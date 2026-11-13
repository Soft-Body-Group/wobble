//! Read thumbnails as little CRT TVs.
//!
//! A Read in TV mode (`ENode::tv`, toggled with shift+H — Blender's
//! "toggle node preview") trades its triangle for a small television whose
//! screen shows the Read's frame at the playhead — the picture untouched
//! (no scanlines, no vignette): the TV is the icon, not a filter. Zoom the
//! graph in and the screen grows: past [`BIG_AT`] pixels it asks for a
//! high-resolution tier,
//! and the pointer moving across it SCRUBS the clip (the browser's
//! hover-scrub, same hands) without touching the comp's playhead.
//!
//! Pictures come off the browser's thumbnail decode (`browser::spawn_thumb`
//! — same decode, same tone curve, no second copy), on rayon, into two
//! small LRU tiers. Nothing here is on the render path: a TV that has not
//! got its picture yet holds the last one it showed, and one that has never
//! had a picture shows static. Following the playhead is throttled to
//! [`FOLLOW_HZ`] per TV, so a playing comp does not queue a CPU decode per
//! Read per frame.
//!
//! The thumbnail is the browser's approximate display (an ACES-ish fit +
//! sRGB on the decoded values, no input transform) — a picture of what is
//! in the file, not a render. The viewer is the arbiter of colour.

use crate::browser::FrameKey;
use egui::{Color32, Pos2, Rect, Stroke, Vec2};
use engine::decode::ExrDecoder;
use engine::player::Sequence;
use std::collections::{HashMap, VecDeque};
use std::sync::{mpsc, Arc};

/// Cabinet half-extents in graph units at node size 1 (a node's circle has
/// radius 24, so a TV is a little wider than two of them).
pub const TV_HALF: Vec2 = Vec2::new(40.0, 30.0);

/// Screen width, in screen pixels, past which a TV asks for the big tier.
pub const BIG_AT: f32 = 200.0;
/// Below this screen width the pointer does not scrub: a TV that small is
/// a glance, and a stray pass of the cursor must not flicker it.
pub const SCRUB_AT: f32 = 90.0;
const SMALL_W: usize = 256;
const SMALL_CAP: usize = 192;
const BIG_W: usize = 1280;
const BIG_CAP: usize = 16;
/// Concurrent decodes per tier: the TVs share the machine with playback.
const WORKERS: usize = 2;
/// Playhead-driven picture changes per TV per second, at most.
const FOLLOW_HZ: f64 = 8.0;

/// What the app tells the editor about one TV-mode Read, each frame.
#[derive(Clone)]
pub struct TvFeed {
    pub seq: Arc<Sequence>,
    /// The Read's frame index at the playhead.
    pub now: usize,
    /// The file's own number for index 0 (1001-numbered plates read as
    /// 1001..), so the scrub readout names the frame the way the files do.
    pub first: Option<i64>,
}

impl TvFeed {
    pub fn len(&self) -> usize {
        self.seq.frames.len().max(1)
    }

    /// The scrub readout for index `i`: the file's frame number when the
    /// sequence has one, else a 1-based index.
    pub fn label(&self, i: usize) -> String {
        match self.first {
            Some(f) => format!("{}", f + i as i64),
            None => format!("{} / {}", i + 1, self.len()),
        }
    }

    /// Frame `i`'s key: a still sequence is a path per frame, a movie is
    /// one path at an index (offset for a trimmed movie) — the browser's
    /// addressing, so the decode path is shared.
    pub fn key(&self, i: usize) -> FrameKey {
        let i = i.min(self.len() - 1);
        if self.seq.movie {
            (self.seq.frames[0].clone(), self.seq.offset + i as u64)
        } else {
            (self.seq.frames[i].clone(), 0)
        }
    }
}

/// Which frame the pointer asks for, `x` across a screen spanning
/// `left..right`, for a clip of `len` frames. The ends are generous: the
/// first and last frames own a half-cell each, so both are reachable
/// without landing exactly on the bezel.
pub fn scrub_index(x: f32, left: f32, right: f32, len: usize) -> usize {
    if len <= 1 || right <= left {
        return 0;
    }
    let f = ((x - left) / (right - left)).clamp(0.0, 1.0);
    ((f * (len - 1) as f32).round() as usize).min(len - 1)
}

/// The screen inside a cabinet: left of the knob panel, under the top rail.
pub fn screen_of(cab: Rect) -> Rect {
    let (w, h) = (cab.width(), cab.height());
    Rect::from_min_max(
        cab.min + Vec2::new(0.07 * w, 0.11 * h),
        Pos2::new(cab.max.x - 0.25 * w, cab.max.y - 0.13 * h),
    )
}

/// One resolution tier: textures by key, LRU-evicted, decoded on rayon.
struct Tier {
    tex: HashMap<FrameKey, egui::TextureHandle>,
    order: VecDeque<FrameKey>,
    inflight: Vec<FrameKey>,
    wanted: Vec<FrameKey>,
    tx: mpsc::Sender<(FrameKey, egui::ColorImage)>,
    rx: mpsc::Receiver<(FrameKey, egui::ColorImage)>,
    max_w: usize,
    cap: usize,
}

impl Tier {
    fn new(max_w: usize, cap: usize) -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            tex: HashMap::new(),
            order: VecDeque::new(),
            inflight: Vec::new(),
            wanted: Vec::new(),
            tx,
            rx,
            max_w,
            cap,
        }
    }

    fn peek(&self, k: &FrameKey) -> Option<egui::TextureHandle> {
        self.tex.get(k).cloned()
    }

    fn want(&mut self, k: &FrameKey) {
        if !self.tex.contains_key(k) && !self.wanted.contains(k) {
            self.wanted.push(k.clone());
        }
    }

    /// Land finished decodes; start the newest wants. Returns whether work
    /// is still in flight (the caller keeps repainting until it lands).
    fn pump(&mut self, ctx: &egui::Context, dec: &Arc<dyn ExrDecoder>) -> bool {
        while let Ok((key, img)) = self.rx.try_recv() {
            self.inflight.retain(|k| *k != key);
            let t = ctx.load_texture("tv", img, egui::TextureOptions::LINEAR);
            if self.tex.insert(key.clone(), t).is_none() {
                self.order.push_back(key);
            }
            while self.order.len() > self.cap {
                if let Some(old) = self.order.pop_front() {
                    self.tex.remove(&old);
                }
            }
        }
        // Newest first: while scrubbing, the frame under the pointer NOW
        // matters and the one from three frames ago does not.
        let mut wanted = std::mem::take(&mut self.wanted);
        wanted.reverse();
        for key in wanted {
            if self.inflight.len() >= WORKERS {
                break;
            }
            if self.tex.contains_key(&key) || self.inflight.contains(&key) {
                continue;
            }
            self.inflight.push(key.clone());
            crate::browser::spawn_thumb(dec.clone(), key, self.max_w, self.tx.clone());
        }
        !self.inflight.is_empty()
    }
}

/// The TVs' picture cache: two tiers, plus the picture each TV last showed
/// (held while the next decodes, so a TV never blanks mid-scrub).
pub struct TvCache {
    small: Tier,
    big: Tier,
    shown: HashMap<u64, egui::TextureHandle>,
    /// (when, which frame) each TV last took from the playhead.
    followed: HashMap<u64, (f64, usize)>,
}

impl Default for TvCache {
    fn default() -> Self {
        Self {
            small: Tier::new(SMALL_W, SMALL_CAP),
            big: Tier::new(BIG_W, BIG_CAP),
            shown: HashMap::new(),
            followed: HashMap::new(),
        }
    }
}

/// What one TV shows this frame.
pub struct Picture {
    pub tex: Option<egui::TextureHandle>,
    /// The frame index actually being asked for (the scrub or the
    /// throttled playhead).
    pub idx: usize,
    /// `tex` IS that frame (not a held stand-in).
    pub exact: bool,
    pub big: bool,
}

impl TvCache {
    pub fn pump(&mut self, ctx: &egui::Context, dec: &Arc<dyn ExrDecoder>) -> bool {
        let a = self.small.pump(ctx, dec);
        let b = self.big.pump(ctx, dec);
        a || b
    }

    /// Forget TVs that are gone (their held picture and throttle state).
    pub fn retain(&mut self, alive: impl Fn(u64) -> bool) {
        self.shown.retain(|id, _| alive(*id));
        self.followed.retain(|id, _| alive(*id));
    }

    /// The picture for TV `id`: frame `scrub` if the pointer is scrubbing,
    /// else the playhead frame (throttled), at the tier `screen_w` asks for.
    pub fn picture(
        &mut self,
        id: u64,
        feed: &TvFeed,
        scrub: Option<usize>,
        screen_w: f32,
        time: f64,
    ) -> Picture {
        let idx = match scrub {
            Some(i) => i,
            None => {
                let want = feed.now.min(feed.len() - 1);
                match self.followed.get(&id).copied() {
                    Some((t, i)) if i != want && time - t < 1.0 / FOLLOW_HZ => i,
                    Some((_, i)) if i == want => want,
                    _ => {
                        self.followed.insert(id, (time, want));
                        want
                    }
                }
            }
        };
        let key = feed.key(idx);
        let big = screen_w >= BIG_AT;
        let exact = if big {
            match self.big.peek(&key) {
                Some(t) => Some(t),
                None => {
                    self.big.want(&key);
                    // The small tier's copy of the same frame is still the
                    // RIGHT picture, just softer — better than a held one.
                    self.small.peek(&key)
                }
            }
        } else {
            match self.small.peek(&key) {
                Some(t) => Some(t),
                None => {
                    self.small.want(&key);
                    None
                }
            }
        };
        if let Some(t) = &exact {
            self.shown.insert(id, t.clone());
        }
        Picture {
            exact: exact.is_some(),
            tex: exact.or_else(|| self.shown.get(&id).cloned()),
            idx,
            big,
        }
    }
}

/// Draw one TV into cabinet `cab` (screen coords). `base` is the Read's
/// type colour, `stroke` its selection-aware outline. `scrub` is the
/// overlay text while the pointer is scrubbing (drawn with a playhead at
/// `scrub_x`). `seed` + `time` animate the static when there is no picture.
#[allow(clippy::too_many_arguments)]
pub fn draw(
    painter: &egui::Painter,
    cab: Rect,
    base: Color32,
    stroke: Stroke,
    pic: Option<&egui::TextureHandle>,
    scrub: Option<(f32, String)>,
    seed: u64,
    time: f32,
) {
    let (w, h) = (cab.width(), cab.height());
    if w < 3.0 {
        return;
    }
    // GRAPHIC, like every other node glyph: flat type colour, bold ink,
    // nothing modelled. One ink weight for every mark keeps it an icon.
    let wt = (h * 0.075).clamp(1.0, 9.0);
    // The SILHOUETTE is one colour, the cabinet's — antennas and feet
    // included (ink out on the graph's near-black would vanish); ink is
    // only for marks ON the body. Antennas first, so the cabinet covers
    // their root: a bold V with round ends.
    let root = Pos2::new(cab.center().x, cab.min.y + wt);
    for tip in [root + Vec2::new(-w * 0.24, -h * 0.40), root + Vec2::new(w * 0.24, -h * 0.40)] {
        painter.line_segment([root, tip], Stroke::new(wt, base));
        painter.circle_filled(tip, wt * 0.5, base);
    }
    for fx in [0.16f32, 0.84] {
        painter.rect_filled(
            Rect::from_center_size(
                Pos2::new(cab.min.x + fx * w, cab.max.y + h * 0.03),
                Vec2::new(w * 0.13, h * 0.11),
            ),
            wt * 0.3,
            base,
        );
    }
    body(painter, cab, screen_of(cab), h * 0.2, base, stroke, pic, scrub, seed, time, 4.0, true);
}

/// Boring mode's glass: the whole card, inset by its outline.
pub fn plain_screen(cab: Rect) -> Rect {
    cab.shrink((cab.height() * 0.06).max(1.0))
}

/// Boring mode's Load: a flat 4:3 card in the node's colour with its frame
/// fitted on it — no cabinet, antennas, knobs or snow.
pub fn plain(
    painter: &egui::Painter,
    cab: Rect,
    base: Color32,
    stroke: Stroke,
    pic: Option<&egui::TextureHandle>,
    scrub: Option<(f32, String)>,
) {
    if cab.width() < 3.0 {
        return;
    }
    let rad = cab.height() * 0.15;
    painter.rect(cab, rad, base, stroke, egui::StrokeKind::Middle);
    let screen = plain_screen(cab);
    if let Some(t) = pic {
        let [tw, th] = t.size();
        let k = (screen.width() / tw.max(1) as f32).min(screen.height() / th.max(1) as f32);
        let img = Rect::from_center_size(screen.center(), Vec2::new(tw as f32 * k, th as f32 * k));
        painter.image(t.id(), img, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
    }
    if let Some((x, _)) = scrub {
        let x = x.clamp(screen.min.x, screen.max.x);
        painter.line_segment(
            [Pos2::new(x, screen.min.y), Pos2::new(x, screen.max.y)],
            Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 190)),
        );
    }
}

/// The cabinet and everything on it — no antennas, no feet (the app icon
/// is this alone). `screen` is the glass (`screen_of` for the graph's
/// cabinet), `round` the cabinet's corner radius, `cell` the
/// static's cell size, `caption` whether a big enough empty screen says
/// "no signal".
#[allow(clippy::too_many_arguments)]
pub fn body(
    painter: &egui::Painter,
    cab: Rect,
    screen: Rect,
    round: f32,
    base: Color32,
    stroke: Stroke,
    pic: Option<&egui::TextureHandle>,
    scrub: Option<(f32, String)>,
    seed: u64,
    time: f32,
    cell: f32,
    caption: bool,
) {
    let h = cab.height();
    let ink = crate::theme::node_ink();
    let wt = (h * 0.075).clamp(1.0, 9.0);
    // The cabinet: the Read's own colour, full strength, like any node.
    painter.rect(cab, round, base, stroke, egui::StrokeKind::Outside);

    let (sw, sh) = (screen.width(), screen.height());
    let rad = sh * 0.22;
    // The bezel is a bold ink ring, CONCENTRIC with the glass (radius +
    // margin), and the margin is at least (√2 − 1)·radius, so the glass's
    // square corners — where the fillets below paint — sit inside it.
    let e = (rad * 0.45).max(wt).max(0.8);
    painter.rect_filled(screen.expand(e), rad + e, ink);
    painter.rect_filled(screen, rad, Color32::from_gray(0x04));

    // The picture, fitted (letterboxed) inside the glass.
    match pic {
        Some(t) => {
            let [tw, th] = t.size();
            let k = (sw / tw.max(1) as f32).min(sh / th.max(1) as f32);
            let img = Rect::from_center_size(screen.center(), Vec2::new(tw as f32 * k, th as f32 * k));
            painter.image(
                t.id(),
                img,
                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );
        }
        None => snow(&painter.with_clip_rect(screen), screen, seed, time, cell),
    }

    // Round the glass: fill the four corners outside the rounded rect.
    fillets(painter, screen, rad, ink);

    // Two knobs: solid ink dots, centred in the panel right of the glass.
    let px = (screen.max.x + e + cab.max.x) * 0.5;
    let kr = ((cab.max.x - screen.max.x - e) * 0.28).max(0.8);
    for ky in [0.36f32, 0.64] {
        painter.circle_filled(Pos2::new(px, cab.min.y + ky * h), kr, ink);
    }

    if caption && pic.is_none() && sh > 60.0 {
        painter.text(
            screen.center(),
            egui::Align2::CENTER_CENTER,
            "no signal",
            egui::FontId::monospace((sh * 0.09).clamp(9.0, 16.0)),
            Color32::from_white_alpha(170),
        );
    }
    if let Some((x, label)) = scrub {
        let x = x.clamp(screen.min.x, screen.max.x);
        painter.line_segment(
            [Pos2::new(x, screen.min.y), Pos2::new(x, screen.max.y)],
            Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 190)),
        );
        let at = Pos2::new(screen.min.x + 6.0, screen.max.y - 5.0);
        let font = egui::FontId::monospace(11.0);
        painter.text(at + Vec2::splat(1.0), egui::Align2::LEFT_BOTTOM, &label, font.clone(), Color32::BLACK);
        painter.text(at, egui::Align2::LEFT_BOTTOM, &label, font, Color32::WHITE);
    }
}

/// Static: a grid of grey cells that reshuffles at 24 Hz. Cheap (one mesh),
/// and it says "tuned to nothing yet" better than a black rectangle does.
fn snow(painter: &egui::Painter, r: Rect, seed: u64, time: f32, cell: f32) {
    let cols = ((r.width() / cell) as usize).clamp(6, 48);
    let rows = ((r.height() / cell) as usize).clamp(4, 36);
    let tick = (time * 24.0) as u64;
    let (cw, ch) = (r.width() / cols as f32, r.height() / rows as f32);
    let mut mesh = egui::epaint::Mesh::default();
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ tick.wrapping_mul(0xD1B5_4A32_D192_ED03);
    for j in 0..rows {
        for i in 0..cols {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let g = (s >> 56) as u8 / 2 + 40;
            let c = Rect::from_min_size(
                Pos2::new(r.min.x + i as f32 * cw, r.min.y + j as f32 * ch),
                Vec2::new(cw + 0.5, ch + 0.5),
            );
            mesh.add_colored_rect(c, Color32::from_gray(g));
        }
    }
    painter.add(egui::Shape::mesh(mesh));
}

/// Fill the four corners of `r` that lie OUTSIDE a rounded rect of radius
/// `rad`, in `color` — how the glass gets its CRT curve over a square image.
fn fillets(painter: &egui::Painter, r: Rect, rad: f32, color: Color32) {
    let rad = rad.min(r.width() * 0.5).min(r.height() * 0.5);
    if rad < 0.8 {
        return;
    }
    const N: usize = 8;
    let mut mesh = egui::epaint::Mesh::default();
    // (corner, arc centre, start angle)
    let corners = [
        (r.left_top(), r.left_top() + Vec2::new(rad, rad), std::f32::consts::PI),
        (r.right_top(), r.right_top() + Vec2::new(-rad, rad), -std::f32::consts::FRAC_PI_2),
        (r.right_bottom(), r.right_bottom() + Vec2::new(-rad, -rad), 0.0),
        (r.left_bottom(), r.left_bottom() + Vec2::new(rad, -rad), std::f32::consts::FRAC_PI_2),
    ];
    for (corner, c, a0) in corners {
        let i0 = mesh.vertices.len() as u32;
        mesh.colored_vertex(corner, color);
        for k in 0..=N {
            let a = a0 + std::f32::consts::FRAC_PI_2 * k as f32 / N as f32;
            mesh.colored_vertex(c + Vec2::new(a.cos(), a.sin()) * rad, color);
        }
        for k in 0..N as u32 {
            mesh.add_triangle(i0, i0 + 1 + k, i0 + 2 + k);
        }
    }
    painter.add(egui::Shape::mesh(mesh));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn seq(n: usize, movie: bool, offset: u64) -> Arc<Sequence> {
        let frames = (0..n)
            .map(|i| {
                if movie {
                    PathBuf::from("clip.mov")
                } else {
                    PathBuf::from(format!("plate.{:04}.exr", 1001 + i))
                }
            })
            .collect();
        Arc::new(Sequence {
            frames,
            info: engine::decode::ImageInfo { width: 64, height: 36 },
            proxy: 1,
            movie,
            offset,
            pack: Default::default(),
        })
    }

    #[test]
    fn a_scrub_spans_the_whole_clip_and_clamps_at_the_bezel() {
        assert_eq!(scrub_index(100.0, 100.0, 200.0, 11), 0);
        assert_eq!(scrub_index(200.0, 100.0, 200.0, 11), 10);
        assert_eq!(scrub_index(150.0, 100.0, 200.0, 11), 5);
        assert_eq!(scrub_index(-50.0, 100.0, 200.0, 11), 0, "off the left holds the first frame");
        assert_eq!(scrub_index(999.0, 100.0, 200.0, 11), 10, "off the right holds the last");
        assert_eq!(scrub_index(150.0, 100.0, 200.0, 1), 0, "a still has one frame");
    }

    #[test]
    fn keys_address_sequences_by_path_and_movies_by_index() {
        let s = TvFeed { seq: seq(5, false, 0), now: 0, first: None };
        assert_eq!(s.key(2), (PathBuf::from("plate.1003.exr"), 0));
        assert_eq!(s.key(99), (PathBuf::from("plate.1005.exr"), 0), "clamped to the clip");
        let m = TvFeed { seq: seq(5, true, 40), now: 0, first: None };
        assert_eq!(m.key(2), (PathBuf::from("clip.mov"), 42), "a trimmed movie keeps its offset");
    }

    #[test]
    fn the_screen_sits_inside_the_cabinet_left_of_the_knobs() {
        let cab = Rect::from_center_size(Pos2::ZERO, TV_HALF * 2.0);
        let s = screen_of(cab);
        assert!(cab.contains_rect(s));
        assert!(s.max.x < cab.max.x - 0.2 * cab.width(), "room for the knob panel");
        assert!(s.width() > s.height(), "a landscape screen");
    }

    /// The throttle: a playing playhead moves the picture at most FOLLOW_HZ
    /// times a second per TV, and a scrub is never throttled.
    #[test]
    fn following_the_playhead_is_throttled_but_scrubbing_is_not() {
        let mut c = TvCache::default();
        let mut f = TvFeed { seq: seq(100, false, 0), now: 10, first: Some(1001) };
        assert_eq!(c.picture(1, &f, None, 60.0, 0.0).idx, 10);
        f.now = 11;
        assert_eq!(c.picture(1, &f, None, 60.0, 0.01).idx, 10, "too soon: hold");
        assert_eq!(c.picture(1, &f, None, 60.0, 0.2).idx, 11, "after the interval it follows");
        assert_eq!(c.picture(1, &f, Some(77), 60.0, 0.21).idx, 77, "a scrub is immediate");
        let p = c.picture(1, &f, Some(77), 300.0, 0.22);
        assert!(p.big && !p.exact, "a big screen asks the big tier; nothing decoded yet");
    }

    /// `cargo test -p app tv_graph_render -- --ignored` writes
    /// `target/tv-graph.png`: the REAL editor drawing a comp whose Reads
    /// are TVs, fed the repo's own plates (`data/2k_zip_bg`, `…_fg`) —
    /// once at a working zoom and once zoomed in on one TV (big tier).
    /// Decodes are awaited up front, so the shot shows pictures, not snow.
    #[test]
    #[ignore]
    fn tv_graph_render() {
        use crate::editor::{EditorGraph, NodeEditor};
        use engine::graph::NodeKind;
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let dec: Arc<dyn ExrDecoder> = Arc::new(engine::decode::AnyDecoder::new(false));
        let load = |dir: &str| -> Option<Arc<Sequence>> {
            let mut frames: Vec<PathBuf> = std::fs::read_dir(root.join(dir))
                .ok()?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "exr"))
                .collect();
            frames.sort();
            let info = dec.read_header(frames.first()?).ok()?;
            Some(Arc::new(Sequence { frames, info, proxy: 1, movie: false, offset: 0, pack: Default::default() }))
        };
        let (Some(bg), Some(fg)) = (load("data/2k_zip_bg"), load("data/2k_zip_fg")) else {
            println!("SKIP: no test plates (generate data/ first)");
            return;
        };
        let mut g = EditorGraph::default();
        let ra = g.add(NodeKind::read(0, ""), Pos2::new(-150.0, -150.0));
        let rb = g.add(NodeKind::read(1, ""), Pos2::new(150.0, -150.0));
        let gr = g.add(NodeKind::Grade(Default::default()), Pos2::new(-150.0, 0.0));
        let mg = g.add(NodeKind::blend(), Pos2::new(0.0, 130.0));
        g.connect(gr, 0, ra);
        g.connect(mg, 0, gr);
        g.connect(mg, 1, rb);
        for id in [ra, rb] {
            g.node_mut(id).unwrap().tv = true;
        }
        let feeds = [(ra, bg.clone(), 5usize), (rb, fg.clone(), 12usize)];
        for (name, zoom, pan) in [("tv-graph", 1.3f32, Vec2::ZERO), ("tv-zoom", 4.6, Vec2::new(150.0, 150.0) * 4.6)] {
            let mut ed = NodeEditor::default();
            ed.zoom = zoom;
            ed.pan = pan;
            ed.tv_decoder = Some(dec.clone());
            ed.tv_feeds = feeds
                .iter()
                .map(|(id, s, now)| (*id, TvFeed { seq: s.clone(), now: *now, first: Some(1) }))
                .collect();
            let (mut sel, mut act, mut pin, mut slot, mut rects) =
                (Vec::new(), Vec::new(), Vec::new(), 2usize, Vec::new());
            let mut waited = false;
            crate::uishot::render(name, 900.0, 560.0, 2.0, 4, |ui, _| {
                // First frame: ask, then block until the pictures land (a
                // test may wait; the app never does).
                if !waited {
                    waited = true;
                    ed.show(ui, &mut g, &mut sel, &mut act, &mut pin, &mut slot, &mut rects);
                    for _ in 0..400 {
                        std::thread::sleep(web_time::Duration::from_millis(15));
                        let busy = ed.test_tv_pump(ui.ctx());
                        let done = [ra, rb].iter().all(|id| {
                            ed.test_tv(*id).is_none_or(|(_, _, exact, _)| exact)
                        });
                        if !busy && done {
                            break;
                        }
                        ed.show(ui, &mut g, &mut sel, &mut act, &mut pin, &mut slot, &mut rects);
                    }
                }
                ed.show(ui, &mut g, &mut sel, &mut act, &mut pin, &mut slot, &mut rects);
            });
        }
    }

    /// `cargo test -p app tv_render -- --ignored` writes `target/tv.png`:
    /// TVs at three zooms, with colour bars as the picture and one with no
    /// signal.
    #[test]
    #[ignore]
    fn tv_render() {
        let out = crate::uishot::render("tv", 900.0, 420.0, 2.0, 3, |ui, _| {
            let area = ui.max_rect();
            ui.painter().rect_filled(area, 0.0, crate::theme::graph_bg());
            let (w, h) = (64usize, 36usize);
            let mut px = vec![0u8; w * h * 4];
            let bars = [
                [192, 192, 192], [192, 192, 0], [0, 192, 192], [0, 192, 0],
                [192, 0, 192], [192, 0, 0], [0, 0, 192],
            ];
            for y in 0..h {
                for x in 0..w {
                    let b = bars[x * bars.len() / w];
                    let d = (y * w + x) * 4;
                    px[d..d + 4].copy_from_slice(&[b[0], b[1], b[2], 255]);
                }
            }
            let img = egui::ColorImage::from_rgba_unmultiplied([w, h], &px);
            let tex = ui.ctx().load_texture("bars", img, egui::TextureOptions::LINEAR);
            let base = Color32::from_rgb(0xa8, 0x2e, 0x7e);
            let st = Stroke::new(1.0, Color32::from_gray(0x0c));
            let p = ui.painter();
            let mut x = area.min.x + 30.0;
            for z in [0.8f32, 1.6, 4.0] {
                let half = TV_HALF * z;
                let cab = Rect::from_center_size(Pos2::new(x + half.x, area.center().y), half * 2.0);
                let s = screen_of(cab);
                let scrub = (z > 3.0).then(|| (s.center().x, "f 12 / 48".to_string()));
                draw(p, cab, base, st, Some(&tex), scrub, 7, 0.0);
                x += half.x * 2.0 + 40.0;
            }
            let cab = Rect::from_center_size(Pos2::new(x + 60.0, area.center().y), TV_HALF * 3.0);
            draw(p, cab, base, st, None, None, 7, 0.3);
        });
        if out.is_none() {
            println!("SKIP: no adapter");
        }
    }
}
