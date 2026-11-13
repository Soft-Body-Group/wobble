//! The Layer node's two instruments.
//!
//! **The mini timeline** (`timeline`) is the panel: one bar per layer, drawn
//! against the comp's own frame range. Drag a bar along to slide the layer in
//! time, drag its ends to trim it, click it to select it. Bars only move
//! left and right (by design): a vertical restack fired on every slightly
//! diagonal slide. It is deliberately small — an After Effects timeline with the
//! keyframe machinery left out, because this app already has a curve editor
//! and a timeline of its own.
//!
//! **The transform gizmo** (`transform_gizmo`) is the viewer manipulator: a
//! pivot to translate by, a ring to rotate by, eight border boxes and an orbit
//! cluster to scale by. The Transform node draws it. (Layers once carried a
//! transform and drew one each; they are placed by upstream Transforms now.)
//!
//! Nothing in here knows about the app struct: the gizmo takes the two
//! coordinate closures it needs and the timeline takes a slice of layers, so
//! both are usable by anything that can draw an egui frame.

use crate::ink::InkShapes;
use crate::editor::EditorGraph;
use crate::theme;
use engine::graph::{BlendOp, LayerItem, NodeKind, TransformParams};

/// Which layer of which stack is selected. Selection is per NODE so that
/// clicking between two Layer panels doesn't lose your place, and the viewer
/// can draw the right manipulator for each stack it is showing.
pub type LayerSel = std::collections::HashMap<u64, usize>;

/// What one frame of the mini timeline did, applied by the caller once the
/// node borrow ends (a reorder has to move edges and animation as well as
/// layers, and none of that is reachable from inside a `&mut LayerParams`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TimelineAction {
    /// Selection moved to this layer.
    Select(usize),
    /// Scrub: the playhead should go to this comp frame.
    Seek(u64),
}

/// A layer's SOURCE frames, inclusive, in its input's own clock: a plate
/// placed at 1018 reports `1018..=1080`, because that is what the layer's
/// time map (`source = (comp − start) × speed`) asks the input for.
/// `play` is what the input plays (its range: the bar untrimmed), `clip`
/// the whole clip behind it (how far an end can be dragged out). A bare
/// length is footage that starts at 0 (`Src::from(24)` is `0..=23`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Src {
    pub play: (i64, i64),
    pub clip: (i64, i64),
}

impl From<i64> for Src {
    fn from(len: i64) -> Self {
        Src::from((0, len.max(1) - 1))
    }
}

impl From<(i64, i64)> for Src {
    fn from((a, b): (i64, i64)) -> Self {
        let r = (a.min(b), a.max(b));
        Src { play: r, clip: r }
    }
}

impl Src {
    /// `play` inside a longer `clip` (the clip always covers the range).
    pub fn new(play: (i64, i64), clip: (i64, i64)) -> Self {
        let play = (play.0.min(play.1), play.0.max(play.1));
        Src { play, clip: (clip.0.min(clip.1).min(play.0), clip.0.max(clip.1).max(play.1)) }
    }
}

/// Where source frames `r` sit in comp frames at this layer's start and
/// rate: a plate at 1018 with start 0 sits at 1018, not at 0.
fn place(l: &LayerItem, r: (i64, i64)) -> (i64, i64) {
    let rate = l.rate() as f64;
    let lead = (r.0 as f64 / rate).min(r.1 as f64 / rate).round() as i64;
    let a = l.start + lead;
    (a, a + l.stretched((r.1 - r.0 + 1).max(1)) - 1)
}

/// Where the layer's whole CLIP sits in comp frames — the ghost drawn
/// behind the bar, and the limit a trim stops at.
fn footage(l: &LayerItem, s: impl Into<Src>) -> (i64, i64) {
    place(l, s.into().clip)
}

/// Geometry of one bar, in comp frames (inclusive). Untrimmed it is what
/// the input plays, stretched by the layer's rate (which is why pulling an
/// end longer is a retime and not just a longer rectangle); a trim wins.
fn extent(l: &LayerItem, s: impl Into<Src>) -> (i64, i64) {
    match l.trim {
        Some((a, b)) => (a.min(b), a.max(b)),
        None => place(l, s.into().play),
    }
}

/// The source frames comp frames `a..=b` of layer `l` show, rounded out.
pub fn source_span(l: &LayerItem, (a, b): (i64, i64)) -> (i64, i64) {
    let r = l.rate() as f64;
    let (x, y) = ((a - l.start) as f64 * r, (b - l.start) as f64 * r);
    (x.min(y).floor() as i64, x.max(y).ceil() as i64)
}

/// What a press on a bar does, After Effects' way round: an END trims
/// (in/out move, the footage does not), alt on an end STRETCHES (the rate
/// changes, the other end holds), the body slides, shift on the body SLIPS
/// (the footage moves under a window that stays).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BarEdit {
    Trim(usize),
    Stretch(usize),
    Slide,
    Slip,
}

/// `l` as it was at the press, edited by `d` comp frames. Absolute from
/// the press (never summed per frame), so a slow drag still moves and a
/// drag back lands exactly where it began. `len` is the source length the
/// bar is drawn from; `bounded` is false for anything endless (a
/// generator), which a trim never stops.
pub fn edit_bar(l: &LayerItem, s: impl Into<Src>, bounded: bool, edit: BarEdit, d: i64) -> LayerItem {
    let s = s.into();
    let mut o = l.clone();
    match edit {
        BarEdit::Slide => {
            o.start += d;
            if let Some((a, b)) = l.trim {
                o.trim = Some((a + d, b + d));
            }
        }
        BarEdit::Slip => {
            // Only a trimmed layer has a window to slip under; untrimmed,
            // the bar IS the footage and a slip is a slide.
            o.start += d;
        }
        BarEdit::Trim(k) => {
            let (a, b) = extent(l, s);
            // A trim never reaches past the footage (AE's rule): there is
            // nothing out there but a held frame. One that already did
            // keeps what it had.
            let (fa, fb) = if bounded {
                footage(l, s)
            } else {
                (i64::MIN / 4, i64::MAX / 4)
            };
            let (lo, hi) = (fa.min(a), fb.max(b));
            let t = if k == 0 {
                ((a + d).clamp(lo, b), b)
            } else {
                (a, (b + d).clamp(a, hi))
            };
            o.trim = Some(t);
        }
        BarEdit::Stretch(k) => {
            let n = (s.play.1 - s.play.0 + 1).max(1);
            let (a, b) = extent(l, s);
            let w = b - a + 1;
            let w2 = if k == 0 { w - d } else { w + d }.max(1);
            if w2 == w {
                return o;
            }
            // The OTHER end holds: dragging the out point keeps the in
            // point, dragging the in point keeps the out point.
            let anchor = if k == 0 { b + 1 } else { a } as f64;
            let f = w2 as f64 / w as f64;
            let sign = if l.speed < 0.0 { -1.0 } else { 1.0 };
            match l.trim {
                None => {
                    // Untrimmed the bar is the footage: fit it EXACTLY.
                    o.speed = sign * (n as f32 / w2 as f32);
                    let lo = if k == 0 { b + 1 - w2 } else { a };
                    o.start += lo - place(&o, s.play).0;
                }
                Some(_) => {
                    o.speed = (l.speed as f64 / f) as f32;
                    o.start = (anchor + (l.start as f64 - anchor) * f).round() as i64;
                    o.trim = Some(if k == 0 { (b + 1 - w2, b) } else { (a, a + w2 - 1) });
                }
            }
        }
    }
    o
}

/// Stretch about the IN point by a factor (AE's Stretch % field: 200% is
/// twice as long, half as fast). The in point and what plays there hold.
pub fn stretch_to(l: &LayerItem, s: impl Into<Src>, speed: f32) -> LayerItem {
    let (a, _) = extent(l, s);
    let mut o = l.clone();
    if speed.abs() < 1e-3 || l.speed == speed {
        return o;
    }
    let f = (l.rate() / speed) as f64; // how much longer the layer gets
    o.speed = speed;
    o.start = (a as f64 + (l.start - a) as f64 * f).round() as i64;
    if let Some((ta, tb)) = l.trim.map(|(x, y)| (x.min(y), x.max(y))) {
        let w = ((tb - ta + 1) as f64 * f.abs()).round().max(1.0) as i64;
        o.trim = Some((ta, ta + w - 1));
    }
    o
}

/// Layer `i`'s source frames: the input's own range, or for anything
/// endless (a generator) the comp's, which is what it fills.
fn src_of(lens: &[Option<Src>], i: usize, comp_span: (u64, u64)) -> Src {
    lens.get(i)
        .copied()
        .flatten()
        .unwrap_or_else(|| (comp_span.0 as i64, comp_span.1 as i64).into())
}

/// The mini timeline. `lens[i]` is layer i's source frame range in its
/// input's clock when the app knows it (a Read's first..last), `None` for
/// anything endless.
///
/// Returns what the caller must apply afterwards, whether any layer's own
/// values changed (start/trim edits, undoable like any other parameter
/// edit), and where each bar landed on screen — which is how the self-test
/// aims at one, and how anything else could point at a layer.
#[allow(clippy::too_many_arguments)]
pub fn timeline(
    ui: &mut egui::Ui,
    layers: &mut [LayerItem],
    names: &[String],
    lens: &[Option<Src>],
    sel: usize,
    comp_span: (u64, u64),
    playhead: u64,
    row_h: f32,
    ranges: crate::minibar::Ranges,
) -> (Option<TimelineAction>, bool, Vec<egui::Rect>) {
    let mut action = None;
    let mut changed = false;
    let mut bars = Vec::new();
    let n = layers.len();
    if n == 0 {
        ui.weak("no layers — add one below");
        return (None, false, bars);
    }
    // The comp occupies `c0..c1` COMP FRAMES (a shot can start at 1001).
    let (c0, c1) = (comp_span.0 as i64, comp_span.1 as i64 + 1);
    // "node range" on the pie: the comp and every layer, however far one
    // has been slid or stretched — an end you cannot reach is an end you
    // cannot drag back.
    let node_range = {
        let (mut lo, mut hi) = (c0, c1 - 1);
        for (i, l) in layers.iter().enumerate() {
            let (a, b) = extent(l, src_of(lens, i, comp_span));
            lo = lo.min(a);
            hi = hi.max(b);
        }
        (lo, hi)
    };

    // The head holds the layer number and its M(ute) / S(olo) switches.
    let head_w = 62.0_f32.min(ui.available_width() * 0.25);
    let full = ui.available_width();
    let track_w = (full - head_w).max(40.0);
    let height = row_h * (n as f32 + 1.0);
    let (rect, _) =
        ui.allocate_exact_size(egui::vec2(full, height), egui::Sense::hover());
    let track = egui::Rect::from_min_max(
        egui::pos2(rect.min.x + head_w, rect.min.y + row_h),
        egui::pos2(rect.min.x + head_w + track_w, rect.max.y),
    );
    // The view: the main time bar's hands (`minibar`), and it never
    // re-fits itself — a drag that let go used to re-frame the bar under
    // the hand. An alt press on a trim handle is the trim's, not a pan.
    let nav_area = egui::Rect::from_min_max(egui::pos2(track.min.x, rect.min.y), track.max);
    let handles_id = ui.id().with("lay-handles");
    // Read last frame's handles BEFORE the input lock (egui's lock is not
    // re-entrant).
    let handles: Vec<egui::Rect> = ui.data(|d| d.get_temp(handles_id)).unwrap_or_default();
    let handle_press = ui.input(|i| {
        i.modifiers.alt
            && i.pointer.primary_pressed()
            && i.pointer.press_origin().is_some_and(|p| handles.iter().any(|h| h.contains(p)))
    });
    let mv = crate::minibar::view(
        ui,
        ui.id().with("lay-view"),
        nav_area,
        &crate::minibar::Ranges { node: node_range, ..ranges },
        false,
        handle_press,
    );
    let nav = mv.nav;
    let (lo, hi) = mv.view;
    let span = (hi - lo + 1.0).max(1.0) as f32;
    let lo = lo as f32;
    let f_to_x = |f: f32| track.min.x + (f - lo) / span * track_w;
    let x_to_f = |x: f32| (x - track.min.x) / track_w * span + lo;
    let px_per_frame = track_w / span;
    let mut handle_rects = Vec::new();

    // Bars, ruler and playhead clip to the track: panned, a bar must not
    // run over the layer numbers.
    let p = ui.painter_at(nav_area);
    let head = ui.painter_at(rect);
    let vis = ui.visuals().clone();
    let dim = vis.weak_text_color();

    // --- ruler ------------------------------------------------------------
    // Ticks at a round interval that keeps them ~60px apart, so the labels
    // stay readable at any panel width or zoom.
    let step = {
        let want = 60.0 / px_per_frame.max(1e-6);
        [1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
            .into_iter()
            .find(|s| *s >= want)
            .unwrap_or(1000.0) as i64
    };
    let ruler = egui::Rect::from_min_max(
        egui::pos2(track.min.x, rect.min.y),
        egui::pos2(track.max.x, rect.min.y + row_h),
    );
    p.rect_filled(ruler, 2.0, vis.extreme_bg_color);
    let mut f = (lo.floor() as i64).div_euclid(step) * step;
    while (f as f64) <= hi {
        let x = f_to_x(f as f32);
        p.line_segment(
            [egui::pos2(x, ruler.max.y - 4.0), egui::pos2(x, ruler.max.y)],
            egui::Stroke::new(1.0, dim),
        );
        p.text(
            egui::pos2(x + 2.0, ruler.min.y + 1.0),
            egui::Align2::LEFT_TOP,
            f.to_string(),
            egui::FontId::monospace(9.0),
            dim,
        );
        f += step;
    }
    // The comp's own range, so "off the end" is visible rather than implied.
    let (cx0, cx1) = (f_to_x(c0 as f32), f_to_x(c1 as f32));
    p.rect_filled(
        egui::Rect::from_min_max(egui::pos2(cx0, track.min.y), egui::pos2(cx1, track.max.y)),
        0.0,
        vis.faint_bg_color,
    );

    // --- rows -------------------------------------------------------------
    for i in 0..n {
        let y = track.min.y + i as f32 * row_h;
        let row = egui::Rect::from_min_max(
            egui::pos2(rect.min.x, y),
            egui::pos2(track.max.x, y + row_h - 2.0),
        );
        let selected = i == sel;
        // Layer number, AE style: 1 is the top of the stack.
        head.text(
            egui::pos2(rect.min.x + 3.0, row.center().y),
            egui::Align2::LEFT_CENTER,
            format!("{}", i + 1),
            egui::FontId::monospace(10.0),
            if selected { vis.strong_text_color() } else { dim },
        );

        // Mute and solo, a mixing desk's switches: a click flips one. The
        // number column is 18px, the switches share what is left.
        let sw = ((head_w - 20.0) / 2.0 - 2.0).clamp(6.0, 16.0);
        for (k, letter) in [(0usize, "M"), (1, "S")] {
            let x0 = rect.min.x + 18.0 + k as f32 * (sw + 2.0);
            let r = egui::Rect::from_min_size(
                egui::pos2(x0, row.min.y + 2.0),
                egui::vec2(sw, (row.height() - 3.0).max(4.0)),
            );
            let resp = ui
                .interact(r, ui.id().with(("lay-sw", i, k)), egui::Sense::click())
                .on_hover_text(if k == 0 {
                    "mute: leave this layer out of the comp"
                } else {
                    "solo: while any layer is soloed, only soloed layers composite"
                });
            let on = if k == 0 { layers[i].mute } else { layers[i].solo };
            if resp.clicked() {
                if k == 0 {
                    layers[i].mute = !on;
                } else {
                    layers[i].solo = !on;
                }
                changed = true;
            }
            let col = if k == 0 { theme::layer_mute() } else { theme::layer_solo() };
            let on = if k == 0 { layers[i].mute } else { layers[i].solo };
            if on {
                head.rect_filled(r, 2.0, col);
            } else {
                head.rect_stroke(
                    r,
                    2.0,
                    egui::Stroke::new(1.0, if resp.hovered() { col } else { dim }),
                    egui::StrokeKind::Inside,
                );
            }
            head.text(
                r.center(),
                egui::Align2::CENTER_CENTER,
                letter,
                egui::FontId::monospace(9.0),
                if on { vis.extreme_bg_color } else { dim },
            );
        }
        let solo_any = layers.iter().any(|l| l.solo);
        let audible = !layers[i].mute && (layers[i].solo || !solo_any);

        let len = src_of(lens, i, comp_span);
        let (a, b) = extent(&layers[i], len);
        let bar = egui::Rect::from_min_max(
            egui::pos2(f_to_x(a as f32), y + 1.0),
            egui::pos2(f_to_x(b as f32 + 1.0).max(f_to_x(a as f32) + 6.0), y + row_h - 3.0),
        );
        bars.push(bar);

        // AE's hands: an END trims (in/out move, the footage stays put),
        // alt on an end STRETCHES (the other end holds), the body slides,
        // shift on the body slips the footage under its window. Every drag
        // is measured from the PRESS against the layer as it was then, so
        // a slow hand still moves and a drag back lands where it began.
        //
        // The body is registered FIRST and the end handles after, because
        // egui hands a press to the LAST widget that claimed the spot: the
        // ends have to sit ON TOP of the body or they can never be grabbed.
        let br = ui.interact(
            bar.intersect(nav_area),
            ui.id().with(("lay-bar", i)),
            egui::Sense::click_and_drag(),
        );
        // The whole clip behind the bar, faint: what a trim (or a range
        // upstream) hid and how far an end can still be pulled out.
        let clipped = lens.get(i).copied().flatten().is_some();
        // A layer that runs on past its clip (hold / loop / bounce) has no
        // edge for a trim to stop at.
        let bounded = clipped && layers[i].extend.is_none();
        if clipped && footage(&layers[i], len) != (a, b) {
            let (fa, fb) = footage(&layers[i], len);
            let ghost = egui::Rect::from_min_max(
                egui::pos2(f_to_x(fa as f32), bar.min.y + 3.0),
                egui::pos2(f_to_x(fb as f32 + 1.0), bar.max.y - 3.0),
            );
            p.rect_filled(ghost, 1.0, theme::layer_bar().gamma_multiply(0.12));
        }
        let press_id = ui.id().with(("lay-press", i));
        let (alt, shift, hand) = ui.input(|inp| {
            (
                inp.modifiers.alt,
                inp.modifiers.shift,
                inp.pointer
                    .press_origin()
                    .zip(inp.pointer.interact_pos())
                    .map(|(o, q)| q.x - o.x),
            )
        });
        // Frames the hand has travelled since the press.
        let travel = hand.map(|dx| (dx / px_per_frame).round() as i64).unwrap_or(0);
        let apply = |ui: &egui::Ui, r: &egui::Response, layer: &mut LayerItem, edit: BarEdit| {
            if r.drag_started() {
                ui.data_mut(|d| d.insert_temp(press_id, (layer.clone(), edit)));
            }
            if r.drag_stopped() {
                // A snapshot left behind would REVERT the layer on the next
                // drag that missed its start (one begun as a pan, say).
                ui.data_mut(|d| d.remove::<(LayerItem, BarEdit)>(press_id));
            }
            if nav || !r.dragged() {
                return false;
            }
            let Some((was, edit)) = ui.data(|d| d.get_temp::<(LayerItem, BarEdit)>(press_id))
            else {
                return false;
            };
            let now = edit_bar(&was, len, bounded, edit, travel);
            if now != *layer {
                *layer = now;
                return true;
            }
            false
        };
        let hit = 6.0_f32.min(bar.width() * 0.3);
        let mut drag_body = true;
        for (k, hx) in [(0usize, bar.min.x), (1, bar.max.x)] {
            let hr = egui::Rect::from_min_max(
                egui::pos2(hx - hit, bar.min.y),
                egui::pos2(hx + hit, bar.max.y),
            );
            let hr = hr.intersect(nav_area);
            handle_rects.push(hr);
            let r = ui.interact(hr, ui.id().with(("lay-trim", i, k)), egui::Sense::drag());
            if r.hovered() || r.dragged() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
                drag_body = false;
            }
            let edit = if alt { BarEdit::Stretch(k) } else { BarEdit::Trim(k) };
            changed |= apply(ui, &r, &mut layers[i], edit);
            p.rect_filled(
                egui::Rect::from_min_max(
                    egui::pos2(hx - 1.5, bar.min.y + 2.0),
                    egui::pos2(hx + 1.5, bar.max.y - 2.0),
                ),
                0.0,
                if layers[i].trim.is_some() { vis.strong_text_color() } else { dim },
            );
        }

        if br.clicked() {
            action = Some(TimelineAction::Select(i));
        }
        if drag_body && (br.dragged() || br.drag_started()) {
            let edit = if shift { BarEdit::Slip } else { BarEdit::Slide };
            changed |= apply(ui, &br, &mut layers[i], edit);
        }
        // The bar itself. Opacity reads as fill alpha, so a faded layer
        // looks faded — no number to cross-reference.
        let hot = br.hovered() || selected;
        let base = if selected {
            theme::layer_bar_selected()
        } else {
            theme::layer_bar()
        };
        // A layer out of the comp (muted, or not soloed) is a ghost of itself.
        let fade = if audible { 1.0 } else { 0.3 };
        p.rect_filled(bar, 2.0, base.gamma_multiply((0.25 + 0.6 * layers[i].opacity) * fade));
        p.rect_stroke(
            bar,
            2.0,
            egui::Stroke::new(if selected { 1.5 } else { 1.0 }, if hot { base } else { dim }),
            egui::StrokeKind::Inside,
        );
        let label = if layers[i].name.is_empty() {
            names.get(i).cloned().unwrap_or_default()
        } else {
            layers[i].name.clone()
        };
        if !label.is_empty() {
            let mut text = label;
            if layers[i].op != BlendOp::Over {
                text = format!("{text}  ·{}", layers[i].op.name());
            }
            // A retime is invisible in the picture and invisible in the bar
            // (a longer bar could be a longer plate), so it is written on.
            if (layers[i].speed - 1.0).abs() > 1e-3 {
                text = format!("{text}  {:.2}×", 1.0 / layers[i].rate());
            }
            p.with_clip_rect(bar.shrink(3.0)).text(
                egui::pos2(bar.min.x + 4.0, bar.center().y),
                egui::Align2::LEFT_CENTER,
                text,
                egui::FontId::proportional(10.0),
                vis.text_color(),
            );
        }
    }
    ui.data_mut(|d| d.insert_temp(handles_id, handle_rects));

    // --- playhead ---------------------------------------------------------
    // Drawn last so it sits over the bars, and draggable, so the timeline
    // scrubs the comp like the transport bar does.
    let px = f_to_x(playhead as f32 + 0.5);
    p.line_segment(
        [egui::pos2(px, ruler.min.y), egui::pos2(px, track.max.y)],
        egui::Stroke::new(1.0, theme::timeline_playhead()),
    );
    let scrub = ui.interact(
        egui::Rect::from_min_max(
            egui::pos2(track.min.x, ruler.min.y),
            egui::pos2(track.max.x, ruler.max.y),
        ),
        ui.id().with("lay-scrub"),
        egui::Sense::click_and_drag(),
    );
    if !nav && (scrub.dragged() || scrub.clicked()) && action.is_none() {
        if let Some(pos) = scrub.interact_pointer_pos() {
            let f = x_to_f(pos.x).floor().clamp(c0 as f32, (c1 - 1) as f32).max(0.0) as u64;
            action = Some(TimelineAction::Seek(f));
        }
    }
    (action, changed, bars)
}

/// Move a layer within its stack, taking its WIRE and its ANIMATION with
/// it. The three live in different places — the params, the edge list and
/// the curve map — and a reorder that moved only the first would silently
/// repoint every layer's footage.
pub fn swap(g: &mut EditorGraph, node: u64, i: usize, j: usize) {
    if i == j {
        return;
    }
    let Some(n) = g.node_mut(node) else { return };
    if let NodeKind::Layer(p) = &mut n.kind {
        if i >= p.layers.len() || j >= p.layers.len() {
            return;
        }
        p.layers.swap(i, j);
    } else {
        return;
    }
    for (a, b) in crate::anim::layer_channels(i as u16).into_iter().zip(crate::anim::layer_channels(j as u16))
    {
        let (ca, cb) = (n.anim.remove(&a), n.anim.remove(&b));
        if let Some(c) = cb {
            n.anim.insert(a, c);
        }
        if let Some(c) = ca {
            n.anim.insert(b, c);
        }
    }
    for e in &mut g.edges {
        if e.0 == node {
            if e.1 == i {
                e.1 = j;
            } else if e.1 == j {
                e.1 = i;
            }
        }
    }
    // A wire off layer i's own output follows layer i.
    let (oi, oj) = (i as u32 + 1, j as u32 + 1);
    remap_outputs(g, node, |o| {
        Some(if o == oi {
            oj
        } else if o == oj {
            oi
        } else {
            o
        })
    });
}

/// Re-point every wire (and viewer slot) taking one of `node`'s outputs
/// through `f`: `None` drops the wire (a viewer slot falls back to the
/// comp). A stack's per-layer outputs are numbered by layer, so a restack
/// has to renumber them the same way it renumbers the ports, or a wire on
/// "layer 2" would silently show whatever layer lands in slot 2.
pub fn remap_outputs(g: &mut EditorGraph, node: u64, f: impl Fn(u32) -> Option<u32>) {
    let mut drop = Vec::new();
    for &(d, i, s) in &g.edges {
        if s != node {
            continue;
        }
        let o = g.edge_out.get(&(d, i)).copied().unwrap_or(0);
        match f(o) {
            None => drop.push((d, i)),
            Some(0) => {
                g.edge_out.remove(&(d, i));
            }
            Some(n) => {
                g.edge_out.insert((d, i), n);
            }
        }
    }
    g.edges.retain(|&(d, i, s)| !(s == node && drop.contains(&(d, i))));
    for (d, i) in drop {
        g.edge_out.remove(&(d, i));
    }
    for (v, o) in [
        (g.view, &mut g.view_out),
        (g.view_b, &mut g.view_b_out),
        (g.view_c, &mut g.view_c_out),
        (g.view_d, &mut g.view_d_out),
    ] {
        if v == Some(node) {
            *o = f(*o).unwrap_or(0);
        }
    }
}

/// Drop the wires on outputs a stack no longer has: every per-layer output
/// once "layer outputs" is switched off. Dropped, not quietly moved onto the
/// comp — a Grade that was correcting one layer must not start correcting
/// the whole stack without anyone asking.
pub fn prune_outputs(g: &mut EditorGraph, node: u64) {
    let n = match g.node(node).map(|n| &n.kind) {
        Some(k @ NodeKind::Layer(_)) => k.output_count() as u32,
        _ => return,
    };
    remap_outputs(g, node, |o| (o < n).then_some(o));
}

/// Insert a layer at the TOP of the stack (index 0, AE numbering), or
/// remove the one at `del`. Both renumber everything below the change:
/// the ports, the wires and the per-layer animation channels.
pub fn restack(g: &mut EditorGraph, node: u64, del: Option<usize>) {
    match del {
        Some(k) => restack_at(g, node, Restack::Remove(k)),
        None => restack_at(g, node, Restack::Insert(0)),
    }
}

/// A change to a stack's layer list: a new empty layer at an index, or the
/// layer at one taken out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Restack {
    Insert(usize),
    Remove(usize),
}

/// [`restack`] at any index: the ports, wires and per-layer channels below
/// the change renumber with it.
pub fn restack_at(g: &mut EditorGraph, node: u64, op: Restack) {
    let Some(n) = g.node_mut(node) else { return };
    let NodeKind::Layer(p) = &mut n.kind else { return };
    let n_before = p.layers.len();
    let (del, at) = match op {
        Restack::Remove(k) if k < n_before && n_before > 1 => {
            p.layers.remove(k);
            (Some(k), 0)
        }
        Restack::Remove(_) => return,
        Restack::Insert(k) if n_before < engine::graph::LAYER_MAX => {
            let k = k.min(n_before);
            p.layers.insert(k, LayerItem::default());
            (None, k)
        }
        Restack::Insert(_) => return,
    };
    // Animation: drop the removed layer's curves, then slide the rest.
    // Ascending for a delete, descending for an insert, so a shift never
    // lands on a slot it has not vacated yet.
    let shift: Vec<(usize, usize)> = match del {
        Some(k) => {
            for ch in crate::anim::layer_channels(k as u16) {
                n.anim.remove(&ch);
            }
            (k + 1..n_before).map(|i| (i, i - 1)).collect()
        }
        None => (at..n_before).rev().map(|i| (i, i + 1)).collect(),
    };
    for (from, to) in &shift {
        for (a, b) in crate::anim::layer_channels(*from as u16)
            .into_iter()
            .zip(crate::anim::layer_channels(*to as u16))
        {
            if let Some(c) = n.anim.remove(&a) {
                n.anim.insert(b, c);
            }
        }
    }
    // Wires: the removed layer's edge goes, the rest slide with it.
    if let Some(k) = del {
        g.edges.retain(|e| !(e.0 == node && e.1 == k));
    }
    for (from, to) in &shift {
        for e in &mut g.edges {
            if e.0 == node && e.1 == *from {
                e.1 = *to;
            }
        }
    }
    // The per-layer outputs renumber the same way: the removed layer's wires
    // go, the rest follow their layers.
    remap_outputs(g, node, |o| {
        if o == 0 {
            return Some(0);
        }
        let l = o as usize - 1;
        if del == Some(l) {
            return None;
        }
        Some(shift.iter().find(|(f, _)| *f == l).map_or(o, |(_, t)| *t as u32 + 1))
    });
}

/// The transform manipulator, drawn in the viewer and dragged against the
/// evaluated params. `to_screen` maps this node's OUTPUT pixels to screen
/// points through whatever deformation lies downstream; `from_screen` is its
/// exact inverse. `size` is the image's own format — the frame the outline
/// and the eight border boxes span.
///
/// Returns true if `t` changed, which is the caller's cue to write it back
/// (keys on animated channels, authored values otherwise).
/// Put the pivot at `q` (the node's OUTPUT pixels) without moving the
/// picture: the new centre is the SOURCE point that already lands at `q`,
/// and translate is re-derived so the forward affine comes out unchanged.
/// Rotate and scale stay as they are; only what they turn about moves.
pub fn move_pivot(t: &mut TransformParams, size: [f32; 2], q: egui::Vec2) {
    let m = engine::codegen::forward_affine(t, size);
    let c = crate::apply_aff(&crate::invert_aff(&m), q);
    t.center = [c.x / size[0].max(1e-6), c.y / size[1].max(1e-6)];
    // On screen the pivot sits at translate + centre (pixels), so this puts
    // it exactly under `q`; with the linear part untouched it also leaves
    // the affine's tx/ty where they were.
    t.translate = [q.x - c.x, q.y - c.y];
}

pub fn transform_gizmo(
    ui: &mut egui::Ui,
    ink: &mut InkShapes,
    id: egui::Id,
    t: &mut TransformParams,
    size: (f32, f32),
    frame: egui::Rect,
    to_screen: &dyn Fn(egui::Vec2) -> egui::Pos2,
    from_screen: &dyn Fn(egui::Pos2) -> egui::Vec2,
) -> bool {
    let before = *t;
    let (lwf, lhf) = size;
    // The box the outline and the eight border handles span, in the node's
    // input pixels: the whole format, or a SoftCrop's box upstream. The
    // pivot and the maths stay the format's — only the handles move.
    let (fx0, fy0, fx1, fy1) = (frame.min.x, frame.min.y, frame.max.x, frame.max.y);
    let fpt = |h: i8, lo: f32, hi: f32| lo + (h as f32 + 1.0) * 0.5 * (hi - lo);
    let pointer_pos = ui.ctx().pointer_latest_pos();
    let m = engine::codegen::forward_affine(t, [lwf, lhf]);
    // The whole map — the corner pin, then the affine — for drawing; the
    // move/rotate/scale maths below works after the pin, in `m`'s space.
    let img_pt =
        |t: &TransformParams, pimg: [f32; 2]| -> egui::Pos2 {
            let h = engine::codegen::transform_forward_px(t, [lwf, lhf]);
            to_screen(crate::apply_h(&h, egui::vec2(pimg[0], pimg[1])))
        };
    // Where the pin puts a point of the frame (before the affine).
    let pin_h = engine::codegen::corner_pin_forward_px(
        &engine::graph::CornerPinParams::pinned(t.pin),
        [lwf, lhf],
    );
    let pinned = |v: egui::Vec2| crate::apply_h(&pin_h, v);

    // Transformed frame outline. Homographies keep lines straight, but a
    // UV Warp downstream bends them, so each edge is walked in steps.
    let corners = [[fx0, fy0], [fx1, fy0], [fx1, fy1], [fx0, fy1]];
    const EDGE_STEPS: usize = 16;
    let mut outline = Vec::with_capacity(4 * EDGE_STEPS);
    for k in 0..4 {
        let (a, b) = (corners[k], corners[(k + 1) % 4]);
        let at = |i: usize| {
            let s = i as f32 / EDGE_STEPS as f32;
            img_pt(t, [a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s])
        };
        outline.extend((0..EDGE_STEPS).map(at));
    }
    ink.path(outline, true, 1.0, 0.55, None);

    // Pivot in the node's output space and on screen.
    let pivot_out =
        egui::vec2(t.center[0] * lwf + t.translate[0], t.center[1] * lhf + t.translate[1]);
    let pivot_pts = to_screen(pivot_out);

    let rad = t.rotate.to_radians();
    let (sn, cs) = rad.sin_cos();
    let rot_inv = |v: egui::Vec2| egui::vec2(cs * v.x + sn * v.y, -sn * v.x + cs * v.y);
    // Screen direction of a node-local angle, via finite differences through
    // the downstream chain (exact for the projective case).
    let ndir = |a: f32| {
        let v = egui::vec2(a.cos(), a.sin());
        let w = egui::vec2(cs * v.x - sn * v.y, sn * v.x + cs * v.y);
        let d = to_screen(pivot_out + w * 4.0) - pivot_pts;
        let len = d.length();
        if len > 1e-5 { d / len } else { w }
    };

    // 1) translate: open circle at the pivot.
    let rt = ui.interact(
        egui::Rect::from_center_size(pivot_pts, egui::vec2(28.0, 28.0)),
        id.with("g-move"),
        egui::Sense::drag(),
    );
    ink.circle_stroke(pivot_pts, 12.0, 2.0, 1.0, rt.hovered());
    // drag_started is skipped everywhere: egui's first-frame delta spans all
    // motion since before the press (a hitch or coalesced input turns that
    // into a teleport).
    if rt.dragged() && !rt.drag_started() {
        if let Some(p) = pointer_pos {
            let d = from_screen(p)
                - from_screen(p - crate::widgets::rated_delta(ui, &rt));
            // D held: move the PIVOT, not the picture (the usual ctrl-drag on
            // the centre; ctrl/cmd-CLICK belong to the viewer's pick). The
            // plain drag on this handle translates, so the key is what says
            // "just the point".
            if ui.input(|i| i.key_down(egui::Key::D)) {
                move_pivot(t, [lwf, lhf], pivot_out + d);
            } else {
                t.translate[0] += d.x;
                t.translate[1] += d.y;
            }
        }
    }

    // 2) rotate: the WHOLE ring is grabbable (no ticks). The interact only
    // registers while the pointer is in the annulus band or its drag is live,
    // so it never steals the pivot / nav.
    let ring_r = 92.0;
    let ring_id = id.with("g-rot");
    let ring_live = ui.ctx().dragged_id() == Some(ring_id);
    let near_ring =
        pointer_pos.is_some_and(|p| ((p - pivot_pts).length() - ring_r).abs() <= 14.0);
    let mut ring_hot = ring_live;
    if near_ring || ring_live {
        let rr = ui.interact(
            egui::Rect::from_center_size(pivot_pts, egui::Vec2::splat((ring_r + 14.0) * 2.0)),
            ring_id,
            egui::Sense::drag(),
        );
        ring_hot |= rr.hovered();
        if rr.dragged() && !rr.drag_started() {
            if let Some(p) = pointer_pos {
                let prev = p - crate::widgets::rated_delta(ui, &rr);
                let mut d = (from_screen(p) - pivot_out).angle()
                    - (from_screen(prev) - pivot_out).angle();
                use std::f32::consts::{PI, TAU};
                if d > PI {
                    d -= TAU;
                } else if d < -PI {
                    d += TAU;
                }
                t.rotate += d.to_degrees();
            }
        }
    }
    ink.circle_stroke(
        pivot_pts,
        ring_r,
        if ring_hot { 2.5 } else { 1.5 },
        if ring_hot { 1.0 } else { 0.8 },
        ring_hot,
    );
    // Faint orientation ticks riding rotation + deformation — drawing only,
    // the grab is the whole ring.
    for k in 0..12 {
        let dir = ndir(k as f32 * std::f32::consts::TAU / 12.0);
        ink.seg(
            pivot_pts + dir * (ring_r - 3.5),
            pivot_pts + dir * (ring_r + 3.5),
            1.0,
            if ring_hot { 0.7 } else { 0.35 },
            ring_hot,
        );
    }

    // Sign-preserving: dragging across the anchor flips (negative scale =
    // flip-flop); only the magnitude is clamped.
    let cl = |v: f32| {
        let sg = if v < 0.0 { -1.0 } else { 1.0 };
        sg * v.abs().clamp(0.05, 20.0)
    };
    const HANDLES: [(i8, i8); 8] =
        [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)];

    // 3) border boxes: Illustrator-style, anchored at the OPPOSITE handle.
    // Ratios are computed back in the node's own space, so the drag stays 1:1
    // even through a downstream corner pin.
    for (k, (hx, hy)) in HANDLES.iter().enumerate() {
        let ix = fpt(*hx, fx0, fx1);
        let iy = fpt(*hy, fy0, fy1);
        let hp = img_pt(t, [ix, iy]);
        let rs = ui.interact(
            egui::Rect::from_center_size(hp, egui::vec2(24.0, 24.0)),
            id.with(("g-scale", k)),
            egui::Sense::drag(),
        );
        ink.rect(
            egui::Rect::from_center_size(hp, egui::vec2(16.0, 16.0)),
            2.0,
            true,
            0.0,
            1.0,
            rs.hovered(),
        );
        // Ctrl on a CORNER is Photoshop's distort: that corner goes where
        // the pointer is and the other three stay put — a corner pin, stored
        // on the Transform and solved around the handle box (the soft crop's,
        // when there is one). Absolute, so an animated pin keys cleanly.
        let distort = *hx != 0 && *hy != 0 && ui.input(|i| i.modifiers.ctrl);
        if distort && rs.dragged() {
            if let Some(p) = pointer_pos {
                // The pointer in the pin's space: back through the affine.
                let a = crate::apply_aff(&crate::invert_aff(&m), from_screen(p));
                let box_n = [fx0 / lwf, fy0 / lhf, fx1 / lwf, fy1 / lhf];
                let mut pins = crate::nodes::softcrop::pins_on_crop(t.pin, box_n);
                let i = match (hx, hy) {
                    (-1, -1) => 0,
                    (1, -1) => 1,
                    (1, 1) => 2,
                    _ => 3,
                };
                pins[i] = [a.x / lwf, a.y / lhf];
                t.pin = crate::nodes::softcrop::to_from_crop_pins(pins, box_n);
            }
            continue;
        }
        if rs.dragged() && !rs.drag_started() {
            if let Some(p) = pointer_pos {
                // Cmd: Photoshop-style, anchored at the PIVOT instead of the
                // opposite handle (still 1:1 under the cursor). The pivot
                // already lives after the pin; a handle is pinned first.
                let about_pivot = ui.input(|i| i.modifiers.mac_cmd && !i.modifiers.ctrl);
                let a_img = if about_pivot {
                    egui::vec2(t.center[0] * lwf, t.center[1] * lhf)
                } else {
                    pinned(egui::vec2(fpt(-*hx, fx0, fx1), fpt(-*hy, fy0, fy1)))
                };
                let a_out = crate::apply_aff(&m, a_img);
                let prev = p - crate::widgets::rated_delta(ui, &rs);
                let ln = rot_inv(from_screen(p) - a_out);
                let lp = rot_inv(from_screen(prev) - a_out);
                let mut fx = 1.0f32;
                let mut fy = 1.0f32;
                // Negative ratio = the drag crossed the anchor: flip.
                if *hx != 0 && lp.x.abs() > 1.0 {
                    fx = ln.x / lp.x;
                }
                if *hy != 0 && lp.y.abs() > 1.0 {
                    fy = ln.y / lp.y;
                }
                if fx != 0.0 && fy != 0.0 {
                    t.scale[0] = cl(t.scale[0] * fx);
                    t.scale[1] = cl(t.scale[1] * fy);
                    // Re-pin the anchor exactly, in output space.
                    let m2 = engine::codegen::forward_affine(t, [lwf, lhf]);
                    let a_new = crate::apply_aff(&m2, a_img);
                    t.translate[0] += a_out.x - a_new.x;
                    t.translate[1] += a_out.y - a_new.y;
                }
            }
        }
    }

    // 4) orbit handles (scale cluster up-left, translate arrows down-right),
    // all in the deformed local frame.
    #[derive(Clone, Copy, PartialEq)]
    enum Orbit {
        ScaleX,
        ScaleY,
        ScaleXY,
        ScaleUniform,
        TransX,
        TransY,
        TransDiag,
    }
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};
    let handles = [
        (PI, ring_r + 26.0, Orbit::ScaleX),
        (-FRAC_PI_2, ring_r + 26.0, Orbit::ScaleY),
        (-PI + FRAC_PI_4, ring_r + 26.0, Orbit::ScaleXY),
        (-PI + FRAC_PI_4, ring_r + 54.0, Orbit::ScaleUniform),
        (0.0, ring_r + 26.0, Orbit::TransX),
        (FRAC_PI_2, ring_r + 26.0, Orbit::TransY),
        (FRAC_PI_4, ring_r + 26.0, Orbit::TransDiag),
    ];
    let lx_axis = egui::vec2(cs, sn);
    let ly_axis = egui::vec2(-sn, cs);
    for (k, (ang, dist, kind)) in handles.iter().enumerate() {
        let dir = ndir(*ang);
        let bp = pivot_pts + dir * *dist;
        let rs = ui.interact(
            egui::Rect::from_center_size(bp, egui::vec2(20.0, 20.0)),
            id.with(("g-orbit", k)),
            egui::Sense::drag(),
        );
        let hov = rs.hovered();
        let is_translate = matches!(kind, Orbit::TransX | Orbit::TransY | Orbit::TransDiag);
        if is_translate {
            let a0 = bp - dir * 11.0;
            let a1 = bp + dir * 11.0;
            ink.seg(a0, a1, 2.0, 1.0, hov);
            for (tip, d) in [(a1, dir), (a0, -dir)] {
                let n = egui::vec2(-d.y, d.x);
                ink.seg(tip, tip - d * 6.0 + n * 4.0, 2.0, 1.0, hov);
                ink.seg(tip, tip - d * 6.0 - n * 4.0, 2.0, 1.0, hov);
            }
        } else {
            let side = if *kind == Orbit::ScaleUniform { 15.0 } else { 12.0 };
            ink.rect(
                egui::Rect::from_center_size(bp, egui::vec2(side, side)),
                1.0,
                false,
                1.5,
                1.0,
                hov,
            );
        }
        if !rs.dragged() || rs.drag_started() {
            continue;
        }
        let Some(p) = pointer_pos else { continue };
        let prev = p - crate::widgets::rated_delta(ui, &rs);
        let pn = from_screen(p);
        let pv = from_screen(prev);
        match kind {
            Orbit::TransX | Orbit::TransY | Orbit::TransDiag => {
                let d = rot_inv(pn - pv);
                let world = match kind {
                    Orbit::TransX => lx_axis * d.x,
                    Orbit::TransY => ly_axis * d.y,
                    _ => {
                        let s = (d.x + d.y) * std::f32::consts::FRAC_1_SQRT_2;
                        (lx_axis + ly_axis) * std::f32::consts::FRAC_1_SQRT_2 * s
                    }
                };
                t.translate[0] += world.x;
                t.translate[1] += world.y;
            }
            _ => {
                let ln = rot_inv(pn - pivot_out);
                let lp = rot_inv(pv - pivot_out);
                let half = |f: f32| 1.0 + (f - 1.0) * 0.5;
                match kind {
                    Orbit::ScaleX => {
                        if lp.x.abs() > 1.0 {
                            let f = half((ln.x / lp.x).max(0.0));
                            if f > 0.0 {
                                t.scale[0] = cl(t.scale[0] * f);
                            }
                        }
                    }
                    Orbit::ScaleY => {
                        if lp.y.abs() > 1.0 {
                            let f = half((ln.y / lp.y).max(0.0));
                            if f > 0.0 {
                                t.scale[1] = cl(t.scale[1] * f);
                            }
                        }
                    }
                    Orbit::ScaleXY => {
                        if lp.x.abs() > 1.0 {
                            let f = half((ln.x / lp.x).max(0.0));
                            if f > 0.0 {
                                t.scale[0] = cl(t.scale[0] * f);
                            }
                        }
                        if lp.y.abs() > 1.0 {
                            let f = half((ln.y / lp.y).max(0.0));
                            if f > 0.0 {
                                t.scale[1] = cl(t.scale[1] * f);
                            }
                        }
                    }
                    _ => {
                        let (dn, dp) =
                            ((pn - pivot_out).length(), (pv - pivot_out).length());
                        if dp > 1.0 {
                            let f = half((dn / dp).max(0.0));
                            if f > 0.0 {
                                t.scale = [cl(t.scale[0] * f), cl(t.scale[1] * f)];
                            }
                        }
                    }
                }
            }
        }
    }
    *t != before
}


/// What the panel asked the app to do, applied once the node borrow ends.
#[derive(Default)]
pub struct PanelOut {
    pub action: Option<TimelineAction>,
    /// A layer's own values changed (a slide, a stretch, a trim, a fade).
    pub changed: bool,
    /// Where each bar landed, for the self-test and for anything that wants
    /// to point at one.
    pub bars: Vec<egui::Rect>,
    pub add: bool,
    pub del: Option<usize>,
    /// A row was ctrl-dragged out to become a viewer HUD.
    pub hud: Option<crate::widgets::HudSlider>,
    /// The pop-out button was pressed.
    pub popout: bool,
}

/// The Layer node's whole panel: the mini timeline, the stack's controls, and
/// the selected layer's own. ONE function, because it is drawn in two places
/// — docked in the properties strip and popped out into a window you can
/// size — and a panel that drifted between the two would be two panels.
#[allow(clippy::too_many_arguments)]
pub fn panel(
    ui: &mut egui::Ui,
    lp: &mut engine::graph::LayerParams,
    rcx: &mut crate::widgets::RowCtx<'_>,
    node: u64,
    sel: usize,
    names: &[String],
    lens: &[Option<Src>],
    comp_span: (u64, u64),
    playhead: u64,
    ranges: crate::minibar::Ranges,
    canvas: &mut Option<(u32, u32)>,
    default_canvas: (u32, u32),
    global: Option<(u32, u32)>,
    offer_popout: bool,
) -> PanelOut {
    use crate::anim::Channel::*;
    let mut out = PanelOut::default();
    let n = lp.layers.len();
    let sel = sel.min(n.saturating_sub(1));

    // The timeline first: it is the map of the node, and everything under it
    // edits whatever it has selected.
    let (act, moved, bars) =
        timeline(ui, &mut lp.layers, names, lens, sel, comp_span, playhead, 20.0, ranges);
    out.action = act;
    out.changed |= moved;
    out.bars = bars;

    ui.horizontal(|ui| {
        if ui
            .add_enabled(n < engine::graph::LAYER_MAX, egui::Button::new("+ layer"))
            .on_hover_text("add a layer on top of the stack")
            .clicked()
        {
            out.add = true;
        }
        if ui
            .add_enabled(n > 1, egui::Button::new("− layer"))
            .on_hover_text("remove the selected layer")
            .clicked()
        {
            out.del = Some(sel);
        }
        ui.weak(format!("{n} layers · 1 is the top"));
        if offer_popout {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                out.popout = ui
                    .button("⤴")
                    .on_hover_text("pop the stack out into a window you can size")
                    .clicked();
            });
        }
    });
    // A stack is a comp, and a comp usually wants a STATED size rather than
    // whatever the bottom plate happens to be. Same control a Blend has.
    ui.horizontal(|ui| {
        let mut on = canvas.is_some();
        if ui
            .checkbox(&mut on, "canvas size")
            .on_hover_text(
                "fix this stack's output size; else it inherits the bottom \
                 layer's resolution",
            )
            .changed()
        {
            *canvas = on.then_some(default_canvas);
        }
        if let Some((w, h)) = canvas {
            crate::widgets::num_field(ui, egui::DragValue::new(w).update_while_editing(false).speed(crate::widgets::rated(ui, 4)).range(16..=16384),
            );
            crate::widgets::num_field(ui, egui::DragValue::new(h).update_while_editing(false).speed(crate::widgets::rated(ui, 4)).range(16..=16384),
            );
        }
        // The script's format (the globals… window, else the plates'):
        // one click from whatever the bottom layer is to the shot's frame.
        let tip = match global {
            Some((w, h)) => format!("set the canvas to the global format, {w}x{h}"),
            None => "no global format: set one in globals…, or load a plate".to_string(),
        };
        if ui
            .add_enabled(global.is_some() && *canvas != global, egui::Button::new("global"))
            .on_hover_text(tip)
            .on_disabled_hover_text(match global {
                Some(_) => "the canvas is already the global format",
                None => "no global format: set one in globals…, or load a plate",
            })
            .clicked()
        {
            *canvas = global;
        }
    });
    if ui
        .checkbox(&mut lp.outputs, "layer outputs")
        .on_hover_text(
            "an output per layer: the layer alone, with its start, speed and \
             trim but no opacity, blend or other layers. Switching it off \
             drops the wires on those outputs",
        )
        .changed()
    {
        out.changed = true;
    }
    ui.separator();

    if let Some(l) = lp.layers.get_mut(sel) {
        let i = sel as u16;
        let len = src_of(lens, sel, comp_span);
        ui.horizontal(|ui| {
            ui.strong(format!("layer {}", sel + 1));
            let mut name = l.name.clone();
            if ui
                .add(
                    egui::TextEdit::singleline(&mut name)
                        .desired_width(110.0)
                        .hint_text("name"),
                )
                .changed()
            {
                l.name = name;
            }
            egui::ComboBox::from_id_salt(("lay-op", node, sel))
                .selected_text(l.op.name())
                .show_ui(ui, |ui| {
                    for op in BlendOp::ALL {
                        ui.selectable_value(&mut l.op, op, op.name());
                    }
                });
        });
        // Placement is not the layer's: an upstream Transform moves it.
        let h =
            crate::widgets::slider(ui, rcx, "opacity", &mut l.opacity, 0.0, 1.0, &[LayerOpacity(i)]);
        ui.horizontal(|ui| {
            // The bar's position and its rate, typed rather than dragged —
            // the timeline is for feel, these are for stating a number.
            let mut st = l.start as f32;
            if ui
                .add(egui::DragValue::new(&mut st).update_while_editing(false).speed(crate::widgets::rated(ui, 1.0)).prefix("offset "))
                .on_hover_text("how far the layer is slid in comp frames: 0 plays the input at its own frame numbers")
                .changed()
            {
                l.start = st.round() as i64;
                out.changed = true;
            }
            // Stretch, AE's way: 200% is twice as long and half as fast,
            // about the in point. Negative runs the layer backwards.
            let mut pct = 100.0 / l.rate();
            if ui
                .add(
                    egui::DragValue::new(&mut pct).update_while_editing(false)
                        .speed(crate::widgets::rated(ui, 1.0))
                        .range(-10000.0..=10000.0)
                        .max_decimals(1)
                        .prefix("stretch ")
                        .suffix("%"),
                )
                .on_hover_text(
                    "how long the layer takes against its own length: 200% is \
                     twice as long at half speed (alt-drag an end does the same). \
                     The in point holds",
                )
                .changed()
                && pct.abs() >= 1.0
            {
                *l = stretch_to(l, len, 100.0 / pct);
                out.changed = true;
            }
        });
        // Past the clip: stop there (a trim cannot reach further), or run
        // on with the clip held, looped or bounced. Only for footage — a
        // generator never ends.
        if lens.get(sel).copied().flatten().is_some() {
            use engine::graph::{LayerExtend, ReadEdge};
            let clip = len.clip;
            let cur = l.extend.map(|e| e.edge);
            ui.horizontal(|ui| {
                ui.label("past the clip").on_hover_text(
                    "what the layer shows beyond its clip's first and last frame: \
                     stop keeps a trim inside the clip; hold, loop and bounce let \
                     an end drag on past it",
                );
                let mut pick = cur;
                for (v, name) in [
                    (None, "stop"),
                    (Some(ReadEdge::Hold), "hold"),
                    (Some(ReadEdge::Loop), "loop"),
                    (Some(ReadEdge::Bounce), "bounce"),
                ] {
                    ui.selectable_value(&mut pick, v, name);
                }
                let want = pick.map(|edge| LayerExtend { edge, lo: clip.0, hi: clip.1 });
                // Also follows the clip itself (a reload, a longer render).
                if want != l.extend {
                    l.extend = want;
                    if want.is_none() {
                        // Back inside the clip: a trim past it would show
                        // frames that are no longer there.
                        if let Some((ta, tb)) = l.trim {
                            let (fa, fb) = footage(l, len);
                            let (ta, tb) = (ta.min(tb).max(fa).min(fb), ta.max(tb).min(fb).max(fa));
                            l.trim = Some((ta, tb));
                        }
                    }
                    out.changed = true;
                }
            });
        }
        // In / out / duration: the trim, typed. Editing any of them trims an
        // untrimmed layer to its own footage first, as dragging an end does.
        ui.horizontal(|ui| {
            let (a, b) = extent(l, len);
            let (mut ia, mut ib, mut dur) = (a as f32, b as f32, (b - a + 1) as f32);
            let field = |ui: &mut egui::Ui, v: &mut f32, pre: &str, tip: &str| {
                ui.add(
                    egui::DragValue::new(v)
                        .update_while_editing(false)
                        .speed(crate::widgets::rated(ui, 1.0))
                        .max_decimals(0)
                        .prefix(pre),
                )
                .on_hover_text(tip)
                .changed()
            };
            let mut t = None;
            if field(ui, &mut ia, "in ", "the first comp frame this layer shows") {
                t = Some(((ia.round() as i64).min(b), b));
            }
            if field(ui, &mut ib, "out ", "the last comp frame this layer shows") {
                t = Some((a, (ib.round() as i64).max(a)));
            }
            if field(ui, &mut dur, "dur ", "how many comp frames it shows, from the in point") {
                t = Some((a, a + (dur.round() as i64).max(1) - 1));
            }
            if let Some(t) = t {
                l.trim = Some(t);
                out.changed = true;
            }
            if l.trim.is_some()
                && ui
                    .button("untrim")
                    .on_hover_text("show this layer for the whole comp")
                    .clicked()
            {
                l.trim = None;
                out.changed = true;
            }
        });
        if h.is_some() {
            out.hud = h;
        }
    }
    ui.weak(
        "drag a bar to slide (shift: slip) · its ends to trim (alt: stretch)",
    );
    out
}

/// The popped-out stack: the same panel, in a window you can size. A floating
/// window CAN occlude the viewer's gizmos (see the trap in CLAUDE.md), so it
/// opens away from the middle and is only ever there because you asked.
#[allow(clippy::too_many_arguments)]
pub fn window(
    ctx: &egui::Context,
    node: u64,
    title: &str,
    lp: &mut engine::graph::LayerParams,
    rcx: &mut crate::widgets::RowCtx<'_>,
    sel: usize,
    names: &[String],
    lens: &[Option<Src>],
    comp_span: (u64, u64),
    playhead: u64,
    ranges: crate::minibar::Ranges,
    canvas: &mut Option<(u32, u32)>,
    default_canvas: (u32, u32),
    global: Option<(u32, u32)>,
) -> (PanelOut, bool) {
    let mut open = true;
    let mut out = PanelOut::default();
    egui::Window::new(format!("{title} — layers"))
        .id(egui::Id::new(("layer-window", node)))
        .open(&mut open)
        .resizable(true)
        .default_size([620.0, 420.0])
        .default_pos([40.0, 60.0])
        .show(ctx, |ui| {
            out = panel(
                ui,
                lp,
                rcx,
                node,
                sel,
                names,
                lens,
                comp_span,
                playhead,
                ranges,
                canvas,
                default_canvas,
                global,
                false,
            );
        });
    (out, open)
}


#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use egui::epaint;

    fn item(start: i64, trim: Option<(i64, i64)>) -> LayerItem {
        LayerItem { start, trim, ..Default::default() }
    }

    /// Software-rasterize an egui frame so a panel's LAYOUT can be looked
    /// at, not just asserted about. Glyphs are skipped (they need the font
    /// atlas); everything else is flat-shaded from the vertex colours.
    /// `keyerui`'s tests carry a copy of this — worth merging into one
    /// helper next time either is touched.
    pub(crate) fn rasterize(
        prims: &[egui::ClippedPrimitive],
        w: usize,
        h: usize,
        ss: f32,
    ) -> Vec<u8> {
        let mut buf = vec![18u8; w * h * 3];
        for p in prims {
            let egui::epaint::Primitive::Mesh(m) = &p.primitive else { continue };
            // egui carries the clip as metadata on the primitive; a
            // rasterizer that ignores it shows shapes the app never draws
            // (this test claimed the plots leaked across the panel).
            let clip = [
                (p.clip_rect.min.x * ss).max(0.0) as usize,
                (p.clip_rect.min.y * ss).max(0.0) as usize,
                (p.clip_rect.max.x * ss).min(w as f32) as usize,
                (p.clip_rect.max.y * ss).min(h as f32) as usize,
            ];
            for tri in m.indices.chunks_exact(3) {
                let v: Vec<&epaint::Vertex> =
                    tri.iter().map(|i| &m.vertices[*i as usize]).collect();
                // Non-white UVs mean a font-atlas lookup: skip the glyphs.
                if v.iter().any(|q| (q.uv.x - epaint::WHITE_UV.x).abs() > 1e-3) {
                    continue;
                }
                let pts: Vec<[f32; 2]> =
                    v.iter().map(|q| [q.pos.x * ss, q.pos.y * ss]).collect();
                let (x0, x1) = (
                    (pts.iter().map(|q| q[0]).fold(f32::MAX, f32::min).floor().max(0.0)
                        as usize)
                        .max(clip[0]),
                    (pts.iter().map(|q| q[0]).fold(f32::MIN, f32::max).ceil().min(w as f32)
                        as usize)
                        .min(clip[2]),
                );
                let (y0, y1) = (
                    (pts.iter().map(|q| q[1]).fold(f32::MAX, f32::min).floor().max(0.0)
                        as usize)
                        .max(clip[1]),
                    (pts.iter().map(|q| q[1]).fold(f32::MIN, f32::max).ceil().min(h as f32)
                        as usize)
                        .min(clip[3]),
                );
                let d = (pts[1][0] - pts[0][0]) * (pts[2][1] - pts[0][1])
                    - (pts[2][0] - pts[0][0]) * (pts[1][1] - pts[0][1]);
                if d.abs() < 1e-9 {
                    continue;
                }
                for y in y0..y1 {
                    for x in x0..x1 {
                        let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                        let b1 = ((px - pts[0][0]) * (pts[2][1] - pts[0][1])
                            - (pts[2][0] - pts[0][0]) * (py - pts[0][1]))
                            / d;
                        let b2 = ((pts[1][0] - pts[0][0]) * (py - pts[0][1])
                            - (px - pts[0][0]) * (pts[1][1] - pts[0][1]))
                            / d;
                        let b0 = 1.0 - b1 - b2;
                        if b0 < 0.0 || b1 < 0.0 || b2 < 0.0 {
                            continue;
                        }
                        let ws = [b0, b1, b2];
                        let mut c = [0.0f32; 4];
                        for (k, q) in v.iter().enumerate() {
                            let a = q.color.to_array();
                            for j in 0..4 {
                                c[j] += a[j] as f32 * ws[k];
                            }
                        }
                        // egui colours are premultiplied.
                        let a = c[3] / 255.0;
                        let o = (y * w + x) * 3;
                        for j in 0..3 {
                            buf[o + j] =
                                (c[j] + buf[o + j] as f32 * (1.0 - a)).clamp(0.0, 255.0) as u8;
                        }
                    }
                }
            }
        }
        buf
    }

    /// Draw a four-layer stack the way the panel does — one layer slid late,
    /// one trimmed, one faded — and write `target/layer-timeline.png`. The
    /// assertions pin what the picture is FOR: bars sit where their frames
    /// say, a slid layer is further right than an unslid one, and a trimmed
    /// one is shorter. The picture itself is for a human to judge.
    #[test]
    fn timeline_renders_to_a_png() {
        let (w_pt, h_pt, ss) = (620.0f32, 120.0f32, 2.0f32);
        let ctx = egui::Context::default();
        let mut layers = vec![
            LayerItem { name: "title".into(), start: 40, ..Default::default() },
            LayerItem { name: "smoke".into(), opacity: 0.35, ..Default::default() },
            LayerItem { name: "fg".into(), trim: Some((20, 60)), ..Default::default() },
            // Stretched to half speed: twice the bar for the same footage.
            LayerItem { name: "bg plate".into(), speed: 0.5, ..Default::default() },
        ];
        let names: Vec<String> = vec![String::new(); 4];
        let lens = vec![Some(Src::from(48)), None, Some(Src::from(96)), Some(Src::from(96))];
        let mut input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(w_pt, h_pt),
            )),
            ..Default::default()
        };
        input.viewports.values_mut().for_each(|v| v.native_pixels_per_point = Some(1.0));
        let mut bars = Vec::new();
        let out = ctx.run_ui(input, |ui| {
            let (_, _, b) =
                timeline(
                    ui,
                    &mut layers,
                    &names,
                    &lens,
                    2,
                    (0, 95),
                    30,
                    20.0,
                    crate::minibar::Ranges { comp: (0, 95), node: (0, 95), inout: None },
                );
            bars = b;
        });
        assert_eq!(bars.len(), 4, "one bar per layer");
        assert!(bars[0].min.x > bars[3].min.x, "a layer starting at 40 must sit right of one at 0");
        assert!(bars[2].width() < bars[3].width(), "a trimmed layer must be shorter");
        assert!(
            bars[3].width() > bars[0].width() * 1.5,
            "a half-speed layer must be about twice as long a bar"
        );
        assert!(bars[1].min.y < bars[2].min.y, "rows must run top to bottom in stack order");

        let prims = ctx.tessellate(out.shapes, 1.0);
        let (w, h) = ((w_pt * ss) as usize, (h_pt * ss) as usize);
        let buf = rasterize(&prims, w, h, ss);
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../target/layer-timeline.png");
        let file = std::fs::File::create(&path).unwrap();
        let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w as u32, h as u32);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header().unwrap().write_image_data(&buf).unwrap();
        eprintln!("wrote {}", path.display());
    }

    #[test]
    fn untrimmed_bar_follows_the_start() {
        assert_eq!(extent(&item(0, None), 24), (0, 23));
        assert_eq!(extent(&item(10, None), 24), (10, 33));
        assert_eq!(extent(&item(-5, None), 24), (-5, 18));
    }

    /// A plate placed at 1018..1080 sits THERE with start 0, because the
    /// layer asks its input for the comp frame's own number; a slide, a
    /// trim and a stretch all keep working from where it really is.
    #[test]
    fn a_placed_plate_sits_on_its_own_frames() {
        let plate = Src::from((1018, 1080));
        let l = item(0, None);
        assert_eq!(extent(&l, plate), (1018, 1080));
        assert_eq!(extent(&item(-18, None), plate), (1000, 1062));
        let t = edit_bar(&l, plate, true, BarEdit::Trim(1), 100);
        assert_eq!(t.trim, Some((1018, 1080)), "a trim stops at the plate's own end");
        // Stretched to twice its length from the out end: the in point holds
        // and still plays frame 1018.
        let s = edit_bar(&l, plate, true, BarEdit::Stretch(1), 63);
        assert_eq!(extent(&s, plate), (1018, 1143));
        let src = |l: &LayerItem, t: i64| (t - l.start) as f32 * l.speed;
        assert!((src(&s, 1018) - 1018.0).abs() < 0.6);
        let u = stretch_to(&l, plate, 0.5);
        assert_eq!(extent(&u, plate).0, 1018);
    }

    /// A range set upstream is the bar; the clip behind it is how far an
    /// end drags out.
    #[test]
    fn an_end_drags_out_to_the_clip_behind_a_range() {
        let s = Src::new((1018, 1080), (1001, 1100));
        let l = item(0, None);
        assert_eq!(extent(&l, s), (1018, 1080), "the bar is the range");
        assert_eq!(footage(&l, s), (1001, 1100), "the ghost is the clip");
        assert_eq!(edit_bar(&l, s, true, BarEdit::Trim(0), -50).trim, Some((1001, 1080)));
        assert_eq!(edit_bar(&l, s, true, BarEdit::Trim(1), 50).trim, Some((1018, 1100)));
        let slid = item(10, Some((1011, 1110)));
        assert_eq!(source_span(&slid, (1011, 1110)), (1001, 1100));
    }

    #[test]
    fn a_trim_wins_and_normalizes() {
        assert_eq!(extent(&item(0, Some((4, 9))), 24), (4, 9));
        // Dragged past itself, an end still describes a real window.
        assert_eq!(extent(&item(0, Some((9, 4))), 24), (4, 9));
    }

    /// AE's trim: an end moves the in or out point and NOTHING else — the
    /// footage stays where it was, the rate does not move, and the end
    /// stops at the footage's own edge.
    #[test]
    fn a_trim_moves_one_end_and_leaves_the_footage() {
        let l = item(10, None); // 24 frames: 10..33
        let t = edit_bar(&l, 24, true, BarEdit::Trim(0), 5);
        assert_eq!((t.start, t.speed, t.trim), (10, 1.0, Some((15, 33))));
        let t = edit_bar(&l, 24, true, BarEdit::Trim(1), -4);
        assert_eq!((t.start, t.trim), (10, Some((10, 29))));
        // Past the footage: stops at its edge.
        assert_eq!(edit_bar(&l, 24, true, BarEdit::Trim(1), 50).trim, Some((10, 33)));
        assert_eq!(edit_bar(&l, 24, true, BarEdit::Trim(0), -50).trim, Some((10, 33)));
        // An end never crosses the other.
        assert_eq!(edit_bar(&l, 24, true, BarEdit::Trim(0), 99).trim, Some((33, 33)));
        // Endless (a generator): no edge to stop at.
        assert_eq!(edit_bar(&l, 24, false, BarEdit::Trim(1), 50).trim, Some((10, 83)));
        // A drag back lands exactly where it began (absolute from the press).
        let t = edit_bar(&l, 24, true, BarEdit::Trim(0), 0);
        assert_eq!(t.trim, Some((10, 33)));
    }

    /// AE's stretch: the dragged end moves, the OTHER end holds, and what
    /// plays at the held end is the same frame as before.
    #[test]
    fn a_stretch_holds_the_other_end() {
        // Untrimmed: the bar is the footage, fitted exactly.
        let l = item(10, None);
        let s = edit_bar(&l, 24, true, BarEdit::Stretch(1), 24);
        assert_eq!(s.start, 10);
        assert!((s.speed - 0.5).abs() < 1e-6);
        assert_eq!(extent(&s, 24), (10, 57));
        let s = edit_bar(&l, 24, true, BarEdit::Stretch(0), -24);
        assert_eq!(extent(&s, 24), (-14, 33), "the out point holds");
        assert!(s.trim.is_none(), "a stretch is not a trim");

        // Trimmed 20..29 of footage at 10: the window AND the footage scale
        // about the held end, so the in point still shows source frame 10.
        let l = item(10, Some((20, 29)));
        let s = edit_bar(&l, 24, true, BarEdit::Stretch(1), 10);
        assert_eq!(s.trim, Some((20, 39)));
        assert!((s.speed - 0.5).abs() < 1e-6);
        let src = |l: &LayerItem, t: i64| (t - l.start) as f32 * l.speed;
        assert!((src(&s, 20) - src(&l, 20)).abs() < 1e-3);
        // Left end: the out point and its frame hold.
        let s = edit_bar(&l, 24, true, BarEdit::Stretch(0), -10);
        assert_eq!(s.trim, Some((10, 29)));
        assert!((src(&s, 30) - src(&l, 30)).abs() < 0.6);
    }

    #[test]
    fn a_slide_carries_the_window_and_a_slip_does_not() {
        let l = item(10, Some((20, 29)));
        let s = edit_bar(&l, 24, true, BarEdit::Slide, 5);
        assert_eq!((s.start, s.trim), (15, Some((25, 34))));
        let s = edit_bar(&l, 24, true, BarEdit::Slip, 5);
        assert_eq!((s.start, s.trim), (15, Some((20, 29))));
    }

    /// The Stretch % field: about the in point, window and all.
    #[test]
    fn stretch_percent_holds_the_in_point() {
        let l = item(10, Some((20, 29)));
        let s = stretch_to(&l, 24, 0.5);
        assert_eq!(s.trim, Some((20, 39)));
        assert_eq!(s.start, 0);
        let u = stretch_to(&item(10, None), 24, 2.0);
        assert_eq!(extent(&u, 24), (10, 21));
    }

    /// A three-layer stack, each layer wired to its own source and carrying
    /// a keyed translate that names it (layer i's curve holds the value i).
    fn rig() -> (EditorGraph, u64) {
        let mut g = EditorGraph::default();
        let srcs: Vec<u64> = (0..3)
            .map(|i| g.add(engine::graph::NodeKind::read(i, ""), egui::Pos2::ZERO))
            .collect();
        let id = g.add(
            engine::graph::NodeKind::Layer(engine::graph::LayerParams {
                outputs: false,
                layers: (0..3).map(|i| item(i as i64 * 10, None)).collect(),
            }),
            egui::Pos2::ZERO,
        );
        for (i, s) in srcs.iter().enumerate() {
            g.connect(id, i, *s);
        }
        if let Some(n) = g.node_mut(id) {
            for i in 0..3u16 {
                n.anim
                    .entry(crate::anim::Channel::LayerOpacity(i))
                    .or_default()
                    .set_key(0, i as f32);
            }
        }
        (g, id)
    }

    fn state(g: &EditorGraph, id: u64) -> Vec<(i64, Option<u64>, Option<f32>)> {
        let n = g.node(id).unwrap();
        let engine::graph::NodeKind::Layer(p) = &n.kind else { panic!("not a stack") };
        (0..p.layers.len())
            .map(|i| {
                (
                    p.layers[i].start,
                    g.input_of(id, i),
                    n.anim
                        .get(&crate::anim::Channel::LayerOpacity(i as u16))
                        .and_then(|c| c.eval(0)),
                )
            })
            .collect()
    }

    #[test]
    fn a_restack_carries_the_wire_and_the_curve() {
        let (mut g, id) = rig();
        let before = state(&g, id);
        swap(&mut g, id, 0, 2);
        let after = state(&g, id);
        assert_eq!(after[0], before[2], "layer 3 did not land on top whole");
        assert_eq!(after[2], before[0], "layer 1 did not land at the bottom whole");
        assert_eq!(after[1], before[1], "the untouched layer moved");
    }

    #[test]
    fn adding_a_layer_slides_everything_below_it() {
        let (mut g, id) = rig();
        let before = state(&g, id);
        restack(&mut g, id, None);
        let after = state(&g, id);
        assert_eq!(after.len(), 4);
        assert_eq!(after[0], (0, None, None), "the new top layer is empty");
        for i in 0..3 {
            assert_eq!(after[i + 1], before[i], "layer {i} lost something on the way down");
        }
    }

    #[test]
    fn deleting_a_layer_takes_only_its_own_wire_and_curve() {
        let (mut g, id) = rig();
        let before = state(&g, id);
        restack(&mut g, id, Some(1));
        let after = state(&g, id);
        assert_eq!(after.len(), 2);
        assert_eq!(after[0], before[0]);
        assert_eq!(after[1], before[2], "the layer below the deleted one lost its wire");
        // The deleted layer's edge is GONE, not left dangling on a port that
        // no longer exists.
        assert_eq!(g.edges.iter().filter(|e| e.0 == id).count(), 2);
    }

    #[test]
    fn the_last_layer_cannot_be_deleted() {
        let (mut g, id) = rig();
        restack(&mut g, id, Some(0));
        restack(&mut g, id, Some(0));
        restack(&mut g, id, Some(0));
        assert_eq!(state(&g, id).len(), 1, "a stack with no layers has no ports at all");
    }

    #[test]
    fn visibility_follows_the_trim() {
        assert!(item(0, None).visible_at(1000));
        let l = item(0, Some((4, 9)));
        assert!(!l.visible_at(3));
        assert!(l.visible_at(4));
        assert!(l.visible_at(9));
        assert!(!l.visible_at(10));
    }
}
#[cfg(test)]
mod pivot_tests {
    use super::*;

    /// A D-drag on the pivot re-centres rotate/scale WITHOUT moving the
    /// picture: the forward affine is the same before and after, and the
    /// pivot lands exactly where it was dragged to.
    #[test]
    fn moving_the_pivot_leaves_the_picture_where_it_was() {
        let size = [2048.0, 1080.0];
        let mut t = TransformParams {
            translate: [120.0, -40.0],
            rotate: 30.0,
            scale: [1.5, 0.8],
            center: [0.5, 0.5],
            scale_gang: false,
            ..Default::default()
        };
        let before = engine::codegen::forward_affine(&t, size);
        let q = egui::vec2(300.0, 700.0);
        move_pivot(&mut t, size, q);
        let after = engine::codegen::forward_affine(&t, size);
        for k in 0..6 {
            assert!(
                (after[k] - before[k]).abs() < 1e-2,
                "affine[{k}] moved {} -> {}",
                before[k],
                after[k]
            );
        }
        let pivot = egui::vec2(
            t.center[0] * size[0] + t.translate[0],
            t.center[1] * size[1] + t.translate[1],
        );
        assert!((pivot - q).length() < 1e-2, "pivot at {pivot:?}, wanted {q:?}");
        assert!((t.center[0] - 0.5).abs() > 0.01 || (t.center[1] - 0.5).abs() > 0.01);
        assert_eq!(t.rotate, 30.0);
        assert_eq!(t.scale, [1.5, 0.8]);
    }

    /// A wire on a layer's own output stays on THAT layer through a
    /// restack, goes with it when it is removed, and every such wire goes
    /// when "layer outputs" is switched off.
    #[test]
    fn layer_output_wires_follow_their_layer() {
        let mut g = EditorGraph::default();
        let stack = g.add(
            NodeKind::Layer(engine::graph::LayerParams {
                layers: vec![LayerItem::default(); 3],
                outputs: true,
            }),
            egui::Pos2::ZERO,
        );
        let a = g.add(NodeKind::Grade(Default::default()), egui::Pos2::ZERO);
        let b = g.add(NodeKind::Grade(Default::default()), egui::Pos2::ZERO);
        let c = g.add(NodeKind::Grade(Default::default()), egui::Pos2::ZERO);
        g.connect_out(a, 0, stack, 1); // layer 1
        g.connect_out(b, 0, stack, 3); // layer 3
        g.connect(c, 0, stack); // the comp
        let out_of = |g: &EditorGraph, d: u64| {
            g.edges.iter().any(|e| e.0 == d && e.2 == stack).then(|| g.edge_out.get(&(d, 0)).copied().unwrap_or(0))
        };
        swap(&mut g, stack, 0, 2);
        assert_eq!((out_of(&g, a), out_of(&g, b), out_of(&g, c)), (Some(3), Some(1), Some(0)));
        // Insert on top: everything moves down one.
        restack_at(&mut g, stack, Restack::Insert(0));
        assert_eq!((out_of(&g, a), out_of(&g, b)), (Some(4), Some(2)));
        // Remove a's layer: a's wire goes, b's stays on its layer.
        restack_at(&mut g, stack, Restack::Remove(3));
        assert_eq!((out_of(&g, a), out_of(&g, b), out_of(&g, c)), (None, Some(2), Some(0)));
        // Off: the per-layer wires go, the comp's stays.
        if let Some(NodeKind::Layer(p)) = g.node_mut(stack).map(|n| &mut n.kind) {
            p.outputs = false;
        }
        prune_outputs(&mut g, stack);
        assert_eq!((out_of(&g, b), out_of(&g, c)), (None, Some(0)));
    }
}
