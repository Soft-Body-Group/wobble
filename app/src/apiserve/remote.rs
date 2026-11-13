//! The properties panel drawn for ANOTHER screen: wobble_pad's iPad app,
//! through its Mac host. Not a copy of the panel — the panel itself: the
//! app runs `properties_ui` a second time per frame in an egui context of
//! its own, sized to the iPad, and sends what egui drew (the tessellated
//! meshes and the texture updates, LZ4'd). The iPad only rasterises them
//! and sends its touches back as pointer events. One app, one document,
//! so the two panels can never disagree.
//!
//! Like the scripting API it rides with: localhost only, and only while
//! the app runs with `--api` (port: the API's + 1, or WOBBLE_REMOTE_PORT).
//! One screen at a time; a new connection replaces the old and starts a
//! fresh context, so its first frame carries every texture.
//!
//! Which pane the screen gets is its choice (`Pane`): the properties, the
//! node graph (with its OWN pan, zoom and drags — a `NodeEditor` of its
//! own swapped in for the pass, so the Mac's graph view never moves), or
//! the viewer: wobble's own viewer pass (the HUDs, gizmos, the brush — a
//! Pencil paints with its pressure through `pen.rs`), with ITS OWN pan and
//! zoom swapped in the same way, over the picture rendered into the same
//! rect by a Blit of its own, read back without blocking and sent as JPEG.
//! That one is a per-frame readback, which is why it only runs while a
//! screen asks for it.
//!
//! Wire, both ways: `[u32 len][u8 type][payload]`, little-endian.
//! In: 1 hello (w, h points, pixels per point), 2 pointer (phase, x, y),
//! 3 scroll (dx, dy), 4 text (utf-8), 5 key (code, pressed), 6 pane (u8).
//! Out: 10 frame (u32 raw length, then the LZ4 block of `encode_frame`),
//! 11 picture (u32 w, u32 h, JPEG).

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::{Read, Write};
use std::net::{Ipv4Addr, TcpListener, TcpStream};
use std::sync::mpsc::{self, Receiver, Sender};

use egui::epaint::{ClippedPrimitive, ImageData, Primitive, TextureId};
use egui::{pos2, vec2, Event, Modifiers, PointerButton, Pos2, Rect, TexturesDelta};
use web_time::{Duration, Instant};

pub const FRAME: u8 = 10;
pub const PICTURE: u8 = 11;
/// The viewer's frame rate on the remote screen, at most.
const PICTURE_EVERY: Duration = Duration::from_millis(50);
/// The picture's long side, at most, in pixels.
const PICTURE_MAX: u32 = 2048;

/// What the remote screen shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pane {
    Properties,
    Nodes,
    Viewer,
}
/// A frame larger than this is a broken peer.
const MAX_IN: usize = 1 << 16;
/// How often the panel is redrawn with nothing new from the iPad: values
/// change on the Mac too (a drag there, playback keying through).
const IDLE: Duration = Duration::from_millis(100);

pub enum In {
    /// A new screen: its writer.
    Connected(Sender<Vec<u8>>),
    Hello { w: f32, h: f32, ppp: f32 },
    /// `pressure`: the Pencil's, or None for a finger.
    Pointer { phase: u8, x: f32, y: f32, pressure: Option<f32> },
    Scroll { dx: f32, dy: f32 },
    Text(String),
    Key { code: u8, pressed: bool },
    Pane(Pane),
    Gone,
}

pub struct Remote {
    rx: Receiver<In>,
    out: Option<Sender<Vec<u8>>>,
    pub ctx: egui::Context,
    screen: Option<(f32, f32, f32)>,
    events: Vec<Event>,
    pointer: Pos2,
    next_due: Instant,
    last_hash: u64,
    started: Instant,
    pub pane: Pane,
    /// The graph's view state on this screen (`Pane::Nodes`).
    pub editor: Option<crate::editor::NodeEditor>,
    pub picture: Picture,
    /// The Pencil's samples since the last pass, in this screen's physical
    /// pixels (what `pen.rs` holds for the window), and its pressure now.
    pub pens: Vec<crate::pen::Sample>,
    pub pressure: Option<f32>,
    /// The viewer's pan, zoom and rects on this screen.
    pub view: Option<super::RemoteView>,
}

impl Remote {
    /// Listen on 127.0.0.1:`port` (0 picks one). Logged and skipped if the
    /// port is taken: the app runs without it.
    pub fn start(port: u16) -> Option<Self> {
        let listener = match TcpListener::bind((Ipv4Addr::LOCALHOST, port)) {
            Ok(l) => l,
            Err(e) => {
                log::error!("remote properties: cannot listen on port {port}: {e}");
                return None;
            }
        };
        let port = listener.local_addr().ok()?.port();
        let (tx, rx) = mpsc::channel();
        std::thread::Builder::new()
            .name("remote-accept".into())
            .spawn(move || {
                for stream in listener.incoming().flatten() {
                    let tx = tx.clone();
                    let _ = std::thread::Builder::new().name("remote-conn".into()).spawn(move || connection(stream, tx));
                }
            })
            .ok()?;
        log::warn!("remote properties panel listening on 127.0.0.1:{port}");
        Some(Self {
            rx,
            out: None,
            ctx: fresh_ctx(),
            screen: None,
            events: Vec::new(),
            pointer: Pos2::ZERO,
            next_due: Instant::now(),
            last_hash: 0,
            started: Instant::now(),
            pane: Pane::Properties,
            editor: None,
            picture: Picture::default(),
            pens: Vec::new(),
            pressure: None,
            view: None,
        })
    }

    /// Take what the screen sent; the input for a panel pass if one is due.
    pub fn poll(&mut self) -> Option<egui::RawInput> {
        while let Ok(m) = self.rx.try_recv() {
            match m {
                In::Connected(w) => {
                    self.out = Some(w);
                    self.ctx = fresh_ctx();
                    self.screen = None;
                    self.events.clear();
                    self.last_hash = 0;
                }
                In::Gone => {
                    self.out = None;
                    self.screen = None;
                }
                In::Pane(p) => {
                    self.pane = p;
                    self.next_due = Instant::now();
                    self.last_hash = 0;
                    self.picture.last_hash = 0;
                }
                In::Hello { w, h, ppp } => {
                    self.screen = Some((w, h, ppp));
                    self.next_due = Instant::now();
                    self.last_hash = 0;
                }
                other => {
                    if let (In::Pointer { phase, x, y, pressure }, Some((_, _, ppp))) = (&other, self.screen) {
                        // The brush reads the pen through pen.rs, every
                        // sample with its pressure; a lift is no pen.
                        self.pressure = if *phase >= 2 { None } else { *pressure };
                        if pressure.is_some() && *phase < 2 {
                            self.pens.push(crate::pen::Sample { phys: [x * ppp, y * ppp], pressure: *pressure });
                        }
                    }
                    self.events.extend(to_events(other, &mut self.pointer))
                }
            }
        }
        let (w, h, ppp) = self.screen?;
        self.out.as_ref()?;
        if self.events.is_empty() && Instant::now() < self.next_due {
            return None;
        }
        let mut raw = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(w, h))),
            max_texture_side: Some(4096),
            time: Some(self.started.elapsed().as_secs_f64()),
            events: std::mem::take(&mut self.events),
            focused: true,
            ..Default::default()
        };
        raw.viewports.entry(egui::ViewportId::ROOT).or_default().native_pixels_per_point = Some(ppp);
        Some(raw)
    }

    /// Send what a pass drew (only if it differs from the last frame, or
    /// carries texture updates), and note when the next pass is due.
    pub fn finish(&mut self, mut out: egui::FullOutput, ink: Option<Ink>) {
        let delay = out.viewport_output.get(&egui::ViewportId::ROOT).map(|v| v.repaint_delay).unwrap_or(IDLE);
        self.next_due = Instant::now() + delay.min(IDLE);
        let ppp = out.pixels_per_point;
        let wants_keyboard = out.platform_output.ime.is_some();
        let prims = self.ctx.tessellate(std::mem::take(&mut out.shapes), ppp);
        let raw = encode_frame(ppp, wants_keyboard, &out.textures_delta, &prims, ink.as_ref());
        let mut h = DefaultHasher::new();
        raw.hash(&mut h);
        let hash = h.finish();
        if hash == self.last_hash && out.textures_delta.is_empty() {
            return;
        }
        self.last_hash = hash;
        let packed = lz4_flex::block::compress(&raw);
        let mut body = Vec::with_capacity(packed.len() + 9);
        body.extend_from_slice(&((packed.len() + 5) as u32).to_le_bytes());
        body.push(FRAME);
        body.extend_from_slice(&(raw.len() as u32).to_le_bytes());
        body.extend_from_slice(&packed);
        if let Some(o) = &self.out {
            if o.send(body).is_err() {
                self.out = None;
            }
        }
    }
}

impl Remote {
    /// The screen's size in pixels, while one is connected.
    pub fn screen_px(&self) -> Option<(u32, u32)> {
        self.out.as_ref()?;
        let (w, h, ppp) = self.screen?;
        Some(((w * ppp) as u32, (h * ppp) as u32))
    }

    /// The screen's size in points.
    pub fn screen_points(&self) -> Option<(f32, f32)> {
        self.screen.map(|(w, h, _)| (w, h))
    }

    pub fn sender(&self) -> Option<Sender<Vec<u8>>> {
        self.out.clone()
    }
}

/// The viewer's picture for the remote screen: a Blit of its own (the
/// viewer's is compiled for the swapchain), a target at the screen's size,
/// and ONE readback in flight, polled rather than waited on.
#[derive(Default)]
pub struct Picture {
    pub blit: Option<engine::blit::Blit>,
    pub target: Option<(wgpu::Texture, wgpu::Buffer, u32, u32, u32)>,
    /// A copy submitted and mapping: set true once mapped.
    pub mapping: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    pub encoding: std::sync::Arc<std::sync::atomic::AtomicBool>,
    pub last_start: Option<Instant>,
    pub last_hash: u64,
}

impl Picture {
    /// A new readback may start.
    pub fn due(&self) -> bool {
        self.mapping.is_none()
            && !self.encoding.load(std::sync::atomic::Ordering::Acquire)
            && self.last_start.is_none_or(|t| t.elapsed() >= PICTURE_EVERY)
    }

    /// The picture's target for a `sw`x`sh` pixel screen: the screen,
    /// scaled down to fit `PICTURE_MAX`, and that scale. The viewer's image
    /// rect is drawn into it as the Mac's viewer draws it into its surface,
    /// so pan, zoom and the NEAREST-past-1:1 rule all carry over.
    pub fn target(sw: u32, sh: u32) -> (u32, u32, f32) {
        let s = (PICTURE_MAX as f32 / sw.max(sh).max(1) as f32).min(1.0);
        (((sw as f32 * s) as u32).max(1), ((sh as f32 * s) as u32).max(1), s)
    }

    /// The mapped readback, if it has landed: hand its pixels to a thread
    /// that JPEGs and sends them (skipped when nothing changed).
    pub fn collect(&mut self, out: Option<Sender<Vec<u8>>>) {
        use std::sync::atomic::Ordering;
        let Some(flag) = &self.mapping else { return };
        if !flag.load(Ordering::Acquire) {
            return;
        }
        self.mapping = None;
        let Some((_, buf, w, h, bpr)) = &self.target else { return };
        let (w, h, bpr) = (*w, *h, *bpr as usize);
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        {
            let view = buf.slice(..).get_mapped_range().expect("a mapped buffer");
            for y in 0..h as usize {
                rgba[y * w as usize * 4..(y + 1) * w as usize * 4]
                    .copy_from_slice(&view[y * bpr..y * bpr + w as usize * 4]);
            }
        }
        buf.unmap();
        let mut hasher = DefaultHasher::new();
        rgba.hash(&mut hasher);
        let hash = hasher.finish();
        let Some(out) = out else { return };
        if hash == self.last_hash {
            return;
        }
        self.last_hash = hash;
        let busy = self.encoding.clone();
        busy.store(true, Ordering::Release);
        crate::bg::spawn(move || {
            let mut jpeg = Vec::new();
            let enc = jpeg_encoder::Encoder::new(&mut jpeg, 82);
            if enc.encode(&rgba, w as u16, h as u16, jpeg_encoder::ColorType::Rgba).is_ok() {
                let mut body = Vec::with_capacity(jpeg.len() + 13);
                body.extend_from_slice(&((jpeg.len() + 9) as u32).to_le_bytes());
                body.push(PICTURE);
                body.extend_from_slice(&w.to_le_bytes());
                body.extend_from_slice(&h.to_le_bytes());
                body.extend_from_slice(&jpeg);
                let _ = out.send(body);
            }
            busy.store(false, Ordering::Release);
        });
    }
}

fn fresh_ctx() -> egui::Context {
    let ctx = egui::Context::default();
    ctx.set_theme(egui::Theme::Dark);
    ctx.all_styles_mut(|s| s.animation_time /= 3.0);
    ctx
}

/// The viewer's HUD marks, which ink.rs draws with a wgpu callback that
/// picks black or white per pixel from the picture under them. Sent in the
/// callback's place (texture kind 2) with the palette and frost, so the
/// screen runs the same pick over its own copy of the picture.
pub struct Ink {
    pub mesh: egui::epaint::Mesh,
    pub palette: [[f32; 4]; 16],
    pub darken: f32,
}

/// One screen's input to egui events. A touch is a press and a release
/// with the pointer gone after, as egui-winit makes of a finger.
fn to_events(m: In, pointer: &mut Pos2) -> Vec<Event> {
    let none = Modifiers::default();
    match m {
        In::Pointer { phase, x, y, .. } => {
            let pos = pos2(x, y);
            *pointer = pos;
            let button = |pressed| Event::PointerButton { pos, button: PointerButton::Primary, pressed, modifiers: none };
            match phase {
                0 => vec![Event::PointerMoved(pos), button(true)],
                1 => vec![Event::PointerMoved(pos)],
                2 => vec![Event::PointerMoved(pos), button(false), Event::PointerGone],
                _ => vec![Event::PointerGone],
            }
        }
        In::Scroll { dx, dy } => vec![Event::MouseWheel {
            unit: egui::MouseWheelUnit::Point,
            delta: vec2(dx, dy),
            phase: egui::TouchPhase::Move,
            modifiers: none,
        }],
        In::Text(t) => vec![Event::Text(t)],
        In::Key { code, pressed } => {
            let key = match code {
                0 => egui::Key::Backspace,
                1 => egui::Key::Enter,
                2 => egui::Key::Escape,
                _ => egui::Key::Tab,
            };
            vec![Event::Key { key, physical_key: None, pressed, repeat: false, modifiers: none }]
        }
        In::Connected(_) | In::Hello { .. } | In::Pane(_) | In::Gone => Vec::new(),
    }
}

/// What a panel pass drew, as the iPad reads it (little-endian):
/// `f32 ppp, u8 wants_keyboard`,
/// `u32 n` texture sets: `u8 kind, u64 id, u8 has_pos, u32 x, u32 y, u32 w, u32 h, u8 linear, w*h*4 RGBA`,
/// `u32 n` frees: `u8 kind, u64 id`,
/// `u32 n` meshes: `f32 clip[4], u8 kind, u64 id, u32 nv, nv * (f32 x, y, u, v, u8 rgba[4]), u32 ni, ni * u32`.
/// Positions are points; colours premultiplied, gamma-encoded (egui's own).
/// Paint callbacks have no meaning off this machine and are left out.
pub fn encode_frame(
    ppp: f32,
    wants_keyboard: bool,
    delta: &TexturesDelta,
    prims: &[ClippedPrimitive],
    ink: Option<&Ink>,
) -> Vec<u8> {
    let mut b = Vec::with_capacity(64 * 1024);
    b.extend_from_slice(&ppp.to_le_bytes());
    b.push(wants_keyboard as u8);
    let tex = |b: &mut Vec<u8>, id: &TextureId| match id {
        TextureId::Managed(n) => {
            b.push(0);
            b.extend_from_slice(&n.to_le_bytes());
        }
        TextureId::User(n) => {
            b.push(1);
            b.extend_from_slice(&n.to_le_bytes());
        }
    };
    let sets: Vec<_> = delta.set.iter().flat_map(|(id, ds)| ds.iter().map(move |d| (id, d))).collect();
    b.extend_from_slice(&(sets.len() as u32).to_le_bytes());
    for (id, d) in sets {
        tex(&mut b, id);
        let [x, y] = d.pos.unwrap_or([0, 0]);
        b.push(d.pos.is_some() as u8);
        b.extend_from_slice(&(x as u32).to_le_bytes());
        b.extend_from_slice(&(y as u32).to_le_bytes());
        let ImageData::Color(img) = &d.image;
        b.extend_from_slice(&(img.size[0] as u32).to_le_bytes());
        b.extend_from_slice(&(img.size[1] as u32).to_le_bytes());
        b.push((d.options.magnification == egui::TextureFilter::Linear) as u8);
        for c in &img.pixels {
            b.extend_from_slice(&c.to_array());
        }
    }
    b.extend_from_slice(&(delta.free.len() as u32).to_le_bytes());
    for id in &delta.free {
        tex(&mut b, id);
    }
    // A paint callback is the viewer's ink (the only one a remote pass
    // makes): its mesh goes in its place, flagged texture kind 2.
    let mut ink_left = ink;
    let meshes: Vec<_> = prims
        .iter()
        .filter_map(|p| match &p.primitive {
            Primitive::Mesh(m) if !m.indices.is_empty() => Some((p.clip_rect, m, false)),
            Primitive::Callback(_) => ink_left.take().filter(|i| !i.mesh.indices.is_empty()).map(|i| (p.clip_rect, &i.mesh, true)),
            _ => None,
        })
        .collect();
    b.extend_from_slice(&(meshes.len() as u32).to_le_bytes());
    for (clip, m, is_ink) in meshes {
        for v in [clip.min.x, clip.min.y, clip.max.x, clip.max.y] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        if is_ink {
            b.push(2);
            b.extend_from_slice(&0u64.to_le_bytes());
        } else {
            tex(&mut b, &m.texture_id);
        }
        b.extend_from_slice(&(m.vertices.len() as u32).to_le_bytes());
        for v in &m.vertices {
            for f in [v.pos.x, v.pos.y, v.uv.x, v.uv.y] {
                b.extend_from_slice(&f.to_le_bytes());
            }
            b.extend_from_slice(&v.color.to_array());
        }
        b.extend_from_slice(&(m.indices.len() as u32).to_le_bytes());
        for i in &m.indices {
            b.extend_from_slice(&i.to_le_bytes());
        }
    }
    // The ink's palette (16 rgba, display encoded) and frost darkening,
    // trailing: a reader that predates them stops at the meshes.
    if let Some(i) = ink {
        for c in &i.palette {
            for v in c {
                b.extend_from_slice(&v.to_le_bytes());
            }
        }
        b.extend_from_slice(&i.darken.to_le_bytes());
    }
    b
}

/// One screen: read its frames into `In`s; write what the app sends it.
fn connection(stream: TcpStream, tx: Sender<In>) {
    let _ = stream.set_nodelay(true);
    let Ok(mut write) = stream.try_clone() else { return };
    let (otx, orx) = mpsc::channel::<Vec<u8>>();
    if tx.send(In::Connected(otx)).is_err() {
        return;
    }
    std::thread::spawn(move || {
        for buf in orx {
            if write.write_all(&buf).is_err() {
                break;
            }
        }
        let _ = write.shutdown(std::net::Shutdown::Both);
    });
    let mut read = stream;
    loop {
        let mut len = [0u8; 4];
        if read.read_exact(&mut len).is_err() {
            break;
        }
        let len = u32::from_le_bytes(len) as usize;
        if len == 0 || len > MAX_IN {
            break;
        }
        let mut body = vec![0u8; len];
        if read.read_exact(&mut body).is_err() {
            break;
        }
        let Some(m) = decode_in(&body) else { break };
        if tx.send(m).is_err() {
            return;
        }
    }
    let _ = tx.send(In::Gone);
}

pub fn decode_in(body: &[u8]) -> Option<In> {
    let f = |i: usize| body.get(i..i + 4).map(|s| f32::from_le_bytes(s.try_into().unwrap()));
    match *body.first()? {
        1 => Some(In::Hello { w: f(1)?, h: f(5)?, ppp: f(9)? }),
        // An older pad sends no pressure; a negative one is "not a pen".
        2 => Some(In::Pointer {
            phase: *body.get(1)?,
            x: f(2)?,
            y: f(6)?,
            pressure: f(10).filter(|p| *p >= 0.0),
        }),
        3 => Some(In::Scroll { dx: f(1)?, dy: f(5)? }),
        4 => Some(In::Text(String::from_utf8(body[1..].to_vec()).ok()?)),
        5 => Some(In::Key { code: *body.get(1)?, pressed: *body.get(2)? != 0 }),
        6 => Some(In::Pane(match *body.get(1)? {
            0 => Pane::Properties,
            1 => Pane::Nodes,
            _ => Pane::Viewer,
        })),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_touch_is_a_press_a_release_and_the_pointer_gone() {
        let mut p = Pos2::ZERO;
        let down = to_events(In::Pointer { phase: 0, x: 10.0, y: 20.0, pressure: None }, &mut p);
        assert!(matches!(down[1], Event::PointerButton { pressed: true, .. }));
        assert_eq!(p, pos2(10.0, 20.0));
        let up = to_events(In::Pointer { phase: 2, x: 12.0, y: 20.0, pressure: None }, &mut p);
        assert!(matches!(up[1], Event::PointerButton { pressed: false, .. }));
        assert!(matches!(up[2], Event::PointerGone));
    }

    #[test]
    fn the_picture_is_the_screen_capped() {
        assert_eq!(Picture::target(1600, 1000), (1600, 1000, 1.0));
        let (w, h, s) = Picture::target(4096, 2048);
        assert_eq!((w, h), (2048, 1024), "capped at PICTURE_MAX");
        assert_eq!(s, 0.5);
        let mut p = Pos2::ZERO;
        let mut hello = vec![2u8, 0];
        for v in [10.0f32, 20.0, 0.4] {
            hello.extend_from_slice(&v.to_le_bytes());
        }
        assert!(matches!(decode_in(&hello), Some(In::Pointer { pressure: Some(x), .. }) if (x - 0.4).abs() < 1e-6));
        assert!(matches!(decode_in(&hello[..10]), Some(In::Pointer { pressure: None, .. })), "an older pad");
        let _ = to_events(In::Pointer { phase: 1, x: 0.0, y: 0.0, pressure: None }, &mut p);
        assert!(matches!(decode_in(&[6, 2]), Some(In::Pane(Pane::Viewer))));
        assert!(matches!(decode_in(&[6, 1]), Some(In::Pane(Pane::Nodes))));
    }

    /// The viewer's ink rides in its callback's place, flagged kind 2, and
    /// its palette and frost trail the meshes.
    #[test]
    fn ink_goes_where_its_callback_was() {
        let mut mesh = egui::epaint::Mesh::default();
        mesh.colored_vertex(pos2(0.0, 0.0), egui::Color32::BLACK);
        mesh.colored_vertex(pos2(4.0, 0.0), egui::Color32::BLACK);
        mesh.colored_vertex(pos2(0.0, 4.0), egui::Color32::BLACK);
        mesh.add_triangle(0, 1, 2);
        let clip = Rect::from_min_size(Pos2::ZERO, vec2(100.0, 100.0));
        let prims = vec![ClippedPrimitive {
            clip_rect: clip,
            primitive: Primitive::Callback(egui::PaintCallback { rect: clip, callback: std::sync::Arc::new(()) }),
        }];
        let ink = Ink { mesh, palette: [[0.5; 4]; 16], darken: 0.25 };
        let b = encode_frame(2.0, false, &TexturesDelta::default(), &prims, Some(&ink));
        // ppp, kb, 0 sets, 0 frees, 1 mesh: clip, kind 2, id, 3 verts, 3 indices; palette; darken.
        let at = 4 + 1 + 4 + 4;
        assert_eq!(u32::from_le_bytes(b[at..at + 4].try_into().unwrap()), 1);
        assert_eq!(b[at + 4 + 16], 2, "kind 2: ink");
        assert_eq!(b.len(), at + 4 + 16 + 9 + 4 + 3 * 20 + 4 + 3 * 4 + 16 * 16 + 4);
        assert_eq!(f32::from_le_bytes(b[b.len() - 4..].try_into().unwrap()), 0.25);
        // Without the ink a callback is left out.
        let b = encode_frame(2.0, false, &TexturesDelta::default(), &prims, None);
        assert_eq!(u32::from_le_bytes(b[at..at + 4].try_into().unwrap()), 0);
    }

    #[test]
    fn input_frames_decode_and_garbage_does_not() {
        let mut hello = vec![1u8];
        for v in [1180.0f32, 820.0, 2.0] {
            hello.extend_from_slice(&v.to_le_bytes());
        }
        assert!(matches!(decode_in(&hello), Some(In::Hello { w, h, ppp }) if w == 1180.0 && h == 820.0 && ppp == 2.0));
        assert!(decode_in(&[2, 0, 1]).is_none(), "a short pointer frame");
        assert!(decode_in(&[99]).is_none());
    }

    /// A real panel pass encodes to what the iPad expects: the font atlas
    /// goes out on the first frame, and the meshes' byte count adds up.
    #[test]
    fn a_frame_carries_the_atlas_then_meshes() {
        let ctx = fresh_ctx();
        let mut raw = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(400.0, 300.0))),
            ..Default::default()
        };
        raw.viewports.entry(egui::ViewportId::ROOT).or_default().native_pixels_per_point = Some(2.0);
        let mut out = ctx.run_ui(raw, |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                ui.label("blur");
                let mut v = 4.0f32;
                ui.add(egui::Slider::new(&mut v, 0.0..=10.0));
            });
        });
        let prims = ctx.tessellate(std::mem::take(&mut out.shapes), out.pixels_per_point);
        let b = encode_frame(out.pixels_per_point, false, &out.textures_delta, &prims, None);
        assert_eq!(f32::from_le_bytes(b[0..4].try_into().unwrap()), 2.0);
        let sets = u32::from_le_bytes(b[5..9].try_into().unwrap());
        assert!(sets >= 1, "the font atlas");
        let meshes: usize = prims.iter().filter(|p| matches!(&p.primitive, Primitive::Mesh(m) if !m.indices.is_empty())).count();
        assert!(meshes >= 1);
        let tail: usize = prims
            .iter()
            .filter_map(|p| match &p.primitive {
                Primitive::Mesh(m) if !m.indices.is_empty() => Some(16 + 9 + 4 + m.vertices.len() * 20 + 4 + m.indices.len() * 4),
                _ => None,
            })
            .sum();
        let n_free = 0;
        assert!(b.len() > tail + 4 + 4 + n_free, "texture bytes + meshes");
        assert_eq!(u32::from_le_bytes(b[b.len() - tail - 4..b.len() - tail].try_into().unwrap()) as usize, meshes);
    }
}
