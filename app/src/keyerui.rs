//! Colour-keyer UI: the two colour-space plots, the blob list, and the
//! viewer-sampling machinery behind them.
//!
//! Owned by instance E (see .agents/board.md). Pure UI + readback over
//! `engine::keyer` — the engine knows nothing about any of this.
//!
//! The plots are the point of the node. Flame shows its 3D Keyer's selection
//! as ellipsoids floating in an RGB cube, which is honest but hard to aim in;
//! this shows the same ellipsoid as its two meaningful PROJECTIONS, each one
//! a plane an artist already thinks in:
//!
//! - **hue/chroma** (the a/b plane at the blob's lightness) — where the
//!   colour is and how wide the hue and saturation grabs are;
//! - **lightness/chroma** (a slice through the blob's own hue) — how far up
//!   and down the exposure range the selection reaches, and how it leans.
//!
//! Both are painted with the actual colours they represent, both carry a
//! scatter of the image's own pixels (so you can see where the subject lives
//! before you aim at it), and both shade the LIVE matte — union, subtractive
//! blobs, levels and all — as a translucent overlay. What you see selected in
//! the plot is what the shader selects.

use crate::anim::Channel;
use crate::theme;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{epaint, Color32, Pos2, Rect, Sense, Stroke, Vec2};
use engine::keyer::{self, KeyBlob, KeyerParams, KeyerView, MAX_BLOBS};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Chroma at the rim of the hue/chroma disc, in Oklab units. Wide enough to
/// hold a saturated primary without pushing everyday subjects into the middle.
const CMAX: f32 = 0.28;
/// Top of the lightness axis. Oklab L is 1.0 at linear 1.0, so this leaves
/// room to aim at highlights without the plot becoming mostly empty.
const LMAX: f32 = 1.25;
/// Plot square side at zoom 1 (geometry scales with the grid, type does not).
const PLOT: f32 = 196.0;

// ---------------------------------------------------------------------------
// colour helpers
// ---------------------------------------------------------------------------

fn srgb_encode(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    if x <= 0.003_130_8 {
        x * 12.92
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

fn in_gamut(c: [f32; 3]) -> bool {
    c.iter().all(|v| *v >= -1e-4 && *v <= 1.0 + 1e-4)
}

/// Oklch -> linear rgb, gamut-mapped by pulling chroma in rather than
/// clipping channels. Also reports how far in it had to pull: clipping turns
/// everything past the boundary into the same flat primary, and the outer
/// half of a disc drawn that way is a neon lie.
fn oklch_mapped(l: f32, c: f32, h: f32) -> ([f32; 3], f32) {
    let lab = |c: f32| [l, c * h.cos(), c * h.sin()];
    if c < 1e-5 || in_gamut(keyer::linear_from_oklab(lab(c))) {
        return (keyer::linear_from_oklab(lab(c)), 1.0);
    }
    let (mut lo, mut hi) = (0.0f32, c);
    for _ in 0..12 {
        let mid = 0.5 * (lo + hi);
        if in_gamut(keyer::linear_from_oklab(lab(mid))) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (keyer::linear_from_oklab(lab(lo)), lo / c)
}

fn pack(rgb: [f32; 3], dim: f32) -> Color32 {
    Color32::from_rgb(
        (srgb_encode(rgb[0] * dim) * 255.0) as u8,
        (srgb_encode(rgb[1] * dim) * 255.0) as u8,
        (srgb_encode(rgb[2] * dim) * 255.0) as u8,
    )
}

/// A plot background colour. Anything the display cannot actually show is
/// DARKENED in proportion to how far out it is, so the gamut boundary reads
/// as a shape on the plot instead of a flat wall of primaries — and the
/// scene-linear colours that live beyond it stay honestly out of reach.
pub(crate) fn oklch_color(l: f32, c: f32, h: f32) -> Color32 {
    let (rgb, reach) = oklch_mapped(l, c, h);
    pack(rgb, 0.45 + 0.55 * reach)
}

/// The swatch for a blob: its own centre colour, undimmed — a swatch is an
/// identity, not a measurement.
pub fn blob_color(b: &KeyBlob) -> Color32 {
    let (c, h) = b.chroma_hue();
    pack(oklch_mapped(b.center[0].clamp(0.05, 1.0), c, h).0, 1.0)
}

// ---------------------------------------------------------------------------
// app-side state
// ---------------------------------------------------------------------------

/// One async readback of the comp output, in flight.
struct PixelRead {
    node: u64,
    buffer: wgpu::Buffer,
    w: u32,
    h: u32,
    /// Row stride in bytes (256-aligned for the copy).
    stride: u32,
    ready: Arc<AtomicBool>,
    mapped: bool,
    cloud: bool,
}

/// A live sampling gesture: the pixels dragged over so far, and which blob
/// they are refitting.
pub struct SampleDrag {
    pub node: u64,
    pub subtract: bool,
    pub blob: usize,
    pub samples: Vec<[f32; 3]>,
    /// Set on mouse-up; the drag survives one more poll so a click (whose
    /// only readback lands after release) still produces a blob.
    pub ended: bool,
}

/// Everything the keyer needs from the app shell, in one field so `lib.rs`
/// carries one member and four call sites.
#[derive(Default)]
pub struct KeyerState {
    /// Selected blob per node — pure UI, deliberately NOT in the params:
    /// it would otherwise churn undo and the render-cache key.
    pub sel: HashMap<u64, usize>,
    /// Scatter cloud per node: linear-RGB samples of that node's input.
    pub cloud: HashMap<u64, Vec<[f32; 3]>>,
    /// A node whose cloud should be refreshed on the next encode.
    pub want_cloud: Option<u64>,
    pub drag: Option<SampleDrag>,
    /// Cursor in OUTPUT pixels, refreshed by the viewer while a sample drag
    /// is live — the encode step reads the square under it.
    pub cursor: Option<(i32, i32)>,
    /// A node the panel asked to become the view target (sampling needs it).
    pub want_view: Option<u64>,
    /// The comp's working space, refreshed from the graph each frame.
    ///
    /// Blob centres MUST be fitted on the same Oklab frame the shader keys
    /// on. Fit in Rec.709 while the shader reads AP1 and the selection lands
    /// somewhere other than where the artist sampled — with nothing on
    /// screen to say so.
    pub working: engine::color::Colorspace,
    reads: Vec<PixelRead>,
}

/// How many pixels a cloud keeps. Enough to read as a distribution, small
/// enough that the plots stay a paint job rather than a particle system.
const CLOUD_MAX: usize = 6000;
/// Side of the square read back around the cursor per sampling frame.
const SAMPLE_PX: u32 = 17;

impl KeyerState {
    pub fn begin_drag(&mut self, node: u64, subtract: bool, blob: usize) {
        self.drag = Some(SampleDrag {
            node,
            subtract,
            blob,
            samples: Vec::new(),
            ended: false,
        });
    }

    /// Mouse-up. In-flight reads are dropped for a real drag (hundreds of
    /// samples are already in, and letting a straggler land after release
    /// would cost a second undo step for nothing); a bare click keeps
    /// waiting, because its one readback IS the sample.
    pub fn end_drag(&mut self) {
        if let Some(d) = &mut self.drag {
            if d.samples.is_empty() {
                d.ended = true;
            } else {
                let node = d.node;
                self.reads.retain(|r| r.cloud || r.node != node);
                self.drag = None;
            }
        }
    }

    pub fn dragging(&self) -> bool {
        self.drag.is_some()
    }

    /// Queue readbacks: the square under the cursor while a sample drag is
    /// live, and a full-frame subsample when a cloud has been asked for.
    ///
    /// SLOW PATH, deliberately visible: the cloud is a whole-frame
    /// texture->buffer copy. It runs only when the user asks for one (or
    /// enters the node), never per frame.
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        tex: &wgpu::Texture,
        out: (u32, u32),
    ) {
        let (ow, oh) = out;
        let mut jobs: Vec<(u64, u32, u32, u32, u32, bool)> = Vec::new();
        if let Some(d) = &self.drag {
            // A bare click ends the gesture before its one and only read has
            // been queued, so an ended drag with nothing sampled still gets
            // one; the in-flight check below keeps it to exactly one.
            if !d.ended || d.samples.is_empty() {
                if let Some((cx, cy)) = self.cursor {
                    let s = SAMPLE_PX.min(ow).min(oh);
                    let x = (cx - s as i32 / 2).clamp(0, (ow - s) as i32) as u32;
                    let y = (cy - s as i32 / 2).clamp(0, (oh - s) as i32) as u32;
                    jobs.push((d.node, x, y, s, s, false));
                }
            }
        }
        if let Some(node) = self.want_cloud.take() {
            jobs.push((node, 0, 0, ow, oh, true));
        }
        for (node, x, y, w, h, cloud) in jobs {
            // One sample read in flight per node is plenty: they land in a
            // frame or two and the drag accumulates over hundreds.
            if !cloud && self.reads.iter().any(|r| r.node == node && !r.cloud) {
                continue;
            }
            let stride = (w * 8).div_ceil(256) * 256;
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("keyer-pick"),
                size: (stride * h) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x, y, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(stride),
                        rows_per_image: Some(h),
                    },
                },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
            self.reads.push(PixelRead {
                node,
                buffer,
                w,
                h,
                stride,
                ready: Arc::new(AtomicBool::new(false)),
                mapped: false,
                cloud,
            });
        }
    }

    /// Map the buffers AFTER the encoder is submitted — mapping one that is
    /// still the destination of an unsubmitted copy is a validation error,
    /// and the read silently never lands.
    pub fn after_submit(&mut self) {
        for r in &mut self.reads {
            if !r.mapped {
                r.mapped = true;
                let flag = r.ready.clone();
                r.buffer.slice(..).map_async(wgpu::MapMode::Read, move |res| {
                    if res.is_ok() {
                        flag.store(true, Ordering::Release);
                    }
                });
            }
        }
    }

    /// Drain finished readbacks. Cloud reads fill the scatter; sample reads
    /// extend the live drag and return a refitted blob per node, which the
    /// shell writes into the graph (so the matte moves under the cursor as
    /// you drag, which is the whole feel of the tool).
    pub fn poll(&mut self) -> Vec<(u64, usize, KeyBlob)> {
        let mut fits = Vec::new();
        let mut done = Vec::new();
        for (i, r) in self.reads.iter().enumerate() {
            if !r.mapped || !r.ready.load(Ordering::Acquire) {
                continue;
            }
            done.push(i);
            let px = {
                let view = r.buffer.slice(..).get_mapped_range().expect("a mapped buffer");
                let step = if r.cloud {
                    // Subsample a full frame down to a readable cloud.
                    (((r.w as usize * r.h as usize) / CLOUD_MAX).max(1) as f32)
                        .sqrt()
                        .ceil() as usize
                } else {
                    1
                };
                let mut px: Vec<[f32; 3]> = Vec::new();
                for row in (0..r.h as usize).step_by(step) {
                    let base = row * r.stride as usize;
                    let line: &[half::f16] =
                        bytemuck::cast_slice(&view[base..base + (r.w * 8) as usize]);
                    for c in line.chunks_exact(4).step_by(step) {
                        px.push([c[0].to_f32(), c[1].to_f32(), c[2].to_f32()]);
                    }
                }
                px
            };
            r.buffer.unmap();
            if r.cloud {
                self.cloud.insert(r.node, px);
                continue;
            }
            let Some(d) = &mut self.drag else { continue };
            if d.node != r.node {
                continue;
            }
            d.samples.extend(px);
            // Cap the accumulation: a long drag would otherwise refit over a
            // growing vector every frame for no extra accuracy.
            if d.samples.len() > CLOUD_MAX {
                let cut = d.samples.len() - CLOUD_MAX;
                d.samples.drain(0..cut);
            }
            let mut blob = keyer::fit_in(&d.samples, 2.0, keyer::oklab_matrix(self.working));
            blob.weight = if d.subtract { -1.0 } else { 1.0 };
            fits.push((d.node, d.blob, blob));
            if d.ended {
                self.drag = None;
            }
        }
        for i in done.into_iter().rev() {
            self.reads.remove(i);
        }
        fits
    }
}

// ---------------------------------------------------------------------------
// the panel
// ---------------------------------------------------------------------------

/// What the panel wants the shell to do after drawing.
#[derive(Default)]
pub struct KeyerOut {
    pub hud: Option<HudSlider>,
    /// Refresh this node's scatter cloud.
    pub want_cloud: bool,
    /// Make this node the view target (sampling needs it).
    pub want_view: bool,
}

/// Draw the keyer panel. `viewing` is whether this node is the current view
/// target — sampling and the cloud are only meaningful then, and the panel
/// says so rather than quietly reading the wrong pixels.
pub fn keyer_ui(
    ui: &mut egui::Ui,
    k: &mut KeyerParams,
    cx: &mut RowCtx,
    sel: &mut usize,
    cloud: Option<&Vec<[f32; 3]>>,
    viewing: bool,
    zoom: f32,
    working: engine::color::Colorspace,
) -> KeyerOut {
    let mut out = KeyerOut::default();
    let lms = keyer::oklab_matrix(working);
    if k.blobs.is_empty() {
        k.blobs.push(KeyBlob::default());
    }
    *sel = (*sel).min(k.blobs.len() - 1);
    let side = PLOT * zoom;
    ui.horizontal_top(|ui| {
        hue_chroma_plot(ui, k, *sel, cloud, side, cx.node, lms);
        ui.add_space(6.0);
        light_chroma_plot(ui, k, *sel, cloud, side, cx.node, lms);
        ui.add_space(6.0);
        ui.vertical(|ui| {
            out = controls(ui, k, cx, sel, viewing);
        });
    });
    // The two plots above are PROJECTIONS of one volume; the volume itself
    // goes underneath, full width, so the thing they are projections OF is
    // on screen rather than assembled in your head. Sized from the TILE, not
    // from whatever room the pane has left.
    ui.add_space(4.0);
    space3d(
        ui,
        ui.id().with(("keyer-space", cx.node)),
        k,
        cloud,
        *sel,
        egui::vec2(
            ui.available_width(),
            (crate::widgets::TILE * 0.96 * zoom).max(120.0),
        ),
    );
    // Shoulder drags on the range brackets park their softness here (a
    // bracket only borrows one axis and cannot reach the blob).
    if let Some(v) = take_soft(ui, cx.node) {
        k.blobs[*sel].soft = v;
    }
    out
}

/// Shared plot chrome: background plus a CLIP to the plot's own square. The
/// clip is load-bearing — a leaned or very soft selection ellipse is
/// legitimately much larger than the plot, and unclipped it sweeps across
/// the whole properties panel.
fn plot_frame(ui: &mut egui::Ui, side: f32) -> (egui::Response, egui::Painter, Rect) {
    let (resp, painter) =
        ui.allocate_painter(Vec2::splat(side), Sense::click_and_drag());
    let r = resp.rect;
    painter.rect_filled(r, 4.0, theme::well());
    (resp, painter.with_clip_rect(r), r)
}

/// Knocks back everything the key does NOT select, by the LIVE matte —
/// union, subtraction, levels, gamma and invert included. Shading the
/// selection instead would fight the plot's own colours; taking light away
/// from the rest makes the selection read instantly, and it is the same
/// gesture the range brackets use. Trustworthy because it runs the real
/// matte, not an idealised outline of one ellipse.
fn matte_overlay(
    painter: &egui::Painter,
    r: Rect,
    k: &KeyerParams,
    lab_at: &dyn Fn(f32, f32) -> Option<[f32; 3]>,
    n: usize,
) {
    let mut mesh = epaint::Mesh::default();
    for iy in 0..=n {
        for ix in 0..=n {
            let (u, v) = (ix as f32 / n as f32, iy as f32 / n as f32);
            let a = match lab_at(u, v) {
                Some(lab) => keyer::matte(k, keyer::linear_from_oklab(lab)),
                None => 0.0,
            };
            mesh.vertices.push(epaint::Vertex {
                pos: Pos2::new(r.min.x + u * r.width(), r.min.y + v * r.height()),
                uv: epaint::WHITE_UV,
                color: Color32::from_black_alpha(((1.0 - a) * 190.0) as u8),
            });
        }
    }
    let w = n as u32 + 1;
    for iy in 0..n as u32 {
        for ix in 0..n as u32 {
            let i = iy * w + ix;
            mesh.indices.extend([i, i + 1, i + w, i + 1, i + w + 1, i + w]);
        }
    }
    painter.add(egui::Shape::mesh(mesh));
}

/// The a/b plane at the selected blob's lightness: hue around, chroma out.
fn hue_chroma_plot(
    ui: &mut egui::Ui,
    k: &mut KeyerParams,
    sel: usize,
    cloud: Option<&Vec<[f32; 3]>>,
    side: f32,
    node: u64,
    lms: engine::color::Mat3,
) {
    let (resp, painter, r) = plot_frame(ui, side);
    let c = r.center();
    let rad = side * 0.5 - 8.0;
    let l = k.blobs[sel].center[0].clamp(0.05, 1.15);
    // Painted disc: a polar mesh so the rim is a circle, not a staircase.
    const RINGS: usize = 9;
    const SECT: usize = 48;
    let mut mesh = epaint::Mesh::default();
    mesh.vertices.push(epaint::Vertex {
        pos: c,
        uv: epaint::WHITE_UV,
        color: oklch_color(l, 0.0, 0.0),
    });
    for ring in 1..=RINGS {
        let t = ring as f32 / RINGS as f32;
        for s in 0..SECT {
            let a = s as f32 / SECT as f32 * std::f32::consts::TAU;
            mesh.vertices.push(epaint::Vertex {
                // y is flipped so +b (yellow) is UP, like every vectorscope.
                pos: c + Vec2::new(a.cos(), -a.sin()) * rad * t,
                uv: epaint::WHITE_UV,
                color: oklch_color(l, t * CMAX, a),
            });
        }
    }
    let idx = |ring: usize, s: usize| (1 + (ring - 1) * SECT + s % SECT) as u32;
    for s in 0..SECT {
        mesh.indices.extend([0, idx(1, s), idx(1, s + 1)]);
    }
    for ring in 1..RINGS {
        for s in 0..SECT {
            mesh.indices.extend([
                idx(ring, s),
                idx(ring, s + 1),
                idx(ring + 1, s),
                idx(ring, s + 1),
                idx(ring + 1, s + 1),
                idx(ring + 1, s),
            ]);
        }
    }
    painter.add(egui::Shape::mesh(mesh));

    let to_ab = |p: Pos2| {
        let d = (p - c) / rad * CMAX;
        [d.x, -d.y]
    };
    let to_scr = |a: f32, b: f32| c + Vec2::new(a, -b) / CMAX * rad;

    matte_overlay(
        &painter,
        r,
        k,
        &|u, v| {
            let p = Pos2::new(r.min.x + u * r.width(), r.min.y + v * r.height());
            if c.distance(p) > rad {
                return None;
            }
            let ab = to_ab(p);
            Some([l, ab[0], ab[1]])
        },
        40,
    );
    scatter(&painter, cloud, lms, &|lab| {
        let p = to_scr(lab[1], lab[2]);
        (c.distance(p) <= rad).then_some(p)
    });
    painter.circle_stroke(c, rad, Stroke::new(1.0, theme::border()));
    // Scale marks: the neutral axis and the half-chroma ring, so saturation
    // can be read off the plot instead of guessed.
    // Faint enough that it can never be mistaken for a selection outline.
    painter.circle_stroke(c, rad * 0.5, Stroke::new(1.0, theme::grid_line()));
    for d in [Vec2::new(4.0, 0.0), Vec2::new(0.0, 4.0)] {
        painter.line_segment([c - d, c + d], Stroke::new(1.0, theme::axis_line()));
    }

    // Every blob outlined; the selected one gets handles.
    for (i, b) in k.blobs.iter().enumerate() {
        let (u, v) = b.frame();
        let cen = to_scr(b.center[1], b.center[2]);
        let axes = |scale: f32| {
            (
                Vec2::new(u[0], -u[1]) / CMAX * rad * b.tol[1] * scale,
                Vec2::new(v[0], -v[1]) / CMAX * rad * b.tol[2] * scale,
            )
        };
        let dim = if i == sel { 1.0 } else { 0.45 };
        let col = if b.weight < 0.0 {
            theme::keyer_subtract()
        } else {
            theme::keyer_blob()
        };
        let (ax, bx) = axes(1.0);
        ellipse(&painter, cen, ax, bx, Stroke::new(1.6, col.gamma_multiply(dim)));
        let (ax2, bx2) = axes(1.0 + b.tol_soft());
        ellipse(
            &painter,
            cen,
            ax2,
            bx2,
            Stroke::new(1.0, col.gamma_multiply(0.4 * dim)),
        );
    }

    // Handles on the selected blob: centre moves the colour, the two rim
    // pucks are the saturation and hue tolerances on their own axes.
    let b = &mut k.blobs[sel];
    let (u, v) = b.frame();
    let cen = to_scr(b.center[1], b.center[2]);
    let ax = Vec2::new(u[0], -u[1]) / CMAX * rad;
    let bx = Vec2::new(v[0], -v[1]) / CMAX * rad;
    if let Some(d) = handle(ui, &painter, cen, node, ("kc", sel), 6.0) {
        let ab = to_ab(cen + d);
        b.center[1] = ab[0];
        b.center[2] = ab[1];
    }
    // The four pucks sit on four different directions — centre, +u
    // (saturation), +v (hue), -u (softness). Stacking two of them on the
    // same axis makes them unpickable the moment the key gets tight, which
    // is exactly when a key needs the most aiming.
    if let Some(d) = handle(ui, &painter, cen + ax * b.tol[1], node, ("ks", sel), 4.5) {
        b.tol[1] = (b.tol[1] + d.dot(ax.normalized()) / rad * CMAX).max(1e-3);
    }
    if let Some(d) = handle(ui, &painter, cen + bx * b.tol[2], node, ("kh", sel), 4.5) {
        b.tol[2] = (b.tol[2] + d.dot(bx.normalized()) / rad * CMAX).max(1e-3);
    }
    // The outer ring is a handle too: pull it out to widen the falloff
    // without moving what is solidly selected. Tolerance and softness are
    // two rings you grab, which is the whole "soft but directable" idea in
    // one gesture.
    let so = cen - ax * (b.tol[1] * (1.0 + b.tol_soft()));
    if let Some(d) = handle(ui, &painter, so, node, ("kso", sel), 3.6) {
        let outer = (b.tol[1] * (1.0 + b.soft) - d.dot(ax.normalized()) / rad * CMAX)
            .max(b.tol[1] * 1.001);
        b.soft = (outer / b.tol[1].max(1e-4) - 1.0).clamp(0.0, 8.0);
    }
    if resp.hovered() {
        resp.on_hover_text(
            "hue / chroma at the blob's lightness\ncentre puck moves the colour · rim pucks are the saturation and hue tolerances",
        );
    }
}

/// A slice through the selected blob's hue: chroma across, lightness up.
fn light_chroma_plot(
    ui: &mut egui::Ui,
    k: &mut KeyerParams,
    sel: usize,
    cloud: Option<&Vec<[f32; 3]>>,
    side: f32,
    node: u64,
    lms: engine::color::Mat3,
) {
    let (resp, painter, r) = plot_frame(ui, side);
    let (_, hue) = k.blobs[sel].chroma_hue();
    const NX: usize = 28;
    const NY: usize = 28;
    let mut mesh = epaint::Mesh::default();
    for iy in 0..=NY {
        for ix in 0..=NX {
            let (u, v) = (ix as f32 / NX as f32, iy as f32 / NY as f32);
            mesh.vertices.push(epaint::Vertex {
                pos: Pos2::new(r.min.x + u * r.width(), r.min.y + v * r.height()),
                uv: epaint::WHITE_UV,
                color: oklch_color((1.0 - v) * LMAX, u * CMAX, hue),
            });
        }
    }
    let w = NX as u32 + 1;
    for iy in 0..NY as u32 {
        for ix in 0..NX as u32 {
            let i = iy * w + ix;
            mesh.indices.extend([i, i + 1, i + w, i + 1, i + w + 1, i + w]);
        }
    }
    painter.add(egui::Shape::mesh(mesh));

    // Points map by (chroma, lightness); hue is the plane, not an axis.
    let to_scr = |c: f32, l: f32| {
        Pos2::new(
            r.min.x + (c / CMAX).clamp(0.0, 1.0) * r.width(),
            r.max.y - (l / LMAX).clamp(0.0, 1.0) * r.height(),
        )
    };
    matte_overlay(
        &painter,
        r,
        k,
        &|u, v| {
            let c = u * CMAX;
            Some([(1.0 - v) * LMAX, c * hue.cos(), c * hue.sin()])
        },
        36,
    );
    scatter(&painter, cloud, lms, &|lab| {
        Some(to_scr(lab[1].hypot(lab[2]), lab[0]))
    });

    let mut tilt = k.tilt;
    let b = &mut k.blobs[sel];
    let (c0, _) = b.chroma_hue();
    let cen = to_scr(c0, b.center[0]);
    let sx = r.width() / CMAX;
    let sy = r.height() / LMAX;
    // The tilt shears the saturation axis on lightness, so the outline has
    // to shear with it or the plot would misreport what is selected.
    let ax = Vec2::new(b.tol[1] * sx, 0.0);
    let by = Vec2::new(b.tol[0] * tilt * sx, -b.tol[0] * sy);
    ellipse(&painter, cen, ax, by, Stroke::new(1.6, theme::keyer_blob()));
    let s = 1.0 + b.tol_soft();
    ellipse(
        &painter,
        cen,
        ax * s,
        by * s,
        Stroke::new(1.0, theme::keyer_blob().gamma_multiply(0.4)),
    );
    if let Some(d) = handle(ui, &painter, cen, node, ("klc", sel), 6.0) {
        b.center[0] = (b.center[0] - d.y / sy).clamp(0.0, LMAX);
        let nc = (c0 + d.x / sx).max(0.0);
        let (_, h) = b.chroma_hue();
        b.center[1] = nc * h.cos();
        b.center[2] = nc * h.sin();
    }
    // One puck, two axes: pull it up to reach further into highlight, push
    // it sideways to LEAN the whole ellipsoid. Real subjects lose or gain
    // chroma as they brighten, and leaning beats widening every time —
    // a fatter selection grabs things you did not ask for.
    if let Some(d) = handle(ui, &painter, cen + by, node, ("kl", sel), 4.5) {
        b.tol[0] = (b.tol[0] - d.y / sy).max(1e-3);
        tilt = (tilt + d.x / sx / b.tol[0].max(1e-3)).clamp(-3.0, 3.0);
    }
    if let Some(d) = handle(ui, &painter, cen + ax, node, ("kcx", sel), 4.5) {
        b.tol[1] = (b.tol[1] + d.x / sx).max(1e-3);
    }
    let so = cen - by * (1.0 + b.tol_soft());
    if let Some(d) = handle(ui, &painter, so, node, ("kso2", sel), 3.6) {
        let outer = (b.tol[0] * (1.0 + b.soft) + d.y / sy).max(b.tol[0] * 1.001);
        b.soft = (outer / b.tol[0].max(1e-4) - 1.0).clamp(0.0, 8.0);
    }
    k.tilt = tilt;
    if resp.hovered() {
        resp.on_hover_text(
            "lightness (up) against saturation (across), sliced at the blob's hue\nthe lean is the tilt control",
        );
    }
}

/// The image's own pixels, so you can see where the subject lives before
/// aiming at it. Drawn ON TOP of the knock-back (a cloud underneath it would
/// vanish in exactly the regions you are trying to judge), in light dots —
/// density reads as a glow.
fn scatter(
    painter: &egui::Painter,
    cloud: Option<&Vec<[f32; 3]>>,
    lms: engine::color::Mat3,
    project: &dyn Fn([f32; 3]) -> Option<Pos2>,
) {
    let Some(cloud) = cloud else { return };
    for rgb in cloud {
        if let Some(p) = project(keyer::oklab_from_linear_in(*rgb, lms)) {
            painter.rect_filled(
                Rect::from_min_size(p, Vec2::splat(2.0)),
                0.0,
                Color32::from_white_alpha(105),
            );
        }
    }
}

/// A parametric ellipse from two (not necessarily perpendicular) semi-axes —
/// the tilted plot needs the sheared form.
fn ellipse(painter: &egui::Painter, c: Pos2, ax: Vec2, by: Vec2, stroke: Stroke) {
    let pts: Vec<Pos2> = (0..=48)
        .map(|i| {
            let t = i as f32 / 48.0 * std::f32::consts::TAU;
            c + ax * t.cos() + by * t.sin()
        })
        .collect();
    painter.add(egui::Shape::line(pts, stroke));
}

/// A draggable puck. Returns this frame's delta while dragged.
fn handle(
    ui: &mut egui::Ui,
    painter: &egui::Painter,
    p: Pos2,
    node: u64,
    id: (&'static str, usize),
    r: f32,
) -> Option<Vec2> {
    let resp = ui.interact(
        Rect::from_center_size(p, Vec2::splat(r * 2.6)),
        ui.id().with((id.0, node, id.1)),
        Sense::click_and_drag(),
    );
    let hot = resp.hovered() || resp.dragged();
    painter.circle(
        p,
        r,
        if hot { theme::handle_hot() } else { theme::handle() },
        Stroke::new(1.5, theme::handle_rim()),
    );
    resp.dragged().then(|| resp.drag_delta())
}

fn controls(
    ui: &mut egui::Ui,
    k: &mut KeyerParams,
    cx: &mut RowCtx,
    sel: &mut usize,
    viewing: bool,
) -> KeyerOut {
    let mut out = KeyerOut::default();
    let w = ui.available_width().max(150.0);
    sample_strip(ui, k, sel);
    let i = *sel;
    let idx = i as u16;

    // Three range brackets, one per axis of the ellipsoid, each drawn over
    // the colours that axis actually sweeps. A bracket says what it grabs:
    // the solid span is the core, the fading shoulders are the softness, and
    // both edges are draggable. This is the tolerance/softness pair Flame
    // puts in two numeric fields, made into one object you can grab.
    let (l0, c0, h0) = {
        let b = &k.blobs[i];
        let (c, h) = b.chroma_hue();
        (b.center[0], c, h)
    };
    let soft = k.blobs[i].soft;
    out.hud = out.hud.or(range_bar(
        ui,
        cx,
        RangeBar {
            label: "light",
            w,
            center: k.blobs[i].center[0],
            tol: k.blobs[i].tol[0],
            soft,
            span: (0.0, LMAX),
            chans: [Channel::KeyCenterL(idx), Channel::KeyTolL(idx)],
            paint: &|t| oklch_color(t * LMAX, c0, h0),
        },
        &mut |c, t| {
            k.blobs[i].center[0] = c.clamp(0.0, LMAX);
            k.blobs[i].tol[0] = t.max(1e-3);
        },
    ));
    out.hud = out.hud.or(range_bar(
        ui,
        cx,
        RangeBar {
            label: "sat",
            w,
            center: c0,
            tol: k.blobs[i].tol[1],
            soft,
            span: (0.0, CMAX),
            chans: [Channel::KeyCenterA(idx), Channel::KeyTolC(idx)],
            paint: &|t| oklch_color(l0.clamp(0.05, 1.15), t * CMAX, h0),
        },
        &mut |c, t| {
            let nc = c.max(0.0);
            k.blobs[i].center[1] = nc * h0.cos();
            k.blobs[i].center[2] = nc * h0.sin();
            k.blobs[i].tol[1] = t.max(1e-3);
        },
    ));
    // Hue reads in DEGREES around the blob's own hue, because that is the
    // only way the number means anything to a person — but it is stored as a
    // distance in the a/b plane, so it keeps shrinking as a pixel desaturates.
    let cref = c0.max(0.02);
    let deg = |x: f32| (x / cref).atan();
    out.hud = out.hud.or(range_bar(
        ui,
        cx,
        RangeBar {
            label: "hue",
            w,
            center: 0.0,
            tol: deg(k.blobs[i].tol[2]),
            soft,
            span: (-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2),
            chans: [Channel::KeyCenterB(idx), Channel::KeyTolH(idx)],
            paint: &|t| {
                let a = h0 + (t - 0.5) * std::f32::consts::PI;
                oklch_color(l0.clamp(0.05, 1.15), c0.max(0.08), a)
            },
        },
        &mut |c, t| {
            // Sliding the bracket sideways rotates the hue itself.
            let nh = h0 + c;
            k.blobs[i].center[1] = c0 * nh.cos();
            k.blobs[i].center[2] = c0 * nh.sin();
            k.blobs[i].tol[2] = (cref * t.clamp(0.0, 1.5).tan()).max(1e-3);
        },
    ));
    // Softness is one number shared by the three shoulders above; grabbing
    // any shoulder edits it, and this readout is where it lands.
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(format!(
                "  soft ×{:.2}   ±{:.0}°   tilt {:+.2}",
                k.blobs[i].soft,
                k.blobs[i].hue_degrees(),
                k.tilt
            ))
            .small()
            .weak(),
        );
        widgets::key_dot(ui, cx, &[Channel::KeySoft(idx), Channel::KeyTilt]);
    });

    out.hud = out.hud.or(falloff_ramp(ui, cx, k, w));

    ui.horizontal(|ui| {
        chip(ui, &mut k.invert, "invert", "flip the matte");
        let mut is_matte = k.view == KeyerView::Matte;
        if chip(
            ui,
            &mut is_matte,
            "matte",
            "show the matte instead of the source WHILE THIS NODE IS VIEWED — downstream always gets the matte either way",
        ) {
            k.view = if is_matte { KeyerView::Matte } else { KeyerView::Source };
        }
        if viewing {
            if ui
                .small_button("resample")
                .on_hover_text("re-read the image behind the plots (a full-frame readback)")
                .clicked()
            {
                out.want_cloud = true;
            }
        } else if ui
            .small_button("view")
            .on_hover_text(
                "sampling reads the pixels on screen, so this node has to be the one being viewed",
            )
            .clicked()
        {
            out.want_view = true;
        }
    });
    ui.label(
        egui::RichText::new(if viewing {
            "drag on the image to refit · shift-drag adds · alt-drag subtracts"
        } else {
            "view this node to sample from the image"
        })
        .small()
        .weak(),
    );
    out
}

/// A latching chip: a button that reads as on or off. Returns true on change.
fn chip(ui: &mut egui::Ui, v: &mut bool, label: &str, hover: &str) -> bool {
    let r = ui.selectable_label(*v, egui::RichText::new(label).small());
    let hit = r.on_hover_text(hover).clicked();
    if hit {
        *v = !*v;
    }
    hit
}

/// The blob strip: one chip per sample. Click selects, VERTICAL DRAG sets its
/// strength (the chip fills from the bottom), right-click flips it between
/// adding to the key and carving out of it, and the ✕ removes it.
fn sample_strip(ui: &mut egui::Ui, k: &mut KeyerParams, sel: &mut usize) {
    let mut remove: Option<usize> = None;
    let mut flip: Option<usize> = None;
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 3.0;
        for i in 0..k.blobs.len() {
            let (rect, resp) =
                ui.allocate_exact_size(Vec2::new(26.0, 22.0), Sense::click_and_drag());
            let b = &mut k.blobs[i];
            let mag = b.weight.abs().clamp(0.0, 1.0);
            let p = ui.painter();
            p.rect_filled(rect.shrink(1.0), 3.0, theme::well());
            // Strength reads as a fill height — the chip is the fader.
            let fill = Rect::from_min_max(
                Pos2::new(rect.left() + 1.0, rect.bottom() - 1.0 - (rect.height() - 2.0) * mag),
                Pos2::new(rect.right() - 1.0, rect.bottom() - 1.0),
            );
            p.rect_filled(fill, 3.0, blob_color(b));
            if b.weight < 0.0 {
                // Subtractive blobs wear a minus bar.
                p.line_segment(
                    [
                        Pos2::new(rect.left() + 7.0, rect.center().y),
                        Pos2::new(rect.right() - 7.0, rect.center().y),
                    ],
                    Stroke::new(2.5, Color32::from_gray(0x12)),
                );
            }
            if i == *sel {
                p.rect_stroke(
                    rect,
                    3.0,
                    Stroke::new(1.5, theme::selection()),
                    egui::StrokeKind::Inside,
                );
            }
            if resp.dragged() {
                let d = -resp.drag_delta().y / rect.height();
                let m = (mag + d).clamp(0.0, 1.0);
                b.weight = if b.weight < 0.0 { -m } else { m };
            } else if resp.clicked() {
                *sel = i;
            }
            if resp.secondary_clicked() {
                flip = Some(i);
            }
            if resp.middle_clicked() {
                remove = Some(i);
            }
            resp.on_hover_text(format!(
                "sample {}  ·  strength {:.2}{}\ndrag up/down for strength · right-click to {} · middle-click removes",
                i + 1,
                mag,
                if b.weight < 0.0 { " (subtracting)" } else { "" },
                if b.weight < 0.0 { "add" } else { "subtract" },
            ));
        }
        if k.blobs.len() < MAX_BLOBS {
            let (rect, resp) =
                ui.allocate_exact_size(Vec2::new(20.0, 22.0), Sense::click());
            let p = ui.painter();
            p.rect_stroke(
                rect.shrink(1.0),
                3.0,
                Stroke::new(1.0, theme::lift(theme::border(), 20)),
                egui::StrokeKind::Inside,
            );
            let c = rect.center();
            for d in [Vec2::new(4.0, 0.0), Vec2::new(0.0, 4.0)] {
                p.line_segment([c - d, c + d], Stroke::new(1.4, theme::text()));
            }
            if resp
                .on_hover_text("add a sample (adding one recompiles once; moving one never does)")
                .clicked()
            {
                let b = k.blobs.get(*sel).copied().unwrap_or_default();
                k.blobs.push(b);
                *sel = k.blobs.len() - 1;
            }
        }
    });
    if let Some(i) = flip {
        k.blobs[i].weight = -k.blobs[i].weight;
    }
    if let Some(i) = remove {
        if k.blobs.len() > 1 {
            k.blobs.remove(i);
            *sel = (*sel).min(k.blobs.len() - 1);
        }
    }
}

/// Everything a range bracket needs. `paint` maps 0..1 across the bar to the
/// colour that position stands for.
struct RangeBar<'a> {
    label: &'static str,
    w: f32,
    center: f32,
    tol: f32,
    soft: f32,
    span: (f32, f32),
    chans: [Channel; 2],
    paint: &'a dyn Fn(f32) -> Color32,
}

/// One axis of the ellipsoid as a draggable bracket over a live colour ramp.
///
/// Grab the middle to slide the whole selection, an inner edge to set the
/// tolerance, an outer shoulder to set the softness. Ctrl-drag throws the row
/// onto the viewer as a HUD slider, like every other parameter row.
fn range_bar(
    ui: &mut egui::Ui,
    cx: &mut RowCtx,
    b: RangeBar,
    set: &mut dyn FnMut(f32, f32),
) -> Option<HudSlider> {
    const H: f32 = 20.0;
    let mut hud = None;
    let mut spawn = None;
    ui.horizontal(|ui| {
        ui.add_sized(
            [30.0, 16.0],
            egui::Label::new(egui::RichText::new(b.label).small()),
        );
        let bw = (b.w - 62.0).max(60.0);
        let (rect, resp) =
            ui.allocate_exact_size(Vec2::new(bw, H), Sense::click_and_drag());
        let p = ui.painter();
        let (lo, hi) = b.span;
        let to_x = |v: f32| rect.left() + ((v - lo) / (hi - lo)).clamp(0.0, 1.0) * rect.width();
        let per_px = (hi - lo) / rect.width();

        // Colour ramp behind the bracket: 40 slabs is plenty at this size and
        // keeps the gamut bisection off the frame budget.
        const N: usize = 40;
        for i in 0..N {
            let t = (i as f32 + 0.5) / N as f32;
            let x0 = rect.left() + i as f32 / N as f32 * rect.width();
            let x1 = rect.left() + (i + 1) as f32 / N as f32 * rect.width();
            p.rect_filled(
                Rect::from_min_max(Pos2::new(x0, rect.top()), Pos2::new(x1, rect.bottom())),
                0.0,
                (b.paint)(t),
            );
        }
        // Everything outside the selection is knocked back, so the bracket
        // reads as a window cut in the ramp rather than a line drawn on it.
        let outer = b.tol * (1.0 + b.soft.max(0.0));
        let (ci, co) = (b.tol, outer);
        for (a, z) in [
            (rect.left(), to_x(b.center - co)),
            (to_x(b.center + co), rect.right()),
        ] {
            if z > a {
                p.rect_filled(
                    Rect::from_min_max(Pos2::new(a, rect.top()), Pos2::new(z, rect.bottom())),
                    0.0,
                    Color32::from_black_alpha(190),
                );
            }
        }
        // Shoulders: a few steps of fade so the softness is visible as a
        // gradient, not just a wider box.
        const STEPS: usize = 6;
        for side in [-1.0f32, 1.0] {
            for s in 0..STEPS {
                let t0 = s as f32 / STEPS as f32;
                let t1 = (s + 1) as f32 / STEPS as f32;
                let v0 = b.center + side * (ci + (co - ci) * t0);
                let v1 = b.center + side * (ci + (co - ci) * t1);
                let a = 190.0 * (0.5 * (t0 + t1));
                let (x0, x1) = (to_x(v0), to_x(v1));
                p.rect_filled(
                    Rect::from_min_max(
                        Pos2::new(x0.min(x1), rect.top()),
                        Pos2::new(x0.max(x1), rect.bottom()),
                    ),
                    0.0,
                    Color32::from_black_alpha(a as u8),
                );
            }
        }
        p.rect_stroke(
            rect,
            2.0,
            Stroke::new(1.0, theme::border()),
            egui::StrokeKind::Inside,
        );
        // Grab marks: two solid core edges, two hollow shoulder edges.
        for (v, solid) in [
            (b.center - ci, true),
            (b.center + ci, true),
            (b.center - co, false),
            (b.center + co, false),
        ] {
            let x = to_x(v);
            p.line_segment(
                [Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())],
                Stroke::new(
                    if solid { 1.8 } else { 1.0 },
                    if solid { theme::handle_hot() } else { theme::lift(theme::handle(), -68) },
                ),
            );
        }
        p.line_segment(
            [
                Pos2::new(to_x(b.center), rect.center().y - 3.0),
                Pos2::new(to_x(b.center), rect.center().y + 3.0),
            ],
            Stroke::new(1.0, theme::handle_hot()),
        );

        // Which part of the bracket the gesture grabbed — and which SIDE of
        // it — is decided ONCE, at press. Re-deciding per frame makes edges
        // swap under the cursor the moment a drag crosses one.
        let key = ui.id().with(("kbar", cx.node, b.label));
        if resp.drag_started() {
            let g = resp
                .interact_pointer_pos()
                .map(|q| {
                    let v = lo + (q.x - rect.left()) * per_px;
                    let d = (v - b.center).abs();
                    let part = if d < ci * 0.45 {
                        0.0 // middle: slide the whole window
                    } else if (d - ci).abs() <= (d - co).abs() {
                        1.0 // an inner edge: tolerance
                    } else {
                        2.0 // a shoulder: softness
                    };
                    // Dragging right on the RIGHT edge widens; on the left
                    // edge it narrows. One sign carries that.
                    [part, if v < b.center { -1.0 } else { 1.0 }]
                })
                .unwrap_or([0.0, 1.0]);
            ui.data_mut(|d| d.insert_temp(key, g));
        }
        let (suppress, sp) = ctrl_drag(ui, cx, &resp);
        if sp {
            spawn = Some((b.chans.to_vec(), b.tol));
        }
        if resp.dragged() && !suppress {
            let fine = if ui.input(|i| i.modifiers.shift) { 0.25 } else { 1.0 };
            let dv = resp.drag_delta().x * per_px * fine;
            let g = ui.data(|d| d.get_temp::<[f32; 2]>(key)).unwrap_or([0.0, 1.0]);
            match g[0] as u8 {
                0 => set(b.center + dv, b.tol),
                1 => set(b.center, (b.tol + dv * g[1]).max(1e-4)),
                _ => {
                    let outer = (co + dv * g[1]).max(b.tol * 1.001);
                    // Softness is one number on the blob, shared by all three
                    // brackets; the row only borrows one axis, so it parks the
                    // new value for `keyer_ui` to write back.
                    soft_out(ui, cx.node, (outer / b.tol.max(1e-4) - 1.0).clamp(0.0, 8.0));
                }
            }
        }
        widgets::key_dot(ui, cx, &b.chans);
    });
    if let Some((chans, v)) = spawn {
        hud = Some(HudSlider {
            node: cx.node,
            chans,
            label: b.label.to_string(),
            lo: 0.0,
            hi: (v * 4.0).max(0.1),
            pos: Vec2::ZERO,
            scale: Vec2::new(1.0, 1.0),
            ..Default::default()
        });
    }
    hud
}

/// Softness is shared by all three brackets, so a shoulder drag parks its new
/// value here and `keyer_ui` writes it back after the row returns — the
/// bracket only borrows one axis and cannot reach the blob itself.
fn soft_out(ui: &egui::Ui, node: u64, v: f32) {
    ui.data_mut(|d| d.insert_temp(egui::Id::new(("keyer-soft", node)), v));
}

fn take_soft(ui: &egui::Ui, node: u64) -> Option<f32> {
    let id = egui::Id::new(("keyer-soft", node));
    let v = ui.data(|d| d.get_temp::<f32>(id));
    if v.is_some() {
        ui.data_mut(|d| d.remove::<f32>(id));
    }
    v
}

/// Ctrl-drag handling, mirroring `widgets::ctrl_drag` (which is private).
fn ctrl_drag(ui: &egui::Ui, cx: &RowCtx, r: &egui::Response) -> (bool, bool) {
    let ctrl = ui.input(|i| i.modifiers.ctrl || i.modifiers.command);
    let touching = r.drag_started() || r.dragged() || r.drag_stopped();
    (touching && (cx.hud_live || ctrl), r.drag_started() && ctrl)
}

/// The matte's own response curve, as one object instead of three sliders:
/// a black-to-white ramp with the finished response drawn across it, a stop
/// at each end (the levels) and a middle puck you drag up and down to bend
/// the roll-off. This is where "very soft" gets turned into "soft but it
/// still goes solid where I need it to".
fn falloff_ramp(
    ui: &mut egui::Ui,
    cx: &mut RowCtx,
    k: &mut KeyerParams,
    w: f32,
) -> Option<HudSlider> {
    const H: f32 = 44.0;
    let mut hud = None;
    let bw = (w - 26.0).max(80.0);
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(bw, H), Sense::click_and_drag());
    let p = ui.painter();
    p.rect_filled(rect, 3.0, theme::well());
    let x_of = |t: f32| rect.left() + t.clamp(0.0, 1.0) * rect.width();
    let y_of = |m: f32| rect.bottom() - m.clamp(0.0, 1.0) * rect.height();
    // The response, sampled exactly as the shader computes it.
    let resp_at = |x: f32| {
        let m = ((x - k.lo) / (k.hi - k.lo).max(1e-4)).clamp(0.0, 1.0);
        let m = m.powf(k.gamma.max(1e-3));
        if k.invert {
            1.0 - m
        } else {
            m
        }
    };
    const N: usize = 56;
    for i in 0..N {
        let t = (i as f32 + 0.5) / N as f32;
        let g = (resp_at(t) * 255.0) as u8;
        p.rect_filled(
            Rect::from_min_max(
                Pos2::new(x_of(i as f32 / N as f32), rect.bottom() - 8.0),
                Pos2::new(x_of((i + 1) as f32 / N as f32), rect.bottom()),
            ),
            0.0,
            Color32::from_gray(g),
        );
    }
    let pts: Vec<Pos2> = (0..=N)
        .map(|i| {
            let t = i as f32 / N as f32;
            Pos2::new(x_of(t), y_of(resp_at(t)))
        })
        .collect();
    p.add(egui::Shape::line(pts, Stroke::new(1.6, theme::curve_line())));
    p.rect_stroke(
        rect,
        3.0,
        Stroke::new(1.0, theme::border()),
        egui::StrokeKind::Inside,
    );

    // Three pucks: the two stops on the floor, the bend in the middle.
    let mid = 0.5 * (k.lo + k.hi);
    let pucks = [
        (Pos2::new(x_of(k.lo), rect.bottom() - 4.0), 0u8),
        (Pos2::new(x_of(k.hi), rect.bottom() - 4.0), 1),
        (Pos2::new(x_of(mid), y_of(resp_at(mid))), 2),
    ];
    for (pos, which) in pucks {
        let r = ui.interact(
            Rect::from_center_size(pos, Vec2::splat(15.0)),
            ui.id().with(("kramp", cx.node, which)),
            Sense::click_and_drag(),
        );
        let hot = r.hovered() || r.dragged();
        p.circle(
            pos,
            if which == 2 { 4.5 } else { 4.0 },
            if hot { theme::handle_hot() } else { theme::handle() },
            Stroke::new(1.2, theme::handle_rim()),
        );
        let (suppress, spawn) = ctrl_drag(ui, cx, &r);
        if spawn {
            hud = Some(HudSlider {
                node: cx.node,
                chans: vec![match which {
                    0 => Channel::KeyLo,
                    1 => Channel::KeyHi,
                    _ => Channel::KeyGamma,
                }],
                label: ["lo", "hi", "edge"][which as usize].to_string(),
                lo: 0.0,
                hi: if which == 2 { 4.0 } else { 1.0 },
                pos: Vec2::ZERO,
                scale: Vec2::new(1.0, 1.0),
                ..Default::default()
            });
        }
        if r.dragged() && !suppress {
            let d = r.drag_delta();
            let fine = ui.input(|i| i.modifiers.shift) as u8 as f32 * 0.75;
            let k_scale = 1.0 - fine;
            match which {
                0 => k.lo = (k.lo + d.x / rect.width() * k_scale).clamp(0.0, k.hi - 0.01),
                1 => k.hi = (k.hi + d.x / rect.width() * k_scale).clamp(k.lo + 0.01, 1.0),
                // Up bends the roll-off toward solid, down toward soft; the
                // exponent moves geometrically so both directions feel equal.
                _ => {
                    k.gamma =
                        (k.gamma * (d.y / rect.height() * 3.0 * k_scale).exp()).clamp(0.15, 6.0)
                    }
            }
        }
    }
    if resp.hovered() {
        resp.on_hover_text(
            "matte response: the two floor pucks are the black and white points, the middle puck bends the roll-off",
        );
    }
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(format!(
                "  levels {:.2}–{:.2}  edge {:.2}",
                k.lo, k.hi, k.gamma
            ))
            .small()
            .weak(),
        );
        widgets::key_dot(ui, cx, &[Channel::KeyLo, Channel::KeyHi, Channel::KeyGamma]);
    });
    hud
}

/// Softness as the plots draw it — a blob with zero softness still needs a
/// visible outer ring, or the falloff ellipse would sit exactly on the core.
trait TolSoft {
    fn tol_soft(&self) -> f32;
}
impl TolSoft for KeyBlob {
    fn tol_soft(&self) -> f32 {
        self.soft.max(0.02)
    }
}

// ---------------------------------------------------------------------------
// the keyed volume, in three dimensions
// ---------------------------------------------------------------------------

/// Where the view sits. Yaw spins around the neutral axis, pitch tips it.
#[derive(Clone, Copy)]
pub struct Orbit {
    pub yaw: f32,
    pub pitch: f32,
}

impl Default for Orbit {
    fn default() -> Self {
        // Three-quarter view: enough tilt to read the lightness axis as
        // vertical, enough spin that the a and b axes are distinguishable.
        Self { yaw: 0.6, pitch: 0.45 }
    }
}

/// Lightness the view is centred on, so an orbit spins about the middle of
/// the volume rather than about black.
const L_MID: f32 = 0.62;
/// Lightness is squashed into chroma-comparable units for DISPLAY only.
/// Oklab L runs to about 1.3 over a plate while chroma rarely leaves ±0.28,
/// so drawn to a common scale the volume is a thin vertical spike with
/// everything interesting crushed into it. Flame's cube normalises its axes
/// for the same reason. The metric is untouched — this is the eye's scale,
/// not the key's.
const L_SCALE: f32 = 0.45;

/// Project an Oklab point to the view plane. Returns the offset from the
/// centre in view units, plus a DEPTH — bigger is nearer the eye, which the
/// scatter sorts on so the cloud reads as a solid volume instead of a mess.
fn project(lab: [f32; 3], o: Orbit) -> (Vec2, f32) {
    let (x, y, z) = (lab[1], lab[2], (lab[0] - L_MID) * L_SCALE);
    let (cy, sy) = (o.yaw.cos(), o.yaw.sin());
    let (x1, y1) = (x * cy - y * sy, x * sy + y * cy);
    let (cp, sp) = (o.pitch.cos(), o.pitch.sin());
    (Vec2::new(x1, -(z * cp - y1 * sp)), y1 * cp + z * sp)
}

/// A point on a blob's tolerance surface, from a point on the unit sphere.
///
/// The metric shears chroma by lightness (`tilt`), so the surface is not an
/// upright ellipsoid and cannot be drawn as one: undo the shear the same way
/// the shader applies it, or the wireframe describes a different volume from
/// the one being keyed.
fn blob_point(b: &KeyBlob, tilt: f32, unit: [f32; 3], scale: f32) -> [f32; 3] {
    let (u, v) = b.frame();
    let dl = b.tol[0] * unit[2] * scale;
    let dc = b.tol[1] * unit[0] * scale + tilt * dl;
    let dh = b.tol[2] * unit[1] * scale;
    [
        b.center[0] + dl,
        b.center[1] + dc * u[0] + dh * v[0],
        b.center[2] + dc * u[1] + dh * v[1],
    ]
}

/// The keyed volume as a solid you can turn over: Flame's RGB viewer, in a
/// perceptual space and showing the actual selection rather than a hull of
/// the samples.
///
/// The picture's own colours are the cloud, depth-sorted so it reads as a
/// volume; each sample is a wireframe of its tolerance surface with its
/// softness surface ghosted outside it. Drag to orbit — it is a READ-ONLY
/// view, because a 3D handle you drag with a 2D mouse is a guess, and the
/// plots and brackets above already give every axis a real one.
pub fn space3d(
    ui: &mut egui::Ui,
    id: egui::Id,
    k: &KeyerParams,
    cloud: Option<&Vec<[f32; 3]>>,
    sel: usize,
    size: Vec2,
) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(size, Sense::drag());
    let mut o = ui.data(|d| d.get_temp::<[f32; 2]>(id)).map_or_else(
        Orbit::default,
        |v| Orbit { yaw: v[0], pitch: v[1] },
    );
    if resp.dragged() {
        let d = resp.drag_delta();
        o.yaw += d.x * 0.01;
        // Stop short of the poles: past vertical the volume turns inside out
        // and the lightness axis stops reading as up.
        o.pitch = (o.pitch + d.y * 0.01).clamp(-1.4, 1.4);
    }
    ui.data_mut(|d| d.insert_temp(id, [o.yaw, o.pitch]));
    draw_space3d(&ui.painter().with_clip_rect(rect), rect, k, cloud, sel, o, true);
    resp.on_hover_text("the keyed volume in Oklab — drag to orbit")
}

/// The volume, painted. Split out from the interactive widget so the viewer
/// HUD can draw the same picture without allocating a Ui — the HUD is
/// something you glance at mid-gesture, not something you drive.
pub fn draw_space3d(
    p: &egui::Painter,
    rect: Rect,
    k: &KeyerParams,
    cloud: Option<&Vec<[f32; 3]>>,
    sel: usize,
    o: Orbit,
    opaque: bool,
) {
    if opaque {
        p.rect_filled(rect, 4.0, theme::well());
    }
    let c = rect.center();
    // Fit the VOLUME to the box: the widest thing in it is the hue ring at
    // full chroma, and with the axes normalized the lightness axis is about
    // the same span, so half the shorter side over CMAX (plus a small margin
    // for the wireframes leaning out of it) fills the panel.
    let s = rect.size().min_elem() * 0.5 / (CMAX * 1.12);
    let to_scr = |lab: [f32; 3]| {
        let (v, depth) = project(lab, o);
        (c + v * s, depth)
    };

    // The neutral axis, and a hue ring at mid lightness: without a reference
    // an orbited cloud is impossible to orient.
    let axis: Vec<Pos2> = (0..=24)
        .map(|i| to_scr([i as f32 / 24.0 * 1.3, 0.0, 0.0]).0)
        .collect();
    p.add(egui::Shape::line(axis, Stroke::new(1.0, theme::axis_line())));
    for i in 0..48 {
        let (a0, a1) = (
            i as f32 / 48.0 * std::f32::consts::TAU,
            (i + 1) as f32 / 48.0 * std::f32::consts::TAU,
        );
        let q = |a: f32| to_scr([L_MID, CMAX * a.cos(), CMAX * a.sin()]).0;
        p.line_segment([q(a0), q(a1)], Stroke::new(1.0, oklch_color(0.7, 0.13, a0)));
    }

    // The picture's colours, painted as themselves, far to near.
    if let Some(cloud) = cloud {
        let mut pts: Vec<([f32; 3], Pos2, f32)> = cloud
            .iter()
            .map(|rgb| {
                let lab = keyer::oklab_from_linear(*rgb);
                let (sp, d) = to_scr(lab);
                (lab, sp, d)
            })
            .filter(|(_, sp, _)| rect.contains(*sp))
            .collect();
        pts.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        for (lab, sp, _) in &pts {
            let ch = lab[1].hypot(lab[2]);
            let col = oklch_color(lab[0].clamp(0.05, 1.15), ch, lab[2].atan2(lab[1]));
            p.rect_filled(Rect::from_center_size(*sp, Vec2::splat(2.0)), 0.0, col);
        }
    }

    // Each sample's tolerance surface, as three principal rings, with the
    // softness surface ghosted outside it.
    for (i, b) in k.blobs.iter().enumerate() {
        let hot = i == sel;
        let base = if b.weight < 0.0 {
            theme::keyer_subtract()
        } else {
            theme::keyer_blob()
        };
        for (scale, alpha) in [(1.0f32, if hot { 220 } else { 90 }), (1.0 + b.soft, if hot { 70 } else { 30 })] {
            for axis in 0..3 {
                let ring: Vec<Pos2> = (0..=40)
                    .map(|j| {
                        let t = j as f32 / 40.0 * std::f32::consts::TAU;
                        let (ct, st) = (t.cos(), t.sin());
                        let unit = match axis {
                            0 => [ct, st, 0.0],
                            1 => [ct, 0.0, st],
                            _ => [0.0, ct, st],
                        };
                        to_scr(blob_point(b, k.tilt, unit, scale)).0
                    })
                    .collect();
                p.add(egui::Shape::line(
                    ring,
                    Stroke::new(if hot { 1.4 } else { 1.0 }, base.gamma_multiply(alpha as f32 / 255.0)),
                ));
            }
        }
    }
    p.rect_stroke(
        rect,
        4.0,
        Stroke::new(1.0, theme::border()),
        egui::StrokeKind::Inside,
    );
}

/// The orbit a node's 3D view is currently at, so the HUD and the panel show
/// the same angle rather than two unrelated views of one volume.
pub fn orbit_of(ctx: &egui::Context, id: egui::Id) -> Orbit {
    ctx.data(|d| d.get_temp::<[f32; 2]>(id))
        .map_or_else(Orbit::default, |v| Orbit { yaw: v[0], pitch: v[1] })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anim::AnimCurve;
    use engine::graph::NodeKind;
    use std::collections::BTreeMap;

    /// The disc's colours have to be the colours they claim to be: gamut
    /// mapping pulls chroma in, so hue and lightness must survive it even
    /// where the requested chroma cannot.
    #[test]
    fn wheel_colours_keep_their_hue_and_lightness() {
        // Oklab hue 0 is reddish, 2.0 rad is greenish, -1.6 is blue-ish.
        let red = oklch_color(0.6, 0.15, 0.0);
        assert!(red.r() > red.g() && red.r() > red.b(), "{red:?}");
        let green = oklch_color(0.7, 0.15, 2.4);
        assert!(green.g() > green.r() && green.g() > green.b(), "{green:?}");
        let blue = oklch_color(0.5, 0.15, -1.6);
        assert!(blue.b() > blue.r(), "{blue:?}");
        // Neutral stays neutral, and lightness is monotone up the axis.
        let mut last = 0u8;
        for l in [0.1f32, 0.3, 0.5, 0.7, 0.9] {
            let c = oklch_color(l, 0.0, 0.0);
            assert!(c.r() == c.g() && c.g() == c.b(), "neutral must be grey: {c:?}");
            assert!(c.r() > last, "lightness must rise");
            last = c.r();
        }
        // Far outside any gamut: still a sane colour, not a clipped mess.
        let wild = oklch_color(0.6, 2.0, 1.0);
        assert!(wild.r() as u16 + wild.g() as u16 + wild.b() as u16 > 30);
    }

    /// The whole panel, driven headlessly through egui: meshes, brackets,
    /// pucks and all. Catches a panic or an empty draw without a window.
    #[test]
    fn panel_draws_headless() {
        let ctx = egui::Context::default();
        let mut params = KeyerParams::default();
        params.blobs.push(KeyBlob { weight: -0.7, ..KeyBlob::default() });
        let cloud: Vec<[f32; 3]> =
            (0..500).map(|i| { let t = i as f32 / 500.0; [t, t * 0.6, 0.2] }).collect();
        let ekind = NodeKind::Keyer(params.clone());
        let mut anim: BTreeMap<Channel, AnimCurve> = BTreeMap::new();
        let mut sel = 1usize;
        let out = ctx.run_ui(egui::RawInput::default(), |ui| {
            let mut cx = RowCtx {
                node: 7,
                anim: &mut anim,
                ekind: &ekind,
                t: 0,
                hud_live: false,
                driven: Vec::new(),
            };
            keyer_ui(
                ui,
                &mut params,
                &mut cx,
                &mut sel,
                Some(&cloud),
                true,
                1.0,
                engine::color::Colorspace::ACESCG,
            );
        });
        let shapes: usize = out
            .shapes
            .iter()
            .map(|c| match &c.shape {
                egui::Shape::Vec(v) => v.len(),
                _ => 1,
            })
            .sum();
        assert!(shapes > 50, "the panel drew almost nothing ({shapes} shapes)");
        // A pure draw must not edit the params it was handed.
        assert_eq!(params.blobs.len(), 2);
        assert_eq!(params.lo, 0.0);
    }

    /// A keyer whose blob list was emptied (a hand-edited setup, or an undo
    /// landing mid-gesture) must heal rather than index out of bounds.
    #[test]
    fn empty_blob_list_heals() {
        let ctx = egui::Context::default();
        let mut params = KeyerParams { blobs: Vec::new(), ..Default::default() };
        let ekind = NodeKind::Keyer(params.clone());
        let mut anim: BTreeMap<Channel, AnimCurve> = BTreeMap::new();
        let mut sel = 9usize;
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            let mut cx = RowCtx {
                node: 1,
                anim: &mut anim,
                ekind: &ekind,
                t: 0,
                hud_live: false,
                driven: Vec::new(),
            };
            keyer_ui(
                ui,
                &mut params,
                &mut cx,
                &mut sel,
                None,
                false,
                1.0,
                engine::color::Colorspace::ACESCG,
            );
        });
        assert_eq!(params.blobs.len(), 1);
        assert_eq!(sel, 0);
    }

    /// Draws the panel with a plausible skin-tone key over a scatter of
    /// image colours and writes `target/keyer-panel.png`. Asserts the plots
    /// are actually PAINTED (a wide spread of hues, not a flat rectangle) —
    /// and leaves a picture behind for a human to judge.
    #[test]
    fn panel_renders_to_a_png() {
        // A skin-ish key: warm hue, wide in lightness, tight in hue, plus a
        // subtractive bite and a lean.
        let skin = keyer::oklab_from_linear([0.36, 0.19, 0.13]);
        let mut params = KeyerParams {
            blobs: vec![
                KeyBlob { center: skin, tol: [0.30, 0.045, 0.030], soft: 1.4, weight: 1.0 },
                KeyBlob {
                    center: keyer::oklab_from_linear([0.55, 0.08, 0.06]),
                    tol: [0.10, 0.03, 0.02],
                    soft: 0.8,
                    weight: -0.7,
                },
            ],
            tilt: 0.5,
            lo: 0.05,
            hi: 0.9,
            gamma: 1.3,
            ..Default::default()
        };
        // A cloud that looks like a real frame: a warm subject against a
        // cool background, plus a spread of neutrals.
        let mut cloud = Vec::new();
        for i in 0..2400 {
            let t = i as f32 / 2400.0;
            let e = 0.05 + t * 0.9;
            cloud.push(match (i % 3) as u32 {
                0 => [e * 0.9, e * 0.52, e * 0.38],  // skin
                1 => [e * 0.35, e * 0.5, e * 0.95],  // sky
                _ => [e, e * 0.98, e * 0.96],        // neutrals
            });
        }
        let ekind = NodeKind::Keyer(params.clone());
        let mut anim: BTreeMap<Channel, AnimCurve> = BTreeMap::new();
        let mut sel = 0usize;
        // 3 tiles wide, 2 tall — the plots row plus the volume row.
        let shot = crate::snapshot::shoot(766.0, 508.0, |ui| {
            let mut cx = RowCtx {
                node: 7,
                anim: &mut anim,
                ekind: &ekind,
                t: 0,
                hud_live: false,
                driven: Vec::new(),
            };
            keyer_ui(
                ui,
                &mut params,
                &mut cx,
                &mut sel,
                Some(&cloud),
                true,
                1.0,
                engine::color::Colorspace::ACESCG,
            );
        });

        // The hue disc must be a WHEEL: sample a ring of it and demand the
        // colours actually go round (a flat or grey plot is the failure this
        // catches, and it is invisible to a "did it draw" assertion).
        let (cx0, cy0) = (98.0f32, 125.0f32);
        let hues: Vec<[u8; 3]> = (0..8)
            .map(|k| {
                let a = k as f32 / 8.0 * std::f32::consts::TAU;
                shot.at(cx0 + a.cos() * 60.0, cy0 - a.sin() * 60.0)
            })
            .collect();
        let reddest = hues.iter().enumerate().max_by_key(|(_, c)| c[0] as i32 - c[1] as i32);
        let bluest = hues.iter().enumerate().max_by_key(|(_, c)| c[2] as i32 - c[1] as i32);
        assert_ne!(reddest.map(|r| r.0), bluest.map(|b| b.0), "disc is not a hue wheel");
        let spread = hues
            .iter()
            .map(|c| c[0] as i32 - c[2] as i32)
            .fold((i32::MAX, i32::MIN), |(lo, hi), v| (lo.min(v), hi.max(v)));
        assert!(spread.1 - spread.0 > 80, "disc barely varies in hue: {spread:?}");
        shot.save("keyer-panel");
    }

    /// The volume view has to survive a degenerate key: a neutral blob has
    /// no hue to build its frame from, and a zero-softness one puts the
    /// ghost surface exactly on the core.
    #[test]
    fn the_volume_draws_for_degenerate_keys() {
        for (center, tol, soft) in [
            ([0.5f32, 0.0, 0.0], [0.2f32; 3], 0.0f32),   // neutral, hard
            ([0.5, 0.0, 0.0], [0.001; 3], 4.0),          // neutral, tiny, soft
            ([1.4, 0.3, -0.2], [0.5, 0.5, 0.5], 1.0),    // way off the plate
        ] {
            let k = KeyerParams {
                blobs: vec![KeyBlob { center, tol, soft, weight: 1.0 }],
                tilt: 1.2,
                ..Default::default()
            };
            let shot = crate::snapshot::shoot(200.0, 200.0, |ui| {
                space3d(ui, egui::Id::new("v"), &k, None, 0, egui::vec2(190.0, 190.0));
            });
            // Something was drawn, and nothing became NaN (a NaN vertex
            // takes the whole mesh with it, so the panel would go empty).
            let painted = (0..shot.h)
                .step_by(4)
                .filter(|y| (0..shot.w).step_by(4).any(|x| shot.px(x, *y) != [18, 18, 18]))
                .count();
            assert!(painted > 8, "volume did not draw for {center:?}/{tol:?}");
        }
    }

    /// The wireframe must describe the volume the SHADER keys, tilt and all.
    /// Drawing an upright ellipsoid while the metric shears one would be a
    /// picture of a different key.
    #[test]
    fn the_wireframe_follows_the_tilt_shear() {
        let b = KeyBlob {
            center: [0.6, 0.1, 0.0],
            tol: [0.2, 0.05, 0.05],
            soft: 1.0,
            weight: 1.0,
        };
        let p = KeyerParams { blobs: vec![b], tilt: 1.5, ..Default::default() };
        // Top and bottom of the tolerance surface, along the lightness axis.
        let top = blob_point(&b, p.tilt, [0.0, 0.0, 1.0], 1.0);
        let bot = blob_point(&b, p.tilt, [0.0, 0.0, -1.0], 1.0);
        // A sheared ellipsoid leans: its poles differ in CHROMA, not just in
        // lightness. An upright one would put them at the same chroma.
        let ch = |q: [f32; 3]| q[1].hypot(q[2]);
        assert!(
            (ch(top) - ch(bot)).abs() > 0.05,
            "wireframe is upright but the metric is sheared: {top:?} {bot:?}"
        );
        // And the two drawn surfaces must be the two the shader means: the
        // inner ring is where coverage is still SOLID (the core boundary is
        // where falloff begins, not where it is half), the ghosted outer one
        // is where it has fallen to nothing.
        for unit in [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
            let core = blob_point(&b, p.tilt, unit, 1.0);
            let cov = keyer::blob_coverage_tilted(&b, core, p.tilt);
            assert!(cov > 0.99, "core surface is not solid: coverage {cov} at {core:?}");
            let edge = blob_point(&b, p.tilt, unit, 1.0 + b.soft);
            let cov = keyer::blob_coverage_tilted(&b, edge, p.tilt);
            assert!(cov < 0.01, "softness surface still selects: coverage {cov}");
        }
    }
}
