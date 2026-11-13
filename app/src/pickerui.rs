//! Viewer pixel sampler.
//!
//! Reads the comp's own pixels back off the GPU and reports them, so a value
//! can be checked rather than eyeballed. Two numbers matter and both are
//! shown:
//!
//! - the **scene-linear** value in the comp's working space, which is what
//!   every node downstream actually operates on, and
//! - the **display** value the view transform is putting on screen, which is
//!   what you are looking at.
//!
//! Confusing those two is the single easiest mistake to make in a
//! scene-referred comp, so they are labelled and never merged.
//!
//! The read is asynchronous and one-at-a-time: a sample lands a frame or two
//! late, which is invisible while hovering and keeps the picker off the
//! critical path.

use crate::theme;
use engine::color::Colorspace;
use engine::viewtf::ViewSettings;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Sample box sizes offered, in comp pixels. An average over a small box is
/// what you want on anything with grain in it; 1x1 is what you want when
/// checking a specific pixel.
pub const SIZES: [u32; 3] = [1, 3, 5];

/// One resolved sample: where it came from and what was there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    pub x: i32,
    pub y: i32,
    /// Box edge in comp pixels.
    pub n: u32,
    /// Mean scene-linear RGBA over the box, in the working space.
    pub rgba: [f32; 4],
}

struct PixelRead {
    buffer: wgpu::Buffer,
    x: i32,
    y: i32,
    w: u32,
    h: u32,
    stride: u32,
    ready: Arc<AtomicBool>,
    mapped: bool,
}

#[derive(Default)]
pub struct PickerState {
    /// Whether the readout is showing. Toggled with P.
    pub on: bool,
    /// Cursor in comp output pixels, refreshed by the viewer each frame.
    /// `None` when the pointer is off the image.
    pub cursor: Option<(i32, i32)>,
    /// Index into [`SIZES`].
    pub size: usize,
    /// The sample under the cursor right now.
    pub live: Option<Sample>,
    /// Samples held for comparison. Pinned with shift+P.
    pub pinned: Vec<Sample>,
    reads: Vec<PixelRead>,
}

impl PickerState {
    pub fn box_size(&self) -> u32 {
        SIZES[self.size.min(SIZES.len() - 1)]
    }

    /// Hold the current sample for comparison.
    pub fn pin(&mut self) -> bool {
        match self.live {
            Some(s) => {
                self.pinned.push(s);
                true
            }
            None => false,
        }
    }

    /// Queue a readback of the box under the cursor, if there is one and
    /// nothing is already in flight.
    ///
    /// One read at a time on purpose: they land in a frame or two, and a
    /// queue of them would just report staler and staler pixels while
    /// costing more.
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        tex: &wgpu::Texture,
        out: (u32, u32),
    ) {
        if !self.on || !self.reads.is_empty() {
            return;
        }
        let Some((cx, cy)) = self.cursor else { return };
        let (ow, oh) = out;
        if ow == 0 || oh == 0 || cx < 0 || cy < 0 || cx >= ow as i32 || cy >= oh as i32 {
            return;
        }
        let n = self.box_size().min(ow).min(oh);
        let x = (cx - n as i32 / 2).clamp(0, (ow - n) as i32) as u32;
        let y = (cy - n as i32 / 2).clamp(0, (oh - n) as i32) as u32;
        let stride = (n * 8).div_ceil(256) * 256;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("picker-read"),
            size: (stride * n) as u64,
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
                    rows_per_image: Some(n),
                },
            },
            wgpu::Extent3d { width: n, height: n, depth_or_array_layers: 1 },
        );
        self.reads.push(PixelRead {
            buffer,
            x: cx,
            y: cy,
            w: n,
            h: n,
            stride,
            ready: Arc::new(AtomicBool::new(false)),
            mapped: false,
        });
    }

    /// Map whatever has landed and update [`Self::live`]. Call once a frame.
    pub fn poll(&mut self) {
        for r in self.reads.iter_mut() {
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
        let mut done = Vec::new();
        for (i, r) in self.reads.iter().enumerate() {
            if !r.ready.load(Ordering::Acquire) {
                continue;
            }
            let view = r.buffer.slice(..).get_mapped_range().expect("a mapped buffer");
            let mut sum = [0.0f64; 4];
            let mut count = 0.0f64;
            for row in 0..r.h as usize {
                let off = row * r.stride as usize;
                let px: &[half::f16] =
                    bytemuck::cast_slice(&view[off..off + r.w as usize * 8]);
                for c in px.chunks_exact(4) {
                    for k in 0..4 {
                        sum[k] += c[k].to_f32() as f64;
                    }
                    count += 1.0;
                }
            }
            drop(view);
            r.buffer.unmap();
            if count > 0.0 {
                self.live = Some(Sample {
                    x: r.x,
                    y: r.y,
                    n: r.w,
                    rgba: std::array::from_fn(|k| (sum[k] / count) as f32),
                });
            }
            done.push(i);
        }
        for i in done.into_iter().rev() {
            self.reads.remove(i);
        }
    }

    /// Drop the live sample when the pointer leaves the image, so a stale
    /// value never sits there looking current.
    pub fn set_cursor(&mut self, c: Option<(i32, i32)>) {
        if c.is_none() {
            self.live = None;
        }
        self.cursor = c;
    }
}

/// The display code values a sample is showing on screen, 0..1, or `None`
/// when the chosen transform exists only as WGSL (OpenDRT).
fn display_of(s: &Sample, view: &ViewSettings) -> Option<[f32; 3]> {
    view.display_rgb([s.rgba[0], s.rgba[1], s.rgba[2]])
}

fn swatch_color(d: Option<[f32; 3]>) -> egui::Color32 {
    match d {
        Some(d) => egui::Color32::from_rgb(
            (d[0].clamp(0.0, 1.0) * 255.0).round() as u8,
            (d[1].clamp(0.0, 1.0) * 255.0).round() as u8,
            (d[2].clamp(0.0, 1.0) * 255.0).round() as u8,
        ),
        None => theme::border(),
    }
}

/// Scene-linear values span a huge range, so a fixed decimal count is wrong
/// at both ends: 0.0001 reads as 0.000 and 4096 wastes four columns on
/// nothing. Switch to exponent notation outside the range where decimals
/// actually say something.
fn fmt_linear(v: f32) -> String {
    if v != 0.0 && (v.abs() < 1e-3 || v.abs() >= 1e4) {
        format!("{v:>10.3e}")
    } else {
        format!("{v:>10.4}")
    }
}

/// Channel letters, in the order the numbers are drawn.
const CHAN: [&str; 4] = ["r", "g", "b", "a"];
/// The r/g/b letters take the patch bay's label tints — the bright lane
/// colours made for type, so a letter reads over the dark HUD panel.
const LANE_HINT: [egui::Color32; 3] = crate::patchbay::LANE_LABEL;
/// Where the display line's first column starts, and each column's width
/// (a letter, a gap, up to three digits).
const DISPLAY_X: f32 = 166.0;
const DISPLAY_COL: f32 = 32.0;
/// Each scene-linear column: a letter, a gap, and a number up to ten
/// characters wide (`-1.234e-5`).
const LINEAR_COL: f32 = 76.0;

/// A number right-aligned so it ENDS at `end`, with its channel letter in
/// the lane's colour just before it — the letter sits against the number
/// whatever its width, so it never drifts off to the far side of a column.
#[allow(clippy::too_many_arguments)]
fn tagged(
    painter: &egui::Painter,
    end: f32,
    y: f32,
    num: &str,
    letter: &str,
    hint: egui::Color32,
    font: &egui::FontId,
    ink: egui::Color32,
) {
    let r = painter.text(egui::pos2(end, y), egui::Align2::RIGHT_TOP, num, font.clone(), ink);
    painter.text(
        egui::pos2(r.left() - 3.0, y),
        egui::Align2::RIGHT_TOP,
        letter,
        font.clone(),
        hint,
    );
}

/// Paint the readout into the viewer's bottom-left.
///
/// Painted, not an egui window: floating windows over the viewer silently
/// occlude gizmos and eat their clicks, and a readout that follows the
/// pointer is exactly where a gizmo wants to be.
pub fn paint(painter: &egui::Painter, picker: &PickerState, working: Colorspace, view: &ViewSettings) {
    if !picker.on {
        return;
    }
    let font = egui::FontId::monospace(11.0);
    let dim = theme::text();
    let bright = theme::text_strong();
    let line_h = 14.0;
    let pad = 6.0;

    let mut rows: Vec<(bool, Option<Sample>)> = vec![(true, picker.live)];
    rows.extend(picker.pinned.iter().map(|p| (false, Some(*p))));

    let clip = painter.clip_rect();
    let h = pad * 2.0 + line_h * (rows.len() as f32 * 2.0 + 1.0);
    let w = 340.0f32.min(clip.width() - 16.0);
    let rect = egui::Rect::from_min_size(
        egui::pos2(clip.min.x + 8.0, clip.max.y - 8.0 - h),
        egui::vec2(w, h),
    );
    painter.rect_filled(rect, 4.0, theme::hud_bg().gamma_multiply_u8(190));

    let mut y = rect.min.y + pad;
    let x = rect.min.x + pad;
    let label = match view.display_rgb([0.18, 0.18, 0.18]) {
        Some(_) => format!("{} · scene-linear / display", working.label()),
        None => format!("{} · scene-linear (display: OpenDRT is shader-only)", working.label()),
    };
    painter.text(
        egui::pos2(x, y),
        egui::Align2::LEFT_TOP,
        label,
        font.clone(),
        dim,
    );
    y += line_h;

    for (live, s) in rows {
        let Some(s) = s else {
            painter.text(
                egui::pos2(x, y),
                egui::Align2::LEFT_TOP,
                "  (pointer off the image)",
                font.clone(),
                dim,
            );
            y += line_h * 2.0;
            continue;
        };
        let d = display_of(&s, view);
        let head = if live { theme::text_strong() } else { theme::text() };
        // Swatch, then where the sample came from.
        painter.rect_filled(
            egui::Rect::from_min_size(egui::pos2(x, y + 1.0), egui::vec2(10.0, 10.0)),
            2.0,
            swatch_color(d),
        );
        let n = if s.n > 1 { format!(" {0}x{0}", s.n) } else { String::new() };
        painter.text(
            egui::pos2(x + 16.0, y),
            egui::Align2::LEFT_TOP,
            format!("{}x {:<6} y {:<6}{n}", if live { "" } else { "· " }, s.x, s.y),
            font.clone(),
            head,
        );
        let ink = if live { bright } else { dim };
        // Display line: 0-255 per channel, then the hex. Each number is
        // right-aligned in its own column with its channel letter just
        // before it, so a glance says which is which.
        match d {
            Some(d) => {
                let b = d.map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8);
                for i in 0..3 {
                    let end = x + DISPLAY_X + (i as f32 + 1.0) * DISPLAY_COL;
                    tagged(painter, end, y, &b[i].to_string(), CHAN[i], LANE_HINT[i], &font, ink);
                }
                painter.text(
                    egui::pos2(x + DISPLAY_X + 3.0 * DISPLAY_COL + 10.0, y),
                    egui::Align2::LEFT_TOP,
                    format!("#{:02X}{:02X}{:02X}", b[0], b[1], b[2]),
                    font.clone(),
                    ink,
                );
            }
            None => {
                painter.text(
                    egui::pos2(x + DISPLAY_X, y),
                    egui::Align2::LEFT_TOP,
                    "   —",
                    font.clone(),
                    ink,
                );
            }
        }
        y += line_h;
        // Scene-linear line: r, g, b, then alpha, the same way.
        for i in 0..4 {
            let end = x + 16.0 + (i as f32 + 1.0) * LINEAR_COL;
            let v = fmt_linear(s.rgba[i]);
            let hint = if i < 3 { LANE_HINT[i] } else { dim };
            tagged(painter, end, y, v.trim(), CHAN[i], hint, &font, ink);
        }
        y += line_h;
    }
}

/// Picker controls for the colour panel: box size and the pinned list.
pub fn panel_ui(ui: &mut egui::Ui, picker: &mut PickerState) {
    ui.horizontal(|ui| {
        ui.checkbox(&mut picker.on, "pixel readout")
            .on_hover_text("P in the viewer");
        ui.add_space(8.0);
        ui.label("sample");
        for (i, n) in SIZES.iter().enumerate() {
            ui.selectable_value(&mut picker.size, i, format!("{n}x{n}"));
        }
    });
    if picker.pinned.is_empty() {
        ui.weak("shift+P pins the sample under the pointer");
        return;
    }
    let mut drop = None;
    for (i, s) in picker.pinned.iter().enumerate() {
        ui.horizontal(|ui| {
            if ui.small_button("×").clicked() {
                drop = Some(i);
            }
            ui.monospace(format!(
                "{:>5},{:<5} {} {} {}",
                s.x,
                s.y,
                fmt_linear(s.rgba[0]).trim(),
                fmt_linear(s.rgba[1]).trim(),
                fmt_linear(s.rgba[2]).trim()
            ));
        });
    }
    if let Some(i) = drop {
        picker.pinned.remove(i);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The readout with every kind of number in it — a live sample, a
    /// pinned one, a negative, a highlight past 1 and a tiny value in
    /// exponent form — rendered with text, to LOOK at the r/g/b hints.
    /// `cargo test -p app readout_render -- --ignored` writes
    /// `target/pixel-readout-gpu.png`.
    #[test]
    #[ignore]
    fn readout_render() {
        let picker = PickerState {
            on: true,
            live: Some(Sample { x: 1024, y: 540, n: 3, rgba: [0.6523, 0.4812, -0.0312, 1.0] }),
            pinned: vec![Sample { x: 88, y: 911, n: 1, rgba: [4.25, 0.18, 0.00004, 0.5] }],
            ..Default::default()
        };
        crate::uishot::render("pixel-readout-gpu", 420.0, 120.0, 2.0, 2, |ui, _| {
            paint(ui.painter(), &picker, Colorspace::ACESCG, &ViewSettings::default());
        });
    }

    /// Scene-linear values run from grain in the blacks to blown highlights;
    /// a fixed format lies at one end or the other.
    #[test]
    fn linear_formatting_survives_the_whole_range() {
        assert!(fmt_linear(0.18).contains("0.18"));
        assert!(fmt_linear(0.0).contains('0'));
        assert!(fmt_linear(0.00004).contains('e'), "tiny values need an exponent");
        assert!(fmt_linear(65504.0).contains('e'), "huge values need an exponent");
        // Nothing ever comes back as an unreadable blank.
        for v in [-1.0f32, 0.0, 1e-9, 0.5, 1.0, 1e9] {
            assert!(!fmt_linear(v).trim().is_empty());
        }
    }

    #[test]
    fn box_size_is_always_one_of_the_offered_sizes() {
        let mut p = PickerState::default();
        for i in 0..10 {
            p.size = i;
            assert!(SIZES.contains(&p.box_size()));
        }
    }

    /// Leaving the image must clear the value, not leave a stale one sitting
    /// there looking live.
    #[test]
    fn losing_the_pointer_clears_the_sample() {
        let mut p = PickerState { on: true, ..Default::default() };
        p.live = Some(Sample { x: 1, y: 2, n: 1, rgba: [0.5; 4] });
        p.set_cursor(Some((3, 4)));
        assert!(p.live.is_some(), "moving within the image keeps the sample");
        p.set_cursor(None);
        assert!(p.live.is_none());
    }

    #[test]
    fn pinning_needs_something_to_pin() {
        let mut p = PickerState::default();
        assert!(!p.pin());
        assert!(p.pinned.is_empty());
        p.live = Some(Sample { x: 7, y: 8, n: 3, rgba: [0.1, 0.2, 0.3, 1.0] });
        assert!(p.pin());
        assert_eq!(p.pinned.len(), 1);
        assert_eq!(p.pinned[0].x, 7);
    }
}
