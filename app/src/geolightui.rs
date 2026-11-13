//! The 2dini node's **light** and **effects** pages: the look
//! (`engine::geolight::GeoLook`) that lights the drawing with radiance
//! cascades and lays SDF effects over it. Every number is a channel named
//! `look.<param>` (`Channel::Custom`), so it keys, cables and drives.

use crate::anim::{self, Channel};
use crate::rgbslider::RgbSpec;
use crate::widgets::{self, HudSlider, RowCtx};
use engine::color::{primaries_matrix, Colorspace, Primaries};
use engine::geolight::GeoLook;

/// The channel of a look param, by its name after `look.`.
pub fn chan(name: &str) -> Channel {
    Channel::Custom(anim::intern(&format!("{}{name}", engine::geolight::PREFIX)))
}

/// Every look param as a channel, in panel order.
pub fn channels() -> impl Iterator<Item = Channel> {
    engine::geolight::PARAMS.iter().map(|n| chan(n))
}

struct Rows<'a, 'b> {
    rcx: &'a mut RowCtx<'b>,
    working: Colorspace,
    out: Option<HudSlider>,
}

impl Rows<'_, '_> {
    fn slider(&mut self, ui: &mut egui::Ui, label: &str, v: &mut f32, lo: f32, hi: f32, name: &str) {
        if let Some(h) = widgets::slider(ui, self.rcx, label, v, lo, hi, &[chan(name)]) {
            self.out = Some(h);
        }
    }

    fn xy(&mut self, ui: &mut egui::Ui, label: &str, v: &mut [f32; 2], head: &str) {
        let [x, y] = v;
        if let Some(h) = widgets::drags(
            ui,
            self.rcx,
            label,
            &mut [x, y],
            0.25,
            (-64.0, 64.0),
            &[chan(&format!("{head}.x")), chan(&format!("{head}.y"))],
        ) {
            self.out = Some(h);
        }
    }

    /// An rgb row and its alpha (a straight colour); `alpha` false for a
    /// colour with no alpha (the sky).
    fn color(&mut self, ui: &mut egui::Ui, label: &str, c: &mut [f32], head: &str, alpha: bool) {
        let mut rgb = [c[0], c[1], c[2]];
        let mut gang = true;
        let spec = RgbSpec::scale(0.0, 4.0, self.working.primaries.luma_weights())
            .with_display(primaries_matrix(self.working.primaries, Primaries::Rec709));
        let chans = [chan(&format!("{head}.r")), chan(&format!("{head}.g")), chan(&format!("{head}.b"))];
        if let Some(h) = widgets::rgb_row(ui, self.rcx, label, &mut rgb, &mut gang, spec, &chans) {
            self.out = Some(h);
        }
        c[..3].copy_from_slice(&rgb);
        if alpha {
            self.slider(ui, "opacity", &mut c[3], 0.0, 1.0, &format!("{head}.a"));
        }
    }
}

/// The page for `tab`: "light", else the effects.
pub fn panel(ui: &mut egui::Ui, rcx: &mut RowCtx, look: &mut GeoLook, tab: &str, working: Colorspace) -> Option<HudSlider> {
    let mut r = Rows { rcx, working, out: None };
    if tab == "light" {
        ui.checkbox(&mut look.light, "light")
            .on_hover_text("shapes a Light node marks give off light; every shape casts shadows (holographic radiance cascades)");
        r.slider(ui, "intensity", &mut look.intensity, 0.0, 8.0, "intensity");
        r.slider(ui, "ambient", &mut look.ambient, 0.0, 1.0, "ambient");
        r.slider(ui, "air", &mut look.air, 0.0, 2.0, "air");
        r.slider(ui, "cell px", &mut look.spacing, 1.0, 16.0, "spacing");
        r.color(ui, "sky", &mut look.sky, "sky", false);
        let grid = engine::geolight::Grid::new(1920, 1080, look.spacing);
        ui.weak(format!(
            "slow path when on: {} cascades over a {}² grid, rerun whenever the drawing moves. A Light node inside makes a shape glow",
            grid.count, grid.extent
        ));
    } else {
        r.slider(ui, "offset", &mut look.offset, -32.0, 32.0, "offset");
        ui.checkbox(&mut look.outline, "outline");
        if look.outline {
            r.slider(ui, "width", &mut look.outline_width, 0.0, 32.0, "outline.width");
            r.slider(ui, "in / out", &mut look.outline_pos, -1.0, 1.0, "outline.pos");
            r.color(ui, "colour", &mut look.outline_color, "outline", true);
        }
        ui.checkbox(&mut look.glow, "outer glow");
        if look.glow {
            r.slider(ui, "size", &mut look.glow_size, 0.0, 128.0, "glow.size");
            r.color(ui, "colour", &mut look.glow_color, "glow", true);
        }
        ui.checkbox(&mut look.shadow, "drop shadow");
        if look.shadow {
            r.xy(ui, "offset", &mut look.shadow_offset, "shadow");
            r.slider(ui, "softness", &mut look.shadow_soft, 0.0, 64.0, "shadow.soft");
            r.color(ui, "colour", &mut look.shadow_color, "shadow", true);
        }
        ui.checkbox(&mut look.inner, "inner shadow");
        if look.inner {
            r.xy(ui, "offset", &mut look.inner_offset, "inner");
            r.slider(ui, "softness", &mut look.inner_soft, 0.0, 64.0, "inner.soft");
            r.color(ui, "colour", &mut look.inner_color, "inner", true);
        }
        ui.checkbox(&mut look.bevel, "bevel");
        if look.bevel {
            r.slider(ui, "depth", &mut look.bevel_depth, 0.0, 64.0, "bevel.depth");
            r.slider(ui, "angle", &mut look.bevel_angle, 0.0, 360.0, "bevel.angle");
            r.slider(ui, "strength", &mut look.bevel_strength, 0.0, 2.0, "bevel.strength");
        }
        ui.weak("read off one distance field of the drawing, made on the GPU when the drawing changes");
    }
    r.out
}
