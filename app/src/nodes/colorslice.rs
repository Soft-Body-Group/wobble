//! The Tinter's app half — the panel, the glyph and the channel bindings.
//! Internally it is still ColorSlice (Resolve's name for the tool, and the
//! key setups store — see `engine::nodes::colorslice`, which holds the model,
//! the WGSL and the params).

use crate::anim::Channel;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke, Vec2};
use engine::color::{Colorspace, Curve};
use engine::graph::{ColorSliceParams, SliceVector};
use engine::nodes::colorslice::{base_hue, lean_of, NAMES, SKIN, VECTORS};

/// Every animatable param, in panel order: each vector's center, hue, sat
/// and density, then the globals.
pub const CHANNELS: &[Channel] = &[
    Channel::SliceRedCenter,
    Channel::SliceRedHue,
    Channel::SliceRedSat,
    Channel::SliceRedDensity,
    Channel::SliceSkinCenter,
    Channel::SliceSkinHue,
    Channel::SliceSkinSat,
    Channel::SliceSkinDensity,
    Channel::SliceYellowCenter,
    Channel::SliceYellowHue,
    Channel::SliceYellowSat,
    Channel::SliceYellowDensity,
    Channel::SliceGreenCenter,
    Channel::SliceGreenHue,
    Channel::SliceGreenSat,
    Channel::SliceGreenDensity,
    Channel::SliceCyanCenter,
    Channel::SliceCyanHue,
    Channel::SliceCyanSat,
    Channel::SliceCyanDensity,
    Channel::SliceBlueCenter,
    Channel::SliceBlueHue,
    Channel::SliceBlueSat,
    Channel::SliceBlueDensity,
    Channel::SliceMagentaCenter,
    Channel::SliceMagentaHue,
    Channel::SliceMagentaSat,
    Channel::SliceMagentaDensity,
    Channel::SliceSat,
    Channel::SliceDensity,
    Channel::SliceSatDepth,
    Channel::SliceDenDepth,
    Channel::SliceSatBalance,
];

/// Panel size: three tiles wide and TWO tall. A row per vector with four
/// real sliders across it needs the width — one tile high, with the
/// globals beside the grid, a rail came out ~36px — and the globals sit
/// under the grid instead. Measured, not guessed: see the fit test.
pub const TILES: usize = 3;
pub const TILE_ROWS: usize = 2;

pub fn get(p: &ColorSliceParams, ch: Channel) -> Option<f32> {
    use Channel::*;
    Some(match ch {
        SliceRedCenter => p.vectors[0].center,
        SliceRedHue => p.vectors[0].hue,
        SliceRedSat => p.vectors[0].sat,
        SliceRedDensity => p.vectors[0].density,
        SliceSkinCenter => p.vectors[1].center,
        SliceSkinHue => p.vectors[1].hue,
        SliceSkinSat => p.vectors[1].sat,
        SliceSkinDensity => p.vectors[1].density,
        SliceYellowCenter => p.vectors[2].center,
        SliceYellowHue => p.vectors[2].hue,
        SliceYellowSat => p.vectors[2].sat,
        SliceYellowDensity => p.vectors[2].density,
        SliceGreenCenter => p.vectors[3].center,
        SliceGreenHue => p.vectors[3].hue,
        SliceGreenSat => p.vectors[3].sat,
        SliceGreenDensity => p.vectors[3].density,
        SliceCyanCenter => p.vectors[4].center,
        SliceCyanHue => p.vectors[4].hue,
        SliceCyanSat => p.vectors[4].sat,
        SliceCyanDensity => p.vectors[4].density,
        SliceBlueCenter => p.vectors[5].center,
        SliceBlueHue => p.vectors[5].hue,
        SliceBlueSat => p.vectors[5].sat,
        SliceBlueDensity => p.vectors[5].density,
        SliceMagentaCenter => p.vectors[6].center,
        SliceMagentaHue => p.vectors[6].hue,
        SliceMagentaSat => p.vectors[6].sat,
        SliceMagentaDensity => p.vectors[6].density,
        SliceSat => p.sat,
        SliceDensity => p.density,
        SliceSatDepth => p.sat_depth,
        SliceDenDepth => p.den_depth,
        SliceSatBalance => p.sat_balance,
        _ => return None,
    })
}

pub fn set(p: &mut ColorSliceParams, ch: Channel, v: f32) {
    use Channel::*;
    match ch {
        SliceRedCenter => p.vectors[0].center = v,
        SliceRedHue => p.vectors[0].hue = v,
        SliceRedSat => p.vectors[0].sat = v,
        SliceRedDensity => p.vectors[0].density = v,
        SliceSkinCenter => p.vectors[1].center = v,
        SliceSkinHue => p.vectors[1].hue = v,
        SliceSkinSat => p.vectors[1].sat = v,
        SliceSkinDensity => p.vectors[1].density = v,
        SliceYellowCenter => p.vectors[2].center = v,
        SliceYellowHue => p.vectors[2].hue = v,
        SliceYellowSat => p.vectors[2].sat = v,
        SliceYellowDensity => p.vectors[2].density = v,
        SliceGreenCenter => p.vectors[3].center = v,
        SliceGreenHue => p.vectors[3].hue = v,
        SliceGreenSat => p.vectors[3].sat = v,
        SliceGreenDensity => p.vectors[3].density = v,
        SliceCyanCenter => p.vectors[4].center = v,
        SliceCyanHue => p.vectors[4].hue = v,
        SliceCyanSat => p.vectors[4].sat = v,
        SliceCyanDensity => p.vectors[4].density = v,
        SliceBlueCenter => p.vectors[5].center = v,
        SliceBlueHue => p.vectors[5].hue = v,
        SliceBlueSat => p.vectors[5].sat = v,
        SliceBlueDensity => p.vectors[5].density = v,
        SliceMagentaCenter => p.vectors[6].center = v,
        SliceMagentaHue => p.vectors[6].hue = v,
        SliceMagentaSat => p.vectors[6].sat = v,
        SliceMagentaDensity => p.vectors[6].density = v,
        SliceSat => p.sat = v,
        SliceDensity => p.density = v,
        SliceSatDepth => p.sat_depth = v,
        SliceDenDepth => p.den_depth = v,
        SliceSatBalance => p.sat_balance = v,
        _ => {}
    }
}

/// The curve editor's name for one of this node's channels.
pub fn label(ch: Channel) -> Option<&'static str> {
    use Channel::*;
    Some(match ch {
        SliceRedCenter => "red center",
        SliceRedHue => "red hue",
        SliceRedSat => "red sat",
        SliceRedDensity => "red density",
        SliceSkinCenter => "skin center",
        SliceSkinHue => "skin hue",
        SliceSkinSat => "skin sat",
        SliceSkinDensity => "skin density",
        SliceYellowCenter => "yellow center",
        SliceYellowHue => "yellow hue",
        SliceYellowSat => "yellow sat",
        SliceYellowDensity => "yellow density",
        SliceGreenCenter => "green center",
        SliceGreenHue => "green hue",
        SliceGreenSat => "green sat",
        SliceGreenDensity => "green density",
        SliceCyanCenter => "cyan center",
        SliceCyanHue => "cyan hue",
        SliceCyanSat => "cyan sat",
        SliceCyanDensity => "cyan density",
        SliceBlueCenter => "blue center",
        SliceBlueHue => "blue hue",
        SliceBlueSat => "blue sat",
        SliceBlueDensity => "blue density",
        SliceMagentaCenter => "magenta center",
        SliceMagentaHue => "magenta hue",
        SliceMagentaSat => "magenta sat",
        SliceMagentaDensity => "magenta density",
        SliceSat => "tinter saturation",
        SliceDensity => "tinter density",
        SliceSatDepth => "sat depth",
        SliceDenDepth => "density depth",
        SliceSatBalance => "sat balance",
        _ => return None,
    })
}

/// The grid's columns, in the order each vector's channels are listed.
const COLUMNS: [&str; 4] = ["center", "hue", "saturation", "density"];
/// The vector column: chip, name and the highlight light.
const NAME_W: f32 = 104.0;
/// The row's reset, at the end.
const RESET_W: f32 = 20.0;
/// A vector row: taller than a plain row, so seven of them read as a list
/// of colours rather than a wall of rails.
const ROW_H: f32 = 38.0;
/// The space the rails' pictures are worked out in. A picture, not a
/// measurement: the comp's own space would only move the hues a hair.
const WS: Colorspace = Colorspace::LINEAR_REC709;

/// What each column's rail spans. Center's depends on the vector: skin may
/// lean less than the six.
fn span(i: usize, k: usize) -> (f32, f32) {
    match k {
        0 => (-lean_of(i), lean_of(i)),
        1 => (-60.0, 60.0),
        2 => (0.0, 3.0),
        _ => (-2.0, 2.0),
    }
}

fn field(v: &mut SliceVector, k: usize) -> &mut f32 {
    match k {
        0 => &mut v.center,
        1 => &mut v.hue,
        2 => &mut v.sat,
        _ => &mut v.density,
    }
}

/// A colour on the RGB hue circle, in LINEAR light — the circle the node's
/// vectors are laid on, so a pixel built here sits where the model says.
fn hue_linear(deg: f32, sat: f32, val: f32) -> [f32; 3] {
    let h = deg.rem_euclid(360.0) / 60.0;
    let c = val * sat;
    let x = c * (1.0 - (h % 2.0 - 1.0).abs());
    let m = val - c;
    let (r, g, b) = match h as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    [r + m, g + m, b + m]
}

/// Linear Rec.709 as a display colour, for painting.
fn shown(c: [f32; 3]) -> Color32 {
    let e = |v: f32| (Curve::Srgb.from_linear(v.clamp(0.0, 1.0)) * 255.0).round() as u8;
    Color32::from_rgb(e(c[0]), e(c[1]), e(c[2]))
}

/// A typical pixel of vector `i`: its hue where its Center has put it, at
/// a moderate colourfulness (skin is paler than the six).
fn typical(i: usize, p: &ColorSliceParams) -> [f32; 3] {
    let h = base_hue(i, WS) + p.center_of(i);
    hue_linear(h, if i == SKIN { 0.5 } else { 0.75 }, 0.5)
}

/// What rail `k` of vector `i` shows at `x`: CENTER, the hues the vector
/// can sit on; the other three, what the node — with every OTHER setting
/// as it is now — does to the vector's typical pixel with this control at
/// `x`. Run through the CPU mirror of the shader, so the rail cannot drift
/// from what the viewer shows.
fn rail_colour(p: &ColorSliceParams, i: usize, k: usize, x: f32) -> Color32 {
    if k == 0 {
        return shown(hue_linear(base_hue(i, WS) + x, if i == SKIN { 0.5 } else { 0.75 }, 0.5));
    }
    let mut q = *p;
    *field(&mut q.vectors[i], k) = x;
    shown(q.apply(typical(i, p), WS))
}

/// The chip a vector wears in the panel: the colour it controls, as DATA —
/// except skin, which is shown as a pale rose-beige rather than its true
/// hue. A skin tone at full saturation is orange, and orange stays out of
/// this UI (CLAUDE.md); the chip is a label, the picture is the truth.
fn chip_colour(i: usize) -> Color32 {
    match i {
        0 => Color32::from_rgb(0xd8, 0x3a, 0x3a),
        SKIN => Color32::from_rgb(0xd6, 0xae, 0xa8),
        2 => Color32::from_rgb(0xd8, 0xc8, 0x3a),
        3 => Color32::from_rgb(0x4a, 0xb8, 0x4a),
        4 => Color32::from_rgb(0x3a, 0xc0, 0xc8),
        5 => Color32::from_rgb(0x4a, 0x6a, 0xd8),
        _ => Color32::from_rgb(0xc8, 0x4a, 0xc8),
    }
}

/// Keep a HUD slider a row spawned. Assigning through a reference, not
/// `h = h.or(..)`: that MOVES `h` into the first closure that touches it,
/// and the panel's rows live inside several.
fn keep(slot: &mut Option<HudSlider>, got: Option<HudSlider>) {
    if got.is_some() {
        *slot = got;
    }
}

/// The vector's name cell: chip, name, and the highlight light. The whole
/// cell is the highlight toggle — Resolve's per-vector Highlight, with a
/// target the size of the name rather than a speck.
fn name_cell(ui: &mut egui::Ui, i: usize, highlight: &mut Option<u8>) {
    let (r, resp) = ui.allocate_exact_size(egui::vec2(NAME_W, ROW_H), egui::Sense::click());
    let on = *highlight == Some(i as u8);
    let pt = ui.painter();
    let chip = chip_colour(i);
    if on || resp.hovered() {
        let a = if on { 46 } else { 20 };
        pt.rect_filled(r.shrink2(egui::vec2(0.0, 1.0)), 4.0, chip.gamma_multiply_u8(a));
    }
    let sq = egui::Rect::from_center_size(r.left_center() + egui::vec2(11.0, 0.0), egui::vec2(12.0, 12.0));
    pt.rect_filled(sq, 3.0, chip);
    let mut name = NAMES[i].to_string();
    name[..1].make_ascii_uppercase();
    let ink = ui.visuals().text_color();
    pt.text(
        r.left_center() + egui::vec2(24.0, 0.0),
        egui::Align2::LEFT_CENTER,
        name,
        egui::FontId::proportional(13.0),
        ink,
    );
    // An EYE, not a dot: every value beside it already has a round key dot,
    // and a second circle read as one more. Open and lit in the vector's
    // colour while it is the one shown.
    let lc = r.right_center() - egui::vec2(12.0, 0.0);
    let (hw, hh) = (8.0, 4.5);
    let lid = |sign: f32| {
        (0..=12).map(move |k| {
            let x = -hw + 2.0 * hw * k as f32 / 12.0;
            lc + egui::vec2(x, sign * hh * (1.0 - (x / hw).powi(2)))
        })
    };
    let outline: Vec<Pos2> = lid(-1.0).chain(lid(1.0).collect::<Vec<_>>().into_iter().rev()).collect();
    let weak = ui.visuals().weak_text_color();
    let (edge, pupil) = if on { (Color32::WHITE, chip) } else { (weak, Color32::TRANSPARENT) };
    pt.add(egui::Shape::closed_line(outline, Stroke::new(1.2, edge)));
    pt.circle(lc, 2.4, pupil, Stroke::new(1.0, edge));
    if resp
        .on_hover_text(if on {
            "highlighting: only this vector's pixels keep their colour — click to stop"
        } else {
            "click to highlight: show only the pixels this vector moves"
        })
        .clicked()
    {
        *highlight = if on { None } else { Some(i as u8) };
    }
}

/// The properties panel, laid out the way Resolve's ColorSlice palette
/// reads: one ROW per vector — its colour, its name and its Highlight —
/// with Center, Hue, Saturation and Density across it as sliders whose
/// rails show what each does to that vector's colours; the globals under
/// the grid. Every slider is a full parameter row in a cell — keyable,
/// patchable, lockable, a ctrl-drag away from a HUD slider.
pub fn panel(
    ui: &mut egui::Ui,
    rcx: &mut RowCtx,
    p: &mut ColorSliceParams,
    viewing: bool,
) -> Option<HudSlider> {
    let mut h: Option<HudSlider> = None;
    // The rails are drawn from the settings as they stood at the top of the
    // frame: a copy, because the cells hold the live fields mutably.
    let snap = *p;
    let gap = ui.spacing().item_spacing.x;
    let cell_w = ((ui.available_width() - NAME_W - RESET_W - 5.0 * gap) / 4.0).max(60.0);
    ui.horizontal(|ui| {
        ui.add_sized([NAME_W, 14.0], egui::Label::new(egui::RichText::new("vector").small().weak()));
        for name in COLUMNS {
            ui.add_sized([cell_w, 14.0], egui::Label::new(egui::RichText::new(name).small().weak()));
        }
    });
    for i in 0..VECTORS {
        ui.horizontal(|ui| {
            ui.set_min_height(ROW_H);
            name_cell(ui, i, &mut p.highlight);
            for k in 0..COLUMNS.len() {
                let ch = CHANNELS[i * 4 + k];
                let rail = |x: f32| rail_colour(&snap, i, k, x);
                keep(
                    &mut h,
                    widgets::slider_cell(
                        ui,
                        rcx,
                        label(ch).unwrap_or(COLUMNS[k]),
                        field(&mut p.vectors[i], k),
                        span(i, k),
                        &[ch],
                        cell_w,
                        &rail,
                    ),
                );
            }
            if p.vectors[i] != SliceVector::default() {
                let b = ui.add_sized([RESET_W, 18.0], egui::Button::new("⟲").small());
                if b.on_hover_text(format!("reset {}", NAMES[i])).clicked() {
                    p.vectors[i] = SliceVector::default();
                }
            } else {
                ui.add_space(RESET_W);
            }
        });
    }
    ui.add_space(10.0);
    ui.label(egui::RichText::new("global").small().weak());
    ui.columns(2, |cols| {
        use Channel::*;
        let (l, r) = cols.split_at_mut(1);
        let (l, r) = (&mut l[0], &mut r[0]);
        keep(&mut h, widgets::slider(l, rcx, "saturation", &mut p.sat, 0.0, 3.0, &[SliceSat]));
        keep(&mut h, widgets::slider(l, rcx, "density", &mut p.density, -2.0, 2.0, &[SliceDensity]));
        keep(&mut h, widgets::slider(l, rcx, "sat balance", &mut p.sat_balance, -1.0, 1.0, &[SliceSatBalance]));
        keep(&mut h, widgets::slider(r, rcx, "sat depth", &mut p.sat_depth, 0.0, 2.0, &[SliceSatDepth]));
        keep(&mut h, widgets::slider(r, rcx, "den depth", &mut p.den_depth, 0.0, 2.0, &[SliceDenDepth]));
    });
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        if ui
            .small_button("reset all")
            .on_hover_text("every vector and every global back to no change")
            .clicked()
        {
            let hl = p.highlight;
            *p = ColorSliceParams { highlight: hl, ..Default::default() };
        }
        // A highlight only ever replaces the picture of the node being
        // VIEWED — that is what keeps it out of renders — so say so rather
        // than let it look broken.
        if p.highlight.is_some() && !viewing {
            ui.weak("highlight: view this node (V) to see it");
        } else {
            ui.weak("subtractive: more saturation deepens a colour instead of lighting it");
        }
    });
    h
}

/// The graph glyph: a disc cut into six slices, one of them lifted — the
/// wheel, and the one part of it you are working on. Drawn in the node's
/// own colours, not a hue wheel: a hue wheel is half orange.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let rr = r * 0.9;
    painter.circle(c, rr, fill, stroke);
    let lifted = fill.lerp_to_gamma(Color32::WHITE, 0.45);
    let mut wedge = vec![c];
    for k in 0..=6 {
        let a = std::f32::consts::TAU * (-0.25 + k as f32 / 36.0);
        wedge.push(c + Vec2::new(a.cos(), a.sin()) * rr);
    }
    painter.add(egui::Shape::convex_polygon(wedge, lifted, Stroke::NONE));
    let ink = Stroke::new((r * 0.1).max(1.0), crate::theme::node_ink());
    for k in 0..6 {
        let a = std::f32::consts::TAU * (-0.25 + k as f32 / 6.0);
        painter.line_segment([c, c + Vec2::new(a.cos(), a.sin()) * rr], ink);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::graph::NodeKind;

    /// Every channel reads and writes the field it names, and nothing else.
    #[test]
    fn every_channel_round_trips_to_its_own_field() {
        for (n, ch) in CHANNELS.iter().enumerate() {
            let mut p = ColorSliceParams::default();
            set(&mut p, *ch, 7.25);
            assert_eq!(get(&p, *ch), Some(7.25), "{ch:?}");
            let touched = CHANNELS.iter().filter(|c| get(&p, **c) == Some(7.25)).count();
            assert_eq!(touched, 1, "{ch:?} (#{n}) wrote more than its own field");
            assert!(label(*ch).is_some(), "{ch:?} has no curve-editor label");
        }
        assert_eq!(CHANNELS.len(), VECTORS * 4 + 5);
    }

    /// The panel WITH its text, on the GPU, to `target/tinter-panel-gpu.png`
    /// — the plain snapshot below draws no glyphs. For a look, not a check:
    /// `cargo test -p app tinter_panel_render -- --ignored`.
    #[test]
    #[ignore]
    fn tinter_panel_render() {
        let mut p = ColorSliceParams { highlight: Some(3), sat: 1.1, ..Default::default() };
        p.vectors[0].hue = 12.0;
        p.vectors[1].sat = 1.4;
        p.vectors[5].density = 0.6;
        let mut anim = Default::default();
        let ekind = NodeKind::ColorSlice(p);
        let tiles = |n: usize| {
            crate::widgets::TILE * n as f32 + crate::widgets::TILE_GAP * (n as f32 - 1.0)
        };
        let (w, hgt) = (tiles(TILES) - 14.0, tiles(TILE_ROWS) - 14.0 - 22.0);
        crate::uishot::render("tinter-panel-gpu", w, hgt, 2.0, 4, |ui, _| {
            let mut rcx = RowCtx {
                node: 1,
                anim: &mut anim,
                ekind: &ekind,
                t: 0,
                hud_live: false,
                driven: Vec::new(),
            };
            let _ = panel(ui, &mut rcx, &mut p, false);
        });
    }

    /// The panel must fit the tiles it asks for, both ways: a row in the
    /// scroll area cannot be clicked, and anything past the right edge is
    /// clipped.
    #[test]
    fn the_panel_fits_the_tiles_it_asks_for() {
        let mut p = ColorSliceParams { highlight: Some(3), ..Default::default() };
        let mut anim = Default::default();
        let ekind = NodeKind::ColorSlice(p);
        let (over, wide) = (std::cell::Cell::new(0.0f32), std::cell::Cell::new(0.0f32));
        let tiles = |n: usize| {
            crate::widgets::TILE * n as f32 + crate::widgets::TILE_GAP * (n as f32 - 1.0)
        };
        let (w, hgt) = (tiles(TILES) - 14.0, tiles(TILE_ROWS) - 14.0 - 22.0);
        let shot = crate::snapshot::shoot(w, hgt, |ui| {
            let out = egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    let mut rcx = RowCtx {
                        node: 1,
                        anim: &mut anim,
                        ekind: &ekind,
                        t: 0,
                        hud_live: false,
                        driven: Vec::new(),
                    };
                    let _ = panel(ui, &mut rcx, &mut p, false);
                });
            over.set(out.content_size.y - out.inner_rect.height());
            wide.set(out.content_size.x - out.inner_rect.width());
        });
        shot.save("tinter-panel");
        assert!(over.get() <= 0.5, "overflows its tile DOWN by {}pt", over.get());
        assert!(wide.get() <= 0.5, "overflows its {TILES} tiles SIDEWAYS by {}pt", wide.get());
    }
}
