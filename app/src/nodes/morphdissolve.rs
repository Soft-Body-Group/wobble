//! Morph Dissolve's app half: the panel, the channels and the glyph. The
//! params and the WGSL are `engine::nodes::morphdissolve`; the analysis runs
//! through the Flow runner (`crate::flowui`), and the weights gate sits above
//! this panel, drawn for every node that declares models.

use crate::anim::Channel;
use crate::flowui::{Action, View};
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke, Vec2};
use engine::graph::{FlowDetail, FlowEngine, MorphDissolveParams};

pub const CHANNELS: &[Channel] = &[
    Channel::MorphMix,
    Channel::MorphWarp,
    Channel::MorphLead,
    Channel::MorphWidth,
    Channel::MorphOcclusion,
];

pub const TILES: usize = 2;
pub const TILE_ROWS: usize = 2;

pub fn get(p: &MorphDissolveParams, ch: Channel) -> Option<f32> {
    match ch {
        Channel::MorphMix => Some(p.mix),
        Channel::MorphWarp => Some(p.warp),
        Channel::MorphLead => Some(p.lead),
        Channel::MorphWidth => Some(p.width),
        Channel::MorphOcclusion => Some(p.occlusion),
        _ => None,
    }
}

pub fn set(p: &mut MorphDissolveParams, ch: Channel, v: f32) {
    match ch {
        Channel::MorphMix => p.mix = v,
        Channel::MorphWarp => p.warp = v,
        Channel::MorphLead => p.lead = v,
        Channel::MorphWidth => p.width = v,
        Channel::MorphOcclusion => p.occlusion = v,
        _ => {}
    }
}

pub fn label(ch: Channel) -> Option<&'static str> {
    match ch {
        Channel::MorphMix => Some("morph mix"),
        Channel::MorphWarp => Some("morph warp"),
        Channel::MorphLead => Some("morph dissolve lead"),
        Channel::MorphWidth => Some("morph dissolve width"),
        Channel::MorphOcclusion => Some("morph occlusion"),
        _ => None,
    }
}

/// The properties panel. `span` is the frames an analysis would solve; the
/// click it hands back is acted on after the node borrow ends.
pub fn panel(
    ui: &mut egui::Ui,
    rcx: &mut RowCtx,
    p: &mut MorphDissolveParams,
    v: &View,
    span: Option<(u64, u64)>,
) -> (Option<HudSlider>, Option<Action>) {
    let mut h = widgets::slider(ui, rcx, "mix", &mut p.mix, 0.0, 1.0, &[Channel::MorphMix]);
    h = h.or(widgets::slider(ui, rcx, "warp", &mut p.warp, 0.0, 1.0, &[Channel::MorphWarp]));
    h = h.or(widgets::slider(ui, rcx, "lead", &mut p.lead, -1.0, 1.0, &[Channel::MorphLead]));
    h = h.or(widgets::slider(ui, rcx, "width", &mut p.width, 0.01, 1.0, &[Channel::MorphWidth]));
    h = h.or(widgets::slider(ui, rcx, "occlusion", &mut p.occlusion, 0.0, 1.0, &[Channel::MorphOcclusion]));
    let node = rcx.node;
    ui.horizontal(|ui| {
        ui.label("engine");
        egui::ComboBox::from_id_salt(("morph-engine", node))
            .width(90.0)
            .selected_text(p.engine.label())
            .show_ui(ui, |ui| {
                for e in FlowEngine::ALL {
                    ui.selectable_value(&mut p.engine, e, e.label());
                }
            });
        ui.label("detail");
        egui::ComboBox::from_id_salt(("morph-detail", node))
            .width(90.0)
            .selected_text(p.detail.label())
            .show_ui(ui, |ui| {
                for d in FlowDetail::ALL {
                    ui.selectable_value(&mut p.detail, d, d.label());
                }
            });
    });
    match span {
        Some((from, len)) => {
            let n = len.saturating_sub(from);
            ui.weak(format!("analysis solves {n} frame{}, twice each · then it plays like a Read", if n == 1 { "" } else { "s" }));
        }
        None => {
            ui.weak("wire A and B, with frames in common");
        }
    }
    let act = crate::nodes::flow::run_rows(
        ui,
        v,
        "not analysed: a plain dissolve",
        "renders A and B over the frames where mix is between 0 and 1 (all their shared frames if it is not keyed), solves the motion both ways at each one, and plays the vectors here. Changing mix or the sliders above never solves again",
    );
    (h, act)
}

/// Two overlapping discs, an arrow through them: one picture becoming
/// another along its motion.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c - Vec2::new(r * 0.32, 0.0), r * 0.62, fill, stroke);
    painter.circle(c + Vec2::new(r * 0.32, 0.0), r * 0.62, fill, stroke);
    let ink = Stroke::new((r * 0.12).max(1.0), crate::theme::node_ink());
    let (a, b) = (c - Vec2::new(r * 0.5, 0.0), c + Vec2::new(r * 0.5, 0.0));
    painter.line_segment([a, b], ink);
    painter.line_segment([b, b + Vec2::new(-r * 0.2, -r * 0.18)], ink);
    painter.line_segment([b, b + Vec2::new(-r * 0.2, r * 0.18)], ink);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowui::ViewState;
    use engine::graph::NodeKind;

    #[test]
    fn the_panel_fits_the_tiles_it_asks_for() {
        let span = |n: usize| crate::widgets::TILE * n as f32 + crate::widgets::TILE_GAP * (n as f32 - 1.0);
        let (w, hgt) = (span(TILES) - 14.0, span(TILE_ROWS) - 14.0 - 22.0);
        let v = View {
            state: ViewState::Solving(7, 24),
            has_input: true,
            weights: true,
            loaded: Some((24, false)),
            on_disk: false,
            busy_elsewhere: false,
        };
        let mut p = MorphDissolveParams::new(0);
        let kind = NodeKind::MorphDissolve(p.clone());
        let mut anim = Default::default();
        let over = std::cell::Cell::new(0.0f32);
        let wide = std::cell::Cell::new(0.0f32);
        let shot = crate::snapshot::shoot(w, hgt, |ui| {
            let mut rcx = RowCtx { node: 1, anim: &mut anim, ekind: &kind, t: 0, hud_live: false, driven: Vec::new() };
            let out = egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                let _ = panel(ui, &mut rcx, &mut p, &v, Some((10, 34)));
            });
            over.set(out.content_size.y - out.inner_rect.height());
            wide.set(out.content_size.x - out.inner_rect.width());
        });
        shot.save("morphdissolve-panel");
        assert!(over.get() <= 0.5, "overflows its tiles by {}pt", over.get());
        assert!(wide.get() <= 0.5, "overflows sideways by {}pt", wide.get());
    }
}
