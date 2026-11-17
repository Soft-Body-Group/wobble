//! Uncrop's app half: the Crop link, the panel, the glyph, and the two
//! places the link is honoured — [`lower`] (the linked Crop's box, as it is
//! at that moment, copied into the engine params) and [`format_from`] (the
//! full format when no reference is wired). The model and the WGSL are
//! `engine::nodes::uncrop`. No channels: the box is normally the Crop's,
//! animated there.

use crate::editor::EditorGraph;
use crate::widgets::{HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke};
use engine::graph::{NodeKind, UncropParams};

pub const TILES: usize = 1;

/// The Crop nodes an Uncrop can link to: (id, shown name).
pub fn crops(g: &EditorGraph) -> Vec<(u64, String)> {
    g.nodes
        .iter()
        .filter(|(_, n)| matches!(n.kind, NodeKind::SoftCrop(_)))
        .map(|(id, n)| (*id, if n.ui_label.is_empty() { format!("Crop {id}") } else { n.ui_label.clone() }))
        .collect()
}

/// The linked Crop's box at tick `t` (its own animation applied), if the
/// link still names a Crop.
pub fn linked_rect(g: &EditorGraph, p: &UncropParams, t: Option<f32>) -> Option<[f32; 4]> {
    let n = g.node(p.crop?)?;
    let mut k = n.kind.clone();
    if let Some(t) = t {
        crate::anim::apply_at(&mut k, &n.anim, t);
    }
    match k {
        NodeKind::SoftCrop(c) => Some(c.rect),
        _ => None,
    }
}

/// Lowering: a linked Uncrop takes its Crop's box.
pub fn lower(g: &EditorGraph, p: &mut UncropParams, t: Option<f32>) {
    if let Some(r) = linked_rect(g, p, t) {
        p.rect = r;
    }
}

/// The node whose format is the full one when the reference port is empty:
/// the linked Crop's input.
pub fn format_from(g: &EditorGraph, p: &UncropParams) -> Option<u64> {
    let c = p.crop?;
    matches!(g.node(c).map(|n| &n.kind), Some(NodeKind::SoftCrop(_))).then_some(c)
}

pub fn panel(
    ui: &mut egui::Ui,
    rcx: &mut RowCtx,
    p: &mut UncropParams,
    crops: &[(u64, String)],
    linked: Option<[f32; 4]>,
) -> Option<HudSlider> {
    let shown = match p.crop {
        None => "none (own box)".to_string(),
        Some(id) => crops
            .iter()
            .find(|(c, _)| *c == id)
            .map(|(_, s)| s.clone())
            .unwrap_or_else(|| "missing".into()),
    };
    ui.horizontal(|ui| {
        ui.label("crop");
        egui::ComboBox::from_id_salt(("uncrop-link", rcx.node))
            .width(150.0)
            .selected_text(shown)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut p.crop, None, "none (own box)");
                for (id, name) in crops {
                    ui.selectable_value(&mut p.crop, Some(*id), name);
                }
            });
    });
    match linked {
        Some(r) => {
            ui.weak(format!(
                "box follows the Crop: {:.3} {:.3} - {:.3} {:.3}",
                r[0], r[1], r[2], r[3]
            ));
        }
        None => {
            if p.crop.is_some() {
                ui.weak("the linked Crop is gone: using the box below");
            }
            ui.horizontal(|ui| {
                for (i, l) in ["l", "t", "r", "b"].iter().enumerate() {
                    ui.label(*l);
                    ui.add(egui::DragValue::new(&mut p.rect[i]).speed(0.001).range(0.0..=1.0).max_decimals(3));
                }
            });
        }
    }
    ui.weak("full format: the f port, else the Crop's input");
    None
}

/// The graph glyph: a small box with its corners pushed out — a Crop's
/// brackets turned inside out.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.rect(
        egui::Rect::from_center_size(c, egui::vec2(r * 1.7, r * 1.7)),
        r * 0.3,
        fill,
        stroke,
        egui::StrokeKind::Middle,
    );
    let ink = Stroke::new((r * 0.14).max(1.0), crate::theme::node_ink());
    let (h, k) = (r * 0.25, r * 0.3);
    painter.rect_stroke(
        egui::Rect::from_center_size(c, egui::vec2(h * 2.0, h * 2.0)),
        0.0,
        ink,
        egui::StrokeKind::Middle,
    );
    for (sx, sy) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
        let p = c + egui::vec2(sx * h, sy * h);
        painter.line_segment([p, p + egui::vec2(sx * k, sy * k)], ink);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::graph::SoftCropParams;

    /// A linked Uncrop reaches the engine with its Crop's box, and follows
    /// it when the Crop moves; unlinked it keeps its own.
    #[test]
    fn a_linked_uncrop_lowers_with_the_crops_box() {
        let mut g = EditorGraph::default();
        let rect = [0.1, 0.2, 0.6, 0.7];
        let crop = g.add(NodeKind::SoftCrop(SoftCropParams { rect, ..Default::default() }), egui::pos2(0.0, 0.0));
        let un = g.add(NodeKind::Uncrop(UncropParams { crop: Some(crop), ..Default::default() }), egui::pos2(0.0, 80.0));
        let box_of = |g: &EditorGraph| match g.to_engine_graph_at(Some(0)).nodes.get(&un) {
            Some(NodeKind::Uncrop(p)) => p.rect,
            k => panic!("lowered to {k:?}"),
        };
        assert_eq!(box_of(&g), rect);
        if let Some(NodeKind::SoftCrop(c)) = g.node_mut(crop).map(|n| &mut n.kind) {
            c.rect = [0.3, 0.3, 0.9, 0.9];
        }
        assert_eq!(box_of(&g), [0.3, 0.3, 0.9, 0.9]);
        assert_eq!(format_from(&g, &UncropParams { crop: Some(crop), ..Default::default() }), Some(crop));
        if let Some(NodeKind::Uncrop(p)) = g.node_mut(un).map(|n| &mut n.kind) {
            p.crop = None;
        }
        assert_eq!(box_of(&g), [0.0, 0.0, 1.0, 1.0]);
    }
}
