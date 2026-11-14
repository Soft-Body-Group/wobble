//! 2dini Brush: a drag in the viewer paints a stroke into it.
//! Moved out of `viewer_ui` (lib.rs) unchanged.

use crate::*;

impl Gfx {
    /// 2dini Brush: a drag in the viewer paints a stroke into it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gizmo_geo_brush(
        &mut self,
        ui: &mut egui::Ui,
        cur_t: u64,
        hud_hide: bool,
        in_geo: bool,
        owf: f32,
        pointer_pos: Option<egui::Pos2>,
        pts_to_out: &impl Fn(egui::Pos2) -> egui::Vec2,
        rect: egui::Rect,
        geo_handle_hot: &mut bool,
    ) {
        // BRUSH: with one Brush selected, a drag in the viewer paints a
        // stroke into it, a point every couple of pixels; a fast hand
        // thins the line, a slow one fills it out.
        if in_geo && !hud_hide && self.geo.tool.is_none() {
            let brush = (self.selected.len() == 1)
                .then(|| self.selected[0])
                .filter(|id| matches!(self.egraph.node(*id).map(|n| &n.kind), Some(NodeKind::GeoNode(engine::geo::GeoNode::Brush { .. }))));
            let over_bar = pointer_pos.is_some_and(|p| self.geo.tool_bar.is_some_and(|b| b.contains(p)));
            let (pressed, down, mods) =
                ui.input(|i| (i.pointer.primary_pressed(), i.pointer.primary_down(), i.modifiers));
            if let Some(b) = brush {
                if pointer_pos.is_some_and(|p| rect.contains(p) && !over_bar) {
                    *geo_handle_hot = true; // a click paints a dab, it selects nothing
                }
                let to_g = |p: egui::Pos2| {
                    let q = pts_to_out(p);
                    [q.x / owf, q.y / owf]
                };
                if let Some(p) = pointer_pos.filter(|p| pressed && mods.is_none() && rect.contains(*p) && !over_bar && self.geo.xf_drag.is_none()) {
                    let g = to_g(p);
                    if let Some(NodeKind::GeoNode(engine::geo::GeoNode::Brush { strokes, .. })) =
                        self.egraph.node_mut(b).map(|n| &mut n.kind)
                    {
                        strokes.push(engine::geo::BrushStroke { pts: vec![[g[0], g[1], 1.0]] });
                    }
                    self.geo.brush_drag = Some((b, p));
                    self.playback_dirty = true;
                }
            }
            if let Some((b, last)) = self.geo.brush_drag {
                *geo_handle_hot = true;
                if let Some(p) = pointer_pos.filter(|p| (*p - last).length() >= 2.0) {
                    let dt = ui.input(|i| i.stable_dt).max(1e-3);
                    let speed = (p - last).length() / dt; // points a second
                    let pressure = (1.0 - speed / 4000.0).clamp(0.35, 1.0);
                    let q = pts_to_out(p);
                    if let Some(NodeKind::GeoNode(engine::geo::GeoNode::Brush { strokes, .. })) =
                        self.egraph.node_mut(b).map(|n| &mut n.kind)
                    {
                        if let Some(s) = strokes.last_mut() {
                            s.pts.push([q.x / owf, q.y / owf, pressure]);
                        }
                    }
                    self.geo.brush_drag = Some((b, p));
                    self.playback_dirty = true;
                }
                if !down {
                    self.geo.brush_drag = None;
                    // A click that never moved, on a shape some OTHER
                    // node made: that is a pick, not a dab.
                    let dab = matches!(
                        self.egraph.node(b).map(|n| &n.kind),
                        Some(NodeKind::GeoNode(engine::geo::GeoNode::Brush { strokes, .. }))
                            if strokes.last().is_some_and(|s| s.pts.len() == 1)
                    );
                    if dab {
                        let q = pts_to_out(last);
                        let at = [q.x / owf, q.y / owf];
                        if let Some(src) = self.geo_pick(at, 6.0 / owf, cur_t).filter(|s| *s != b) {
                            if let Some(NodeKind::GeoNode(engine::geo::GeoNode::Brush { strokes, .. })) =
                                self.egraph.node_mut(b).map(|n| &mut n.kind)
                            {
                                strokes.pop();
                            }
                            let g = self.live_geo();
                            self.geo_select(g.output().map_or(src, |out| geo_click_target(&g, out, src)));
                        }
                    }
                }
            }
        } else {
            self.geo.brush_drag = None;
        }
    }
}
