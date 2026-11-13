//! 2dini's tool bar and the armed tool's gesture.
//! Moved out of `viewer_ui` (lib.rs) unchanged.

use crate::*;

impl Gfx {
    /// 2dini's tool bar and the armed tool's gesture.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gizmo_geo_tools(
        &mut self,
        ui: &mut egui::Ui,
        hud_hide: bool,
        in_geo: bool,
        owf: f32,
        pointer_pos: Option<egui::Pos2>,
        out_to_pts: &impl Fn(egui::Vec2) -> egui::Pos2,
        pts_to_out: &impl Fn(egui::Pos2) -> egui::Vec2,
        rect: egui::Rect,
        geo_handle_hot: &mut bool,
    ) {
        if in_geo && !hud_hide {
            let mut clicked: Option<GeoTool> = None;
            let armed = self.geo.tool;
            let swatch = self.geo_style_target().map(|(_, st)| (st.fill, st.stroke)).unwrap_or(([1.0; 4], [0.0; 4]));
            let mut spots = Vec::new();
            let resp = egui::Area::new(egui::Id::new("geo-tool-bar"))
                .order(egui::Order::Foreground)
                .fixed_pos(rect.left_top() + egui::vec2(22.0, 36.0))
                .show(ui.ctx(), |ui| {
                    egui::Frame::popup(ui.style()).inner_margin(3.0).show(ui, |ui| {
                        ui.spacing_mut().item_spacing = egui::vec2(0.0, 2.0);
                        // Fill and stroke of the selected shape, the way a
                        // vector app keeps them at the foot of its tools.
                        // A click restyles through a Style node.
                        let (fill, stroke) = swatch;
                        for (k, c) in [(0u8, fill), (1, stroke)] {
                            let (r, resp) = ui.allocate_exact_size(egui::vec2(30.0, 26.0), egui::Sense::click());
                            let c32 = egui::Color32::from_rgba_unmultiplied(
                                (c[0].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8,
                                (c[1].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8,
                                (c[2].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8,
                                (c[3].clamp(0.0, 1.0) * 255.0) as u8,
                            );
                            let sq = egui::Rect::from_center_size(r.center(), egui::vec2(18.0, 18.0));
                            let edge = if resp.hovered() { theme::selection() } else { theme::text() };
                            if k == 0 {
                                ui.painter().rect_filled(sq, 2.0, c32);
                                ui.painter().rect_stroke(sq, 2.0, egui::Stroke::new(1.0, edge), egui::StrokeKind::Outside);
                            } else {
                                // A stroke swatch is a frame: the colour as a thick ring.
                                ui.painter().rect_stroke(sq.shrink(2.5), 1.0, egui::Stroke::new(5.0, c32), egui::StrokeKind::Middle);
                                ui.painter().rect_stroke(sq, 2.0, egui::Stroke::new(1.0, edge), egui::StrokeKind::Outside);
                            }
                            if c[3] <= 0.0 {
                                // None: the usual red slash.
                                ui.painter().line_segment([sq.left_bottom(), sq.right_top()], egui::Stroke::new(1.5, theme::cost_slow()));
                            }
                            spots.push((if k == 0 { GeoTool::Fill } else { GeoTool::StrokeColour }, r));
                            if resp.on_hover_text(if k == 0 { "fill: restyle the selection (a Style node, its panel opened)" } else { "stroke: restyle the selection (a Style node, its panel opened)" }).clicked() {
                                clicked = Some(if k == 0 { GeoTool::Fill } else { GeoTool::StrokeColour });
                            }
                        }
                        ui.separator();
                        for t in GeoTool::ALL {
                            if t == GeoTool::Circle {
                                ui.separator();
                            }
                            let (r, resp) = ui.allocate_exact_size(egui::vec2(30.0, 26.0), egui::Sense::click());
                            let on = armed == Some(t);
                            let vis = ui.style().interact_selectable(&resp, on);
                            if on || resp.hovered() {
                                ui.painter().rect_filled(r, 3.0, vis.weak_bg_fill);
                            }
                            geo_tool_icon(ui.painter(), r.shrink(6.0), t, vis.fg_stroke.color);
                            spots.push((t, r));
                            if resp.on_hover_text(t.tip()).clicked() {
                                clicked = Some(t);
                            }
                        }
                    });
                });
            self.geo.tool_bar = Some(resp.response.rect);
            self.geo.tool_spots = spots;
            match clicked {
                Some(GeoTool::Transform) => self.geo_tool_transform(),
                Some(GeoTool::Direct) => self.geo_tool_edit(),
                Some(GeoTool::Fill | GeoTool::StrokeColour) => self.geo_tool_style(),
                Some(t @ (GeoTool::Line | GeoTool::Spline | GeoTool::Brush)) => {
                    self.geo.tool = None;
                    let n = self.geo_tool_add(t.make());
                    self.geo_select(n);
                }
                Some(t) => self.geo.tool = if armed == Some(t) { None } else { Some(t) },
                None => {}
            }
            if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                self.geo.tool = None;
            }
            let over_bar = pointer_pos.is_some_and(|p| self.geo.tool_bar.is_some_and(|b| b.contains(p)));
            let s2g = |p: egui::Pos2| {
                let q = pts_to_out(p);
                [q.x / owf, q.y / owf]
            };
            let (pressed, down) = ui.input(|i| (i.pointer.primary_pressed(), i.pointer.primary_down()));
            let free = ui.input(|i| i.modifiers.ctrl);
            let px_scr = (out_to_pts(egui::vec2(owf, 0.0)) - out_to_pts(egui::vec2(0.0, 0.0))).x; // screen points per geo unit
            if let Some(t) = self.geo.tool.filter(|_| self.geo.tool_drag.is_none()) {
                *geo_handle_hot = true; // the press is the tool's
                if let Some(p) = pointer_pos.filter(|p| pressed && rect.contains(*p) && !over_bar) {
                    let at = self.geo_snap(s2g(p), owf, px_scr, None, free).0;
                    let n = self.geo_tool_add(geo_tool_place(t.make(), at, at));
                    self.geo_select(n);
                    self.geo.tool_drag = Some((n, at));
                }
            }
            if let Some((n, at)) = self.geo.tool_drag {
                *geo_handle_hot = true;
                if let (Some(p), Some(t)) = (pointer_pos, self.geo.tool) {
                    let to = self.geo_snap(s2g(p), owf, px_scr, Some(n), free).0;
                    let moved = ((to[0] - at[0]).powi(2) + (to[1] - at[1]).powi(2)).sqrt() > 3.0 / owf;
                    let k = if moved { geo_tool_place(t.make(), at, to) } else { geo_tool_place(t.make(), at, at) };
                    if let Some(node) = self.egraph.node_mut(n) {
                        node.kind = NodeKind::GeoNode(k);
                    }
                    self.playback_dirty = true;
                }
                if !down {
                    self.geo.tool_drag = None;
                    self.geo.tool = None;
                }
            }
        }
    }
}
