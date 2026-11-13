//! 2dini's display bar: one toggle per overlay of the vector HUD.
//! Moved out of `viewer_ui` (lib.rs) unchanged.

use crate::*;

impl Gfx {
    /// 2dini's display bar: one toggle per overlay of the vector HUD.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gizmo_geo_viz_bar(
        &mut self,
        ui: &mut egui::Ui,
        geo_in_play: bool,
        hud_hide: bool,
        owf: f32,
        pointer_pos: Option<egui::Pos2>,
        out_to_pts: &impl Fn(egui::Vec2) -> egui::Pos2,
        rect: egui::Rect,
    ) {
        if geo_in_play && !hud_hide {
            let mut viz = self.prefs.geo_viz;
            let mut hud_on = self.prefs.geo_hud.unwrap_or(true);
            let before = (viz, hud_on);
            let resp = egui::Area::new(egui::Id::new("geo-viz-bar"))
                .order(egui::Order::Foreground)
                .fixed_pos(rect.right_top() + egui::vec2(-46.0, 36.0))
                .show(ui.ctx(), |ui| {
                    egui::Frame::popup(ui.style()).inner_margin(3.0).show(ui, |ui| {
                        // One button wide, whatever is in it (the onion
                        // fields would otherwise let the bar sprawl).
                        ui.set_max_width(34.0);
                        ui.spacing_mut().item_spacing = egui::vec2(0.0, 2.0);
                        let tog = |ui: &mut egui::Ui, on: &mut bool, label: &str, tip: &str| {
                            if ui
                                .add_sized([34.0, 18.0], egui::Button::selectable(*on, egui::RichText::new(label).small()))
                                .on_hover_text(tip)
                                .clicked()
                            {
                                *on = !*on;
                            }
                        };
                        tog(ui, &mut hud_on, "HUD", "the vector HUD (H)");
                        ui.separator();
                        tog(ui, &mut viz.points, "pts", "display points");
                        tog(ui, &mut viz.point_nums, "#pt", "display point numbers");
                        tog(ui, &mut viz.prim_nums, "#pr", "display primitive numbers");
                        tog(ui, &mut viz.normals, "N", "display normals");
                        tog(ui, &mut viz.ticks, "orient", "display point orientation");
                        tog(ui, &mut viz.outlines, "wire", "display outlines");
                        tog(ui, &mut viz.bbox, "bbox", "display the bounding box");
                        ui.separator();
                        tog(ui, &mut viz.rulers, "rule", "rulers along the image, in output pixels");
                        tog(ui, &mut viz.snap, "snap", "handles and drawing tools snap to shape centres and the ruler ticks (ctrl held: free)");
                        ui.separator();
                        tog(ui, &mut viz.onion, "onion", "onion skins: the drawing a few frames before (red) and after (green) the playhead");
                        if viz.onion {
                            // Before, after, step, falloff: four small fields.
                            let num = |ui: &mut egui::Ui, v: &mut u32, tip: &str| {
                                ui.add_sized([34.0, 16.0], egui::DragValue::new(v).range(0..=12).speed(0.1)).on_hover_text(tip);
                            };
                            num(ui, &mut viz.onion_before, "frames before");
                            num(ui, &mut viz.onion_after, "frames after");
                            ui.add_sized([34.0, 16.0], egui::DragValue::new(&mut viz.onion_step).range(1..=24).speed(0.1).prefix("x"))
                                .on_hover_text("frames between skins");
                            ui.add_sized([34.0, 16.0], egui::DragValue::new(&mut viz.onion_falloff).range(0.05..=1.0).speed(0.01))
                                .on_hover_text("how much fainter each skin further out is");
                        }
                    });
                });
            self.geo_viz_bar = Some(resp.response.rect);
            // RULERS along the image's top and left, in output pixels,
            // the pointer marked on both.
            if viz.rulers {
                let o = out_to_pts(egui::vec2(0.0, 0.0));
                let scr = (out_to_pts(egui::vec2(owf, 0.0)) - o).x;
                let scr_px = scr / owf.max(1.0);
                let (minor, major) = ruler_steps(scr_px);
                const T: f32 = 16.0;
                let pt = ui.painter().clone().with_layer_id(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("geo-rulers")));
                let top = egui::Rect::from_min_max(rect.min, egui::pos2(rect.max.x, rect.min.y + T));
                let left = egui::Rect::from_min_max(egui::pos2(rect.min.x, rect.min.y + T), egui::pos2(rect.min.x + T, rect.max.y));
                let bg = theme::hud_bg().gamma_multiply_u8(220);
                pt.rect_filled(top, 0.0, bg);
                pt.rect_filled(left, 0.0, bg);
                let ink = theme::text();
                let font = egui::FontId::proportional(9.0);
                let ticks = |lo: f32, hi: f32, origin: f32| {
                    let k0 = ((lo - origin) / (minor * scr_px)).floor() as i64;
                    let k1 = ((hi - origin) / (minor * scr_px)).ceil() as i64;
                    (k0..=k1.min(k0 + 4000)).map(move |k| (k, origin + k as f32 * minor * scr_px))
                };
                for (k, x) in ticks(top.left() + T, top.right(), o.x) {
                    let v = k as f32 * minor;
                    let big = (v / major).fract().abs() < 1e-4;
                    let h = if big { T * 0.6 } else { T * 0.25 };
                    pt.line_segment([egui::pos2(x, top.bottom() - h), egui::pos2(x, top.bottom())], egui::Stroke::new(1.0, ink));
                    if big {
                        pt.text(egui::pos2(x + 2.0, top.top() + 1.0), egui::Align2::LEFT_TOP, format!("{v:.0}"), font.clone(), ink);
                    }
                }
                for (k, y) in ticks(left.top(), left.bottom(), o.y) {
                    let v = k as f32 * minor;
                    let big = (v / major).fract().abs() < 1e-4;
                    let w = if big { T * 0.6 } else { T * 0.25 };
                    pt.line_segment([egui::pos2(left.right() - w, y), egui::pos2(left.right(), y)], egui::Stroke::new(1.0, ink));
                    if big {
                        pt.text(egui::pos2(left.left() + 1.0, y + 2.0), egui::Align2::LEFT_TOP, format!("{v:.0}"), font.clone(), ink);
                    }
                }
                if let Some(p) = pointer_pos.filter(|p| rect.contains(*p)) {
                    let hot = theme::selection();
                    pt.line_segment([egui::pos2(p.x, top.top()), egui::pos2(p.x, top.bottom())], egui::Stroke::new(1.0, hot));
                    pt.line_segment([egui::pos2(left.left(), p.y), egui::pos2(left.right(), p.y)], egui::Stroke::new(1.0, hot));
                }
            }
            if (viz, hud_on) != before {
                self.prefs.geo_viz = viz;
                self.prefs.geo_hud = Some(hud_on);
                save_prefs(&self.prefs);
            }
        }
    }
}
