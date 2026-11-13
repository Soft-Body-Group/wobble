//! The vector HUD of every 2dini in play: points, handles, rulers.
//! Moved out of `viewer_ui` (lib.rs) unchanged.

use crate::*;

impl Gfx {
    /// The vector HUD of every 2dini in play: points, handles, rulers.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gizmo_geo_hud(
        &mut self,
        ui: &mut egui::Ui,
        cur_t: u64,
        hud_hide: bool,
        owf: f32,
        ohf: f32,
        out_to_pts: &impl Fn(egui::Vec2) -> egui::Pos2,
        rect: egui::Rect,
        selected: &Vec<u64>,
        ink: &mut ink::InkShapes,
    ) {
        let viz = self.prefs.geo_viz;
        if self.prefs.geo_hud.unwrap_or(true) && !hud_hide {
            let px = rect.width() / owf;
            self.geo_hud_refresh(&selected, owf, ohf, cur_t);
            for (owner, geo, hd) in self.geo_hud_items(&selected, owf, ohf, cur_t) {
                ink.set_tint(match owner {
                    Some(n) => self.hud_ink(n),
                    None => self.dive_hud_ink(),
                });
                // Geo space is field space: y in WIDTH units.
                let map = |p: [f32; 2]| -> egui::Pos2 {
                    let v = egui::vec2(p[0] * owf, p[1] * owf);
                    match &hd {
                        Some(h) => out_to_pts(apply_h(h, v)),
                        None => out_to_pts(v),
                    }
                };
                // How much the downstream chain scales lengths, for the
                // rings: the linear part's area root.
                let ring_scale = hd
                    .map(|h| (h[0][0] * h[1][1] - h[0][1] * h[1][0]).abs().sqrt())
                    .unwrap_or(1.0)
                    * px;
                for pr in &geo.prims {
                    match pr.kind {
                        engine::geo::PrimKind::Point => {
                            let Some(q) =
                                pr.verts.first().and_then(|&v| geo.points.get(v as usize))
                            else {
                                continue;
                            };
                            ink.circle_stroke(
                                map(q.p),
                                (q.pscale * ring_scale).max(2.0),
                                0.9,
                                0.7,
                                false,
                            );
                        }
                        engine::geo::PrimKind::Poly { .. } if !viz.outlines => {}
                        engine::geo::PrimKind::Poly { closed } => {
                            // One path per outline. Plain ink for a
                            // closed shape; the red lane — the HUD's
                            // warning colour — for one that did not
                            // close, so it reads at a glance.
                            // One path per ring: a glyph's counters
                            // are outlines of their own.
                            let n = pr.verts.len();
                            let mut cuts: Vec<usize> =
                                pr.rings.iter().map(|&r| r as usize).filter(|&r| r > 0 && r < n).collect();
                            cuts.sort_unstable();
                            let mut start = 0;
                            for end in cuts.into_iter().chain(std::iter::once(n)) {
                                let pts: Vec<egui::Pos2> = pr.verts[start..end]
                                    .iter()
                                    .filter_map(|&v| geo.points.get(v as usize))
                                    .map(|q| map(q.p))
                                    .collect();
                                ink.path(pts, closed, 1.0, 0.8, if closed { None } else { Some(0) });
                                start = end;
                            }
                        }
                    }
                }
                // Outline normals: out of every closed shape, one side of
                // an open line; blue so they read apart from the ticks.
                if viz.normals {
                    let len = 12.0;
                    let mut segs = Vec::new();
                    for i in 0..geo.prims.len() {
                        for (q, n) in geo.polyline(i).iter().zip(geo.normals(i)) {
                            let a = map(*q);
                            let b = map([q[0] + n[0] * 1e-3, q[1] + n[1] * 1e-3]) - a;
                            if b.length_sq() > 1e-9 {
                                segs.push((a, a + b.normalized() * len));
                            }
                        }
                    }
                    ink.segs_lane(segs.into_iter(), 1.0, 0.8, 2);
                }
                // Every point as a dot with a tick along its orientation
                // — the attributes an Instance reads — as ONE mesh. The
                // tick is projected like the point, so a downstream
                // rotation turns both.
                if viz.points || viz.ticks {
                    ink.dots(
                        geo.points.iter().map(|q| {
                            let c = map(q.p);
                            let (s, co) = q.orient.sin_cos();
                            let e = map([q.p[0] + co * 1e-3, q.p[1] + s * 1e-3]) - c;
                            let len = 5.0 + 2.0 * q.pscale.clamp(0.0, 4.0);
                            let tick = (viz.ticks && e.length_sq() > 1e-9).then(|| c + e.normalized() * len);
                            (c, tick)
                        }),
                        if viz.points { 1.6 } else { 0.0 },
                        0.85,
                        0.8,
                    );
                }
                if viz.bbox {
                    if let Some((lo, hi)) = geo.bbox() {
                        let corners = [map(lo), map([hi[0], lo[1]]), map(hi), map([lo[0], hi[1]])];
                        ink.path(corners.to_vec(), true, 1.0, 0.6, Some(1));
                    }
                }
                // Numbers, as Houdini draws them: point indices beside
                // the points, shape indices at their centres. Only the
                // ones on screen, and not thousands of them.
                if viz.point_nums || viz.prim_nums {
                    let painter = ui.painter_at(rect);
                    let font = egui::FontId::monospace(10.0);
                    if viz.point_nums {
                        let mut n = 0;
                        for (i, q) in geo.points.iter().enumerate() {
                            let c = map(q.p);
                            if !rect.contains(c) {
                                continue;
                            }
                            painter.text(c + egui::vec2(4.0, -3.0), egui::Align2::LEFT_BOTTOM, i.to_string(), font.clone(), egui::Color32::from_rgb(0xff, 0xd8, 0x6a));
                            n += 1;
                            if n >= 2000 {
                                break;
                            }
                        }
                    }
                    if viz.prim_nums {
                        for i in 0..geo.prims.len().min(2000) {
                            let c = map(geo.centroid(i));
                            if rect.contains(c) {
                                painter.text(c, egui::Align2::CENTER_CENTER, i.to_string(), font.clone(), egui::Color32::from_rgb(0x7f, 0xe0, 0xff));
                            }
                        }
                    }
                }
            }
        }
    }
}
