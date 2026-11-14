//! Onion skins inside a 2dini.
//! Moved out of `viewer_ui` (lib.rs) unchanged.

use crate::*;

impl Gfx {
    /// Onion skins inside a 2dini.
    pub(crate) fn gizmo_geo_onion(
        &mut self,
        ui: &mut egui::Ui,
        cur_t: u64,
        hud_hide: bool,
        owf: f32,
        out_to_pts: &impl Fn(egui::Vec2) -> egui::Pos2,
        rect: egui::Rect,
    ) {
        // ONION SKINS inside a 2dini: the drawing at frames either side
        // of the playhead (animation applied), as tinted outlines, the
        // nearest strongest. Evaluated once per frame and cached on the
        // graph as it stood, so a still graph pays nothing.
        if self.dive.last().is_some_and(|d| d.kind == DiveKind::Geo) && self.prefs.geo_viz.onion && !hud_hide {
            let v = self.prefs.geo_viz;
            let step = v.onion_step.max(1) as i64;
            let frames: Vec<(i64, bool, u32)> = (1..=v.onion_before as i64)
                .map(|k| (cur_t as i64 - k * step, true, k as u32))
                .chain((1..=v.onion_after as i64).map(|k| (cur_t as i64 + k * step, false, k as u32)))
                .filter(|(f, ..)| *f >= 0)
                .collect();
            let g2s = |p: [f32; 2]| out_to_pts(egui::vec2(p[0] * owf, p[1] * owf));
            let painter = ui.painter().clone();
            // What does not move is not skinned: a still shape would only
            // draw its own outline again, in colour.
            let now = self.geo_onion(cur_t);
            // Shapes are matched by maker and order within it, so a
            // particle count that changes does not shift the rest.
            let key_of = |g: &engine::geo::Geometry| {
                let mut seen: std::collections::HashMap<u64, u32> = Default::default();
                g.prims
                    .iter()
                    .map(|p| {
                        let n = seen.entry(p.src).or_insert(0);
                        *n += 1;
                        (p.src, *n)
                    })
                    .collect::<Vec<_>>()
            };
            let now_keys: std::collections::HashMap<(u64, u32), usize> = now
                .as_ref()
                .map(|n| key_of(n).into_iter().enumerate().map(|(i, k)| (k, i)).collect())
                .unwrap_or_default();
            let same = |geo: &engine::geo::Geometry, keys: &[(u64, u32)], i: usize| {
                let Some(n) = now.as_ref() else { return false };
                let Some(&j) = keys.get(i).and_then(|k| now_keys.get(k)) else { return false };
                match (geo.prims.get(i), n.prims.get(j)) {
                    (Some(a), Some(b)) => {
                        a.verts.len() == b.verts.len()
                            && a.verts.iter().zip(&b.verts).all(|(x, y)| {
                                geo.points.get(*x as usize).map(|q| q.p) == n.points.get(*y as usize).map(|q| q.p)
                            })
                    }
                    _ => false,
                }
            };
            for (f, before, k) in frames {
                let Some(geo) = self.geo_onion(f as u64) else { continue };
                let keys = key_of(&geo);
                let col = if before { theme::onion_before() } else { theme::onion_after() };
                let a = 0.75 * v.onion_falloff.powi(k as i32 - 1);
                let c = col.gamma_multiply(a);
                for (pi, pr) in geo.prims.iter().enumerate() {
                    if same(&geo, &keys, pi) {
                        continue;
                    }
                    match pr.kind {
                        engine::geo::PrimKind::Point => {
                            if let Some(q) = pr.verts.first().and_then(|v| geo.points.get(*v as usize)) {
                                let r = (q.pscale * rect.width() / owf.max(1.0)).max(1.5);
                                painter.circle_stroke(g2s(q.p), r, egui::Stroke::new(1.0, c));
                            }
                        }
                        engine::geo::PrimKind::Poly { closed } => {
                            for ring in geo.rings(pi) {
                                let mut pts: Vec<egui::Pos2> = ring.iter().map(|p| g2s(*p)).collect();
                                if closed && pts.len() > 2 {
                                    pts.push(pts[0]);
                                }
                                painter.add(egui::Shape::line(pts, egui::Stroke::new(1.2, c)));
                            }
                        }
                    }
                }
            }
        }
    }
}
