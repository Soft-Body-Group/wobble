//! `--test-gridwarp`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-cornerpin: a marquee picks two pins, E held scales them about
    /// the press, R held turns them a quarter round it, and with `from`
    /// armed a pin drag moves the FROM quad and leaves the pin alone. Every
    /// expectation is anchored to the press, like the spline suite's.
    /// --test-gridwarp driver, through the real viewer input: a destination
    /// point dragged lands where the pointer went and leaves its source
    /// alone; in source mode the source moves instead; a box on the
    /// background picks the points it covers and a drag moves the pick as
    /// one; and on an ANIMATED point, a drag at a frame between two keys
    /// keys that frame at the curve's value there plus the travel (the
    /// gizmo rule: absolute from the press, never summed). Wall-clock
    /// deadline.
    pub(crate) fn drive_test_gridwarp(&mut self, raw: &mut egui::RawInput) {
        use nodes::gridwarp::Edit;
        let Some((stage, id, press, deadline)) = self.test.gridwarp else { return };
        let fail = |m: String| -> ! {
            println!("GRIDWARP TEST FAIL: {m}");
            std::process::exit(1);
        };
        if Instant::now() > deadline {
            fail(format!("timeout at stage {stage}"));
        }
        let click = |raw: &mut egui::RawInput, pos: egui::Pos2, down: bool| {
            raw.events.push(egui::Event::PointerMoved(pos));
            raw.events.push(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: down,
                modifiers: Default::default(),
            });
        };
        let at = |s: &Self, f: [f32; 2]| {
            let r = s.test.rect;
            egui::pos2(r.min.x + r.width() * f[0], r.min.y + r.height() * f[1])
        };
        let gw = |s: &Self| match s.egraph.node(id).map(|n| n.kind_at(s.eval_tick())) {
            Some(NodeKind::GridWarp(g)) => g,
            _ => Default::default(),
        };
        let per = |s: &Self, d: egui::Vec2| [d.x / s.test.rect.width(), d.y / s.test.rect.height()];
        let near = |a: [f32; 2], b: [f32; 2]| (a[0] - b[0]).abs() < 2e-3 && (a[1] - b[1]).abs() < 2e-3;
        let ctx = self.egui_ctx.clone();
        // The default lattice is 4 x 4 cells: point (i, j) is 5 j + i.
        const MID: usize = 12;
        let (d1, d2, d3) = (egui::vec2(60.0, 24.0), egui::vec2(-20.0, 30.0), egui::vec2(40.0, 0.0));
        let mut st = (stage + 1, id, press, deadline);
        match stage {
            0 => {
                if !self.layout_settled() || self.playback.is_none() {
                    st.0 = 0;
                } else {
                    self.test.rect = self.image_rect;
                    let n = self.egraph.add(NodeKind::GridWarp(Default::default()), egui::Pos2::new(240.0, 0.0));
                    self.selected = vec![n];
                    self.playing = false;
                    st.1 = n;
                }
            }
            1 => {}
            // A destination point dragged.
            2 => {
                st.2 = at(self, [0.5, 0.5]);
                click(raw, st.2, true);
            }
            3 | 4 => raw.events.push(egui::Event::PointerMoved(press + d1 * (stage - 2) as f32 * 0.5)),
            5 => click(raw, press + d1, false),
            6 => {}
            7 => {
                let g = gw(self);
                let want = [0.5 + per(self, d1)[0], 0.5 + per(self, d1)[1]];
                if !near(g.dst[MID], want) {
                    fail(format!("the dragged destination point is at {:?}, wanted {want:?}", g.dst[MID]));
                }
                if g.src[MID] != [0.5, 0.5] {
                    fail(format!("dragging a destination point moved its source to {:?}", g.src[MID]));
                }
                let pick = nodes::gridwarp::picked(&ctx, id, g.count());
                if pick.iter().enumerate().any(|(k, b)| *b != (k == MID)) {
                    fail("the dragged point is not the whole pick".into());
                }
                nodes::gridwarp::set_edit(&ctx, id, Edit::Src);
            }
            // Source mode: the source moves, the destination stays.
            8 => {}
            9 => {
                st.2 = at(self, [0.25, 0.25]);
                click(raw, st.2, true);
            }
            10 | 11 => raw.events.push(egui::Event::PointerMoved(press + d2 * (stage - 9) as f32 * 0.5)),
            12 => click(raw, press + d2, false),
            13 => {}
            14 => {
                let g = gw(self);
                let want = [0.25 + per(self, d2)[0], 0.25 + per(self, d2)[1]];
                if !near(g.src[6], want) || g.dst[6] != [0.25, 0.25] {
                    fail(format!("in source mode point 6 went src {:?} dst {:?}, wanted src {want:?}", g.src[6], g.dst[6]));
                }
                nodes::gridwarp::set_edit(&ctx, id, Edit::Dst);
            }
            // A box over the second row's first two inner points.
            15 => {}
            16 => click(raw, at(self, [0.12, 0.12]), true),
            17 => raw.events.push(egui::Event::PointerMoved(at(self, [0.35, 0.2]))),
            18 => raw.events.push(egui::Event::PointerMoved(at(self, [0.6, 0.32]))),
            19 => click(raw, at(self, [0.6, 0.32]), false),
            20 => {}
            21 => {
                let g = gw(self);
                let pick = nodes::gridwarp::picked(&ctx, id, g.count());
                let got: Vec<usize> = (0..pick.len()).filter(|k| pick[*k]).collect();
                if got != [6, 7] {
                    fail(format!("the box picked {got:?}, wanted [6, 7]"));
                }
                if g.dst[MID] == [0.5, 0.5] || g.dst[6] != [0.25, 0.25] {
                    fail("the box moved a point".into());
                }
            }
            // ...and a drag on one of them moves both.
            22 => {
                st.2 = at(self, [0.5, 0.25]);
                click(raw, st.2, true);
            }
            23 | 24 => raw.events.push(egui::Event::PointerMoved(press + d3 * (stage - 22) as f32 * 0.5)),
            25 => click(raw, press + d3, false),
            26 => {}
            27 => {
                let g = gw(self);
                let dx = per(self, d3)[0];
                if !near(g.dst[6], [0.25 + dx, 0.25]) || !near(g.dst[7], [0.5 + dx, 0.25]) || !near(g.dst[8], [0.75, 0.25]) {
                    fail(format!("the pick drag left {:?} {:?} {:?}", g.dst[6], g.dst[7], g.dst[8]));
                }
                // The box: four points round (2, 2), untouched so far.
                nodes::gridwarp::set_pick(&ctx, id, g.count(), &[11, 13, 16, 18]);
            }
            28 | 29 => {}
            // A corner handle scales about the opposite corner.
            30 => {
                let Some(hot) = nodes::gridwarp::box_hot(&ctx, id) else { fail("four picked points drew no box".into()) };
                st.2 = hot.scale[4];
                self.test.gridwarp_snap = gw(self).dst.clone();
                click(raw, st.2, true);
            }
            31 | 32 => raw.events.push(egui::Event::PointerMoved(press + egui::vec2(40.0, 20.0) * (stage - 30) as f32 * 0.5)),
            33 => click(raw, press + egui::vec2(40.0, 20.0), false),
            34 => {}
            35 => {
                let g = gw(self);
                let s0 = &self.test.gridwarp_snap;
                let sc = |q: [f32; 2]| at(self, q);
                // In screen points the map is the same affine (the image
                // fills the viewer, no downstream chain): pivot = point 11.
                let pv = sc(s0[11]);
                let (fx, fy) = ((press.x + 40.0 - pv.x) / (press.x - pv.x), (press.y + 20.0 - pv.y) / (press.y - pv.y));
                for k in [11, 13, 16, 18] {
                    let p = sc(s0[k]);
                    let want = egui::pos2(pv.x + (p.x - pv.x) * fx, pv.y + (p.y - pv.y) * fy);
                    if (sc(g.dst[k]) - want).length() > 0.75 {
                        fail(format!("the corner scale put point {k} at {:?}, wanted {want:?}", sc(g.dst[k])));
                    }
                }
                if g.dst[12] != s0[12] {
                    fail("the box scale moved a point outside the pick".into());
                }
            }
            // ctrl on the top edge skews: x slides along y about the bottom.
            36 => {
                let Some(hot) = nodes::gridwarp::box_hot(&ctx, id) else { fail("the box went away".into()) };
                st.2 = hot.scale[1];
                self.test.gridwarp_snap = gw(self).dst.clone();
                raw.events.insert(0, egui::Event::ModifiersChanged(egui::Modifiers::COMMAND));
                click(raw, st.2, true);
            }
            37 | 38 => {
                raw.events.insert(0, egui::Event::ModifiersChanged(egui::Modifiers::COMMAND));
                raw.events.push(egui::Event::PointerMoved(press + egui::vec2(30.0, 0.0) * (stage - 36) as f32 * 0.5));
            }
            39 => {
                raw.events.insert(0, egui::Event::ModifiersChanged(egui::Modifiers::COMMAND));
                click(raw, press + egui::vec2(30.0, 0.0), false);
            }
            40 => raw.events.push(egui::Event::ModifiersChanged(egui::Modifiers::NONE)),
            41 => {
                let g = gw(self);
                let s0 = &self.test.gridwarp_snap;
                let sc = |q: [f32; 2]| at(self, q);
                let bottom = sc(s0[16]).y.max(sc(s0[18]).y);
                let k = 30.0 / (press.y - bottom);
                for i in [11, 13, 16, 18] {
                    let p = sc(s0[i]);
                    let want = egui::pos2(p.x + k * (p.y - bottom), p.y);
                    if (sc(g.dst[i]) - want).length() > 0.75 {
                        fail(format!("the skew put point {i} at {:?}, wanted {want:?}", sc(g.dst[i])));
                    }
                }
            }
            // The pivot dragged onto point 11, then the knob a quarter round
            // it: point 11 stays, the rest turn about it.
            42 => {
                let Some(hot) = nodes::gridwarp::box_hot(&ctx, id) else { fail("the box went away".into()) };
                st.2 = hot.pivot;
                click(raw, st.2, true);
            }
            43 | 44 => {
                let to = at(self, gw(self).dst[11]);
                raw.events.push(egui::Event::PointerMoved(press + (to - press) * (stage - 42) as f32 * 0.5));
            }
            45 => click(raw, at(self, gw(self).dst[11]), false),
            46 => {}
            47 => {
                let Some(hot) = nodes::gridwarp::box_hot(&ctx, id) else { fail("the box went away".into()) };
                if (hot.pivot - at(self, gw(self).dst[11])).length() > 0.75 {
                    fail(format!("the pivot drag left it at {:?}", hot.pivot));
                }
                st.2 = hot.knob;
                self.test.gridwarp_snap = gw(self).dst.clone();
                self.test.gridwarp_pivot = hot.pivot;
                click(raw, st.2, true);
            }
            48..=53 => {
                let pv = self.test.gridwarp_pivot;
                let r = press - pv;
                let a = r.angle() + std::f32::consts::FRAC_PI_2 * (stage - 47) as f32 / 6.0;
                raw.events.push(egui::Event::PointerMoved(pv + egui::vec2(a.cos(), a.sin()) * r.length()));
            }
            54 => {
                let pv = self.test.gridwarp_pivot;
                let r = press - pv;
                click(raw, pv + egui::vec2(-r.y, r.x), false);
            }
            55 => {}
            56 => {
                let g = gw(self);
                let s0 = &self.test.gridwarp_snap;
                let sc = |q: [f32; 2]| at(self, q);
                let pv = self.test.gridwarp_pivot;
                for i in [11, 13, 16, 18] {
                    let v = sc(s0[i]) - pv;
                    let want = pv + egui::vec2(-v.y, v.x);
                    if (sc(g.dst[i]) - want).length() > 1.0 {
                        fail(format!("the quarter turn put point {i} at {:?}, wanted {want:?}", sc(g.dst[i])));
                    }
                }
                nodes::gridwarp::set_pick(&ctx, id, g.count(), &[]);
            }
            57..=66 => {}
            67 => {
                // Animate point 18 (3, 3) across x and stop between the keys,
                // on a fresh lattice (the box moved it).
                if let Some(n) = self.egraph.node_mut(id) {
                    n.kind = NodeKind::GridWarp(Default::default());
                    let c = n.anim.entry(anim::Channel::GwDstX(18)).or_default();
                    c.set_key(0, 0.75);
                    c.set_key(12, 0.85);
                }
                self.seek_to(6);
            }
            68 => {
                if self.force_display || self.eval_tick() != 6 {
                    st.0 = 68;
                }
            }
            69 => {
                let x = gw(self).dst[18][0];
                st.2 = at(self, [x, 0.75]);
                click(raw, st.2, true);
            }
            70 | 71 => raw.events.push(egui::Event::PointerMoved(press + d3 * (stage - 69) as f32 * 0.5)),
            72 => click(raw, press + d3, false),
            73 => {}
            74 => {
                let n = self.egraph.node(id).expect("the GridWarp");
                let authored = match &n.kind {
                    NodeKind::GridWarp(g) => g.dst[18][0],
                    _ => f32::NAN,
                };
                let c = n.anim.get(&anim::Channel::GwDstX(18));
                let k6 = c.and_then(|c| c.keys.iter().find(|k| k.frame == 6.0).map(|k| k.value));
                let keys = c.map_or(0, |c| c.keys.len());
                let want = 0.8 + per(self, d3)[0];
                match k6 {
                    Some(v) if (v - want).abs() < 2e-3 && keys == 3 && authored == 0.75 => {}
                    _ => fail(format!(
                        "dragging an animated point at frame 6 left key {k6:?} ({keys} keys, authored {authored}), wanted {want}"
                    )),
                }
                println!(
                    "GRIDWARP TEST PASS: a destination drag landed under the pointer with its source held; \
                     source mode moved the source alone; a box picked the two points it covered and a drag \
                     moved both; the box scaled four points about the opposite corner, skewed them with ctrl \
                     on an edge, and turned them a quarter round a pivot dragged onto one of them; on an animated point a drag between keys keyed that frame at the curve plus the travel"
                );
                std::process::exit(0);
            }
            _ => {}
        }
        self.test.gridwarp = Some(st);
    }
}
