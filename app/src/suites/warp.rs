//! `--test-warp`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-warp driver, through the real viewer input: three S-clicks
    /// draw a curve whose two sides start together (no warp), a drag on its
    /// middle destination point moves that point under the pointer and
    /// leaves its source, a drag inside the box round all three moves every
    /// destination by the travel, and a D-click removes a point (the curve
    /// keeps the other two). Wall-clock deadline.
    pub(crate) fn drive_test_warp(&mut self, raw: &mut egui::RawInput) {
        let Some((stage, id, press, deadline)) = self.test_warp else { return };
        let fail = |m: String| -> ! {
            println!("WARP TEST FAIL: {m}");
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
        let key = |raw: &mut egui::RawInput, key: egui::Key, down: bool| {
            raw.events.push(egui::Event::Key { key, physical_key: None, pressed: down, repeat: false, modifiers: Default::default() });
        };
        let at = |s: &Self, f: [f32; 2]| {
            let r = s.test_rect;
            egui::pos2(r.min.x + r.width() * f[0], r.min.y + r.height() * f[1])
        };
        let wp = |s: &Self| match s.egraph.node(id).map(|n| n.kind_at(s.eval_tick())) {
            Some(NodeKind::Warp(w)) => w,
            _ => Default::default(),
        };
        let per = |s: &Self, d: egui::Vec2| [d.x / s.test_rect.width(), d.y / s.test_rect.height()];
        let near = |a: [f32; 2], b: [f32; 2]| (a[0] - b[0]).abs() < 2e-3 && (a[1] - b[1]).abs() < 2e-3;
        let ctx = self.egui_ctx.clone();
        const XS: [f32; 3] = [0.3, 0.5, 0.7];
        let (d1, d2) = (egui::vec2(0.0, 40.0), egui::vec2(30.0, 0.0));
        let mut st = (stage + 1, id, press, deadline);
        match stage {
            0 => {
                if !self.layout_settled() || self.playback.is_none() {
                    st.0 = 0;
                } else {
                    self.test_rect = self.image_rect;
                    let n = self.egraph.add(NodeKind::Warp(Default::default()), egui::Pos2::new(240.0, 0.0));
                    self.selected = vec![n];
                    self.playing = false;
                    st.1 = n;
                }
            }
            1 => {}
            // Three S-clicks, five steps each: S down, press, release, S up,
            // a beat.
            2..=16 => {
                let k = ((stage - 2) / 5) as usize;
                let p = at(self, [XS[k], 0.5]);
                match (stage - 2) % 5 {
                    0 => key(raw, egui::Key::S, true),
                    1 => click(raw, p, true),
                    2 => click(raw, p, false),
                    3 => key(raw, egui::Key::S, false),
                    _ => {}
                }
            }
            17 => {
                let w = wp(self);
                if w.pairs.len() != 1 || w.pairs[0].len() != 3 {
                    fail(format!("three S-clicks made {} curve(s), {:?} points", w.pairs.len(), w.pairs.iter().map(|p| p.len()).collect::<Vec<_>>()));
                }
                for (k, x) in XS.iter().enumerate() {
                    if !near(w.pairs[0].dst[k], [*x, 0.5]) || !near(w.pairs[0].src[k], [*x, 0.5]) {
                        fail(format!("point {k} landed at src {:?} dst {:?}, wanted both at ({x}, 0.5)", w.pairs[0].src[k], w.pairs[0].dst[k]));
                    }
                }
            }
            // The middle destination point dragged down.
            18 => {
                st.2 = at(self, [0.5, 0.5]);
                click(raw, st.2, true);
            }
            19 | 20 => raw.events.push(egui::Event::PointerMoved(press + d1 * (stage - 18) as f32 * 0.5)),
            21 => click(raw, press + d1, false),
            22 => {}
            23 => {
                let w = wp(self);
                let want = [0.5 + per(self, d1)[0], 0.5 + per(self, d1)[1]];
                if !near(w.pairs[0].dst[1], want) || !near(w.pairs[0].src[1], [0.5, 0.5]) {
                    fail(format!("the drag left dst {:?} src {:?}, wanted dst {want:?}", w.pairs[0].dst[1], w.pairs[0].src[1]));
                }
                nodes::warp::set_pick(&ctx, id, &w, &[(0, 0), (0, 1), (0, 2)]);
            }
            24 | 25 => {}
            // A drag inside the box (off every point) moves all three.
            26 => {
                if nodes::warp::box_hot(&ctx, id).is_none() {
                    fail("three picked points drew no box".into());
                }
                st.2 = at(self, [0.4, 0.5]) + egui::vec2(0.0, 20.0);
                click(raw, st.2, true);
            }
            27 | 28 => raw.events.push(egui::Event::PointerMoved(press + d2 * (stage - 26) as f32 * 0.5)),
            29 => click(raw, press + d2, false),
            30 => {}
            31 => {
                let w = wp(self);
                let dx = per(self, d2)[0];
                let dy1 = per(self, d1)[1];
                let want = [[0.3 + dx, 0.5], [0.5 + dx, 0.5 + dy1], [0.7 + dx, 0.5]];
                for k in 0..3 {
                    if !near(w.pairs[0].dst[k], want[k]) || !near(w.pairs[0].src[k], [XS[k], 0.5]) {
                        fail(format!("the box move left point {k} dst {:?} src {:?}, wanted dst {:?}", w.pairs[0].dst[k], w.pairs[0].src[k], want[k]));
                    }
                }
                if nodes::warp::picked(&ctx, id, &w).len() != 3 {
                    fail("the box move dropped the pick".into());
                }
                st.2 = at(self, w.pairs[0].dst[0]);
            }
            // D-click on the first point removes it.
            32 => key(raw, egui::Key::D, true),
            33 => click(raw, press, true),
            34 => click(raw, press, false),
            35 => key(raw, egui::Key::D, false),
            36 => {}
            37 => {
                let w = wp(self);
                if w.pairs[0].len() != 2 || !near(w.pairs[0].src[0], [0.5, 0.5]) {
                    fail(format!("a D-click left {:?}", w.pairs[0]));
                }
                println!(
                    "WARP TEST PASS: three S-clicks drew a curve with its sides together; a destination drag moved \
                     that point under the pointer and left its source; a drag in the box moved every destination \
                     by the travel; a D-click removed a point"
                );
                std::process::exit(0);
            }
            _ => {}
        }
        self.test_warp = Some(st);
    }
}
