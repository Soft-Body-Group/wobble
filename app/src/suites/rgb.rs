//! `--test-rgb`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-rgb driver: the rgb slider (`rgbslider.rs`) under real input,
    /// on a Mult node. Three things only a live run can show: a ganged drag
    /// moves all three lanes and KEEPS THEIR OFFSETS, a colour axis held
    /// mid-gesture moves the triple without touching its luma, and the
    /// widget claims the keyboard while it is dragged — which is what stops
    /// D (viewer back to Normal) from firing under the temperature axis.
    pub(crate) fn drive_test_rgb(&mut self, raw: &mut egui::RawInput) {
        let Some((stage, mut frames, id)) = self.test.rgb else { return };
        frames += 1;
        if frames > 1200 {
            println!("RGB TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let fail = |m: String| -> ! {
            println!("RGB TEST FAIL: {m}");
            std::process::exit(1);
        };
        let press = |raw: &mut egui::RawInput, pos: egui::Pos2, down: bool| {
            raw.events.push(egui::Event::PointerMoved(pos));
            raw.events.push(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: down,
                modifiers: Default::default(),
            });
        };
        let key = |raw: &mut egui::RawInput, k: egui::Key, down: bool| {
            raw.events.push(egui::Event::Key {
                key: k,
                physical_key: None,
                pressed: down,
                repeat: false,
                modifiers: Default::default(),
            });
        };
        let value = |s: &Self| -> [f32; 3] {
            match s.egraph.node(id).map(|n| &n.kind) {
                Some(NodeKind::Mult(m)) => m.value,
                _ => [f32::NAN; 3],
            }
        };
        let find_kind = |s: &Self, want: &str| -> Option<u64> {
            s.egraph
                .nodes
                .iter()
                .find(|(_, n)| n.kind.name() == want)
                .map(|(id, _)| *id)
        };
        let gain_of = |s: &Self| -> [f32; 3] {
            match s.egraph.node(id).map(|n| &n.kind) {
                Some(NodeKind::Grade(g)) => g.gain,
                _ => [f32::NAN; 3],
            }
        };
        // The axes preserve the WORKING SPACE's luma, so the assertion has
        // to use the same weights the widget does.
        let w = self.egraph.working.primaries.luma_weights();
        let luma = move |v: [f32; 3]| w[0] * v[0] + w[1] * v[1] + w[2] * v[2];
        let geom = rgbslider::last_geom(&self.egui_ctx, (id, "value"));
        // Every drag is the same shape: press, three moves, release. The
        // first move is the frame egui calls drag_started, and the widget
        // skips that one (its delta is the threshold, not an edit). The
        // step is a fraction of the TRACK, so the gesture means the same
        // thing whatever the panel is sized or zoomed to.
        // Each gesture: press, three moves an eighth of the TRACK apart,
        // release — so it means the same thing whatever the panel is sized
        // or zoomed to, and the net travel is a known 3/8 of the track.
        let mut next = stage;
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 3 => {
                // A deliberately UNEVEN triple: "ganged" means the offsets
                // survive, which a neutral 1,1,1 could never show.
                let nid = self.egraph.add(
                    NodeKind::Mult(engine::graph::MultParams {
                        value: [1.0, 1.4, 0.6],
                        alpha: false,
                        matte: false,
                    }),
                    egui::Pos2::new(300.0, 0.0),
                );
                self.egraph.connect(nid, 0, 0); // Mult <- Read A
                self.egraph.view = Some(nid);
                self.egraph.view_active_b = false;
                self.prop_order = vec![nid];
                self.props_visible = true;
                self.bottom = BottomTab::Properties;
                self.playing = false;
                self.seek_to(0);
                self.test.rgb = Some((1, frames, nid));
                return;
            }
            // Ganged drag on the middle lane: the gang box sits over the
            // knobs, so this grab moves all three.
            1 if frames > 20 => {
                let Some(g) = geom else { return };
                // A slider needs TRACK. The fixed chrome (gang box, lane
                // letters, number field, colour chip) eats ~126px, so a row
                // in a one-tile panel can end up with almost nothing to aim
                // along — precision the numbers cannot give back.
                let tw = g.track_x1 - g.track_x0;
                if tw < 120.0 {
                    fail(format!(
                        "the Mult's rgb track is only {tw:.0}px wide — the panel \
                         is too narrow for the widget's chrome"
                    ));
                }
                self.test.rgb_before = value(self);
                self.test.rgb_anchor =
                    (egui::pos2(g.knob_x[1], g.lane_y[1]), g.track_x1 - g.track_x0);
                press(raw, self.test.rgb_anchor.0, true);
                next = 2;
            }
            2 | 3 | 4 => {
                let (a, tw) = self.test.rgb_anchor;
                raw.events
                    .push(egui::Event::PointerMoved(a + egui::vec2((stage - 1) as f32 * tw / 8.0, 0.0)));
                if stage == 3 && !rgbslider::grabbing(&self.egui_ctx) {
                    fail("the widget is being dragged but does not claim the \
                          keyboard — D and F would fire the app's hotkeys".into());
                }
                next = stage + 1;
            }
            5 => {
                let (a, tw) = self.test.rgb_anchor;
                press(raw, a + egui::vec2(3.0 * tw / 8.0, 0.0), false);
                next = 6;
            }
            6 => {
                let (v, b) = (value(self), self.test.rgb_before);
                let d = [v[0] - b[0], v[1] - b[1], v[2] - b[2]];
                if d[0] < 0.05 {
                    fail(format!("ganged drag right moved red by {:.4} — nothing happened", d[0]));
                }
                if (d[1] - d[0]).abs() > 1e-3 || (d[2] - d[0]).abs() > 1e-3 {
                    fail(format!(
                        "ganged drag was not relative: {b:?} -> {v:?} (deltas {d:?})"
                    ));
                }
                // The knob has to stay under the finger, exactly: the
                // gesture ends three track-eighths right of where it was
                // pressed, so the value ends three eighths of 0..16 up.
                // (egui starts the drag on the press, so no move frame is
                // lost to a threshold.)
                let want = 3.0 / 8.0 * 16.0;
                if (d[0] - want).abs() > 0.2 {
                    fail(format!(
                        "a {:.0}px drag moved the value {:.3}, want {want:.3} — the \
                         knob is not tracking the pointer",
                        3.0 * self.test.rgb_anchor.1 / 8.0,
                        d[0]
                    ));
                }
                self.test.rgb_before = v;
                next = 7;
            }
            // Same gesture with D held: the temperature axis. Warmer to the
            // right, and the luma must not move at all.
            7 => {
                let Some(g) = geom else { fail("widget gone".into()) };
                key(raw, egui::Key::D, true);
                self.test.rgb_anchor =
                    (egui::pos2(g.knob_x[1], g.lane_y[1]), g.track_x1 - g.track_x0);
                press(raw, self.test.rgb_anchor.0, true);
                next = 8;
            }
            8 | 9 | 10 => {
                let (a, tw) = self.test.rgb_anchor;
                raw.events
                    .push(egui::Event::PointerMoved(a + egui::vec2((stage - 7) as f32 * tw / 8.0, 0.0)));
                next = stage + 1;
            }
            11 => {
                let (a, tw) = self.test.rgb_anchor;
                press(raw, a + egui::vec2(3.0 * tw / 8.0, 0.0), false);
                key(raw, egui::Key::D, false);
                next = 12;
            }
            12 => {
                let (v, b) = (value(self), self.test.rgb_before);
                if !(v[0] > b[0] && v[2] < b[2]) {
                    fail(format!("D dragged right must warm the triple: {b:?} -> {v:?}"));
                }
                if self.viewer_mode != ViewerMode::Normal {
                    fail("holding D changed the viewer mode".into());
                }
                let (l0, l1) = (luma(b), luma(v));
                if (l1 - l0).abs() > 2e-3 * l0.max(1.0) {
                    fail(format!("temperature axis moved the luma {l0:.4} -> {l1:.4}"));
                }
                self.test.rgb_before = v;
                next = 13;
            }
            // And with G held the gang inverts: one lane, alone.
            // G goes down a frame EARLY, the way a hand holds it: egui
            // routes a press by the PREVIOUS frame's rects, so the gang box
            // has to be out of the layout before the press lands or it
            // swallows a grab meant for the lane.
            13 => {
                key(raw, egui::Key::G, true);
                next = 14;
            }
            14 => {
                let Some(g) = geom else { fail("widget gone".into()) };
                self.test.rgb_anchor =
                    (egui::pos2(g.knob_x[2], g.lane_y[2]), g.track_x1 - g.track_x0);
                press(raw, self.test.rgb_anchor.0, true);
                next = 15;
            }
            15 | 16 | 17 => {
                let (a, tw) = self.test.rgb_anchor;
                raw.events
                    .push(egui::Event::PointerMoved(a + egui::vec2((stage - 14) as f32 * tw / 8.0, 0.0)));
                next = stage + 1;
            }
            18 => {
                let (a, tw) = self.test.rgb_anchor;
                press(raw, a + egui::vec2(3.0 * tw / 8.0, 0.0), false);
                key(raw, egui::Key::G, false);
                next = 19;
            }
            19 => {
                let (v, b) = (value(self), self.test.rgb_before);
                if (v[0] - b[0]).abs() > 1e-4 || (v[1] - b[1]).abs() > 1e-4 {
                    fail(format!("G-held drag was not per-lane: {b:?} -> {v:?}"));
                }
                if v[2] - b[2] < 0.05 {
                    fail(format!("G-held drag on blue moved it by {:.4}", v[2] - b[2]));
                }
                // On to the Grade's slider mode: same widget, a node whose
                // panel normally shows wheels.
                let gid = self.egraph.add(
                    NodeKind::Grade(engine::graph::GradeParams {
                        gain: [1.0, 1.2, 0.8],
                        ..Default::default()
                    }),
                    egui::Pos2::new(460.0, 0.0),
                );
                self.egraph.connect(gid, 0, 0);
                self.egraph.view = Some(gid);
                if let Some(n) = self.egraph.node_mut(gid) {
                    n.grade_sliders = true;
                }
                self.prop_order = vec![gid];
                self.test.rgb = Some((20, 0, gid));
                return;
            }
            // The mode has to actually draw rgb rows: no geometry under
            // (node, "gain") means the panel is still showing wheels.
            20 if frames > 20 => {
                let Some(g) = rgbslider::last_geom(&self.egui_ctx, (id, "gain")) else {
                    return;
                };
                // Two columns of two, not one tall column: four stacked rows
                // overflow the tile, and an overflowed row cannot be clicked
                // at all (it is inside the tile's scroll area). This is the
                // assertion that catches that regression.
                let geo = |n: &str| rgbslider::last_geom(&self.egui_ctx, (id, n));
                let (Some(lift), Some(gamma), Some(offset)) =
                    (geo("lift"), geo("gamma"), geo("offset"))
                else {
                    fail("slider mode did not draw all four rows".into())
                };
                if (gamma.track_x0 - lift.track_x0).abs() > 1.0
                    || (offset.track_x0 - g.track_x0).abs() > 1.0
                {
                    fail("lift/gamma and gain/offset should each share a column".into());
                }
                if g.track_x0 <= lift.track_x0 + 40.0 {
                    fail(format!(
                        "the second column starts at {:.0}, barely right of the \
                         first at {:.0} — they are stacked, not side by side",
                        g.track_x0, lift.track_x0
                    ));
                }
                if gamma.lane_y[0] <= lift.lane_y[0] || (g.lane_y[0] - lift.lane_y[0]).abs() > 1.0
                {
                    fail("rows are not laid out two per column".into());
                }
                self.test.rgb_before = gain_of(self);
                self.test.rgb_anchor =
                    (egui::pos2(g.knob_x[1], g.lane_y[1]), g.track_x1 - g.track_x0);
                press(raw, self.test.rgb_anchor.0, true);
                next = 21;
            }
            21 | 22 | 23 => {
                let (a, tw) = self.test.rgb_anchor;
                raw.events.push(egui::Event::PointerMoved(
                    a + egui::vec2((stage - 20) as f32 * tw / 8.0, 0.0),
                ));
                next = stage + 1;
            }
            24 => {
                let (a, tw) = self.test.rgb_anchor;
                press(raw, a + egui::vec2(3.0 * tw / 8.0, 0.0), false);
                next = 25;
            }
            25 => {
                let (v, b) = (gain_of(self), self.test.rgb_before);
                let d = [v[0] - b[0], v[1] - b[1], v[2] - b[2]];
                if d[0] < 0.05 {
                    fail(format!("Grade slider mode: gain did not move ({b:?} -> {v:?})"));
                }
                if (d[1] - d[0]).abs() > 1e-3 || (d[2] - d[0]).abs() > 1e-3 {
                    fail(format!("Grade slider gang was not relative: {b:?} -> {v:?}"));
                }
                self.test.rgb_before = v;
                // Every node with a colour triple has to be driving THIS
                // widget: a tint drawn with a colour button cannot be ganged,
                // keyed or pushed along the S/D/F axes. Each panel is pinned
                // alone — a strip of them scrolls, and a panel scrolled out
                // of view proves nothing.
                for (kind, pos) in [
                    (NodeKind::Glow(Default::default()), 620.0),
                ] {
                    let n = self.egraph.add(kind, egui::Pos2::new(pos, 0.0));
                    self.egraph.connect(n, 0, 0);
                }
                let Some(first) = find_kind(self, "Glow") else {
                    fail("Glow node vanished".into())
                };
                self.prop_order = vec![first];
                next = 26;
            }
            26 if frames > 20 => {
                let Some(nid) = find_kind(self, "Glow") else {
                    fail("Glow node vanished".into())
                };
                if rgbslider::last_geom(&self.egui_ctx, (nid, "tint")).is_none() {
                    fail(
                        "Glow's tint is not drawn by the rgb slider — it is still a \
                         colour button (no rgb-slider geometry for ({nid}, \"tint\"))"
                            .replace("{nid}", &nid.to_string()),
                    );
                }
                let v = self.test.rgb_before;
                println!(
                    "RGB TEST PASS: ganged drag stayed relative, D held \
                     slid orange<->cyan at constant luma, G held moved \
                     blue alone, a Grade in slider mode ganged its gain \
                     to {:.3},{:.3},{:.3}, and the Glow tint draws the \
                     same rgb slider",
                    v[0], v[1], v[2]
                );
                std::process::exit(0);
            }
            // A frame of settle between panels: the pinned panel changes, and
            // the next stage reads geometry the new one has not drawn yet.
            27 | 29 => {
                next = stage + 1;
            }
            _ => {}
        }
        self.test.rgb = Some((next, if next == stage { frames } else { 0 }, id));
    }
}
