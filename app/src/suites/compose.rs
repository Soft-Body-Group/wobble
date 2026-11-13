//! `--test-compose`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-compose driver: the subwob panel composer on a subwob the
    /// suite COLLAPSES from a Grade (the alt+G path) and fills with four rgb
    /// controls ([4,3] cells), widened to 4 tiles.
    /// The composer's grid must be the width of its columns (it was a third
    /// of it); a body drag follows the pointer PER PIXEL and lands on the
    /// nearest cell only at release; the right bar scales; a drop onto
    /// another control puts it back.
    pub(crate) fn drive_test_compose(&mut self, raw: &mut egui::RawInput) {
        let Some((stage, mut frames)) = self.test_compose else { return };
        let ids = self.test_compose_ids;
        frames += 1;
        if frames > 1200 {
            println!("COMPOSE TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let fail = |m: String| -> ! {
            println!("COMPOSE TEST FAIL: {m}");
            std::process::exit(1);
        };
        let button = |raw: &mut egui::RawInput, pos: egui::Pos2, down: bool| {
            raw.events.push(egui::Event::PointerMoved(pos));
            raw.events.push(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: down,
                modifiers: Default::default(),
            });
        };
        const TILES: usize = 4;
        let (cols, rows) = (tools::cols(TILES), tools::rows(1));
        let cell_of = |s: &Self, id: u64| -> ([i32; 2], [u8; 2]) {
            tools::layout_cells(&s.egraph, cols, rows)
                .into_iter()
                .find(|(i, _, _)| *i == id)
                .map(|(_, c, sp)| (c, sp))
                .unwrap_or_else(|| fail(format!("control {id} is not on the panel")))
        };
        let rect_of = |s: &Self, id: u64| {
            let (c, sp) = cell_of(s, id);
            egui::Rect::from_min_size(
                s.compose_area.min
                    + egui::vec2(c[0] as f32 * tools::CELL_W, c[1] as f32 * tools::CELL_H),
                egui::vec2(sp[0] as f32 * tools::CELL_W, sp[1] as f32 * tools::CELL_H),
            )
        };
        const STEP: f32 = 7.0;
        let mut next = stage;
        // A drag of `want` px along x (stage `s`) or y, from the press
        // stored in `test_compose_at`, a step a frame, checking the live
        // rect trails the pointer by exactly one frame's travel.
        let drag_step = |s: &mut Self, raw: &mut egui::RawInput, want: egui::Vec2, id: u64, check: bool| -> bool {
            let (press, last, start) = s.test_compose_at;
            let done = last - press;
            if check && done.length() > 0.0 {
                // What the pass that saw `last` drew: the control moved by
                // exactly the pointer's travel, not by whole cells.
                let Some(live) = s.compose_live else {
                    fail(format!("mid-drag ({done:?} px) the composer shows no live control"))
                };
                let moved = live.min - start.min;
                if (moved - done).length() > 1.0 {
                    fail(format!("the control moved {moved:?} for a {done:?} px drag: it should follow per pixel"));
                }
                let at_press = start.min - s.compose_area.min;
                let c0 = [
                    (at_press.x / tools::CELL_W).round() as i32,
                    (at_press.y / tools::CELL_H).round() as i32,
                ];
                if cell_of(s, id).0 != c0 {
                    fail("the cell changed MID-drag; it should only snap on release".into());
                }
            }
            if done.length() >= want.length() {
                button(raw, last, false);
                return true;
            }
            let dir = want / want.length();
            let p = press + dir * (done.length() + STEP).min(want.length());
            raw.events.push(egui::Event::PointerMoved(p));
            s.test_compose_at.1 = p;
            false
        };
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 2 => {
                if !self.layout_settled() {
                    return;
                }
                // The fixture is built the way a hand builds one: a Grade on
                // the plate COLLAPSED into a subwob (alt+G's own path),
                // opened, and four rgb controls laid out where they sit on a
                // four-tile panel: [0,0] [5,0] [0,3] [5,3], each [4,3].
                let read = self
                    .egraph
                    .nodes
                    .iter()
                    .find(|(_, n)| matches!(n.kind, NodeKind::Read { .. }))
                    .map(|(id, _)| *id)
                    .unwrap_or_else(|| fail("needs a Read".into()));
                let gr = self.egraph.add(NodeKind::Grade(Default::default()), egui::pos2(360.0, 120.0));
                self.egraph.edges.push((gr, 0, read));
                self.selected = vec![gr];
                self.collapse_selection();
                let n = self.selected.first().copied().unwrap_or(gr);
                if !matches!(self.egraph.node(n).map(|e| &e.kind),
                    Some(NodeKind::Custom { type_id, .. }) if type_id.starts_with(tools::LOCAL_PREFIX))
                {
                    fail(format!("the collapse made no subwob: {:?}", self.tool_error));
                }
                if self.egraph.input_of(n, 0) != Some(read) {
                    fail("the subwob is not fed by the plate the Grade was".into());
                }
                if let Err(e) = self.dive_into(n) {
                    fail(format!("dive: {e}"));
                }
                if let Some(d) = self.dive.last_mut() {
                    d.tiles = TILES;
                    d.tiles_h = 1;
                }
                for (k, cell) in [[0, 0], [5, 0], [0, 3], [5, 3]].into_iter().enumerate() {
                    let id = self.egraph.add(
                        NodeKind::Custom {
                            type_id: tools::ControlKind::Rgb.type_id().into(),
                            params: Default::default(),
                        },
                        egui::pos2(-300.0, k as f32 * 60.0),
                    );
                    if let Some(e) = self.egraph.nodes.get_mut(&id) {
                        e.ui_cell = Some(cell);
                        e.ui_span = Some([4, 3]);
                        e.ui_label = ["offset", "lift", "gamma", "gain"][k].into();
                    }
                    self.test_compose_ids[k] = id;
                }
                let mut g = std::mem::take(&mut self.egraph);
                tools::place_controls(&mut g, cols, rows);
                self.egraph = g;
                self.layout_open = true;
                next = 1;
            }
            // The window is up and its grid is the width of its columns.
            1 if frames > 20 && self.compose_area.width() > 0.0 => {
                let want = cols as f32 * tools::CELL_W;
                if (self.compose_area.width() - want).abs() > 1.0 {
                    fail(format!(
                        "the composer grid is {:.0}px wide for {cols} columns of {:.0}px ({want:.0}px)",
                        self.compose_area.width(),
                        tools::CELL_W
                    ));
                }
                let (c, sp) = cell_of(self, ids[1]);
                if c != [5, 0] || sp != [4, 3] {
                    fail(format!("control 3 starts at {c:?} {sp:?}, the fixture put it at [5,0] [4,3]"));
                }
                // Grab control 3 by its BODY (left of centre, clear of the
                // bars) — anywhere on it moves it now.
                let r = rect_of(self, ids[1]);
                let press = egui::pos2(r.min.x + r.width() * 0.35, r.center().y);
                button(raw, press, true);
                self.test_compose_at = (press, press, r);
                next = 2;
            }
            // Just short of three columns right (the panel edge): snaps to [8,0].
            2 if frames > 1 => {
                if self.layout_drag.is_none() {
                    fail("a press on the control's body did not pick it up".into());
                }
                if drag_step(self, raw, egui::vec2(3.0 * tools::CELL_W - 14.0, 4.0), ids[1], true) {
                    next = 3;
                }
            }
            3 if frames > 3 => {
                let (c, _) = cell_of(self, ids[1]);
                if c != [8, 0] || self.layout_drag.is_some() {
                    fail(format!("released at ~2.85 columns, the control landed on {c:?} (want [8, 0])"));
                }
                // Scale control 2 from its RIGHT bar by two columns.
                let r = rect_of(self, ids[0]);
                let press = egui::pos2(r.max.x - 5.0, r.center().y);
                button(raw, press, true);
                self.test_compose_at = (press, press, r);
                next = 4;
            }
            4 if frames > 1 => {
                if !matches!(self.layout_drag, Some((i, PanelGrab::Right, ..)) if i == ids[0]) {
                    fail("a press on the right bar did not take the scale grip".into());
                }
                if drag_step(self, raw, egui::vec2(2.0 * tools::CELL_W + 6.0, 0.0), ids[0], false) {
                    next = 5;
                }
            }
            5 if frames > 3 => {
                let (c, sp) = cell_of(self, ids[0]);
                if c != [0, 0] || sp != [6, 3] {
                    fail(format!("scaled right by two columns: {c:?} {sp:?} (want [0,0] [6,3])"));
                }
                // Drop control 4 ([0,3]) onto control 2: refused.
                let r = rect_of(self, ids[2]);
                let press = r.center() - egui::vec2(r.width() * 0.1, 0.0);
                button(raw, press, true);
                self.test_compose_at = (press, press, r);
                next = 6;
            }
            6 if frames > 1 => {
                if drag_step(self, raw, egui::vec2(0.0, -3.0 * tools::CELL_H), ids[2], false) {
                    next = 7;
                }
            }
            7 if frames > 3 => {
                let (c, _) = cell_of(self, ids[2]);
                if c != [0, 3] {
                    fail(format!("dropped onto another control, it landed on {c:?}: it should go back to [0, 3]"));
                }
                // Now PULL a knob in: two tiles tall for a wheel's six
                // rows, and the Grade inside the tool raised in properties.
                if let Some(d) = self.dive.last_mut() {
                    d.tiles_h = 2;
                }
                let grade = self
                    .egraph
                    .nodes
                    .iter()
                    .find(|(_, n)| matches!(n.kind, NodeKind::Grade(_)))
                    .map(|(id, _)| *id)
                    .unwrap_or_else(|| fail("simple_grade has no Grade inside".into()));
                self.selected = vec![grade];
                self.prop_order = vec![grade];
                self.props_visible = true;
                self.bottom = BottomTab::Properties;
                next = 8;
            }
            // ctrl-press on the Gain wheel.
            8 if frames > 30 => {
                let Some((r, clip)) =
                    widgets::noted_rect(&self.egui_ctx, egui::Id::new(("wheel-col", "Gain")))
                else {
                    fail("the Grade's Gain wheel was never drawn".into())
                };
                let press = r.center() + egui::vec2(0.0, -r.height() * 0.15);
                if !clip.contains(press) {
                    fail(format!("the Gain wheel is clipped ({r:?} in {clip:?})"));
                }
                let ctrl = egui::Modifiers { ctrl: true, command: true, ..Default::default() };
                raw.events.insert(0, egui::Event::ModifiersChanged(ctrl));
                raw.events.push(egui::Event::PointerMoved(press));
                raw.events.push(egui::Event::PointerButton {
                    pos: press,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: ctrl,
                });
                self.test_compose_at = (press, press, r);
                next = 9;
            }
            // Pull it: a short drag starts the pull, then carry it to an
            // empty cell of the composer ([0, 7], below the four rgb rows).
            9 | 10 => {
                let ctrl = egui::Modifiers { ctrl: true, command: true, ..Default::default() };
                raw.events.insert(0, egui::Event::ModifiersChanged(ctrl));
                let (press, last, _) = self.test_compose_at;
                let to = self.compose_area.min
                    + egui::vec2(0.5 * tools::CELL_W, 7.5 * tools::CELL_H);
                if stage == 9 {
                    let p = last + egui::vec2(6.0, 4.0);
                    raw.events.push(egui::Event::PointerMoved(p));
                    self.test_compose_at.1 = p;
                    if self.hud_drag.is_some() {
                        if !self.hud_drag.as_ref().is_some_and(|h| h.wheel && h.rgb && h.chans.len() == 3) {
                            fail("the pulled wheel is not marked as a wheel triple".into());
                        }
                        next = 10;
                    } else if (p - press).length() > 80.0 {
                        fail("a ctrl-drag on the Gain wheel never started a pull".into());
                    }
                } else {
                    let d = to - last;
                    let p = if d.length() <= 24.0 { to } else { last + d.normalized() * 24.0 };
                    raw.events.push(egui::Event::PointerMoved(p));
                    self.test_compose_at.1 = p;
                    if p == to && frames > 2 {
                        raw.events.push(egui::Event::PointerButton {
                            pos: to,
                            button: egui::PointerButton::Primary,
                            pressed: false,
                            modifiers: ctrl,
                        });
                        next = 11;
                    }
                }
            }
            11 if frames > 3 => {
                let gain = anim::grade_triple(None, 2);
                let grade = self.selected.first().copied().unwrap_or(0);
                let src = |c: anim::Channel| {
                    self.egraph.patch.link_to(editor::PDst::Chan(grade, c)).map(|l| l.from)
                };
                let Some(editor::PSrc::Chan(ctl, _)) = src(gain[0]) else {
                    fail("the dropped wheel is not cabled into the Grade's gain".into())
                };
                if gain.iter().any(|c| !matches!(src(*c), Some(editor::PSrc::Chan(i, _)) if i == ctl)) {
                    fail("the gain lanes are not all fed by the one new control".into());
                }
                let n = self.egraph.node(ctl).unwrap_or_else(|| fail("no control node".into()));
                if tools::control_of(&n.kind) != Some(tools::ControlKind::Wheel) {
                    fail(format!("the drop made a {:?}, not a Colour Wheel", tools::control_of(&n.kind)));
                }
                let (c, _) = tools::layout_cells(&self.egraph, cols, tools::rows(2))
                    .into_iter()
                    .find(|(i, _, _)| *i == ctl)
                    .map(|(_, c, sp)| (c, sp))
                    .unwrap_or_else(|| fail("the new wheel is not on the panel".into()));
                if c != [0, 7] {
                    fail(format!("dropped on [0, 7], the wheel landed on {c:?}"));
                }
                if std::env::var_os("WOBBLE_TEST_HOLD").is_some() {
                    return;
                }
                println!(
                    "COMPOSE TEST PASS: grid {:.0}px = {cols} columns; a body drag follows the pointer per pixel and snaps on release (~2.85 cols -> [8,0]); the right bar scales (+2 cols -> [6,3]); a drop onto another control goes back; the Grade's Gain wheel ctrl-dragged onto the panel became a Colour Wheel cabled into its gain, at the drop cell",
                    self.compose_area.width()
                );
                std::process::exit(0);
            }
            _ => {}
        }
        if next != stage {
            self.test_compose = Some((next, 0));
        } else {
            self.test_compose = Some((stage, frames));
        }
    }
}
