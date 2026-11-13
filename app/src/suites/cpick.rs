//! `--test-cpick`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    pub(crate) fn drive_test_cpick(&mut self, raw: &mut egui::RawInput) {
        let Some((stage, mut frames, id)) = self.test.cpick else { return };
        // Stage 0 waits on real decodes, so it runs on the wall clock; every
        // later stage is UI-only and a frame budget is honest there.
        if stage == 0 && self.test.cpick_t0.elapsed().as_secs() > 60 {
            println!("CPICK TEST FAIL: no frames on screen after 60s");
            std::process::exit(1);
        }
        if stage > 0 {
            frames += 1;
        }
        // WOBBLE_TEST_HOLD parks the suite with the picker open, for a look.
        let hold = std::env::var("WOBBLE_TEST_HOLD").is_ok();
        // Stage 23 waits on the resize settling, which is wall-clock too.
        if frames > 1200 && !hold && stage != 23 {
            println!("CPICK TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let fail = |m: String| -> ! {
            println!("CPICK TEST FAIL: {m}");
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
        let key = |raw: &mut egui::RawInput, k: egui::Key, down: bool| {
            raw.events.push(egui::Event::Key {
                key: k,
                physical_key: None,
                pressed: down,
                repeat: false,
                modifiers: Default::default(),
            });
        };
        let line = |s: &Self| -> [f32; 4] {
            match s.egraph.node(id).map(|n| &n.kind) {
                Some(NodeKind::Grid(g)) => g.line,
                _ => [f32::NAN; 4],
            }
        };
        let back = |s: &Self| -> [f32; 4] {
            match s.egraph.node(id).map(|n| &n.kind) {
                Some(NodeKind::Grid(g)) => g.back,
                _ => [f32::NAN; 4],
            }
        };
        // L is judged through the matrices the picker used, so this is the
        // round trip in the comp's own working space, not Rec.709's.
        let spec = pick_spec(self.egraph.working, true);
        let lightness = |v: [f32; 4]| colorpicker::value_to_lch([v[0], v[1], v[2]], &spec)[0];
        let geom = colorpicker::last_geom(&self.egui_ctx, (id, "grid-line"));
        let geom_back = colorpicker::last_geom(&self.egui_ctx, (id, "grid-back"));
        // The Grid's HUD chip swatch — the same salt the chip uses.
        let chip = colorpicker::last_geom(&self.egui_ctx, ("hud-ink", id, 0usize));
        // A value as the palette keeps it: linear Rec.709, through the
        // picker's own matrix.
        let d = spec.display;
        let as709 = |v: [f32; 4]| [0, 1, 2].map(|r| d[r][0] * v[0] + d[r][1] * v[1] + d[r][2] * v[2]);
        let near = |a: [f32; 3], b: [f32; 3]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4);
        let along = |r: egui::Rect, t: f32| egui::pos2(r.min.x + t * r.width(), r.center().y);
        // The free corner, just inside the frame: the grips' own ground.
        let corner_in = |g: &colorpicker::Geom| {
            let above = g.popup.bottom() <= g.swatch.top();
            let y = if above { g.popup.top() + 3.0 } else { g.popup.bottom() - 3.0 };
            egui::pos2(g.popup.right() - 3.0, y)
        };
        let mut next = stage;
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 3 => {
                // A saturated blue: at L 0.9 it cannot keep its chroma, so
                // the drag below runs through the gamut pull-back.
                let nid = self.egraph.add(
                    NodeKind::Grid(engine::graph::GridParams {
                        line: [0.05, 0.2, 0.9, 1.0],
                        ..Default::default()
                    }),
                    egui::Pos2::new(300.0, 0.0),
                );
                self.egraph.view = Some(nid);
                self.egraph.view_active_b = false;
                self.prop.order = vec![nid];
                self.props_visible = true;
                self.bottom = BottomTab::Properties;
                self.playing = false;
                self.seek_to(0);
                self.test.cpick = Some((1, 0, nid));
                return;
            }
            // Panels animate in: click the swatch only once its rect has
            // held still for a few frames.
            1 => {
                if let Some(g) = geom {
                    let (r, still) = self.test.cpick_at;
                    self.test.cpick_at = (g.swatch, if r == g.swatch { still + 1 } else { 0 });
                    if still >= 5 {
                        self.test.cpick_before = line(self);
                        button(raw, g.swatch.center(), true);
                        next = 2;
                    }
                }
            }
            2 => {
                button(raw, self.test.cpick_at.0.center(), false);
                next = 3;
            }
            3 if frames > 3 => {
                let Some(g) = geom.filter(|g| g.open) else {
                    fail("clicking the swatch did not open the picker".into())
                };
                let screen = self.egui_ctx.content_rect();
                if !screen.contains_rect(g.popup) {
                    fail(format!("the popup is off screen: {:?} in {screen:?}", g.popup));
                }
                if g.popup.intersects(g.swatch) {
                    fail("the popup covers its own swatch".into());
                }
                // Hermetic prefs: the default size, which this window has room for.
                if colorpicker::scale() != colorpicker::scale_default()
                    || (g.scale - colorpicker::scale_default()).length() > 1e-3
                {
                    fail(format!(
                        "the picker opened at scale {:?} (app {:?}), want the default {:?}",
                        g.scale,
                        colorpicker::scale(),
                        colorpicker::scale_default()
                    ));
                }
                self.test.cpick_at = (g.tabs[3], 0);
                button(raw, g.tabs[3].center(), true);
                next = 4;
            }
            4 => {
                button(raw, self.test.cpick_at.0.center(), false);
                next = 5;
            }
            // Absolute: press the L lane 30% along, drag to 70%, release.
            5 if frames > 3 => {
                if colorpicker::mode() != colorpicker::Mode::Lch {
                    fail("the lch tab did not switch the picker to lch".into());
                }
                let Some(g) = geom else { fail("the picker is gone".into()) };
                let want = colorpicker::track_len(g.scale);
                if (g.lanes[0].width() - want).abs() > 0.5 {
                    fail(format!(
                        "the L lane is {:.1} long at scale {}, want {want:.1}",
                        g.lanes[0].width(),
                        g.scale
                    ));
                }
                self.test.cpick_at = (g.lanes[0], 0);
                button(raw, along(g.lanes[0], 0.3), true);
                next = 6;
            }
            6..=8 => {
                let t = 0.3 + 0.4 * (stage - 5) as f32 / 3.0;
                raw.events.push(egui::Event::PointerMoved(along(self.test.cpick_at.0, t)));
                next = stage + 1;
            }
            9 => {
                button(raw, along(self.test.cpick_at.0, 0.7), false);
                next = 10;
            }
            10 if frames > 2 => {
                let v = line(self);
                if v.iter().any(|x| !(0.0..=1.0).contains(x)) {
                    fail(format!("the picker wrote a colour outside 0..1: {v:?}"));
                }
                let l = lightness(v);
                if (l - 0.7).abs() > 0.01 {
                    fail(format!(
                        "released the L lane 70% along but L is {l:.4} — the knob is \
                         not under the pointer, or the working-space round trip is off"
                    ));
                }
                if v[3] != self.test.cpick_before[3] {
                    fail("an L drag moved alpha".into());
                }
                key(raw, egui::Key::V, true);
                next = 11;
            }
            // Relative at a held rate: V goes down BEFORE the press, so the
            // press must not jump the knob, and 2% of travel moves L by 2% at
            // V's rate (read from THE table, which gets retuned). V is also
            // the props-bar toggle — if the popup did not claim it, the panel
            // the popup lives in vanishes right here.
            11 if frames > 2 => {
                button(raw, along(self.test.cpick_at.0, 0.2), true);
                next = 12;
            }
            12 | 13 => {
                let t = 0.2 + 0.01 * (stage - 11) as f32;
                raw.events.push(egui::Event::PointerMoved(along(self.test.cpick_at.0, t)));
                next = stage + 1;
            }
            14 => {
                button(raw, along(self.test.cpick_at.0, 0.22), false);
                key(raw, egui::Key::V, false);
                next = 15;
            }
            15 if frames > 2 => {
                if !self.props_visible {
                    fail("holding V over the picker hid the properties bar — the \
                          popup did not claim its rate keys"
                        .into());
                }
                let v_rate = rgbslider::RATE_KEYS
                    .iter()
                    .find(|(k, _)| *k == egui::Key::V)
                    .map_or(1.0, |&(_, f)| f);
                let want = (0.7 + 0.02 * v_rate).min(1.0);
                let l = lightness(line(self));
                if (l - want).abs() > 0.02 {
                    fail(format!(
                        "a V-held drag left L at {l:.4}; want {want:.2} (no jump on \
                         the press, then 2% of the track at x{v_rate})"
                    ));
                }
                let Some(g) = geom.filter(|g| g.open) else {
                    fail("the picker closed during the V drag".into())
                };
                if !hold {
                    self.test.cpick_at = (g.old_chip, 0);
                    button(raw, g.old_chip.center(), true);
                    next = 16;
                }
            }
            16 => {
                button(raw, self.test.cpick_at.0.center(), false);
                next = 17;
            }
            // Then the right-edge grip, dragged 172pt in — about a unit of
            // scale. Shrinking always fits, so the edge must follow the
            // pointer while the swatch-side edges stay where they are.
            17 if frames > 2 => {
                let v = line(self);
                if v != self.test.cpick_before {
                    fail(format!(
                        "the old chip did not restore: {v:?}, want {:?}",
                        self.test.cpick_before
                    ));
                }
                let Some(g) = geom.filter(|g| g.open) else {
                    fail("the old chip closed the picker".into())
                };
                self.test.cpick_scale = g.scale;
                self.test.cpick_at = (g.popup, 0);
                // Inside the frame margin, where the grip has no widget above it.
                button(raw, egui::pos2(g.popup.right() - 3.0, g.popup.center().y), true);
                next = 18;
            }
            18..=20 => {
                let r = self.test.cpick_at.0;
                let x = r.right() - 3.0 - 172.0 * (stage - 17) as f32 / 3.0;
                raw.events.push(egui::Event::PointerMoved(egui::pos2(x, r.center().y)));
                next = stage + 1;
            }
            21 => {
                let r = self.test.cpick_at.0;
                button(raw, egui::pos2(r.right() - 175.0, r.center().y), false);
                next = 22;
            }
            22 if frames > 2 => {
                let Some(g) = geom.filter(|g| g.open) else {
                    fail("the picker closed during a resize".into())
                };
                let (was, s0) = (self.test.cpick_at.0, self.test.cpick_scale);
                if !(s0.x - 1.2..s0.x - 0.8).contains(&g.scale.x) {
                    fail(format!(
                        "dragging the right edge 172pt in took the width scale {} to {}",
                        s0.x, g.scale.x
                    ));
                }
                // The axes are independent: the edge that was not dragged
                // must not have moved a thing.
                if (g.scale.y - s0.y).abs() > 1e-3 {
                    fail(format!(
                        "the right edge changed the HEIGHT scale too: {} -> {}",
                        s0.y, g.scale.y
                    ));
                }
                if (g.popup.right() - (was.right() - 172.0)).abs() > 2.5 {
                    fail(format!(
                        "the right edge ended at {:.1}, not under the pointer at {:.1}",
                        g.popup.right(),
                        was.right() - 172.0
                    ));
                }
                let above = g.popup.bottom() <= g.swatch.top();
                let facing = if above {
                    (g.popup.bottom() - was.bottom()).abs()
                } else {
                    (g.popup.top() - was.top()).abs()
                };
                if (g.popup.left() - was.left()).abs() > 0.5 || facing > 0.5 {
                    fail(format!("a resize moved the swatch-side edges: {was:?} -> {:?}", g.popup));
                }
                self.test.cpick_scale = g.scale;
                // The OTHER axis, then a move, before the prefs wait. The
                // free edge is the one AWAY from the swatch, so which one
                // that is travels with the rect.
                self.test.cpick_at = (g.popup, u32::from(above));
                next = 70;
            }
            // The free horizontal edge, 60pt IN: the height, and nothing
            // else. Inward, because shrinking always fits — outward stops
            // wherever the room on that side of the swatch runs out.
            70..=74 => {
                let (r, above) = (self.test.cpick_at.0, self.test.cpick_at.1 == 1);
                let (y0, dir) =
                    if above { (r.top() + 3.0, 1.0) } else { (r.bottom() - 3.0, -1.0) };
                // Full travel by the frame BEFORE the release: a drag
                // that moves and lets go in one frame never applies its
                // last hop (see --test-spline's press-anchoring rule).
                let y = y0 + dir * 60.0 * ((stage - 70) as f32 / 3.0).min(1.0);
                match stage {
                    70 => button(raw, egui::pos2(r.center().x, y), true),
                    74 => button(raw, egui::pos2(r.center().x, y), false),
                    _ => raw.events.push(egui::Event::PointerMoved(egui::pos2(r.center().x, y))),
                }
                next = stage + 1;
            }
            75 if frames > 2 => {
                let Some(g) = geom.filter(|g| g.open) else {
                    fail("the picker closed during the height resize".into())
                };
                let (was, s0) = (self.test.cpick_at.0, self.test.cpick_scale);
                if g.scale.y >= s0.y - 0.2 {
                    fail(format!(
                        "dragging the free edge 60pt in left the height scale at {} (was {})",
                        g.scale.y, s0.y
                    ));
                }
                if (g.scale.x - s0.x).abs() > 1e-3 {
                    fail(format!(
                        "the free edge changed the WIDTH scale too: {} -> {}",
                        s0.x, g.scale.x
                    ));
                }
                if (g.popup.width() - was.width()).abs() > 2.0 {
                    fail(format!(
                        "the free edge changed the popup's width: {:.1} -> {:.1}",
                        was.width(),
                        g.popup.width()
                    ));
                }
                let grew = was.height() - g.popup.height();
                if (grew - 60.0).abs() > 8.0 {
                    fail(format!(
                        "the free edge moved {grew:.1}pt, not the 60 the pointer did \
                         (height scale {} -> {}, popup {:.1} -> {:.1} tall)",
                        s0.y,
                        g.scale.y,
                        was.height(),
                        g.popup.height()
                    ));
                }
                // ...and the edge the swatch faces stayed put.
                let facing = if self.test.cpick_at.1 == 1 {
                    (g.popup.bottom() - was.bottom()).abs()
                } else {
                    (g.popup.top() - was.top()).abs()
                };
                if facing > 1.0 {
                    fail(format!("a height resize moved the swatch-side edge by {facing:.1}pt"));
                }
                // Now MOVE it: press the left frame margin, where no widget
                // and no grip sits, and drag up and left.
                self.test.cpick_at = (g.popup, 0);
                button(raw, egui::pos2(g.popup.left() + 3.0, g.popup.center().y), true);
                next = 76;
            }
            76..=78 => {
                let r = self.test.cpick_at.0;
                let k = (stage - 75) as f32 / 3.0;
                let p = egui::pos2(r.left() + 3.0 - 40.0 * k, r.center().y - 30.0 * k);
                raw.events.push(egui::Event::PointerMoved(p));
                next = stage + 1;
            }
            79 => {
                let r = self.test.cpick_at.0;
                button(raw, egui::pos2(r.left() - 37.0, r.center().y - 30.0), false);
                next = 80;
            }
            80 if frames > 2 => {
                let Some(g) = geom.filter(|g| g.open) else {
                    fail("dragging the picker's chrome closed it".into())
                };
                let was = self.test.cpick_at.0;
                let moved = g.popup.min - was.min;
                if (moved - egui::vec2(-40.0, -30.0)).length() > 4.0 {
                    fail(format!(
                        "dragging the chrome moved the popup by {moved:?}, not the (-40, -30) \
                         the pointer travelled"
                    ));
                }
                if (g.popup.size() - was.size()).length() > 1.0 {
                    fail("moving the popup resized it".into());
                }
                let screen = self.egui_ctx.content_rect();
                if !screen.contains_rect(g.popup) {
                    fail(format!("the moved popup left the screen: {:?}", g.popup));
                }
                self.test.cpick_scale = g.scale;
                self.test.cpick_t0 = web_time::Instant::now();
                next = 23;
            }
            // The size reaches prefs once the gesture settles (wall clock).
            23 => {
                let s = colorpicker::scale();
                let landed = self.prefs.picker_scale.is_some_and(|p| (p - s.x).abs() < 1e-5)
                    && self.prefs.picker_scale_y.is_some_and(|p| (p - s.y).abs() < 1e-5);
                if landed {
                    key(raw, egui::Key::Escape, true);
                    next = 24;
                } else if self.test.cpick_t0.elapsed().as_secs_f32() > 3.0 {
                    fail(format!(
                        "the resize never reached prefs: {:?}/{:?} with the picker at {s:?}",
                        self.prefs.picker_scale, self.prefs.picker_scale_y
                    ));
                }
            }
            24 => {
                key(raw, egui::Key::Escape, false);
                next = 25;
            }
            25 if frames > 3 => {
                let Some(g) = geom.filter(|g| !g.open) else {
                    fail("Escape did not close the picker".into())
                };
                self.test.cpick_at = (g.swatch, 0);
                button(raw, g.swatch.center(), true);
                next = 26;
            }
            26 => {
                button(raw, self.test.cpick_at.0.center(), false);
                next = 27;
            }
            // Reopened at the size it was left at; then a pinch over it.
            27 if frames > 3 => {
                let Some(g) = geom.filter(|g| g.open) else {
                    fail("the swatch did not reopen the picker".into())
                };
                if (g.scale - self.test.cpick_scale).length() > 1e-3 {
                    fail(format!(
                        "reopened at scale {:?}, not the {:?} it was resized to",
                        g.scale, self.test.cpick_scale
                    ));
                }
                self.test.cpick_zoom = (self.view_zoom, self.node_editor.zoom);
                raw.events.push(egui::Event::PointerMoved(g.popup.center()));
                raw.events.push(egui::Event::Zoom(1.25));
                next = 28;
            }
            28 if frames > 2 => {
                let Some(g) = geom.filter(|g| g.open) else {
                    fail("a pinch closed the picker".into())
                };
                let want = self.test.cpick_scale * 1.25;
                if (colorpicker::scale() - want).length() > 0.02
                    || (g.scale - want).length() > 0.02
                {
                    fail(format!(
                        "pinching x1.25 over the picker left it at {:?} (app {:?}), want {want:?}",
                        g.scale,
                        colorpicker::scale()
                    ));
                }
                if (self.view_zoom, self.node_editor.zoom) != self.test.cpick_zoom {
                    fail("pinching over the picker also zoomed the viewer or graph under it".into());
                }
                let at = corner_in(&g);
                self.test.cpick_at = (egui::Rect::from_center_size(at, egui::Vec2::ZERO), 0);
                button(raw, at, true);
                next = 29;
            }
            // Double-click the corner grip: the default size.
            29..=31 => {
                button(raw, self.test.cpick_at.0.center(), stage == 30);
                next = stage + 1;
            }
            32 if frames > 2 => {
                let Some(g) = geom.filter(|g| g.open) else {
                    fail("double-clicking the corner grip closed the picker".into())
                };
                if colorpicker::scale() != colorpicker::scale_default()
                    || (g.scale - colorpicker::scale_default()).length() > 1e-3
                {
                    fail(format!(
                        "double-clicking the corner grip left the picker at {:?} (app {:?})",
                        g.scale,
                        colorpicker::scale()
                    ));
                }
                // The palette: hermetic prefs, so it starts empty and its
                // only slot is +. Keep the current colour.
                if !colorpicker::swatches().is_empty() || g.swatch_count != 0 {
                    fail(format!("the palette did not start empty: {:?}", colorpicker::swatches()));
                }
                self.test.cpick_at = (g.swatch_rect(0), 0);
                button(raw, g.swatch_rect(0).center(), true);
                next = 33;
            }
            // Every palette click is press, release, then a look.
            33 | 35 | 37 | 39 | 45 | 49 | 51 => {
                button(raw, self.test.cpick_at.0.center(), false);
                next = stage + 1;
            }
            34 if frames > 2 => {
                let Some(g) = geom.filter(|g| g.open) else {
                    fail("keeping a swatch closed the picker".into())
                };
                let (sw, want) = (colorpicker::swatches(), as709(line(self)));
                if sw.len() != 1 || !near(sw[0], want) || g.swatch_count != 1 {
                    fail(format!("+ kept {sw:?}, want the current colour {want:?} in Rec.709"));
                }
                if self.prefs.picker_swatches != sw {
                    fail(format!("the swatch never reached prefs: {:?}", self.prefs.picker_swatches));
                }
                // Move away from it: a click on the L lane jumps there.
                self.test.cpick_before = line(self);
                self.test.cpick_at = (egui::Rect::from_center_size(along(g.lanes[0], 0.3), egui::Vec2::ZERO), 0);
                button(raw, self.test.cpick_at.0.center(), true);
                next = 35;
            }
            36 if frames > 2 => {
                let Some(g) = geom.filter(|g| g.open) else { fail("the picker closed".into()) };
                if (lightness(line(self)) - 0.3).abs() > 0.01 {
                    fail(format!("a click on the L lane left L at {:.3}", lightness(line(self))));
                }
                self.test.cpick_at = (g.swatch_rect(0), 0);
                button(raw, g.swatch_rect(0).center(), true);
                next = 37;
            }
            // Using the swatch: its colour, the value's own alpha.
            38 if frames > 2 => {
                let Some(g) = geom.filter(|g| g.open) else { fail("using a swatch closed the picker".into()) };
                let (v, b) = (line(self), self.test.cpick_before);
                if !near([v[0], v[1], v[2]], [b[0], b[1], b[2]]) || v[3] != b[3] {
                    fail(format!("using the swatch gave {v:?}, want the kept colour {b:?}"));
                }
                self.test.cpick_at = (egui::Rect::from_center_size(along(g.lanes[0], 0.5), egui::Vec2::ZERO), 0);
                button(raw, self.test.cpick_at.0.center(), true);
                next = 39;
            }
            // Shift-click replaces the swatch with the current colour.
            40 if frames > 2 => {
                let Some(g) = geom.filter(|g| g.open) else { fail("the picker closed".into()) };
                self.test.cpick_before = line(self);
                self.test.cpick_at = (g.swatch_rect(0), 0);
                raw.events.insert(0, egui::Event::ModifiersChanged(egui::Modifiers::SHIFT));
                button(raw, g.swatch_rect(0).center(), true);
                next = 41;
            }
            41 => {
                raw.events.insert(0, egui::Event::ModifiersChanged(egui::Modifiers::SHIFT));
                button(raw, self.test.cpick_at.0.center(), false);
                next = 42;
            }
            42 if frames > 2 => {
                let Some(g) = geom.filter(|g| g.open) else { fail("the picker closed".into()) };
                let (sw, v) = (colorpicker::swatches(), line(self));
                if sw.len() != 1 || !near(sw[0], as709(v)) {
                    fail(format!("shift-click left the palette at {sw:?}, want the current {:?}", as709(v)));
                }
                if v != self.test.cpick_before {
                    fail(format!("shift-click on a swatch changed the value: {v:?}"));
                }
                self.test.cpick_at = (g.swatch_rect(0), 0);
                raw.events.insert(0, egui::Event::ModifiersChanged(egui::Modifiers::ALT));
                button(raw, g.swatch_rect(0).center(), true);
                next = 43;
            }
            // Alt-click removes it.
            43 => {
                raw.events.insert(0, egui::Event::ModifiersChanged(egui::Modifiers::ALT));
                button(raw, self.test.cpick_at.0.center(), false);
                next = 44;
            }
            44 if frames > 2 => {
                let Some(g) = geom.filter(|g| g.open) else { fail("alt-click closed the picker".into()) };
                if !colorpicker::swatches().is_empty() || !self.prefs.picker_swatches.is_empty() {
                    fail(format!(
                        "alt-click left {:?} (prefs {:?})",
                        colorpicker::swatches(),
                        self.prefs.picker_swatches
                    ));
                }
                // Keep one more, for the app-wide check on another swatch.
                self.test.cpick_at = (g.swatch_rect(0), 0);
                button(raw, g.swatch_rect(0).center(), true);
                next = 45;
            }
            46 if frames > 2 => {
                if colorpicker::swatches().len() != 1 {
                    fail("+ did not keep the colour the second time".into());
                }
                key(raw, egui::Key::Escape, true);
                next = 47;
            }
            47 => {
                key(raw, egui::Key::Escape, false);
                next = 48;
            }
            // App-wide: the Grid's OTHER swatch offers the same palette.
            48 if frames > 3 => {
                if geom.is_some_and(|g| g.open) {
                    fail("Escape did not close the picker".into());
                }
                let Some(gb) = geom_back else { fail("the Grid's back swatch is not drawn".into()) };
                self.test.cpick_before = back(self);
                self.test.cpick_at = (gb.swatch, 0);
                button(raw, gb.swatch.center(), true);
                next = 49;
            }
            50 if frames > 3 => {
                let Some(gb) = geom_back.filter(|g| g.open) else {
                    fail("the back swatch did not open a picker".into())
                };
                if gb.swatch_count != 1 {
                    fail(format!("the back picker offers {} swatches, not the app's 1", gb.swatch_count));
                }
                self.test.cpick_at = (gb.swatch_rect(0), 0);
                button(raw, gb.swatch_rect(0).center(), true);
                next = 51;
            }
            52 if frames > 2 => {
                let (b, l, was) = (back(self), line(self), self.test.cpick_before);
                if !near([b[0], b[1], b[2]], [l[0], l[1], l[2]]) || b[3] != was[3] {
                    fail(format!(
                        "the swatch put {b:?} on the back colour, want the line's colour {l:?} with alpha {}",
                        was[3]
                    ));
                }
                key(raw, egui::Key::Escape, true);
                next = 53;
            }
            53 => {
                key(raw, egui::Key::Escape, false);
                next = 54;
            }
            54 if frames > 3 => {
                if geom_back.is_some_and(|g| g.open) {
                    fail("Escape did not close the back picker".into());
                }
                // The HUD chip: the selected Grid has one at the viewer's left
                // edge, and its swatch is the same picker. The click lands on
                // the NAME, well right of the dot — the dot is too small to be
                // the target, so the reach across the name is the feature.
                self.selected = vec![id];
                next = 55;
            }
            55 if frames > 3 => {
                let Some(c) = chip else { return };
                let at = egui::pos2(c.swatch.right() + 12.0, c.swatch.center().y);
                self.test.cpick_at = (c.swatch, 0);
                button(raw, at, true);
                next = 56;
            }
            56 if frames > 2 => {
                button(raw, self.test.cpick_at.0.right_center() + egui::vec2(12.0, 0.0), false);
                next = 57;
            }
            57 if frames > 3 => {
                let Some(c) = chip.filter(|g| g.open) else {
                    fail("clicking the HUD chip's name did not open its picker".into())
                };
                if c.swatch_count != 1 {
                    fail(format!("the chip's picker offers {} swatches, not the app's 1", c.swatch_count));
                }
                self.test.cpick_at = (c.swatch_rect(0), 0);
                button(raw, c.swatch_rect(0).center(), true);
                next = 58;
            }
            58 => {
                button(raw, self.test.cpick_at.0.center(), false);
                next = 59;
            }
            59 if frames > 2 => {
                // A swatch past the gamut clamps on use (the palette kept the
                // ACEScg line colour, which is outside Rec.709's).
                let want = colorpicker::swatches()[0].map(|x| x.clamp(0.0, 1.0));
                match self.egraph.node(id).and_then(|n| n.hud_color) {
                    Some(c) if near(c, want) => {}
                    other => fail(format!(
                        "the palette swatch put {other:?} on the node's HUD colour, want {want:?}"
                    )),
                }
                key(raw, egui::Key::Escape, true);
                next = 60;
            }
            60 => {
                key(raw, egui::Key::Escape, false);
                next = 61;
            }
            61 if frames > 3 => {
                let Some(c) = chip else { fail("the chip vanished".into()) };
                if c.open {
                    fail(format!(
                        "Escape did not close the chip's picker (selected {:?}, normal {:?}, hud_hide {:?}, chip swatch {:?}, popup {:?})",
                        self.selected, self.viewer_mode == ViewerMode::Normal, self.prefs.hud_hide, c.swatch, c.popup
                    ));
                }
                // Right-click on the dot: back to the adaptive ink.
                let at = c.swatch.center();
                for pressed in [true, false] {
                    raw.events.push(egui::Event::PointerMoved(at));
                    raw.events.push(egui::Event::PointerButton {
                        pos: at,
                        button: egui::PointerButton::Secondary,
                        pressed,
                        modifiers: Default::default(),
                    });
                }
                next = 62;
            }
            62 if frames > 3 => {
                if let Some(c) = self.egraph.node(id).and_then(|n| n.hud_color) {
                    fail(format!("right-click left the HUD colour at {c:?}"));
                }
                if chip.is_some_and(|g| g.open) {
                    fail("a right-click opened the chip's picker".into());
                }
                println!(
                    "CPICK TEST PASS: opened at the default size, the lch L lane \
                     tracked the pointer to L 0.70 through the working space, a \
                     V-held drag went relative at V's rate without hiding the panel, \
                     the old chip restored, the right edge resized it under the \
                     pointer with the swatch side pinned and the HEIGHT untouched, \
                     the free edge did the same to the height alone, a drag on the \
                     chrome MOVED it, the size reached prefs \
                     and survived a reopen, a pinch scaled it without zooming \
                     what is under it, a double-click restored the default; the \
                     palette kept a colour into prefs, used it (alpha kept), \
                     shift-click replaced it, alt-click removed it, and the Grid's \
                     other swatch offered the same palette and took its colour; \
                     the HUD chip's NAME opened the same picker, a palette click \
                     landed on the node's HUD colour, and a right-click cleared it"
                );
                std::process::exit(0);
            }
            _ => {}
        }
        self.test.cpick = Some((next, if next == stage { frames } else { 0 }, id));
    }
}
