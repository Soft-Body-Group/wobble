//! `--test-keymap`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-keymap: the hotkey remapper end to end in the LIVE app.
    ///
    /// Parsing, matching and conflicts are unit-tested in `keymap`; what
    /// only a live run shows is the wiring — that a rebind is honoured by
    /// the app's real input path the SAME frame (old key inert, new key
    /// firing), and that the keys window's capture takes a real key event
    /// while the ordinary hotkeys stand down for the press.
    pub(crate) fn drive_test_keymap(&mut self, raw: &mut egui::RawInput) {
        use keymap::Act;
        let Some((stage, mut frames)) = self.test_keymap else { return };
        frames += 1;
        if frames > 2400 {
            println!("KEYMAP TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let fail = |m: String| -> ! {
            println!("KEYMAP TEST FAIL: {m}");
            std::process::exit(1);
        };
        // Press AND release in one frame: `pressed` reads the down event,
        // and the release keeps egui's keys_down bookkeeping clean.
        let key = |raw: &mut egui::RawInput, k: egui::Key| {
            for pressed in [true, false] {
                raw.events.push(egui::Event::Key {
                    key: k,
                    physical_key: None,
                    pressed,
                    repeat: false,
                    modifiers: Default::default(),
                });
            }
        };
        // Where each pane sits in the main window: the leaf it is in.
        static DOCK_SNAP: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
        fn dock_snap(g: &Gfx) -> String {
            use dockui::Pane;
            let Some(d) = g.docks.as_ref() else { return String::new() };
            let t = &d.hosts[dockui::MAIN].tree;
            format!("{:?}", [Pane::Nodes, Pane::Viewer, Pane::Timeline, Pane::Properties, Pane::Curves].map(|p| t.find(p)))
        }
        // True once `done`, or once two seconds have gone without it (the
        // stage then judges, and fails); false while still waiting.
        static GLIDE: std::sync::Mutex<Option<web_time::Instant>> = std::sync::Mutex::new(None);
        fn glided(done: bool) -> bool {
            let mut g = GLIDE.lock().unwrap();
            let t0 = *g.get_or_insert_with(web_time::Instant::now);
            if done || t0.elapsed().as_secs_f32() > 2.0 {
                *g = None;
                return true;
            }
            false
        }
        fn same_dock(g: &Gfx, what: &str) -> Option<String> {
            let was = DOCK_SNAP.lock().unwrap().clone().unwrap_or_default();
            let now = dock_snap(g);
            (now != was).then(|| format!("{what} left the layout as {now}, not {was}"))
        }
        let mut next = stage;
        match stage {
            // Settle, then hit the default Y: the curve editor comes up.
            0 if frames >= 6 => {
                if self.bottom == BottomTab::Curves {
                    fail("the bar opened on curves before any key".into());
                }
                key(raw, egui::Key::Y);
                next = 1;
            }
            1 => {
                if self.bottom != BottomTab::Curves {
                    fail("default Y did not open the curve editor".into());
                }
                // Rebind through the same entry the window's rows use.
                keymap::set(Act::CurveEditor, keymap::Binding::plain(egui::Key::K));
                key(raw, egui::Key::Y);
                next = 2;
            }
            2 => {
                if self.bottom != BottomTab::Curves {
                    fail("Y still fired after being rebound away".into());
                }
                key(raw, egui::Key::K);
                next = 3;
            }
            3 => {
                if self.bottom != BottomTab::Properties {
                    fail("the rebound key K did not fire".into());
                }
                // The capture path: arm the keys window on graph tidy and
                // press Y.
                self.keys_ui.open = true;
                self.keys_ui.begin_capture(Act::GraphTidy);
                next = 4;
            }
            4 => {
                // The window drew once with the capture armed; the guard
                // must be up before the key lands.
                if !keymap::capturing() {
                    fail("an armed capture does not report capturing()".into());
                }
                key(raw, egui::Key::Y);
                next = 5;
            }
            5 => {
                if keymap::binding(Act::GraphTidy) != keymap::Binding::plain(egui::Key::Y) {
                    fail(format!(
                        "capture did not take Y: {:?}",
                        keymap::binding(Act::GraphTidy)
                    ));
                }
                if keymap::capturing() {
                    fail("capture still live after the key came back up".into());
                }
                // Same-scope conflict is flagged…
                keymap::set(Act::GraphSwapInputs, keymap::Binding::plain(egui::Key::Y));
                let c = keymap::conflicts();
                if !c.iter().any(|(a, b)| {
                    (*a == Act::GraphTidy && *b == Act::GraphSwapInputs)
                        || (*a == Act::GraphSwapInputs && *b == Act::GraphTidy)
                }) {
                    fail(format!("tidy and swap on one key not flagged: {c:?}"));
                }
                // …and the prefs strings round-trip.
                let saved = keymap::overrides();
                keymap::reset_all();
                if keymap::binding(Act::GraphTidy)
                    != keymap::Binding::plain(egui::Key::T)
                {
                    fail("reset all did not restore the default".into());
                }
                keymap::install(&saved);
                if keymap::binding(Act::GraphTidy) != keymap::Binding::plain(egui::Key::Y)
                    || keymap::binding(Act::GraphSwapInputs)
                        != keymap::Binding::plain(egui::Key::Y)
                {
                    fail("overrides did not round-trip through prefs strings".into());
                }
                keymap::reset_all();
                next = 100;
            }
            // Q fullscreens the pane under the cursor, ALONE: the graph over
            // the graph, the properties over the properties, the image over
            // the image. Q again, or Esc, puts the layout back.
            100 if frames > 3 => {
                *DOCK_SNAP.lock().unwrap() = Some(dock_snap(self));
                raw.events.push(egui::Event::PointerMoved(self.graph_panel_rect.center()));
                key(raw, egui::Key::Q);
                next = 101;
            }
            101 => {
                if self.viewer_mode != ViewerMode::FullGraph {
                    fail(format!("Q over the graph gave {:?}, not the graph alone", self.viewer_mode));
                }
                next = 102;
            }
            // The dock GLIDES the leaf out to the whole window: wait for
            // it on the wall clock (the test loop is unpaced).
            102 if !glided(self.graph_panel_rect.width() >= 0.8 * self.egui_ctx.content_rect().width()) => {}
            102 => {
                let screen = self.egui_ctx.content_rect();
                if self.graph_panel_rect.width() < 0.8 * screen.width() {
                    fail(format!(
                        "the fullscreen graph is {:.0} wide in a {:.0} window",
                        self.graph_panel_rect.width(),
                        screen.width()
                    ));
                }
                raw.events.push(egui::Event::PointerMoved(self.graph_panel_rect.center()));
                key(raw, egui::Key::Q);
                next = 103;
            }
            103 => {
                if self.viewer_mode != ViewerMode::Normal || !self.props_visible {
                    fail(format!("Q again gave {:?}, not the layout", self.viewer_mode));
                }
                next = 1031;
            }
            // The dock applies its asks between frames: judge the layout
            // a frame later. Every pane back in the leaf it started in.
            1031 => {
                if let Some(m) = same_dock(self, "Q, Q") {
                    fail(m);
                }
                next = 104;
            }
            // ...and glides the layout back: aim at the properties once
            // the graph has let go of them.
            104 if !glided(!self.graph_panel_rect.contains(self.bottom_rect.center())) => {}
            104 => {
                raw.events.push(egui::Event::PointerMoved(self.bottom_rect.center()));
                key(raw, egui::Key::Q);
                next = 105;
            }
            105 => {
                if self.viewer_mode != ViewerMode::FullProps {
                    fail(format!("Q over the properties gave {:?}", self.viewer_mode));
                }
                next = 106;
            }
            106 if !glided(self.bottom_rect.height() >= 0.8 * self.egui_ctx.content_rect().height()) => {}
            106 => {
                let screen = self.egui_ctx.content_rect();
                if self.bottom_rect.height() < 0.8 * screen.height() {
                    fail(format!(
                        "the fullscreen properties are {:.0} tall in a {:.0} window",
                        self.bottom_rect.height(),
                        screen.height()
                    ));
                }
                key(raw, egui::Key::Escape);
                next = 107;
            }
            107 => {
                if self.viewer_mode != ViewerMode::Normal {
                    fail(format!("Esc gave {:?}, not the layout", self.viewer_mode));
                }
                next = 1071;
            }
            1071 => {
                if let Some(m) = same_dock(self, "Q over the properties, Esc") {
                    fail(m);
                }
                next = 108;
            }
            108 if !glided({
                let c = self.viewer_rect.center();
                !self.graph_panel_rect.contains(c) && !self.bottom_rect.contains(c)
            }) => {}
            108 => {
                raw.events.push(egui::Event::PointerMoved(self.viewer_rect.center()));
                key(raw, egui::Key::Q);
                next = 109;
            }
            109 => {
                if self.viewer_mode != ViewerMode::Full {
                    fail(format!("Q over the image gave {:?}", self.viewer_mode));
                }
                key(raw, egui::Key::Q);
                next = 110;
            }
            110 => {
                if self.viewer_mode != ViewerMode::Normal {
                    fail(format!("Q out of the image gave {:?}", self.viewer_mode));
                }
                next = 111;
            }
            // W closes the properties and the animation; W again opens
            // them where they were, not wherever a new tab lands.
            111 => {
                if let Some(m) = same_dock(self, "Q, Q over the image") {
                    fail(m);
                }
                raw.events.push(egui::Event::PointerMoved(self.viewer_rect.center()));
                key(raw, egui::Key::W);
                next = 112;
            }
            112 => {
                if self.props_visible {
                    fail("W did not close the properties".into());
                }
                key(raw, egui::Key::W);
                next = 113;
            }
            113 => {
                if !self.props_visible {
                    fail("W again did not open the properties".into());
                }
                next = 117;
            }
            117 => {
                if let Some(m) = same_dock(self, "W, W") {
                    fail(m);
                }
                // The hotkeys window: a key pressed over it finds its row
                // and does NOT do what it is bound to.
                self.help_open = true;
                next = 118;
            }
            118 if self.help_rect.is_none() || !self.layout_settled() => {}
            118 => {
                let r = self.help_rect.unwrap();
                raw.events.push(egui::Event::PointerMoved(r.center()));
                next = 119;
            }
            119 => {
                key(raw, egui::Key::Q);
                next = 120;
            }
            120 => {
                if self.viewer_mode != ViewerMode::Normal {
                    fail(format!("Q over the hotkeys window fired: {:?}", self.viewer_mode));
                }
                let want = keymap::label(&self.egui_ctx, Act::FullscreenPane);
                match self.help_jump {
                    Some((egui::Key::Q, 0, row, _)) if row == 0 => {}
                    j => fail(format!("Q over the hotkeys window jumped to {j:?}, not the {want} row")),
                }
                self.help_open = false;
                println!(
                    "KEYMAP TEST PASS: a rebind is honoured live (old key inert, new \
                     key fires), the keys window's capture takes a real key event and \
                     stands the hotkeys down while armed, same-scope conflicts are \
                     flagged, overrides round-trip through the prefs strings, Q \
                     fullscreens the graph, the properties or the image alone, and \
                     Q again and W twice put every pane back in its own leaf, and a key \
                     pressed over the hotkeys window finds its row without firing"
                );
                std::process::exit(0);
            }
            _ => {}
        }
        self.test_keymap = Some((next, frames));
    }
}
