//! `--test-blendnav`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-blendnav driver: Blender's navigation chords (manifesto 11)
    /// against the REAL panels — graph MMB pan, ctrl+MMB drag-zoom, alt+LMB
    /// pan STARTING ON A NODE (the chord outranks the node drag), then the
    /// wheel family (Line = mouse wheel zooms, Point = trackpad pans,
    /// ctrl+wheel pans sideways), then the viewer's alt+LMB pan and
    /// ctrl+MMB zoom. Everything is asserted relative to a baseline packed
    /// into the state tuple, so the suite is indifferent to window size.
    pub(crate) fn drive_test_blendnav(&mut self, raw: &mut egui::RawInput) {
        let Some((stage, mut frames, base_pan, mut base_zoom)) = self.test.blendnav else {
            return;
        };
        frames += 1;
        if frames > 1800 {
            println!("BLENDNAV TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let fail = |msg: String| -> ! {
            println!("BLENDNAV TEST FAIL: {msg}");
            std::process::exit(1);
        };
        let pack2 = |a: f32, b: f32| ((a.to_bits() as u64) << 32) | b.to_bits() as u64;
        let unpack2 =
            |v: u64| (f32::from_bits((v >> 32) as u32), f32::from_bits(v as u32));
        // A background spot on the graph panel, clear of the default nodes
        // (they cluster around the panel's centre).
        let bg = {
            let r = self.graph_panel_rect;
            egui::pos2(r.min.x + 40.0, r.min.y + 40.0)
        };
        let button = |raw: &mut egui::RawInput,
                      p: egui::Pos2,
                      b: egui::PointerButton,
                      down: bool,
                      m: egui::Modifiers| {
            raw.events.insert(0, egui::Event::ModifiersChanged(m));
            raw.events.push(egui::Event::PointerMoved(p));
            raw.events.push(egui::Event::PointerButton {
                pos: p,
                button: b,
                pressed: down,
                modifiers: m,
            });
        };
        let mv = |raw: &mut egui::RawInput, p: egui::Pos2, m: egui::Modifiers| {
            raw.events.insert(0, egui::Event::ModifiersChanged(m));
            raw.events.push(egui::Event::PointerMoved(p));
        };
        let wheel = |raw: &mut egui::RawInput,
                     unit: egui::MouseWheelUnit,
                     d: egui::Vec2,
                     m: egui::Modifiers| {
            raw.events.insert(0, egui::Event::ModifiersChanged(m));
            raw.events.push(egui::Event::MouseWheel {
                unit,
                delta: d,
                modifiers: m,
                phase: egui::TouchPhase::Move,
            });
        };
        const NONE: egui::Modifiers = egui::Modifiers::NONE;
        const CTRL: egui::Modifiers = egui::Modifiers::CTRL;
        const ALT: egui::Modifiers = egui::Modifiers::ALT;
        let ctrl_alt = egui::Modifiers { ctrl: true, alt: true, ..NONE };
        let mid = egui::PointerButton::Middle;
        let prim = egui::PointerButton::Primary;
        let mut next = stage;
        let set_base = |s: &Self| {
            (pack2(s.node_editor.pan.x, s.node_editor.pan.y), pack2(s.node_editor.zoom, 0.0))
        };
        match stage {
            0 => {
                if self.layout_settled() && frames >= 6 {
                    let (p, z) = set_base(self);
                    self.test.blendnav = Some((1, frames, p, z));
                    return;
                }
            }
            // --- graph: MMB drag pans ---
            1 => {
                button(raw, bg, mid, true, NONE);
                next = 2;
            }
            2 => {
                mv(raw, bg + egui::vec2(60.0, 25.0), NONE);
                next = 3;
            }
            3 => {
                button(raw, bg + egui::vec2(60.0, 25.0), mid, false, NONE);
                next = 4;
            }
            4 => {
                let (bx, by) = unpack2(base_pan);
                let (bz, _) = unpack2(base_zoom);
                let d = self.node_editor.pan - egui::vec2(bx, by);
                if (d - egui::vec2(60.0, 25.0)).length() > 3.0 {
                    fail(format!("MMB pan moved {d:?}, wanted (60, 25)"));
                }
                if (self.node_editor.zoom - bz).abs() > 1e-4 {
                    fail("MMB pan must not zoom".into());
                }
                let (p, z) = set_base(self);
                self.test.blendnav = Some((5, frames, p, z));
                return;
            }
            // --- graph: ctrl+MMB drag-zooms (up = in) ---
            5 => {
                button(raw, bg, mid, true, CTRL);
                next = 6;
            }
            6 => {
                mv(raw, bg + egui::vec2(0.0, -80.0), CTRL);
                next = 7;
            }
            7 => {
                button(raw, bg + egui::vec2(0.0, -80.0), mid, false, CTRL);
                next = 8;
            }
            8 => {
                let (bz, _) = unpack2(base_zoom);
                if self.node_editor.zoom <= bz * 1.2 {
                    fail(format!(
                        "ctrl+MMB up should zoom in ({bz} -> {})",
                        self.node_editor.zoom
                    ));
                }
                // Home the view: the zoom just pushed the node cluster off
                // the panel, and the next gesture starts ON a node.
                self.node_editor.pan = egui::Vec2::ZERO;
                self.node_editor.zoom = 1.0;
                let (p, z) = set_base(self);
                self.test.blendnav = Some((9, frames, p, z));
                return;
            }
            // --- graph: alt+LMB pans even when the drag STARTS ON A NODE ---
            9 => {
                let Some((_, r)) = self.node_rects.iter().find(|(i, _)| *i == 4) else {
                    return;
                };
                button(raw, r.center(), prim, true, ALT);
                next = 10;
            }
            10 => {
                if let Some((_, r)) = self.node_rects.iter().find(|(i, _)| *i == 4) {
                    mv(raw, r.center() + egui::vec2(50.0, 30.0), ALT);
                }
                next = 11;
            }
            11 => {
                if let Some((_, r)) = self.node_rects.iter().find(|(i, _)| *i == 4) {
                    button(raw, r.center() + egui::vec2(50.0, 30.0), prim, false, ALT);
                }
                next = 12;
            }
            12 => {
                let (bx, by) = unpack2(base_pan);
                let d = self.node_editor.pan - egui::vec2(bx, by);
                if (d - egui::vec2(50.0, 30.0)).length() > 3.0 {
                    fail(format!("alt+LMB pan moved {d:?}, wanted (50, 30)"));
                }
                let blend = self.egraph.node(4).map(|n| n.pos);
                if blend != Some(egui::pos2(0.0, 150.0)) {
                    fail(format!(
                        "alt+LMB over the blend node must pan, not move it ({blend:?})"
                    ));
                }
                let (p, z) = set_base(self);
                self.test.blendnav = Some((13, frames, p, z));
                return;
            }
            // --- graph: ctrl+alt+LMB drag-zooms ---
            13 => {
                button(raw, bg, prim, true, ctrl_alt);
                next = 14;
            }
            14 => {
                mv(raw, bg + egui::vec2(0.0, -60.0), ctrl_alt);
                next = 15;
            }
            15 => {
                button(raw, bg + egui::vec2(0.0, -60.0), prim, false, ctrl_alt);
                next = 16;
            }
            16 => {
                let (bz, _) = unpack2(base_zoom);
                if self.node_editor.zoom <= bz * 1.15 {
                    fail(format!(
                        "ctrl+alt+LMB up should zoom in ({bz} -> {})",
                        self.node_editor.zoom
                    ));
                }
                let (p, z) = set_base(self);
                self.test.blendnav = Some((17, frames, p, z));
                return;
            }
            // --- graph wheel family ---
            17 => {
                // A mouse wheel notch zooms.
                mv(raw, bg, NONE);
                wheel(raw, egui::MouseWheelUnit::Line, egui::vec2(0.0, 2.0), NONE);
                next = 18;
            }
            18 => {
                let (bz, _) = unpack2(base_zoom);
                if self.node_editor.zoom <= bz * 1.1 {
                    fail(format!("wheel should zoom ({bz} -> {})", self.node_editor.zoom));
                }
                let (p, z) = set_base(self);
                self.test.blendnav = Some((19, frames, p, z));
                return;
            }
            19 => {
                // Trackpad fingers pan.
                wheel(raw, egui::MouseWheelUnit::Point, egui::vec2(35.0, -18.0), NONE);
                next = 20;
            }
            20 => {
                let (bx, by) = unpack2(base_pan);
                let (bz, _) = unpack2(base_zoom);
                let d = self.node_editor.pan - egui::vec2(bx, by);
                if (d - egui::vec2(35.0, -18.0)).length() > 1.0 {
                    fail(format!("trackpad should pan (35, -18), moved {d:?}"));
                }
                if (self.node_editor.zoom - bz).abs() > 1e-4 {
                    fail("trackpad pan must not zoom".into());
                }
                let (p, z) = set_base(self);
                self.test.blendnav = Some((21, frames, p, z));
                return;
            }
            21 => {
                // ctrl+wheel pans sideways.
                wheel(raw, egui::MouseWheelUnit::Line, egui::vec2(0.0, 2.0), CTRL);
                next = 22;
            }
            22 => {
                let (bx, by) = unpack2(base_pan);
                let (bz, _) = unpack2(base_zoom);
                let d = self.node_editor.pan - egui::vec2(bx, by);
                if d.x.abs() < 20.0 || d.y.abs() > 0.5 {
                    fail(format!("ctrl+wheel should pan sideways only, moved {d:?}"));
                }
                if (self.node_editor.zoom - bz).abs() > 1e-4 {
                    fail("ctrl+wheel must not zoom".into());
                }
                // Viewer baselines ride the same slots from here.
                let vp = pack2(self.view_pan.x, self.view_pan.y);
                let vz = pack2(self.view_zoom, 0.0);
                self.test.blendnav = Some((23, frames, vp, vz));
                return;
            }
            // --- viewer: alt+LMB pans, ctrl+MMB zooms ---
            23 => {
                button(raw, self.viewer_rect.center(), prim, true, ALT);
                next = 24;
            }
            24 => {
                mv(raw, self.viewer_rect.center() + egui::vec2(40.0, 20.0), ALT);
                next = 25;
            }
            25 => {
                button(
                    raw,
                    self.viewer_rect.center() + egui::vec2(40.0, 20.0),
                    prim,
                    false,
                    ALT,
                );
                next = 26;
            }
            26 => {
                let (bx, by) = unpack2(base_pan);
                let d = self.view_pan - egui::vec2(bx, by);
                if (d - egui::vec2(40.0, 20.0)).length() > 3.0 {
                    fail(format!("viewer alt+LMB pan moved {d:?}, wanted (40, 20)"));
                }
                next = 27;
            }
            27 => {
                button(raw, self.viewer_rect.center(), mid, true, CTRL);
                next = 28;
            }
            28 => {
                mv(raw, self.viewer_rect.center() + egui::vec2(0.0, -70.0), CTRL);
                next = 29;
            }
            29 => {
                button(
                    raw,
                    self.viewer_rect.center() + egui::vec2(0.0, -70.0),
                    mid,
                    false,
                    CTRL,
                );
                next = 30;
            }
            30 => {
                let (bz, _) = unpack2(base_zoom);
                if self.view_zoom <= bz * 1.15 {
                    fail(format!(
                        "viewer ctrl+MMB up should zoom in ({bz} -> {})",
                        self.view_zoom
                    ));
                }
                // The scrub chord: hold Tab, drag across the picture. Tab is
                // stripped from egui at the raw level (the search steal), so
                // this is the one gesture that would break silently if the
                // held state were read from egui instead of the raw events.
                self.playing = false;
                base_zoom = pack2(self.timeline_pos as f32, self.view_pan.x);
                raw.events.push(egui::Event::Key {
                    key: egui::Key::Tab,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::NONE,
                });
                button(raw, self.viewer_rect.center(), prim, true, egui::Modifiers::NONE);
                next = 31;
            }
            31 => {
                mv(raw, self.viewer_rect.center() + egui::vec2(120.0, 0.0), egui::Modifiers::NONE);
                next = 32;
            }
            32 => {
                button(
                    raw,
                    self.viewer_rect.center() + egui::vec2(120.0, 0.0),
                    prim,
                    false,
                    egui::Modifiers::NONE,
                );
                raw.events.push(egui::Event::Key {
                    key: egui::Key::Tab,
                    physical_key: None,
                    pressed: false,
                    repeat: false,
                    modifiers: egui::Modifiers::NONE,
                });
                next = 33;
            }
            33 => {
                let (t0, pan_x) = unpack2(base_zoom);
                if self.timeline_pos == t0 as u64 {
                    fail(format!("viewer Tab+LMB drag did not scrub (still at {t0})"));
                }
                if (self.view_pan.x - pan_x).abs() > 0.5 {
                    fail("viewer Tab+LMB drag panned instead of scrubbing".into());
                }
                println!(
                    "BLENDNAV TEST PASS: MMB pan, ctrl+MMB zoom, alt+LMB pan over a \
                     node, ctrl+alt+LMB zoom, wheel zoom, trackpad pan, ctrl+wheel \
                     sideways — graph and viewer; Tab+LMB scrubs the viewer"
                );
                std::process::exit(0);
            }
            _ => {}
        }
        self.test.blendnav = Some((next, frames, base_pan, base_zoom));
    }
}
