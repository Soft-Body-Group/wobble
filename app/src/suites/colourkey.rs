//! `--test-colourkey`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-colourkey driver: the colour keyer end to end. Adds one, views
    /// it, drags a sample across the middle of the plate, and then proves the
    /// three things only a live run can: that the readback→fit→graph path
    /// actually moves the blob off its placeholder, that the fitted key is
    /// SOLID on the pixels it was fitted to, and that the preview shows the
    /// source image while the node is viewed but a downstream consumer still
    /// gets a matte.
    pub(crate) fn drive_test_colourkey(&mut self) {
        let Some((stage, mut frames, id)) = self.test_colourkey else { return };
        frames += 1;
        if frames > 2000 {
            println!("COLOURKEY TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let fail = |m: String| -> ! {
            println!("COLOURKEY TEST FAIL: {m}");
            std::process::exit(1);
        };
        let mut next = stage;
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 3 => {
                let id = self
                    .egraph
                    .add(NodeKind::Keyer(Default::default()), egui::Pos2::new(300.0, 0.0));
                self.egraph.connect(id, 0, 0); // Keyer <- Read A
                self.egraph.view = Some(id);
                self.egraph.view_active_b = false;
                // Pin its properties: the panel's plots build meshes from
                // the live params every frame, and an unopened panel would
                // leave all of that untested.
                self.prop_order = vec![id];
                self.props_visible = true;
                self.bottom = BottomTab::Properties;
                // PIN THE PLATE. Later stages compare two probes of the same
                // pixels (matte, then its inverse); with playback running
                // they land on different source frames and the comparison is
                // nonsense — it read 1.000 here and 0.000 twenty frames on.
                self.playing = false;
                self.seek_to(0);
                self.test_colourkey = Some((1, frames, id));
                return;
            }
            // Let the compile land, then drag a sample across the middle.
            1 if frames > 20 => {
                self.keyer.begin_drag(id, false, 0);
                next = 2;
            }
            2 => {
                let (ow, oh) = self.comp.output_size().unwrap_or((1920, 1080));
                // Walk the cursor across the centre band, exactly as the
                // viewer gesture would.
                // Sweep a short path through the middle — the same 64x64
                // the probe reads, so "solid where I sampled" is testable.
                let t = (frames % 24) as f32 / 24.0;
                self.keyer.cursor = Some((
                    (ow as f32 * 0.5 + (t - 0.5) * 40.0) as i32,
                    (oh as f32 * 0.5 + (t - 0.5) * 40.0) as i32,
                ));
                if frames > 90 {
                    self.keyer.end_drag();
                    next = 3;
                }
            }
            3 if frames > 40 => {
                let Some(NodeKind::Keyer(k)) = self.egraph.node(id).map(|n| &n.kind) else {
                    fail("keyer node vanished".into())
                };
                if k.blobs.len() != 1 {
                    fail(format!("expected one sample, got {}", k.blobs.len()));
                }
                if k.blobs[0] == engine::keyer::KeyBlob::default() {
                    fail("the sample never left its placeholder — readback or fit is dead".into());
                }
                // Preview is on (this node IS the view), so the viewer shows
                // the plate. The matte is in alpha, invisible to this probe.
                let src = self
                    .probe_output_mean()
                    .unwrap_or_else(|| fail("no output to probe".into()));
                if src <= 1e-4 {
                    fail(format!("preview should show the source plate, probed {src:.4}"));
                }
                // Flip to the matte: the key was fitted to these very pixels,
                // so the centre must come back solid.
                if let Some(editor::ENode { kind: NodeKind::Keyer(k), .. }) =
                    self.egraph.node_mut(id)
                {
                    k.view = engine::keyer::KeyerView::Matte;
                }
                next = 4;
            }
            4 if frames > 40 => {
                let m = self
                    .probe_output_mean()
                    .unwrap_or_else(|| fail("no output to probe".into()));
                if m < 0.5 {
                    fail(format!(
                        "matte reads {m:.3} over the pixels it was fitted to (expected mostly solid)"
                    ));
                }
                // Invert is a uniform, so it must land WITHOUT a recompile —
                // and the matte must be the exact complement.
                if let Some(editor::ENode { kind: NodeKind::Keyer(k), .. }) =
                    self.egraph.node_mut(id)
                {
                    k.invert = true;
                }
                self.test_colourkey_m = m;
                next = 5;
            }
            5 if frames > 20 => {
                let m = self.test_colourkey_m;
                let inv = self
                    .probe_output_mean()
                    .unwrap_or_else(|| fail("no output to probe".into()));
                if (inv - (1.0 - m)).abs() > 0.02 {
                    fail(format!("invert gave {inv:.3}, expected {:.3}", 1.0 - m));
                }
                if self.comp.params_stale() {
                    fail("a param edit left the uniforms stale — did it recompile?".into());
                }
                let blob = match self.egraph.node(id).map(|n| &n.kind) {
                    Some(NodeKind::Keyer(k)) => k.blobs[0],
                    _ => fail("keyer node vanished".into()),
                };
                println!(
                    "COLOURKEY TEST PASS: dragged sample fitted an Oklab blob ({:.3},{:.3},{:.3} tol {:.3}/{:.3}/{:.3}), preview showed the plate, matte reads {m:.3} where it was sampled, invert {inv:.3}",
                    blob.center[0],
                    blob.center[1],
                    blob.center[2],
                    blob.tol[0],
                    blob.tol[1],
                    blob.tol[2]
                );
                std::process::exit(0);
            }
            _ => {}
        }
        // Stages count their OWN frames: the first one waits on a real
        // sequence loading, which on a cold cache is longer than any of the
        // later waits and would otherwise expire them all at once.
        self.test_colourkey = Some((next, if next == stage { frames } else { 0 }, id));
    }
}
