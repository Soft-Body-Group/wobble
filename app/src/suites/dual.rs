//! `--test-dual`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    pub(crate) fn drive_test_dual(&mut self) {
        let Some((stage, mut frames, grade)) = self.test_dual else { return };
        // No synthetic input here, so nothing else keeps frames coming.
        self.egui_ctx.request_repaint();
        frames += 1;
        if frames > 1200 {
            let o = self.other_pane.as_ref().map(|o| {
                (o.shown, o.last_view, o.comp.output_size(), o.last_ready)
            });
            println!(
                "DUAL TEST FAIL: timeout at stage {stage}; other pane (shown, view, size, ready) = {o:?}, dual {}, live pane {}",
                self.dual_view,
                self.live_pane()
            );
            std::process::exit(1);
        }
        let fail = |m: String| -> ! {
            println!("DUAL TEST FAIL: {m}");
            std::process::exit(1);
        };
        let mut grade = grade;
        let mut next = stage;
        // Both outputs at the same relative spot.
        let sample = |s: &Self| -> Option<([f32; 4], [f32; 4])> {
            let a = s.comp.output_texture()?;
            let o = s.other_pane.as_ref().filter(|o| o.shown)?;
            let b = o.comp.output_texture()?;
            let at = |t: &wgpu::Texture| {
                ((t.width() as f32 * 0.41) as u32, (t.height() as f32 * 0.47) as u32)
            };
            let (ax, ay) = at(a);
            let (bx, by) = at(b);
            Some((s.test_read_texel(a, ax, ay), s.test_read_texel(b, bx, by)))
        };
        let ratio_ok = |a: [f32; 4], b: [f32; 4], k: f32| {
            (0..3).all(|i| a[i] < 0.02 || ((b[i] / a[i]) - k).abs() < 0.03)
        };
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 2 => {
                let read = self
                    .egraph
                    .nodes
                    .iter()
                    .find(|(_, n)| matches!(n.kind, NodeKind::Read { .. }))
                    .map(|(id, _)| *id)
                    .unwrap_or_else(|| fail("needs a Read".into()));
                let g = self.egraph.add(
                    NodeKind::Grade(engine::graph::GradeParams { gain: [0.5; 3], ..Default::default() }),
                    egui::pos2(300.0, 120.0),
                );
                self.egraph.connect(g, 0, read);
                self.egraph.view = Some(read);
                self.egraph.view_active_b = false;
                self.egraph.view_c = Some(g);
                self.egraph.view2_active_d = false;
                self.egraph.view_pane = 0;
                self.dual_view = true;
                self.playing = false;
                self.seek_to(5);
                grade = g;
                next = 1;
            }
            1 if frames > 60 => {
                let Some((a, b)) = sample(self) else {
                    self.test_dual = Some((stage, frames, grade));
                    return;
                };
                if a[..3].iter().all(|v| v.abs() < 0.02) {
                    fail(format!("the plate reads black: {a:?}"));
                }
                if !ratio_ok(a, b, 0.5) {
                    fail(format!("right pane {b:?} is not half the left {a:?}"));
                }
                if self.live_pane() != 0 {
                    fail("the left pane is not the live one".into());
                }
                if let Some(editor::ENode { kind: NodeKind::Grade(p), .. }) =
                    self.egraph.node_mut(grade)
                {
                    p.gain = [0.25; 3];
                }
                self.playback_dirty = true;
                next = 2;
            }
            2 if frames > 60 => {
                let Some((a, b)) = sample(self) else {
                    self.test_dual = Some((stage, frames, grade));
                    return;
                };
                if !ratio_ok(a, b, 0.25) {
                    fail(format!(
                        "after the edit the right pane reads {b:?} against {a:?}: not live"
                    ));
                }
                println!(
                    "DUAL TEST PASS: both panes render — left the plate {:?}, right its Grade, and the right follows an edit while the left is live",
                    [a[0], a[1], a[2]]
                );
                std::process::exit(0);
            }
            _ => {}
        }
        self.test_dual = Some((next, if next == stage { frames } else { 0 }, grade));
    }
}
