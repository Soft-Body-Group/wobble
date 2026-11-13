//! `--test-viewswitch`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// Repro for "switching the view target shows black until you touch the
    /// UI": pause on a real frame, retarget the view (merge -> transform,
    /// resolutions differ), and after a few frames the output must NOT be
    /// black — with no UI interaction at all.
    pub(crate) fn drive_test_viewswitch(&mut self) {
        let Some((stage, mut frames)) = self.test_viewswitch else { return };
        frames += 1;
        if frames > 2400 {
            println!("VIEWSWITCH TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 3 => {
                self.playing = false;
                // Force differing resolutions: merge fixed small, sources big.
                if let Some(n) = self.egraph.node_mut(4) {
                    n.blend_res = Some((640, 360));
                }
                self.test_viewswitch = Some((1, frames));
            }
            1 if frames > 20 => {
                let base = self.output_mean_luma().unwrap_or(0.0);
                if base < 0.01 {
                    println!("VIEWSWITCH TEST FAIL: baseline view is already black");
                    std::process::exit(1);
                }
                // Gizmo space: viewed through the 640x360 merge, the
                // transform still lives in its SOURCE's resolution — the
                // truncated-outline bug was using the view size here.
                let local = self.node_res_f32(2);
                let view = self.comp.output_size().map(|(w, h)| (w as f32, h as f32));
                if local == view || local.is_none() {
                    println!(
                        "VIEWSWITCH TEST FAIL: transform local res {local:?} should be its source's, not the merge view {view:?}"
                    );
                    std::process::exit(1);
                }
                // The switch, exactly as the 1 key does it — no other input.
                self.egraph.view = Some(2);
                self.egraph.view_active_b = false;
                self.test_viewswitch = Some((2, frames));
            }
            2 if frames > 40 => {
                let l = self.output_mean_luma().unwrap_or(0.0);
                let sz = self.comp.output_size();
                if l < 0.01 {
                    println!(
                        "VIEWSWITCH TEST FAIL: black after view switch (luma {l:.4}, output {sz:?})"
                    );
                    std::process::exit(1);
                }
                // And back to the merge.
                self.egraph.view = Some(4);
                self.test_viewswitch = Some((3, frames));
            }
            3 if frames > 60 => {
                let l = self.output_mean_luma().unwrap_or(0.0);
                if l < 0.01 {
                    println!("VIEWSWITCH TEST FAIL: black after switching BACK (luma {l:.4})");
                    std::process::exit(1);
                }
                println!(
                    "VIEWSWITCH TEST PASS: view retarget across resolutions renders without UI interaction (luma {l:.3})"
                );
                std::process::exit(0);
            }
            _ => self.test_viewswitch = Some((stage, frames)),
        }
    }
}
