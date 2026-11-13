//! `--test-setup`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-setup driver: save -> wipe -> load -> verify graph + sequences
    /// come back.
    pub(crate) fn drive_test_setup(&mut self) {
        let Some((stage, mut frames, nodes, loads)) = self.test_setup else { return };
        frames += 1;
        if frames > 1800 {
            println!("SETUP TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let path = std::env::temp_dir().join("wobble_test_setup.wbl");
        match stage {
            0 if frames >= 3 => {
                let nodes = self.egraph.nodes.len();
                let loads = self.loaded.len();
                if let Err(e) = self.save_setup(&path) {
                    println!("SETUP TEST FAIL: save: {e:#}");
                    std::process::exit(1);
                }
                // Wipe everything the setup should restore.
                self.egraph = EditorGraph::default();
                self.loaded.clear();
                self.playback_dirty = true;
                self.fps = 99.0;
                self.test_setup = Some((1, frames, nodes, loads));
                return;
            }
            1 => {
                if let Err(e) = self.load_setup(&path) {
                    println!("SETUP TEST FAIL: load: {e:#}");
                    std::process::exit(1);
                }
                let n = self.egraph.nodes.len();
                if n != nodes || (self.fps - 24.0).abs() > 0.01 {
                    println!("SETUP TEST FAIL: roundtrip nodes {n}/{nodes} fps {}", self.fps);
                    std::process::exit(1);
                }
                self.test_setup = Some((2, frames, nodes, loads));
                return;
            }
            2 => {
                if self.loaded.len() == loads && (loads == 0 || self.playback.is_some()) {
                    // Now the other half of the contract: a history jump
                    // (undo/redo) restores nodes whose loads went with them,
                    // so it has to re-adopt the footage as well.
                    let g = self.egraph.clone();
                    self.loaded.clear();
                    self.load_errors.clear();
                    self.apply_history(g);
                    self.test_setup = Some((3, frames, nodes, loads));
                    return;
                }
            }
            3 => {
                let back = self.loaded.len();
                if back != loads {
                    println!(
                        "SETUP TEST FAIL: history jump left {back} of {loads} sequences loaded"
                    );
                    std::process::exit(1);
                }
                println!(
                    "SETUP TEST PASS: {nodes} nodes, {loads} sequences reloaded, fps restored, history jump re-adopted them"
                );
                std::process::exit(0);
            }
            _ => {}
        }
        self.test_setup = Some((stage, frames, nodes, loads));
    }
}
