//! `--test-bin`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    pub(crate) fn drive_test_bin(&mut self) {
        let Some((stage, mut frames, before)) = self.test_bin else { return };
        frames += 1;
        if frames > 2400 {
            println!("BIN TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 2 => {
                let n = self.egraph.nodes.len();
                self.browser = Some(browser::SeqBrowser::new(
                    None,
                    std::path::PathBuf::from("data"),
                    self.decoder.clone(),
                ));
                if let Some(b) = &mut self.browser {
                    b.test_set_grid(true);
                    b.test_set_tile(140.0);
                    b.test_search(2);
                }
                self.test_bin = Some((1, frames, n));
            }
            // Wait for the walk to finish, then bin three of its hits.
            1 => {
                let done = self
                    .browser
                    .as_ref()
                    .map(|b| b.test_search_status())
                    .is_some_and(|(hits, _, running)| !running && hits >= 3);
                // Thumbnails decode on workers, so give the tiles a moment
                // to actually come back with one — that is the proof the
                // recursive TILE path rendered, not just the list.
                let (_, _, thumbs) = self
                    .browser
                    .as_ref()
                    .map(|b| b.test_counts())
                    .unwrap_or((0, 0, 0));
                if done && thumbs > 0 {
                    let (hits, dirs, _) =
                        self.browser.as_ref().unwrap().test_search_status();
                    log::info!("bin: {thumbs} thumbnails over {hits} hits");
                    let act = self.browser.as_mut().unwrap().test_bin_hits(3);
                    let browser::BrowserAction::PickMany(picks) = act else {
                        println!("BIN TEST FAIL: bin did not commit a batch");
                        std::process::exit(1);
                    };
                    if picks.len() != 3 {
                        println!("BIN TEST FAIL: binned {} of 3", picks.len());
                        std::process::exit(1);
                    }
                    log::info!("bin: {hits} hits over {dirs} dirs");
                    self.commit_picks(picks);
                    self.test_bin = Some((2, frames, before));
                }
            }
            // The new Reads must exist AND resolve to real frames.
            2 if frames > 30 => {
                let made = self.egraph.nodes.len().saturating_sub(before);
                let reads = self
                    .egraph
                    .nodes
                    .iter()
                    .filter(|(_, n)| matches!(n.kind, NodeKind::Read { .. }))
                    .count();
                let loaded_new = self
                    .egraph
                    .nodes
                    .iter()
                    .filter(|(id, n)| {
                        matches!(n.kind, NodeKind::Read { .. })
                            && self.loaded.contains_key(id)
                    })
                    .count();
                if made != 3 {
                    println!("BIN TEST FAIL: {made} nodes created, want 3");
                    std::process::exit(1);
                }
                if loaded_new < 3 {
                    println!(
                        "BIN TEST FAIL: only {loaded_new} of {reads} reads loaded frames"
                    );
                    std::process::exit(1);
                }
                if self.browser.is_some() {
                    println!("BIN TEST FAIL: browser stayed open after committing the bin");
                    std::process::exit(1);
                }
                println!(
                    "BIN TEST PASS: recursive search -> bin -> {made} Read nodes created and loaded, browser closed ({reads} reads total)"
                );
                std::process::exit(0);
            }
            _ => self.test_bin = Some((stage, frames, before)),
        }
    }
}
