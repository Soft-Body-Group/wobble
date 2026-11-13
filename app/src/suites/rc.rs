//! `--test-rc`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-rendercache driver: bake the comp to a scratch NVMe dir
    /// (storage = NVMe so playback streams from disk, RAM untouched), play
    /// 48 frames from the cache, then edit a param and expect invalidation.
    pub(crate) fn drive_test_rc(&mut self) {
        let Some((stage, mut frames, mark)) = self.test_rc else { return };
        frames += 1;
        // Wall clock, not iterations: this suite bakes 24 frames and then
        // plays 48 at 60fps — nearly a second of REAL time — while the
        // test-mode loop spins with no vsync to pace it.
        let t0 = *self.test_rc_t0.get_or_insert_with(web_time::Instant::now);
        if t0.elapsed().as_secs_f32() > 90.0 {
            println!(
                "RENDERCACHE TEST FAIL: timeout at stage {stage} (status {:?})",
                self.render_status
            );
            std::process::exit(1);
        }
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 5 => {
                self.render_ram = false;
                self.start_render_bake();
                if self.render_bake.is_none() && self.render_play.is_none() {
                    println!(
                        "RENDERCACHE TEST FAIL: bake didn't start ({:?})",
                        self.render_status
                    );
                    std::process::exit(1);
                }
                self.test_rc = Some((1, frames, 0));
                return;
            }
            1 => {
                if self.render_play.is_some() {
                    self.playing = true;
                    self.fps = 60.0;
                    self.skipped = 0;
                    self.cached_frames_shown = 0;
                    self.test_rc = Some((2, frames, self.displayed_total));
                    return;
                }
                if self.render_bake.is_none() {
                    println!(
                        "RENDERCACHE TEST FAIL: bake aborted ({:?})",
                        self.render_status
                    );
                    std::process::exit(1);
                }
            }
            2 if frames % 240 == 0 => {
                let rp = self.render_play.as_ref();
                eprintln!(
                    "rc-debug: displayed {} (mark {mark}), cached_shown {}, rp {:?}, ahead {:?}, decoded {:?} ({:.1} ms/frame), stalls {:?}, ring free {:?}, playing {}",
                    self.displayed_total,
                    self.cached_frames_shown,
                    rp.is_some(),
                    rp.map(|r| r.player.frames_ahead()),
                    rp.map(|r| r.player.stats.frames_decoded),
                    rp.map(|r| r.player.stats.per_frame_ms.avg()).unwrap_or(0.0),
                    rp.map(|r| r.player.stats.stalls),
                    rp.map(|r| r.ring.free_count()),
                    self.playing,
                );
            }
            2 if self.displayed_total >= mark + 48 => {
                if self.cached_frames_shown < 40 || self.render_play.is_none() {
                    println!(
                        "RENDERCACHE TEST FAIL: only {} of 48 frames came from the cache",
                        self.cached_frames_shown
                    );
                    std::process::exit(1);
                }
                println!(
                    "RENDERCACHE: played {} frames from NVMe at 60 target ({} skipped)",
                    self.cached_frames_shown, self.skipped
                );
                let gid = self
                    .egraph
                    .nodes
                    .iter()
                    .find(|(_, n)| matches!(n.kind, NodeKind::Grade(_)))
                    .map(|(id, _)| *id);
                match gid.and_then(|id| self.egraph.node_mut(id)) {
                    Some(editor::ENode { kind: NodeKind::Grade(g), .. }) => g.gain[0] += 0.1,
                    _ => {
                        println!("RENDERCACHE TEST FAIL: no grade node to edit");
                        std::process::exit(1);
                    }
                }
                self.test_rc = Some((3, frames, 0));
                return;
            }
            3 => {
                if self.render_play.is_none() {
                    println!(
                        "RENDERCACHE: NVMe bake + streamed playback + edit invalidation ok"
                    );
                    // Round 2: RAM mode — the read caches must drain while
                    // the viewer cache holds the RAM, and refill on release.
                    self.render_ram = true;
                    self.start_render_bake();
                    self.test_rc = Some((4, frames, 0));
                    return;
                }
                if frames > 7000 {
                    println!("RENDERCACHE TEST FAIL: cache survived a param edit");
                    std::process::exit(1);
                }
            }
            4 => {
                if let Some(rp) = &self.render_play {
                    let read_budget =
                        self.playback.as_ref().map(|pb| pb.player.cache_budget());
                    if !rp.ram || read_budget != Some(0) {
                        println!(
                            "RENDERCACHE TEST FAIL: RAM mode didn't drain reads (ram={}, budget={read_budget:?})",
                            rp.ram
                        );
                        std::process::exit(1);
                    }
                    self.release_render_cache();
                    let restored =
                        self.playback.as_ref().map(|pb| pb.player.cache_budget());
                    if restored != Some(self.ram_tier_budget()) {
                        println!(
                            "RENDERCACHE TEST FAIL: release didn't restore read budget ({restored:?})"
                        );
                        std::process::exit(1);
                    }
                    println!(
                        "RENDERCACHE TEST PASS: NVMe streamed + invalidated; RAM mode drained reads and release restored them"
                    );
                    std::process::exit(0);
                }
                if self.render_bake.is_none() {
                    println!(
                        "RENDERCACHE TEST FAIL: RAM bake aborted ({:?})",
                        self.render_status
                    );
                    std::process::exit(1);
                }
            }
            _ => {}
        }
        self.test_rc = Some((stage, frames, mark));
    }
}
