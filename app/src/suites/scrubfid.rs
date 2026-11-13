//! `--test-scrubfid`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// Scrub FIDELITY: what the hand asked for versus what went on screen.
    ///
    /// `--test-scrubperf` counts frames put on screen during a drag; it
    /// passes at 59 fps while every other picture is the wrong frame,
    /// because it never asks WHICH frame. This one does: per displayed
    /// frame it records the hand's target (`seek_target`) against the tick
    /// uploaded (`shown_tick`) and counts the things an eye sees as
    /// jitter — a WRONG frame, a REGRESSION (the new picture is farther
    /// from the hand than the one it replaced), a FLIP (the picture moved
    /// against the drag), how many junk frames a stationary hand is shown
    /// before its frame lands, and whether a relative drag keeps its base.
    /// Every pattern runs three times: cold, with the RAM tier warm, and
    /// on the viewer cache (RAM, warm) — the buffered cases are the ones
    /// that must be perfect.
    pub(crate) fn drive_test_scrubfid(&mut self) {
        let Some(mut fid) = self.test.scrubfid.take() else { return };
        fid.redraws += 1;
        if fid.t0.elapsed().as_secs_f32() > 150.0 {
            for f in &fid.fails {
                println!("SCRUBFID: {f}");
            }
            println!(
                "SCRUBFID TEST FAIL: timeout at stage {} tier {} scenario {} (status {:?})",
                fid.stage, FID_TIERS[fid.tier], fid.scen, self.render_status
            );
            std::process::exit(1);
        }
        let (a, b) = self.shot_span();
        let len = (b - a + 1).max(1);
        let el = fid.clock.elapsed().as_secs_f32();
        let now = web_time::Instant::now();

        // A seek onto the tick already on screen is satisfied without an
        // upload (`Scrub::OnScreen`): the display loop clears the pending
        // seek and shows nothing new. Count it as landed.
        if fid.stage == 2
            && !fid.cur_exact
            && !self.force_display
            && self.displayed_total > 0
            && self.shown_tick == self.seek_target.unwrap_or(self.timeline_pos)
        {
            fid.cur_exact = true;
            fid.exact_hits += 1;
            let ms = fid.seek_at.elapsed().as_secs_f32() * 1e3;
            fid.lat_sum += ms;
            fid.lat_max = fid.lat_max.max(ms);
            fid.lat_n += 1;
            let rd = fid.redraws - fid.seek_redraw;
            fid.lag_rd_sum += rd;
            fid.lag_rd_max = fid.lag_rd_max.max(rd);
        }
        // Record the picture that went on screen this redraw, if one did.
        if fid.stage == 2 && self.displayed_total != fid.last_displayed {
            let target = self.seek_target.unwrap_or(self.timeline_pos);
            let shown = self.shown_tick;
            let err = shown.abs_diff(target);
            let (_, pat, _) = FID_SCENARIOS[fid.scen];
            fid.shown += 1;
            if err > 0 {
                fid.wrong += 1;
            }
            fid.sum_err += err;
            fid.max_err = fid.max_err.max(err);
            if fid.last_target.is_some_and(|lt| lt.abs_diff(target) > len / 2) {
                // The pattern wrapped: a jump, and a new monotonic segment.
                fid.prev_shown = None;
            }
            fid.last_target = Some(target);
            if let Some(p) = fid.prev_shown {
                if err > p.abs_diff(target) {
                    fid.regress += 1;
                }
                let dir: i64 = match pat {
                    FidPattern::Fwd60 | FidPattern::SlowFwd | FidPattern::RelDrag => 1,
                    FidPattern::Back60 | FidPattern::SlowBack | FidPattern::RelBack => -1,
                    _ => 0,
                };
                let d = shown as i64 - p as i64;
                // Against the drag AND away from the hand: a picture that
                // has overshot the hand (a cold plate lagging across a
                // wrap of the pattern) comes back toward it the "wrong"
                // way, and that is right. A wrap itself is a jump.
                if dir != 0
                    && d != 0
                    && d.unsigned_abs() < len / 2
                    && d.signum() != dir
                    && err > p.abs_diff(target)
                {
                    fid.flips += 1;
                }
            }
            if err == 0 && !fid.cur_exact {
                fid.cur_exact = true;
                fid.exact_hits += 1;
                let ms = fid.seek_at.elapsed().as_secs_f32() * 1e3;
                fid.lat_sum += ms;
                fid.lat_max = fid.lat_max.max(ms);
                fid.lat_n += 1;
                let rd = fid.redraws - fid.seek_redraw;
                fid.lag_rd_sum += rd;
                fid.lag_rd_max = fid.lag_rd_max.max(rd);
                if std::env::var("WOBBLE_FID_TRACE").is_ok() {
                    println!("fid   exact {target} after {rd} redraws {ms:.1}ms");
                }
            }
            if pat == FidPattern::Hold && fid.settle_ms.is_none() && fid.steps > 0 {
                if err == 0 {
                    fid.settle_ms = Some(fid.seek_at.elapsed().as_secs_f32() * 1e3);
                } else if !fid
                    .prev_shown
                    .is_some_and(|p| (p.min(target)..=p.max(target)).contains(&shown))
                {
                    // A frame on the way from the old picture to the new
                    // position is the scrub landing; one outside that span
                    // is junk.
                    fid.settle_junk += 1;
                }
            }
            fid.prev_shown = Some(shown);
        }
        fid.last_displayed = self.displayed_total;

        match fid.stage {
            0 if self.playback.is_some() && self.displayed_total >= 3 => {
                self.playing = false;
                fid.stage = 1;
                fid.clock = now;
                self.seek_to(Self::fid_start(FID_SCENARIOS[fid.scen].1, a, len));
            }
            // Settle on the scenario's start frame — or go anyway after
            // 1.5s, and let the numbers say the landing never came.
            1 => {
                let start = Self::fid_start(FID_SCENARIOS[fid.scen].1, a, len);
                let landed = !self.force_display && self.shown_tick == start;
                if landed || el > 1.5 {
                    fid.reset_counters();
                    // The picture on screen as the pattern begins is the
                    // predecessor of its first frame.
                    fid.prev_shown = (self.displayed_total > 0).then_some(self.shown_tick);
                    fid.rel_base = self.timeline_pos;
                    fid.dec0 = self.fid_decoded();
                    fid.stage = 2;
                    fid.clock = now;
                    fid.last_displayed = self.displayed_total;
                }
            }
            2 => {
                let (name, pat, secs) = FID_SCENARIOS[fid.scen];
                if el < secs {
                    let n60 = (el * 60.0) as u64;
                    let slow = (el / 0.15) as u64;
                    let want = match pat {
                        FidPattern::Fwd60 => Some(a + n60 % len),
                        FidPattern::Back60 => Some(a + (len - 1 - n60 % len)),
                        FidPattern::SlowFwd => Some(a + slow % len),
                        FidPattern::SlowBack => Some(a + (len - 1 - slow % len)),
                        FidPattern::Reversal => Some(
                            a + if ((el / 0.12) as u64) % 2 == 0 {
                                4u64.min(len - 1)
                            } else {
                                10u64.min(len - 1)
                            },
                        ),
                        FidPattern::Hold => Some(a + 12u64.min(len - 1)),
                        FidPattern::RelDrag | FidPattern::RelBack => {
                            if n60 > fid.steps {
                                // Exactly the Tab-drag's step (see the
                                // viewer's `scrubbing` block).
                                let step: i64 = if pat == FidPattern::RelDrag { 1 } else { -1 };
                                let cur = self.timeline_pos as i64;
                                let tick =
                                    a as i64 + (cur - a as i64 + step).rem_euclid(len as i64);
                                self.seek_to(tick as u64);
                                fid.steps += 1;
                                fid.cur_exact = false;
                                fid.seek_at = now;
                                fid.seek_redraw = fid.redraws;
                                let expected = a as i64
                                    + ((fid.rel_base - a) as i64 + step * fid.steps as i64)
                                        .rem_euclid(len as i64);
                                if self.seek_target != Some(expected as u64) {
                                    fid.drift += 1;
                                }
                            }
                            None
                        }
                    };
                    if let Some(w) = want {
                        if fid.last_want != Some(w) {
                            if std::env::var("WOBBLE_FID_TRACE").is_ok() {
                                let (pr, st) = self
                                    .playback
                                    .as_ref()
                                    .map(|pb| (pb.player.debug_pressure(), pb.player.debug_state()))
                                    .unwrap_or_default();
                                println!(
                                    "fid seek {w} shown {} back {} pressure(pend,infl,free,complete) {pr:?} state(disp,issue,pend,gen) {st:?}",
                                    self.shown_tick, self.scrub_back
                                );
                            }
                            self.seek_to(w);
                            fid.last_want = Some(w);
                            fid.steps += 1;
                            fid.cur_exact = false;
                            fid.seek_at = now;
                            fid.seek_redraw = fid.redraws;
                        }
                    }
                } else {
                    let tier = FID_TIERS[fid.tier];
                    let decodes = self.fid_decoded().saturating_sub(fid.dec0);
                    let row = fid.row(tier, name, pat, secs, decodes);
                    println!("{row}");
                    fid.rows.push(row);
                    fid.judge(tier, name, pat, secs);
                    fid.scen += 1;
                    if fid.scen < FID_SCENARIOS.len() {
                        fid.stage = 1;
                        fid.clock = now;
                        self.seek_to(Self::fid_start(FID_SCENARIOS[fid.scen].1, a, len));
                    } else if fid.tier == 0 {
                        // The RAM tier on for every Read (off by default,
                        // exactly as a user opts in on the panel), then one
                        // pass through the clip to fill it.
                        let reads: Vec<u64> = self
                            .egraph
                            .nodes
                            .iter()
                            .filter(|(_, n)| matches!(n.kind, NodeKind::Read { .. }))
                            .map(|(id, _)| *id)
                            .collect();
                        for id in reads {
                            if let Some(editor::ENode {
                                kind: NodeKind::Read { ram_cache, .. }, ..
                            }) = self.egraph.node_mut(id)
                            {
                                *ram_cache = true;
                            }
                        }
                        self.playback_dirty = true;
                        self.seek_to(a);
                        self.playing = true;
                        self.fps = 60.0;
                        fid.stage = 3;
                        fid.clock = now;
                    } else if fid.tier == 1 {
                        self.render_ram = true;
                        self.start_render_bake();
                        if self.render_bake.is_none() && self.render_play.is_none() {
                            println!(
                                "SCRUBFID TEST FAIL: viewer bake didn't start ({:?})",
                                self.render_status
                            );
                            std::process::exit(1);
                        }
                        fid.stage = 4;
                        fid.clock = now;
                    } else {
                        fid.stage = 9;
                    }
                }
            }
            // Warm the ACTIVE player's RAM tier: every frame of every reader
            // resident, so the next tier really is buffered.
            3 => {
                let (cached, need) = if let Some(rp) = &self.render_play {
                    // The viewer cache's RAM tier is the ring itself.
                    (rp.player.ring_residency(0).iter().filter(|b| **b).count(), len as usize)
                } else if let Some(pb) = &self.playback {
                    (pb.player.cache_stats().1, pb.player.sequences().len() * len as usize)
                } else {
                    (0, 1)
                };
                if cached >= need || el > 8.0 {
                    if cached < need {
                        fid.fails.push(format!(
                            "[{}] RAM tier warmed only {cached}/{need} frames in {el:.1}s",
                            FID_TIERS[fid.tier + 1]
                        ));
                    }
                    self.playing = false;
                    fid.tier += 1;
                    fid.scen = 0;
                    fid.stage = 1;
                    fid.clock = now;
                    self.seek_to(Self::fid_start(FID_SCENARIOS[0].1, a, len));
                }
            }
            // The viewer bake: when it lands, warm its RAM tier like the
            // read tier above.
            4 => {
                if self.render_play.is_some() {
                    self.seek_to(a);
                    self.playing = true;
                    self.fps = 60.0;
                    fid.stage = 3;
                    fid.clock = now;
                } else if self.render_bake.is_none() {
                    println!(
                        "SCRUBFID TEST FAIL: viewer bake aborted ({:?})",
                        self.render_status
                    );
                    std::process::exit(1);
                }
            }
            9 => {
                for f in &fid.fails {
                    println!("SCRUBFID: {f}");
                }
                if fid.fails.is_empty() {
                    println!(
                        "SCRUBFID TEST PASS: {} patterns x {} tiers — no regressions, no flips, \
                         a still hand sees no junk, drags keep their base",
                        FID_SCENARIOS.len(),
                        FID_TIERS.len()
                    );
                    std::process::exit(0);
                }
                println!("SCRUBFID TEST FAIL: {} problems (above)", fid.fails.len());
                std::process::exit(1);
            }
            _ => {}
        }
        self.test.scrubfid = Some(fid);
    }
}
