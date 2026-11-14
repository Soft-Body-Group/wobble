//! `--test-proxy`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// Proxy-toggle fidelity: flipping `proxy ½` either way must put the comp
    /// on screen at the new size WITH its effects, given no further input.
    /// Reported: "when switching between half res proxy mode and regular,
    /// sometimes I have to switch focus or change a variable for the system
    /// to update ... or the effects won't be applied until I change
    /// something". The tell is a NO-OP EDIT: nudge a Blur's size by 0.001
    /// and, if the picture changes materially, what was on screen before
    /// the nudge was stale. Measured on the output texture itself (the
    /// viewer's source) as the mean horizontal gradient, normalised to
    /// full-res pixels so half-res and full-res numbers compare.
    pub(crate) fn drive_test_proxy(&mut self) {
        let Some(mut pt) = self.test.proxy.take() else { return };
        let fail_now = |m: String| -> ! {
            println!("PROXY TEST FAIL: {m}");
            std::process::exit(1);
        };
        if pt.t0.elapsed().as_secs_f32() > 90.0 {
            for f in &pt.fails {
                println!("PROXY: {f}");
            }
            fail_now(format!("timeout at stage {} after {} toggles", pt.stage, pt.toggles));
        }
        let el = pt.clock.elapsed().as_secs_f32();
        let now = web_time::Instant::now();
        match pt.stage {
            0 if self.playback.is_some() && self.displayed_total >= 3 => {
                // A Blur off Read A, viewed — bypassed first, for the sharp
                // reference the metric is judged against.
                let b = self
                    .egraph
                    .add(NodeKind::Blur(Default::default()), egui::pos2(320.0, 40.0));
                self.egraph.connect(b, 0, 0);
                self.egraph.view = Some(b);
                if let Some(n) = self.egraph.node_mut(b) {
                    if let NodeKind::Blur(p) = &mut n.kind {
                        p.size = 40.0;
                    }
                    n.disabled = true;
                }
                self.playing = false;
                pt.blur = b;
                pt.stage = 1;
                pt.clock = now;
            }
            1 if el > 1.0 => {
                pt.w_full = self.comp.output_size().map(|s| s.0).unwrap_or(0);
                pt.sharp = self.proxy_sharpness(pt.w_full).unwrap_or(0.0);
                if let Some(n) = self.egraph.node_mut(pt.blur) {
                    n.disabled = false;
                }
                pt.stage = 2;
                pt.clock = now;
            }
            // Enabled, at full res, no edit since: is the Blur applied? Then
            // the no-op edit, and the blurred picture that follows is the
            // reference the metric is judged against.
            2 if el > 1.0 => {
                let s = self.proxy_sharpness(pt.w_full).unwrap_or(0.0);
                println!("PROXY: full res: sharp plate {:.4}, Blur just enabled {s:.4}", pt.sharp);
                if s > pt.sharp * 0.6 {
                    pt.fails.push(format!(
                        "enabling the Blur did not apply it ({s:.4} vs sharp plate {:.4}) — \
                         a fresh stage chain rendered as if its params were zero",
                        pt.sharp
                    ));
                }
                if let Some(NodeKind::Blur(p)) =
                    self.egraph.node_mut(pt.blur).map(|n| &mut n.kind)
                {
                    p.size += 0.001;
                }
                pt.stage = 20;
                pt.clock = now;
            }
            20 if el > 0.6 => {
                let s = self.proxy_sharpness(pt.w_full).unwrap_or(0.0);
                if pt.sharp <= 0.0 || s > pt.sharp * 0.6 {
                    fail_now(format!(
                        "the Blur has no measurable effect on this plate even after an edit \
                         (sharp {:.4}, blurred {s:.4}) — the metric cannot see the bug",
                        pt.sharp
                    ));
                }
                println!("PROXY: full res after an edit: blurred {s:.4}");
                pt.stage = 3;
            }
            // Toggle exactly as the toolbar chip does, and nothing else.
            3 => {
                let on = self.proxy < 1.0;
                self.proxy = if on { 1.0 } else { 0.5 };
                self.playback_dirty = true;
                self.last_engine_graph = None;
                pt.shown_at_toggle = self.displayed_total;
                pt.toggles += 1;
                pt.stage = 4;
                pt.clock = now;
            }
            // The new canvas is up and a picture has landed on it; give it a
            // moment more, then measure BEFORE touching anything.
            4 => {
                let want = self.desired_output();
                let have = self.comp.output_size();
                let landed = want.is_some()
                    && have == want
                    && self.displayed_total > pt.shown_at_toggle;
                if landed && el > 0.8 {
                    pt.pre = self.proxy_sharpness(pt.w_full).unwrap_or(0.0);
                    // The no-op edit: a Blur size nobody could see move.
                    if let Some(NodeKind::Blur(p)) =
                        self.egraph.node_mut(pt.blur).map(|n| &mut n.kind)
                    {
                        p.size += 0.001;
                    }
                    pt.stage = 5;
                    pt.clock = now;
                } else if el > 10.0 {
                    fail_now(format!(
                        "toggle {} to {}: canvas {have:?} wanted {want:?}, pictures since the \
                         toggle {}",
                        pt.toggles,
                        if self.proxy < 1.0 { "proxy" } else { "full" },
                        self.displayed_total - pt.shown_at_toggle
                    ));
                }
            }
            5 if el > 0.5 => {
                let post = self.proxy_sharpness(pt.w_full).unwrap_or(0.0);
                let label = if self.proxy < 1.0 { "full -> proxy" } else { "proxy -> full" };
                println!(
                    "PROXY: toggle {} {label}: canvas {:?}, sharpness before the edit {:.4}, \
                     after {:.4} (sharp plate {:.4})",
                    pt.toggles,
                    self.comp.output_size(),
                    pt.pre,
                    post,
                    pt.sharp
                );
                if (pt.pre - post).abs() / post.max(1e-6) > 0.05 {
                    pt.fails.push(format!(
                        "toggle {} {label}: a no-op edit changed the picture ({:.4} -> {:.4}): \
                         the frame after the toggle was STALE",
                        pt.toggles, pt.pre, post
                    ));
                }
                if pt.pre > pt.sharp * 0.6 {
                    pt.fails.push(format!(
                        "toggle {} {label}: the Blur was not applied after the toggle \
                         (sharpness {:.4} vs sharp plate {:.4})",
                        pt.toggles, pt.pre, pt.sharp
                    ));
                }
                pt.stage = if pt.toggles < 6 { 3 } else { 9 };
                pt.clock = now;
            }
            9 => {
                for f in &pt.fails {
                    println!("PROXY: {f}");
                }
                if pt.fails.is_empty() {
                    println!(
                        "PROXY TEST PASS: 6 toggles, the comp re-rendered at the new size with \
                         its effects every time, no edit needed"
                    );
                    std::process::exit(0);
                }
                println!("PROXY TEST FAIL: {} problems (above)", pt.fails.len());
                std::process::exit(1);
            }
            _ => {}
        }
        self.test.proxy = Some(pt);
    }
}
