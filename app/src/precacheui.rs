//! Precache Reads: decode every frame of some Reads' ranges into the cache
//! tiers in the background, so playback starts warm.
//!
//! The work is `engine::player::Player::precache`; this is the app's side —
//! which Reads, the budget check, the progress window and cancel. A run is
//! keyed by NODE ids, not reader indices: a rig rebuild (a load, a cache
//! toggle, a range trim) makes a new player with a new, empty RAM tier, so
//! a run still going when that happens stops and starts again on the new
//! rig. Frames it already wrote to the NVMe tier come back from there.
//!
//! A Load's own "cache in–out" / "cache all" leave its toggles alone (RAM
//! off = NVMe only).
//! From the menu, RAM is the tier a precache is for, so a Read whose RAM toggle is off has
//! it turned ON (an authored, visible, undoable edit — the checkbox never
//! lies about what the Read uses). With no RAM budget at all (the viewer
//! cache is holding it) the run fills the NVMe tier only, and says so.

use crate::Gfx;
use engine::graph::NodeKind;
use engine::player::Precache;
use std::sync::atomic::Ordering;

pub struct Run {
    /// The Reads, one progress row each.
    nodes: Vec<u64>,
    names: Vec<String>,
    job: Option<Precache>,
    /// `Gfx::rig_builds` the job was started on.
    rig: u64,
    /// Said once at start: a budget that will not hold it all, no tier, …
    warn: Option<String>,
    started: web_time::Instant,
    cancelled: bool,
    /// Every file of each Load, not just its in..out (NVMe for the rest).
    all_files: bool,
    /// Seconds it took, frozen when it finished.
    took: Option<f32>,
}

impl Run {
    pub(crate) fn job(&self) -> Option<&Precache> {
        self.job.as_ref()
    }
    pub(crate) fn rig(&self) -> u64 {
        self.rig
    }
}

impl Gfx {
    /// Precache these Reads (anything else in the list is ignored). A run
    /// already going is stopped and replaced.
    pub(crate) fn precache_start(&mut self, ids: Vec<u64>) {
        self.precache_start_with(ids, true);
    }

    /// `turn_on_ram`: flip each Read's RAM toggle on first (the menu's
    /// precache). Off, a Read's tiers are left as set: "cache all" with RAM
    /// off fills the NVMe tier only.
    fn precache_start_with(&mut self, ids: Vec<u64>, turn_on_ram: bool) {
        if let Some(r) = &self.precache {
            if let Some(j) = &r.job {
                j.stop();
            }
        }
        let nodes: Vec<u64> = ids
            .into_iter()
            .filter(|id| {
                matches!(self.egraph.node(*id).map(|n| &n.kind), Some(NodeKind::Read { .. }))
            })
            .collect();
        if nodes.is_empty() {
            self.precache = None;
            return;
        }
        let ram_ok = self.ram_tier_budget() > 0
            && !self.render_play.as_ref().is_some_and(|rp| rp.ram);
        let mut flipped = false;
        if ram_ok && turn_on_ram {
            for id in &nodes {
                if let Some(NodeKind::Read { ram_cache, .. }) =
                    self.egraph.node_mut(*id).map(|n| &mut n.kind)
                {
                    if !*ram_cache {
                        *ram_cache = true;
                        flipped = true;
                    }
                }
            }
        }
        if flipped {
            self.playback_dirty = true;
        }
        let names = nodes
            .iter()
            .map(|id| match self.egraph.node(*id) {
                Some(n) if !n.ui_label.is_empty() => n.ui_label.clone(),
                Some(n) => match &n.kind {
                    NodeKind::Read { path, .. } if !path.is_empty() => {
                        crate::editor::guess_read_label(path)
                    }
                    k => crate::editor::node_label(k),
                },
                None => format!("node {id}"),
            })
            .collect();
        self.precache = Some(Run {
            nodes,
            names,
            job: None,
            rig: 0,
            warn: None,
            started: web_time::Instant::now(),
            cancelled: false,
            all_files: false,
            took: None,
        });
    }

    /// The Load's "cache all": its in..out as a precache does, plus every
    /// other file of the clip into the NVMe tier.
    /// The Load's "cache in–out": its in..out, tiers left as set.
    pub(crate) fn precache_start_inout(&mut self, ids: Vec<u64>) {
        self.precache_start_with(ids, false);
    }

    pub(crate) fn precache_start_files(&mut self, ids: Vec<u64>) {
        self.precache_start_with(ids, false);
        if let Some(r) = &mut self.precache {
            r.all_files = true;
        }
    }

    /// Per frame: take the graph menu's ask, (re)start the job on the
    /// current rig, draw the progress window.
    pub(crate) fn precache_frame(&mut self, ctx: &egui::Context) {
        if let Some(ids) = self.node_editor.precache_req.take() {
            self.precache_start(ids);
        }
        let Some(run) = &self.precache else { return };
        let running = run.job.as_ref().is_some_and(|j| !j.finished());
        let stale = run.job.is_none() || (running && run.rig != self.rig_builds);
        if stale && !run.cancelled && !self.playback_dirty {
            self.precache_launch();
        }
        self.precache_window(ctx);
    }

    fn precache_launch(&mut self) {
        let Some(mut run) = self.precache.take() else { return };
        if let Some(j) = &run.job {
            j.stop();
        }
        run.job = None;
        run.rig = self.rig_builds;
        let Some(pb) = &self.playback else {
            run.warn = Some("nothing loaded to precache".into());
            run.cancelled = true;
            self.precache = Some(run);
            return;
        };
        // Every reader fed by the Read: its own instances (one per time
        // path), and the channel selects and Cryptomatte ranks off it.
        let instances = self.reader_instances();
        let groups: Vec<Vec<usize>> = run
            .nodes
            .iter()
            .map(|id| {
                instances
                    .iter()
                    .filter(|(inst, _)| {
                        inst.node == *id || self.egraph.channel_source(inst.node) == Some(*id)
                    })
                    .filter_map(|(_, slot)| pb.slots.iter().position(|s| s == slot))
                    .collect()
            })
            .collect();
        let all: Vec<usize> = groups.iter().flatten().copied().collect();
        let need = pb.player.precache_ram_bytes(&all);
        let budget = pb.player.cache_budget() as u64;
        let gb = |b: u64| b as f64 / 1e9;
        run.warn = if all.is_empty() {
            Some("not part of the comp's rig yet (no frames loaded)".into())
        } else if budget == 0 {
            Some("no RAM budget free (the viewer cache holds it): NVMe tier only".into())
        } else if need > budget {
            Some(format!(
                "needs {:.1} GB, the RAM tier holds {:.1} GB: it fills to the budget, \
                 the rest go to NVMe only — raise the budget in cache…",
                gb(need),
                gb(budget)
            ))
        } else {
            None
        };
        // Half the cores: the other half keep the playback decoder and the
        // UI responsive while this runs.
        let threads = std::thread::available_parallelism()
            .map(|n| (n.get() / 2).max(1))
            .unwrap_or(2);
        let full: Vec<Option<(Vec<std::path::PathBuf>, u64)>> = if run.all_files {
            run.nodes
                .iter()
                .map(|id| self.loaded.get(id).map(|l| (l.seq.frames.clone(), l.seq.offset)))
                .collect()
        } else {
            Vec::new()
        };
        let job = pb.player.precache_files(groups, threads, full);
        #[cfg(target_arch = "wasm32")]
        {
            run.warn = Some("precache needs threads, which a browser page does not have".into());
        }
        if job.total() == 0 && run.warn.is_none() {
            run.warn = Some("these Loads use no cache tier (RAM and NVMe both off)".into());
        }
        run.job = Some(job);
        run.started = web_time::Instant::now();
        run.took = None;
        self.precache = Some(run);
    }

    fn precache_window(&mut self, ctx: &egui::Context) {
        let Some(run) = &mut self.precache else { return };
        let mut close = false;
        let finished = run.job.as_ref().is_none_or(|j| j.finished());
        if finished && run.took.is_none() {
            run.took = Some(run.started.elapsed().as_secs_f32());
        }
        egui::Window::new("precache")
            .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-12.0, -12.0))
            .resizable(false)
            .collapsible(false)
            .default_width(300.0)
            .show(ctx, |ui| {
                if let Some(j) = &run.job {
                    for (i, name) in run.names.iter().enumerate() {
                        let Some((total, done)) = j.groups.get(i) else { continue };
                        let done = done.load(Ordering::Relaxed).min(*total);
                        let frac = if *total == 0 { 1.0 } else { done as f32 / *total as f32 };
                        ui.label(name);
                        ui.add(
                            egui::ProgressBar::new(frac)
                                .desired_width(280.0)
                                .text(format!("{done} / {total}")),
                        );
                    }
                    if j.ram_full.load(Ordering::Relaxed) {
                        ui.colored_label(crate::theme::warn(), "RAM tier full — later frames NVMe only");
                    }
                    let failed = j.failed.load(Ordering::Relaxed);
                    if failed > 0 {
                        ui.colored_label(
                            crate::theme::error(),
                            format!("{failed} frames would not decode"),
                        );
                    }
                }
                if let Some(w) = &run.warn {
                    ui.colored_label(crate::theme::warn(), w);
                }
                ui.horizontal(|ui| {
                    if finished || run.cancelled {
                        let what = if run.cancelled { "cancelled" } else { "done" };
                        let secs = run.took.unwrap_or(0.0);
                        if run.cancelled || run.job.is_none() {
                            ui.weak(what);
                        } else {
                            ui.weak(format!("{what} in {secs:.1}s"));
                        }
                        if ui.button("close").clicked() {
                            close = true;
                        }
                    } else if ui.button("cancel").clicked() {
                        if let Some(j) = &run.job {
                            j.stop();
                        }
                        run.cancelled = true;
                    }
                });
            });
        if close {
            self.precache = None;
        } else if !finished {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
    }
}
