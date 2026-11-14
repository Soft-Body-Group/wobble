//! The optimise command: the dialog that starts a transcode (destination
//! and ProRes flavour, each with a default), the background job and its
//! line in the HUD, and the optimised-media window (cache size, cap,
//! purge). The work itself is `media::optimise`.

use media::optimise::{self, CacheDir, Dest, Job, Profile, Progress, Settings};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;

struct Running {
    source: PathBuf,
    dest: Dest,
    progress: Arc<Progress>,
    handle: JoinHandle<anyhow::Result<PathBuf>>,
    started: Instant,
}

pub struct OptimiseUi {
    pub settings: Settings,
    job: Option<Running>,
    /// The start dialog: open, and its picks.
    asking: bool,
    dest: Dest,
    profile: Profile,
    make_default: bool,
    pub show_cache: bool,
    cap_text: String,
    /// The last outcome, for the HUD.
    notice: Option<(String, Instant)>,
}

impl OptimiseUi {
    pub fn new() -> Self {
        let settings = Settings::load();
        Self {
            dest: settings.dest,
            profile: settings.profile,
            cap_text: settings.cache_cap_gb.map(|g| g.to_string()).unwrap_or_default(),
            settings,
            job: None,
            asking: false,
            make_default: false,
            show_cache: false,
            notice: None,
        }
    }

    pub fn busy(&self) -> bool {
        self.job.is_some()
    }

    /// Open the start dialog with the defaults.
    pub fn ask(&mut self) {
        self.asking = true;
        self.dest = self.settings.dest;
        self.profile = self.settings.profile;
    }

    pub fn start(&mut self, source: &Path, dest: Dest, profile: Profile) {
        if self.job.is_some() {
            self.notice = Some(("an optimise is already running".into(), Instant::now()));
            return;
        }
        let progress = Arc::new(Progress::default());
        let job = Job {
            source: source.to_path_buf(),
            dest,
            profile,
            cache_dir: self.settings.cache_dir.clone(),
            cache_cap_gb: self.settings.cache_cap_gb,
        };
        let p = progress.clone();
        let handle = std::thread::Builder::new()
            .name("optimise".into())
            .spawn(move || optimise::run(&job, &p))
            .expect("spawn optimise");
        self.job = Some(Running { source: source.to_path_buf(), dest, progress, handle, started: Instant::now() });
    }

    pub fn cancel(&mut self) {
        if let Some(j) = &self.job {
            j.progress.cancel.store(true, Ordering::Relaxed);
        }
    }

    /// A finished job: the SOURCE it optimised, when it succeeded.
    pub fn poll(&mut self) -> Option<PathBuf> {
        if !self.job.as_ref().is_some_and(|j| j.handle.is_finished()) {
            return None;
        }
        let j = self.job.take().expect("checked");
        let secs = j.started.elapsed().as_secs_f32();
        match j.handle.join() {
            Ok(Ok(out)) => {
                let n = j.progress.done.load(Ordering::Relaxed);
                self.notice = Some((
                    format!("optimised {} frames in {secs:.1}s ({:.0} fps): {}", n, n as f32 / secs, out.display()),
                    Instant::now(),
                ));
                Some(j.source)
            }
            Ok(Err(e)) => {
                self.notice = Some((format!("optimise stopped: {e:#}"), Instant::now()));
                None
            }
            Err(_) => {
                self.notice = Some(("optimise crashed".into(), Instant::now()));
                None
            }
        }
    }

    pub fn hud_line(&self) -> Option<String> {
        if let Some(j) = &self.job {
            let done = j.progress.done.load(Ordering::Relaxed);
            let fps = done as f32 / j.started.elapsed().as_secs_f32().max(0.01);
            let to = match j.dest {
                Dest::Beside => "beside the original",
                Dest::Cache => "to the cache",
            };
            return Some(format!("optimising {:.0}%  {fps:.0} fps  {to}", j.progress.fraction() * 100.0));
        }
        self.notice.as_ref().filter(|(_, at)| at.elapsed().as_secs() < 8).map(|(s, _)| s.clone())
    }

    /// Draw the dialog and the cache window. `current`: the open file, is
    /// it already read through an optimised copy, is it intra. True when
    /// the current clip's optimised copy was purged (reopen it).
    pub fn windows(&mut self, ctx: &egui::Context, current: Option<(&Path, bool, bool)>) -> bool {
        self.dialog(ctx, current);
        self.cache_window(ctx, current)
    }

    fn dialog(&mut self, ctx: &egui::Context, current: Option<(&Path, bool, bool)>) {
        if !self.asking {
            return;
        }
        let mut open = true;
        let mut go = false;
        egui::Window::new("optimise").open(&mut open).collapsible(false).resizable(false).show(ctx, |ui| {
            let Some((path, optimised, intra)) = current else {
                ui.label("open a clip first");
                return;
            };
            ui.label(path.display().to_string());
            if optimised {
                ui.label("already playing from its optimised copy");
                return;
            }
            if intra {
                ui.label("every frame of this clip already decodes on its own: nothing to gain");
                return;
            }
            ui.add_space(4.0);
            ui.label("transcode to ProRes, so any frame decodes on its own. The original is never touched.");
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label("put it");
                ui.radio_value(&mut self.dest, Dest::Beside, "beside the original");
                ui.radio_value(&mut self.dest, Dest::Cache, "in the optimised cache");
            });
            ui.horizontal(|ui| {
                ui.label("ProRes");
                ui.radio_value(&mut self.profile, Profile::Lt, "422 LT");
                ui.radio_value(&mut self.profile, Profile::Standard, "422");
                ui.radio_value(&mut self.profile, Profile::Hq, "422 HQ");
            });
            ui.checkbox(&mut self.make_default, "make these the default");
            ui.add_space(4.0);
            go = ui.button("optimise  (enter)").clicked() || ui.input(|i| i.key_pressed(egui::Key::Enter));
        });
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            open = false;
        }
        if go {
            if let Some((path, ..)) = current {
                if self.make_default {
                    self.settings.dest = self.dest;
                    self.settings.profile = self.profile;
                    let _ = self.settings.save();
                }
                let path = path.to_path_buf();
                self.start(&path, self.dest, self.profile);
            }
            open = false;
        }
        self.asking = open;
    }

    fn cache_window(&mut self, ctx: &egui::Context, current: Option<(&Path, bool, bool)>) -> bool {
        if !self.show_cache {
            return false;
        }
        let mut open = true;
        let mut purged_current = false;
        let cache = CacheDir::new(&self.settings.cache_dir);
        egui::Window::new("optimised media").open(&mut open).default_width(460.0).show(ctx, |ui| {
            if let Some(j) = &self.job {
                ui.horizontal(|ui| {
                    ui.label(format!("optimising {} — {:.0}%", j.source.display(), j.progress.fraction() * 100.0));
                });
                if ui.button("cancel the optimise").clicked() {
                    self.cancel();
                }
                ui.separator();
            }
            ui.label("Scrub's cache folder (files beside originals are yours to delete):");
            let mut dir = self.settings.cache_dir.display().to_string();
            if ui.text_edit_singleline(&mut dir).lost_focus() {
                self.settings.cache_dir = imfram::filebrowser::expand_home(&dir);
                let _ = self.settings.save();
            }
            let entries = cache.entries();
            let gb = cache.size() as f64 / (1u64 << 30) as f64;
            ui.label(format!("{} files, {gb:.2} GB", entries.len()));
            ui.horizontal(|ui| {
                ui.label("cap (GB, empty for none)");
                if ui.add(egui::TextEdit::singleline(&mut self.cap_text).desired_width(60.0)).lost_focus() {
                    self.settings.cache_cap_gb = self.cap_text.trim().parse::<f64>().ok().filter(|g| *g > 0.0);
                    let _ = self.settings.save();
                    let _ = cache.enforce_cap(self.settings.cache_cap_gb);
                }
            });
            ui.horizontal(|ui| {
                ui.label("default destination");
                let mut d = self.settings.dest;
                ui.radio_value(&mut d, Dest::Beside, "beside the original");
                ui.radio_value(&mut d, Dest::Cache, "the cache");
                if d != self.settings.dest {
                    self.settings.dest = d;
                    let _ = self.settings.save();
                }
            });
            ui.horizontal(|ui| {
                if ui.button("purge all").clicked() {
                    let _ = cache.purge();
                    purged_current = current.is_some_and(|(_, opt, _)| opt);
                }
                if let Some((path, ..)) = current {
                    let has = entries.iter().any(|e| e.source == media::optimise::canonical(path));
                    if ui.add_enabled(has, egui::Button::new("purge this clip's copy")).clicked() {
                        let _ = cache.purge_source(path);
                        purged_current = true;
                    }
                }
            });
            // The local copies of frames read off network storage (image
            // sequences on a share): the same kind of managed cache.
            ui.separator();
            ui.strong("network frames, copied to this disk");
            match media::netcache::get() {
                Some(nc) => {
                    let (b, cap, n) = nc.usage();
                    ui.label(format!("{n} frames, {:.2} of {:.0} GB  ({})", b as f64 / 1e9, cap as f64 / 1e9, nc.dir().display()));
                    ui.weak("filled as sequences on a share play, and by sequence > cache shot locally; EDIFY_NETCACHE_GB sets the cap");
                    if ui.button("clear the network copies").clicked() {
                        nc.clear();
                    }
                }
                None => {
                    ui.weak("off (EDIFY_NETCACHE_GB=0)");
                }
            }
        });
        self.show_cache = open;
        purged_current
    }
}
