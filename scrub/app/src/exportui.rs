//! The render command: the dialog (preset, fit or fill, whole sequence or
//! in..out, where it goes), the background job and its HUD line.
//!
//! Inside an organisicator marketing post the file goes to the post's
//! `out/` (`<slug>_<preset>_v001.mp4`, the next free version) and, when
//! asked, a copy to the marketing folder's `result/<slug>_v###.mp4`, which
//! is how organisicator finds a post's results. Outside a post a save
//! dialog asks. Nothing is ever overwritten and nothing is published.

use crate::editor::Editor;
use crate::render::{self, Fit, Job, Preset, Progress, PRESETS};
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warnings_say_what_the_platforms_dislike() {
        let reels = crate::render::preset("reels").unwrap();
        let yt = crate::render::preset("youtube-1080").unwrap();
        assert!(render_warnings(reels, Some("9:16"), 30.0, 30.0).is_empty());
        assert!(render_warnings(yt, Some("9:16"), 30.0, 30.0)[0].contains("frame shape is 9:16"));
        assert!(render_warnings(reels, None, 120.0, 30.0).iter().any(|w| w.contains("90 s")));
        assert!(render_warnings(yt, None, 10.0, 120.0).iter().any(|w| w.contains("60")));
        assert!(render_warnings(yt, Some("original"), 120.0, 30.0).is_empty());
    }
}

pub struct Running {
    progress: Arc<Progress>,
    handle: JoinHandle<anyhow::Result<render::Done>>,
    started: Instant,
    name: String,
}

pub struct Dialog {
    preset: usize,
    fit: Fit,
    inout: bool,
}

#[derive(Default)]
pub struct ExportUi {
    pub dialog: Option<Dialog>,
    pub job: Option<Running>,
    /// How the last render ended, in words (the scripting API reports it).
    pub last: Option<String>,
}

impl ExportUi {
    pub fn busy(&self) -> bool {
        self.job.is_some()
    }

    pub fn cancel(&self) {
        if let Some(j) = &self.job {
            j.progress.cancel.store(true, Ordering::Relaxed);
        }
    }

    /// (file name, frames done, total, fps) of the running render.
    pub fn progress(&self) -> Option<(String, u64, u64, f64)> {
        let j = self.job.as_ref()?;
        let (d, t) = (j.progress.done.load(Ordering::Relaxed), j.progress.total.load(Ordering::Relaxed));
        Some((j.name.clone(), d, t, d as f64 / j.started.elapsed().as_secs_f64().max(0.01)))
    }

    pub fn hud_line(&self) -> Option<String> {
        let j = self.job.as_ref()?;
        let (d, t) = (j.progress.done.load(Ordering::Relaxed), j.progress.total.load(Ordering::Relaxed).max(1));
        let fps = d as f64 / j.started.elapsed().as_secs_f64().max(0.01);
        Some(format!("rendering {}  {:.0}%  {fps:.0} fps", j.name, d as f64 * 100.0 / t as f64))
    }
}

/// What the platforms will not like about a render (warnings, not refusals).
pub fn render_warnings(p: Preset, post_frame: Option<&str>, secs: f64, fps: f64) -> Vec<String> {
    let mut w = Vec::new();
    let shape = crate::marketing::shape_of((p.width, p.height));
    if let (Some(pf), Some(s)) = (post_frame, shape) {
        if !pf.is_empty() && pf != "original" && pf != s {
            w.push(format!("the post's frame shape is {pf}; this renders {s}"));
        }
    }
    if fps > 60.0 + 1e-6 {
        w.push(format!("{fps:.2} fps: YouTube and Instagram take at most 60"));
    }
    let instagram = p.id != "youtube-1080" && p.id != "youtube-4k";
    if p.id == "reels" && secs > 90.0 {
        w.push(format!("{secs:.0} s long: Reels made in the app stop at 90 s (uploads up to 15 min post as Reels; Stories cut at 60 s)"));
    }
    if instagram && secs > 60.0 * 60.0 {
        w.push("over 60 minutes: Instagram will not take it".into());
    }
    if instagram && secs < 3.0 {
        w.push(format!("{secs:.1} s: Instagram wants at least 3 s"));
    }
    if !instagram && secs > 12.0 * 3600.0 {
        w.push("over 12 hours: YouTube will not take it".into());
    }
    w
}

impl Editor {
    /// cmd+M: the render dialog, defaulting to the preset that matches the
    /// sequence's shape.
    pub fn ask_render(&mut self) {
        if self.rec.seq.is_empty() {
            self.say("nothing to render: the sequence is empty");
            return;
        }
        let raster = self.rec.seq.raster.unwrap_or((1920, 1080));
        let want = render::preset_for(raster);
        let preset = PRESETS.iter().position(|p| *p == want).unwrap_or(0);
        let inout = self.rec.marks.inp.is_some() || self.rec.marks.out.is_some();
        self.export.dialog = Some(Dialog { preset, fit: Fit::Fit, inout });
    }

    /// The record frames to render: in..out (marks set), else everything.
    fn render_range(&self, inout: bool) -> (i64, i64) {
        let d = self.rec.seq.duration();
        if !inout {
            return (0, d);
        }
        let m = self.rec.marks;
        (m.inp.unwrap_or(0).clamp(0, d), m.out.map_or(d, |o| o + 1).clamp(0, d))
    }

    /// A job for the current sequence (the files clips are actually read from).
    pub fn render_job(&self, preset: Preset, fit: Fit, inout: bool, out: PathBuf, copy_to: Option<PathBuf>) -> Job {
        let seq = self.rec.seq.clone();
        let read = self.read_paths(&seq);
        Job { seq, read, range: self.render_range(inout), preset, fit, out, copy_to }
    }

    /// The file each media item of `seq` is read from: the one playback
    /// reads, else its optimised copy, else the original.
    pub(crate) fn read_paths(&self, seq: &model::Sequence) -> Vec<PathBuf> {
        seq.media
            .iter()
            .map(|m| {
                self.rec
                    .program
                    .read_path(&m.path)
                    .map(|p| p.to_path_buf())
                    .or_else(|| media::optimise::find(&m.path, &self.opt.settings))
                    .unwrap_or_else(|| m.path.clone())
            })
            .collect()
    }

    /// Where a render goes: inside a post, its `out/` (and `result/` when
    /// asked); else a save dialog. `None`: cancelled.
    fn render_dest(&self, preset: Preset) -> Option<(PathBuf, Option<PathBuf>)> {
        if let Some(p) = &self.post {
            // Only out/: publishing into result/ is organisicator's step.
            return Some((p.out_path(preset.id), None));
        }
        let name = format!("{}_{}.mp4", self.project.name(), preset.id);
        let mut d = imfram::filedialog::FileDialog::new().add_filter("MP4", &["mp4"]).set_file_name(name);
        if let Some(dir) = self.project.path.as_ref().and_then(|p| p.parent()) {
            d = d.set_directory(dir);
        }
        d.save_file().map(|p| (p, None))
    }

    pub fn start_render(&mut self, job: Job) {
        if self.export.busy() {
            self.say("a render is already running");
            return;
        }
        let progress = Arc::new(Progress::default());
        let p = progress.clone();
        let name = job.out.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let handle = std::thread::Builder::new().name("render".into()).spawn(move || render::run(&job, &p)).expect("spawn render");
        self.export.job = Some(Running { progress, handle, started: Instant::now(), name });
    }

    pub fn export_windows(&mut self, ctx: &egui::Context) {
        // A finished job.
        if self.export.job.as_ref().is_some_and(|j| j.handle.is_finished()) {
            let j = self.export.job.take().expect("checked");
            match j.handle.join() {
                Ok(Ok(d)) => {
                    let also = d.copy.as_ref().map_or(String::new(), |c| format!(" and {}", c.display()));
                    self.say(format!("rendered {} frames in {:.1}s ({:.0} fps, {}): {}{also}", d.frames, d.secs, d.frames as f64 / d.secs, d.encoder, d.out.display()));
                }
                Ok(Err(e)) => self.say(format!("render stopped: {e:#}")),
                Err(_) => self.say("render crashed"),
            }
            self.export.last = Some(self.last_notice());
        }
        let fps = self.rec.seq.fps.as_f64();
        let lens = [self.render_range(false), self.render_range(true)].map(|(a, b)| (b - a) as f64 / fps);
        let Some(dlg) = &mut self.export.dialog else { return };
        let mut open = true;
        let mut go = false;
        let marks = self.rec.marks.inp.is_some() || self.rec.marks.out.is_some();
        let in_post = self.post.as_ref().map(|p| p.title.clone());
        let post_frame = self.post.as_ref().map(|p| p.frame.clone());

        egui::Window::new("render").open(&mut open).collapsible(false).resizable(false).show(ctx, |ui| {
            for (i, p) in PRESETS.iter().enumerate() {
                ui.radio_value(&mut dlg.preset, i, format!("{}  ({}x{}, {} Mbit/s)", p.name, p.width, p.height, p.bitrate / 1_000_000));
            }
            ui.separator();
            ui.horizontal(|ui| {
                ui.label("sequence into the frame");
                ui.radio_value(&mut dlg.fit, Fit::Fit, "fit (bars)");
                ui.radio_value(&mut dlg.fit, Fit::Fill, "fill (crop)");
            });
            ui.horizontal(|ui| {
                ui.radio_value(&mut dlg.inout, false, "whole sequence");
                ui.add_enabled(marks, egui::RadioButton::new(dlg.inout, "in to out")).clicked().then(|| dlg.inout = true);
            });
            match &in_post {
                Some(t) => {
                    ui.label(format!("to post \"{t}\": out/, the next version (publish it from organisicator)"));
                }
                None => {
                    ui.weak("not in a marketing post: you'll be asked where to save");
                }
            }
            let preset = PRESETS[dlg.preset];
            let secs = lens[dlg.inout as usize];
            for w in render_warnings(preset, post_frame.as_deref(), secs, fps) {
                ui.colored_label(imfram::theme::warn(), w);
            }
            ui.weak("H.264 + AAC MP4; never overwrites, never publishes");
            go = ui.button("render  (enter)").clicked() || ui.input(|i| i.key_pressed(egui::Key::Enter));
        });
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            open = false;
        }
        if go {
            let (preset, fit, inout) = (PRESETS[dlg.preset], dlg.fit, dlg.inout);
            self.export.dialog = None;
            if let Some((out, copy_to)) = self.render_dest(preset) {
                let job = self.render_job(preset, fit, inout, out, copy_to);
                self.start_render(job);
            }
            return;
        }
        if !open {
            self.export.dialog = None;
        }
    }
}
