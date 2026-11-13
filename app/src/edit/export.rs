//! Export: conform-copy the edit to a single renumbered sequence of the
//! ORIGINAL files (true data — the ProRes cache is never a source). Frames
//! keep their source format (EXR/DPX); a Wobble comp is the place for
//! anything fancier.

use crate::edit::model::{Media, Project};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Clone, Default)]
pub struct ExportStatus(pub Arc<Mutex<Option<String>>>);

pub fn export_copy(project: &Project, out_dir: PathBuf, status: ExportStatus) {
    let jobs: Vec<(PathBuf, usize)> = {
        let mut v = Vec::new();
        let mut n = 1usize;
        for clip in &project.clips {
            match clip.media() {
                Media::Frames(f) => {
                    for local in 0..clip.len() {
                        v.push((f[clip.in_off + local].clone(), n));
                        n += 1;
                    }
                }
                // A movie is copied whole (trims are play-range only in the
                // POC); the numbering still advances by the clip's length.
                Media::Movie { path, .. } => {
                    v.push((path.clone(), n));
                    n += clip.len();
                }
                // No files to copy; the numbering still keeps its place.
                Media::Blank { .. } => n += clip.len(),
            }
        }
        v
    };
    let total = jobs.len();
    std::thread::spawn(move || {
        std::fs::create_dir_all(&out_dir).ok();
        for (i, (src, n)) in jobs.iter().enumerate() {
            let ext = src
                .extension()
                .map(|e| e.to_string_lossy().into_owned())
                .unwrap_or_else(|| "exr".into());
            let dst = out_dir.join(format!("edit.{n:05}.{ext}"));
            if let Err(e) = std::fs::copy(src, &dst) {
                *status.0.lock().unwrap() =
                    Some(format!("export failed at {}: {e}", src.display()));
                return;
            }
            if i % 10 == 0 {
                *status.0.lock().unwrap() =
                    Some(format!("exporting {}/{total}…", i + 1));
            }
        }
        *status.0.lock().unwrap() = Some(format!("exported {total} frames ✓"));
    });
}
