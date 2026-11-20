//! A movie Render's encoder: one thread owning an `engine::movie::MovieWriter`,
//! fed the frames in order through a short queue.
//!
//! An EXR Render writes each frame on any worker, in any order; a movie is
//! ONE stream, so its frames go through here instead. The queue is short so
//! a slow encoder holds the render back rather than piling frames up in
//! memory. Dropping a `MovieOut` (a cancelled render) closes the file with
//! what it has, which is a playable movie.

use std::sync::{Arc, Mutex};

/// One rendered frame on its way to the encoder: the readback's padded
/// rows of half RGBA, output space already applied.
pub(crate) struct Frame {
    pub px: Vec<half::f16>,
    /// Halves per padded row.
    pub stride: usize,
    /// One channel as grey (the Render's channel pick): weights over the
    /// four lanes, alpha then opaque. `None` = the picture as it is.
    pub mono: Option<[f32; 4]>,
}

#[cfg_attr(not(feature = "movie"), allow(dead_code))]
pub(crate) struct MovieOut {
    tx: Option<std::sync::mpsc::SyncSender<Frame>>,
    handle: Option<std::thread::JoinHandle<Result<i64, String>>>,
}

impl MovieOut {
    /// Open `path` for a movie of `w`×`h` (already even where the codec
    /// needs it; a larger rendered frame is cropped to it) and start the
    /// encoder. A failure on the way (a write that fails mid-render) lands
    /// in `err`, which the render loop checks every slice.
    #[cfg(feature = "movie")]
    pub fn start(
        path: &std::path::Path,
        (w, h): (u32, u32),
        fps: f32,
        format: engine::graph::WriteFormat,
        rec709: bool,
        err: Arc<Mutex<Option<String>>>,
    ) -> Result<Self, String> {
        let (tx, rx) = std::sync::mpsc::sync_channel::<Frame>(3);
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();
        let path = path.to_path_buf();
        // FFmpeg's contexts live and die on this one thread.
        let handle = std::thread::Builder::new()
            .name("movie-render".into())
            .spawn(move || -> Result<i64, String> {
                let mut mw = match engine::movie::MovieWriter::create(&path, w, h, fps, format, rec709) {
                    Ok(mw) => {
                        let _ = ready_tx.send(Ok(()));
                        mw
                    }
                    Err(e) => {
                        let msg = format!("{e:#}");
                        let _ = ready_tx.send(Err(msg.clone()));
                        return Err(msg);
                    }
                };
                let mut tight = vec![0u16; (w * h * 4) as usize];
                for f in rx {
                    to_u16(&f, w as usize, h as usize, &mut tight);
                    if let Err(e) = mw.push(&tight) {
                        let msg = format!("{e:#}");
                        err.lock().unwrap().get_or_insert(msg.clone());
                        return Err(msg);
                    }
                }
                let n = mw.frames();
                mw.finish().map_err(|e| {
                    let msg = format!("{e:#}");
                    err.lock().unwrap().get_or_insert(msg.clone());
                    msg
                })?;
                Ok(n)
            })
            .map_err(|e| format!("encoder thread: {e}"))?;
        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self { tx: Some(tx), handle: Some(handle) }),
            Ok(Err(e)) => Err(e),
            Err(_) => Err("the encoder thread died opening the file".into()),
        }
    }

    /// No FFmpeg in this build: say how to get one.
    #[cfg(not(feature = "movie"))]
    pub fn start(
        _path: &std::path::Path,
        _size: (u32, u32),
        _fps: f32,
        format: engine::graph::WriteFormat,
        _rec709: bool,
        _err: Arc<Mutex<Option<String>>>,
    ) -> Result<Self, String> {
        Err(format!(
            "{} needs a movie build (cargo build -p app --features app/movie)",
            format.label()
        ))
    }

    /// Can this build write `format`? `Err` says why not.
    pub fn can_encode(format: engine::graph::WriteFormat) -> Result<(), String> {
        #[cfg(feature = "movie")]
        return engine::movie::can_encode(format);
        #[cfg(not(feature = "movie"))]
        return Err(format!("{} needs a movie build (--features app/movie)", format.label()));
    }

    /// Queue the next frame; blocks while the queue is full. `false` when
    /// the encoder has stopped (its error is in `err`).
    pub fn send(&self, f: Frame) -> bool {
        self.tx.as_ref().is_some_and(|tx| tx.send(f).is_ok())
    }

    /// Every frame sent: flush and close the file. The frame count, or why
    /// it failed.
    pub fn finish(mut self) -> Result<i64, String> {
        drop(self.tx.take());
        match self.handle.take().map(|h| h.join()) {
            Some(Ok(r)) => r,
            Some(Err(_)) => Err("the movie encoder panicked".into()),
            None => Err("no encoder".into()),
        }
    }
}

impl Drop for MovieOut {
    fn drop(&mut self) {
        // Closing the queue lets the thread write the trailer for what it
        // has; it is not waited for.
        drop(self.tx.take());
    }
}

/// Padded half RGBA to tight 16-bit code values at `w`×`h` (the top-left of
/// the frame, which is at least that big), clamped to [0, 1]: a movie holds
/// display code values, and anything outside the range is clipped as any
/// display file would.
#[cfg_attr(not(feature = "movie"), allow(dead_code))]
fn to_u16(f: &Frame, w: usize, h: usize, out: &mut [u16]) {
    use rayon::prelude::*;
    let q = |v: f32| (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
    out.par_chunks_mut(w * 4).enumerate().for_each(|(y, dst)| {
        let src = &f.px[y * f.stride..y * f.stride + w * 4];
        for (d, s) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
            let s = [s[0].to_f32(), s[1].to_f32(), s[2].to_f32(), s[3].to_f32()];
            match f.mono {
                None => {
                    for c in 0..4 {
                        d[c] = q(s[c]);
                    }
                }
                Some(wt) => {
                    let g = q(s[0] * wt[0] + s[1] * wt[1] + s[2] * wt[2] + s[3] * wt[3]);
                    d[..3].fill(g);
                    d[3] = u16::MAX;
                }
            }
        }
    });
    debug_assert!(h * w * 4 == out.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_are_cropped_clamped_and_greyed() {
        let f16 = half::f16::from_f32;
        // 3x2 padded to a stride of 16 halves, cropped to 2x2.
        let mut px = vec![f16(0.0); 32];
        px[0..4].copy_from_slice(&[f16(1.5), f16(-0.2), f16(0.5), f16(1.0)]);
        px[16..20].copy_from_slice(&[f16(0.25), f16(0.5), f16(0.75), f16(0.0)]);
        let mut out = vec![0u16; 16];
        to_u16(&Frame { px: px.clone(), stride: 16, mono: None }, 2, 2, &mut out);
        assert_eq!(&out[0..4], &[65535, 0, 32768, 65535]);
        assert_eq!(out[8], 16384);
        to_u16(&Frame { px, stride: 16, mono: Some([0.0, 1.0, 0.0, 0.0]) }, 2, 2, &mut out);
        assert_eq!(&out[8..12], &[32768, 32768, 32768, 65535]);
    }
}
