//! Background ProRes playback cache.
//!
//! Each clip VERSION (the full sequence, not the trim) is encoded once to a
//! ProRes 4444 .mov in the cache dir: scene-linear ACEScg in, ACEScct log out,
//! so 12-bit integer holds the range and the view transform still runs live
//! at the blit on playback. Encode is `ffmpeg` with `prores_videotoolbox`
//! (Apple Silicon media engine), falling back to `prores_ks` (CPU).
//!
//! The cache is derived data, keyed by content (paths + mtimes + sizes):
//! a new version hashes differently and gets its own file. Sources are never
//! touched.

use anyhow::{anyhow, Context as _, Result};
use std::sync::mpsc::{channel, Sender};
use engine::color::{Colorspace, Curve, Transform};
use engine::decode::{AnyDecoder, ExrDecoder as _};
use half::f16;
use rayon::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io::Write as _;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

pub type CacheKey = u64;

#[derive(Clone)]
pub enum CacheState {
    Encoding(f32), // 0..1 progress
    #[allow(dead_code)]
    Ready(PathBuf),
    Failed(String),
}

struct Job {
    key: CacheKey,
    frames: Vec<PathBuf>,
    fps: f32,
    out: PathBuf,
    /// Colourspace the frames decode as; transformed to working (ACEScg)
    /// before the ACEScct encode.
    input: Colorspace,
}

pub struct CacheManager {
    dir: PathBuf,
    states: Arc<Mutex<HashMap<CacheKey, CacheState>>>,
    #[allow(dead_code)]
    tx: Sender<Job>,
}

impl CacheManager {
    pub fn new() -> Self {
        let dir = dirs_cache_dir();
        std::fs::create_dir_all(&dir).ok();
        // A killed encode leaves a .part behind; they are worthless on the
        // next run (the job restarts from scratch), so sweep them.
        if let Ok(rd) = std::fs::read_dir(&dir) {
            for e in rd.flatten() {
                if e.path().extension().is_some_and(|x| x == "part") {
                    std::fs::remove_file(e.path()).ok();
                }
            }
        }
        let states: Arc<Mutex<HashMap<CacheKey, CacheState>>> = Default::default();
        let (tx, rx) = channel::<Job>();
        let st = states.clone();
        std::thread::Builder::new()
            .name("prores-cache".into())
            .spawn(move || {
                for job in rx {
                    let key = job.key;
                    let result = encode_job(&job, &st);
                    let mut s = st.lock().unwrap();
                    match result {
                        Ok(path) => s.insert(key, CacheState::Ready(path)),
                        Err(e) => {
                            log::error!("cache encode failed: {e:#}");
                            s.insert(key, CacheState::Failed(format!("{e:#}")))
                        }
                    };
                }
            })
            .unwrap();
        Self { dir, states, tx }
    }

    /// Content hash of a sequence + its input colourspace: identity of the
    /// cache file. A different input pick is different cached pixels.
    #[allow(dead_code)]
    pub fn key_for(frames: &[PathBuf], input: Colorspace) -> CacheKey {
        let mut h = DefaultHasher::new();
        // v3: proxy budgeted by PIXEL COUNT — a long-side cap left square
        // formats (4096x4096 -> 2048x2048) with 2x the pixels of a 16:9
        // proxy, decoding just UNDER playback speed.
        "acescct-prores-v3-proxy2mpx".hash(&mut h);
        input.hash(&mut h);
        for p in frames {
            p.hash(&mut h);
            if let Ok(md) = std::fs::metadata(p) {
                md.len().hash(&mut h);
                if let Ok(t) = md.modified() {
                    t.hash(&mut h);
                }
            }
        }
        h.finish()
    }

    /// Make sure a cache encode exists or is underway for this sequence.
    #[allow(dead_code)]
    pub fn ensure(&self, frames: &[PathBuf], fps: f32, input: Colorspace) -> CacheKey {
        let key = Self::key_for(frames, input);
        let mut s = self.states.lock().unwrap();
        if s.contains_key(&key) {
            return key;
        }
        let out = self.dir.join(format!("{key:016x}.mov"));
        if out.is_file() {
            s.insert(key, CacheState::Ready(out));
            return key;
        }
        s.insert(key, CacheState::Encoding(0.0));
        self.tx
            .send(Job { key, frames: frames.to_vec(), fps, out, input })
            .ok();
        key
    }

    pub fn state(&self, key: CacheKey) -> Option<CacheState> {
        self.states.lock().unwrap().get(&key).cloned()
    }

    /// Forget every state and delete the cache files; returns how many files
    /// went. Callers re-ensure afterwards, which re-encodes what's in use.
    pub fn clear(&self) -> usize {
        let mut n = 0;
        if let Ok(rd) = std::fs::read_dir(&self.dir) {
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().is_some_and(|x| x == "mov" || x == "part")
                    && std::fs::remove_file(&p).is_ok()
                {
                    n += 1;
                }
            }
        }
        self.states.lock().unwrap().clear();
        n
    }
}

fn dirs_cache_dir() -> PathBuf {
    // Windows: %LOCALAPPDATA%. Without it the Mac path was joined onto an
    // empty HOME, and the cache landed in `Library\Caches` under whatever
    // directory the app was started from.
    if cfg!(windows) {
        if let Some(d) = std::env::var_os("LOCALAPPDATA") {
            return PathBuf::from(d).join("WobbleEdit").join("Cache");
        }
    }
    let home = engine::paths::home().unwrap_or_default();
    home.join("Library/Caches/WobbleEdit")
}

fn encoder_args(encoder: &str) -> Vec<String> {
    match encoder {
        "prores_videotoolbox" => vec![
            "-c:v".into(),
            "prores_videotoolbox".into(),
            "-profile:v".into(),
            "4444".into(),
            "-allow_sw".into(),
            "1".into(),
        ],
        _ => vec![
            "-c:v".into(),
            "prores_ks".into(),
            "-profile:v".into(),
            "4444".into(),
            "-pix_fmt".into(),
            "yuv444p10le".into(),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::decode::ImageInfo;

    /// End-to-end colour round trip: EXR (scene-linear) -> ACEScct ProRes
    /// 4444 -> decode -> back to linear must match the source. Catches a
    /// wrong YUV matrix, range mishandling, or a broken cct curve at once.
    #[test]
    fn prores_roundtrip_matches_source() {
        let seq_dir = std::path::Path::new("testdata/shotA/v001");
        if !seq_dir.is_dir() {
            eprintln!("testdata missing; skipping");
            return;
        }
        let mut frames: Vec<PathBuf> = std::fs::read_dir(seq_dir)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "exr"))
            .collect();
        frames.sort();
        frames.truncate(4);

        let mgr = CacheManager::new();
        let key = mgr.ensure(&frames, 24.0, Colorspace::ACESCG);
        let deadline = web_time::Instant::now() + web_time::Duration::from_secs(120);
        let mov = loop {
            match mgr.state(key) {
                Some(CacheState::Ready(p)) => break p,
                Some(CacheState::Failed(e)) => panic!("encode failed: {e}"),
                _ => {
                    assert!(web_time::Instant::now() < deadline, "encode timed out");
                    std::thread::sleep(web_time::Duration::from_millis(100));
                }
            }
        };

        let decoder = AnyDecoder::new(true);
        let info: ImageInfo = decoder.read_header(&frames[0]).unwrap();
        let stride = info.row_stride_f16();
        let n = stride * info.height as usize;
        let mut exr = vec![f16::ZERO; n];
        let mut movf = vec![f16::ZERO; n];
        decoder.decode_into(&frames[0], info, stride, &mut exr).unwrap();
        decoder
            .decode_frame_into(&mov, info, stride, None, 0, &mut movf)
            .unwrap();

        // Compare in the cct domain (what the file stores): uniform tolerance.
        let mut max_err = 0f32;
        let mut sum_err = 0f64;
        let mut count = 0u64;
        for (w, row) in (0..info.height as usize).map(|y| (y, y * stride)) {
            let _ = w;
            for x in 0..info.width as usize {
                for c in 0..3 {
                    let a = Curve::AcesCct
                        .from_linear(exr[row + x * 4 + c].to_f32())
                        .clamp(0.0, 1.0);
                    let b = movf[row + x * 4 + c].to_f32().clamp(0.0, 1.0);
                    let e = (a - b).abs();
                    max_err = max_err.max(e);
                    sum_err += e as f64;
                    count += 1;
                }
            }
        }
        let mean = sum_err / count as f64;
        eprintln!("cct-domain error: mean {mean:.5} max {max_err:.5}");
        assert!(mean < 0.004, "mean cct error too high: {mean}");
        assert!(max_err < 0.05, "max cct error too high: {max_err}");
    }
}

fn encode_job(job: &Job, states: &Arc<Mutex<HashMap<CacheKey, CacheState>>>) -> Result<PathBuf> {
    for (i, enc) in ["prores_videotoolbox", "prores_ks"].iter().enumerate() {
        match encode_with(job, enc, states) {
            Ok(p) => return Ok(p),
            Err(e) if i == 0 => log::warn!("{enc} failed ({e:#}), falling back to prores_ks"),
            Err(e) => return Err(e),
        }
    }
    unreachable!()
}

fn encode_with(
    job: &Job,
    encoder: &str,
    states: &Arc<Mutex<HashMap<CacheKey, CacheState>>>,
) -> Result<PathBuf> {
    let decoder = AnyDecoder::new(false);
    let info = decoder.read_header(&job.frames[0])?;
    let (w, h) = (info.width as usize, info.height as usize);
    // The cache is a PLAYBACK proxy budgeted by PIXELS (aspect-independent):
    // software ProRes decodes ~90M px/s here, so ~2M px keeps ~45fps —
    // comfortable over any playback rate. Exports always re-read the true
    // frames anyway.
    const PROXY_PX_BUDGET: f64 = 2.0e6;
    let (pw, ph) = if (w * h) as f64 > PROXY_PX_BUDGET {
        let s = (PROXY_PX_BUDGET / (w * h) as f64).sqrt();
        (
            ((w as f64 * s / 2.0).round() as usize * 2).max(2),
            ((h as f64 * s / 2.0).round() as usize * 2).max(2),
        )
    } else {
        (w, h)
    };
    let tmp = job.out.with_extension("mov.part");

    let mut args: Vec<String> = vec![
        "-y".into(),
        "-loglevel".into(),
        "error".into(),
        "-f".into(),
        "rawvideo".into(),
        "-pixel_format".into(),
        "rgb48le".into(),
        "-video_size".into(),
        format!("{w}x{h}"),
        "-framerate".into(),
        format!("{}", job.fps),
        "-i".into(),
        "pipe:0".into(),
    ];
    if (pw, ph) != (w, h) {
        args.extend(["-vf".into(), format!("scale={pw}:{ph}:flags=bilinear")]);
    }
    args.extend(encoder_args(encoder));
    // Tag the stream so decode uses the same matrix we were converted with.
    args.extend(
        ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
            .map(String::from),
    );
    // The temp file ends in .part, so the container must be explicit.
    args.extend(["-f".into(), "mov".into()]);
    args.push(tmp.to_string_lossy().into_owned());

    let mut child = Command::new("ffmpeg")
        .args(&args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .context("spawn ffmpeg (is it installed? brew install ffmpeg)")?;
    let mut stdin = child.stdin.take().unwrap();

    let stride = info.row_stride_f16();
    let mut f16buf = vec![f16::ZERO; stride * h];
    let mut out = vec![0u8; w * h * 6]; // rgb48le, tight rows
    let total = job.frames.len();

    let mut feed = || -> Result<()> {
        for (fi, path) in job.frames.iter().enumerate() {
            let finfo = decoder.read_header(path)?;
            if finfo.width != info.width || finfo.height != info.height {
                return Err(anyhow!(
                    "frame size changed mid-sequence at {}",
                    path.display()
                ));
            }
            decoder.decode_into(path, info, stride, &mut f16buf)?;
            // Input space -> scene-linear AP1 -> ACEScct, clamped to [0,1]
            // of the 16-bit pipe. (Clips linear above ~223; irrelevant for
            // playback.)
            let tf = Transform::new(job.input, Colorspace::ACESCG);
            let identity = tf.is_identity();
            out.par_chunks_mut(w * 6)
                .zip(f16buf.par_chunks(stride))
                .for_each(|(orow, irow)| {
                    for x in 0..w {
                        let mut rgb = [
                            irow[x * 4].to_f32(),
                            irow[x * 4 + 1].to_f32(),
                            irow[x * 4 + 2].to_f32(),
                        ];
                        if !identity {
                            rgb = tf.apply(rgb);
                        }
                        for c in 0..3 {
                            let cct = Curve::AcesCct.from_linear(rgb[c]).clamp(0.0, 1.0);
                            let v = (cct * 65535.0 + 0.5) as u16;
                            orow[x * 6 + c * 2..x * 6 + c * 2 + 2]
                                .copy_from_slice(&v.to_le_bytes());
                        }
                    }
                });
            stdin.write_all(&out)?;
            states
                .lock()
                .unwrap()
                .insert(job.key, CacheState::Encoding((fi + 1) as f32 / total as f32));
        }
        Ok(())
    };
    let fed = feed();
    drop(stdin);
    let output = child.wait_with_output()?;
    // A feed error is usually just the broken pipe after ffmpeg died;
    // ffmpeg's own stderr is the useful message, so report that first.
    if !output.status.success() {
        std::fs::remove_file(&tmp).ok();
        return Err(anyhow!(
            "ffmpeg ({encoder}) exited {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    fed?;
    std::fs::rename(&tmp, &job.out)?;
    Ok(job.out.clone())
}

