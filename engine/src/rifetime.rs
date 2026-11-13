//! The Rifetime node's work: a directory of rendered frames in, a directory
//! of retimed frames out, named by output comp frame.
//!
//! Per output frame, `RifetimeParams::plan` says which source frame and how
//! far towards the next. On a source frame the file is copied. Between two,
//! RIFE (`crate::rife`) is run at that time on the sRGB-encoded, clamped
//! pair — what it was trained on — and its flow and blend mask are applied
//! to the LINEAR, unclamped frames: `out = m·I0(p + F→0) + (1 − m)·I1(p +
//! F→1)`, bilinear with the edge held (torch's `grid_sample`, border). So
//! the network decides where things went and the pixels keep their HDR.
//! Alpha is warped with the colour.
//!
//! Runs on the caller's thread (the app puts it on a worker), cancellable
//! between frames; the done-marker is written last.

use crate::decode::ExrDecoder as _;
use crate::rife::gpu::RifeGpu;
use crate::rife::RifeModel;
use anyhow::{Context, Result};
use half::f16;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub struct Job {
    /// RIFE's `flownet.pkl`.
    pub weights: PathBuf,
    /// Per output frame: two input frames and how far from the first
    /// towards the second (`adjacent` of `RifetimeParams::plan` or
    /// `plan_times`; a Frame Fix pairs the good frames round a bad run).
    /// A fraction of 0 is a copy of the first.
    pub plan: Vec<(usize, usize, f32)>,
    /// Analyse at `1 / divisor` of the size.
    pub divisor: usize,
    /// The input frames, in time order.
    pub frames: Vec<PathBuf>,
    /// The comp frame the first output lands on (the input's first).
    pub first: u64,
    pub out: PathBuf,
}

/// A Rifetime plan (frame `i`, fraction towards `i + 1`) as job steps.
pub fn adjacent(plan: Vec<(usize, f32)>) -> Vec<(usize, usize, f32)> {
    plan.into_iter().map(|(i, t)| (i, i + 1, t)).collect()
}

/// Copy `from` to `to` as a hard link where the filesystem allows (a good
/// frame is the rendered input byte for byte, so it need not take the disk
/// twice), else a plain copy.
fn link_or_copy(from: &Path, to: &Path) -> std::io::Result<()> {
    let _ = std::fs::remove_file(to);
    std::fs::hard_link(from, to).or_else(|_| std::fs::copy(from, to).map(|_| ()))
}

/// The output file for comp frame `tick`.
pub fn frame_name(tick: u64) -> String {
    format!("f_{tick:05}.exr")
}

/// A decoded frame: `[4][h][w]` linear RGBA, and the network's view of it,
/// `[3][sh][sw]` encoded and shrunk.
struct Frame {
    rgba: Vec<f32>,
    net: Vec<f32>,
}

fn load(dec: &crate::decode::AnyDecoder, path: &Path, d: usize) -> Result<(usize, usize, Frame)> {
    let info = dec.read_header(path).with_context(|| format!("reading {}", path.display()))?;
    let (w, h) = (info.width as usize, info.height as usize);
    let stride = info.row_stride_f16();
    let mut buf = vec![f16::ZERO; stride * h];
    dec.decode_into(path, info, stride, &mut buf)?;
    let n = w * h;
    let mut rgba = vec![0f32; 4 * n];
    let mut enc = vec![0f32; 3 * n];
    for y in 0..h {
        for x in 0..w {
            let px = &buf[y * stride + x * 4..][..4];
            for c in 0..4 {
                let v = px[c].to_f32();
                rgba[c * n + y * w + x] = v;
                if c < 3 {
                    enc[c * n + y * w + x] = crate::flowsolve::srgb_encode(v);
                }
            }
        }
    }
    let net = crate::flowsolve::shrink(&enc, 3, w, h, d).2;
    Ok((w, h, Frame { rgba, net }))
}

/// One plane at `sw × sh` grown to `w × h`, bilinear about pixel centres,
/// values unscaled (a mask, not a motion).
fn grow_plane(src: &[f32], sw: usize, sh: usize, w: usize, h: usize, d: usize) -> Vec<f32> {
    if d <= 1 {
        return src.to_vec();
    }
    // `grow` scales by d (it is for offsets): undo that for a mask.
    let two: Vec<f32> = src.iter().chain(src.iter()).copied().collect();
    let g = crate::flowsolve::grow(&two, sw, sh, w, h, d);
    g[..w * h].iter().map(|v| v / d as f32).collect()
}

/// The in-between at `t` from `a` to `b`.
fn between(gpu: &RifeGpu, a: &Frame, b: &Frame, t: f32, (w, h): (usize, usize), (sw, sh): (usize, usize), d: usize) -> Result<Vec<f32>> {
    let out = gpu.run(&a.net, &b.net, t)?;
    let sn = sw * sh;
    let flow = crate::flowsolve::grow(&out.flow[..2 * sn], sw, sh, w, h, d);
    let flow1 = crate::flowsolve::grow(&out.flow[2 * sn..4 * sn], sw, sh, w, h, d);
    let mask = grow_plane(&out.mask, sw, sh, w, h, d);
    let w0 = crate::rife::reference::warp(&a.rgba, 4, h, w, &flow, 0, false);
    let w1 = crate::rife::reference::warp(&b.rgba, 4, h, w, &flow1, 0, false);
    let n = w * h;
    let mut px = vec![0f32; 4 * n];
    for c in 0..4 {
        for i in 0..n {
            let m = mask[i];
            px[c * n + i] = m * w0[c * n + i] + (1.0 - m) * w1[c * n + i];
        }
    }
    Ok(px)
}

fn write(path: &Path, w: usize, h: usize, planes: &[f32]) -> Result<()> {
    let n = w * h;
    let mut buf = vec![f16::ZERO; 4 * n];
    for i in 0..n {
        for c in 0..4 {
            buf[i * 4 + c] = f16::from_f32(planes[c * n + i]);
        }
    }
    crate::bakes::write_rgba_f16_exr_tagged(path, w, h, 4 * w, &buf, None)
        .with_context(|| format!("writing {}", path.display()))
}

/// Make every output frame of `job`, reporting `(done, total)` after each.
pub fn run(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    job: &Job,
    progress: &mut dyn FnMut(usize, usize),
    cancel: &AtomicBool,
) -> Result<()> {
    let plan = &job.plan;
    anyhow::ensure!(!plan.is_empty() && !job.frames.is_empty(), "the input has no frames");
    anyhow::ensure!(
        plan.iter().all(|(i, k, t)| *i < job.frames.len() && (*t == 0.0 || *k < job.frames.len())),
        "the plan reads past the input"
    );
    std::fs::create_dir_all(&job.out).with_context(|| format!("creating {}", job.out.display()))?;
    let d = job.divisor.max(1);
    let dec = crate::decode::AnyDecoder::default();
    // The model and a planned GPU only once an in-between is wanted.
    let mut gpu: Option<(RifeGpu, (usize, usize), (usize, usize))> = None;
    let mut pair: Option<((usize, usize), Frame, Frame)> = None;
    let total = plan.len();
    for (j, &(i, k, t)) in plan.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            anyhow::bail!("cancelled");
        }
        let file = job.out.join(frame_name(job.first + j as u64));
        if t == 0.0 {
            link_or_copy(&job.frames[i], &file).with_context(|| format!("copying {}", job.frames[i].display()))?;
        } else {
            if pair.as_ref().map(|p| p.0) != Some((i, k)) {
                let (w, h, a) = load(&dec, &job.frames[i], d)?;
                let (w1, h1, b) = load(&dec, &job.frames[k], d)?;
                anyhow::ensure!((w, h) == (w1, h1), "frame {k} is {w1}x{h1}, frame {i} {w}x{h}");
                if gpu.as_ref().is_none_or(|g| g.1 != (w, h)) {
                    let m = RifeModel::load_pkl(&job.weights).context("loading the RIFE weights")?;
                    let (sw, sh) = (w.div_ceil(d), h.div_ceil(d));
                    gpu = Some((RifeGpu::new(device.clone(), queue.clone(), &m, sw, sh)?, (w, h), (sw, sh)));
                }
                pair = Some(((i, k), a, b));
            }
            let (g, size, small) = gpu.as_ref().expect("planned above");
            let (_, a, b) = pair.as_ref().expect("loaded above");
            write(&file, size.0, size.1, &between(g, a, b, t, *size, *small, d)?)?;
        }
        progress(j + 1, total);
    }
    crate::bakes::mark_baked(&job.out).context("marking the frames done")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A texture moving 4 px a frame, retimed to half speed: the made frame
    /// halfway between two sources is the texture moved 2 px, not a blend
    /// of the two (which is what a Speed node would show). Needs
    /// `data/rife/flownet.pkl` and a GPU.
    #[test]
    fn a_half_speed_in_between_is_the_halfway_picture() {
        let weights = Path::new("../data/rife/flownet.pkl");
        if !weights.exists() {
            eprintln!("SKIP: no data/rife/flownet.pkl");
            return;
        }
        let Ok(ctx) = crate::GpuContext::new() else {
            eprintln!("SKIP: no GPU");
            return;
        };
        let dir = std::env::temp_dir().join(format!("wobble-retime-test-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("in")).unwrap();
        let (w, h) = (192usize, 128usize);
        // Smooth, aperiodic, HDR in places (values past 1 must survive).
        let tex = |x: f32, y: f32| {
            let v = 0.5 + 0.25 * (x * 0.13 + y * 0.05).sin() + 0.2 * (x * 0.041 - y * 0.11 + 1.0).sin();
            v * v * 2.5
        };
        let shot = |shift: f32| {
            let mut px = vec![f16::ZERO; w * h * 4];
            for y in 0..h {
                for x in 0..w {
                    let v = tex(x as f32 - shift, y as f32);
                    let i = (y * w + x) * 4;
                    px[i] = f16::from_f32(v);
                    px[i + 1] = f16::from_f32(tex(x as f32 - shift + 30.0, y as f32 + 9.0));
                    px[i + 2] = f16::from_f32(v * 0.4);
                    px[i + 3] = f16::ONE;
                }
            }
            px
        };
        let mut frames = vec![];
        for f in 0..2 {
            let p = dir.join("in").join(frame_name(f));
            crate::bakes::write_rgba_f16_exr_tagged(&p, w, h, w * 4, &shot(4.0 * f as f32), None).unwrap();
            frames.push(p);
        }
        let job = Job {
            weights: weights.into(),
            plan: adjacent(crate::nodes::rifetime::RifetimeParams::default().plan(frames.len())),
            divisor: 1,
            frames,
            first: 10,
            out: dir.join("out"),
        };
        run(ctx.device.clone(), ctx.queue.clone(), &job, &mut |_, _| {}, &AtomicBool::new(false)).unwrap();
        assert!(crate::bakes::is_baked(&job.out));
        let names: Vec<String> = (10..13).map(frame_name).collect();
        for n in &names {
            assert!(job.out.join(n).is_file(), "{n} missing");
        }
        let dec = crate::decode::AnyDecoder::default();
        let read = |p: &Path| {
            let info = dec.read_header(p).unwrap();
            let mut buf = vec![f16::ZERO; info.row_stride_f16() * h];
            dec.decode_into(p, info, info.row_stride_f16(), &mut buf).unwrap();
            buf
        };
        let mid = read(&job.out.join(&names[1]));
        let want = shot(2.0);
        let (mut made, mut blend, mut n) = (0.0f32, 0.0f32, 0.0f32);
        let (a, b) = (shot(0.0), shot(4.0));
        for y in 16..h - 16 {
            for x in 16..w - 16 {
                let i = (y * w + x) * 4;
                made += (mid[i].to_f32() - want[i].to_f32()).abs();
                blend += (0.5 * (a[i].to_f32() + b[i].to_f32()) - want[i].to_f32()).abs();
                n += 1.0;
            }
        }
        let (made, blend) = (made / n, blend / n);
        eprintln!("mean error of the made frame {made:.4}, of a cross-fade {blend:.4}");
        assert!(made < blend * 0.25, "the in-between is no better than a cross-fade ({made} vs {blend})");
        // Frames on source frames are the sources, byte for byte.
        assert_eq!(std::fs::read(&job.frames[1]).unwrap(), std::fs::read(job.out.join(&names[2])).unwrap());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
