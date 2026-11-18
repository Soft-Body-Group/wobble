//! The Flow node's analysis: a directory of rendered frames in, a
//! directory of motion-vector EXRs out, one per frame, same names, by
//! either network (`crate::searaft` or `crate::rife`).
//!
//! Input frames are linear Rec.709 (the app renders the node's input that
//! way, as it does for every tool that wants a picture); the network was
//! trained on sRGB-encoded 0..1 pictures, so they are encoded and clamped
//! on the way in. Output files are RGBA half: `rg` = forward motion (where
//! the pixel is one frame later), `ba` = backward (where it was one frame
//! earlier), in full-resolution pixels, y down.
//!
//! SEA-RAFT's flow is the motion from the first frame to the second, as it
//! stands. RIFE's needs care. IFNet is asked for the MIDDLE of the pair (`t = 0.5`), where it was
//! trained to be good: `F_t→1 - F_t→0` there is the whole frame's motion,
//! anchored half way. Asked at `t = 0` instead (the obvious reading, and
//! what the first port did) it under-reads anything past a few pixels —
//! measured on a texture moved (6, 4) px: (5.5, 2.3) at `t = 0`, (6.0, 4.1)
//! from the middle. The midpoint field is then carried back to the frame
//! by a fixed point, `f(p) = M(p + f(p)/2)`, so a moving edge's vectors sit
//! on the edge where it IS, not half a move ahead. Backward is the same
//! with the pair reversed (a second run). At the ends of the range the
//! missing neighbour is the constant-velocity guess, the other vector
//! negated; with `backward` off it always is.
//!
//! Runs on the caller's thread and blocks on the GPU per frame: the app
//! puts it on a worker. Cancellable between frames; a partial run leaves
//! no done-marker, so it is never mistaken for a finished one.

use crate::decode::ExrDecoder as _;
use crate::nodes::flow::FlowEngine;
use crate::rife::gpu::RifeGpu;
use crate::rife::RifeModel;
use anyhow::{Context, Result};
use half::f16;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// What to solve.
pub struct Job {
    pub engine: FlowEngine,
    /// The engine's weights (`models::ready_path(engine.model())`).
    pub weights: PathBuf,
    /// The frames to analyse, in time order.
    pub frames: Vec<PathBuf>,
    /// Where each frame's vectors go: same file name, this directory.
    pub out: PathBuf,
    /// Analyse at `1 / divisor` of the size.
    pub divisor: usize,
    pub backward: bool,
}

/// What a Morph Dissolve solves: two clips, the motion BETWEEN them at each
/// tick rather than along either one.
pub struct PairJob {
    pub engine: FlowEngine,
    pub weights: PathBuf,
    /// A's frames and B's, same length, index for index the same tick.
    pub a: Vec<PathBuf>,
    pub b: Vec<PathBuf>,
    /// One file per tick, named after A's frame: `rg` = A→B, `ba` = B→A.
    pub out: PathBuf,
    pub divisor: usize,
    /// Multiplies the vectors on the way out: full-res pixels per rendered
    /// pixel, so a solve made under proxy is in the same units as any other.
    pub scale: f32,
}

/// Decode one frame to `[3][h][w]` planes, sRGB-encoded and clamped to 0..1
/// for the network.
pub(crate) fn load(dec: &crate::decode::AnyDecoder, path: &Path) -> Result<(usize, usize, Vec<f32>)> {
    let info = dec.read_header(path).with_context(|| format!("reading {}", path.display()))?;
    let (w, h) = (info.width as usize, info.height as usize);
    let stride = info.row_stride_f16();
    let mut buf = vec![f16::ZERO; stride * h];
    dec.decode_into(path, info, stride, &mut buf)?;
    let mut rgb = vec![0f32; 3 * w * h];
    for y in 0..h {
        for x in 0..w {
            let px = &buf[y * stride + x * 4..][..4];
            for c in 0..3 {
                rgb[c * w * h + y * w + x] = srgb_encode(px[c].to_f32());
            }
        }
    }
    Ok((w, h, rgb))
}

pub(crate) fn srgb_encode(v: f32) -> f32 {
    let v = if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 };
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// A `d`-times box reduction of `[c][h][w]` planes; the size rounds up and
/// the last row and column average what they cover.
pub fn shrink(src: &[f32], c: usize, w: usize, h: usize, d: usize) -> (usize, usize, Vec<f32>) {
    if d <= 1 {
        return (w, h, src.to_vec());
    }
    let (sw, sh) = (w.div_ceil(d), h.div_ceil(d));
    let mut out = vec![0f32; c * sw * sh];
    for ch in 0..c {
        for y in 0..sh {
            for x in 0..sw {
                let (mut acc, mut n) = (0.0, 0.0);
                for yy in y * d..((y + 1) * d).min(h) {
                    for xx in x * d..((x + 1) * d).min(w) {
                        acc += src[ch * w * h + yy * w + xx];
                        n += 1.0;
                    }
                }
                out[ch * sw * sh + y * sw + x] = acc / n;
            }
        }
    }
    (sw, sh, out)
}

/// Two flow planes at `[2][sh][sw]` back up to `w × h`, bilinear about
/// pixel centres, the offsets multiplied by the same factor.
pub fn grow(src: &[f32], sw: usize, sh: usize, w: usize, h: usize, d: usize) -> Vec<f32> {
    if d <= 1 {
        return src.to_vec();
    }
    let mut out = vec![0f32; 2 * w * h];
    let k = d as f32;
    for ch in 0..2 {
        let plane = &src[ch * sw * sh..][..sw * sh];
        for y in 0..h {
            let fy = ((y as f32 + 0.5) / k - 0.5).clamp(0.0, (sh - 1) as f32);
            let (y0, ty) = (fy.floor() as usize, fy.fract());
            let y1 = (y0 + 1).min(sh - 1);
            for x in 0..w {
                let fx = ((x as f32 + 0.5) / k - 0.5).clamp(0.0, (sw - 1) as f32);
                let (x0, tx) = (fx.floor() as usize, fx.fract());
                let x1 = (x0 + 1).min(sw - 1);
                let a = plane[y0 * sw + x0] * (1.0 - tx) + plane[y0 * sw + x1] * tx;
                let b = plane[y1 * sw + x0] * (1.0 - tx) + plane[y1 * sw + x1] * tx;
                out[ch * w * h + y * w + x] = (a * (1.0 - ty) + b * ty) * k;
            }
        }
    }
    out
}

/// Bilinear read of a `[2][h][w]` field at continuous pixel `(x, y)`
/// (centres at +0.5), clamped to the edge.
fn sample2(f: &[f32], w: usize, h: usize, x: f32, y: f32) -> [f32; 2] {
    let fx = (x - 0.5).clamp(0.0, (w - 1) as f32);
    let fy = (y - 0.5).clamp(0.0, (h - 1) as f32);
    let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
    let n = w * h;
    let at = |c: usize| {
        let p = &f[c * n..];
        let a = p[y0 * w + x0] * (1.0 - tx) + p[y0 * w + x1] * tx;
        let b = p[y1 * w + x0] * (1.0 - tx) + p[y1 * w + x1] * tx;
        a * (1.0 - ty) + b * ty
    };
    [at(0), at(1)]
}

/// A field anchored at the pair's middle, carried back onto the first
/// frame: `f(p) = M(p + f(p) / 2)`, a few fixed-point steps from `M(p)`.
pub fn anchor(mid: &[f32], w: usize, h: usize) -> Vec<f32> {
    let n = w * h;
    let mut f = mid.to_vec();
    for _ in 0..3 {
        let mut next = vec![0f32; 2 * n];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let v = sample2(mid, w, h, x as f32 + 0.5 + f[i] * 0.5, y as f32 + 0.5 + f[n + i] * 0.5);
                next[i] = v[0];
                next[n + i] = v[1];
            }
        }
        f = next;
    }
    f
}

/// A network planned at one size.
enum Net {
    Rife(RifeGpu),
    SeaRaft(crate::searaft::Gpu),
}

/// Where each pixel of `from` is in `to`, in the network's pixels.
fn motion(net: &Net, from: &[f32], to: &[f32], w: usize, h: usize) -> Result<Vec<f32>> {
    let n = w * h;
    match net {
        Net::Rife(gpu) => {
            let out = gpu.run(from, to, 0.5)?;
            let mid: Vec<f32> = (0..2 * n).map(|i| out.flow[2 * n + i] - out.flow[i]).collect();
            Ok(anchor(&mid, w, h))
        }
        Net::SeaRaft(gpu) => {
            let mut out = gpu.run(from, to)?;
            out.truncate(2 * n);
            Ok(out)
        }
    }
}

fn write(path: &Path, w: usize, h: usize, fwd: &[f32], bwd: &[f32]) -> Result<()> {
    use exr::prelude::{Encoding, Image, Layer, LayerAttributes, SpecificChannels, Vec2, WritableImage as _};
    let n = w * h;
    let layer = Layer::new(
        (w, h),
        LayerAttributes::named("rgba"),
        Encoding::FAST_LOSSLESS,
        SpecificChannels::rgba(|p: Vec2<usize>| {
            let i = p.y() * w + p.x();
            (f16::from_f32(fwd[i]), f16::from_f32(fwd[n + i]), f16::from_f32(bwd[i]), f16::from_f32(bwd[n + i]))
        }),
    );
    Image::from_layer(layer).write().to_file(path).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// The divisor actually used: the one asked for, or a coarser one when a
/// GPU's buffers cannot hold the picture. Scale down rather than refuse
/// (manifesto rule 3).
fn analysis_divisor(device: &wgpu::Device, engine: FlowEngine, w: usize, h: usize, asked: usize) -> usize {
    let mut d = asked.max(1);
    if engine == FlowEngine::SeaRaft {
        let max = device.limits().max_storage_buffer_binding_size as u64;
        while crate::searaft::largest_buffer(w.div_ceil(d), h.div_ceil(d)) > max && d < 16 {
            d *= 2;
        }
        if d != asked.max(1) {
            log::warn!("flow: {w}x{h} analysed at 1/{d}, the most this GPU's buffers hold");
        }
    }
    d
}

fn open_net(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    engine: FlowEngine,
    weights: &Path,
    sw: usize,
    sh: usize,
) -> Result<Net> {
    Ok(match engine {
        FlowEngine::Rife => {
            let m = RifeModel::load_pkl(weights).context("loading the RIFE weights")?;
            Net::Rife(RifeGpu::new(device, queue, &m, sw, sh)?)
        }
        FlowEngine::SeaRaft => {
            let m = crate::searaft::Model::load(weights).context("loading the SEA-RAFT weights")?;
            Net::SeaRaft(crate::searaft::Gpu::new(device, queue, &m, sw, sh)?)
        }
    })
}

/// Solve A against B at every tick of `job`, both directions, reporting
/// `(done, total)` after each. A tick whose two pictures match the previous
/// tick's (held frames, stills) reuses its vectors instead of solving again.
pub fn run_pair(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    job: &PairJob,
    progress: &mut dyn FnMut(usize, usize),
    cancel: &AtomicBool,
) -> Result<()> {
    let total = job.a.len();
    anyhow::ensure!(total >= 1, "nothing to solve");
    anyhow::ensure!(job.b.len() == total, "{total} frames of A against {} of B", job.b.len());
    std::fs::create_dir_all(&job.out).with_context(|| format!("creating {}", job.out.display()))?;
    let dec = crate::decode::AnyDecoder::default();
    let first = dec.read_header(&job.a[0]).with_context(|| format!("reading {}", job.a[0].display()))?;
    let (w, h) = (first.width as usize, first.height as usize);
    let d = analysis_divisor(&device, job.engine, w, h, job.divisor);
    let (sw, sh) = (w.div_ceil(d), h.div_ceil(d));
    let get = |p: &Path| -> Result<Vec<f32>> {
        let (fw, fh, rgb) = load(&dec, p)?;
        anyhow::ensure!((fw, fh) == (w, h), "{} is {fw}x{fh}, the first frame was {w}x{h}", p.display());
        Ok(shrink(&rgb, 3, w, h, d).2)
    };
    let net = open_net(device, queue, job.engine, &job.weights, sw, sh)?;
    let mut last: Option<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> = None;
    for i in 0..total {
        if cancel.load(Ordering::Relaxed) {
            anyhow::bail!("cancelled");
        }
        let (a, b) = (get(&job.a[i])?, get(&job.b[i])?);
        let (ab, ba) = match &last {
            Some((pa, pb, ab, ba)) if *pa == a && *pb == b => (ab.clone(), ba.clone()),
            _ => (motion(&net, &a, &b, sw, sh)?, motion(&net, &b, &a, sw, sh)?),
        };
        let full = |f: &[f32]| -> Vec<f32> { grow(f, sw, sh, w, h, d).iter().map(|v| v * job.scale).collect() };
        let name = job.a[i].file_name().context("a frame without a name")?;
        write(&job.out.join(name), w, h, &full(&ab), &full(&ba))?;
        last = Some((a, b, ab, ba));
        progress(i + 1, total);
    }
    crate::bakes::mark_baked(&job.out).context("marking the vectors done")?;
    Ok(())
}

/// Solve every frame of `job`, reporting `(done, total)` after each.
pub fn run(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    job: &Job,
    progress: &mut dyn FnMut(usize, usize),
    cancel: &AtomicBool,
) -> Result<()> {
    let n_frames = job.frames.len();
    anyhow::ensure!(n_frames >= 2, "motion needs at least two frames, the range has {n_frames}");
    std::fs::create_dir_all(&job.out).with_context(|| format!("creating {}", job.out.display()))?;
    let dec = crate::decode::AnyDecoder::default();
    let first = dec
        .read_header(&job.frames[0])
        .with_context(|| format!("reading {}", job.frames[0].display()))?;
    let (w, h) = (first.width as usize, first.height as usize);
    let d = analysis_divisor(&device, job.engine, w, h, job.divisor);
    let (sw, sh) = (w.div_ceil(d), h.div_ceil(d));
    let n = sw * sh;

    // A sliding window of three decoded, shrunk frames.
    let get = |i: usize| -> Result<Vec<f32>> {
        let (fw, fh, rgb) = load(&dec, &job.frames[i])?;
        anyhow::ensure!((fw, fh) == (w, h), "frame {i} is {fw}x{fh}, the first was {w}x{h}");
        Ok(shrink(&rgb, 3, w, h, d).2)
    };
    let mut cur = get(0)?;
    let mut next = Some(get(1)?);
    let mut prev: Option<Vec<f32>> = None;
    let net = open_net(device, queue, job.engine, &job.weights, sw, sh)?;
    let mut last_fwd: Option<Vec<f32>> = None;
    let neg = |v: &[f32]| v.iter().map(|x| -x).collect::<Vec<f32>>();

    for i in 0..n_frames {
        if cancel.load(Ordering::Relaxed) {
            anyhow::bail!("cancelled");
        }
        let fwd = match &next {
            Some(nx) => Some(motion(&net, &cur, nx, sw, sh)?),
            None => None,
        };
        let bwd = match (&prev, job.backward) {
            (Some(pv), true) => Some(motion(&net, &cur, pv, sw, sh)?),
            _ => None,
        };
        // The ends, and backward off: the constant-velocity guess.
        let fwd = match (fwd, &bwd) {
            (Some(f), _) => f,
            (None, Some(b)) => neg(b),
            (None, None) => last_fwd.clone().unwrap_or_else(|| vec![0.0; 2 * n]),
        };
        let bwd = bwd.unwrap_or_else(|| neg(&fwd));
        let name = job.frames[i].file_name().context("a frame without a name")?;
        write(&job.out.join(name), w, h, &grow(&fwd, sw, sh, w, h, d), &grow(&bwd, sw, sh, w, h, d))?;
        last_fwd = Some(fwd);
        progress(i + 1, n_frames);
        // Slide the window one frame on.
        if let Some(nx) = next.take() {
            prev = Some(std::mem::replace(&mut cur, nx));
            next = if i + 2 < n_frames { Some(get(i + 2)?) } else { None };
        }
    }
    crate::bakes::mark_baked(&job.out).context("marking the vectors done")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shrink_then_grow_keeps_a_constant_motion() {
        let (w, h) = (10, 6);
        let planes: Vec<f32> = (0..2 * w * h).map(|i| if i < w * h { 3.0 } else { -1.0 }).collect();
        let (sw, sh, s) = shrink(&planes, 2, w, h, 2);
        assert_eq!((sw, sh), (5, 3));
        // A motion of 1.5 px at half size is 3 px at full size.
        let half: Vec<f32> = s.iter().map(|v| v / 2.0).collect();
        let back = grow(&half, sw, sh, w, h, 2);
        assert!(back[..w * h].iter().all(|v| (v - 3.0).abs() < 1e-5));
        assert!(back[w * h..].iter().all(|v| (v + 1.0).abs() < 1e-5));
    }

    /// A uniform field is its own anchor; a field that varies is read
    /// where the pixel will be half way.
    #[test]
    fn anchoring_reads_the_middle_field_half_a_move_on() {
        let (w, h) = (8, 4);
        let n = w * h;
        let uni: Vec<f32> = (0..2 * n).map(|i| if i < n { 2.0 } else { -1.0 }).collect();
        assert!(anchor(&uni, w, h).iter().zip(&uni).all(|(a, b)| (a - b).abs() < 1e-6));
        // x motion = 4 at x >= 4.5 (a ramp below), so a pixel at x = 2.5
        // moving 2 reads the field at 3.5.
        let ramp: Vec<f32> = (0..2 * n).map(|i| if i < n { (i % w) as f32 * 0.5 } else { 0.0 }).collect();
        let a = anchor(&ramp, w, h);
        // Fixed point of f = 0.5 (x + f/2 - 0.5) at x = 2.5: f = 4/3 (minus clamping).
        assert!((a[2] - 4.0 / 3.0).abs() < 0.02, "{}", a[2]);
    }

    #[test]
    fn srgb_encoding_clamps_what_the_network_cannot_read() {
        assert_eq!(srgb_encode(-1.0), 0.0);
        assert!((srgb_encode(4.0) - 1.0).abs() < 1e-6);
        assert_eq!(srgb_encode(f32::NAN), 0.0);
        assert!((srgb_encode(0.18) - 0.4613).abs() < 1e-3);
    }

    /// Both real networks on a real-ish texture moved by a known amount:
    /// forward vectors read the move, backward ones the move back. Needs
    /// the weights in `data/` and a GPU; each engine self-skips without.
    #[test]
    fn a_translated_texture_reads_as_its_translation() {
        let Ok(ctx) = crate::GpuContext::new() else {
            eprintln!("SKIP: no GPU");
            return;
        };
        let dir = std::env::temp_dir().join(format!("wobble-flow-test-{}", std::process::id()));
        let input = dir.join("in");
        std::fs::create_dir_all(&input).unwrap();
        let (w, h) = (192usize, 128usize);
        let (dx, dy) = (3.0f32, -2.0f32);
        // Value noise over four octaves: structure at every scale, like a
        // photograph, and no period for the match to slip by.
        let hash = |x: i32, y: i32| {
            let mut v = (x as u32).wrapping_mul(0x8da6_b343) ^ (y as u32).wrapping_mul(0xd816_3841);
            v ^= v >> 13;
            v = v.wrapping_mul(0x5bd1_e995);
            (v >> 8) as f32 / (1u32 << 24) as f32
        };
        let noise = |x: f32, y: f32| {
            let (x0, y0) = (x.floor(), y.floor());
            let (tx, ty) = (x - x0, y - y0);
            let (sx, sy) = (tx * tx * (3.0 - 2.0 * tx), ty * ty * (3.0 - 2.0 * ty));
            let (i, j) = (x0 as i32, y0 as i32);
            let a = hash(i, j) * (1.0 - sx) + hash(i + 1, j) * sx;
            let b = hash(i, j + 1) * (1.0 - sx) + hash(i + 1, j + 1) * sx;
            a * (1.0 - sy) + b * sy
        };
        let tex = |x: f32, y: f32| {
            let v = 0.45 * noise(x / 24.0, y / 24.0)
                + 0.3 * noise(x / 9.0 + 17.0, y / 9.0)
                + 0.15 * noise(x / 4.0, y / 4.0 + 5.0)
                + 0.1 * noise(x / 2.0 + 3.0, y / 2.0);
            v.clamp(0.0, 1.0).powf(2.2)
        };
        let mut frames = vec![];
        for f in 0..3 {
            let (ox, oy) = (dx * f as f32, dy * f as f32);
            let mut px = vec![f16::ZERO; w * h * 4];
            for y in 0..h {
                for x in 0..w {
                    // Frame f shows the texture moved by f × (dx, dy).
                    let (sx, sy) = (x as f32 - ox, y as f32 - oy);
                    let i = (y * w + x) * 4;
                    px[i] = f16::from_f32(tex(sx, sy));
                    px[i + 1] = f16::from_f32(tex(sx + 40.0, sy));
                    px[i + 2] = f16::from_f32(tex(sx, sy + 70.0) * 0.5);
                    px[i + 3] = f16::ONE;
                }
            }
            let p = input.join(format!("f_{f:05}.exr"));
            crate::bakes::write_rgba_f16_exr_tagged(&p, w, h, w * 4, &px, None).unwrap();
            frames.push(p);
        }
        for (engine, weights, tol) in [
            (FlowEngine::Rife, "../data/rife/flownet.pkl", 0.3f32),
            (FlowEngine::SeaRaft, "../data/searaft/C-T-M.safetensors", 0.1),
        ] {
        if !Path::new(weights).exists() {
            eprintln!("SKIP {engine:?}: no {weights}");
            continue;
        }
        let out = dir.join(format!("out-{engine:?}"));
        let job = Job { engine, weights: weights.into(), frames: frames.clone(), out, divisor: 1, backward: true };
        run(ctx.device.clone(), ctx.queue.clone(), &job, &mut |_, _| {}, &AtomicBool::new(false)).unwrap();
        assert!(crate::bakes::is_baked(&job.out));
        // The middle frame, away from the borders.
        let dec = crate::decode::AnyDecoder::default();
        let p = job.out.join("f_00001.exr");
        let info = dec.read_header(&p).unwrap();
        let stride = info.row_stride_f16();
        let mut buf = vec![f16::ZERO; stride * h];
        dec.decode_into(&p, info, stride, &mut buf).unwrap();
        let mut acc = [0f32; 4];
        let mut n = 0.0;
        for y in 32..h - 32 {
            for x in 32..w - 32 {
                for (c, a) in acc.iter_mut().enumerate() {
                    *a += buf[y * stride + x * 4 + c].to_f32();
                }
                n += 1.0;
            }
        }
        let m = acc.map(|a| a / n);
        eprintln!("{engine:?}: mean vectors {m:?}, true forward ({dx}, {dy})");
        assert!((m[0] - dx).abs() < tol && (m[1] - dy).abs() < tol, "{engine:?} forward {m:?}");
        assert!((m[2] + dx).abs() < tol && (m[3] + dy).abs() < tol, "{engine:?} backward {m:?}");

        // Two CLIPS, a Morph Dissolve's solve: frame 0 against frame 2 at
        // both ticks (twice the move, and the repeat reuses the solve),
        // written in full-res units for a half-res proxy render.
        let pair = PairJob {
            engine,
            weights: weights.into(),
            a: vec![frames[0].clone(), frames[0].clone()],
            b: vec![frames[2].clone(), frames[2].clone()],
            out: dir.join(format!("pair-{engine:?}")),
            divisor: 1,
            scale: 2.0,
        };
        let mut reports = 0;
        run_pair(ctx.device.clone(), ctx.queue.clone(), &pair, &mut |_, _| reports += 1, &AtomicBool::new(false)).unwrap();
        assert_eq!(reports, 2);
        assert!(crate::bakes::is_baked(&pair.out));
        let p = pair.out.join("f_00000.exr");
        let info = dec.read_header(&p).unwrap();
        dec.decode_into(&p, info, stride, &mut buf).unwrap();
        let mut acc = [0f32; 4];
        for y in 32..h - 32 {
            for x in 32..w - 32 {
                for (c, a) in acc.iter_mut().enumerate() {
                    *a += buf[y * stride + x * 4 + c].to_f32();
                }
            }
        }
        let m = acc.map(|a| a / n);
        let want = [4.0 * dx, 4.0 * dy];
        eprintln!("{engine:?}: A->B / B->A means {m:?}, true A->B {want:?}");
        let tol = tol * 4.0;
        assert!((m[0] - want[0]).abs() < tol && (m[1] - want[1]).abs() < tol, "{engine:?} A->B {m:?}");
        assert!((m[2] + want[0]).abs() < tol && (m[3] + want[1]).abs() < tol, "{engine:?} B->A {m:?}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod probe {
    use super::*;
    /// FLOW_PROBE_DIR=<dir of f_NNNNN.exr>: prints the centre vector of
    /// frame 20 -> 21 under each way of asking RIFE, and SEA-RAFT's.
    #[test]
    #[ignore]
    fn probe_centre() {
        let Ok(dir) = std::env::var("FLOW_PROBE_DIR") else { return };
        let dir = PathBuf::from(dir);
        let ctx = crate::GpuContext::new().unwrap();
        let model = RifeModel::load_pkl(Path::new("../data/rife/flownet.pkl")).unwrap();
        let dec = crate::decode::AnyDecoder::default();
        let (w, h, a) = load(&dec, &dir.join("f_00020.exr")).unwrap();
        let (_, _, b) = load(&dec, &dir.join("f_00021.exr")).unwrap();
        for d in [1usize, 2] {
            let (sw, sh, a2) = shrink(&a, 3, w, h, d);
            let (_, _, b2) = shrink(&b, 3, w, h, d);
            let gpu = RifeGpu::new(ctx.device.clone(), ctx.queue.clone(), &model, sw, sh).unwrap();
            let n = sw * sh;
            let mean = |f: &[f32], c: usize| {
                let (cx, cy) = (sw / 2, sh / 2);
                let mut s = 0.0;
                for y in cy - 2..cy + 2 { for x in cx - 2..cx + 2 { s += f[c * n + y * sw + x]; } }
                s / 16.0 * d as f32
            };
            for t in [0.0f32, 0.5] {
                let o = gpu.run(&a2, &b2, t).unwrap();
                eprintln!("d={d} t={t}: F_t0=({:.2},{:.2}) F_t1=({:.2},{:.2}) diff=({:.2},{:.2})",
                    mean(&o.flow,0), mean(&o.flow,1), mean(&o.flow,2), mean(&o.flow,3),
                    mean(&o.flow,2)-mean(&o.flow,0), mean(&o.flow,3)-mean(&o.flow,1));
            }
            let net = Net::Rife(gpu);
            let m = motion(&net, &a2, &b2, sw, sh).unwrap();
            eprintln!("d={d} anchored: ({:.2},{:.2})", mean(&m, 0), mean(&m, 1));
            let sm = crate::searaft::Model::load(Path::new("../data/searaft/C-T-M.safetensors")).unwrap();
            let net = Net::SeaRaft(crate::searaft::Gpu::new(ctx.device.clone(), ctx.queue.clone(), &sm, sw, sh).unwrap());
            let m = motion(&net, &a2, &b2, sw, sh).unwrap();
            eprintln!("d={d} SEA-RAFT: ({:.2},{:.2})", mean(&m, 0), mean(&m, 1));
        }
    }
}
