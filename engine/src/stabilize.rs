//! The Stabilize node's analysis: rendered frames in, a camera path out.
//!
//! Per neighbouring pair, SEA-RAFT's flow at `1 / divisor` of the size
//! (`crate::searaft`, the Flow node's network and weights), sampled on a
//! grid away from the borders; points under the matte (if one was
//! rendered) are left out. ONE global motion is fitted to the grid
//! (`fit`): least squares in normalised coordinates, started from the
//! median displacement and re-weighted with Tukey's biweight on the
//! residual, so a subject moving against the camera is outvoted rather
//! than averaged in. The pair motions chain into the path the node
//! smooths (`nodes::stabilize::StabPath`): `motion[t]` carries frame 0's
//! pixels onto frame t's.
//!
//! The result is small (nine numbers a frame): it is written as text to
//! `<out>/motion.txt` with the bake's done-marker, so the same recipe never
//! runs twice, and the app copies it onto the node.
//!
//! Runs on the caller's thread and blocks on the GPU per pair: the app puts
//! it on a worker. Cancellable between pairs.

use crate::decode::ExrDecoder as _;
use crate::nodes::stabilize::StabModel;
use crate::planar::{h_apply, h_mul, h_translation, H};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// What to analyse.
pub struct Job {
    /// SEA-RAFT's weights (`models::ready_path("searaft-m")`).
    pub weights: PathBuf,
    /// The frames, in time order (linear Rec.709 EXRs).
    pub frames: Vec<PathBuf>,
    /// Per frame, a matte whose red > 0.5 keeps a pixel out of the fit.
    pub mattes: Option<Vec<PathBuf>>,
    /// Where `motion.txt` and the done-marker go.
    pub out: PathBuf,
    pub divisor: usize,
    pub model: StabModel,
}

/// Grid points across the shorter side of the analysed picture.
const GRID: usize = 48;
/// The border left out, as a fraction of each side: flow there sees
/// pixels coming in from nowhere.
const MARGIN: f64 = 0.04;

/// Fit `model` to correspondences `p -> q` (pixels), robustly. None when
/// there are too few or they are degenerate.
pub fn fit(pts: &[([f64; 2], [f64; 2])], model: StabModel) -> Option<H> {
    let need = match model {
        StabModel::Translate => 1,
        StabModel::Similarity => 2,
        StabModel::Affine => 3,
        StabModel::Perspective => 4,
    };
    if pts.len() < need * 3 {
        return None;
    }
    // Normalise: centroid at the origin, mean distance sqrt 2.
    let n = pts.len() as f64;
    let c = [pts.iter().map(|p| p.0[0]).sum::<f64>() / n, pts.iter().map(|p| p.0[1]).sum::<f64>() / n];
    let spread = pts.iter().map(|p| (p.0[0] - c[0]).hypot(p.0[1] - c[1])).sum::<f64>() / n;
    let k = std::f64::consts::SQRT_2 / spread.max(1e-9);
    let norm = |v: [f64; 2]| [(v[0] - c[0]) * k, (v[1] - c[1]) * k];
    let np: Vec<([f64; 2], [f64; 2])> = pts.iter().map(|(p, q)| (norm(*p), norm(*q))).collect();
    // Start from the median displacement: half the points can be wrong.
    let med = |mut v: Vec<f64>| -> f64 {
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };
    let t0 = [
        med(np.iter().map(|(p, q)| q[0] - p[0]).collect()),
        med(np.iter().map(|(p, q)| q[1] - p[1]).collect()),
    ];
    let mut h = h_translation(t0[0], t0[1]);
    let mut w = vec![1.0f64; np.len()];
    for _ in 0..20 {
        // Tukey weights on the residual, the cut at 4.685 robust sigmas
        // (never under a twentieth of a pixel, in normalised units).
        let r: Vec<f64> = np
            .iter()
            .map(|(p, q)| {
                let (x, y) = h_apply(&h, p[0], p[1]);
                (x - q[0]).hypot(y - q[1])
            })
            .collect();
        let sigma = 1.4826 * med(r.clone());
        let cut = (4.685 * sigma).max(0.05 * k);
        for (wi, ri) in w.iter_mut().zip(&r) {
            let u = ri / cut;
            *wi = if u < 1.0 { (1.0 - u * u).powi(2) } else { 0.0 };
        }
        let next = solve(&np, &w, model)?;
        let moved = np.iter().take(64).map(|(p, _)| {
            let (a, b) = h_apply(&h, p[0], p[1]);
            let (c2, d) = h_apply(&next, p[0], p[1]);
            (a - c2).hypot(b - d)
        });
        let delta = moved.fold(0.0f64, f64::max);
        h = next;
        if delta < 1e-4 * k {
            break;
        }
    }
    // Back to pixels: T⁻¹ · h · T.
    let t = h_mul(&[k, 0.0, 0.0, 0.0, k, 0.0, 0.0, 0.0, 1.0], &h_translation(-c[0], -c[1]));
    let ti = h_mul(&h_translation(c[0], c[1]), &[1.0 / k, 0.0, 0.0, 0.0, 1.0 / k, 0.0, 0.0, 0.0, 1.0]);
    Some(crate::planar::h_norm(&h_mul(&ti, &h_mul(&h, &t))))
}

/// Weighted least squares for one model.
fn solve(pts: &[([f64; 2], [f64; 2])], w: &[f64], model: StabModel) -> Option<H> {
    let n = match model {
        StabModel::Translate => 2,
        StabModel::Similarity => 4,
        StabModel::Affine => 6,
        StabModel::Perspective => 8,
    };
    let mut ata = vec![0.0f64; n * n];
    let mut atb = vec![0.0f64; n];
    let mut add = |row: &[f64], rhs: f64, wt: f64| {
        for i in 0..n {
            atb[i] += wt * row[i] * rhs;
            for j in 0..n {
                ata[i * n + j] += wt * row[i] * row[j];
            }
        }
    };
    for ((p, q), &wt) in pts.iter().zip(w) {
        if wt <= 0.0 {
            continue;
        }
        let (x, y) = (p[0], p[1]);
        match model {
            StabModel::Translate => {
                add(&[1.0, 0.0], q[0] - x, wt);
                add(&[0.0, 1.0], q[1] - y, wt);
            }
            StabModel::Similarity => {
                add(&[x, -y, 1.0, 0.0], q[0], wt);
                add(&[y, x, 0.0, 1.0], q[1], wt);
            }
            StabModel::Affine => {
                add(&[x, y, 1.0, 0.0, 0.0, 0.0], q[0], wt);
                add(&[0.0, 0.0, 0.0, x, y, 1.0], q[1], wt);
            }
            StabModel::Perspective => {
                add(&[x, y, 1.0, 0.0, 0.0, 0.0, -q[0] * x, -q[0] * y], q[0], wt);
                add(&[0.0, 0.0, 0.0, x, y, 1.0, -q[1] * x, -q[1] * y], q[1], wt);
            }
        }
    }
    let s = crate::planar::solve_linear(&mut ata, &mut atb, n)?;
    Some(match model {
        StabModel::Translate => h_translation(s[0], s[1]),
        StabModel::Similarity => [s[0], -s[1], s[2], s[1], s[0], s[3], 0.0, 0.0, 1.0],
        StabModel::Affine => [s[0], s[1], s[2], s[3], s[4], s[5], 0.0, 0.0, 1.0],
        StabModel::Perspective => [s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], 1.0],
    })
}

/// The grid of correspondences one flow field gives, in FULL-size pixels.
/// `flow` is `[2][sh][sw]` in analysed pixels; `keep(x, y)` says whether a
/// full-size pixel may vote.
pub fn correspondences(
    flow: &[f32],
    sw: usize,
    sh: usize,
    d: usize,
    keep: &dyn Fn(f64, f64) -> bool,
) -> Vec<([f64; 2], [f64; 2])> {
    let n = sw * sh;
    let step = (sw.min(sh) / GRID).max(1);
    let (mx, my) = ((sw as f64 * MARGIN) as usize, (sh as f64 * MARGIN) as usize);
    let mut out = Vec::new();
    let k = d as f64;
    let mut y = my;
    while y < sh.saturating_sub(my) {
        let mut x = mx;
        while x < sw.saturating_sub(mx) {
            let i = y * sw + x;
            let (fx, fy) = (flow[i] as f64, flow[n + i] as f64);
            let p = [(x as f64 + 0.5) * k, (y as f64 + 0.5) * k];
            if fx.is_finite() && fy.is_finite() && keep(p[0], p[1]) {
                out.push((p, [p[0] + fx * k, p[1] + fy * k]));
            }
            x += step;
        }
        y += step;
    }
    out
}

/// A matte frame's red channel at full size.
pub(crate) fn load_matte(dec: &crate::decode::AnyDecoder, path: &Path) -> Result<(usize, usize, Vec<f32>)> {
    let info = dec.read_header(path).with_context(|| format!("reading {}", path.display()))?;
    let (w, h) = (info.width as usize, info.height as usize);
    let stride = info.row_stride_f16();
    let mut buf = vec![half::f16::ZERO; stride * h];
    dec.decode_into(path, info, stride, &mut buf)?;
    let mut m = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            m[y * w + x] = buf[y * stride + x * 4].to_f32();
        }
    }
    Ok((w, h, m))
}

/// Analyse every pair of `job`, reporting `(done, total)` after each.
/// Returns the path (`motion[0]` the identity) and the frames' size.
pub fn run(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    job: &Job,
    progress: &mut dyn FnMut(usize, usize),
    cancel: &AtomicBool,
) -> Result<(Vec<H>, [u32; 2])> {
    let n_frames = job.frames.len();
    anyhow::ensure!(n_frames >= 2, "stabilising needs at least two frames, the range has {n_frames}");
    std::fs::create_dir_all(&job.out).with_context(|| format!("creating {}", job.out.display()))?;
    let dec = crate::decode::AnyDecoder::default();
    let first = dec
        .read_header(&job.frames[0])
        .with_context(|| format!("reading {}", job.frames[0].display()))?;
    let (w, h) = (first.width as usize, first.height as usize);
    // Scale down rather than refuse (manifesto rule 3), and never below
    // what SEA-RAFT can see.
    let mut d = job.divisor.max(1);
    let max = device.limits().max_storage_buffer_binding_size as u64;
    while crate::searaft::largest_buffer(w.div_ceil(d), h.div_ceil(d)) > max && d < 16 {
        d *= 2;
    }
    while d > 1 && (w.div_ceil(d) < 128 || h.div_ceil(d) < 128) {
        d /= 2;
    }
    let (sw, sh) = (w.div_ceil(d), h.div_ceil(d));
    let get = |i: usize| -> Result<Vec<f32>> {
        let (fw, fh, rgb) = crate::flowsolve::load(&dec, &job.frames[i])?;
        anyhow::ensure!((fw, fh) == (w, h), "frame {i} is {fw}x{fh}, the first was {w}x{h}");
        Ok(crate::flowsolve::shrink(&rgb, 3, w, h, d).2)
    };
    let m = crate::searaft::Model::load(&job.weights).context("loading the SEA-RAFT weights")?;
    let net = crate::searaft::Gpu::new(device, queue, &m, sw, sh)?;
    let mut path: Vec<H> = vec![[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]];
    let mut cur = get(0)?;
    let mut last = path[0];
    for i in 0..n_frames - 1 {
        if cancel.load(Ordering::Relaxed) {
            anyhow::bail!("cancelled");
        }
        let next = get(i + 1)?;
        let mut flow = net.run(&cur, &next)?;
        flow.truncate(2 * sw * sh);
        let matte = match &job.mattes {
            Some(ms) => Some(load_matte(&dec, &ms[i])?),
            None => None,
        };
        let keep = |x: f64, y: f64| -> bool {
            match &matte {
                Some((mw, mh, m)) => {
                    let (ix, iy) = ((x as usize).min(mw - 1), (y as usize).min(mh - 1));
                    m[iy * mw + ix] <= 0.5
                }
                None => true,
            }
        };
        let pts = correspondences(&flow, sw, sh, d, &keep);
        // A pair the fit cannot use (all matte, a cut to black) holds the
        // last motion rather than inventing one.
        let pair = fit(&pts, job.model).unwrap_or(last);
        let pair = if pair.iter().all(|v| v.is_finite()) { pair } else { last };
        last = pair;
        let acc = crate::planar::h_norm(&h_mul(&pair, path.last().unwrap()));
        path.push(acc);
        cur = next;
        progress(i + 1, n_frames - 1);
    }
    write_motion(&job.out, &path, [w as u32, h as u32])?;
    crate::bakes::mark_baked(&job.out).context("marking the analysis done")?;
    Ok((path, [w as u32, h as u32]))
}

/// `motion.txt`: `size w h`, then nine numbers a frame.
pub fn write_motion(dir: &Path, path: &[H], size: [u32; 2]) -> Result<()> {
    let mut s = format!("size {} {}\n", size[0], size[1]);
    for m in path {
        let row: Vec<String> = m.iter().map(|v| format!("{v:e}")).collect();
        s.push_str(&row.join(" "));
        s.push('\n');
    }
    std::fs::write(dir.join("motion.txt"), s).with_context(|| format!("writing {}", dir.display()))
}

/// Read what `write_motion` wrote.
pub fn read_motion(dir: &Path) -> Result<(Vec<H>, [u32; 2])> {
    let text = std::fs::read_to_string(dir.join("motion.txt")).with_context(|| format!("reading {}", dir.display()))?;
    let mut lines = text.lines();
    let head: Vec<u32> =
        lines.next().unwrap_or("").split_whitespace().skip(1).filter_map(|v| v.parse().ok()).collect();
    anyhow::ensure!(head.len() == 2, "motion.txt has no size");
    let mut path = Vec::new();
    for l in lines {
        let v: Vec<f64> = l.split_whitespace().filter_map(|v| v.parse().ok()).collect();
        anyhow::ensure!(v.len() == 9, "a motion row of {} numbers", v.len());
        path.push(std::array::from_fn(|k| v[k]));
    }
    Ok((path, [head[0], head[1]]))
}

/// A synthetic shaky shot for tests and the suite: `planar::synth`'s
/// texture, framed 1.25x so the shake never shows its edge, under a
/// jittering similarity. Writes `frame.NNNN.png` and returns, per frame,
/// the TRUE camera path (frame 0's pixels onto frame t's).
pub fn write_shaky(dir: &Path, w: usize, h: usize, frames: usize) -> Result<Vec<H>> {
    use crate::planar::synth::{render, texture, view, RenderOpts};
    std::fs::create_dir_all(dir)?;
    let tex = texture(w, h, 11);
    let pose = |t: usize| {
        let tf = t as f64;
        view(
            w,
            h,
            1.2 * (tf * 1.3).sin() + 0.6 * (tf * 0.37).cos(),
            1.25 * (1.0 + 0.008 * (tf * 0.9).cos()),
            1.5 * tf + 6.0 * (tf * 1.7).sin(),
            4.0 * (tf * 2.3).cos() - 4.0 + 2.0 * (tf * 0.6).sin(),
            0.0,
            0.0,
        )
    };
    let inv0 = crate::planar::h_inv(&pose(0)).context("pose 0")?;
    let mut truth = Vec::with_capacity(frames);
    for t in 0..frames {
        let v = pose(t);
        truth.push(h_mul(&v, &inv0));
        let f = render(&tex, &v, w, h, &RenderOpts { noise: 0.01, seed: t as u32 + 1, ..Default::default() });
        let mut data = Vec::with_capacity(w * h * 2);
        for &px in &f.px {
            data.extend_from_slice(&((px.clamp(0.0, 1.0) * 65535.0).round() as u16).to_be_bytes());
        }
        let file = std::io::BufWriter::new(std::fs::File::create(dir.join(format!("frame.{t:04}.png")))?);
        let mut enc = png::Encoder::new(file, w as u32, h as u32);
        enc.set_color(png::ColorType::Grayscale);
        enc.set_depth(png::BitDepth::Sixteen);
        enc.write_header()?.write_image_data(&data)?;
    }
    Ok(truth)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid_of(m: &H, out: Option<(&H, f64)>) -> Vec<([f64; 2], [f64; 2])> {
        let mut pts = Vec::new();
        for j in 0..30 {
            for i in 0..50 {
                let p = [20.0 + i as f64 * 12.0, 20.0 + j as f64 * 11.0];
                // `out`: a subject moving its own way over the left third.
                let mm = match out {
                    Some((o, x)) if p[0] < x => o,
                    _ => m,
                };
                let (x, y) = h_apply(mm, p[0], p[1]);
                pts.push((p, [x, y]));
            }
        }
        pts
    }

    fn err(a: &H, b: &H) -> f64 {
        let mut e = 0.0f64;
        for p in [[0.0, 0.0], [640.0, 0.0], [640.0, 360.0], [0.0, 360.0]] {
            let (x, y) = h_apply(a, p[0], p[1]);
            let (u, v) = h_apply(b, p[0], p[1]);
            e = e.max((x - u).hypot(y - v));
        }
        e
    }

    #[test]
    fn every_model_recovers_its_own_motion() {
        let sim = crate::nodes::stabilize::sim_h(&[4.0, -3.0, 0.03, 0.02f64.ln_1p()], [320.0, 180.0]);
        let aff = h_mul(&sim, &[1.0, 0.02, 0.0, -0.01, 1.01, 0.0, 0.0, 0.0, 1.0]);
        let per = h_mul(&aff, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2e-5, -1e-5, 1.0]);
        for (model, m) in [
            (StabModel::Translate, h_translation(3.5, -2.25)),
            (StabModel::Similarity, sim),
            (StabModel::Affine, aff),
            (StabModel::Perspective, per),
        ] {
            let got = fit(&grid_of(&m, None), model).unwrap();
            assert!(err(&got, &m) < 1e-6, "{model:?} off by {}", err(&got, &m));
        }
    }

    /// A third of the picture moving its own way is outvoted, not averaged.
    #[test]
    fn a_moving_subject_is_outvoted() {
        let cam = crate::nodes::stabilize::sim_h(&[2.0, 1.0, 0.01, 0.0], [320.0, 180.0]);
        let subject = h_translation(-25.0, 12.0);
        let got = fit(&grid_of(&cam, Some((&subject, 200.0))), StabModel::Similarity).unwrap();
        assert!(err(&got, &cam) < 1e-3, "off by {}", err(&got, &cam));
    }

    #[test]
    fn the_matte_keeps_points_out_and_the_grid_is_in_full_pixels() {
        let (sw, sh, d) = (200usize, 120usize, 2usize);
        let mut flow = vec![0f32; 2 * sw * sh];
        flow[..sw * sh].fill(1.5);
        let all = correspondences(&flow, sw, sh, d, &|_, _| true);
        assert!(all.len() > 500);
        assert!(all.iter().all(|(p, q)| (q[0] - p[0] - 3.0).abs() < 1e-9 && q[1] == p[1]));
        let left = correspondences(&flow, sw, sh, d, &|x, _| x > 200.0);
        assert!(left.iter().all(|(p, _)| p[0] > 200.0) && left.len() < all.len());
    }

    #[test]
    fn the_motion_file_round_trips() {
        let dir = std::env::temp_dir().join(format!("wobble-stab-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = vec![[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], h_translation(1.0 / 3.0, -7.25)];
        write_motion(&dir, &path, [960, 540]).unwrap();
        let (back, size) = read_motion(&dir).unwrap();
        assert_eq!(size, [960, 540]);
        assert_eq!(back, path);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
