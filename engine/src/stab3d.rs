//! 3D stabilisation, the cheap version (a PROTOTYPE, driven by
//! `bin/stab3d`; no node yet): every frame becomes a cloud of Gaussian
//! splats, one per pixel, lifted off a monocular depth map into the
//! camera solve's world; the solved path is smoothed; and each frame is
//! drawn again through its smoothed camera with the Splat node's
//! rasteriser (`splatgpu`).
//!
//! What the reprojection cannot see (the picture's border, and whatever a
//! nearer thing hid) comes out as alpha 0. Neighbouring frames' clouds,
//! drawn UNDER the frame's own, fill what they saw; the rest is a hole,
//! and the share of holes is what the harness reports.
//!
//! Conventions are the solver's (`camtrack`): CV camera (x right, y down,
//! z forward), `x_c = R x + t`, world = the first camera's frame. The
//! rasteriser looks down −Z with y up, so its model-view is `FLIP · [R|t]`.

use crate::camtrack::{add, m_mul, m_t, m_v, scl, so3_exp, so3_log, sub, Pose, Solve, Track, M3, V3};
use crate::decode::ExrDecoder;
use crate::splat::{Cloud, Native, SPLAT_WORDS};
use crate::splatgpu::{SplatLayer, SplatRaster, UNIFORM_BYTES};
use anyhow::{Context, Result};
use half::f16;
use std::path::Path;

/// One frame: the picture (linear, as decoded) and its depth along the
/// optical axis, in the depth map's own units.
pub struct Frame {
    pub w: usize,
    pub h: usize,
    /// `[h][w][4]`.
    pub rgba: Vec<f32>,
    /// `[h][w]`.
    pub z: Vec<f32>,
}

fn decode(dec: &crate::decode::AnyDecoder, path: &Path) -> Result<(usize, usize, usize, Vec<f16>)> {
    let info = dec.read_header(path).with_context(|| format!("reading {}", path.display()))?;
    let (w, h) = (info.width as usize, info.height as usize);
    let stride = info.row_stride_f16();
    let mut buf = vec![f16::ZERO; stride * h];
    dec.decode_into(path, info, stride, &mut buf).with_context(|| format!("decoding {}", path.display()))?;
    Ok((w, h, stride, buf))
}

/// Read a plate and the depth map rendered for it (a Depth node's `Z`).
pub fn read_frame(dec: &crate::decode::AnyDecoder, plate: &Path, depth: &Path) -> Result<Frame> {
    let (w, h, stride, buf) = decode(dec, plate)?;
    let mut rgba = vec![0f32; w * h * 4];
    for y in 0..h {
        for (o, v) in rgba[y * w * 4..][..w * 4].iter_mut().zip(&buf[y * stride..][..w * 4]) {
            let f = v.to_f32();
            *o = if f.is_finite() { f.max(0.0) } else { 0.0 };
        }
    }
    let (dw, dh, dstride, dbuf) = decode(dec, depth)?;
    anyhow::ensure!((dw, dh) == (w, h), "the depth is {dw}x{dh}, the plate {w}x{h}");
    let mut z = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            z[y * w + x] = dbuf[y * dstride + x * 4].to_f32();
        }
    }
    Ok(Frame { w, h, rgba, z })
}

fn median(v: &mut [f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let m = v.len() / 2;
    v.select_nth_unstable_by(m, |a, b| a.total_cmp(b));
    Some(v[m])
}

/// What one unit of the depth map is in the solve's units at `frame`: the
/// median, over the solved points the frame sees, of the solve's depth
/// over the map's. Also the median relative misfit once scaled (how well
/// the two agree about the SHAPE of the scene). None with too few points.
pub fn depth_scale(solve: &Solve, tracks: &[Track], frame: usize, f: &Frame) -> Option<(f64, f64)> {
    let pose = solve.poses.get(frame)?;
    let mut pairs = Vec::new();
    for (p, ti) in solve.points.iter().zip(&solve.point_track) {
        let Some(px) = tracks.get(*ti).and_then(|t| t.at(frame)) else { continue };
        let zc = pose.apply(*p)[2];
        let (x, y) = (px[0] as usize, px[1] as usize);
        if zc <= 1e-9 || x >= f.w || y >= f.h {
            continue;
        }
        let zm = f.z[y * f.w + x] as f64;
        if zm > 1e-6 && zm < 0.5 * crate::nodes::depth::FAR as f64 {
            pairs.push((zc, zm));
        }
    }
    if pairs.len() < 8 {
        return None;
    }
    let s = median(&mut pairs.iter().map(|(a, b)| a / b).collect::<Vec<_>>())?;
    let misfit = median(&mut pairs.iter().map(|(a, b)| (s * b / a - 1.0).abs()).collect::<Vec<_>>())?;
    Some((s, misfit))
}

/// A Gaussian low-pass of a camera path, `sigma` frames wide: centres
/// averaged, rotations by the weighted mean on the group. The window is
/// renormalised at the ends, so the first and last frames lean inward.
pub fn smooth_path(poses: &[Pose], sigma: f64) -> Vec<Pose> {
    if sigma <= 1e-6 {
        return poses.to_vec();
    }
    let n = poses.len();
    let reach = (3.0 * sigma).ceil() as isize;
    let centres: Vec<V3> = poses.iter().map(|p| p.centre()).collect();
    (0..n as isize)
        .map(|i| {
            let win: Vec<(usize, f64)> = (i - reach..=i + reach)
                .filter(|j| *j >= 0 && *j < n as isize)
                .map(|j| (j as usize, (-0.5 * ((j - i) as f64 / sigma).powi(2)).exp()))
                .collect();
            let total: f64 = win.iter().map(|w| w.1).sum();
            let mut c = [0.0; 3];
            for (j, w) in &win {
                c = add(c, scl(centres[*j], w / total));
            }
            let mut r: M3 = poses[i as usize].r;
            for _ in 0..4 {
                let mut d = [0.0; 3];
                for (j, w) in &win {
                    d = add(d, scl(so3_log(&m_mul(&poses[*j].r, &m_t(&r))), w / total));
                }
                r = m_mul(&so3_exp(d), &r);
            }
            Pose { r, t: scl(m_v(&r, c), -1.0) }
        })
        .collect()
}

/// How far a smoothed path strays from the solved one: (mean, max) of the
/// centres' distance, and of the rotation between them in degrees.
pub fn path_delta(a: &[Pose], b: &[Pose]) -> ([f64; 2], [f64; 2]) {
    let (mut c, mut r) = ([0.0f64; 2], [0.0f64; 2]);
    for (p, q) in a.iter().zip(b) {
        let d = crate::camtrack::norm(sub(p.centre(), q.centre()));
        let ang = crate::camtrack::norm(so3_log(&m_mul(&p.r, &m_t(&q.r)))).to_degrees();
        c = [c[0] + d / a.len() as f64, c[1].max(d)];
        r = [r[0] + ang / a.len() as f64, r[1].max(ang)];
    }
    (c, r)
}

/// How a frame becomes splats.
#[derive(Clone, Copy, Debug)]
pub struct CloudOpts {
    /// A splat's sigma, in source pixels.
    pub sigma_px: f32,
    /// Leave out a pixel whose depth differs from a 4-neighbour's by more
    /// than this share: the mixed pixels along a silhouette, which hang in
    /// mid air between the two surfaces. 0 keeps everything.
    pub edge: f32,
    /// Every `step`-th pixel each way (the sigma grows to match).
    pub step: usize,
}

fn h2(a: f32, b: f32) -> u32 {
    f16::from_f32(a).to_bits() as u32 | ((f16::from_f32(b).to_bits() as u32) << 16)
}

/// The frame as a cloud in the solve's world: one isotropic splat a pixel,
/// at `zscale ×` its depth along the pixel's ray through `pose`.
pub fn cloud(fr: &Frame, zscale: f64, pose: &Pose, focal_px: f64, o: &CloudOpts) -> Cloud {
    let (w, h) = (fr.w, fr.h);
    let step = o.step.max(1);
    let (cx, cy) = (w as f64 * 0.5, h as f64 * 0.5);
    let rt = m_t(&pose.r);
    let origin = pose.centre();
    let mut splats = Vec::with_capacity(w.div_ceil(step) * h.div_ceil(step) * SPLAT_WORDS);
    let ident = (h2(1.0, 0.0), h2(0.0, 0.0));
    let mut count = 0;
    for y in (step / 2..h).step_by(step) {
        for x in (step / 2..w).step_by(step) {
            let z = fr.z[y * w + x];
            if !(z > 1e-6) {
                continue;
            }
            if o.edge > 0.0 {
                let (mut lo, mut hi) = (z, z);
                for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let (xx, yy) = (x as i32 + dx * step as i32, y as i32 + dy * step as i32);
                    if xx >= 0 && yy >= 0 && (xx as usize) < w && (yy as usize) < h {
                        let v = fr.z[yy as usize * w + xx as usize];
                        lo = lo.min(v);
                        hi = hi.max(v);
                    }
                }
                if hi > lo * (1.0 + o.edge) {
                    continue;
                }
            }
            let zc = z as f64 * zscale;
            let pc = [(x as f64 + 0.5 - cx) / focal_px * zc, (y as f64 + 0.5 - cy) / focal_px * zc, zc];
            let p = add(m_v(&rt, pc), origin);
            let ls = ((o.sigma_px as f64 * step as f64 * zc / focal_px).max(1e-12)).ln() as f32;
            let c = &fr.rgba[(y * w + x) * 4..][..3];
            let cl = |v: f32| v.min(60000.0);
            splats.extend_from_slice(&[
                (p[0] as f32).to_bits(),
                (p[1] as f32).to_bits(),
                (p[2] as f32).to_bits(),
                ident.0,
                ident.1,
                h2(ls, ls),
                h2(ls, 1.0),
                h2(cl(c[0]), cl(c[1])),
                h2(cl(c[2]), 0.0),
            ]);
            count += 1;
        }
    }
    Cloud {
        count,
        splats,
        sh_degree: 0,
        sh_words: 0,
        sh: Vec::new(),
        native: Native::Rdf,
        center: [0.0; 3],
        radius: 1.0,
        antialiased: false,
        notes: Vec::new(),
        file_count: count,
    }
}

/// One round, opaque splat of a flat colour at `p`, `sigma` world units
/// wide: what a showcase view draws camera paths and frusta out of.
pub fn marker(c: &mut Cloud, p: V3, sigma: f64, rgb: [f32; 3]) {
    let ls = sigma.max(1e-12).ln() as f32;
    c.splats.extend_from_slice(&[
        (p[0] as f32).to_bits(),
        (p[1] as f32).to_bits(),
        (p[2] as f32).to_bits(),
        h2(1.0, 0.0),
        h2(0.0, 0.0),
        h2(ls, ls),
        h2(ls, 1.0),
        h2(rgb[0], rgb[1]),
        h2(rgb[2], 0.0),
    ]);
    c.count += 1;
    c.file_count += 1;
}

/// A camera at `eye` looking at `target`, with `down` (any vector not
/// along the view) as the picture's down. CV convention, like the solve.
pub fn look_at(eye: V3, target: V3, down: V3) -> Pose {
    let norm = |v: V3| scl(v, 1.0 / (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-12));
    let cross = |a: V3, b: V3| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let z = norm(sub(target, eye));
    let x = norm(cross(down, z));
    let y = cross(z, x);
    let r = [x, y, z];
    Pose { r, t: scl(m_v(&r, eye), -1.0) }
}

/// Several clouds as one.
pub fn merge(clouds: Vec<Cloud>) -> Cloud {
    let mut it = clouds.into_iter();
    let mut a = it.next().expect("a cloud");
    for b in it {
        a.count += b.count;
        a.file_count += b.file_count;
        a.splats.extend(b.splats);
    }
    a
}

/// The rasteriser's uniform (`struct U` in `splatgpu`) for a CV pose.
pub fn uniform(pose: &Pose, focal_px: f64, w: u32, h: u32, near: f64, far: f64, count: usize) -> [u8; UNIFORM_BYTES] {
    let (wf, hf, f) = (w as f32, h as f32, focal_px as f32);
    let mut v = [0f32; 28];
    for r in 0..3 {
        let s = if r == 0 { 1.0 } else { -1.0 };
        for c in 0..3 {
            v[r * 4 + c] = (s * pose.r[r][c]) as f32;
        }
        v[r * 4 + 3] = (s * pose.t[r]) as f32;
    }
    let c = pose.centre();
    v[12..15].copy_from_slice(&[c[0] as f32, c[1] as f32, c[2] as f32]);
    v[16..20].copy_from_slice(&[f, 0.5 * wf, 0.5 * hf, near as f32]);
    v[20..24].copy_from_slice(&[wf, hf, far as f32, (far / near).ln() as f32]);
    v[24..28].copy_from_slice(&[1.0, 1.0, 1.3 * 0.5 * wf / f, 1.3 * 0.5 * hf / f]);
    let tail = [count as u32, 0, 0, 0];
    let mut out = [0u8; UNIFORM_BYTES];
    out[..112].copy_from_slice(bytemuck::cast_slice(&v));
    out[112..].copy_from_slice(bytemuck::cast_slice(&tail));
    out
}

/// The Splat node's rasteriser on its own: a cloud in, premultiplied
/// pixels back.
pub struct Renderer {
    device: std::sync::Arc<wgpu::Device>,
    queue: std::sync::Arc<wgpu::Queue>,
    raster: SplatRaster,
    layer: SplatLayer,
    gen: u64,
}

impl Renderer {
    pub fn new(device: std::sync::Arc<wgpu::Device>, queue: std::sync::Arc<wgpu::Queue>, w: u32, h: u32) -> Self {
        let raster = SplatRaster::new(&device);
        let layer = raster.make_layer(&device, w, h);
        Self { device, queue, raster, layer, gen: 0 }
    }

    /// Draw `cloud` through `pose`; `[h][w][4]`, premultiplied.
    pub fn render(&mut self, cloud: &Cloud, pose: &Pose, focal_px: f64, near: f64, far: f64) -> Vec<f32> {
        let (w, h) = (self.layer.width, self.layer.height);
        self.gen += 1;
        let uni = uniform(pose, focal_px, w, h, near, far, cloud.count);
        self.raster.set(&self.device, &self.queue, &mut self.layer, Some((cloud, self.gen)), &uni, false);
        let mut enc = self.device.create_command_encoder(&Default::default());
        self.raster.encode(&mut enc, &mut self.layer);
        let padded = (w * 8).div_ceil(256) * 256;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stab3d-readback"),
            size: (padded * h) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.layer.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(h) },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let data = slice.get_mapped_range().expect("a mapped buffer");
        let mut out = vec![0f32; (w * h * 4) as usize];
        for y in 0..h as usize {
            let row: &[f16] = bytemuck::cast_slice(&data[y * padded as usize..][..(w * 8) as usize]);
            for (o, v) in out[y * w as usize * 4..][..w as usize * 4].iter_mut().zip(row) {
                *o = v.to_f32();
            }
        }
        drop(data);
        staging.unmap();
        out
    }
}

/// `a` over `b`, both premultiplied, into `a`.
pub fn over(a: &mut [f32], b: &[f32]) {
    for (p, q) in a.chunks_exact_mut(4).zip(b.chunks_exact(4)) {
        let k = 1.0 - p[3];
        for c in 0..4 {
            p[c] += k * q[c];
        }
    }
}

/// The holes in a premultiplied picture (alpha under a half), told apart:
/// the share that reaches the picture's border (what a crop removes), the
/// share that does not (what a nearer thing hid: needs filling), and the
/// centred crop, as a share of the frame's size, that clears the border
/// kind (all but a thousandth of the frame).
pub fn holes(px: &[f32], w: usize, h: usize) -> (f64, f64, f64) {
    let hole: Vec<bool> = px.chunks_exact(4).map(|p| p[3] < 0.5).collect();
    let mut edge = vec![false; w * h];
    let mut stack: Vec<usize> = Vec::new();
    for x in 0..w {
        stack.extend([x, (h - 1) * w + x]);
    }
    for y in 0..h {
        stack.extend([y * w, y * w + w - 1]);
    }
    while let Some(i) = stack.pop() {
        if !hole[i] || edge[i] {
            continue;
        }
        edge[i] = true;
        let (x, y) = (i % w, i / w);
        if x > 0 {
            stack.push(i - 1);
        }
        if x + 1 < w {
            stack.push(i + 1);
        }
        if y > 0 {
            stack.push(i - w);
        }
        if y + 1 < h {
            stack.push(i + w);
        }
    }
    let n = (w * h) as f64;
    let border = edge.iter().filter(|e| **e).count() as f64;
    let inner = hole.iter().zip(&edge).filter(|(a, b)| **a && !**b).count() as f64;
    // Each border hole's distance from the centre, as a share of the half
    // size on its worse axis: the crop must come inside all but a few.
    let mut d: Vec<f64> = (0..w * h)
        .filter(|i| edge[*i])
        .map(|i| {
            let (x, y) = ((i % w) as f64 + 0.5, (i / w) as f64 + 0.5);
            ((x / w as f64 - 0.5).abs() * 2.0).max((y / h as f64 - 0.5).abs() * 2.0)
        })
        .collect();
    let spare = (n * 0.001) as usize;
    let crop = if d.len() <= spare {
        1.0
    } else {
        d.select_nth_unstable_by(spare, |a, b| a.total_cmp(b));
        d[spare]
    };
    (border / n, inner / n, crop)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoothing_a_smooth_path_leaves_it() {
        // A straight dolly with a steady pan: the low-pass has nothing to
        // take, away from the ends.
        let poses: Vec<Pose> = (0..60)
            .map(|i| {
                let r = so3_exp([0.0, 0.002 * i as f64, 0.0]);
                Pose { r, t: scl(m_v(&r, [0.01 * i as f64, 0.0, 0.02 * i as f64]), -1.0) }
            })
            .collect();
        let s = smooth_path(&poses, 4.0);
        let (c, r) = path_delta(&poses[15..45], &s[15..45]);
        assert!(c[1] < 1e-9 && r[1] < 1e-6, "{c:?} {r:?}");
    }

    #[test]
    fn smoothing_takes_the_shake_out() {
        let shake = |i: usize| ((i * 7919) % 13) as f64 / 13.0 - 0.5;
        let poses: Vec<Pose> = (0..80)
            .map(|i| {
                let r = so3_exp([0.02 * shake(i), 0.02 * shake(i + 3), 0.0]);
                Pose { r, t: scl(m_v(&r, [0.05 * shake(i + 5), 0.0, 0.02 * i as f64]), -1.0) }
            })
            .collect();
        let s = smooth_path(&poses, 8.0);
        let rough = |p: &[Pose]| -> f64 {
            p.windows(3)
                .map(|w| crate::camtrack::norm(sub(add(w[0].centre(), w[2].centre()), scl(w[1].centre(), 2.0))))
                .sum()
        };
        assert!(rough(&s[10..70]) < 0.1 * rough(&poses[10..70]));
    }

    #[test]
    fn holes_are_told_apart() {
        let (w, h) = (100, 50);
        let mut px = vec![1f32; w * h * 4];
        for y in 0..h {
            for x in 0..5 {
                px[(y * w + x) * 4 + 3] = 0.0; // a strip on the left edge
            }
        }
        for y in 20..25 {
            for x in 40..50 {
                px[(y * w + x) * 4 + 3] = 0.0; // an island
            }
        }
        let (border, inner, crop) = holes(&px, w, h);
        assert!((border - 0.05).abs() < 1e-9 && (inner - 0.01).abs() < 1e-9);
        assert!((crop - 0.9).abs() < 0.02, "{crop}");
    }

    #[test]
    fn a_pixel_goes_out_and_comes_back() {
        // The cloud's point, taken through the uniform's model-view and
        // lens the way the project shader does, lands on its own pixel.
        let fr = Frame { w: 8, h: 6, rgba: vec![0.5; 8 * 6 * 4], z: vec![3.0; 8 * 6] };
        let r = so3_exp([0.1, -0.2, 0.05]);
        let pose = Pose { r, t: [0.3, -0.1, 0.4] };
        let c = cloud(&fr, 2.0, &pose, 10.0, &CloudOpts { sigma_px: 0.5, edge: 0.0, step: 1 });
        assert_eq!(c.count, 48);
        let u = uniform(&pose, 10.0, 8, 6, 0.1, 100.0, c.count);
        let v: &[f32] = bytemuck::cast_slice(&u[..112]);
        let k = 2 * 8 + 5; // pixel (5, 2)
        let p = [0, 1, 2].map(|i| f32::from_bits(c.splats[k * SPLAT_WORDS + i]));
        let e = [0, 1, 2].map(|r| v[r * 4] * p[0] + v[r * 4 + 1] * p[1] + v[r * 4 + 2] * p[2] + v[r * 4 + 3]);
        let t = -e[2];
        assert!((t - 6.0).abs() < 1e-4);
        let (sx, sy) = (v[17] + v[16] * e[0] / t, v[18] - v[16] * e[1] / t);
        assert!((sx - 5.5).abs() < 1e-3 && (sy - 2.5).abs() < 1e-3, "{sx} {sy}");
    }
}
