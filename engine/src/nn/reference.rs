//! The CPU statement of a [`Net`]: slow, plain f32, one tile the size of
//! the whole picture (so it is exactly what torch computes). It is what the
//! GPU kernels are tested against, and what an architecture builder is
//! checked with before a GPU is involved.

use super::{Epi, Net, Op, Out, Slice};

/// Run `net` over an RGB picture in 0..1 (`[3][h][w]`, planar) and return
/// its output, `[3][h*s][w*s]`, with no colour handling at all.
pub fn run(net: &Net, rgb: &[f32], w: usize, h: usize) -> Vec<f32> {
    assert_eq!(rgb.len(), 3 * w * h);
    let mut bufs: Vec<Vec<f32>> = net
        .bufs
        .iter()
        .map(|b| {
            assert!(w * b.res as usize % b.div as usize == 0 && h * b.res as usize % b.div as usize == 0, "the picture is not on the net's grid");
            vec![0.0; b.ch as usize * (w * b.res as usize / b.div as usize) * (h * b.res as usize / b.div as usize)]
        })
        .collect();
    let dims = |b: usize| {
        let (r, d) = (net.bufs[b].res as usize, net.bufs[b].div as usize);
        (w * r / d, h * r / d)
    };
    // The input slice: normalised rgb, alpha (1) in channel 3.
    {
        let (bw, bh) = dims(net.input.buf);
        assert_eq!((bw, bh), (w, h), "the input buffer is at the tile's size");
        let base = net.input.ch as usize * w * h;
        for c in 0..4 {
            for i in 0..w * h {
                bufs[net.input.buf][base + c * w * h + i] = if c < 3 {
                    (rgb[c * w * h + i] - net.mean[c]) * net.range
                } else {
                    1.0
                };
            }
        }
    }
    for op in &net.ops {
        match op {
            Op::Pool { src, dst } => {
                let (sw, sh) = dims(src.buf);
                let (dw, dh) = dims(dst.buf);
                assert_eq!((dw * 2, dh * 2), (sw, sh));
                for c in 0..src.n as usize {
                    for y in 0..dh {
                        for x in 0..dw {
                            let at = |xx: usize, yy: usize| bufs[src.buf][(src.ch as usize + c) * sw * sh + yy * sw + xx];
                            let v = at(2 * x, 2 * y).max(at(2 * x + 1, 2 * y)).max(at(2 * x, 2 * y + 1)).max(at(2 * x + 1, 2 * y + 1));
                            bufs[dst.buf][(dst.ch as usize + c) * dw * dh + y * dw + x] = v;
                        }
                    }
                }
            }
            Op::LogSkip { src, dst, scale, offset } => {
                let (bw, bh) = dims(src.buf);
                assert_eq!(dims(dst.buf), (bw, bh));
                for c in 0..src.n as usize {
                    for i in 0..bw * bh {
                        let v = bufs[src.buf][(src.ch as usize + c) * bw * bh + i] * scale + offset[c % 4];
                        bufs[dst.buf][(dst.ch as usize + c) * bw * bh + i] = (v * v + 1.0 / 255.0).ln();
                    }
                }
            }
            Op::Deconv(c) => {
                let (sw, sh) = dims(c.src.buf);
                let (dw, dh) = dims(c.dst.buf);
                assert_eq!((sw * 2, sh * 2), (dw, dh));
                assert_eq!(c.k, 4);
                let mut out = vec![0.0f32; c.cout as usize * dw * dh];
                for o in 0..c.cout as usize {
                    for y in 0..dh {
                        for x in 0..dw {
                            let mut acc = c.bias[o];
                            // Inputs i with tap o - 2i + 1 in 0..4.
                            for iy in (y as isize - 2) / 2..=(y as isize + 1) / 2 {
                                let ky = y as isize - 2 * iy + 1;
                                if !(0..4).contains(&ky) || iy < 0 || iy >= sh as isize {
                                    continue;
                                }
                                for ix in (x as isize - 2) / 2..=(x as isize + 1) / 2 {
                                    let kx = x as isize - 2 * ix + 1;
                                    if !(0..4).contains(&kx) || ix < 0 || ix >= sw as isize {
                                        continue;
                                    }
                                    for i in 0..c.cin as usize {
                                        let v = bufs[c.src.buf][(c.src.ch as usize + i) * sw * sh + iy as usize * sw + ix as usize];
                                        acc += c.weight[((o * c.cin as usize + i) * 4 + ky as usize) * 4 + kx as usize] * v;
                                    }
                                }
                            }
                            out[o * dw * dh + y * dw + x] = acc;
                        }
                    }
                }
                epilogue(&c.epi, &mut out, &bufs, dw * dh);
                let d = c.dst.ch as usize * dw * dh;
                bufs[c.dst.buf][d..d + out.len()].copy_from_slice(&out);
            }
            Op::Copy { src, dst } => {
                let (bw, bh) = dims(src.buf);
                let n = src.n as usize * bw * bh;
                let s = src.ch as usize * bw * bh;
                let d = dst.ch as usize * bw * bh;
                let tmp = bufs[src.buf][s..s + n].to_vec();
                bufs[dst.buf][d..d + n].copy_from_slice(&tmp);
            }
            Op::Conv(c) => {
                let (dw, dh) = dims(c.dst.buf);
                let (sw, sh) = dims(c.src.buf);
                assert_eq!((sw * c.up as usize, sh * c.up as usize), (dw, dh));
                let k = c.k as usize;
                let r = (k / 2) as isize;
                let read = |bufs: &Vec<Vec<f32>>, s: &Slice, ch: usize, x: usize, y: usize, pw: usize, ph: usize| {
                    bufs[s.buf][(s.ch as usize + ch) * pw * ph + y * pw + x]
                };
                let mut out = vec![0.0f32; c.cout as usize * dw * dh];
                for o in 0..c.cout as usize {
                    for y in 0..dh {
                        for x in 0..dw {
                            let mut acc = c.bias[o];
                            for i in 0..c.cin as usize {
                                for ky in 0..k {
                                    for kx in 0..k {
                                        let yy = y as isize + ky as isize - r;
                                        let xx = x as isize + kx as isize - r;
                                        if yy < 0 || xx < 0 || yy >= dh as isize || xx >= dw as isize {
                                            continue;
                                        }
                                        let sx = xx as usize / c.up as usize;
                                        let sy = yy as usize / c.up as usize;
                                        let mut v = read(&bufs, &c.src, i, sx, sy, sw, sh);
                                        if c.silu_in {
                                            v = silu(v);
                                        }
                                        acc += c.weight[((o * c.cin as usize + i) * k + ky) * k + kx] * v;
                                    }
                                }
                            }
                            out[o * dw * dh + y * dw + x] = acc;
                        }
                    }
                }
                epilogue(&c.epi, &mut out, &bufs, dw * dh);
                let d = c.dst.ch as usize * dw * dh;
                bufs[c.dst.buf][d..d + out.len()].copy_from_slice(&out);
            }
        }
    }
    let s = net.scale as usize;
    let (ow, oh) = (w * s, h * s);
    let mut res = vec![0.0f32; 3 * ow * oh];
    for c in 0..3 {
        for y in 0..oh {
            for x in 0..ow {
                let mut v = match net.out {
                    Out::Direct(sl) => bufs[sl.buf][(sl.ch as usize + c) * ow * oh + y * ow + x],
                    Out::Shuffle(sl) => {
                        let ch = c * s * s + (y % s) * s + (x % s);
                        bufs[sl.buf][(sl.ch as usize + ch) * w * h + (y / s) * w + x / s]
                    }
                };
                if net.base {
                    v += rgb[c * w * h + (y / s) * w + x / s];
                }
                res[c * ow * oh + y * ow + x] = v;
            }
        }
    }
    res
}

/// A conv's epilogue over its raw output (`plane` values per channel).
fn epilogue(epi: &Epi, out: &mut [f32], bufs: &[Vec<f32>], plane: usize) {
    for (j, slot) in out.iter_mut().enumerate() {
        let (o, i) = (j / plane, j % plane);
        let v = *slot;
        *slot = match epi {
            Epi::Plain { alpha, res, slope } => {
                let mut v = alpha * v;
                for (s, beta) in res {
                    v += beta * bufs[s.buf][(s.ch as usize + o) * plane + i];
                }
                match slope {
                    Some(sl) if v < 0.0 => v * sl[o],
                    _ => v,
                }
            }
            Epi::Gate { res } => {
                let r = bufs[res.buf][(res.ch as usize + o) * plane + i];
                (v + r) * (sigmoid(v) - 0.5)
            }
            Epi::Silu => silu(v),
        };
    }
}

/// HDRCNN's own last step (`get_final` upstream), on one pixel: `x` is
/// what the net was given (gamma-2 encoded, 0..1), `y` its log
/// prediction. Linear out, relative to the clip level: the input's `x²`
/// where it is far from the clip, the prediction where it reaches it.
pub fn hdr_blend(x: [f32; 3], y: [f32; 3]) -> [f32; 3] {
    let a = hdr_alpha(x);
    std::array::from_fn(|c| (1.0 - a) * x[c] * x[c] + a * (y[c].exp() - 1.0 / 255.0))
}

/// The highlight mask: 0 under 0.95 in the brightest channel, 1 at 1.
pub fn hdr_alpha(x: [f32; 3]) -> f32 {
    let m = x[0].max(x[1]).max(x[2]);
    ((m - 1.0 + HDR_THR).max(0.0) / HDR_THR).min(1.0)
}

/// Where the mask starts: 0.05 under the top of the encoded range.
pub const HDR_THR: f32 = 0.05;

pub fn silu(v: f32) -> f32 {
    v * sigmoid(v)
}

pub fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}
