//! CPU mirror of the IFNet forward pass. Test oracle for the WGSL
//! implementation, and the `rife --cpu` debug path. Deliberately written as
//! the same pass sequence the GPU records, with PyTorch-exact resize/warp
//! semantics; rayon only over conv output channels so `--cpu` stays usable.

use super::*;
use rayon::prelude::*;

/// `F.interpolate(mode="bilinear", align_corners=False)` over a channel range,
/// with a post-multiply (flow rescaling folds in here, like the GPU pass).
/// Reads `src` channels `src_c0..src_c0+nch`, writes into `dst` at `dst_c0`.
#[allow(clippy::too_many_arguments)]
pub fn resize_into(
    src: &[f32],
    sw: usize,
    sh: usize,
    src_c0: usize,
    dst: &mut [f32],
    dw: usize,
    dh: usize,
    dst_c0: usize,
    nch: usize,
    post_mul: f32,
) {
    let sx_scale = sw as f32 / dw as f32;
    let sy_scale = sh as f32 / dh as f32;
    for c in 0..nch {
        let sp = &src[(src_c0 + c) * sw * sh..][..sw * sh];
        let dp = &mut dst[(dst_c0 + c) * dw * dh..][..dw * dh];
        for oy in 0..dh {
            // align_corners=False: src = (dst + 0.5) * (in/out) - 0.5, clamped low.
            let sy = ((oy as f32 + 0.5) * sy_scale - 0.5).max(0.0);
            let y0 = (sy as usize).min(sh - 1);
            let y1 = (y0 + 1).min(sh - 1);
            let fy = sy - y0 as f32;
            for ox in 0..dw {
                let sx = ((ox as f32 + 0.5) * sx_scale - 0.5).max(0.0);
                let x0 = (sx as usize).min(sw - 1);
                let x1 = (x0 + 1).min(sw - 1);
                let fx = sx - x0 as f32;
                let top = sp[y0 * sw + x0] * (1.0 - fx) + sp[y0 * sw + x1] * fx;
                let bot = sp[y1 * sw + x0] * (1.0 - fx) + sp[y1 * sw + x1] * fx;
                dp[oy * dw + ox] = (top * (1.0 - fy) + bot * fy) * post_mul;
            }
        }
    }
}

/// Conv3x3 (zero pad 1) + per-channel-slope activation. `stride` 1 or 2.
pub fn conv3x3(
    src: &[f32],
    in_c: usize,
    ih: usize,
    iw: usize,
    layer: &ConvLayer,
    stride: usize,
) -> (Vec<f32>, usize, usize) {
    let out_c = layer.b.data.len();
    let oh = (ih + 2 - 3) / stride + 1;
    let ow = (iw + 2 - 3) / stride + 1;
    let mut out = vec![0f32; out_c * oh * ow];
    out.par_chunks_mut(oh * ow).enumerate().for_each(|(oc, op)| {
        let wslice = &layer.w.data[oc * in_c * 9..][..in_c * 9];
        let bias = layer.b.data[oc];
        let slope = layer.slope[oc];
        for oy in 0..oh {
            for ox in 0..ow {
                let mut acc = bias;
                for ic in 0..in_c {
                    let sp = &src[ic * ih * iw..];
                    let wk = &wslice[ic * 9..][..9];
                    for ky in 0..3 {
                        let iy = (oy * stride + ky) as isize - 1;
                        if iy < 0 || iy >= ih as isize {
                            continue;
                        }
                        for kx in 0..3 {
                            let ix = (ox * stride + kx) as isize - 1;
                            if ix < 0 || ix >= iw as isize {
                                continue;
                            }
                            acc += sp[iy as usize * iw + ix as usize] * wk[ky * 3 + kx];
                        }
                    }
                }
                op[oy * ow + ox] = if acc >= 0.0 { acc } else { acc * slope };
            }
        }
    });
    (out, oh, ow)
}

/// `relu(conv3x3(x) * beta + x)`.
pub fn resconv(src: &[f32], c: usize, h: usize, w: usize, layer: &ResConvLayer) -> Vec<f32> {
    let mut out = vec![0f32; c * h * w];
    out.par_chunks_mut(h * w).enumerate().for_each(|(oc, op)| {
        let wslice = &layer.w.data[oc * c * 9..][..c * 9];
        let bias = layer.b.data[oc];
        let beta = layer.beta[oc];
        let slope = layer.slope[oc];
        let res = &src[oc * h * w..][..h * w];
        for oy in 0..h {
            for ox in 0..w {
                let mut acc = bias;
                for ic in 0..c {
                    let sp = &src[ic * h * w..];
                    let wk = &wslice[ic * 9..][..9];
                    for ky in 0..3 {
                        let iy = (oy + ky) as isize - 1;
                        if iy < 0 || iy >= h as isize {
                            continue;
                        }
                        for kx in 0..3 {
                            let ix = (ox + kx) as isize - 1;
                            if ix < 0 || ix >= w as isize {
                                continue;
                            }
                            acc += sp[iy as usize * w + ix as usize] * wk[ky * 3 + kx];
                        }
                    }
                }
                let v = acc * beta + res[oy * w + ox];
                op[oy * w + ox] = if v >= 0.0 { v } else { v * slope };
            }
        }
    });
    out
}

/// ConvTranspose2d(c → 24, k4 s2 p1) + PixelShuffle(2), fused: input
/// `[c, h, w]` → output `[6, 4h, 4w]`. Same gather the WGSL kernel does.
pub fn deconv_ps(src: &[f32], c: usize, h: usize, w: usize, layer: &DeconvLayer) -> Vec<f32> {
    let oh = 4 * h;
    let ow = 4 * w;
    let mut out = vec![0f32; HEAD_C * oh * ow];
    out.par_chunks_mut(oh * ow).enumerate().for_each(|(oc6, op)| {
        for oy in 0..oh {
            for ox in 0..ow {
                // PixelShuffle: which deconv channel and pixel feeds this output.
                let dc = oc6 * 4 + (oy % 2) * 2 + (ox % 2);
                let dy = oy / 2; // position in the [24, 2h, 2w] deconv result
                let dx = ox / 2;
                let mut acc = layer.b.data[dc];
                for ky in 0..4usize {
                    let ty = dy as isize + 1 - ky as isize;
                    if ty < 0 || ty % 2 != 0 {
                        continue;
                    }
                    let iy = (ty / 2) as usize;
                    if iy >= h {
                        continue;
                    }
                    for kx in 0..4usize {
                        let tx = dx as isize + 1 - kx as isize;
                        if tx < 0 || tx % 2 != 0 {
                            continue;
                        }
                        let ix = (tx / 2) as usize;
                        if ix >= w {
                            continue;
                        }
                        for ic in 0..c {
                            acc += src[ic * h * w + iy * w + ix]
                                * layer.w.data[((ic * 24 + dc) * 4 + ky) * 4 + kx];
                        }
                    }
                }
                op[oy * ow + ox] = acc;
            }
        }
    });
    out
}

/// Backward warp: sample `img` at `(x + fx, y + fy)` — RIFE's
/// `grid_sample(padding_mode='border', align_corners=True)`. `clamp01`
/// reproduces the net's input clamp; off for the HDR-honest blend.
pub fn warp(
    img: &[f32],
    c: usize,
    h: usize,
    w: usize,
    flow: &[f32],
    flow_c0: usize,
    clamp01: bool,
) -> Vec<f32> {
    let mut out = vec![0f32; c * h * w];
    let fx_p = &flow[flow_c0 * h * w..][..h * w];
    let fy_p = &flow[(flow_c0 + 1) * h * w..][..h * w];
    for ch in 0..c {
        let sp = &img[ch * h * w..][..h * w];
        let op = &mut out[ch * h * w..][..h * w];
        for y in 0..h {
            for x in 0..w {
                let px = (x as f32 + fx_p[y * w + x]).clamp(0.0, (w - 1) as f32);
                let py = (y as f32 + fy_p[y * w + x]).clamp(0.0, (h - 1) as f32);
                let x0 = (px as usize).min(w - 1);
                let y0 = (py as usize).min(h - 1);
                let x1 = (x0 + 1).min(w - 1);
                let y1 = (y0 + 1).min(h - 1);
                let fx = px - x0 as f32;
                let fy = py - y0 as f32;
                let s = |xx: usize, yy: usize| {
                    let v = sp[yy * w + xx];
                    if clamp01 { v.clamp(0.0, 1.0) } else { v }
                };
                op[y * w + x] = (s(x0, y0) * (1.0 - fx) + s(x1, y0) * fx) * (1.0 - fy)
                    + (s(x0, y1) * (1.0 - fx) + s(x1, y1) * fx) * fy;
            }
        }
    }
    out
}

/// Full forward on `[3][h][w]` inputs; pads to /32 internally, crops back.
pub fn forward(model: &RifeModel, img0: &[f32], img1: &[f32], w: usize, h: usize, t: f32) -> RifeOutput {
    forward_traced(model, img0, img1, w, h, t, |_, _| {})
}

/// `forward` with a checkpoint callback after every stage — the GPU parity
/// bisection hook. Checkpoint names match the GPU pass labels.
pub fn forward_traced(
    model: &RifeModel,
    img0: &[f32],
    img1: &[f32],
    w: usize,
    h: usize,
    t: f32,
    mut trace: impl FnMut(&str, &[f32]),
) -> RifeOutput {
    let pw = pad32(w);
    let ph = pad32(h);
    let i0 = pad_planes(img0, 3, w, h, pw, ph);
    let i1 = pad_planes(img1, 3, w, h, pw, ph);
    let full = pw * ph;

    let clamp3 = |src: &[f32]| -> Vec<f32> { src.iter().map(|v| v.clamp(0.0, 1.0)).collect() };
    let i0c = clamp3(&i0);
    let i1c = clamp3(&i1);

    let mut flow = vec![0f32; 4 * full];
    let mut mask = vec![0f32; full];
    let mut warped0 = i0c.clone();
    let mut warped1 = i1c.clone();

    for (i, &scale) in SCALES.iter().enumerate() {
        let blk = &model.blocks[i];
        let (rw, rh) = (pw / scale, ph / scale);
        // Pack the full-res input stack, then downsample — matches
        // interpolate-then-conv in the reference net (resize is per-channel).
        let base = if i == 0 { 7 } else { 8 };
        let mut xfull = vec![0f32; base * full];
        if i == 0 {
            xfull[..3 * full].copy_from_slice(&i0c);
            xfull[3 * full..6 * full].copy_from_slice(&i1c);
            xfull[6 * full..7 * full].fill(t);
        } else {
            xfull[..3 * full].copy_from_slice(&warped0);
            xfull[3 * full..6 * full].copy_from_slice(&warped1);
            xfull[6 * full..7 * full].fill(t);
            xfull[7 * full..8 * full].copy_from_slice(&mask);
        }
        let in_c = blk.in_planes;
        let mut xs = vec![0f32; in_c * rw * rh];
        resize_into(&xfull, pw, ph, 0, &mut xs, rw, rh, 0, base, 1.0);
        if i > 0 {
            resize_into(&flow, pw, ph, 0, &mut xs, rw, rh, 8, 4, 1.0 / scale as f32);
        }
        trace(&format!("xs{i}"), &xs);

        let (h1, h1h, h1w) = conv3x3(&xs, in_c, rh, rw, &blk.conv0[0], 2);
        trace(&format!("c0a{i}"), &h1);
        let (mut feat, fh, fw) = conv3x3(&h1, blk.c / 2, h1h, h1w, &blk.conv0[1], 2);
        trace(&format!("c0b{i}"), &feat);
        for rc in &blk.convblock {
            feat = resconv(&feat, blk.c, fh, fw, rc);
        }
        trace(&format!("rc{i}"), &feat);
        let tmp = deconv_ps(&feat, blk.c, fh, fw, &blk.lastconv); // [6, rh, rw]
        trace(&format!("dc{i}"), &tmp);

        if i == 0 {
            resize_into(&tmp, rw, rh, 0, &mut flow, pw, ph, 0, 4, scale as f32);
            resize_into(&tmp, rw, rh, 4, &mut mask, pw, ph, 0, 1, 1.0);
        } else {
            let mut fd = vec![0f32; 4 * full];
            let mut md = vec![0f32; full];
            resize_into(&tmp, rw, rh, 0, &mut fd, pw, ph, 0, 4, scale as f32);
            resize_into(&tmp, rw, rh, 4, &mut md, pw, ph, 0, 1, 1.0);
            for (a, b) in flow.iter_mut().zip(&fd) {
                *a += b;
            }
            for (a, b) in mask.iter_mut().zip(&md) {
                *a += b;
            }
        }

        trace(&format!("flow{i}"), &flow);
        trace(&format!("mask{i}"), &mask);
        warped0 = warp(&i0, 3, ph, pw, &flow, 0, true);
        warped1 = warp(&i1, 3, ph, pw, &flow, 2, true);
        trace(&format!("warp{i}"), &warped0);
    }

    let m: Vec<f32> = mask.iter().map(|&v| sigmoid(v)).collect();
    let blend = |a: &[f32], b: &[f32]| -> Vec<f32> {
        let mut out = vec![0f32; 3 * full];
        for c in 0..3 {
            for p in 0..full {
                out[c * full + p] = a[c * full + p] * m[p] + b[c * full + p] * (1.0 - m[p]);
            }
        }
        out
    };
    let merged = blend(&warped0, &warped1);
    let hw0 = warp(&i0, 3, ph, pw, &flow, 0, false);
    let hw1 = warp(&i1, 3, ph, pw, &flow, 2, false);
    let merged_hdr = blend(&hw0, &hw1);

    RifeOutput {
        merged: crop_planes(&merged, 3, pw, ph, w, h),
        merged_hdr: crop_planes(&merged_hdr, 3, pw, ph, w, h),
        flow: crop_planes(&flow, 4, pw, ph, w, h),
        mask: crop_planes(&m, 1, pw, ph, w, h),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_matches_torch_semantics() {
        // 2 -> 1 downsample, align_corners=False: src = (0+0.5)*2-0.5 = 0.5 -> mid.
        let src = [1.0f32, 3.0];
        let mut dst = [0f32];
        resize_into(&src, 2, 1, 0, &mut dst, 1, 1, 0, 1, 1.0);
        assert!((dst[0] - 2.0).abs() < 1e-6);
        // 1 -> 2 upsample: both outputs clamp to the single sample.
        let mut up = [0f32; 2];
        resize_into(&[5.0f32], 1, 1, 0, &mut up, 2, 1, 0, 1, 1.0);
        assert_eq!(up, [5.0, 5.0]);
        // 4 -> 2: src = (0.5)*2 - 0.5 = 0.5 and (1.5)*2 - 0.5 = 2.5.
        let src4 = [0.0f32, 10.0, 20.0, 30.0];
        let mut d2 = [0f32; 2];
        resize_into(&src4, 4, 1, 0, &mut d2, 2, 1, 0, 1, 1.0);
        assert!((d2[0] - 5.0).abs() < 1e-5 && (d2[1] - 25.0).abs() < 1e-5);
    }

    #[test]
    fn warp_zero_flow_is_identity_and_shift_shifts() {
        let (w, h) = (4, 3);
        let img: Vec<f32> = (0..w * h).map(|i| i as f32).collect();
        let flow = vec![0f32; 4 * w * h];
        assert_eq!(warp(&img, 1, h, w, &flow, 0, false), img);
        let mut shift = vec![0f32; 4 * w * h];
        shift[..w * h].fill(1.0); // fx = +1: sample one pixel right
        let out = warp(&img, 1, h, w, &shift, 0, false);
        assert_eq!(out[0], img[1]);
        assert_eq!(out[w - 1], img[w - 1]); // border clamp
    }

    #[test]
    fn conv_stride2_shapes_and_bias() {
        let layer = ConvLayer {
            w: Tensor { dims: vec![2, 1, 3, 3], data: vec![0.0; 18] },
            b: Tensor { dims: vec![2], data: vec![1.5, -2.0] },
            slope: vec![0.5, 0.5],
        };
        let (out, oh, ow) = conv3x3(&vec![0f32; 8 * 6], 1, 6, 8, &layer, 2);
        assert_eq!((oh, ow), (3, 4));
        assert_eq!(out[0], 1.5); // positive bias passes
        assert_eq!(out[oh * ow], -1.0); // negative bias through slope 0.5
    }

    #[test]
    fn forward_shapes_finite() {
        let model = RifeModel::random(11);
        let (w, h) = (64, 32);
        let mut rng = Lcg(5);
        let img0: Vec<f32> = (0..3 * w * h).map(|_| rng.uniform()).collect();
        let img1: Vec<f32> = (0..3 * w * h).map(|_| rng.uniform()).collect();
        let out = forward(&model, &img0, &img1, w, h, 0.5);
        assert_eq!(out.merged.len(), 3 * w * h);
        assert_eq!(out.flow.len(), 4 * w * h);
        assert_eq!(out.mask.len(), w * h);
        assert!(out.merged.iter().chain(&out.flow).chain(&out.mask).all(|v| v.is_finite()));
        assert!(out.mask.iter().all(|&m| (0.0..=1.0).contains(&m)));
    }
}
