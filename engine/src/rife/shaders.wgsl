// RIFE v4.6 IFNet compute kernels. gpu.rs splits this file on the
// `---kernel:` comment markers into one shader module per kernel, so
// binding slots never collide.
// All tensors are flat f32 storage buffers, plane-major: idx = c*h*w + y*w + x.
// Semantics mirror reference.rs exactly (which mirrors PyTorch); any change
// here must keep the CPU/GPU parity test green.

//---kernel:pack
// Build a block's full-res input stack.
//   mode 0 (block0): [clamp(img0) 3][clamp(img1) 3][t]           -> 7 planes
//   mode 1 (later):  [warped0 3][warped1 3][t][mask]             -> 8 planes
struct PackU { w: u32, h: u32, mode: u32, _p: u32, t: f32, _p1: f32, _p2: f32, _p3: f32 }
@group(0) @binding(0) var<uniform> u: PackU;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> mask: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn pack(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.w || gid.y >= u.h) { return; }
    let plane = u.w * u.h;
    let p = gid.y * u.w + gid.x;
    for (var c = 0u; c < 3u; c++) {
        var va = a[c * plane + p];
        var vb = b[c * plane + p];
        if (u.mode == 0u) {
            va = clamp(va, 0.0, 1.0);
            vb = clamp(vb, 0.0, 1.0);
        }
        dst[c * plane + p] = va;
        dst[(c + 3u) * plane + p] = vb;
    }
    dst[6u * plane + p] = u.t;
    if (u.mode == 1u) {
        dst[7u * plane + p] = mask[p];
    }
}

//---kernel:resize
// F.interpolate(mode="bilinear", align_corners=False) over a channel range,
// times post_mul. Reads src channels sc0.., writes dst channels dc0.. .
struct ResizeU {
    sw: u32, sh: u32, dw: u32, dh: u32,
    sc0: u32, dc0: u32, nch: u32, _p: u32,
    post_mul: f32, _p1: f32, _p2: f32, _p3: f32,
}
@group(0) @binding(0) var<uniform> u: ResizeU;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn resize(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.dw || gid.y >= u.dh || gid.z >= u.nch) { return; }
    let sp = (u.sc0 + gid.z) * u.sw * u.sh;
    let sx = max((f32(gid.x) + 0.5) * f32(u.sw) / f32(u.dw) - 0.5, 0.0);
    let sy = max((f32(gid.y) + 0.5) * f32(u.sh) / f32(u.dh) - 0.5, 0.0);
    let x0 = min(u32(sx), u.sw - 1u);
    let y0 = min(u32(sy), u.sh - 1u);
    let x1 = min(x0 + 1u, u.sw - 1u);
    let y1 = min(y0 + 1u, u.sh - 1u);
    let fx = sx - f32(x0);
    let fy = sy - f32(y0);
    let top = src[sp + y0 * u.sw + x0] * (1.0 - fx) + src[sp + y0 * u.sw + x1] * fx;
    let bot = src[sp + y1 * u.sw + x0] * (1.0 - fx) + src[sp + y1 * u.sw + x1] * fx;
    dst[(u.dc0 + gid.z) * u.dw * u.dh + gid.y * u.dw + gid.x] =
        (top * (1.0 - fy) + bot * fy) * u.post_mul;
}

//---kernel:conv
// Conv3x3, zero pad 1, stride 1 or 2, + per-out-channel-slope activation
// (LeakyReLU/PReLU unified). One thread per (ox, oy, oc).
struct ConvU {
    in_c: u32, out_c: u32, ih: u32, iw: u32,
    oh: u32, ow: u32, stride: u32, _p: u32,
}
@group(0) @binding(0) var<uniform> u: ConvU;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> wgt: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> slope: array<f32>;
@group(0) @binding(5) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn conv(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.ow || gid.y >= u.oh) { return; }
    let oc = gid.z;
    var acc = bias[oc];
    let wbase = oc * u.in_c * 9u;
    for (var ic = 0u; ic < u.in_c; ic++) {
        let sp = ic * u.ih * u.iw;
        let wb = wbase + ic * 9u;
        for (var ky = 0u; ky < 3u; ky++) {
            let iy = i32(gid.y * u.stride + ky) - 1;
            if (iy < 0 || iy >= i32(u.ih)) { continue; }
            for (var kx = 0u; kx < 3u; kx++) {
                let ix = i32(gid.x * u.stride + kx) - 1;
                if (ix < 0 || ix >= i32(u.iw)) { continue; }
                acc += src[sp + u32(iy) * u.iw + u32(ix)] * wgt[wb + ky * 3u + kx];
            }
        }
    }
    if (acc < 0.0) { acc *= slope[oc]; }
    dst[oc * u.oh * u.ow + gid.y * u.ow + gid.x] = acc;
}

//---kernel:resconv
// relu(conv3x3(x) * beta + x): the v4.6 residual block, stride 1, c -> c.
struct ResConvU { c: u32, h: u32, w: u32, _p: u32 }
@group(0) @binding(0) var<uniform> u: ResConvU;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> wgt: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> beta: array<f32>;
@group(0) @binding(5) var<storage, read> slope: array<f32>;
@group(0) @binding(6) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn resconv(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.w || gid.y >= u.h) { return; }
    let oc = gid.z;
    let plane = u.h * u.w;
    var acc = bias[oc];
    let wbase = oc * u.c * 9u;
    for (var ic = 0u; ic < u.c; ic++) {
        let sp = ic * plane;
        let wb = wbase + ic * 9u;
        for (var ky = 0u; ky < 3u; ky++) {
            let iy = i32(gid.y + ky) - 1;
            if (iy < 0 || iy >= i32(u.h)) { continue; }
            for (var kx = 0u; kx < 3u; kx++) {
                let ix = i32(gid.x + kx) - 1;
                if (ix < 0 || ix >= i32(u.w)) { continue; }
                acc += src[sp + u32(iy) * u.w + u32(ix)] * wgt[wb + ky * 3u + kx];
            }
        }
    }
    let p = gid.y * u.w + gid.x;
    var v = acc * beta[oc] + src[oc * plane + p];
    if (v < 0.0) { v *= slope[oc]; }
    dst[oc * plane + p] = v;
}

//---kernel:deconv_ps
// ConvTranspose2d(c -> 24, k4 s2 p1) + PixelShuffle(2), fused gather:
// input [c, h, w] -> output [6, 4h, 4w]. No activation.
struct DeconvU { c: u32, h: u32, w: u32, _p: u32 }
@group(0) @binding(0) var<uniform> u: DeconvU;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> wgt: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn deconv_ps(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ow = 4u * u.w;
    let oh = 4u * u.h;
    if (gid.x >= ow || gid.y >= oh) { return; }
    // PixelShuffle source: deconv channel dc at deconv-res position (dx, dy).
    let dc = gid.z * 4u + (gid.y % 2u) * 2u + (gid.x % 2u);
    let dy = i32(gid.y / 2u);
    let dx = i32(gid.x / 2u);
    var acc = bias[dc];
    for (var ky = 0; ky < 4; ky++) {
        let ty = dy + 1 - ky;
        if (ty < 0 || (ty % 2) != 0) { continue; }
        let iy = u32(ty / 2);
        if (iy >= u.h) { continue; }
        for (var kx = 0; kx < 4; kx++) {
            let tx = dx + 1 - kx;
            if (tx < 0 || (tx % 2) != 0) { continue; }
            let ix = u32(tx / 2);
            if (ix >= u.w) { continue; }
            for (var ic = 0u; ic < u.c; ic++) {
                acc += src[ic * u.h * u.w + iy * u.w + ix]
                    * wgt[((ic * 24u + dc) * 4u + u32(ky)) * 4u + u32(kx)];
            }
        }
    }
    dst[gid.z * oh * ow + gid.y * ow + gid.x] = acc;
}

//---kernel:warp
// Backward warp: sample img (3ch) at (x + flow.x, y + flow.y) — RIFE's
// grid_sample(border, align_corners=True). clamp01 reproduces the net's
// input clamp; 0 for the HDR-honest pass.
struct WarpU { w: u32, h: u32, flow_c0: u32, clamp01: u32 }
@group(0) @binding(0) var<uniform> u: WarpU;
@group(0) @binding(1) var<storage, read> img: array<f32>;
@group(0) @binding(2) var<storage, read> flow: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn warp(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.w || gid.y >= u.h) { return; }
    let plane = u.w * u.h;
    let p = gid.y * u.w + gid.x;
    let px = clamp(f32(gid.x) + flow[u.flow_c0 * plane + p], 0.0, f32(u.w - 1u));
    let py = clamp(f32(gid.y) + flow[(u.flow_c0 + 1u) * plane + p], 0.0, f32(u.h - 1u));
    let x0 = min(u32(px), u.w - 1u);
    let y0 = min(u32(py), u.h - 1u);
    let x1 = min(x0 + 1u, u.w - 1u);
    let y1 = min(y0 + 1u, u.h - 1u);
    let fx = px - f32(x0);
    let fy = py - f32(y0);
    for (var c = 0u; c < 3u; c++) {
        let sp = c * plane;
        var s00 = img[sp + y0 * u.w + x0];
        var s10 = img[sp + y0 * u.w + x1];
        var s01 = img[sp + y1 * u.w + x0];
        var s11 = img[sp + y1 * u.w + x1];
        if (u.clamp01 == 1u) {
            s00 = clamp(s00, 0.0, 1.0);
            s10 = clamp(s10, 0.0, 1.0);
            s01 = clamp(s01, 0.0, 1.0);
            s11 = clamp(s11, 0.0, 1.0);
        }
        dst[sp + p] = (s00 * (1.0 - fx) + s10 * fx) * (1.0 - fy)
            + (s01 * (1.0 - fx) + s11 * fx) * fy;
    }
}

//---kernel:add
// dst[c,y,x] += src[c,y,x] — coarse-to-fine flow/mask accumulation.
struct AddU { w: u32, h: u32, nch: u32, _p: u32 }
@group(0) @binding(0) var<uniform> u: AddU;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn add(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.w || gid.y >= u.h || gid.z >= u.nch) { return; }
    let i = gid.z * u.w * u.h + gid.y * u.w + gid.x;
    dst[i] = dst[i] + src[i];
}

//---kernel:merge
// out = warped0 * sigmoid(mask) + warped1 * (1 - sigmoid(mask)).
struct MergeU { w: u32, h: u32, _p: u32, _p1: u32 }
@group(0) @binding(0) var<uniform> u: MergeU;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> mask: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn merge(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.w || gid.y >= u.h) { return; }
    let plane = u.w * u.h;
    let p = gid.y * u.w + gid.x;
    let m = 1.0 / (1.0 + exp(-mask[p]));
    for (var c = 0u; c < 3u; c++) {
        dst[c * plane + p] = a[c * plane + p] * m + b[c * plane + p] * (1.0 - m);
    }
}
