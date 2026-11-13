// SEA-RAFT compute kernels. searaft.rs splits this file on the
// `---kernel:` markers into one module per kernel.
// Every tensor is a flat f32 storage buffer, plane-major:
// idx = c*h*w + y*w + x. A kernel reads and writes channel RANGES of its
// buffers (`*_off`), which is how the network's concatenations are laid
// out without a copy.

//---kernel:conv
// A k×k convolution, any stride, zero padding, with an epilogue:
//   v = conv + bias
//   flags & 1: v = relu(v)       flags & 2: v = gelu(v)
//   flags & 4: v = v * scale[o]  flags & 8: v = v + res[res_off + o]
//   flags & 16: v = relu(v)      (after the residual)
// BatchNorm is folded into the weights on load.
//
// Tiled: a workgroup is 8×8 output pixels × 16 output channels. The
// workgroup stages the input patch it covers in shared memory, as many
// input channels at a time as fit, and every thread reuses each staged value for all 16
// of its output channels. Weights are laid out [oc/16][ic][k][k][16] on
// upload, so a tap's 16 weights are four vec4 loads, the same for every
// thread. (A 2×2 block of pixels per thread measured 3x SLOWER on an M4
// Pro: the extra accumulators spill.)
struct ConvU {
    ic: u32, oc: u32, ih: u32, iw: u32,
    oh: u32, ow: u32, k: u32, stride: u32,
    pad: u32, in_off: u32, out_off: u32, res_off: u32,
    flags: u32, _p0: u32, _p1: u32, _p2: u32,
}
@group(0) @binding(0) var<uniform> u: ConvU;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> wt: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> scale: array<f32>;
@group(0) @binding(5) var<storage, read> res: array<f32>;
@group(0) @binding(6) var<storage, read_write> dst: array<f32>;

// Room for four channels of the largest patch, k = 7 at stride 2:
// (8 - 1) * 2 + 7 = 21 a side.
var<workgroup> tile: array<f32, 1764>;

fn gelu(x: f32) -> f32 {
    // The exact (erf) GELU torch's nn.GELU() computes, through a
    // polynomial erf good to 1.5e-7.
    let z = x * 0.70710678;
    let a = abs(z);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
    let erf = select(-y, y, z >= 0.0);
    return 0.5 * x * (1.0 + erf);
}

@compute @workgroup_size(8, 8, 1)
fn conv(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wg: vec3<u32>) {
    let og = wg.z;
    let span = 7u * u.stride + u.k;
    let tx0 = i32(wg.x * 8u * u.stride) - i32(u.pad);
    let ty0 = i32(wg.y * 8u * u.stride) - i32(u.pad);
    let iplane = u.ih * u.iw;
    let kk = u.k * u.k;
    let li = lid.y * 8u + lid.x;
    var acc: array<vec4<f32>, 4>;
    let area = span * span;
    let bx = lid.x * u.stride;
    let by = lid.y * u.stride;
    // As many input channels a round as the tile holds, up to 16: a 1×1
    // conv over 512 channels is otherwise a barrier every four.
    let per = clamp(1764u / area, 1u, 16u);
    for (var c0 = 0u; c0 < u.ic; c0 += per) {
        let nc = min(per, u.ic - c0);
        workgroupBarrier();
        for (var t = li; t < nc * area; t += 64u) {
            let cc = t / area;
            let r = t % area;
            let x = tx0 + i32(r % span);
            let y = ty0 + i32(r / span);
            var v = 0.0;
            if (x >= 0 && x < i32(u.iw) && y >= 0 && y < i32(u.ih)) {
                v = src[(u.in_off + c0 + cc) * iplane + u32(y) * u.iw + u32(x)];
            }
            tile[t] = v;
        }
        workgroupBarrier();
        for (var cc = 0u; cc < nc; cc++) {
            let wbase = ((og * u.ic + c0 + cc) * kk) * 4u;
            let tbase = cc * area;
            for (var ky = 0u; ky < u.k; ky++) {
                for (var kx = 0u; kx < u.k; kx++) {
                    let v = tile[tbase + (by + ky) * span + bx + kx];
                    let wi = wbase + (ky * u.k + kx) * 4u;
                    acc[0] += v * wt[wi];
                    acc[1] += v * wt[wi + 1u];
                    acc[2] += v * wt[wi + 2u];
                    acc[3] += v * wt[wi + 3u];
                }
            }
        }
    }
    if (gid.x >= u.ow || gid.y >= u.oh) { return; }
    let p = gid.y * u.ow + gid.x;
    let oplane = u.oh * u.ow;
    for (var j = 0u; j < 16u; j++) {
        let o = og * 16u + j;
        if (o >= u.oc) { break; }
        var v = acc[j / 4u][j % 4u] + bias[o];
        if ((u.flags & 1u) != 0u) { v = max(v, 0.0); }
        if ((u.flags & 2u) != 0u) { v = gelu(v); }
        if ((u.flags & 4u) != 0u) { v = v * scale[o]; }
        if ((u.flags & 8u) != 0u) { v = v + res[(u.res_off + o) * oplane + p]; }
        if ((u.flags & 16u) != 0u) { v = max(v, 0.0); }
        dst[(u.out_off + o) * oplane + p] = v;
    }
}

//---kernel:dwconv
// Depthwise k×k, stride 1, padding k/2, with bias.
struct DwU { c: u32, h: u32, w: u32, k: u32 }
@group(0) @binding(0) var<uniform> u: DwU;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> wt: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn dwconv(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.w || gid.y >= u.h || gid.z >= u.c) { return; }
    let c = gid.z;
    let plane = u.h * u.w;
    let r = i32(u.k / 2u);
    var acc = bias[c];
    for (var ky = 0u; ky < u.k; ky++) {
        let y = i32(gid.y) + i32(ky) - r;
        if (y < 0 || y >= i32(u.h)) { continue; }
        for (var kx = 0u; kx < u.k; kx++) {
            let x = i32(gid.x) + i32(kx) - r;
            if (x < 0 || x >= i32(u.w)) { continue; }
            acc += src[c * plane + u32(y) * u.w + u32(x)] * wt[(c * u.k + ky) * u.k + kx];
        }
    }
    dst[c * plane + gid.y * u.w + gid.x] = acc;
}

//---kernel:layernorm
// LayerNorm over the channels of each pixel (ConvNeXt's channels-last
// norm), eps 1e-6, with weight and bias.
struct LnU { c: u32, h: u32, w: u32, _p: u32 }
@group(0) @binding(0) var<uniform> u: LnU;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> wt: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn layernorm(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.w || gid.y >= u.h) { return; }
    let plane = u.h * u.w;
    let p = gid.y * u.w + gid.x;
    var mean = 0.0;
    for (var c = 0u; c < u.c; c++) { mean += src[c * plane + p]; }
    mean /= f32(u.c);
    var var_ = 0.0;
    for (var c = 0u; c < u.c; c++) {
        let d = src[c * plane + p] - mean;
        var_ += d * d;
    }
    var_ /= f32(u.c);
    let inv = 1.0 / sqrt(var_ + 1e-6);
    for (var c = 0u; c < u.c; c++) {
        dst[c * plane + p] = (src[c * plane + p] - mean) * inv * wt[c] + bias[c];
    }
}

//---kernel:pool2
// F.interpolate(scale_factor=0.5, bilinear, align_corners=False): the
// average of each 2×2 block, the size floored.
struct PoolU { c: u32, ih: u32, iw: u32, oh: u32, ow: u32, src_off: u32, dst_off: u32, _p: u32 }
@group(0) @binding(0) var<uniform> u: PoolU;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn pool2(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.ow || gid.y >= u.oh || gid.z >= u.c) { return; }
    let s = u.src_off + gid.z * u.ih * u.iw;
    let x = gid.x * 2u;
    let y = gid.y * 2u;
    let v = src[s + y * u.iw + x] + src[s + y * u.iw + x + 1u]
          + src[s + (y + 1u) * u.iw + x] + src[s + (y + 1u) * u.iw + x + 1u];
    dst[u.dst_off + gid.z * u.oh * u.ow + gid.y * u.ow + gid.x] = 0.25 * v;
}

//---kernel:corr
// The correlation lookup, computed on the fly: for each 1/8 pixel and each
// pyramid level, the (2r+1)² neighbourhood around where the flow says the
// pixel went, `dot(f1(p), bilinear(f2_level, q)) / sqrt(C)` with zero
// padding — exactly bilinear sampling of RAFT's all-pairs volume, since
// the dot product is linear, without ever building it. Channel order is
// RAFT's: level-major, then i*(2r+1)+j with the X offset from i.
//
// One thread per (pixel, level, i): its 9 samples share a column pair and
// ten rows, so each channel costs 20 loads for 9 outputs.
struct CorrU {
    c: u32, h: u32, w: u32, r: u32,
    lw0: u32, lh0: u32, lw1: u32, lh1: u32,
    lw2: u32, lh2: u32, lw3: u32, lh3: u32,
    off1: u32, off2: u32, off3: u32, _p: u32,
}
@group(0) @binding(0) var<uniform> u: CorrU;
@group(0) @binding(1) var<storage, read> f1: array<f32>;
@group(0) @binding(2) var<storage, read> f2: array<f32>;
@group(0) @binding(3) var<storage, read> flow: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn corr(@builtin(global_invocation_id) gid: vec3<u32>) {
    // r is 4 in every SEA-RAFT model: the arrays below are sized for it.
    let n = 9u;
    if (gid.x >= u.w || gid.y >= u.h || gid.z >= 4u * n) { return; }
    let lvl = gid.z / n;
    let i = gid.z % n;
    var lw = u.lw0; var lh = u.lh0; var off = 0u;
    if (lvl == 1u) { lw = u.lw1; lh = u.lh1; off = u.off1; }
    if (lvl == 2u) { lw = u.lw2; lh = u.lh2; off = u.off2; }
    if (lvl == 3u) { lw = u.lw3; lh = u.lh3; off = u.off3; }
    let plane = u.h * u.w;
    let p = gid.y * u.w + gid.x;
    let s = 1.0 / f32(1u << lvl);
    let cx = (f32(gid.x) + flow[p]) * s + f32(i32(i) - 4);
    let cy = (f32(gid.y) + flow[plane + p]) * s - 4.0;
    let x0 = floor(cx);
    let y0 = floor(cy);
    let tx = cx - x0;
    let ty = cy - y0;
    let ix = i32(x0);
    let iy = i32(y0);
    // Rows iy..iy+9, columns ix and ix+1; a coordinate outside reads 0.
    let in0 = ix >= 0 && ix < i32(lw);
    let in1 = ix + 1 >= 0 && ix + 1 < i32(lw);
    var acc: array<f32, 9>;
    let lplane = lw * lh;
    for (var c = 0u; c < u.c; c++) {
        let a = f1[c * plane + p];
        let base = off + c * lplane;
        var col: array<f32, 10>;
        for (var k = 0u; k < 10u; k++) {
            let y = iy + i32(k);
            var v = 0.0;
            if (y >= 0 && y < i32(lh)) {
                let row = base + u32(y) * lw;
                if (in0) { v += (1.0 - tx) * f2[row + u32(ix)]; }
                if (in1) { v += tx * f2[row + u32(ix + 1)]; }
            }
            col[k] = v;
        }
        for (var j = 0u; j < 9u; j++) {
            acc[j] += a * ((1.0 - ty) * col[j] + ty * col[j + 1u]);
        }
    }
    let norm = 1.0 / sqrt(f32(u.c));
    for (var j = 0u; j < 9u; j++) {
        dst[(lvl * 81u + i * 9u + j) * plane + p] = acc[j] * norm;
    }
}

//---kernel:flowstep
// flow += update[0..2]; the flow also goes into the context stack's last
// two channels, where the motion encoder's output is concatenated with it.
struct StepU { h: u32, w: u32, x_off: u32, reset: u32 }
@group(0) @binding(0) var<uniform> u: StepU;
@group(0) @binding(1) var<storage, read> upd: array<f32>;
@group(0) @binding(2) var<storage, read_write> flow: array<f32>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn flowstep(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.w || gid.y >= u.h) { return; }
    let plane = u.h * u.w;
    let p = gid.y * u.w + gid.x;
    for (var c = 0u; c < 2u; c++) {
        var f = upd[c * plane + p];
        if (u.reset == 0u) { f += flow[c * plane + p]; }
        flow[c * plane + p] = f;
        x[(u.x_off + c) * plane + p] = f;
    }
}

//---kernel:upsample
// RAFT's convex upsampling: each full-res pixel is a softmax-weighted
// mix of the 3×3 coarse neighbourhood (zero padded), flow ×8. Writes the
// flow (2) and, unscaled, the uncertainty head (4: two mixture weights,
// two log scales).
struct UpU { h: u32, w: u32, _p0: u32, _p1: u32 }
@group(0) @binding(0) var<uniform> u: UpU;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read> flow: array<f32>;
@group(0) @binding(3) var<storage, read> upd: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn upsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let fw = u.w * 8u;
    let fh = u.h * 8u;
    if (gid.x >= fw || gid.y >= fh) { return; }
    let cx = gid.x / 8u;
    let cy = gid.y / 8u;
    let a = gid.y % 8u;
    let b = gid.x % 8u;
    let plane = u.h * u.w;
    let p = cy * u.w + cx;
    var m: array<f32, 9>;
    var mx = -1e30;
    for (var k = 0u; k < 9u; k++) {
        m[k] = 0.25 * mask[(k * 64u + a * 8u + b) * plane + p];
        mx = max(mx, m[k]);
    }
    var sum = 0.0;
    for (var k = 0u; k < 9u; k++) {
        m[k] = exp(m[k] - mx);
        sum += m[k];
    }
    var out = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (var k = 0u; k < 9u; k++) {
        let x = i32(cx) + i32(k % 3u) - 1;
        let y = i32(cy) + i32(k / 3u) - 1;
        if (x < 0 || x >= i32(u.w) || y < 0 || y >= i32(u.h)) { continue; }
        let q = u32(y) * u.w + u32(x);
        let wk = m[k] / sum;
        out[0] += wk * 8.0 * flow[q];
        out[1] += wk * 8.0 * flow[plane + q];
        for (var c = 0u; c < 4u; c++) {
            out[2u + c] += wk * upd[(2u + c) * plane + q];
        }
    }
    let fplane = fw * fh;
    let fp = gid.y * fw + gid.x;
    for (var c = 0u; c < 6u; c++) {
        dst[c * fplane + fp] = out[c];
    }
}

//---kernel:copy
// dst[dst_off + i] = src[i], for n floats.
struct CopyU { dst_off: u32, n: u32, _p0: u32, _p1: u32 }
@group(0) @binding(0) var<uniform> u: CopyU;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn copy(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.n) { return; }
    dst[u.dst_off + gid.x] = src[gid.x];
}
