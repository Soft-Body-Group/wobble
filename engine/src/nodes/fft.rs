//! A 2D FFT convolution as stage passes: what lets a Convolve take a kernel
//! IMAGE (a star, a bokeh, a diffraction pattern hundreds of pixels wide)
//! at O(n log n) instead of the O(n k^2) a gather would cost.
//!
//! The picture and the kernel are laid into power-of-two grids
//! (`grid`): the picture at the top-left, zero-padded to at least twice its
//! size so the circular convolution IS the linear one, the kernel with its
//! centre at the origin and its negative offsets wrapped to the far edge.
//! Both are transformed, multiplied and transformed back.
//!
//! Each 1D transform is ONE dispatch: a workgroup takes whole rows (or
//! columns) through registers and shared memory and runs every radix-8
//! pass there, so a 2D transform is two passes over the grid rather than
//! two per stage. A grid texel is `rgba32float` holding two complex lanes
//! (re, im, re, im), and each lane carries TWO real channels: r + i g and
//! b + i a. A real signal's spectrum is conjugate-symmetric, so the
//! multiply pass separates the two from one transform (`X_r = (Z[k] +
//! conj Z[-k]) / 2`), multiplies each by its own kernel and packs them
//! back — rgba is one texture and ONE 2D transform each way, six FFT
//! passes a frame where the lane-per-channel layout took twelve. A
//! workgroup holds [`MAX_N`] values, which caps a side; the grid steps
//! down in resolution to fit (scale down, never refuse).
//!
//! The textures are read as READ-ONLY STORAGE, not sampled: a float32
//! texture is not filterable everywhere, and a pass that only loads must
//! not depend on the adapter offering that.

/// The longest transform a workgroup holds in shared memory.
pub const MAX_N: u32 = 2048;

/// The grid for a canvas at `1 / 2^shift`: (shift, nx, ny), the shift
/// raised until both sides fit [`MAX_N`]. Each side is at least twice the
/// picture's, so nothing wraps.
pub fn grid(canvas: (u32, u32), shift: u32) -> (u32, u32, u32) {
    let mut s = shift;
    loop {
        let w = canvas.0.div_ceil(1 << s).max(1);
        let h = canvas.1.div_ceil(1 << s).max(1);
        let nx = (2 * w).next_power_of_two().max(16);
        let ny = (2 * h).next_power_of_two().max(16);
        if (nx <= MAX_N && ny <= MAX_N) || s >= 8 {
            return (s, nx.min(MAX_N), ny.min(MAX_N));
        }
        s += 1;
    }
}

/// Lines of a grid one workgroup transforms: the 256 threads hold eight
/// values each, so a workgroup covers [`MAX_N`] values — one 2048 line,
/// or 2048 / n shorter ones.
pub fn lines_per_group(n: u32) -> u32 {
    (MAX_N / n.max(16)).max(1)
}

/// Workgroups for one 1D pass over an nx x ny grid (rows or columns: the
/// same count either way).
pub fn groups(nx: u32, ny: u32) -> u32 {
    (nx * ny).div_ceil(MAX_N)
}

/// 1D transforms, rows or columns, both lanes of a texel at once.
/// Uniform: (n, log2 n, axis 0 = rows / 1 = columns, sign -1 forward / +1
/// inverse — unscaled; the output pass divides by the grid's area).
///
/// A Stockham radix-8 transform (the last pass radix 4 or 2 where log2 n
/// is not a multiple of 3), natural order in and out: each thread keeps
/// its eight values in REGISTERS and the passes exchange through shared
/// memory one float component at a time. The exchange buffer is 8 KiB on
/// purpose: on Apple GPUs a workgroup holding 16 KiB ran the same copy
/// ten times slower (occupancy), and the old in-place radix-2 kept 16.
/// `fft_micro` measures it.
pub const FFT_WGSL: &str = r#"
struct FU { n: f32, logn: f32, axis: f32, sign: f32 }
@group(0) @binding(0) var<uniform> U: FU;
@group(0) @binding(1) var src: texture_storage_2d<rgba32float, read>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba32float, write>;

const PI: f32 = 3.14159265358979;
const H: f32 = 0.70710678118654752;
var<workgroup> sh: array<f32, 2048>;

fn at(line: u32, i: u32) -> vec2u {
    if (U.axis == 0.0) { return vec2u(i, line); }
    return vec2u(line, i);
}

fn cm(a: vec2f, b: vec2f) -> vec2f {
    return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Both lanes times one complex number.
fn cm2(a: vec4f, w: vec2f) -> vec4f {
    return vec4f(cm(a.xy, w), cm(a.zw, w));
}

// Both lanes times s * i.
fn rot(a: vec4f, s: f32) -> vec4f {
    return vec4f(-s * a.y, s * a.x, -s * a.w, s * a.z);
}

// log2 of the radix of pass p: 3, then what log2 n leaves.
fn lradix(p: u32, logn: u32) -> u32 {
    let full = logn / 3u;
    if (p < full) { return 3u; }
    return logn - 3u * full;
}

// Register e holds input k of the radix-r butterfly of virtual thread j
// (all powers of two: shifts and masks, never a division).
fn in_idx(e: u32, t: u32, lper: u32, lr: u32, logn: u32) -> u32 {
    let j = t + ((e >> lr) << lper);
    return j + ((e & ((1u << lr) - 1u)) << (logn - lr));
}

fn out_idx(e: u32, t: u32, lper: u32, lr: u32, lns: u32) -> u32 {
    let j = t + ((e >> lr) << lper);
    let m = (1u << lns) - 1u;
    return ((j >> lns) << (lns + lr)) + (j & m) + ((e & ((1u << lr) - 1u)) << lns);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) tid: u32) {
    let logn = u32(U.logn);
    let n = 1u << logn;
    let s = U.sign;
    let lper = logn - 3u;
    let ll = tid >> lper;
    let t = tid & ((1u << lper) - 1u);
    let line = (wg.x << (8u - lper)) + ll;
    let dims = textureDimensions(src);
    let lines = select(dims.x, dims.y, U.axis == 0.0);
    let live = line < lines;
    let base = ll << logn;
    let passes = (logn + 2u) / 3u;
    var v: array<vec4f, 8>;
    var lr = lradix(0u, logn);
    for (var e = 0u; e < 8u; e++) {
        v[e] = vec4f(0.0);
        if (live) { v[e] = textureLoad(src, at(line, in_idx(e, t, lper, lr, logn))); }
    }
    var lns = 0u;
    for (var p = 0u; p < passes; p++) {
        lr = lradix(p, logn);
        let r = 1u << lr;
        let ns = 1u << lns;
        for (var e = 0u; e < 8u; e++) {
            let k = e & (r - 1u);
            if (k != 0u) {
                let j = t + ((e >> lr) << lper);
                let a = s * 2.0 * PI * f32((j & (ns - 1u)) * k) / f32(ns << lr);
                v[e] = cm2(v[e], vec2f(cos(a), sin(a)));
            }
        }
        if (r == 8u) {
            let a0 = v[0] + v[4]; let b0 = v[0] - v[4];
            let c0 = v[2] + v[6]; let d0 = rot(v[2] - v[6], s);
            let a1 = v[1] + v[5]; let b1 = v[1] - v[5];
            let c1 = v[3] + v[7]; let d1 = rot(v[3] - v[7], s);
            let e0 = a0 + c0; let e1 = b0 + d0; let e2 = a0 - c0; let e3 = b0 - d0;
            let o0 = a1 + c1;
            let o1 = cm2(b1 + d1, vec2f(H, s * H));
            let o2 = rot(a1 - c1, s);
            let o3 = cm2(b1 - d1, vec2f(-H, s * H));
            v[0] = e0 + o0; v[4] = e0 - o0;
            v[1] = e1 + o1; v[5] = e1 - o1;
            v[2] = e2 + o2; v[6] = e2 - o2;
            v[3] = e3 + o3; v[7] = e3 - o3;
        } else if (r == 4u) {
            for (var g = 0u; g < 8u; g += 4u) {
                let a = v[g] + v[g + 2u]; let b = v[g] - v[g + 2u];
                let c = v[g + 1u] + v[g + 3u]; let d = rot(v[g + 1u] - v[g + 3u], s);
                v[g] = a + c; v[g + 1u] = b + d; v[g + 2u] = a - c; v[g + 3u] = b - d;
            }
        } else {
            for (var g = 0u; g < 8u; g += 2u) {
                let a = v[g]; let b = v[g + 1u];
                v[g] = a + b; v[g + 1u] = a - b;
            }
        }
        var oi: array<u32, 8>;
        for (var e = 0u; e < 8u; e++) { oi[e] = out_idx(e, t, lper, lr, lns); }
        if (p + 1u == passes) {
            for (var e = 0u; e < 8u; e++) {
                if (live) { textureStore(dst, at(line, oi[e]), v[e]); }
            }
        } else {
            let lrn = lradix(p + 1u, logn);
            var ii: array<u32, 8>;
            for (var e = 0u; e < 8u; e++) { ii[e] = base + in_idx(e, t, lper, lrn, logn); }
            for (var e = 0u; e < 8u; e++) { sh[base + oi[e]] = v[e].x; }
            workgroupBarrier();
            for (var e = 0u; e < 8u; e++) { v[e].x = sh[ii[e]]; }
            workgroupBarrier();
            for (var e = 0u; e < 8u; e++) { sh[base + oi[e]] = v[e].y; }
            workgroupBarrier();
            for (var e = 0u; e < 8u; e++) { v[e].y = sh[ii[e]]; }
            workgroupBarrier();
            for (var e = 0u; e < 8u; e++) { sh[base + oi[e]] = v[e].z; }
            workgroupBarrier();
            for (var e = 0u; e < 8u; e++) { v[e].z = sh[ii[e]]; }
            workgroupBarrier();
            for (var e = 0u; e < 8u; e++) { sh[base + oi[e]] = v[e].w; }
            workgroupBarrier();
            for (var e = 0u; e < 8u; e++) { v[e].w = sh[ii[e]]; }
            workgroupBarrier();
        }
        lns += lr;
    }
}
"#;

/// The picture into its grid: the mean of each 2^shift block, top-left,
/// zero beyond it, as (r, g, b, a) = r + i g and b + i a. Uniform: (block,
/// picture w, picture h, _).
pub const LOAD_IMAGE_WGSL: &str = r#"
struct LU { block: f32, w: f32, h: f32, _p: f32 }
@group(0) @binding(0) var<uniform> U: LU;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var grid: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(grid);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    var c = vec4f(0.0);
    if (f32(gid.x) < U.w && f32(gid.y) < U.h) {
        let b = i32(U.block);
        let sd = vec2i(textureDimensions(src)) - 1;
        let o = vec2i(gid.xy) * b;
        for (var j = 0; j < b; j++) {
            for (var i = 0; i < b; i++) {
                c += textureLoad(src, min(o + vec2i(i, j), sd), 0);
            }
        }
        c /= f32(b * b);
    }
    textureStore(grid, gid.xy, c);
}
"#;

/// The kernel into its grid, centred on the origin: grid offset `d` (the
/// upper half of each axis wrapped negative) reads the 2^shift block of
/// the kernel picture around its centre pixel + d * block, SUMMED, so a
/// kernel's weights stay per output pixel at any grid resolution. Alpha's
/// kernel is the mean of rgb's. Packed like the picture: r + i g, b + i a.
/// Uniform: (block, nx, ny, _), (the kernel's centre pixel, _, _) — the
/// middle of its FORMAT, which sits at the top-left of the canvas like any
/// picture.
pub const LOAD_KERNEL_WGSL: &str = r#"
struct LU { block: f32, nx: f32, ny: f32, _p: f32, cx: f32, cy: f32, _p1: f32, _p2: f32 }
@group(0) @binding(0) var<uniform> U: LU;
@group(0) @binding(1) var kern: texture_2d<f32>;
@group(0) @binding(2) var grid: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(grid);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let n = vec2i(i32(U.nx), i32(U.ny));
    var d = vec2i(gid.xy);
    if (d.x >= n.x / 2) { d.x -= n.x; }
    if (d.y >= n.y / 2) { d.y -= n.y; }
    let b = i32(U.block);
    let kd = vec2i(textureDimensions(kern));
    let o = vec2i(i32(U.cx), i32(U.cy)) + d * b - vec2i(b / 2);
    var c = vec3f(0.0);
    for (var j = 0; j < b; j++) {
        for (var i = 0; i < b; i++) {
            let q = o + vec2i(i, j);
            if (all(q >= vec2i(0)) && all(q < kd)) {
                c += textureLoad(kern, q, 0).rgb;
            }
        }
    }
    let a = (c.r + c.g + c.b) / 3.0;
    textureStore(grid, gid.xy, vec4f(c, a));
}
"#;

/// Spectrum times spectrum, channel by channel. Each lane holds two real
/// channels (`p + i q`), whose spectra come apart by conjugate symmetry:
/// `P[k] = (Z[k] + conj Z[-k]) / 2`, `Q[k] = (Z[k] - conj Z[-k]) / 2i`.
/// The products pack back the same way, `P K_p + i Q K_q`, so one
/// inverse transform returns both channels real. With `normalize`,
/// divided by the kernel's DC — its sum — averaged over r, g and b, so a
/// tinted kernel keeps its tint and the picture keeps its level. That mean
/// is exactly the alpha channel's DC (alpha's kernel IS the mean of rgb):
/// the imaginary part of the second lane at (0, 0). Uniform: (normalize,
/// nx, ny, _).
pub const MULTIPLY_WGSL: &str = r#"
struct MU { normalize: f32, nx: f32, ny: f32, _p: f32 }
@group(0) @binding(0) var<uniform> U: MU;
@group(0) @binding(1) var f: texture_storage_2d<rgba32float, read>;
@group(0) @binding(2) var k: texture_storage_2d<rgba32float, read>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba32float, write>;

fn cmul(a: vec2f, b: vec2f) -> vec2f {
    return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// The two channels' spectra of a packed lane: z at k, zm at -k.
fn split(z: vec2f, zm: vec2f) -> vec4f {
    let c = vec2f(zm.x, -zm.y);
    let p = (z + c) * 0.5;
    let d = (z - c) * 0.5;
    return vec4f(p, d.y, -d.x);
}

// Both channels' products, packed back as p + i q.
fn lane(z: vec2f, zm: vec2f, w: vec2f, wm: vec2f) -> vec2f {
    let x = split(z, zm);
    let y = split(w, wm);
    let p = cmul(x.xy, y.xy);
    let q = cmul(x.zw, y.zw);
    return p + vec2f(-q.y, q.x);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(dst);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let n = vec2u(u32(U.nx), u32(U.ny));
    let m = (n - gid.xy) % n;
    let a = textureLoad(f, gid.xy);
    let am = textureLoad(f, m);
    let b = textureLoad(k, gid.xy);
    let bm = textureLoad(k, m);
    var s = 1.0;
    if (U.normalize != 0.0) {
        let dc = textureLoad(k, vec2u(0u)).w;
        s = select(1.0 / dc, 1.0, abs(dc) < 1e-8);
    }
    textureStore(dst, gid.xy, vec4f(lane(a.xy, am.xy, b.xy, bm.xy), lane(a.zw, am.zw, b.zw, bm.zw)) * s);
}
"#;

macro_rules! out_body {
    () => {
        r#"
// The grid back onto the canvas: bilinear between grid texels (centres at
// (q + 0.5) * block), the texel being (r, g, b, a) as the lanes unpack,
// scaled by 1 / grid area — the inverse transform is unscaled.
fn grid_at(p: vec2i) -> vec4f {
    let b = U.block;
    let q = (vec2f(p) + 0.5) / b - 0.5;
    let q0 = vec2i(floor(q));
    let f = q - vec2f(q0);
    let hi = vec2i(i32(U.w) - 1, i32(U.h) - 1);
    var acc = vec4f(0.0);
    for (var j = 0; j < 2; j++) {
        for (var i = 0; i < 2; i++) {
            let t = vec2u(clamp(q0 + vec2i(i, j), vec2i(0), hi));
            let w = select(1.0 - f.x, f.x, i == 1) * select(1.0 - f.y, f.y, j == 1);
            acc += w * textureLoad(grid, t);
        }
    }
    return acc * U.inv_area;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = vec2i(textureDimensions(dst));
    if (gid.x >= u32(dims.x) || gid.y >= u32(dims.y)) { return; }
    let p = vec2i(gid.xy);
    let c = textureLoad(src, p, 0);
    var v = grid_at(p);
    if (U.alpha == 0.0) { v.a = c.a; }
    textureStore(dst, gid.xy, mix(c, v, amount(p)));
}
"#
    };
}

/// Uniform for the output pass: (block, grid w, grid h, 1 / area), (mix,
/// alpha, _, _).
macro_rules! out_head {
    () => {
        r#"struct OU { block: f32, w: f32, h: f32, inv_area: f32, mix: f32, alpha: f32, _p0: f32, _p1: f32 }
@group(0) @binding(0) var<uniform> U: OU;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var grid: texture_storage_2d<rgba32float, read>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;
"#
    };
}

pub const OUT_WGSL: &str = concat!(
    out_head!(),
    "fn amount(p: vec2i) -> f32 { return U.mix; }\n",
    out_body!()
);

/// [`OUT_WGSL`] with a modulate matte scaling the mix per pixel.
pub const OUT_MOD_WGSL: &str = concat!(
    out_head!(),
    "@group(0) @binding(4) var modt: texture_2d<f32>;\n",
    "fn amount(p: vec2i) -> f32 { return U.mix * clamp(textureLoad(modt, p, 0).x, 0.0, 1.0); }\n",
    out_body!()
);

#[cfg(test)]
mod tests {
    use super::*;

    type C = (f64, f64);
    fn cm(a: C, b: C) -> C {
        (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
    }
    fn add(a: C, b: C) -> C {
        (a.0 + b.0, a.1 + b.1)
    }
    fn sub(a: C, b: C) -> C {
        (a.0 - b.0, a.1 - b.1)
    }
    fn rot(a: C, s: f64) -> C {
        (-s * a.1, s * a.0)
    }

    /// The shader's scheme on the CPU, register for register: Stockham,
    /// radix 8 then what is left, n / 8 threads of eight values.
    fn fft(x: &[C], sign: f64) -> Vec<C> {
        let n = x.len();
        let logn = n.trailing_zeros() as usize;
        let radix = |p: usize| if p < logn / 3 { 8 } else { 1 << (logn - 3 * (logn / 3)) };
        let per = n / 8;
        let in_idx = |e: usize, t: usize, r: usize| t + (e / r) * per + (e % r) * (n / r);
        let out_idx = |e: usize, t: usize, r: usize, ns: usize| {
            let j = t + (e / r) * per;
            (j / ns) * ns * r + j % ns + (e % r) * ns
        };
        let passes = logn.div_ceil(3);
        let mut regs: Vec<[C; 8]> =
            (0..per).map(|t| std::array::from_fn(|e| x[in_idx(e, t, radix(0))])).collect();
        let mut ns = 1;
        let mut out = vec![(0.0, 0.0); n];
        let h = std::f64::consts::FRAC_1_SQRT_2;
        let s = sign;
        for p in 0..passes {
            let r = radix(p);
            for (t, v) in regs.iter_mut().enumerate() {
                for (e, ve) in v.iter_mut().enumerate() {
                    let k = e % r;
                    if k != 0 {
                        let j = t + (e / r) * per;
                        let a = s * 2.0 * std::f64::consts::PI * ((j % ns) * k) as f64 / (ns * r) as f64;
                        *ve = cm(*ve, (a.cos(), a.sin()));
                    }
                }
                if r == 8 {
                    let (a0, b0) = (add(v[0], v[4]), sub(v[0], v[4]));
                    let (c0, d0) = (add(v[2], v[6]), rot(sub(v[2], v[6]), s));
                    let (a1, b1) = (add(v[1], v[5]), sub(v[1], v[5]));
                    let (c1, d1) = (add(v[3], v[7]), rot(sub(v[3], v[7]), s));
                    let e = [add(a0, c0), add(b0, d0), sub(a0, c0), sub(b0, d0)];
                    let o = [
                        add(a1, c1),
                        cm(add(b1, d1), (h, s * h)),
                        rot(sub(a1, c1), s),
                        cm(sub(b1, d1), (-h, s * h)),
                    ];
                    for k in 0..4 {
                        v[k] = add(e[k], o[k]);
                        v[k + 4] = sub(e[k], o[k]);
                    }
                } else if r == 4 {
                    for g in [0, 4] {
                        let (a, b) = (add(v[g], v[g + 2]), sub(v[g], v[g + 2]));
                        let (c, d) = (add(v[g + 1], v[g + 3]), rot(sub(v[g + 1], v[g + 3]), s));
                        v[g] = add(a, c);
                        v[g + 1] = add(b, d);
                        v[g + 2] = sub(a, c);
                        v[g + 3] = sub(b, d);
                    }
                } else {
                    for g in [0, 2, 4, 6] {
                        let (a, b) = (v[g], v[g + 1]);
                        v[g] = add(a, b);
                        v[g + 1] = sub(a, b);
                    }
                }
                for (e, ve) in v.iter().enumerate() {
                    out[out_idx(e, t, r, ns)] = *ve;
                }
            }
            ns *= r;
            if p + 1 < passes {
                let rn = radix(p + 1);
                for (t, v) in regs.iter_mut().enumerate() {
                    *v = std::array::from_fn(|e| out[in_idx(e, t, rn)]);
                }
            }
        }
        out
    }

    #[test]
    fn the_passes_are_a_dft() {
        for n in [16usize, 32, 64, 128, 256, 512, 1024, 2048] {
            let x: Vec<C> = (0..n).map(|i| ((i as f64 * 0.37).sin(), (i as f64 * 1.3).cos())).collect();
            for sign in [-1.0, 1.0] {
                let got = fft(&x, sign);
                for (k, g) in got.iter().enumerate() {
                    let mut want = (0.0, 0.0);
                    for (j, v) in x.iter().enumerate() {
                        let a = sign * 2.0 * std::f64::consts::PI * ((j * k) % n) as f64 / n as f64;
                        want = add(want, cm(*v, (a.cos(), a.sin())));
                    }
                    let tol = 1e-9 * n as f64;
                    assert!((g.0 - want.0).abs() < tol && (g.1 - want.1).abs() < tol, "n {n} sign {sign} bin {k}");
                }
            }
        }
    }

    #[test]
    fn a_group_covers_whole_lines() {
        for n in [16u32, 64, 512, 2048] {
            assert_eq!(lines_per_group(n) * n, MAX_N);
            assert_eq!(groups(n, 4096 / n) * MAX_N, n * (4096 / n));
        }
    }

    #[test]
    fn the_grid_never_wraps_and_fits_shared_memory() {
        for canvas in [(1920, 1080), (2048, 1080), (4096, 2160), (256, 256), (7, 3)] {
            for shift in 0..3 {
                let (s, nx, ny) = grid(canvas, shift);
                assert!(s >= shift && nx <= MAX_N && ny <= MAX_N);
                assert!(nx >= 2 * canvas.0.div_ceil(1 << s), "{canvas:?} at {s}: {nx}");
                assert!(ny >= 2 * canvas.1.div_ceil(1 << s), "{canvas:?} at {s}: {ny}");
            }
        }
        assert_eq!(grid((1920, 1080), 0).0, 1, "2K steps down to half to fit");
    }
}
