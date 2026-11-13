//! The kernels, generated per specialisation. Every feature buffer of a
//! tile lives in ONE storage buffer (the arena), bound read-write once, so
//! a dense block can read channels 0..k of a buffer and append channels
//! k..k+32 to it in the same dispatch — two bindings of one buffer, one of
//! them writable, is a usage conflict wgpu refuses.

use std::fmt::Write;

/// What a conv kernel is specialised on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ConvKey {
    pub k: u32,
    /// Output channel groups (vec4s) one thread computes.
    pub ocb: u32,
    pub up: u32,
    pub silu_in: bool,
    /// 0 plain, 1 leaky (slope per channel), 2 gate, 3 silu.
    pub epi: u32,
    /// Residual slices (plain only): 0..=2.
    pub nres: u32,
    pub half: bool,
}

/// Threads per workgroup across and down; each computes a 2x2 quad, so a
/// workgroup covers `TSX x TSY` output pixels.
pub const TX: u32 = 8;
pub const TY: u32 = 8;
pub const TSX: u32 = 2 * TX;
pub const TSY: u32 = 2 * TY;

/// The per-dispatch uniform of a conv, 64 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ConvU {
    pub dw: u32,
    pub dh: u32,
    pub sw: u32,
    pub sh: u32,
    pub src: u32,
    pub dst: u32,
    pub ic4: u32,
    pub w: u32,
    pub b: u32,
    pub oc4: u32,
    pub alpha: f32,
    pub r0: u32,
    pub beta0: f32,
    pub r1: u32,
    pub beta1: f32,
    pub _pad: u32,
}

fn header(half: bool) -> String {
    if half {
        "enable f16;\nalias T = vec4<f16>;\n".into()
    } else {
        "alias T = vec4<f32>;\n".into()
    }
}

/// Clamp into f16's range before the store: an overflow is an inf, and an
/// inf poisons every pixel the next layer's kernel touches.
fn store(half: bool, v: &str) -> String {
    if half {
        format!("T(clamp({v}, vec4f(-65000.0), vec4f(65000.0)))")
    } else {
        format!("T({v})")
    }
}

pub fn conv(key: ConvKey) -> String {
    let ConvKey { k, ocb, up, half, .. } = key;
    let r = k / 2;
    let (psx, psy) = (TSX + 2 * r, TSY + 2 * r);
    let wchunk = k * k * ocb * 4;
    let mut s = header(half);
    write!(
        s,
        r#"
struct U {{
    dw: u32, dh: u32, sw: u32, sh: u32,
    src: u32, dst: u32, ic4: u32, w: u32,
    b: u32, oc4: u32, alpha: f32, r0: u32,
    beta0: f32, r1: u32, beta1: f32, _pad: u32,
}}
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read_write> A: array<T>;
@group(0) @binding(2) var<storage, read> W: array<vec4f>;

const R: i32 = {r};
const PSX: u32 = {psx}u;
const PSY: u32 = {psy}u;
const OCB: u32 = {ocb}u;
const UP: u32 = {up}u;
const WCHUNK: u32 = {wchunk}u;

var<workgroup> tilep: array<vec4f, {pn}>;
var<workgroup> wsh: array<vec4f, {wchunk}>;

fn silu(v: vec4f) -> vec4f {{ return v / (1.0 + exp(-v)); }}

@compute @workgroup_size({TX}, {TY}, 1)
fn main(
    @builtin(local_invocation_id) l: vec3u,
    @builtin(workgroup_id) g: vec3u,
    @builtin(local_invocation_index) li: u32,
) {{
{body}}}
"#,
        pn = psx * psy,
        body = conv_body(key),
    )
    .unwrap();
    s
}

/// The body of `main`. Each thread computes a 2x2 quad of pixels for
/// `OCB` output groups: per input group and tap it reads the quad's four
/// inputs, then walks the output groups with one group's four weight
/// vectors live at a time — everything else in registers is accumulators.
/// Unrolled in Rust, so nothing indexed dynamically has to live in memory.
///
/// Measured on an M4 Pro (`examples/nn_micro`, 48 channels, 512²), against
/// a pure-FMA ceiling through wgpu of ~2.3 TFLOP/s: this shape at OCB 1-3
/// ~1.05; OCB 4 0.68 (register pressure); the same quad holding a whole
/// tap's weights at once 0.11 (it spilled); an 8-pixel row per thread
/// 0.45 (shared-memory bank conflicts: neighbouring threads 8 vec4s apart);
/// an 8-pixel column per thread 0.3-0.66; explicit fma() chains slower than
/// the sum the compiler contracts itself; f16 arithmetic no faster than f32
/// on this GPU, so features are STORED as f16 and computed in f32.
fn conv_body(key: ConvKey) -> String {
    let ConvKey { k, ocb, silu_in, epi, nres, half, .. } = key;
    let r = k / 2;
    let psx = TSX + 2 * r;
    let mut b = String::new();
    let w = |b: &mut String, l: String| {
        b.push_str("    ");
        b.push_str(&l);
        b.push('\n');
    };
    w(&mut b, "let og = g.z;".into());
    w(&mut b, format!("let x0 = i32(g.x * {TSX}u);"));
    w(&mut b, format!("let y0 = i32(g.y * {TSY}u);"));
    w(&mut b, "let splane = u.sw * u.sh;".into());
    for p in 0..4 {
        for o in 0..ocb {
            w(&mut b, format!("var a{p}_{o} = vec4f(0.0);"));
        }
    }
    w(&mut b, "for (var ic = 0u; ic < u.ic4; ic++) {".into());
    w(&mut b, "    let sbase = u.src + ic * splane;".into());
    w(&mut b, "    for (var i = li; i < PSX * PSY; i += 64u) {".into());
    w(&mut b, "        let px = x0 + i32(i % PSX) - R;".into());
    w(&mut b, "        let py = y0 + i32(i / PSX) - R;".into());
    w(&mut b, "        var v = vec4f(0.0);".into());
    w(&mut b, "        if (px >= 0 && py >= 0 && px < i32(u.dw) && py < i32(u.dh)) {".into());
    w(&mut b, "            v = vec4f(A[sbase + (u32(py) / UP) * u.sw + u32(px) / UP]);".into());
    if silu_in {
        w(&mut b, "            v = silu(v);".into());
    }
    w(&mut b, "        }".into());
    w(&mut b, "        tilep[i] = v;".into());
    w(&mut b, "    }".into());
    w(&mut b, "    let wbase = u.w + (og * u.ic4 + ic) * WCHUNK;".into());
    w(&mut b, "    for (var i = li; i < WCHUNK; i += 64u) { wsh[i] = W[wbase + i]; }".into());
    w(&mut b, "    workgroupBarrier();".into());
    w(&mut b, "    let q = l.y * 2u * PSX + l.x * 2u;".into());
    for t in 0..k * k {
        let (ty, tx) = (t / k, t % k);
        w(&mut b, "    {".into());
        for p in 0..4u32 {
            let off = (p / 2 + ty) * psx + p % 2 + tx;
            w(&mut b, format!("    let v{p} = tilep[q + {off}u];"));
        }
        for o in 0..ocb {
            w(&mut b, "    {".into());
            for j in 0..4 {
                w(&mut b, format!("        let w{j} = wsh[{}u];", (t * ocb + o) * 4 + j));
            }
            for p in 0..4u32 {
                w(&mut b, format!("        a{p}_{o} += w0 * v{p}.x + w1 * v{p}.y + w2 * v{p}.z + w3 * v{p}.w;"));
            }
            w(&mut b, "    }".into());
        }
        w(&mut b, "    }".into());
    }
    w(&mut b, "    workgroupBarrier();".into());
    w(&mut b, "}".into());
    w(&mut b, "let dplane = u.dw * u.dh;".into());
    for p in 0..4u32 {
        w(&mut b, "{".into());
        w(&mut b, format!("    let x = u32(x0) + l.x * 2u + {}u;", p % 2));
        w(&mut b, format!("    let y = u32(y0) + l.y * 2u + {}u;", p / 2));
        w(&mut b, "    if (x < u.dw && y < u.dh) {".into());
        w(&mut b, "        let at = y * u.dw + x;".into());
        for o in 0..ocb {
            w(&mut b, "        {".into());
            w(&mut b, format!("            let oc = og * OCB + {o}u;"));
            w(&mut b, format!("            var v = a{p}_{o} + W[u.b + oc];"));
            w(&mut b, format!("            {}", epilogue(epi, nres)));
            w(&mut b, format!("            A[u.dst + oc * dplane + at] = {};", store(half, "v")));
            w(&mut b, "        }".into());
        }
        w(&mut b, "    }".into());
        w(&mut b, "}".into());
    }
    b
}

fn epilogue(epi: u32, nres: u32) -> String {
    if epi == 3 {
        return "v = v / (1.0 + exp(-v));".into();
    }
    if epi == 2 {
        return "let rr = vec4f(A[u.r0 + oc * dplane + at]);\n            \
                v = (v + rr) * (1.0 / (1.0 + exp(-v)) - 0.5);"
            .into();
    }
    let mut s = String::from("v = u.alpha * v;");
    if nres >= 1 {
        s += "\n            v += u.beta0 * vec4f(A[u.r0 + oc * dplane + at]);";
    }
    if nres >= 2 {
        s += "\n            v += u.beta1 * vec4f(A[u.r1 + oc * dplane + at]);";
    }
    if epi == 1 {
        s += "\n            let sl = W[u.b + u.oc4 + oc];\n            v = select(v * sl, v, v >= vec4f(0.0));";
    }
    s
}

/// Copy whole channel groups within the arena.
pub fn copy(half: bool) -> String {
    let mut s = header(half);
    s += r#"
struct C { n: u32, src: u32, dst: u32, _p: u32 }
@group(0) @binding(0) var<uniform> c: C;
@group(0) @binding(1) var<storage, read_write> A: array<T>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let i = gid.x + gid.y * nw.x * 64u;
    if (i >= c.n) { return; }
    A[c.dst + i] = A[c.src + i];
}
"#;
    s
}

/// A 4x4 stride-2 transposed conv (TensorFlow `SAME`): one thread per
/// output pixel and output group. Output `o` takes input `i` through tap
/// `o - 2i + 1`, so each output pixel sees a 2x2 of the input — a pair of
/// rows `(o + 1) / 2` and one above. Weights are laid out
/// `[og][ic4][tap 16][lane 4]`, each a vec4 over the group's four output
/// channels. Plain and simple next to `conv`: HDRCNN's decoder runs five
/// of these against thirteen 3x3 convs of VGG, so it is not the time.
pub fn deconv(key: ConvKey) -> String {
    let mut s = header(key.half);
    write!(
        s,
        r#"
struct U {{
    dw: u32, dh: u32, sw: u32, sh: u32,
    src: u32, dst: u32, ic4: u32, w: u32,
    b: u32, oc4: u32, alpha: f32, r0: u32,
    beta0: f32, r1: u32, beta1: f32, _pad: u32,
}}
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read_write> A: array<T>;
@group(0) @binding(2) var<storage, read> W: array<vec4f>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {{
    let x = gid.x;
    let y = gid.y;
    let oc = gid.z;
    if (x >= u.dw || y >= u.dh) {{ return; }}
    let splane = u.sw * u.sh;
    let iy0 = (y + 1u) / 2u;
    let ix0 = (x + 1u) / 2u;
    let ky0 = y + 1u - 2u * iy0;
    let kx0 = x + 1u - 2u * ix0;
    var acc = vec4f(0.0);
    for (var ic = 0u; ic < u.ic4; ic++) {{
        let sbase = u.src + ic * splane;
        let wbase = u.w + (oc * u.ic4 + ic) * 64u;
        for (var dy = 0u; dy < 2u; dy++) {{
            let iy = i32(iy0) - i32(dy);
            if (iy < 0 || iy >= i32(u.sh)) {{ continue; }}
            let ky = ky0 + 2u * dy;
            for (var dx = 0u; dx < 2u; dx++) {{
                let ix = i32(ix0) - i32(dx);
                if (ix < 0 || ix >= i32(u.sw)) {{ continue; }}
                let kx = kx0 + 2u * dx;
                let v = vec4f(A[sbase + u32(iy) * u.sw + u32(ix)]);
                let wb = wbase + (ky * 4u + kx) * 4u;
                acc += W[wb] * v.x + W[wb + 1u] * v.y + W[wb + 2u] * v.z + W[wb + 3u] * v.w;
            }}
        }}
    }}
    let dplane = u.dw * u.dh;
    let at = y * u.dw + x;
    var v = acc + W[u.b + oc];
    {epi}
    A[u.dst + oc * dplane + at] = {st};
}}
"#,
        epi = epilogue(key.epi, key.nres),
        st = store(key.half, "v"),
    )
    .unwrap();
    s
}

/// The per-dispatch uniform of a pool or a log skip.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MapU {
    /// Destination size (pool), or values to map (log skip, `dw` alone).
    pub dw: u32,
    pub dh: u32,
    pub sw: u32,
    pub sh: u32,
    pub src: u32,
    pub dst: u32,
    pub _p0: u32,
    pub scale: f32,
    pub offset: [f32; 4],
}

/// 2x2 max pool, stride 2, one thread per destination pixel and group.
pub fn pool(half: bool) -> String {
    let mut s = header(half);
    s += r#"
struct M { dw: u32, dh: u32, sw: u32, sh: u32, src: u32, dst: u32, _p0: u32, scale: f32, offset: vec4f }
@group(0) @binding(0) var<uniform> u: M;
@group(0) @binding(1) var<storage, read_write> A: array<T>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= u.dw || gid.y >= u.dh) { return; }
    let sb = u.src + gid.z * u.sw * u.sh + 2u * gid.y * u.sw + 2u * gid.x;
    let v = max(max(A[sb], A[sb + 1u]), max(A[sb + u.sw], A[sb + u.sw + 1u]));
    A[u.dst + gid.z * u.dw * u.dh + gid.y * u.dw + gid.x] = v;
}
"#;
    s
}

/// `ln((v * scale + offset)² + 1/255)` over `dw` vec4s; the offset is per
/// lane. Computed in f32.
pub fn logskip(half: bool) -> String {
    let mut s = header(half);
    s += r#"
struct M { dw: u32, dh: u32, sw: u32, sh: u32, src: u32, dst: u32, _p0: u32, scale: f32, offset: vec4f }
@group(0) @binding(0) var<uniform> u: M;
@group(0) @binding(1) var<storage, read_write> A: array<T>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let i = gid.x + gid.y * nw.x * 64u;
    if (i >= u.dw) { return; }
    let v = vec4f(A[u.src + i]) * u.scale + u.offset;
    A[u.dst + i] = T(log(v * v + vec4f(1.0 / 255.0)));
}
"#;
    s
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CopyU {
    pub n: u32,
    pub src: u32,
    pub dst: u32,
    pub _p: u32,
}

/// The tile's way in and out share their colour maths: working -> linear
/// Rec.709 (`m`) and the sRGB curve. The models were trained on
/// display-referred 0..1 pictures, so that is what they are given — the
/// picture CLAMPED to 0..1 — and what lies outside (highlights past 1,
/// out-of-gamut negatives) is carried around them: upscaled bilinearly and
/// added back. A bounded tonemap with an exact inverse was the first
/// design, and is wrong: the model overshoots near the top of its range as
/// a matter of course (sharpening), and the inverse of anything that maps
/// [k, inf) into [k, 1) turns that overshoot into an unbounded value —
/// measured, an input of 0.87 came back as 400.
const COLOUR: &str = r#"
fn oetf(v: vec3f) -> vec3f {
    return select(12.92 * v, 1.055 * pow(v, vec3f(1.0 / 2.4)) - 0.055, v > vec3f(0.0031308));
}
fn eotf(v: vec3f) -> vec3f {
    return select(v / 12.92, pow((v + 0.055) / 1.055, vec3f(2.4)), v > vec3f(0.04045));
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LoadU {
    pub ox: u32,
    pub oy: u32,
    pub tw: u32,
    pub th: u32,
    pub dst: u32,
    /// 0 raw, 1 sRGB (an upscaler), 2 gamma 2 over the clip level (HDRCNN).
    pub encode: u32,
    pub clip: f32,
    pub range: f32,
    pub mean: [f32; 4],
    /// Working -> linear Rec.709, rows.
    pub m: [[f32; 4]; 3],
    /// The picture's size: reads past it repeat the edge (a net cut on a
    /// grid reads past the picture into its padding).
    pub lim: [u32; 4],
}

pub fn load(half: bool) -> String {
    let mut s = header(half);
    s += COLOUR;
    s += r#"
struct L {
    ox: u32, oy: u32, tw: u32, th: u32,
    dst: u32, encode: u32, clip: f32, range: f32,
    mean: vec4f, m0: vec4f, m1: vec4f, m2: vec4f,
    lim: vec4u,
}
@group(0) @binding(0) var<uniform> u: L;
@group(0) @binding(1) var<storage, read_write> A: array<T>;
@group(0) @binding(2) var src: texture_2d<f32>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= u.tw || gid.y >= u.th) { return; }
    let c = textureLoad(src, min(vec2u(u.ox + gid.x, u.oy + gid.y), u.lim.xy - 1u), 0);
    var rgb = c.rgb;
    if (u.encode != 0u) {
        // Straight colour: the model never saw premultiplied edges.
        if (c.a > 1e-5) { rgb = rgb / c.a; }
        let lin = vec3f(dot(u.m0.xyz, rgb), dot(u.m1.xyz, rgb), dot(u.m2.xyz, rgb));
        if (u.encode == 2u) {
            rgb = sqrt(clamp(lin / u.clip, vec3f(0.0), vec3f(1.0)));
        } else {
            rgb = oetf(clamp(lin, vec3f(0.0), vec3f(1.0)));
        }
    }
    rgb = (rgb - u.mean.xyz) * u.range;
    A[u.dst + gid.y * u.tw + gid.x] = T(vec4f(rgb, c.a));
}
"#;
    s
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EmitU {
    /// Tile origin and size, input pixels.
    pub ox: u32,
    pub oy: u32,
    pub tw: u32,
    pub th: u32,
    /// The region this tile writes, output pixels.
    pub vx: u32,
    pub vy: u32,
    pub vw: u32,
    pub vh: u32,
    /// The net's output slice, and its input's (for the base).
    pub obase: u32,
    pub ibase: u32,
    /// bit 0 shuffle, bit 1 base, bit 2 colour, bit 3 half (2x) output,
    /// bit 4 HDRCNN's blend.
    pub mode: u32,
    pub clip: f32,
    pub in_w: u32,
    pub in_h: u32,
    /// The net's own scale.
    pub s: u32,
    pub _p: u32,
    /// Linear Rec.709 -> working, rows.
    pub mi: [[f32; 4]; 3],
    /// Working -> linear Rec.709, rows (for the excess).
    pub m: [[f32; 4]; 3],
}

pub fn emit(half: bool) -> String {
    let mut s = header(half);
    s += COLOUR;
    s += r#"
struct E {
    ox: u32, oy: u32, tw: u32, th: u32,
    vx: u32, vy: u32, vw: u32, vh: u32,
    obase: u32, ibase: u32, mode: u32, clip: f32,
    in_w: u32, in_h: u32, s: u32, _p: u32,
    mi0: vec4f, mi1: vec4f, mi2: vec4f,
    m0: vec4f, m1: vec4f, m2: vec4f,
}
@group(0) @binding(0) var<uniform> u: E;
@group(0) @binding(1) var<storage, read_write> A: array<T>;
@group(0) @binding(2) var src: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var dst: texture_storage_2d<rgba16float, write>;

// The stage input, bilinear, at an input-pixel position held inside the
// picture (the input texture is canvas-sized): premultiplied rgba.
fn input_at(p: vec2f) -> vec4f {
    let q = clamp(p, vec2f(0.5), vec2f(f32(u.in_w), f32(u.in_h)) - 0.5);
    return textureSampleLevel(src, samp, q / vec2f(textureDimensions(src)), 0.0);
}

// The net's answer at a global output-grid pixel (at the net's scale), as
// straight working-space colour.
fn net_at(gx: u32, gy: u32) -> vec3f {
    let s = u.s;
    let x = min(gx, u.in_w * s - 1u) - u.ox * s;
    let y = min(gy, u.in_h * s - 1u) - u.oy * s;
    var v: vec3f;
    if ((u.mode & 1u) != 0u) {
        // Pixel shuffle: channel c*s*s + (y%s)*s + (x%s) at (x/s, y/s).
        let plane = u.tw * u.th;
        let at = (y / s) * u.tw + x / s;
        for (var c = 0u; c < 3u; c++) {
            let ch = c * s * s + (y % s) * s + (x % s);
            v[c] = f32(A[u.obase + (ch / 4u) * plane + at][ch % 4u]);
        }
    } else {
        v = vec3f(A[u.obase + y * (u.tw * s) + x].xyz);
    }
    if ((u.mode & 2u) != 0u) {
        v += vec3f(A[u.ibase + (y / s) * u.tw + x / s].xyz);
    }
    if ((u.mode & 4u) != 0u) {
        var lin = eotf(clamp(v, vec3f(0.0), vec3f(1.0)));
        // What the model never saw: the input outside 0..1, bilinear.
        let c = input_at((vec2f(f32(gx), f32(gy)) + 0.5) / f32(s));
        var rgb = c.rgb;
        if (c.a > 1e-5) { rgb = rgb / c.a; }
        let li = vec3f(dot(u.m0.xyz, rgb), dot(u.m1.xyz, rgb), dot(u.m2.xyz, rgb));
        lin += li - clamp(li, vec3f(0.0), vec3f(1.0));
        v = vec3f(dot(u.mi0.xyz, lin), dot(u.mi1.xyz, lin), dot(u.mi2.xyz, lin));
    }
    return v;
}

fn alpha_at(p: vec2f) -> f32 {
    return input_at(p).a;
}

// HDRCNN: the net's log prediction at a picture pixel, and the method's
// blend with the input it was given. Where the mask is 0 the input texel
// is stored AS IT CAME.
fn hdr(X: u32, Y: u32) {
    let c = textureLoad(src, vec2u(X, Y), 0);
    let y = vec3f(A[u.obase + (Y - u.oy) * u.tw + (X - u.ox)].xyz);
    if ((u.mode & 4u) == 0u) {
        textureStore(dst, vec2u(X, Y), vec4f(y, 1.0));
        return;
    }
    var rgb = c.rgb;
    if (c.a > 1e-5) { rgb = rgb / c.a; }
    let lin = vec3f(dot(u.m0.xyz, rgb), dot(u.m1.xyz, rgb), dot(u.m2.xyz, rgb));
    let x = sqrt(clamp(lin / u.clip, vec3f(0.0), vec3f(1.0)));
    let a = clamp((max(x.r, max(x.g, x.b)) - 1.0 + 0.05) / 0.05, 0.0, 1.0);
    if (a <= 0.0) {
        textureStore(dst, vec2u(X, Y), c);
        return;
    }
    let o = mix(lin, u.clip * (exp(y) - 1.0 / 255.0), a);
    let w = vec3f(dot(u.mi0.xyz, o), dot(u.mi1.xyz, o), dot(u.mi2.xyz, o));
    textureStore(dst, vec2u(X, Y), vec4f(w * c.a, c.a));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= u.vw || gid.y >= u.vh) { return; }
    let X = u.vx + gid.x;
    let Y = u.vy + gid.y;
    if ((u.mode & 16u) != 0u) {
        hdr(X, Y);
        return;
    }
    let fs = f32(u.s);
    var out: vec4f;
    if ((u.mode & 8u) != 0u) {
        // Half the net's scale: a [1 3 3 1] decimation of its grid,
        // premultiplied, centred between the pixel pair it halves.
        let w = array<f32, 4>(0.125, 0.375, 0.375, 0.125);
        var acc = vec4f(0.0);
        for (var j = 0u; j < 4u; j++) {
            let gy = u32(max(i32(2u * Y + j) - 1, 0));
            for (var i = 0u; i < 4u; i++) {
                let gx = u32(max(i32(2u * X + i) - 1, 0));
                let a = alpha_at((vec2f(f32(gx), f32(gy)) + 0.5) / fs);
                acc += w[i] * w[j] * vec4f(net_at(gx, gy) * a, a);
            }
        }
        out = acc;
    } else {
        let a = alpha_at((vec2f(f32(X), f32(Y)) + 0.5) / fs);
        out = vec4f(net_at(X, Y) * a, a);
    }
    textureStore(dst, vec2u(X, Y), out);
}
"#;
    s
}
