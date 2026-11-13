//! LUT: a 1D and/or 3D lookup table from a file, applied to the picture.
//!
//! The comp is scene-linear and nearly every LUT is not: a "Rec.709 look"
//! cube wants sRGB-ish code values in 0..1, a camera LUT wants LogC or
//! S-Log3. So the node SHAPES the picture into the space the LUT expects
//! ([`LutParams::input`], or a log2 allocation over a stops range when the
//! LUT was built for one), looks it up, and brings the result back
//! ([`LutOut`]: decode it the same way, read it as another space, or leave
//! the LUT's numbers as they are). Values the shaping puts outside the
//! table's domain clamp, as they do in OCIO.
//!
//! The table cannot ride the 16 KB uniform block, so it is a per-node
//! TEXTURE (`SourceBinding::Lut`), laid out by [`crate::lut::Layout`] and
//! uploaded by the compositor when the file's bytes change. Its shape (1D,
//! 3D, both, sizes, domain) is UNIFORMS: picking another file never
//! recompiles. Only the shaping is topology — the input space, the shaper
//! kind and the output pick — because those are colour transforms written
//! into the WGSL. A path that does not load is a passthrough and the panel
//! says why ([`LutParams::status`]).

use crate::codegen::Emit;
use crate::color::{Colorspace, Curve, Transform};
use crate::invview::InverseView;
use crate::lut::{Interp, Lut};
use crate::nodes::NodeDef;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

const TOPO_BASE: u64 = 0x1A7;

/// How the picture is put into the LUT's domain. Structural.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LutShaper {
    /// Encode with the input space's own curve (sRGB, LogC3, …).
    #[default]
    Curve,
    /// The input space's primaries, then log2 about 18% grey over
    /// `log_min..log_max` stops into 0..1 — what a LUT baked for a log2
    /// allocation (Blender's Filmic and AgX, OCIO `lg2` shapers) expects.
    Log2,
}

impl LutShaper {
    pub const ALL: [Self; 2] = [Self::Curve, Self::Log2];
    pub fn label(self) -> &'static str {
        match self {
            Self::Curve => "the space's curve",
            Self::Log2 => "log2 over stops",
        }
    }
}

/// What the LUT's output is. Structural.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LutOut {
    /// The same space it went in as: undo the shaping, so a look LUT
    /// stays a look in the comp's scene-linear working space.
    #[default]
    Same,
    /// Another space (a camera-to-display LUT's output, say), decoded into
    /// the working space.
    Space(Colorspace),
    /// The table's numbers as they are — display-referred code values,
    /// to be viewed with the view transform off.
    Raw,
}

/// Where the table comes from.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LutSource {
    /// The file at [`LutParams::path`].
    #[default]
    File,
    /// A view transform run backwards ([`crate::invview`]): display codes
    /// in, scene-linear out. The node is then the "Inverse View".
    InverseView(InverseView),
}

/// A [`crate::graph::NodeKind::Lut`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize), serde(default))]
pub struct LutParams {
    /// As typed: a leading `~` is expanded where it is read
    /// ([`crate::lut::load`]), so a setup stays portable.
    pub path: String,
    /// 0 = the input, 1 = the LUT.
    pub mix: f32,
    /// The space the LUT expects its input in.
    pub input: Colorspace,
    pub shaper: LutShaper,
    /// The log2 shaper's range, in stops about 0.18.
    pub log_min: f32,
    pub log_max: f32,
    pub output: LutOut,
    /// How the cube is interpolated: match Resolve's project setting
    /// ("3D lookup table interpolation") to match its picture. A uniform.
    pub interp: Interp,
    pub source: LutSource,
    /// Triangular dither added to the code values before the lookup, in
    /// 8-bit steps (0 = off): an 8-bit source stretched by an inverse view
    /// bands in the highlights without it. A uniform.
    pub dither: f32,
    /// Soft ceiling on the result, scene-linear (0 = off): the brightest
    /// channel rolls off from half of it towards it, hue kept. A uniform.
    pub limit: f32,
    /// TRANSIENT, never authored: the loaded file's content hash, set only
    /// on the copy a render-cache key is made from, so an edited file
    /// under the same path is a new key. Zero is skipped when saving.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "is_zero"))]
    pub stamp: u64,
}

#[cfg(feature = "serde")]
fn is_zero(v: &u64) -> bool {
    *v == 0
}

impl Default for LutParams {
    fn default() -> Self {
        Self {
            path: String::new(),
            mix: 1.0,
            input: Colorspace::SRGB,
            shaper: LutShaper::Curve,
            log_min: -8.0,
            log_max: 6.0,
            output: LutOut::Same,
            interp: Interp::Tetrahedral,
            source: LutSource::File,
            dither: 0.0,
            limit: 0.0,
            stamp: 0,
        }
    }
}

/// Mid grey, the log2 shaper's anchor.
const GREY: f32 = 0.18;

impl LutParams {
    /// An Inverse View: the viewer's default rendering undone, from a plate
    /// a Load decoded as Rec.709 (the movie default), with a code value of
    /// dither.
    pub fn inverse_view() -> Self {
        Self {
            source: LutSource::InverseView(InverseView::default()),
            input: Colorspace::REC709,
            dither: 1.0,
            ..Default::default()
        }
    }

    pub fn inverse(&self) -> Option<&InverseView> {
        match &self.source {
            LutSource::InverseView(v) => Some(v),
            LutSource::File => None,
        }
    }

    /// What the graph calls this node.
    pub fn name(&self) -> &'static str {
        if self.inverse().is_some() { "Inverse View" } else { Self::NAME }
    }

    /// The table, or why there is none (`Err("")` = no path).
    pub fn status(&self) -> Result<Arc<Lut>, String> {
        match &self.source {
            LutSource::File => crate::lut::load(&self.path),
            LutSource::InverseView(v) => v.table(),
        }
    }

    /// The shaper that runs: an inverse view's table is over the codes.
    fn shaper(&self) -> LutShaper {
        if self.inverse().is_some() { LutShaper::Curve } else { self.shaper }
    }

    /// The output that runs: an inverse view's table holds ACEScct.
    fn output(&self) -> LutOut {
        if self.inverse().is_some() { LutOut::Space(Colorspace::ACESCCT) } else { self.output }
    }

    /// The space the shaping encodes into before the lookup.
    fn shaped(&self) -> Colorspace {
        match self.shaper() {
            LutShaper::Curve => self.input,
            LutShaper::Log2 => Colorspace::new(self.input.primaries, Curve::Linear),
        }
    }

    /// The space the LUT's output is decoded from, or `None` for raw.
    fn out_space(&self) -> Option<Colorspace> {
        match self.output() {
            LutOut::Same => Some(self.shaped()),
            LutOut::Space(cs) => Some(cs),
            LutOut::Raw => None,
        }
    }

    fn log_range(&self) -> (f32, f32) {
        (self.log_min, self.log_max)
    }

    /// The CPU mirror of the generated code, for `lut` as loaded, with
    /// the dither off.
    pub fn apply(&self, lut: Option<&Lut>, working: Colorspace, c: [f32; 3]) -> [f32; 3] {
        self.apply_at(lut, working, c, None)
    }

    /// [`Self::apply`] at a pixel, which is what the dither needs.
    pub fn apply_at(&self, lut: Option<&Lut>, working: Colorspace, c: [f32; 3], px: Option<[f32; 2]>) -> [f32; 3] {
        let Some(lut) = lut else { return c };
        let mut v = Transform::new(working, self.shaped()).apply(c);
        if let (Some(p), true) = (px, self.dither > 0.0) {
            let n = tpdf(p);
            v = std::array::from_fn(|i| v[i] + n[i] * self.dither / 255.0);
        }
        let (lo, hi) = self.log_range();
        let log2 = self.shaper() == LutShaper::Log2;
        if log2 {
            v = v.map(|x| ((x.max(1e-10) / GREY).log2() - lo) / (hi - lo).max(1e-6));
        }
        v = lut.apply_with(v, self.interp);
        if log2 && self.output() == LutOut::Same {
            v = v.map(|y| GREY * (lo + y * (hi - lo)).exp2());
        }
        if let Some(cs) = self.out_space() {
            v = Transform::new(cs, working).apply(v);
        }
        if self.limit > 0.0 {
            v = soft_limit(v, self.limit);
        }
        if self.mix == 0.0 {
            return c;
        }
        std::array::from_fn(|i| c[i] + (v[i] - c[i]) * self.mix)
    }
}

impl NodeDef for LutParams {
    const NAME: &'static str = "LUT";
    /// Image, then the mask.
    const INPUTS: usize = 2;
    const MASK_INPUT: Option<usize> = Some(1);
    /// [has1d, n1, has3d (2 = trilinear), n3], [mix, tiles, width, row1d],
    /// [1d min, log min], [1d max, log max], [3d min, dither], [3d max, limit].
    const VEC4_SLOTS: usize = 6;

    /// The shaping is code; the table, its shape and the mix are not.
    fn topo_tag(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (TOPO_BASE, self.input, self.shaper(), self.output()).hash(&mut h);
        h.finish()
    }

    fn pack(&self, v: &mut [f32]) {
        v.fill(0.0);
        v[4] = self.mix;
        v[11] = self.log_min;
        v[15] = self.log_max;
        v[19] = self.dither.max(0.0) / 255.0;
        v[23] = self.limit.max(0.0);
        let Ok(lut) = self.status() else { return };
        let lay = lut.layout();
        if let Some(s) = &lut.shaper {
            v[0] = 1.0;
            v[1] = s.data.len() as f32;
            v[8..11].copy_from_slice(&s.min);
            v[12..15].copy_from_slice(&s.max);
        }
        if let Some(c) = &lut.cube {
            v[2] = if self.interp == Interp::Trilinear { 2.0 } else { 1.0 };
            v[3] = c.size as f32;
            v[16..19].copy_from_slice(&c.min);
            v[20..23].copy_from_slice(&c.max);
        }
        v[5] = lay.tiles as f32;
        v[6] = lay.width as f32;
        v[7] = lay.row1d as f32;
    }

    fn emit(&self, e: &mut Emit) -> String {
        let input = e.input(0);
        let slot = e.slot();
        let pos = e.pos().to_string();
        let table = e.lut_table();
        let working = e.working();
        let (curves, fns) = e.color_helpers();
        let f = wgsl(self, working, curves, fns);
        let v = e.fresh();
        e.line(format_args!("let {v} = {f}({input}, {slot}u, {table}, {pos});"));
        e.masked(Self::MASK_INPUT, &input, v)
    }
}

/// The lookups, shared by every LUT node: `lut_b_1d` (linear per channel),
/// `lut_b_3d` (tetrahedral, exactly [`crate::lut::Lut3d::sample`]) and
/// `lut_b_3t` (trilinear, [`crate::lut::Lut3d::sample_trilinear`]).
/// Named so the map's alphabetical order declares each before its caller.
const LOOKUPS: &str = r#"
fn lut_a_at(t: texture_2d<f32>, i: vec3i, n: i32, k: i32) -> vec3f {
    return textureLoad(t, vec2i(i.x + n * (i.z % k), i.y + n * (i.z / k)), 0).rgb;
}
fn lut_a_norm(c: vec3f, mn: vec3f, mx: vec3f) -> vec3f {
    let d = mx - mn;
    return clamp(select(vec3f(0.0), (c - mn) / d, (abs(d) > vec3f(1e-20))), vec3f(0.0), vec3f(1.0));
}
fn lut_a_hash(q: vec3u) -> vec3u {
    var v = q * 1664525u + vec3u(1013904223u);
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v = v ^ (v >> vec3u(16u));
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}
fn lut_a_tpdf(p: vec2f) -> vec3f {
    let q = bitcast<vec2u>(vec2i(floor(p)));
    let a = vec3f(lut_a_hash(vec3u(q, 1u)) >> vec3u(8u)) / 16777216.0;
    let b = vec3f(lut_a_hash(vec3u(q, 2u)) >> vec3u(8u)) / 16777216.0;
    return a + b - vec3f(1.0);
}
fn lut_a_limit(c: vec3f, lim: f32) -> vec3f {
    let m = max(c.r, max(c.g, c.b));
    let k = 0.5 * lim;
    if (m <= k) {
        return c;
    }
    let s = k + (lim - k) * (1.0 - exp(-(m - k) / (lim - k)));
    return c * (s / m);
}
fn lut_a_tap(t: texture_2d<f32>, i: i32, w: i32, row: i32) -> vec3f {
    return textureLoad(t, vec2i(i % w, row + i / w), 0).rgb;
}
fn lut_b_1d(t: texture_2d<f32>, c: vec3f, n: i32, w: i32, row: i32, mn: vec3f, mx: vec3f) -> vec3f {
    let u = lut_a_norm(c, mn, mx) * f32(n - 1);
    let i = clamp(vec3i(floor(u)), vec3i(0), vec3i(n - 2));
    let f = u - vec3f(i);
    let r = mix(lut_a_tap(t, i.x, w, row).r, lut_a_tap(t, i.x + 1, w, row).r, f.x);
    let g = mix(lut_a_tap(t, i.y, w, row).g, lut_a_tap(t, i.y + 1, w, row).g, f.y);
    let b = mix(lut_a_tap(t, i.z, w, row).b, lut_a_tap(t, i.z + 1, w, row).b, f.z);
    return vec3f(r, g, b);
}
fn lut_b_3t(t: texture_2d<f32>, c: vec3f, n: i32, k: i32, mn: vec3f, mx: vec3f) -> vec3f {
    let p = lut_a_norm(c, mn, mx) * f32(n - 1);
    let i = clamp(vec3i(floor(p)), vec3i(0), vec3i(n - 2));
    let f = p - vec3f(i);
    let x00 = mix(lut_a_at(t, i, n, k), lut_a_at(t, i + vec3i(1, 0, 0), n, k), f.x);
    let x10 = mix(lut_a_at(t, i + vec3i(0, 1, 0), n, k), lut_a_at(t, i + vec3i(1, 1, 0), n, k), f.x);
    let x01 = mix(lut_a_at(t, i + vec3i(0, 0, 1), n, k), lut_a_at(t, i + vec3i(1, 0, 1), n, k), f.x);
    let x11 = mix(lut_a_at(t, i + vec3i(0, 1, 1), n, k), lut_a_at(t, i + vec3i(1, 1, 1), n, k), f.x);
    return mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
}
fn lut_b_3d(t: texture_2d<f32>, c: vec3f, n: i32, k: i32, mn: vec3f, mx: vec3f) -> vec3f {
    let p = lut_a_norm(c, mn, mx) * f32(n - 1);
    let i = clamp(vec3i(floor(p)), vec3i(0), vec3i(n - 2));
    let f = p - vec3f(i);
    let c000 = lut_a_at(t, i, n, k);
    let c111 = lut_a_at(t, i + vec3i(1, 1, 1), n, k);
    if (f.x > f.y) {
        if (f.y > f.z) {
            return c000 * (1.0 - f.x) + lut_a_at(t, i + vec3i(1, 0, 0), n, k) * (f.x - f.y) + lut_a_at(t, i + vec3i(1, 1, 0), n, k) * (f.y - f.z) + c111 * f.z;
        } else if (f.x > f.z) {
            return c000 * (1.0 - f.x) + lut_a_at(t, i + vec3i(1, 0, 0), n, k) * (f.x - f.z) + lut_a_at(t, i + vec3i(1, 0, 1), n, k) * (f.z - f.y) + c111 * f.y;
        } else {
            return c000 * (1.0 - f.z) + lut_a_at(t, i + vec3i(0, 0, 1), n, k) * (f.z - f.x) + lut_a_at(t, i + vec3i(1, 0, 1), n, k) * (f.x - f.y) + c111 * f.y;
        }
    } else {
        if (f.z > f.y) {
            return c000 * (1.0 - f.z) + lut_a_at(t, i + vec3i(0, 0, 1), n, k) * (f.z - f.y) + lut_a_at(t, i + vec3i(0, 1, 1), n, k) * (f.y - f.x) + c111 * f.x;
        } else if (f.z > f.x) {
            return c000 * (1.0 - f.y) + lut_a_at(t, i + vec3i(0, 1, 0), n, k) * (f.y - f.z) + lut_a_at(t, i + vec3i(0, 1, 1), n, k) * (f.z - f.x) + c111 * f.x;
        } else {
            return c000 * (1.0 - f.y) + lut_a_at(t, i + vec3i(0, 1, 0), n, k) * (f.y - f.x) + lut_a_at(t, i + vec3i(1, 1, 0), n, k) * (f.x - f.z) + c111 * f.z;
        }
    }
}
"#;

/// Emit the node's function (one per shaping) and the shared lookups;
/// returns its name.
pub fn wgsl(
    p: &LutParams,
    working: Colorspace,
    curves: &mut std::collections::BTreeMap<String, String>,
    fns: &mut std::collections::BTreeMap<String, String>,
) -> String {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (working, p.topo_tag()).hash(&mut h);
    let id = h.finish();
    let name = format!("lut_op_{id:016x}");
    if fns.contains_key(&name) {
        return name;
    }
    fns.entry("lut_a_lookups".into()).or_insert_with(|| LOOKUPS.to_string());
    let to = format!("lut_c_in_{id:016x}");
    let def = Transform::new(working, p.shaped()).wgsl(&to, curves);
    fns.insert(to.clone(), def);
    let from = p.out_space().map(|cs| {
        let n = format!("lut_c_out_{id:016x}");
        let def = Transform::new(cs, working).wgsl(&n, curves);
        fns.insert(n.clone(), def);
        n
    });
    let log2 = p.shaper() == LutShaper::Log2;
    let shape_in = if log2 {
        "    v = (log2(max(v, vec3f(1e-10)) / 0.18) - vec3f(lg)) / max(hi - lg, 1e-6);\n"
    } else {
        ""
    };
    let shape_out = if log2 && p.output() == LutOut::Same {
        "    v = 0.18 * exp2(vec3f(lg) + v * (hi - lg));\n"
    } else {
        ""
    };
    let decode = from.map(|f| format!("    v = {f}(v);\n")).unwrap_or_default();
    let def = format!(
        r#"
// LUT: shape into the table's space, 1D then 3D, back. Slots: see
// `nodes::lut::LutParams::VEC4_SLOTS`.
fn {name}(c: vec4f, slot: u32, t: texture_2d<f32>, p: vec2f) -> vec4f {{
    let h = P.v[slot];
    let g = P.v[slot + 1u];
    if ((h.x < 0.5) && (h.z < 0.5)) {{
        return c;
    }}
    let lg = P.v[slot + 2u].w;
    let hi = P.v[slot + 3u].w;
    var v = {to}(c.rgb);
    let dz = P.v[slot + 4u].w;
    if (dz > 0.0) {{
        v = v + lut_a_tpdf(p) * dz;
    }}
{shape_in}    if (h.x > 0.5) {{
        v = lut_b_1d(t, v, i32(h.y), i32(g.z), i32(g.w), P.v[slot + 2u].xyz, P.v[slot + 3u].xyz);
    }}
    if (h.z > 1.5) {{
        v = lut_b_3t(t, v, i32(h.w), i32(g.y), P.v[slot + 4u].xyz, P.v[slot + 5u].xyz);
    }} else if (h.z > 0.5) {{
        v = lut_b_3d(t, v, i32(h.w), i32(g.y), P.v[slot + 4u].xyz, P.v[slot + 5u].xyz);
    }}
{shape_out}{decode}    let lim = P.v[slot + 5u].w;
    if (lim > 0.0) {{
        v = lut_a_limit(v, lim);
    }}
    return vec4f(select(mix(c.rgb, v, g.x), c.rgb, (g.x == 0.0)), c.a);
}}
"#
    );
    fns.insert(name.clone(), def);
    name
}

/// The shader's triangular dither at a pixel, in -1..1 per channel.
pub fn tpdf(p: [f32; 2]) -> [f32; 3] {
    fn hash(q: [u32; 3]) -> [u32; 3] {
        let mut v = q.map(|x| x.wrapping_mul(1664525).wrapping_add(1013904223));
        v[0] = v[0].wrapping_add(v[1].wrapping_mul(v[2]));
        v[1] = v[1].wrapping_add(v[2].wrapping_mul(v[0]));
        v[2] = v[2].wrapping_add(v[0].wrapping_mul(v[1]));
        v = v.map(|x| x ^ (x >> 16));
        v[0] = v[0].wrapping_add(v[1].wrapping_mul(v[2]));
        v[1] = v[1].wrapping_add(v[2].wrapping_mul(v[0]));
        v[2] = v[2].wrapping_add(v[0].wrapping_mul(v[1]));
        v
    }
    let q = [p[0].floor() as i32 as u32, p[1].floor() as i32 as u32];
    let a = hash([q[0], q[1], 1]).map(|x| (x >> 8) as f32 / 16777216.0);
    let b = hash([q[0], q[1], 2]).map(|x| (x >> 8) as f32 / 16777216.0);
    std::array::from_fn(|i| a[i] + b[i] - 1.0)
}

/// The shader's soft ceiling: the brightest channel eases from `lim / 2`
/// towards `lim`, the others scaled with it so the hue holds.
pub fn soft_limit(c: [f32; 3], lim: f32) -> [f32; 3] {
    let m = c[0].max(c[1]).max(c[2]);
    let k = 0.5 * lim;
    if m <= k {
        return c;
    }
    let s = k + (lim - k) * (1.0 - (-(m - k) / (lim - k)).exp());
    c.map(|v| v * (s / m))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn swap() -> Lut {
        crate::lut::parse(
            "cube",
            b"LUT_3D_SIZE 2\n0 0 0\n0 0 1\n0 1 0\n0 1 1\n1 0 0\n1 0 1\n1 1 0\n1 1 1\n",
        )
        .unwrap()
    }

    #[test]
    fn same_out_round_trips_an_identity_cube() {
        let id = crate::lut::parse(
            "cube",
            b"LUT_3D_SIZE 2\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n",
        )
        .unwrap();
        for shaper in LutShaper::ALL {
            let p = LutParams { shaper, ..Default::default() };
            for c in [[0.18f32, 0.5, 0.02], [0.4, 0.3, 0.2]] {
                let o = p.apply(Some(&id), Colorspace::LINEAR_REC709, c);
                for k in 0..3 {
                    assert!((o[k] - c[k]).abs() < 1e-3, "{shaper:?}: {c:?} -> {o:?}");
                }
            }
        }
    }

    #[test]
    fn mix_zero_and_no_table_are_the_input() {
        let c = [0.3, 0.6, 0.1];
        let p = LutParams { mix: 0.0, ..Default::default() };
        assert_eq!(p.apply(Some(&swap()), Colorspace::ACESCG, c), c);
        assert_eq!(LutParams::default().apply(None, Colorspace::ACESCG, c), c);
        let raw = LutParams { output: LutOut::Raw, input: Colorspace::LINEAR_REC709, ..Default::default() };
        let o = raw.apply(Some(&swap()), Colorspace::LINEAR_REC709, c);
        assert!((o[0] - 0.1).abs() < 1e-5 && (o[2] - 0.3).abs() < 1e-5, "{o:?}");
    }

    #[test]
    fn an_inverse_view_is_one_shader_whatever_its_knobs() {
        let a = LutParams::inverse_view();
        let b = LutParams { dither: 0.0, limit: 40.0, mix: 0.5, ..a.clone() };
        assert_eq!(a.topo_tag(), b.topo_tag());
        // The authored shaper/output are ignored while it is an inverse view.
        let c = LutParams { shaper: LutShaper::Log2, output: LutOut::Raw, ..a.clone() };
        assert_eq!(a.topo_tag(), c.topo_tag());
        assert_eq!(a.name(), "Inverse View");
        assert_eq!(LutParams::default().name(), "LUT");
    }

    #[test]
    fn the_limit_eases_and_keeps_the_hue() {
        assert_eq!(soft_limit([0.2, 0.4, 1.0], 10.0), [0.2, 0.4, 1.0]);
        let o = soft_limit([4.0, 8.0, 80.0], 10.0);
        assert!(o[2] < 10.0 && o[2] > 9.0, "{o:?}");
        assert!((o[0] / o[2] - 0.05).abs() < 1e-5 && (o[1] / o[2] - 0.1).abs() < 1e-5);
    }

    #[test]
    fn the_dither_is_triangular_and_centred() {
        let mut sum = 0.0f64;
        let mut inside_half = 0;
        let n = 200 * 200;
        for y in 0..200 {
            for x in 0..200 {
                let d = tpdf([x as f32 + 0.5, y as f32 + 0.5]);
                for v in d {
                    assert!((-1.0..=1.0).contains(&v));
                    sum += v as f64;
                    if v.abs() < 0.5 {
                        inside_half += 1;
                    }
                }
            }
        }
        assert!((sum / (3 * n) as f64).abs() < 0.01, "mean {}", sum / (3 * n) as f64);
        // A triangle puts 3/4 of its mass inside half its reach.
        let f = inside_half as f64 / (3 * n) as f64;
        assert!((f - 0.75).abs() < 0.01, "{f}");
    }

    #[test]
    fn only_the_shaping_is_topology() {
        let a = LutParams::default();
        let b = LutParams {
            path: "/x.cube".into(),
            mix: 0.5,
            log_min: -3.0,
            stamp: 9,
            interp: Interp::Trilinear,
            ..a.clone()
        };
        assert_eq!(a.topo_tag(), b.topo_tag());
        assert_ne!(a.topo_tag(), LutParams { shaper: LutShaper::Log2, ..a.clone() }.topo_tag());
        assert_ne!(a.topo_tag(), LutParams { output: LutOut::Raw, ..a.clone() }.topo_tag());
        assert_ne!(a.topo_tag(), LutParams { input: Colorspace::LOGC3, ..a }.topo_tag());
    }
}
