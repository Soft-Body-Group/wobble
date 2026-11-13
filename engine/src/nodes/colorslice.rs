//! ColorSlice: Resolve's secondary for "the reds", "the skin", "the sky" —
//! six hue vectors and a skin vector, each with its own hue, saturation and
//! density, done SUBTRACTIVELY.
//!
//! The app calls it the **Tinter** (its `NAME`, the menus, the panel). The
//! code and the SETUP key stay `ColorSlice`: a setup stores the variant
//! name, and "Tinter" is what setups from before the Sep-23 prune stored for
//! this node's predecessor, whose params are not these — reusing the key
//! would load those as a silently different node.
//!
//! What Resolve documents, and this implements:
//!
//! - **Seven vectors**: red, skin, yellow, green, cyan, blue, magenta. The
//!   six are the vertices of the working space's own RGB hue circle; skin
//!   is not a vertex but an OVERLAPPING region between red and yellow, laid
//!   on top of the other six rather than carved out of them.
//! - **Per vector**: Center (where the vector sits), Hue (turn a red shirt
//!   magenta), Saturation, Density — and a Highlight that shows which pixels
//!   the vector is moving.
//! - **Global**: Saturation, Density, Saturation Depth and Density Depth
//!   ("the effect of saturation / density adjustments on brighter areas"),
//!   and Saturation Balance ("how saturation changes affect the luminance
//!   balance in colours with medium saturation levels").
//! - **A subtractive model**: more saturation means more colour WITHOUT
//!   more light — the dye-in-film behaviour, not the additive RGB one.
//!
//! What Resolve does NOT publish is the arithmetic, so what follows is this
//! node's own, chosen to honour those descriptions and this codebase's
//! habits. It will not match Resolve's numbers, and says so.
//!
//! **Subtractive saturation** is a power law on each channel's ratio to the
//! BRIGHTEST channel: `c' = max * (c / max)^s`. The brightest channel never
//! moves, the others are pulled down (or let up), so saturating deepens a
//! colour instead of lighting it — and because it is a power law it cannot
//! push a channel negative, which is the failure an additive lerp makes at
//! a saturation of two (see [`crate::nodes::saturation`], which makes the
//! same argument about the ratio to the neutral). A grey has every ratio at
//! one and is untouched at any setting.
//!
//! **Density** is exposure, scaled by how colourful the pixel is:
//! `c' = c * 2^(-density * q)` with `q = (max - min) / max`. One stop of
//! density takes a fully saturated colour down a stop and leaves a grey
//! exactly where it was — density is a property of colour.
//!
//! **Vectors blend, they do not stack.** The six vertex vectors are a
//! smooth partition of the hue circle between their (movable) centres, so
//! every coloured pixel's weights sum to one: an orange is half red, half
//! yellow, and setting all six to the same saturation is exactly the
//! global saturation. Skin is the exception, on purpose, and adds on top.
//! Weights fade out toward the neutral axis, where hue is not a fact.
//!
//! **Hue** turns in Oklab, where a rotation holds perceived lightness — a
//! hue change that brightened or darkened the colour would be a second
//! correction nobody asked for.

use crate::codegen::Emit;
use crate::color::Colorspace;
use crate::nodes::NodeDef;
use std::collections::BTreeMap;

const TOPO_BASE: u64 = 0x6D;

/// How many vectors, in hue order around the wheel.
pub const VECTORS: usize = 7;
/// Their names, index-matched to [`ColorSliceParams::vectors`].
pub const NAMES: [&str; VECTORS] = ["red", "skin", "yellow", "green", "cyan", "blue", "magenta"];
/// Which one is the skin region.
pub const SKIN: usize = 1;
/// The six that tile the hue circle, in hue order, and where each sits on
/// the working space's RGB hue circle: the primaries and secondaries.
const PART: [usize; 6] = [0, 2, 3, 4, 5, 6];
const PART_BASE: [f32; 6] = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0];
/// How far a vertex vector's Center may move, in degrees. Under half the 60°
/// spacing, so the six can never pass one another and the partition stays
/// a partition.
pub const LEAN: f32 = 25.0;
/// How far the skin region's Center may move.
pub const SKIN_LEAN: f32 = 15.0;
/// Half the width of the skin region, in degrees of hue.
pub const SKIN_HALF: f32 = 20.0;
/// Colourfulness below which a pixel's hue is not trusted: a near-grey's
/// hue is noise, and a grey card must not take whatever correction happens
/// to sit where its hue happens to fall.
const GATE: f32 = 0.08;
/// The scene-linear level treated as the middle of the tonal range when
/// deciding what "brighter areas" are: mid grey.
const TONE_PIVOT: f32 = 0.18;
/// How far the highlight view sinks everything outside the vector.
const HL_DIM: f32 = 0.22;

/// The reference skin: the X-Rite ColorChecker's "light skin" patch in
/// linear Rec.709. Where the skin region sits is where THIS lands on the
/// working space's hue circle — about 15° in linear Rec.709 and 22° in
/// ACEScg — rather than a number that is only right in one space.
const SKIN_REF_709: [f32; 3] = [0.540, 0.305, 0.223];

/// One vector's correction. Identity by default.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct SliceVector {
    /// Where the vector sits, as an offset from its default, in degrees.
    /// Held to ±[`LEAN`] (±[`SKIN_LEAN`] for skin) when packed.
    pub center: f32,
    /// Hue shift, in degrees. Positive turns toward the next vector along
    /// the wheel (red toward yellow), negative toward the previous one (red
    /// toward magenta).
    pub hue: f32,
    /// Subtractive saturation: 1 is no change, above deepens, below lets
    /// the colour up toward its brightest channel.
    pub sat: f32,
    /// Density in stops at full saturation: positive darkens this vector's
    /// colours, negative lightens them. Greys are never touched.
    pub density: f32,
}

impl Default for SliceVector {
    fn default() -> Self {
        Self { center: 0.0, hue: 0.0, sat: 1.0, density: 0.0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct ColorSliceParams {
    /// Index-matched to [`NAMES`].
    pub vectors: [SliceVector; VECTORS],
    /// Global subtractive saturation, over every colour.
    pub sat: f32,
    /// Global density, in stops at full saturation.
    pub density: f32,
    /// How much of the saturation change reaches the BRIGHTER areas: 1 is
    /// even across the tonal range, 0 leaves the brights alone, 2 pushes
    /// them twice as hard.
    pub sat_depth: f32,
    /// The same, for density.
    pub den_depth: f32,
    /// How much a saturation change is allowed to move the LUMINANCE of
    /// medium-saturation colours. Subtractive saturation darkens as it
    /// deepens; 0 lets it, 1 holds their luminance where it was, negative
    /// darkens them further.
    pub sat_balance: f32,
    /// Which vector the viewer highlights. Authored — it is a knob, it
    /// undoes and saves — and honoured only while `previewing`.
    pub highlight: Option<u8>,
    /// Transient: set by the app when this node IS the view target. Never
    /// authored, never serialized — the colour keyer's contract, and what
    /// keeps a highlight out of every render and every consumer.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub previewing: bool,
}

impl Default for ColorSliceParams {
    fn default() -> Self {
        Self {
            vectors: [SliceVector::default(); VECTORS],
            sat: 1.0,
            density: 0.0,
            sat_depth: 1.0,
            den_depth: 1.0,
            sat_balance: 0.0,
            highlight: None,
            previewing: false,
        }
    }
}

/// `x` wrapped into [0, 360).
fn wrap360(x: f32) -> f32 {
    x - 360.0 * (x / 360.0).floor()
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// A colour's position on the RGB hue circle, in degrees: red at 0, yellow
/// 60, green 120, cyan 180, blue 240, magenta 300. The same branch order as
/// the WGSL, so the two agree on a tie.
pub fn hue_of(c: [f32; 3]) -> f32 {
    let mx = c[0].max(c[1]).max(c[2]);
    let mn = c[0].min(c[1]).min(c[2]);
    let d = mx - mn;
    if d <= 1e-10 {
        return 0.0;
    }
    let h = if mx == c[0] {
        (c[1] - c[2]) / d
    } else if mx == c[1] {
        (c[2] - c[0]) / d + 2.0
    } else {
        (c[0] - c[1]) / d + 4.0
    };
    wrap360(h * 60.0)
}

/// Where vector `i` sits on `working`'s RGB hue circle before its Center:
/// a primary or secondary for the six, the skin reference for skin.
pub fn base_hue(i: usize, working: Colorspace) -> f32 {
    if i == SKIN {
        return skin_hue(working);
    }
    PART.iter().position(|&p| p == i).map(|k| PART_BASE[k]).unwrap_or(0.0)
}

/// How far vector `i`'s Center may move either way, in degrees.
pub fn lean_of(i: usize) -> f32 {
    if i == SKIN {
        SKIN_LEAN
    } else {
        LEAN
    }
}

/// Where the skin region sits on `working`'s hue circle.
pub fn skin_hue(working: Colorspace) -> f32 {
    let m = crate::color::primaries_matrix(crate::color::Primaries::Rec709, working.primaries);
    let r = SKIN_REF_709;
    let c = [
        m[0][0] * r[0] + m[0][1] * r[1] + m[0][2] * r[2],
        m[1][0] * r[0] + m[1][1] * r[1] + m[1][2] * r[2],
        m[2][0] * r[0] + m[2][1] * r[1] + m[2][2] * r[2],
    ];
    hue_of(c)
}

impl ColorSliceParams {
    /// Vector `i`'s Center offset as the shader sees it: held inside the
    /// range that keeps the vectors in order.
    pub fn center_of(&self, i: usize) -> f32 {
        let lean = if i == SKIN { SKIN_LEAN } else { LEAN };
        let c = self.vectors[i].center;
        if c.is_finite() {
            c.clamp(-lean, lean)
        } else {
            0.0
        }
    }

    /// Every vector's weight for a pixel of hue `h` and colourfulness `q`,
    /// neutral gate applied. The six vertex vectors sum to the gate; skin
    /// rides on top.
    pub fn weights(&self, h: f32, q: f32, working: Colorspace) -> [f32; VECTORS] {
        let mut w = [0.0f32; VECTORS];
        for k in 0..6 {
            let kn = (k + 1) % 6;
            let a = PART_BASE[k] + self.center_of(PART[k]);
            let b = PART_BASE[kn] + self.center_of(PART[kn]);
            let len = wrap360(b - a);
            let d = wrap360(h - a);
            if d < len {
                let s = smoothstep(0.0, 1.0, d / len.max(1e-3));
                w[PART[k]] += 1.0 - s;
                w[PART[kn]] += s;
            }
        }
        let sc = skin_hue(working) + self.center_of(SKIN);
        let dsk = (wrap360(h - sc + 180.0) - 180.0).abs();
        w[SKIN] = 1.0 - smoothstep(0.0, SKIN_HALF, dsk);
        let g = smoothstep(0.0, GATE, q);
        w.map(|x| x * g)
    }

    /// The CPU mirror of the generated code, colour only (the highlight view
    /// is [`Self::highlight_weight`]). What the tests check the GPU against,
    /// and what anything can ask without rendering a frame.
    pub fn apply(&self, c: [f32; 3], working: Colorspace) -> [f32; 3] {
        let lw = working.primaries.luma_weights();
        let luma = |v: [f32; 3]| v[0] * lw[0] + v[1] * lw[1] + v[2] * lw[2];
        let mx = c[0].max(c[1]).max(c[2]);
        let mn = c[0].min(c[1]).min(c[2]);
        if mx <= 1e-6 {
            return c;
        }
        let q = ((mx - mn) / mx).clamp(0.0, 1.0);
        let w = self.weights(hue_of(c), q, working);

        let mut hue = 0.0;
        let mut sat = 1.0;
        let mut den = 0.0;
        for (i, v) in self.vectors.iter().enumerate() {
            hue += w[i] * v.hue;
            sat += w[i] * (v.sat - 1.0);
            den += w[i] * v.density;
        }
        let sat = (sat * self.sat).max(0.0);
        let den = den + self.density;

        // Hue, in Oklab: a rotation of the chroma that leaves L alone.
        let c1 = if hue.abs() > 1e-4 {
            let m = crate::keyer::oklab_matrix(working);
            let lab = crate::keyer::oklab_from_linear_in(c, m);
            let (sn, cs) = hue.to_radians().sin_cos();
            let ab = [lab[1] * cs - lab[2] * sn, lab[1] * sn + lab[2] * cs];
            crate::nodes::saturation::linear_from_oklab([lab[0], ab[0], ab[1]], working)
        } else {
            c
        };

        // Where on the tonal range this pixel is, for the two depths.
        let y0 = luma(c1);
        let t = y0.max(0.0) / (y0.max(0.0) + TONE_PIVOT);

        // Subtractive saturation, the brights' share set by the depth.
        let s = (1.0 + (sat - 1.0) * lerp(1.0, self.sat_depth, t)).max(0.0);
        let mx1 = c1[0].max(c1[1]).max(c1[2]);
        let c2 = if (s - 1.0).abs() < 1e-6 || mx1 <= 1e-6 {
            c1
        } else {
            c1.map(|x| if x > 0.0 { mx1 * (x / mx1).powf(s) } else { x })
        };

        // Balance: hand medium-saturation colours back some (or less) of the
        // luminance the subtractive step took.
        let y1 = luma(c2);
        let k = self.sat_balance * 4.0 * q * (1.0 - q);
        let c3 = if k != 0.0 && y0 > 1e-6 && y1 > 1e-6 {
            let f = (y0 / y1).clamp(0.25, 4.0).powf(k);
            c2.map(|x| x * f)
        } else {
            c2
        };

        // Density, on the colour as it now is.
        let mx3 = c3[0].max(c3[1]).max(c3[2]);
        let mn3 = c3[0].min(c3[1]).min(c3[2]);
        let q3 = if mx3 > 1e-6 { ((mx3 - mn3) / mx3).clamp(0.0, 1.0) } else { 0.0 };
        let d = den * lerp(1.0, self.den_depth, t);
        if d.abs() < 1e-6 {
            c3
        } else {
            let f = (-d * q3).exp2();
            c3.map(|x| x * f)
        }
    }

    /// How strongly the highlighted vector owns this pixel, 0..1 — what the
    /// highlight view keeps in colour. 0 when nothing is highlighted or the
    /// node is not being viewed.
    pub fn highlight_weight(&self, c: [f32; 3], working: Colorspace) -> f32 {
        let Some(v) = self.live_highlight() else { return 0.0 };
        let mx = c[0].max(c[1]).max(c[2]);
        let mn = c[0].min(c[1]).min(c[2]);
        if mx <= 1e-6 {
            return 0.0;
        }
        let q = ((mx - mn) / mx).clamp(0.0, 1.0);
        self.weights(hue_of(c), q, working)[v]
    }

    /// The vector the SHADER highlights: the authored one, but only while
    /// this node is the view target, and only if it names a vector.
    pub fn live_highlight(&self) -> Option<usize> {
        let v = self.highlight? as usize;
        (self.previewing && v < VECTORS).then_some(v)
    }
}

/// Emit the node's WGSL function once per shader and return its name. The
/// working space picks the skin region's centre and the luma weights, and is
/// already part of the topology hash, so baking them in costs nothing.
fn wgsl(
    working: Colorspace,
    helpers: &mut BTreeMap<String, String>,
    fns: &mut BTreeMap<String, String>,
) -> String {
    let name = "color_slice_op".to_string();
    if fns.contains_key(&name) {
        return name;
    }
    // The way back from Oklab, emitted once and shared.
    let to_lin = crate::nodes::saturation::oklab_inverse_wgsl(working, helpers);
    let w = working.primaries.luma_weights();
    let lw = format!("vec3f({:.7}, {:.7}, {:.7})", w[0], w[1], w[2]);
    let skin = skin_hue(working);

    // The six segments of the partition, straight-line: segment k runs from
    // vertex vector PART[k] to the next one along.
    let mut segments = String::new();
    for k in 0..6 {
        let kn = (k + 1) % 6;
        let (i, j) = (PART[k], PART[kn]);
        segments.push_str(&format!(
            "    {{
        let a = {:.4} + P.v[slot + {}u].x;
        let b = {:.4} + P.v[slot + {}u].x;
        let len = cs_wrap360(b - a);
        let d = cs_wrap360(h - a);
        if (d < len) {{
            let s = smoothstep(0.0, 1.0, d / max(len, 1e-3));
            w[{i}] += 1.0 - s;
            w[{j}] += s;
        }}
    }}
",
            PART_BASE[k],
            2 + i,
            PART_BASE[kn],
            2 + j,
        ));
    }
    // Blend every vector's correction by its weight, and pick out the
    // highlighted vector's weight on the way.
    let mut blend = String::new();
    for i in 0..VECTORS {
        blend.push_str(&format!(
            "    let v{i} = P.v[slot + {}u];
    hue = hue + w[{i}] * v{i}.y;
    sat = sat + w[{i}] * (v{i}.z - 1.0);
    den = den + w[{i}] * v{i}.w;
    pw = select(pw, w[{i}], hl.z == {i}.0);
",
            2 + i
        ));
    }

    let def = format!(
        r#"
fn cs_wrap360(x: f32) -> f32 {{
    return x - 360.0 * floor(x / 360.0);
}}
// The RGB hue circle, degrees: red 0, yellow 60 ... magenta 300. Branch
// order matches `colorslice::hue_of`, so the CPU mirror agrees on a tie.
fn cs_hue(c: vec3f) -> f32 {{
    let mx = max(c.r, max(c.g, c.b));
    let mn = min(c.r, min(c.g, c.b));
    let d = mx - mn;
    if (d <= 1e-10) {{ return 0.0; }}
    var h = 0.0;
    if (mx == c.r) {{ h = (c.g - c.b) / d; }}
    else if (mx == c.g) {{ h = (c.b - c.r) / d + 2.0; }}
    else {{ h = (c.r - c.g) / d + 4.0; }}
    return cs_wrap360(h * 60.0);
}}
// ColorSlice. slot+0 = (sat, density, sat depth, density depth), slot+1 =
// (sat balance, highlight armed, highlight vector, -), slot+2+i = vector i's
// (center offset, hue, sat, density), i in red, skin, yellow, green, cyan,
// blue, magenta.
fn {name}(c: vec4f, slot: u32) -> vec4f {{
    let g0 = P.v[slot];
    let hl = P.v[slot + 1u];
    let rgb = c.rgb;
    let mx = max(rgb.r, max(rgb.g, rgb.b));
    let mn = min(rgb.r, min(rgb.g, rgb.b));
    if (mx <= 1e-6) {{ return c; }}
    let q = clamp((mx - mn) / mx, 0.0, 1.0);
    let h = cs_hue(rgb);

    var w = array<f32, 7>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
{segments}    let sc = {skin:.4} + P.v[slot + {skin_slot}u].x;
    let dsk = abs(cs_wrap360(h - sc + 180.0) - 180.0);
    w[{SKIN}] = 1.0 - smoothstep(0.0, {SKIN_HALF:.4}, dsk);
    let gate = smoothstep(0.0, {GATE:.4}, q);
    for (var i = 0u; i < 7u; i = i + 1u) {{ w[i] = w[i] * gate; }}

    var hue = 0.0;
    var sat = 1.0;
    var den = 0.0;
    var pw = 0.0;
{blend}    sat = max(sat * g0.x, 0.0);
    den = den + g0.y;

    // Hue in Oklab: the chroma turns, the lightness does not.
    var c1 = rgb;
    if (abs(hue) > 1e-4) {{
        let lab = oklab_from_linear(rgb);
        let th = radians(hue);
        let cs = cos(th);
        let sn = sin(th);
        c1 = {to_lin}(vec3f(lab.x, lab.y * cs - lab.z * sn, lab.y * sn + lab.z * cs));
    }}

    let y0 = dot(c1, {lw});
    let t = max(y0, 0.0) / (max(y0, 0.0) + {TONE_PIVOT:.4});

    // Subtractive saturation: a power law on the ratio to the brightest
    // channel, so it deepens instead of lighting and never goes negative.
    let s = max(1.0 + (sat - 1.0) * mix(1.0, g0.z, t), 0.0);
    let mx1 = max(c1.r, max(c1.g, c1.b));
    var c2 = c1;
    if (abs(s - 1.0) >= 1e-6 && mx1 > 1e-6) {{
        let r = pow(max(c1 / mx1, vec3f(1e-20)), vec3f(s)) * mx1;
        c2 = select(c1, r, c1 > vec3f(0.0));
    }}

    // Balance: how much luminance a medium-saturation colour may lose.
    let y1 = dot(c2, {lw});
    let k = hl.x * 4.0 * q * (1.0 - q);
    var c3 = c2;
    if (k != 0.0 && y0 > 1e-6 && y1 > 1e-6) {{
        c3 = c2 * pow(clamp(y0 / y1, 0.25, 4.0), k);
    }}

    // Density: exposure, scaled by how colourful the pixel now is.
    let mx3 = max(c3.r, max(c3.g, c3.b));
    let mn3 = min(c3.r, min(c3.g, c3.b));
    let q3 = select(0.0, clamp((mx3 - mn3) / max(mx3, 1e-6), 0.0, 1.0), mx3 > 1e-6);
    let dd = den * mix(1.0, g0.w, t);
    var c4 = c3;
    if (abs(dd) >= 1e-6) {{
        c4 = c3 * exp2(-dd * q3);
    }}

    // Highlight: the vector's pixels as graded, everything else sunk grey.
    let lit = mix(vec3f(dot(c4, {lw})) * {HL_DIM:.4}, c4, pw);
    return vec4f(select(c4, lit, hl.y > 0.5), c.a);
}}
"#,
        skin_slot = 2 + SKIN,
    );
    fns.insert(name.clone(), def);
    name
}

impl NodeDef for ColorSliceParams {
    const NAME: &'static str = "Tinter";
    /// Image, then the mask.
    const INPUTS: usize = 2;
    const MASK_INPUT: Option<usize> = Some(1);
    /// Two global vec4s, then one per vector.
    const VEC4_SLOTS: usize = 2 + VECTORS;

    /// Every control is a uniform: the vector count is fixed, so there is one
    /// shape of code and no value can change it. (The working space picks
    /// the skin centre, and it is already structural for the whole graph.)
    fn topo_tag(&self) -> u64 {
        TOPO_BASE
    }

    fn pack(&self, v: &mut [f32]) {
        v[0] = self.sat.max(0.0);
        v[1] = self.density;
        v[2] = self.sat_depth.max(0.0);
        v[3] = self.den_depth.max(0.0);
        v[4] = self.sat_balance;
        match self.live_highlight() {
            Some(i) => {
                v[5] = 1.0;
                v[6] = i as f32;
            }
            None => {
                v[5] = 0.0;
                v[6] = 0.0;
            }
        }
        for i in 0..VECTORS {
            let o = (2 + i) * 4;
            let vec = &self.vectors[i];
            v[o] = self.center_of(i);
            v[o + 1] = vec.hue;
            v[o + 2] = vec.sat.max(0.0);
            v[o + 3] = vec.density;
        }
    }

    fn emit(&self, e: &mut Emit) -> String {
        let input = e.input(0);
        let slot = e.slot();
        let working = e.working();
        let (helpers, fns) = e.color_helpers();
        let f = wgsl(working, helpers, fns);
        let v = e.fresh();
        e.line(format_args!("let {v} = {f}({input}, {slot}u);"));
        e.masked(Self::MASK_INPUT, &input, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WS: Colorspace = Colorspace::ACESCG;

    fn luma(c: [f32; 3]) -> f32 {
        let w = WS.primaries.luma_weights();
        c[0] * w[0] + c[1] * w[1] + c[2] * w[2]
    }

    fn close3(a: [f32; 3], b: [f32; 3], eps: f32) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() <= eps)
    }

    fn with(i: usize, f: impl Fn(&mut SliceVector)) -> ColorSliceParams {
        let mut p = ColorSliceParams::default();
        f(&mut p.vectors[i]);
        p
    }

    /// A colour exactly on a vector's own hue, fairly saturated.
    fn at_hue(deg: f32) -> [f32; 3] {
        // Walk the RGB hue circle at q = 0.7, max 0.6.
        let (mx, q) = (0.6f32, 0.7f32);
        let mn = mx * (1.0 - q);
        let x = (deg / 60.0).rem_euclid(6.0);
        let f = x - x.floor();
        let (up, down) = (mn + (mx - mn) * f, mx - (mx - mn) * f);
        match x as i32 {
            0 => [mx, up, mn],
            1 => [down, mx, mn],
            2 => [mn, mx, up],
            3 => [mn, down, mx],
            4 => [up, mn, mx],
            _ => [mx, mn, down],
        }
    }

    #[test]
    fn the_hue_circle_is_rgbs_own() {
        for (deg, want) in [(0.0, 0.0), (60.0, 60.0), (120.0, 120.0), (200.0, 200.0), (330.0, 330.0)] {
            let got = hue_of(at_hue(deg));
            assert!((got - want).abs() < 1e-3, "{deg}: got {got}");
        }
        assert_eq!(hue_of([0.3, 0.3, 0.3]), 0.0, "a grey has no hue to report");
    }

    /// A fresh node is the identity, on colours, on greys and on the
    /// out-of-gamut negatives a scene-linear plate carries.
    #[test]
    fn the_default_is_an_identity() {
        let p = ColorSliceParams::default();
        for c in [
            [0.18, 0.18, 0.18],
            [0.8, 0.3, 0.1],
            [0.05, 0.4, 0.9],
            [4.0, 1.0, 0.2],
            [0.3, -0.02, 0.1],
        ] {
            assert!(close3(p.apply(c, WS), c, 1e-6), "{c:?} moved to {:?}", p.apply(c, WS));
        }
    }

    /// Density is a property of colour: no setting of any vector or of the
    /// globals moves a grey.
    #[test]
    fn a_grey_is_untouched_by_everything() {
        let mut p = ColorSliceParams {
            sat: 2.0,
            density: 1.5,
            sat_balance: 1.0,
            ..Default::default()
        };
        for v in &mut p.vectors {
            *v = SliceVector { center: 10.0, hue: 40.0, sat: 0.2, density: -1.0 };
        }
        for g in [0.01f32, 0.18, 1.0, 6.0] {
            let c = [g; 3];
            assert!(close3(p.apply(c, WS), c, 1e-6), "grey {g} moved to {:?}", p.apply(c, WS));
        }
    }

    /// The six vertex vectors TILE the hue circle: every coloured pixel's
    /// weights sum to one, so setting all six alike is the global control.
    #[test]
    fn the_six_vectors_are_a_partition_and_skin_rides_on_top() {
        let p = ColorSliceParams::default();
        for deg in (0..360).step_by(7) {
            let w = p.weights(deg as f32, 1.0, WS);
            let six: f32 = PART.iter().map(|i| w[*i]).sum();
            assert!((six - 1.0).abs() < 1e-5, "at {deg}° the six sum to {six}");
        }
        // All six at 1.5 is the global 1.5, on colours skin does not reach.
        let mut all = ColorSliceParams::default();
        for i in PART {
            all.vectors[i].sat = 1.5;
        }
        let global = ColorSliceParams { sat: 1.5, ..Default::default() };
        for c in [at_hue(120.0), at_hue(210.0), at_hue(290.0)] {
            assert!(close3(all.apply(c, WS), global.apply(c, WS), 1e-4), "{c:?}");
        }
        // Skin peaks where the working space puts skin, and overlaps red.
        let sk = skin_hue(WS);
        assert!((10.0..35.0).contains(&sk), "skin sits at {sk}° in ACEScg");
        let w = p.weights(sk, 1.0, WS);
        assert!((w[SKIN] - 1.0).abs() < 1e-5 && w[0] > 0.0, "{w:?}");
        assert_eq!(p.weights(200.0, 1.0, WS)[SKIN], 0.0, "skin reached cyan");
    }

    /// A vector moves its own colours and nothing across the wheel.
    #[test]
    fn a_vector_acts_on_its_own_colours() {
        let p = with(5, |v| v.sat = 2.0); // blue
        let blue = at_hue(240.0);
        let red = at_hue(0.0);
        assert!(!close3(p.apply(blue, WS), blue, 1e-3), "blue did not move");
        assert!(close3(p.apply(red, WS), red, 1e-6), "red moved for a blue correction");
    }

    /// SUBTRACTIVE: saturating deepens rather than lights. The brightest
    /// channel holds, the others go down, luminance falls — and nothing is
    /// pushed negative however hard it is pushed.
    #[test]
    fn saturation_deepens_instead_of_lighting() {
        let p = ColorSliceParams { sat: 3.0, ..Default::default() };
        let c = [0.8, 0.4, 0.2];
        let out = p.apply(c, WS);
        assert!((out[0] - 0.8).abs() < 1e-5, "the brightest channel moved: {out:?}");
        assert!(out[1] < c[1] && out[2] < c[2], "{out:?}");
        assert!(luma(out) < luma(c), "saturating lit the colour");
        assert!(out.iter().all(|x| *x >= 0.0), "went negative: {out:?}");
        // Letting it up (below 1) rises toward the brightest channel.
        let up = ColorSliceParams { sat: 0.0, ..Default::default() }.apply(c, WS);
        assert!(close3(up, [0.8; 3], 1e-5), "{up:?}");
    }

    /// Density darkens in proportion to colourfulness: more for the more
    /// saturated of two colours, one stop at full saturation per stop set.
    #[test]
    fn density_darkens_colour_by_how_much_colour_it_has() {
        let p = ColorSliceParams { density: 1.0, ..Default::default() };
        let full = [0.5, 0.0, 0.0];
        assert!(close3(p.apply(full, WS), [0.25, 0.0, 0.0], 1e-5), "{:?}", p.apply(full, WS));
        let mild = [0.5, 0.4, 0.4];
        let (a, b) = (p.apply(full, WS)[0] / 0.5, p.apply(mild, WS)[0] / 0.5);
        assert!(a < b && b < 1.0, "full {a} vs mild {b}");
        let lighter = ColorSliceParams { density: -1.0, ..Default::default() };
        assert!(lighter.apply(full, WS)[0] > 0.5);
    }

    /// Hue turns toward the neighbours, and holds perceived lightness.
    #[test]
    fn hue_turns_red_toward_yellow_or_magenta_and_keeps_its_lightness() {
        let red = at_hue(0.0);
        let warm = with(0, |v| v.hue = 20.0).apply(red, WS);
        let cool = with(0, |v| v.hue = -20.0).apply(red, WS);
        let h = |c: [f32; 3]| {
            let x = hue_of(c);
            if x > 180.0 { x - 360.0 } else { x }
        };
        assert!(h(warm) > 2.0, "positive hue did not go toward yellow: {}", h(warm));
        assert!(h(cool) < -2.0, "negative hue did not go toward magenta: {}", h(cool));
        let m = crate::keyer::oklab_matrix(WS);
        let l = |c: [f32; 3]| crate::keyer::oklab_from_linear_in(c, m)[0];
        assert!((l(warm) - l(red)).abs() < 1e-4, "hue moved the lightness");
    }

    /// The depths set how much of a change reaches the BRIGHTER areas.
    #[test]
    fn the_depths_decide_what_reaches_the_brights() {
        let (dark, bright) = ([0.08, 0.03, 0.01], [8.0, 3.0, 1.0]);
        let sat = |d: f32| ColorSliceParams { sat: 2.0, sat_depth: d, ..Default::default() };
        let moved = |p: &ColorSliceParams, c: [f32; 3]| {
            let o = p.apply(c, WS);
            (o[1] / o[0] - c[1] / c[0]).abs()
        };
        assert!(moved(&sat(1.0), bright) > 0.05, "full depth did not reach the brights");
        assert!(
            moved(&sat(0.0), bright) < moved(&sat(1.0), bright) * 0.2,
            "zero depth still saturated the brights"
        );
        assert!(
            (moved(&sat(0.0), dark) - moved(&sat(1.0), dark)).abs() < moved(&sat(1.0), dark) * 0.4,
            "the depth moved the darks as much as the brights"
        );
        let den = |d: f32| ColorSliceParams { density: 1.0, den_depth: d, ..Default::default() };
        let drop = |p: &ColorSliceParams, c: [f32; 3]| 1.0 - p.apply(c, WS)[0] / c[0];
        assert!(drop(&den(0.0), bright) < drop(&den(1.0), bright) * 0.2);
    }

    /// Balance hands luminance back to medium-saturation colours — the ones
    /// subtractive saturation darkens most visibly — and leaves greys and
    /// fully saturated colours to the model.
    #[test]
    fn balance_holds_the_luminance_of_medium_saturation_colours() {
        let mid = [0.6, 0.4, 0.3]; // q = 0.5
        let plain = ColorSliceParams { sat: 2.0, ..Default::default() };
        let held = ColorSliceParams { sat: 2.0, sat_balance: 1.0, ..Default::default() };
        let (y, yp, yh) = (luma(mid), luma(plain.apply(mid, WS)), luma(held.apply(mid, WS)));
        assert!(yp < y * 0.97, "the subtractive step did not darken it: {yp} vs {y}");
        assert!((yh - y).abs() < (y - yp) * 0.1, "balance 1 did not hold it: {yh} vs {y}");
        // Balance with no saturation change is no change.
        let idle = ColorSliceParams { sat_balance: 1.0, ..Default::default() };
        assert!(close3(idle.apply(mid, WS), mid, 1e-6));
    }

    /// Center moves where a vector sits, and is held so the vectors can
    /// never pass one another.
    #[test]
    fn center_moves_the_vector_and_is_held_in_order() {
        let moved = with(3, |v| v.center = 20.0); // green, to 140°
        let w = moved.weights(140.0, 1.0, WS);
        assert!((w[3] - 1.0).abs() < 1e-5, "green does not peak at its new centre: {w:?}");
        let wild = with(3, |v| v.center = 500.0);
        assert_eq!(wild.center_of(3), LEAN);
        let six: f32 = PART.iter().map(|i| wild.weights(170.0, 1.0, WS)[*i]).sum();
        assert!((six - 1.0).abs() < 1e-5, "a held centre broke the partition");
        assert_eq!(with(SKIN, |v| v.center = -99.0).center_of(SKIN), -SKIN_LEAN);
    }

    /// Near the neutral axis hue is noise, so the vectors fade out there.
    #[test]
    fn the_vectors_fade_toward_the_neutral_axis() {
        let p = ColorSliceParams::default();
        let near = p.weights(0.0, 0.01, WS);
        let far = p.weights(0.0, 0.5, WS);
        assert!(near[0] < 0.1 && (far[0] - 1.0).abs() < 1e-5, "{near:?} {far:?}");
    }

    /// The highlight is authored but only acts while the node is viewed.
    #[test]
    fn a_highlight_acts_only_when_armed() {
        let mut p = ColorSliceParams { highlight: Some(5), ..Default::default() };
        assert_eq!(p.live_highlight(), None, "authored is not armed");
        p.previewing = true;
        assert_eq!(p.live_highlight(), Some(5));
        assert!(p.highlight_weight(at_hue(240.0), WS) > 0.99);
        assert!(p.highlight_weight(at_hue(60.0), WS) < 0.01);
        p.highlight = Some(99);
        assert_eq!(p.live_highlight(), None, "a vector that is not there");
    }

    /// Nothing is topology, and the span carries what the shader reads.
    #[test]
    fn no_control_is_topology_and_the_span_is_laid_out() {
        let a = ColorSliceParams::default();
        let mut b = with(4, |v| *v = SliceVector { center: 9.0, hue: 30.0, sat: 1.7, density: 0.4 });
        b.sat = 1.3;
        b.highlight = Some(2);
        b.previewing = true;
        assert_eq!(a.topo_tag(), b.topo_tag());
        let mut v = [0.0f32; ColorSliceParams::VEC4_SLOTS * 4];
        b.pack(&mut v);
        assert_eq!(v[0], 1.3);
        assert_eq!((v[5], v[6]), (1.0, 2.0));
        let o = (2 + 4) * 4;
        assert_eq!(&v[o..o + 4], &[9.0, 30.0, 1.7, 0.4]);
    }
}
