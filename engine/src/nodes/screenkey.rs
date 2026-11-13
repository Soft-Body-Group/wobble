//! ScreenKey: a green / blue screen keyer, the Keylight shape of it.
//!
//! One node does the three jobs a screen key is: it **pulls a matte**, it
//! **despills** the foreground, and it hands both on as **premultiplied
//! RGBA** so a Blend over the background is the next node and nothing
//! else. Everything is pointwise, so it fuses into the pass with its
//! neighbours and costs a few dozen ALU per pixel — cyan in the graph.
//!
//! **The matte** is a colour-difference key, the Vlahos idea that every
//! screen keyer since Ultimatte is built on: in a green screen the green
//! channel stands above the other two, and *how far* above is a measure of
//! how much screen is in the pixel. What makes it usable rather than a
//! toy is two normalisations:
//!
//! - **to the screen colour**, sampled off the plate. A screen is never
//!   pure green; its own colour difference, as a fraction of its dominant
//!   channel (`k` below), is what "100% screen" means. A pixel's screen
//!   fraction is its colour difference divided by the colour difference a
//!   pure screen would have AT THAT PIXEL'S LEVEL;
//! - **to the pixel's own brightness**, which falls out of the same ratio:
//!   `(p_d - others) / (p_d * k)` is invariant to exposure, so a lit patch
//!   of screen and a shadowed one both read as 1, and a grey — any grey —
//!   reads as 0. A key that compares absolute differences (the naive
//!   `G - max(R, B)` with a threshold) keys the bright screen and leaves
//!   the dark corners, and that is the difference between the two.
//!
//! `others` is where Keylight's **screen balance** lives: the two
//! non-dominant channels are combined as a mix between their `min` and
//! their `max`. Toward `max` is conservative — only a pixel whose screen
//! channel beats BOTH others counts as screen — and holds a foreground
//! with one strong channel (a red coat) out of the key; toward `min` is
//! aggressive and takes more of the edge. The midpoint is Keylight's
//! default and ours.
//!
//! Then **screen gain** scales the fraction (more gain, more of the edge
//! pixels are called screen), and the **clip black / clip white** pair
//! remaps the matte so its core can be made solid and its screen made
//! clean without touching the soft edge between them. A **holdout** matte
//! forces alpha to 1 where it covers (the green shirt you want to keep), a
//! **garbage** matte forces 0 (the light stands).
//!
//! **Despill** limits the dominant channel to the same `others` the key
//! measured against — the amount by which a pixel is greener than its
//! own red and blue say it should be is exactly the screen bounce, and it
//! is subtracted. Subtracting light darkens: a blonde edge goes dull,
//! skin goes muddy. **Restore** puts the luminance that was taken back as
//! neutral light, in the working space's own luma weights, so a despilled
//! pixel keeps its brightness and only loses its cast. That is the whole of
//! what the "luminance preserving despill" gizmos do, and it is the right
//! default.
//!
//! Everything here is a uniform. The screen colour, the balance, the clips
//! and the output mode all ride the uniform block, so sampling the screen
//! with the eyedropper and dragging any control never recompiles.

use crate::codegen::Emit;
use crate::color::Colorspace;
use crate::nodes::NodeDef;

/// Topology tags are one flat u64 space shared by every kind; 0x5C is clear.
const TOPO_BASE: u64 = 0x5C;

/// Below this dominant-channel level a pixel is black, and black is
/// foreground: a ratio out of nothing is noise, not a measurement.
const BLACK: f32 = 1e-4;

/// What the node hands downstream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ScreenOut {
    /// Despilled foreground multiplied by the matte, matte in alpha: what a
    /// Blend wants.
    #[default]
    Premult,
    /// Despilled foreground as it is, matte in alpha.
    Straight,
    /// The matte in rgb, alpha 1: a picture of the key, readable by any
    /// mask port and drawn plainly by the viewer.
    Matte,
    /// The despilled foreground with the input's alpha: the despill alone.
    Despill,
}

impl ScreenOut {
    pub const ALL: [ScreenOut; 4] =
        [ScreenOut::Premult, ScreenOut::Straight, ScreenOut::Matte, ScreenOut::Despill];

    pub fn label(self) -> &'static str {
        match self {
            ScreenOut::Premult => "premultiplied",
            ScreenOut::Straight => "straight (unpremultiplied)",
            ScreenOut::Matte => "matte",
            ScreenOut::Despill => "despill only",
        }
    }

    /// The uniform's code for the mode.
    fn code(self) -> f32 {
        match self {
            ScreenOut::Premult => 0.0,
            ScreenOut::Straight => 1.0,
            ScreenOut::Matte => 2.0,
            ScreenOut::Despill => 3.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct ScreenKeyParams {
    /// The screen colour, working-space linear: sample it off the plate
    /// with the eyedropper. Its dominant channel decides whether this is
    /// a green or a blue (or a red) key.
    pub screen: [f32; 3],
    /// How the two non-screen channels combine into the level the screen
    /// channel is measured against: 0 = their min, 1 = their max, 0.5 =
    /// their mean. Keylight's screen balance.
    pub balance: f32,
    /// Multiplies the screen fraction before it becomes alpha: above 1
    /// calls more of the edge screen. Keylight's screen gain.
    pub gain: f32,
    /// Alpha at or below this becomes 0: cleans the screen.
    pub clip_black: f32,
    /// Alpha at or above this becomes 1: solidifies the core.
    pub clip_white: f32,
    /// How much of the screen bounce to take out of the foreground.
    pub despill: f32,
    /// How much of the luminance the despill took to put back, as neutral
    /// light.
    pub restore: f32,
    /// What goes downstream.
    pub output: ScreenOut,
}

impl Default for ScreenKeyParams {
    fn default() -> Self {
        Self {
            // A lit green screen, roughly, so a fresh node keys SOMETHING
            // on a green plate before the first sample.
            screen: [0.16, 0.55, 0.14],
            balance: 0.5,
            gain: 1.0,
            clip_black: 0.0,
            clip_white: 1.0,
            despill: 1.0,
            restore: 1.0,
            output: ScreenOut::Premult,
        }
    }
}

/// The channel roles the screen colour picks: the dominant channel and
/// the two others, as one-hot weight vectors (so the shader dots rather
/// than indexes).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Roles {
    pub dom: [f32; 3],
    pub o1: [f32; 3],
    pub o2: [f32; 3],
}

impl ScreenKeyParams {
    /// Which channel is the screen: the largest of the sampled colour. A
    /// tie or a non-finite sample falls to green, the common case.
    pub fn dominant(&self) -> usize {
        let s = self.screen;
        if !s.iter().all(|v| v.is_finite()) {
            return 1;
        }
        let mut d = 1;
        if s[0] > s[d] {
            d = 0;
        }
        if s[2] > s[d] {
            d = 2;
        }
        d
    }

    pub fn roles(&self) -> Roles {
        let d = self.dominant();
        let one = |i: usize| {
            let mut v = [0.0; 3];
            v[i] = 1.0;
            v
        };
        let (a, b) = match d {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        Roles { dom: one(d), o1: one(a), o2: one(b) }
    }

    /// The two non-screen channels combined by the balance.
    fn others(&self, o1: f32, o2: f32) -> f32 {
        let b = self.balance.clamp(0.0, 1.0);
        let (lo, hi) = (o1.min(o2), o1.max(o2));
        lo + (hi - lo) * b
    }

    /// The screen's own colour difference as a fraction of its dominant
    /// channel: what a screen fraction of 1 means. Floored so a sampled
    /// grey (no screen at all) keys nothing rather than everything.
    fn k(&self) -> f32 {
        let d = self.dominant();
        let s = self.screen;
        let (o1, o2) = match d {
            0 => (s[1], s[2]),
            1 => (s[0], s[2]),
            _ => (s[0], s[1]),
        };
        let sd = s[d].max(BLACK);
        ((sd - self.others(o1, o2)) / sd).max(0.02)
    }

    /// The raw matte for one pixel (before the holdout / garbage mattes
    /// and the input alpha): 1 foreground, 0 screen.
    pub fn matte(&self, rgb: [f32; 3]) -> f32 {
        let d = self.dominant();
        let (o1, o2) = match d {
            0 => (rgb[1], rgb[2]),
            1 => (rgb[0], rgb[2]),
            _ => (rgb[0], rgb[1]),
        };
        let pd = rgb[d];
        let frac = if pd > BLACK {
            ((pd - self.others(o1, o2)) / (pd * self.k()) * self.gain).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let a = 1.0 - frac;
        let (cb, cw) = (self.clip_black, self.clip_white.max(self.clip_black + 1e-3));
        ((a - cb) / (cw - cb)).clamp(0.0, 1.0)
    }

    /// The despilled colour for one pixel.
    pub fn despilled(&self, rgb: [f32; 3], working: Colorspace) -> [f32; 3] {
        let d = self.dominant();
        let (o1, o2) = match d {
            0 => (rgb[1], rgb[2]),
            1 => (rgb[0], rgb[2]),
            _ => (rgb[0], rgb[1]),
        };
        let spill = (rgb[d] - self.others(o1, o2)).max(0.0) * self.despill.clamp(0.0, 1.0);
        let w = working.primaries.luma_weights();
        let lost = spill * w[d] * self.restore.clamp(0.0, 1.0);
        let mut out = rgb;
        out[d] -= spill;
        for c in out.iter_mut() {
            *c += lost;
        }
        out
    }

    /// The CPU mirror of the generated shader, value for value: the
    /// input pixel, the holdout and garbage coverages (0 when unwired),
    /// and what comes out.
    pub fn apply(&self, c: [f32; 4], hold: f32, garbage: f32, working: Colorspace) -> [f32; 4] {
        let rgb = [c[0], c[1], c[2]];
        let mut a = self.matte(rgb);
        a = a.max(hold.clamp(0.0, 1.0));
        a *= 1.0 - garbage.clamp(0.0, 1.0);
        let a_out = a * c[3];
        let fg = self.despilled(rgb, working);
        match self.output {
            ScreenOut::Premult => [fg[0] * a_out, fg[1] * a_out, fg[2] * a_out, a_out],
            ScreenOut::Straight => [fg[0], fg[1], fg[2], a_out],
            ScreenOut::Matte => [a_out, a_out, a_out, 1.0],
            ScreenOut::Despill => [fg[0], fg[1], fg[2], c[3]],
        }
    }
}

/// The WGSL entry point, emitted once per shader (the luma weights are the
/// working space's, which is fixed for a shader).
fn wgsl(working: Colorspace, fns: &mut std::collections::BTreeMap<String, String>) -> &'static str {
    const NAME: &str = "screenkey_op";
    if fns.contains_key(NAME) {
        return NAME;
    }
    let w = working.primaries.luma_weights();
    let def = format!(
        r#"
// ScreenKey: a colour-difference matte normalised to the sampled screen
// and to the pixel's own level, then a luma-preserving despill.
// slot+0 = screen rgb, balance; slot+1 = gain, clip black, clip white,
// despill; slot+2 = restore, output mode; slot+3 = one-hot dominant
// channel; slot+4 = one-hot first other channel (the second is the rest).
fn {NAME}(c: vec4f, hold: f32, garbage: f32, slot: u32) -> vec4f {{
    let s = P.v[slot];
    let k1 = P.v[slot + 1u];
    let k2 = P.v[slot + 2u];
    let wd = P.v[slot + 3u].xyz;
    let w1 = P.v[slot + 4u].xyz;
    let w2 = vec3f(1.0) - wd - w1;
    let bal = clamp(s.w, 0.0, 1.0);
    // The screen's own colour difference, as a fraction of its level.
    let sd = max(dot(s.xyz, wd), {BLACK});
    let so1 = dot(s.xyz, w1);
    let so2 = dot(s.xyz, w2);
    let sk = max((sd - mix(min(so1, so2), max(so1, so2), bal)) / sd, 0.02);
    // The pixel's, against it.
    let pd = dot(c.rgb, wd);
    let po = mix(min(dot(c.rgb, w1), dot(c.rgb, w2)), max(dot(c.rgb, w1), dot(c.rgb, w2)), bal);
    var frac = 0.0;
    if (pd > {BLACK}) {{
        frac = clamp((pd - po) / (pd * sk) * k1.x, 0.0, 1.0);
    }}
    let cb = k1.y;
    let cw = max(k1.z, cb + 1e-3);
    var a = clamp((1.0 - frac - cb) / (cw - cb), 0.0, 1.0);
    a = max(a, clamp(hold, 0.0, 1.0));
    a = a * (1.0 - clamp(garbage, 0.0, 1.0));
    let a_out = a * c.a;
    // Despill: the screen channel down to what the others say, the lost
    // light back as neutral.
    let spill = max(pd - po, 0.0) * clamp(k1.w, 0.0, 1.0);
    let lost = spill * dot(wd, vec3f({:.7}, {:.7}, {:.7})) * clamp(k2.x, 0.0, 1.0);
    let fg = c.rgb - wd * spill + vec3f(lost);
    let mode = k2.y;
    if (mode < 0.5) {{
        return vec4f(fg * a_out, a_out);
    }} else if (mode < 1.5) {{
        return vec4f(fg, a_out);
    }} else if (mode < 2.5) {{
        return vec4f(vec3f(a_out), 1.0);
    }}
    return vec4f(fg, c.a);
}}
"#,
        w[0], w[1], w[2]
    );
    fns.insert(NAME.to_string(), def);
    NAME
}

impl NodeDef for ScreenKeyParams {
    const NAME: &'static str = "ScreenKey";
    /// Image, holdout matte, garbage matte.
    const INPUTS: usize = 3;
    /// Neither matte is a mask in the "mix the effect back" sense: one
    /// forces the key, the other forces the hole.
    const MASK_INPUT: Option<usize> = None;
    const VEC4_SLOTS: usize = 5;

    /// Everything is a uniform: one shape of code, never a recompile.
    fn topo_tag(&self) -> u64 {
        TOPO_BASE
    }

    fn pack(&self, v: &mut [f32]) {
        let r = self.roles();
        v[0..3].copy_from_slice(&self.screen);
        v[3] = self.balance;
        v[4] = self.gain;
        v[5] = self.clip_black;
        v[6] = self.clip_white;
        v[7] = self.despill;
        v[8] = self.restore;
        v[9] = self.output.code();
        v[12..15].copy_from_slice(&r.dom);
        v[16..19].copy_from_slice(&r.o1);
    }

    fn emit(&self, e: &mut Emit) -> String {
        let input = e.input(0);
        let hold = e.mask(1).map(|m| format!("{m}.x")).unwrap_or_else(|| "0.0".into());
        let garbage = e.mask(2).map(|m| format!("{m}.x")).unwrap_or_else(|| "0.0".into());
        let slot = e.slot();
        let working = e.working();
        let (_, fns) = e.color_helpers();
        let f = wgsl(working, fns);
        let v = e.fresh();
        e.line(format_args!("let {v} = {f}({input}, {hold}, {garbage}, {slot}u);"));
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WS: Colorspace = Colorspace::ACESCG;

    fn p() -> ScreenKeyParams {
        ScreenKeyParams::default()
    }

    fn luma(c: [f32; 3]) -> f32 {
        let w = WS.primaries.luma_weights();
        c[0] * w[0] + c[1] * w[1] + c[2] * w[2]
    }

    /// The sampled screen is screen at any exposure; greys are foreground
    /// at any exposure. That is the normalisation the node is about.
    #[test]
    fn the_screen_keys_and_greys_hold_whatever_the_exposure() {
        let k = p();
        for gain in [0.05, 0.3, 1.0, 4.0] {
            let s = k.screen.map(|v| v * gain);
            assert!(k.matte(s) < 1e-4, "screen at {gain}x: {}", k.matte(s));
            let g = [0.18 * gain; 3];
            assert!((k.matte(g) - 1.0).abs() < 1e-5, "grey at {gain}x: {}", k.matte(g));
        }
        // Black is foreground, not a division by nothing.
        assert_eq!(k.matte([0.0; 3]), 1.0);
    }

    /// Greener than the screen is still screen; half screen is half.
    #[test]
    fn the_matte_is_a_fraction_of_screen() {
        let k = p();
        assert!(k.matte([0.05, 0.9, 0.05]) < 1e-4);
        // A 50/50 mix of a grey and the screen at the same green level
        // sits well inside the soft edge.
        let s = k.screen;
        let g = [s[1]; 3];
        let mix = std::array::from_fn(|i| 0.5 * s[i] + 0.5 * g[i]);
        let a = k.matte(mix);
        assert!(a > 0.2 && a < 0.8, "half screen keyed at {a}");
    }

    /// Blue screens work by the same rule: the sampled colour picks the
    /// channel, and a strong-red foreground is not screen either way.
    #[test]
    fn a_blue_screen_keys_blue() {
        let mut k = p();
        k.screen = [0.1, 0.15, 0.6];
        assert_eq!(k.dominant(), 2);
        assert!(k.matte(k.screen.map(|v| v * 0.5)) < 1e-4);
        assert!(k.matte([0.05, 0.08, 0.3]) < 0.02, "near the screen: {}", k.matte([0.05, 0.08, 0.3]));
        assert!((k.matte([0.6, 0.1, 0.1]) - 1.0).abs() < 1e-5);
        // ...and the green one leaves a blue coat alone.
        assert!((p().matte([0.05, 0.1, 0.6]) - 1.0).abs() < 1e-5);
    }

    /// Balance toward max is the conservative key: a pixel whose screen
    /// channel beats one other channel but not both is foreground there
    /// and part screen at the mean.
    #[test]
    fn balance_decides_the_edge() {
        let c = [0.5, 0.4, 0.1];
        let mut k = p();
        k.balance = 1.0;
        assert!((k.matte(c) - 1.0).abs() < 1e-5, "max: {}", k.matte(c));
        k.balance = 0.0;
        assert!(k.matte(c) < 0.95, "min: {}", k.matte(c));
    }

    /// The clips remap the matte without moving its ends.
    #[test]
    fn clips_remap_the_soft_edge() {
        let mut k = p();
        let s = k.screen;
        let g = [s[1]; 3];
        let mix: [f32; 3] = std::array::from_fn(|i| 0.5 * s[i] + 0.5 * g[i]);
        let soft = k.matte(mix);
        k.clip_black = soft + 0.05;
        assert_eq!(k.matte(mix), 0.0);
        k.clip_black = 0.0;
        k.clip_white = soft - 0.05;
        assert_eq!(k.matte(mix), 1.0);
        assert_eq!(k.matte(s), 0.0);
        assert_eq!(k.matte(g), 1.0);
    }

    /// Despill takes the bounce and restore puts the brightness back.
    #[test]
    fn despill_removes_the_cast_and_restore_keeps_the_luma() {
        let mut k = p();
        let c = [0.3, 0.5, 0.3];
        k.restore = 0.0;
        let d = k.despilled(c, WS);
        assert!((d[1] - 0.3).abs() < 1e-6 && d[0] == 0.3 && d[2] == 0.3, "{d:?}");
        k.restore = 1.0;
        let r = k.despilled(c, WS);
        assert!((luma(r) - luma(c)).abs() < 1e-5, "{r:?} vs {c:?}");
        assert!((r[0] - r[2]).abs() < 1e-6 && r[1] < c[1], "{r:?}");
        // Nothing to take from a pixel with no cast.
        let n = [0.5, 0.2, 0.4];
        assert_eq!(k.despilled(n, WS), n);
        k.despill = 0.0;
        assert_eq!(k.despilled(c, WS), c);
    }

    /// Holdout forces foreground, garbage forces the hole, and the modes
    /// are what they say.
    #[test]
    fn mattes_and_modes() {
        let mut k = p();
        let s = [k.screen[0], k.screen[1], k.screen[2], 1.0];
        assert_eq!(k.apply(s, 0.0, 0.0, WS)[3], 0.0);
        assert_eq!(k.apply(s, 1.0, 0.0, WS)[3], 1.0);
        let g = [0.18, 0.18, 0.18, 1.0];
        assert_eq!(k.apply(g, 0.0, 1.0, WS)[3], 0.0);
        // Premult: a grey stays a grey at alpha 1, and a half-alpha input
        // halves it.
        assert_eq!(k.apply(g, 0.0, 0.0, WS), g);
        let h = k.apply([0.18, 0.18, 0.18, 0.5], 0.0, 0.0, WS);
        assert!((h[0] - 0.09).abs() < 1e-6 && h[3] == 0.5);
        k.output = ScreenOut::Straight;
        assert_eq!(k.apply([0.18, 0.18, 0.18, 0.5], 0.0, 0.0, WS), [0.18, 0.18, 0.18, 0.5]);
        k.output = ScreenOut::Matte;
        assert_eq!(k.apply(s, 0.0, 0.0, WS), [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(k.apply(g, 0.0, 0.0, WS), [1.0; 4]);
        k.output = ScreenOut::Despill;
        let out = k.apply([0.3, 0.5, 0.3, 0.7], 0.0, 0.0, WS);
        assert!(out[1] < 0.5 && out[3] == 0.7);
    }

    /// The pack lays the roles where the shader reads them.
    #[test]
    fn pack_puts_the_roles_in_the_last_two_slots() {
        let k = p();
        let mut v = [0.0; 20];
        k.pack(&mut v);
        assert_eq!(&v[12..15], &[0.0, 1.0, 0.0]);
        assert_eq!(&v[16..19], &[1.0, 0.0, 0.0]);
        assert_eq!(v[9], 0.0);
    }
}
