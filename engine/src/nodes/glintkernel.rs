//! Glint Kernel: a generator that draws a star — the kernel a Convolve
//! turns highlights into glints with. Wire it into a Convolve's kernel
//! port (K) under a highlight key and every ping wears this star.
//!
//! The star sits at the middle of the node's format (on the centre pixel,
//! where Convolve reads a kernel's origin), its lengths measured in
//! fractions of half the format's short side, so a proxy halves it for
//! free and it means the same at any format. Every control is a uniform:
//! the ray COUNT included (a loop to [`MAX_RAYS`] that stops early), so it
//! keys and never recompiles.
//!
//! Each ray is a line from the centre with a Gaussian cross-section of
//! `thickness` px and an exponential fade along it, ending softly at
//! `length`. `alternate` shortens every other ray (the classic 4 + 4
//! star), `chroma` spreads the channels' lengths so the tips fringe red
//! out and blue in like real diffraction, and `core` adds a soft centre
//! dot.

use crate::codegen::Emit;
use crate::nodes::NodeDef;

/// Rays the loop can draw.
pub const MAX_RAYS: usize = 16;

const TOPO: u64 = 0x6C00;

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct GlintKernelParams {
    /// The node's format (see `inherit`).
    pub size: (u32, u32),
    /// Take the comp's canvas instead of `size`.
    pub inherit: bool,
    /// Rays, evenly spaced. Rounded, 1..=MAX_RAYS.
    pub rays: f32,
    /// Ray length, as a fraction of half the format's short side.
    pub length: f32,
    /// Width of a ray's core, in pixels (scaled by the proxy).
    pub thickness: f32,
    /// Fade along a ray: 0 holds to the tip, 1 is gone early.
    pub falloff: f32,
    /// Degrees, counter-clockwise.
    pub rotation: f32,
    /// Every other ray's length relative to the rest: 1 = all equal.
    pub alternate: f32,
    /// Centre dot strength.
    pub core: f32,
    /// Centre dot radius, as a fraction of the ray length.
    pub core_size: f32,
    /// Spread of the channels' lengths: red reaches further, blue less.
    pub chroma: f32,
    /// Overall multiplier.
    pub intensity: f32,
    /// Per-channel tint, linear.
    pub tint: [f32; 3],
}

impl Default for GlintKernelParams {
    fn default() -> Self {
        Self {
            size: (512, 512),
            inherit: false,
            rays: 4.0,
            length: 0.9,
            thickness: 1.5,
            falloff: 0.5,
            rotation: 0.0,
            alternate: 1.0,
            core: 1.0,
            core_size: 0.03,
            chroma: 0.0,
            intensity: 1.0,
            tint: [1.0, 1.0, 1.0],
        }
    }
}

impl GlintKernelParams {
    /// [`NodeDef::pack`] with the node's format and the proxy.
    pub fn pack_fmt(&self, v: &mut [f32], fmt: [f32; 2], proxy: f32) {
        let half = 0.5 * fmt[0].min(fmt[1]);
        v[0] = (fmt[0] * 0.5).floor() + 0.5;
        v[1] = (fmt[1] * 0.5).floor() + 0.5;
        v[2] = (self.length.max(0.0) * half).max(1e-3);
        v[3] = self.rays.round().clamp(1.0, MAX_RAYS as f32);
        v[4] = (self.thickness * proxy).max(0.35);
        v[5] = self.falloff.clamp(0.0, 1.0) * 6.0;
        v[6] = self.rotation.to_radians();
        v[7] = self.alternate.max(0.0);
        v[8] = self.core.max(0.0);
        v[9] = self.core_size.max(1e-4);
        v[10] = self.chroma;
        v[11] = 0.0;
        for c in 0..3 {
            v[12 + c] = self.tint[c] * self.intensity;
        }
        v[15] = 0.0;
    }
}

/// The star at pixel `p`. Slots: (centre, length px, rays), (thickness px,
/// falloff nepers, rotation, alternate), (core, core size, chroma, _),
/// (tint * intensity, _).
pub const WGSL: &str = r#"
fn glint_kernel(p: vec2f, slot: u32) -> vec4f {
    let a = P.v[slot];
    let b = P.v[slot + 1u];
    let c = P.v[slot + 2u];
    let tint = P.v[slot + 3u].rgb;
    let d = p - a.xy;
    let n = u32(a.w);
    let len = a.z;
    // Per-channel length: chroma spreads red out, blue in.
    let lens = len * max(vec3f(1.0) + c.z * vec3f(0.12, 0.0, -0.12), vec3f(0.05));
    var acc = vec3f(0.0);
    for (var i = 0u; i < 16u; i++) {
        if (i >= n) { break; }
        let ang = b.z + f32(i) * 6.2831853 / f32(n);
        // y is down in the picture: a positive angle turns counter-clockwise.
        let dir = vec2f(cos(ang), -sin(ang));
        let along = dot(d, dir);
        if (along <= 0.0) { continue; }
        let across = abs(d.x * dir.y - d.y * dir.x);
        let width = exp(-(across * across) / (b.x * b.x));
        var l = lens;
        if ((i & 1u) == 1u) { l = lens * b.w; }
        let t = vec3f(along) / max(l, vec3f(1e-3));
        let fade = exp(-b.y * t) * (vec3f(1.0) - smoothstep(vec3f(0.75), vec3f(1.0), t));
        acc += width * fade;
    }
    let r = length(d) / max(c.y * len, 1e-3);
    acc += vec3f(c.x * exp(-r * r));
    let rgb = acc * tint;
    return vec4f(rgb, clamp(max(rgb.r, max(rgb.g, rgb.b)), 0.0, 1.0));
}
"#;

impl NodeDef for GlintKernelParams {
    const NAME: &'static str = "Glint Kernel";
    const INPUTS: usize = 0;
    const MASK_INPUT: Option<usize> = None;
    const VEC4_SLOTS: usize = 4;

    fn topo_tag(&self) -> u64 {
        TOPO
    }

    fn pack(&self, v: &mut [f32]) {
        self.pack_fmt(v, [self.size.0 as f32, self.size.1 as f32], 1.0);
    }

    fn emit(&self, e: &mut Emit) -> String {
        let slot = e.slot();
        let pos = e.pos().to_string();
        let (_, fns) = e.color_helpers();
        fns.entry("glint_kernel".into()).or_insert_with(|| WGSL.to_string());
        let v = e.fresh();
        e.line(format_args!("let {v} = glint_kernel({pos}, {slot}u);"));
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_control_is_a_uniform() {
        let a = GlintKernelParams::default();
        let b = GlintKernelParams { rays: 9.0, length: 0.3, chroma: 1.0, ..a };
        assert_eq!(a.topo_tag(), b.topo_tag());
    }

    #[test]
    fn the_star_sits_on_the_centre_pixel() {
        let mut v = [0.0; 16];
        GlintKernelParams::default().pack_fmt(&mut v, [512.0, 256.0], 1.0);
        assert_eq!((v[0], v[1]), (256.5, 128.5));
        assert!((v[2] - 0.9 * 128.0).abs() < 1e-4, "length is of half the short side");
    }
}
