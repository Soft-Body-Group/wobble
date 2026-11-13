//! Render3D: cards drawn through a camera (`crate::scene3d`).
//!
//! In the editor a Render3D has two scene ports, the camera and the
//! cards. `scene3d::expand` lowers it before codegen: the scene wiring
//! goes, each card's PICTURE is wired to a port of its own (1..=n, port 0
//! left empty so every card is an aux texture and has a known format), and
//! the cameras and card poses for the frame land in [`Lowered`].
//!
//! The stage is ONE compute pass: a ray per pixel against every card, the
//! hits sorted by depth and laid front to back with "over", so depth order
//! is exact per pixel and cards may cut through one another. Motion blur
//! is shutter samples of the camera and the card poses INSIDE the same
//! pass — the card pictures are rendered once — averaged. The depth output
//! is the same render keeping its distance (along the camera's −Z, to the
//! first hit at least half opaque; the far clip where nothing is): a
//! lowered node of its own, so a picture-only render never pays for it.
//!
//! Card count picks the module (a binding per card, padded to a power of
//! two); everything else is a uniform.

use crate::codegen::Emit;
use crate::cost::Speed;
use crate::nodes::stage::{Bind, Dispatch, Pass, StageCx, StageDef, StageProgram, TexSpec, Wired};
use crate::nodes::NodeDef;
use crate::scene3d::{CameraParams, CardParams, MAX_CARDS, MAX_SAMPLES};

/// Editor ports.
pub const CAMERA_PORT: usize = 0;
pub const SCENE_PORT: usize = 1;

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct Render3dParams {
    /// The picture's format; a generator's row.
    pub size: (u32, u32),
    pub inherit: bool,
    /// Shutter samples; 1 is no motion blur.
    pub samples: u32,
    /// Shutter open, as a fraction of a frame (0.5 = 180 degrees).
    pub shutter: f32,
    /// Where the shutter sits about the frame: 0 centred, +0.5 opening on
    /// it.
    pub offset: f32,
    /// What `scene3d::expand` made of the scene for this frame. Never
    /// saved: it is derived per frame from the scene wiring.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub lowered: Option<Lowered>,
}

impl Default for Render3dParams {
    fn default() -> Self {
        Self { size: (1920, 1080), inherit: true, samples: 1, shutter: 0.5, offset: 0.0, lowered: None }
    }
}

/// A card as the stage sees it: one pose per shutter sample.
#[derive(Clone, Debug, PartialEq)]
pub struct LoweredCard {
    pub poses: Vec<CardParams>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Lowered {
    /// One camera per shutter sample; empty = no camera wired.
    pub cams: Vec<CameraParams>,
    /// The cards that draw, their pictures on ports 1..=n.
    pub cards: Vec<LoweredCard>,
    /// Cards the scene reached (more than `cards` past the cap).
    pub reached: usize,
    /// This copy writes depth, not colour.
    pub depth: bool,
}

impl Render3dParams {
    /// The shutter sample times, in frames about the frame — `[0]` without
    /// motion blur.
    pub fn shutter_times(&self) -> Vec<f32> {
        let n = self.samples.clamp(1, MAX_SAMPLES as u32);
        if n == 1 || self.shutter <= 0.0 {
            return vec![0.0];
        }
        (0..n)
            .map(|i| self.offset + self.shutter * ((i as f32 + 0.5) / n as f32 - 0.5))
            .collect()
    }

    pub fn input_count(&self) -> usize {
        match &self.lowered {
            Some(l) => 1 + l.cards.len(),
            None => 2,
        }
    }

    fn drawing(&self) -> Option<&Lowered> {
        self.lowered.as_ref().filter(|l| !l.cams.is_empty() && !l.cards.is_empty())
    }

    /// The stage, only when there is something to draw.
    pub fn stage(&self) -> Option<&Self> {
        self.drawing().map(|_| self)
    }
}

impl NodeDef for Render3dParams {
    const NAME: &'static str = "Render3D";
    const INPUTS: usize = 2;
    const MASK_INPUT: Option<usize> = None;
    const VEC4_SLOTS: usize = 0;

    /// Card count and which output: the binding layout. Poses, focal,
    /// samples are uniforms.
    fn topo_tag(&self) -> u64 {
        match self.drawing() {
            Some(l) => 0x3D_00 ^ ((l.cards.len() as u64) << 8) ^ ((l.depth as u64) << 16),
            None => 0x3D_00,
        }
    }

    fn pack(&self, _span: &mut [f32]) {}

    fn emit(&self, e: &mut Emit) -> String {
        if self.drawing().is_some() {
            e.stage_result()
        } else {
            "vec4f(0.0)".into()
        }
    }
}

/// Bindings a module is built for: the card count rounded up to a power
/// of two, so five modules cover every scene.
pub fn slots_for(cards: usize) -> usize {
    cards.clamp(1, MAX_CARDS).next_power_of_two()
}

/// vec4s of uniform: a header, two per shutter sample, then per card an
/// info row and three matrix rows per sample.
const HEAD: usize = 2;
const CAM_BASE: usize = HEAD;
const CARD_BASE: usize = CAM_BASE + 2 * MAX_SAMPLES;
const CARD_STRIDE: usize = 1 + 3 * MAX_SAMPLES;
pub const UNIFORM_VEC4: usize = CARD_BASE + MAX_CARDS * CARD_STRIDE;

fn module(slots: usize) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    writeln!(s, "struct U {{ v: array<vec4f, {UNIFORM_VEC4}> }}").unwrap();
    s.push_str("@group(0) @binding(0) var<uniform> u: U;\n@group(0) @binding(1) var samp: sampler;\n");
    for i in 0..slots {
        writeln!(s, "@group(0) @binding({}) var t{i}: texture_2d<f32>;", 2 + i).unwrap();
    }
    writeln!(s, "@group(0) @binding({}) var dst: texture_storage_2d<rgba16float, write>;", 2 + slots).unwrap();
    writeln!(s, "@group(0) @binding({}) var zdst: texture_storage_2d<rgba16float, write>;", 3 + slots).unwrap();
    s.push_str("fn card_tex(i: u32, uv: vec2f) -> vec4f {\n    switch i {\n");
    for i in 0..slots {
        writeln!(s, "        case {i}u: {{ return textureSampleLevel(t{i}, samp, uv, 0.0); }}").unwrap();
    }
    s.push_str("        default: { return vec4f(0.0); }\n    }\n}\n");
    write!(
        s,
        r#"
const CAM_BASE: u32 = {CAM_BASE}u;
const CARD_BASE: u32 = {CARD_BASE}u;
const CARD_STRIDE: u32 = {CARD_STRIDE}u;
const MAXC: u32 = {MAX_CARDS}u;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {{
    let dims = textureDimensions(dst);
    if (gid.x >= dims.x || gid.y >= dims.y) {{ return; }}
    let head = u.v[0];
    let p = vec2f(gid.xy) + vec2f(0.5);
    let far0 = u.v[CAM_BASE + 1u].y;
    // Outside the render's own format: nothing.
    if (p.x > head.z || p.y > head.w) {{
        textureStore(dst, gid.xy, vec4f(0.0));
        textureStore(zdst, gid.xy, vec4f(far0, far0, far0, 1.0));
        return;
    }}
    let nc = min(u32(head.x), MAXC);
    let ns = max(u32(head.y), 1u);
    let mid = ns / 2u;
    var acc = vec4f(0.0);
    var depth = far0;
    for (var s = 0u; s < ns; s = s + 1u) {{
        let cam = u.v[CAM_BASE + 2u * s];
        let clip = u.v[CAM_BASE + 2u * s + 1u];
        let dir = vec3f((p.x - cam.y) / cam.x, -(p.y - cam.z) / cam.x, -1.0);
        var ts: array<f32, {MAX_CARDS}>;
        var cs: array<vec4f, {MAX_CARDS}>;
        var k = 0u;
        for (var c = 0u; c < nc; c = c + 1u) {{
            let base = CARD_BASE + c * CARD_STRIDE;
            let info = u.v[base];
            let r0 = u.v[base + 1u + 3u * s];
            let r1 = u.v[base + 2u + 3u * s];
            let r2 = u.v[base + 3u + 3u * s];
            let o = vec3f(r0.w, r1.w, r2.w);
            let d = vec3f(dot(r0.xyz, dir), dot(r1.xyz, dir), dot(r2.xyz, dir));
            if (abs(d.z) < 1e-12) {{ continue; }}
            let t = -o.z / d.z;
            if (t < clip.x || t > clip.y) {{ continue; }}
            let q = o + d * t;
            let uv = vec2f(q.x + 0.5, 0.5 - q.y);
            if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) {{ continue; }}
            let col = card_tex(c, uv * info.xy) * info.z;
            if (col.a <= 0.0 && col.r == 0.0 && col.g == 0.0 && col.b == 0.0) {{ continue; }}
            var j = k;
            while (j > 0u && ts[j - 1u] > t) {{
                ts[j] = ts[j - 1u];
                cs[j] = cs[j - 1u];
                j = j - 1u;
            }}
            ts[j] = t;
            cs[j] = col;
            k = k + 1u;
        }}
        var c4 = vec4f(0.0);
        for (var j = 0u; j < k; j = j + 1u) {{
            c4 = c4 + (1.0 - c4.a) * cs[j];
            if (c4.a >= 0.9995) {{ break; }}
        }}
        acc = acc + c4;
        if (s == mid) {{
            depth = clip.y;
            for (var j = 0u; j < k; j = j + 1u) {{
                if (cs[j].a >= 0.5) {{ depth = ts[j]; break; }}
            }}
        }}
    }}
    textureStore(dst, gid.xy, acc / f32(ns));
    textureStore(zdst, gid.xy, vec4f(depth, depth, depth, 1.0));
}}
"#
    )
    .unwrap();
    s
}

static MODULES: std::sync::LazyLock<Vec<String>> =
    std::sync::LazyLock::new(|| [1, 2, 4, 8, 16].into_iter().map(module).collect());

/// The pass module for `slots` card bindings (a power of two to 16).
pub fn wgsl(slots: usize) -> &'static str {
    let i = slots.clamp(1, MAX_CARDS).trailing_zeros() as usize;
    &MODULES[i]
}

impl StageDef for Render3dParams {
    /// Every card picture; port 0 is never wired once lowered.
    fn aux_ports(&self, wired: Wired) -> Vec<usize> {
        match self.drawing() {
            Some(l) => (1..=l.cards.len()).filter(|p| wired.has(*p)).collect(),
            None => Vec::new(),
        }
    }

    fn program(&self, _canvas: (u32, u32), wired: Wired) -> StageProgram {
        let n = self.aux_ports(wired).len();
        if n == 0 {
            return StageProgram { textures: Vec::new(), passes: Vec::new(), result: "a".into() };
        }
        let slots = slots_for(n);
        let mut binds = vec![Bind::Uniform, Bind::Sampler];
        for i in 0..slots {
            binds.push(Bind::Tex(format!("aux{}", i.min(n - 1))));
        }
        binds.push(Bind::Storage("b".into()));
        binds.push(Bind::Storage("z".into()));
        let depth = self.drawing().is_some_and(|l| l.depth);
        StageProgram {
            textures: vec![TexSpec::full("b"), TexSpec::full("z")],
            passes: vec![Pass {
                label: "render3d",
                wgsl: wgsl(slots),
                binds,
                dispatch: Dispatch::Pixels("b".into()),
                uniform_floats: UNIFORM_VEC4 * 4,
            }],
            result: if depth { "z".into() } else { "b".into() },
        }
    }

    fn uniforms(&self, pass: usize, cx: &StageCx) -> Vec<f32> {
        self.uniforms_aux(pass, cx, &[])
    }

    /// `aux_fmt` is each card picture's format, then this node's own.
    fn uniforms_aux(&self, _pass: usize, cx: &StageCx, aux_fmt: &[(u32, u32)]) -> Vec<f32> {
        let mut v = vec![0.0f32; UNIFORM_VEC4 * 4];
        let Some(l) = self.drawing() else { return v };
        let n = l.cards.len();
        let own = aux_fmt.get(n).copied().unwrap_or(cx.canvas);
        let fmt = [own.0 as f32, own.1 as f32];
        let ns = l.cams.len().min(MAX_SAMPLES);
        v[0] = n.min(MAX_CARDS) as f32;
        v[1] = ns as f32;
        v[2] = fmt[0];
        v[3] = fmt[1];
        v[4] = cx.canvas.0 as f32;
        v[5] = cx.canvas.1 as f32;
        let canvas = [cx.canvas.0.max(1) as f32, cx.canvas.1.max(1) as f32];
        for (s, cam) in l.cams.iter().take(ns).enumerate() {
            let b = (CAM_BASE + 2 * s) * 4;
            v[b] = cam.focal_px(fmt[0]);
            v[b + 1] = fmt[0] * 0.5;
            v[b + 2] = fmt[1] * 0.5;
            v[b + 4] = cam.near.max(0.0);
            v[b + 5] = cam.far.max(cam.near);
        }
        for (c, card) in l.cards.iter().take(MAX_CARDS).enumerate() {
            let b = (CARD_BASE + c * CARD_STRIDE) * 4;
            let f = aux_fmt.get(c).copied().unwrap_or(cx.canvas);
            let f = [f.0.max(1) as f32, f.1.max(1) as f32];
            v[b] = f[0] / canvas[0];
            v[b + 1] = f[1] / canvas[1];
            v[b + 2] = card.poses.first().map(|p| p.opacity.clamp(0.0, 1.0)).unwrap_or(1.0);
            for (s, (pose, cam)) in card.poses.iter().zip(&l.cams).take(ns).enumerate() {
                let aspect = if pose.aspect > 0.0 { pose.aspect } else { f[0] / f[1] };
                let to_card = pose.xform.inverse_scaled([1.0, 1.0 / aspect, 1.0]);
                let cam_m = crate::scene3d::Xform { scale: [1.0; 3], ..cam.xform }.matrix();
                let m = crate::scene3d::mat_mul(&to_card, &cam_m);
                for r in 0..3 {
                    let o = b + 4 + (3 * s + r) * 4;
                    v[o..o + 4].copy_from_slice(&m[r]);
                }
            }
        }
        v
    }

    fn speed(&self) -> Speed {
        Speed::Slow
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_module_is_valid_wgsl() {
        for slots in [1, 2, 4, 8, 16] {
            let m = naga::front::wgsl::parse_str(wgsl(slots)).unwrap_or_else(|e| {
                panic!("{slots} slots: {}", e.emit_to_string(wgsl(slots)))
            });
            naga::valid::Validator::new(Default::default(), Default::default())
                .validate(&m)
                .unwrap_or_else(|e| panic!("{slots} slots: {e:?}"));
        }
    }

    #[test]
    fn slots_round_up_to_a_module() {
        assert_eq!(slots_for(1), 1);
        assert_eq!(slots_for(3), 4);
        assert_eq!(slots_for(9), 16);
        assert_eq!(slots_for(40), 16);
    }

    #[test]
    fn the_shutter_is_centred() {
        let p = Render3dParams { samples: 4, shutter: 1.0, ..Default::default() };
        assert_eq!(p.shutter_times(), vec![-0.375, -0.125, 0.125, 0.375]);
        assert_eq!(Render3dParams::default().shutter_times(), vec![0.0]);
    }
}
