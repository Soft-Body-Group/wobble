//! LTX HDR: Lightricks' LTX-2.5 SDR-to-HDR IC-LoRA as a node whose result
//! plays back like a bake — generative highlight recovery that stays
//! consistent over a clip.
//!
//! A 22B video diffusion model is not something the engine runs. The app
//! renders the node's input to disk (ACEScg EXRs), runs the tool as a
//! process on an explicit click (the CorridorKey / SeedVR2 / Depth
//! precedent), and the node samples the result through its reader slot.
//!
//! **The result is composited back only where the input is clipped.** The
//! model round-trips the WHOLE picture through its VAE, which shifts every
//! pixel a little; a pixel the plate recorded properly must not pay for
//! that. So the loaded node is `mix(input, result, m)`, `m` a soft mask
//! off the input's brightest channel: 0 under `clip * (1 - softness)`, 1
//! at `clip`. `raw` shows the result alone (and is then the only mode that
//! reads no plate). Unloaded the node passes its input through.
//!
//! The model may be run at a fraction of the plate's size (`scale`): the
//! result is stretched back over the plate's format, which a highlight —
//! smooth by nature — barely notices, and a 22B model at HD does not fit a
//! 48 GB Mac.
//!
//! Licence (manifesto rule 10): the LTX-2.x Community Licence — free
//! commercial use under $10M annual revenue, with use restrictions and a
//! provenance clause. Not plain non-commercial; tagged `Restricted`, which
//! warns at a Write like non-commercial does, and a bake does not clear it.

/// The fraction of the plate's size the model runs at.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LtxScale {
    #[default]
    Full,
    Half,
    Quarter,
}

impl LtxScale {
    pub const ALL: [LtxScale; 3] = [LtxScale::Full, LtxScale::Half, LtxScale::Quarter];

    pub fn label(self) -> &'static str {
        match self {
            LtxScale::Full => "full size",
            LtxScale::Half => "proxy 1/2",
            LtxScale::Quarter => "proxy 1/4",
        }
    }

    pub fn factor(self) -> f32 {
        match self {
            LtxScale::Full => 1.0,
            LtxScale::Half => 0.5,
            LtxScale::Quarter => 0.25,
        }
    }
}

/// How the tool keeps its weights while it runs (`--offload`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LtxOffload {
    /// Everything on the GPU: the fastest, and the most memory.
    #[default]
    None,
    /// Weights pinned in system RAM, streamed to the GPU per layer.
    Cpu,
    /// Weights read from disk on demand: the least memory, the slowest.
    Disk,
}

impl LtxOffload {
    pub const ALL: [LtxOffload; 3] = [LtxOffload::None, LtxOffload::Cpu, LtxOffload::Disk];

    pub fn label(self) -> &'static str {
        match self {
            LtxOffload::None => "none (all on the GPU)",
            LtxOffload::Cpu => "cpu",
            LtxOffload::Disk => "disk",
        }
    }

    /// The tool's `--offload` value.
    pub fn arg(self) -> &'static str {
        match self {
            LtxOffload::None => "none",
            LtxOffload::Cpu => "cpu",
            LtxOffload::Disk => "disk",
        }
    }
}

/// The node's authored state, plus two transient fields.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct LtxHdrParams {
    /// Reader slot the result plays through.
    pub source: usize,
    /// The value the plate clipped at, in the working space: where the
    /// result is composited in.
    pub clip: f32,
    /// How far under the clip the composite fades in, as a fraction of it.
    pub softness: f32,
    /// Show the result alone, every pixel through the model's VAE.
    pub raw: bool,
    pub scale: LtxScale,
    /// Twice the frames internally, every other one kept: steadier, ~2x
    /// the time (`--high-quality`).
    pub high_quality: bool,
    /// Seam keyframes (the tool's default; off is plain IC-LoRA).
    pub keyframes: bool,
    pub seed: u32,
    pub offload: LtxOffload,
    /// TRANSIENT: a result is loaded in the reader slot.
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    pub loaded: bool,
    /// TRANSIENT: the comp frame the first result file holds.
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    pub start: Option<i64>,
}

impl Default for LtxHdrParams {
    fn default() -> Self {
        Self {
            source: 0,
            clip: 1.0,
            softness: 0.1,
            raw: false,
            scale: LtxScale::Full,
            high_quality: false,
            keyframes: true,
            seed: 42,
            offload: LtxOffload::None,
            loaded: false,
            start: None,
        }
    }
}

impl LtxHdrParams {
    pub const NAME: &'static str = "LTX HDR";
    pub const INPUTS: usize = 1;
    /// The plate's format (the result is stretched over it), then the
    /// clip level and the knee.
    pub const VEC4_SLOTS: usize = 1;

    pub fn new(source: usize) -> Self {
        Self { source, ..Default::default() }
    }

    /// Loaded or not, and raw or composited: those change what is sampled.
    /// The clip and softness are uniforms.
    pub fn topo_tag(&self) -> u64 {
        self.loaded as u64 | (self.raw as u64) << 1
    }

    /// Does the loaded node read no plate at all? Only raw: the composite
    /// needs the input under the result.
    pub fn reads_no_plate(&self) -> bool {
        self.loaded && self.raw
    }

    /// Everything that shapes the result's pixels, for the app's result
    /// key. NOT the composite (clip, softness, raw: uniforms and a branch
    /// over the same files) and not the offload (where the weights sit).
    pub fn recipe(&self) -> (u8, bool, bool, u32) {
        (self.scale as u8, self.high_quality, self.keyframes, self.seed)
    }

    /// The clip level and the knee the composite fades in from.
    pub fn knee(&self) -> (f32, f32) {
        let clip = self.clip.max(1e-4);
        (clip, clip * (1.0 - self.softness.clamp(0.0, 0.95)))
    }

    /// Output px -> the file's uv is the plate's format; then the knee.
    /// The format is packed as x/y and the clip/knee pair rides in z/w.
    pub fn pack(&self, fmt: [f32; 2], out: &mut [f32]) {
        let (clip, knee) = self.knee();
        out[0] = fmt[0].max(1.0);
        out[1] = fmt[1].max(1.0);
        out[2] = clip;
        out[3] = knee;
    }

    /// The WGSL for a loaded result: the file stretched over the plate's
    /// format and through `idt` (ACEScg -> working), then either itself
    /// (`raw`) or composited over `input` by the clip mask. `input` is the
    /// plate's expression, None in raw mode.
    pub fn emit_loaded(
        src: &str,
        pos: &str,
        slot: usize,
        idt: Option<&str>,
        input: Option<&str>,
        body: &mut String,
        fresh: &mut dyn FnMut() -> String,
    ) -> String {
        use std::fmt::Write;
        let q = fresh();
        writeln!(body, "    let {q} = {pos} / P.v[{slot}].xy * vec2f(textureDimensions({src}));").unwrap();
        let v = fresh();
        writeln!(body, "    let {v} = sample_px({src}, {q});").unwrap();
        let h = fresh();
        match idt {
            Some(f) => writeln!(body, "    let {h} = {f}({v}.rgb);").unwrap(),
            None => writeln!(body, "    let {h} = {v}.rgb;").unwrap(),
        }
        let Some(inp) = input else {
            let o = fresh();
            writeln!(body, "    let {o} = vec4f({h}, {v}.a);").unwrap();
            return o;
        };
        let m = fresh();
        writeln!(
            body,
            "    let {m} = smoothstep(P.v[{slot}].w, max(P.v[{slot}].z, P.v[{slot}].w + 1e-6), max({inp}.r, max({inp}.g, {inp}.b)));"
        )
        .unwrap();
        let o = fresh();
        writeln!(body, "    let {o} = vec4f(mix({inp}.rgb, {h}, {m}), {inp}.a);").unwrap();
        o
    }

    /// Frames the tool is given for a clip of `n`: the next 8k + 1, the
    /// tail held on the last frame (and trimmed back off the result).
    pub fn padded_frames(n: usize) -> usize {
        if n <= 1 {
            1
        } else {
            (n - 1).div_ceil(8) * 8 + 1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_composite_is_uniforms_and_loading_or_raw_is_topology() {
        let mut a = LtxHdrParams::new(3);
        let b = a.clone();
        a.clip = 0.8;
        a.softness = 0.3;
        a.seed = 7;
        a.scale = LtxScale::Half;
        assert_eq!(a.topo_tag(), b.topo_tag());
        a.loaded = true;
        assert_ne!(a.topo_tag(), b.topo_tag());
        let loaded = a.topo_tag();
        a.raw = true;
        assert_ne!(a.topo_tag(), loaded);
    }

    #[test]
    fn the_recipe_is_what_shapes_the_result_and_not_the_composite() {
        let a = LtxHdrParams::default();
        let same = LtxHdrParams {
            clip: 0.5,
            softness: 0.4,
            raw: true,
            offload: LtxOffload::Disk,
            source: 9,
            loaded: true,
            start: Some(1001),
            ..a.clone()
        };
        assert_eq!(a.recipe(), same.recipe());
        for other in [
            LtxHdrParams { scale: LtxScale::Half, ..a.clone() },
            LtxHdrParams { high_quality: true, ..a.clone() },
            LtxHdrParams { keyframes: false, ..a.clone() },
            LtxHdrParams { seed: 1, ..a.clone() },
        ] {
            assert_ne!(a.recipe(), other.recipe());
        }
    }

    #[test]
    fn frames_pad_to_eight_k_plus_one() {
        for (n, want) in [(0, 1), (1, 1), (2, 9), (9, 9), (10, 17), (48, 49), (97, 97), (100, 105)] {
            assert_eq!(LtxHdrParams::padded_frames(n), want, "{n}");
            assert_eq!((want - 1) % 8, 0);
        }
    }

    #[test]
    fn the_shader_composites_by_the_mask_or_shows_the_result_raw() {
        let mut n = 0;
        let mut fresh = || {
            n += 1;
            format!("t{n}")
        };
        let mut body = String::new();
        let out = LtxHdrParams::emit_loaded("src0", "pos", 5, Some("idt"), Some("inp"), &mut body, &mut fresh);
        assert!(body.contains("idt(t2.rgb)"), "{body}");
        assert!(body.contains("smoothstep(P.v[5].w"), "{body}");
        assert!(body.contains("mix(inp.rgb, t3, t4)") && body.contains("inp.a"), "{body}");
        assert_eq!(out, "t5");
        let mut body = String::new();
        let out = LtxHdrParams::emit_loaded("src0", "pos", 5, None, None, &mut body, &mut fresh);
        assert!(!body.contains("smoothstep"), "raw reads no plate: {body}");
        assert!(body.contains(&format!("let {out} = vec4f(")));
        let mut v = [0.0; 4];
        LtxHdrParams { clip: 2.0, softness: 0.25, ..Default::default() }.pack([1920.0, 1080.0], &mut v);
        assert_eq!(v, [1920.0, 1080.0, 2.0, 1.5]);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn transients_are_never_saved() {
        let p = LtxHdrParams { loaded: true, start: Some(7), ..LtxHdrParams::new(2) };
        let j = serde_json::to_string(&p).unwrap();
        assert!(!j.contains("loaded") && !j.contains("start"), "{j}");
        let old: LtxHdrParams = serde_json::from_str(r#"{"source":1}"#).unwrap();
        assert!(old.keyframes && old.clip == 1.0);
    }
}
