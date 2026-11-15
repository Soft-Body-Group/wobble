//! Normals: NormalCrafter (Bin et al., ICCV 2025) as a node whose result
//! plays back like a bake — per-pixel surface normals from a plate,
//! temporally stable because the network is a video diffusion model that
//! sees the clip in overlapping windows.
//!
//! The Depth pattern: the app renders the node's input to disk, runs the
//! tool as a process on an explicit click, and the node then samples the
//! result through its reader slot. A loaded result CLEARS the slow taint
//! above it like a GraphBake: the pixels come off disk.
//!
//! What the result files hold: `R G B` = the unit normal's x y z in the
//! network's camera space — x right, y up, z towards the camera — on
//! -1..1, half float. The output is that, as numbers on the image pipe (a
//! normal is multi-channel data, which stays on the image pipe until it
//! earns a pipe of its own: `PortKind::Data`'s note), alpha 1. Two
//! uniforms re-express it without a re-run:
//!
//! - **flip x / flip y**: other conventions (DirectX-style y down, or a
//!   mirrored x) for whatever reads it;
//! - **encode 0..1**: `n * 0.5 + 0.5`, the familiar normal-map colours, for
//!   looking at or for a tool that wants them. Relight reads either.
//!
//! Licence (manifesto rule 10): the code is MIT. The weights are a
//! fine-tune of Stable Video Diffusion, so whatever their own card says
//! they inherit the Stability AI Community Licence (commercial use under
//! a revenue cap, registration, an acceptable-use policy): the node is
//! tagged `Restricted` (`cost::licence_of`), whether or not a result is
//! loaded, and a bake does not clear it.

/// The node's authored state, plus two transient fields.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct NormalsParams {
    /// Reader slot the result plays through.
    pub source: usize,
    /// The network's long side in pixels (a multiple of 64). Larger is
    /// finer, slower and hungrier: ~20 GB of GPU memory at 1024 upstream;
    /// the driver steps down on its own when it runs out.
    pub max_res: u32,
    /// Frames the network sees at once, and how far each window moves on:
    /// the overlap (window - stride) is what ties the windows together.
    pub window: u32,
    pub stride: u32,
    /// Uniforms: re-express the stored normal.
    pub flip_x: bool,
    pub flip_y: bool,
    pub encode01: bool,
    /// TRANSIENT: a result is loaded in the reader slot.
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    pub loaded: bool,
    /// TRANSIENT: the comp frame the first result file holds.
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    pub start: Option<i64>,
}

impl Default for NormalsParams {
    fn default() -> Self {
        Self {
            source: 0,
            max_res: 1024,
            window: 14,
            stride: 10,
            flip_x: false,
            flip_y: false,
            encode01: false,
            loaded: false,
            start: None,
        }
    }
}

impl NormalsParams {
    pub const NAME: &'static str = "Normals";
    pub const INPUTS: usize = 1;
    pub const VEC4_SLOTS: usize = 1;

    pub fn new(source: usize) -> Self {
        Self { source, ..Default::default() }
    }

    /// Only whether a result is loaded changes the shader.
    pub fn topo_tag(&self) -> u64 {
        self.loaded as u64
    }

    /// The long side the tool is given: a multiple of 64, 256..=2048.
    pub fn res_arg(&self) -> u32 {
        (self.max_res.clamp(256, 2048) + 32) / 64 * 64
    }

    /// Window and stride as the tool is given them: a window of 2..=25
    /// frames (SVD-XT was trained on 25), a stride inside it.
    pub fn window_arg(&self) -> (u32, u32) {
        let w = self.window.clamp(2, 25);
        (w, self.stride.clamp(1, w))
    }

    /// Everything that shapes the pixels, for the app's result key. NOT
    /// the flips or the encoding, which are uniforms.
    pub fn recipe(&self) -> (u32, u32, u32) {
        let (w, s) = self.window_arg();
        (self.res_arg(), w, s)
    }

    pub fn pack(&self, out: &mut [f32]) {
        out[0] = if self.flip_x { -1.0 } else { 1.0 };
        out[1] = if self.flip_y { -1.0 } else { 1.0 };
        out[2] = if self.encode01 { 1.0 } else { 0.0 };
        out[3] = 0.0;
    }

    /// The WGSL for a loaded result: the file's rgb sampled at the output
    /// pixel, flipped and encoded by the slot's uniforms, alpha 1. No input
    /// transform: these are numbers.
    pub fn emit_loaded(src: &str, pos: &str, slot: usize, body: &mut String, fresh: &mut dyn FnMut() -> String) -> String {
        use std::fmt::Write;
        let (n, o) = (fresh(), fresh());
        let q = format!("P.v[{slot}]");
        writeln!(body, "    let {n} = sample_px({src}, {pos}).rgb * vec3f({q}.x, {q}.y, 1.0);").unwrap();
        writeln!(body, "    let {o} = vec4f(select({n}, {n} * 0.5 + 0.5, {q}.z > 0.5), 1.0);").unwrap();
        o
    }

    /// The CPU mirror of `emit_loaded`.
    pub fn derive(&self, n: [f32; 3]) -> [f32; 3] {
        let v = [n[0] * if self.flip_x { -1.0 } else { 1.0 }, n[1] * if self.flip_y { -1.0 } else { 1.0 }, n[2]];
        if self.encode01 {
            v.map(|c| c * 0.5 + 0.5)
        } else {
            v
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_loading_changes_the_topology() {
        let mut a = NormalsParams::new(3);
        let b = a.clone();
        a.flip_y = true;
        a.encode01 = true;
        a.max_res = 512;
        assert_eq!(a.topo_tag(), b.topo_tag());
        a.loaded = true;
        assert_ne!(a.topo_tag(), b.topo_tag());
    }

    #[test]
    fn the_recipe_is_the_run_and_not_the_presentation() {
        let a = NormalsParams::default();
        let b = NormalsParams { flip_x: true, flip_y: true, encode01: true, source: 4, loaded: true, ..a.clone() };
        assert_eq!(a.recipe(), b.recipe());
        assert_ne!(a.recipe(), NormalsParams { max_res: 768, ..a.clone() }.recipe());
        assert_ne!(a.recipe(), NormalsParams { stride: 7, ..a.clone() }.recipe());
        assert_eq!(a.recipe(), NormalsParams { max_res: 1030, ..a.clone() }.recipe(), "same multiple of 64");
    }

    #[test]
    fn the_arguments_stay_in_range() {
        for r in [0, 100, 1000, 9000] {
            let v = NormalsParams { max_res: r, ..Default::default() }.res_arg();
            assert!(v % 64 == 0 && (256..=2048).contains(&v), "{r} -> {v}");
        }
        let (w, s) = NormalsParams { window: 99, stride: 99, ..Default::default() }.window_arg();
        assert_eq!((w, s), (25, 25));
        let (w, s) = NormalsParams { window: 0, stride: 0, ..Default::default() }.window_arg();
        assert_eq!((w, s), (2, 1));
    }

    #[test]
    fn the_loaded_shader_flips_and_encodes_by_uniform() {
        let mut body = String::new();
        let mut k = 0;
        let out = NormalsParams::emit_loaded("src0", "pos", 2, &mut body, &mut || {
            k += 1;
            format!("t{k}")
        });
        assert!(body.contains("sample_px(src0, pos).rgb * vec3f(P.v[2].x, P.v[2].y, 1.0)"), "{body}");
        assert!(body.contains("P.v[2].z > 0.5"), "{body}");
        assert_eq!(out, "t2");
        let mut v = [9.0; 4];
        NormalsParams { flip_y: true, encode01: true, ..Default::default() }.pack(&mut v);
        assert_eq!(v, [1.0, -1.0, 1.0, 0.0]);
        let p = NormalsParams { flip_y: true, encode01: true, ..Default::default() };
        assert_eq!(p.derive([0.0, 1.0, 0.0]), [0.5, 0.0, 0.5]);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn transients_are_never_saved() {
        let p = NormalsParams { loaded: true, start: Some(7), ..NormalsParams::new(2) };
        let j = serde_json::to_string(&p).unwrap();
        assert!(!j.contains("loaded") && !j.contains("start"), "{j}");
        let old: NormalsParams = serde_json::from_str(r#"{"source":1}"#).unwrap();
        assert_eq!(old.max_res, 1024);
    }
}
