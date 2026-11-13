//! SeedVR2: ByteDance-Seed's one-step diffusion video restoration and
//! upscaling, as a node whose result plays back like a bake.
//!
//! A 3B / 7B diffusion transformer is not something the engine runs. The
//! app renders the node's input to disk, runs the tool as a process on an
//! explicit click (the CorridorKey precedent, through the community
//! `numz/ComfyUI-SeedVR2_VideoUpscaler` CLI, which runs on CUDA and on
//! Apple Silicon), and the node then samples the result through its reader
//! slot — a Read's cost, and a loaded result CLEARS the slow taint above
//! it like a GraphBake (the pixels come off disk).
//!
//! What it renders:
//!
//! - **A result loaded** (`loaded`, transient, set by the app like a
//!   GraphBake's `baked`): the tool's frames, linear Rec.709 premultiplied
//!   (the driver's convention), at THEIR size — the node's format is the
//!   result's, larger than its input.
//! - **Nothing loaded**: input 0 passes through untouched.
//!
//! Licence (manifesto rule 10): SeedVR2's code and weights are Apache-2.0
//! (ByteDance-Seed), the CLI is Apache-2.0 (numz): free to deliver, not
//! tagged.

/// Which network.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SeedModel {
    /// 3B: the one that fits a laptop.
    #[default]
    B3,
    /// 7B: better detail, twice the memory and time.
    B7,
    /// 7B "sharp": the 7B fine-tuned for crisper output.
    B7Sharp,
}

impl SeedModel {
    pub const ALL: [SeedModel; 3] = [SeedModel::B3, SeedModel::B7, SeedModel::B7Sharp];

    pub fn label(self) -> &'static str {
        match self {
            SeedModel::B3 => "3B",
            SeedModel::B7 => "7B",
            SeedModel::B7Sharp => "7B sharp",
        }
    }

    /// The driver's word for it.
    pub fn arg(self) -> &'static str {
        match self {
            SeedModel::B3 => "3b",
            SeedModel::B7 => "7b",
            SeedModel::B7Sharp => "7b_sharp",
        }
    }
}

/// The tool's colour correction against the input (it fixes the colour
/// drift a diffusion model introduces).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SeedColour {
    #[default]
    Lab,
    Wavelet,
    WaveletAdaptive,
    Hsv,
    Adain,
    None,
}

impl SeedColour {
    pub const ALL: [SeedColour; 6] = [
        SeedColour::Lab,
        SeedColour::Wavelet,
        SeedColour::WaveletAdaptive,
        SeedColour::Hsv,
        SeedColour::Adain,
        SeedColour::None,
    ];

    pub fn arg(self) -> &'static str {
        match self {
            SeedColour::Lab => "lab",
            SeedColour::Wavelet => "wavelet",
            SeedColour::WaveletAdaptive => "wavelet_adaptive",
            SeedColour::Hsv => "hsv",
            SeedColour::Adain => "adain",
            SeedColour::None => "none",
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
pub struct SeedVrParams {
    /// Reader slot the result plays through.
    pub source: usize,
    pub model: SeedModel,
    /// Output size over input size (the short side; the tool keeps the
    /// aspect and rounds to what it can make).
    pub scale: f32,
    pub seed: u32,
    /// Frames the model sees at once: temporal consistency against memory.
    /// The tool wants 4n + 1.
    pub batch: u32,
    pub colour: SeedColour,
    /// TRANSIENT: a result is loaded in the reader slot.
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    pub loaded: bool,
    /// TRANSIENT: the comp frame the first result file holds.
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    pub start: Option<i64>,
}

impl Default for SeedVrParams {
    fn default() -> Self {
        Self {
            source: 0,
            model: SeedModel::B3,
            scale: 2.0,
            seed: 42,
            batch: 5,
            colour: SeedColour::Lab,
            loaded: false,
            start: None,
        }
    }
}

impl SeedVrParams {
    pub const NAME: &'static str = "SeedVR2";
    pub const INPUTS: usize = 1;
    pub const VEC4_SLOTS: usize = 0;

    pub fn new(source: usize) -> Self {
        Self { source, ..Default::default() }
    }

    /// Only whether a result is loaded changes the shader.
    pub fn topo_tag(&self) -> u64 {
        self.loaded as u64
    }

    /// The batch the tool is given: 4n + 1, at least 1.
    pub fn batch_arg(&self) -> u32 {
        let b = self.batch.max(1);
        (b - 1) / 4 * 4 + 1
    }

    /// The short side the tool is asked for, for an input's.
    pub fn resolution_for(&self, input: (u32, u32)) -> u32 {
        let short = input.0.min(input.1).max(1) as f32;
        ((short * self.scale.clamp(0.25, 8.0)).round() as u32 / 2 * 2).max(16)
    }

    /// Everything that shapes the pixels, for the app's result key.
    pub fn recipe(&self) -> (u8, u32, u32, u32, u8) {
        (
            self.model as u8,
            self.scale.to_bits(),
            self.seed,
            self.batch_arg(),
            self.colour as u8,
        )
    }

    /// The WGSL for a loaded result: the file sampled at the output pixel,
    /// then its input transform (linear to linear, safe premultiplied).
    pub fn emit_loaded(
        src: &str,
        pos: &str,
        idt: Option<&str>,
        body: &mut String,
        fresh: &mut dyn FnMut() -> String,
    ) -> String {
        use std::fmt::Write;
        let v = fresh();
        writeln!(body, "    let {v} = sample_px({src}, {pos});").unwrap();
        match idt {
            Some(f) => {
                let c = fresh();
                writeln!(body, "    let {c} = vec4f({f}({v}.rgb), {v}.a);").unwrap();
                c
            }
            None => v,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_loading_changes_the_topology() {
        let mut a = SeedVrParams::new(3);
        let b = a.clone();
        a.scale = 4.0;
        a.model = SeedModel::B7;
        assert_eq!(a.topo_tag(), b.topo_tag());
        a.loaded = true;
        assert_ne!(a.topo_tag(), b.topo_tag());
    }

    #[test]
    fn the_recipe_is_every_pixel_shaping_setting() {
        let a = SeedVrParams::default();
        let mut b = a.clone();
        b.source = 4;
        b.loaded = true;
        b.start = Some(1001);
        assert_eq!(a.recipe(), b.recipe());
        for f in [
            &|p: &mut SeedVrParams| p.model = SeedModel::B7Sharp,
            &|p: &mut SeedVrParams| p.scale = 3.0,
            &|p: &mut SeedVrParams| p.seed = 7,
            &|p: &mut SeedVrParams| p.batch = 9,
            &|p: &mut SeedVrParams| p.colour = SeedColour::Wavelet,
        ] as [&dyn Fn(&mut SeedVrParams); 5]
        {
            let mut c = a.clone();
            f(&mut c);
            assert_ne!(a.recipe(), c.recipe());
        }
        // 6 and 5 are the same batch to the tool.
        let mut d = a.clone();
        d.batch = 6;
        assert_eq!(a.recipe(), d.recipe());
    }

    #[test]
    fn the_tool_gets_its_batch_shape_and_an_even_resolution() {
        let p = |batch| SeedVrParams { batch, ..Default::default() }.batch_arg();
        assert_eq!((p(0), p(1), p(4), p(5), p(8), p(9)), (1, 1, 1, 5, 5, 9));
        let r = SeedVrParams { scale: 2.0, ..Default::default() }.resolution_for((1920, 1081));
        assert_eq!(r, 2162);
        assert_eq!(SeedVrParams { scale: 1.5, ..Default::default() }.resolution_for((640, 361)), 542);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn transients_are_never_saved() {
        let p = SeedVrParams { loaded: true, start: Some(7), ..SeedVrParams::new(2) };
        let j = serde_json::to_string(&p).unwrap();
        assert!(!j.contains("loaded") && !j.contains("start"), "{j}");
        let old: SeedVrParams = serde_json::from_str(r#"{"source":1}"#).unwrap();
        assert_eq!(old.scale, 2.0);
    }
}
