//! DiffusionLight: DiffusionLight-Turbo (Chinchuthakun, Phongthawee et
//! al.) as a node whose result plays back like a bake — an HDR latlong
//! environment estimated from the plate alone, one per frame, to light and
//! reflect CG with where no HDRI was shot.
//!
//! The method inpaints a chrome ball into the frame with SDXL at three
//! exposures and unwraps it; none of that is something the engine runs. The
//! app renders the node's input to disk, runs the tool as a process on an
//! explicit click (the CorridorKey / SeedVR2 / Depth precedent), and the
//! node samples the result through its reader slot.
//!
//! What it renders:
//!
//! - **A result loaded** (`loaded`, transient, set by the app like a
//!   GraphBake's `baked`): the latlong, at ITS size (`2 x rows` by `rows`),
//!   linear Rec.709 in the plate's units. Blender's equirectangular, which
//!   is the Env node's: the centre is where the camera looks. It is
//!   relative to each frame's CAMERA, not to the world.
//! - **Nothing loaded**: black. The plate is not a latlong, and passing it
//!   on to an Env would light a scene with it.
//!
//! Every frame is its own estimate, so the raw result flickers. The tool's
//! driver smooths it over time, and — because in the shots this is for the
//! lighting really does change every frame — can take each frame's level
//! and colour from the plate itself rather than from the smoothed estimate
//! (`follow`; the maths is in `tools/wobble_difflight.py`). Smoothing and
//! `stride` are a second pass over estimates kept on disk: changing them
//! re-runs no diffusion.
//!
//! Licence (manifesto rule 10): DiffusionLight's code and both LoRAs are
//! MIT, SDXL and its depth ControlNet are CreativeML Open RAIL++-M
//! (commercial use allowed, with use restrictions on the model that do not
//! reach a lighting estimate), DPT-Large is Apache-2.0. Free to deliver,
//! not tagged.

/// The latlong's rows (it is twice as wide). The ball it is unwrapped from
/// is 256 px across, so more rows are a smoother picture, not more detail.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DiffLightSize {
    #[default]
    R256,
    R512,
    R1024,
}

impl DiffLightSize {
    pub const ALL: [DiffLightSize; 3] = [DiffLightSize::R256, DiffLightSize::R512, DiffLightSize::R1024];

    pub fn rows(self) -> u32 {
        match self {
            DiffLightSize::R256 => 256,
            DiffLightSize::R512 => 512,
            DiffLightSize::R1024 => 1024,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            DiffLightSize::R256 => "512 x 256",
            DiffLightSize::R512 => "1024 x 512",
            DiffLightSize::R1024 => "2048 x 1024",
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
pub struct DiffLightParams {
    /// Reader slot the result plays through.
    pub source: usize,
    /// Stops applied to the plate before the model sees it (it is shown an
    /// sRGB picture clipped at 1), and divided back out of the result.
    pub exposure: f32,
    /// One seed for every exposure and every frame.
    pub seed: u32,
    /// Denoising steps (the authors' 30).
    pub steps: u32,
    pub size: DiffLightSize,
    /// Estimate every `stride`th frame; the ones between are filled by the
    /// smoothing pass.
    pub stride: u32,
    /// Temporal Gaussian over the estimates, sigma in frames. 0 is off.
    pub smooth: f32,
    /// Each frame's level and colour come from the plate, unsmoothed; only
    /// the picture of the surroundings is steadied.
    pub follow: bool,
    /// TRANSIENT: a result is loaded in the reader slot.
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    pub loaded: bool,
    /// TRANSIENT: the comp frame the first result file holds.
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    pub start: Option<i64>,
}

impl Default for DiffLightParams {
    fn default() -> Self {
        Self {
            source: 0,
            exposure: 0.0,
            seed: 200,
            steps: 30,
            size: DiffLightSize::R256,
            stride: 1,
            smooth: 2.0,
            follow: true,
            loaded: false,
            start: None,
        }
    }
}

impl DiffLightParams {
    pub const NAME: &'static str = "DiffusionLight";
    pub const INPUTS: usize = 1;
    pub const VEC4_SLOTS: usize = 0;

    pub fn new(source: usize) -> Self {
        Self { source, ..Default::default() }
    }

    /// Loaded or not is the whole of the topology.
    pub fn topo_tag(&self) -> u64 {
        self.loaded as u64
    }

    pub fn steps_arg(&self) -> u32 {
        self.steps.clamp(4, 100)
    }

    pub fn stride_arg(&self) -> u32 {
        self.stride.clamp(1, 64)
    }

    pub fn smooth_arg(&self) -> f32 {
        if self.smooth.is_finite() {
            self.smooth.clamp(0.0, 50.0)
        } else {
            0.0
        }
    }

    /// Everything that shapes an ESTIMATE, for the app's result key: what
    /// the model is shown and how it is run. Not the smoothing pass.
    pub fn recipe(&self) -> (u32, u32, u32, u32) {
        (self.exposure.to_bits(), self.seed, self.steps_arg(), self.size.rows())
    }

    /// The smoothing pass's own directory under the estimate's: a new
    /// stride, sigma or `follow` is a new name beside the same estimates.
    pub fn out_name(&self) -> String {
        format!(
            "Env-k{}-s{}-{}",
            self.stride_arg(),
            (self.smooth_arg() * 100.0).round() as u32,
            if self.follow { "plate" } else { "plain" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loading_is_the_only_topology() {
        let mut a = DiffLightParams::new(3);
        let b = a.clone();
        a.exposure = 2.0;
        a.seed = 7;
        a.smooth = 9.0;
        a.stride = 4;
        a.follow = false;
        a.size = DiffLightSize::R512;
        assert_eq!(a.topo_tag(), b.topo_tag());
        a.loaded = true;
        assert_ne!(a.topo_tag(), b.topo_tag());
    }

    #[test]
    fn smoothing_is_a_new_name_and_never_a_new_estimate() {
        let a = DiffLightParams::default();
        for other in [
            DiffLightParams { smooth: 0.0, ..a.clone() },
            DiffLightParams { stride: 4, ..a.clone() },
            DiffLightParams { follow: false, ..a.clone() },
        ] {
            assert_eq!(a.recipe(), other.recipe());
            assert_ne!(a.out_name(), other.out_name());
        }
        for other in [
            DiffLightParams { exposure: 1.0, ..a.clone() },
            DiffLightParams { seed: 1, ..a.clone() },
            DiffLightParams { steps: 20, ..a.clone() },
            DiffLightParams { size: DiffLightSize::R512, ..a.clone() },
        ] {
            assert_ne!(a.recipe(), other.recipe());
            assert_eq!(a.out_name(), other.out_name());
        }
        let same = DiffLightParams { source: 9, loaded: true, start: Some(1001), ..a.clone() };
        assert_eq!((a.recipe(), a.out_name()), (same.recipe(), same.out_name()));
        assert_eq!(a.out_name(), "Env-k1-s200-plate");
    }

    #[test]
    fn the_arguments_are_clamped_to_what_the_tool_takes() {
        let p = DiffLightParams { steps: 0, stride: 0, smooth: f32::NAN, ..Default::default() };
        assert_eq!((p.steps_arg(), p.stride_arg(), p.smooth_arg()), (4, 1, 0.0));
        assert_eq!(DiffLightSize::R1024.rows(), 1024);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn transients_are_never_saved() {
        let p = DiffLightParams { loaded: true, start: Some(7), ..DiffLightParams::new(2) };
        let j = serde_json::to_string(&p).unwrap();
        assert!(!j.contains("loaded") && !j.contains("start"), "{j}");
        let old: DiffLightParams = serde_json::from_str(r#"{"source":1}"#).unwrap();
        assert!(old.follow && old.steps == 30);
    }
}
