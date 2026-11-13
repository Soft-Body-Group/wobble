//! HDRCNN: clipped highlights reconstructed by a CNN (Eilertsen et al.
//! 2017, BSD-3), as a node whose result plays back like a bake.
//!
//! The network runs in our own WGSL (`crate::nn`, built from the upstream
//! `.npz` by `crate::hdrcnn`), but never on an edit: the app renders the
//! node's input over its frame range, reconstructs every frame on a GPU
//! worker on a CLICK (`crate::hdrcnn::run`, the Upscale precedent), and the
//! node samples the result through its reader slot. A loaded result clears
//! the slow taint above it, like a GraphBake.
//!
//! - **A result loaded**: the reconstructed frames, linear Rec.709
//!   premultiplied, the input's size. Pixels the plate recorded properly
//!   (more than 5% under the clip, in the net's encoding) are the rendered
//!   input exactly; only the highlights are the network's.
//! - **Nothing loaded**: input 0 passes through untouched.
//!
//! `clip` is the level the plate clipped at, linear: the plate is divided
//! by it, clamped, and handed to the net; the prediction comes back times
//! it. It shapes the pixels, so it is in the recipe — a new clip is a new
//! render, never a uniform.

pub use crate::hdrcnn::Model as HdrCnnModel;

/// The node's authored state, plus two transient fields.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct HdrCnnParams {
    /// Reader slot the result plays through.
    pub source: usize,
    pub model: HdrCnnModel,
    /// The level the plate clipped at, linear.
    pub clip: f32,
    /// TRANSIENT: a result is loaded in the reader slot.
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    pub loaded: bool,
    /// TRANSIENT: the comp frame the first result file holds.
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    pub start: Option<i64>,
}

impl Default for HdrCnnParams {
    fn default() -> Self {
        Self { source: 0, model: HdrCnnModel::Regularized, clip: 1.0, loaded: false, start: None }
    }
}

impl HdrCnnParams {
    pub const NAME: &'static str = "HDRCNN";
    pub const INPUTS: usize = 1;
    pub const VEC4_SLOTS: usize = 0;

    pub fn new(source: usize) -> Self {
        Self { source, ..Default::default() }
    }

    /// Only whether a result is loaded changes the shader.
    pub fn topo_tag(&self) -> u64 {
        self.loaded as u64
    }

    /// The clip level the job is run at (never zero).
    pub fn clip_arg(&self) -> f32 {
        self.clip.max(1e-3)
    }

    /// Everything that shapes the pixels, for the app's result key.
    pub fn recipe(&self) -> (&'static str, u32) {
        (self.model.model_id(), self.clip_arg().to_bits())
    }

    /// The weights the picked model needs.
    pub fn models(&self) -> &'static [&'static str] {
        match self.model {
            HdrCnnModel::Regularized => &["hdrcnn-compr-regularized"],
            HdrCnnModel::Original => &["hdrcnn"],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_loading_changes_the_topology() {
        let mut a = HdrCnnParams::new(3);
        let b = a.clone();
        a.model = HdrCnnModel::Original;
        a.clip = 0.8;
        assert_eq!(a.topo_tag(), b.topo_tag());
        a.loaded = true;
        assert_ne!(a.topo_tag(), b.topo_tag());
    }

    #[test]
    fn the_recipe_is_the_model_and_the_clip() {
        let a = HdrCnnParams::default();
        let b = HdrCnnParams { source: 4, loaded: true, start: Some(1001), ..a.clone() };
        assert_eq!(a.recipe(), b.recipe());
        assert_ne!(a.recipe(), HdrCnnParams { model: HdrCnnModel::Original, ..a.clone() }.recipe());
        assert_ne!(a.recipe(), HdrCnnParams { clip: 0.9, ..a.clone() }.recipe());
        for m in HdrCnnModel::ALL {
            assert_eq!(HdrCnnParams { model: m, ..a.clone() }.models(), &[m.model_id()]);
        }
    }
}
