//! Upscale: neural super-resolution (Real-ESRGAN compact / x4plus, SPAN)
//! as a node whose result plays back like a bake.
//!
//! The networks run in our own WGSL (`crate::nn`, built from the upstream
//! checkpoints by `crate::upscale`), but at seconds to a minute a frame
//! they never run on an edit. The app renders the node's input over its
//! frame range, upscales every frame on a GPU worker on a CLICK
//! (`crate::upscale::run`, the Flow precedent), and the node then samples
//! the result through its reader slot — a Read's cost, and a loaded result
//! CLEARS the slow taint above it like a GraphBake.
//!
//! What it renders:
//!
//! - **A result loaded** (`loaded`, transient, set by the app like a
//!   GraphBake's `baked`): the upscaled frames, linear Rec.709
//!   premultiplied, at THEIR size — the node's format is the result's, its
//!   input's times the scale.
//! - **Nothing loaded**: input 0 passes through untouched, at its size.
//!
//! "2x" runs the model at its native 4x and halves the result with a
//! [1 3 3 1] decimation, which beats asking a 4x model for half the
//! detail. Colour: the models were trained on display-referred sRGB 0..1,
//! so they are given the picture clamped there, and what lies outside
//! (highlights, negatives) is upscaled bilinearly and added back — nothing
//! is clipped (see `nn::wgsl`). Alpha is upscaled bilinearly beside the
//! colour and the result premultiplied again.

pub use crate::upscale::Model as UpscaleModel;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum UpscaleScale {
    /// The model's own 4x.
    #[default]
    X4,
    /// 4x, then halved.
    X2,
}

impl UpscaleScale {
    pub fn factor(self) -> u32 {
        match self {
            UpscaleScale::X4 => 4,
            UpscaleScale::X2 => 2,
        }
    }
}

/// The node's authored state, plus two transient fields.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct UpscaleParams {
    /// Reader slot the result plays through.
    pub source: usize,
    pub model: UpscaleModel,
    pub scale: UpscaleScale,
    /// TRANSIENT: a result is loaded in the reader slot. Set by the app on
    /// the engine-graph copy only, like a GraphBake's `baked`.
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    pub loaded: bool,
    /// TRANSIENT: the comp frame the first result file holds.
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    pub start: Option<i64>,
}

impl UpscaleParams {
    pub const NAME: &'static str = "Upscale";
    pub const INPUTS: usize = 1;
    pub const VEC4_SLOTS: usize = 0;

    pub fn new(source: usize) -> Self {
        Self { source, ..Default::default() }
    }

    /// Only whether a result is loaded changes the shader.
    pub fn topo_tag(&self) -> u64 {
        self.loaded as u64
    }

    /// Everything that shapes the pixels, for the app's result key.
    pub fn recipe(&self) -> (&'static str, u32) {
        (self.model.model_id(), self.scale.factor())
    }

    /// The weights the PICKED model needs: only those are gated.
    pub fn models(&self) -> &'static [&'static str] {
        match self.model {
            UpscaleModel::Compact => &["realesr-general-x4v3"],
            UpscaleModel::X4Plus => &["realesrgan-x4plus"],
            UpscaleModel::Span => &["span-nomosuni-x4"],
        }
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
        let mut a = UpscaleParams::new(3);
        let b = a.clone();
        a.model = UpscaleModel::X4Plus;
        a.scale = UpscaleScale::X2;
        assert_eq!(a.topo_tag(), b.topo_tag());
        a.loaded = true;
        assert_ne!(a.topo_tag(), b.topo_tag());
    }

    #[test]
    fn the_recipe_is_the_model_and_the_scale() {
        let a = UpscaleParams::default();
        let b = UpscaleParams { source: 4, loaded: true, start: Some(1001), ..a.clone() };
        assert_eq!(a.recipe(), b.recipe());
        assert_ne!(a.recipe(), UpscaleParams { model: UpscaleModel::Span, ..a.clone() }.recipe());
        assert_ne!(a.recipe(), UpscaleParams { scale: UpscaleScale::X2, ..a.clone() }.recipe());
        for m in UpscaleModel::ALL {
            assert_eq!(UpscaleParams { model: m, ..a.clone() }.models(), &[m.model_id()]);
        }
    }

    #[test]
    fn the_loaded_shader_samples_and_transforms() {
        let mut body = String::new();
        let mut n = 0;
        let out = UpscaleParams::emit_loaded("src1", "pos", Some("cs_idt_x"), &mut body, &mut || {
            n += 1;
            format!("t{n}")
        });
        assert!(body.contains("sample_px(src1, pos)") && body.contains("cs_idt_x(t1.rgb)"), "{body}");
        assert_eq!(out, "t2");
    }

    #[cfg(feature = "serde")]
    #[test]
    fn transients_are_never_saved() {
        let p = UpscaleParams { loaded: true, start: Some(7), ..UpscaleParams::new(2) };
        let j = serde_json::to_string(&p).unwrap();
        assert!(!j.contains("loaded") && !j.contains("start"), "{j}");
        let old: UpscaleParams = serde_json::from_str(r#"{"source":1}"#).unwrap();
        assert_eq!(old.model, UpscaleModel::Compact);
    }
}
