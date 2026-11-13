//! The GenAI node: a generative-AI pipeline whose result plays back like a
//! bake.
//!
//! The node carries a [`genai::Pipeline`] — a small step graph the app draws
//! INSIDE the node — and the takes that running it produced. The engine
//! never runs anything: the pipeline is data, the app runs it on an
//! explicit click (see the `genai` crate), and the engine only ever sees the
//! picked take as a file in this node's reader slot.
//!
//! What it renders:
//!
//! - **A take loaded** (`loaded`, transient, set by the app like a
//!   GraphBake's `baked`): the picture, STRETCHED over the node's own
//!   format. A generated picture comes back at the model's size — a
//!   megapixel, a reference capped at two — and the point of the node is a
//!   plate that lines up with the one it was made from, so it fills the
//!   format its input has (or the script's, unwired) rather than sitting in
//!   the top-left corner at whatever size the model chose.
//! - **Nothing loaded**: input 0 passes through untouched, so dropping the
//!   node into a chain changes nothing until something is generated.
//!
//! The file is display-referred sRGB (every model we call is), so the
//! reader leaf applies the sRGB input transform — on UNPREMULTIPLIED
//! colour, because a cutout's soft edge is exactly where a non-linear curve
//! on premultiplied values goes wrong.
//!
//! Cost, in both currencies `cost.rs` keeps: fast (a reader like any
//! other), and a licence taken from the take on screen and from the models
//! the pipeline names (`genai::catalog`, rule 10).

/// The node's authored state, plus one transient flag.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GenAiParams {
    /// Reader slot the picked take plays through.
    pub source: usize,
    #[cfg_attr(feature = "serde", serde(default))]
    pub pipeline: genai::Pipeline,
    /// TRANSIENT: a take is loaded in the reader slot. Set by the app on the
    /// engine-graph copy only; flipping it recompiles (sample vs pass
    /// through), exactly like a GraphBake's `baked`.
    #[cfg_attr(feature = "serde", serde(default, skip_serializing))]
    pub loaded: bool,
}

impl Default for GenAiParams {
    fn default() -> Self {
        Self { source: 0, pipeline: genai::Pipeline::default(), loaded: false }
    }
}

impl GenAiParams {
    pub const NAME: &'static str = "GenAI";
    /// The image the pipeline's Reference step reads, the matte its Matte
    /// step reads, and the text its `prompt in` step reads (off a Prompt
    /// node, on the text pipe).
    pub const INPUTS: usize = 3;
    /// One slot: the node's format, which the take is stretched over.
    pub const VEC4_SLOTS: usize = 1;

    pub fn new(source: usize) -> Self {
        Self { source, ..Default::default() }
    }

    /// Topology: only whether a take is loaded changes the shader. Which
    /// take, the pipeline, the prompts — none of it is code.
    pub fn topo_tag(&self) -> u64 {
        if self.loaded {
            1
        } else {
            0
        }
    }

    /// Rule 10, per instance. Tagged when the picture on screen was made by
    /// a model not free for commercial use, AND when the pipeline as it
    /// stands would run one — the licence is shown before anything is sent.
    pub fn commercial(&self) -> bool {
        self.pipeline.picked().is_none_or(|t| t.commercial) && self.pipeline.all_commercial()
    }

    pub fn input_name(input: usize) -> &'static str {
        match input {
            0 => "image (the reference)",
            1 => "matte (where to fill)",
            _ => "prompt (a String node's text)",
        }
    }

    /// Output px -> the file's uv: the node's format.
    pub fn pack(fmt: [f32; 2], out: &mut [f32]) {
        out[0] = fmt[0].max(1.0);
        out[1] = fmt[1].max(1.0);
        out[2] = 0.0;
        out[3] = 0.0;
    }

    /// The WGSL for a loaded take: stretch the file over the format, then
    /// the sRGB input transform on straight colour. `idt` is the transform
    /// function codegen generated for this node (None when the working
    /// space happens to be sRGB-encoded, which it never is, but the
    /// identity case is still honest).
    pub fn emit_loaded(
        src: &str,
        pos: &str,
        slot: usize,
        idt: Option<&str>,
        body: &mut String,
        fresh: &mut dyn FnMut() -> String,
    ) -> String {
        use std::fmt::Write;
        let q = fresh();
        writeln!(
            body,
            "    let {q} = {pos} / P.v[{slot}].xy * vec2f(textureDimensions({src}));"
        )
        .unwrap();
        let v = fresh();
        writeln!(body, "    let {v} = sample_px({src}, {q});").unwrap();
        let Some(f) = idt else { return v };
        let a = fresh();
        writeln!(body, "    let {a} = max({v}.a, 1e-6);").unwrap();
        let c = fresh();
        writeln!(
            body,
            "    let {c} = select(vec4f(0.0), vec4f({f}({v}.rgb / {a}) * {v}.a, {v}.a), {v}.a > 0.0);"
        )
        .unwrap();
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_loading_a_take_changes_the_topology() {
        let mut a = GenAiParams::new(3);
        let b = a.clone();
        if let genai::Op::Prompt { text } = &mut a.pipeline.steps[0].op {
            *text = "a different prompt".into();
        }
        assert_eq!(a.topo_tag(), b.topo_tag(), "a prompt is not code");
        a.loaded = true;
        assert_ne!(a.topo_tag(), b.topo_tag());
    }

    #[test]
    fn a_non_commercial_take_stays_tagged_after_the_pipeline_changes() {
        let mut p = GenAiParams::new(0);
        assert!(p.commercial());
        p.pipeline.takes.push(genai::Take {
            key: "k".into(),
            file: "result.png".into(),
            tick: 0,
            commercial: false,
            usd: None,
            label: String::new(),
            when: 0,
        });
        p.pipeline.take = 0;
        // The pipeline itself is all-commercial now; the pixels on screen
        // were not made that way.
        assert!(p.pipeline.all_commercial());
        assert!(!p.commercial());
    }

    #[test]
    fn a_video_take_is_placed_at_its_capture_tick_and_a_still_is_not() {
        let mut p = GenAiParams::new(0);
        let take = |file: &str| genai::Take {
            key: "k".into(),
            file: file.into(),
            tick: 1012,
            commercial: true,
            usd: None,
            label: String::new(),
            when: 0,
        };
        p.pipeline.takes = vec![take("result.png"), take("result.mp4")];
        p.pipeline.take = 0;
        assert_eq!(crate::graph::NodeKind::GenAi(p.clone()).placement(), None);
        p.pipeline.take = 1;
        assert_eq!(crate::graph::NodeKind::GenAi(p).placement(), Some(1012));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn loaded_is_never_saved() {
        let p = GenAiParams { loaded: true, ..GenAiParams::new(2) };
        let j = serde_json::to_string(&p).unwrap();
        assert!(!j.contains("loaded"), "{j}");
        let back: GenAiParams = serde_json::from_str(&j).unwrap();
        assert!(!back.loaded);
        assert_eq!(back.source, 2);
    }
}
