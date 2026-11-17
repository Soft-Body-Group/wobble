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
    /// A ONE-MODEL node: the catalog id it was made for, from the add
    /// menu's per-model entries. Its pipeline is `Pipeline::for_model`'s,
    /// fixed — the panel shows the model's own fields and no step editor,
    /// and the node is called by the model's name. `None` = the free-form
    /// pipeline node, steps and all.
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Option::is_none"))]
    pub model: Option<String>,
    /// How a run is multiplied (fan-out to other models): `genai::Vary`.
    /// Empty = one run, one take.
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "genai::Vary::is_empty"))]
    pub vary: genai::Vary,
    /// App mode (`genai::app`): the fields a simple form shows, and
    /// whether the node is showing it with everything else locked.
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "genai::AppForm::is_empty"))]
    pub app: genai::AppForm,
    /// TRANSIENT: a take is loaded in the reader slot. Set by the app on the
    /// engine-graph copy only; flipping it recompiles (sample vs pass
    /// through), exactly like a GraphBake's `baked`.
    #[cfg_attr(feature = "serde", serde(default, skip_serializing))]
    pub loaded: bool,
}

impl Default for GenAiParams {
    fn default() -> Self {
        Self {
            source: 0,
            pipeline: genai::Pipeline::default(),
            model: None,
            vary: Default::default(),
            app: Default::default(),
            loaded: false,
        }
    }
}

impl GenAiParams {
    pub const NAME: &'static str = "GenAI";
    /// The image the pipeline's Reference step reads, the matte its Matte
    /// step reads, the text its `prompt in` step reads (off a Prompt
    /// node, on the text pipe), and the second picture its `Second` step
    /// reads (a reference for colour, style or content).
    pub const INPUTS: usize = 4;
    /// One slot: the node's format, which the take is stretched over.
    pub const VEC4_SLOTS: usize = 1;

    pub fn new(source: usize) -> Self {
        Self { source, ..Default::default() }
    }

    /// A one-model node for catalog id `id`; the free-form node when the
    /// id is unknown (so a menu built from the catalog can never panic).
    pub fn for_model(source: usize, id: &str) -> Self {
        match genai::Pipeline::for_model(id) {
            Some(pipeline) => {
                Self { source, pipeline, model: Some(id.to_string()), ..Self::new(source) }
            }
            None => Self::new(source),
        }
    }

    /// The catalog model of a one-model node.
    pub fn remote_model(&self) -> Option<&'static genai::catalog::RemoteModel> {
        self.model.as_deref().and_then(genai::catalog::get)
    }

    /// Topology: only whether a take is loaded changes the shader. Which
    /// take, the pipeline, the prompts — none of it is code.
    pub fn topo_tag(&self) -> u64 {
        match (self.loaded, self.data()) {
            (false, _) => 0,
            (true, false) => 1,
            // Raw values skip the sRGB transform: different code.
            (true, true) => 2,
        }
    }

    /// The take on screen holds MEASUREMENTS (depth, normals): it plays
    /// back as stored, with no sRGB input transform.
    pub fn data(&self) -> bool {
        self.pipeline.picked().is_some_and(|t| t.data)
    }

    /// Rule 10, per instance. Tagged when the picture on screen was made by
    /// a model not free for commercial use, AND when the pipeline as it
    /// stands would run one — the licence is shown before anything is sent.
    pub fn commercial(&self) -> bool {
        self.pipeline.picked().is_none_or(|t| t.commercial)
            && self.pipeline.all_commercial()
            // A fan-out would run these too: said before anything is sent.
            && self.vary.all_commercial(&self.pipeline)
    }

    /// The node takes its RESULT's size as its format (an upscaler feeds
    /// Output), rather than its input's. The app's format walk asks.
    pub fn resizes(&self) -> bool {
        self.pipeline.resizes()
    }

    pub fn input_name(input: usize) -> &'static str {
        match input {
            0 => "image (the reference, or the clip)",
            1 => "matte (where to fill, a frame or a clip)",
            2 => "prompt (a String node's text)",
            _ => "second picture (a reference: colour, style, content)",
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
    fn a_data_take_is_its_own_code_and_skips_the_srgb_transform() {
        let mut p = GenAiParams::new(0);
        p.pipeline.takes.push(genai::Take {
            key: "k".into(),
            file: "result.png".into(),
            tick: 0,
            commercial: true,
            usd: None,
            label: String::new(),
            when: 0,
            data: true,
        });
        p.pipeline.take = 0;
        p.loaded = true;
        assert!(p.data());
        let still = GenAiParams { pipeline: genai::Pipeline::default(), ..p.clone() };
        assert_ne!(p.topo_tag(), still.topo_tag(), "raw values are different code");
    }

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
            data: false,
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
            data: false,
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
