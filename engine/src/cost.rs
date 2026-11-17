//! What a node costs, in the two currencies that are not pixels: **frame
//! time** and **rights**.
//!
//! Both are per-node facts here, and both TAINT DOWNSTREAM — a comp is only
//! as fast as its slowest branch, and only as freely usable as its most
//! restricted one. The propagation lives in the app (`costviz`), because it
//! is a property of a graph being drawn; this module is the classification
//! everything else agrees on.
//!
//! # Speed
//!
//! [`Speed::Slow`] means exactly one thing: **the node takes its own GPU
//! pass or passes.** It cannot fuse into the main shader, so it costs an
//! extra full-canvas dispatch (and usually a texture) every frame, whether
//! or not anything about it changed.
//!
//! That definition is not a judgement about how heavy a node feels — it is
//! the same question `codegen::filter_stages` asks when it decides to split
//! the graph, and `slow_matches_codegen` in the tests below asserts the two
//! answers agree for every variant. A new stage kind that forgets to come
//! back here fails that test rather than quietly rendering cyan.
//!
//! Manifesto rule 8: slow paths are welcome when they are worth it. They are
//! never allowed to be invisible.
//!
//! # Licence
//!
//! [`Licence::NonCommercial`] means the node's WEIGHTS are research-only.
//! Rule 10: such nodes are supported and tagged, never refused.
//!
//! The classification is derived from the model registry rather than a list
//! kept by hand here, so making a node non-commercial is a data change and
//! cannot be forgotten. It reads the exact ids a node declares in
//! `NodeKind::models()` — per instance, so a node offering a research-only
//! option is tagged only while that option is picked.

use crate::graph::NodeKind;

/// What a node costs to RENDER, per frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Speed {
    /// Fuses into the main pass: free beyond its own ALU.
    Fast,
    /// Takes its own pass(es) and intermediate textures.
    Slow,
    /// REALLY slow: a pass CHAIN that dominates the frame rather than a
    /// pass — dozens of dispatches, or a gather whose per-pixel work is
    /// unbounded. A kind of slow (everything said of `Slow` holds), with one
    /// extra tell in the graph: the ring gets a halo and the wires below it
    /// glow. Classified by what the compositor actually dispatches
    /// (`codegen::FilterKind`'s own accounting), never by how a node feels.
    Heavy,
}

/// A paired Loop is heavy from this many iterations: each one is a
/// full-canvas pass of the whole body, so eight is a Blur four times over.
pub const HEAVY_LOOP: f32 = 8.0;
/// A user Shader is heavy from this many passes, each full-canvas and
/// arbitrary.
pub const HEAVY_SHADER_PASSES: usize = 3;

/// What a node's output costs to USE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Licence {
    Permissive,
    /// Research-only weights. Supported, tagged, warned about.
    NonCommercial,
    /// Commercial use allowed only under terms a Write cannot check for
    /// you — a revenue cap, use restrictions, a provenance clause (the
    /// LTX-2.x Community Licence). Not research-only, but warned about
    /// exactly like it: the moment output leaves the building is the
    /// moment somebody has to have read those terms.
    Restricted,
}

impl Licence {
    /// Is this a licence the graph tags and a Write warns about? Every
    /// check that used to ask "non-commercial?" asks this.
    pub fn warns(self) -> bool {
        self != Licence::Permissive
    }
}

/// Does this node take its own pass?
///
/// Note what is NOT slow: the baked nodes. A Rife, a Tracker, a Blender or a
/// loaded GraphBake all cost an explicit, user-triggered bake and then play
/// through a reader slot exactly like a Read. At render time they are the
/// cheapest thing in the graph, and colouring them red would be a lie that
/// taught people to avoid the very mechanism that fixes slowness.
pub fn speed_of(kind: &NodeKind) -> Speed {
    // A node that declares its own chain says what it costs.
    if let Some(stage) = kind.stage_def() {
        return stage.speed();
    }
    match kind {
        // The prefix-sum disc is O(r) per pixel and a few passes; the
        // accurate spiral gather is an arbitrary kernel's worth of taps per
        // pixel — the mode the panel already marks "(slow)".
        NodeKind::Defocus(d) => {
            if d.mode.is_slow() {
                Speed::Heavy
            } else {
                Speed::Slow
            }
        }
        // n full-canvas passes when paired; an unpaired LoopOut is the
        // body evaluated once, fused (and that is what every_kind's
        // default-params probe sees).
        NodeKind::LoopOut(p) => {
            if p.pair == 0 {
                Speed::Fast
            } else if p.n.round() >= HEAVY_LOOP {
                Speed::Heavy
            } else {
                Speed::Slow
            }
        }
        // A user shader is slow only when it actually runs: one that does
        // not compile is a passthrough, and gets no stage (see
        // `codegen::filter_stages`). Every pass is a full-canvas dispatch of
        // whatever the author wrote, so a stack of them is heavy.
        NodeKind::Shader(p) => {
            let def = p.def();
            if !def.ok() {
                Speed::Fast
            } else if def.passes >= HEAVY_SHADER_PASSES {
                Speed::Heavy
            } else {
                Speed::Slow
            }
        }
        // A Cryptomatte's cost is not a PASS, it is a DECODE: six rank
        // layers pulled out of the EXR every frame, on top of the plate,
        // before a matte exists at all — the heaviest thing in a comp that
        // has one, and the one place where the stage rule above cannot see
        // the work. It is also exactly what the node's own bake takes
        // away, so it glows until the matte is on disk (`clears_slow`).
        NodeKind::Cryptomatte(_) => Speed::Heavy,
        // A Splat projects, depth-sorts and draws its whole capture in
        // passes of its own whenever the camera moves: millions of splats,
        // not a fused expression. A Bake below it clears the red.
        NodeKind::Splat(_) => Speed::Slow,
        // A Blender node's cost is a PROCESS: every frame is a round trip
        // to Blender — a viewport draw and a copy back (slow), or a real
        // render, seconds a frame in Cycles (heavy). No pass of ours, so
        // the stage rule cannot see it; it is the second entry in the
        // tests' `decode_cost` list, and a GraphBake below it is what turns
        // the round trips into files.
        NodeKind::Blender(p) => match p.quality {
            crate::blender::Quality::Viewport => Speed::Slow,
            crate::blender::Quality::Final => Speed::Heavy,
        },
        _ => Speed::Fast,
    }
}

/// Are this node's weights free to use commercially?
pub fn licence_of(kind: &NodeKind) -> Licence {
    // A GenAI node's models are REMOTE — never downloaded, so not in
    // `kind.models()` — and their licences live in `genai::catalog`. The
    // take on screen carries the licence it was MADE under, and a bake
    // below it does not clear this (see `clears_slow`: it never mentions
    // licence at all).
    // A Sammie node's matting weights are fetched by the TOOL, not the
    // catalog; the node knows whether it uses them, and whether the matte
    // it shows was made with them.
    if let NodeKind::Sammie(p) = kind {
        if !p.commercial() {
            return Licence::NonCommercial;
        }
    }
    if let NodeKind::GenAi(p) = kind {
        if !p.commercial() {
            return Licence::NonCommercial;
        }
    }
    // LTX's weights are fetched by its tool (gated on Hugging Face), not
    // the catalog; the node's purpose is to run them, so it is tagged
    // whether or not a result is loaded — the terms are shown before the
    // fetch, and a bake does not clear them.
    if let NodeKind::LtxHdr(_) = kind {
        return Licence::Restricted;
    }
    if crate::models::all_commercial(kind.models()) {
        Licence::Permissive
    } else {
        Licence::NonCommercial
    }
}

/// Does this node CLEAR the slow taint for everything downstream of it?
///
/// Only a node SERVING FROM DISK does, and only because the claim is true:
/// the pixels downstream really do come off disk instead of the branch
/// being re-evaluated. Two of them, and `is_loaded` means the same thing
/// for both — "this node's frames are really there":
///
/// - a **GraphBake** with frames written;
/// - a **Cryptomatte** whose matte is baked into the NVMe tier, which is
///   what replaces the per-frame rank decodes with one matte read.
///
/// `is_loaded` is passed in because whether a bake exists is app state, not
/// graph state — the authored `baked` flag stays false.
pub fn clears_slow(kind: &NodeKind, is_loaded: bool) -> bool {
    is_loaded
        && matches!(
            kind,
            NodeKind::GraphBake { .. }
                // A Crop playing its own bake (`SoftCropParams::bake`).
                | NodeKind::SoftCrop(_)
                | NodeKind::Cryptomatte(_)
                | NodeKind::CorridorKey(_)
                | NodeKind::Flow(_)
                | NodeKind::Rifetime(_)
                // Flow-mode frames on disk (only ever loaded in flow mode).
                | NodeKind::FrameFix(_)
                | NodeKind::Upscale(_)
                | NodeKind::HdrCnn(_)
                | NodeKind::SeedVr(_)
                | NodeKind::Depth(_)
        )
        // A matte off disk reads no plate; a cutout still evaluates it.
        || is_loaded && matches!(kind, NodeKind::Sammie(p) if p.output == crate::graph::SammieOut::Matte)
        // Raw, the result alone off disk; composited, the plate under it is
        // still evaluated, so it clears nothing.
        || is_loaded && matches!(kind, NodeKind::LtxHdr(p) if p.raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Graph, NodeKind};

    fn up(model: crate::graph::UpscaleModel) -> crate::graph::UpscaleParams {
        crate::graph::UpscaleParams { model, ..Default::default() }
    }

    /// Every variant, so a new node has to be classified rather than
    /// defaulting to cyan by omission.
    fn every_kind() -> Vec<NodeKind> {
        vec![
            NodeKind::read(0, ""),
            NodeKind::Camera3D(Default::default()),
            NodeKind::Card3D(Default::default()),
            NodeKind::Scene3D(Default::default()),
            NodeKind::Render3D(Default::default()),
            NodeKind::Transform(Default::default()),
            NodeKind::Grade(Default::default()),
            NodeKind::ColorCompress(Default::default()),
            NodeKind::HueRotate(Default::default()),
            NodeKind::Lut(Default::default()),
            NodeKind::Text(Default::default()),
            NodeKind::Burnin(Default::default()),
            NodeKind::Mirror(Default::default()),
            NodeKind::FlipFlop(Default::default()),
            NodeKind::Turn(Default::default()),
            NodeKind::Uncrop(Default::default()),
            NodeKind::Compare(Default::default()),
            NodeKind::Gamma(Default::default()),
            NodeKind::ToneMap(Default::default()),
            NodeKind::UvWarp(Default::default()),
            NodeKind::Env(Default::default()),
            NodeKind::Refract(Default::default()),
            NodeKind::GridWarp(Default::default()),
            NodeKind::LensDistort(Default::default()),
            NodeKind::Warp(Default::default()),
            NodeKind::Expression(Default::default()),
            NodeKind::Despill(Default::default()),
            NodeKind::Lightwrap(Default::default()),
            // Rendered on a click and played from disk: never a stage.
            NodeKind::Upscale(up(crate::graph::UpscaleModel::Compact)),
            NodeKind::Upscale(up(crate::graph::UpscaleModel::Span)),
            NodeKind::Upscale(up(crate::graph::UpscaleModel::X4Plus)),
            NodeKind::SeedVr(Default::default()),
            NodeKind::Depth(Default::default()),
            NodeKind::Depth(crate::graph::DepthParams { model: crate::graph::DepthModel::Large, ..Default::default() }),
            NodeKind::LtxHdr(Default::default()),
            NodeKind::HdrCnn(Default::default()),
            NodeKind::HdrCnn(crate::graph::HdrCnnParams { model: crate::graph::HdrCnnModel::Original, ..Default::default() }),
            NodeKind::ScreenKey(Default::default()),
            NodeKind::ColorSlice(Default::default()),
            NodeKind::Blur(Default::default()),
            NodeKind::Defocus(Default::default()),
            NodeKind::Glow(Default::default()),
            NodeKind::Godrays(Default::default()),
            NodeKind::LensFlare(Default::default()),
            NodeKind::Contrast(Default::default()),
            NodeKind::EdgeExtend(Default::default()),
            NodeKind::Glint(Default::default()),
            NodeKind::EdgeDetect(Default::default()),
            NodeKind::Median(Default::default()),
            NodeKind::HighlightRebuild(Default::default()),
            NodeKind::Stabilize(Default::default()),
            NodeKind::CamTrack(Default::default()),
            NodeKind::Convolve(Default::default()),
            NodeKind::EdgeDetect(crate::graph::EdgeDetectParams {
                output: crate::graph::EdgeOutput::Matte,
                ..Default::default()
            }),
            NodeKind::Diffuse(Default::default()),
            NodeKind::Dilate(Default::default()),
            NodeKind::Shader(Default::default()),
            NodeKind::blend(),
            NodeKind::Breakout(Default::default()),
            NodeKind::Recombine(Default::default()),
            NodeKind::Extract(Default::default()),
            NodeKind::Mix(Default::default()),
            NodeKind::Geo(Default::default()),
            NodeKind::GeoNode(Default::default()),
            NodeKind::Sdf(Default::default()),
            NodeKind::SdfNode(Default::default()),
            NodeKind::Curve(Default::default()),
            NodeKind::Mask,
            NodeKind::LumaKey(Default::default()),
            NodeKind::Cryptomatte(Default::default()),
            NodeKind::Spline(Default::default()),
            NodeKind::Paint(Default::default()),
            NodeKind::Blender(Default::default()),
            NodeKind::Blender(crate::blender::BlenderParams {
                quality: crate::blender::Quality::Final,
                ..Default::default()
            }),
            NodeKind::Shuffle(crate::graph::ShuffleParams::IDENTITY),
            NodeKind::Premult { premult: false },
            NodeKind::Mult(Default::default()),
            NodeKind::ToMatte(Default::default()),
            NodeKind::ToData(Default::default()),
            NodeKind::ToRgba(Default::default()),
            NodeKind::SoftCrop(Default::default()),
            NodeKind::MotionBlur(Default::default()),
            NodeKind::MatteIn,
            NodeKind::Reformat(Default::default()),
            NodeKind::Grid(Default::default()),
            NodeKind::Blank(Default::default()),
            NodeKind::Starfield(Default::default()),
            NodeKind::Planet(Default::default()),
            NodeKind::Plasma(Default::default()),
            NodeKind::Splat(Default::default()),
            NodeKind::Particles(Default::default()),
            NodeKind::ParticleNode(Default::default()),
            NodeKind::GlintKernel(Default::default()),
            NodeKind::BokehKernel(Default::default()),
            NodeKind::Grain(Default::default()),
            NodeKind::Gradient(Default::default()),
            NodeKind::Gradient(crate::graph::GradientParams {
                interp: crate::graph::GradInterp::Lch,
                ..Default::default()
            }),
            NodeKind::Slip(Default::default()),
            NodeKind::Speed(Default::default()),
            NodeKind::TimeRemap(Default::default()),
            NodeKind::Write(Default::default()),
            NodeKind::GraphBake { source: 0, baked: false, start: None, mask: false, window: None, matte_half: false, single: false },
            NodeKind::GenAi(Default::default()),
            NodeKind::CorridorKey(Default::default()),
            NodeKind::Sammie(Default::default()),
            NodeKind::Flow(Default::default()),
            NodeKind::Rifetime(Default::default()),
            NodeKind::Freeze(Default::default()),
            NodeKind::FrameFix(crate::graph::FrameFixParams::new(0, 5)),
            NodeKind::FrameFix(crate::graph::FrameFixParams { loaded: true, ..crate::graph::FrameFixParams::new(0, 5) }),
            NodeKind::VectorBlur(Default::default()),
            NodeKind::VectorWarp(Default::default()),
            NodeKind::Tracker(Default::default()),
            NodeKind::PlanarTrack(Default::default()),
            NodeKind::LoopIn,
            NodeKind::LoopOut(Default::default()),
        ]
    }

    /// Kinds whose per-frame cost is a DECODE or a PROCESS rather than a
    /// GPU pass. Codegen gives them no stage and they are slow anyway,
    /// which is the one thing the stage rule cannot see. Deliberately
    /// short, and listed here rather than waved through: every other node
    /// earns its colour from `stage_def`, and an addition to this list is
    /// a claim somebody has to justify. Three entries: a Cryptomatte's rank
    /// decodes, a Blender node's round trip to another process, and a
    /// Splat's layer — drawn outside the stages like a Plasma's, but a
    /// projection, a depth sort and a draw of the whole capture (millions
    /// of splats, tens of milliseconds) every time the camera moves.
    fn decode_cost(kind: &NodeKind) -> bool {
        matches!(kind, NodeKind::Cryptomatte(_) | NodeKind::Blender(_) | NodeKind::Splat(_))
    }

    /// THE invariant: "slow" means "codegen gives it a stage" — or, for the
    /// short list above, a decode nobody can see in the stages. Two answers
    /// to one question drift; this makes them one question with a witness.
    #[test]
    fn slow_matches_codegen() {
        for kind in every_kind() {
            let name = kind.name();
            if decode_cost(&kind) {
                assert_ne!(speed_of(&kind), Speed::Fast, "{name}: a decode cost is still a cost");
                continue;
            }
            let mut g = Graph::default();
            g.nodes.insert(1, NodeKind::read(0, ""));
            g.nodes.insert(2, kind.clone());
            g.edges.insert((2, 0), 1);
            g.view = Some(2);
            let staged = !crate::codegen::generate(&g).blurs.is_empty();
            let says = speed_of(&kind) != Speed::Fast;
            assert_eq!(
                says, staged,
                "{name}: speed_of says slow={says}, codegen gives {} stage(s)",
                staged as u8
            );
        }
    }

    /// Heavy is the pass CHAINS that dominate a frame, by the compositor's
    /// own accounting — not a feeling. This pins the set, both ways.
    #[test]
    fn heavy_is_the_pass_chains_that_dominate_a_frame() {
        use crate::graph::{DefocusMode, DefocusParams, ShaderParams};
        let mut d = DefocusParams::default();
        assert_eq!(speed_of(&NodeKind::Defocus(d.clone())), Speed::Slow, "prefix-sum disc");
        d.mode = DefocusMode::Accurate;
        assert_eq!(speed_of(&NodeKind::Defocus(d)), Speed::Heavy, "spiral gather");
        assert_eq!(speed_of(&NodeKind::Blur(Default::default())), Speed::Slow);

        let NodeKind::LoopOut(mut lp) = NodeKind::LoopOut(Default::default()) else {
            unreachable!()
        };
        lp.pair = 1;
        lp.n = 2.0;
        assert_eq!(speed_of(&NodeKind::LoopOut(lp.clone())), Speed::Slow, "two passes");
        lp.n = HEAVY_LOOP;
        assert_eq!(speed_of(&NodeKind::LoopOut(lp)), Speed::Heavy, "eight passes");

        let mut one = ShaderParams::default();
        one.set_source(
            "//! passes: 1\nfn pass1(uv: vec2f, frag: vec2f) -> vec4f { return vec4f(1.0); }\n"
                .into(),
        );
        assert_eq!(speed_of(&NodeKind::Shader(one)), Speed::Slow, "one pass");
        let mut three = ShaderParams::default();
        three.set_source(
            "//! passes: 3\n\
             fn pass1(uv: vec2f, frag: vec2f) -> vec4f { return vec4f(1.0); }\n\
             fn pass2(uv: vec2f, frag: vec2f) -> vec4f { return tex(result1, uv); }\n\
             fn pass3(uv: vec2f, frag: vec2f) -> vec4f { return tex(result2, uv); }\n"
                .into(),
        );
        assert_eq!(speed_of(&NodeKind::Shader(three)), Speed::Heavy, "three passes");

        // A MotionBlur renders its whole upstream once per sample.
        use crate::nodes::motionblur::HEAVY_SAMPLES;
        let mb = |samples| NodeKind::MotionBlur(crate::graph::MotionBlurParams {
            samples,
            ..Default::default()
        });
        assert_eq!(speed_of(&mb(HEAVY_SAMPLES - 1)), Speed::Slow, "a few samples");
        assert_eq!(speed_of(&mb(HEAVY_SAMPLES)), Speed::Heavy, "the upstream, eight times over");

    }

    /// A shader that does not compile is a passthrough, so it is not slow —
    /// and the colour has to agree, or a broken node looks expensive.
    #[test]
    fn a_broken_shader_is_not_slow() {
        let mut p = crate::graph::ShaderParams::default();
        p.set_source("fn pass1(uv: vec2f, frag: vec2f) -> vec4f { return nope(uv); }".into());
        assert_eq!(speed_of(&NodeKind::Shader(p)), Speed::Fast);
    }

    /// A Cryptomatte GLOWS until its matte is baked — the rank decodes it
    /// costs every frame are the work, and the NVMe tier is what takes
    /// them away. Same shape as a Bake, same clearing rule, and nothing
    /// about it touches the licence taint (rule 10: the two clear
    /// differently, and this is the speed one).
    #[test]
    fn a_cryptomatte_glows_until_its_matte_is_baked() {
        let c = NodeKind::Cryptomatte(Default::default());
        assert_eq!(speed_of(&c), Speed::Heavy, "an unbaked cryptomatte is the frame's worst node");
        assert!(!clears_slow(&c, false), "an unbaked matte decodes every rank, every frame");
        assert!(clears_slow(&c, true), "a baked matte is one read off disk");
        assert_eq!(licence_of(&c), Licence::Permissive, "a bake must not touch rights");
    }

    /// A Blender node is a round trip to another process every frame:
    /// slow as a viewport draw, heavy as a real render — and nothing on
    /// the node clears it. The bake that does is the GraphBake below it.
    #[test]
    fn a_blender_node_is_a_round_trip_and_a_render_is_a_heavy_one() {
        use crate::blender::{BlenderParams, Quality};
        let live = NodeKind::Blender(Default::default());
        assert_eq!(speed_of(&live), Speed::Slow, "a viewport draw and a copy back, every frame");
        let fin = NodeKind::Blender(BlenderParams { quality: Quality::Final, ..Default::default() });
        assert_eq!(speed_of(&fin), Speed::Heavy, "an F12 render a frame");
        assert!(!clears_slow(&live, true) && !clears_slow(&fin, true));
        assert_eq!(licence_of(&live), Licence::Permissive);
    }

    #[test]
    fn only_a_loaded_bake_clears_slow() {
        let bake = NodeKind::GraphBake { source: 0, baked: false, start: None, mask: false, window: None, matte_half: false, single: false };
        assert!(clears_slow(&bake, true));
        assert!(!clears_slow(&bake, false), "an unbaked Bake still evaluates live");
        assert!(!clears_slow(&NodeKind::Blur(Default::default()), true));
    }

    /// The weights convention's invariant: every id a node declares is in
    /// the catalog AND hash-pinned. An uncatalogued id would leave the node
    /// gated forever behind a button with nothing to download; an unpinned
    /// one would let "ready" mean only "the right number of bytes".
    #[test]
    fn a_genai_node_is_tagged_by_its_remote_models_and_by_its_take() {
        let mut p = crate::graph::GenAiParams::default();
        assert_eq!(licence_of(&NodeKind::GenAi(p.clone())), Licence::Permissive);
        // Naming a research-only model tags it before anything is sent.
        p.pipeline.set_model(1, "fal/flux-dev");
        assert_eq!(licence_of(&NodeKind::GenAi(p.clone())), Licence::NonCommercial);
        // Back to a commercial model: the TAKE made with flux-dev still
        // carries the tag.
        p.pipeline.set_model(1, genai::catalog::DEFAULT_MODEL);
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
        assert_eq!(licence_of(&NodeKind::GenAi(p.clone())), Licence::NonCommercial);
        // And it is fast: a reader like any other, whatever it cost to make.
        assert_eq!(speed_of(&NodeKind::GenAi(p)), Speed::Fast);
    }

    #[test]
    fn every_declared_model_is_catalogued_and_pinned() {
        for kind in every_kind() {
            for id in kind.models() {
                let m = crate::models::spec(id).unwrap_or_else(|| {
                    panic!("{} declares model \"{id}\", which is not in models::CATALOG", kind.name())
                });
                assert!(
                    m.verifiable(),
                    "{} needs {id}, which has no sha256 pinned — `models get {id}` prints it",
                    kind.name()
                );
            }
        }
    }

    /// The non-commercial models are exactly these, each with a node that
    /// is tagged while it is picked. This is not a wish — it fails the day
    /// another one lands, which is exactly when someone needs to go and look
    /// at whether its node is tagged.
    #[test]
    fn the_non_commercial_models_are_listed_and_their_nodes_tagged() {
        use crate::graph::{DepthModel, DepthParams};
        let listed = ["vda-large", "vda-large-metric"];
        for m in crate::models::all() {
            assert_eq!(!m.commercial, listed.contains(&m.id), "{} is non-commercial: is its node tagged?", m.id);
        }
        for model in DepthModel::ALL {
            let k = NodeKind::Depth(DepthParams { model, ..Default::default() });
            let nc = listed.contains(&model.model_id());
            assert_eq!(licence_of(&k) == Licence::NonCommercial, nc, "{model:?}");
            // Loaded (frames on disk) clears slow; it never clears the tag.
            let loaded = NodeKind::Depth(DepthParams { model, loaded: true, ..Default::default() });
            assert!(clears_slow(&loaded, true));
            assert_eq!(licence_of(&loaded) == Licence::NonCommercial, nc);
        }
    }

    /// LTX HDR: restricted (warned like non-commercial), loaded or not, and
    /// a loaded result clears slow only when it is shown raw — composited,
    /// the plate under it is still rendered.
    #[test]
    fn an_ltx_hdr_node_is_restricted_and_clears_slow_only_raw() {
        use crate::graph::LtxHdrParams;
        for p in [
            LtxHdrParams::default(),
            LtxHdrParams { loaded: true, ..Default::default() },
            LtxHdrParams { loaded: true, raw: true, ..Default::default() },
        ] {
            let k = NodeKind::LtxHdr(p.clone());
            assert_eq!(licence_of(&k), Licence::Restricted);
            assert!(licence_of(&k).warns());
            assert_eq!(clears_slow(&k, p.loaded), p.raw && p.loaded, "{p:?}");
            assert_eq!(speed_of(&k), Speed::Fast, "a reader, never a stage");
        }
        assert!(!Licence::Permissive.warns() && Licence::NonCommercial.warns());
    }
}
