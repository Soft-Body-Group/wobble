//! Fused WGSL codegen: topo-walk the graph from the Viewer and string-assemble
//! compute shaders.
//!
//! Pointwise nodes (grade, merge, transforms-as-samples) fuse into one shader:
//! coordinates flow DOWN the recursion, colors flow UP, every Read leaf is one
//! bilinear sample. Non-pointwise nodes (Blur) split the graph: the blur's
//! input subtree becomes its own fused pass into an intermediate texture,
//! fixed separable H/V passes filter it (sigma is a uniform — never a
//! recompile), and downstream shaders sample the result like a Read.
//!
//! One uniform layout is shared by every pass of a plan: each node instance
//! gets a fixed range of vec4 slots, so one packed buffer serves all passes.
//! Parameter edits rewrite that buffer; only topology changes regenerate WGSL.

use crate::graph::{
    BlurParams, ColorCompressParams, ColorSliceParams, DilateParams, GlowParams, Graph,
    HueRotateParams, GammaParams, ToneMapParams, UvWarpParams, DespillParams, LightwrapParams, NodeId, ScreenKeyParams, CorridorKeyParams,
    BlankParams, GradientParams, GrainParams, NodeKind, SatParams,
    MotionBlurParams, TimeKey, ToDataParams, ToRgbaParams, TIME_KEY_ROOT,
    GenAiParams,
};
use crate::nodes::NodeDef;
use std::collections::HashMap;
use std::fmt::Write;

/// Where a node's parameters live in the shared uniform buffer (vec4 slots).
#[derive(Clone, Debug, Default)]
pub struct ParamLayout {
    /// node -> first vec4 slot index
    pub node_slot: HashMap<NodeId, usize>,
    /// total vec4 slots (including the header slot 0 = out_size)
    pub total_slots: usize,
    /// Transforms whose MODULATE port is wired: their span carries the raw
    /// params after the inverse rows (`MOD_XFORM_SLOTS` more), which is
    /// what a per-pixel partial transform is rebuilt from.
    pub modulated: std::collections::HashSet<NodeId>,
    /// One slot per reader instance whose texture is not simply the plate
    /// (`Graph::reader_windows`): xy is the corner of the window it holds,
    /// zw the scale it is stored at, and the leaf samples at
    /// `position * zw - xy`. Per instance, not per node — one Load reached
    /// on four time paths is four textures with four windows.
    pub window_slot: HashMap<(NodeId, TimeKey), usize>,
}

/// Extra vec4s a modulated Transform carries: (tx, ty, rot, mode),
/// (sx, sy, cx, cy), pin corners 0-1, pin corners 2-3, (w, h, pinned, 0).
pub const MOD_XFORM_SLOTS: usize = 5;

/// Whether `id` is a Transform or CornerPin with its modulate port wired.
fn modulated_transform(graph: &Graph, id: NodeId) -> bool {
    matches!(graph.nodes.get(&id), Some(NodeKind::Transform(_) | NodeKind::CornerPin(_)))
        && graph.input_of(id, 1).is_some_and(|m| graph.nodes.contains_key(&m))
}

/// Zeroed vec4s past the last node's span. A variable-length block that is
/// EMPTY — a spline with no points, say — still addresses `slot + k + i`,
/// one past its own span. Its loop never runs, but DX12's shader compiler
/// (FXC) proves every such index is >= the array length and refuses the
/// whole shader ("X3504: array index out of bounds"), where Metal and
/// Vulkan accept it: on Windows a comp holding a fresh spline did not
/// compile at all. The slack makes those indices land inside the array;
/// nothing is ever read from it.
///
/// ONE slot, deliberately: an empty block's first address is exactly the
/// end of its own span, never further, and the uniform is tight — seven
/// dense shapes are 1023 of its 1024 slots (`a_chain_of_shapes_fits_the_
/// uniform`), so every slot here comes straight off the roto ceiling.
pub const TAIL_SLOTS: usize = 1;

impl ParamLayout {
    /// The uniform array's length in vec4s — the ONE number the shader's
    /// declaration, the packer and the buffer size all use, so they cannot
    /// disagree (they had: 2, 2 and 1 for an empty graph).
    pub fn array_len(&self) -> usize {
        self.total_slots.max(1) + TAIL_SLOTS
    }

    pub fn byte_size(&self) -> u64 {
        (self.array_len() * 16) as u64
    }
}

/// Which filter machinery a split-out stage runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterKind {
    /// A node's own chain, declared by its `StageDef`.
    Node,
    /// Defocus. `mode` picks the pass chain: Fast is row-prefix-sums + a
    /// strip sum (exact flat disc, O(r) per pixel); Accurate is a single
    /// spiral-gather pass with an arbitrary kernel. `depth` is whether the
    /// depth port is wired (an aux subtree pass renders it); `modulate`
    /// whether the modulate matte is (another aux, after depth); `kernel` is
    /// the bokeh input's reader slot when it is wired to a source node;
    /// `kernel_aux` is a kernel from any OTHER node (a Bokeh Kernel, a graded
    /// plate), rendered as the last aux — Accurate only, since Fast is a
    /// flat disc and must not pay for a kernel it cannot use.
    Defocus {
        mode: crate::graph::DefocusMode,
        depth: bool,
        modulate: bool,
        kernel: Option<usize>,
        kernel_aux: bool,
    },
    /// Iterate a fused body: the stage's own subtree shader (the loop body,
    /// reading its LoopIn as `SourceBinding::LoopPrev`) is dispatched n
    /// times, ping-ponging the stage's two intermediates; the aux pass
    /// renders the SEED that primes iteration 0. n is an encode-time pass
    /// count, never part of the shader.
    Loop,
    /// User-authored WGSL (`crate::usershader`). Every declared image input
    /// is resolved to its own texture — input 0 through the stage's normal
    /// subtree pass, the rest through aux passes — and then `passes` user
    /// passes run, each writing its own full-canvas result that later
    /// passes may sample. The shader itself lives in `GeneratedPlan::shaders`
    /// (a `FilterKind` is `Copy`; a shader is not).
    Shader { inputs: usize, passes: usize },
}

/// A filter stage's identity: the node, plus the time path it is being
/// evaluated on. A blur upstream of two different retimes is two stages —
/// same node, two intermediate textures, each filtering its own frame.
pub type StageId = (NodeId, TimeKey);

/// What a shader's texture binding refers to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceBinding {
    /// Reader slot (uploaded plate).
    Reader(usize),
    /// A filter stage's output texture.
    Blur(StageId),
    /// A Paint node's rasterized stroke layer. One layer serves every time
    /// path — a stroke's lifetime is resolved when the layer is rasterized,
    /// not in the shader.
    Paint(NodeId),
    /// The same node's SOURCE layer (clone + smear): `rg` is where to fetch
    /// the input from, an offset in output pixels premultiplied by `a`; `a`
    /// is the coverage. Only bound for nodes that have such strokes.
    PaintSource(NodeId),
    /// The colour layer's MULTIPLIER (`M`; the colour layer is `B`): only
    /// bound when a stroke blends by a colour (`PaintParams::has_blend`),
    /// and then the node composites `below * M + B`.
    PaintMul(NodeId),
    /// The same node's EFFECT layer (blur + dodge + burn), additive: `r` is
    /// exposure in stops, `g` blur radius px, `b` blur coverage, `a` total
    /// coverage. Only bound for nodes that have such strokes.
    PaintEffect(NodeId),
    /// A 2dini node's rasterised geometry layer: drawn outside this pass,
    /// only when the geometry changes (see `georaster`).
    Geo(NodeId),
    /// A Blender node's layer — (the node whose session draws, the pass
    /// id; 0 = combined): the last frame the session handed over,
    /// uploaded by the app (`Compositor::blender_upload`).
    Blender(NodeId, u8),
    /// A LUT node's table (`crate::lut::Layout`), uploaded by the
    /// compositor when the file's bytes change; the dummy until one loads.
    Lut(NodeId),
    /// A Text or Burnin node's glyph atlas (`crate::textraster`), uploaded
    /// by the compositor when the words change; the dummy until then.
    Text(NodeId),
    /// A Plasma node's particle layer (`crate::plasmagpu`), re-drawn by
    /// the compositor when its uniform changes.
    Plasma(NodeId),
    /// A Particles node's layer (`crate::particlegpu`), simulated and
    /// re-drawn by the compositor when its frame or look changes.
    Particles(NodeId),
    /// A Splat node's layer (`crate::splatgpu`) — (the node that draws,
    /// true for its depth target) — re-drawn when its uniform changes.
    Splat(NodeId, bool),
    /// A loop's previous-iteration texture, named by the LoopIn's id. Inside
    /// the loop stage the compositor rebinds it per iteration (ping-pong);
    /// sampled anywhere else it resolves to the loop's finished result.
    LoopPrev(NodeId),
}

#[derive(Clone, Debug)]
pub struct GeneratedShader {
    pub wgsl: String,
    /// Texture binding order; binding for sources[i] is `2 + i`, the output
    /// storage texture is `2 + len`.
    pub sources: Vec<SourceBinding>,
}

#[derive(Clone, Debug)]
pub struct GeneratedPlan {
    pub layout: ParamLayout,
    /// Filter stages in dependency order:
    /// (stage id, filter kind, its input-subtree shader, its AUX subtree
    /// shaders). Each aux entry resolves a FURTHER image port to its own
    /// full-canvas texture before the stage runs — Defocus's depth input was
    /// the first user, a Shader node's inputs 1.. are the second. Empty for
    /// stages that take one image input.
    pub blurs: Vec<(StageId, FilterKind, GeneratedShader, Vec<GeneratedShader>)>,
    /// The final pass (viewer subtree + view transform).
    pub main: GeneratedShader,
    /// Paint nodes in the plan (compositor keeps a stroke layer per node).
    pub paints: Vec<NodeId>,
    /// The compiled shader behind each `FilterKind::Shader` stage.
    pub shaders: HashMap<StageId, std::sync::Arc<crate::usershader::ShaderDef>>,
    /// Parallel to `blurs`: the node each stage belongs to, as it was when
    /// the plan was generated.
    pub stage_nodes: Vec<NodeKind>,
    /// Parallel to `stage_nodes`: which of each node's inputs are wired —
    /// what its `StageDef` builds its program from (an unwired optional
    /// port costs nothing).
    pub stage_wired: Vec<Vec<bool>>,
}

/// A Grade zone's span in the uniform block: its five adjustment vec4s
/// (lift + saturation, gamma, gain, offset, contrast + pivot), one curve
/// header (count,
/// smooth, invert) and a FIXED run of `ZONE_PTS` points, two per vec4.
/// Fixed so the span never moves with the curve's shape: shaping a zone's
/// band is a uniform write, and only the zone COUNT recompiles.
pub const ZONE_SLOTS: usize = GRADE_SLOTS + 1 + crate::graph::ZONE_PTS / 2;

/// The Grade's own span (and each zone's adjustments): lift + saturation,
/// gamma, gain, offset, contrast + pivot.
pub const GRADE_SLOTS: usize = 5;

fn vec4_slots_for(kind: &NodeKind) -> usize {
    match kind {

        // The live input is compiled in, so there is nothing to pass.
        NodeKind::Switch(_) => 0,
        NodeKind::Read { .. } => 0,
        // A windowed bake carries where its box sits.
        NodeKind::GraphBake { baked, window, .. } => (*baked && window.is_some()) as usize,
        NodeKind::GenAi(_) => GenAiParams::VEC4_SLOTS,
        NodeKind::CorridorKey(_) => CorridorKeyParams::VEC4_SLOTS,
        NodeKind::Normals(_) => crate::graph::NormalsParams::VEC4_SLOTS,
        NodeKind::Relight(_) => <crate::graph::RelightParams as crate::nodes::NodeDef>::VEC4_SLOTS,
        NodeKind::Flow(_) => crate::nodes::flow::FlowParams::VEC4_SLOTS,
        NodeKind::Rifetime(_) => 0,
        // The dissolve between its two branches (unused once loaded).
        NodeKind::FrameFix(_) => 1,
        NodeKind::Freeze(_) => 0,
        NodeKind::VectorBlur(_) => crate::nodes::vectorblur::VectorBlurParams::VEC4_SLOTS,
        NodeKind::VectorWarp(_) => crate::nodes::vectorwarp::VectorWarpParams::VEC4_SLOTS,
        NodeKind::Sammie(_) => crate::nodes::sammie::SammieParams::VEC4_SLOTS,
        NodeKind::Transform(_) => 3, // three rows of the inverse homography (pin + affine)
        // lift, gamma, gain, offset, contrast, then a fixed span per zone.
        NodeKind::Grade(g) => GRADE_SLOTS + g.zone_count() * ZONE_SLOTS,

        // Every user param lives in the Shader node's OWN uniform buffer,
        // laid out by the shader's declaration rather than by this table.
        NodeKind::Shader(_) => 0,
        NodeKind::Blur(_) => BlurParams::VEC4_SLOTS,
        NodeKind::LoopIn | NodeKind::LoopOut(_) => 0, // n is an encode-time pass count
        NodeKind::ChannelSelect(_) => 0, // lowered to a Read before codegen
        NodeKind::Breakout(_) => 0, // lowered to its ChannelSelect kids before codegen
        // One lane per string for its gain (n is topology, so the span is
        // stable between recompiles).
        NodeKind::Recombine(p) => (p.n.clamp(1, 16) as usize).div_ceil(4),
        NodeKind::Defocus(_) => 0,
        NodeKind::Glow(_) => GlowParams::VEC4_SLOTS,
        NodeKind::Godrays(_) => crate::graph::GodraysParams::VEC4_SLOTS,
        NodeKind::LensFlare(_) => crate::graph::LensFlareParams::VEC4_SLOTS,
        NodeKind::EdgeExtend(_) => crate::graph::EdgeExtendParams::VEC4_SLOTS,
        NodeKind::Glint(_) => crate::graph::GlintParams::VEC4_SLOTS,
        NodeKind::EdgeDetect(_) => crate::graph::EdgeDetectParams::VEC4_SLOTS,
        NodeKind::Median(_) => crate::graph::MedianParams::VEC4_SLOTS,
        NodeKind::HighlightRebuild(_) => crate::graph::HighlightRebuildParams::VEC4_SLOTS,
        NodeKind::Convolve(_) => crate::graph::ConvolveParams::VEC4_SLOTS,
        NodeKind::Diffuse(_) => crate::graph::DiffuseParams::VEC4_SLOTS,
        // One header vec4 (scale, seed) then a block per field node: a
        // whole sub-graph's numbers live inside the owning node's span, so
        // editing any of them is a param write and never a recompile.
        NodeKind::Noise(n) => 1 + crate::field::slot_map(&n.graph).1,
        // A field node has no slot of its own — its numbers are inside the
        // Noise node that owns it.
        NodeKind::Field(_) => 0,
        // Geometry never goes near the uniform block: it is a layer.
        NodeKind::Geo(_) | NodeKind::GeoNode(_) => 0,
        // A header, then the SDF graph's blocks — inside the owner's span
        // like a noise field's, so its numbers are param writes.
        NodeKind::Sdf(p) => p.slots(),
        NodeKind::SdfNode(_) => 0,
        // A Blender picture is a layer too; its one slot is the layer's
        // size over the canvas's — 1 normally, 0.5 while a moving eye is
        // drawn at half resolution — written by the compositor, which is
        // the only thing that knows the layer's size.
        NodeKind::Blender(_) => 1,
        NodeKind::Dilate(_) => DilateParams::VEC4_SLOTS,
        NodeKind::Blend { .. } => 1, // mix
        NodeKind::Mix(_) => 1, // crossfade amount
        // Two affine rows per layer (the inverse, like a Transform), with
        // the layer's opacity riding in the first row's unused .w.
        NodeKind::Layer(p) => p.layers.len().min(crate::graph::LAYER_MAX),
        NodeKind::Mask => 0,
        NodeKind::Extract(_) => 0, // lowered to Masks before codegen
        NodeKind::MatteCombine(_) => 1, // mix
        // Header (count, smooth, invert) + the response points, 2 per vec4.
        NodeKind::LumaKey(k) => 1 + (k.curve().len() + 1) / 2,
        // Two header slots (tilt/levels/gamma, then count/invert/preview)
        // plus two per blob (centre+weight, tolerances+softness).
        NodeKind::Keyer(k) => 2 + 2 * k.blobs.len().min(crate::keyer::MAX_BLOBS),
        NodeKind::Cryptomatte(_) => crate::cryptomatte::CryptomatteParams::SLOTS,
        // Two header slots (per-curve counts, then flags) + the master run
        // and each split channel's own, every run padded to a whole vec4.
        NodeKind::Curve(c) => {
            2 + (c.points.len() + 1) / 2
                + c.rgb.iter().map(|r| (r.len() + 1) / 2).sum::<usize>()
        }
        NodeKind::CornerPin(_) => 3, // three rows of the inverse homography
        NodeKind::Stabilize(_) => crate::graph::StabilizeParams::VEC4_SLOTS,
        NodeKind::CamTrack(_) => crate::graph::CamTrackParams::VEC4_SLOTS,
        NodeKind::Paint(_) => 0,     // strokes live in the layer texture
        // Registry-declared params, 4 per vec4. Slot span joins the
        // topology hash via the plugin source hash, so a manifest edit that
        // changes the param count recompiles cleanly.
        NodeKind::Custom { type_id, .. } => crate::plugins::get(type_id)
            .map(|d| d.params.len().div_ceil(4))
            .unwrap_or(0),
        // Header (count, bounding circle) + matte controls + the
        // subdivided polyline, 2 points per vec4, then the per-sample
        // feathers 4 per vec4.
        NodeKind::Spline(s) => {
            let m = spline_samples(s.points.len());
            2 + m.div_ceil(2) + m.div_ceil(4)
        }
        NodeKind::Mult(_) => 1, // value, alpha flag
        NodeKind::ToMatte(_) => 2, // channel weights, then range + invert
        NodeKind::MatteIn => 0,   // pure rewiring, nothing to parameterise
        NodeKind::Grid(_) => 4, // spacing+width, offset, line rgba, back rgba
        NodeKind::Gradient(_) => GradientParams::VEC4_SLOTS,
        NodeKind::Blank(_) => BlankParams::VEC4_SLOTS,
        NodeKind::Starfield(_) => crate::graph::StarfieldParams::VEC4_SLOTS,
        NodeKind::Planet(_) => crate::graph::PlanetParams::VEC4_SLOTS,
        NodeKind::Plasma(_) => crate::graph::PlasmaParams::VEC4_SLOTS,
        NodeKind::Splat(_) => crate::graph::SplatParams::VEC4_SLOTS,
        NodeKind::Particles(_) => crate::graph::ParticlesParams::VEC4_SLOTS,
        NodeKind::ParticleNode(_) | NodeKind::GenAiStep(_) | NodeKind::RotoNode(_) => 0,
        NodeKind::GlintKernel(_) => crate::graph::GlintKernelParams::VEC4_SLOTS,
        NodeKind::BokehKernel(_) => crate::graph::BokehKernelParams::VEC4_SLOTS,
        NodeKind::Prompt(_) => 0,
        NodeKind::Camera3D(_) | NodeKind::Card3D(_) | NodeKind::Scene3D(_) | NodeKind::Render3D(_) => 0,
        NodeKind::Grain(_) => GrainParams::VEC4_SLOTS,
        NodeKind::Reformat(_) => 1, // source-side scale + offset
        NodeKind::Saturation(_) => SatParams::VEC4_SLOTS,
        NodeKind::Contrast(_) => crate::graph::ContrastParams::VEC4_SLOTS,
        NodeKind::ColorCompress(_) => ColorCompressParams::VEC4_SLOTS,
        NodeKind::HueRotate(_) => HueRotateParams::VEC4_SLOTS,
        NodeKind::Lut(_) => crate::graph::LutParams::VEC4_SLOTS,
        NodeKind::Text(_) => crate::graph::TextParams::VEC4_SLOTS,
        NodeKind::Burnin(_) => crate::graph::BurninParams::VEC4_SLOTS,
        NodeKind::Mirror(_) => crate::graph::MirrorParams::VEC4_SLOTS,
        NodeKind::FlipFlop(_) => crate::graph::FlipFlopParams::VEC4_SLOTS,
        NodeKind::Turn(_) => crate::graph::TurnParams::VEC4_SLOTS,
        NodeKind::Uncrop(_) => crate::graph::UncropParams::VEC4_SLOTS,
        NodeKind::Compare(_) => crate::graph::CompareParams::VEC4_SLOTS,
        NodeKind::MorphDissolve(_) => crate::graph::MorphDissolveParams::VEC4_SLOTS,
        NodeKind::Gamma(_) => GammaParams::VEC4_SLOTS,
        NodeKind::ToneMap(_) => ToneMapParams::VEC4_SLOTS,
        NodeKind::UvWarp(_) => UvWarpParams::VEC4_SLOTS,
        NodeKind::Env(_) => crate::graph::EnvParams::VEC4_SLOTS,
        NodeKind::Refract(_) => crate::graph::RefractParams::VEC4_SLOTS,
        // The header, then a vec4 per lattice point: the span follows the
        // lattice's SHAPE, which is in the topology tag.
        NodeKind::GridWarp(p) => p.slots(),
        NodeKind::LensDistort(_) => crate::graph::LensDistortParams::VEC4_SLOTS,
        NodeKind::Warp(p) => p.slots(),
        NodeKind::Expression(_) => crate::graph::ExpressionParams::VEC4_SLOTS,
        NodeKind::Despill(_) => DespillParams::VEC4_SLOTS,
        NodeKind::Lightwrap(_) => LightwrapParams::VEC4_SLOTS,
        NodeKind::Upscale(_) => crate::graph::UpscaleParams::VEC4_SLOTS,
        NodeKind::HdrCnn(_) => crate::graph::HdrCnnParams::VEC4_SLOTS,
        NodeKind::SeedVr(_) => crate::graph::SeedVrParams::VEC4_SLOTS,
        NodeKind::Depth(_) => crate::graph::DepthParams::VEC4_SLOTS,
        NodeKind::LtxHdr(_) => crate::graph::LtxHdrParams::VEC4_SLOTS,
        NodeKind::DiffLight(_) => crate::graph::DiffLightParams::VEC4_SLOTS,
        NodeKind::ScreenKey(_) => ScreenKeyParams::VEC4_SLOTS,
        NodeKind::ColorSlice(_) => ColorSliceParams::VEC4_SLOTS,
        NodeKind::ToData(_) => ToDataParams::VEC4_SLOTS,
        NodeKind::ToRgba(_) => ToRgbaParams::VEC4_SLOTS,
        NodeKind::SoftCrop(p) => p.slots(),
        NodeKind::MotionBlur(_) => MotionBlurParams::VEC4_SLOTS,
        // Passthroughs: the track is animation, the time nodes remap the
        // player, and premult is a formula with no parameters.
        // Four one-hot rows plus the constant column.
        NodeKind::Shuffle(_) => 5,
        NodeKind::Write(_)
        | NodeKind::Tracker(_)
        | NodeKind::PlanarTrack(_)
        | NodeKind::Slip(_)
        | NodeKind::Premult { .. }
        // A dot is elided by the editor and never reaches here; if one ever
        // did, it is a wire, and a wire has no parameters.
        | NodeKind::Dot { .. }
        | NodeKind::TimeRange(_) => 0,
        // A snapping time node emits nothing; a blending one dissolves
        // between two frames and needs the weight as a uniform.
        NodeKind::TimeRemap(_) | NodeKind::Speed(_) | NodeKind::RangeRemap(_) => {
            usize::from(kind.time_interp() == crate::graph::TimeInterp::Blend)
        }
    }
}

/// Outline samples a spline aims to spend in total, spread over its
/// segments. A flat 8-per-segment was visibly faceted on anything big;
/// a budget buys a 3-point blob real smoothness without letting a
/// 40-point shape grow an unbounded per-pixel loop (and uniform span) to
/// match — the mask walks every sample for every pixel.
pub const SPLINE_SAMPLE_BUDGET: usize = 192;
/// Subdivision floor and ceiling. The floor keeps many-point shapes bending
/// between their points; the ceiling stops a 3-point spline from spending
/// the whole budget on three nearly straight edges.
pub const SPLINE_SUBD_MIN: usize = 4;
pub const SPLINE_SUBD_MAX: usize = 32;

/// Samples per control-point segment: a pure function of the point count,
/// so the slot span (and therefore the compiled layout) still depends on
/// nothing but `points.len()`. Mode never enters it — switching mode must
/// not recompile.
pub fn spline_subd(n_points: usize) -> usize {
    if n_points == 0 {
        return SPLINE_SUBD_MAX;
    }
    (SPLINE_SAMPLE_BUDGET / n_points).clamp(SPLINE_SUBD_MIN, SPLINE_SUBD_MAX)
}

/// Upper bound on outline samples for a point count — what the layout is
/// sized from. Degenerate shapes (< 3 points) emit fewer and pack short.
pub fn spline_samples(n_points: usize) -> usize {
    n_points * spline_subd(n_points)
}

/// The closed catmull-rom polyline the mask shader walks; also drawn by the
/// app's viewer gizmo so the overlay matches the GPU exactly.
/// The closed outline the mask shader walks: one sample list, whatever the
/// mode. Also drawn by the app's viewer gizmo, so the overlay is exactly
/// what the GPU rasterizes. Each sample carries the feather interpolated
/// from its segment's endpoints.
///
/// The sample COUNT is `spline_samples(points)` in every mode, so switching
/// mode never changes the uniform layout (and never recompiles).
pub fn spline_polyline(
    sp: &crate::graph::SplineParams,
    size: [f32; 2],
) -> Vec<([f32; 2], f32)> {
    spline_outline(sp, size).into_iter().map(|(p, f, _)| (p, f)).collect()
}

/// The same samples as [`spline_polyline`], each with its soft BAND: the
/// signed distances from the outline (output pixels, + outward) where
/// coverage is full (`lo`) and where it reaches zero (`hi`). A point's band
/// is its feather about its bias; the shape-wide `feather_in` / `feather_out`
/// widen every band on their side. What the shader ramps across and what
/// the gizmo's two rings draw.
pub fn spline_band(
    sp: &crate::graph::SplineParams,
    size: [f32; 2],
) -> Vec<([f32; 2], [f32; 2])> {
    // A shape still being drawn has no inside yet: no band, no matte.
    if sp.open {
        return Vec::new();
    }
    let (fi, fo) = (sp.feather_in.max(0.0), sp.feather_out.max(0.0));
    spline_outline(sp, size)
        .into_iter()
        .map(|(p, f, b)| (p, [b - 0.5 * f - fi, b + 0.5 * f + fo]))
        .collect()
}

/// The walk both of the above read: (position, feather, bias) per sample.
/// A roto shape the lowering placed (`SplineParams::proj`: its axes and
/// camera) has every SAMPLE carried through that homography — samples, not
/// control points, because a projected curve is not the curve of the
/// projected points.
fn spline_outline(
    sp: &crate::graph::SplineParams,
    size: [f32; 2],
) -> Vec<([f32; 2], f32, f32)> {
    let mut out = spline_outline_flat(sp, size);
    if let Some(pr) = sp.proj.filter(|p| !p.is_identity()) {
        let hm = pr.homography(size);
        for s in &mut out {
            s.0 = crate::graph::homography_point(&hm, s.0, size);
        }
    }
    out
}

fn spline_outline_flat(
    sp: &crate::graph::SplineParams,
    size: [f32; 2],
) -> Vec<([f32; 2], f32, f32)> {
    use crate::graph::SplineMode;
    // The shape transform is affine, and every interpolation below is an
    // affine combination of control points — so transforming the CONTROL
    // points is exactly transforming every sample, for a fraction of the
    // work. Handles are offsets, so they take the linear part only — which
    // transforming the handle's TIP and re-subtracting gives for free.
    let owned: Vec<crate::graph::SplinePoint>;
    let pts: &[crate::graph::SplinePoint] = if sp.xf.is_identity() {
        &sp.points
    } else {
        owned = sp
            .points
            .iter()
            .map(|q| {
                let p = sp.xf.point(q.p, size);
                let tip = |h: [f32; 2]| {
                    let t = sp.xf.point([q.p[0] + h[0], q.p[1] + h[1]], size);
                    [t[0] - p[0], t[1] - p[1]]
                };
                crate::graph::SplinePoint {
                    p,
                    feather: q.feather,
                    bias: q.bias,
                    h_in: tip(q.h_in),
                    h_out: tip(q.h_out),
                    tension: q.tension,
                }
            })
            .collect();
        &owned
    };
    let n = pts.len();
    if n == 0 {
        return Vec::new();
    }
    if n < 3 {
        return pts.iter().map(|q| (q.p, q.feather, q.bias)).collect();
    }
    let subd = spline_subd(n);
    let mut out = Vec::with_capacity(n * subd);
    for i in 0..n {
        let p0 = pts[(i + n - 1) % n];
        let p1 = pts[i];
        let p2 = pts[(i + 1) % n];
        let p3 = pts[(i + 2) % n];
        for k in 0..subd {
            let t = k as f32 / subd as f32;
            let (t2, t3) = (t * t, t * t * t);
            let xy = match sp.mode {
                SplineMode::Linear => [
                    p1.p[0] + (p2.p[0] - p1.p[0]) * t,
                    p1.p[1] + (p2.p[1] - p1.p[1]) * t,
                ],
                SplineMode::CatmullRom if p1.tension == 0.0 && p2.tension == 0.0 => {
                    let cr = |a: f32, b: f32, c: f32, d: f32| {
                        0.5 * (2.0 * b
                            + (c - a) * t
                            + (2.0 * a - 5.0 * b + 4.0 * c - d) * t2
                            + (3.0 * b - a - 3.0 * c + d) * t3)
                    };
                    [
                        cr(p0.p[0], p1.p[0], p2.p[0], p3.p[0]),
                        cr(p0.p[1], p1.p[1], p2.p[1], p3.p[1]),
                    ]
                }
                // Tension: the same curve as a Hermite segment whose end
                // tangents are Catmull-Rom's scaled by (1 - tension) — a
                // Cardinal spline with a tension per point. At 0 it IS the
                // arm above (kept separate so untensioned shapes stay bit-
                // identical); at 1 the tangent vanishes and the point is a
                // corner.
                SplineMode::CatmullRom => {
                    let (s1, s2) = (0.5 * (1.0 - p1.tension), 0.5 * (1.0 - p2.tension));
                    let (h00, h10) = (2.0 * t3 - 3.0 * t2 + 1.0, t3 - 2.0 * t2 + t);
                    let (h01, h11) = (-2.0 * t3 + 3.0 * t2, t3 - t2);
                    let herm = |a: f32, b: f32, c: f32, d: f32| {
                        h00 * b + h10 * s1 * (c - a) + h01 * c + h11 * s2 * (d - b)
                    };
                    [
                        herm(p0.p[0], p1.p[0], p2.p[0], p3.p[0]),
                        herm(p0.p[1], p1.p[1], p2.p[1], p3.p[1]),
                    ]
                }
                SplineMode::Bezier => {
                    // p1 -> p2 with p1.h_out and p2.h_in as the controls.
                    let u = 1.0 - t;
                    let (w0, w1, w2, w3) =
                        (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t3);
                    let c1 = [p1.p[0] + p1.h_out[0], p1.p[1] + p1.h_out[1]];
                    let c2 = [p2.p[0] + p2.h_in[0], p2.p[1] + p2.h_in[1]];
                    [
                        w0 * p1.p[0] + w1 * c1[0] + w2 * c2[0] + w3 * p2.p[0],
                        w0 * p1.p[1] + w1 * c1[1] + w2 * c2[1] + w3 * p2.p[1],
                    ]
                }
                SplineMode::BSpline => {
                    // Uniform cubic B-spline basis over p0..p3.
                    let b0 = ((1.0 - t).powi(3)) / 6.0;
                    let b1 = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0;
                    let b2 = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0;
                    let b3 = t3 / 6.0;
                    [
                        b0 * p0.p[0] + b1 * p1.p[0] + b2 * p2.p[0] + b3 * p3.p[0],
                        b0 * p0.p[1] + b1 * p1.p[1] + b2 * p2.p[1] + b3 * p3.p[1],
                    ]
                }
            };
            out.push((
                xy,
                p1.feather + (p2.feather - p1.feather) * t,
                p1.bias + (p2.bias - p1.bias) * t,
            ));
        }
    }
    out
}

/// One spline sample's band as the shader's lane: both edges quantised to
/// quarter pixels in ±512, packed `lo * 4096 + hi` (24 bits, exact in f32).
/// `sband` in the prelude is the inverse.
pub fn spline_band_lane(b: [f32; 2]) -> f32 {
    let q = |x: f32| ((x * 4.0).round() + 2048.0).clamp(0.0, 4095.0) as u32;
    (q(b[0]) * 4096 + q(b[1])) as f32
}

pub fn generate(graph: &Graph) -> GeneratedPlan {
    let order = graph.topo_from_view().unwrap_or_default();

    // Shared uniform layout. Slot 0 is the header (out_size).
    let mut layout = ParamLayout {
        node_slot: HashMap::new(),
        total_slots: 1,
        modulated: Default::default(),
        window_slot: HashMap::new(),
    };
    {
        // Deterministic: the layout must not move with a HashMap's order.
        let mut wins: Vec<_> = graph.reader_windows.keys().copied().collect();
        wins.sort_unstable();
        for w in wins {
            layout.window_slot.insert(w, layout.total_slots);
            layout.total_slots += 1;
        }
    }
    for id in &order {
        let kind = &graph.nodes[id];
        let mut n = vec4_slots_for(kind);
        if modulated_transform(graph, *id) {
            layout.modulated.insert(*id);
            n += MOD_XFORM_SLOTS;
        }
        if n > 0 {
            layout.node_slot.insert(*id, layout.total_slots);
            layout.total_slots += n;
        }
    }

    // One pass per filter stage, in dependency order, then the main pass.
    // A filter reached on two different time paths is TWO stages: each
    // filters its own frame of the same node.
    let mut shaders: HashMap<StageId, std::sync::Arc<crate::usershader::ShaderDef>> =
        HashMap::new();
    let blurs: Vec<_> = filter_stages(graph)
        .into_iter()
        .map(|(stage, kind)| {
            let (id, tkey) = stage;
            let shader =
                gen_shader(graph, &layout, graph.input_of(id, 0), tkey, Some(stage));
            // Aux subtrees resolve FURTHER image ports into their own
            // textures. Same exclude throughout, so a branch that loops back
            // through this node reads transparent black rather than the
            // texture the stage is about to write.
            let aux_ports: Vec<usize> = match kind {
                FilterKind::Node => graph
                    .nodes
                    .get(&id)
                    .and_then(|k| k.stage_def())
                    .map(|s| s.aux_ports(crate::nodes::stage::Wired(&wired_of(graph, id))))
                    .unwrap_or_default(),
                // Depth first (the focus pick reads `aux[0]`), then the
                // modulate matte.
                FilterKind::Defocus { depth, modulate, kernel_aux, .. } => {
                    [
                        depth.then_some(2),
                        modulate.then_some(DEFOCUS_MOD_INPUT),
                        kernel_aux.then_some(DEFOCUS_KERNEL_INPUT),
                    ]
                        .into_iter()
                        .flatten()
                        .collect()
                }
                // The seed that primes iteration 0.
                FilterKind::Loop => vec![1],
                // Input 0 is the subtree pass above; 1.. are aux.
                FilterKind::Shader { inputs, .. } => (1..inputs).collect(),
            };
            let aux = aux_ports
                .into_iter()
                .map(|i| gen_shader(graph, &layout, graph.input_of(id, i), tkey, Some(stage)))
                .collect();
            if let (FilterKind::Shader { .. }, Some(NodeKind::Shader(p))) =
                (kind, graph.nodes.get(&id))
            {
                shaders.insert(stage, p.def());
            }
            (stage, kind, shader, aux)
        })
        .collect();
    let paints = order
        .iter()
        .filter(|id| {
            !graph.disabled.contains(id)
                && matches!(graph.nodes.get(id), Some(NodeKind::Paint(_)))
        })
        .copied()
        .collect();
    // The view target node itself is the render root, on the root timebase.
    let main = gen_shader(graph, &layout, graph.view_node(), TIME_KEY_ROOT, None);

    // A snapshot of each stage's node, for its `StageDef::program` — the
    // chain depends only on structural params, which this plan is keyed on.
    let stage_nodes = blurs.iter().map(|(s, ..)| graph.nodes[&s.0].clone()).collect();
    let stage_wired = blurs.iter().map(|(s, ..)| wired_of(graph, s.0)).collect();
    GeneratedPlan { layout, blurs, main, paints, shaders, stage_nodes, stage_wired }
}

/// Is this LoopOut a compilable loop? The paired LoopIn must exist and be
/// reachable through the body (input 0), and no node BETWEEN them — i.e. no
/// node that depends on the tap — may take its own pass: a stage runs once
/// per frame, so it cannot sit inside the iteration. (A stage feeding the
/// body from OUTSIDE the tap's cone is fine: it renders once and the body
/// samples its result every iteration, which is exactly what you want.)
///
/// The stage-kind list mirrors `filter_stages::kind_of`; `cost.rs`'s
/// `slow_matches_codegen` idea applies — if a new stage kind lands, add it
/// here or a loop containing it will silently iterate a stale texture.
pub fn loop_ok(graph: &Graph, out_id: NodeId, pair: NodeId) -> bool {
    use std::collections::HashSet;
    if pair == 0 || !matches!(graph.nodes.get(&pair), Some(NodeKind::LoopIn)) {
        return false;
    }
    // Upstream closure of the body input.
    let mut body: HashSet<NodeId> = HashSet::new();
    let mut stack: Vec<NodeId> = graph.input_of(out_id, 0).into_iter().collect();
    while let Some(n) = stack.pop() {
        if n == out_id || !body.insert(n) {
            continue;
        }
        let Some(k) = graph.nodes.get(&n) else { continue };
        for i in 0..k.input_count() {
            stack.extend(graph.input_of(n, i));
        }
    }
    if !body.contains(&pair) {
        return false;
    }
    // Which body nodes depend on the tap (reverse reachability from pair).
    let mut taps: HashSet<NodeId> = HashSet::new();
    taps.insert(pair);
    let mut grew = true;
    while grew {
        grew = false;
        for &n in &body {
            if taps.contains(&n) {
                continue;
            }
            let k = &graph.nodes[&n];
            if (0..k.input_count())
                .any(|i| graph.input_of(n, i).is_some_and(|s| taps.contains(&s)))
            {
                taps.insert(n);
                grew = true;
            }
        }
    }
    for &n in &taps {
        let k = &graph.nodes[&n];
        let staged = k.stage_def().is_some()
            || match k {
                NodeKind::Defocus(_) | NodeKind::LoopOut(_) => true,
                NodeKind::Shader(p) => p.def().ok(),
                _ => false,
            };
        if staged {
            return false;
        }
    }
    true
}

/// Which of `id`'s inputs are wired to a node that exists, by index. A
/// stage's program may depend on it: wiring is topology.
pub fn wired_of(graph: &Graph, id: NodeId) -> Vec<bool> {
    let n = graph.nodes.get(&id).map(|k| k.input_count()).unwrap_or(0);
    (0..n)
        .map(|i| graph.input_of(id, i).is_some_and(|m| graph.nodes.contains_key(&m)))
        .collect()
}

/// Which filter node needs a stage, on which time path, in dependency order
/// (deepest first). Mirrors `emit`'s descent exactly — anything it would not
/// sample gets no stage.
fn filter_stages(graph: &Graph) -> Vec<(StageId, FilterKind)> {
    fn kind_of(graph: &Graph, id: NodeId, tkey: TimeKey) -> Option<FilterKind> {
        if graph.nodes.get(&id).is_some_and(|k| k.stage_def().is_some()) {
            return Some(FilterKind::Node);
        }
        match graph.nodes.get(&id) {
            Some(NodeKind::Defocus(d)) => {
                // A Read on the kernel port is sampled straight off its own
                // texture; anything else renders as an aux. Unwired =
                // procedural.
                let kernel = graph.input_of(id, DEFOCUS_KERNEL_INPUT).and_then(|k| {
                    let slot = graph.nodes.get(&k).and_then(|n| match n {
                        NodeKind::Read { source, .. } => Some(*source),
                        _ => None,
                    })?;
                    Some(graph.slot_for(k, tkey, slot))
                });
                Some(FilterKind::Defocus {
                    mode: d.mode,
                    depth: graph.input_of(id, 2).is_some(),
                    modulate: graph.input_of(id, DEFOCUS_MOD_INPUT).is_some(),
                    kernel_aux: kernel.is_none()
                        && d.mode == crate::graph::DefocusMode::Accurate
                        && graph
                            .input_of(id, DEFOCUS_KERNEL_INPUT)
                            .is_some_and(|k| graph.nodes.contains_key(&k)),
                    kernel,
                })
            }
            // A shader that does not validate gets NO stage: the node
            // falls back to a passthrough (see `emit`), so a typo mid-edit
            // costs you the effect, never the comp.
            Some(NodeKind::Shader(p)) => {
                let def = p.def();
                def.ok().then(|| FilterKind::Shader {
                    inputs: def.inputs.len(),
                    passes: def.passes,
                })
            }
            Some(NodeKind::LoopOut(p)) => {
                loop_ok(graph, id, p.pair).then_some(FilterKind::Loop)
            }
            _ => None,
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn walk(
        graph: &Graph,
        node: NodeId,
        tkey: TimeKey,
        out: &mut Vec<(StageId, FilterKind)>,
        seen: &mut std::collections::HashSet<StageId>,
        stack: &mut Vec<StageId>,
    ) {
        let stage = (node, tkey);
        if stack.contains(&stage) || !seen.insert(stage) {
            return;
        }
        let Some(kind) = graph.nodes.get(&node) else { return };
        // A blending time node samples its input on TWO keys, one per
        // branch, exactly as `emit` does: walking only the plain key left a
        // stage above it unbound on both.
        let inners: Vec<TimeKey> = if graph.disabled.contains(&node) || !kind.is_time_node() {
            vec![tkey]
        } else if kind.time_interp() == crate::graph::TimeInterp::Blend {
            vec![crate::graph::time_key_push_phase(tkey, node, 0), crate::graph::time_key_push_phase(tkey, node, 1)]
        } else {
            vec![crate::graph::time_key_push(tkey, node)]
        };
        if !graph.inputs_unsampled_pub(node, kind) {
            stack.push(stage);
            for &inner in &inners {
                for i in 0..kind.input_count() {
                    if !kind.samples_input(i) {
                        continue; // a Switch's dead branches have no stages
                    }
                    if let Some(src) = graph.input_of(node, i) {
                        // A Layer gives each input its own time branch, and
                        // `emit` samples the stage on THAT key: a stage found
                        // on the stack's own key would never be the one bound,
                        // and the layer read the dummy (it vanished).
                        let k = if kind.per_input_time() && !graph.disabled.contains(&node) {
                            crate::graph::input_time_key(inner, node, i)
                        } else {
                            inner
                        };
                        walk(graph, src, k, out, seen, stack);
                    }
                }
            }
            stack.pop();
        }
        // Post-order: a stage lands after everything it samples, so the
        // compositor can encode the list front to back.
        if !graph.disabled.contains(&node) {
            if let Some(k) = kind_of(graph, node, tkey) {
                out.push((stage, k));
            }
        }
    }
    let Some(view) = graph.view_node() else { return Vec::new() };
    let mut out = Vec::new();
    walk(
        graph,
        view,
        TIME_KEY_ROOT,
        &mut out,
        &mut std::collections::HashSet::new(),
        &mut Vec::new(),
    );
    out
}

/// Generate one fused compute shader evaluating `root` at every output pixel.
/// Output is SCENE-LINEAR: the view transform lives at the blit (see
/// `engine::viewtf`) so cached and exported frames stay scene-referred.
/// `exclude` makes a node emit transparent black — used so a Blur's own
/// subtree can never sample the texture the same pass is writing.
fn gen_shader(
    graph: &Graph,
    layout: &ParamLayout,
    root: Option<NodeId>,
    root_key: TimeKey,
    exclude: Option<StageId>,
) -> GeneratedShader {
    let mut body = String::new();
    let mut sources: Vec<SourceBinding> = Vec::new();
    let mut tmp = 0usize;
    let mut on_path = std::collections::HashSet::new();
    let mut plugin_helpers = std::collections::BTreeMap::new();
    let mut color_curves = std::collections::BTreeMap::new();
    let mut color_fns = std::collections::BTreeMap::new();
    let mut cx = EmitCx {
        graph,
        layout,
        sources: &mut sources,
        exclude,
        plugin_helpers: &mut plugin_helpers,
        color_curves: &mut color_curves,
        color_fns: &mut color_fns,
    };
    let expr = match root {
        Some(r) => emit(&mut cx, r, root_key, "p0", &mut body, &mut tmp, &mut on_path),
        None => "vec4f(0.0)".to_string(),
    };

    let mut w = String::with_capacity(4096);
    let slots = layout.array_len();
    writeln!(w, "struct Params {{ v: array<vec4f, {slots}> }};").unwrap();
    writeln!(w, "@group(0) @binding(0) var<uniform> P: Params;").unwrap();
    writeln!(w, "@group(0) @binding(1) var samp: sampler;").unwrap();
    for (i, _) in sources.iter().enumerate() {
        writeln!(w, "@group(0) @binding({}) var src{}: texture_2d<f32>;", 2 + i, i).unwrap();
    }
    writeln!(
        w,
        "@group(0) @binding({}) var out_tex: texture_storage_2d<rgba16float, write>;",
        2 + sources.len()
    )
    .unwrap();

    w.push_str(
        r#"
fn affine2(r0: vec4f, r1: vec4f, p: vec2f) -> vec2f {
    let q = vec3f(p, 1.0);
    return vec2f(dot(r0.xyz, q), dot(r1.xyz, q));
}
fn grade4(c: vec4f, lift: vec4f, gamma: vec4f, gain: vec4f, offset: vec4f, ct: vec4f, lw: vec3f) -> vec4f {
    let g = max((c.rgb + offset.rgb) * gain.rgb + lift.rgb, vec3f(0.0));
    let t0 = pow(g, 1.0 / max(gamma.rgb, vec3f(1e-4)));
    // Contrast (ct.x) about a pivot (ct.y): a power about the pivot, which
    // is a straight-line contrast in log — right for scene-linear, never
    // crosses zero, no ceiling. Exactly 1 takes the old value untouched, so
    // a grade that never touched contrast renders bit for bit as before.
    let pv = max(ct.y, 1e-4);
    let t = select(pv * pow(t0 / pv, vec3f(ct.x)), t0, ct.x == 1.0);
    // Saturation rides in lift.w, applied last: the pixel lerped about its
    // own luminance, with the working space's weights so the luminance it
    // preserves is the real one.
    let y = dot(t, lw);
    return vec4f(y + (t - y) * lift.w, c.a);
}
fn over(a: vec4f, b: vec4f) -> vec4f {
    return a + b * (1.0 - a.a);
}
// Spline mask: signed distance to a closed polyline packed in the uniform
// (2 points per vec4 after the header and control slots), feathered to
// coverage. Points are PIXELS of the shape's own format, never scaled by
// the output size: a matte on a small plate merged over a big one must stay
// on the plate, not stretch to the big frame.
fn spt(slot: u32, i: u32) -> vec2f {
    let v = P.v[slot + 2u + (i >> 1u)];
    return select(v.xy, v.zw, (i & 1u) == 1u);
}
// Per-sample soft BAND, packed 4 to a vec4 after the positions: the block
// starts 2 + ceil(n / 2) vec4s from the node's own slot. Each lane holds
// both edges (signed px from the outline, + outward) as two 12-bit
// quarter-pixel integers, lo * 4096 + hi — exact in an f32, so the band
// costs the one lane the feather always did (see `spline_band_lane`).
fn sband(slot: u32, i: u32) -> vec2f {
    let q = P.v[slot + 2u + (u32(P.v[slot].x) + 1u) / 2u + (i >> 2u)];
    let k = i & 3u;
    var l = q.w;
    if (k == 0u) { l = q.x; }
    if (k == 1u) { l = q.y; }
    if (k == 2u) { l = q.z; }
    let v = u32(l + 0.5);
    return (vec2f(f32(v >> 12u), f32(v & 4095u)) - 2048.0) * 0.25;
}
fn spline_mask(p: vec2f, slot: u32) -> f32 {
    let h = P.v[slot];
    // (opacity, expand, falloff, invert)
    let ctl = P.v[slot + 1u];
    let n = u32(h.x);
    // No shape at all still honours invert: an inverted empty matte is a
    // full one, which is what makes "invert" a usable garbage switch.
    if (n < 3u) { return select(0.0, ctl.x, ctl.w > 0.5); }
    // Outside the bounding circle (header: radius and centre in pixels, the
    // radius already grown by half the widest feather + a positive expand)
    // no segment can lift coverage off zero, so the walk is skipped: a
    // small shape stops costing its whole outline at every pixel.
    let q = p - h.zw;
    if (dot(q, q) > h.y * h.y) { return select(0.0, ctl.x, ctl.w > 0.5); }
    var d = 1e20;
    var band = vec2f(-0.5, 0.5);
    var sgn = 1.0;
    var j = n - 1u;
    for (var i = 0u; i < n; i = i + 1u) {
        let a = spt(slot, j);
        let b = spt(slot, i);
        let e = a - b;
        let w = p - b;
        let t = clamp(dot(w, e) / max(dot(e, e), 1e-8), 0.0, 1.0);
        let bq = w - e * t;
        let dd = dot(bq, bq);
        if (dd < d) {
            d = dd;
            // The band follows the outline: blend the two endpoints of
            // whichever segment is nearest.
            band = mix(sband(slot, i), sband(slot, j), t);
        }
        let c = vec3<bool>((p.y >= b.y), (p.y < a.y), (e.x * w.y > e.y * w.x));
        if (all(c) || !any(c)) { sgn = -sgn; }
        j = i;
    }
    // Expand moves the outline itself along its own normal (a signed
    // distance offset), so the soft edge rides with it instead of being
    // eaten by it.
    let sd = sgn * sqrt(d) - ctl.y;
    // Full coverage at the band's inner edge, none at its outer: a centred
    // band is the old symmetric feather exactly (0.5 on the outline).
    var cov = clamp((band.y - sd) / max(band.y - band.x, 1e-3), 0.0, 1.0);
    // Falloff shapes the band between linear and smoothstep. Both hit the
    // band's midpoint at 0.5, so the gizmo's rings mean the same thing
    // either way.
    cov = mix(cov, cov * cov * (3.0 - 2.0 * cov), clamp(ctl.z, 0.0, 1.0));
    cov = select(cov, 1.0 - cov, ctl.w > 0.5);
    return cov * ctl.x;
}
// Sample in OUTPUT-PIXEL space; converts to the texture's own UV.
// Projective (homography) coordinate map: rows of the inverse matrix.
fn persp(r0: vec4f, r1: vec4f, r2: vec4f, p: vec2f) -> vec2f {
    let q = vec3f(p, 1.0);
    var w = dot(r2.xyz, q);
    if (abs(w) < 1e-6) { w = 1e-6; }
    return vec2f(dot(r0.xyz, q), dot(r1.xyz, q)) / w;
}
// A MODULATED Transform at strength m (0 = untouched, 1 = the full move),
// output pixel -> source pixel. Span from `s`: three inverse rows, then
// (tx, ty, rot, mode), (sx, sy, cx, cy), pin 0-1, pin 2-3, (w, h, pinned).
// mode 1 is the UV Warp trick (mix the positions); mode 0 rebuilds the
// transform at strength m and inverts it here, so a rotation sweeps arcs.
// At m = 1 both are exactly the unmodulated map (`codegen::transform_
// forward_px`: pin, then pivot-rotate-scale, then translate).
fn xf_mod(s: u32, m: f32, p: vec2f) -> vec2f {
    let a = P.v[s + 3u];
    if (a.w != 0.0) {
        return mix(p, persp(P.v[s], P.v[s + 1u], P.v[s + 2u], p), m);
    }
    let b = P.v[s + 4u];
    let c0 = P.v[s + 5u];
    let c1 = P.v[s + 6u];
    let d = P.v[s + 7u];
    // Scale to the power m (a zoom halfway is geometric, not linear); a
    // flip cannot be half-applied that way, so negative scale mixes.
    var sc = select(mix(vec2f(1.0), b.xy, m), pow(abs(b.xy), vec2f(m)), b.xy > vec2f(0.0));
    sc = select(sc, vec2f(1e-6), abs(sc) < vec2f(1e-6));
    let r = a.z * m;
    let cs = cos(r);
    let sn = sin(r);
    let m00 = cs * sc.x;
    let m01 = -sn * sc.y;
    let m10 = sn * sc.x;
    let m11 = cs * sc.y;
    let tx = a.x * m + b.z - (m00 * b.z + m01 * b.w);
    let ty = a.y * m + b.w - (m10 * b.z + m11 * b.w);
    // Undo the affine.
    let q = p - vec2f(tx, ty);
    let det = m00 * m11 - m01 * m10;
    let u = vec2f(m11 * q.x - m01 * q.y, m00 * q.y - m10 * q.x) / det;
    if (d.z == 0.0) { return u; }
    // Undo the pin, its corners m of the way from the frame's own: build
    // the square->quad homography (Heckbert, as on the CPU) and apply its
    // adjugate — the determinant cancels in the divide.
    let sz = d.xy;
    let q0 = mix(vec2f(0.0, 0.0), c0.xy, m) * sz;
    let q1 = mix(vec2f(1.0, 0.0), c0.zw, m) * sz;
    let q2 = mix(vec2f(1.0, 1.0), c1.xy, m) * sz;
    let q3 = mix(vec2f(0.0, 1.0), c1.zw, m) * sz;
    let d1 = q1 - q2;
    let d2 = q3 - q2;
    let sv = q0 - q1 + q2 - q3;
    var g = 0.0;
    var h = 0.0;
    if (abs(sv.x) > 1e-6 || abs(sv.y) > 1e-6) {
        var den = d1.x * d2.y - d2.x * d1.y;
        if (abs(den) < 1e-9) { den = 1e-9; }
        g = (sv.x * d2.y - d2.x * sv.y) / den;
        h = (d1.x * sv.y - sv.x * d1.y) / den;
    }
    // Rows of H (unit square -> pixels).
    let h0 = vec3f(q1.x - q0.x + g * q1.x, q3.x - q0.x + h * q3.x, q0.x);
    let h1 = vec3f(q1.y - q0.y + g * q1.y, q3.y - q0.y + h * q3.y, q0.y);
    let h2 = vec3f(g, h, 1.0);
    // adj(H) * (u, 1): the columns of adj(H) are cross products of rows.
    let uv = vec3f(u, 1.0);
    let w = cross(h1, h2) * uv.x + cross(h2, h0) * uv.y + cross(h0, h1) * uv.z;
    var wz = w.z;
    if (abs(wz) < 1e-12) { wz = 1e-12; }
    return w.xy / wz * sz;
}
// Control-point curve packed in the shared uniform: points 2 per vec4
// (xy, zw), sorted by x, starting at `base`. Counts and flags live in the
// node's own header, so several curves can share one packed run.
fn curve_pt(base: u32, i: u32) -> vec2f {
    let q = P.v[base + i / 2u];
    return select(q.zw, q.xy, (i & 1u) == 0u);
}
// Secant of segment i -> i+1.
fn curve_sec(base: u32, i: u32) -> f32 {
    let a = curve_pt(base, i);
    let b = curve_pt(base, i + 1u);
    return (b.y - a.y) / max(b.x - a.x, 1e-5);
}
// Fritsch-Carlson tangent: the average of the two secants, clamped so the
// segment can never overshoot (a colour curve that dips below a point it
// passes through reads as a bug). The END tangents depend on `flat`: a
// response curve (a band, a LumaKey; clamped to 0..1) has FLAT ends, which
// makes a two-point smooth curve exactly a smoothstep; a colour curve's end
// tangent is its end segment's secant — the slope the extrapolation past
// that end carries on with, so the curve meets it without a kink (and the
// default two-point ramp stays a straight line).
fn curve_tan(base: u32, n: u32, i: u32, flat: f32) -> f32 {
    if (i == 0u || i + 1u >= n) {
        if (flat != 0.0 || n < 2u) { return 0.0; }
        return curve_sec(base, select(n - 2u, 0u, i == 0u));
    }
    let a = curve_sec(base, i - 1u);
    let b = curve_sec(base, i);
    if (a * b <= 0.0) { return 0.0; }
    let m = 0.5 * (a + b);
    let lim = 3.0 * min(abs(a), abs(b));
    return clamp(m, -lim, lim);
}
// NOTE: the smoothing parameter is `soft`, NOT `smooth` — `smooth` is a WGSL
// reserved keyword (an interpolation qualifier), and naming it that makes
// EVERY generated shader fail to parse, not just one with a Curve in it.
fn curve_eval(base: u32, n: u32, x: f32, soft: f32, flat: f32) -> f32 {
    if (n == 0u) { return x; }
    var prev = curve_pt(base, 0u);
    if (n == 1u) { return prev.y; }
    // Outside the ends the curve EXTRAPOLATES along the end segment rather
    // than flattening: a colour curve that clamps at its last point crushes
    // every highlight above it, and this comp is scene-linear so there are
    // plenty. Consumers that need a bounded result clamp for themselves.
    if (x <= prev.x) { return prev.y + (x - prev.x) * curve_sec(base, 0u); }
    let fin = curve_pt(base, n - 1u);
    if (x >= fin.x) { return fin.y + (x - fin.x) * curve_sec(base, n - 2u); }
    for (var i = 1u; i < n; i++) {
        let p = curve_pt(base, i);
        if (x <= p.x) {
            let h = max(p.x - prev.x, 1e-5);
            let t = (x - prev.x) / h;
            if (soft == 0.0) { return mix(prev.y, p.y, t); }
            // Cubic Hermite on the clamped tangents.
            let m0 = curve_tan(base, n, i - 1u, flat) * h;
            let m1 = curve_tan(base, n, i, flat) * h;
            let t2 = t * t;
            let t3 = t2 * t;
            return (2.0 * t3 - 3.0 * t2 + 1.0) * prev.y
                + (t3 - 2.0 * t2 + t) * m0
                + (-2.0 * t3 + 3.0 * t2) * p.y
                + (t3 - t2) * m1;
        }
        prev = p;
    }
    return prev.y;
}
// Curve node: header [n_master, n_r, n_g, n_b], then [smooth, split, ..],
// then the master run followed by each channel's own. Every run is padded
// to an even point count, so each starts on its own vec4.
fn curve3(c: vec4f, slot: u32) -> vec4f {
    let n = P.v[slot];
    let f = P.v[slot + 1u];
    let base = slot + 2u;
    let nm = u32(n.x);
    var v = vec3f(0.0);
    for (var k = 0u; k < 3u; k++) {
        var first = base;
        var cnt = nm;
        let own = u32(n[k + 1u]);
        if (f.y != 0.0 && own > 0u) {
            var vec4s = (nm + 1u) / 2u;
            for (var j = 0u; j < k; j++) { vec4s += (u32(n[j + 1u]) + 1u) / 2u; }
            first = base + vec4s;
            cnt = own;
        }
        v[k] = curve_eval(first, cnt, c[k], f.x, 0.0);
    }
    return vec4f(v, c.a);
}
// One channel of a plate being read as a displacement vector.
fn disp_chan(v: vec4f, k: f32, lw: vec3f) -> f32 {
    if (k == 1.0) { return v.g; }
    if (k == 2.0) { return v.b; }
    if (k == 3.0) { return v.a; }
    if (k == 4.0) { return dot(v.rgb, lw); }
    return v.r;
}
// Star flare, drawn in polar coordinates about a centre: a periodic
// function of the ANGLE makes the rays, a function of the RADIUS gives them
// their reach, and an inner glow sits under both. Pointwise, so it fuses
// into the main pass like any other generator.
fn flare_lobe(a: f32, n: f32, sharp: f32, soft: f32) -> f32 {
    // |cos(a * n/2)| has exactly n lobes around the circle.
    let base = abs(cos(a * n * 0.5));
    var v = pow(base, sharp);
    if (soft > 0.0) {
        // A far wider, dimmer lobe under the sharp one: the haze that makes
        // a ray read as light rather than as a drawn line.
        v = max(v, pow(base, max(sharp * 0.15, 1.0)) * soft * 0.6);
    }
    return v;
}
// Which ray this angle belongs to — 1 for every other one.
fn flare_odd(a: f32, n: f32) -> bool {
    let k = floor(a * n / 6.2831853 + 0.5);
    return fract(k * 0.5) > 0.25;
}
fn flare(p: vec2f, slot: u32) -> vec4f {
    let size = max(P.v[0].xy, vec2f(1.0));
    let h0 = P.v[slot];        // centre.xy (normalized), radius, intensity
    let h1 = P.v[slot + 1u];   // scale.xy, gamma, rotation (radians)
    let h2 = P.v[slot + 2u];   // rays, ray intensity, ray gamma, thickness
    let h3 = P.v[slot + 3u];   // softness, odd length, falloff-to-zero, glow
    let col = P.v[slot + 4u].rgb;

    let c = h0.xy * size;
    let d = (p - c) / max(h1.xy, vec2f(1e-3));
    let reach = max(h0.z, 1e-4) * size.x;
    let r = length(d) / reach;                 // 1.0 AT the radius
    // Minus, so a POSITIVE rotation turns the star the positive way round.
    let a = atan2(d.y, d.x) - h1.w;

    // Radial reach. Falloff-to-zero is what a convolution kernel needs (no
    // edge where it is cropped); the bounded inverse square is what looks
    // right drawn in place, and it is BOUNDED so the centre is not an inf.
    let lin = max(1.0 - r, 0.0);
    let inv = 1.0 / (1.0 + 16.0 * r * r);
    let fall = select(inv, lin * lin, h3.z != 0.0);

    let n = clamp(h2.x, 1.0, 64.0);
    let sharp = max(1.0, 1.0 / max(h2.w, 1e-3));
    // The ray, angular profile times radial reach. The odd-ray gamma goes
    // on the PRODUCT: at the very centre of a ray the angular term is 1, so
    // a gamma on that alone would only thin the ray — on the product it
    // eats into the length, which is the control's whole point.
    var ray = flare_lobe(a, n, sharp, max(h3.x, 0.0)) * pow(fall, max(h2.z, 1e-3));
    let odd = max(h3.y, 1e-3);
    if (odd != 1.0 && flare_odd(a, n)) {
        ray = pow(ray, odd);
    }
    let rays = ray * max(h2.y, 0.0);
    // The glow is the same falloff without the angular term, pulled in
    // tight so it reads as a core rather than a wash.
    let glow = pow(fall, 3.0) * max(h3.w, 0.0);

    let v = pow(max(rays + glow, 0.0), 1.0 / max(h1.z, 1e-3)) * max(h0.w, 0.0);
    let rgb = col * v;
    // Emissive, but alpha carries the coverage so an Over reads as well as
    // an Add — premultiplied, so rgb never exceeds it by construction.
    return vec4f(rgb, clamp(v, 0.0, 1.0));
}
// Procedural grid: pixel distance to the nearest line in each axis, so the
// line stays `width` px thick whatever the spacing.
fn grid_px(p: vec2f, slot: u32) -> vec4f {
    let h = P.v[slot];
    let off = P.v[slot + 1u];
    let sp = max(h.xy, vec2f(1e-3));
    let q = abs(fract((p - off.xy) / sp + 0.5) - 0.5) * sp;
    let d = min(q.x, q.y) - max(h.z, 0.0) * 0.5;
    // Half a pixel of coverage either side keeps thin lines from crawling.
    let cov = clamp(0.5 - d, 0.0, 1.0);
    return mix(P.v[slot + 3u], P.v[slot + 2u], cov);
}
fn rgb2hsv(c: vec3f) -> vec3f {
    let mx = max(c.r, max(c.g, c.b));
    let mn = min(c.r, min(c.g, c.b));
    let d = mx - mn;
    var h = 0.0;
    if (d > 1e-7) {
        if (mx == c.r) { h = (c.g - c.b) / d; }
        else if (mx == c.g) { h = 2.0 + (c.b - c.r) / d; }
        else { h = 4.0 + (c.r - c.g) / d; }
        h = fract(h / 6.0 + 1.0);
    }
    // Value is kept unclamped so HDR survives the round trip.
    return vec3f(h, select(0.0, d / mx, mx > 1e-7), mx);
}
fn hsv2rgb(c: vec3f) -> vec3f {
    let h = fract(c.x) * 6.0;
    let k = abs(h - vec3f(3.0, 2.0, 4.0)) - vec3f(1.0, 1.0, 1.0);
    let rgb = clamp(vec3f(k.x, 2.0 - k.y, 2.0 - k.z), vec3f(0.0), vec3f(1.0));
    return c.z * mix(vec3f(1.0), rgb, c.y);
}
// Six hue bands 60 degrees apart, each with its own hue shift, saturation
// and gain, blended by hue distance so corrections cross-fade rather than
// banding. header slot = (global sat, falloff, .., ..).
fn tint(c: vec4f, slot: u32) -> vec4f {
    var hsv = rgb2hsv(max(c.rgb, vec3f(0.0)));
    var wsum = 0.0;
    var shift = 0.0;
    var sat = 0.0;
    var gain = 0.0;
    var dens = 0.0;
    let g = P.v[slot + 6u];
    // falloff 0 = broad cosine lobes, 1 = narrow wedges.
    let sharp = 1.0 + 7.0 * clamp(g.y, 0.0, 1.0);
    for (var i = 0u; i < 6u; i = i + 1u) {
        let centre = f32(i) / 6.0;
        var d = abs(hsv.x - centre);
        d = min(d, 1.0 - d) * 6.0; // 0 at the centre, 1 at a neighbour
        let w = pow(max(1.0 - d, 0.0), sharp);
        let b = P.v[slot + i];
        wsum += w;
        shift += w * b.x;
        sat += w * b.y;
        gain += w * b.z;
        dens += w * b.w;
    }
    if (wsum > 1e-6) {
        // Fade the band correction out as the pixel desaturates: a neutral
        // has no meaningful hue, so a red-band gain must not tint the greys.
        // The GLOBAL saturation still applies to everything.
        let w = clamp(hsv.y, 0.0, 1.0);
        // Band hue leans fade out on neutrals; the GLOBAL hue is a spin of
        // the whole wheel, and on a grey it is a no-op anyway.
        hsv.x = fract(hsv.x + (shift / wsum) * w / 360.0 + g.z / 360.0 + 1.0);
        hsv.y = clamp(hsv.y * mix(1.0, sat / wsum, w) * g.x, 0.0, 1.0);
        hsv.z = hsv.z * mix(1.0, gain / wsum, w);
        // Density, the control a six-vector normally lacks: DEEPEN the
        // colour — darken it in proportion to how colourful it already is,
        // and enrich what is left. Both halves are scaled by saturation, so
        // a grey cannot be made denser: density is a property of colour.
        let d = (dens / wsum) * w + g.w;
        hsv.z = hsv.z * (1.0 - d * hsv.y);
        hsv.y = clamp(hsv.y * (1.0 + d), 0.0, 1.0);
    }
    return vec4f(hsv2rgb(hsv), c.a);
}
// Channel rewiring: four one-hot rows pick an input channel each, and the
// constant column supplies 0/1 where an output is pinned rather than picked.
// Branch-free, and re-pointing a channel is a uniform write.
fn shuffle4(c: vec4f, slot: u32) -> vec4f {
    return vec4f(
        dot(P.v[slot], c),
        dot(P.v[slot + 1u], c),
        dot(P.v[slot + 2u], c),
        dot(P.v[slot + 3u], c),
    ) + P.v[slot + 4u];
}
// --- fractal noise -------------------------------------------------------
// An INTEGER hash, not the usual sin() trick: sin's precision varies between
// drivers, and a compositor's noise has to be the same picture on every
// machine that opens the setup.
fn fbm_hash(x: u32) -> u32 {
    var h = x;
    h ^= h >> 16u;
    h *= 0x7feb352du;
    h ^= h >> 15u;
    h *= 0x846ca68bu;
    h ^= h >> 16u;
    return h;
}
// A unit vector per lattice corner, drawn uniformly over the sphere.
fn fbm_grad(c: vec3f, seed: u32) -> vec3f {
    let ix = bitcast<u32>(i32(c.x));
    let iy = bitcast<u32>(i32(c.y));
    let iz = bitcast<u32>(i32(c.z));
    var h = fbm_hash(ix ^ fbm_hash(iy ^ fbm_hash(iz ^ seed)));
    let a = f32(h & 0xffffu) / 65535.0 * 6.28318530718;
    h = fbm_hash(h);
    let z = f32(h & 0xffffu) / 65535.0 * 2.0 - 1.0;
    let r = sqrt(max(1.0 - z * z, 0.0));
    return vec3f(cos(a) * r, sin(a) * r, z);
}
// Gradient (Perlin) noise with the quintic fade, roughly -1..1.
fn fbm_noise(p: vec3f, seed: u32) -> f32 {
    let i = floor(p);
    let f = p - i;
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    var acc = 0.0;
    for (var k = 0u; k < 8u; k = k + 1u) {
        let c = vec3f(f32(k & 1u), f32((k >> 1u) & 1u), f32((k >> 2u) & 1u));
        let g = fbm_grad(i + c, seed);
        let d = f - c;
        let w = mix(vec3f(1.0) - u, u, c);
        acc += dot(g, d) * w.x * w.y * w.z;
    }
    return acc * 1.6;
}
// Summed octaves. The octave COUNT is a uniform loop bound, so turning it up
// is a param edit and never a recompile; the cap keeps the worst case bounded.
// The same sum with its parameters passed IN rather than read from a slot.
// The noise-field graph needs this: there, lacunarity or gain may themselves
// be the output of another noise, so there is no uniform lane to read.
fn fbm_core(p0: vec3f, oct_in: u32, lacunarity: f32, gain: f32, seed: u32, mode: f32) -> f32 {
    let oct = clamp(oct_in, 1u, __FBM_MAX__u);
    var amp = 1.0;
    var freq = 1.0;
    var sum = 0.0;
    var norm = 0.0;
    var norm2 = 0.0;
    // Each mode folds its octaves differently AND lands in a different
    // range: rectifying the noise (turbulence, ridged) leaves a DC bias that
    // a single shared mapping would just turn into a washed-out field.
    for (var i = 0u; i < oct; i = i + 1u) {
        let n = fbm_noise(p0 * freq, seed + i * 8191u);
        if (mode == 1.0) {
            sum += abs(n) * amp;              // billows: dark, bright veins
        } else if (mode == 2.0) {
            let r = 1.0 - abs(n);
            sum += r * r * amp;               // squared: sharp ridges
        } else {
            sum += n * amp;                   // signed, symmetric
        }
        norm += amp;
        norm2 += amp * amp;
        amp *= gain;
        freq *= lacunarity;
    }
    // Fill 0..1. Summed noise never reaches its theoretical bounds — the
    // octaves would all have to peak together — so the raw field lands well
    // inside the range: fbm ~0.09..0.88, turbulence ~0..0.5, ridged
    // ~0.26..0.98. A generator that cannot output 1.0 is a bad matte: mask a
    // grade with it and the gain never reaches full strength. These spans
    // are MEASURED at the default octave count (see the fbm_range test) and
    // are approximate for other settings, so the result is clamped.
    if (mode == 0.0) {
        // Independent octaves add in QUADRATURE. Normalising by the sum of
        // amplitudes (the obvious choice) squeezes the field toward mid grey.
        let x = sum / max(sqrt(norm2), 1e-6);
        return clamp(0.5 + 0.5 * x * 1.3, 0.0, 1.0);
    }
    let t = sum / max(norm, 1e-6);
    if (mode == 1.0) {
        return clamp(t * 2.0, 0.0, 1.0);          // turbulence sits low
    }
    return clamp((t - 0.25) * 1.42, 0.0, 1.0);    // ridged sits high
}
fn fbm_field(p0: vec3f, slot: u32, seed_shift: u32) -> f32 {
    let h = P.v[slot];
    let g = P.v[slot + 2u];
    return fbm_core(p0, u32(h.x), h.y, h.z, u32(max(h.w, 0.0)) + seed_shift, g.y);
}
fn fbm_rgba(pos: vec2f, slot: u32) -> vec4f {
    let sc = P.v[slot + 1u];
    let g = P.v[slot + 2u];
    // Feature SIZE in pixels, so the look holds when the format changes.
    let q = vec3f(
        (pos.x + sc.z) / max(sc.x, 1e-3),
        (pos.y + sc.w) / max(sc.y, 1e-3),
        g.x,
    );
    let r = fbm_field(q, slot, 0u);
    if (g.z != 0.0) {
        return vec4f(r, r, r, 1.0);
    }
    // Decorrelated channels: three fields, three seeds.
    return vec4f(
        r,
        fbm_field(q, slot, 0x9e3779b9u),
        fbm_field(q, slot, 0x85ebca6bu),
        1.0,
    );
}
fn sample_px(t: texture_2d<f32>, p: vec2f) -> vec4f {
    let ts = vec2f(textureDimensions(t));
    let uv = p / ts;
    let inside = all(uv >= vec2f(0.0)) && all(uv <= vec2f(1.0));
    let c = textureSampleLevel(t, samp, uv, 0.0);
    return select(vec4f(0.0), c, inside);
}
// --- cryptomatte ------------------------------------------------------------
// Unfiltered texel under the output pixel (plate scaled onto the canvas the
// way sample_px does, but never interpolated: an id is not a colour).
fn crypto_load(t: texture_2d<f32>, p: vec2f) -> vec4f {
    let ts = vec2f(textureDimensions(t));
    let q = floor(p / P.v[0].xy * ts);
    let inside = all(q >= vec2f(0.0)) && all(q < ts);
    let c = textureLoad(t, vec2i(q), 0);
    return select(vec4f(0.0), c, inside);
}
// Three 11-bit chunks back into the id's 32 bits (see cryptomatte.rs).
fn crypto_id(c: vec3f) -> u32 {
    let k = vec3u(round(c));
    return k.x | (k.y << 11u) | (k.z << 22u);
}
// An 11-bit chunk as a value in [1,4), exact in f16 (cryptomatte::encode_lit).
fn crypto_lit(k: f32) -> f32 {
    return select(2.0 + (k - 1024.0) / 512.0, 1.0 + k / 1024.0, k < 1024.0);
}
// 1.0 when `id` is one of the node's picks: header at `slot` holds the
// count, then (hi, lo) 16-bit halves two picks per vec4.
fn crypto_hit(id: u32, slot: u32) -> f32 {
    let n = u32(P.v[slot].y);
    var hit = false;
    for (var i = 0u; i < n; i++) {
        let p = P.v[slot + 1u + (i >> 1u)];
        let odd = (i & 1u) == 1u;
        let hi = u32(select(p.x, p.z, odd));
        let lo = u32(select(p.y, p.w, odd));
        hit = hit || ((hi << 16u) | lo) == id;
    }
    return select(0.0, 1.0, hit);
}
"#,
    );
    // One source of truth for the octave cap: the shader text carries the
    // same constant the CPU clamps the param to.
    let cap = crate::field::FBM_MAX_OCTAVES;
    let fixed = w.replace("__FBM_MAX__", &cap.to_string());
    w = fixed;
    w.push_str(&crate::keyer::keyer_wgsl(graph.working));
    // The noise-field helpers, ahead of the per-node functions a Noise node
    // emits into `color_fns`.
    w.push_str(crate::field::FIELD_WGSL);
    for h in color_curves.values() {
        w.push_str(h);
    }
    for h in color_fns.values() {
        w.push_str(h);
    }
    for h in plugin_helpers.values() {
        w.push_str(h);
        w.push('\n');
    }
    let tail = expr;
    writeln!(
        w,
        r#"
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {{
    let dims = textureDimensions(out_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {{ return; }}
    let p0 = vec2f(gid.xy) + vec2f(0.5);
{body}    textureStore(out_tex, gid.xy, {tail});
}}
"#
    )
    .unwrap();

    GeneratedShader { wgsl: w, sources }
}

/// The function converting `node`'s plate into the comp's working space,
/// emitted into the shared helper sets. `None` when the plate is already in
/// the working space, in which case no code is generated at all — an
/// already-ACEScg plate costs exactly what it did before colour management.
fn idt_fn(cx: &mut EmitCx, node: NodeId) -> Option<String> {
    use std::hash::{Hash, Hasher};
    let input = match cx.graph.nodes.get(&node) {
        Some(NodeKind::Read { input, .. }) => *input,
        Some(NodeKind::Blender(p)) => p.effective_space(),
        // Every model we call hands back display-referred sRGB, except a
        // depth or normals take: stored values, read as they are.
        Some(NodeKind::GenAi(p)) if p.data() => crate::color::InputSpace::Working,
        Some(NodeKind::GenAi(_)) => {
            crate::color::InputSpace::Explicit(crate::color::Colorspace::SRGB)
        }
        // The tool writes linear Rec.709 (its own convention: sRGB primaries,
        // no curve, premultiplied).
        // Retimed frames are the input as it was rendered: linear Rec.709.
        Some(NodeKind::Rifetime(_)) => crate::color::InputSpace::Explicit(crate::color::Colorspace::LINEAR_REC709),
        // So are a Frame Fix's flow-mode frames.
        Some(NodeKind::FrameFix(_)) => crate::color::InputSpace::Explicit(crate::color::Colorspace::LINEAR_REC709),
        Some(NodeKind::CorridorKey(_)) => {
            crate::color::InputSpace::Explicit(crate::color::Colorspace::LINEAR_REC709)
        }
        // Their results are written linear Rec.709, premultiplied.
        // (and a DiffusionLight latlong: linear Rec.709, alpha 1)
        Some(NodeKind::Upscale(_)) | Some(NodeKind::SeedVr(_)) | Some(NodeKind::HdrCnn(_)) | Some(NodeKind::DiffLight(_)) => {
            crate::color::InputSpace::Explicit(crate::color::Colorspace::LINEAR_REC709)
        }
        // The tool writes scene-linear ACEScg (`--exr-colorspace acescg`).
        Some(NodeKind::LtxHdr(_)) => crate::color::InputSpace::Explicit(crate::color::Colorspace::ACESCG),
        _ => crate::color::InputSpace::Working,
    };
    let working = cx.graph.working;
    let src = input.resolve(working);
    let t = crate::color::Transform::new(src, working);
    if t.is_identity() {
        return None;
    }
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (src, working).hash(&mut h);
    let name = format!("cs_idt_{:016x}", h.finish());
    if !cx.color_fns.contains_key(&name) {
        let def = t.wgsl(&name, cx.color_curves);
        cx.color_fns.insert(name.clone(), def);
    }
    Some(name)
}

struct EmitCx<'a> {
    graph: &'a Graph,
    layout: &'a ParamLayout,
    sources: &'a mut Vec<SourceBinding>,
    exclude: Option<StageId>,
    /// Plugin helper WGSL, emitted once per definition (keyed by source
    /// hash) ahead of the entry point.
    plugin_helpers: &'a mut std::collections::BTreeMap<u64, String>,
    /// Transfer-curve helpers (`cs_srgb_to_lin` and friends), keyed by
    /// function name so two Reads sharing a curve emit it once.
    color_curves: &'a mut std::collections::BTreeMap<String, String>,
    /// The per-space input transforms that CALL those helpers. Kept separate
    /// so they can be emitted strictly after them: WGSL wants a function
    /// declared before it is used, and one map's alphabetical order would
    /// put `cs_idt_*` ahead of `cs_srgb_to_lin`.
    color_fns: &'a mut std::collections::BTreeMap<String, String>,
}

impl EmitCx<'_> {
    fn bind(&mut self, s: SourceBinding) -> usize {
        match self.sources.iter().position(|b| *b == s) {
            Some(i) => i,
            None => {
                self.sources.push(s);
                self.sources.len() - 1
            }
        }
    }
}

/// What a node module emits itself with — see [`crate::nodes::NodeDef::emit`].
/// One per node visit: `emit` builds it over its own state and hands it to
/// the node, so a node never sees the recursion, the bindings or the
/// temporaries directly.
pub struct Emit<'e, 'a> {
    cx: &'e mut EmitCx<'a>,
    node: NodeId,
    tkey: TimeKey,
    pos: &'e str,
    body: &'e mut String,
    tmp: &'e mut usize,
    on_path: &'e mut std::collections::HashSet<NodeId>,
}

impl<'e, 'a> Emit<'e, 'a> {
    fn new(
        cx: &'e mut EmitCx<'a>,
        node: NodeId,
        tkey: TimeKey,
        pos: &'e str,
        body: &'e mut String,
        tmp: &'e mut usize,
        on_path: &'e mut std::collections::HashSet<NodeId>,
    ) -> Self {
        Emit { cx, node, tkey, pos, body, tmp, on_path }
    }

    /// Input `i` as a WGSL expression; transparent black when unwired.
    pub fn input(&mut self, i: usize) -> String {
        match self.cx.graph.input_of(self.node, i) {
            Some(src) => emit(self.cx, src, self.tkey, self.pos, self.body, self.tmp, self.on_path),
            None => "vec4f(0.0)".into(),
        }
    }

    /// Input `i` fused again at ANOTHER position (a WGSL expression in
    /// pixels) — what a node that bends its input reads through (the SDF
    /// node's refraction). The subtree is re-evaluated there, ALU only.
    pub fn input_at(&mut self, i: usize, pos: &str) -> String {
        match self.cx.graph.input_of(self.node, i) {
            Some(src) => emit(self.cx, src, self.tkey, pos, self.body, self.tmp, self.on_path),
            None => "vec4f(0.0)".into(),
        }
    }

    /// Reader slot `source` as seen by this node, as a WGSL texture name —
    /// what a node playing frames off disk samples.
    pub fn reader(&mut self, source: usize) -> String {
        let slot = self.cx.graph.slot_for(self.node, self.tkey, source);
        let idx = self.cx.bind(SourceBinding::Reader(slot));
        format!("src{idx}")
    }

    /// The node being emitted — what a node naming its own WGSL functions
    /// keys them on.
    pub fn node_id(&self) -> NodeId {
        self.node
    }

    /// The mask on input `i`, when one is wired. A dangling edge counts as
    /// unwired.
    pub fn mask(&mut self, i: usize) -> Option<String> {
        let m = self.cx.graph.input_of(self.node, i).filter(|m| self.cx.graph.nodes.contains_key(m))?;
        Some(emit(self.cx, m, self.tkey, self.pos, self.body, self.tmp, self.on_path))
    }

    /// `v` masked: crossfaded against the untouched `input` by the mask's
    /// coverage, the way every masked pointwise node does it. With no mask
    /// port, or nothing wired to it, `v` stands as it is.
    pub fn masked(&mut self, mask_input: Option<usize>, input: &str, v: String) -> String {
        let Some(mv) = mask_input.and_then(|i| self.mask(i)) else { return v };
        let v2 = self.fresh();
        self.line(format_args!("let {v2} = mix({input}, {v}, {mv}.x);"));
        v2
    }

    /// Whether anything is wired to input `i`. A dangling edge (source
    /// deleted) counts as unwired.
    pub fn has_input(&self, i: usize) -> bool {
        self.cx.graph.input_of(self.node, i).is_some_and(|m| self.cx.graph.nodes.contains_key(&m))
    }

    /// This node's stage result — the texture its passes left behind —
    /// sampled at the current position. A pass boundary is just a read.
    pub fn stage_result(&mut self) -> String {
        let idx = self.cx.bind(SourceBinding::Blur((self.node, self.tkey)));
        let v = self.fresh();
        let pos = self.pos;
        self.line(format_args!("let {v} = sample_px(src{idx}, {pos});"));
        v
    }

    /// A stage result masked: crossfaded against the input subtree, which
    /// is RE-FUSED inline here — ALU only when a mask is wired, no extra
    /// texture binding, nothing for the compositor to do.
    pub fn masked_stage(&mut self, mask_input: Option<usize>, v: String) -> String {
        let Some(m) = mask_input.filter(|i| self.has_input(*i)) else { return v };
        let orig = self.input(0);
        let mv = self.mask(m).expect("checked wired above");
        let v2 = self.fresh();
        self.line(format_args!("let {v2} = mix({orig}, {v}, {mv}.x);"));
        v2
    }

    /// The WGSL expression for the pixel position being evaluated — what a
    /// generator reads its picture from.
    pub fn pos(&self) -> &str {
        self.pos
    }

    /// This node's LUT table texture (`SourceBinding::Lut`), as a WGSL
    /// texture name.
    pub fn lut_table(&mut self) -> String {
        let idx = self.cx.bind(SourceBinding::Lut(self.node));
        format!("src{idx}")
    }

    /// This node's glyph atlas (`SourceBinding::Text`), as a WGSL texture
    /// name.
    pub fn text_table(&mut self) -> String {
        let idx = self.cx.bind(SourceBinding::Text(self.node));
        format!("src{idx}")
    }

    /// This node's particle layer (`SourceBinding::Plasma`), as a WGSL
    /// texture name.
    pub fn plasma_layer(&mut self) -> String {
        let idx = self.cx.bind(SourceBinding::Plasma(self.node));
        format!("src{idx}")
    }

    /// This node's particle layer (`SourceBinding::Particles`), as a WGSL
    /// texture name.
    pub fn particles_layer(&mut self) -> String {
        let idx = self.cx.bind(SourceBinding::Particles(self.node));
        format!("src{idx}")
    }

    /// A Splat node's layer (`SourceBinding::Splat`) — `host`'s, its
    /// depth target when `depth` — as a WGSL texture name.
    pub fn splat_layer(&mut self, host: NodeId, depth: bool) -> String {
        let idx = self.cx.bind(SourceBinding::Splat(host, depth));
        format!("src{idx}")
    }

    /// The node being emitted.
    pub fn node(&self) -> NodeId {
        self.node
    }

    /// This node's first vec4 slot in the uniform block.
    pub fn slot(&self) -> usize {
        self.cx.layout.node_slot[&self.node]
    }

    /// The comp's working space.
    pub fn working(&self) -> crate::color::Colorspace {
        self.cx.graph.working
    }

    /// A fresh temporary.
    pub fn fresh(&mut self) -> String {
        fresh(self.tmp)
    }

    /// One statement in the fused body.
    pub fn line(&mut self, s: std::fmt::Arguments) {
        writeln!(self.body, "    {s}").unwrap();
    }

    /// The shared helper maps: transfer-curve helpers, then the functions
    /// that call them — emitted strictly in that order, so a node that
    /// declares one of each gets them in the order WGSL needs.
    pub fn color_helpers(
        &mut self,
    ) -> (
        &mut std::collections::BTreeMap<String, String>,
        &mut std::collections::BTreeMap<String, String>,
    ) {
        let cx = &mut *self.cx;
        (&mut *cx.color_curves, &mut *cx.color_fns)
    }
}

/// Recursively emit statements computing `node`'s color at position var
/// `pos` (output-pixel space). Returns the WGSL expression for the color.
/// `on_path` guards against cycles (the UI can wire one).
/// `a` composited onto `b` by one blend op, as a WGSL expression. ONE place
/// for the formulas: a Blend node and every layer of a Layer stack are the
/// same operation, and two copies of "screen" would eventually disagree.
fn blend_expr(op: crate::graph::BlendOp, a: &str, b: &str) -> String {
    use crate::graph::BlendOp;
    match op {
        BlendOp::Over => format!("over({a}, {b})"),
        BlendOp::Add => format!("{a} + {b}"),
        BlendOp::Multiply => format!("{a} * {b}"),
        BlendOp::Screen => {
            format!("vec4f({a}.rgb + {b}.rgb - {a}.rgb * {b}.rgb, over({a}, {b}).a)")
        }
        BlendOp::Difference => format!("vec4f(abs({a}.rgb - {b}.rgb), over({a}, {b}).a)"),
        // No compositing at all: A's pixels stand in for B's. WHERE that
        // happens is the matte's job (a Blend's third input); in a layer
        // stack it is the layer's own alpha and opacity.
        BlendOp::Replace => a.to_string(),
    }
}

fn emit(
    cx: &mut EmitCx,
    node: NodeId,
    tkey: TimeKey,
    pos: &str,
    body: &mut String,
    tmp: &mut usize,
    on_path: &mut std::collections::HashSet<NodeId>,
) -> String {
    if cx.exclude == Some((node, tkey)) {
        return "vec4f(0.0)".into();
    }
    let Some(kind) = cx.graph.nodes.get(&node) else {
        return "vec4f(0.0)".into();
    };
    if !on_path.insert(node) {
        return "vec4f(0.0)".into(); // cycle
    }
    // Bypassed node: its pass-through input (0, a Blend's background)
    // flows straight through (Reads go black).
    if cx.graph.disabled.contains(&node) {
        // A bypassed mask source means "mask off": coverage 1 everywhere,
        // so the consuming effect applies to ALL pixels, not none.
        // A node that merely FOLLOWS a matte pipe (an Invert of a matte)
        // passes its matte through instead, like a bypassed picture op.
        if kind.output_kind() == crate::graph::PortKind::Mask && !kind.follows_pipe(0) {
            on_path.remove(&node);
            return "vec4f(1.0)".into();
        }
        let through = kind.bypass_input();
        let result = match cx.graph.input_of(node, through) {
            Some(input) if kind.input_count() > through => {
                emit(cx, input, tkey, pos, body, tmp, on_path)
            }
            _ => "vec4f(0.0)".into(),
        };
        on_path.remove(&node);
        return result;
    }
    let result = match kind {
        // Only the live branch is emitted, so the others cost nothing —
        // no sampling, no filter stages, no decode.
        NodeKind::Switch(p) => match cx.graph.input_of(node, p.live()) {
            Some(input) => emit(cx, input, tkey, pos, body, tmp, on_path),
            None => "vec4f(0.0)".into(),
        },
        // Safety net: the editor resolves wires THROUGH dots, so codegen
        // never sees one. If it ever does, it passes its input straight on —
        // no temporary, no cost, exactly what a wire does.
        NodeKind::Dot { .. } | NodeKind::TimeRange(_) => match cx.graph.input_of(node, 0) {
            Some(i) => emit(cx, i, tkey, pos, body, tmp, on_path),
            None => "vec4f(0.0)".into(),
        },
        NodeKind::Read { source, single, .. } => {
            // Which plate this leaf samples depends on the time path that
            // reached it: the same Read on two paths is two reader slots,
            // each holding its own frame.
            let idx = cx.bind(SourceBinding::Reader(cx.graph.slot_for(node, tkey, *source)));
            let v = fresh(tmp);
            // The texture may be a smaller copy of the plate and may hold
            // one box of it: scale into its pixels, then step back by the
            // box's corner (sample_px is black outside it).
            let at = cx
                .layout
                .window_slot
                .get(&(node, tkey))
                .map(|s| format!("({pos} * P.v[{s}].zw - P.v[{s}].xy)"));
            let pos = at.as_deref().unwrap_or(pos);
            if single.is_on() {
                // One stored channel (an R texture, `decode::Pack`): the
                // value is the whole pixel, alpha included — a white
                // element over black, which is what a matte plate is.
                writeln!(body, "    let {v} = vec4f(sample_px(src{idx}, {pos}).r);").unwrap();
            } else {
                writeln!(body, "    let {v} = sample_px(src{idx}, {pos});").unwrap();
            }
            // Input transform: the plate enters the comp's working space
            // here, at the only place its own encoding is still known.
            //
            // RGB only. On a premultiplied plate with a NON-linear input
            // curve the mathematically correct order is unpremult ->
            // transform -> repremult; a linear matrix commutes with the
            // premultiply and needs no such care. Log-encoded sources
            // essentially never carry alpha, and the Premult node exists for
            // the case that does, so this matches what a compositor's Read does
            // rather than silently guessing at the plate's convention.
            match idt_fn(cx, node) {
                Some(f) => {
                    let c = fresh(tmp);
                    writeln!(body, "    let {c} = vec4f({f}({v}.rgb), {v}.a);").unwrap();
                    c
                }
                None => v,
            }
        }
        NodeKind::Custom { type_id, .. } => {
            match crate::plugins::get(type_id) {
                // Not installed: input 0 passes through, exactly like a
                // bypass — the comp stays renderable, the data stays.
                None => match cx.graph.input_of(node, 0) {
                    Some(input) => emit(cx, input, tkey, pos, body, tmp, on_path),
                    None => "vec4f(0.0)".into(),
                },
                Some(def) => {
                    let slot = cx.layout.node_slot.get(&node).copied().unwrap_or(0);
                    let inputs: Vec<String> = (0..def.inputs)
                        .map(|i| match cx.graph.input_of(node, i) {
                            Some(input) => emit(cx, input, tkey, pos, body, tmp, on_path),
                            None => "vec4f(0.0)".into(),
                        })
                        .collect();
                    if !def.helpers.is_empty() {
                        cx.plugin_helpers
                            .entry(def.source_hash)
                            .or_insert_with(|| def.helpers.clone());
                    }
                    let e = def.instantiate(&inputs, slot, pos);
                    let v = fresh(tmp);
                    writeln!(body, "    let {v} = ({e});").unwrap();
                    v
                }
            }
        }
        NodeKind::Spline(sp) => {
            let slot = cx.layout.node_slot[&node];
            let m = fresh(tmp);
            writeln!(body, "    let {m} = spline_mask({pos}, {slot}u);").unwrap();
            // The matte this shape combines with. Unwired (or dangling) is
            // empty coverage, which under the default Union leaves the node
            // exactly the bare shape it is without an input.
            let inm = match cx
                .graph
                .input_of(node, 0)
                .filter(|i| cx.graph.nodes.contains_key(i))
            {
                Some(i) => {
                    let iv = emit(cx, i, tkey, pos, body, tmp, on_path);
                    let a = fresh(tmp);
                    writeln!(body, "    let {a} = ({iv}).a;").unwrap();
                    a
                }
                None => "0.0".to_string(),
            };
            // One expression per op, chosen at compile time like a Blend's:
            // a uniform-driven switch would cost every pixel of every shape
            // the branch, for a control nobody scrubs.
            let c = fresh(tmp);
            let expr = sp.combine.expr(&inm, &m);
            writeln!(body, "    let {c} = {expr};").unwrap();
            // Matte in ALPHA as well as RGB, the same convention LumaKey
            // uses: rgb so the shape reads on its own, alpha so the viewer's
            // alpha toggle — and anything downstream that composites — sees
            // a real matte instead of a solid 1.
            let v = fresh(tmp);
            writeln!(body, "    let {v} = vec4f(vec3f({c}), {c});").unwrap();
            v
        }
        NodeKind::Paint(p) => {
            use crate::graph::PaintOutput;
            // The stroke layers are rasterized outside the fused pass (only
            // when strokes change); here they are textures. A colour stroke
            // is just composited over the input — the node stays pointwise
            // — and that is the whole cost of a node that only paints.
            //
            // The other classes READ THE INPUT AGAIN, which is the whole
            // upstream chain again, per pixel: once for a clone/smear (at
            // the offset the source layer carries), PAINT_BLUR_TAPS times
            // for a blur. Each class is a topology bit, so a shader only
            // carries the reads its strokes need, and the panel says what
            // the node is paying.
            let idx = cx.bind(SourceBinding::Paint(node));
            let matte = p.output == PaintOutput::Matte;
            let below = match cx.graph.input_of(node, 0) {
                Some(input) => emit(cx, input, tkey, pos, body, tmp, on_path),
                None => "vec4f(0.0)".into(),
            };
            let paint = fresh(tmp);
            writeln!(body, "    let {paint} = sample_px(src{idx}, {pos});").unwrap();
            // The colour layer is the affine map `below * M + B` its blends
            // composed (`graph::PaintBlend`). Normal and erase alone leave
            // `M = 1 - B.a`, which is `over` — one texture, the code every
            // painted node always ran.
            let mul_idx = p.has_blend().then(|| cx.bind(SourceBinding::PaintMul(node)));
            let apply = |b: &str, m: Option<&str>, below: &str| match m {
                Some(m) => format!("({below} * {m} + {b})"),
                None => format!("over({b}, {below})"),
            };
            let mul = mul_idx.map(|mi| {
                let m = fresh(tmp);
                writeln!(body, "    let {m} = sample_px(src{mi}, {pos});").unwrap();
                m
            });
            // Coverage terms for the matte output, one per bound layer.
            let mut covs = vec![format!("{paint}.a")];
            let mut base = below.clone();
            // Bound up front so the clone source can read it too; `bind`
            // hands back the same index when the effect block asks again.
            let effect_idx = p.has_effect().then(|| cx.bind(SourceBinding::PaintEffect(node)));
            if p.has_source() {
                let ci = cx.bind(SourceBinding::PaintSource(node));
                let cl = fresh(tmp);
                writeln!(body, "    let {cl} = sample_px(src{ci}, {pos});").unwrap();
                covs.push(format!("{cl}.a"));
                if !matte {
                    // The offset arrives premultiplied by coverage (the
                    // raster pass blends stamps), so divide it back out;
                    // overlapping strokes therefore average their offsets
                    // instead of fighting over the pixel.
                    let cpos = fresh(tmp);
                    writeln!(
                        body,
                        "    let {cpos} = {pos} + select(vec2f(0.0), {cl}.xy / max({cl}.a, 1e-4), {cl}.a > 1e-4);"
                    )
                    .unwrap();
                    // The source is the PAINTED picture at the offset, not
                    // the bare input: a clone that could not pick up a
                    // colour stroke already laid down is not much of a
                    // clone. That means the input there, dodged/burned as
                    // it is there, with the colour layer over it. NOT the
                    // blur (its taps would multiply) and not other clones
                    // (a clone of a clone is a loop) — both are documented.
                    let inpx = match cx.graph.input_of(node, 0) {
                        Some(input) => emit(cx, input, tkey, &cpos, body, tmp, on_path),
                        None => "vec4f(0.0)".into(),
                    };
                    let toned = match effect_idx {
                        Some(ei) if p.has_tone() => {
                            let e2 = fresh(tmp);
                            writeln!(body, "    let {e2} = sample_px(src{ei}, {cpos});").unwrap();
                            let t = fresh(tmp);
                            writeln!(
                                body,
                                "    let {t} = vec4f({inpx}.rgb * exp2({e2}.r), {inpx}.a);"
                            )
                            .unwrap();
                            t
                        }
                        _ => inpx,
                    };
                    let mul_at = mul_idx.map(|mi| format!("sample_px(src{mi}, {cpos})"));
                    let srcpx = fresh(tmp);
                    writeln!(
                        body,
                        "    let {srcpx} = {};",
                        apply(&format!("sample_px(src{idx}, {cpos})"), mul_at.as_deref(), &toned)
                    )
                    .unwrap();
                    let b = fresh(tmp);
                    writeln!(
                        body,
                        "    let {b} = mix({base}, {srcpx}, clamp({cl}.a, 0.0, 1.0));"
                    )
                    .unwrap();
                    base = b;
                }
            }
            if let Some(ei) = effect_idx {
                let ef = fresh(tmp);
                writeln!(body, "    let {ef} = sample_px(src{ei}, {pos});").unwrap();
                covs.push(format!("clamp({ef}.a, 0.0, 1.0)"));
                if !matte && p.has_blur() {
                    // A ring of taps around the pixel already there, spread
                    // to the mean blur radius under it. The tap COUNT is
                    // fixed (the layer is additive: `g / b` is the weighted
                    // mean radius), so a wide blur brush is sparse rather
                    // than slow.
                    let brad = fresh(tmp);
                    writeln!(body, "    let {brad} = {ef}.g / max({ef}.b, 1e-4);").unwrap();
                    let mut acc = below.clone();
                    let n = crate::paint::PAINT_BLUR_TAPS;
                    for k in 0..n {
                        let a = k as f32 * std::f32::consts::TAU / n as f32;
                        let tp = fresh(tmp);
                        writeln!(
                            body,
                            "    let {tp} = {pos} + vec2f({:.6}, {:.6}) * {brad};",
                            a.cos(),
                            a.sin()
                        )
                        .unwrap();
                        let tap = match cx.graph.input_of(node, 0) {
                            Some(input) => emit(cx, input, tkey, &tp, body, tmp, on_path),
                            None => "vec4f(0.0)".into(),
                        };
                        let sum = fresh(tmp);
                        writeln!(body, "    let {sum} = {acc} + {tap};").unwrap();
                        acc = sum;
                    }
                    let b = fresh(tmp);
                    writeln!(
                        body,
                        "    let {b} = mix({base}, {acc} / {:.1}, clamp({ef}.b, 0.0, 1.0));",
                        (n + 1) as f32
                    )
                    .unwrap();
                    base = b;
                }
                if !matte && p.has_tone() {
                    // Exposure, in stops, already premultiplied by coverage
                    // and summed over strokes: a dodge brush builds up.
                    let b = fresh(tmp);
                    writeln!(
                        body,
                        "    let {b} = vec4f({base}.rgb * exp2({ef}.r), {base}.a);"
                    )
                    .unwrap();
                    base = b;
                }
            }
            let v = fresh(tmp);
            if matte {
                // Coverage as a matte, unioned with whatever alpha came in:
                // touching up a keyer's matte by hand is the point. In rgb
                // AND alpha, the convention every matte source here uses.
                let m = fresh(tmp);
                let mut expr = format!("{below}.a");
                for c in &covs {
                    expr = format!("max({expr}, {c})");
                }
                writeln!(body, "    let {m} = clamp({expr}, 0.0, 1.0);").unwrap();
                writeln!(body, "    let {v} = vec4f(vec3f({m}), {m});").unwrap();
            } else {
                writeln!(body, "    let {v} = {};", apply(&paint, mul.as_deref(), &base)).unwrap();
            }
            v
        }
        NodeKind::Shader(p) => {
            // Broken shader = passthrough of input 0. The ports are still
            // there, the wires are still there, and the comp still renders
            // while the text is being typed.
            if !p.def().ok() {
                let result = match cx.graph.input_of(node, 0) {
                    Some(input) => emit(cx, input, tkey, pos, body, tmp, on_path),
                    None => "vec4f(0.0)".into(),
                };
                on_path.remove(&node);
                return result;
            }
            let idx = cx.bind(SourceBinding::Blur((node, tkey)));
            let v = fresh(tmp);
            writeln!(body, "    let {v} = sample_px(src{idx}, {pos});").unwrap();
            v
        }
        NodeKind::Blur(b) => b.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::Glow(g) => g.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::Godrays(g) => g.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::LensFlare(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::EdgeExtend(g) => g.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::Glint(g) => g.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::Convolve(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::Median(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::HighlightRebuild(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::EdgeDetect(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::Diffuse(d) => d.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::Defocus(_) => {
            // A filter's output texture is a pass boundary: just sample it.
            let idx = cx.bind(SourceBinding::Blur((node, tkey)));
            let v = fresh(tmp);
            writeln!(body, "    let {v} = sample_px(src{idx}, {pos});").unwrap();
            // Masked filter: crossfade against the input subtree, RE-FUSED
            // inline here (costs ALU only when a mask is wired — no extra
            // texture binding, no compositor changes). A dangling mask edge
            // (source deleted) counts as unmasked.
            if let Some(mask) =
                cx.graph.input_of(node, 1).filter(|m| cx.graph.nodes.contains_key(m))
            {
                let orig = match cx.graph.input_of(node, 0) {
                    Some(input) => emit(cx, input, tkey, pos, body, tmp, on_path),
                    None => "vec4f(0.0)".into(),
                };
                let mv = emit(cx, mask, tkey, pos, body, tmp, on_path);
                let v2 = fresh(tmp);
                writeln!(body, "    let {v2} = mix({orig}, {v}, {mv}.x);").unwrap();
                return v2;
            }
            v
        }
        NodeKind::GraphBake { source, baked, window, mask, single, .. } => {
            if *baked {
                // The cached subtree, played through this node's reader slot.
                let idx =
                    cx.bind(SourceBinding::Reader(cx.graph.slot_for(node, tkey, *source)));
                let v = fresh(tmp);
                let at = if window.is_some() {
                    // Only a crop's box was stored: put it back where the
                    // box sits (sample_px is black outside the texture).
                    let slot = cx.layout.node_slot[&node];
                    format!("{pos} - P.v[{slot}].xy")
                } else {
                    pos.to_string()
                };
                if *mask || *single {
                    // A matte (or a one-channel picture) plays back as one
                    // stored channel (an R texture): the value is the whole
                    // pixel, broadcast like a single-channel Load's.
                    writeln!(body, "    let {v} = vec4f(sample_px(src{idx}, {at}).r);").unwrap();
                } else {
                    writeln!(body, "    let {v} = sample_px(src{idx}, {at});").unwrap();
                }
                v
            } else {
                // Live: transparent passthrough of the subtree.
                match cx.graph.input_of(node, 0) {
                    Some(input) => emit(cx, input, tkey, pos, body, tmp, on_path),
                    None => "vec4f(0.0)".into(),
                }
            }
        }
        NodeKind::GenAi(p) => {
            if p.loaded {
                // The picked take, stretched over this node's format, into
                // the working space through the sRGB input transform.
                let idx = cx.bind(SourceBinding::Reader(cx.graph.slot_for(node, tkey, p.source)));
                let slot = cx.layout.node_slot[&node];
                let idt = idt_fn(cx, node);
                GenAiParams::emit_loaded(
                    &format!("src{idx}"),
                    pos,
                    slot,
                    idt.as_deref(),
                    body,
                    &mut || fresh(tmp),
                )
            } else {
                // Nothing generated yet: the node is transparent.
                match cx.graph.input_of(node, 0) {
                    Some(input) => emit(cx, input, tkey, pos, body, tmp, on_path),
                    None => "vec4f(0.0)".into(),
                }
            }
        }
        // A loaded result plays from its reader; nothing loaded, the node
        // is transparent.
        NodeKind::Upscale(crate::graph::UpscaleParams { loaded: true, source, .. })
        | NodeKind::SeedVr(crate::graph::SeedVrParams { loaded: true, source, .. })
        | NodeKind::HdrCnn(crate::graph::HdrCnnParams { loaded: true, source, .. }) => {
            let idx = cx.bind(SourceBinding::Reader(cx.graph.slot_for(node, tkey, *source)));
            let idt = idt_fn(cx, node);
            crate::nodes::upscale::UpscaleParams::emit_loaded(&format!("src{idx}"), pos, idt.as_deref(), body, &mut || fresh(tmp))
        }
        NodeKind::Upscale(_) | NodeKind::SeedVr(_) | NodeKind::HdrCnn(_) => match cx.graph.input_of(node, 0) {
            Some(input) => emit(cx, input, tkey, pos, body, tmp, on_path),
            None => "vec4f(0.0)".into(),
        },
        // Loaded: the result stretched over the plate's format, composited
        // where the plate clipped (or alone, raw). Unloaded: transparent.
        NodeKind::LtxHdr(p) => {
            if p.loaded {
                let idx = cx.bind(SourceBinding::Reader(cx.graph.slot_for(node, tkey, p.source)));
                let slot = cx.layout.node_slot[&node];
                let idt = idt_fn(cx, node);
                let input = match (p.raw, cx.graph.input_of(node, 0)) {
                    (false, Some(input)) => Some(emit(cx, input, tkey, pos, body, tmp, on_path)),
                    (false, None) => Some("vec4f(0.0)".to_string()),
                    (true, _) => None,
                };
                crate::graph::LtxHdrParams::emit_loaded(
                    &format!("src{idx}"),
                    pos,
                    slot,
                    idt.as_deref(),
                    input.as_deref(),
                    body,
                    &mut || fresh(tmp),
                )
            } else {
                match cx.graph.input_of(node, 0) {
                    Some(input) => emit(cx, input, tkey, pos, body, tmp, on_path),
                    None => "vec4f(0.0)".into(),
                }
            }
        }
        // A loaded latlong plays from its reader at its own size; nothing
        // run yet, black (the plate is not a latlong).
        NodeKind::DiffLight(p) => {
            if p.loaded {
                let idx = cx.bind(SourceBinding::Reader(cx.graph.slot_for(node, tkey, p.source)));
                let idt = idt_fn(cx, node);
                crate::nodes::upscale::UpscaleParams::emit_loaded(&format!("src{idx}"), pos, idt.as_deref(), body, &mut || fresh(tmp))
            } else {
                "vec4f(0.0, 0.0, 0.0, 1.0)".into()
            }
        }
        // A loaded result is numbers off disk; nothing run yet, no depth.
        NodeKind::Depth(p) => {
            if p.loaded {
                let idx = cx.bind(SourceBinding::Reader(cx.graph.slot_for(node, tkey, p.source)));
                let slot = cx.layout.node_slot[&node];
                crate::graph::DepthParams::emit_loaded(&format!("src{idx}"), pos, slot, body, &mut || fresh(tmp))
            } else {
                "vec4f(0.0)".into()
            }
        }
        NodeKind::Rifetime(p) => {
            if p.loaded {
                let idx = cx.bind(SourceBinding::Reader(cx.graph.slot_for(node, tkey, p.source)));
                let idt = idt_fn(cx, node);
                let v = fresh(tmp);
                body.push_str(&format!("    let {v} = sample_px(src{idx}, {pos});\n"));
                match idt {
                    Some(f) => {
                        let c = fresh(tmp);
                        body.push_str(&format!("    let {c} = vec4f({f}({v}.rgb), {v}.a);\n"));
                        c
                    }
                    None => v,
                }
            } else {
                // Not rendered yet: the input, untouched.
                match cx.graph.input_of(node, 0) {
                    Some(input) => emit(cx, input, tkey, pos, body, tmp, on_path),
                    None => "vec4f(0.0)".into(),
                }
            }
        }
        NodeKind::Flow(p) => {
            if p.loaded {
                let idx = cx.bind(SourceBinding::Reader(cx.graph.slot_for(node, tkey, p.source)));
                let slot = cx.layout.node_slot[&node];
                crate::nodes::flow::FlowParams::emit_loaded(&format!("src{idx}"), pos, slot, body, &mut || fresh(tmp))
            } else {
                // Nothing analysed yet: no motion anywhere.
                "vec4f(0.0)".into()
            }
        }
        NodeKind::VectorBlur(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::VectorWarp(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::CorridorKey(p) => {
            if p.loaded {
                let idx = cx.bind(SourceBinding::Reader(cx.graph.slot_for(node, tkey, p.source)));
                let slot = cx.layout.node_slot[&node];
                let idt = idt_fn(cx, node);
                CorridorKeyParams::emit_loaded(
                    &format!("src{idx}"),
                    pos,
                    slot,
                    idt.as_deref(),
                    body,
                    &mut || fresh(tmp),
                )
            } else {
                // No key yet: the node is transparent.
                match cx.graph.input_of(node, 0) {
                    Some(input) => emit(cx, input, tkey, pos, body, tmp, on_path),
                    None => "vec4f(0.0)".into(),
                }
            }
        }
        // A loaded result is numbers off disk; nothing run yet, no normal
        // (a Relight reads a zero normal as "no surface" and adds nothing).
        NodeKind::Normals(p) => {
            if p.loaded {
                let idx = cx.bind(SourceBinding::Reader(cx.graph.slot_for(node, tkey, p.source)));
                let slot = cx.layout.node_slot[&node];
                crate::graph::NormalsParams::emit_loaded(&format!("src{idx}"), pos, slot, body, &mut || fresh(tmp))
            } else {
                "vec4f(0.0)".into()
            }
        }
        NodeKind::Sammie(p) => {
            let plate = if p.samples_plate() {
                cx.graph.input_of(node, 0).map(|input| emit(cx, input, tkey, pos, body, tmp, on_path))
            } else {
                None
            };
            if p.loaded {
                let idx = cx.bind(SourceBinding::Reader(cx.graph.slot_for(node, tkey, p.source)));
                p.emit_loaded(&format!("src{idx}"), pos, plate.as_deref(), body, &mut || fresh(tmp))
            } else {
                // No matte yet: a cutout shows the plate (so its points can
                // be placed on it), a matte is empty.
                match (p.output, plate) {
                    (crate::nodes::sammie::SammieOut::Cutout, Some(plate)) => plate,
                    _ => "vec4f(0.0)".into(),
                }
            }
        }
        NodeKind::Dilate(d) => d.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::ChannelSelect(_) => {
            // Never reaches a compiled graph (`to_engine_graph_at` lowers it
            // to a Read); an UNLOWERED one — unwired, or its slot not yet
            // minted — reads as conspicuous transparent black.
            "vec4f(0.0)".into()
        }
        NodeKind::Breakout(_) => {
            // Same contract: the editor re-points every consumer at the
            // hidden ChannelSelect kids, so a Breakout that reaches codegen
            // is unlowered and reads as conspicuous black.
            "vec4f(0.0)".into()
        }
        NodeKind::Recombine(p) => {
            // The Add ladder as one fused expression: unwired strings
            // contribute nothing, so a half-plugged harp still sums. A
            // signed input SUBTRACTS — the remainder pass is built from
            // exactly this (base rgba plus, every AOV minus).
            let slot = cx.layout.node_slot[&node];
            let parts: Vec<(bool, String)> = (0..p.strings())
                .filter_map(|i| cx.graph.input_of(node, i).map(|inp| (i, inp)))
                .map(|(i, input)| {
                    let e = emit(cx, input, tkey, pos, body, tmp, on_path);
                    // Per-string gain (the horseshoe arc), a uniform lane.
                    let lane = ["x", "y", "z", "w"][i % 4];
                    let g = fresh(tmp);
                    writeln!(body, "    let {g} = {e} * P.v[{}].{lane};", slot + i / 4)
                        .unwrap();
                    (p.sub.get(i).copied().unwrap_or(false), g)
                })
                .collect();
            let v = fresh(tmp);
            if parts.is_empty() {
                writeln!(body, "    let {v} = vec4f(0.0);").unwrap();
            } else {
                let mut expr = String::new();
                for (i, (sub, e)) in parts.iter().enumerate() {
                    if i == 0 && !*sub {
                        expr.push_str(e);
                    } else if *sub {
                        expr.push_str(" - ");
                        expr.push_str(e);
                    } else {
                        expr.push_str(" + ");
                        expr.push_str(e);
                    }
                }
                // A leading minus needs something to hang off.
                if parts[0].0 {
                    expr = format!("vec4f(0.0){expr}");
                }
                writeln!(body, "    let {v} = {expr};").unwrap();
            }
            // Summing AOVs rebuilds COLOUR. Their alphas are the
            // renderer's coverage per pass and adding those is meaningless,
            // so by default the fourth channel comes off the ORIGINAL rgba
            // on the porch input — the beauty these strings were broken out
            // of. Unwired, there is nothing better than the sum.
            let base = cx.graph.input_of(node, p.base_input());
            match (p.alpha, base) {
                (crate::graph::AlphaFrom::Base, Some(src)) => {
                    let b = emit(cx, src, tkey, pos, body, tmp, on_path);
                    let a = fresh(tmp);
                    writeln!(body, "    let {a} = vec4f({v}.rgb, {b}.a);").unwrap();
                    a
                }
                (crate::graph::AlphaFrom::Opaque, _) => {
                    let a = fresh(tmp);
                    writeln!(body, "    let {a} = vec4f({v}.rgb, 1.0);").unwrap();
                    a
                }
                _ => v,
            }
        }
        NodeKind::LoopIn => {
            // The previous-iteration tap. Inside its loop's body shader the
            // compositor rebinds this per iteration; anywhere else it
            // resolves to the loop's finished result (or black when no
            // LoopOut pairs it — a dangling tap is conspicuous, not magic).
            let idx = cx.bind(SourceBinding::LoopPrev(node));
            let v = fresh(tmp);
            writeln!(body, "    let {v} = sample_px(src{idx}, {pos});").unwrap();
            v
        }
        NodeKind::LoopOut(p) => {
            if loop_ok(cx.graph, node, p.pair) {
                // A pass boundary: the iterations already ran.
                let idx = cx.bind(SourceBinding::Blur((node, tkey)));
                let v = fresh(tmp);
                writeln!(body, "    let {v} = sample_px(src{idx}, {pos});").unwrap();
                v
            } else {
                // No compilable loop (unpaired, or a stage inside the tap's
                // cone): evaluate the body ONCE inline. The panel says why.
                match cx.graph.input_of(node, 0) {
                    Some(input) => emit(cx, input, tkey, pos, body, tmp, on_path),
                    None => "vec4f(0.0)".into(),
                }
            }
        }
        NodeKind::Grid(_) => {
            let slot = cx.layout.node_slot[&node];
            let v = fresh(tmp);
            writeln!(body, "    let {v} = grid_px({pos}, {slot}u);").unwrap();
            v
        }
        NodeKind::Noise(n) => {
            let slot = cx.layout.node_slot[&node];
            // The sub-graph compiles to one WGSL function per field node,
            // callee first. `color_fns` is emitted in KEY order, so the keys
            // carry a zero-padded sequence: sorting by name would put
            // `nz_10` before `nz_2` and hand naga a call to something not
            // yet declared.
            let mut fns = Vec::new();
            // The animation phase lives in the node's own header; each
            // generator inside scales it by its own multiplier.
            let phase = format!("P.v[{slot}].z");
            let entry =
                crate::field::compile(node, &n.graph, slot + 1, &phase, &mut fns);
            // The mask the field reads, sampled HERE, at the pixel — the
            // compiled functions take it as an argument rather than
            // sampling it themselves, so a Warp inside the graph moves the
            // noise through the mask rather than dragging the mask with it.
            // Unwired or dangling reads 1, like every other matte input.
            let mask = match cx
                .graph
                .input_of(node, 0)
                .filter(|m| cx.graph.nodes.contains_key(m))
            {
                Some(m) => {
                    let mv = emit(cx, m, tkey, pos, body, tmp, on_path);
                    format!("{mv}.x")
                }
                None => "1.0".into(),
            };
            for (name, def) in fns {
                // Fusion inlines a node once per consumer (a Noise masking
                // two Mults in a chain is emitted twice), but its functions
                // are named by node and must be declared ONCE: a second copy
                // is a redefinition, and the whole shader fails to compile.
                if cx.color_fns.keys().any(|k| k.split_once('_').map(|(_, n)| n) == Some(&*name)) {
                    continue;
                }
                let key = format!("{:04}_{name}", cx.color_fns.len());
                cx.color_fns.insert(key, def);
            }
            let v = fresh(tmp);
            match entry {
                Some(f) => {
                    let q = fresh(tmp);
                    // Field space is normalized on the WIDTH, so a graph
                    // keeps its look when the format changes and y simply
                    // runs to the aspect ratio.
                    writeln!(
                        body,
                        "    let {q} = {pos} / max(P.v[0].x, 1.0) * max(P.v[{slot}].x, 1e-4);"
                    )
                    .unwrap();
                    let n0 = fresh(tmp);
                    writeln!(body, "    let {n0} = {f}({q}, P.v[{slot}].y, {mask});").unwrap();
                    // `var` only when the 8-bit step reassigns it, so a
                    // plain Noise emits exactly the WGSL it always did.
                    let kw = if n.eight_bit { "var" } else { "let" };
                    if n.single {
                        // One channel, in all four lanes like a Load's
                        // single channel; the second field is not evaluated.
                        writeln!(body, "    {kw} {v} = vec4f({n0});").unwrap();
                    } else if n.two_d {
                        // The second dimension: the same graph at the
                        // offset seed, into green.
                        let n1 = fresh(tmp);
                        writeln!(
                            body,
                            "    let {n1} = {f}({q}, P.v[{slot}].y + P.v[{slot}].w, {mask});"
                        )
                        .unwrap();
                        writeln!(body, "    {kw} {v} = vec4f({n0}, {n1}, 0.0, 1.0);").unwrap();
                    } else {
                        writeln!(body, "    {kw} {v} = vec4f(vec3f({n0}), 1.0);").unwrap();
                    }
                    if n.eight_bit {
                        // 8 bits: what is drawn is what an 8-bit Bake keeps.
                        writeln!(body, "    {v} = round(clamp({v}, vec4f(0.0), vec4f(1.0)) * 255.0) / 255.0;").unwrap();
                    }
                }
                // No output marked, or a cycle: black, and conspicuously so.
                None => writeln!(body, "    let {v} = vec4f(0.0, 0.0, 0.0, 1.0);").unwrap(),
            }
            v
        }
        // A field node cannot be reached from a comp graph — it only ever
        // lives inside a Noise node's own graph.
        NodeKind::Field(_) => "vec4f(0.0)".into(),
        NodeKind::Sdf(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::SdfNode(_) => "vec4f(0.0)".into(),
        NodeKind::Geo(_) => {
            // The drawing lives in a layer rasterised outside the fused pass
            // (only when the geometry changes); here it is a texture read.
            // The field port is the CPU evaluator's business, not this
            // pass's: nothing upstream of a Geo node is sampled here.
            let idx = cx.bind(SourceBinding::Geo(node));
            let v = fresh(tmp);
            writeln!(body, "    let {v} = sample_px(src{idx}, {pos});").unwrap();
            v
        }
        NodeKind::GeoNode(_) => "vec4f(0.0)".into(),
        NodeKind::Blender(bp) => {
            // The live picture, in a layer the app fills from the session;
            // its inputs went TO Blender and are not sampled here. The
            // pixels arrive in Blender's scene-linear primaries and enter
            // the working space the way a Read's do. A lowered pass
            // samples its HOST's layer for that pass.
            let (host, pass) = bp.layer_key(node);
            let idx = cx.bind(SourceBinding::Blender(host, pass));
            let slot = cx.layout.node_slot[&node];
            let v = fresh(tmp);
            // The layer may be smaller than the canvas (a half-res draw
            // while the eye moves): the slot scales the position onto it.
            // Its rows are bottom-up, as the GPU read them back in Blender
            // and as they sit in the ring: the flip is here, for free.
            let q = fresh(tmp);
            writeln!(body, "    let {q} = {pos} * P.v[{slot}].xy;").unwrap();
            writeln!(
                body,
                "    let {v} = sample_px(src{idx}, vec2f({q}.x, f32(textureDimensions(src{idx}).y) - {q}.y));"
            )
            .unwrap();
            match idt_fn(cx, node) {
                Some(f) => {
                    let v2 = fresh(tmp);
                    writeln!(body, "    let {v2} = vec4f({f}({v}.rgb), {v}.a);").unwrap();
                    v2
                }
                None => v,
            }
        }
        // A time node emits no code at all: it puts everything above it on
        // its own timebase, and the readers up there are bound to slots
        // holding that timebase's frame. The pixels flow through unchanged.
        // Flow-mode frames on disk: the reader, like a Rifetime's.
        NodeKind::FrameFix(p) if p.loaded => {
            let idx = cx.bind(SourceBinding::Reader(cx.graph.slot_for(node, tkey, p.source)));
            let v = fresh(tmp);
            body.push_str(&format!("    let {v} = sample_px(src{idx}, {pos});\n"));
            match idt_fn(cx, node) {
                Some(f) => {
                    let c = fresh(tmp);
                    body.push_str(&format!("    let {c} = vec4f({f}({v}.rgb), {v}.a);\n"));
                    c
                }
                None => v,
            }
        }
        NodeKind::Slip(_)
        | NodeKind::TimeRemap(_)
        | NodeKind::Speed(_)
        | NodeKind::RangeRemap(_)
        | NodeKind::Freeze(_)
        | NodeKind::FrameFix(_) => {
            let Some(input) = cx.graph.input_of(node, 0) else { return "vec4f(0.0)".into() };
            if kind.time_interp() == crate::graph::TimeInterp::Blend {
                // Both sides of the sub-frame time are live readers; dissolve
                // between them by the weight the app packed for this tick.
                let a = emit(
                    cx,
                    input,
                    crate::graph::time_key_push_phase(tkey, node, 0),
                    pos,
                    body,
                    tmp,
                    on_path,
                );
                let b = emit(
                    cx,
                    input,
                    crate::graph::time_key_push_phase(tkey, node, 1),
                    pos,
                    body,
                    tmp,
                    on_path,
                );
                let slot = cx.layout.node_slot[&node];
                let v = fresh(tmp);
                writeln!(body, "    let {v} = mix({a}, {b}, P.v[{slot}].x);").unwrap();
                v
            } else {
                emit(cx, input, crate::graph::time_key_push(tkey, node), pos, body, tmp, on_path)
            }
        }
        // Pure passthroughs: the tracker's output is animation data, and a
        // Write renders to disk on a button, never in the fused pass.
        NodeKind::Tracker(_) | NodeKind::PlanarTrack(_) | NodeKind::Write(_) => match cx
            .graph
            .input_of(node, 0)
        {
            Some(input) => emit(cx, input, tkey, pos, body, tmp, on_path),
            None => "vec4f(0.0)".into(),
        },
        NodeKind::Premult { premult } => {
            let input = match cx.graph.input_of(node, 0) {
                Some(i) => emit(cx, i, tkey, pos, body, tmp, on_path),
                None => "vec4f(0.0)".into(),
            };
            let v = fresh(tmp);
            if *premult {
                writeln!(body, "    let {v} = vec4f({input}.rgb * {input}.a, {input}.a);")
                    .unwrap();
            } else {
                // Fully transparent pixels have no colour to recover: leave
                // them at zero instead of dividing toward infinity.
                writeln!(
                    body,
                    "    let {v} = select({input}, vec4f({input}.rgb / {input}.a, {input}.a), {input}.a > 1e-6);"
                )
                .unwrap();
            }
            v
        }
        NodeKind::Shuffle(_) => {
            let input = match cx.graph.input_of(node, 0) {
                Some(i) => emit(cx, i, tkey, pos, body, tmp, on_path),
                None => "vec4f(0.0)".into(),
            };
            let slot = cx.layout.node_slot[&node];
            let v = fresh(tmp);
            writeln!(body, "    let {v} = shuffle4({input}, {slot}u);").unwrap();
            v
        }
        NodeKind::MatteIn => {
            let img = match cx.graph.input_of(node, 0) {
                Some(i) => emit(cx, i, tkey, pos, body, tmp, on_path),
                None => "vec4f(0.0)".into(),
            };
            // An unwired matte is coverage 1, the same as everywhere else,
            // so a MatteIn with nothing on input 2 makes the stream opaque.
            let cov = match cx
                .graph
                .input_of(node, 1)
                .filter(|m| cx.graph.nodes.contains_key(m))
            {
                Some(m) => {
                    let mv = emit(cx, m, tkey, pos, body, tmp, on_path);
                    format!("{mv}.x")
                }
                None => "1.0".into(),
            };
            let v = fresh(tmp);
            writeln!(body, "    let {v} = vec4f({img}.rgb, {cov});").unwrap();
            v
        }
        NodeKind::ToMatte(_) => {
            let input = match cx.graph.input_of(node, 0) {
                Some(i) => emit(cx, i, tkey, pos, body, tmp, on_path),
                None => "vec4f(0.0)".into(),
            };
            let slot = cx.layout.node_slot[&node];
            let m = fresh(tmp);
            // Channel pick is a dot with a one-hot (or luma) vector; the
            // second slot is (lo, hi, invert) for re-ranging a source that
            // is not already 0..1.
            writeln!(
                body,
                "    let {m} = clamp((dot({input}, P.v[{slot}]) - P.v[{}].x) / max(P.v[{}].y - P.v[{}].x, 1e-5), 0.0, 1.0);",
                slot + 1,
                slot + 1,
                slot + 1
            )
            .unwrap();
            let c = fresh(tmp);
            writeln!(
                body,
                "    let {c} = select({m}, 1.0 - {m}, P.v[{}].z != 0.0);",
                slot + 1
            )
            .unwrap();
            // Coverage rides in rgb like every other mask source (consumers
            // read .x) AND in alpha, so the viewer's alpha toggle shows the
            // matte rather than a solid 1.
            let v = fresh(tmp);
            writeln!(body, "    let {v} = vec4f(vec3f({c}), {c});").unwrap();
            v
        }
        NodeKind::Mult(_) => {
            let input = match cx.graph.input_of(node, 0) {
                Some(i) => emit(cx, i, tkey, pos, body, tmp, on_path),
                None => "vec4f(0.0)".into(),
            };
            let slot = cx.layout.node_slot[&node];
            let v = fresh(tmp);
            // P.v[slot] = (r, g, b, multiply-alpha). Colour always; alpha
            // only when asked, so the default makes things hot rather than
            // fading them. An rgb multiplier has no single scalar to fade
            // alpha by, so it fades by its WORKING-SPACE luma — same rule
            // as everything else luminance-shaped in here.
            let w = cx.graph.working.primaries.luma_weights();
            writeln!(
                body,
                "    let {v} = vec4f({input}.rgb * P.v[{slot}].rgb, select({input}.a, {input}.a * dot(P.v[{slot}].rgb, vec3f({:.7}, {:.7}, {:.7})), P.v[{slot}].w != 0.0));",
                w[0], w[1], w[2]
            )
            .unwrap();
            // Masked: crossfade against the untouched input, the same way a
            // masked Grade does. A dangling mask edge counts as unmasked.
            if let Some(mask) =
                cx.graph.input_of(node, 1).filter(|m| cx.graph.nodes.contains_key(m))
            {
                let mv = emit(cx, mask, tkey, pos, body, tmp, on_path);
                let v2 = fresh(tmp);
                writeln!(body, "    let {v2} = mix({input}, {v}, {mv}.x);").unwrap();
                return v2;
            }
            v
        }
        NodeKind::Saturation(sp) => sp.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::Contrast(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::Gradient(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::Blank(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::Starfield(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::Planet(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::Plasma(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::Splat(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::Particles(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::ParticleNode(_) | NodeKind::GenAiStep(_) | NodeKind::RotoNode(_) => "vec4f(0.0)".into(),
        NodeKind::GlintKernel(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::BokehKernel(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::Prompt(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::CamTrack(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::Camera3D(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::Card3D(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::Scene3D(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::Render3D(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::Grain(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::ColorCompress(p) => {
            p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path))
        }
        NodeKind::HueRotate(p) => {
            p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path))
        }
        NodeKind::Lut(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::Text(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::Burnin(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::Mirror(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::FlipFlop(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::Turn(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::Uncrop(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::Compare(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::MorphDissolve(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::Gamma(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::ToneMap(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::UvWarp(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::Env(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::Refract(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::GridWarp(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::LensDistort(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::Warp(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::Expression(p) => {
            p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path))
        }
        NodeKind::Despill(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::Lightwrap(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::Relight(p) => crate::nodes::NodeDef::emit(p, &mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::ScreenKey(p) => {
            p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path))
        }
        NodeKind::ColorSlice(p) => {
            p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path))
        }
        NodeKind::ToData(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::ToRgba(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::SoftCrop(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::MotionBlur(p) => p.emit(&mut Emit::new(cx, node, tkey, pos, body, tmp, on_path)),
        NodeKind::Reformat(_) => {
            // Output pixel -> source pixel: a scale and an offset, so the
            // node stays one sample (no resampling pass).
            let slot = cx.layout.node_slot[&node];
            let p = fresh(tmp);
            writeln!(
                body,
                "    let {p} = {pos} * P.v[{slot}].xy + P.v[{slot}].zw;"
            )
            .unwrap();
            match cx.graph.input_of(node, 0) {
                Some(input) => emit(cx, input, tkey, &p, body, tmp, on_path),
                None => "vec4f(0.0)".into(),
            }
        }
        NodeKind::Transform(_) => {
            // A homography, always: the corner pin rides inside the same
            // map, so pinning a corner is a uniform write, never a recompile.
            let slot = cx.layout.node_slot[&node];
            let p = fresh(tmp);
            let matte = cx.graph.input_of(node, 1).filter(|_| cx.layout.modulated.contains(&node));
            if let Some(mid) = matte {
                // MODULATED: the matte, evaluated at THIS output pixel,
                // says how much of the transform applies here. Still one
                // position expression — the upstream is sampled once, at
                // the blended position, with no UV image and no resample.
                let mv = emit(cx, mid, tkey, pos, body, tmp, on_path);
                writeln!(body, "    let {p} = xf_mod({slot}u, clamp({mv}.x, 0.0, 1.0), {pos});")
                    .unwrap();
            } else {
                writeln!(
                    body,
                    "    let {p} = persp(P.v[{slot}], P.v[{}], P.v[{}], {pos});",
                    slot + 1,
                    slot + 2
                )
                .unwrap();
            }
            match cx.graph.input_of(node, 0) {
                Some(input) => emit(cx, input, tkey, &p, body, tmp, on_path),
                None => "vec4f(0.0)".into(),
            }
        }
        // A corner pin with no modulate port: the pin is the correction at
        // this frame (`StabilizeParams::pin`, packed like a CornerPin's).
        NodeKind::Stabilize(_) => {
            let slot = cx.layout.node_slot[&node];
            let p = fresh(tmp);
            writeln!(body, "    let {p} = persp(P.v[{slot}], P.v[{}], P.v[{}], {pos});", slot + 1, slot + 2).unwrap();
            match cx.graph.input_of(node, 0) {
                Some(input) => emit(cx, input, tkey, &p, body, tmp, on_path),
                None => "vec4f(0.0)".into(),
            }
        }
        NodeKind::CornerPin(_) => {
            let slot = cx.layout.node_slot[&node];
            let p = fresh(tmp);
            // Modulated exactly like a Transform: to `xf_mod` a pin is a
            // Transform that neither moves, turns nor scales.
            let matte = cx.graph.input_of(node, 1).filter(|_| cx.layout.modulated.contains(&node));
            if let Some(mid) = matte {
                let mv = emit(cx, mid, tkey, pos, body, tmp, on_path);
                writeln!(body, "    let {p} = xf_mod({slot}u, clamp({mv}.x, 0.0, 1.0), {pos});")
                    .unwrap();
            } else {
                writeln!(
                    body,
                    "    let {p} = persp(P.v[{slot}], P.v[{}], P.v[{}], {pos});",
                    slot + 1,
                    slot + 2
                )
                .unwrap();
            }
            match cx.graph.input_of(node, 0) {
                Some(input) => emit(cx, input, tkey, &p, body, tmp, on_path),
                None => "vec4f(0.0)".into(),
            }
        }
        NodeKind::Grade(g) => {
            let slot = cx.layout.node_slot[&node];
            let input = match cx.graph.input_of(node, 0) {
                Some(input) => emit(cx, input, tkey, pos, body, tmp, on_path),
                None => "vec4f(0.0)".into(),
            };
            // The working space's own luma weights, so the saturation the
            // grade applies preserves the luminance it claims to.
            let lw = cx.graph.working.primaries.luma_weights();
            let lwv = format!("vec3f({:.7}, {:.7}, {:.7})", lw[0], lw[1], lw[2]);
            let v = fresh(tmp);
            writeln!(
                body,
                "    let {v} = grade4({input}, P.v[{slot}], P.v[{}], P.v[{}], P.v[{}], P.v[{}], {lwv});",
                slot + 1,
                slot + 2,
                slot + 3,
                slot + 4,
            )
            .unwrap();
            // Zones: a LumaKey with N bands feeding N masked Grades in
            // series, fused. Each band's response — the same curve, clamp
            // and invert the LumaKey emits — gates that zone's own grade4
            // over the running result. The luma is the INPUT's, before the
            // master, as it would be with the key wired off the same plate.
            let mut out = v;
            let nz = g.zone_count();
            if nz > 0 {
                let l = fresh(tmp);
                writeln!(body, "    let {l} = dot({input}.rgb, {lwv});").unwrap();
                // The previewed zone's coverage, picked out as the zones are
                // walked: which one is a uniform, so the pick is a `select`
                // per zone rather than a second pass or a recompile.
                let pw = fresh(tmp);
                writeln!(body, "    var {pw} = 0.0;").unwrap();
                // The ZONES preview: every zone's weight times its colour,
                // summed; and the HARD one, the colour of whichever zone
                // weighs most (strictly, so a tie goes to the lower zone).
                // The colours are constants — they depend on the zone count
                // and the working space, both already structural — so the
                // maps cost a multiply-add and a compare per zone, no slot.
                let pc = fresh(tmp);
                writeln!(body, "    var {pc} = vec3f(0.0);").unwrap();
                let (hw, hc) = (fresh(tmp), fresh(tmp));
                writeln!(body, "    var {hw} = 0.0;").unwrap();
                writeln!(body, "    var {hc} = vec3f(0.0);").unwrap();
                for i in 0..nz {
                    let zb = slot + GRADE_SLOTS + i * ZONE_SLOTS;
                    let hdr = zb + GRADE_SLOTS;
                    let c = fresh(tmp);
                    writeln!(
                        body,
                        "    let {c} = clamp(curve_eval({}u, u32(P.v[{hdr}].x), {l}, P.v[{hdr}].y, 1.0), 0.0, 1.0);",
                        hdr + 1
                    )
                    .unwrap();
                    let w = fresh(tmp);
                    writeln!(body, "    let {w} = mix({c}, 1.0 - {c}, P.v[{hdr}].z);").unwrap();
                    writeln!(
                        body,
                        "    {pw} = select({pw}, {w}, P.v[{}].w == {i}.0);",
                        slot + 2
                    )
                    .unwrap();
                    let col = crate::graph::band_linear(i, nz, cx.graph.working);
                    let cv = fresh(tmp);
                    writeln!(
                        body,
                        "    let {cv} = vec3f({:.7}, {:.7}, {:.7});",
                        col[0], col[1], col[2]
                    )
                    .unwrap();
                    writeln!(body, "    {pc} = {pc} + {w} * {cv};").unwrap();
                    // Starting the best at 0 and taking only a STRICTLY
                    // heavier zone leaves a pixel no zone covers black.
                    writeln!(body, "    {hc} = select({hc}, {cv}, {w} > {hw});").unwrap();
                    writeln!(body, "    {hw} = max({hw}, {w});").unwrap();
                    let z = fresh(tmp);
                    writeln!(
                        body,
                        "    let {z} = grade4({out}, P.v[{zb}], P.v[{}], P.v[{}], P.v[{}], P.v[{}], {lwv});",
                        zb + 1,
                        zb + 2,
                        zb + 3,
                        zb + 4,
                    )
                    .unwrap();
                    let m = fresh(tmp);
                    writeln!(body, "    let {m} = mix({out}, {z}, {w});").unwrap();
                    out = m;
                }
                // Preview (gamma.w: 0 off, 1 matte, 2 highlight, 3 zones,
                // 4 hard). MATTE is the coverage in rgb AND alpha, the
                // convention every matte node here follows; HIGHLIGHT holds
                // the picture where the zone lands and sinks the rest toward
                // grey, so the same coverage can be read against the shot;
                // ZONES and HARD are the colour maps built above, alpha kept.
                // All are one select off
                // values already computed, and all vanish when the mode is
                // 0 — which is every render, every consumer, and every frame
                // this node is not the one being viewed.
                let py = fresh(tmp);
                writeln!(body, "    let {py} = dot({out}.rgb, {lwv});").unwrap();
                let pv = fresh(tmp);
                writeln!(
                    body,
                    "    let {pv} = select(vec4f(vec3f({pw}), {pw}), \
                     vec4f(mix(vec3f({py}) * 0.22, {out}.rgb, {pw}), {out}.a), \
                     P.v[{}].w > 1.5);",
                    slot + 1
                )
                .unwrap();
                let pm = fresh(tmp);
                writeln!(
                    body,
                    "    let {pm} = select({pv}, vec4f({pc}, {out}.a), P.v[{}].w > 2.5);",
                    slot + 1
                )
                .unwrap();
                let ph = fresh(tmp);
                writeln!(
                    body,
                    "    let {ph} = select({pm}, vec4f({hc}, {out}.a), P.v[{}].w > 3.5);",
                    slot + 1
                )
                .unwrap();
                let po = fresh(tmp);
                writeln!(
                    body,
                    "    let {po} = select({out}, {ph}, P.v[{}].w > 0.5);",
                    slot + 1
                )
                .unwrap();
                out = po;
            }
            // Masked grade: blend toward the untouched input. A dangling
            // mask edge (source deleted) counts as unmasked.
            if let Some(mask) =
                cx.graph.input_of(node, 1).filter(|m| cx.graph.nodes.contains_key(m))
            {
                let mv = emit(cx, mask, tkey, pos, body, tmp, on_path);
                let v2 = fresh(tmp);
                writeln!(body, "    let {v2} = mix({input}, {out}, {mv}.x);").unwrap();
                return v2;
            }
            out
        }

        NodeKind::Blend { op, .. } => {
            let a0 = match cx.graph.input_of(node, 0) {
                Some(input) => emit(cx, input, tkey, pos, body, tmp, on_path),
                None => "vec4f(0.0)".into(),
            };
            let b0 = match cx.graph.input_of(node, 1) {
                Some(input) => emit(cx, input, tkey, pos, body, tmp, on_path),
                None => "vec4f(0.0)".into(),
            };
            // Per-input gains (the graph's horseshoe sliders): applied to
            // the samples BEFORE the op, so `mix` still dissolves toward
            // the (gained) background.
            let slot_g = cx.layout.node_slot[&node];
            let a = fresh(tmp);
            writeln!(body, "    let {a} = {a0} * P.v[{slot_g}].y;").unwrap();
            let b = fresh(tmp);
            writeln!(body, "    let {b} = {b0} * P.v[{slot_g}].z;").unwrap();
            let raw = fresh(tmp);
            writeln!(body, "    let {raw} = {};", blend_expr(*op, &a, &b)).unwrap();
            // `mix` dissolves the result back toward B, and the matte on
            // input 2 limits it to where the matte covers. Both land on the
            // same B, so they multiply into ONE lerp — a mixed, matted
            // blend costs exactly what a plain one does.
            let slot = cx.layout.node_slot[&node];
            let v = fresh(tmp);
            match cx.graph.input_of(node, 2).filter(|m| cx.graph.nodes.contains_key(m)) {
                Some(mask) => {
                    let mv = emit(cx, mask, tkey, pos, body, tmp, on_path);
                    writeln!(
                        body,
                        "    let {v} = mix({b}, {raw}, {mv}.x * P.v[{slot}].x);"
                    )
                    .unwrap();
                }
                None => {
                    writeln!(body, "    let {v} = mix({b}, {raw}, P.v[{slot}].x);")
                        .unwrap();
                }
            }
            v
        }
        NodeKind::Mix(_) => {
            let a = match cx.graph.input_of(node, 0) {
                Some(input) => emit(cx, input, tkey, pos, body, tmp, on_path),
                None => "vec4f(0.0)".into(),
            };
            let b = match cx.graph.input_of(node, 1) {
                Some(input) => emit(cx, input, tkey, pos, body, tmp, on_path),
                None => "vec4f(0.0)".into(),
            };
            let slot = cx.layout.node_slot[&node];
            let v = fresh(tmp);
            writeln!(body, "    let {v} = mix({a}, {b}, P.v[{slot}].x);").unwrap();
            v
        }
        NodeKind::Layer(lp) => {
            let n = lp.layers.len().min(crate::graph::LAYER_MAX);
            // An empty stack owns no slots, so it must not look one up.
            let slot = if n == 0 { 0 } else { cx.layout.node_slot[&node] };
            // Bottom-up. The last layer is the BASE — its blend op is
            // ignored, because there is nothing under it to land on (a
            // bottom layer set to multiply would otherwise multiply itself
            // into black). Everything above composites onto the running
            // result in stack order.
            let mut acc = String::from("vec4f(0.0)");
            for i in (0..n).rev() {
                let s0 = slot + i;
                // No per-layer placement: a layer lands where its input
                // already is (an upstream Transform places it), so it is
                // sampled at the stack's own pixel.
                let q = pos;
                // Each layer is its own branch of TIME (`input_time_key`),
                // so the same plate on two layers at two offsets decodes two
                // frames, both live at once.
                let c = match cx.graph.input_of(node, i) {
                    Some(src) => emit(
                        cx,
                        src,
                        crate::graph::input_time_key(tkey, node, i),
                        q,
                        body,
                        tmp,
                        on_path,
                    ),
                    None => "vec4f(0.0)".into(),
                };
                // Opacity folded with the trim gate the app stamps per
                // tick, one slot a layer. Premultiplied, so it fades rgb and
                // alpha together.
                let o = fresh(tmp);
                writeln!(body, "    let {o} = {c} * P.v[{s0}].x;").unwrap();
                acc = if i + 1 == n {
                    o
                } else {
                    let v = fresh(tmp);
                    writeln!(
                        body,
                        "    let {v} = {};",
                        blend_expr(lp.layers[i].op, &o, &acc)
                    )
                    .unwrap();
                    v
                };
            }
            acc
        }
        NodeKind::Curve(_) => {
            let input = match cx.graph.input_of(node, 0) {
                Some(i) => emit(cx, i, tkey, pos, body, tmp, on_path),
                None => "vec4f(0.0)".into(),
            };
            let slot = cx.layout.node_slot[&node];
            let v = fresh(tmp);
            writeln!(body, "    let {v} = curve3({input}, {slot}u);").unwrap();
            v
        }
        NodeKind::MatteCombine(mc) => {
            // Both ports are mattes. An UNWIRED one reads as EMPTY coverage,
            // not full: the identity for a union is 0, and a combine node
            // with one input should be that input, not a solid frame.
            let operand = |cx: &mut EmitCx,
                               body: &mut String,
                               tmp: &mut usize,
                               on_path: &mut std::collections::HashSet<NodeId>,
                               slot: usize| {
                match cx
                    .graph
                    .input_of(node, slot)
                    .filter(|i| cx.graph.nodes.contains_key(i))
                {
                    Some(i) => {
                        let v = emit(cx, i, tkey, pos, body, tmp, on_path);
                        let c = fresh(tmp);
                        writeln!(body, "    let {c} = ({v}).x;").unwrap();
                        c
                    }
                    None => "0.0".to_string(),
                }
            };
            let a = operand(cx, body, tmp, on_path, 0);
            let b = operand(cx, body, tmp, on_path, 1);
            let r = fresh(tmp);
            writeln!(body, "    let {r} = {};", mc.op.expr(&a, &b)).unwrap();
            // `mix` dissolves back to A, so 0 is a bypass of the whole node.
            let slot = cx.layout.node_slot[&node];
            let m = fresh(tmp);
            writeln!(body, "    let {m} = mix({a}, {r}, P.v[{slot}].x);").unwrap();
            // Matte in rgb AND alpha, the convention every matte source
            // here uses, so the result is legible on its own and composites.
            let v = fresh(tmp);
            writeln!(body, "    let {v} = vec4f(vec3f({m}), {m});").unwrap();
            v
        }
        // An unlowered Extract reads as its output 0: the Mask it lowers to.
        NodeKind::Mask | NodeKind::Extract(_) => {
            let img = match cx.graph.input_of(node, 0) {
                Some(i) => emit(cx, i, tkey, pos, body, tmp, on_path),
                None => "vec4f(0.0)".into(),
            };
            // Unwired / dangling mask counts as coverage 1 (passthrough).
            let cov = match cx
                .graph
                .input_of(node, 1)
                .filter(|m| cx.graph.nodes.contains_key(m))
            {
                Some(m) => {
                    let mv = emit(cx, m, tkey, pos, body, tmp, on_path);
                    format!("{mv}.x")
                }
                None => "1.0".into(),
            };
            let v = fresh(tmp);
            writeln!(body, "    let {v} = {img} * {cov};").unwrap();
            v
        }
        NodeKind::LumaKey(_) => {
            let img = match cx.graph.input_of(node, 0) {
                Some(i) => emit(cx, i, tkey, pos, body, tmp, on_path),
                None => "vec4f(0.0)".into(),
            };
            let slot = cx.layout.node_slot[&node];
            let l = fresh(tmp);
            // Luminance weights are the Y row of the working space's RGB->XYZ
            // matrix, by definition. The Rec.709 triple everyone hardcodes is
            // only correct when the comp IS Rec.709; in ACEScg the green
            // weight is 0.674, not 0.715, and a key pulled with the wrong
            // ones drifts on saturated colour.
            let w = cx.graph.working.primaries.luma_weights();
            writeln!(
                body,
                "    let {l} = dot({img}.rgb, vec3f({:.7}, {:.7}, {:.7}));",
                w[0], w[1], w[2]
            )
            .unwrap();
            // The response is a control-point curve of luma -> coverage;
            // two points with `smooth` on reproduce the old smoothstep
            // exactly, so existing keys are untouched.
            let c = fresh(tmp);
            writeln!(
                body,
                "    let {c} = curve_eval({slot}u + 1u, u32(P.v[{slot}].x), {l}, P.v[{slot}].y, 1.0);"
            )
            .unwrap();
            let m = fresh(tmp);
            writeln!(
                body,
                "    let {m} = mix(clamp({c}, 0.0, 1.0), 1.0 - clamp({c}, 0.0, 1.0), P.v[{slot}].z);"
            )
            .unwrap();
            // Coverage in ALPHA as well as rgb — the convention Spline and
            // Keyer already follow, and what the viewer's alpha toggle reads.
            // A solid 1 here is why A over a LumaKey showed flat white.
            let v = fresh(tmp);
            writeln!(body, "    let {v} = vec4f(vec3f({m}), {m});").unwrap();
            v
        }
        // Not viewed: the finished matte off the node's ONE matte reader (a
        // decode-side bake, see `CryptomatteParams::matte_source`). The id
        // layers are not in the rig at all.
        NodeKind::Cryptomatte(c) if !c.previewing => {
            let v = fresh(tmp);
            match c.matte_source {
                usize::MAX => writeln!(body, "    let {v} = vec4f(0.0);").unwrap(),
                s => {
                    let idx = cx.bind(SourceBinding::Reader(cx.graph.slot_for_sub(
                        node,
                        tkey,
                        crate::cryptomatte::MATTE_SUB,
                        s,
                    )));
                    let m = fresh(tmp);
                    writeln!(body, "    let {m} = sample_px(src{idx}, {pos}).x;").unwrap();
                    writeln!(body, "    let {v} = vec4f(vec3f({m}), {m});").unwrap();
                }
            }
            v
        }
        NodeKind::Cryptomatte(c) => {
            // VIEWED. The node is its own reader set: sub 0 is the file's
            // rgba, 1.. the ranks. Its one input is the cable,
            // dropped at lowering; nothing here samples an input. Unminted
            // (unwired, or before the app assigned slots) renders black.
            let bind = |cx: &mut EmitCx, sub: usize| {
                c.source(sub)
                    .map(|s| cx.bind(SourceBinding::Reader(cx.graph.slot_for_sub(node, tkey, sub, s))))
            };
            let img = match bind(cx, 0) {
                Some(idx) => {
                    let v = fresh(tmp);
                    writeln!(body, "    let {v} = sample_px(src{idx}, {pos});").unwrap();
                    v
                }
                None => "vec4f(0.0)".into(),
            };
            let slot = cx.layout.node_slot[&node];
            let m = fresh(tmp);
            writeln!(body, "    var {m} = 0.0;").unwrap();
            // Rank 0's texel doubles as the ids view.
            let mut first: Option<String> = None;
            for r in 0..crate::cryptomatte::RANKS {
                let Some(idx) = bind(cx, 1 + r) else { continue };
                // A rank is loaded UNFILTERED at the texel under the pixel:
                // an id interpolated with its neighbour is a third id that
                // belongs to nothing.
                let rv = fresh(tmp);
                writeln!(body, "    let {rv} = crypto_load(src{idx}, {pos});").unwrap();
                writeln!(
                    body,
                    "    {m} += {rv}.a * crypto_hit(crypto_id({rv}.rgb), {slot}u);"
                )
                .unwrap();
                first.get_or_insert(rv);
            }
            let ids = first.unwrap_or_else(|| "vec4f(0.0)".into());
            let mm = fresh(tmp);
            writeln!(
                body,
                "    let {mm} = mix(clamp({m}, 0.0, 1.0), 1.0 - clamp({m}, 0.0, 1.0), P.v[{slot}].z);"
            )
            .unwrap();
            // The ids picture is a uniform the app sets only for the node
            // being VIEWED (see `previewing`); consumers always get the
            // matte. It is rank 0's chunks scaled into [0,1) — exact in f16,
            // so a viewer readback decodes straight back to the id — and a
            // PICKED object is lit: red and green pushed into [1,4) by
            // `crypto_lit`, blue quartered — bright yellow on screen (the usual
            // colour for a picked matte) and still decodable (see
            // `cryptomatte::decode_view`). Alpha is the matte either way,
            // so the viewer's alpha toggle shows it.
            let lit = fresh(tmp);
            writeln!(
                body,
                "    let {lit} = crypto_hit(crypto_id({ids}.rgb), {slot}u) > 0.5;"
            )
            .unwrap();
            let idv = fresh(tmp);
            writeln!(
                body,
                "    let {idv} = select({ids}.rgb / {scale}, vec3f(crypto_lit({ids}.r), crypto_lit({ids}.g), {ids}.b / {scale2}), {lit});",
                scale = crate::cryptomatte::CHUNK_SCALE,
                scale2 = 2.0 * crate::cryptomatte::CHUNK_SCALE
            )
            .unwrap();
            let v = fresh(tmp);
            writeln!(
                body,
                "    let {v} = select(vec4f(vec3f({mm}), {mm}), vec4f({idv}, {mm}), P.v[{slot}].x > 0.5);"
            )
            .unwrap();
            let _ = img;
            v
        }
        NodeKind::Keyer(_) => {
            let img = match cx.graph.input_of(node, 0) {
                Some(i) => emit(cx, i, tkey, pos, body, tmp, on_path),
                None => "vec4f(0.0)".into(),
            };
            let slot = cx.layout.node_slot[&node];
            let m = fresh(tmp);
            writeln!(body, "    let {m} = keyer_matte({img}.rgb, {slot}u);").unwrap();
            // Garbage matte (input 1) multiplies in; unwired or dangling
            // leaves the key alone.
            let m = match cx
                .graph
                .input_of(node, 1)
                .filter(|g| cx.graph.nodes.contains_key(g))
            {
                Some(g) => {
                    let gv = emit(cx, g, tkey, pos, body, tmp, on_path);
                    let m2 = fresh(tmp);
                    writeln!(body, "    let {m2} = {m} * {gv}.x;").unwrap();
                    m2
                }
                None => m,
            };
            // Preview is a LERP on a uniform, not a branch and never a
            // recompile: rgb crossfades from the matte to the source image.
            // The app only ever sets that uniform for the node being VIEWED,
            // so every downstream consumer still reads a pure matte.
            let v = fresh(tmp);
            writeln!(
                body,
                "    let {v} = vec4f(mix(vec3f({m}), {img}.rgb, P.v[{}].z), {m});",
                slot + 1
            )
            .unwrap();
            v
        }
    };
    // A pointwise op in a matte pipe computed on the coverage it was given
    // (r = g = b = a); hand back a matte again, the coverage in all four,
    // so the viewer's alpha and anything reading .a agree with the .x every
    // mask consumer reads. One splat, only when the node is in matte mode.
    let result = if kind.recolours_matte() {
        let m = fresh(tmp);
        writeln!(body, "    let {m} = vec4f({result}.x);").unwrap();
        m
    } else {
        result
    };
    on_path.remove(&node);
    result
}

fn fresh(tmp: &mut usize) -> String {
    *tmp += 1;
    format!("t{}", *tmp)
}

/// The fixed separable gaussian pass (H or V decided by the `dir` uniform;
/// sigma is a uniform too, so blur-size edits never recompile anything).

/// The vector-blur pass: N taps along the baked motion vector at each pixel
/// (gathered with the DESTINATION pixel's vector — the classic cheap vector
/// blur). Vectors come from a reader texture in flow-plate pixels; a
/// resolution mismatch with the canvas rescales both lookup and vector.
/// `valid = 0` (vector input not a source node) or `amount = 0` copies.
/// Motion blur, half one: add the frame just rendered into the accumulator.
///
/// A STORAGE BUFFER, not a read-write storage texture: read-write textures
/// need an adapter feature that rgba16float does not universally have,
/// whereas a storage buffer works everywhere and accumulates in f32, which
/// matters when a long shutter sums a dozen samples of an HDR frame.
pub const ACCUM_WGSL: &str = r#"
struct AP { weight: f32, clear: f32, w: f32, h: f32 }
@group(0) @binding(0) var<uniform> A: AP;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> acc: array<vec4f>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = vec2u(u32(A.w), u32(A.h));
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let i = gid.y * dims.x + gid.x;
    let c = textureLoad(src, vec2i(gid.xy), 0) * A.weight;
    // The first sample seeds the accumulator rather than needing a separate
    // clear pass over the whole frame.
    acc[i] = select(acc[i] + c, c, A.clear != 0.0);
}
"#;

/// Motion blur, half two: the accumulated frame becomes the output.
///
/// The weights are applied on the way IN (each sample carries its own), so
/// this is a straight copy — which is what lets a non-uniform shutter weight
/// its samples without this pass knowing anything about it.
pub const ACCUM_RESOLVE_WGSL: &str = r#"
struct AP { weight: f32, clear: f32, w: f32, h: f32 }
@group(0) @binding(0) var<uniform> A: AP;
@group(0) @binding(1) var<storage, read_write> acc: array<vec4f>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = vec2u(u32(A.w), u32(A.h));
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    textureStore(dst, gid.xy, acc[gid.y * dims.x + gid.x]);
}
"#;


/// Glow, pass type 1: 13-tap downsample (Jimenez/CoD) to half resolution,
/// with the highlight prefilter folded into the FIRST one. This is where
/// the cost saving lives: an octave of extra radius is a quarter of the
/// pixels at a fixed tap count, where a widening convolution would be the
/// same pixels at double the taps.

/// Glow, pass type 2: 3x3 tent upsample of the octave above, plus this
/// octave's own weighted contribution. Running it from the top down builds
/// S_k = w_k*mip_k + up(S_k+1), so the half-res result is the whole
/// weighted sum of octaves — the compounded-blur stack, one add per level.

/// Glow, pass type 3: combine the half-res glow back over the full-res
/// original. Alpha is the original's throughout — a glow adds light, it
/// does not change what is covered.

/// A Defocus's MODULATE port: a matte that scales the circle of confusion
/// per pixel, on top of whatever the depth pass says.
pub const DEFOCUS_MOD_INPUT: usize = 4;
/// A Defocus's bokeh KERNEL port: the aperture's shape, Accurate mode only.
pub const DEFOCUS_KERNEL_INPUT: usize = 3;

/// The defocus uniform, shared by every defocus pass. 80 bytes.
/// `has_depth`/`has_kernel` gate the optional bindings so an unwired port is
/// a uniform edit, never a recompile. `has_depth` means the CoC MAP exists —
/// a depth pass OR a modulate matte made it; `has_z` and `has_mod` say which.
/// Without a Z the map holds the uniform radius (scaled by the matte) at
/// the focus distance, so everything is one far layer.
pub const DEFOCUS_COMMON_WGSL: &str = r#"
struct DP {
    radius: f32, has_depth: f32, chan: f32, inv_depth: f32,
    focus: f32, f_stop: f32, focal: f32, filmback: f32,
    blades: f32, rotation: f32, catseye: f32, samples: f32,
    has_kernel: f32, unit: f32, view: f32, edge: f32,
    clampv: f32, has_z: f32, has_mod: f32, pad2: f32,
    // The kernel picture's centre and radius, in texels of `kern`: the
    // middle of its format and half its short side.
    kcx: f32, kcy: f32, kr: f32, pad3: f32,
}

// Linear view-space Z from the depth texel, whichever convention it is in.
// Disparity 0 is infinity, not zero distance — clamped, not special-cased,
// so the CoC saturates at its far value instead of collapsing to sharp.
fn depth_z(raw: vec4f) -> f32 {
    var d = raw.r;
    if (P.chan == 1.0) { d = raw.g; }
    else if (P.chan == 2.0) { d = raw.b; }
    else if (P.chan == 3.0) { d = raw.a; }
    else if (P.chan == 4.0) { d = dot(raw.rgb, vec3f(0.2126, 0.7152, 0.0722)); }
    return depth_lin(d);
}

// A raw pass value to linear Z. `focus` goes through this too: it is a
// number read off the pass, not a distance.
fn depth_lin(d: f32) -> f32 {
    if (P.inv_depth != 0.0) { return 1.0 / clamp(abs(d), 1e-6, 1e6); }
    return d;
}

// Thin-lens circle of confusion, in output pixels. Focus distance is scene
// units read as metres against a millimetre focal length (the comp
// convention); filmback converts sensor mm to pixels across `canvas_w`.
// Mirrored by DefocusParams::coc_px on the CPU — a test asserts they agree.
fn coc_px(z: f32, canvas_w: f32) -> f32 {
    let f = max(P.focal, 1e-3);
    let mm = 1000.0 * max(P.unit, 1e-9);
    let focus_mm = max(depth_lin(P.focus) * mm, f * 1.001);
    let z_mm = z * mm;
    if (z_mm <= 0.0 || z_mm != z_mm) { return 0.0; }
    let aperture = f / max(P.f_stop, 1e-3);
    let coc_mm = aperture * (f / max(focus_mm - f, 1e-3)) * (abs(z_mm - focus_mm) / z_mm);
    return clamp(coc_mm * (canvas_w / max(P.filmback, 1e-3)), 0.0, max(P.radius, 0.0));
}

// Focal-plane view: the SHARP input, greyed, tinted by where each pixel sits
// against the plane — green in focus (under a pixel of CoC), red in front,
// blue behind, strengthening with the CoC toward `radius`. The classic focal-plane
// setup colours. Replaces the gather, so it costs less than the blur it
// stands in for.
fn focal_viz(src: vec4f, z: f32, coc: f32) -> vec4f {
    let l = dot(max(src.rgb, vec3f(0.0)), vec3f(0.2126, 0.7152, 0.0722));
    let grey = vec3f(l / (1.0 + l)) * 0.6;
    let sharp = 1.0 - smoothstep(0.5, 1.5, coc);
    let k = clamp(coc / max(P.radius, 1e-3), 0.0, 1.0);
    var side = vec3f(0.0, 0.25, 1.0);
    if (z < depth_lin(P.focus)) { side = vec3f(1.0, 0.1, 0.1); }
    let tint = mix(side * (0.35 + 0.65 * k), vec3f(0.1, 1.0, 0.2), sharp);
    return vec4f(grey + tint * 0.5, 1.0);
}

// How much of the NEAR layer a pixel is, from its signed CoC (negative =
// in front of focus). Only a BLURRY near pixel joins the layer: a near
// pixel under half a pixel of CoC is sharp, and gathering it at the
// layer's dilated radius would blur what should be crisp — it stays in
// the far/focus layer and gathers at its own tiny CoC instead.
fn near_w(signed_coc: f32) -> f32 {
    return smoothstep(0.5, 2.0, -signed_coc);
}

// Interleaved gradient noise: a per-pixel value in 0..1 with no visible
// structure, which is what breaks the spiral's lock-step across the frame.
fn ign(p: vec2i) -> f32 {
    let f = vec2f(p);
    return fract(52.9829189 * fract(0.06711056 * f.x + 0.00583715 * f.y));
}

const TILE: i32 = 8;
"#;

/// Defocus pass 0 (depth wired): the CoC MAP. Per pixel: the signed circle
/// of confusion (negative in front of focus) and the Z it came from —
/// optionally the NEAREST Z within `edge` pixels, so an antialiased edge
/// pixel travels with the foreground instead of wearing the background's
/// depth over a half-foreground colour. Every later pass reads this instead
/// of re-deriving from the depth texture.
pub const DEFOCUS_COC_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> P: DP;
@group(0) @binding(1) var depth: texture_2d<f32>;
@group(0) @binding(2) var modt: texture_2d<f32>;
@group(0) @binding(3) var cmap: texture_storage_2d<rg32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = vec2i(textureDimensions(cmap));
    if (gid.x >= u32(dims.x) || gid.y >= u32(dims.y)) { return; }
    let p = vec2i(gid.xy);
    // No Z: the uniform disc, every pixel at the focus distance.
    var z = depth_lin(P.focus);
    var coc = max(P.radius, 0.0);
    if (P.has_z != 0.0) {
        let ddims = vec2i(textureDimensions(depth));
        let dp = clamp(vec2i(vec2f(p) * vec2f(ddims) / vec2f(dims)), vec2i(0), ddims - 1);
        z = depth_z(textureLoad(depth, dp, 0));
        let e = clamp(i32(round(P.edge)), 0, 8);
        if (e > 0) {
            for (var dy = -e; dy <= e; dy = dy + 1) {
                for (var dx = -e; dx <= e; dx = dx + 1) {
                    let q = clamp(dp + vec2i(dx, dy), vec2i(0), ddims - 1);
                    z = min(z, depth_z(textureLoad(depth, q, 0)));
                }
            }
        }
        coc = coc_px(z, f32(dims.x));
    }
    // The modulate matte scales the CoC by hand, under the clamp: it can
    // only take blur away, so the tiles' reach stays honest.
    if (P.has_mod != 0.0) {
        let mdims = vec2i(textureDimensions(modt));
        let mp = clamp(vec2i(vec2f(p) * vec2f(mdims) / vec2f(dims)), vec2i(0), mdims - 1);
        coc = coc * clamp(textureLoad(modt, mp, 0).x, 0.0, 1.0);
    }
    var signed = coc;
    if (z < depth_lin(P.focus)) { signed = -coc; }
    textureStore(cmap, gid.xy, vec4f(signed, z, 0.0, 0.0));
}
"#;

/// Defocus pass 1 (depth wired): reduce the CoC map to 8x8 TILES — the
/// largest near CoC and the largest CoC of any kind in each. The next two
/// passes dilate the tile grid by the max radius, which turns it into "the
/// widest CoC that could reach this pixel": Fast mode's near-layer gather
/// radius, Accurate mode's search radius. Conservative by a tile, which only
/// costs taps, never correctness.
pub const DEFOCUS_TILE_WGSL: &str = r#"
@group(0) @binding(0) var cmap: texture_2d<f32>;
@group(0) @binding(1) var tiles: texture_storage_2d<rg32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let tdims = vec2i(textureDimensions(tiles));
    if (gid.x >= u32(tdims.x) || gid.y >= u32(tdims.y)) { return; }
    let dims = vec2i(textureDimensions(cmap));
    let o = vec2i(gid.xy) * 8;
    var near = 0.0;
    var any = 0.0;
    for (var dy = 0; dy < 8; dy = dy + 1) {
        for (var dx = 0; dx < 8; dx = dx + 1) {
            let q = o + vec2i(dx, dy);
            if (q.x >= dims.x || q.y >= dims.y) { continue; }
            let m = textureLoad(cmap, q, 0).r;
            near = max(near, -m);
            any = max(any, abs(m));
        }
    }
    textureStore(tiles, gid.xy, vec4f(near, any, 0.0, 0.0));
}
"#;

/// Defocus passes 2 + 3 (depth wired): separable max over the tile grid,
/// `ceil(radius / 8) + 1` tiles each way. `DIR` is substituted per pass.
pub const DEFOCUS_TILEMAX_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> P: DP;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rg32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let tdims = vec2i(textureDimensions(dst));
    if (gid.x >= u32(tdims.x) || gid.y >= u32(tdims.y)) { return; }
    let p = vec2i(gid.xy);
    let n = i32(ceil(max(P.radius, 0.0) / 8.0)) + 1;
    var m = vec2f(0.0);
    for (var k = -n; k <= n; k = k + 1) {
        let q = clamp(p + vec2i(DIR) * k, vec2i(0), tdims - 1);
        m = max(m, textureLoad(src, q, 0).rg);
    }
    textureStore(dst, gid.xy, vec4f(m, 0.0, 0.0));
}
"#;

/// Defocus Fast pass 4: per-row prefix sums into rgba32float textures (f16
/// would overflow long HDR rows). One thread walks each row — trivially
/// parallel across rows and cheap next to the strip pass it enables.
///
/// With depth wired it ALSO sums the NEAR LAYER: each pixel's colour
/// weighted by how much of the layer it is, and that weight alone. A
/// strip of the near sums is the near layer's contribution to a disc; the
/// plain sums minus it is the far layer's — which is what lets the far
/// gather EXCLUDE the foreground and fill in behind it from what is around.
pub const DEFOCUS_PREFIX_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> P: DP;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var cmap: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var dstn: texture_storage_2d<rgba32float, write>;
@group(0) @binding(5) var dstw: texture_storage_2d<r32float, write>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(src);
    let y = gid.x;
    if (y >= dims.y) { return; }
    var sum = vec4f(0.0);
    var sumn = vec4f(0.0);
    var sumw = 0.0;
    for (var x = 0u; x < dims.x; x = x + 1u) {
        let c = textureLoad(src, vec2u(x, y), 0);
        sum += c;
        textureStore(dst, vec2u(x, y), sum);
        if (P.has_depth != 0.0) {
            let w = near_w(textureLoad(cmap, vec2u(x, y), 0).r);
            sumn += c * w;
            sumw += w;
            textureStore(dstn, vec2u(x, y), sumn);
            textureStore(dstw, vec2u(x, y), vec4f(sumw, 0.0, 0.0, 0.0));
        }
    }
}
"#;

/// Defocus, Fast mode pass 5: exact flat-disc averages via row strips — each
/// of the 2r+1 rows costs TWO prefix lookups instead of 2r+1 taps, so the
/// whole disc is O(r) per pixel instead of O(r^2), and r may vary per pixel
/// for free. Still LINEAR in radius, not flat: 4K measures 7.0 ms at r=8,
/// 20 ms at r=32 and 73 ms at r=128.
///
/// With depth it is TWO LAYERS. The NEAR layer (blurry pixels in front of
/// focus) is gathered at the dilated near radius — the widest near CoC that
/// could reach this pixel — so a foreground object bleeds OUT past its own
/// edge, which a plain gather cannot do. The FAR layer gathers at the
/// pixel's own CoC with the near pixels EXCLUDED from the average, so under
/// a foreground edge it is filled from the background around it — the
/// best guess for pixels the plate never had. Near is composited over far
/// by the near layer's blurred coverage. A disc with no far pixel in it at
/// all (deep inside a foreground object) has nothing to fill from and shows
/// the near colour — the limit of a single-layer plate.
///
/// Normalizes by the in-bounds texel count (no edge darkening).
pub const DEFOCUS_DISC_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> P: DP;
@group(0) @binding(1) var pre: texture_2d<f32>;
@group(0) @binding(2) var pren: texture_2d<f32>;
@group(0) @binding(3) var prew: texture_2d<f32>;
@group(0) @binding(4) var cmap: texture_2d<f32>;
@group(0) @binding(5) var tiles: texture_2d<f32>;
@group(0) @binding(6) var dst: texture_storage_2d<rgba16float, write>;

struct Disc { acc: vec4f, accn: vec4f, accw: f32, count: f32 }

// The pixel itself, off the prefix sums.
fn texel(px: i32, py: i32) -> vec4f {
    var s = textureLoad(pre, vec2u(u32(px), u32(py)), 0);
    if (px > 0) { s -= textureLoad(pre, vec2u(u32(px - 1), u32(py)), 0); }
    return s;
}

// Sums over the disc of radius r about (px, py): plain, near-weighted and
// the near weight itself. r = 0 is the pixel.
fn disc(px: i32, py: i32, r: i32, dims: vec2i, layered: bool) -> Disc {
    var d: Disc;
    if (r == 0) {
        d.acc = texel(px, py);
        d.count = 1.0;
        if (layered) {
            let w = near_w(textureLoad(cmap, vec2i(px, py), 0).r);
            d.accn = d.acc * w;
            d.accw = w;
        }
        return d;
    }
    for (var dy = -r; dy <= r; dy = dy + 1) {
        let yy = py + dy;
        if (yy < 0 || yy >= dims.y) { continue; }
        let wd = i32(floor(sqrt(f32(r * r - dy * dy) + 0.25)));
        let x1 = min(px + wd, dims.x - 1);
        let x0 = max(px - wd - 1, -1);
        if (x1 < 0 || x0 >= x1) { continue; }
        let a = vec2u(u32(x1), u32(yy));
        var s = textureLoad(pre, a, 0);
        var sn = vec4f(0.0);
        var sw = 0.0;
        if (layered) {
            sn = textureLoad(pren, a, 0);
            sw = textureLoad(prew, a, 0).r;
        }
        if (x0 >= 0) {
            let b = vec2u(u32(x0), u32(yy));
            s -= textureLoad(pre, b, 0);
            if (layered) {
                sn -= textureLoad(pren, b, 0);
                sw -= textureLoad(prew, b, 0).r;
            }
        }
        d.acc += s;
        d.accn += sn;
        d.accw += sw;
        d.count += f32(x1 - x0);
    }
    return d;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = vec2i(textureDimensions(dst));
    if (gid.x >= u32(dims.x) || gid.y >= u32(dims.y)) { return; }
    let px = i32(gid.x);
    let py = i32(gid.y);
    if (P.has_depth == 0.0) {
        // No depth: one uniform disc, exactly the node before layers.
        let r = clamp(i32(round(max(P.radius, 0.0))), 0, 256);
        let d = disc(px, py, r, dims, false);
        textureStore(dst, gid.xy, d.acc / max(d.count, 1.0));
        return;
    }
    let m = textureLoad(cmap, vec2i(px, py), 0);
    let coc = abs(m.r);
    if (P.view != 0.0 && P.has_z != 0.0) {
        textureStore(dst, gid.xy, focal_viz(texel(px, py), m.g, coc));
        return;
    }
    let wn = near_w(m.r);
    let t = textureLoad(tiles, vec2i(px, py) / TILE, 0);
    // The near layer reaches as far as the widest near CoC around here;
    // the far layer gathers at the pixel's own CoC — or, where the pixel
    // IS foreground, as far as the near layer, to fill in behind it.
    let rn_f = clamp(t.r, 0.0, max(P.radius, 0.0));
    let rf_f = clamp(mix(coc, rn_f, wn), 0.0, max(P.radius, 0.0));
    let rn = clamp(i32(round(rn_f)), 0, 256);
    let rf = clamp(i32(round(rf_f)), 0, 256);
    if (rn == 0 && rf == 0) {
        textureStore(dst, gid.xy, texel(px, py));
        return;
    }
    let n = disc(px, py, rn, dims, true);
    let f = disc(px, py, rf, dims, true);
    // Near coverage after the blur, and the average near colour under it.
    let cover = clamp(n.accw / max(n.count, 1.0), 0.0, 1.0);
    let near_col = n.accn / max(n.accw, 1e-5);
    let far_w = f.count - f.accw;
    var far_col = (f.acc - f.accn) / max(far_w, 1e-5);
    if (far_w < 1e-3) {
        // Nothing but foreground in reach: no data behind it.
        far_col = near_col;
    }
    textureStore(dst, gid.xy, near_col * cover + far_col * (1.0 - cover));
}
"#;

/// Defocus, Accurate mode: one pass, a golden-angle spiral over the widest
/// CoC that could reach the pixel. Cost is O(samples) per pixel at the max
/// radius and shrinks with the search disc; under-sampling a wide CoC shows
/// as grain, which is what the `samples` knob is for.
///
/// Two things it does that the flat disc cannot:
/// * an arbitrary bokeh kernel — a wired plate (sampled per-channel, so a
///   coloured kernel gives coloured fringing) or a procedural blade
///   polygon with optical vignetting;
/// * the scatter-as-gather occlusion test — a tap contributes only where its
///   OWN CoC reaches this pixel, and a tap BEHIND the centre may spread no
///   further than the centre is itself blurry. Blurry foreground therefore
///   bleeds outward, and sharp foreground does not get bled on.
///
/// Four things that keep it from looking like a fingerprint (it did):
/// * the search radius is per pixel, off the dilated tile map, so a 10 px
///   CoC under a 64 px max is not 128 taps over a 64 px disc with three of
///   them landing;
/// * the spiral is rotated per pixel by noise, so the under-sampling is
///   grain rather than a pattern the eye locks onto across the frame;
/// * the tap count follows the search disc's area (floor 32), so the budget
///   goes where the blur is;
/// * taps sample bilinearly instead of snapping to texels.
///
/// Premultiplied throughout: rgb and a ride the same weights, so a soft
/// matte edge stays neutral.
pub const DEFOCUS_BOKEH_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> P: DP;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var src: texture_2d<f32>;
@group(0) @binding(3) var cmap: texture_2d<f32>;
@group(0) @binding(4) var tiles: texture_2d<f32>;
@group(0) @binding(5) var kern: texture_2d<f32>;
@group(0) @binding(6) var dst: texture_storage_2d<rgba16float, write>;

const TAU: f32 = 6.283185307;
// Golden angle: consecutive taps land maximally out of phase, so a partial
// spiral is still an even disc.
const GOLDEN: f32 = 2.399963230;

// Unit-disc coverage of the aperture polygon. 0 blades = plain circle.
fn blade_mask(v: vec2f) -> f32 {
    let n = P.blades;
    if (n < 3.0) { return step(length(v), 1.0); }
    let seg = TAU / n;
    var a = atan2(v.y, v.x) - radians(P.rotation);
    a = a - seg * floor(a / seg) - seg * 0.5;
    let r = length(v) * cos(a) / cos(seg * 0.5);
    return step(r, 1.0);
}

// Kernel weight for a tap at unit-disc position `v`, at frame position
// `ndc` (-1..1 from centre) for the optical-vignetting term.
fn kernel_w(v: vec2f, ndc: vec2f) -> vec3f {
    // Optical vignetting: clip the aperture against a circle displaced
    // toward the frame edge — round in the middle, cat's-eye at the corners.
    var occl = 1.0;
    if (P.catseye > 0.0) {
        occl = step(length(v - ndc * P.catseye), 1.0);
    }
    if (P.has_kernel != 0.0) {
        let uv = (vec2f(P.kcx, P.kcy) + v * P.kr) / vec2f(textureDimensions(kern));
        if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) { return vec3f(0.0); }
        return max(textureSampleLevel(kern, samp, uv, 0.0).rgb, vec3f(0.0)) * occl;
    }
    return vec3f(blade_mask(v) * occl);
}

// CoC and linear Z at a pixel, off the map. With NO depth wired this
// degenerates to a uniform disc of `radius` at a constant Z, so the mode
// still matches Fast instead of quietly collapsing to a sharp image.
fn coc_z_at(p: vec2i) -> vec2f {
    if (P.has_depth == 0.0) { return vec2f(max(P.radius, 0.0), 0.0); }
    let m = textureLoad(cmap, p, 0);
    return vec2f(abs(m.r), m.g);
}

// Soft highlight clamp: a tap brighter than `clampv` is scaled down to it
// before it spreads, so an HDR speck is a disc rather than a scatter of
// dots at low tap counts. Off at 0.
fn tame(c: vec4f) -> vec4f {
    if (P.clampv <= 0.0) { return c; }
    let l = dot(max(c.rgb, vec3f(0.0)), vec3f(0.2126, 0.7152, 0.0722));
    return c * min(1.0, P.clampv / max(l, 1e-6));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = vec2i(textureDimensions(dst));
    if (gid.x >= u32(dims.x) || gid.y >= u32(dims.y)) { return; }
    let p = vec2i(gid.xy);
    let centre = textureLoad(src, p, 0);
    let cz = coc_z_at(p);
    let coc_c = cz.x;
    let zc = cz.y;
    if (P.view != 0.0 && P.has_z != 0.0) {
        textureStore(dst, gid.xy, focal_viz(centre, zc, coc_c));
        return;
    }
    // Search the widest CoC any neighbour in reach could have: a blurry
    // pixel that far away can still splat onto this one. Per pixel, off
    // the dilated tiles — not the max radius everywhere.
    let rmax = max(P.radius, 0.0);
    var search = rmax;
    if (P.has_depth != 0.0) {
        search = clamp(max(textureLoad(tiles, p / TILE, 0).g, coc_c), 0.0, rmax);
    }
    if (search < 0.5) {
        textureStore(dst, gid.xy, centre);
        return;
    }
    let ndc = (vec2f(p) + 0.5) / vec2f(dims) * 2.0 - 1.0;
    // Taps follow the disc's AREA: the authored count is for the max
    // radius. Floored at 32 — a small disc over fine texture at 8 taps is
    // grain, and 32 taps on a small disc is cheap.
    let smax = clamp(i32(P.samples), 1, 1024);
    let frac = search / max(rmax, 1e-3);
    let n = clamp(i32(ceil(P.samples * frac * frac)), min(32, smax), smax);
    // Per-pixel rotation and radial offset: the same spiral everywhere
    // under-samples the same way everywhere, which reads as a pattern.
    let rot = ign(p) * TAU;
    let jr = ign(p + vec2i(37, 17));
    // The centre tap is always in, so a fully sharp pixel is untouched.
    var acc = tame(centre);
    var wrgb = vec3f(1.0);
    var wa = 1.0;
    let inv = 1.0 / vec2f(dims);
    for (var i = 0; i < n; i = i + 1) {
        let t = (f32(i) + jr) / f32(n);
        // sqrt keeps the spiral area-uniform instead of centre-heavy.
        let rr = sqrt(t);
        let ang = f32(i) * GOLDEN + rot;
        let off = vec2f(cos(ang), sin(ang)) * rr * search;
        let qf = vec2f(p) + 0.5 + off;
        let q = clamp(vec2i(floor(qf)), vec2i(0), dims - 1);
        let tz = coc_z_at(q);
        let coc_t = tz.x;
        let zt = tz.y;
        // A tap behind the centre may spread no further than the centre is
        // itself blurry: that is what stops a soft background washing over
        // a sharp foreground edge.
        var coc_eff = coc_t;
        if (zt >= zc) { coc_eff = min(coc_t, coc_c); }
        if (coc_eff < 0.5) { continue; }
        let d = length(off);
        let reach = smoothstep(coc_eff + 1.0, coc_eff - 1.0, d);
        if (reach <= 0.0) { continue; }
        let w3 = kernel_w(off / coc_eff, ndc) * reach;
        let col = tame(textureSampleLevel(src, samp, clamp(qf * inv, vec2f(0.0), vec2f(1.0)), 0.0));
        acc = vec4f(acc.rgb + col.rgb * w3, acc.a + col.a * dot(w3, vec3f(1.0 / 3.0)));
        wrgb += w3;
        wa += dot(w3, vec3f(1.0 / 3.0));
    }
    textureStore(
        dst,
        gid.xy,
        vec4f(acc.rgb / max(wrgb, vec3f(1e-5)), acc.a / max(wa, 1e-5)),
    );
}
"#;

/// CPU-side mirror of the shared uniform buffer. Rebuilt from graph params on
/// any param edit; uploading it is a single small write_buffer.
pub fn pack_params(graph: &Graph, layout: &ParamLayout, out_size: [f32; 2]) -> Vec<f32> {
    let mut v = vec![0.0f32; layout.array_len() * 4];
    v[0] = out_size[0];
    v[1] = out_size[1];
    for (inst, slot) in &layout.window_slot {
        // Absent (a layout a moment older than the graph): the plate as it
        // is, not a scale of zero.
        let o = graph.reader_windows.get(inst).copied().unwrap_or([0.0, 0.0, 1.0, 1.0]);
        v[slot * 4..slot * 4 + 4].copy_from_slice(&o);
    }
    for (id, slot) in &layout.node_slot {
        let base = slot * 4;
        // The layout may momentarily belong to an older topology than `graph`
        // (a node was just deleted while the new pipeline still compiles) —
        // skip ids the graph no longer has instead of panicking.
        let Some(kind) = graph.nodes.get(id) else { continue };
        // Nor may an id the old layout gave a smaller span: loading a setup
        // reuses ids, so a Grade can arrive where the outgoing pipeline had
        // a one-slot node at the end of its block. Kinds whose span grows
        // with their data clamp themselves to what is left (a stale spline
        // decimates); a Grade's zones do, its own four slots do not.
        let fixed = match kind {
            NodeKind::Spline(_)
            | NodeKind::Curve(_)
            | NodeKind::Custom { .. }
            | NodeKind::Layer(_)
            | NodeKind::LumaKey(_)
            | NodeKind::Cryptomatte(_)
            | NodeKind::Keyer(_)
            | NodeKind::GridWarp(_)
            | NodeKind::Warp(_) => 0,
            NodeKind::Grade(_) => GRADE_SLOTS,
            k => vec4_slots_for(k),
        };
        if base + fixed * 4 > v.len() {
            continue;
        }
        match kind {
            // Unit scale until the compositor knows the layer's size.
            NodeKind::Blender(_) => {
                v[base] = 1.0;
                v[base + 1] = 1.0;
            }
            NodeKind::Transform(t) => {
                // Pivot in the NODE's own format (fallback: pass size).
                let sz = graph
                    .formats
                    .get(id)
                    .map(|(w, h)| [*w as f32, *h as f32])
                    .unwrap_or(out_size);
                // Translate is PIXELS; under proxy the pixels are smaller.
                let mut t = t.clone();
                t.translate[0] *= graph.proxy_scale();
                t.translate[1] *= graph.proxy_scale();
                let hinv = mat3_inv(&transform_applied_px(&t, sz));
                for r in 0..3 {
                    v[base + r * 4..base + r * 4 + 3].copy_from_slice(&hinv[r]);
                }
                if layout.modulated.contains(id) && v.len() >= base + 32 {
                    // The shader rebuilds a PARTIAL transform from the plain
                    // params, which knows no shear or taper: a bent one
                    // blends positions instead (the full map is exact).
                    // Nor does it know an inverse: inverted blends positions.
                    let mode = match t.mod_blend {
                        crate::graph::TransformBlend::Params if !t.bends() && !t.invert => 0.0,
                        _ => 1.0,
                    };
                    let pinned = (t.pin != crate::graph::UNIT_QUAD) as u32 as f32;
                    let p = t.pin;
                    v[base + 12..base + 32].copy_from_slice(&[
                        t.translate[0],
                        t.translate[1],
                        t.rotate.to_radians(),
                        mode,
                        t.scale[0],
                        t.scale[1],
                        t.center[0] * sz[0],
                        t.center[1] * sz[1],
                        p[0][0],
                        p[0][1],
                        p[1][0],
                        p[1][1],
                        p[2][0],
                        p[2][1],
                        p[3][0],
                        p[3][1],
                        sz[0],
                        sz[1],
                        pinned,
                        0.0,
                    ]);
                }
            }
            NodeKind::Stabilize(st) => {
                let sz = graph.formats.get(id).map(|(w, h)| [*w as f32, *h as f32]).unwrap_or(out_size);
                let hinv = corner_pin_inverse(&st.as_corner_pin(), sz);
                for r in 0..3 {
                    v[base + r * 4..base + r * 4 + 3].copy_from_slice(&hinv[r]);
                }
            }
            NodeKind::CornerPin(cp) => {
                let sz = graph
                    .formats
                    .get(id)
                    .map(|(w, h)| [*w as f32, *h as f32])
                    .unwrap_or(out_size);
                let hinv = corner_pin_inverse(cp, sz);
                for r in 0..3 {
                    v[base + r * 4..base + r * 4 + 3].copy_from_slice(&hinv[r]);
                }
                if layout.modulated.contains(id) && v.len() >= base + 32 {
                    // The Transform span with an identity move: no translate,
                    // rotate or scale, pivot anywhere, the pin always on.
                    let mode = match cp.mod_blend {
                        crate::graph::TransformBlend::Params => 0.0,
                        crate::graph::TransformBlend::Coords => 1.0,
                    };
                    let p = cp.effective_to();
                    v[base + 12..base + 32].copy_from_slice(&[
                        0.0, 0.0, 0.0, mode, 1.0, 1.0, 0.0, 0.0, p[0][0], p[0][1], p[1][0],
                        p[1][1], p[2][0], p[2][1], p[3][0], p[3][1], sz[0], sz[1], 1.0, 0.0,
                    ]);
                }
            }
            NodeKind::Grade(g) => {
                v[base..base + 3].copy_from_slice(&g.lift);
                // lift.w is the saturation; it must ALWAYS be written, or a
                // zeroed mirror slot reads as "fully desaturated".
                v[base + 3] = g.saturation.max(0.0);
                // Belt and braces with the UI range: gamma <= 0 turns pow()
                // into inf/NaN downstream.
                let gamma = g.gamma.map(|x| x.max(0.05));
                v[base + 4..base + 7].copy_from_slice(&gamma);
                v[base + 8..base + 11].copy_from_slice(&g.gain);
                v[base + 12..base + 15].copy_from_slice(&g.offset);
                v[base + 16] = g.contrast.max(0.0);
                v[base + 17] = g.pivot;
                // The ranges editor's preview rides in the .w lanes `grade4`
                // does not read (it takes rgb from gamma and gain, and
                // lift.w is the saturation), so it costs no slot and no
                // recompile: gamma.w is the mode, gain.w the zone. Transient
                // by construction — `live_preview` is None unless the app
                // marked this node the view target.
                let (pv, pz) = match g.zones.as_ref().and_then(|z| z.live_preview()) {
                    Some((view, zone)) => (view.code(), zone as f32),
                    None => (0.0, 0.0),
                };
                v[base + 7] = pv;
                v[base + 11] = pz;
                // Zones, each in its fixed span: the four adjustments laid
                // out exactly like the master's, then the band's response
                // packed exactly like a LumaKey's. Same layout-clamp
                // discipline as the LumaKey: the live layout can be one
                // topology behind while the recompile for a new zone lands,
                // so never write past the mirror.
                if let Some(z) = &g.zones {
                    let end = v.len();
                    for i in 0..z.count() {
                        let zb = base + GRADE_SLOTS * 4 + i * ZONE_SLOTS * 4;
                        if zb + ZONE_SLOTS * 4 > end {
                            break;
                        }
                        let a = z.zone(i);
                        v[zb..zb + 3].copy_from_slice(&a.lift);
                        v[zb + 3] = a.saturation.max(0.0);
                        v[zb + 4..zb + 7].copy_from_slice(&a.gamma.map(|x| x.max(0.05)));
                        v[zb + 8..zb + 11].copy_from_slice(&a.gain);
                        v[zb + 12..zb + 15].copy_from_slice(&a.offset);
                        v[zb + 16] = a.contrast.max(0.0);
                        v[zb + 17] = a.pivot;
                        let b = z.key.band(i);
                        let pts = b.curve();
                        let fit = pts.len().min(crate::graph::ZONE_PTS);
                        let hdr = zb + GRADE_SLOTS * 4;
                        v[hdr] = fit as f32;
                        v[hdr + 1] =
                            if b.interp == crate::graph::CurveInterp::Smooth { 1.0 } else { 0.0 };
                        v[hdr + 2] = if b.invert { 1.0 } else { 0.0 };
                        for (k, p) in pts.iter().take(fit).enumerate() {
                            v[hdr + 4 + k * 2] = p[0];
                            v[hdr + 4 + k * 2 + 1] = p[1];
                        }
                    }
                }
            }

            // The dissolve weight for a blending retime, computed per tick
            // by the app (it depends on the tick, not the authored value).
            NodeKind::Speed(p) => v[base] = p.blend_w,
            NodeKind::RangeRemap(p) => v[base] = p.blend_w,
            NodeKind::TimeRemap(p) => v[base] = p.blend_w,
            NodeKind::FrameFix(p) => v[base] = p.blend_w,
            NodeKind::Spline(sp) => {
                // The shape transform is applied HERE, on the control points,
                // in the node's own pixel space — the same call the viewer
                // gizmo makes, so the outline drawn and the outline
                // rasterized cannot drift apart. A shape a roto container
                // lowered into lives in the container's format.
                let sz = graph
                    .format_of(*id)
                    .map(|(w, h)| [w as f32, h as f32])
                    .unwrap_or(out_size);
                let poly = spline_band(sp, sz);
                let end = v.len();
                // The live layout can be one topology behind while the async
                // recompile lands (point count changes the slot span), so
                // never write past the mirror — the swap re-packs next frame.
                if base + 8 <= end {
                    // Matte controls, always written: a zeroed mirror slot
                    // would read as opacity 0, i.e. no matte at all.
                    v[base + 4] = sp.opacity;
                    v[base + 5] = sp.expand;
                    v[base + 6] = sp.falloff.clamp(0.0, 1.0);
                    v[base + 7] = f32::from(sp.invert);
                    // Positions (2 per vec4) then feathers (4 per vec4);
                    // both blocks have to fit inside the live span.
                    let room = end - base - 8;
                    let mut fit = poly.len();
                    while fit > 0 && (fit.div_ceil(2) + fit.div_ceil(4)) * 4 > room {
                        fit -= 1;
                    }
                    // The feather block's vec4 offset from `base`; the shader
                    // derives the same number from the count.
                    let foff = 2 + fit.div_ceil(2);
                    v[base] = fit as f32;
                    // A short span DECIMATES rather than dropping the tail:
                    // a truncated outline closes with a chord straight across
                    // the shape, an evenly thinned one is the same shape at a
                    // coarser step.
                    let pick = |i: usize| poly[i * poly.len() / fit.max(1)];
                    for i in 0..fit {
                        let o = base + 8 + i * 2;
                        // Pixels of the node's own format (see `spt`).
                        let p = pick(i).0;
                        v[o] = p[0] * sz[0];
                        v[o + 1] = p[1] * sz[1];
                    }
                    for i in 0..fit {
                        v[base + foff * 4 + i] = spline_band_lane(pick(i).1);
                    }
                    // Bounding circle of the samples the shader walks: the
                    // box's centre, and the farthest sample, in the node's
                    // pixels (the space the shader measures in), plus how far the
                    // soft edge reaches past the outline — half the widest
                    // feather, a positive expand, and a pixel of slack.
                    let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
                    for i in 0..fit {
                        let p = pick(i).0;
                        for k in 0..2 {
                            lo[k] = lo[k].min(p[k]);
                            hi[k] = hi[k].max(p[k]);
                        }
                    }
                    let c = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5];
                    let (mut r, mut reach) = (0.0f32, 1e-3f32);
                    for i in 0..fit {
                        let (p, b) = pick(i);
                        let dx = (p[0] - c[0]) * sz[0];
                        let dy = (p[1] - c[1]) * sz[1];
                        r = r.max((dx * dx + dy * dy).sqrt());
                        // How far past the outline coverage can still be
                        // above zero: the band's outer edge.
                        reach = reach.max(b[1]);
                    }
                    v[base + 1] = r + reach + sp.expand.max(0.0) + 1.0;
                    v[base + 2] = c[0] * sz[0];
                    v[base + 3] = c[1] * sz[1];
                }
            }
            NodeKind::Curve(cv) => {
                // Same layout-clamp discipline as Spline: never write past
                // the (possibly one-topology-old) mirror. Runs are padded to
                // whole vec4s so each channel's curve starts on a boundary.
                let end = v.len();
                if base + 8 <= end {
                    v[base + 4] =
                        if cv.interp == crate::graph::CurveInterp::Smooth { 1.0 } else { 0.0 };
                    v[base + 5] = if cv.split { 1.0 } else { 0.0 };
                    let mut at = base + 8;
                    for (c, src) in std::iter::once(&cv.points)
                        .chain(cv.rgb.iter())
                        .enumerate()
                    {
                        let mut pts = src.clone();
                        pts.sort_by(|a, b| {
                            a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        let room = end.saturating_sub(at) / 2;
                        let fit = pts.len().min(room);
                        v[base + c] = fit as f32;
                        for (i, p) in pts.iter().take(fit).enumerate() {
                            v[at + i * 2] = p[0];
                            v[at + i * 2 + 1] = p[1];
                        }
                        // Pad to the next vec4: an odd run would otherwise
                        // put the following channel on a half-slot.
                        at += fit.div_ceil(2) * 4;
                    }
                }
            }
            NodeKind::Custom { type_id, params } => {
                // Manifest order, 4 per vec4 — the same order instantiate()
                // used for its accessors. Missing values take the default.
                if let Some(def) = crate::plugins::get(type_id) {
                    let end = v.len();
                    for (k, spec) in def.params.iter().enumerate() {
                        let at = base + (k / 4) * 4 + (k % 4);
                        if at < end {
                            v[at] =
                                params.get(&spec.name).copied().unwrap_or(spec.default);
                        }
                    }
                }
            }
            NodeKind::Mix(m) => {
                v[base] = m.amount.clamp(0.0, 1.0);
            }
            NodeKind::Blend { mix, gain, .. } => {
                v[base] = mix.clamp(0.0, 1.0);
                v[base + 1] = gain[0];
                v[base + 2] = gain[1];
            }
            NodeKind::Recombine(p) => {
                for i in 0..p.n.clamp(1, 16) as usize {
                    v[base + i] = p.gain.get(i).copied().unwrap_or(1.0);
                }
            }
            NodeKind::MatteCombine(m) => {
                v[base] = m.mix.clamp(0.0, 1.0);
            }
            NodeKind::Noise(n) => {
                v[base] = n.scale.max(1e-4);
                v[base + 1] = n.seed;
                // One number for the whole sub-graph: the tick times the
                // global speed. Resolved on the CPU because the engine has
                // no idea what time it is.
                v[base + 2] = n.phase();
                v[base + 3] = n.seed_offset;
                crate::field::pack(&n.graph, base + 4, &mut v[..]);
            }
            NodeKind::Field(_) => {}
            NodeKind::Sdf(p) => {
                // Spline points go to the node's OWN render pixels; lengths
                // ride raw and the shader applies the proxy.
                let w = graph.format_of(*id).map(|(w, _)| w as f32).unwrap_or(out_size[0]);
                p.pack(base, w.max(1.0), graph.proxy_scale(), &mut v[..]);
            }
            NodeKind::SdfNode(_) => {}
            NodeKind::Layer(lp) => {
                // Same layout-clamp discipline as Spline/Curve: the live
                // layout can be one topology behind while an async recompile
                // lands, so never write past the mirror.
                let end = v.len();
                for (i, l) in
                    lp.layers.iter().take(crate::graph::LAYER_MAX).enumerate()
                {
                    let b = base + i * 4;
                    if b + 4 > end {
                        break;
                    }
                    // Opacity folded with the trim gate and the mute/solo
                    // switches: one slot a layer, never a recompile.
                    let on = if lp.audible(i) { 1.0 } else { 0.0 };
                    v[b] = l.opacity.clamp(0.0, 1.0) * l.gate.clamp(0.0, 1.0) * on;
                }
            }
            NodeKind::LumaKey(k) => {
                // Header (count, smooth, invert) then the response points.
                let end = v.len();
                if base + 4 <= end {
                    let pts = k.curve();
                    let fit = pts.len().min((end - base - 4) / 2);
                    v[base] = fit as f32;
                    v[base + 1] =
                        if k.interp() == crate::graph::CurveInterp::Smooth { 1.0 } else { 0.0 };
                    v[base + 2] = if k.invert() { 1.0 } else { 0.0 };
                    for (i, p) in pts.iter().take(fit).enumerate() {
                        v[base + 4 + i * 2] = p[0];
                        v[base + 4 + i * 2 + 1] = p[1];
                    }
                }
            }
            NodeKind::Cryptomatte(c) => {
                let end = (base + crate::cryptomatte::CryptomatteParams::SLOTS * 4).min(v.len());
                c.pack(&mut v[base..end]);
            }
            NodeKind::Keyer(k) => {
                // Same layout-clamp discipline as Spline/Curve: the live
                // layout can be one topology behind while an async recompile
                // lands, so never write past the mirror.
                let end = v.len();
                if base + 8 <= end {
                    let fit = k
                        .blobs
                        .len()
                        .min(crate::keyer::MAX_BLOBS)
                        .min((end - base - 8) / 8);
                    v[base] = k.tilt;
                    v[base + 1] = k.lo;
                    v[base + 2] = k.hi.max(k.lo + 1e-4);
                    v[base + 3] = k.gamma.max(1e-3);
                    v[base + 4] = fit as f32;
                    v[base + 5] = if k.invert { 1.0 } else { 0.0 };
                    // Preview is transient: only the node being viewed gets
                    // it, and only in Source mode.
                    v[base + 6] = if k.previewing
                        && k.view == crate::keyer::KeyerView::Source
                    {
                        1.0
                    } else {
                        0.0
                    };
                    for (i, b) in k.blobs.iter().take(fit).enumerate() {
                        let o = base + 8 + i * 8;
                        v[o] = b.center[0];
                        v[o + 1] = b.center[1];
                        v[o + 2] = b.center[2];
                        v[o + 3] = b.weight;
                        v[o + 4] = b.tol[0].max(1e-4);
                        v[o + 5] = b.tol[1].max(1e-4);
                        v[o + 6] = b.tol[2].max(1e-4);
                        v[o + 7] = b.soft.max(0.0);
                    }
                }
            }
            NodeKind::Grid(g) => {
                v[base] = g.spacing[0];
                v[base + 1] = g.spacing[1];
                v[base + 2] = g.width;
                v[base + 4] = g.offset[0];
                v[base + 5] = g.offset[1];
                v[base + 8..base + 12].copy_from_slice(&g.line);
                v[base + 12..base + 16].copy_from_slice(&g.back);
            }
            NodeKind::ToMatte(t) => {
                v[base..base + 4].copy_from_slice(&t.channel.weights());
                v[base + 4] = t.lo;
                v[base + 5] = t.hi;
                v[base + 6] = if t.invert { 1.0 } else { 0.0 };
            }
            NodeKind::Mult(m) => {
                v[base..base + 3].copy_from_slice(&m.value);
                v[base + 3] = if m.alpha { 1.0 } else { 0.0 };
            }
            NodeKind::CorridorKey(p) => p.pack(&mut v[base..base + 4]),
            NodeKind::Normals(p) => p.pack(&mut v[base..base + 4]),
            NodeKind::Depth(p) => p.pack(&mut v[base..base + 4]),
            NodeKind::LtxHdr(p) => {
                let f = graph.format_of(*id).map(|(w, h)| [w as f32, h as f32]).unwrap_or(out_size);
                p.pack(f, &mut v[base..base + 4]);
            }
            NodeKind::Flow(p) => p.pack(graph.proxy_scale(), &mut v[base..base + 4]),
            NodeKind::MorphDissolve(p) => {
                p.pack(graph.proxy_scale(), &mut v[base..base + crate::graph::MorphDissolveParams::VEC4_SLOTS * 4])
            }
            NodeKind::VectorWarp(p) => p.pack(&mut v[base..base + 4]),
            NodeKind::GenAi(_) => {
                let f = graph
                    .format_of(*id)
                    .map(|(w, h)| [w as f32, h as f32])
                    .unwrap_or(out_size);
                GenAiParams::pack(f, &mut v[base..base + 4]);
            }
            NodeKind::Reformat(rf) => {
                // Output px -> source px. The source's format is its own
                // (formats are stamped per node), so the fit is exact even
                // when this node is viewed through something larger.
                let src = graph
                    .input_of(*id, 0)
                    .and_then(|s| graph.formats.get(&s))
                    .map(|(w, h)| [*w as f32, *h as f32])
                    .unwrap_or(out_size);
                // Size 0 is "the format this node is stamped with": what a
                // lowering mints to fit a picture to another node's format
                // (`nodes::relight::lower`, by `format_alias`).
                let dst = if !rf.by_factor && rf.size == (0, 0) {
                    graph.format_of(*id).map(|(w, h)| [w as f32, h as f32]).unwrap_or(out_size)
                } else {
                    let out = rf.out_size((src[0].max(1.0) as u32, src[1].max(1.0) as u32));
                    [out.0.max(1) as f32, out.1.max(1) as f32]
                };
                let [sx, sy] = reformat_scale(rf, src, dst);
                // The shader maps forward through the INVERSE of the fit.
                let (ix, iy) = (1.0 / sx.max(1e-6), 1.0 / sy.max(1e-6));
                let (tx, ty) = if rf.center {
                    (
                        -(dst[0] - src[0] * sx) * 0.5 * ix,
                        -(dst[1] - src[1] * sy) * 0.5 * iy,
                    )
                } else {
                    (0.0, 0.0)
                };
                v[base] = ix;
                v[base + 1] = iy;
                v[base + 2] = tx;
                v[base + 3] = ty;
            }
            NodeKind::Shuffle(sh) => {
                let (rows, consts) = sh.matrix();
                for (r, row) in rows.iter().enumerate() {
                    v[base + r * 4..base + r * 4 + 4].copy_from_slice(row);
                }
                v[base + 16..base + 20].copy_from_slice(&consts);
            }
            NodeKind::Saturation(sp) => sp.pack(&mut v[base..base + SatParams::VEC4_SLOTS * 4]),
            NodeKind::Contrast(p) => p.pack(&mut v[base..base + crate::graph::ContrastParams::VEC4_SLOTS * 4]),
            NodeKind::Blank(p) => p.pack(&mut v[base..base + BlankParams::VEC4_SLOTS * 4]),
            // Lengths are frame heights, packed as pixels of the node's
            // own format (proxy included).
            NodeKind::LensFlare(p) => {
                let fmt = graph.formats.get(id).map(|(w, h)| [*w as f32, *h as f32]).unwrap_or(out_size);
                p.pack_fmt(&mut v[base..base + crate::graph::LensFlareParams::VEC4_SLOTS * 4], fmt)
            }
            NodeKind::Particles(p) => {
                p.pack(&mut v[base..base + crate::graph::ParticlesParams::VEC4_SLOTS * 4])
            }
            NodeKind::ParticleNode(_) => {}
            NodeKind::Splat(p) => p.pack(&mut v[base..base + crate::graph::SplatParams::VEC4_SLOTS * 4]),
            NodeKind::Plasma(p) => {
                p.pack_fmt(&mut v[base..base + crate::graph::PlasmaParams::VEC4_SLOTS * 4], graph.proxy_scale())
            }
            // The disc is canvas pixels (scaled by the proxy); a latlong
            // spans the node's own format.
            NodeKind::Planet(p) => {
                let fmt = graph.formats.get(id).map(|(w, h)| [*w as f32, *h as f32]).unwrap_or(out_size);
                p.pack_fmt(&mut v[base..base + crate::graph::PlanetParams::VEC4_SLOTS * 4], fmt, graph.proxy_scale())
            }
            // The field of view spans the node's OWN format (proxy
            // included); star size is pixels, scaled by the proxy.
            NodeKind::Starfield(p) => {
                let fmt = graph.formats.get(id).map(|(w, h)| [*w as f32, *h as f32]).unwrap_or(out_size);
                p.pack_fmt(
                    &mut v[base..base + crate::graph::StarfieldParams::VEC4_SLOTS * 4],
                    fmt,
                    graph.proxy_scale(),
                )
            }
            // The kernels are drawn in their OWN format (proxy included).
            NodeKind::GlintKernel(p) => {
                let fmt = graph.format_of(*id).map(|(w, h)| [w as f32, h as f32]).unwrap_or(out_size);
                p.pack_fmt(&mut v[base..base + 16], fmt, graph.proxy_scale())
            }
            NodeKind::BokehKernel(p) => {
                let fmt = graph.format_of(*id).map(|(w, h)| [w as f32, h as f32]).unwrap_or(out_size);
                p.pack_fmt(&mut v[base..base + 16], fmt, graph.proxy_scale())
            }
            NodeKind::Grain(p) => {
                // Grain sizes are pixels: a proxy halves them.
                p.pack_scaled(&mut v[base..base + GrainParams::VEC4_SLOTS * 4], graph.proxy_scale())
            }
            NodeKind::Gradient(p) => {
                // The ramp spans the node's OWN format (proxy included).
                let fmt = graph
                    .formats
                    .get(id)
                    .map(|(w, h)| [*w as f32, *h as f32])
                    .unwrap_or(out_size);
                p.pack_fmt(&mut v[base..base + GradientParams::VEC4_SLOTS * 4], fmt)
            }
            NodeKind::ColorCompress(p) => {
                p.pack(&mut v[base..base + ColorCompressParams::VEC4_SLOTS * 4])
            }
            NodeKind::Lut(p) => {
                p.pack(&mut v[base..base + crate::graph::LutParams::VEC4_SLOTS * 4])
            }
            // A text layer is drawn outside the pass (`textgpu`): no uniforms.
            NodeKind::Text(_) => {}
            // The corners are the node's OWN format (proxy included).
            NodeKind::Burnin(p) => {
                let fmt = graph.formats.get(id).map(|(w, h)| [*w as f32, *h as f32]).unwrap_or(out_size);
                p.pack_fmt(
                    &mut v[base..base + crate::graph::BurninParams::VEC4_SLOTS * 4],
                    fmt,
                    graph.proxy_scale(),
                )
            }
            NodeKind::HueRotate(p) => {
                p.pack(&mut v[base..base + HueRotateParams::VEC4_SLOTS * 4])
            }
            // The line and the turn-over span the node's OWN format (proxy
            // included), like a Gradient's ramp.
            NodeKind::Mirror(p) => {
                let fmt = graph.formats.get(id).map(|(w, h)| [*w as f32, *h as f32]).unwrap_or(out_size);
                p.pack_fmt(&mut v[base..base + crate::graph::MirrorParams::VEC4_SLOTS * 4], fmt)
            }
            NodeKind::FlipFlop(p) => {
                let fmt = graph.formats.get(id).map(|(w, h)| [*w as f32, *h as f32]).unwrap_or(out_size);
                p.pack_fmt(&mut v[base..base + crate::graph::FlipFlopParams::VEC4_SLOTS * 4], fmt)
            }
            // The turn is read in the node's OWN (turned) format.
            NodeKind::Turn(p) => {
                let fmt = graph.formats.get(id).map(|(w, h)| [*w as f32, *h as f32]).unwrap_or(out_size);
                p.pack_fmt(&mut v[base..base + crate::graph::TurnParams::VEC4_SLOTS * 4], fmt)
            }
            // The box in the node's OWN (full) format, the scale from the
            // cropped picture's (proxy included either way).
            NodeKind::Uncrop(p) => {
                let fmt_of = |n: NodeId| {
                    graph.formats.get(&n).map(|(w, h)| [*w as f32, *h as f32]).unwrap_or(out_size)
                };
                let fmt = fmt_of(*id);
                let input = graph.input_of(*id, 0).map_or(fmt, fmt_of);
                p.pack_fmt(&mut v[base..base + crate::graph::UncropParams::VEC4_SLOTS * 4], fmt, input)
            }
            // Each input's fit into the node's own format, and the wipe in
            // its pixels (proxy included throughout).
            NodeKind::Compare(p) => {
                let fmt_of = |n: NodeId| {
                    graph.formats.get(&n).map(|(w, h)| [*w as f32, *h as f32]).unwrap_or(out_size)
                };
                let fmt = fmt_of(*id);
                let a = graph.input_of(*id, 0).map_or(fmt, fmt_of);
                let b = graph.input_of(*id, 1).map_or(fmt, fmt_of);
                p.pack_fmt(&mut v[base..base + crate::graph::CompareParams::VEC4_SLOTS * 4], fmt, a, b)
            }
            // A windowed bake's box corner, in the node's own pixels.
            NodeKind::GraphBake { baked: true, window: Some(o), .. } => {
                let fmt = graph.formats.get(id).map(|(w, h)| [*w as f32, *h as f32]).unwrap_or(out_size);
                v[base] = (o[0] * fmt[0]).round();
                v[base + 1] = (o[1] * fmt[1]).round();
            }
            // A crop's box in the node's own format, a reformat's corner in
            // its input's (proxy included either way).
            NodeKind::SoftCrop(p) => {
                let fmt_of = |n: NodeId| {
                    graph.formats.get(&n).map(|(w, h)| [*w as f32, *h as f32]).unwrap_or(out_size)
                };
                let fmt = fmt_of(*id);
                let input = graph.input_of(*id, 0).map_or(fmt, fmt_of);
                p.pack_fmt_in(&mut v[base..base + p.slots() * 4], fmt, input)
            }
            NodeKind::Gamma(p) => p.pack(&mut v[base..base + GammaParams::VEC4_SLOTS * 4]),
            NodeKind::ToneMap(p) => p.pack(&mut v[base..base + ToneMapParams::VEC4_SLOTS * 4]),
            // U and V are normalised over the SOURCE's own format (the usual convention),
            // so the span carries it: the shader scales by it.
            NodeKind::UvWarp(p) => {
                let src = graph
                    .input_of(*id, 0)
                    .and_then(|s| graph.formats.get(&s))
                    .map(|(w, h)| [*w as f32, *h as f32])
                    .unwrap_or(out_size);
                p.pack_sized(src, &mut v[base..base + UvWarpParams::VEC4_SLOTS * 4])
            }
            // The view spans the node's OWN format (proxy included); the
            // lookup lands in the latlong's.
            NodeKind::Env(p) => {
                let fmt_of = |n: NodeId| graph.formats.get(&n).map(|(w, h)| [*w as f32, *h as f32]);
                let fmt = fmt_of(*id).unwrap_or(out_size);
                let src = graph.input_of(*id, crate::nodes::env::IMAGE_INPUT).and_then(fmt_of).unwrap_or(fmt);
                p.pack_fmt(&mut v[base..base + crate::graph::EnvParams::VEC4_SLOTS * 4], fmt, src)
            }
            // The latlong view spans the node's OWN format; the lookup lands
            // in input 0's. Thickness and relief are pixels (the proxy's).
            NodeKind::Refract(p) => {
                let fmt_of = |n: NodeId| graph.formats.get(&n).map(|(w, h)| [*w as f32, *h as f32]);
                let fmt = fmt_of(*id).unwrap_or(out_size);
                let src =
                    graph.input_of(*id, crate::nodes::refract::IMAGE_INPUT).and_then(fmt_of).unwrap_or(fmt);
                let end = base + crate::graph::RefractParams::VEC4_SLOTS * 4;
                p.pack_fmt(&mut v[base..end], fmt, src, graph.proxy_scale())
            }
            // The model is normalised over the PLATE, which is the input
            // undistorting and the output distorting; both formats go in.
            NodeKind::LensDistort(p) => {
                let src = graph
                    .input_of(*id, 0)
                    .and_then(|s| graph.formats.get(&s))
                    .map(|(w, h)| [*w as f32, *h as f32]);
                let out = graph.formats.get(id).map(|(w, h)| [*w as f32, *h as f32]);
                let (src, out) = match (src, out) {
                    (Some(s), Some(o)) => (s, o),
                    (Some(s), None) => (s, s),
                    (None, Some(o)) => (o, o),
                    (None, None) => (out_size, out_size),
                };
                p.pack_geom(&p.geom(src, out), &mut v[base..base + crate::graph::LensDistortParams::VEC4_SLOTS * 4])
            }
            // Points are normalised over the node's own format.
            NodeKind::GridWarp(p) => {
                let fmt = graph
                    .formats
                    .get(id)
                    .map(|(w, h)| [*w as f32, *h as f32])
                    .unwrap_or(out_size);
                // Same layout-clamp discipline as Spline/Curve: while a
                // re-grid's recompile lands the live layout is one topology
                // behind, so never write past the mirror.
                let end = (base + p.slots() * 4).min(v.len());
                if base + crate::graph::GridWarpParams::VEC4_SLOTS * 4 <= end {
                    p.pack_fmt(fmt, &mut v[base..end])
                }
            }
            // Curves normalised over the node's format; falloff in full-res
            // pixels, scaled by the proxy here.
            NodeKind::Warp(p) => {
                let fmt = graph
                    .formats
                    .get(id)
                    .map(|(w, h)| [*w as f32, *h as f32])
                    .unwrap_or(out_size);
                // Same layout-clamp discipline: a new curve is a new
                // topology, and until its recompile lands the live layout
                // (and the mirror) is one curve short. Writing the new span
                // anyway was the crash on the first S-click of a curve.
                let end = (base + p.slots() * 4).min(v.len());
                if base + crate::graph::WarpParams::VEC4_SLOTS * 4 <= end {
                    p.pack_fmt(fmt, graph.proxy_scale(), &mut v[base..end])
                }
            }
            // x, y, width and height are FULL-resolution pixels: the shader
            // divides by the proxy it is handed here.
            NodeKind::Expression(p) => p.pack_scaled(
                &mut v[base..base + crate::graph::ExpressionParams::VEC4_SLOTS * 4],
                graph.proxy_scale(),
            ),
            NodeKind::Despill(p) => p.pack(&mut v[base..base + DespillParams::VEC4_SLOTS * 4]),
            NodeKind::ScreenKey(p) => {
                p.pack(&mut v[base..base + ScreenKeyParams::VEC4_SLOTS * 4])
            }
            NodeKind::ColorSlice(p) => {
                p.pack(&mut v[base..base + ColorSliceParams::VEC4_SLOTS * 4])
            }
            NodeKind::ToData(p) => p.pack(&mut v[base..base + ToDataParams::VEC4_SLOTS * 4]),
            NodeKind::ToRgba(p) => p.pack(&mut v[base..base + ToRgbaParams::VEC4_SLOTS * 4]),
            _ => {}
        }
    }
    v
}

/// The per-axis scale a Reformat applies to its source, by mode. Public so
/// the app can draw the resulting frame.
pub fn reformat_scale(
    rf: &crate::graph::ReformatParams,
    src: [f32; 2],
    dst: [f32; 2],
) -> [f32; 2] {
    use crate::graph::ResizeMode;
    let (sw, sh) = (src[0].max(1.0), src[1].max(1.0));
    let (rx, ry) = (dst[0] / sw, dst[1] / sh);
    let k = rf.scale.abs().max(1e-4);
    if rf.by_factor {
        // The format followed the input, so the picture fills it exactly.
        return [rx * k, ry * k];
    }
    match rf.mode {
        ResizeMode::Distort => [rx * k, ry * k],
        ResizeMode::Fit => [rx.min(ry) * k; 2],
        ResizeMode::Fill => [rx.max(ry) * k; 2],
        ResizeMode::None => [k; 2],
    }
}

/// Forward homography of a CornerPin in PIXEL space: source px -> dest px.
/// (Heckbert unit-square -> quad, pre-scaled by 1/size on the source side.)
/// Public so the app can deform upstream manipulators through it.
pub fn corner_pin_forward_px(
    cp: &crate::graph::CornerPinParams,
    out: [f32; 2],
) -> [[f32; 3]; 3] {
    let h = corner_pin_forward_unit(cp, out);
    let mut hp = h;
    for r in 0..3 {
        hp[r][0] /= out[0];
        hp[r][1] /= out[1];
    }
    hp
}

/// 3x3 inverse (adjugate/det, det guarded). Public for the app's gizmos.
pub fn mat3_inv(h: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let det = h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1])
        - h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0])
        + h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0]);
    let d = if det.abs() < 1e-12 { 1e-12 } else { det };
    [
        [
            (h[1][1] * h[2][2] - h[1][2] * h[2][1]) / d,
            (h[0][2] * h[2][1] - h[0][1] * h[2][2]) / d,
            (h[0][1] * h[1][2] - h[0][2] * h[1][1]) / d,
        ],
        [
            (h[1][2] * h[2][0] - h[1][0] * h[2][2]) / d,
            (h[0][0] * h[2][2] - h[0][2] * h[2][0]) / d,
            (h[0][2] * h[1][0] - h[0][0] * h[1][2]) / d,
        ],
        [
            (h[1][0] * h[2][1] - h[1][1] * h[2][0]) / d,
            (h[0][1] * h[2][0] - h[0][0] * h[2][1]) / d,
            (h[0][0] * h[1][1] - h[0][1] * h[1][0]) / d,
        ],
    ]
}

/// Inverse homography for the shader: OUTPUT pixel -> SOURCE pixel.
fn corner_pin_inverse(cp: &crate::graph::CornerPinParams, out: [f32; 2]) -> [[f32; 3]; 3] {
    mat3_inv(&corner_pin_forward_px(cp, out))
}

/// Heckbert square->quad, dest in pixels, source in the unit square. The
/// pin's EFFECTIVE quad, so a `from` other than the frame is folded in.
fn corner_pin_forward_unit(cp: &crate::graph::CornerPinParams, out: [f32; 2]) -> [[f32; 3]; 3] {
    square_to_quad(cp.effective_to().map(|c| [c[0] * out[0], c[1] * out[1]]))
}

/// Heckbert's unit-square -> quad homography (corners TL, TR, BR, BL), in
/// whatever units the quad is in.
pub fn square_to_quad(q: [[f32; 2]; 4]) -> [[f32; 3]; 3] {
    let (x0, y0) = (q[0][0], q[0][1]);
    let (x1, y1) = (q[1][0], q[1][1]);
    let (x2, y2) = (q[2][0], q[2][1]);
    let (x3, y3) = (q[3][0], q[3][1]);
    let (dx1, dy1) = (x1 - x2, y1 - y2);
    let (dx2, dy2) = (x3 - x2, y3 - y2);
    let (sx, sy) = (x0 - x1 + x2 - x3, y0 - y1 + y2 - y3);
    if sx.abs() < 1e-6 && sy.abs() < 1e-6 {
        [[x1 - x0, x3 - x0, x0], [y1 - y0, y3 - y0, y0], [0.0, 0.0, 1.0]]
    } else {
        let den = dx1 * dy2 - dx2 * dy1;
        let den = if den.abs() < 1e-9 { 1e-9 } else { den };
        let g = (sx * dy2 - dx2 * sy) / den;
        let hh = (dx1 * sy - sx * dy1) / den;
        [
            [x1 - x0 + g * x1, x3 - x0 + hh * x3, x0],
            [y1 - y0 + g * y1, y3 - y0 + hh * y3, y0],
            [g, hh, 1.0],
        ]
    }
}

/// A Transform's whole forward map in its own pixels, source -> output: its
/// corner pin (`pin`, the unit square when unpinned) and then its affine.
/// Public for the viewer's gizmo and the downstream-deformation chain.
pub fn transform_forward_px(t: &crate::graph::TransformParams, out_size: [f32; 2]) -> [[f32; 3]; 3] {
    let [m00, m01, tx, m10, m11, ty] = forward_affine(t, out_size);
    let mut a = [[m00, m01, tx], [m10, m11, ty], [0.0, 0.0, 1.0]];
    // Shear and taper bend the picture about the pivot BEFORE it turns and
    // scales; neither set, the matrix is the plain affine it always was.
    if t.bends() {
        a = mat3_mul3(&a, &shear_taper_px(t, out_size));
    }
    if t.pin == crate::graph::UNIT_QUAD {
        return a;
    }
    let p = corner_pin_forward_px(&crate::graph::CornerPinParams::pinned(t.pin), out_size);
    mat3_mul3(&a, &p)
}

/// The map a Transform actually applies, source -> output in its own
/// pixels: `transform_forward_px`, or its inverse when `invert` is on. The
/// shader and anything that deforms THROUGH the node use this; the gizmo,
/// which edits the authored knobs, keeps `transform_forward_px`.
pub fn transform_applied_px(t: &crate::graph::TransformParams, out_size: [f32; 2]) -> [[f32; 3]; 3] {
    let h = transform_forward_px(t, out_size);
    if t.invert { mat3_inv(&h) } else { h }
}

fn mat3_mul3(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut m = [[0.0f32; 3]; 3];
    for (r, row) in m.iter_mut().enumerate() {
        for (c, v) in row.iter_mut().enumerate() {
            *v = (0..3).map(|k| a[r][k] * b[k][c]).sum();
        }
    }
    m
}

/// A Transform's shear and taper as one homography in the node's pixels,
/// about its pivot. Shear is `[[1, sx], [sy, 1]]`. The taper works in the
/// axis frame (u along the axis, v across it, L the format's half extent
/// along u): `u' = (u + L t) / w`, `v' = v / w`, `w = 1 + t u / L` — the
/// ends u = ±L stay put, their widths go to 1 / (1 ± t), lines stay lines.
pub fn shear_taper_px(t: &crate::graph::TransformParams, size: [f32; 2]) -> [[f32; 3]; 3] {
    let (cx, cy) = (t.center[0] * size[0], t.center[1] * size[1]);
    let sh = [[1.0, t.shear[0], 0.0], [t.shear[1], 1.0, 0.0], [0.0, 0.0, 1.0]];
    let tp = t.taper.clamp(-0.95, 0.95);
    let mut k = sh;
    if tp != 0.0 {
        let (s, c) = t.taper_angle.to_radians().sin_cos();
        // The axis (y down), and the frame turning pixels into (u, v).
        let (ax, ay) = (c, -s);
        let rf = [[ax, ay, 0.0], [-ay, ax, 0.0], [0.0, 0.0, 1.0]];
        let rt = [[ax, -ay, 0.0], [ay, ax, 0.0], [0.0, 0.0, 1.0]];
        let l = (0.5 * (ax.abs() * size[0] + ay.abs() * size[1])).max(1.0);
        let p = [[1.0, 0.0, l * tp], [0.0, 1.0, 0.0], [tp / l, 0.0, 1.0]];
        k = mat3_mul3(&mat3_mul3(&rt, &mat3_mul3(&p, &rf)), &k);
    }
    // About the pivot.
    let to = [[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]];
    let from = [[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]];
    mat3_mul3(&to, &mat3_mul3(&k, &from))
}

/// Forward affine in output-pixel space: where the source image lands.
/// Returns [m00, m01, tx, m10, m11, ty]; p' = M*p + t. Public for the
/// viewer's transform gizmo.
pub fn forward_affine(t: &crate::graph::TransformParams, out_size: [f32; 2]) -> [f32; 6] {
    // Negative scale is a flip; only guard the magnitude away from zero.
    let guard = |v: f32| if v.abs() < 1e-6 { 1e-6 } else { v };
    let (sx, sy) = (guard(t.scale[0]), guard(t.scale[1]));
    let rad = t.rotate.to_radians();
    let (s, c) = rad.sin_cos();
    let (cx, cy) = (t.center[0] * out_size[0], t.center[1] * out_size[1]);
    // translate(t) * translate(pivot) * rot * scale * translate(-pivot)
    let m00 = c * sx;
    let m01 = -s * sy;
    let m10 = s * sx;
    let m11 = c * sy;
    let tx = t.translate[0] + cx - (m00 * cx + m01 * cy);
    let ty = t.translate[1] + cy - (m10 * cx + m11 * cy);
    [m00, m01, tx, m10, m11, ty]
}

/// Inverse of the forward affine: maps OUTPUT pixel -> SOURCE pixel (what the
/// fused shader samples with).
#[cfg(test)]
fn inverse_affine(t: &crate::graph::TransformParams, out_size: [f32; 2]) -> [f32; 6] {
    let [m00, m01, tx, m10, m11, ty] = forward_affine(t, out_size);
    // Invert 2x3.
    let det = m00 * m11 - m01 * m10;
    let inv_det = if det.abs() < 1e-12 { 0.0 } else { 1.0 / det };
    let i00 = m11 * inv_det;
    let i01 = -m01 * inv_det;
    let i10 = -m10 * inv_det;
    let i11 = m00 * inv_det;
    let itx = -(i00 * tx + i01 * ty);
    let ity = -(i10 * tx + i11 * ty);
    [i00, i01, itx, i10, i11, ity]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Graph, NodeKind};

    fn apply(h: &[[f32; 3]; 3], p: [f32; 2]) -> [f32; 2] {
        let w = h[2][0] * p[0] + h[2][1] * p[1] + h[2][2];
        [
            (h[0][0] * p[0] + h[0][1] * p[1] + h[0][2]) / w,
            (h[1][0] * p[0] + h[1][1] * p[1] + h[1][2]) / w,
        ]
    }

    fn near(a: [f32; 2], b: [f32; 2]) -> bool {
        (a[0] - b[0]).abs() < 1e-3 && (a[1] - b[1]).abs() < 1e-3
    }

    /// No shear and no taper is the plain affine, bit for bit: every older
    /// setup renders what it always did.
    /// A leaf whose texture is a smaller copy of the plate scales the
    /// position into it, from a uniform: the size changes no shader.
    #[test]
    fn a_scaled_reader_samples_through_its_slot() {
        let mut g = Graph::default_poc();
        let read = *g.nodes.iter().find(|(_, k)| matches!(k, NodeKind::Read { .. })).unwrap().0;
        g.reader_windows.insert((read, TIME_KEY_ROOT), [3.0, 4.0, 0.5, 0.25]);
        let plan = generate(&g);
        let slot = plan.layout.window_slot[&(read, TIME_KEY_ROOT)];
        assert!(plan.main.wgsl.contains(&format!("* P.v[{slot}].zw - P.v[{slot}].xy")));
        let v = pack_params(&g, &plan.layout, [64.0, 64.0]);
        assert_eq!(v[slot * 4..slot * 4 + 4], [3.0, 4.0, 0.5, 0.25]);
        g.reader_windows.clear();
        let v = pack_params(&g, &plan.layout, [64.0, 64.0]);
        assert_eq!(v[slot * 4..slot * 4 + 4], [0.0, 0.0, 1.0, 1.0], "a stale layout samples the plate");
    }

    #[test]
    fn an_unbent_transform_is_the_affine_it_always_was() {
        let t = crate::graph::TransformParams { rotate: 30.0, scale: [1.5, 0.5], ..Default::default() };
        let [m00, m01, tx, m10, m11, ty] = forward_affine(&t, [200.0, 100.0]);
        assert_eq!(transform_forward_px(&t, [200.0, 100.0]), [[m00, m01, tx], [m10, m11, ty], [0.0, 0.0, 1.0]]);
    }

    #[test]
    fn shear_moves_x_by_y_about_the_pivot() {
        let t = crate::graph::TransformParams { shear: [0.5, 0.0], ..Default::default() };
        let h = transform_forward_px(&t, [200.0, 100.0]);
        // The pivot stays; 20 px below it moves 10 px right.
        assert!(near(apply(&h, [100.0, 50.0]), [100.0, 50.0]));
        assert!(near(apply(&h, [100.0, 70.0]), [110.0, 70.0]));
    }

    /// The default axis (up) narrows the TOP to 1/(1+t) and widens the
    /// bottom to 1/(1-t); the top and bottom edges stay where they were.
    #[test]
    fn taper_narrows_the_end_its_axis_points_to() {
        let t = crate::graph::TransformParams { taper: 0.25, ..Default::default() };
        let (w, hh) = (200.0, 100.0);
        let h = transform_forward_px(&t, [w, hh]);
        let (tl, tr) = (apply(&h, [0.0, 0.0]), apply(&h, [w, 0.0]));
        let (bl, br) = (apply(&h, [0.0, hh]), apply(&h, [w, hh]));
        assert!((tl[1]).abs() < 1e-3 && (bl[1] - hh).abs() < 1e-3, "ends move: {tl:?} {bl:?}");
        assert!(((tr[0] - tl[0]) - w / 1.25).abs() < 1e-2, "top {tl:?} {tr:?}");
        assert!(((br[0] - bl[0]) - w / 0.75).abs() < 1e-2, "bottom {bl:?} {br:?}");
        // Still centred on the pivot's column.
        assert!(((tl[0] + tr[0]) * 0.5 - 100.0).abs() < 1e-3);
    }

    #[test]
    fn default_graph_generates() {
        let g = Graph::default_poc();
        let plan = generate(&g);
        assert!(plan.main.wgsl.contains("@compute"));
        assert!(plan.blurs.is_empty());
        assert_eq!(
            plan.main.sources,
            vec![SourceBinding::Reader(0), SourceBinding::Reader(1)]
        );
        assert_eq!(plan.layout.node_slot.len(), 3); // transform + grade + blend mix
        let params = pack_params(&g, &plan.layout, [1920.0, 1080.0]);
        assert_eq!(params.len(), plan.layout.array_len() * 4);
    }

    /// Read(0) -> view, with a declared input colour space.
    fn idt_graph(input: crate::color::InputSpace, working: crate::color::Colorspace) -> Graph {
        let mut g = Graph::default();
        g.working = working;
        let mut read = NodeKind::read(0, "plate");
        if let NodeKind::Read { input: i, .. } = &mut read {
            *i = input;
        }
        g.nodes.insert(0, read);
        g.view = Some(0);
        g
    }

    /// A plate declared as something other than the working space gets a
    /// conversion compiled in at the point it is sampled.
    #[test]
    fn a_declared_input_space_emits_an_input_transform() {
        use crate::color::{Colorspace, InputSpace};
        let g = idt_graph(InputSpace::Explicit(Colorspace::SRGB), Colorspace::ACESCG);
        let src = generate(&g).main.wgsl;
        assert!(src.contains("cs_srgb_to_lin"), "sRGB curve missing:\n{src}");
        assert!(src.contains("cs_idt_"), "input transform missing:\n{src}");
        assert!(src.contains("mat3x3<f32>"), "gamut matrix missing:\n{src}");
        // WGSL wants a function declared before it is called.
        assert!(
            src.find("fn cs_srgb_to_lin").unwrap() < src.find("fn cs_idt_").unwrap(),
            "curve helper must precede the transform that calls it:\n{src}"
        );
        validate_wgsl("srgb idt", &src);
    }

    /// The default costs nothing: a plate already in the working space
    /// compiles to exactly the shader it compiled to before colour
    /// management existed.
    #[test]
    fn a_working_space_plate_emits_no_transform() {
        use crate::color::{Colorspace, InputSpace};
        let g = idt_graph(InputSpace::Working, Colorspace::ACESCG);
        let src = generate(&g).main.wgsl;
        assert!(!src.contains("cs_idt_"), "identity should emit nothing:\n{src}");
        assert!(!src.contains("cs_srgb_to_lin"));
        validate_wgsl("no idt", &src);
    }

    /// Declaring ACEScg while working in ACEScg is the same no-op as
    /// declaring nothing — the transform is identity, not a wasted matrix.
    #[test]
    fn declaring_the_working_space_explicitly_is_still_a_no_op() {
        use crate::color::{Colorspace, InputSpace};
        let g = idt_graph(InputSpace::Explicit(Colorspace::ACESCG), Colorspace::ACESCG);
        assert!(!generate(&g).main.wgsl.contains("cs_idt_"));
    }

    /// Every preset a user can pick must compile, in both working spaces we
    /// expect people to run.
    #[test]
    fn every_input_preset_compiles_in_the_shader() {
        use crate::color::{Colorspace, InputSpace};
        for working in [Colorspace::ACESCG, Colorspace::LINEAR_REC709] {
            for (name, cs) in Colorspace::PRESETS {
                let g = idt_graph(InputSpace::Explicit(cs), working);
                validate_wgsl(name, &generate(&g).main.wgsl);
            }
        }
    }

    /// Two Reads declaring the same space share one emitted function, and a
    /// third declaring a different one adds exactly one more.
    #[test]
    fn input_transforms_are_shared_between_reads() {
        use crate::color::{Colorspace, InputSpace};
        let mut g = Graph::default();
        g.working = Colorspace::ACESCG;
        for (id, cs) in [
            (0u64, Colorspace::SRGB),
            (1, Colorspace::SRGB),
            (2, Colorspace::SLOG3),
        ] {
            let mut r = NodeKind::read(id as usize, "plate");
            if let NodeKind::Read { input, .. } = &mut r {
                *input = InputSpace::Explicit(cs);
            }
            g.nodes.insert(id, r);
        }
        g.nodes.insert(10, NodeKind::blend());
        g.nodes.insert(11, NodeKind::blend());
        g.edges.insert((10, 0), 0);
        g.edges.insert((10, 1), 1);
        g.edges.insert((11, 0), 10);
        g.edges.insert((11, 1), 2);
        g.view = Some(11);
        let src = generate(&g).main.wgsl;
        assert_eq!(src.matches("fn cs_idt_").count(), 2, "one per SPACE:\n{src}");
        assert_eq!(src.matches("fn cs_srgb_to_lin").count(), 1);
        validate_wgsl("shared idts", &src);
    }

    /// The working space is part of the shader, so changing it must move the
    /// topology hash — otherwise a cached pipeline outlives its own matrix.
    #[test]
    fn working_space_moves_the_topology_hash() {
        use crate::color::{Colorspace, InputSpace};
        let a = idt_graph(InputSpace::Explicit(Colorspace::SRGB), Colorspace::ACESCG);
        let b = idt_graph(InputSpace::Explicit(Colorspace::SRGB), Colorspace::LINEAR_REC709);
        assert_ne!(a.topology_hash(), b.topology_hash());
        assert_ne!(generate(&a).main.wgsl, generate(&b).main.wgsl);
    }

    /// One Noise masking two Mults in a chain: fusion inlines the Noise
    /// twice (once inside the first Mult, once for the second's mask), and
    /// its field functions must still be declared ONCE — a second copy is a
    /// redefinition naga rejects, and the app kept the last good shader, so
    /// the second Mult silently did nothing.
    #[test]
    fn a_noise_feeding_two_consumers_declares_its_functions_once() {
        use crate::graph::{Graph, MultParams, NodeKind};
        let mut g = Graph::default();
        g.nodes.insert(1, NodeKind::read(0, ""));
        let mut noise = crate::graph::NoiseParams::default();
        noise.graph.nodes.insert(1, crate::field::FieldNode::default());
        g.nodes.insert(2, NodeKind::Noise(noise));
        g.nodes.insert(3, NodeKind::ToMatte(Default::default()));
        g.edges.insert((3, 0), 2);
        let m = || NodeKind::Mult(MultParams { value: [4.0; 3], ..Default::default() });
        g.nodes.insert(4, m());
        g.edges.insert((4, 0), 1);
        g.edges.insert((4, 1), 3);
        g.nodes.insert(5, m());
        g.edges.insert((5, 0), 4);
        g.edges.insert((5, 1), 3);
        g.view = Some(5);
        let src = generate(&g).main.wgsl;
        assert_eq!(src.matches("fn nz2_1(").count(), 1, "declared once:\n{src}");
        validate_wgsl("noise x2", &src);
    }

    fn validate_wgsl(name: &str, src: &str) {
        let module = naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("{name} parse: {e}\n{src}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name} validate: {e:?}"));
    }

    #[test]
    fn glow_pass_shaders_validate() {
        validate_wgsl("glow::DOWN_WGSL", crate::nodes::glow::DOWN_WGSL);
        validate_wgsl("glow::UP_WGSL", crate::nodes::glow::UP_WGSL);
        validate_wgsl("glow::COMP_WGSL", crate::nodes::glow::COMP_WGSL);
        validate_wgsl("godrays::EXTRACT_WGSL", crate::nodes::godrays::EXTRACT_WGSL);
        validate_wgsl("godrays::EXTRACT_MATTE_WGSL", crate::nodes::godrays::EXTRACT_MATTE_WGSL);
        validate_wgsl("godrays::RADIAL_WGSL", crate::nodes::godrays::RADIAL_WGSL);
        validate_wgsl("glint::EXTRACT_WGSL", crate::nodes::glint::EXTRACT_WGSL);
        validate_wgsl("glint::STREAK_WGSL", crate::nodes::glint::STREAK_WGSL);
        validate_wgsl("diffuse::PREP_WGSL", crate::nodes::diffuse::PREP_WGSL);
        validate_wgsl("diffuse::PREP_MOD_WGSL", crate::nodes::diffuse::PREP_MOD_WGSL);
        validate_wgsl("diffuse::PREP_BOTH_WGSL", crate::nodes::diffuse::PREP_BOTH_WGSL);
        validate_wgsl("diffuse::DOWN_WGSL", crate::nodes::diffuse::DOWN_WGSL);
        validate_wgsl("diffuse::UP_WGSL", crate::nodes::diffuse::UP_WGSL);
        validate_wgsl("diffuse::COMP_WGSL", crate::nodes::diffuse::COMP_WGSL);
        validate_wgsl("blur::WGSL", crate::nodes::blur::WGSL);
        validate_wgsl("dilate::WGSL", crate::nodes::dilate::WGSL);
    }

    #[test]
    fn glow_splits_into_prepass() {
        // Same split as Blur: a glow between grade(4) and blend.+(5).
        let mut g = Graph::default_poc();
        g.nodes.insert(7, NodeKind::Glow(crate::graph::GlowParams::default()));
        g.edges.insert((7, 0), 4);
        g.edges.insert((5, 0), 7);
        let plan = generate(&g);
        assert_eq!(plan.blurs.len(), 1);
        let (stage, kind, sub, _) = &plan.blurs[0];
        assert_eq!(*stage, (7, TIME_KEY_ROOT));
        assert_eq!(*kind, FilterKind::Node);
        validate_wgsl("glow-subtree", &sub.wgsl);
        validate_wgsl("glow-main", &plan.main.wgsl);
        assert!(plan.main.sources.contains(&SourceBinding::Blur((7, TIME_KEY_ROOT))));
    }

    #[test]
    fn defocus_pass_shaders_validate() {
        validate_wgsl(
            "DEFOCUS_PREFIX_WGSL",
            &format!("{DEFOCUS_COMMON_WGSL}{DEFOCUS_PREFIX_WGSL}"),
        );
        validate_wgsl("DEFOCUS_COC_WGSL", &format!("{DEFOCUS_COMMON_WGSL}{DEFOCUS_COC_WGSL}"));
        validate_wgsl("DEFOCUS_TILE_WGSL", DEFOCUS_TILE_WGSL);
        validate_wgsl(
            "DEFOCUS_TILEMAX_WGSL",
            &format!("{DEFOCUS_COMMON_WGSL}{}", DEFOCUS_TILEMAX_WGSL.replace("DIR", "1, 0")),
        );
        // Both defocus passes are compiled with the shared prelude spliced
        // in front, so that is what has to parse.
        validate_wgsl(
            "DEFOCUS_DISC_WGSL",
            &format!("{DEFOCUS_COMMON_WGSL}{DEFOCUS_DISC_WGSL}"),
        );
        validate_wgsl(
            "DEFOCUS_BOKEH_WGSL",
            &format!("{DEFOCUS_COMMON_WGSL}{DEFOCUS_BOKEH_WGSL}"),
        );
    }

    /// The depth port is a full image input: an arbitrary branch resolves
    /// into its own texture through the plan's aux slot, and nested filters
    /// in that branch still get their own stages.
    #[test]
    fn defocus_depth_branch_gets_an_aux_shader() {
        let mut g = Graph::default_poc();
        g.nodes.insert(7, NodeKind::Defocus(crate::graph::DefocusParams::default()));
        g.edges.insert((7, 0), 4);
        g.edges.insert((5, 0), 7);
        // No depth wired: no aux pass, and the stage says so.
        let plan = generate(&g);
        assert!(plan.blurs[0].3.is_empty());
        assert!(matches!(plan.blurs[0].1, FilterKind::Defocus { depth: false, .. }));

        // Depth wired through a Blur, which must get a stage of its own.
        g.nodes.insert(8, NodeKind::Blur(crate::graph::BlurParams { size: 2.0, matte: false }));
        g.edges.insert((8, 0), 4);
        g.edges.insert((7, 2), 8);
        let plan = generate(&g);
        let defocus = plan.blurs.iter().find(|b| b.0 .0 == 7).expect("defocus stage");
        assert!(matches!(defocus.1, FilterKind::Defocus { depth: true, .. }));
        let aux = defocus.3.first().expect("aux shader for the depth branch");
        validate_wgsl("defocus-aux", &aux.wgsl);
        assert!(
            plan.blurs.iter().any(|b| b.0 .0 == 8),
            "a filter inside the depth branch needs its own stage"
        );
    }

    /// Mode is the only defocus control that may change the pass chain.
    #[test]
    fn defocus_mode_recompiles_and_params_do_not() {
        let mut g = Graph::default_poc();
        let base = crate::graph::DefocusParams::default();
        g.nodes.insert(7, NodeKind::Defocus(base));
        g.edges.insert((7, 0), 4);
        g.edges.insert((5, 0), 7);
        let h0 = g.topology_hash();
        let mut wide = base;
        wide.radius = 120.0;
        wide.f_stop = 1.4;
        wide.blades = 6;
        wide.samples = 512;
        g.nodes.insert(7, NodeKind::Defocus(wide));
        assert_eq!(g.topology_hash(), h0, "defocus params must be uniforms");
        let mut slow = base;
        slow.mode = crate::graph::DefocusMode::Accurate;
        g.nodes.insert(7, NodeKind::Defocus(slow));
        assert_ne!(g.topology_hash(), h0, "mode picks the pass chain");
    }

    #[test]
    fn graphbake_baked_prunes_subtree_and_samples_reader() {
        // read A -> Blur(7) -> Bake(8, baked) as view target.
        let mut g = Graph::default_poc();
        g.nodes.insert(7, NodeKind::Blur(crate::graph::BlurParams::default()));
        g.nodes.insert(8, NodeKind::GraphBake { source: 3, baked: true, start: None, mask: false, window: None, matte_half: false, single: false });
        g.edges.insert((7, 0), 1);
        g.edges.insert((8, 0), 7);
        g.view = Some(8);
        let plan = generate(&g);
        assert!(plan.blurs.is_empty(), "baked bake must prune upstream stages");
        assert_eq!(plan.main.sources, vec![SourceBinding::Reader(3)]);
        validate_wgsl("graphbake-baked", &plan.main.wgsl);

        // Live: transparent passthrough, and the upstream stage is back.
        g.nodes.insert(8, NodeKind::GraphBake { source: 3, baked: false, start: None, mask: false, window: None, matte_half: false, single: false });
        let plan = generate(&g);
        assert_eq!(plan.blurs.len(), 1, "live bake must evaluate the subtree");
        assert_eq!(plan.main.sources, vec![SourceBinding::Blur((7, TIME_KEY_ROOT))]);
        validate_wgsl("graphbake-live", &plan.main.wgsl);
        // live vs baked is a different shader: the hash must move.
        let h_live = g.topology_hash();
        g.nodes.insert(8, NodeKind::GraphBake { source: 3, baked: true, start: None, mask: false, window: None, matte_half: false, single: false });
        assert_ne!(g.topology_hash(), h_live);
    }

    #[test]
    fn blur_splits_into_prepass() {
        // Insert a blur between grade(4) and blend.+(5).
        let mut g = Graph::default_poc();
        g.nodes.insert(7, NodeKind::Blur(crate::graph::BlurParams::default()));
        g.edges.insert((7, 0), 4); // blur <- grade
        g.edges.insert((5, 0), 7); // blend.+ <- blur
        let plan = generate(&g);
        assert_eq!(plan.blurs.len(), 1);
        let (bnode, kind, sub, _) = &plan.blurs[0];
        assert_eq!(*bnode, (7, TIME_KEY_ROOT));
        assert_eq!(*kind, FilterKind::Node);
        // Subtree sees only read A (through transform+grade).
        assert_eq!(sub.sources, vec![SourceBinding::Reader(0)]);
        assert!(!sub.wgsl.contains("view_tf"), "nothing tonemaps in the fused pass");
        // Main samples the blur output + read B.
        assert_eq!(
            plan.main.sources,
            vec![SourceBinding::Blur((7, TIME_KEY_ROOT)), SourceBinding::Reader(1)]
        );
        assert!(
            !plan.main.wgsl.contains("view_tf"),
            "main pass is scene-linear too; the blit transforms"
        );
    }

    #[test]
    fn nested_blur_orders_dependencies() {
        // read A -> blur 7 -> blur 8 -> viewer
        let mut g = Graph::default_poc();
        g.nodes.insert(7, NodeKind::Blur(crate::graph::BlurParams::default()));
        g.nodes.insert(8, NodeKind::Blur(crate::graph::BlurParams::default()));
        g.edges.insert((7, 0), 1);
        g.edges.insert((8, 0), 7);
        g.view = Some(8);
        let plan = generate(&g);
        assert_eq!(plan.blurs.len(), 2);
        assert_eq!(plan.blurs[0].0, (7, TIME_KEY_ROOT), "inner blur must compile first");
        assert_eq!(plan.blurs[1].0, (8, TIME_KEY_ROOT));
        assert_eq!(plan.blurs[1].2.sources, vec![SourceBinding::Blur((7, TIME_KEY_ROOT))]);
        assert_eq!(plan.main.sources, vec![SourceBinding::Blur((8, TIME_KEY_ROOT))]);
    }

    #[test]
    fn self_referential_blur_is_black_not_conflicting() {
        // blur wired to itself must not bind its own output in its subtree.
        let mut g = Graph::default_poc();
        g.nodes.insert(7, NodeKind::Blur(crate::graph::BlurParams::default()));
        g.edges.insert((7, 0), 7);
        g.view = Some(7);
        let plan = generate(&g);
        let (_, _, sub, _) = &plan.blurs[0];
        assert!(sub.sources.is_empty());
    }

    #[test]
    fn disabled_node_bypasses() {
        let mut g = Graph::default_poc();
        let h1 = g.topology_hash();
        g.disabled.insert(4); // grade
        assert_ne!(g.topology_hash(), h1, "disable must change topology");
        let plan = generate(&g);
        assert!(!plan.main.wgsl.contains("= grade4("), "bypassed grade still called");
        // Disabled blur produces no filter stage.
        let mut g2 = Graph::default_poc();
        g2.nodes.insert(7, NodeKind::Blur(crate::graph::BlurParams::default()));
        g2.edges.insert((7, 0), 4);
        g2.edges.insert((5, 0), 7);
        g2.disabled.insert(7);
        let plan2 = generate(&g2);
        assert!(plan2.blurs.is_empty());
        assert!(plan2.main.wgsl.contains("= grade4("), "blur bypass must pass grade through");
    }

    /// A filter under a Layer is found on the LAYER's time branch, the key
    /// `emit` binds it by. Found on the stack's own key, the binding missed
    /// and the layer sampled the dummy: a Defocus on the top layer vanished.
    #[test]
    fn a_stage_under_a_layer_is_the_one_bound() {
        use crate::graph::{LayerItem, LayerParams};
        let mut g = Graph::default();
        g.nodes.insert(1, NodeKind::read(0, ""));
        g.nodes.insert(2, NodeKind::Defocus(crate::graph::DefocusParams::default()));
        g.nodes.insert(3, NodeKind::Blur(crate::graph::BlurParams::default()));
        g.nodes.insert(
            4,
            NodeKind::Layer(LayerParams { layers: vec![LayerItem::default(); 3], outputs: false }),
        );
        g.edges.insert((2, 0), 1);
        g.edges.insert((3, 0), 1);
        g.edges.insert((4, 0), 2);
        g.edges.insert((4, 1), 3);
        g.edges.insert((4, 2), 1);
        g.view = Some(4);
        let plan = generate(&g);
        assert_eq!(plan.blurs.len(), 2);
        for (stage, ..) in &plan.blurs {
            assert!(
                plan.main.sources.contains(&SourceBinding::Blur(*stage)),
                "stage {stage:?} is not what the layer samples: {:?}",
                plan.main.sources
            );
        }
    }

    /// A bypassed Blend is "the merge is off": the BACKGROUND stays and the
    /// foreground branch is not even emitted.
    #[test]
    fn disabled_blend_passes_the_background() {
        let mut g = Graph::default_poc();
        g.disabled.insert(5); // blend: fg = grade <- read A, bg = read B
        let plan = generate(&g);
        assert!(!plan.main.wgsl.contains("= grade4("), "bypassed blend kept the fg branch");
        assert!(!plan.main.wgsl.contains("= over("), "bypassed blend still merged");
        g.disabled.clear();
        assert!(generate(&g).main.wgsl.contains("= over("), "the probe must see a live merge");
        assert!(plan.main.wgsl.contains("sample_px("), "bypassed blend lost the bg");
    }

    #[test]
    fn defocus_is_a_disc_stage() {
        let mut g = Graph::default_poc();
        g.nodes.insert(7, NodeKind::Defocus(crate::graph::DefocusParams::default()));
        g.edges.insert((7, 0), 4);
        g.edges.insert((5, 0), 7);
        let plan = generate(&g);
        assert_eq!(plan.blurs.len(), 1);
        assert!(matches!(
            plan.blurs[0].1,
            FilterKind::Defocus {
                mode: crate::graph::DefocusMode::Fast,
                depth: false,
                modulate: false,
                kernel: None,
                kernel_aux: false,
            }
        ));
        assert_eq!(plan.main.sources[0], SourceBinding::Blur((7, TIME_KEY_ROOT)));
    }

    #[test]
    fn topology_hash_ignores_params() {
        let g1 = Graph::default_poc();
        let mut g2 = Graph::default_poc();
        if let Some(NodeKind::Grade(gr)) = g2.nodes.get_mut(&4) {
            gr.gain = [2.0, 2.0, 2.0];
        }
        assert_eq!(g1.topology_hash(), g2.topology_hash());
        g2.edges.remove(&(5, 1));
        assert_ne!(g1.topology_hash(), g2.topology_hash());
    }

    #[test]
    fn cyclic_graph_terminates() {
        let mut g = Graph::default_poc();
        // grade -> blend.+ already exists; wire the blend back into grade's input.
        g.edges.insert((4, 0), 5);
        let plan = generate(&g); // must not stack-overflow
        assert!(plan.main.wgsl.contains("@compute"));
        // pack_params with a layout whose node was deleted must not panic.
        let mut g2 = Graph::default_poc();
        g2.nodes.remove(&4);
        let _ = pack_params(&g2, &plan.layout, [100.0, 100.0]);
    }

    #[test]
    fn view_target_is_structural() {
        let mut g = Graph::default_poc();
        let h1 = g.topology_hash();
        g.view = Some(4); // view the grade directly
        assert_ne!(g.topology_hash(), h1, "view change must recompile");
        let plan = generate(&g);
        // Rooted at the grade: read B never sampled.
        assert_eq!(plan.main.sources, vec![SourceBinding::Reader(0)]);
        // Dangling view renders black (no sources).
        g.view = Some(999);
        let plan2 = generate(&g);
        assert!(plan2.main.sources.is_empty());
    }

    #[test]
    fn corner_pin_identity_roundtrip() {
        let cp = crate::graph::CornerPinParams::default();
        let m = corner_pin_inverse(&cp, [200.0, 100.0]);
        // Identity corners: output pixel maps to itself.
        let p = [37.0f32, 81.0];
        let w = m[2][0] * p[0] + m[2][1] * p[1] + m[2][2];
        let x = (m[0][0] * p[0] + m[0][1] * p[1] + m[0][2]) / w;
        let y = (m[1][0] * p[0] + m[1][1] * p[1] + m[1][2]) / w;
        assert!((x - p[0]).abs() < 1e-2 && (y - p[1]).abs() < 1e-2, "{x} {y}");
        // A node with a pin gets 3 uniform slots and a persp() call.
        let mut g = Graph::default_poc();
        g.nodes.insert(7, NodeKind::CornerPin(cp));
        g.edges.insert((7, 0), 4);
        g.edges.insert((5, 0), 7);
        let plan = generate(&g);
        assert!(plan.main.wgsl.contains("= persp("));
    }

    #[test]
    fn identity_transform_inverts_to_identity() {
        let t = crate::graph::TransformParams::default();
        let m = inverse_affine(&t, [100.0, 100.0]);
        assert!((m[0] - 1.0).abs() < 1e-5 && (m[4] - 1.0).abs() < 1e-5);
        assert!(m[2].abs() < 1e-3 && m[5].abs() < 1e-3);
    }

    #[test]
    fn paint_layer_binds_and_composites() {
        let mut g = Graph::default_poc();
        g.nodes.insert(7, NodeKind::Paint(Default::default()));
        g.edges.insert((7, 0), 4); // paint <- grade
        g.edges.insert((5, 0), 7); // blend.+ <- paint
        let plan = generate(&g);
        assert_eq!(plan.paints, vec![7]);
        assert!(plan.main.sources.contains(&SourceBinding::Paint(7)));
        assert!(plan.main.wgsl.contains("= over("), "the layer composites over");
        // Disabled paint: passthrough, no layer bound.
        g.disabled.insert(7);
        let plan = generate(&g);
        assert!(plan.paints.is_empty());
        assert!(!plan.main.sources.contains(&SourceBinding::Paint(7)));
    }

    /// A clone stroke is the one thing a Paint node does that changes the
    /// SHAPE of the generated shader: its input is evaluated a second time,
    /// at an offset the clone layer carries. So it has to be topology (a
    /// recompile when the first one is drawn), the clone layer has to be
    /// bound, and the plain node must be untouched by all of it.
    #[test]
    fn a_clone_stroke_binds_a_second_layer_and_re_reads_the_input() {
        use crate::graph::{PaintParams, PaintStroke, StrokeLife};
        let paint = |strokes: Vec<PaintStroke>| {
            let mut g = Graph::default_poc();
            g.nodes.insert(
                7,
                NodeKind::Paint(PaintParams { strokes, ..Default::default() }),
            );
            g.edges.insert((7, 0), 4); // paint <- grade <- transform <- read
            g.edges.insert((5, 0), 7);
            g
        };
        let colour = PaintStroke::new(vec![[0.2, 0.2], [0.4, 0.4]], 10.0, 0.5, [1.0; 4]);
        let mut cloned = colour.clone();
        cloned.clone_off = Some([0.1, 0.0]);

        // Colour strokes only: exactly the old shader, one layer.
        let plain = generate(&paint(vec![colour.clone()]));
        assert!(plain.main.sources.contains(&SourceBinding::Paint(7)));
        assert!(!plain.main.sources.contains(&SourceBinding::PaintSource(7)));
        validate_wgsl("paint", &plain.main.wgsl);

        let cg = paint(vec![colour.clone(), cloned.clone()]);
        let clone = generate(&cg);
        assert!(clone.main.sources.contains(&SourceBinding::PaintSource(7)));
        validate_wgsl("paint-clone", &clone.main.wgsl);
        // The upstream chain really is emitted a second time: the clone
        // source is read live from above the node, not lifted off the pixel
        // that is already there. `= grade4(` counts uses, not the prelude's
        // definition.
        let evals = |w: &str| w.matches("= grade4(").count();
        assert_eq!(evals(&plain.main.wgsl), 1, "plain paint reads its input once");
        assert_eq!(
            evals(&clone.main.wgsl),
            2,
            "a cloning paint reads it twice — once under the stroke, once at \
             the offset"
        );
        // And the offset comes out of the clone layer, divided back out of
        // the premultiplied blend.
        assert!(
            clone.main.wgsl.contains(".xy / max(") && clone.main.wgsl.contains("mix("),
            "{}",
            clone.main.wgsl
        );

        // Drawing the first clone stroke recompiles; drawing more, moving
        // them, or changing their lifetimes never does.
        let h = paint(vec![colour.clone()]).topology_hash();
        assert_ne!(cg.topology_hash(), h, "the first clone stroke recompiles");
        let mut more = cloned.clone();
        more.clone_off = Some([-0.3, 0.2]);
        more.life = StrokeLife::Single(12);
        more.points.push([0.9, 0.9]);
        assert_eq!(
            paint(vec![colour.clone(), cloned.clone(), more]).topology_hash(),
            cg.topology_hash(),
            "more clone strokes are just more stamps"
        );
        // ...and neither does anything at all about a colour stroke.
        let mut moved = colour.clone();
        moved.points = vec![[0.7, 0.1]];
        moved.radius = 99.0;
        moved.life = StrokeLife::Range(2, 8);
        assert_eq!(paint(vec![moved]).topology_hash(), h, "painting never recompiles");
    }

    /// Every brush class compiles, binds exactly the layers it needs, and
    /// re-reads the input exactly as often as its cost tier says: colour /
    /// dodge / burn never, clone / smear once, blur PAINT_BLUR_TAPS times.
    #[test]
    fn each_brush_class_reads_the_input_as_often_as_it_says() {
        use crate::graph::{BrushMode, PaintOutput, PaintParams, PaintStroke};
        use crate::paint::PAINT_BLUR_TAPS;
        let stroke = |mode: BrushMode| {
            let mut s = PaintStroke::new(vec![[0.2, 0.2], [0.4, 0.4]], 10.0, 0.5, [1.0; 4]);
            s.mode = mode;
            s.strength = 8.0;
            if mode == BrushMode::Clone {
                s.clone_off = Some([0.1, 0.0]);
            }
            s
        };
        let plan_for = |strokes: Vec<PaintStroke>, output: PaintOutput| {
            let mut g = Graph::default_poc();
            g.nodes.insert(
                7,
                NodeKind::Paint(PaintParams { strokes, output, ..Default::default() }),
            );
            g.edges.insert((7, 0), 4); // paint <- grade <- transform <- read
            g.edges.insert((5, 0), 7);
            generate(&g)
        };
        // `= grade4(` counts USES of the upstream chain, not the prelude's
        // definition of it.
        let evals = |w: &str| w.matches("= grade4(").count();
        for (mode, extra) in [
            (BrushMode::Color, 0),
            (BrushMode::Dodge, 0),
            (BrushMode::Burn, 0),
            (BrushMode::Clone, 1),
            (BrushMode::Smear, 1),
            (BrushMode::Blur, PAINT_BLUR_TAPS),
        ] {
            let plan = plan_for(vec![stroke(mode)], PaintOutput::Rgb);
            validate_wgsl(&format!("paint-{}", mode.name()), &plan.main.wgsl);
            assert_eq!(
                evals(&plan.main.wgsl),
                1 + extra,
                "{}: input reads",
                mode.name()
            );
            let src = plan.main.sources.contains(&SourceBinding::PaintSource(7));
            let eff = plan.main.sources.contains(&SourceBinding::PaintEffect(7));
            assert_eq!(src, matches!(mode, BrushMode::Clone | BrushMode::Smear), "{mode:?} source layer");
            assert_eq!(
                eff,
                matches!(mode, BrushMode::Blur | BrushMode::Dodge | BrushMode::Burn),
                "{mode:?} effect layer"
            );
            assert!(plan.main.sources.contains(&SourceBinding::Paint(7)));
        }
        // All three classes at once: every layer bound, reads add up.
        let all = plan_for(
            vec![stroke(BrushMode::Clone), stroke(BrushMode::Blur), stroke(BrushMode::Dodge)],
            PaintOutput::Rgb,
        );
        validate_wgsl("paint-all", &all.main.wgsl);
        assert_eq!(evals(&all.main.wgsl), 1 + 1 + PAINT_BLUR_TAPS);
        assert!(all.main.wgsl.contains("exp2("), "dodge is an exposure");

        // A clone's source is the PAINTED picture at the offset: the colour
        // layer sampled there and composited over the input there — so a
        // colour stroke under the source shows up in the clone. Still one
        // extra read of the INPUT; the layer is a texture.
        let clone = plan_for(vec![stroke(BrushMode::Clone)], PaintOutput::Rgb);
        let w = &clone.main.wgsl;
        let src_idx = clone.main.sources.iter().position(|b| *b == SourceBinding::Paint(7)).unwrap();
        assert_eq!(
            w.matches(&format!("sample_px(src{src_idx}, ")).count(),
            2,
            "the colour layer is read under the brush AND under the source"
        );
        assert_eq!(evals(w), 2);
        // ...and a dodge under the source is applied there too, without a
        // third read of the input.
        let both = plan_for(vec![stroke(BrushMode::Clone), stroke(BrushMode::Dodge)], PaintOutput::Rgb);
        validate_wgsl("paint-clone-dodge", &both.main.wgsl);
        assert_eq!(evals(&both.main.wgsl), 2);
        assert_eq!(both.main.wgsl.matches("exp2(").count(), 2, "exposure under the brush and under the source");

        // MATTE output: the layers are read for their coverage only, so
        // none of the input re-reads exist — a matte painted with a blur
        // brush costs what a colour one does.
        let matte = plan_for(
            vec![stroke(BrushMode::Clone), stroke(BrushMode::Blur), stroke(BrushMode::Dodge)],
            PaintOutput::Matte,
        );
        validate_wgsl("paint-matte", &matte.main.wgsl);
        assert_eq!(evals(&matte.main.wgsl), 1, "matte output never re-reads the input");
        assert!(matte.main.sources.contains(&SourceBinding::PaintSource(7)));
        assert!(matte.main.sources.contains(&SourceBinding::PaintEffect(7)));
        assert!(!matte.main.wgsl.contains("exp2("));
    }
}

#[cfg(test)]
mod node_pack_tests {
    use super::*;
    use crate::graph::*;

    fn validate_wgsl(name: &str, src: &str) {
        let module = naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("{name} parse: {e}\n{src}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name} validate: {e:?}"));
    }

    /// Read A -> node 7 -> view.
    fn one_node(kind: NodeKind) -> Graph {
        let mut g = Graph::default_poc();
        g.nodes.insert(7, kind);
        g.edges.insert((7, 0), 1);
        g.view = Some(7);
        g
    }

    #[test]
    fn every_new_node_compiles_to_valid_wgsl() {
        for kind in [
            NodeKind::Cryptomatte(Default::default()),
            NodeKind::Grid(GridParams::default()),
            NodeKind::Blank(Default::default()),
            NodeKind::Starfield(Default::default()),
            NodeKind::Planet(Default::default()),
            NodeKind::Planet(crate::graph::PlanetParams { mode: crate::graph::PlanetMode::Latlong, ..Default::default() }),
            NodeKind::Plasma(Default::default()),
            NodeKind::Splat(Default::default()),
            NodeKind::Splat(crate::graph::SplatParams { depth_of: Some(1), ..Default::default() }),
            NodeKind::LensFlare(Default::default()),
            NodeKind::Contrast(Default::default()),
            NodeKind::LensFlare(crate::graph::LensFlareParams { occlude: true, ..Default::default() }),
            NodeKind::GlintKernel(Default::default()),
            NodeKind::BokehKernel(Default::default()),
            NodeKind::Grain(Default::default()),
            NodeKind::Gradient(Default::default()),
            NodeKind::Gradient(crate::graph::GradientParams {
                interp: crate::graph::GradInterp::Lch,
                ..Default::default()
            }),
            NodeKind::Reformat(ReformatParams::default()),
            NodeKind::Tracker(TrackerParams::default()),
            NodeKind::Slip(SlipParams::default()),
            NodeKind::Speed(SpeedParams::default()),
            NodeKind::TimeRemap(TimeRemapParams::default()),
            NodeKind::Premult { premult: false },
            NodeKind::Premult { premult: true },
        ] {
            let name = kind.name();
            let g = one_node(kind);
            let plan = generate(&g);
            validate_wgsl(name, &plan.main.wgsl);
            assert!(plan.blurs.is_empty(), "{name} must stay fused");
            let params = pack_params(&g, &plan.layout, [1920.0, 1080.0]);
            assert_eq!(params.len(), plan.layout.array_len() * 4);
            assert!(params.iter().all(|v| v.is_finite()), "{name} packed a NaN");
        }
    }

    #[test]
    fn passthrough_nodes_cost_nothing() {
        // Time nodes, the tracker and premult take no uniform slots, so the
        // fused pass is exactly the upstream one plus (for premult) ALU.
        for kind in [
            NodeKind::Slip(SlipParams { delay: -12 }),
            NodeKind::Speed(SpeedParams { speed: 0.5, anchor: 3.0, ..Default::default() }),
            NodeKind::Tracker(TrackerParams::default()),
            NodeKind::Premult { premult: false },
        ] {
            let g = one_node(kind.clone());
            let plan = generate(&g);
            assert!(
                !plan.layout.node_slot.contains_key(&7),
                "{} should take no slots",
                kind.name()
            );
        }
        // A time param edit is not a recompile.
        let a = one_node(NodeKind::Slip(SlipParams { delay: 0 }));
        let b = one_node(NodeKind::Slip(SlipParams { delay: -40 }));
        assert_eq!(a.topology_hash(), b.topology_hash());
        // But unpremult and premult are different formulas.
        let un = one_node(NodeKind::Premult { premult: false });
        let pre = one_node(NodeKind::Premult { premult: true });
        assert_ne!(un.topology_hash(), pre.topology_hash());
        assert!(generate(&un).main.wgsl.contains(".rgb / "));
        assert!(generate(&pre).main.wgsl.contains(".rgb * "));
    }

    #[test]
    fn reformat_fits_the_source_into_the_new_format() {
        let src = [1000.0, 1000.0];
        let dst = [200.0, 100.0];
        let fit = |mode| {
            reformat_scale(
                &ReformatParams { size: (200, 100), mode, scale: 1.0, center: true, ..Default::default() },
                src,
                dst,
            )
        };
        // Fit takes the smaller ratio (letterbox), fill the larger (crop),
        // distort each axis on its own.
        assert_eq!(fit(ResizeMode::Fit), [0.1, 0.1]);
        assert_eq!(fit(ResizeMode::Fill), [0.2, 0.2]);
        assert_eq!(fit(ResizeMode::Distort), [0.2, 0.1]);
        assert_eq!(fit(ResizeMode::None), [1.0, 1.0]);

        // Packed as an output->source map: the centre of the new format must
        // land on the centre of the source.
        let mut g = one_node(NodeKind::Reformat(ReformatParams {
            size: (200, 100),
            mode: ResizeMode::Fit,
            scale: 1.0,
            center: true,
            ..Default::default()
        }));
        g.formats.insert(1, (1000, 1000));
        g.formats.insert(7, (200, 100));
        let plan = generate(&g);
        let v = pack_params(&g, &plan.layout, [200.0, 100.0]);
        let b = plan.layout.node_slot[&7] * 4;
        let map = |p: [f32; 2]| [p[0] * v[b] + v[b + 2], p[1] * v[b + 1] + v[b + 3]];
        let c = map([100.0, 50.0]);
        assert!((c[0] - 500.0).abs() < 1e-3 && (c[1] - 500.0).abs() < 1e-3, "{c:?}");
    }

    /// Sized by a factor: the output is the input's format times it, and
    /// the picture fills it whatever the mode says.
    #[test]
    fn reformat_by_factor_scales_the_input_format() {
        let rf = ReformatParams { by_factor: true, factor: 0.5, mode: ResizeMode::None, ..Default::default() };
        assert_eq!(rf.out_size((2048, 1080)), (1024, 540));
        assert_eq!(rf.out_size((1, 1)), (1, 1));
        assert_eq!(reformat_scale(&rf, [2048.0, 1080.0], [1024.0, 540.0]), [0.5, 0.5]);
        // Off, the absolute size stands.
        assert_eq!(ReformatParams::default().out_size((2048, 1080)), (1920, 1080));
        let mut g = one_node(NodeKind::Reformat(rf));
        g.formats.insert(1, (1000, 800));
        g.formats.insert(7, (500, 400));
        let plan = generate(&g);
        let v = pack_params(&g, &plan.layout, [500.0, 400.0]);
        let b = plan.layout.node_slot[&7] * 4;
        // Output px -> source px: a plain x2, no offset.
        assert_eq!([v[b], v[b + 1], v[b + 2], v[b + 3]], [2.0, 2.0, 0.0, 0.0]);
    }

    /// A file saved before the factor existed loads with it off.
    #[cfg(feature = "serde")]
    #[test]
    fn old_reformat_loads_without_a_factor() {
        let rf: ReformatParams =
            serde_json::from_str(r#"{"size":[1280,720],"mode":"Fit","scale":1.0,"center":true}"#).unwrap();
        assert!(!rf.by_factor);
        assert_eq!(rf.out_size((4096, 2160)), (1280, 720));
    }

    #[test]
    fn time_maps_compose_down_the_chain() {
        // Read(1) -> offset(7, +5) -> warp(8, x2) -> view.
        let mut g = Graph::default();
        g.nodes.insert(1, NodeKind::read(0, ""));
        g.nodes.insert(7, NodeKind::Slip(SlipParams { delay: -5 }));
        g.nodes.insert(8, NodeKind::Speed(SpeedParams { speed: 2.0, anchor: 0.0, ..Default::default() }));
        g.edges.insert((7, 0), 1);
        g.edges.insert((8, 0), 7);
        g.view = Some(8);
        let inst = g.reader_instances();
        // One source, one path: doubled, then shifted five SOURCE frames.
        assert_eq!(inst.len(), 1);
        assert_eq!(inst[0].node, 1);
        assert_eq!(inst[0].map.apply_f(10.0) - inst[0].map.apply_f(9.0), 2.0);
        assert_eq!(inst[0].map.apply(10), 25);
        assert_ne!(inst[0].key, TIME_KEY_ROOT, "it is on a retimed path");

        // Bypassing the offset takes it out of the chain (and off its path).
        g.disabled.insert(7);
        let bypassed = g.reader_instances();
        assert_eq!(bypassed[0].map.apply(10), 20);
        g.disabled.clear();
    }

    #[test]
    fn a_source_on_two_time_paths_becomes_two_readers() {
        // The same Read feeding a merge twice: once straight, once through a
        // +5 offset. Both frames are wanted at once, so both exist.
        let mut g = Graph::default();
        g.nodes.insert(1, NodeKind::read(0, ""));
        g.nodes.insert(7, NodeKind::Slip(SlipParams { delay: -5 }));
        g.nodes.insert(9, NodeKind::blend());
        g.edges.insert((7, 0), 1);
        g.edges.insert((9, 0), 7);
        g.edges.insert((9, 1), 1);
        g.view = Some(9);

        let inst = g.reader_instances();
        assert_eq!(inst.len(), 2, "one reader per time path: {inst:?}");
        let straight = inst.iter().find(|i| i.key == TIME_KEY_ROOT).unwrap();
        let shifted = inst.iter().find(|i| i.key != TIME_KEY_ROOT).unwrap();
        assert_eq!(straight.map.apply(10), 10);
        assert_eq!(shifted.map.apply(10), 15);

        // The app assigns a slot per instance; codegen then binds each leaf
        // to the slot its own path owns, so ONE shader samples both frames.
        g.time_slots.insert((1, straight.key), 0);
        g.time_slots.insert((1, shifted.key), 4);
        let plan = generate(&g);
        assert!(plan.main.sources.contains(&SourceBinding::Reader(0)));
        assert!(plan.main.sources.contains(&SourceBinding::Reader(4)));
        validate_wgsl("two-timebases", &plan.main.wgsl);

        // Dragging the offset must NOT recompile: only the frame each
        // reader decodes moves, and that is the player's business.
        let h = g.topology_hash();
        g.nodes.insert(7, NodeKind::Slip(SlipParams { delay: -99 }));
        assert_eq!(g.topology_hash(), h, "a time param edit is a uniform-free no-op");
        // Passing through zero is still its own branch of time.
        g.nodes.insert(7, NodeKind::Slip(SlipParams { delay: 0 }));
        assert_eq!(g.topology_hash(), h);
        assert_eq!(g.reader_instances().len(), 2);
        // But REMOVING the time node collapses the two paths into one.
        g.nodes.remove(&7);
        g.edges.insert((9, 0), 1);
        assert_eq!(g.reader_instances().len(), 1);
    }

    #[test]
    fn a_filter_upstream_of_two_timebases_is_two_stages() {
        // Read -> Blur -> { offset -> blend.+, blend.− }: the blur has to be
        // computed at both frames, so it gets an intermediate per path.
        let mut g = Graph::default();
        g.nodes.insert(1, NodeKind::read(0, ""));
        g.nodes.insert(2, NodeKind::Blur(crate::graph::BlurParams::default()));
        g.nodes.insert(7, NodeKind::Slip(SlipParams { delay: -5 }));
        g.nodes.insert(9, NodeKind::blend());
        g.edges.insert((2, 0), 1);
        g.edges.insert((7, 0), 2);
        g.edges.insert((9, 0), 7);
        g.edges.insert((9, 1), 2);
        g.view = Some(9);
        let inst = g.reader_instances();
        g.time_slots.insert((1, inst[0].key), 0);
        g.time_slots.insert((1, inst[1].key), 4);

        let plan = generate(&g);
        assert_eq!(plan.blurs.len(), 2, "one blur stage per timebase");
        let stages: Vec<StageId> = plan.blurs.iter().map(|(s, _, _, _)| *s).collect();
        assert!(stages.iter().all(|(n, _)| *n == 2));
        assert_ne!(stages[0].1, stages[1].1, "distinct time paths");
        // Each stage filters its OWN reader.
        let mut read_slots: Vec<&SourceBinding> =
            plan.blurs.iter().flat_map(|(_, _, sh, _)| sh.sources.iter()).collect();
        read_slots.sort_by_key(|b| format!("{b:?}"));
        assert_eq!(
            read_slots,
            vec![&SourceBinding::Reader(0), &SourceBinding::Reader(4)]
        );
        // And the main pass samples both intermediates.
        for s in stages {
            assert!(plan.main.sources.contains(&SourceBinding::Blur(s)));
        }
        validate_wgsl("two-stage-main", &plan.main.wgsl);
        for (_, _, sh, _) in &plan.blurs {
            validate_wgsl("two-stage-sub", &sh.wgsl);
        }
    }

    #[test]
    fn shuffle_rewires_channels_without_recompiling() {
        // bgra: colour reversed, alpha kept.
        let sh = ShuffleParams {
            out: [ShuffleSrc::B, ShuffleSrc::G, ShuffleSrc::R, ShuffleSrc::A],
        };
        let g = one_node(NodeKind::Shuffle(sh));
        let plan = generate(&g);
        validate_wgsl("shuffle", &plan.main.wgsl);
        let v = pack_params(&g, &plan.layout, [64.0, 64.0]);
        let b = plan.layout.node_slot[&7] * 4;

        // The packed matrix, applied on the CPU exactly as the shader does.
        let apply = |v: &[f32], b: usize, c: [f32; 4]| -> [f32; 4] {
            let mut out = [0.0f32; 4];
            for r in 0..4 {
                let row = &v[b + r * 4..b + r * 4 + 4];
                out[r] = (0..4).map(|i| row[i] * c[i]).sum::<f32>() + v[b + 16 + r];
            }
            out
        };
        assert_eq!(apply(&v, b, [0.1, 0.2, 0.3, 0.4]), [0.3, 0.2, 0.1, 0.4]);

        // Constants: alpha to 1, colour from alpha (the "look at the matte"
        // wiring), which no reordering alone could express.
        let matte = ShuffleParams {
            out: [ShuffleSrc::A, ShuffleSrc::A, ShuffleSrc::A, ShuffleSrc::One],
        };
        let g2 = one_node(NodeKind::Shuffle(matte));
        let plan2 = generate(&g2);
        let v2 = pack_params(&g2, &plan2.layout, [64.0, 64.0]);
        let b2 = plan2.layout.node_slot[&7] * 4;
        assert_eq!(apply(&v2, b2, [0.1, 0.2, 0.3, 0.75]), [0.75, 0.75, 0.75, 1.0]);
        // Zero really is zero, not a passthrough.
        let killed = ShuffleParams {
            out: [ShuffleSrc::R, ShuffleSrc::Zero, ShuffleSrc::Zero, ShuffleSrc::A],
        };
        let g3 = one_node(NodeKind::Shuffle(killed));
        let plan3 = generate(&g3);
        let v3 = pack_params(&g3, &plan3.layout, [64.0, 64.0]);
        let b3 = plan3.layout.node_slot[&7] * 4;
        assert_eq!(apply(&v3, b3, [0.5, 0.6, 0.7, 0.8]), [0.5, 0.0, 0.0, 0.8]);

        // The identity is the default, and rewiring is a UNIFORM edit: the
        // topology must not move when a channel is re-pointed.
        assert_eq!(ShuffleParams::default(), ShuffleParams::IDENTITY);
        assert_eq!(g.topology_hash(), g2.topology_hash());
        assert_eq!(g.topology_hash(), g3.topology_hash());
    }

    #[test]
    fn write_range_modes_resolve_and_old_setups_keep_their_meaning() {
        use crate::graph::WriteRange;
        let input = (1001, 1100);
        let global = (0, 99);
        let w = WriteParams::default();
        assert_eq!(w.mode(), WriteRange::Input);
        assert_eq!(w.resolve_range(input, global), input);
        let g = WriteParams { range_mode: Some(WriteRange::Global), ..w.clone() };
        assert_eq!(g.resolve_range(input, global), global);
        let r = WriteParams { range_mode: Some(WriteRange::Range), range: Some((5, 9)), ..w.clone() };
        assert_eq!(r.resolve_range(input, global), (5, 9));
        // A remembered window does not apply outside range mode.
        let back = WriteParams { range_mode: Some(WriteRange::Input), ..r.clone() };
        assert_eq!(back.resolve_range(input, global), input);
        // Range mode with no window yet renders the input, not nothing.
        let empty = WriteParams { range_mode: Some(WriteRange::Range), range: None, ..w.clone() };
        assert_eq!(empty.resolve_range(input, global), input);
        // Setups from before the modes: a window meant range, none meant input.
        let old_win = WriteParams { range_mode: None, range: Some((2, 3)), ..w.clone() };
        assert_eq!(old_win.mode(), WriteRange::Range);
        let old_none = WriteParams { range_mode: None, range: None, ..w };
        assert_eq!(old_none.mode(), WriteRange::Input);
    }

    #[test]
    fn write_paths_resolve_and_the_node_is_a_passthrough() {
        use std::path::PathBuf;
        // A `####` run takes the frame number at its own width.
        let w = WriteParams {
            path: "out/comp.####.exr".into(),
            start: 1001,
            range: None,
            range_mode: None,
            output: None,
            output_default: false,
            ..WriteParams::default()
        };
        assert_eq!(w.resolve_frame(0), PathBuf::from("out/comp.1001.exr"));
        assert_eq!(w.resolve_frame(12), PathBuf::from("out/comp.1013.exr"));
        assert_eq!(w.dir(), PathBuf::from("out"));
        // A wider run pads wider; a narrower one does not truncate.
        let wide = WriteParams { path: "s/f.######.exr".into(), ..w.clone() };
        assert_eq!(wide.resolve_frame(0), PathBuf::from("s/f.001001.exr"));
        // A bare .exr gets the number inserted before the extension.
        let bare =
            WriteParams { path: "out/comp.exr".into(), start: 1, ..w.clone() };
        assert_eq!(bare.resolve_frame(0), PathBuf::from("out/comp.0001.exr"));
        // Anything else is a directory.
        let dir =
            WriteParams { path: "renders/v003".into(), start: 1, ..w.clone() };
        assert_eq!(dir.resolve_frame(2), PathBuf::from("renders/v003/frame.0003.exr"));
        assert_eq!(dir.dir(), PathBuf::from("renders/v003"));

        // The node itself changes no pixels and takes no uniform slots.
        let g = one_node(NodeKind::Write(w));
        let plan = generate(&g);
        assert!(!plan.layout.node_slot.contains_key(&7));
        validate_wgsl("write", &plan.main.wgsl);
        // Editing the path is not a recompile.
        let g2 = one_node(NodeKind::Write(WriteParams {
            path: "elsewhere/####.exr".into(),
            start: 7,
            range: Some((2, 9)),
            range_mode: None,
            output: None,
            output_default: false,
            ..WriteParams::default()
        }));
        assert_eq!(g.topology_hash(), g2.topology_hash());
    }

    #[test]
    fn a_render_compiles_its_path_from_location_name_and_padding() {
        use std::path::PathBuf;
        // A new node: subfolder of the same name, four digits.
        let w = WriteParams { location: "renders".into(), name: "shot".into(), ..WriteParams::default() };
        assert!(w.subfolder && w.padding == 4 && w.path.is_empty());
        assert_eq!(PathBuf::from(w.pattern()), PathBuf::from("renders/shot/shot.####.exr"));
        assert_eq!(w.resolve_frame(0), PathBuf::from("renders/shot/shot.1001.exr"));
        assert_eq!(w.dir(), PathBuf::from("renders/shot"));
        // No subfolder; padding is its own setting.
        let flat = WriteParams { subfolder: false, padding: 6, start: 1, ..w.clone() };
        assert_eq!(flat.resolve_frame(2), PathBuf::from("renders/shot.000003.exr"));
        // A name is ONE name: separators and frame runs cannot sneak in.
        let odd = WriteParams { name: "a/b#".into(), ..w.clone() };
        assert_eq!(PathBuf::from(odd.pattern()), PathBuf::from("renders/a_b/a_b.####.exr"));
        let empty = WriteParams { name: "  ".into(), ..w.clone() };
        assert_eq!(PathBuf::from(empty.pattern()), PathBuf::from("renders/comp/comp.####.exr"));
        // Nowhere to write until there is a location.
        assert!(!WriteParams::default().is_set());
        assert!(w.is_set());
        // `~` is home, expanded at resolution and never in the authored text.
        if let Some(home) = std::env::var_os("HOME") {
            let t = WriteParams { location: "~/out".into(), ..w.clone() };
            assert_eq!(t.location, "~/out");
            assert_eq!(t.resolve_frame(0), PathBuf::from(home).join("out/shot/shot.1001.exr"));
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn old_render_paths_migrate_to_the_same_files() {
        use std::path::PathBuf;
        let load = |json: &str| -> WriteParams { serde_json::from_str(json).unwrap() };
        let legacy = |path: &str| WriteParams { path: path.into(), ..WriteParams::default() };
        for (path, location, name, padding) in [
            ("out/comp.####.exr", "out", "comp", 4),
            ("/abs/s/f.######.exr", "/abs/s", "f", 6),
            ("comp.####.exr", "", "comp", 4),
            ("out/comp.exr", "out", "comp", 4),
            ("renders/v003", "renders/v003", "frame", 4),
        ] {
            let w = load(&format!(r#"{{"path":"{path}","start":1001}}"#));
            assert_eq!((w.path.as_str(), w.location.as_str(), w.name.as_str(), w.padding, w.subfolder),
                ("", location, name, padding, false), "{path}");
            for i in [0, 7, 120] {
                assert_eq!(w.resolve_frame(i), legacy(path).resolve_frame(i), "{path} frame {i}");
            }
        }
        // One the fields cannot say stays a legacy path, writing where it did.
        for path in ["out/comp_####.exr", "out/####.exr", "out/c.####.v2.exr"] {
            let w = load(&format!(r#"{{"path":"{path}","start":1}}"#));
            assert_eq!(w.path, path);
            let old = WriteParams { start: 1, ..legacy(path) };
            assert_eq!(w.resolve_frame(3), old.resolve_frame(3), "{path}");
        }
        assert_eq!(load(r#"{"path":"out/c_####.exr","start":1}"#).resolve_frame(0), PathBuf::from("out/c_0001.exr"));
        // A node that never pointed anywhere comes back as a new one.
        let blank = load(r#"{"path":"","start":1001}"#);
        assert_eq!((blank.name.as_str(), blank.subfolder), ("comp", true));
        // And a new-style save round-trips untouched.
        let w = WriteParams { location: "r".into(), name: "x".into(), padding: 5, subfolder: false, ..WriteParams::default() };
        assert_eq!(load(&serde_json::to_string(&w).unwrap()), w);
    }

    #[test]
    fn grid_is_a_source_with_no_inputs() {
        let kind = NodeKind::Grid(GridParams::default());
        assert_eq!(kind.input_count(), 0);
        let mut g = Graph::default();
        g.nodes.insert(1, kind);
        g.view = Some(1);
        let plan = generate(&g);
        assert!(plan.main.sources.is_empty(), "a grid samples no texture");
        validate_wgsl("grid-only", &plan.main.wgsl);
        let v = pack_params(&g, &plan.layout, [64.0, 64.0]);
        let b = plan.layout.node_slot[&1] * 4;
        assert_eq!(&v[b..b + 3], &[64.0, 64.0, 2.0], "spacing + width");
        assert_eq!(&v[b + 8..b + 12], &[1.0, 1.0, 1.0, 1.0], "line colour");
    }
}

#[cfg(test)]
mod spline_curve_tests {
    use super::*;
    use crate::graph::{Graph, NodeKind, SplineMode, SplineParams, SplinePoint};

    /// The format these shapes are authored against. Square, so a rotation
    /// in normalized space and one in pixel space agree — the aspect cases
    /// get their own test.
    const SZ: [f32; 2] = [100.0, 100.0];

    fn validate_wgsl(name: &str, src: &str) {
        let module = naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("{name} parse: {e}\n{src}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name} validate: {e:?}"));
    }

    fn tri(mode: SplineMode) -> SplineParams {
        SplineParams {
            points: vec![
                SplinePoint::new([0.2, 0.2], 10.0),
                SplinePoint::new([0.8, 0.2], 20.0),
                SplinePoint::new([0.5, 0.8], 30.0),
            ],
            mode,
            ..Default::default()
        }
    }

    #[test]
    fn sample_count_is_mode_independent() {
        // The uniform layout is sized from point count alone, so switching
        // mode must never change the sample count (it would recompile).
        for m in SplineMode::ALL {
            let poly = spline_polyline(&tri(m), SZ);
            assert_eq!(poly.len(), 3 * spline_subd(3), "{m:?}");
        }
        assert!(spline_polyline(&SplineParams::default(), SZ).is_empty());
    }

    #[test]
    fn feather_interpolates_along_the_outline() {
        let poly = spline_polyline(&tri(SplineMode::Linear), SZ);
        // Segment 0 runs pt0 (10) -> pt1 (20): starts at 10, mid ~15.
        assert!((poly[0].1 - 10.0).abs() < 1e-4);
        let mid = poly[spline_subd(3) / 2].1;
        assert!((mid - 15.0).abs() < 1.0, "mid feather {mid}");
        // Wraps: the last segment heads back toward pt0's 10.
        assert!(poly.last().unwrap().1 < 30.0);
    }

    #[test]
    fn subdivision_spends_a_budget_and_stays_bounded() {
        // Dense while a shape has few points...
        assert_eq!(spline_subd(3), SPLINE_SUBD_MAX);
        assert!(spline_subd(6) > 8, "must beat the old flat 8 on hand-drawn shapes");
        // ...thinning as it gains them, so neither the per-pixel loop nor
        // the slot span that feeds it runs away.
        for n in 1..64usize {
            assert!(spline_subd(n) >= SPLINE_SUBD_MIN, "n={n}");
            assert!(spline_subd(n) <= SPLINE_SUBD_MAX, "n={n}");
            let m = spline_samples(n);
            assert!(
                m <= SPLINE_SAMPLE_BUDGET.max(n * SPLINE_SUBD_MIN),
                "n={n} spends {m}"
            );
            assert!(spline_subd(n) >= spline_subd(n + 1), "n={n} not monotonic");
        }
    }

    #[test]
    fn two_dense_splines_fit_the_uniform() {
        // The budget bump exists so one mask cannot reject a whole topology
        // by crowding out the next one.
        let dense = |seed: f32| {
            let pts = (0..6)
                .map(|i| {
                    let a = i as f32 * 1.047 + seed;
                    SplinePoint::new([0.5 + 0.3 * a.cos(), 0.5 + 0.3 * a.sin()], 20.0)
                })
                .collect();
            NodeKind::Spline(SplineParams {
                points: pts,
                mode: SplineMode::CatmullRom,
                ..Default::default()
            })
        };
        let mut g = Graph::default_poc();
        g.nodes.insert(7, dense(0.0));
        g.edges.insert((7, 0), 1);
        g.nodes.insert(8, dense(0.4));
        g.edges.insert((8, 0), 7);
        g.view = Some(8);
        assert!(generate(&g).layout.byte_size() <= 16384);
    }

    /// Chaining shapes is how a roto with several shapes is built here, so
    /// the uniform block has to hold a realistic stack of them. A shape
    /// spends the sample budget whatever its point count, so this is close
    /// to the worst case per shape — and it is the ceiling on how many
    /// shapes one chain can carry before the topology is rejected outright.
    #[test]
    fn a_chain_of_shapes_fits_the_uniform() {
        let dense = |seed: f32| {
            let pts = (0..24)
                .map(|i| {
                    let a = seed + i as f32 * std::f32::consts::TAU / 24.0;
                    SplinePoint::new([0.5 + 0.3 * a.cos(), 0.5 + 0.3 * a.sin()], 20.0)
                })
                .collect();
            NodeKind::Spline(SplineParams {
                points: pts,
                mode: SplineMode::CatmullRom,
                combine: crate::graph::MatteOp::Subtract,
                ..Default::default()
            })
        };
        let chain = |n: u64| {
            let mut g = Graph::default();
            for k in 0..n {
                g.nodes.insert(10 + k, dense(k as f32 * 0.3));
                if k > 0 {
                    g.edges.insert((10 + k, 0), 9 + k);
                }
            }
            g.view = Some(9 + n);
            g
        };
        assert!(
            generate(&chain(6)).layout.byte_size() <= 16384,
            "six chained shapes is a plausible roto and must render"
        );
        // Where the wall actually is, so a change that halves it is loud.
        let over = (1..24u64).find(|n| generate(&chain(*n)).layout.byte_size() > 16384);
        assert!(over.is_some_and(|n| n >= 8), "chain ceiling fell to {over:?}");
    }

    /// A setup load reuses ids: the outgoing layout's LAST slot may be a
    /// one-slot node's while the new graph has a Grade there. Packing must
    /// skip it (the params are rewritten once the new pipeline lands), not
    /// slice past the end of the block.
    #[test]
    fn a_reused_id_with_a_bigger_span_is_skipped_not_overrun() {
        let small = {
            let mut g = Graph::default_poc();
            g.nodes.insert(7, NodeKind::Gamma(Default::default()));
            g.edges.insert((7, 0), 1);
            g.view = Some(7);
            g
        };
        let layout = generate(&small).layout;
        let mut big = small.clone();
        big.nodes.insert(7, NodeKind::Grade(Default::default()));
        let v = pack_params(&big, &layout, [100.0, 100.0]);
        assert_eq!(v.len(), layout.array_len() * 4);
    }

    #[test]
    fn a_stale_layout_decimates_rather_than_truncates() {
        // The layout can be one topology behind while the recompile lands.
        // The short pack must still describe the WHOLE outline, coarser —
        // dropping the tail closes the shape with a chord across it.
        let small = {
            let mut g = Graph::default_poc();
            g.nodes.insert(7, NodeKind::Spline(tri(SplineMode::Linear)));
            g.edges.insert((7, 0), 1);
            g.view = Some(7);
            g
        };
        let mut big_sp = tri(SplineMode::Linear);
        for i in 0..9 {
            let a = i as f32 * 0.7;
            big_sp.points.push(SplinePoint::new([0.5 + 0.2 * a.cos(), 0.5 + 0.2 * a.sin()], 5.0));
        }
        let mut big = small.clone();
        big.nodes.insert(7, NodeKind::Spline(big_sp.clone()));

        let layout = generate(&small).layout;
        let v = pack_params(&big, &layout, [100.0, 100.0]);
        let base = layout.node_slot[&7] * 4;
        let fit = v[base] as usize;
        assert!(fit > 2, "packed {fit} samples into a stale span");
        let poly = spline_polyline(&big_sp, SZ);
        assert!(fit < poly.len(), "this case is only interesting when it clamps");
        // The last packed sample belongs to the last stretch of the outline,
        // not to wherever a truncation happened to stop.
        let last = [v[base + 8 + (fit - 1) * 2], v[base + 9 + (fit - 1) * 2]];
        // Packed in pixels: no format here, so the output's 100x100.
        let w = poly[(fit - 1) * poly.len() / fit].0;
        let want = [w[0] * 100.0, w[1] * 100.0];
        assert!(
            (last[0] - want[0]).abs() < 1e-4 && (last[1] - want[1]).abs() < 1e-4,
            "last packed {last:?} wanted {want:?}"
        );
    }

    /// Every combine op has to produce a shader that actually compiles —
    /// each is a different emitted expression, so each is a different
    /// program, and a typo in one of five arms would otherwise only show up
    /// when someone picked that op.
    #[test]
    fn every_combine_op_compiles_and_recompiles() {
        use crate::graph::MatteOp;
        let graph_with = |c: MatteOp, wired: bool| {
            let mut g = Graph::default_poc();
            g.nodes.insert(
                7,
                NodeKind::Spline(SplineParams { combine: c, ..tri(SplineMode::Linear) }),
            );
            if wired {
                // Another shape upstream: mask -> mask, the chain a roto is
                // actually built out of.
                g.nodes.insert(8, NodeKind::Spline(tri(SplineMode::CatmullRom)));
                g.edges.insert((7, 0), 8);
            }
            g.view = Some(7);
            g
        };
        let mut hashes = std::collections::HashSet::new();
        for c in MatteOp::ALL {
            for wired in [false, true] {
                let g = graph_with(c, wired);
                let plan = generate(&g);
                validate_wgsl(&format!("spline-{}-{wired}", c.name()), &plan.main.wgsl);
                hashes.insert(g.topology_hash());
            }
            // Unwired, the shape is still there to combine WITH nothing.
            // (Port 1 is the camera, lowered away before codegen.)
            assert_eq!(NodeKind::Spline(tri(SplineMode::Linear)).input_count(), 2);
        }
        // The op picks the expression, so it is structural, like a Blend's:
        // every op x wired/unwired is a distinct program.
        assert_eq!(
            hashes.len(),
            MatteOp::ALL.len() * 2,
            "combine op must reach the topology hash"
        );
    }

    /// ...and the matte controls must NOT. They are uniforms; a shape whose
    /// opacity is being scrubbed cannot be recompiling per frame.
    #[test]
    fn matte_controls_and_the_transform_stay_uniforms() {
        use crate::graph::ShapeXf;
        let base_g = |sp: SplineParams| {
            let mut g = Graph::default_poc();
            g.nodes.insert(7, NodeKind::Spline(sp));
            g.view = Some(7);
            g
        };
        let g0 = base_g(tri(SplineMode::Linear));
        let h = g0.topology_hash();
        for sp in [
            SplineParams { opacity: 0.3, ..tri(SplineMode::Linear) },
            SplineParams { expand: -7.0, ..tri(SplineMode::Linear) },
            SplineParams { falloff: 1.0, ..tri(SplineMode::Linear) },
            SplineParams { invert: true, ..tri(SplineMode::Linear) },
            SplineParams {
                xf: ShapeXf { translate: [0.2, 0.1], rotate: 45.0, ..Default::default() },
                ..tri(SplineMode::Linear)
            },
            // Mode is CPU-side (it only reshapes the sample list), so it is
            // not structural either.
            tri(SplineMode::BSpline),
        ] {
            assert_eq!(base_g(sp).topology_hash(), h, "this must not recompile");
        }

        // And they reach the uniform where the shader looks for them.
        let sp = SplineParams {
            opacity: 0.25,
            expand: 3.5,
            falloff: 0.75,
            invert: true,
            ..tri(SplineMode::Linear)
        };
        let g = base_g(sp);
        let plan = generate(&g);
        let v = pack_params(&g, &plan.layout, [100.0, 100.0]);
        let base = plan.layout.node_slot[&7] * 4;
        assert_eq!(&v[base + 4..base + 8], &[0.25, 3.5, 0.75, 1.0], "control slot");
        // Header: sample count, then the bounding circle; the feather
        // block sits right after the positions (the shader derives it).
        let m = spline_samples(3);
        assert_eq!(v[base] as usize, m, "sample count");
        assert!(v[base + 1] > 0.0, "bounding radius");
    }

    /// The transform is applied to the CONTROL points at pack time, which is
    /// only legal because every mode is an affine combination of them. Proof:
    /// transforming the shape and transforming its samples agree.
    #[test]
    fn the_shape_transform_commutes_with_every_mode() {
        use crate::graph::ShapeXf;
        const RECT: [f32; 2] = [200.0, 100.0]; // deliberately not square
        let xf = ShapeXf {
            translate: [0.05, -0.1],
            rotate: 25.0,
            scale: [1.3, 0.8],
            center: [0.4, 0.6],
        };
        for m in SplineMode::ALL {
            let plain = spline_polyline(&tri(m), RECT);
            let moved = spline_polyline(&SplineParams { xf, ..tri(m) }, RECT);
            assert_eq!(plain.len(), moved.len(), "{m:?}: sample count must not move");
            for (i, (a, b)) in plain.iter().zip(&moved).enumerate() {
                let want = xf.point(a.0, RECT);
                assert!(
                    (b.0[0] - want[0]).abs() < 1e-4 && (b.0[1] - want[1]).abs() < 1e-4,
                    "{m:?} sample {i}: {:?} != {want:?}",
                    b.0
                );
                assert_eq!(a.1, b.1, "{m:?}: feather is a width, it does not rotate");
            }
        }
        // The identity really is a no-op, not an almost-no-op: an
        // untransformed shape must pack bit-for-bit what it always did.
        for m in SplineMode::ALL {
            assert_eq!(
                spline_polyline(&tri(m), RECT),
                spline_polyline(&SplineParams { xf: ShapeXf::default(), ..tri(m) }, RECT)
            );
        }
    }

    /// The transform packs against the NODE's own format, not the output's —
    /// a mask anchored to a 2:1 plate has to rotate in that plate's pixels.
    #[test]
    fn the_transform_packs_in_the_nodes_own_format() {
        use crate::graph::ShapeXf;
        let sp = SplineParams {
            xf: ShapeXf { rotate: 90.0, ..Default::default() },
            ..tri(SplineMode::Linear)
        };
        let mut g = Graph::default_poc();
        g.nodes.insert(7, NodeKind::Spline(sp.clone()));
        g.view = Some(7);
        g.formats.insert(7, (200, 100));
        let plan = generate(&g);
        let v = pack_params(&g, &plan.layout, [64.0, 64.0]);
        let base = plan.layout.node_slot[&7] * 4;
        let want = spline_polyline(&sp, [200.0, 100.0]);
        // In the node's PIXELS, not normalized against the output size.
        assert!((v[base + 8] - want[0].0[0] * 200.0).abs() < 1e-3, "packed in node pixels");
        assert!((v[base + 9] - want[0].0[1] * 100.0).abs() < 1e-3);
        // Which is NOT what the output-sized transform would have given.
        let other = spline_polyline(&sp, [64.0, 64.0]);
        assert!((want[0].0[0] - other[0].0[0]).abs() > 1e-4, "aspect must matter");
    }

    #[test]
    fn modes_differ_where_they_should() {
        let lin = spline_polyline(&tri(SplineMode::Linear), SZ);
        let cr = spline_polyline(&tri(SplineMode::CatmullRom), SZ);
        let bs = spline_polyline(&tri(SplineMode::BSpline), SZ);
        // Linear passes through its control points; Catmull-Rom does too.
        assert!((lin[0].0[0] - 0.2).abs() < 1e-5 && (cr[0].0[0] - 0.2).abs() < 1e-5);
        // B-spline is approximating: it pulls inside the control polygon.
        assert!((bs[0].0[0] - 0.2).abs() > 1e-3, "b-spline should not interpolate");
        // Catmull-Rom bulges away from the straight chord between points.
        let k = spline_subd(3) / 2;
        assert!((cr[k].0[1] - lin[k].0[1]).abs() > 1e-4);

        // Bezier with zero handles collapses to straight segments...
        let bez = spline_polyline(&tri(SplineMode::Bezier), SZ);
        assert!((bez[k].0[0] - lin[k].0[0]).abs() < 1e-5);
        // ...and bends once a handle is pulled.
        let mut curved = tri(SplineMode::Bezier);
        curved.points[0].h_out = [0.0, -0.4];
        let cb = spline_polyline(&curved, SZ);
        assert!((cb[k].0[1] - bez[k].0[1]).abs() > 1e-3, "handle should bend it");
    }
}

#[cfg(test)]
mod plugin_codegen_tests {
    use super::*;
    use crate::graph::{Graph, NodeKind};
    use crate::plugins::{self, ParamSpec, ShaderNodeDef};

    fn validate_wgsl(name: &str, src: &str) {
        let module = naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("{name} parse: {e}\n{src}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name} validate: {e:?}"));
    }

    fn vignette_def(type_id: &str) -> ShaderNodeDef {
        ShaderNodeDef::new(
            type_id.into(),
            "Vignette".into(),
            [96, 96, 160],
            1,
            1,
            vec![
                ParamSpec { name: "radius".into(), label: "radius".into(), default: 0.7, min: 0.05, max: 1.5 },
                ParamSpec { name: "softness".into(), label: "soft".into(), default: 0.4, min: 0.01, max: 1.0 },
                ParamSpec { name: "gain".into(), label: "gain".into(), default: 0.0, min: 0.0, max: 1.0 },
            ],
            "vec4f($in.rgb * mix(vig_w($pos, $size, $radius, $softness), 1.0, $gain), $in.a)"
                .into(),
            "fn vig_w(pos: vec2f, size: vec2f, r: f32, s: f32) -> f32 {\n\
             let d = length(pos / size - vec2f(0.5));\n\
             return 1.0 - smoothstep(r, r + s, d);\n}"
                .into(),
        )
    }

    /// Read(0) -> custom -> view.
    fn custom_graph(type_id: &str) -> Graph {
        let mut g = Graph::default();
        g.nodes.insert(
            0,
            NodeKind::Read {
                source: 0,
                path: String::new(),
                range: None,
                ram_cache: true,
                nvme_cache: true,
                layer: None,
                input: Default::default(),
                multi: false,
                start: None,
                first: None,
                last: None,
                edges: Default::default(),
                cache_on_play: true,
                single: Default::default(),
                eight_bit: false,
                scale: Default::default(),
            },
        );
        g.nodes.insert(
            1,
            NodeKind::Custom { type_id: type_id.into(), params: Default::default() },
        );
        g.edges.insert((1, 0), 0);
        g.view = Some(1);
        g
    }

    #[test]
    fn keyer_slots_follow_blob_count_and_shader_validates() {
        use crate::keyer::{KeyBlob, KeyerParams};
        let mut g = Graph::default_poc();
        let mut kp = KeyerParams::default();
        kp.blobs.push(KeyBlob::default());
        kp.blobs.push(KeyBlob { weight: -0.8, ..KeyBlob::default() });
        g.nodes.insert(7, NodeKind::Keyer(kp.clone()));
        g.edges.insert((7, 0), 1); // image <- read A
        g.view = Some(7);
        let plan = generate(&g);
        validate_wgsl("keyer", &plan.main.wgsl);
        assert!(plan.main.wgsl.contains("keyer_matte"));
        // 2 header + 2 per blob.
        let span = plan.layout.total_slots - plan.layout.node_slot[&7];
        assert_eq!(span, 2 + 2 * 3);
        let params = pack_params(&g, &plan.layout, [1920.0, 1080.0]);
        let base = plan.layout.node_slot[&7] * 4;
        assert_eq!(params[base + 4], 3.0, "blob count in the header");
        assert_eq!(params[base + 6], 0.0, "preview off unless the app sets it");
        assert_eq!(params[base + 8 + 2 * 8 + 3], -0.8, "third blob's weight");

        // Blob count is topology; everything else is a uniform write.
        let topo = g.topology_hash();
        let mut moved = kp.clone();
        moved.blobs[0].center = [0.4, 0.1, -0.05];
        moved.gamma = 2.0;
        g.nodes.insert(7, NodeKind::Keyer(moved));
        assert_eq!(g.topology_hash(), topo, "param edits must not recompile");
        let mut grown = kp;
        grown.blobs.push(KeyBlob::default());
        g.nodes.insert(7, NodeKind::Keyer(grown));
        assert_ne!(g.topology_hash(), topo, "adding a blob changes the span");
    }

    #[test]
    fn keyer_preview_never_reaches_a_downstream_consumer() {
        use crate::keyer::KeyerParams;
        // A keyer feeding a Grade's mask: the preview lerp is present in the
        // shader but its uniform is 0, so the mask stays a matte.
        let mut g = Graph::default_poc();
        g.nodes.insert(7, NodeKind::Keyer(KeyerParams { previewing: true, ..Default::default() }));
        g.edges.insert((7, 0), 1);
        g.edges.insert((4, 1), 7); // grade.mask <- keyer
        let plan = generate(&g);
        validate_wgsl("keyer-mask", &plan.main.wgsl);
        let params = pack_params(&g, &plan.layout, [1920.0, 1080.0]);
        let base = plan.layout.node_slot[&7] * 4;
        // `previewing` is transient state the APP only sets for the view
        // node; here the keyer is not it, but the flag is still packed —
        // this asserts the app's contract is the only gate.
        assert_eq!(params[base + 6], 1.0);
    }

    #[test]
    fn plugin_shader_is_valid_wgsl_with_helpers_once() {
        plugins::register(vignette_def("test_vig_a"));
        let plan = generate(&custom_graph("test_vig_a"));
        let w = &plan.main.wgsl;
        validate_wgsl("plugin shader", w);
        assert_eq!(w.matches("fn vig_w").count(), 1, "helpers must emit once");
        assert!(w.contains("smoothstep"), "{w}");
    }

    #[test]
    fn params_pack_in_manifest_order() {
        plugins::register(vignette_def("test_vig_b"));
        let mut g = custom_graph("test_vig_b");
        if let Some(NodeKind::Custom { params, .. }) = g.nodes.get_mut(&1) {
            params.insert("softness".into(), 0.9);
        }
        let plan = generate(&g);
        let v = pack_params(&g, &plan.layout, [64.0, 32.0]);
        let base = plan.layout.node_slot[&1] * 4;
        assert_eq!(v[base], 0.7, "radius default");
        assert_eq!(v[base + 1], 0.9, "stored softness");
        assert_eq!(v[base + 2], 0.0, "gain default");
    }

    #[test]
    fn missing_plugin_passes_through_and_still_validates() {
        let plan = generate(&custom_graph("nobody_registered_this"));
        validate_wgsl("passthrough", &plan.main.wgsl);
        // The read's texture still feeds the output.
        assert!(plan.main.wgsl.contains("sample_px"), "{}", plan.main.wgsl);
    }

    #[test]
    fn editing_the_plugin_changes_the_topology_hash() {
        plugins::register(vignette_def("test_vig_c"));
        let g = custom_graph("test_vig_c");
        let h1 = g.topology_hash();
        let mut d = vignette_def("test_vig_c");
        d.expr.push_str(" * 1.0");
        plugins::register(ShaderNodeDef::new(
            d.type_id, d.label, d.color, d.inputs, d.tiles, d.params, d.expr, d.helpers,
        ));
        assert_ne!(g.topology_hash(), h1, "source edit must recompile");
    }
}

#[cfg(test)]
mod loop_tests {
    use super::*;
    use crate::graph::{Graph, LoopParams, NodeKind};

    /// Read 1 -> seed; LoopIn(5) -> Mult(6) -> LoopOut(7).
    fn loop_graph() -> Graph {
        let mut g = Graph::default();
        g.nodes.insert(1, NodeKind::read(0, ""));
        g.nodes.insert(5, NodeKind::LoopIn);
        g.nodes.insert(6, NodeKind::Mult(Default::default()));
        g.edges.insert((6, 0), 5);
        g.nodes.insert(7, NodeKind::LoopOut(LoopParams { pair: 5, n: 4.0 }));
        g.edges.insert((7, 0), 6);
        g.edges.insert((7, 1), 1);
        g.view = Some(7);
        g
    }

    fn validate(wgsl: &str) {
        let module = naga::front::wgsl::parse_str(wgsl).expect("parses");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("validates");
    }

    /// The canonical loop: one stage, its body tapping LoopPrev, one aux
    /// (the seed), everything naga-valid.
    #[test]
    fn a_loop_makes_one_stage_with_a_seed_aux() {
        let plan = generate(&loop_graph());
        assert_eq!(plan.blurs.len(), 1);
        let (sid, kind, body, aux) = &plan.blurs[0];
        assert_eq!(sid.0, 7);
        assert_eq!(*kind, FilterKind::Loop);
        assert!(
            body.sources.contains(&SourceBinding::LoopPrev(5)),
            "the body reads the tap: {:?}",
            body.sources
        );
        assert_eq!(aux.len(), 1, "the seed subtree resolves as aux 0");
        assert!(aux[0].sources.contains(&SourceBinding::Reader(0)));
        validate(&body.wgsl);
        validate(&aux[0].wgsl);
        validate(&plan.main.wgsl);
    }

    /// n is an encode-time pass count: dragging it must never recompile.
    #[test]
    fn iteration_count_is_not_topology() {
        let g = loop_graph();
        let h1 = g.topology_hash();
        let mut g2 = loop_graph();
        if let Some(NodeKind::LoopOut(p)) = g2.nodes.get_mut(&7) {
            p.n = 60.0;
        }
        assert_eq!(g2.topology_hash(), h1);
        // Re-pairing IS topology (it renames the texture binding).
        if let Some(NodeKind::LoopOut(p)) = g2.nodes.get_mut(&7) {
            p.pair = 0;
        }
        assert_ne!(g2.topology_hash(), h1);
    }

    /// A stage node INSIDE the tap's cone cannot iterate (its pass runs once
    /// a frame): the loop degrades to a single fused body evaluation, and
    /// still compiles.
    #[test]
    fn a_stage_in_the_tap_cone_degrades_to_one_pass() {
        let mut g = loop_graph();
        g.nodes.insert(8, NodeKind::Blur(Default::default()));
        g.edges.insert((8, 0), 5); // Blur reads the tap
        g.edges.insert((6, 0), 8); // body goes LoopIn -> Blur -> Mult
        assert!(!loop_ok(&g, 7, 5));
        let plan = generate(&g);
        assert!(
            !plan.blurs.iter().any(|(_, k, _, _)| *k == FilterKind::Loop),
            "no loop stage"
        );
        validate(&plan.main.wgsl);
    }

    /// A stage OUTSIDE the cone is fine: it renders once and the body
    /// samples its result every iteration.
    #[test]
    fn a_stage_outside_the_tap_cone_is_allowed() {
        let mut g = loop_graph();
        // Blur of the READ, blended into the body via a Blend.
        g.nodes.insert(8, NodeKind::Blur(Default::default()));
        g.edges.insert((8, 0), 1);
        g.nodes.insert(9, NodeKind::blend());
        g.edges.insert((9, 0), 6); // over: the tap chain
        g.edges.insert((9, 1), 8); // under: the blurred source
        g.edges.insert((7, 0), 9);
        assert!(loop_ok(&g, 7, 5));
        let plan = generate(&g);
        assert!(plan.blurs.iter().any(|(_, k, _, _)| *k == FilterKind::Loop));
        assert!(plan.blurs.iter().any(|(_, k, _, _)| *k == FilterKind::Node));
    }

    /// Unpaired (pair 0 or the LoopIn deleted): no stage, body emits once.
    #[test]
    fn an_unpaired_loopout_is_a_single_fused_evaluation() {
        let mut g = loop_graph();
        if let Some(NodeKind::LoopOut(p)) = g.nodes.get_mut(&7) {
            p.pair = 0;
        }
        let plan = generate(&g);
        assert!(plan.blurs.is_empty());
        validate(&plan.main.wgsl);
    }
}

#[cfg(test)]
mod zone_tests {
    use super::*;
    use crate::graph::{Graph, GradeParams, GradeZones, NodeKind};

    fn zoned(zones: Option<GradeZones>) -> Graph {
        let mut g = Graph::default();
        g.nodes.insert(0, NodeKind::read(0, ""));
        g.nodes.insert(1, NodeKind::Grade(GradeParams { zones, ..Default::default() }));
        g.edges.insert((1, 0), 0);
        g.view = Some(1);
        g
    }

    /// Zones fuse: one grade4 per zone on top of the master's, one curve
    /// lookup per zone, and no stage of their own.
    #[test]
    fn a_zoned_grade_is_one_fused_pass() {
        let plan = generate(&zoned(Some(GradeZones::default())));
        assert!(plan.blurs.is_empty(), "zones must not take a pass");
        assert_eq!(plan.main.wgsl.matches("= grade4(").count(), 4);
        assert_eq!(plan.main.wgsl.matches("clamp(curve_eval(").count(), 3, "one lookup per zone");
        let plain = generate(&zoned(None));
        assert_eq!(plain.main.wgsl.matches("= grade4(").count(), 1);
    }

    /// The zone count is topology; nothing else about a zone is.
    #[test]
    fn only_the_zone_count_recompiles() {
        let plain = zoned(None);
        let three = zoned(Some(GradeZones::default()));
        assert_ne!(plain.topology_hash(), three.topology_hash());
        let mut moved = three.clone();
        if let Some(NodeKind::Grade(g)) = moved.nodes.get_mut(&1) {
            let z = g.zones.as_mut().unwrap();
            z.adjust[2].gain = [3.0; 3];
            z.key.ranges.as_mut().unwrap().move_edge(1, 1.2);
            z.key.sync_from_ranges();
            z.key.bands[1].points.push([0.4, 0.6]);
        }
        assert_eq!(three.topology_hash(), moved.topology_hash());
        let mut four = three.clone();
        if let Some(NodeKind::Grade(g)) = four.nodes.get_mut(&1) {
            let z = g.zones.as_mut().unwrap();
            let n = z.key.split_band(1);
            z.band_split(n);
        }
        assert_ne!(three.topology_hash(), four.topology_hash());
    }

    /// The ranges preview is a UNIFORM the app ARMS: it never moves the
    /// topology hash, it rides in the lanes `grade4` does not read, and it
    /// stays off until the app marks the node the one being viewed.
    #[test]
    fn the_zone_preview_is_an_armed_uniform() {
        use crate::graph::{ZonePreview, ZoneView};
        let mut g = zoned(Some(GradeZones::default()));
        let plan = generate(&g);
        let slot = plan.layout.node_slot[&1];
        let base = slot * 4;
        let hash = g.topology_hash();
        let packed = |g: &Graph| pack_params(g, &plan.layout, [100.0, 100.0]);
        assert_eq!(packed(&g)[base + 7], 0.0, "off by default");

        let arm = |g: &mut Graph, view, zone, armed| {
            if let Some(NodeKind::Grade(gr)) = g.nodes.get_mut(&1) {
                let z = gr.zones.as_mut().unwrap();
                z.preview = ZonePreview { view, zone };
                z.previewing = armed;
            }
        };
        // Authored but unarmed: nothing to show, and nothing recompiled.
        arm(&mut g, ZoneView::Highlight, 2, false);
        assert_eq!(g.topology_hash(), hash, "a preview must never recompile");
        assert_eq!(packed(&g)[base + 7], 0.0, "not the view target");

        arm(&mut g, ZoneView::Highlight, 2, true);
        assert_eq!(g.topology_hash(), hash);
        let v = packed(&g);
        assert_eq!((v[base + 7], v[base + 11]), (2.0, 2.0), "mode in gamma.w, zone in gain.w");
        // Which are exactly the lanes `grade4` leaves alone — the master's
        // gamma and gain are untouched by a preview.
        assert_eq!(&v[base + 4..base + 7], &[1.0; 3]);
        assert_eq!(&v[base + 8..base + 11], &[1.0; 3]);
        // A zone the partition no longer has clamps to the last one rather
        // than reading past the end of the span.
        arm(&mut g, ZoneView::Matte, 99, true);
        assert_eq!(packed(&g)[base + 11], 2.0);

        // The shader reads those lanes...
        let wgsl = &plan.main.wgsl;
        assert!(wgsl.contains(&format!("P.v[{}].w > 0.5", slot + 1)), "{wgsl}");
        assert!(wgsl.contains(&format!("P.v[{}].w == 0.0", slot + 2)));
        // ...and a Grade with no zones pays for none of it.
        let plain = generate(&zoned(None));
        let ps = plain.layout.node_slot[&1];
        assert!(!plain.main.wgsl.contains(&format!("P.v[{}].w > 0.5", ps + 1)));
    }

    /// The packer lays each zone where the shader reads it: adjustments in
    /// the master's layout, then the band's curve in the LumaKey's.
    #[test]
    fn zones_pack_where_the_shader_reads_them() {
        let mut g = zoned(Some(GradeZones::default()));
        if let Some(NodeKind::Grade(gr)) = g.nodes.get_mut(&1) {
            let z = gr.zones.as_mut().unwrap();
            z.adjust[1].gain = [1.5, 1.6, 1.7];
            z.adjust[1].saturation = 0.25;
            z.adjust[1].contrast = 1.4;
            z.adjust[1].pivot = 0.3;
            gr.contrast = 0.8;
            gr.pivot = 0.5;
        }
        let plan = generate(&g);
        let v = pack_params(&g, &plan.layout, [100.0, 100.0]);
        let slot = plan.layout.node_slot[&1];
        let base = slot * 4;
        assert_eq!(&v[base + 16..base + 18], &[0.8, 0.5], "master contrast + pivot");
        let zb = base + GRADE_SLOTS * 4 + ZONE_SLOTS * 4; // zone 1
        assert_eq!(&v[zb + 8..zb + 11], &[1.5, 1.6, 1.7]);
        assert_eq!(v[zb + 3], 0.25);
        assert_eq!(&v[zb + 16..zb + 18], &[1.4, 0.3], "zone contrast + pivot");
        let NodeKind::Grade(gr) = &g.nodes[&1] else { unreachable!() };
        let pts = gr.zones.as_ref().unwrap().key.band(1).curve();
        let hdr = zb + GRADE_SLOTS * 4;
        assert_eq!(v[hdr] as usize, pts.len());
        assert_eq!(v[hdr + 1], 1.0, "smooth");
        assert_eq!(v[hdr + 2], 0.0, "not inverted");
        for (k, p) in pts.iter().enumerate() {
            assert_eq!(&v[hdr + 4 + k * 2..hdr + 6 + k * 2], p);
        }
        // Three zones after the master's own slots, and nothing past them.
        assert_eq!(vec4_slots_for(&g.nodes[&1]), GRADE_SLOTS + 3 * ZONE_SLOTS);
    }
}


#[cfg(test)]
mod proxy_tests {
    use super::*;
    use crate::graph::{Graph, NodeKind, TransformParams};

    /// Pixel-denominated params scale with the proxy at PACK time; the
    /// shape of the shader (and so the topology hash) never moves.
    #[test]
    fn translate_packs_scaled_and_never_recompiles() {
        let mut g = Graph::default();
        g.nodes.insert(1, NodeKind::read(0, ""));
        g.nodes.insert(
            2,
            NodeKind::Transform(TransformParams {
                matte: false,
                translate: [10.0, -6.0],
                rotate: 0.0,
                scale: [1.0, 1.0],
                center: [0.5, 0.5],
                scale_gang: false,
                pin: crate::graph::UNIT_QUAD,
                mod_blend: Default::default(),
                shear: [0.0, 0.0],
                taper: 0.0,
                taper_angle: 90.0,
                invert: false,
            }),
        );
        g.edges.insert((2, 0), 1);
        g.view = Some(2);
        let plan = generate(&g);
        let full = pack_params(&g, &plan.layout, [128.0, 64.0]);
        let h_full = g.topology_hash();
        g.proxy = 0.5;
        let half = pack_params(&g, &plan.layout, [64.0, 32.0]);
        assert_eq!(g.topology_hash(), h_full, "proxy is never topology");
        let slot = plan.layout.node_slot[&2] * 4;
        // Inverse affine of a pure translate: row0 = (1, 0, -tx).
        assert_eq!(full[slot + 2], -10.0);
        assert_eq!(half[slot + 2], -5.0, "translate halves under proxy");
        assert_eq!(half[slot], 1.0, "scale part untouched");
    }
}


#[cfg(test)]
mod gain_tests {
    use super::*;
    use crate::graph::{BlendOp, Graph, NodeKind};

    /// Gains ride the mix slot as uniforms: packing changes, the hash does
    /// not, and the emitted shader scales each input before the op.
    #[test]
    fn blend_gains_are_uniforms_not_topology() {
        let mut g = Graph::default();
        g.nodes.insert(1, NodeKind::read(0, ""));
        g.nodes.insert(2, NodeKind::read(1, ""));
        g.nodes.insert(
            3,
            NodeKind::Blend { op: BlendOp::Add, mix: 1.0, gain: [1.0, 1.0] },
        );
        g.edges.insert((3, 0), 1);
        g.edges.insert((3, 1), 2);
        g.view = Some(3);
        let plan = generate(&g);
        let h = g.topology_hash();
        let slot = plan.layout.node_slot[&3] * 4;
        let p1 = pack_params(&g, &plan.layout, [8.0, 8.0]);
        assert_eq!((p1[slot + 1], p1[slot + 2]), (1.0, 1.0));
        if let Some(NodeKind::Blend { gain, .. }) = g.nodes.get_mut(&3) {
            *gain = [0.25, 1.5];
        }
        assert_eq!(g.topology_hash(), h, "gains never recompile");
        let p2 = pack_params(&g, &plan.layout, [8.0, 8.0]);
        assert_eq!((p2[slot + 1], p2[slot + 2]), (0.25, 1.5));
        // And the shader actually consumes both lanes of the slot.
        let vp = &plan.main.wgsl;
        assert!(vp.contains(".y;") && vp.contains(".z;"), "gains sampled: {vp}");
    }

    /// The harp's strings: sparse gains default to 1.0 in the packed
    /// uniform, edits land in their lane, and none of it is topology.
    #[test]
    fn recombine_gains_pack_sparse_and_stay_uniforms() {
        use crate::graph::RecombineParams;
        let mut g = Graph::default();
        for i in 0..5u64 {
            g.nodes.insert(i + 1, NodeKind::read(i as usize, ""));
        }
        g.nodes.insert(
            9,
            NodeKind::Recombine(RecombineParams {
                n: 5,
                ..Default::default()
            }),
        );
        for i in 0..5u64 {
            g.edges.insert((9, i as usize), i + 1);
        }
        g.view = Some(9);
        let plan = generate(&g);
        let h = g.topology_hash();
        let slot = plan.layout.node_slot[&9] * 4;
        let p1 = pack_params(&g, &plan.layout, [8.0, 8.0]);
        assert_eq!(&p1[slot..slot + 5], &[1.0; 5], "sparse gains read 1.0");
        if let Some(NodeKind::Recombine(p)) = g.nodes.get_mut(&9) {
            p.gain = vec![1.0, 0.5, 1.0, 2.0];
        }
        assert_eq!(g.topology_hash(), h, "gains never recompile");
        let p2 = pack_params(&g, &plan.layout, [8.0, 8.0]);
        assert_eq!(&p2[slot..slot + 5], &[1.0, 0.5, 1.0, 2.0, 1.0]);
        // Five strings span two vec4s: the second slot's lane is consumed.
        assert!(plan.main.wgsl.contains(&format!("P.v[{}].x", slot / 4 + 1)));
    }
}

#[cfg(test)]
mod shared_noise_tests {
    use super::*;
    use crate::graph::{BlendOp, Graph, NodeKind};

    /// One Noise feeding two branches into a Blend (a real comp's setup) is
    /// reached twice by the fused pass, so its sub-graph is compiled twice.
    /// Its functions must be declared ONCE: naga refused the redefinition,
    /// the whole plan failed, and the viewer kept the last good picture
    /// while no knob downstream did anything.
    #[test]
    fn a_noise_on_two_paths_declares_its_functions_once() {
        let mut g = Graph::default();
        g.nodes.insert(1, NodeKind::Noise(crate::graph::NoiseParams::fresh()));
        g.nodes.insert(2, NodeKind::Gamma(Default::default()));
        g.nodes.insert(3, NodeKind::Gamma(Default::default()));
        g.nodes.insert(4, NodeKind::Blend { op: BlendOp::Add, mix: 1.0, gain: [0.5, 0.25] });
        g.edges.insert((2, 0), 1);
        g.edges.insert((3, 0), 1);
        g.edges.insert((4, 0), 2);
        g.edges.insert((4, 1), 3);
        g.view = Some(4);
        let wgsl = generate(&g).main.wgsl;
        let module = naga::front::wgsl::parse_str(&wgsl)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&wgsl)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("validates");
    }
}
