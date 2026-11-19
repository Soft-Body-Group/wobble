//! Graph model: UI-agnostic description of the node graph.
//!
//! The app crate mirrors its snarl editor state into this model; the engine
//! consumes it for codegen. Parameter values live here too, but they are NOT
//! part of the topology hash — edits to them never trigger a recompile.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crate::nodes::NodeDef;

pub type NodeId = u64;

// A node's params live with the node, in `crate::nodes`; the graph model
// re-exports them so `graph::SatParams` stays the path everyone uses.
pub use crate::blender::BlenderParams;
pub use crate::sdf::SdfParams;
pub use crate::nodes::blur::BlurParams;
pub use crate::nodes::colorcompress::ColorCompressParams;
pub use crate::nodes::colorslice::{ColorSliceParams, SliceVector};
pub use crate::nodes::expression::ExpressionParams;
pub use crate::nodes::huerotate::HueRotateParams;
pub use crate::nodes::lut::{LutOut, LutParams, LutShaper, LutSource};
pub use crate::nodes::text::TextParams;
pub use crate::nodes::burnin::{BurninCorner, BurninParams};
pub use crate::nodes::flipflop::{FlipFlopParams, FlipMode};
pub use crate::nodes::turn::{TurnMode, TurnParams};
pub use crate::nodes::uncrop::UncropParams;
pub use crate::nodes::compare::{CompareFit, CompareMode, CompareParams};
pub use crate::nodes::morphdissolve::MorphDissolveParams;
pub use crate::nodes::mirror::{MirrorParams, MirrorSide};
pub use crate::nodes::gamma::GammaParams;
pub use crate::nodes::tonemap::{ToneMapParams, ToneMode, ToneOp};
pub use crate::nodes::uvwarp::{UvWarpFilter, UvWarpParams};
pub use crate::nodes::env::EnvParams;
pub use crate::nodes::refract::{NormalY, RefractMap, RefractMode, RefractParams};
pub use crate::nodes::lensdistort::{LensDistortParams, LensMode, LensOutput};
pub use crate::nodes::gridwarp::{GridWarpParams, GRIDWARP_MAX};
pub use crate::nodes::warp::{WarpPair, WarpParams, WARP_PAIRS};
pub use crate::nodes::despill::{DespillMethod, DespillParams, DespillScreen};
pub use crate::nodes::lightwrap::{LightwrapParams, WrapMode};
pub use crate::nodes::upscale::{UpscaleModel, UpscaleParams, UpscaleScale};
pub use crate::nodes::seedvr::{SeedColour, SeedModel, SeedVrParams};
pub use crate::nodes::depth::{DepthModel, DepthOut, DepthParams};
pub use crate::nodes::ltxhdr::{LtxHdrParams, LtxOffload, LtxScale};
pub use crate::nodes::hdrcnn::{HdrCnnModel, HdrCnnParams};
pub use crate::nodes::stabilize::{StabMode, StabModel, StabPath, StabPathRef, StabilizeParams};
pub use crate::nodes::camtrack::{CamSolve, CamSolveRef, CamTrackParams};
pub use crate::nodes::screenkey::{ScreenKeyParams, ScreenOut};
pub use crate::nodes::dilate::{DilateParams, DILATE_MAX};
pub use crate::nodes::glow::{GlowMode, GlowParams};
pub use crate::nodes::godrays::GodraysParams;
pub use crate::nodes::lensflare::LensFlareParams;
pub use crate::nodes::edgeextend::{EdgeExtendParams, ExtendMode, EXTEND_MAX};
pub use crate::nodes::glint::GlintParams;
pub use crate::nodes::convolve::{ConvolveParams, ConvolvePreset, ConvolveRes, CONVOLVE_MAX};
pub use crate::nodes::median::{MedianParams, MEDIAN_EXACT, MEDIAN_MAX};
pub use crate::nodes::highlightrebuild::{HighlightRebuildParams, HLR_RADIUS_MAX};
pub use crate::nodes::edgedetect::{EdgeDetectParams, EdgeOp, EdgeOutput, EDGE_SIZE_MAX};
pub use crate::nodes::diffuse::DiffuseParams;
pub use crate::nodes::saturation::{SatDomain, SatLuma, SatParams};
pub use crate::nodes::contrast::{ContrastDomain, ContrastParams};
pub use crate::nodes::blank::BlankParams;
pub use crate::nodes::starfield::{StarCam, StarDist, StarfieldParams};
pub use crate::nodes::planet::{PlanetMode, PlanetNoise, PlanetParams};
pub use crate::nodes::plasma::PlasmaParams;
pub use crate::nodes::splat::SplatParams;
pub use crate::nodes::particles::{ParticlesParams, PNode};
pub use crate::nodes::bokehkernel::BokehKernelParams;
pub use crate::nodes::glintkernel::GlintKernelParams;
pub use crate::nodes::grain::{GrainApply, GrainParams, GrainStock};
pub use crate::nodes::gradient::{GradInterp, GradStop, GradientParams, GRAD_STOPS};
pub use crate::nodes::todata::ToDataParams;
pub use crate::nodes::torgba::ToRgbaParams;
pub use crate::nodes::softcrop::{CropMode, SoftCropParams};
pub use crate::nodes::motionblur::MotionBlurParams;
pub use crate::nodes::genai::GenAiParams;
pub use crate::nodes::prompt::PromptParams;
pub use crate::nodes::render3d::Render3dParams;
pub use crate::scene3d::{CameraParams, CardParams, SceneParams};
pub use crate::nodes::corridorkey::{CorridorKeyParams, CorridorOut, Screen as CorridorScreen};
pub use crate::nodes::flow::{FlowDetail, FlowEngine, FlowParams};
pub use crate::nodes::rifetime::{RifetimeMode, RifetimeParams};
pub use crate::nodes::freeze::FreezeParams;
pub use crate::nodes::framefix::{FrameFixMode, FrameFixParams};
pub use crate::nodes::vectorblur::VectorBlurParams;
pub use crate::nodes::vectorwarp::VectorWarpParams;
pub use crate::nodes::sammie::{Matting as SammieMatting, SamModel, SamPoint, SammieOut, SammieParams};

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NodeKind {
    /// EXR sequence reader. `source` indexes into the engine's reader slots
    /// (unique per Read node); `path` is the sequence directory the app loads
    /// from — a parameter, not topology (editing it never recompiles).
    /// `range` trims to an inclusive (first, last) frame-index window of the
    /// loaded sequence; the cache flags pick which tiers this node may use.
    Read {
        source: usize,
        path: String,
        #[cfg_attr(feature = "serde", serde(default))]
        range: Option<(u64, u64)>,
        /// RAM tier OFF by default: the NVMe tier is the standing cache,
        /// RAM is opt-in per read (setups predating the flag load as off).
        #[cfg_attr(feature = "serde", serde(default))]
        ram_cache: bool,
        #[cfg_attr(feature = "serde", serde(default = "yes"))]
        nvme_cache: bool,
        /// Which EXR layer to decode (None/empty = the file's own RGBA).
        /// A param, not topology: the shader is identical either way, only
        /// the pixels the reader uploads change.
        #[cfg_attr(feature = "serde", serde(default))]
        layer: Option<String>,
        /// What this plate's pixels ARE, so the comp can convert them into
        /// its working space. Defaults to `Working` — no transform — because
        /// a wrong guess here is invisible: the picture still looks like a
        /// picture, it is just the wrong one. A NEW Read is
        /// `InputSpace::Default` (the per-format table decides); a setup
        /// saved without the field predates colour management and loads as
        /// `Working`, exactly as it rendered.
        #[cfg_attr(feature = "serde", serde(default = "legacy_input"))]
        input: crate::color::InputSpace,
        /// Output ALL channels as a lazy multi-channel wire instead of one
        /// decoded layer. Editor typing only: the engine never sees it (a
        /// ChannelSelect downstream lowers to its own Read), and nothing
        /// decodes until a channel is actually selected.
        #[cfg_attr(feature = "serde", serde(default))]
        multi: bool,
        /// Comp frame the sequence's FIRST file lands on. `None` = not yet
        /// resolved — the app fills it from the file names at load (a render
        /// numbered 1001.. starts at comp 1001), and the panel can override
        /// or zero it. Composed into the reader's time map at the leaf, so
        /// ticks before it HOLD the first frame, exactly like a Slip.
        #[cfg_attr(feature = "serde", serde(default))]
        start: Option<i64>,
        /// The Read's frame range, FIRST and LAST, in comp frames. `None` =
        /// not yet known — the app fills both from the files at load (a
        /// 1001..1069 render reads 1001 / 1069) and the panel edits them.
        /// A frame inside the range with no file on disk is black and never
        /// cached; outside the range the edges below decide. Replaces the
        /// index-based `range` trim, which setups still carry and the app
        /// migrates on load.
        #[cfg_attr(feature = "serde", serde(default))]
        first: Option<i64>,
        #[cfg_attr(feature = "serde", serde(default))]
        last: Option<i64>,
        /// What the plate answers for a frame BEFORE / AFTER its own range —
        /// when the viewer (or a retime) asks for more than the clip has.
        /// The usual before/after: hold by default.
        #[cfg_attr(feature = "serde", serde(default))]
        edges: ReadEdges,
        /// Playback writes the frames it decodes into this Load's cache
        /// tiers. Off, only a precache ("cache all" / "cache in–out") fills
        /// them; the tiers are read either way.
        #[cfg_attr(feature = "serde", serde(default = "yes"))]
        cache_on_play: bool,
        /// Keep ONE channel of the plate (broadcast to all four in the
        /// shader): a quarter of the bytes in every cache tier, the staging
        /// ring and VRAM. Topology only in whether it is on (the shader
        /// broadcasts); which channel is picked on the CPU, at the cache.
        #[cfg_attr(feature = "serde", serde(default))]
        single: crate::decode::SingleChannel,
        /// Store the plate 8-bit (unorm, clamped to 0..1): half the bytes
        /// again. Lossless for an 8-bit source. Not topology at all.
        #[cfg_attr(feature = "serde", serde(default))]
        eight_bit: bool,
        /// This Load's own proxy: its frames are kept at this resolution in
        /// every cache tier, the ring and VRAM. The format stays the plate's.
        /// Skipped at the default so an old setup's bakes keep their keys.
        #[cfg_attr(
            feature = "serde",
            serde(default, skip_serializing_if = "crate::decode::LoadScale::is_full")
        )]
        scale: crate::decode::LoadScale,
    },
    Transform(TransformParams),
    Grade(GradeParams),
    /// Separable gaussian; the first non-pointwise node. Codegen splits the
    /// graph here: its input subtree renders to an intermediate texture,
    /// H/V blur passes filter it, and consumers sample the result.
    Blur(BlurParams),
    /// Flat-disc convolution (basic bokeh defocus). Direct NxN taps in one
    /// pass — O(r^2), radius capped small; same graph-splitting as Blur.
    Defocus(DefocusParams),
    /// Bloom/glow. Splits the graph like Blur, but the filter is a mip
    /// PYRAMID, not a widening convolution: threshold -> downsample chain
    /// -> weighted upsample chain. The classic compounded-blur look (a sum of
    /// exponentially spaced gaussians with falling weights) for ~1.3x the
    /// cost of a half-res pass, because each octave halves resolution
    /// instead of doubling tap count.
    Glow(GlowParams),
    /// Light shafts streaming from a source point. See
    /// [`crate::nodes::godrays`].
    Godrays(GodraysParams),
    /// A lens flare off a light point, fused over the picture. See
    /// [`crate::nodes::lensflare`].
    LensFlare(LensFlareParams),
    /// Spreads a matte's edge colours outward, or pulls them in from
    /// further inside. See [`crate::nodes::edgeextend`].
    EdgeExtend(EdgeExtendParams),
    /// Star streaks off the highlights. See [`crate::nodes::glint`].
    Glint(GlintParams),
    /// Edges by a 3x3 gradient (Sobel, Prewitt, Scharr) or a Laplacian, as
    /// a picture or a thresholded matte. See [`crate::nodes::edgedetect`].
    EdgeDetect(EdgeDetectParams),
    /// Per-channel median over a square window: noise and specks out, edges
    /// kept. See [`crate::nodes::median`].
    Median(MedianParams),
    /// Clipped highlights rebuilt from the channels that did not clip, at
    /// the colour around them. See [`crate::nodes::highlightrebuild`].
    HighlightRebuild(HighlightRebuildParams),
    /// A typed kernel of up to 7x7 weights. See [`crate::nodes::convolve`].
    Convolve(ConvolveParams),
    /// Mist and haze: light scattered into a soft halo, ramped by an
    /// optional depth pass. See [`crate::nodes::diffuse`].
    Diffuse(DiffuseParams),
    /// Combine two mattes into one — union, intersect, subtract, difference
    /// (see [`MatteOp`]). Both inputs are matte ports and so is the output,
    /// so a chain of these is matte algebra with wires: keys, splines,
    /// segments and luma mattes all speak the same stream.
    MatteCombine(MatteCombineParams),
    /// A blended with B. The op selects the formula; changing it recompiles
    /// — a different formula IS a different shader. (Called Merge until
    /// 2026-08-29; setups written then still load.)
    #[cfg_attr(feature = "serde", serde(alias = "Merge"))]
    Blend {
        #[cfg_attr(feature = "serde", serde(default))]
        op: BlendOp,
        /// The usual `mix`: dissolves the whole result back toward the B input
        /// (the untouched background), so 0 is a bypass and 1 is the full
        /// blend. A uniform, so dragging it never recompiles.
        #[cfg_attr(feature = "serde", serde(default = "one"))]
        mix: f32,
        /// Per-input multipliers, A then B — the recombine weights the
        /// graph's horseshoe sliders drag. Uniforms in the same vec4 as
        /// `mix`; 1.0 is a no-op.
        #[cfg_attr(feature = "serde", serde(default = "ones2"))]
        gain: [f32; 2],
    },
    /// Four-corner projective warp (homography), CornerPin-style.
    CornerPin(CornerPinParams),
    /// The camera's shake measured off the picture (SEA-RAFT, on a click)
    /// and taken out — or put back — as a fused corner pin. See
    /// [`crate::nodes::stabilize`].
    Stabilize(StabilizeParams),
    /// A 3D camera solved off the picture (SEA-RAFT tracks, our own
    /// bundle adjustment, on a click), read by a Camera node's tracker
    /// source. The picture passes through. See [`crate::nodes::camtrack`].
    CamTrack(CamTrackParams),
    /// Closed spline shape rasterized analytically to a single-channel mask.
    Spline(SplineParams),
    /// Brush strokes rasterized into a persistent layer texture, composited
    /// over the input. Strokes are vector params (setup-serializable, undo
    /// via graph snapshots); the layer is a memoized rasterization that only
    /// re-records when the strokes change — never a recompile.
    Paint(PaintParams),
    /// A window onto a running Blender (`crate::blender`): the scene
    /// camera's picture, live, in a per-node layer the app fills from the
    /// session. Its inputs go TO Blender as named images and are never
    /// sampled by the fused pass, exactly like a Geo node's field port.
    Blender(BlenderParams),
    /// Per-channel piecewise-linear color remap.
    Curve(CurveParams),
    /// Plugin shader node (Tier 0): behavior comes from the plugin registry
    /// (`type_id` -> WGSL + schema); the node carries only its param VALUES.
    /// A setup loaded without the plugin keeps every value and renders as a
    /// passthrough — user data never depends on a plugin being installed.
    Custom {
        type_id: String,
        #[cfg_attr(feature = "serde", serde(default))]
        params: std::collections::BTreeMap<String, f32>,
    },
    /// Multiplies the image input by the mask input's coverage.
    Mask,
    /// Luminance keyer: image in, mask out.
    LumaKey(LumaKeyParams),
    /// Colour keyer: isolates a range of COLOUR (skin, sky, foliage) as a
    /// soft mask to grade through — the secondary-correction job, not green
    /// screen. Selection is a union of soft ellipsoids in Oklab; see
    /// `crate::keyer`. Image in (input 0), optional garbage matte (input 1),
    /// mask out.
    Keyer(crate::keyer::KeyerParams),
    /// Cryptomatte: a matte of picked OBJECTS from a renderer's id layers.
    /// Image in (input 0, for the preview), then up to `RANKS` rank Reads
    /// (inputs 1..), each a Read on a `CryptoXxx00#0`-style layer that the
    /// decoder turns into exact id chunks; mask out. See `crate::cryptomatte`.
    Cryptomatte(crate::cryptomatte::CryptomatteParams),
    /// Caches its input subtree to NVMe: an explicit bake renders every
    /// timeline frame of the upstream comp to an EXR sequence, and while
    /// that bake is loaded this node samples it through its reader slot
    /// instead of evaluating the subtree. `baked` is TRANSIENT state the app
    /// sets while converting to the engine graph (authored value stays
    /// false) — flipping it is a topology change, so live↔baked recompiles.
    GraphBake {
        source: usize,
        #[cfg_attr(feature = "serde", serde(default))]
        baked: bool,
        /// The comp frame the first BAKED file holds, so a cache plays back
        /// where the comp it was baked from lives. TRANSIENT like `baked`:
        /// the app fills it from the loaded sequence's file numbers.
        #[cfg_attr(feature = "serde", serde(default, skip_serializing))]
        start: Option<i64>,
        /// The bake sits in a MATTE pipe: its ports read as Mask so it can
        /// be dropped into one (a matte bakes as grayscale and plays back
        /// identically). Follows whatever is wired into input 0 — the
        /// editor's `connect_out` keeps it current, the Dot pattern.
        #[cfg_attr(feature = "serde", serde(default))]
        mask: bool,
        /// A bake of a full CROP stores only the box: its frames are the
        /// box's size, and they play back placed where the box sits in the
        /// input's format — this is the box's top-left corner, normalised
        /// to that format. TRANSIENT like `baked`: the app sets it when the
        /// loaded frames are the crop's box.
        #[cfg_attr(feature = "serde", serde(default, skip_serializing))]
        window: Option<[f32; 2]>,
        /// A matte bake plays back as ONE 8-bit channel (a matte only ever
        /// reads red, and 1/255 steps hold nearly every matte). On, it plays
        /// back as one half-float channel instead: for a soft gradient over
        /// a large area that would band. Storage on disk is the same either
        /// way (one half channel, RLE), so switching needs no re-bake.
        #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "std::ops::Not::not"))]
        matte_half: bool,
        /// The bake's input is ONE channel (a single-channel Load, through
        /// nodes that keep it): it is stored and played back as one, and
        /// the shader broadcasts it like a matte's. TRANSIENT like `baked`:
        /// the app sets it from the input it reads the format off.
        #[cfg_attr(feature = "serde", serde(default, skip_serializing))]
        single: bool,
    },
    /// A generative-AI pipeline carried as data, run on an explicit click
    /// by the app; the picked take plays through its reader slot like a
    /// bake. See `crate::nodes::genai`.
    GenAi(GenAiParams),
    /// One step INSIDE a GenAI Pipeline's dive (a prompt, the node's input,
    /// a model call, Output…). Never appears in a comp graph: the dive
    /// writes the steps back into the owner's `genai::Pipeline`.
    GenAiStep(genai::Op),
    /// One node INSIDE a Roto's dive that is not a shape or a combine (those
    /// are plain Splines and MatteCombines in there): an Axis, or Out. Never
    /// appears in a comp graph: the dive writes them back into the owner's
    /// `SplineParams`.
    RotoNode(RotoInner),
    /// Text on the text pipe: a prompt as a node of its own.
    Prompt(PromptParams),
    /// A camera on the scene pipe (`crate::scene3d`).
    Camera3D(CameraParams),
    /// A picture on a plane, on the scene pipe.
    Card3D(CardParams),
    /// Gathers cards on the scene pipe. Saved as `CardScene`: "Scene3D" is
    /// what a long-pruned FBX node was stored as (an alias, below).
    #[cfg_attr(feature = "serde", serde(rename = "CardScene"))]
    Scene3D(SceneParams),
    /// Cards drawn through a camera; lowered by `scene3d::expand`.
    Render3D(Render3dParams),
    /// Corridor Digital's neural screen keyer, run by the app as a process;
    /// the result plays through its reader slot. See `crate::nodes::corridorkey`.
    CorridorKey(CorridorKeyParams),
    /// Click-to-roto with SAM 2 (and optionally MatAnyone), run by the app
    /// as a process; the matte plays through its reader slot. See
    /// `crate::nodes::sammie`.
    Sammie(SammieParams),
    /// Optical flow: motion vectors solved by RIFE on a click, played back
    /// through its reader slot. See `crate::nodes::flow`.
    Flow(FlowParams),
    /// Slow-down with real in-between frames, made by RIFE on a click and
    /// played back through its reader slot. See `crate::nodes::retime`.
    /// Saved as `Retime` before the rename.
    #[cfg_attr(feature = "serde", serde(alias = "Retime"))]
    Rifetime(RifetimeParams),
    /// Hold the input on one (keyable) comp frame: a time map with speed
    /// 0, no pass (`nodes::freeze`).
    Freeze(FreezeParams),
    /// Rebuild listed bad frames from their good neighbours: a live blend
    /// of two time branches, or RIFE in-betweens on a click
    /// (`nodes::framefix`).
    FrameFix(FrameFixParams),
    /// Motion blur along motion vectors. See `crate::nodes::vectorblur`.
    VectorBlur(VectorBlurParams),
    /// A picture sampled where motion vectors point. See
    /// `crate::nodes::vectorwarp`.
    VectorWarp(VectorWarpParams),
    /// Procedural grid generator (no inputs). Its own format, so a Grid can
    /// drive the comp resolution the way a Read does.
    Grid(GridParams),
    /// A colour ramp across its own format: stops along an angled axis,
    /// mixed linearly or in OkLCh. See [`crate::nodes::gradient`].
    Gradient(GradientParams),
    /// One colour across its own format: a Constant. See
    /// [`crate::nodes::blank`].
    Blank(BlankParams),
    /// A sky of stars at infinity through a camera (wired, or its own
    /// rotation and lens). See [`crate::nodes::starfield`].
    Starfield(StarfieldParams),
    /// Noise on a sphere, as a lit disc or a latlong: stars, planets.
    /// See [`crate::nodes::planet`].
    Planet(PlanetParams),
    /// GPU particles flowing along magnetic loops and streamers off a
    /// star: a corona. See [`crate::nodes::plasma`].
    Plasma(PlasmaParams),
    /// A 3D Gaussian splat capture (.ply / .spz) seen through a camera:
    /// the picture, then its depth. See [`crate::nodes::splat`].
    Splat(SplatParams),
    /// A general 2D particle system simulated on the GPU: it owns a
    /// sub-graph of [`PNode`]s (emitters, forces, renders). See
    /// [`crate::nodes::particles`].
    Particles(ParticlesParams),
    /// One node INSIDE a Particles sub-graph. Never appears in a comp graph.
    ParticleNode(PNode),
    /// A star, drawn: a Convolve's kernel for glints.
    GlintKernel(GlintKernelParams),
    /// A lens aperture, drawn: a Defocus's (or a Convolve's) kernel.
    BokehKernel(BokehKernelParams),
    /// Procedural film grain: applied to input 0, or across its own
    /// format when nothing is wired. See
    /// [`crate::nodes::grain`].
    Grain(GrainParams),
    /// Redefine the image's format: the output is `size`, and `mode` decides
    /// how the input's own format is scaled into it. The scale is a
    /// coordinate map on the SOURCE side, so it stays fused (one sample).
    Reformat(ReformatParams),
    /// Blob tracker: passes the image through untouched and analyses a small
    /// region over the sequence in an explicit BAKE (never live). The track
    /// lands on the node's TrackX/TrackY animation channels, so patch
    /// cables can drive anything from it.
    Tracker(TrackerParams),
    /// Mocha-style planar tracker: a quad on a plane, followed as a
    /// homography per frame (with a mesh over it on request). A passthrough
    /// in the shader; the results are DATA on the node — see `planar.rs`.
    PlanarTrack(PlanarTrackParams),
    /// Saturation, the way a colour corrector means it: a lerp between the
    /// pixel and its own luminance, plus the things that make Resolve's
    /// version feel like Resolve's — the log domain it actually grades in,
    /// Color Boost, and per-zone weighting. See [`crate::nodes::saturation`].
    Saturation(SatParams),
    /// Contrast about a pivot, shaped (toe, shoulder, per side), per
    /// channel or on luminance, with optional local contrast. See
    /// [`crate::nodes::contrast`].
    Contrast(ContrastParams),
    /// Levels: black and white in, lift and mult out, one straight line per
    /// channel. Compresses a range into a smaller one or expands it into a
    /// larger, and clips at neither end. See [`crate::nodes::colorcompress`].
    ColorCompress(ColorCompressParams),
    /// Every colour turned about the neutral axis by an angle, luminance
    /// held exactly. See [`crate::nodes::huerotate`].
    HueRotate(HueRotateParams),
    /// A 1D / 3D LUT file applied to the picture, shaped into the space
    /// the table expects. See [`crate::nodes::lut`].
    Lut(LutParams),
    /// A string drawn over the input from a glyph atlas. See
    /// [`crate::nodes::text`].
    Text(TextParams),
    /// The frame number printed into the picture. See
    /// [`crate::nodes::burnin`].
    Burnin(BurninParams),
    /// The picture reflected across a line it can move and turn. See
    /// [`crate::nodes::mirror`].
    Mirror(MirrorParams),
    /// The picture turned over: top to bottom, left to right, or both. See
    /// [`crate::nodes::flipflop`].
    FlipFlop(FlipFlopParams),
    /// The picture rotated in quarter turns, its format turned with it
    /// (90° swaps width and height). See [`crate::nodes::turn`].
    Turn(TurnParams),
    /// A cropped picture set back into the full format at the Crop's box,
    /// the rest transparent black. See [`crate::nodes::uncrop`].
    Uncrop(UncropParams),
    /// Two pictures, A / B or split by a wipe, scaled into the larger
    /// one's format. See [`crate::nodes::compare`].
    Compare(CompareParams),
    /// A transition: A and B warped toward each other along their optical
    /// flow while they cross-dissolve. `crate::nodes::morphdissolve`.
    MorphDissolve(MorphDissolveParams),
    /// A power curve per channel, `x^(1/γ)`, master times r/g/b (alpha its
    /// own), negatives untouched. See [`crate::nodes::gamma`].
    Gamma(GammaParams),
    /// Highlight rolloff / tone mapping on scene-linear values (`nodes::tonemap`).
    ToneMap(ToneMapParams),
    /// UV Warp: the source sampled where a UV picture points. See
    /// [`crate::nodes::uvwarp`].
    /// Saved as `StMap` before the rename.
    #[cfg_attr(feature = "serde", serde(alias = "StMap"))]
    UvWarp(UvWarpParams),
    /// A latlong as a background at infinity through a camera (wired,
    /// or its own rotation and lens). See [`crate::nodes::env`].
    Env(EnvParams),
    /// A picture through bumpy glass (`nodes::refract`).
    Refract(RefractParams),
    /// Two lattices of points, the picture moved so what sat under each
    /// source point lands on its destination point. See
    /// [`crate::nodes::gridwarp`].
    GridWarp(GridWarpParams),
    /// Brown-Conrady lens distortion, either way round, with CA and an
    /// STMap output. See [`crate::nodes::lensdistort`].
    LensDistort(LensDistortParams),
    /// Pairs of curves, the picture moved from each source curve onto its
    /// destination curve. See [`crate::nodes::warp`].
    Warp(WarpParams),
    /// A pixel written as arithmetic, one expression per channel, compiled
    /// into the fused pass. See [`crate::nodes::expression`].
    Expression(ExpressionParams),
    /// The screen colour limited back out of a keyed foreground: green or
    /// blue, average / max / min. See [`crate::nodes::despill`].
    Despill(DespillParams),
    /// The background's light bled into a band inside the foreground's
    /// edge. Foreground, background, mask. See [`crate::nodes::lightwrap`].
    Lightwrap(LightwrapParams),
    /// Neural super-resolution (`nodes::upscale`), rendered on a click.
    Upscale(UpscaleParams),
    /// HDRCNN highlight reconstruction, rendered on a click
    /// (`nodes::hdrcnn`).
    HdrCnn(HdrCnnParams),
    /// SeedVR2 diffusion restoration / upscaling, a process on a click.
    SeedVr(SeedVrParams),
    /// Video Depth Anything as a process-backed reader (`nodes::depth`).
    Depth(DepthParams),
    /// LTX-2.5 SDR-to-HDR as a process-backed reader, composited back
    /// where the input clipped (`nodes::ltxhdr`).
    LtxHdr(LtxHdrParams),
    /// A green / blue screen keyer: a colour-difference matte normalised to
    /// the sampled screen, a luma-preserving despill, premultiplied out.
    /// See [`crate::nodes::screenkey`].
    ScreenKey(ScreenKeyParams),
    /// Resolve's ColorSlice: six hue vectors and a skin vector, each with
    /// hue, saturation and density, done subtractively — more colour without
    /// more light. See [`crate::nodes::colorslice`].
    ColorSlice(ColorSliceParams),
    /// Shift the upstream readers in time by a whole number of frames.
    /// Not a shader effect: it remaps which frame each reader in its input
    /// subtree decodes (see `player::TimeMap`).
    /// A TimeOffset, named the editorial way: SLIP the upstream
    /// footage by whole frames. The playhead itself never moves.
    #[cfg_attr(feature = "serde", serde(alias = "TimeOffset"))]
    Slip(SlipParams),
    /// One of N inputs passes through; the rest are not evaluated.
    Switch(SwitchParams),
    /// Naive retime: same reader remapping as TimeOffset with a speed
    /// multiplier and nearest-frame sampling. No motion estimation and no
    /// frame blending — `RifeRetime` is the interpolating one.
    TimeRemap(TimeRemapParams),
    /// Constant rate around an anchor frame.
    Speed(SpeedParams),
    /// Shown as Time Remap: the source's in..out played over the target's
    /// in..out, the speed whatever makes them meet. (The curve-driven
    /// remap is `TimeRemap`, shown as Time Curve: setups store that key.)
    RangeRemap(RangeRemapParams),
    /// The noodle's frame range, set by hand: first and last. Metadata only
    /// — no remap, no pass; the editor elides it like a Dot, so the engine
    /// never sees one. Frames it asks for past its source's own are answered
    /// by that source's before/after (hold, loop, bounce, black).
    TimeRange(TimeRangeParams),
    /// Divide the colour by alpha (or multiply it back). Colour operations
    /// here assume PREMULTIPLIED input, so an unpremult/premult sandwich is
    /// how you grade only the opaque part of a soft edge.
    Premult {
        /// false = unpremult (divide), true = premult (multiply).
        #[cfg_attr(feature = "serde", serde(default))]
        premult: bool,
    },
    /// Rewire the channels: each output picks any input channel, or a
    /// constant. Reordering RGBA, copying alpha into the colour to look at
    /// it, killing a channel — all the same node.
    Shuffle(ShuffleParams),
    /// Render the comp above it to an EXR sequence on disk. The image passes
    /// straight through, so a Write sits inline in a chain; rendering is an
    /// explicit button, never something that happens as you scrub.
    Write(WriteParams),
    /// Multiply by a constant, through a matte. The whole node is one
    /// number: when all you want is "this much brighter, here", a Grade is
    /// four wheels of things you are not using.
    Mult(MultParams),
    /// Image in, MATTE out: picks one channel (or luma) and relabels it as
    /// coverage. Mask inputs only accept mask outputs, so this is the bridge
    /// that lets any image drive one — noise, a pass, a grade, anything.
    ToMatte(ToMatteParams),
    /// One channel of an image on the DATA pipe: full depth, no clamp.
    /// The way in to `PortKind::Data` — see `nodes::todata`.
    ToData(ToDataParams),
    /// A matte back on the image pipe, broadcast to rgba. See
    /// `nodes::torgba`.
    ToRgba(ToRgbaParams),
    /// The pixels, untouched, plus a box saying what the layer IS — which a
    /// Transform downstream sizes its handles to. See `nodes::softcrop`.
    SoftCrop(SoftCropParams),
    /// Everything upstream, averaged across a shutter: sub-frame sampling
    /// scoped to this node's input. See `nodes::motionblur`.
    MotionBlur(MotionBlurParams),
    /// The other direction: copy a matte INTO the stream as its alpha.
    /// Image on input 1, matte on input 2, colour untouched — what you reach
    /// for before a Premult, or before writing a matte out with the plate.
    MatteIn,
    /// Straight crossfade between two images on one slider — the dissolve a
    /// Blend cannot do (Blend composites; this interpolates). 0 = input 1,
    /// 1 = input 2.
    Mix(MixParams),
    /// A procedural NOISE FIELD rasterised to an image: it owns a whole
    /// sub-graph of [`crate::field::FieldNode`]s, which is a different data
    /// type from anything else here — a field is a *function* of position
    /// and seed, not a picture, so noise can drive the parameters of other
    /// noise. This node is where that collapses back into pixels.
    Noise(NoiseParams),
    /// One node INSIDE a noise sub-graph. Never appears in a comp graph:
    /// the main graph moves images, this moves fields.
    Field(crate::field::FieldNode),
    /// Procedural VECTOR GEOMETRY rasterised to an image ("2dini"): it owns
    /// a sub-graph of [`crate::geo::GeoNode`]s — a third data type, points
    /// and primitives rather than pixels or fields — evaluated on the CPU
    /// and drawn into a memoised layer the fused pass samples, on the
    /// Paint precedent. Nothing in the geometry is topology.
    Geo(GeoParams),
    /// One node INSIDE a 2dini sub-graph. Never appears in a comp graph.
    GeoNode(crate::geo::GeoNode),
    /// SIGNED DISTANCE effects: it owns a sub-graph of
    /// [`crate::sdf::SdfNode`]s — splines and shapes as distances, ops that
    /// reshape them, effects that turn a distance into colour — laid over
    /// its input. Fused, like a Noise field. See `sdf.rs`.
    Sdf(SdfParams),
    /// One node INSIDE an SDF sub-graph. Never appears in a comp graph.
    SdfNode(crate::sdf::SdfNode),
    /// Grows (positive) or shrinks (negative) a matte, in SUB-PIXEL steps:
    /// a max/min over a disc whose radius is fractional, so 0.3 px really is
    /// 0.3 px rather than a no-op that snaps to a whole pixel. Matte in,
    /// matte out — its own filter pass, like Blur.
    Dilate(DilateParams),
    /// Render the input through UV-mapped FBX geometry with the file's own
    /// (animated) camera — pure emissive, occlusion by depth buffer. One
    /// raster prepass inside the filter-stage contract; downstream samples
    /// its output like any pass boundary. The lineup node: a noise pass that
    /// follows an asset's UVs, or a frozen camera projection as a grade
    /// matte. Mesh + camera load via `crate::geoload` and are memoised per
    /// (path, mtime); every control here is a uniform.
    #[cfg_attr(feature = "serde", serde(alias = "Scene3D"))]
    /// One layer OUT of a multi-channel wire, as an ordinary hot image
    /// wire. Never reaches the engine as itself: `to_engine_graph_at`
    /// lowers it to a Read of its own reader slot with the chosen layer, so
    /// each SELECTED channel is one reader — decoded, cached (RAM and NVMe
    /// keys carry the layer) and evicted like any plate, and unselected
    /// channels cost nothing anywhere.
    ChannelSelect(ChannelSelectParams),
    /// The AOV comb as one wide node: multi-channel cable in, one output
    /// nodule per chosen layer. Lowered to its hidden ChannelSelect kids
    /// before the engine sees it (see [`BreakoutParams`]).
    Breakout(BreakoutParams),
    /// The Breakout's paired receiver: n inputs summed, fused.
    Recombine(RecombineParams),
    /// N Mask nodes as one: the image on input 0 (drawn on the LEFT), one
    /// matte per input above it, and output k is the image isolated by
    /// matte k. Lowered to plain Masks before the engine sees it
    /// ([`crate::extract`]).
    Extract(ExtractParams),
    /// The "previous iteration" tap of a loop: no inputs, output = what the
    /// paired LoopOut computed LAST iteration (the seed on iteration 0).
    /// Pair identity is this node's own graph id — a LoopOut names it.
    LoopIn,
    /// Iterate a subgraph: input 0 is the BODY (a chain that somewhere reads
    /// the paired LoopIn), input 1 the SEED priming iteration 0. The body
    /// fuses into ONE pass run `n` times, ping-ponging two textures — `n` is
    /// an encode-time count, so dragging it never recompiles. The body must
    /// stay pointwise where it depends on the LoopIn: a stage node (Blur,
    /// Glow…) reading the tap can only run once per frame, so such a loop
    /// degrades to a single body evaluation and says so in the panel.
    LoopOut(LoopParams),
    /// User-authored WGSL, edited in the app. Unlike a `Custom` plugin (one
    /// expression fused into the main pass), this node owns its passes: each
    /// image input is resolved to a real texture first, so the shader may
    /// sample anywhere and may chain up to 4 passes. See
    /// [`crate::usershader`] for the dialect. Its params are uniforms, so
    /// only EDITING THE TEXT recompiles.
    Shader(ShaderParams),
    /// A layer STACK: an arbitrary number of image inputs, each with its own
    /// transform, time offset, opacity and blend op, composited bottom-up in
    /// the one fused pass. Layer 0 is the TOP of the stack (After Effects
    /// numbering); the last layer is the bottom — the base the rest lands on.
    ///
    /// Deliberately bigger in scope than a Blend: a Blend is one formula
    /// between two images, this is a small EDIT of them. The port count
    /// follows `layers.len()`, so adding a layer IS a topology change (one
    /// recompile), while sliding, fading or moving one stays a uniform write
    /// like every other parameter.
    Layer(LayerParams),
    /// A routing anchor: one input, one output, the same pixels out as in.
    /// Purely an EDITOR construct — the app resolves wires through dots and
    /// never emits one, so a dot costs nothing on the GPU and does not move
    /// the topology hash. The arms below exist so a dot that ever did leak
    /// into an engine graph would pass its input through rather than
    /// disappear; `mask` only says which colour its wire draws in.
    Dot {
        #[cfg_attr(feature = "serde", serde(default))]
        mask: bool,
        /// The dot sits on a MULTI-CHANNEL wire: both its ends carry the
        /// fat cable, so a source chip can hand a Cryptomatte (or a
        /// Breakout, or a select) its lazy all-layers reference through a
        /// hidden wire. Derived per frame like `mask`.
        #[cfg_attr(feature = "serde", serde(default))]
        channels: bool,
        /// The dot sits on a DATA wire (a depth, say). Derived per frame
        /// like the other two; `serde(default)` so a setup written before
        /// the data pipe existed loads with its dots on the image pipe,
        /// which is where they were.
        #[cfg_attr(feature = "serde", serde(default))]
        data: bool,
        /// The dot sits on a TEXT wire (a prompt). Derived like the rest.
        #[cfg_attr(feature = "serde", serde(default))]
        text: bool,
        /// The dot sits on a VECTOR wire. Derived like the rest.
        #[cfg_attr(feature = "serde", serde(default))]
        vector: bool,
        /// The dot sits on a SCENE wire (cameras and cards).
        #[cfg_attr(feature = "serde", serde(default))]
        scene: bool,
    },
}

/// Socket glyphs for a layer stack: one character each, because the tag IS
/// the nodule (see `stalk` in the editor).
const LAYER_TAGS: [&str; LAYER_MAX] = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f", "g",
];

/// A Blender node's input ports, named as the images they become in
/// Blender (`blender::input_image_name`), so the socket says where to look.
const BLENDER_INPUT_NAMES: [&str; crate::blender::MAX_INPUTS] = [
    "wobble_in_0", "wobble_in_1", "wobble_in_2", "wobble_in_3",
    "wobble_in_4", "wobble_in_5", "wobble_in_6", "wobble_in_7",
];

/// How many layers one stack may hold. The cap is about the uniform budget
/// and the reader slots a stack of plates costs, not the shader — the fused
/// composite is a straight unrolled chain.
pub const LAYER_MAX: usize = 16;

/// Where a plugin node keeps its pipe (`NodeKind::matte_mode`) among its
/// param values. Not a legal param name, so no manifest can collide with it.
pub const CUSTOM_MATTE_KEY: &str = "~matte";

/// A layer stack. See [`NodeKind::Layer`].
#[derive(Clone, Debug, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LayerParams {
    /// Top of the stack first.
    pub layers: Vec<LayerItem>,
    /// One extra output per layer: output 0 is the comp, output `1 + i` is
    /// layer `i` ALONE — its start, speed and trim applied, no opacity, no
    /// blend op, nothing else in the stack. Lowered by
    /// [`crate::layerouts::expand`] before the engine counts, so the engine
    /// never sees it. Off by default and left out of the file while off, so
    /// a setup from before it loads and hashes exactly as it did.
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "is_false"))]
    pub outputs: bool,
}

#[cfg(feature = "serde")]
fn is_false(b: &bool) -> bool {
    !*b
}

/// One layer of a [`NodeKind::Layer`] stack: what its input image does on
/// the way into the composite.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LayerItem {
    /// Shown on the timeline bar. Empty falls back to the wired node's
    /// label, so a layer only needs a name when the graph doesn't say it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub name: String,
    /// LEGACY: layers once carried their own transform. Placement is an
    /// upstream Transform's job now, so this is READ from old setups only —
    /// the app lifts it into a real Transform node on load
    /// (`EditorGraph::lift_layer_xforms`) and it is never rendered or
    /// written back.
    #[cfg_attr(
        feature = "serde",
        serde(default, rename = "xform", skip_serializing_if = "Option::is_none")
    )]
    pub legacy_xform: Option<TransformParams>,
    /// The COMP frame this layer's source frame 0 lands on — where the bar
    /// sits on the mini timeline, and what dragging it along edits. It is a
    /// per-layer slip (the playhead never moves), stated the editorial way
    /// round: source = comp − start, which `input_time_map` is the one place
    /// that negates.
    #[cfg_attr(feature = "serde", serde(default))]
    pub start: i64,
    pub opacity: f32,
    /// Source frames per comp frame. STRETCHING a layer's bar on the mini
    /// timeline edits this: pull it twice as long and it plays at half
    /// speed. 1 is real time; it is a rate, so it is never zero.
    #[cfg_attr(feature = "serde", serde(default = "one_f32"))]
    pub speed: f32,
    /// How this layer lands on everything below it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub op: BlendOp,
    /// Comp-frame window this layer is visible in, inclusive. `None` = the
    /// whole timeline, which is what dragging a bar's ENDS trims away.
    #[cfg_attr(feature = "serde", serde(default))]
    pub trim: Option<(i64, i64)>,
    /// TRANSIENT: 1 inside the trim, 0 outside. The app stamps it while
    /// converting to the engine graph — the same trick `SpeedParams::blend_w`
    /// uses, because the engine has no idea which tick it is rendering.
    /// Never authored, never serialized, never in the topology hash.
    #[cfg_attr(feature = "serde", serde(skip, default = "one_f32"))]
    pub gate: f32,
    /// Past the clip's own frames: `None` stops a trim at them; otherwise
    /// the layer may run on and the clip holds, loops or bounces there.
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Option::is_none"))]
    pub extend: Option<LayerExtend>,
    /// Left out of the composite (the layer's own output still shows it).
    /// A uniform, folded into the opacity slot: toggling never recompiles.
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "is_false"))]
    pub mute: bool,
    /// While ANY layer of the stack is soloed, only soloed layers composite.
    /// Mute wins over solo, as in a mixing desk.
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "is_false"))]
    pub solo: bool,
}

/// How a layer runs on past its clip: `lo..=hi` is the clip in the
/// layer's SOURCE clock (the input's own frame numbers), kept by the app.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LayerExtend {
    pub edge: ReadEdge,
    pub lo: i64,
    pub hi: i64,
}

impl Default for LayerItem {
    fn default() -> Self {
        Self {
            name: String::new(),
            legacy_xform: None,
            start: 0,
            opacity: 1.0,
            speed: 1.0,
            op: BlendOp::Over,
            trim: None,
            gate: 1.0,
            extend: None,
            mute: false,
            solo: false,
        }
    }
}

impl LayerItem {
    /// The speed, guarded: a rate of zero freezes a layer on frame 0 and
    /// divides by zero everywhere a bar length is worked out from it.
    pub fn rate(&self) -> f32 {
        if self.speed.abs() < 1e-3 { 1e-3_f32.copysign(self.speed) } else { self.speed }
    }

    /// How many COMP frames `src` source frames occupy at this layer's rate.
    pub fn stretched(&self, src: i64) -> i64 {
        ((src.max(1) as f32 / self.rate().abs()).round() as i64).max(1)
    }

    /// Whether this layer contributes at comp frame `t` (its trim window).
    pub fn visible_at(&self, t: i64) -> bool {
        match self.trim {
            Some((a, b)) => t >= a.min(b) && t <= a.max(b),
            None => true,
        }
    }
}

impl LayerParams {
    /// A fresh stack: two layers, because one layer over nothing is a
    /// Transform, and a node should arrive being what it is.
    pub fn new() -> Self {
        Self { layers: vec![LayerItem::default(), LayerItem::default()], outputs: false }
    }

    /// Layer `i` alone, as its own output carries it: the same timing (the
    /// time walk reads `start` and `speed` off the stack it sits in, the
    /// trim gate is already stamped), at full opacity over nothing.
    pub fn only(&self, i: usize) -> Self {
        let mut l = self.layers.get(i).cloned().unwrap_or_default();
        l.opacity = 1.0;
        l.op = BlendOp::Over;
        l.mute = false;
        l.solo = false;
        Self { layers: vec![l], outputs: false }
    }

    /// Whether layer `i` reaches the composite under the stack's mute and
    /// solo switches (the trim is a separate, per-tick gate).
    pub fn audible(&self, i: usize) -> bool {
        let Some(l) = self.layers.get(i) else { return false };
        !l.mute && (l.solo || !self.layers.iter().take(LAYER_MAX).any(|l| l.solo))
    }
}

/// A Shader node's authored state. The source is the behaviour; `values`
/// holds param values by NAME so they survive an edit that reorders or
/// renames nothing else — exactly like `Custom`, and for the same reason:
/// user data must not depend on the shader still parsing.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ShaderParams {
    /// Change it through `set_source` only: `def()` is cached against it.
    pub source: String,
    #[cfg_attr(feature = "serde", serde(default))]
    pub values: std::collections::BTreeMap<String, [f32; 4]>,
    /// `source` compiled. `def()` is asked hundreds of times a frame (port
    /// counts, codegen, cost, the editor), and the memo behind `compile`
    /// keys on a hash of the WHOLE text: a generated shader carrying a
    /// table (the spektrafilm wobbin's 680 KB LUT) cost 1.2 s a frame in
    /// hashing alone. Cloning the node carries the cache with it.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub def_cache: ShaderDefCache,
}

/// A `ShaderParams`' compiled def. Not part of the node's identity: it is
/// derived from `source`, so it compares equal always and never saves.
#[derive(Default)]
pub struct ShaderDefCache(std::sync::OnceLock<std::sync::Arc<crate::usershader::ShaderDef>>);

impl Clone for ShaderDefCache {
    fn clone(&self) -> Self {
        let c = std::sync::OnceLock::new();
        if let Some(d) = self.0.get() {
            let _ = c.set(d.clone());
        }
        Self(c)
    }
}

impl PartialEq for ShaderDefCache {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl std::fmt::Debug for ShaderDefCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.0.get().is_some() { "ShaderDefCache(compiled)" } else { "ShaderDefCache" })
    }
}

impl Default for ShaderParams {
    fn default() -> Self {
        let def = crate::usershader::compile(crate::usershader::STARTER);
        Self {
            source: crate::usershader::STARTER.to_string(),
            values: def.defaults(),
            def_cache: ShaderDefCache::default(),
        }
    }
}

impl ShaderParams {
    pub fn def(&self) -> std::sync::Arc<crate::usershader::ShaderDef> {
        self.def_cache.0.get_or_init(|| crate::usershader::compile(&self.source)).clone()
    }

    /// A node that starts as GLSL, ready for a Matchbox shader to be pasted.
    pub fn glsl() -> Self {
        let mut p = Self { source: String::new(), values: Default::default(), def_cache: Default::default() };
        p.set_source(crate::matchbox::STARTER.to_string());
        p
    }

    /// Adopt a new source, keeping every param value the user has moved whose
    /// name survives. One still at its old default takes the new default, so
    /// a pasted shader is not handed the starter's `amount`.
    pub fn set_source(&mut self, source: String) {
        let def = crate::usershader::compile(&source);
        let old = self.def();
        let mut values = std::collections::BTreeMap::new();
        for p in &def.params {
            let kept = self.values.get(&p.name).copied().filter(|v| {
                old.param(&p.name).is_none_or(|o| o.default != *v)
            });
            values.insert(p.name.clone(), kept.unwrap_or(p.default));
        }
        self.source = source;
        self.values = values;
        self.def_cache = ShaderDefCache::default();
    }
}

#[cfg(test)]
mod shader_def_cache_tests {
    use super::*;

    #[test]
    fn the_cached_def_follows_the_source_and_the_clone() {
        let mut p = ShaderParams::default();
        let a = p.def();
        assert!(std::sync::Arc::ptr_eq(&a, &p.clone().def()), "a clone carries the compiled def");
        let src = format!("{}\n// edited", crate::usershader::STARTER);
        p.set_source(src.clone());
        let b = p.def();
        assert_eq!(b.hash, crate::usershader::source_hash(&src), "set_source drops the old def");
        assert_eq!(
            NodeKind::Shader(p.clone()).topo_tag(),
            0xC0DE_0000_0000_0000 ^ crate::usershader::source_hash(&src),
            "the topology tag is still the source's hash"
        );
        assert_eq!(p, ShaderParams { def_cache: ShaderDefCache::default(), ..p.clone() });
    }
}

/// A noise field, and the format it is rasterised into.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct NoiseParams {
    /// The node's format: a generator declares one, like Grid or fBm.
    pub size: (u32, u32),
    /// Take the comp's canvas instead of `size` (see `GridParams`). A
    /// field default, so an old setup keeps its declared size.
    #[cfg_attr(feature = "serde", serde(default))]
    pub inherit: bool,
    /// The field graph — its own data type, not a comp graph: a field has
    /// no readers, no time base and no resolution, and unlike a comp graph
    /// this one is DATA that serializes into the setup.
    pub graph: crate::field::FieldGraph,
    /// Pixels per unit of field space: the field is authored in normalized
    /// coordinates so a graph survives a change of format.
    pub scale: f32,
    /// Seed handed to the TOP of the graph. Every generator adds its own to
    /// it, so one number reshuffles the whole field.
    pub seed: f32,
    /// How fast the whole field evolves, in cycles per frame. 0 is a still
    /// texture; every generator inside scales this by its own multiplier, so
    /// one knob drives the lot and a single node can still be frozen or run
    /// at double.
    pub speed: f32,
    /// The tick this is being rendered at, resolved per frame like a Layer's
    /// gate — the engine has no idea what time it is, and a field that
    /// evolves needs to know. TRANSIENT: never authored, never saved.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub time: f32,
    /// Where the field is sampled when it is read as DATA rather than drawn:
    /// the `noise` channel evaluates the graph here on the CPU, so a patch
    /// cable can drive any ordinary parameter from it. Normalized like the
    /// field's own coordinates, and animatable, so the probe can be walked
    /// through the field over time.
    pub probe: [f32; 2],
    /// Two dimensions: red is the field, green the SAME graph at
    /// `seed + seed_offset` — one character, two uncorrelated values, so x
    /// and y can be displaced alike but not in lockstep. Evaluates the
    /// graph twice. A field default: an older setup stays one-dimensional.
    #[cfg_attr(feature = "serde", serde(default))]
    pub two_d: bool,
    /// What the second dimension adds to the seed. Seeds hash as whole
    /// numbers, so only the integer part tells.
    #[cfg_attr(feature = "serde", serde(default = "default_seed_offset"))]
    pub seed_offset: f32,
    /// What `speed` is multiplied by on its way to the phase. New nodes run
    /// at a tenth (`SPEED_UNIT`: speed 1 used to be far too quick to be
    /// useful); a setup saved before the change has no field and reads 1,
    /// so it keeps moving exactly as it did — its curves and cables on
    /// `speed` included.
    #[cfg_attr(feature = "serde", serde(default = "legacy_speed_unit"))]
    pub speed_unit: f32,
    /// ONE channel: the field in all four lanes (a Load's single channel
    /// convention), the 2d second field not evaluated. A Bake fed it stores
    /// and plays back one channel (`editor::Reduced`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub single: bool,
    /// 8 bits: every lane clamped to 0..1 and rounded to 1/255 steps, so
    /// what is drawn is exactly what an 8-bit Bake of it keeps.
    #[cfg_attr(feature = "serde", serde(default))]
    pub eight_bit: bool,
}

/// What a new Noise node multiplies its speed by (see `speed_unit`).
pub const SPEED_UNIT: f32 = 0.1;

fn default_seed_offset() -> f32 {
    101.0
}

#[cfg(feature = "serde")]
fn legacy_speed_unit() -> f32 {
    1.0
}

impl Default for NoiseParams {
    fn default() -> Self {
        // A real format and a unit scale from the start: derived defaults
        // gave a fresh node a (0, 0) format (clamped to 16 px) and scale 0,
        // which samples the field at one point — a flat picture until two
        // controls were touched.
        Self {
            size: (1920, 1080),
            inherit: false,
            graph: Default::default(),
            scale: 1.0,
            seed: 0.0,
            speed: 0.0,
            time: 0.0,
            probe: [0.0, 0.0],
            two_d: false,
            seed_offset: default_seed_offset(),
            speed_unit: SPEED_UNIT,
            single: false,
            eight_bit: false,
        }
    }
}

impl NoiseParams {
    /// The animation phase every generator inside scales by its own
    /// multiplier: the tick times the speed, in this node's units.
    pub fn phase(&self) -> f32 {
        self.time * self.speed * self.speed_unit
    }

    /// A node as the add menus make it: the defaults plus a basic fBm, so
    /// it draws noise the moment it lands.
    pub fn fresh() -> Self {
        Self { graph: crate::field::presets::basic_fbm(), ..Default::default() }
    }
}

/// A 2dini node: a geometry graph and the format it is drawn into.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct GeoParams {
    /// The node's format: a generator declares one, like Grid or Noise.
    pub size: (u32, u32),
    /// Take the comp's canvas instead of `size` (see `GridParams`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub inherit: bool,
    /// The geometry graph — DATA that serializes into the setup, like a
    /// noise field's.
    pub graph: crate::geo::GeoGraph,
    /// Seed handed to every Scatter in the graph, so one number reshuffles
    /// the whole drawing.
    pub seed: f32,
    /// The tick being rendered, stamped per frame like a Noise's: what a
    /// Noise node inside animates with and a Physics bake plays at.
    /// TRANSIENT.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub time: f32,
    /// Light (radiance cascades) and SDF effects over the drawing, worked
    /// out on the GPU after it is rasterised (`geolight`). Off by default:
    /// the layer is then drawn exactly as before.
    #[cfg_attr(feature = "serde", serde(default))]
    pub look: crate::geolight::GeoLook,
}

impl Default for GeoParams {
    fn default() -> Self {
        // A real format from the start, like Grid: a (0, 0) generator is
        // clamped to a 16 px canvas, where every shape is sub-pixel.
        Self {
            size: (1920, 1080),
            inherit: false,
            graph: Default::default(),
            seed: 0.0,
            time: 0.0,
            look: Default::default(),
        }
    }
}


/// LoopOut: which LoopIn is the feedback tap, and how many iterations.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LoopParams {
    /// Graph id of the paired LoopIn. 0 = unpaired (renders the body once).
    pub pair: NodeId,
    /// Iterations per frame, clamped to 1..=`LOOP_MAX` at encode. A float so
    /// it animates like any channel; rounded when used.
    pub n: f32,
}

impl Default for LoopParams {
    fn default() -> Self {
        Self { pair: 0, n: 4.0 }
    }
}

/// Iteration ceiling: n full-canvas passes per frame is real money, and the
/// cost readout only helps if the scale stays bounded.
pub const LOOP_MAX: f32 = 200.0;

/// ChannelSelect: which layer, through which reader slot.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ChannelSelectParams {
    /// This node's OWN reader slot (each selected channel is a reader).
    /// `usize::MAX` = not yet minted; the app assigns one before playback.
    pub source: usize,
    /// Layer name as `decode::layers()` lists it; empty = the base RGBA.
    pub layer: String,
}

impl Default for ChannelSelectParams {
    fn default() -> Self {
        Self { source: usize::MAX, layer: String::new() }
    }
}

/// Breakout: the AOV comb as ONE wide node — a harp bridge. Input 0 takes
/// the multi-channel cable; each chosen layer is an output nodule, evenly
/// pitched by `spacing`. Behind the chrome each layer is an ordinary
/// ChannelSelect KID the editor manages and hides
/// (`app::breakout::sync`), and `app::breakout::expand` re-points every
/// consumer at its kid before the engine sees the graph — so the engine,
/// the player, both caches and the cost model never learn the concept, and
/// a breakout costs exactly what the select ladder costs by hand.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BreakoutParams {
    /// Chosen layer names, one output each (`decode::layers()` spelling).
    #[cfg_attr(feature = "serde", serde(default))]
    pub layers: Vec<String>,
    /// The hidden ChannelSelect kid behind each output, index-aligned with
    /// `layers`. Editor bookkeeping — the kids are real graph nodes.
    #[cfg_attr(feature = "serde", serde(default))]
    pub kids: Vec<u64>,
    /// One more output after the layers: the REMAINDER — the base rgba
    /// minus every chosen layer, i.e. whatever the renderer failed to
    /// distribute into the AOVs, surfaced as a pass of its own. Signed on
    /// purpose: mis-distribution can overshoot, and a Recombine adding the
    /// remainder back respects the sign (adding a negative subtracts).
    #[cfg_attr(feature = "serde", serde(default))]
    pub remainder: bool,
    /// The remainder's hidden machinery when it is on: a base-rgba
    /// ChannelSelect and a signed Recombine (base minus every kid).
    #[cfg_attr(feature = "serde", serde(default))]
    pub remainder_kids: Vec<u64>,
    /// Nodule pitch in graph units at size 1 — the harp's string spacing.
    #[cfg_attr(feature = "serde", serde(default = "default_spacing"))]
    pub spacing: f32,
    /// The paired Recombine node riding x-locked below (0 = unpaired).
    #[cfg_attr(feature = "serde", serde(default))]
    pub pair: u64,
    /// Per-string x nudges (graph units at size 1), sparse off the end —
    /// the grab points on the bar. Added on top of the uniform pitch, so a
    /// harp can open a gap where a chain needs elbow room. Mirrored onto
    /// the paired Recombine so the strings hang straight.
    #[cfg_attr(feature = "serde", serde(default))]
    pub offsets: Vec<f32>,
}

/// serde default (dead without the `serde` feature, like `yes`).
#[allow(dead_code)]
fn default_spacing() -> f32 {
    90.0
}

impl Default for BreakoutParams {
    fn default() -> Self {
        Self {
            layers: Vec::new(),
            kids: Vec::new(),
            remainder: false,
            remainder_kids: Vec::new(),
            spacing: 90.0,
            pair: 0,
            offsets: Vec::new(),
        }
    }
}

impl BreakoutParams {
    /// How many strings the bar carries: the layers, plus the remainder's
    /// when it is on. Never zero — an empty bar still shows one nodule.
    pub fn strings(&self) -> usize {
        (self.layers.len() + self.remainder as usize).max(1)
    }
}

/// Where the rebuilt beauty's ALPHA comes from.
///
/// Summing AOVs rebuilds COLOUR: the passes carry the renderer's coverage
/// in their own alphas, and adding those together is meaningless (two
/// half-covered passes do not make a fully covered pixel). The original
/// rgba already holds the right answer, so that is the default — the
/// receiver takes rgb from the sum and `a` from the beauty it was broken
/// out of.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AlphaFrom {
    /// The base rgba on the receiver's porch input (the harp wires it from
    /// the same cable the strings come off). Unwired, this falls back to
    /// the sum — a hand-built Recombine behaves as it always did.
    #[default]
    Base,
    /// Whatever adding the strings produced, alpha included.
    Sum,
    /// Opaque: the rebuild covers the frame.
    Opaque,
}

impl AlphaFrom {
    pub fn label(self) -> &'static str {
        match self {
            AlphaFrom::Base => "from the rgba",
            AlphaFrom::Sum => "summed",
            AlphaFrom::Opaque => "opaque",
        }
    }

    pub const ALL: [AlphaFrom; 3] = [AlphaFrom::Base, AlphaFrom::Sum, AlphaFrom::Opaque];
}

/// Recombine: the Breakout's paired receiver — n image inputs summed in
/// the fused pass (the Add ladder as one node). A real engine node, single
/// output, no state: `n` is topology (it changes the emitted WGSL).
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RecombineParams {
    /// How many inputs — the paired Breakout's layer count, editor-synced.
    #[cfg_attr(feature = "serde", serde(default = "default_recombine_n"))]
    pub n: u32,
    /// Per-input SIGN: true = this input is subtracted. Sparse off the
    /// end (absent = added), so the plain sum stays byte-identical. What
    /// the remainder is built from: base rgba plus, every AOV minus.
    #[cfg_attr(feature = "serde", serde(default))]
    pub sub: Vec<bool>,
    /// Nodule pitch, mirrored from the pair so the strings hang straight.
    #[cfg_attr(feature = "serde", serde(default = "default_spacing"))]
    pub spacing: f32,
    /// The paired Breakout (0 = unpaired).
    #[cfg_attr(feature = "serde", serde(default))]
    pub pair: u64,
    /// Per-string multipliers, sparse off the end (absent = 1.0) — the
    /// graph's horseshoe arcs. Uniforms: dragging never recompiles.
    #[cfg_attr(feature = "serde", serde(default))]
    pub gain: Vec<f32>,
    /// Per-string x nudges, mirrored from the paired Breakout (see
    /// [`BreakoutParams::offsets`]).
    #[cfg_attr(feature = "serde", serde(default))]
    pub offsets: Vec<f32>,
    /// Where the output's alpha comes from. The strings are summed either
    /// way; this decides the fourth channel.
    #[cfg_attr(feature = "serde", serde(default))]
    pub alpha: AlphaFrom,
}

impl RecombineParams {
    /// How many STRINGS it sums.
    pub fn strings(&self) -> usize {
        self.n.clamp(1, 16) as usize
    }

    /// The porch input that carries the original rgba: one past the last
    /// string, drawn on the left bookend like a Blend's matte.
    pub fn base_input(&self) -> usize {
        self.strings()
    }

    /// Strings plus the base porch.
    pub fn inputs(&self) -> usize {
        self.strings() + 1
    }
}

/// serde default (dead without the `serde` feature, like `yes`).
#[allow(dead_code)]
fn default_recombine_n() -> u32 {
    2
}

impl Default for RecombineParams {
    fn default() -> Self {
        Self {
            n: 2,
            sub: Vec::new(),
            spacing: 90.0,
            pair: 0,
            gain: Vec::new(),
            offsets: Vec::new(),
            alpha: AlphaFrom::Base,
        }
    }
}

/// Extract: one image, N mattes, N isolated outputs. `n` is how many
/// matte/output pairs the node shows; the `+` in its panel grows it.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExtractParams {
    #[cfg_attr(feature = "serde", serde(default = "default_extract_n"))]
    pub n: u32,
}

/// Most matte/output pairs one Extract carries.
pub const EXTRACT_MAX: u32 = 32;

impl ExtractParams {
    pub fn pairs(&self) -> usize {
        self.n.clamp(1, EXTRACT_MAX) as usize
    }
}

/// serde default (dead without the `serde` feature, like `yes`).
#[allow(dead_code)]
fn default_extract_n() -> u32 {
    2
}

impl Default for ExtractParams {
    fn default() -> Self {
        Self { n: 2 }
    }
}

impl Default for DilateParams {
    fn default() -> Self {
        Self { amount: 1.0 }
    }
}

/// Crossfade between two images.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MixParams {
    /// 0 = all of input 1, 1 = all of input 2.
    pub amount: f32,
}

impl Default for MixParams {
    fn default() -> Self {
        Self { amount: 0.5 }
    }
}

/// Which channel of the image becomes the matte.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MatteChannel {
    R,
    G,
    B,
    /// The image's own alpha. Nothing else in the pipeline reads alpha as
    /// coverage, so this is the one to reach for with a premultiplied
    /// element — but it is NOT the default: most things you want as a matte
    /// (noise, a pass, a grade) have alpha 1 everywhere, and a flat matte
    /// silently does nothing at all, which is the worst way to be wrong.
    A,
    /// Rec.709 luminance of the colour. The default, because it says
    /// something useful about any image.
    #[default]
    Luma,
}

impl MatteChannel {
    pub const ALL: [MatteChannel; 5] = [
        MatteChannel::R,
        MatteChannel::G,
        MatteChannel::B,
        MatteChannel::A,
        MatteChannel::Luma,
    ];

    pub fn name(self) -> &'static str {
        match self {
            MatteChannel::R => "r",
            MatteChannel::G => "g",
            MatteChannel::B => "b",
            MatteChannel::A => "a",
            MatteChannel::Luma => "luma",
        }
    }

    /// Channel pick as a dot product, so the shader stays branch-free.
    pub fn weights(self) -> [f32; 4] {
        match self {
            MatteChannel::R => [1.0, 0.0, 0.0, 0.0],
            MatteChannel::G => [0.0, 1.0, 0.0, 0.0],
            MatteChannel::B => [0.0, 0.0, 1.0, 0.0],
            MatteChannel::A => [0.0, 0.0, 0.0, 1.0],
            MatteChannel::Luma => [0.2126, 0.7152, 0.0722, 0.0],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ToMatteParams {
    pub channel: MatteChannel,
    pub invert: bool,
    /// Values are passed straight through by default. These clamp and
    /// re-range it when the source is not already 0..1 (a scene-linear pass
    /// that goes well above 1, say).
    #[cfg_attr(feature = "serde", serde(default))]
    pub lo: f32,
    #[cfg_attr(feature = "serde", serde(default = "one_f32"))]
    pub hi: f32,
}

impl Default for ToMatteParams {
    fn default() -> Self {
        Self { channel: MatteChannel::default(), invert: false, lo: 0.0, hi: 1.0 }
    }
}

/// A constant multiply. Image on input 0, matte on input 1.
///
/// The multiplier is rgb, driven by the three-lane slider
/// (`app/src/rgbslider.rs`): ganged it is the one number it always was,
/// ungang it — or hold a colour axis — and the same node tints.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MultParams {
    #[cfg_attr(feature = "serde", serde(deserialize_with = "de_rgb_or_scalar"))]
    pub value: [f32; 3],
    /// Multiply alpha as well. OFF by default: multiplying colour alone is
    /// what makes something HOT (and breaks premultiplication on purpose);
    /// multiplying both is a fade. With an rgb multiplier the alpha rides
    /// the WORKING SPACE's luma of it, so a tinted mult still fades by the
    /// amount it brightens.
    #[cfg_attr(feature = "serde", serde(default))]
    pub alpha: bool,
    /// Which pipe it sits in: wired a MATTE it takes one and gives one
    /// (`NodeKind::follows_pipe`), the SoftCrop precedent. The app keeps it
    /// true to its input (`EditorGraph::connect_out`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub matte: bool,
}

impl Default for MultParams {
    fn default() -> Self {
        Self { value: [1.0; 3], alpha: false, matte: false }
    }
}

/// Setups written while Mult was a single float still load: a bare number
/// spreads across the three channels.
#[cfg(feature = "serde")]
fn de_rgb_or_scalar<'de, D: serde::Deserializer<'de>>(d: D) -> Result<[f32; 3], D::Error> {
    use serde::Deserialize as _;
    #[derive(serde::Deserialize)]
    #[serde(untagged)]
    enum OneOrThree {
        One(f32),
        Three([f32; 3]),
    }
    Ok(match OneOrThree::deserialize(d)? {
        OneOrThree::One(v) => [v; 3],
        OneOrThree::Three(v) => v,
    })
}

/// Where a Render (Write) puts its frames, and which ones.
///
/// A new node names its files with four fields — `location`, `name`,
/// `padding`, `subfolder` — which `pattern()` compiles into
/// `location/[name/]name.####.exr`. `path` is the old single-string form:
/// when it is set it wins, so a setup that could not be split into the new
/// fields (`WriteCompat` tries on load) still writes exactly where it did.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(from = "WriteCompat"))]
pub struct WriteParams {
    /// LEGACY: a directory, or a filename pattern with a run of `#` for the
    /// frame number (`out/comp.####.exr`). Empty on every node made since
    /// the split; see `resolve_frame`.
    pub path: String,
    /// The directory the render goes in (typed, so `~` works).
    pub location: String,
    /// The render's name: the file stem, and the subfolder's name.
    pub name: String,
    /// Digits in the frame number (4 = `1001`, `0001`).
    pub padding: u32,
    /// Put the frames in `location/name/` rather than straight in
    /// `location`.
    pub subfolder: bool,
    /// The number the FIRST rendered frame is written as, so a sequence can
    /// land on 1001 whatever the timeline calls it.
    pub start: u32,
    /// The custom window, inclusive, in comp frames. Read only in
    /// `WriteRange::Range` mode; kept when the mode moves away so a flip
    /// back finds it again.
    pub range: Option<(u64, u64)>,
    /// Which frames a render covers. `None` is a setup from before the
    /// mode existed: `mode()` derives it from `range`, so those keep
    /// rendering exactly what they did.
    pub range_mode: Option<WriteRange>,
    /// What the written files are encoded in. `None` means the comp's
    /// working space, unconverted — the right default for an EXR, which is a
    /// scene-linear container.
    ///
    /// Whatever this ends up being, it is written into the file's
    /// `chromaticities` attribute, because an EXR without one is defined to
    /// be Rec.709 and an untagged ACEScg render is therefore mislabelled
    /// rather than merely undocumented.
    pub output: Option<crate::color::Colorspace>,
    /// Follow the per-format output default (`color::FormatDefaults`)
    /// instead of `output`. New Writes do; a setup saved before the table
    /// existed loads `false` and keeps its own `output`.
    pub output_default: bool,
    /// How the EXR is compressed. The default is what every Render wrote
    /// before the pick existed (RLE in 64px tiles), byte for byte.
    pub compression: ExrCompression,
    /// Write ONE channel instead of rgba: the pick's value, half float.
    /// `None` is the full rgba picture. A data render ignores it (it is one
    /// channel already).
    pub channel: Option<WriteChannel>,
    /// What that channel is called in the file; empty = the pick's own
    /// name (`WriteChannel::default_name`).
    pub channel_name: String,
}

/// EXR compression a Render can write with: every method the `exr` crate
/// can WRITE (DWAA/DWAB it cannot).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ExrCompression {
    /// RLE, 64px tiles: the old fixed choice, fast, lossless.
    #[default]
    Rle,
    None,
    Zip1,
    Zip16,
    Piz,
    Pxr24,
    B44,
    B44a,
}

impl ExrCompression {
    pub const ALL: [ExrCompression; 8] = [
        ExrCompression::None,
        ExrCompression::Rle,
        ExrCompression::Zip1,
        ExrCompression::Zip16,
        ExrCompression::Piz,
        ExrCompression::Pxr24,
        ExrCompression::B44,
        ExrCompression::B44a,
    ];

    pub fn label(self) -> &'static str {
        match self {
            ExrCompression::None => "none",
            ExrCompression::Rle => "RLE",
            ExrCompression::Zip1 => "ZIPS (1 line)",
            ExrCompression::Zip16 => "ZIP (16 lines)",
            ExrCompression::Piz => "PIZ",
            ExrCompression::Pxr24 => "PXR24",
            ExrCompression::B44 => "B44",
            ExrCompression::B44a => "B44A",
        }
    }

    /// Does it lose anything on what a Render writes? PXR24 rounds float32
    /// to 24 bits (only a data render is float32); B44/B44A are lossy on
    /// half, which is every picture.
    pub fn lossy(self, float32: bool) -> bool {
        match self {
            ExrCompression::Pxr24 => float32,
            ExrCompression::B44 | ExrCompression::B44a => !float32,
            _ => false,
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            ExrCompression::None => "no compression: biggest files, quickest to write and read",
            ExrCompression::Rle => "run-length, tiled: fast and lossless, good on flat areas and mattes",
            ExrCompression::Zip1 => "zip per scanline: lossless, small, reads well one line at a time",
            ExrCompression::Zip16 => "zip per 16 lines: lossless and small, the usual delivery choice",
            ExrCompression::Piz => "wavelet: lossless, best on grainy, natural plates",
            ExrCompression::Pxr24 => "lossless on half; rounds float32 (a data render) to 24 bits",
            ExrCompression::B44 => "LOSSY on half: fixed 4x4 blocks, fast to play back",
            ExrCompression::B44a => "LOSSY on half: B44 that also shrinks flat areas",
        }
    }
}

/// Which one channel a single-channel Render writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum WriteChannel {
    R,
    G,
    B,
    A,
    /// Luminance in the OUTPUT space's primaries (the picture as written).
    Luma,
}

impl WriteChannel {
    pub const ALL: [WriteChannel; 5] =
        [WriteChannel::R, WriteChannel::G, WriteChannel::B, WriteChannel::A, WriteChannel::Luma];

    pub fn label(self) -> &'static str {
        match self {
            WriteChannel::R => "red",
            WriteChannel::G => "green",
            WriteChannel::B => "blue",
            WriteChannel::A => "alpha",
            WriteChannel::Luma => "luma",
        }
    }

    /// The channel's name in the file when none is given: the EXR
    /// conventions (`Y` is luminance).
    pub fn default_name(self) -> &'static str {
        match self {
            WriteChannel::R => "R",
            WriteChannel::G => "G",
            WriteChannel::B => "B",
            WriteChannel::A => "A",
            WriteChannel::Luma => "Y",
        }
    }
}

/// What a saved Render looks like on the way in: every field optional, so
/// setups from before the location/name split (just `path`) load, and are
/// split into the new fields when that names exactly the same files.
#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct WriteCompat {
    #[serde(default)]
    path: String,
    #[serde(default)]
    location: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    padding: Option<u32>,
    #[serde(default)]
    subfolder: Option<bool>,
    start: u32,
    #[serde(default)]
    range: Option<(u64, u64)>,
    #[serde(default)]
    range_mode: Option<WriteRange>,
    #[serde(default)]
    output: Option<crate::color::Colorspace>,
    #[serde(default)]
    output_default: bool,
    #[serde(default)]
    compression: ExrCompression,
    #[serde(default)]
    channel: Option<WriteChannel>,
    #[serde(default)]
    channel_name: String,
}

#[cfg(feature = "serde")]
impl From<WriteCompat> for WriteParams {
    fn from(c: WriteCompat) -> Self {
        let split = c.location.is_some() || c.name.is_some();
        let fresh = WriteParams::default();
        let mut w = WriteParams {
            path: c.path,
            location: c.location.unwrap_or_default(),
            name: c.name.unwrap_or_default(),
            padding: c.padding.unwrap_or(fresh.padding),
            subfolder: c.subfolder.unwrap_or(false),
            start: c.start,
            range: c.range,
            range_mode: c.range_mode,
            output: c.output,
            output_default: c.output_default,
            compression: c.compression,
            channel: c.channel,
            channel_name: c.channel_name,
        };
        if !split {
            if w.path.trim().is_empty() {
                // Never pointed anywhere: it may as well be a new node.
                w.name = fresh.name;
                w.subfolder = fresh.subfolder;
            } else {
                w.migrate_legacy();
            }
        }
        w
    }
}

impl Default for WriteParams {
    fn default() -> Self {
        Self {
            path: String::new(),
            location: String::new(),
            name: "comp".into(),
            padding: 4,
            subfolder: true,
            start: 1001,
            range: None,
            range_mode: Some(WriteRange::Input),
            output: None,
            output_default: true,
            compression: ExrCompression::Rle,
            channel: None,
            channel_name: String::new(),
        }
    }
}

/// Where a Write's frame range comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum WriteRange {
    /// The range its input noodle carries (`ranges.rs` in the app).
    #[default]
    Input,
    /// `WriteParams::range`, set on the node.
    Range,
    /// The script's range — the comp's own, not whichever noodle is viewed.
    Global,
}

impl WriteRange {
    pub const ALL: [WriteRange; 3] = [WriteRange::Input, WriteRange::Range, WriteRange::Global];

    pub fn label(self) -> &'static str {
        match self {
            WriteRange::Input => "from input",
            WriteRange::Range => "range",
            WriteRange::Global => "global",
        }
    }
}

impl WriteParams {
    /// The range mode in force: the authored one, else what a setup saved
    /// before modes existed meant (a window if it had one, else its input).
    pub fn mode(&self) -> WriteRange {
        self.range_mode.unwrap_or(if self.range.is_some() {
            WriteRange::Range
        } else {
            WriteRange::Input
        })
    }

    /// The frames a render covers, given what the input carries and the
    /// script's range. Range mode with no window yet falls back to the
    /// input rather than rendering nothing.
    pub fn resolve_range(&self, input: (u64, u64), global: (u64, u64)) -> (u64, u64) {
        match self.mode() {
            WriteRange::Input => input,
            WriteRange::Range => self.range.unwrap_or(input),
            WriteRange::Global => global,
        }
    }

    /// The space the files are encoded in: the node's own pick, or the
    /// table's default for this path's format; `None` in either = working.
    pub fn effective_output(
        &self,
        working: crate::color::Colorspace,
        defaults: &crate::color::FormatDefaults,
    ) -> crate::color::Colorspace {
        let out = if self.output_default { defaults.output_for(&self.pattern()) } else { self.output };
        out.unwrap_or(working)
    }

    /// Is there anywhere to write? (A legacy path, or a location.)
    pub fn is_set(&self) -> bool {
        !self.path.trim().is_empty() || !self.location.trim().is_empty()
    }

    /// The name as it goes into a path: no separators, no `#` (which would
    /// read as a frame run), `comp` when left empty.
    pub fn clean_name(&self) -> String {
        let n: String = self
            .name
            .trim()
            .chars()
            .filter(|c| *c != '#')
            .map(|c| if c == '/' || c == '\\' { '_' } else { c })
            .collect();
        if n.is_empty() { "comp".into() } else { n }
    }

    /// The whole output as one pattern, `#` standing for the frame digits:
    /// the legacy `path` when there is one, else
    /// `location/[name/]name.####.exr` — what a Load reads it back by.
    /// Not home-expanded: that happens at resolution.
    pub fn pattern(&self) -> String {
        if !self.path.trim().is_empty() {
            return self.path.clone();
        }
        let name = self.clean_name();
        let file = format!("{name}.{}.exr", "#".repeat(self.padding.clamp(1, 12) as usize));
        let mut p = std::path::PathBuf::from(self.location.trim());
        if self.subfolder {
            p.push(&name);
        }
        p.push(file);
        p.to_string_lossy().into_owned()
    }

    /// Split a legacy `path` into location / name / padding when they name
    /// EXACTLY the same files (checked, not assumed); otherwise leave it, and
    /// the node keeps writing through `path`.
    pub fn migrate_legacy(&mut self) {
        let p = self.path.trim().to_string();
        if p.is_empty() {
            return;
        }
        let path = std::path::Path::new(&p);
        let file = path.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
        let parent = path.parent().map(|d| d.to_string_lossy().into_owned()).unwrap_or_default();
        let guess = if let Some(at) = file.find('#') {
            let width = file[at..].chars().take_while(|c| *c == '#').count();
            let (head, tail) = (&file[..at], &file[at + width..]);
            match head.strip_suffix('.') {
                Some(stem) if tail == ".exr" && !stem.is_empty() => {
                    Some((parent, stem.to_string(), width as u32))
                }
                _ => None,
            }
        } else if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("exr")) {
            path.file_stem().map(|s| (parent, s.to_string_lossy().into_owned(), 4))
        } else {
            Some((p.clone(), "frame".to_string(), 4))
        };
        let Some((location, name, padding)) = guess else { return };
        let split = WriteParams {
            path: String::new(),
            location,
            name,
            padding,
            subfolder: false,
            ..self.clone()
        };
        let same = [0, 1, 12, 9999].iter().all(|&i| split.resolve_frame(i) == self.resolve_frame(i))
            && split.name == split.clean_name();
        if same {
            *self = split;
        }
    }

    /// The file frame `i` (0-based within the render) is written to.
    ///
    /// The `pattern()`'s `####` run is filled with the frame number at that
    /// width. A legacy path ending in `.exr` with no run gets the number
    /// inserted before the extension; anything else is a directory.
    pub fn resolve_frame(&self, i: u64) -> std::path::PathBuf {
        let n = self.start as u64 + i;
        // A Write path is typed, so `~` means home here too. Expanded
        // BEFORE the `#` substitution, which only ever edits the numeric
        // field and would otherwise leave the tilde in the written path.
        let expanded = crate::paths::expand_home(&self.pattern());
        let p = expanded.to_string_lossy();
        let p = p.as_ref();
        if let Some(start) = p.find('#') {
            let width = p[start..].chars().take_while(|c| *c == '#').count();
            let end = start + width;
            return std::path::PathBuf::from(format!(
                "{}{:0width$}{}",
                &p[..start],
                n,
                &p[end..],
                width = width
            ));
        }
        let path = std::path::Path::new(p);
        if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("exr")) {
            let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned());
            let dir = path.parent().map(|d| d.to_path_buf()).unwrap_or_default();
            return dir.join(format!("{}.{n:04}.exr", stem.unwrap_or_default()));
        }
        path.join(format!("frame.{n:04}.exr"))
    }

    /// The single channel to write and its name in the file, if any.
    pub fn single_channel(&self) -> Option<(WriteChannel, String)> {
        self.channel.map(|c| {
            let n = self.channel_name.trim();
            (c, if n.is_empty() { c.default_name().to_string() } else { n.to_string() })
        })
    }

    /// The directory the frames land in, which is what has to exist.
    pub fn dir(&self) -> std::path::PathBuf {
        self.resolve_frame(0)
            .parent()
            .map(|d| d.to_path_buf())
            .unwrap_or_default()
    }
}

/// Where one output channel gets its value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ShuffleSrc {
    R,
    G,
    B,
    A,
    /// Constant 0 (black, or fully transparent in alpha).
    Zero,
    /// Constant 1 (white, or fully opaque in alpha).
    One,
}

impl ShuffleSrc {
    pub const ALL: [ShuffleSrc; 6] = [
        ShuffleSrc::R,
        ShuffleSrc::G,
        ShuffleSrc::B,
        ShuffleSrc::A,
        ShuffleSrc::Zero,
        ShuffleSrc::One,
    ];

    pub fn name(self) -> &'static str {
        match self {
            ShuffleSrc::R => "r",
            ShuffleSrc::G => "g",
            ShuffleSrc::B => "b",
            ShuffleSrc::A => "a",
            ShuffleSrc::Zero => "0",
            ShuffleSrc::One => "1",
        }
    }

    /// (one-hot pick over the input channels, constant added after).
    fn weights(self) -> ([f32; 4], f32) {
        match self {
            ShuffleSrc::R => ([1.0, 0.0, 0.0, 0.0], 0.0),
            ShuffleSrc::G => ([0.0, 1.0, 0.0, 0.0], 0.0),
            ShuffleSrc::B => ([0.0, 0.0, 1.0, 0.0], 0.0),
            ShuffleSrc::A => ([0.0, 0.0, 0.0, 1.0], 0.0),
            ShuffleSrc::Zero => ([0.0; 4], 0.0),
            ShuffleSrc::One => ([0.0; 4], 1.0),
        }
    }
}

/// Which input channel feeds each of R, G, B, A.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ShuffleParams {
    pub out: [ShuffleSrc; 4],
}

impl Default for ShuffleParams {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl ShuffleParams {
    /// Straight through — a fresh Shuffle changes nothing until you point a
    /// channel somewhere else.
    pub const IDENTITY: Self = Self {
        out: [ShuffleSrc::R, ShuffleSrc::G, ShuffleSrc::B, ShuffleSrc::A],
    };

    /// The wiring as a 4x4 one-hot matrix plus the constant column, which is
    /// what the shader multiplies by. Rows are outputs, columns inputs.
    pub fn matrix(&self) -> ([[f32; 4]; 4], [f32; 4]) {
        let mut rows = [[0.0f32; 4]; 4];
        let mut consts = [0.0f32; 4];
        for (i, src) in self.out.iter().enumerate() {
            let (w, c) = src.weights();
            rows[i] = w;
            consts[i] = c;
        }
        (rows, consts)
    }

    /// Output channel labels, in packing order.
    pub const LABELS: [&'static str; 4] = ["r", "g", "b", "a"];
}

/// Procedural grid. Line spacing/width are in the node's own pixels, so the
/// grid stays put when the comp is viewed at another resolution.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GridParams {
    /// The node's format: a Grid is a source, so it declares one.
    pub size: (u32, u32),
    /// Take the comp's canvas instead of `size` (the format row's
    /// "inherit", resolved by the app like a roto shape's). Old setups
    /// have no field and keep their declared size.
    #[cfg_attr(feature = "serde", serde(default))]
    pub inherit: bool,
    /// Cell size in pixels (x, y).
    pub spacing: [f32; 2],
    /// Line thickness in pixels.
    pub width: f32,
    /// Phase offset in pixels.
    pub offset: [f32; 2],
    pub line: [f32; 4],
    pub back: [f32; 4],
}

impl Default for GridParams {
    fn default() -> Self {
        Self {
            size: (1920, 1080),
            inherit: false,
            spacing: [64.0, 64.0],
            width: 2.0,
            offset: [0.0, 0.0],
            line: [1.0, 1.0, 1.0, 1.0],
            back: [0.0, 0.0, 0.0, 0.0],
        }
    }
}

/// How a Reformat fits its input's format into the new one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ResizeMode {
    /// Scale each axis independently: the image fills the format exactly and
    /// the aspect ratio changes.
    Distort,
    /// Uniform scale, whole image inside the format (letterbox).
    #[default]
    Fit,
    /// Uniform scale, format fully covered (the overflow crops).
    Fill,
    /// No scaling at all: the input sits at its own size, centered.
    None,
}

impl ResizeMode {
    pub const ALL: [ResizeMode; 4] =
        [ResizeMode::Distort, ResizeMode::Fit, ResizeMode::Fill, ResizeMode::None];

    pub fn name(self) -> &'static str {
        match self {
            ResizeMode::Distort => "distort",
            ResizeMode::Fit => "fit",
            ResizeMode::Fill => "fill",
            ResizeMode::None => "none",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ReformatParams {
    /// The format this node outputs.
    pub size: (u32, u32),
    pub mode: ResizeMode,
    /// Extra uniform scale on top of the mode's fit (1 = none).
    #[cfg_attr(feature = "serde", serde(default = "one_f32"))]
    pub scale: f32,
    /// Centre the scaled image in the new format (off = anchor top-left).
    #[cfg_attr(feature = "serde", serde(default = "yes"))]
    pub center: bool,
    /// Size the output by `factor` x the input's format instead of `size`
    /// (0.5 = half res). The picture scales with it, so `mode` is moot.
    #[cfg_attr(feature = "serde", serde(default))]
    pub by_factor: bool,
    #[cfg_attr(feature = "serde", serde(default = "one_f32"))]
    pub factor: f32,
}

impl ReformatParams {
    /// The format this node outputs, given its input's.
    pub fn out_size(&self, src: (u32, u32)) -> (u32, u32) {
        if self.by_factor {
            let f = self.factor.abs().max(1e-4) as f64;
            let r = |v: u32| ((v as f64 * f).round() as u32).clamp(1, 32768);
            (r(src.0), r(src.1))
        } else {
            self.size
        }
    }
}

#[allow(dead_code)] // serde default (feature-gated)
fn one_f32() -> f32 {
    1.0
}

impl Default for ReformatParams {
    fn default() -> Self {
        Self { size: (1920, 1080), mode: ResizeMode::Fit, scale: 1.0, center: true, by_factor: false, factor: 0.5 }
    }
}

/// The region a Tracker follows. Everything normalized to the node's format,
/// so a track survives a resolution change.
///
/// A Tracker holds one or more POINTS. Point 0 is the fields below (and
/// the TrackX/TrackY channels), so a single-point setup is what it always
/// was; points 1.. live in `more` (channels `TrackPtX(i)`/`TrackPtY(i)`).
/// Together they solve a plane motion (`pointsolve`), read out as a corner
/// pin through `pins()` — derived from the tracked positions at whatever
/// frame is evaluated, never keyed, so correcting one point's key corrects
/// the pin.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TrackerParams {
    /// Pattern centre at the reference frame, normalized [0,1].
    pub center: [f32; 2],
    /// Pattern half-size, normalized.
    pub size: [f32; 2],
    /// How far to look for the blob each frame, normalized.
    pub search: f32,
    /// Luma threshold separating blob from background.
    pub threshold: f32,
    /// Track dark blobs on light backgrounds instead.
    #[cfg_attr(feature = "serde", serde(default))]
    pub invert: bool,
    /// How the point is found: a pattern aligned to subpixel precision
    /// (`pointtrack`), or the old thresholded blob centroid.
    #[cfg_attr(feature = "serde", serde(default))]
    pub mode: TrackMode,
    /// What a PATTERN may do between frames (translation or similarity;
    /// the point is its centre either way).
    #[cfg_attr(feature = "serde", serde(default = "track_model_default"))]
    pub model: crate::planar::Model,
    /// Points 1.., each with its own pattern, search and result.
    #[cfg_attr(feature = "serde", serde(default))]
    pub more: Vec<TrackPoint>,
    /// Every point's position at the frame the last track was seeded from:
    /// the solve's reference. Empty until something is tracked.
    #[cfg_attr(feature = "serde", serde(default))]
    pub anchors: Vec<[f32; 2]>,
    /// The picture's size in pixels when tracked: the solve is made in
    /// pixels (a rotation is only a rotation where the axes share a unit).
    #[cfg_attr(feature = "serde", serde(default))]
    pub frame: [f32; 2],
    /// The inner graph (zoom into the node): where each point's node sits,
    /// in the node's own [-1, 1] square. Missing entries get a default row.
    #[cfg_attr(feature = "serde", serde(default))]
    pub layout: Vec<[f32; 2]>,
    /// Where the Solve node sits.
    #[cfg_attr(feature = "serde", serde(default))]
    pub solve_at: Option<[f32; 2]>,
    /// The points wired into Solve, in port order. `None` = every point in
    /// order (a setup from before the wiring, and a fresh node); a point
    /// left unwired is still tracked but does not move the pin.
    #[cfg_attr(feature = "serde", serde(default))]
    pub solve_in: Option<Vec<usize>>,
    /// The tracked position, normalized. This is the node's OUTPUT: the bake
    /// writes it as a keyframe per frame, and a patch cable reads it from
    /// here to drive anything else. Nothing in the shader touches it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub track: [f32; 2],
}

impl Default for TrackerParams {
    fn default() -> Self {
        Self {
            center: [0.5, 0.5],
            size: [0.05, 0.05],
            search: 0.08,
            threshold: 0.5,
            invert: false,
            mode: TrackMode::Pattern,
            model: track_model_default(),
            track: [0.5, 0.5],
            more: Vec::new(),
            anchors: Vec::new(),
            frame: [0.0, 0.0],
            layout: Vec::new(),
            solve_at: None,
            solve_in: None,
        }
    }
}

/// One of a Tracker's points past the first.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TrackPoint {
    pub center: [f32; 2],
    pub size: [f32; 2],
    pub search: f32,
    pub track: [f32; 2],
}

/// A Tracker holds at most this many points (a corner pin needs four; a
/// similarity is overdetermined well before this).
pub const TRACK_POINTS_MAX: usize = 16;

impl TrackerParams {
    pub fn count(&self) -> usize {
        1 + self.more.len()
    }

    /// Point `i` (0 is the node's own fields).
    pub fn point(&self, i: usize) -> Option<TrackPoint> {
        if i == 0 {
            Some(TrackPoint { center: self.center, size: self.size, search: self.search, track: self.track })
        } else {
            self.more.get(i - 1).copied()
        }
    }

    pub fn set_point(&mut self, i: usize, p: TrackPoint) {
        if i == 0 {
            self.center = p.center;
            self.size = p.size;
            self.search = p.search;
            self.track = p.track;
        } else if let Some(q) = self.more.get_mut(i - 1) {
            *q = p;
        }
    }

    /// A new point beside the last one, the same size. Returns its index.
    pub fn add_point(&mut self) -> Option<usize> {
        if self.count() >= TRACK_POINTS_MAX {
            return None;
        }
        let last = self.point(self.count() - 1)?;
        let step = (last.size[0] * 2.5).max(0.02);
        let x = if last.center[0] + step < 0.95 { last.center[0] + step } else { (last.center[0] - step).max(0.05) };
        let c = [x, last.center[1]];
        self.more.push(TrackPoint { center: c, size: last.size, search: last.search, track: c });
        self.anchors.clear();
        let i = self.count() - 1;
        // A new point is wired into the solve (the default is "all of them").
        if let Some(v) = &mut self.solve_in {
            v.push(i);
        }
        Some(i)
    }

    /// Point `i`'s node in the inner graph: laid out, or a default row
    /// along the top.
    pub fn node_at(&self, i: usize) -> [f32; 2] {
        self.layout.get(i).copied().unwrap_or_else(|| {
            let n = self.count().max(1) as f32;
            [-0.8 + 1.6 * (i as f32 + 0.5) / n, -0.45]
        })
    }

    pub fn set_node_at(&mut self, i: usize, at: [f32; 2]) {
        while self.layout.len() <= i {
            let k = self.layout.len();
            let d = self.node_at(k);
            self.layout.push(d);
        }
        self.layout[i] = at;
    }

    pub fn solve_node_at(&self) -> [f32; 2] {
        self.solve_at.unwrap_or([0.0, 0.55])
    }

    /// The points driving the solve, in port order.
    pub fn solve_inputs(&self) -> Vec<usize> {
        match &self.solve_in {
            Some(v) => v.iter().copied().filter(|&i| i < self.count()).collect(),
            None => (0..self.count()).collect(),
        }
    }

    /// Wire point `i` into Solve's port `port` (a port past the last wired
    /// one appends). A point feeds the solve once: wiring it again moves it.
    pub fn wire_solve(&mut self, i: usize, port: usize) {
        if i >= self.count() {
            return;
        }
        let mut v = self.solve_inputs();
        v.retain(|&k| k != i);
        if port < v.len() {
            v[port] = i;
        } else {
            v.push(i);
        }
        self.solve_in = Some(v);
    }

    /// Take whatever feeds Solve's `port` off; later ports close up.
    pub fn unwire_solve(&mut self, port: usize) -> Option<usize> {
        let mut v = self.solve_inputs();
        let out = (port < v.len()).then(|| v.remove(port));
        self.solve_in = Some(v);
        out
    }

    /// Drop point `i`; the last point cannot go (a Tracker tracks
    /// something). Later points slide down.
    pub fn remove_point(&mut self, i: usize) -> bool {
        if self.count() <= 1 || i >= self.count() {
            return false;
        }
        if i == 0 {
            let p = self.more.remove(0);
            self.set_point(0, p);
        } else {
            self.more.remove(i - 1);
        }
        if i < self.anchors.len() {
            self.anchors.remove(i);
        }
        if i < self.layout.len() {
            self.layout.remove(i);
        }
        if let Some(v) = &mut self.solve_in {
            v.retain(|&k| k != i);
            for k in v.iter_mut() {
                if *k > i {
                    *k -= 1;
                }
            }
        }
        true
    }

    /// The corner pin the points' motion since the reference makes, in the
    /// normalised pin convention. At rest (nothing tracked) the unit quad.
    pub fn pins(&self) -> [[f32; 2]; 4] {
        let n = self.count();
        if self.anchors.len() != n || self.frame[0] <= 0.0 || self.frame[1] <= 0.0 {
            return UNIT_QUAD;
        }
        let (w, h) = (self.frame[0] as f64, self.frame[1] as f64);
        let px = |p: [f32; 2]| [p[0] as f64 * w, p[1] as f64 * h];
        // Only the points wired into Solve, in port order — four of them
        // are the pin's corners in that order.
        let used = self.solve_inputs();
        let a: Vec<[f64; 2]> = used.iter().map(|&i| px(self.anchors[i])).collect();
        let b: Vec<[f64; 2]> = used.iter().filter_map(|&i| self.point(i)).map(|p| px(p.track)).collect();
        crate::pointsolve::pins(&crate::pointsolve::solve(&a, &b), [w, h])
    }
}

/// How a Tracker finds its point.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TrackMode {
    /// A pattern around the point, aligned into every frame (ZNCC, subpixel,
    /// occlusion-aware). Works on any texture.
    #[default]
    Pattern,
    /// The luma-weighted centroid of what clears a threshold: a bright (or
    /// dark) blob on a plain ground, and nothing else.
    Blob,
}

fn track_model_default() -> crate::planar::Model {
    crate::planar::Model::Similarity
}

/// The planar tracker's setup and its results. The region is a quad drawn
/// on the frame the track is seeded from; the results are a corner pin per
/// frame (the quad carried by the frame's homography) and, with the mesh
/// on, a vertex set per frame. Everything normalised to the node's format.
/// Nothing here is a channel: the track is data, like a Paint's strokes.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PlanarTrackParams {
    /// The region at the reference frame, normalised, from the top-left.
    pub corners: [[f32; 2]; 4],
    /// What the plane may do: translation / similarity / affine / perspective.
    pub model: crate::planar::Model,
    /// Below this match score the template is re-cut from the current
    /// frame (0 never, 1 every frame).
    pub reanchor: f32,
    /// Track a mesh over the region as well.
    pub mesh: bool,
    /// Mesh cells along the region's longer side.
    pub mesh_cells: u32,
    /// Half-size of each vertex's alignment patch, in pixels.
    pub mesh_patch: u32,
    /// How hard a vertex leans on its neighbours (0 = raw local flow).
    pub mesh_smooth: f32,
    /// Track the detail layer (a per-level high-pass) rather than the raw
    /// picture. Off by default: shading that moves with the view is
    /// already discounted in the solve's weights, and the layer costs
    /// accuracy on clean footage.
    #[cfg_attr(feature = "serde", serde(default))]
    pub detail: bool,
    /// The track as a Transform hears it: the region's motion since the
    /// reference frame as a similarity — dx, dy (pixels), rotation
    /// (degrees), scale — keyed per frame by the bake on the
    /// `PlanarDX/DY/Rot/Scale` channels. Cable them into a Transform whose
    /// pivot sits on the region's reference centre.
    #[cfg_attr(feature = "serde", serde(default = "drive_rest"))]
    pub drive: [f32; 4],
    /// The track as a corner pin: where the FRAME's corners go under the
    /// frame's homography, normalised (`PlanarPin0X..3Y`); the unit square
    /// at the reference frame. Cabled into a Transform's pin or a
    /// CornerPin's corners, their input rides the plane, perspective and all.
    #[cfg_attr(feature = "serde", serde(default = "unit_quad"))]
    pub pin: [[f32; 2]; 4],
    #[cfg_attr(feature = "serde", serde(default))]
    pub track: TrackRef,
}

impl PlanarTrackParams {
    /// The plate, then the IGNORE matte: white where something in front of
    /// the plane (a hand, a pole) must sit out the track.
    pub const INPUTS: usize = 2;
    pub const MATTE_INPUT: usize = 1;
}

/// A similarity that does nothing: the drive channels at the reference.
pub const DRIVE_REST: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

#[cfg(feature = "serde")]
fn drive_rest() -> [f32; 4] {
    DRIVE_REST
}

/// The track behind an `Arc`. A graph is cloned per frame
/// (`to_engine_graph_at`), compared per frame (the undo step) and
/// serialised per frame (the render-cache key), and a track — a corner
/// pin and a mesh per frame, for the whole shot — is the bulkiest thing a
/// node can carry. Sharing makes the clone a pointer bump and the compare
/// a pointer test until something actually changes; mutation goes through
/// `get_mut` (copy on write) so an undo snapshot stays intact.
#[derive(Clone, Debug, Default)]
pub struct TrackRef(std::sync::Arc<PlanarTrackData>);

impl TrackRef {
    pub fn get_mut(&mut self) -> &mut PlanarTrackData {
        std::sync::Arc::make_mut(&mut self.0)
    }
}

impl std::ops::Deref for TrackRef {
    type Target = PlanarTrackData;
    fn deref(&self) -> &PlanarTrackData {
        &self.0
    }
}

impl PartialEq for TrackRef {
    fn eq(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.0, &other.0) || *self.0 == *other.0
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for TrackRef {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        (*self.0).serialize(s)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for TrackRef {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        PlanarTrackData::deserialize(d).map(|t| TrackRef(std::sync::Arc::new(t)))
    }
}

impl Default for PlanarTrackParams {
    fn default() -> Self {
        Self {
            corners: [[0.35, 0.3], [0.65, 0.3], [0.65, 0.7], [0.35, 0.7]],
            model: crate::planar::Model::Perspective,
            reanchor: 0.8,
            mesh: false,
            mesh_cells: 12,
            mesh_patch: 10,
            mesh_smooth: 0.6,
            detail: false,
            drive: DRIVE_REST,
            pin: UNIT_QUAD,
            track: TrackRef::default(),
        }
    }
}

/// One tracked frame, normalised to the node's format.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PlanarFrame {
    /// Comp frame.
    pub frame: i64,
    pub corners: [[f32; 2]; 4],
    /// Match score, −1..1 (1 is the reference itself).
    pub score: f32,
    /// The plane was not found; `corners` hold the previous frame's.
    pub lost: bool,
    /// The template was re-cut from this frame.
    pub reanchored: bool,
    /// Mesh vertex positions (empty without a mesh).
    #[cfg_attr(feature = "serde", serde(default))]
    pub mesh: Vec<[f32; 2]>,
}

/// A planar track's results: one entry per comp frame, kept sorted.
#[derive(Clone, Debug, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PlanarTrackData {
    /// Comp frame the region was drawn on and the track seeded from.
    pub reference: Option<i64>,
    /// The mesh's topology, vertices normalised at the reference frame.
    #[cfg_attr(feature = "serde", serde(default))]
    pub mesh: crate::planar::Mesh,
    #[cfg_attr(feature = "serde", serde(default))]
    pub frames: Vec<PlanarFrame>,
}

impl PlanarTrackData {
    pub fn at(&self, frame: i64) -> Option<&PlanarFrame> {
        self.frames.binary_search_by_key(&frame, |f| f.frame).ok().map(|i| &self.frames[i])
    }

    /// Insert or replace the frame's result.
    pub fn set(&mut self, f: PlanarFrame) {
        match self.frames.binary_search_by_key(&f.frame, |e| e.frame) {
            Ok(i) => self.frames[i] = f,
            Err(i) => self.frames.insert(i, f),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    pub fn lost(&self) -> usize {
        self.frames.iter().filter(|f| f.lost).count()
    }

    pub fn span(&self) -> Option<(i64, i64)> {
        Some((self.frames.first()?.frame, self.frames.last()?.frame))
    }
}


/// A Switch: several inputs, one of them live.
///
/// `which` is TOPOLOGY, not a uniform — it decides which subtree is
/// compiled, so the branches you are not on cost nothing to render and are
/// never even decoded. That is what makes a Switch free, and it is why this
/// param reaches `topo_tag` when the house rule says params must not: the
/// alternative, emitting every branch and selecting between them, would
/// evaluate all of them every frame. Changing `which` recompiles, but
/// pipelines are cached by topology hash, so an animated switch pays once
/// per distinct value and hits the cache from then on.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SwitchParams {
    /// Which input is live, clamped into range at use.
    pub which: u32,
    /// How many inputs the node offers (at least two — one input is a Dot).
    pub inputs: u32,
}

impl Default for SwitchParams {
    fn default() -> Self {
        Self { which: 0, inputs: 2 }
    }
}

impl SwitchParams {
    pub fn n(&self) -> usize {
        (self.inputs.max(2) as usize).min(16)
    }

    /// The live input index, always a real input.
    pub fn live(&self) -> usize {
        (self.which as usize).min(self.n() - 1)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SlipParams {
    /// Frames to shift the upstream readers by (positive = later footage).
    pub offset: i32,
}

impl Default for SlipParams {
    fn default() -> Self {
        Self { offset: 0 }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TimeInterp {
    /// Snap to the closest source frame.
    #[default]
    Nearest,
    /// Dissolve between the two frames the sub-frame time falls between.
    Blend,
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpeedParams {
    /// Constant rate: 0.5 is half speed, 2 double, negative backwards.
    pub speed: f32,
    /// The frame that stays put while the rest stretches around it.
    pub anchor: f32,
    /// Snap to a frame, or dissolve between the two it falls between.
    #[cfg_attr(feature = "serde", serde(default))]
    pub interp: TimeInterp,
    /// How far between those two frames this tick sits, filled in by the app
    /// each build (it is a function of the tick, not of the authored value).
    #[cfg_attr(feature = "serde", serde(skip))]
    pub blend_w: f32,
}

impl Default for SpeedParams {
    fn default() -> Self {
        Self { speed: 1.0, anchor: 0.0, interp: TimeInterp::Nearest, blend_w: 0.0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TimeRangeParams {
    /// In and out, in the INPUT's frame numbers.
    pub first: i64,
    pub last: i64,
    /// The frame the in point plays at (a Load's "start at"). None = where
    /// it is (`first`): metadata only, no slip. Set and different, the node
    /// lowers to a Slip by `first - at`.
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Option::is_none"))]
    pub at: Option<i64>,
}

impl Default for TimeRangeParams {
    fn default() -> Self {
        Self { first: 1001, last: 1100, at: None }
    }
}

impl TimeRangeParams {
    /// Source frame minus output frame, when "start at" moves the footage.
    pub fn slip(&self) -> Option<i64> {
        self.at.map(|a| self.first - a).filter(|o| *o != 0)
    }

    /// The frames this noodle covers, on its OWN timeline.
    pub fn output(&self) -> (i64, i64) {
        let o = self.slip().unwrap_or(0);
        (self.first - o, self.last.max(self.first) - o)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RangeRemapParams {
    /// Source frames, in and out (the upstream's own frame numbers).
    pub src: (f32, f32),
    /// Where they land on this node's timeline, in and out.
    pub dst: (f32, f32),
    #[cfg_attr(feature = "serde", serde(default))]
    pub interp: TimeInterp,
    /// The in pair is switched off: no retime, a slip that puts `src.1` on
    /// `dst.1`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub in_off: bool,
    /// The out pair is switched off: no retime, a slip that puts `src.0` on
    /// `dst.0`. Wins over `in_off` when both are set.
    #[cfg_attr(feature = "serde", serde(default))]
    pub out_off: bool,
    #[cfg_attr(feature = "serde", serde(skip))]
    pub blend_w: f32,
}

impl Default for RangeRemapParams {
    fn default() -> Self {
        Self {
            src: (1001.0, 1100.0),
            dst: (1001.0, 1100.0),
            interp: TimeInterp::Nearest,
            in_off: false,
            out_off: false,
            blend_w: 0.0,
        }
    }
}

impl RangeRemapParams {
    /// Source frames per target frame: 1 with either pair off (a slip). A
    /// target span of one frame (or a reversed one past zero) cannot be
    /// divided by: it holds.
    pub fn speed(&self) -> f32 {
        self.affine().0
    }

    /// `source = k * t + b`. Both pairs on: `src.in + (t - dst.in) * speed`.
    /// One off: a slip through the pair that is still on.
    pub fn affine(&self) -> (f32, f32) {
        if self.out_off {
            return (1.0, self.src.0 - self.dst.0);
        }
        if self.in_off {
            return (1.0, self.src.1 - self.dst.1);
        }
        let d = self.dst.1 - self.dst.0;
        let k = if d.abs() < 1e-6 { 0.0 } else { (self.src.1 - self.src.0) / d };
        (k, self.src.0 - self.dst.0 * k)
    }

    pub fn map(&self) -> TimeMap {
        let (k, b) = self.affine();
        TimeMap::affine(k, b)
    }
}

impl SpeedParams {
    /// `source = anchor + (t - anchor) * speed`, as an affine map.
    pub fn map(&self) -> TimeMap {
        TimeMap::affine(self.speed, self.anchor * (1.0 - self.speed))
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TimeRemapParams {
    /// Which source frame this timeline frame shows. One number, animated
    /// like any other parameter — the SHAPE of the remap is its curve in the
    /// curve editor, not a second editor bolted onto this node.
    pub frame: f32,
    /// The animated curve sampled into (timeline frame, source frame) points
    /// by the app, because time has to be evaluable at every tick a reader
    /// prefetches, not just the one on screen. Derived, never authored:
    /// it is rebuilt on every graph build and never persisted.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub lut: Vec<[f32; 2]>,
    /// Snap to a frame, or dissolve between the two it falls between.
    #[cfg_attr(feature = "serde", serde(default))]
    pub interp: TimeInterp,
    /// Sub-frame position between those two frames, filled per build.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub blend_w: f32,
}

impl TimeRemapParams {
    /// The map this node imposes: its sampled curve when animated, else the
    /// single frame it is parked on (a freeze).
    pub fn map(&self) -> TimeMap {
        if self.lut.is_empty() {
            TimeMap::lut(vec![[0.0, self.frame]])
        } else {
            TimeMap::lut(self.lut.clone())
        }
    }
}

/// Structural identity of a time path: which chain of time nodes lies
/// between a node and the render root. Two paths through the same time nodes
/// share a key (and therefore a reader and its decoded frame); paths through
/// different ones are evaluated separately, at their own frames.
///
/// It is deliberately built from node IDS, never from their VALUES: dragging
/// an offset changes what frame a reader sits on, never the shape of the
/// graph, so time params stay uniform-only edits and never recompile.
pub type TimeKey = u64;

/// The key of a path with no time nodes on it — the ordinary case, and the
/// one every reader falls back to.
pub const TIME_KEY_ROOT: TimeKey = 0;

/// Extend a path with one more time node, on one side of a blend. Codegen
/// and the reader walk both derive their keys through here, so the two
/// halves of a dissolve agree on which reader they are sampling.
pub fn time_key_push_phase(key: TimeKey, node: NodeId, phase: u64) -> TimeKey {
    time_key_push(key, node.wrapping_add(phase.wrapping_mul(0x9E37_79B9)))
}

/// The path key for ONE input of a per-input-time node (a Layer's layer).
/// Codegen and the reader walk both come through here, so the two can never
/// disagree about which reader a given layer samples. Note that a layer is
/// always given its own branch, whatever its offset reads as — the key must
/// never move with a parameter, or dragging an offset past zero would
/// recompile.
pub fn input_time_key(key: TimeKey, node: NodeId, input: usize) -> TimeKey {
    time_key_push_phase(key, node, input as u64)
}

/// Extend a path with one more time node.
pub fn time_key_push(key: TimeKey, node: NodeId) -> TimeKey {
    let mixed = key ^ node.wrapping_add(0x9E37_79B9_7F4A_7C15);
    mixed.rotate_left(27).wrapping_mul(0xC2B2_AE3D_27D4_EB4F) | 1
}

/// The state of one reader-instance walk: what it has found, and whether
/// it takes every input of a Switch (the rig) or only the live one (decode).
struct TimeWalk<'a> {
    every_branch: bool,
    out: &'a mut Vec<ReaderInstance>,
    seen: &'a mut std::collections::HashSet<(NodeId, TimeKey)>,
}

/// One instance of a reader-backed node: the node, the time path that
/// reaches it, and the frame mapping that path implies. A source reached
/// through two different time paths yields two instances — two reader slots,
/// two decoded frames, both live at once.
#[derive(Clone, Debug, PartialEq)]
pub struct ReaderInstance {
    pub node: NodeId,
    /// The time path — already folded with `sub` (see [`sub_key`]), so
    /// `(node, key)` stays the one slot key everywhere.
    pub key: TimeKey,
    /// Which of the node's reader slots this is. 0 for every single-slot
    /// reader; a Cryptomatte owns `cryptomatte::SOURCES` of them.
    pub sub: usize,
    pub map: TimeMap,
}

/// The time key of a node's `sub`-th reader slot on time path `key`. Sub 0
/// IS the path's key, so single-slot readers are untouched.
pub fn sub_key(key: TimeKey, sub: usize) -> TimeKey {
    match sub {
        0 => key,
        s => key ^ (s as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(23),
    }
}

/// Evaluate a lookup curve at `x`: linear between points, extrapolated along
/// the end slopes. Fewer than two points is identity (or a constant).
pub fn lut_eval(pts: &[[f32; 2]], x: f32) -> f32 {
    let slope = |a: [f32; 2], b: [f32; 2]| {
        let dx = b[0] - a[0];
        if dx.abs() < 1e-6 { 0.0 } else { (b[1] - a[1]) / dx }
    };
    match pts.len() {
        0 => x,
        1 => pts[0][1],
        n => {
            let (first, last) = (pts[0], pts[n - 1]);
            if x <= first[0] {
                return first[1] + (x - first[0]) * slope(first, pts[1]);
            }
            if x >= last[0] {
                return last[1] + (x - last[0]) * slope(pts[n - 2], last);
            }
            let i = pts.partition_point(|p| p[0] <= x).clamp(1, n - 1);
            let (a, b) = (pts[i - 1], pts[i]);
            a[1] + (x - a[0]) * slope(a, b)
        }
    }
}

/// One step of a time chain.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TimeOp {
    /// `t * speed + offset`.
    Affine { speed: f32, offset: f32 },
    /// Lookup curve, (output frame, source frame) points sorted by x.
    Lut(Vec<[f32; 2]>),
    /// A Frame Fix branch: a time on one of the `bad` frames (sorted) moves
    /// to the nearest good frame on `side` (-1 before, +1 after); any other
    /// time passes unchanged.
    Fix { bad: Vec<i64>, side: i8 },
    /// Outside `lo..=hi` the time folds back into it the way `edge` says
    /// (a layer running on past its clip). Inside it passes unchanged,
    /// sub-frame and all.
    Fold { lo: i64, hi: i64, edge: ReadEdge },
}

impl TimeOp {
    fn apply(&self, t: f32) -> f32 {
        match self {
            TimeOp::Affine { speed, offset } => t * speed + offset,
            TimeOp::Lut(pts) => lut_eval(pts, t),
            TimeOp::Fix { bad, side } => {
                let r = t.round() as i64;
                if bad.binary_search(&r).is_ok() {
                    crate::nodes::framefix::good_from(bad, r, *side) as f32
                } else {
                    t
                }
            }
            TimeOp::Fold { lo, hi, edge } => {
                let (lo, hi) = (*lo.min(hi), *lo.max(hi));
                if t >= lo as f32 - 0.5 && t < hi as f32 + 0.5 {
                    return t;
                }
                let i = t.round() as i64 - lo;
                let edges = ReadEdges { before: *edge, after: *edge };
                match ReadEdge::resolve(edges, i, (hi - lo + 1) as u64) {
                    Some(k) => (lo + k as i64) as f32,
                    None => t,
                }
            }
        }
    }

    fn is_identity(&self) -> bool {
        match self {
            TimeOp::Affine { speed, offset } => {
                (speed - 1.0).abs() < 1e-6 && offset.abs() < 1e-6
            }
            // A curve is never assumed to be identity: it is the reason the
            // branch exists, and proving it flat is not worth the risk.
            TimeOp::Lut(_) => false,
            TimeOp::Fix { bad, .. } => bad.is_empty(),
            TimeOp::Fold { .. } => false,
        }
    }
}

/// How a time node maps a timeline tick onto a source time: a chain of ops
/// applied outermost (nearest the render root) first, so a Slip above a
/// TimeWarp shifts whole source frames rather than warped ones.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct TimeMap {
    /// Applied in order; empty is identity.
    pub ops: Vec<TimeOp>,
    /// Which of the frames around the sub-frame time this path reads. Blends
    /// walk the same chain twice, once each side, and dissolve between them.
    pub phase: TimePhase,
}

/// Which whole frame a sub-frame time resolves to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TimePhase {
    #[default]
    Nearest,
    Floor,
    Ceil,
}

impl TimeMap {
    pub fn affine(speed: f32, offset: f32) -> Self {
        Self { ops: vec![TimeOp::Affine { speed, offset }], phase: TimePhase::Nearest }
    }

    pub fn lut(points: Vec<[f32; 2]>) -> Self {
        Self { ops: vec![TimeOp::Lut(points)], phase: TimePhase::Nearest }
    }

    /// The same chain read on one side of the sub-frame time.
    pub fn with_phase(mut self, phase: TimePhase) -> Self {
        self.phase = phase;
        self
    }

    pub fn is_identity(&self) -> bool {
        self.ops.iter().all(|o| o.is_identity())
    }

    /// This node's map applied on top of `outer`, the map its own output
    /// already sits under: `t_in = self(outer(tick))`. So an offset above a
    /// 2x warp still shifts by whole SOURCE frames, not doubled ones.
    pub fn then(self, outer: TimeMap) -> TimeMap {
        let mut ops = outer.ops;
        ops.extend(self.ops);
        TimeMap { ops, phase: self.phase }
    }

    /// The source TIME for a timeline tick, unrounded — what animation is
    /// sampled at, so a warp that lands between frames still moves upstream
    /// curves smoothly.
    pub fn apply_f(&self, t: f32) -> f32 {
        self.ops.iter().fold(t, |acc, op| op.apply(acc))
    }

    /// The source FRAME a reader decodes (never negative; the caller wraps it
    /// into the sequence length). Plates only exist on whole frames, so this
    /// rounds where `apply_f` does not.
    pub fn apply(&self, tick: u64) -> u64 {
        self.apply_i(tick).max(0) as u64
    }

    /// `apply`, NOT clamped at zero: a tick before a placed clip's first
    /// frame lands on a negative index, and the source's edge behaviour
    /// (`ReadEdge`) needs to know how far before.
    pub fn apply_i(&self, tick: u64) -> i64 {
        let t = self.apply_f(tick as f32);
        let f = match self.phase {
            TimePhase::Nearest => t.round(),
            TimePhase::Floor => t.floor(),
            TimePhase::Ceil => t.floor() + 1.0,
        };
        f as i64
    }

    /// How far past the floored frame this tick sits — the dissolve weight
    /// between the Floor and Ceil instances of a blended path.
    pub fn blend_weight(&self, tick: u64) -> f32 {
        let t = self.apply_f(tick as f32);
        (t - t.floor()).clamp(0.0, 1.0)
    }
}

/// What a source answers for a frame outside its own range.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ReadEdge {
    /// The nearest frame it has (first before, last after).
    #[default]
    Hold,
    /// Round again: the clip repeats.
    Loop,
    /// Back and forth: the clip plays through, then in reverse.
    Bounce,
    /// Nothing: transparent black.
    Black,
}

impl ReadEdge {
    pub const ALL: [ReadEdge; 4] = [ReadEdge::Hold, ReadEdge::Loop, ReadEdge::Bounce, ReadEdge::Black];

    pub fn label(self) -> &'static str {
        match self {
            ReadEdge::Hold => "hold",
            ReadEdge::Loop => "loop",
            ReadEdge::Bounce => "bounce",
            ReadEdge::Black => "black",
        }
    }

    /// Resolve an UNCLAMPED source index against a clip `len` frames long.
    /// `None` = black. Inside the clip every mode is the identity.
    pub fn resolve(edges: ReadEdges, i: i64, len: u64) -> Option<u64> {
        let n = len.max(1) as i64;
        if (0..n).contains(&i) {
            return Some(i as u64);
        }
        let mode = if i < 0 { edges.before } else { edges.after };
        Some(match mode {
            ReadEdge::Hold => i.clamp(0, n - 1) as u64,
            ReadEdge::Loop => i.rem_euclid(n) as u64,
            ReadEdge::Bounce => {
                if n == 1 {
                    0
                } else {
                    let p = 2 * (n - 1);
                    let k = i.rem_euclid(p);
                    (if k < n { k } else { p - k }) as u64
                }
            }
            ReadEdge::Black => return None,
        })
    }
}

/// A source's two edge behaviours.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ReadEdges {
    #[cfg_attr(feature = "serde", serde(default))]
    pub before: ReadEdge,
    #[cfg_attr(feature = "serde", serde(default))]
    pub after: ReadEdge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BlendOp {
    #[default]
    Over,
    /// A flat-out replaces B — no alpha compositing, so it is the matte
    /// input that decides where. Unmasked it is just A.
    Replace,
    Add,
    Multiply,
    Screen,
    Difference,
}

impl BlendOp {
    pub const ALL: [BlendOp; 6] = [
        BlendOp::Over,
        BlendOp::Replace,
        BlendOp::Add,
        BlendOp::Multiply,
        BlendOp::Screen,
        BlendOp::Difference,
    ];

    pub fn name(self) -> &'static str {
        match self {
            BlendOp::Over => "over",
            BlendOp::Replace => "replace",
            BlendOp::Add => "add",
            BlendOp::Multiply => "multiply",
            BlendOp::Screen => "screen",
            BlendOp::Difference => "difference",
        }
    }
}

/// How a control-point curve joins its points.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CurveInterp {
    Linear,
    /// Monotone cubic (Fritsch–Carlson tangents). A band's response has
    /// flat ends (two points are exactly a smoothstep); a Curve node's ends
    /// take the end segment's slope, so they run into the extrapolation
    /// without a kink.
    #[default]
    Smooth,
}

impl CurveInterp {
    pub const ALL: [CurveInterp; 2] = [CurveInterp::Linear, CurveInterp::Smooth];

    pub fn name(self) -> &'static str {
        match self {
            CurveInterp::Linear => "linear",
            CurveInterp::Smooth => "smooth",
        }
    }
}

/// Colour remap curve. `points` is the master (ganged) curve; with `split`
/// on, each of r/g/b follows its own — an empty per-channel curve falls back
/// to the master, so splitting an existing curve starts from where it was.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
// Container-level default: a curve saved before a field existed loads with
// THIS struct's value for it, not the field type's — which matters for
// `interp`, since every curve written before it was piecewise linear.
#[cfg_attr(feature = "serde", serde(default))]
pub struct CurveParams {
    /// (input, output) control points, sorted by x at pack time. Values
    /// outside them extrapolate, so neither axis is limited to 0..1 — a
    /// scene-linear comp grades well above white.
    pub points: Vec<[f32; 2]>,
    /// Per-channel curves (r, g, b). Empty = follow `points`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub rgb: [Vec<[f32; 2]>; 3],
    /// Drive r/g/b from their own curves instead of the master.
    #[cfg_attr(feature = "serde", serde(default))]
    pub split: bool,
    pub interp: CurveInterp,
    /// Where the curve's high point should sit: the "set" button rebuilds
    /// the curve as a straight line from the origin to (high, 1).
    /// Authoring only — the shader never reads it.
    pub high: f32,
    /// Which pipe it sits in: wired a MATTE it takes one and gives one
    /// (`NodeKind::follows_pipe`), the SoftCrop precedent. The app keeps it
    /// true to its input (`EditorGraph::connect_out`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub matte: bool,
}

impl CurveParams {
    /// The straight line a curve starts as: (0,0) to (`high`, 1), which
    /// keeps going past both ends.
    pub fn ramp(high: f32) -> Vec<[f32; 2]> {
        vec![[0.0, 0.0], [high.max(1e-4), 1.0]]
    }

    /// The curve a channel actually evaluates (0=r, 1=g, 2=b).
    pub fn channel(&self, ch: usize) -> &Vec<[f32; 2]> {
        let own = &self.rgb[ch.min(2)];
        if self.split && !own.is_empty() { own } else { &self.points }
    }
}

impl Default for CurveParams {
    fn default() -> Self {
        Self {
            // x = y, straight, extrapolating: a Curve you have not touched
            // must be a no-op at EVERY value, not just inside 0..1. Smooth
            // would bend two points into a smoothstep, so a colour curve
            // starts linear (LumaKey, which wants the S, does not).
            points: Self::ramp(1.0),
            rgb: Default::default(),
            split: false,
            interp: CurveInterp::Linear,
            high: 1.0,
            matte: false,
        }
    }
}

/// Luminance keyer: image in, MASK out. The response is a curve of luma
/// (0..2) -> coverage; the default two points are the classic lo/hi ramp,
/// and with `Smooth` that is exactly the smoothstep it always was.
/// One isolated luminance range: a response curve turning luma into
/// coverage. A [`NodeKind::LumaKey`] carries one per OUTPUT.
///
/// A classic colour-correct gives you three fixed ranges (shadows, midtones,
/// highlights) that tile the luminance axis between them. These are
/// INDEPENDENT instead — as many as you like, each with its own ends and
/// its own shape, free to overlap and free to leave gaps. Overlapping is
/// the useful case: a "shadows" and a "deep shadows" matte that share
/// territory is a thing you want and a partition cannot express.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct LumaBand {
    pub lo: f32,
    pub hi: f32,
    pub invert: bool,
    /// Extra shape between the ends. EMPTY = the plain lo/hi ramp, which is
    /// what `lo`/`hi` animate; adding points keeps them as the end handles.
    pub points: Vec<[f32; 2]>,
    /// Coverage AT `lo` and at `hi`. Defaults to `[0, 1]` — the rising ramp
    /// this band has always been — so every existing setup is unchanged.
    ///
    /// It exists because a band in the MIDDLE of a partition has to start at
    /// zero and end at zero, with its full strength in between, and a curve
    /// whose ends are pinned to 0 and 1 cannot say that. That is the whole
    /// reason a classic midtones range is not expressible as two thresholds.
    #[cfg_attr(feature = "serde", serde(default = "band_ends"))]
    pub ends: [f32; 2],
    pub interp: CurveInterp,
    /// What the editor calls this band's output port. Empty = numbered.
    pub name: String,
}

impl Default for LumaBand {
    fn default() -> Self {
        Self {
            lo: 0.2,
            hi: 0.8,
            invert: false,
            points: Vec::new(),
            ends: band_ends(),
            interp: CurveInterp::default(),
            name: String::new(),
        }
    }
}

fn band_ends() -> [f32; 2] {
    [0.0, 1.0]
}

impl LumaBand {
    /// The response as control points, ends included: `lo`/`hi` always own
    /// the two ends (so animating them still works), `points` shapes the
    /// middle. X is luma, y is coverage before `invert`.
    pub fn curve(&self) -> Vec<[f32; 2]> {
        let hi = self.hi.max(self.lo + 1e-4);
        let mut out = Vec::with_capacity(self.points.len() + 2);
        out.push([self.lo, self.ends[0]]);
        for p in &self.points {
            if p[0] > self.lo && p[0] < hi {
                out.push(*p);
            }
        }
        out.push([hi, self.ends[1]]);
        out.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    /// Port label: the authored name, or "shadows"-style defaults for the
    /// first few, or just the index.
    pub fn label(&self, i: usize) -> String {
        if !self.name.is_empty() {
            return self.name.clone();
        }
        match i {
            0 => "matte".into(),
            _ => format!("matte {}", i + 1),
        }
    }
}

/// A luminance keyer with one or more independent bands. Band `i` is
/// output `i`; a one-band key is exactly what this node has always been,
/// and is what every existing setup loads as.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(from = "LumaKeyCompat"))]
pub struct LumaKeyParams {
    /// Never empty — [`LumaKeyParams::band`] and codegen both index it, and
    /// a keyer with no output is not a thing. Constructors and the compat
    /// deserializer both guarantee at least one.
    pub bands: Vec<LumaBand>,
    /// When set, the bands are a LOCKED PARTITION of the luma axis and this
    /// is the authored truth: `bands` is derived from it by
    /// [`LumaKeyParams::sync_from_ranges`]. One boundary belongs to two
    /// bands, so moving it moves both, and the weights always sum to 1.
    ///
    /// `None` is the legacy shape — bands with independent ends, free to
    /// overlap and leave gaps. Setups authored that way keep it; nothing is
    /// silently repartitioned under someone's existing work.
    #[cfg_attr(feature = "serde", serde(default))]
    pub ranges: Option<crate::bands::BandRanges>,
}

/// Deserialization shim so setups written before bands existed still load.
/// Those carried the single band's fields FLAT on the node; a file with
/// neither shape gets one default band rather than an error.
#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
#[serde(default)]
pub struct LumaKeyCompat {
    bands: Vec<LumaBand>,
    ranges: Option<crate::bands::BandRanges>,
    lo: Option<f32>,
    hi: Option<f32>,
    invert: bool,
    points: Vec<[f32; 2]>,
    interp: CurveInterp,
}

#[cfg(feature = "serde")]
impl Default for LumaKeyCompat {
    fn default() -> Self {
        Self {
            bands: Vec::new(),
            ranges: None,
            lo: None,
            hi: None,
            invert: false,
            points: Vec::new(),
            interp: CurveInterp::default(),
        }
    }
}

#[cfg(feature = "serde")]
impl From<LumaKeyCompat> for LumaKeyParams {
    fn from(c: LumaKeyCompat) -> Self {
        if !c.bands.is_empty() {
            let mut k = Self { bands: c.bands, ranges: c.ranges };
            if k.ranges.is_none() {
                k.adopt_migrating(LUMA_SPAN);
            }
            return k;
        }
        // The legacy shape. `lo`/`hi` absent entirely means "no LumaKey
        // params were written at all", which is the default band.
        let d = LumaBand::default();
        let mut k = Self {
            bands: vec![LumaBand {
                lo: c.lo.unwrap_or(d.lo),
                hi: c.hi.unwrap_or(d.hi),
                invert: c.invert,
                points: c.points,
                // A pre-bands file is always the rising ramp.
                ends: band_ends(),
                interp: c.interp,
                name: String::new(),
            }],
            ranges: c.ranges,
        };
        if k.ranges.is_none() {
            // A pre-partition file becomes the partition it was describing,
            // rather than keeping a second model alive forever.
            k.adopt_migrating(LUMA_SPAN);
        }
        k
    }
}

#[cfg(feature = "serde")]
impl LumaKeyParams {
    /// `adopt_ranges` for a key loaded from a pre-partition file, saying so
    /// when the migration cannot keep everything. A hand-shaped ramp's
    /// points sit ON the crossfade, and a locked partition has no place for
    /// a point there (the crossfade is the boundary both bands share), so
    /// the key comes back as the plain ramp between the same ends. That is a
    /// real loss of someone's work and must not be silent.
    fn adopt_migrating(&mut self, span: (f32, f32)) {
        let shaped: usize = self.bands.iter().map(|b| b.points.len()).sum();
        self.adopt_ranges(span);
        if shaped > 0 {
            log::warn!(
                "LumaKey from an older setup: its hand-shaped ramp ({shaped} point(s)) cannot \
                 be kept in the locked-band model, so it loads as a plain ramp between the same \
                 ends — reshape it in the band editor"
            );
        }
    }
}

impl LumaKeyParams {
    /// Band `i`, clamped to the last one. Never panics: `bands` is never
    /// empty, and a consumer asking for a band that a param edit has since
    /// removed should get a matte, not a crash.
    pub fn band(&self, i: usize) -> &LumaBand {
        let i = i.min(self.bands.len().saturating_sub(1));
        &self.bands[i]
    }

    pub fn band_mut(&mut self, i: usize) -> &mut LumaBand {
        let i = i.min(self.bands.len().saturating_sub(1));
        &mut self.bands[i]
    }

    /// Band 0's response. Codegen only ever sees ONE-band keys (see
    /// [`crate::lumabands`]), so this is the band it packs.
    pub fn curve(&self) -> Vec<[f32; 2]> {
        self.band(0).curve()
    }

    /// Rebuild `bands` from `ranges`. Call after every edit to the
    /// partition; it is the only thing that writes derived band shapes, so
    /// the editor's picture and the packed curve cannot drift.
    ///
    /// Names and `invert` are the band's OWN — they survive a boundary move,
    /// a split, or a dissolve, because they are what the artist attached to
    /// that output rather than anything the partition implies.
    pub fn sync_from_ranges(&mut self) {
        let Some(r) = self.ranges.clone() else { return };
        let n = r.band_count();
        while self.bands.len() < n {
            let name = String::new();
            self.bands.push(LumaBand { name, ..LumaBand::default() });
        }
        self.bands.truncate(n.max(1));
        let interp = match r.curve {
            crate::bands::BandCurve::Linear => CurveInterp::Linear,
            crate::bands::BandCurve::Smooth => CurveInterp::Smooth,
        };
        for (i, b) in self.bands.iter_mut().enumerate() {
            let pts = crate::bands::band_curve(&r, i);
            b.lo = pts[0][0];
            b.hi = pts[pts.len() - 1][0].max(b.lo + 1e-4);
            b.ends = [pts[0][1], pts[pts.len() - 1][1]];
            b.points = pts[1..pts.len().saturating_sub(1)].to_vec();
            b.interp = interp;
        }
    }

    /// Turn a legacy key into a locked partition, keeping the response it
    /// already had.
    ///
    /// The ordinary single-band key is the case that matters, and it is not
    /// "one band": a threshold is TWO bands with one boundary — below and
    /// above. So it migrates to exactly that, with the boundary sitting on
    /// the old ramp (centre and width taken from `lo`/`hi`), and band 0's
    /// `invert` flipped, because band 0 is the falling side and the old
    /// ramp rose. Output 0 therefore keeps its meaning, which matters: it is
    /// what every existing wire into this node is already reading.
    ///
    /// The bonus is that the complement arrives as output 1 for free.
    pub fn adopt_ranges(&mut self, span: (f32, f32)) {
        let old: Vec<LumaBand> = self.bands.clone();
        let mut r = if old.len() <= 1 {
            let b = old.first().cloned().unwrap_or_default();
            let hi = b.hi.max(b.lo + 1e-4);
            let mut r = crate::bands::BandRanges::even(2, span, false, 0.0);
            r.edges[0] = crate::bands::Edge::new(0.5 * (b.lo + hi), hi - b.lo);
            r
        } else {
            let mut r = crate::bands::BandRanges::even(old.len(), span, false, 0.5);
            for (i, e) in r.edges.iter_mut().enumerate() {
                if let Some(b) = old.get(i) {
                    e.at = 0.5 * (b.lo + b.hi.max(b.lo + 1e-4));
                    e.width = (b.hi - b.lo).abs();
                }
            }
            r
        };
        r.normalize();
        self.ranges = Some(r);
        self.sync_from_ranges();
        if old.len() <= 1 {
            let was = old.first().map(|b| b.invert).unwrap_or(false);
            if let Some(b) = self.bands.get_mut(0) {
                b.invert = !was;
                b.name = old.first().map(|o| o.name.clone()).unwrap_or_default();
            }
        } else {
            for (i, o) in old.iter().enumerate() {
                if let Some(b) = self.bands.get_mut(i) {
                    b.invert = o.invert;
                    b.name = o.name.clone();
                }
            }
        }
    }

    pub fn interp(&self) -> CurveInterp {
        self.band(0).interp
    }

    pub fn invert(&self) -> bool {
        self.band(0).invert
    }

    /// Split band `i` at its centre. The new band is the UPPER half, at
    /// `i + 1`, and comes back unnamed; every band above it moves up one
    /// index carrying its own attributes (name, invert) with it. Returns the
    /// new band's index.
    ///
    /// `sync_from_ranges` alone cannot do this: it re-derives geometry by
    /// index and appends the extra band at the END, so a split in the
    /// middle left every band above wearing the name of the one below it.
    pub fn split_band(&mut self, i: usize) -> usize {
        let Some(r) = self.ranges.as_mut() else { return i };
        let i = i.min(r.band_count().saturating_sub(1));
        let at = r.band_center(i);
        r.split_at(at);
        self.sync_from_ranges();
        // The appended band's own attributes are the blank ones; walk them
        // down to `i + 1` so everything above keeps what it had.
        let new = i + 1;
        let last = self.bands.len().saturating_sub(1);
        if new < last {
            let mut own: Vec<(String, bool)> =
                self.bands[new..].iter().map(|b| (b.name.clone(), b.invert)).collect();
            own.rotate_right(1);
            for (b, (name, invert)) in self.bands[new..].iter_mut().zip(own) {
                b.name = name;
                b.invert = invert;
            }
        }
        new.min(last)
    }

    /// Dissolve band `i` into its neighbour (the one above, or below for the
    /// last band). The neighbour keeps its own attributes; the dissolved
    /// band's go with it. A one-band key stays as it is.
    pub fn dissolve_band(&mut self, i: usize) {
        let Some(r) = self.ranges.as_mut() else { return };
        let n = r.band_count();
        if n <= 1 {
            return;
        }
        let i = i.min(n - 1);
        r.remove_band(i);
        if i < self.bands.len() {
            self.bands.remove(i);
        }
        self.sync_from_ranges();
    }

    /// This key reduced to a single band — what [`crate::lumabands`] gives
    /// each consumer of a multi-band node.
    pub fn only(&self, i: usize) -> Self {
        // The single band is already DERIVED; the partition it came from is
        // authoring state the expansion has no use for, and carrying it
        // would make the clone re-derive itself back into many bands.
        Self { bands: vec![self.band(i).clone()], ranges: None }
    }
}

impl Default for LumaKeyParams {
    fn default() -> Self {
        // A fresh key is the key it has always been — `LumaBand::default()`'s
        // 0.2..0.8 rising ramp — migrated into the partition model: exactly
        // what loading a setup with no LumaKey params at all gives
        // (`LumaKeyCompat` -> `adopt_ranges`). It used to be a ONE-band
        // partition, which has no boundary and so is flat at full coverage:
        // it keyed nothing, and a Grade masked by a fresh key graded the
        // whole frame.
        let mut k = Self { bands: vec![LumaBand::default()], ranges: None };
        k.adopt_ranges(LUMA_SPAN);
        k
    }
}

/// The luminance axis the key is cut on. 0..2 rather than 0..1 because the
/// comp is scene-linear and a highlight band has to reach above white.
pub const LUMA_SPAN: (f32, f32) = (0.0, 2.0);

/// Points a Grade zone's weight curve can carry in the uniform block. A
/// partition band is four corners; the rest is the user's bias points. The
/// capacity is FIXED so the span never moves with the curve's shape —
/// adding a point to a zone is a uniform write, and only the zone count
/// recompiles — and the band editor stops adding points at it rather than
/// let the packer drop the top of the band.
pub const ZONE_PTS: usize = 16;

/// How long a stroke lives. Dust-busting paints on ONE frame; a rig removal
/// covers a section; a logo patch covers everything — so the lifetime is per
/// STROKE, not per node, and the node only carries the default new ones get.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StrokeLife {
    /// Every frame.
    #[default]
    All,
    /// Exactly the frame it was painted on.
    Single(u64),
    /// Inclusive range.
    Range(u64, u64),
}

impl StrokeLife {
    pub fn covers(self, frame: u64) -> bool {
        match self {
            StrokeLife::All => true,
            StrokeLife::Single(f) => f == frame,
            StrokeLife::Range(a, b) => frame >= a.min(b) && frame <= a.max(b),
        }
    }

    /// This lifetime as a NEW stroke painted at `frame` should get it.
    ///
    /// Only `Single` follows the playhead — "this frame" means the frame you
    /// are looking at, so the stored number is a placeholder. A `Range` is
    /// two frame numbers the user TYPED into the panel's in/out fields, so
    /// it is taken as authored: painting at frame 50 with in=10 out=20 gives
    /// a stroke that lives 10..20, not 50..60.
    pub fn at(self, frame: u64) -> Self {
        match self {
            StrokeLife::Single(_) => StrokeLife::Single(frame),
            other => other,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            StrokeLife::All => "all frames",
            StrokeLife::Single(_) => "this frame",
            StrokeLife::Range(..) => "range",
        }
    }
}

/// What a stroke DOES to the pixels under it. Grouped by what it costs the
/// fused pass, which is the thing that actually matters here:
///
/// * `Color`, `Dodge`, `Burn` — pointwise on the pixel already computed.
///   Free.
/// * `Clone`, `Smear` — the node's input read ONCE MORE, at an offset the
///   stroke carries (clone: where you alt-clicked; smear: back along the
///   stroke). The whole upstream chain again, per pixel.
/// * `Blur` — the input read [`crate::paint::PAINT_BLUR_TAPS`] more times.
///   The one genuinely heavy brush.
///
/// Each class is a bit in the node's topology tag: a node only compiles the
/// reads its strokes actually need, and a node that has only ever painted
/// colour costs exactly what it did before any of this existed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BrushMode {
    #[default]
    Color,
    Clone,
    Smear,
    Blur,
    Dodge,
    Burn,
}

/// Which stroke layer a mode rasterizes into — see `paint::PaintStamps`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrushClass {
    /// Premultiplied colour, composited over.
    Color,
    /// A source offset: where to fetch the input FROM (clone, smear).
    Source,
    /// An amount: exposure in stops and/or a blur radius (dodge, burn, blur).
    Effect,
}

impl BrushMode {
    pub const ALL: [BrushMode; 6] = [
        BrushMode::Color,
        BrushMode::Clone,
        BrushMode::Smear,
        BrushMode::Blur,
        BrushMode::Dodge,
        BrushMode::Burn,
    ];

    pub fn name(self) -> &'static str {
        match self {
            BrushMode::Color => "colour",
            BrushMode::Clone => "clone",
            BrushMode::Smear => "smear",
            BrushMode::Blur => "blur",
            BrushMode::Dodge => "dodge",
            BrushMode::Burn => "burn",
        }
    }

    pub fn class(self) -> BrushClass {
        match self {
            BrushMode::Color => BrushClass::Color,
            BrushMode::Clone | BrushMode::Smear => BrushClass::Source,
            BrushMode::Blur | BrushMode::Dodge | BrushMode::Burn => BrushClass::Effect,
        }
    }

    /// What `PaintStroke::strength` means for this mode, for the panel.
    pub fn strength_label(self) -> Option<&'static str> {
        match self {
            BrushMode::Blur => Some("blur px"),
            BrushMode::Smear => Some("drag px"),
            BrushMode::Dodge | BrushMode::Burn => Some("stops"),
            BrushMode::Color | BrushMode::Clone => None,
        }
    }
}

/// How a COLOUR stroke lands on what is under it (`paint::PAINT_WGSL` has
/// the blend states). Every one of them is an affine map per channel,
/// `x -> x * m + b`, and a chain of affine maps is one affine map — so the
/// colour layer stores the composed `(M, B)` and the fused pass applies
/// `below * M + B` once. That is what keeps stroke ORDER across modes (a
/// multiply laid after a normal stroke darkens it; one laid before does
/// not) without the layer ever reading the plate, so it stays memoized
/// however the plate moves.
///
/// `Normal` and `Erase` only ever scale by the stroke's coverage, so `M` is
/// `1 - B.a` and the node needs one texture, exactly as before blend modes
/// existed. The others make `M` a colour, which costs a second layer — a
/// topology bit, `PaintParams::has_blend`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PaintBlend {
    #[default]
    Normal,
    /// Takes earlier PAINT away (back towards the input). Never the plate.
    Erase,
    Add,
    Subtract,
    Multiply,
    Screen,
}

impl PaintBlend {
    pub const ALL: [PaintBlend; 6] = [
        PaintBlend::Normal,
        PaintBlend::Erase,
        PaintBlend::Add,
        PaintBlend::Subtract,
        PaintBlend::Multiply,
        PaintBlend::Screen,
    ];

    pub fn name(self) -> &'static str {
        match self {
            PaintBlend::Normal => "normal",
            PaintBlend::Erase => "erase",
            PaintBlend::Add => "add",
            PaintBlend::Subtract => "subtract",
            PaintBlend::Multiply => "multiply",
            PaintBlend::Screen => "screen",
        }
    }

    /// The id the raster shader switches on (`Stamp::extra[0]`).
    pub fn id(self) -> u32 {
        self as u32
    }

    /// Does this blend need the colour layer's MULTIPLIER texture? Normal
    /// and erase scale by coverage alone, which the one layer's alpha holds.
    pub fn needs_mul(self) -> bool {
        !matches!(self, PaintBlend::Normal | PaintBlend::Erase)
    }
}

/// How a pen's pressure shapes a stroke. Each amount is how far pressure
/// pulls its target from full (0 = ignored, 1 = a feather touch is
/// nothing), and `gamma` bends the response: above 1 the light end is
/// softer, below 1 a light touch already lands hard.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PressureMap {
    pub size: f32,
    pub opacity: f32,
    pub gamma: f32,
}

impl Default for PressureMap {
    /// Pressure moves the size, like a pencil — and is NOT the default for
    /// strokes from before pressure existed (they carry no samples, which
    /// is full pressure whatever the map says).
    fn default() -> Self {
        Self { size: 1.0, opacity: 0.0, gamma: 1.0 }
    }
}

impl PressureMap {
    pub const OFF: PressureMap = PressureMap { size: 0.0, opacity: 0.0, gamma: 1.0 };

    /// The pressure after the curve, 0..1.
    pub fn curve(&self, p: f32) -> f32 {
        p.clamp(0.0, 1.0).powf(self.gamma.clamp(0.1, 10.0))
    }

    /// (radius factor, opacity factor) at raw pressure `p`.
    pub fn factors(&self, p: f32) -> (f32, f32) {
        let c = self.curve(p);
        let lerp = |amt: f32| 1.0 - amt.clamp(0.0, 1.0) * (1.0 - c);
        (lerp(self.size), lerp(self.opacity))
    }
}

/// What the node puts on its output socket.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PaintOutput {
    /// The input with the strokes applied — retouching.
    #[default]
    Rgb,
    /// The strokes' COVERAGE as a matte, unioned with whatever alpha came in
    /// on the input (so a keyer's matte can be touched up by hand). The
    /// socket turns green; nothing about the pixels under the brush matters.
    Matte,
}

#[derive(Clone, Debug, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PaintParams {
    pub strokes: Vec<PaintStroke>,
    /// The lifetime new strokes are authored with — the node's paint MODE,
    /// in effect: leave it on `All` and the node is a matte painting, set it
    /// to `Single` and it is a per-frame dust bust.
    #[cfg_attr(feature = "serde", serde(default))]
    pub life_default: StrokeLife,
    #[cfg_attr(feature = "serde", serde(default))]
    pub output: PaintOutput,
}

impl PaintParams {
    /// Does any stroke read the input at an offset (clone, smear)? This and
    /// the two below decide the SHAPE of the generated shader — which extra
    /// reads of the input exist — so they are topology, and they must not
    /// depend on the frame or scrubbing would recompile: a timed-out stroke
    /// still counts, it just rasterizes to nothing that frame.
    pub fn has_source(&self) -> bool {
        self.strokes.iter().any(|s| s.mode().class() == BrushClass::Source)
    }

    /// Does any stroke blur? The expensive one.
    pub fn has_blur(&self) -> bool {
        self.strokes.iter().any(|s| s.mode() == BrushMode::Blur)
    }

    /// Does any stroke dodge or burn?
    pub fn has_tone(&self) -> bool {
        self.strokes
            .iter()
            .any(|s| matches!(s.mode(), BrushMode::Dodge | BrushMode::Burn))
    }

    /// Does the node need the effect layer at all?
    pub fn has_effect(&self) -> bool {
        self.has_blur() || self.has_tone()
    }

    /// Does any colour stroke blend with a mode that scales by a COLOUR
    /// (add, subtract, multiply, screen)? Then the colour layer carries a
    /// second, multiplier texture and the shader applies `below * M + B`.
    pub fn has_blend(&self) -> bool {
        self.strokes
            .iter()
            .any(|s| s.mode().class() == BrushClass::Color && s.blend.needs_mul())
    }

    /// Strokes live at `frame`, in paint order.
    pub fn live_at(&self, frame: u64) -> impl Iterator<Item = &PaintStroke> {
        self.strokes.iter().filter(move |s| s.life.covers(frame))
    }
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PaintStroke {
    /// Polyline in NORMALIZED [0,1] output space (resolution-independent).
    pub points: Vec<[f32; 2]>,
    /// Brush radius in output pixels.
    pub radius: f32,
    /// 0 = hard disc, 1 = fully soft falloff.
    pub softness: f32,
    /// Straight (unpremultiplied) RGBA. For a CLONE stroke only the alpha
    /// is used — the colour comes from the input, at the offset.
    pub color: [f32; 4],
    /// Which frames this stroke paints on.
    #[cfg_attr(feature = "serde", serde(default))]
    pub life: StrokeLife,
    /// CLONE: take the colour from the node's own input at `p + offset`
    /// rather than from `color`. NORMALIZED, like the points. Setups from
    /// before `mode` existed encode "clone" as `Some` here with no mode at
    /// all — `mode()` reads that as `Clone`, so they keep working.
    #[cfg_attr(feature = "serde", serde(default))]
    pub clone_off: Option<[f32; 2]>,
    /// What the stroke does. Read it through `mode()`, never directly: the
    /// legacy clone encoding above is folded in there.
    #[cfg_attr(feature = "serde", serde(default))]
    pub mode: BrushMode,
    /// Brush shape: minor/major axis ratio, 1 = round. The ellipse's major
    /// axis is `radius`.
    #[cfg_attr(feature = "serde", serde(default = "one"))]
    pub aspect: f32,
    /// Brush rotation, degrees, for an elliptical brush.
    #[cfg_attr(feature = "serde", serde(default))]
    pub angle: f32,
    /// The mode's second knob, next to the opacity that every mode has:
    /// blur radius in px, smear drag distance in px, dodge/burn exposure in
    /// stops. Unused by colour and clone. See `BrushMode::strength_label`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub strength: f32,
    /// How a colour stroke lands (see `PaintBlend`). Other modes ignore it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub blend: PaintBlend,
    /// Pen pressure per point, 0..1, parallel to `points`. EMPTY means
    /// full pressure everywhere — a mouse stroke, or any stroke from before
    /// pressure — so those setups load unchanged and stay small.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Vec::is_empty")
    )]
    pub pressure: Vec<f32>,
    /// What the pressure does to this stroke. Recorded per stroke, so
    /// changing the brush's response later never reshapes what is painted.
    #[cfg_attr(feature = "serde", serde(default = "PressureMap::off"))]
    pub press: PressureMap,
}

impl PressureMap {
    #[cfg(feature = "serde")]
    fn off() -> Self {
        Self::OFF
    }
}

impl PaintStroke {
    /// Pressure at point `i` (1 where the stroke has none).
    pub fn pressure_at(&self, i: usize) -> f32 {
        self.pressure.get(i).copied().unwrap_or(1.0)
    }

    /// A plain round colour stroke, every frame — what the brush made
    /// before lifetimes, cloning, shapes and modes existed.
    pub fn new(points: Vec<[f32; 2]>, radius: f32, softness: f32, color: [f32; 4]) -> Self {
        Self {
            points,
            radius,
            softness,
            color,
            life: StrokeLife::All,
            clone_off: None,
            mode: BrushMode::Color,
            aspect: 1.0,
            angle: 0.0,
            strength: 0.0,
            blend: PaintBlend::Normal,
            pressure: Vec::new(),
            press: PressureMap::OFF,
        }
    }

    /// The stroke's mode, with the pre-`mode` clone encoding folded in.
    pub fn mode(&self) -> BrushMode {
        if self.mode == BrushMode::Color && self.clone_off.is_some() {
            BrushMode::Clone
        } else {
            self.mode
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TransformParams {
    pub translate: [f32; 2],
    /// Degrees.
    pub rotate: f32,
    pub scale: [f32; 2],
    /// Pivot in normalized [0,1] image space.
    pub center: [f32; 2],
    /// UI: scale x/y edited as one value. Not used by codegen.
    #[cfg_attr(feature = "serde", serde(default))]
    pub scale_gang: bool,
    /// A corner pin applied BEFORE the move/rotate/scale, in the node's own
    /// normalised format: where the frame's corners go (TL, TR, BR, BL),
    /// exactly a CornerPin's `to`. The unit square is no pin at all, which
    /// is what every setup written before it loads as.
    #[cfg_attr(feature = "serde", serde(default = "unit_quad"))]
    pub pin: [[f32; 2]; 4],
    /// How a MODULATE matte blends the transform in, per pixel. Only read
    /// while the modulate port is wired; a uniform, so switching it never
    /// recompiles.
    #[cfg_attr(feature = "serde", serde(default))]
    pub mod_blend: TransformBlend,
    /// Shear about the pivot, before the rotate and scale: x moves by
    /// `shear[0]` × y, y by `shear[1]` × x (both from the pivot, in
    /// pixels). 0 is none, what every older setup loads as.
    #[cfg_attr(feature = "serde", serde(default))]
    pub shear: [f32; 2],
    /// Taper (a keystone): the end of the picture the taper axis points to
    /// narrows to `1 / (1 + taper)` of its width and the opposite end
    /// widens to `1 / (1 - taper)`, both ends staying put along the axis
    /// and every line staying straight. -0.95..0.95; 0 is none.
    #[cfg_attr(feature = "serde", serde(default))]
    pub taper: f32,
    /// The taper axis, degrees counter-clockwise on screen from +x: 90
    /// (the default) narrows the top.
    #[cfg_attr(feature = "serde", serde(default = "taper_up"))]
    pub taper_angle: f32,
    /// Which pipe it sits in: wired a MATTE it filters the coverage and
    /// gives a matte (`NodeKind::follows_pipe`). Kept true to its input by
    /// the app (`EditorGraph::connect_out`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub matte: bool,
    /// Apply the INVERSE of the whole map (pin, bend, move): the same knobs
    /// undo what they would do, so one matchmove can be removed and put
    /// back. Still one homography (`codegen::transform_applied_px`), a
    /// uniform write; the gizmo keeps drawing the authored map.
    #[cfg_attr(feature = "serde", serde(default))]
    pub invert: bool,
}

/// How a modulated Transform takes a partial strength `m` at a pixel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TransformBlend {
    /// The transform itself at strength m: move × m, rotate × m, scale to
    /// the power m, pin corners m of the way. A rotation sweeps along arcs.
    #[default]
    Params,
    /// The sample POSITION mixed between untouched and fully transformed —
    /// The UV map + UV Warp trick exactly. Rotated areas cut chords, so
    /// they shrink toward the pivot at partial strength.
    Coords,
}

/// The frame's own corners: a corner pin that does nothing.
pub const UNIT_QUAD: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

#[cfg(feature = "serde")]
fn unit_quad() -> [[f32; 2]; 4] {
    UNIT_QUAD
}

impl Default for TransformParams {
    fn default() -> Self {
        Self {
            translate: [0.0, 0.0],
            rotate: 0.0,
            scale: [1.0, 1.0],
            center: [0.5, 0.5],
            scale_gang: true,
            pin: UNIT_QUAD,
            mod_blend: TransformBlend::default(),
            shear: [0.0, 0.0],
            taper: 0.0,
            taper_angle: 90.0,
            matte: false,
            invert: false,
        }
    }
}

impl TransformParams {
    /// Does it shear or taper? Neither is part of the plain affine.
    pub fn bends(&self) -> bool {
        self.shear != [0.0, 0.0] || self.taper != 0.0
    }
}

#[cfg(feature = "serde")]
fn taper_up() -> f32 {
    90.0
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
// Container-level default so a Grade saved before `saturation` existed
// loads with THIS struct's 1.0 (no change) rather than the field type's 0.0
// (every old grade suddenly monochrome).
#[cfg_attr(feature = "serde", serde(default))]
pub struct GradeParams {
    pub lift: [f32; 3],
    pub gamma: [f32; 3],
    pub gain: [f32; 3],
    /// Printer-lights shift applied before gain: (c + offset) * gain + lift.
    #[cfg_attr(feature = "serde", serde(default))]
    pub offset: [f32; 3],
    /// Overall saturation, applied AFTER lift/gamma/gain: the pixel lerped
    /// about its own luminance (`y + (rgb - y) * s`), so 1 is unchanged, 0
    /// is monochrome and past 1 pushes. Luminance uses the working space's
    /// own weights, which makes it luminance-preserving by construction.
    /// The Saturation NODE is the elaborate version of this — log/perceptual
    /// domains, vibrance, per-zone weighting; this is the one slider a
    /// grade wants in front of it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub saturation: f32,
    /// Contrast about `pivot`, applied after lift/gamma/gain and BEFORE
    /// saturation: `pivot * (c / pivot)^contrast` per channel. A power about
    /// the pivot is a straight-line contrast in log, the form that suits a
    /// scene-linear picture: the pivot stays put, values never cross zero,
    /// and there is no ceiling for highlights to crush against. 1 is no
    /// change (and compiles to exactly the old maths); the container
    /// default keeps a setup saved before it existed loading at 1.
    pub contrast: f32,
    /// The value contrast turns about. Scene-linear mid grey by default.
    pub pivot: f32,
    /// UI: each row's rgb edited as one value. Not used by codegen.
    #[cfg_attr(feature = "serde", serde(default))]
    pub gang: bool,
    /// Luma ZONES — the classic colour-correct shadows / midtones / highlights,
    /// generalised to any number of them. The luma axis is cut into bands
    /// exactly as a LumaKey cuts it (same model, same editor), and each band
    /// carries a correction of its own, applied in series after the master
    /// and weighted by the band's matte. `None` is a plain Grade: what every
    /// setup written before zones existed loads as, and what it compiles to,
    /// byte for byte.
    #[cfg_attr(feature = "serde", serde(default))]
    pub zones: Option<GradeZones>,
    /// Which pipe it sits in: wired a MATTE it takes one and gives one
    /// (`NodeKind::follows_pipe`), the SoftCrop precedent. The app keeps it
    /// true to its input (`EditorGraph::connect_out`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub matte: bool,
}

impl GradeParams {
    /// How many zones the shader has to evaluate. Structural: it sizes the
    /// node's uniform span and joins the topology tag.
    pub fn zone_count(&self) -> usize {
        self.zones.as_ref().map_or(0, |z| z.count())
    }

    /// The five adjustments as ONE mutable view — the master's (`None`) or
    /// zone `i`'s. The panel binds its wheels and sliders to this, so one
    /// instrument serves every tab rather than a master version and a zone
    /// version that drift apart.
    pub fn adjust_mut(&mut self, zone: Option<usize>) -> Option<GradeAdjustMut<'_>> {
        match zone {
            None => Some(GradeAdjustMut {
                lift: &mut self.lift,
                gamma: &mut self.gamma,
                gain: &mut self.gain,
                offset: &mut self.offset,
                saturation: &mut self.saturation,
                contrast: &mut self.contrast,
                pivot: &mut self.pivot,
            }),
            Some(i) => {
                let z = self.zones.as_mut()?.adjust.get_mut(i)?;
                Some(GradeAdjustMut {
                    lift: &mut z.lift,
                    gamma: &mut z.gamma,
                    gain: &mut z.gain,
                    offset: &mut z.offset,
                    saturation: &mut z.saturation,
                    contrast: &mut z.contrast,
                    pivot: &mut z.pivot,
                })
            }
        }
    }
}

/// Where a Grade's contrast pivots by default: scene-linear mid grey.
pub const GRADE_PIVOT: f32 = 0.18;

/// A mutable view of one set of grade adjustments — see
/// [`GradeParams::adjust_mut`].
pub struct GradeAdjustMut<'a> {
    pub lift: &'a mut [f32; 3],
    pub gamma: &'a mut [f32; 3],
    pub gain: &'a mut [f32; 3],
    pub offset: &'a mut [f32; 3],
    pub saturation: &'a mut f32,
    pub contrast: &'a mut f32,
    pub pivot: &'a mut f32,
}

/// One zone's correction: the master's five adjustments over again, for one
/// band of the luma axis. Identity by default, so a fresh zone changes
/// nothing about the picture.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct GradeZone {
    pub lift: [f32; 3],
    pub gamma: [f32; 3],
    pub gain: [f32; 3],
    pub offset: [f32; 3],
    pub saturation: f32,
    pub contrast: f32,
    pub pivot: f32,
}

impl Default for GradeZone {
    fn default() -> Self {
        Self {
            lift: [0.0; 3],
            gamma: [1.0; 3],
            gain: [1.0; 3],
            offset: [0.0; 3],
            saturation: 1.0,
            contrast: 1.0,
            pivot: GRADE_PIVOT,
        }
    }
}

/// What the viewer shows while a Grade's ranges are being configured.
///
/// Authored, like the colour keyer's [`crate::keyer::KeyerView`], and
/// honoured under the same contract: only while the node IS the view target
/// (see `GradeZones::previewing`), so a preview can never reach a render or
/// a downstream consumer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ZoneView {
    /// The graded picture. What a Grade normally shows.
    #[default]
    Off,
    /// The zone's COVERAGE, as a matte: white where its correction lands in
    /// full, black where it does not land at all. The precise answer — a
    /// boundary is read off a matte, not off a picture.
    Matte,
    /// The picture, held inside the zone and dimmed to grey outside it. The
    /// same coverage read in CONTEXT: which faces, which sky, which
    /// highlight — the question a matte alone cannot answer.
    Highlight,
    /// EVERY zone at once, each in its own colour ([`band_srgb`], the
    /// colours of the plot), mixed by the zones' own weights. A plateau is
    /// its zone's flat colour; a crossfade is the blend of its two
    /// neighbours, as wide as the crossfade is; and coverage that does not
    /// add up (a bias point dipping a plateau) shows as that colour DARKENED,
    /// because the weights are summed, never normalised. The partition as a
    /// whole, which one zone at a time cannot show.
    Zones,
    /// The same map CUT HARD: each pixel is the flat colour of the zone that
    /// owns the most of it, so a crossfade becomes a line where its two
    /// weights cross (a tie goes to the lower zone), and black is a pixel
    /// no zone covers at all. Where the boundaries ARE, with the softness
    /// taken away — the blended map shows how wide they are.
    Hard,
}

impl ZoneView {
    pub const ALL: [ZoneView; 5] =
        [ZoneView::Off, ZoneView::Matte, ZoneView::Highlight, ZoneView::Zones, ZoneView::Hard];

    pub fn name(self) -> &'static str {
        match self {
            ZoneView::Off => "off",
            ZoneView::Matte => "matte",
            ZoneView::Highlight => "highlight",
            ZoneView::Zones => "zones",
            ZoneView::Hard => "hard",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            ZoneView::Off => "show the graded picture",
            ZoneView::Matte => "show this zone's coverage as a matte",
            ZoneView::Highlight => "hold this zone's pixels, dim the rest",
            ZoneView::Zones => {
                "every zone in its own colour (the plot's), blended through each crossfade"
            }
            ZoneView::Hard => {
                "every zone in its own colour, cut hard where one takes over from the next"
            }
        }
    }

    /// What the packer writes for the shader: 0 off, 1 matte, 2 highlight,
    /// 3 zones, 4 hard.
    pub fn code(self) -> f32 {
        match self {
            ZoneView::Off => 0.0,
            ZoneView::Matte => 1.0,
            ZoneView::Highlight => 2.0,
            ZoneView::Zones => 3.0,
            ZoneView::Hard => 4.0,
        }
    }
}

/// The colour band `i` of `n` wears, wherever bands are told apart: a
/// Grade's zone maps ([`ZoneView::Zones`], [`ZoneView::Hard`]) and the band
/// editor's plot (`bandui`, which lifts it a little to read over its dark
/// ramp), so the plot is the viewer's legend. Display sRGB, 8 bit.
///
/// CATEGORICAL, and chosen for how it looks THROUGH THE VIEWER: the order is
/// the greedy max-min of a pool of saturated colours as the default ACES
/// view shows them (CIE76 ΔE), under one more rule — two neighbours'
/// crossfade must not pass through an orange, because the blended map mixes
/// them and a green-to-red ramp is orange in the middle (a standing UI
/// rule). So three zones are green / blue / red, every pair at least 117 ΔE
/// apart as shown, where the viridis ramp this replaced managed ~70.
/// Past the end of the list it cycles; `i` past `n` reads as the last band
/// rather than panicking.
pub fn band_srgb(i: usize, n: usize) -> [u8; 3] {
    const SEQ: [[u8; 3]; 10] = [
        [43, 217, 74],   // green
        [47, 107, 255],  // blue
        [255, 48, 64],   // red
        [235, 235, 235], // white
        [255, 69, 208],  // magenta
        [0, 150, 140],   // teal
        [255, 226, 26],  // yellow
        [150, 255, 190], // mint
        [255, 160, 210], // pink
        [155, 77, 255],  // violet
    ];
    SEQ[i.min(n.max(1) - 1) % SEQ.len()]
}

/// [`band_srgb`] as scene-linear light in `working`'s primaries: what the
/// zones preview renders, so the view transform shows it as that colour.
pub fn band_linear(i: usize, n: usize, working: crate::color::Colorspace) -> [f32; 3] {
    use crate::color::{Curve, Primaries};
    let lin = band_srgb(i, n).map(|c| Curve::Srgb.to_linear(c as f32 / 255.0));
    crate::color::mat3_apply(
        crate::color::primaries_matrix(Primaries::Rec709, working.primaries),
        lin,
    )
}

/// Which zone the ranges editor is previewing, and how.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct ZonePreview {
    pub view: ZoneView,
    /// The zone shown. Follows the band selected in the ranges editor, so
    /// the preview is always of the band under the hand.
    pub zone: u32,
}

/// A Grade's zones: the partition of the luma axis, and one correction per
/// band of it.
///
/// The partition is a [`LumaKeyParams`] on purpose. A Grade with zones IS a
/// LumaKey with N bands feeding N masked Grades in series — that equivalence
/// is what the shader implements and what `engine/tests/grade_zones.rs` pins
/// pixel for pixel — so the model is the key's model and the editor is the
/// key's editor, rather than a second partition that would drift from the
/// first. `key.bands[i]` is zone `i`'s weight curve; `adjust[i]` is what it
/// does where it is at full strength.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct GradeZones {
    /// The partition, in the LumaKey's own model.
    pub key: LumaKeyParams,
    /// One correction per band, index-matched to `key.bands`.
    /// [`GradeZones::sync`] keeps the lengths equal; a missing entry reads
    /// as identity rather than panicking.
    pub adjust: Vec<GradeZone>,
    /// What the ranges editor is showing in the viewer. A knob like any
    /// other — it is authored, it undoes, it saves — but it only does
    /// anything while `previewing`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub preview: ZonePreview,
    /// Transient: set by the app when this node IS the view target. Never
    /// authored, never serialized — the same contract as the colour keyer's
    /// own `previewing`, and what makes a preview unable to reach a render
    /// or a downstream consumer.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub previewing: bool,
}

impl Default for GradeZones {
    /// The classic colour-correct defaults, on the key's scene-linear 0..2 axis:
    /// shadows fall off over 0..0.09, highlights rise over 0.5..1.0, and the
    /// midtones are what is left between. Every correction starts identity.
    fn default() -> Self {
        let mut r = crate::bands::BandRanges::even(3, LUMA_SPAN, false, 0.0);
        r.edges[0] = crate::bands::Edge::new(0.045, 0.09);
        r.edges[1] = crate::bands::Edge::new(0.75, 0.5);
        r.normalize();
        let mut key = LumaKeyParams { bands: Vec::new(), ranges: Some(r) };
        key.sync_from_ranges();
        let adjust = vec![GradeZone::default(); key.bands.len()];
        Self { key, adjust, preview: ZonePreview::default(), previewing: false }
    }
}

impl GradeZones {
    pub fn count(&self) -> usize {
        self.key.bands.len().max(1)
    }

    /// The preview the SHADER should apply: the authored one, but only
    /// while this node is the view target, and clamped to a zone that still
    /// exists (a dissolve can leave the selection past the end).
    pub fn live_preview(&self) -> Option<(ZoneView, u32)> {
        if !self.previewing || self.preview.view == ZoneView::Off {
            return None;
        }
        Some((self.preview.view, self.preview.zone.min(self.count() as u32 - 1)))
    }

    /// One correction per band, no more and no fewer. Called after every
    /// edit to the partition; a new band arrives identity.
    pub fn sync(&mut self) {
        self.adjust.resize(self.key.bands.len().max(1), GradeZone::default());
    }

    /// Zone `i`'s correction; identity for an index the file did not
    /// carry, so a hand-edited setup never reads out of bounds.
    pub fn zone(&self, i: usize) -> GradeZone {
        self.adjust.get(i).copied().unwrap_or_default()
    }

    /// The band editor split a band into two, the upper half now at `new`.
    /// The correction the split band had covers BOTH halves, so splitting
    /// changes nothing about the picture — it only makes a place to diverge.
    pub fn band_split(&mut self, new: usize) {
        let z = self.zone(new.saturating_sub(1));
        let at = new.min(self.adjust.len());
        self.adjust.insert(at, z);
        self.sync();
    }

    /// The band editor dissolved band `i` into its neighbour: the neighbour
    /// keeps its correction, the dissolved one's goes with it.
    pub fn band_dissolved(&mut self, i: usize) {
        if i < self.adjust.len() {
            self.adjust.remove(i);
        }
        self.sync();
    }

    /// What zone `i` is called: its band's own name, or the classic names
    /// when the count allows them — shadows / midtones / highlights for
    /// three, shadows / highlights for two, numbered mids beyond that.
    pub fn label(&self, i: usize) -> String {
        if let Some(b) = self.key.bands.get(i) {
            if !b.name.is_empty() {
                return b.name.clone();
            }
        }
        let n = self.count();
        match (n, i) {
            (1, _) => "zone".into(),
            (_, 0) => "shadows".into(),
            (n, i) if i + 1 == n => "highlights".into(),
            (3, 1) => "midtones".into(),
            (_, i) => format!("mids {i}"),
        }
    }
}

/// How the control points are joined into the closed outline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SplineMode {
    /// Straight segments — corners everywhere.
    Linear,
    /// Interpolating Catmull-Rom: passes through every point, no handles.
    #[default]
    CatmullRom,
    /// Cubic Bezier through each point's out/in handles (zero handles read
    /// as corners, so click-drawn shapes start polygonal and smooth as the
    /// handles are pulled).
    Bezier,
    /// Uniform cubic B-spline: approximating, so the curve is tugged toward
    /// the control polygon rather than passing through it — the smoothest
    /// option and the one that never overshoots.
    BSpline,
}

impl SplineMode {
    pub const ALL: [SplineMode; 4] = [
        SplineMode::Linear,
        SplineMode::CatmullRom,
        SplineMode::Bezier,
        SplineMode::BSpline,
    ];

    pub fn name(self) -> &'static str {
        match self {
            SplineMode::Linear => "linear",
            SplineMode::CatmullRom => "catmull-rom",
            SplineMode::Bezier => "bezier",
            SplineMode::BSpline => "b-spline",
        }
    }
}

#[allow(dead_code)] // serde default (feature-gated)
fn default_feather() -> f32 {
    30.0
}

/// One control point: position, its own feather width, and Bezier handles.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SplinePoint {
    /// NORMALIZED [0,1] output space.
    pub p: [f32; 2],
    /// Feather width in output pixels AT THIS POINT; the edge interpolates
    /// between neighbours along the outline.
    #[cfg_attr(feature = "serde", serde(default = "default_feather"))]
    pub feather: f32,
    /// Where this point's soft band sits, in output pixels along the
    /// outward normal: 0 centres it on the outline (half in, half out),
    /// `+feather/2` puts it all OUTSIDE (the hard edge on the outline),
    /// `-feather/2` all inside. Interpolates along the outline like the
    /// feather does.
    #[cfg_attr(feature = "serde", serde(default))]
    pub bias: f32,
    /// Handles as offsets from `p`, normalized space (Bezier mode).
    #[cfg_attr(feature = "serde", serde(default))]
    pub h_in: [f32; 2],
    #[cfg_attr(feature = "serde", serde(default))]
    pub h_out: [f32; 2],
    /// How tightly the curve turns at this point (Catmull-Rom mode): 0 is
    /// the ordinary smooth curve, 1 a corner, below 0 looser and rounder.
    /// It scales the tangent the curve leaves the point with (a Cardinal
    /// spline, per point). Authored, like the handles — not a keyed channel.
    #[cfg_attr(feature = "serde", serde(default))]
    pub tension: f32,
}

impl SplinePoint {
    pub fn new(p: [f32; 2], feather: f32) -> Self {
        Self { p, feather, bias: 0.0, h_in: [0.0, 0.0], h_out: [0.0, 0.0], tension: 0.0 }
    }
}

/// The algebra of coverage: how one matte meets another.
///
/// ONE vocabulary, used by every node that combines mattes — the Spline's
/// input port (where A is the incoming roto and B is this shape) and the
/// MatteCombine node (where A and B are two arbitrary mattes). A Spline
/// with nothing wired reads A as empty coverage, so `Union` — the default —
/// is exactly the bare shape, and chaining Splines builds a roto one node
/// per shape with no multi-shape editor to learn.
///
/// Soft and hard forms of the same idea are both here on purpose: coverage
/// is fractional, so `a*b` and `min(a,b)` genuinely differ on every soft
/// edge in the frame, and which one is right depends on whether the two
/// mattes are independent or the same edge seen twice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MatteOp {
    /// max(a, b) — either. The safe union: soft edges keep their shape
    /// because nothing accumulates.
    #[default]
    Union,
    /// min(a + b, 1) — either, ACCUMULATING: two half-covered edges make
    /// one solid pixel. What you want when the two mattes are independent
    /// coverage of the same area, and what you do not want when they are
    /// the same edge twice (that seams).
    Add,
    /// B alone; A is dropped.
    Replace,
    /// a * (1 - b) — punch B out of A. Holes, garbage cuts, stencils.
    Subtract,
    /// a * b — both, soft. Two independent coverages multiply.
    Intersect,
    /// min(a, b) — both, HARD. Keeps a soft edge intact instead of
    /// squaring it, which is the right choice when one matte is only there
    /// to bound the other (a garbage matte over a key).
    Min,
    /// abs(a - b) — either but not both. The symmetric difference; also the
    /// fastest way to SEE where two mattes disagree.
    Difference,
}

impl MatteOp {
    pub const ALL: [MatteOp; 7] = [
        MatteOp::Union,
        MatteOp::Add,
        MatteOp::Replace,
        MatteOp::Subtract,
        MatteOp::Intersect,
        MatteOp::Min,
        MatteOp::Difference,
    ];

    pub fn name(self) -> &'static str {
        match self {
            MatteOp::Union => "union (max)",
            MatteOp::Add => "add",
            MatteOp::Replace => "replace",
            MatteOp::Subtract => "subtract",
            MatteOp::Intersect => "intersect (mult)",
            MatteOp::Min => "intersect (min)",
            MatteOp::Difference => "difference",
        }
    }

    /// The WGSL for `a <op> b` on two scalar coverages. Chosen at COMPILE
    /// time — a uniform-driven switch would cost every pixel the branch for
    /// a control nobody scrubs — which is why the op joins the topology
    /// hash wherever it is used.
    pub fn expr(self, a: &str, b: &str) -> String {
        match self {
            MatteOp::Union => format!("max({a}, {b})"),
            MatteOp::Add => format!("min({a} + {b}, 1.0)"),
            MatteOp::Replace => b.to_string(),
            MatteOp::Subtract => format!("{a} * (1.0 - {b})"),
            MatteOp::Intersect => format!("{a} * {b}"),
            MatteOp::Min => format!("min({a}, {b})"),
            MatteOp::Difference => format!("abs({a} - {b})"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MatteCombineParams {
    pub op: MatteOp,
    /// Dissolves the result back to the A input, like a Blend's `mix`: 0 is
    /// a bypass, 1 is the full combine. A uniform, so it never recompiles.
    pub mix: f32,
}

impl Default for MatteCombineParams {
    fn default() -> Self {
        Self { op: MatteOp::default(), mix: 1.0 }
    }
}

#[allow(dead_code)] // serde defaults (feature-gated)
fn one() -> f32 {
    1.0
}

#[allow(dead_code)]
fn ones2() -> [f32; 2] {
    [1.0, 1.0]
}

#[allow(dead_code)]
fn unit_scale() -> [f32; 2] {
    [1.0, 1.0]
}
#[allow(dead_code)]
fn centre() -> [f32; 2] {
    [0.5, 0.5]
}

/// An affine shape transform: what a whole roto shape rides on, kept apart
/// from the control points so that moving a shape is five curves and not two
/// per point. `Copy`, because the viewer's gizmo takes a snapshot of it and
/// then edits the points it belongs to.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ShapeXf {
    /// NORMALIZED offset, like every other spatial value here.
    #[cfg_attr(feature = "serde", serde(default))]
    pub translate: [f32; 2],
    /// Degrees.
    #[cfg_attr(feature = "serde", serde(default))]
    pub rotate: f32,
    #[cfg_attr(feature = "serde", serde(default = "unit_scale"))]
    pub scale: [f32; 2],
    /// Pivot for rotate/scale, NORMALIZED — the middle of the frame unless
    /// moved.
    #[cfg_attr(feature = "serde", serde(default = "centre"))]
    pub center: [f32; 2],
}

impl Default for ShapeXf {
    fn default() -> Self {
        Self {
            translate: [0.0, 0.0],
            rotate: 0.0,
            scale: [1.0, 1.0],
            center: [0.5, 0.5],
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct SplineParams {
    /// Closed-loop control points. EMPTY by default: a fresh Spline draws
    /// nothing until points are clicked into the viewer.
    pub points: Vec<SplinePoint>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub mode: SplineMode,
    /// What the shape does to the matte arriving on the input port.
    #[cfg_attr(feature = "serde", serde(default))]
    pub combine: MatteOp,
    /// Matte gain: the shape's coverage scales by this before it combines,
    /// so a shape can be a partial matte without a Mult downstream.
    #[cfg_attr(feature = "serde", serde(default = "one"))]
    pub opacity: f32,
    /// Grow (+) or shrink (-) the outline by this many OUTPUT PIXELS before
    /// the feather ramp — the same signed-distance offset a Dilate applies
    /// to a finished matte, but free, and with the soft edge following it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub expand: f32,
    /// Shape of the ramp across the feather band: 0 = linear in distance,
    /// 1 = smoothstep. Both keep 0.5 coverage ON the outline, so the gizmo's
    /// rings stay honest whatever this is set to.
    #[cfg_attr(feature = "serde", serde(default))]
    pub falloff: f32,
    /// Flip the shape's own matte (before opacity and before combining):
    /// everything outside the outline, nothing inside.
    #[cfg_attr(feature = "serde", serde(default))]
    pub invert: bool,
    /// Shape transform, applied about its own centre in the node's pixel
    /// space. Authored points never move: this is what a tracked or
    /// hand-animated shape rides on, so a shape that merely moves needs five
    /// curves instead of two per control point.
    #[cfg_attr(feature = "serde", serde(default))]
    pub xf: ShapeXf,
    /// The node's own format, like a Grid's. `None` (the default) inherits
    /// — the canvas — which is what every shape did before this existed.
    /// Points stay normalized, so changing it re-scales the shape with the
    /// frame rather than cropping it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub format: Option<(u32, u32)>,
    /// Shape-wide feather added OUTSIDE every point's band, in output
    /// pixels: the soft edge grows outward and the inner edge stays.
    #[cfg_attr(feature = "serde", serde(default))]
    pub feather_out: f32,
    /// Shape-wide feather added INSIDE every point's band.
    #[cfg_attr(feature = "serde", serde(default))]
    pub feather_in: f32,
    /// Still being drawn: the outline has an end, and the shape renders NO
    /// matte until it is closed (a click on its first point). Every shape
    /// that existed before this is closed, hence the default.
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "std::ops::Not::not"))]
    pub open: bool,
    /// A ROTO CONTAINER's shapes after the first. The fields above are
    /// shape 0 (so a one-shape node is exactly what a Spline always was,
    /// and every setup loads unchanged); these are shapes 1.., and `ops`
    /// combine them. See `crate::roto`, which lowers a container into the
    /// plain Splines and MatteCombines it stands for.
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Vec::is_empty"))]
    pub more: Vec<SplineShape>,
    /// The inner graph's combine nodes, each over two of the container's
    /// mini nodes (a shape or another op). What is wired into Out (`out`)
    /// is unioned into the output.
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Vec::is_empty"))]
    pub ops: Vec<RotoOp>,
    /// Where each shape's mini node sits in the node ([-1, 1] square);
    /// missing entries take a default row.
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Vec::is_empty"))]
    pub layout: Vec<[f32; 2]>,
    /// Where the Out mini node sits (`None`: bottom centre).
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Option::is_none"))]
    pub out_at: Option<[f32; 2]>,
    /// What is wired into Out, in port order — unioned into the node's
    /// output. `None` (a fresh node, or one never rewired) means every
    /// loose end; the first edit to Out writes that list down, after which
    /// a shape can sit unconnected: drawn in the node, not rendered.
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Option::is_none"))]
    pub out: Option<Vec<usize>>,
    /// The container's AXES: groups that union what is wired into them and
    /// move it by a 3D transform — a Flame-style hierarchy, an axis wired
    /// into another axis inheriting its parent's move. Mini ids
    /// `ROTO_AXIS + k`. With a Camera on the node's port 1 the shapes live
    /// on planes in 3D and are seen through it; without one an axis is a
    /// 2D move (its Z ignored). See `crate::roto`.
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Vec::is_empty"))]
    pub axes: Vec<RotoAxis>,
    /// TRANSIENT, never saved: where the lowering put this shape — its
    /// axes and the camera — for the frame. `codegen::spline_outline`
    /// carries every outline sample through it. Only `crate::roto` sets it.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub proj: Option<ShapeProj>,
}

/// An axis inside a roto container: a group with a transform. What is wired
/// into it (shapes, combine ops, other axes) is unioned in port order and
/// moved, about `pivot`, by `xform` (scene units: the frame is one unit
/// wide, Y up, centred — the 3D cards' convention, so a default Camera sees
/// an unmoved shape exactly where it was drawn).
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize), serde(default))]
pub struct RotoAxis {
    pub name: String,
    pub xform: crate::scene3d::Xform,
    pub pivot: [f32; 3],
    /// Mini ids wired into it, in port order (`None`: an empty port).
    pub inputs: Vec<Option<usize>>,
    pub at: [f32; 2],
}

impl Default for RotoAxis {
    fn default() -> Self {
        Self {
            name: String::new(),
            xform: Default::default(),
            pivot: [0.0; 3],
            inputs: vec![None, None],
            at: [0.0, 0.0],
        }
    }
}

impl RotoAxis {
    /// Local -> parent: `T(translate) T(pivot) R S T(-pivot)`.
    pub fn matrix(&self) -> crate::scene3d::Mat4 {
        use crate::scene3d::{mat_mul, Xform};
        let t = |v: [f32; 3]| {
            Xform { translate: v, ..Default::default() }.matrix()
        };
        let rs = Xform { translate: [0.0; 3], ..self.xform }.matrix();
        let back = [-self.pivot[0], -self.pivot[1], -self.pivot[2]];
        let piv = [
            self.xform.translate[0] + self.pivot[0],
            self.xform.translate[1] + self.pivot[1],
            self.xform.translate[2] + self.pivot[2],
        ];
        mat_mul(&t(piv), &mat_mul(&rs, &t(back)))
    }
}

/// A camera as a roto shape sees it: world -> camera, and the lens.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotoCam {
    pub view: crate::scene3d::Mat4,
    pub focal: f32,
    pub aperture: f32,
}

impl RotoCam {
    pub fn of(c: &crate::scene3d::CameraParams) -> Self {
        let x = crate::scene3d::Xform { scale: [1.0; 3], ..c.xform };
        Self { view: x.inverse_scaled([1.0; 3]), focal: c.focal, aperture: c.aperture[0] }
    }
}

/// Where a roto shape lands for a frame: its plane moved by `world` (the
/// product of its axes) and, with a camera, projected through it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShapeProj {
    pub world: crate::scene3d::Mat4,
    pub cam: Option<RotoCam>,
}

impl ShapeProj {
    /// Nothing to do: no move, no camera.
    pub fn is_identity(&self) -> bool {
        self.cam.is_none() && self.world == crate::scene3d::identity()
    }

    /// The OUTPUT-PIXEL homography: a pixel of the shape's own (undrawn,
    /// unmoved) frame to where it lands. A plane through a pinhole is a
    /// homography, so this is exact for every sample of the outline.
    pub fn homography(&self, size: [f32; 2]) -> [[f32; 3]; 3] {
        let (w, h) = (size[0].max(1.0), size[1].max(1.0));
        // Pixel -> the plane's scene units (one unit wide, Y up, centred).
        let a = [[1.0 / w, 0.0, -0.5], [0.0, -1.0 / w, 0.5 * h / w], [0.0, 0.0, 1.0]];
        let m3 = |m: &crate::scene3d::Mat4, rows: [usize; 3]| {
            let mut q = [[0.0f32; 3]; 3];
            for (i, r) in rows.iter().enumerate() {
                q[i] = [m[*r][0], m[*r][1], m[*r][3]];
            }
            q
        };
        let (q, k) = match &self.cam {
            Some(c) => {
                let f = c.focal.max(1e-3) / c.aperture.max(1e-3) * w;
                (
                    m3(&crate::scene3d::mat_mul(&c.view, &self.world), [0, 1, 2]),
                    [[f, 0.0, -0.5 * w], [0.0, -f, -0.5 * h], [0.0, 0.0, -1.0]],
                )
            }
            None => (m3(&self.world, [0, 1, 3]), [[w, 0.0, 0.5 * w], [0.0, -w, 0.5 * h], [0.0, 0.0, 1.0]]),
        };
        mat3_mul(&k, &mat3_mul(&q, &a))
    }

    /// A normalized point through [`Self::homography`]. A point that would
    /// land behind the camera is held just in front of it.
    pub fn point(&self, p: [f32; 2], size: [f32; 2]) -> [f32; 2] {
        homography_point(&self.homography(size), p, size)
    }
}

fn mat3_mul(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut m = [[0.0f32; 3]; 3];
    for (r, row) in m.iter_mut().enumerate() {
        for (c, v) in row.iter_mut().enumerate() {
            *v = (0..3).map(|k| a[r][k] * b[k][c]).sum();
        }
    }
    m
}

/// A normalized point through an output-pixel homography.
pub fn homography_point(hm: &[[f32; 3]; 3], p: [f32; 2], size: [f32; 2]) -> [f32; 2] {
    let (x, y) = (p[0] * size[0], p[1] * size[1]);
    let u = hm[0][0] * x + hm[0][1] * y + hm[0][2];
    let v = hm[1][0] * x + hm[1][1] * y + hm[1][2];
    let z = (hm[2][0] * x + hm[2][1] * y + hm[2][2]).max(1e-4);
    [u / z / size[0].max(1e-6), v / z / size[1].max(1e-6)]
}

/// One shape of a roto container: everything a single Spline's shape is,
/// minus what belongs to the node (its format, and how the finished roto
/// meets the matte on the input).
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize), serde(default))]
pub struct SplineShape {
    pub points: Vec<SplinePoint>,
    pub mode: SplineMode,
    pub opacity: f32,
    pub expand: f32,
    pub falloff: f32,
    pub invert: bool,
    pub xf: ShapeXf,
    pub feather_out: f32,
    pub feather_in: f32,
    /// See [`SplineParams::open`].
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "std::ops::Not::not"))]
    pub open: bool,
}

impl Default for SplineShape {
    fn default() -> Self {
        let d = SplineParams::default();
        Self {
            points: Vec::new(),
            mode: d.mode,
            opacity: d.opacity,
            expand: d.expand,
            falloff: d.falloff,
            invert: d.invert,
            xf: d.xf,
            feather_out: d.feather_out,
            feather_in: d.feather_in,
            open: false,
        }
    }
}

/// A combine node inside a roto container: `op(a, b)` with `mix` back to
/// `a`, exactly a MatteCombine. `a` and `b` are mini-node ids (a shape
/// index, or `ROTO_OP + k` for op k); `None` is an unwired port, which
/// reads as empty coverage.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize), serde(default))]
pub struct RotoOp {
    pub op: MatteOp,
    pub mix: f32,
    pub a: Option<usize>,
    pub b: Option<usize>,
    pub at: [f32; 2],
}

impl Default for RotoOp {
    fn default() -> Self {
        Self { op: MatteOp::Union, mix: 1.0, a: None, b: None, at: [0.0, 0.3] }
    }
}

/// Mini-node ids at and above this are ops (`ROTO_OP + k`); below, shapes.
pub const ROTO_OP: usize = 1 << 12;
/// ...and at and above this, axes (`ROTO_AXIS + k`).
pub const ROTO_AXIS: usize = 2 << 12;
/// Axes a container may hold.
pub const ROTO_AXES_MAX: usize = 32;

/// A roto dive's own nodes (see [`NodeKind::RotoNode`]). Each is a bar
/// that takes any number of wires: the last port is always the free one.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RotoInner {
    /// `inputs` stands for the port count only: in the dive the wiring is
    /// the editor's own edges.
    Axis(RotoAxis),
    Out { inputs: usize },
}

impl RotoInner {
    pub fn input_count(&self) -> usize {
        match self {
            RotoInner::Axis(a) => a.inputs.len().clamp(1, 64),
            RotoInner::Out { inputs } => (*inputs).clamp(1, 64),
        }
    }
}

/// What a roto mini id names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotoRef {
    Shape(usize),
    Op(usize),
    Axis(usize),
}

impl RotoRef {
    pub fn of(id: usize) -> Self {
        if id >= ROTO_AXIS {
            RotoRef::Axis(id - ROTO_AXIS)
        } else if id >= ROTO_OP {
            RotoRef::Op(id - ROTO_OP)
        } else {
            RotoRef::Shape(id)
        }
    }
}
/// Shapes a container may hold.
pub const ROTO_SHAPES_MAX: usize = 32;

impl SplineParams {
    pub fn shape_count(&self) -> usize {
        1 + self.more.len()
    }

    /// More than a plain Spline: something for `crate::roto` to lower.
    pub fn is_container(&self) -> bool {
        !self.more.is_empty() || !self.ops.is_empty() || !self.axes.is_empty()
    }

    /// Shape `s` on its own (shape 0 is the node's own fields).
    pub fn shape(&self, s: usize) -> Option<SplineShape> {
        if s == 0 {
            return Some(SplineShape {
                points: self.points.clone(),
                mode: self.mode,
                opacity: self.opacity,
                expand: self.expand,
                falloff: self.falloff,
                invert: self.invert,
                xf: self.xf,
                feather_out: self.feather_out,
                feather_in: self.feather_in,
                open: self.open,
            });
        }
        self.more.get(s - 1).cloned()
    }

    pub fn set_shape(&mut self, s: usize, sh: SplineShape) {
        if s == 0 {
            self.points = sh.points;
            self.mode = sh.mode;
            self.opacity = sh.opacity;
            self.expand = sh.expand;
            self.falloff = sh.falloff;
            self.invert = sh.invert;
            self.xf = sh.xf;
            self.feather_out = sh.feather_out;
            self.feather_in = sh.feather_in;
            self.open = sh.open;
        } else if let Some(slot) = self.more.get_mut(s - 1) {
            *slot = sh;
        }
    }

    /// Shape `s` as a plain one-shape Spline: what the viewer gizmo edits,
    /// the panel shows, and the lowering renders. The node's own format
    /// and combine ride along, so a round trip through `apply_single`
    /// loses nothing.
    pub fn single(&self, s: usize) -> SplineParams {
        let sh = self.shape(s).unwrap_or_default();
        SplineParams {
            points: sh.points,
            mode: sh.mode,
            combine: self.combine,
            opacity: sh.opacity,
            expand: sh.expand,
            falloff: sh.falloff,
            invert: sh.invert,
            xf: sh.xf,
            format: self.format,
            feather_out: sh.feather_out,
            feather_in: sh.feather_in,
            open: sh.open,
            ..SplineParams::default()
        }
    }

    /// The inverse of [`Self::single`]: shape `s` takes `one`'s shape, and
    /// the node takes its format and combine.
    pub fn apply_single(&mut self, s: usize, one: &SplineParams) {
        self.combine = one.combine;
        self.format = one.format;
        self.set_shape(
            s,
            SplineShape {
                points: one.points.clone(),
                mode: one.mode,
                opacity: one.opacity,
                expand: one.expand,
                falloff: one.falloff,
                invert: one.invert,
                xf: one.xf,
                feather_out: one.feather_out,
                feather_in: one.feather_in,
                open: one.open,
            },
        );
    }

    /// Where shape `s`'s mini node sits: authored, or a default row across
    /// the top.
    pub fn shape_at(&self, s: usize) -> [f32; 2] {
        self.layout.get(s).copied().unwrap_or_else(|| {
            let n = self.shape_count().max(1) as f32;
            [-0.6 + 1.2 * (s as f32 + 0.5) / n, -0.5]
        })
    }

    pub fn set_shape_at(&mut self, s: usize, at: [f32; 2]) {
        let n = self.shape_count();
        while self.layout.len() < n {
            let k = self.layout.len();
            let d = self.shape_at(k);
            self.layout.push(d);
        }
        if let Some(slot) = self.layout.get_mut(s) {
            *slot = at;
        }
    }

    pub fn out_node_at(&self) -> [f32; 2] {
        self.out_at.unwrap_or([0.0, 0.62])
    }

    /// A new empty shape, its mini node placed; its index.
    pub fn add_shape(&mut self) -> Option<usize> {
        if self.shape_count() >= ROTO_SHAPES_MAX {
            return None;
        }
        // Pin every existing node where it is drawn now, so the default row
        // re-spacing for the new count does not move them.
        for k in 0..self.shape_count() {
            let at = self.shape_at(k);
            self.set_shape_at(k, at);
        }
        // Out's list is written down BEFORE the shape exists, so what was
        // loose stays wired; the new shape is then wired in too — it shows
        // the moment it is drawn.
        self.out_list();
        self.more.push(SplineShape::default());
        let s = self.shape_count() - 1;
        self.out_list().push(s);
        let at = [(-0.6 + 0.35 * s as f32).min(0.8), -0.5 + 0.12 * (s / 5) as f32];
        self.set_shape_at(s, at);
        Some(s)
    }

    /// Drop shape `s` (never the last one), sliding every later shape's id
    /// down in the ops. Shape 1 becomes the node's own fields when 0 goes.
    pub fn remove_shape(&mut self, s: usize) -> bool {
        if self.shape_count() <= 1 || s >= self.shape_count() {
            return false;
        }
        if s == 0 {
            let next = self.more.remove(0);
            self.set_shape(0, next);
        } else {
            self.more.remove(s - 1);
        }
        if s < self.layout.len() {
            self.layout.remove(s);
        }
        let slide = |r: &mut Option<usize>| {
            *r = match *r {
                Some(x) if x == s => None,
                Some(x) if x < ROTO_OP && x > s => Some(x - 1),
                other => other,
            };
        };
        for o in &mut self.ops {
            slide(&mut o.a);
            slide(&mut o.b);
        }
        for a in &mut self.axes {
            a.inputs.iter_mut().for_each(slide);
        }
        if let Some(out) = &mut self.out {
            out.retain(|x| *x != s);
            for x in out.iter_mut() {
                if *x < ROTO_OP && *x > s {
                    *x -= 1;
                }
            }
        }
        true
    }

    /// A combine node; its mini id.
    pub fn add_op(&mut self, op: RotoOp) -> usize {
        self.ops.push(op);
        ROTO_OP + self.ops.len() - 1
    }

    /// Drop op `k`, sliding later ops' ids down wherever they are read.
    pub fn remove_op(&mut self, k: usize) -> bool {
        if k >= self.ops.len() {
            return false;
        }
        self.ops.remove(k);
        self.slide_out(ROTO_OP + k, ROTO_AXIS);
        true
    }

    /// Mini id `gone` is no more: every reference to it unplugs, and the
    /// ids above it (below `ceil`, its own kind's range) slide down one.
    fn slide_out(&mut self, gone: usize, ceil: usize) {
        let slide = |r: &mut Option<usize>| {
            *r = match *r {
                Some(x) if x == gone => None,
                Some(x) if x > gone && x < ceil => Some(x - 1),
                other => other,
            };
        };
        for o in &mut self.ops {
            slide(&mut o.a);
            slide(&mut o.b);
        }
        for a in &mut self.axes {
            a.inputs.iter_mut().for_each(slide);
        }
        if let Some(out) = &mut self.out {
            out.retain(|x| *x != gone);
            for x in out.iter_mut() {
                if *x > gone && *x < ceil {
                    *x -= 1;
                }
            }
        }
    }

    /// A new axis; its mini id.
    pub fn add_axis(&mut self, axis: RotoAxis) -> Option<usize> {
        if self.axes.len() >= ROTO_AXES_MAX {
            return None;
        }
        self.axes.push(axis);
        Some(ROTO_AXIS + self.axes.len() - 1)
    }

    /// Drop axis `k`, sliding later axes' ids down wherever they are read.
    pub fn remove_axis(&mut self, k: usize) -> bool {
        if k >= self.axes.len() {
            return false;
        }
        self.axes.remove(k);
        self.slide_out(ROTO_AXIS + k, usize::MAX);
        true
    }

    /// The mini ids `r` reads, in port order (`None`: an empty port).
    pub fn mini_inputs(&self, r: usize) -> Vec<Option<usize>> {
        match RotoRef::of(r) {
            RotoRef::Shape(_) => Vec::new(),
            RotoRef::Op(k) => self.ops.get(k).map(|o| vec![o.a, o.b]).unwrap_or_default(),
            RotoRef::Axis(k) => self.axes.get(k).map(|a| a.inputs.clone()).unwrap_or_default(),
        }
    }

    /// Does mini id `r` name something that exists?
    pub fn mini_exists(&self, r: usize) -> bool {
        match RotoRef::of(r) {
            RotoRef::Shape(s) => s < self.shape_count(),
            RotoRef::Op(k) => k < self.ops.len(),
            RotoRef::Axis(k) => k < self.axes.len(),
        }
    }

    /// Where shape `s` lands, through the axes on its way to Out (the
    /// first way, if it is wired more than once) and the camera: what the
    /// gizmo draws it through. `None`: not moved at all.
    pub fn shape_proj(&self, s: usize, cam: Option<RotoCam>) -> Option<ShapeProj> {
        fn walk(sp: &SplineParams, r: usize, s: usize, m: crate::scene3d::Mat4, depth: usize) -> Option<crate::scene3d::Mat4> {
            if r == s {
                return Some(m);
            }
            if depth > 64 {
                return None;
            }
            let m2 = match RotoRef::of(r) {
                RotoRef::Axis(k) => crate::scene3d::mat_mul(&m, &sp.axes.get(k)?.matrix()),
                _ => m,
            };
            sp.mini_inputs(r).into_iter().flatten().find_map(|x| walk(sp, x, s, m2, depth + 1))
        }
        let m = self
            .out_inputs()
            .into_iter()
            .find_map(|r| walk(self, r, s, crate::scene3d::identity(), 0))
            .unwrap_or_else(crate::scene3d::identity);
        let p = ShapeProj { world: m, cam };
        (!p.is_identity()).then_some(p)
    }

    /// Does mini node `from` feed (directly or through ops) mini node `to`?
    /// Wiring `to` into `from` would then be a loop.
    pub fn feeds(&self, from: usize, to: usize) -> bool {
        if from == to {
            return true;
        }
        self.mini_inputs(to).into_iter().flatten().any(|x| self.feeds(from, x))
    }

    /// What Out unions, in port order: the written-down list, or every
    /// loose end while there is none. Ids that no longer exist drop out.
    pub fn out_inputs(&self) -> Vec<usize> {
        let valid = |x: &usize| self.mini_exists(*x);
        match &self.out {
            Some(v) => v.iter().copied().filter(valid).collect(),
            None => self.loose_ends(),
        }
    }

    /// Out's list, written down (from the loose ends) if it was not yet —
    /// for an edit.
    pub fn out_list(&mut self) -> &mut Vec<usize> {
        if self.out.is_none() {
            self.out = Some(self.loose_ends());
        }
        self.out.get_or_insert_with(Vec::new)
    }

    /// Every mini node nothing consumes, shapes first then ops, in id
    /// order: what Out unions until its list is written down.
    pub fn loose_ends(&self) -> Vec<usize> {
        let used: std::collections::HashSet<usize> = self
            .ops
            .iter()
            .flat_map(|o| [o.a, o.b])
            .chain(self.axes.iter().flat_map(|a| a.inputs.iter().copied()))
            .flatten()
            .collect();
        (0..self.shape_count())
            .chain((0..self.ops.len()).map(|k| ROTO_OP + k))
            .chain((0..self.axes.len()).map(|k| ROTO_AXIS + k))
            .filter(|id| !used.contains(id))
            .collect()
    }
}

impl Default for SplineParams {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            mode: SplineMode::default(),
            combine: MatteOp::default(),
            opacity: 1.0,
            expand: 0.0,
            falloff: 0.0,
            invert: false,
            xf: ShapeXf::default(),
            format: None,
            feather_out: 0.0,
            feather_in: 0.0,
            open: false,
            more: Vec::new(),
            ops: Vec::new(),
            layout: Vec::new(),
            out_at: None,
            out: None,
            axes: Vec::new(),
            proj: None,
        }
    }
}

impl ShapeXf {
    /// Nothing to apply? The gizmo and the packer both skip the matrix
    /// entirely when so, and an untransformed shape is bit-for-bit what it
    /// was before the transform existed.
    pub fn is_identity(&self) -> bool {
        self.translate == [0.0, 0.0] && self.rotate == 0.0 && self.scale == [1.0, 1.0]
    }

    /// Forward affine `[m00, m01, tx, m10, m11, ty]` in the node's own PIXEL
    /// space — rotation has to be isotropic, and a normalized-space rotation
    /// on a 2:1 format is a shear.
    pub fn affine(&self, size: [f32; 2]) -> [f32; 6] {
        let r = self.rotate.to_radians();
        let (s, c) = (r.sin(), r.cos());
        let (sx, sy) = (self.scale[0], self.scale[1]);
        let (m00, m01) = (c * sx, -s * sy);
        let (m10, m11) = (s * sx, c * sy);
        let (cx, cy) = (self.center[0] * size[0], self.center[1] * size[1]);
        let (tx, ty) = (self.translate[0] * size[0], self.translate[1] * size[1]);
        [m00, m01, cx + tx - m00 * cx - m01 * cy, m10, m11, cy + ty - m10 * cx - m11 * cy]
    }

    /// A normalized point put through the transform.
    pub fn point(&self, p: [f32; 2], size: [f32; 2]) -> [f32; 2] {
        if self.is_identity() {
            return p;
        }
        let m = self.affine(size);
        let (x, y) = (p[0] * size[0], p[1] * size[1]);
        [
            (m[0] * x + m[1] * y + m[2]) / size[0].max(1e-6),
            (m[3] * x + m[4] * y + m[5]) / size[1].max(1e-6),
        ]
    }

    /// The inverse of `point`: screen-side edits (a dragged control point)
    /// land in AUTHORED space, so dragging a point on a transformed shape
    /// moves that point and not the whole shape.
    pub fn point_inv(&self, p: [f32; 2], size: [f32; 2]) -> [f32; 2] {
        if self.is_identity() {
            return p;
        }
        let m = self.affine(size);
        let det = m[0] * m[4] - m[1] * m[3];
        if det.abs() < 1e-9 {
            return p;
        }
        let (x, y) = (p[0] * size[0] - m[2], p[1] * size[1] - m[5]);
        [
            (m[4] * x - m[1] * y) / det / size[0].max(1e-6),
            (-m[3] * x + m[0] * y) / det / size[1].max(1e-6),
        ]
    }

    /// A DIRECTION (a drag delta) through the inverse — the translation part
    /// drops out, so a delta stays a delta.
    pub fn vec_inv(&self, d: [f32; 2], size: [f32; 2]) -> [f32; 2] {
        if self.is_identity() {
            return d;
        }
        let m = self.affine(size);
        let det = m[0] * m[4] - m[1] * m[3];
        if det.abs() < 1e-9 {
            return d;
        }
        let (x, y) = (d[0] * size[0], d[1] * size[1]);
        [
            (m[4] * x - m[1] * y) / det / size[0].max(1e-6),
            (-m[3] * x + m[0] * y) / det / size[1].max(1e-6),
        ]
    }
}

/// Setups written before per-point feather/handles stored `points` as bare
/// `[x, y]` pairs plus one global `feather`; those still load (as
/// Catmull-Rom, the shape they were authored against).
#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
#[serde(untagged)]
enum SplineRepr {
    New {
        points: Vec<SplinePoint>,
        #[serde(default)]
        mode: SplineMode,
        // Everything below post-dates the shape itself; a setup written
        // before them loads with the values that mean "as it was".
        #[serde(default)]
        combine: MatteOp,
        #[serde(default = "one")]
        opacity: f32,
        #[serde(default)]
        expand: f32,
        #[serde(default)]
        falloff: f32,
        #[serde(default)]
        invert: bool,
        #[serde(default)]
        xf: ShapeXf,
        #[serde(default)]
        format: Option<(u32, u32)>,
        #[serde(default)]
        feather_out: f32,
        #[serde(default)]
        feather_in: f32,
        #[serde(default)]
        open: bool,
        #[serde(default)]
        more: Vec<SplineShape>,
        #[serde(default)]
        ops: Vec<RotoOp>,
        #[serde(default)]
        layout: Vec<[f32; 2]>,
        #[serde(default)]
        out_at: Option<[f32; 2]>,
        #[serde(default)]
        out: Option<Vec<usize>>,
        #[serde(default)]
        axes: Vec<RotoAxis>,
    },
    Old {
        points: Vec<[f32; 2]>,
        feather: f32,
    },
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for SplineParams {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Ok(match SplineRepr::deserialize(d)? {
            SplineRepr::New {
                points,
                mode,
                combine,
                opacity,
                expand,
                falloff,
                invert,
                xf,
                format,
                feather_out,
                feather_in,
                open,
                more,
                ops,
                layout,
                out_at,
                out,
                axes,
            } => Self {
                points,
                mode,
                combine,
                opacity,
                expand,
                falloff,
                invert,
                xf,
                format,
                feather_out,
                feather_in,
                open,
                more,
                ops,
                layout,
                out_at,
                out,
                axes,
                proj: None,
            },
            SplineRepr::Old { points, feather } => Self {
                points: points.into_iter().map(|p| SplinePoint::new(p, feather)).collect(),
                mode: SplineMode::CatmullRom,
                ..Self::default()
            },
        })
    }
}

/// The comp pipe an SDF graph's socket rides: distances and numbers on the
/// field pipe, colour on the image pipe.
fn sdf_pipe(p: crate::sdf::Pipe) -> PortKind {
    match p {
        crate::sdf::Pipe::Dist => PortKind::Field,
        crate::sdf::Pipe::Color => PortKind::Image,
    }
}

/// What a connection carries. The pipe a wire is in decides what may be
/// plugged into what ([`PortKind::can_feed`]) and, in time, what the
/// compositor has to spend to carry it — which is why they are few and why
/// each one means something the others do not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PortKind {
    Image,
    /// COVERAGE: one channel, defined on 0..1. A matte outside that range
    /// is a bug, which is what lets it be kept cheaply.
    Mask,
    /// A MEASUREMENT: one channel, full depth, whatever the number is —
    /// depth, disparity, thickness, an id. The same shape as a matte and
    /// the opposite contract, because clamping or quantising one of these
    /// destroys the thing it is for (`nodes::todata` has the table).
    ///
    /// One channel, deliberately. Multi-channel data — a normal, a
    /// position — stays on the image pipe until it earns a pipe of its own;
    /// a pixel pipe is a cost every node and every gesture pays. Motion
    /// vectors earned theirs (`Vector`).
    Data,
    /// MOTION VECTORS: `rg` forward and `ba` backward motion in canvas
    /// pixels, y down — a Flow node's output, what VectorBlur, VectorWarp
    /// and the like read. Never colour: no input transform, no clamp. A
    /// vector wire may feed an image input (an Expression or a Shader reads
    /// the numbers), never the other way: a picture plugged into a vectors
    /// port would be read as motion.
    Vector,
    /// A procedural field: not pixels but a function of position, evaluated
    /// wherever its consumer asks. Only ever seen inside a noise sub-graph.
    Field,
    /// Vector GEOMETRY: points and the primitives that index them. Only ever
    /// seen inside a 2dini sub-graph.
    Geo,
    /// EVERY channel of a multi-layer source at once — a lazy REFERENCE to
    /// the file, not decoded pixels. Nothing evaluates it; a ChannelSelect
    /// (or a node that declares it) turns one layer of it into an ordinary
    /// image wire, and only THAT layer ever decodes or caches. The fat
    /// cable in the graph.
    Channels,
    /// TEXT: a prompt, read on the CPU by whoever consumes it (a GenAI
    /// node's `prompt in` step). Never a picture, never sampled.
    Text,
    /// A 3D SCENE: cameras and cards (`crate::scene3d`), read on the CPU by
    /// a Render3D and lowered before codegen. Never a picture, never sampled.
    Scene,
}

impl PortKind {
    /// May a wire leaving a `self` output be plugged into a `dst` input?
    ///
    /// THE wiring rule, in one place, so the drag, the magnet, the bridge
    /// and the loader cannot disagree about what is legal. Same kind
    /// always. The one crossing allowed without a node in between is
    /// **matte into data**: coverage is a measurement that happens to live
    /// on 0..1, so widening it loses nothing and asks nothing of the user.
    ///
    /// The reverse is refused on purpose. Data into a matte port needs a
    /// range — which depth is black and which is white — and there is no
    /// answer this code could pick that would not be wrong on somebody's
    /// shot. The editor offers the converter instead (`bridge_for`).
    pub fn can_feed(self, dst: PortKind) -> bool {
        self == dst
            || (self == PortKind::Mask && dst == PortKind::Data)
            || (self == PortKind::Vector && dst == PortKind::Image)
    }

    /// Is this the PICTURE pipe? Image, and Data, which is the same raster
    /// with one channel and no colour meaning. Mask is out: a matte is
    /// usually a modifier rather than the picture. The resolution and range
    /// walks ask `NodeKind::inherits_from`, which also takes a matte that is
    /// the node's subject (a Dilate's).
    pub fn is_picture(self) -> bool {
        matches!(self, PortKind::Image | PortKind::Data | PortKind::Vector)
    }
}


/// How the defocus is computed. The mode decides the PASS CHAIN, so it is
/// folded into the topology hash — switching mode recompiles, every other
/// control is a uniform.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DefocusMode {
    /// Game-style gather: an exact flat disc, per-pixel radius, evaluated
    /// through row prefix sums — O(r) per pixel, not O(r^2). Measured at 4K:
    /// 7.0 ms at r=8, 20 ms at r=32, 73 ms at r=128, so quadrupling the
    /// radius costs ~3.5x rather than 16x. Flat kernel only, and no
    /// occlusion reasoning at all (background bleeds over sharp foreground).
    /// The fast path at ordinary radii, and the one to scrub focus on.
    #[default]
    Fast,
    /// Fixed-count spiral gather with an arbitrary bokeh kernel and the
    /// scatter-as-gather occlusion test. Cost is O(samples) per pixel and
    /// FLAT in radius — 4K measures 32 ms at 128 samples whether the CoC is
    /// 8 px or 128 px (16 ms at 64, 124 ms at 512). So it overtakes Fast
    /// somewhere around r=20, but the constant is large and it is the
    /// CONSPICUOUS slow path (see `is_slow`).
    Accurate,
}

impl DefocusMode {
    /// Whether this mode is the marked slow path. Manifesto rule: a slow
    /// path never hides.
    pub fn is_slow(self) -> bool {
        matches!(self, DefocusMode::Accurate)
    }
}

/// Which channel of the depth input carries Z.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DepthChannel {
    #[default]
    R,
    G,
    B,
    A,
    Luma,
}

impl DepthChannel {
    pub const ALL: [DepthChannel; 5] = [
        DepthChannel::R,
        DepthChannel::G,
        DepthChannel::B,
        DepthChannel::A,
        DepthChannel::Luma,
    ];
    pub fn label(self) -> &'static str {
        match self {
            DepthChannel::R => "R",
            DepthChannel::G => "G",
            DepthChannel::B => "B",
            DepthChannel::A => "A",
            DepthChannel::Luma => "Luma",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DefocusParams {
    /// Max CoC radius in output pixels. With no depth wired this IS the
    /// radius (uniform disc); with depth it clamps the near/far extremes and
    /// so bounds the cost.
    pub radius: f32,
    #[cfg_attr(feature = "serde", serde(default))]
    pub mode: DefocusMode,
    // --- thin-lens CoC, used only when the depth input is wired ---
    /// The value IN THE DEPTH PASS that lands sharp — exactly what a pixel
    /// of the pass reads, so a sample off the pass is the knob's value with
    /// no conversion (with `inverse_depth` that is the 1/Z number, not Z).
    /// World-space passes run to thousands; nothing here bounds it.
    #[cfg_attr(feature = "serde", serde(default = "default_focus"))]
    pub focus: f32,
    /// Metres per depth unit (1 = metres, 0.01 = cm). The lens is in mm, so
    /// the CoC needs to know what a unit of Z is; depth AND focus are both
    /// scaled by it, so it never moves the focal plane, only the blur.
    #[cfg_attr(feature = "serde", serde(default = "default_unit"))]
    pub unit: f32,
    /// Focal-plane view: while this node is the view target, show the sharp
    /// input tinted by CoC — green in focus, red in front, blue behind —
    /// instead of the defocus. Authored (it undoes and saves); it acts only
    /// with `previewing`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub focal_view: bool,
    /// Transient: set on the engine graph for the view target only (the
    /// keyer's contract), so the view never reaches a render or a consumer.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub previewing: bool,
    /// f-number. Smaller = shallower depth of field.
    #[cfg_attr(feature = "serde", serde(default = "default_fstop"))]
    pub f_stop: f32,
    /// Focal length in mm.
    #[cfg_attr(feature = "serde", serde(default = "default_focal"))]
    pub focal_len: f32,
    /// Filmback / sensor width in mm — what converts CoC millimetres to
    /// pixels against the canvas width.
    #[cfg_attr(feature = "serde", serde(default = "default_filmback"))]
    pub filmback: f32,
    /// Depth channel holds 1/Z (disparity) rather than Z.
    #[cfg_attr(feature = "serde", serde(default))]
    pub inverse_depth: bool,
    #[cfg_attr(feature = "serde", serde(default))]
    pub depth_channel: DepthChannel,
    // --- bokeh, Accurate mode only ---
    /// Aperture blades: 0 is a circle, 3..=11 a polygon.
    #[cfg_attr(feature = "serde", serde(default))]
    pub blades: u32,
    /// Blade rotation in degrees.
    #[cfg_attr(feature = "serde", serde(default))]
    pub rotation: f32,
    /// Optical vignetting: 0 leaves discs round everywhere, 1 squeezes them
    /// into cat's-eyes toward the frame edge.
    #[cfg_attr(feature = "serde", serde(default))]
    pub catseye: f32,
    /// Spiral taps per pixel AT THE MAX RADIUS; a pixel whose search disc
    /// is smaller takes proportionally fewer (never under 8), so the
    /// budget lands where the blur is.
    #[cfg_attr(feature = "serde", serde(default = "default_samples"))]
    pub samples: u32,
    /// Depth-edge treatment, in pixels: the NEAREST depth within this
    /// many pixels stands for the pixel. The depth pass is one sample a
    /// pixel while the colour is antialiased, so an edge pixel's mixed
    /// colour wears a hard depth and halos; taking the nearer depth makes
    /// it travel with the foreground (the erode-the-Z habit). 0 = off.
    #[cfg_attr(feature = "serde", serde(default))]
    pub edge: f32,
    /// Accurate only: soft clamp on a tap's luminance before it spreads.
    /// An HDR highlight through too few taps is a scatter of white dots;
    /// clamping tames it at the cost of a dimmer bokeh. 0 = off.
    #[cfg_attr(feature = "serde", serde(default))]
    pub bright_clamp: f32,
}

fn default_focus() -> f32 {
    10.0
}
fn default_fstop() -> f32 {
    2.8
}
fn default_unit() -> f32 {
    1.0
}
fn default_focal() -> f32 {
    50.0
}
fn default_filmback() -> f32 {
    36.0
}
fn default_samples() -> u32 {
    128
}

impl Default for DefocusParams {
    fn default() -> Self {
        Self {
            radius: 8.0,
            mode: DefocusMode::default(),
            focus: default_focus(),
            unit: default_unit(),
            focal_view: false,
            previewing: false,
            f_stop: default_fstop(),
            focal_len: default_focal(),
            filmback: default_filmback(),
            inverse_depth: false,
            depth_channel: DepthChannel::default(),
            blades: 0,
            rotation: 0.0,
            catseye: 0.0,
            samples: default_samples(),
            edge: 0.0,
            bright_clamp: 0.0,
        }
    }
}

impl DefocusParams {
    /// CoC radius in pixels for a raw depth sample (channel selection
    /// already applied), by the thin-lens model. Mirrors `depth_z` +
    /// `coc_px` in `codegen::DEFOCUS_COMMON_WGSL`; `coc_matches_wgsl` in
    /// engine/tests/defocus.rs asserts the two agree.
    pub fn coc_px(&self, depth: f32, canvas_w: f32) -> f32 {
        let z = self.depth_z(depth);
        let f = self.focal_len.max(1e-3);
        let mm = 1000.0 * self.unit.max(1e-9);
        let focus_mm = (self.depth_z(self.focus) * mm).max(f * 1.001);
        let z_mm = z * mm;
        if z_mm <= 0.0 || z_mm.is_nan() {
            return 0.0;
        }
        let aperture = f / self.f_stop.max(1e-3);
        let coc_mm =
            aperture * (f / (focus_mm - f).max(1e-3)) * ((z_mm - focus_mm).abs() / z_mm);
        (coc_mm * (canvas_w / self.filmback.max(1e-3))).clamp(0.0, self.radius.max(0.0))
    }

    /// Linear Z (depth units) from a raw pass value. Mirrors `depth_z` in
    /// the WGSL, and is applied to `focus` too — focus is a pass value.
    pub fn depth_z(&self, raw: f32) -> f32 {
        if self.inverse_depth {
            1.0 / raw.abs().clamp(1e-6, 1e6)
        } else {
            raw
        }
    }

    /// Whether the focal-plane view is showing: authored AND the view target.
    pub fn live_focal_view(&self) -> bool {
        self.focal_view && self.previewing
    }
}


#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CornerPinParams {
    /// Destination corners in NORMALIZED output space, order TL, TR, BR, BL.
    pub to: [[f32; 2]; 4],
    /// Source corners, same space and order: the warp is the homography
    /// that carries `from` onto `to`, so it is the DELTA between the two
    /// quads. The frame (the unit square) by default, which is the classic
    /// pin; set both to the same quad anywhere and nothing moves.
    #[cfg_attr(feature = "serde", serde(default = "unit_quad"))]
    pub from: [[f32; 2]; 4],
    /// How a MODULATE matte blends the pin in: params moves the corners m of
    /// the way, coords mixes the sample positions (a Transform's choice).
    #[cfg_attr(feature = "serde", serde(default))]
    pub mod_blend: TransformBlend,
    /// Which pipe it sits in: wired a MATTE it filters the coverage and
    /// gives a matte (`NodeKind::follows_pipe`). Kept true to its input by
    /// the app (`EditorGraph::connect_out`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub matte: bool,
    /// Apply the INVERSE warp: `to` back onto `from`. Folded into
    /// `effective_to`, so the shader, the modulate span and every gizmo
    /// that deforms through the pin follow without knowing about it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub invert: bool,
}

impl Default for CornerPinParams {
    fn default() -> Self {
        Self::pinned([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    }
}

impl CornerPinParams {
    /// A pin to `to`, blend at its default — what the maths helpers build.
    pub fn pinned(to: [[f32; 2]; 4]) -> Self {
        Self { to, from: UNIT_QUAD, mod_blend: TransformBlend::default(), matte: false, invert: false }
    }

    /// The `to` of the plain frame-to-quad pin that warps exactly as this
    /// one does: where the FRAME's corners go under the `from` -> `to`
    /// homography. A homography is fixed by four corners, so this is not an
    /// approximation — it is how `from` reaches the shader for free (the
    /// same uniforms, the same `persp`, the same modulate span).
    ///
    /// Inverted, the warp carries `to` onto `from`: the same thing with the
    /// two quads swapped.
    pub fn effective_to(&self) -> [[f32; 2]; 4] {
        let (from, to) = if self.invert { (self.to, self.from) } else { (self.from, self.to) };
        if from == UNIT_QUAD {
            return to;
        }
        if from == to {
            return UNIT_QUAD;
        }
        let h = crate::codegen::square_to_quad(to);
        let f_inv = crate::codegen::mat3_inv(&crate::codegen::square_to_quad(from));
        UNIT_QUAD.map(|c| h_apply(&h, h_apply(&f_inv, c)))
    }

    /// The `to` that makes this pin's warp `eff` (a frame-to-quad pin's
    /// corners) with `from` held: `effective_to` run backwards, for an edit
    /// made in effective terms (the soft-crop pins).
    pub fn to_for_effective(&self, eff: [[f32; 2]; 4]) -> [[f32; 2]; 4] {
        if self.invert {
            // to -> from must be unit -> eff, so to is `from` carried back
            // through eff's inverse.
            let h_inv = crate::codegen::mat3_inv(&crate::codegen::square_to_quad(eff));
            return self.from.map(|c| h_apply(&h_inv, c));
        }
        if self.from == UNIT_QUAD {
            return eff;
        }
        let h = crate::codegen::square_to_quad(eff);
        self.from.map(|c| h_apply(&h, c))
    }
}

/// A homography applied to a point, the divide guarded (a corner mapped to
/// the horizon lands far away rather than at NaN).
fn h_apply(h: &[[f32; 3]; 3], p: [f32; 2]) -> [f32; 2] {
    let w = h[2][0] * p[0] + h[2][1] * p[1] + h[2][2];
    let w = if w.abs() < 1e-9 { 1e-9f32.copysign(w) } else { w };
    [
        (h[0][0] * p[0] + h[0][1] * p[1] + h[0][2]) / w,
        (h[1][0] * p[0] + h[1][1] * p[1] + h[1][2]) / w,
    ]
}

impl Default for GradeParams {
    fn default() -> Self {
        Self {
            lift: [0.0; 3],
            gamma: [1.0; 3],
            gain: [1.0; 3],
            offset: [0.0; 3],
            saturation: 1.0,
            contrast: 1.0,
            pivot: GRADE_PIVOT,
            gang: true,
            zones: None,
            matte: false,
        }
    }
}

/// serde default for the cache flags (new fields on old setups read as on).
#[allow(dead_code)]
fn yes() -> bool {
    true
}

#[cfg(feature = "serde")]
fn legacy_input() -> crate::color::InputSpace {
    crate::color::InputSpace::Working
}

/// Fold a source node's input colour space into its topology tag: two Reads
/// that differ only in what their plate IS compile to two different shaders.
///
/// `Working` deliberately tags as 0 so a setup predating input colour
/// management hashes exactly as it did before, and its baked caches survive.
fn idt_tag(input: &crate::color::InputSpace) -> u64 {
    use std::hash::{Hash, Hasher};
    // `Default` reaching the engine is `Working` (see `InputSpace::Default`).
    if matches!(input, crate::color::InputSpace::Working | crate::color::InputSpace::Default) {
        return 0;
    }
    let mut h = DefaultHasher::new();
    input.hash(&mut h);
    // Never 0: that value means "no transform".
    h.finish() | 1
}

impl NodeKind {
    /// Read with default trim (full) and both cache tiers enabled.
    pub fn read(source: usize, path: impl Into<String>) -> Self {
        NodeKind::Read {
            source,
            path: path.into(),
            range: None,
            ram_cache: false,
            nvme_cache: true,
            layer: None,
            input: crate::color::InputSpace::default(),
            multi: false,
            start: None,
            first: None,
            last: None,
            edges: ReadEdges::default(),
            cache_on_play: true,
            single: crate::decode::SingleChannel::Off,
            eight_bit: false,
            scale: Default::default(),
        }
    }

    /// A default Blend (over).
    pub fn blend() -> Self {
        NodeKind::Blend { op: BlendOp::Over, mix: 1.0, gain: [1.0, 1.0] }
    }

    /// A fresh layer stack (two layers — see [`LayerParams::new`]).
    pub fn layer() -> Self {
        NodeKind::Layer(LayerParams::new())
    }

    /// How this node remaps time for everything upstream of it (identity for
    /// all but the time nodes).
    pub fn time_map(&self) -> TimeMap {
        match self {
            NodeKind::Slip(p) => TimeMap::affine(1.0, p.offset as f32),
            NodeKind::TimeRemap(p) => p.map(),
            NodeKind::Speed(p) => p.map(),
            NodeKind::RangeRemap(p) => p.map(),
            NodeKind::Freeze(p) => p.map(),
            _ => TimeMap::default(),
        }
    }

    /// A blending time node's two branches, when they are not the Floor and
    /// Ceil sides of one map: a Frame Fix reads the good frame BEFORE a bad
    /// run on one and the one AFTER it on the other.
    pub fn blend_branch_maps(&self) -> Option<[TimeMap; 2]> {
        match self {
            NodeKind::FrameFix(p) => Some(p.branch_maps()),
            _ => None,
        }
    }

    /// The dissolve a blending time node packs for `tick`, its output
    /// sitting under `outer`.
    pub fn blend_weight_at(&self, outer: &TimeMap, tick: u64) -> f32 {
        match self {
            NodeKind::FrameFix(p) => p.weight_at(outer.apply_f(tick as f32).round() as i64),
            _ => self.time_map().then(outer.clone()).blend_weight(tick),
        }
    }

    /// Nearest or blend, for the nodes that can land between frames. Slip
    /// shifts by whole frames, so it never has a fraction to dissolve.
    pub fn time_interp(&self) -> TimeInterp {
        match self {
            NodeKind::Speed(p) => p.interp,
            NodeKind::RangeRemap(p) => p.interp,
            NodeKind::TimeRemap(p) => p.interp,
            // Two frames dissolved, until flow-mode frames are loaded.
            NodeKind::FrameFix(p) if !p.loaded => TimeInterp::Blend,
            _ => TimeInterp::Nearest,
        }
    }

    /// Does this node put its input subtree on its own timebase? True even
    /// when the map currently reads as identity — an offset of 0 is still a
    /// separate branch of time, so passing through zero never restructures
    /// the graph (or recompiles).
    pub fn is_time_node(&self) -> bool {
        matches!(
            self,
            NodeKind::Slip(_)
                | NodeKind::TimeRemap(_)
                | NodeKind::Speed(_)
                | NodeKind::RangeRemap(_)
                | NodeKind::Freeze(_)
                | NodeKind::FrameFix(FrameFixParams { loaded: false, .. })
        )
    }

    /// Does each input sit on its OWN timebase? A Slip puts its whole
    /// subtree on one; a Layer gives every layer a different offset, so the
    /// time walk has to branch per input instead of once per node.
    pub fn per_input_time(&self) -> bool {
        matches!(self, NodeKind::Layer(_))
    }

    /// The map this node applies to ONE input. Everything but a Layer treats
    /// its inputs alike, so this is `time_map()` for them.
    pub fn input_time_map(&self, input: usize) -> TimeMap {
        match self {
            // source = (comp − start) × speed: a layer that STARTS at frame
            // 5 shows its own frame 0 there, and one stretched to twice its
            // length gets there half as fast. The only place the sign is
            // flipped, and the only place a stretch becomes a rate.
            NodeKind::Layer(p) => match p.layers.get(input) {
                Some(l) => {
                    let sp = l.rate();
                    let mut m = TimeMap::affine(sp, -l.start as f32 * sp);
                    if let Some(e) = l.extend {
                        m.ops.push(TimeOp::Fold { lo: e.lo, hi: e.hi, edge: e.edge });
                    }
                    m
                }
                None => TimeMap::default(),
            },
            _ => self.time_map(),
        }
    }

    /// Whether input `i` is actually sampled to render this node.
    ///
    /// `inputs_unsampled` answers that for a whole node; this answers it per
    /// input, which is what a Switch needs: the branches it is not on must
    /// not be walked, or their Reads get reader slots and decode frames
    /// nobody looks at. Everything else samples every input it declares.
    pub fn samples_input(&self, i: usize) -> bool {
        match self {
            NodeKind::Switch(p) => i == p.live(),
            // The reference only names the full format; its pixels are
            // never read.
            NodeKind::Uncrop(_) => i != crate::nodes::uncrop::REF_INPUT,
            _ => true,
        }
    }

    /// The reader slot this node owns, for the kinds whose output is an
    /// uploaded plate (a Read, or anything that plays a bake through one).
    pub fn reader_slot(&self) -> Option<usize> {
        match self {
            NodeKind::ChannelSelect(p) => Some(p.source),
            NodeKind::Read { source, .. }
            | NodeKind::GraphBake { source, .. } => Some(*source),
            NodeKind::GenAi(p) => Some(p.source),
            NodeKind::CorridorKey(p) => Some(p.source),
            NodeKind::Sammie(p) => Some(p.source),
            NodeKind::Flow(p) => Some(p.source),
            NodeKind::MorphDissolve(p) => Some(p.source),
            NodeKind::Rifetime(p) => Some(p.source),
            NodeKind::FrameFix(p) => Some(p.source),
            NodeKind::Upscale(p) => Some(p.source),
            NodeKind::HdrCnn(p) => Some(p.source),
            NodeKind::SeedVr(p) => Some(p.source),
            NodeKind::Depth(p) => Some(p.source),
            NodeKind::LtxHdr(p) => Some(p.source),
            // A Crop owns a slot once it has baked itself.
            NodeKind::SoftCrop(p) => p.bake,
            _ => None,
        }
    }

    /// Where this node's media sits on the timeline: the comp frame its
    /// own first frame plays at, `None` for "frame zero, like everything
    /// that has never been placed".
    ///
    /// ONE rule for every reader-bearing node, because a plate, the channel
    /// selected out of it, its crypto ranks and a bake of the comp above it
    /// must all agree about what frame a frame is.
    pub fn placement(&self) -> Option<i64> {
        match self {
            // A TRIM does not slide the clip: frames 1005..1010 of a plate
            // placed at 1001 play at comp 1006..1011, not from the start of
            // the shot. The player hands the reader the TRIMMED sequence, so
            // its frame one is the trim's first frame — the placement has to
            // carry that offset, or the picture and the range bar
            // (`ranges.rs`, which shifts by both) disagree about where the
            // clip is.
            // With an explicit first frame the reader's sequence IS the
            // first..last window (the app builds it so), so index 0 plays
            // at `first`.
            NodeKind::Read { start, range, first, .. } => Some(first.unwrap_or_else(|| {
                start.unwrap_or(0) + range.map(|(a, _)| a as i64).unwrap_or(0)
            })),
            NodeKind::Cryptomatte(c) => c.start,
            NodeKind::GraphBake { start, .. } => *start,
            NodeKind::CorridorKey(p) => p.start,
            NodeKind::Sammie(p) => p.start,
            NodeKind::Flow(p) => p.start,
            NodeKind::MorphDissolve(p) => p.start,
            NodeKind::Rifetime(p) => p.start,
            NodeKind::FrameFix(p) => p.start,
            NodeKind::Upscale(p) => p.start,
            NodeKind::HdrCnn(p) => p.start,
            NodeKind::SeedVr(p) => p.start,
            NodeKind::Depth(p) => p.start,
            NodeKind::LtxHdr(p) => p.start,
            // A still take holds everywhere; a VIDEO take plays from the
            // tick it was captured at, which is where its first frame
            // belongs.
            NodeKind::GenAi(p) => {
                p.shown_tick.or_else(|| p.pipeline.picked().filter(|t| t.is_video()).map(|t| t.tick))
            }
            _ => None,
        }
    }

    /// Every reader slot a node owns, in `sub` order. Single-slot readers
    /// have one; a Cryptomatte has the beauty plus one per rank.
    pub fn reader_slots(&self) -> Vec<usize> {
        match self {
            NodeKind::Cryptomatte(c) => c.reader_slots(),
            _ => self.reader_slot().into_iter().collect(),
        }
    }

    /// Whether reader slot `sub` is in the rig right now. Always, except
    /// for a Cryptomatte: its live id readers while it is viewed, its one
    /// matte reader otherwise — never both, which is the whole saving.
    pub fn reader_sub_active(&self, sub: usize) -> bool {
        match self {
            NodeKind::Cryptomatte(c) => c.reader_sub_active(sub),
            _ => true,
        }
    }

    pub fn input_count(&self) -> usize {
        match self {
            NodeKind::Read { .. }
            | NodeKind::Grid(_)
            | NodeKind::Gradient(_)
            | NodeKind::Blank(_)
            | NodeKind::GlintKernel(_)
            | NodeKind::BokehKernel(_)
            | NodeKind::Prompt(_)
            | NodeKind::Camera3D(_)
            | NodeKind::LoopIn => 0,
            NodeKind::Card3D(_) => 1,
            NodeKind::Starfield(_) => StarfieldParams::INPUTS,
            NodeKind::Planet(_) => PlanetParams::INPUTS,
            NodeKind::Plasma(_) => PlasmaParams::INPUTS,
            NodeKind::Splat(_) => SplatParams::INPUTS,
            NodeKind::Particles(_) => ParticlesParams::INPUTS,
            NodeKind::ParticleNode(n) => n.inputs().len(),
            NodeKind::GenAiStep(op) => op.ports().len(),
            NodeKind::RotoNode(r) => r.input_count(),
            NodeKind::Scene3D(p) => p.input_count(),
            NodeKind::Render3D(p) => p.input_count(),
            NodeKind::Grain(_) => GrainParams::INPUTS,
            // Body + seed.
            NodeKind::LoopOut(_) => 2,
            // The multi-channel wire in.
            NodeKind::ChannelSelect(_) => 1,
            NodeKind::Breakout(_) => 1,
            // One image input per string of the harp.
            NodeKind::Recombine(p) => p.inputs(),
            // The image, then one matte per output.
            NodeKind::Extract(p) => 1 + p.pairs(),
            NodeKind::Switch(p) => p.n(),
            // A generator that can also consume: the optional matte input is
            // what the shape combines WITH, so several Splines chain into one
            // roto. Unwired it reads as empty coverage and the node is the
            // bare shape it always was.
            // The matte it combines with, and a Camera its shapes are seen
            // through (lowered away by `crate::roto`).
            NodeKind::Spline(_) => 2,
            // A generator with ONE input, and it is a matte: the field graph
            // reads it as a number per pixel, which is how a noise gets to
            // be finer inside a shape.
            NodeKind::Noise(_) => 1,
            // A field node's sockets are its own drivable parameters.
            NodeKind::Field(f) => f.inputs().len(),
            // The one input is the field port: a Noise the geometry reads.
            NodeKind::Geo(_) => 1,
            // Pictures for Blender: named images, one per port.
            NodeKind::Blender(p) => p.input_count(),
            NodeKind::GeoNode(n) => n.inputs().len(),
            // The picture it draws over, then a mask.
            NodeKind::Sdf(_) => SdfParams::INPUTS,
            NodeKind::SdfNode(n) => n.inputs().len(),
            NodeKind::GraphBake { .. } => 1,
            NodeKind::GenAi(p) => p.inputs(),
            NodeKind::CorridorKey(_) => CorridorKeyParams::INPUTS,
            NodeKind::Sammie(_) => SammieParams::INPUTS,
            NodeKind::Flow(_) => FlowParams::INPUTS,
            NodeKind::Rifetime(_) => RifetimeParams::INPUTS,
            NodeKind::Freeze(_) => FreezeParams::INPUTS,
            NodeKind::FrameFix(_) => FrameFixParams::INPUTS,
            NodeKind::VectorBlur(_) => VectorBlurParams::INPUTS,
            NodeKind::VectorWarp(_) => VectorWarpParams::INPUTS,
            // Image + modulate.
            NodeKind::Transform(_) | NodeKind::CornerPin(_) => 2,
            NodeKind::Stabilize(_) => StabilizeParams::INPUTS,
            NodeKind::CamTrack(_) => CamTrackParams::INPUTS,
            NodeKind::PlanarTrack(_) => PlanarTrackParams::INPUTS,
            NodeKind::Paint(_)
            | NodeKind::Curve(_)
            | NodeKind::LumaKey(_)
            | NodeKind::Reformat(_)
            | NodeKind::Tracker(_)
            | NodeKind::Slip(_)
            | NodeKind::TimeRemap(_)
            | NodeKind::Speed(_)
            | NodeKind::RangeRemap(_)
            | NodeKind::Premult { .. }
            | NodeKind::Shuffle(_)
            | NodeKind::Write(_)
            | NodeKind::ToMatte(_)
            | NodeKind::ToData(_)
            | NodeKind::ToRgba(_)
            | NodeKind::SoftCrop(_)
            | NodeKind::MotionBlur(_)
            // A routing anchor: exactly what came in, one wire onward.
            | NodeKind::Dot { .. }
            | NodeKind::TimeRange(_) => 1,
            // Image input 0 + mask input 1.
            NodeKind::Grade(_) | NodeKind::Mult(_) | NodeKind::MatteIn => 2,
            NodeKind::Saturation(_) => SatParams::INPUTS,
            NodeKind::Contrast(_) => ContrastParams::INPUTS,
            NodeKind::ColorCompress(_) => ColorCompressParams::INPUTS,
            NodeKind::HueRotate(_) => HueRotateParams::INPUTS,
            NodeKind::Lut(_) => LutParams::INPUTS,
            NodeKind::Text(_) => TextParams::INPUTS,
            NodeKind::Burnin(_) => BurninParams::INPUTS,
            NodeKind::Mirror(_) => MirrorParams::INPUTS,
            NodeKind::FlipFlop(_) => FlipFlopParams::INPUTS,
            NodeKind::Turn(_) => TurnParams::INPUTS,
            NodeKind::Uncrop(_) => UncropParams::INPUTS,
            NodeKind::Compare(_) => CompareParams::INPUTS,
            NodeKind::MorphDissolve(_) => MorphDissolveParams::INPUTS,
            NodeKind::Gamma(_) => GammaParams::INPUTS,
            NodeKind::ToneMap(_) => ToneMapParams::INPUTS,
            NodeKind::UvWarp(_) => UvWarpParams::INPUTS,
            NodeKind::Env(_) => EnvParams::INPUTS,
            NodeKind::Refract(_) => RefractParams::INPUTS,
            NodeKind::GridWarp(_) => GridWarpParams::INPUTS,
            NodeKind::LensDistort(_) => LensDistortParams::INPUTS,
            NodeKind::Warp(_) => WarpParams::INPUTS,
            NodeKind::Expression(_) => ExpressionParams::INPUTS,
            NodeKind::Despill(_) => DespillParams::INPUTS,
            NodeKind::Lightwrap(_) => LightwrapParams::INPUTS,
            NodeKind::Upscale(_) => UpscaleParams::INPUTS,
            NodeKind::HdrCnn(_) => HdrCnnParams::INPUTS,
            NodeKind::SeedVr(_) => SeedVrParams::INPUTS,
            NodeKind::Depth(_) => DepthParams::INPUTS,
            NodeKind::LtxHdr(_) => LtxHdrParams::INPUTS,
            NodeKind::ScreenKey(_) => ScreenKeyParams::INPUTS,
            NodeKind::ColorSlice(_) => ColorSliceParams::INPUTS,
            NodeKind::Blur(_) => BlurParams::INPUTS,
            NodeKind::Glow(_) => GlowParams::INPUTS,
            NodeKind::Godrays(_) => GodraysParams::INPUTS,
            NodeKind::LensFlare(_) => LensFlareParams::INPUTS,
            NodeKind::EdgeExtend(_) => EdgeExtendParams::INPUTS,
            NodeKind::Glint(_) => GlintParams::INPUTS,
            NodeKind::EdgeDetect(_) => EdgeDetectParams::INPUTS,
            NodeKind::Median(_) => MedianParams::INPUTS,
            NodeKind::HighlightRebuild(_) => HighlightRebuildParams::INPUTS,
            NodeKind::Convolve(_) => ConvolveParams::INPUTS,
            NodeKind::Diffuse(_) => DiffuseParams::INPUTS,
            NodeKind::Dilate(_) => DilateParams::INPUTS,
            // Image 0 + mask 1 + depth 2 + bokeh kernel 3 + modulate 4.
            // Depth is a full image port (an arbitrary branch renders into
            // it); the kernel must be a SOURCE node, so it stays at its own
            // small size instead of being resolved at canvas resolution.
            NodeKind::Defocus(_) => 5,
            // Image input 0 + garbage matte on input 1 (multiplied into the
            // key, so a rough shape can rule out the half of the frame that
            // happens to share the colour).
            NodeKind::Keyer(_) => 2,
            // ONE input, the multi-channel cable. The ranks come off it as
            // the node's own reader slots; the edge itself is dropped at
            // lowering, so the engine never samples it.
            NodeKind::Cryptomatte(_) => 1,
            NodeKind::Mask => 2,
            // Two MATTE inputs (neither is a mask that limits the node —
            // they are the operands), so `mask_input` stays None and
            // `input_kind` answers Mask for both.
            NodeKind::MatteCombine(_) => 2,
            // A on 0, B on 1, and a matte on 2 that limits the blend to
            // where it covers (the only way Replace knows where to land).
            NodeKind::Blend { .. } => 3,
            NodeKind::Mix(_) => 2,
            // Declared by the plugin; a missing plugin keeps one input so
            // bypass/passthrough still has something to pass through.
            NodeKind::Custom { type_id, .. } => {
                crate::plugins::get(type_id).map(|d| d.inputs).unwrap_or(1)
            }
            // Declared by the shader's own `input:` pragmas. Read from the
            // PARSE, not the compile, so a shader that is mid-edit and does
            // not validate keeps its ports — and therefore its wires.
            NodeKind::Shader(s) => s.def().inputs.len().max(1),
            // One port per layer — the whole point of the node. Adding a
            // layer is therefore a topology change, like a spline gaining a
            // point; moving one never is.
            NodeKind::Layer(p) => p.layers.len().min(LAYER_MAX),
        }
    }

    /// The pass chain a node runs before the main pass, for the kinds that
    /// take one. `None` is a node that fuses.
    pub fn stage_def(&self) -> Option<&dyn crate::nodes::stage::StageDef> {
        match self {
            NodeKind::Blur(p) => Some(p),
            NodeKind::Dilate(p) => Some(p),
            NodeKind::Glow(p) => Some(p),
            NodeKind::Contrast(p) => p.local.then_some(p as &dyn crate::nodes::stage::StageDef),
            NodeKind::Godrays(p) => Some(p),
            NodeKind::EdgeExtend(p) => Some(p),
            NodeKind::Glint(p) => Some(p),
            NodeKind::EdgeDetect(p) => Some(p),
            NodeKind::Median(p) => Some(p),
            NodeKind::HighlightRebuild(p) => Some(p),
            NodeKind::Convolve(p) => Some(p),
            NodeKind::Diffuse(p) => Some(p),
            NodeKind::Lightwrap(p) => Some(p),
            NodeKind::MotionBlur(p) => Some(p),
            NodeKind::VectorBlur(p) => Some(p),
            NodeKind::Render3D(p) => p.stage().map(|p| p as &dyn crate::nodes::stage::StageDef),
            _ => None,
        }
    }

    /// The input slot carrying a mask, for kinds that take one.
    pub fn mask_input(&self) -> Option<usize> {
        match self {
            NodeKind::Grade(_) | NodeKind::Defocus(_) | NodeKind::Mult(_) | NodeKind::MatteIn => Some(1),
            NodeKind::Saturation(_) => SatParams::MASK_INPUT,
            NodeKind::Contrast(_) => ContrastParams::MASK_INPUT,
            NodeKind::ColorCompress(_) => ColorCompressParams::MASK_INPUT,
            NodeKind::HueRotate(_) => HueRotateParams::MASK_INPUT,
            NodeKind::Lut(_) => LutParams::MASK_INPUT,
            NodeKind::Text(_) => TextParams::MASK_INPUT,
            NodeKind::Burnin(_) => BurninParams::MASK_INPUT,
            NodeKind::Mirror(_) => MirrorParams::MASK_INPUT,
            NodeKind::FlipFlop(_) => FlipFlopParams::MASK_INPUT,
            NodeKind::Turn(_) => TurnParams::MASK_INPUT,
            NodeKind::Uncrop(_) => UncropParams::MASK_INPUT,
            NodeKind::Compare(_) => CompareParams::MASK_INPUT,
            NodeKind::MorphDissolve(_) => MorphDissolveParams::MASK_INPUT,
            NodeKind::Gamma(_) => GammaParams::MASK_INPUT,
            NodeKind::ToneMap(_) => ToneMapParams::MASK_INPUT,
            NodeKind::UvWarp(_) => UvWarpParams::MASK_INPUT,
            NodeKind::Env(_) => EnvParams::MASK_INPUT,
            NodeKind::Refract(_) => RefractParams::MASK_INPUT,
            NodeKind::VectorWarp(_) => VectorWarpParams::MASK_INPUT,
            NodeKind::VectorBlur(_) => VectorBlurParams::MASK_INPUT,
            NodeKind::GridWarp(_) => GridWarpParams::MASK_INPUT,
            NodeKind::LensDistort(_) => LensDistortParams::MASK_INPUT,
            NodeKind::Warp(_) => WarpParams::MASK_INPUT,
            NodeKind::Expression(_) => ExpressionParams::MASK_INPUT,
            NodeKind::Despill(_) => DespillParams::MASK_INPUT,
            NodeKind::Lightwrap(_) => LightwrapParams::MASK_INPUT,
            NodeKind::ScreenKey(_) => ScreenKeyParams::MASK_INPUT,
            NodeKind::ColorSlice(_) => ColorSliceParams::MASK_INPUT,
            NodeKind::Blur(_) => BlurParams::MASK_INPUT,
            NodeKind::Glow(_) => GlowParams::MASK_INPUT,
            NodeKind::Godrays(_) => GodraysParams::MASK_INPUT,
            NodeKind::LensFlare(_) => LensFlareParams::MASK_INPUT,
            NodeKind::EdgeExtend(_) => EdgeExtendParams::MASK_INPUT,
            NodeKind::Glint(_) => GlintParams::MASK_INPUT,
            NodeKind::EdgeDetect(_) => EdgeDetectParams::MASK_INPUT,
            NodeKind::Median(_) => MedianParams::MASK_INPUT,
            NodeKind::HighlightRebuild(_) => HighlightRebuildParams::MASK_INPUT,
            NodeKind::Convolve(_) => ConvolveParams::MASK_INPUT,
            NodeKind::Diffuse(_) => DiffuseParams::MASK_INPUT,
            NodeKind::Grain(_) => GrainParams::MASK_INPUT,
            NodeKind::Dilate(_) => DilateParams::MASK_INPUT,
            NodeKind::Mask | NodeKind::Keyer(_) => Some(1),
            NodeKind::Noise(_) => Some(0),
            NodeKind::Sdf(_) => SdfParams::MASK_INPUT,
            NodeKind::Blend { .. } => Some(2),
            // The Spline's only input IS a matte — the one it combines with.
            NodeKind::Spline(_) => Some(0),
            // Declared by the plugin: a tool with a matte input proxy.
            NodeKind::Custom { type_id, .. } => {
                crate::plugins::get(type_id).and_then(|d| d.mask_input)
            }
            _ => None,
        }
    }

    /// The input a BYPASSED node passes through. Input 0 for everything
    /// but a Blend, whose bypass is "the merge is off": the background
    /// (input 1) stays and the foreground goes, as in every compositor.
    pub fn bypass_input(&self) -> usize {
        match self {
            NodeKind::Blend { .. } => 1,
            _ => 0,
        }
    }

    /// The input slot carrying a MODULATE matte, for kinds that take one:
    /// coverage that scales the effect per pixel rather than mixing it back
    /// against the input (that is `mask_input`). The wire is a matte either
    /// way — `input_kind` says Mask — and the socket is what tells them
    /// apart.
    pub fn mod_input(&self) -> Option<usize> {
        match self {
            NodeKind::Blur(_) => BlurParams::MOD_INPUT,
            NodeKind::Dilate(_) => DilateParams::MOD_INPUT,
            NodeKind::VectorBlur(_) => VectorBlurParams::MOD_INPUT,
            NodeKind::Diffuse(_) => DiffuseParams::MOD_INPUT,
            NodeKind::Convolve(_) => ConvolveParams::MOD_INPUT,
            // A matte scaling the circle of confusion, past the depth pass.
            NodeKind::Defocus(_) => Some(crate::codegen::DEFOCUS_MOD_INPUT),
            // Image, then the matte that says how much of the move applies.
            NodeKind::Transform(_) | NodeKind::CornerPin(_) => Some(1),
            _ => None,
        }
    }

    /// Whether input `i` rests on the node's SIDE: a mask or a modulate
    /// matte — something that limits or scales the effect, not the picture
    /// it acts on.
    pub fn is_side_input(&self, i: usize) -> bool {
        self.mask_input() == Some(i) || self.mod_input() == Some(i)
    }

    /// Does input `i` FOLLOW ITS PIPE — take a picture or a matte, and be
    /// that on both sides? A Bake and a SoftCrop pass pixels through; the
    /// pointwise ops (Gamma, Mult, Grade, Curve, and a plugin whose manifest
    /// says `"mattes": true`, like Invert) work on a matte's coverage as
    /// they would on a picture's rgb. Which pipe one is in right now is
    /// [`Self::matte_mode`], kept true to the wire by the app.
    pub fn follows_pipe(&self, input: usize) -> bool {
        input == 0
            && match self {
                NodeKind::GraphBake { .. }
                | NodeKind::SoftCrop(_)
                | NodeKind::Gamma(_)
                | NodeKind::Mult(_)
                | NodeKind::Grade(_)
                | NodeKind::Curve(_)
                // Filters and transforms move coverage as they move rgb.
                | NodeKind::Blur(_)
                | NodeKind::Median(_)
                | NodeKind::Convolve(_)
                | NodeKind::EdgeDetect(_)
                | NodeKind::VectorBlur(_)
                | NodeKind::MotionBlur(_)
                | NodeKind::Transform(_)
                | NodeKind::CornerPin(_)
                | NodeKind::Mirror(_)
                | NodeKind::FlipFlop(_)
                | NodeKind::Turn(_) => true,
                NodeKind::Custom { type_id, .. } => {
                    crate::plugins::get(type_id).is_some_and(|d| d.mattes)
                }
                _ => false,
            }
    }

    /// Is this pipe-following node sitting in a MATTE pipe? Its input 0 and
    /// its output are then `PortKind::Mask`.
    pub fn matte_mode(&self) -> bool {
        match self {
            NodeKind::GraphBake { mask, .. } => *mask,
            NodeKind::SoftCrop(p) => p.matte,
            NodeKind::Gamma(p) => p.matte,
            NodeKind::Mult(p) => p.matte,
            NodeKind::Grade(p) => p.matte,
            NodeKind::Curve(p) => p.matte,
            NodeKind::Blur(p) => p.matte,
            NodeKind::Median(p) => p.matte,
            NodeKind::Convolve(p) => p.matte,
            NodeKind::EdgeDetect(p) => p.matte,
            NodeKind::VectorBlur(p) => p.matte,
            NodeKind::MotionBlur(p) => p.matte,
            NodeKind::Transform(p) => p.matte,
            NodeKind::CornerPin(p) => p.matte,
            NodeKind::Mirror(p) => p.matte,
            NodeKind::FlipFlop(p) => p.matte,
            NodeKind::Turn(p) => p.matte,
            NodeKind::Custom { params, .. } => {
                params.get(CUSTOM_MATTE_KEY).is_some_and(|v| *v != 0.0)
            }
            _ => false,
        }
    }

    /// Put a pipe-following node in the matte pipe or take it out. A no-op
    /// on anything that does not follow its pipe.
    pub fn set_matte_mode(&mut self, on: bool) {
        if !self.follows_pipe(0) {
            return;
        }
        match self {
            NodeKind::GraphBake { mask, .. } => *mask = on,
            NodeKind::SoftCrop(p) => p.matte = on,
            NodeKind::Gamma(p) => p.matte = on,
            NodeKind::Mult(p) => p.matte = on,
            NodeKind::Grade(p) => p.matte = on,
            NodeKind::Curve(p) => p.matte = on,
            NodeKind::Blur(p) => p.matte = on,
            NodeKind::Median(p) => p.matte = on,
            NodeKind::Convolve(p) => p.matte = on,
            NodeKind::EdgeDetect(p) => p.matte = on,
            NodeKind::VectorBlur(p) => p.matte = on,
            NodeKind::MotionBlur(p) => p.matte = on,
            NodeKind::Transform(p) => p.matte = on,
            NodeKind::CornerPin(p) => p.matte = on,
            NodeKind::Mirror(p) => p.matte = on,
            NodeKind::FlipFlop(p) => p.matte = on,
            NodeKind::Turn(p) => p.matte = on,
            NodeKind::Custom { params, .. } => {
                if on {
                    params.insert(CUSTOM_MATTE_KEY.into(), 1.0);
                } else {
                    params.remove(CUSTOM_MATTE_KEY);
                }
            }
            _ => {}
        }
    }

    /// Does this node COMPUTE on a matte (rather than pass it through)? Its
    /// result is then re-broadcast so the matte stays a matte: coverage in
    /// rgb (what consumers read) AND alpha (what the viewer's alpha shows).
    /// Structural: it joins the topology hash.
    pub fn recolours_matte(&self) -> bool {
        self.matte_mode() && !matches!(self, NodeKind::GraphBake { .. } | NodeKind::SoftCrop(_))
    }

    /// May a wire of `given` plug into `input`? [`PortKind::can_feed`],
    /// plus a pipe-following port's either-pipe welcome (the app retypes
    /// the node on connect). Every wiring gesture asks this.
    pub fn accepts(&self, input: usize, given: PortKind) -> bool {
        given.can_feed(self.input_kind(input))
            || (self.follows_pipe(input) && matches!(given, PortKind::Image | PortKind::Mask))
            || self.writes_data(input, given)
    }

    /// A Render (Write) also takes DATA: it writes it as one float channel,
    /// `Z`, with no colour transform. Its own port, not a `can_feed` rule:
    /// data into a colour node would be read as a picture.
    pub fn writes_data(&self, input: usize, given: PortKind) -> bool {
        matches!(self, NodeKind::Write(_)) && input == 0 && given == PortKind::Data
    }

    /// Does this node take its FORMAT and frame range from input `i`? A
    /// picture input, or a matte that is the node's main operand rather
    /// than a side input: a Dilate's or a MatteCombine's matte IS the
    /// picture there, and skipping it handed them the canvas instead.
    /// The resolution and range walks take the first connected one.
    pub fn inherits_from(&self, i: usize) -> bool {
        let k = self.input_kind(i);
        k.is_picture() || (k == PortKind::Mask && !self.is_side_input(i))
    }

    /// Are this node's inputs PEERS for its frame range — pictures of equal
    /// standing whose spans union (a Blend's A and B, a Layer stack, a
    /// harp's strings)? Everything else has a PLATE on input 0 and its
    /// other inputs are elements it reads (a Convolve's kernel, a Defocus's
    /// depth, a VectorBlur's vectors, a Keyer's garbage matte, a
    /// Lightwrap's background): those must not stretch the shot, or a
    /// one-frame kernel at frame 0 drags a 1001 plate's range back to 0.
    pub fn range_peers(&self) -> bool {
        matches!(
            self,
            NodeKind::Blend { .. }
                | NodeKind::Mix(_)
                | NodeKind::MorphDissolve(_)
                | NodeKind::MatteCombine(_)
                | NodeKind::Switch(_)
                | NodeKind::Recombine(_)
                | NodeKind::Layer(_)
                | NodeKind::LoopOut(_)
                | NodeKind::Scene3D(_)
                | NodeKind::Render3D(_)
                | NodeKind::Blender(_)
                // A plugin or subwob declares its ports; we cannot tell a
                // plate from an element there, so its inputs stay peers.
                | NodeKind::Custom { .. }
        )
    }

    /// The inputs the frame-range walk takes, given which are wired. Peers
    /// all count; otherwise the plate (input 0) alone, falling back to the
    /// other inputs only when no plate is wired.
    pub fn range_inputs(&self, wired: impl Fn(usize) -> bool) -> Vec<usize> {
        let all: Vec<usize> = (0..self.input_count())
            .filter(|&i| self.inherits_from(i) && self.samples_input(i) && wired(i))
            .collect();
        if self.range_peers() || !all.contains(&0) {
            all
        } else {
            vec![0]
        }
    }

    /// How many outputs this kind has. One for everything except a
    /// multi-band LumaKey — and it is one for THAT too by the time codegen
    /// sees it, because [`crate::lumabands`] lowers a multi-band key into
    /// one single-band key per consumed band first (the same trick the tool
    /// nodes and Dots use: the engine never learns the concept).
    pub fn output_count(&self) -> usize {
        match self {
            NodeKind::LumaKey(k) => k.bands.len().max(1),
            // The combined picture, then one per pass (lowered by
            // `blender::expand` before the engine counts).
            NodeKind::Blender(p) => 1 + p.passes.len(),
            // The comp, then each layer alone (lowered by `layerouts::expand`).
            NodeKind::Layer(p) if p.outputs => 1 + p.layers.len().min(LAYER_MAX),
            // One nodule per chosen layer, plus the remainder's (lowered
            // to the kids before the engine ever counts).
            NodeKind::Breakout(p) => p.strings(),
            // One isolated image per matte (lowered to Masks, like the key).
            NodeKind::Extract(p) => p.pairs(),
            // The picture, then its depth (lowered by `scene3d::expand`).
            NodeKind::Render3D(p) if p.lowered.is_none() => 2,
            // The same for a Splat (lowered by `nodes::splat::lower`).
            NodeKind::Splat(p) if p.depth_of.is_none() => 2,
            _ => 1,
        }
    }

    /// Image inputs only (mask and modulate slots excluded) —
    /// layout/eligibility rules.
    pub fn image_input_count(&self) -> usize {
        self.input_count()
            - self.mask_input().is_some() as usize
            - self.mod_input().is_some() as usize
    }

    /// Where `input` sits among the IMAGE inputs, ignoring the mask slot.
    /// Ports splay across the top by this ordinal, not by the raw index —
    /// otherwise a kind whose mask sits in the middle (Defocus: image 0,
    /// mask 1, depth 2, kernel 3) splays its last port off the end.
    pub fn image_slot(&self, input: usize) -> usize {
        let below = |p: Option<usize>| p.is_some_and(|m| input > m) as usize;
        input - below(self.mask_input()) - below(self.mod_input())
    }

    /// A single glyph painted ON the input socket, for kinds where the ports
    /// are not interchangeable and the shape alone will not tell you which
    /// is which. `None` draws a plain socket. Keep these to one character:
    /// the tag IS the nodule (see `stalk` in the editor).
    pub fn input_tag(&self, input: usize) -> Option<&'static str> {
        if self.mask_input() == Some(input) {
            return None; // masks are already a distinct colour
        }
        // A modulate matte is the same colour as a mask (it IS one on the
        // wire), so it takes the glyph: it varies the effect.
        if self.mod_input() == Some(input) {
            return Some("~");
        }
        match self {
            // + is the foreground (what goes OVER), - the background (what
            // it lands on): the sign says which way the blend runs without
            // needing to remember an alphabet.
            NodeKind::Blend { .. } => Some(if input == 0 { "+" } else { "−" }),
            // The bookend carries the rgba the strings were broken out of;
            // it is where the rebuild's alpha comes from.
            NodeKind::Recombine(p) => (input == p.base_input()).then_some("a"),
            // The UV map wears S: which port is the warp.
            NodeKind::UvWarp(_) => (input == 1).then_some("uv"),
            NodeKind::Uncrop(_) => (input == crate::nodes::uncrop::REF_INPUT).then_some("f"),
            NodeKind::Compare(_) => Some(if input == 0 { "A" } else { "B" }),
            NodeKind::MorphDissolve(_) => (input < 2).then_some(if input == 0 { "A" } else { "B" }),
            // The surface map wears N: which port is the glass.
            NodeKind::Refract(_) => (input == crate::nodes::refract::MAP_INPUT).then_some("n"),
            // The vectors port wears its letter: which wire says how
            // things moved.
            NodeKind::VectorBlur(_) | NodeKind::VectorWarp(_) => (input == 1).then_some("v"),
            NodeKind::Defocus(_) => match input {
                2 => Some("Z"),
                3 => Some("B"),
                _ => None,
            },
            NodeKind::Convolve(_) => {
                (input == crate::nodes::convolve::KERNEL_INPUT).then_some("K")
            }
            NodeKind::Diffuse(_) => {
                (input == crate::nodes::diffuse::DEPTH_INPUT).then_some("Z")
            }
            // The stack order IS the port order, so each socket wears its
            // layer number (1 = top, AE numbering).
            NodeKind::Layer(_) => Some(LAYER_TAGS.get(input).copied().unwrap_or("+")),
            // Each port is the image `wobble_in_N` in Blender: the socket
            // wears N.
            NodeKind::Blender(_) => Some(["0", "1", "2", "3", "4", "5", "6", "7"][input.min(7)]),
            // A shader's ports are named by its own manifest, and those
            // names are not 'static — the socket wears its index and the
            // name shows in `input_label`.
            NodeKind::Shader(s) => (s.def().inputs.len() > 1)
                .then(|| LAYER_TAGS.get(input).copied().unwrap_or("+")),
            _ => None,
        }
    }

    /// Display name of an input, including the ones a node names at runtime
    /// (a Shader node's declared ports, a Switch's or a Layer's numbered
    /// ones). `input_name` stays `&'static str` for everything that can be
    /// static.
    pub fn input_label(&self, input: usize) -> String {
        match self {
            NodeKind::Shader(s) => s
                .def()
                .inputs
                .get(input)
                .cloned()
                .unwrap_or_else(|| "input".into()),
            NodeKind::Switch(_) => format!("input {}", input + 1),
            NodeKind::Layer(_) => format!("layer {} (1 = top)", input + 1),
            NodeKind::Extract(_) if input > 0 => format!("matte {input}"),
            NodeKind::Recombine(p) if input < p.base_input() => format!("string {}", input + 1),
            NodeKind::Recombine(_) => "base rgba".into(),
            NodeKind::Scene3D(_) => format!("scene input {}", input + 1),
            _ => self.input_name(input).to_string(),
        }
    }

    /// What an input is FOR, in a sentence: the hover tip on its socket.
    /// `None` leaves the tip to the pipe's own words — a mask or a modulate
    /// port, which mean the same thing on every node that has one.
    pub fn input_about(&self, input: usize) -> Option<String> {
        let s = |t: &str| Some(t.to_string());
        if self.mod_input() == Some(input) {
            return None;
        }
        // Mask ports that are NOT "mix the effect back": the node's whole
        // job is the matte, so the generic words would be wrong.
        match self {
            NodeKind::Noise(_) => {
                return s("a matte the field graph reads as a number per pixel (its \
                          Field In) — e.g. a finer noise inside a shape")
            }
            NodeKind::Spline(_) if input == 1 => {
                return s("a Camera the shapes are seen through: with one, the roto's \
                          Axis nodes move its shapes in 3D; unwired, flat")
            }
            NodeKind::Spline(_) => {
                return s("a matte this shape combines with, so several Splines chain \
                          into one roto; unwired, the bare shape")
            }
            NodeKind::MatteIn if input == 1 => {
                return s("the matte that becomes the picture's alpha (colour untouched); \
                          unwired, opaque")
            }
            NodeKind::Mask if input == 1 => {
                return s("coverage the picture is multiplied by: white keeps, black \
                          clears; unwired, everything kept")
            }
            NodeKind::Keyer(_) if input == 1 => {
                return s("garbage matte, multiplied into the key: a rough shape that \
                          rules out the parts of the frame sharing the key colour")
            }
            _ => {}
        }
        if self.mask_input() == Some(input) {
            return None;
        }
        // Sub-graph nodes: a socket IS a parameter (or the thing worked on).
        match self {
            NodeKind::Field(_) | NodeKind::SdfNode(_) | NodeKind::GeoNode(_) => {
                let name = self.input_name(input);
                return match self.input_kind(input) {
                    PortKind::Geo => Some(format!("the geometry this node works on ({name})")),
                    PortKind::Image => Some(format!("{name}: a colour, per pixel")),
                    // The operand itself, not a parameter.
                    _ if matches!(self, NodeKind::SdfNode(_))
                        && matches!(name, "shape" | "a" | "b") =>
                    {
                        Some(format!("the shape (a distance) this node works on ({name})"))
                    }
                    _ if name == "field" => Some("the field this node works on".to_string()),
                    _ => Some(format!(
                        "drives {name}: wire a value or a field to vary it; \
                         unwired, the panel's value"
                    )),
                };
            }
            _ => {}
        }
        match self {
            NodeKind::Geo(_) => s("a Noise node's field, read by the drawing (scatter \
                                   density, displacement…)"),
            NodeKind::Sdf(_) => s("the picture the shapes are drawn over (and that \
                                   Refract bends)"),
            NodeKind::GenAi(p) => match (p.pipeline.in_kind(input), input) {
                (genai::InKind::Image, 0) => s("the picture sent with the request, captured at the \
                        playhead (or over a range, as a clip) — and passed through until \
                        something is generated"),
                (genai::InKind::Image, _) => s("another picture for a model that takes one: the look \
                        to match, the style or the object to draw on"),
                (genai::InKind::Matte, _) => s("where to work: white is generated anew, black kept"),
            },
            NodeKind::Card3D(_) => s("the picture on the card"),
            NodeKind::Splat(_) => s("the Camera to look through (a CameraTrack's, a file's, Blender's); \
                                    unwired, the default Camera"),
            NodeKind::Starfield(_) => s("a Camera to look through (a CameraTrack's, a file's, Blender's): \
                                        its rotation and lens drive the sky; unwired, the node's own"),
            NodeKind::Scene3D(_) => s("a Camera, a Card or another Scene, gathered \
                                       into this one"),
            NodeKind::Render3D(_) => {
                if input == crate::nodes::render3d::CAMERA_PORT {
                    s("the Camera to render through")
                } else {
                    s("the cards to render: a Card or a Scene")
                }
            }
            NodeKind::Defocus(_) => match input {
                0 => s("the picture to defocus"),
                2 => s("depth (Z) per pixel — a Depth node or a render's Z: each \
                        pixel's blur comes from its distance to the focus; unwired, \
                        one even blur"),
                _ => s("the aperture shape every highlight takes (a Bokeh Kernel); \
                        unwired, a disc"),
            },
            NodeKind::Convolve(_) => {
                if input == crate::nodes::convolve::KERNEL_INPUT {
                    s("an image used as the convolution kernel, centred on its middle \
                       (a Glint Kernel, or any picture)")
                } else {
                    s("the picture to convolve")
                }
            }
            NodeKind::Diffuse(_) => {
                if input == crate::nodes::diffuse::DEPTH_INPUT {
                    s("depth (Z) per pixel: ramps the diffusion from near to far")
                } else {
                    s("the picture to diffuse")
                }
            }
            NodeKind::Godrays(_) => {
                if input == crate::nodes::godrays::SOURCE_INPUT {
                    s("a matte of what casts the rays (a sky key, a window); unwired, \
                       the picture's own brights")
                } else {
                    s("the picture the shafts are added to")
                }
            }
            NodeKind::EdgeExtend(_) => {
                if input == crate::nodes::edgeextend::MATTE_INPUT {
                    s("a matte that replaces the picture's alpha as the edge to extend \
                       from")
                } else {
                    s("the picture whose edge colour is pushed outward")
                }
            }
            NodeKind::Lightwrap(_) => {
                if input == 0 {
                    s("the foreground (premultiplied, with its alpha)")
                } else {
                    s("the background it will sit over: its light bleeds round the \
                       foreground's edge")
                }
            }
            NodeKind::MorphDissolve(_) => match input {
                0 => s("A: the outgoing picture, what the transition starts on"),
                1 => s("B: the incoming picture, what the transition lands on"),
                _ => s("where the transition applies; outside it, A"),
            },
            NodeKind::Compare(_) => {
                if input == 0 {
                    s("A: shown alone, or on the front of the wipe (left of a vertical line)")
                } else {
                    s("B: shown alone, or behind the wipe; scaled into the larger input's format")
                }
            }
            NodeKind::Uncrop(_) => {
                if input == 0 {
                    s("the cropped picture coming back (a render of the Crop's box)")
                } else {
                    s("names the full format to put it back into; its pixels are never \
                       read (the plate the Crop cut from)")
                }
            }
            NodeKind::UvWarp(_) => {
                if input == 0 {
                    s("the picture to warp")
                } else {
                    s("where each pixel samples from: u in red, v in green, 0..1 \
                       across the frame (a Gradient, a render's UV pass)")
                }
            }
            NodeKind::Env(_) => {
                if input == crate::nodes::env::IMAGE_INPUT {
                    s("the latlong (equirectangular) picture: an HDRI, a sky, a 360 plate")
                } else {
                    s("a Camera to look through (a CameraTrack's, a file's, Blender's): \
                       its rotation and lens drive the view; unwired, the node's own")
                }
            }
            NodeKind::Refract(_) => match input {
                crate::nodes::refract::IMAGE_INPUT => {
                    s("what is behind the glass: a plate, or a latlong environment (pick \
                       which in the panel)")
                }
                crate::nodes::refract::MAP_INPUT => {
                    s("the glass's surface: a height map (red) or a tangent-space normal \
                       map. Load a normal map raw, with no colour transform")
                }
                _ => s("a Camera for the latlong view: its rotation and lens; unwired, the \
                        node's own"),
            },
            NodeKind::VectorBlur(_) | NodeKind::VectorWarp(_) => {
                if input == 0 {
                    s("the picture to move")
                } else {
                    s("motion vectors, off a Flow node (or a render's vector pass)")
                }
            }
            NodeKind::Stabilize(_) => {
                if input == 0 {
                    s("the plate to stabilise")
                } else {
                    s("pixels under this matte are kept out of the camera solve \
                       (actors, cars — anything that moves on its own)")
                }
            }
            NodeKind::CamTrack(_) => {
                if input == 0 {
                    s("the plate to solve the camera from (it passes through)")
                } else {
                    s("pixels under this matte are not tracked \
                       (actors, cars — anything that moves on its own)")
                }
            }
            NodeKind::ScreenKey(_) => match input {
                0 => s("the green- or blue-screen plate"),
                1 => s("holdout: always kept, whatever the colour says"),
                _ => s("garbage: always dropped, whatever the colour says"),
            },
            NodeKind::CorridorKey(_) => {
                if input == 0 {
                    s("the screen plate to key")
                } else {
                    s("a coarse matte to guide the network; unwired, a ScreenKey \
                       makes one")
                }
            }
            NodeKind::Sammie(_) => s("the plate to roto; click points on it in the \
                                      viewer"),
            NodeKind::Flow(_) => s("the plate whose motion is analysed into vectors"),
            NodeKind::Rifetime(_) => s("the clip to retime: in-betweens are made from \
                                        its frames"),
            NodeKind::Freeze(_) => s("the clip to hold: every frame shows its \
                                      frame at the freeze frame"),
            NodeKind::FrameFix(_) => s("the clip to repair: the listed frames are \
                                        rebuilt from their good neighbours"),
            NodeKind::Upscale(_) => s("the picture to upscale, rendered at its own \
                                       format"),
            NodeKind::HdrCnn(_) => s("the SDR plate whose clipped highlights are \
                                      rebuilt"),
            NodeKind::SeedVr(_) => s("the plate to restore"),
            NodeKind::Depth(_) => s("the plate to estimate depth from"),
            NodeKind::LtxHdr(_) => s("the SDR plate to lift to HDR"),
            NodeKind::Cryptomatte(_) => s("a multi-layer Load's channels cable, \
                                           carrying the Cryptomatte layers"),
            NodeKind::Breakout(_) | NodeKind::ChannelSelect(_) => {
                s("a multi-layer Load's channels cable: every layer of the file")
            }
            NodeKind::MatteCombine(_) => {
                if input == 0 {
                    s("A: the matte B is combined into (mix 0 gives A back); \
                       unwired, empty")
                } else {
                    s("B: combined into A by the op; unwired, empty")
                }
            }
            NodeKind::Mask => s("the picture to cut out"),
            NodeKind::Extract(_) => {
                if input == 0 {
                    s("the picture each matte isolates")
                } else {
                    Some(format!("isolates the picture inside this matte, on output {input}"))
                }
            }
            NodeKind::Dilate(_) => s("the matte to grow or shrink"),
            NodeKind::Blend { .. } => {
                if input == 0 {
                    s("what goes over")
                } else {
                    s("what it lands on")
                }
            }
            NodeKind::Mix(_) => {
                if input == 0 {
                    s("shown at amount 0")
                } else {
                    s("shown at amount 1")
                }
            }
            NodeKind::Layer(p) => {
                if input + 1 == p.layers.len().min(LAYER_MAX) {
                    s("the bottom of the stack; placed by an upstream Transform")
                } else {
                    s("stacked over the layers below it; placed by an upstream \
                       Transform")
                }
            }
            NodeKind::Switch(_) => Some(format!("shown when the switch picks {}", input + 1)),
            NodeKind::Recombine(p) => {
                if input == p.base_input() {
                    s("the rgba the strings were broken out of: alpha comes from here")
                } else {
                    s("one string of the harp, rebuilt into the rgba")
                }
            }
            NodeKind::LoopOut(_) => {
                if input == 0 {
                    s("the end of the loop body, fed back into the LoopIn each \
                       iteration")
                } else {
                    s("the picture the first iteration starts from")
                }
            }
            NodeKind::Shader(_) => Some(format!(
                "{}: a picture the shader samples (its `input:` pragma)",
                self.input_label(input)
            )),
            NodeKind::Custom { .. } => s("a picture this plugin reads"),
            NodeKind::Blender(_) => Some(format!(
                "sent to Blender as the image {}",
                self.input_name(input)
            )),
            NodeKind::Write(_) => s("the picture rendered to files"),
            NodeKind::GraphBake { .. } => s("the branch baked to disk; downstream then \
                                             plays the files"),
            NodeKind::Paint(_) => s("the picture painted over"),
            NodeKind::Tracker(_) => s("the plate to track"),
            NodeKind::PlanarTrack(_) => {
                if input == 0 {
                    s("the plate to track")
                } else {
                    s("pixels under this matte are not the plane: a hand, a pole, \
                       anything in front of it sits out the track")
                }
            }
            NodeKind::Dot { .. } => s("passed on unchanged: a routing anchor"),
            NodeKind::Slip(_)
            | NodeKind::Speed(_)
            | NodeKind::TimeRemap(_)
            | NodeKind::RangeRemap(_)
            | NodeKind::TimeRange(_) => s("the clip to retime"),
            NodeKind::MotionBlur(_) => s("the branch to blur: everything upstream \
                                          renders once per shutter sample"),
            NodeKind::ToMatte(_) => s("the picture to turn into a matte"),
            NodeKind::ToData(_) => s("the picture to read as one channel of data"),
            NodeKind::ToRgba(_) => s("the matte to show as a picture"),
            NodeKind::LumaKey(_) => s("the picture keyed by its brightness"),
            NodeKind::Keyer(_) => s("the picture to key by colour"),
            _ => Some(format!("the picture this {} works on", self.name())),
        }
    }

    /// Full name of an input, for tooltips and property panels.
    pub fn input_name(&self, input: usize) -> &'static str {
        // Mask ports whose matte is the node's own operand, not a limit.
        match self {
            NodeKind::Noise(_) => return "matte (read by the field)",
            NodeKind::Spline(_) if input == 1 => return "camera",
            NodeKind::Spline(_) => return "matte (combined with)",
            NodeKind::MatteIn if input == 1 => return "alpha",
            NodeKind::Keyer(_) if input == 1 => return "garbage",
            _ => {}
        }
        if self.mask_input() == Some(input) {
            return "mask";
        }
        if self.mod_input() == Some(input) {
            return "modulate";
        }
        match self {
            NodeKind::Blend { .. } => {
                if input == 0 { "foreground (over)" } else { "background (under)" }
            }
            NodeKind::Mix(_) => {
                if input == 0 { "from (amount 0)" } else { "to (amount 1)" }
            }
            NodeKind::Diffuse(_) if input == crate::nodes::diffuse::DEPTH_INPUT => {
                "depth (ramps the effect near -> far)"
            }
            NodeKind::Defocus(_) => match input {
                0 => "image",
                2 => "depth",
                3 => "bokeh kernel",
                _ => "input",
            },
            NodeKind::Convolve(_) => match input {
                0 => "image",
                crate::nodes::convolve::KERNEL_INPUT => "kernel (a picture, centred on its middle)",
                _ => "input",
            },
            NodeKind::LoopOut(_) => {
                if input == 0 { "body" } else { "seed" }
            }
            NodeKind::Render3D(_) => {
                if input == crate::nodes::render3d::CAMERA_PORT { "camera" } else { "scene" }
            }
            NodeKind::Starfield(_) => "camera",
            NodeKind::Splat(_) => "camera",
            NodeKind::Godrays(_) if input == crate::nodes::godrays::SOURCE_INPUT => "source",
            NodeKind::EdgeExtend(_) if input == crate::nodes::edgeextend::MATTE_INPUT => {
                "edge matte"
            }
            NodeKind::Lightwrap(_) => {
                if input == 0 { "foreground" } else { "background" }
            }
            NodeKind::MatteCombine(_) => {
                if input == 0 { "A" } else { "B" }
            }
            NodeKind::Dilate(_) | NodeKind::ToRgba(_) => "matte",
            NodeKind::Breakout(_) | NodeKind::ChannelSelect(_) => "channels",
            NodeKind::UvWarp(_) => {
                if input == 0 { "source" } else { "uv map (u = red, v = green)" }
            }
            NodeKind::Uncrop(_) => {
                if input == 0 { "cropped" } else { "full format" }
            }
            NodeKind::MorphDissolve(_) => match input {
                0 => "A",
                1 => "B",
                _ => "matte",
            },
            NodeKind::Compare(_) => {
                if input == 0 { "A" } else { "B" }
            }
            NodeKind::Env(_) => {
                if input == crate::nodes::env::IMAGE_INPUT { "latlong" } else { "camera" }
            }
            NodeKind::Refract(_) => match input {
                crate::nodes::refract::IMAGE_INPUT => "background / latlong",
                crate::nodes::refract::MAP_INPUT => "height / normal map",
                _ => "camera",
            },
            NodeKind::Cryptomatte(_) => "channels",
            NodeKind::ScreenKey(_) => match input {
                0 => "image",
                1 => "holdout (keep)",
                _ => "garbage (drop)",
            },
            NodeKind::GenAi(p) => p.input_name(input),
            NodeKind::CorridorKey(_) => CorridorKeyParams::input_name(input),
            NodeKind::Sammie(_) => SammieParams::input_name(input),
            NodeKind::Flow(_) => "image (to analyse)",
            NodeKind::Stabilize(_) => {
                if input == 0 { "image" } else { "matte (kept out of the solve)" }
            }
            NodeKind::CamTrack(_) => {
                if input == 0 { "image" } else { "matte (not tracked)" }
            }
            NodeKind::PlanarTrack(_) => {
                if input == 0 { "image" } else { "ignore matte (not the plane)" }
            }
            NodeKind::Rifetime(_) => "image (to retime)",
            NodeKind::Freeze(_) => "image (to hold)",
            NodeKind::FrameFix(_) => "image (to repair)",
            NodeKind::VectorBlur(_) | NodeKind::VectorWarp(_) => match input {
                0 => "image",
                1 => "motion vectors (a Flow)",
                _ => "matte",
            },
            NodeKind::Layer(p) => {
                if input + 1 == p.layers.len() { "bottom layer" } else { "layer" }
            }
            // A field node's sockets ARE its parameters, and which is which
            // is the whole question when seven of them fan out of one node.
            NodeKind::Field(f) => {
                f.inputs().get(input).map(|i| i.name).unwrap_or("input")
            }
            NodeKind::Geo(_) => "field (wire a Noise)",
            NodeKind::Blender(_) => BLENDER_INPUT_NAMES.get(input).copied().unwrap_or("wobble_in_N"),
            NodeKind::GeoNode(n) => {
                n.inputs().get(input).map(|i| i.name).unwrap_or("input")
            }
            NodeKind::SdfNode(n) => {
                n.inputs().get(input).map(|i| i.name).unwrap_or("input")
            }
            NodeKind::ParticleNode(n) => n.inputs().get(input).copied().unwrap_or("input"),
            // A model's second port is its source picture: named, so it
            // does not read as a bare "image input" beside the prompt.
            NodeKind::GenAiStep(op) => match op.ports().get(input).map(|p| p.1) {
                Some("image") if input > 0 => "source",
                Some(n) => n,
                None => "input",
            },
            _ => {
                if input == 0 { "image" } else { "input" }
            }
        }
    }

    /// Whether the sockets are worth spelling out beside the node.
    ///
    /// Off almost everywhere on purpose. A Blend's `+` and `−` already say
    /// which is which, and a Defocus's `Z` and `B` are glyphs drawn ON the
    /// nodule — a word beside them would be a second label for the same
    /// fact. It earns its place where the sockets are PARAMETERS and there
    /// are more of them than a glyph each could carry: a field fBm fans out
    /// seven, and nothing about their shape says which one is lacunarity.
    pub fn labels_inputs(&self) -> bool {
        matches!(
            self,
            NodeKind::Field(_)
                | NodeKind::GeoNode(_)
                | NodeKind::SdfNode(_)
                | NodeKind::ParticleNode(_)
                | NodeKind::GenAiStep(_)
        )
    }

    /// A routing dot on a wire of `kind`. The ONE place that knows how a
    /// pipe maps onto a dot's flags, so a new pipe is an arm here rather
    /// than another bool at every site that cuts one.
    pub fn dot(kind: PortKind) -> Self {
        NodeKind::Dot {
            mask: kind == PortKind::Mask,
            channels: kind == PortKind::Channels,
            data: kind == PortKind::Data,
            text: kind == PortKind::Text,
            vector: kind == PortKind::Vector,
            scene: kind == PortKind::Scene,
        }
    }

    pub fn input_kind(&self, input: usize) -> PortKind {
        if matches!(self, NodeKind::Field(_)) {
            return PortKind::Field;
        }
        // Inside an SDF graph a distance (or a number) rides the field
        // pipe and a colour the image pipe.
        if let NodeKind::SdfNode(n) = self {
            return n.inputs().get(input).map(|i| sdf_pipe(i.pipe)).unwrap_or(PortKind::Field);
        }
        // A plugin that types its ports: a subwob, whose ports carry what
        // its input nodes are the portals for.
        if let NodeKind::Custom { type_id, .. } = self {
            if let Some(k) =
                crate::plugins::get(type_id).and_then(|d| d.input_kinds.get(input).copied())
            {
                return k;
            }
        }
        // A modulate port takes a matte, like a mask does.
        if self.mod_input() == Some(input) {
            return PortKind::Mask;
        }
        // The inputs that take the fat cable.
        if matches!(
            self,
            NodeKind::ChannelSelect(_) | NodeKind::Cryptomatte(_) | NodeKind::Breakout(_)
        ) {
            return PortKind::Channels;
        }
        // Scene ports: a Scene's every port, a Render3D's both until it is
        // lowered (then the card pictures, on ports 1..).
        match self {
            NodeKind::Scene3D(_) => return PortKind::Scene,
            NodeKind::Render3D(p) if p.lowered.is_none() => return PortKind::Scene,
            // An Env's camera (lowered away before codegen).
            NodeKind::Env(_) if input == crate::nodes::env::CAMERA_INPUT => return PortKind::Scene,
            NodeKind::Refract(_) if input == crate::nodes::refract::CAMERA_INPUT => return PortKind::Scene,
            // A Starfield's camera (lowered away before codegen).
            NodeKind::Starfield(_) => return PortKind::Scene,
            // A Roto's camera, the same (`crate::roto`).
            NodeKind::Spline(_) if input == 1 => return PortKind::Scene,
            // A Splat's camera, the same.
            NodeKind::Splat(_) => return PortKind::Scene,
            _ => {}
        }
        // Inside a pipeline's dive: a step's prompt port is the text pipe,
        // everything else a picture.
        if let NodeKind::GenAiStep(op) = self {
            return match op.ports().get(input) {
                Some((genai::Pipe::Text, _)) => PortKind::Text,
                // A matte: what a Matte In or a Spline drawn in there
                // gives. (A picture wired in gets the usual ToMatte.)
                Some((_, "mask")) => PortKind::Mask,
                _ => PortKind::Image,
            };
        }
        // A Text's words can come off one too.
        if matches!(self, NodeKind::Text(_)) && input == crate::nodes::text::WORDS_INPUT {
            return PortKind::Text;
        }
        // The pipeline's Matte step reads port 1: coverage, not a picture.
        // A CorridorKey's alpha hint is coverage too.
        if matches!(self, NodeKind::CorridorKey(_)) && input == 1 {
            return PortKind::Mask;
        }
        // A GenAI's matte inputs, wherever on its bar they are.
        if let NodeKind::GenAi(p) = self {
            if p.pipeline.in_kind(input) == genai::InKind::Matte {
                return PortKind::Mask;
            }
        }
        // An EdgeExtend's matte says where the edge is.
        if matches!(self, NodeKind::EdgeExtend(_))
            && input == crate::nodes::edgeextend::MATTE_INPUT
        {
            return PortKind::Mask;
        }
        // A Godrays' source matte says what casts the rays.
        if matches!(self, NodeKind::Godrays(_))
            && input == crate::nodes::godrays::SOURCE_INPUT
        {
            return PortKind::Mask;
        }
        // A Stabilize's matte says which pixels may not vote on the camera.
        if matches!(self, NodeKind::Stabilize(_)) && input == StabilizeParams::MATTE_INPUT {
            return PortKind::Mask;
        }
        if matches!(self, NodeKind::CamTrack(_)) && input == CamTrackParams::MATTE_INPUT {
            return PortKind::Mask;
        }
        // A PlanarTrack's ignore matte says what is in front of the plane.
        if matches!(self, NodeKind::PlanarTrack(_)) && input == PlanarTrackParams::MATTE_INPUT {
            return PortKind::Mask;
        }
        // A bake parked in a matte pipe is a matte on both sides; so is a
        // soft crop, and any other node that follows its pipe.
        if input == 0 && self.matte_mode() {
            return PortKind::Mask;
        }
        // A geometry node's sockets say what they carry: geometry or a
        // scalar. The wiring refuses to cross the two.
        if let NodeKind::GeoNode(n) = self {
            return n.inputs().get(input).map(|i| i.kind).unwrap_or(PortKind::Field);
        }
        // Dilate is a matte operator end to end: its ONE input is a matte,
        // not an image with a matte beside it — so it is not `mask_input`,
        // which means "the mask that limits this node's effect".
        if matches!(self, NodeKind::Dilate(_) | NodeKind::MatteCombine(_)) {
            return PortKind::Mask;
        }
        // Every Extract input past the image is an operand matte (not a
        // `mask_input` limiting the node — each one IS an output).
        if matches!(self, NodeKind::Extract(_)) && input > 0 {
            return PortKind::Mask;
        }
        // A ScreenKey's holdout and garbage mattes: coverage that forces the
        // key one way or the other, not a mask limiting the node.
        if matches!(self, NodeKind::ScreenKey(_)) && input > 0 {
            return PortKind::Mask;
        }
        // A dot wears the kind of the wire it was cut into, both ends.
        // The vectors port of the nodes that read motion.
        if matches!(self, NodeKind::VectorBlur(_) | NodeKind::VectorWarp(_)) && input == 1 {
            return PortKind::Vector;
        }
        if let NodeKind::Dot { mask, channels, data, text, vector, scene } = self {
            return if *scene {
                PortKind::Scene
            } else if *vector {
                PortKind::Vector
            } else if *text {
                PortKind::Text
            } else if *channels {
                PortKind::Channels
            } else if *data {
                PortKind::Data
            } else if *mask {
                PortKind::Mask
            } else {
                PortKind::Image
            };
        }
        // A DEPTH port is a measurement, not a picture: it takes the data
        // pipe, so nothing can hand it a clamped matte's worth of range by
        // accident. What used to arrive here as an image now comes through
        // a ToData, which is what the loader's migration inserts.
        if matches!(self, NodeKind::Defocus(_)) && input == 2 {
            return PortKind::Data;
        }
        if matches!(self, NodeKind::Diffuse(_)) && input == crate::nodes::diffuse::DEPTH_INPUT {
            return PortKind::Data;
        }
        // ToRgba's one input IS the matte it converts.
        if matches!(self, NodeKind::ToRgba(_)) {
            return PortKind::Mask;
        }
        if self.mask_input() == Some(input) {
            PortKind::Mask
        } else {
            PortKind::Image
        }
    }

    /// The kind of ONE output. A node with several outputs need not have
    /// them all the same, and a wiring gate that asks `output_kind()` is
    /// judging every socket by the first one — latent while the only
    /// multi-output node is a LumaKey whose bands are all mattes, and wrong
    /// the moment that stops being true.
    pub fn output_kind_at(&self, slot: usize) -> PortKind {
        // A scalar pass (Z, AO, shadow) rides the data pipe; output 0 is
        // the fat cable when the node offers one.
        if matches!(self, NodeKind::Render3D(p) if p.lowered.is_none()) && slot == 1 {
            return PortKind::Data;
        }
        if matches!(self, NodeKind::Splat(p) if p.depth_of.is_none()) && slot == 1 {
            return PortKind::Data;
        }
        if let NodeKind::Blender(p) = self {
            if slot > 0 {
                return match p.passes.get(slot - 1) {
                    Some(pass) if pass.scalar() => PortKind::Data,
                    _ => PortKind::Image,
                };
            }
        }
        self.output_kind()
    }

    pub fn output_kind(&self) -> PortKind {
        if matches!(self, NodeKind::Field(_)) {
            return PortKind::Field;
        }
        if let NodeKind::SdfNode(n) = self {
            return sdf_pipe(n.out_pipe());
        }
        if let NodeKind::GenAiStep(op) = self {
            return match op.output() {
                Some(genai::Pipe::Text) => PortKind::Text,
                _ if matches!(op, genai::Op::Input { matte: true, .. }) => PortKind::Mask,
                _ => PortKind::Image,
            };
        }
        // A Read switched to all-channels output carries the fat cable.
        if matches!(self, NodeKind::Read { multi: true, .. }) {
            return PortKind::Channels;
        }
        // A lowered scalar pass of a Blender node is data; a Blender node
        // offering its passes by name carries the fat cable.
        if let NodeKind::Blender(p) = self {
            if p.pass.is_some_and(|x| x.scalar()) {
                return PortKind::Data;
            }
            if p.fat && p.pass.is_none() && p.host.is_none() {
                return PortKind::Channels;
            }
        }
        if self.matte_mode()
            || matches!(self, NodeKind::Sammie(p) if p.output == SammieOut::Matte)
        {
            return PortKind::Mask;
        }
        if let NodeKind::GeoNode(n) = self {
            return n.output_kind();
        }
        // A plugin that declares its output's pipe: a subwob's input node
        // emits the pipe it is the portal for.
        if let NodeKind::Custom { type_id, .. } = self {
            if let Some(d) = crate::plugins::get(type_id) {
                if let Some(k) = d.out_kind {
                    return k;
                }
                if d.mask_out {
                    return PortKind::Mask;
                }
            }
        }
        match self {
            NodeKind::Spline(_)
            | NodeKind::RotoNode(_)
            | NodeKind::LumaKey(_)
            | NodeKind::Keyer(_)
            | NodeKind::Cryptomatte(_)
            | NodeKind::ToMatte(_)
            | NodeKind::Dilate(_)
            | NodeKind::MatteCombine(_) => PortKind::Mask,
            // The one way on to the data pipe.
            NodeKind::ToData(_) => PortKind::Data,
            // Depth is a measurement: one value, full depth.
            NodeKind::Depth(_) => PortKind::Data,
            NodeKind::Prompt(_) => PortKind::Text,
            NodeKind::Camera3D(_) | NodeKind::Card3D(_) | NodeKind::Scene3D(_) => PortKind::Scene,
            // The lowered copy that keeps its depth.
            NodeKind::Render3D(p) if p.lowered.as_ref().is_some_and(|l| l.depth) => PortKind::Data,
            NodeKind::Flow(_) => PortKind::Vector,
            // Thresholded edges are coverage, like a Paint in matte mode.
            NodeKind::EdgeDetect(p) if p.output == EdgeOutput::Matte => PortKind::Mask,
            // A Paint switched to matte output paints coverage, not pixels.
            NodeKind::Paint(p) if p.output == PaintOutput::Matte => PortKind::Mask,
            NodeKind::Dot { mask, channels, data, text, vector, scene } => {
                if *scene {
                    PortKind::Scene
                } else if *vector {
                    PortKind::Vector
                } else if *text {
                    PortKind::Text
                } else if *channels {
                    PortKind::Channels
                } else if *data {
                    PortKind::Data
                } else if *mask {
                    PortKind::Mask
                } else {
                    PortKind::Image
                }
            }
            _ => PortKind::Image,
        }
    }

    /// The downloadable weights this node needs, as `models::CATALOG` ids.
    /// Declaring them here is the WHOLE of a node's download handling: the
    /// panel gate, the graph badge, the on-open prompt and the licence tag
    /// all read this (see `models` and CLAUDE.md "Adding a node that needs
    /// weights"). Takes `&self` so a node offering a choice of weights
    /// (small vs accurate) answers for the one its params pick.
    #[allow(clippy::match_single_binding)] // the arms land with the first ML node
    pub fn models(&self) -> &'static [&'static str] {
        match self {
            NodeKind::Upscale(p) => p.models(),
            NodeKind::HdrCnn(p) => p.models(),
            NodeKind::Depth(p) => p.models(),
            NodeKind::Stabilize(p) => p.models(),
            NodeKind::CamTrack(p) => p.models(),
            NodeKind::Flow(p) => p.models(),
            NodeKind::MorphDissolve(p) => p.models(),
            NodeKind::Rifetime(p) => p.models(),
            NodeKind::FrameFix(p) => p.models(),
            _ => &[],
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            NodeKind::Switch(_) => "Switch",
            NodeKind::Read { .. } => "Load",
            NodeKind::Dot { .. } => "Dot",
            NodeKind::Transform(_) => "Transform",
            NodeKind::Grade(_) => "Grade",
            NodeKind::Blur(_) => BlurParams::NAME,
            NodeKind::Defocus(_) => "Defocus",
            NodeKind::MatteCombine(_) => "MatteCombine",
            NodeKind::Glow(_) => GlowParams::NAME,
            NodeKind::Godrays(_) => GodraysParams::NAME,
            NodeKind::LensFlare(_) => LensFlareParams::NAME,
            NodeKind::EdgeExtend(_) => EdgeExtendParams::NAME,
            NodeKind::Glint(_) => GlintParams::NAME,
            NodeKind::EdgeDetect(_) => EdgeDetectParams::NAME,
            NodeKind::Median(_) => MedianParams::NAME,
            NodeKind::HighlightRebuild(_) => HighlightRebuildParams::NAME,
            NodeKind::Convolve(_) => ConvolveParams::NAME,
            NodeKind::Diffuse(_) => DiffuseParams::NAME,
            NodeKind::Blend { .. } => "Blend",
            NodeKind::CornerPin(_) => "CornerPin",
            NodeKind::Stabilize(_) => StabilizeParams::NAME,
            NodeKind::CamTrack(_) => CamTrackParams::NAME,
            NodeKind::Spline(_) => "Spline",
            NodeKind::Paint(_) => "Paint",
            NodeKind::Curve(_) => "Curve",
            NodeKind::Mask => "Mask",
            NodeKind::LumaKey(_) => "LumaKey",
            NodeKind::Keyer(_) => "Keyer",
            NodeKind::Cryptomatte(_) => "Cryptomatte",
            NodeKind::GraphBake { .. } => "Bake",
            NodeKind::GenAi(_) => GenAiParams::NAME,
            NodeKind::Prompt(_) => PromptParams::NAME,
            NodeKind::Camera3D(_) => CameraParams::NAME,
            NodeKind::Card3D(_) => CardParams::NAME,
            NodeKind::Scene3D(_) => SceneParams::NAME,
            NodeKind::Render3D(_) => Render3dParams::NAME,
            NodeKind::CorridorKey(_) => CorridorKeyParams::NAME,
            NodeKind::Sammie(_) => SammieParams::NAME,
            NodeKind::Flow(_) => FlowParams::NAME,
            NodeKind::Rifetime(_) => RifetimeParams::NAME,
            NodeKind::Freeze(_) => FreezeParams::NAME,
            NodeKind::FrameFix(_) => FrameFixParams::NAME,
            NodeKind::VectorBlur(_) => VectorBlurParams::NAME,
            NodeKind::VectorWarp(_) => VectorWarpParams::NAME,
            NodeKind::Dilate(_) => DilateParams::NAME,
            NodeKind::ChannelSelect(_) => "ChannelSelect",
            NodeKind::Breakout(_) => "Breakout",
            NodeKind::Recombine(_) => "Recombine",
            NodeKind::Extract(_) => "Extract",
            NodeKind::LoopIn => "LoopIn",
            NodeKind::LoopOut(_) => "LoopOut",
            NodeKind::Grid(_) => "Grid",
            NodeKind::Gradient(_) => GradientParams::NAME,
            NodeKind::Blank(_) => BlankParams::NAME,
            NodeKind::Starfield(_) => StarfieldParams::NAME,
            NodeKind::Planet(_) => PlanetParams::NAME,
            NodeKind::Plasma(_) => PlasmaParams::NAME,
            NodeKind::Splat(_) => SplatParams::NAME,
            NodeKind::Particles(_) => ParticlesParams::NAME,
            NodeKind::ParticleNode(n) => n.name(),
            NodeKind::GenAiStep(op) => op.name(),
            NodeKind::RotoNode(RotoInner::Axis(_)) => "Axis",
            NodeKind::RotoNode(RotoInner::Out { .. }) => "Out",
            NodeKind::GlintKernel(_) => GlintKernelParams::NAME,
            NodeKind::BokehKernel(_) => BokehKernelParams::NAME,
            NodeKind::Grain(_) => GrainParams::NAME,
            NodeKind::Reformat(_) => "Reformat",
            NodeKind::Tracker(_) => "Tracker",
            NodeKind::PlanarTrack(_) => "PlanarTrack",
            NodeKind::Saturation(_) => SatParams::NAME,
            NodeKind::Contrast(_) => ContrastParams::NAME,
            NodeKind::ColorCompress(_) => ColorCompressParams::NAME,
            NodeKind::HueRotate(_) => HueRotateParams::NAME,
            NodeKind::Lut(p) => p.name(),
            NodeKind::Text(_) => TextParams::NAME,
            NodeKind::Burnin(_) => BurninParams::NAME,
            NodeKind::Mirror(_) => MirrorParams::NAME,
            NodeKind::FlipFlop(_) => FlipFlopParams::NAME,
            NodeKind::Turn(_) => TurnParams::NAME,
            NodeKind::Uncrop(_) => UncropParams::NAME,
            NodeKind::Compare(_) => CompareParams::NAME,
            NodeKind::MorphDissolve(_) => MorphDissolveParams::NAME,
            NodeKind::Gamma(_) => GammaParams::NAME,
            NodeKind::ToneMap(_) => ToneMapParams::NAME,
            NodeKind::UvWarp(_) => UvWarpParams::NAME,
            NodeKind::Env(_) => EnvParams::NAME,
            NodeKind::Refract(_) => RefractParams::NAME,
            NodeKind::GridWarp(_) => GridWarpParams::NAME,
            NodeKind::LensDistort(_) => LensDistortParams::NAME,
            NodeKind::Warp(_) => WarpParams::NAME,
            NodeKind::Expression(_) => ExpressionParams::NAME,
            NodeKind::Despill(_) => DespillParams::NAME,
            NodeKind::Lightwrap(_) => LightwrapParams::NAME,
            NodeKind::Upscale(_) => UpscaleParams::NAME,
            NodeKind::HdrCnn(_) => HdrCnnParams::NAME,
            NodeKind::SeedVr(_) => SeedVrParams::NAME,
            NodeKind::Depth(_) => DepthParams::NAME,
            NodeKind::LtxHdr(_) => LtxHdrParams::NAME,
            NodeKind::ScreenKey(_) => ScreenKeyParams::NAME,
            NodeKind::ColorSlice(_) => ColorSliceParams::NAME,
            NodeKind::Slip(_) => "Slip",
            NodeKind::TimeRemap(_) => "Time Curve",
            NodeKind::Speed(_) => "Speed",
            NodeKind::RangeRemap(_) => "Time Remap",
            NodeKind::TimeRange(_) => "Time Range",
            NodeKind::Premult { premult } => {
                if *premult { "Premult" } else { "Unpremult" }
            }
            NodeKind::Shuffle(_) => "Reorder",
            NodeKind::Write(_) => "Render",
            NodeKind::Noise(_) => "Noise",
            NodeKind::Field(f) => f.name(),
            NodeKind::Geo(_) => "2dini",
            NodeKind::Blender(_) => "Blender",
            NodeKind::GeoNode(n) => n.name(),
            NodeKind::Sdf(_) => "SDF",
            NodeKind::SdfNode(n) => n.name(),
            NodeKind::Mult(_) => "Mult",
            NodeKind::Mix(_) => "Mix",
            NodeKind::Layer(_) => "Layer",
            NodeKind::ToMatte(_) => "ToMatte",
            NodeKind::ToData(_) => ToDataParams::NAME,
            NodeKind::ToRgba(_) => ToRgbaParams::NAME,
            NodeKind::SoftCrop(_) => SoftCropParams::NAME,
            NodeKind::MotionBlur(_) => MotionBlurParams::NAME,
            NodeKind::MatteIn => "MatteIn",
            NodeKind::Shader(_) => "Shader",
            NodeKind::Custom { .. } => "Custom",
        }
    }

    /// Discriminant used in the topology hash (parameter values excluded;
    /// Read.source included because it selects a different texture binding).
    /// A pointwise op on a matte emits a different epilogue, so the pipe
    /// joins the tag — and only then, so nothing already cached moves.
    fn topo_tag(&self) -> u64 {
        let t = self.kind_tag();
        if self.recolours_matte() { t ^ 0x3A77_E000_0000_0000 } else { t }
    }

    fn kind_tag(&self) -> u64 {
        match self {
            // Both params are structural: `inputs` changes the node's shape
            // and `which` changes which subtree is compiled. See
            // `SwitchParams` for why that is the point rather than a slip.
            NodeKind::Switch(p) => 0xc00 + (p.live() as u64) * 32 + p.n() as u64,
            NodeKind::Read { source, input, single, .. } => {
                (idt_tag(input) << 32)
                    | ((single.is_on() as u64) << 31)
                    | (0x100 + *source as u64)
            }
            NodeKind::Transform(_) => 1,
            // The zone COUNT sizes the span (each zone is a fixed run of
            // slots — see `codegen::ZONE_SLOTS`), so it recompiles; every
            // number inside a zone, its band's edges included, is a uniform.
            // No zones is exactly the 2 it always was, so a setup that never
            // touched zones keeps its compiled pipeline and its caches.
            NodeKind::Grade(g) => 2 | ((g.zone_count() as u64) << 32),

            // Elided before the engine ever sees a graph; the tag exists so
            // the match is total, and is deliberately shared with nothing.
            NodeKind::Dot { .. } => 0x2A,
            NodeKind::TimeRange(_) => 0x2A,
            // Different op = different blend formula = different shader.
            NodeKind::Blend { op, .. } => 0x300 + *op as u64,
            NodeKind::Blur(b) => b.topo_tag(),
            // Mode picks the pass chain (prefix+strip vs spiral gather), so
            // it must recompile; every other defocus control is a uniform.
            NodeKind::Defocus(d) => 6 + ((d.mode as u64) << 8),
            // The op picks the expression the fused shader emits, so it
            // joins the hash exactly as a Blend's and a Spline's do.
            NodeKind::MatteCombine(m) => 0xb00 + m.op as u64,
            NodeKind::Glow(gl) => gl.topo_tag(),
            NodeKind::Godrays(p) => p.topo_tag(),
            NodeKind::LensFlare(p) => p.topo_tag(),
            NodeKind::EdgeExtend(p) => p.topo_tag(),
            NodeKind::Glint(gl) => gl.topo_tag(),
            NodeKind::EdgeDetect(p) => p.topo_tag(),
            NodeKind::Median(p) => p.topo_tag(),
            NodeKind::HighlightRebuild(p) => p.topo_tag(),
            NodeKind::Convolve(p) => p.topo_tag(),
            NodeKind::Diffuse(d) => d.topo_tag(),
            NodeKind::CornerPin(_) => 7,
            NodeKind::Stabilize(p) => p.topo_tag(),
            NodeKind::CamTrack(p) => p.topo_tag(),
            // Strokes are NOT here: painting must never recompile. Whether
            // ANY of them clones is, though — a cloning Paint evaluates its
            // input a second time, which is a different program. Bit 40 is
            // clear of every other tag with 8 in its low bits.
            // Strokes are NOT here: painting must never recompile. Which
            // CLASSES of stroke exist are, one bit each, because each is a
            // different set of reads of the input (see `BrushMode`); and
            // the output kind, because a matte is a different program from
            // a composite. Bits 40+ are clear of every other tag with 8 in
            // its low bits.
            NodeKind::Paint(p) => {
                8 | ((p.has_source() as u64) << 40)
                    | ((p.has_blur() as u64) << 41)
                    | ((p.has_tone() as u64) << 42)
                    | (((p.output == PaintOutput::Matte) as u64) << 43)
                    | ((p.has_blend() as u64) << 44)
            }
            NodeKind::Mask => 9,
            // Only the point COUNT moves the slot span; lo/hi/invert/interp
            // are uniforms, so shaping the key never recompiles.
            // Band COUNT lives high so it cannot run into 0xC00; band 0's
            // point count moves the slot span exactly as it always did.
            NodeKind::LumaKey(k) => {
                0xB00 + k.band(0).points.len() as u64 + ((k.bands.len() as u64) << 32)
            }
            // Slot span follows the blob count, like Spline's points: adding
            // a sample recompiles once, moving one never does.
            NodeKind::Keyer(k) => 0xA00 + k.blobs.len() as u64,
            // Fixed slot span (MAX_PICKS); picks, view and invert are all
            // uniforms, so picking never recompiles.
            // ...but WHICH readers it samples is structural: the live id
            // set while viewed, the one matte reader otherwise. Two
            // programs, both cached; a view hop is a cache hit.
            NodeKind::Cryptomatte(c) => 0xE00 + c.previewing as u64,
            // Slot span depends on the point counts, like Spline. `split`
            // and `interp` are uniforms — ganging or smoothing is free.
            NodeKind::Curve(c) => {
                let mut t = 0x400 + c.points.len() as u64;
                for (i, ch) in c.rgb.iter().enumerate() {
                    t += (ch.len() as u64) << (12 + i * 6);
                }
                t
            }
            // Slot count depends on the point count: recompile on add/remove.
            // The point count sizes the layout; the combine op picks which
            // expression the fused shader emits, exactly like a Blend's.
            // Everything else about a Spline — mode, feather, opacity,
            // expand, falloff, invert, the shape transform — is a uniform.
            NodeKind::Spline(s) => {
                0x200 + s.points.len() as u64 + (s.combine as u64) * 0x1_0000
            }
            // live vs baked IS a different shader (passthrough vs reader).
            NodeKind::GraphBake { source, baked, window, mask, single, .. } => {
                0x800
                    + (*source as u64) * 2
                    + *baked as u64
                    + ((window.is_some() as u64) << 40)
                    + (((*baked && (*mask || *single)) as u64) << 41)
            }
            // Same shape: which slot, and whether a take is loaded. Nothing
            // in the pipeline is code.
            NodeKind::GenAi(p) => 0x6A1_0000 + (p.source as u64) * 2 + p.topo_tag(),
            NodeKind::CorridorKey(p) => 0x6C2_0000 + (p.source as u64) * 2 + p.topo_tag(),
            NodeKind::Sammie(p) => 0x5A3_0000 + (p.source as u64) * 4 + p.topo_tag(),
            NodeKind::Flow(p) => 0xF10_0000 + (p.source as u64) * 2 + p.topo_tag(),
            NodeKind::Rifetime(p) => 0x7E7_0000 + (p.source as u64) * 2 + p.topo_tag(),
            NodeKind::Freeze(p) => 0xF3E_0000 + p.topo_tag(),
            NodeKind::FrameFix(p) => 0xF1F_0000 + (p.source as u64) * 2 + p.topo_tag(),
            NodeKind::VectorBlur(p) => p.topo_tag(),
            NodeKind::VectorWarp(p) => p.topo_tag(),
            // The text IS the behaviour: editing it must recompile, and
            // must invalidate every render cache keyed on topology.
            NodeKind::Shader(s) => {
                0xC0DE_0000_0000_0000 ^ s.def().hash
            }
            NodeKind::Dilate(d) => d.topo_tag(),
            // Lowered to a Read before the engine ever hashes a graph; the
            // tag only matters for the black-passthrough safety net.
            NodeKind::ChannelSelect(_) => 43,
            // Never compiled (the editor lowers it to the kids); the tag
            // only exists so a leaked one hashes stably.
            NodeKind::Breakout(_) => 44,
            // Never compiled either (lowered to Masks by `extract::expand`).
            NodeKind::Extract(p) => 0x4500 + p.pairs() as u64,
            // n IS topology (how many inputs the sum folds), and so are
            // the signs (which of them subtract).
            NodeKind::Recombine(p) => {
                let signs = p
                    .sub
                    .iter()
                    .enumerate()
                    .fold(0u64, |a, (i, s)| a | ((*s as u64) << (i & 31)));
                // The alpha source picks a different expression, so it is
                // structural like the signs.
                0xd00 + p.n as u64 + (signs << 16) + ((p.alpha as u64) << 48)
            }
            NodeKind::LoopIn => 41,
            // WHICH LoopIn the stage taps is wiring (it names the texture
            // binding); the iteration COUNT is an encode-time pass count —
            // dragging n must never recompile.
            NodeKind::LoopOut(p) => 0x2900 ^ p.pair.rotate_left(8),
            NodeKind::Grid(_) => 13,
            NodeKind::Gradient(p) => p.topo_tag(),
            NodeKind::Blank(p) => p.topo_tag(),
            NodeKind::Starfield(p) => p.topo_tag(),
            NodeKind::Planet(p) => p.topo_tag(),
            NodeKind::Plasma(p) => p.topo_tag(),
            NodeKind::Splat(p) => p.topo_tag(),
            NodeKind::Particles(p) => p.topo_tag(),
            // A particle node never reaches a comp graph's topology.
            NodeKind::ParticleNode(n) => 0x9A00 ^ n.topo_tag(),
            // Never compiled (dive-only); a stable tag all the same.
            NodeKind::GenAiStep(_) => 0x6A2_0000,
            NodeKind::RotoNode(_) => 0x7A2_0000,
            NodeKind::GlintKernel(p) => p.topo_tag(),
            NodeKind::BokehKernel(p) => p.topo_tag(),
            NodeKind::Prompt(p) => p.topo_tag(),
            NodeKind::Camera3D(p) => p.topo_tag(),
            NodeKind::Card3D(p) => p.topo_tag(),
            NodeKind::Scene3D(p) => p.topo_tag(),
            NodeKind::Render3D(p) => p.topo_tag(),
            NodeKind::Grain(p) => p.topo_tag(),
            NodeKind::Reformat(_) => 14,
            NodeKind::Saturation(sp) => sp.topo_tag(),
            NodeKind::Contrast(p) => p.topo_tag(),
            NodeKind::ColorCompress(p) => p.topo_tag(),
            NodeKind::HueRotate(p) => p.topo_tag(),
            NodeKind::Lut(p) => p.topo_tag(),
            NodeKind::Text(p) => p.topo_tag(),
            NodeKind::Burnin(p) => p.topo_tag(),
            NodeKind::Mirror(p) => p.topo_tag(),
            NodeKind::FlipFlop(p) => p.topo_tag(),
            NodeKind::Turn(p) => p.topo_tag(),
            NodeKind::Uncrop(p) => p.topo_tag(),
            NodeKind::Compare(p) => p.topo_tag(),
            NodeKind::MorphDissolve(p) => 0xF20_0000 + (p.source as u64) * 2 + p.topo_tag(),
            NodeKind::Gamma(p) => p.topo_tag(),
            NodeKind::ToneMap(p) => p.topo_tag(),
            NodeKind::UvWarp(p) => p.topo_tag(),
            NodeKind::Env(p) => p.topo_tag(),
            NodeKind::Refract(p) => p.topo_tag(),
            NodeKind::GridWarp(p) => p.topo_tag(),
            NodeKind::LensDistort(p) => p.topo_tag(),
            NodeKind::Warp(p) => p.topo_tag(),
            NodeKind::Expression(p) => p.topo_tag(),
            NodeKind::Despill(p) => p.topo_tag(),
            NodeKind::Lightwrap(p) => p.topo_tag(),
            NodeKind::Upscale(p) => 0x5C4_0000 + (p.source as u64) * 2 + p.topo_tag(),
            NodeKind::HdrCnn(p) => 0x4DC_0000 + (p.source as u64) * 2 + p.topo_tag(),
            NodeKind::SeedVr(p) => 0x5EE_0000 + (p.source as u64) * 2 + p.topo_tag(),
            NodeKind::Depth(p) => 0xDE9_0000 + (p.source as u64) * 2 + p.topo_tag(),
            NodeKind::LtxHdr(p) => 0x17C_0000 + (p.source as u64) * 4 + p.topo_tag(),
            NodeKind::ScreenKey(p) => p.topo_tag(),
            NodeKind::ColorSlice(p) => p.topo_tag(),
            // Time nodes are shader passthroughs; the remap happens in the
            // player. Nothing about the WGSL changes with their params.
            NodeKind::Slip(_) | NodeKind::TimeRemap(_) | NodeKind::Speed(_) | NodeKind::RangeRemap(_) => 17,
            // Tracking is a bake; the node itself is always a passthrough.
            NodeKind::Tracker(_) => 18,
            // The planar track too: its results are data, never uniforms.
            NodeKind::PlanarTrack(_) => 60,
            // Divide vs multiply IS a different formula.
            NodeKind::Premult { premult } => 19 + *premult as u64,
            // The wiring is a uniform matrix, so re-pointing a channel is a
            // param edit — never a recompile.
            NodeKind::Shuffle(_) => 21,
            // A passthrough in the shader: rendering is a button, and the
            // path is a param like any other.
            NodeKind::Write(_) => 22,
            // The SHAPE of the noise sub-graph is topology (it decides which
            // functions get emitted and who calls whom); every number in it
            // is a uniform, so turning a knob deep in a field graph is still
            // a param edit. Two dimensions emit a second call: topology too.
            NodeKind::Noise(n) => {
                0xC00 ^ crate::field::topo_tag(&n.graph) ^ ((n.two_d as u64) << 40) ^ ((n.single as u64) << 41) ^ ((n.eight_bit as u64) << 42)
            }
            // A field node never reaches a comp graph's topology.
            NodeKind::Field(_) => 32,
            // Geometry is a LAYER, never uniforms and never topology: the
            // tag is a constant whatever the graph inside holds.
            NodeKind::Geo(_) => 37,
            // A layer like Geo's; only the space its pixels arrive in
            // reaches the shader.
            NodeKind::Blender(p) => 39 ^ p.topo_tag(),
            NodeKind::GeoNode(_) => 38,
            // The graph's wiring, kinds and spline point counts; every
            // number inside is a uniform.
            NodeKind::Sdf(p) => p.topo_tag(),
            // An SDF node never reaches a comp graph's topology.
            NodeKind::SdfNode(_) => 0x5DF1,
            // Value and the alpha flag are both uniforms.
            NodeKind::Mult(_) => 24,
            // The crossfade amount is a uniform; nothing here recompiles.
            NodeKind::Mix(_) => 29,
            // Port count and each layer's OP shape the composite chain, so
            // both belong here. Offsets, opacity, trims and transforms are
            // uniforms: sliding a layer along the timeline never recompiles.
            NodeKind::Layer(p) => p
                .layers
                .iter()
                .take(LAYER_MAX)
                .fold(0xC00u64 + p.layers.len().min(LAYER_MAX) as u64, |h, l| {
                    h.wrapping_mul(7).wrapping_add(l.op as u64 + 1)
                }),
            // Channel pick and range are uniforms.
            NodeKind::ToMatte(_) => 25,
            NodeKind::ToData(p) => p.topo_tag(),
            NodeKind::ToRgba(p) => p.topo_tag(),
            NodeKind::SoftCrop(p) => p.topo_tag(),
            NodeKind::MotionBlur(p) => p.topo_tag(),
            NodeKind::MatteIn => 26,
            // Behavior lives in the registry: the tag folds in the plugin's
            // SOURCE hash, so editing a plugin recompiles its consumers and
            // invalidates cached renders. Unregistered types hash to a
            // passthrough identity per type_id.
            NodeKind::Custom { type_id, .. } => {
                let base = type_id
                    .bytes()
                    .fold(0x900u64, |h, b| h.wrapping_mul(31).wrapping_add(b as u64));
                match crate::plugins::get(type_id) {
                    Some(def) => base ^ def.source_hash,
                    None => base,
                }
            }
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Graph {
    pub nodes: HashMap<NodeId, NodeKind>,
    /// edges[(dst_node, dst_input)] = src_node.
    pub edges: HashMap<(NodeId, usize), NodeId>,
    /// Which OUTPUT of the source an edge takes, for the few kinds that
    /// have more than one (see [`NodeKind::output_count`]).
    ///
    /// A SPARSE side table rather than a second field on `edges`: absent
    /// means output 0, so every edge ever written stays exactly what it
    /// was, every setup file loads unchanged, and `input_of` keeps its
    /// signature — which matters, because 119 call sites use it and only a
    /// handful care which output they got. Structural.
    pub edge_out: HashMap<(NodeId, usize), u32>,
    /// Bypassed nodes: codegen passes input 0 through (Reads go black).
    /// Structural — participates in the topology hash.
    pub disabled: std::collections::HashSet<NodeId>,
    /// The node whose output is rendered (there is no Viewer node; any node
    /// can be the view target). Structural — participates in the hash.
    pub view: Option<NodeId>,
    /// Which of the view node's outputs is rendered. 0 for everything that
    /// has only one, which is everything but a multi-band LumaKey.
    pub view_out: u32,
    /// PROXY scale the whole comp evaluates at (0 or 1 = full res; 0.5 =
    /// half). Stamped by the app alongside `formats`; like them it is a
    /// param concern and ABSENT from the topology hash — toggling proxy
    /// never recompiles. Spatial params denominated in PIXELS multiply by
    /// it at pack/write time (`proxy_scale`), normalized params ride the
    /// halved formats for free.
    pub proxy: f32,
    /// Per-node FORMAT (the resolution a node's own image lives in), filled
    /// by the app from what is actually loaded. Spatial params (a
    /// transform's pivot, corner-pin points) anchor to their node's format,
    /// as compositors do — NOT to the viewed output — so what a node does cannot
    /// change with the view target. A param concern, deliberately absent
    /// from the topology hash; empty falls back to the pass out_size.
    pub formats: HashMap<NodeId, (u32, u32)>,
    /// Nodes a lowering MINTED (`crate::roto`) that live in another node's
    /// format: the minted id -> the node whose `formats` entry applies.
    /// The app stamps formats by its own node ids, which minted ones are
    /// not. Not structural, like `formats`.
    pub format_alias: HashMap<NodeId, NodeId>,
    /// Reader slot per (source node, time path), filled by the app from
    /// `reader_instances()`. STRUCTURAL — it joins the topology hash,
    /// because which slot a leaf samples is part of the shader. It moves
    /// only when the time-path structure does, never when a time param is
    /// dragged. Empty falls back to each node's own `source`.
    pub time_slots: HashMap<(NodeId, TimeKey), usize>,
    /// Reader instances whose texture is not simply the plate: it holds a
    /// WINDOW of it (`decode::Window`), a smaller copy (`decode::LoadScale`),
    /// or both. `[x, y, kx, ky]` from `Pack::placement`: the leaf samples at
    /// `position * k - xy`. Filled by the app from the rig it built. WHICH
    /// instances have one is structural; the numbers are a uniform.
    pub reader_windows: HashMap<(NodeId, TimeKey), [f32; 4]>,
    /// The space every node's pixels are in between the input transforms and
    /// the Write. STRUCTURAL: it picks the matrix compiled into every input
    /// transform, and it sets the luma weights and the keyer's Oklab frame,
    /// so it belongs in the topology hash and in the render-cache key.
    pub working: crate::color::Colorspace,
    /// Bake-backed nodes whose inputs ARE to be walked this frame, against
    /// the usual rule that a bake replaces its subtree: the app puts a
    /// Blender node here while exporting its inputs, so the readers above
    /// it exist in the playback rig for exactly as long as the export
    /// needs them. Transient app state, never authored, not in the hash.
    pub sample_through: std::collections::HashSet<NodeId>,
    /// The OTHER viewer slots, lowered like `view` (dots resolved, tools
    /// and harps unfolded): what could be on screen after one keypress.
    /// Their readers stay decoded so an A/B flip is instant; everything
    /// else in the graph is instanced but not decoded (see
    /// `Player::set_active`). Presentation state — not in the hash.
    pub hot_roots: Vec<NodeId>,
}

impl Graph {
    /// A node's format, through `format_alias` for minted nodes.
    pub fn format_of(&self, id: NodeId) -> Option<(u32, u32)> {
        self.formats
            .get(&id)
            .or_else(|| self.format_alias.get(&id).and_then(|h| self.formats.get(h)))
            .copied()
    }

    /// The active view target, if it still exists in the graph.
    pub fn view_node(&self) -> Option<NodeId> {
        self.view.filter(|id| self.nodes.contains_key(id))
    }

    pub fn input_of(&self, node: NodeId, input: usize) -> Option<NodeId> {
        self.edges.get(&(node, input)).copied()
    }

    /// The source of `input` AND which of its outputs the edge takes.
    /// Use this wherever the answer matters; `input_of` is the same thing
    /// with the output dropped, which is correct for every single-output
    /// kind and so for almost every caller.
    pub fn input_port(&self, node: NodeId, input: usize) -> Option<(NodeId, u32)> {
        let src = self.input_of(node, input)?;
        Some((src, self.edge_out.get(&(node, input)).copied().unwrap_or(0)))
    }

    /// Wire `src`'s output `out` into `dst`'s `input`. Output 0 writes no
    /// side-table entry, so a plain connection leaves `edge_out` empty.
    pub fn connect_out(&mut self, dst: NodeId, input: usize, src: NodeId, out: u32) {
        self.edges.insert((dst, input), src);
        if out == 0 {
            self.edge_out.remove(&(dst, input));
        } else {
            self.edge_out.insert((dst, input), out);
        }
    }

    /// How many outputs `node` actually has right now, 1 if it is gone.
    pub fn outputs_of(&self, node: NodeId) -> usize {
        self.nodes.get(&node).map(|k| k.output_count()).unwrap_or(1)
    }

    /// Hash of topology only: node ids, kinds (not params), and wiring.
    /// Two graphs with the same hash compile to the same WGSL.
    /// The proxy factor with the unset default mapped to full res —
    /// `Graph::default()` zeroes the field, and a comp must never multiply
    /// its pixels by zero.
    pub fn proxy_scale(&self) -> f32 {
        if self.proxy > 0.0 { self.proxy } else { 1.0 }
    }

    pub fn topology_hash(&self) -> u64 {
        let mut h = DefaultHasher::new();
        let mut nodes: Vec<_> = self
            .nodes
            .iter()
            .map(|(id, k)| {
                // A Noise that only feeds CPU-side consumers (a 2dini's
                // field port) never reaches the shader, so the SHAPE of its
                // field graph is not this graph's topology: editing it must
                // not recompile a byte-identical pipeline.
                let tag = match k {
                    NodeKind::Noise(_) if self.feeds_only_unsampled(*id) => 0xC00,
                    _ => k.topo_tag(),
                };
                (*id, tag)
            })
            .collect();
        nodes.sort_unstable();
        nodes.hash(&mut h);
        let mut edges: Vec<_> = self.edges.iter().map(|((d, i), s)| (*d, *i, *s)).collect();
        edges.sort_unstable();
        edges.hash(&mut h);
        // Which OUTPUT an edge takes is as structural as which node it came
        // from: a different band is different generated code.
        let mut eout: Vec<_> = self.edge_out.iter().map(|((d, i), o)| (*d, *i, *o)).collect();
        eout.sort_unstable();
        eout.hash(&mut h);
        self.view_out.hash(&mut h);
        let mut disabled: Vec<_> = self.disabled.iter().copied().collect();
        disabled.sort_unstable();
        disabled.hash(&mut h);
        // Which slot each time path samples IS part of the shader. Purely
        // structural: it changes when a time node is added, removed or
        // rewired, never when one is dragged.
        let mut slots: Vec<_> =
            self.time_slots.iter().map(|((n, k), s)| (*n, *k, *s)).collect();
        slots.sort_unstable();
        slots.hash(&mut h);
        // Which leaves sample a windowed texture (never where the window
        // sits: that is a uniform).
        let mut wins: Vec<_> = self.reader_windows.keys().copied().collect();
        wins.sort_unstable();
        wins.hash(&mut h);
        self.view_node().hash(&mut h);
        // Changing the working space rewrites every input transform.
        self.working.hash(&mut h);
        h.finish()
    }

    /// Is every consumer of `id` a node that never samples its inputs (and
    /// `id` not the view)? Then nothing about `id` is in the generated
    /// shader. False for a node nothing consumes: unwired, it is still what
    /// the view might land on next.
    fn feeds_only_unsampled(&self, id: NodeId) -> bool {
        if self.view_node() == Some(id) {
            return false;
        }
        let mut any = false;
        for ((dst, _), src) in &self.edges {
            if *src != id {
                continue;
            }
            any = true;
            match self.nodes.get(dst) {
                Some(k) if self.inputs_unsampled(*dst, k) => {}
                _ => return false,
            }
        }
        any
    }

    /// Every (source node, time path) pair this graph needs in order to
    /// render, with the frame mapping each path implies.
    ///
    /// A Read reached both directly and through a TimeOffset appears TWICE:
    /// the two paths want different frames of it, so they get a reader
    /// instance each and both are live at the same time. Deterministic
    /// order, so slot assignment is stable frame to frame.
    ///
    /// Every input of a Switch is walked here, live or not: this is the
    /// set the playback rig is built from, and a rig that followed the
    /// switch was torn down and rebuilt — its decoded frames thrown away —
    /// every time an animated switch flipped. Which branch is DECODED is
    /// [`Self::reader_instances_from`]'s answer.
    pub fn reader_instances(&self) -> Vec<ReaderInstance> {
        self.instances(&self.time_roots(), true)
    }

    /// The reader instances reachable from `roots` alone, through the LIVE
    /// input of each Switch — the same walk and the same time keys, so
    /// every instance here is also one of [`Self::reader_instances`] and
    /// matches it by `(node, key)`. This is how the app asks which readers
    /// what is ON SCREEN actually samples.
    pub fn reader_instances_from(&self, roots: &[NodeId]) -> Vec<ReaderInstance> {
        self.instances(roots, false)
    }

    fn instances(&self, roots: &[NodeId], every_branch: bool) -> Vec<ReaderInstance> {
        let mut out: Vec<ReaderInstance> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut walk = TimeWalk { every_branch, out: &mut out, seen: &mut seen };
        for &root in roots {
            self.walk_time(root, TimeMap::default(), TIME_KEY_ROOT, &mut walk, &mut Vec::new());
        }
        out.sort_by_key(|i| (i.node, i.key));
        out
    }

    /// For every Load instance: the part of its frame anything downstream
    /// can sample, as `[left, top, right, bottom]` fractions of its format —
    /// `None` when that is all of it. This is what lets a reader keep only a
    /// section of a big plate (`decode::Window`).
    ///
    /// The answer is a box only where EVERY path from the instance to a sink
    /// runs through a Crop (crop or reformat mode) with nothing between the
    /// two but nodes that read their input at the pixel they write — a
    /// Transform, a Blur or anything unlisted in between asks for the whole
    /// frame. Keys are derived exactly as [`Self::reader_instances`] derives
    /// them, so `(node, key)` names the same instance in both.
    pub fn reader_rois(&self) -> HashMap<(NodeId, TimeKey), Option<[f32; 4]>> {
        let mut out = HashMap::new();
        let mut seen = std::collections::HashSet::new();
        for root in self.time_roots() {
            self.walk_roi(root, TIME_KEY_ROOT, None, &mut out, &mut seen, &mut Vec::new());
        }
        out
    }

    fn walk_roi(
        &self,
        node: NodeId,
        key: TimeKey,
        roi: Option<[f32; 4]>,
        out: &mut HashMap<(NodeId, TimeKey), Option<[f32; 4]>>,
        seen: &mut std::collections::HashSet<(NodeId, TimeKey, Option<[u32; 4]>)>,
        stack: &mut Vec<NodeId>,
    ) {
        if stack.contains(&node) {
            return;
        }
        let Some(kind) = self.nodes.get(&node) else { return };
        if !seen.insert((node, key, roi.map(|r| r.map(f32::to_bits)))) {
            return;
        }
        if matches!(kind, NodeKind::Read { .. }) {
            // The union over every path; one path wanting it all wins.
            out.entry((node, key))
                .and_modify(|have| {
                    *have = match (*have, roi) {
                        (Some(a), Some(b)) => {
                            Some([a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])])
                        }
                        _ => None,
                    }
                })
                .or_insert(roi);
        }
        if self.inputs_unsampled(node, kind) {
            return;
        }
        let off = self.disabled.contains(&node);
        let keys: Vec<TimeKey> = if off {
            vec![key]
        } else {
            match kind.time_map() {
                m if m.is_identity() && !kind.is_time_node() => vec![key],
                _ if kind.time_interp() == TimeInterp::Blend => {
                    vec![time_key_push_phase(key, node, 0), time_key_push_phase(key, node, 1)]
                }
                _ => vec![time_key_push(key, node)],
            }
        };
        // What input 0 is asked for, given what this node is asked for.
        let through: Option<[f32; 4]> = if off {
            roi
        } else {
            match kind {
                NodeKind::SoftCrop(p) => {
                    let [l, t, r, b] = p.rect;
                    let bx = [l.min(r), t.min(b), l.max(r), t.max(b)];
                    match p.mode() {
                        CropMode::BoundingBox => roi,
                        // Outside the box is never read.
                        CropMode::Crop => Some(match roi {
                            Some(q) => [q[0].max(bx[0]), q[1].max(bx[1]), q[2].min(bx[2]), q[3].min(bx[3])],
                            None => bx,
                        }),
                        // The node's whole format IS the box.
                        CropMode::Reformat => Some(match roi {
                            Some(q) => {
                                let (w, h) = (bx[2] - bx[0], bx[3] - bx[1]);
                                [bx[0] + q[0] * w, bx[1] + q[1] * h, bx[0] + q[2] * w, bx[1] + q[3] * h]
                            }
                            None => bx,
                        }),
                    }
                }
                // Read the input at the pixel they write, in its format.
                NodeKind::Grade(_)
                | NodeKind::Curve(_)
                | NodeKind::Mult(_)
                | NodeKind::Gamma(_)
                | NodeKind::Saturation(_)
                | NodeKind::Contrast(_)
                | NodeKind::ToneMap(_)
                | NodeKind::HueRotate(_)
                | NodeKind::ColorCompress(_)
                | NodeKind::Lut(_)
                | NodeKind::Shuffle(_)
                | NodeKind::Dot { .. }
                | NodeKind::TimeRange(_)
                | NodeKind::Slip(_)
                | NodeKind::TimeRemap(_)
                | NodeKind::Speed(_)
                | NodeKind::RangeRemap(_)
                | NodeKind::Freeze(_)
                | NodeKind::GraphBake { .. } => roi,
                _ => None,
            }
        };
        stack.push(node);
        if kind.per_input_time() && !off {
            let okey = keys[0];
            for i in 0..kind.input_count() {
                if let Some(src) = self.input_of(node, i) {
                    self.walk_roi(src, input_time_key(okey, node, i), None, out, seen, stack);
                }
            }
        } else {
            for &k in &keys {
                for i in 0..kind.input_count() {
                    if let Some(src) = self.input_of(node, i) {
                        let r = if i == 0 { through } else { None };
                        self.walk_roi(src, k, r, out, seen, stack);
                    }
                }
            }
        }
        stack.pop();
    }

    /// The time map reaching EVERY node, by the same walk `reader_instances`
    /// uses — what each node's animated parameters are evaluated at.
    ///
    /// A node reachable through two different time paths keeps the first one
    /// found (view target first): the graph holds one kind per node id, so
    /// one node's params cannot sit on two timebases at once. Readers do not
    /// share that limit — they instance per time key.
    pub fn node_time_maps(&self) -> std::collections::BTreeMap<NodeId, TimeMap> {
        let mut out = std::collections::BTreeMap::new();
        for root in self.time_roots() {
            self.walk_node_times(root, TimeMap::default(), &mut out, &mut Vec::new());
        }
        out
    }

    fn walk_node_times(
        &self,
        node: NodeId,
        map: TimeMap,
        out: &mut std::collections::BTreeMap<NodeId, TimeMap>,
        stack: &mut Vec<NodeId>,
    ) {
        if stack.contains(&node) {
            return; // cycle
        }
        let Some(kind) = self.nodes.get(&node) else { return };
        if out.contains_key(&node) {
            return; // first path wins, and it already carried the subtree
        }
        out.insert(node, map.clone());
        let inner = if self.disabled.contains(&node) {
            map
        } else {
            match kind.time_map() {
                m if m.is_identity() && !kind.is_time_node() => map,
                // Animation is sampled at the unrounded time either way, so
                // the phase is irrelevant here — only readers snap to frames.
                m => m.then(map),
            }
        };
        if self.inputs_unsampled(node, kind) {
            return;
        }
        stack.push(node);
        // A layer stack hands each input its own timebase, so the walk
        // branches per input rather than once for the node.
        let per_input = kind.per_input_time() && !self.disabled.contains(&node);
        for i in 0..kind.input_count() {
            if !kind.samples_input(i) {
                continue;
            }
            if let Some(src) = self.input_of(node, i) {
                // `inner` is this node's own timebase (a Layer has no
                // node-level map of its own); each layer composes its offset
                // on top of it.
                let m = if per_input {
                    kind.input_time_map(i).then(inner.clone())
                } else {
                    inner.clone()
                };
                self.walk_node_times(src, m, out, stack);
            }
        }
        stack.pop();
    }

    /// Where a time walk starts: the view target first (what is on screen
    /// matters most), then anything else nothing consumes.
    fn time_roots(&self) -> Vec<NodeId> {
        let consumed: std::collections::HashSet<NodeId> =
            self.edges.values().copied().collect();
        let mut roots: Vec<NodeId> = self
            .nodes
            .keys()
            .copied()
            .filter(|id| !consumed.contains(id))
            .collect();
        roots.sort_unstable();
        if let Some(v) = self.view_node() {
            roots.retain(|id| *id != v);
            roots.insert(0, v);
        }
        roots
    }

    fn walk_time(
        &self,
        node: NodeId,
        map: TimeMap,
        key: TimeKey,
        walk: &mut TimeWalk,
        stack: &mut Vec<NodeId>,
    ) {
        if stack.contains(&node) {
            return; // cycle
        }
        let Some(kind) = self.nodes.get(&node) else { return };
        // The same node on the same time path is one instance however many
        // ways the graph arrives at it (a diamond is not two readers).
        if !walk.seen.insert((node, key)) {
            return;
        }
        // An unminted slot (`usize::MAX`, a ChannelSelect or Cryptomatte
        // the app has not assigned yet) is not a reader: instancing it
        // would hand the rig a slot that does not exist.
        // Timeline PLACEMENT: a Read whose files start at 1001 starts at
        // comp 1001 — a leaf-level shift composed like a Slip sitting just
        // above the reader, so every consumer (player, residency, caches)
        // follows for free and ticks before the start hold frame one.
        let placed = match kind.placement() {
            Some(s) if s != 0 => Some(TimeMap::affine(1.0, -(s as f32))),
            None | Some(_) => None,
        };
        for (sub, s) in kind.reader_slots().iter().enumerate() {
            if *s == usize::MAX || !kind.reader_sub_active(sub) {
                continue;
            }
            let leaf = match &placed {
                Some(p) => p.clone().then(map.clone()),
                None => map.clone(),
            };
            walk.out.push(ReaderInstance { node, key: sub_key(key, sub), sub, map: leaf });
        }
        // A bypassed time node passes time through unchanged, like its image.
        // A blending one splits: the subtree is walked once per side of the
        // sub-frame time, so both frames are decoded and live at once.
        let branches: Vec<(TimeMap, TimeKey)> = if self.disabled.contains(&node) {
            vec![(map.clone(), key)]
        } else {
            match kind.time_map() {
                m if m.is_identity() && !kind.is_time_node() => vec![(map.clone(), key)],
                m if kind.time_interp() == TimeInterp::Blend => {
                    let [a, b] = kind.blend_branch_maps().unwrap_or_else(|| {
                        [m.clone().with_phase(TimePhase::Floor), m.with_phase(TimePhase::Ceil)]
                    });
                    vec![
                        (a.then(map.clone()), time_key_push_phase(key, node, 0)),
                        (b.then(map.clone()), time_key_push_phase(key, node, 1)),
                    ]
                }
                m => vec![(m.then(map.clone()), time_key_push(key, node))],
            }
        };
        // Inputs that are never sampled don't carry time either (same rule
        // as `visit`: a baked GraphBake replaces its subtree, and a
        // Rife/Retime input wire only names the sequence its bake analysed).
        if self.inputs_unsampled(node, kind) {
            return;
        }
        stack.push(node);
        // A layer stack branches PER INPUT: every layer is its own time
        // path, so the same plate wired into two layers at two offsets
        // decodes two frames and both are live at once — which is the whole
        // point of a stack.
        if kind.per_input_time() && !self.disabled.contains(&node) {
            // One branch out of `branches` (a Layer has no node-level map),
            // with each layer's offset composed on top of it.
            let (outer, okey) = branches[0].clone();
            for i in 0..kind.input_count() {
                if !walk.every_branch && !kind.samples_input(i) {
                    continue;
                }
                if let Some(src) = self.input_of(node, i) {
                    let m = kind.input_time_map(i).then(outer.clone());
                    let k = input_time_key(okey, node, i);
                    self.walk_time(src, m, k, walk, stack);
                }
            }
            stack.pop();
            return;
        }
        for (inner_map, inner_key) in &branches {
            for i in 0..kind.input_count() {
                if !walk.every_branch && !kind.samples_input(i) {
                    continue;
                }
                if let Some(src) = self.input_of(node, i) {
                    self.walk_time(src, inner_map.clone(), *inner_key, walk, stack);
                }
            }
        }
        stack.pop();
    }

    /// `inputs_unsampled` for codegen, which walks the graph the same way.
    pub fn inputs_unsampled_pub(&self, node: NodeId, kind: &NodeKind) -> bool {
        self.inputs_unsampled(node, kind)
    }

    /// True when a node's wired inputs are not actually sampled to render it.
    fn inputs_unsampled(&self, node: NodeId, kind: &NodeKind) -> bool {
        !self.disabled.contains(&node)
            && !self.sample_through.contains(&node)
            && matches!(
                kind,
                NodeKind::GraphBake { baked: true, .. }
                    // A loaded key is the frames on disk; its plate and hint
                    // were rendered when it was run.
                    | NodeKind::CorridorKey(CorridorKeyParams { loaded: true, .. })
                    // A loaded matte on the mask pipe reads no plate.
                    | NodeKind::Sammie(SammieParams { loaded: true, output: SammieOut::Matte, .. })
                    // Vectors are frames on disk, or zero motion: its input
                    // is only ever read by the analysis render.
                    | NodeKind::Flow(_)
                    // Retimed frames are on disk; its input was rendered
                    // when it was run.
                    | NodeKind::Rifetime(RifetimeParams { loaded: true, .. })
                    // So are a Frame Fix's, in flow mode.
                    | NodeKind::FrameFix(FrameFixParams { loaded: true, .. })
                    // A loaded result is frames on disk; the input was
                    // rendered when it was run.
                    | NodeKind::Upscale(UpscaleParams { loaded: true, .. })
                    | NodeKind::HdrCnn(HdrCnnParams { loaded: true, .. })
                    | NodeKind::SeedVr(SeedVrParams { loaded: true, .. })
                    | NodeKind::Depth(DepthParams { loaded: true, .. })
                    // Raw, the result alone; composited it still reads the
                    // plate under it.
                    | NodeKind::LtxHdr(LtxHdrParams { loaded: true, raw: true, .. })
                    // The field port is read on the CPU by the layer
                    // evaluator, never by the shader: the Noise wired there
                    // is not part of this node's pixel subtree.
                    | NodeKind::Geo(_)
                    // Its inputs are exported TO Blender by the app; the
                    // fused pass samples only the layer that comes back.
                    | NodeKind::Blender(_)
            )
    }

    /// The reader slot a node instance binds: the instanced slot when the
    /// app has assigned one for this time path, else the node's own.
    pub fn slot_for(&self, node: NodeId, key: TimeKey, fallback: usize) -> usize {
        self.time_slots.get(&(node, key)).copied().unwrap_or(fallback)
    }

    /// [`slot_for`] for a node's `sub`-th reader slot.
    pub fn slot_for_sub(&self, node: NodeId, key: TimeKey, sub: usize, fallback: usize) -> usize {
        self.slot_for(node, sub_key(key, sub), fallback)
    }

    /// Nodes reachable upstream from the viewer, deepest-first (topo order).
    /// Returns None if there is no viewer. Cycles are broken by the visited
    /// set (a cycle upstream simply truncates — codegen treats the repeated
    /// node as unconnected input, and the UI should prevent cycles anyway).
    pub fn topo_from_view(&self) -> Option<Vec<NodeId>> {
        let view = self.view_node()?;
        let mut order = Vec::new();
        let mut visited = std::collections::HashSet::new();
        self.visit(view, &mut visited, &mut order);
        Some(order)
    }

    fn visit(
        &self,
        node: NodeId,
        visited: &mut std::collections::HashSet<NodeId>,
        order: &mut Vec<NodeId>,
    ) {
        if !visited.insert(node) {
            return;
        }
        if let Some(kind) = self.nodes.get(&node) {
            // Inputs that are never sampled don't render: a BAKED GraphBake
            // replaces its subtree entirely, and a Rife/Retime input wire
            // only names the sequence its bake analyses. Descending would
            // needlessly run upstream filter stages every frame. (Bypassed
            // nodes pass input 0 through, so they still descend.)
            if !self.inputs_unsampled(node, kind) {
                for i in 0..kind.input_count() {
                    if !kind.samples_input(i) {
                        continue;
                    }
                    if let Some(src) = self.input_of(node, i) {
                        self.visit(src, visited, order);
                    }
                }
            }
            order.push(node);
        }
    }

    /// The default POC graph:
    /// Read(0) -> Transform -> Grade -\
    ///                                 Blend (view target)
    /// Read(1) ----------------------/
    pub fn default_poc() -> Self {
        let mut g = Graph::default();
        g.nodes.insert(1, NodeKind::read(0, ""));
        g.nodes.insert(2, NodeKind::read(1, ""));
        g.nodes.insert(3, NodeKind::Transform(TransformParams::default()));
        g.nodes.insert(4, NodeKind::Grade(GradeParams::default()));
        g.nodes.insert(5, NodeKind::blend());
        g.edges.insert((3, 0), 1); // transform <- read A
        g.edges.insert((4, 0), 3); // grade <- transform
        g.edges.insert((5, 0), 4); // blend.+ <- grade
        g.edges.insert((5, 1), 2); // blend.− <- read B
        g.view = Some(5);
        g
    }
}

#[cfg(test)]
mod reader_roi_tests {
    use super::*;

    fn crop(rect: [f32; 4], mode: CropMode) -> NodeKind {
        NodeKind::SoftCrop(SoftCropParams { rect, mode, ..Default::default() })
    }

    /// One plate, two reformat crops, each into its own layer: two reader
    /// instances, each asked for its own box — and the keys are the ones
    /// `reader_instances` gives, so the app can match them to slots.
    #[test]
    fn a_plate_cut_into_layers_is_asked_for_each_box() {
        let mut g = Graph::default();
        g.nodes.insert(1, NodeKind::read(0, ""));
        g.nodes.insert(2, crop([0.0, 0.0, 0.5, 1.0], CropMode::Reformat));
        g.nodes.insert(3, crop([0.5, 0.0, 1.0, 1.0], CropMode::Reformat));
        g.nodes.insert(4, NodeKind::Grade(GradeParams::default()));
        g.nodes.insert(
            5,
            NodeKind::Layer(LayerParams { layers: vec![LayerItem::default(); 2], outputs: false }),
        );
        g.edges.insert((2, 0), 1);
        g.edges.insert((4, 0), 1); // a pointwise node between the crop and the plate
        g.edges.insert((3, 0), 4);
        g.edges.insert((5, 0), 2);
        g.edges.insert((5, 1), 3);
        g.view = Some(5);
        let rois = g.reader_rois();
        let inst = g.reader_instances();
        assert_eq!(inst.len(), 2);
        let mut boxes: Vec<[f32; 4]> =
            inst.iter().map(|i| rois[&(i.node, i.key)].expect("a box")).collect();
        boxes.sort_by(|a, b| a[0].total_cmp(&b[0]));
        assert_eq!(boxes, vec![[0.0, 0.0, 0.5, 1.0], [0.5, 0.0, 1.0, 1.0]]);
    }

    /// Anything that samples elsewhere between the crop and the plate, a
    /// second consumer that wants it all, or a bounding box (a statement,
    /// not a cut) asks for the whole frame.
    #[test]
    fn the_whole_frame_whenever_a_box_cannot_be_proven() {
        let whole = |g: &Graph| g.reader_rois().values().all(|r| r.is_none());
        let base = |mode| {
            let mut g = Graph::default();
            g.nodes.insert(1, NodeKind::read(0, ""));
            g.nodes.insert(2, crop([0.25, 0.25, 0.75, 0.75], mode));
            g.edges.insert((2, 0), 1);
            g.view = Some(2);
            g
        };
        let g = base(CropMode::Crop);
        assert_eq!(g.reader_rois().values().next(), Some(&Some([0.25, 0.25, 0.75, 0.75])));
        assert!(whole(&base(CropMode::BoundingBox)));
        // A Transform under the crop reads the plate anywhere.
        let mut t = base(CropMode::Crop);
        t.nodes.insert(3, NodeKind::Transform(TransformParams::default()));
        t.edges.insert((3, 0), 1);
        t.edges.insert((2, 0), 3);
        assert!(whole(&t));
        // A second consumer on the same time path, uncropped.
        let mut two = base(CropMode::Crop);
        two.nodes.insert(3, NodeKind::Grade(GradeParams::default()));
        two.edges.insert((3, 0), 1);
        assert!(whole(&two));
        // A bypassed crop cuts nothing.
        let mut off = base(CropMode::Crop);
        off.disabled.insert(2);
        assert!(whole(&off));
        // A reformat inside a crop composes: the inner box, in plate terms.
        let mut nest = base(CropMode::Reformat);
        nest.nodes.insert(3, crop([0.5, 0.0, 1.0, 1.0], CropMode::Crop));
        nest.edges.insert((3, 0), 2);
        nest.view = Some(3);
        assert_eq!(nest.reader_rois().values().next(), Some(&Some([0.5, 0.25, 0.75, 0.75])));
    }

    /// A setup from before Load scales loads at full size, and a Load left
    /// at full size saves (and so keys its bakes) exactly as it did.
    #[cfg(feature = "serde")]
    #[test]
    fn a_load_scale_is_absent_until_set() {
        let mut json = serde_json::to_value(NodeKind::read(0, "plate.exr")).unwrap();
        assert!(json["Read"].get("scale").is_none(), "{json}");
        let back: NodeKind = serde_json::from_value(json.clone()).unwrap();
        assert!(matches!(back, NodeKind::Read { scale: crate::decode::LoadScale::Full, .. }));
        json["Read"]["scale"] = "Half".into();
        let back: NodeKind = serde_json::from_value(json).unwrap();
        assert!(matches!(back, NodeKind::Read { scale: crate::decode::LoadScale::Half, .. }));
    }

    /// Which instances are windowed is topology; where the window sits is not.
    #[test]
    fn a_window_corner_is_not_topology() {
        let mut g = Graph::default_poc();
        let h0 = g.topology_hash();
        g.reader_windows.insert((1, TIME_KEY_ROOT), [64.0, 0.0, 1.0, 1.0]);
        let h1 = g.topology_hash();
        g.reader_windows.insert((1, TIME_KEY_ROOT), [128.0, 64.0, 0.5, 0.5]);
        assert_ne!(h0, h1);
        assert_eq!(h1, g.topology_hash());
    }
}

#[cfg(all(test, feature = "serde"))]
mod generator_format_tests {
    use super::*;

    /// A generator saved before the format row had "inherit" must keep the
    /// size it declared: the field defaults to false PER FIELD, even where
    /// the struct's own serde default would say otherwise.
    #[test]
    fn old_generators_keep_their_declared_size() {
        let mut g = serde_json::to_value(GridParams { size: (64, 32), ..Default::default() }).unwrap();
        g.as_object_mut().unwrap().remove("inherit");
        let g: GridParams = serde_json::from_value(g).unwrap();
        assert!(!g.inherit && g.size == (64, 32));
        let gr: GradientParams = serde_json::from_str(r#"{"size":[64,32]}"#).unwrap();
        assert!(!gr.inherit && gr.size == (64, 32));
        let n: NoiseParams = serde_json::from_str(r#"{"size":[64,32]}"#).unwrap();
        assert!(!n.inherit);
        // A Noise saved before the 0.1x rate keeps moving as it did; a new
        // one runs at a tenth, and that survives a round trip.
        assert_eq!(n.speed_unit, 1.0, "a legacy Noise keeps its rate");
        assert!(!n.single && !n.eight_bit);
        let fresh: NoiseParams =
            serde_json::from_value(serde_json::to_value(NoiseParams::default()).unwrap()).unwrap();
        assert_eq!(fresh.speed_unit, SPEED_UNIT);
        let p = NoiseParams { time: 10.0, speed: 2.0, ..Default::default() };
        assert!((p.phase() - 2.0).abs() < 1e-6, "a new node runs at a tenth");
        let b: BlankParams = serde_json::from_str(r#"{"size":[64,32]}"#).unwrap();
        assert!(!b.inherit && b.color == [0.0, 0.0, 0.0, 1.0]);
    }
}

#[cfg(all(test, feature = "serde"))]
mod spline_tests {
    use super::*;

    #[test]
    fn old_spline_setups_migrate_to_per_point_feather() {
        // Pre-handles form: bare [x, y] points + one global feather.
        let old = r#"{"points":[[0.1,0.2],[0.8,0.2],[0.5,0.9]],"feather":42.0}"#;
        let sp: SplineParams = serde_json::from_str(old).unwrap();
        assert_eq!(sp.mode, SplineMode::CatmullRom, "old shapes keep their look");
        assert_eq!(sp.points.len(), 3);
        assert!(sp.points.iter().all(|q| q.feather == 42.0));
        assert_eq!(sp.points[1].p, [0.8, 0.2]);
        assert_eq!(sp.points[0].h_in, [0.0, 0.0]);

        // A shape written before the matte controls and the transform
        // existed loads with the values that mean "unchanged".
        assert_eq!(sp.combine, MatteOp::Union, "and combines as itself");
        assert_eq!(sp.opacity, 1.0);
        assert_eq!(sp.expand, 0.0);
        assert_eq!(sp.falloff, 0.0);
        assert!(!sp.invert);
        assert!(sp.xf.is_identity(), "and sits where it was authored");
        assert_eq!(sp.xf.center, [0.5, 0.5]);

        // Current form round-trips, handles, controls and transform and all.
        let mut cur = SplineParams {
            points: vec![SplinePoint { p: [0.2, 0.3], feather: 5.0, bias: 1.5, h_in: [-0.1, 0.0], h_out: [0.1, 0.0], tension: 0.0 }],
            mode: SplineMode::Bezier,
            combine: MatteOp::Subtract,
            opacity: 0.4,
            expand: -3.5,
            falloff: 0.75,
            invert: true,
            xf: ShapeXf {
                translate: [0.1, -0.2],
                rotate: 30.0,
                scale: [1.5, 0.5],
                center: [0.25, 0.75],
            },
            format: Some((640, 360)),
            feather_out: 4.0,
            feather_in: 2.5,
            ..SplineParams::default()
        };
        cur.points.push(SplinePoint::new([0.6, 0.7], 9.0));
        let json = serde_json::to_string(&cur).unwrap();
        let back: SplineParams = serde_json::from_str(&json).unwrap();
        assert_eq!(back, cur);

        // A shape written by THIS build minus the newest keys still loads:
        // the untagged repr only needs `points`.
        let partial = r#"{"points":[{"p":[0.1,0.1],"feather":4.0}],"mode":"Bezier"}"#;
        let p2: SplineParams = serde_json::from_str(partial).unwrap();
        assert_eq!(p2.mode, SplineMode::Bezier);
        assert_eq!(p2.opacity, 1.0, "opacity defaults to fully on, not zero");
        assert_eq!(p2.xf.scale, [1.0, 1.0], "and scale to 1, not zero");
    }

    #[test]
    fn the_shape_transform_is_isotropic_and_invertible() {
        // A 2:1 format: a rotation applied in NORMALIZED space would shear
        // the shape, so the affine has to be built in pixels.
        const SZ: [f32; 2] = [200.0, 100.0];
        let xf = ShapeXf { rotate: 90.0, center: [0.5, 0.5], ..Default::default() };
        // A point one pixel-unit right of the centre lands one pixel-unit
        // BELOW it (y down on screen), not half a frame away.
        let p = xf.point([0.5 + 20.0 / 200.0, 0.5], SZ);
        assert!((p[0] - 0.5).abs() < 1e-5, "x {p:?}");
        assert!((p[1] - (0.5 + 20.0 / 100.0)).abs() < 1e-5, "y {p:?}");

        // Every transform round-trips through its inverse, so a dragged
        // point lands where the cursor is.
        let xf = ShapeXf {
            translate: [0.13, -0.07],
            rotate: -37.0,
            scale: [1.7, 0.6],
            center: [0.3, 0.8],
        };
        for q in [[0.0, 0.0], [0.25, 0.9], [1.0, 1.0], [0.5, 0.5]] {
            let back = xf.point_inv(xf.point(q, SZ), SZ);
            assert!(
                (back[0] - q[0]).abs() < 1e-4 && (back[1] - q[1]).abs() < 1e-4,
                "{q:?} -> {back:?}"
            );
        }
        // A delta comes back as a delta: the translation drops out.
        let d = xf.vec_inv([0.1, 0.0], SZ);
        let two = xf.vec_inv([0.2, 0.0], SZ);
        assert!((two[0] - 2.0 * d[0]).abs() < 1e-5 && (two[1] - 2.0 * d[1]).abs() < 1e-5);

        // The identity is exactly free — and exactly the identity.
        assert!(ShapeXf::default().is_identity());
        assert_eq!(ShapeXf::default().point([0.31, 0.62], SZ), [0.31, 0.62]);
    }

    #[test]
    fn fresh_spline_has_no_shape() {
        let sp = SplineParams::default();
        assert!(sp.points.is_empty(), "a new Spline draws nothing until clicked");
    }
}

#[cfg(test)]
mod brush_mode_tests {
    use super::*;

    fn stroke(mode: BrushMode) -> PaintStroke {
        PaintStroke { mode, ..PaintStroke::new(vec![[0.5, 0.5]], 10.0, 0.5, [1.0; 4]) }
    }

    /// Each read class is its own topology bit, so a node compiles exactly
    /// the extra reads its strokes need — and nothing else about a stroke
    /// is topology at all.
    #[test]
    fn read_classes_are_topology_and_nothing_else_is() {
        let tag = |strokes: Vec<PaintStroke>, output: PaintOutput| {
            NodeKind::Paint(PaintParams { strokes, output, ..Default::default() }).topo_tag()
        };
        let plain = tag(vec![stroke(BrushMode::Color)], PaintOutput::Rgb);
        assert_eq!(plain, tag(vec![], PaintOutput::Rgb), "a colour stroke is not topology");
        // Pointwise modes cost nothing and change nothing.
        assert_eq!(plain, tag(vec![stroke(BrushMode::Dodge)], PaintOutput::Rgb) & !(1 << 42));
        let clone = tag(vec![stroke(BrushMode::Clone)], PaintOutput::Rgb);
        let smear = tag(vec![stroke(BrushMode::Smear)], PaintOutput::Rgb);
        assert_ne!(clone, plain);
        assert_eq!(clone, smear, "clone and smear are the same read");
        let blur = tag(vec![stroke(BrushMode::Blur)], PaintOutput::Rgb);
        let burn = tag(vec![stroke(BrushMode::Burn)], PaintOutput::Rgb);
        let dodge = tag(vec![stroke(BrushMode::Dodge)], PaintOutput::Rgb);
        assert_eq!(burn, dodge, "dodge and burn are one exposure read");
        assert!(blur != plain && blur != clone && blur != burn);
        // Everything at once is the union of the bits.
        let all = tag(
            vec![stroke(BrushMode::Clone), stroke(BrushMode::Blur), stroke(BrushMode::Burn)],
            PaintOutput::Rgb,
        );
        assert_eq!(all, plain | (1 << 40) | (1 << 41) | (1 << 42));
        // The output kind is structural too, and it is what the socket says.
        let matte = tag(vec![stroke(BrushMode::Color)], PaintOutput::Matte);
        assert_eq!(matte, plain | (1 << 43));
        let p = NodeKind::Paint(PaintParams { output: PaintOutput::Matte, ..Default::default() });
        assert_eq!(p.output_kind(), PortKind::Mask);
        assert_eq!(NodeKind::Paint(Default::default()).output_kind(), PortKind::Image);
        // Shape, strength, lifetime, points: none of it reaches the tag.
        let mut moved = stroke(BrushMode::Blur);
        moved.aspect = 0.3;
        moved.angle = 45.0;
        moved.strength = 99.0;
        moved.life = StrokeLife::Single(7);
        moved.points.push([0.1, 0.9]);
        assert_eq!(tag(vec![moved], PaintOutput::Rgb), blur);
    }

    /// A stroke written before `mode` existed says "clone" by carrying an
    /// offset and no mode; it must still read as a clone, or every clone
    /// patch in every saved setup silently turns into a black smudge.
    #[cfg(feature = "serde")]
    #[test]
    fn legacy_clone_strokes_still_clone() {
        let legacy = r#"{"points":[[0.5,0.5]],"radius":10.0,"softness":0.5,
            "color":[0.0,0.0,0.0,1.0],"clone_off":[0.1,0.0]}"#;
        let s: PaintStroke = serde_json::from_str(legacy).unwrap();
        assert_eq!(s.mode, BrushMode::Color, "the field itself is the default");
        assert_eq!(s.mode(), BrushMode::Clone, "but the accessor knows better");
        assert_eq!(s.aspect, 1.0, "and the brush is round");
        let p = PaintParams { strokes: vec![s], ..Default::default() };
        assert!(p.has_source() && !p.has_blur() && !p.has_tone());
        // And a stroke from before clone existed at all is a colour stroke.
        let older = r#"{"points":[[0.5,0.5]],"radius":10.0,"softness":0.5,
            "color":[1.0,0.0,0.0,1.0]}"#;
        let s: PaintStroke = serde_json::from_str(older).unwrap();
        assert_eq!(s.mode(), BrushMode::Color);
        assert_eq!(s.life, StrokeLife::All);
        // The current form round-trips every field.
        let mut full = stroke(BrushMode::Smear);
        full.aspect = 0.4;
        full.angle = -30.0;
        full.strength = 24.0;
        full.life = StrokeLife::Range(2, 8);
        let back: PaintStroke =
            serde_json::from_str(&serde_json::to_string(&full).unwrap()).unwrap();
        assert_eq!(back, full);
    }
}

#[cfg(test)]
mod stroke_life_tests {
    use super::*;

    #[test]
    fn only_single_follows_the_playhead() {
        // "this frame" means the frame you are looking at, so the stored
        // number is a placeholder that gets replaced at paint time.
        assert_eq!(StrokeLife::Single(0).at(42), StrokeLife::Single(42));
        // A range is two numbers the user typed into the panel's in/out
        // fields. Rebasing them onto the playhead would silently ignore what
        // they typed the moment they painted anywhere else.
        assert_eq!(StrokeLife::Range(10, 20).at(50), StrokeLife::Range(10, 20));
        assert_eq!(StrokeLife::All.at(7), StrokeLife::All);
    }

    #[test]
    fn coverage_is_inclusive_and_survives_a_backwards_range() {
        assert!(StrokeLife::All.covers(0) && StrokeLife::All.covers(u64::MAX));
        assert!(StrokeLife::Single(5).covers(5));
        assert!(!StrokeLife::Single(5).covers(4) && !StrokeLife::Single(5).covers(6));
        let r = StrokeLife::Range(5, 9);
        assert!(!r.covers(4) && r.covers(5) && r.covers(9) && !r.covers(10));
        // Dragging `out` below `in` is a thing a DragValue lets you do; it
        // must not silently become an empty stroke.
        let back = StrokeLife::Range(9, 5);
        assert!(back.covers(5) && back.covers(7) && back.covers(9));
        assert!(!back.covers(4) && !back.covers(10));
    }
}

#[cfg(test)]
mod time_tests {
    use super::*;

    /// A TimeRemap parked on a sampled curve, the way the app hands one to
    /// the engine.
    fn warp(pts: Vec<[f32; 2]>) -> NodeKind {
        NodeKind::TimeRemap(TimeRemapParams { frame: 0.0, lut: pts, ..Default::default() })
    }

    #[test]
    fn lut_interpolates_extrapolates_and_freezes() {
        let half = [[0.0, 0.0], [100.0, 50.0]];
        assert_eq!(lut_eval(&half, 0.0), 0.0);
        assert_eq!(lut_eval(&half, 10.0), 5.0);
        // Extrapolation continues the end slope in both directions.
        assert_eq!(lut_eval(&half, 200.0), 100.0);
        assert_eq!(lut_eval(&half, -10.0), -5.0);

        // A flat span freezes; a falling span runs backwards.
        let hold = [[0.0, 0.0], [10.0, 10.0], [20.0, 10.0], [30.0, 0.0]];
        assert_eq!(lut_eval(&hold, 5.0), 5.0);
        assert_eq!(lut_eval(&hold, 15.0), 10.0, "flat span must freeze");
        assert_eq!(lut_eval(&hold, 25.0), 5.0, "falling span must reverse");

        // Degenerate inputs stay sane.
        assert_eq!(lut_eval(&[], 7.0), 7.0);
        assert_eq!(lut_eval(&[[0.0, 3.0]], 7.0), 3.0);
        assert!(lut_eval(&[[5.0, 1.0], [5.0, 9.0]], 5.0).is_finite());
    }

    #[test]
    fn a_range_remap_plays_source_in_out_over_target_in_out() {
        // 100 source frames over 50 target ones: double speed, ends meeting.
        let p = RangeRemapParams { src: (414.0, 513.0), dst: (1001.0, 1050.5), ..Default::default() };
        let m = p.map();
        assert_eq!(m.apply_f(1001.0), 414.0);
        assert!((m.apply_f(1050.5) - 513.0).abs() < 1e-3);
        assert!((p.speed() - 2.0).abs() < 1e-6);
        // A one-frame target holds rather than dividing by zero.
        let hold = RangeRemapParams { dst: (5.0, 5.0), ..p };
        // Out off: a slip putting src.in on dst.in, whatever the outs say.
        let slip = RangeRemapParams { out_off: true, ..p };
        assert_eq!(slip.affine(), (1.0, 414.0 - 1001.0));
        // In off: a slip through the outs.
        let slip_out = RangeRemapParams { in_off: true, ..p };
        assert_eq!(slip_out.affine(), (1.0, 513.0 - 1050.5));
        assert_eq!(hold.map().apply_f(99.0), 414.0);
    }

    #[test]
    fn unanimated_remap_freezes_on_its_frame() {
        // No curve sampled in: the node parks the whole shot on one frame.
        let p = TimeRemapParams { frame: 12.0, lut: Vec::new(), ..Default::default() };
        let m = p.map();
        for t in [0.0f32, 5.0, 99.0] {
            assert_eq!(m.apply_f(t), 12.0, "a static remap is a freeze");
        }
        assert_eq!(m.apply(99), 12);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn the_sampled_lut_is_never_persisted() {
        // It is rebuilt from the animation on every graph build; writing it
        // into setups would just be a stale copy of the curve.
        let p = TimeRemapParams { frame: 3.0, lut: vec![[0.0, 0.0], [10.0, 5.0]], ..Default::default() };
        let json = serde_json::to_string(&p).unwrap();
        assert!(!json.contains("lut"), "lut leaked into the setup: {json}");
        let back: TimeRemapParams = serde_json::from_str(&json).unwrap();
        assert_eq!(back.frame, 3.0);
        assert!(back.lut.is_empty());
    }

    #[test]
    fn chain_composes_outermost_first() {
        // A Slip of +4 sitting BELOW a half-speed warp shifts whole source
        // frames: tick -> slip -> warp.
        let slip = TimeMap::affine(1.0, 4.0);
        let warp = TimeMap::lut(vec![[0.0, 0.0], [100.0, 50.0]]);
        let chained = warp.clone().then(slip.clone());
        // tick 10 -> slip 14 -> warp 7
        assert_eq!(chained.apply_f(10.0), 7.0);
        assert_eq!(chained.apply(10), 7);
        // Order matters: the other way round is 4 + 5 = 9.
        assert_eq!(slip.then(warp).apply_f(10.0), 9.0);
        // Identity composes away.
        assert!(TimeMap::default().is_identity());
        assert_eq!(TimeMap::default().apply(42), 42);
        // apply() rounds to a real frame, apply_f() does not.
        let quarter = TimeMap::lut(vec![[0.0, 0.0], [4.0, 1.0]]);
        assert_eq!(quarter.apply_f(3.0), 0.75);
        assert_eq!(quarter.apply(3), 1);
        // Never negative: readers wrap, they don't index backwards.
        assert_eq!(TimeMap::affine(1.0, -50.0).apply(3), 0);
    }

    #[test]
    fn node_time_maps_cover_only_whats_upstream() {
        // read -> transform -> warp -> grade(view)
        let mut g = Graph::default();
        g.nodes.insert(1, NodeKind::read(0, ""));
        g.nodes.insert(2, NodeKind::Transform(TransformParams::default()));
        g.nodes.insert(3, warp(vec![[0.0, 0.0], [100.0, 50.0]]));
        g.nodes.insert(4, NodeKind::Grade(GradeParams::default()));
        g.edges.insert((2, 0), 1);
        g.edges.insert((3, 0), 2);
        g.edges.insert((4, 0), 3);
        g.view = Some(4);
        let maps = g.node_time_maps();
        // Below/at the warp: untouched clock.
        assert!(maps[&4].is_identity(), "grade below the warp");
        assert!(maps[&3].is_identity(), "the warp reads its own output clock");
        // Above it: warped, and sub-frame.
        assert_eq!(maps[&2].apply_f(10.0), 5.0);
        assert_eq!(maps[&1].apply_f(5.0), 2.5);

        // A bypassed time node passes time through, like its image.
        g.disabled.insert(3);
        let maps = g.node_time_maps();
        assert!(maps[&2].is_identity(), "bypassed warp must not remap");
    }
}

#[cfg(test)]
mod blend_tests {
    use super::*;

    /// read -> speed(half, blend) -> view
    fn blended_graph(interp: TimeInterp) -> Graph {
        let mut g = Graph::default();
        g.nodes.insert(1, NodeKind::read(0, ""));
        g.nodes.insert(
            2,
            NodeKind::Speed(SpeedParams { speed: 0.5, anchor: 0.0, interp, blend_w: 0.0 }),
        );
        g.edges.insert((2, 0), 1);
        g.view = Some(2);
        g
    }

    #[test]
    fn blending_instances_the_source_on_both_sides() {
        // Snapping: one reader, one frame.
        let near = blended_graph(TimeInterp::Nearest).reader_instances();
        assert_eq!(near.len(), 1, "nearest must not double the reads");
        assert_eq!(near[0].map.apply(5), 3, "2.5 rounds to 3");

        // Blending: the same source twice, one frame either side.
        let both = blended_graph(TimeInterp::Blend).reader_instances();
        assert_eq!(both.len(), 2, "blend needs both frames live");
        assert!(both.iter().all(|i| i.node == 1), "same source, two instances");
        let mut frames: Vec<u64> = both.iter().map(|i| i.map.apply(5)).collect();
        frames.sort();
        assert_eq!(frames, vec![2, 3], "tick 5 is source 2.5: frames 2 and 3");
        // Two distinct time paths, so they get their own reader slots.
        assert_ne!(both[0].key, both[1].key);
        // ...and the dissolve sits halfway between them.
        assert_eq!(both[0].map.blend_weight(5), 0.5);
    }

    #[test]
    fn blend_weight_tracks_the_sub_frame_position() {
        let g = blended_graph(TimeInterp::Blend);
        let inst = g.reader_instances();
        let m = &inst[0].map;
        // Quarter speed of a sort: 0.5x means every other tick is exact.
        assert_eq!(m.blend_weight(4), 0.0, "tick 4 -> source 2.0, no dissolve");
        assert_eq!(m.blend_weight(5), 0.5);
        assert_eq!(m.blend_weight(6), 0.0);
        // On an exact frame both sides still resolve sanely (floor == value,
        // ceil is one past) and the weight of 0 shows only the floor.
        let floors: Vec<u64> = inst.iter().map(|i| i.map.apply(4)).collect();
        assert!(floors.contains(&2));
    }

    #[test]
    fn a_snapping_node_emits_no_uniform_and_a_blending_one_does() {
        // Count the emitted dissolve STATEMENTS: the prelude's helper
        // functions use mix() for their own reasons, so match the shape the
        // time node emits — a mix whose weight comes from the uniform.
        let dissolves = |w: &str| {
            w.lines().filter(|l| l.contains("= mix(") && l.contains("P.v[")).count()
        };
        let near = crate::codegen::generate(&blended_graph(TimeInterp::Nearest));
        assert_eq!(dissolves(&near.main.wgsl), 0, "nearest must not dissolve");
        let mut bg = blended_graph(TimeInterp::Blend);
        let blend = crate::codegen::generate(&bg);
        assert_eq!(dissolves(&blend.main.wgsl), 1, "blend must dissolve once");
        assert!(blend.layout.node_slot.contains_key(&2), "weight needs a slot");

        // Give the two instances their own reader slots the way the app does,
        // and the pass samples both plates instead of one twice.
        bg.time_slots = bg
            .reader_instances()
            .iter()
            .enumerate()
            .map(|(i, inst)| ((inst.node, inst.key), i))
            .collect();
        let wired = crate::codegen::generate(&bg);
        assert_eq!(wired.main.sources.len(), 2, "{:?}", wired.main.sources);
    }
}

#[cfg(test)]
mod geo_topology_tests {
    use super::*;

    /// A Noise wired only into a 2dini's field port is read on the CPU, so
    /// reshaping its field graph must not change the comp's topology — and
    /// the moment it is also drawn, it must.
    #[test]
    fn a_noise_feeding_only_a_geo_field_port_is_not_topology() {
        let mut g = Graph::default();
        g.nodes.insert(1, NodeKind::Noise(Default::default()));
        g.nodes.insert(2, NodeKind::Geo(Default::default()));
        g.edges.insert((2, 0), 1);
        g.view = Some(2);
        let before = g.topology_hash();
        if let Some(NodeKind::Noise(n)) = g.nodes.get_mut(&1) {
            n.graph.nodes.insert(1, crate::field::FieldNode::default());
        }
        assert_eq!(before, g.topology_hash(), "a field edit behind a geo port recompiled");
        // Editing the geometry graph is never topology either.
        if let Some(NodeKind::Geo(p)) = g.nodes.get_mut(&2) {
            p.graph.nodes.insert(1, crate::geo::GeoNode::default());
        }
        assert_eq!(before, g.topology_hash(), "a geometry edit recompiled");
        // Viewed, the Noise is drawn by the shader again: its shape counts.
        g.view = Some(1);
        let viewed = g.topology_hash();
        if let Some(NodeKind::Noise(n)) = g.nodes.get_mut(&1) {
            n.graph.nodes.insert(2, crate::field::FieldNode::Output);
        }
        assert_ne!(viewed, g.topology_hash(), "a drawn noise's shape is topology");
        // Single channel and 8 bits change the emitted code; the speed unit
        // is a uniform.
        let shaped = g.topology_hash();
        if let Some(NodeKind::Noise(n)) = g.nodes.get_mut(&1) {
            n.speed_unit = 1.0;
        }
        assert_eq!(shaped, g.topology_hash(), "the speed unit recompiled");
        if let Some(NodeKind::Noise(n)) = g.nodes.get_mut(&1) {
            n.single = true;
        }
        let single = g.topology_hash();
        assert_ne!(shaped, single, "single channel is topology");
        if let Some(NodeKind::Noise(n)) = g.nodes.get_mut(&1) {
            n.eight_bit = true;
        }
        assert_ne!(single, g.topology_hash(), "8 bits is topology");
    }
}

#[cfg(test)]
mod switch_tests {
    use super::*;

    /// read(1) -> switch(3).0, read(2) -> switch.1, viewing the switch.
    fn switched(which: u32) -> Graph {
        let mut g = Graph::default();
        g.nodes.insert(1, NodeKind::read(0, ""));
        g.nodes.insert(2, NodeKind::read(1, ""));
        g.nodes.insert(3, NodeKind::Switch(SwitchParams { which, inputs: 2 }));
        g.edges.insert((3, 0), 1);
        g.edges.insert((3, 1), 2);
        g.view = Some(3);
        g
    }

    #[test]
    fn only_the_live_branch_is_decoded() {
        // The whole point: the branch you are not on is not decoded. If both
        // Reads were live, a Switch over two plates would cost both.
        for (which, want) in [(0u32, 1u64), (1, 2)] {
            let inst = switched(which).reader_instances_from(&[3]);
            assert_eq!(inst.len(), 1, "which={which} instanced {inst:?}");
            assert_eq!(inst[0].node, want, "which={which} took the wrong branch");
        }
    }

    #[test]
    fn the_rig_holds_every_branch_so_a_flip_rebuilds_nothing() {
        // The rig is built from `reader_instances`; if it followed the
        // switch, an animated one rebuilt the rig (dropping every decoded
        // frame) each time it flipped.
        let a = switched(0).reader_instances();
        let b = switched(1).reader_instances();
        assert_eq!(a.len(), 2, "{a:?}");
        assert_eq!(a, b, "the rig must not depend on the live input");
    }

    #[test]
    fn only_the_live_branch_is_compiled() {
        for (which, other) in [(0u32, 2u64), (1, 1)] {
            let mut g = switched(which);
            // Give the instances their slots the way the app does, so the
            // two branches would bind different textures if both were live.
            g.time_slots = g
                .reader_instances()
                .iter()
                .enumerate()
                .map(|(i, inst)| ((inst.node, inst.key), i))
                .collect();
            let plan = crate::codegen::generate(&g);
            assert_eq!(
                plan.main.sources.len(),
                1,
                "which={which} bound {:?}",
                plan.main.sources
            );
            // And the dead branch contributes no filter stage either.
            assert!(plan.blurs.is_empty(), "which={which} built stages for a dead branch");
            let _ = other;
        }
    }

    #[test]
    fn which_is_structural_so_the_pipeline_cache_can_hold_both() {
        // `which` MUST reach the topology hash — it picks the program, not a
        // uniform. Two values, two hashes, both cacheable.
        let a = switched(0).topology_hash();
        let b = switched(1).topology_hash();
        assert_ne!(a, b, "flipping the switch must recompile, not reuse");
        assert_eq!(a, switched(0).topology_hash(), "same switch, same hash");
        // Input count is structural too.
        let mut wide = switched(0);
        if let Some(NodeKind::Switch(p)) = wide.nodes.get_mut(&3) {
            p.inputs = 4;
        }
        assert_ne!(wide.topology_hash(), a);
    }

    #[test]
    fn out_of_range_and_degenerate_values_still_pick_a_real_input() {
        // `which` past the end clamps rather than emitting nothing, and a
        // Switch is never narrower than two inputs.
        let p = SwitchParams { which: 9, inputs: 2 };
        assert_eq!(p.live(), 1);
        assert_eq!(SwitchParams { which: 0, inputs: 0 }.n(), 2);
        assert_eq!(SwitchParams { which: 0, inputs: 99 }.n(), 16);
        let g = switched(9);
        let inst = g.reader_instances_from(&[3]);
        assert_eq!(inst.len(), 1);
        assert_eq!(inst[0].node, 2, "clamped to the last input");
        // An unwired live input renders nothing rather than falling back to
        // a neighbour — a Switch is not a Merge.
        let mut bare = switched(1);
        bare.edges.remove(&(3, 1));
        let plan = crate::codegen::generate(&bare);
        assert!(plan.main.sources.is_empty(), "{:?}", plan.main.sources);
    }
}

#[cfg(test)]
mod zone_tests {
    use super::*;
    use crate::bands::BandRanges;

    fn named(names: &[&str]) -> LumaKeyParams {
        let mut k = LumaKeyParams {
            bands: Vec::new(),
            ranges: Some(BandRanges::even(names.len(), LUMA_SPAN, false, 0.5)),
        };
        k.sync_from_ranges();
        for (b, n) in k.bands.iter_mut().zip(names) {
            b.name = n.to_string();
        }
        k
    }

    fn names(k: &LumaKeyParams) -> Vec<String> {
        k.bands.iter().map(|b| b.name.clone()).collect()
    }

    fn gain(v: f32) -> GradeZone {
        GradeZone { gain: [v; 3], ..Default::default() }
    }

    fn gains(z: &GradeZones) -> Vec<f32> {
        z.adjust.iter().map(|a| a.gain[0]).collect()
    }

    /// A Grade that never touched zones is the Grade it always was: no
    /// zones, the tag it always had, so its pipeline and caches survive.
    #[test]
    fn a_plain_grade_keeps_its_tag_and_zones_are_the_only_structure() {
        let plain = NodeKind::Grade(GradeParams::default());
        assert_eq!(plain.topo_tag(), 2);
        let mut zoned = GradeParams { zones: Some(GradeZones::default()), ..Default::default() };
        let t = NodeKind::Grade(zoned.clone()).topo_tag();
        assert_ne!(t, 2, "adding zones must recompile");
        assert_eq!(t & 0xffff_ffff, 2, "the low word is still the Grade's own tag");
        // Every number inside a zone is a uniform.
        let z = zoned.zones.as_mut().unwrap();
        z.adjust[1].gain = [2.0; 3];
        z.key.ranges.as_mut().unwrap().move_edge(0, 0.3);
        z.key.sync_from_ranges();
        z.key.bands[1].points.push([0.5, 0.7]);
        assert_eq!(NodeKind::Grade(zoned.clone()).topo_tag(), t);
        // One more zone is one more span.
        zoned.zones.as_mut().unwrap().key.split_band(1);
        assert_ne!(NodeKind::Grade(zoned).topo_tag(), t);
    }

    /// The default zones are the classic three, identity, tiling the axis.
    #[test]
    fn default_zones_are_the_classic_three() {
        let z = GradeZones::default();
        assert_eq!(z.count(), 3);
        assert_eq!(z.adjust.len(), 3);
        assert!(z.adjust.iter().all(|a| *a == GradeZone::default()));
        assert_eq!(
            (0..3).map(|i| z.label(i)).collect::<Vec<_>>(),
            ["shadows", "midtones", "highlights"]
        );
        let r = z.key.ranges.as_ref().unwrap();
        for k in 0..=40 {
            let s: f32 = r.weights(k as f32 / 20.0).iter().sum();
            assert!((s - 1.0).abs() < 1e-4);
        }
        let w = |x: f32| r.weights(x);
        assert!(w(0.0)[0] > 0.99, "black is all shadows: {:?}", w(0.0));
        assert!(w(0.3)[1] > 0.99, "0.3 is all midtones: {:?}", w(0.3));
        assert!(w(1.5)[2] > 0.99, "1.5 is all highlights: {:?}", w(1.5));
        // A band's own name wins over the classic one.
        let mut z = z;
        z.key.bands[1].name = "skin".into();
        assert_eq!(z.label(1), "skin");
        assert_eq!(GradeZones { key: named(&["", ""]), adjust: vec![], ..Default::default() }.label(1), "highlights");
    }

    /// Splitting a band puts the new band ABOVE it, unnamed; the bands above
    /// keep their own names, and a zone's correction covers both halves.
    #[test]
    fn splitting_keeps_names_in_place_and_copies_the_correction() {
        let mut k = named(&["a", "b", "c"]);
        let new = k.split_band(0);
        assert_eq!(new, 1);
        assert_eq!(names(&k), ["a", "", "b", "c"]);
        let r = k.ranges.as_ref().unwrap();
        assert_eq!(r.band_count(), 4);
        // Split the last: nothing above to shift.
        let new = k.split_band(3);
        assert_eq!(new, 4);
        assert_eq!(names(&k), ["a", "", "b", "c", ""]);

        let mut z = GradeZones {
            key: named(&["a", "b", "c"]),
            adjust: vec![gain(1.0), gain(2.0), gain(3.0)],
            ..Default::default()
        };
        let new = z.key.split_band(1);
        z.band_split(new);
        assert_eq!(gains(&z), [1.0, 2.0, 2.0, 3.0]);
        assert_eq!(z.adjust.len(), z.count());
    }

    /// Dissolving a band merges it into its neighbour, which keeps its own
    /// name and correction; the last band merges downward.
    #[test]
    fn dissolving_keeps_the_neighbour() {
        let mut k = named(&["a", "b", "c"]);
        k.dissolve_band(1);
        assert_eq!(names(&k), ["a", "c"]);
        let mut k = named(&["a", "b", "c"]);
        k.dissolve_band(2);
        assert_eq!(names(&k), ["a", "b"]);
        let mut k = named(&["a"]);
        k.dissolve_band(0);
        assert_eq!(names(&k), ["a"], "a one-band key cannot dissolve");

        let mut z = GradeZones {
            key: named(&["a", "b", "c"]),
            adjust: vec![gain(1.0), gain(2.0), gain(3.0)],
            ..Default::default()
        };
        z.key.dissolve_band(0);
        z.band_dissolved(0);
        assert_eq!(gains(&z), [2.0, 3.0]);
        assert_eq!(z.adjust.len(), z.count());
    }

    /// One view for every page: the master's five or a zone's.
    #[test]
    fn adjust_mut_addresses_the_master_or_a_zone() {
        let mut g = GradeParams { zones: Some(GradeZones::default()), ..Default::default() };
        *g.adjust_mut(None).unwrap().gain = [2.0; 3];
        *g.adjust_mut(Some(2)).unwrap().saturation = 0.5;
        assert!(g.adjust_mut(Some(3)).is_none());
        assert_eq!(g.gain, [2.0; 3]);
        assert_eq!(g.zones.as_ref().unwrap().adjust[2].saturation, 0.5);
        assert_eq!(g.zones.as_ref().unwrap().adjust[0].saturation, 1.0);
        let mut plain = GradeParams::default();
        assert!(plain.adjust_mut(Some(0)).is_none());
        assert!(plain.adjust_mut(None).is_some());
    }

    /// CIE76 ΔE between two sRGB colours (D65).
    fn delta_e(a: [u8; 3], b: [u8; 3]) -> f32 {
        let lab = |c: [u8; 3]| {
            let l = c.map(|v| crate::color::Curve::Srgb.to_linear(v as f32 / 255.0));
            let xyz = [
                (0.4124 * l[0] + 0.3576 * l[1] + 0.1805 * l[2]) / 0.9505,
                0.2126 * l[0] + 0.7152 * l[1] + 0.0722 * l[2],
                (0.0193 * l[0] + 0.1192 * l[1] + 0.9505 * l[2]) / 1.089,
            ];
            let f = xyz.map(|t| if t > 0.008856 { t.cbrt() } else { 7.787 * t + 16.0 / 116.0 });
            [116.0 * f[1] - 16.0, 500.0 * (f[0] - f[1]), 200.0 * (f[1] - f[2])]
        };
        let (p, q) = (lab(a), lab(b));
        ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
    }

    /// The band palette, judged the way it is SEEN — through the default
    /// ACES view, in an ACEScg comp, as CIE76 ΔE: neighbours plainly apart
    /// everywhere in the sequence (the cycle included), every pair far apart
    /// for the zone counts people use, three zones very far apart. And never
    /// an orange: not in a colour (a standing UI rule — the plot tint once
    /// ended in salmon) and not anywhere along a neighbour's crossfade,
    /// because the blended map mixes the two. A zone past the end reads as
    /// the last rather than panicking.
    #[test]
    fn the_band_palette_is_distinct_as_shown_and_has_no_orange() {
        let orange = |c: [f32; 3]| {
            let [r, g, b] = c;
            let (mx, mn) = (r.max(g).max(b), r.min(g).min(b));
            let d = (mx - mn).max(1e-6);
            let h = if mx == r {
                (g - b) / d
            } else if mx == g {
                2.0 + (b - r) / d
            } else {
                4.0 + (r - g) / d
            };
            let h = ((h * 60.0) + 360.0) % 360.0;
            (mx - mn) / mx.max(1e-3) > 0.25 && (15.0..48.0).contains(&h)
        };
        let working = crate::color::Colorspace::ACESCG;
        let mut view = crate::viewtf::ViewSettings::default();
        view.mirror_working(working);
        let shown = |lin: [f32; 3]| {
            view.display_rgb(lin).unwrap().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        };
        const N: usize = 12; // past the list, so the cycle's seam is judged too
        let lin: Vec<[f32; 3]> = (0..N).map(|i| band_linear(i, N, working)).collect();
        let seen: Vec<[u8; 3]> = lin.iter().map(|&l| shown(l)).collect();
        for i in 0..N {
            let c = band_srgb(i, N);
            assert!(!orange(c.map(|v| v as f32 / 255.0)), "band {i} is orange: {c:?}");
            assert!(!orange(seen[i].map(|v| v as f32 / 255.0)), "band {i} is SHOWN orange: {:?}", seen[i]);
            if i + 1 < N {
                let d = delta_e(seen[i], seen[i + 1]);
                assert!(d >= 25.0, "bands {i} and {} are {d:.0} ΔE apart as shown", i + 1);
                for k in 0..=20 {
                    let t = k as f32 / 20.0;
                    let mixed: [f32; 3] = std::array::from_fn(|c| lin[i][c] * (1.0 - t) + lin[i + 1][c] * t);
                    let m = shown(mixed);
                    assert!(
                        !orange(m.map(|v| v as f32 / 255.0)),
                        "the crossfade from band {i} to {} passes through orange {m:?} at {t}",
                        i + 1
                    );
                }
            }
        }
        for (n, floor) in [(2usize, 100.0f32), (3, 100.0), (4, 60.0), (5, 55.0)] {
            for a in 0..n {
                for b in a + 1..n {
                    let d = delta_e(shown(band_linear(a, n, working)), shown(band_linear(b, n, working)));
                    assert!(d >= floor, "{n} zones: {a} and {b} are only {d:.0} ΔE apart as shown");
                }
            }
        }
        for n in 1..=16usize {
            assert_eq!(band_srgb(n + 3, n), band_srgb(n - 1, n));
        }
        // Linear light in the working primaries: Rec.709 is the sRGB decode.
        let l = band_linear(1, 3, crate::color::Colorspace::LINEAR_REC709);
        let s = band_srgb(1, 3);
        for c in 0..3 {
            let want = crate::color::Curve::Srgb.to_linear(s[c] as f32 / 255.0);
            assert!((l[c] - want).abs() < 1e-5);
        }
    }

    /// The preview is a knob — authored, undoable, saved — that only ACTS
    /// when the app arms it for the node being viewed, and never names a
    /// zone the partition no longer has.
    #[test]
    fn a_preview_is_authored_but_only_acts_when_armed() {
        let mut z = GradeZones::default();
        assert_eq!(z.preview.view, ZoneView::Off);
        assert!(z.live_preview().is_none());
        z.preview = ZonePreview { view: ZoneView::Matte, zone: 1 };
        assert!(z.live_preview().is_none(), "authored is not armed");
        z.previewing = true;
        assert_eq!(z.live_preview(), Some((ZoneView::Matte, 1)));
        // A dissolve can leave the selection past the end.
        z.key.dissolve_band(2);
        z.band_dissolved(2);
        z.preview.zone = 7;
        assert_eq!(z.live_preview(), Some((ZoneView::Matte, z.count() as u32 - 1)));
        // Off wins over armed.
        z.preview.view = ZoneView::Off;
        assert!(z.live_preview().is_none());
    }

    /// The MODE saves with the setup, like the keyer's own view; the arming
    /// never does, so nothing opens already replacing its picture.
    #[cfg(feature = "serde")]
    #[test]
    fn the_preview_mode_saves_but_the_arming_never_does() {
        let mut z = GradeZones::default();
        z.preview = ZonePreview { view: ZoneView::Highlight, zone: 1 };
        z.previewing = true;
        let back: GradeZones =
            serde_json::from_str(&serde_json::to_string(&z).unwrap()).unwrap();
        assert_eq!(back.preview, z.preview);
        assert!(!back.previewing, "a setup must not open already previewing");
    }

    /// A setup written before zones existed loads as a plain Grade, and a
    /// zoned one round-trips.
    #[cfg(feature = "serde")]
    #[test]
    fn zones_survive_serde_and_old_grades_load_plain() {
        let old: GradeParams = serde_json::from_str(r#"{"gain":[2.0,2.0,2.0]}"#).unwrap();
        assert!(old.zones.is_none());
        assert_eq!(old.gain, [2.0; 3]);
        let mut g = GradeParams { zones: Some(GradeZones::default()), ..Default::default() };
        g.zones.as_mut().unwrap().adjust[0].lift = [0.1, 0.2, 0.3];
        g.zones.as_mut().unwrap().key.bands[0].name = "toe".into();
        let s = serde_json::to_string(&g).unwrap();
        let back: GradeParams = serde_json::from_str(&s).unwrap();
        assert_eq!(back, g);
    }
}


#[cfg(test)]
mod placement_tests {
    use super::*;

    fn read_at(start: Option<i64>) -> Graph {
        let mut g = Graph::default();
        let mut k = NodeKind::read(0, "x");
        if let NodeKind::Read { start: s, .. } = &mut k {
            *s = start;
        }
        g.nodes.insert(1, k);
        g.view = Some(1);
        g
    }

    /// Files numbered 1001.. sit at comp 1001..: the leaf map turns comp
    /// ticks into sequence indices, and ticks before the start clamp to
    /// frame one (the hold).
    #[test]
    fn a_read_placed_at_its_file_number() {
        let g = read_at(Some(1001));
        let inst = g.reader_instances();
        assert_eq!(inst.len(), 1);
        assert_eq!(inst[0].map.apply(1001), 0, "first file at its own number");
        assert_eq!(inst[0].map.apply(1050), 49);
        assert_eq!(inst[0].map.apply(500), 0, "before the start holds frame one");
    }

    /// A TRIM does not slide the clip. The player hands the reader the
    /// trimmed sequence, so its frame one is the trim's first frame — and
    /// the picture has to stay at the frames the range bar shows.
    #[test]
    fn a_trim_keeps_the_clip_where_its_frames_are() {
        let mut g = read_at(Some(1001));
        if let Some(NodeKind::Read { range, .. }) = g.nodes.get_mut(&1) {
            *range = Some((4, 9));
        }
        let inst = g.reader_instances();
        // Comp 1005 is the trim's first frame — index 0 of the trimmed
        // sequence — and the frames before it hold it.
        assert_eq!(inst[0].map.apply(1005), 0);
        assert_eq!(inst[0].map.apply(1010), 5);
        assert_eq!(inst[0].map.apply(1001), 0, "before the trim holds its first frame");
    }

    /// Placement composes with the time nodes exactly like a Slip under
    /// them: a Slip +3 above a 1001-placed Read reads three frames later.
    #[test]
    fn placement_composes_with_a_slip() {
        let mut g = read_at(Some(1001));
        g.nodes.insert(2, NodeKind::Slip(SlipParams { offset: 3 }));
        g.edges.insert((2, 0), 1);
        g.view = Some(2);
        let inst = g.reader_instances();
        assert_eq!(inst.len(), 1);
        assert_eq!(inst[0].map.apply(1001), 3);
    }

    /// Unplaced (None or 0) is byte-identical to the old behaviour.
    #[test]
    fn unplaced_reads_are_unchanged() {
        for start in [None, Some(0)] {
            let g = read_at(start);
            let inst = g.reader_instances();
            assert_eq!(inst[0].map.apply(7), 7);
            assert!(inst[0].map.is_identity() || inst[0].map.apply(0) == 0);
        }
    }
}

#[cfg(test)]
mod pipe_tests {
    use super::*;

    /// The wiring rule, both directions pinned. Widening coverage into a
    /// measurement is free; narrowing a measurement into coverage needs a
    /// range nobody can pick for you, so it is refused.
    #[test]
    fn matte_widens_into_data_and_never_back() {
        use PortKind::*;
        for k in [Image, Mask, Data, Field, Geo, Channels] {
            assert!(k.can_feed(k), "{k:?} must feed itself");
        }
        assert!(Mask.can_feed(Data), "a matte is valid data");
        assert!(!Data.can_feed(Mask), "data into a matte needs a range: refuse");
        assert!(!Image.can_feed(Data), "an image needs a channel pick first");
        assert!(!Data.can_feed(Image), "data has no colour meaning");
    }

    /// A depth is a measurement, and the one way onto its pipe is ToData.
    #[test]
    fn the_depth_port_takes_data_and_todata_gives_it() {
        let d = NodeKind::Defocus(Default::default());
        assert_eq!(d.input_kind(2), PortKind::Data, "Z");
        assert_eq!(d.input_kind(0), PortKind::Image, "the picture itself stays an image");
        assert_eq!(d.input_kind(3), PortKind::Image, "the bokeh kernel is a picture");
        let t = NodeKind::ToData(Default::default());
        assert_eq!((t.input_kind(0), t.output_kind()), (PortKind::Image, PortKind::Data));
        // The depth is still the PICTURE pipe for resolution and ranges.
        assert!(PortKind::Data.is_picture() && !PortKind::Mask.is_picture());
    }

    /// A dot wears its wire's pipe, the data pipe included, and a dot
    /// written before the pipe existed still reads as an image dot.
    #[test]
    fn a_dot_carries_the_data_pipe() {
        let dot = NodeKind::dot(PortKind::Data);
        assert_eq!((dot.input_kind(0), dot.output_kind()), (PortKind::Data, PortKind::Data));
        #[cfg(feature = "serde")]
        {
            let old: NodeKind =
                serde_json::from_str(r#"{"Dot":{"mask":false,"channels":false}}"#).unwrap();
            assert_eq!(old.output_kind(), PortKind::Image);
        }
    }
}

#[cfg(test)]
mod corner_pin_from_tests {
    use super::*;

    /// Invert undoes the pin: the inverted node after the plain one is the
    /// identity, and an edit made in effective terms round-trips.
    #[test]
    fn an_inverted_pin_undoes_the_plain_one() {
        let fwd = CornerPinParams { to: B, from: A, ..Default::default() };
        let inv = CornerPinParams { invert: true, ..fwd };
        for p in [[0.5, 0.5], [0.1, 0.9], [0.33, 0.2]] {
            let q = warp(&inv, warp(&fwd, p));
            assert!((q[0] - p[0]).abs() < 1e-3 && (q[1] - p[1]).abs() < 1e-3, "{p:?} -> {q:?}");
        }
        let back = inv.effective_to();
        let edited = CornerPinParams { to: inv.to_for_effective(back), ..inv };
        for (a, b) in edited.to.iter().zip(inv.to.iter()) {
            assert!((a[0] - b[0]).abs() < 1e-3 && (a[1] - b[1]).abs() < 1e-3, "{a:?} vs {b:?}");
        }
    }

    /// A Transform's inverse is the inverse of its whole map, pin and bend
    /// included.
    #[test]
    fn an_inverted_transform_undoes_the_plain_one() {
        let t = TransformParams {
            translate: [30.0, -12.0],
            rotate: 17.0,
            scale: [1.3, 0.8],
            shear: [0.1, 0.0],
            taper: 0.2,
            pin: A,
            ..Default::default()
        };
        let sz = [200.0, 100.0];
        let f = crate::codegen::transform_applied_px(&t, sz);
        let i = crate::codegen::transform_applied_px(&TransformParams { invert: true, ..t.clone() }, sz);
        for p in [[100.0, 50.0], [10.0, 90.0], [170.0, 20.0]] {
            let q = h_apply(&i, h_apply(&f, p));
            assert!((q[0] - p[0]).abs() < 1e-2 && (q[1] - p[1]).abs() < 1e-2, "{p:?} -> {q:?}");
        }
    }

    fn warp(cp: &CornerPinParams, p: [f32; 2]) -> [f32; 2] {
        h_apply(&crate::codegen::corner_pin_forward_px(cp, [1.0, 1.0]), p)
    }

    const A: [[f32; 2]; 4] = [[0.2, 0.3], [0.7, 0.25], [0.8, 0.9], [0.1, 0.7]];
    const B: [[f32; 2]; 4] = [[0.1, 0.1], [0.9, 0.2], [0.85, 0.95], [0.3, 0.8]];

    #[test]
    fn the_same_quad_twice_moves_nothing() {
        let cp = CornerPinParams { to: A, from: A, ..Default::default() };
        assert_eq!(cp.effective_to(), UNIT_QUAD);
        for p in [[0.5, 0.5], [0.0, 1.0], [0.33, 0.1]] {
            let q = warp(&cp, p);
            assert!((q[0] - p[0]).abs() < 1e-5 && (q[1] - p[1]).abs() < 1e-5, "{p:?} -> {q:?}");
        }
    }

    #[test]
    fn the_warp_carries_from_onto_to() {
        let cp = CornerPinParams { to: B, from: A, ..Default::default() };
        for i in 0..4 {
            let q = warp(&cp, A[i]);
            assert!((q[0] - B[i][0]).abs() < 1e-4 && (q[1] - B[i][1]).abs() < 1e-4, "corner {i}: {q:?}");
        }
        // And back: the `to` an effective edit asks for keeps `from`.
        let back = cp.to_for_effective(cp.effective_to());
        for i in 0..4 {
            assert!((back[i][0] - B[i][0]).abs() < 1e-4 && (back[i][1] - B[i][1]).abs() < 1e-4);
        }
    }

    #[test]
    fn the_frame_as_from_is_the_classic_pin() {
        let cp = CornerPinParams::pinned(B);
        assert_eq!(cp.effective_to(), B);
    }
}

#[cfg(test)]
mod format_input_tests {
    use super::*;

    /// A Dilate's matte IS its picture: the format and frame-range walks
    /// must follow it, or a Dilate takes the canvas and changes the
    /// resolution of the matte it grows. A matte that LIMITS a node (a
    /// Grade's mask, a Blur's modulate) is not where the format comes from.
    #[test]
    fn a_matte_operand_carries_the_format_and_a_side_matte_does_not() {
        let dilate = NodeKind::Dilate(Default::default());
        assert!(dilate.inherits_from(0), "Dilate's matte is its subject");
        assert!(!dilate.inherits_from(1), "Dilate's modulate is a side input");
        let comb = NodeKind::MatteCombine(Default::default());
        assert!(comb.inherits_from(0));
        let grade = NodeKind::Grade(Default::default());
        assert!(grade.inherits_from(0));
        assert!(!grade.inherits_from(grade.mask_input().unwrap()));
        let blur = NodeKind::Blur(Default::default());
        assert!(!blur.inherits_from(blur.mod_input().unwrap()));
        let mut gamma = NodeKind::Gamma(Default::default());
        gamma.set_matte_mode(true);
        assert!(gamma.inherits_from(0), "a Gamma of a matte keeps the matte's format");
    }

    /// Following a pipe is the narrow thing it says: input 0 of the nodes
    /// that work on coverage as they work on rgb, image or matte only.
    #[test]
    fn only_pipe_followers_welcome_a_matte() {
        let gamma = NodeKind::Gamma(Default::default());
        assert!(gamma.accepts(0, PortKind::Mask));
        assert!(gamma.accepts(0, PortKind::Image));
        assert!(!gamma.accepts(0, PortKind::Data), "data has no colour meaning");
        // A Render writes data as data (one float channel, no transform);
        // nothing else learns to take it from this.
        let write = NodeKind::Write(Default::default());
        assert!(write.accepts(0, PortKind::Data) && write.accepts(0, PortKind::Image));
        assert!(!PortKind::Data.can_feed(PortKind::Image), "the pipe rule itself is unchanged");
        assert!(!gamma.accepts(0, PortKind::Channels));
        // Filters and transforms move coverage as they move pixels.
        for k in [
            NodeKind::Blur(Default::default()),
            NodeKind::Median(Default::default()),
            NodeKind::Transform(Default::default()),
            NodeKind::CornerPin(Default::default()),
        ] {
            assert!(k.accepts(0, PortKind::Mask), "{} should take a matte", k.name());
        }
        let sat = NodeKind::Saturation(Default::default());
        assert!(!sat.accepts(0, PortKind::Mask), "saturation means nothing on a matte");
        let mut dilate = NodeKind::Dilate(Default::default());
        dilate.set_matte_mode(false); // not a follower: a no-op
        assert_eq!(dilate.input_kind(0), PortKind::Mask);
    }
}

#[cfg(test)]
mod layer_extend_tests {
    use super::*;

    fn map(edge: ReadEdge) -> TimeMap {
        let l = LayerItem { extend: Some(LayerExtend { edge, lo: 10, hi: 13 }), ..Default::default() };
        NodeKind::Layer(LayerParams { layers: vec![l], outputs: false }).input_time_map(0)
    }

    /// Past its clip a layer holds, loops or bounces it; inside it plays
    /// the input's own frames untouched.
    #[test]
    fn a_layer_runs_on_past_its_clip() {
        let at = |m: &TimeMap, t: u64| m.apply_i(t);
        let hold = map(ReadEdge::Hold);
        assert_eq!((at(&hold, 11), at(&hold, 20), at(&hold, 3)), (11, 13, 10));
        let lp = map(ReadEdge::Loop);
        assert_eq!((at(&lp, 14), at(&lp, 17), at(&lp, 9)), (10, 13, 13));
        let b = map(ReadEdge::Bounce);
        assert_eq!((at(&b, 14), at(&b, 16), at(&b, 17)), (12, 10, 11));
        assert!(!lp.is_identity());
    }
}
