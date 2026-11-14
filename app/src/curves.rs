//! Curve editor: a value-over-time graph of the ANIMATED channels on the
//! selected nodes (static channels never appear — an empty graph means
//! nothing is keyed).
//!
//! Navigation is Blender's, shared with every canvas (`crate::nav`): MMB /
//! alt+LMB pan, wheel zooms about the cursor, ctrl/shift+wheel pan by axis,
//! trackpad pans, pinch zooms; R, Home or double-click frames. The two axes
//! are decoupled like Blender's graph editor: ctrl+MMB (ctrl+alt+LMB)
//! stretches time and value separately, alt+wheel zooms value alone.
//!
//! Owned by instance B (see .agents/board.md); the only lib.rs footprint is
//! `mod curves;`, one App field and the panel block.

use crate::anim::{frame_id, id_frame, AnimCurve, Channel, Ease, EaseDir, Extend, FrameId, Interp, Key};
use crate::curveops::{self, Slide};
use crate::editor::EditorGraph;
use crate::keymap::{self, Act};
use crate::theme;
use egui::{Color32, Pos2, Rect, Stroke, Vec2};
use std::collections::{BTreeMap, HashMap, HashSet};

/// A key's identity: which curve it belongs to and which frame it sits on.
type KeyId = (u64, Channel, FrameId);

/// One visible curve.
struct Track {
    node: u64,
    ch: Channel,
    color: Color32,
    /// The channel's own name, without the node (the list groups by node).
    short: String,
}

/// A bezier handle: the key it hangs off, and whether it is the outgoing one.
type HandleId = (u64, Channel, FrameId, bool);

/// Grabbed keys: node, channel, and the WHOLE key as the gesture found it
/// (interpolation and handles travel with it).
type Grabbed = Vec<(u64, Channel, Key)>;

/// Curves as they were when a gesture began.
type Snapshot = HashMap<(u64, Channel), AnimCurve>;

/// Normalize view: per curve (mid, half), shown value = (value - mid) / half.
type Norms = HashMap<(u64, Channel), (f32, f32)>;

enum Drag {
    Pan,
    /// Dragging a bezier handle.
    Handle(HandleId),
    /// Axis dolly (alt + right-drag): dx scales time, dy scales value.
    Dolly { anchor: Pos2 },
    Marquee { start: Pos2 },
    /// Scaling the selected keys about the opposite edge of their box.
    Scale {
        snapshot: Snapshot,
        grabbed: Grabbed,
        /// Fixed corner in SHOWN space (frame, shown value).
        anchor: (f32, f32),
        origin: Pos2,
        ax: bool,
        ay: bool,
    },
    /// Moving keys. Curves are restored from the snapshot every frame and
    /// re-offset, so the gesture stays exact no matter how it wanders.
    Keys { snapshot: Snapshot, grabbed: Grabbed, origin: Pos2 },
    /// Houdini's ladder (D held): up / down at the press picks a step,
    /// then left / right counts steps onto every grabbed value.
    Ladder { snapshot: Snapshot, grabbed: Grabbed, origin: Pos2, rung: usize, lock_x: Option<f32> },
}

/// Steps of the ladder's rungs, top to bottom; the press lands on `LADDER_HOME`.
const LADDER: [f32; 6] = [100.0, 10.0, 1.0, 0.1, 0.01, 0.001];
const LADDER_HOME: usize = 2;
const RUNG_H: f32 = 20.0;
/// Pointer travel per step once a rung is locked.
const LADDER_PX: f32 = 10.0;

/// Blender's modal transforms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Xform {
    Grab,
    Scale,
    Rotate,
}

/// What scale and rotate turn about (Blender's pivot point).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Pivot {
    /// The centre of the selected keys' box.
    #[default]
    Box,
    /// The playhead's frame (at the box's middle value).
    Playhead,
    /// Every key about itself: keys stay put, their HANDLES scale / turn.
    Each,
}

impl Pivot {
    const ALL: [Pivot; 3] = [Pivot::Box, Pivot::Playhead, Pivot::Each];
    fn name(self) -> &'static str {
        match self {
            Pivot::Box => "box centre",
            Pivot::Playhead => "playhead",
            Pivot::Each => "each key",
        }
    }
}

/// A modal G / S / R: the pointer drives it, a click or Enter keeps it,
/// right-click or Esc puts everything back.
struct Modal {
    kind: Xform,
    snapshot: Snapshot,
    grabbed: Grabbed,
    origin: Pos2,
    /// (frame, shown value).
    pivot: (f32, f32),
}

/// A key slider in progress: left / right sets the factor.
struct SliderRun {
    kind: Slide,
    snapshot: Snapshot,
    /// Per curve: its selected frames and its default value.
    curves: Vec<((u64, Channel), Vec<f64>, Option<f32>)>,
    origin_x: f32,
    factor: f32,
}
/// Pointer travel for a full slider throw.
const SLIDE_PX: f32 = 250.0;

/// A frozen copy of a curve, drawn faintly behind the live one.
struct Ghost {
    node: u64,
    ch: Channel,
    color: Color32,
    curve: AnimCurve,
}

/// What a slice of the curve-ops pie (or a header button, or a key) does.
#[derive(Clone, Copy, Debug, PartialEq)]
enum PieOp {
    /// Both sides of every selected key.
    Interp(Interp),
    /// Bezier with level handles: the key eases in and out flat.
    Flatten,
    /// Tangent: handles aligned (and bezier both sides).
    Unify,
    /// Broken tangent: handles independent.
    Break,
    /// Vector handles: each points at its neighbour, and follows it.
    Vector,
    Snap,
    Frame,
    FrameSelected,
    SelectAll,
    SelectLeft,
    SelectRight,
    SelectColumn,
    Delete,
    Copy,
    Paste,
    /// Blender's Noise modifier on the targeted curves: added where
    /// missing, taken off when they all have one.
    Noise,
    /// Blender's Stepped modifier, the same way.
    Stepped,
    /// Cycle both ends on (or off, when every target already cycles).
    Cycle,
    /// Set the extrapolation: before (`Some(false)`), after
    /// (`Some(true)`) or both (`None`).
    Extend(Option<bool>, Extend),
    /// Start a key slider.
    Slide(Slide),
    /// Undo ±360° jumps.
    Unwrap,
    MirrorTime,
    MirrorValue,
    /// Start a modal transform.
    Xform(Xform),
    /// Freeze the shown curves as ghosts, or clear them.
    Ghost,
}

const PIE: [PieOp; 13] = [
    PieOp::Interp(Interp::Smooth),
    PieOp::Interp(Interp::Linear),
    PieOp::Interp(Interp::Hold),
    PieOp::Interp(Interp::Bezier),
    PieOp::Unify,
    PieOp::Break,
    PieOp::Vector,
    PieOp::Flatten,
    PieOp::Slide(Slide::Smooth),
    PieOp::Cycle,
    PieOp::Snap,
    PieOp::Frame,
    PieOp::SelectAll,
];
/// Same orbit as the graph's ops pie.
const PIE_R: f32 = 62.0;
/// Width of the channel list left of the plot.
const LIST_W: f32 = 180.0;
const ROW_H: f32 = 18.0;

pub struct CurveEditor {
    /// Kept for setups/handles that still poke it; the bottom bar's tab is
    /// what actually decides visibility now.
    #[allow(dead_code)]
    pub open: bool,
    /// Frame at the left edge, and points per frame.
    x_off: f32,
    x_scale: f32,
    /// SHOWN value at the vertical center, and points per shown unit (the
    /// shown value is the value itself unless the view is normalized).
    y_off: f32,
    y_scale: f32,
    drag: Option<Drag>,
    selection: HashSet<KeyId>,
    /// The key the typed fields edit — Blender's active keyframe: the last
    /// one clicked.
    active: Option<KeyId>,
    /// A modal G / S / R in progress.
    modal: Option<Modal>,
    /// A key slider in progress.
    slider: Option<SliderRun>,
    /// The click that confirmed a modal must not also start a gesture:
    /// presses are ignored until every button is up again.
    eat_press: bool,
    /// What scale and rotate turn about.
    pub pivot: Pivot,
    /// Copied keys, frames relative to the earliest: paste lands it on the
    /// playhead.
    clip: Vec<(u64, Channel, Key)>,
    /// NORMALIZE: every curve drawn into -1..1 so a 0.01 opacity and a
    /// 2000 px move read together. View only — the keys keep their values.
    pub normalize: bool,
    /// The per-curve fit of the last normalized paint; held still while a
    /// gesture runs, or the view would slide under the hand.
    norms: Norms,
    /// Frozen copies of curves to compare against (Blender's ghost curves).
    ghosts: Vec<Ghost>,
    /// What the last op did, in words (the read-out).
    note: String,
    /// Hidden / pinned channels and saved sets — SHARED with the dope
    /// sheet, which lists exactly what this editor shows.
    pub view: crate::chanview::ChannelView,
    /// FRAME SNAP: key moves, scales and inserts land on whole frames. Off
    /// (shift+S, the pie, the header), keys may sit between frames.
    pub snap: bool,
    /// The curve-ops pie: centre while its key is held, and its spring.
    pie: Option<Pos2>,
    pie_anim: (f32, f32),
    /// Hotkey strip along the plot's top (the viewer's `hud_keys` pref).
    pub show_keys: bool,
    keys_reveal: f32,
    /// Frame the view on the first paint that has curves to show.
    framed: bool,
    /// Last painted plot rect and track count (self-tests target these).
    pub last_rect: Rect,
    pub last_tracks: usize,
    /// Group-box handles from the last frame: (screen rect, shown anchor,
    /// scales x, scales y). Hit-tested before keys, same one-frame-stale
    /// scheme the node editor uses.
    pub box_handles: Vec<(Rect, (f32, f32), bool, bool)>,
}

impl Default for CurveEditor {
    fn default() -> Self {
        Self {
            open: false,
            x_off: 0.0,
            x_scale: 8.0,
            y_off: 0.0,
            y_scale: 60.0,
            drag: None,
            selection: HashSet::new(),
            active: None,
            modal: None,
            slider: None,
            eat_press: false,
            pivot: Pivot::default(),
            clip: Vec::new(),
            normalize: false,
            norms: Norms::new(),
            ghosts: Vec::new(),
            note: String::new(),
            view: Default::default(),
            snap: true,
            pie: None,
            pie_anim: (0.0, 0.0),
            show_keys: true,
            keys_reveal: 0.0,
            framed: false,
            last_rect: Rect::NOTHING,
            last_tracks: 0,
            box_handles: Vec::new(),
        }
    }
}

/// Channel display name. Indexed channels (spline points) name their index,
/// so the label has to be owned.
/// A channel's label given the node it is on. Field sockets are named by
/// the node's kind (an fBm's socket 5 is "gain", a Shape's is "feather"),
/// which a bare channel cannot know.
pub(crate) fn ch_label_for(kind: &engine::graph::NodeKind, ch: Channel) -> std::borrow::Cow<'static, str> {
    if let (engine::graph::NodeKind::Field(f), Channel::Field(i)) = (kind, ch) {
        return std::borrow::Cow::Borrowed(
            f.inputs().get(i as usize).map(|s| s.name).unwrap_or("socket"),
        );
    }
    if let (engine::graph::NodeKind::GeoNode(n), Channel::Field(i)) = (kind, ch) {
        return std::borrow::Cow::Borrowed(n.channel_name(i as usize));
    }
    if let (engine::graph::NodeKind::SdfNode(n), Channel::Field(i)) = (kind, ch) {
        return std::borrow::Cow::Borrowed(
            n.inputs().get(i as usize).map(|s| s.name).unwrap_or("value"),
        );
    }
    // A zone channel names its zone the way the panel's tab does —
    // "shadows lift r" — which only the node knows.
    if let (engine::graph::NodeKind::Grade(g), Some(i)) = (kind, zone_of(ch)) {
        if let Some(z) = &g.zones {
            let bare = ch_label(ch);
            let what = bare.split_once(' ').map(|(_, w)| w).unwrap_or(&bare);
            return std::borrow::Cow::Owned(format!("{} {what}", z.label(i as usize)));
        }
    }
    ch_label(ch)
}

/// A track's "node · channel" text — also the sort key of the shared
/// listing (`chanview::ChannelView::listed`), so both editors agree.
pub(crate) fn track_label(graph: &EditorGraph, node: u64, ch: Channel) -> String {
    graph
        .nodes
        .get(&node)
        .map(|n| format!("{} · {}", crate::editor::node_label(&n.kind), ch_label_for(&n.kind, ch)))
        .unwrap_or_default()
}

/// The zone a Grade zone channel belongs to.
fn zone_of(ch: Channel) -> Option<u16> {
    use Channel::*;
    match ch {
        ZoneLiftR(i) | ZoneLiftG(i) | ZoneLiftB(i) | ZoneGammaR(i) | ZoneGammaG(i)
        | ZoneGammaB(i) | ZoneGainR(i) | ZoneGainG(i) | ZoneGainB(i) | ZoneOffsetR(i)
        | ZoneOffsetG(i) | ZoneOffsetB(i) | ZoneSat(i) | ZoneContrast(i) | ZonePivot(i) => Some(i),
        _ => None,
    }
}

pub(crate) fn ch_label(ch: Channel) -> std::borrow::Cow<'static, str> {
    use Channel::*;
    if let Custom(i) | Shader(i) = ch {
        return std::borrow::Cow::Owned(crate::anim::interned(i));
    }
    match ch {
        TrackPtX(i) => return std::borrow::Cow::Owned(format!("track{i} x")),
        TrackPtY(i) => return std::borrow::Cow::Owned(format!("track{i} y")),
        TrackPinX(k) => return std::borrow::Cow::Owned(format!("pin{k} x")),
        TrackPinY(k) => return std::borrow::Cow::Owned(format!("pin{k} y")),
        _ => {}
    }
    if let Some((s, base)) = crate::anim::shape_base(ch) {
        return std::borrow::Cow::Owned(format!("shape {} {}", s + 1, ch_label(base)));
    }
    if let SplinePtX(i) | SplinePtY(i) | SplinePtF(i) | SplinePtB(i) = ch {
        let what = match ch {
            SplinePtX(_) => "x",
            SplinePtY(_) => "y",
            SplinePtB(_) => "feather bias",
            _ => "feather",
        };
        return std::borrow::Cow::Owned(format!("pt{i} {what}"));
    }
    if let KeyCenterL(i) | KeyCenterA(i) | KeyCenterB(i) | KeyTolL(i) | KeyTolC(i)
    | KeyTolH(i) | KeySoft(i) = ch
    {
        let what = match ch {
            KeyCenterL(_) => "lightness",
            KeyCenterA(_) => "a",
            KeyCenterB(_) => "b",
            KeyTolL(_) => "light tol",
            KeyTolC(_) => "sat tol",
            KeyTolH(_) => "hue tol",
            _ => "softness",
        };
        return std::borrow::Cow::Owned(format!("sample{} {what}", i + 1));
    }
    if let Some(l) = crate::nodes::gridwarp::point_label(ch) {
        return std::borrow::Cow::Owned(l);
    }
    if let Some(l) = crate::nodes::warp::point_label(ch) {
        return std::borrow::Cow::Owned(l);
    }
    if let Some(l) = crate::nodes::gradient::stop_label(ch) {
        return std::borrow::Cow::Owned(l);
    }
    if let RecombineGain(i) = ch {
        return std::borrow::Cow::Owned(format!("string {} gain", i + 1));
    }
    if let LayerTx(i) | LayerTy(i) | LayerRot(i) | LayerSx(i) | LayerSy(i)
    | LayerCx(i) | LayerCy(i) | LayerOpacity(i) = ch
    {
        let what = match ch {
            LayerTx(_) => "translate x",
            LayerTy(_) => "translate y",
            LayerRot(_) => "rotate",
            LayerSx(_) => "scale x",
            LayerSy(_) => "scale y",
            LayerCx(_) => "center x",
            LayerCy(_) => "center y",
            _ => "opacity",
        };
        return std::borrow::Cow::Owned(format!("L{} {what}", i + 1));
    }
    // Without the node there is no name to give a field socket.
    if let Field(i) = ch {
        return std::borrow::Cow::Owned(format!("socket {}", i + 1));
    }
    if let Some(i) = zone_of(ch) {
        let what = match ch {
            ZoneLiftR(_) => "lift r",
            ZoneLiftG(_) => "lift g",
            ZoneLiftB(_) => "lift b",
            ZoneGammaR(_) => "gamma r",
            ZoneGammaG(_) => "gamma g",
            ZoneGammaB(_) => "gamma b",
            ZoneGainR(_) => "gain r",
            ZoneGainG(_) => "gain g",
            ZoneGainB(_) => "gain b",
            ZoneOffsetR(_) => "offset r",
            ZoneOffsetG(_) => "offset g",
            ZoneOffsetB(_) => "offset b",
            ZoneContrast(_) => "contrast",
            ZonePivot(_) => "pivot",
            _ => "sat",
        };
        return std::borrow::Cow::Owned(format!("zone{} {what}", i + 1));
    }
    // Per-node labels live with the node.
    if let Some(l) = crate::nodes::label(ch) {
        return std::borrow::Cow::Borrowed(l);
    }
    std::borrow::Cow::Borrowed(match ch {
        Custom(_) | Shader(_) => unreachable!(),
        SplinePtX(_) | SplinePtY(_) | SplinePtF(_) | SplinePtB(_) | RotoPt(..) | RotoSh(..) => {
            unreachable!()
        }
        TrackPtX(_) | TrackPtY(_) | TrackPinX(_) | TrackPinY(_) => unreachable!(),
        ZoneLiftR(_) | ZoneLiftG(_) | ZoneLiftB(_) | ZoneGammaR(_) | ZoneGammaG(_)
        | ZoneGammaB(_) | ZoneGainR(_) | ZoneGainG(_) | ZoneGainB(_) | ZoneOffsetR(_)
        | ZoneOffsetG(_) | ZoneOffsetB(_) | ZoneSat(_) | ZoneContrast(_) | ZonePivot(_) => {
            unreachable!()
        }

        Field(_) => unreachable!(),
        LayerTx(_) | LayerTy(_) | LayerRot(_) | LayerSx(_) | LayerSy(_) => unreachable!(),
        LayerCx(_) | LayerCy(_) | LayerOpacity(_) => unreachable!(),
        RecombineGain(_) => unreachable!(),
        GradPos(_) | GradR(_) | GradG(_) | GradB(_) | GradA(_) => unreachable!(),
        GwSrcX(_) | GwSrcY(_) | GwDstX(_) | GwDstY(_) | GwAmount => unreachable!(),
        WarpPt(..) | WarpFalloff | WarpAmount => unreachable!(),
        KeyCenterL(_) | KeyCenterA(_) | KeyCenterB(_) => unreachable!(),
        KeyTolL(_) | KeyTolC(_) | KeyTolH(_) | KeySoft(_) => unreachable!(),
        TranslateX => "translate x",
        TranslateY => "translate y",
        Rotate => "rotate",
        ScaleX => "scale x",
        ScaleY => "scale y",
        CenterX => "center x",
        CenterY => "center y",
        ShearX => "shear x",
        ShearY => "shear y",
        Taper => "taper",
        TaperAngle => "taper angle",
        LiftR => "lift r",
        LiftG => "lift g",
        LiftB => "lift b",
        GammaR => "gamma r",
        GammaG => "gamma g",
        GammaB => "gamma b",
        GainR => "gain r",
        GainG => "gain g",
        GainB => "gain b",
        OffsetR => "offset r",
        OffsetG => "offset g",
        OffsetB => "offset b",
        DefocusRadius => "defocus radius",
        DefocusFocus => "focus distance",
        DefocusFStop => "f-stop",
        DefocusRotation => "iris rotation",
        DefocusCatseye => "cat's-eye",
        DefocusEdge => "edge",
        DefocusClamp => "highlight clamp",
        BlendMix => "blend mix",
        MatteMix => "matte mix",
        LumaLo => "luma lo",
        LumaHi => "luma hi",
        Pin0X => "pin0 x",
        Pin0Y => "pin0 y",
        Pin1X => "pin1 x",
        Pin1Y => "pin1 y",
        Pin2X => "pin2 x",
        Pin2Y => "pin2 y",
        Pin3X => "pin3 x",
        Pin3Y => "pin3 y",
        From0X => "from0 x",
        From0Y => "from0 y",
        From1X => "from1 x",
        From1Y => "from1 y",
        From2X => "from2 x",
        From2Y => "from2 y",
        From3X => "from3 x",
        From3Y => "from3 y",
        MixAmount => "mix",
        SplineTx => "shape x",
        SplineTy => "shape y",
        SplineRot => "shape rotate",
        SplineSx => "shape scale.x",
        SplineSy => "shape scale.y",
        SplineCx => "shape centre.x",
        SplineCy => "shape centre.y",
        SplineOpacity => "shape opacity",
        SplineExpand => "shape expand",
        SplineFalloff => "shape falloff",
        SplineFeatherOut => "shape feather out",
        SplineFeatherIn => "shape feather in",
        SatAmount | SatBoost | SatShadow | SatMid | SatHigh | SatPivot | SatWidth | SatPreserve
        | SatCompress => unreachable!(),
        EdgeSize | EdgeGain | EdgeThreshold | EdgeSoftness | MedianRadius | ConvolveMix => unreachable!(),
        HlrClip | HlrSoftness | HlrRadius | LtxClip | LtxSoftness => unreachable!(),
        StabSmooth | StabZoom => unreachable!(),
        BlurSize | DilateAmount | SoftCropL | SoftCropT | SoftCropR | SoftCropB
        | MotionBlurAngle | MotionBlurOffset | HueRotateAngle => unreachable!(),
        LutMix | LutLogMin | LutLogMax | LutDither | LutLimit => unreachable!(),
        TextX | TextY | TextSize | BurninSize | BurninMargin | BurninX | BurninY => unreachable!(),
        MirrorX | MirrorY | MirrorAngle => unreachable!(),
        LensK1 | LensK2 | LensK3 | LensP1 | LensP2 | LensCenterX | LensCenterY | LensSqueeze | LensZoom
        | LensCaR | LensCaB => unreachable!(),
        VectorBlurAmount | VectorBlurOffset | VectorWarpAmount | RifetimeFrame | FreezeFrame => unreachable!(),
        Cam3dTx | Cam3dTy | Cam3dTz | Cam3dRx | Cam3dRy | Cam3dRz | Cam3dFocal | Cam3dApW | Cam3dApH
        | Cam3dNear | Cam3dFar | Card3dTx | Card3dTy | Card3dTz | Card3dRx | Card3dRy | Card3dRz
        | Card3dSx | Card3dSy | Card3dSz | Card3dAspect | Card3dOpacity | Render3dShutter
        | Render3dOffset => unreachable!(),
        GammaNodeMaster | GammaNodeR | GammaNodeG | GammaNodeB | GammaNodeA | GammaNodeMix => unreachable!(),
        ToneExposure | ToneThreshold | ToneWhite | ToneSoftness | ToneToe | ToneToePivot | ToneMix => unreachable!(),
        DespillStrength | DespillBias | DespillRestore | LightwrapWidth | LightwrapBgBlur
        | LightwrapIntensity => unreachable!(),
        ScreenKeyR | ScreenKeyG | ScreenKeyB | ScreenKeyBalance | ScreenKeyGain | ScreenKeyClipBlack
        | ScreenKeyClipWhite | ScreenKeyDespill | ScreenKeyRestore => unreachable!(),
        GradAngle | GradCenterX | GradCenterY | GradScale => unreachable!(),
        GrainSizeR | GrainSizeG | GrainSizeB | GrainAmountR | GrainAmountG | GrainAmountB
        | GrainIrrR | GrainIrrG | GrainIrrB | GrainSoftness | GrainMono | GrainAspect
        | GrainBase | GrainShadows | GrainMids | GrainHighs | GrainSeed => unreachable!(),
        CompressBlackR | CompressBlackG | CompressBlackB | CompressWhiteR | CompressWhiteG
        | CompressWhiteB | CompressLiftR | CompressLiftG | CompressLiftB | CompressMultR
        | CompressMultG | CompressMultB => unreachable!(),
        SliceRedCenter | SliceRedHue | SliceRedSat | SliceRedDensity | SliceSkinCenter |
        SliceSkinHue | SliceSkinSat | SliceSkinDensity | SliceYellowCenter | SliceYellowHue |
        SliceYellowSat | SliceYellowDensity | SliceGreenCenter | SliceGreenHue |
        SliceGreenSat | SliceGreenDensity | SliceCyanCenter | SliceCyanHue | SliceCyanSat |
        SliceCyanDensity | SliceBlueCenter | SliceBlueHue | SliceBlueSat | SliceBlueDensity |
        SliceMagentaCenter | SliceMagentaHue | SliceMagentaSat | SliceMagentaDensity |
        SliceSat | SliceDensity | SliceSatDepth | SliceDenDepth |
        SliceSatBalance => unreachable!(),
        GlowSize | GlowFalloff | GlowThreshold | GlowKnee | GlowIntensity | GlowTintR | GlowTintG
        | GlowTintB => unreachable!(),
        GodraysX | GodraysY | GodraysLength | GodraysFalloff | GodraysThreshold | GodraysKnee
        | GodraysIntensity | GodraysTintR | GodraysTintG | GodraysTintB => unreachable!(),
        EdgeExtendDistance | EdgeExtendInset | EdgeExtendSoftness => unreachable!(),
        GlintThreshold | GlintKnee | GlintRays | GlintLength | GlintRotation | GlintFalloff
        | GlintIntensity | GlintTintR | GlintTintG | GlintTintB | GlintErode
        | GlintSpread | GlintShift => unreachable!(),
        GlintKernelRays | GlintKernelLength | GlintKernelThickness | GlintKernelFalloff
        | GlintKernelRotation | GlintKernelAlternate | GlintKernelCore | GlintKernelCoreSize
        | GlintKernelChroma | GlintKernelIntensity | GlintKernelTintR | GlintKernelTintG
        | GlintKernelTintB => unreachable!(),
        BokehKernelBlades | BokehKernelRoundness | BokehKernelRotation | BokehKernelRadius
        | BokehKernelSoftness | BokehKernelCatseye | BokehKernelCatseyeAngle | BokehKernelRing
        | BokehKernelFringe | BokehKernelSqueeze | BokehKernelIntensity => unreachable!(),
        DiffuseSize | DiffuseFalloff | DiffuseAmount | DiffuseHaze | DiffuseMistR | DiffuseMistG
        | DiffuseMistB | DiffuseNear | DiffuseFar => unreachable!(),
        BlendGainA => "gain +",
        BlendGainB => "gain −",
        LoopN => "loop iterations",
        GradeSat => "saturation",
        GradeContrast => "contrast",
        GradePivot => "contrast pivot",
        NoiseProbeX => "probe.x",
        NoiseProbeY => "probe.y",
        NoiseScale => "noise scale",
        NoiseSeed => "noise seed",
        NoiseSpeed => "noise speed",
        NoiseValue => "noise value",
        NoiseValueY => "noise value y",
        NoiseSeedOffset => "noise seed offset",
        GeoSeed => "geo seed",
        MultR => "mult.r",
        MultG => "mult.g",
        MultB => "mult.b",
        // The time lane's channel, so this exhaustive match keeps compiling.
        TimeRemapFrame => "source frame",
        SwitchWhich => "switch input",
        GridSpacingX => "spacing x",
        GridSpacingY => "spacing y",
        GridWidth => "line width",
        GridOffsetX => "offset x",
        GridOffsetY => "offset y",
        ReformatScale => "scale",
        TrackX => "track x",
        TrackY => "track y",
        PlanarDX => "planar dx",
        PlanarDY => "planar dy",
        PlanarRot => "planar rot",
        PlanarScale => "planar scale",
        PlanarPin0X => "planar pin 0 x",
        PlanarPin0Y => "planar pin 0 y",
        PlanarPin1X => "planar pin 1 x",
        PlanarPin1Y => "planar pin 1 y",
        PlanarPin2X => "planar pin 2 x",
        PlanarPin2Y => "planar pin 2 y",
        PlanarPin3X => "planar pin 3 x",
        PlanarPin3Y => "planar pin 3 y",
        KeyTilt => "key tilt",
        KeyLo => "matte lo",
        KeyHi => "matte hi",
        KeyGamma => "matte edge",
    })
}

/// Curve color: x/r lanes red, y/g green, b blue, scalars neutral — the same
/// channel-tint convention the grade HUD uses.
fn ch_color(ch: Channel) -> Color32 {
    let l = ch_label(ch);
    if l.ends_with(" x") || l.ends_with(" r") {
        Color32::from_rgb(0xd0, 0x5f, 0x55)
    } else if l.ends_with(" y") || l.ends_with(" g") {
        Color32::from_rgb(0x63, 0xb8, 0x62)
    } else if l.ends_with(" b") {
        Color32::from_rgb(0x62, 0x8a, 0xd8)
    } else {
        theme::curve_scalar()
    }
}

/// Grid step: the 1/2/5 x 10^n step nearest `target` data units.
fn nice_step(target: f32) -> f32 {
    if !target.is_finite() || target <= 0.0 {
        return 1.0;
    }
    let mag = 10f32.powf(target.log10().floor());
    let n = target / mag;
    mag * if n < 1.5 {
        1.0
    } else if n < 3.5 {
        2.0
    } else if n < 7.5 {
        5.0
    } else {
        10.0
    }
}

fn fmt_val(v: f32, step: f32) -> String {
    let dec = if step >= 10.0 {
        0
    } else if step >= 1.0 {
        1
    } else if step >= 0.1 {
        2
    } else {
        3
    };
    format!("{v:.dec$}")
}

impl CurveEditor {
    fn x_of(&self, rect: &Rect, frame: f32) -> f32 {
        rect.min.x + (frame - self.x_off) * self.x_scale
    }
    fn frame_at(&self, rect: &Rect, x: f32) -> f32 {
        self.x_off + (x - rect.min.x) / self.x_scale
    }
    /// Screen y of a SHOWN value.
    fn y_of(&self, rect: &Rect, v: f32) -> f32 {
        rect.center().y - (v - self.y_off) * self.y_scale
    }
    /// The SHOWN value at a screen y.
    fn value_at(&self, rect: &Rect, y: f32) -> f32 {
        self.y_off + (rect.center().y - y) / self.y_scale
    }

    /// A curve's normalisation, identity when the view is not normalized.
    fn norm(&self, n: u64, ch: Channel) -> (f32, f32) {
        self.norms.get(&(n, ch)).copied().unwrap_or((0.0, 1.0))
    }
    /// A value as shown.
    fn disp(&self, n: u64, ch: Channel, v: f32) -> f32 {
        let (m, h) = self.norm(n, ch);
        (v - m) / h
    }
    /// A shown value back to the curve's own units.
    fn undisp(&self, n: u64, ch: Channel, d: f32) -> f32 {
        let (m, h) = self.norm(n, ch);
        d * h + m
    }
    /// Screen y of a curve's value.
    fn yk(&self, rect: &Rect, n: u64, ch: Channel, v: f32) -> f32 {
        self.y_of(rect, self.disp(n, ch, v))
    }
    /// Points per value unit ON this curve.
    fn sy(&self, n: u64, ch: Channel) -> f32 {
        self.y_scale / self.norm(n, ch).1
    }

    /// Fit every listed curve into -1..1 (normalize on), or nothing.
    fn compute_norms(&mut self, tracks: &[Track], graph: &EditorGraph) {
        self.norms.clear();
        if !self.normalize {
            return;
        }
        for t in tracks {
            let Some(c) = graph.nodes.get(&t.node).and_then(|n| n.anim.get(&t.ch)) else { continue };
            let swing = c.noise.as_ref().map_or(0.0, |n| n.strength.abs() * 0.5);
            let (mut lo, mut hi) = (f32::MAX, f32::MIN);
            for k in &c.keys {
                lo = lo.min(k.value - swing);
                hi = hi.max(k.value + swing);
            }
            if lo > hi || !lo.is_finite() || !hi.is_finite() {
                continue;
            }
            // A flat curve sits on 0 rather than blowing up.
            let span = hi - lo;
            let fit = if span < 1e-9 { (lo, 1.0) } else { ((lo + hi) * 0.5, span * 0.5) };
            self.norms.insert((t.node, t.ch), fit);
        }
    }

    /// Fit the view to a box in (frame, shown value).
    fn frame_box(&mut self, rect: &Rect, f0: f32, f1: f32, v0: f32, v1: f32) {
        let fspan = (f1 - f0).max(1.0);
        let vspan = (v1 - v0).max(1e-3);
        self.x_scale = (rect.width() * 0.86 / fspan).clamp(0.02, 400.0);
        self.y_scale = (rect.height() * 0.72 / vspan).clamp(1e-4, 100_000.0);
        self.x_off = f0 - rect.width() * 0.07 / self.x_scale;
        self.y_off = (v0 + v1) * 0.5;
    }

    /// Fit both axes (independently) around the SHOWN curves (R, Home,
    /// a double-click): every key in time, and in value the keys AND the
    /// curve between them, so a bezier that overshoots its keys stays on
    /// screen. A channel hidden in the list does not count — its values
    /// would squash the curves you are looking at flat. With every channel
    /// hidden, all of them do.
    fn frame_all(&mut self, rect: &Rect, tracks: &[Track], graph: &EditorGraph) {
        let shown: Vec<&Track> =
            tracks.iter().filter(|t| !self.view.hidden.contains(&(t.node, t.ch))).collect();
        let shown = if shown.is_empty() { tracks.iter().collect() } else { shown };
        let (mut f0, mut f1) = (f32::MAX, f32::MIN);
        let (mut v0, mut v1) = (f32::MAX, f32::MIN);
        let mut any = false;
        for t in shown {
            let Some(c) = graph.nodes.get(&t.node).and_then(|n| n.anim.get(&t.ch)) else {
                continue;
            };
            // A noised curve swings ± strength / 2 about its keys.
            let swing = c.noise.as_ref().map_or(0.0, |n| n.strength.abs() * 0.5);
            let mut take = |v: f32| {
                v0 = v0.min(self.disp(t.node, t.ch, v - swing));
                v1 = v1.max(self.disp(t.node, t.ch, v + swing));
            };
            for k in &c.keys {
                any = true;
                f0 = f0.min(k.frame as f32);
                f1 = f1.max(k.frame as f32);
                take(k.value);
            }
            // The curve between its first and last key, sampled.
            if let (Some(a), Some(b)) = (c.keys.first(), c.keys.last()) {
                let (a, b) = (a.frame as f32, b.frame as f32);
                const STEPS: usize = 256;
                for i in 1..STEPS {
                    let f = a + (b - a) * i as f32 / STEPS as f32;
                    if let Some(v) = c.eval_base_at(f).filter(|v| v.is_finite()) {
                        take(v);
                    }
                }
            }
        }
        if any {
            // A single key, or keys on one frame, gets some room in time.
            let pad = if f1 - f0 < 1.0 { 10.0 } else { 0.0 };
            // A flat curve sits mid-height instead of zooming into nothing.
            let pad_v = if v1 - v0 < 1e-3 { v0.abs().max(1.0) * 0.5 } else { 0.0 };
            self.frame_box(rect, f0 - pad, f1 + pad, v0 - pad_v, v1 + pad_v);
        }
    }

    /// Fit the view around the selected keys (F).
    fn frame_selected(&mut self, rect: &Rect, graph: &EditorGraph) {
        let pts = self.selected_shown(graph);
        if pts.is_empty() {
            return;
        }
        let (mut f0, mut f1, mut v0, mut v1) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for (f, v) in pts {
            f0 = f0.min(f);
            f1 = f1.max(f);
            v0 = v0.min(v);
            v1 = v1.max(v);
        }
        // A single key gets some room around it.
        let pad_f = if f1 - f0 < 1.0 { 10.0 } else { 0.0 };
        let pad_v = if v1 - v0 < 1e-3 { v0.abs().max(1.0) * 0.5 } else { 0.0 };
        self.frame_box(rect, f0 - pad_f, f1 + pad_f, v0 - pad_v, v1 + pad_v);
    }

    /// The selected keys as (frame, shown value).
    fn selected_shown(&self, graph: &EditorGraph) -> Vec<(f32, f32)> {
        self.selection
            .iter()
            .filter_map(|(n, ch, f)| {
                let c = graph.nodes.get(n).and_then(|nd| nd.anim.get(ch))?;
                let f = id_frame(*f);
                let v = c.keys.iter().find(|k| k.frame == f)?.value;
                Some((f as f32, self.disp(*n, *ch, v)))
            })
            .collect()
    }

    /// The selected keys' frames, curve by curve (ordered, so ops run the
    /// same way every time).
    fn sel_by_curve(&self) -> BTreeMap<(u64, Channel), Vec<f64>> {
        let mut out: BTreeMap<(u64, Channel), Vec<f64>> = BTreeMap::new();
        for (n, ch, f) in &self.selection {
            out.entry((*n, *ch)).or_default().push(id_frame(*f));
        }
        out
    }

    /// Draw the editor and edit keys in place. `t` is the playhead frame.
    ///
    /// Keys are FRAME NUMBERS, so nothing here clamps them to a length: a
    /// key belongs at frame 1050 when the shot plays 1001..1069, and the
    /// old "clamp into the media's length" pinned exactly those keys to
    /// frame 68.
    pub fn ui(&mut self, ui: &mut egui::Ui, graph: &mut EditorGraph, sel: &[u64], t: u64) {
        // --- which curves: selected nodes if any, else everything animated,
        // plus whatever is PINNED in the list ---
        self.view.prune(graph);
        let chans = self.view.listed(graph, sel, |n, ch| track_label(graph, n, ch));
        let mut tracks: Vec<Track> = Vec::new();
        for &(id, ch) in &chans {
            let Some(n) = graph.nodes.get(&id) else { continue };
            if n.anim.get(&ch).is_none_or(|c| c.keys.is_empty()) {
                continue;
            }
            let short = ch_label_for(&n.kind, ch).into_owned();
            tracks.push(Track {
                node: id,
                ch,
                color: ch_color(ch),
                short,
            });
        }
        // Forget selected keys that no longer exist (undo, the dope sheet).
        self.selection.retain(|(n, ch, f)| {
            graph.nodes.get(n).and_then(|nd| nd.anim.get(ch)).is_some_and(|c| c.has_key(id_frame(*f)))
        });
        if self.active.is_some_and(|a| !self.selection.contains(&a)) {
            self.active = None;
        }
        self.ghosts.retain(|g| graph.nodes.contains_key(&g.node));
        let busy = self.drag.is_some() || self.modal.is_some() || self.slider.is_some();
        if !busy {
            self.compute_norms(&tracks, graph);
        }

        let over_panel = ui.rect_contains_pointer(ui.max_rect()) && !keymap::capturing();
        let typing = ui.ctx().egui_wants_keyboard_input();
        if over_panel && !typing && ui.input(|i| keymap::pressed(i, Act::CurvesSnap)) {
            self.snap = !self.snap;
        }

        // --- header: interpolation of the selection, modifiers, ops, view ---
        let mut op: Option<PieOp> = None;
        let mut out_only = false;
        let small = |t: &str| egui::RichText::new(t.to_string()).small();
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("curves").strong());
            self.view.sets_menu(ui, &chans);
            if tracks.is_empty() {
                ui.weak(if sel.is_empty() {
                    "nothing animated yet — key a parameter to see its curve"
                } else {
                    "no animated channels on the selection"
                });
                return;
            }
            // Interpolation of the selected keys: the shared mode reads as
            // active, a mixed selection shows none active.
            ui.separator();
            let has_sel = !self.selection.is_empty();
            let current = {
                let mut it = self.selection.iter().filter_map(|(n, ch, f)| {
                    graph
                        .nodes
                        .get(n)
                        .and_then(|nd| nd.anim.get(ch))
                        .and_then(|c| c.keys.iter().find(|k| k.frame == id_frame(*f)))
                        .map(|k| k.interp)
                });
                let first = it.next();
                match first {
                    Some(m) if it.all(|o| o == m) => Some(m),
                    _ => None,
                }
            };
            for (name, mode) in [
                ("hold", Interp::Hold),
                ("linear", Interp::Linear),
                ("smooth", Interp::Smooth),
                ("bezier", Interp::Bezier),
            ] {
                let active = current == Some(mode);
                let r = ui.add_enabled(has_sel, egui::Button::new(small(name)).selected(active));
                if r.on_hover_text(if has_sel {
                    "set the selected keys' interpolation, both sides\n\
                     shift-click: only the segment leaving each key"
                } else {
                    "select keys first"
                })
                .clicked()
                {
                    op = Some(PieOp::Interp(mode));
                    out_only = ui.input(|i| i.modifiers.shift);
                }
            }
            let eased = matches!(current, Some(Interp::Ease(..)));
            let ease_title = match current {
                Some(Interp::Ease(e, d)) => format!("{} {} ⏷", e.name(), d.name()),
                _ => "ease ⏷".to_string(),
            };
            let er = ui.add_enabled_ui(has_sel, |ui| {
                ui.menu_button(egui::RichText::new(ease_title).small().color(if eased {
                    theme::selection()
                } else {
                    theme::text()
                }), |ui| {
                    ui.set_min_width(200.0);
                    egui::Grid::new("curves-ease").spacing([4.0, 2.0]).show(ui, |ui| {
                        for e in Ease::ALL {
                            ui.label(small(e.name()));
                            for d in EaseDir::ALL {
                                let on = current == Some(Interp::Ease(e, d));
                                if ui.add(egui::Button::new(small(d.name())).selected(on)).clicked() {
                                    op = Some(PieOp::Interp(Interp::Ease(e, d)));
                                    out_only = true;
                                    ui.close();
                                }
                            }
                            ui.end_row();
                        }
                    });
                })
            });
            er.response.on_hover_text(
                "Blender's easing presets on the segment LEAVING each selected key — \
                 back overshoots, bounce and elastic ring",
            );
            ui.separator();
            if ui
                .add(egui::Button::new(small("snap")).selected(self.snap))
                .on_hover_text(format!(
                    "frame snap ({}) — off, keys can sit between frames",
                    keymap::label(ui.ctx(), Act::CurvesSnap)
                ))
                .clicked()
            {
                self.snap = !self.snap;
            }
            ui.separator();
            // Modifiers and extrapolation.
            let targets = self.noise_targets(&tracks);
            let noised = !targets.is_empty()
                && targets.iter().all(|(n, ch)| {
                    graph.nodes.get(n).and_then(|nd| nd.anim.get(ch)).is_some_and(|c| c.noise.is_some())
                });
            if ui
                .add_enabled(!targets.is_empty(), egui::Button::new(small("noise")).selected(noised))
                .on_hover_text(format!(
                    "Blender's Noise modifier ({}): a wiggle on top of the keys — \
                     on the curves with selected keys, or the one curve shown. \
                     Its settings sit under the curve in the list.",
                    keymap::label(ui.ctx(), Act::CurvesNoise)
                ))
                .clicked()
            {
                op = Some(PieOp::Noise);
            }
            let ctargets = self.curve_targets(&tracks);
            let curve = |n: &u64, ch: &Channel| graph.nodes.get(n).and_then(|nd| nd.anim.get(ch));
            let stepped = !ctargets.is_empty()
                && ctargets.iter().all(|(n, ch)| curve(n, ch).is_some_and(|c| c.stepped.is_some()));
            if ui
                .add(egui::Button::new(small("stepped")).selected(stepped))
                .on_hover_text(
                    "Blender's Stepped modifier: the curve is read every few frames and \
                     held between — animation on twos without re-keying. On the curves \
                     with selected keys, else every one shown; settings under the curve.",
                )
                .clicked()
            {
                op = Some(PieOp::Stepped);
            }
            let ext = |c: &AnimCurve| (c.pre, c.post);
            let shared = {
                let mut it = ctargets.iter().filter_map(|(n, ch)| curve(n, ch).map(ext));
                let first = it.next();
                first.filter(|f| it.all(|o| o == *f))
            };
            let ext_title = match shared {
                Some((a, b)) if a == b => format!("extend: {} ⏷", a.name()),
                Some((a, b)) => format!("extend: {} / {} ⏷", a.name(), b.name()),
                None => "extend: mixed ⏷".to_string(),
            };
            ui.menu_button(small(&ext_title), |ui| {
                ui.set_min_width(240.0);
                ui.weak("past the keys — on the curves with selected keys, else every one shown");
                egui::Grid::new("curves-extend").spacing([4.0, 2.0]).show(ui, |ui| {
                    for e in Extend::ALL {
                        ui.label(small(e.name()));
                        for (side, nm) in [(None, "both"), (Some(false), "before"), (Some(true), "after")] {
                            let on = match (shared, side) {
                                (Some((a, b)), None) => a == e && b == e,
                                (Some((a, _)), Some(false)) => a == e,
                                (Some((_, b)), Some(true)) => b == e,
                                _ => false,
                            };
                            if ui.add(egui::Button::new(small(nm)).selected(on)).clicked() {
                                op = Some(PieOp::Extend(side, e));
                                ui.close();
                            }
                        }
                        ui.end_row();
                    }
                });
                ui.weak(format!("{}: cycle on / off", keymap::label(ui.ctx(), Act::CurvesCycle)));
            })
            .response
            .on_hover_text("what the curves do before their first key and after their last");
            ui.separator();
            // Key operations.
            ui.menu_button(small("keys ⏷"), |ui| {
                ui.set_min_width(250.0);
                let item = |ui: &mut egui::Ui, label: &str, key: Option<Act>, on: bool| -> bool {
                    let text = match key {
                        Some(a) => format!("{label}   {}", keymap::label(ui.ctx(), a)),
                        None => label.to_string(),
                    };
                    ui.add_enabled(on, egui::Button::new(small(&text))).clicked()
                };
                ui.weak("sliders: move left / right, click keeps, Esc cancels");
                for s in Slide::ALL {
                    let key = (s == Slide::Smooth).then_some(Act::CurvesSmooth);
                    if item(ui, &format!("{}…", s.name()), key, has_sel) {
                        op = Some(PieOp::Slide(s));
                        ui.close();
                    }
                }
                ui.separator();
                for (label, key, o) in [
                    ("vector handles", None, PieOp::Vector),
                    ("unwrap rotation (±360°)", None, PieOp::Unwrap),
                    ("mirror in time (playhead)", Some(Act::CurvesMirror), PieOp::MirrorTime),
                    ("mirror in value (selection centre)", None, PieOp::MirrorValue),
                ] {
                    if item(ui, label, key, has_sel) {
                        op = Some(o);
                        ui.close();
                    }
                }
                ui.separator();
                for (label, key, o, on) in [
                    ("grab", Some(Act::CurvesGrab), PieOp::Xform(Xform::Grab), has_sel),
                    ("scale", Some(Act::CurvesScale), PieOp::Xform(Xform::Scale), has_sel),
                    ("rotate", Some(Act::CurvesRotate), PieOp::Xform(Xform::Rotate), has_sel),
                    ("copy", Some(Act::CurvesCopy), PieOp::Copy, has_sel),
                    ("paste at the playhead", Some(Act::CurvesPaste), PieOp::Paste, !self.clip.is_empty()),
                ] {
                    if item(ui, label, key, on) {
                        op = Some(o);
                        ui.close();
                    }
                }
                ui.separator();
                for (label, key, o) in [
                    ("select at / before the playhead", Act::CurvesSelectLeft, PieOp::SelectLeft),
                    ("select at / after the playhead", Act::CurvesSelectRight, PieOp::SelectRight),
                    ("select columns", Act::CurvesSelectColumn, PieOp::SelectColumn),
                    ("frame selected", Act::CurvesFrameSelected, PieOp::FrameSelected),
                ] {
                    if item(ui, label, Some(key), true) {
                        op = Some(o);
                        ui.close();
                    }
                }
            });
            egui::ComboBox::from_id_salt("curves-pivot")
                .selected_text(small(&format!("pivot: {}", self.pivot.name())))
                .show_ui(ui, |ui| {
                    for p in Pivot::ALL {
                        ui.selectable_value(&mut self.pivot, p, small(p.name()));
                    }
                })
                .response
                .on_hover_text("what scale (S) and rotate (shift+R) turn about");
            ui.separator();
            if ui
                .add(egui::Button::new(small("normalize")).selected(self.normalize))
                .on_hover_text(
                    "draw every curve into -1..1, so channels of very different sizes \
                     read together — view only, the values do not change",
                )
                .clicked()
            {
                self.normalize = !self.normalize;
                self.framed = false;
            }
            if ui
                .add(egui::Button::new(small("ghost")).selected(!self.ghosts.is_empty()))
                .on_hover_text(if self.ghosts.is_empty() {
                    "freeze a faint copy of the curves shown, to compare your edits against"
                } else {
                    "clear the ghost curves"
                })
                .clicked()
            {
                op = Some(PieOp::Ghost);
            }
            ui.separator();
            let shown = tracks.iter().filter(|t| !self.view.hidden.contains(&(t.node, t.ch))).count();
            if ui
                .add(egui::Button::new(small("all")).selected(shown == tracks.len()))
                .on_hover_text("show every listed channel")
                .clicked()
            {
                self.view.hidden.clear();
            }
            if ui
                .add(egui::Button::new(small("none")))
                .on_hover_text("hide them all, then pick the ones you want")
                .clicked()
            {
                self.view.hidden = tracks.iter().map(|t| (t.node, t.ch)).collect();
            }
        });
        if !tracks.is_empty() {
            self.key_fields(ui, graph);
        }

        let full = ui.available_rect_before_wrap();
        if full.width() < LIST_W + 32.0 || full.height() < 32.0 {
            return;
        }
        // The channel list on the left, the plot beside it.
        let list_rect = Rect::from_min_max(full.min, Pos2::new(full.min.x + LIST_W, full.max.y));
        let rect = Rect::from_min_max(Pos2::new(list_rect.max.x + 1.0, full.min.y), full.max);
        self.channel_list(ui, list_rect, &tracks, graph);

        // A FIXED id: an auto id would change with the widgets above (the
        // key row, a menu), and egui would drop a drag in flight.
        ui.advance_cursor_after_rect(rect);
        let resp = ui.interact(rect, ui.id().with("curves-plot"), egui::Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, theme::well());
        if tracks.is_empty() {
            return;
        }
        let visible: Vec<&Track> =
            tracks.iter().filter(|t| !self.view.hidden.contains(&(t.node, t.ch))).collect();
        self.last_rect = rect;
        self.last_tracks = visible.len();
        if !self.framed {
            self.framed = true;
            self.frame_all(&rect, &tracks, graph);
        }

        // --- navigation (decoupled axes) ---
        let pointer = resp.hover_pos().or_else(|| ui.ctx().pointer_latest_pos());
        let mods = ui.input(|i| i.modifiers);
        let over_plot = ui.rect_contains_pointer(rect) && !keymap::capturing();
        let gesture = self.drag.is_some() || self.modal.is_some() || self.slider.is_some();
        // Z / X held: the drag keeps to one axis (time / value).
        let (only_time, only_value) = if over_plot || gesture {
            ui.input(|i| (keymap::down(i, Act::CurvesOnlyTime), keymap::down(i, Act::CurvesOnlyValue)))
        } else {
            (false, false)
        };
        if resp.hovered() {
            // Blender's wheel rules (`crate::nav`): wheel zooms, trackpad
            // fingers pan, ctrl/shift+wheel pan by axis. Alt+wheel keeps
            // this editor's value-only zoom (Blender leaves alt+wheel
            // unused); the old shift+wheel time-zoom retired to Blender's
            // vertical pan — stretch one axis with ctrl+MMB instead.
            let w = crate::nav::wheel_nav(ui);
            self.x_off -= w.pan.x / self.x_scale;
            self.y_off += w.pan.y / self.y_scale;
            let factor = w.zoom;
            if (factor - 1.0).abs() > 1e-4 {
                if let Some(c) = pointer {
                    let (fx, fy) = if w.alt { (1.0, factor) } else { (factor, factor) };
                    let (fr, va) = (self.frame_at(&rect, c.x), self.value_at(&rect, c.y));
                    self.x_scale = (self.x_scale * fx).clamp(0.02, 400.0);
                    self.y_scale = (self.y_scale * fy).clamp(1e-4, 100_000.0);
                    self.x_off = fr - (c.x - rect.min.x) / self.x_scale;
                    self.y_off = va - (rect.center().y - c.y) / self.y_scale;
                }
            }
        }
        if (resp.double_clicked() && !gesture)
            || (over_plot && !typing && ui.input(|i| keymap::pressed(i, Act::CurvesFrame)))
        {
            self.frame_all(&rect, &tracks, graph);
        }

        // --- keys (only while the pointer is over the plot, nothing modal) ---
        let free = over_plot && !typing && !gesture && self.pie.is_none();
        if free {
            let has_sel = !self.selection.is_empty();
            let pressed = |a: Act| ui.input(|i| keymap::pressed(i, a));
            for (a, o, needs_sel) in [
                (Act::CurvesUnify, PieOp::Unify, true),
                (Act::CurvesBreak, PieOp::Break, true),
                (Act::CurvesNoise, PieOp::Noise, false),
                (Act::CurvesGrab, PieOp::Xform(Xform::Grab), true),
                (Act::CurvesScale, PieOp::Xform(Xform::Scale), true),
                (Act::CurvesRotate, PieOp::Xform(Xform::Rotate), true),
                (Act::CurvesCycle, PieOp::Cycle, false),
                (Act::CurvesSmooth, PieOp::Slide(Slide::Smooth), true),
                (Act::CurvesMirror, PieOp::MirrorTime, true),
                (Act::CurvesSelectLeft, PieOp::SelectLeft, false),
                (Act::CurvesSelectRight, PieOp::SelectRight, false),
                (Act::CurvesSelectColumn, PieOp::SelectColumn, false),
                (Act::CurvesFrameSelected, PieOp::FrameSelected, true),
                (Act::CurvesCopy, PieOp::Copy, true),
                (Act::CurvesPaste, PieOp::Paste, false),
            ] {
                if (has_sel || !needs_sel) && pressed(a) {
                    op = Some(o);
                }
            }
            // ⌘C / ⌘V arrive as egui's own events, never key presses.
            let (copy, paste) = ui.input(|i| {
                i.events.iter().fold((false, false), |acc, e| match e {
                    egui::Event::Copy => (true, acc.1),
                    egui::Event::Paste(_) => (acc.0, true),
                    _ => acc,
                })
            });
            if copy && has_sel {
                op = Some(PieOp::Copy);
            }
            if paste && !self.clip.is_empty() {
                op = Some(PieOp::Paste);
            }
        }

        // --- the curve-ops pie: held, release over a slice runs it ---
        if free && ui.input(|i| keymap::pressed(i, Act::CurvesPie)) {
            if let Some(p) = pointer {
                self.pie = Some(p);
                self.pie_anim = (0.0, 0.0);
            }
        }

        // --- modal transform / key slider: the pointer drives them ---
        let (confirm, cancel) = ui.input(|i| {
            (
                i.pointer.primary_pressed() || i.key_pressed(egui::Key::Enter),
                i.pointer.secondary_pressed() || i.key_pressed(egui::Key::Escape),
            )
        });
        if let Some(m) = self.modal.take() {
            if cancel {
                restore(graph, &m.snapshot);
                self.selection = m.grabbed.iter().map(|(n, ch, k)| (*n, *ch, frame_id(k.frame))).collect();
                self.note = "cancelled".into();
                self.eat_press = true;
            } else {
                if let Some(p) = pointer {
                    let moved = self.xformed(&rect, m.kind, m.pivot, &m.grabbed, m.origin, p, only_time, only_value);
                    let ids = apply_moved(graph, &m.snapshot, &m.grabbed, moved);
                    self.reselect(&m.grabbed, &ids);
                }
                if confirm {
                    self.eat_press = true;
                } else {
                    self.modal = Some(m);
                }
            }
        }
        if let Some(mut s) = self.slider.take() {
            if cancel {
                restore(graph, &s.snapshot);
                self.note = "cancelled".into();
                self.eat_press = true;
            } else {
                if let Some(p) = pointer {
                    let raw = (p.x - s.origin_x) / SLIDE_PX;
                    s.factor = if s.kind.two_sided() { raw.clamp(-1.0, 1.0) } else { raw.clamp(0.0, 1.0) };
                    restore(graph, &s.snapshot);
                    self.selection.clear();
                    for ((n, ch), frames, def) in &s.curves {
                        if let Some(c) = graph.nodes.get_mut(n).and_then(|nd| nd.anim.get_mut(ch)) {
                            for f in curveops::slide(c, frames, s.kind, s.factor, *def) {
                                self.selection.insert((*n, *ch, frame_id(f)));
                            }
                        }
                    }
                }
                if confirm {
                    self.note = format!("{} {:.0}%", s.kind.name(), s.factor * 100.0);
                    self.eat_press = true;
                } else {
                    self.slider = Some(s);
                }
            }
        }
        if self.eat_press && !ui.input(|i| i.pointer.any_down()) {
            self.eat_press = false;
        }
        let eat = self.eat_press || self.modal.is_some() || self.slider.is_some();

        // --- key hit map (screen positions of every visible key) ---
        let mut hits: Vec<(KeyId, Pos2)> = Vec::new();
        for tr in &visible {
            let Some(c) = graph.nodes.get(&tr.node).and_then(|n| n.anim.get(&tr.ch)) else {
                continue;
            };
            // A LOCKED channel takes no key edits: its keys are drawn but
            // cannot be hit, so nothing selects, drags, scales or deletes
            // them. Right-click its row in the list to unlock.
            if c.locked {
                continue;
            }
            for k in &c.keys {
                hits.push((
                    (tr.node, tr.ch, frame_id(k.frame)),
                    Pos2::new(self.x_of(&rect, k.frame as f32), self.yk(&rect, tr.node, tr.ch, k.value)),
                ));
            }
        }
        // Handles are only live on selected keys whose segment is bezier.
        let mut hhits: Vec<(HandleId, Pos2)> = Vec::new();
        for tr in &visible {
            let Some(c) = graph.nodes.get(&tr.node).and_then(|n| n.anim.get(&tr.ch)) else {
                continue;
            };
            let sy = self.sy(tr.node, tr.ch);
            for (i, k) in c.keys.iter().enumerate() {
                let id = frame_id(k.frame);
                if !self.selection.contains(&(tr.node, tr.ch, id)) {
                    continue;
                }
                let kp = Pos2::new(self.x_of(&rect, k.frame as f32), self.yk(&rect, tr.node, tr.ch, k.value));
                let off = |d: (f32, f32)| Pos2::new(kp.x + d.0 * self.x_scale, kp.y - d.1 * sy);
                if k.interp == Interp::Bezier {
                    hhits.push(((tr.node, tr.ch, id, true), off(c.out_handle(i))));
                }
                let prev_bezier = i
                    .checked_sub(1)
                    .and_then(|j| c.keys.get(j))
                    .is_some_and(|p| p.interp == Interp::Bezier);
                if prev_bezier {
                    hhits.push(((tr.node, tr.ch, id, false), off(c.in_handle(i))));
                }
            }
        }
        let hhit_at = |p: Pos2| -> Option<HandleId> {
            hhits
                .iter()
                .filter(|(_, sp)| (*sp - p).length() <= 7.0)
                .min_by(|a, b| (a.1 - p).length().total_cmp(&(b.1 - p).length()))
                .map(|(h, _)| *h)
        };
        let hit_at = |p: Pos2| -> Option<KeyId> {
            hits.iter()
                .filter(|(_, sp)| (*sp - p).length() <= 7.0)
                .min_by(|a, b| (a.1 - p).length().total_cmp(&(b.1 - p).length()))
                .map(|(k, _)| *k)
        };

        // --- drag state machine (the pie owns the pointer while it is up) ---
        let pie_up = self.pie.is_some();
        if !pie_up && !eat && resp.drag_started_by(egui::PointerButton::Secondary) && (mods.command || mods.alt) {
            self.drag = pointer.map(|p| Drag::Dolly { anchor: p });
        }
        // Blender's zoom chord: ctrl+MMB (and its alt+LMB emulation, below)
        // runs the SAME per-axis dolly — drag right to stretch time, up to
        // stretch value.
        if !pie_up && resp.drag_started_by(egui::PointerButton::Middle) && (mods.ctrl || mods.command) {
            self.drag = pointer.map(|p| Drag::Dolly { anchor: p });
        }
        if !pie_up && !eat && resp.drag_started_by(egui::PointerButton::Primary) {
            let origin = ui.input(|i| i.pointer.press_origin()).or(pointer);
            // Group box first: its handles sit outside the keys, so they can
            // never steal a key that is under the cursor.
            let box_hit = origin.and_then(|p| {
                self.box_handles.iter().find(|(r, ..)| r.contains(p)).copied()
            });
            let ladder = ui.input(|i| keymap::down(i, Act::CurvesLadder));
            let key_hit = origin.and_then(hit_at);
            self.drag = if mods.alt && (mods.ctrl || mods.command) {
                origin.map(|p| Drag::Dolly { anchor: p })
            } else if mods.command || mods.alt {
                // cmd+LMB is this editor's old pan; alt+LMB is Blender's
                // emulated MMB — same thing, and it outranks keys/handles.
                Some(Drag::Pan)
            } else if ladder && (key_hit.is_some() || !self.selection.is_empty()) {
                if let Some(k) = key_hit.filter(|k| !self.selection.contains(k)) {
                    self.selection.clear();
                    self.selection.insert(k);
                    self.active = Some(k);
                }
                let (snapshot, grabbed) = self.snapshot_selection(graph);
                origin.map(|p| Drag::Ladder { snapshot, grabbed, origin: p, rung: LADDER_HOME, lock_x: None })
            } else if let Some((_, anchor, ax, ay)) = box_hit {
                let (snapshot, grabbed) = self.snapshot_selection(graph);
                origin.map(|p| Drag::Scale { snapshot, grabbed, anchor, origin: p, ax, ay })
            } else if let Some(h) = origin.and_then(hhit_at) {
                Some(Drag::Handle(h))
            } else if let Some(k) = key_hit {
                if !self.selection.contains(&k) {
                    if !mods.shift {
                        self.selection.clear();
                    }
                    self.selection.insert(k);
                }
                self.active = Some(k);
                let (snapshot, grabbed) = self.snapshot_selection(graph);
                origin.map(|p| Drag::Keys { snapshot, grabbed, origin: p })
            } else {
                origin.map(|p| Drag::Marquee { start: p })
            };
        }
        if resp.dragged_by(egui::PointerButton::Middle) && !(mods.ctrl || mods.command) {
            self.x_off -= resp.drag_delta().x / self.x_scale;
            self.y_off += resp.drag_delta().y / self.y_scale;
        }

        let snap = self.snap;
        let snapped = |f: f64| if snap { f.round() } else { f }.max(0.0);
        let mut marquee: Option<Rect> = None;
        let mut drag = self.drag.take();
        match &mut drag {
            Some(Drag::Pan) if resp.dragged() => {
                self.x_off -= resp.drag_delta().x / self.x_scale;
                self.y_off += resp.drag_delta().y / self.y_scale;
            }
            Some(Drag::Dolly { anchor }) if resp.dragged() => {
                let a = *anchor;
                let d = resp.drag_delta();
                let (fr, va) = (self.frame_at(&rect, a.x), self.value_at(&rect, a.y));
                self.x_scale = (self.x_scale * (d.x * 0.005).exp()).clamp(0.02, 400.0);
                self.y_scale = (self.y_scale * (-d.y * 0.005).exp()).clamp(1e-4, 100_000.0);
                self.x_off = fr - (a.x - rect.min.x) / self.x_scale;
                self.y_off = va - (rect.center().y - a.y) / self.y_scale;
            }
            Some(Drag::Handle((n, ch, f, out))) if resp.dragged() => {
                let d = resp.drag_delta();
                let sy = self.sy(*n, *ch);
                let (mut dx, mut dy) = (d.x / self.x_scale, -d.y / sy);
                if only_time {
                    dy = 0.0;
                } else if only_value {
                    dx = 0.0;
                }
                let sx = self.x_scale;
                if let Some(c) = graph.nodes.get_mut(n).and_then(|nd| nd.anim.get_mut(ch)) {
                    if let Some(i) = c.index_of(id_frame(*f)) {
                        // A vector handle becomes a plain one where it is.
                        let (hi, ho) = (c.in_handle(i), c.out_handle(i));
                        let k = &mut c.keys[i];
                        if *out && k.vec_out {
                            k.tan_out = ho;
                            k.vec_out = false;
                        }
                        if !*out && k.vec_in {
                            k.tan_in = hi;
                            k.vec_in = false;
                        }
                        // Out handles stay to the right of the key, in
                        // handles to the left: x-monotonic, always solvable.
                        if *out {
                            k.tan_out.0 = (k.tan_out.0 + dx).max(0.0);
                            k.tan_out.1 += dy;
                        } else {
                            k.tan_in.0 = (k.tan_in.0 + dx).min(0.0);
                            k.tan_in.1 += dy;
                        }
                        // An unbroken tangent swings its other handle along.
                        if !k.broken {
                            swing(k, *out, sx, sy);
                        }
                    }
                }
            }
            Some(Drag::Scale { snapshot, grabbed, anchor, origin, ax, ay }) if resp.dragged() => {
                if let Some(p) = pointer {
                    // Scale about the opposite edge, 1:1 with the cursor.
                    let (a_f, a_v) = *anchor;
                    let f_now = self.frame_at(&rect, p.x) - a_f;
                    let f_was = self.frame_at(&rect, origin.x) - a_f;
                    let v_now = self.value_at(&rect, p.y) - a_v;
                    let v_was = self.value_at(&rect, origin.y) - a_v;
                    let (ax, ay) = (*ax && !only_value, *ay && !only_time);
                    let sf = if ax && f_was.abs() > 1e-4 { f_now / f_was } else { 1.0 };
                    let sv = if ay && v_was.abs() > 1e-4 { v_now / v_was } else { 1.0 };
                    let moved: Vec<(u64, Channel, Key)> = grabbed
                        .iter()
                        .map(|(n, ch, k)| {
                            // Unscaled time keeps its exact frame (a sub-frame
                            // key must not snap just because the box moved in v).
                            let frame = if ax {
                                snapped(a_f as f64 + (k.frame - a_f as f64) * sf as f64)
                            } else {
                                k.frame
                            };
                            let d = self.disp(*n, *ch, k.value);
                            let value = self.undisp(*n, *ch, a_v + (d - a_v) * sv);
                            let mut nk = Key { frame, value, ..*k };
                            nk.tan_in = (k.tan_in.0 * sf, k.tan_in.1 * sv);
                            nk.tan_out = (k.tan_out.0 * sf, k.tan_out.1 * sv);
                            (*n, *ch, nk)
                        })
                        .collect();
                    let ids = apply_moved(graph, snapshot, grabbed, moved);
                    self.reselect(grabbed, &ids);
                }
            }
            Some(Drag::Marquee { start }) if resp.dragged() => {
                if let Some(p) = pointer {
                    marquee = Some(Rect::from_two_pos(*start, p));
                }
            }
            Some(Drag::Keys { snapshot, grabbed, origin }) if resp.dragged() => {
                if let Some(p) = pointer {
                    // Z / X pin the move to one axis; shift locks it to
                    // whichever way you started pulling — the usual way to
                    // slide keys in time without nudging their values.
                    let (mut ot, mut ov) = (only_time, only_value);
                    if !ot && !ov && mods.shift {
                        let d = p - *origin;
                        if d.x.abs() >= d.y.abs() {
                            ot = true;
                        } else {
                            ov = true;
                        }
                    }
                    let moved = self.xformed(&rect, Xform::Grab, (0.0, 0.0), grabbed, *origin, p, ot, ov);
                    let ids = apply_moved(graph, snapshot, grabbed, moved);
                    self.reselect(grabbed, &ids);
                }
            }
            Some(Drag::Ladder { snapshot, grabbed, origin, rung, lock_x }) if resp.dragged() => {
                if let Some(p) = pointer {
                    if lock_x.is_none() {
                        // Up / down picks the rung until the hand moves sideways.
                        let r = LADDER_HOME as f32 + ((p.y - origin.y) / RUNG_H).round();
                        *rung = r.clamp(0.0, (LADDER.len() - 1) as f32) as usize;
                        if (p.x - origin.x).abs() > 6.0 {
                            *lock_x = Some(p.x);
                        }
                    }
                    if let Some(lx) = *lock_x {
                        let steps = ((p.x - lx) / LADDER_PX).trunc();
                        let delta = steps * LADDER[*rung];
                        restore(graph, snapshot);
                        for (n, ch, k) in grabbed.iter() {
                            if let Some(c) = graph.nodes.get_mut(n).and_then(|nd| nd.anim.get_mut(ch)) {
                                if let Some(i) = c.index_of(k.frame) {
                                    c.keys[i].value = k.value + delta;
                                }
                            }
                        }
                        self.note = format!("{:+}", round_to(delta, LADDER[*rung]));
                    }
                }
            }
            _ => {}
        }
        if resp.drag_stopped() {
            if let Some(Drag::Marquee { start }) = drag.take() {
                if let Some(p) = pointer {
                    let r = Rect::from_two_pos(start, p);
                    if !mods.shift {
                        self.selection.clear();
                    }
                    for (k, sp) in &hits {
                        if r.contains(*sp) {
                            self.selection.insert(*k);
                        }
                    }
                }
            }
            drag = None;
        }
        self.drag = drag;

        // --- click gestures: shift = insert, ctrl/right = delete ---
        if pie_up || eat {
            // Clicks belong to the pie, or to the modal that just ended.
        } else if resp.clicked() && mods.shift {
            if let Some(p) = pointer {
                if hit_at(p).is_none() {
                    // Insert on the nearest visible curve at this frame.
                    let f = snapped(self.frame_at(&rect, p.x) as f64);
                    let mut best: Option<(f32, u64, Channel)> = None;
                    for tr in &visible {
                        let Some(c) = graph.nodes.get(&tr.node).and_then(|n| n.anim.get(&tr.ch))
                        else {
                            continue;
                        };
                        if c.locked {
                            continue; // no keys onto a locked channel
                        }
                        if let Some(v) = c.eval_at(f as f32) {
                            let d = (self.yk(&rect, tr.node, tr.ch, v) - p.y).abs();
                            if best.is_none_or(|(bd, _, _)| d < bd) {
                                best = Some((d, tr.node, tr.ch));
                            }
                        }
                    }
                    if let Some((d, n, ch)) = best {
                        if d < 48.0 {
                            // The key goes on the curve the keys make, so
                            // inserting one never changes the shape.
                            let v = graph
                                .nodes
                                .get(&n)
                                .and_then(|nd| nd.anim.get(&ch))
                                .and_then(|c| c.eval_base_at(f as f32));
                            if let (Some(v), Some(c)) =
                                (v, graph.nodes.get_mut(&n).and_then(|nd| nd.anim.get_mut(&ch)))
                            {
                                c.set_key(f, v);
                                self.selection.clear();
                                self.selection.insert((n, ch, frame_id(f)));
                                self.active = Some((n, ch, frame_id(f)));
                            }
                        }
                    }
                }
            }
        // Alt-click removes, the same gesture the property curve editors
        // use; ctrl-click and right-click still do too.
        } else if (resp.clicked() && (mods.ctrl || mods.alt)) || resp.secondary_clicked() {
            if let Some(k) = pointer.and_then(hit_at) {
                self.remove_keys(graph, std::iter::once(k));
            }
        } else if resp.clicked() {
            match pointer.and_then(hit_at) {
                Some(k) => {
                    if !mods.shift {
                        self.selection.clear();
                    }
                    self.selection.insert(k);
                    self.active = Some(k);
                }
                None => self.selection.clear(),
            }
        }
        // Delete removes the selected keys, but only while the pointer is
        // over this panel — the node editor owns Delete everywhere else.
        if (resp.hovered() || ui.rect_contains_pointer(rect)) && !typing {
            let del = ui.input_mut(|i| {
                i.consume_key(egui::Modifiers::NONE, egui::Key::Delete)
                    || i.consume_key(egui::Modifiers::NONE, egui::Key::Backspace)
            });
            if del && !self.selection.is_empty() && !gesture {
                op = Some(PieOp::Delete);
            }
        }

        // --- grid ---
        let fstep = nice_step(90.0 / self.x_scale).max(1.0);
        let vstep = nice_step(46.0 / self.y_scale);
        let grid = Stroke::new(1.0, theme::grid_line());
        let axis = Stroke::new(1.0, theme::axis_line());
        let f_lo = (self.frame_at(&rect, rect.min.x) / fstep).floor() * fstep;
        let mut f = f_lo;
        while self.x_of(&rect, f) < rect.max.x {
            let x = self.x_of(&rect, f);
            if x >= rect.min.x {
                painter.line_segment([Pos2::new(x, rect.min.y), Pos2::new(x, rect.max.y)], grid);
                painter.text(
                    Pos2::new(x + 3.0, rect.max.y - 2.0),
                    egui::Align2::LEFT_BOTTOM,
                    format!("{}", f as i64),
                    egui::FontId::proportional(9.0),
                    theme::text_weak(),
                );
            }
            f += fstep;
        }
        let v_lo = (self.value_at(&rect, rect.max.y) / vstep).floor() * vstep;
        let mut v = v_lo;
        while self.y_of(&rect, v) > rect.min.y {
            let y = self.y_of(&rect, v);
            if y <= rect.max.y {
                let zero = v.abs() < vstep * 0.01;
                painter.line_segment(
                    [Pos2::new(rect.min.x, y), Pos2::new(rect.max.x, y)],
                    if zero { axis } else { grid },
                );
                painter.text(
                    Pos2::new(rect.min.x + 3.0, y - 2.0),
                    egui::Align2::LEFT_BOTTOM,
                    fmt_val(v, vstep),
                    egui::FontId::proportional(9.0),
                    theme::text_weak(),
                );
            }
            v += vstep;
        }

        // --- ghosts: frozen copies, faint, under everything live ---
        for g in &self.ghosts {
            if self.view.hidden.contains(&(g.node, g.ch)) {
                continue;
            }
            let pts = self.sampled(&rect, |t| g.curve.eval_at(t), g.node, g.ch, rect.min.x, rect.max.x);
            painter.add(egui::Shape::line(pts, Stroke::new(3.0, g.color.gamma_multiply(0.22))));
        }

        // --- curves + keys ---
        for tr in &visible {
            let Some(c) = graph.nodes.get(&tr.node).and_then(|n| n.anim.get(&tr.ch)) else {
                continue;
            };
            let (n, ch) = (tr.node, tr.ch);
            // A locked curve is drawn in the lock colour, not its channel's:
            // you can read it, you cannot touch it.
            let color = if c.locked { theme::locked() } else { tr.color };
            let kp = |k: &Key| Pos2::new(self.x_of(&rect, k.frame as f32), self.yk(&rect, n, ch, k.value));
            // Each segment is drawn the way it evaluates: steps for Hold,
            // straight for Linear, sampled for everything curved.
            if c.keys.len() >= 2 {
                let mut pts: Vec<Pos2> = Vec::new();
                for w in c.keys.windows(2) {
                    let (a, b) = (&w[0], &w[1]);
                    let pa = kp(a);
                    pts.push(pa);
                    match a.interp {
                        Interp::Hold => pts.push(Pos2::new(self.x_of(&rect, b.frame as f32), pa.y)),
                        Interp::Linear => {}
                        _ => {
                            let px = self.x_of(&rect, b.frame as f32) - pa.x;
                            // Bounce and elastic ring: sample them finely.
                            let fine = matches!(a.interp, Interp::Ease(..));
                            let n_s = (px.abs() / if fine { 2.0 } else { 5.0 })
                                .clamp(6.0, if fine { 400.0 } else { 64.0 }) as usize;
                            let span = (b.frame - a.frame) as f32;
                            for j in 1..n_s {
                                let t = a.frame as f32 + span * (j as f32 / n_s as f32);
                                if let Some(v) = c.eval_base_at(t) {
                                    pts.push(Pos2::new(self.x_of(&rect, t), self.yk(&rect, n, ch, v)));
                                }
                            }
                        }
                    }
                }
                if let Some(l) = c.keys.last() {
                    pts.push(kp(l));
                }
                // Under a modifier the keys' own curve is the faint one; the
                // modified curve below is what the channel reads.
                let base = if c.modified() { color.gamma_multiply(0.35) } else { color };
                painter.add(egui::Shape::line(pts, Stroke::new(1.6, base)));
            }
            // Past the keys: hold tails, or the extrapolation sampled.
            if let (Some(k0), Some(k1)) = (c.keys.first(), c.keys.last()) {
                let tail = Stroke::new(1.0, color.gamma_multiply(if c.modified() { 0.25 } else { 0.5 }));
                let x0 = self.x_of(&rect, k0.frame as f32);
                let x1 = self.x_of(&rect, k1.frame as f32);
                if c.pre == Extend::Hold {
                    let y = kp(k0).y;
                    painter.line_segment([Pos2::new(rect.min.x, y), Pos2::new(x0, y)], tail);
                } else if x0 > rect.min.x {
                    let pts = self.sampled(&rect, |t| c.eval_base_at(t), n, ch, rect.min.x, x0);
                    painter.add(egui::Shape::line(pts, tail));
                }
                if c.post == Extend::Hold {
                    let y = kp(k1).y;
                    painter.line_segment([Pos2::new(x1, y), Pos2::new(rect.max.x, y)], tail);
                } else if x1 < rect.max.x {
                    let pts = self.sampled(&rect, |t| c.eval_base_at(t), n, ch, x1, rect.max.x);
                    painter.add(egui::Shape::line(pts, tail));
                }
            }
            if c.modified() {
                // What the channel READS (noise, steps), across the whole
                // view: a modifier runs before the first key and after the
                // last as well.
                let pts = self.sampled(&rect, |t| c.eval_at(t), n, ch, rect.min.x, rect.max.x);
                painter.add(egui::Shape::line(pts, Stroke::new(1.4, color)));
            }
            // Bezier handles on selected keys.
            for (h, hp) in &hhits {
                if h.0 != n || h.1 != ch {
                    continue;
                }
                let Some(k) = c.keys.iter().find(|k| k.frame == id_frame(h.2)) else { continue };
                let kpos = kp(k);
                let hot = pointer.is_some_and(|q| (q - *hp).length() <= 7.0);
                painter.line_segment([kpos, *hp], Stroke::new(1.0, color.gamma_multiply(0.7)));
                // Square ends = broken tangent, round = aligned, hollow =
                // vector (it follows its neighbour).
                let r = if hot { 4.5 } else { 3.0 };
                let vector = if h.3 { k.vec_out } else { k.vec_in };
                if vector {
                    painter.rect_stroke(
                        Rect::from_center_size(*hp, egui::vec2(r * 2.0, r * 2.0)),
                        0.0,
                        Stroke::new(1.4, color),
                        egui::StrokeKind::Middle,
                    );
                } else if k.broken {
                    painter.rect_filled(Rect::from_center_size(*hp, egui::vec2(r * 1.8, r * 1.8)), 0.0, color);
                } else {
                    painter.circle_filled(*hp, r, color);
                }
            }
            for k in &c.keys {
                let p = kp(k);
                if !rect.expand(8.0).contains(p) {
                    continue;
                }
                let id = (n, ch, frame_id(k.frame));
                let sel = self.selection.contains(&id);
                let hot = pointer.is_some_and(|q| (q - p).length() <= 7.0);
                let fill = if sel { theme::selection() } else { color };
                if sel || hot {
                    painter.circle_filled(p, 5.5, theme::handle_hot());
                }
                if self.active == Some(id) {
                    painter.circle_stroke(p, 7.5, Stroke::new(1.2, theme::selection()));
                }
                match k.interp {
                    // Square = stepped, diamond = curved, dot = linear.
                    Interp::Hold => {
                        painter.rect_filled(Rect::from_center_size(p, egui::vec2(6.0, 6.0)), 0.0, fill);
                    }
                    Interp::Linear => {
                        painter.circle_filled(p, 3.5, fill);
                    }
                    _ => {
                        let d = 4.6;
                        painter.add(egui::Shape::convex_polygon(
                            vec![
                                Pos2::new(p.x, p.y - d),
                                Pos2::new(p.x + d, p.y),
                                Pos2::new(p.x, p.y + d),
                                Pos2::new(p.x - d, p.y),
                            ],
                            fill,
                            Stroke::NONE,
                        ));
                    }
                }
            }
        }

        // --- group box: scale a whole selection in time and value ---
        self.box_handles.clear();
        let sel_pts = self.selected_shown(graph);
        let scaling = matches!(self.drag, Some(Drag::Scale { .. }));
        let moving = matches!(self.drag, Some(Drag::Keys { .. } | Drag::Ladder { .. }))
            || self.modal.is_some()
            || self.slider.is_some();
        if sel_pts.len() >= 2 && !moving {
            let (mut f0, mut f1) = (f32::MAX, f32::MIN);
            let (mut v0, mut v1) = (f32::MAX, f32::MIN);
            for (f, v) in &sel_pts {
                f0 = f0.min(*f);
                f1 = f1.max(*f);
                v0 = v0.min(*v);
                v1 = v1.max(*v);
            }
            // A flat or single-frame selection still gets a grabbable box.
            // Clamped into the plot: a selection reaching past the edge
            // would otherwise hang its handles somewhere unclickable. The
            // scale maths works in SHOWN space off f0/f1/v0/v1, so pinning
            // the drawn box to the view changes where you grab, never the result.
            let bx = Rect::from_two_pos(
                Pos2::new(self.x_of(&rect, f0), self.y_of(&rect, v1)),
                Pos2::new(self.x_of(&rect, f1), self.y_of(&rect, v0)),
            )
            .expand(9.0)
            .intersect(rect.shrink(4.0));
            let bx = if bx.width() > 2.0 && bx.height() > 2.0 { bx } else { rect.shrink(4.0) };
            painter.rect_stroke(
                bx,
                0.0,
                Stroke::new(1.0, if scaling { theme::handle() } else { theme::selection_box() }),
                egui::StrokeKind::Outside,
            );
            // Handles on the edges and corners; each scales about the
            // opposite side, so the far edge stays pinned.
            for (hx, hy) in [(-1i8, -1i8), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)] {
                let hp = Pos2::new(
                    bx.center().x + hx as f32 * bx.width() * 0.5,
                    bx.center().y + hy as f32 * bx.height() * 0.5,
                );
                // The anchor is the OPPOSITE edge, in shown space.
                let anchor = (
                    if hx > 0 { f0 } else if hx < 0 { f1 } else { f0 },
                    if hy > 0 { v1 } else if hy < 0 { v0 } else { v0 },
                );
                painter.rect_filled(Rect::from_center_size(hp, egui::vec2(7.0, 7.0)), 1.0, theme::handle());
                self.box_handles.push((
                    Rect::from_center_size(hp, egui::vec2(13.0, 13.0)),
                    anchor,
                    hx != 0,
                    hy != 0,
                ));
            }
        }

        // --- playhead ---
        let px = self.x_of(&rect, t as f32);
        if px >= rect.min.x && px <= rect.max.x {
            painter.line_segment(
                [Pos2::new(px, rect.min.y), Pos2::new(px, rect.max.y)],
                Stroke::new(1.0, theme::timeline_playhead().gamma_multiply(0.8)),
            );
        }
        if let Some(m) = marquee {
            painter.rect_filled(m, 0.0, theme::marquee_fill());
            painter.rect_stroke(m, 0.0, Stroke::new(1.0, theme::marquee()), egui::StrokeKind::Outside);
        }
        // The ladder: rungs at the press, the live one lit.
        if let Some(Drag::Ladder { origin, rung, lock_x, .. }) = &self.drag {
            let x = origin.x + 18.0;
            for (i, step) in LADDER.iter().enumerate() {
                let y = origin.y + (i as f32 - LADDER_HOME as f32) * RUNG_H;
                let r = Rect::from_min_size(Pos2::new(x, y - RUNG_H * 0.5 + 1.0), egui::vec2(54.0, RUNG_H - 2.0));
                let on = i == *rung;
                let col = if on { theme::text_hot() } else { theme::text_weak() };
                if lock_x.is_none() || on {
                    painter.rect(
                        r,
                        3.0,
                        theme::graph_bg(),
                        Stroke::new(if on { 1.5 } else { 1.0 }, col.gamma_multiply(0.8)),
                        egui::StrokeKind::Inside,
                    );
                    painter.text(
                        r.center(),
                        egui::Align2::CENTER_CENTER,
                        fmt_step(*step),
                        egui::FontId::proportional(11.0),
                        col,
                    );
                }
            }
        }
        // The slider's gauge, along the plot's top.
        if let Some(s) = &self.slider {
            let w = 260.0f32.min(rect.width() - 20.0);
            let bar = Rect::from_center_size(Pos2::new(rect.center().x, rect.min.y + 30.0), egui::vec2(w, 6.0));
            painter.rect_filled(bar.expand2(egui::vec2(8.0, 14.0)), 4.0, theme::graph_bg().gamma_multiply(0.9));
            painter.rect_filled(bar, 3.0, theme::grid_line());
            let (lo, span) = if s.kind.two_sided() { (-1.0, 2.0) } else { (0.0, 1.0) };
            let kx = bar.min.x + (s.factor - lo) / span * bar.width();
            if s.kind.two_sided() {
                painter.line_segment(
                    [Pos2::new(bar.center().x, bar.min.y - 3.0), Pos2::new(bar.center().x, bar.max.y + 3.0)],
                    Stroke::new(1.0, theme::text_weak()),
                );
            }
            painter.circle_filled(Pos2::new(kx, bar.center().y), 6.0, theme::selection());
            painter.text(
                Pos2::new(bar.center().x, bar.min.y - 4.0),
                egui::Align2::CENTER_BOTTOM,
                format!("{}  {:.0}%   click keeps · Esc cancels", s.kind.name(), s.factor * 100.0),
                egui::FontId::proportional(11.0),
                theme::text(),
            );
        }
        // Read-out: the op in progress, how many keys are selected, and the
        // hover position.
        let mut status = String::new();
        if let Some(m) = &self.modal {
            status = format!(
                "{} — click / Enter keeps, right-click / Esc cancels, Z / X one axis",
                match m.kind {
                    Xform::Grab => "grab",
                    Xform::Scale => "scale",
                    Xform::Rotate => "rotate",
                }
            );
        } else if !self.selection.is_empty() {
            status = format!("{} key(s) selected", self.selection.len());
        }
        if !self.note.is_empty() {
            status = format!("{status}   {}", self.note);
        }
        if !self.snap {
            status = format!("{status}   snap off");
        }
        if self.normalize {
            status = format!("{status}   normalized");
        }
        if let Some(p) = pointer.filter(|p| rect.contains(*p)) {
            let f = self.frame_at(&rect, p.x);
            let f = if self.snap { format!("{}", f.round() as i64) } else { format!("{f:.2}") };
            let val = self.value_at(&rect, p.y);
            status = format!("{status}   f {f}  v {}", fmt_val(val, vstep));
        }
        if !status.is_empty() {
            painter.text(
                Pos2::new(rect.max.x - 6.0, rect.min.y + 2.0),
                egui::Align2::RIGHT_TOP,
                status.trim_start(),
                egui::FontId::proportional(10.0),
                theme::text(),
            );
        }

        // --- hotkey strip: caps along the plot's top edge, words fanning out
        // while the hand hovers them — the same strip the graph and viewer
        // carry (and the same pref hides it).
        if self.show_keys && self.slider.is_none() {
            let ctx = ui.ctx().clone();
            let item = |a: Act, label: &str| {
                (
                    keymap::label(&ctx, a).replace("Backtick", "`"),
                    label.to_string(),
                    ui.input(|i| keymap::down(i, a)),
                )
            };
            let items = vec![
                item(Act::CurvesPie, "hold: curve-ops pie"),
                item(Act::CurvesGrab, "grab"),
                item(Act::CurvesScale, "scale"),
                item(Act::CurvesRotate, "rotate"),
                item(Act::CurvesLadder, "hold + drag: ladder (nudge in steps)"),
                item(Act::CurvesOnlyTime, "hold + drag: time only"),
                item(Act::CurvesOnlyValue, "hold + drag: value only"),
                item(Act::CurvesUnify, "tangent (align handles)"),
                item(Act::CurvesBreak, "break tangent"),
                item(Act::CurvesCycle, "cycle on / off"),
                item(Act::CurvesSmooth, "smooth keys"),
                item(Act::CurvesNoise, "noise on / off"),
                item(Act::CurvesSnap, if self.snap { "frame snap: on" } else { "frame snap: off" }),
                item(Act::CurvesSelectLeft, "select before the playhead"),
                item(Act::CurvesSelectRight, "select after the playhead"),
                item(Act::CurvesSelectColumn, "select columns"),
                item(Act::CurvesFrameSelected, "frame selected"),
                item(Act::CurvesFrame, "frame all"),
                ("shift-click".to_string(), "add a key on the nearest curve".to_string(), mods.shift),
                ("alt-click".to_string(), "delete a key".to_string(), mods.alt),
                ("shift-drag".to_string(), "lock to the axis you pull".to_string(), false),
            ];
            let strip = crate::hudkeys::TopStrip::new(&painter, rect, &items);
            let hot = self.drag.is_none()
                && pointer.is_some_and(|p| {
                    strip.row().expand(6.0).contains(p)
                        || (self.keys_reveal > 0.3 && strip.fan(&painter).contains(p))
                });
            let target = if hot { 1.0 } else { 0.0 };
            let dt = ui.input(|i| i.stable_dt).min(0.1);
            self.keys_reveal += (target - self.keys_reveal) * (1.0 - (-dt * 12.0).exp());
            if (target - self.keys_reveal).abs() > 0.004 {
                ui.ctx().request_repaint();
            } else {
                self.keys_reveal = target;
            }
            strip.draw_caps(&painter);
            strip.draw_words(&painter, self.keys_reveal);
        }

        // --- the pie: an orbit of labels, spring-opened ---
        if let Some(c) = self.pie {
            let dt = ui.input(|i| i.stable_dt).min(0.1);
            let (mut s, mut v) = self.pie_anim;
            let acc = (1.0 - s) * 260.0 - v * 10.0;
            v += acc * dt;
            s += v * dt;
            self.pie_anim = (s, v);
            if (1.0 - s).abs() > 1e-3 || v.abs() > 1e-3 {
                ui.ctx().request_repaint();
            }
            let ring = PIE_R * s.max(0.0);
            let n = PIE.len();
            let hot = pointer.and_then(|p| crate::regions::pie_pick(c, p, n));
            let has_sel = !self.selection.is_empty();
            painter.circle_filled(c, 2.0, theme::node_label());
            for (i, o) in PIE.iter().enumerate() {
                let at = crate::regions::pie_slice_pos(c, i, n, ring);
                let is_hot = hot == Some(i);
                let live = has_sel || matches!(o, PieOp::Snap | PieOp::Frame | PieOp::SelectAll | PieOp::Cycle);
                let col = if is_hot {
                    theme::text_hot()
                } else if live {
                    theme::text()
                } else {
                    theme::text_weak()
                };
                let galley = painter.layout_no_wrap(self.pie_label(*o), egui::FontId::proportional(12.0), col);
                let r = Rect::from_center_size(at, galley.size() + egui::vec2(10.0, 4.0));
                painter.rect(
                    r,
                    4.0,
                    theme::graph_bg(),
                    Stroke::new(if is_hot { 2.0 } else { 1.0 }, col.gamma_multiply(0.7)),
                    egui::StrokeKind::Outside,
                );
                painter.galley(r.center() - galley.size() * 0.5, galley, col);
            }
            let released = ui.input(|i| keymap::released(i, Act::CurvesPie));
            let held = ui.input(|i| keymap::down(i, Act::CurvesPie));
            if released || !held {
                if released {
                    if let Some(i) = hot {
                        op = Some(PIE[i]);
                        out_only = mods.shift;
                    }
                }
                self.pie = None;
            }
        }
        if self.modal.is_some() || self.slider.is_some() {
            ui.ctx().request_repaint();
        }

        if let Some(o) = op {
            self.run(o, out_only, graph, &rect, &tracks, &hits, t, pointer);
        }
    }

    /// A curve sampled every 2 px between two screen xs.
    fn sampled(
        &self,
        rect: &Rect,
        f: impl Fn(f32) -> Option<f32>,
        n: u64,
        ch: Channel,
        xa: f32,
        xb: f32,
    ) -> Vec<Pos2> {
        let (fa, fb) = (self.frame_at(rect, xa), self.frame_at(rect, xb));
        let step = (2.0 / self.x_scale).min(0.5).max(1e-3);
        let cnt = (((fb - fa) / step).ceil().max(1.0) as usize).min(4000);
        (0..=cnt)
            .filter_map(|j| {
                let t = fa + (fb - fa) * j as f32 / cnt as f32;
                f(t).map(|v| Pos2::new(self.x_of(rect, t), self.yk(rect, n, ch, v)))
            })
            .collect()
    }

    /// Where the grabbed keys go for a transform with the pointer at `p`.
    /// Pure in the editor's view: (frame, shown value) space for keys,
    /// screen space for rotation, so a turn looks like a turn however the
    /// axes are scaled.
    #[allow(clippy::too_many_arguments)]
    fn xformed(
        &self,
        rect: &Rect,
        kind: Xform,
        pivot: (f32, f32),
        grabbed: &Grabbed,
        origin: Pos2,
        p: Pos2,
        only_time: bool,
        only_value: bool,
    ) -> Vec<(u64, Channel, Key)> {
        let snap = self.snap;
        let snapf = |f: f64| if snap { f.round() } else { f };
        let pv = Pos2::new(self.x_of(rect, pivot.0), self.y_of(rect, pivot.1));
        let each = self.pivot == Pivot::Each;
        match kind {
            Xform::Grab => {
                let raw = (self.frame_at(rect, p.x) - self.frame_at(rect, origin.x)) as f64;
                // Snapped, a move is whole frames — a sub-frame key keeps
                // its fraction.
                let mut df = if snap { raw.round() } else { raw };
                let mut dd = self.value_at(rect, p.y) - self.value_at(rect, origin.y);
                if only_time {
                    dd = 0.0;
                } else if only_value {
                    df = 0.0;
                }
                grabbed
                    .iter()
                    .map(|(n, ch, k)| {
                        let value = self.undisp(*n, *ch, self.disp(*n, *ch, k.value) + dd);
                        (*n, *ch, Key { frame: k.frame + df, value, ..*k })
                    })
                    .collect()
            }
            Xform::Scale => {
                let (a, b) = (origin - pv, p - pv);
                let s = if a.length() > 1.0 { b.length() / a.length() } else { 1.0 };
                let (sx, sv) = (if only_value { 1.0 } else { s }, if only_time { 1.0 } else { s });
                grabbed
                    .iter()
                    .map(|(n, ch, k)| {
                        let mut nk = *k;
                        nk.tan_in = (k.tan_in.0 * sx, k.tan_in.1 * sv);
                        nk.tan_out = (k.tan_out.0 * sx, k.tan_out.1 * sv);
                        if !each {
                            nk.frame = snapf(pivot.0 as f64 + (k.frame - pivot.0 as f64) * sx as f64);
                            let d = self.disp(*n, *ch, k.value);
                            nk.value = self.undisp(*n, *ch, pivot.1 + (d - pivot.1) * sv);
                        }
                        (*n, *ch, nk)
                    })
                    .collect()
            }
            Xform::Rotate => {
                let ang = (p - pv).angle() - (origin - pv).angle();
                let (s, c) = ang.sin_cos();
                let rot = |v: Vec2| Vec2::new(v.x * c - v.y * s, v.x * s + v.y * c);
                grabbed
                    .iter()
                    .map(|(n, ch, k)| {
                        let (sx, sy) = (self.x_scale, self.sy(*n, *ch));
                        let to_s = |t: (f32, f32)| Vec2::new(t.0 * sx, -t.1 * sy);
                        let back = |v: Vec2| (v.x / sx, -v.y / sy);
                        let mut nk = *k;
                        if !k.vec_out {
                            let o = back(rot(to_s(k.tan_out)));
                            nk.tan_out = (o.0.max(0.0), o.1);
                        }
                        if !k.vec_in {
                            let i = back(rot(to_s(k.tan_in)));
                            nk.tan_in = (i.0.min(0.0), i.1);
                        }
                        if !each {
                            let kp = Pos2::new(self.x_of(rect, k.frame as f32), self.yk(rect, *n, *ch, k.value));
                            let q = pv + rot(kp - pv);
                            nk.frame = snapf(self.frame_at(rect, q.x) as f64);
                            nk.value = self.undisp(*n, *ch, self.value_at(rect, q.y));
                        }
                        (*n, *ch, nk)
                    })
                    .collect()
            }
        }
    }

    /// After keys moved: the selection is where they landed, and the active
    /// key follows its own key.
    fn reselect(&mut self, grabbed: &Grabbed, ids: &[KeyId]) {
        let was = self.active;
        self.selection = ids.iter().copied().collect();
        self.active = None;
        for ((n, ch, k), id) in grabbed.iter().zip(ids) {
            if was == Some((*n, *ch, frame_id(k.frame))) {
                self.active = Some(*id);
            }
        }
    }

    /// The active key's numbers, typed (Blender's sidebar F-Curve panel):
    /// frame, value and — where its segments are bezier — the handles.
    ///
    /// The row is ALWAYS there (saying how to get an active key when there
    /// is none), so clicking a key never shifts the plot under the hand.
    fn key_fields(&mut self, ui: &mut egui::Ui, graph: &mut EditorGraph) {
        let small = |t: &str| egui::RichText::new(t.to_string()).small();
        let found = self.active.and_then(|(n, ch, fid)| {
            let c = graph.nodes.get(&n).and_then(|nd| nd.anim.get(&ch))?;
            let i = c.index_of(id_frame(fid))?;
            (!c.locked).then_some((n, ch, fid, i))
        });
        let Some((n, ch, fid, i)) = found else {
            ui.horizontal(|ui| {
                ui.set_min_height(ROW_H + 2.0);
                ui.label(small("active key: click a key to type its frame, value and handles").color(theme::text_weak()));
            });
            return;
        };
        let c = &graph.nodes[&n].anim[&ch];
        let k = c.keys[i];
        let has_in = i > 0 && c.keys[i - 1].interp == Interp::Bezier;
        let has_out = k.interp == Interp::Bezier && i + 1 < c.keys.len();
        let (mut f, mut v) = (k.frame, k.value);
        let (mut hi, mut ho) = (c.in_handle(i), c.out_handle(i));
        let (f0, v0, hi0, ho0) = (f, v, hi, ho);
        let vs = (v.abs() * 0.005).max(0.001);
        fn field<'a, T: egui::emath::Numeric>(x: &'a mut T, pre: &str, speed: f64) -> egui::DragValue<'a> {
            egui::DragValue::new(x).update_while_editing(false).prefix(pre).speed(speed).max_decimals(4)
        }
        ui.horizontal(|ui| {
            ui.set_min_height(ROW_H + 2.0);
            ui.label(small("active key").color(theme::text_weak()));
            ui.add(field(&mut f, "frame ", 0.1).range(0.0..=1e9)).on_hover_text("move the key to this frame");
            ui.add(field(&mut v, "value ", vs as f64)).on_hover_text("the key's value");
            if has_in {
                ui.label(small("in").color(theme::text_weak()));
                ui.add(field(&mut hi.0, "f ", 0.05).range(-1e6..=0.0));
                ui.add(field(&mut hi.1, "v ", vs as f64));
            }
            if has_out {
                ui.label(small("out").color(theme::text_weak()));
                ui.add(field(&mut ho.0, "f ", 0.05).range(0.0..=1e6));
                ui.add(field(&mut ho.1, "v ", vs as f64));
            }
        });
        let (sx, sy) = (self.x_scale, self.sy(n, ch));
        let Some(c) = graph.nodes.get_mut(&n).and_then(|nd| nd.anim.get_mut(&ch)) else { return };
        let k = &mut c.keys[i];
        if v != v0 {
            k.value = v;
        }
        if hi != hi0 {
            k.vec_in = false;
            k.tan_in = (hi.0.min(0.0), hi.1);
            if !k.broken {
                swing(k, false, sx, sy);
            }
        }
        if ho != ho0 {
            k.vec_out = false;
            k.tan_out = (ho.0.max(0.0), ho.1);
            if !k.broken {
                swing(k, true, sx, sy);
            }
        }
        if f != f0 {
            let mut moved = curveops::take(c, &[f0]);
            if let Some(m) = moved.first_mut() {
                m.frame = f.max(0.0) + 0.0;
                let id = (n, ch, frame_id(m.frame));
                curveops::place(c, moved);
                self.selection.remove(&(n, ch, fid));
                self.selection.insert(id);
                self.active = Some(id);
            }
        }
    }

    fn pie_label(&self, o: PieOp) -> String {
        match o {
            PieOp::Interp(Interp::Hold) => "hold".into(),
            PieOp::Interp(Interp::Linear) => "linear".into(),
            PieOp::Interp(Interp::Smooth) => "smooth".into(),
            PieOp::Interp(Interp::Bezier) => "bezier".into(),
            PieOp::Interp(Interp::Ease(e, d)) => format!("{} {}", e.name(), d.name()),
            PieOp::Flatten => "flatten".into(),
            PieOp::Unify => "tangent".into(),
            PieOp::Break => "break tangent".into(),
            PieOp::Vector => "vector".into(),
            PieOp::Snap => if self.snap { "snap off" } else { "snap on" }.into(),
            PieOp::Frame => "frame all".into(),
            PieOp::FrameSelected => "frame selected".into(),
            PieOp::SelectAll => "select all".into(),
            PieOp::SelectLeft => "select left".into(),
            PieOp::SelectRight => "select right".into(),
            PieOp::SelectColumn => "select columns".into(),
            PieOp::Delete => "delete".into(),
            PieOp::Copy => "copy".into(),
            PieOp::Paste => "paste".into(),
            PieOp::Noise => "noise".into(),
            PieOp::Stepped => "stepped".into(),
            PieOp::Cycle => "cycle".into(),
            PieOp::Extend(_, e) => format!("extend {}", e.name()),
            PieOp::Slide(Slide::Smooth) => "smooth keys".into(),
            PieOp::Slide(s) => s.name().into(),
            PieOp::Unwrap => "unwrap".into(),
            PieOp::MirrorTime => "mirror time".into(),
            PieOp::MirrorValue => "mirror value".into(),
            PieOp::Xform(Xform::Grab) => "grab".into(),
            PieOp::Xform(Xform::Scale) => "scale".into(),
            PieOp::Xform(Xform::Rotate) => "rotate".into(),
            PieOp::Ghost => "ghost".into(),
        }
    }

    /// One curve op — the header, the pie, the keys and Delete all land here.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        o: PieOp,
        out_only: bool,
        graph: &mut EditorGraph,
        rect: &Rect,
        tracks: &[Track],
        hits: &[(KeyId, Pos2)],
        t: u64,
        pointer: Option<Pos2>,
    ) {
        let each = |sel: &HashSet<KeyId>, graph: &mut EditorGraph, f: &dyn Fn(&mut AnimCurve, f64)| {
            for (n, ch, id) in sel.iter().copied() {
                if let Some(c) = graph.nodes.get_mut(&n).and_then(|nd| nd.anim.get_mut(&ch)) {
                    if !c.locked {
                        f(c, id_frame(id));
                    }
                }
            }
        };
        // Per curve, unlocked only.
        let per_curve = |s: &Self, graph: &mut EditorGraph, f: &mut dyn FnMut((u64, Channel), &mut AnimCurve, &[f64])| {
            for (key, frames) in s.sel_by_curve() {
                if let Some(c) = graph.nodes.get_mut(&key.0).and_then(|nd| nd.anim.get_mut(&key.1)) {
                    if !c.locked {
                        f(key, c, &frames);
                    }
                }
            }
        };
        self.note.clear();
        match o {
            PieOp::Interp(mode) if out_only => each(&self.selection, graph, &|c, f| c.set_interp(f, mode)),
            PieOp::Interp(mode) => each(&self.selection, graph, &|c, f| c.set_interp_both(f, mode)),
            PieOp::Unify => {
                let (sx, ys) = (self.x_scale, self.y_scale);
                let norms = self.norms.clone();
                for (key, frames) in self.sel_by_curve() {
                    let sy = ys / norms.get(&key).map_or(1.0, |n| n.1);
                    if let Some(c) = graph.nodes.get_mut(&key.0).and_then(|nd| nd.anim.get_mut(&key.1)) {
                        if c.locked {
                            continue;
                        }
                        for f in frames {
                            c.ensure_bezier(f);
                            if let Some(i) = c.index_of(f) {
                                let k = &mut c.keys[i];
                                k.broken = false;
                                k.vec_in = false;
                                k.vec_out = false;
                                unify(k, sx, sy);
                            }
                        }
                    }
                }
            }
            PieOp::Break => each(&self.selection, graph, &|c, f| {
                c.ensure_bezier(f);
                if let Some(i) = c.index_of(f) {
                    c.keys[i].broken = true;
                }
            }),
            PieOp::Vector => per_curve(self, graph, &mut |_, c, frames| curveops::vector_handles(c, frames)),
            PieOp::Flatten => each(&self.selection, graph, &|c, f| {
                c.set_interp_both(f, Interp::Bezier);
                if let Some(i) = c.index_of(f) {
                    let k = &mut c.keys[i];
                    k.vec_in = false;
                    k.vec_out = false;
                    k.tan_in.1 = 0.0;
                    k.tan_out.1 = 0.0;
                }
            }),
            PieOp::Snap => self.snap = !self.snap,
            PieOp::Frame => self.frame_all(rect, tracks, graph),
            PieOp::FrameSelected => self.frame_selected(rect, graph),
            // Every key the plot can hit: shown and unlocked.
            PieOp::SelectAll => self.selection = hits.iter().map(|(k, _)| *k).collect(),
            PieOp::SelectLeft | PieOp::SelectRight => {
                let tf = t as f64;
                self.selection = hits
                    .iter()
                    .map(|(k, _)| *k)
                    .filter(|k| {
                        let f = id_frame(k.2);
                        if o == PieOp::SelectLeft { f <= tf + 1e-6 } else { f >= tf - 1e-6 }
                    })
                    .collect();
            }
            PieOp::SelectColumn => {
                let mut cols: Vec<f64> = self.selection.iter().map(|k| id_frame(k.2)).collect();
                if cols.is_empty() {
                    cols.push(t as f64);
                }
                self.selection = hits
                    .iter()
                    .map(|(k, _)| *k)
                    .filter(|k| cols.iter().any(|c| (id_frame(k.2) - c).abs() < 1e-6))
                    .collect();
            }
            PieOp::Delete => {
                let doomed: Vec<KeyId> = self.selection.iter().copied().collect();
                self.remove_keys(graph, doomed.into_iter());
                self.selection.clear();
            }
            PieOp::Copy => {
                let (_, mut grabbed) = self.snapshot_selection(graph);
                let f0 = grabbed.iter().map(|(.., k)| k.frame).reduce(f64::min).unwrap_or(0.0);
                for (.., k) in &mut grabbed {
                    k.frame -= f0;
                }
                self.note = format!("copied {} key(s)", grabbed.len());
                self.clip = grabbed;
            }
            PieOp::Paste => {
                if !self.clip.is_empty() {
                    self.selection = crate::dopesheet::paste_keys(graph, &self.clip, t);
                    self.note = format!("pasted {} key(s) at {t}", self.selection.len());
                }
            }
            PieOp::Noise => {
                let targets = self.noise_targets(tracks);
                let all_have = !targets.is_empty()
                    && targets.iter().all(|(n, ch)| {
                        graph.nodes.get(n).and_then(|nd| nd.anim.get(ch)).is_some_and(|c| c.noise.is_some())
                    });
                for (n, ch) in targets {
                    if let Some(c) = graph.nodes.get_mut(&n).and_then(|nd| nd.anim.get_mut(&ch)) {
                        c.noise = if all_have {
                            None
                        } else {
                            c.noise.take().or_else(|| {
                                Some(crate::animnoise::CurveNoise::sized_for(c.keys.iter().map(|k| k.value)))
                            })
                        };
                    }
                }
            }
            PieOp::Stepped => {
                let targets = self.curve_targets(tracks);
                let all_have = !targets.is_empty()
                    && targets.iter().all(|(n, ch)| {
                        graph.nodes.get(n).and_then(|nd| nd.anim.get(ch)).is_some_and(|c| c.stepped.is_some())
                    });
                for (n, ch) in targets {
                    if let Some(c) = graph.nodes.get_mut(&n).and_then(|nd| nd.anim.get_mut(&ch)) {
                        c.stepped = if all_have { None } else { c.stepped.or(Some(Default::default())) };
                    }
                }
            }
            PieOp::Cycle => {
                let targets = self.curve_targets(tracks);
                let all_cycle = !targets.is_empty()
                    && targets.iter().all(|(n, ch)| {
                        graph
                            .nodes
                            .get(n)
                            .and_then(|nd| nd.anim.get(ch))
                            .is_some_and(|c| c.pre == Extend::Cycle && c.post == Extend::Cycle)
                    });
                let e = if all_cycle { Extend::Hold } else { Extend::Cycle };
                for (n, ch) in targets {
                    if let Some(c) = graph.nodes.get_mut(&n).and_then(|nd| nd.anim.get_mut(&ch)) {
                        c.pre = e;
                        c.post = e;
                    }
                }
                self.note = format!("extend: {}", e.name());
            }
            PieOp::Extend(side, e) => {
                for (n, ch) in self.curve_targets(tracks) {
                    if let Some(c) = graph.nodes.get_mut(&n).and_then(|nd| nd.anim.get_mut(&ch)) {
                        if side != Some(true) {
                            c.pre = e;
                        }
                        if side != Some(false) {
                            c.post = e;
                        }
                    }
                }
            }
            PieOp::Slide(kind) => {
                if self.selection.is_empty() {
                    return;
                }
                let mut snapshot = Snapshot::new();
                let mut curves = Vec::new();
                for (key, frames) in self.sel_by_curve() {
                    let Some(nd) = graph.nodes.get(&key.0) else { continue };
                    let Some(c) = nd.anim.get(&key.1) else { continue };
                    if c.locked {
                        continue;
                    }
                    snapshot.insert(key, c.clone());
                    let def = if kind == Slide::Default { default_value(&nd.kind, key.1) } else { None };
                    curves.push((key, frames, def));
                }
                let origin_x = pointer.map_or(rect.center().x, |p| p.x);
                self.slider = Some(SliderRun { kind, snapshot, curves, origin_x, factor: 0.0 });
            }
            PieOp::Unwrap => per_curve(self, graph, &mut |_, c, frames| curveops::unwrap(c, frames, 360.0)),
            PieOp::MirrorTime => {
                let mut sel = HashSet::new();
                per_curve(self, graph, &mut |key, c, frames| {
                    for f in curveops::mirror_time(c, frames, t as f64) {
                        sel.insert((key.0, key.1, frame_id(f)));
                    }
                });
                self.selection = sel;
                self.active = None;
            }
            PieOp::MirrorValue => per_curve(self, graph, &mut |_, c, frames| {
                let vals: Vec<f32> = frames.iter().filter_map(|f| c.index_of(*f)).map(|i| c.keys[i].value).collect();
                let lo = vals.iter().copied().fold(f32::MAX, f32::min);
                let hi = vals.iter().copied().fold(f32::MIN, f32::max);
                if lo <= hi {
                    curveops::mirror_value(c, frames, (lo + hi) * 0.5);
                }
            }),
            PieOp::Xform(kind) => {
                if self.selection.is_empty() {
                    return;
                }
                let (snapshot, grabbed) = self.snapshot_selection(graph);
                if grabbed.is_empty() {
                    return;
                }
                let pts = self.selected_shown(graph);
                let (mut f0, mut f1, mut v0, mut v1) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
                for (f, v) in pts {
                    f0 = f0.min(f);
                    f1 = f1.max(f);
                    v0 = v0.min(v);
                    v1 = v1.max(v);
                }
                let mid_v = (v0 + v1) * 0.5;
                let pivot = match self.pivot {
                    Pivot::Playhead => (t as f32, mid_v),
                    Pivot::Box | Pivot::Each => ((f0 + f1) * 0.5, mid_v),
                };
                let origin = pointer.unwrap_or(rect.center());
                self.modal = Some(Modal { kind, snapshot, grabbed, origin, pivot });
            }
            PieOp::Ghost => {
                if self.ghosts.is_empty() {
                    for tr in tracks.iter().filter(|t| !self.view.hidden.contains(&(t.node, t.ch))) {
                        if let Some(c) = graph.nodes.get(&tr.node).and_then(|n| n.anim.get(&tr.ch)) {
                            self.ghosts.push(Ghost { node: tr.node, ch: tr.ch, color: tr.color, curve: c.clone() });
                        }
                    }
                    self.note = format!("{} ghost curve(s)", self.ghosts.len());
                } else {
                    self.ghosts.clear();
                }
            }
        }
    }

    /// The curves a noise op acts on: those holding a selected key, else
    /// the one curve on show when only one is.
    fn noise_targets(&self, tracks: &[Track]) -> Vec<(u64, Channel)> {
        let mut out: Vec<(u64, Channel)> = Vec::new();
        for (n, ch, _) in &self.selection {
            if !out.contains(&(*n, *ch)) {
                out.push((*n, *ch));
            }
        }
        if out.is_empty() {
            let shown: Vec<(u64, Channel)> = tracks
                .iter()
                .map(|t| (t.node, t.ch))
                .filter(|k| !self.view.hidden.contains(k))
                .collect();
            if shown.len() == 1 {
                out = shown;
            }
        }
        out
    }

    /// The curves a whole-curve setting (extend, stepped) acts on: those
    /// holding a selected key, else every curve shown.
    fn curve_targets(&self, tracks: &[Track]) -> Vec<(u64, Channel)> {
        let mut out: Vec<(u64, Channel)> = Vec::new();
        for (n, ch, _) in &self.selection {
            if !out.contains(&(*n, *ch)) {
                out.push((*n, *ch));
            }
        }
        if out.is_empty() {
            out = tracks.iter().map(|t| (t.node, t.ch)).filter(|k| !self.view.hidden.contains(k)).collect();
        }
        out
    }

    /// The channel list: channels grouped under their node. Click a row to
    /// show only that curve, shift-click to add / remove it, right-click to
    /// lock it against keyframing. The pin on the right keeps a channel in
    /// the editor after the node selection moves on.
    fn channel_list(&mut self, ui: &mut egui::Ui, area: Rect, tracks: &[Track], graph: &mut EditorGraph) {
        let painter = ui.painter_at(area);
        painter.rect_filled(area, 0.0, theme::raised_fill());
        painter.line_segment(
            [Pos2::new(area.max.x, area.min.y), Pos2::new(area.max.x, area.max.y)],
            Stroke::new(1.0, theme::grid_line()),
        );
        let shown = tracks.iter().filter(|t| !self.view.hidden.contains(&(t.node, t.ch))).count();
        let mut isolate: Option<Vec<(u64, Channel)>> = None;
        let mut child = ui.new_child(
            egui::UiBuilder::new()
                .max_rect(area.shrink2(egui::vec2(0.0, 2.0)))
                .layout(egui::Layout::top_down(egui::Align::LEFT)),
        );
        egui::ScrollArea::vertical().id_salt("curves_list").auto_shrink([false, false]).show(
            &mut child,
            |ui| {
                ui.spacing_mut().item_spacing.y = 0.0;
                let w = ui.available_width();
                let mut i = 0;
                while i < tracks.len() {
                    let node = tracks[i].node;
                    let group: Vec<&Track> =
                        tracks[i..].iter().take_while(|t| t.node == node).collect();
                    i += group.len();
                    let keys: Vec<(u64, Channel)> = group.iter().map(|t| (t.node, t.ch)).collect();

                    // Node row: its name; the pin pins every channel under it.
                    let (r, resp) = ui.allocate_exact_size(egui::vec2(w, ROW_H), egui::Sense::click());
                    let all_pinned = keys.iter().all(|k| self.view.pinned.contains(k));
                    let name = graph
                        .nodes
                        .get(&node)
                        .map(|n| crate::editor::node_label(&n.kind))
                        .unwrap_or_default();
                    ui.painter().text(
                        Pos2::new(r.min.x + 6.0, r.center().y),
                        egui::Align2::LEFT_CENTER,
                        name,
                        egui::FontId::proportional(11.0),
                        theme::text_strong(),
                    );
                    if pin_toggle(ui, r, egui::Id::new(("curves_pin_node", node)), all_pinned, resp.hovered()) {
                        for k in &keys {
                            if all_pinned {
                                self.view.pinned.remove(k);
                            } else {
                                self.view.pinned.insert(*k);
                            }
                        }
                    }
                    if resp
                        .on_hover_text("click: show only this node's curves · shift-click: add / remove them")
                        .clicked()
                    {
                        if ui.input(|i| i.modifiers.shift) {
                            let all_on = keys.iter().all(|k| !self.view.hidden.contains(k));
                            for k in &keys {
                                if all_on {
                                    self.view.hidden.insert(*k);
                                } else {
                                    self.view.hidden.remove(k);
                                }
                            }
                        } else {
                            isolate = Some(keys.clone());
                        }
                    }

                    for tr in group {
                        let key = (tr.node, tr.ch);
                        let on = !self.view.hidden.contains(&key);
                        let locked = graph
                            .nodes
                            .get(&tr.node)
                            .is_some_and(|n| crate::anim::is_locked(&n.anim, tr.ch));
                        let (r, resp) =
                            ui.allocate_exact_size(egui::vec2(w, ROW_H), egui::Sense::click());
                        if resp.hovered() {
                            ui.painter().rect_filled(r, 0.0, theme::hover_fill());
                        }
                        let col = if locked {
                            theme::locked()
                        } else if on {
                            tr.color
                        } else {
                            theme::text_weak()
                        };
                        // Swatch: filled while the curve is drawn.
                        let sw = Rect::from_center_size(
                            Pos2::new(r.min.x + 16.0, r.center().y),
                            egui::vec2(8.0, 8.0),
                        );
                        if on {
                            ui.painter().rect_filled(sw, 1.5, col);
                        } else {
                            ui.painter().rect_stroke(sw, 1.5, Stroke::new(1.0, col), egui::StrokeKind::Inside);
                        }
                        let text = if locked { format!("{} ·lock", tr.short) } else { tr.short.clone() };
                        ui.painter().text(
                            Pos2::new(r.min.x + 26.0, r.center().y),
                            egui::Align2::LEFT_CENTER,
                            text,
                            egui::FontId::proportional(11.0),
                            if on { theme::text() } else { theme::text_weak() },
                        );
                        let pinned = self.view.pinned.contains(&key);
                        if pin_toggle(ui, r, egui::Id::new(("curves_pin", key)), pinned, resp.hovered()) {
                            if pinned {
                                self.view.pinned.remove(&key);
                            } else {
                                self.view.pinned.insert(key);
                            }
                        }
                        let resp = resp.on_hover_text(if locked {
                            "LOCKED against keyframing — its keys cannot be edited\n\
                             click: show only this one · shift-click: add / remove\n\
                             right-click: unlock"
                        } else {
                            "click: show only this one · shift-click: add / remove\n\
                             right-click: lock against keyframing"
                        });
                        if resp.secondary_clicked() {
                            if let Some(n) = graph.nodes.get_mut(&tr.node) {
                                crate::anim::set_locked(&mut n.anim, tr.ch, !locked);
                            }
                        }
                        if resp.clicked() {
                            if ui.input(|i| i.modifiers.shift) {
                                if on {
                                    self.view.hidden.insert(key);
                                } else {
                                    self.view.hidden.remove(&key);
                                }
                            } else {
                                isolate = Some(vec![key]);
                            }
                        }
                        // The curve's noise, where it has one: its settings
                        // live under it, like Blender's modifier panel.
                        if let Some(c) = graph.nodes.get_mut(&tr.node).and_then(|n| n.anim.get_mut(&tr.ch)) {
                            if let Some(nz) = c.noise.as_mut() {
                                if noise_rows(ui, w, key, nz) {
                                    c.noise = None;
                                }
                            }
                            if let Some(st) = c.stepped.as_mut() {
                                if stepped_rows(ui, w, key, st) {
                                    c.stepped = None;
                                }
                            }
                        }
                    }
                }
            },
        );
        if let Some(keep) = isolate {
            // Clicking what is already isolated puts everything back.
            let already = shown == keep.len() && keep.iter().all(|k| !self.view.hidden.contains(k));
            if already {
                self.view.hidden.clear();
            } else {
                self.view.hidden =
                    tracks.iter().map(|t| (t.node, t.ch)).filter(|k| !keep.contains(k)).collect();
            }
        }
    }

    /// Select every key on a channel — the self-test's way in.
    pub fn select_channel(&mut self, graph: &EditorGraph, node: u64, ch: Channel) {
        self.selection.clear();
        if let Some(c) = graph.nodes.get(&node).and_then(|n| n.anim.get(&ch)) {
            for k in &c.keys {
                self.selection.insert((node, ch, frame_id(k.frame)));
            }
        }
    }

    /// Snapshot every curve the selection touches plus the keys as they
    /// are, so a gesture is recomputed from scratch each frame instead of
    /// accumulating drift. Locked curves are left out.
    fn snapshot_selection(&self, graph: &EditorGraph) -> (Snapshot, Grabbed) {
        let mut snapshot = Snapshot::new();
        let mut grabbed = Vec::new();
        for (n, ch, f) in self.selection.iter().copied() {
            let Some(curve) = graph.nodes.get(&n).and_then(|nd| nd.anim.get(&ch)) else {
                continue;
            };
            if curve.locked {
                continue;
            }
            let f = id_frame(f);
            let Some(k) = curve.keys.iter().find(|k| k.frame == f) else {
                continue;
            };
            snapshot.entry((n, ch)).or_insert_with(|| curve.clone());
            grabbed.push((n, ch, *k));
        }
        (snapshot, grabbed)
    }

    /// The frame a screen x lands on in the last painted view.
    pub fn frame_at_x(&self, x: f32) -> f32 {
        self.frame_at(&self.last_rect, x)
    }

    /// Where a (frame, value) pair sits on screen in the last painted view.
    /// Used by `--test-curves` to aim synthetic drags.
    pub fn key_pos(&self, frame: f32, value: f32) -> Pos2 {
        Pos2::new(self.x_of(&self.last_rect, frame), self.y_of(&self.last_rect, value))
    }

    fn remove_keys(&mut self, graph: &mut EditorGraph, keys: impl Iterator<Item = KeyId>) {
        for (n, ch, f) in keys {
            let Some(nd) = graph.nodes.get_mut(&n) else { continue };
            if let Some(c) = nd.anim.get_mut(&ch) {
                c.remove_key(id_frame(f));
                // An emptied curve goes too — unless it is locked: the lock
                // lives on the curve and must survive (the dope sheet's rule).
                if c.keys.is_empty() && !c.locked {
                    nd.anim.remove(&ch);
                }
            }
            self.selection.remove(&(n, ch, f));
        }
    }
}

/// Put the gesture's curves back the way they were when it began.
fn restore(graph: &mut EditorGraph, snapshot: &Snapshot) {
    for ((n, ch), curve) in snapshot {
        if let Some(nd) = graph.nodes.get_mut(n) {
            nd.anim.insert(*ch, curve.clone());
        }
    }
}

/// Re-place moved keys: the curves go back to the snapshot, the grabbed
/// keys come out, and the moved ones go in (replacing any key on the
/// frame they land on). Returns where each landed, in `moved`'s order.
fn apply_moved(
    graph: &mut EditorGraph,
    snapshot: &Snapshot,
    grabbed: &Grabbed,
    moved: Vec<(u64, Channel, Key)>,
) -> Vec<KeyId> {
    restore(graph, snapshot);
    let mut out: BTreeMap<(u64, Channel), Vec<f64>> = BTreeMap::new();
    for (n, ch, k) in grabbed {
        out.entry((*n, *ch)).or_default().push(k.frame);
    }
    for ((n, ch), frames) in out {
        if let Some(c) = graph.nodes.get_mut(&n).and_then(|nd| nd.anim.get_mut(&ch)) {
            curveops::take(c, &frames);
        }
    }
    let mut ids = Vec::with_capacity(moved.len());
    let mut put: BTreeMap<(u64, Channel), Vec<Key>> = BTreeMap::new();
    for (n, ch, k) in moved {
        let k = Key { frame: k.frame.max(0.0) + 0.0, ..k };
        ids.push((n, ch, frame_id(k.frame)));
        put.entry((n, ch)).or_default().push(k);
    }
    for ((n, ch), keys) in put {
        if let Some(c) = graph.nodes.get_mut(&n).and_then(|nd| nd.anim.get_mut(&ch)) {
            curveops::place(c, keys);
        }
    }
    ids
}

/// `v` rounded to a multiple of `step` (kills float dust in the read-out).
fn round_to(v: f32, step: f32) -> f32 {
    (v / step).round() * step
}

/// A ladder rung's label.
fn fmt_step(s: f32) -> String {
    if s >= 1.0 {
        format!("{}", s as i64)
    } else {
        format!("{s}")
    }
}

/// A channel's default value: what a fresh node of the same kind holds
/// (`None` for a plugin's, whose fresh node the catalog cannot make).
fn default_value(kind: &engine::graph::NodeKind, ch: Channel) -> Option<f32> {
    if matches!(kind, engine::graph::NodeKind::Custom { .. }) {
        return None;
    }
    let d = std::mem::discriminant(kind);
    crate::editor::node_catalog()
        .into_iter()
        .map(|e| (e.make)(0))
        .find(|k| std::mem::discriminant(k) == d)
        .and_then(|k| crate::anim::get(&k, ch))
}

/// Keep an unbroken key's handles collinear after one moved: the other
/// swings to point straight away from it, keeping its own on-screen
/// length. Collinear on screen is collinear in data (the axes only scale).
fn swing(k: &mut Key, moved_out: bool, sx: f32, sy: f32) {
    let to_s = |t: (f32, f32)| egui::vec2(t.0 * sx, t.1 * sy);
    let (m, o) = if moved_out { (to_s(k.tan_out), to_s(k.tan_in)) } else { (to_s(k.tan_in), to_s(k.tan_out)) };
    if m.length() < 1e-3 {
        return;
    }
    let len = if o.length() > 1e-3 { o.length() } else { m.length() };
    let n = -m.normalized() * len;
    let n = (n.x / sx, n.y / sy);
    if moved_out {
        k.tan_in = (n.0.min(0.0), n.1);
    } else {
        k.tan_out = (n.0.max(0.0), n.1);
    }
}

/// Align a key's two handles on their average direction, each keeping its
/// on-screen length — `T` on a key whose tangent was broken.
fn unify(k: &mut Key, sx: f32, sy: f32) {
    let to_s = |t: (f32, f32)| egui::vec2(t.0 * sx, t.1 * sy);
    let (i, o) = (to_s(k.tan_in), to_s(k.tan_out));
    let dir = o.normalized() - i.normalized();
    if dir.length() < 1e-3 || !dir.x.is_finite() {
        return;
    }
    let d = dir.normalized();
    let (li, lo) = (i.length(), o.length());
    k.tan_out = ((d.x * lo / sx).max(0.0), d.y * lo / sy);
    k.tan_in = ((-d.x * li / sx).min(0.0), -d.y * li / sy);
}

/// The pin at a list row's right end: a filled dot on a stem when pinned, a
/// faint ring while the row is hovered, nothing otherwise. Returns clicked.
/// A curve's noise settings, indented under its row: scale, strength,
/// phase, offset, depth, and × to take it off (returns true then).
fn noise_rows(ui: &mut egui::Ui, w: f32, key: (u64, Channel), nz: &mut crate::animnoise::CurveNoise) -> bool {
    let mut remove = false;
    let half = ((w - 34.0) * 0.5).max(40.0);
    fn drag<'a>(v: &'a mut f32, pre: &str, speed: f32) -> egui::DragValue<'a> {
        egui::DragValue::new(v).update_while_editing(false).prefix(pre).speed(speed).max_decimals(3)
    }
    let small = |t: &str| egui::RichText::new(t).small();
    ui.horizontal(|ui| {
        ui.add_space(24.0);
        ui.label(small("~ noise").color(theme::text_weak()));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add_space(6.0);
            remove = ui
                .add(egui::Button::new(small("×")).frame(false))
                .on_hover_text("remove the noise — the keys are untouched")
                .clicked();
        });
    });
    ui.push_id(("curve-noise", key), |ui| {
        ui.horizontal(|ui| {
            ui.add_space(24.0);
            ui.add_sized([half, ROW_H], drag(&mut nz.scale, "scale ", 0.1).range(0.1..=10_000.0))
                .on_hover_text("frames per wiggle — bigger is slower");
            let sp = (nz.strength.abs() * 0.01).max(0.001);
            ui.add_sized([half, ROW_H], drag(&mut nz.strength, "str ", sp))
                .on_hover_text("strength: it swings ± half of this, in the channel's units");
        });
        ui.horizontal(|ui| {
            ui.add_space(24.0);
            ui.add_sized([half, ROW_H], drag(&mut nz.phase, "phase ", 0.01))
                .on_hover_text("where in the noise it reads: a smooth seed");
            ui.add_sized([half, ROW_H], drag(&mut nz.offset, "offset ", 0.1))
                .on_hover_text("slides the wiggle along the timeline, in frames");
        });
        ui.horizontal(|ui| {
            ui.add_space(24.0);
            ui.add_sized(
                [half, ROW_H],
                egui::DragValue::new(&mut nz.depth).update_while_editing(false)
                    .prefix("depth ")
                    .speed(0.05)
                    .range(0..=crate::animnoise::MAX_DEPTH),
            )
            .on_hover_text("extra octaves of finer detail");
        });
    });
    remove
}

/// A curve's Stepped settings, under its row like the noise's: frames per
/// step and where the steps start, and × to take it off (returns true then).
fn stepped_rows(ui: &mut egui::Ui, w: f32, key: (u64, Channel), st: &mut crate::anim::Stepped) -> bool {
    let mut remove = false;
    let half = ((w - 34.0) * 0.5).max(40.0);
    let small = |t: &str| egui::RichText::new(t).small();
    ui.horizontal(|ui| {
        ui.add_space(24.0);
        ui.label(small("~ stepped").color(theme::text_weak()));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add_space(6.0);
            remove = ui
                .add(egui::Button::new(small("×")).frame(false))
                .on_hover_text("remove the steps — the keys are untouched")
                .clicked();
        });
    });
    ui.push_id(("curve-stepped", key), |ui| {
        ui.horizontal(|ui| {
            ui.add_space(24.0);
            ui.add_sized(
                [half, ROW_H],
                egui::DragValue::new(&mut st.step)
                    .update_while_editing(false)
                    .prefix("step ")
                    .speed(0.05)
                    .range(0.1..=1000.0)
                    .max_decimals(2),
            )
            .on_hover_text("frames each value is held — 2 animates on twos");
            ui.add_sized(
                [half, ROW_H],
                egui::DragValue::new(&mut st.offset)
                    .update_while_editing(false)
                    .prefix("offset ")
                    .speed(0.05)
                    .max_decimals(2),
            )
            .on_hover_text("where the steps start, in frames");
        });
    });
    remove
}

fn pin_toggle(ui: &mut egui::Ui, row: Rect, id: egui::Id, on: bool, row_hot: bool) -> bool {
    let r = Rect::from_center_size(Pos2::new(row.max.x - 12.0, row.center().y), egui::vec2(16.0, 16.0));
    let resp = ui.interact(r, id, egui::Sense::click());
    let hot = resp.hovered();
    let c = r.center() + egui::vec2(0.0, -2.0);
    if on || hot {
        let col = if on { theme::selection() } else { theme::text() };
        ui.painter().line_segment([c, c + egui::vec2(0.0, 7.0)], Stroke::new(1.2, col));
        ui.painter().circle_filled(c, 3.2, col);
    } else if row_hot {
        ui.painter().circle_stroke(c, 3.2, Stroke::new(1.0, theme::text_weak()));
    }
    resp.on_hover_text(if on {
        "pinned: stays in the editor whatever is selected — click to unpin"
    } else {
        "pin: keep this in the editor after the selection moves on"
    })
    .clicked()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anim::Channel;
    use engine::graph::NodeKind;

    fn graph_with(keys: &[(u64, f32)]) -> (EditorGraph, Vec<Track>) {
        let mut g = EditorGraph::default();
        let id = g.add(NodeKind::Transform(Default::default()), Pos2::ZERO);
        let c = g.nodes.get_mut(&id).unwrap().anim.entry(Channel::TranslateX).or_default();
        for (f, v) in keys {
            c.set_key(*f, *v);
        }
        let tracks = vec![Track {
            node: id,
            ch: Channel::TranslateX,
            color: Color32::WHITE,
            short: "t".into(),
        }];
        (g, tracks)
    }

    /// The noise op: on for the curve with a selected key (sized to it),
    /// off again when run on a curve that has one; with nothing selected it
    /// takes the one curve on show.
    #[test]
    fn the_noise_op_toggles_a_sized_noise() {
        let (mut g, tracks) = graph_with(&[(0, 0.0), (20, 500.0)]);
        let id = tracks[0].node;
        let mut ed = CurveEditor::default();
        let noise = |g: &EditorGraph| g.nodes[&id].anim[&Channel::TranslateX].noise.clone();
        ed.selection.insert((id, Channel::TranslateX, frame_id(20.0)));
        ed.run(PieOp::Noise, false, &mut g, &Rect::ZERO, &tracks, &[], 0, None);
        assert_eq!(noise(&g).map(|n| n.strength), Some(50.0), "a tenth of the move");
        ed.run(PieOp::Noise, false, &mut g, &Rect::ZERO, &tracks, &[], 0, None);
        assert_eq!(noise(&g), None, "run again: off");
        ed.selection.clear();
        ed.run(PieOp::Noise, false, &mut g, &Rect::ZERO, &tracks, &[], 0, None);
        assert!(noise(&g).is_some(), "the only curve shown");
        // The keys never moved.
        let c = &g.nodes[&id].anim[&Channel::TranslateX];
        assert_eq!(c.keys.iter().map(|k| k.value).collect::<Vec<_>>(), vec![0.0, 500.0]);
    }

    /// Not an assertion — a way to LOOK at it: a keyed move with noise on,
    /// drawn over its faint key curve, the settings under it in the list.
    /// `cargo test -p app noisy_curve_render -- --ignored` → target/curves-noise.png
    #[test]
    #[ignore]
    fn noisy_curve_render() {
        let mut g = EditorGraph::default();
        let a = g.add(NodeKind::Transform(Default::default()), Pos2::ZERO);
        let c = g.nodes.get_mut(&a).unwrap().anim.entry(Channel::TranslateX).or_default();
        for (f, v) in [(0.0, 0.0), (30.0, 200.0), (60.0, 120.0)] {
            c.set_key(f, v);
        }
        c.set_interp_both(30.0, Interp::Smooth);
        c.noise = Some(crate::animnoise::CurveNoise { strength: 40.0, scale: 6.0, depth: 2, ..Default::default() });
        let mut ed = CurveEditor { snap: false, ..Default::default() };
        let out = crate::uishot::render("curves-noise", 1000.0, 320.0, 2.0, 3, |ui, _| {
            ed.ui(ui, &mut g, &[a], 18);
        });
        if let Some(p) = out {
            eprintln!("wrote {}", p.display());
        }
    }

    #[test]
    fn screen_data_roundtrip() {
        let ed = CurveEditor::default();
        let r = Rect::from_min_size(Pos2::new(10.0, 20.0), egui::vec2(600.0, 300.0));
        for (f, v) in [(0.0, 0.0), (12.0, 240.0), (7.5, -3.25)] {
            let (x, y) = (ed.x_of(&r, f), ed.y_of(&r, v));
            assert!((ed.frame_at(&r, x) - f).abs() < 1e-3, "frame roundtrip {f}");
            assert!((ed.value_at(&r, y) - v).abs() < 1e-3, "value roundtrip {v}");
        }
    }

    #[test]
    fn frame_all_scales_axes_independently() {
        // 40 frames wide but only 0.5 value units tall: the two scales must
        // be fitted separately (a shared scale would squash one flat).
        let (g, tracks) = graph_with(&[(0, 1.0), (40, 1.5)]);
        let mut ed = CurveEditor::default();
        let r = Rect::from_min_size(Pos2::ZERO, egui::vec2(800.0, 400.0));
        ed.frame_all(&r, &tracks, &g);
        // Both spans land inside the rect...
        let (x0, x1) = (ed.x_of(&r, 0.0), ed.x_of(&r, 40.0));
        let (y0, y1) = (ed.y_of(&r, 1.0), ed.y_of(&r, 1.5));
        assert!(r.x_range().contains(x0) && r.x_range().contains(x1), "x fit {x0}..{x1}");
        assert!(r.y_range().contains(y0) && r.y_range().contains(y1), "y fit {y0}..{y1}");
        // ...and each uses most of its axis, which only works decoupled.
        assert!((x1 - x0) > r.width() * 0.7, "time span too small: {}", x1 - x0);
        assert!((y0 - y1) > r.height() * 0.5, "value span too small: {}", y0 - y1);
        assert!(ed.y_scale > ed.x_scale * 10.0, "axes not independent");
    }

    #[test]
    fn flat_curve_frames_without_dividing_by_zero() {
        let (g, tracks) = graph_with(&[(5, 2.0), (5, 2.0)]);
        let mut ed = CurveEditor::default();
        let r = Rect::from_min_size(Pos2::ZERO, egui::vec2(800.0, 400.0));
        ed.frame_all(&r, &tracks, &g);
        assert!(ed.x_scale.is_finite() && ed.y_scale.is_finite());
        assert!(ed.x_of(&r, 5.0).is_finite() && ed.y_of(&r, 2.0).is_finite());
    }

    #[test]
    fn unbroken_handles_stay_collinear() {
        let mut k = Key::new(0, 0.0);
        k.tan_in = (-3.0, 0.0);
        k.tan_out = (3.0, 6.0);
        swing(&mut k, true, 8.0, 60.0);
        // In points: out = (24, 360); in must be the exact opposite ray.
        let (o, i) = (egui::vec2(k.tan_out.0 * 8.0, k.tan_out.1 * 60.0), egui::vec2(k.tan_in.0 * 8.0, k.tan_in.1 * 60.0));
        assert!((o.normalized() + i.normalized()).length() < 1e-4, "{o:?} vs {i:?}");
        assert!((i.length() - 24.0).abs() < 1e-3, "in kept its own length");
        // Unify of a broken pair lines them up without changing lengths.
        let mut b = Key::new(0, 0.0);
        b.tan_in = (-2.0, 1.0);
        b.tan_out = (2.0, 1.0);
        unify(&mut b, 8.0, 60.0);
        let (o, i) = (egui::vec2(b.tan_out.0 * 8.0, b.tan_out.1 * 60.0), egui::vec2(b.tan_in.0 * 8.0, b.tan_in.1 * 60.0));
        assert!((o.normalized() + i.normalized()).length() < 1e-4);
        assert!(o.y.abs() < 1e-3, "a V-shaped pair averages to level: {o:?}");
    }

    fn grabbed_all(g: &EditorGraph, id: u64) -> Grabbed {
        g.nodes[&id].anim[&Channel::TranslateX].keys.iter().map(|k| (id, Channel::TranslateX, *k)).collect()
    }

    /// G, S and R through `xformed` + `apply_moved`: whole keys travel
    /// (interpolation and handles with them), scale works about the pivot,
    /// and rotate with pivot "each" turns handles while the keys stay put.
    #[test]
    fn modal_transforms_move_whole_keys() {
        let (mut g, _) = graph_with(&[(0, 0.0), (10, 10.0)]);
        let id = g.nodes.keys().copied().next().unwrap();
        g.nodes.get_mut(&id).unwrap().anim.get_mut(&Channel::TranslateX).unwrap().set_interp(0.0, Interp::Bezier);
        let r = Rect::from_min_size(Pos2::ZERO, egui::vec2(800.0, 400.0));
        let mut ed = CurveEditor::default();
        let grabbed = grabbed_all(&g, id);
        let snap: Snapshot = [((id, Channel::TranslateX), g.nodes[&id].anim[&Channel::TranslateX].clone())].into();
        let o = Pos2::new(ed.x_of(&r, 5.0), ed.y_of(&r, 5.0));
        // Grab: +3 frames, +2 value.
        let p = Pos2::new(ed.x_of(&r, 8.0), ed.y_of(&r, 7.0));
        let moved = ed.xformed(&r, Xform::Grab, (5.0, 5.0), &grabbed, o, p, false, false);
        apply_moved(&mut g, &snap, &grabbed, moved);
        let c = &g.nodes[&id].anim[&Channel::TranslateX];
        assert_eq!(c.keys.iter().map(|k| (k.frame, k.value)).collect::<Vec<_>>(), vec![(3.0, 2.0), (13.0, 12.0)]);
        assert_eq!(c.keys[0].interp, Interp::Bezier, "the key's own interpolation travelled");
        // Scale x2 about the box centre (5, 5), time only (Z held).
        let o2 = Pos2::new(ed.x_of(&r, 7.0), ed.y_of(&r, 5.0));
        let p2 = Pos2::new(ed.x_of(&r, 9.0), ed.y_of(&r, 5.0));
        let moved = ed.xformed(&r, Xform::Scale, (5.0, 5.0), &grabbed, o2, p2, true, false);
        apply_moved(&mut g, &snap, &grabbed, moved);
        let c = &g.nodes[&id].anim[&Channel::TranslateX];
        assert_eq!(c.keys.iter().map(|k| (k.frame, k.value)).collect::<Vec<_>>(), vec![(0.0, 0.0), (15.0, 10.0)], "frames 5 ∓ 5·2 (the left clamps at 0)");
        // Rotate a quarter turn, pivot each: keys stay, the out handle turns.
        ed.pivot = Pivot::Each;
        let mut gr = grabbed.clone();
        gr[0].2.tan_out = (2.0, 0.0);
        let pv = Pos2::new(ed.x_of(&r, 5.0), ed.y_of(&r, 5.0));
        let moved = ed.xformed(&r, Xform::Rotate, (5.0, 5.0), &gr, pv + egui::vec2(100.0, 0.0), pv + egui::vec2(0.0, -100.0), false, false);
        let k = moved[0].2;
        assert_eq!((k.frame, k.value), (0.0, 0.0), "the key stayed");
        // (16 px, 0) turned to point up: x 0, y +16 px = 16/60 value units.
        assert!(k.tan_out.0.abs() < 1e-4 && (k.tan_out.1 - 16.0 / 60.0).abs() < 1e-4, "{:?}", k.tan_out);
    }

    /// Normalize: every curve is SHOWN in -1..1, and moving a key in shown
    /// units lands it in the curve's own units.
    #[test]
    fn normalize_shows_curves_in_unit_range() {
        let (g, tracks) = graph_with(&[(0, 100.0), (10, 300.0)]);
        let id = tracks[0].node;
        let mut ed = CurveEditor { normalize: true, ..Default::default() };
        ed.compute_norms(&tracks, &g);
        assert_eq!(ed.disp(id, Channel::TranslateX, 100.0), -1.0);
        assert_eq!(ed.disp(id, Channel::TranslateX, 300.0), 1.0);
        assert_eq!(ed.undisp(id, Channel::TranslateX, 0.5), 250.0);
        ed.normalize = false;
        ed.compute_norms(&tracks, &g);
        assert_eq!(ed.disp(id, Channel::TranslateX, 300.0), 300.0, "off: identity");
    }

    #[test]
    fn nice_step_picks_1_2_5_decades() {
        assert_eq!(nice_step(1.0), 1.0);
        assert_eq!(nice_step(3.0), 2.0);
        assert_eq!(nice_step(6.0), 5.0);
        assert_eq!(nice_step(0.03), 0.02);
        assert_eq!(nice_step(230.0), 200.0);
        assert_eq!(nice_step(0.0), 1.0); // degenerate input is safe
    }
}

#[cfg(test)]
mod render_tests {
    use super::*;
    use engine::graph::NodeKind;

    /// Look at it: `cargo test -p app curves_render -- --ignored`, then
    /// open `target/curves.png`. Two nodes, one only PINNED (not selected),
    /// a sub-frame key, snap off.
    #[test]
    #[ignore]
    fn curves_render() {
        let mut g = EditorGraph::default();
        let a = g.add(NodeKind::Transform(Default::default()), Pos2::ZERO);
        let b = g.add(NodeKind::Transform(Default::default()), Pos2::ZERO);
        for (id, ch, keys) in [
            (a, Channel::TranslateX, vec![(0.0, 0.0), (12.5, 40.0), (24.0, 40.0), (40.0, 0.0)]),
            (a, Channel::TranslateY, vec![(6.0, 10.0), (30.0, -20.0)]),
            (b, Channel::ScaleX, vec![(10.0, 1.0), (20.0, 30.0), (30.0, 1.0)]),
        ] {
            let c = g.nodes.get_mut(&id).unwrap().anim.entry(ch).or_default();
            for (f, v) in keys {
                c.set_key(f, v);
            }
            c.set_interp_both(12.5, Interp::Smooth);
        }
        // Frame 24: a BROKEN tangent (square handle ends), the out handle
        // kicked up on its own.
        let c = g.nodes.get_mut(&a).unwrap().anim.get_mut(&Channel::TranslateX).unwrap();
        c.ensure_bezier(24.0);
        let i = c.index_of(24.0).unwrap();
        c.keys[i].broken = true;
        c.keys[i].tan_out = (4.0, 15.0);
        let mut ed = CurveEditor { snap: false, ..Default::default() };
        ed.selection.insert((a, Channel::TranslateX, frame_id(24.0)));
        ed.view.pinned.insert((b, Channel::ScaleX));
        ed.selection.insert((a, Channel::TranslateX, frame_id(12.5)));
        let out = crate::uishot::render("curves", 1000.0, 320.0, 2.0, 3, |ui, _| {
            ed.ui(ui, &mut g, &[a], 18);
        });
        if let Some(p) = out {
            eprintln!("wrote {}", p.display());
        }
    }

    /// The round-1 additions in one picture: a cycling curve (tails drawn
    /// past the keys), a bounce ease, vector handles, a stepped curve, a
    /// ghost, and the active key's typed fields in the header.
    /// `cargo test -p app curves_more_render -- --ignored` → target/curves-more.png
    #[test]
    #[ignore]
    fn curves_more_render() {
        use crate::anim::{Ease, EaseDir, Extend, Stepped};
        let mut g = EditorGraph::default();
        let a = g.add(NodeKind::Transform(Default::default()), Pos2::ZERO);
        let anim = &mut g.nodes.get_mut(&a).unwrap().anim;
        let c = anim.entry(Channel::TranslateX).or_default();
        for (f, v) in [(10.0, 0.0), (20.0, 40.0), (30.0, 10.0)] {
            c.set_key(f, v);
        }
        crate::curveops::vector_handles(c, &[20.0]);
        c.pre = Extend::Cycle;
        c.post = Extend::CycleOffset;
        let ghost = c.clone();
        c.keys[1].value = 55.0;
        let y = anim.entry(Channel::TranslateY).or_default();
        for (f, v) in [(5.0, -20.0), (40.0, 20.0)] {
            y.set_key(f, v);
        }
        y.stepped = Some(Stepped { step: 4.0, offset: 0.0 });
        let sx = anim.entry(Channel::ScaleX).or_default();
        for (f, v) in [(12.0, -30.0), (34.0, 0.0)] {
            sx.set_key(f, v);
        }
        sx.set_interp(12.0, Interp::Ease(Ease::Bounce, EaseDir::Out));
        let mut ed = CurveEditor { snap: false, ..Default::default() };
        ed.selection.insert((a, Channel::TranslateX, frame_id(20.0)));
        ed.active = Some((a, Channel::TranslateX, frame_id(20.0)));
        ed.ghosts.push(Ghost { node: a, ch: Channel::TranslateX, color: ch_color(Channel::TranslateX), curve: ghost });
        let out = crate::uishot::render("curves-more", 1100.0, 380.0, 2.0, 3, |ui, _| {
            ed.ui(ui, &mut g, &[a], 18);
        });
        if let Some(p) = out {
            eprintln!("wrote {}", p.display());
        }
    }
}
