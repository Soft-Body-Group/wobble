//! Keyframe animation: authored curves live on ENode (`anim` map), and every
//! frame the app evaluates them AT THE DISPLAYED TICK into a transient engine
//! Graph (`EditorGraph::to_engine_graph_at`). The engine never learns what a
//! keyframe is; evaluated values are never written back into the editor graph
//! (write-back would flood the undo stack and thrash the render-cache key).
//!
//! Keyframes are in TIMELINE FRAMES (the engine has no seconds); playback
//! ticks are unbounded, so evaluation happens at `tick % timeline_len`.
//! A key's frame is FRACTIONAL (`f64`): the curve editor can drop keys
//! between frames with its snap off. Whole-frame keys still write as plain
//! integers, so a setup that never used sub-frames reads exactly as before.

use engine::graph::{NodeKind, SplinePoint};

/// The per-node curve map: channel -> keys.
pub type Anim = std::collections::BTreeMap<Channel, AnimCurve>;

/// One animatable scalar on a node. Flat unit variants so the map serializes
/// with plain-string keys (JSON object keys must be strings).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Channel {
    TranslateX,
    TranslateY,
    Rotate,
    ScaleX,
    ScaleY,
    CenterX,
    CenterY,
    /// A Transform's shear (x by y, y by x) and taper (amount, axis angle).
    ShearX,
    ShearY,
    Taper,
    TaperAngle,
    LiftR,
    LiftG,
    LiftB,
    GammaR,
    GammaG,
    GammaB,
    GainR,
    GainG,
    GainB,
    OffsetR,
    OffsetG,
    OffsetB,
    BlurSize,
    SoftCropL,
    SoftCropT,
    SoftCropR,
    SoftCropB,
    MotionBlurAngle,
    MotionBlurOffset,
    /// The 3D cards nodes (`nodes::scene3d`): a Camera's transform, focal
    /// length, filmback and clip; a Card's transform, aspect and opacity;
    /// a Render3D's shutter.
    Cam3dTx,
    Cam3dTy,
    Cam3dTz,
    Cam3dRx,
    Cam3dRy,
    Cam3dRz,
    Cam3dFocal,
    Cam3dApW,
    Cam3dApH,
    Cam3dNear,
    Cam3dFar,
    Card3dTx,
    Card3dTy,
    Card3dTz,
    Card3dRx,
    Card3dRy,
    Card3dRz,
    Card3dSx,
    Card3dSy,
    Card3dSz,
    Card3dAspect,
    Card3dOpacity,
    Render3dShutter,
    Render3dOffset,
    /// VectorBlur's shutter and its offset.
    VectorBlurAmount,
    VectorBlurOffset,
    /// VectorWarp: frames of motion to follow.
    VectorWarpAmount,
    /// Rifetime in curve mode: the source frame this output frame shows.
    RifetimeFrame,
    /// Freeze: the comp frame every tick holds.
    FreezeFrame,
    /// HueRotate's angle, degrees.
    HueRotateAngle,
    /// Text: where the block sits and its em, canvas pixels.
    TextX,
    TextY,
    TextSize,
    /// Burnin: its em, the margin from the frame's edge, and a custom place.
    BurninSize,
    BurninMargin,
    BurninX,
    BurninY,
    /// LUT: the mix, and the log2 shaper's range in stops.
    LutMix,
    LutLogMin,
    LutLogMax,
    /// Inverse View: the dither in 8-bit steps, the soft ceiling.
    LutDither,
    LutLimit,
    /// Mirror: a point on the line (normalised) and its angle, degrees.
    MirrorX,
    MirrorY,
    MirrorAngle,
    /// Compare: a point on the wipe line (normalised) and its angle.
    CompareX,
    CompareY,
    CompareAngle,
    /// Lens Distortion: radial, tangential, principal point (normalised),
    /// squeeze, zoom, red and blue magnification.
    LensK1,
    LensK2,
    LensK3,
    LensP1,
    LensP2,
    LensCenterX,
    LensCenterY,
    LensSqueeze,
    LensZoom,
    LensCaR,
    LensCaB,
    /// Gamma: the master, r/g/b on top of it, alpha's own, the mix.
    GammaNodeMaster,
    GammaNodeR,
    GammaNodeG,
    GammaNodeB,
    GammaNodeA,
    GammaNodeMix,
    /// Tone Map: exposure in, the knee, the toe, the mix.
    ToneExposure,
    ToneThreshold,
    ToneWhite,
    ToneSoftness,
    ToneToe,
    ToneToePivot,
    ToneMix,
    /// Despill: how much comes out, the limit's scale, the luma put back.
    DespillStrength,
    DespillBias,
    DespillRestore,
    /// Lightwrap: band width and background blur (pixels), intensity.
    LightwrapWidth,
    LightwrapBgBlur,
    LightwrapIntensity,
    /// ScreenKey: the screen colour, then its knobs.
    ScreenKeyR,
    ScreenKeyG,
    ScreenKeyB,
    ScreenKeyBalance,
    ScreenKeyGain,
    ScreenKeyClipBlack,
    ScreenKeyClipWhite,
    ScreenKeyDespill,
    ScreenKeyRestore,
    /// ColorCompress: the two levels measured, then the two they land on.
    CompressBlackR,
    CompressBlackG,
    CompressBlackB,
    CompressWhiteR,
    CompressWhiteG,
    CompressWhiteB,
    CompressLiftR,
    CompressLiftG,
    CompressLiftB,
    CompressMultR,
    CompressMultG,
    CompressMultB,
    /// ColorSlice: each vector's center, hue, sat and density, then the
    /// globals. Flat, like every channel: the NAME is what a setup stores.
    SliceRedCenter,
    SliceRedHue,
    SliceRedSat,
    SliceRedDensity,
    SliceSkinCenter,
    SliceSkinHue,
    SliceSkinSat,
    SliceSkinDensity,
    SliceYellowCenter,
    SliceYellowHue,
    SliceYellowSat,
    SliceYellowDensity,
    SliceGreenCenter,
    SliceGreenHue,
    SliceGreenSat,
    SliceGreenDensity,
    SliceCyanCenter,
    SliceCyanHue,
    SliceCyanSat,
    SliceCyanDensity,
    SliceBlueCenter,
    SliceBlueHue,
    SliceBlueSat,
    SliceBlueDensity,
    SliceMagentaCenter,
    SliceMagentaHue,
    SliceMagentaSat,
    SliceMagentaDensity,
    SliceSat,
    SliceDensity,
    SliceSatDepth,
    SliceDenDepth,
    SliceSatBalance,
    DefocusRadius,
    DefocusFocus,
    DefocusFStop,
    DefocusRotation,
    DefocusCatseye,
    DefocusEdge,
    DefocusClamp,
    BlendMix,
    MatteMix,
    GlowSize,
    GlowFalloff,
    GlowThreshold,
    GlowKnee,
    GlowIntensity,
    /// Per-channel tints. They are rgb triples like any other, so they get
    /// lanes, keys and patch jacks rather than a colour-picker button.
    GlowTintR,
    GlowTintG,
    GlowTintB,
    /// Godrays: the source point (normalised), the shafts and their tint.
    GodraysX,
    GodraysY,
    GodraysLength,
    GodraysFalloff,
    GodraysThreshold,
    GodraysKnee,
    GodraysIntensity,
    GodraysTintR,
    GodraysTintG,
    GodraysTintB,
    /// EdgeExtend: how far the colour reaches, how far in it is taken from,
    /// and the ramp on both.
    EdgeExtendDistance,
    EdgeExtendInset,
    EdgeExtendSoftness,
    /// Glint: the key, the star's shape, and what it adds.
    GlintThreshold,
    GlintKnee,
    GlintRays,
    GlintLength,
    GlintRotation,
    GlintFalloff,
    GlintIntensity,
    GlintTintR,
    GlintTintG,
    GlintTintB,
    /// Glint's isolate size (pixels).
    GlintErode,
    /// Glint's dispersion and hue drift.
    GlintSpread,
    GlintShift,
    /// Glint Kernel: the star a Convolve stamps.
    GlintKernelRays,
    GlintKernelLength,
    GlintKernelThickness,
    GlintKernelFalloff,
    GlintKernelRotation,
    GlintKernelAlternate,
    GlintKernelCore,
    GlintKernelCoreSize,
    GlintKernelChroma,
    GlintKernelIntensity,
    GlintKernelTintR,
    GlintKernelTintG,
    GlintKernelTintB,
    /// Bokeh Kernel: the aperture.
    BokehKernelBlades,
    BokehKernelRoundness,
    BokehKernelRotation,
    BokehKernelRadius,
    BokehKernelSoftness,
    BokehKernelCatseye,
    BokehKernelCatseyeAngle,
    BokehKernelRing,
    BokehKernelFringe,
    BokehKernelSqueeze,
    BokehKernelIntensity,
    /// Diffuse: the scatter, the veil, the mist, the depth ramp.
    DiffuseSize,
    DiffuseFalloff,
    DiffuseAmount,
    DiffuseHaze,
    DiffuseMistR,
    DiffuseMistG,
    DiffuseMistB,
    DiffuseNear,
    DiffuseFar,
    LumaLo,
    LumaHi,
    /// TimeRemap: which source frame this timeline frame shows. Animating
    /// THIS is the time remap — its curve in the curve editor is the warp.
    TimeRemapFrame,
    /// A Switch's live input. Whole numbers: a curve through it snaps at the
    /// midpoint, so Hold keys are how you place a cut exactly.
    SwitchWhich,
    /// Spline control point `i`: position and its own feather. Dynamic, so
    /// these serialize as "SplinePtX:3" — JSON object keys must be strings.
    SplinePtX(u16),
    SplinePtY(u16),
    SplinePtF(u16),
    /// Where point `i`'s feather band sits (+ outward), px.
    SplinePtB(u16),
    /// A roto container's shape `s` (>= 1; shape 0 is the Spline* channels),
    /// point `i`: what 0 x, 1 y, 2 feather, 3 bias — the SplinePt* four.
    RotoPt(u16, u16, u8),
    /// A roto container's shape `s` (>= 1), shape-level value `w`: an index
    /// into `ROTO_SHAPE_CHANS`, the shape-0 channel it stands in for.
    RotoSh(u16, u8),
    /// A roto container's axis `k`, transform value `w`: 0..3 translate,
    /// 3..6 rotate, 6..9 scale (`rotodive::AXIS_CHANNELS` inside the dive).
    RotoAx(u16, u8),
    /// The Spline's SHAPE TRANSFORM, which every control point rides on.
    /// A shape that merely moves keys these five instead of two curves per
    /// point — and they are what a Tracker patches into.
    SplineTx,
    SplineTy,
    SplineRot,
    SplineSx,
    SplineSy,
    SplineCx,
    SplineCy,
    /// The Spline's matte controls.
    SplineOpacity,
    SplineExpand,
    SplineFalloff,
    /// Shape-wide feather added outside / inside every point's band.
    SplineFeatherOut,
    SplineFeatherIn,
    /// GridWarp point `i`: where it sits in the SOURCE grid and in the
    /// DESTINATION grid, normalised. Indexed like spline points
    /// ("GwDstX:7"); a re-grid keeps the warp and drops these curves.
    GwSrcX(u16),
    GwSrcY(u16),
    GwDstX(u16),
    GwDstY(u16),
    /// GridWarp's 0..1 blend from no warp to the full one.
    GwAmount,
    /// Warp curve pair `.0`, point `.1`: lane 0/1 its source x/y, 2/3
    /// its destination x/y, normalised ("WarpPt:0.3.2").
    WarpPt(u16, u16, u8),
    /// Warp's falloff (full-res pixels) and its 0..1 amount.
    WarpFalloff,
    WarpAmount,
    /// Colour-keyer blob `i`: its Oklab centre, its three tolerances and its
    /// softness. Indexed, so they serialize as "KeyTolL:2" like the rest.
    KeyCenterL(u16),
    KeyCenterA(u16),
    KeyCenterB(u16),
    KeyTolL(u16),
    KeyTolC(u16),
    KeyTolH(u16),
    KeySoft(u16),
    /// Layer `i` of a Layer stack. Indexed like spline points, so they
    /// serialize as "LayerOpacity:2". The stack ORDER is authored data, not
    /// a curve — reordering renames these, exactly as inserting a spline
    /// point shifts its neighbours.
    ///
    /// LEGACY: the layer transform channels (Tx..Cy). A layer has no transform any more
    /// (an upstream Transform places it); these parse only so an old
    /// setup's curves can be lifted onto that Transform on load
    /// (`EditorGraph::lift_layer_xforms`). No node reports them.
    LayerTx(u16),
    LayerTy(u16),
    LayerRot(u16),
    LayerSx(u16),
    LayerSy(u16),
    LayerCx(u16),
    LayerCy(u16),
    LayerOpacity(u16),
    /// A Recombine string's gain (the harp's horseshoe arcs).
    RecombineGain(u16),
    /// Tinter hue band `i` (0 = red, 60° apart from there).
    /// Socket `k` of a noise-field node — the literal an unwired socket
    /// reads. Indexed rather than one variant per parameter, because which
    /// parameters a field node has depends on its kind.
    Field(u16),
    /// Tinter band `i`'s density (Color Slice's film-print control).
    /// The Mult node's rgb multiplier (one lane each — the widget gangs
    /// them, the curves keep them apart).
    MultR,
    MultG,
    MultB,
    /// The Mix node's crossfade.
    MixAmount,
    /// The Saturation node. `SatShadow`/`SatMid`/`SatHigh` are the
    /// sat-vs-luminance zones; the domain and luma basis are not channels
    /// because they are shader shape, not values.
    /// The Tinter's global row. The per-band ones are indexed above.
    SatAmount,
    SatBoost,
    SatShadow,
    SatMid,
    SatHigh,
    SatPivot,
    SatWidth,
    SatPreserve,
    SatCompress,
    /// The Dilate node's signed pixel amount.
    DilateAmount,
    /// Edge Detect: tap spacing, gain, and the matte's threshold + ramp.
    EdgeSize,
    EdgeGain,
    EdgeThreshold,
    EdgeSoftness,
    /// The Median node's window radius.
    MedianRadius,
    /// Highlight Rebuild: clip level, soft knee, search radius.
    HlrClip,
    HlrSoftness,
    HlrRadius,
    /// LTX HDR: the composite's clip level and soft knee.
    LtxClip,
    LtxSoftness,
    /// Stabilize: the smoothing's sigma in frames, the zoom.
    StabSmooth,
    StabZoom,
    /// The Convolve node's mix (its weights are typed, not channels).
    ConvolveMix,
    /// A Blend's per-input gains (the graph's horseshoe sliders).
    BlendGainA,
    BlendGainB,
    /// LoopOut iterations per frame (rounded, clamped at encode).
    LoopN,
    /// GeoLoad: FBX camera frame = comp frame + offset.
    /// GeoLoad: the absolute FBX frame the projector is frozen at.
    /// The Grade's overall saturation.
    GradeSat,
    /// The Grade's contrast, and the value it turns about.
    GradeContrast,
    GradePivot,
    /// A Grade ZONE's corrections, by zone index: the master's fifteen over
    /// again for one band of the luma axis. See `GradeZones`.
    ZoneLiftR(u16),
    ZoneLiftG(u16),
    ZoneLiftB(u16),
    ZoneGammaR(u16),
    ZoneGammaG(u16),
    ZoneGammaB(u16),
    ZoneGainR(u16),
    ZoneGainG(u16),
    ZoneGainB(u16),
    ZoneOffsetR(u16),
    ZoneOffsetG(u16),
    ZoneOffsetB(u16),
    ZoneSat(u16),
    ZoneContrast(u16),
    ZonePivot(u16),

    /// The Noise generator. `NoiseValue` is READ-ONLY: it is the field
    /// itself, sampled at the probe, which is how a noise graph drives an
    /// ordinary linked parameter.
    NoiseProbeX,
    NoiseProbeY,
    NoiseScale,
    NoiseSeed,
    NoiseSpeed,
    NoiseValue,
    /// The second dimension of a 2-d Noise: READ-ONLY like `NoiseValue`,
    /// the field at the probe with the seed moved by `NoiseSeedOffset`.
    NoiseValueY,
    NoiseSeedOffset,
    /// The 2dini node's seed.
    GeoSeed,
    /// The Displace node.
    /// The Flare generator: the handful worth animating.
    /// The noise's third axis: key it and the pattern boils in place.
    GridSpacingX,
    GridSpacingY,
    GridWidth,
    GridOffsetX,
    GridOffsetY,
    /// The Gradient's axis.
    GradAngle,
    GradCenterX,
    GradCenterY,
    GradScale,
    /// Gradient stop `i`: its position and straight rgba. Indexed like
    /// spline points ("GradPos:2"); removing a stop slides the later
    /// stops' curves down (`nodes::gradient::remove_stop`).
    GradPos(u16),
    GradR(u16),
    GradG(u16),
    GradB(u16),
    GradA(u16),
    /// The Grain generator's knobs; see `nodes::grain`.
    GrainSizeR,
    GrainSizeG,
    GrainSizeB,
    GrainAmountR,
    GrainAmountG,
    GrainAmountB,
    GrainIrrR,
    GrainIrrG,
    GrainIrrB,
    GrainSoftness,
    GrainMono,
    GrainAspect,
    GrainBase,
    GrainShadows,
    GrainMids,
    GrainHighs,
    GrainSeed,
    ReformatScale,
    /// A Tracker's tracked position. The bake writes it as keyframes, so a
    /// patch cable from here drives anything with the motion of the plate.
    TrackX,
    TrackY,
    /// A Tracker's point `i` (1..; point 0 is TrackX/TrackY), keyed by the
    /// bake like point 0. Removing a point slides the later ones' curves
    /// down (`tracker::remove_point`).
    TrackPtX(u16),
    TrackPtY(u16),
    /// Corner `k` (0..4) of the pin the points' motion solves — DERIVED
    /// from the tracked positions at the evaluated frame, never keyed
    /// (`TrackerParams::pins`); a cable from here drives a Transform's pin
    /// or a CornerPin.
    TrackPinX(u16),
    TrackPinY(u16),
    /// A PlanarTrack's motion since its reference frame, keyed by the bake:
    /// a similarity (pixels, pixels, degrees, a factor) for a Transform's
    /// translate/rotate/scale, and the frame's corners under the frame's
    /// homography (normalised) for a Transform's pin or a CornerPin.
    PlanarDX,
    PlanarDY,
    PlanarRot,
    PlanarScale,
    PlanarPin0X,
    PlanarPin0Y,
    PlanarPin1X,
    PlanarPin1Y,
    PlanarPin2X,
    PlanarPin2Y,
    PlanarPin3X,
    PlanarPin3Y,
    /// Colour keyer, node-wide: the ellipsoid's lean and the matte's levels.
    KeyTilt,
    KeyLo,
    KeyHi,
    KeyGamma,
    /// A plugin node's named param. The u16 is a process-lifetime interner
    /// id (name-based, stable across manifest reorder); serializes as
    /// "Custom:<name>" so setups survive interner order changes.
    Custom(u16),
    /// A user Shader node's named param. Same interner as `Custom`; the
    /// name carries a `.<lane>` suffix for the vector types (`tint.1`), so
    /// one Channel still addresses exactly one float.
    Shader(u16),
    Pin0X,
    Pin0Y,
    Pin1X,
    Pin1Y,
    Pin2X,
    Pin2Y,
    Pin3X,
    Pin3Y,
    /// A CornerPin's SOURCE quad (`CornerPinParams::from`), TL TR BR BL.
    From0X,
    From0Y,
    From1X,
    From1Y,
    From2X,
    From2Y,
    From3X,
    From3Y,
}

/// Plugin param names, interned for the process lifetime so Channel stays
/// Copy. Small and append-only; lookups are linear over a handful of names.
fn interner() -> &'static std::sync::RwLock<Vec<String>> {
    static I: std::sync::OnceLock<std::sync::RwLock<Vec<String>>> =
        std::sync::OnceLock::new();
    I.get_or_init(Default::default)
}

pub fn intern(name: &str) -> u16 {
    if let Some(i) = interner().read().unwrap().iter().position(|n| n == name) {
        return i as u16;
    }
    let mut w = interner().write().unwrap();
    // Re-check under the write lock (another thread may have added it).
    if let Some(i) = w.iter().position(|n| n == name) {
        return i as u16;
    }
    w.push(name.to_string());
    (w.len() - 1) as u16
}

pub fn interned(id: u16) -> String {
    interner().read().unwrap().get(id as usize).cloned().unwrap_or_default()
}

impl Channel {
    /// Map key: unit variants use their own name, indexed ones append ":i".
    pub fn key(self) -> String {
        match self {
            Channel::SplinePtX(i) => format!("SplinePtX:{i}"),
            Channel::SplinePtY(i) => format!("SplinePtY:{i}"),
            Channel::SplinePtF(i) => format!("SplinePtF:{i}"),
            Channel::SplinePtB(i) => format!("SplinePtB:{i}"),
            Channel::RotoPt(s, i, w) => format!("RotoPt:{s}.{i}.{w}"),
            Channel::RotoSh(s, w) => format!("RotoSh:{s}.{w}"),
            Channel::RotoAx(k, w) => format!("RotoAx:{k}.{w}"),
            Channel::KeyCenterL(i) => format!("KeyCenterL:{i}"),
            Channel::KeyCenterA(i) => format!("KeyCenterA:{i}"),
            Channel::KeyCenterB(i) => format!("KeyCenterB:{i}"),
            Channel::KeyTolL(i) => format!("KeyTolL:{i}"),
            Channel::KeyTolC(i) => format!("KeyTolC:{i}"),
            Channel::KeyTolH(i) => format!("KeyTolH:{i}"),
            Channel::KeySoft(i) => format!("KeySoft:{i}"),
            Channel::LayerTx(i) => format!("LayerTx:{i}"),
            Channel::LayerTy(i) => format!("LayerTy:{i}"),
            Channel::LayerRot(i) => format!("LayerRot:{i}"),
            Channel::LayerSx(i) => format!("LayerSx:{i}"),
            Channel::LayerSy(i) => format!("LayerSy:{i}"),
            Channel::LayerCx(i) => format!("LayerCx:{i}"),
            Channel::LayerCy(i) => format!("LayerCy:{i}"),
            Channel::LayerOpacity(i) => format!("LayerOpacity:{i}"),
            Channel::RecombineGain(i) => format!("RecombineGain:{i}"),
            Channel::Field(i) => format!("Field:{i}"),
            Channel::ZoneLiftR(i) => format!("ZoneLiftR:{i}"),
            Channel::ZoneLiftG(i) => format!("ZoneLiftG:{i}"),
            Channel::ZoneLiftB(i) => format!("ZoneLiftB:{i}"),
            Channel::ZoneGammaR(i) => format!("ZoneGammaR:{i}"),
            Channel::ZoneGammaG(i) => format!("ZoneGammaG:{i}"),
            Channel::ZoneGammaB(i) => format!("ZoneGammaB:{i}"),
            Channel::ZoneGainR(i) => format!("ZoneGainR:{i}"),
            Channel::ZoneGainG(i) => format!("ZoneGainG:{i}"),
            Channel::ZoneGainB(i) => format!("ZoneGainB:{i}"),
            Channel::ZoneOffsetR(i) => format!("ZoneOffsetR:{i}"),
            Channel::ZoneOffsetG(i) => format!("ZoneOffsetG:{i}"),
            Channel::ZoneOffsetB(i) => format!("ZoneOffsetB:{i}"),
            Channel::ZoneSat(i) => format!("ZoneSat:{i}"),
            Channel::ZoneContrast(i) => format!("ZoneContrast:{i}"),
            Channel::ZonePivot(i) => format!("ZonePivot:{i}"),
            Channel::GradPos(i) => format!("GradPos:{i}"),
            Channel::GwSrcX(i) => format!("GwSrcX:{i}"),
            Channel::WarpPt(p, i, w) => format!("WarpPt:{p}.{i}.{w}"),
            Channel::GwSrcY(i) => format!("GwSrcY:{i}"),
            Channel::GwDstX(i) => format!("GwDstX:{i}"),
            Channel::GwDstY(i) => format!("GwDstY:{i}"),
            Channel::GradR(i) => format!("GradR:{i}"),
            Channel::GradG(i) => format!("GradG:{i}"),
            Channel::GradB(i) => format!("GradB:{i}"),
            Channel::GradA(i) => format!("GradA:{i}"),
            Channel::TrackPtX(i) => format!("TrackPtX:{i}"),
            Channel::TrackPtY(i) => format!("TrackPtY:{i}"),
            Channel::TrackPinX(i) => format!("TrackPinX:{i}"),
            Channel::TrackPinY(i) => format!("TrackPinY:{i}"),
            Channel::Custom(i) => format!("Custom:{}", interned(i)),

            Channel::Shader(i) => format!("Shader:{}", interned(i)),
            other => format!("{other:?}"),
        }
    }

    pub fn from_key(s: &str) -> Option<Channel> {
        if let Some((name, idx)) = s.split_once(':') {
            if name == "Custom" {
                return Some(Channel::Custom(intern(idx)));
            }
            if name == "Shader" {
                return Some(Channel::Shader(intern(idx)));
            }
            if name == "WarpPt" {
                let n: Vec<u16> = idx.split('.').map(|x| x.parse().ok()).collect::<Option<_>>()?;
                return match n.as_slice() {
                    [p, i, w] => Some(Channel::WarpPt(*p, *i, (*w).min(3) as u8)),
                    _ => None,
                };
            }
            if name == "RotoPt" || name == "RotoSh" || name == "RotoAx" {
                let n: Vec<u16> = idx.split('.').map(|x| x.parse().ok()).collect::<Option<_>>()?;
                return match (name, n.as_slice()) {
                    ("RotoAx", [k, w]) => Some(Channel::RotoAx(*k, (*w).min(8) as u8)),
                    ("RotoPt", [s, i, w]) => Some(Channel::RotoPt(*s, *i, (*w).min(3) as u8)),
                    ("RotoSh", [s, w]) => Some(Channel::RotoSh(*s, (*w).min(255) as u8)),
                    _ => None,
                };
            }
            let i: u16 = idx.parse().ok()?;
            return match name {
                "SplinePtX" => Some(Channel::SplinePtX(i)),
                "SplinePtY" => Some(Channel::SplinePtY(i)),
                "SplinePtF" => Some(Channel::SplinePtF(i)),
                "SplinePtB" => Some(Channel::SplinePtB(i)),
                "KeyCenterL" => Some(Channel::KeyCenterL(i)),
                "KeyCenterA" => Some(Channel::KeyCenterA(i)),
                "KeyCenterB" => Some(Channel::KeyCenterB(i)),
                "KeyTolL" => Some(Channel::KeyTolL(i)),
                "KeyTolC" => Some(Channel::KeyTolC(i)),
                "KeyTolH" => Some(Channel::KeyTolH(i)),
                "KeySoft" => Some(Channel::KeySoft(i)),
                "LayerTx" => Some(Channel::LayerTx(i)),
                "LayerTy" => Some(Channel::LayerTy(i)),
                "LayerRot" => Some(Channel::LayerRot(i)),
                "LayerSx" => Some(Channel::LayerSx(i)),
                "LayerSy" => Some(Channel::LayerSy(i)),
                "LayerCx" => Some(Channel::LayerCx(i)),
                "LayerCy" => Some(Channel::LayerCy(i)),
                "LayerOpacity" => Some(Channel::LayerOpacity(i)),
                "RecombineGain" => Some(Channel::RecombineGain(i)),
                "Field" => Some(Channel::Field(i)),
                "ZoneLiftR" => Some(Channel::ZoneLiftR(i)),
                "ZoneLiftG" => Some(Channel::ZoneLiftG(i)),
                "ZoneLiftB" => Some(Channel::ZoneLiftB(i)),
                "ZoneGammaR" => Some(Channel::ZoneGammaR(i)),
                "ZoneGammaG" => Some(Channel::ZoneGammaG(i)),
                "ZoneGammaB" => Some(Channel::ZoneGammaB(i)),
                "ZoneGainR" => Some(Channel::ZoneGainR(i)),
                "ZoneGainG" => Some(Channel::ZoneGainG(i)),
                "ZoneGainB" => Some(Channel::ZoneGainB(i)),
                "ZoneOffsetR" => Some(Channel::ZoneOffsetR(i)),
                "ZoneOffsetG" => Some(Channel::ZoneOffsetG(i)),
                "ZoneOffsetB" => Some(Channel::ZoneOffsetB(i)),
                "ZoneSat" => Some(Channel::ZoneSat(i)),
                "ZoneContrast" => Some(Channel::ZoneContrast(i)),
                "ZonePivot" => Some(Channel::ZonePivot(i)),
                "GradPos" => Some(Channel::GradPos(i)),
                // Before the prune a GridWarp had one lattice, the
                // source positions, keyed as GwPt.
                "GwSrcX" | "GwPtX" => Some(Channel::GwSrcX(i)),
                "GwSrcY" | "GwPtY" => Some(Channel::GwSrcY(i)),
                "GwDstX" => Some(Channel::GwDstX(i)),
                "GwDstY" => Some(Channel::GwDstY(i)),
                "GradR" => Some(Channel::GradR(i)),
                "GradG" => Some(Channel::GradG(i)),
                "GradB" => Some(Channel::GradB(i)),
                "GradA" => Some(Channel::GradA(i)),
                "TrackPtX" => Some(Channel::TrackPtX(i)),
                "TrackPtY" => Some(Channel::TrackPtY(i)),
                "TrackPinX" => Some(Channel::TrackPinX(i)),
                "TrackPinY" => Some(Channel::TrackPinY(i)),
                _ => None,

            };
        }
        // The Mult node's multiplier was a single float for a day; a
        // curve on it lands on red, where a ganged edit spreads it back.
        if s == "MultValue" {
            return Some(Channel::MultR);
        }
        ALL.iter().copied().chain(crate::nodes::channels()).find(|c| format!("{c:?}") == s)
    }
}

impl serde::Serialize for Channel {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.key())
    }
}

impl<'de> serde::Deserialize<'de> for Channel {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Channel::from_key(&s)
            .ok_or_else(|| serde::de::Error::custom(format!("unknown channel {s}")))
    }
}

/// Animation maps tolerate channels this build doesn't know (a setup from a
/// newer build, or one naming a channel that has since been retired): the
/// unknown tracks are dropped instead of failing the whole load.
pub fn de_anim_map<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<std::collections::BTreeMap<Channel, AnimCurve>, D::Error> {
    use serde::Deserialize as _;
    let raw = <std::collections::BTreeMap<String, AnimCurve>>::deserialize(d)?;
    let mut out = std::collections::BTreeMap::new();
    for (k, v) in raw {
        match Channel::from_key(&k) {
            Some(c) => {
                out.insert(c, v);
            }
            None => log::warn!("dropping unknown animation channel {k}"),
        }
    }
    Ok(out)
}

/// Is this channel locked against keyframing?
pub fn is_locked(
    anim: &std::collections::BTreeMap<Channel, AnimCurve>,
    ch: Channel,
) -> bool {
    anim.get(&ch).is_some_and(|c| c.locked)
}

/// Any of these channels locked? A ganged control (one number, several
/// channels) is locked when any of its channels is.
pub fn any_locked(
    anim: &std::collections::BTreeMap<Channel, AnimCurve>,
    chans: &[Channel],
) -> bool {
    chans.iter().any(|c| is_locked(anim, *c))
}

/// Lock or unlock a channel. Locking a static channel leaves an EMPTY curve
/// behind to carry the flag; unlocking one drops it again, so a lock that
/// was only ever a lock leaves nothing in the setup.
pub fn set_locked(
    anim: &mut std::collections::BTreeMap<Channel, AnimCurve>,
    ch: Channel,
    on: bool,
) {
    if on {
        anim.entry(ch).or_default().locked = true;
    } else if let Some(c) = anim.get_mut(&ch) {
        c.locked = false;
        if c.keys.is_empty() {
            anim.remove(&ch);
        }
    }
}

/// How a segment LEAVES a key.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize,
)]
pub enum Interp {
    /// Step: hold this key's value until the next one.
    Hold,
    #[default]
    Linear,
    /// Auto-smoothed: Catmull-Rom tangents, flattened at local extrema and
    /// at the curve's ends, and slope-limited — so it eases without ever
    /// overshooting past neighbouring keys.
    Smooth,
    /// Free handles: this key's `tan_out` and the next key's `tan_in`.
    Bezier,
    /// Blender's easing presets: a fixed shape between the two values,
    /// Penner's curves (`ease_shape`). Back overshoots, Bounce and Elastic
    /// ring — on purpose, for motion graphics.
    Ease(Ease, EaseDir),
}

/// The easing family of an `Interp::Ease` segment (Blender's names).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Ease {
    Sine,
    Quad,
    Cubic,
    Quart,
    Quint,
    Expo,
    Circ,
    Back,
    Bounce,
    Elastic,
}

impl Ease {
    pub const ALL: [Ease; 10] = [
        Ease::Sine,
        Ease::Quad,
        Ease::Cubic,
        Ease::Quart,
        Ease::Quint,
        Ease::Expo,
        Ease::Circ,
        Ease::Back,
        Ease::Bounce,
        Ease::Elastic,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Ease::Sine => "sine",
            Ease::Quad => "quad",
            Ease::Cubic => "cubic",
            Ease::Quart => "quart",
            Ease::Quint => "quint",
            Ease::Expo => "expo",
            Ease::Circ => "circ",
            Ease::Back => "back",
            Ease::Bounce => "bounce",
            Ease::Elastic => "elastic",
        }
    }
    /// The ease-IN shape, 0 at 0 and 1 at 1.
    fn ease_in(self, u: f32) -> f32 {
        use std::f32::consts::PI;
        match self {
            Ease::Sine => 1.0 - (u * PI * 0.5).cos(),
            Ease::Quad => u * u,
            Ease::Cubic => u * u * u,
            Ease::Quart => u * u * u * u,
            Ease::Quint => u * u * u * u * u,
            Ease::Expo => {
                if u <= 0.0 {
                    0.0
                } else {
                    2f32.powf(10.0 * (u - 1.0))
                }
            }
            Ease::Circ => 1.0 - (1.0 - u * u).max(0.0).sqrt(),
            Ease::Back => {
                const S: f32 = 1.70158;
                u * u * ((S + 1.0) * u - S)
            }
            Ease::Bounce => 1.0 - bounce_out(1.0 - u),
            Ease::Elastic => {
                if u <= 0.0 || u >= 1.0 {
                    return u.clamp(0.0, 1.0);
                }
                const P: f32 = 0.3;
                -(2f32.powf(10.0 * (u - 1.0))) * ((u - 1.0 - P / 4.0) * (2.0 * PI) / P).sin()
            }
        }
    }
}

fn bounce_out(u: f32) -> f32 {
    const N: f32 = 7.5625;
    const D: f32 = 2.75;
    if u < 1.0 / D {
        N * u * u
    } else if u < 2.0 / D {
        let u = u - 1.5 / D;
        N * u * u + 0.75
    } else if u < 2.5 / D {
        let u = u - 2.25 / D;
        N * u * u + 0.9375
    } else {
        let u = u - 2.625 / D;
        N * u * u + 0.984375
    }
}

/// Which end of an `Interp::Ease` segment the easing is on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum EaseDir {
    In,
    Out,
    InOut,
}

impl EaseDir {
    pub const ALL: [EaseDir; 3] = [EaseDir::In, EaseDir::Out, EaseDir::InOut];
    pub fn name(self) -> &'static str {
        match self {
            EaseDir::In => "in",
            EaseDir::Out => "out",
            EaseDir::InOut => "in-out",
        }
    }
}

/// Penner's easing at `u` in 0..1 (0 → 0, 1 → 1; Back / Elastic leave
/// that range in between).
pub fn ease_shape(e: Ease, d: EaseDir, u: f32) -> f32 {
    let u = u.clamp(0.0, 1.0);
    match d {
        EaseDir::In => e.ease_in(u),
        EaseDir::Out => 1.0 - e.ease_in(1.0 - u),
        EaseDir::InOut => {
            if u < 0.5 {
                0.5 * e.ease_in(2.0 * u)
            } else {
                1.0 - 0.5 * e.ease_in(2.0 - 2.0 * u)
            }
        }
    }
}

/// What a curve does PAST its first or last key: Blender's extrapolation
/// (constant / linear / the Cycles modifier), Houdini's extend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Extend {
    /// Keep the end key's value (what every curve did before).
    #[default]
    Hold,
    /// Carry on along the end's slope.
    Linear,
    /// Repeat the keyed span.
    Cycle,
    /// Repeat it, each repeat starting where the last one ended — a ramp
    /// that keeps climbing.
    CycleOffset,
    /// Repeat it back and forth.
    Oscillate,
}

impl Extend {
    pub const ALL: [Extend; 5] =
        [Extend::Hold, Extend::Linear, Extend::Cycle, Extend::CycleOffset, Extend::Oscillate];
    pub fn name(self) -> &'static str {
        match self {
            Extend::Hold => "hold",
            Extend::Linear => "linear",
            Extend::Cycle => "cycle",
            Extend::CycleOffset => "cycle + offset",
            Extend::Oscillate => "oscillate",
        }
    }
    fn is_hold(&self) -> bool {
        *self == Extend::Hold
    }
}

/// Blender's Stepped modifier: the curve is read only every `step` frames
/// and held in between — animation on twos without re-keying.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Stepped {
    /// Frames per held value.
    pub step: f32,
    /// Where the steps start, in frames.
    pub offset: f32,
}

impl Default for Stepped {
    fn default() -> Self {
        Self { step: 2.0, offset: 0.0 }
    }
}

impl Stepped {
    /// The frame `t` actually reads.
    pub fn at(&self, t: f32) -> f32 {
        let s = self.step.abs().max(1e-3);
        ((t - self.offset) / s).floor() * s + self.offset
    }
}

/// Anything a key can be addressed at: a whole timeline frame (every
/// autokey / gizmo / bake path) or a fractional one (the curve editor with
/// its snap off). Lets `set_key(f, v)` take either without a cast at every
/// call site.
pub trait AtFrame: Copy {
    fn at(self) -> f64;
}
macro_rules! at_frame {
    ($($t:ty),*) => {$(impl AtFrame for $t { fn at(self) -> f64 { self as f64 } })*};
}
at_frame!(u64, u32, i32, i64, usize, f32, f64);

/// A key's frame as a hashable identity — selections are sets of these.
/// Exact bits: two keys on one curve never share a frame. A newtype, not a
/// bare u64, so an id can never be passed where a FRAME is meant (u64 is an
/// `AtFrame`). Frames are never negative, so the bits order like frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FrameId(u64);

/// `+ 0.0` folds -0.0 into 0.0 so one frame has one id.
pub fn frame_id(f: f64) -> FrameId {
    FrameId((f + 0.0).to_bits())
}
pub fn id_frame(id: FrameId) -> f64 {
    f64::from_bits(id.0)
}

/// Whole frames serialize as integers (the pre-sub-frame format, and what
/// older builds read); only a genuinely fractional key writes a float.
fn ser_frame<S: serde::Serializer>(f: &f64, s: S) -> Result<S::Ok, S::Error> {
    if f.fract() == 0.0 && *f >= 0.0 && *f < u64::MAX as f64 {
        s.serialize_u64(*f as u64)
    } else {
        s.serialize_f64(*f)
    }
}

/// One keyframe. Handles are offsets from the key in (frames, value units).
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct Key {
    #[serde(serialize_with = "ser_frame")]
    pub frame: f64,
    pub value: f32,
    pub interp: Interp,
    pub tan_in: (f32, f32),
    pub tan_out: (f32, f32),
    /// BROKEN tangent: the two handles move independently. Unbroken (the
    /// default) the curve editor keeps them collinear, so the key passes
    /// through smoothly — dragging one swings the other.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub broken: bool,
    /// VECTOR handles (Blender's): the in / out handle points a third of
    /// the way at the neighbouring key, and keeps doing so as keys move —
    /// derived at evaluation (`AnimCurve::in_handle` / `out_handle`), so
    /// `tan_in` / `tan_out` are ignored while set. Dragging the handle
    /// clears it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub vec_in: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub vec_out: bool,
}

impl Key {
    pub fn new(frame: impl AtFrame, value: f32) -> Self {
        Self {
            frame: frame.at(),
            value,
            interp: Interp::default(),
            tan_in: (0.0, 0.0),
            tan_out: (0.0, 0.0),
            broken: false,
            vec_in: false,
            vec_out: false,
        }
    }
}

/// Setups written before interpolation modes stored keys as `[frame, value]`
/// pairs; those still load (as Linear) alongside the current object form.
#[derive(serde::Deserialize)]
#[serde(untagged)]
enum KeyRepr {
    Pair(f64, f32),
    Full {
        frame: f64,
        value: f32,
        #[serde(default)]
        interp: Interp,
        #[serde(default)]
        tan_in: (f32, f32),
        #[serde(default)]
        tan_out: (f32, f32),
        #[serde(default)]
        broken: bool,
        #[serde(default)]
        vec_in: bool,
        #[serde(default)]
        vec_out: bool,
    },
}

impl<'de> serde::Deserialize<'de> for Key {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Ok(match KeyRepr::deserialize(d)? {
            KeyRepr::Pair(frame, value) => Key::new(frame, value),
            KeyRepr::Full { frame, value, interp, tan_in, tan_out, broken, vec_in, vec_out } => {
                Key { frame, value, interp, tan_in, tan_out, broken, vec_in, vec_out }
            }
        })
    }
}

fn hermite(v0: f32, v1: f32, m0: f32, m1: f32, u: f32) -> f32 {
    let (u2, u3) = (u * u, u * u * u);
    (2.0 * u3 - 3.0 * u2 + 1.0) * v0
        + (u3 - 2.0 * u2 + u) * m0
        + (-2.0 * u3 + 3.0 * u2) * v1
        + (u3 - u2) * m1
}

/// A value track: keys sorted by frame, each carrying the interpolation used
/// to reach the next one. Values hold before the first key and after the last.
#[derive(Clone, Debug, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct AnimCurve {
    pub keys: Vec<Key>,
    /// LOCKED: this channel takes no keyframes. Nothing that writes a key —
    /// a properties row, a gizmo, a HUD slider, autokey, the curve editor —
    /// may touch it, so one axis of a transform can be pinned while the
    /// other animates. The flag lives on the curve rather than beside it so
    /// every write path already has it in hand (they all hold the node's
    /// anim map), and a locked channel with no keys is simply an empty
    /// curve — which every "is it animated?" test already reads as static.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub locked: bool,
    /// Blender's Noise modifier: a wiggle ADDED to what the keys say, in
    /// `eval_at`, so every reader sees it and the keys never move.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub noise: Option<crate::animnoise::CurveNoise>,
    /// Before the first key / after the last (`Extend::Hold`: the old,
    /// still default, behaviour). Part of the keys' own curve, so
    /// `eval_base_at` applies it.
    #[serde(default, skip_serializing_if = "Extend::is_hold")]
    pub pre: Extend,
    #[serde(default, skip_serializing_if = "Extend::is_hold")]
    pub post: Extend,
    /// Blender's Stepped modifier: the whole channel — keys AND noise — is
    /// read on a lattice of frames, so it holds in between.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stepped: Option<Stepped>,
}

impl AnimCurve {
    /// Evaluate at an integer timeline frame (what playback and the engine
    /// graph use).
    pub fn eval(&self, t: u64) -> Option<f32> {
        self.eval_at(t as f32)
    }

    /// Evaluate at a fractional frame: the keys, plus the noise when the
    /// curve has one, read through the Stepped modifier. What the channel
    /// READS.
    pub fn eval_at(&self, t: f32) -> Option<f32> {
        let t = self.stepped.map_or(t, |s| s.at(t));
        self.eval_base_at(t).map(|v| v + self.noise_at(t))
    }

    /// True when what the channel reads differs from the keys' own curve
    /// (the editor then draws both).
    pub fn modified(&self) -> bool {
        self.noise.is_some() || self.stepped.is_some()
    }

    /// The effective OUT handle of key `i` (a vector handle is derived from
    /// its neighbour; `tan_out` otherwise).
    pub fn out_handle(&self, i: usize) -> (f32, f32) {
        let k = &self.keys[i];
        match self.keys.get(i + 1) {
            Some(n) if k.vec_out => {
                (((n.frame - k.frame) / 3.0) as f32, (n.value - k.value) / 3.0)
            }
            _ => k.tan_out,
        }
    }

    /// The effective IN handle of key `i`.
    pub fn in_handle(&self, i: usize) -> (f32, f32) {
        let k = &self.keys[i];
        match i.checked_sub(1).and_then(|j| self.keys.get(j)) {
            Some(p) if k.vec_in => {
                (((p.frame - k.frame) / 3.0) as f32, (p.value - k.value) / 3.0)
            }
            _ => k.tan_in,
        }
    }

    /// The slope (value per frame) the curve leaves its LAST key with, and
    /// arrives at its FIRST with — what linear extrapolation carries on.
    /// A bezier end follows its handle, a hold end is flat, anything else
    /// follows the end segment's chord.
    fn end_slope(&self, last: bool) -> f32 {
        let n = self.keys.len();
        if n < 2 {
            return 0.0;
        }
        let (a, b) = if last { (n - 2, n - 1) } else { (0, 1) };
        let (ka, kb) = (&self.keys[a], &self.keys[b]);
        let chord = (kb.value - ka.value) / ((kb.frame - ka.frame).max(1e-6)) as f32;
        match ka.interp {
            Interp::Hold => 0.0,
            Interp::Bezier => {
                let h = if last { self.in_handle(b) } else { self.out_handle(a) };
                if h.0.abs() > 1e-6 {
                    h.1 / h.0
                } else {
                    chord
                }
            }
            _ => chord,
        }
    }

    /// What the noise adds at `t` (0 without one).
    pub fn noise_at(&self, t: f32) -> f32 {
        self.noise.as_ref().map_or(0.0, |n| n.at(t))
    }

    /// The keys alone, without the noise — the curve the keys sit on. The
    /// curve editor samples segments with this so the drawn shape is the
    /// real one.
    pub fn eval_base_at(&self, t: f32) -> Option<f32> {
        let tt = t as f64;
        let first = self.keys.first()?;
        let last = self.keys.last()?;
        let (f0, f1) = (first.frame, last.frame);
        let (mode, edge) = if tt < f0 {
            (self.pre, first)
        } else if tt > f1 {
            (self.post, last)
        } else {
            return self.eval_inside(t);
        };
        let span = f1 - f0;
        Some(match mode {
            Extend::Hold => edge.value,
            Extend::Linear => edge.value + self.end_slope(tt > f1) * (tt - edge.frame) as f32,
            _ if span <= 1e-9 => edge.value,
            Extend::Cycle | Extend::CycleOffset | Extend::Oscillate => {
                let n = ((tt - f0) / span).floor();
                let mut inner = tt - n * span;
                if mode == Extend::Oscillate && (n as i64).rem_euclid(2) == 1 {
                    inner = f1 - (inner - f0);
                }
                let v = self.eval_inside(inner as f32)?;
                if mode == Extend::CycleOffset {
                    v + n as f32 * (last.value - first.value)
                } else {
                    v
                }
            }
        })
    }

    /// The keyed span alone (`t` clamped into it).
    fn eval_inside(&self, t: f32) -> Option<f32> {
        let t = t as f64;
        let first = self.keys.first()?;
        if t <= first.frame {
            return Some(first.value);
        }
        let last = self.keys.last()?;
        if t >= last.frame {
            return Some(last.value);
        }
        let i = self.keys.partition_point(|k| k.frame <= t);
        let (k0, k1) = (&self.keys[i - 1], &self.keys[i]);
        let dt = (k1.frame - k0.frame).max(1e-6);
        let u = ((t - k0.frame) / dt).clamp(0.0, 1.0) as f32;
        Some(match k0.interp {
            Interp::Hold => k0.value,
            Interp::Linear => k0.value + (k1.value - k0.value) * u,
            Interp::Smooth => {
                let (m0, m1) = (self.slope_at(i - 1), self.slope_at(i));
                hermite(k0.value, k1.value, m0 * dt as f32, m1 * dt as f32, u)
            }
            Interp::Bezier => {
                bezier_eval(k0, k1, self.out_handle(i - 1), self.in_handle(i), t as f32)
            }
            Interp::Ease(e, d) => k0.value + (k1.value - k0.value) * ease_shape(e, d, u),
        })
    }

    /// Auto-tangent (value per frame) at a key: Catmull-Rom, flattened at
    /// local extrema and Fritsch-Carlson limited so `Smooth` stays tame.
    pub fn slope_at(&self, i: usize) -> f32 {
        let Some(k) = self.keys.get(i) else { return 0.0 };
        let prev = i.checked_sub(1).and_then(|j| self.keys.get(j));
        let next = self.keys.get(i + 1);
        let slope =
            |a: &Key, b: &Key| (b.value - a.value) / (b.frame - a.frame).max(1e-3) as f32;
        match (prev, next) {
            (Some(p), Some(n)) => {
                let (sp, sn) = (slope(p, k), slope(k, n));
                if sp * sn <= 0.0 {
                    0.0 // local extremum: flat, so the ease can't overshoot
                } else {
                    let lim = 3.0 * sp.abs().min(sn.abs());
                    (0.5 * (sp + sn)).clamp(-lim, lim)
                }
            }
            // Curve ends flatten, so two smoothed keys read as a proper
            // ease instead of a straight line (chord tangents would make
            // Smooth identical to Linear there).
            (None, Some(_)) | (Some(_), None) | (None, None) => 0.0,
        }
    }

    /// Where the key at `frame` sits, if there is one.
    pub fn index_of(&self, frame: impl AtFrame) -> Option<usize> {
        let f = frame.at();
        self.keys.binary_search_by(|k| k.frame.total_cmp(&f)).ok()
    }

    /// Key so the channel READS `shown` at `frame`: the noise there comes
    /// back off first. For writes made from an evaluated value (a property
    /// row, a gizmo, a HUD) — `set_key` stores the key value itself.
    pub fn set_key_shown(&mut self, frame: impl AtFrame, shown: f32) {
        let f = frame.at();
        let n = self.noise_at(f as f32);
        self.set_key(f, shown - n);
    }

    /// Insert or replace the key at `frame`, keeping keys sorted. A new key
    /// inherits the interpolation of the segment it lands in, so autokeying
    /// into a smoothed curve doesn't punch a linear corner in it.
    pub fn set_key(&mut self, frame: impl AtFrame, value: f32) {
        let f = frame.at() + 0.0;
        match self.keys.binary_search_by(|k| k.frame.total_cmp(&f)) {
            Ok(i) => self.keys[i].value = value,
            Err(i) => {
                let interp = i
                    .checked_sub(1)
                    .and_then(|j| self.keys.get(j))
                    .or_else(|| self.keys.get(i))
                    .map(|k| k.interp)
                    .unwrap_or_default();
                self.keys.insert(i, Key { interp, ..Key::new(f, value) });
            }
        }
    }

    /// Set the interpolation leaving `frame`. Switching to Bezier seeds the
    /// handles from the curve's current slope, so the shape doesn't jump.
    pub fn set_interp(&mut self, frame: impl AtFrame, interp: Interp) {
        let Some(i) = self.index_of(frame) else { return };
        if interp == Interp::Bezier {
            let m = self.slope_at(i);
            let f = self.keys[i].frame;
            let out_dt =
                self.keys.get(i + 1).map(|n| (n.frame - f) as f32).unwrap_or(10.0) / 3.0;
            let in_dt = i
                .checked_sub(1)
                .and_then(|j| self.keys.get(j))
                .map(|p| (f - p.frame) as f32)
                .unwrap_or(10.0)
                / 3.0;
            let k = &mut self.keys[i];
            k.tan_out = (out_dt, m * out_dt);
            k.tan_in = (-in_dt, -m * in_dt);
        }
        self.keys[i].interp = interp;
    }

    /// Set the interpolation on BOTH sides of the key at `frame`: the
    /// segment leaving it and the one arriving (which the previous key
    /// owns). What "make this key smooth" means to a hand — a key eased on
    /// one side only reads as a kink. The first key has no arriving side.
    pub fn set_interp_both(&mut self, frame: impl AtFrame, interp: Interp) {
        let Some(i) = self.index_of(frame) else { return };
        if let Some(prev) = i.checked_sub(1).map(|j| self.keys[j].frame) {
            self.set_interp(prev, interp);
        }
        self.set_interp(self.keys[i].frame, interp);
    }

    /// Make both sides of the key at `frame` bezier WITHOUT touching a
    /// side that already is: an existing handle keeps its shape, a new one
    /// is seeded from the curve's slope (as `set_interp` would). What the
    /// tangent keys need before they can align or break anything.
    pub fn ensure_bezier(&mut self, frame: impl AtFrame) {
        let Some(i) = self.index_of(frame) else { return };
        let f = self.keys[i].frame;
        let m = self.slope_at(i);
        if self.keys[i].interp != Interp::Bezier {
            let dt = self.keys.get(i + 1).map_or(10.0, |n| (n.frame - f) as f32) / 3.0;
            self.keys[i].interp = Interp::Bezier;
            self.keys[i].tan_out = (dt, m * dt);
        }
        if let Some(j) = i.checked_sub(1) {
            if self.keys[j].interp != Interp::Bezier {
                let mp = self.slope_at(j);
                let dt = (f - self.keys[j].frame) as f32 / 3.0;
                self.keys[j].interp = Interp::Bezier;
                self.keys[j].tan_out = (dt, mp * dt);
                self.keys[i].tan_in = (-dt, -m * dt);
            }
        } else if self.keys[i].tan_in == (0.0, 0.0) {
            // First key: no arriving segment, but a handle to align against.
            let dt = self.keys[i].tan_out.0.max(1.0);
            self.keys[i].tan_in = (-dt, -m * dt);
        }
    }

    pub fn remove_key(&mut self, frame: impl AtFrame) {
        let f = frame.at();
        self.keys.retain(|k| k.frame != f);
    }

    pub fn has_key(&self, frame: impl AtFrame) -> bool {
        self.index_of(frame).is_some()
    }
}

/// Cubic bezier through (k0, k0+tan_out, k1+tan_in, k1), solved for the
/// parameter whose x is `t`. Handle x offsets are clamped into the segment
/// so x stays monotonic and the solve always converges.
fn bezier_eval(k0: &Key, k1: &Key, out: (f32, f32), inn: (f32, f32), t: f32) -> f32 {
    let (f0, f1) = (k0.frame as f32, k1.frame as f32);
    let dt = (f1 - f0).max(1e-6);
    let x1 = f0 + out.0.clamp(0.0, dt);
    let x2 = f1 + inn.0.clamp(-dt, 0.0);
    let (y0, y1) = (k0.value, k0.value + out.1);
    let (y2, y3) = (k1.value + inn.1, k1.value);
    let cubic = |a: f32, b: f32, c: f32, d: f32, s: f32| {
        let r = 1.0 - s;
        r * r * r * a + 3.0 * r * r * s * b + 3.0 * r * s * s * c + s * s * s * d
    };
    let (mut lo, mut hi) = (0.0f32, 1.0f32);
    for _ in 0..24 {
        let mid = 0.5 * (lo + hi);
        if cubic(f0, x1, x2, f1, mid) < t {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    cubic(y0, y1, y2, y3, 0.5 * (lo + hi))
}

/// Channel groups per node kind (gizmo write-back + key capture).
pub const TRANSFORM: &[Channel] = &[
    Channel::TranslateX,
    Channel::TranslateY,
    Channel::Rotate,
    Channel::ScaleX,
    Channel::ScaleY,
    Channel::CenterX,
    Channel::CenterY,
    Channel::ShearX,
    Channel::ShearY,
    Channel::Taper,
    Channel::TaperAngle,
    // The Transform's corner pin shares the CornerPin's channels: the same
    // four corners, the same curve-editor names.
    Channel::Pin0X,
    Channel::Pin0Y,
    Channel::Pin1X,
    Channel::Pin1Y,
    Channel::Pin2X,
    Channel::Pin2Y,
    Channel::Pin3X,
    Channel::Pin3Y,
];
/// Every channel belonging to layer `i` of a stack — what a restack has to
/// carry with the layer, and what deleting one has to take away.
pub fn layer_channels(i: u16) -> [Channel; 1] {
    [Channel::LayerOpacity(i)]
}

/// The LEGACY transform channels of layer `i`, in `TRANSFORM` order — what
/// the load-time lift moves onto the Transform it splices in.
pub fn legacy_layer_transform(i: u16) -> [Channel; 7] {
    [
        Channel::LayerTx(i),
        Channel::LayerTy(i),
        Channel::LayerRot(i),
        Channel::LayerSx(i),
        Channel::LayerSy(i),
        Channel::LayerCx(i),
        Channel::LayerCy(i),
    ]
}

pub const GRADE: &[Channel] = &[
    Channel::LiftR,
    Channel::LiftG,
    Channel::LiftB,
    Channel::GammaR,
    Channel::GammaG,
    Channel::GammaB,
    Channel::GainR,
    Channel::GainG,
    Channel::GainB,
    Channel::OffsetR,
    Channel::OffsetG,
    Channel::OffsetB,
    Channel::GradeSat,
    Channel::GradeContrast,
    Channel::GradePivot,
];

/// The fifteen channels of ONE Grade zone — `GRADE`, per band.
pub fn zone_channels(i: u16) -> [Channel; 15] {
    [
        Channel::ZoneLiftR(i),
        Channel::ZoneLiftG(i),
        Channel::ZoneLiftB(i),
        Channel::ZoneGammaR(i),
        Channel::ZoneGammaG(i),
        Channel::ZoneGammaB(i),
        Channel::ZoneGainR(i),
        Channel::ZoneGainG(i),
        Channel::ZoneGainB(i),
        Channel::ZoneOffsetR(i),
        Channel::ZoneOffsetG(i),
        Channel::ZoneOffsetB(i),
        Channel::ZoneSat(i),
        Channel::ZoneContrast(i),
        Channel::ZonePivot(i),
    ]
}

/// The Grade's channels for one page of its panel: the master's (`None`)
/// or a zone's. One instrument, one list, whichever tab is up.
pub fn grade_channels(zone: Option<u16>) -> Vec<Channel> {
    match zone {
        None => GRADE.to_vec(),
        Some(i) => zone_channels(i).to_vec(),
    }
}

/// The rgb triple of row `row` (0 lift, 1 gamma, 2 gain, 3 offset) on the
/// master or on a zone.
pub fn grade_triple(zone: Option<u16>, row: usize) -> [Channel; 3] {
    let all = grade_channels(zone);
    let k = row.min(3) * 3;
    [all[k], all[k + 1], all[k + 2]]
}

/// The saturation channel of the master or of a zone.
pub fn grade_sat(zone: Option<u16>) -> Channel {
    match zone {
        None => Channel::GradeSat,
        Some(i) => Channel::ZoneSat(i),
    }
}

/// The contrast and pivot channels of the master or of a zone.
pub fn grade_contrast(zone: Option<u16>) -> [Channel; 2] {
    match zone {
        None => [Channel::GradeContrast, Channel::GradePivot],
        Some(i) => [Channel::ZoneContrast(i), Channel::ZonePivot(i)],
    }
}

pub const PINS: &[Channel] = &[
    Channel::Pin0X,
    Channel::Pin0Y,
    Channel::Pin1X,
    Channel::Pin1Y,
    Channel::Pin2X,
    Channel::Pin2Y,
    Channel::Pin3X,
    Channel::Pin3Y,
];

/// A CornerPin's `from` quad, laid out like `PINS`.
pub const FROMS: &[Channel] = &[
    Channel::From0X,
    Channel::From0Y,
    Channel::From1X,
    Channel::From1Y,
    Channel::From2X,
    Channel::From2Y,
    Channel::From3X,
    Channel::From3Y,
];

/// Every channel not owned by a node module (those come in through
/// `nodes::channels()`). The properties post-pass sweeps the union; get()
/// returns None for kinds a channel doesn't apply to, so the sweep is cheap
/// and safe.
pub const ALL: &[Channel] = &[
    Channel::TranslateX,
    Channel::TranslateY,
    Channel::Rotate,
    Channel::ScaleX,
    Channel::ScaleY,
    Channel::CenterX,
    Channel::CenterY,
    Channel::ShearX,
    Channel::ShearY,
    Channel::Taper,
    Channel::TaperAngle,
    Channel::LiftR,
    Channel::LiftG,
    Channel::LiftB,
    Channel::GammaR,
    Channel::GammaG,
    Channel::GammaB,
    Channel::GainR,
    Channel::GainG,
    Channel::GainB,
    Channel::OffsetR,
    Channel::OffsetG,
    Channel::OffsetB,
    Channel::DefocusRadius,
    Channel::DefocusFocus,
    Channel::DefocusFStop,
    Channel::DefocusRotation,
    Channel::DefocusCatseye,
    Channel::DefocusEdge,
    Channel::DefocusClamp,
    Channel::BlendMix,
    Channel::MatteMix,
    Channel::LumaLo,
    Channel::LumaHi,
    Channel::TimeRemapFrame,
    Channel::SwitchWhich,
    Channel::Pin0X,
    Channel::Pin0Y,
    Channel::Pin1X,
    Channel::Pin1Y,
    Channel::Pin2X,
    Channel::Pin2Y,
    Channel::Pin3X,
    Channel::Pin3Y,
    Channel::From0X,
    Channel::From0Y,
    Channel::From1X,
    Channel::From1Y,
    Channel::From2X,
    Channel::From2Y,
    Channel::From3X,
    Channel::From3Y,
    Channel::MultR,
    Channel::MultG,
    Channel::MultB,
    Channel::MixAmount,
    Channel::SplineTx,
    Channel::SplineTy,
    Channel::SplineRot,
    Channel::SplineSx,
    Channel::SplineSy,
    Channel::SplineCx,
    Channel::SplineCy,
    Channel::SplineOpacity,
    Channel::SplineExpand,
    Channel::SplineFalloff,
    Channel::SplineFeatherOut,
    Channel::SplineFeatherIn,
    Channel::BlendGainA,
    Channel::BlendGainB,
    Channel::LoopN,
    Channel::GradeSat,
    Channel::GradeContrast,
    Channel::GradePivot,
    Channel::NoiseProbeX,
    Channel::NoiseProbeY,
    Channel::NoiseScale,
    Channel::NoiseSeed,
    Channel::NoiseSpeed,
    Channel::NoiseValue,
    Channel::NoiseValueY,
    Channel::NoiseSeedOffset,
    Channel::GeoSeed,
    Channel::GridSpacingX,
    Channel::GridSpacingY,
    Channel::GridWidth,
    Channel::GridOffsetX,
    Channel::GridOffsetY,
    Channel::ReformatScale,
    Channel::TrackX,
    Channel::TrackY,
    Channel::PlanarDX,
    Channel::PlanarDY,
    Channel::PlanarRot,
    Channel::PlanarScale,
    Channel::PlanarPin0X,
    Channel::PlanarPin0Y,
    Channel::PlanarPin1X,
    Channel::PlanarPin1Y,
    Channel::PlanarPin2X,
    Channel::PlanarPin2Y,
    Channel::PlanarPin3X,
    Channel::PlanarPin3Y,
    Channel::KeyTilt,
    Channel::KeyLo,
    Channel::KeyHi,
    Channel::KeyGamma,
];

/// A spline animates as ONE shape: every control point's position keys on
/// the same frame. Points that key apart drift into a different shape
/// between keys, so until a point can carry its own timing, these two
/// groups — position and feather — are what the key dots and the viewer
/// gizmo write back.
pub fn spline_shape(n_points: usize) -> Vec<Channel> {
    (0..n_points.min(u16::MAX as usize) as u16)
        .flat_map(|i| [Channel::SplinePtX(i), Channel::SplinePtY(i)])
        .collect()
}

/// The feather group — same all-at-once rule, its own row. A point's band
/// is its feather AND where it sits, so the two key together.
pub fn spline_feather(n_points: usize) -> Vec<Channel> {
    (0..n_points.min(u16::MAX as usize) as u16)
        .flat_map(|i| [Channel::SplinePtF(i), Channel::SplinePtB(i)])
        .collect()
}

/// The shape-level channels of a spline shape, in the order `RotoSh`
/// indexes them.
pub const ROTO_SHAPE_CHANS: [Channel; 12] = [
    Channel::SplineTx,
    Channel::SplineTy,
    Channel::SplineRot,
    Channel::SplineSx,
    Channel::SplineSy,
    Channel::SplineCx,
    Channel::SplineCy,
    Channel::SplineOpacity,
    Channel::SplineExpand,
    Channel::SplineFalloff,
    Channel::SplineFeatherOut,
    Channel::SplineFeatherIn,
];

/// Shape `s`'s name for a shape-0 spline channel: itself for shape 0, a
/// `RotoPt`/`RotoSh` for the others. Anything that is not a per-shape
/// channel comes back unchanged.
pub fn for_shape(s: usize, ch: Channel) -> Channel {
    if s == 0 {
        return ch;
    }
    let s = s.min(u16::MAX as usize) as u16;
    match ch {
        Channel::SplinePtX(i) => Channel::RotoPt(s, i, 0),
        Channel::SplinePtY(i) => Channel::RotoPt(s, i, 1),
        Channel::SplinePtF(i) => Channel::RotoPt(s, i, 2),
        Channel::SplinePtB(i) => Channel::RotoPt(s, i, 3),
        c => match ROTO_SHAPE_CHANS.iter().position(|x| *x == c) {
            Some(w) => Channel::RotoSh(s, w as u8),
            None => c,
        },
    }
}

/// The inverse for shapes >= 1: (shape, the shape-0 channel it stands for).
pub fn shape_base(ch: Channel) -> Option<(usize, Channel)> {
    match ch {
        Channel::RotoPt(s, i, w) => Some((
            s as usize,
            match w {
                0 => Channel::SplinePtX(i),
                1 => Channel::SplinePtY(i),
                2 => Channel::SplinePtF(i),
                _ => Channel::SplinePtB(i),
            },
        )),
        Channel::RotoSh(s, w) => ROTO_SHAPE_CHANS.get(w as usize).map(|c| (s as usize, *c)),
        _ => None,
    }
}

/// Run `f` on shape `s`'s curves under their SHAPE-0 names, so every
/// helper written for one spline (point insert/remove, keyed write-back,
/// clearing) works on any shape of a container. Shape 0 is the map itself.
pub fn in_shape<R>(anim: &mut Anim, s: usize, f: impl FnOnce(&mut Anim) -> R) -> R {
    if s == 0 {
        return f(anim);
    }
    let mine: Vec<Channel> = anim
        .keys()
        .filter(|c| shape_base(**c).is_some_and(|(ss, _)| ss == s))
        .copied()
        .collect();
    let mut tmp = Anim::new();
    for c in mine {
        if let (Some((_, base)), Some(curve)) = (shape_base(c), anim.remove(&c)) {
            tmp.insert(base, curve);
        }
    }
    let r = f(&mut tmp);
    for (c, curve) in tmp {
        anim.insert(for_shape(s, c), curve);
    }
    r
}

/// Shape `s` left a container: its curves go, and every later shape's
/// slide down one (shape 1 becoming shape 0's own channels when 0 went).
/// Returns the renaming, for the caller to apply to patch cables too.
pub fn shape_removed(anim: &mut Anim, s: usize) -> impl Fn(Channel) -> Option<Channel> {
    let rename = move |c: Channel| -> Option<Channel> {
        let (cs, base) = match shape_base(c) {
            Some(x) => x,
            None if is_shape_chan(c) => (0, c),
            None => return Some(c),
        };
        if cs == s {
            None
        } else if cs > s {
            Some(for_shape(cs - 1, base))
        } else {
            Some(c)
        }
    };
    let old = std::mem::take(anim);
    for (c, curve) in old {
        if let Some(n) = rename(c) {
            anim.insert(n, curve);
        }
    }
    rename
}

/// A shape-0 spline channel (per point or per shape).
pub fn is_shape_chan(c: Channel) -> bool {
    matches!(
        c,
        Channel::SplinePtX(_) | Channel::SplinePtY(_) | Channel::SplinePtF(_) | Channel::SplinePtB(_)
    ) || ROTO_SHAPE_CHANS.contains(&c)
}

fn chan_at(anim: &Anim, ch: Channel, f: f64, fallback: f32) -> f32 {
    anim.get(&ch).and_then(|c| c.eval_at(f as f32)).unwrap_or(fallback)
}

/// Every frame the shape is keyed on, in order.
fn spline_key_frames(anim: &Anim, n_points: usize) -> Vec<f64> {
    let mut fs: Vec<f64> = spline_shape(n_points)
        .iter()
        .filter_map(|ch| anim.get(ch))
        .flat_map(|c| c.keys.iter().map(|k| k.frame))
        .collect();
    fs.sort_unstable_by(f64::total_cmp);
    fs.dedup();
    fs
}

/// Per-point channels are indexed by position in the list, so inserting a
/// point renumbers every channel after it. Slide the curves along with
/// their points, then give the new point (already in `pts` at `at`) its own
/// curve: an animated shape would otherwise sprout one frozen vertex that
/// tears the outline as the rest moves. It follows its neighbours, holding
/// the offset it was born with at `t`.
pub fn spline_point_inserted(anim: &mut Anim, pts: &[SplinePoint], at: usize, t: u64) {
    let n = pts.len();
    if n < 2 || at >= n {
        return;
    }
    for i in (at + 1..n).rev() {
        let (d, s) = (i as u16, i as u16 - 1);
        for (to, from) in [
            (Channel::SplinePtX(d), Channel::SplinePtX(s)),
            (Channel::SplinePtY(d), Channel::SplinePtY(s)),
            (Channel::SplinePtF(d), Channel::SplinePtF(s)),
            (Channel::SplinePtB(d), Channel::SplinePtB(s)),
        ] {
            match anim.remove(&from) {
                Some(c) => {
                    anim.insert(to, c);
                }
                None => {
                    anim.remove(&to);
                }
            }
        }
    }
    let frames = spline_key_frames(anim, n);
    if frames.is_empty() {
        return; // static shape: the authored point is the whole story
    }
    let (a, b) = ((at + n - 1) % n, (at + 1) % n);
    // Where the neighbours' midpoint sits at `t` vs where the point was
    // actually dropped — the offset it keeps for the whole curve.
    let mid = |anim: &Anim, f: f64, axis: usize| {
        let ch = |i: usize| {
            if axis == 0 {
                Channel::SplinePtX(i as u16)
            } else {
                Channel::SplinePtY(i as u16)
            }
        };
        0.5 * (chan_at(anim, ch(a), f, pts[a].p[axis]) + chan_at(anim, ch(b), f, pts[b].p[axis]))
    };
    let off = [
        pts[at].p[0] - mid(anim, t as f64, 0),
        pts[at].p[1] - mid(anim, t as f64, 1),
    ];
    for f in frames {
        let (x, y) = (mid(anim, f, 0) + off[0], mid(anim, f, 1) + off[1]);
        anim.entry(Channel::SplinePtX(at as u16)).or_default().set_key(f, x);
        anim.entry(Channel::SplinePtY(at as u16)).or_default().set_key(f, y);
    }
    // Feather rides along only if the feather group is animated at all.
    let ff: Vec<f64> = spline_feather(n)
        .iter()
        .filter_map(|ch| anim.get(ch))
        .flat_map(|c| c.keys.iter().map(|k| k.frame))
        .collect();
    for f in ff {
        let v = 0.5
            * (chan_at(anim, Channel::SplinePtF(a as u16), f, pts[a].feather)
                + chan_at(anim, Channel::SplinePtF(b as u16), f, pts[b].feather));
        anim.entry(Channel::SplinePtF(at as u16)).or_default().set_key(f, v);
        let b = 0.5
            * (chan_at(anim, Channel::SplinePtB(a as u16), f, pts[a].bias)
                + chan_at(anim, Channel::SplinePtB(b as u16), f, pts[b].bias));
        anim.entry(Channel::SplinePtB(at as u16)).or_default().set_key(f, b);
    }
}

/// The mirror: drop the removed point's curves and slide the rest down.
/// `n_before` is the point count BEFORE the removal.
pub fn spline_point_removed(anim: &mut Anim, at: usize, n_before: usize) {
    if at >= n_before {
        return;
    }
    for i in at..n_before {
        let d = i as u16;
        for (to, from) in [
            (Channel::SplinePtX(d), Channel::SplinePtX(d + 1)),
            (Channel::SplinePtY(d), Channel::SplinePtY(d + 1)),
            (Channel::SplinePtF(d), Channel::SplinePtF(d + 1)),
            (Channel::SplinePtB(d), Channel::SplinePtB(d + 1)),
        ] {
            match anim.remove(&from) {
                Some(c) => {
                    anim.insert(to, c);
                }
                None => {
                    anim.remove(&to);
                }
            }
        }
    }
}

/// Every spline channel gone (the "clear" button drops the points too).
pub fn spline_channels_cleared(anim: &mut Anim) {
    anim.retain(|c, _| {
        !matches!(
            c,
            Channel::SplinePtX(_)
                | Channel::SplinePtY(_)
                | Channel::SplinePtF(_)
                | Channel::SplinePtB(_)
        )
    });
}

/// Every channel that applies to this node right now — the fixed table plus
/// dynamic ones (spline points). Properties/keying sweep this.
pub fn channels_for(kind: &NodeKind) -> Vec<Channel> {
    let mut v: Vec<Channel> = ALL.iter().copied().chain(crate::nodes::channels()).collect();
    if let NodeKind::Grade(g) = kind {
        for i in 0..g.zone_count().min(u16::MAX as usize) as u16 {
            v.extend(zone_channels(i));
        }
    }
    if let NodeKind::Spline(sp) = kind {

        for i in 0..sp.points.len().min(u16::MAX as usize) as u16 {
            v.push(Channel::SplinePtX(i));
            v.push(Channel::SplinePtY(i));
            v.push(Channel::SplinePtF(i));
            v.push(Channel::SplinePtB(i));
        }
        // Its axes' transforms.
        for k in 0..sp.axes.len().min(u16::MAX as usize) as u16 {
            v.extend((0..9).map(|w| Channel::RotoAx(k, w)));
        }
        // A container's other shapes: the same set, renamed per shape.
        for (k, sh) in sp.more.iter().enumerate() {
            let s = k + 1;
            v.extend(ROTO_SHAPE_CHANS.iter().map(|c| for_shape(s, *c)));
            for i in 0..sh.points.len().min(u16::MAX as usize) as u16 {
                for c in [
                    Channel::SplinePtX(i),
                    Channel::SplinePtY(i),
                    Channel::SplinePtF(i),
                    Channel::SplinePtB(i),
                ] {
                    v.push(for_shape(s, c));
                }
            }
        }
    }
    if let NodeKind::Keyer(k) = kind {
        for i in 0..k.blobs.len().min(u16::MAX as usize) as u16 {
            v.push(Channel::KeyCenterL(i));
            v.push(Channel::KeyCenterA(i));
            v.push(Channel::KeyCenterB(i));
            v.push(Channel::KeyTolL(i));
            v.push(Channel::KeyTolC(i));
            v.push(Channel::KeyTolH(i));
            v.push(Channel::KeySoft(i));
        }
    }
    if let NodeKind::Layer(p) = kind {
        for i in 0..p.layers.len().min(engine::graph::LAYER_MAX) as u16 {
            v.push(Channel::LayerOpacity(i));
        }
    }
    if let NodeKind::Tracker(t) = kind {
        for i in 1..t.count().min(engine::graph::TRACK_POINTS_MAX) as u16 {
            v.push(Channel::TrackPtX(i));
            v.push(Channel::TrackPtY(i));
        }
        for k in 0..4 {
            v.push(Channel::TrackPinX(k));
            v.push(Channel::TrackPinY(k));
        }
    }
    if let NodeKind::GridWarp(p) = kind {
        for i in 0..p.count() as u16 {
            v.extend(crate::nodes::gridwarp::point_channels(i));
        }
    }
    if let NodeKind::Warp(p) = kind {
        v.extend(crate::nodes::warp::all_point_channels(p));
    }
    if let NodeKind::Gradient(p) = kind {
        for i in 0..p.stops.len().min(engine::graph::GRAD_STOPS) as u16 {
            v.extend(crate::nodes::gradient::stop_channels(i));
        }
    }
    if let NodeKind::Recombine(p) = kind {
        for i in 0..p.n.clamp(1, 16) as u16 {
            v.push(Channel::RecombineGain(i));
        }
    }
    if let NodeKind::Field(f) = kind {
        for i in 0..f.channel_count() as u16 {
            v.push(Channel::Field(i));
        }
    }
    // A geometry node's sockets are channels the same way (per node, so the
    // index never clashes with a field node's).
    if let NodeKind::GeoNode(n) = kind {
        for i in 0..n.channel_count() as u16 {
            v.push(Channel::Field(i));
        }
    }
    // So are an SDF node's.
    if let NodeKind::SdfNode(n) = kind {
        for i in 0..n.channel_count() as u16 {
            v.push(Channel::Field(i));
        }
    }
    if let NodeKind::Shader(p) = kind {
        // Declared order when the shader parses; stored keys otherwise, so a
        // shader that is mid-edit keeps its curves.
        let def = p.def();
        if def.params.is_empty() {
            for name in p.values.keys() {
                v.push(Channel::Shader(intern(name)));
            }
        } else {
            for spec in &def.params {
                for lane in 0..spec.ty.lanes() {
                    v.push(Channel::Shader(intern(&shader_key(&spec.name, spec.ty, lane))));
                }
            }
        }
    }
    // A Noise node exposes its sub-graph's Sliders and Inputs by name, as
    // the same channel kind a plugin param uses — so the panel, the patch
    // bay and link mode all reach them with no new machinery.
    if let NodeKind::Noise(n) = kind {
        for (name, _) in n.graph.exposed() {
            v.push(Channel::Custom(intern(&name)));
        }
    }
    // A 2dini node exposes its graph's Sliders and Inputs exactly as a
    // Noise does: same channel kind, same widgets, same jacks.
    if let NodeKind::Geo(g) = kind {
        v.push(Channel::GeoSeed);
        for (name, _) in g.graph.exposed() {
            v.push(Channel::Custom(intern(&name)));
        }
        // Its light and SDF effects, `look.<param>` (`geolightui`).
        v.extend(crate::geolightui::channels());
    }
    // A Text node's numbers past position and size, animators included,
    // as `text.<name>` (`nodes::text`).
    if let NodeKind::Text(p) = kind {
        v.extend(crate::nodes::text::dyn_channels(p));
    }
    // An Env's view, spin and exposure, as `env.<name>` (`nodes::env`).
    if let NodeKind::Env(_) = kind {
        v.extend(crate::nodes::env::dyn_channels());
    }
    // A Refract's numbers, as `refract.<name>` (`nodes::refract`).
    if let NodeKind::Refract(_) = kind {
        v.extend(crate::nodes::refract::dyn_channels());
    }
    // A Lens Flare's numbers, as `flare.<name>` (`nodes::lensflare`).
    if let NodeKind::LensFlare(_) = kind {
        v.extend(crate::nodes::lensflare::dyn_channels());
    }
    // A Contrast's numbers, as `contrast.<name>` (`nodes::contrast`).
    if let NodeKind::Contrast(_) = kind {
        v.extend(crate::nodes::contrast::dyn_channels());
    }
    // A Particles node's own numbers, as `particles.<name>`.
    if let NodeKind::Particles(_) = kind {
        v.extend(crate::nodes::particles::dyn_channels());
    }
    // A Plasma's numbers, as `plasma.<name>` (`nodes::plasma`).
    if let NodeKind::Plasma(_) = kind {
        v.extend(crate::nodes::plasma::dyn_channels());
    }
    // A Planet's numbers, as `planet.<name>` (`nodes::planet`).
    if let NodeKind::Planet(_) = kind {
        v.extend(crate::nodes::planet::dyn_channels());
    }
    // A Splat's numbers, as `splat.<name>` (`nodes::splat`).
    if let NodeKind::Splat(_) = kind {
        v.extend(crate::nodes::splat::dyn_channels());
    }
    // A Starfield's numbers, as `star.<name>` (`nodes::starfield`).
    if let NodeKind::Starfield(_) = kind {
        v.extend(crate::nodes::starfield::dyn_channels());
    }
    // An SDF node the same way: its graph's Sliders and Inputs by name.
    if let NodeKind::Sdf(p) = kind {
        for (name, _) in p.graph.exposed() {
            v.push(Channel::Custom(intern(&name)));
        }
    }
    if let NodeKind::Custom { type_id, params } = kind {
        // Registry order when installed; stored keys keep an uninstalled
        // plugin's animation intact.
        match engine::plugins::get(type_id) {
            Some(def) => {
                for p in &def.params {
                    v.push(Channel::Custom(intern(&p.name)));
                }
            }
            None => {
                // The pipe flag rides in the map; it is not a channel.
                for name in params.keys().filter(|k| *k != engine::graph::CUSTOM_MATTE_KEY) {
                    v.push(Channel::Custom(intern(name)));
                }
            }
        }
    }
    v
}

/// Apply every curve in `anim` to `kind` at a FRACTIONAL frame — what a
/// time-warped branch samples, so upstream animation moves smoothly even
/// when the warp lands between frames.
/// One sample of an animated element's path: where it was, on which frame,
/// and whether that frame carries a key.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathPoint {
    pub frame: f64,
    pub at: [f32; 2],
    pub key: bool,
}

/// How many samples a motion path may take, however long the span is. A
/// path is a picture, not a data set: past a few hundred points the extra
/// samples land inside the same pixel and cost a clone of the node's params
/// each.
pub const PATH_SAMPLES: u64 = 360;

/// Sample the path an animated element follows over `span`.
///
/// `point_of` reads the element's position out of the node's kind EVALUATED
/// at a frame — so any kind with a point that curves can have a path by
/// answering there, and anything that moves it (a translate, a centre, a
/// tracked position) shows up without being enumerated here.
///
/// Returns nothing when the point never moves: a static element has no path,
/// and a dot drawn on top of its own gizmo is only noise. Key frames of ANY
/// animated channel on the node are marked and always sampled — they are
/// where the shape of the path is decided — even when the span is strided.
pub fn motion_path(
    kind: &NodeKind,
    anim: &std::collections::BTreeMap<Channel, AnimCurve>,
    span: (u64, u64),
    point_of: &dyn Fn(&NodeKind) -> Option<[f32; 2]>,
) -> Vec<PathPoint> {
    if point_of(kind).is_none() {
        return Vec::new();
    }
    let (a, b) = (span.0 as f64, span.1.max(span.0) as f64);
    let keys: Vec<f64> = {
        let mut k: Vec<f64> = anim
            .values()
            .flat_map(|c| c.keys.iter().map(|k| k.frame))
            .filter(|f| (a..=b).contains(f))
            .collect();
        k.sort_by(|x, y| x.total_cmp(y));
        k.dedup_by(|x, y| (*x - *y).abs() < 1e-6);
        k
    };
    if keys.is_empty() {
        return Vec::new();
    }
    let stride = (((b - a) as u64 + 1).div_ceil(PATH_SAMPLES)).max(1) as f64;
    let mut frames: Vec<f64> = Vec::with_capacity(PATH_SAMPLES as usize + keys.len());
    let mut f = a;
    while f < b {
        frames.push(f);
        f += stride;
    }
    frames.push(b);
    frames.extend(keys.iter().copied());
    frames.sort_by(|x, y| x.total_cmp(y));
    frames.dedup_by(|x, y| (*x - *y).abs() < 1e-6);
    let is_key = |f: f64| keys.iter().any(|k| (k - f).abs() < 1e-6);
    let mut out = Vec::with_capacity(frames.len());
    for f in frames {
        let mut k = kind.clone();
        apply_at(&mut k, anim, f as f32);
        if let Some(at) = point_of(&k) {
            out.push(PathPoint { frame: f, at, key: is_key(f) });
        }
    }
    let moved = out
        .iter()
        .any(|p| (p.at[0] - out[0].at[0]).abs() > 1e-3 || (p.at[1] - out[0].at[1]).abs() > 1e-3);
    if moved { out } else { Vec::new() }
}

pub fn apply_at(
    kind: &mut NodeKind,
    anim: &std::collections::BTreeMap<Channel, AnimCurve>,
    t: f32,
) {
    for (ch, curve) in anim {
        if let Some(v) = curve.eval_at(t) {
            set(kind, *ch, v);
        }
    }
}

/// Apply every curve in `anim` to `kind` at frame `t` (transient use only).
pub fn apply(
    kind: &mut NodeKind,
    anim: &std::collections::BTreeMap<Channel, AnimCurve>,
    t: u64,
) {
    for (ch, curve) in anim {
        if let Some(v) = curve.eval(t) {
            set(kind, *ch, v);
        }
    }
}

/// Read a channel's current value off a node kind.
/// Write one field of keyer blob `i`, if it is still there — a curve can
/// outlive the sample it was keyed on (undo, a setup edited by hand).
fn blob(
    k: &mut engine::keyer::KeyerParams,
    i: u16,
    f: impl FnOnce(&mut engine::keyer::KeyBlob),
) {
    if let Some(b) = k.blobs.get_mut(i as usize) {
        f(b);
    }
}

/// Channel name for one lane of a Shader param: scalars use the bare name,
/// vectors append the lane so each float is its own channel.
pub fn shader_key(name: &str, ty: engine::usershader::ParamTy, lane: usize) -> String {
    if ty.lanes() > 1 {
        format!("{name}.{lane}")
    } else {
        name.to_string()
    }
}

/// Split a Shader channel name back into (param, lane).
fn shader_lane(s: &str) -> (String, usize) {
    match s.rsplit_once('.') {
        Some((n, l)) => match l.parse::<usize>() {
            Ok(i) if i < 4 && !n.is_empty() => (n.to_string(), i),
            _ => (s.to_string(), 0),
        },
        None => (s.to_string(), 0),
    }
}

/// Write one field of layer `i`, if it is still there — a curve can outlive
/// the layer it was keyed on (undo, a reorder, a setup edited by hand).
fn layer(
    p: &mut engine::graph::LayerParams,
    i: u16,
    f: impl FnOnce(&mut engine::graph::LayerItem),
) {
    if let Some(l) = p.layers.get_mut(i as usize) {
        f(l);
    }
}

fn axis_get(x: &engine::scene3d::Xform, w: usize) -> f32 {
    match w {
        0..=2 => x.translate[w],
        3..=5 => x.rotate[w - 3],
        _ => x.scale[(w - 6).min(2)],
    }
}

fn axis_set(x: &mut engine::scene3d::Xform, w: usize, v: f32) {
    match w {
        0..=2 => x.translate[w] = v,
        3..=5 => x.rotate[w - 3] = v,
        _ => x.scale[(w - 6).min(2)] = v,
    }
}

pub fn get(kind: &NodeKind, ch: Channel) -> Option<f32> {
    use Channel::*;
    // A roto axis: on its owner by index, inside the dive as itself.
    if let (NodeKind::Spline(sp), RotoAx(k, w)) = (kind, ch) {
        return sp.axes.get(k as usize).map(|a| axis_get(&a.xform, w as usize));
    }
    if let NodeKind::RotoNode(engine::graph::RotoInner::Axis(a)) = kind {
        let w = crate::rotodive::AXIS_CHANNELS.iter().position(|c| *c == ch)?;
        return Some(axis_get(&a.xform, w));
    }
    // A container's other shapes read through the one-shape view.
    if let Some((s, base)) = shape_base(ch) {
        let NodeKind::Spline(sp) = kind else { return None };
        if s >= sp.shape_count() {
            return None;
        }
        return get(&NodeKind::Spline(sp.single(s)), base);
    }
    // User shader params: the stored value, else the shader's own default.
    if let (NodeKind::Shader(p), Shader(i)) = (kind, ch) {
        let (name, lane) = shader_lane(&interned(i));
        if let Some(v) = p.values.get(&name) {
            return Some(v[lane]);
        }
        return p.def().param(&name).map(|s| s.default[lane]);
    }
    // Plugin params: the stored value, else the manifest default.
    if let (NodeKind::Noise(n), Custom(i)) = (kind, ch) {
        return n.graph.exposed_value(&interned(i));
    }
    if let (NodeKind::Geo(g), Custom(i)) = (kind, ch) {
        let name = interned(i);
        if let Some(p) = name.strip_prefix(engine::geolight::PREFIX) {
            return g.look.get(p);
        }
        if engine::geo::parse_inner(&name).is_some() {
            return g.graph.inner_value(&name);
        }
        return g.graph.exposed_value(&name);
    }
    if let (NodeKind::Sdf(p), Custom(i)) = (kind, ch) {
        return p.graph.exposed_value(&interned(i));
    }
    if let (NodeKind::Custom { type_id, params }, Custom(i)) = (kind, ch) {
        let name = interned(i);
        if let Some(v) = params.get(&name) {
            return Some(*v);
        }
        return engine::plugins::get(type_id)?
            .params
            .iter()
            .find(|p| p.name == name)
            .map(|p| p.default);
    }
    Some(match (kind, ch) {
        (NodeKind::Transform(t), TranslateX) => t.translate[0],
        (NodeKind::Transform(t), TranslateY) => t.translate[1],
        (NodeKind::Transform(t), Rotate) => t.rotate,
        (NodeKind::Transform(t), ScaleX) => t.scale[0],
        (NodeKind::Transform(t), ScaleY) => t.scale[1],
        (NodeKind::Transform(t), CenterX) => t.center[0],
        (NodeKind::Transform(t), CenterY) => t.center[1],
        (NodeKind::Transform(t), ShearX) => t.shear[0],
        (NodeKind::Transform(t), ShearY) => t.shear[1],
        (NodeKind::Transform(t), Taper) => t.taper,
        (NodeKind::Transform(t), TaperAngle) => t.taper_angle,
        (NodeKind::Grade(g), LiftR) => g.lift[0],
        (NodeKind::Grade(g), LiftG) => g.lift[1],
        (NodeKind::Grade(g), LiftB) => g.lift[2],
        (NodeKind::Grade(g), GammaR) => g.gamma[0],
        (NodeKind::Grade(g), GammaG) => g.gamma[1],
        (NodeKind::Grade(g), GammaB) => g.gamma[2],
        (NodeKind::Grade(g), GainR) => g.gain[0],
        (NodeKind::Grade(g), GainG) => g.gain[1],
        (NodeKind::Grade(g), GainB) => g.gain[2],
        (NodeKind::Grade(g), OffsetR) => g.offset[0],
        (NodeKind::Grade(g), OffsetG) => g.offset[1],
        (NodeKind::Grade(g), OffsetB) => g.offset[2],
        (NodeKind::Grade(g), ZoneLiftR(i)) => g.zones.as_ref()?.adjust.get(i as usize)?.lift[0],
        (NodeKind::Grade(g), ZoneLiftG(i)) => g.zones.as_ref()?.adjust.get(i as usize)?.lift[1],
        (NodeKind::Grade(g), ZoneLiftB(i)) => g.zones.as_ref()?.adjust.get(i as usize)?.lift[2],
        (NodeKind::Grade(g), ZoneGammaR(i)) => g.zones.as_ref()?.adjust.get(i as usize)?.gamma[0],
        (NodeKind::Grade(g), ZoneGammaG(i)) => g.zones.as_ref()?.adjust.get(i as usize)?.gamma[1],
        (NodeKind::Grade(g), ZoneGammaB(i)) => g.zones.as_ref()?.adjust.get(i as usize)?.gamma[2],
        (NodeKind::Grade(g), ZoneGainR(i)) => g.zones.as_ref()?.adjust.get(i as usize)?.gain[0],
        (NodeKind::Grade(g), ZoneGainG(i)) => g.zones.as_ref()?.adjust.get(i as usize)?.gain[1],
        (NodeKind::Grade(g), ZoneGainB(i)) => g.zones.as_ref()?.adjust.get(i as usize)?.gain[2],
        (NodeKind::Grade(g), ZoneOffsetR(i)) => g.zones.as_ref()?.adjust.get(i as usize)?.offset[0],
        (NodeKind::Grade(g), ZoneOffsetG(i)) => g.zones.as_ref()?.adjust.get(i as usize)?.offset[1],
        (NodeKind::Grade(g), ZoneOffsetB(i)) => g.zones.as_ref()?.adjust.get(i as usize)?.offset[2],
        (NodeKind::Grade(g), ZoneSat(i)) => g.zones.as_ref()?.adjust.get(i as usize)?.saturation,
        (NodeKind::Grade(g), ZoneContrast(i)) => g.zones.as_ref()?.adjust.get(i as usize)?.contrast,
        (NodeKind::Grade(g), ZonePivot(i)) => g.zones.as_ref()?.adjust.get(i as usize)?.pivot,
        (NodeKind::Blur(b), _) => return crate::nodes::blur::get(b, ch),
        (NodeKind::SoftCrop(p), _) => return crate::nodes::softcrop::get(p, ch),
        (NodeKind::MotionBlur(p), _) => return crate::nodes::motionblur::get(p, ch),
        (NodeKind::VectorBlur(p), _) => return crate::nodes::vectorblur::get(p, ch),
        (NodeKind::Camera3D(p), _) => return crate::nodes::scene3d::camera_get(p, ch),
        (NodeKind::Card3D(p), _) => return crate::nodes::scene3d::card_get(p, ch),
        (NodeKind::Render3D(p), _) => return crate::nodes::scene3d::render_get(p, ch),
        (NodeKind::VectorWarp(p), _) => return crate::nodes::vectorwarp::get(p, ch),
        (NodeKind::Rifetime(p), _) => return crate::nodes::rifetime::get(p, ch),
        (NodeKind::Freeze(p), _) => return crate::nodes::freeze::get(p, ch),
        (NodeKind::ColorCompress(p), _) => return crate::nodes::colorcompress::get(p, ch),
        (NodeKind::HueRotate(p), _) => return crate::nodes::huerotate::get(p, ch),
        (NodeKind::Lut(p), _) => return crate::nodes::lut::get(p, ch),
        (NodeKind::Text(p), _) => return crate::nodes::text::get(p, ch),
        (NodeKind::Env(p), _) => return crate::nodes::env::get(p, ch),
        (NodeKind::Refract(p), _) => return crate::nodes::refract::get(p, ch),
        (NodeKind::Starfield(p), _) => return crate::nodes::starfield::get(p, ch),
        (NodeKind::Splat(p), _) => return crate::nodes::splat::get(p, ch),
        (NodeKind::Planet(p), _) => return crate::nodes::planet::get(p, ch),
        (NodeKind::Plasma(p), _) => return crate::nodes::plasma::get(p, ch),
        (NodeKind::Particles(p), _) => return crate::nodes::particles::get(p, ch),
        (NodeKind::LensFlare(p), _) => return crate::nodes::lensflare::get(p, ch),
        (NodeKind::Contrast(p), _) => return crate::nodes::contrast::get(p, ch),
        (NodeKind::Burnin(p), _) => return crate::nodes::burnin::get(p, ch),
        (NodeKind::Mirror(p), _) => return crate::nodes::mirror::get(p, ch),
        (NodeKind::Compare(p), _) => return crate::nodes::compare::get(p, ch),
        (NodeKind::LensDistort(p), _) => return crate::nodes::lensdistort::get(p, ch),
        (NodeKind::Gamma(p), _) => return crate::nodes::gamma::get(p, ch),
        (NodeKind::ToneMap(p), _) => return crate::nodes::tonemap::get(p, ch),
        (NodeKind::Despill(p), _) => return crate::nodes::despill::get(p, ch),
        (NodeKind::Lightwrap(p), _) => return crate::nodes::lightwrap::get(p, ch),
        (NodeKind::ScreenKey(p), _) => return crate::nodes::screenkey::get(p, ch),
        (NodeKind::ColorSlice(p), _) => return crate::nodes::colorslice::get(p, ch),

        (NodeKind::Defocus(d), DefocusRadius) => d.radius,
        (NodeKind::Defocus(d), DefocusFocus) => d.focus,
        (NodeKind::Defocus(d), DefocusFStop) => d.f_stop,
        (NodeKind::Defocus(d), DefocusRotation) => d.rotation,
        (NodeKind::Defocus(d), DefocusCatseye) => d.catseye,
        (NodeKind::Defocus(d), DefocusEdge) => d.edge,
        (NodeKind::Defocus(d), DefocusClamp) => d.bright_clamp,
        (NodeKind::Blend { mix, .. }, BlendMix) => *mix,
        (NodeKind::MatteCombine(m), MatteMix) => m.mix,
        (NodeKind::Glow(g), _) => return crate::nodes::glow::get(g, ch),
        (NodeKind::Godrays(g), _) => return crate::nodes::godrays::get(g, ch),
        (NodeKind::EdgeExtend(g), _) => return crate::nodes::edgeextend::get(g, ch),
        (NodeKind::Glint(g), _) => return crate::nodes::glint::get(g, ch),
        (NodeKind::GlintKernel(p), _) => return crate::nodes::glintkernel::get(p, ch),
        (NodeKind::BokehKernel(p), _) => return crate::nodes::bokehkernel::get(p, ch),
        (NodeKind::Diffuse(d), _) => return crate::nodes::diffuse::get(d, ch),
        (NodeKind::LumaKey(k), LumaLo) => k.band(0).lo,
        (NodeKind::LumaKey(k), LumaHi) => k.band(0).hi,
        (NodeKind::TimeRemap(p), TimeRemapFrame) => p.frame,
        (NodeKind::Switch(p), SwitchWhich) => p.which as f32,
        (NodeKind::Keyer(k), KeyTilt) => k.tilt,
        (NodeKind::Keyer(k), KeyLo) => k.lo,
        (NodeKind::Keyer(k), KeyHi) => k.hi,
        (NodeKind::Keyer(k), KeyGamma) => k.gamma,
        (NodeKind::Keyer(k), KeyCenterL(i)) => k.blobs.get(i as usize)?.center[0],
        (NodeKind::Keyer(k), KeyCenterA(i)) => k.blobs.get(i as usize)?.center[1],
        (NodeKind::Keyer(k), KeyCenterB(i)) => k.blobs.get(i as usize)?.center[2],
        (NodeKind::Keyer(k), KeyTolL(i)) => k.blobs.get(i as usize)?.tol[0],
        (NodeKind::Keyer(k), KeyTolC(i)) => k.blobs.get(i as usize)?.tol[1],
        (NodeKind::Keyer(k), KeyTolH(i)) => k.blobs.get(i as usize)?.tol[2],
        (NodeKind::Keyer(k), KeySoft(i)) => k.blobs.get(i as usize)?.soft,
        (NodeKind::Spline(s), SplinePtX(i)) => s.points.get(i as usize)?.p[0],
        (NodeKind::Spline(s), SplinePtY(i)) => s.points.get(i as usize)?.p[1],
        (NodeKind::Spline(s), SplinePtF(i)) => s.points.get(i as usize)?.feather,
        (NodeKind::Spline(s), SplineTx) => s.xf.translate[0],
        (NodeKind::Spline(s), SplineTy) => s.xf.translate[1],
        (NodeKind::Spline(s), SplineRot) => s.xf.rotate,
        (NodeKind::Spline(s), SplineSx) => s.xf.scale[0],
        (NodeKind::Spline(s), SplineSy) => s.xf.scale[1],
        (NodeKind::Spline(s), SplineCx) => s.xf.center[0],
        (NodeKind::Spline(s), SplineCy) => s.xf.center[1],
        (NodeKind::Spline(s), SplineOpacity) => s.opacity,
        (NodeKind::Spline(s), SplineExpand) => s.expand,
        (NodeKind::Spline(s), SplineFalloff) => s.falloff,
        (NodeKind::Spline(s), SplineFeatherOut) => s.feather_out,
        (NodeKind::Spline(s), SplineFeatherIn) => s.feather_in,
        (NodeKind::Spline(s), SplinePtB(i)) => s.points.get(i as usize)?.bias,
        (NodeKind::Transform(t), Pin0X) => t.pin[0][0],
        (NodeKind::Transform(t), Pin0Y) => t.pin[0][1],
        (NodeKind::Transform(t), Pin1X) => t.pin[1][0],
        (NodeKind::Transform(t), Pin1Y) => t.pin[1][1],
        (NodeKind::Transform(t), Pin2X) => t.pin[2][0],
        (NodeKind::Transform(t), Pin2Y) => t.pin[2][1],
        (NodeKind::Transform(t), Pin3X) => t.pin[3][0],
        (NodeKind::Transform(t), Pin3Y) => t.pin[3][1],
        (NodeKind::CornerPin(cp), Pin0X) => cp.to[0][0],
        (NodeKind::CornerPin(cp), Pin0Y) => cp.to[0][1],
        (NodeKind::CornerPin(cp), Pin1X) => cp.to[1][0],
        (NodeKind::CornerPin(cp), Pin1Y) => cp.to[1][1],
        (NodeKind::CornerPin(cp), Pin2X) => cp.to[2][0],
        (NodeKind::CornerPin(cp), Pin2Y) => cp.to[2][1],
        (NodeKind::CornerPin(cp), Pin3X) => cp.to[3][0],
        (NodeKind::CornerPin(cp), Pin3Y) => cp.to[3][1],
        (NodeKind::CornerPin(cp), From0X) => cp.from[0][0],
        (NodeKind::CornerPin(cp), From0Y) => cp.from[0][1],
        (NodeKind::CornerPin(cp), From1X) => cp.from[1][0],
        (NodeKind::CornerPin(cp), From1Y) => cp.from[1][1],
        (NodeKind::CornerPin(cp), From2X) => cp.from[2][0],
        (NodeKind::CornerPin(cp), From2Y) => cp.from[2][1],
        (NodeKind::CornerPin(cp), From3X) => cp.from[3][0],
        (NodeKind::CornerPin(cp), From3Y) => cp.from[3][1],
        (NodeKind::Mix(m), MixAmount) => m.amount,
        (NodeKind::Saturation(sp), _) => return crate::nodes::saturation::get(sp, ch),
        (NodeKind::Gradient(p), _) => return crate::nodes::gradient::get(p, ch),
        (NodeKind::GridWarp(p), _) => return crate::nodes::gridwarp::get(p, ch),
        (NodeKind::Warp(p), _) => return crate::nodes::warp::get(p, ch),
        (NodeKind::Grain(p), _) => return crate::nodes::grain::get(p, ch),
        (NodeKind::Dilate(d), _) => return crate::nodes::dilate::get(d, ch),
        (NodeKind::EdgeDetect(p), _) => return crate::nodes::edgedetect::get(p, ch),
        (NodeKind::Median(p), _) => return crate::nodes::median::get(p, ch),
        (NodeKind::HighlightRebuild(p), _) => return crate::nodes::highlightrebuild::get(p, ch),
        (NodeKind::LtxHdr(p), _) => return crate::nodes::ltxhdr::get(p, ch),
        (NodeKind::Stabilize(p), _) => return crate::nodes::stabilize::get(p, ch),
        (NodeKind::Convolve(p), _) => return crate::nodes::convolve::get(p, ch),
        (NodeKind::Blend { gain, .. }, BlendGainA) => gain[0],
        (NodeKind::Blend { gain, .. }, BlendGainB) => gain[1],
        (NodeKind::LoopOut(p), LoopN) => p.n,
        (NodeKind::Grade(g), GradeSat) => g.saturation,
        (NodeKind::Grade(g), GradeContrast) => g.contrast,
        (NodeKind::Grade(g), GradePivot) => g.pivot,
        (NodeKind::Noise(n), NoiseProbeX) => n.probe[0],
        (NodeKind::Noise(n), NoiseProbeY) => n.probe[1],
        (NodeKind::Noise(n), NoiseScale) => n.scale,
        (NodeKind::Noise(n), NoiseSeed) => n.seed,
        (NodeKind::Noise(n), NoiseSpeed) => n.speed,
        // The field AS DATA: evaluated on the CPU at the probe, so a patch
        // cable out of here can drive any parameter in the comp. Same code
        // path the shader takes, so what drives a value is what renders.
        (NodeKind::Noise(n), NoiseValue) => {
            // Mask 1: the image wired into the node is a GPU-side picture
            // and there is nothing to sample on this side, so the probe
            // reads the field as if the mask were fully on — the same
            // "unwired means 1" convention the shader uses.
            engine::field::eval_cpu(&n.graph, n.probe, n.seed, n.phase(), 1.0)
        }
        // Green's number: only a 2-d Noise has one.
        (NodeKind::Noise(n), NoiseValueY) if n.two_d => engine::field::eval_cpu(
            &n.graph,
            n.probe,
            n.seed + n.seed_offset,
            n.phase(),
            1.0,
        ),
        (NodeKind::Noise(n), NoiseSeedOffset) => n.seed_offset,
        (NodeKind::Mult(m), MultR) => m.value[0],
        (NodeKind::Mult(m), MultG) => m.value[1],
        (NodeKind::Mult(m), MultB) => m.value[2],
        (NodeKind::Grid(g), GridSpacingX) => g.spacing[0],
        (NodeKind::Grid(g), GridSpacingY) => g.spacing[1],
        (NodeKind::Grid(g), GridWidth) => g.width,
        (NodeKind::Grid(g), GridOffsetX) => g.offset[0],
        (NodeKind::Grid(g), GridOffsetY) => g.offset[1],
        (NodeKind::Reformat(r), ReformatScale) => r.scale,
        (NodeKind::Tracker(t), TrackX) => t.track[0],
        (NodeKind::Tracker(t), TrackY) => t.track[1],
        (NodeKind::Tracker(t), TrackPtX(i)) if i > 0 => t.point(i as usize)?.track[0],
        (NodeKind::Tracker(t), TrackPtY(i)) if i > 0 => t.point(i as usize)?.track[1],
        (NodeKind::Tracker(t), TrackPinX(k)) if k < 4 => t.pins()[k as usize][0],
        (NodeKind::Tracker(t), TrackPinY(k)) if k < 4 => t.pins()[k as usize][1],
        (NodeKind::PlanarTrack(p), PlanarDX) => p.drive[0],
        (NodeKind::PlanarTrack(p), PlanarDY) => p.drive[1],
        (NodeKind::PlanarTrack(p), PlanarRot) => p.drive[2],
        (NodeKind::PlanarTrack(p), PlanarScale) => p.drive[3],
        (NodeKind::PlanarTrack(p), PlanarPin0X) => p.pin[0][0],
        (NodeKind::PlanarTrack(p), PlanarPin0Y) => p.pin[0][1],
        (NodeKind::PlanarTrack(p), PlanarPin1X) => p.pin[1][0],
        (NodeKind::PlanarTrack(p), PlanarPin1Y) => p.pin[1][1],
        (NodeKind::PlanarTrack(p), PlanarPin2X) => p.pin[2][0],
        (NodeKind::PlanarTrack(p), PlanarPin2Y) => p.pin[2][1],
        (NodeKind::PlanarTrack(p), PlanarPin3X) => p.pin[3][0],
        (NodeKind::PlanarTrack(p), PlanarPin3Y) => p.pin[3][1],
        (NodeKind::Layer(p), LayerOpacity(i)) => p.layers.get(i as usize)?.opacity,
        (NodeKind::Recombine(p), RecombineGain(i)) => {
            p.gain.get(i as usize).copied().unwrap_or(1.0)
        }
        (NodeKind::Field(f), Field(i)) => f.literal(i as usize)?,
        (NodeKind::GeoNode(n), Field(i)) => n.literal(i as usize)?,
        (NodeKind::SdfNode(n), Field(i)) => n.literal(i as usize)?,
        (NodeKind::Geo(g), GeoSeed) => g.seed,
        _ => return None,
    })
}

/// Write a channel value into a node kind (used by transient evaluation and
/// by key capture). A mismatched channel/kind pair is a no-op.
pub fn set(kind: &mut NodeKind, ch: Channel, v: f32) {
    if let (NodeKind::Spline(sp), Channel::RotoAx(k, w)) = (&mut *kind, ch) {
        if let Some(a) = sp.axes.get_mut(k as usize) {
            axis_set(&mut a.xform, w as usize, v);
        }
        return;
    }
    if let NodeKind::RotoNode(engine::graph::RotoInner::Axis(a)) = kind {
        if let Some(w) = crate::rotodive::AXIS_CHANNELS.iter().position(|c| *c == ch) {
            axis_set(&mut a.xform, w, v);
        }
        return;
    }
    if let Some((s, base)) = shape_base(ch) {
        if let NodeKind::Spline(sp) = kind {
            if s < sp.shape_count() {
                let mut one = NodeKind::Spline(sp.single(s));
                set(&mut one, base, v);
                if let NodeKind::Spline(one) = one {
                    sp.apply_single(s, &one);
                }
            }
        }
        return;
    }
    if let (NodeKind::Noise(n), Channel::Custom(i)) = (&mut *kind, ch) {
        n.graph.set_exposed(&interned(i), v);
        return;
    }
    if let (NodeKind::Geo(g), Channel::Custom(i)) = (&mut *kind, ch) {
        let name = interned(i);
        if let Some(p) = name.strip_prefix(engine::geolight::PREFIX) {
            g.look.set(p, v);
            return;
        }
        if engine::geo::parse_inner(&name).is_some() {
            g.graph.set_inner(&name, v);
            return;
        }
        g.graph.set_exposed(&name, v);
        return;
    }
    if let (NodeKind::Sdf(p), Channel::Custom(i)) = (&mut *kind, ch) {
        p.graph.set_exposed(&interned(i), v);
        return;
    }
    if let (NodeKind::Custom { params, .. }, Channel::Custom(i)) = (&mut *kind, ch) {
        params.insert(interned(i), v);
        return;
    }
    if let (NodeKind::Shader(p), Channel::Shader(i)) = (&mut *kind, ch) {
        let (name, lane) = shader_lane(&interned(i));
        // A curve can outlive the param it was keyed on (the text was
        // edited); writing into the map anyway keeps the value for the
        // shader that brings the name back.
        let cell = p.values.entry(name).or_insert([0.0; 4]);
        cell[lane] = v;
        return;
    }
    use Channel::*;
    match (kind, ch) {
        (NodeKind::Transform(t), TranslateX) => t.translate[0] = v,
        (NodeKind::Transform(t), TranslateY) => t.translate[1] = v,
        (NodeKind::Transform(t), Rotate) => t.rotate = v,
        (NodeKind::Transform(t), ScaleX) => t.scale[0] = v,
        (NodeKind::Transform(t), ScaleY) => t.scale[1] = v,
        (NodeKind::Transform(t), CenterX) => t.center[0] = v,
        (NodeKind::Transform(t), CenterY) => t.center[1] = v,
        (NodeKind::Transform(t), ShearX) => t.shear[0] = v,
        (NodeKind::Transform(t), ShearY) => t.shear[1] = v,
        (NodeKind::Transform(t), Taper) => t.taper = v,
        (NodeKind::Transform(t), TaperAngle) => t.taper_angle = v,
        (NodeKind::Grade(g), LiftR) => g.lift[0] = v,
        (NodeKind::Grade(g), LiftG) => g.lift[1] = v,
        (NodeKind::Grade(g), LiftB) => g.lift[2] = v,
        (NodeKind::Grade(g), GammaR) => g.gamma[0] = v,
        (NodeKind::Grade(g), GammaG) => g.gamma[1] = v,
        (NodeKind::Grade(g), GammaB) => g.gamma[2] = v,
        (NodeKind::Grade(g), GainR) => g.gain[0] = v,
        (NodeKind::Grade(g), GainG) => g.gain[1] = v,
        (NodeKind::Grade(g), GainB) => g.gain[2] = v,
        (NodeKind::Grade(g), OffsetR) => g.offset[0] = v,
        (NodeKind::Grade(g), OffsetG) => g.offset[1] = v,
        (NodeKind::Grade(g), OffsetB) => g.offset[2] = v,
        (
            NodeKind::Grade(g),
            ZoneLiftR(i) | ZoneLiftG(i) | ZoneLiftB(i) | ZoneGammaR(i) | ZoneGammaG(i)
            | ZoneGammaB(i) | ZoneGainR(i) | ZoneGainG(i) | ZoneGainB(i) | ZoneOffsetR(i)
            | ZoneOffsetG(i) | ZoneOffsetB(i) | ZoneSat(i) | ZoneContrast(i) | ZonePivot(i),
        ) => {
            // A zone the file does not carry is left alone: a curve keyed on
            // a zone that was since dissolved must not conjure one back.
            if let Some(z) = g.zones.as_mut().and_then(|z| z.adjust.get_mut(i as usize)) {
                match ch {
                    ZoneLiftR(_) => z.lift[0] = v,
                    ZoneLiftG(_) => z.lift[1] = v,
                    ZoneLiftB(_) => z.lift[2] = v,
                    ZoneGammaR(_) => z.gamma[0] = v,
                    ZoneGammaG(_) => z.gamma[1] = v,
                    ZoneGammaB(_) => z.gamma[2] = v,
                    ZoneGainR(_) => z.gain[0] = v,
                    ZoneGainG(_) => z.gain[1] = v,
                    ZoneGainB(_) => z.gain[2] = v,
                    ZoneOffsetR(_) => z.offset[0] = v,
                    ZoneOffsetG(_) => z.offset[1] = v,
                    ZoneOffsetB(_) => z.offset[2] = v,
                    ZoneContrast(_) => z.contrast = v,
                    ZonePivot(_) => z.pivot = v,
                    _ => z.saturation = v,
                }
            }
        }
        (NodeKind::Blur(b), _) => crate::nodes::blur::set(b, ch, v),
        (NodeKind::SoftCrop(p), _) => crate::nodes::softcrop::set(p, ch, v),
        (NodeKind::MotionBlur(p), _) => crate::nodes::motionblur::set(p, ch, v),
        (NodeKind::VectorBlur(p), _) => crate::nodes::vectorblur::set(p, ch, v),
        (NodeKind::Camera3D(p), _) => crate::nodes::scene3d::camera_set(p, ch, v),
        (NodeKind::Card3D(p), _) => crate::nodes::scene3d::card_set(p, ch, v),
        (NodeKind::Render3D(p), _) => crate::nodes::scene3d::render_set(p, ch, v),
        (NodeKind::VectorWarp(p), _) => crate::nodes::vectorwarp::set(p, ch, v),
        (NodeKind::Rifetime(p), _) => crate::nodes::rifetime::set(p, ch, v),
        (NodeKind::Freeze(p), _) => crate::nodes::freeze::set(p, ch, v),
        (NodeKind::ColorCompress(p), _) => crate::nodes::colorcompress::set(p, ch, v),
        (NodeKind::HueRotate(p), _) => crate::nodes::huerotate::set(p, ch, v),
        (NodeKind::Lut(p), _) => crate::nodes::lut::set(p, ch, v),
        (NodeKind::Text(p), _) => crate::nodes::text::set(p, ch, v),
        (NodeKind::Env(p), _) => crate::nodes::env::set(p, ch, v),
        (NodeKind::Refract(p), _) => crate::nodes::refract::set(p, ch, v),
        (NodeKind::Starfield(p), _) => crate::nodes::starfield::set(p, ch, v),
        (NodeKind::Splat(p), _) => crate::nodes::splat::set(p, ch, v),
        (NodeKind::Planet(p), _) => crate::nodes::planet::set(p, ch, v),
        (NodeKind::Plasma(p), _) => crate::nodes::plasma::set(p, ch, v),
        (NodeKind::Particles(p), _) => crate::nodes::particles::set(p, ch, v),
        (NodeKind::LensFlare(p), _) => crate::nodes::lensflare::set(p, ch, v),
        (NodeKind::Contrast(p), _) => crate::nodes::contrast::set(p, ch, v),
        (NodeKind::Burnin(p), _) => crate::nodes::burnin::set(p, ch, v),
        (NodeKind::Mirror(p), _) => crate::nodes::mirror::set(p, ch, v),
        (NodeKind::Compare(p), _) => crate::nodes::compare::set(p, ch, v),
        (NodeKind::LensDistort(p), _) => crate::nodes::lensdistort::set(p, ch, v),
        (NodeKind::Gamma(p), _) => crate::nodes::gamma::set(p, ch, v),
        (NodeKind::ToneMap(p), _) => crate::nodes::tonemap::set(p, ch, v),
        (NodeKind::Despill(p), _) => crate::nodes::despill::set(p, ch, v),
        (NodeKind::Lightwrap(p), _) => crate::nodes::lightwrap::set(p, ch, v),
        (NodeKind::ScreenKey(p), _) => crate::nodes::screenkey::set(p, ch, v),
        (NodeKind::ColorSlice(p), _) => crate::nodes::colorslice::set(p, ch, v),

        (NodeKind::Defocus(d), DefocusRadius) => d.radius = v,
        (NodeKind::Defocus(d), DefocusFocus) => d.focus = v,
        (NodeKind::Defocus(d), DefocusFStop) => d.f_stop = v,
        (NodeKind::Defocus(d), DefocusRotation) => d.rotation = v,
        (NodeKind::Defocus(d), DefocusCatseye) => d.catseye = v,
        (NodeKind::Defocus(d), DefocusEdge) => d.edge = v,
        (NodeKind::Defocus(d), DefocusClamp) => d.bright_clamp = v,
        (NodeKind::Blend { mix, .. }, BlendMix) => *mix = v,
        (NodeKind::MatteCombine(m), MatteMix) => m.mix = v,
        (NodeKind::Glow(g), _) => crate::nodes::glow::set(g, ch, v),
        (NodeKind::Godrays(g), _) => crate::nodes::godrays::set(g, ch, v),
        (NodeKind::EdgeExtend(g), _) => crate::nodes::edgeextend::set(g, ch, v),
        (NodeKind::Glint(g), _) => crate::nodes::glint::set(g, ch, v),
        (NodeKind::GlintKernel(p), _) => crate::nodes::glintkernel::set(p, ch, v),
        (NodeKind::BokehKernel(p), _) => crate::nodes::bokehkernel::set(p, ch, v),
        (NodeKind::Diffuse(d), _) => crate::nodes::diffuse::set(d, ch, v),
        (NodeKind::LumaKey(k), LumaLo) => k.band_mut(0).lo = v,
        (NodeKind::LumaKey(k), LumaHi) => k.band_mut(0).hi = v,
        (NodeKind::TimeRemap(p), TimeRemapFrame) => p.frame = v,
        (NodeKind::Switch(p), SwitchWhich) => {
            p.which = v.round().clamp(0.0, (p.n() - 1) as f32) as u32
        }
        (NodeKind::Keyer(k), KeyTilt) => k.tilt = v,
        (NodeKind::Keyer(k), KeyLo) => k.lo = v,
        (NodeKind::Keyer(k), KeyHi) => k.hi = v,
        (NodeKind::Keyer(k), KeyGamma) => k.gamma = v,
        (NodeKind::Keyer(k), KeyCenterL(i)) => blob(k, i, |b| b.center[0] = v),
        (NodeKind::Keyer(k), KeyCenterA(i)) => blob(k, i, |b| b.center[1] = v),
        (NodeKind::Keyer(k), KeyCenterB(i)) => blob(k, i, |b| b.center[2] = v),
        (NodeKind::Keyer(k), KeyTolL(i)) => blob(k, i, |b| b.tol[0] = v),
        (NodeKind::Keyer(k), KeyTolC(i)) => blob(k, i, |b| b.tol[1] = v),
        (NodeKind::Keyer(k), KeyTolH(i)) => blob(k, i, |b| b.tol[2] = v),
        (NodeKind::Keyer(k), KeySoft(i)) => blob(k, i, |b| b.soft = v),
        (NodeKind::Spline(s), SplinePtX(i)) => {
            if let Some(q) = s.points.get_mut(i as usize) {
                q.p[0] = v;
            }
        }
        (NodeKind::Spline(s), SplinePtY(i)) => {
            if let Some(q) = s.points.get_mut(i as usize) {
                q.p[1] = v;
            }
        }
        (NodeKind::Spline(s), SplinePtF(i)) => {
            if let Some(q) = s.points.get_mut(i as usize) {
                q.feather = v;
            }
        }
        (NodeKind::Spline(s), SplineTx) => s.xf.translate[0] = v,
        (NodeKind::Spline(s), SplineTy) => s.xf.translate[1] = v,
        (NodeKind::Spline(s), SplineRot) => s.xf.rotate = v,
        (NodeKind::Spline(s), SplineSx) => s.xf.scale[0] = v,
        (NodeKind::Spline(s), SplineSy) => s.xf.scale[1] = v,
        (NodeKind::Spline(s), SplineCx) => s.xf.center[0] = v,
        (NodeKind::Spline(s), SplineCy) => s.xf.center[1] = v,
        (NodeKind::Spline(s), SplineOpacity) => s.opacity = v,
        (NodeKind::Spline(s), SplineExpand) => s.expand = v,
        (NodeKind::Spline(s), SplineFalloff) => s.falloff = v,
        (NodeKind::Spline(s), SplineFeatherOut) => s.feather_out = v.max(0.0),
        (NodeKind::Spline(s), SplineFeatherIn) => s.feather_in = v.max(0.0),
        (NodeKind::Spline(s), SplinePtB(i)) => {
            if let Some(q) = s.points.get_mut(i as usize) {
                q.bias = v;
            }
        }
        (NodeKind::Transform(t), Pin0X) => t.pin[0][0] = v,
        (NodeKind::Transform(t), Pin0Y) => t.pin[0][1] = v,
        (NodeKind::Transform(t), Pin1X) => t.pin[1][0] = v,
        (NodeKind::Transform(t), Pin1Y) => t.pin[1][1] = v,
        (NodeKind::Transform(t), Pin2X) => t.pin[2][0] = v,
        (NodeKind::Transform(t), Pin2Y) => t.pin[2][1] = v,
        (NodeKind::Transform(t), Pin3X) => t.pin[3][0] = v,
        (NodeKind::Transform(t), Pin3Y) => t.pin[3][1] = v,
        (NodeKind::CornerPin(cp), Pin0X) => cp.to[0][0] = v,
        (NodeKind::CornerPin(cp), Pin0Y) => cp.to[0][1] = v,
        (NodeKind::CornerPin(cp), Pin1X) => cp.to[1][0] = v,
        (NodeKind::CornerPin(cp), Pin1Y) => cp.to[1][1] = v,
        (NodeKind::CornerPin(cp), Pin2X) => cp.to[2][0] = v,
        (NodeKind::CornerPin(cp), Pin2Y) => cp.to[2][1] = v,
        (NodeKind::CornerPin(cp), Pin3X) => cp.to[3][0] = v,
        (NodeKind::CornerPin(cp), Pin3Y) => cp.to[3][1] = v,
        (NodeKind::CornerPin(cp), From0X) => cp.from[0][0] = v,
        (NodeKind::CornerPin(cp), From0Y) => cp.from[0][1] = v,
        (NodeKind::CornerPin(cp), From1X) => cp.from[1][0] = v,
        (NodeKind::CornerPin(cp), From1Y) => cp.from[1][1] = v,
        (NodeKind::CornerPin(cp), From2X) => cp.from[2][0] = v,
        (NodeKind::CornerPin(cp), From2Y) => cp.from[2][1] = v,
        (NodeKind::CornerPin(cp), From3X) => cp.from[3][0] = v,
        (NodeKind::CornerPin(cp), From3Y) => cp.from[3][1] = v,
        (NodeKind::Mix(m), MixAmount) => m.amount = v,
        (NodeKind::Saturation(sp), _) => crate::nodes::saturation::set(sp, ch, v),
        (NodeKind::Gradient(p), _) => crate::nodes::gradient::set(p, ch, v),
        (NodeKind::GridWarp(p), _) => crate::nodes::gridwarp::set(p, ch, v),
        (NodeKind::Warp(p), _) => crate::nodes::warp::set(p, ch, v),
        (NodeKind::Grain(p), _) => crate::nodes::grain::set(p, ch, v),
        (NodeKind::Dilate(d), _) => crate::nodes::dilate::set(d, ch, v),
        (NodeKind::EdgeDetect(p), _) => crate::nodes::edgedetect::set(p, ch, v),
        (NodeKind::Median(p), _) => crate::nodes::median::set(p, ch, v),
        (NodeKind::HighlightRebuild(p), _) => crate::nodes::highlightrebuild::set(p, ch, v),
        (NodeKind::LtxHdr(p), _) => crate::nodes::ltxhdr::set(p, ch, v),
        (NodeKind::Stabilize(p), _) => crate::nodes::stabilize::set(p, ch, v),
        (NodeKind::Convolve(p), _) => crate::nodes::convolve::set(p, ch, v),
        (NodeKind::Blend { gain, .. }, BlendGainA) => gain[0] = v,
        (NodeKind::Blend { gain, .. }, BlendGainB) => gain[1] = v,
        (NodeKind::LoopOut(p), LoopN) => p.n = v,
        (NodeKind::Grade(g), GradeSat) => g.saturation = v,
        (NodeKind::Grade(g), GradeContrast) => g.contrast = v,
        (NodeKind::Grade(g), GradePivot) => g.pivot = v,
        (NodeKind::Noise(n), NoiseProbeX) => n.probe[0] = v,
        (NodeKind::Noise(n), NoiseProbeY) => n.probe[1] = v,
        (NodeKind::Noise(n), NoiseScale) => n.scale = v,
        (NodeKind::Noise(n), NoiseSeed) => n.seed = v,
        (NodeKind::Noise(n), NoiseSpeed) => n.speed = v,
        (NodeKind::Noise(n), NoiseSeedOffset) => n.seed_offset = v,
        (NodeKind::Mult(m), MultR) => m.value[0] = v,
        (NodeKind::Mult(m), MultG) => m.value[1] = v,
        (NodeKind::Mult(m), MultB) => m.value[2] = v,
        (NodeKind::Grid(g), GridSpacingX) => g.spacing[0] = v,
        (NodeKind::Grid(g), GridSpacingY) => g.spacing[1] = v,
        (NodeKind::Grid(g), GridWidth) => g.width = v,
        (NodeKind::Grid(g), GridOffsetX) => g.offset[0] = v,
        (NodeKind::Grid(g), GridOffsetY) => g.offset[1] = v,
        (NodeKind::Reformat(r), ReformatScale) => r.scale = v,
        (NodeKind::Tracker(t), TrackX) => t.track[0] = v,
        (NodeKind::Tracker(t), TrackY) => t.track[1] = v,
        (NodeKind::Tracker(t), TrackPtX(i) | TrackPtY(i)) if i > 0 => {
            if let Some(mut p) = t.point(i as usize) {
                p.track[matches!(ch, TrackPtY(_)) as usize] = v;
                t.set_point(i as usize, p);
            }
        }
        (NodeKind::PlanarTrack(p), PlanarDX) => p.drive[0] = v,
        (NodeKind::PlanarTrack(p), PlanarDY) => p.drive[1] = v,
        (NodeKind::PlanarTrack(p), PlanarRot) => p.drive[2] = v,
        (NodeKind::PlanarTrack(p), PlanarScale) => p.drive[3] = v,
        (NodeKind::PlanarTrack(p), PlanarPin0X) => p.pin[0][0] = v,
        (NodeKind::PlanarTrack(p), PlanarPin0Y) => p.pin[0][1] = v,
        (NodeKind::PlanarTrack(p), PlanarPin1X) => p.pin[1][0] = v,
        (NodeKind::PlanarTrack(p), PlanarPin1Y) => p.pin[1][1] = v,
        (NodeKind::PlanarTrack(p), PlanarPin2X) => p.pin[2][0] = v,
        (NodeKind::PlanarTrack(p), PlanarPin2Y) => p.pin[2][1] = v,
        (NodeKind::PlanarTrack(p), PlanarPin3X) => p.pin[3][0] = v,
        (NodeKind::PlanarTrack(p), PlanarPin3Y) => p.pin[3][1] = v,
        (NodeKind::Layer(p), LayerOpacity(i)) => layer(p, i, |l| l.opacity = v),
        (NodeKind::Recombine(p), RecombineGain(i)) => {
            let i = i as usize;
            if p.gain.len() <= i {
                p.gain.resize(i + 1, 1.0);
            }
            p.gain[i] = v;
        }
        (NodeKind::Field(f), Field(i)) => {
            f.set_literal(i as usize, v);
        }
        (NodeKind::GeoNode(n), Field(i)) => {
            n.set_literal(i as usize, v);
        }
        (NodeKind::SdfNode(n), Field(i)) => {
            n.set_literal(i as usize, v);
        }
        (NodeKind::Geo(g), GeoSeed) => g.seed = v,
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Noise on a curve: what the channel reads is the keys plus the
    /// wiggle, a value-space write lands exactly where it was aimed, and a
    /// curve without noise saves as it always did.
    #[test]
    fn a_noised_curve_reads_keys_plus_noise() {
        let mut c = AnimCurve::default();
        c.set_key(0u64, 10.0);
        c.set_key(20u64, 30.0);
        let plain = serde_json::to_string(&c).unwrap();
        assert!(!plain.contains("noise"), "{plain}");
        c.noise = Some(crate::animnoise::CurveNoise { strength: 8.0, ..Default::default() });
        for t in [0.0, 7.5, 13.0, 40.0] {
            let want = c.eval_base_at(t).unwrap() + c.noise.as_ref().unwrap().at(t);
            assert_eq!(c.eval_at(t), Some(want));
        }
        assert!((0..40).any(|t| (c.eval_at(t as f32).unwrap() - c.eval_base_at(t as f32).unwrap()).abs() > 0.5));
        // A property row showing 25 at frame 13 keys so it READS 25 there.
        c.set_key_shown(13u64, 25.0);
        assert!((c.eval_at(13.0).unwrap() - 25.0).abs() < 1e-4);
        let back: AnimCurve = serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
        assert_eq!(back, c);
        // No keys, no value: noise alone never invents one.
        let empty = AnimCurve { noise: c.noise.clone(), ..Default::default() };
        assert_eq!(empty.eval_at(3.0), None);
    }

    /// A 2-d Noise hands out two numbers at the probe — x the field, y the
    /// same field at the offset seed — so two cables can drive x and y with
    /// one character. A 1-d Noise has no y to cable.
    #[test]
    fn a_two_d_noise_cables_out_two_values() {
        let mut p = engine::graph::NoiseParams {
            graph: engine::field::presets::NoisePreset::Clouds.graph(),
            probe: [0.3, 0.2],
            ..Default::default()
        };
        assert_eq!(get(&NodeKind::Noise(p.clone()), Channel::NoiseValueY), None);
        p.two_d = true;
        let mut k = NodeKind::Noise(p);
        let x = get(&k, Channel::NoiseValue).unwrap();
        let y = get(&k, Channel::NoiseValueY).unwrap();
        assert!((x - y).abs() > 1e-3, "y is x again ({x})");
        set(&mut k, Channel::NoiseSeedOffset, 7.0);
        assert_eq!(get(&k, Channel::NoiseSeedOffset), Some(7.0));
        assert_ne!(get(&k, Channel::NoiseValueY), Some(y), "the offset moves y");
        assert_eq!(get(&k, Channel::NoiseValue), Some(x), "and leaves x alone");
        for c in [Channel::NoiseValueY, Channel::NoiseSeedOffset] {
            assert_eq!(Channel::from_key(&c.key()), Some(c));
        }
    }

    /// A motion path samples where the element ACTUALLY is, frame by frame:
    /// keys marked, the curve's shape between them, and nothing at all for
    /// an element that does not move.
    #[test]
    fn a_motion_path_follows_the_curve_and_marks_its_keys() {
        use engine::graph::TransformParams;
        let point = |k: &NodeKind| match k {
            NodeKind::Transform(t) => Some([t.translate[0], t.translate[1]]),
            _ => None,
        };
        let kind = NodeKind::Transform(TransformParams::default());
        let mut a = Anim::new();
        // x goes 0 -> 100 over ten frames, y stays put: a straight run.
        let c = a.entry(Channel::TranslateX).or_default();
        c.set_key(0, 0.0);
        c.set_key(10, 100.0);
        let path = motion_path(&kind, &a, (0, 10), &point);
        assert_eq!(path.len(), 11, "one sample a frame over a short span");
        assert_eq!(path.first().map(|p| p.at[0]), Some(0.0));
        assert_eq!(path.last().map(|p| p.at[0]), Some(100.0));
        assert!(path[5].at[0] > 40.0 && path[5].at[0] < 60.0, "sampled mid-curve");
        assert!(path[0].key && path[10].key, "both keys marked");
        assert!(!path[5].key, "and the frames between them are not");

        // A node with no position has no path, however animated it is.
        let mut g = Anim::new();
        g.entry(Channel::TranslateX).or_default().set_key(0, 2.0);
        assert!(motion_path(&kind, &g, (0, 10), &|_| None).is_empty());

        // Neither does an element whose curves never move it.
        let mut still = Anim::new();
        let c = still.entry(Channel::TranslateX).or_default();
        c.set_key(0, 7.0);
        c.set_key(10, 7.0);
        assert!(motion_path(&kind, &still, (0, 10), &point).is_empty());

        // A long span is strided to a few hundred samples — but never over a
        // key, which is where the shape is decided.
        let mut long = Anim::new();
        let c = long.entry(Channel::TranslateX).or_default();
        c.set_key(0, 0.0);
        c.set_key(1234.5, 100.0);
        let path = motion_path(&kind, &long, (0, 5000), &point);
        assert!(
            path.len() <= PATH_SAMPLES as usize + 4,
            "strided, not one sample a frame: {}",
            path.len()
        );
        assert!(
            path.iter().any(|p| (p.frame - 1234.5).abs() < 1e-6 && p.key),
            "the sub-frame key was sampled exactly"
        );
    }

    fn pt(x: f32, y: f32) -> SplinePoint {
        SplinePoint::new([x, y], 10.0)
    }

    /// Two animated points, keyed at frames 0 and 10.
    fn animated_pair() -> Anim {
        let mut a = Anim::new();
        for (i, x0, x1) in [(0u16, 0.0f32, 0.2f32), (1, 1.0, 1.2)] {
            let c = a.entry(Channel::SplinePtX(i)).or_default();
            c.set_key(0, x0);
            c.set_key(10, x1);
            let c = a.entry(Channel::SplinePtY(i)).or_default();
            c.set_key(0, 0.0);
            c.set_key(10, 0.0);
        }
        a
    }

    #[test]
    fn inserting_a_point_slides_the_curves_along() {
        let mut a = animated_pair();
        // [p0, NEW, p1, p2] — the old point 1's curve has to end up on 2.
        let pts = [pt(0.0, 0.0), pt(0.5, 0.0), pt(1.0, 0.0), pt(0.5, 1.0)];
        spline_point_inserted(&mut a, &pts, 1, 0);
        assert_eq!(a[&Channel::SplinePtX(2)].eval(0), Some(1.0), "old curve moved up");
        assert_eq!(a[&Channel::SplinePtX(2)].eval(10), Some(1.2));
        // The new point follows its neighbours, holding the offset it was
        // dropped with (here: exactly the midpoint at t=0).
        let nx = &a[&Channel::SplinePtX(1)];
        assert_eq!(nx.keys.len(), 2, "keyed on the shape's own frames");
        assert!((nx.eval(0).unwrap() - 0.5).abs() < 1e-5);
        assert!((nx.eval(10).unwrap() - 0.7).abs() < 1e-5, "midpoint of 0.2 and 1.2");
    }

    #[test]
    fn an_offset_new_point_keeps_its_offset() {
        let mut a = animated_pair();
        // Dropped 0.1 above the midpoint: it should stay 0.1 above at every
        // key, not snap onto the chord.
        let pts = [pt(0.0, 0.0), pt(0.5, 0.1), pt(1.0, 0.0), pt(0.5, 1.0)];
        spline_point_inserted(&mut a, &pts, 1, 0);
        let ny = &a[&Channel::SplinePtY(1)];
        assert!((ny.eval(0).unwrap() - 0.1).abs() < 1e-5);
        assert!((ny.eval(10).unwrap() - 0.1).abs() < 1e-5);
    }

    #[test]
    fn a_static_shape_gains_no_curves() {
        let mut a = Anim::new();
        let pts = [pt(0.0, 0.0), pt(0.5, 0.0), pt(1.0, 0.0)];
        spline_point_inserted(&mut a, &pts, 1, 0);
        assert!(a.is_empty(), "an unanimated spline stays unanimated");
    }

    #[test]
    fn removing_a_point_slides_the_curves_back() {
        let mut a = animated_pair();
        spline_point_removed(&mut a, 0, 2);
        assert_eq!(a[&Channel::SplinePtX(0)].eval(0), Some(1.0), "point 1 became 0");
        assert!(!a.contains_key(&Channel::SplinePtX(1)), "no orphan left behind");
        spline_channels_cleared(&mut a);
        assert!(a.is_empty());
    }

    #[test]
    fn the_shape_group_is_every_point() {
        assert_eq!(spline_shape(3).len(), 6);
        // A point's band is its feather AND where it sits: they key as one.
        assert_eq!(spline_feather(3), vec![
            Channel::SplinePtF(0),
            Channel::SplinePtB(0),
            Channel::SplinePtF(1),
            Channel::SplinePtB(1),
            Channel::SplinePtF(2),
            Channel::SplinePtB(2)
        ]);
    }

    #[test]
    fn curve_holds_and_lerps() {
        let mut c = AnimCurve::default();
        assert_eq!(c.eval(5), None);
        c.set_key(10, 0.0);
        c.set_key(20, 100.0);
        assert_eq!(c.eval(0), Some(0.0)); // hold before
        assert_eq!(c.eval(10), Some(0.0));
        assert_eq!(c.eval(15), Some(50.0)); // lerp
        assert_eq!(c.eval(20), Some(100.0));
        assert_eq!(c.eval(99), Some(100.0)); // hold after
        // Replacing a key keeps order; out-of-order insert sorts in.
        c.set_key(10, 10.0);
        c.set_key(5, -5.0);
        assert_eq!(c.keys.iter().map(|k| k.frame).collect::<Vec<_>>(), vec![5.0, 10.0, 20.0]);
        assert_eq!(c.eval(10), Some(10.0));
        c.remove_key(10);
        assert!(!c.has_key(10));
        assert_eq!(c.keys.len(), 2);
    }

    #[test]
    fn interp_modes_shape_the_segment() {
        let mut c = AnimCurve::default();
        c.set_key(0, 0.0);
        c.set_key(10, 100.0);
        // Linear is the default and hits the midpoint exactly.
        assert_eq!(c.eval_at(5.0), Some(50.0));

        // Hold steps: the value stays put until the next key lands.
        c.set_interp(0, Interp::Hold);
        assert_eq!(c.eval_at(5.0), Some(0.0));
        assert_eq!(c.eval_at(9.99), Some(0.0));
        assert_eq!(c.eval_at(10.0), Some(100.0));

        // Smooth eases: slower than linear at the start, still monotonic,
        // still exact on the keys.
        c.set_interp(0, Interp::Smooth);
        let q = c.eval_at(2.5).unwrap();
        assert!(q < 20.0, "ease-in should lag linear, got {q}");
        assert_eq!(c.eval_at(0.0), Some(0.0));
        assert_eq!(c.eval_at(10.0), Some(100.0));
        let (a, b) = (c.eval_at(3.0).unwrap(), c.eval_at(7.0).unwrap());
        assert!(a < b, "smooth segment must stay monotonic");

        // Bezier seeds handles from the current slope, so switching modes
        // doesn't jump the shape, and it stays pinned to both keys.
        c.set_interp(0, Interp::Bezier);
        assert_eq!(c.eval_at(0.0), Some(0.0));
        assert_eq!(c.eval_at(10.0), Some(100.0));
        let mid = c.eval_at(5.0).unwrap();
        assert!((mid - 50.0).abs() < 20.0, "seeded bezier way off: {mid}");
    }

    #[test]
    fn smooth_does_not_overshoot_a_plateau() {
        // The classic auto-tangent failure: rise then hold flat. A naive
        // Catmull-Rom bulges above 100 between the last two keys.
        let mut c = AnimCurve::default();
        for (f, v) in [(0u64, 0.0f32), (10, 100.0), (20, 100.0)] {
            c.set_key(f, v);
            c.set_interp(f, Interp::Smooth);
        }
        for i in 0..=200 {
            let t = i as f32 * 0.1;
            let v = c.eval_at(t).unwrap();
            assert!(v <= 100.001 && v >= -0.001, "overshoot at t={t}: {v}");
        }
    }

    #[test]
    fn bezier_handles_bend_the_curve_and_stay_solvable() {
        let mut c = AnimCurve::default();
        c.set_key(0, 0.0);
        c.set_key(10, 100.0);
        c.set_interp(0, Interp::Bezier);
        // Yank the outgoing handle far past the segment: eval must stay
        // finite, pinned at the keys, and inside a sane band.
        c.keys[0].tan_out = (999.0, -500.0);
        c.keys[1].tan_in = (-999.0, 500.0);
        assert_eq!(c.eval_at(0.0), Some(0.0));
        assert_eq!(c.eval_at(10.0), Some(100.0));
        for i in 0..=100 {
            let v = c.eval_at(i as f32 * 0.1).unwrap();
            assert!(v.is_finite(), "non-finite bezier value");
        }
    }

    #[test]
    fn new_keys_inherit_neighbouring_interpolation() {
        let mut c = AnimCurve::default();
        c.set_key(0, 0.0);
        c.set_key(20, 1.0);
        c.set_interp(0, Interp::Hold);
        // Autokey drops a key inside the held segment: it stays held.
        c.set_key(10, 0.5);
        assert_eq!(c.keys[1].interp, Interp::Hold);
        // Replacing an existing key keeps its mode and only moves the value.
        c.set_key(10, 0.75);
        assert_eq!(c.keys[1].interp, Interp::Hold);
        assert_eq!(c.keys[1].value, 0.75);
    }

    #[test]
    fn legacy_pair_keys_still_deserialize() {
        // Setups written before interpolation modes existed.
        let old: AnimCurve = serde_json::from_str(r#"{"keys":[[0,1.0],[12,2.0]]}"#).unwrap();
        assert_eq!(old.keys.len(), 2);
        assert_eq!(old.keys[0].frame, 0.0);
        assert_eq!(old.keys[1].value, 2.0);
        assert_eq!(old.keys[0].interp, Interp::Linear);
        assert_eq!(old.eval(6), Some(1.5)); // unchanged behaviour
        // And the round trip through the new form survives.
        let json = serde_json::to_string(&old).unwrap();
        // Whole frames still write as integers — older builds read them.
        assert!(json.contains(r#""frame":12,"#), "{json}");
        let back: AnimCurve = serde_json::from_str(&json).unwrap();
        assert_eq!(back, old);
    }

    #[test]
    fn sub_frame_keys_evaluate_and_round_trip() {
        let mut c = AnimCurve::default();
        c.set_key(0, 0.0);
        c.set_key(2.5, 10.0);
        c.set_key(5, 0.0);
        assert_eq!(c.keys.iter().map(|k| k.frame).collect::<Vec<_>>(), vec![0.0, 2.5, 5.0]);
        assert_eq!(c.eval_at(2.5), Some(10.0));
        assert!((c.eval(1).unwrap() - 4.0).abs() < 1e-5, "linear toward the 2.5 key");
        assert!(c.has_key(2.5) && !c.has_key(2));
        let json = serde_json::to_string(&c).unwrap();
        assert!(json.contains(r#""frame":2.5"#), "{json}");
        let back: AnimCurve = serde_json::from_str(&json).unwrap();
        assert_eq!(back, c);
        c.remove_key(2.5);
        assert_eq!(c.keys.len(), 2);
    }

    #[test]
    fn ensure_bezier_keeps_existing_handles() {
        let mut c = AnimCurve::default();
        for (f, v) in [(0u64, 0.0), (10, 10.0), (20, 0.0)] {
            c.set_key(f, v);
        }
        c.set_interp(10, Interp::Bezier);
        c.keys[1].tan_out = (2.0, -7.0);
        c.ensure_bezier(10);
        assert_eq!(c.keys[1].tan_out, (2.0, -7.0), "the existing out handle survives");
        assert_eq!(c.keys[0].interp, Interp::Bezier, "the arriving side became bezier");
        assert!(c.keys[1].tan_in.0 < 0.0, "and got an in handle");
    }

    #[test]
    fn set_interp_both_sets_the_arriving_segment_too() {
        let mut c = AnimCurve::default();
        for f in [0u64, 10, 20] {
            c.set_key(f, f as f32);
        }
        c.set_interp_both(10, Interp::Smooth);
        let modes: Vec<_> = c.keys.iter().map(|k| k.interp).collect();
        assert_eq!(modes, vec![Interp::Smooth, Interp::Smooth, Interp::Linear]);
        // The first key has no arriving side: only its own changes.
        c.set_interp_both(0, Interp::Hold);
        assert_eq!(c.keys[0].interp, Interp::Hold);
        assert_eq!(c.keys[1].interp, Interp::Smooth);
    }

    #[test]
    fn channel_get_set_roundtrip() {
        let mut k = NodeKind::Transform(Default::default());
        set(&mut k, Channel::Rotate, 33.0);
        assert_eq!(get(&k, Channel::Rotate), Some(33.0));
        // Mismatched pair: no-op, get None.
        set(&mut k, Channel::BlurSize, 5.0);
        assert_eq!(get(&k, Channel::BlurSize), None);
        let mut g = NodeKind::Grade(Default::default());
        set(&mut g, Channel::GainG, 2.5);
        assert_eq!(get(&g, Channel::GainG), Some(2.5));
        let mut cp = NodeKind::CornerPin(Default::default());
        set(&mut cp, Channel::Pin2X, 0.9);
        assert_eq!(get(&cp, Channel::Pin2X), Some(0.9));
    }

    fn ramp() -> AnimCurve {
        let mut c = AnimCurve::default();
        c.set_key(10u64, 0.0);
        c.set_key(20u64, 10.0);
        c
    }

    #[test]
    fn extrapolation_carries_on_past_the_keys() {
        let mut c = ramp();
        assert_eq!((c.eval_at(0.0), c.eval_at(30.0)), (Some(0.0), Some(10.0)), "hold, as before");
        c.pre = Extend::Linear;
        c.post = Extend::Linear;
        assert_eq!((c.eval_at(5.0), c.eval_at(30.0)), (Some(-5.0), Some(20.0)), "linear: the chord's slope");
        c.post = Extend::Cycle;
        assert_eq!(c.eval_at(25.0), Some(5.0), "cycle repeats the span");
        assert_eq!(c.eval_at(30.0), Some(0.0), "and starts again at the first key");
        c.post = Extend::CycleOffset;
        assert_eq!(c.eval_at(25.0), Some(15.0), "cycle + offset climbs");
        assert_eq!(c.eval_at(45.0), Some(35.0), "three spans on");
        c.post = Extend::Oscillate;
        assert_eq!(c.eval_at(25.0), Some(5.0));
        assert_eq!(c.eval_at(28.0), Some(2.0), "on the way back down");
        assert_eq!(c.eval_at(32.0), Some(2.0), "and up again");
        c.pre = Extend::Cycle;
        assert_eq!(c.eval_at(5.0), Some(5.0), "before the first key too");
        // A bezier end extrapolates along its handle.
        let mut b = ramp();
        b.set_interp(10u64, Interp::Bezier);
        b.keys[1].tan_in = (-2.0, -1.0);
        b.post = Extend::Linear;
        assert!((b.eval_at(24.0).unwrap() - 12.0).abs() < 1e-4, "{:?}", b.eval_at(24.0));
        // Hold ends save as before (no new fields in the file).
        let json = serde_json::to_string(&ramp()).unwrap();
        assert!(!json.contains("pre") && !json.contains("post"), "{json}");
        let back: AnimCurve = serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
        assert_eq!(back, c);
    }

    #[test]
    fn stepped_holds_the_curve_and_its_noise_between_steps() {
        let mut c = ramp();
        c.stepped = Some(Stepped { step: 4.0, offset: 0.0 });
        assert_eq!(c.eval_at(13.0), Some(2.0), "12 is read until 16");
        assert_eq!(c.eval_at(15.9), Some(2.0));
        assert_eq!(c.eval_at(16.0), Some(6.0));
        assert_eq!(c.eval_base_at(13.0), Some(3.0), "the keys' own curve is unstepped");
        c.noise = Some(crate::animnoise::CurveNoise::default());
        assert_eq!(c.eval_at(13.0), c.eval_at(14.5), "the noise holds with it");
        c.stepped = Some(Stepped { step: 4.0, offset: 1.0 });
        assert_eq!(c.eval_base_at(13.0), c.eval_at(13.0).map(|v| v - c.noise_at(13.0)), "offset moves the lattice");
    }

    #[test]
    fn eases_run_from_value_to_value_and_keep_their_shape() {
        for e in Ease::ALL {
            for d in EaseDir::ALL {
                assert!(ease_shape(e, d, 0.0).abs() < 1e-5, "{e:?} {d:?} at 0");
                assert!((ease_shape(e, d, 1.0) - 1.0).abs() < 1e-5, "{e:?} {d:?} at 1");
            }
        }
        // In is slow first, out fast first, in-out symmetric.
        assert!(ease_shape(Ease::Cubic, EaseDir::In, 0.25) < 0.25);
        assert!(ease_shape(Ease::Cubic, EaseDir::Out, 0.25) > 0.25);
        assert!((ease_shape(Ease::Sine, EaseDir::InOut, 0.5) - 0.5).abs() < 1e-5);
        // Back overshoots, bounce never leaves 0..1.
        assert!((0..100).any(|i| ease_shape(Ease::Back, EaseDir::Out, i as f32 / 100.0) > 1.0));
        assert!((0..=100).all(|i| (0.0..=1.0001).contains(&ease_shape(Ease::Bounce, EaseDir::Out, i as f32 / 100.0))));
        let mut c = ramp();
        c.set_interp(10u64, Interp::Ease(Ease::Quad, EaseDir::In));
        assert_eq!(c.eval_at(15.0), Some(2.5));
        let back: AnimCurve = serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
        assert_eq!(back, c);
    }

    #[test]
    fn a_vector_handle_follows_its_neighbour_and_saves() {
        let mut c = ramp();
        c.set_interp(10u64, Interp::Bezier);
        c.keys[0].vec_out = true;
        c.keys[1].vec_in = true;
        // Both handles on the chord: the bezier IS the straight line.
        assert!((c.eval_at(15.0).unwrap() - 5.0).abs() < 1e-3);
        c.keys[1].value = 30.0;
        assert!((c.eval_at(15.0).unwrap() - 15.0).abs() < 1e-3, "followed the move");
        let back: AnimCurve = serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
        assert_eq!(back, c);
        let plain = serde_json::to_string(&ramp()).unwrap();
        assert!(!plain.contains("vec_"), "{plain}");
    }
}
