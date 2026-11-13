//! User-remappable hotkeys.
//!
//! One table ([`DEFS`]) declares every remappable action — a stable string
//! id (what prefs store), a label, the pane that owns it, the default
//! binding and how it matches. Sites read LIVE through [`pressed`] /
//! [`released`] / [`down`] / [`consume`], so a rebind takes effect the same
//! frame with nothing to invalidate. Prefs hold only the OVERRIDES, as
//! human-readable strings (`"graph.tidy": "cmd+T"`): an action whose
//! binding never moved follows any future default, and a stale id from an
//! old build is skipped with a warning rather than wedging the map.
//!
//! Matching, because the old hardcoded reads had two different shapes:
//!
//! * [`MatchMode::Exact`] compares the key event's OWN modifiers with
//!   `Modifiers::matches_exact`. That keeps `P` and `shift+P` two separate
//!   bindings, makes a plain-letter binding ignore its cmd chord cousin
//!   without the `!i.modifiers.command` guards the call sites used to
//!   carry, and — because the modifiers come off the EVENT, not the global
//!   state — works for synthetic test input exactly the way the picker's
//!   hand-rolled event scan did.
//! * [`MatchMode::KeyOnly`] matches the bare key, for sites that read the
//!   modifiers themselves as part of the gesture: the brush pair (shift =
//!   softness, alt = opacity), Z/X pinning (shift = replace), the held
//!   align swipe, play/pause (matched on release).
//!
//! Deliberately NOT here: the navigation chords (`nav.rs` — Blender's
//! hands, manifesto 11) and the rgb slider's held colour axes (S/D/F/G, a
//! design invariant the F-framing collision already paid for once). The
//! keys window says so rather than offering rows that would break them.

use crate::theme;
use egui::{Key, Modifiers};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::RwLock;

/// A concrete key chord. `mods` is NORMALIZED (see [`norm`]): `command`
/// covers ⌘/ctrl cross-platform, `ctrl` alone means the mac ctrl key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Binding {
    pub key: Key,
    pub mods: Modifiers,
}

impl Binding {
    pub const fn plain(key: Key) -> Self {
        Self { key, mods: Modifiers::NONE }
    }
    pub const fn cmd(key: Key) -> Self {
        Self { key, mods: Modifiers::COMMAND }
    }
    pub const fn cmd_shift(key: Key) -> Self {
        Self {
            key,
            mods: Modifiers {
                alt: false,
                ctrl: false,
                shift: true,
                mac_cmd: false,
                command: true,
            },
        }
    }
    /// The Control key itself — on a Mac, the one beside ⌥, not ⌘.
    pub const fn ctrl(key: Key) -> Self {
        Self {
            key,
            mods: Modifiers {
                alt: false,
                ctrl: true,
                shift: false,
                mac_cmd: false,
                command: false,
            },
        }
    }
    /// Control + shift — Blender's modifier chords on a Mac too.
    pub const fn ctrl_shift(key: Key) -> Self {
        Self {
            key,
            mods: Modifiers {
                alt: false,
                ctrl: true,
                shift: true,
                mac_cmd: false,
                command: false,
            },
        }
    }
    /// ⌥ on a Mac.
    pub const fn alt(key: Key) -> Self {
        Self {
            key,
            mods: Modifiers {
                alt: true,
                ctrl: false,
                shift: false,
                mac_cmd: false,
                command: false,
            },
        }
    }
    pub const fn shift(key: Key) -> Self {
        Self {
            key,
            mods: Modifiers {
                alt: false,
                ctrl: false,
                shift: true,
                mac_cmd: false,
                command: false,
            },
        }
    }
}

/// Drop `mac_cmd` (an event detail `command` already carries) and fold the
/// non-mac ctrl mirror into `command`, so a binding captured on one
/// platform reads back on another. `ctrl` survives alone only where it is
/// a real, separate key: the mac ctrl.
fn norm(m: Modifiers) -> Modifiers {
    Modifiers {
        alt: m.alt,
        ctrl: m.ctrl && !m.command,
        shift: m.shift,
        mac_cmd: false,
        command: m.command,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatchMode {
    /// Event modifiers must equal the binding's (`matches_exact`).
    Exact,
    /// Bare key; the site reads the modifiers itself.
    KeyOnly,
}

/// The pane a binding belongs to. Conflicts are flagged WITHIN a scope
/// only: the pane under the cursor owns single-letter keys (see CLAUDE.md),
/// so graph `D` (bypass) and app `D` (back to layout) coexisting is the
/// design, not a clash.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Scope {
    App,
    Viewer,
    Graph,
    Curves,
    Paint,
    /// The properties panels (docked strip or popped out).
    Props,
}

impl Scope {
    pub fn title(self) -> &'static str {
        match self {
            Scope::App => "app",
            Scope::Viewer => "viewer",
            Scope::Graph => "graph",
            Scope::Curves => "curves",
            Scope::Paint => "paint",
            Scope::Props => "properties",
        }
    }
    /// The keyboard's scope tint: the dots on a keycap, the ring on a
    /// selected command's key, the legend chips and the list headers.
    pub fn color(self) -> egui::Color32 {
        match self {
            Scope::App => theme::keys_app(),
            Scope::Viewer => theme::keys_viewer(),
            Scope::Graph => theme::keys_graph(),
            Scope::Curves => theme::keys_curves(),
            Scope::Paint => theme::keys_paint(),
            Scope::Props => theme::keys_props(),
        }
    }
    pub const ALL: [Scope; 6] =
        [Scope::App, Scope::Viewer, Scope::Graph, Scope::Curves, Scope::Paint, Scope::Props];
}

/// Every remappable action. The enum is the call-site handle; the string
/// id in its [`Def`] is the stable name prefs use. Adding one = one row in
/// [`DEFS`] + reading it somewhere.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Act {
    Save,
    SaveAs,
    Open,
    Undo,
    Redo,
    Record,
    TimelineIn,
    TimelineOut,
    TimelinePie,
    TimelineFrame,
    FullscreenPane,
    GraphOverlay,
    BackToLayout,
    ToggleProps,
    HudOff,
    CurveEditor,
    Browse,
    PlayPause,
    StepBack,
    StepForward,
    ToggleAlpha,
    ViewRed,
    ViewGreen,
    ViewBlue,
    PixelReadout,
    PinSample,
    FrameView,
    View3dFront,
    View3dBack,
    View3dRight,
    View3dLeft,
    View3dTop,
    View3dBottom,
    View3dOrbitLeft,
    View3dOrbitRight,
    View3dOrbitUp,
    View3dOrbitDown,
    View3dOpposite,
    VectorHud,
    DiveIn,
    DiveOut,
    GraphSearch,
    GraphDelete,
    GraphLinkMode,
    GraphPatchMode,
    GraphOpsPie,
    PropsPie,
    PropsCurveFrame,
    ViewerPie,
    GraphNodePie,
    GraphRename,
    GraphSwipeIn,
    GraphSwipeOut,
    GraphSwipeBlend,
    GraphTv,
    GraphSourcePie,
    GraphRegionPie,
    GraphRegionMenu,
    GraphBypass,
    GraphFrame,
    GraphUnifySize,
    GraphPinPanels,
    GraphPinGizmos,
    GraphGrab,
    GraphWireZDown,
    GraphWireZUp,
    GraphSwapInputs,
    GraphTidy,
    GraphRanges,
    GraphDot,
    GraphGroup,
    GraphUngroup,
    GraphSubwob,
    GraphBakeSelected,
    GraphCacheSelected,
    GraphView1,
    GraphView2,
    GraphView3,
    GraphView4,
    GraphViewLock,
    GraphSwapUp,
    GraphSwapDown,
    GraphAlign,
    GraphSelectAll,
    GraphCopy,
    GraphPaste,
    CurvesFrame,
    CurvesPie,
    CurvesSnap,
    CurvesOnlyTime,
    CurvesOnlyValue,
    CurvesUnify,
    CurvesBreak,
    CurvesNoise,
    CurvesGrab,
    CurvesScale,
    CurvesRotate,
    CurvesCycle,
    CurvesSmooth,
    CurvesMirror,
    CurvesSelectLeft,
    CurvesSelectRight,
    CurvesSelectColumn,
    CurvesLadder,
    CurvesCopy,
    CurvesPaste,
    CurvesFrameSelected,
    BrushDown,
    BrushUp,
    /// Held: pointer motion sets the brush (Blender's F).
    BrushAdjust,
}

pub struct Def {
    pub act: Act,
    /// Stable prefs id. Never rename one that has shipped.
    pub id: &'static str,
    pub label: &'static str,
    pub scope: Scope,
    pub default: Binding,
    pub mode: MatchMode,
    /// A FIXED synonym key (matched with the same rule, never remapped,
    /// shown greyed in the window). Delete/Backspace, D/Escape.
    pub also: Option<Key>,
    /// A FIXED second chord: the default key with these modifiers fires too,
    /// while the binding is the default. Select-all is ctrl+A AND ⌘A.
    pub also_mods: Option<Modifiers>,
}

const fn def(
    act: Act,
    id: &'static str,
    label: &'static str,
    scope: Scope,
    default: Binding,
    mode: MatchMode,
) -> Def {
    Def { act, id, label, scope, default, mode, also: None, also_mods: None }
}

const fn def_also(
    act: Act,
    id: &'static str,
    label: &'static str,
    scope: Scope,
    default: Binding,
    mode: MatchMode,
    also: Key,
) -> Def {
    Def { act, id, label, scope, default, mode, also: Some(also), also_mods: None }
}

const fn def_chords(
    act: Act,
    id: &'static str,
    label: &'static str,
    scope: Scope,
    default: Binding,
    also_mods: Modifiers,
) -> Def {
    Def { act, id, label, scope, default, mode: Exact, also: None, also_mods: Some(also_mods) }
}

use MatchMode::{Exact, KeyOnly};

pub const DEFS: &[Def] = &[
    def(Act::Save, "app.save", "save setup", Scope::App, Binding::cmd(Key::S), Exact),
    def(Act::SaveAs, "app.save_as", "save setup as…", Scope::App, Binding::cmd_shift(Key::S), Exact),
    def(Act::Open, "app.open", "open setup…", Scope::App, Binding::cmd(Key::O), Exact),
    def(Act::Undo, "app.undo", "undo", Scope::App, Binding::cmd(Key::Z), Exact),
    def(Act::Redo, "app.redo", "redo", Scope::App, Binding::cmd_shift(Key::Z), Exact),
    def(Act::Record, "app.record", "record a clip / stop", Scope::App, Binding::cmd_shift(Key::R), Exact),
    // Held over the time bar: what it spans (comp / node / in–out).
    def(
        Act::TimelinePie,
        "timeline.pie",
        "time bar pie: zoom to the comp, node or in–out range (over the time bar)",
        Scope::App,
        Binding::plain(Key::Backtick),
        Exact,
    ),
    // Over the time bar or a Time Remap bar: frame its in/out.
    def(
        Act::TimelineFrame,
        "timeline.frame",
        "frame the in/out (over the time bar or a Time Remap bar)",
        Scope::App,
        Binding::plain(Key::R),
        Exact,
    ),
    // Over the timeline, or while Tab-scrubbing the viewer: the playhead
    // becomes the playback in / out point. Elsewhere I and O are the
    // graph's swipes and roto's feather drags, so the site gates on that.
    def(
        Act::TimelineIn,
        "timeline.in",
        "set the playback in point at the playhead (over the timeline or while scrubbing)",
        Scope::App,
        Binding::plain(Key::I),
        Exact,
    ),
    def(
        Act::TimelineOut,
        "timeline.out",
        "set the playback out point at the playhead (over the timeline or while scrubbing)",
        Scope::App,
        Binding::plain(Key::O),
        Exact,
    ),
    def(
        Act::FullscreenPane,
        "app.fullscreen",
        "fullscreen the pane under the cursor",
        Scope::App,
        Binding::plain(Key::Q),
        Exact,
    ),
    def(
        Act::GraphOverlay,
        "app.overlay",
        "fullscreen + node overlay",
        Scope::App,
        Binding::cmd(Key::Q),
        Exact,
    ),
    def(
        Act::BackToLayout,
        "app.back_to_layout",
        "back to the panelled layout",
        Scope::App,
        Binding::plain(Key::Escape),
        Exact,
    ),
    def(
        Act::ToggleProps,
        "app.props",
        "show / hide the properties bar",
        Scope::App,
        Binding::plain(Key::W),
        Exact,
    ),
    // Every viewer HUD at once — the menubar's "hud off" — on shift of W,
    // the other chrome toggle (W hides the props bar), so the two hide keys
    // sit under one finger.
    def(
        Act::HudOff,
        "app.hud_off",
        "hide all viewer HUD (manipulators, sliders, chips)",
        Scope::App,
        Binding::shift(Key::W),
        Exact,
    ),
    def(
        Act::CurveEditor,
        "app.curves",
        "toggle the curve editor",
        Scope::App,
        Binding::plain(Key::Y),
        Exact,
    ),
    def(
        Act::Browse,
        "app.browse",
        "browse for media (bin to Loads)",
        Scope::App,
        Binding::plain(Key::L),
        Exact,
    ),
    // Matched on RELEASE (a tap must not be swallowed by a drag), and the
    // graph checks it as a HELD guard — both key-only.
    def(
        Act::PlayPause,
        "app.play",
        "play / pause",
        Scope::App,
        Binding::plain(Key::Space),
        KeyOnly,
    ),
    // Read with `pressed_or_repeat`: holding the key keeps stepping.
    def(
        Act::StepBack,
        "app.step_back",
        "step back one frame",
        Scope::App,
        Binding::plain(Key::ArrowLeft),
        Exact,
    ),
    def(
        Act::StepForward,
        "app.step_forward",
        "step forward one frame",
        Scope::App,
        Binding::plain(Key::ArrowRight),
        Exact,
    ),
    def(
        Act::ToggleAlpha,
        "viewer.alpha",
        "show the alpha channel",
        Scope::Viewer,
        Binding::plain(Key::A),
        Exact,
    ),
    // Shift, because plain R frames the view (and is the roto rotate
    // gesture while held); again returns to rgb, like A.
    def(
        Act::ViewRed,
        "viewer.red",
        "show the red channel",
        Scope::Viewer,
        Binding::shift(Key::R),
        Exact,
    ),
    def(
        Act::ViewGreen,
        "viewer.green",
        "show the green channel",
        Scope::Viewer,
        Binding::shift(Key::G),
        Exact,
    ),
    def(
        Act::ViewBlue,
        "viewer.blue",
        "show the blue channel",
        Scope::Viewer,
        Binding::shift(Key::B),
        Exact,
    ),
    def(
        Act::PixelReadout,
        "viewer.readout",
        "pixel readout under the pointer",
        Scope::Viewer,
        Binding::plain(Key::P),
        Exact,
    ),
    def(
        Act::PinSample,
        "viewer.pin_sample",
        "hold the current sample beside the live one",
        Scope::Viewer,
        Binding::shift(Key::P),
        Exact,
    ),
    def_also(
        Act::FrameView,
        "viewer.frame",
        "fit the image to the viewer",
        Scope::Viewer,
        Binding::plain(Key::R),
        Exact,
        Key::Home,
    ),
    // A Card or Blend 3D on the viewer is a 3D view: Blender's numpad
    // (egui reads the numpad and the top row as one key, which is also
    // Blender's "emulate numpad").
    def(
        Act::View3dFront,
        "viewer.3d_front",
        "front view (3D)",
        Scope::Viewer,
        Binding::plain(Key::Num1),
        Exact,
    ),
    def(
        Act::View3dBack,
        "viewer.3d_back",
        "back view (3D)",
        Scope::Viewer,
        Binding::ctrl(Key::Num1),
        Exact,
    ),
    def(
        Act::View3dRight,
        "viewer.3d_right",
        "right view (3D)",
        Scope::Viewer,
        Binding::plain(Key::Num3),
        Exact,
    ),
    def(
        Act::View3dLeft,
        "viewer.3d_left",
        "left view (3D)",
        Scope::Viewer,
        Binding::ctrl(Key::Num3),
        Exact,
    ),
    def(
        Act::View3dTop,
        "viewer.3d_top",
        "top view (3D)",
        Scope::Viewer,
        Binding::plain(Key::Num7),
        Exact,
    ),
    def(
        Act::View3dBottom,
        "viewer.3d_bottom",
        "bottom view (3D)",
        Scope::Viewer,
        Binding::ctrl(Key::Num7),
        Exact,
    ),
    def(
        Act::View3dOrbitLeft,
        "viewer.3d_orbit_left",
        "orbit 15° left (3D)",
        Scope::Viewer,
        Binding::plain(Key::Num4),
        Exact,
    ),
    def(
        Act::View3dOrbitRight,
        "viewer.3d_orbit_right",
        "orbit 15° right (3D)",
        Scope::Viewer,
        Binding::plain(Key::Num6),
        Exact,
    ),
    def(
        Act::View3dOrbitUp,
        "viewer.3d_orbit_up",
        "orbit 15° up (3D)",
        Scope::Viewer,
        Binding::plain(Key::Num8),
        Exact,
    ),
    def(
        Act::View3dOrbitDown,
        "viewer.3d_orbit_down",
        "orbit 15° down (3D)",
        Scope::Viewer,
        Binding::plain(Key::Num2),
        Exact,
    ),
    def(
        Act::View3dOpposite,
        "viewer.3d_opposite",
        "the view from the other side (3D)",
        Scope::Viewer,
        Binding::plain(Key::Num9),
        Exact,
    ),
    def(
        Act::VectorHud,
        "viewer.vectors",
        "vector HUD (a 2dini's geometry as data)",
        Scope::Viewer,
        Binding::plain(Key::H),
        Exact,
    ),
    def(
        Act::DiveIn,
        "graph.dive",
        "open the selected subwob / Noise / 2dini",
        Scope::Graph,
        Binding::plain(Key::Enter),
        Exact,
    ),
    def(
        Act::DiveOut,
        "graph.dive_out",
        "up one level (out of a subwob / Noise / 2dini)",
        Scope::Graph,
        Binding::plain(Key::U),
        Exact,
    ),
    // Any-mods on purpose: the Tab steal always fired whatever was held,
    // and a search popup opening on shift+Tab hurts nobody.
    def(
        Act::GraphSearch,
        "graph.search",
        "node search at the cursor",
        Scope::Graph,
        Binding::plain(Key::Tab),
        KeyOnly,
    ),
    def_also(
        Act::GraphDelete,
        "graph.delete",
        "delete the selection",
        Scope::Graph,
        Binding::plain(Key::Delete),
        Exact,
        Key::Backspace,
    ),
    def(
        Act::GraphLinkMode,
        "graph.link_mode",
        "expression-link mode",
        Scope::Graph,
        Binding::plain(Key::E),
        Exact,
    ),
    def(
        Act::GraphPatchMode,
        "graph.patch_mode",
        "node-patch mode (hidden wires)",
        Scope::Graph,
        Binding::shift(Key::Backtick),
        Exact,
    ),
    // Held like the source pies: release over a slice fires that op.
    def(
        Act::GraphOpsPie,
        "graph.ops_pie",
        "node-ops pie (unify, group, tidy, bypass…)",
        Scope::Graph,
        Binding::plain(Key::Backtick),
        Exact,
    ),
    // The properties twin of the ops pie: held over a row (or a panel),
    // release on a slice to apply it there.
    def(
        Act::PropsPie,
        "props.pie",
        "properties pie (lock keyframes…)",
        Scope::Props,
        Binding::plain(Key::Backtick),
        Exact,
    ),
    // Over a curve point editor (a LumaKey's bands, a colour curve):
    // frame the picked points, or all of them with none picked (Q).
    // The pane fullscreen stands down while a curve has the pointer.
    def(
        Act::PropsCurveFrame,
        "props.curve_frame",
        "frame the picked curve points (all when none) — over a curve editor",
        Scope::Props,
        Binding::plain(Key::Q),
        Exact,
    ),
    // The viewer's twin: held over the picture, release on a slice.
    def(
        Act::ViewerPie,
        "viewer.pie",
        "viewer pie (frame, 1:1, 1/2, channels)",
        Scope::Viewer,
        Binding::plain(Key::Backtick),
        Exact,
    ),
    // The usual compositor rename key (Blender's F2 is the global patch pie here, by
    // request). Double-clicking a node's NAME renames it too.
    // Blender's "toggle node preview": the selected Reads become little
    // CRTs showing their frame (see `crate::tv`).
    def(
        Act::GraphTv,
        "graph.tv",
        "Load thumbnails (toggle node preview)",
        Scope::Graph,
        Binding::shift(Key::H),
        Exact,
    ),
    // Held while dragging a MATTE across a node: isolate it by the matte
    // (a mask) or cut the matte out of it (a stencil). Same
    // swipe as M's blend; hold-only, so no tap to share the key with.
    def(
        Act::GraphSwipeIn,
        "graph.swipe_in",
        "hold + swipe a matte over a node: mask it (in)",
        Scope::Graph,
        Binding::plain(Key::I),
        Exact,
    ),
    def(
        Act::GraphSwipeOut,
        "graph.swipe_out",
        "hold + swipe a matte over a node: stencil it (out)",
        Scope::Graph,
        Binding::plain(Key::O),
        Exact,
    ),
    // B is M's blend swipe without M's tap: hold-only. B is also the
    // "drag rate x10" key, but the graph drags no values (the Z/X precedent).
    def(
        Act::GraphSwipeBlend,
        "graph.swipe_blend",
        "hold + swipe a dragged node over another: blend",
        Scope::Graph,
        Binding::plain(Key::B),
        Exact,
    ),
    def(
        Act::GraphRename,
        "graph.rename",
        "rename the selected node",
        Scope::Graph,
        Binding::plain(Key::N),
        Exact,
    ),
    def(
        Act::GraphNodePie,
        "graph.node_pie",
        "most-used nodes pie",
        Scope::Graph,
        Binding::plain(Key::F1),
        Exact,
    ),
    def(
        Act::GraphSourcePie,
        "graph.source_pie",
        "the global patch-source pie",
        Scope::Graph,
        Binding::plain(Key::F2),
        Exact,
    ),
    def(
        Act::GraphRegionPie,
        "graph.region_pie",
        "the region's patch-source pie",
        Scope::Graph,
        Binding::plain(Key::F3),
        Exact,
    ),
    def(
        Act::GraphRegionMenu,
        "graph.region_menu",
        "tap: selection to region popup · hold + swipe a dragged node over another: blend",
        Scope::Graph,
        Binding::plain(Key::M),
        Exact,
    ),
    def(
        Act::GraphBypass,
        "graph.bypass",
        "bypass the selection",
        Scope::Graph,
        Binding::plain(Key::D),
        Exact,
    ),
    def_also(
        Act::GraphFrame,
        "graph.frame",
        "tap: frame the selection (everything when empty) · hold + drag: rotate its layout",
        Scope::Graph,
        Binding::plain(Key::R),
        Exact,
        Key::Home,
    ),
    def(
        Act::GraphUnifySize,
        "graph.unify_size",
        "unify the selection's node size",
        Scope::Graph,
        Binding::shift(Key::U),
        Exact,
    ),
    // Shift (replace instead of add) and the tap-vs-click split are read
    // by the gesture.
    def(
        Act::GraphPinPanels,
        "graph.pin_panels",
        "pin properties panels (tap = selection, click = node)",
        Scope::Graph,
        Binding::plain(Key::Z),
        KeyOnly,
    ),
    def(
        Act::GraphPinGizmos,
        "graph.pin_gizmos",
        "pin viewer manipulators (tap = selection, click = node)",
        Scope::Graph,
        Binding::plain(Key::X),
        KeyOnly,
    ),
    def(
        Act::GraphGrab,
        "graph.grab",
        "pick a noodle up off the selected port / drop it",
        Scope::Graph,
        Binding::plain(Key::G),
        Exact,
    ),
    // Inside a 2dini: the noodle under the pointer down / up the stack (its
    // z), Illustrator's send backward / bring forward.
    def(
        Act::GraphWireZDown,
        "graph.wire_z_down",
        "2dini: the noodle under the pointer down the stack (z - 1)",
        Scope::Graph,
        Binding::plain(Key::OpenBracket),
        Exact,
    ),
    def(
        Act::GraphWireZUp,
        "graph.wire_z_up",
        "2dini: the noodle under the pointer up the stack (z + 1)",
        Scope::Graph,
        Binding::plain(Key::CloseBracket),
        Exact,
    ),
    def(
        Act::GraphSwapInputs,
        "graph.swap_inputs",
        "tap: swap a node's two named inputs (+/− …) · hold + drag: scale the selection's spacing",
        Scope::Graph,
        Binding::plain(Key::S),
        Exact,
    ),
    def(
        Act::GraphTidy,
        "graph.tidy",
        "tidy — lay the selection out as a clean flow",
        Scope::Graph,
        Binding::plain(Key::T),
        Exact,
    ),
    // R for range; plain R frames the graph.
    def(
        Act::GraphRanges,
        "graph.ranges",
        "range view: every noodle's first–last frame",
        Scope::Graph,
        Binding::shift(Key::R),
        Exact,
    ),
    // Held: the click (or drag) on the wire is the gesture.
    def(
        Act::GraphDot,
        "graph.dot",
        "hold + click a wire: routing dot (drag for tension)",
        Scope::Graph,
        Binding::plain(Key::Period),
        KeyOnly,
    ),
    def(
        Act::GraphGroup,
        "graph.group",
        "group the selection behind a backdrop",
        Scope::Graph,
        Binding::cmd(Key::G),
        Exact,
    ),
    def(
        Act::GraphUngroup,
        "graph.ungroup",
        "ungroup the selection's backdrops",
        Scope::Graph,
        Binding::cmd_shift(Key::G),
        Exact,
    ),
    def(
        Act::GraphSubwob,
        "graph.subwob",
        "collapse the selection into one subwob node",
        Scope::Graph,
        Binding::alt(Key::G),
        Exact,
    ),
    def(
        Act::GraphBakeSelected,
        "graph.bake_selected",
        "bake every selected Bake and Crop, one after another",
        Scope::Graph,
        Binding::alt(Key::B),
        Exact,
    ),
    def(
        Act::GraphCacheSelected,
        "graph.cache_selected",
        "cache every selected Load (all its files) at once",
        Scope::Graph,
        Binding::alt(Key::C),
        Exact,
    ),
    def(
        Act::GraphView1,
        "graph.view_a",
        "view in slot A (bare: flip to A)",
        Scope::Graph,
        Binding::plain(Key::Num1),
        Exact,
    ),
    def(
        Act::GraphView2,
        "graph.view_b",
        "view in slot B (bare: flip to B)",
        Scope::Graph,
        Binding::plain(Key::Num2),
        Exact,
    ),
    def(
        Act::GraphView3,
        "graph.view_c",
        "view in the second pane's slot A",
        Scope::Graph,
        Binding::plain(Key::Num3),
        Exact,
    ),
    def(
        Act::GraphView4,
        "graph.view_d",
        "view in the second pane's slot B",
        Scope::Graph,
        Binding::plain(Key::Num4),
        Exact,
    ),
    // alt, not shift: shift+1 arrives as "!" on most layouts.
    def(
        Act::GraphViewLock,
        "graph.view_lock",
        "lock viewer 1 on this node — it stays there inside subwobs (again: unlock)",
        Scope::Graph,
        Binding::alt(Key::Num1),
        Exact,
    ),
    def(
        Act::GraphSwapUp,
        "graph.swap_up",
        "swap the node with its parent",
        Scope::Graph,
        Binding::cmd(Key::ArrowUp),
        Exact,
    ),
    def(
        Act::GraphSwapDown,
        "graph.swap_down",
        "swap the node with its child",
        Scope::Graph,
        Binding::cmd(Key::ArrowDown),
        Exact,
    ),
    // Held while a drag starts; the stroke picks the axis. Read with
    // `down`, which ignores modifiers whatever the mode; Exact says A's
    // chords are free for other actions (select-all is ctrl+A).
    def(
        Act::GraphAlign,
        "graph.align",
        "align swipe (hold + drag)",
        Scope::Graph,
        Binding::plain(Key::A),
        Exact,
    ),
    def_chords(
        Act::GraphSelectAll,
        "graph.select_all",
        "select every node",
        Scope::Graph,
        Binding::ctrl(Key::A),
        Modifiers::COMMAND,
    ),
    // ⌘C / ⌘X / ⌘V are the system clipboard's own events and always work
    // (egui turns them into Copy / Cut / Paste before any key event exists,
    // so they are not bindings — see FIXED); these are the ctrl chords.
    def(
        Act::GraphCopy,
        "graph.copy",
        "copy the selected nodes",
        Scope::Graph,
        Binding::ctrl(Key::C),
        Exact,
    ),
    def(
        Act::GraphPaste,
        "graph.paste",
        "paste nodes at the pointer",
        Scope::Graph,
        Binding::ctrl(Key::V),
        Exact,
    ),
    def_also(
        Act::CurvesFrame,
        "curves.frame",
        "frame the visible curves",
        Scope::Curves,
        Binding::plain(Key::R),
        Exact,
        Key::Home,
    ),
    // Held like the graph's ops pie: release over a slice runs it.
    def(
        Act::CurvesPie,
        "curves.pie",
        "curve-ops pie (interpolation, snap, frame…)",
        Scope::Curves,
        Binding::plain(Key::Backtick),
        Exact,
    ),
    // Blender's snap key.
    def(
        Act::CurvesSnap,
        "curves.snap",
        "toggle frame snap (off: keys can sit between frames)",
        Scope::Curves,
        Binding::shift(Key::S),
        Exact,
    ),
    // HELD during a key / handle / box drag. Key-only so they combine with
    // shift. Z and X are the sensitivity row elsewhere; the curve editor
    // drags no slider values, so it may reuse them like the graph does.
    def(
        Act::CurvesOnlyTime,
        "curves.only_time",
        "hold while dragging: move in time only (x)",
        Scope::Curves,
        Binding::plain(Key::Z),
        KeyOnly,
    ),
    def(
        Act::CurvesOnlyValue,
        "curves.only_value",
        "hold while dragging: move in value only (y)",
        Scope::Curves,
        Binding::plain(Key::X),
        KeyOnly,
    ),
    def(
        Act::CurvesUnify,
        "curves.unify",
        "tangent: align the selected keys' handles (they swing together)",
        Scope::Curves,
        Binding::plain(Key::T),
        Exact,
    ),
    def(
        Act::CurvesBreak,
        "curves.break",
        "break tangent: the selected keys' handles move independently",
        Scope::Curves,
        Binding::shift(Key::T),
        Exact,
    ),
    // Blender's add-F-curve-modifier key.
    def(
        Act::CurvesNoise,
        "curves.noise",
        "noise modifier on / off for the curves with selected keys",
        Scope::Curves,
        Binding::ctrl_shift(Key::M),
        Exact,
    ),
    // Blender's modal transforms: the pointer drives, click / Enter
    // confirms, right-click / Esc puts it back. R is this editor's frame
    // key, so rotate takes shift.
    def(
        Act::CurvesGrab,
        "curves.grab",
        "grab the selected keys (modal: click confirms, Esc cancels)",
        Scope::Curves,
        Binding::plain(Key::G),
        Exact,
    ),
    def(
        Act::CurvesScale,
        "curves.scale",
        "scale the selected keys about the pivot (modal)",
        Scope::Curves,
        Binding::plain(Key::S),
        Exact,
    ),
    def(
        Act::CurvesRotate,
        "curves.rotate",
        "rotate the selected keys (pivot 'each': their handles) (modal)",
        Scope::Curves,
        Binding::shift(Key::R),
        Exact,
    ),
    // Blender's extrapolation key.
    def(
        Act::CurvesCycle,
        "curves.cycle",
        "cycle on / off: repeat the curves past their keys",
        Scope::Curves,
        Binding::shift(Key::E),
        Exact,
    ),
    // Blender's smooth-keys key.
    def(
        Act::CurvesSmooth,
        "curves.smooth",
        "smooth the selected keys (drag sets how much)",
        Scope::Curves,
        Binding::alt(Key::O),
        Exact,
    ),
    def(
        Act::CurvesMirror,
        "curves.mirror",
        "mirror the selected keys in time about the playhead",
        Scope::Curves,
        Binding::ctrl(Key::M),
        Exact,
    ),
    def(
        Act::CurvesSelectLeft,
        "curves.select_left",
        "select the keys at or before the playhead",
        Scope::Curves,
        Binding::plain(Key::OpenBracket),
        Exact,
    ),
    def(
        Act::CurvesSelectRight,
        "curves.select_right",
        "select the keys at or after the playhead",
        Scope::Curves,
        Binding::plain(Key::CloseBracket),
        Exact,
    ),
    def(
        Act::CurvesSelectColumn,
        "curves.select_column",
        "select every key on the selected keys' frames (none: the playhead's)",
        Scope::Curves,
        Binding::plain(Key::K),
        Exact,
    ),
    // Houdini's ladder: held while a drag starts on the keys.
    def(
        Act::CurvesLadder,
        "curves.ladder",
        "hold + drag: nudge values in steps (up / down picks the step)",
        Scope::Curves,
        Binding::plain(Key::D),
        KeyOnly,
    ),
    // ⌘C / ⌘V are the system clipboard's events and always work there.
    def(
        Act::CurvesCopy,
        "curves.copy",
        "copy the selected keys",
        Scope::Curves,
        Binding::ctrl(Key::C),
        Exact,
    ),
    def(
        Act::CurvesPaste,
        "curves.paste",
        "paste keys at the playhead",
        Scope::Curves,
        Binding::ctrl(Key::V),
        Exact,
    ),
    def(
        Act::CurvesFrameSelected,
        "curves.frame_selected",
        "frame the selected keys",
        Scope::Curves,
        Binding::plain(Key::F),
        Exact,
    ),
    // Shift = softness, alt = opacity — read by the site per event.
    def(
        Act::BrushDown,
        "paint.brush_down",
        "brush smaller (shift softness, alt opacity)",
        Scope::Paint,
        Binding::plain(Key::OpenBracket),
        KeyOnly,
    ),
    def(
        Act::BrushUp,
        "paint.brush_up",
        "brush larger (shift softness, alt opacity)",
        Scope::Paint,
        Binding::plain(Key::CloseBracket),
        KeyOnly,
    ),
    // Blender's F: hold it and MOVE — x sets size, y softness; with shift,
    // x opacity and y angle; with ctrl, x the ellipse's aspect. Read by the
    // site while held, modifiers off the live state.
    def(
        Act::BrushAdjust,
        "paint.brush_adjust",
        "hold + move: brush size/softness (shift opacity/angle, ctrl aspect)",
        Scope::Paint,
        Binding::plain(Key::F),
        KeyOnly,
    ),
];

pub fn def_of(a: Act) -> &'static Def {
    DEFS.iter().find(|d| d.act == a).expect("every Act has a DEFS row")
}

// ---------------------------------------------------------------------------
// Overrides (the only mutable state) + capture flag.

static OVERRIDES: RwLock<BTreeMap<&'static str, Binding>> = RwLock::new(BTreeMap::new());
/// True from the frame a capture is armed until the captured key is
/// RELEASED — the app block and the graph keys stand down for the whole
/// press, or capturing "Q" would also fullscreen the pane (and releasing a
/// captured Space would start playback).
static CAPTURING: AtomicBool = AtomicBool::new(false);

pub fn binding(a: Act) -> Binding {
    let d = def_of(a);
    OVERRIDES.read().unwrap().get(d.id).copied().unwrap_or(d.default)
}

/// Set a binding. Setting an action back to its default REMOVES the
/// override, so prefs stay a diff.
pub fn set(a: Act, b: Binding) {
    let d = def_of(a);
    let b = Binding { key: b.key, mods: norm(b.mods) };
    let mut o = OVERRIDES.write().unwrap();
    if b == d.default {
        o.remove(d.id);
    } else {
        o.insert(d.id, b);
    }
}

pub fn reset(a: Act) {
    OVERRIDES.write().unwrap().remove(def_of(a).id);
}

pub fn reset_all() {
    OVERRIDES.write().unwrap().clear();
}

pub fn is_overridden(a: Act) -> bool {
    OVERRIDES.read().unwrap().contains_key(def_of(a).id)
}

/// The overrides as prefs strings, ready to serialize.
pub fn overrides() -> BTreeMap<String, String> {
    OVERRIDES
        .read()
        .unwrap()
        .iter()
        .map(|(id, b)| (id.to_string(), format_binding(*b)))
        .collect()
}

/// Replace the overrides from prefs strings. An id this build does not
/// know, or a string that does not parse, is skipped with a warning — a
/// downgraded build must not wedge on a newer map.
pub fn install(map: &BTreeMap<String, String>) {
    let mut o = OVERRIDES.write().unwrap();
    o.clear();
    for (id, s) in map {
        let Some(d) = DEFS.iter().find(|d| d.id == *id) else {
            log::warn!("keymap: unknown action id {id:?} in prefs, skipped");
            continue;
        };
        match parse_binding(s) {
            Some(b) if b != d.default => {
                o.insert(d.id, b);
            }
            Some(_) => {} // stored default: treat as no override
            None => log::warn!("keymap: unparsable binding {s:?} for {id}, skipped"),
        }
    }
}

pub fn capturing() -> bool {
    CAPTURING.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Matching.

fn event_hits(b: Binding, mode: MatchMode, also: Option<Key>, key: Key, mods: Modifiers) -> bool {
    let key_ok = key == b.key || also == Some(key);
    key_ok
        && match mode {
            Exact => mods.matches_exact(b.mods),
            KeyOnly => true,
        }
}

/// The fixed second chord, while the binding is still the default (a
/// rebind means the user chose the key; the synonym does not follow it).
fn also_chord(a: Act) -> Option<Binding> {
    let d = def_of(a);
    if is_overridden(a) {
        return None;
    }
    d.also_mods.map(|mods| Binding { key: d.default.key, mods })
}

/// Every chord that fires `a`: its binding, and the fixed second chord.
fn chords(a: Act) -> Vec<Binding> {
    std::iter::once(binding(a)).chain(also_chord(a)).collect()
}

/// The action's key went DOWN this frame (repeats excluded; modifiers off
/// the event, so synthetic test input needs no global modifier state).
pub fn pressed(i: &egui::InputState, a: Act) -> bool {
    let d = def_of(a);
    let b = binding(a);
    let second = also_chord(a);
    i.events.iter().any(|e| {
        matches!(e, egui::Event::Key { key, pressed: true, repeat: false, modifiers, .. }
            if event_hits(b, d.mode, d.also, *key, *modifiers)
                || second.is_some_and(|c| *key == c.key && modifiers.matches_exact(c.mods)))
    })
}

/// `pressed`, plus the OS key repeat: for actions that keep going while
/// the key is held (frame stepping).
pub fn pressed_or_repeat(i: &egui::InputState, a: Act) -> bool {
    let d = def_of(a);
    let b = binding(a);
    let second = also_chord(a);
    i.events.iter().any(|e| {
        matches!(e, egui::Event::Key { key, pressed: true, modifiers, .. }
            if event_hits(b, d.mode, d.also, *key, *modifiers)
                || second.is_some_and(|c| *key == c.key && modifiers.matches_exact(c.mods)))
    })
}

/// The action's key came UP this frame. Key-only: the modifier state at
/// release is nobody's contract.
pub fn released(i: &egui::InputState, a: Act) -> bool {
    let d = def_of(a);
    let b = binding(a);
    i.events.iter().any(|e| {
        matches!(e, egui::Event::Key { key, pressed: false, .. }
            if *key == b.key || d.also == Some(*key))
    })
}

/// The action's key is HELD (key-only, for gesture guards).
pub fn down(i: &egui::InputState, a: Act) -> bool {
    let d = def_of(a);
    let b = binding(a);
    i.key_down(b.key) || d.also.is_some_and(|k| i.key_down(k))
}

/// Consume-flavoured press for the chord actions (save/open/undo): removes
/// the event so nothing downstream sees it, exactly like the
/// `consume_key` calls it replaces.
pub fn consume(i: &mut egui::InputState, a: Act) -> bool {
    let d = def_of(a);
    let b = binding(a);
    i.consume_key(b.mods, b.key)
        || d.also.is_some_and(|k| i.consume_key(b.mods, k))
        || also_chord(a).is_some_and(|c| i.consume_key(c.mods, c.key))
}

/// Tab is held: the viewer's scrub modifier (hold Tab, drag across the
/// picture). Tracked from RAW input because `take_search` strips Tab
/// before egui sees it, so `key_down(Tab)` is never true in a pass.
static SCRUB_KEY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Is the scrub modifier (Tab) held right now?
pub fn scrub_key_down() -> bool {
    SCRUB_KEY.load(std::sync::atomic::Ordering::Relaxed)
}

/// The Tab steal, shared by the main window and the detached-pane path.
/// Tab is two fixed things at once — the viewer's scrub modifier while
/// held, and the default node search on press — and neither survives
/// egui's focus traversal seeing it, so every Tab event is stripped here
/// after its held state is noted. Returns whether the SEARCH binding went
/// down (a letter binding types into a focused field like any other
/// single-letter key and needs no strip).
pub fn take_search(ctx: &egui::Context, raw: &mut egui::RawInput) -> bool {
    use std::sync::atomic::Ordering;
    // A number field is being typed into: Tab is the spreadsheet move to
    // the next one (`num_tab_pre`), not the search or the scrub.
    if num_tab_pre(ctx, raw) {
        return false;
    }
    for e in &raw.events {
        match e {
            egui::Event::Key { key: Key::Tab, pressed, .. } => {
                SCRUB_KEY.store(*pressed, Ordering::Relaxed);
            }
            // A release lost to another window must not leave Tab "held".
            // The TRANSITION, not `raw.focused`: a suite's window is an
            // inactive accessory and never focused at all.
            egui::Event::WindowFocused(false) => SCRUB_KEY.store(false, Ordering::Relaxed),
            _ => {}
        }
    }
    // A capture in the keys window outranks the steal: with Tab eaten at
    // the RAW level it could never be taught as a binding.
    if capturing() {
        return false;
    }
    let b = binding(Act::GraphSearch);
    let hit = raw.events.iter().any(|e| {
        matches!(e, egui::Event::Key { key, pressed: true, .. } if *key == b.key)
    });
    raw.events.retain(|e| !matches!(e, egui::Event::Key { key: Key::Tab, .. }));
    hit
}

// ---------------------------------------------------------------------------
// Tab between number fields.
//
// While a numeric field (any `DragValue`, which every slider's value box
// is) is being typed into, Tab moves to the NEXT numeric field and
// shift+Tab to the previous — a spreadsheet's hands. egui's own focus
// traversal does the moving (a DragValue that gains focus opens for
// typing), but it walks every focusable widget; so when the focus lands on
// a button or a checkbox this hops again, one widget per frame, until a
// number field has it. State lives in the egui context, so each detached
// pane tabs on its own.

/// Tabbing: (backwards, hops so far).
#[derive(Clone, Copy, Default)]
struct NumTab(Option<(bool, u8)>);

fn num_tab_id() -> egui::Id {
    egui::Id::new("keymap-num-tab")
}

/// Hops before giving up (a panel with no other number field).
const NUM_TAB_MAX_HOPS: u8 = 64;

/// Is this widget a number field being typed into? A `DragValue` in edit
/// mode draws a TextEdit under its OWN id, so the TextEdit's state there is
/// the public trace it leaves (egui 0.36 keeps the text itself in a private
/// type). Buttons and checkboxes leave none.
fn is_number_field(ctx: &egui::Context, id: egui::Id) -> bool {
    egui::text_edit::TextEditState::load(ctx, id).is_some()
}

/// Before the pass. Returns true when Tab belongs to number-field
/// traversal this frame, i.e. its events must reach egui.
fn num_tab_pre(ctx: &egui::Context, raw: &mut egui::RawInput) -> bool {
    let mut st = ctx.data(|d| d.get_temp::<NumTab>(num_tab_id())).unwrap_or_default();
    let focused = ctx.memory(|m| m.focused());
    let number = focused.is_some_and(|id| is_number_field(ctx, id));
    let tab_down = raw.events.iter().find_map(|e| match e {
        egui::Event::Key { key: Key::Tab, pressed: true, modifiers, .. }
            if !modifiers.any() || modifiers.shift_only() =>
        {
            Some(modifiers.shift)
        }
        _ => None,
    });
    let ours = match (st.0, focused) {
        // Mid-hop and landed on something that is not a number: move on.
        (Some((back, n)), Some(_)) if !number && n < NUM_TAB_MAX_HOPS => {
            raw.events.retain(|e| !matches!(e, egui::Event::Key { key: Key::Tab, .. }));
            let modifiers = if back { Modifiers::SHIFT } else { Modifiers::NONE };
            for pressed in [true, false] {
                raw.events.push(egui::Event::Key {
                    key: Key::Tab,
                    physical_key: None,
                    pressed,
                    repeat: false,
                    modifiers,
                });
            }
            st.0 = Some((back, n + 1));
            true
        }
        _ => {
            // Arrived, lost focus, or gave up.
            st.0 = None;
            match tab_down {
                Some(back) if number => {
                    st.0 = Some((back, 0));
                    true
                }
                _ => false,
            }
        }
    };
    ctx.data_mut(|d| d.insert_temp(num_tab_id(), st));
    ours
}

// ---------------------------------------------------------------------------
// Strings.

/// Prefs form: `cmd+shift+S`, `[`, `Space`, `cmd+Up`. `Key::name()` spells
/// the key; the round trip over every default is unit-tested.
pub fn format_binding(b: Binding) -> String {
    let mut s = String::new();
    if b.mods.command {
        s.push_str("cmd+");
    }
    if b.mods.ctrl {
        s.push_str("ctrl+");
    }
    if b.mods.shift {
        s.push_str("shift+");
    }
    if b.mods.alt {
        s.push_str("alt+");
    }
    s.push_str(b.key.name());
    s
}

pub fn parse_binding(s: &str) -> Option<Binding> {
    let mut mods = Modifiers::NONE;
    let mut key = None;
    for part in s.split('+') {
        let part = part.trim();
        match part.to_ascii_lowercase().as_str() {
            "cmd" | "command" | "super" => mods.command = true,
            "ctrl" | "control" => mods.ctrl = true,
            "shift" => mods.shift = true,
            "alt" | "option" => mods.alt = true,
            _ => key = Key::from_name(part),
        }
    }
    key.map(|key| Binding { key, mods: norm(mods) })
}

/// Display form for panels and the help window ("⌘⇧Z" on a Mac). The
/// egui context knows the platform.
pub fn label(ctx: &egui::Context, a: Act) -> String {
    let b = binding(a);
    ctx.format_shortcut(&egui::KeyboardShortcut { modifiers: b.mods, logical_key: b.key })
}

// ---------------------------------------------------------------------------
// Conflicts.

/// Pairs of actions in the SAME scope whose bindings collide. A key-only
/// action collides on the bare key (it would swallow every modifier
/// flavour); two exact actions collide only on the full chord.
pub fn conflicts() -> Vec<(Act, Act)> {
    let mut out = Vec::new();
    for (n, a) in DEFS.iter().enumerate() {
        for b in &DEFS[n + 1..] {
            if a.scope != b.scope {
                continue;
            }
            let loose = a.mode == KeyOnly || b.mode == KeyOnly;
            let clash = chords(a.act).iter().any(|ca| {
                chords(b.act)
                    .iter()
                    .any(|cb| ca.key == cb.key && (loose || ca.mods == cb.mods))
            });
            if clash {
                out.push((a.act, b.act));
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// The keyboard.

/// One keycap on the drawn keyboard: a real egui key, a modifier (which is
/// not a `Key` in egui — it can only be part of a chord), or a dead cap
/// (caps lock) that exists so the rows read as the keyboard on the desk.
struct Cap {
    key: Option<Key>,
    modifier: Option<ModKey>,
    label: &'static str,
    /// Width in key units (a letter is 1.0).
    w: f32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ModKey {
    Cmd,
    Shift,
    Alt,
    Ctrl,
}

fn kcap(key: Key, label: &'static str, w: f32) -> Cap {
    Cap { key: Some(key), modifier: None, label, w }
}
fn mcap(m: ModKey, label: &'static str, w: f32) -> Cap {
    Cap { key: None, modifier: Some(m), label, w }
}
fn dcap(label: &'static str, w: f32) -> Cap {
    Cap { key: None, modifier: None, label, w }
}

/// The ANSI rows, 15 units wide each, esc kept (it is the fixed back /
/// cancel synonym) and the arrow cluster flattened onto the bottom row.
/// Every default binding must be drawable here, via its key or its `also`
/// synonym — `every_default_key_is_on_the_keyboard` pins that, so a new
/// DEFS row cannot ship a key the visualization silently has no cap for.
fn keyboard_rows() -> Vec<Vec<Cap>> {
    use Key::*;
    use ModKey::*;
    vec![
        vec![
            kcap(Escape, "esc", 1.3),
            kcap(F1, "F1", 1.0),
            kcap(F2, "F2", 1.0),
            kcap(F3, "F3", 1.0),
        ],
        vec![
            kcap(Backtick, "`", 1.0),
            kcap(Num1, "1", 1.0),
            kcap(Num2, "2", 1.0),
            kcap(Num3, "3", 1.0),
            kcap(Num4, "4", 1.0),
            kcap(Num5, "5", 1.0),
            kcap(Num6, "6", 1.0),
            kcap(Num7, "7", 1.0),
            kcap(Num8, "8", 1.0),
            kcap(Num9, "9", 1.0),
            kcap(Num0, "0", 1.0),
            kcap(Minus, "-", 1.0),
            kcap(Equals, "=", 1.0),
            kcap(Backspace, "delete", 2.0),
        ],
        vec![
            kcap(Tab, "tab", 1.5),
            kcap(Q, "Q", 1.0),
            kcap(W, "W", 1.0),
            kcap(E, "E", 1.0),
            kcap(R, "R", 1.0),
            kcap(T, "T", 1.0),
            kcap(Y, "Y", 1.0),
            kcap(U, "U", 1.0),
            kcap(I, "I", 1.0),
            kcap(O, "O", 1.0),
            kcap(P, "P", 1.0),
            kcap(OpenBracket, "[", 1.0),
            kcap(CloseBracket, "]", 1.0),
            kcap(Backslash, "\\", 1.5),
        ],
        vec![
            dcap("caps", 1.75),
            kcap(A, "A", 1.0),
            kcap(S, "S", 1.0),
            kcap(D, "D", 1.0),
            kcap(F, "F", 1.0),
            kcap(G, "G", 1.0),
            kcap(H, "H", 1.0),
            kcap(J, "J", 1.0),
            kcap(K, "K", 1.0),
            kcap(L, "L", 1.0),
            kcap(Semicolon, ";", 1.0),
            kcap(Quote, "'", 1.0),
            kcap(Enter, "return", 2.25),
        ],
        vec![
            mcap(Shift, "shift", 2.25),
            kcap(Z, "Z", 1.0),
            kcap(X, "X", 1.0),
            kcap(C, "C", 1.0),
            kcap(V, "V", 1.0),
            kcap(B, "B", 1.0),
            kcap(N, "N", 1.0),
            kcap(M, "M", 1.0),
            kcap(Comma, ",", 1.0),
            kcap(Period, ".", 1.0),
            kcap(Slash, "/", 1.0),
            mcap(Shift, "shift", 2.75),
        ],
        vec![
            mcap(Ctrl, "ctrl", 1.25),
            mcap(Alt, "alt", 1.25),
            mcap(Cmd, "⌘", 1.5),
            kcap(Space, "", 5.5),
            mcap(Cmd, "⌘", 1.5),
            kcap(ArrowLeft, "⏴", 1.0),
            kcap(ArrowUp, "⏶", 1.0),
            kcap(ArrowDown, "⏷", 1.0),
            kcap(ArrowRight, "⏵", 1.0),
        ],
    ]
}

/// A word ("tab", "return") small; one glyph ("Q", "⌘", "⏶") larger.
/// Proportional for both: the monospace face has no backtick.
fn cap_font(label: &str) -> egui::FontId {
    if label.chars().count() > 1 {
        egui::FontId::proportional(9.5 * KB_SCALE)
    } else {
        egui::FontId::proportional(13.0 * KB_SCALE)
    }
}

fn mod_in(m: Modifiers, which: ModKey) -> bool {
    match which {
        ModKey::Cmd => m.command || m.mac_cmd,
        ModKey::Shift => m.shift,
        ModKey::Alt => m.alt,
        ModKey::Ctrl => m.ctrl,
    }
}

/// Commands that are keys or modifier gestures but deliberately NOT
/// remappable (the module docs say why). Listed read-only so the list and
/// the board tell the whole truth about a key; `how` stands where a
/// remappable row shows its binding. Wording follows the hotkeys window.
struct Fixed {
    how: &'static str,
    label: &'static str,
    key: Option<Key>,
    mods: Modifiers,
}

const CTRL_ALT: Modifiers =
    Modifiers { alt: true, ctrl: true, shift: false, mac_cmd: false, command: false };
const CTRL_SHIFT: Modifiers =
    Modifiers { alt: false, ctrl: true, shift: true, mac_cmd: false, command: false };

const FIXED: &[Fixed] = &[
    Fixed { how: "cmd+C", label: "graph: copy the selected nodes", key: Some(Key::C), mods: Modifiers::COMMAND },
    Fixed { how: "cmd+X", label: "graph: cut the selected nodes", key: Some(Key::X), mods: Modifiers::COMMAND },
    Fixed { how: "cmd+V", label: "graph: paste nodes at the pointer", key: Some(Key::V), mods: Modifiers::COMMAND },
    Fixed { how: "shift-hover a rail", label: "slider: edit its range", key: None, mods: Modifiers::SHIFT },
    Fixed { how: "ctrl+shift on a rail", label: "slider: pin the range editor open", key: None, mods: CTRL_SHIFT },
    Fixed { how: "hold G", label: "rgb slider: invert the gang", key: Some(Key::G), mods: Modifiers::NONE },
    Fixed { how: "hold S", label: "rgb slider: slide saturation", key: Some(Key::S), mods: Modifiers::NONE },
    Fixed { how: "hold D", label: "rgb slider: slide orange/cyan", key: Some(Key::D), mods: Modifiers::NONE },
    Fixed { how: "hold F", label: "rgb slider: slide green/magenta", key: Some(Key::F), mods: Modifiers::NONE },
    Fixed { how: "alt-drag", label: "pan, every canvas (or middle-drag)", key: None, mods: Modifiers::ALT },
    Fixed { how: "ctrl+alt-drag", label: "drag-zoom (or ctrl+middle-drag)", key: None, mods: CTRL_ALT },
    Fixed { how: "cmd+right-drag", label: "dolly zoom", key: None, mods: Modifiers::COMMAND },
    Fixed { how: "ctrl-drag", label: "graph: slice wires", key: None, mods: Modifiers::CTRL },
    Fixed { how: "shift", label: "graph: wire the courting pair", key: None, mods: Modifiers::SHIFT },
    Fixed { how: "hold tab + drag", label: "viewer: scrub the timeline", key: Some(Key::Tab), mods: Modifiers::NONE },
    Fixed { how: "alt-click", label: "paint: set the clone source", key: None, mods: Modifiers::ALT },
    Fixed { how: "ctrl-click", label: "viewer: pick the layer's transform, show its HUD", key: None, mods: Modifiers::CTRL },
    Fixed { how: "cmd-click", label: "viewer: pick the layer's transform, raise its properties", key: None, mods: Modifiers::COMMAND },
    Fixed { how: "shift+ctrl/cmd-click", label: "viewer: add the pick instead of soloing it", key: None, mods: Modifiers::SHIFT },
    Fixed { how: "hold D", label: "viewer pivot: drag moves the centre only", key: Some(Key::D), mods: Modifiers::NONE },
    Fixed { how: "hold E + drag", label: "roto pick: scale about the click", key: Some(Key::E), mods: Modifiers::NONE },
    Fixed { how: "hold R + drag", label: "roto pick: rotate by circling the click (R frames otherwise)", key: Some(Key::R), mods: Modifiers::NONE },
    Fixed { how: "hold O + drag point", label: "roto: drag a point to feather it outward", key: Some(Key::O), mods: Modifiers::NONE },
    Fixed { how: "hold I + drag point", label: "roto: drag a point to feather it inward", key: Some(Key::I), mods: Modifiers::NONE },
    Fixed { how: "hold T + drag", label: "roto pick: move the pick from anywhere", key: Some(Key::T), mods: Modifiers::NONE },
    Fixed { how: "G", label: "3D view: move manipulator", key: Some(Key::G), mods: Modifiers::NONE },
    Fixed { how: "R", label: "3D view: rotate manipulator (Home frames there)", key: Some(Key::R), mods: Modifiers::NONE },
    Fixed { how: "S", label: "3D view: scale manipulator", key: Some(Key::S), mods: Modifiers::NONE },
    // The drag-rate row. Held over any draggable value — slider, number
    // field, rgb lane, viewer gizmo — these scale the throw. Fixed for the
    // same reason as the colour axes: they are one instrument spread over
    // six adjacent keys, and rebinding one of them alone breaks the row.
    // `rgbslider::RATE_KEYS` is the table; keep these labels in step.
    Fixed { how: "hold Z", label: "drag rate ×0.01 (finest)", key: Some(Key::Z), mods: Modifiers::NONE },
    Fixed { how: "hold X", label: "drag rate ×0.1", key: Some(Key::X), mods: Modifiers::NONE },
    Fixed { how: "hold C", label: "drag rate ×1/3", key: Some(Key::C), mods: Modifiers::NONE },
    Fixed { how: "hold V", label: "drag rate ×3", key: Some(Key::V), mods: Modifiers::NONE },
    Fixed { how: "hold B", label: "drag rate ×10", key: Some(Key::B), mods: Modifiers::NONE },
    Fixed { how: "hold N", label: "drag rate ×30 (coarsest)", key: Some(Key::N), mods: Modifiers::NONE },
];

/// The dot and ring of a fixed command.
fn fixed_color() -> egui::Color32 {
    theme::text()
}

/// One command in the list: a remappable action or a fixed one. The list,
/// the lit caps, the inspector and click-to-select all go through this, so
/// they cannot disagree about what a key does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pick {
    Act(Act),
    Fixed(usize),
}

impl Pick {
    fn color(self) -> egui::Color32 {
        match self {
            Pick::Act(a) => def_of(a).scope.color(),
            Pick::Fixed(_) => fixed_color(),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Pick::Act(a) => def_of(a).label,
            Pick::Fixed(i) => FIXED[i].label,
        }
    }

    /// The chord as the list shows it — the LIVE binding for an action.
    fn chord(self, ctx: &egui::Context) -> String {
        match self {
            Pick::Act(a) => {
                let mut t = label(ctx, a);
                if let Some(k) = def_of(a).also {
                    t = format!("{t} / {}", k.name());
                }
                if let Some(c) = also_chord(a) {
                    let s = ctx.format_shortcut(&egui::KeyboardShortcut {
                        modifiers: c.mods,
                        logical_key: c.key,
                    });
                    t = format!("{t} / {s}");
                }
                t
            }
            Pick::Fixed(i) => FIXED[i].how.to_string(),
        }
    }

    /// Does this command's chord use `cap` — its key, its fixed synonym, or
    /// a modifier it holds?
    fn lights(self, cap: &Cap) -> bool {
        let (key, also, mods) = match self {
            Pick::Act(a) => {
                let b = binding(a);
                // Light the modifiers of the second chord as well: both are
                // how this command is reached.
                let mut mods = b.mods;
                if let Some(c) = also_chord(a) {
                    mods = mods | c.mods;
                }
                (Some(b.key), def_of(a).also, mods)
            }
            Pick::Fixed(i) => (FIXED[i].key, None, FIXED[i].mods),
        };
        match (cap.key, cap.modifier) {
            (Some(k), _) => key == Some(k) || also == Some(k),
            (None, Some(m)) => mod_in(mods, m),
            _ => false,
        }
    }

    /// The list filter. `q` arrives lowercased; see [`row_matches`].
    fn matches(self, q: &str) -> bool {
        match self {
            Pick::Act(a) => row_matches(def_of(a), q),
            Pick::Fixed(i) => {
                q.is_empty()
                    || FIXED[i].label.to_lowercase().contains(q)
                    || FIXED[i].how.to_lowercase().contains(q)
                    || "fixed".contains(q)
            }
        }
    }
}

/// Every command that uses `cap`: remappable ones in DEFS order, then the
/// fixed ones.
fn picks_on(cap: &Cap) -> Vec<Pick> {
    DEFS.iter()
        .map(|d| Pick::Act(d.act))
        .chain((0..FIXED.len()).map(Pick::Fixed))
        .filter(|p| p.lights(cap))
        .collect()
}

/// The list filter. Label, prefs id, pane name and the CURRENT key's
/// spelling all match, so "tidy", "graph" and "cmd+t" each find their
/// rows. `q` arrives lowercased.
fn row_matches(d: &Def, q: &str) -> bool {
    if q.is_empty() {
        return true;
    }
    d.label.to_lowercase().contains(q)
        || d.id.contains(q)
        || d.scope.title().contains(q)
        || format_binding(binding(d.act)).to_lowercase().contains(q)
}

/// The board's one size knob: caps, gaps, labels, dots, strokes and the
/// legend all scale with it. The list and the inspector do not.
const KB_SCALE: f32 = 2.0;
const KB_U: f32 = 30.0 * KB_SCALE;
const KB_GAP: f32 = 3.0 * KB_SCALE;
const KB_H: f32 = 28.0 * KB_SCALE;
const LIST_W: f32 = 300.0;
const ROW_H: f32 = 22.0;
const INSPECT_H: f32 = 132.0;

fn cap_w(units: f32) -> f32 {
    units * KB_U + (units - 1.0) * KB_GAP
}

fn board_size(rows: &[Vec<Cap>]) -> egui::Vec2 {
    egui::vec2(cap_w(15.0), rows.len() as f32 * (KB_H + KB_GAP) - KB_GAP)
}

/// Where cap (row, col) sits, relative to the board's top-left. A wide cap
/// absorbs the gaps it spans, so the columns line up across rows. Shared by
/// the painter and the pixel test, so the test looks where the paint went.
fn cap_rect(rows: &[Vec<Cap>], ri: usize, ci: usize) -> egui::Rect {
    let x: f32 = rows[ri][..ci].iter().map(|c| cap_w(c.w) + KB_GAP).sum();
    let y = ri as f32 * (KB_H + KB_GAP);
    egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(cap_w(rows[ri][ci].w), KB_H))
}

/// An OPAQUE blend. `gamma_multiply(..).to_opaque()` looks like a tint but
/// is not one: `to_opaque` un-premultiplies, which hands back the colour at
/// full strength.
fn mix(a: egui::Color32, b: egui::Color32, t: f32) -> egui::Color32 {
    let l = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    egui::Color32::from_rgb(l(a.r(), b.r()), l(a.g(), b.g()), l(a.b(), b.b()))
}

// ---------------------------------------------------------------------------
// The keys… window.

enum Capture {
    None,
    /// Waiting for a key to bind to this action.
    Armed(Act),
    /// Bound; waiting for that key to come back UP so its release cannot
    /// reach the app (releasing a captured Space must not start playback).
    WaitRelease(Key),
}

pub struct KeysUi {
    pub open: bool,
    capture: Capture,
    search: String,
    /// The command whose chord the board lights.
    selected: Option<Pick>,
    /// The list row under the pointer this frame; the board previews it.
    list_hover: Option<Pick>,
    /// The cap under the pointer this frame; the inspector describes it.
    cap_hover: Option<(usize, usize)>,
    /// Where the board was last drawn, for the pixel test.
    board: egui::Rect,
}

impl Default for KeysUi {
    fn default() -> Self {
        Self {
            open: false,
            capture: Capture::None,
            search: String::new(),
            selected: None,
            list_hover: None,
            cap_hover: None,
            board: egui::Rect::NOTHING,
        }
    }
}

impl KeysUi {
    /// Test hook + the rebind buttons' entry point.
    pub fn begin_capture(&mut self, a: Act) {
        self.capture = Capture::Armed(a);
        self.selected = Some(Pick::Act(a));
    }

    /// Draw the window (when open) and run the capture. Returns true when a
    /// binding changed — the caller mirrors [`overrides`] into prefs and
    /// saves.
    pub fn window(&mut self, ctx: &egui::Context, fullscreen: bool) -> bool {
        let mut changed = false;
        if !self.open || fullscreen {
            self.capture = Capture::None;
            CAPTURING.store(false, Ordering::Relaxed);
            return false;
        }
        // Capture BEFORE drawing, so the row shows its new key this frame.
        if let Capture::Armed(act) = self.capture {
            let hit = ctx.input(|i| {
                i.events.iter().find_map(|e| match e {
                    egui::Event::Key { key, pressed: true, repeat: false, modifiers, .. } => {
                        Some((*key, *modifiers))
                    }
                    _ => None,
                })
            });
            if let Some((key, mods)) = hit {
                if key != Key::Escape {
                    let mode = def_of(act).mode;
                    let mods = if mode == KeyOnly { Modifiers::NONE } else { mods };
                    set(act, Binding { key, mods });
                    changed = true;
                }
                self.capture = Capture::WaitRelease(key);
            }
        }
        // Not an else: a tap delivers press and release in ONE frame, and
        // holding the guard past a key that is already up would eat the
        // next real keystroke.
        if let Capture::WaitRelease(k) = self.capture {
            let up = ctx.input(|i| {
                !i.key_down(k)
                    || i.events.iter().any(|e| {
                        matches!(e, egui::Event::Key { key, pressed: false, .. } if *key == k)
                    })
            });
            if up {
                self.capture = Capture::None;
            }
        }

        let mut open = self.open;
        egui::Window::new("keys")
            .open(&mut open)
            .resizable(false)
            .show(ctx, |ui| changed |= self.body(ui));
        self.open = open;
        CAPTURING.store(!matches!(self.capture, Capture::None), Ordering::Relaxed);
        changed
    }

    /// Search on top; the command list left, the board and its inspector
    /// right. Split out of `window` so a test can rasterize it bare.
    fn body(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        self.list_hover = None;
        self.cap_hover = None;
        let conflicted = conflicts();
        let mut enter = false;
        ui.horizontal(|ui| {
            let r = ui.add(
                egui::TextEdit::singleline(&mut self.search)
                    .hint_text("search commands: name, pane or key…")
                    .desired_width(LIST_W - 28.0),
            );
            // Enter commits the search: the first row shown is selected.
            enter = r.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter));
            if !self.search.is_empty() && ui.small_button("×").on_hover_text("clear").clicked() {
                self.search.clear();
            }
            ui.add_space(12.0);
            if ui
                .small_button("reset all")
                .on_hover_text("every remappable key back to its default")
                .clicked()
            {
                reset_all();
                self.capture = Capture::None;
                changed = true;
            }
        });
        ui.separator();
        let right_h = board_size(&keyboard_rows()).y + ui.spacing().item_spacing.y + INSPECT_H;
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                ui.set_width(LIST_W);
                let (c, first) = self.command_list(ui, &conflicted, right_h);
                changed |= c;
                if enter && first.is_some() {
                    self.selected = first;
                }
            });
            ui.separator();
            ui.vertical(|ui| {
                changed |= self.keyboard(ui);
                changed |= self.inspector(ui, &conflicted);
            });
        });
        changed
    }

    /// The searchable list, grouped by pane with the fixed commands last.
    /// Returns (a binding changed, the first row shown).
    fn command_list(
        &mut self,
        ui: &mut egui::Ui,
        conflicted: &[(Act, Act)],
        height: f32,
    ) -> (bool, Option<Pick>) {
        let mut changed = false;
        let mut first = None;
        let q = self.search.trim().to_lowercase();
        let mut groups: Vec<(&str, egui::Color32, Vec<Pick>)> = Scope::ALL
            .iter()
            .map(|s| {
                let picks = DEFS.iter().filter(|d| d.scope == *s).map(|d| Pick::Act(d.act));
                (s.title(), s.color(), picks.collect())
            })
            .collect();
        groups.push((
            "fixed — not remappable",
            fixed_color(),
            (0..FIXED.len()).map(Pick::Fixed).collect(),
        ));
        egui::ScrollArea::vertical()
            .id_salt("keys-list")
            .max_height(height)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for (title, color, picks) in groups {
                    let shown: Vec<Pick> = picks.into_iter().filter(|p| p.matches(&q)).collect();
                    if shown.is_empty() {
                        continue;
                    }
                    first = first.or(shown.first().copied());
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        scope_dot(ui, color);
                        ui.strong(title);
                    });
                    for (n, p) in shown.into_iter().enumerate() {
                        changed |= self.row(ui, p, n % 2 == 1, conflicted);
                    }
                }
                if first.is_none() {
                    ui.add_space(8.0);
                    ui.weak("nothing matches");
                }
            });
        (changed, first)
    }

    fn row(&mut self, ui: &mut egui::Ui, p: Pick, stripe: bool, conflicted: &[(Act, Act)]) -> bool {
        let mut changed = false;
        // The row's own click is registered BEFORE its button, so the button
        // on top still takes its click (the `drag_row` rule in CLAUDE.md).
        let (rect, resp) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), ROW_H),
            egui::Sense::click(),
        );
        let sel = self.selected == Some(p);
        let hover = ui.rect_contains_pointer(rect);
        if hover {
            self.list_hover = Some(p);
        }
        if sel {
            ui.painter().rect_filled(rect, 3.0, p.color().gamma_multiply(0.22));
            let bar = egui::Rect::from_min_size(rect.min, egui::vec2(3.0, ROW_H));
            ui.painter().rect_filled(bar, 1.5, p.color());
        } else if hover {
            ui.painter().rect_filled(rect, 3.0, theme::hover_fill());
        } else if stripe {
            ui.painter().rect_filled(rect, 0.0, theme::raised_fill());
        }
        if resp.clicked() {
            self.selected = if sel { None } else { Some(p) };
        }
        let inner = egui::UiBuilder::new()
            .max_rect(rect.shrink2(egui::vec2(8.0, 0.0)))
            .layout(egui::Layout::right_to_left(egui::Align::Center));
        ui.scope_builder(inner, |ui| {
            // Right to left: reset, key, then the name fills what is left.
            match p {
                Pick::Act(a) => {
                    if is_overridden(a) {
                        if ui
                            .small_button("⟲")
                            .on_hover_text(format!("back to {}", format_binding(def_of(a).default)))
                            .clicked()
                        {
                            reset(a);
                            changed = true;
                        }
                    } else {
                        ui.add_space(18.0);
                    }
                    let armed = matches!(self.capture, Capture::Armed(x) if x == a);
                    let clash = conflicted.iter().find(|(x, y)| *x == a || *y == a);
                    let mut text = if armed {
                        egui::RichText::new("press a key…").italics()
                    } else {
                        egui::RichText::new(p.chord(ui.ctx())).strong()
                    };
                    if clash.is_some() && !armed {
                        text = text.color(theme::error());
                    }
                    let mut btn = ui.button(text);
                    btn = match clash {
                        Some((x, y)) => {
                            let other = if *x == a { *y } else { *x };
                            btn.on_hover_text(format!(
                                "collides with \"{}\" in the same pane",
                                def_of(other).label
                            ))
                        }
                        None => btn.on_hover_text("click, then press the new key (or click one on the board)"),
                    };
                    if btn.clicked() {
                        self.capture = if armed { Capture::None } else { Capture::Armed(a) };
                        self.selected = Some(p);
                    }
                }
                Pick::Fixed(_) => {
                    ui.add_space(18.0);
                    ui.weak(p.chord(ui.ctx()));
                }
            }
            ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                let mut name = egui::RichText::new(p.label());
                if sel {
                    name = name.strong();
                }
                ui.add(egui::Label::new(name).truncate().selectable(false));
            });
        });
        changed
    }

    /// The QWERTY board. The selected command's chord is lit in its pane's
    /// colour, and a hovered list row previews as an outline; the dots on a
    /// cap say which panes use it; hovering a cap describes it in the
    /// inspector below. Clicking a cap selects what lives there (again =
    /// the next command on a shared key) — or, while a row waits for a key,
    /// assigns that key with whatever modifiers are held.
    fn keyboard(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        let rows = keyboard_rows();
        let (board, _) = ui.allocate_exact_size(board_size(&rows), egui::Sense::hover());
        self.board = board;
        for (ri, row) in rows.iter().enumerate() {
            for (ci, cap) in row.iter().enumerate() {
                let r = cap_rect(&rows, ri, ci).translate(board.min.to_vec2());
                changed |= self.cap_ui(ui, r, cap, (ri, ci));
            }
        }
        let esc = cap_rect(&rows, 0, rows[0].len() - 1);
        legend(ui, board.min + egui::vec2(esc.max.x + 12.0 * KB_SCALE, 0.0), KB_H);
        changed
    }

    fn cap_ui(&mut self, ui: &mut egui::Ui, r: egui::Rect, cap: &Cap, at: (usize, usize)) -> bool {
        let dead = cap.key.is_none() && cap.modifier.is_none();
        let sense = if dead { egui::Sense::hover() } else { egui::Sense::click() };
        let resp = ui.interact(r, ui.id().with(("kbcap", at)), sense);
        let picks = picks_on(cap);
        if resp.hovered() {
            self.cap_hover = Some(at);
        }
        let lit = self.selected.filter(|p| p.lights(cap));
        let preview = self.list_hover.filter(|p| Some(*p) != self.selected && p.lights(cap));

        let base = if dead {
            theme::lift(theme::key_cap(), -22)
        } else if picks.is_empty() {
            theme::lift(theme::key_cap(), -13)
        } else {
            theme::key_cap()
        };
        let mut fill = base;
        if resp.hovered() && !dead {
            fill = mix(fill, egui::Color32::WHITE, 0.10);
        }
        if let Some(p) = lit {
            fill = mix(base, p.color(), 0.38);
        }
        let stroke = match (lit, preview) {
            (Some(p), _) => egui::Stroke::new(1.8 * KB_SCALE, p.color()),
            (None, Some(p)) => egui::Stroke::new(1.4 * KB_SCALE, mix(theme::border(), p.color(), 0.8)),
            _ => egui::Stroke::new(1.0 * KB_SCALE, theme::border()),
        };
        let text = if dead {
            theme::text_weak()
        } else if lit.is_some() {
            theme::text_hot()
        } else if picks.is_empty() {
            theme::text()
        } else {
            theme::text_strong()
        };
        let painter = ui.painter();
        painter.rect_filled(r, 4.0 * KB_SCALE, fill);
        painter.rect_stroke(r, 4.0 * KB_SCALE, stroke, egui::StrokeKind::Inside);
        painter.text(r.center(), egui::Align2::CENTER_CENTER, cap.label, cap_font(cap.label), text);
        // Dots along the top edge, right to left: which panes use this cap.
        let mut dots: Vec<egui::Color32> = Vec::new();
        for p in &picks {
            if !dots.contains(&p.color()) {
                dots.push(p.color());
            }
        }
        for (n, c) in dots.iter().enumerate() {
            let at = egui::pos2(r.max.x - (5.0 + n as f32 * 6.0) * KB_SCALE, r.min.y + 5.0 * KB_SCALE);
            painter.circle_filled(at, 2.2 * KB_SCALE, *c);
        }

        if !resp.clicked() {
            return false;
        }
        // A row waiting for a key takes the clicked one.
        if let (Capture::Armed(act), Some(key)) = (&self.capture, cap.key) {
            let act = *act;
            let mods = if def_of(act).mode == KeyOnly {
                Modifiers::NONE
            } else {
                ui.input(|i| i.modifiers)
            };
            set(act, Binding { key, mods });
            self.capture = Capture::None;
            return true;
        }
        if !picks.is_empty() {
            self.selected = Some(match self.selected.and_then(|s| picks.iter().position(|p| *p == s)) {
                Some(i) => picks[(i + 1) % picks.len()],
                None => picks[0],
            });
        }
        false
    }

    /// Under the board, and instant — tooltips here wait half a second,
    /// which is far too slow for sweeping across a keyboard. The hovered
    /// cap's every use, else the selected command, else how to drive this.
    fn inspector(&mut self, ui: &mut egui::Ui, conflicted: &[(Act, Act)]) -> bool {
        let mut changed = false;
        let rows = keyboard_rows();
        let (rect, _) = ui.allocate_exact_size(
            egui::vec2(board_size(&rows).x, INSPECT_H),
            egui::Sense::hover(),
        );
        ui.painter().rect_filled(rect, 4.0, theme::raised_fill());
        let inner = egui::UiBuilder::new().max_rect(rect.shrink(8.0));
        ui.scope_builder(inner, |ui| {
            ui.set_clip_rect(rect);
            if let Capture::Armed(a) = self.capture {
                ui.colored_label(
                    def_of(a).scope.color(),
                    format!("press a key for \"{}\" — or click one above", def_of(a).label),
                );
                ui.weak("esc cancels");
            } else if let Some((ri, ci)) = self.cap_hover {
                let cap = &rows[ri][ci];
                let title = match (cap.key, cap.modifier) {
                    (Some(k), _) => k.symbol_or_name().to_string(),
                    (_, Some(ModKey::Cmd)) => "cmd ⌘".into(),
                    (_, Some(ModKey::Shift)) => "shift".into(),
                    (_, Some(ModKey::Alt)) => "alt / option".into(),
                    (_, Some(ModKey::Ctrl)) => "ctrl".into(),
                    _ => cap.label.to_string(),
                };
                let picks = picks_on(cap);
                const SHOW: usize = 10;
                ui.horizontal(|ui| {
                    ui.strong(title);
                    if picks.len() > SHOW {
                        ui.weak(format!("+{} more", picks.len() - SHOW));
                    }
                });
                if picks.is_empty() {
                    ui.weak(if cap.key.is_none() && cap.modifier.is_none() {
                        "not a key the app reads"
                    } else {
                        "unassigned"
                    });
                } else if picks.len() <= 5 {
                    for p in &picks {
                        entry(ui, *p);
                    }
                } else {
                    ui.columns(2, |cols| {
                        for (n, p) in picks.iter().take(SHOW).enumerate() {
                            entry(&mut cols[n % 2], *p);
                        }
                    });
                }
            } else if let Some(p) = self.selected {
                entry(ui, p);
                match p {
                    Pick::Act(a) => {
                        ui.horizontal(|ui| {
                            if ui.button("rebind").clicked() {
                                self.capture = Capture::Armed(a);
                            }
                            if is_overridden(a)
                                && ui
                                    .button(format!("⟲ back to {}", format_binding(def_of(a).default)))
                                    .clicked()
                            {
                                reset(a);
                                changed = true;
                            }
                            ui.weak(def_of(a).scope.title());
                        });
                        if let Some((x, y)) = conflicted.iter().find(|(x, y)| *x == a || *y == a) {
                            let other = if *x == a { *y } else { *x };
                            ui.colored_label(
                                theme::error(),
                                format!("collides with \"{}\" in the same pane", def_of(other).label),
                            );
                        }
                    }
                    Pick::Fixed(_) => {
                        ui.weak("fixed: nav gestures and the rgb slider's held axes are not remappable");
                    }
                }
            } else {
                ui.weak("hover a key to see what it does");
                ui.weak("click a command, or a key, to light its chord");
                ui.weak("a row's key button rebinds it: press the new key, or click one");
            }
        });
        changed
    }
}

/// One command, one line: its pane dot, its chord, its name.
fn entry(ui: &mut egui::Ui, p: Pick) {
    ui.horizontal(|ui| {
        scope_dot(ui, p.color());
        ui.strong(p.chord(ui.ctx()));
        ui.add(egui::Label::new(p.label()).truncate().selectable(false));
    });
}

/// The colour key for the cap dots, beside esc.
fn legend(ui: &egui::Ui, at: egui::Pos2, h: f32) {
    let painter = ui.painter();
    let y = at.y + h * 0.5;
    let mut x = at.x;
    let names = Scope::ALL.iter().map(|s| (s.title(), s.color()));
    for (name, col) in names.chain(std::iter::once(("fixed", fixed_color()))) {
        painter.circle_filled(egui::pos2(x + 3.0 * KB_SCALE, y), 3.0 * KB_SCALE, col);
        let g = painter.text(
            egui::pos2(x + 10.0 * KB_SCALE, y),
            egui::Align2::LEFT_CENTER,
            name,
            egui::FontId::proportional(11.0 * KB_SCALE),
            theme::text(),
        );
        x = g.max.x + 12.0 * KB_SCALE;
    }
}

fn scope_dot(ui: &mut egui::Ui, col: egui::Color32) {
    let (r, _) = ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
    ui.painter().circle_filled(r.center(), 3.0, col);
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod num_tab_tests {
    use super::*;

    /// Runs one frame of: number, button, checkbox, number, number. Returns
    /// the ids of the three number fields and what has focus after.
    /// Focus starts from `request_focus` between frames — what a click does,
    /// and the case where egui reports no FocusGained at all.
    fn frame(ctx: &egui::Context, events: Vec<egui::Event>) -> ([egui::Id; 3], Option<egui::Id>) {
        let mut raw = egui::RawInput { events, ..Default::default() };
        let _ = take_search(ctx, &mut raw);
        let mut ids = [egui::Id::NULL; 3];
        let (mut a, mut b, mut c, mut on) = (1.0f32, 2.0f32, 3.0f32, false);
        let _ = ctx.run_ui(raw, |ui| {
            ids[0] = ui.add(egui::DragValue::new(&mut a).update_while_editing(false)).id;
            let _ = ui.button("not a number");
            ui.checkbox(&mut on, "nor this");
            ids[1] = ui.add(egui::DragValue::new(&mut b).update_while_editing(false)).id;
            ids[2] = ui.add(egui::DragValue::new(&mut c).update_while_editing(false)).id;
        });
        (ids, ctx.memory(|m| m.focused()))
    }

    fn tab(shift: bool) -> Vec<egui::Event> {
        let modifiers = if shift { Modifiers::SHIFT } else { Modifiers::NONE };
        [true, false]
            .map(|pressed| egui::Event::Key {
                key: Key::Tab,
                physical_key: None,
                pressed,
                repeat: false,
                modifiers,
            })
            .to_vec()
    }

    fn settle(ctx: &egui::Context) -> ([egui::Id; 3], Option<egui::Id>) {
        let mut last = frame(ctx, vec![]);
        for _ in 0..12 {
            last = frame(ctx, vec![]);
        }
        last
    }

    #[test]
    fn tab_in_a_number_goes_to_the_next_number() {
        let ctx = egui::Context::default();
        let (ids, _) = frame(&ctx, vec![]);
        ctx.memory_mut(|m| m.request_focus(ids[0]));
        let (_, f) = settle(&ctx);
        assert_eq!(f, Some(ids[0]), "the first number took focus");
        let _ = frame(&ctx, tab(false));
        let (_, f) = settle(&ctx);
        assert_eq!(f, Some(ids[1]), "Tab skipped the button and checkbox to the next number");
        let _ = frame(&ctx, tab(false));
        let (_, f) = settle(&ctx);
        assert_eq!(f, Some(ids[2]), "and on to the one after");
        let _ = frame(&ctx, tab(true));
        let (_, f) = settle(&ctx);
        assert_eq!(f, Some(ids[1]), "shift+Tab goes back");
    }

    #[test]
    fn with_no_number_focused_tab_is_still_stolen() {
        let ctx = egui::Context::default();
        let _ = frame(&ctx, vec![]);
        let mut raw = egui::RawInput { events: tab(false), ..Default::default() };
        assert!(take_search(&ctx, &mut raw), "Tab opens the search");
        assert!(raw.events.is_empty(), "and never reaches egui's focus traversal");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Prefs ids are the contract with every setup already on disk.
    #[test]
    fn ids_are_unique() {
        let mut ids: Vec<&str> = DEFS.iter().map(|d| d.id).collect();
        ids.sort_unstable();
        let n = ids.len();
        ids.dedup();
        assert_eq!(n, ids.len(), "duplicate action id in DEFS");
    }

    /// What `format_binding` writes, `parse_binding` must read back — for
    /// every default, so a new row cannot ship a key `Key::from_name`
    /// spells differently.
    #[test]
    fn every_default_round_trips() {
        for d in DEFS {
            let s = format_binding(d.default);
            let b = parse_binding(&s)
                .unwrap_or_else(|| panic!("{}: {s:?} did not parse", d.id));
            assert_eq!(b, d.default, "{}: {s:?} round-tripped wrong", d.id);
        }
    }

    #[test]
    fn parse_is_forgiving_and_bounded() {
        assert_eq!(
            parse_binding("cmd+shift+Z"),
            Some(Binding::cmd_shift(Key::Z))
        );
        assert_eq!(parse_binding("Cmd + T"), Some(Binding::cmd(Key::T)));
        // ctrl normalizes into command only when command is also set;
        // alone it stays the mac ctrl key.
        assert_eq!(
            parse_binding("ctrl+K").map(|b| (b.mods.ctrl, b.mods.command)),
            Some((true, false))
        );
        assert_eq!(parse_binding("cmd+"), None);
        assert_eq!(parse_binding("notakey"), None);
        assert_eq!(parse_binding(""), None);
    }

    /// Exact matching reads the EVENT's modifiers: P and shift+P are two
    /// actions, a chord's letter does not leak into the plain binding, and
    /// a key-only action takes every flavour.
    #[test]
    fn matching_reads_event_modifiers() {
        let press = |key, modifiers| egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers,
        };
        let mut i = egui::InputState::default();
        i.events = vec![press(Key::P, Modifiers::SHIFT)];
        assert!(!pressed(&i, Act::PixelReadout), "shift+P is not P");
        assert!(pressed(&i, Act::PinSample));
        i.events = vec![press(Key::G, Modifiers::COMMAND)];
        assert!(!pressed(&i, Act::GraphGrab), "cmd+G is the group chord");
        assert!(pressed(&i, Act::GraphGroup));
        i.events = vec![press(Key::Z, Modifiers::SHIFT)];
        assert!(pressed(&i, Act::GraphPinPanels), "key-only takes any mods");
        // The Delete row's fixed synonym.
        i.events = vec![press(Key::Backspace, Modifiers::NONE)];
        assert!(pressed(&i, Act::GraphDelete));
        // Repeats are not presses: a held D must not flicker bypass.
        i.events = vec![egui::Event::Key {
            key: Key::D,
            physical_key: None,
            pressed: true,
            repeat: true,
            modifiers: Modifiers::NONE,
        }];
        assert!(!pressed(&i, Act::GraphBypass));
    }

    /// One test owns every assertion that touches the PROCESS-WIDE override
    /// map, because `cargo test` runs tests in parallel and two tests
    /// racing `set`/`reset_all` would flake.
    #[test]
    fn overrides_are_diffs_and_round_trip() {
        reset_all();
        assert!(overrides().is_empty());
        assert_eq!(binding(Act::GraphTidy), Binding::plain(Key::T));

        // An override shows up; setting the default back clears it.
        set(Act::GraphTidy, Binding::plain(Key::Y));
        assert_eq!(binding(Act::GraphTidy), Binding::plain(Key::Y));
        assert_eq!(overrides().len(), 1);
        set(Act::GraphTidy, Binding::plain(Key::T));
        assert!(overrides().is_empty(), "a default is not an override");

        // Same-scope conflict is reported; cross-scope is the design.
        set(Act::GraphTidy, Binding::plain(Key::S));
        let c = conflicts();
        assert!(
            c.contains(&(Act::GraphSwapInputs, Act::GraphTidy))
                || c.contains(&(Act::GraphTidy, Act::GraphSwapInputs)),
            "T moved onto S: {c:?}"
        );
        reset_all();
        assert!(
            !conflicts()
                .iter()
                .any(|(a, b)| *a == Act::GraphTidy || *b == Act::GraphTidy),
            "defaults must not self-conflict"
        );
        // Viewer A (alpha) and graph A (align) coexist by design: the pane
        // under the pointer owns the key.
        assert!(conflicts().is_empty(), "defaults conflict: {:?}", conflicts());

        // install: prefs strings in, same map out; junk skipped.
        let mut m = std::collections::BTreeMap::new();
        m.insert("graph.tidy".to_string(), "cmd+T".to_string());
        m.insert("no.such.action".to_string(), "Q".to_string());
        m.insert("graph.grab".to_string(), "not a key".to_string());
        install(&m);
        assert_eq!(binding(Act::GraphTidy), Binding::cmd(Key::T));
        assert_eq!(binding(Act::GraphGrab), Binding::plain(Key::G), "junk skipped");
        let out = overrides();
        assert_eq!(out.len(), 1);
        assert_eq!(out.get("graph.tidy").map(String::as_str), Some("cmd+T"));
        install(&BTreeMap::new());
        assert!(overrides().is_empty());
    }

    /// Every default binding must be drawable on the keys window's board —
    /// by its own key or its fixed synonym — or a new DEFS row would be
    /// invisible to the keyboard visualization.
    #[test]
    fn every_default_key_is_on_the_keyboard() {
        let caps: Vec<Key> =
            keyboard_rows().iter().flatten().filter_map(|c| c.key).collect();
        for d in DEFS {
            assert!(
                caps.contains(&d.default.key)
                    || d.also.is_some_and(|k| caps.contains(&k)),
                "{}: {} has no keycap",
                d.id,
                d.default.key.name()
            );
        }
    }

    /// No missing-glyph boxes in the keys window. egui's default fonts
    /// carry far fewer symbols than a keyboard legend reaches for — no
    /// \u{232b}, \u{23ce}, \u{21e7}, \u{2192} — and a missing one draws as tofu
    /// rather than failing. Every non-ASCII character in a string literal of
    /// this file must exist in the proportional family, and every keycap
    /// label in the face `cap_font` actually draws it with.
    /// Characters egui's built-in faces draw as an empty box (checked by
    /// rendering them: `has_glyph` says no to some that DO draw, through
    /// the emoji fallback, so it cannot be the judge across the app).
    /// ← ↑ → ↓ ↵ ⇄ ⇧ ⇱ ⏎ ▦ ▲ ▼ ◆ ◌ ● ✓
    const BOXES: &str = "\u{2190}\u{2191}\u{2192}\u{2193}\u{21b5}\u{21c4}\u{21e7}\u{21f1}\u{23ce}\u{25a6}\u{25b2}\u{25bc}\u{25c6}\u{25cc}\u{25cf}\u{2713}";

    #[test]
    fn no_ui_string_uses_a_character_that_draws_as_a_box() {
        let mut bad = Vec::new();
        for d in ["src", "src/nodes", "src/recorder"] {
            let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(d);
            for e in std::fs::read_dir(dir).unwrap() {
                let p = e.unwrap().path();
                if p.extension().is_none_or(|x| x != "rs") || p.ends_with("keymap.rs") {
                    continue;
                }
                let src = std::fs::read_to_string(&p).unwrap();
                for (ln, line) in src.lines().enumerate() {
                    // Terminal output and test messages are not drawn.
                    if ["println", "TEST", "log::", "assert", "fail(", "panic"].iter().any(|s| line.contains(s)) {
                        continue;
                    }
                    let (mut in_str, mut esc, mut prev) = (false, false, '\0');
                    for c in line.chars() {
                        if in_str {
                            if esc {
                                esc = false;
                            } else if c == '\\' {
                                esc = true;
                            } else if c == '"' {
                                in_str = false;
                            } else if BOXES.contains(c) {
                                bad.push(format!("{}:{} {c}", p.display(), ln + 1));
                            }
                        } else if c == '/' && prev == '/' {
                            break;
                        } else if c == '"' {
                            in_str = true;
                        }
                        prev = c;
                    }
                }
            }
        }
        assert!(bad.is_empty(), "these draw as an empty box:\n{}", bad.join("\n"));
    }

    #[test]
    fn every_drawn_character_has_a_glyph() {
        let ctx = egui::Context::default();
        let _ = ctx.run_ui(Default::default(), |_| {});
        let has = |font: &egui::FontId, c: char| ctx.fonts_mut(|f| f.has_glyph(font, c));

        // String literals only: comments are free to say what they like.
        let src = include_str!("keymap.rs");
        let (mut in_str, mut in_comment, mut escaped) = (false, false, false);
        let mut prev = '\0';
        let prop = egui::FontId::proportional(12.0);
        let mut missing = Vec::new();
        for c in src.chars() {
            if in_comment {
                in_comment = c != '\n';
            } else if in_str {
                if escaped {
                    escaped = false;
                } else if c == '\\' {
                    escaped = true;
                } else if c == '\u{22}' {
                    in_str = false;
                } else if !c.is_ascii() && !has(&prop, c) {
                    missing.push(c);
                }
            } else if c == '/' && prev == '/' {
                in_comment = true;
            } else if c == '\u{22}' {
                in_str = true;
            }
            prev = c;
        }
        missing.dedup();
        assert!(missing.is_empty(), "no glyph in the proportional font for {missing:?}");

        for cap in keyboard_rows().iter().flatten() {
            let font = cap_font(cap.label);
            for c in cap.label.chars() {
                assert!(has(&font, c), "keycap {:?}: no glyph for {c:?} in {font:?}", cap.label);
            }
        }
    }

    /// The rows stay a rectangle — 15 units each — or the columns shear
    /// the first time someone edits the layout. (Row 0 is esc alone.)
    #[test]
    fn keyboard_rows_line_up() {
        for (n, row) in keyboard_rows().iter().enumerate().skip(1) {
            let w: f32 = row.iter().map(|c| c.w).sum();
            assert!((w - 15.0).abs() < 1e-3, "row {n} is {w} units wide");
        }
    }

    /// The list filter finds a row by name, pane and key spelling. Uses
    /// actions the override-mutating test never touches, so the defaults
    /// this reads through `binding()` hold under parallel tests.
    #[test]
    fn search_matches_name_pane_and_key() {
        assert!(row_matches(def_of(Act::Save), "save"));
        assert!(row_matches(def_of(Act::Save), "cmd+s"));
        assert!(row_matches(def_of(Act::FrameView), "viewer"));
        assert!(row_matches(def_of(Act::FrameView), ""));
        assert!(!row_matches(def_of(Act::FrameView), "zzz"));
        // Fixed commands search too, by name, by chord and as a group.
        let sat = (0..FIXED.len()).find(|i| FIXED[*i].label.contains("saturation")).unwrap();
        assert!(Pick::Fixed(sat).matches("saturation"));
        assert!(Pick::Fixed(sat).matches("hold s"));
        assert!(Pick::Fixed(sat).matches("fixed"));
        assert!(!Pick::Fixed(sat).matches("tidy"));
    }

    /// A fixed command's key must be a cap, or selecting it lights nothing.
    #[test]
    fn fixed_keys_are_on_the_keyboard() {
        let rows = keyboard_rows();
        for f in FIXED {
            if let Some(k) = f.key {
                assert!(
                    rows.iter().flatten().any(|c| c.key == Some(k)),
                    "{}: {} has no keycap",
                    f.label,
                    k.name()
                );
            }
        }
    }

    /// The keys window as the app draws it — text and all — rendered through
    /// egui-wgpu on a headless device into `target/keys-window-gpu.png`.
    /// A picture to LOOK at, not an assertion (the snapshot test below makes
    /// those); ignored so CI never needs a GPU for it:
    /// `cargo test -p app keys_window_render -- --ignored`
    #[test]
    #[ignore]
    fn keys_window_render() {
        let mut keys = KeysUi::default();
        keys.selected = Some(Pick::Act(Act::Undo));
        let rows = keyboard_rows();
        let r_cap = rows
            .iter()
            .enumerate()
            .find_map(|(ri, row)| {
                row.iter().position(|c| c.key == Some(Key::R)).map(|ci| (ri, ci))
            })
            .unwrap();
        crate::uishot::render("keys-window-gpu", 1360.0, 600.0, 2.0, 4, |ui, frame| {
            // Once the board has a position, park the pointer on R so the
            // inspector shows a shared key.
            if frame > 0 {
                let at = cap_rect(&rows, r_cap.0, r_cap.1)
                    .translate(keys.board.min.to_vec2())
                    .center();
                ui.ctx().input_mut(|i| i.events.push(egui::Event::PointerMoved(at)));
            }
            egui::Frame::window(ui.style()).show(ui, |ui| {
                keys.body(ui);
            });
        });
    }

    /// The board lights what the list selects — read off the pixels of the
    /// window's body, rasterized headlessly. Selecting undo (cmd+Z) lights
    /// the Z cap AND both cmd caps in the app pane's blue; a key it does not
    /// use stays neutral. Undo is never rebound by the override test, so the
    /// binding read here is the default even with tests in parallel.
    #[test]
    fn selecting_a_command_lights_its_chord() {
        let rows = keyboard_rows();
        let caps = |want: &dyn Fn(&Cap) -> bool| -> Vec<egui::Rect> {
            let mut out = Vec::new();
            for (ri, row) in rows.iter().enumerate() {
                for (ci, c) in row.iter().enumerate() {
                    if want(c) {
                        out.push(cap_rect(&rows, ri, ci));
                    }
                }
            }
            out
        };
        let mut keys = KeysUi::default();
        keys.selected = Some(Pick::Act(Act::Undo));
        let shot = crate::snapshot::shoot(1360.0, 600.0, |ui| {
            keys.body(ui);
        });
        shot.save("keys-window");
        assert!(keys.board.is_positive(), "the board was never laid out");
        let origin = keys.board.min.to_vec2();
        // Below centre and left of it, in proportion to the cap so it stays
        // clear of the stroke and the dots at any KB_SCALE (the label is
        // skipped by the rasterizer anyway).
        let sample = |r: egui::Rect| {
            let r = r.translate(origin);
            shot.at(r.min.x + r.width() * 0.25, r.center().y + r.height() * 0.2)
        };
        let blue = |px: [u8; 3]| px[2] as i32 - px[0] as i32 > 25;
        let neutral = |px: [u8; 3]| (px[2] as i32 - px[0] as i32).abs() <= 3;

        let z = caps(&|c| c.key == Some(Key::Z));
        assert_eq!(z.len(), 1);
        assert!(blue(sample(z[0])), "Z cap not lit: {:?}", sample(z[0]));
        let cmd = caps(&|c| c.modifier == Some(ModKey::Cmd));
        assert_eq!(cmd.len(), 2, "both cmd caps");
        for r in cmd {
            assert!(blue(sample(r)), "cmd cap not lit: {:?}", sample(r));
        }
        for r in caps(&|c| c.key == Some(Key::J) || c.modifier == Some(ModKey::Shift)) {
            assert!(neutral(sample(r)), "a cap undo does not use is lit: {:?}", sample(r));
        }
    }
}
