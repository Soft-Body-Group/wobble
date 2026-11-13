//! The UI colour theme: every colour the interface paints that is chrome
//! rather than data, in ONE table a user can recolour.
//!
//! [`DEFS`] declares each slot — a stable id (what prefs store), the group
//! and label the colours… window lists it under, and a default. A default
//! is either a literal or a LINK to another slot, so recolouring a role
//! carries the colours that belong to it: move `background` and the windows,
//! fields and canvases move with it; move `accent` and the highlights built
//! on it follow. Links come in two strengths:
//!
//! * `like(S::x)` — the same colour as `x`, whatever that is.
//! * `shade(S::x, c)` — its own colour `c`, shifted per channel by however
//!   far `x` has been moved from ITS default. A canvas a little darker than
//!   the panels stays a little darker than the panels in any theme, and with
//!   nothing overridden every slot is exactly its literal.
//!
//! Sites read LIVE through the generated accessors (`theme::wire()`): one
//! atomic load, so an edit repaints the same frame with nothing to
//! invalidate. The statics are const-initialised to the defaults, so code
//! that paints before [`install`] (or a unit test that never calls it) sees
//! the real colours. Prefs hold only the OVERRIDES, as hex (`"wire":
//! "#b88a40"`) — the keymap's rule: an untouched slot follows any future
//! default, and an id this build does not know is skipped with a warning.
//!
//! egui's own widgets follow through [`sync`], which writes the interface
//! slots into a context's dark `Visuals` whenever the theme has moved.
//!
//! Deliberately NOT here — colours that ARE data, where a theme would make
//! the UI lie: r/g/b channel lanes and their tints, swatches and gradients
//! showing a value, plugin-declared node colours, cryptomatte ids, region and
//! backdrop colours a user picked. `rgbslider` and `colorpicker` stay
//! egui-only and take their chrome from `ui.visuals()`, which follows.
//!
//! Adding UI colour? Add a slot here and call it; do not paint a literal.

use egui::Color32;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::RwLock;

/// How a slot gets its colour when nothing overrides it.
#[derive(Clone, Copy, Debug)]
pub enum Base {
    /// A fixed colour.
    Rgb(Color32),
    /// Exactly another slot's colour.
    Like(Slot),
    /// Its own colour, shifted by however far the linked slot has moved from
    /// its default.
    Shade(Slot, Color32),
}

impl Base {
    pub const fn parent(&self) -> Option<Slot> {
        match *self {
            Base::Rgb(_) => None,
            Base::Like(p) | Base::Shade(p, _) => Some(p),
        }
    }
}

pub struct Def {
    /// The prefs key. Renaming one orphans every stored override of it.
    pub id: &'static str,
    pub group: &'static str,
    pub label: &'static str,
    /// The slot's doc comment, shown on hover.
    pub tip: &'static str,
    pub base: Base,
    /// The picker offers an alpha lane. Always true for a translucent
    /// default; `[alpha]` in the table grants it to an opaque one whose
    /// sites fade it themselves (`gamma_multiply_u8`), so a theme's alpha
    /// scales every one of them.
    pub alpha: bool,
}

const fn rgb(r: u8, g: u8, b: u8) -> Base {
    Base::Rgb(Color32::from_rgb(r, g, b))
}
const fn gray(l: u8) -> Base {
    Base::Rgb(Color32::from_gray(l))
}
/// Unmultiplied alpha, like a hex code.
const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Base {
    Base::Rgb(Color32::from_rgba_unmultiplied_const(r, g, b, a))
}
const fn like(s: Slot) -> Base {
    Base::Like(s)
}
/// A literal's colour, for the table's alpha flag (`like` answers opaque;
/// the flag is read off the slot's OWN literal).
const fn default_rgb(b: Base) -> Color32 {
    match b {
        Base::Rgb(c) | Base::Shade(_, c) => c,
        Base::Like(_) => Color32::BLACK,
    }
}
const fn shade(s: Slot, own: Base) -> Base {
    match own {
        Base::Rgb(c) => Base::Shade(s, c),
        _ => panic!("shade() takes a literal colour"),
    }
}

/// Declares the slots: the [`Slot`] enum, [`DEFS`], the live storage and one
/// accessor per slot, named after it.
macro_rules! slots {
    (@alpha $base:expr) => { !default_rgb($base).is_opaque() };
    (@alpha $base:expr, alpha) => { true };
    ($( $group:literal {
        $( $(#[doc = $doc:literal])* $id:ident : $label:literal $([$flag:ident])? = $base:expr ; )*
    } )*) => {
        /// One themable colour. The variant name is its prefs id.
        #[allow(non_camel_case_types)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum Slot { $($( $id, )*)* }

        pub const DEFS: &[Def] = &[
            $($( Def {
                id: stringify!($id),
                group: $group,
                label: $label,
                tip: concat!($( $doc, "\n", )* ""),
                base: $base,
                alpha: slots!(@alpha $base $(, $flag)?),
            }, )*)*
        ];

        pub const ALL: [Slot; N] = [ $($( Slot::$id, )*)* ];

        static LIVE: [AtomicU32; N] = [
            $($( AtomicU32::new(pack(default_of(Slot::$id))), )*)*
        ];

        $($(
            $(#[doc = $doc])*
            #[inline]
            // Some slots only ever reach egui's visuals, never a call site.
            #[allow(dead_code)]
            pub fn $id() -> Color32 {
                get(Slot::$id)
            }
        )*)*
    };
}

use Slot as S;

slots! {
    "interface" {
        /// Panels, and egui's frame fill behind non-interactive widgets.
        bg: "background" = gray(27);
        /// Floating windows and popups.
        window: "windows" = shade(S::bg, gray(27));
        /// Text fields, drag values and other sunk wells.
        field: "fields" = shade(S::bg, gray(10));
        /// Separators, window outlines, indentation lines.
        border: "borders" = gray(60);
        /// Outline of a hovered box.
        border_hover: "borders, hovered" = shade(S::border, gray(0x6a));
        /// Labels — every hover/active text shade egui keeps follows it.
        text: "text" = gray(140);
        /// Hints and placeholders.
        text_weak: "text, faint" = shade(S::text, gray(90));
        /// Labels painted on canvases and HUDs that must read clearly.
        text_strong: "text, bright" = shade(S::text, gray(0xd8));
        /// A hovered or active painted label.
        text_hot: "text, highlighted" = shade(S::text, gray(0xff));
        /// Button, checkbox and slider-handle fill.
        button: "buttons" = gray(60);
        button_hover: "buttons, hovered" = shade(S::button, gray(70));
        button_active: "buttons, pressed" = shade(S::button, gray(55));
        /// Selected text, selectable labels, toggled buttons.
        accent: "accent" = rgb(0, 92, 128);
        /// Text and outlines on the accent; the text cursor.
        accent_text: "accent text" = rgb(192, 222, 255);
        link: "links" = shade(S::accent_text, rgb(90, 170, 255));
        /// Outline of a selected thing on any canvas: nodes, clips, keys.
        selection: "selection" = gray(0xff);
        /// An on or pinned state that is not a selection: the hud toggle,
        /// X-pinned handles, the tool you are inside.
        highlight: "highlight" = rgb(0xd8, 0xa0, 0x30);
        /// A hovered row, tile or keycap, laid over whatever it sits on.
        hover_fill: "hover fill" = rgba(0xff, 0xff, 0xff, 12);
        /// Faintly raised: idle cards, striped rows, inset boxes.
        raised_fill: "raised fill" = shade(S::hover_fill, rgba(0xff, 0xff, 0xff, 5));
        /// The plate behind floating readouts, chips, pies and scrims. Each
        /// site keeps its own opacity; this colour's alpha scales them all.
        hud_bg: "HUD backdrop" [alpha] = gray(0);
        /// Drop shadow under floating boxes.
        shadow: "shadows" = shade(S::hud_bg, rgba(0, 0, 0, 96));
    }
    "status" {
        ok: "ok" = rgb(0x6c, 0xc2, 0x8a);
        /// Neutral state worth a colour (a movie that plays direct).
        info: "info" = rgb(110, 150, 200);
        warn: "warning" = rgb(0xff, 0xc4, 0x4d);
        error: "error" = rgb(0xe0, 0x60, 0x50);
        /// The slicer's trail, and the strike through a bypassed node.
        cut: "cut" = shade(S::error, rgb(0xe0, 0x50, 0x50));
    }
    "canvases" {
        /// The dark well behind a plot, a curve editor, a poster.
        well: "wells" = shade(S::field, gray(0x12));
        grid_line: "grid lines" = shade(S::well, gray(0x22));
        /// Zero, neutral and identity lines.
        axis_line: "axis lines" = shade(S::well, gray(0x3a));
        /// Pucks, knobs, points and box handles.
        handle: "handles" = gray(0xd4);
        handle_hot: "handles, hovered" = shade(S::handle, gray(0xff));
        /// The dark rim that keeps a handle readable on anything.
        handle_rim: "handle rims" = shade(S::hud_bg, gray(0x14));
        marquee: "box select" = shade(S::handle, gray(0x9a));
        marquee_fill: "box select fill" = shade(S::hover_fill, rgba(0xff, 0xff, 0xff, 6));
        /// Outline of a scalable selection and the rotate stalk.
        selection_box: "selection box" = shade(S::border, gray(0x55));
        /// A parameter curve with no channel colour of its own.
        curve_line: "curves" = like(S::handle);
        /// Scalar channels in the animation curve editor.
        curve_scalar: "animation curves" = shade(S::highlight, rgb(0xc8, 0xb0, 0x60));
        curve_selected: "selected points" = like(S::warn);
    }
    "node graph" {
        graph_bg: "background" = shade(S::bg, gray(0x14));
        graph_grid: "grid dots" = shade(S::graph_bg, gray(0x22));
        /// A wire with no cost colouring to wear, and the dots that bend one.
        wire: "wires" = rgb(0xb8, 0x8a, 0x40);
        /// White, like the matte input sockets they land in.
        wire_mask: "mask wires" = gray(0xff);
        /// The fat multi-channel cable off a Read.
        wire_channels: "multi-channel cables" = rgb(0x3d, 0x8f, 0x8f);
        /// The DATA pipe: one channel at full depth (a depth pass, a
        /// disparity). Violet, so it cannot be mistaken for the matte
        /// pipe's white at a glance — the two are the same shape and
        /// opposite contracts.
        wire_data: "data wires" = rgb(0x9a, 0x86, 0xdc);
        /// The TEXT pipe: words off a String node. Rose, clear of the
        /// picture pipes and of every status colour.
        wire_text: "text wires" = rgb(0xe0, 0x8c, 0xb8);
        wire_vector: "vector wires" = rgb(0xe6, 0xd4, 0x4a);
        /// A MODULATE matte (Blur, Convolve, Transform…): the matte pipe,
        /// but it SCALES the effect rather than mixing it, so its wire and
        /// socket wear yellow rather than the mask's white.
        wire_modulate: "modulate wires + sockets" = like(S::wire_vector);
        /// The SCENE pipe: cameras and cards, read by a Render3D. Green:
        /// clear of the matte's white, the vectors' yellow and the cyan
        /// every fast image wire wears (its old light blue read as one).
        wire_scene: "3D wires" = rgb(0x4c, 0xd8, 0x6a);
        /// Inside a 2dini, a noodle wears what it CARRIES: shapes,
        /// points (a scatter, before it is instanced), bodies (tagged for
        /// physics), a plain number, or a field (a number that varies over
        /// the canvas: a noise, or maths on one).
        wire_geo: "2dini shape wires" = rgb(0xe0, 0x7a, 0xa0);
        wire_points: "2dini point wires" = rgb(0x6c, 0xd0, 0x9c);
        wire_bodies: "2dini body wires" = rgb(0x5a, 0x9c, 0xf0);
        wire_number: "2dini number wires" = gray(0xc8);
        wire_field: "2dini field wires" = rgb(0xa8, 0x7c, 0xf0);
        /// Onion skins in a 2dini: frames before the playhead, and after.
        onion_before: "onion skin, before" = rgb(0xe0, 0x50, 0x70);
        onion_after: "onion skin, after" = rgb(0x60, 0xd0, 0x90);
        /// Ordinary wires stepped back while node-patch mode edits the hidden layer.
        wire_locked: "wires, locked" = shade(S::graph_bg, gray(0x3c));
        wire_dragged: "wire being dragged" = shade(S::selection, gray(0xcc));
        /// A noodle picked up with G, its cursor bead and port ring.
        wire_held: "wire held" = shade(S::highlight, rgb(0xf0, 0xd0, 0x60));
        /// Image sockets, inputs and outputs alike: the cup says which end.
        port_in: "sockets" = rgb(0x55, 0xb8, 0xd8);
        /// A node's matte input socket: white, like the matte wires, so it
        /// stands apart from the image inputs.
        port_mask: "matte sockets" = gray(0xff);
        /// Viewer A's halo, gizmo pins and channel-output nodules. No longer
        /// an output SOCKET: those wear `port_in` like the inputs.
        port_out: "viewer halo + pins" = rgb(0xd8, 0x3a, 0xae);
        /// Expression cables, channel nodules and patch-bay inputs — one
        /// amber family across the app.
        cable: "expression cables" = like(S::highlight);
        /// Hidden wires (node-patch mode) and the badge on hidden-fed nodes.
        hidden_wire: "hidden wires" = like(S::cable);
        /// A value SOURCE: patch-bay outputs and a tool's own knob feeding a
        /// channel. Deliberately not the cable amber.
        cable_source: "expression sources" = shade(S::ok, rgb(0x78, 0xbe, 0x96));
        node_label: "node labels" = shade(S::text, gray(0xb8));
        /// Node outlines and badge rims.
        node_outline: "node outlines" = shade(S::hud_bg, gray(0x0c));
        /// The dark glyph drawn on a node's body.
        node_ink: "node glyphs" = shade(S::node_outline, gray(0x18));
        /// A glyph that has to read light on the body (the Switch lever).
        node_ink_light: "node glyphs, light" = shade(S::selection, gray(0xe8));
        /// The light a node scaled down to a speck twinkles with, so it
        /// stays findable when its shape is a couple of pixels across.
        node_twinkle: "tiny-node twinkle" = shade(S::selection, rgb(0xf2, 0xf8, 0xff));
        /// The align swipe and the axis the nodes will snap to.
        align_guide: "align guide" = shade(S::link, rgb(0x60, 0xc0, 0xf0));
        /// The pip on nodes whose properties are pinned.
        pinned: "pinned pip" = shade(S::selection, gray(0xf0));
    }
    "cost + licence" {
        /// Everything upstream fuses. Deliberately the input cyan's cousin.
        cost_fast: "fast" = shade(S::port_in, rgb(0x46, 0xc8, 0xd8));
        /// A pass of its own somewhere upstream.
        cost_slow: "slow" = shade(S::error, rgb(0xe0, 0x4a, 0x3c));
        /// Non-commercial weights upstream: a restriction, not a cost.
        cost_nc: "non-commercial" = shade(S::wire_data, rgb(0xc0, 0x78, 0xe8));
        /// A panel saying, in words, what a node costs.
        cost_note: "cost notes" = shade(S::highlight, rgb(0xd0, 0x92, 0x3a));
        /// A node that bills when you press run (a GenAI pipeline): money,
        /// not frame time — its own hue, never the slow red.
        cost_money: "costs money" = rgb(0x5c, 0xc0, 0x6a);
    }
    "timeline" {
        /// The playhead on the transport track, the cache meter and scrub strips.
        playhead: "playhead" = like(S::selection);
        /// The playhead in the edit, layer and curve timelines.
        timeline_playhead: "playhead, timelines" = shade(S::error, rgb(0xd8, 0x5a, 0x4a));
        keyframe: "keyframes" = like(S::timeline_playhead);
        /// A channel that is animated, with no key on this frame.
        animated: "animated" = like(S::highlight);
        /// A channel LOCKED against keyframing: the bar under its field, its
        /// key dot, its curve in the editor.
        locked: "locked channel" = shade(S::border, rgb(0x6a, 0x74, 0x84));
        row_locked: "locked row wash" = shade(S::error, rgba(0xe0, 0x40, 0x40, 22));
        /// The stretch of the transport track that plays.
        range_in: "play range" = rgb(0x3c, 0x4a, 0x5a);
        /// A Load's own in/out: on the track while a Load is viewed, and
        /// its panel's bar. Muted green, so it never reads as the
        /// player's in/out.
        range_load: "Load in/out" = rgb(0x36, 0x5c, 0x3e);
        /// The playhead's frame-number block on the time bar.
        playhead_box: "playhead frame box" = like(S::ok);
        /// Tracks and strips outside the play range.
        range_out: "track" = shade(S::bg, gray(0x2e));
        cache_ram: "cached in RAM" = like(S::ok);
        cache_nvme: "cached on NVMe" = shade(S::info, rgb(0x4a, 0x74, 0xb8));
    }
    "edit timeline" {
        /// The source viewer: its tag, the card being viewed, in/out marks.
        timeline_source: "source viewer" = shade(S::highlight, rgb(255, 140, 60));
        /// Live edit feedback: drop targets and ghosts, trim readouts, a
        /// newer version on disk.
        timeline_active: "drops + readouts" = shade(S::warn, rgb(255, 215, 0));
        timeline_move: "clip being moved" = shade(S::link, rgb(173, 216, 230));
        timeline_ruler: "ruler" = shade(S::bg, gray(36));
        /// Media past a clip's in/out and its edge grips. Sites fade it.
        timeline_handles: "clip handles" [alpha] = shade(S::selection, gray(255));
        timeline_clip_1: "clips 1" = rgb(70, 100, 140);
        timeline_clip_2: "clips 2" = shade(S::timeline_clip_1, rgb(140, 90, 70));
        timeline_clip_3: "clips 3" = shade(S::timeline_clip_1, rgb(80, 130, 90));
        timeline_clip_4: "clips 4" = shade(S::timeline_clip_1, rgb(120, 80, 130));
        timeline_clip_5: "clips 5" = shade(S::timeline_clip_1, rgb(140, 120, 60));
        timeline_clip_6: "clips 6" = shade(S::timeline_clip_1, rgb(70, 125, 130));
    }
    "patch bay" {
        patch_mult: "Mult" = rgb(0x4f, 0x8f, 0xa8);
        patch_add: "Add" = shade(S::patch_mult, rgb(0x6f, 0xa8, 0x5f));
        patch_max: "Max" = shade(S::patch_mult, rgb(0xa8, 0x8f, 0x4f));
        patch_min: "Min" = shade(S::patch_mult, rgb(0x8a, 0x5f, 0xa8));
        patch_expr: "Expr" = shade(S::patch_mult, rgb(0xc9, 0x7a, 0x3a));
    }
    "viewer" {
        /// Around the image. Deliberately not linked to the interface
        /// background: this is where colour gets judged.
        viewer_bg: "backdrop" = gray(0);
        /// Outline of the comp's frame over the image.
        viewer_frame: "comp frame" = shade(S::border, gray(0x6a));
        /// The label of the pane a dual view is driving.
        viewer_live: "live pane" = shade(S::timeline_source, rgb(255, 170, 70));
        /// The 3D view's floor grid (a Card or Blend 3D on the viewer).
        view3d_grid: "3D floor grid" = gray(0x6e);
        /// The 3D view's axes, on the floor and in the gizmo: Blender's.
        view3d_x: "3D X axis" = rgb(232, 62, 82);
        view3d_y: "3D Y axis" = rgb(130, 200, 30);
        view3d_z: "3D Z axis" = rgb(46, 132, 232);
        /// A card's outline in the 3D view, and the picked card's.
        view3d_card: "3D card outline" = gray(0x9a);
        view3d_card_picked: "3D card outline, picked" = like(S::selection);
        view3d_camera: "3D camera frustum" = rgb(0x8c, 0xc8, 0xe8);
        view3d_points: "3D solved points" = rgb(0x6e, 0xd8, 0x9a);
    }
    "keyer" {
        keyer_blob: "key blobs" = shade(S::handle, gray(0xe8));
        keyer_subtract: "key blobs, subtract" = like(S::timeline_playhead);
    }
    "grade wheels" {
        wheel_well: "well" = shade(S::field, rgb(0x17, 0x18, 0x1a));
        wheel_disc: "disc" = shade(S::bg, rgb(0x1c, 0x1d, 0x20));
        /// The ring's bevel edges and the dial's outer edge.
        wheel_bevel: "bevel" = shade(S::wheel_well, rgb(0x0e, 0x0e, 0x10));
    }
    "browser" {
        browser_accent: "accent" = shade(S::accent, rgb(0x4c, 0x8b, 0xf5));
        browser_selected: "selected tile" = shade(S::browser_accent, rgb(0x2b, 0x3a, 0x53));
    }
    "layers" {
        layer_bar: "layer bars" = shade(S::locked, rgb(0x6a, 0x74, 0x88));
        layer_bar_selected: "layer bar, selected" = shade(S::highlight, rgb(0xc8, 0xa8, 0x5a));
        /// The M switch on a muted layer's row.
        layer_mute: "mute switch" = shade(S::error, rgb(0xd8, 0x50, 0x48));
        /// The S switch on a soloed layer's row.
        layer_solo: "solo switch" = shade(S::wire_mask, rgb(0xb0, 0xe0, 0x40));
    }
    "keys" {
        /// A keycap with a binding; unbound and dead caps sit darker.
        key_cap: "keycaps" = shade(S::bg, gray(46));
        keys_app: "app scope" = shade(S::link, rgb(108, 168, 255));
        keys_viewer: "viewer scope" = shade(S::ok, rgb(118, 208, 158));
        keys_graph: "graph scope" = shade(S::highlight, rgb(255, 188, 92));
        keys_curves: "curves scope" = shade(S::cost_nc, rgb(198, 142, 255));
        keys_paint: "paint scope" = shade(S::error, rgb(255, 128, 148));
        keys_props: "properties scope" = shade(S::keys_graph, rgb(120, 214, 176));
    }
    "node families" {
        /// Moving a family recolours every node type in it; each type keeps
        /// its own offset from the family colour.
        family_source: "sources" = rgb(0xa8, 0x2e, 0x7e);
        family_colour: "colour" = rgb(0x8a, 0x4f, 0x9e);
        family_filter: "filters" = rgb(0x3f, 0x8a, 0x7a);
        family_transform: "transforms" = rgb(0x3a, 0x6e, 0xa5);
        family_matte: "mattes + keys" = rgb(0x5e, 0x8a, 0x5e);
        family_channels: "channels" = rgb(0x3f, 0x9e, 0x6a);
        family_merge: "merges" = gray(0x77);
        family_time: "time" = rgb(0x6a, 0x76, 0x8c);
        family_track: "tracking" = rgb(0xd0, 0x4a, 0x4a);
        family_other: "output + other" = rgb(0xd0, 0x6a, 0x2e);
    }
    "nodes" {
        /// Production is plain text, so it follows the text colour.
        maturity_production: "production tag" = shade(S::text, gray(0xe8));
        maturity_beta: "beta tag" = shade(S::wire_mask, rgb(0xb0, 0xe0, 0x40));
        maturity_alpha: "alpha tag" = shade(S::error, rgb(0xe0, 0x50, 0x50));
        node_geo: "2dini + geometry" = shade(S::family_source, rgb(0xc4, 0x6f, 0x8a));
        node_bake: "Bake" = shade(S::family_time, rgb(0x8f, 0x8f, 0x4f));
        node_blend: "Blend" = shade(S::family_merge, gray(0x77));
        node_blender: "Blender" = shade(S::family_source, rgb(0x4f, 0x8f, 0xd6));
        node_genai: "GenAI" = shade(S::family_source, rgb(0xb4, 0x5a, 0xd2));
        node_prompt: "String" = like(S::wire_text);
        node_blur: "Blur" = shade(S::family_filter, rgb(0x3f, 0x8a, 0x7a));
        node_breakout: "Breakout" = shade(S::family_channels, rgb(0x46, 0x9a, 0x96));
        node_card3d: "Card" = like(S::wire_scene);
        node_camera3d: "Camera" = shade(S::family_source, rgb(0x5a, 0x7c, 0xc8));
        node_scene3d: "Blend 3D" = like(S::wire_scene);
        node_render3d: "Render3D" = shade(S::family_source, rgb(0x4a, 0x66, 0xb8));
        node_env: "Env" = shade(S::family_source, rgb(0x5a, 0x9a, 0xb4));
        node_channel_select: "ChannelSelect" = shade(S::family_channels, rgb(0x3d, 0x8f, 0x8f));
        node_color_compress: "ColorCompress" = shade(S::family_colour, rgb(0x6a, 0x5f, 0xc0));
        node_color_warp: "ColorWarp" = shade(S::family_colour, rgb(0xc0, 0x5a, 0x72));
        node_convolve: "Convolve" = shade(S::family_filter, rgb(0x86, 0x6c, 0x38));
        node_corner_pin: "CornerPin" = shade(S::family_transform, rgb(0x9a, 0x3a, 0x4a));
        node_stabilize: "Stabilize" = shade(S::family_transform, rgb(0x8a, 0x4e, 0x6e));
        node_camera_track: "CameraTrack" = shade(S::family_transform, rgb(0x5e, 0x6e, 0x9a));
        node_cryptomatte: "Cryptomatte" = shade(S::family_matte, rgb(0xc9, 0x7b, 0x3a));
        node_curve: "Curve" = shade(S::family_colour, rgb(0xb8, 0x9c, 0x2e));
        node_defocus: "Defocus" = shade(S::family_filter, rgb(0x2f, 0x6f, 0x9a));
        node_dilate: "Dilate" = shade(S::family_filter, rgb(0x4a, 0x8f, 0x6a));
        node_edge_extend: "EdgeExtend" = shade(S::family_filter, rgb(0x5a, 0x9a, 0x5e));
        node_displace: "Displace" = shade(S::family_transform, rgb(0x46, 0x93, 0xa8));
        node_expo_blur: "ExpoBlur" = shade(S::family_filter, rgb(0x3f, 0x7a, 0x8a));
        node_extract: "Extract" = shade(S::family_matte, rgb(0x5e, 0x8a, 0x72));
        node_fbm: "fBm" = shade(S::family_source, rgb(0x7e, 0x4f, 0xa8));
        node_flare: "Flare" = shade(S::family_filter, rgb(0xc9, 0x9a, 0x33));
        node_geo_load: "GeoLoad" = shade(S::family_source, rgb(0x5f, 0x4a, 0xc0));
        node_glint: "Glint" = shade(S::family_filter, rgb(0xb0, 0x92, 0x46));
        node_median: "Median" = shade(S::family_filter, rgb(0x6e, 0x86, 0x5a));
        node_highlight_rebuild: "HighlightRebuild" = shade(S::family_filter, rgb(0xb8, 0xa8, 0x5c));
        node_edge_detect: "EdgeDetect" = shade(S::family_filter, rgb(0x5c, 0x9c, 0xa8));
        node_diffuse: "Diffuse" = shade(S::family_filter, rgb(0x6f, 0x86, 0x96));
        node_glow: "Glow" = shade(S::family_filter, rgb(0x9a, 0x7f, 0x2f));
        node_godrays: "Godrays" = shade(S::family_filter, rgb(0xb4, 0xa0, 0x52));
        node_lens_flare: "Lens Flare" = shade(S::family_filter, rgb(0x9c, 0xb0, 0xc8));
        node_grade: "Grade" = shade(S::family_colour, rgb(0x8a, 0x4f, 0x9e));
        node_grid: "Grid" = shade(S::family_source, rgb(0x8e, 0x3a, 0x6e));
        node_gradient: "Gradient" = shade(S::family_source, rgb(0x9a, 0x44, 0x7e));
        node_blank: "Blank" = shade(S::family_source, rgb(0x7a, 0x7a, 0x86));
        node_starfield: "Starfield" = shade(S::family_source, rgb(0x5a, 0x6c, 0xb8));
        node_planet: "Planet" = shade(S::family_source, rgb(0x4e, 0x8a, 0x9c));
        node_plasma: "Plasma" = shade(S::family_source, rgb(0xc8, 0x5a, 0x6e));
        node_particles: "Particles" = shade(S::family_source, rgb(0xd0, 0x7a, 0x8c));
        node_glint_kernel: "Glint Kernel" = shade(S::family_source, rgb(0xa8, 0x9a, 0x6a));
        node_bokeh_kernel: "Bokeh Kernel" = shade(S::family_source, rgb(0x86, 0x7a, 0xa8));
        node_grain: "Grain" = shade(S::family_source, rgb(0x8c, 0x80, 0x92));
        node_hue_rotate: "HueRotate" = shade(S::family_colour, rgb(0x6c, 0x5c, 0xc4));
        node_lut: "LUT" = shade(S::family_colour, rgb(0x5a, 0x6e, 0xc0));
        node_text: "Text" = shade(S::family_source, rgb(0xb0, 0x5a, 0x9c));
        node_burnin: "Burnin" = shade(S::family_source, rgb(0x9a, 0x6a, 0xa8));
        node_mirror: "Mirror" = shade(S::family_transform, rgb(0x3e, 0x8e, 0x9e));
        node_lens_distort: "Lens Distortion" = shade(S::family_transform, rgb(0x4a, 0x86, 0xb4));
        node_flip_flop: "FlipFlop" = shade(S::family_transform, rgb(0x46, 0x7e, 0xa8));
        node_gamma: "Gamma" = shade(S::family_colour, rgb(0x7e, 0x56, 0xa8));
        node_tonemap: "Tone Map" = shade(S::family_colour, rgb(0x9a, 0x6a, 0xb8));
        node_expression: "Expression" = shade(S::family_other, rgb(0x7a, 0x86, 0x9a));
        node_lightwrap: "Lightwrap" = shade(S::family_filter, rgb(0x7a, 0x8a, 0xb8));
        node_upscale: "Upscale" = shade(S::family_filter, rgb(0x4f, 0x9c, 0xc4));
        node_seedvr: "SeedVR2" = shade(S::family_filter, rgb(0x6a, 0x8c, 0x5a));
        node_depth: "Depth" = shade(S::family_filter, rgb(0x5a, 0x76, 0xa8));
        node_ltx_hdr: "LTX HDR" = shade(S::family_filter, rgb(0xa8, 0x8e, 0xb8));
        node_hdrcnn: "HDRCNN" = shade(S::family_filter, rgb(0x8e, 0xa0, 0x6c));
        node_grid_warp: "GridWarp" = shade(S::family_transform, rgb(0x35, 0x9a, 0x9a));
        node_warp: "Warp" = shade(S::family_transform, rgb(0x3f, 0x92, 0x8a));
        node_haze: "Haze" = shade(S::family_filter, rgb(0x6f, 0x8a, 0x9a));
        node_keyer: "Keyer" = shade(S::family_matte, rgb(0xcc, 0x4a, 0x6e));
        node_screen_key: "ScreenKey" = shade(S::family_matte, rgb(0x4a, 0xa0, 0x62));
        node_despill: "Despill" = shade(S::family_matte, rgb(0x5a, 0x9a, 0x7a));
        node_corridor_key: "CorridorKey" = shade(S::family_matte, rgb(0x3c, 0xb0, 0x8e));
        node_sammie: "Sammie" = shade(S::family_matte, rgb(0x5a, 0x9c, 0xc8));
        node_layer: "Layer" = shade(S::family_merge, rgb(0x8a, 0x82, 0x6e));
        node_loop: "LoopIn / LoopOut" = shade(S::family_time, rgb(0xc0, 0x6a, 0x2e));
        node_luma_key: "LumaKey" = shade(S::family_matte, rgb(0x7a, 0xa8, 0x9a));
        node_mask: "Mask" = shade(S::family_matte, rgb(0x5e, 0x8a, 0x5e));
        node_matte_combine: "MatteCombine" = shade(S::family_matte, rgb(0x5a, 0x8f, 0x7f));
        node_matte_in: "MatteIn" = shade(S::family_matte, rgb(0x52, 0x8a, 0x86));
        node_mix: "Mix" = shade(S::family_merge, gray(0x6b));
        node_mult: "Mult" = shade(S::family_colour, rgb(0x8a, 0x6f, 0x9e));
        node_noise: "Noise + fields" = shade(S::family_source, rgb(0x5f, 0x7d, 0xc4));
        node_sdf: "SDF + shapes" = shade(S::family_source, rgb(0x4f, 0xa0, 0x96));
        node_paint: "Paint" = shade(S::family_source, rgb(0xc2, 0x5e, 0x35));
        /// A plugin node whose plugin is not installed. Installed plugins
        /// wear the colour their manifest declares.
        node_plugin_missing: "plugin, missing" = shade(S::family_other, gray(0x55));
        /// A tool's control nodes (`ui.*`) wear the cable amber they drive.
        node_tool_control: "subwob controls" = like(S::cable);
        node_tool_input: "subwob inputs" = shade(S::family_other, rgb(0x6a, 0x94, 0xd8));
        node_premult: "Premult" = shade(S::family_matte, rgb(0x62, 0x6a, 0x9e));
        node_read: "Load" = shade(S::family_source, rgb(0xa8, 0x2e, 0x7e));
        node_recombine: "Recombine" = shade(S::family_channels, rgb(0x35, 0x7f, 0x84));
        node_reformat: "Reformat" = shade(S::family_transform, rgb(0x2f, 0x82, 0xb5));
        node_rifetime: "Rifetime" = shade(S::family_time, rgb(0x3e, 0x7f, 0xd9));
        node_freeze: "Freeze" = shade(S::family_time, rgb(0x5a, 0x8c, 0xc0));
        node_framefix: "Frame Fix" = shade(S::family_time, rgb(0x2f, 0x9a, 0xb0));
        node_flow: "Flow" = shade(S::family_time, rgb(0x2e, 0x9f, 0xc9));
        node_sam: "Sam" = shade(S::family_matte, rgb(0x6a, 0x9e, 0xc8));
        node_saturation: "Saturation" = shade(S::family_colour, rgb(0x7a, 0x4f, 0xb0));
        node_contrast: "Contrast" = shade(S::family_colour, rgb(0x6c, 0x5a, 0xa8));
        node_shader: "Shader" = shade(S::family_other, rgb(0x9a, 0x7a, 0xc8));
        /// Muted while the shader does not compile: it is passing through.
        node_shader_broken: "Shader, not compiling" = shade(S::node_shader, rgb(0x6a, 0x5a, 0x72));
        node_shuffle: "Reorder" = shade(S::family_channels, rgb(0x3f, 0x9e, 0x6a));
        node_time: "Slip / Speed / TimeRemap" = shade(S::family_time, rgb(0x6a, 0x76, 0x8c));
        node_smear: "Smear" = shade(S::family_filter, rgb(0x8a, 0x6f, 0x9a));
        node_spline: "Spline" = shade(S::family_matte, rgb(0x86, 0xb8, 0x4a));
        node_switch: "Switch" = shade(S::family_merge, rgb(0x8c, 0x86, 0x62));
        node_tinter: "Tinter" = shade(S::family_colour, rgb(0xa8, 0x4f, 0x86));
        node_to_scrub: "To Scrub" = shade(S::family_time, rgb(0x2e, 0xa5, 0xd0));
        node_to_matte: "ToMatte" = shade(S::family_matte, rgb(0x6e, 0xa8, 0x72));
        node_to_data: "ToData" = like(S::wire_data);
        node_to_rgba: "ToRgba" = like(S::node_to_matte);
        node_soft_crop: "SoftCrop" = shade(S::family_transform, rgb(0x5a, 0x8a, 0xb5));
        node_motion_blur: "MotionBlur" = shade(S::family_filter, rgb(0x4a, 0x7c, 0xc4));
        node_tracker: "Tracker" = shade(S::family_track, rgb(0xd0, 0x4a, 0x4a));
        node_planar: "PlanarTrack" = shade(S::family_track, rgb(0xb8, 0x4a, 0x9c));
        node_transform: "Transform" = shade(S::family_transform, rgb(0x3a, 0x6e, 0xa5));
        node_vector_blur: "VectorBlur" = shade(S::family_filter, rgb(0x4f, 0x8a, 0x3f));
        node_uvwarp: "UV Warp" = shade(S::family_transform, rgb(0x3a, 0x86, 0xa8));
        node_vector_warp: "VectorWarp" = shade(S::family_transform, rgb(0x4e, 0x72, 0xc8));
        node_vector_distort: "VectorDistort" = shade(S::family_transform, rgb(0x9a, 0x7a, 0x35));
        node_write: "Render" = shade(S::family_other, rgb(0xd0, 0x6a, 0x2e));
    }
}

const N: usize = DEFS.len();

// ---------------------------------------------------------------------------
// Storage. LIVE holds the resolved colours (premultiplied rgba packed into a
// u32); OVERRIDES is the only authored state.

const fn pack(c: Color32) -> u32 {
    u32::from_le_bytes(c.to_array())
}

fn unpack(u: u32) -> Color32 {
    let [r, g, b, a] = u.to_le_bytes();
    Color32::from_rgba_premultiplied(r, g, b, a)
}

/// A slot's colour with nothing overridden anywhere.
const fn default_of(s: Slot) -> Color32 {
    match DEFS[s as usize].base {
        Base::Rgb(c) | Base::Shade(_, c) => c,
        Base::Like(p) => default_of(p),
    }
}

static OVERRIDES: RwLock<[Option<Color32>; N]> = RwLock::new([None; N]);

/// Bumped whenever LIVE changes, so [`sync`] knows a context is stale.
static GENERATION: AtomicU32 = AtomicU32::new(1);

#[inline]
pub fn get(s: Slot) -> Color32 {
    unpack(LIVE[s as usize].load(Ordering::Relaxed))
}

/// `c` lighter by `by` levels per channel (darker when negative), alpha kept.
/// For a site that sits a fixed step off a slot — an unbound keycap a shade
/// under a bound one — so the step holds in any theme.
pub fn lift(c: Color32, by: i16) -> Color32 {
    let [r, g, b, a] = c.to_srgba_unmultiplied();
    let m = |x: u8| (x as i16 + by).clamp(0, 255) as u8;
    Color32::from_rgba_unmultiplied(m(r), m(g), m(b), a)
}

/// `c` moved by however far `to` is from `from`, per channel, in the sRGB
/// bytes the eye judges; alpha stays `c`'s.
fn shift(c: Color32, from: Color32, to: Color32) -> Color32 {
    if from == to {
        return c;
    }
    let [r, g, b, a] = c.to_srgba_unmultiplied();
    let [fr, fg, fb, _] = from.to_srgba_unmultiplied();
    let [tr, tg, tb, _] = to.to_srgba_unmultiplied();
    let m = |x: u8, f: u8, t: u8| (x as i16 + t as i16 - f as i16).clamp(0, 255) as u8;
    Color32::from_rgba_unmultiplied(m(r, fr, tr), m(g, fg, tg), m(b, fb, tb), a)
}

/// What slot `i` is without an override of its own, given every slot before
/// it already resolved (links only point backwards — a test holds that).
fn inherited(i: usize, out: &[Color32]) -> Color32 {
    match DEFS[i].base {
        Base::Rgb(c) => c,
        Base::Like(p) => out[p as usize],
        Base::Shade(p, c) => shift(c, default_of(p), out[p as usize]),
    }
}

fn resolve(over: &[Option<Color32>; N]) -> [Color32; N] {
    let mut out = [Color32::TRANSPARENT; N];
    for i in 0..N {
        out[i] = over[i].unwrap_or_else(|| inherited(i, &out));
    }
    out
}

fn apply(over: &[Option<Color32>; N]) {
    for (slot, c) in LIVE.iter().zip(resolve(over)) {
        slot.store(pack(c), Ordering::Relaxed);
    }
    GENERATION.fetch_add(1, Ordering::Relaxed);
}

/// Recolour a slot. Setting it to what it would inherit anyway REMOVES the
/// override, so prefs stay a diff and the link comes back.
pub fn set(s: Slot, c: Color32) {
    let mut o = OVERRIDES.write().unwrap();
    o[s as usize] = None;
    if resolve(&o)[s as usize] != c {
        o[s as usize] = Some(c);
    }
    apply(&o);
}

pub fn reset(s: Slot) {
    let mut o = OVERRIDES.write().unwrap();
    o[s as usize] = None;
    apply(&o);
}

pub fn reset_all() {
    let mut o = OVERRIDES.write().unwrap();
    *o = [None; N];
    apply(&o);
}

pub fn is_overridden(s: Slot) -> bool {
    OVERRIDES.read().unwrap()[s as usize].is_some()
}

pub fn any_overridden() -> bool {
    OVERRIDES.read().unwrap().iter().any(Option::is_some)
}

// ---------------------------------------------------------------------------
// Prefs.

/// `#rrggbb`, or `#rrggbbaa` when there is an alpha to say.
fn hex(c: Color32) -> String {
    let [r, g, b, a] = c.to_srgba_unmultiplied();
    if a == 255 {
        format!("#{r:02x}{g:02x}{b:02x}")
    } else {
        format!("#{r:02x}{g:02x}{b:02x}{a:02x}")
    }
}

fn format_overrides(over: &[Option<Color32>; N]) -> BTreeMap<String, String> {
    over.iter()
        .enumerate()
        .filter_map(|(i, c)| c.map(|c| (DEFS[i].id.to_string(), hex(c))))
        .collect()
}

/// An id this build does not know, or a colour that does not parse, is
/// skipped with a warning — a downgraded build must not wedge on a newer
/// theme.
fn parse_overrides(map: &BTreeMap<String, String>) -> [Option<Color32>; N] {
    let mut over = [None; N];
    for (id, s) in map {
        // Slots renamed since, under the id old prefs saved.
        let id = match id.as_str() {
            "node_stmap" => "node_uvwarp",
            "node_rife" => "node_flow",
            "node_retime" => "node_rifetime",
            other => other,
        };
        let Some(i) = DEFS.iter().position(|d| d.id == id) else {
            log::warn!("theme: unknown colour id {id:?} in prefs, skipped");
            continue;
        };
        let t = s.trim();
        let t = if t.starts_with('#') { t.to_string() } else { format!("#{t}") };
        match Color32::from_hex(&t) {
            Ok(c) => over[i] = Some(c),
            Err(_) => log::warn!("theme: unparsable colour {s:?} for {id}, skipped"),
        }
    }
    // A stored value equal to what the slot inherits is no override at all.
    let mut o = [None; N];
    for i in 0..N {
        if let Some(c) = over[i] {
            if resolve(&o)[i] != c {
                o[i] = Some(c);
            }
        }
    }
    o
}

/// The overrides as prefs strings, ready to serialize.
pub fn overrides() -> BTreeMap<String, String> {
    format_overrides(&OVERRIDES.read().unwrap())
}

/// Replace the overrides from prefs strings.
pub fn install(map: &BTreeMap<String, String>) {
    let mut o = OVERRIDES.write().unwrap();
    *o = parse_overrides(map);
    apply(&o);
}

// ---------------------------------------------------------------------------
// egui.

/// egui's dark-mode colours, driven by the interface slots. Every shade egui
/// keeps that has no slot of its own moves with the slot it belongs to, so
/// the defaults reproduce `Visuals::dark()` exactly (bar the status colours,
/// which are the app's own).
fn write_visuals(v: &mut egui::Visuals, c: &[Color32; N]) {
    let at = |s: Slot| c[s as usize];
    let follow = |own: Color32, s: Slot| shift(own, default_of(s), at(s));
    let w = &mut v.widgets;

    v.panel_fill = at(S::bg);
    w.noninteractive.bg_fill = at(S::bg);
    w.noninteractive.weak_bg_fill = at(S::bg);
    w.open.bg_fill = at(S::bg);
    v.window_fill = at(S::window);
    v.extreme_bg_color = at(S::field);
    v.window_shadow.color = at(S::shadow);
    v.popup_shadow.color = at(S::shadow);
    v.code_bg_color = follow(Color32::from_gray(64), S::bg);

    v.window_stroke.color = at(S::border);
    w.noninteractive.bg_stroke.color = at(S::border);
    w.open.bg_stroke.color = at(S::border);
    w.hovered.bg_stroke.color = follow(Color32::from_gray(150), S::border);

    w.noninteractive.fg_stroke.color = at(S::text);
    w.inactive.fg_stroke.color = follow(Color32::from_gray(180), S::text);
    w.hovered.fg_stroke.color = follow(Color32::from_gray(240), S::text);
    w.open.fg_stroke.color = follow(Color32::from_gray(210), S::text);
    w.active.fg_stroke.color = follow(Color32::WHITE, S::text);
    w.active.bg_stroke.color = follow(Color32::WHITE, S::text);

    w.inactive.bg_fill = at(S::button);
    w.inactive.weak_bg_fill = at(S::button);
    w.open.weak_bg_fill = follow(Color32::from_gray(45), S::button);
    w.hovered.bg_fill = at(S::button_hover);
    w.hovered.weak_bg_fill = at(S::button_hover);
    w.active.bg_fill = at(S::button_active);
    w.active.weak_bg_fill = at(S::button_active);

    v.selection.bg_fill = at(S::accent);
    v.selection.stroke.color = at(S::accent_text);
    v.text_cursor.stroke.color = at(S::accent_text);
    v.hyperlink_color = at(S::link);

    v.warn_fg_color = at(S::warn);
    v.error_fg_color = at(S::error);
}

fn live() -> [Color32; N] {
    std::array::from_fn(|i| unpack(LIVE[i].load(Ordering::Relaxed)))
}

/// Bring a context's visuals up to date with the theme. Cheap when nothing
/// moved (one atomic load and a memory read), so call it before every
/// `run_ui` — the main window and each detached pane own a context.
pub fn sync(ctx: &egui::Context) {
    let gen = GENERATION.load(Ordering::Relaxed);
    let id = egui::Id::new("wobble-theme-generation");
    if ctx.data(|d| d.get_temp::<u32>(id)) == Some(gen) {
        return;
    }
    ctx.data_mut(|d| d.insert_temp(id, gen));
    let c = live();
    ctx.style_mut_of(egui::Theme::Dark, |s| write_visuals(&mut s.visuals, &c));
}

// ---------------------------------------------------------------------------
// The colours… window.

#[derive(Default)]
pub struct ThemeUi {
    pub open: bool,
    search: String,
    /// Edited but not yet written to prefs: a picker drag recolours every
    /// frame, and prefs are written once, when the pointer lets go.
    dirty: bool,
    /// List every slot, not just the roots the rest follow.
    show_all: bool,
}

/// A slot the short list shows: one with a colour of its own (nothing to
/// follow), or one somebody has already set.
fn is_root(i: usize) -> bool {
    DEFS[i].base.parent().is_none() || is_overridden(ALL[i])
}

/// How many slots move when slot `i` does, following links transitively.
fn carried(i: usize) -> usize {
    let mut moves = [false; N];
    moves[i] = true;
    let mut n = 0;
    // Links only point backwards, so one forward pass sees every chain.
    for j in i + 1..N {
        if DEFS[j].base.parent().is_some_and(|p| moves[p as usize]) {
            moves[j] = true;
            n += 1;
        }
    }
    n
}

fn row_matches(d: &Def, q: &str) -> bool {
    q.is_empty()
        || d.label.to_lowercase().contains(q)
        || d.group.to_lowercase().contains(q)
        || d.id.replace('_', " ").contains(q)
}

impl ThemeUi {
    /// Draw the window (when open). Returns true when the overrides should
    /// be written to prefs — the caller mirrors [`overrides`] and saves.
    pub fn window(&mut self, ctx: &egui::Context, fullscreen: bool) -> bool {
        if self.open && !fullscreen {
            let mut open = self.open;
            egui::Window::new("colours")
                .open(&mut open)
                .default_width(360.0)
                .show(ctx, |ui| self.contents(ui));
            self.open = open;
        }
        if self.dirty {
            // egui's own widgets pick the change up at the next `sync`.
            ctx.request_repaint();
        }
        let settled = !self.open || ctx.input(|i| !i.pointer.any_down());
        if self.dirty && settled {
            self.dirty = false;
            return true;
        }
        false
    }

    fn contents(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.search)
                    .hint_text("search colours…")
                    .desired_width(200.0),
            );
            if !self.search.is_empty() && ui.small_button("×").clicked() {
                self.search.clear();
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .add_enabled(any_overridden(), egui::Button::new("reset all").small())
                    .on_hover_text("every colour back to its default")
                    .clicked()
                {
                    reset_all();
                    self.dirty = true;
                }
            });
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.show_all, format!("all {N}"))
                .on_hover_text("also list the colours that follow another one");
            ui.weak("setting a colour moves everything that follows it");
        });
        ui.separator();
        let q = self.search.trim().to_lowercase();
        // A search looks through everything; otherwise the short list.
        let short = q.is_empty() && !self.show_all;
        let mut any = false;
        egui::ScrollArea::vertical().id_salt("theme-list").max_height(480.0).show(ui, |ui| {
            let mut start = 0;
            while start < N {
                let group = DEFS[start].group;
                let end = (start..N).find(|&i| DEFS[i].group != group).unwrap_or(N);
                let rows: Vec<usize> = (start..end)
                    .filter(|&i| row_matches(&DEFS[i], &q) && (!short || is_root(i)))
                    .collect();
                start = end;
                if rows.is_empty() {
                    continue;
                }
                any = true;
                let header = egui::CollapsingHeader::new(group)
                    .id_salt(("theme-group", group))
                    .default_open(group == DEFS[0].group);
                // A search opens every group it hits, and closes nothing.
                let header = if q.is_empty() { header } else { header.open(Some(true)) };
                header.show(ui, |ui| {
                    egui::Grid::new(("theme-grid", group))
                        .num_columns(3)
                        .spacing([8.0, 4.0])
                        .show(ui, |ui| {
                            for i in rows {
                                self.row(ui, i);
                                ui.end_row();
                            }
                        });
                });
            }
        });
        if !any {
            ui.weak("no colour matches");
        }
    }

    fn row(&mut self, ui: &mut egui::Ui, i: usize) {
        use crate::colorpicker::{dec1, enc1, PickSpec};
        let d = &DEFS[i];
        let slot = ALL[i];
        let cur = get(slot);
        let [r, g, b, a] = cur.to_srgba_unmultiplied();
        let byte = |x: u8| dec1(x as f32 / 255.0);
        let mut v = [byte(r), byte(g), byte(b), a as f32 / 255.0];
        let spec = if d.alpha { PickSpec::rgba() } else { PickSpec::rgb() };
        if crate::colorpicker::button(ui, ("theme", i), &mut v, spec).changed() {
            let e = |x: f32| (enc1(x).clamp(0.0, 1.0) * 255.0).round() as u8;
            let alpha = if spec.alpha { (v[3].clamp(0.0, 1.0) * 255.0).round() as u8 } else { a };
            let c = Color32::from_rgba_unmultiplied(e(v[0]), e(v[1]), e(v[2]), alpha);
            if c != cur {
                set(slot, c);
                self.dirty = true;
            }
        }
        let label = ui.label(d.label);
        if !d.tip.trim().is_empty() {
            label.on_hover_ui(|ui| {
                ui.label(d.tip.lines().map(str::trim).collect::<Vec<_>>().join(" ").trim());
            });
        }
        ui.horizontal(|ui| {
            if is_overridden(slot) {
                if ui.small_button("⟲").on_hover_text("back to its default").clicked() {
                    reset(slot);
                    self.dirty = true;
                }
            } else if let Some(p) = d.base.parent() {
                let p = &DEFS[p as usize];
                if p.group == d.group {
                    ui.weak(format!("follows {}", p.label));
                } else {
                    ui.weak(format!("follows {} › {}", p.group, p.label));
                }
            } else {
                let n = carried(i);
                if n > 0 {
                    ui.weak(format!("+{n} follow"));
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn links_point_backwards() {
        for (i, d) in DEFS.iter().enumerate() {
            if let Some(p) = d.base.parent() {
                assert!((p as usize) < i, "{} links forward to {}", d.id, DEFS[p as usize].id);
            }
        }
    }

    #[test]
    fn ids_and_labels_are_unique() {
        let mut ids = std::collections::HashSet::new();
        let mut labels = std::collections::HashSet::new();
        for d in DEFS {
            assert!(ids.insert(d.id), "duplicate id {}", d.id);
            assert!(labels.insert((d.group, d.label)), "duplicate label {} in {}", d.label, d.group);
        }
    }

    #[test]
    fn nothing_overridden_is_every_literal() {
        let c = resolve(&[None; N]);
        for (i, s) in ALL.iter().enumerate() {
            assert_eq!(c[i], default_of(*s), "{}", DEFS[i].id);
            assert_eq!(get(*s), default_of(*s), "LIVE starts at the defaults: {}", DEFS[i].id);
        }
    }

    #[test]
    fn links_follow_their_slot() {
        let mut over = [None; N];
        over[S::bg as usize] = Some(Color32::from_rgb(40, 50, 90));
        over[S::button as usize] = Some(Color32::from_gray(250));
        let c = resolve(&over);
        // shade: own default + the parent's move, per channel.
        assert_eq!(c[S::field as usize], Color32::from_rgb(23, 33, 73));
        // clamped at the top, not wrapped.
        assert_eq!(c[S::button_hover as usize], Color32::from_gray(255));
        // an override of the child wins over the link.
        over[S::field as usize] = Some(Color32::RED);
        assert_eq!(resolve(&over)[S::field as usize], Color32::RED);
    }

    #[test]
    fn prefs_round_trip() {
        let mut map = BTreeMap::new();
        map.insert("accent".to_string(), "#ff8800".to_string());
        map.insert("field".to_string(), "#10203040".to_string());
        map.insert("not_a_slot".to_string(), "#ffffff".to_string());
        map.insert("text".to_string(), "chartreuse".to_string());
        // What the slot has anyway is not stored.
        map.insert("border".to_string(), "#3c3c3c".to_string());
        let over = parse_overrides(&map);
        assert_eq!(over[S::accent as usize], Some(Color32::from_rgb(0xff, 0x88, 0x00)));
        assert_eq!(over[S::text as usize], None);
        assert_eq!(over[S::border as usize], None);
        let back = format_overrides(&over);
        assert_eq!(back.len(), 2);
        assert_eq!(back["accent"], "#ff8800");
        assert_eq!(back["field"], "#10203040");
        assert_eq!(parse_overrides(&back), over);
    }

    /// `cargo test -p app theme_window_render -- --ignored` renders the
    /// colours window twice, text and all — the default theme, then a
    /// recoloured one — to `target/theme-window-gpu.png` and
    /// `target/theme-window-custom-gpu.png`. It moves the LIVE theme, so it
    /// is ignored and runs alone.
    #[test]
    #[ignore]
    fn theme_window_render() {
        let gpu = match engine::GpuContext::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("SKIP theme_window_render: no adapter ({e})");
                return;
            }
        };
        render_window(&gpu, "theme-window-gpu");
        let custom: BTreeMap<String, String> = [
            ("bg", "#1c2230"),
            ("border", "#3a4660"),
            ("text", "#a8b4cc"),
            ("button", "#34405a"),
            ("accent", "#6d3fd0"),
            ("wire", "#50c8a0"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        install(&custom);
        render_window(&gpu, "theme-window-custom-gpu");
        install(&BTreeMap::new());
    }

    fn render_window(gpu: &engine::GpuContext, name: &str) {
        let (w_pt, h_pt, ppp) = (420.0f32, 640.0f32, 2.0f32);
        let ctx = egui::Context::default();
        ctx.set_theme(egui::Theme::Dark);
        sync(&ctx);
        let mut theme_ui = ThemeUi { open: true, ..Default::default() };
        let mut deltas = Vec::new();
        let mut last = None;
        for _ in 0..4 {
            let mut input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w_pt, h_pt))),
                ..Default::default()
            };
            input.viewports.values_mut().for_each(|v| v.native_pixels_per_point = Some(ppp));
            let out = ctx.run_ui(input, |ui| {
                egui::Frame::window(ui.style()).show(ui, |ui| theme_ui.contents(ui));
            });
            deltas.extend(out.textures_delta.set.clone());
            last = Some(out);
        }
        let out = last.unwrap();
        let prims = ctx.tessellate(out.shapes, out.pixels_per_point);
        let (w, h) = ((w_pt * ppp) as u32, (h_pt * ppp) as u32);
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("theme-window"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        let mut renderer =
            egui_wgpu::Renderer::new(&gpu.device, format, egui_wgpu::RendererOptions::default());
        for (id, ds) in &deltas {
            for d in ds {
                renderer.update_texture(&gpu.device, &gpu.queue, *id, d);
            }
        }
        let screen = egui_wgpu::ScreenDescriptor { size_in_pixels: [w, h], pixels_per_point: ppp };
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        let user = renderer.update_buffers(&gpu.device, &gpu.queue, &mut enc, &prims, &screen);
        {
            let clear = egui::Rgba::from(bg());
            let mut rp = enc
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("theme-window"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: clear.r() as f64,
                                g: clear.g() as f64,
                                b: clear.b() as f64,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            renderer.render(&mut rp, &prims, &screen);
        }
        let unpadded = w as usize * 4;
        let padded = unpadded.div_ceil(256) * 256;
        let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("theme-window-rb"),
            size: (padded * h as usize) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded as u32),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        gpu.queue.submit(user.into_iter().chain([enc.finish()]));
        let (tx, rx) = std::sync::mpsc::channel();
        buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();
        let mut rgb = Vec::with_capacity(w as usize * h as usize * 3);
        {
            let data = buf.slice(..).get_mapped_range().expect("a mapped buffer");
            for y in 0..h as usize {
                for px in data[y * padded..y * padded + unpadded].chunks_exact(4) {
                    rgb.extend_from_slice(&px[..3]);
                }
            }
        }
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../target/{name}.png"));
        let file = std::fs::File::create(&path).unwrap();
        let mut png = png::Encoder::new(std::io::BufWriter::new(file), w, h);
        png.set_color(png::ColorType::Rgb);
        png.set_depth(png::BitDepth::Eight);
        png.write_header().unwrap().write_image_data(&rgb).unwrap();
        eprintln!("wrote {}", path.display());
    }

    #[test]
    fn default_visuals_are_egui_dark() {
        let mut v = egui::Visuals::dark();
        write_visuals(&mut v, &resolve(&[None; N]));
        let mut want = egui::Visuals::dark();
        want.warn_fg_color = default_of(S::warn);
        want.error_fg_color = default_of(S::error);
        assert_eq!(v, want);
    }
}
