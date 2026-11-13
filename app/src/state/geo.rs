//! 2dini's viewer state: the tool in hand, the drags in flight, the HUD and onion caches.
//! These were loose fields on `Gfx`; grouped, not changed.

use crate::*;

/// 2dini's viewer state: the tool in hand, the drags in flight, the HUD and onion caches.
pub(crate) struct GeoUiState {
    /// What the vector HUD last evaluated inside a 2dini dive, so a still
    /// drawing is evaluated once and an animated one once per tick.
    pub(crate) hud_cache: Option<GeoHudCache>,
    /// The last 2dini web-page export: (node, what happened).
    pub(crate) web_status: Option<(u64, String)>,
    /// Export web pages that evaluate live and play forever (vs loop the
    /// play range's frames).
    pub(crate) web_live: bool,
    /// A 2dini Transform handle being dragged: (node, handle, press point,
    /// the node's translate/rotate/scale/pivot at the press).
    pub(crate) xf_drag: Option<GeoXfDrag>,
    /// Where the display bar was last frame, so a click on it is not a
    /// click-through into the drawing under it.
    pub(crate) viz_bar: Option<egui::Rect>,
    /// The 2dini tool bar down the viewer's left edge, last drawn.
    pub(crate) tool_bar: Option<egui::Rect>,
    /// The armed shape tool: the next press in the viewer draws one.
    pub(crate) tool: Option<GeoTool>,
    /// A shape being drawn with the armed tool: the node and where the
    /// press landed, geo space.
    pub(crate) tool_drag: Option<(u64, [f32; 2])>,
    /// A brush stroke being painted: the Brush node and the last point
    /// laid (screen).
    pub(crate) brush_drag: Option<(u64, egui::Pos2)>,
    /// The Edit node's picked points, (shape, vertex).
    pub(crate) edit_pick: Vec<(u32, u32)>,
    /// Which Edit node the pick belongs to.
    pub(crate) edit_owner: Option<u64>,
    /// Onion skins drawn lately: (frame, the graph at it, its geometry).
    pub(crate) onion_cache: Vec<(u64, engine::geo::GeoGraph, engine::geo::Geometry)>,
    /// An Edit gesture in flight: a point drag (the edits as they were at
    /// the press) or a box (its first corner).
    pub(crate) edit_drag: Option<GeoEditDrag>,
    pub(crate) bone_drag: Option<BoneDrag>,
    /// A gradient handle being dragged: node, 0 start / 1 end, the press,
    /// and the point as it was.
    pub(crate) grad_drag: Option<(u64, u8, egui::Pos2, [f32; 2])>,
    /// Last frame's tool buttons (screen), for the suites.
    pub(crate) tool_spots: Vec<(GeoTool, egui::Rect)>,
    /// --test-geo: the rendered mean before the exposed slider was driven.
    pub(crate) mean: f32,
    /// --test-geo: the Physics node under test and the most frames its live
    /// bake was seen with.
    pub(crate) phys_node: u64,
    pub(crate) phys_seen: u32,
    /// --test-geo: viewer clicks retried, capped so a miss fails loudly.
    pub(crate) click_tries: u32,
}
