//! Standardized properties widgets: every parameter row is label + control +
//! keyframe dot, sized for the tile grid. Any row can be ctrl-dragged out of
//! the panel and dropped on the viewer to become a floating HUD slider bound
//! to its channels (e.g. drag "rotate" out of a Transform).

use crate::anim::{self, AnimCurve, Channel};
use crate::theme;
use engine::graph::NodeKind;
use std::collections::BTreeMap;

/// Numeric field geometry. Every float control is EXACTLY this size, no
/// matter what it is displaying — a panel is n whole tiles wide and must
/// stay that way, so a value like 0.28385 can never push the grid apart.
pub const NUM_W: f32 = 54.0;
pub const NUM_H: f32 = 18.0;
/// A number field at EXACTLY `NUM_W` x `NUM_H`, whatever it shows. See
/// [`num_field_w`].
pub fn num_field(ui: &mut egui::Ui, dv: egui::DragValue<'_>) -> egui::Response {
    num_field_w(ui, NUM_W, dv)
}

/// A number field at EXACTLY `w` x `NUM_H` (`crate::numfield`): the text
/// fits the box, never the other way round.
pub fn num_field_w(ui: &mut egui::Ui, w: f32, dv: egui::DragValue<'_>) -> egui::Response {
    crate::numfield::fixed(ui, w, NUM_H, dv)
}

/// Fixed label column, for the same reason as `NUM_W`: every row's control
/// starts at the same x whatever its label says. Sized for the longest row
/// label in the app ("odd ray length"); anything longer truncates rather
/// than shoving its own row out of the column.
pub const LABEL_W: f32 = 74.0;
/// The keyframe dot's fixed slot. The three glyphs (● ◌ ○) are not the
/// same advance width, and a row is only as aligned as its widest-varying
/// part.
pub const DOT_W: f32 = 18.0;
/// Digits after the point, bounded so the text can't outgrow the field.
/// CONFIGURABLE: three is noise on a pixel count and not enough on a gain,
/// and which it is depends on the comp, not on the widget.
///
/// A process-wide atomic rather than a threaded parameter — it is a display
/// preference, and every row in the app would otherwise have to carry it
/// down from wherever the setting lives. Same pattern as `TEST_RUN`.
pub const NUM_DECIMALS: usize = 3;
static DECIMALS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(NUM_DECIMALS);

pub fn decimals() -> usize {
    DECIMALS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Clamped to what a `NUM_W`-wide field can actually show.
pub fn set_decimals(n: usize) {
    DECIMALS.store(n.clamp(0, 6), std::sync::atomic::Ordering::Relaxed);
}

/// Zoom bounds for the properties grid. Zoom scales the tile — and with
/// it every control whose PRECISION comes from its size (slider tracks,
/// colour wheels) — while type, drag fields and knobs keep their size.
///
/// Wide enough that the patch canvas, which zooms on the plain wheel like
/// the node graph, can pull back far enough to see a whole rig. Not as wide
/// as the graph's: below a quarter, a tile is too small to hold a row.
pub const ZOOM_MIN: f32 = 0.25;
pub const ZOOM_MAX: f32 = 6.0;
/// Per-axis tile stretch in the properties grid (on top of the zoom).
pub const STRETCH_MIN: f32 = 0.5;
pub const STRETCH_MAX: f32 = 4.0;

/// Tile module: the grid cell is EXACTLY SQUARE, and every panel is a whole
/// number of tiles — panel design conforms to the module, not the other way
/// round. Sized so the tallest content (the Grade wheels, wheels.rs
/// COL_H = 187, plus header + gang row) fits one tile of height.
///
/// The GRID is twice as fine as the tile: a tile is 2x2 `CELL`s plus the
/// gap between them, so panels still declare whole tiles (`tiles_for`,
/// `tile_rows`, plugin and tool manifests) while the patch view snaps and
/// packs on the finer cell. `patch_span_of` is the one place tiles become
/// cells.
pub const CELL: f32 = 128.0;
pub const TILE_GAP: f32 = 8.0;
/// Grid cells per tile, per axis.
pub const CELLS_PER_TILE: usize = 2;
pub const TILE: f32 = CELL * CELLS_PER_TILE as f32 + TILE_GAP * (CELLS_PER_TILE - 1) as f32;

/// Where a control landed AND the clip it was drawn under, by id, for the
/// UI suites to aim at — the `rgbslider::last_geom` pattern, one small write
/// a frame. The rect says where to press; the clip says whether a hand
/// could. A control pushed past the bottom of its tile still has a rect,
/// just no pixels: the Grade's zone preview sat there, pressable by a suite
/// and invisible to everyone else.
pub fn note_rect(ui: &egui::Ui, id: egui::Id, rect: egui::Rect) {
    let clip = ui.clip_rect();
    ui.ctx().data_mut(|d| d.insert_temp(id, (rect, clip)));
}

/// What [`note_rect`] recorded: `(rect, clip)`.
pub fn noted_rect(ctx: &egui::Context, id: egui::Id) -> Option<(egui::Rect, egui::Rect)> {
    ctx.data(|d| d.get_temp(id))
}

/// How many grid tiles a node's panel spans.
pub fn tiles_for(kind: &NodeKind) -> usize {
    match kind {
        // Six hue bands of three values each need the room a Grade needs.
        // A full curve editor plus the band strip wants Grade's room.
        NodeKind::Grade(_) | NodeKind::LumaKey(_) => 3,
        // A 2dini Text's animate page is the Text node's.
        NodeKind::GeoNode(engine::geo::GeoNode::Text { .. }) => crate::nodes::text::TILES,
        // A timeline needs TRACK: a bar per layer plus the selected layer's
        // controls under it. Three tiles is the width a mini timeline can
        // actually be dragged along.
        NodeKind::Layer(_) => 3,
        // Two colour-space plots side by side plus the controls column.
        NodeKind::Keyer(_) => 3,
        // Layer row, view row, the pick list and the manifest.
        NodeKind::Cryptomatte(_) => 2,
        NodeKind::ToData(_) => crate::nodes::todata::TILES,
        NodeKind::UvWarp(_) => crate::nodes::uvwarp::TILES,
        NodeKind::Env(_) => crate::nodes::env::TILES,
        NodeKind::Refract(_) => crate::nodes::refract::TILES,
        NodeKind::GridWarp(_) => crate::nodes::gridwarp::TILES,
        NodeKind::Warp(_) => crate::nodes::warp::TILES,
        NodeKind::ToRgba(_) => crate::nodes::torgba::TILES,
        NodeKind::SoftCrop(_) => crate::nodes::softcrop::TILES,
        NodeKind::MotionBlur(_) => crate::nodes::motionblur::TILES,
        NodeKind::VectorBlur(_) => crate::nodes::vectorblur::TILES,
        NodeKind::Camera3D(_) => crate::nodes::scene3d::CAMERA_TILES,
        NodeKind::Card3D(_) => crate::nodes::scene3d::CARD_TILES,
        NodeKind::Scene3D(_) => crate::nodes::scene3d::SCENE_TILES,
        NodeKind::Render3D(_) => crate::nodes::scene3d::RENDER_TILES,
        NodeKind::VectorWarp(_) => crate::nodes::vectorwarp::TILES,
        NodeKind::Flow(_) => crate::nodes::flow::TILES,
        NodeKind::Rifetime(_) => crate::nodes::rifetime::TILES,
        NodeKind::Freeze(_) => crate::nodes::freeze::TILES,
        NodeKind::FrameFix(_) => crate::nodes::framefix::TILES,
        // A tick per AOV layer plus the spacing slider: taller than one
        // tile as soon as a real multi-layer EXR arrives.
        NodeKind::Breakout(_) => 2,
        // A row of rgb lanes needs track to aim along: the widget's fixed
        // chrome is ~126px, so these do not fit in one tile.
        NodeKind::Saturation(_) => crate::nodes::saturation::TILES,
        NodeKind::ColorCompress(_) => crate::nodes::colorcompress::TILES,
        NodeKind::HueRotate(_) => crate::nodes::huerotate::TILES,
        NodeKind::Lut(_) => crate::nodes::lut::TILES,
        NodeKind::Text(_) => crate::nodes::text::TILES,
        NodeKind::Burnin(_) => crate::nodes::burnin::TILES,
        NodeKind::Mirror(_) => crate::nodes::mirror::TILES,
        NodeKind::LensDistort(_) => crate::nodes::lensdistort::TILES,
        NodeKind::FlipFlop(_) => crate::nodes::flipflop::TILES,
        NodeKind::Gamma(_) => crate::nodes::gamma::TILES,
        NodeKind::ToneMap(_) => crate::nodes::tonemap::TILES,
        NodeKind::Expression(_) => crate::nodes::expression::TILES,
        NodeKind::Despill(_) => crate::nodes::despill::TILES,
        NodeKind::Lightwrap(_) => crate::nodes::lightwrap::TILES,
        NodeKind::Upscale(_) => crate::nodes::upscale::TILES,
        NodeKind::HdrCnn(_) => crate::nodes::hdrcnn::TILES,
        NodeKind::SeedVr(_) => crate::nodes::seedvr::TILES,
        NodeKind::Depth(_) => crate::nodes::depth::TILES,
        NodeKind::LtxHdr(_) => crate::nodes::ltxhdr::TILES,
        NodeKind::ScreenKey(_) => crate::nodes::screenkey::TILES,
        NodeKind::ColorSlice(_) => crate::nodes::colorslice::TILES,
        NodeKind::Blur(_) => crate::nodes::blur::TILES,
        NodeKind::Dilate(_) => crate::nodes::dilate::TILES,
        NodeKind::Glow(_) => crate::nodes::glow::TILES,
        NodeKind::Godrays(_) => crate::nodes::godrays::TILES,
        NodeKind::LensFlare(_) => crate::nodes::lensflare::TILES,
        NodeKind::Contrast(_) => crate::nodes::contrast::TILES,
        NodeKind::EdgeExtend(_) => crate::nodes::edgeextend::TILES,
        NodeKind::Glint(_) => crate::nodes::glint::TILES,
        NodeKind::GlintKernel(_) => crate::nodes::glintkernel::TILES,
        NodeKind::BokehKernel(_) => crate::nodes::bokehkernel::TILES,
        NodeKind::EdgeDetect(_) => crate::nodes::edgedetect::TILES,
        NodeKind::Median(_) => crate::nodes::median::TILES,
        NodeKind::HighlightRebuild(_) => crate::nodes::highlightrebuild::TILES,
        NodeKind::Stabilize(_) => crate::nodes::stabilize::TILES,
        NodeKind::CamTrack(_) => crate::nodes::camtrack::TILES,
        NodeKind::Convolve(_) => crate::nodes::convolve::TILES,
        NodeKind::Diffuse(_) => crate::nodes::diffuse::TILES,
        NodeKind::Gradient(_) => crate::nodes::gradient::TILES,
        NodeKind::Grain(_) => crate::nodes::grain::TILES,
        NodeKind::Starfield(_) => crate::nodes::starfield::TILES,
        NodeKind::Splat(_) => crate::nodes::splat::TILES,
        NodeKind::Planet(_) => crate::nodes::planet::TILES,
        NodeKind::Plasma(_) => crate::nodes::plasma::TILES,
        NodeKind::Particles(_) | NodeKind::ParticleNode(_) => crate::nodes::particles::TILES,
        NodeKind::Mult(_)
        | NodeKind::Read { .. }
        | NodeKind::Transform(_)
        | NodeKind::CornerPin(_)
        | NodeKind::Curve(_)
        | NodeKind::Grid(_)
        // A drawing's node carries its exposed sliders; the nodes inside
        // fan out socket rows.
        | NodeKind::Geo(_)
        | NodeKind::GeoNode(_)
        | NodeKind::Sdf(_)
        | NodeKind::SdfNode(_)
        | NodeKind::Reformat(_)
        | NodeKind::Tracker(_)
        | NodeKind::PlanarTrack(_)
        // Shape controls, matte controls and a transform: a tile's width of
        // slider track is not enough to aim along.
        | NodeKind::Spline(_)
        // Brush, clone and stroke-lifetime rows.
        | NodeKind::Paint(_)
        // Session buttons, the scene's names and a status line in words.
        | NodeKind::Blender(_)
        | NodeKind::Shuffle(_) => 2,
        NodeKind::Custom { type_id, .. } => engine::plugins::get(type_id)
            .map(|d| d.tiles)
            .unwrap_or(1),
        // Its param list is whatever the shader declares, so give it room;
        // the code itself lives in its own window, not in the tile.
        NodeKind::Shader(_) => 2,
        // Two range bars with the remap drawn between them want the width.
        NodeKind::RangeRemap(_) => 2,
        // Steps on the left, billing + run + takes on the right.
        NodeKind::GenAi(_) => crate::genaiui::TILES,
        NodeKind::Prompt(_) => crate::nodes::prompt::TILES,
        NodeKind::CorridorKey(_) => crate::nodes::corridorkey::TILES,
        NodeKind::Sammie(_) => crate::nodes::sammie::TILES,
        _ => 1,
    }
}

/// One tab of a node's properties panel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PanelTab {
    /// Stable identity, what the selection is stored by.
    pub id: String,
    /// What the strip shows.
    pub label: String,
}

impl PanelTab {
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self { id: id.into(), label: label.into() }
    }
}

/// The tabs a node's panel is divided into. Most panels are one page and
/// declare none; a panel whose controls are a few INSTRUMENTS rather than a
/// list — the Grade's master, each of its zones, and the ranges that cut
/// them — declares one per instrument. `properties_ui` draws the strip in
/// the title row (or under it, when it does not fit beside the name) and
/// hands the node's arm the selected id.
///
/// Ids rather than indices, so the selection survives a tab being added
/// ahead of it: adding zones from the ranges page must not bounce you off
/// the ranges page.
pub fn panel_tabs(kind: &NodeKind) -> Vec<PanelTab> {
    match kind {
        NodeKind::Grade(g) => {
            let mut v = vec![PanelTab::new("master", "master")];
            if let Some(z) = &g.zones {
                for i in 0..z.count() {
                    v.push(PanelTab::new(format!("zone:{i}"), z.label(i)));
                }
            }
            v.push(PanelTab::new("ranges", "ranges"));
            v
        }
        // A 2dini shape (or Style) with a fill: its gradient on a page of
        // its own, where the ramp editor has room.
        NodeKind::GeoNode(engine::geo::GeoNode::Text { .. }) => vec![
            PanelTab::new("shape", "shape"),
            PanelTab::new("style", "style"),
            PanelTab::new("animate", "animate"),
            PanelTab::new("gradient", "gradient"),
        ],
        NodeKind::GeoNode(k)
            if matches!(
                k,
                engine::geo::GeoNode::Spline { .. }
                    | engine::geo::GeoNode::Text { .. }
                    | engine::geo::GeoNode::Polygon { .. }
                    | engine::geo::GeoNode::Star { .. }
                    | engine::geo::GeoNode::Circle { .. }
                    | engine::geo::GeoNode::Rectangle { .. }
                    | engine::geo::GeoNode::Style { .. }
            ) =>
        {
            vec![PanelTab::new("shape", "shape"), PanelTab::new("gradient", "gradient")]
        }
        // The drawing, its light, its SDF effects (`geolightui`).
        NodeKind::Geo(_) => vec![
            PanelTab::new("drawing", "drawing"),
            PanelTab::new("light", "light"),
            PanelTab::new("effects", "effects"),
        ],
        NodeKind::LensDistort(_) => crate::nodes::lensdistort::tabs(),
        NodeKind::Text(_) => crate::nodes::text::tabs(),
        NodeKind::Starfield(_) => crate::nodes::starfield::tabs(),
        NodeKind::Planet(_) => crate::nodes::planet::tabs(),
        NodeKind::Plasma(_) => crate::nodes::plasma::tabs(),
        NodeKind::LensFlare(_) => crate::nodes::lensflare::tabs(),
        NodeKind::Contrast(_) => crate::nodes::contrast::tabs(),
        NodeKind::Refract(_) => crate::nodes::refract::tabs(),
        // The box, and the Crop's own bake.
        NodeKind::SoftCrop(_) => vec![PanelTab::new("crop", "crop"), PanelTab::new("bake", "bake")],
        NodeKind::Paint(_) => vec![
            PanelTab::new("brush", "brush"),
            PanelTab::new("pen", "pen"),
            PanelTab::new("strokes", "strokes"),
        ],
        _ => Vec::new(),
    }
}

/// Room one tab takes beyond its text: button padding both sides plus the
/// gap to the next.
const TAB_PAD: f32 = 16.0;

fn tab_text(label: &str) -> egui::RichText {
    egui::RichText::new(label).small()
}

/// The width the strip will take, so the title row can decide whether the
/// tabs fit beside the name or need a row of their own.
pub fn tab_strip_width(ui: &egui::Ui, tabs: &[PanelTab]) -> f32 {
    tabs.iter()
        .map(|t| {
            egui::WidgetText::from(tab_text(&t.label))
                .into_galley(ui, Some(egui::TextWrapMode::Extend), f32::INFINITY, egui::TextStyle::Small)
                .size()
                .x
                + TAB_PAD
        })
        .sum()
}

/// The strip itself. `sel` is the selected id; it is normalised to a live
/// one on the way out (the first tab when it names nothing), so the caller
/// can match on it without a fallback arm. Returns true when a click
/// changed it.
pub fn tab_strip(ui: &mut egui::Ui, tabs: &[PanelTab], sel: &mut String) -> bool {
    if !tabs.iter().any(|t| t.id == *sel) {
        *sel = tabs.first().map(|t| t.id.clone()).unwrap_or_default();
    }
    let mut changed = false;
    // Tight between tabs; scoped, so the row's own spacing (the buttons
    // after the strip) is left alone.
    ui.scope(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        for t in tabs {
            let on = t.id == *sel;
            let r = ui.add(egui::Button::selectable(on, tab_text(&t.label)));
            note_rect(ui, egui::Id::new(("panel-tab-rect", t.id.as_str())), r.rect);
            if r.clicked() && !on {
                *sel = t.id.clone();
                changed = true;
            }
        }
    });
    changed
}

/// A row that is a DRAG HANDLE and also holds buttons — a panel's title row,
/// which reorders the panel when dragged and carries its tabs, hud toggle
/// and × in the same strip.
///
/// The drag sense is registered BEFORE the row's widgets, on last frame's
/// row rect, so the buttons sit on top of it. egui hands a click to the
/// top-most widget under the pointer, and a drag-only rect registered after
/// the buttons is on top of them and swallows every click (that is how the
/// tab strip arrived dead, and the × beside it). Registering the handle
/// first is the order egui's own scroll areas use. First show, with no rect
/// remembered yet, guesses one row at the cursor.
/// A panel's frame at EXACTLY `size`, contents clipped inside it.
///
/// `Frame::group` draws around whatever its contents ended up taking, so
/// one row wider than the tile (a fixed-width field, a long label) pushed
/// the frame past the grid and over its neighbour in the patch view.
/// Panels are whole tiles; the tile wins and the overflow is clipped.
pub fn tile_frame<R>(
    ui: &mut egui::Ui,
    size: egui::Vec2,
    add: impl FnOnce(&mut egui::Ui) -> R,
) -> R {
    let frame = egui::Frame::group(ui.style());
    let rect = egui::Rect::from_min_size(ui.cursor().min, size);
    ui.painter().rect(
        rect,
        frame.corner_radius,
        frame.fill,
        frame.stroke,
        egui::StrokeKind::Inside,
    );
    let pad = frame.inner_margin.left as f32 + frame.stroke.width;
    let inner = rect.shrink(pad);
    let mut child = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(inner)
            .layout(egui::Layout::top_down(egui::Align::Min)),
    );
    child.shrink_clip_rect(inner);
    let r = add(&mut child);
    // Space only, no interaction: a rect registered over the contents
    // would sit on top of their hover.
    ui.advance_cursor_after_rect(rect);
    r
}

pub fn drag_row<R>(
    ui: &mut egui::Ui,
    id: egui::Id,
    add: impl FnOnce(&mut egui::Ui) -> R,
) -> (egui::Response, egui::InnerResponse<R>) {
    let rect_key = id.with("row-rect");
    let guess = egui::Rect::from_min_size(
        ui.cursor().min,
        egui::vec2(ui.available_width(), ui.spacing().interact_size.y),
    );
    let rect = ui.data(|d| d.get_temp::<egui::Rect>(rect_key)).unwrap_or(guess);
    // Click as well as drag: a click on the bare row selects the panel. The
    // buttons drawn over it are later, so they still win their own clicks.
    let drag = ui.interact(rect, id, egui::Sense::click_and_drag());
    let row = ui.horizontal(add);
    ui.data_mut(|d| d.insert_temp(rect_key, row.response.rect));
    (drag, row)
}

/// How many grid tiles TALL a node's panel is. One unless the panel is

/// mostly a plot, in which case the extra row is all graph — a curve is
/// read by its shape, and a shape gets easier to read the more of it you
/// can see at once.
pub fn tile_rows(kind: &NodeKind) -> usize {
    match kind {
        NodeKind::LumaKey(_) => 2,
        NodeKind::Text(_) | NodeKind::GeoNode(engine::geo::GeoNode::Text { .. }) => crate::nodes::text::TILE_ROWS,
        NodeKind::ColorSlice(_) => crate::nodes::colorslice::TILE_ROWS,
        NodeKind::ColorCompress(_) => crate::nodes::colorcompress::TILE_ROWS,
        NodeKind::Godrays(_) => crate::nodes::godrays::TILE_ROWS,
        NodeKind::LensFlare(_) => crate::nodes::lensflare::TILE_ROWS,
        NodeKind::Planet(_) => crate::nodes::planet::TILE_ROWS,
        NodeKind::Grain(_) => crate::nodes::grain::TILE_ROWS,
        NodeKind::Convolve(_) => crate::nodes::convolve::TILE_ROWS,
        NodeKind::Glint(_) => crate::nodes::glint::TILE_ROWS,
        NodeKind::GlintKernel(_) => crate::nodes::glintkernel::TILE_ROWS,
        NodeKind::BokehKernel(_) => crate::nodes::bokehkernel::TILE_ROWS,
        // Session, scene, picture, frames, inputs, colour and a status in
        // words: more rows than one tile holds.
        NodeKind::Blender(_) => crate::blenderui::TILE_ROWS,
        NodeKind::GenAi(_) => crate::genaiui::TILE_ROWS,
        NodeKind::CorridorKey(_) => crate::nodes::corridorkey::TILE_ROWS,
        NodeKind::Sammie(_) => crate::nodes::sammie::TILE_ROWS,
        NodeKind::Flow(_) => crate::nodes::flow::TILE_ROWS,
        NodeKind::Stabilize(_) => crate::nodes::stabilize::TILE_ROWS,
        NodeKind::CamTrack(_) => crate::nodes::camtrack::TILE_ROWS,
        NodeKind::Rifetime(_) => crate::nodes::rifetime::TILE_ROWS,
        NodeKind::FrameFix(_) => crate::nodes::framefix::TILE_ROWS,
        NodeKind::Upscale(_) => crate::nodes::upscale::TILE_ROWS,
        NodeKind::HdrCnn(_) => crate::nodes::hdrcnn::TILE_ROWS,
        NodeKind::SeedVr(_) => crate::nodes::seedvr::TILE_ROWS,
        NodeKind::Depth(_) => crate::nodes::depth::TILE_ROWS,
        NodeKind::LtxHdr(_) => crate::nodes::ltxhdr::TILE_ROWS,
        // Two projections and the volume they project: the plots want a row
        // and the 3D view wants its own.
        NodeKind::Keyer(_) => 2,
        _ => 1,
    }
}

/// What a standard row needs from its node (borrowed disjointly from ENode).
pub struct RowCtx<'a> {
    pub node: u64,
    pub anim: &'a mut BTreeMap<Channel, AnimCurve>,
    pub ekind: &'a NodeKind,
    pub t: u64,
    /// A HUD drag-out is in flight: rows suppress edits for its whole drag
    /// (not just the first frame — ctrl can even be released mid-gesture).
    pub hud_live: bool,
    /// Channels on this node that a patch cable drives, with their LIVE
    /// values — the widget still binds the authored value, so the row's
    /// label turns amber and carries the real number in its hover.
    pub driven: Vec<(Channel, f32)>,
}

impl RowCtx<'_> {
    pub fn driven_of(&self, chans: &[Channel]) -> Option<f32> {
        chans
            .iter()
            .find_map(|c| self.driven.iter().find(|(dc, _)| dc == c).map(|(_, v)| *v))
    }
}

fn unit_scale() -> egui::Vec2 {
    egui::vec2(1.0, 1.0)
}

/// A floating on-viewer slider bound to node channels, spawned by
/// ctrl-dragging a row out of the properties panel.
#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HudSlider {
    pub node: u64,
    pub chans: Vec<Channel>,
    pub label: String,
    pub lo: f32,
    pub hi: f32,
    /// Viewer-panel-relative position in points.
    #[serde(default)]
    pub pos: egui::Vec2,
    /// Per-axis size (corner-drag resize; alt = uniform).
    #[serde(default = "unit_scale")]
    pub scale: egui::Vec2,
    /// These three channels are an rgb TRIPLE (this HUD was dragged out of
    /// an rgb row), so in the viewer it gangs and takes the colour axes
    /// instead of being three unrelated sliders. Marked rather than guessed:
    /// a Card3D's xyz is also three channels and is not a colour.
    #[serde(default)]
    pub rgb: bool,
    /// Scale-type (a multiplier, ratios) rather than offset-type.
    #[serde(default)]
    pub rgb_scale: bool,
    /// Ganged — only meaningful with `rgb`.
    #[serde(default = "yes")]
    pub gang: bool,
    /// Pulled off a colour WHEEL (a Grade's lift/gamma/gain/offset), so a
    /// drop on a tool's panel composer makes a Colour Wheel control rather
    /// than an rgb slider. The viewer HUD ignores it.
    #[serde(default)]
    pub wheel: bool,
}

fn yes() -> bool {
    true
}

impl Default for HudSlider {
    /// So a spawn site only has to name what it cares about
    /// (`..Default::default()`): a plain channel slider is not rgb, and an
    /// rgb one starts ganged.
    fn default() -> Self {
        Self {
            node: 0,
            chans: Vec::new(),
            label: String::new(),
            lo: 0.0,
            hi: 1.0,
            pos: egui::Vec2::ZERO,
            scale: unit_scale(),
            rgb: false,
            rgb_scale: false,
            gang: true,
            wheel: false,
        }
    }
}

/// The number a parameter row should SHOW.
///
/// An animated channel's authored value is dead data: the curve drives the
/// render, and the properties post-pass turns any edit into a key and puts
/// authored back. A row that showed authored would sit frozen while you drag
/// it, and would start each drag from the wrong base — nudge +2 at a frame
/// whose curve reads 40 and the key would land on 2. So animated rows show
/// the evaluated value, everything else shows what is authored.
pub fn shown_value(cx: &RowCtx, chans: &[Channel], authored: f32) -> f32 {
    chans
        .iter()
        .find(|c| cx.anim.get(c).is_some_and(|k| !k.keys.is_empty()))
        .and_then(|c| anim::get(cx.ekind, *c))
        .unwrap_or(authored)
}

/// Right-click any parameter field to LOCK its channel against keyframing.
///
/// A locked channel takes no keys from anything — this row, a gizmo, a HUD
/// slider, autokey, a tracker bake, the curve editor — so one axis of a
/// transform can be pinned while the other keeps animating. Per FIELD, not
/// per row: `translate` offers x and y separately because that is the whole
/// point of it.
fn lock_menu(cx: &mut RowCtx, r: &egui::Response, chans: &[Channel]) {
    let locked = anim::any_locked(cx.anim, chans);
    r.context_menu(|ui| {
        let label = if locked { "unlock keyframing" } else { "lock keyframing" };
        if ui
            .button(label)
            .on_hover_text("a locked channel takes no keyframes, from anything")
            .clicked()
        {
            for c in chans {
                anim::set_locked(cx.anim, *c, !locked);
            }
            ui.close();
        }
    });
}

/// A bar under a locked field — the lock has to be visible on the control
/// it locks, not only in a menu nobody opens.
fn lock_marker(ui: &egui::Ui, rect: egui::Rect, locked: bool) {
    if !locked {
        return;
    }
    let y = rect.bottom() - 1.0;
    ui.painter().line_segment(
        [egui::pos2(rect.left() + 2.0, y), egui::pos2(rect.right() - 2.0, y)],
        egui::Stroke::new(1.5, theme::locked()),
    );
}

/// What a locked field says when you hover it.
const LOCKED_TIP: &str =
    "locked: this channel takes no keyframes (right-click to unlock)";

/// The keyframe dot: ● keys at t (click removes), ◌ animated elsewhere,
/// ○ static. Right-click clears the channels' animation.
pub fn key_dot(ui: &mut egui::Ui, cx: &mut RowCtx, chans: &[Channel]) {
    let animated = chans
        .iter()
        .any(|c| cx.anim.get(c).is_some_and(|k| !k.keys.is_empty()));
    let keyed_now = animated
        && chans
            .iter()
            .all(|c| cx.anim.get(c).is_some_and(|k| k.has_key(cx.t)));
    // Every channel of the row locked: the dot says so and does nothing.
    let all_locked = !chans.is_empty() && chans.iter().all(|c| anim::is_locked(cx.anim, *c));
    let (glyph, color) = if all_locked {
        ("○", theme::locked())
    } else if keyed_now {
        ("⏺", theme::keyframe())
    } else if animated {
        ("○", theme::animated())
    } else {
        ("○", theme::text())
    };
    let r = ui
        .add_sized(
            [DOT_W, NUM_H],
            egui::Button::new(egui::RichText::new(glyph).color(color)).frame(false),
        )
        .on_hover_text(if all_locked {
            LOCKED_TIP.to_string()
        } else {
            format!(
                "keyframe at frame {}\nclick: set / remove key here\nright-click: clear animation",
                cx.t
            )
        });
    if r.clicked() && !all_locked {
        if keyed_now {
            for c in chans {
                if let Some(k) = cx.anim.get_mut(c) {
                    k.remove_key(cx.t);
                    if k.keys.is_empty() {
                        cx.anim.remove(c);
                    }
                }
            }
        } else {
            // Key what's on screen: the EVALUATED value at this frame.
            for c in chans {
                if anim::is_locked(cx.anim, *c) {
                    continue;
                }
                if let Some(v) = anim::get(cx.ekind, *c) {
                    cx.anim.entry(*c).or_default().set_key(cx.t, v);
                }
            }
        }
    }
    if r.secondary_clicked() {
        for c in chans {
            // Clearing the keys leaves the LOCK where it was: it is a
            // property of the channel, not of the curve that was on it.
            match cx.anim.get_mut(c) {
                Some(k) if k.locked => k.keys.clear(),
                _ => {
                    cx.anim.remove(c);
                }
            }
        }
    }
}

/// Row label; amber when a patch cable drives the channel (the live value
/// rides the hover — the control itself still shows the authored number).
fn row_label(ui: &mut egui::Ui, cx: &RowCtx, label: &str, chans: &[Channel]) {
    let drv = cx.driven_of(chans);
    let expr = drv.and_then(|_| crate::chanexpr::row_text(cx.node, chans));
    let mut text = egui::RichText::new(label).small();
    if expr.is_some() {
        text = text.color(theme::patch_expr());
    } else if drv.is_some() {
        text = text.color(theme::cable());
    }
    // Clickable for its menu: expressions and "copy path" (`chanexpr`).
    let r = ui.add_sized(
        [LABEL_W, 16.0],
        egui::Label::new(text).truncate().sense(egui::Sense::click()),
    );
    crate::chanexpr::label_menu(&r, cx.node, chans, cx.ekind);
    match (drv, expr) {
        (Some(v), Some(x)) => {
            r.on_hover_text(format!("= {x}\nlive value {:.*}", decimals(), v));
        }
        (Some(v), None) => {
            r.on_hover_text(format!(
                "driven by patch cable — live value {:.*}",
                decimals(),
                v
            ));
        }
        _ => {}
    }
}

fn hud_from(cx: &RowCtx, label: &str, chans: &[Channel], lo: f32, hi: f32) -> HudSlider {
    HudSlider {
        node: cx.node,
        chans: chans.to_vec(),
        label: label.to_string(),
        lo,
        hi,
        pos: egui::Vec2::ZERO,
        scale: unit_scale(),
        rgb: false,
        rgb_scale: false,
        gang: true,
        wheel: false,
    }
}

/// Ctrl-drag handling on a control: while the gesture (or an in-flight HUD
/// drag) is live, the edit is reverted EVERY frame; drag-start reports the
/// spawn. Returns (suppress_edit, spawn).
fn ctrl_drag(ui: &egui::Ui, cx: &RowCtx, r: &egui::Response) -> (bool, bool) {
    let ctrl = ui.input(|i| i.modifiers.ctrl || i.modifiers.command);
    let touching = r.drag_started() || r.dragged() || r.drag_stopped();
    let suppress = touching && (cx.hud_live || ctrl);
    let spawn = r.drag_started() && ctrl;
    (suppress, spawn)
}

/// The held z/x/c/v rate factor, if any. The table and the typing guard
/// live in [`crate::rgbslider::rate`] — the rgb lanes take the same keys,
/// and one table means one retune. The keys window capturing a binding
/// stands the keys down here too.
pub fn sens_factor(ui: &egui::Ui) -> Option<f32> {
    if crate::keymap::capturing() {
        return None;
    }
    crate::rgbslider::rate(ui)
}

/// A drag speed with the held rate key folded in — what every raw
/// `DragValue::new(..).speed(..)` in the app passes, so z/x/c/v work over
/// any draggable float, not just the standard rows.
pub fn rated(ui: &egui::Ui, base: impl Into<f64>) -> f64 {
    base.into() * sens_factor(ui).unwrap_or(1.0) as f64
}

/// A drag's frame delta with the held rate key folded in — the hook for
/// VIEWER GIZMOS (transform handles, spline points, warp grids), which all
/// derive their edit from `pointer` and `pointer - drag_delta()`. Damping
/// the delta damps the increment whatever the gesture computes from it:
/// a translate scales linearly, a rotate by angle, a scale by ratio.
///
/// The handle stops tracking the cursor exactly while a rate is held —
/// that is the same bargain a slider knob makes, and it is what "finer
/// than the screen" costs.
///
/// Only for drags that edit a VALUE. Moving a HUD, resizing one or panning
/// a canvas is furniture, and furniture goes where it is put.
pub fn rated_delta(ui: &egui::Ui, r: &egui::Response) -> egui::Vec2 {
    r.drag_delta() * sens_factor(ui).unwrap_or(1.0)
}

/// While the pointer is over a parameter row (or a rate drag is live),
/// z/x/c/v belong to the row and their single-letter HOTKEYS are dead —
/// the props-bar, curve-editor and graph-pin sites check here before
/// firing. A TIMESTAMP rather than a flag, exactly like
/// [`crate::rgbslider::grabbing`] and for the same reason: many rows write
/// it every frame, and a claim that expires cannot race a clear.
pub fn sens_grabbing(ctx: &egui::Context) -> bool {
    let now = ctx.input(|i| i.time);
    if ctx
        .data(|d| d.get_temp::<f64>(egui::Id::new("sens-grab")))
        .is_some_and(|c| now - c < 0.25)
    {
        return true;
    }
    // Any DECIDED drag also owns the keys: raw DragValues all over the
    // panels ride the rate through their speed and cannot each claim.
    // "Decidedly" is what keeps the Z/X pin CLICK alive — a click on a
    // node briefly owns a drag id, but never a decided drag.
    ctx.dragged_id().is_some() && ctx.input(|i| i.pointer.is_decidedly_dragging())
}

pub fn claim_sens(ctx: &egui::Context) {
    let now = ctx.input(|i| i.time);
    ctx.data_mut(|d| d.insert_temp(egui::Id::new("sens-grab"), now));
}

/// Relative rail drag while a rate key is down. A drag surface registered
/// OVER the rail owns the gesture, so the slider's jump-to-pointer never
/// runs: a press with the key already held lands here (added after the
/// slider, so on top), and a drag already running on the rail migrates
/// here the frame a key goes down. Either way the surface keeps the drag
/// to the end of the gesture — releasing the key mid-drag must not snap
/// the value to the pointer — and the rate is read live, so z→x→released
/// steps fine → coarser → 1:1 inside one drag.
///
/// Returns the value delta while it owns the gesture; None leaves the
/// frame to the slider's own positional editing.
fn sens_drag(
    ui: &mut egui::Ui,
    rail: &egui::Response,
    lo: f32,
    hi: f32,
    sens: Option<f32>,
) -> Option<f32> {
    let id = rail.id.with("sens");
    let stolen = ui.ctx().dragged_id() == Some(id);
    if sens.is_none() && !stolen {
        return None;
    }
    let fr = ui.interact(rail.rect, id, egui::Sense::drag());
    if !stolen && !fr.dragged() {
        if rail.dragged() {
            // Key went down mid-gesture: take the drag over, and eat the
            // jump the slider already applied this frame.
            ui.ctx().set_dragged_id(id);
            return Some(0.0);
        }
        return None;
    }
    Some(fr.drag_delta().x / rail.rect.width().max(1.0) * (hi - lo) * sens.unwrap_or(1.0))
}

/// The range a row's rail spans: the authored default until the user
/// redefines it from the shift-hover bound fields ([`range_edit`]). Kept
/// per node AND label — two Glints keep separate throws — for the session.
fn range_id(node: u64, label: &str) -> egui::Id {
    egui::Id::new(("slider-range", node, label.to_owned()))
}

fn slider_range(ui: &egui::Ui, node: u64, label: &str, lo: f32, hi: f32) -> (f32, f32) {
    ui.data(|d| d.get_temp(range_id(node, label))).unwrap_or((lo, hi))
}

/// Shift-hover a rail and its two bounds appear over its ends as drag
/// fields: drag (or click and type) to re-aim what the rail spans. The
/// VALUE is untouched — the range was soft to begin with (clamping Never),
/// this only redistributes the rail's precision. Right-click a bound to
/// go back to the built-in range.
///
/// ctrl+shift over the rail (the chord alone, no other key down) PINS the
/// editor open, so it needs no holding: it stays until Escape, a click
/// anywhere off the rail and its fields, or the chord again. The fields
/// wear the accent either way — they float over the panel and should not
/// be mistaken for the row's own value.
///
/// `rail` is the rail's rect and whether the rail itself is hovered (a
/// plain slider's own response; the rgb lanes' track block, which answers
/// to the same chord and spans one range for all three lanes).
fn range_edit(ui: &mut egui::Ui, node: u64, label: &str, (rail, rail_hovered): (egui::Rect, bool), cur: (f32, f32)) {
    let id = range_id(node, label);
    let live_id = id.with("live");
    // WHICH row opens is the rail's own hover — the same layer-aware
    // judgement that lights the knob, so exactly one row answers. (A
    // hand-rolled pointer-in-band test sat here first; its band reached up
    // over the row above, so two editors fought for every hover.) The flag
    // then keeps the editor up while the pointer is ON the bound fields,
    // or a field is being dragged or typed into (shift may well come up
    // mid-edit).
    let shift = ui.input(|i| i.modifiers.shift);
    let keep: bool = ui.data(|d| d.get_temp(live_id)).unwrap_or(false);
    // The pin: toggled on the chord's ARRIVAL over this rail, so holding it
    // does not flicker the mode every frame.
    let sticky_id = id.with("sticky");
    let prev_id = id.with("chord-prev");
    let chord = ui.input(|i| {
        (i.modifiers.ctrl || i.modifiers.command) && i.modifiers.shift && i.keys_down.is_empty()
    });
    let prev: bool = ui.data(|d| d.get_temp(prev_id)).unwrap_or(false);
    let mut sticky: bool = ui.data(|d| d.get_temp(sticky_id)).unwrap_or(false);
    // Over the rail by GEOMETRY, or with the editor already up: egui's
    // hover flickers off the rail for a frame when the bound fields pop in,
    // and the usual hand is shift first (editor opens) then ctrl — the
    // chord lands in exactly that frame, or with the pointer already up on
    // the fields.
    let over = rail_hovered || keep || ui.rect_contains_pointer(rail);
    if chord && !prev && over {
        sticky = !sticky;
    }
    ui.data_mut(|d| {
        d.insert_temp(prev_id, chord);
        d.insert_temp(sticky_id, sticky);
    });
    if !(keep || sticky || (shift && rail_hovered)) {
        return;
    }
    let (mut lo, mut hi) = cur;
    // A bound drags at the rail's own rate.
    let speed = (((hi - lo).abs() / rail.width().max(1.0)) as f64).max(1e-4);
    let gap = ui.spacing().item_spacing.x;
    let (rl, rh) = egui::Area::new(id.with("edit"))
        .order(egui::Order::Tooltip)
        // Flush against the rail's top: a gap here is a strip where the
        // pointer hovers NOTHING on its way up to a field, and the editor
        // blinks out under it.
        .fixed_pos(rail.left_top() - egui::vec2(0.0, NUM_H))
        .show(ui.ctx(), |ui| {
            // Accent: these are the rail's bounds, not the row's value.
            let v = ui.visuals_mut();
            v.override_text_color = Some(theme::accent_text());
            for w in [&mut v.widgets.inactive, &mut v.widgets.hovered, &mut v.widgets.active] {
                w.bg_fill = theme::accent();
                w.weak_bg_fill = theme::accent();
            }
            v.widgets.hovered.weak_bg_fill = theme::lift(theme::accent(), 24);
            v.widgets.hovered.bg_fill = theme::lift(theme::accent(), 24);
            if sticky {
                // Pinned: an outline says it will stay.
                v.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, theme::accent_text());
            }
            ui.horizontal(|ui| {
                let rl = ui
                    .add_sized(
                        [NUM_W, NUM_H],
                        egui::DragValue::new(&mut lo).update_while_editing(false).speed(speed).max_decimals(decimals()),
                    )
                    .on_hover_text(
                        "slider range minimum — drag or type\nright-click: built-in range",
                    );
                ui.add_space((rail.width() - 2.0 * NUM_W - 2.0 * gap).max(4.0));
                let rh = ui
                    .add_sized(
                        [NUM_W, NUM_H],
                        egui::DragValue::new(&mut hi).update_while_editing(false).speed(speed).max_decimals(decimals()),
                    )
                    .on_hover_text(
                        "slider range maximum — drag or type\nright-click: built-in range",
                    );
                (rl, rh)
            })
            .inner
        })
        .inner;
    if rl.secondary_clicked() || rh.secondary_clicked() {
        ui.data_mut(|d| d.remove::<(f32, f32)>(id));
    } else if rl.changed() || rh.changed() {
        // Bounds stay ordered; the edited one gives way.
        let eps = ((cur.1 - cur.0).abs() * 1e-3).max(1e-6);
        if rl.changed() {
            lo = lo.min(hi - eps);
        }
        if rh.changed() {
            hi = hi.max(lo + eps);
        }
        ui.data_mut(|d| d.insert_temp(id, (lo, hi)));
    }
    // Evaluated BEFORE the data_mut closure: has_focus() takes the ctx
    // lock, and egui's lock is not re-entrant — inside data_mut it
    // deadlocks the app, no panic, no message. Hovering a field sustains
    // only while shift is held; a drag or an open text edit outlives it.
    // The rail's hover counts too: on the way from rail to field the
    // pointer changes owner between frames, and without it the editor
    // blinks out mid-crossing.
    // A pinned editor lets go on Escape or a click off the rail and fields.
    if sticky {
        let off = ui.input(|i| {
            i.key_pressed(egui::Key::Escape)
                || (i.pointer.any_pressed()
                    && i.pointer
                        .interact_pos()
                        .is_some_and(|p| !(rail.contains(p) || rl.rect.contains(p) || rh.rect.contains(p))))
        });
        if off {
            ui.data_mut(|d| d.insert_temp(sticky_id, false));
        }
    }
    let live = rl.dragged()
        || rh.dragged()
        || rl.has_focus()
        || rh.has_focus()
        || (shift && (rail_hovered || rl.hovered() || rh.hovered()));
    ui.data_mut(|d| d.insert_temp(live_id, live));
}

/// One keyable row as the properties pie sees it: where it was drawn,
/// whose it is, what it is called and which channels it drives.
#[derive(Clone, Debug)]
pub struct RowHit {
    pub rect: egui::Rect,
    pub node: u64,
    pub label: String,
    pub chans: Vec<Channel>,
}

fn rows_id() -> egui::Id {
    egui::Id::new("widgets-row-hits")
}

/// Forget last frame's rows. The properties bin calls it before drawing.
pub fn clear_rows(ctx: &egui::Context) {
    ctx.data_mut(|d| d.insert_temp(rows_id(), Vec::<RowHit>::new()));
}

/// Every keyable row registered so far (the self-tests aim at these).
pub fn rows(ctx: &egui::Context) -> Vec<RowHit> {
    ctx.data(|d| d.get_temp::<Vec<RowHit>>(rows_id())).unwrap_or_default()
}

/// The keyable row under `p`, if any, as drawn this frame so far (or last
/// frame, when read before the bin redraws).
pub fn row_at(ctx: &egui::Context, p: egui::Pos2) -> Option<RowHit> {
    ctx.data(|d| {
        d.get_temp::<Vec<RowHit>>(rows_id())
            .and_then(|v| v.into_iter().rev().find(|r| r.rect.contains(p)))
    })
}

/// A background slot under a row, filled in `end_row` once the row's rect
/// is known: the wash has to sit UNDER the widgets it tints.
fn begin_row(ui: &egui::Ui) -> egui::layers::ShapeIdx {
    ui.painter().add(egui::Shape::Noop)
}

/// Close a keyable row: a LOCKED row (any of its channels) wears a faint
/// red wash, so a lock reads across the whole row and not only in the
/// dot and the bar under the field; and the row registers itself for the
/// properties pie.
fn end_row(
    ui: &egui::Ui,
    bg: egui::layers::ShapeIdx,
    rect: egui::Rect,
    cx: &RowCtx,
    label: &str,
    chans: &[Channel],
) {
    if chans.is_empty() {
        return;
    }
    if anim::any_locked(cx.anim, chans) {
        ui.painter().set(bg, egui::Shape::rect_filled(rect.expand(2.0), 3.0, theme::row_locked()));
    }
    let hit = RowHit { rect, node: cx.node, label: label.to_string(), chans: chans.to_vec() };
    ui.ctx().data_mut(|d| d.get_temp_mut_or_default::<Vec<RowHit>>(rows_id()).push(hit));
    // The expression editor, when it is open on this row.
    crate::chanexpr::editor_ui(ui, cx.node, chans, rect);
}

/// Bounded scalar: label + slider + key dot. Returns a HUD spawn when the
/// row is ctrl-dragged toward the viewer.
pub fn slider(
    ui: &mut egui::Ui,
    cx: &mut RowCtx,
    label: &str,
    v: &mut f32,
    lo: f32,
    hi: f32,
    chans: &[Channel],
) -> Option<HudSlider> {
    let mut out = None;
    let bg = begin_row(ui);
    let row = ui.horizontal(|ui| {
        row_label(ui, cx, label, chans);
        out = slider_body(ui, cx, label, v, lo, hi, chans).0;
    });
    end_row(ui, bg, row.response.rect, cx, label, chans);
    out
}

/// [`slider`] without its label, in `width`: one CELL of a table whose rows
/// and columns are labelled around it (the Tinter's vector grid). Everything
/// else is the row's — rail, value field, key dot, keys, locks, patch
/// cables, the ctrl-drag HUD (named `label`), the rate keys and the
/// shift-hover range editor — so a cell is exactly as capable as a row.
///
/// `rail` colours the rail by value: the rail IS a picture of what the
/// control does (a hue sweep, grey to colour), painted where egui's plain
/// rail would be, with egui's handle on top.
pub fn slider_cell(
    ui: &mut egui::Ui,
    cx: &mut RowCtx,
    label: &str,
    v: &mut f32,
    (lo, hi): (f32, f32),
    chans: &[Channel],
    width: f32,
    rail: &dyn Fn(f32) -> egui::Color32,
) -> Option<HudSlider> {
    let bg = begin_row(ui);
    let under = ui.painter().add(egui::Shape::Noop);
    let layout = egui::Layout::left_to_right(egui::Align::Center);
    let (hud, rect) = ui
        .allocate_ui_with_layout(egui::vec2(width, NUM_H), layout, |ui| {
            ui.set_width(width);
            // egui's own rail goes clear; the picture below stands in.
            ui.visuals_mut().widgets.inactive.bg_fill = egui::Color32::TRANSPARENT;
            ui.visuals_mut().slider_trailing_fill = false;
            slider_body(ui, cx, label, v, lo, hi, chans)
        })
        .inner;
    end_row(ui, bg, rect, cx, label, chans);
    // The same geometry egui's slider uses: the rail spans the rect, the
    // handle travels it inset by its radius (height / 2.5).
    let (lo, hi) = slider_range(ui, cx.node, label, lo, hi);
    let h = ui.spacing().slider_rail_height.max(8.0);
    let band = egui::Rect::from_center_size(rect.center(), egui::vec2(rect.width(), h));
    let inset = rect.height() / 2.5;
    let n = 32;
    let mut mesh = egui::Mesh::default();
    for k in 0..=n {
        let x = band.left() + band.width() * k as f32 / n as f32;
        let t = ((x - rect.left() - inset) / (rect.width() - 2.0 * inset).max(1.0)).clamp(0.0, 1.0);
        let c = rail(lo + (hi - lo) * t);
        mesh.colored_vertex(egui::pos2(x, band.top()), c);
        mesh.colored_vertex(egui::pos2(x, band.bottom()), c);
        if k > 0 {
            let i = 2 * k as u32;
            mesh.add_triangle(i - 2, i - 1, i);
            mesh.add_triangle(i - 1, i + 1, i);
        }
    }
    ui.painter().set(under, egui::Shape::mesh(mesh));
    hud
}

/// Everything [`slider`] draws after its label: rail, value field, key dot.
/// Hands back the HUD spawn and the rail's rect.
fn slider_body(
    ui: &mut egui::Ui,
    cx: &mut RowCtx,
    label: &str,
    v: &mut f32,
    lo: f32,
    hi: f32,
    chans: &[Channel],
) -> (Option<HudSlider>, egui::Rect) {
    let mut out = None;
    // Reserve EXACTLY what follows the rail: value field, key dot and
    // the gaps between. The old hardcoded 66 was short of the real
    // width, so every row overflowed its panel — and egui widens a
    // ui's max_rect to include an overflowing child, so each row then
    // laid out one step wider than the row above it: the staircase
    // the panels used to show.
    let gap = ui.spacing().item_spacing.x;
    ui.spacing_mut().slider_width =
        (ui.available_width() - (NUM_W + DOT_W + 2.0 * gap)).max(40.0);
    // The authored range is only the default: shift-hover the rail and
    // its bounds appear as drag fields (range_edit, below).
    let (lo, hi) = slider_range(ui, cx.node, label, lo, hi);
    // A driven channel shows its LIVE value, read-only — the cable
    // overrides any edit every frame, so the widget says so.
    if let Some(mut dv) = cx.driven_of(chans) {
        // Driven by a typed EXPRESSION: the rail shows the text and both
        // halves open it for editing (`chanexpr`).
        if let Some(x) = crate::chanexpr::row_text(cx.node, chans) {
            let w = ui.spacing().slider_width;
            let dr = crate::chanexpr::rail(ui, w, cx.node, chans, &x);
            crate::chanexpr::value_field(ui, NUM_W, cx.node, chans, dv, &x);
            key_dot(ui, cx, chans);
            return (None, dr.rect);
        }
        let dr = ui
            .add_enabled(
                false,
                egui::Slider::new(&mut dv, lo..=hi)
                    .clamping(egui::SliderClamping::Never)
                    .max_decimals(decimals())
                    .show_value(false),
            )
            .on_disabled_hover_text("driven by patch cable (live value)");
        ui.add_enabled_ui(false, |ui| {
            num_field(ui, egui::DragValue::new(&mut dv).update_while_editing(false).max_decimals(decimals()),
            )
        })
        .inner
        .on_disabled_hover_text("driven by patch cable (live value)");
        key_dot(ui, cx, chans);
        return (None, dr.rect);
    }
    // Same as `drags`: an animated row shows its curve's value. And the
    // same lock rule: no keys, and read-only once a curve is on it.
    let locked = anim::any_locked(cx.anim, chans);
    let frozen = locked
        && chans.iter().any(|c| cx.anim.get(c).is_some_and(|k| !k.keys.is_empty()));
    if frozen {
        let mut shown = shown_value(cx, chans, *v);
        let r = ui
            .add_enabled(
                false,
                egui::Slider::new(&mut shown, lo..=hi)
                    .clamping(egui::SliderClamping::Never)
                    .max_decimals(decimals())
                    .show_value(false),
            )
            .on_disabled_hover_text(LOCKED_TIP);
        lock_marker(ui, r.rect, true);
        lock_menu(cx, &r, chans);
        let vr = ui
            .add_enabled_ui(false, |ui| {
                num_field(ui, egui::DragValue::new(&mut shown).update_while_editing(false).max_decimals(decimals()),
                )
            })
            .inner
            .on_disabled_hover_text(LOCKED_TIP);
        lock_menu(cx, &vr, chans);
        key_dot(ui, cx, chans);
        return (None, r.rect);
    }
    let mut shown = shown_value(cx, chans, *v);
    let pre = shown;
    let r = ui.add(
        egui::Slider::new(&mut shown, lo..=hi)
            .clamping(egui::SliderClamping::Never)
            .max_decimals(decimals())
            .show_value(false),
    );
    let rail = r.rect;
    let sens = sens_factor(ui);
    // The value field is our own, NUM_W like every float field in the
    // app — the slider's built-in one is auto-width, the one field
    // that could break the column. Its drag speed is the rail's
    // per-pixel rate, riding the same z/x/c/v factor.
    let vr = num_field(ui, egui::DragValue::new(&mut shown).update_while_editing(false)
            .speed(((hi - lo) / rail.width().max(1.0) * sens.unwrap_or(1.0)) as f64)
            .max_decimals(decimals()),
    );
    lock_marker(ui, vr.rect, locked);
    lock_menu(cx, &r, chans);
    lock_menu(cx, &vr, chans);
    let (s1, spawn1) = ctrl_drag(ui, cx, &r);
    let (s2, spawn2) = ctrl_drag(ui, cx, &vr);
    let fine = sens_drag(ui, &r, lo, hi, sens);
    // The ctrl gesture spawns a HUD, it doesn't edit.
    if !(s1 || s2) {
        if let Some(d) = fine {
            if d != 0.0 {
                *v = pre + d;
            }
        } else if shown != pre {
            *v = shown;
        }
    }
    if spawn1 || spawn2 {
        out = Some(hud_from(cx, label, chans, lo, hi));
    }
    // Typed into: `../node/param` opens the expression editor, `10/2` is 5.
    if let Some(k) = crate::chanexpr::field_typing(ui, &vr, cx.node, chans) {
        *v = k;
    }
    key_dot(ui, cx, chans);
    // z/x/c/v belong to this row while the pointer is over it.
    let over = ui
        .input(|i| i.pointer.hover_pos())
        .is_some_and(|p| rail.union(vr.rect).contains(p));
    if over || fine.is_some() {
        claim_sens(ui.ctx());
    }
    range_edit(ui, cx.node, label, (r.rect, r.hovered()), (lo, hi));
    (out, rail)
}


/// The space that puts `n` value fields (and then the key dot) flush with
/// the right edge — exactly where `slider` puts its one value field.
fn value_column_pad(ui: &egui::Ui, n: usize) -> f32 {
    let gap = ui.spacing().item_spacing.x;
    // `add_space` advances without an item gap; each field is followed by one.
    (ui.available_width() - DOT_W - n as f32 * (NUM_W + gap)).max(0.0)
}

/// A parameter row for a value that is NOT a channel (a brush setting, a
/// tracker's search radius, a layout knob): the same geometry as `slider`
/// — `LABEL_W` label, rail, `NUM_W` field, an empty `DOT_W` slot — so it
/// lines up with the keyable rows around it. No keys, no HUD, no patch.
/// `integer` rounds the value (an octave count, an input count).
pub fn plain_slider(
    ui: &mut egui::Ui,
    label: &str,
    v: &mut f32,
    lo: f32,
    hi: f32,
    integer: bool,
) -> egui::Response {
    ui.horizontal(|ui| {
        ui.add_sized([LABEL_W, 16.0], egui::Label::new(egui::RichText::new(label).small()).truncate());
        let gap = ui.spacing().item_spacing.x;
        ui.spacing_mut().slider_width =
            (ui.available_width() - (NUM_W + DOT_W + 2.0 * gap)).max(40.0);
        let mut s = egui::Slider::new(v, lo..=hi).show_value(false);
        if integer {
            s = s.integer();
        }
        let r = ui.add(s);
        let dec = if integer { 0 } else { decimals() };
        let mut f = egui::DragValue::new(v).update_while_editing(false)
            .speed(((hi - lo) / r.rect.width().max(1.0)) as f64)
            .range(lo..=hi)
            .max_decimals(dec);
        if integer {
            f = f.fixed_decimals(0);
        }
        let vr = num_field(ui, f);
        ui.add_space(DOT_W);
        r.union(vr)
    })
    .inner
}

/// A bare bounded slider for chrome outside the parameter rows (the viewer
/// bar): the rail takes whatever width is left after its value field and
/// `trailing` (whatever the caller adds after it), and z/x/c/v rate the
/// drag exactly as on a parameter row. Returns true when `v` changed.
pub fn bar_slider(ui: &mut egui::Ui, v: &mut f32, lo: f32, hi: f32, trailing: f32) -> bool {
    let gap = ui.spacing().item_spacing.x;
    ui.spacing_mut().slider_width =
        (ui.available_width() - (NUM_W + gap + trailing)).max(40.0);
    let pre = *v;
    let mut shown = pre;
    let r = ui.add(
        egui::Slider::new(&mut shown, lo..=hi)
            .clamping(egui::SliderClamping::Never)
            .max_decimals(decimals())
            .show_value(false),
    );
    let rail = r.rect;
    let sens = sens_factor(ui);
    let vr = num_field(ui, egui::DragValue::new(&mut shown).update_while_editing(false)
            .speed(((hi - lo) / rail.width().max(1.0) * sens.unwrap_or(1.0)) as f64)
            .max_decimals(decimals()),
    );
    let fine = sens_drag(ui, &r, lo, hi, sens);
    if let Some(d) = fine {
        if d != 0.0 {
            *v = pre + d;
        }
    } else if shown != pre {
        *v = shown;
    }
    let over = ui
        .input(|i| i.pointer.hover_pos())
        .is_some_and(|p| rail.union(vr.rect).contains(p));
    if over || fine.is_some() {
        claim_sens(ui.ctx());
    }
    *v != pre
}

/// Unbounded scalars: label + N drag-values + key dot. `hud` gives the
/// range a spawned HUD slider uses.
pub fn drags(
    ui: &mut egui::Ui,
    cx: &mut RowCtx,
    label: &str,
    vs: &mut [&mut f32],
    speed: f32,
    hud: (f32, f32),
    chans: &[Channel],
) -> Option<HudSlider> {
    drags_impl(ui, cx, label, vs, speed, hud, chans, None)
}

/// [`drags`] for values that go together (scale x/y, a filmback): a
/// padlock after the first field, and while it is closed an edit to one
/// moves the others by the same factor (`imfram::widgets::ratiolock`).
/// The followers are written like any edit, so on an animated channel the
/// post-pass keys them too. `locked` is the caller's: a param when the
/// node saves it (a Transform's `scale_gang`), else [`ratio_lock_state`].
#[allow(clippy::too_many_arguments)]
pub fn drags_locked(
    ui: &mut egui::Ui,
    cx: &mut RowCtx,
    label: &str,
    vs: &mut [&mut f32],
    speed: f32,
    hud: (f32, f32),
    chans: &[Channel],
    locked: &mut bool,
) -> Option<HudSlider> {
    drags_impl(ui, cx, label, vs, speed, hud, chans, Some(locked))
}

/// A ratio lock's state when the node does not save one: UI memory, per
/// row, off until clicked.
pub fn ratio_lock_state(ui: &egui::Ui, id: impl std::hash::Hash + std::fmt::Debug) -> (egui::Id, bool) {
    let id = egui::Id::new(("ratio-lock", id));
    (id, ui.data(|d| d.get_temp::<bool>(id)).unwrap_or(false))
}

/// [`drags_locked`] with its state in UI memory, keyed by node and label.
#[allow(clippy::too_many_arguments)]
pub fn drags_linked(
    ui: &mut egui::Ui,
    cx: &mut RowCtx,
    label: &str,
    vs: &mut [&mut f32],
    speed: f32,
    hud: (f32, f32),
    chans: &[Channel],
) -> Option<HudSlider> {
    let (id, mut locked) = ratio_lock_state(ui, (cx.node, label));
    let out = drags_impl(ui, cx, label, vs, speed, hud, chans, Some(&mut locked));
    ui.data_mut(|d| d.insert_temp(id, locked));
    out
}

#[allow(clippy::too_many_arguments)]
fn drags_impl(
    ui: &mut egui::Ui,
    cx: &mut RowCtx,
    label: &str,
    vs: &mut [&mut f32],
    speed: f32,
    hud: (f32, f32),
    chans: &[Channel],
    mut lock: Option<&mut bool>,
) -> Option<HudSlider> {
    let mut out = None;
    let nvs = vs.len();
    // What each field showed before this pass, and whether it is free to
    // follow a locked edit (not cabled, not frozen by a channel lock).
    let mut shown_all = vec![0.0f32; nvs];
    let mut free = vec![false; nvs];
    let mut now = vec![0.0f32; nvs];
    let bg = begin_row(ui);
    let row = ui.horizontal(|ui| {
        row_label(ui, cx, label, chans);
        // Fields sit RIGHT-JUSTIFIED, in the same column as a slider row's
        // value field: a panel mixing `drags` and `slider` rows reads as one
        // table, number under number, whatever each row's control is.
        let lock_w = if lock.is_some() {
            imfram::widgets::ratiolock::W + ui.spacing().item_spacing.x
        } else {
            0.0
        };
        ui.add_space((value_column_pad(ui, nvs) - lock_w).max(0.0));
        // The z/x/c/v drag-rate keys work here too, scaling the field's
        // own speed; the claim keeps their hotkeys quiet over the row.
        let sens = sens_factor(ui);
        let mut zone = egui::Rect::NOTHING;
        for (i, v) in vs.iter_mut().enumerate() {
            let auth0 = **v;
            if i == 1 {
                if let Some(l) = lock.as_deref_mut() {
                    imfram::widgets::ratiolock::button(ui, l);
                }
            }
            // Per-element driven check: translate.x can be cabled while
            // translate.y stays hand-editable. A ganged control (one value,
            // several channels) is driven if any of them is.
            let dch: &[Channel] = if nvs == chans.len() { &chans[i..=i] } else { chans };
            if let Some(x) = cx
                .driven_of(dch)
                .and_then(|dv| Some((dv, crate::chanexpr::row_text(cx.node, dch)?)))
            {
                let r = crate::chanexpr::value_field(ui, NUM_W, cx.node, dch, x.0, &x.1);
                zone = zone.union(r.rect);
                continue;
            }
            if let Some(mut dv) = cx.driven_of(dch) {
                let r = ui
                    .add_enabled_ui(false, |ui| {
                        num_field(ui, egui::DragValue::new(&mut dv).update_while_editing(false)
                                .speed(speed)
                                .max_decimals(decimals()),
                        )
                    })
                    .inner
                    .on_disabled_hover_text("driven by patch cable (live value)");
                zone = zone.union(r.rect);
                continue;
            }
            // An animated channel's AUTHORED number is a dead value: the
            // curve drives the render, and the post-pass turns any edit into
            // a key and puts authored back. Showing authored therefore makes
            // the row look frozen while dragging, and starts each drag from
            // the wrong base — drag +2 at a frame the curve reads 40 and it
            // would land on 2. Show what the curve says, write only on a
            // real edit, and let the post-pass key it.
            let mut shown = shown_value(cx, dch, **v);
            let pre = shown;
            shown_all[i] = pre;
            // LOCKED: no keys, ever. With keys already on the channel the
            // curve keeps driving, so the field is read-only too — editing
            // the authored number under a curve does nothing visible and
            // pretending otherwise is worse than saying so.
            let locked = anim::any_locked(cx.anim, dch);
            let frozen = locked
                && dch.iter().any(|c| cx.anim.get(c).is_some_and(|k| !k.keys.is_empty()));
            let r = if frozen {
                ui.add_enabled_ui(false, |ui| {
                    num_field(ui, egui::DragValue::new(&mut shown).update_while_editing(false).max_decimals(decimals()),
                    )
                })
                .inner
                .on_disabled_hover_text(LOCKED_TIP)
            } else {
                num_field(ui, egui::DragValue::new(&mut shown).update_while_editing(false)
                        .speed(speed * sens.unwrap_or(1.0))
                        .max_decimals(decimals()),
                )
            };
            lock_marker(ui, r.rect, locked);
            lock_menu(cx, &r, dch);
            if locked {
                zone = zone.union(r.rect);
                if !frozen && shown != pre {
                    **v = shown; // static and locked: still a number to set
                }
                continue;
            }
            zone = zone.union(r.rect);
            free[i] = true;
            let (suppress, spawn) = ctrl_drag(ui, cx, &r);
            if !suppress && shown != pre {
                **v = shown;
            }
            if let Some(k) = crate::chanexpr::field_typing(ui, &r, cx.node, dch) {
                **v = k;
            }
            if spawn {
                out = Some(hud_from(cx, label, chans, hud.0, hud.1));
            }
            // The field's value now: the edit if there was one, else what
            // it showed (authored is not that on an animated channel).
            now[i] = if **v != auth0 { **v } else { pre };
        }
        let over = ui
            .input(|i| i.pointer.hover_pos())
            .is_some_and(|p| zone.contains(p));
        if over {
            claim_sens(ui.ctx());
        }
        key_dot(ui, cx, chans);
    });
    end_row(ui, bg, row.response.rect, cx, label, chans);
    if lock.is_some_and(|l| *l) {
        // Only an edit to a free field leads, and only free fields follow.
        let before: Vec<f64> = shown_all.iter().map(|&x| x as f64).collect();
        let mut after: Vec<f64> = (0..nvs)
            .map(|i| if free[i] { now[i] as f64 } else { before[i] })
            .collect();
        if let Some(lead) = imfram::widgets::ratiolock::keep_ratio(&before, &mut after) {
            for j in (0..nvs).filter(|&j| j != lead && free[j]) {
                *vs[j] = after[j] as f32;
            }
        }
    }
    out
}

/// An rgb triple: label + the three-lane slider + one key dot for all
/// three. The gesture (gang, and the S/D/F colour axes) lives in
/// [`crate::rgbslider`]; this is the standard row around it.
pub fn rgb_row(
    ui: &mut egui::Ui,
    cx: &mut RowCtx,
    label: &str,
    v: &mut [f32; 3],
    gang: &mut bool,
    spec: crate::rgbslider::RgbSpec,
    chans: &[Channel; 3],
) -> Option<HudSlider> {
    let mut out = None;
    // The spec's range is only the default, as on a plain row: shift-hover
    // the lanes and the bounds appear as drag fields (`range_edit`), one
    // range for all three lanes.
    let (lo, hi) = slider_range(ui, cx.node, label, spec.lo, spec.hi);
    let spec = crate::rgbslider::RgbSpec { lo, hi, ..spec };
    let bg = begin_row(ui);
    let row = ui.horizontal(|ui| {
        row_label(ui, cx, label, chans);
        // Any cabled lane makes the whole widget read-only: a gang moves
        // three channels at once, and two of them being authored while the
        // third is overridden every frame is a value nobody can predict.
        let driven = chans.iter().any(|c| cx.driven_of(&[*c]).is_some()).then(|| {
            let mut d = *v;
            for (i, c) in chans.iter().enumerate() {
                if let Some(dv) = cx.driven_of(&[*c]) {
                    d[i] = dv;
                }
            }
            d
        });
        // The lanes take what is left AFTER the key dot. Handed the whole
        // row they pushed the dot past the edge, and egui then widened the
        // container to cover it — so every later row grew by a dot, and a
        // second column ran off the tile.
        let lanes_w = (ui.available_width() - DOT_W - ui.spacing().item_spacing.x)
            .max(crate::rgbslider::CHROME_W + crate::rgbslider::MIN_TRACK);
        let lanes = ui.allocate_ui_with_layout(
            egui::vec2(lanes_w, 0.0),
            egui::Layout::top_down(egui::Align::Min),
            |ui| {
            ui.set_width(lanes_w);
            let r = crate::rgbslider::show(
                ui,
                (cx.node, label),
                v,
                gang,
                spec,
                driven,
                cx.hud_live,
            );
            if r.ctrl_drag {
                // The HUD it spawns is the same instrument: ganged, with the
                // colour axes on the same keys.
                out = Some(HudSlider {
                    rgb: true,
                    rgb_scale: spec.mode == crate::rgbslider::RgbMode::Scale,
                    gang: *gang,
                    ..hud_from(cx, label, chans, spec.lo, spec.hi)
                });
            }
            (r.dragging, r.track)
        },
        );
        let (dragging, track) = lanes.inner;
        if driven.is_none() {
            range_edit(ui, cx.node, label, (track, ui.rect_contains_pointer(track)), (lo, hi));
        }
        // z/x/c/v are drag-rate keys on the lanes too: claim them while the
        // pointer is over the widget, so a key pressed BEFORE the grab does
        // not fire its hotkey (mid-drag the rgbslider grab already stands
        // the whole block down).
        let over = ui
            .input(|i| i.pointer.hover_pos())
            .is_some_and(|p| lanes.response.rect.contains(p));
        if over || dragging {
            claim_sens(ui.ctx());
        }
        // E over the lanes: sample the viewer into the triple. A cabled
        // row is read-only, so it neither arms nor takes a sample.
        let target = egui::Id::new(("eyedrop-rgb", cx.node, label));
        let drop = crate::eyedrop::hook(ui, target, over && driven.is_none());
        if let Some(c) = drop.sample.filter(|_| driven.is_none()) {
            *v = [c[0], c[1], c[2]];
        }
        if drop.armed {
            ui.painter().rect_stroke(
                lanes.response.rect.expand(2.0),
                3.0,
                egui::Stroke::new(1.5, ui.visuals().selection.stroke.color),
                egui::StrokeKind::Outside,
            );
        }
        key_dot(ui, cx, chans);
    });
    end_row(ui, bg, row.response.rect, cx, label, chans);
    out
}

/// ---- Patch-bay geometry -------------------------------------------------
/// Panels are puzzle pieces: whole-tile rectangles (1x1, 2x1, 2x2 ... never
/// 1.5x1) on a cell grid, and pieces never overlap.

pub fn tiles_overlap(a: [i32; 2], asp: [u8; 2], b: [i32; 2], bsp: [u8; 2]) -> bool {
    a[0] < b[0] + bsp[0] as i32
        && b[0] < a[0] + asp[0] as i32
        && a[1] < b[1] + bsp[1] as i32
        && b[1] < a[1] + asp[1] as i32
}

pub fn any_overlap(cell: [i32; 2], span: [u8; 2], others: &[([i32; 2], [u8; 2])]) -> bool {
    others.iter().any(|(c, s)| tiles_overlap(cell, span, *c, *s))
}

/// First cell (row-major scan, `cols` wide) where `span` fits cleanly.
pub fn first_free_cell(span: [u8; 2], others: &[([i32; 2], [u8; 2])], cols: i32) -> [i32; 2] {
    let cols = cols.max(span[0] as i32);
    for row in 0..1024 {
        for col in 0..=(cols - span[0] as i32) {
            let c = [col, row];
            if !any_overlap(c, span, others) {
                return c;
            }
        }
    }
    [0, 0]
}

#[cfg(test)]
mod patch_tests {
    use super::*;

    #[test]
    fn overlap_edges_are_exclusive() {
        // Side-by-side 1x1s touch but don't overlap.
        assert!(!tiles_overlap([0, 0], [1, 1], [1, 0], [1, 1]));
        assert!(!tiles_overlap([0, 0], [1, 1], [0, 1], [1, 1]));
        assert!(tiles_overlap([0, 0], [1, 1], [0, 0], [1, 1]));
    }

    #[test]
    fn overlap_multi_tile() {
        // A 2x2 at origin collides with anything inside it...
        assert!(tiles_overlap([0, 0], [2, 2], [1, 1], [1, 1]));
        assert!(tiles_overlap([0, 0], [2, 2], [1, 0], [2, 1]));
        // ...but not with a piece seated just past its edge.
        assert!(!tiles_overlap([0, 0], [2, 2], [2, 0], [1, 3]));
        assert!(!tiles_overlap([0, 0], [2, 2], [0, 2], [3, 1]));
    }

    #[test]
    fn first_free_skips_occupied() {
        let others = [([0, 0], [2, 1]), ([2, 0], [1, 1])];
        // Row 0 is full up to col 3 in a 3-wide grid: a 1x1 lands on row 1.
        assert_eq!(first_free_cell([1, 1], &others, 3), [0, 1]);
        // A 2x1 in a 4-wide grid still fits at the end of row 0.
        assert_eq!(first_free_cell([2, 1], &others, 5), [3, 0]);
    }

    #[test]
    fn first_free_fits_tall_pieces() {
        let others = [([0, 0], [1, 2])];
        // A 2x2 can't share cols 0 rows 0-1; next fit is col 1.
        assert_eq!(first_free_cell([2, 2], &others, 4), [1, 0]);
    }
}

#[cfg(test)]
mod tab_tests {
    use super::*;
    use engine::graph::{GradeParams, GradeZones};

    /// A plain panel is one page; a Grade is a master page, a page per zone
    /// and the ranges, in that order, with ids that name the zone rather
    /// than its position.
    #[test]
    fn a_grade_declares_master_zones_and_ranges() {
        assert!(panel_tabs(&NodeKind::Blur(Default::default())).is_empty());
        let plain = panel_tabs(&NodeKind::Grade(GradeParams::default()));
        assert_eq!(
            plain.iter().map(|t| t.id.as_str()).collect::<Vec<_>>(),
            ["master", "ranges"]
        );
        let zoned = panel_tabs(&NodeKind::Grade(GradeParams {
            zones: Some(GradeZones::default()),
            ..Default::default()
        }));
        assert_eq!(
            zoned.iter().map(|t| t.id.as_str()).collect::<Vec<_>>(),
            ["master", "zone:0", "zone:1", "zone:2", "ranges"]
        );
        assert_eq!(
            zoned.iter().map(|t| t.label.as_str()).collect::<Vec<_>>(),
            ["master", "shadows", "midtones", "highlights", "ranges"]
        );
    }

    /// The selection is an ID: adding tabs ahead of it leaves it where it
    /// was, and a dead id falls back to the first tab rather than nowhere.
    #[test]
    fn the_selection_survives_tabs_being_added_ahead_of_it() {
        let mut sel = "ranges".to_string();
        let before = panel_tabs(&NodeKind::Grade(GradeParams::default()));
        let after = panel_tabs(&NodeKind::Grade(GradeParams {
            zones: Some(GradeZones::default()),
            ..Default::default()
        }));
        let mut width = 0.0;
        let _ = crate::snapshot::shoot(600.0, 100.0, |ui| {
            ui.horizontal(|ui| {
                tab_strip(ui, &before, &mut sel);
                assert_eq!(sel, "ranges");
                tab_strip(ui, &after, &mut sel);
                assert_eq!(sel, "ranges", "adding zones bounced the selection");
                let mut dead = "zone:9".to_string();
                tab_strip(ui, &after, &mut dead);
                assert_eq!(dead, "master");
                width = tab_strip_width(ui, &after);
            });
        });
        // Five small tabs fit in a three-tile title row with room for the
        // name and the buttons; the width is what the title row decides by.
        assert!(width > 100.0 && width < 400.0, "strip width {width}");
    }

    /// Drive a button through press and release across three frames, with
    /// the row's drag handle registered either after the button (the old
    /// title row) or before it (`drag_row`). Returns whether the click
    /// landed.
    fn click_lands(handle_first: bool) -> bool {
        let ctx = egui::Context::default();
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(400.0, 100.0),
            )),
            ..Default::default()
        };
        let clicked = std::cell::Cell::new(false);
        let at: std::cell::Cell<Option<egui::Pos2>> = std::cell::Cell::new(None);
        let frame = |events: Vec<egui::Event>| {
            let mut input = input.clone();
            input.events = events;
            let _ = ctx.run_ui(input, |ui| {
                let build = |ui: &mut egui::Ui| {
                    let r = ui.button("ranges");
                    at.set(Some(r.rect.center()));
                    if r.clicked() {
                        clicked.set(true);
                    }
                };
                if handle_first {
                    let _ = drag_row(ui, egui::Id::new("row"), build);
                } else {
                    let row = ui.horizontal(build);
                    let _ = ui.interact(row.response.rect, egui::Id::new("row"), egui::Sense::drag());
                }
            });
        };
        frame(Vec::new());
        let p = at.get().expect("button laid out");
        let button = |pressed: bool| egui::Event::PointerButton {
            pos: p,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: Default::default(),
        };
        frame(vec![egui::Event::PointerMoved(p), button(true)]);
        frame(vec![button(false)]);
        clicked.get()
    }

    /// egui gives a click to the TOP-MOST widget under the pointer, and a
    /// drag-only rect registered after a button is on top of it: the panel
    /// header's reorder handle, registered after the row, ate every click on
    /// the tabs. `drag_row` registers it first, and the click lands.
    #[test]
    fn buttons_in_a_drag_row_still_take_the_click() {
        assert!(
            !click_lands(false),
            "egui now lets a click through a later drag rect; drag_row's ordering is moot"
        );
        assert!(click_lands(true), "a button under the row's drag handle lost its click");
    }

    /// The strip draws a selected tab differently from the rest — a fill
    /// under the live one — so which page you are on can be read at a glance.
    #[test]
    fn the_selected_tab_is_painted_and_the_others_are_not() {
        let tabs = vec![PanelTab::new("a", "master"), PanelTab::new("b", "ranges")];
        let mut sel = "b".to_string();
        let shot = crate::snapshot::shoot(300.0, 40.0, |ui| {
            ui.horizontal(|ui| {
                tab_strip(ui, &tabs, &mut sel);
            });
        });
        shot.save("tab-strip");
        // Scan the row for a run of non-background pixels: the selected
        // tab's fill. Text is not rasterised, so an unselected tab leaves
        // the background alone.
        let bg = shot.at(290.0, 20.0);
        let painted = (0..300).filter(|x| shot.at(*x as f32, 14.0) != bg).count();
        assert!(painted > 20, "no tab fill drawn ({painted} px differ)");
        assert!(painted < 200, "both tabs painted alike ({painted} px differ)");
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::anim::{AnimCurve, Channel};
    use engine::graph::NodeKind;

    #[test]
    fn animated_rows_show_the_curve_not_the_authored_value() {
        // A TimeRemap parked on frame 0 in its authored data, but keyed to
        // read frame 40 at the tick on screen.
        let mut anim: BTreeMap<Channel, AnimCurve> = BTreeMap::new();
        let c = anim.entry(Channel::TimeRemapFrame).or_default();
        c.set_key(0, 0.0);
        c.set_key(10, 80.0);
        let mut ekind = NodeKind::TimeRemap(engine::graph::TimeRemapParams::default());
        crate::anim::apply(&mut ekind, &anim, 5);
        let cx = RowCtx {
            node: 1,
            anim: &mut anim.clone(),
            ekind: &ekind,
            t: 5,
            hud_live: false,
            driven: Vec::new(),
        };
        // The row must show what the curve says at frame 5 (40), never the
        // authored 0 — dragging from 0 would jump the retime.
        assert_eq!(shown_value(&cx, &[Channel::TimeRemapFrame], 0.0), 40.0);
    }

    #[test]
    fn unanimated_rows_show_what_is_authored() {
        let mut anim: BTreeMap<Channel, AnimCurve> = BTreeMap::new();
        let ekind = NodeKind::TimeRemap(engine::graph::TimeRemapParams::default());
        let cx = RowCtx {
            node: 1,
            anim: &mut anim,
            ekind: &ekind,
            t: 5,
            hud_live: false,
            driven: Vec::new(),
        };
        assert_eq!(shown_value(&cx, &[Channel::TimeRemapFrame], 7.5), 7.5);
        // An empty curve counts as unanimated, not as "evaluates to 0".
        let mut anim2: BTreeMap<Channel, AnimCurve> = BTreeMap::new();
        anim2.insert(Channel::TimeRemapFrame, AnimCurve::default());
        let cx2 = RowCtx {
            node: 1,
            anim: &mut anim2,
            ekind: &ekind,
            t: 5,
            hud_live: false,
            driven: Vec::new(),
        };
        assert_eq!(shown_value(&cx2, &[Channel::TimeRemapFrame], 7.5), 7.5);
    }

    /// Rows must form COLUMNS. The old layout reserved a hardcoded 66 for
    /// what follows the rail — less than those widgets actually take — so
    /// every slider row overflowed its panel, and egui widens a ui's
    /// max_rect to include an overflowing child, so each row then laid out
    /// one step wider than the row above: the staircase the panels used to
    /// show. A long label ("odd ray length") shoved its rail right on top
    /// of that. Eight rows in a fixed-width column must end flush instead.
    #[test]
    fn slider_rows_stay_inside_their_panel() {
        let ctx = egui::Context::default();
        let ekind = NodeKind::Blur(Default::default());
        let mut anim: BTreeMap<Channel, AnimCurve> = BTreeMap::new();
        let mut vals = [1.0f32; 8];
        let labels = [
            "threshold",
            "knee",
            "angle",
            "length",
            "odd ray length",
            "falloff",
            "chroma",
            "intensity",
        ];
        const W: f32 = 480.0;
        // Two frames: the overflow also fed back through the panel's
        // remembered size, growing it frame over frame.
        for frame in 0..2 {
            let mut used = 0.0f32;
            let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
                ui.scope(|ui| {
                    ui.set_max_width(W);
                    let mut cx = RowCtx {
                        node: 7,
                        anim: &mut anim,
                        ekind: &ekind,
                        t: 0,
                        hud_live: false,
                        driven: Vec::new(),
                    };
                    for (l, v) in labels.iter().zip(vals.iter_mut()) {
                        slider(ui, &mut cx, l, v, 0.0, 8.0, &[Channel::BlurSize]);
                    }
                    used = ui.min_rect().width();
                });
            });
            assert!(
                used <= W + 0.5,
                "frame {frame}: slider rows take {used} of a {W} column"
            );
        }
    }

    /// One shift-hover frame over a rail must COMPLETE. The range editor's
    /// bookkeeping once read Response::has_focus() inside a data_mut
    /// closure, and egui's context lock is not re-entrant — the app froze
    /// the moment the bound fields appeared, no panic, no message. The
    /// live flag is written at the very end of that path, so its presence
    /// is the proof the frame ran out. (Pre-fix this test HANGS rather
    /// than fails; a test stuck on this name is this bug.)
    #[test]
    fn shift_hover_range_editor_completes_the_frame() {
        let ctx = egui::Context::default();
        let ekind = NodeKind::Blur(Default::default());
        let mut anim: BTreeMap<Channel, AnimCurve> = BTreeMap::new();
        let mut val = 1.0f32;
        let frame = |shift: bool| {
            let mut input = egui::RawInput::default();
            input.events.push(egui::Event::ModifiersChanged(egui::Modifiers { shift, ..Default::default() }));
            input.events.push(egui::Event::PointerMoved(egui::pos2(200.0, 10.0)));
            input
        };
        for shift in [false, true] {
            let _ = ctx.run_ui(frame(shift), |ui| {
                ui.scope(|ui| {
                    ui.set_max_width(480.0);
                    let mut cx = RowCtx {
                        node: 7,
                        anim: &mut anim,
                        ekind: &ekind,
                        t: 0,
                        hud_live: false,
                        driven: Vec::new(),
                    };
                    slider(ui, &mut cx, "threshold", &mut val, 0.0, 8.0, &[
                        Channel::BlurSize,
                    ]);
                });
            });
        }
        assert!(
            ctx.data(|d| d.get_temp::<bool>(range_id(7, "threshold").with("live")))
                .is_some(),
            "the shift-hover range editor never ran to the end of its frame"
        );
    }

    /// ctrl+shift on the rail PINS the range editor: it is still up after
    /// both keys come off, and Escape puts it away.
    #[test]
    fn ctrl_shift_pins_the_range_editor_and_escape_unpins_it() {
        let ctx = egui::Context::default();
        let ekind = NodeKind::Blur(Default::default());
        let mut anim: BTreeMap<Channel, AnimCurve> = BTreeMap::new();
        let mut val = 1.0f32;
        let mut run = |mods: egui::Modifiers, esc: bool| {
            let mut input = egui::RawInput::default();
            input.events.push(egui::Event::ModifiersChanged(mods));
            input.events.push(egui::Event::PointerMoved(egui::pos2(200.0, 10.0)));
            if esc {
                input.events.push(egui::Event::Key {
                    key: egui::Key::Escape,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: mods,
                });
            }
            let _ = ctx.run_ui(input, |ui| {
                ui.scope(|ui| {
                    ui.set_max_width(480.0);
                    let mut cx = RowCtx { node: 7, anim: &mut anim, ekind: &ekind, t: 0, hud_live: false, driven: Vec::new() };
                    slider(ui, &mut cx, "threshold", &mut val, 0.0, 8.0, &[Channel::BlurSize]);
                });
            });
        };
        let chord = egui::Modifiers { ctrl: true, shift: true, ..Default::default() };
        let none = egui::Modifiers::default();
        run(none, false);
        run(chord, false);
        run(chord, false); // held: toggles once, on arrival
        run(none, false);
        run(none, false);
        let sticky = || ctx.data(|d| d.get_temp::<bool>(range_id(7, "threshold").with("sticky")));
        assert_eq!(sticky(), Some(true), "the chord pinned the editor, and it outlived the keys");
        run(none, true);
        assert_eq!(sticky(), Some(false), "escape unpins it");
    }
}

/// What a generator's "inherit" resolves to right now, for the format row's
/// label: set once per frame by the properties bin (the plates' canvas),
/// read by every generator panel so none of them needs it threaded in.
static INHERIT_CANVAS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new((1920u64 << 32) | 1080);

pub fn set_inherit_canvas((w, h): (u32, u32)) {
    INHERIT_CANVAS.store(((w as u64) << 32) | h as u64, std::sync::atomic::Ordering::Relaxed);
}

pub fn inherit_canvas() -> (u32, u32) {
    let v = INHERIT_CANVAS.load(std::sync::atomic::Ordering::Relaxed);
    ((v >> 32) as u32, v as u32)
}

/// A GENERATOR's format row: roto's `format_row` (inherit / named presets /
/// custom) over the declared `size` plus an `inherit` flag — the engine
/// params keep a size either way, so switching to inherit and back
/// restores the size you had.
pub fn gen_format_row(
    ui: &mut egui::Ui,
    id: impl std::hash::Hash + std::fmt::Debug,
    size: &mut (u32, u32),
    inherit: &mut bool,
) {
    let mut f = (!*inherit).then_some(*size);
    format_row(ui, id, &mut f, inherit_canvas());
    *inherit = f.is_none();
    if let Some(s) = f {
        *size = s;
    }
}

/// Formats a format row offers by name. Plain literals: these are the
/// sizes, not a colour or a style.
pub const FORMAT_PRESETS: &[(&str, (u32, u32))] = &[
    ("HD", (1920, 1080)),
    ("UHD", (3840, 2160)),
    ("2K", (2048, 1080)),
    ("4K", (4096, 2160)),
    ("square 1K", (1024, 1024)),
];

/// A node's format: inherit (`None` — `inherit` is what that resolves to,
/// shown so the choice is informed), a preset, or custom width × height.
/// Choosing custom starts from the size in force, so nothing jumps.
pub fn format_row(
    ui: &mut egui::Ui,
    id: impl std::hash::Hash + std::fmt::Debug,
    format: &mut Option<(u32, u32)>,
    inherit: (u32, u32),
) {
    let lock_id = egui::Id::new(("format-size", &id));
    ui.horizontal(|ui| {
        ui.add_sized([LABEL_W, 16.0], egui::Label::new(egui::RichText::new("format").small()));
        let name = |f: Option<(u32, u32)>| match f {
            None => format!("inherit {}×{}", inherit.0, inherit.1),
            Some(s) => FORMAT_PRESETS
                .iter()
                .find(|(_, p)| *p == s)
                .map(|(n, _)| (*n).to_string())
                .unwrap_or_else(|| "custom".into()),
        };
        egui::ComboBox::from_id_salt(id)
            .selected_text(name(*format))
            .show_ui(ui, |ui| {
                ui.selectable_value(format, None, name(None));
                for (n, s) in FORMAT_PRESETS {
                    ui.selectable_value(format, Some(*s), format!("{n}  {}×{}", s.0, s.1));
                }
                let custom = format.is_some_and(|s| !FORMAT_PRESETS.iter().any(|(_, p)| *p == s));
                if ui.selectable_label(custom, "custom").clicked() && !custom {
                    *format = Some(format.unwrap_or(inherit));
                }
            });
        if let Some(s) = format {
            size_fields(ui, lock_id, s);
        }
    });
}

/// Width × height fields with a ratio lock between them (16..=16384,
/// the format range). Locked, a new width brings the height along and
/// vice versa. The lock is UI state, per `id`.
pub fn size_fields(ui: &mut egui::Ui, id: egui::Id, size: &mut (u32, u32)) {
    let (lid, mut locked) = ratio_lock_state(ui, id);
    let before = [size.0, size.1];
    num_field(ui, egui::DragValue::new(&mut size.0).update_while_editing(false).speed(rated(ui, 4)).range(16..=16384));
    imfram::widgets::ratiolock::button(ui, &mut locked);
    num_field(ui, egui::DragValue::new(&mut size.1).update_while_editing(false).speed(rated(ui, 4)).range(16..=16384));
    if locked {
        let mut after = [size.0, size.1];
        if imfram::widgets::ratiolock::keep_ratio_u32(&before, &mut after, 16).is_some() {
            *size = (after[0].min(16384), after[1].min(16384));
        }
    }
    ui.data_mut(|d| d.insert_temp(lid, locked));
}

#[cfg(test)]
mod column_tests {
    use super::*;

    /// A `drags` row's fields end where a `slider` row's value field ends:
    /// the rows of a panel read as one table (the Spline's feather /
    /// opacity / expand / falloff).
    #[test]
    fn drags_fields_share_the_slider_value_column() {
        let ctx = egui::Context::default();
        let mut anim = BTreeMap::new();
        let kind = NodeKind::Dot { mask: false, channels: false, data: false, text: false, vector: false, scene: false };
        let mut ends = (0.0f32, 0.0f32);
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            ui.set_width(400.0);
            let mut cx = RowCtx { node: 1, anim: &mut anim, ekind: &kind, t: 0, hud_live: false, driven: Vec::new() };
            let (mut a, mut b) = (0.5f32, 0.0f32);
            ends.0 = ui.scope(|ui| slider(ui, &mut cx, "opacity", &mut a, 0.0, 1.0, &[])).response.rect.max.x;
            ends.1 = ui.scope(|ui| drags(ui, &mut cx, "expand", &mut [&mut b], 0.1, (0.0, 1.0), &[])).response.rect.max.x;
        });
        assert!((ends.0 - ends.1).abs() < 0.5, "slider row ends at {}, drags row at {}", ends.0, ends.1);
    }

    /// The ratio lock's padlock comes out of the pad, not off the end: a
    /// locked pair still ends in the value column.
    #[test]
    fn a_locked_pair_keeps_the_value_column() {
        let ctx = egui::Context::default();
        let mut anim = BTreeMap::new();
        let kind = NodeKind::Dot { mask: false, channels: false, data: false, text: false, vector: false, scene: false };
        let mut ends = (0.0f32, 0.0f32);
        imfram::snapshot::discard(ctx.run_ui(egui::RawInput::default(), |ui| {
            ui.set_width(400.0);
            let mut cx = RowCtx { node: 1, anim: &mut anim, ekind: &kind, t: 0, hud_live: false, driven: Vec::new() };
            let (mut a, mut b, mut c, mut d) = (1.0f32, 2.0f32, 1.0f32, 2.0f32);
            let mut locked = true;
            ends.0 = ui.scope(|ui| drags(ui, &mut cx, "plain", &mut [&mut a, &mut b], 0.1, (0.0, 1.0), &[])).response.rect.max.x;
            ends.1 = ui
                .scope(|ui| drags_locked(ui, &mut cx, "locked", &mut [&mut c, &mut d], 0.1, (0.0, 1.0), &[], &mut locked))
                .response
                .rect
                .max
                .x;
        }));
        assert!((ends.0 - ends.1).abs() < 0.5, "plain pair ends at {}, locked at {}", ends.0, ends.1);
    }
}

#[cfg(test)]
mod lock_row_tests {
    use super::*;

    fn washed(locked: bool) -> (bool, Option<RowHit>) {
        let ctx = egui::Context::default();
        let mut anim = BTreeMap::new();
        if locked {
            anim::set_locked(&mut anim, Channel::SplineOpacity, true);
        }
        let kind = NodeKind::Dot { mask: false, channels: false, data: false, text: false, vector: false, scene: false };
        let mut centre = egui::Pos2::ZERO;
        let out = ctx.run_ui(egui::RawInput::default(), |ui| {
            clear_rows(ui.ctx());
            ui.set_width(400.0);
            let mut cx = RowCtx { node: 7, anim: &mut anim, ekind: &kind, t: 0, hud_live: false, driven: Vec::new() };
            let mut v = 0.5f32;
            centre = ui
                .scope(|ui| slider(ui, &mut cx, "opacity", &mut v, 0.0, 1.0, &[Channel::SplineOpacity]))
                .response
                .rect
                .center();
        });
        let wash = theme::row_locked();
        let hit = out.shapes.iter().any(|c| {
            matches!(&c.shape, egui::Shape::Rect(r) if r.fill == wash)
        });
        (hit, row_at(&ctx, centre))
    }

    /// A locked row goes red across its whole width, and every keyable row
    /// registers itself so the properties pie can find what is under the
    /// pointer.
    #[test]
    fn a_locked_row_wears_the_wash_and_every_row_registers() {
        let (wash, row) = washed(true);
        assert!(wash, "a locked row paints the red wash");
        let row = row.expect("the row registered");
        assert_eq!((row.node, row.label.as_str(), row.chans.clone()), (7, "opacity", vec![Channel::SplineOpacity]));
        let (wash, row) = washed(false);
        assert!(!wash, "an unlocked row paints no wash");
        assert!(row.is_some());
    }
}
