//! The mock layouts: stills of Wobble's screens, painted from the live theme
//! with every element tagged by the slot it wears. Not interactive as Wobble
//! is — clicking anything just picks its colour.

use crate::canvas::{wire_points, Canvas};
use crate::glyphs::{self, Icon, NODES};
use crate::theme::{self, Slot as S};
use egui::{Align2, Color32, Mesh, Pos2, Shape};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Scene {
    Workspace,
    Nodes,
    Curves,
    Edit,
    Windows,
}

impl Scene {
    pub const ALL: [Scene; 5] = [Scene::Workspace, Scene::Nodes, Scene::Curves, Scene::Edit, Scene::Windows];
    pub fn label(self) -> &'static str {
        match self {
            Scene::Workspace => "workspace",
            Scene::Nodes => "nodes",
            Scene::Curves => "curves + wheels",
            Scene::Edit => "edit timeline",
            Scene::Windows => "windows + keys",
        }
    }
    pub fn paint(self, c: &mut Canvas) {
        match self {
            Scene::Workspace => workspace(c),
            Scene::Nodes => nodes(c),
            Scene::Curves => curves(c),
            Scene::Edit => edit(c),
            Scene::Windows => windows(c),
        }
    }
}

fn a(slot: S, alpha: u8) -> Color32 {
    theme::get(slot).gamma_multiply_u8(alpha)
}

// ---------------------------------------------------------------------------
// shared bits

/// A graph node's radius on the workspace mock, and how far its ports sit
/// from its centre.
const NODE_R: f32 = 13.0;
const PORT: f32 = NODE_R + 9.0;

/// A graph node centred on (x, y), top-to-bottom flow: input port above,
/// output below, name to the right.
#[allow(clippy::too_many_arguments)]
fn node(c: &mut Canvas, x: f32, y: f32, icon: Icon, slot: S, name: &str, selected: bool, ports: (bool, bool)) {
    if ports.0 {
        c.line(&[(x, y - NODE_R), (x, y - PORT)], 1.5, S::port_in, "input port");
        c.dot(x, y - PORT, 3.5, S::port_in, "input port");
    }
    if ports.1 {
        c.line(&[(x, y + NODE_R), (x, y + PORT)], 1.5, S::port_out, "output port");
        c.dot(x, y + PORT, 3.5, S::port_out, "output port");
    }
    let e = c.node(x, y, NODE_R, icon, slot, selected, "node");
    c.text(e.max.x + 8.0, y - 7.5, name, 12.0, S::node_label, "node label");
}

fn button(c: &mut Canvas, x: f32, y: f32, w: f32, label: &str, slot: S, fg: S) {
    c.fill(x, y, w, 20.0, 3.0, slot, "button");
    c.text_a(x + w * 0.5, y + 10.0, Align2::CENTER_CENTER, label, 11.5, theme::get(fg), fg, "button label");
}

fn field(c: &mut Canvas, x: f32, y: f32, w: f32, value: &str) {
    c.fill(x, y, w, 20.0, 3.0, S::field, "field");
    c.text_a(x + w - 6.0, y + 10.0, Align2::RIGHT_CENTER, value, 11.5, theme::get(S::text), S::text, "field value");
}

fn grid_dots(c: &mut Canvas, x0: f32, y0: f32, x1: f32, y1: f32, step: f32, slot: S) {
    let mut y = y0 + step * 0.5;
    while y < y1 {
        let mut x = x0 + step * 0.5;
        while x < x1 {
            c.dot(x, y, 1.3, slot, "grid dots");
            x += step;
        }
        y += step;
    }
}

/// Something painted over a plate: a gradient standing in for the image.
fn plate(c: &mut Canvas, x: f32, y: f32, w: f32, h: f32) {
    let r = c.r(x, y, w, h);
    let mut m = Mesh::default();
    let cols = [
        Color32::from_rgb(38, 52, 70),
        Color32::from_rgb(120, 96, 70),
        Color32::from_rgb(24, 30, 38),
        Color32::from_rgb(70, 60, 52),
    ];
    m.colored_vertex(r.left_top(), cols[0]);
    m.colored_vertex(r.right_top(), cols[1]);
    m.colored_vertex(r.left_bottom(), cols[2]);
    m.colored_vertex(r.right_bottom(), cols[3]);
    m.add_triangle(0, 1, 2);
    m.add_triangle(1, 3, 2);
    c.plain(Shape::mesh(m));
    // A sun and a horizon so it reads as a picture.
    c.plain(Shape::circle_filled(c.p(x + w * 0.68, y + h * 0.36), h * 0.09 * c.s, Color32::from_rgb(230, 190, 130)));
    c.plain(Shape::rect_filled(c.r(x, y + h * 0.62, w, h * 0.38), 0.0, Color32::from_rgb(28, 32, 30)));
}

fn menubar(c: &mut Canvas) {
    c.fill(0.0, 0.0, 1200.0, 28.0, 0.0, S::bg, "menu bar");
    let mut x = 12.0;
    for m in ["file", "edit", "view", "prefs…", "globals…"] {
        let r = c.text(x, 7.0, m, 12.5, S::text, "menu item");
        x += r.width() / c.s + 18.0;
    }
    button(c, x + 8.0, 4.0, 86.0, "cache viewer", S::button, S::text);
    c.fill(x + 102.0, 4.0, 46.0, 20.0, 3.0, S::accent, "toggled button");
    c.text_a(x + 125.0, 14.0, Align2::CENTER_CENTER, "rgb", 11.5, theme::get(S::accent_text), S::accent_text, "toggled label");
    c.outline(x + 156.0, 4.0, 40.0, 20.0, 3.0, 1.2, S::highlight, "on / pinned state");
    c.text_a(x + 176.0, 14.0, Align2::CENTER_CENTER, "hud", 11.5, theme::get(S::highlight), S::highlight, "on / pinned state");
    c.text_a(1188.0, 14.0, Align2::RIGHT_CENTER, "24.0 fps", 11.5, theme::get(S::ok), S::ok, "ok status");
    c.text_a(1110.0, 14.0, Align2::RIGHT_CENTER, "NC", 11.5, theme::get(S::cost_nc), S::cost_nc, "non-commercial badge");
    c.line(&[(0.0, 28.0), (1200.0, 28.0)], 1.0, S::border, "separator");
}

// ---------------------------------------------------------------------------
// workspace: graph | viewer over properties, transport underneath

fn workspace(c: &mut Canvas) {
    c.fill(0.0, 0.0, 1200.0, 760.0, 0.0, S::bg, "panels");
    menubar(c);

    // --- node graph
    c.fill(0.0, 29.0, 560.0, 571.0, 0.0, S::graph_bg, "graph background");
    grid_dots(c, 0.0, 29.0, 560.0, 600.0, 24.0, S::graph_grid);

    // wires first, nodes over them
    c.wire((122.0, 83.0 + PORT), (122.0, 203.0 - PORT), 4.0, S::wire_channels, "multi-channel cable");
    c.wire((342.0, 83.0 + PORT), (364.0, 213.0 - PORT), 2.0, S::wire, "wire");
    c.wire((342.0, 83.0 + PORT), (380.0, 213.0 - PORT), 2.0, S::wire_data, "data wire");
    c.wire((272.0, 203.0 + PORT), (122.0 - PORT, 203.0), 2.0, S::wire_mask, "mask wire");
    c.wire((122.0, 203.0 + PORT), (196.0, 333.0 - PORT), 2.0, S::wire, "wire");
    c.wire((372.0, 213.0 + PORT), (372.0, 313.0 - PORT), 2.0, S::wire, "wire");
    c.wire((372.0, 313.0 + PORT), (208.0, 333.0 - PORT), 2.0, S::wire, "wire");
    c.wire((202.0, 333.0 + PORT), (202.0, 453.0 - PORT), 2.0, S::wire, "wire");
    // expression cable: a value link up into the Transform, under its name
    let cable: Vec<(f32, f32)> = (0..=20)
        .map(|i| {
            let t = i as f32 / 20.0;
            let u = 1.0 - t;
            (u * u * 462.0 + 2.0 * u * t * 452.0 + t * t * 382.0, u * u * 460.0 + 2.0 * u * t * 350.0 + t * t * 326.0)
        })
        .collect();
    c.line(&cable, 1.5, S::cable, "expression cable");

    node(c, 122.0, 83.0, Icon::Read, S::node_read, "Load  plate_bg", false, (false, true));
    node(c, 342.0, 83.0, Icon::Read, S::node_read, "Load  char_fg", false, (false, true));
    node(c, 122.0, 203.0, Icon::Grade, S::node_grade, "Grade", true, (true, true));
    c.line(&[(122.0 - NODE_R, 203.0), (122.0 - PORT, 203.0)], 1.5, S::wire_mask, "mask input");
    c.dot(122.0 - PORT, 203.0, 3.5, S::wire_mask, "mask input");
    c.dot(138.0, 187.0, 3.0, S::pinned, "pinned pip");
    node(c, 272.0, 203.0, Icon::Spline, S::node_spline, "Spline", false, (false, true));
    node(c, 372.0, 213.0, Icon::Defocus, S::node_defocus, "Defocus", false, (true, true));
    node(c, 372.0, 313.0, Icon::Transform, S::node_transform, "Transform", false, (true, true));
    node(c, 202.0, 333.0, Icon::Blend, S::node_blend, "Blend  over", false, (true, true));
    node(c, 202.0, 453.0, Icon::Write, S::node_write, "Render", false, (true, false));
    node(c, 462.0, 473.0, Icon::SubwobControl, S::node_tool_control, "slider", false, (false, false));
    // bypassed node: struck through
    node(c, 82.0, 453.0, Icon::Blur, S::node_blur, "Blur", false, (true, true));
    c.line(&[(60.0, 474.0), (104.0, 432.0)], 2.0, S::cut, "bypass strike");
    // a wire being dragged off the Write
    let pts = wire_points((202.0, 453.0 + NODE_R), (300.0, 540.0));
    c.line(&pts, 1.5, S::wire_dragged, "wire being dragged");
    c.dot(300.0, 540.0, 3.5, S::wire_dragged, "wire being dragged");
    // a speck of a node, scaled down
    c.dot(500.0, 120.0, 2.0, S::family_filter, "tiny node");
    c.dot_c(500.0, 120.0, 5.0, a(S::node_twinkle, 80), S::node_twinkle, "tiny-node twinkle");
    c.dot_c(500.0, 120.0, 1.2, theme::get(S::node_twinkle), S::node_twinkle, "tiny-node twinkle");

    // --- viewer
    c.fill(564.0, 29.0, 636.0, 361.0, 0.0, S::viewer_bg, "viewer backdrop");
    plate(c, 610.0, 60.0, 544.0, 300.0);
    c.outline(610.0, 60.0, 544.0, 300.0, 0.0, 1.0, S::viewer_frame, "comp frame");
    c.text(574.0, 36.0, "A", 12.0, S::viewer_live, "live pane label");
    c.text(592.0, 36.0, "B", 12.0, S::text_weak, "faint text");
    // transform gizmo
    c.outline(760.0, 120.0, 220.0, 150.0, 0.0, 1.0, S::selection_box, "selection box");
    c.line(&[(870.0, 120.0), (870.0, 92.0)], 1.0, S::selection_box, "rotate stalk");
    for (hx, hy) in [(760.0, 120.0), (980.0, 120.0), (760.0, 270.0), (980.0, 270.0), (870.0, 270.0), (760.0, 195.0), (980.0, 195.0)] {
        c.fill(hx - 5.0, hy - 5.0, 10.0, 10.0, 1.0, S::handle_rim, "handle rim");
        c.fill(hx - 3.5, hy - 3.5, 7.0, 7.0, 1.0, S::handle, "handle");
    }
    c.dot(870.0, 92.0, 6.0, S::handle_rim, "handle rim");
    c.dot(870.0, 92.0, 4.5, S::handle_hot, "hovered handle");
    c.dot(870.0, 195.0, 3.0, S::handle, "handle");
    // readout
    c.fill_px(c.r(580.0, 356.0, 250.0, 24.0).translate(egui::vec2(3.0, 3.0)), 5.0, theme::get(S::shadow), S::shadow, "drop shadow");
    c.fill_px(c.r(580.0, 356.0, 250.0, 24.0), 5.0, a(S::hud_bg, 160), S::hud_bg, "HUD backdrop");
    c.mono(590.0, 368.0, Align2::LEFT_CENTER, "4096×2160  f1012  0.84 0.61 0.40", 11.0, S::text_strong, "bright text");

    // --- properties
    c.fill(564.0, 394.0, 636.0, 206.0, 0.0, S::bg, "properties panel");
    c.line(&[(564.0, 392.0), (1200.0, 392.0)], 1.0, S::border, "separator");
    // tile 1: Grade
    c.fill(574.0, 402.0, 300.0, 190.0, 5.0, S::window, "tile");
    c.outline(574.0, 402.0, 300.0, 190.0, 5.0, 1.0, S::border, "tile outline");
    c.text(584.0, 408.0, "Grade", 13.0, S::text_strong, "tile title");
    let rows = [("lift", "0.012"), ("gamma", "1.000"), ("gain", "1.180"), ("offset", "0.000")];
    for (i, (l, val)) in rows.iter().enumerate() {
        let y = 432.0 + i as f32 * 26.0;
        if i == 1 {
            c.fill(578.0, y - 3.0, 292.0, 26.0, 3.0, S::hover_fill, "hovered row");
        }
        if i == 3 {
            c.fill(578.0, y - 3.0, 292.0, 26.0, 3.0, S::raised_fill, "raised fill");
        }
        c.text(588.0, y + 3.0, l, 12.0, S::text, "label");
        field(c, 660.0, y, 86.0, val);
        // slider rail + handle
        c.fill(756.0, y + 8.0, 90.0, 4.0, 2.0, S::button, "slider rail");
        c.dot(756.0 + 20.0 + i as f32 * 18.0, y + 10.0, 6.0, if i == 2 { S::button_hover } else { S::button }, if i == 2 { "hovered slider handle" } else { "slider handle" });
        if i == 0 {
            c.diamond(858.0, y + 10.0, 5.0, S::keyframe, "keyframe");
        }
        if i == 2 {
            c.fill(660.0, y + 18.0, 86.0, 2.0, 0.0, S::animated, "animated channel");
        }
        if i == 3 {
            c.fill(660.0, y + 18.0, 86.0, 2.0, 0.0, S::locked, "locked channel");
        }
    }
    c.text(588.0, 540.0, "drag to change · ⌘-drag fine", 11.0, S::text_weak, "faint text");
    c.text(588.0, 560.0, "docs ↗", 11.5, S::link, "link");
    button(c, 700.0, 558.0, 60.0, "reset", S::button, S::text);
    button(c, 766.0, 558.0, 60.0, "pressed", S::button_active, S::text);
    // tile 2: Blur, checkbox + selectable
    c.fill(884.0, 402.0, 306.0, 190.0, 5.0, S::window, "tile");
    c.outline(884.0, 402.0, 306.0, 190.0, 5.0, 1.0, S::border, "tile outline");
    c.text(894.0, 408.0, "Transform", 13.0, S::text_strong, "tile title");
    c.fill(896.0, 436.0, 16.0, 16.0, 3.0, S::button, "checkbox");
    c.text_a(904.0, 444.0, Align2::CENTER_CENTER, "✔", 11.0, theme::get(S::text_hot), S::text_hot, "highlighted text");
    c.text(920.0, 437.0, "motion blur", 12.0, S::text, "label");
    c.fill(896.0, 462.0, 80.0, 20.0, 3.0, S::accent, "selected item");
    c.text_a(936.0, 472.0, Align2::CENTER_CENTER, "bilinear", 11.5, theme::get(S::accent_text), S::accent_text, "text on accent");
    c.text_a(1016.0, 472.0, Align2::CENTER_CENTER, "cubic", 11.5, theme::get(S::text), S::text, "label");
    c.text_a(1066.0, 472.0, Align2::CENTER_CENTER, "lanczos", 11.5, theme::get(S::text), S::text, "label");
    c.outline(896.0, 492.0, 180.0, 22.0, 3.0, 1.0, S::border_hover, "hovered outline");
    c.fill(897.0, 493.0, 178.0, 20.0, 3.0, S::field, "field");
    c.text(904.0, 496.0, "centre 2048, 1080", 11.5, S::text, "field value");
    c.text_a(900.0, 530.0, Align2::LEFT_CENTER, "⚠ scale is extreme", 11.5, theme::get(S::warn), S::warn, "warning");
    c.text_a(900.0, 550.0, Align2::LEFT_CENTER, "✖ missing frames 1040-1046", 11.5, theme::get(S::error), S::error, "error");
    c.text_a(900.0, 570.0, Align2::LEFT_CENTER, "ⓘ plays direct", 11.5, theme::get(S::info), S::info, "info");

    // --- transport
    c.fill(0.0, 604.0, 1200.0, 156.0, 0.0, S::bg, "transport panel");
    c.line(&[(0.0, 602.0), (1200.0, 602.0)], 1.0, S::border, "separator");
    c.fill(20.0, 620.0, 1160.0, 40.0, 2.0, S::well, "track well");
    c.fill(40.0, 624.0, 1120.0, 32.0, 0.0, S::range_out, "track");
    c.fill(180.0, 624.0, 840.0, 32.0, 0.0, S::range_in, "play range");
    c.fill(180.0, 657.0, 520.0, 3.0, 0.0, S::cache_ram, "cached in RAM");
    c.fill(700.0, 657.0, 160.0, 3.0, 0.0, S::cache_nvme, "cached on NVMe");
    for k in [260.0, 420.0, 610.0, 860.0] {
        c.diamond(k, 640.0, 5.0, S::keyframe, "keyframe");
    }
    c.line(&[(512.0, 616.0), (512.0, 664.0)], 1.5, S::playhead, "playhead");
    c.outline(20.0, 620.0, 1160.0, 40.0, 2.0, 1.0, S::border, "border");
    for (i, f) in [1001, 1025, 1049, 1073, 1097, 1121].iter().enumerate() {
        c.text(40.0 + i as f32 * 224.0, 666.0, &f.to_string(), 10.5, S::text_weak, "frame numbers");
    }
    let mut x = 20.0;
    for b in ["|◀", "◀", "▶", "▶|", "loop in/out"] {
        let w = if b.len() > 4 { 84.0 } else { 34.0 };
        button(c, x, 690.0, w, b, S::button, S::text);
        x += w + 6.0;
    }
    field(c, x + 10.0, 690.0, 60.0, "1012");
    c.text(x + 80.0, 693.0, "/ 1121   24 fps", 12.0, S::text, "label");
}

// ---------------------------------------------------------------------------
// nodes: every node Wobble draws, family by family, on the graph's ground

fn nodes(c: &mut Canvas) {
    c.fill(0.0, 0.0, 1200.0, 760.0, 0.0, S::graph_bg, "graph background");
    grid_dots(c, 0.0, 0.0, 1200.0, 760.0, 24.0, S::graph_grid);
    // Four columns; `None` gathers the nodes that follow no family.
    let columns: [&[Option<S>]; 4] = [
        &[Some(S::family_source), Some(S::family_filter), Some(S::family_track)],
        &[Some(S::family_matte), Some(S::family_channels)],
        &[Some(S::family_colour), Some(S::family_transform), Some(S::family_merge)],
        &[Some(S::family_time), Some(S::family_other), None],
    ];
    let r = 13.0;
    for (col, fams) in columns.iter().enumerate() {
        let x = 40.0 + col as f32 * 292.0;
        let mut y = 34.0;
        for &fam in fams.iter() {
            match fam {
                Some(f) => {
                    c.fill(x, y + 1.0, 30.0, 14.0, 7.0, f, "family colour");
                    c.text(x + 40.0, y, crate::family_name(f), 13.0, S::text_strong, "bright text");
                }
                None => {
                    c.text(x, y, "borrowed colours", 13.0, S::text_strong, "bright text");
                }
            }
            y += 34.0;
            for n in NODES.iter().filter(|n| crate::family_of(n.slot) == fam) {
                // Left edges line up, however wide the node.
                let left = glyphs::extent(n.icon, Pos2::ZERO, r / glyphs::R).min.x;
                let e = c.node(x - left, y + r, r, n.icon, n.slot, false, "node");
                c.text(e.max.x + 8.0, y + r - 7.5, n.name, 12.0, S::node_label, "node label");
                y += 32.0;
            }
            y += 22.0;
        }
    }
}

// ---------------------------------------------------------------------------
// curve editor + grade wheels + keyer

fn curves(c: &mut Canvas) {
    c.fill(0.0, 0.0, 1200.0, 760.0, 0.0, S::bg, "panels");
    menubar(c);

    // channel list
    let chans: [(&str, S); 6] = [
        ("transform.x", S::text),
        ("transform.y", S::accent_text),
        ("rotate", S::text),
        ("grade.gain", S::text),
        ("blur.size", S::locked),
        ("mix", S::text_weak),
    ];
    for (i, (n, slot)) in chans.iter().enumerate() {
        let y = 50.0 + i as f32 * 26.0;
        if i == 1 {
            c.fill(12.0, y - 4.0, 196.0, 24.0, 3.0, S::accent, "selected channel");
        }
        if i == 3 {
            c.fill(12.0, y - 4.0, 196.0, 24.0, 3.0, S::hover_fill, "hovered row");
        }
        c.dot(26.0, y + 8.0, 4.0, if i == 4 { S::locked } else { S::animated }, if i == 4 { "locked channel" } else { "animated" });
        c.text(38.0, y + 1.0, n, 12.0, *slot, if i == 4 { "locked channel" } else { "channel name" });
    }
    c.outline(12.0, 202.0, 60.0, 20.0, 3.0, 1.2, S::highlight, "isolate (on state)");
    c.text_a(42.0, 212.0, Align2::CENTER_CENTER, "isolate", 11.0, theme::get(S::highlight), S::highlight, "isolate (on state)");
    button(c, 80.0, 202.0, 40.0, "all", S::button, S::text);
    button(c, 126.0, 202.0, 40.0, "none", S::button, S::text);

    // the curve well
    let (x0, y0, w, h) = (220.0, 40.0, 960.0, 420.0);
    c.fill(x0, y0, w, h, 3.0, S::well, "curve editor well");
    for i in 1..12 {
        let x = x0 + i as f32 * w / 12.0;
        c.line(&[(x, y0), (x, y0 + h)], 1.0, S::grid_line, "grid lines");
    }
    for i in 1..6 {
        let y = y0 + i as f32 * h / 6.0;
        c.line(&[(x0, y), (x0 + w, y)], 1.0, if i == 3 { S::axis_line } else { S::grid_line }, if i == 3 { "zero line" } else { "grid lines" });
    }
    for i in 0..12 {
        c.text(x0 + 4.0 + i as f32 * w / 12.0, y0 + h - 16.0, &(1001 + i * 10).to_string(), 10.0, S::text_weak, "frame numbers");
    }
    let curve = |t: f32, ph: f32, amp: f32, mid: f32| -> (f32, f32) {
        (x0 + t * w, y0 + mid + (t * 7.0 + ph).sin() * amp)
    };
    let c1: Vec<_> = (0..=120).map(|i| curve(i as f32 / 120.0, 0.0, 110.0, 200.0)).collect();
    c.line(&c1, 2.0, S::curve_scalar, "animation curve");
    let c2: Vec<_> = (0..=120).map(|i| curve(i as f32 / 120.0, 1.7, 60.0, 250.0)).collect();
    c.line(&c2, 1.5, S::curve_line, "parameter curve");
    let c3: Vec<_> = (0..=120).map(|i| curve(i as f32 / 120.0, 3.0, 30.0, 330.0)).collect();
    c.line(&c3, 1.5, S::locked, "locked curve");
    // keys on curve 1
    for (i, t) in [0.05f32, 0.27, 0.49, 0.72, 0.94].iter().enumerate() {
        let (kx, ky) = curve(*t, 0.0, 110.0, 200.0);
        let (slot, what) = match i {
            1 | 2 => (S::curve_selected, "selected key"),
            3 => (S::handle_hot, "hovered handle"),
            _ => (S::handle, "key / handle"),
        };
        c.dot(kx, ky, 6.0, S::handle_rim, "handle rim");
        c.dot(kx, ky, 4.5, slot, what);
    }
    // group box around the selected keys
    let (ax, ay) = curve(0.27, 0.0, 110.0, 200.0);
    let (bx, by) = curve(0.49, 0.0, 110.0, 200.0);
    let (gx, gy, gw, gh) = (ax - 14.0, ay.min(by) - 14.0, bx - ax + 28.0, (ay - by).abs() + 28.0);
    c.outline(gx, gy, gw, gh, 0.0, 1.0, S::selection_box, "group box");
    for (hx, hy) in [(gx, gy), (gx + gw, gy), (gx, gy + gh), (gx + gw, gy + gh)] {
        c.fill(hx - 4.0, hy - 4.0, 8.0, 8.0, 1.0, S::handle, "box handle");
    }
    // a marquee in progress
    c.fill(860.0, 250.0, 160.0, 110.0, 0.0, S::marquee_fill, "box select fill");
    c.outline(860.0, 250.0, 160.0, 110.0, 0.0, 1.0, S::marquee, "box select");
    // playhead
    c.line(&[(640.0, y0), (640.0, y0 + h)], 1.5, S::timeline_playhead, "playhead, timelines");
    c.fill(622.0, y0, 36.0, 16.0, 2.0, S::timeline_playhead, "playhead, timelines");
    c.text_a(640.0, y0 + 8.0, Align2::CENTER_CENTER, "1054", 10.0, theme::get(S::text_strong), S::text_strong, "bright text");

    // key strip
    c.fill(220.0, 470.0, 960.0, 22.0, 2.0, S::range_out, "track");
    for t in [0.05f32, 0.27, 0.49, 0.72, 0.94] {
        c.diamond(220.0 + t * 960.0, 481.0, 6.0, S::keyframe, "keyframe");
    }

    // grade wheels
    c.text(20.0, 510.0, "Grade", 13.0, S::text_strong, "tile title");
    for (i, n) in ["lift", "gamma", "gain", "offset"].iter().enumerate() {
        let x = 20.0 + i as f32 * 170.0;
        c.fill(x, 532.0, 160.0, 210.0, 4.0, S::wheel_well, "wheel well");
        let (cx, cy) = (x + 80.0, 612.0);
        c.dot(cx, cy, 64.0, S::wheel_bevel, "wheel bevel");
        c.dot(cx, cy, 60.0, S::wheel_disc, "wheel disc");
        c.ring(cx, cy, 44.0, 1.0, S::grid_line, "grid lines");
        c.line(&[(cx - 60.0, cy), (cx + 60.0, cy)], 1.0, S::axis_line, "axis line");
        c.line(&[(cx, cy - 60.0), (cx, cy + 60.0)], 1.0, S::axis_line, "axis line");
        let off = [(8.0, -12.0), (-14.0, 6.0), (20.0, 10.0), (0.0, 0.0)][i];
        c.dot(cx + off.0, cy + off.1, 7.0, S::handle_rim, "handle rim");
        c.dot(cx + off.0, cy + off.1, 5.0, S::handle, "puck");
        c.text_a(cx, 690.0, Align2::CENTER_CENTER, n, 12.0, theme::get(S::text), S::text, "label");
        field(c, x + 40.0, 708.0, 80.0, "0.000");
    }

    // keyer
    c.text(710.0, 510.0, "Keyer", 13.0, S::text_strong, "tile title");
    c.fill(710.0, 532.0, 470.0, 210.0, 4.0, S::well, "well");
    for i in 1..8 {
        let x = 710.0 + i as f32 * 470.0 / 8.0;
        c.line(&[(x, 532.0), (x, 742.0)], 1.0, S::grid_line, "grid lines");
    }
    for (bx, by, r) in [(860.0, 610.0, 38.0), (930.0, 660.0, 26.0)] {
        c.dot_c(bx, by, r, a(S::keyer_blob, 70), S::keyer_blob, "key blob");
        c.ring(bx, by, r, 1.5, S::keyer_blob, "key blob");
    }
    c.dot_c(1040.0, 620.0, 30.0, a(S::keyer_subtract, 70), S::keyer_subtract, "key blob, subtract");
    c.ring(1040.0, 620.0, 30.0, 1.5, S::keyer_subtract, "key blob, subtract");
}

// ---------------------------------------------------------------------------
// edit timeline: source + record viewers, ruler, tracks, layer panel

fn edit(c: &mut Canvas) {
    c.fill(0.0, 0.0, 1200.0, 760.0, 0.0, S::bg, "panels");
    menubar(c);
    for (i, (x, tag, slot)) in [(10.0, "source", S::timeline_source), (604.0, "record", S::viewer_live)].iter().enumerate() {
        c.fill(*x, 36.0, 586.0, 300.0, 0.0, S::viewer_bg, "viewer backdrop");
        plate(c, x + 30.0, 56.0, 526.0, 260.0);
        c.outline(x + 30.0, 56.0, 526.0, 260.0, 0.0, 1.0, S::viewer_frame, "comp frame");
        c.text(x + 8.0, 40.0, tag, 12.0, *slot, if i == 0 { "source viewer" } else { "live pane label" });
    }
    // ruler
    c.fill(160.0, 346.0, 1030.0, 24.0, 0.0, S::timeline_ruler, "ruler");
    for i in 0..11 {
        let x = 160.0 + i as f32 * 103.0;
        c.line(&[(x, 360.0), (x, 370.0)], 1.0, S::text_weak, "ruler ticks");
        c.text(x + 3.0, 348.0, &format!("00:00:{:02}:00", i * 2), 10.0, S::text, "ruler label");
    }
    // layer panel
    let tracks = ["V3", "V2", "V1", "A1"];
    for (i, t) in tracks.iter().enumerate() {
        let y = 376.0 + i as f32 * 58.0;
        c.fill(10.0, y, 144.0, 52.0, 3.0, if i == 1 { S::raised_fill } else { S::hover_fill }, if i == 1 { "raised fill" } else { "hover fill" });
        c.text(20.0, y + 8.0, t, 12.0, S::text_strong, "track name");
        c.fill(20.0, y + 30.0, 110.0, 8.0, 2.0, if i == 2 { S::layer_bar_selected } else { S::layer_bar }, if i == 2 { "layer bar, selected" } else { "layer bars" });
        c.fill(160.0, y, 1030.0, 52.0, 0.0, S::range_out, "track");
    }
    // clips
    let clips: [(usize, f32, f32, S, &str); 9] = [
        (0, 420.0, 180.0, S::timeline_clip_5, "title_v2"),
        (1, 200.0, 260.0, S::timeline_clip_2, "fg_char_010"),
        (1, 700.0, 200.0, S::timeline_clip_4, "fx_smoke"),
        (2, 160.0, 240.0, S::timeline_clip_1, "sh010_bg"),
        (2, 404.0, 300.0, S::timeline_clip_3, "sh020_bg"),
        (2, 708.0, 220.0, S::timeline_clip_1, "sh030_bg"),
        (2, 932.0, 250.0, S::timeline_clip_6, "sh040_bg"),
        (3, 160.0, 560.0, S::timeline_clip_6, "dialogue.wav"),
        (3, 740.0, 440.0, S::timeline_clip_2, "music.wav"),
    ];
    for (row, x, w, slot, name) in clips {
        let y = 376.0 + row as f32 * 58.0 + 4.0;
        c.fill(x, y, w, 44.0, 3.0, slot, "clip");
        c.text(x + 8.0, y + 6.0, name, 11.5, S::text_strong, "clip name");
    }
    // handles past a clip's in/out
    c.fill_px(c.r(904.0, 438.0, 40.0, 44.0), 3.0, a(S::timeline_handles, 50), S::timeline_handles, "clip handles");
    c.fill(900.0, 438.0, 4.0, 44.0, 0.0, S::timeline_handles, "clip edge grip");
    // a clip being moved, and where it would land
    c.outline(520.0, 380.0, 180.0, 44.0, 3.0, 1.5, S::timeline_move, "clip being moved");
    c.text(528.0, 386.0, "title_v2", 11.5, S::timeline_move, "clip being moved");
    c.outline(640.0, 380.0, 180.0, 44.0, 3.0, 2.0, S::timeline_active, "drop target");
    c.fill_px(c.r(648.0, 356.0, 70.0, 18.0), 3.0, a(S::hud_bg, 180), S::hud_bg, "HUD backdrop");
    c.text_a(683.0, 365.0, Align2::CENTER_CENTER, "+00:12", 10.5, theme::get(S::timeline_active), S::timeline_active, "trim readout");
    // source in/out marks on the ruler
    c.line(&[(260.0, 346.0), (260.0, 370.0)], 2.0, S::timeline_source, "source in mark");
    c.line(&[(470.0, 346.0), (470.0, 370.0)], 2.0, S::timeline_source, "source out mark");
    // playhead
    c.line(&[(560.0, 346.0), (560.0, 610.0)], 1.5, S::timeline_playhead, "playhead, timelines");
    c.fill(552.0, 346.0, 16.0, 10.0, 2.0, S::timeline_playhead, "playhead, timelines");

    // bottom bar
    c.fill(0.0, 620.0, 1200.0, 140.0, 0.0, S::bg, "panels");
    c.line(&[(0.0, 618.0), (1200.0, 618.0)], 1.0, S::border, "separator");
    let mut x = 20.0;
    for b in ["|◀", "◀", "▶", "▶|"] {
        button(c, x, 636.0, 34.0, b, S::button, S::text);
        x += 40.0;
    }
    c.text(x + 10.0, 639.0, "00:00:07:12", 13.0, S::text_strong, "timecode");
    c.text_a(1180.0, 646.0, Align2::RIGHT_CENTER, "newer version on disk: sh020_bg v3", 11.5, theme::get(S::timeline_active), S::timeline_active, "newer version on disk");
    c.fill(20.0, 680.0, 1160.0, 60.0, 3.0, S::well, "well");
    c.text(30.0, 690.0, "bin", 12.0, S::text_weak, "faint text");
    for (i, n) in ["sh010_bg", "sh020_bg", "fg_char_010", "fx_smoke", "title_v2"].iter().enumerate() {
        let x = 70.0 + i as f32 * 150.0;
        let sel = i == 1;
        c.fill(x, 688.0, 140.0, 44.0, 3.0, if sel { S::browser_selected } else { S::raised_fill }, if sel { "selected tile" } else { "raised fill" });
        if sel {
            c.outline(x, 688.0, 140.0, 44.0, 3.0, 1.5, S::browser_accent, "browser accent");
        }
        c.text(x + 8.0, 702.0, n, 11.5, S::text, "label");
    }
}

// ---------------------------------------------------------------------------
// floating windows, status colours, keyboard map, file browser

fn windows(c: &mut Canvas) {
    c.fill(0.0, 0.0, 1200.0, 760.0, 0.0, S::bg, "panels");
    menubar(c);

    // a floating window with a shadow
    c.fill_px(c.r(40.0, 60.0, 420.0, 380.0).translate(egui::vec2(6.0, 8.0)), 8.0, theme::get(S::shadow), S::shadow, "drop shadow");
    c.fill(40.0, 60.0, 420.0, 380.0, 6.0, S::window, "window");
    c.outline(40.0, 60.0, 420.0, 380.0, 6.0, 1.0, S::border, "window outline");
    c.text(54.0, 70.0, "prefs…", 13.0, S::text_strong, "window title");
    c.text_a(446.0, 76.0, Align2::RIGHT_CENTER, "✕", 12.0, theme::get(S::text_hot), S::text_hot, "hovered label");
    c.line(&[(40.0, 92.0), (460.0, 92.0)], 1.0, S::border, "separator");
    let rows = ["display transform", "working space", "cache budget", "picker size", "selection cap"];
    for (i, r) in rows.iter().enumerate() {
        let y = 104.0 + i as f32 * 28.0;
        if i % 2 == 1 {
            c.fill(46.0, y - 3.0, 408.0, 26.0, 2.0, S::raised_fill, "striped row");
        }
        if i == 2 {
            c.fill(46.0, y - 3.0, 408.0, 26.0, 2.0, S::hover_fill, "hovered row");
        }
        c.text(58.0, y + 2.0, r, 12.0, S::text, "label");
        field(c, 300.0, y, 140.0, ["ACES 1.3", "ACEScg", "24 GB", "3.0 × 3.0", "400"][i]);
    }
    button(c, 58.0, 258.0, 80.0, "normal", S::button, S::text);
    button(c, 146.0, 258.0, 80.0, "hovered", S::button_hover, S::text_hot);
    button(c, 234.0, 258.0, 80.0, "pressed", S::button_active, S::text);
    c.fill(322.0, 258.0, 80.0, 20.0, 3.0, S::accent, "selected item");
    c.text_a(362.0, 268.0, Align2::CENTER_CENTER, "selected", 11.5, theme::get(S::accent_text), S::accent_text, "text on accent");
    c.text(58.0, 292.0, "search keeps every colour; the list shows roots", 11.0, S::text_weak, "faint text");
    c.text(58.0, 312.0, "open the manual ↗", 11.5, S::link, "link");
    c.fill(58.0, 336.0, 380.0, 22.0, 3.0, S::field, "field");
    c.text(66.0, 340.0, "type to filter…", 11.5, S::text_weak, "placeholder");
    c.fill(58.0, 366.0, 80.0, 18.0, 2.0, S::accent, "selected text");
    c.text(62.0, 368.0, "ACEScg", 11.5, S::accent_text, "text on accent");
    c.line(&[(141.0, 366.0), (141.0, 384.0)], 1.5, S::accent_text, "text cursor");
    c.text(58.0, 402.0, "status:", 11.5, S::text, "label");
    c.text(110.0, 402.0, "ok", 11.5, S::ok, "ok");
    c.text(140.0, 402.0, "info", 11.5, S::info, "info");
    c.text(180.0, 402.0, "warning", 11.5, S::warn, "warning");
    c.text(246.0, 402.0, "error", 11.5, S::error, "error");

    // tooltip + pie
    c.fill_px(c.r(500.0, 70.0, 230.0, 46.0), 5.0, a(S::hud_bg, 200), S::hud_bg, "HUD backdrop");
    c.text(510.0, 76.0, "Grade · gain", 12.0, S::text_strong, "bright text");
    c.text(510.0, 94.0, "multiplies the image", 11.0, S::text, "label");
    c.dot_c(620.0, 230.0, 80.0, a(S::hud_bg, 140), S::hud_bg, "HUD backdrop");
    for (i, n) in ["Grade", "Blur", "Merge", "Load", "Render", "Transform"].iter().enumerate() {
        let ang = i as f32 / 6.0 * std::f32::consts::TAU - 1.57;
        let (x, y) = (620.0 + ang.cos() * 56.0, 230.0 + ang.sin() * 56.0);
        let slot = if i == 1 { S::text_hot } else { S::text };
        c.text_a(x, y, Align2::CENTER_CENTER, n, 11.0, theme::get(slot), slot, if i == 1 { "hovered label" } else { "label" });
    }
    c.outline(720.0, 150.0, 70.0, 70.0, 0.0, 1.5, S::selection, "selection outline");
    c.text(720.0, 226.0, "selected", 11.0, S::text_weak, "faint text");

    // keyboard map
    let (kx, ky) = (40.0, 470.0);
    c.text(kx, ky - 20.0, "keys", 12.0, S::text_strong, "window title");
    let scopes = [S::keys_app, S::keys_viewer, S::keys_graph, S::keys_curves, S::keys_paint];
    let names = ["app", "viewer", "graph", "curves", "paint"];
    let rows = ["QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM"];
    for (r, row) in rows.iter().enumerate() {
        for (i, ch) in row.chars().enumerate() {
            let x = kx + r as f32 * 18.0 + i as f32 * 46.0;
            let y = ky + r as f32 * 46.0;
            let bound = (i + r) % 4 != 3;
            let cap = theme::get(S::key_cap);
            let cap = if bound { cap } else { theme::lift(cap, -12) };
            c.fill_px(c.r(x, y, 40.0, 40.0), 4.0, cap, S::key_cap, if bound { "keycap" } else { "keycap, unbound" });
            c.text(x + 5.0, y + 4.0, &ch.to_string(), 11.0, S::text, "label");
            if bound {
                let s = scopes[(i * 3 + r) % scopes.len()];
                c.fill(x + 4.0, y + 32.0, 32.0, 4.0, 1.0, s, "key scope");
            }
        }
    }
    for (i, (s, n)) in scopes.iter().zip(names).enumerate() {
        let x = kx + i as f32 * 90.0;
        c.fill(x, ky + 150.0, 10.0, 10.0, 2.0, *s, "key scope");
        c.text(x + 16.0, ky + 147.0, n, 11.0, *s, "key scope");
    }

    // file browser
    let (bx, by) = (560.0, 330.0);
    c.fill(bx, by, 620.0, 400.0, 4.0, S::well, "well");
    c.text(bx + 10.0, by + 8.0, "~/shots/sh020/plates", 12.0, S::text, "label");
    c.text_a(bx + 610.0, by + 14.0, Align2::RIGHT_CENTER, "name ↓   date   frames", 11.0, theme::get(S::text_weak), S::text_weak, "column header");
    for i in 0..12 {
        let (col, row) = (i % 4, i / 4);
        let x = bx + 12.0 + col as f32 * 150.0;
        let y = by + 36.0 + row as f32 * 118.0;
        let sel = i == 5;
        c.fill(x, y, 140.0, 108.0, 4.0, if sel { S::browser_selected } else { S::raised_fill }, if sel { "selected tile" } else { "tile" });
        plate(c, x + 8.0, y + 8.0, 124.0, 70.0);
        if sel {
            c.outline(x, y, 140.0, 108.0, 4.0, 1.5, S::browser_accent, "browser accent");
        }
        c.text(x + 8.0, y + 84.0, &format!("plate_{:02}.####.exr", i + 1), 10.0, if sel { S::text_strong } else { S::text }, "label");
    }
}
