//! Resolve-style primaries wheels for the Grade node's properties panel.
//!
//! Pure UI over `GradeParams` — no engine involvement. Each wheel edits one
//! rgb triple: the color disc adds a zero-sum chroma offset (drag toward a
//! hue to push the triple that way), the dial strip below shifts all three
//! channels together (the "master"), and the boxes are master/R/G/B numeric
//! entry, Resolve layout. The arc outside the hue ring shows how far master
//! sits from its default, wrapping clockwise for increase.
//!
//! Owned by instance B (see .agents/board.md). To add a wheel, extend the
//! spec table in `adjust_wheels_ui`. The set is bound to four triples rather
//! than to `GradeParams` so the Grade's master and each of its zones draw
//! the same instrument.

use crate::theme;
use egui::{
    epaint, vec2, Align2, Color32, CursorIcon, FontId, Pos2, Rect, Sense, Stroke, Vec2,
};

/// Per-wheel config: value semantics differ (lift is additive around 0,
/// gamma/gain multiplicative around 1) but the widget is the same.
struct WheelSpec {
    /// Owned, not `&'static str`: a tool's Colour Wheel control carries a
    /// user-authored caption, and the Grade's four are just literals.
    label: String,
    default: f32,
    lo: f32,
    hi: f32,
    /// Chroma offset when the puck reaches the disc rim.
    chroma_range: f32,
    /// Master units per px of dial drag (before shift/alt modifiers).
    master_speed: f32,
    /// Master delta that wraps the indicator arc a full turn.
    arc_span: f32,
}

// Screen-space directions of the R/G/B primaries on the wheel (y down):
// red up, green lower-left, blue lower-right — matches Resolve's ring.
const DIR_R: Vec2 = Vec2::new(0.0, -1.0);
const DIR_G: Vec2 = Vec2::new(-0.866_025_4, 0.5);
const DIR_B: Vec2 = Vec2::new(0.866_025_4, 0.5);
const DIRS: [Vec2; 3] = [DIR_R, DIR_G, DIR_B];

/// Geometry at zoom 1. The properties grid can zoom, and everything a
/// drag reads position from (wheel, ring, arc) scales with it — bigger
/// means finer control. Type, value boxes, pucks and strokes do NOT
/// scale: they are read, not aimed.
pub const COL_W: f32 = 150.0;
pub const COL_H: f32 = 187.0;
/// Between wheel columns, at zoom 1.
const WHEEL_GAP: f32 = 12.0;
const WHEEL_A: f32 = 118.0; // wheel square side
const R_ARC: f32 = 52.0;
const R_RING_OUT: f32 = 47.0;
const R_RING_IN: f32 = 42.0;
const R_DISC: f32 = 36.0;

/// One wheel's scaled geometry.
#[derive(Clone, Copy)]
struct Geo {
    s: f32,
    col: Vec2,
    wheel_a: f32,
    r_arc: f32,
    r_ring_out: f32,
    r_ring_in: f32,
    r_disc: f32,
}

impl Geo {
    fn new(s: f32) -> Self {
        Self {
            s,
            col: vec2(COL_W * s, COL_H * s),
            wheel_a: WHEEL_A * s,
            r_arc: R_ARC * s,
            r_ring_out: R_RING_OUT * s,
            r_ring_in: R_RING_IN * s,
            r_disc: R_DISC * s,
        }
    }
}

const UNDERLINES: [Color32; 4] = [
    Color32::from_rgb(0xd9, 0xd9, 0xd9), // master
    Color32::from_rgb(0xd8, 0x4a, 0x4a),
    Color32::from_rgb(0x3f, 0xae, 0x4a),
    Color32::from_rgb(0x4a, 0x6f, 0xd8),
];

/// The four wheels over ANY lift / gamma / gain / offset — the Grade's
/// master or one of its zones, which are the same five adjustments per band.
/// `salt` keeps the sets on one node apart in egui's id space (a zone's
/// wheels must not inherit the master's drag state).
pub fn adjust_wheels_ui(
    ui: &mut egui::Ui,
    v: [&mut [f32; 3]; 4],
    salt: impl std::hash::Hash + std::fmt::Debug,
    scale: f32,
) -> Option<usize> {
    // The Grade is the hero panel and is never clipped: the four columns
    // take the zoom, but no more than the width they were given (a tile
    // stretched narrower, a small zoomed-out rack). Row = 4 cols + 3 gaps.
    let fit = ui.available_width() / (4.0 * COL_W + 3.0 * WHEEL_GAP);
    let geo = Geo::new(scale.min(fit).max(0.2));

    // gamma <= 0 explodes pow() into NaN (same guard as codegen / old grid).
    let specs = [
        WheelSpec {
            label: "Lift".into(),
            default: 0.0,
            lo: -4.0,
            hi: 8.0,
            chroma_range: 0.35,
            master_speed: 0.003,
            arc_span: 1.0,
        },
        WheelSpec {
            label: "Gamma".into(),
            default: 1.0,
            lo: 0.05,
            hi: 8.0,
            chroma_range: 0.4,
            master_speed: 0.003,
            arc_span: 2.0,
        },
        WheelSpec {
            label: "Gain".into(),
            default: 1.0,
            lo: -4.0,
            hi: 8.0,
            chroma_range: 0.35,
            master_speed: 0.004,
            arc_span: 2.0,
        },
        WheelSpec {
            label: "Offset".into(),
            default: 0.0,
            lo: -4.0,
            hi: 8.0,
            chroma_range: 0.35,
            master_speed: 0.003,
            arc_span: 1.0,
        },
    ];
    let mut pulled = None;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = WHEEL_GAP * geo.s;
        for ((wi, spec), v) in specs.iter().enumerate().zip(v) {
            if ui.push_id((&salt, "wheel", wi), |ui| one_wheel(ui, spec, v, geo)).inner {
                pulled = Some(wi);
            }
        }
    });
    pulled
}

/// True on the frame a ctrl-drag starts on the wheel: the wheel is being
/// PULLED out (into a viewer HUD, or onto a tool's panel composer), the
/// gesture every properties row has.
fn one_wheel(ui: &mut egui::Ui, spec: &WheelSpec, v: &mut [f32; 3], geo: Geo) -> bool {
    let (col, _) = ui.allocate_exact_size(geo.col, Sense::hover());
    if !ui.is_rect_visible(col) {
        return false;
    }
    // Suites aim the ctrl-drag pull at this (`widgets::noted_rect`).
    crate::widgets::note_rect(ui, egui::Id::new(("wheel-col", &spec.label)), col);
    let p = ui.painter().clone();
    let fine = ui.input(|i| i.modifiers.alt);
    let coarse = ui.input(|i| i.modifiers.shift);
    let speed_mod = if fine { 0.2 } else if coarse { 4.0 } else { 1.0 };
    // The z/x/c/v/b/n rate row belongs to this wheel while the pointer is
    // over it, exactly as it does over any parameter row: the keys are ALSO
    // hotkeys (V hides the properties bar), and without the claim, holding
    // one over the wheel to set a finer drag would fire it instead.
    if ui
        .input(|i| i.pointer.hover_pos())
        .is_some_and(|pos| col.contains(pos))
    {
        crate::widgets::claim_sens(ui.ctx());
    }
    // E over the wheel: sample the viewer into its triple.
    let hovered = ui.input(|i| i.pointer.hover_pos()).is_some_and(|pos| col.contains(pos));
    let drop = crate::eyedrop::hook(ui, ui.id().with("eyedrop"), hovered);
    if let Some(c) = drop.sample {
        *v = [c[0], c[1], c[2]];
    }
    if drop.armed {
        p.rect_stroke(
            col.shrink(1.0),
            4.0,
            egui::Stroke::new(1.5, ui.visuals().selection.stroke.color),
            egui::StrokeKind::Inside,
        );
    }

    // -- header: centered label, reset at the right ---------------------------
    let header = Rect::from_min_size(col.min, vec2(geo.col.x, 18.0));
    p.text(
        header.center(),
        Align2::CENTER_CENTER,
        spec.label.as_str(),
        FontId::proportional(12.0),
        theme::text_strong(),
    );
    let reset_rect = Rect::from_center_size(
        Pos2::new(header.max.x - 12.0, header.center().y),
        vec2(16.0, 16.0),
    );
    let reset = ui
        .put(
            reset_rect,
            egui::Button::new(egui::RichText::new("↺").size(12.0)).frame(false),
        )
        .on_hover_text(format!("reset {}", spec.label.to_lowercase()));
    if reset.clicked() {
        *v = [spec.default; 3];
    }

    // -- wheel ----------------------------------------------------------------
    let wheel_rect = Rect::from_center_size(
        Pos2::new(col.center().x, header.max.y + geo.wheel_a * 0.5),
        Vec2::splat(geo.wheel_a),
    );
    let c = wheel_rect.center();
    // ctrl is the pull (see the end): it moves nothing.
    let ctrl = ui.input(|i| i.modifiers.ctrl || i.modifiers.command);
    let resp = ui
        .interact(wheel_rect, ui.id().with("disc"), Sense::click_and_drag())
        .on_hover_cursor(CursorIcon::Crosshair);
    if resp.dragged() && !ctrl {
        // `rated_delta`, not `drag_delta`: the held rate key scales the
        // throw here as it does on every other draggable value.
        let d = crate::widgets::rated_delta(ui, &resp) * (spec.chroma_range / 220.0) * speed_mod
            / geo.s;
        for (ch, dir) in v.iter_mut().zip(DIRS) {
            *ch = (*ch + d.dot(dir)).clamp(spec.lo, spec.hi);
        }
    }
    if resp.double_clicked() {
        // Kill chroma, keep master.
        let m = mean(v);
        *v = [m.clamp(spec.lo, spec.hi); 3];
    }

    paint_wheel(&p, c, v, spec, geo);

    // -- value boxes: master, r, g, b ----------------------------------------
    let boxes_top = wheel_rect.max.y + 6.0;
    let bw = 32.0;
    let gap = 4.0;
    let row_w = bw * 4.0 + gap * 3.0;
    let x0 = col.center().x - row_w * 0.5;
    let m0 = mean(v);
    let mut m = m0;
    for i in 0..4 {
        let r = Rect::from_min_size(
            Pos2::new(x0 + i as f32 * (bw + gap), boxes_top),
            vec2(bw, 16.0),
        );
        let dv = if i == 0 {
            egui::DragValue::new(&mut m).update_while_editing(false)
        } else {
            egui::DragValue::new(&mut v[i - 1]).update_while_editing(false)
        };
        ui.put(
            r,
            dv.speed(crate::widgets::rated(ui, 0.005 * speed_mod as f64))
                .range(spec.lo..=spec.hi)
                .fixed_decimals(2),
        );
        // Resolve's channel underline.
        let ul = Rect::from_min_size(Pos2::new(r.min.x + 2.0, r.max.y + 2.0), vec2(bw - 4.0, 3.0));
        ui.painter().rect_filled(ul, 1.0, UNDERLINES[i]);
    }
    if m != m0 {
        let d = m - m0;
        for ch in v.iter_mut() {
            *ch = (*ch + d).clamp(spec.lo, spec.hi);
        }
    }

    // -- master dial strip ----------------------------------------------------
    let dial_w = row_w.max(geo.col.x - 10.0);
    let dial = Rect::from_min_size(
        Pos2::new(col.center().x - dial_w * 0.5, boxes_top + 16.0 + 9.0),
        vec2(dial_w, 13.0),
    );
    #[cfg(test)]
    ui.ctx().data_mut(|d| d.insert_temp(egui::Id::new(TEST_DIAL), dial));
    let dresp = ui
        .interact(dial, ui.id().with("dial"), Sense::click_and_drag())
        .on_hover_cursor(CursorIcon::ResizeHorizontal);
    if dresp.dragged() && !ctrl {
        // The master jog strip: the rate row scales it like any other drag.
        let d = crate::widgets::rated_delta(ui, &dresp).x * spec.master_speed * speed_mod / geo.s;
        for ch in v.iter_mut() {
            *ch = (*ch + d).clamp(spec.lo, spec.hi);
        }
    }
    if dresp.double_clicked() {
        let d = spec.default - mean(v);
        for ch in v.iter_mut() {
            *ch = (*ch + d).clamp(spec.lo, spec.hi);
        }
    }
    paint_dial(ui.painter(), dial, mean(v), dresp.hovered() || dresp.dragged());
    let started = resp.drag_started() || dresp.drag_started();
    dresp.on_hover_text(
        "drag: master (all channels) — hold Z X C V B N for the drag rate, \
         ⇧ coarse, ⌥ fine\nctrl-drag: pull the wheel out (viewer HUD, or a subwob's panel)\n\
         double-click: reset master",
    );
    // ctrl held: the whole column is one grip, registered LAST so it is
    // topmost and the disc / dial underneath never see the drag.
    // The disc and dial count too: egui hit-tests against LAST frame's
    // widgets, so a ctrl that goes down on the very frame of the press finds
    // no grip yet and the drag lands on whatever is under it.
    let grip = ctrl
        && ui
            .interact(col, ui.id().with("pull"), Sense::drag())
            .on_hover_cursor(CursorIcon::Grab)
            .drag_started();
    ctrl && (grip || started)
}

fn mean(v: &[f32; 3]) -> f32 {
    (v[0] + v[1] + v[2]) / 3.0
}

/// Vivid ring color in the given screen direction (unit vector from center).
fn hue_at(dir: Vec2) -> Color32 {
    let ch = |d: Vec2| ((dir.dot(d) + 0.5).clamp(0.0, 1.0) * 235.0) as u8;
    Color32::from_rgb(ch(DIR_R), ch(DIR_G), ch(DIR_B))
}

fn lerp_rgb(a: Color32, b: Color32, t: f32) -> Color32 {
    let l = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t) as u8;
    Color32::from_rgb(l(a.r(), b.r()), l(a.g(), b.g()), l(a.b(), b.b()))
}

/// Angle in degrees, clockwise from 12 o'clock, to a screen direction.
fn cw_dir(deg: f32) -> Vec2 {
    let a = deg.to_radians();
    Vec2::new(a.sin(), -a.cos())
}

fn paint_wheel(p: &egui::Painter, c: Pos2, v: &[f32; 3], spec: &WheelSpec, geo: Geo) {
    let well = theme::wheel_well();
    let disc_bg = theme::wheel_disc();

    // Recessed well behind ring + disc.
    p.circle_filled(c, geo.r_ring_out + 3.0, well);

    // Hue ring: annulus mesh, per-vertex vivid hue.
    let mut mesh = epaint::Mesh::default();
    const SEGS: usize = 72;
    for s in 0..=SEGS {
        let dir = cw_dir(s as f32 / SEGS as f32 * 360.0);
        let col = hue_at(dir);
        mesh.colored_vertex(c + dir * geo.r_ring_in, col);
        mesh.colored_vertex(c + dir * geo.r_ring_out, col);
        if s > 0 {
            let i = (s * 2) as u32;
            mesh.add_triangle(i - 2, i - 1, i);
            mesh.add_triangle(i - 1, i + 1, i);
        }
    }
    p.add(egui::Shape::mesh(mesh));
    // Bevel edges keep the ring crisp against well + disc.
    p.circle_stroke(c, geo.r_ring_out, Stroke::new(1.0, theme::wheel_bevel()));
    p.circle_stroke(c, geo.r_ring_in, Stroke::new(1.0, theme::wheel_bevel()));

    // Disc: dark center fading to a faint hue tint at the rim.
    let mut fan = epaint::Mesh::default();
    fan.colored_vertex(c, disc_bg);
    for s in 0..=SEGS {
        let dir = cw_dir(s as f32 / SEGS as f32 * 360.0);
        fan.colored_vertex(c + dir * geo.r_disc, lerp_rgb(disc_bg, hue_at(dir), 0.20));
        if s > 0 {
            let i = (s + 1) as u32;
            fan.add_triangle(0, i - 1, i);
        }
    }
    p.add(egui::Shape::mesh(fan));

    // Crosshair.
    let cross = Stroke::new(1.0, theme::grid_line());
    p.line_segment([c - vec2(geo.r_disc, 0.0), c + vec2(geo.r_disc, 0.0)], cross);
    p.line_segment([c - vec2(0.0, geo.r_disc), c + vec2(0.0, geo.r_disc)], cross);

    // Master arc: stub at the bottom, wrapping clockwise as master rises.
    let arc_col = theme::lift(theme::handle(), -69);
    let stub = theme::lift(theme::axis_line(), 32);
    p.line(arc_pts(c, 170.0, 190.0, geo), Stroke::new(2.0, stub));
    let sweep = ((mean(v) - spec.default) / spec.arc_span * 360.0).clamp(-352.0, 352.0);
    if sweep.abs() > 1.0 {
        p.line(arc_pts(c, 180.0, 180.0 + sweep, geo), Stroke::new(2.5, arc_col));
        p.circle_filled(
            c + cw_dir(180.0 + sweep) * geo.r_arc,
            2.4,
            theme::lift(theme::handle(), 26),
        );
    }

    // Puck at the chroma offset (zero-sum part of the triple).
    let m = mean(v);
    let k = 2.0 / (3.0 * spec.chroma_range);
    let mut puck: Vec2 = DIRS
        .iter()
        .zip(v)
        .map(|(dir, ch)| *dir * (ch - m) * k)
        .fold(Vec2::ZERO, |a, b| a + b);
    if puck.length() > 0.94 {
        puck = puck.normalized() * 0.94;
    }
    let pp = c + puck * geo.r_disc;
    p.circle_filled(pp + vec2(0.0, 1.0), 4.5, theme::shadow());
    p.circle(pp, 4.0, theme::handle(), Stroke::new(1.0, theme::handle_rim()));
}

fn arc_pts(c: Pos2, from_deg: f32, to_deg: f32, geo: Geo) -> Vec<Pos2> {
    let n = ((to_deg - from_deg).abs() / 4.0).ceil().max(1.0) as usize;
    (0..=n)
        .map(|i| c + cw_dir(from_deg + (to_deg - from_deg) * i as f32 / n as f32) * geo.r_arc)
        .collect()
}

fn paint_dial(p: &egui::Painter, r: Rect, master: f32, hot: bool) {
    p.rect_filled(r, 3.0, theme::lift(theme::wheel_well(), -8));
    p.rect_stroke(
        r,
        3.0,
        Stroke::new(1.0, theme::lift(theme::wheel_bevel(), -9)),
        epaint::StrokeKind::Outside,
    );
    let ridge = if hot {
        theme::lift(theme::wheel_disc(), 49)
    } else {
        theme::lift(theme::wheel_disc(), 23)
    };
    // Ridges scroll with the value so drags give tactile feedback.
    let spacing = 5.0;
    let off = (-master * 160.0).rem_euclid(spacing);
    let mut x = r.min.x + 2.0 + off;
    while x < r.max.x - 2.0 {
        p.line_segment(
            [Pos2::new(x, r.min.y + 2.5), Pos2::new(x, r.max.y - 2.5)],
            Stroke::new(1.0, ridge),
        );
        x += spacing;
    }
}

/// Where the last wheel drawn put its master dial — so a test can aim a
/// real drag at it rather than re-derive the layout.
#[cfg(test)]
const TEST_DIAL: &str = "wheels-test-dial";

#[cfg(test)]
mod tests {
    use super::{Geo, COL_H, COL_W};

    /// Drag the master dial `px` to the right over several frames with
    /// `key` held (if any), and return how far the master moved.
    fn dial_travel(key: Option<egui::Key>, px: f32) -> f32 {
        let ctx = egui::Context::default();
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(400.0, 300.0));
        let mut v = [1.0f32; 3];
        let frame = |events: Vec<egui::Event>, v: &mut [f32; 3]| {
            let input = egui::RawInput {
                screen_rect: Some(screen),
                events,
                ..Default::default()
            };
            let _ = ctx.run_ui(input, |ui| {
                super::bound_wheel_ui(ui, "gain", v, -4.0, 8.0, 1.0, true);
            });
        };
        frame(Vec::new(), &mut v);
        let dial: egui::Rect = ctx
            .data(|d| d.get_temp(egui::Id::new(super::TEST_DIAL)))
            .expect("the wheel recorded its dial");
        let start = dial.center();
        let before = v;
        let btn = |pos, pressed| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: Default::default(),
        };
        let keyev = |k, pressed| egui::Event::Key {
            key: k,
            physical_key: None,
            pressed,
            repeat: false,
            modifiers: Default::default(),
        };
        let mut down = vec![egui::Event::PointerMoved(start)];
        if let Some(k) = key {
            down.push(keyev(k, true));
        }
        frame(down, &mut v);
        frame(vec![btn(start, true)], &mut v);
        // Anchored to the PRESS point, in steps: a real hand's path does not
        // depend on where the value went.
        let steps = 5;
        for s in 1..=steps {
            let p = start + egui::vec2(px * s as f32 / steps as f32, 0.0);
            frame(vec![egui::Event::PointerMoved(p)], &mut v);
        }
        let end = start + egui::vec2(px, 0.0);
        let mut up = vec![btn(end, false)];
        if let Some(k) = key {
            up.push(keyev(k, false));
        }
        frame(up, &mut v);
        let m = |x: [f32; 3]| (x[0] + x[1] + x[2]) / 3.0;
        m(v) - m(before)
    }

    /// The bug this pins: the rate row (Z X C V B N) did nothing to the
    /// horizontal master dial, because it read the raw `drag_delta` while
    /// every other draggable value in the app reads the rated one.
    #[test]
    fn the_rate_keys_scale_the_master_dial() {
        let plain = dial_travel(None, 40.0);
        assert!(plain > 0.01, "a plain drag did not move the master: {plain}");
        let fast = dial_travel(Some(egui::Key::V), 40.0);
        let slow = dial_travel(Some(egui::Key::X), 40.0);
        let (vf, xf) = (
            super::super::rgbslider::RATE_KEYS.iter().find(|(k, _)| *k == egui::Key::V).unwrap().1,
            super::super::rgbslider::RATE_KEYS.iter().find(|(k, _)| *k == egui::Key::X).unwrap().1,
        );
        assert!(
            (fast / plain - vf).abs() < 0.05 * vf,
            "V should scale the dial by {vf}: {fast} vs {plain}"
        );
        assert!(
            (slow / plain - xf).abs() < 0.05 * xf.max(0.1),
            "X should scale the dial by {xf}: {slow} vs {plain}"
        );
    }

    #[test]
    fn geometry_scales_but_stays_proportional() {
        let a = Geo::new(1.0);
        let b = Geo::new(2.0);
        assert_eq!(a.col.x, COL_W);
        assert_eq!(a.col.y, COL_H);
        assert_eq!(b.col.x, COL_W * 2.0);
        assert_eq!(b.wheel_a, a.wheel_a * 2.0);
        assert_eq!(b.r_disc, a.r_disc * 2.0);
        // The ring stays an annulus around the disc at any zoom.
        for g in [a, b] {
            assert!(g.r_disc < g.r_ring_in);
            assert!(g.r_ring_in < g.r_ring_out);
            assert!(g.r_ring_out < g.r_arc);
            assert!(g.r_arc * 2.0 <= g.wheel_a);
        }
    }
}

/// One wheel bound to ANY rgb triple — the Grade's instrument without a
/// `GradeParams` behind it. This is what a tool's Colour Wheel control draws,
/// and the reason `WheelSpec::label` is owned.
pub fn bound_wheel_ui(
    ui: &mut egui::Ui,
    label: &str,
    v: &mut [f32; 3],
    lo: f32,
    hi: f32,
    scale: f32,
    // A MULTIPLIER (gain, gamma: neutral at 1) or an OFFSET (lift, offset:
    // neutral at 0). The wheel is drawn and dragged about its neutral, so
    // an offset treated as a multiplier jumps on the first touch.
    mult: bool,
) {
    let spec = WheelSpec {
        label: label.to_string(),
        default: if mult { 1.0 } else { 0.0 },
        lo,
        hi,
        chroma_range: 0.35,
        master_speed: if mult { 0.004 } else { 0.003 },
        arc_span: if mult { 2.0 } else { 1.0 },
    };
    one_wheel(ui, &spec, v, Geo::new(scale.max(0.2)));
}
