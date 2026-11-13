//! GridWarp's app half: the panel (lattice size, which grid the viewer
//! edits, the picked point's numbers), the viewer gizmo, the glyph and the
//! channel bindings. The lattice maths, the WGSL and the params are
//! `engine::nodes::gridwarp`.
//!
//! Points are indexed channels (`GwSrcX(i)`, `GwDstY(i)`…), so every point
//! of both grids keys, patches and HUD-slides like any other value. The
//! panel and the gizmo share one per-node state in egui temp memory — which
//! grid is being edited and which points are picked — so picking a point on
//! the picture puts its numbers in the panel.
//!
//! The gizmo follows the gizmo rule: a drag is computed ABSOLUTELY from the
//! points as they were at the press, never summed per frame, because a
//! point's channel may be animated and the working copy is re-evaluated at
//! the playhead every frame.

use crate::anim::{AnimCurve, Channel};
use crate::hudkeys;
use crate::ink::InkShapes;
use crate::pointpick;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Rect, Stroke, Vec2};
use engine::graph::{GridWarpParams, NodeKind, GRIDWARP_MAX};
use std::collections::BTreeMap;

/// The fixed channel. The points' own are indexed ([`point_channels`]).
pub const CHANNELS: &[Channel] = &[Channel::GwAmount];

/// Two tiles: the edit row and the picked point's two pairs of numbers
/// want width.
pub const TILES: usize = 2;

/// Point `i`'s channels: source x, y, then destination x, y.
pub fn point_channels(i: u16) -> [Channel; 4] {
    [Channel::GwSrcX(i), Channel::GwSrcY(i), Channel::GwDstX(i), Channel::GwDstY(i)]
}

pub fn get(p: &GridWarpParams, ch: Channel) -> Option<f32> {
    use Channel::*;
    Some(match ch {
        GwAmount => p.amount,
        GwSrcX(i) => p.src.get(i as usize)?[0],
        GwSrcY(i) => p.src.get(i as usize)?[1],
        GwDstX(i) => p.dst.get(i as usize)?[0],
        GwDstY(i) => p.dst.get(i as usize)?[1],
        _ => return None,
    })
}

pub fn set(p: &mut GridWarpParams, ch: Channel, v: f32) {
    use Channel::*;
    let (grid, i, lane) = match ch {
        GwAmount => return p.amount = v,
        GwSrcX(i) => (&mut p.src, i, 0),
        GwSrcY(i) => (&mut p.src, i, 1),
        GwDstX(i) => (&mut p.dst, i, 0),
        GwDstY(i) => (&mut p.dst, i, 1),
        _ => return,
    };
    if let Some(q) = grid.get_mut(i as usize) {
        q[lane] = v;
    }
}

/// The curve editor's name for the fixed channel.
pub fn label(ch: Channel) -> Option<&'static str> {
    match ch {
        Channel::GwAmount => Some("warp amount"),
        _ => None,
    }
}

/// The curve editor's name for a point's channel.
pub fn point_label(ch: Channel) -> Option<String> {
    use Channel::*;
    let (i, what) = match ch {
        GwSrcX(i) => (i, "source x"),
        GwSrcY(i) => (i, "source y"),
        GwDstX(i) => (i, "dest x"),
        GwDstY(i) => (i, "dest y"),
        _ => return None,
    };
    Some(format!("pt{i} {what}"))
}

/// Drop every point curve: a re-grid renumbers the points, so the old
/// curves would land on the wrong ones.
pub fn clear_point_curves(anim: &mut BTreeMap<Channel, AnimCurve>) {
    anim.retain(|ch, _| !matches!(ch, Channel::GwSrcX(_) | Channel::GwSrcY(_) | Channel::GwDstX(_) | Channel::GwDstY(_)));
}

/// Which grid the viewer's handles move.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Edit {
    /// Where things go: the warp itself.
    #[default]
    Dst,
    /// What gets picked up.
    Src,
    /// Both together: re-place a control point without changing the warp
    /// there.
    Both,
}

/// The per-node state the panel and the gizmo share.
#[derive(Clone, Debug, Default)]
struct Ui {
    edit: Edit,
    pp: pointpick::Pick,
}

fn ui_id(node: u64) -> egui::Id {
    egui::Id::new(("gridwarp-ui", node))
}

fn load(ctx: &egui::Context, node: u64, n: usize) -> Ui {
    let mut s: Ui = ctx.data(|d| d.get_temp::<Ui>(ui_id(node))).unwrap_or_default();
    // A re-grid renumbered the points.
    s.pp.fit(n);
    s
}

fn store(ctx: &egui::Context, node: u64, s: Ui) {
    ctx.data_mut(|d| d.insert_temp(ui_id(node), s));
}

/// Which points are picked (`--test-gridwarp` reads it).
pub fn picked(ctx: &egui::Context, node: u64, n: usize) -> Vec<bool> {
    load(ctx, node, n).pp.pick
}

/// Replace the pick (`--test-gridwarp` sets one).
pub fn set_pick(ctx: &egui::Context, node: u64, n: usize, pick: &[usize]) {
    let mut s = load(ctx, node, n);
    s.pp.set(n, pick);
    store(ctx, node, s);
}

/// Where the pick's box drew its grips last frame (`--test-gridwarp` aims
/// at them).
pub fn box_hot(ctx: &egui::Context, node: u64) -> Option<pointpick::BoxHot> {
    ctx.data(|d| d.get_temp::<Ui>(ui_id(node))).and_then(|s| s.pp.hot)
}

/// Point the viewer's handles at one grid (`--test-gridwarp` sets it).
pub fn set_edit(ctx: &egui::Context, node: u64, edit: Edit) {
    let mut s: Ui = ctx.data(|d| d.get_temp::<Ui>(ui_id(node))).unwrap_or_default();
    s.edit = edit;
    store(ctx, node, s);
}

/// What the panel hands back.
#[derive(Default)]
pub struct PanelOut {
    pub hud: Option<HudSlider>,
    /// Both grids replaced wholesale (reset, swap): written back through
    /// the keyed path by the caller, so an animated point gets a key.
    pub replace: Option<GridWarpParams>,
}

pub fn panel(ui: &mut egui::Ui, rcx: &mut RowCtx, p: &mut GridWarpParams) -> PanelOut {
    let mut out = PanelOut::default();
    let node = rcx.node;
    // What is on screen: the curves evaluated at the playhead.
    let ev = match rcx.ekind {
        NodeKind::GridWarp(g) => g.clone(),
        _ => p.clone(),
    };
    let ctx = ui.ctx().clone();
    let mut st = load(&ctx, node, p.count());

    ui.horizontal(|ui| {
        ui.add_sized([widgets::LABEL_W, 16.0], egui::Label::new(egui::RichText::new("grid").small()));
        let (mut c, mut r) = (p.cols, p.rows);
        let field = |ui: &mut egui::Ui, v: &mut u32| {
            widgets::num_field(ui, egui::DragValue::new(v).update_while_editing(false).range(1..=GRIDWARP_MAX).speed(0.05))
        };
        let a = field(ui, &mut c).on_hover_text("cells across");
        ui.label("×");
        let b = field(ui, &mut r).on_hover_text("cells down");
        let _ = (a, b);
        if (c, r) != (p.cols, p.rows) {
            // Re-grid from what is ON SCREEN, so the warp stays put; the
            // points are renumbered, so their curves go.
            let mut g = ev.clone();
            g.resize(c, r);
            *p = g;
            clear_point_curves(rcx.anim);
            st.pp = Default::default();
            st.pp.fit(p.count());
        }
        ui.checkbox(&mut p.smooth, "smooth").on_hover_text(
            "bend smoothly through the points (Catmull-Rom); off, straight lines between them",
        );
    });

    ui.horizontal(|ui| {
        ui.add_sized([widgets::LABEL_W, 16.0], egui::Label::new(egui::RichText::new("edit").small()));
        ui.selectable_value(&mut st.edit, Edit::Dst, "destination")
            .on_hover_text("move where the picture goes");
        ui.selectable_value(&mut st.edit, Edit::Src, "source")
            .on_hover_text("move what gets picked up");
        ui.selectable_value(&mut st.edit, Edit::Both, "both")
            .on_hover_text("move a point's source and destination together: re-places it without warping");
    });

    out.hud = widgets::slider(ui, rcx, "amount", &mut p.amount, 0.0, 1.0, &[Channel::GwAmount]);

    // The picked point's numbers.
    let n = p.count();
    let picked = st.pp.count();
    if let Some(k) = st.pp.first().filter(|k| *k < n) {
        let i = k as u16;
        let what = if picked > 1 { format!("pt {k} (+{} picked)", picked - 1) } else { format!("pt {k}") };
        ui.weak(what);
        let [sx, sy] = &mut p.src[k];
        out.hud = out.hud.or(widgets::drags(
            ui,
            rcx,
            "source",
            &mut [sx, sy],
            0.001,
            (0.0, 1.0),
            &[Channel::GwSrcX(i), Channel::GwSrcY(i)],
        ));
        let [dx, dy] = &mut p.dst[k];
        out.hud = out.hud.or(widgets::drags(
            ui,
            rcx,
            "destination",
            &mut [dx, dy],
            0.001,
            (0.0, 1.0),
            &[Channel::GwDstX(i), Channel::GwDstY(i)],
        ));
    } else {
        ui.weak("pick points in the viewer: click, S-click to add, drag a box");
    }

    ui.horizontal(|ui| {
        if ui.button("no warp").on_hover_text("destination back onto the source: nothing moves").clicked() {
            let mut g = ev.clone();
            g.dst = g.src.clone();
            out.replace = Some(g);
        }
        if ui.button("reset").on_hover_text("both grids back to the regular lattice").clicked() {
            out.replace = Some(GridWarpParams { smooth: ev.smooth, amount: ev.amount, ..GridWarpParams::identity(ev.cols, ev.rows) });
        }
        if ui.button("swap").on_hover_text("source and destination trade places: the warp turned round").clicked() {
            let mut g = ev.clone();
            std::mem::swap(&mut g.src, &mut g.dst);
            out.replace = Some(g);
        }
    });
    store(&ctx, node, st);
    out
}

/// What the gizmo needs from the viewer that it cannot work out itself.
pub struct Viewer<'a> {
    /// The node's format, in the pixels `to_screen` takes.
    pub size: (f32, f32),
    /// Node pixels to screen points and back, through whatever deforms the
    /// picture downstream.
    pub to_screen: &'a dyn Fn(Vec2) -> Pos2,
    pub from_screen: &'a dyn Fn(Pos2) -> Vec2,
    /// The viewer's rect: a click there off every handle drops the pick.
    pub rect: Rect,
    /// A plain primary drag on the viewer background this frame (no pan,
    /// no scrub): it bands points.
    pub banding: bool,
}

/// The lattice on the picture: both grids drawn through their own
/// interpolant (so a smooth lattice draws curved), a thread from each moved
/// point's source to its destination, and the points of the grid being
/// edited, picked and moved through [`pointpick`] (its box included).
/// Returns whether `gw` changed.
pub fn gizmo(
    ui: &mut egui::Ui,
    ink: &mut InkShapes,
    node: u64,
    gw: &mut GridWarpParams,
    v: &Viewer,
    hud_keys: &mut Vec<hudkeys::HudKey>,
) -> bool {
    let before = gw.clone();
    let n = gw.count();
    if gw.src.len() != n || gw.dst.len() != n {
        return false;
    }
    let ctx = ui.ctx().clone();
    let mut st = load(&ctx, node, n);
    let (w, h) = v.size;
    let px = |q: [f32; 2]| Vec2::new(q[0] * w, q[1] * h);
    let scr = |q: [f32; 2]| (v.to_screen)(px(q));

    // The two grids along their lattice lines, sampled through the
    // interpolant. The edited one strong, the other faint.
    let shown = GridWarpParams { amount: 1.0, ..gw.clone() };
    let steps = if gw.smooth { 8 } else { 1 };
    let line = |ink: &mut InkShapes, src: bool, alpha: f32| {
        let at = |g: [f32; 2]| {
            let e = shown.eval(g);
            scr(if src { e.src } else { e.dst })
        };
        for j in 0..=gw.rows {
            for i in 0..gw.cols {
                for s in 0..steps {
                    let (a, b) = (i as f32 + s as f32 / steps as f32, i as f32 + (s + 1) as f32 / steps as f32);
                    ink.seg(at([a, j as f32]), at([b, j as f32]), 1.0, alpha, false);
                }
            }
        }
        for i in 0..=gw.cols {
            for j in 0..gw.rows {
                for s in 0..steps {
                    let (a, b) = (j as f32 + s as f32 / steps as f32, j as f32 + (s + 1) as f32 / steps as f32);
                    ink.seg(at([i as f32, a]), at([i as f32, b]), 1.0, alpha, false);
                }
            }
        }
    };
    let edit_src = st.edit == Edit::Src;
    line(ink, !edit_src, 0.25);
    line(ink, edit_src, 0.7);
    // Each moved point: a thread from where it picks up to where it lands.
    for k in 0..n {
        let (s, d) = (gw.src[k], gw.dst[k]);
        if (s[0] - d[0]).abs() + (s[1] - d[1]).abs() > 1e-5 {
            ink.seg(scr(s), scr(d), 1.0, 0.5, false);
        }
    }

    let pts: Vec<Vec2> = (0..n).map(|k| px(if edit_src { gw.src[k] } else { gw.dst[k] })).collect();
    let frame = pointpick::Frame {
        to_screen: v.to_screen,
        from_screen: v.from_screen,
        rect: v.rect,
        banding: v.banding,
        id: ui.id().with(("gridwarp", node)),
    };
    let out = pointpick::run(ui, ink, &mut st.pp, &pts, &frame, hud_keys);
    if let Some(d) = out.drag {
        // The snapshot: both grids at the press, src then dst.
        if d.started || st.pp.snap.len() != 2 * n {
            st.pp.snap = gw.src.iter().chain(&gw.dst).copied().collect();
        }
        for k in (0..n).filter(|k| d.moving[*k]) {
            if st.edit != Edit::Dst {
                gw.src[k] = d.map.apply_norm(st.pp.snap[k], v.size);
            }
            if st.edit != Edit::Src {
                gw.dst[k] = d.map.apply_norm(st.pp.snap[n + k], v.size);
            }
        }
    }
    store(&ctx, node, st);
    *gw != before
}

/// A lattice with its middle point pulled: the warp itself.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let h = r * 0.8;
    painter.rect(
        Rect::from_center_size(c, Vec2::splat(h * 2.0)),
        2.0,
        fill,
        stroke,
        egui::StrokeKind::Middle,
    );
    let ink = Stroke::new((r * 0.1).max(1.0), crate::theme::node_ink());
    // A 2x2 lattice, its centre pulled up and right; each line bows
    // toward it.
    let pull = Vec2::new(h * 0.28, -h * 0.28);
    let at = |x: f32, y: f32| {
        let fall = (1.0 - x.abs()) * (1.0 - y.abs());
        c + Vec2::new(x * h * 0.8, y * h * 0.8) + pull * fall
    };
    const N: usize = 10;
    for k in [-1.0f32, 0.0, 1.0] {
        let across: Vec<Pos2> = (0..=N).map(|i| at(i as f32 / N as f32 * 2.0 - 1.0, k)).collect();
        let down: Vec<Pos2> = (0..=N).map(|i| at(k, i as f32 / N as f32 * 2.0 - 1.0)).collect();
        painter.add(egui::Shape::line(across, ink));
        painter.add(egui::Shape::line(down, ink));
    }
    painter.circle_filled(at(0.0, 0.0), (r * 0.14).max(1.0), crate::theme::node_ink());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channels_round_trip_and_name_their_point() {
        let mut p = GridWarpParams::identity(2, 2);
        set(&mut p, Channel::GwDstX(4), 0.7);
        set(&mut p, Channel::GwSrcY(3), 0.2);
        assert_eq!(get(&p, Channel::GwDstX(4)), Some(0.7));
        assert_eq!(get(&p, Channel::GwSrcY(3)), Some(0.2));
        assert_eq!(get(&p, Channel::GwDstX(9)), None, "no such point");
        assert_eq!(point_label(Channel::GwDstY(4)).as_deref(), Some("pt4 dest y"));
        assert_eq!(Channel::from_key(&Channel::GwDstY(4).key()), Some(Channel::GwDstY(4)));
        // A pre-prune lattice keyed its source positions as GwPt.
        assert_eq!(Channel::from_key("GwPtX:3"), Some(Channel::GwSrcX(3)));
    }

    #[test]
    fn every_point_is_a_channel_of_the_node() {
        let p = GridWarpParams::identity(3, 2);
        let chans = crate::anim::channels_for(&NodeKind::GridWarp(p.clone()));
        for i in 0..p.count() as u16 {
            for ch in point_channels(i) {
                assert!(chans.contains(&ch), "{ch:?} missing");
            }
        }
        assert!(chans.contains(&Channel::GwAmount));
    }

    /// The panel fits its tiles, both ways, with a point picked.
    #[test]
    fn the_panel_fits_its_tiles() {
        let mut p = GridWarpParams::default();
        let mut anim = Default::default();
        let ekind = NodeKind::GridWarp(p.clone());
        let (over, wide) = (std::cell::Cell::new(0.0f32), std::cell::Cell::new(0.0f32));
        let t = crate::widgets::TILE;
        let wpx = t * TILES as f32 + crate::widgets::TILE_GAP * (TILES as f32 - 1.0);
        let shot = crate::snapshot::shoot(wpx - 14.0, t - 14.0 - 22.0, |ui| {
            let mut st = load(ui.ctx(), 1, p.count());
            st.pp.pick[7] = true;
            store(ui.ctx(), 1, st);
            let out = egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                let mut rcx = RowCtx { node: 1, anim: &mut anim, ekind: &ekind, t: 0, hud_live: false, driven: Vec::new() };
                let _ = panel(ui, &mut rcx, &mut p);
            });
            over.set(out.content_size.y - out.inner_rect.height());
            wide.set(out.content_size.x - out.inner_rect.width());
        });
        shot.save("gridwarp-panel");
        assert!(over.get() <= 0.5, "overflows its tiles DOWN by {}pt", over.get());
        assert!(wide.get() <= 0.5, "overflows its tiles SIDEWAYS by {}pt", wide.get());
    }
}
