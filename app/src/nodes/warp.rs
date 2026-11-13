//! Warp's app half: the panel (the curve pairs, which side the viewer
//! edits, falloff and amount, the picked point's numbers), the viewer gizmo
//! (draw curves, pick and move their points through [`pointpick`], its
//! affine box included), the glyph and the channel bindings. The maths, the
//! WGSL and the params are `engine::nodes::warp`.
//!
//! Every point of every curve is an indexed channel, `WarpPt(pair, point,
//! lane)` with lanes source x, source y, destination x, destination y, so
//! they key, patch and HUD-slide like any value. Adding or removing a point
//! or a curve is STRUCTURE: it goes straight to the authored params, and a
//! removal slides the later points' curves down so each keeps its own
//! animation ([`remove_point`], [`remove_pair`]).

use crate::anim::{AnimCurve, Channel};
use crate::hudkeys;
use crate::ink::InkShapes;
use crate::pointpick;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Rect, Stroke, Vec2};
use engine::graph::{NodeKind, WarpPair, WarpParams, WARP_PAIRS};
use std::collections::BTreeMap;

/// The fixed channels. The points' own are indexed ([`point_channels`]).
pub const CHANNELS: &[Channel] = &[Channel::WarpAmount, Channel::WarpFalloff];

/// Two tiles: the curve row and the point's two pairs of numbers want width.
pub const TILES: usize = 2;

/// Point `i` of curve pair `pair`: source x, y, destination x, y.
pub fn point_channels(pair: u16, i: u16) -> [Channel; 4] {
    std::array::from_fn(|w| Channel::WarpPt(pair, i, w as u8))
}

pub fn get(p: &WarpParams, ch: Channel) -> Option<f32> {
    Some(match ch {
        Channel::WarpAmount => p.amount,
        Channel::WarpFalloff => p.falloff,
        Channel::WarpPt(pr, i, w) => {
            let pair = p.pairs.get(pr as usize)?;
            let side = if w < 2 { &pair.src } else { &pair.dst };
            side.get(i as usize)?[(w & 1) as usize]
        }
        _ => return None,
    })
}

pub fn set(p: &mut WarpParams, ch: Channel, v: f32) {
    match ch {
        Channel::WarpAmount => p.amount = v,
        Channel::WarpFalloff => p.falloff = v,
        Channel::WarpPt(pr, i, w) => {
            if let Some(pair) = p.pairs.get_mut(pr as usize) {
                let side = if w < 2 { &mut pair.src } else { &mut pair.dst };
                if let Some(q) = side.get_mut(i as usize) {
                    q[(w & 1) as usize] = v;
                }
            }
        }
        _ => {}
    }
}

/// The curve editor's name for a fixed channel.
pub fn label(ch: Channel) -> Option<&'static str> {
    match ch {
        Channel::WarpAmount => Some("warp amount"),
        Channel::WarpFalloff => Some("warp falloff"),
        _ => None,
    }
}

/// The curve editor's name for a point's channel.
pub fn point_label(ch: Channel) -> Option<String> {
    let Channel::WarpPt(pr, i, w) = ch else { return None };
    let what = ["source x", "source y", "dest x", "dest y"][(w & 3) as usize];
    Some(format!("curve{} pt{i} {what}", pr + 1))
}

/// Every channel of the node's points.
pub fn all_point_channels(p: &WarpParams) -> Vec<Channel> {
    let mut v = Vec::new();
    for (pr, pair) in p.pairs.iter().enumerate().take(u16::MAX as usize) {
        for i in 0..pair.len().min(u16::MAX as usize) {
            v.extend(point_channels(pr as u16, i as u16));
        }
    }
    v
}

/// Move every point curve through `f` (`None` drops it).
fn remap(anim: &mut BTreeMap<Channel, AnimCurve>, f: impl Fn(u16, u16) -> Option<(u16, u16)>) {
    let moved: Vec<(Channel, AnimCurve)> = anim
        .iter()
        .filter(|(c, _)| matches!(c, Channel::WarpPt(..)))
        .map(|(c, k)| (*c, k.clone()))
        .collect();
    for (c, _) in &moved {
        anim.remove(c);
    }
    for (c, k) in moved {
        if let Channel::WarpPt(pr, i, w) = c {
            if let Some((pr, i)) = f(pr, i) {
                anim.insert(Channel::WarpPt(pr, i, w), k);
            }
        }
    }
}

/// Remove point `i` of pair `pair` from both sides, sliding the later
/// points' curves down one.
pub fn remove_point(p: &mut WarpParams, anim: &mut BTreeMap<Channel, AnimCurve>, pair: usize, i: usize) {
    let Some(pr) = p.pairs.get_mut(pair) else { return };
    if i >= pr.len() {
        return;
    }
    pr.src.remove(i);
    pr.dst.remove(i);
    let (pair, i) = (pair as u16, i as u16);
    remap(anim, |q, k| {
        if q != pair || k < i {
            Some((q, k))
        } else if k == i {
            None
        } else {
            Some((q, k - 1))
        }
    });
}

/// Remove a whole pair, sliding the later pairs' curves down one.
pub fn remove_pair(p: &mut WarpParams, anim: &mut BTreeMap<Channel, AnimCurve>, pair: usize) {
    if pair >= p.pairs.len() {
        return;
    }
    p.pairs.remove(pair);
    let pair = pair as u16;
    remap(anim, |q, k| match q.cmp(&pair) {
        std::cmp::Ordering::Less => Some((q, k)),
        std::cmp::Ordering::Equal => None,
        std::cmp::Ordering::Greater => Some((q - 1, k)),
    });
}

/// Which side the viewer's handles move.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Edit {
    /// Where things go: the warp itself.
    #[default]
    Dst,
    /// What gets picked up.
    Src,
    /// Both together: re-place a point without warping.
    Both,
}

/// The per-node state the panel and the gizmo share.
#[derive(Clone, Debug, Default)]
struct Ui {
    edit: Edit,
    /// The curve pair new points go on.
    pair: usize,
    pp: pointpick::Pick,
}

fn ui_id(node: u64) -> egui::Id {
    egui::Id::new(("warp-ui", node))
}

fn load(ctx: &egui::Context, node: u64) -> Ui {
    ctx.data(|d| d.get_temp::<Ui>(ui_id(node))).unwrap_or_default()
}

fn store(ctx: &egui::Context, node: u64, s: Ui) {
    ctx.data_mut(|d| d.insert_temp(ui_id(node), s));
}

/// (pair, point) of every point, in the order the gizmo lists them.
fn index(p: &WarpParams) -> Vec<(usize, usize)> {
    p.pairs.iter().enumerate().flat_map(|(pr, pair)| (0..pair.len()).map(move |i| (pr, i))).collect()
}

/// Which (pair, point)s are picked (`--test-warp` reads it).
pub fn picked(ctx: &egui::Context, node: u64, p: &WarpParams) -> Vec<(usize, usize)> {
    let mut s = load(ctx, node);
    let ix = index(p);
    s.pp.fit(ix.len());
    ix.into_iter().zip(s.pp.pick).filter(|(_, b)| *b).map(|(k, _)| k).collect()
}

/// Replace the pick (`--test-warp` sets one).
pub fn set_pick(ctx: &egui::Context, node: u64, p: &WarpParams, pick: &[(usize, usize)]) {
    let mut s = load(ctx, node);
    let ix = index(p);
    let flat: Vec<usize> = ix.iter().enumerate().filter(|(_, k)| pick.contains(k)).map(|(f, _)| f).collect();
    s.pp.set(ix.len(), &flat);
    store(ctx, node, s);
}

/// Where the pick's box drew its grips last frame.
pub fn box_hot(ctx: &egui::Context, node: u64) -> Option<pointpick::BoxHot> {
    load(ctx, node).pp.hot
}

/// What the panel hands back.
#[derive(Default)]
pub struct PanelOut {
    pub hud: Option<HudSlider>,
    /// Every point replaced (no warp, swap): written back through the keyed
    /// path by the caller, so an animated point gets a key.
    pub replace: Option<WarpParams>,
    /// A pair to remove once the node borrow is released.
    pub remove_pair: Option<usize>,
}

pub fn panel(ui: &mut egui::Ui, rcx: &mut RowCtx, p: &mut WarpParams) -> PanelOut {
    let mut out = PanelOut::default();
    let node = rcx.node;
    let ev = match rcx.ekind {
        NodeKind::Warp(g) => g.clone(),
        _ => p.clone(),
    };
    let ctx = ui.ctx().clone();
    let mut st = load(&ctx, node);

    // The curves: one chip each, the picked one takes new points.
    ui.horizontal_wrapped(|ui| {
        ui.add_sized([widgets::LABEL_W, 16.0], egui::Label::new(egui::RichText::new("curves").small()));
        for (k, pair) in p.pairs.iter().enumerate() {
            let on = st.pair == k;
            if ui
                .add(egui::Button::selectable(on, format!("{}", k + 1)))
                .on_hover_text(format!("{} points; S-click in the viewer adds to this curve", pair.len()))
                .clicked()
            {
                st.pair = k;
            }
        }
        if ui
            .add_enabled(p.pairs.len() < WARP_PAIRS, egui::Button::new("+"))
            .on_hover_text("a new curve: S-click in the viewer to draw it")
            .clicked()
        {
            p.pairs.push(WarpPair::default());
            st.pair = p.pairs.len() - 1;
        }
    });
    if let Some(pair) = p.pairs.get_mut(st.pair) {
        ui.horizontal(|ui| {
            ui.add_space(widgets::LABEL_W + ui.spacing().item_spacing.x);
            ui.checkbox(&mut pair.closed, "closed").on_hover_text("the curve loops back to its first point");
            if ui.button("remove curve").clicked() {
                out.remove_pair = Some(st.pair);
            }
        });
    }

    ui.horizontal(|ui| {
        ui.add_sized([widgets::LABEL_W, 16.0], egui::Label::new(egui::RichText::new("edit").small()));
        ui.selectable_value(&mut st.edit, Edit::Dst, "destination").on_hover_text("move where the picture goes");
        ui.selectable_value(&mut st.edit, Edit::Src, "source").on_hover_text("move what gets picked up");
        ui.selectable_value(&mut st.edit, Edit::Both, "both")
            .on_hover_text("move a point's source and destination together: re-places it without warping");
    });
    out.hud = widgets::slider(ui, rcx, "falloff", &mut p.falloff, 1.0, 1000.0, &[Channel::WarpFalloff]);
    out.hud = out.hud.or(widgets::slider(ui, rcx, "amount", &mut p.amount, 0.0, 1.0, &[Channel::WarpAmount]));

    // The first picked point's numbers.
    let ix = index(p);
    st.pp.fit(ix.len());
    if let Some(&(pr, i)) = st.pp.first().and_then(|f| ix.get(f)) {
        let n = st.pp.count();
        ui.weak(if n > 1 { format!("curve {} pt {i} (+{} picked)", pr + 1, n - 1) } else { format!("curve {} pt {i}", pr + 1) });
        let c = point_channels(pr as u16, i as u16);
        let [sx, sy] = &mut p.pairs[pr].src[i];
        out.hud = out.hud.or(widgets::drags(ui, rcx, "source", &mut [sx, sy], 0.001, (0.0, 1.0), &c[0..2]));
        let [dx, dy] = &mut p.pairs[pr].dst[i];
        out.hud = out.hud.or(widgets::drags(ui, rcx, "destination", &mut [dx, dy], 0.001, (0.0, 1.0), &c[2..4]));
    } else {
        ui.weak("S-click in the viewer draws a curve · D-click removes a point");
    }

    ui.horizontal(|ui| {
        if ui.button("no warp").on_hover_text("every destination back onto its source").clicked() {
            let mut g = ev.clone();
            for pair in &mut g.pairs {
                pair.dst = pair.src.clone();
            }
            out.replace = Some(g);
        }
        if ui.button("swap").on_hover_text("sources and destinations trade places: the warp turned round").clicked() {
            let mut g = ev.clone();
            for pair in &mut g.pairs {
                std::mem::swap(&mut pair.src, &mut pair.dst);
            }
            out.replace = Some(g);
        }
    });
    store(&ctx, node, st);
    out
}

/// The viewer as the gizmo needs it.
pub struct Viewer<'a> {
    /// The node's format, in the pixels `to_screen` takes.
    pub size: (f32, f32),
    pub to_screen: &'a dyn Fn(Vec2) -> Pos2,
    pub from_screen: &'a dyn Fn(Pos2) -> Vec2,
    pub rect: Rect,
    pub banding: bool,
    /// The falloff in the same pixels (the proxy applied): where a new
    /// point's source is placed so it does not jump.
    pub falloff: f32,
}

/// Structural edits the gizmo asks for, applied by the caller to the
/// AUTHORED params (they change how many points there are).
#[derive(Default)]
pub struct Edits {
    /// Append a point (src, dst) to pair `.0`, creating it if it is new.
    pub add: Option<(usize, [f32; 2], [f32; 2])>,
    pub remove: Option<(usize, usize)>,
}

/// Both sides of every curve on the picture (the edited side strong), a
/// thread from each moved point's source to its destination, and the
/// points of the edited side, picked and moved through [`pointpick`].
/// S-click on the picture adds a point to the current curve, D-click on a
/// point removes it. Returns whether `w` changed, and the structural edits.
pub fn gizmo(
    ui: &mut egui::Ui,
    ink: &mut InkShapes,
    node: u64,
    w: &mut WarpParams,
    v: &Viewer,
    hud_keys: &mut Vec<hudkeys::HudKey>,
) -> (bool, Edits) {
    let before = w.clone();
    let mut edits = Edits::default();
    let ctx = ui.ctx().clone();
    let mut st = load(&ctx, node);
    let (sw, sh) = v.size;
    let px = |q: [f32; 2]| Vec2::new(q[0] * sw, q[1] * sh);
    let scr = |q: [f32; 2]| (v.to_screen)(px(q));
    let edit_src = st.edit == Edit::Src;

    for (k, pair) in w.pairs.iter().enumerate() {
        let cur = k == st.pair;
        for (src, strong) in [(!edit_src, false), (edit_src, true)] {
            let s = pair.samples(!src);
            let a = if strong { if cur { 0.9 } else { 0.6 } } else { 0.25 };
            for seg in s.windows(2) {
                ink.seg(scr(seg[0]), scr(seg[1]), if strong { 1.5 } else { 1.0 }, a, false);
            }
        }
        for i in 0..pair.len() {
            let (s, d) = (pair.src[i], pair.dst[i]);
            if (s[0] - d[0]).abs() + (s[1] - d[1]).abs() > 1e-5 {
                ink.seg(scr(s), scr(d), 1.0, 0.4, false);
            }
        }
    }

    let ix = index(w);
    let pts: Vec<Vec2> =
        ix.iter().map(|&(pr, i)| px(if edit_src { w.pairs[pr].src[i] } else { w.pairs[pr].dst[i] })).collect();
    let frame = pointpick::Frame {
        to_screen: v.to_screen,
        from_screen: v.from_screen,
        rect: v.rect,
        banding: v.banding,
        id: ui.id().with(("warp", node)),
    };
    let out = pointpick::run(ui, ink, &mut st.pp, &pts, &frame, hud_keys);
    hud_keys.push(hudkeys::key("S", "click the picture: add a point", egui::Key::S));
    hud_keys.push(hudkeys::key("D", "click a point: remove it", egui::Key::D));
    if let Some((f, _)) = out.clicked {
        let (pr, i) = ix[f];
        st.pair = pr;
        if ui.input(|inp| inp.key_down(egui::Key::D)) {
            edits.remove = Some((pr, i));
        }
    }
    if let Some(d) = out.drag {
        // The snapshot: every point at the press, src then dst, flat.
        let n = ix.len();
        if d.started || st.pp.snap.len() != 2 * n {
            st.pp.snap = ix.iter().map(|&(pr, i)| w.pairs[pr].src[i]).chain(ix.iter().map(|&(pr, i)| w.pairs[pr].dst[i])).collect();
        }
        for (f, &(pr, i)) in ix.iter().enumerate() {
            if !d.moving[f] {
                continue;
            }
            if st.edit != Edit::Dst {
                w.pairs[pr].src[i] = d.map.apply_norm(st.pp.snap[f], v.size);
            }
            if st.edit != Edit::Src {
                w.pairs[pr].dst[i] = d.map.apply_norm(st.pp.snap[n + f], v.size);
            }
        }
    }
    // S-click on the picture, off every handle: a new point on the current
    // curve. Its other side goes where the warp already puts it, so adding
    // a point moves nothing.
    let clicked = ui.input(|i| i.pointer.primary_clicked() && i.key_down(egui::Key::S));
    if clicked && !out.hot && !st.pp.grabbing() {
        if let Some(p) = ctx.pointer_latest_pos().filter(|p| v.rect.contains(*p)) {
            let o = (v.from_screen)(p);
            let here = [o.x / sw.max(1.0), o.y / sh.max(1.0)];
            let full = WarpParams { amount: 1.0, ..w.clone() };
            let (src, dst) = if edit_src {
                (here, here)
            } else {
                let s = full.sample_at([o.x, o.y], [sw, sh], v.falloff);
                ([s[0] / sw.max(1.0), s[1] / sh.max(1.0)], here)
            };
            let pair = st.pair.min(w.pairs.len());
            edits.add = Some((pair, src, dst));
            st.pair = pair;
        }
    }
    store(&ctx, node, st);
    (*w != before, edits)
}

/// Apply the gizmo's structural edits to the authored params.
pub fn apply_edits(p: &mut WarpParams, anim: &mut BTreeMap<Channel, AnimCurve>, e: Edits) {
    if let Some((pr, i)) = e.remove {
        remove_point(p, anim, pr, i);
    }
    if let Some((pr, s, d)) = e.add {
        if pr >= p.pairs.len() && p.pairs.len() < WARP_PAIRS {
            p.pairs.push(WarpPair::default());
        }
        if let Some(pair) = p.pairs.get_mut(pr) {
            pair.src.push(s);
            pair.dst.push(d);
        }
    }
}

/// Two curves, one bent off the other, and the tick between them.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = Stroke::new((r * 0.1).max(1.0), crate::theme::node_ink());
    let faint = Stroke::new((r * 0.08).max(0.8), crate::theme::node_ink().gamma_multiply(0.5));
    const N: usize = 12;
    let curve = |bend: f32| -> Vec<Pos2> {
        (0..=N)
            .map(|i| {
                let t = i as f32 / N as f32 * 2.0 - 1.0;
                c + Vec2::new(t * r * 0.6, r * 0.15 - (1.0 - t * t) * r * bend)
            })
            .collect()
    };
    painter.add(egui::Shape::line(curve(0.0), faint));
    painter.add(egui::Shape::line(curve(0.5), ink));
    painter.line_segment([c + Vec2::new(0.0, r * 0.15), c + Vec2::new(0.0, r * 0.15 - r * 0.5)], faint);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_pairs() -> WarpParams {
        let pair = |y: f32| WarpPair { src: vec![[0.1, y], [0.5, y], [0.9, y]], dst: vec![[0.1, y], [0.5, y + 0.1], [0.9, y]], closed: false };
        WarpParams { pairs: vec![pair(0.3), pair(0.6)], ..Default::default() }
    }

    #[test]
    fn channels_round_trip_and_name_their_point() {
        let mut p = two_pairs();
        set(&mut p, Channel::WarpPt(1, 2, 3), 0.77);
        assert_eq!(get(&p, Channel::WarpPt(1, 2, 3)), Some(0.77));
        assert_eq!(get(&p, Channel::WarpPt(0, 1, 0)), Some(0.5));
        assert_eq!(get(&p, Channel::WarpPt(4, 0, 0)), None);
        assert_eq!(point_label(Channel::WarpPt(1, 2, 3)).as_deref(), Some("curve2 pt2 dest y"));
        assert_eq!(Channel::from_key(&Channel::WarpPt(1, 2, 3).key()), Some(Channel::WarpPt(1, 2, 3)));
        let chans = crate::anim::channels_for(&NodeKind::Warp(p.clone()));
        for ch in all_point_channels(&p) {
            assert!(chans.contains(&ch), "{ch:?}");
        }
    }

    #[test]
    fn removing_slides_the_later_curves_down() {
        let mut p = two_pairs();
        let mut anim = BTreeMap::new();
        let curve = |v: f32| {
            let mut c = AnimCurve::default();
            c.set_key(0, v);
            c
        };
        anim.insert(Channel::WarpPt(0, 2, 2), curve(0.9));
        anim.insert(Channel::WarpPt(0, 1, 2), curve(0.5));
        anim.insert(Channel::WarpPt(1, 0, 0), curve(0.1));
        remove_point(&mut p, &mut anim, 0, 1);
        assert_eq!(p.pairs[0].len(), 2);
        assert!(anim.contains_key(&Channel::WarpPt(0, 1, 2)), "pt2's curve slid to pt1");
        assert_eq!(anim[&Channel::WarpPt(0, 1, 2)].eval(0), Some(0.9));
        assert!(anim.contains_key(&Channel::WarpPt(1, 0, 0)), "another curve's points are untouched");
        remove_pair(&mut p, &mut anim, 0);
        assert_eq!(p.pairs.len(), 1);
        assert!(anim.contains_key(&Channel::WarpPt(0, 0, 0)) && anim.len() == 1, "{anim:?}");
    }

    /// The panel fits its tiles, both ways, with curves and a point picked.
    #[test]
    fn the_panel_fits_its_tiles() {
        let mut p = two_pairs();
        let mut anim = Default::default();
        let ekind = NodeKind::Warp(p.clone());
        let (over, wide) = (std::cell::Cell::new(0.0f32), std::cell::Cell::new(0.0f32));
        let t = crate::widgets::TILE;
        let wpx = t * TILES as f32 + crate::widgets::TILE_GAP * (TILES as f32 - 1.0);
        let shot = crate::snapshot::shoot(wpx - 14.0, t - 14.0 - 22.0, |ui| {
            set_pick(ui.ctx(), 1, &p, &[(1, 1)]);
            let out = egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                let mut rcx = RowCtx { node: 1, anim: &mut anim, ekind: &ekind, t: 0, hud_live: false, driven: Vec::new() };
                let _ = panel(ui, &mut rcx, &mut p);
            });
            over.set(out.content_size.y - out.inner_rect.height());
            wide.set(out.content_size.x - out.inner_rect.width());
        });
        shot.save("warp-panel");
        assert!(over.get() <= 0.5, "overflows its tiles DOWN by {}pt", over.get());
        assert!(wide.get() <= 0.5, "overflows its tiles SIDEWAYS by {}pt", wide.get());
    }
}
