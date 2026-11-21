//! The LumaKey's band panel.
//!
//! There is no bespoke widget here any more. The partition is authored in
//! `curveui`, the shared curve editor, because everything a band strip
//! wanted — a backdrop painted in the colours of the axis, filled ghosts
//! for the other bands, points that move together — turned out to be
//! something every curve editor wanted, and those now live in `curveui` as
//! API features rather than in one node's private widget.
//!
//! What is left here is the part that is genuinely about BANDS: turning a
//! partition into curves, and reading an edited curve back into the
//! partition so that moving a boundary moves the band on both sides of it.
//!
//! Owned by instance E (see .agents/board.md).

use crate::curveui;
use egui::Color32;
use engine::bands::BandCurve;
#[cfg(test)]
use engine::bands::BandRanges;

/// The linear/smooth pair, as a chip row. Every band node needs it and it is
/// one control, not a combo box worth of ceremony.
pub fn curve_chips(ui: &mut egui::Ui, c: &mut BandCurve) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        for k in BandCurve::ALL {
            if ui
                .selectable_label(*c == k, egui::RichText::new(k.name()).small())
                .clicked()
                && *c != k
            {
                *c = k;
                changed = true;
            }
        }
    });
    changed
}

/// What the LumaKey panel wants the shell to do afterwards.
#[derive(Default)]
pub struct LumaOut {
    pub changed: bool,
    /// Show this band's matte in the viewer.
    pub view_band: Option<u32>,
    /// A band was split in two; this is the NEW band's index (the upper
    /// half). Whatever the caller keeps per band has to grow to match.
    pub split: Option<usize>,
    /// Band `i` was dissolved into its neighbour; the caller's per-band
    /// state for `i` goes with it.
    pub dissolved: Option<usize>,
    /// The whole partition was asked away (`BandOpts::removable`).
    pub remove_all: bool,
}

/// How the band panel is being used. The partition and its editor are the
/// same for a keyer and for a Grade's zones; what differs is the row under
/// the plot and what a band is called.
pub struct BandOpts<'a> {
    /// The keyer's own controls: per-band invert, and "view this matte".
    pub keyer: bool,
    /// Chip labels, one per band. `None` = the keyer's matte labels.
    pub labels: Option<&'a [String]>,
    /// Hint in the band's name field.
    pub name_hint: &'static str,
    /// Offer to drop the whole partition (reported as `LumaOut::remove_all`).
    pub removable: bool,
    /// Stop adding points at this many on one band. The Grade packs a zone's
    /// curve into a FIXED span, and a point that would not fit is worse left
    /// unadded than silently dropped off the top of the band.
    pub max_points: Option<usize>,
    /// Room the CALLER draws in under this panel, in points at zoom 1
    /// (~24 per row of small controls, spacing included). The PLOT gives it
    /// up: this panel sizes itself to fill exactly what the tile has left,
    /// so anything a caller adds below without reserving it lands in the
    /// tile's scroll area, where it cannot even be clicked. A control about
    /// the selected band needs no row at all: see [`luma_panel_with`].
    pub reserve: f32,
}

impl BandOpts<'static> {
    /// The LumaKey's own panel.
    pub const KEYER: BandOpts<'static> = BandOpts {
        keyer: true,
        labels: None,
        name_hint: "port label",
        removable: false,
        max_points: None,
        reserve: 0.0,
    };
}

/// The LumaKey's band panel, built on the SHARED curve editor.
///
/// It was a bespoke strip; it is now `curveui` with a backdrop, filled
/// ghosts and mirrored points, because everything the strip did specially
/// was something every curve editor wanted anyway. What it buys by moving:
/// decoupled x/y pan and zoom, marquee selection, the scale box, and
/// arbitrary points — a luma axis in a scene-linear comp runs to 2.0 and
/// beyond, and being able to zoom into the toe is the difference between
/// authoring a shadow band and guessing at one.
///
/// The **lock** is the model's, not the widget's: corner positions are read
/// back into the partition's edges and every band is re-derived, so moving
/// a boundary moves the band on each side of it because there is only one
/// boundary. Within the edited band the two corners of one boundary are a
/// mirror group, so dragging it carries its crossfade rather than smearing
/// it; alt-drag one corner alone to set the softness.
/// Room the row under the plot needs (the invert/name/view strip), plus the
/// spacing either side of it.
const BOTTOM_ROW: f32 = 30.0;
/// Smallest plot worth showing. Above this the plot takes exactly the room
/// the tile has left, so a panel that FITS shows no scrollbar; below it the
/// plot keeps this height and the grid scrolls, which is what a short docked
/// strip should do rather than shrink the editor to nothing. It was 150,
/// which a Grade's one-tile ranges page (206pt under its tab strip) could not
/// afford: the floor held and the band row went below the fold. The plot
/// zooms and pans, so 100 is still an instrument.
const PLOT_MIN: f32 = 100.0;

pub fn luma_panel(
    ui: &mut egui::Ui,
    k: &mut engine::graph::LumaKeyParams,
    node: u64,
    sel: &mut usize,
    zoom: f32,
    opts: &BandOpts<'_>,
) -> LumaOut {
    luma_panel_with(ui, k, node, sel, zoom, opts, |_, _| {})
}

/// [`luma_panel`] with `tail` drawn in the selected band's own row under
/// the plot, where the keyer keeps its "view" button. A control about the
/// selected band belongs there rather than in a row of its own: a row under
/// the panel is room the plot has to give up ([`BandOpts::reserve`]), and a
/// Grade's one-tile page had none to spare — its preview row landed below
/// the fold. `tail` is handed the band selected after this frame's edits.
pub fn luma_panel_with(
    ui: &mut egui::Ui,
    k: &mut engine::graph::LumaKeyParams,
    node: u64,
    sel: &mut usize,
    zoom: f32,
    opts: &BandOpts<'_>,
    tail: impl FnOnce(&mut egui::Ui, usize),
) -> LumaOut {
    use engine::graph::LUMA_SPAN;
    let mut out = LumaOut::default();
    let Some(r) = k.ranges.clone() else { return out };
    let n = r.band_count();
    *sel = (*sel).min(n.saturating_sub(1));
    let span = LUMA_SPAN;

    // Wrapped: a key can carry many bands, and a row that overflows its
    // tile drags the whole grid's layout out of shape.
    ui.horizontal_wrapped(|ui| {
        for i in 0..n {
            let label = match opts.labels.and_then(|l| l.get(i)) {
                Some(l) => l.clone(),
                None => k.bands.get(i).map(|b| b.label(i)).unwrap_or_default(),
            };
            if ui.selectable_label(*sel == i, egui::RichText::new(label).small()).clicked() {
                *sel = i;
            }
        }
        ui.separator();
        let mut c = r.curve;
        if curve_chips(ui, &mut c) {
            k.ranges.as_mut().unwrap().curve = c;
            k.sync_from_ranges();
            out.changed = true;
        }
        let split = ui.small_button("+");
        crate::widgets::note_rect(ui, egui::Id::new("band-split-rect"), split.rect);
        if split.on_hover_text("split this band in two").clicked() {
            // The model does the bookkeeping: the new band is the upper
            // half, and every band above keeps its own name.
            out.split = Some(k.split_band(*sel));
            out.changed = true;
        }
        let dissolve = ui.add_enabled(n > 1, egui::Button::new("−").small());
        crate::widgets::note_rect(ui, egui::Id::new("band-dissolve-rect"), dissolve.rect);
        if dissolve.on_hover_text("dissolve it into its neighbour").clicked() {
            k.dissolve_band(*sel);
            out.dissolved = Some(*sel);
            *sel = (*sel).min(k.ranges.as_ref().unwrap().band_count() - 1);
            out.changed = true;
        }
    });

    // Every other band, filled, behind the one being edited.
    let ghost_pts: Vec<(usize, Vec<[f32; 2]>)> = (0..n)
        .filter(|i| *i != *sel)
        .map(|i| (i, k.bands.get(i).map(|b| b.curve()).unwrap_or_default()))
        .collect();
    let ghosts: Vec<curveui::Ghost<'_>> = ghost_pts
        .iter()
        .map(|(i, p)| curveui::Ghost {
            pts: p,
            color: band_tint(*i, n).gamma_multiply(0.5),
            fill: true,
        })
        .collect();

    let (mut pts, roles) = layout(k, *sel);
    // A plain drag moves the point you grabbed. SHIFT-drag carries the
    // whole crossfade — both corners and its slope points — as one object.
    // That way round because the single-point move is the one you want most
    // of the time, and having both move by default made every small
    // correction feel like it was fighting back.
    let mirror = fade_groups(&roles, !ui.input(|i| i.modifiers.shift));
    // The band's JOINTS. A boundary corner is at full coverage or at none —
    // there is no such thing as a boundary at 0.4 — so these move in x only
    // and cannot be deleted. Dragging one off its rail was the easiest way
    // to make a near-vertical drop that then had to be repaired by hand.
    let pinned = structural(&roles);
    // A new point can go on either slope (it shapes that boundary, so it
    // lands in the neighbour's curve too) or in the plateau (the band's own
    // bias). Where the consumer packs a fixed number of points, every band
    // the point lands in needs room for it.
    let room = |len: usize| opts.max_points.is_none_or(|m| len < m);
    let len_of = |i: Option<usize>| i.and_then(|i| k.bands.get(i)).map_or(0, |b| b.curve().len());
    let room_here = room(pts.len());
    let room_below = room(len_of(sel.checked_sub(1)));
    let room_above = room(len_of(Some(*sel + 1)));
    let (rpts, rroles) = (pts.clone(), roles.clone());
    let add_ok = move |x: f32| match region(&rpts, &rroles, x) {
        Some(Region::Plateau) => room_here,
        Some(Region::Slope(false)) => room_here && room_below,
        Some(Region::Slope(true)) => room_here && room_above,
        None => false,
    };

    let cfg = curveui::CurveCfg {
        // Whatever the tile has left after the row below, floored so a
        // short pane scrolls instead of squashing this to nothing.
        size: egui::vec2(
            ui.available_width(),
            (ui.available_height() - (BOTTOM_ROW + opts.reserve) * zoom)
                .max(PLOT_MIN * zoom),
        ),
        view: [span.0, span.1, 0.0, 1.0],
        interp: match r.curve {
            BandCurve::Linear => engine::graph::CurveInterp::Linear,
            BandCurve::Smooth => engine::graph::CurveInterp::Smooth,
        },
        min_points: 2,
        pin_ends: false,
        clamp_y: Some((0.0, 1.0)),
        tint: band_tint(*sel, n),
        backdrop: Some(&|x, _y| luma_tone(x, span)),
        pin_y: &pinned,
        add_ok: Some(&add_ok),
        mirror: &mirror,
        // Points stop at their neighbours: a corner swapping past a slope
        // point (or the far corner) would silently re-mean both.
        keep_order: true,
        ..Default::default()
    };
    let res = curveui::curve_edit(ui, ui.id().with(("luma-curve", node)), &cfg, &ghosts, &mut pts);
    if res.changed {
        read_back(k, *sel, &pts);
        out.changed = true;
    }

    let at = *sel;
    if let Some(b) = k.bands.get_mut(*sel) {
        ui.horizontal(|ui| {
            if opts.keyer && ui.checkbox(&mut b.invert, "invert").changed() {
                out.changed = true;
            }
            ui.add(
                egui::TextEdit::singleline(&mut b.name)
                    .desired_width(90.0)
                    .hint_text(opts.name_hint),
            );
            if opts.keyer
                && ui
                    .small_button("view")
                    .on_hover_text("show this band's matte in the viewer")
                    .clicked()
            {
                out.view_band = Some(*sel as u32);
            }
            ui.add_space(6.0);
            tail(ui, at);
            // Pinned to the right edge: what the tail draws may come and go
            // (the Grade's "view this node" hint), and a destructive button
            // must not slide under the pointer when it does.
            if opts.removable {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .small_button("remove zones")
                        .on_hover_text("back to one correction for the whole frame")
                        .clicked()
                    {
                        out.remove_all = true;
                    }
                });
            }
        });
    }
    out
}

/// The grey a luma value shows as, for the backdrop.
fn luma_tone(x: f32, span: (f32, f32)) -> Color32 {
    let t = ((x - span.0) / (span.1 - span.0)).clamp(0.0, 1.0);
    Color32::from_gray((t.powf(0.6) * 205.0) as u8 + 16)
}

/// A band's colour: the ONE palette (`engine::graph::band_srgb`), shared
/// with a Grade's zone maps so the plot is the viewer's legend. Lifted 15%
/// toward white here so a saturated blue still reads as a stroke over the
/// plot's near-black shadows end; same hues, so the legend still reads. It
/// used to be a tint of its own — lavender, grey, salmon — which told the
/// middle bands apart badly and ended in an orange.
fn band_tint(i: usize, n: usize) -> Color32 {
    let [r, g, b] = engine::graph::band_srgb(i, n).map(|v| v + ((255 - v) as f32 * 0.15) as u8);
    Color32::from_rgb(r, g, b)
}

/// What a point of the edited band's curve IS. The curve the editor shows
/// is the partition's corners and slope points plus the band's own plateau
/// bias, and reading an edit back needs to know which is which — by role,
/// never by guessing from heights (a slope point may sit at 0 or 1 too).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Role {
    /// An end of the axis: an end band's outer point. Structural.
    End,
    /// A crossfade corner, on the boundary below the band (`above: false`)
    /// or above it; `hi` is the corner at the higher luma. Structural.
    Corner { above: bool, hi: bool },
    /// A point shaping that boundary's slope ([`engine::bands::Edge::shape`]).
    Slope { above: bool },
    /// The user's own bias point in the plateau.
    Bias,
}

impl Role {
    fn structural(self) -> bool {
        matches!(self, Role::End | Role::Corner { .. })
    }
}

/// Where on the edited band's curve a new point would land.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Region {
    /// On the slope of the boundary below (`false`) or above (`true`).
    Slope(bool),
    Plateau,
}

/// The edited band's curve, and what each point is.
fn layout(k: &engine::graph::LumaKeyParams, sel: usize) -> (Vec<[f32; 2]>, Vec<Role>) {
    let Some(r) = k.ranges.as_ref() else { return (Vec::new(), Vec::new()) };
    let n = r.band_count();
    let sel = sel.min(n.saturating_sub(1));
    let base = engine::bands::band_curve(r, sel);
    let shape_len = |i: usize| r.edges.get(i).filter(|e| e.width > 0.0).map_or(0, |e| e.shape.len());
    let mut roles: Vec<Role> = Vec::with_capacity(base.len());
    let has_below = !r.edges.is_empty() && n > 1 && sel > 0;
    let has_above = !r.edges.is_empty() && n > 1 && sel < r.edges.len();
    if has_below {
        roles.push(Role::Corner { above: false, hi: false });
        roles.extend(std::iter::repeat_n(Role::Slope { above: false }, shape_len(sel - 1)));
        roles.push(Role::Corner { above: false, hi: true });
    } else {
        roles.push(Role::End);
    }
    let split = roles.len();
    if has_above {
        roles.push(Role::Corner { above: true, hi: false });
        roles.extend(std::iter::repeat_n(Role::Slope { above: true }, shape_len(sel)));
        roles.push(Role::Corner { above: true, hi: true });
    } else {
        roles.push(Role::End);
    }
    if roles.len() != base.len() {
        // Cannot happen while `band_curve` emits what it documents; show the
        // curve as all structure rather than mislabel it.
        return (base.clone(), vec![Role::End; base.len()]);
    }
    // The band's bias: whatever of its points sits strictly inside the
    // plateau, which no derived point ever does.
    let (pl, ph) = (base[split - 1][0], base[split][0]);
    let mut bias: Vec<[f32; 2]> = k
        .bands
        .get(sel)
        .map(|b| b.points.iter().copied().filter(|p| p[0] > pl && p[0] < ph).collect())
        .unwrap_or_default();
    bias.sort_by(|a, c| a[0].partial_cmp(&c[0]).unwrap_or(std::cmp::Ordering::Equal));
    let mut pts = base[..split].to_vec();
    pts.extend(bias.iter().copied());
    pts.extend(base[split..].iter().copied());
    let mut out_roles = roles[..split].to_vec();
    out_roles.extend(std::iter::repeat_n(Role::Bias, bias.len()));
    out_roles.extend(roles[split..].iter().copied());
    (pts, out_roles)
}

/// Indices of the structural points: the corners and the axis ends. They
/// move in x only and cannot be deleted — a boundary corner is at full
/// coverage or at none, there is no such thing as a corner at 0.4.
fn structural(roles: &[Role]) -> Vec<usize> {
    (0..roles.len()).filter(|i| roles[*i].structural()).collect()
}

/// Each crossfade as one group — both corners and every slope point
/// between them — so a SHIFT-drag carries the boundary whole, shape and
/// all. Empty when `off`: a plain drag moves the point you grabbed.
fn fade_groups(roles: &[Role], off: bool) -> Vec<Vec<usize>> {
    if off {
        return Vec::new();
    }
    [false, true]
        .into_iter()
        .map(|above| {
            (0..roles.len())
                .filter(|i| {
                    matches!(roles[*i], Role::Corner { above: a, .. } | Role::Slope { above: a } if a == above)
                })
                .collect::<Vec<usize>>()
        })
        .filter(|g| g.len() >= 2)
        .collect()
}

/// Which part of the band `x` falls in, if any: a slope (between a
/// boundary's two corners) or the plateau (between the band's inner
/// corners or axis ends). Outside the band there is nothing to shape.
fn region(pts: &[[f32; 2]], roles: &[Role], x: f32) -> Option<Region> {
    let at = |want: Role| roles.iter().position(|r| *r == want).map(|i| pts[i][0]);
    let lo_c = at(Role::Corner { above: false, hi: false });
    let hi_c = at(Role::Corner { above: false, hi: true });
    let lo_a = at(Role::Corner { above: true, hi: false });
    let hi_a = at(Role::Corner { above: true, hi: true });
    if let (Some(a), Some(b)) = (lo_c, hi_c) {
        if x > a && x < b {
            return Some(Region::Slope(false));
        }
    }
    if let (Some(a), Some(b)) = (lo_a, hi_a) {
        if x > a && x < b {
            return Some(Region::Slope(true));
        }
    }
    let pl = hi_c.unwrap_or(pts.first().map_or(f32::MAX, |p| p[0]));
    let ph = lo_a.unwrap_or(pts.last().map_or(f32::MIN, |p| p[0]));
    (x > pl && x < ph).then_some(Region::Plateau)
}

/// The roles of an edited curve. A drag or a scale keeps the count (and,
/// with `keep_order`, the order), so the roles are the ones it started
/// with; an add or a delete moves no structural point, so those are found
/// again by x and everything else is named by where it sits between them.
fn roles_after(before: &[[f32; 2]], was: &[Role], pts: &[[f32; 2]]) -> Vec<Option<Role>> {
    if pts.len() == before.len() {
        return was.iter().copied().map(Some).collect();
    }
    let joints: Vec<usize> = structural(was);
    let mut j = 0;
    let mut last: Option<Role> = None;
    pts.iter()
        .map(|p| {
            if let Some(&s) = joints.get(j) {
                if (p[0] - before[s][0]).abs() < 1e-6 {
                    j += 1;
                    last = Some(was[s]);
                    return last;
                }
            }
            match last {
                Some(Role::Corner { above, hi: false }) => Some(Role::Slope { above }),
                Some(Role::Corner { above: false, hi: true }) => Some(Role::Bias),
                Some(Role::End) if j == 1 => Some(Role::Bias),
                _ => None,
            }
        })
        .collect()
}

/// Read an edited band curve back into the partition, then re-derive every
/// band from it. This is where the lock lives: the corners become edge
/// positions, slope points become the edge's shape, and both bands either
/// side of an edge are rebuilt from it.
///
/// Points inside the plateau are the user's own bias points and are kept;
/// they are the only thing the partition does not imply.
fn read_back(k: &mut engine::graph::LumaKeyParams, sel: usize, pts: &[[f32; 2]]) {
    let (before, was) = layout(k, sel);
    let Some(r) = k.ranges.as_mut() else { return };
    let n = r.band_count();
    if pts.len() < 2 || sel >= n {
        return;
    }
    let roles = roles_after(&before, &was, pts);
    let find = |want: Role| roles.iter().position(|r| *r == Some(want)).map(|i| pts[i]);
    let slope = |above: bool| -> Vec<[f32; 2]> {
        roles
            .iter()
            .zip(pts)
            .filter(|(r, _)| **r == Some(Role::Slope { above }))
            .map(|(_, p)| *p)
            .collect()
    };
    for (above, slot) in [(false, sel.checked_sub(1)), (true, (sel < r.edges.len()).then_some(sel))] {
        let Some(slot) = slot else { continue };
        let (Some(a), Some(b)) = (
            find(Role::Corner { above, hi: false }),
            find(Role::Corner { above, hi: true }),
        ) else {
            continue;
        };
        // The neighbouring boundaries are not on this curve; stop at their
        // corners rather than run through them and let `normalize` re-sort.
        let floor = if slot > 0 {
            let e = &r.edges[slot - 1];
            e.at + 0.5 * e.width
        } else {
            r.span.0
        };
        let ceil = if slot + 1 < r.edges.len() {
            let e = &r.edges[slot + 1];
            e.at - 0.5 * e.width
        } else {
            r.span.1
        };
        let x0 = a[0].clamp(floor, ceil);
        let x1 = b[0].clamp(x0, ceil);
        let w = x1 - x0;
        let e = &mut r.edges[slot];
        e.at = 0.5 * (x0 + x1);
        e.width = w;
        if w > 1e-6 {
            e.shape = slope(above)
                .into_iter()
                .map(|p| {
                    let v = p[1].clamp(0.0, 1.0);
                    [(p[0] - x0) / w, if above { 1.0 - v } else { v }]
                })
                .collect();
        }
    }
    r.normalize();
    k.sync_from_ranges();
    // Put the user's bias points back, where they can still mean something:
    // inside the (possibly moved) plateau, and never at or below zero (which
    // would punch a hole a boundary is supposed to own).
    let (now, now_roles) = layout(k, sel);
    let bias: Vec<[f32; 2]> = roles
        .iter()
        .zip(pts)
        .filter(|(r, _)| **r == Some(Role::Bias))
        .map(|(_, p)| [p[0], p[1].clamp(0.02, 1.0)])
        .filter(|p| p[1] < 0.999 && region(&now, &now_roles, p[0]) == Some(Region::Plateau))
        .collect();
    if let Some(b) = k.bands.get_mut(sel) {
        b.points.extend(bias);
        b.points
            .sort_by(|a, c| a[0].partial_cmp(&c[0]).unwrap_or(std::cmp::Ordering::Equal));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real panel, as the properties grid draws it.
    #[test]
    fn the_luma_key_panel_renders() {
        let mut k = engine::graph::LumaKeyParams::default();
        k.adopt_ranges(engine::graph::LUMA_SPAN);
        *k.ranges.as_mut().unwrap() =
            BandRanges::even(3, engine::graph::LUMA_SPAN, false, 0.65);
        k.sync_from_ranges();
        k.bands[1].name = "mids".into();
        let mut sel = 1usize;
        // The REAL size the properties grid gives it: 3 tiles wide, one
        // tall (widgets::TILE = 250, TILE_GAP = 8).
        // Hosted exactly as the properties grid hosts it: a tile-sized
        // pane with a scroll area inside. egui reports the content size and
        // the viewport, so whether the panel is CUT OFF is a fact we can
        // read rather than a scrollbar we have to find in the pixels (its
        // bars float over the content, so they change nothing measurable).
        let host = |h: f32, k: &mut engine::graph::LumaKeyParams, sel: &mut usize| {
            let over = std::cell::Cell::new(0.0f32);
            let shot = crate::snapshot::shoot(766.0, h, |ui| {
                let out = egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        luma_panel(ui, k, 7, sel, 1.0, &BandOpts::KEYER);
                    });
                over.set(out.content_size.y - out.inner_rect.height());
            });
            (shot, over.get())
        };
        let (shot, tall_over) = host(508.0, &mut k, &mut sel);
        shot.save("lumakey-panel");
        let mut short_sel = 1usize;
        let (short, short_over) = host(120.0, &mut k, &mut short_sel);

        // A panel that fits its tile must not be scrollable at all — the
        // bar showing when there is room to spare is the complaint.
        assert!(
            tall_over <= 0.5,
            "panel overflows a full tile by {tall_over}pt, so it shows a scrollbar with room to spare"
        );
        // And a short pane must SCROLL rather than squash the editor.
        assert!(short_over > 0.5, "short pane did not scroll: {short_over}");
        let plot_rows = |sh: &crate::snapshot::Shot, h: usize| {
            (0..h).filter(|y| sh.px(sh.w / 2, *y) != [18, 18, 18]).count()
        };
        let short_rows = plot_rows(&short, 240);
        assert!(
            short_rows > 180,
            "the editor shrank in a short pane instead of scrolling: {short_rows} rows"
        );
        // The middle band is a PLATEAU: zero at both its own ends, solid
        // between them. Two independent thresholds cannot say that.
        let mid = &k.bands[1];
        assert_eq!(mid.ends, [0.0, 0.0]);
        assert!(mid.points.iter().any(|p| p[1] > 0.99), "no plateau: {:?}", mid.points);
    }

    /// The Grade's ranges page: the same panel in its zone flavour, hosted in
    /// what a one-tile Grade panel has left under its title row. It has to
    /// FIT — the plot floors at `PLOT_MIN`, and a page that scrolled would
    /// put the name row where it cannot be clicked.
    #[test]
    fn the_zones_page_fits_a_grade_tile() {
        // Three zones, and enough that the chip row wraps onto a second line.
        for n in [3usize, 14] {
            let mut z = engine::graph::GradeZones::default();
            z.key.bands[1].name = "skin".into();
            while z.count() < n {
                let top = z.count() - 1;
                let new = z.key.split_band(top);
                z.band_split(new);
            }
            let labels: Vec<String> = (0..z.count()).map(|i| z.label(i)).collect();
            assert_eq!(labels.len(), n);
            assert_eq!(labels[..2], ["shadows", "skin"]);
            let opts = BandOpts {
                keyer: false,
                labels: Some(&labels),
                name_hint: "zone name",
                removable: true,
                max_points: Some(engine::graph::ZONE_PTS),
                reserve: 0.0,
            };
            let mut sel = 1usize;
            let over = std::cell::Cell::new(egui::Vec2::ZERO);
            // A one-tile Grade panel's page under its own tab row: the
            // 264pt tile less the group frame, the title row and the tab
            // strip. MEASURED, not estimated — `--test-zones` reads the
            // page's clip as 206pt — because the estimate this test once
            // used (214) was the reason it passed while the app's preview
            // row sat 23pt below the fold.
            let shot = crate::snapshot::shoot(736.0, 206.0, |ui| {
                let out = egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        // The Grade's real tail, at its widest: the chips
                        // AND the hint a preview off the view target shows.
                        let o = luma_panel_with(ui, &mut z.key, 9, &mut sel, 1.0, &opts, |ui, _| {
                            ui.label(egui::RichText::new("preview").small());
                            for name in ["off", "matte", "highlight", "zones", "hard"] {
                                let _ = ui.add(egui::Button::selectable(
                                    name == "matte",
                                    egui::RichText::new(name).small(),
                                ));
                            }
                            ui.label(egui::RichText::new("— view this node (V)").small().weak());
                        });
                        assert!(o.split.is_none() && o.dissolved.is_none() && !o.remove_all);
                    });
                over.set(out.content_size - out.inner_rect.size());
            });
            shot.save(&format!("grade-zones-page-{n}"));
            let o = over.get();
            assert!(o.y <= 0.5, "{n} zones: the page overflows a Grade tile by {}pt", o.y);
            assert!(o.x <= 0.5, "{n} zones: the band row runs {}pt past the tile", o.x);
        }
    }

    fn key3(soft: f32) -> engine::graph::LumaKeyParams {
        let mut k = engine::graph::LumaKeyParams::default();
        k.adopt_ranges((0.0, 2.0));
        *k.ranges.as_mut().unwrap() = BandRanges::even(3, (0.0, 2.0), false, soft);
        k.sync_from_ranges();
        k
    }

    /// The point cap: with a full band the plateau accepts nothing more.
    #[test]
    fn the_point_cap_closes_the_plateau() {
        let k = key3(0.5);
        let (pts, roles) = layout(&k, 1);
        let (lo, hi) = (pts[1][0], pts[2][0]);
        let mid = 0.5 * (lo + hi);
        assert_eq!(region(&pts, &roles, mid), Some(Region::Plateau));
        let room = Some(pts.len()).is_none_or(|m| pts.len() < m);
        assert!(!room, "a band at the cap still offered room");
        let room = Some(pts.len() + 1).is_none_or(|m| pts.len() < m);
        assert!(room && mid > lo && mid < hi);
    }

    /// A crossfade is a group: two corners and every slope point between
    /// them. It moves as one only when asked (shift at the call site) — a
    /// plain drag moves the one point you grabbed.
    #[test]
    fn crossfade_corners_pair_up_only_on_request() {
        let mut k = key3(0.5);
        k.ranges.as_mut().unwrap().edges[0].shape = vec![[0.5, 0.3]];
        k.sync_from_ranges();
        let (pts, roles) = layout(&k, 1);
        let groups = fade_groups(&roles, false);
        assert_eq!(groups.len(), 2, "a middle band has two crossfades: {pts:?}");
        assert_eq!(groups[0], vec![0, 1, 2], "the slope point rides with its corners");
        assert_eq!(groups[1].len(), 2);
        assert!(fade_groups(&roles, true).is_empty());
    }

    /// Editing a band's curve must land back on the PARTITION, so the band
    /// next door moves too. This is the lock, tested end to end.
    #[test]
    fn dragging_a_boundary_moves_the_band_on_the_other_side() {
        let mut k = key3(0.4);
        let neighbour_before = k.bands[2].curve();
        // Shove band 1's UPPER crossfade to the right.
        let mut pts = k.bands[1].curve();
        let last = pts.len() - 1;
        pts[last - 1][0] += 0.3;
        pts[last][0] += 0.3;
        read_back(&mut k, 1, &pts);
        let e = k.ranges.as_ref().unwrap().edges[1].at;
        assert!(e > 1.3, "the boundary did not move: {e}");
        assert_ne!(k.bands[2].curve(), neighbour_before, "band 2 did not follow");
        // And the partition is still a partition.
        let r = k.ranges.as_ref().unwrap();
        for i in 0..=40 {
            let s: f32 = r.weights(i as f32 / 20.0).iter().sum();
            assert!((s - 1.0).abs() < 1e-4, "weights broke: {s}");
        }
    }

    /// A point the user adds inside the plateau is theirs — the partition
    /// has no opinion about it, so a re-derive must not eat it.
    #[test]
    fn bias_points_survive_a_partition_edit() {
        let mut k = key3(0.4);
        let mut pts = k.bands[1].curve();
        // A bias point in the middle of the plateau.
        let mid = 0.5 * (pts[1][0] + pts[2][0]);
        pts.insert(2, [mid, 0.6]);
        read_back(&mut k, 1, &pts);
        assert!(
            k.bands[1].points.iter().any(|p| (p[0] - mid).abs() < 1e-3 && p[1] < 0.9),
            "the bias point was eaten: {:?}",
            k.bands[1].points
        );
    }

    /// The backdrop must be KNOCKED BACK, not painted at full strength: a
    /// light ramp at full strength swallows the grid, the ghosts and the
    /// curve, and the caller should not have to pre-dim its own colours.
    #[test]
    fn a_bright_backdrop_is_scrimmed_so_the_curve_still_reads() {
        let mut pts = vec![[0.2f32, 0.9], [1.0, 0.95], [1.8, 0.98]];
        let shot = crate::snapshot::shoot(400.0, 200.0, |ui| {
            let cfg = curveui::CurveCfg {
                size: egui::vec2(380.0, 180.0),
                view: [0.0, 2.0, 0.0, 1.0],
                clamp_y: Some((0.0, 1.0)),
                backdrop: Some(&|_x, _y| Color32::from_gray(250)),
                ..Default::default()
            };
            curveui::curve_edit(ui, egui::Id::new("bd"), &cfg, &[], &mut pts);
        });
        shot.save("curve-backdrop");
        // Well away from the curve, the plate must be mid grey, not white.
        let plate = shot.at(200.0, 140.0);
        assert!(
            plate[0] > 90 && plate[0] < 190,
            "backdrop not scrimmed (or scrimmed to nothing): {plate:?}"
        );
        // And somewhere in that column the curve must still be brighter
        // than the plate. Scanned rather than sampled at a guessed y: the
        // widget's origin depends on the panel's own layout.
        let brightest = (0..190)
            .map(|y| shot.at(200.0, y as f32)[0])
            .max()
            .unwrap_or(0);
        assert!(
            brightest > plate[0] + 30,
            "curve lost in the backdrop: brightest {brightest} vs plate {}",
            plate[0]
        );
    }

    /// Evaluate a band exactly as the shader does: curve, clamp, invert.
    fn response(k: &engine::graph::LumaKeyParams, band: usize, x: f32) -> f32 {
        let b = &k.bands[band];
        let mut v = curveui::eval(&b.curve(), x, b.interp, true).clamp(0.0, 1.0);
        if b.invert {
            v = 1.0 - v;
        }
        v
    }

    /// An ordinary threshold key authored before partitions existed must
    /// keep the response it had. Output 0 is what every existing wire into
    /// the node is reading, so migration cannot quietly redefine it.
    ///
    /// A threshold is not "one band": it is TWO, below and above, with one
    /// boundary. That is why the migration produces two.
    #[test]
    fn a_legacy_single_band_key_keeps_its_response() {
        for (lo, hi, inv) in [(0.3f32, 0.7f32, false), (0.2, 0.9, true), (0.0, 0.02, false)] {
            let mut k = engine::graph::LumaKeyParams {
                bands: vec![engine::graph::LumaBand {
                    lo,
                    hi,
                    invert: inv,
                    ..Default::default()
                }],
                ranges: None,
            };
            let before: Vec<f32> = (0..=40).map(|i| response(&k, 0, i as f32 / 20.0)).collect();
            k.adopt_ranges(engine::graph::LUMA_SPAN);
            assert!(k.ranges.is_some(), "migration did not happen");
            assert_eq!(k.bands.len(), 2, "a threshold is two bands");
            let after: Vec<f32> = (0..=40).map(|i| response(&k, 0, i as f32 / 20.0)).collect();
            for (i, (a, b)) in before.iter().zip(&after).enumerate() {
                assert!(
                    (a - b).abs() < 0.03,
                    "({lo},{hi},{inv}) response changed at x={}: {a} -> {b}",
                    i as f32 / 20.0
                );
            }
        }
    }

    /// There is no second model left to fall into: whatever a file said, a
    /// key comes back with a partition, so the panel has one shape.
    #[test]
    fn every_key_has_a_partition_after_load() {
        for src in [
            r#"{"lo":0.1,"hi":0.4}"#,
            r#"{"bands":[{"lo":0.0,"hi":1.0}]}"#,
            r#"{}"#,
        ] {
            let k: engine::graph::LumaKeyParams = serde_json::from_str(src).unwrap();
            assert!(k.ranges.is_some(), "{src} did not migrate");
        }
    }

    /// A band's joints are its corners and the axis ends. Bias and slope
    /// points are the user's, and stay free.
    #[test]
    fn the_structural_points_are_the_joints_and_only_the_joints() {
        let mut k = key3(0.5);
        let (pts, roles) = layout(&k, 1);
        // Two crossfades, four corners.
        assert_eq!(structural(&roles), vec![0, 1, 2, 3]);
        // Add a bias point in the plateau and a slope point on the upper
        // crossfade, the way the editor would.
        let mid = 0.5 * (pts[1][0] + pts[2][0]);
        let slope = 0.5 * (pts[2][0] + pts[3][0]);
        let mut edited = pts.clone();
        edited.insert(2, [mid, 0.55]);
        edited.insert(4, [slope, 0.8]);
        read_back(&mut k, 1, &edited);
        let (out, roles) = layout(&k, 1);
        assert_eq!(roles.len(), 6, "{out:?} {roles:?}");
        assert_eq!(roles[2], Role::Bias);
        assert_eq!(roles[4], Role::Slope { above: true });
        assert_eq!(structural(&roles), vec![0, 1, 3, 5]);
        for i in structural(&roles) {
            assert!(out[i][1] < 0.01 || out[i][1] > 0.99, "joint off its rail: {:?}", out[i]);
        }
    }

    /// A new point goes on a slope (shaping that boundary) or in the
    /// plateau (the band's bias); outside the band there is nothing to shape.
    #[test]
    fn points_can_be_added_on_a_slope_or_the_plateau() {
        let k = key3(0.5);
        let (pts, roles) = layout(&k, 1);
        let mid = |a: usize, b: usize| 0.5 * (pts[a][0] + pts[b][0]);
        assert_eq!(region(&pts, &roles, mid(0, 1)), Some(Region::Slope(false)));
        assert_eq!(region(&pts, &roles, mid(1, 2)), Some(Region::Plateau));
        assert_eq!(region(&pts, &roles, mid(2, 3)), Some(Region::Slope(true)));
        assert_eq!(region(&pts, &roles, 0.05), None, "outside the band accepted a point");
        assert_eq!(region(&pts, &roles, 1.95), None);
        // An end band's plateau runs to the end of the axis.
        let (p0, r0) = layout(&k, 0);
        assert_eq!(region(&p0, &r0, 0.05), Some(Region::Plateau));
    }

    /// A point added on a slope is the BOUNDARY's: it lands in the edge's
    /// shape, the band next door falls through its complement, the
    /// partition still sums to 1, and it survives an edit made from the
    /// other side.
    #[test]
    fn a_slope_point_shapes_the_boundary_for_both_bands() {
        let mut k = key3(0.6);
        let (pts, _) = layout(&k, 1);
        let x = pts[2][0] + 0.25 * (pts[3][0] - pts[2][0]);
        let mut edited = pts.clone();
        edited.insert(3, [x, 0.3]);
        read_back(&mut k, 1, &edited);
        let r = k.ranges.as_ref().unwrap();
        assert_eq!(r.edges[1].shape.len(), 1, "{:?}", r.edges);
        let s = r.edges[1].shape[0];
        assert!((s[0] - 0.25).abs() < 1e-4 && (s[1] - 0.7).abs() < 1e-4, "{s:?}");
        let w = r.weights(x);
        assert!((w[1] - 0.3).abs() < 1e-3 && (w[2] - 0.7).abs() < 1e-3, "{w:?}");
        for i in 0..=80 {
            let t: f32 = r.weights(i as f32 / 40.0).iter().sum();
            assert!((t - 1.0).abs() < 1e-4, "weights broke: {t}");
        }
        // The neighbour's own curve carries the complement.
        let up = k.bands[2].curve();
        assert!(up.iter().any(|p| (p[0] - x).abs() < 1e-4 && (p[1] - 0.7).abs() < 1e-4), "{up:?}");
        // Read band 2 back untouched: the shape is the edge's, so it stays.
        let (p2, _) = layout(&k, 2);
        read_back(&mut k, 2, &p2);
        assert_eq!(k.ranges.as_ref().unwrap().edges[1].shape.len(), 1);
        // And deleting it from band 2's side removes it.
        let (p2, r2) = layout(&k, 2);
        let at = r2.iter().position(|r| *r == Role::Slope { above: false }).unwrap();
        let mut del = p2.clone();
        del.remove(at);
        read_back(&mut k, 2, &del);
        assert!(k.ranges.as_ref().unwrap().edges[1].shape.is_empty());
    }

    /// A corner dragged past the far corner of the NEXT boundary (which is
    /// not on this band's curve) stops at it instead of swapping order.
    #[test]
    fn a_corner_stops_at_the_next_boundary() {
        let mut k = key3(0.3);
        let before = k.ranges.as_ref().unwrap().edges.clone();
        let (pts, _) = layout(&k, 1);
        let mut edited = pts.clone();
        // Shove the upper crossfade's far corner past the top of the axis
        // and the lower one's outer corner below zero.
        edited[3][0] = 5.0;
        edited[0][0] = -5.0;
        read_back(&mut k, 1, &edited);
        let r = k.ranges.as_ref().unwrap();
        assert!(r.edges[0].at < r.edges[1].at, "{:?}", r.edges);
        assert!(r.edges[1].at + 0.5 * r.edges[1].width <= 2.0 + 1e-5);
        assert!(r.edges[0].at - 0.5 * r.edges[0].width >= -1e-5);
        assert_eq!(r.edges.len(), before.len());
    }

    /// The invariant, end to end: whatever the curve came back looking like,
    /// what gets stored has its joints on 0 and 1.
    #[test]
    fn read_back_repairs_a_joint_dragged_off_its_rail() {
        let mut k = key3(0.4);
        let mut pts = k.bands[1].curve();
        // Shove a joint to a nonsense height and a bias point below zero.
        pts[1][1] = 0.42;
        let mid = 0.5 * (pts[1][0] + pts[2][0]);
        pts.insert(2, [mid, -3.0]);
        read_back(&mut k, 1, &pts);
        let (out, roles) = layout(&k, 1);
        for p in &out {
            assert!(
                p[1] >= 0.0 && p[1] <= 1.0,
                "stored point outside coverage: {p:?} in {out:?}"
            );
        }
        for i in structural(&roles) {
            assert!(
                out[i][1] < 0.01 || out[i][1] > 0.99,
                "joint not repaired: {:?} in {out:?}",
                out[i]
            );
        }
    }

    /// Dragging one corner of a crossfade must SLIDE the crossfade, not
    /// collapse it. A mirrored group moves by a shared delta; forcing its
    /// members onto the moved point's x snapped the two corners together and
    /// turned every boundary vertical the instant it was touched.
    #[test]
    fn dragging_a_crossfade_corner_slides_it_instead_of_collapsing_it() {
        let k = key3(0.5);
        let (start, roles) = layout(&k, 1);
        let mirror = fade_groups(&roles, false);
        assert!(!mirror.is_empty());
        let width = |p: &[[f32; 2]]| p[1][0] - p[0][0];
        let w0 = width(&start);
        assert!(w0 > 0.01, "no crossfade to test: {start:?}");

        // The gesture: grab point 0 (the outer corner of the lower fade)
        // and pull it right.
        let mut next = start.clone();
        next[0][0] += 0.2;
        let cfg = curveui::CurveCfg { mirror: &mirror, ..Default::default() };
        curveui::mirror_x(&cfg, &mut next, &start, &[0]);

        assert!(
            (width(&next) - w0).abs() < 1e-5,
            "crossfade width changed: {w0} -> {}",
            width(&next)
        );
        assert!(
            (next[1][0] - (start[1][0] + 0.2)).abs() < 1e-5,
            "the partner did not follow by the same delta: {:?}",
            next
        );
        // Grabbing the OTHER member carries the first the same way.
        let mut back = start.clone();
        back[1][0] -= 0.15;
        curveui::mirror_x(&cfg, &mut back, &start, &[1]);
        assert!((width(&back) - w0).abs() < 1e-5);
        assert!((back[0][0] - (start[0][0] - 0.15)).abs() < 1e-5);
    }

    /// Two points that genuinely are ONE place still land on one place —
    /// the delta rule has to subsume the shared-x case, not replace it.
    #[test]
    fn a_zero_offset_group_still_shares_its_x() {
        let start = [[0.5f32, 0.0], [0.5, 1.0], [1.5, 1.0]];
        let mut next = start;
        next[0][0] = 0.8;
        let cfg = curveui::CurveCfg { mirror: &[vec![0, 1]], ..Default::default() };
        curveui::mirror_x(&cfg, &mut next, &start, &[0]);
        assert_eq!(next[0][0], next[1][0]);
    }
}
