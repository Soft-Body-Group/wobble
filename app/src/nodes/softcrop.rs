//! SoftCrop's app half: the box's four edges as channels, the panel, the
//! glyph, and the one thing the node exists for — [`frame_for`], which
//! tells a Transform downstream which rectangle its handles should span.
//! The params are `engine::nodes::softcrop`.

use crate::anim::Channel;
use crate::editor::EditorGraph;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke};
use engine::graph::{CropMode, NodeKind, SoftCropParams};

pub const CHANNELS: &[Channel] =
    &[Channel::SoftCropL, Channel::SoftCropT, Channel::SoftCropR, Channel::SoftCropB];

pub const TILES: usize = 1;

pub fn get(p: &SoftCropParams, ch: Channel) -> Option<f32> {
    CHANNELS.iter().position(|c| *c == ch).map(|i| p.rect[i])
}

pub fn set(p: &mut SoftCropParams, ch: Channel, v: f32) {
    if let Some(i) = CHANNELS.iter().position(|c| *c == ch) {
        p.rect[i] = v;
    }
}

pub fn label(ch: Channel) -> Option<&'static str> {
    match ch {
        Channel::SoftCropL => Some("crop left"),
        Channel::SoftCropT => Some("crop top"),
        Channel::SoftCropR => Some("crop right"),
        Channel::SoftCropB => Some("crop bottom"),
        _ => None,
    }
}

pub fn panel(ui: &mut egui::Ui, rcx: &mut RowCtx, p: &mut SoftCropParams) -> Option<HudSlider> {
    let [l, t, r, b] = &mut p.rect;
    let mut h = widgets::drags(
        ui,
        rcx,
        "left / right",
        &mut [l, r],
        0.002,
        (0.0, 1.0),
        &[Channel::SoftCropL, Channel::SoftCropR],
    );
    h = h.or(widgets::drags(
        ui,
        rcx,
        "top / bottom",
        &mut [t, b],
        0.002,
        (0.0, 1.0),
        &[Channel::SoftCropT, Channel::SoftCropB],
    ));
    ui.horizontal(|ui| {
        let mut mode = p.mode();
        for m in CropMode::ALL {
            ui.selectable_value(&mut mode, m, m.label()).on_hover_text(match m {
                CropMode::BoundingBox => "keep every pixel: the box only sizes the Transform handles below",
                CropMode::Crop => "remove the pixels outside the box; the format stays",
                CropMode::Reformat => "the stream becomes the box: its format is the box's size",
            });
        }
        if mode != p.mode() {
            p.mode = mode;
            p.full = false;
        }
        if ui.small_button("reset").clicked() {
            // The box resets; the mode stays.
            *p = SoftCropParams { mode: p.mode(), matte: p.matte, ..Default::default() };
        }
    });
    ui.weak(match p.mode() {
        CropMode::BoundingBox => "keeps every pixel; sizes the Transform handles below it",
        CropMode::Crop => "removes the pixels outside; sizes the Transform handles below it",
        CropMode::Reformat => "the format is the box: its bake (the bake tab) stores and plays back that size",
    });
    h
}

/// A frame drawn in corner brackets only: a box that is there, but does not
/// cut anything.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.rect(
        egui::Rect::from_center_size(c, egui::vec2(r * 1.7, r * 1.7)),
        r * 0.3,
        fill,
        stroke,
        egui::StrokeKind::Middle,
    );
    let (h, k) = (r * 0.55, r * 0.3);
    let ink = Stroke::new((r * 0.14).max(1.0), crate::theme::node_ink());
    for (sx, sy) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
        let p = c + egui::vec2(sx * h, sy * h);
        painter.line_segment([p, p - egui::vec2(sx * k, 0.0)], ink);
        painter.line_segment([p, p - egui::vec2(0.0, sy * k)], ink);
    }
}

/// The box a Transform's handles should span, in the Transform's own pixels
/// (the space `layerui::transform_gizmo` works in): the nearest SoftCrop up
/// its image chain — or up the MATTE that cuts a plate out on the way (a
/// Mask, a MatteIn, a Grade's mask), since a crop on the cut-out's matte is
/// the cut-out's box — carried through every Transform / CornerPin /
/// Reformat in between. `None` when there is none, or when a merge comes
/// first: a crop on one branch of a merge says nothing about the other.
///
/// `res` is each node's format and `step` what one node does to the pixels
/// passing through it (`node_step_h`); `kind_at` is the node's EVALUATED
/// params, so an animated box is where it is on screen.
pub fn frame_for(
    g: &EditorGraph,
    transform: u64,
    res: &dyn Fn(u64) -> (f32, f32),
    step: &dyn Fn(u64) -> crate::H3,
    kind_at: &dyn Fn(u64) -> Option<NodeKind>,
) -> Option<egui::Rect> {
    // A single-input chain from `from` upward to a SoftCrop, with the nodes
    // passed on the way (nearest first). A merge or a leaf ends it.
    let up = |from: u64, seen: &mut std::collections::HashSet<u64>| {
        let mut chain = Vec::new();
        let mut cur = from;
        loop {
            if !seen.insert(cur) {
                return None;
            }
            let n = g.node(cur)?;
            if matches!(n.kind, NodeKind::SoftCrop(_)) && !n.disabled {
                return Some((cur, chain));
            }
            // Mask and modulate mattes are not the picture.
            let images: Vec<usize> =
                (0..n.kind.input_count()).filter(|i| !n.kind.is_side_input(*i)).collect();
            if images.len() != 1 {
                return None;
            }
            chain.push(cur);
            cur = g.input_of(cur, images[0])?;
        }
    };
    let mut seen = std::collections::HashSet::new();
    let mut chain: Vec<u64> = Vec::new();
    let mut cur = g.input_of(transform, 0)?;
    // Which output of `cur` the walk came up through.
    let mut out = g.out_of(transform, 0) as usize;
    let (crop, mut path) = loop {
        let n = g.node(cur)?;
        if matches!(n.kind, NodeKind::SoftCrop(_)) && !n.disabled {
            break (cur, Vec::new());
        }
        // An Extract's output k is its image cut by matte k (input k + 1):
        // that matte's crop is the output's box, and nothing else is.
        if matches!(n.kind, NodeKind::Extract(_)) {
            let (c, mchain) = up(g.input_of(cur, out + 1)?, &mut seen)?;
            break (c, mchain);
        }
        // The matte this node cuts with, if a crop sits up it.
        if let Some(m) = n.kind.mask_input().and_then(|i| g.input_of(cur, i)) {
            if let Some((c, mchain)) = up(m, &mut seen.clone()) {
                break (c, mchain);
            }
        }
        if !seen.insert(cur) {
            return None;
        }
        let images: Vec<usize> =
            (0..n.kind.input_count()).filter(|i| !n.kind.is_side_input(*i)).collect();
        if images.len() != 1 {
            return None;
        }
        chain.push(cur);
        out = g.out_of(cur, images[0]) as usize;
        cur = g.input_of(cur, images[0])?;
    };
    let Some(NodeKind::SoftCrop(p)) = kind_at(crop) else { return None };
    let (w, h) = res(crop);
    // A reformat's stream IS the box: below it the box is the whole frame
    // (`res` is the crop's own format, already the box's size). Its rect is
    // normalised to the INPUT's format, so read against this one it came
    // out a box inside a box.
    let [x0, y0, x1, y1] = if p.mode() == CropMode::Reformat {
        [0.0, 0.0, w, h]
    } else {
        p.px(w, h)
    };
    // Crop pixels -> the Transform's input pixels: up the matte side (if
    // that is where it was), then the image chain, farthest node first.
    path.reverse();
    path.extend(chain.iter().rev());
    let mut m = crate::IDENTITY_H;
    for id in &path {
        m = crate::h_mul(&step(*id), &m);
    }
    let pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        .map(|(x, y)| crate::apply_h(&m, egui::vec2(x, y)).to_pos2());
    // A corner pin in between makes it a quad; its bounds are the box.
    Some(egui::Rect::from_points(&pts))
}

/// A CornerPin under a soft crop pins the CROP's corners, not the frame's:
/// where each corner of `crop` (normalised x0, y0, x1, y1) lands under the
/// pin's `to` quad — the four points the viewer draws and you drag. TL, TR,
/// BR, BL, like `to`.
pub fn pins_on_crop(to: [[f32; 2]; 4], crop: [f32; 4]) -> [[f32; 2]; 4] {
    let h = engine::codegen::corner_pin_forward_px(&engine::graph::CornerPinParams::pinned(to), [1.0, 1.0]);
    let [x0, y0, x1, y1] = crop;
    [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].map(|[x, y]| {
        let p = crate::apply_h(&h, egui::vec2(x, y));
        [p.x, p.y]
    })
}

/// The inverse: the `to` quad that puts the crop's corners at `pins`. The
/// node still stores what it always stored (where the whole frame lands),
/// so a pin dragged here renders like any other and a setup is unchanged.
pub fn to_from_crop_pins(pins: [[f32; 2]; 4], crop: [f32; 4]) -> [[f32; 2]; 4] {
    // Unit square -> pins, after the frame's corners are expressed in the
    // crop's own unit square.
    let q = engine::codegen::corner_pin_forward_px(
        &engine::graph::CornerPinParams::pinned(pins),
        [1.0, 1.0],
    );
    let [x0, y0, x1, y1] = crop;
    let (w, h) = ((x1 - x0).max(1e-6), (y1 - y0).max(1e-6));
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]].map(|[x, y]| {
        let p = crate::apply_h(&q, egui::vec2((x - x0) / w, (y - y0) / h));
        [p.x, p.y]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn crop(rect: [f32; 4]) -> NodeKind {
        NodeKind::SoftCrop(SoftCropParams { rect, ..Default::default() })
    }

    #[test]
    fn a_transform_finds_the_crop_above_it_through_a_grade() {
        let mut g = EditorGraph::default();
        let z = Pos2::ZERO;
        let src = g.add(NodeKind::Grid(Default::default()), z);
        let sc = g.add(crop([0.25, 0.0, 0.75, 0.5]), z);
        let gr = g.add(NodeKind::Grade(Default::default()), z);
        let xf = g.add(NodeKind::Transform(Default::default()), z);
        g.connect(sc, 0, src);
        g.connect(gr, 0, sc);
        g.connect(xf, 0, gr);
        let kind_at = |id: u64| g.node(id).map(|n| n.kind.clone());
        let r = frame_for(&g, xf, &|_| (200.0, 100.0), &|_| crate::IDENTITY_H, &kind_at)
            .expect("a crop upstream");
        assert_eq!(r, egui::Rect::from_min_max(Pos2::new(50.0, 0.0), Pos2::new(150.0, 50.0)));
    }

    /// Below a reformat the stream IS the box: the handles span the whole
    /// new frame (`res` of the crop is already the box's size), not the
    /// rect read a second time against it.
    #[test]
    fn below_a_reformat_the_box_is_the_whole_new_frame() {
        let mut g = EditorGraph::default();
        let z = Pos2::ZERO;
        let src = g.add(NodeKind::Grid(Default::default()), z);
        let sc = g.add(
            NodeKind::SoftCrop(SoftCropParams {
                rect: [0.25, 0.25, 0.75, 0.6],
                mode: CropMode::Reformat,
                ..Default::default()
            }),
            z,
        );
        let xf = g.add(NodeKind::Transform(Default::default()), z);
        g.connect(sc, 0, src);
        g.connect(xf, 0, sc);
        let kind_at = |id: u64| g.node(id).map(|n| n.kind.clone());
        let r = frame_for(&g, xf, &|_| (256.0, 101.0), &|_| crate::IDENTITY_H, &kind_at)
            .expect("a crop upstream");
        assert_eq!(r, egui::Rect::from_min_max(Pos2::ZERO, Pos2::new(256.0, 101.0)));
    }

    #[test]
    fn a_node_in_between_carries_the_box_and_a_merge_stops_the_walk() {
        let mut g = EditorGraph::default();
        let z = Pos2::ZERO;
        let src = g.add(NodeKind::Grid(Default::default()), z);
        let sc = g.add(crop([0.0, 0.0, 0.5, 0.5]), z);
        let mid = g.add(NodeKind::Transform(Default::default()), z);
        let xf = g.add(NodeKind::Transform(Default::default()), z);
        g.connect(sc, 0, src);
        g.connect(mid, 0, sc);
        g.connect(xf, 0, mid);
        let kind_at = |id: u64| g.node(id).map(|n| n.kind.clone());
        let shift = [[1.0, 0.0, 10.0], [0.0, 1.0, 5.0], [0.0, 0.0, 1.0]];
        let step = |id: u64| if id == mid { shift } else { crate::IDENTITY_H };
        let r = frame_for(&g, xf, &|_| (100.0, 100.0), &step, &kind_at).unwrap();
        assert_eq!(r.min, Pos2::new(10.0, 5.0));
        assert_eq!(r.max, Pos2::new(60.0, 55.0));

        // A crop on one side of a merge is not the merge's box.
        let bl = g.add(NodeKind::blend(), z);
        let top = g.add(NodeKind::Transform(Default::default()), z);
        g.connect(bl, 0, sc);
        g.connect(top, 0, bl);
        let kind_at = |id: u64| g.node(id).map(|n| n.kind.clone());
        assert!(frame_for(&g, top, &|_| (100.0, 100.0), &|_| crate::IDENTITY_H, &kind_at).is_none());
    }

    /// Pins on the crop and the `to` they solve for are one map, both ways:
    /// dragging nothing changes nothing, and the identity pin puts the
    /// crop's corners exactly on the crop.
    #[test]
    fn crop_pins_round_trip_through_the_quad() {
        let crop = [0.2, 0.1, 0.6, 0.7];
        let ident = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let pins = pins_on_crop(ident, crop);
        assert!((pins[2][0] - 0.6).abs() < 1e-5 && (pins[2][1] - 0.7).abs() < 1e-5);
        let to = [[0.05, 0.02], [0.9, 0.1], [1.1, 0.95], [-0.1, 0.85]];
        let back = to_from_crop_pins(pins_on_crop(to, crop), crop);
        for (a, b) in to.iter().zip(back) {
            assert!((a[0] - b[0]).abs() < 1e-3 && (a[1] - b[1]).abs() < 1e-3, "{to:?} vs {back:?}");
        }
    }

    /// Each output of an Extract takes the crop on ITS matte.
    #[test]
    fn an_extract_output_takes_the_crop_on_its_own_matte() {
        let mut g = EditorGraph::default();
        let z = Pos2::ZERO;
        let plate = g.add(NodeKind::Grid(Default::default()), z);
        let ex = g.add(NodeKind::Extract(engine::graph::ExtractParams { n: 2 }), z);
        g.connect(ex, 0, plate);
        let mut crops = Vec::new();
        for k in 0..2 {
            let sp = g.add(NodeKind::Spline(Default::default()), z);
            let sc = g.add(crop([0.1 * k as f32, 0.0, 0.5, 0.5]), z);
            g.connect(sc, 0, sp);
            g.connect(ex, 1 + k, sc);
            crops.push(sc);
        }
        let xf = g.add(NodeKind::Transform(Default::default()), z);
        g.connect_out(xf, 0, ex, 1);
        let kind_at = |id: u64| g.node(id).map(|n| n.kind.clone());
        let r = frame_for(&g, xf, &|_| (100.0, 100.0), &|_| crate::IDENTITY_H, &kind_at).unwrap();
        assert!((r.min.x - 10.0).abs() < 1e-3, "output 1 is cut by matte 2: {r:?}");
    }

    /// A spline cutting a plate out through a Mask, the soft crop on the
    /// SPLINE: the Transform under the Mask gets the cut-out's box, and the
    /// crop took the matte pipe when the spline was wired into it.
    #[test]
    fn a_crop_on_the_cutting_matte_is_the_cut_outs_box() {
        let mut g = EditorGraph::default();
        let z = Pos2::ZERO;
        let plate = g.add(NodeKind::Grid(Default::default()), z);
        let spline = g.add(NodeKind::Spline(Default::default()), z);
        let sc = g.add(crop([0.1, 0.2, 0.3, 0.4]), z);
        let mask = g.add(NodeKind::Mask, z);
        let xf = g.add(NodeKind::Transform(Default::default()), z);
        g.connect(sc, 0, spline);
        assert!(
            matches!(g.node(sc).map(|n| &n.kind), Some(NodeKind::SoftCrop(p)) if p.matte),
            "a spline wired in makes it a matte soft crop"
        );
        let mi = NodeKind::Mask.mask_input().unwrap();
        g.connect(mask, 1 - mi, plate);
        g.connect(mask, mi, sc);
        g.connect(xf, 0, mask);
        let kind_at = |id: u64| g.node(id).map(|n| n.kind.clone());
        let r = frame_for(&g, xf, &|_| (100.0, 100.0), &|_| crate::IDENTITY_H, &kind_at)
            .expect("the matte's crop");
        let want = egui::Rect::from_min_max(Pos2::new(10.0, 20.0), Pos2::new(30.0, 40.0));
        assert!((r.min - want.min).length() < 1e-3 && (r.max - want.max).length() < 1e-3, "{r:?}");
    }
}
