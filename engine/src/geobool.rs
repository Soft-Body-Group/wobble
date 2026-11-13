//! The 2dini Boolean node's core: union, intersect, subtract and exclude
//! on the FILLED AREA of two geometries (Houdini's Boolean SOP, in 2D).
//!
//! The clipping is `i_overlay`. Each closed prim is first cleaned under its
//! OWN fill rule, so a glyph keeps its counters however its rings happen
//! to wind; the cleaned shapes come back in one canonical winding, so the
//! pile of them then overlays under non-zero as a plain union of areas.
//! A bare point counts as the disc it draws.
//!
//! Subtract and Intersect run per A shape, so each piece keeps its shape's
//! style, physics and maker; Union and Exclude fuse everything and wear
//! the first shape's style. An open line in A is CUT by B's area (kept
//! inside for Intersect, outside for Subtract) and passes through Union
//! and Exclude untouched; open lines in B have no area and cut nothing.

use i_overlay::core::fill_rule::FillRule as Fr;
use i_overlay::core::overlay_rule::OverlayRule;
use i_overlay::float::clip::FloatClip;
use i_overlay::float::simplify::SimplifyShape;
use i_overlay::float::single::SingleFloatOverlay;
use i_overlay::string::clip::ClipRule;

use crate::geo::{FillRule, Geometry, Point, PointId, Prim, PrimKind};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BoolOp {
    /// A and B fused into one area (A alone: its overlaps dissolved).
    #[default]
    Union,
    /// Only where A and B overlap.
    Intersect,
    /// A with B cut out of it.
    Subtract,
    /// Where exactly one of them is.
    Exclude,
}

impl BoolOp {
    pub const ALL: [BoolOp; 4] =
        [BoolOp::Union, BoolOp::Intersect, BoolOp::Subtract, BoolOp::Exclude];
    pub fn name(self) -> &'static str {
        match self {
            BoolOp::Union => "union",
            BoolOp::Intersect => "intersect",
            BoolOp::Subtract => "subtract",
            BoolOp::Exclude => "exclude",
        }
    }
}

type Contour = Vec<[f64; 2]>;
type Shapes = Vec<Vec<Contour>>;

/// Segments in the polygon standing in for a point's disc.
const DISC_SEGS: usize = 32;

fn fr(r: FillRule) -> Fr {
    match r {
        FillRule::NonZero => Fr::NonZero,
        FillRule::EvenOdd => Fr::EvenOdd,
    }
}

/// One prim's filled area as clean shapes (empty for an open line).
fn prim_area(g: &Geometry, i: usize, px_per_unit: f32) -> Shapes {
    let pr = &g.prims[i];
    match pr.kind {
        PrimKind::Poly { closed: true } => {
            let rings: Vec<Contour> = g
                .rings(i)
                .into_iter()
                .filter(|r| r.len() >= 3)
                .map(|r| r.into_iter().map(|p| [p[0] as f64, p[1] as f64]).collect())
                .collect();
            if rings.is_empty() {
                return Vec::new();
            }
            rings.simplify_shape(fr(pr.style.rule))
        }
        PrimKind::Point => {
            let Some(q) = pr.verts.first().and_then(|&v| g.points.get(v as usize)) else {
                return Vec::new();
            };
            let r = (q.pscale / px_per_unit.max(1e-6)) as f64;
            if r <= 0.0 {
                return Vec::new();
            }
            let disc: Contour = (0..DISC_SEGS)
                .map(|k| {
                    let a = k as f64 / DISC_SEGS as f64 * std::f64::consts::TAU;
                    [q.p[0] as f64 + r * a.cos(), q.p[1] as f64 + r * a.sin()]
                })
                .collect();
            vec![vec![disc]]
        }
        PrimKind::Poly { closed: false } => Vec::new(),
    }
}

/// Every prim's area, piled up (overlays under non-zero as their union).
fn area(g: &Geometry, px_per_unit: f32) -> Shapes {
    (0..g.prims.len()).flat_map(|i| prim_area(g, i, px_per_unit)).collect()
}

fn is_open(p: &Prim) -> bool {
    matches!(p.kind, PrimKind::Poly { closed: false })
}

/// Append `shapes` as closed prims (outer ring first, holes as rings)
/// cloned from `like`'s style and tags.
fn emit(out: &mut Geometry, shapes: &Shapes, like: &Prim) {
    for shape in shapes {
        let mut verts = Vec::new();
        let mut rings = Vec::new();
        for (k, c) in shape.iter().enumerate() {
            if c.len() < 3 {
                continue;
            }
            if k > 0 && !verts.is_empty() {
                rings.push(verts.len() as u32);
            }
            for p in c {
                verts.push(push_point(out, *p));
            }
        }
        if verts.len() < 3 {
            continue;
        }
        let mut style = like.style;
        // The pieces come back wound canonically, holes against their
        // outer ring: even-odd reads them right whatever the source rule.
        style.rule = FillRule::EvenOdd;
        out.prims.push(Prim {
            layer: Default::default(),
            kind: PrimKind::Poly { closed: true },
            verts,
            style,
            phys: like.phys,
            src: like.src,
            rings,
        });
    }
}

fn push_point(out: &mut Geometry, p: [f64; 2]) -> PointId {
    let id = out.points.len() as PointId;
    out.points.push(Point { p: [p[0] as f32, p[1] as f32], id, ..Default::default() });
    id
}

/// Append an open line, its style and tags from `like`.
fn emit_line(out: &mut Geometry, path: &[[f64; 2]], like: &Prim) {
    if path.len() < 2 {
        return;
    }
    let verts = path.iter().map(|p| push_point(out, *p)).collect();
    out.prims.push(Prim { verts, rings: Vec::new(), ..like.clone() });
}

/// `a` op `b`. `px_per_unit` sizes a bare point's disc, as it draws.
pub fn boolean(a: &Geometry, b: &Geometry, op: BoolOp, px_per_unit: f32) -> Geometry {
    let mut out = Geometry::default();
    let b_area = area(b, px_per_unit);
    let cut = |path: &Contour, inside: bool| -> Vec<Contour> {
        if b_area.is_empty() {
            return if inside { Vec::new() } else { vec![path.clone()] };
        }
        let rule = ClipRule { invert: !inside, boundary_included: false };
        path.clip_by(&b_area, Fr::NonZero, rule)
    };
    let line = |g: &Geometry, i: usize| -> Contour {
        g.polyline(i).into_iter().map(|p| [p[0] as f64, p[1] as f64]).collect()
    };
    match op {
        BoolOp::Subtract | BoolOp::Intersect => {
            let (rule, inside) = match op {
                BoolOp::Subtract => (OverlayRule::Difference, false),
                _ => (OverlayRule::Intersect, true),
            };
            for (i, pr) in a.prims.iter().enumerate() {
                if is_open(pr) {
                    for piece in cut(&line(a, i), inside) {
                        emit_line(&mut out, &piece, pr);
                    }
                    continue;
                }
                let mine = prim_area(a, i, px_per_unit);
                if mine.is_empty() {
                    continue;
                }
                let res = if b_area.is_empty() {
                    if inside { Vec::new() } else { mine }
                } else {
                    mine.overlay(&b_area, rule, Fr::NonZero)
                };
                emit(&mut out, &res, pr);
            }
        }
        BoolOp::Union | BoolOp::Exclude => {
            let a_area = area(a, px_per_unit);
            let rule = if op == BoolOp::Union { OverlayRule::Union } else { OverlayRule::Xor };
            let res = a_area.overlay(&b_area, rule, Fr::NonZero);
            // The fused area wears the first filled shape's style (A's,
            // else B's); `src` 0 so the Boolean itself is what a click on
            // it selects.
            let like = a
                .prims
                .iter()
                .chain(b.prims.iter())
                .find(|p| !is_open(p))
                .map(|p| Prim { src: 0, ..p.clone() });
            if let Some(like) = like {
                emit(&mut out, &res, &like);
            }
            for (i, pr) in a.prims.iter().enumerate().filter(|(_, p)| is_open(p)) {
                emit_line(&mut out, &line(a, i), pr);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo::Style;

    fn square(g: &mut Geometry, lo: [f32; 2], hi: [f32; 2], fill: [f32; 4]) {
        let base = g.points.len() as PointId;
        for p in [[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]] {
            g.points.push(Point { p, ..Default::default() });
        }
        g.prims.push(Prim {
            layer: Default::default(),
            kind: PrimKind::Poly { closed: true },
            verts: (base..base + 4).collect(),
            style: Style { fill, ..Default::default() },
            phys: None,
            src: 7,
            rings: Vec::new(),
        });
    }

    fn area_of(g: &Geometry) -> f32 {
        // Shoelace per ring; holes wind against their outer, so the signed
        // sum of a prim's rings is its filled area.
        let mut t = 0.0;
        for i in 0..g.prims.len() {
            let mut s = 0.0;
            for r in g.rings(i) {
                let n = r.len();
                let mut a = 0.0;
                for k in 0..n {
                    let (p, q) = (r[k], r[(k + 1) % n]);
                    a += p[0] * q[1] - q[0] * p[1];
                }
                s += a * 0.5;
            }
            t += s.abs();
        }
        t
    }

    fn two() -> (Geometry, Geometry) {
        let (mut a, mut b) = (Geometry::default(), Geometry::default());
        square(&mut a, [0.0, 0.0], [0.4, 0.4], [1.0, 0.0, 0.0, 1.0]);
        square(&mut b, [0.2, 0.2], [0.6, 0.6], [0.0, 0.0, 1.0, 1.0]);
        (a, b)
    }

    #[test]
    fn the_four_ops_measure_right() {
        let (a, b) = two();
        let near = |g: &Geometry, want: f32| {
            let got = area_of(g);
            assert!((got - want).abs() < 1e-3, "area {got}, want {want}");
        };
        near(&boolean(&a, &b, BoolOp::Union, 1000.0), 0.16 + 0.16 - 0.04);
        near(&boolean(&a, &b, BoolOp::Intersect, 1000.0), 0.04);
        near(&boolean(&a, &b, BoolOp::Subtract, 1000.0), 0.12);
        near(&boolean(&a, &b, BoolOp::Exclude, 1000.0), 0.32 - 0.08);
    }

    #[test]
    fn subtract_keeps_each_shapes_style_and_maker() {
        let (a, b) = two();
        let g = boolean(&a, &b, BoolOp::Subtract, 1000.0);
        assert!(!g.prims.is_empty());
        for p in &g.prims {
            assert_eq!(p.style.fill, [1.0, 0.0, 0.0, 1.0]);
            assert_eq!(p.src, 7);
        }
    }

    #[test]
    fn a_hole_punched_through_is_a_ring() {
        let (mut a, mut b) = (Geometry::default(), Geometry::default());
        square(&mut a, [0.0, 0.0], [1.0, 1.0], [1.0; 4]);
        square(&mut b, [0.4, 0.4], [0.6, 0.6], [1.0; 4]);
        let g = boolean(&a, &b, BoolOp::Subtract, 1000.0);
        assert_eq!(g.prims.len(), 1);
        assert_eq!(g.prims[0].rings.len(), 1, "the cut is a counter of the one shape");
        assert!(!g.contains([0.5, 0.5]) && g.contains([0.1, 0.1]));
    }

    #[test]
    fn a_line_is_cut_by_the_area() {
        let (mut a, mut b) = (Geometry::default(), Geometry::default());
        for p in [[0.0, 0.5], [1.0, 0.5]] {
            a.points.push(Point { p, ..Default::default() });
        }
        a.prims.push(Prim {
            layer: Default::default(),
            kind: PrimKind::Poly { closed: false },
            verts: vec![0, 1],
            style: Style::default(),
            phys: None,
            src: 0,
            rings: Vec::new(),
        });
        square(&mut b, [0.25, 0.0], [0.75, 1.0], [1.0; 4]);
        let out = boolean(&a, &b, BoolOp::Subtract, 1000.0);
        assert_eq!(out.prims.len(), 2, "one each side of the cut");
        let inn = boolean(&a, &b, BoolOp::Intersect, 1000.0);
        assert_eq!(inn.prims.len(), 1);
        let pl = inn.polyline(0);
        let len = (pl[1][0] - pl[0][0]).abs();
        assert!((len - 0.5).abs() < 1e-3, "{len}");
    }

    #[test]
    fn a_point_cuts_as_its_disc() {
        let (mut a, mut b) = (Geometry::default(), Geometry::default());
        square(&mut a, [0.0, 0.0], [1.0, 1.0], [1.0; 4]);
        b.points.push(Point { p: [0.5, 0.5], pscale: 100.0, ..Default::default() });
        b.prims.push(Prim {
            layer: Default::default(),
            kind: PrimKind::Point,
            verts: vec![0],
            style: Style::default(),
            phys: None,
            src: 0,
            rings: Vec::new(),
        });
        // 100 px at 1000 px per unit: a 0.1 disc.
        let g = boolean(&a, &b, BoolOp::Subtract, 1000.0);
        let hole = std::f32::consts::PI * 0.01;
        assert!((area_of(&g) - (1.0 - hole)).abs() < 2e-3, "{}", area_of(&g));
    }
}
