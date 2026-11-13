//! Mirror: reflect the picture across a line.
//!
//! The line passes through `center` (normalised over the node's own
//! format) at `angle` degrees — 90 is vertical, the default, so a fresh
//! Mirror folds the left half onto the right. `side` says which half is
//! kept: the FRONT (the side the handle's arrow points to, the line's
//! normal), the BACK, or BOTH, which reflects the whole picture across the
//! line. It fuses like a UV Warp: the input is evaluated at the reflected
//! position (`Emit::input_at`), no pass and no texture. The side is a
//! uniform, so switching it never recompiles.

use crate::codegen::Emit;
use crate::nodes::NodeDef;

/// Which half of the picture the mirror keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MirrorSide {
    /// The half the line's normal points into, reflected onto the other.
    #[default]
    Front,
    /// The other half, reflected onto the front.
    Back,
    /// No half kept: the whole picture reflected across the line.
    Both,
}

impl MirrorSide {
    pub const ALL: [MirrorSide; 3] = [MirrorSide::Front, MirrorSide::Back, MirrorSide::Both];

    pub fn name(self) -> &'static str {
        match self {
            MirrorSide::Front => "keep front",
            MirrorSide::Back => "keep back",
            MirrorSide::Both => "reflect all",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct MirrorParams {
    /// A point on the line, normalised over the node's format (y down).
    pub center: [f32; 2],
    /// The line's direction in degrees, counter-clockwise on screen from
    /// +x (a Gradient's convention). 90 is a vertical line.
    pub angle: f32,
    pub side: MirrorSide,
    /// Which pipe it sits in: wired a MATTE it mirrors the coverage and
    /// gives a matte (`NodeKind::follows_pipe`).
    pub matte: bool,
}

impl Default for MirrorParams {
    fn default() -> Self {
        Self { center: [0.5, 0.5], angle: 90.0, side: MirrorSide::Front, matte: false }
    }
}

impl MirrorParams {
    /// The line's direction in pixels (y down).
    pub fn direction(&self) -> [f32; 2] {
        let a = if self.angle.is_finite() { self.angle.to_radians() } else { 0.0 };
        let (s, c) = a.sin_cos();
        [c, -s]
    }

    /// The line's unit normal in pixels: the direction turned a quarter
    /// counter-clockwise on screen. The FRONT is where it points — left of
    /// a vertical line, above a horizontal one.
    pub fn normal(&self) -> [f32; 2] {
        let d = self.direction();
        [d[1], -d[0]]
    }

    /// The angle whose normal points along `v` (pixels, y down): what the
    /// viewer handle's arrow sets.
    pub fn angle_facing(v: [f32; 2]) -> f32 {
        (-v[0]).atan2(-v[1]).to_degrees()
    }

    /// Where output pixel `p` samples the input, in a `fmt`-pixel format —
    /// the CPU mirror of the generated code.
    pub fn lookup(&self, p: [f32; 2], fmt: [f32; 2]) -> [f32; 2] {
        let c = [self.center[0] * fmt[0], self.center[1] * fmt[1]];
        let n = self.normal();
        let s = (p[0] - c[0]) * n[0] + (p[1] - c[1]) * n[1];
        let reflect = match self.side {
            MirrorSide::Front => s < 0.0,
            MirrorSide::Back => s > 0.0,
            MirrorSide::Both => true,
        };
        if reflect {
            [p[0] - 2.0 * s * n[0], p[1] - 2.0 * s * n[1]]
        } else {
            p
        }
    }

    /// The uniform span: (cx, cy, nx, ny) in the format's pixels, then
    /// (side, -, -, -).
    pub fn pack_fmt(&self, v: &mut [f32], fmt: [f32; 2]) {
        let n = self.normal();
        v[0] = self.center[0] * fmt[0];
        v[1] = self.center[1] * fmt[1];
        v[2] = n[0];
        v[3] = n[1];
        v[4] = self.side as u32 as f32;
        v[5] = 0.0;
        v[6] = 0.0;
        v[7] = 0.0;
    }
}

impl NodeDef for MirrorParams {
    const NAME: &'static str = "Mirror";
    /// The picture, then the mask.
    const INPUTS: usize = 2;
    const MASK_INPUT: Option<usize> = Some(1);
    const VEC4_SLOTS: usize = 2;

    fn topo_tag(&self) -> u64 {
        0x313E
    }

    fn pack(&self, v: &mut [f32]) {
        self.pack_fmt(v, [1.0, 1.0]);
    }

    fn emit(&self, e: &mut Emit) -> String {
        if !e.has_input(0) {
            return "vec4f(0.0)".into();
        }
        let slot = e.slot();
        let pos = e.pos().to_string();
        let (s, side, flip, q) = (e.fresh(), e.fresh(), e.fresh(), e.fresh());
        e.line(format_args!("let {s} = dot({pos} - P.v[{slot}].xy, P.v[{slot}].zw);"));
        e.line(format_args!("let {side} = P.v[{}].x;", slot + 1));
        e.line(format_args!(
            "let {flip} = select(select(({s} < 0.0), ({s} > 0.0), ({side} > 0.5)), true, ({side} > 1.5));"
        ));
        e.line(format_args!(
            "let {q} = select({pos}, {pos} - 2.0 * {s} * P.v[{slot}].zw, {flip});"
        ));
        let out = e.input_at(0, &q);
        if e.has_input(1) {
            let orig = e.input(0);
            e.masked(Self::MASK_INPUT, &orig, out)
        } else {
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_mirror_folds_the_left_half_onto_the_right() {
        let m = MirrorParams::default();
        let fmt = [100.0, 50.0];
        // The normal of a vertical line points LEFT: the left is kept.
        assert_eq!(m.lookup([20.0, 10.0], fmt), [20.0, 10.0]);
        let r = m.lookup([70.0, 10.0], fmt);
        assert!((r[0] - 30.0).abs() < 1e-4 && (r[1] - 10.0).abs() < 1e-4, "{r:?}");
        let back = MirrorParams { side: MirrorSide::Back, ..m };
        assert_eq!(back.lookup([70.0, 10.0], fmt), [70.0, 10.0]);
        let all = MirrorParams { side: MirrorSide::Both, ..m };
        let r = all.lookup([70.0, 10.0], fmt);
        assert!((r[0] - 30.0).abs() < 1e-4);
    }

    #[test]
    fn a_turned_line_reflects_across_itself() {
        // A horizontal line (0°) a quarter down: y reflects about 12.5.
        let m = MirrorParams { center: [0.5, 0.25], angle: 0.0, side: MirrorSide::Both, ..Default::default() };
        let r = m.lookup([10.0, 20.0], [100.0, 50.0]);
        assert!((r[0] - 10.0).abs() < 1e-4 && (r[1] - 5.0).abs() < 1e-4, "{r:?}");
        // -45° (down-right on screen): (x, y) about the corner swaps.
        let d = MirrorParams { center: [0.0, 0.0], angle: -45.0, side: MirrorSide::Both, ..Default::default() };
        let r = d.lookup([30.0, 10.0], [100.0, 100.0]);
        assert!((r[0] - 10.0).abs() < 1e-3 && (r[1] - 30.0).abs() < 1e-3, "{r:?}");
    }

    #[test]
    fn the_arrow_sets_the_angle_whose_normal_it_is() {
        for a in [-170.0f32, -90.0, -30.0, 0.0, 45.0, 90.0, 135.0] {
            let n = MirrorParams { angle: a, ..Default::default() }.normal();
            let back = MirrorParams::angle_facing(n);
            let d = (back - a).rem_euclid(360.0);
            assert!(d < 1e-3 || d > 360.0 - 1e-3, "{a} came back {back}");
        }
        // A horizontal line keeps what is above it.
        let m = MirrorParams { angle: 0.0, ..Default::default() };
        assert_eq!(m.lookup([10.0, 5.0], [100.0, 50.0]), [10.0, 5.0]);
    }
}
