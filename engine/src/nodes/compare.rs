//! Compare: two pictures, shown one at a time (A / B) or split by a wipe.
//!
//! Input 0 is A, input 1 is B. The node's format is the LARGER of the two
//! (by pixel count; the app's `node_resolution` decides it), and each input
//! is scaled into it about its centre — so a 1K proxy compared against the
//! 2K plate it came from lines up pixel for pixel. When the two share an
//! aspect that is a plain fill; when they do not, [`CompareFit`] says which
//! dimension is matched (width or height) and the rest of the picture is
//! letterboxed or overhangs, transparent black where an input has no pixels.
//!
//! The wipe is a line through `center` (normalised over the node's format)
//! at `angle` degrees, Mirror's convention: 90 is vertical, and A shows on
//! the side the line's normal points to (left of a vertical line). The
//! viewer draws the line and its handles; the pixels carry no line.
//!
//! Everything — the mode included — is a uniform, so flicking A/B or
//! dragging the wipe never recompiles. Both inputs are always evaluated
//! (a toggle that had to wait on a decode would not be a toggle); it fuses,
//! two `input_at`s, no pass, no texture.

use crate::codegen::Emit;
use crate::nodes::NodeDef;

/// What the node shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CompareMode {
    A,
    B,
    /// A on the front of the wipe line, B behind it.
    #[default]
    Wipe,
}

impl CompareMode {
    pub const ALL: [CompareMode; 3] = [CompareMode::A, CompareMode::B, CompareMode::Wipe];

    pub fn name(self) -> &'static str {
        match self {
            CompareMode::A => "A",
            CompareMode::B => "B",
            CompareMode::Wipe => "wipe",
        }
    }
}

/// Which dimension an input is scaled to match when the aspects differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CompareFit {
    #[default]
    Width,
    Height,
}

impl CompareFit {
    pub const ALL: [CompareFit; 2] = [CompareFit::Width, CompareFit::Height];

    pub fn name(self) -> &'static str {
        match self {
            CompareFit::Width => "fit width",
            CompareFit::Height => "fit height",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct CompareParams {
    pub mode: CompareMode,
    /// A point on the wipe line, normalised over the node's format (y down).
    pub center: [f32; 2],
    /// The wipe line's direction in degrees, counter-clockwise on screen
    /// from +x. 90 is a vertical line.
    pub angle: f32,
    pub fit: CompareFit,
}

impl Default for CompareParams {
    fn default() -> Self {
        Self { mode: CompareMode::Wipe, center: [0.5, 0.5], angle: 90.0, fit: CompareFit::Width }
    }
}

/// The format a Compare outputs: the larger input's (by pixel count; A on a
/// tie), whichever is wired.
pub fn output_format(a: Option<(u32, u32)>, b: Option<(u32, u32)>) -> Option<(u32, u32)> {
    match (a, b) {
        (Some(a), Some(b)) => {
            Some(if (b.0 as u64) * (b.1 as u64) > (a.0 as u64) * (a.1 as u64) { b } else { a })
        }
        (a, b) => a.or(b),
    }
}

impl CompareParams {
    fn line(&self) -> crate::nodes::mirror::MirrorParams {
        crate::nodes::mirror::MirrorParams { center: self.center, angle: self.angle, ..Default::default() }
    }

    /// The wipe line's direction in pixels (y down).
    pub fn direction(&self) -> [f32; 2] {
        self.line().direction()
    }

    /// The wipe line's unit normal in pixels: A is the side it points to.
    pub fn normal(&self) -> [f32; 2] {
        self.line().normal()
    }

    /// The angle whose normal points along `v`: what the viewer arrow sets.
    pub fn angle_facing(v: [f32; 2]) -> f32 {
        crate::nodes::mirror::MirrorParams::angle_facing(v)
    }

    /// Input pixels per output pixel, and the offset, for an `input`-pixel
    /// picture scaled into the `out`-pixel format about the centres:
    /// `q = p * s + off`.
    pub fn fit_of(&self, out: [f32; 2], input: [f32; 2]) -> (f32, [f32; 2]) {
        let s = match self.fit {
            CompareFit::Width => input[0] / out[0].max(1.0),
            CompareFit::Height => input[1] / out[1].max(1.0),
        };
        (s, [input[0] * 0.5 - out[0] * 0.5 * s, input[1] * 0.5 - out[1] * 0.5 * s])
    }

    /// Where output pixel `p` reads input A (`b = false`) or B, and whether
    /// that input has a pixel there — the CPU mirror of the generated code.
    pub fn lookup(&self, p: [f32; 2], out: [f32; 2], input: [f32; 2]) -> Option<[f32; 2]> {
        let (s, o) = self.fit_of(out, input);
        let q = [p[0] * s + o[0], p[1] * s + o[1]];
        (q[0] >= 0.0 && q[1] >= 0.0 && q[0] < input[0] && q[1] < input[1]).then_some(q)
    }

    /// Whether output pixel `p` shows A (else B), in an `out`-pixel format.
    pub fn shows_a(&self, p: [f32; 2], out: [f32; 2]) -> bool {
        match self.mode {
            CompareMode::A => true,
            CompareMode::B => false,
            CompareMode::Wipe => {
                let n = self.normal();
                let c = [self.center[0] * out[0], self.center[1] * out[1]];
                (p[0] - c[0]) * n[0] + (p[1] - c[1]) * n[1] >= 0.0
            }
        }
    }

    /// The uniform span, all in pixels of the node's own format:
    /// (sA, offA.x, offA.y, mode), (sB, offB.x, offB.y, 0),
    /// (wA, hA, wB, hB), (cx, cy, nx, ny).
    pub fn pack_fmt(&self, v: &mut [f32], out: [f32; 2], a: [f32; 2], b: [f32; 2]) {
        if v.len() < 16 {
            return;
        }
        let (sa, oa) = self.fit_of(out, a);
        let (sb, ob) = self.fit_of(out, b);
        let n = self.normal();
        v[..16].copy_from_slice(&[
            sa,
            oa[0],
            oa[1],
            self.mode as u32 as f32,
            sb,
            ob[0],
            ob[1],
            0.0,
            a[0],
            a[1],
            b[0],
            b[1],
            self.center[0] * out[0],
            self.center[1] * out[1],
            n[0],
            n[1],
        ]);
    }
}

impl NodeDef for CompareParams {
    const NAME: &'static str = "Compare";
    const INPUTS: usize = 2;
    const MASK_INPUT: Option<usize> = None;
    const VEC4_SLOTS: usize = 4;

    fn topo_tag(&self) -> u64 {
        0xC0B9
    }

    fn pack(&self, v: &mut [f32]) {
        self.pack_fmt(v, [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]);
    }

    fn emit(&self, e: &mut Emit) -> String {
        let (slot, pos) = (e.slot(), e.pos().to_string());
        let side = |e: &mut Emit, i: usize, fit: usize, size: &str| -> String {
            if !e.has_input(i) {
                return "vec4f(0.0)".into();
            }
            let q = e.fresh();
            e.line(format_args!("let {q} = {pos} * P.v[{fit}u].x + P.v[{fit}u].yz;"));
            let px = e.input_at(i, &q);
            let v = e.fresh();
            e.line(format_args!(
                "let {v} = select(vec4f(0.0), {px}, all({q} >= vec2f(0.0)) && all({q} < P.v[{}u].{size}));",
                slot + 2
            ));
            v
        };
        let a = side(e, 0, slot, "xy");
        let b = side(e, 1, slot + 1, "zw");
        let (d, m, out) = (e.fresh(), e.fresh(), e.fresh());
        e.line(format_args!("let {d} = dot({pos} - P.v[{}u].xy, P.v[{}u].zw);", slot + 3, slot + 3));
        e.line(format_args!("let {m} = P.v[{slot}u].w;"));
        e.line(format_args!(
            "let {out} = select(select({b}, {a}, {d} >= 0.0), select({a}, {b}, {m} > 0.5), {m} < 1.5);"
        ));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_larger_input_names_the_format() {
        assert_eq!(output_format(Some((1024, 540)), Some((2048, 1080))), Some((2048, 1080)));
        assert_eq!(output_format(Some((2048, 1080)), Some((1024, 540))), Some((2048, 1080)));
        assert_eq!(output_format(None, Some((640, 480))), Some((640, 480)));
        assert_eq!(output_format(None, None), None);
    }

    /// Same aspect: the half-size input is scaled up to fill, pixel centre
    /// to pixel centre, whichever fit is picked.
    #[test]
    fn a_smaller_same_aspect_input_fills_the_frame() {
        for fit in CompareFit::ALL {
            let p = CompareParams { fit, ..Default::default() };
            let out = [200.0, 100.0];
            let half = [100.0, 50.0];
            assert_eq!(p.lookup([100.0, 50.0], out, half), Some([50.0, 25.0]));
            assert_eq!(p.lookup([0.0, 0.0], out, half), Some([0.0, 0.0]));
            assert!(p.lookup([199.9, 99.9], out, half).is_some());
            // The larger input is read 1:1.
            assert_eq!(p.lookup([10.5, 20.5], out, out), Some([10.5, 20.5]));
        }
    }

    /// Different aspects: fit width matches widths and letterboxes a
    /// squarer input's overhang; fit height matches heights.
    #[test]
    fn differing_aspects_fit_by_the_picked_dimension() {
        let out = [200.0, 100.0];
        let square = [100.0, 100.0];
        let w = CompareParams { fit: CompareFit::Width, ..Default::default() };
        // 2x up: the square is 200 tall, centred, so only its middle shows.
        assert_eq!(w.lookup([100.0, 50.0], out, square), Some([50.0, 50.0]));
        assert_eq!(w.lookup([0.0, 0.0], out, square), Some([0.0, 25.0]));
        let h = CompareParams { fit: CompareFit::Height, ..Default::default() };
        // 1:1, centred: 50 px of black either side.
        assert!(h.lookup([25.0, 50.0], out, square).is_none());
        assert_eq!(h.lookup([60.0, 50.0], out, square), Some([10.0, 50.0]));
        assert!(h.lookup([175.0, 50.0], out, square).is_none());
    }

    #[test]
    fn a_fresh_wipe_shows_a_on_the_left() {
        let p = CompareParams::default();
        let out = [100.0, 50.0];
        assert!(p.shows_a([20.0, 10.0], out));
        assert!(!p.shows_a([80.0, 10.0], out));
        assert!(CompareParams { mode: CompareMode::A, ..p }.shows_a([80.0, 10.0], out));
        assert!(!CompareParams { mode: CompareMode::B, ..p }.shows_a([20.0, 10.0], out));
    }

    #[test]
    fn nothing_but_the_kind_is_topology() {
        let a = CompareParams::default();
        let b = CompareParams { mode: CompareMode::B, center: [0.1, 0.9], angle: 12.0, fit: CompareFit::Height };
        assert_eq!(a.topo_tag(), b.topo_tag());
    }
}
