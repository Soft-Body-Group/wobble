//! Gradient: a generator that fills its format with a colour ramp — a
//! Constant, with the one colour spread along an axis.
//!
//! The ramp is a list of STOPS (position 0..1 along the axis, colour + alpha
//! each) and an axis: an `angle`, a `center` normalised to the format, and a
//! `scale`. At scale 1 the ramp runs exactly corner to corner of the format
//! along the axis, whatever the angle — position 0 lands on the first corner
//! the axis meets and 1 on the last — so turning the angle never leaves an
//! unfilled band or pushes an end off the frame.
//!
//! **Interpolation** is the one structural choice. `Linear` mixes the
//! scene-linear numbers, which is what light does and what a classic Ramp
//! does. `Lch` mixes in OkLCh — lightness, chroma, hue — taking the short
//! way round the hue circle, so red to blue passes through a real magenta
//! instead of the muddy mid-grey a linear mix of complements falls into.
//! A stop with no chroma lends the other end its hue, so a fade to grey
//! does not swing through every hue on the way.
//!
//! Stops are uniforms at a FIXED capacity ([`GRAD_STOPS`]): adding one,
//! removing one or dragging it past a neighbour is a uniform write, never a
//! recompile. Only the interpolation picks the code. The Vec keeps stops in
//! AUTHORED order (a stop's index is its identity, so its keyed position
//! channel stays its own when it is dragged past another); `pack` sorts.
//!
//! Output is premultiplied: the colours are authored straight, mixed
//! straight, and multiplied by their alpha at the end.

use crate::codegen::Emit;
use crate::color::Colorspace;
use crate::nodes::NodeDef;
use std::collections::BTreeMap;

/// Fixed stop capacity. The span is sized for it so the stop COUNT is a
/// uniform; the editor stops adding at the cap.
pub const GRAD_STOPS: usize = 12;
/// vec4 slots of stop positions, four to a slot.
const POS_SLOTS: usize = GRAD_STOPS.div_ceil(4);

/// Topology base, clear of every other node's span.
const TOPO_BASE: u64 = 0x6A00;

/// What the stops are mixed in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GradInterp {
    /// Scene-linear RGB, straight: light mixing.
    #[default]
    Linear,
    /// OkLCh, shortest hue: perceptual, keeps its saturation through the
    /// middle.
    Lch,
}

impl GradInterp {
    pub const ALL: [GradInterp; 2] = [GradInterp::Linear, GradInterp::Lch];

    pub fn label(self) -> &'static str {
        match self {
            GradInterp::Linear => "linear",
            GradInterp::Lch => "lch",
        }
    }
}

/// One colour on the ramp.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GradStop {
    /// Where along the axis, 0..1.
    pub pos: f32,
    /// Straight (unpremultiplied) colour in the working space, and alpha.
    pub color: [f32; 4],
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct GradientParams {
    /// The node's format: a generator declares one, like Grid.
    pub size: (u32, u32),
    /// Take the comp's canvas instead of `size` (see `GridParams`). A
    /// field default, so an old setup keeps its declared size.
    #[cfg_attr(feature = "serde", serde(default))]
    pub inherit: bool,
    /// Authored order; see the module docs. Never empty in practice, and
    /// an empty list renders transparent rather than failing.
    pub stops: Vec<GradStop>,
    /// Axis direction in degrees, counter-clockwise on screen; 0 runs left
    /// to right, 90 bottom to top.
    pub angle: f32,
    /// Where position 0.5 sits, normalised to the format.
    pub center: [f32; 2],
    /// Ramp length relative to the corner-to-corner span along the axis.
    pub scale: f32,
    pub interp: GradInterp,
}

impl Default for GradientParams {
    fn default() -> Self {
        Self {
            size: (1920, 1080),
            inherit: false,
            stops: vec![
                GradStop { pos: 0.0, color: [0.0, 0.0, 0.0, 1.0] },
                GradStop { pos: 1.0, color: [1.0, 1.0, 1.0, 1.0] },
            ],
            angle: 0.0,
            center: [0.5, 0.5],
            scale: 1.0,
            interp: GradInterp::Linear,
        }
    }
}

impl GradientParams {
    /// Unit axis direction in pixel space (y down).
    pub fn dir(&self) -> [f32; 2] {
        let a = self.angle.to_radians();
        [a.cos(), -a.sin()]
    }

    /// Half the ramp's length in pixels of a `w` x `h` format: half the
    /// format's extent along the axis, times `scale`.
    pub fn half_len(&self, w: f32, h: f32) -> f32 {
        let [dx, dy] = self.dir();
        (0.5 * (dx.abs() * w + dy.abs() * h) * self.scale).max(1e-3)
    }

    /// The pixel positions of ramp position 0 and 1 in a `w` x `h` format.
    pub fn ends(&self, w: f32, h: f32) -> ([f32; 2], [f32; 2]) {
        let [dx, dy] = self.dir();
        let l = self.half_len(w, h);
        let c = [self.center[0] * w, self.center[1] * h];
        ([c[0] - dx * l, c[1] - dy * l], [c[0] + dx * l, c[1] + dy * l])
    }

    /// The ramp position of pixel `p` in a `w` x `h` format (unclamped).
    pub fn t_at(&self, p: [f32; 2], w: f32, h: f32) -> f32 {
        let [dx, dy] = self.dir();
        let q = [p[0] - self.center[0] * w, p[1] - self.center[1] * h];
        (q[0] * dx + q[1] * dy) / (2.0 * self.half_len(w, h)) + 0.5
    }

    /// Stop indices in position order (ties keep authored order).
    pub fn order(&self) -> Vec<usize> {
        let mut o: Vec<usize> = (0..self.stops.len().min(GRAD_STOPS)).collect();
        o.sort_by(|a, b| self.stops[*a].pos.total_cmp(&self.stops[*b].pos));
        o
    }

    /// The STRAIGHT colour at ramp position `t` — the CPU mirror of the
    /// shader, for the editor's bar and for tests.
    pub fn sample(&self, t: f32, working: Colorspace) -> [f32; 4] {
        let o = self.order();
        let Some(&first) = o.first() else { return [0.0; 4] };
        let last = *o.last().unwrap();
        if t <= self.stops[first].pos {
            return self.stops[first].color;
        }
        if t >= self.stops[last].pos {
            return self.stops[last].color;
        }
        for w in o.windows(2) {
            let (a, b) = (self.stops[w[0]], self.stops[w[1]]);
            if t <= b.pos {
                let f = ((t - a.pos) / (b.pos - a.pos).max(1e-6)).clamp(0.0, 1.0);
                return mix(a.color, b.color, f, self.interp, working);
            }
        }
        self.stops[last].color
    }

    /// Insert a stop at `t` carrying the colour the ramp has there, so the
    /// picture does not change until it is edited. `None` at capacity.
    pub fn add_stop(&mut self, t: f32, working: Colorspace) -> Option<usize> {
        if self.stops.len() >= GRAD_STOPS {
            return None;
        }
        let color = self.sample(t, working);
        self.stops.push(GradStop { pos: t.clamp(0.0, 1.0), color });
        Some(self.stops.len() - 1)
    }
}

/// OkLCh of a working-space colour.
fn lch_of(c: [f32; 3], working: Colorspace) -> [f32; 3] {
    let lab = crate::keyer::oklab_from_linear_in(c, crate::keyer::oklab_matrix(working));
    [lab[0], lab[1].hypot(lab[2]), lab[2].atan2(lab[1])]
}

/// Chroma under which a stop is treated as having no hue of its own.
const GREY_C: f32 = 1e-4;

/// Two straight colours mixed at `f`.
fn mix(a: [f32; 4], b: [f32; 4], f: f32, interp: GradInterp, working: Colorspace) -> [f32; 4] {
    let alpha = a[3] + (b[3] - a[3]) * f;
    match interp {
        GradInterp::Linear => {
            let m = |i: usize| a[i] + (b[i] - a[i]) * f;
            [m(0), m(1), m(2), alpha]
        }
        GradInterp::Lch => {
            let la = lch_of([a[0], a[1], a[2]], working);
            let lb = lch_of([b[0], b[1], b[2]], working);
            let (ha, hb) = match (la[1] < GREY_C, lb[1] < GREY_C) {
                (true, false) => (lb[2], lb[2]),
                (false, true) => (la[2], la[2]),
                _ => (la[2], lb[2]),
            };
            let tau = std::f32::consts::TAU;
            let mut d = (hb - ha) % tau;
            if d > std::f32::consts::PI {
                d -= tau;
            } else if d < -std::f32::consts::PI {
                d += tau;
            }
            let l = la[0] + (lb[0] - la[0]) * f;
            let c = la[1] + (lb[1] - la[1]) * f;
            let h = ha + d * f;
            let rgb = crate::nodes::saturation::linear_from_oklab([l, c * h.cos(), c * h.sin()], working);
            [rgb[0], rgb[1], rgb[2], alpha]
        }
    }
}

/// Emit the node's WGSL function once per shader and return its name.
fn wgsl(
    interp: GradInterp,
    working: Colorspace,
    helpers: &mut BTreeMap<String, String>,
    fns: &mut BTreeMap<String, String>,
) -> String {
    let name = format!("grad_px_{}", interp.label());
    if fns.contains_key(&name) {
        return name;
    }
    let mix_fn = match interp {
        GradInterp::Linear => "mix(a, b, f)".to_string(),
        GradInterp::Lch => {
            let to_lin = crate::nodes::saturation::oklab_inverse_wgsl(working, helpers);
            if !fns.contains_key("grad_lch_mix") {
                fns.insert(
                    "grad_lch_mix".into(),
                    format!(
                        "// Two straight colours mixed in OkLCh, the short way round the hue
// circle; a stop with no chroma borrows the other's hue.
fn grad_lch_mix(a: vec4f, b: vec4f, f: f32) -> vec4f {{
    let la = oklab_from_linear(a.rgb);
    let lb = oklab_from_linear(b.rgb);
    let ca = length(la.yz);
    let cb = length(lb.yz);
    var ha = atan2(la.z, la.y);
    var hb = atan2(lb.z, lb.y);
    if (ca < {GREY_C:e}) {{ ha = hb; }}
    if (cb < {GREY_C:e}) {{ hb = ha; }}
    var d = hb - ha;
    d = d - 6.2831853 * round(d / 6.2831853);
    let h = ha + d * f;
    let c = mix(ca, cb, f);
    let l = mix(la.x, lb.x, f);
    return vec4f({to_lin}(vec3f(l, c * cos(h), c * sin(h))), mix(a.a, b.a, f));
}}
"
                    ),
                );
            }
            "grad_lch_mix(a, b, f)".to_string()
        }
    };
    let pos0 = 2;
    let col0 = 2 + POS_SLOTS;
    fns.insert(
        name.clone(),
        format!(
            "// Gradient: the ramp position of this pixel along the node's axis, then
// the two stops either side of it mixed ({interp}). Stops arrive sorted.
fn {name}(p: vec2f, slot: u32) -> vec4f {{
    let h = P.v[slot];
    let g = P.v[slot + 1u];
    let n = u32(h.z);
    if (n == 0u) {{ return vec4f(0.0); }}
    let fmt = max(vec2f(h.w, g.w), vec2f(1.0));
    let dir = h.xy;
    let half = max(0.5 * (abs(dir.x) * fmt.x + abs(dir.y) * fmt.y) * g.z, 1e-3);
    let t = dot(p - g.xy * fmt, dir) / (2.0 * half) + 0.5;
    var c = P.v[slot + {col0}u];
    if (t > P.v[slot + {pos0}u].x) {{
        c = P.v[slot + {col0}u + n - 1u];
        for (var i = 1u; i < n; i = i + 1u) {{
            let pb = P.v[slot + {pos0}u + (i >> 2u)][i & 3u];
            if (t <= pb) {{
                let j = i - 1u;
                let pa = P.v[slot + {pos0}u + (j >> 2u)][j & 3u];
                let a = P.v[slot + {col0}u + j];
                let b = P.v[slot + {col0}u + i];
                let f = clamp((t - pa) / max(pb - pa, 1e-6), 0.0, 1.0);
                c = {mix_fn};
                break;
            }}
        }}
    }}
    return vec4f(c.rgb * c.a, c.a);
}}
",
            interp = interp.label(),
        ),
    );
    name
}

impl GradientParams {
    /// [`NodeDef::pack`] with the node's format, which only the codegen
    /// arm knows (formats are stamped per node, proxy included).
    pub fn pack_fmt(&self, v: &mut [f32], fmt: [f32; 2]) {
        let [dx, dy] = self.dir();
        let o = self.order();
        v[0] = dx;
        v[1] = dy;
        v[2] = o.len() as f32;
        v[3] = fmt[0];
        v[4] = self.center[0];
        v[5] = self.center[1];
        v[6] = self.scale;
        v[7] = fmt[1];
        for (k, i) in o.iter().enumerate() {
            let s = self.stops[*i];
            v[8 + k] = s.pos;
            let c = (2 + POS_SLOTS + k) * 4;
            v[c..c + 4].copy_from_slice(&s.color);
        }
    }
}

impl NodeDef for GradientParams {
    const NAME: &'static str = "Gradient";
    const INPUTS: usize = 0;
    const MASK_INPUT: Option<usize> = None;
    /// (dir, count, fmt.x), (center, scale, fmt.y), positions, colours.
    const VEC4_SLOTS: usize = 2 + POS_SLOTS + GRAD_STOPS;

    /// Only the interpolation picks the code; stops, their count and the
    /// axis are all uniforms.
    fn topo_tag(&self) -> u64 {
        TOPO_BASE + self.interp as u64
    }

    fn pack(&self, v: &mut [f32]) {
        self.pack_fmt(v, [self.size.0 as f32, self.size.1 as f32]);
    }

    fn emit(&self, e: &mut Emit) -> String {
        let slot = e.slot();
        let working = e.working();
        let pos = e.pos().to_string();
        let (curves, fns) = e.color_helpers();
        let f = wgsl(self.interp, working, curves, fns);
        let v = e.fresh();
        e.line(format_args!("let {v} = {f}({pos}, {slot}u);"));
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::{Curve, Primaries};

    fn ws() -> Colorspace {
        Colorspace::new(Primaries::Rec709, Curve::Linear)
    }

    #[test]
    fn scale_one_spans_corner_to_corner_at_any_angle() {
        let (w, h) = (1920.0, 1080.0);
        for a in [0.0, 30.0, 90.0, 135.0, 200.0, 333.0] {
            let p = GradientParams { angle: a, ..Default::default() };
            let ts: Vec<f32> =
                [[0.0, 0.0], [w, 0.0], [0.0, h], [w, h]].iter().map(|c| p.t_at(*c, w, h)).collect();
            let lo = ts.iter().cloned().fold(f32::MAX, f32::min);
            let hi = ts.iter().cloned().fold(f32::MIN, f32::max);
            assert!(lo.abs() < 1e-4 && (hi - 1.0).abs() < 1e-4, "angle {a}: {lo}..{hi}");
        }
    }

    #[test]
    fn zero_degrees_runs_left_to_right_and_ninety_bottom_to_top() {
        let p = GradientParams::default();
        assert!(p.t_at([0.0, 540.0], 1920.0, 1080.0).abs() < 1e-5);
        let p = GradientParams { angle: 90.0, ..Default::default() };
        assert!(p.t_at([960.0, 1080.0], 1920.0, 1080.0).abs() < 1e-4);
        assert!((p.t_at([960.0, 0.0], 1920.0, 1080.0) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn stops_are_found_in_position_order_whatever_the_authored_order() {
        let mut p = GradientParams::default();
        p.stops.swap(0, 1);
        p.stops[0].pos = 1.0;
        p.stops[1].pos = 0.0;
        assert_eq!(p.sample(0.25, ws())[0], 0.25);
    }

    #[test]
    fn lch_keeps_chroma_through_the_middle_where_linear_greys_out() {
        let red = [1.0, 0.0, 0.0, 1.0];
        let cyan = [0.0, 1.0, 1.0, 1.0];
        let mut p = GradientParams::default();
        p.stops[0].color = red;
        p.stops[1].color = cyan;
        let lin = p.sample(0.5, ws());
        p.interp = GradInterp::Lch;
        let lch = p.sample(0.5, ws());
        let chroma = |c: [f32; 4]| lch_of([c[0], c[1], c[2]], ws())[1];
        assert!(chroma(lin) < 0.02, "linear red/cyan midpoint is grey");
        assert!(chroma(lch) > 0.1, "lch keeps its chroma: {}", chroma(lch));
        // The ends are the stops, exactly.
        let e = p.sample(0.0, ws());
        assert!((e[0] - 1.0).abs() < 1e-3 && e[1].abs() < 1e-3);
    }

    #[test]
    fn a_grey_stop_lends_no_hue() {
        let mut p = GradientParams { interp: GradInterp::Lch, ..Default::default() };
        p.stops[0].color = [0.5, 0.5, 0.5, 1.0];
        p.stops[1].color = [0.0, 0.0, 1.0, 1.0];
        let blue_h = lch_of([0.0, 0.0, 1.0], ws())[2];
        for t in [0.2, 0.5, 0.8] {
            let c = p.sample(t, ws());
            let h = lch_of([c[0], c[1], c[2]], ws())[2];
            assert!((h - blue_h).abs() < 0.02, "t {t}: hue {h} vs {blue_h}");
        }
    }

    #[test]
    fn a_new_stop_changes_nothing_until_edited() {
        let mut p = GradientParams::default();
        let before = p.sample(0.3, ws());
        p.add_stop(0.3, ws()).unwrap();
        for t in [0.1, 0.3, 0.6] {
            let a = GradientParams::default().sample(t, ws());
            let b = p.sample(t, ws());
            assert!((a[0] - b[0]).abs() < 1e-5, "t {t}");
        }
        assert_eq!(p.stops[2].color, before);
    }

    #[test]
    fn stops_stop_at_capacity() {
        let mut p = GradientParams::default();
        while p.add_stop(0.5, ws()).is_some() {}
        assert_eq!(p.stops.len(), GRAD_STOPS);
    }

    #[test]
    fn the_topology_is_the_interpolation_alone() {
        let a = GradientParams::default();
        let mut b = a.clone();
        b.angle = 40.0;
        b.stops.push(GradStop { pos: 0.4, color: [1.0, 0.0, 0.0, 1.0] });
        assert_eq!(a.topo_tag(), b.topo_tag());
        b.interp = GradInterp::Lch;
        assert_ne!(a.topo_tag(), b.topo_tag());
    }
}
