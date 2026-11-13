//! UV Warp. The second input is a UV map — red is U, green is
//! V, both normalised 0..1 over the SOURCE's format — and every output
//! pixel is the source sampled where its UV pixel points. A lens
//! distortion baked out of a solver, a refraction pass, a re-projection:
//! any warp that someone else computed as a picture.
//!
//! The standard convention exactly: V runs UP (the origin is the bottom-left
//! corner), and coordinates name pixel CENTRES at `(x + 0.5) / w`, so an
//! identity map built the usual way gives back the source pixel for pixel.
//! `v_up` off reads V top-down for maps made by tools that think that way.
//!
//! It fuses. The source subtree is re-evaluated at the looked-up position
//! (`Emit::input_at`) — the same thing a Transform does, with the position
//! read out of a picture instead of a matrix — so there is no pass, no
//! intermediate texture and no stage. Sampling is whatever the leaves do,
//! which is bilinear; `Filter::Nearest` snaps the lookup onto the source's
//! pixel centre, where bilinear IS the texel — a uniform, not a recompile.
//! Alpha comes from the source like every other channel; the map's own
//! alpha is not read.

use crate::codegen::Emit;
use crate::nodes::NodeDef;

const TOPO_BASE: u64 = 0x5717;

/// How the source is sampled at the looked-up position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum UvWarpFilter {
    /// Interpolated between the four nearest pixels.
    #[default]
    Bilinear,
    /// The one pixel the lookup lands in — for maps over ids or data,
    /// where a blend of two values is a value nobody wrote.
    Nearest,
}

/// A [`crate::graph::NodeKind::UvWarp`]'s controls. Both are uniforms.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct UvWarpParams {
    pub filter: UvWarpFilter,
    /// V runs up from the bottom edge (the standard). Off: down from the top.
    pub v_up: bool,
}

impl Default for UvWarpParams {
    fn default() -> Self {
        Self { filter: UvWarpFilter::Bilinear, v_up: true }
    }
}

impl UvWarpParams {
    /// Where (u, v) lands in a `size` source, in pixels (y down, centres at
    /// +0.5) — the CPU mirror of the generated code.
    pub fn lookup(&self, uv: [f32; 2], size: [f32; 2]) -> [f32; 2] {
        let v = if self.v_up { 1.0 - uv[1] } else { uv[1] };
        let q = [uv[0] * size[0], v * size[1]];
        match self.filter {
            UvWarpFilter::Bilinear => q,
            UvWarpFilter::Nearest => q.map(|c| c.floor() + 0.5),
        }
    }

    /// The span with the source's format in it: the codegen's central
    /// packer knows the per-node formats, `pack` does not.
    pub fn pack_sized(&self, size: [f32; 2], v: &mut [f32]) {
        self.pack(v);
        v[0] = size[0];
        v[1] = size[1];
    }
}

impl NodeDef for UvWarpParams {
    const NAME: &'static str = "UV Warp";
    /// The source, the UV map, then the mask.
    const INPUTS: usize = 3;
    const MASK_INPUT: Option<usize> = Some(2);
    /// (source w, source h, v up, nearest).
    const VEC4_SLOTS: usize = 1;

    /// Filter and orientation are uniforms: one shape of code, ever.
    fn topo_tag(&self) -> u64 {
        TOPO_BASE
    }

    /// The flags. The size lanes are written by [`Self::pack_sized`].
    fn pack(&self, v: &mut [f32]) {
        v[2] = f32::from(u8::from(self.v_up));
        v[3] = f32::from(u8::from(self.filter == UvWarpFilter::Nearest));
    }

    fn emit(&self, e: &mut Emit) -> String {
        if !e.has_input(0) {
            return "vec4f(0.0)".into();
        }
        // No map wired: nothing says where to look, so the source passes.
        if !e.has_input(1) {
            return e.input(0);
        }
        let slot = e.slot();
        let uv = e.input(1);
        let (vy, q, qs) = (e.fresh(), e.fresh(), e.fresh());
        e.line(format_args!("let {vy} = select({uv}.y, 1.0 - {uv}.y, P.v[{slot}].z > 0.5);"));
        e.line(format_args!("let {q} = vec2f({uv}.x, {vy}) * P.v[{slot}].xy;"));
        e.line(format_args!(
            "let {qs} = select({q}, floor({q}) + vec2f(0.5), P.v[{slot}].w > 0.5);"
        ));
        let v = e.input_at(0, &qs);
        if e.has_input(2) {
            let orig = e.input(0);
            e.masked(Self::MASK_INPUT, &orig, v)
        } else {
            v
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A map built the standard way — `(x + 0.5) / w`, V up — lands every
    /// pixel on its own centre.
    #[test]
    fn the_identity_map_lands_on_pixel_centres() {
        let (w, h) = (64.0, 36.0);
        for filter in [UvWarpFilter::Bilinear, UvWarpFilter::Nearest] {
            let p = UvWarpParams { filter, v_up: true };
            for (x, y) in [(0.0, 0.0), (13.0, 7.0), (63.0, 35.0)] {
                let uv = [(x + 0.5) / w, 1.0 - (y + 0.5) / h];
                let q = p.lookup(uv, [w, h]);
                assert!((q[0] - (x + 0.5)).abs() < 1e-4 && (q[1] - (y + 0.5)).abs() < 1e-4, "{q:?}");
            }
        }
    }

    #[test]
    fn v_down_reads_from_the_top_and_nearest_snaps() {
        let p = UvWarpParams { filter: UvWarpFilter::Bilinear, v_up: false };
        assert_eq!(p.lookup([0.0, 0.25], [8.0, 8.0]), [0.0, 2.0]);
        let n = UvWarpParams { filter: UvWarpFilter::Nearest, v_up: false };
        assert_eq!(n.lookup([0.26, 0.9], [10.0, 10.0]), [2.5, 9.5]);
    }

    /// Comps saved before the rename stored the node as `StMap`.
    #[cfg(feature = "serde")]
    #[test]
    fn an_old_stmap_loads_as_a_uv_warp() {
        use crate::graph::NodeKind;
        let p = UvWarpParams { filter: UvWarpFilter::Nearest, v_up: false };
        let new = serde_json::to_string(&NodeKind::UvWarp(p)).unwrap();
        assert!(new.contains("UvWarp"), "{new}");
        let old = new.replace("UvWarp", "StMap");
        let k: NodeKind = serde_json::from_str(&old).unwrap();
        assert_eq!(k, NodeKind::UvWarp(p));
    }
}
