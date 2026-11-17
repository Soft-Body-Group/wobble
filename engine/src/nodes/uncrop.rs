//! Uncrop: put a cropped picture back where it came from.
//!
//! The round trip it closes: a Crop in **reformat** mode cuts a region out
//! (its format becomes the box), that region goes off somewhere — a render
//! farm, another app, a model that only takes small frames — and comes back
//! as a new plate. The Uncrop sets it back into the full format at the box,
//! with the area the crop threw away transparent black.
//!
//! Two inputs: 0 is the cropped picture, 1 is a REFERENCE whose only job is
//! to say what the full format is (the plate the Crop cut from, usually).
//! The reference is never sampled (`NodeKind::samples_input`), so wiring it
//! decodes nothing. The box is normalised to the full format, like the
//! Crop's, and is normally the Crop's own: the app links an Uncrop to a Crop
//! node ([`UncropParams::crop`]) and copies that Crop's box in when it
//! lowers the graph, so moving the crop moves the uncrop with it. With no
//! reference wired the full format is the linked Crop's input's.
//!
//! If the picture that comes back is not the box's size (rendered at twice
//! the resolution, say) it is scaled to fill the box, so a 1:1 return is
//! exactly one pixel for one pixel and anything else still lands in the
//! right place. It fuses: one `input_at`, no pass, no texture. The box is a
//! uniform; nothing about it is topology.

use crate::codegen::Emit;
use crate::nodes::NodeDef;

/// The port that names the full format.
pub const REF_INPUT: usize = 1;

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct UncropParams {
    /// Left, top, right, bottom, as fractions of the FULL format — the
    /// Crop's own `rect`. Used as authored when no Crop is linked; when one
    /// is, the app overwrites it with that Crop's box on every lowering.
    pub rect: [f32; 4],
    /// The Crop node this undoes (an editor node id). App data: the engine
    /// never reads it.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub crop: Option<u64>,
}

impl Default for UncropParams {
    fn default() -> Self {
        Self { rect: [0.0, 0.0, 1.0, 1.0], crop: None }
    }
}

impl UncropParams {
    /// The box in whole pixels of the full `w` x `h` format: corner and
    /// size, exactly as a reformatting Crop of that format places it.
    pub fn px_box(&self, w: u32, h: u32) -> ((u32, u32), (u32, u32)) {
        crate::nodes::softcrop::SoftCropParams { rect: self.rect, ..Default::default() }.px_box(w, h)
    }

    /// Where full-format pixel `p` reads the cropped picture, and whether
    /// it is inside the box at all — the CPU mirror of the generated code.
    /// `full` is the node's own format, `input` the cropped picture's.
    pub fn lookup(&self, p: [f32; 2], full: [f32; 2], input: [f32; 2]) -> Option<[f32; 2]> {
        let mut v = [0.0; 8];
        self.pack_fmt(&mut v, full, input);
        let inside = p[0] >= v[0] && p[1] >= v[1] && p[0] < v[0] + v[2] && p[1] < v[1] + v[3];
        inside.then(|| [(p[0] - v[0]) * v[4], (p[1] - v[1]) * v[5]])
    }

    /// (corner x, corner y, box w, box h) in the node's own (full, proxy
    /// included) pixels, then (input w / box w, input h / box h, 0, 0).
    pub fn pack_fmt(&self, v: &mut [f32], full: [f32; 2], input: [f32; 2]) {
        if v.len() < 8 {
            return;
        }
        let ((x0, y0), (bw, bh)) = self.px_box(full[0].round() as u32, full[1].round() as u32);
        let (bw, bh) = (bw as f32, bh as f32);
        v[..8].copy_from_slice(&[x0 as f32, y0 as f32, bw, bh, input[0] / bw, input[1] / bh, 0.0, 0.0]);
    }
}

impl NodeDef for UncropParams {
    const NAME: &'static str = "Uncrop";
    const INPUTS: usize = 2;
    const MASK_INPUT: Option<usize> = None;
    const VEC4_SLOTS: usize = 2;

    fn topo_tag(&self) -> u64 {
        0x7043
    }

    fn pack(&self, v: &mut [f32]) {
        self.pack_fmt(v, [1.0, 1.0], [1.0, 1.0]);
    }

    fn emit(&self, e: &mut Emit) -> String {
        if !e.has_input(0) {
            return "vec4f(0.0)".into();
        }
        let (slot, pos) = (e.slot(), e.pos().to_string());
        let q = e.fresh();
        e.line(format_args!("let {q} = ({pos} - P.v[{slot}u].xy) * P.v[{slot}u + 1u].xy;"));
        let input = e.input_at(0, &q);
        let v = e.fresh();
        e.line(format_args!(
            "let {v} = select(vec4f(0.0), {input}, all({pos} >= P.v[{slot}u].xy) && \
             all({pos} < P.v[{slot}u].xy + P.v[{slot}u].zw));"
        ));
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nodes::softcrop::{CropMode, SoftCropParams};

    /// Crop then uncrop is the identity inside the box: every pixel of a
    /// reformatting Crop's output lands back on the pixel it came from.
    #[test]
    fn uncrop_undoes_a_reformat_crop() {
        let rect = [0.25, 0.25, 0.75, 0.6];
        let crop = SoftCropParams { rect, mode: CropMode::Reformat, ..Default::default() };
        let un = UncropParams { rect, crop: None };
        let full = (512u32, 288u32);
        let ((cx, cy), (bw, bh)) = crop.px_box(full.0, full.1);
        let fullf = [full.0 as f32, full.1 as f32];
        let inf = [bw as f32, bh as f32];
        for (x, y) in [(cx, cy), (cx + bw - 1, cy + bh - 1), (cx + 7, cy + 3)] {
            let p = [x as f32 + 0.5, y as f32 + 0.5];
            // The crop's pixel q is the input's q + corner; the uncrop sends
            // p back to q.
            let q = un.lookup(p, fullf, inf).expect("inside the box");
            assert_eq!([q[0] + cx as f32, q[1] + cy as f32], p);
        }
        assert!(un.lookup([cx as f32 - 0.5, cy as f32 + 0.5], fullf, inf).is_none());
        assert!(un.lookup([(cx + bw) as f32 + 0.5, cy as f32 + 0.5], fullf, inf).is_none());
    }

    /// A return at twice the box's size still fills exactly the box.
    #[test]
    fn a_bigger_return_is_scaled_into_the_box() {
        let un = UncropParams { rect: [0.5, 0.0, 1.0, 0.5], crop: None };
        let q = un.lookup([75.0, 25.0], [100.0, 100.0], [100.0, 100.0]).unwrap();
        assert_eq!(q, [50.0, 50.0]);
    }

    #[test]
    fn the_box_is_not_topology() {
        let a = UncropParams::default();
        let b = UncropParams { rect: [0.1, 0.2, 0.7, 0.9], crop: Some(3) };
        assert_eq!(a.topo_tag(), b.topo_tag());
    }
}
