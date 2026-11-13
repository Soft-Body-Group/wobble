//! SoftCrop: a crop that throws nothing away.
//!
//! A hard crop is a statement about PIXELS — outside the box they are gone.
//! A soft crop is a statement about the LAYER: "this rectangle is what the
//! image is". The pixels pass through untouched (a soft crop costs exactly
//! nothing in the fused pass: its emit IS its input), and the rectangle is
//! read by the app instead — a Transform downstream sizes its handles to it,
//! so a plate with a letterbox, or a cutout in a big transparent canvas,
//! scales and rotates by the edges of what you actually see rather than by
//! its format's.
//!
//! The rectangle is NORMALISED to the node's own format, so it means the
//! same box at full resolution and under proxy. It is not a uniform (the
//! shader never reads it) and not topology, so dragging it neither
//! recompiles nor re-renders anything.
//!
//! The node is called **Crop** in the UI and has three modes
//! ([`CropMode`]; the variant and setup key stay `SoftCrop`):
//!
//! - **bounding box** — the soft crop above: every pixel passes.
//! - **crop** — the hard crop: the same box for the Transform below, but
//!   outside it the pixels ARE gone (transparent black), the format kept.
//!   The box is a uniform (one slot, in the node's own pixels).
//! - **reformat** — the stream BECOMES the box: the node's format is the
//!   box's size and its pixel (0, 0) is the box's corner (one slot: the
//!   corner, in the input's pixels, and the box's size). Outside its own
//!   format it is transparent black, like a crop: a Transform or a Layer
//!   reading past its edge must not find the input beyond the box. A Bake
//!   after it caches and plays back at that smaller format.
//!
//! The mode is topology; the box never is.
//!
//! A Crop can also **bake** itself ([`SoftCropParams::bake`]): the Bake
//! node's whole machinery, with no second node. The app lowers a baking
//! Crop into the live crop (at a transient id) under a `GraphBake` that
//! keeps the Crop's id, so the cache, its key, proxy handling and the
//! box-only storage are exactly the Bake's.

use crate::codegen::Emit;
use crate::nodes::NodeDef;

const TOPO_BASE: u64 = 0x5C;

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct SoftCropParams {
    /// Left, top, right, bottom, as fractions of the node's format. The
    /// full frame by default: a fresh soft crop changes nothing until it
    /// is pulled in.
    pub rect: [f32; 4],
    /// Which pipe it sits in. A soft crop takes whatever is wired into it —
    /// a picture, or the MATTE that will cut one out (a spline's box is the
    /// cut-out's box) — and is that on both sides, like a Bake or a Dot.
    /// The app keeps it true to its input (`EditorGraph::connect_out`); the
    /// pixels pass through either way.
    pub matte: bool,
    /// What the box does to the pixels. A bounding box is skipped when
    /// saved, so a soft crop's saved params and render keys are unchanged.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "CropMode::is_box"))]
    pub mode: CropMode,
    /// The hard-crop toggle setups saved before the modes: read, never
    /// written — `mode()` turns it into `CropMode::Crop`.
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    pub full: bool,
    /// The reader slot the Crop's own bake plays through, minted the first
    /// time "bake" is pressed (`None`: the Crop has never baked, and is the
    /// plain node it always was — nothing saved, nothing lowered). Not
    /// topology and not in the bake's key: which texture slot it plays
    /// through changes no pixel.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub bake: Option<usize>,
}

/// What a Crop node's box does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CropMode {
    /// A statement about the layer: every pixel passes.
    #[default]
    BoundingBox,
    /// Outside the box is gone; the format is kept.
    Crop,
    /// The stream becomes the box: its format is the box's size.
    Reformat,
}

impl CropMode {
    pub const ALL: [CropMode; 3] = [CropMode::BoundingBox, CropMode::Crop, CropMode::Reformat];

    pub fn label(self) -> &'static str {
        match self {
            CropMode::BoundingBox => "bounding box",
            CropMode::Crop => "crop",
            CropMode::Reformat => "reformat",
        }
    }

    pub fn is_box(&self) -> bool {
        *self == CropMode::BoundingBox
    }
}

impl Default for SoftCropParams {
    fn default() -> Self {
        Self { rect: [0.0, 0.0, 1.0, 1.0], matte: false, mode: CropMode::BoundingBox, full: false, bake: None }
    }
}

impl SoftCropParams {
    /// The box in pixels of a `w` x `h` format, ordered (a crop dragged
    /// inside out is still the same box) and never thinner than a pixel.
    pub fn px(&self, w: f32, h: f32) -> [f32; 4] {
        let [l, t, r, b] = self.rect;
        let (x0, x1) = (l.min(r) * w, l.max(r) * w);
        let (y0, y1) = (t.min(b) * h, t.max(b) * h);
        [x0, y0, x1.max(x0 + 1.0), y1.max(y0 + 1.0)]
    }

    /// The mode, with a pre-modes setup's `full` read as a crop.
    pub fn mode(&self) -> CropMode {
        match self.mode {
            CropMode::BoundingBox if self.full => CropMode::Crop,
            m => m,
        }
    }

    /// The box in whole pixels of a `w` x `h` format: corner and size, the
    /// size never zero. What a reformat's format is (`size`) and where its
    /// pixel (0, 0) sits in the input (`corner`).
    pub fn px_box(&self, w: u32, h: u32) -> ((u32, u32), (u32, u32)) {
        let [x0, y0, x1, y1] = self.px(w as f32, h as f32);
        let x0 = (x0.round().max(0.0) as u32).min(w.saturating_sub(1));
        let y0 = (y0.round().max(0.0) as u32).min(h.saturating_sub(1));
        let x1 = (x1.round() as u32).clamp(x0 + 1, w.max(x0 + 1));
        let y1 = (y1.round() as u32).clamp(y0 + 1, h.max(y0 + 1));
        ((x0, y0), (x1 - x0, y1 - y0))
    }

    /// Uniform slots: the box, only when it cuts or reformats.
    pub fn slots(&self) -> usize {
        !self.mode().is_box() as usize
    }

    /// The uniform span: a crop's box in the node's own (proxy) pixels,
    /// `fmt` being that format; a reformat's corner in its INPUT's pixels,
    /// `input` being the input's format, then its own size (`fmt`).
    pub fn pack_fmt_in(&self, v: &mut [f32], fmt: [f32; 2], input: [f32; 2]) {
        if v.len() < 4 {
            return;
        }
        match self.mode() {
            CropMode::BoundingBox => {}
            CropMode::Crop => v[..4].copy_from_slice(&self.px(fmt[0], fmt[1])),
            CropMode::Reformat => {
                let ((x0, y0), _) = self.px_box(input[0].round() as u32, input[1].round() as u32);
                v[..4].copy_from_slice(&[x0 as f32, y0 as f32, fmt[0], fmt[1]]);
            }
        }
    }

    /// `pack_fmt_in` for a node whose format is its input's (every mode but
    /// reformat).
    pub fn pack_fmt(&self, v: &mut [f32], fmt: [f32; 2]) {
        self.pack_fmt_in(v, fmt, fmt);
    }

    /// Whether pixel centre `p` survives a full crop of a `fmt` format —
    /// the CPU mirror of the generated test.
    pub fn keeps(&self, p: [f32; 2], fmt: [f32; 2]) -> bool {
        let [x0, y0, x1, y1] = self.px(fmt[0], fmt[1]);
        self.mode() != CropMode::Crop || (p[0] >= x0 && p[0] < x1 && p[1] >= y0 && p[1] < y1)
    }
}

impl NodeDef for SoftCropParams {
    const NAME: &'static str = "Crop";
    const INPUTS: usize = 1;
    const MASK_INPUT: Option<usize> = None;
    /// A soft crop sends nothing to the shader (the rectangle is the app's
    /// to read); a full crop sends its box — see [`SoftCropParams::slots`].
    const VEC4_SLOTS: usize = 0;

    fn topo_tag(&self) -> u64 {
        // Crop is 1, the tag the old `full` toggle had: nothing cached moves.
        let m = match self.mode() {
            CropMode::BoundingBox => 0u64,
            CropMode::Crop => 1,
            CropMode::Reformat => 2,
        };
        TOPO_BASE ^ (m << 8)
    }

    fn pack(&self, v: &mut [f32]) {
        self.pack_fmt(v, [1.0, 1.0]);
    }

    fn emit(&self, e: &mut Emit) -> String {
        let mode = self.mode();
        if mode == CropMode::Reformat {
            // The box is the whole stream: this node's pixel p is the
            // input's pixel p + the box's corner, and nothing outside it.
            let (slot, pos) = (e.slot(), e.pos().to_string());
            let at = format!("({pos} + P.v[{slot}u].xy)");
            let input = e.input_at(0, &at);
            let v = e.fresh();
            e.line(format_args!(
                "let {v} = select(vec4f(0.0), {input}, all({pos} >= vec2f(0.0)) && all({pos} < P.v[{slot}u].zw));"
            ));
            return v;
        }
        // A bounding box: the pixels are the input's, every one of them.
        let input = e.input(0);
        if mode == CropMode::BoundingBox {
            return input;
        }
        let (slot, pos, v) = (e.slot(), e.pos().to_string(), e.fresh());
        e.line(format_args!(
            "let {v} = select(vec4f(0.0), {input}, all({pos} >= P.v[{slot}u].xy) && all({pos} < P.v[{slot}u].zw));"
        ));
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Moving the box is neither a recompile nor a uniform: a soft crop
    /// has nothing to say to the shader.
    #[test]
    fn the_box_is_not_topology() {
        let a = SoftCropParams::default();
        let b = SoftCropParams { rect: [0.1, 0.2, 0.7, 0.9], ..Default::default() };
        assert_eq!(a.topo_tag(), b.topo_tag());
    }

    /// The toggle is topology (it changes the code); the box under a full
    /// crop is still only a uniform.
    #[test]
    fn full_is_topology_and_its_box_is_a_uniform() {
        let soft = SoftCropParams::default();
        let full = SoftCropParams { mode: CropMode::Crop, ..soft };
        assert_ne!(soft.topo_tag(), full.topo_tag());
        let moved = SoftCropParams { rect: [0.1, 0.1, 0.5, 0.5], ..full };
        assert_eq!(full.topo_tag(), moved.topo_tag());
        assert_eq!((soft.slots(), full.slots()), (0, 1));
        assert!(moved.keeps([20.5, 20.5], [100.0, 100.0]));
        assert!(!moved.keeps([60.5, 20.5], [100.0, 100.0]));
        assert!(SoftCropParams { mode: CropMode::BoundingBox, ..moved }.keeps([60.5, 20.5], [100.0, 100.0]));
    }

    /// A setup saved with the old toggle loads as a crop, keeps the old
    /// toggle's tag, and saves as a mode.
    #[cfg(feature = "serde")]
    #[test]
    fn the_old_full_toggle_loads_as_crop() {
        let old: SoftCropParams =
            serde_json::from_str(r#"{"rect":[0.1,0.1,0.5,0.5],"matte":false,"full":true}"#).unwrap();
        assert_eq!(old.mode(), CropMode::Crop);
        assert_eq!(old.topo_tag(), TOPO_BASE ^ (1 << 8));
        let soft: SoftCropParams = serde_json::from_str(r#"{"rect":[0,0,1,1],"matte":false}"#).unwrap();
        assert_eq!(soft.mode(), CropMode::BoundingBox);
        assert!(!serde_json::to_string(&soft).unwrap().contains("mode"));
        assert!(!serde_json::to_string(&soft).unwrap().contains("bake"));
        let baking = SoftCropParams { bake: Some(4), ..soft };
        assert_eq!(baking.topo_tag(), soft.topo_tag());
        let back: SoftCropParams = serde_json::from_str(&serde_json::to_string(&baking).unwrap()).unwrap();
        assert_eq!(back.bake, Some(4));
        let re = SoftCropParams { mode: CropMode::Reformat, ..soft };
        assert_ne!(re.topo_tag(), old.topo_tag());
        assert_eq!(re.slots(), 1);
    }

    #[test]
    fn a_box_in_whole_pixels() {
        let p = SoftCropParams { rect: [0.25, 0.25, 0.75, 0.6], ..Default::default() };
        assert_eq!(p.px_box(512, 288), ((128, 72), (256, 101)));
    }

    #[test]
    fn px_orders_the_box_and_keeps_it_a_pixel_wide() {
        let p = SoftCropParams { rect: [0.75, 0.5, 0.25, 0.5], ..Default::default() };
        assert_eq!(p.px(100.0, 40.0), [25.0, 20.0, 75.0, 21.0]);
    }
}
