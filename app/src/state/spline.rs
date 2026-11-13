//! The Spline gizmo's state: the pick, the drags in flight and the temporary transform box.
//! These were loose fields on `Gfx`; grouped, not changed.

#[allow(unused_imports)] // a group of plain types needs nothing from the crate
use crate::*;

/// The Spline gizmo's state: the pick, the drags in flight and the temporary transform box.
pub(crate) struct SplineUiState {
    /// Spline control points picked out for a group edit, by index, and
    /// which spline they belong to. Indices, so they slide when points are
    /// added or removed (`anim::spline_point_*` does the same to the curves).
    /// A spline point drag, as it was at the press: the pointer position and
    /// the picked points' (index, position, feather). Gizmo drags compute
    /// ABSOLUTELY from this rather than summing per-frame deltas, because the
    /// working copy they edit is re-derived from the animation curve every
    /// frame — summed deltas would each be applied to the same base and only
    /// the last would survive.
    pub(crate) drag: Option<(egui::Pos2, Vec<(usize, [f32; 2], f32, f32)>)>,
    /// Each dragged point's tension when the drag began (K-drag).
    pub(crate) tension0: Vec<(usize, f32)>,
    /// The group box's pivot, moved off the box centre by a D-drag on its
    /// circle (the Transform gizmo's gesture): (node, the pick it was set
    /// for, NORMALIZED authored position). Rotate and alt-scale turn about
    /// it; a different pick or node drops it back to the centre.
    pub(crate) pivot: Option<(u64, Vec<usize>, [f32; 2])>,
    /// A ctrl-drag on a group box CORNER, as it was at the press: the
    /// pointer, the box's four corners (authored px, TL TR BR BL), which
    /// corner, and each picked point's (index, p, h_in, h_out). Absolute
    /// from the press, like `spline_drag`, so an animated shape keys cleanly.
    pub(crate) distort: Option<(egui::Pos2, [egui::Vec2; 4], usize, Vec<(usize, [f32; 2], [f32; 2], [f32; 2])>)>,
    /// Where the pivot was when the distort pressed (authored px).
    pub(crate) distort_pivot: egui::Vec2,
    /// The group box as the last edit left it: (node, pick, the quad in
    /// authored px TL TR BR BL, the picked points it was left around). A
    /// distort bends it and a rotate turns it; it is rebuilt from the
    /// bounds when the pick changes or the points move some other way.
    pub(crate) box_: Option<(u64, Vec<usize>, [egui::Vec2; 4], Vec<egui::Vec2>)>,
    /// A held-key edit of the point pick (E scale, R rotate, T translate),
    /// as it was at the press: node, which key, the pointer, the TEMPORARY
    /// pivot (authored px — wherever the press landed), and each picked
    /// point's (index, p, h_in, h_out). Absolute from the press.
    pub(crate) temp: Option<(u64, egui::Key, egui::Pos2, egui::Vec2, Vec<(usize, [f32; 2], [f32; 2], [f32; 2])>)>,
    /// The group box and pivot as the held-key edit found them, so the
    /// edit carries them too (taken on its first frame).
    pub(crate) temp_box: Option<([egui::Vec2; 4], Option<egui::Vec2>)>,
    /// R's turn so far: (the cursor's last angle round the pivot, the
    /// unwrapped total). Set once the cursor has left the pivot.
    pub(crate) temp_rot: Option<(f32, f32)>,
    pub(crate) sel: std::collections::BTreeSet<usize>,
    pub(crate) sel_node: Option<u64>,
    /// Point count the pick was made against; a change we did not make
    /// (the properties +/- buttons) invalidates every index in it.
    pub(crate) sel_n: usize,
    /// Which shape of a roto container the point pick belongs to.
    pub(crate) sel_shape: usize,
    /// The live band while a background drag is marquee-selecting points.
    pub(crate) marquee: Option<egui::Rect>,
}
