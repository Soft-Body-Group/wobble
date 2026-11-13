//! The Curve node's canvas, as the properties pass last drew it.
//! These were loose fields on `Gfx`; grouped, not changed.

#[allow(unused_imports)] // a group of plain types needs nothing from the crate
use crate::*;

/// The Curve node's canvas, as the properties pass last drew it.
pub(crate) struct CurveEditState {
    /// --test-curveedit: the curve canvas rect, and the shape before a drag.
    pub(crate) rect: Option<egui::Rect>,
    /// Screen position of each curve point, as the widget drew them.
    pub(crate) screen: Vec<egui::Pos2>,
    pub(crate) before: Vec<[f32; 2]>,
    /// Where the driver grabbed a point, so the move and release match it.
    pub(crate) grab: egui::Pos2,
}
