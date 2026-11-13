//! `Gfx`'s state, grouped by subsystem: one struct per file.

mod test;
pub(crate) use test::SuiteState;
mod brush;
pub(crate) use brush::BrushState;
mod geo;
pub(crate) use geo::GeoUiState;
mod spline;
pub(crate) use spline::SplineUiState;
