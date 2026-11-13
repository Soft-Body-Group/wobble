//! `Gfx`'s state, grouped by subsystem: one struct per file.

mod test;
pub(crate) use test::SuiteState;
mod brush;
pub(crate) use brush::BrushState;
mod geo;
pub(crate) use geo::GeoUiState;
mod spline;
pub(crate) use spline::SplineUiState;
mod pin;
pub(crate) use pin::PinState;
mod prop;
pub(crate) use prop::PropState;
mod patch;
pub(crate) use patch::PatchState;
mod layer;
pub(crate) use layer::LayerUiState;
mod hud;
pub(crate) use hud::HudState;
mod cache;
pub(crate) use cache::CacheUiState;
mod curve;
pub(crate) use curve::CurveEditState;
mod help;
pub(crate) use help::HelpState;
