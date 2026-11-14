//! The cache settings: the command-line overrides, the drive benchmark and what the meter last showed.
//! These were loose fields on `Gfx`; grouped, not changed.

#[allow(unused_imports)] // a group of plain types needs nothing from the crate
use crate::*;

/// The cache settings: the command-line overrides, the drive benchmark and what the meter last showed.
pub(crate) struct CacheUiState {
    /// First launch, after the welcome + consent windows: pick the SSD
    /// cache (`settingsui::cache_setup_window`); the quick start waits on it.
    pub(crate) setup_open: bool,
    pub(crate) bytes: usize,
    /// True when --cache-dir overrode prefs (UI edits then don't persist).
    pub(crate) dir_cli: bool,
    /// Why the NVMe tier is off, when a dir IS set but unusable (an
    /// unwritable volume — a Time Machine backup drive, say — fails with a
    /// bare EPERM that otherwise only reaches stderr).
    pub(crate) dir_err: Option<String>,
    /// --cache-gb was given: the UI edits prefs but this session's budget
    /// stays the flag's.
    pub(crate) gb_cli: bool,
    /// NVMe benchmark in flight / last result.
    pub(crate) bench_rx: Option<std::sync::mpsc::Receiver<anyhow::Result<engine::diskcache::Bench>>>,
    pub(crate) bench: Option<engine::diskcache::Bench>,
    /// Why the last benchmark failed — shown in the window, not only logged.
    pub(crate) bench_err: Option<String>,
}
