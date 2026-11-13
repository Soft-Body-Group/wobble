//! The `--test-*` suites: one file per suite, each an `impl Gfx` block
//! holding that suite's driver. main.rs only dispatches to them.
mod slice;
