//! The `--test-*` suites: one file per suite, each an `impl Gfx` block
//! holding that suite's driver. main.rs only dispatches to them.
mod compose;
mod curveedit;
mod dual;
mod eyedrop;
mod frames;
mod inout;
mod layer;
mod record;
mod shot;
mod slice;
mod subwob;
mod viewlock;
