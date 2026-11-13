//! The `--test-*` suites: one file per suite, each an `impl Gfx` block
//! holding that suite's driver. main.rs only dispatches to them.
mod bake;
mod r#box;
mod compose;
mod curveedit;
mod curves;
mod dual;
mod eyedrop;
mod frames;
mod inout;
mod layer;
mod paint;
mod range;
mod record;
mod shot;
mod slice;
mod subwob;
mod viewlock;
