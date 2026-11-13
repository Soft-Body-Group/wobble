# Modified copy

This is egui-winit 0.36.2 from <https://github.com/emilk/egui>
(`crates/egui-winit`), by Emil Ernerfeldt and the egui contributors, under
MIT OR Apache-2.0 (`LICENSE-MIT`, `LICENSE-APACHE`).

It has been modified for Wobble: the dropped-file handling in
`src/dropped_file.rs` is changed so the crate compiles for
`wasm32-unknown-unknown`. The root `Cargo.toml` patches the crates.io
egui-winit to this copy until upstream ships the fix.
