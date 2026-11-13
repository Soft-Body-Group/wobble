# Contributing to Wobble

Outside contribution is the point of this project. This page is the short
version: where to start, what to run, and which part of the tree to stay
out of until it has been cleaned up.

## Easy entry points

**A plugin node (no Rust, no rebuild).** A folder in `plugins/` holding a
`plugin.json` and a WGSL expression is a node. It fuses into the main
shader pass like a built-in, and its params animate and patch.
[plugins/invert](plugins/invert/plugin.json) is the whole of one; copy it,
change `type_id`, `label` and `expr`, and restart the app.
[plugins/vignette](plugins/vignette) shows more params.

**A built-in node.** Two files with the same name, one per crate, plus
one-line arms where `NodeKind` is dispatched:

- `engine/src/nodes/<node>.rs` holds the params, the WGSL and `impl NodeDef`.
- `app/src/nodes/<node>.rs` holds the channels, the tiles and the panel.

Copy `saturation` for a pointwise node or `blur` for a filter. The full
recipe, step by step, is in
[docs/claude/recipes.md](docs/claude/recipes.md#adding-a-node).

**An addon** (Python talking to the scripting API):
[plugins/hello_addon](plugins/hello_addon).

**A test suite.** Each `--test-*` suite is one file in
[app/src/suites](app/src/suites). The rules for writing one are in
[docs/claude/testing.md](docs/claude/testing.md).

## Known debt: `app/src/lib.rs`

`app/src/lib.rs` is still tens of thousands of lines: the window shell,
the frame loop, what is left of the viewer and properties passes, and one
very large `Gfx` struct. It is known debt and it is being split up. Out
already: the suites (`app/src/suites`), the node panels (`app/src/nodes`),
the viewer's gizmo passes (`app/src/viewer`), the document model
(`app/src/doc`) and about half of `Gfx`'s fields (`app/src/state`). Still
to come: the document model as its own egui-free crate, and the rest of
`Gfx`. You do not need to understand `lib.rs` to add a node or a plugin,
and a change that adds to it will be asked to live in its own file
instead. If your change needs to touch it, keep the footprint to a `mod`
line and a call.

## Before you send a change

```sh
cargo build --release      # zero warnings is the standard
cargo test                 # engine + app unit tests
```

Test media is generated, not committed (see the README), and the UI suites
run one at a time:

```sh
./target/release/wobble data/2k_zip_bg data/2k_zip_bg --test-click
```

Each prints one `PASS` or `FAIL` line and exits. Run the ones near what
you touched.

## What decides arguments

Performance first, work on the GPU, slow paths allowed but never hidden,
and Blender's conventions for the hands. The longer list, and the
invariants that have cost real time when broken, are in
[CLAUDE.md](CLAUDE.md) and [docs/claude](docs/claude). The architecture
reference is [docs/reference.md](docs/reference.md).

## Licence

Wobble is GPL-3.0-or-later with an additional permission for plugins and
addons; see [LICENSING.md](LICENSING.md). Contributions are accepted under
the same terms.
