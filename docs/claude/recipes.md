# Recipes: adding nodes, models, weights

<!-- Moved out of CLAUDE.md to keep it under Claude Code's size limit. Same rules, same weight: read the section for any subsystem you touch. -->

### Adding a downloadable model

One row in `CATALOG` in [engine/src/models.rs](engine/src/models.rs) — id,
url, exact byte size, SHA-256, licence, source. Nothing else in the app may
hold a URL. Fetch it once with `target/release/models get <id>`, which prints
the hash to paste back into the row; until a hash is pinned, `State::Ready`
only means "the right number of bytes" and the UI says so.

The size in the row is checked against the server's `Content-Length` before
a byte is written, so a moved or replaced upstream file fails loudly instead
of caching something else under the same name. `engine/tests/models.rs`
drives the whole fetcher against a throwaway localhost server — resume,
corrupt-prefix rejection, cancel — with no network.

### Adding a node that needs weights

**A node declares its weights; it never handles downloading.** The whole
convention, in order:

1. **Catalog the weights** — one `CATALOG` row per file (above), with the
   **sha256 pinned**. `cost::every_declared_model_is_catalogued_and_pinned`
   fails on a declared id that is missing or unpinned. Host weights at a
   stable direct URL (upstream, or a mirror this project controls — a
   Google Drive folder does not qualify). New weights = a new id; never
   change what an existing id's bytes are, or old comps silently render
   different pixels.
2. **Declare them** — one arm in `NodeKind::models()` (`graph.rs`)
   returning the ids, e.g. `NodeKind::Tapir(p) => p.variant.models()`. It
   takes `&self`, so a node offering small vs accurate weights returns the
   picked one. That arm alone gets the node:
   - the **download gate** at the top of its properties panel
     (`modelui::node_gate`, drawn by `properties_ui` for any node that
     declares models — do NOT draw a model row in the node's own panel);
   - the **missing-weights badge** (warn-coloured down-arrow, lower left) in
     the graph;
   - a line in the **"missing weights" prompt** when a setup opens without
     them (`modelui::needed_by` → `setup_prompt`);
   - its **licence tag** (`cost::licence_of`, rule 10).
3. **Load through `models::ready_path(id)`** and nothing else. It returns the
   cache path only once the file is catalogued and the right size, and its
   error is written for a person — show it as the node's status. Never build
   a path from `models_dir()` yourself, never take a path param.
4. **Missing weights = passthrough, never an error.** Until `ready_path`
   answers `Ok`, the node passes its input through (or emits nothing, for a
   generator) and does no work. The gate says why; the node must not also
   pop dialogs, log per frame, or start a fetch. **Nothing downloads without
   a click** — `modelui` tests assert the gate and prompt never do.
5. **Inference is an explicit action with a result on disk** (the bake /
   Tracker / planar precedent) when it is expensive, so `cost.rs` can colour
   it and a bake can clear the slow taint. Non-commercial weights stay
   tagged after a bake.
6. **Readers stay safe.** Checkpoints go through `npy.rs` (restricted pickle)
   or a format with no code in it (safetensors, ONNX, a Wobble container).
   Never add a general unpickler, and never let a setup file carry a URL —
   a comp names catalog ids, the app decides where they come from.

### Reading a checkpoint

[engine/src/npy.rs](engine/src/npy.rs) parses JAX/numpy `.npy` checkpoints
(a pickled tree of arrays) into `name -> Tensor`, keys joined with `/`.
`models inspect <id> [substring]` is the way to look at one. The BootsTAPIR
causal checkpoint parses to 218 tensors / 54.7M parameters in ~0.3s.

**It is not a general unpickler and must never become one.** Pickle is an
RCE format; what runs is a stack machine over the opcode subset numpy emits,
allowing exactly `numpy.core.multiarray._reconstruct` and `numpy.dtype`, and
refusing every other `GLOBAL` by name. There is a test that a pickle naming
`os.system` is rejected. If a checkpoint needs a wider subset, widen it
deliberately and keep that boundary.

### Adding a node

The add menus are built from ONE table — `editor::node_catalog()` — which the
right-click menu groups into submenus by `NodeCat` and the Tab search
flattens. Add your entry there and both menus get it; there is no second list
to keep in step. `editor::catalog_tests` checks the invariant that actually
bites: `reader: true` must agree with `NodeKind::reader_slot()`, because a
reader-backed node added without it hands two nodes the same slot and they
fight over one texture — which looks like a decode bug, not a menu bug.

A built-in node is TWO FILES with the same name — one per crate — plus
one-line arms in the matches that dispatch on `NodeKind`. Copy `saturation`
on both sides for a pointwise node, `blur` for a filter; they are the
templates:

1. `engine/src/nodes/<node>.rs` — the params (re-exported through
   `graph.rs`, so `graph::XParams` stays the path everyone uses), the WGSL,
   and `impl NodeDef`: `NAME`, `INPUTS`, `MASK_INPUT`, `VEC4_SLOTS`,
   `topo_tag` (STRUCTURAL params only — a uniform that reaches it makes
   every slider drag a recompile), `pack` (the uniform span, in the order
   the WGSL reads it) and `emit` (over `codegen::Emit`: inputs, the slot,
   the helper maps, `masked`). `pub mod` it in `nodes/mod.rs`.
   **A filter — anything that samples its neighbourhood — also
   `impl StageDef`** (`engine/src/nodes/stage.rs`): its scratch textures,
   its passes in order (each a complete WGSL compute module, its bindings
   in `@binding` order, its dispatch shape, its uniform size), which
   texture holds the result, the uniform floats per pass per frame, and
   its `Speed`. The compositor allocates, binds and dispatches from that
   description; the node never sees wgpu. `program()` may depend on the
   canvas and on structural params only — a uniform that changed the
   chain would make a slider drag rebuild textures. `emit` is then
   `e.stage_result()` (+ `masked_stage`), and the node's WGSL modules go
   in `nodes::builtin_pass_wgsl()` so they compile at startup. One arm
   in `NodeKind::stage_def()` is the whole of the central wiring: codegen
   (`kind_of`, aux ports, `loop_ok`), the cost colouring and the
   compositor all read it. Glow is the template for a chain whose length
   depends on the canvas.
2. `graph.rs` — the variant, and one-line arms in `input_count`,
   `mask_input`, `name` and `topo_tag` (`output_kind`, `reader_slot` and
   `placement` only when it is not a plain image node).
3. `codegen.rs` — one-line arms in `vec4_slots_for`, `emit` and
   `pack_params`. WGSL helpers go in the node file, not the prelude.
4. `app/src/nodes/<node>.rs` — `CHANNELS`, `TILES`, `get` / `set` /
   `label`, `panel`, `glyph`. `pub mod` it and add it to
   `nodes::channels()`, which is how `anim` learns the channels exist.
5. `anim.rs` — the `Channel` variants (they stay in the one flat enum: the
   variant NAME is what a patch cable and a setup file store) and one-line
   arms in `get` and `set`. `curves.rs` — the variants join the
   `unreachable!` group of the label match; the label comes from the node.
6. `editor.rs` — `type_color` (+ a `theme.rs` slot), the glyph arm in
   `paint_body`, the `node_catalog()` entry. The colour designer keeps a
   COPY of every glyph: add the node to `colors/src/glyphs.rs` (`Icon`,
   `NODES`, the arm in `body` — copy the shape code verbatim).
   `glyph_parity` fails, naming the node, until the copy matches.
   `lib.rs` — the `properties_ui` arm (four lines: call `panel`, keep its
   HUD slider); `node_resolution` only if the node declares its own
   format (a generator or a Reformat).
   `widgets.rs::tiles_for` — one arm; `panel_tabs` only if the panel is
   several pages.
7. `editor::status_of` — nothing to do for a NEW node: it is alpha by
   default and the catalog reads its status from there. Promoting one to
   beta or production is an edit to that match AND to
   `catalog_tests::every_entry_has_the_status_its_kind_declares_and_the_policy_holds`,
   which pins the policy so a graduation is never accidental.

Nodes that already live in `nodes/`: LUT, Text (a per-glyph layer, see **Text node**) and Burnin (a per-node
distance-field glyph atlas from `engine::textraster`, `SourceBinding::Text`;
size, place, colour and the frame are uniforms, only the words
re-rasterise), Flow, Lens Distortion, Stabilize, VectorBlur, VectorWarp, Saturation, Contrast (fused; its optional local-contrast pyramid is a stage only while `local` is on), ScreenKey, CorridorKey (a process-backed reader, see **CorridorKey** below), Sammie (the same, see **Sammie**), Depth (the same, see **Depth node**), Blur, Dilate, Glow, Glint, Diffuse, EdgeDetect, Median, HighlightRebuild, Convolve,
SoftCrop, UV Warp, ToData, ToRgba, ColorCompress, MotionBlur, ColorSlice (shown as the Tinter), HueRotate, Gamma, Expression, Despill, Lightwrap, Gradient, Grain. Every other node still carries its body
inside the central files (Defocus, Loop and Shader build their passes by
hand in `compositor.rs` — the next stages to move).
Move one when you touch it — same shape, and `cargo run -p engine --example
wgsl_dump` (edit the graph it builds) diffed against `main` is how you know
the move changed nothing.
