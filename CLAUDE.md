# Wobble — working notes for Claude

GPU-resident 2D compositor: Rust + wgpu, fused WGSL generated from a node
graph. `docs/reference.md` (the long-form reference; "README" below means
it) describes what it does and how the pipeline works — read it for
architecture. This file is the operating manual: conventions, hard
invariants, and the traps that have actually cost time.

**Detail lives in `docs/claude/`** (moved there to keep this file under
Claude Code's size limit; the rules are not optional because they moved):

- [docs/claude/invariants.md](docs/claude/invariants.md) — the full invariants; headlines below.
- [docs/claude/traps.md](docs/claude/traps.md) — the full traps; headlines below.
- [docs/claude/subsystems.md](docs/claude/subsystems.md) — what must stay true per subsystem/node. **Read the section before touching that subsystem.**
- [docs/claude/testing.md](docs/claude/testing.md) — what each `--test-*` suite covers, adding a suite (seven registrations), rules for suites, CI, web build.
- [docs/claude/recipes.md](docs/claude/recipes.md) — adding a node, a downloadable model, a node that needs weights, reading a checkpoint.

## Manifesto

The principles that decide arguments. When a change trades one against
another, these are the tiebreakers.

1. **Performance above all else.** It is the thing being proven. A feature
   that costs frame rate needs a very good reason, and usually a different
   design.
2. **GPU above all else.** Work belongs on the GPU; the CPU schedules and
   feeds it. Readbacks, per-frame CPU passes and round trips are suspect by
   default. Data stays resident.
3. **Target workstation hardware — 4090-tier GPU, plenty of NVMe — but
   never fail outright on less.** Scale down (smaller caches, fewer frames
   in flight, lower resolution) rather than refuse to run. macOS is the
   development machine, not the target.
4. **This will be open source, and outside contribution is the point.**
   Internal libraries — the UI above all — stay modular, so someone can use
   or replace a piece without taking the whole app.
5. **Nodes must be authorable outside the main build.** *Tier 0 is real*:
   `plugins/<name>/plugin.json` + a WGSL expression = a node (see
   `plugins/invert`), fused like any built-in, params animatable/patchable,
   uninstalled types render as a passthrough with data preserved. Built-ins
   are still enum arms; migrate toward `NodeKind::Custom` + registry, and do
   not entrench the enum further.
6. **Plugins go beyond nodes.** *Also aspiration.* A plugin should be able
   to ship custom UI and behaviour a node cannot express — panels, viewer
   overlays and gizmos, browser integrations, expression functions, patch
   value nodes. Registries over enums everywhere a plugin would plausibly
   want to add a case (`PaneKind`, `ModKind`, shortcuts, panel builders).
7. **We serve the high-end compositing use case, within our scope.** We would rather omit a
   feature than ship one that makes us as slow as the incumbents. Scope is a
   performance decision.
8. **Slow paths are welcome when they are worth it — and they are always
   VISIBLE.** The rule was once "no slow paths"; that was too blunt. A
   defocus with a real bokeh, a diffraction convolve, a user shader with four
   passes: these cannot fuse, and refusing to ship them would only mean
   people do the work somewhere worse. What is forbidden is a slow path
   nobody can SEE. So:
   - **Fast is cyan, slow is red, in the graph itself.** A node that takes
     its own pass wears a red ring, and every noodle downstream of it runs
     red until something resolves it. You can read the cost of a comp off
     the wires without opening a panel.
   - **A Bake clears the red.** That is the answer the colour is steering
     you towards: put the expensive branch behind a `GraphBake` and
     everything downstream is cyan again, because it really is — the pixels
     now come off disk.
   - The panel still says what the node costs, in words, at the point of use.

   The invariant is not "be fast", it is **"never be slow silently"**.

9. **UX is a first-class concern**, explicitly in contrast to the incumbents' PyQt
   front ends: direct manipulation, live feedback, gestures that respect the
   user's hands. A compositor people enjoy driving.

10. **Non-commercially-licensed models are supported, tagged and warned
    about — never refused.** Some of the best tools in this field ship under
    research-only licences, and a compositor that pretends they do not exist
    is less useful, not more principled. The policy is:
    - **We ship them.** If a non-commercial model is the right tool, it gets
      a node like anything else.
    - **The node is tagged**, and the licence is shown before a single byte
      is downloaded (`models.rs` already does this).
    - **Anything downstream of one is flagged**, and a Write is warned before
      it renders, because that is the moment output leaves the building.
    - **A Bake does NOT clear that flag.** Baking changes what the pixels
      cost, not what they are: a frame rendered through a research-only model
      is still a frame rendered through a research-only model. Speed taint
      and licence taint propagate the same way and clear differently, and
      conflating them would be the one bug in this feature that actually
      hurts somebody.
11. **We are a Blender companion app.** When a UI decision has a Blender
    convention, we take it: muscle memory must carry across the two apps in
    both directions. Navigation is the loudest case — pan/zoom modifiers
    everywhere, and Blender's tumble the day a 3D view exists — but the
    rule covers any gesture, shortcut or layout where a Blender habit
    exists and does not collide with rules 7 and 9. When compositing convention and Blender
    disagree, Blender wins the hands, compositing wins the scope.

## Ground rules

- **Commit your own finished work locally; never `git push` unless asked
  in that turn.** Commit only the files you changed (other lanes leave
  uncommitted work in the tree — never sweep it into your commit).
- **A second Claude may be working this repo.** Claim your lane and files in
  [.agents/board.md](.agents/board.md), leave notes in
  [.agents/messages.md](.agents/messages.md), and **re-read a file before
  editing it** — the other lane commits too, so your view can be stale.
- Commit messages: concise, no AI-authorship trailer.

## Build, run, test

**`data/` is gitignored — a fresh clone has no test media.** Generate it
before running anything that needs plates:

```bash
cargo build --release
target/release/gen_exr data/2k_zip_bg 2048 1080 24 zip bg
target/release/gen_exr data/2k_zip_fg 2048 1080 24 zip fg
target/release/gen_exr data/2k_passes 512 288 8 zip passes   # multi-layer
```

```bash
cargo build --release                  # zero warnings is the standard
cargo test                             # engine + app unit tests
./target/release/wobble data/2k_zip_bg data/2k_zip_bg   # interactive
cargo build --release -p app --features app/opendrt      # GPLv3 when enabled;
                                        # a bare --features opendrt at the
                                        # workspace root silently does nothing
cargo build --release -p app --features app/movie       # ProRes/H.264 via a
                                        # SYSTEM ffmpeg (brew install ffmpeg).
                                        # Off by default so a stock build and
                                        # CI need no non-Rust dependency.
cargo test -p engine --features movie                   # incl. the encode ->
                                        # decode round trip in tests/movie.rs
```

**imfram** (the shared egui crate) is a git dependency pinned by `rev` in
`app/Cargo.toml`, so a plain clone builds with nothing else checked out.
It is public (`Soft-Body-Group/imfram`), fetched over https. On the maintainer's Mac, a
`.cargo/config.toml` one level up `[patch]`es it to the sibling checkout
`../imfram`, so imfram edits show up without a push — and the
committed `Cargo.lock` is the patched one (imfram has no `source` line;
a clone without the patch adds it locally, which is harmless). **Before
pushing wobble work that needs new imfram code:** push imfram, then set
`rev` in `app/Cargo.toml` to that commit's full hash. Forget, and a clean
clone builds the OLD imfram. To check what a stranger gets, clone into a
directory outside the patched parent (so the patch does not apply) and build.

Self-test suites drive the real UI with synthetic input and print one
`PASS`/`FAIL` line, then exit:

```bash
./target/release/wobble data/2k_zip_bg data/2k_zip_bg --test-click
```

The whole battery, one at a time (back-to-back launches contend for the GPU
and time out spuriously):

```bash
B=./target/release/wobble; D=data/2k_zip_bg
for t in click box slice magnet nodeux blendnav tv mswipe tab paint anim gizmokey curves keyer setup \
         spline patchlive zoom cachemeter bin browser thumbs dpx plugin \
         curveedit noise geo sdf mblur \
         nodepack track planar colourkey channels write picker layer shader cost focus keymap cpick zones eyedrop cornerpin gridwarp warp compose subwob blender genai corridor sammie flow seedvr upscale depth ltxhdr stabilize text; do  # + record, with WOBBLE_TEST_FOCUS=1
  $B $D $D --test-$t 2>/dev/null | grep -E "TEST (PASS|FAIL)"; sleep 1
done
```

**A failure that looks like a timeout is often environmental — re-run it
standalone before diagnosing.** (`play` and `rendercache` measure throughput
and are the flakiest under load; they are also the ones worth running by hand
after touching the player or cache.) **But check the budget before blaming
the environment**: `seek`, `scrub`, `scrubperf` and `rendercache` all wait on
real time and all used to time out on iteration counts, which is why `seek`
spent a long stretch documented here as a deadlock it never had.

**Under test the loop is unpaced, so a frame count is not a duration**: any suite waiting on real time must time out on an `Instant`. Suites also: are listed in `Options::any_test()` (hermetic, default prefs, never save prefs), aim at controls through `widgets::note_rect` and check the clip, anchor gestures to a settled layout and to the PRESS point, set `raw.modifiers` every frame of a chord, and gate on state, not throughput.

Suites, what each covers, and the rules for writing one: [docs/claude/testing.md](docs/claude/testing.md).
**Adding a suite** means registering the flag in SEVEN places in `main.rs` (see testing.md). **Adding a node / model / weights**: [docs/claude/recipes.md](docs/claude/recipes.md) — one `CATALOG` row per file with sha256 pinned, declare in `NodeKind::models()`, load only via `models::ready_path`, missing weights = passthrough, nothing downloads without a click; a built-in node is two files (`engine/src/nodes/<n>.rs`, `app/src/nodes/<n>.rs`) plus one-line arms, and the add menus come from `editor::node_catalog()` alone.
Check wasm with `cargo check -p app --target wasm32-unknown-unknown` (no threads, no blocking, no `std::time`, no filesystem in a page — see testing.md).

## Invariants worth protecting (headlines)

Full text, with the why and the measurements: [docs/claude/invariants.md](docs/claude/invariants.md).

- A tick IS a comp frame number.
- A wire's geometry is a `WirePath`, built in ONE place.
- Frame ranges are noodle metadata, resolved in ONE place.
- Mask MIXES, modulate SCALES — two roles, one pipe.
- Pipes are few, and ONE function says what feeds what.
- Parameters are uniforms; only topology changes recompile.
- Uniform block is 16384 bytes with layout clamping.
- Geometry is a layer, never uniforms.
- An SDF graph is fused, and a shape is read at the pixel ONCE.
- A LOCKED channel takes no keys, from anything.
- Animation and patch links evaluate transiently.
- The render-cache key must cover all authored data.
- Comp output is scene-linear.
- The comp has ONE working space.
- Colour matrices come from `engine::color`, derived from published chromaticities.
- The viewer samples NEAREST once magnified past 1:1.
- The pixel readout reports scene-linear AND display values, labelled.
- `InputSpace::Working` is the default and emits no code.
- Plugin behavior lives in the registry; plugin DATA lives on the node.
- A SUBWOB is a node whose behaviour is a graph.
- The display tick is always decoded, or on its way.
- The rig is whole-graph; DECODE is view-scoped.
- A Read's range may run PAST its source, on purpose.
- The ring is the scrub cache: a frame is SHOWN, never taken.
- A seek must not throw away work in flight.
- A frame behind the playhead is dead weight when PLAYING; scrubbing, the line is the PICTURE.
- While chasing, do not speculate.
- A seek is issued in the redraw it was asked for.
- The scrub branch never runs while playing.
- Releasing the viewer cache re-seeks the read player, and a bake retires the last seek.
- A rig rebuild between the take and the upload orphans the taken set.
- Motion-blur samples must see THIS frame's plates.
- A scrub shows a frame BETWEEN the picture and the hand, or keeps the picture.
- An approximate frame moves the picture, not the playhead.
- `--test-scrubperf` cannot see a scrub bug.
- Typed paths go through `engine::paths::expand_home`.
- Both load routes must agree.
- `--test-scrub` gates on WALL TIME, not frames.
- Supported formats live in one list.
- Decoders hand back CODE VALUES; `InputSpace` transforms them.
- A MOVIE is a `Sequence` whose `frames` are all the same path.
- Movie frame indexing counts, it does not divide.
- The properties pane has a pie and a key strip, like the graph.
- Panel rows are one table.
- Every generator takes roto's format row.
- Properties tiles are square at stretch 1.
- Panel tabs are declared, not drawn by the arm.
- A Grade with zones IS a LumaKey feeding masked Grades in series.
- Every colour triple in the app uses `rgbslider`.
- One rgb slider, three places.
- Every "pick a colour" swatch uses `colorpicker`, never egui's stock `color_edit_button_*`.
- A HUD lists its gestures in the hotkey strip.
- The Tinter IS `NodeKind::ColorSlice`: rename strings, never the variant.
- Never use orange in the UI.
- Node names are placed, not just drawn.
- A node too small to read as a shape TWINKLES.
- Z X C V are the sensitivity (drag-rate) row — keep other hotkeys off them.
- A single-letter hotkey belongs to the pane under the cursor.
- Hotkeys are remappable: a new one is a `keymap::DEFS` row, never a raw `key_pressed`.
- UI colours are theme slots: a new one is a `theme.rs` row, never a literal.
- A node can hold a small node graph of its own, edited IN PLACE (`app/src/innards.rs`) — reusable on purpose, roto shapes are next.
- A multi-point Tracker's solve is DERIVED, not keyed.

## Direction already decided (don't re-litigate, don't build early)

- **Scripting is ONE language-neutral protocol, not a Python API.** JSON
  over a local socket, described by a schema that clients are generated
  from; served by the GUI app AND a headless `wobble --serve`. Python is the
  first thin client, not a privileged one (pyo3 in-process only if zero-copy
  pixels ever earn it; an embedded interpreter last, if ever). It edits
  AUTHORED data (never the engine graph), one call is one undo step, a
  non-commercial Write still warns (headless: an explicit flag), and no
  script runs per frame or per pixel — no script expressions (`exprs.rs`
  stays), no synchronous knob-changed callbacks. Prerequisite: the document
  model (`EditorGraph`, `anim`, patch, ranges, tools' `expand`) moves into an
  egui-free `doc` crate the server can link without the UI.

- **Navigation chords live in `app/src/nav.rs` and nowhere else** (manifesto
  11). A new canvas takes `nav::wheel_nav` + the drag table rather than
  inventing its own; a canvas tool that needs alt (paint's clone source,
  the keyer's subtract sample) stands alt-nav down for THAT view target
  and leaves MMB/wheel alone. The day a 3D view exists it uses **Blender's
  tumble**: MMB orbit, shift+MMB pan, ctrl+MMB zoom — which is why shift+MMB
  deliberately stays a plain pan in 2D (same fingers, no retraining).

- **TimeRemap** will follow Slip: same playhead-never-moves contract, but a
  lookup curve instead of a constant — sub-frame source sampling
  (interpolation: blend/nearest, motion-estimated later) and sub-frame
  evaluation of animation curves. Anything touching `TimeMap`, the player's
  tick→frame mapping, or curve evaluation should stay compatible with
  fractional frames.

## Traps (headlines)

Full text: [docs/claude/traps.md](docs/claude/traps.md).

- Leaving an interactive run without `report::session_end()` reads as a crash.
- `report.rs` / `report_win.rs` must stay warning-free on all three OSes.
- Autosave compares against a baseline.
- Stage chains are built during ENCODE, after the frame's `write_params`, and their uniform buffers come up ZERO.
- A PAUSED re-show must not advance the playhead.
- A gizmo that sums per-frame deltas is wrong on an ANIMATED channel.
- A multi-output node courts with EVERY output.
- ⌘C / ⌘X / ⌘V never arrive as key presses.
- A UI test that re-reads what it is dragging measures its own bug.
- On the GPU a test for NaN or infinity must be done on BITS.
- egui's context lock is not re-entrant: never call a `ctx` method inside a `data` / `data_mut` / `input` / `memory` closure.
- Floating egui windows silently occlude viewer gizmos.
- A drag-only `ui.interact` registered AFTER the widgets it covers eats their clicks.
- `ui.separator()` inside a horizontal row is a VERTICAL rule that claims the row's whole available height.
- egui's default fonts lack most symbols, and a missing glyph draws as a box, not an error.
- `gamma_multiply(t).to_opaque()` is not a tint.
- Text painted without clipping bleeds across neighbours.
- egui API drift.
- Disk cache.
- Borrow patterns that recur.

## Subsystem notes

In [docs/claude/subsystems.md](docs/claude/subsystems.md), one section each:

- Cost + licence taint (`cost.rs` / `costviz.rs`)
- Defocus (`codegen::DEFOCUS_*`, `compositor::defocus_bind_groups`)
- Shader nodes (user-authored WGSL)
- Upscale + `nn` (`engine/src/nn/`, `engine/src/upscale.rs`, `app/src/upscaleui.rs`)
- SeedVR2 node (`engine/src/nodes/seedvr.rs`, `app/src/seedvr.rs`, `tools/wobble_seedvr.py`)
- Depth node (`engine/src/nodes/depth.rs`, `app/src/depth.rs`, `tools/wobble_depth.py`)
- HDRCNN node (`engine/src/hdrcnn.rs`, `engine/src/nodes/hdrcnn.rs`, `app/src/hdrcnnui.rs`)
- LTX HDR node (`engine/src/nodes/ltxhdr.rs`, `app/src/ltxhdr.rs`, `tools/wobble_ltxhdr.py`)
- Planar tracker (`engine/src/planar.rs`, `app/src/planarui.rs`)
- Flow + motion vectors (`engine/src/searaft*`, `engine/src/rife*`, `engine/src/flowsolve.rs`, `engine/src/nodes/flow.rs`, `app/src/flowui.rs`)
- Stabilize node (`engine/src/stabilize.rs`, `engine/src/nodes/stabilize.rs`, `app/src/stabilizeui.rs`)
- CameraTrack (`engine/src/camtrack.rs`, `engine/src/nodes/camtrack.rs`, `app/src/camtrackui.rs`)
- 2dini light + SDF effects (`engine/src/geolight.rs`, `app/src/geolightui.rs`)
- 2dini fluid (`engine/src/geofluid.rs`, `app/src/geofluidui.rs`)
- Text node (`engine/src/textfx.rs`, `engine/src/textgpu.rs`, `engine/src/nodes/text.rs`, `app/src/nodes/text.rs`)
- Plasma node (`engine/src/nodes/plasma.rs`, `engine/src/plasmagpu.rs`, `app/src/nodes/plasma.rs`)
- Particles node (`engine/src/nodes/particles.rs`, `engine/src/particlegpu.rs`, `app/src/nodes/particles.rs`)
- LUT node (`engine/src/lut.rs`, `engine/src/nodes/lut.rs`, `app/src/nodes/lut.rs`)
- Reader windows + shared decodes (`decode::Window`, `player.rs`, `main::sync_reader_windows`)
- Loops (LoopIn / LoopOut)
- 3D cards (`engine/src/scene3d.rs`, `engine/src/nodes/render3d.rs`, `app/src/nodes/scene3d.rs`)
- Blender node (`engine/src/blender.rs`, `app/src/blenderui.rs`)
- Scripting API (`api/`, `app/src/apiserve.rs`)
- Curve editor (`curves.rs`, `curveops.rs`, `anim.rs`)
- Subwob panel composer (`tool_layout_body`, `compose_drop`)
- Addons (`app/src/addons.rs`)
- GenAI node (`genai/`, `engine/src/nodes/genai.rs`, `app/src/genaiui.rs`)
- CorridorKey node (`engine/src/nodes/corridorkey.rs`, `app/src/corridorkey.rs`, `app/src/nodes/corridorkey.rs`)
- Sammie node (`engine/src/nodes/sammie.rs`, `app/src/sammie.rs`, `app/src/nodes/sammie.rs`)
- Screen recorder (`app/src/recorder.rs`, `recorder/`)
- Plugin store (`app/src/pluginstore.rs`, `app/src/urlhook.rs`)
- One build, runtime consent (`app/src/report.rs`, `tools/release_public.sh`)
- Project assets on a Load (`app/src/assets.rs`)
- Object storage reads (`engine/src/objstore.rs`)
- Dockable panels (`app/src/dockui.rs`, imfram)

## Where things live

`engine/`: `codegen.rs` (WGSL generation, packing), `compositor.rs` (passes,
`write_params`), `graph.rs` (node kinds), `player.rs` (decode scheduling, RAM
cache), `diskcache.rs` (NVMe tier), `decode.rs` + `dpx.rs` (readers),
`viewtf*` (display transform), `keyer.rs` (colour-keyer maths + the WGSL the
codegen prelude splices in), `nodes/` (one file per built-in node — see
**Adding a node**), `paint.rs`, `bakes.rs` (the bake-done marker + the
tagged EXR writer every subtree render lands through), `sdf.rs` (the SDF
sub-graph: shapes as distances, ops, effects, compiled into the fused pass).

`app/`: `main.rs` (shell, viewer, properties, suites — large), `editor.rs`
(node graph, patch model), `widgets.rs` (tile/row widgets), `curves.rs`
(curve editor), `wheels.rs` (grade wheels), `browser.rs` (sequence browser),
`anim.rs` (channels, curves), `exprs.rs` (patch expression language),
`patchbay.rs` (the patch canvas: jacks, cables, value nodes — the graph's
gestures with the flow turned left-to-right),
`regions.rs` (render-layer-scale grouping: washes, common sources, the
backtick pie's maths; hidden wires are `EditorGraph.hidden_wires`, a
presentation flag over ordinary edges),
`breakout.rs` (the Breakout/Recombine harp: hidden ChannelSelect kids,
the x-locked pair, and the lowering that keeps the engine ignorant),
`layerui.rs` (the Layer stack's mini timeline AND the transform gizmo, which
a Transform draws; layers are placed by upstream Transforms), `viewui.rs`,
`ink.rs`,
`keyerui.rs` (colour-keyer panel: colour-space plots, range brackets, the
sampling readbacks), `sdfui.rs` (the SDF node's panels, glyph, the dive's
graph round trip, the demo and `--test-sdf`).
