# Traps

<!-- Moved out of CLAUDE.md to keep it under Claude Code's size limit. Same rules, same weight: read the section for any subsystem you touch. -->


- **Leaving an interactive run without `report::session_end()` reads as a
  crash.** Each interactive run holds `~/.wobble/sessions/<id>.lock`; the next
  launch reports any session that ended without `session_end` as an unclean
  exit (report.rs). Window close / `exiting` / the end of `main` already call
  it — a NEW `std::process::exit` on a non-headless path must call it first.
  Headless and `--test-*` runs never start a session, so suites are exempt.
- **`report.rs` / `report_win.rs` must stay warning-free on all three
  OSes** (CI's `-D warnings`), and Windows-only code cannot be built for
  Windows on this Mac (C deps). Check it in a scratch crate that `#[path]`s
  both files in, calls them the way `main.rs` does, and stubs `wgpu`
  (only `AdapterInfo`) + `ureq` (`default-features = false`):
  `WOBBLE_GIT=x WOBBLE_STAMP=0 RUSTFLAGS="-D warnings" cargo check --offline
  --target x86_64-pc-windows-msvc` (and the linux + darwin targets). The
  real Windows verdict is `tools/crash_smoke.sh` in CI.
- **Autosave compares against a baseline.** `save_setup`/`load_setup` and the
  end of `build_gfx_with` baseline the comp; a new path that REPLACES the comp
  wholesale (like crash recovery) must either baseline it (it is on disk) or
  `unbaseline()` it (it is not), or recovery will offer back the wrong thing.

- **Stage chains are built during ENCODE, after the frame's `write_params`,
  and their uniform buffers come up ZERO.** `encode_comp` allocates a
  node's `StageRt` (and a Defocus/Loop/Shader `BlurTex`) on first sight and
  on a canvas change; the params written moments earlier went into the
  buffers it replaced, and nothing wrote the new ones until the next edit.
  So a freshly added Blur rendered SHARP, and the proxy toggle dropped every
  effect (`proxy -> full` showed the bare plate; `full -> proxy` a wrong
  blur) until a slider moved — the report was "the effects won't be applied until I
  change something". The rebuild now sets `params_stale()`, and every
  `encode_comp` caller (the display loop, each motion-blur sample, both
  bake loops) writes params again when it is set, BEFORE the submit — a
  staged write lands ahead of the recorded passes, so the same frame is
  right. Keep that pairing when you add a caller. `--test-proxy` pins it.
- **A PAUSED re-show must not advance the playhead.** `try_take_display_set`
  moves the player on — right for playback, wrong for the `refresh_display`
  path, which exists to show the CURRENT tick again after a reader wakes (a
  view flip, a landing Blender render, a bake being adopted). Left advanced,
  the next take — the one that ends a scrub — lands on `t + 1`, so the
  picture and the playhead sit one frame past where the hand stopped. It
  re-seeks to the tick it just showed; keep that if you touch the display
  loop. `--test-bake` catches it. The refresh take also clears
  `force_display`: the scrub branch runs after it and would otherwise swap
  the exact re-show for an in-span neighbour.

- **A gizmo that sums per-frame deltas is wrong on an ANIMATED channel.**
  Measured on `--test-spline`: a drag across several frames moved the point
  by the LAST frame's delta alone, not by the travel. The working copy comes
  from `live_kind(node, t)` = `kind_at(t)` — the curves EVALUATED at the
  playhead — so every frame starts from the same curve-derived base rather
  than from where the previous frame left it. Unanimated channels evaluate
  to their authored values and accumulate fine, which is why this hides.
  **Snapshot at press and compute absolutely** (`pointer - press_origin`
  against the values as they were), the way `curveui`'s
  `Drag::Points { origin, start }` and the spline drag now do. That also
  retires `if hr.dragged() && !hr.drag_started()`, an idiom used across the
  gizmos to dodge egui's pre-press teleport which costs a frame of travel.
  **The cheap tell:** a gesture whose write-back is `keyed_write_back`,
  `spline_keyed_write_back` or `write_hud_triple` is editing a channel
  `live_kind` may own, so it must not sum; one writing straight to a plain
  field cannot hit this. Still summed at the time of writing: transform
  pivot/scale/rotate, corner pins, HUD sliders — nothing tests the animated
  case for any of them. A test that would: key a channel, drag across
  several frames with the path anchored to the press, then assert the net
  equals the travel AND that a key landed at the playhead.
- **A multi-output node courts with EVERY output.** The magnet's pairs are
  (dst, input, src, OUTPUT): it once measured, marked taken and shift-joined
  output 0 only, so no LumaKey band past the first could ever be joined
  that way — while a wire DRAGGED from the band worked, which is what made it
  look like a bug in the band. `band_matte_tests` drives both gestures
  headless and fails on the old join.
- **⌘C / ⌘X / ⌘V never arrive as key presses.** egui-winit turns them into
  `Event::Copy` / `Cut` / `Paste(text)` first, so a keymap binding on ⌘C can
  never fire — the graph's node clipboard reads the events (and binds only
  the ctrl chords). A clip is JSON (`editor::NodeClip`); pasted readers get
  fresh slots (`reslot`); suites never write the user's system clipboard
  (`NodeEditor.system_clipboard`). A synthetic key TAP must be down AND up:
  a second press of a key never released arrives flagged `repeat`, which no
  binding answers.
- **A UI test that re-reads what it is dragging measures its own bug.**
  `--test-spline` moved to `point_position + 20` each hop, which only worked
  while the point lagged the cursor; once the drag tracked exactly, the path
  compounded and the same assertion failed at 1.5x instead of 0.5x. Anchor a
  synthetic gesture to its PRESS point — a real cursor path does not depend
  on what follows it.
- **On the GPU a test for NaN or infinity must be done on BITS.** Metal's
  fast math assumes NaN never happens and folds `x == x` to TRUE for a NaN
  pixel (measured: the Expression node's first GPU test). So
  `engine::nodes::expression` tests special values with `bitcast<u32>`:
  `== inf` is the exact bit pattern, `isnan` is `(bits & 0x7fffffff) >
  0x7f800000`, every comparison is guarded false when either side is NaN
  (true for `!=`), and "non-zero" is `bits & 0x7fffffff != 0`. Arithmetic ON
  a special value (`sin(nan)`, `min(nan, 2)`) stays whatever the GPU makes
  it, and `engine/tests/expression.rs` judges those on ordinary pixels only.
  An infinite WGSL constant is a shader-creation error: `inf`/`nan` ride the
  uniform block. And a half-float output SATURATES on Metal (2.8e14 stores
  as 65504, not inf).
- **egui's context lock is not re-entrant: never call a `ctx` method inside
  a `data` / `data_mut` / `input` / `memory` closure.** It deadlocks — no
  panic, no crash report, a frozen window that users report as a crash.
  `curveui`'s wheel claim read `cumulative_frame_nr()` inside `data_mut`,
  which froze the app the moment a band plot was hovered, and "add zones"
  draws one right under the pointer. Read what you need first, then take
  the lock.
- **Floating egui windows silently occlude viewer gizmos** and eat clicks —
  the stats window did this and broke a suite invisibly. Suspect them when
  viewer interaction dies.
- **A drag-only `ui.interact` registered AFTER the widgets it covers eats
  their clicks.** egui's hit test gives a click to the top-most widget under
  the pointer, and a drag-only rect on top of a button answers "no click"
  rather than passing it down (`hit_test.rs`, the `(Some, Some)` case). The
  panel title row's reorder handle was registered after the row, so the tabs,
  the hud toggle and × were all dead. `widgets::drag_row` registers the
  handle FIRST, on last frame's row rect — the order egui's own scroll areas
  use — and `buttons_in_a_drag_row_still_take_the_click` pins the rule in
  both directions.
- **`ui.separator()` inside a horizontal row is a VERTICAL rule that claims
  the row's whole available height.** Put one between two columns of a
  panel and the row becomes as tall as the tile, so whatever follows it is
  pushed into the scroll area where it cannot be clicked — ColorSlice's (now the Tinter's)
  reset row did exactly this (21pt over, found by its fit test). Use
  `add_space` between columns, and give any new panel the fit test
  ColorCompress and ColorSlice carry: host it at its declared tiles in a
  vertical `ScrollArea` and assert `content_size - inner_rect` is ~0 on
  BOTH axes. It is the only check that sees overflow, because a clipped
  widget still records its geometry.
- **egui's default fonts lack most symbols, and a missing glyph draws as a
  box, not an error.** No `⌫ ⏎ ⇧ ⌥ → ↔ ▲ ▶`; the proportional face has
  `⌘ ⏴⏶⏷⏵ ⟲ × … · — −`, the monospace face far less (not even a backtick).
  egui's own `format_shortcut` checks and falls back to `Shift+Cmd+Z`; your
  literals get no such check. `keymap::tests::every_drawn_character_has_a_glyph`
  scans that file's string literals — copy it for any symbol-heavy UI.
- **`gamma_multiply(t).to_opaque()` is not a tint.** `to_opaque`
  un-premultiplies, which hands the colour back at full strength. Blend
  explicitly for an opaque tint (`keymap::mix`), or draw the translucent
  colour over what is behind it.
- **Text painted without clipping bleeds across neighbours** (the tile-view
  path badges did). Elide *and* clip anything drawn into a cell.
- **egui API drift**: `all_styles_mut` (not `style_mut`),
  `egui_is_using_pointer`, `ui.close()`, `Slider` clamps to its range,
  `SelectableLabel` is gone — `egui::Button::selectable(on, text)`.
- **Disk cache**: `.gcf` v2 pads its header to 4096 so direct IO stays
  aligned (v1 cost 3.5× on reads). `WOBBLE_NOCACHE=r|w|rw` bypasses the OS
  page cache for honest measurement; `w` (uncached writes, buffered reads)
  is the one that doesn't lie.
- **Borrow patterns that recur**: take a collection out with `std::mem::take`
  for the render pass and put it back (tile drawing needs `&mut self` while
  iterating), and defer `&mut self` work (loads, bakes) past a node borrow.
