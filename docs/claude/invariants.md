# Invariants worth protecting

<!-- Moved out of CLAUDE.md to keep it under Claude Code's size limit. Same rules, same weight: read the section for any subsystem you touch. -->


- **A tick IS a comp frame number.** One numbering for the playhead
  (`timeline_pos`/`shown_tick`/`eval_tick`), keyframes, in/out, the shot
  range, handles, Read placement + trims, Write ranges and every bake. No
  modulo, no "index into whatever media is loaded", no offset by
  `comp_range.0` — those all meant a 1001-numbered plate drew its playhead
  off the left edge of the track and evaluated animation at a different
  frame than it displayed. Readers HOLD outside their own frames
  (`source_frame_of`), playback loops on the player's `set_loop` span
  (`play_span()`), and anything reader-bearing declares where it sits
  through `NodeKind::placement()` — extend THAT, do not add a second
  offset somewhere else. `--test-frames` guards it.
- **A wire's geometry is a `WirePath`, built in ONE place.** Every graph
  wire draw and every wire hit test (slicer, `.` dot cut, drop-on-a-wire,
  the magnet's tether, ghosts, glows, value-link cables) takes the path from
  `wire_path` / `plain_wire_path` / `cable_path` in `editor.rs`: the bezier,
  or in right-angles mode (`NodeEditor::ortho_wires`) the `ortho::route`
  polyline. Draw a new kind of wire through them, never a hand-built
  `[Pos2; 4]`, or the mode skips it and its hit test disagrees with what is
  drawn. A side socket's entry comes from `socket_of` (the same code that
  places it). Dashes and colour ramps go piecewise (`WirePath::pieces`):
  chords between resampled points cut a corner on a long cable.
  `ortho_hit_tests` drive the real editor on a spot that is on the route
  but off the curve. **Lanes are computed once per frame** (`ortho::lanes`,
  at the top of the hit tests in `show`, keyed by exact screen ends in
  `ORTHO_LANES`) and `wire_path` serves them, so a draw and a hit test see
  the same laned route; only runs between two corners move, never an end,
  and never past a neighbouring run's corner. Tidy in right angles uses
  `nodeux::ORTHO_ROW_GAP` + a lane's room per jogging output
  (`auto_layout_for(.., ortho)`). `graph_lanes_render` (ignored) draws
  before / tidy / tidy+lanes to `target/graph-lanes-*.png`.
- **Frame ranges are noodle metadata, resolved in ONE place.** `ranges.rs`
  answers every node's first/last frame (a Read's explicit `first`/`last`,
  a generator's script range or its own, else inherited); the viewer range
  (`shot_span`) is the viewed noodle's unless `view_range` pins it, and the
  SCRIPT range (`comp_range` / `script_range`) is a separate thing —
  generators and shot delivery, never the viewer. A plate's frames are
  windowed in ONE place too, `read_window`: the Read, a channel off it and
  its crypto ranks all come through it, frames with no file get the path
  they WOULD have and decode black without being cached, and outside
  first..last the Read's `edges` (hold/loop/bounce/black) answer, in the
  player (`source_index_of`). `--test-range` guards it.
- **Mask MIXES, modulate SCALES — two roles, one pipe.** `mask_input()`
  crossfades the finished effect against the input; `mod_input()`
  (`NodeDef::MOD_INPUT`: Blur, Dilate, Transform, CornerPin) scales the
  effect per pixel. A Blur has NO mask — its one matte port (1) modulates;
  `bridge_mistyped` moves a matte off the briefly-existing port 2. Both
  are `PortKind::Mask` on the wire, so the wiring rules never learned about
  it; the editor tells them apart (`editor::side_dir`: modulate on the
  mirror side, `~` glyph). A modulate port that costs something must be
  free when unwired: `StageDef::aux_ports`/`program` take the node's
  `Wired`, and `engine/tests/modulate.rs` pins that the unwired plan has
  no aux subtree and binds nothing extra. **A Transform's modulate is not a
  stage** — it blends the sample POSITION in the fused pass (`xf_mod` in the
  prelude), with `ParamLayout::modulated` giving a wired Transform
  `MOD_XFORM_SLOTS` more slots of raw params; unwired it is the same three
  rows and `persp` it always was (`engine/tests/transform_mod.rs`).
  CornerPin rides the same `xf_mod` span with an identity move (its
  `mod_blend` too); build a maths-only pin with `CornerPinParams::pinned`. A walk
  that wants "the image input" must skip `is_side_input`, not just the
  mask — the soft-crop walk broke on exactly that.
- **Pipes are few, and ONE function says what feeds what.**
  `PortKind` is Image / Mask / Data / Vector / Channels / Text (+ Field and Geo
  inside sub-graphs; Text is a Prompt node's, read on the CPU by a GenAI
  port, never a picture or a sampled input), and `PortKind::can_feed` is the only wiring rule: same
  kind, plus Mask → Data (widening coverage into a measurement loses
  nothing). Data → Mask is refused on purpose — a range has to be chosen
  and no default is right for every shot. The drag, the magnet, the bridge,
  the Tab splice and the loader's `bridge_mistyped` all ask it; a new
  check that compares kinds with `==` is a bug waiting for the next pipe.
  **Data is ONE channel at full depth, deliberately** (the user's call:
  normals and positions stay on the image pipe until they earn a pipe;
  motion vectors earned theirs, `Vector`).
  `nodes::todata` is the way in and must never clamp or re-range — a
  source scan in its tests fails if it does — and it BROADCASTS the value
  to rgb, so a consumer's own channel pick reads the same number whichever
  it names. `PortKind::is_picture` (Image | Data) is what the resolution
  and frame-range walks follow, so a retyped port keeps its behaviour
  there. Colours come from `editor::pipe_color`, and a dot is made with
  `NodeKind::dot(kind)` — one place each, so the next pipe is an arm, not
  a bool at every site. A port that changes pipe needs a migration:
  `EditorGraph::bridge_mistyped` runs on every load and splices the
  bridge in (the Defocus Z port is the precedent, pinned on the GPU by
  `a_todata_depth_renders_what_the_direct_wire_did`).
- **Parameters are uniforms; only topology changes recompile.** Pipelines
  are cached on a topology hash and never evicted. A param edit that
  triggers a recompile is a bug.
- **Uniform block is 16384 bytes with layout clamping.** Packers must fit the
  budget and keep slot counts stable — e.g. a spline's sample count is
  constant across modes, so switching mode never recompiles.
- **Geometry is a layer, never uniforms.** A 2dini node's `topo_tag` is a
  constant whatever its graph holds; the drawing is evaluated on the CPU
  and rasterised into a per-node layer only when it changes
  (`Compositor::update_geo_layers`), on the Paint precedent. A geometry
  edit that recompiles, or a static drawing that re-records, is a bug.
- **An SDF graph is fused, and a shape is read at the pixel ONCE**
  (`engine/src/sdf.rs`). Distance nodes compile twice over: as functions
  (`sd{owner}_{node}(p)`) for reads at another point — a Shadow's offset, a
  Transform's domain, a Bevel's slope — and inline in ONE entry function
  (`sde{owner}`) as SSA locals for reads at the pixel, so a spline feeding a
  fill, a stroke and a glow walks its outline once. Keep new effects on
  `Cx::inp`/`Cx::arg` and that holds by construction. Lengths are PIXELS,
  packed raw and scaled by the proxy in the shader (`Cx::arg`), and a
  driven socket is scaled only when its driver is a Slider/Input (authored
  units) — a distance is already in render pixels. Spline points stay in
  FIELD space (the roto gizmo's) and become pixels at pack time; the point
  COUNT is the only structure a spline has. Refract needs the node's input
  somewhere else, which a function cannot fuse, so every Refract's bend is
  summed into `sdd{owner}` and the comp emit fuses input 0 a second time
  there (`Emit::input_at`). `SdfParams::preview` (what `1` looks at inside
  the dive) is transient and topology — it only moves while authoring.
- **A LOCKED channel takes no keys, from anything.** The flag lives on the
  channel's `AnimCurve` (`anim::is_locked`), so every write path already has
  it in hand, and the gate sits in the ONE place they all pass through:
  `keyed_write_back` (plus its spline and rgb-triple siblings, and the
  tracker bake). A new way to write a value goes through those or it is a
  hole in the lock. Locking a static channel leaves an EMPTY curve carrying
  the flag — which is why every "is it animated?" test in the app asks
  `!keys.is_empty()` rather than `contains_key`. `--test-gizmokey` guards it.
- **Animation and patch links evaluate transiently.** `to_engine_graph_at(t)`
  applies curves and cables into a throwaway graph; evaluated values are
  **never** written back into `EditorGraph` (that floods undo and thrashes
  the render-cache key). Gizmos/HUDs read `live_kind()`, which recomputes
  from authored data — not from `last_engine_graph`, which lags a frame and
  makes handles ungrabbable right after a seek.
- **The render-cache key must cover all authored data** (nodes, anim, patch,
  ranges, canvas). Miss something and a stale bake replays.
- **Comp output is scene-linear.** The view transform lives at the blit;
  anything reading `comp.output_texture()` must apply one before judging
  brightness.
- **The comp has ONE working space** (`Graph::working`, ACEScg for new
  setups, Linear Rec.709 for setups that predate it). Plates enter it at the
  reader leaf via `color::Transform`; Writes leave it at the file. Between
  those two points every node may assume scene-linear pixels in the working
  primaries — so anything luminance- or hue-shaped derives its constants
  from the working space rather than hardcoding Rec.709. `LumaKey` uses
  `working.primaries.luma_weights()`, the keyer uses
  `keyer::oklab_matrix(working)`. The working space is STRUCTURAL: it is in
  the topology hash and in the render-cache key, because it changes what a
  cached pixel means.
- **Colour matrices come from `engine::color`, derived from published
  chromaticities.** `viewtf.rs` keeps a SECOND, D65-normalised set to match
  OpenDRT; those are display matrices and are not the published ACES ones.
  Never reuse them for an input or output transform — go through
  `color::primaries_matrix`, which adapts white points with Bradford and is
  unit-tested against the published values.
- **The viewer samples NEAREST once magnified past 1:1.** Zoomed in you are
  looking at pixels, and a bilinear ramp between them is a lie about the
  data; minified it still filters, because there the blur is doing real work.
  The switch is made in the blit shader from the ratio of the destination
  rect to the texture size — no second sampler, no bind-group swap.
- **The pixel readout reports scene-linear AND display values, labelled.**
  Confusing the two is the easiest mistake to make in a scene-referred comp.
  `ViewSettings::display_rgb` is the CPU mirror of the shader's display side
  and returns `None` for OpenDRT, which exists only as generated WGSL — the
  readout says so rather than substituting a different curve.
- **`InputSpace::Working` is the default and emits no code.** A wrong colour
  space is invisible — the picture still looks like a picture — so nothing
  is ever guessed silently; a format guess is offered in the UI, never
  applied on load.
- **Plugin behavior lives in the registry; plugin DATA lives on the node.**
  `NodeKind::Custom` carries `type_id` + param values only, so setups never
  depend on a plugin being installed (missing = labelled passthrough). The
  topology hash folds in the plugin's `source_hash` — editing a plugin's
  WGSL recompiles consumers and invalidates cached renders. Param edits are
  uniform writes, never recompiles, same as built-ins.
- **A SUBWOB is a node whose behaviour is a graph.** The UI says
  *subwob* (a subnet); the code still says tool — `app/src/tools.rs`,
  `ToolDef`, `tool.json` — on the Tinter precedent: rename strings, never
  what a file or a setup stores. There is NO `NodeKind::Tool` — an
  instance is a `NodeKind::Custom` like any Tier 0 plugin, whose registry
  entry holds a body instead of WGSL, which is what gives it the properties
  panel, keyable and patchable params, the add menus and missing-definition
  passthrough for free. `tools::expand()` unfolds instances at the top of
  `to_engine_graph_at`, so **the engine never sees one** and a subwob costs
  exactly what its nodes cost by hand. The expanded instance is kept as a
  bypassed orphan because host cables may read its params — un-bypassing
  it would make expansion diverge. Not to be confused with `nodeux::Group`,
  which is a backdrop: decoration that owns nothing.
  **Two homes for a definition.** A COLLAPSE (the ` pie's "subwob", alt+G,
  `tools::collapse`) defines it IN the comp: `EditorGraph::subwobs`, keyed
  `subwob.N` (a `.` never comes out of a library id), a `LocalDef` behind
  `SubwobRef` (Arc + ptr_eq, the `TrackRef` pattern — the graph is cloned
  and compared every frame). So it saves with the setup, rides undo, and
  nests. The registry is a CACHE of the document: `Gfx::sync_subwobs`
  re-registers after a load, an undo/redo, a collapse, a dive out and a
  plugin reload, and `dive_out` writes a local body back into the graph
  that defines it — never to disk. A library subwob is a
  `plugins/<id>/tool.json` (the top bar's **subwobs** dialog writes one),
  shared by every comp. A collapse must leave the PICTURE unchanged
  (`--test-subwob` compares the output before and after).
  **Inputs are nodes, one type per pipe** (`tools::INPUTS`: `input.image`,
  `input.matte`, `input.data`, `input.channels`). An input node is a
  PORTAL: zero input sockets, an output of its pipe
  (`ShaderNodeDef::with_output_kind`). Each is one instance socket typed by
  `ShaderNodeDef::input_kinds`, ordered by its `port` param (a collapse
  numbers them) then id; the first matte port is also `mask_input`, so it
  draws and defaults like a Grade's. `tool.in1..4` / `tool.matte` bodies are
  migrated on load (`migrate_legacy_inputs`). Input nodes and controls are
  registered as tool furniture (`tool: true`) but are NOT subwobs:
  `editor::is_subwob_type` excludes them, and the add menus give inputs
  their own **input** submenu (inside a subwob only).
  **Readers do not cross the boundary yet.** Reader bookkeeping (slots,
  `channel_source`, `instance_loaded`, `compute_reader_instances`) is keyed
  by COMP node ids, so a reader-backed node that only exists once a subwob
  unfolds would bind slot 0 — a plausible, wrong plate. A body holding one
  (`tools::body_reads_footage`) does not unfold or preview, and the
  instance's panel says why; a collapse refuses them. Carrying readers
  through expansion (fresh slots per instance, lookups on the expanded
  graph) is what the fat pipe input is waiting on.
  **Diving into a subwob SWAPS `egraph`** (parking the comp, both undo
  stacks, the selection and the view in a `Dive`) rather than pointing the
  editor at another graph — every gesture and panel already works on
  `self.egraph`, so the swap needs no call-site changes and no site can act
  on the wrong graph. While a body is open the viewer shows
  `tools::preview`: the comp with the LIVE body spliced back in at its
  instance, so a subwob is built against its real inputs. That path skips
  the registry on purpose — the body is unsaved, and re-registering per
  frame would re-hash it per frame.
- **The display tick is always decoded, or on its way.** `pump` enforces it
  by rewinding `issue_tick` onto any hole, and restarts the cursor from the
  playhead whenever it sits outside the prefetch window (the walk skips what
  is pending or in flight, so that costs nothing). Playback therefore steps
  the PLAYHEAD past the frame on screen before asking for the next one
  (`show_display_set`): a request for `display_tick + 1` while the cursor
  was parked behind by a backward prefetch stalled `--test-seek` forever. A
  tick left behind the cursor — by an eviction, a drop, an errored decode —
  is otherwise a frame nobody will ever produce and the display waits for it
  forever. I
  built exactly that deadlock while adding scrub caching (an eviction took
  ONE reader's half of a tick) and it presented as `play`, `anim` and
  `rendercache` all hanging early, nothing like a scrub bug. Two rules keep
  it unrepresentable: evict whole TICKS, never half a pair, and rewind onto
  a hole.
- **The rig is whole-graph; DECODE is view-scoped.** `reader_instances`
  walks every sink, so the rig holds every reader the graph could show, and
  its slots, time maps and the RAM cache's reader indices never move on a
  view flip. But the player decodes only readers upstream of the viewer
  slots: `sync_time_maps` walks `Graph.view` + `Graph.hot_roots` (the other
  slots, lowered exactly like `view` — A/B flips stay instant) with
  `reader_instances_from` and pushes `Player::set_active`; `pump`'s issue
  loop and hole check, `have_all` and `take_set` skip inactive readers, so a
  tick is complete when every ACTIVE reader has it. Before this a nine-string
  harp decoded nine layers per tick whatever was on screen — and so did every
  Read in a comp. **A render decodes everything**: both bake loops call
  `set_active(Vec::new())` before they pump, because a bake's pump runs
  before `sync_time_maps` has seen it, and a set missing a reader bakes stale
  pixels to disk. A rebuilt player inherits the mask by slot
  (`inactive_slots`); a slot it has never seen starts active, so nothing is
  ever starved for a frame. `--test-harp` pins the cost AND the pixels: a
  mask that activated the wrong reader would be exactly as cheap.
- **A Read's range may run PAST its source, on purpose.** That is how you
  watch a render arrive: set OUT beyond what exists and the timeline covers
  frames nobody has written yet. `Sequence::trimmed` extends rather than
  clamps, filling the tail with the paths those frames WILL have
  (`bump_frame_number` advances the last digit run, keeping the padding).
  The rule that makes it work is in `spawn_decode`: **a frame that is absent,
  or that fails to decode, is delivered BLACK and is never cached** — not in
  RAM, not on NVMe. Cache a missing frame and it stays missing forever, which
  is the entire feature broken. Existence is re-checked per decode, so a file
  appears the next time the playhead passes it, with no rescan and no reload.
  A decode failure is treated the same way and logged at debug, because the
  common cause is a frame the renderer is still writing.
- **The ring is the scrub cache: a frame is SHOWN, never taken.** A decoded
  frame stays in its staging slot after the copy to its reader texture, so a
  tick the hand comes back to is one GPU copy — no decode, no CPU memcpy —
  and a clip that fits the ring scrubs exactly at 60Hz at zero decodes
  (`--test-scrubfid`: 0 decodes, 1 redraw, every tier, 2K–8K). The contract:
  `Player::show_set(tick)` (a re-show, a bake's frame), `show_display_set()`
  (playback: steps the playhead past the frame on screen, shows the next,
  advances) and `show_nearest_set()` (a scrub: `Scrub::{Exact, Between,
  OnScreen, NotYet}`) hand out a `FrameSet` naming slots that stay in
  `pending`; the slots are HELD until the app calls `frame_submitted()`
  after the submit that carries the copies — every redraw, on every player,
  and in both bake loops. Nothing remaps a held slot: an eviction skips the
  tick, `invalidate`/`forget` defer the release to `frame_submitted`. The
  Player owns every release (`evict_farthest`, whole ticks, never the frame
  on screen); the app never calls `release_after_copy`, and there is no
  `discard_set` — a set shown and superseded in the same redraw simply stays
  resident. The ring is sized by a byte budget (`App::ring_slots_for`: half
  of `cache_bytes` per player, never below the classic twelve) and grows
  lazily — `StagingRing::new` allocates nothing, `try_acquire` creates slots
  as decodes need them, `available()` counts what a decode could take. A
  slot is freed only to give its bytes to a frame of another size (see
  **Reader windows + shared decodes**);
  a clip larger than the ring evicts by distance from the playhead. The CPU
  RAM tier (`FrameCache`) remains a second tier at its own budget — retiring
  it is the next step, and the reason the ring takes only half (at 8K the
  half-budget holds 22 frames of the test rig's 24: the viewer cache is
  fully resident and exact at 60Hz, the read rig re-decodes ~55 per pass).
  Rules the residency brings, each one a suite or a review finding:
  **the ring is floored at `readers × (ahead + 3)` slots** (`ring_slots_for`)
  — the prefetch window, the pinned tick on screen and one spare for the
  eviction round-trip; the classic twelve as a scrub cache played a 48-frame
  clip at 90 sets/s against 120, and one tick's worth deadlocks outright.
  **Once the ring is at budget every decode needs an eviction, so `pump`
  evicts AHEAD of need** — `window_holes()` counts what the prefetch window
  still lacks and the end-of-pump top-up frees that many ticks plus the
  reserve plus one (an evicted slot returns a poll later); with speculation
  merely gated at the reserve and nothing evicted, a clip longer than the
  ring stopped prefetching and played at one late decode per tick (measured
  24 sets/s). **"The hand is on the picture" is only an answer while the
  picture is current** — `Scrub::OnScreen` satisfies a seek without an
  upload, so `Player::picture_stale` is set by every event that makes the
  on-screen pixels stale for their tick (a rig rebuild, a viewer-cache swap,
  `invalidate`, `forget` of the shown tick) and cleared by any successful
  show; while set, the exact tick or a between-frame is answered instead,
  and playback does not step past the shown frame. The app also sets
  `refresh_display` on those events so the exact re-show is preferred, but
  the scrub branch keeps running — a drag begun right after a rebuild still
  tracks the hand. `--test-proxy`, `--test-range` and `--test-nodepack` each
  failed on the stale-picture case before the rule. **A black stand-in is
  never resident**: `DecodedFrame::transient` marks an absent or undecodable
  frame; `pump` drops it unless it is the display tick, and `show` releases
  it at `frame_submitted` — so the next pass re-checks the disk and a render
  written mid-run is picked up (`--test-grow`), exactly as the RAM and NVMe
  tiers never cached it either. **The budget is honest**: the ring takes
  `cache_bytes / 2` and the CPU RAM tier the other half
  (`ram_tier_budget`), so the two together are what the user set.
  `--test-harp` counts reader frames UPLOADED per shown tick
  (`uploaded_total`), not decodes: a second pass over a resident clip
  decodes nothing, and the claim is about which readers a view keeps fed.
- **A seek must not throw away work in flight.** `Player::seek` moves the
  playhead and NOTHING else: decoded frames stay in `pending` (a drag comes
  back over them constantly), in-flight decodes land and are used, and ring
  pressure is relieved by evicting whatever is furthest from the playhead.
  The old seek discarded everything, which cannot work: a scrub re-targets
  every ~16ms and a 2K decode takes ~27ms, so nothing survived — measured at
  **0 frames displayed across a 3s drag, from 970 decodes**. Work-preserving
  seek measures **53fps from 469 decodes** on the same drag. `pump` skips a
  frame already pending or in flight, so a re-targeted playhead never orders
  the same decode twice.
  **`Player::invalidate` is the version that discards**, and it exists for
  exactly one caller: a time-map change, where a tick now names a different
  source frame so the decoded pixels are genuinely wrong. Do not reach for it
  from a seek.
- **A frame behind the playhead is dead weight when PLAYING; scrubbing, the
  line is the PICTURE.** The display only moves forward during playback, so
  keeping frames behind it starves the ring. Chasing a scrub (`ahead == 0`)
  the useful set is the span between the tick on screen (`Player::shown`,
  pushed by the app every frame) and the playhead: a landed frame outside
  it can never be shown by the between rule below, and holding its slot is
  what starved the ring — landings arrive out of order, the ones the
  picture had passed piled up, and every new target waited on an
  eviction's remap (measured: 46 pictures for 117 seeks, +15ms on every
  landing). PLUS whatever lies past the playhead in the drag's direction
  (`scrub_wants` = the span, or `ahead_of_hand`): speculation lands there,
  and a first cut that kept only the span dropped every prefetched tick
  that landed during the redraw or two the player was chasing, then
  decoded it again — 572 decodes for 120 backward seeks against 244.
  `pump`'s arrival rule and `evict_farthest` both branch on `ahead == 0`
  and score by `scrub_wants`. The RAM tier is the cache for a hand that
  comes back, not `pending`.
- **While chasing, do not speculate.** Prefetching `target..target+ahead`
  during a drag fills every ring slot with decodes for ticks the hand has
  already left; they land obsolete and are evicted, so the tick actually
  wanted can never get a slot (measured on 4K: pending 0, inflight 12, free
  0, permanently, 1fps). The app sets `ahead = 0` while `force_display`, and
  `pump` caps in-flight work at two thirds of the ring so landed frames have
  somewhere to sit. That cap is a CLIFF, not a slope: 8 in-flight jobs on a
  12-slot ring measured ~17fps at 4K, 10 jobs fell back to 7. The moment the
  exact frame lands speculation resumes — in the DIRECTION of the drag
  (`Player::backward`, from the app's last two seek targets): forward-only
  prefetch made a backward drag pay a full decode per step and queue it
  behind five speculative forward ones (384 decodes for 120 backward seeks
  against 253 forward). The scrub take steps the playhead the same way, or
  the frame just shown sits one step "ahead" and is decoded again (500
  decodes for 120 seeks). **Speculation never takes the last tick's worth
  of slots** (`Player::reserve`, one tick while scrubbing, 0 while playing):
  in-flight work cannot be evicted, so a reversal used to queue its first
  steps behind ten wrong-way decodes — 3.1 redraws per reversal against
  1.0 for a straight step; with the reserve the target is issued at once
  while the wrong-way landings are dropped.
- **A seek is issued in the redraw it was asked for.** The redraw pumps at
  its top, then runs the UI (where a drag seeks), so a seek used to wait for
  the NEXT redraw's pump — one vsync (16.7ms) on every scrub landing, the
  whole floor on a buffered clip. `pump_after_ui` pumps again after the UI
  pass and after `sync_time_maps` (which may invalidate), with the same
  flags recomputed. `--test-scrubfid` reports landings in REDRAWS as well
  as ms because the unpaced test loop redraws thousands of times a second:
  forward sweeps went 1.9 -> 1.0 redraws.
- **The scrub branch never runs while playing.** Playing, a neighbour is
  simply the wrong frame, and the playback take lands the seek exactly (and
  clears `force_display`). It used to run at startup too — nothing on
  screen yet, so any decoded tick was "between" — and could show tick 1
  before tick 0 landed; with the playhead staying on the hand, a rebuild
  then snapped the picture back to 0 (`--test-mblur` failed one run in four
  on that race).
- **Releasing the viewer cache re-seeks the read player, and a bake retires
  the last seek.** The read player sat idle while the cache played — or was
  walked to the END of the range by the bake that filled it — so any comp
  edit while cached used to teleport the picture there. `release_render_cache`
  seeks it to `timeline_pos`; both bake loops set `seek_target = None` when
  they move the playhead, so everything that asks "where is the user"
  (`refresh_display`, the release) reads the playhead. Bakes also reset
  `ahead`/`backward`/`reserve` on the player they borrow: a bake started
  right after a backward scrub walked backwards with no prefetch.
- **A rig rebuild between the take and the upload orphans the taken set.**
  The properties bin can set `playback_dirty` in the UI pass; the rebuild
  that follows drops the ring whose slots the set names. The set is let go
  (not uploaded), and the rebuild's own seek + `force_display` land the tick
  next redraw. The viewer cache's player is untouched by a read-rig rebuild.
- **Motion-blur samples must see THIS frame's plates.** The uploads sit in
  the frame encoder, which is submitted last; the samples submit their own
  encoders first, so they blurred last redraw's plates. The upload encoder
  is flushed ahead of the sample loop (submission order is execution order;
  the timer's query set resolves in a later submit, which is fine).
- **A scrub shows a frame BETWEEN the picture and the hand, or keeps the
  picture.** `try_take_nearest_set` once handed over whatever pending tick
  was nearest the playhead, and consumed it. `pending` during a drag is
  speculation plus decodes for ticks the hand already left, so every UI
  frame the exact tick was still in flight the next-nearest of those was
  shown — the picture walked AWAY from the hand one frame at a time, and a
  backward step was shown the prefetched FORWARD neighbours first. That was
  the reported "jittery or random images": `--test-scrubfid` measured a 60Hz
  backward drag at 49 regressions and 47 flips per 120 seeks (170 pictures
  for 120 targets, up to 19 frames off), IDENTICAL with the RAM tier warm
  and on the viewer cache, because the junk was already decoded. Playback
  was fine because playback never takes a neighbour, and a forward drag
  was nearly fine because the prefetch happened to run its way. Candidates
  are now restricted to the shown..playhead span, so the picture may LAG
  the hand but never leads it, jumps past it or moves against it; with
  nothing in the span the caller keeps what it has. Do not reintroduce a
  "nearest anywhere" fallback, however tempting on a cold 4K plate — the
  cold case wants frames landing between, which the between rule shows.
- **An approximate frame moves the picture, not the playhead.** The display
  loop sets `timeline_pos` from a set only when it is exact (or playing);
  `shown_tick` always follows the pixels. The viewer's Tab-drag steps
  from `timeline_pos`, so a stale neighbour on screen used to rebase the
  drag onto itself: `--test-scrubfid`'s `relback` measured a backward
  drag landing off-course 79 steps in 89. The hand's position is
  `seek_target`; anything that wants "where the user is" reads that.
- **`--test-scrubperf` cannot see a scrub bug.** It counts frames put on
  screen, not WHICH frames, and passed at 59fps while every other picture
  was wrong. `--test-scrubfid` records (hand target, shown tick) per
  displayed frame across eight patterns (60Hz sweeps both ways, slow steps
  both ways, reversals, a stationary hand, relative drags both ways) in
  three tiers (cold, RAM tier warm, viewer cache warm) and fails on any
  regression (the new picture farther from the hand than the old), flip
  (against the drag), junk to a still hand, or a hijacked drag base. Run it
  after touching `try_take_nearest_set`, the `force_display` branch,
  `pump`'s arrival/eviction rules or `seek_to`.
- **Typed paths go through `engine::paths::expand_home`.** Every field that
  takes a path as TEXT rather than from a file dialog — the browser's
  directory bar, a Read's path, a Write's path, the NVMe cache directory — expands a leading `~`, because that is what
  people type. Only a LEADING tilde, and only as a whole segment: `~sam/…`
  is left alone rather than guessed at (it needs a password-database lookup
  we do not do, and gluing a name onto our own home's parent would silently
  read the wrong directory). Expansion happens at the RESOLUTION site
  (`load_read`, `WriteParams::resolve_frame`),
  not where the text is edited, so the authored string stays what the user
  typed and setups stay portable between machines.
- **Both load routes must agree.** A Read loads either by PATH
  (`load_read`, the "load" button, a typed path) or by a picked FRAME LIST
  (`load_read_files`, the browser). The browser hands a single-file pick over
  as a one-entry list, so a movie is indistinguishable from a still there —
  `picked_movie()` is what stops the two routes disagreeing. They did:
  picking a clip gave a one-frame Read and only pressing "load" fixed it.
- **`--test-scrub` gates on WALL TIME, not frames.** Under test the event
  loop spins ~13k iterations/second with no surface, so a frame count is not
  a duration — and that suite waits on a 24fps transport and real decodes. A
  frame-count timeout there looks like a broken player and is not one.
- **Supported formats live in one list**: `engine::decode::SUPPORTED_EXTS`,
  shared by the browser filter, the scanner and the decoder. Add formats
  there, plus a backend in `AnyDecoder`. A format that only some builds can
  decode is `#[cfg]`-ed OUT of the list (movies are), because listing one we
  cannot read is worse than not offering it.
- **Decoders hand back CODE VALUES; `InputSpace` transforms them.** `still.rs`
  (PNG/JPEG) and `movie.rs` do exactly that. `dpx.rs` predates the input
  transforms and linearises internally, so DPX + an accepted Cineon guess
  applies the curve twice — a known inconsistency, not a pattern to copy.
- **A MOVIE is a `Sequence` whose `frames` are all the same path.** The
  index into that list is the frame number, handed to the new
  `ExrDecoder::decode_frame_into`; every other backend inherits the default
  (frame ignored, `frame_count` 1) and does not care. That is what lets time
  maps, residency, the staging ring and both caches work on a movie unchanged.
  Trims must go through `Sequence::trimmed`, NOT a hand-rolled slice of
  `frames`: a movie's frame numbers live in the index, so a trim has to carry
  `offset` or the clip silently replays from its start.
- **Movie frame indexing counts, it does not divide.** `avg_frame_rate` is
  derived and comes back as nonsense like 576/23 on short clips, so
  `movie.rs` indexes by the NOMINAL `r_frame_rate` and, once located, by
  counting pictures out of the decoder. pts is used only to place the frame a
  seek landed on — and because avformat seeks on DECODE timestamps, that seek
  can overshoot past the target in a stream with B-frames, which is detected
  and retried from further back rather than silently returning the wrong
  frame.
- **The properties pane has a pie and a key strip, like the graph.**
  `props.pie` (`` ` ``, scope `Props`) opens on the keyable row under the
  pointer — every `slider`/`drags`/`rgb_row`/`slider_cell` registers its
  rect and channels (`widgets::RowHit`) — or on the whole panel off any
  row. A new common action is a `PropsOp` variant + a `PROPS_PIE` row + an
  arm in `props_pie_ui`. A LOCKED row wears a red wash (`theme::row_locked`,
  painted under the row by `end_row`). The bar's key strip
  (`props_keys_strip`) leaves out the Z X C V B N sensitivity row.
- **Panel rows are one table.** Every row is `LABEL_W` label, control,
  `NUM_W` value field, `DOT_W` key slot: `widgets::slider` for keyable
  bounded values, `widgets::drags` for unbounded ones (its fields are
  RIGHT-JUSTIFIED into the slider's value column), `widgets::plain_slider`
  for a value that is not a channel. Never a bare `egui::Slider` or a
  hand-sized label in a panel — its auto-width value box breaks the column.
  A number box is `widgets::num_field` (`num_field_w` for a wider one),
  never `add_sized([NUM_W, NUM_H], DragValue)`: `add_sized` is only a
  MINIMUM and egui grows a DragValue to its text. `numfield::fixed`
  (egui-only, shared with the colour designer) allocates the exact box and
  drops decimals until the number fits it.
  A column of fields laid out side by side (a table) calls
  `widgets::pack_left` so its rows do not pad across the panel.
- **Every generator takes roto's format row** (`widgets::gen_format_row`:
  inherit / named presets / custom). The params keep a `size` AND an
  `inherit` flag (field-level serde default false, so old setups keep their
  size); inherit is resolved in ONE place — `node_resolution_full` skips
  the declared size and the node takes the canvas like a format-less roto
  shape. The engine never reads `inherit`: generators pack from the stamped
  per-node `formats`. A new generator: both fields, one guarded arm there.
- **Properties tiles are square at stretch 1** (`widgets::TILE`) and panels
  occupy whole tiles. The GRID is twice as fine: a tile is 2x2
  `widgets::CELL`s + gap, panels still declare tiles, and `patch_span_of`
  is the one place tiles become cells (old setups' tile cells are scaled
  on load by `migrate_cell`). Panels are drawn at exactly their span by
  `widgets::tile_frame` and clip anything wider — the Grade wheels shrink
  to fit instead. Zoom scales geometry (slider tracks, colour wheels,
  gaps) but never type, drag fields or knobs. On top of it, a per-axis
  STRETCH (`prop_stretch`, the strip's w/h fields, ctrl+MMB drag) resizes
  only the tile BOX — widgets keep their scale, so x lengthens tracks and y
  shows rows a tile cut off. Size a panel's contents from the space it is
  given, never from `TILE * zoom`.
- **Panel tabs are declared, not drawn by the arm.** `widgets::panel_tabs`
  returns a node's `PanelTab`s (id + label); `properties_ui` draws the strip
  in the title row (a row of its own when it will not fit beside the name),
  keeps the selection per node in egui frame memory BY ID, and hands the
  arm the selected id in `tab`. Ids, not indices: adding zones from the
  Grade's ranges page must not bounce the selection off the ranges page.
  A one-page panel declares nothing and draws nothing new.
- **A Grade with zones IS a LumaKey feeding masked Grades in series.**
  `GradeZones.key` is a `LumaKeyParams` — the same partition model, edited
  by the same `bandui::luma_panel` (with `BandOpts` picking the zone flavour
  of the row under the plot) — and `adjust[i]` is what band `i` does. The
  shader evaluates the band's response exactly as the LumaKey packs it and
  applies each zone as `mix(so far, grade4(so far), w)` after the master;
  `engine/tests/grade_zones.rs` pins it against the hand-built graph. Two
  things that must stay true: a plain Grade (`zones: None`) has the tag `2`
  and four slots it always had, so nothing already cached is invalidated;
  and a zone's span is FIXED (`codegen::ZONE_SLOTS`, `ZONE_PTS` points) so
  only the zone COUNT recompiles — the band editor stops adding points at
  the cap rather than let the packer drop the top of a band. Splits and
  dissolves keep the corrections and the band names with their bands
  (`LumaKeyParams::split_band`/`dissolve_band` do the bookkeeping the raw
  `sync_from_ranges` cannot; `luma_panel` reports them in `LumaOut`).
  **The ranges PREVIEW follows the keyer's contract, not a new one.**
  `GradeZones::preview` (a `ZonePreview`: matte / highlight / zones / off,
  plus the zone) is AUTHORED — a knob that undoes and saves — and `previewing` is
  transient, set in `to_engine_graph_at` for the view target exactly like
  `KeyerParams::previewing`; `live_preview()` is the one place that ANDs
  them and clamps a zone a dissolve has removed. So a preview cannot reach
  a render, a bake or a consumer, because a node that is the view root has
  nothing downstream and a bake re-roots `view` at what it is baking. It
  costs no slot and no recompile: the mode rides in `gamma.w` and the zone
  in `gain.w`, the two lanes `grade4` does not read. The ZONES map is
  Σ weight × colour with the colours as shader CONSTANTS (they depend on the
  zone count and the working space, both already structural), so it too is
  a uniform flip; HARD is the colour of the heaviest zone (strictly, so a
  tie goes low and an uncovered pixel stays black), one compare per zone.
  **One band palette**: `graph::band_srgb` is what the maps render AND
  what `bandui` draws the plot in (lifted there to read over its dark
  ramp), so the plot is the viewer's legend — a second palette would make
  it lie. It is CATEGORICAL and judged AS SHOWN: the greedy max-min of a
  saturated pool through the default ACES view (CIE76 ΔE), with no
  neighbours whose crossfade passes through orange (a green→red blend is
  orange in the middle, and the blended map draws exactly that). Re-derive
  it, do not hand-tweak it: `graph::zone_tests` measures neighbours, pairs
  and every neighbour blend through the view, and fails on a tweak that
  reads fine in sRGB and not on screen. `grade_zones.rs` pins both maps
  against the mattes off the same GPU.
- **Every colour triple in the app uses `rgbslider`.** `MultParams::value`,
  the Grade's four adjustments (in slider mode) and the Glow tint all draw
  the same widget; the tint gained `GlowTintR/G/B` channels on the way,
  because a colour-picker button cannot be ganged, keyed or patched.
  What deliberately does NOT use it: the RGBA swatches (`GridParams::line/back`, paint
  stroke colour) — those are "pick a colour" with an alpha, which the slider
  has no lane for, and they use `colorpicker` (below). Plugin manifests only declare scalar floats, so a plugin
  has no triple to draw yet. A node carrying an rgb row needs TWO TILES:
  `rgbslider::CHROME_W` is ~126px, and in a one-tile panel the track came out
  at 52px — a slider with nothing to aim along. `--test-rgb` pins both the
  minimum track and the fact that each tint node draws this widget.
- **One rgb slider, three places.** Any rgb triple — a properties row, a
  Grade HUD row, a dragged-out HUD — is the same instrument: ganged and
  relative by default, `G` inverts the gang while held, and `S`/`D`/`F` arm
  saturation / orange↔cyan / green↔magenta. **The lane NUMBERS obey the
  gang too** — a field dragged or typed into moves all three relatively,
  `G` peels it back to its own lane — so they go through `nudge` off a
  SCRATCH copy rather than binding `v[lane]`, which is what once made the
  fields the one part of the instrument that could not move together. The
  colour axes stay on the rails: they are aimed gestures whose units are
  not a lane's number. The colour axes preserve the
  WORKING SPACE's luma exactly (additively for offset-type triples, in log
  space for scale-type ones, which also keeps a multiplier positive), so
  brightness is the one thing they never change. An armed axis turns the LANE
  LINES into a picture of it (what it does to mid grey across a full drag,
  centre = no change) rather than laying a band over the row — these appear
  on HUDs that sit on the shot. `app/src/rgbslider.rs` owns the maths and
  that preview and depends only on egui — the app hands it
  luma weights and a working→Rec.709 matrix rather than the widget reaching
  into `engine`. A Grade switches between wheels and sliders
  (`ENode.grade_sliders`, app-side UI state like `blend_res`) and the two
  views share both the node's gang flag and the wheels' ranges, so neither
  can reach a value the other cannot. Slider mode lays out TWO COLUMNS of
  two: four stacked rgb rows are taller than a tile, and the overflow lands
  inside the tile's scroll area where it cannot even be clicked.
- **Every "pick a colour" swatch uses `colorpicker`, never egui's stock
  `color_edit_button_*`.** `app/src/colorpicker.rs` is a swatch that opens a
  popup with wheel / rgb / hsl / lch modes over ONE stored value (linear,
  working primaries, 0..1). Like `rgbslider` it depends only on egui: the app
  builds its `PickSpec` in `main::pick_spec` — the working→Rec.709 matrix
  pair and the z/x/c/v hooks (`widgets::sens_factor`, `widgets::claim_sens`).
  Three rules that are easy to break from outside. **What each mode means is
  fixed**: rgb is the stored numbers; wheel/hsl are those numbers under ONE
  display encode, exact sRGB, which the previews and hex share (so the swatch
  and its hex are the same bytes); lch and hex go through Rec.709 (Oklab is
  defined there) — change one and every setup's colours shift.
  **The popup must claim the rate keys while hovered** (`PickSpec::claim`):
  V is a rate key AND the props-bar toggle, and without the claim holding V
  over the popup hides the panel the popup lives in. **Coordinates outlive
  their degenerate points**: the view being dragged keeps its own
  coordinates and the others re-derive with fallbacks, so a hue survives
  saturation 0 and lch keeps an authored chroma the gamut cannot hold (and
  underlines where). **Every palette slot is an eyedropper target in its
  own right** (`E` over it, then click the viewer): the sample lands in
  that slot — in the `+` slot as a new swatch — without passing through
  the value, which is how a palette gets built from the shot rather than
  from the picker. Two things make it work and are easy to undo: the
  swatch's own hook arms from anywhere over the popup, so it stands down
  over `Geom::swatch_grid` (whoever runs first consumes `E`), and the
  popup honours `Geom::swatch_dropping` alongside its own `dropping`, or
  the viewer click carrying the sample reads as "clicked elsewhere,
  close" and the slot never receives it.
  **Swatches are app-wide and live in linear Rec.709**
  (`Prefs::picker_swatches`), never in a comp's working space: a swatch
  kept in an ACEScg comp must be the same colour in a Rec.709 one, so
  `to_palette`/`from_palette` go through `PickSpec::display`, and a
  swatch past the comp's gamut clamps on use rather than on keep. Using a
  swatch sets the colour and never the alpha. **Its size is arithmetic, and the layout must match
  it**: one app-wide scale (`colorpicker::scale`, a `Vec2`, default 3 both
  ways, persisted as `Prefs::picker_scale`/`_y` once a resize settles)
  drives `Geo`, where aimed geometry grows and type and number fields do
  not (the properties-zoom rule). **The axes are independent**: x is
  everything measured ACROSS (lane length, chips, the value strip's
  thickness — `Geo::strip_w`, which is why the height scale cannot widen
  the wheel row) and y everything STACKED (track height, lane height,
  gaps), so the popup's outer WIDTH is a function of `s.x` alone and
  `fit_scale` can solve one axis and then the other. The wheel takes the
  smaller axis and stays round. A drag on the frame margin MOVES the popup
  (`move_zones` — bands, never a rect over the content: a drag rect over a
  widget takes its hover the moment the button goes down, and the old|new
  chip stops answering clicks), press-anchored like every other gesture
  here, and the offset clears on the next open. The popup is placed, fitted to the screen and given its resize
  grips from `Geo::size` BEFORE anything is laid out, so a row added
  without updating `size` misplaces it — `size_is_the_layout` measures
  every mode against real egui passes and fails first. `--test-cpick`
  drives the popup live (incl. an edge resize, reopen, pinch and the
  double-click reset); the unit tests pin the maths and snapshot the modes
  to `target/cpick_*.png`.
- **A HUD lists its gestures in the hotkey strip** (`app/src/hudkeys.rs`):
  a HUD with held-key or modifier gestures pushes `hudkeys::key`/`chord`
  rows into `hud_key_list` while it is drawn, and the viewer paints one
  strip along its top — ink caps, words rising at 45°, a held key lights
  its cap. Roto is the first (`S`/`D`/`O`/`I`/`E`/`R`/`T` + the box chords); the
  menubar's `hud keys` (`Prefs::hud_keys`) hides it. Keep the list in step
  with the gestures, and with `keymap::FIXED`.
- **The Tinter IS `NodeKind::ColorSlice`: rename strings, never the
  variant.** The UI name (`NAME`, the catalog, the panel) is Tinter; the
  setup key stays `ColorSlice`, because "Tinter" is what setups from before
  the Sep-23 prune stored for this node's PREDECESSOR, whose params are not
  these — reusing the key would load those as a silently different node.
  Its panel is Resolve-shaped: a row per vector, sliders whose rails are
  pictures of the effect (`widgets::slider_cell`, a full parameter row
  without its label, painting the rail from a colour-at-value function run
  through the node's CPU mirror). `tinter_panel_render` (ignored) draws it
  with text to `target/tinter-panel-gpu.png`.
- **Never use orange in the UI.** A user rule, blanket. The status palette
  is red for alpha, lime for beta and plain white for production
  (`editor::NodeStatus::color`, pinned by a catalog test); reach for those,
  or for the existing type/pin colours, never an orange.
- **Node names are placed, not just drawn.** They are collected during the
  node loop and drawn AFTER it, once every node's rect exists, by
  `editor::place_label`: level first, then swinging up around the node's
  centre (`LABEL_ANGLES_DEG`) until the oriented box clears every other
  node and every label already placed (separating axes, `quads_overlap`,
  behind an AABB broad phase). Left to right, so a row fans up in step.
  `label_tests` pins it; `label_collision_render` (ignored) draws the case.
- **A node too small to read as a shape TWINKLES** (`editor::twinkle_star`,
  under `SPECK_R` drawn px — scaled down, zoomed out, or both): a hot core
  that never blinks out, four spikes and a halo tinted with its type colour,
  each node on its own phase so a field of specks shimmers rather than
  pulsing in lockstep. The maturity pip fades out under it — at that size
  the dots are bigger than the node. `a_speck_twinkles` pins it.
- **Z X C V are the sensitivity (drag-rate) row — keep other hotkeys off
  them.** Held over any draggable value they scale the throw
  (`rgbslider::RATE_KEYS`, with B/N). Only the graph, which drags no values,
  may reuse Z/X (pin panels / gizmos). App keys: `Q` fullscreens the pane under the
  cursor ALONE (`ViewerMode::FullGraph` / `FullProps` / `Full`; the central
  panel draws that pane, and `graph_panel_rect` / `bottom_rect` follow it so
  hover-gated keys keep working), ⌘Q
  graph overlay (the macOS default menu is OFF so ⌘Q reaches egui — no quit
  key), `W` props bar (shift+`W` every viewer HUD — the menubar's `hud
  off`), `Y` curve editor, `A` alpha, Esc back. Graph `` ` ``
  = node-ops pie (`editor::OPS_PIE`; a pick sets `ops_pick` and fires
  through the op's own key site — and counts as HOVERED on that frame,
  because the slice it was released on can sit past the pane's edge and
  every op there gates on `over`; `--test-subwob` caught subwob never
  firing), ⌥` global sources, ⌘` region sources.
- **A single-letter hotkey belongs to the pane under the cursor.** The
  graph's editing keys (Delete, `E`, `D`, `U`, `R`) fire only while the graph
  is hovered, and `rgbslider::grabbing()` suspends the app's whole hotkey
  block during a slider drag — otherwise the colour axes, which are held
  letters, would also frame the viewer and bypass nodes. Framing lives on
  `R` (it moved off `F`, which the tint axis needs, then off `M`).
- **Hotkeys are remappable: a new one is a `keymap::DEFS` row, never a raw
  `key_pressed`.** `app/src/keymap.rs` declares every remappable action
  (id, label, pane scope, default, match rule); sites read live through
  `keymap::pressed/released/down/consume`, prefs store only overrides as
  strings, and the keys… window (menubar) rebinds with capture + same-scope
  conflict flags. `keymap::capturing()` joins `rgbslider::grabbing()` in
  the stand-down guards. Deliberately fixed: nav chords (manifesto 11) and
  the rgb colour axes — listed read-only in `keymap::FIXED`, so give a new
  held-key gesture a row there. The window's board, list, inspector and
  click-select all ask `Pick::lights`, so they cannot disagree about a
  key. The Tab steal stands down while a number field is being typed
  into: Tab / shift+Tab then walk to the next / previous NUMBER field
  (`keymap::num_tab_pre`, hopping past buttons one per frame;
  `num_tab_tests` pin it). `--test-keymap` drives the loop end to end;
  `cargo test -p app keys_window_render -- --ignored` renders the window,
  text and all, to `target/keys-window-gpu.png`.
- **UI colours are theme slots: a new one is a `theme.rs` row, never a
  literal.** `app/src/theme.rs` declares every colour that is chrome rather
  than data (interface, status, canvases, graph, each node type, timeline,
  patch bay…) with a default that is a literal or a LINK (`like` /
  `shade`) to another slot, so recolouring "background" or "accent" carries
  its family. Sites call the accessor (`theme::cable()`, one atomic load);
  prefs store only overrides as hex; the colours… window (menubar) edits
  them live; `theme::sync` pushes the interface slots into each egui
  context's Visuals. Fades keep the site's alpha
  (`theme::hud_bg().gamma_multiply_u8(160)`), fixed steps off a slot use
  `theme::lift`. A new NodeKind needs a `node_*` slot for `type_color`.
  Deliberately NOT slots: anything that IS data — rgb lanes and tints,
  swatches and picker gradients, plugin manifest colours, user-picked
  region/backdrop colours, cryptomatte ids, band tints — and the viewer's
  image-adaptive HUD ink (`ink.rs` and the lib.rs greys that mirror it).
  With nothing overridden every slot is its old literal, and
  `theme::tests::default_visuals_are_egui_dark` pins egui's look.

- **A node can hold a small node graph of its own, edited IN PLACE
  (`app/src/innards.rs`) — reusable on purpose, roto shapes are next.**
  Zoom a node past twice its zoom-1 size (`SHOW_AT`/`FULL_AT`) and its
  glyph fades into a window onto that graph; once mostly in (`live`) it
  takes a SUBSET of the graph's gestures: drag a node, drag output→input
  to wire, drag a wired input off to unplug/re-plug, click selects, `+`
  adds, alt-click or Delete (over the node) removes the picked node,
  ctrl-drag slices inner wires (the graph's own slice stroke), and a
  kind may answer the M/I/O swipe (`Provider::swipe`, defaulted off). Nothing else — a graph that needs Tab, marquee
  or dots is a DIVE (Noise, 2dini, tools), and stays one. The inner graph
  is DATA on the node, never graph nodes. A kind joins by implementing
  `innards::Provider` (`mini` describes nodes/ports/edges; `move_to`,
  `connect`, `disconnect`, `add`, `remove` apply to its own data — a
  remove must slide later items' indexed curves AND patch cables down,
  `tracker::Points` is the template) and one arm in `innards::provider()`.
  The editor only routes: a press inside a live inner graph becomes
  `DragKind::Innard` after the nav chords and before everything else, so
  the host never drags. `--test-track` drives every gesture through the
  real input path; `innards_render` (ignored) draws it to
  `target/innards-*.png`.
- **A multi-point Tracker's solve is DERIVED, not keyed.** Its inner
  graph is a node per point plus Solve; the points WIRED into Solve, in
  port order, drive it (`solve_inputs`; `solve_in: None` = all, which is
  every old setup and a fresh node; a new point is wired in). Points are
  `TrackX/Y` (point 0, so old setups are unchanged) and `TrackPtX/Y(i)`;
  `TrackPinX/Y(k)` read `TrackerParams::pins()`, which runs
  `engine::pointsolve` (1 point move, 4 corner pin, else similarity) on
  the positions at the evaluated frame against `anchors` (every point at
  the seed frame, in the plate's pixels) — so a hand-corrected key
  corrects the pin, and a cable from them drives a Transform/CornerPin
  through `drive_pins` (shared with the planar tracker). A re-track seeds
  a point from its track only where it has a key AT the playhead; any key
  at all let a stale curve override the box just placed.
