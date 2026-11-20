# Testing: suites, CI, web build

<!-- Moved out of CLAUDE.md to keep it under Claude Code's size limit. Same rules, same weight: read the section for any subsystem you touch. -->

**`--test-seek` was NOT a real deadlock — it was the suite's own budget.**
It used to fail deterministically at stage 5 (`ring free Some(0)`), and this
file used to describe it as hold-and-wait in the repeated-seek path. That
diagnosis was wrong. Stage 5 waits for 24 displayed frames at 24fps — a
whole SECOND — while the timeout was 2400 loop iterations, and under test
(no surface, no vsync) the loop spins ~13k iterations/second, so the budget
elapsed in about 0.2s. Instrumenting the scheduler showed the player
perfectly healthy at the moment it "failed": pending held exactly the ticks
the display wanted. It now times out on the WALL CLOCK and passes 3/3.

**The general trap, because it has now bitten twice:** under test the event
loop is not paced by anything, so *a frame count is not a duration*. Any
suite waiting on real-time behaviour — a transport, playback, a decode —
must time out on an `Instant`, not on iterations. `--test-seek`,
`--test-scrub`, `--test-scrubperf` and `--test-rendercache` all do — the
last three were converted after this bit three separate times.

What each covers, so you know what to re-run:

| suite | covers |
|---|---|
| `click` | node select/deselect, transform gizmo drag, per-pixel ink, ctrl+cmd-click focus, D-drag on the pivot moves the centre with the picture held still (affine unchanged), ctrl-drag on a CORNER handle pins that corner (Transform `pin`, the other three stay), proxy drag 1:1 |
| `box` `slice` `magnet` `tab` | marquee + chain swaps + held-R drag rotating the selection's layout, noodle cut/splice, socket magnetism (INPUTS reach for an output; outputs hold still, so the moving tip is always the input), node search (FUZZY — "grd" reaches Grade by subsequence — with ↑/↓ walking the hits and Enter taking the marked one; with nothing selected the new node arrives unwired; one selected feeds its input 0 AND the new node inherits its consumers — a Tab-add splices into the chain; two selected feed a Blend, first pick as fg) |
| `quickadd` | ctrl+Tab quick create through raw input: over the graph it arms without opening the Tab search, G adds a Grade (unwired, selected), Escape cancels AND is eaten (the selection survives), an unmapped key cancels with nothing added, and with ctrl still held ctrl+B adds a Blur fed by the selected Grade; wall-clock deadline |
| `anim` `curves` `gizmokey` | curve evaluation at the playhead, curve editor, keying from a gizmo |
| `patchlive` | expression links drive values without touching authored data |
| `spline` `paint` `keyer` | spline authoring (S-click add, D-click remove — ctrl/cmd is the layer pick, and the viewer scrub is Tab+drag; marquee + group scale/rotate, D-drag on the box centre moves its pivot, ctrl-drag on a box corner distorts the pick; E/R/T held at a press scale/rotate/move the pick about the press point; O/I held on a point drag its feather outward/inward, moving that point's `bias` so the band's hard side holds; new shapes start at feather 0), feather, shape keys (one keyframe covers every control point), points still grabbable under a shape transform, paint strokes + the brush keys ([ / ] with shift and alt) and the held-F motion gesture, per-frame stroke lifetimes, clone offsets, a dodge stroke carrying mode/stops/shape and its own topology bit, matte output turning the socket into a mask, LumaKey topology |
| `picker` | pixel readout: Z toggles, hover samples the pixel actually under the pointer, shift+Z pins for comparison |
| `colourkey` | colour keyer end to end: sample gesture -> readback -> Oklab fit -> matte, plus the transient preview |
| `zoom` | properties zoom: geometry scales, type does not, cursor anchor holds; then per-panel width: a click on a panel's name selects it, dragging its right edge widens THAT panel by exactly the drag (the next keeps its width), a double-click on the edge hands back the shared width |
| `mblur` | the MotionBlur node: a gain keyed as a triangle UPSTREAM of it averages to exactly 3/4 of its peak across a whole-frame shutter, and the same gain moved DOWNSTREAM of it stays sharp (waits for the new topology to be live before probing) |
| `noise` | the noise authoring loop: dive into a Noise node, build a field graph, step out, render what was authored |
| `geo` | the 2dini authoring loop: dive into a 2dini node, build a noise-driven scatter + instance drawing with an exposed Slider, step out, render it, and drive the Slider through its channel |
| `sdf` | the SDF authoring loop: an SDF node over the plate opens onto Spline -> Fill -> Output with the Spline picked, three S-clicks in the viewer draw a triangle that renders filled white over the plate, `1` over the Spline shows its distance map (violet inside), and out of the dive the comp renders the drawing; pixel reads wait on the wall clock for the picture to LEAVE the previous reading and hold for 400 ms |
| `curveedit` | curve editor gestures: shift-click adds, marquee selects, the selection drags as one group, alt-click removes |
| `browser` `thumbs` `bin` | sequence browser, thumbnails + type filter, recursive search → bin → Reads |
| `cachemeter` `rendercache` `play` `seek` | cache residency reporting, bake/replay, playback rate, scrubbing |
| `frames` `shot` `inout` | the frame-number convention (a placed Read viewed directly: playhead, source frame, in/out loop), the shot's own range, in/out + loop modes |
| `bake` | a GraphBake plays back the live picture frame for frame, a drag lands where it stopped, and the same holds with the plate PLACED at 1011 |
| `range` | noodle frame ranges downstream of a placed plate, the viewer + playhead following the viewed noodle, generators on the script range vs their own, a Read's before/after (black, hold, loop) on real pixels, first/last filled from the files, missing frames black and never cached |
| `dpx` | DPX decode end to end, incl. the log→linear curve |
| `still` | PNG/JPEG decode end to end: a still sequence loads through Read, and code values survive (an sRGB decode in the reader would be invisible otherwise) |
| `grow` | a Read whose range runs PAST its source: the reader's sequence is the range, absent frames do not stall playback, the extrapolated paths are the ones the renderer will write, and a frame written MID-RUN is picked up with no reload |
| `stress` | the crop-into-Layer workflow under load, MEASURED rather than judged (`app/src/stress.rs`; give it a big plate — `gen_exr data/8k_zip_bg 7680 4320 24 zip bg` — and `--cache-dir` for the NVMe tier): one plate cut into `WOBBLE_STRESS_SECTIONS` (4) reformat Crops into a Layer, slipped and stretched (`WOBBLE_STRESS_MEDIA=two`: the sections come from the two plates in turn; `=whole`: the two plates stacked uncropped, a full frame a layer); a cold and a warm pass, a 60Hz scrub, slip / stretch / crop-box edits, proxy both ways, then a Bake of the stack held to the live picture at three ticks and played back; with `WOBBLE_STRESS_STALE=1` (the plate must be a scratch COPY) it writes another frame over the file under the picture and waits for the picture to follow. Prints pictures/s, reader frames uploaded, files really decoded and rig rebuilds per stage; fails only on a wrong bake, a stale picture or a timeout |
| `scrubperf` | how many frames a CONTINUOUS viewer scrub actually puts on screen (measures, and floors at 15fps — the regression it guards is total, not gradual; it does NOT check which frames — `scrubfid` does) |
| `scrubfid` | scrub FIDELITY: per displayed frame, the hand's target against the tick on screen, over eight patterns (60Hz sweeps both ways, 150ms steps both ways, reversals, a stationary hand, relative drags both ways) in three tiers (cold, RAM tier warm, viewer cache warm); fails on a regression, a flip, junk to a still hand or a hijacked drag base, and prints seeks/decodes/exact/latency per pattern — latency in ms AND in redraws (the unpaced test loop redraws thousands of times a second, so ms is not what a vsync display sees); `WOBBLE_FID_TRACE=1` prints per-seek ring pressure and landing lag; runs are noisy while another wobble process holds the GPU |
| `proxy` | the proxy toggle, either way, re-renders the comp at the new size WITH its stage effects and no further input: a Blur is enabled, the output is read back through the viewer transform, and a no-op edit (size += 0.001) must not change the picture; also catches a freshly enabled Blur rendering sharp. Then a Load's own proxy at 0.5, alone and under the comp proxy: the texture shrinks, the canvas does not, and the picture stays in place |
| `scrub` | the browser's preview transport: playing a selected sequence advances the position AND decodes distinct frames behind it (a position that moves over a frozen picture is the failure worth catching) |
| `plugin` | Tier 0 plugin nodes: manifest load, fused compile, keyed params, missing-plugin passthrough |
| `cost` | slow-path colouring: a Blur taints itself and downstream, the Read above stays fast, bypass clears it, and the editor draws from the app's same-frame answer |
| `focus` | the Defocus focus pick: armed, a held pointer lands the node's own Z texel in `focus` (checked against the ToData's output at that pixel), and the focal-plane view then shows that pixel green |
| `keymap` | the hotkey remapper: a rebind is honoured by the live input path (old key inert, new key fires), the keys… window's capture takes a real key event while the hotkeys stand down, same-scope conflicts are flagged, overrides round-trip through the prefs strings; Q fullscreens the graph / properties / image alone and Q again, Esc and W twice (properties off and on) put every pane back in the leaf it started in (the dock tree, not just the flags — a reopened tab once landed in the graph's leaf); a key pressed over the hotkeys window finds its row and does not fire. Waits on the dock's glide are wall-clock |
| `harp` | view-scoped decode on a harp (needs `data/2k_passes`): a select viewed decodes one reader of three; a harp string viewed still decodes one while the rig grows by every string, and shows its select's exact picture (a different string shows a different one); a select on A with a string parked on B decodes two and assigning them rebuilds nothing; flipping to B rebuilds nothing and shows the string |
| `region` | regions + the hidden-wire layer: the view centre activates a region, cmd+` opens the REGION pie and a pick spawns a source chip hidden-wired to its registered source, the hidden edge reaches the engine as an ordinary edge, plain ` opens the GLOBAL pie (add mode registers the selected node; picking one spawns its chip with no region in sight), shift+` flips patch mode |
| `shader` | user-authored WGSL nodes: the starter compiles, a param is a keyed channel and moves NO topology hash, a text edit does, a two-pass shader comes up, a broken one passes through with the comp still rendering |
| `blender` | the Blender node against a stand-in bridge (`blenderui::fake`, a real socket + a real ring): connect, the spec goes out at the comp format, a seek puts Blender's frame ON THE CANVAS under the node's offset (the frame is encoded in the colour), Blender's own scrub moves our playhead, a scrub back over a settled frame is served from the cache and Blender is told not to draw, an edit in Blender empties the cache and the next request draws again; a 3D Camera and Card with a Blender source read its `scene3d` export and follow an edit; every wait is wall-clock. The REAL Blender is `cargo test -p engine --test blender_live -- --ignored --nocapture` (opens a window, ~5s): viewport draw and F12 both come back scene-linear at the spec's size |
| `blendnav` | Blender's nav chords on the real panels: MMB pan, ctrl+MMB drag-zoom, alt+LMB pan starting ON a node, ctrl+alt+LMB zoom, wheel zoom, trackpad pan, ctrl+wheel sideways — graph and viewer; and the viewer's scrub chord, hold Tab + LMB drag (Tab is stripped from egui by the search steal, so its held state comes from the raw events — `keymap::scrub_key_down`) |
| `mswipe` | hold M + drag a node across another: Blend (dragged over, swiped-onto under) takes the under node's place downstream, dragged node goes home, Blend lands at release, spent M opens no popup, a tap does; then I/O swipe a matte into a mask/stencil, and holding B (hold-only, no tap) makes the same Blend |
| `tv` | Read TV thumbnails: shift+H toggles, cabinet-wide hit rect, big tier when zoomed, hover-scrub lands exact decoded frames, off-screen returns to the playhead |
| `nodeux` | node-UX gestures: magnet one-port-per-output rule, S input swap (a tap; held S + drag on a selected backdrop spreads its members, sizes held), hold-`.` dots (a tap alone cuts nothing; press+drag on a wire cuts one and pulls its tension; click places at tension 1; drag on a dot re-pulls) + engine elision (same topology hash), Z/X tap pinning (and shift+Z+X-click soloing a node's panel AND manipulators in one click — both keys held is both gestures), G-grab noodle landing, ⌘G shaped backdrop (member blob; grip scales the members), A-swipe align, T tidy |
| `nodepack` | Grid/Reformat/Premult render fused; a Reformat's format drives the output; a TimeOffset moves the reader's source frame; an fBm generates a non-flat field |
| `track` | BLOB mode follows a disc and keys TrackX/TrackY, and a second pass started mid-sequence keys from the PLAYHEAD onward, retracing the first (needs `data/2k_zip_fg`); then PATTERN mode (`engine::pointtrack`, the default) on the planar suite's plate PLACED at 1001 (the playhead tick is not an index into the frame list), every key within 0.5 px of the truth through a turning, growing plane, no frame lost, and the viewer followed the bake to its last frame; then the MULTI-POINT pass through the real graph gestures (`drive_test_track_ui`): no innards at zoom 1, zoomed in `+` clicked three times and point 3 clicked to select it, four points tracked to the truth, a CornerPin driven from the solve carries them within 0.75 px (judged AT the points — the frame's corners are an extrapolation); waits are wall-clock. `cargo run --release -p engine --example point -- <dir> cx cy hx hy` tracks a point outside the app |
| `planar` | the planar tracker: writes `data/planar_synth` (a textured plane under a known homography) on first run, tracks it forward from frame 0 through the app and checks every frame's corners against the truth (< 0.5 px), then with the mesh on (every vertex on the plane within 1 px) — the viewer must have followed each bake to its last frame — then drives a Transform (a new one, eight pin cables, its pin puts the region on the truth within 0.75 px), then clears and drags a corner handle (that corner follows by the drag, the other three stay); waits are wall-clock. `WOBBLE_PLANAR_FRAMES=240` is the timing mode: a longer shot in its own directory, accuracy checks off, pass durations printed |
| `channels` | Shuffle rewires rgba; a Read decodes a NAMED EXR layer (needs `data/2k_passes`) |
| `crypto` | Cryptomatte end to end: the manifest is read from the header, the fat cable feeds the node's one input and its seven reader slots are minted and decode id-exact, a pick by name mattes the cube, a downstream MatteIn reads it off the node's ONE matte reader (the decode-side bake) and the requested frame lands in the NVMe tier, invert clears it, the ids picture is exact, a plain click on the image picks nothing, cmd-click adds "cube" then "sphere" through the manifest, and cmd+shift-click on the lit (yellow, still id-exact) objects removes them; waits are wall-clock (writes its own ONE-frame plate at `data/crypto_still` via `cryptomatte::write_plate` — the transport runs under test, so the discs must not orbit) |
| `write` | a Write node renders EXRs that read back graded, correctly sized and moving; then (a movie build with `prores_ks`) the same Render as a ProRes `.mov` under proxy, named `_prx`, 3 frames at the canvas size |
| `cpick` | the colour picker on a Grid's line swatch: the swatch opens the popup at the default size (on screen, off its own swatch), the lch L lane puts the knob under the pointer and lands that exact L through the working space, V held before a press drags relatively at V's rate (read from `RATE_KEYS`) without the V hotkey hiding the panel, the old chip restores exactly, the right-edge grip resizes with the edge under the pointer and the swatch side pinned, the size reaches prefs (wall clock) and survives a close + reopen, a pinch scales it without zooming the viewer or graph under it, a double-click on the corner grip restores the default; then the palette: + keeps the colour (into prefs, in Rec.709), a click uses it (alpha kept), shift-click replaces it, alt-click removes it, and the Grid's OTHER swatch offers the same palette and takes its colour; Escape closes; then the viewer's HUD chip: clicking its NAME (the dot is no target) opens the same picker, a palette click lands on `ENode::hud_color`, right-click on the dot clears it. `WOBBLE_TEST_HOLD=1` parks it open for a look |
| `rgb` | the rgb slider on a Mult: a ganged drag stays relative and tracks the pointer, D held slides orange↔cyan at constant luma, G held peels off one lane; plus a Grade in slider mode, including that its rows land two-per-column (an overflowed row is unclickable, not just ugly) |
| `zones` | a Grade's ranges through the REAL panel: the ranges tab, add zones (3), every zone tab, split (4), dissolve (3), the matte, ZONES and HARD previews armed on the view target, the blended map on screen plainly not the graded picture and full of colours (the plate's grain runs through its crossfade), the hard one flat (its zones+1 commonest colours cover 99% of the frame) — read through the viewer after a wall-clock beat (the render is surface-paced; 24 loop iterations read a stale frame) and dumped to `target/zones-preview.png` / `zones-hard.png` — and every control it presses must lie inside its visible clip (`widgets::note_rect`): the preview row once sat 23pt below its tile's fold, pressable by a suite and invisible to a hand |
| `layer` | the Layer stack: a bar drag slides a layer in time, ALT on an end stretches it (the other end holds), a plain end drag TRIMS its in point (footage and rate stay put), a bar sits on its input's own frames and a trim out past a Load's range widens that range (never past the clip), a vertical drag does NOT restack it (bars move left and right only; the graph is where order changes), a ctrl-click on the picture selects a layer (a stack places nothing — upstream Transforms do), the panel pops out into a window, and one plate on two layers decodes into two reader slots |
| `propspie` | the properties pie: `` ` `` held over a row opens it on that row (found through the rows' own registry, `widgets::row_at`), "lock keyframes" locks the row's channels and the same pick unlocks them |
| `setup` | save/load round-trip |
| `roto` | the roto CONTAINER through real input: zoomed in, `+` adds shape 2 and picks it, S-clicks draw it (shape 1 untouched) and leave it open, a plain press-drag adds a point and pulls its tension, a click on the first point closes it, an O swipe inside the node cuts it out of shape 1 and the node lowers to one Subtract MatteCombine, the soft crop boxes what the roto draws, a ctrl-drag cuts the op's wire into Out, Delete over the node takes the picked op; wall-clock deadline |
| `layerpick` | ctrl/cmd-click on the PICTURE picks the layer under the pointer by stacking order AND alpha (a gap in a see-through grid clicks through to the one below) and selects its Transform; ctrl shows the HUD, cmd raises properties, both do both — each SOLOS (selection, pins, rack = that node), shift+chord ADDS |
| `cornerpin` | CornerPin pin picking and group edits: a marquee picks two pins, E held scales them about the press, R held turns them a quarter round it (the others hold), and with `from` armed a pin drag moves the FROM quad and leaves the pin alone, and with X held a 100px drag moves it 10px (the Z/X/C/V rate rides a rated pointer, still anchored to the press), and on the DEFAULT pins (the frame corners) the box knob it presses is on the viewer and turns a pick of three a quarter round the box centre — handles that would land off the viewer or under a floating panel come inside the box (the warp is `from` -> `to`; `CornerPinParams::effective_to` folds it into a plain pin, so no shader change) |
| `gridwarp` | the GridWarp gizmo through real input: a destination point dragged lands under the pointer with its source untouched, source mode moves the source alone, a background box picks the two points it covers and a drag moves the pick as one, the pick's affine box scales four points about the opposite corner, skews them (ctrl on an edge) and turns them a quarter round a pivot dragged onto one of them, and on an ANIMATED point a drag between two keys keys that frame at the curve's value plus the travel (absolute from the press); wall-clock deadline. Pixels are `engine/tests/gridwarp.rs` (the shader's Newton inverse against the CPU mirror) |
| `warp` | the Warp node through real input: three S-clicks draw a curve whose sides start together, a drag on a destination point moves it under the pointer with its source held, a drag inside the box round all three moves every destination by the travel (the shared `pointpick` box), a D-click removes a point; wall-clock deadline. Pixels are `engine/tests/warp.rs` |
| `subwob` | the ` pie's **subwob** on a selected Grade → Blur: both leave, ONE subwob node takes their place fed by the Read and viewed, the picture's mean is unchanged across the collapse (wall-clock settle), undo brings the two back and redo the subwob, and ⏎ opens it onto a PORTAL input node (no input socket, image out) feeding the Grade |
| `compose` | the subwob panel composer on a subwob the suite COLLAPSES from a Grade and fills with four rgb controls, widened to 4 tiles: the grid is exactly its columns wide (it was once a third of it — widgets overflowed a wide empty window and drags only hit the first third); a press anywhere on a control's BODY grabs it, the control follows the pointer PER PIXEL and snaps to the nearest cell only on release; the right bar scales by whole columns; a drop onto another control goes back; then the Grade INSIDE the subwob is raised and its Gain wheel is ctrl-dragged onto an empty cell: it becomes a Colour Wheel control, cabled into the Grade's three gain lanes, at the drop cell |
| `genai` | the GenAI node, offline: Reference -> Output runs with nothing billed (capture by a one-frame bake at the playhead, sRGB PNG, take lands, reader loads) and the picture through the node is the plate within the 8-bit round trip; then Prompt -> FLUX schnell -> Output against an in-process stand-in fal (`genaiui::fake`): `run…` stops at a $0.003 quote with ZERO submits, pay & run submits exactly once, the returned sRGB-128 card reads 0.2159 in the working space, and a re-run of the unchanged pipeline is a cache hit that sends nothing and adds no take. Hermetic (`WOBBLE_GENAI_DIR`/`_CONFIG`/`WOBBLE_FAL_URL` set by the suite); wall-clock waits |
| `flow` | the Flow node end to end on the planar suite's synthetic plate (written to `data/planar_synth` if missing): `analyse` renders the plate, solves it with the REAL network and plays the vectors back — SEA-RAFT first (centre vector within 0.25 px of the truth; 0.02 measured), then switched to RIFE, a new recipe and a new analysis (within 1 px; 0.57 measured); a VectorBlur over the plate along those vectors moves the texture; a half-speed Retime runs 0..94 for 0..47, tick 40 IS source 20 and tick 41 is a made frame unlike either neighbour. Needs `data/rife/flownet.pkl` and `data/searaft/C-T-M.safetensors` (the catalog's files, byte for byte), staged into a private `WOBBLE_MODELS_DIR` so nothing downloads. ~2.5 min; wall-clock waits |
| `upscale` | the Upscale node with the REAL compact network (`data/upscale/realesr-general-x4v3.pth`, served through a private models dir): a 128x68 Reformat of the plate rendered at its own format, upscaled over the range and played at 2x with the picture intact; 4x is a new recipe, STALE until re-run, and the re-run renders the input at its format under the larger loaded canvas; wall-clock waits |
| `seedvr` | the SeedVR2 node against the stand-in tool (`--seedvr-fake`): `run` renders the plate, the process is asked for twice its short side, the result is adopted at twice the plate's format with the picture intact through the Rec.709 round trip, a new seed is STALE; wall-clock waits |
| `stabilize` | the Stabilize node with the REAL SEA-RAFT (`data/searaft/C-T-M.safetensors`, staged into a private models dir; SKIP without it) on a synthetic shaky shot written to `data/stabilize_synth` with its truth: the path within 0.75 px, lock-off through the node's pin at every frame < 0.5 px, nine patches of the stabilised picture still where the plate's move, a reversed second Stabilize puts the plate back (pins compose to identity), and a matte wired into port 1 renders, re-keys the run and keeps the path; wall-clock waits |
| `depth` | the Depth node against the stand-in tool (`--depth-fake`): unrun it outputs 0, `run` renders the plate, runs the process and adopts `Depth/`; the data output derives the fake's depth, disparity and near from its ONE stored value and `meta.json` through the uniform pick with no new directory, a Render fed it writes one exact float32 `Z` channel with no colour tag (read back), it can feed a Defocus's Z port, and a model change is STALE; wall-clock waits |
| `ltxhdr` | the LTX HDR node against the stand-in tool (`--ltxhdr-fake`, result = plate x 4): unrun it passes the plate, `run` renders it as ACEScg, runs the process and adopts `HDR/`; a clip under the plate composites the result, a clip over it shows the plate, raw shows the result, none of the three a new directory; tagged Restricted and carrying the licence taint; a new seed is STALE; wall-clock waits |
| `difflight` | the DiffusionLight node against the stand-in tool (`--difflight-fake`, a flat 512x256 latlong): unrun it is black (never the plate), `run` renders the plate, runs the process and adopts the latlong at the LATLONG's size, untagged; turning `follow` off keeps the estimate directory, changes the result's name and shows STALE, and the re-run estimates nothing again (the stand-in's `Raw/` markers) and plays the new pass; a new seed is a new estimate directory; picking LuxDiT is a new estimate directory and tags the node non-commercial; wall-clock waits |
| `hdrcnn` | the HDRCNN node with the REAL network (`data/hdrcnn/hdrcnn_params_compr_regularized.npz`, staged in a private models dir): a 128x68 Reformat of the plate; clip 1000 renders a result that IS the plate (the blend leaves it alone), a new clip is STALE, clip at half the probe's level re-runs and the reconstruction is neither the plate nor darker than the clipped picture; wall-clock waits |
| `corridor` | the CorridorKey node against the stand-in tool (`--corridor-fake`, the binary itself): `key` renders the plate (linear Rec.709) and a coarse ScreenKey hint to the shot dir, runs the process, adopts `Processed/`; premultiplied output = plate × matte at the probe (the Rec.709 round trip), the matte output is the matte; a Gradient wired into the hint re-keys into a NEW directory and is the matte on screen (left < right), premult following it; wall-clock waits |
| `sammie` | the Sammie node against the stand-in tool (`--sammie-fake`, the binary itself: a disc round each include point) through the REAL viewer input: S-click lands an include point where clicked on the frame on screen, shift+S-click an exclude, D-click on the exclude removes it; `track` renders the plate, runs the process, adopts `Matte/`; the cutout is the plate inside the segment (same frame as read before) and black outside, the matte output is 1 on the MASK pipe, and a new click marks the loaded matte stale without starting a run; before tracking, `preview` puts the frame's segment on the viewer (coverage = one disc, not adopted) and a second S-click re-runs it by itself (two discs). Gesture steps are 120ms apart so press/release and key/click never fold into one UI pass; wall-clock waits |
| `text` | the Text node through the app: a Text over the plate draws its four glyphs solid in its colour where its layout says; an animator over the first half lifts exactly glyphs 0-1 by its offset with no recompile; a key on `text.a0.offset` moves the selection with the playhead (off the end at 10, back at 5); two S-clicks in the viewer lay a path the glyphs then run down; wall-clock waits |
| `eyedrop` | E over a colour swatch arms the eyedropper; a plain drag follows the pointer live, a shift-drag averages a box; a viewer click lands the comp's scene-linear value under it (matching the pixel readout there), alpha untouched; E + Esc arms and stands down |
| `viewlock` | viewer 1 locked (alt+1) on a node DOWNSTREAM of a subwob stays on it when you dive in, and the picture follows the body's live edit (inner gain halved → locked tail halves); stepping out keeps the lock |
| `dual` | dual view renders BOTH panes live: left the plate, right a Grade of it at 0.5 through the second compositor; with the left live, setting the Grade to 0.25 moves the right pane to a quarter (a held still would not) |
| `record` | the screen recorder: ⌘⇧R through the real keymap starts a take of the window, a pointer circles the viewer while the comp plays, ⌘⇧R stops it, at least 45 frames came back off the GPU, and the take saves to an MP4 (the OS encoder) and a GIF. Needs a window that DRAWS: `WOBBLE_TEST_FOCUS=1` (an accessory window behind others produces no surface frames); `WOBBLE_RECORD_DIR` is where the clip lands, `WOBBLE_RECORD_TEST_SECS` how long it records (2.5) |

### Adding a suite

A flag has to be registered in **seven** places in `lib.rs` — miss one and
it fails in a way that looks like anything but a missing registration:
`Options` field, its initialiser, the `--test-x` arm of the arg parser,
`any_test()`, the `x` state field in `SuiteState` (`app/src/state/test.rs`,
reached as `self.test.x`), its initialiser in `build_gfx`, and the
`drive_test_x()` call in the frame loop. Add it to the `options_tests` table
too, which is what catches the `any_test()` omission.

The driver itself is NOT in `lib.rs`: `fn drive_test_x` lives in
`app/src/suites/x.rs` (an `impl Gfx` block, `use crate::*;`), with one
`mod x;` line in `app/src/suites/mod.rs`. Helpers a suite shares with the
app stay where they are.

### Rules for suites

- **A deadlock cannot fail its own suite.** The wall-clock budgets are
  checked by the loop that is stuck, so a hang used to read as a suite
  running forever with no output. `spawn_test_watchdog` watches the UI
  passes from another thread: none for 45s prints `WATCHDOG TEST FAIL` and
  exits. On a live hang, `sample <pid> 3` names the lock in one go.
- **Aim at a control through `widgets::note_rect`, and check its clip.** It
  records the rect AND the clip it was drawn under. A control pushed past
  its tile's edge still has a rect, so a suite that reads rects alone
  presses what no hand can see; `--test-zones` fails on it.

- **Every `--test-*` flag must be in `Options::any_test()`.** That is what
  makes a run hermetic. A unit test (`options_tests`) enforces it because
  the symptom otherwise is baffling: a suite silently reads the user's real
  prefs, and e.g. `hud_follow: false` hides the manipulators it clicks.
- Suites run on `Prefs::default()` and **never save prefs** (`TEST_RUN`
  guard in `save_prefs`). Do not bypass this — earlier versions overwrote
  the user's real settings with defaults.
- Suites run as a macOS *accessory* so they never steal focus.
  `WOBBLE_TEST_FOCUS=1` opts back into a normal window. Never use
  `with_visible(false)`: a hidden window stalls the surface.
- Under test the egui pass runs **without a surface** (occluded windows stop
  producing frames), so texture deltas must still be applied.
- Synthetic gestures must anchor to a **settled** layout: capture the rect
  once (`layout_settled()`, `test_rect`) and reuse it for the whole gesture.
  Panels animate, and a press computed from a moving rect misses its handle.
- Adding panel chrome shifts viewport origins and can push a hard-coded test
  coordinate off-screen. Place test nodes relative to existing ones.
- **`raw.modifiers` is per FRAME.** A synthetic chord (shift+Z+X-click, say)
  must set it on every frame of the gesture — the release too, because a
  click resolves on the release. `--test-nodeux`'s Z+X stage held shift
  only on the press and measured "both pinned, nothing cleared".
- Gate on **state, not throughput**: assert "residency reached N" rather than
  "N frames displayed". Display rate varies with machine load, and CI or a
  busy laptop will fail an otherwise correct test.
- Never assume POSIX path separators in assertions — build expectations with
  `Path::join`. (Windows CI caught exactly this in a Blender arg test.)

### CI

[.github/workflows/ci.yml](.github/workflows/ci.yml) builds and unit-tests on
**Linux, macOS and Windows**, uploads a binary per platform, keeps the
`opendrt` feature compiling (check only — GPLv3, no artifact), and runs
clippy as **advisory** (the bar is zero rustc warnings, not zero clippy
lints). `RUSTFLAGS: -D warnings` on the build job holds that bar.

The UI suites are not in CI: they need a real window and GPU. Run them
locally before pushing. Before changing anything platform-specific, a cheap
local check of the other targets:

```bash
rustup target add x86_64-pc-windows-msvc x86_64-unknown-linux-gnu
cargo check --workspace --target x86_64-pc-windows-msvc
cargo check --workspace --target x86_64-unknown-linux-gnu
```

Caveat: the workspace now carries C deps (`ring` via ureq, `ufbx`), and this
Mac has no cross C toolchains, so those two targets fail in their BUILD
SCRIPTS before any Rust is judged. Real CI compiles natively per-OS and both
libs support MSVC/gcc/clang; the cross-check above still catches Rust-side
breakage if you let it run past the C errors (`--keep-going`), but the real
verdict for C is CI's.

### Web build (`web/`)

The WHOLE app runs in a browser on WebGPU: `web/build-app.sh` builds
`wobble` for `wasm32-unknown-unknown` into `web/www/app/`; serve `web/www`
(`python3 -m http.server 8765 --directory web/www`) and open `/app/`.
(`web/build.sh` + `web/www/index.html` is the older engine-only page.)
Browser glue lives in `app/src/webmain.rs`; every other change is a
`cfg(target_arch = "wasm32")` gate, so native builds are unchanged. A
browser breaks five things silently, and each has ONE answer — use it, or
the page dies with a panic the desktop never shows:

- **No threads.** `std::thread::spawn` panics. A job that ENDS goes
  through `crate::bg::spawn` (a thread on native, rayon's one-thread pool
  in a page, drained by `webmain::pump_rayon` each frame). A loop waiting
  on a channel must not run there at all (see the browser's header worker).
  `Builder::spawn(..).ok()` merely fails, which is fine.
- **No blocking.** `pollster::block_on`, `PollType::Wait` + a mapped read
  hang or panic: the device comes from `GpuContext::new_async`, the compile
  validates through naga (`web_validate`), bakes are refused.
- **No `std::time`.** `Instant::now` / `SystemTime::now` panic; use
  `web_time::` (a plain re-export of std's on native). File mtimes stay
  `std::time::SystemTime` — that is the type `fs::Metadata` returns.
- **No filesystem.** Decoders read through `engine::vfs` (`open`, `read`,
  `exists`, `list`) — the filesystem natively, the page's in-memory table
  in a browser, filled by drops and the file picker (`wobble_put_file` /
  `wobble_open`). `std::env::temp_dir()` PANICS there.
- **No ring/ureq, no arboard, no rfd.** HTTP, the system clipboard and file
  dialogs are native-only deps; `crate::filedialog::FileDialog` is the
  dialog (picks nothing in a page, whose picker is `webmain::pick`).

Check a change with
`cargo check -p app --target wasm32-unknown-unknown` — compile errors are
the cheap half; the panics above only show in the browser console.

## The LIVE GenAI run (real fal, real cents)

`WOBBLE_TEST_GENAI_LIVE=1 wobble <clip> <clip> --test-genai` (a movie
build; a plate of 2 s or more, e.g. `data/hcr080`) is `--test-genai`'s
opt-in live branch (`genaiui::drive_test_genai_live`): the user's own fal
key, about $0.21 a run, refusing any quote over `LIVE_CAP_USD`. A plate
into a GenAI Pipeline; Wan 2.7 video edit run from its own node inside on
two seconds of it; FLUX schnell paid and run BESIDE it; the pipeline's own
run landing the video as a take without paying twice; the take loaded and
read by a node downstream. It logs every run's state as it changes and
keeps its results in `$TMPDIR/wobble-test-genai-live`. `=dry` pays
nothing: it captures the clip twice and checks both key the same.
It found, the first time it ran: clips one frame short (the mp4's last
packet had no duration), and a clip keyed on its encoded bytes (never the
same twice), which re-billed every re-capture.
