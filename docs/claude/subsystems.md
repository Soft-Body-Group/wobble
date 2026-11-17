# Subsystem notes

<!-- Moved out of CLAUDE.md to keep it under Claude Code's size limit. Same rules, same weight: read the section for any subsystem you touch. -->

Each section is what must stay true in one subsystem. Ctrl-F the file or node name.

## Cost + licence taint (`cost.rs` / `costviz.rs`)

Manifesto rules 8 and 10 in code. [engine/src/cost.rs](engine/src/cost.rs)
classifies one node; [app/src/costviz.rs](app/src/costviz.rs) propagates it
downstream and picks which colour a wire wears; the colours themselves are
the `cost_*` slots in `theme.rs`. Four things not to break:

- **"Slow" means "codegen gives it a stage", and there is a test that says
  so — with ONE listed exception.** A node on `StageDef` cannot get this wrong: `kind_of` and
  `speed_of` both read `NodeKind::stage_def()`, so declaring the chain IS
  declaring the colour. `cost::tests::slow_matches_codegen` builds a one-node graph for every
  variant and asserts `speed_of` agrees with whether `generate()` produced a
  filter stage. Add a stage kind without coming back here and it fails —
  which is the point, because the alternative is a node that renders red
  everywhere except where it matters. The exception is a cost the stages
  cannot see: a **DECODE** or a **PROCESS**. An unbaked Cryptomatte pulls
  six rank layers out of the EXR every frame and takes no pass at all; a
  Blender node is a round trip to another process a frame (heavy when
  that trip is a real render). Both are listed in
  `cost::tests::decode_cost` — a hand-kept list of exactly those two,
  that the same test walks rather than waving through.
- **`Speed::Heavy` is "really slow": a pass CHAIN that dominates the frame.**
  Defocus in accurate mode, a paired Loop
  at `HEAVY_LOOP` iterations or more, a Shader of three passes or more, and
  a Cryptomatte until its matte is BAKED, and a MotionBlur from
  `HEAVY_SAMPLES` (its whole upstream, rendered once per sample).
  It is a kind of slow (`heavy ⊆ slow` in `costviz::Flow`, same bake clears
  both) with one extra tell: it GLOWS — a red-hot ring, a halo breathing
  around the node and light along every wire below it. Classify by what the
  compositor actually dispatches, not by how a node feels;
  `cost::tests::heavy_is_the_pass_chains_that_dominate_a_frame` pins the set.
  **The glow is additive LIGHT, not paint** (`editor::glow_light` /
  `glow_ring` / `glow_ribbon`): a premultiplied colour with zero alpha ADDS
  to what is under it, and the falloff is a gradient mesh — stacked
  translucent strokes band, and banding reads as rings rather than as light.
  Three pixel tests in `editor::cost_look_tests` measure it (halo outshines a
  merely slow node, falls off smoothly, lit wires beat plain ones), because
  "it glows" is the claim a stack of alpha tweaks quietly loses.
- **The two taints clear differently.** Serving from disk clears SLOW — a
  loaded Bake (the branch above is not evaluated) or a Cryptomatte whose
  matte is in the NVMe tier (the rank decodes stop), which is why
  `clears_slow` takes `is_loaded` and the app answers it for both
  (`crypto_baked`, recomputed at the top of `ui()` from `bake_ui`'s
  throttled count; ALL frames, because a half-baked matte still costs the
  decodes on the frames it does not have). Nothing clears NON-COMMERCIAL
  (the pixels were still made that way). Conflating them is the one bug here that could cost
  somebody a delivery, so `taint()` applies exactly one clearing predicate,
  to one of them, and two tests pin the directions.
- **The flow is computed once, at the TOP of `ui()`** — before the editor,
  the panels or the menubar draw. It used to sit in `redraw()`, which runs on
  its own schedule, and the editor was drawing last frame's answer. That is
  invisible in a screenshot; `--test-cost` asserts the editor's flow equals
  the app's flow in the same frame, and it caught exactly that.
- **Licence is derived from the model registry, never a list here.**
  `ModelSpec.commercial` is the source of truth, read for the exact ids a
  node declares in `NodeKind::models()` — per instance, so a node whose
  params pick a research-only option is tagged only while it is picked.

## Defocus (`codegen::DEFOCUS_*`, `compositor::defocus_bind_groups`)

Still hand-built in the compositor (the next stage to move to `StageDef`).
With depth wired every frame runs FOUR map passes before the gather — the
CoC map (`DEFOCUS_COC_WGSL`: signed CoC + Z, `edge` = nearest Z within n
px), an 8x8 tile max, and a separable tile dilation by the max radius —
and every later pass reads `cmap`/`tiles` rather than the depth texture.
Without depth none of it runs and the map bindings take the 1x1 dummies
(`dummy_rg32`/`_rgba32`/`_r32`, in the storage formats the bindings
declare — a dummy of the wrong format is a validation error). Fast mode
is two layers off THREE prefix textures (plain, near-weighted, near
weight; `near_w` is the one definition of "near": blurry AND in front);
`engine/tests/defocus.rs` pins the bleed-out, the far fill, the edge
option, Accurate's per-pixel search radius and the clamp. The DP uniform
is 80 bytes (5 vec4) — `make_uniform(.., 80, "stage-params")`.
The MODULATE matte (port 4, `codegen::DEFOCUS_MOD_INPUT`, the aux after
depth) scales the CoC in the map pass, so a wired matte builds the maps
even with no depth: `has_depth` means "the map exists", `has_z` /
`has_mod` (the old pad lanes) say which made it, and without Z the map
holds `radius` × matte at the focus distance. Focal-plane view gates on
`has_z`.

## Shader nodes (user-authored WGSL)

`NodeKind::Shader` is text the user types into the app; the dialect,
codegen, naga validation and uniform packing all live in
[engine/src/usershader.rs](engine/src/usershader.rs), the UI in
[app/src/shaderui.rs](app/src/shaderui.rs). README has the user-facing
reference. The invariants that are easy to break from outside:

- **It is not a plugin.** A Tier 0 plugin is one expression fused into the
  main pass. A Shader node is a filter STAGE: each image input resolves to
  its own texture (input 0 through the stage subtree pass, inputs 1.. through
  aux passes) and then N user passes run, each writing a full-canvas result
  later passes may sample. That is what buys arbitrary-coordinate sampling,
  and it is why the node is marked a slow path in its panel.
- **Params are uniforms; text is topology.** `topo_tag` hashes the SOURCE, so
  editing the text recompiles and invalidates render caches; moving a param
  must never do either. There is a test for exactly this.
- **Validation happens before topology.** `usershader::compile` runs naga and
  memoizes on the source hash. A shader with errors gets NO stage and `emit`
  falls back to input 0 — the comp keeps rendering while the user types. Do
  not "fix" this by letting a broken shader reach `compile_plan`; one typo
  would take the whole plan down.
- **Ports come from the pragmas, not the compile.** `input_count` reads the
  parse so a shader that is mid-edit keeps its wires.
- **All passes share one module and one bind group layout.** Pass i binds
  every input and every result texture, with its OWN result slot pointed at
  the dummy (a texture cannot be sampled and written by the same pass), and
  selects itself by entry point (`wob_pass{i}`). Keep the binding count
  constant across passes or the layout stops being shareable and the module
  gets compiled N times — which on Metal is ~100-150ms each.
- **`shaderui::make_controls` is the bridge to the tool composer.** It spawns
  control nodes UNPLACED and hands them to `tools::place_controls` rather
  than positioning them itself — the composer owns panel layout, and a second
  placer would drift from it. It is idempotent by checking
  `patch.link_to(PDst::Chan(node, chan))`, so it tops up after a param is
  added instead of duplicating.
- **Uniform layout is one 16-byte cell per member**, `@align(16)` on all of
  them, because the struct's shape is authored at runtime and the WGSL
  layout rules are not worth getting subtly wrong to save 12 bytes.
  `BUILTIN_CELLS` must match the prelude's built-in list.

## Upscale + `nn` (`engine/src/nn/`, `engine/src/upscale.rs`, `app/src/upscaleui.rs`)

Neural super-resolution (Real-ESRGAN compact / x4plus, SPAN) in our own
WGSL, **click to render** on the Flow precedent: `upscaleui` renders the
input (a `SubtreeBake`, `BakeSink::Upscale`), `engine::upscale::run` upscales
the frames on a worker, the node plays `<dir>/Upscaled` through its reader
slot. `nn::Net` is the network as DATA (feature buffers, convs over channel
SLICES, an epilogue); `upscale.rs` builds one per architecture from the
upstream `.pth` (read by `torchpt`, the restricted pickle reader);
`nn::gpu::Runner` runs it tile by tile. README has the user story. What
must stay true:

- **It never runs on an edit** (by design: at 1-60 s a frame, live froze
  the app). Reuse the Flow runner's types (`flowui::{Run, Phase, View,
  Action}`), not a third set; `Gfx::render_input_frames` is the shared
  input render (SeedVR2 uses it too).
- **The input is rendered at ITS format, not the canvas's.** A loaded
  result makes the node's canvas larger than its input, and a re-run at the
  canvas size would upscale a mostly-black frame. The bake loop grows the
  output to hold a render larger than the viewer's canvas (the pixels are
  1:1, so the frame is the top-left), and the next redraw puts it back.
- **`nn::reference` is the spec.** A new op lands there first, then in
  `nn::wgsl`; `engine/tests/upscale.rs` holds the CPU nets against torch
  (`upscale_ref.rs`, made ONCE by `tools/upscale_reference.py` through
  spandrel, ~1e-5) and the GPU against both (f32 and f16, ~5e-3), and runs
  `upscale::run` on EXRs. The tests need the weights (`WOBBLE_MODELS_DIR`)
  and skip, saying so, without them; `--test-upscale` needs
  `data/upscale/realesr-general-x4v3.pth`, byte for byte (the Flow
  precedent: a private models dir, nothing downloaded).
- **SPAN's SiLU is `inplace=True` upstream.** Block 6's "pre-activation"
  output that conv_cat reads is therefore ACTIVATED. Building it from the
  forward() as written is off by 0.15 and the mean hides it.
- **One arena, bound read-write once.** Every feature buffer of a tile is a
  range of one storage buffer: a dense block reads channels 0..k and
  appends k..k+32 of the SAME buffer, and two bindings of one buffer with
  one writable is a usage conflict wgpu refuses.
- **The conv kernel's shape is measured, not guessed** (`wgsl::conv_body`
  lists it): a 2x2 quad per thread, one output group's weights live at a
  time. Holding a whole tap's weights spilled (2% of the ALU); row/column
  runs per thread lost to bank conflicts. **Benchmark best-of-20**: the GPU
  clocks up over the first runs and a remote-desktop session shares it,
  so a single run reads 2-5x slow (`examples/nn_micro`, `upscale_bench`).
  f16 is STORAGE only — f16 arithmetic is no faster on Apple GPUs.
- **The halo is the receptive field** (`Net::receptive`, capped at
  `MAX_HALO`), so a compact net's tiles join EXACTLY; tiles all have one
  size (the last row/column slides in), so a conv's uniform is per op, not
  per tile.
- **No bounded tonemap on the way in.** Its inverse turns the model's
  routine overshoot near 1 into an unbounded value (0.87 came back as
  400). The model gets the picture CLAMPED to 0..1 sRGB; the excess
  (highlights, negatives) rides around it bilinearly (`nn::wgsl::emit`).
  The SeedVR2 driver does the same in numpy.
- **`Runner::encode` may SUBMIT.** Past `SUBMIT_MACS` it ends the caller's
  encoder, submits it and hands back a fresh one: Windows resets a GPU
  whose submission runs ~2 s (TDR), and an RRDB frame is a minute on a
  laptop.

## SeedVR2 node (`engine/src/nodes/seedvr.rs`, `app/src/seedvr.rs`, `tools/wobble_seedvr.py`)

ByteDance-Seed's diffusion restorer as a process-backed reader, the
CorridorKey pattern (its `spawn`/`Msg`/`uv` plumbing, Upscale's input
render). The tool is the community numz CLI at a PINNED commit
(`seedvr::COMMIT`): the driver calls its `parse_arguments` and
`_process_frames_core` directly, which a later commit may rename — bump the
pin and re-test, never float it. The driver picks the DiT file by device
(fp16 on MPS, fp8 on CUDA under 20 GB) and processes whole batches at a
time (SeedVR2 is consistent within a batch and independent across, so a
chunk on a batch boundary changes nothing), with the VAE tiled (same
speed, 28 GB peak instead of 50 at 1080p out). Measured on an M4 Pro: 3B
runs on MPS, ~32 s a 1080p frame. `--test-seedvr` drives it against
`--seedvr-fake` (pixel replication at the asked resolution); the real tool
needs `install…` and its weights, which no suite fetches.

## Depth node (`engine/src/nodes/depth.rs`, `app/src/depth.rs`, `tools/wobble_depth.py`)

Video Depth Anything as a process-backed reader, the SeedVR2 pattern
(CorridorKey's `spawn`/`Msg`/`uv`, Upscale's `render_input_frames`). README
has the user story. What must stay true:

- **Weights are the catalog's, never the tool's.** Four `vda-*` rows,
  pinned by HF commit and sha256; `NodeKind::models()` declares the picked
  one, so the gate, badge, prompt and licence tag come for free. The app
  passes `ready_path` to the driver as `--weights`;
  `the_driver_never_downloads` scans it. Only the stand-in
  (`$WOBBLE_DEPTH_CMD`) may run without them.
- **Licences, read off the model cards**: small and small metric
  Apache-2.0, large and large metric CC-BY-NC-4.0 (`commercial: false`).
  `cost::tests::the_non_commercial_models_are_listed_and_their_nodes_tagged`
  replaced the old "everything is permissive" tripwire and pins that a
  loaded result clears slow but never the tag.
- **The pin is a commit AND a pyproject.** Upstream has no pyproject and
  pins torch 2.1 + xformers (no MPS autocast, no macOS wheels); `install…`
  writes `depth::PYPROJECT` into the checkout; `installed()` is false
  while the checkout's pyproject differs from it, so an update brings
  `install…` back. **`opencv-python<5`**: 5.x's Windows wheel has no EXR
  codec (a Windows run: every `imread` None), so EXR I/O is the
  OpenEXR package, cv2 only the fallback. SeedVR2's unpinned upstream
  pyproject gets the same pin appended (`seedvr::CONSTRAINTS`);
  CorridorKey and Sammie install from upstream locks on 4.x.
- **The driver re-does `infer_video_depth` window by window**, with the
  tool's own constants and `compute_scale_and_shift` /
  `get_interpolate_frames`: the tool's call holds every frame at full size
  and every result at the plate's (~50 GB for 700 4K frames). Inputs are
  shrunk to the network's size on read, aligned results wait on disk at
  that size (`depth_raw.npy`), and are scaled up as written. Same pixels to
  0.02% on hcr080; RSS flat with clip length (3.7 GB at 50 and 100). An
  unreadable Input frame is REMOVED (a half-written render would else be
  skipped by the resume forever) and the error names it, its size and
  both readers' reasons. Bump `COMMIT` and re-test, never float it.
- **GPU memory: SDPA, a cap, and a step down.** Without xformers (no Mac
  wheels; not in our pyproject) DINOv2 materialises its attention matrix,
  and Large at a 4K plate's 490x924 input peaked at 45 GB on MPS (on a
  4090: 21 GB VRAM + 11 GB silently spilled to shared memory, 0 frames in
  9 min). The driver swaps in `sdpa_forward` (the same maths through
  `F.scaled_dot_product_attention`): 19 GB on MPS, half the time, small
  6.2 -> 4.2 GB at 518. Chunking the ENCODER does not help (the 32-frame
  temporal head is the rest). On CUDA `set_per_process_memory_fraction(0.9)`
  makes an overflow an OOM instead of a spill, and an OOM restarts the clip
  at 0.8x the input size (logged), down to 140, then a named error.
  Progress: phase lines (loading, window 1 read/infer/time + GPU memory)
  before the first `WOBBLE frame`, so 0/N is never a silent wait.
- **The cache is ONE half-float channel `Z` plus `Depth/meta.json`**
  (`depth::LAYOUT` 2: `top` = the disparity's 99.9th-percentile clip
  maximum, `metric`). The network's own output is stored (disparity, or
  metres); `DepthParams::emit_loaded` derives depth / disparity / near from
  it with `top` + `metric` on the node's slot (transient `top` /
  `metric_take`, set where `loaded` is, from `Jobs::stats`, which
  `depth_load` fills from `meta.json`), so the pick is a UNIFORM and not in
  the recipe; `DepthParams::derive` is the CPU mirror the tests use.
  Relative depth = 1 / (disparity / top), capped at `FAR`. `LAYOUT` is in
  the directory key, so a layout-1 result (three float32 channels, 74 MB a
  4K frame) is never adopted: it shows STALE and a run makes a new one.
  Half, not float32: the reader decodes to half whatever the file holds.
- **A Render takes the Data pipe** (`NodeKind::accepts` →
  `writes_data`, NOT `can_feed`: data into a colour node would still be
  read as a picture) and writes one float32 `Z` channel per frame, no
  colour transform, no chromaticities (`BakeSink::Write { data }`,
  `bakes::write_z_f32_exr`). The image path is untouched and its bytes are
  pinned (`bakes::tests::the_image_writer_is_byte_identical`).
  `--test-depth` renders the fake's result through one and reads it back.
- **Output is the Data pipe** (`output_kind`), broadcast to rgb, alpha 1;
  no input transform. Unloaded it is 0 and does no work. Loaded it is
  `inputs_unsampled` and `clears_slow`.
- `--test-depth` runs against `--depth-fake` (`FAKE_RAW` + `FAKE_TOP`: a
  distinct number per output). The real check is the driver on its own:
  `uv run --project ~/.wobble/VideoDepthAnything python wobble_depth.py
  --shot <dir with Input/> --weights <pth> --encoder vits`.

## HDRCNN node (`engine/src/hdrcnn.rs`, `engine/src/nodes/hdrcnn.rs`, `app/src/hdrcnnui.rs`)

Eilertsen et al.'s HDRCNN on the `nn` runtime, click to render on the
Upscale pattern. What must stay true:

- **`nn` grew an encoder-decoder vocabulary**: `Buf::div` (a buffer below
  the tile's size), `Op::Pool` (2x2 max), `Op::Deconv` (4x4 stride 2, TF
  `SAME`: output `o` takes input `i` through tap `o - 2i + 1`),
  `Op::LogSkip`, and `Io::Hdr` (gamma-2 load over the clip, the method's
  blend at emit). A net with `div > 1` is cut on its grid (`Net::align`):
  tile size, origins and halo are multiples of it and the picture is padded
  to one (`LoadU::lim` repeats the edge). Reference first, then WGSL, as
  for every op.
- **f32 storage, always**: the bottleneck's BatchNorm stats say its input
  is ~1e9. `Kernels::new(device, false)`.
- **Weights are read by POSITION** from TensorLayer's `params.npy` (80
  tensors, conv `W, b`, BatchNorm `beta, gamma, mean, var`), each shape
  checked. The npz's arrays are pickled as `_codecs.encode(text, "latin1")`
  (Python 3, protocol 2): `npy.rs` allows exactly that callable and that
  encoding, nothing else from `_codecs`.
- **Pixels the mask leaves alone are stored as they came** (`hdr()` in
  `nn::wgsl::emit`, textureLoad not a filtered sample);
  `engine/tests/hdrcnn.rs` asserts it bit for bit.
- `engine/tests/hdrcnn_ref.rs` is made ONCE by `tools/hdrcnn_reference.py`:
  the upstream graph re-written in NumPy (TF1 does not install), run on the
  same npz files. CPU net ~3e-6 off it, GPU ~5e-3 (half-float output).
  Tiles vs the whole frame at `HDR_HALO` 96: ~3% at worst inside a
  highlight (the receptive field is hundreds of px; the cap is a
  measurement, like Upscale's `MAX_HALO`). Tests need the weights
  (`WOBBLE_MODELS_DIR`); `--test-hdrcnn` needs `data/hdrcnn/` (the catalog's
  files, byte for byte).
- The plate goes in linear Rec.709 (`BakeSink::HdrCnn`), divided by `clip`,
  clamped, `sqrt`-encoded: the net linearises with `x²`, so its blend's
  `x²` IS the plate. A clip far under the whole plate gives the net flat
  white and it predicts ~1.5x the clip: that is the method, not a bug.

## LTX HDR node (`engine/src/nodes/ltxhdr.rs`, `app/src/ltxhdr.rs`, `tools/wobble_ltxhdr.py`)

LTX-2.5's SDR-to-HDR IC-LoRA as a process-backed reader, the Depth
pattern. README has the user story. What must stay true:

- **Composite, never replace, by default.** The VAE round trip shifts
  every pixel; the loaded node is `mix(input, result, smoothstep(knee,
  clip, max(rgb)))`, so an unclipped pixel is the plate. Clip/softness are
  uniforms, `raw` is topology (it stops sampling the plate). Only a raw
  loaded result is `inputs_unsampled` and `clears_slow` (the Sammie
  cutout/matte precedent); composited, the plate under it still renders.
- **The node's format is the INPUT's** (it is left out of
  `node_resolution`'s loaded-size list): a proxy result is smaller than
  the plate and is stretched over the format in the emit.
- **Licence is `Licence::Restricted`, always.** The LTX-2 Community
  Licence is not research-only but is not plain permissive either; it
  warns exactly like NC (`Licence::warns`, which every "is it NC?" check
  now asks) and nothing clears it. The Write's words say which.
- **Weights are the tool's, fetched under the user's own HF login**
  (gated repos, so not catalog rows): `--fetch` is the ONLY network code
  in the driver (`only_the_fetch_mode_touches_the_network`), pinned to two
  HF commits plus the LTX-2 `COMMIT`. Bump all three together and re-test.
- **The driver builds the pipeline the way `hdr_ic_lora.main()` does and
  writes the EXRs itself** (`decode_hdr_video` + `save_exr_tensor`), not
  through the HLG mp4 encoder: a missing x265 there would lose the EXRs.
  Frames pad to 8k+1 by holding the last (`LtxHdrParams::padded_frames`,
  the driver's copy is checked) and trim back.
- `--test-ltxhdr` runs against `--ltxhdr-fake` (result = input x
  `FAKE_GAIN`). The driver was run against the pinned checkout up to the
  weight load (imports, the tool's own arg parser, proxy downscale, 8k+1
  padding, the gated-fetch error); never with real weights.

## Planar tracker (`engine/src/planar.rs`, `app/src/planarui.rs`)

Mocha-style planar + mesh tracking, CPU, research stage. README has the
user-facing story. The things that cost time, so nobody pays twice:

- **Never subsample a level on a lattice.** The finest level once took
  every other pixel to cap the cost, and the objective's optimum moved by
  0.5 px — the lattice aliased the texture, and the fit preferred the
  wrong answer (`probe_scale` shows it level by level). Levels are dense up
  to `MAX_LEVEL_PIXELS` and RANDOMLY subsampled past it.
- **Robust weights are a second phase on a BLURRED residual, and both
  sides are normalised over the weighted set.** Per-pixel Tukey from the
  first iteration rejected the misaligned edges (the position signal) and
  bent clean fits by 0.4 px; a per-pixel test cannot tell an occluder (a
  blob) from a misaligned edge (a line), the box blur can. Phase 1
  converges unweighted, phase 2 reweights; finer levels start weighted so
  an occluder cannot pull a converged coarse fit off again. And the gain +
  bias the residual is judged against come from the WEIGHTED pixels — with
  the occluder in the statistics the fit sat 0.5 px off with every weight
  right (2.2 px unweighted, 0.035 px now). `probe_robust` prints all of it.
- **The mesh is conservative on purpose.** A vertex's local solve seeds
  from its OWN last position carried by the plane's frame-to-frame motion
  (the planar prediction wobbles on a surface that is not a plane); it
  votes only with a patch that is nearly all in frame, well matched,
  moved less than its own radius, NOT weighted out by the planar solve
  (`Fit::weights`, the occlusion map), and scoring better off the plane
  than on it by `PATCH_GAIN` — a 21 px patch on fine repetitive texture
  settles into the nearest wrong minimum with full confidence otherwise
  (vertex 37, 1.5 px, every frame). Patches are 29 px / 3 levels.
  `probe_mesh_*` prints the worst vertex and why vertices declined.
- **The bake also KEYS channels, and the poll mirrors them into
  `undo_last`.** `poll_planar` turns each frame's corners into the twelve
  `Planar*` channels (`planarui::drive_of`: a Procrustes similarity in the
  Transform's own convention — rotate/scale about the reference centroid,
  translate by the centroids' difference, `[[c,−s],[s,c]]` in y-down
  pixels — and the frame's corners under `h_from_quad`), skips locked
  ones, and writes the same keys into `undo_last` so a poll is still not a
  history step. `drive_planar` cables the pin channels into a Transform's
  or CornerPin's `Pin0..3` with `patch.link` (which replaces whatever fed
  the input). The suite judges the driven pin at the REGION's corners: at
  the frame's corners a 0.2 px track reads as 2 px, an extrapolation
  three times further out.
- **Results are keyed by COMP FRAME.** `start_planar` maps the DISPLAYED
  tick onto the windowed frame list through the Read's `first`
  (else `start`), the same rule `input_frames` cut the list by; index 0 is
  `tick0`. The overlay looks the frame up by tick, so a placed plate draws
  its track where it plays.
- **Trust is decided per frame, and a lost frame COASTS.** A fit with
  score ≥ `CONFIDENT` and little weighted out is taken as it is; anything
  weaker must land within `CONSISTENT` of the region's extent from where
  the velocity put it; otherwise the frame is lost, the plane coasts on
  its velocity for `COAST_MAX` frames (then holds) while the coarse search
  tries to re-acquire, and no template is ever cut from it
  (`may_reanchor`: no sudden score dip, little rejected). The first
  rendered brick-wall shot showed why: a ball entering the region cost a
  67 px slide onto the next brick row at score 0.67 (a lattice has a
  fair minimum one period over), the old policy then re-anchored ON the
  occluded frame and carried the ball along for the rest of the shot. With
  the policy the same shot tracks to 0.74 px worst, two frames coasted.
  `a_big_occluder_coasts_and_never_becomes_the_template` pins it.
- **Bulky node data rides behind an `Arc`, stays out of the per-frame
  keys, and a poll is not an undo step.** Reported: "it starts off fairly
  quick but gets super slow". The app clones the graph per frame
  (`to_engine_graph_at`), deep-compares it per frame for the undo step,
  and serialises every node's params to JSON per frame for the render-cache
  and subtree keys — all linear in the track, and the undo stack pushed a
  full graph copy per polled batch. `graph::TrackRef` (Arc, `ptr_eq` fast
  path, copy-on-write `get_mut`), `main::key_kind` (a `Cow` that strips
  the results, which cannot change a pixel) and `poll_planar` syncing
  `undo_last` fix the three. Measured through `--test-planar` with
  `WOBBLE_PLANAR_FRAMES=240`: 14.0 s → 8.2 s for the planar pass, now equal
  to the CLI. Paint strokes have the same shape and no such treatment yet.
- **Shading is handled in the RESIDUAL and the WEIGHTS, not by
  high-passing the picture.** A cloth shot (`data/hcr080`: a
  necklace on folded cloth, the sheen moving with the camera) was lost
  from frame 8 as "covered": the raw residual of that shading is a broad
  smooth blob the Tukey cut read as an occluder, and at the coarse levels a
  Gauss-Newton step reads it as motion. `centre_residuals` subtracts each
  residual's mean over `SHADE_RADIUS` in every iteration (only where the
  region dwarfs the window — on a mesh vertex's 29 px patch it took the
  signal too, 2.4 px off on a plane); the Tukey cut then sees only the
  structure. A LOCAL VARIANCE gate (`robust_weights`, once per level)
  drops windows where the picture has no structure while the template has
  some — a flat grey ball — because a smooth occluder has a smooth
  residual and the centring alone waved the ball through. NOT a
  correlation gate: on fine cloth the correlation length is a couple of
  pixels, the folds' parallax misaligns that much everywhere, and it threw
  away 90% of a region whose coarse structure matched. NOT a high-passed
  picture (`detail`, kept as an opt-in): 0.15 → 0.44 px on the synthetic
  shot, and the band-passed scores never reach `CONFIDENT`. And the SCORE
  stays the raw ZNCC: a shading-invariant score rated a 123 px miss at
  0.86 — the smooth part of a misalignment is most of it.
  `a_sliding_sheen_is_not_an_occluder` pins the shading case.
- **A re-anchor cuts through the weights** (`Template::build_masked`,
  mesh patches under the mask get no vote): with the cut gated on
  `rejected < 0.3` the cloth shot never re-anchored and its score decayed
  to nothing; gated on 0.5 the rendered shot re-anchored with the ball
  half in the region and lost the rest of the shot (667 px). Masked, the
  gate can sit at 0.7. `a_re_anchor_leaves_the_occluder_out_of_the_template`.
- **The solve starts from the velocity's prediction and stops on
  stagnation.** It used to start from the last frame and add a second
  solve from the prediction on every frame under score 0.9 — the cloth
  shot's honest scores are 0.85, so every frame paid twice — and on a
  surface that is not quite a plane the step never fell under `eps`, so
  every level ran to its cap (90–130 iterations a frame). Now: one solve
  from the prediction, a second from the last frame only on a score drop
  or a fit that disagrees with the prediction, and a level stops after
  two iterations without a thousandth of improvement. 788 → 120 ms a
  frame on that shot. `WOBBLE_PLANAR_TRACE=1` prints the phases.
- **Measure the tracker, not the harness.** The first "slow" numbers on
  the 2K shot were the CLI writing a 2048² overlay PNG per frame (~400 ms)
  and cold reads of 168 MB multi-part EXRs off a cloud-synced folder; warm, one decodes
  in 19 ms (`cargo run --release -p engine --example decode_time -- <dir>`).
  In the app the job reads the NVMe tier first (`planarui::spawn` takes
  the `DiskCache`).
- **The solve is parallel (rayon) and a lost frame is capped.** Samples
  live in one `Vec<Sample>` so a chunk is one slice; warp, statistics,
  residuals and the normal equations are chunked map-reduces, the mesh
  runs one task per vertex, the coarse search one per row of shifts, the
  pyramid one per row. A lost frame used to cost eight tracked ones: the
  coarse search plus three solves, each running to its iteration cap with
  the region half off the picture. After a lost frame the solves are
  short (`cheap`), on hold the coarse search runs every fourth frame, and
  the coast extrapolates the CORNERS linearly (composing the frame-to-frame
  homography compounded its perspective terms and sent a region 150k px
  away in twelve frames).
- Coordinates are continuous pixels with centres at `+0.5` (the viewer's
  output space); the pyramid's `[1 3 3 1]` decimation is centred between
  pixel pairs so a coordinate halves EXACTLY per level. A sampling
  convention slip here reads as a constant 0.7 px error, not as a bug.

## Flow + motion vectors (`engine/src/searaft*`, `engine/src/rife*`, `engine/src/flowsolve.rs`, `engine/src/nodes/flow.rs`, `app/src/flowui.rs`)

Two flow networks in WGSL behind the Flow node (`FlowEngine`): SEA-RAFT
(default) and RIFE v4.6; `flowsolve` runs either over a directory of
rendered frames. VectorBlur and VectorWarp read what it writes. README has
the user-facing story. What must stay true:

- **Both ports are checked against torch, not against themselves.**
  `searaft::tests::golden_pair_matches_torch` (`data/searaft/golden.wifw`,
  from `tools/searaft_golden.py` run with any python that has torch +
  safetensors — CorridorKey's venv does) agrees to ~1e-6;
  `rife::gpu::tests::golden_bundle_matches_torch` likewise. Both self-skip
  without `data/`. Change a kernel, run them.
- **SEA-RAFT never builds the all-pairs volume** (4.8 GB at 2K): the
  `corr` kernel samples the pooled features bilinearly and dots them, which
  IS bilinear sampling of the volume. RAFT's channel order quirk is kept:
  the X offset comes from the SLOWER index. BatchNorm is folded on load;
  `downsample.1` is an alias of `bn3` in the file.
- **The conv kernel is the time.** 8×8 pixels × 16 output channels a
  workgroup, input staged in shared memory, weights re-laid
  `[oc/16][ic][k][k][16]`: 18.7 s → 0.96 s a pair at 960×544 against the
  naive one. A 2×2 pixel block per thread measured 3× SLOWER (spills).
  `searaft::bench::timing` (ignored) prints the per-pass cost.
- **Ask RIFE for the MIDDLE of a pair.** `F_t→1 − F_t→0` at `t = 0.5`,
  then `flowsolve::anchor` carries the midpoint field back onto the frame
  (`f(p) = M(p + f/2)`). At `t = 0` IFNet under-reads anything past a few
  px (a (6, 4) move read (5.5, 2.3)). SEA-RAFT's flow is used as it is.
  `flowsolve::probe` (ignored, `FLOW_PROBE_DIR=<Input dir>`) prints every
  way of asking on a real pair.
- **Weights are the upstream files, read in Rust**: RIFE's `flownet.pkl`
  by `torchpt` (a restricted torch-zip unpickler — keep it restricted),
  SEA-RAFT's `model.safetensors` by `safetensors`. Both catalog rows are
  pinned by HF commit AND sha256; RIFE's is a third-party byte-identical
  copy (upstream is a Google Drive zip) a mirror we control should replace.
  Only the C+T SEA-RAFT checkpoint is catalogued: the KITTI-tuned ones are
  trained on non-commercial data (rule 10 would tag them).
- **Scale down, never refuse**: `flowsolve` halves the analysis size until
  SEA-RAFT's largest buffer fits the adapter's storage binding limit.
- **Vectors are data: no input transform, ever**, stored in FULL-res
  pixels; the Flow's emit multiplies by `proxy_scale` (packed in its slot),
  so consumers see canvas pixels and need no proxy handling of their own.
- **The Flow node never samples its input** (`inputs_unsampled`, loaded or
  not): the input is read by the analysis render only. Unanalysed it is
  zero motion, so a VectorBlur below it is a no-op.
- **`rg` forward, `ba` backward, always both.** With `backward` off, and at
  the ends of the range, the missing one is the other negated (constant
  velocity), so a consumer can always read both halves of a shutter.
- **Rifetime (was Retime; setups load either name) shares the Flow runner** (`flowui`: one run slot, the same
  render → worker → adopt; `out_dir_name` says `Frames` vs `Vectors`).
  `engine::rifetime` makes the in-betweens: RIFE judges motion on the ENCODED
  frames, `reference::warp` moves the LINEAR ones with its flow + mask, and
  a frame on a source frame is a file copy. Output files are named by
  OUTPUT comp frame from the input's first; `ranges::time_shift` gives the
  node its stretched range (speed mode), anchored there. Curve mode keys
  `Channel::RifetimeFrame` (a source frame per output tick); the app samples
  it over the input's range (`flowui::rifetime_times`) into the RECIPE and
  the plan (`plan_times`), so a curve edit is a new directory, never stale
  frames. Unrendered, it passes through.
  `rifetime::tests` pins a made frame 4× closer to the true halfway picture
  than a cross-fade; `--test-flow` pins the range and the in-betweens.
- **Frame Fix is the third runner client** (`out_dir_name` `Fixed`). A
  job step names BOTH frames (`(a, b, t)`; `rifetime::adjacent` turns a
  Rifetime plan into steps), so a bad run is made from the good frames
  round it; good frames hard-link across. Its blend mode is a time node
  whose two branches are NOT Floor/Ceil of one map:
  `NodeKind::blend_branch_maps` (`TimeOp::Fix`, before / after a bad run)
  and `blend_weight_at` give them, and both the reader walk and codegen's
  stage walk branch on the two phase keys (the stage walk used to take one
  key, leaving a stage above any blending time node unbound).
  `engine/tests/framefix.rs` pins all of it. Freeze is a speed-0 map
  (`FreezeParams::map`, keyed = a sampled lut held flat past its ends).
- **Vectors have their own pipe** (`PortKind::Vector`, yellow): a Flow's
  output and the vectors port (1) of VectorBlur/VectorWarp. Vector feeds
  Image (an Expression reads the numbers), never the reverse — a picture
  in a vectors port is read as motion. A new motion consumer types its port
  in `input_kind`; `flow::tests::vectors_ride_the_vector_pipe` pins it.
- **A loaded Flow clears slow** (`cost::clears_slow`), like a GraphBake.
  VectorBlur is a one-pass stage (Slow) whose vectors and modulate matte
  cost nothing until wired; VectorWarp fuses like UV Warp.

## Stabilize node (`engine/src/stabilize.rs`, `engine/src/nodes/stabilize.rs`, `app/src/stabilizeui.rs`)

Click-to-analyse on the Flow runner's types (`flowui::{Run, Phase, View,
Action, Msg}`, its own `Gfx::stab` slot): render input 0 (and the matte,
port 1) as `BakeSink::Stabilize { port }`, `engine::stabilize::run` on a
worker (SEA-RAFT pair flow → grid → robust `fit` → chained path), then
the path lands ON THE NODE. README has the user story. What must stay
true:

- **The node is a fused corner pin, never a stage.** `StabilizeParams::pin`
  is TRANSIENT, set per tick in `to_engine_graph_at_f` from
  `pin_at(t)`, packed as a CornerPin's inverse homography (3 slots,
  constant `topo_tag`). So it is cyan by `slow_matches_codegen`, and
  smoothing / reference / axes / reverse / zoom are uniforms: changing
  them never recompiles or re-analyses. `live_kind` does NOT set the pin
  — read it off `to_engine_graph_at` (the suite once measured 58 px of
  "residual" that way).
- **The path is data on the node, behind an Arc** (`StabPathRef`, the
  `TrackRef` pattern), saved with the setup; `motion[t]` carries the
  first frame's pixels onto frame t (path pixels, `size`). `key_kind`
  strips `motion` and keeps `recipe` (the run key = the directory name):
  same key, same analysis, same pixels. The landing writes into
  `undo_last` too — a click's result, not an undo step.
- **Lock-off with all four axes is the full model** (`C_ref · C_t⁻¹`);
  anything else goes through similarity parameters about the centre
  (rotation unwrapped), so only the similarity part is corrected.
  `reverse` is the exact inverse INCLUDING zoom, so reverse ∘ stabilise
  is the identity (`reverse_undoes_the_stabilise`).
- **The matte is port 1, a plain Mask input, not `mask_input`**: it never
  reaches the shader, it only leaves points out of the vote (red > 0.5).
  Both renders cover the PICTURE's range, so frame i is frame i.
- `engine/tests/stabilize.rs` runs the real network on `write_shaky`
  (skips without `data/searaft/`); `--test-stabilize` does it through the
  app, then judges the pin, the picture, reverse and a matted re-run.

## CameraTrack (`engine/src/camtrack.rs`, `engine/src/nodes/camtrack.rs`, `app/src/camtrackui.rs`)

A 3D camera solved off the picture, click-to-solve on the Stabilize
pattern (`flowui::{Run, Phase, View, Action, Msg}`, its own `Gfx::camtrack`
slot, `BakeSink::CamTrack { port }`): render input 0 (and the matte, port
1), `engine::camtrack::track` (SEA-RAFT forward AND backward per pair, a
grid of points carried along, forward-backward checked, re-seeded where
thin) then `engine::camtrack::solve` (our own SfM: eight-point + RANSAC
init on the pair with the most parallax, robust pose refinement per
frame, DLT triangulation, Levenberg–Marquardt bundle with the points
eliminated through their 3x3 blocks, windowed as keyframes arrive and
globally at the end, the focal optionally free). README has the user
story. What must stay true:

- **The node is a passthrough, cyan, and the solve is data on the node**
  (`CamSolve` behind `CamSolveRef`, the `StabPath` pattern; `key_kind`
  strips it, keeps `recipe`). A Camera reads it through
  `Source::Tracker { node }` and `scene3d::tracker_solve`, a registry
  `camtrack_sync` keeps in step every redraw (pointer compare per node),
  so `apply_file_source` needs no graph. `bake to keys` works unchanged.
- **Two recipes.** The TRACKS directory is keyed on the inputs' subtree
  keys, the detail and the range; the SOLVE directory (`Solve-<key>`
  inside it) on that plus `recipe_solve()` (focal to 0.1 mm, refine). A
  lens change re-solves off `tracks.txt` with no render and no network;
  `scale` and `show` are read at use. `--test-camtrack` pins both.
  **`guess`** (in the recipe, focal ignored) is `solve_guess`: a solve
  from each of `GUESS_STARTS` (14–135 mm) with the focal free, the best
  mean error kept — a lens from nothing, pinned by
  `the_lens_is_guessed_from_nothing` and the suite's last stage.
  `LENS_PRESETS` are 35 mm-equivalent on a 36 mm back (the solve only
  needs the ratio).
- **The 3D view draws every Camera a Blend 3D reaches** (`view3d_cameras`,
  `paint_camera`: a frustum sized `0.12 × the eye's distance`), and for a
  tracker-sourced one the whole path and the point cloud (`paint_track`);
  `view3d_bounds` takes them in, and a solve landing re-frames a 3D view
  that shows it. The API's `node_get` returns the `key_kind`-stripped
  node, or a solve is half a megabyte of JSON.
- **Picks + alignment are read at use, never a re-run.** `picked`
  (point indices) belong to `picked_for` (a solve recipe) and `picks()`
  is empty for any other solve. `align_rot` / `align_off` / `scale` make
  `align_mat()`: `R · T(−scale·off)` over the scaled solve, and EVERY
  scene read goes through `camera_mat` / `point_world` (the Camera's
  `camera_at`, the 3D view's path, cloud and bounds, `card_from_picks`);
  the viewer overlay projects in the camera's own frame and needs none of
  it. `set_ground` fits a plane (`plane_normal`, the scatter's smallest
  axis) and turns its normal to +Y with the FIRST camera on the normal's
  side; `set_distance` sets `scale` from two picks and `ruler`.
  `the_alignment_ops_put_the_picks_where_they_say` and the suite's last
  stage (61 floor points within 0.007 of y = 0) pin it. S-click over the
  viewer is `camtrack_gizmo` (nearest point within 12 px, roto's S).
- **Conventions.** Inside the solver it is CV (camera down +Z, image y
  down, `x_c = R x + t`), world = frame 0's camera, median point depth 1.
  `Solve::camera` / `Solve::point` are the ONE place that flips to the
  cards' Y-up / −Z (`FLIP`); `the_cards_camera_looks_at_the_points_it_solved`
  pins it. Rotation updates are left-multiplied `exp([δ]×) R`.
- **Full-size flow by default.** Measured on the synthetic three-plane
  dolly (`synth::write_shot`, `engine/tests/camtrack.rs`, needs
  `data/searaft/`): half-res tracks solve to 0.15 px yet land 0.6° /
  2.8%-of-travel off the truth — a bias the solve cannot tell from
  motion; full-size: 0.06 px, 0.1°, 1.1%, focal refined to 0.01% from a
  10% wrong start. ~1.8 s a frame at 960x540 for both directions.
- **What limits the camera is the COUNT of tracks, not the fit**: on the
  synthetic tracks, 170 points under 0.4 px jitter leave a 1° / 10%
  ambiguity that explains the data BETTER than the truth (the unit test
  asserts solve cost < truth cost, so a stall and an ambiguity are told
  apart); 450 points bring it to 0.03°. Hence the grid seeding (`CELL`
  24 analysed px) and every frame in the bundle up to `max_keyframes`
  (100). `WOBBLE_CAMTRACK_TRACE=1` prints every bundle step.
- **A pan is refused with words** (`a_pan_is_refused_with_words`): the
  init needs a keyframe pair with ≥ 1% of the width of median
  displacement AND parallax; without one the error says MOVE.
- Not yet: lens distortion (undistort upstream), cards from picked
  points, ground-plane / origin tools, pattern-tracker refinement of
  the tracks. The matte keeps a moving subject out; nothing else does.

## 2dini light + SDF effects (`engine/src/geolight.rs`, `app/src/geolightui.rs`)

A 2dini node's `look` (`GeoLook`): radiance-cascade light and SDF
effects, GPU passes after the layer is rasterised. README has the user
story. What must stay true:

- **Off is exactly the old path.** `GeoLook::active()` false: no light
  target, no passes, the fused pass samples the bare layer. On, the layer
  is drawn by the LIT pipelines (`georaster`, `fs_lit`: a second target,
  rgb given off + a = stands in light's way, packed from `Style.emit` into
  `PrimGpu`'s pad lanes) and the fused pass samples `LitLayer::out`.
- **One SDF, two readers.** Seed + jump flood + finalise (storage BUFFERS
  for the flood: rgba32float textures are not filterable and auto layout
  would refuse them) give r = signed px distance, gb = the nearest outline
  point (what a grown pixel takes its colour from). The cascades march it;
  the finish pass reads it for every effect.
- **The cascades are graav_rs's `hrc.wgsl`** (same author), marching the SDF
  instead of the world's, with no bounce or temporal blend (a comp frame
  renders once). Keep the four-frustum layout and the "mix AFTER merging"
  rule. `Grid` caps the square grid at `MAX_EXTENT` 1024 by coarsening
  cells, never by refusing.
- **Params are uniforms and `Need` is the memo.** `set_look` compares the
  look and marks Finish (any param), Light (`hrc_key`: light on, cell px,
  sky) or the compositor marks All (the layer re-recorded). A still drawing
  with still params runs nothing. The look's numbers are channels
  `look.<param>` (`Channel::Custom`, `anim.rs` routes the prefix to
  `GeoLook::get/set`), so the Geo node's `topo_tag` never moves.
- `engine/tests/geolight.rs`: a lamp and a wall (shadow, lit side, air),
  light-off leaves pixels exact, each effect lands where its params say,
  and a look edit is not a recompile.

## 2dini fluid (`engine/src/geofluid.rs`, `app/src/geofluidui.rs`)

The Physics node's FLUID mode: graav's CPU position-based fluid
(`graav_core::fluid`) ported, baked like rigid/soft and played back from the
node. What must stay true:

- **It runs in graav's units.** Geo space is scaled so the spacing is
  `SOLVER_SPACING` (0.135) and the smoothing radius `H` (0.35); graav's tuned
  constants then carry over unchanged. The rest density is MEASURED off a
  lattice (`rest_density`), not graav's 55.
- **A push out of a solid lands `SKIN` (1e-4) past the surface.** graav's
  0.01 became an upward kick ten times gravity's on every substep a particle
  rested on something, because the push turns into velocity.
- **The move is walked against the solids** (`Solid::collide`, steps of 0.4
  spacings): an open line is ~1.5 spacings thick and fast water tunnelled
  through it otherwise.
- **The look is read at playback** (`frame_geometry_look`), never hashed:
  only `spacing`, `viscosity`, `iterations` reach `input_hash`, and only in
  fluid mode, so rigid and soft bakes from before the mode are not stale.
- **The drawing fits 2dini's caps** (4096 points/prims, 512 verts a prim):
  outline rings are grouped by containment (marching squares gives no
  winding), smoothed and resampled to `MAX_SHAPE_POINTS` a shape /
  `MAX_SURFACE_POINTS` a volume; dots stride past `MAX_DOTS`. Particles cap
  at `MAX_PARTICLES` and particle-frames, coarsening the spacing to fit.
- `GEO_DUMP_ONLY=dam GEO_DUMP_FRAMES=0,12,24 cargo test --release -p engine
  --test geo dump_presets -- --ignored` renders the dam break through the
  real compositor.

## Text node (`engine/src/textfx.rs`, `engine/src/textgpu.rs`, `engine/src/nodes/text.rs`, `app/src/nodes/text.rs`)

A design-app text block with TextMeshPro-style per-glyph animators and a
path. README has the user story. What must stay true:

- **The text is a LAYER, never uniforms or topology.** `textfx` lays the
  glyphs out (ems, block space), runs the animators and the path, and
  emits `Instance`s; `textgpu` draws them as quads over a per-font SDF
  atlas into a per-node rgba16f layer at the node's format (proxy
  included) — the Geo layer precedent. The fused pass is one
  `sample_px` + over (`SourceBinding::Text` resolves the layer for a
  Text, the old block atlas for a Burnin). `topo_tag` is a constant.
  The layer is re-recorded only when `Built::hash` (instances + uniform
  + atlas) moves; `--test-text` and `a_still_title_is_drawn_once` pin it.
- **Cells are cached per (font, char), atlases per char SET**: a new char
  costs one glyph's field. `SPREAD` (a quarter em at `EM` 96) is the budget
  outline + weight + glow + shadow softness share; the CPU clamps to it.
- **Animator numbers are channels by NAME** (`text.<name>`,
  `text.a<i>.<name>` on `Channel::Custom`, the 2dini look's precedent);
  position and size keep `TextX/TextY/TextSize` so old curves land.
  `nodes::text::remove_animator` slides later animators' curves down —
  patch cables onto a removed animator are not re-pointed yet.
- **The Burnin still uses `textraster`'s block atlas**, and its test holds
  it to a Text of the same words pixel for pixel — keep the two layouts
  agreeing (kerning, ascent, alignment) or that test fails.
- Wiggles read the transient `time` (set in `to_engine_graph_at_f`) and
  the compositor's clock fps.
- **2dini's Text shares the typesetting** (`textfx::places`, the placement
  half both use; `geo::text_fx_geometry` places each glyph's OUTLINE with
  it). `GeoTextFx` all default and no `path` socket wired = the old
  `geotext::layout` path, byte for byte. The path socket is APPENDED (5) so
  old wires keep their ports; literals past it are the path offset
  (`TEXT_PATH_OFFSET`), the style block (`GEO_STYLE_PARAMS`,
  `text_style_index`), then each animator's `ANIM_PARAMS`
  (`text_anim_index`), keyed as `Channel::Field(i)`. The standalone
  node's styling is GEOMETRY here (`text_fx_geometry`: shadow, outline
  ring and weight are stroked copies of every glyph, layered under all
  the faces), so 2dini's own fill/gradient/effects still apply; glow and
  softness stay the drawing's effects pages. Both panels draw
  `nodes::text::animators_page`. A **words** port (1, the text pipe) takes a
  Prompt; `EditorGraph::prompt_on` resolves it into the engine copy.
- **The Prompt node is called String** (the Tinter precedent: `NAME`,
  the catalog, the panel; `NodeKind::Prompt` / `PromptParams` and the
  setup key stay, so old setups load untouched). Inside a 2dini it is
  `GeoNode::String` (output `PortKind::Text`, draws nothing), wired into
  the 2dini Text's **words** socket (`geo::TEXT_WORDS_INPUT`, 6, appended
  after the path so old wires keep their ports; `geo::words_on` reads it
  in the eval). Socket 6 shares its index with the path-offset CHANNEL,
  so `channel_name` skips a text socket, and "draws geometry" is
  `GeoNode::is_geometry`, never `!is_scalar` (a dangling String must not
  become the drawing).

## Splat node (`engine/src/splat.rs`, `engine/src/splatgpu.rs`, `engine/src/nodes/splat.rs`, `app/src/nodes/splat.rs`)

A 3D Gaussian splat capture (`.ply`, `.spz` v1–4) drawn through a Camera:
the picture, and its depth on output 2. No training. README has the user
story. What must stay true:

- **The capture is a LAYER, never topology** (the Plasma / Text
  precedent): `splatgpu` draws into a per-node texture, the fused pass
  samples it (`SourceBinding::Splat(host, depth)`). Camera, placement,
  opacity are the layer's 128-byte uniform; `topo_tag` only tells the
  picture from the depth copy.
- **Everything per frame is on the GPU and nothing reads back**: project
  (cull, EWA covariance, SH for the view, atomic append), an LSD radix
  sort on 24-bit log-depth keys (6 passes of 4 bits; per-block counts, one
  scan, a stable scatter that sorts each 512-block in workgroup memory —
  kept under 8 KiB, see the Apple cliff), then instanced quads nearest
  first, blended UNDER (`1 - dst.a`). Dispatch and draw counts are
  indirect. One compute pass, one render pass.
- **Drawn only when dirty AND sampled.** The layer's hash is its uniform +
  cloud generation + whether depth is wanted; the encode skips layers the
  current plan does not bind. A still camera costs nothing
  (`a_still_camera_draws_once`).
- **The camera is lowered, like a Starfield's** (`nodes::splat::lower`,
  after sources, next to `scene3d::expand`): the Camera3D lands in the
  transient `cam`, the wire goes. Unwired = the default Camera. The same
  pass stamps `gen` (the cloud's load generation), so a capture that
  finishes loading CHANGES the graph and the viewer redraws.
- **Depth is a second lowered node** (Render3D's pattern, `depth_of` +
  `format_alias`) reading the host layer's depth target, which is only
  drawn while `depth_used`. Weighted distance where coverage ≥ 0.5, the
  far clip elsewhere.
- **The cloud keeps the file's frame.** A `.ply` is right-down-forward, an
  `.spz` right-up-back (ours); the model matrix converts (`convert`), so SH
  is evaluated in the file's frame with the camera taken INTO it. Do not
  rewrite positions at load without rotating the SH.
- **Colours are blended as trained (display sRGB), then decoded** to
  scene-linear working primaries in the fused pass (`splat_px`); `linear`
  skips the decode.
- **Loading is off the UI thread, cached on path + mtime**
  (`splat::request`); never refuse a big one: `splat::pack` drops SH bands,
  then keeps every k-th splat, under the device's largest binding
  (`set_gpu_budget`), and the panel says what went.
- It is a SLOW path (`cost::speed_of`): red in the graph, a Bake clears it.
- `engine/tests/splat.rs` pins: nearest wins in either file order, depth =
  distance, a `.ply` is turned Y up, a shuffled 150k capture renders the
  same, a still camera draws once. `WOBBLE_SPLAT_PERF=1` prints the time for
  4M splats at 1080p (≈41 ms on the M-series dev Mac).
- Not built: compressed `.ply`, SH band 4 (read, not drawn), motion blur across the
  shutter, a viewer gizmo for the placement, a tile rasteriser (big
  near-camera splats are fill-rate bound).

## Plasma node (`engine/src/nodes/plasma.rs`, `engine/src/plasmagpu.rs`, `app/src/nodes/plasma.rs`)

GPU particles along magnetic loops and streamers: a corona. README has the
user story. What must stay true:

- **There is no particle buffer and no state.** A particle is its
  instance index: the vertex shader hashes it into a field line, a strand
  and a place along it, and the clock moves the place. So any frame draws
  on its own (scrub, cache, farm), and the only upload is the 192-byte
  uniform (`PlasmaParams::uniform`, `UNIFORM_VEC4S`). Anything that would
  need last frame's positions does not belong here.
- **The uniform's bytes are the layer's memo** (`PlasmaRaster::set`): a
  still frame with still params draws nothing. `time` is in it — stamped
  at both of `editor.rs`'s stamp sites, like a Noise's.
- **Everything is a uniform.** One topology (`TOPO`); the count is the
  draw call's instance count. The layer is `SourceBinding::Plasma`, drawn
  additively (One + One on rgb AND alpha) into rgba16f before the fused
  pass, on the Text layer's pattern; the fused half (`plasma_px`) adds
  the star's disc under it.
- **Pixel lengths scale by the proxy in ONE place** (`uniform` /
  `pack_fmt`); angles and radii-relative numbers do not. Light per
  particle is divided by the count, so the count is quality, not exposure.
- **A streamer's falloff is split between placement and brightness**
  (half each, the inverse CDF in `vs`). All in brightness, the dim outer
  light is below a half-float's step next to the bright limb and never
  adds up; all in placement, the few far particles show one by one.
- A Glow over it wants `knee: 0`: a soft knee lifts the faint tail into a
  flat grey plateau (it looked like a precision bug and was not).
- The shader's `u.v[i]` reads and `uniform`'s `v[4i..]` writes are the
  same table: change them together.
- `engine/tests/plasma.rs`: light hugs the limb, the disc hides the far
  side, the clock moves it without a recompile. `PLASMA_DUMP=<dir> cargo
  test --release -p engine --test plasma look -- --ignored` renders the
  corona comp for a look (`PLASMA_FRAMES=72` an APNG, `PLASMA_CLOSE=1` the
  limb close up).

## Particles node (`engine/src/nodes/particles.rs`, `engine/src/particlegpu.rs`, `app/src/nodes/particles.rs`)

A general 2D particle system: a node that owns a graph of `PNode`s (Emit,
Force, Merge, Render), opened by a dive (`DiveKind::Particles`, the SDF
pattern: `to_editor`/`from_editor`, inner kind `NodeKind::ParticleNode`,
its own add-menu world `NodeCat::Particles` / `NodeEditor::in_particles`).
README has the user story. What must stay true:

- **A frame is a pure function of authored data + frame number.** Slot
  `i` of an emitter is reborn every `cap / rate` frames offset by
  `i / rate`; every random is a hash of (slot, generation, seed); the sim
  starts EMPTY at `start - preroll`. That is what lets the compositor jump
  anywhere. `engine/tests/particles.rs` asserts played == jumped ==
  scrubbed-back bit for bit; keep it that way (no atomics, no
  order-dependent writes, nothing read from the previous frame's draw).
- **State is GPU-resident; nothing in the sim is a `queue.write_buffer`.**
  The clock lives in vec4 0 of the state buffer and is advanced by a
  one-thread `tick` dispatch; reset/restore are encoder copies. So a
  thousand-frame catch-up is one encoder, and two sims in one submit can't
  clobber each other's frame number. The table (`Compiled::sim`) is
  written only when `sim_hash` changes, and that also resets the state.
- **A run of steps is ONE compute pass.** wgpu spends a backend command
  buffer per pass and Metal treats the 4097th outstanding one as device
  lost: a pass per step crashed the Mac on a comp whose playhead sat a
  thousand frames past `start`. `ParticleRaster::run` alternates step and
  tick inside one pass; the pass breaks only for a checkpoint copy, and a
  long catch-up checkpoints coarsely (at most ~32 breaks).
- **Checkpoints thin, never truncate.** Every `interval` (8) frames from
  the origin, under `CHECKPOINT_BUDGET` per node; over budget the interval
  doubles and odd checkpoints go. A node too big for one checkpoint still
  works, re-simulating from the origin.
- **Sim and look are hashed apart.** A Render edit redraws; an emitter or
  force edit re-simulates. The sim runs in FULL-RES px; proxy only scales
  the draw (`RENDER_VEC4S` lanes 2, 12–14), so a proxy flip is never a
  re-sim.
- **Nothing recompiles.** Forces are records the `step` kernel loops over
  (kind code + emitter MASK: a force acts on the emitters upstream of it,
  in flow order by `depth_of`); `topo_tag` is a constant. A new force kind
  is a `ForceKind` arm, a code in the kernel's if-chain, and panel rows —
  keep the `ForceParams` record one struct so switching kind keeps numbers.
- **Field Lines' poles are placed on the CPU** (`field_charges`, the
  shader's `rnd` twin) into the table's charge block (`MAX_CHARGES`).
  `foot` re-homes newborns onto a pole and flips their direction to flow
  away from it; a particle that reaches a pole after 3 frames is absorbed
  (it would otherwise stall there as a bright dot).
- **A crisp thread is many particles on the SAME line.** `strands` picks
  one of N launch directions a pole; continuous directions are a haze (the
  first corona preset was fog for exactly this reason). The charge's 4th
  lane names its partner (±1 limb, ±2 face, 0 none); `fan` is measured from
  the chord to the partner on the OUTWARD side, and because a lone pair's
  lines are circles through both poles it bounds the loop height. `own_pair`
  makes a rooted particle feel only its pole and partner — the pole's table
  index rides in `p1.w` above the emitter's six bits (`& 63` to read the
  emitter); off, all poles superpose and the lines tangle.
- **Turbulence that `follow`s must evolve SLOWLY next to a line's travel
  time.** A particle keeps the push it was given, so a field that changes
  while particles run down a line reads as a ripple along it, not a sway.
- **Depth is a third vec4 a slot** (`(z, vz)`, z towards the eye; 48
  bytes a slot). Only Field Lines with `sphere` > 0 moves it: poles are 3D
  (`field_charges`, a charge is TWO vec4s `(x, y, z, q)`, `(link)`), the
  field is summed in 3D, and a loop's plane holds its chord and the
  outward normal. Every other force is flat and leaves z alone. Flat
  (`sphere` 0) must stay exactly the old picture: all z are 0. The disc
  that hides the far side is the RENDER's (`occlude`, drawn-time only, so
  toggling it is not a re-sim).
- `p1.z` is two things: below 0.5 = flows against the field, and its
  fractional double is the line's own random (`RenderParams::vary`).
- Inner numbers are NOT channels in v1 (a dive drops inner curves, like
  SDF's). The owner's are `particles.<name>`.
- `PARTICLES_DUMP=<dir> cargo test --release -p engine --test particles
  look -- --ignored` renders every preset at frame 90.

## LUT node (`engine/src/lut.rs`, `engine/src/nodes/lut.rs`, `app/src/nodes/lut.rs`)

A LUT file applied in the fused pass. What must stay true:

- **The table is a TEXTURE, its shape is UNIFORMS, only the shaping is
  topology.** `SourceBinding::Lut(node)` binds a per-node 2D texture
  (`Compositor::update_lut_tables`, re-uploaded when `Lut::hash` moves,
  rgba32float where `FLOAT32_FILTERABLE`, else half); 1D/3D, sizes and
  domains are packed, so another file never recompiles. `topo_tag` is
  `input` + `shaper` + `output` (colour transforms written into WGSL).
- **One layout, two readers.** `lut::Layout` (cube slices tiled
  `size`x`size`, `tiles` a row; the 1D table from `row1d`, `width` a row)
  is what `Lut::texture` writes and `nodes::lut::LOOKUPS` reads, and
  `Lut::apply` is the CPU mirror `engine/tests/lut.rs` holds the GPU to.
- **The file's bytes are in the render key**: `LutParams::stamp` is
  transient (never authored, skipped when 0) and set only by
  `Gfx::key_kind` to the loaded hash. `lut::load` is cached (stat every
  500 ms, re-read on a new mtime/size), so asking per frame is free.
- **Resolve compatibility is checked against OCIO, not eyeballed.**
  `lut::tests::resolve_cube_and_csp_match_opencolorio` holds a Resolve-
  layout shaper+cube (`LUT_1D/3D_INPUT_RANGE`) and a `.csp` to OpenColorIO
  2.6's numbers, both interpolations. Interpolation (`lut::Interp`) is a
  uniform riding the has-3D flag (1 = tetrahedral, 2 = trilinear, Resolve's
  default project setting): no recompile, no new slot. A `.csp` prelut is a
  CUBIC (Rising Sun's Hermite, as OCIO draws it), not a polyline; it is
  resampled to `CSP_RESAMPLE` points. DaVinci Wide Gamut / Intermediate is
  a `color` preset so a DWG-project LUT shapes like Resolve's timeline.
- **A file that does not load is a passthrough** (the dummy binds, the
  shader's has-table flags are 0) and the panel says why. AgX's cubes go
  through the same parser (`agx::Lut3d` is `lut::Lut3d`).
- **Inverse View is a LUT whose table is SOLVED** (`engine/src/invview.rs`):
  `ViewSettings::display_rgb` inverted per lattice point by bounded LM,
  warm-started from the grey's inverse in the DISPLAY primaries, with a
  walk-out from grey when that misses (a start in a clamped region fell to
  black). Change the solve, bump `GEN`. Dither and limit are uniforms.

## Reader windows + shared decodes (`decode::Window`, `player.rs`, `main::sync_reader_windows`)

One big plate cut into sections by Crops and stacked in a Layer is one
reader INSTANCE a section (each layer is its own time path). Each used to
decode, cache and upload the whole frame. `--test-stress` is the
instrument; on an 8K plate in four slipped sections it measured 1.7
pictures/s warm and a scrub that showed nothing, against 47 and 53 now.
Its scrub goes back and forth like a hand: a sweep that WRAPS from the
last frame to the first puts every frame in flight behind the picture.
What must stay true:

- **A reader keeps only the box its consumers can sample.**
  `Graph::reader_rois` answers per instance (same keys as
  `reader_instances`): a box only where EVERY path to a sink runs through
  a Crop in crop/reformat mode with nothing but listed pointwise nodes
  between it and the Load; a Transform, a stage, a second uncropped
  consumer or anything unlisted asks for the whole frame. Add a node to
  that list only if it reads input 0 at the pixel it writes, in the
  input's format. `Pack::window` carries the box into the decode, both
  cache tiers (the NVMe key folds it in), the ring slot, the upload extent
  and the texture size; `seq.info` stays the PLATE (format resolution
  reads it), `Pack::dims` is the texture.
- **Which instances are windowed is topology; where the box sits is a
  uniform.** `Graph::reader_windows` → `ParamLayout::window_slot`, one slot
  an instance, subtracted from the leaf's sample position. Windows snap
  outward to `WINDOW_GRID` and are padded `WINDOW_PAD` px (a filtered
  read reaches half a pixel past the box).
- **A crop drag never waits on a decode.** While the asked-for boxes are
  moving a window is never tightened: it stays while it covers the need
  and is dropped for the whole frame the moment it does not; the rig
  tightens `WINDOW_SETTLE` after they hold still. A window change carries
  the pixels both textures hold across on the GPU
  (`ensure_reader_packed`), so the picture does not blink.
- **Every graph handed to a compositor fed by the rig goes through
  `stamp_readers`** (slots AND windows): the display's, both bake loops',
  the other pane's. The bake loops used to stamp nothing, and a Bake or
  Render of a Layer stacking one plate came out BLACK; `--test-stress`
  holds the bake to the live picture.
- **One decode feeds every reader of the file** (`shared_full`): a packed
  or windowed reader packs from the full frame in a small shared store
  keyed by file, frame, layer, proxy and the file's stamp; a second ask
  while it decodes WAITS for that decode, and the job then fills its
  SIBLINGS' RAM and NVMe entries (same frames, other windows), so a file
  is decoded once however far apart the sections are slipped.
- **Decode jobs run on plain threads, not rayon** (`spawn_job`): a job
  forks into rayon and may block on a shared decode; a rayon worker that
  blocks there can be the thread holding the lock. Off a job thread
  `shared_full` never waits (it decodes again), which is what keeps
  `precache_one` safe.
- **Big frames decode in parallel, a few at a time.** At
  `decode::PARALLEL_FRAME_BYTES` and up the EXR decoder ignores
  `single_threaded`, `BigDecode` admits `BIG_DECODES` at once and the
  full-frame buffers are recycled (a fresh 265 MB allocation costs more
  than the decode: 374 ms a frame against 185). `cargo run --release -p
  engine --example decode_conc -- <dir>` (`REUSE=1`) is the measurement.
- **NVMe writes happen after the frame is handed over, on ONE thread**
  (`queue_cache_write`), never where the decode ran: the compress took
  the cores the next decodes wanted (a first pass with the tier on:
  11–18 s, now 7). A backlog past 4 GiB drops the write, and a RAM hit
  whose NVMe entry is missing owes it again, so the tier converges on
  what is played. A suite that asks "is it on NVMe yet" must wait on the
  wall clock for it.
- **The ring is a BYTE budget; a slot is the size its frame needs**
  (`StagingRing::try_acquire_sized`). One slot size made every reader pay
  for the largest: a 2K element beside an 8K plate took a 265 MB slot a
  frame, and a parked 8K plate starved a scrub of small sections. A free
  slot of the wrong size is RETIRED for the asker (its index reused), so
  the mix follows what is being decoded with no rig rebuild. The player
  counts in bytes of one TICK (the active readers' frames): `reserve`,
  the chase cap and the evict-ahead top-up are all that many ticks.
  `spawn_decode` still refuses a frame its slot cannot hold (black, never
  an overrun). `staging::tests::slots_come_in_the_size_asked_for_under_one_budget`.
- **Chasing a scrub, a tick is issued whole and a showable tick is never
  evicted.** On a ring a few ticks deep (four whole 8K layers: 1 GB a
  picture, fifteen slots) the hand re-targets faster than a tick lands.
  Two things then showed NOTHING for a whole drag: the cursor followed
  the hand and left ticks with only some readers issued; and a tick that
  had just landed whole was evicted for the next target in the pump it
  completed in. So `pump` issues a chased tick only when all of its
  missing readers fit, and `evict_farthest` skips a tick the between rule
  can show this redraw (whole, in the shown..hand span) or one still
  landing there. The target waits; the picture follows at the rate the
  ring can carry (0 -> 8 pictures/s on that comp). `--test-scrubfid` is
  the guard that this cost nothing where the ring is roomy.
- **A bake decodes the readers above it** (`bake_reader_mask`), and all
  of them whenever the walk cannot be matched to the rig exactly.
- **A plate written over on disk is noticed** — RAM entries and resident
  frames carry the file's size + mtime (`file_stamp`): a RAM hit against a
  changed file is a miss, and the app asks `Player::stale_at` about the
  tick on screen when it changes and twice a second. Other resident
  ticks are caught when they are next decoded or shown and checked; the
  ring is not swept.

## Loops (LoopIn / LoopOut)

A subgraph iterated n times per frame. The graph stays a DAG: a wire dragged
from an output back into an upstream input (or a node into itself) lowers to
a LoopIn/LoopOut pair (`editor::insert_loop`) — the back-port's old feed
becomes the seed. The body must stay POINTWISE where it depends on the tap
(`codegen::loop_ok` is the single truth; its stage-kind list mirrors
`filter_stages::kind_of` — grow both together). n is an encode-time pass
count: animatable, clamped to `LOOP_MAX`, never a recompile. The compositor
skips the generic subtree dispatch for `FilterKind::Loop` and ping-pongs
a/b, always ending in `b`. A loop that cannot iterate runs its body once and
says so in the panel — never silently.

## 3D cards (`engine/src/scene3d.rs`, `engine/src/nodes/render3d.rs`, `app/src/nodes/scene3d.rs`)

Camera / Card / Scene / Render3D on `PortKind::Scene`. README has the user
story. What must stay true:

- **The scene pipe is CPU data; the engine sees a stage over pictures.**
  `scene3d::expand` (last in `to_engine_graph_at_f`, after animation) drops
  a Render3D's scene wiring, wires each card's PICTURE to ports 1..=n (port
  0 stays empty so every card is an aux texture with a known format) and
  fills the transient `Render3dParams::lowered`. Camera/Card/Scene stay as
  orphans (a viewed Card shows its picture). Render3D's `input_count` /
  `input_kind` / `output_kind` answer differently lowered — keep both.
- **Card count is topology, poses are uniforms.** One module per power of
  two of cards (`render3d::wgsl`, padded by re-binding `aux0`); 16 is the
  device's sampled-texture limit, not a choice. `uniforms_aux` builds
  every matrix (camera space → card's unit square) from `aux_fmt`, which
  carries each card's format and then the node's OWN (the compositor
  appends it).
- **Motion blur is poses, not re-renders.** `expand` asks its `at(node, dt)`
  callback for each shutter sample; the app answers with the pose this
  build resolved plus the curves' travel over `dt` (`editor.rs`), so a
  patched value holds still rather than vanishing.
- **Depth is a second lowered node** (`Lowered::depth`, `format_alias` to
  the host), so a picture-only render never writes it for a consumer.
- **Saved as `CardScene`, not `Scene3D`**: `Scene3D` is an alias a
  pruned FBX node left on `ChannelSelect`.
- `engine/tests/render3d.rs` pins: default camera on default card = the
  picture, nearer card wins in either scene order, depth = distance, a
  card moving across the shutter smears.
- **File sources are read per frame through `import::load`** (cached on
  path + mtime) by `apply_file_source`, in the lowering's per-node loop and
  in the shutter callback at each sample's own time. Our convention on the
  way in: Z-up files turn −90° about X; USD lengths are tenths of a unit
  (`metersPerUnit`, 0.01 when absent); a USD Xform holding a Camera is the
  camera's, and only prims with ops of their own are transforms. The
  importer tests include what Blender 5.2's USD exporter writes.
- **Blender sync is the same importer.** A `Source::Blender` reads the JSON
  the Blender node's script writes on a `scene3d` request
  (`export_scene3d`: cameras, empties, 4-vertex meshes, world matrices per
  frame, Z up). `blender_step` asks whenever `Live::scene` moves and a
  Camera/Card names that node, and registers the file
  (`scene3d::set_blender_scene`, offset = −the node's offset).
  `--test-blender` ends with it against the fake bridge;
  `engine/tests/blender_scene3d.rs` (ignored) runs a real Blender.

## Blender node (`engine/src/blender.rs`, `app/src/blenderui.rs`)

Blender as the comp's 3D system, LIVE: wobble drives Blender's playhead,
Blender draws the scene camera and hands the pixels back through shared
memory, the comp's pictures go the other way as named images. A first-class
node (`NodeKind::Blender`), not an addon — the Blender SIDE is an addon
(`engine/src/blender/wobble_blender.py`, also the launch script). README
has the user-facing story; what must stay true:

- **Pixels never touch disk or an encoder.** `blender/shm.rs` is a ring
  both processes map — POSIX shared memory (`shm_open`, `shm::Source::Posix`,
  Python's `multiprocessing.shared_memory`) on Unix, so the pages have no
  file behind them and the kernel can never write them back (a file
  mapping DID: 35 MB a frame of dirty pages at 2K, flushed to the SSD on
  the OS's schedule); a temp file only on Windows (`Source::File`). A
  frame is announced on the socket AFTER the write completes (a syscall
  is the fence), and a slot lapped mid-copy is dropped by its sequence,
  never shown torn. macOS shm names are 31 chars at most: `wob_<port>_<n>`. The Python
  `RingWriter` mirrors the layout byte for byte — change one, change both,
  bump `VERSION`. **On macOS the ring IS a GPU buffer** (`gpuimport.rs`:
  `newBufferWithBytesNoCopy` over the page-aligned mapping, imported
  through `wgpu::hal`), so a frame is ONE blit from the shared pages into
  the layer and no CPU copy anywhere in wobble; Blender writes float32
  rows padded to 256 bytes, bottom-up, straight off its readback (no
  convert, no flip — the shader flips), and a slot is HELD until wobble
  acks it (the blit's `on_submitted_work_done`, plus the cache copy a
  helper thread makes of a settled frame), a frame being skipped rather
  than torn when every slot is held. Sampling float32 needs
  `Features::FLOAT32_FILTERABLE` (`device.rs` takes it where offered);
  without it the CPU path converts to half on the way in. Measured through
  the app: a 2K viewport frame draws in 4ms and a seek lands in ~0.2s
  round trip.
- **The node is a LAYER, not a reader.** Frames land in a per-node GPU
  texture (`Compositor::blender_upload`, the Paint/Geo precedent) stamped
  with the Blender frame and scene version; `SourceBinding::Blender`
  samples it. No reader slot, no player, no time map: the node always
  shows the latest picture, and the panel says in words whether that is
  the frame the playhead asks for. A `GraphBake` below it is what turns
  it into files — both bake loops wait on `blender_ready` (settled frame,
  right tick, current scene) before writing a frame.
- **Everything about the frame is derived from the stamp.** Settled
  frames are cached per node per scene version (`blenderui::Live`); a
  scrub back uploads from the cache and sends `cached: true` so Blender
  moves its playhead without drawing; any edit in Blender bumps the
  version (`depsgraph_update_post`, pixel-moving updates only) and empties
  it. The scene versions are in BOTH bake keys (`blender_versions()`).
- **Two timelines, no echo.** A tick IS a comp frame; the node's `offset`
  maps it (`frame_of`/`tick_of`). Blender's own moves arrive as
  `frame_changed` and become `seek_to`; ours go out as `frame`, and the
  script skips a `frame_set` to the frame it is already on, so neither
  side ever answers its own move. `Sync` picks who follows.
- **Inputs go TO Blender and are never sampled here.** `inputs_unsampled`
  lists the node like Geo's field port; `blender_export` (END of the
  redraw, so this tick's plates are in the reader textures) renders each
  wired port as the view target, reads it back (`read_output_rgba16`) and
  writes the wobble→Blender ring, keyed on (subtree key, tick). It is the
  one readback on a per-frame path and the node is RED for it.
- **The colour space is the only topology.** `BlenderParams::topo_tag`
  hashes `space` alone (an input transform is WGSL); every other knob is
  a `spec` message, idempotent, re-sent when it changes.
- **Cost: viewport = Slow, final = Heavy**, and nothing on the node clears
  it. Second entry in `cost::tests::decode_cost`.
- The offscreen draw needs a 3D viewport in Blender's window (it borrows
  the first one's space; overlays off for the draw, restored after) and
  runs with `do_color_management=False` — VERIFIED exact scene-linear on
  5.2.1. **One draw is the whole viewport render** (measured: every
  sample, a second draw byte-identical), so a frame is settled the moment
  it exists and `samples` is the viewport's `taa_samples`. **Film >
  Transparent does not reach the viewport draw**; `draw_background=False`
  is what makes the world alpha 0. **Cycles has no offscreen viewport
  render** (RENDERED shading draws black): the live picture under Cycles
  is the material preview, `final` gives the Cycles pixels. A final render
  is `INVOKE_DEFAULT` (Blender's UI stays alive) read off a Viewer node
  the script wires to whatever feeds the user's compositor output.
- **The native module is the zero-copy path, and everything else is its
  fallback.** `blender_native/` (a pyo3 `cdylib`, stable ABI, built with
  the workspace as `libwobble_native.dylib`; `install_script` writes it
  beside the script as `wobble_native.abi3.so` when `native_module_path`
  finds it) reaches Blender's offscreen texture through three symbols
  Blender's binary EXPORTS but does not promise —
  `GPU_offscreen_color_texture`, `MTLTexture::get_metal_handle`,
  `GPU_finish` (mangled names in `Syms`) — resolved with `dlsym` at load;
  any missing one makes `available()` false and the script never asks.
  The `GPUOffScreen*` is read out of the Python object right after its
  `PyObject_HEAD` (type name checked first). `SurfaceRing` is N global
  IOSurfaces (`kIOSurfaceIsGlobal`: deprecated as insecure, but the only
  way another process finds them by id without a mach channel); a frame
  is `GPU_finish` + one Metal blit + `waitUntilCompleted`, then the
  `frame` message carries the stamp on the wire (no ring descriptor). The
  app imports them once per `surfaces` message (`import_surfaces`:
  `IOSurfaceLookup` → `newTextureWithDescriptor:iosurface:` →
  `hal::metal::Device::texture_from_raw`) and answers `surfaces_ok`; a
  `false` sends Blender back to the ring. Ring, CPU and surface frames all
  end in `blender_layer`, so nothing downstream knows which path fed it.
  Z stays on the ring (its scale is applied on the way in). Measured
  through the live test: the texture round trip fell from 514 to 168 ms.
- **The eye is one record both ways** (`blender::ViewState`: Blender's
  `view_location` / `view_rotation` / `view_distance` + lens; the orbit,
  pan and dolly maths in `blender/view.rs`, turntable like Blender's,
  unit-tested). The script builds the projection — a 36mm sensor fitted
  to the larger side and DOUBLED for a free view, so a 100mm free lens is
  a 50mm camera's field — VERIFIED against `RegionView3D.window_matrix`
  (PERSP and ORTHO; an ortho `view_matrix` omits the distance
  translation). A frame change fires `frame_change_post` and NOT
  `depsgraph_update_post` (measured), which is why a scrub in Blender
  never empties the cache while an edit does. wobble's drag steps go out
  flagged `moving` (one-sample draws); Blender's own reports are ignored
  while `Live::nav_active`, or the move would come back one step late.
- **Passes are outputs, lowered like LumaKey bands.** `BlenderParams
  .passes` gives the node `1 + n` outputs; `blender::expand` (called next
  to `lumabands::expand` in `to_engine_graph_at_f`) puts each consumed
  pass on a transient node of its own (`pass: Some`, `host: Some`, both
  `serde(skip)`) whose emit samples the HOST's layer for that pass —
  layers are keyed `(node, pass id)` and the ring stamp carries the id
  (`Pass::id`, fixed forever: a setup names it). Blender draws each pass
  with `View3DShading.render_pass` set (the view-layer `use_pass_*` flag
  enabled first, or the pass is black) — viewport mode only; a final
  render's viewer node is the combined picture. Scalar passes (Z, AO,
  shadow) are `PortKind::Data` on their socket (`output_kind_at`) and on
  the lowered node (`output_kind`). **Z is the mist pass** (the viewport
  has no Z): the script sets the world's mist linear over the clip range
  and multiplies by it on the way into the ring (float32 offscreen, or
  depth loses its precision). **`fat`** makes output 0 `Channels`:
  `EditorGraph::channel_source` accepts a fat Blender node as a source,
  `Gfx::channel_layers` lists its passes by `Pass::key` where a Read
  lists EXR layers, and a ChannelSelect off it lowers (in
  `to_engine_graph_at_f`, before the Read branch) to a pass node of the
  host — so a Breakout harp works on Blender passes with no reader
  slots at all.
- **Cards ride with the camera** (`wobble_card_N`, parented, `CARD_DIST`
  + port units in front): back the camera off and the card comes with it
  — which is what the live test had to learn.
- `engine/tests/blender_live.rs` is the real thing (ignored; needs
  Blender + a display; kills its own Blender at the end because the dirty
  scene makes `quit` stop at the save dialog). Its `wait` scans the WHOLE
  poll batch before returning: a `view` rides in the same batch as the
  hello and an early return dropped it.

## Scripting API (`api/`, `app/src/apiserve.rs`)

The protocol is ONE table, `api::methods()`; the server, the MCP bridge
(`wobble-mcp`) and the Python client (`clients/python`) all read it, so a
method is added in exactly two places: a row there and an arm in
`apiserve::api_call` (`the_app_implements_every_method` fails otherwise).
`blender_session` / `blender_status` are the precedent for a method about
state that lives on the app (a session), not on a node.
Calls are drained at the TOP of `ui()`, before the cost flow, so a script's
edit is coloured and drawn the frame it lands. A mutating call goes through
`api_recorded` (one undo step; a pending hand edit is recorded first).
`viewer_capture` and `pixel_sample` are deferred (`api::DEFERRED`): they wait
`SETTLE_PASSES` UI passes AND for the shown tick to reach the playhead before
reading back, or the answer is last frame's picture. The headless `--serve`
waits on the `doc` crate split. Rules the table keeps:

- **A method is read-only, an edit, or an ACT** — `mutates` (one undo
  step), else `api::ACTS` (moves the viewer, writes a file, starts work:
  not undoable, not read-only), else a read. `DESTRUCTIVE` marks the ones
  a client should confirm. The MCP bridge turns these into tool
  annotations; a new method that acts and is left off `ACTS` is
  auto-approvable by an agent's client, which is the bug the lists exist
  to prevent.
- **Work that takes time is `node_run` + `jobs_get`**, never a call that
  blocks: `node_run` makes the SAME call the node's panel button makes
  (`start_write`, `start_graph_bake`, `start_track`, …) and returns; a new
  run button is an arm in `api_run` and a row in `api_jobs`. It never
  pays (GenAI `Confirm`), downloads or installs — those stay clicks.
- **`batch` is all or nothing**: it runs inside `api_recorded` (one undo
  step) and restores the graph on the first error. Dives, undo/redo,
  `setup_load` and the deferred calls are refused inside one.
- **An expression set by script is validated first** (`api_expr`): a path
  that names nothing is an error there, because at render time it is a
  silent 0.

## Curve editor (`curves.rs`, `curveops.rs`, `anim.rs`)

- **Extrapolation, Stepped and vector handles live on the CURVE, read in
  `AnimCurve::eval_*`** — never in the editor — so playback, the engine
  graph, motion paths and the properties rows all see them. `pre`/`post`
  (`Extend`) are part of the keys' own curve (`eval_base_at`); `stepped`
  wraps keys AND noise (`eval_at`). A vector handle is derived at
  evaluation (`in_handle`/`out_handle`): read handles through those, never
  `tan_in`/`tan_out` directly. Hold ends, no steps, no vectors serialize
  exactly as before.
- **A key move carries the WHOLE key** (`Grabbed` holds `Key`s;
  `apply_moved` restores the snapshot, takes the grabbed keys out and
  `curveops::place`s the moved ones), so interpolation and handles travel
  — the old drag re-`set_key`ed values and dropped them.
- **Normalize is view-only**: the editor works in SHOWN values (`disp` /
  `undisp`, `yk`, `sy` per curve); `norms` freeze while a gesture runs.
- **The plot's response id is fixed** (`curves-plot`) and the active-key
  row is always drawn: an auto id shifted when the row appeared on a click
  and egui dropped the drag (`--test-curves` caught it).
- Selection ops are pure in `curveops` (sliders, smooth, decimate, unwrap,
  mirror, vector) and unit-tested there; G/S/R maths is `xformed`.

## Subwob panel composer (`tool_layout_body`, `compose_drop`)

- **The composer's grid is `cols × CELL_W` wide** — the same size
  `tool_panel_ui` allocates. It once read `tiles × CELL_W`, a third of it,
  and everything about it broke at once (overflowing window, dead drags,
  clipped boxes).
- **A drag follows the pointer per pixel and snaps on RELEASE**; the
  whole body is the move grip (the widgets are disabled in there), the
  side bars scale. `tool_panel_ui`'s `live` arg draws the dragged control
  at its live rect.
- **ctrl-drag a properties row onto the composer** (inside a subwob) and it
  becomes a knob: `HudSlider` (the viewer drag-out's payload) → Slider /
  RGB Slider / Colour Wheel (`HudSlider::wheel`), seeded with the row's
  value and range, cabled with `patch.link`. A drop on a taken cell packs
  the NEW control, never the old one. The Grade wheels grew the same
  ctrl-drag pull; it also answers from the disc and dial, because egui
  hit-tests on last frame's widgets and a ctrl that goes down on the press
  frame finds no grip yet (`--test-compose` failed 3/3 on exactly that).

## Addons (`app/src/addons.rs`)

Manifesto rule 6's first real step: `plugins/<name>/addon.json` declares
commands + panels and may `run` a process, which drives the app through the
scripting API above (`addon_events` long-poll in, ordinary calls out).
Rules that are easy to break:

- **Enabled is opt-in, in prefs** (`Prefs::addons_enabled`). Enabling runs
  code on the machine; never auto-enable one, and suites (default prefs)
  start none.
- **The app draws addon UI from the manifest; a process never draws.** A
  new row kind is a `RowDef` arm + `panel_rows` arm + README line.
- **`addon_events` is held, like `viewer_capture`** (`ApiServe::polls`),
  answered in `api_drain` the pass an event lands or at its deadline.
  Slider events coalesce while queued (`Addon::push`), and the queue is
  capped, so a process that stopped listening costs nothing.
- **Tilt-menu ids are `&'static str`**: addon items are INTERNED
  (`addons::intern`), so a rescan does not leak a new copy each time.
- Without `--api`, an enabled addon with `run` opens the API on port 0 and
  the port reaches the process as `WOBBLE_API_PORT`.
- Next: addon hotkeys need `keymap` to take rows at runtime; viewer
  overlays need a declarative shape list; Tab search should list commands.

## GenAI node (`genai/`, `engine/src/nodes/genai.rs`, `app/src/genaiui.rs`)

A generative-AI pipeline as a node: a small step graph (Prompt, Reference,
Matte, Generate, Comfy, Output) carried as DATA on the node and edited by
a DIVE (`DiveKind::GenAi`, the Particles/SDF pattern: `genaiui::to_editor`
/ `from_editor`, inner kind `NodeKind::GenAiStep(Op)`, its own add-menu
world `NodeCat::GenAiStep` / `NodeEditor::in_genai` / `editor::offered_in`;
step index + 1 is the node id going in, id order is the step order coming
out, takes never move), run on a click, its takes
played back through a reader slot like a GraphBake. `.agents/genai.md` is
the design note (relay protocol, what is next). The `genai` crate has no
egui, no wgpu and no engine; `http.rs` is the only file that knows wasm
(a stub transport there — the app needs no cfg of its own).

- **A one-model node is the same `NodeKind::GenAi`, not a new kind.**
  `GenAiParams::model = Some(catalog id)` and its pipeline is
  `Pipeline::for_model` (sources → Generate → Output), fixed: no innards
  (`innards::provider` returns None), no model menu, `node_label` is the
  model's label. The add menu gets one entry per `genaiui::offered_models()`
  row, which is why `NodeEntry::make` is an `Arc<dyn Fn>`, not a fn
  pointer. A wired String swaps Prompt for PromptIn only in
  `genai_runnable` (the run's copy, and what staleness compares) — never
  written back. `genai_unrun_upstream` refuses a capture through a GenAI
  node with no take (it would pass the plate through and bill for that).
- **Depth and normals are DATA takes.** `Task::is_data()` (Depth,
  Normals) → `Take::data` at run time → `GenAiParams::data()` →
  `idt_fn` reads it as `Working` (no transform) and `topo_tag` changes.
  A model whose picture is not `images[0]`/`image` names its field in
  `Dialect::output_key` (MoGe `normal_map`, MiDaS `depth_map`/`normal_map`).
  MoGe's `depth_map` is a colourised visualisation: never offer it as depth.

- **An upscaler RESIZES the node.** `Pipeline::resizes()` (Output fed by
  a `Task::resizes()` model: Upscale, VideoUpscale) makes
  `node_resolution_walk` answer the loaded take's own size, so
  `emit_loaded`'s stretch is 1:1; everything else still fills the input's
  format. `Pipeline::reference_mp` raises the capture cap to
  `UPSCALE_REF_MP` when an image upscaler is among the steps. Upscaler
  dialects (`Topaz`, `Clarity`, `SeedVr`, `Recraft`, `TopazVideo`,
  `SeedVrVideo`) each have their own request arm; Real-ESRGAN keeps
  `scale` (its keys must not move). `VideoUpscale` is NOT `takes_clip()`:
  its video port may be fed by a video step, not only the captured clip.
  `genai/tests/live.rs` (ignored, free) asks fal's pricing API about every
  catalog endpoint; run it after adding rows.
- **The node has a FOURTH input, the second picture** (`genaiui::SECOND`
  = 3, after the prompt's 2): `Op::Second` reads it, captured and sent
  like the image (`Inputs::second`), keyed on its bytes. A catalog port
  named `"reference"` gets an `Op::Second` in `Pipeline::for_model`
  (`Task::EditRef`). A ready-made node over a general model is a DIALECT
  with a fixed instruction (`Dialect::GeminiRef { instruction }`,
  `Dialect::instruct`), not a new field on every catalog row.
  `Task::Outpaint` resizes like an upscaler; Bria Expand's canvas is
  computed from the picture's real size at run time (`run::expand_canvas`;
  the request builder only has the comp's, which is what a quote uses).
- **A 3D take is a FILE the node cannot draw.** `Task::is_3d()` models
  output `Pipe::Scene` (only Output accepts it) and their take is
  `result.ply` (`Take::is_splat`, named by the task: the download has no
  useful content type). `genai_show` never loads one; on landing
  `genai_take_to_splat` adds a `NodeKind::Splat` of the path beside the
  node, once per capture. LICENCE: a Splat node carries no licence tag, so
  only catalog rows with `commercial: true` may be `is_3d()` until Splat
  can carry one (rule 10; `no_3d_model_is_research_only` holds the line).
- **Templates are STEPS, nothing else** (`genai::templates`,
  `app/src/genaitmpl.rs`). Built-ins are code (`BUILTIN`, held by a test to
  "plans, every port fed, every model known"); the user's are JSON files in
  `~/.wobble/genai-templates` (`$WOBBLE_GENAI_TEMPLATES`), which is also
  how one is shared. Applying one replaces `pipeline.steps` and keeps the
  takes. Each built-in this build can run is also an add-menu entry making
  a wired GenAI Pipeline node; a template NAME is therefore a node label,
  and `editor::catalog_tests` will tell you when one steals a search.
- **A fan-out is N ordinary runs** (`genai::vary`, `GenAiParams::vary`,
  `app/src/genaivary.rs`). `Vary::expand` gives the pipeline plus one
  variant per ticked model (same TASK only; unknown ids skipped); each is
  planned, cached, licensed and recipe'd as a single run would be.
  `Run::more` holds the extra `(Pipeline, Plan)`s; `run::quote_many`
  concatenates the quotes in variant order and `genai_confirm` cuts the
  approved prices back apart by `Plan::billed` lengths: keep those two in
  step. `run::start_many` walks the variants on one thread
  (`Event::Take` for all but the last). An edit to `vary` makes a waiting
  quote stale. `GenAiParams::commercial` asks the fan-out's models too.
  The recipe is now written by the run thread, per variant, not the app.
- **The history is the takes, as pictures (`app/src/genaihist.rs`, a child
  module of `genaiui`).** Thumbnails decode off the UI thread, once per
  file, as sRGB code values (no tone curve: a take is display-referred).
  A take's RECIPE (the steps that ran) is written beside its result as
  `recipe.json` when the run lands (`genai::history`), never into the
  setup; "settings" puts it back, as an edit, and runs nothing. A
  one-model node restores only seed, options and typed prompt, and
  refuses another model's recipe. "load" / "feed" make a Load of the
  take's file (`Action::TakeToLoad`); feed wires it into input 0.
- **Nothing runs or bills without a click, and the quote comes first.**
  `run…` = capture + `run::plan` + `run::quote`, all free; only "pay & run"
  calls `run::start` with the APPROVED per-step prices, and the relay
  refuses a job priced above them. A plan with nothing billed runs
  straight through. Never add an auto-run, a per-frame evaluation or a
  retry that re-bills. `--test-genai` counts the stand-in's submits.
- **The cache key is the anti-rebill.** `run::plan` hashes, per step,
  everything that shapes pixels (model id AND endpoint, seed, size,
  scale, extra JSON, the reference's PNG bytes, input keys) and nothing
  that does not (position, the licence flag). A result dir is complete
  when `meta.json` exists (written last, renamed into place). A field that
  changes the request but not the key re-bills AND serves stale results;
  one that changes the key but not the request re-bills for nothing. Bump
  `KEY_VERSION` if the recipe changes.
- **A setup names catalog ids, never a URL or a key.** Endpoints, tasks
  and licences live in `genai::catalog`; keys, route and the ComfyUI
  address in `genai::config` (`~/.wobble/genai.json`, 0600, env wins) —
  not in a setup, not in prefs. New endpoint = new catalog id.
- **Licence is recorded on the TAKE.** `GenAiParams::commercial()` =
  picked take's `commercial` AND the pipeline's models (`cost::licence_of`
  reads it). Re-pointing a step at a commercial model must not clear a
  take made by a research-only one; `cost` and `nodes::genai` tests pin it.
  Comfy defaults to NOT commercial (we cannot read a licence out of a
  workflow); the user's checkbox asserts it.
- **Capture is a one-frame GraphBake** (`BakeSink::Capture { port }`): the
  bake loop renders the node's INPUT at the playhead, hands tight
  working-space RGBA to `genai.captured` (a field write — `pb` holds
  `self.playback`) and writes no file; `finish_graph_bake` calls
  `genai_capture_finished`. Do not add a second render path. Working ->
  Rec.709 -> `shrink` to `REF_MP` -> sRGB PNG; a matte takes red.
- **Codegen**: loaded (transient, `apply_bake_state` sets it like `baked`)
  = the take STRETCHED over the node's format (one slot: the format) and
  the sRGB input transform on UNPREMULTIPLIED colour (cutout edges);
  unloaded = input 0 passes through. `topo_tag` is source + loaded only —
  a prompt is not code.
- **A take landing is an undo step** (one generation = one action — unlike
  a tracker's per-frame poll); a cache-hit re-run re-picks the existing
  take rather than duplicating it. `genai_adopt` (end of `adopt_loads`)
  re-points every GenAI reader after a load/undo.
- **Video takes** (`Task::ImageToVideo`, Seedance): only Output takes a
  `Pipe::Video` (`Op::accepts`); the take is an mp4 played as a movie
  placed at `Take.tick` (`placement()`); offered, run and loaded only
  where `genaiui::video_ok()` (a movie build) — else the panel says so in
  red. Seconds/resolution join a step's key ONLY for video models, so
  every still's key is unchanged (no `KEY_VERSION` bump); keep it so.
  Request quirks per model live in `catalog::Dialect`, not in endpoint
  string tests.
- **Clips** (`Op::ClipIn` / `Op::MatteClip`, `Task::VideoEdit` /
  `VideoInpaint`): the image / matte input over `Pipeline::clip_seconds`
  from the playhead, on the `Pipe::Video` pipe. Captured by the SAME
  capture bake over N ticks (`genaiui::CLIP`/`MATTE_CLIP` capture ports =
  input + 10); each frame is converted ON ARRIVAL in `Runs::take_capture`
  (Rec.709, box-averaged to `CLIP_SHORT` = 720 short side, even sides,
  8-bit) — never keep float frames per tick — then
  `engine::movie::encode_mp4` makes the upload. Release FFmpeg is
  decode-only, so clips are offered only where `genaiui::clip_ok()`.
  A model's clip limits (`Dialect::check_clip`) are refused in the PLAN,
  before anything is sent. Masks come from outside (Roto/Paint/Sammie into
  the matte input); pixel nodes inside the dive are not built.
- **Known cost**: a loaded GenAI is not `inputs_unsampled`, so its
  upstream stays in the rig and is decoded when only the GenAI is viewed.
  Making it unsampled needs `sample_through` during capture (the Blender
  precedent). For the same reason it does not `clears_slow`.
- Tests: `cargo test -p genai` (unit + `tests/services.rs`: fal queue,
  422 reasons, cancel reaching the provider, relay cap + charge, ComfyUI
  uploads/placeholders, all on localhost); `genaiui::tests` (inner graph,
  panel fit incl. keys open); `genai_panel_render` (ignored) draws the
  real panel to `target/genai-panel-gpu.png`.

## CorridorKey node (`engine/src/nodes/corridorkey.rs`, `app/src/corridorkey.rs`, `app/src/nodes/corridorkey.rs`)

Corridor Digital's neural keyer as a node. README has the user-facing
story. The engine never runs the network: the node is a READER (slot,
`placement` from `start`, `loaded` transient like a GraphBake's `baked`)
and the app does the work on a click. What must stay true:

- **Three phases, all explicit.** `corridor_start` → a `SubtreeBake` with
  `BakeSink::Corridor { port: 0 }` (the plate, converted working →
  LINEAR_REC709 on the way out exactly as a Write does, into
  `<dir>/Input/f_NNNNN.exr`) → `corridor_render_finished(0)` → port 1
  (the wired hint into `<dir>/AlphaHint`, or the plate again with `hint:
  Some(ScreenKeyParams)`, which the bake loop splices in as
  `corridorkey::HINT_NODE` and views) → `corridor_render_finished(1)`
  spawns the process. Files are named by COMP FRAME, so `first_frame`
  becomes `start` and the key plays where the comp does.
- **The tool is driven by `tools/wobble_corridorkey.py`** (embedded,
  written next to the checkout every run, run with `uv run --project`
  from the checkout because the tool's checkpoint dir is RELATIVE to its
  cwd). It calls the published `create_engine` / `process_frame` API with
  `input_is_linear=True` and writes the `processed` result (linear
  Rec.709, premultiplied, despilled, despeckled) — the reader leaf's IDT
  is `Explicit(LINEAR_REC709)` in `codegen::idt_fn`. Protocol: `WOBBLE
  frame i n` / `WOBBLE done` / `WOBBLE error …` on stdout; everything
  else is a log tail. `$WOBBLE_CORRIDORKEY_CMD` replaces the command
  (the suite points it at `wobble --corridor-fake`).
- **The directory is the recipe.** `corridor_dir` hashes both inputs'
  subtree keys, `CorridorKeyParams::recipe()` (screen, despill, refiner,
  despeckle — NOT the output pick, which is a uniform) and the range. A
  changed recipe is a new dir and the old key keeps playing, flagged
  STALE; `corridor_adopt` (end of `adopt_loads`) loads a finished dir for
  the current recipe on open/undo. `mark_baked` goes on `Processed/`.
- **Nothing installs or downloads without a click.** `install…` is
  `git clone` + `uv sync` on a thread; weights are fetched by the tool on
  its first run and the button's hover says so. Suites never touch it
  (`installed()` is true under `$WOBBLE_CORRIDORKEY_CMD`).
- **Licence: the pixels are free, the tool is not.** CC BY-NC-SA with an
  explicit commercial-output grant, so `licence_of` stays Permissive and
  the panel links the licence; do not bundle the tool, do not serve it.
- A loaded key `clears_slow` (it is frames on disk) and is
  `inputs_unsampled` — like a GraphBake, unlike a GenAI take.

## Sammie node (`engine/src/nodes/sammie.rs`, `app/src/sammie.rs`, `app/src/nodes/sammie.rs`)

Click-to-roto: SAM 2 (via Sammie-Roto 2's checkout) as a node, on the
CorridorKey pattern — a READER (slot, `start`, transient `loaded`), the
work a process on a click, the process plumbing (`spawn`, `Msg`, `uv`)
shared with `corridorkey.rs`. README has the user-facing story.

- **The driver imports only `sam2` and `matanyone`, never `sammie.*`.**
  Sammie's app is GPL-3.0; its vendored model packages are Apache-2.0 and
  S-Lab. Weights come from running `sammie/model_downloader.py` as a
  PROGRAM. `the_driver_never_imports_sammies_gpl_code` pins it. Do not
  "simplify" by importing Sammie's managers (they also need Qt).
- **Points are normalised plate coords on the node, per comp frame**, and
  the WHOLE click set is the recipe (sorted, so click order is not a re-run)
  — a new click is a new run dir and the loaded matte goes STALE. The gizmo
  measures a point drag from the press (`insert_temp` at `drag_started`),
  never summed deltas.
- **SAM 2 backward needs a FRESH inference state.** Reverse propagation
  over a state the forward pass filled came back all-empty (measured);
  `prompted()` rebuilds it per direction.
- **Licence is per-instance, and survives the switch-off.** `matting:
  MatAnyone2` tags the node; `nc_take` (AUTHORED, saved) records that the
  LOADED matte was made with it, set on every adopt from the run's recipe.
  `cost::licence_of` reads `SammieParams::commercial()`.
- **Internal resolution is MATTING's only** (`matting_res`, Sammie's
  `matany_res`): the driver cuts the shorter side to it before MatAnyone and
  scales the matte back. SAM 2 always runs at 1024x1024, so segmentation has
  no knob. It rides in `recipe()`'s matting field (`matting_key`) only while
  matting is on, and at 0 that field is the old number, so old runs load.
- **Output picks the pipe.** `Cutout` (image, fuses input 0 × matte;
  unloaded = plate pass-through) or `Matte` (mask; unloaded = 0; loaded it
  is `inputs_unsampled` and `clears_slow`). The pick is in `topo_tag`.
- **A preview is one frame through a RESIDENT segmenter, never adopted.**
  `sammie_preview` renders the frame on screen once into
  `frames-<plate key>` (a rendered frame finishes the bake at once) and asks
  `sammie::Server` — the driver in `--serve` mode, SAM 2's image predictor
  loaded once, one JSON line in and `WOBBLE done` out a request, the picture
  embedded once per frame — for `preview-<key>/Matte` (key = plate subtree,
  models, THAT frame's clicks). The mask becomes an egui overlay
  (`preview_texture`: green fill + white outline, because green alone
  vanishes on a green subject), drawn under the points. **Auto preview**
  (`SammieParams::auto_preview`, on by default, not in the recipe): a click
  edit re-asks 80 ms after the last change (`edited` debounce in
  `sammie_poll`); ~0.5 s a click on an M4 once the model is up. The server
  restarts on a model change and dies with the app (`Drop`).
- **A track covers the plate's range cut to the script range — no
  handles** (`sammie_range`): a 67-frame trim of a long clip rendered 247
  with them. `--test-sammie` sets handles and checks the frame count.
- `torch_extra()` picks the tool's uv extra (macos / cu126 / cpu,
  `$WOBBLE_SAMMIE_EXTRA`). Real end-to-end check: the driver on its own
  (`uv run --project ~/.wobble/Sammie-Roto-2 python wobble_sammie.py --shot
  <dir> --model Efficient [--matting matanyone2]` against an `Input/` of
  EXRs and a `points.json`).

## Screen recorder (`app/src/recorder.rs`, `recorder/`)

A record button in the menubar (⌘⇧R; right-click for window / viewer,
pointer, mark, GIF) for clips users post about wobble. What must stay true:

- **Capture is the app's own frames, never an OS grab.** The surface
  texture is copied to a readback buffer just before the submit that draws
  it (`Recorder::capture`), so no screen-recording permission, nothing
  else on the desktop, same on every OS. The surface carries `COPY_SRC`
  ONLY while recording (`redraw` swaps it before acquiring a frame; on
  Metal it costs the framebuffer-only path). Three buffers in flight; a
  redraw that finds none free drops its frame rather than wait on a map.
- **A take is LZ4'd frames on disk** (temp dir, removed with the take),
  box-downscaled to fit 1920x1200 with even sides. The trim is exact and
  the pointer + "Made with Wobble" mark are laid on at SAVE (`overlay.rs`),
  so both can be switched after recording.
- **Encoding is the OS's H.264, never a GPL encoder**: AVAssetWriter on
  macOS (`encode_mac.rs`, has a real-encoder unit test), Media Foundation's
  sink writer on Windows (`encode_win.rs`, compile-checked only from this
  Mac). Elsewhere a user-installed `ffmpeg` on PATH, else the GIF alone.
  The public build's FFmpeg is decoders only and is not involved.
- **Sharing never uploads and holds no accounts or keys** (`share.rs`):
  the OS share sheet (macOS), reveal, copy the FILE to the clipboard, or a
  site's compose page with the caption in the URL (the user attaches the
  file). Clips land in `~/Movies/Wobble` (`Videos` elsewhere),
  screenshots in `~/Pictures/Wobble`, unless the recorder's options name
  a folder (`Opts::clip_dir` / `shot_dir`, saved in prefs;
  `WOBBLE_RECORD_DIR` wins over both).
- **The one upload is the wobble blog's** (`app/src/blogpost.rs`): a
  "blog…" button on the share panel and "submit to the wobble blog…" on a
  Write that has rendered open one dialog (title, about, name, attach the
  .wbl), which sends to the site's own `/api/wobbles` steps — the same as
  wobble.ing/submit/, `Origin` set to the site, `t` = when the dialog
  opened (send stays off for the site's 3s minimum). Nothing shows until
  approved at /admin/wobbles. A Write's EXRs go through the viewer's
  transform on the CPU (`display_rgb`; OpenDRT falls back to Filmic) and
  the recorder's `open_mp4`, a PNG for one frame. The .wbl has the home
  dir written as `~`. `WOBBLE_SITE_URL` points it at a `wrangler dev`;
  `blogpost::tests::live_upload` (ignored) is the end-to-end check.

## Plugin store (`app/src/pluginstore.rs`, `app/src/urlhook.rs`)

User-facing name: **wobbins** (site, dialogs, help menu); code and disk
stay "plugin" so installed folders never move. The website
(`website_wobble`: `public/wobbins/`, old `/plugins/` 301s there, built from `registry/<id>/`
by `build_registry.py`) is for browsing; its install button opens
`wobble://install/<id>`. Rules:

- **A link carries an ID, never a URL.** Bytes come only from
  `pluginstore::REGISTRY_URL` (`WOBBLE_PLUGIN_REGISTRY` overrides for a local
  copy); a bundle's `file` must be relative to it. Same rule as model ids.
- **Nothing installs without a click in the app**, and an addon stays
  disabled (`Prefs::addons_enabled` is untouched). Size + sha256 from the
  index are checked, paths must be plain relative components (no `..`, no
  dotfiles), and the folder is staged then renamed into place.
- **Bundles are JSON text, not zips** — no archive semantics to get wrong.
  `.store.json` in the installed folder is the receipt (version); a bundle
  can never ship it because dotfiles are refused.
- **macOS needs the bundle** (`tools/bundle_app.sh`, writes `dist/Wobble.app`
  with `CFBundleURLTypes`). winit does not surface GetURL, so `urlhook`
  registers an `NSAppleEventManager` handler before the loop AND in
  `resumed`. Windows/Linux get the URL as argv (`main`'s `wobble://` arm);
  registering the scheme there is not done yet.
- `cargo test -p app pluginstore` (4 tests); `live_registry_installs_everything`
  (ignored) runs against a served copy of the site.
- **Accounts + paid wobbins** (`app/src/account.rs`; the site's Worker in
  `website_wobble/src`, tested by `npm test` there). The app signs in with a
  DEVICE CODE (opens `/device?code=…`, polls) and never sees a password or an
  OAuth secret; its token lives in `~/.wobble/account.json` (0600) and is
  revocable on the account page. `TEST_RUN` hides it from suites. A paid
  wobbin has `file: null` in the index and downloads from
  `/api/wobbins/<id>/download` with the token; the site checks ownership.
  **Buying happens on the website only** — the app opens the wobbin's page.
  Prices live in D1 (from `registry/<id>/meta.json`); the page's copy is
  display. Downloads are gated, the files are not DRM'd, and say so.

## One build, runtime consent (`app/src/report.rs`, `tools/release_public.sh`)

There is ONE distributed build. Alpha testing (crash reports, logs, session
stats) and the new-build check are switches the user answers on first run
and in prefs → "alpha testing + updates"; both are OFF until answered
(`Prefs::seen_consent`), and off means no request at all. What must stay true:

- **Money is the one compile-time cut.** The `commerce` feature (app
  default, `genai/relay` with it) holds accounts + sign-in, paid wobbins,
  prices and tip buttons, and GenAI relay credits. `release_public.sh` builds
  `--no-default-features`, so none of it ships; `engine` and `app` depend on
  `genai` with `default-features = false` or feature unification turns the
  relay back on. Check a release with `strings` for "wobble credits".
- **Sending feedback is its own consent**: the help menu's bug / feature /
  node / feedback forms send with testing off, anonymous unless an email or
  name is filled in. A node request is a `feature` with a `[node] ` title,
  general feedback `[feedback] ` — no worker change needed.
- **The new-build check is anonymous**: `download.wobble.ing/latest.json`
  (the download worker derives it from the newest public zip's name),
  compared by SemVer. THE version is `app/Cargo.toml`'s (`0.1.0-alpha.N`);
  bump with `tools/bump_version.sh`; a clean release tags `v<version>` locally.
- First launch runs welcome → consent → SSD cache window → quick start, and
  a new install starts on right-angle wires and wobble 0.5 (saved prefs
  keep theirs). `WOBBLE_SHOW_CONSENT=1|testing|welcome|cache` puts those
  windows up under a suite, for `WOBBLE_SHOT`.

## Project assets on a Load (`app/src/assets.rs`)

A Load can name an organisicator ASSET instead of a path: `ENode::asset`
(`AssetRef`: organisicator's database id for the indexed file/sequence, plus
the names it had when last seen, for display only). The panel shows project
/ shot / asset and a tree picker (`assets::Picker`), never the path. What
must stay true:

- **Database-driven: the id is the truth, organisicator answers.** It serves
  a read-only API on 127.0.0.1 while running (`organisicator/src-tauri/src/api.rs`;
  port + token in `~/.organisicator/api.json`, `$ORGANISICATOR_API_FILE`
  overrides). `load_asset` asks `/v1/assets/{id}`, writes the path into
  `Read.path` and the fresh names into `asset`, mirrored into `undo_last`
  (never an undo step). `adopt_loads` re-asks for every asset Load, so a
  project moved or a shot/file renamed THROUGH organisicator keeps loading,
  with its range. organisicator not running is an error that says so; never
  fall back to the stale path or a folder scan.
- **Ids survive because organisicator rewrites rows in place**: a project
  move and a Files-view rename `rekey_dir` the index, `instance_rename` /
  `follow_dir_rename` keep a shot's record. A Finder rename does not.
- The picker lists only what the index holds (a project organisicator has
  not indexed says so) and only what `decode::is_supported` opens.
- The addon's "add plates" looks the plates' ids up through the same API
  (`shots.fetch_asset_ids`); `node_create` takes `asset: {id}`.
- **A plain Load can be INGESTED** (`ingest…`, the picker's destination
  mode, `assets::ingest`): organisicator's `POST /v1/ingest` COPIES the
  footage into the picked shot's folder (never moves it, never
  overwrites), indexes it and answers its id; the Load then names that id
  (one undo step) and keeps its range. The copy runs on a worker
  (`start_asset_ingest` / `poll_asset_ingest`). Cross-process check:
  organisicator's `serve_a_fixture_until_killed` + wobble's
  `assets::tests::live_ingest_then_resolve` (both ignored).

## Object storage reads (`engine/src/objstore.rs`)

`s3://bucket/prefix` is a directory to a Read; README "Reading from object
storage" is the user-facing story. What must stay true:

- **Everything goes through `vfs`, and `vfs::Reader` is an enum.** Decoders
  never see a URL: `vfs::open/read/exists/list/is_dir/is_file` route an
  `s3://` path to `objstore`, and `Path::is_dir` / `is_file` are FALSE for
  one — anywhere a typed path may be remote asks `vfs` (`load_read` does).
- **Fetch what a read touches, in `UNIT`s, and cache it on disk.**
  `RemoteReader` is `Read + Seek`; a seek is free, a read makes the units it
  covers resident (`Cached::ensure`, one ranged GET per run of missing
  units, readahead doubling to `MAX_SPAN` while reads stay sequential). The
  EXR decoder's filtered part read is what makes this selective — do not
  add a "read the whole object first" path, and do not shrink `UNIT` under
  a chunk row. `engine/tests/objstore.rs` pins one part of two costing
  under 3/4 of the file, and a re-read costing zero requests.
- **The cache entry is keyed by ETag + size**, so an overwritten frame is a
  new entry, never a stale mix. HEADs are trusted `POSITIVE_TTL` (30s) /
  `NEGATIVE_TTL` (2s) — the short negative is the render-arriving case.
- **The signer is in-tree and pinned** to AWS's `get-vanilla` vector
  (`sigv4_get_vanilla_with_date_header`). Only `host`, `x-amz-date` and
  `x-amz-content-sha256` are signed; `range` is not (it need not be).
- **A setup carries the `s3://` path only.** Endpoints and keys come from
  `~/.wobble/storage.json` + `AWS_*` env (`Config::load`); tests inject a
  `Config` with `set_config`, which also drops every in-process cache. The
  file needs the engine `serde` feature (the app has it); without it env is
  the whole config.
- The app crate is `cfg(not(wasm32))` for all of this: ureq has no wasm
  build, and a page's files are `vfs`'s in-memory table anyway.

## Dockable panels (`app/src/dockui.rs`, imfram)

The main window's graph, viewer, properties and animation are tabs of
imfram's dock (the project's shared egui crate, a pinned git dependency — see
**imfram** under Build, run, test). The main
window is an "external" host: `Gfx::ui` draws the menubar and hands the
rest to `Docks::draw_external` (`dock_ui`). **The viewer tab is
`see_through`**: the picture is blitted to the surface BEFORE egui, so a
filled leaf body paints over it (the dock port shipped exactly that — a
blank viewer). **The timeline is a tab too** (`Pane::Timeline`),
a STRIP (`imfram::dock::Tab::strip`): a leaf of strips keeps its height
in a split above or below it whatever the window, and that split has no
divider. A prefs layout from before it was a tab gets it above the
properties on load (`dockui::add_timeline`). A tab dragged out, or detached from its right-click menu,
gets a window of imfram's own, drawn by `Docks::frame` from
`App::window_event`; the asks (detach, move, close) are taken up in
`about_to_wait`. `Gfx` is the `Panels` impl: each tab body is the old
pane's drawing (`pane_ui`, `viewer_ui`). Q / W (`viewer_mode`) show one
leaf alone (`Docks::solo`); V (`props_visible`) opens or closes the
properties and animation tabs; `bottom` and the dock's active tab are
kept in step both ways. The layout is saved in prefs (`dock`) on exit
and restored on the first turn.

`WOBBLE_SHOT=<png>` with any `--test-*` run writes the interface as
drawn (the picture itself left out) after 20 frames and exits: the way
to look at the window without touching the user's.


## Refract node (`engine/src/nodes/refract.rs`, `app/src/nodes/refract.rs`)

A plate or a latlong seen through bumpy glass. Input 0 is what is behind
(`RefractMode::Background`: a plate; `Latlong`: an equirectangular
environment through a camera, Env's convention and Env's `skyview` rows),
input 1 the surface (a height map read as its slope, or a tangent-space
normal map), input 2 a Camera on the scene pipe, lowered away by
`refract::lower` next to Env's.

- **It fuses, and must stay fused.** Dispersion and roughness are ONE WGSL
  loop around `Emit::input_at`: each tap has its own index (its place in
  the spectrum) and its own nudge of the normal (the R2 sequence, so the
  disc's radius does not march with the spectrum). The tap count is a
  uniform; the loop body is one copy of the background's subtree whatever
  the count. Unrolling it would multiply the shader by the count.
- **Only the mode and the map's kind are topology.** Everything else,
  conventions and the tap count included, is a uniform.
- **Flat glass is the identity**: the plate where it stands, and under a
  latlong exactly what an Env shows (in through the bumpy face, OUT
  through a flat back face). `engine/tests/refract.rs` pins both against
  `RefractParams::lookup`, the CPU mirror.
- **Thickness is Background-only.** A parallel shift of a ray reads the
  same sky at infinity; the panel disables the row and says why.
- **Pixels scale with the proxy at pack time**: thickness, and a height
  map's relief (a half-res slope is twice as steep per pixel).
- **Format**: the map's under a latlong (the glass is the frame), the
  plate's otherwise (`node_resolution`).
- A normal map must be loaded raw; the port's tooltip and the panel say so.

## Roto dive + axes (`engine/src/roto.rs`, `app/src/rotodive.rs`)

The Roto is still `NodeKind::Spline`; its shapes, combine ops, **axes** and
Out list are DATA on `SplineParams` (`more`, `ops`, `axes`, `out`,
`layout`). Mini ids: shapes `0..`, ops `ROTO_OP + k`, axes `ROTO_AXIS + k`
(`RotoRef::of` decodes; never test `id >= ROTO_OP` to mean "op" — axes are
above it too).

- **Enter dives in** (`DiveKind::Roto`). The body is built by
  `rotodive::to_editor` and written back by `from_editor`, every frame for
  the preview and once on the way out. Inside, a shape IS a plain
  `NodeKind::Spline` (so the gizmo, the pen tool and the panel work
  unchanged), a combine IS a `MatteCombine`, and only Axis and Out are new
  (`NodeKind::RotoNode`, dive-only, never compiled). Shape order = body id
  order. Curves move with the items: a shape's `RotoPt`/`RotoSh` on the
  owner are its `Spline*` inside; an axis's `RotoAx(k, w)` its `Card3d*`
  transform channels. A Spline inside a Roto dive does NOT dive again.
- **An Axis unions what is wired into it and moves it** (`RotoAxis::matrix`:
  T·T(pivot)·R·S·T(−pivot), scene units: frame one unit wide, Y up, the 3D
  cards' convention). Axes nest; the product is applied leaf-ward.
- **Port 1 is a Camera** (scene pipe, lowered away in `roto::expand`, which
  also drops the wire on a plain Spline). With one, shapes are planes in 3D;
  a default Camera sees an unmoved shape exactly where it was drawn.
  Without, the projection is orthographic (axis Z ignored).
- **Projection is per outline SAMPLE**: `roto::expand` puts a transient
  `SplineParams::proj` (`ShapeProj`, serde-skipped) on each minted Spline;
  `codegen::spline_outline` carries every sample through its homography.
  Never apply it to control points (a projected Bezier is not the Bezier of
  projected points). Feather stays in output pixels.
- **The gizmo composes the same homography** into its downstream chain
  (`Gfx::roto_gizmo_h`), so points are drawn and dragged where they render,
  and edits land in authored space.
- Known gaps: `roto::bounds` (the SoftCrop box) ignores the camera; patch
  cables on a shape's channels stay on the owner while dived; the in-node
  `innards` view shows and wires axes but cannot add one (add them in the
  dive).
