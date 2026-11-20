# Wobble — GPU-resident compositing POC

[![CI](https://github.com/Soft-Body-Group/wobble/actions/workflows/ci.yml/badge.svg)](https://github.com/Soft-Body-Group/wobble/actions/workflows/ci.yml)

(repo `wobble`; the app binary is `wobble`)

Proves the throughput story for a GPU-resident compositing engine: EXR
sequence in, one fused compute pass, frame on screen, with numbers for every
stage.

## Layout

- `engine/` — UI-agnostic: wgpu device (native limits), EXR decode behind a
  trait (`exr` pure-Rust backend), persistent staging ring, fused WGSL codegen
  from the node graph, frame scheduler, GPU timestamp instrumentation.
- `app/` — winit + egui shell with a bespoke node editor: nodes are
  geometric shapes (Load = down-triangle, Blend = up-triangle, Transform =
  square, Blur = diamond, Grade = circle, Viewer = slab), flow runs top ->
  down, inputs (cyan) and outputs (gold) sit on stalks emitted from the
  shape. Node editor and viewer sit side by side at full height; properties
  run along the bottom, timeline above them. Drag sockets to wire (grabbing
  a connected input detaches it), right-click to add (grouped into source /
  transform / colour / filter / matte / composite / time / utility submenus)
  or remove, Tab to search the same set by name. Nothing is listed until
  you type, then the best few matches — the search is FUZZY, so the letters
  only have to appear in order and "chsl" finds ChannelSelect, "tfm"
  Transform, "grd" Grade. A literal match always leads a scattered one:
  the name spelled exactly, then prefixes, then names that contain the
  query, then the fuzzy hits. **↑/↓** walk the list (they wrap) and
  **Enter** adds whichever row is marked. A node added with Tab is WIRED
  to the selection: one selected node feeds its input AND the new node
  takes its place downstream — whatever the selection was feeding now
  reads from the new node, so a Tab-add INSERTS into the chain rather than
  hanging off it (ports that would not take its output keep their wires).
  Two selected into a Blend (or a Mix) go in pick order, the first as the
  foreground — so select, Tab, select, Tab chains a comp without touching
  a socket.
  **Ctrl+A** (or ⌘A) selects every node; **⌘C / ⌘X / ⌘V** (ctrl+C / ctrl+V
  too) copy, cut and paste them, wires between them included, landing at
  the pointer — through the system clipboard, so a branch can travel
  between sessions.
  Middle-drag or
  alt-drag to pan, scroll to zoom (0.02×–40×: zoom right in to tiny nodes),
  double-click to re-center. Dragging empty space draws a marquee: selected
  nodes move together. Hold S and drag to scale their spacing about the
  selection's centre (Blender's scale: the pointer's distance from the
  centre over where you pressed; node sizes stay), and hold R and drag to
  rotate the layout about it (the nodes stay upright). A tap of S still
  swaps a node's inputs, a tap of R still frames. ⌘↑/⌘↓
  swaps the selected node with its parent/child in the chain (1-in-1-out
  connections only), exchanging positions with a short glide animation.
  Rewires compile on a worker thread and swap in when ready.

Node set: Load (stored as `Read` in setups and still accepted as `Read` by the API; each owns a sequence-directory path), Transform (affine,
bilinear), Grade (lift/gamma/gain/offset plus an overall saturation, and optional luma zones — see **Grade zones**), ColorCompress (black/white in, lift/mult out — see **ColorCompress**), Tinter (Resolve's ColorSlice: a seven-vector hue secondary, subtractive — see **Tinter**), HueRotate (every colour turned about grey by an angle, luminance held exactly — see **HueRotate**), Gamma (`x^(1/γ)` per channel: a master times r/g/b, alpha its own, negatives passed through, with a mix), Expression (a pixel written as arithmetic, one expression per channel — see **Expression**), ScreenKey (a green / blue screen keyer: matte, despill and premult in one fused node — see **ScreenKey**), Despill (green / blue, average / max / min — see **Despill**), Lightwrap (background light bled into the foreground's edge — see **Lightwrap**), UV Warp (the source sampled where a UV picture points — see **UV Warp**), Lens Distortion (Brown-Conrady undistort / distort with overscan, CA and an STMap out — see **Lens Distortion**), GridWarp (a source and a destination lattice of points, the picture moved from one to the other — see **GridWarp**), Warp (the same with curves — see **Warp**), Blur (separable gaussian, sigma in px),
Glow (see **Glow** below), Blend (+ over −, or replace, with a `mix` back to −), Mix (crossfade two inputs on one slider), MatteCombine (matte algebra — see **Mattes**), Dilate (sub-pixel matte grow/shrink), Noise (a procedural field graph — see **Noise fields**), Viewer. The viewer node marks the render root only — the
view transform is display state and lives at the blit (see **View transform**
below), so the fused pass output is scene-linear. Blur is the first non-pointwise node: codegen splits the graph there —
the blur's input subtree fuses into its own pass rendering an intermediate
texture, fixed H/V passes filter it (sigma is a uniform: size drags never
recompile), and downstream shaders sample the result like a Load. The HUD
reports these "prepass" ms separately (~2.4 ms at 4K for a 10 px sigma). Parameters are edited in the
properties bin (right panel) for whatever nodes are selected in the graph —
click a node to select, shift-click adds, Esc/background click clears. A
Load's `dir` (typed or via browse…) points it at any EXR sequence; loading
rebuilds the playback rig (staging ring resized to the largest plate).
Parameter edits are uniform writes — never a recompile; only topology
changes regenerate WGSL (cache keyed on topology hash).

### Glow

The classic way to get a pleasing glow is a stack of blurs at exponentially
increasing sizes with falling weights. The look is right and the cost is
not: a separable gaussian of radius r is 2r taps per pixel at FULL
resolution, so each doubling of radius doubles the work, and the stack ends
up costing about as much as its widest member twice over. Wobble's **Glow**
keeps the sum of exponentially spaced kernels and changes what "one octave
wider" buys: a mip PYRAMID. Threshold the highlights, then a chain of 13-tap
downsamples (each level half the resolution, so a quarter of the pixels at a
FIXED tap count), then a chain of 3x3 tent upsamples that adds each octave's
weighted contribution back on the way down. Octave i behaves like a gaussian
of radius ~2^(i+1) px; the whole pyramid is ~1/3 of a full-res pass on top of
the half-res prefilter.

The consequence worth internalising: `size` does not buy taps, it only moves
weight further up a pyramid that is already being built. A 500 px halo costs
what a 30 px one does. Measured at 4K on an M-series laptop, the glow prepass
is 2.9 ms — *less* than a third of the 9.8 ms a single 10 px-sigma Blur
costs there, and a sixth of a Fast Defocus at r=32 (17.7 ms).

`size` sets how far up the octaves the weights reach (fading each one in
continuously, so the control is smooth); `falloff` is the shape — the weight
ratio between one octave and the next, 0.5 for a tight core, 1.0 for a wide
soft veil; `threshold`/`knee` pick what glows (the comp is scene-linear, so
1.0 means highlights only); `intensity` and `tint` scale it; `combine` is
add / screen / glow-only. Every one of them is a uniform — the pass chain's
shape depends only on canvas size, so no glow edit ever recompiles, and the
mask input crossfades like Blur's.

### Glint

Star streaks off the highlights. `threshold`/`knee` key what shines (the
same soft knee as Glow, each full-res pixel keyed before it is averaged, so a
lone hot pixel is not lost); `rays` (0 to 8) are spread evenly round the
circle and turned by `rotation`; `length` is in pixels; `falloff` shapes the
tail, 0 full strength to the tip, 1 a short spark; `intensity`, `tint` and
`combine` work as on Glow. Each ray runs at half resolution as four 4-tap
passes at strides s, 4s, 16s, 64s, which together visit 256 positions along
the ray with the exact exponential tail — so `length` costs no extra taps.
Every control is a uniform, the ray count included: the program always holds
eight rays and the unused ones return at once, so rays can be keyed.

### Diffuse

Atmospheric diffusion — mist, haze, a diffusion filter. Each pixel gives up
`amount` of its light to a wide soft halo of its neighbourhood (the Glow
pyramid, run over the whole picture; `size` and `falloff` work as there),
so contrast softens and highlights bloom without the frame getting
brighter: the light is scattered, not added, and in scene-linear light a
bright source dominates its halo, which is what a blur mix never looks
like. `haze` lays a veil of the `mist` colour on top. Wire a depth pass to
**Z** and both ramp from `near` to `far` (defaults 0 and 1, a Blender mist
pass; tick **far is low** for a pass that falls with distance): the near
end stays sharp, and because the scattered light is weighted by the same
ramp and normalised by it, a sharp foreground never haloes out over the
mist behind it. The `~` port is a modulate matte that scales the same
weight by hand. Unwired, depth and modulate cost nothing; wired, one
full-res pass.

### Input formats

`engine::decode::SUPPORTED_EXTS` is the single list the browser filter, the
sequence scanner and the decoder all read, so they can never disagree about
what counts as media.

| format | notes |
|---|---|
| **EXR** | scene-linear, the native case. Named layers selectable per Load; multi-part files are listed but only part 0 decodes. |
| **DPX** | 10-bit packed plus 8/12/16-bit RGB(A), either byte order. Log files ride the Cineon curve to linear in the decoder. |
| **PNG** | 8- and 16-bit, grey/greyA/RGB/RGBA, palettes and `tRNS` expanded. |
| **JPEG** | baseline and progressive, greyscale/YCbCr/CMYK. |
| **Movies** | ProRes, H.264 and whatever else the linked FFmpeg decodes — `.mov .mp4 .m4v .mxf .avi .mkv`. Behind the `movie` feature. |

PNG, JPEG and movies decode to **code values**, not to linear: what the file
stores is what lands in the buffer, and the reader leaf's input transform
does the rest (see *Colour management*). That is deliberate — a decoder that
helpfully linearised sRGB would still produce a picture, so the mistake would
be invisible. The one thing movies DO apply is the YCbCr matrix and the
studio/full range, because the stream states them rather than us guessing.

**Movies are the slow path, and the Load panel says so.** A frame sequence
answers "give me frame N" by opening a file; a container has to seek to a
keyframe and decode forward. Wobble keeps a pool of warm decoder sessions per
file, each remembering the frame it would produce next, so playing forward
costs one `receive_frame` (~1 ms on a 512x288 ProRes here) and only a scrub
outside every warm run pays for a seek (~20 ms). It is still not a frame
sequence: cache to RAM or NVMe, or transcode, if you need to scrub hard.

Movie support needs a system FFmpeg (`libavcodec/libavformat/libavutil/
libswscale`) and is therefore **off by default**, so a stock build has no
non-Rust dependency and CI is unchanged:

```bash
brew install ffmpeg                                  # or the distro equivalent
cargo build --release -p app --features app/movie
```

Without the feature the movie extensions are not in `SUPPORTED_EXTS` at all —
the browser never lists a format the build cannot decode.

### Reading from object storage (`s3://`)

A Load's path can be an object-storage prefix instead of a directory:

```
s3://plates/shots/sh010/plate_v003
```

and it loads like a folder of frames — Cloudflare R2, AWS S3, MinIO,
Garage, anything that speaks the S3 API. Nothing else in the app knows: the
decoders read through `engine::vfs`, and an `s3://` path is served there by
`engine::objstore`. What makes it usable rather than a download button:

- **Byte-range reads.** Object storage has no `seek`, but a `GET` can ask
  for a range. The reader fetches the bytes a decode actually touches, in
  1 MiB units, growing the fetch while reads stay sequential (to 32 MiB).
  The EXR decoder reads the header and offset tables, then seeks to the
  chunks of the one part it wants — so a Load of one layer of a 20-AOV
  multi-part pulls that layer's bytes, not the file. The test pins it: one
  part of a two-part file costs under three quarters of the object.
- **An NVMe cache of what was fetched.** Every unit lands in
  `~/.wobble/objcache` (one sparse file + a bitmap per object, keyed by
  ETag so a re-rendered frame is a new entry, 20 GiB budget, least recently
  touched evicted first). The second decode of a frame makes no request;
  a second *process* makes none either. `WOBBLE_OBJCACHE_DIR` /
  `WOBBLE_OBJCACHE_BYTES` move and size it.
- **A frame that is not there yet is checked again**, exactly as a local
  render-in-progress is: a `HEAD` that found nothing is trusted for two
  seconds, one that found the object for thirty.
- **Setups carry the `s3://` path and nothing else** — never a key, never
  an endpoint — so a comp moves between machines and each machine's
  credentials stay on it.

Credentials and endpoints live in `~/.wobble/storage.json`
(`$WOBBLE_STORAGE_CONFIG` overrides the location), with the usual `AWS_*`
environment variables winning over the file:

```json
{
  "endpoint": "https://<account>.r2.cloudflarestorage.com",
  "region": "auto",
  "access_key": "…",
  "secret_key": "…",
  "path_style": true,
  "buckets": {
    "public-plates": { "access_key": null, "secret_key": null },
    "aws-archive":   { "endpoint": null, "region": "eu-west-2", "path_style": false }
  },
  "cache_dir": "~/.wobble/objcache",
  "cache_bytes": 21474836480
}
```

`endpoint` omitted means AWS itself (`https://<bucket>.s3.<region>.amazonaws.com`);
with an endpoint, requests are path-style (`endpoint/bucket/key`) unless
`path_style` is false. No key = anonymous requests (a public bucket).
Requests are signed with AWS Signature V4 (the signer is pinned to AWS's
published test vector). Env: `WOBBLE_S3_ENDPOINT` or `AWS_ENDPOINT_URL`,
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`.

Not yet: the sequence browser does not walk buckets (type the `s3://` path
into the Load), and Writes are local only. `cargo test -p engine --test
objstore` runs the whole path against a throwaway S3 on localhost, and

```bash
cargo run --release -p engine --features serde --example s3probe -- s3://bucket/prefix
```

is the first thing to run when a Load of an `s3://` path fails: it prints
where the bucket resolves, whether requests are signed, and what LIST /
HEAD / GET came back with. (`--features serde` is what reads
`storage.json`; the app has it, a bare engine example does not.) The
server side for testing — a local S3, Postgres and an asset registry —
is the separate `wobble-server` repo.

The browser (**L**) opens full-window. Its right sidebar lists every file
type found — in the folder, or across the whole tree when `recursive` is on
— with counts; untick a type (say `.png`) to drop it from the listing, on
top of the `supported media` / `all files` base filter, the duration floor
and a width floor (`at least N px wide`, read off each file's header on a
worker — a clip stays listed until it has been measured). Rows select the
way any list does: click one, shift-click a range, ⌘/ctrl-click to toggle,
⌘A for all; **add N selected to bin** in the bottom bar puts the selection
in the bin (the `+` on a row or tile still bins one). **Space** over a clip
(or on the selected one) opens a quick look: the frame at full window size
with a scrub strip, ▶ to play it there, arrow keys to step, `+ bin` / `use`
to take it; Space or Esc puts it away.

Viewer: drag (left or middle) pans, scroll/pinch zooms about the cursor,
double-click refits. Selecting a Transform node shows an on-viewer gizmo:
drag the pivot to translate, the corner squares to scale (uniform, about the
pivot), the knob on the stick to rotate. **Ctrl-drag the pivot to move the
pivot itself**: the point rotate and scale turn about goes where you put it
and the picture stays exactly where it was (translate is re-derived to
compensate, the way a classic centre handle works).

## Routing dots

A **dot** is a bend in a wire: it routes a noodle where you want it and is
free at render time (the engine never sees one). **Hold `.` and click a
wire** to drop one there. **Hold `.` and drag** while placing it to set its
**tension** — how far the wire's curve reaches through the bead. The handle
reach follows the pointer: close in pulls the wires into a sharp corner,
further out swings them wide, and a faint ring marks the automatic curve
(tension 1, which the pull snaps to). Hold `.` and drag an existing dot to
change its tension later. Shift-click on a wire, or "cut a dot in here" on
its right-click menu, also drops a dot at the automatic curve. Tapping `.`
on its own does nothing: the click on the wire is the gesture.

## Right-angle noodles

The graph bar's **right angles** toggle draws every noodle as straight runs
joined by rounded corners, never a diagonal. A wire leaves its output
downward, runs across halfway, and drops into the input; one whose input sits
level with or above its source hooks round underneath instead of cutting
back up. Side sockets (a Grade's matte) are entered sideways, and expression
cables run sideways too. **Dots become the corners**: route a wire down to a
dot, across to another, and down again, the way dots are usually used (a
dot's tension only shapes curves). The slicer, `.` dot cuts and dropping a
node onto a wire all follow the drawn route, so what you click is what you
see. Runs that would lie on top of one another — four branches dropping
into a Layer along one line — are spread into **lanes**, like lines on a
subway map: one lane per output, ordered so the wires do not cross on their
way in and out, and wires from one output still share a lane (a line that
branches). **Tidy** in this mode leaves taller gaps between rows, growing
with the number of noodles that jog through each gap, so the lanes have
room. The mode is saved in prefs (`graph_ortho`).

## Expression links in the graph (L)

Press `E` (or use the graph's right-click menu) and the node view becomes a
patch bay. The image wiring steps aside, and every node splays its animatable
channels on tentacles — **inputs left, outputs right** — long enough, and
spaced far enough apart, that individual nodules stay pickable on a Grade's
twelve as readily as on a Blur's one. Drag output to input to patch a value,
grab a driven input to pull its cable off, ctrl-drag to slice cables the same
way the slicer cuts noodles. A driven input and a driving output both burn
amber, so what is actually patched reads at a glance against what merely
could be; the channel's name rides its nodule when there is room, and always
on the one under the pointer.

It is the same patch model as the properties rack (`EditorGraph::patch`), so
the two views edit one thing. Cables whose other end is a value node (mult,
expression, ...) live on the rack and are not drawn here — the rack is where
those get built.

## Expressions typed into a field

The third way to patch a value is to TYPE it, Houdini-style. Click any
parameter's value field and type an address:

```text
../grade12/gainr * 2
```

`../node/param` reads another node's parameter, `./param` one on the same
node, `ch("../node/param")` is accepted for Houdini hands, and `$F` (or `t`)
is the frame. The usual arithmetic and functions apply (`+ - * / % ^`, `sin`,
`clamp`, `mix`, `smoothstep`, …). As soon as the text is more than a number
the field hands over to an editor under the row:

- **Enter** sets it (whenever the text reads as it stands), **Esc** cancels,
  emptying it takes the expression off.
- **shift+Enter** starts a new line. A line `name = …` names a value for the
  lines below, the LAST line is the result, `#` starts a comment, and a line
  ending in an operator (or inside brackets) continues on the next:

  ```text
  key = ../grade12/gainr      # the key light
  key * 2 + sin($F / 10)
  ```

- **Tab** completes the address under the cursor, as a shell's menu does:
  first as far as every candidate agrees, then STEPPING through them, each
  written into the text as it is reached (shift+Tab steps back). **Enter**
  takes the one you stopped on, and sets the expression when that makes it
  whole — so Tab, Tab, Enter on a node offers its params, and Tab, Tab,
  Enter on a param sets. Half an address and Enter finishes it with the
  highlighted candidate; the arrows move that highlight. Only addresses
  that exist are offered.
- Addresses in the text are coloured by whether they resolve, and a line
  that cannot be read is named in the error under the text. An expression
  that cannot be read is never set.

A node is named by its own label when it has one (`Hero Key` is `hero_key`),
otherwise by its type and id (`grade12`). While the editor is open the graph
labels every node with exactly that name and rings the ones the text reads.
Right-click a row's label for "expression…" and "copy path" — the way in for
rgb rows, whose lanes take no typing.

It is the same patch model again: the text lives in an `expr` value node
wired to the channel, so it shows on the rack, undoes, saves, and is refused
by everything a cable is refused by (`app/src/chanexpr.rs`).

## Motion blur

Motion blur is a NODE, and it blurs what is upstream of it. `MotionBlur`
(Filter menu) evaluates its input's whole branch at several sub-frame
moments while the shutter is open and averages them; everything downstream
runs once, at the frame, on the averaged picture. Put it under the moving
element and above the grade over the finished shot, and only the element
smears. That works because of the invariant the whole engine runs on —
params are uniforms, so N sub-frame samples are N uniform writes and N
dispatches of the SAME pipelines, never a recompile.

The shutter is the node's: `samples`, `angle` (open time as a fraction of a
frame — 0.5 is a 180 degree shutter, animatable) and `offset` (0 centred,
-0.5 closing on the frame, +0.5 opening on it, animatable). Samples land at
the CENTRE of each slice of the open shutter, so the ends of a smear carry
the same weight as its middle.

Only the node's own upstream is re-dispatched per sample (the compositor
walks the plan's stage dependencies), and a stage that feeds ONLY a resolved
MotionBlur is not run again at the frame. Accumulation happens in a storage
BUFFER in f32 — a read-write storage texture needs an adapter feature
rgba16float does not universally have, and a long shutter sums a dozen HDR
samples. The resolve writes into the node's own stage texture, so the fused
pass samples it like any other stage result.

It is a slow path, and says so: red at a few samples, GLOWING from eight,
because its whole upstream is rendered that many times over. Nested
MotionBlurs are resolved upstream first; the inner one is held at its
frame's average while the outer one samples through it.

Two things it deliberately does not do. A comp with nothing animated opts
out, since paying N times for N identical renders is pure waste. And a
moving PLATE does not smear: which frame a reader decodes is a fact about the
whole tick, and there is no half frame on disk for a sub-sample to land on —
blurring a retime would mean decoding between frames, which is a different
feature. Motion IN the plate is what **VectorBlur** is for (next section).

**A trap worth knowing if you touch this:** each sample must be its own
submit. `queue.write_buffer` is not ordered against commands recorded into an
encoder — the writes are staged and flushed before the command buffer runs —
so recording all N passes into one encoder renders N copies of the LAST
sample's params. The failure is invisible: every sample runs, the frame looks
blurred, it just blends the same moment N times.

## Motion vectors: Flow, VectorBlur, VectorWarp

**Flow** (AI menu) solves optical flow: per-pixel motion vectors for its
input. Two engines, both ported to WGSL and run on the GPU with no Python
or torch, picked in the panel:

- **SEA-RAFT** (default; [princeton-vl/SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT),
  BSD-3, ECCV 2024) — real two-frame optical flow, trained and judged on
  flow. The M model, on the checkpoint trained on TartanAir, FlyingChairs
  and FlyingThings only (`searaft-m`, 79 MB). About a second a frame pair
  at 960×540 on an M4 Pro.
- **RIFE** v4.6 ([hzwer](https://github.com/hzwer/Practical-RIFE), MIT) —
  a frame interpolator's flow: lighter weights (`rife-4.6`, 21 MB), rougher
  at edges and occlusions.

On the planar suite's synthetic plate the forward vector at the centre
lands 0.02 px from the true motion with SEA-RAFT, 0.57 px with RIFE. The
weights come through the model gate at the top of the panel; nothing
downloads until you press it, and only the picked engine's are asked for.

Press **analyse** and the node renders its input over the input's frame
range (handles included), solves the motion between every pair of
neighbouring frames, writes the vectors to disk and plays them back
through its reader slot, like a bake. So the analysis is the slow path and
it is explicit; once it has run the node costs what a Read costs. The
result is keyed on everything upstream and on the settings: an edit shows
as STALE and nothing re-runs behind your back. Unanalysed, the node is
zero motion everywhere and reads nothing.

What comes out rides its own VECTOR pipe (yellow wires), in canvas pixels, y down (the
viewer's own pixel space): `rg` is FORWARD motion (where the pixel is one
frame later), `ba` BACKWARD (where it was one frame earlier). Settings:
**detail** (full / half / quarter: analyse at that fraction of the size —
half suits 4K) and **backward** (a second solve per frame; off, `ba` is
`-rg`). Under proxy the vectors are scaled on the way out, so consumers
never have to know.

Anything that reads a picture can use the vectors — an Expression or a
Shader node reads them like any input. Two nodes are built for them:

- **VectorBlur** (Filter) — motion blur along the vectors: each pixel
  averages the picture along its own motion across the shutter, the forward
  vector after the frame and the backward one before it. `shutter` (0.5 =
  180°, animatable), `offset` (where the shutter sits, animatable),
  `samples`, and a matte on port `~` that scales the shutter. One pass, red
  in the graph. It smears what moved IN THE PLATE — the thing MotionBlur
  cannot.
- **Morph Dissolve** (Composite) — a transition between A and B: each is
  warped toward the other along the optical flow between them while they
  cross-dissolve. Key `mix` from 0 to 1 and press `analyse`, which solves both
  directions at each frame where `mix` is between its ends. Unanalysed it is a
  plain dissolve; once analysed, moving `mix` or any slider solves nothing.
- **VectorWarp** (Transform) — the picture sampled where the vectors point,
  `amount` frames along them (+ forward, − backward). Amount 1 over the NEXT
  frame (a TimeOffset of +1 upstream) pulls that frame back onto this one:
  the aligned neighbour a temporal denoise or a clean-plate fill wants. It
  fuses, like UV Warp.

### Rifetime: slow-motion with made frames

**Rifetime** (Time menu) slows a shot down with REAL in-between frames. A
Speed node at 0.5 can only repeat frames or cross-fade two, which ghosts
anything that moves; Rifetime renders its input on a click and, for every
output frame that falls between two source frames, runs RIFE at that
fractional time and warps both neighbours onto it along the motion. The
network judges the motion on the sRGB-encoded picture; the pixels are
warped from the linear, unclamped frames, so highlights survive. Frames
that land on source frames are copied as they were.

**speed** is source frames per output frame (0.5 = half speed, twice as
long; above 1 it speeds up), anchored on the input's first frame, and the
timeline, the viewer and a Write below all see the longer shot. Or switch
**time** to **curve**: a keyable **source frame**, TimeRemap's contract —
output frame T shows source frame `frame(T)`, so ramps, freezes and
reverses are a curve shaped in C. Curve mode keeps the input's range; the
curve is sampled at every frame when you press render, and editing it
shows the node STALE until you render again. **detail**
judges the motion at a fraction of the size (the frames are always made at
full size). Like Flow, the result is keyed on everything upstream: an edit
shows as STALE and nothing re-runs on its own. Uses the `rife-4.6`
weights.

## Stabilize — the camera's shake, measured and taken out

A **Stabilize** node (transform menu) measures how the camera moved and
undoes it. Wire a plate in and press **analyse**: the node renders its
input over the input's range, runs SEA-RAFT (the Flow node's network and
weights — nothing new to download) between every pair of neighbouring
frames at half size, and fits ONE global motion to a grid of that flow
per pair — translate, similarity, affine or perspective (**fit**). The fit
is robust: it starts from the median move and re-weights with Tukey's
biweight, so a subject walking across the frame is outvoted rather than
averaged into the camera. Wire a matte into the second port (a Sammie
cutout of the actor, say) and the pixels under it do not vote at all.
The pairs chain into a camera path, kept on the node and saved with the
setup.

Everything after that is a few numbers a frame, evaluated at the playhead,
so changing it never re-runs the analysis:

- **lock-off** warps every frame onto a reference frame (**set to
  playhead**; the first analysed frame by default) — a camera on a tripod;
- **smooth** keeps the move and loses the shake: a Gaussian over the path,
  **smoothing** frames wide;
- **remove x / y / rotate / scale** pick which parts of the motion are
  taken out (lock-off with all four holds the reference exactly, perspective
  included);
- **zoom** scales about the centre to push the edges the warp drags in off
  the frame;
- **reverse** puts the shake BACK — exactly the inverse of the same node
  with the same settings: stabilise, do the work (paint, a sign
  replacement), and reverse it after, or reverse a CG element onto the
  plate.

The correction is a corner pin applied in the fused pass, like a
CornerPin: one resample, no pass of its own, so the node stays cyan. Two
Stabilizes in a row (stabilise, then reverse) compose their sample
positions before anything is sampled, so the round trip gives the plate
back with no second resample. Edit anything upstream and the panel says **STALE**; the old
path keeps working until you re-analyse.

Measured on a synthetic 960x540 shot under a known hand-held-style shake
(36 frames): the solved path within 0.26 px of the truth, lock-off holding
the true camera to 0.26 px at every frame; with half the frame matted out,
0.50 px.

For a tracked region rather than the whole frame, the planar tracker and
the multi-point Tracker have a **stabilise** drive button: it cables the
track into a CornerPin's FROM corners, and the region holds still.

## Noise fields

A second data type, and a second kind of graph. An image port carries pixels
that have already been decided; a **field** port carries a *function* —
`f(p, seed) -> f32`, callable at any coordinate its consumer likes. That
distinction is the whole paradigm, and it is what makes a parameter drivable
by another noise: if an fBm's scale is itself a field, and a Warp above it
rewrites the coordinates its input is read at, then a node cannot be
evaluated once and cached. The same node may be called twice in one pixel at
two different coordinates. So every field node compiles to its own WGSL
function and consumers CALL it: pixels flow down the graph, but the
coordinate and the seed flow UP it.

A `Noise` node owns a whole field sub-graph and is where that collapses back
into pixels. Inside it: **fBm** (fbm / turbulence / ridged, sharing the
image-side fBm's noise exactly), **Voronoi** (F1, F2, F2-F1, cell id),
**Coord** (x, y, radius, angle), **Const**, **Math**, **Mix**, **Remap**,
**Warp** (offset, or the divergence-free curl of a potential field),
**Seed**, and **Gradient**. Every numeric parameter on those is a literal
that an input can override — wire a field into `scale` and the literal stops
being read.

Three properties fall out of the design and are worth relying on:

- **Nothing is rasterised on the way.** A field is pointwise, so a whole
  sub-graph collapses into the fused pass with no intermediate texture and
  no bandwidth. A deep noise graph costs ALU, not memory.
- **Params stay uniforms.** A sub-graph's numbers live in one contiguous
  block reserved by the Noise node, so turning a knob deep inside a field
  graph is a param write. Only rewiring recompiles, exactly as everywhere
  else here.
- **No recursion.** WGSL has none, so the graph must be a DAG; the compiler
  refuses a cycle rather than emitting a shader that cannot build.

### Masking one

A **Roto** node draws a closed spline IN the field graph — the comp Spline's
outline, in field units. Select it and click in the viewer to draw, exactly
as with the comp Spline; the same gizmo drives both. Because it is a field it
is read wherever the graph asks, so a Warp above it bends the roto's edge, and
its offset and feather are sockets — a roto whose position is itself noise
is a perfectly good mask. Its points are authored (edit them in the viewer),
not keyframed, in this first version.

A **Shape** node draws a mask IN the field graph — an ellipse or a box, with
a centre, a size and a feather — so a region can be defined where it is used
rather than round-tripped out to the comp and back. Every one of its numbers
is a drivable socket like any other, so the shape can be moved, resized or
softened by another field: a circle whose radius is itself noise is a
perfectly good mask.

The other kind of mask comes from outside. `Noise` takes a matte, and inside the graph a **Mask** node reads it as a
number per pixel. That is the only thing in a field graph that comes from
outside it, and it is what lets a field change only where you say: mix a
coarse feature size toward a fine one through the Mask and the noise is
finer inside a spline and untouched outside it. "Apply it where the mask is
1, leave it as it was where the mask is 0" is exactly a Mix with the Mask as
its blend amount.

The matte is sampled at the PIXEL, not at the field coordinate, so a Warp
further up the graph moves the noise THROUGH the mask rather than dragging
the mask along with it — a mask is a region of the picture. Unwired it reads
1 everywhere, the convention every other matte input here follows. The CPU
probe (see **Noise as data**) reads 1 too: the matte is a GPU-side picture
and there is nothing to sample on that side.

### Animating one

`Noise` has a global **speed** in cycles per frame, and every generator
inside has its own multiplier on it: 0 freezes that one node while the rest
of the graph keeps moving, 2 runs it at double. One knob drives the lot and a
single node can still opt out.

fBm evolves along its third axis — the same dimension the image-side fBm
calls `evolve`, in the same noise-lattice units, so a small number is a
normal setting. Voronoi is 2D and has no third axis, so instead each feature
point ORBITS inside its own cell: the pattern boils in place rather than
sliding past.

The tick reaches the field the way a Layer's trim does, resolved per frame
into a transient on the node, because the engine has no idea what time it is.
That also means a noise field motion-blurs correctly for free: sub-frame
samples evaluate it at fractional ticks.

### Authoring one

An **Output** node marks what the graph evaluates to, and an empty Noise
opens with one already in it. The viewer marker inside a dive stays a
PREVIEW — park it on an intermediate node to inspect that node, and what the
Noise renders does not change underneath you.

A `Noise` node is opened the way a subwob is: select it and press ⏎ (or the
button in its panel) to step INSIDE its field graph, and Esc to step back
out. Field nodes are ordinary nodes to the editor — selection, wiring, undo,
the tile layout and the properties grid all work unchanged — which is the
whole reason the sub-graph is edited in the same canvas rather than in a
second bespoke one. While you are inside, the add menus offer field nodes and
nothing else (a Blur means nothing in a field graph, and an fBm field means
nothing in a comp), and the viewer renders THE FIELD, so you are looking at
the thing you are building rather than the comp it will end up in.

Stepping out folds the graph back into the node that owns it. A subwob's
body is one definition shared by every instance of it; a noise field
belongs to its one node and saves with the setup, layout included.

A socket that is wired overrides its literal, so a driven parameter's number
is shown greyed and labelled `driven` rather than left looking live while
doing nothing — the confusion this paradigm invites most.

**Sliders are the node's panel.** Name a Slider and it appears as a widget on
the Noise node's properties panel outside, and as a channel — the same kind a
plugin param is — so the comp's patch bay and link mode (`E`) can drive it
with a cable. An **Input** node is the same channel with no widget: a jack on
the Noise node for data coming in from the comp, nothing to twiddle. Two
nodes sharing a name share the channel.

**One wiring system.** Inside a noise graph there are no patch jacks and no
`E` link mode. A value you want to drive several sockets with — or animate —
is a **Slider** node: a keyframeable float, wired into any socket by an
ordinary edge. That works without conversion because a slider is a special
case of a field, a flat one: a field is one float PER POINT, a slider is the
same float at every point. Socket literals still key (the key dot on every
socket row), but a cable is always an edge. Two ways to drive the same float
was exactly the confusion to avoid.

### Noise as data

`Noise` also has a **probe** and a read-only `noise value` channel: the field
evaluated on the CPU at the probe point. Patch a cable out of that channel
and a noise graph drives any ordinary parameter in the comp — a blur size, a
transform, a grade. `field::eval_cpu` mirrors the shader function for
function (same hash, same fade, same per-mode normalisation) and
`cpu_matches_the_gpu` holds the two to each other, so what drives a value is
what renders. The probe is animatable, so it can be walked through the field
over time.

## Blend and Mix

Blend composites `+` (input 1) onto `−` (input 2): over, replace, add,
multiply, screen, difference. **Replace** does no alpha compositing at all —
A's pixels stand in for B's — so it is the **matte input** (input 3) that
decides where, and unmasked it is simply A. That matte limits every other op
the same way (`mix(b, blended, matte)`), which is how a masked Over lands its
foreground only where the matte covers.

Mix is the other thing "merge" is often asked for and is not: a straight
crossfade between two images on one slider, `mix(a, b, amount)`, with 0 all
of input 1 and 1 all of input 2. The amount animates like any other param.

## 2dini (procedural vector graphics)

A **2dini** node is a third data type: not pixels, not a field, but
*geometry* — points carrying attributes (position, scale, orientation,
colour, id) and primitives that index them, Houdini's model cut down to two
dimensions. Open the node (⏎, like a Noise) to build its graph: Spline
(drawn in the viewer with the roto gizmo), Polygon (a `step` above 1 walks
the vertices as a star polygon — {5/2} is the pentagram), Star, Resample,
Scatter, Displace, Instance (copy-to-points: input 1 is the template, input
2 the points), Transform, Merge, Style, Output. Every shape carries a fill,
a stroke colour and a stroke width in output pixels, and the node's output
is an ordinary premultiplied RGBA image that composites like any other.

The graph is evaluated on the CPU and drawn into a **layer**, on the Paint
precedent: the fused pass merely samples a texture, so nothing in a drawing
is topology — editing a shape re-records the layer and never recompiles —
and a static drawing costs nothing per frame on either side. The layer is
re-evaluated only when the node (or the Noise it reads) changes; the panel
reports what it produced and what it cost — including the raster cost of a
re-record (pixels × segments), flagged when a full-frame, many-vertex shape
would make an *animated* drawing expensive. Big shapes are tiled so their
empty corners and flat interiors skip the segment walk.

Coordinates are the field graph's: x runs 0..1 across the node's format
width, y in the same units, so the roto gizmo serves a geometry Spline
unchanged.

**Scalars are nodes**, exactly as in a noise graph: Slider, Input, Field
In, Math and Remap wire into any numeric socket by an ordinary edge, and a
socket that is wired stops reading its literal (its row says so). Sliders
and Inputs are what the node **exposes**: every Slider is a widget on the
2dini node's panel and a channel — keyable, HUD-draggable, patchable — and
every Input is a jack with no widget, cabled in link mode. Same data-input
system as the root graph.

**Field In** is how a drawing reads a noise: wire a Noise into the 2dini
node's one input (the *field* port) and a Field In node inside samples it at
the point being evaluated — feed it to Scatter's `density` and the points
follow the noise, to Displace's `x`/`y` and the outline is warped by it. Two
Field Ins with different seed shifts decorrelate. Unwired it reads 0.5.

**Stroke** is the Spline's line-drawing twin: always open, drawn as a
line of its own colour and width (a socket, so noise can vary it), never
filled — the Spline is for shapes, the Stroke for lines. Its points are
laid down in the viewer the same way, and the overlay shows the open curve
with no leg back to the start.

**Noise inside the drawing.** A **Noise** node is also offered *inside* a
2dini: a scalar that owns its own field graph, as the comp's Noise node
does. Open it (⏎) to build the noise — fBm, Voronoi, Warp and the rest live
in there, not in the drawing, kept apart the way Houdini keeps a VOP net
apart from its SOPs. Wire it straight into a Displace's `x` or a Scatter's
`density` with no Noise in the comp at all. It is sampled at the point
being asked about with the comp Noise node's own rule (`scale`, `seed`) and
animates with its `speed` against the 2dini's tick; at speed 0 the drawing
holds still however the playhead moves.

**Text** draws a string in a font — the built-in sans (Ubuntu Light) or
mono (Hack), or any .ttf/.otf by file — one closed shape per glyph outline
with its counters as holes, so an "o" is a ring. x/y is the first
baseline, size the em (the canvas is 1 wide), with alignment, tracking,
leading and rotation; everything downstream treats it like any shape
(Resample + Displace for wobbly type, Rigid Body + Physics to drop it).

**Merge** has ONE wide input bar (Houdini's Merge): drop wires anywhere on
it, they land along it in order, and pulling one out closes the gap.
**Boolean** unions, intersects, subtracts or excludes the filled areas of
`a` and `b`; subtract/intersect keep each shape's style, cut open lines,
and treat bare points as the discs they draw (preset: **stencil**).
`examples/jelly_title.wbl` (preset **jelly title**): "wobble" filled with
pulsing circles, unioned into blobs, dropped as soft bodies — baked, press
play. `wobble <setup> --play --dump-after N out.png` grabs a frame N redraws
into playback, headless.
**Export web page…** on a 2dini node writes its play range as ONE static
HTML file: every frame's shapes as packed data plus a small Canvas2D player
that loops it — no WebGPU, no other files, host it anywhere
(`examples/jelly_title.html`). Colours go out as sRGB; the page background
is the `--bg` CSS variable at the top.
Tick **play forever** beside it and the page evaluates the drawing itself
every frame instead (the 2dini evaluator compiled to wasm, `geoplayer/`,
embedded in the page): moving noise never repeats, a baked simulation plays
once and holds (`examples/plant_live.html`). Keyed sliders are frozen at the
in point. The app embeds `geoplayer/geoplayer.wasm` — rebuild it with
`geoplayer/build.sh` after changing the evaluator.
**Transform** has on-screen handles inside a 2dini —
centre moves, the knob on the ring rotates, the squares scale, alt-drag
places the pivot without moving the shape. **Click-through**: inside a
2dini, clicking a drawn shape selects the node that made it (a copy's
maker — an instanced star picks the Star). **Any node can be viewed**:
set the view on a node inside the 2dini and the viewer shows its
geometry, not just the Output's. **Display options**, Houdini-style: while
a 2dini is in play a display bar sits on the viewer's right edge — HUD on/
off, points, point numbers, primitive numbers, normals (out of each shape;
a counter's point into its hole), orientation, outlines, bounding box —
each a toggle, remembered across launches.

**Presets**: the add menu inside a 2dini has *presets* — ready-wired
networks dropped at the cursor: star field, wobbly title, wobbly ring,
rosette, noise dots; and physics ones (unbaked — press bake): falling
blocks, bouncy balls, jelly pile, plinko, falling letters.

**Light and effects.** The 2dini node's **light** tab lights the drawing
in 2D: shapes cast soft shadows and light bounces round them as global
illumination (holographic radiance cascades, ported from graav). A
**Light** node inside marks shapes as lamps: they give off their own colour
times *tint* times *strength*, and may be set not to cast shadows.
*intensity*, *ambient*, *air* (how much light shows in empty space, as
added light), *cell px* (the light grid's resolution: finer is dearer) and
*sky* (what an escaping ray sees). The **effects** tab lays SDF effects
over the drawing: *offset* (grow or shrink every shape), *outline*,
*outer glow*, *drop shadow*, *inner shadow* and *bevel*. Both read one
signed distance field of the drawing, built on the GPU (a jump flood) only
when the drawing changes; every number is a keyable channel and a uniform,
so dragging one never re-evaluates the drawing or recompiles. Light on is a
slow path (a pass chain), paid every frame only while the drawing moves.

**Physics** simulates the drawing. Input 1 is the *bodies* (closed shapes
and bare points), input 2 the *colliders*: closed shapes are solid, open
lines (a Stroke is ideal) are walls a body cannot cross, and the canvas
edges can be walls too. Two solvers: **rigid** (rapier2d — shapes fall,
stack, bounce and topple without deforming) and **soft** (an XPBD particle
ring per body, squish2d's approach — shapes squash and wobble, *blobbiness*
from firm to jiggly). Nothing simulates until you press **bake**: it runs
from the *start* frame for *frames* frames and stores the motion in the
node, so it saves with the setup and plays back per frame as a lookup. Edit
the bodies, colliders or settings and the panel says **stale** until you
bake again. Settings are physical: gravity in m/s² with the canvas 10 m
wide. **You watch it simulate**: the bake runs in the background and the
viewer follows the newest frame, with progress and *cancel* on the panel;
the finished motion lands in the node as one undo step. *preset…* sets up
the whole simulation (falling blocks, bouncy balls, ice rink, moon, zero-g,
jelly pile, water balloons, stiff rubber).

**Rigid Body** and **Soft Body** set physics PER SHAPE, upstream of the
Physics node (Houdini's per-primitive attributes): density / friction /
bounce / pinned, or blobbiness / pressure / friction, plus a start
velocity (m/s) and spin (rad/s). Each socket is read at the shape's centre,
so a Noise can vary them across a scatter; *material…* fills them in
(stone, wood, metal, rubber, ice, cardboard, balloon; jelly, water balloon,
dough, stiff rubber, slime, pillow). A pinned shape never moves but is
still drawn and still in the way. Untagged shapes take the Physics node's
settings. A glyph's counters ride along with a rigid body; a soft body is
its outline.

Scatter is deterministic: an R2 low-discrepancy sequence accepted per
candidate against the density, so lowering the count drops the tail and
changing the seed reshuffles. Points that reach the output as bare points
draw as discs of their `pscale`.

**Demo**: `examples/plant.wbl` (⌘O) is a procedural plant — a stem that
sways in a Noise wind, branches instanced along it on alternating sides
(the far side gets the branch mirrored, so both lift), leaves instanced
along the branches, a flower on top — with seven controls exposed on its
node: *wind*, *stem width*, *branch spacing*, *branch length*, *leaf
spacing*, *leaf size*, *bloom*. Press play and it moves; open the node (⏎) and press
`H` to see the graph as data. The file is regenerated by
`cargo test -p app plant_demo` from `app/src/demos.rs`, so it never drifts
from the nodes it uses.

**Vector HUD** (`H`, or *vectors* in the HUD menu; remembered across
launches): the viewer overlays the geometry *as data* beside
the render — every point as a dot with a tick along its orientation, every
primitive as its outline (closed shapes tinted green, open ones red so an
outline that did not close reads at a glance, bare points as rings of
their `pscale`). Outside the node it shows the selected 2dini's
output, exactly what its layer was drawn from, projected through whatever
sits downstream like any other gizmo. Inside the node it shows the
*selected* node's own geometry (Houdini's display flag): select a Scatter
and you see its points, select the Instance below it and you see the
copies — the output when nothing is selected.

## Plasma (GPU particles: a star's corona)

A **Plasma** node (Source menu) is a particle system whose particles flow
along magnetic field lines off a star: closed **loops** that arch between
two footpoints, and open **streamers** that leave. Add enough of them up
and you have a corona. `examples/corona.wbl` is three of them (surface,
streamers, active regions) added together under a Glow: open it and press
play.

Nothing is simulated. Each particle is worked out on the GPU from its
number and the frame, so any frame draws on its own, scrubbing is live,
and a million particles cost the CPU nothing. Every control is a uniform
and a keyable channel (`plasma.<name>`); none recompiles.

The star is a sphere and the picture is 2D: field lines are rooted on the
sphere and looked at straight on. **depth** 0 keeps every root on the limb,
so the loops are flat arches seen side-on; 1 spreads them over the sphere,
so some lie across the face and the disc hides the ones behind it. Four
pages:

- **star** — the format, *preset* buttons, the disc's *centre* and
  *radius* (pixels), how many *particles*, *seed*, *depth*, *brightness*
  and *colour*, and the star's own disc: *body* (its brightness, 0 = none),
  *limb* (above 0 darkens the edge, below 0 brightens it) and *mottle*
  (granulation that lives on the sphere and boils at *mottle speed*).
  The light is additive; *alpha* says how much of it counts as coverage
  (0 = pure emission, for an Add).
- **loops** — how many *loops*, each a bundle of *strands* nested by
  *thickness* and spread side by side by *arcade* (a tunnel of arches);
  *regions* and *cluster* gather them into active regions; *span* is the
  footpoint separation and *height* the apex (radii), each with a ±;
  *lean* tips them over, *feet* brightens the footpoints, *life* is the
  frames before a loop fades and is reborn somewhere else.
- **streamers** — the share of the particles on open lines, their *reach*
  (radii), how fast their light falls off with height, and *pinch*: how
  hard a region's lines bend together as they climb, closing into a helmet
  with a cusp.
- **motion** — *flow* along the line (pixels a frame), *pulse* (brightness
  waves riding the flow), the particle's *streak* and *width*, *fuzz*,
  *turbulence* (how far lines are pushed, the field's scale and speed; it
  is nothing at the feet, so a loop sways without leaving the surface) and
  *spin* (the sphere turning, degrees a frame).

Presets: **active regions** (the default: big arcades and a fan of open
lines), **surface** (the disc under a carpet of thousands of tiny loops
and a fringe of spicules), **streamers** (long faint rays only) and
**flat arches** (depth 0: a 2D sun with nothing across its face). Two more
make a total eclipse (`examples/eclipse.wbl`): **eclipse** is the
white-light corona — streamers only, a million and a half soft particles
in fine rays gathered into helmets — behind a black disc (the *eclipse*
tick on the star page: the moon, in front of everything), and
**prominences** is a few small pink loops on the limb to add over it. Pixel
sizes are authored for the radius the node starts with: push in close and
the lines want more particles and more width.

## Particles (a general 2D particle system on the GPU)

A **Particles** node (Source menu) is a particle simulation you build
from parts. Open it (↩, or "open" on the dive bar) and it shows its own
graph, the way an SDF or 2dini node does. Particle **streams** flow along
the wires inside:

- **Emit** makes a stream. It emits from a point, a line, a ring, a disc
  or a box, at a rate (particles a frame), with a life in frames, a
  launch speed, a direction and spread, and an optional push along the
  shape's outward normal. **Spots** gather births into hot spots along
  the shape.
- A **Force** acts on every stream that flows through it. The forces are
  **Gravity**, **Radial** (push or pull about a centre, fading to a
  radius), **Vortex**, **Drag**, **Turbulence** (curl noise: swirls that
  neither gather nor spread), **Field Lines** and **Kill** (a circle that
  removes what enters it, or what leaves it). Forces in a chain apply in
  order, so Gravity then Drag runs gravity first.
- **Merge** joins two streams, so the forces after it act on both.
- **Render** draws the streams that reach it as streaks of light: length
  plus stretch times speed, a width, a colour that runs from birth to
  death, and fades in and out. Every Render adds into the same layer.

**Field Lines** places ± pole pairs just under a ring. With follow at 1
it locks each particle's velocity onto its field line. Particles arch
between the poles as loops and leave the unpaired ones (open) as
streamers. A foot greater than 0 moves every newborn particle onto a
pole, so the loops are bundles rooted at their footpoints. **Strands**
sets how many lines leave each pole: many particles sharing one line is
what draws a crisp thread, and 0 gives a haze. **Fan** is the cone the
lines leave in, which sets how tall the loops stand. **Fill** puts some
pairs on the face of the ring. With "each line feels only its own pair"
on, loops are clean arcs and streamers are straight rays. A Render's
**vary** makes a few threads bright among many faint ones.

**Sphere** gives the field depth. At 0 every pole sits on the ring and
the loops are flat arches seen side-on. At 1 the poles sit all over a
sphere of that radius seen from the front: loops stand off the surface,
lie foreshortened across the face, and streamers leave in every
direction including towards you. Tick a Render's "hide what is behind
the disc" and give it the same centre and radius, and the far side is
hidden by the star. This is the Plasma node's depth. The
**corona** preset is the Plasma look built from these parts. Load
`examples/particles_corona.wbl` to see it under a black disc with a Glow.

**It is a real simulation, and scrubbing still works.** State lives on
the GPU and starts empty at `start - preroll` (preroll fills the screen
before the first frame). Births are scheduled by the frame number and
every random number is a hash, so a frame is the same however it was
reached. Every 8 frames the state is checkpointed under a VRAM budget.
When the checkpoints don't fit, the interval doubles rather than the
far end being dropped. Scrubbing back restores the nearest checkpoint
and steps forward. Playing forward costs one step a frame. Changing an
emitter or a force throws the simulation away. Changing a Render only
redraws. Nothing recompiles: the emitters, forces and renders are rows
in a table the compute shader loops over.

**Quality** multiplies every rate and divides every Render's light, so it
adds particles without changing exposure. The simulation runs in the
node's full-resolution pixels, so proxy changes the drawing and never
the motion. The owner's numbers key as `particles.<name>`. The numbers
inside the graph are authored data in v1: they don't key, because a dive
drops inner curves.

## Text — titles, subtitles and per-letter animation

A Text node draws a block of type over its input (or over nothing, unwired).
Its panel has five pages:

- **type** — the words (a new line starts a paragraph), font (the built-in
  sans or mono, or any .ttf/.otf), alignment (left / centre / right, plus
  **justify** for wrapped lines), the **anchor** `position` names (the
  block's top, middle, bottom or first baseline), case (as typed, upper,
  lower), kerning, position, size (the em in canvas pixels), tracking
  (thousandths of an em, as in a design app), leading, **wrap px** (a
  paragraph box; 0 = no wrapping), and the block's rotation, scale and
  italic lean.
- **style** — fill colour and opacity, **weight** (a faux bold, or thinner
  when negative), edge softness, and an **outline** ring with its colour.
- **shadow + glow** — a drop shadow (offset in pixels, softness in ems) and
  an outer glow.
- **animate** — per-glyph **animators**, TextMeshPro / After Effects
  style. Each is a range selector over chars, letters (spaces left out),
  words or lines: a start / end / offset window in percent, a shape
  (square with softness, ramp up, ramp down, triangle, round, smooth), an
  amount, random order, and a **wiggle** that turns each unit's share into
  a random walk over time. What the selection drives: an offset, scale,
  rotation, skew, tracking, opacity, a fill tint, blur and weight. Key the
  **offset** to sweep the effect through the text. "+ add" offers presets:
  rise in (key offset -20 → 100), typewriter (key start 0 → 100), wave and
  jitter. Animators stack in order.
- **path** — the baseline follows a bezier path: S-click in the viewer adds
  a point at the end, drag moves one, alt-drag pulls out its handles
  (mirrored), a handle's dot drags alone, D-click removes a point. Arc,
  circle and wave buttons lay one down round the text. The alignment picks
  where on the path the text starts (its start, middle or end); `offset`
  slides it along (keyable); "turn with it" rotates the glyphs to the path.

Every number is keyable and patchable. Lengths in ems (weight, outline,
glow, shadow softness) scale with the text and share a quarter em between
them; pixel offsets do not scale.

**In 2dini** the Text node inside a drawing shares all of this: its
**animate** tab is the same page, its **style** tab has the anchor,
kerning, italic, scale, weight, outline and drop shadow (drawn as shapes
under the glyphs, every number keyable; glow and soft shadows are the
drawing's effects pages), case / wrap / justify sit on its shape tab, and a shape wired into its **path** socket (a Spline, say) is the path
its baseline runs along (turn with it, reverse, a keyable path offset).
Each glyph comes out as an outline, so 2dini's fills, strokes, gradients,
light and effects style it — an animator's tint and opacity land on each
glyph's fill.

Under the hood each glyph is a distance-field cell in a small per-font
atlas, drawn as an instanced quad into the node's own layer — so nothing
about the text ever recompiles the comp, a still title costs nothing per
frame, and an animated one costs one draw call.

## SDF (distance effects)

An **SDF** node draws with *distance*. Open it (⏎) and its graph is shapes,
ops and effects: every shape is a **signed distance** — at each pixel, how
far it is from the outline, negative inside — and an effect is a question
asked of that number. One spline gives a stroke (`|d|` under half a width),
its glow (light falling off with `d`), a drop shadow (the shape read
somewhere else), ripples running out from it (iso-lines of `d` travelling
with the playhead) and a bevel (the slope of `d` lit as a surface) — all
from the same outline, which is the point of the type.

- **Shapes**: **Spline** — drawn in the viewer with the roto gizmo (S-click
  adds, D-click removes, K-drag sets tension); *closed* it has an inside,
  *open* it is a path and the distance is to the curve itself. **Circle**,
  **Rect** (rounded).
- **Distance ops**: **Combine** (union, subtract, intersect, xor; `smooth`
  melts the seam over that many pixels), **Offset** (grow/shrink),
  **Shell** (the outline as a band — a ring from a disc), **Morph** (A to
  B), **Transform** (move, turn, scale about a pivot), **Displace** (fBm
  wobble on the outline, evolving with `speed`).
- **Effects** (distance → colour): **Fill**, **Stroke**, **Glow** (light:
  zero alpha, it *adds* to what it lands on), **Shadow**, **Ripples**
  (spacing, width, speed in pixels a frame, falloff, inside/outside/both),
  **Bevel** (rim width, light angle and elevation, depth), **Refract** (the
  node's input seen through the shape, bent at the rim), **Ramp** (the
  distance as a colour ramp).
- **Colour**: **Merge** (over, add, multiply, screen), **Image** (the node's
  input at this pixel), **Output** — what the node draws, laid *over* its
  input. Unwired, the node is a generator at its own format; wired, it draws
  over the plate at the plate's format. Input 2 is a mask.

Distances ride the field pipe and colours the image pipe, so a shape plugs
into any number socket too — wire a distance into a stroke's width and it
varies along the picture. **Sliders** and **Inputs** work as in a noise
graph: every Slider is a keyable, patchable row on the SDF node's panel,
which is how the graph animates from outside. Lengths are pixels of the
node's format (a proxy halves them, driven or not).

**Looking inside.** While the node is open the viewer shows it over its
real input; press `1` over any node inside to look at *that* node instead —
an effect over the input, a shape as a **distance map** (teal outside,
violet inside, a white zero line and a contour every 16 px). `1` over the
Output (or nothing viewed) shows what the node will draw.

**Cost.** The whole graph is fused into the main pass — no textures, no
stages, cyan like a Grade — and every shape read at the pixel is evaluated
once however many effects read it. A spline is a walk over its outline
(~200 segments) per pixel per read; Bevel reads its shape three times (the
slope), Shadow once more at its offset, and **Refract fuses the node's input
a second time** at the bent position. Params are uniforms: nothing but the
wiring and a spline's point *count* recompiles.

**Demo**: `examples/sdf.wbl` — a grid seen through a glass blob melted
into a disc, wobbling, with a rim stroke, shadow, glow and ripples; two
controls on the node (*wobble*, *glow*). Regenerated by
`cargo test -p app sdf_demo`.

## 3D cards (Camera, Card, Blend 3D, Render3D)

Pictures on planes, seen through a camera: projection compositing, and
nothing more — no meshes, no lights, no shading. Four nodes in the **3D**
menu, on their own pipe (green **3D** wires):

- **Camera** — translate, rotate, focal length (mm), filmback (mm), near
  and far, all keyable. The filmback's width is fitted to the render's
  width; pixels are square.
- **Card** — a picture in, a plane out. One unit wide before its scale,
  as tall as the picture's aspect makes it (or `aspect` if set), centred
  on its origin, facing +Z. Keyable transform and opacity.
- **Blend 3D** (was Scene) — gathers cards (and other Blend 3Ds) so a big
  setup stays tidy. One bar, like 2dini's Merge: it takes any number of
  wires and grows a free input as they land; their order means nothing,
  because the render sorts by depth.
- **Render3D** — camera into the first port, a Card or Blend 3D into the
  second. Two outputs: the picture, and its **depth** (the distance along
  the camera's −Z to the first hit at least half opaque, the far clip where
  there is nothing) on the data pipe, ready for a Defocus's Z port.

Y is up and the camera looks down −Z; rotations are degrees applied X,
then Y, then Z. The default camera frames the default card exactly, so a
Card wired straight into a Render3D shows its picture untouched.

Each pixel is a ray tested against every card, the hits sorted by
distance and laid front to back, so depth order is exact per pixel and
cards may cut through one another. **Motion blur** (`blur samples` above
1) samples the camera and card poses across the shutter inside the same
pass: the card pictures render once. A Render3D draws up to **16 cards**
(a GPU binding each); the panel says how many it left out. It is a pass of
its own, so its wire runs red; moving a card or the camera is a uniform
write, never a recompile — only adding or removing a card is.

`examples/cards3d.wbl` is a room of cards with a camera dollying through it.

### From files

A Camera or Card's **source** can be a file instead of its own keys: pick
`file`, the file, the object in it, and a frame offset (file frame = comp
frame + offset). A camera takes the object's pose and optics; a card keeps
its own transform ON TOP of the object's (so a two-unit plane from another
app takes a card scale of 2). Three text formats, read in wobble with no
library:

- **.usda** (USD's text form): Camera and Xform prims, their `xformOp`s in
  order, time samples, parents composed, `upAxis` and `metersPerUnit`
  honoured. Blender's File > Export > USD (as .usda) reads as is. Binary
  USD, references and layers are not read.
- **.json**: `{ "fps": 24, "up": "y", "objects": [ { "name", "type":
  "camera" | "xform", "frames": [ { "frame", "matrix" (16, row-major) or
  "translate"/"rotate"/"scale", and for a camera "focal", "aperture",
  "near", "far" } ] } ] }`. `"up": "z"` takes a Z-up scene.
- **.chan**: one line a frame, `frame tx ty tz rx ry rz [vfov]`.

Alembic is not read: send it through Blender, or export .usda. **bake to
keys** samples the file over the script range into the node's own curves
and goes back to keyed; on a Camera, **cards for the rest…** makes a Card
for every other object in the file, gathered in a Blend 3D.

### From Blender

Pick **Blender** as the source and the Blender node to follow: the Camera
takes Blender's scene camera (or a named one), a Card a named plane or
empty. With the Blender node connected, Blender sends its cameras, empties
and four-vertex planes over the frame range — no pixels — and sends them
again whenever the scene is edited, so the cards stay in sync. Frames map
through the Blender node's offset, as its picture does. A plane's own
width is folded in, so a default card fills it; Blender's camera sensor
fit is honoured, its lens shift is not.

## Splat — a Gaussian splat capture through a camera

**3D → Splat.** Pick a `.ply` (the layout 3DGS training writes) or an
`.spz` and the capture is drawn on the GPU through whatever
**Camera** is wired into the node's input — a CameraTrack's solve, a
file's, Blender's. Unwired, it is the default Camera. Output 1 is the
picture (premultiplied RGBA, scene-linear); output 2 is **depth**, the
distance from the camera, the far clip where there is nothing.

- A capture you pick in the panel is **fitted to the frame** once it has
  loaded; `fit to frame` does it again. Translate / rotate / scale place it
  in the scene and are keyable.
- `opacity` and `splat size` are looks; `view-dependent colour` turns the
  spherical harmonics past the first band on or off.
- A `.ply` is turned Y up on the way in (training writes Y down). Colours
  are decoded from sRGB unless `colours are linear`.
- Loading happens in the background. A capture too big for the GPU loses
  SH bands first, then is thinned, and the panel says so — it still draws.
- The depth sort is on the GPU every time the camera moves, and a still
  camera costs nothing. It is still a pass of its own, so the node is
  **red**; a Bake below it clears that.

Training a capture is not something wobble does.

## CameraTrack — a 3D camera solved off the picture

A **CameraTrack** node (3D menu) solves where the camera was from the
footage alone, and hands the result to the 3D cards. Wire a plate in,
set the **lens** — a preset (the iPhone's main, ultra wide, 2× and
telephoto lenses in 35 mm-equivalent terms, the common full-frame
primes), or a focal length in mm on a filmback width in mm, with
**refine** letting the solve correct it; or **guess**, which solves from
six starting lengths wide to long and keeps the best fit — and press
**solve**. The node renders its input over the input's range, runs
SEA-RAFT (the Flow node's network and weights — nothing new to download)
forward and backward between every pair of neighbouring frames at half
size, and follows a grid of features along it — a point whose backward
flow does not bring it home is dropped, and the grid is re-seeded where
it thins. Those tracks go into a structure-from-motion solve of our own:
two frames with parallax initialise it, every other frame is placed
against the points so far, and the whole thing is bundle-adjusted. Out
comes a camera per frame, a point cloud, and a reprojection error in
pixels, all kept on the node and saved with the setup. The picture
passes through untouched (the node is cyan; only the click costs), and
with **points** on the solved cloud is drawn over the viewer through the
solved camera — if the dots sit on the texture they were tracked on, the
solve is good.

Wire a matte into the second port (a Sammie cutout of the actor, say) and
the pixels under it are not tracked: a subject moving on its own would
otherwise pull the camera. The camera has to MOVE — a pan or a lock-off
has no parallax and the node says so instead of inventing depth.

The solve is in its own units: the world is the first frame's camera
(Y up, looking down −Z) and the median point depth from it is 1;
**scale** multiplies that into scene units and re-runs nothing. To put
it in the shot's terms, **pick** points: hold S and click a solved point
over the viewer (picked points draw bigger, in the 3D view too). Three
or more on the floor and **ground** fits a plane to them and turns the
scene so it is the ground, Y up, with the origin on their centre;
**origin** alone moves the origin; two picks a known distance apart and
**set scale** sizes the scene; **card…** makes a Card on the picks — on
their plane, or facing the camera for one or two — sized to cover them
and wired into the Blend 3D your tracked Camera feeds. None of these
re-solve: they are a frame laid over the solve, and **reset align**
takes it off. Press
**camera…** for a Camera node whose source is the tracker: it reads the
pose and lens per frame, follows a re-solve live, and **bake to keys** on
its panel turns the solve into ordinary keyframes. Wire it into a
Render3D with your Cards and the CG sits in the plate.

Two recipes, so a re-solve never re-tracks: the tracks are keyed on the
inputs, the detail and the range; the solve on those plus the lens
settings. Change the focal length (or ask for a guess) and the re-solve takes
seconds off the tracks on disk. An edit upstream shows as STALE, and the old solve keeps
working until you press the button again. Lens distortion is not
modelled: undistort upstream with a Lens Distortion node first.

## Blender (live 3D)

A **Blender** node is a window onto a running Blender. The comp's playhead
drives Blender's; Blender draws its scene camera and the picture lands in
the node, live, with no file in between; the comp's own pictures go the
other way as images any material can use. Blender is the 3D system — a
matte painting positioned against the plate, a look-dev pass over a scene
that is really being lit, a card carrying a comp'd element back into the
render — and the two timelines follow each other.

**Getting a Blender.** `open in blender` launches Blender on the node's
`.blend` (empty = a fresh scene; save it in Blender and the node learns the
path) with the wobble bridge. Or install the bridge as an addon (`addon
file` writes `wobble_blender.py` and copies the path: Blender > preferences
> add-ons > install from disk) and connect from its `wobble` tab in the 3D
view's sidebar to the port shown in the panel — a Blender you already have
open joins the same way.

**What comes back.** `picture`:

- **viewport** — the 3D view drawn through the scene camera at the comp's
  format, colour management off, so the pixels are scene-linear exactly as
  a float render writes them. Refined across draws until it settles; the
  panel says *refining, N samples* until then. `shading` sets the 3D view's
  shading (rendered by default) so Blender shows what wobble gets.
- **final** — an F12 render per frame, read back when it completes. Seconds
  a frame in Cycles, and the node glows for it.

The node is **red** either way: every frame is a round trip to another
process. A **Bake** below it turns the shot into files and clears the red,
and the bake waits for each frame to settle before writing it. `samples`
caps the viewport settle (or the render's samples); `engine` overrides the
file's; `transparent` sets film > transparent so the world is alpha 0 and
the picture drops straight into an over.

**Frames.** A tick is a comp frame; `offset` is added to Blender's frame:
a scene animated 1..48 with 1000 here plays at 1001..1048. Scrub in wobble
and Blender follows; scrub in Blender and wobble follows (`playhead`
picks who follows whom; neither side answers its own move). Frames that
settled are cached per node, so a scrub back over them costs one upload
and no draw — Blender's playhead still moves — and any edit in Blender
empties that cache and redraws the frame you are on, once your hand pauses.

**Textures out.** `inputs` adds image ports; each arrives in Blender as the
image `wobble_in_N` (scene-linear, Blender's primaries) for an Image
Texture node in any material, and is sent again whenever the picture above
the port changes — a render and a readback each time, which is the slow
path this node is red for. `cards` mounts every port on a plane in front
of the camera — unlit emission through the plate's own alpha, fitted to
the frame, one unit further back per port, riding with the camera — so a
plate is behind the scene (or an element in front of it) without building
a material.

**Passes.** Tick passes on the node — Z (depth), normal, position,
emission, environment, AO, shadow, diffuse/specular light and colour,
volume light, transparent — and the node grows one output per pass, in
the order picked, after the combined picture on output 0. Blender's
viewport draws each pass on its own, so a pass is one more ~4ms draw a
frame through the same ring. Z is depth along the camera axis in scene
units (Blender's mist, set linear over the clip range and scaled back,
drawn at float32 so it keeps its precision); Z, AO and shadow ride the
data pipe, so Z wires straight into a Defocus Z port. Tick **fat cable**
and output 0 carries the combined picture and every pass by name, like a
multi-layer Load: wire it into a Breakout (a harp of passes) or a
ChannelSelect. Viewport mode only for now; a final render carries the
combined picture.

**The eye.** `eye` picks whose picture it is: the scene **camera**, or
**free** — the 3D view's own eye, what Blender's viewport shows, not saved
with the file. With `orbit here` on, Blender's hands over the picture in
wobble's viewer move that eye: MMB orbits, shift+MMB pans, ctrl+MMB or the
wheel dollies (the 2D pan of the picture stays on alt+LMB / ⌘-drag). Orbit
in wobble and Blender's viewport follows; orbit in Blender and wobble's
picture follows — the same record travels both ways, and neither side
answers its own move. In camera mode the hands move the camera itself,
and with autokey on in Blender an orbit becomes camera animation. While
the hand moves, Blender draws at one sample and returns to the scene's
count when it stops, which is what its own viewport does.

**From a script.** `blender_session` (open / connect / detach / quit /
refresh) and `blender_status` (what the panel says, as JSON — poll it after
a seek to know the frame has landed) are in the scripting API, so an agent
can drive a session over the MCP bridge.

**Colour.** Pixels come back in Blender's scene-linear space (Linear
Rec.709 unless its OCIO config says otherwise); the node's `input` row is
the same transform a Load has, into the working space.

Pixels never touch a disk or an encoder. With the bridge's native module
(built with wobble, written beside the add-on) the frame never leaves the
GPU: Blender's own texture is blitted into a surface both apps share and
wobble copies it into the node — no readback, no CPU copy anywhere. A
Blender without the module, or a GPU that cannot share, falls back to a
shared-memory ring (one memcpy on Blender's side; on a Mac the ring is
GPU-visible as it is, so wobble still copies nothing). The panel's info
line says which path is serving. A 2K viewport frame draws in about 4ms.

## Viewer manipulators

Manipulators follow the selection (or the nodes X-clicked in the graph, in
pin mode). **Ctrl/cmd-clicking the picture goes the other way** — from the picture
back to the node. The viewer works out which branch put that pixel there
(stacking order AND alpha, so a transparent hole clicks through) and selects
its Transform, centring the graph on it; ctrl shows its HUD, cmd raises its
properties panel, both do both. The same chords on a manipulator's pivot go
straight to that node. D-DRAG on the pivot moves the pivot without moving the
picture. A Layer stack has no transform of its own: each layer is placed by
the Transform upstream of it, so a pick in a stack lands on that Transform
(or on the layer, when nothing upstream places it). Setups from before carry
their layer transforms forward as Transform nodes spliced in on load.

**Every HUD element on screen has a chip** in the column down the viewer's
left edge — one per node showing manipulators, one per floating slider —
and the × on it is where a pin comes off or a slider is dismissed. The dot
at the chip's front is a **swatch: the colour that node's HUD is drawn in.**
By default it shows the node's type colour and the marks use the adaptive
ink (light over dark footage, dark over light). Click the dot or the name
(anywhere on the chip but its ×) to pick a colour, and the node's
manipulators, its floating sliders and their labels wear it from then on —
hover still snaps white. The colour lives on the node
(`hud_color`), so it survives deselecting, undo and a save, and a node with
two chips (manipulators and a slider) has one colour. Right-click the swatch
for the adaptive ink again. Up to fifteen distinct colours can be on screen
in one frame; a sixteenth draws adaptive.

## Colour picker

Every colour swatch (a Grid's line and background, the paint brush, a
2dini style) opens the same picker. Four modes, one value; the last
mode you used is the one every swatch opens in.

- **wheel**: hue and saturation on the disc (red up, the same ring as the
  grade wheels), value on the strip beside it.
- **rgb**: the stored linear numbers.
- **hsl**: hue, saturation, lightness.
- **lch**: Oklch, where equal steps look equal. L, C and H stay as you set
  them. If the working space cannot make that chroma, the colour is pulled
  back to the gamut edge and the lanes **underline** where that is happening.
  Move back and you get your chroma back.

Every lane's track shows what the colour becomes along it, so you aim at a
colour rather than a number. A hue lane is never a grey bar, even when the
colour has no chroma. Click anywhere on a lane to jump there. z / x / c / v
held set the drag rate, the same as every other row, and releasing one never
snaps the value back. Take a hue to zero saturation and back and the hue
survives.

The footer has old | new (click old to restore) and a hex field (exact
sRGB). Hover a swatch or the open picker: ⌘C copies the numbers as
`[r, g, b, a]`, and ⌘V pastes numbers or a hex code. Esc or a click outside
closes it.

**Swatches, shared by the whole app.** Under the sliders sits a palette
that every picker shows. Click **+** to keep the current colour. Click a
swatch to use it: it sets the colour and leaves alpha alone. Shift-click a
swatch to replace it with the current colour, and alt-click to remove it.
The swatch you're on has a ring. Swatches are saved between launches and
mean the same colour in every comp, whatever its working space. There are
up to 48, twelve to a row.

**It is big, it moves, and it resizes in both directions.** It opens at
three times its original size, hanging off its swatch. **Drag its frame**
— the margin around the edge — to move it anywhere; it goes back to its
swatch next time you open it. **Drag its right edge for width, its far
edge for height, or the corner for both** (the corner has a grip), or
pinch / ⌘-scroll to scale it evenly. Width runs the lanes' length, height
their thickness, so a picker dragged wide gets long sliders and one
dragged tall gets fat ones — aim with whichever you need. Double-click a
grip to get the default back. What you AIM at grows; text stays the same
size, and the wheel stays round. Both sizes are remembered between
launches, and near a screen edge the picker comes up smaller rather than
off screen.

## Sensitivity keys: Z X C V B N

**Z, X, C, V, B and N are sensitivity modifiers while you drag a value.**
Hold one during a drag of a slider, number field, rgb lane or viewer gizmo
and the drag slows or speeds up: Z ×0.01, X ×0.1, C ×⅓, V ×3, B ×10, N ×30.
They do nothing else and swallow no key press, so outside a drag the same
letters are free for ordinary hotkeys: in the graph Z and X pin a node's
properties panels / viewer manipulators, B is swipe-blend and N renames; in
the viewer B shows the blue channel.

Viewer keys: `F` frames the image (Home too; with a Paint node selected, held
`F` is the brush adjust instead), `R` / `G` / `B` show one channel as grey and
`A` the alpha — the same key again goes back to rgb. `Z` toggles the pixel readout (scene-linear and display values under the pointer) and shift+`Z` pins a sample to compare against; Z still sets the finest drag rate while a value drag is under way.

Main keys: `Q` fullscreens the pane under the cursor, alone — the graph,
the properties or the image (`Q` again or Esc puts the layout back), ⌘Q
fullscreens the image with the node graph on top (there is no quit key — close the window), `W`
show/hide the properties bar, `Y` curve editor, `A` alpha, Esc back to the
panelled layout. In the graph, hold `` ` `` for the **node-ops pie** (group,
unify size, tidy, swap inputs, bypass, frame, region…, ungroup — release
over one), ⌥` for the global source pie, ⌘` for the region's. `F` or Home frames the canvas under the cursor (graph, viewer; `R` in the curve editor); in the graph held `R` + drag rotates the selection's layout, as in Blender.

## Recording a clip

The **rec** button in the menubar (or ⌘⇧R) records wobble itself, straight
from the frames it draws: no screen-recording permission, nothing else on
your desktop. Right-click it to record the whole window or just the viewer,
and to switch the pointer, the small "Made with Wobble" mark and a GIF copy.
Press it again to stop; a window opens where you drag the ends of the strip
to trim, then **save** writes an MP4 (your OS's own H.264 encoder) to
`~/Movies/Wobble` (`Videos\Wobble` on Windows). From there: **share…** (the
macOS share sheet), **reveal**, **copy** the file, **publish to wobble…** (the wobble
blog, even before saving), or open Bluesky's compose page and
attach the clip there. Nothing else uploads, and wobble signs in nowhere. A take stops itself at five
minutes.

## Remappable hotkeys

**keys…** in the menubar opens the remapper: a searchable command list on
the left, a keyboard on the right. **Hover a key** and the panel under the
board lists everything it does — every pane, every chord, and the fixed
gestures that hold it. **Click a command** and its whole chord lights up on
the board (hovering a row previews it); clicking a key selects what lives
there, again for the next command on a shared key. Coloured dots on each
cap say which panes use it. Search matches a command's name, its pane or
its key (`tidy`, `graph`, `cmd+s`), and Enter selects the first match.

To rebind, click a row's key, then press the new one — or click it on the
board, holding any modifiers. Esc cancels, ⟲ takes a row back to its
default, and two actions landing on one key **in the same pane** flag red
(the same key in different panes is fine: the pane under the cursor owns
its letters, which is how graph `A` aligns while `A` over the viewer shows
alpha). The `?` hotkey list always shows the live bindings.

Only the changes are saved (`~/.wobble/prefs.json`), so an untouched key
follows whatever a future build ships. Deliberately not remappable: the
navigation chords — Blender's hands are the point — and the rgb sliders'
held colour axes (G/S/D/F). Those are listed too, read-only, so the board
tells the whole truth about a key.

## Interface colours

**colours…** in the menubar recolours the interface: backgrounds, text,
buttons and accent, the node graph (wires, ports, selection), every node
type, the timelines, the patch bay, the curve editors and HUD backdrops.
Groups fold; search matches a colour's name or its group. Each swatch opens
the colour picker and the change shows at once.

Colours are linked where they belong together. Move **background** and the
windows, fields, canvases and wells move by the same amount, so a canvas
that sat a little darker than the panels still does. A linked colour says
what it follows until you set it yourself; ⟲ takes a colour
back to its default (and its link), and **reset all** takes everything back.

Only the changes are saved (`~/.wobble/prefs.json`, under `theme`, as hex),
so an untouched colour follows whatever a future build ships. Deliberately
not in the list: colours that are data — the r/g/b lanes of the rgb sliders
and wheels, swatches, plugin node colours, the colours you give regions and
backdrops — and the viewer's HUD ink, which picks light or dark against the
footage under it (or wears the colour a node chose on its chip; see *Viewer
manipulators*).

### The colour designer (`wobble-colors`)

A separate app for designing a scheme without hunting through the list:

```
cargo run -p colors --release
```

You pick 12 shared colours (background, text, lines, buttons, accent,
highlight, selection, wires, handles, good, warning, error). Every other
colour in Wobble is a shade linked to one of them, so it moves along without
a decision. Stills of Wobble's screens (workspace, nodes, curves + wheels,
edit timeline, windows + keys) show the result live; click anything on them
to open the colour it comes from, and everything sharing it pulses. Port
colours are not in the set yet. ⌘Z / ⌘⇧Z undo.

**Nodes** go one level deeper. The *nodes* page draws every node Wobble has,
with its real icon, grouped by family. The list's *node colours* section
holds the ten family colours (sources, colour, filters, …), each node type
being a shade of its family, plus the node outline and ink colours: several
icons (Shader, LoopIn / LoopOut, Glow's rays) are drawn in the outline colour
alone, so that one decides whether they read. Click a node to pick its
family; double-click it (or its chip under the family) to tune that node
alone, and **⟲ follow … again** hands it back to the family.

**save to Wobble** writes the changes into `~/.wobble/prefs.json` (`theme`
plus the picker swatches, nothing else). Quit Wobble first: a running
Wobble writes its own prefs back on its next save. **more** exports / imports
a scheme around as `{"theme": {...}}` JSON (import also reads a whole
prefs.json). It shares `app/src/theme.rs` and `app/src/colorpicker.rs` by
`#[path]`; the 12 colours are `CORE` in `colors/src/main.rs`, and a slot
joins one by linking to it in the theme table. The node icons are a copy of
the graph's (`colors/src/glyphs.rs`); Wobble's `glyph_parity` test paints
every node both ways and fails when they differ. Dev flags for checks:
`--scene N --select slot_id --set slot_id=#hex --shot out.png` (renders with
no window), plus `--click x,y` / `--dclick x,y` to click before the shot.

## Grain

Procedural film grain (Filter ▸ Grain). Wire a plate into input 0 and it
grains the plate; input 1 is the usual mask. With nothing wired it is a
generator: grain on a flat `base` in its own format.

- **stock** loads a look shaped like a film stock (35mm 50D / 250D / 500T,
  16mm, super 8, b&w 400), with grain sizes scaled to the format.
- **size / amount / irregular** are per channel (rgb rows, gang or split
  them): a colour negative's blue layer is typically the coarsest and
  loudest. `amount` is the grain's standard deviation, exactly.
- **apply** (wired): `film` multiplies, `in · exp(d / 0.18)` — a
  negative's grain is a density fluctuation, so in linear light it scales
  with the signal, black stays black, and `amount` is the grain at 18%
  grey. `add` puts the same grain everywhere (scaled by alpha, so clear
  stays clear).
- **shad · mid · high** (wired) weights the grain by the plate's
  luminance, crossfaded over stops around 18% grey.
- **softness** is the grain edge, **mono** blends colour grain to b&w,
  **aspect** stretches it, **seed** picks the pattern, **new grain every
  frame** re-seeds from the frame number.

The model is sparse convolution — a jittered soft disc per lattice cell,
each with its own radius and signed density, plus a finer layer and a
clumping field that `irregular` brings in — not blurred noise, which is
why it reads as grain. It fuses into the main pass (cyan, not red), no
knob recompiles, and grain sizes follow the proxy.

## Dilate

Sub-pixel matte dilate / erode, in its own gather pass. Grayscale morphology
with a FLAT disc: dilation is the max over the disc, erosion the min. On an
anti-aliased matte — coverage ramping about 1 per pixel across the edge — a
flat disc of radius r shifts the whole ramp by exactly r, fractions included,
which is the property the node is for.

The whole part of r is covered by integer taps; the fraction rides on a ring
of bilinear taps at exactly radius r, because max is monotone in radius and
the tap that decides the answer is the one along the gradient. A 0.25 px
dilate really does move the edge a quarter of a pixel rather than snapping to
0 or 1 (`engine/tests/dilate.rs` measures the 0.5 crossing to prove it).
Matte in, matte out — drag an image at it and the editor bridges a ToMatte in,
like any other mask port. Amount is signed (positive grows) and capped at
`DILATE_MAX` = 16, since the integer part of the gather is O(r²).

## EdgeExtend

Pixel spread and edge expander, one node: both push a matte's edge colours
outward so an unpremultiplied edge never picks up black or the old
background. **spread** fills the whole frame outside the matte; **expand**
fills `distance` px past the edge (0 still covers the anti-aliased rim).
**inset** takes the colour from that far INSIDE the edge and replaces the
band too — the fix for a dark or light key edge. **softness** ramps both
boundaries. The edge is the input's alpha, or a matte wired into port 1
(the colour is then read straight); `premult in`, `keep alpha` and
`premult out` say how alpha comes in and goes out.

On the GPU: a jump flood (eight passes, steps 128..1, carrying the sub-pixel
offset to the matte's 0.5 line) gives the signed distance; a push-pull
pyramid of the core's weighted colour gives a smooth, streak-free fill of
the whole frame. Every param is a uniform (a jump-flood step past what the
params can reach is a copy, so the pass count never moves); reach is capped
at `EXTEND_MAX` = 250 px. `engine/tests/edgeextend.rs` pins spread, expand,
the inset fix and the wired matte.

## Curves

Curve (colour remap) and LumaKey (luma -> coverage) author the same thing: a
short list of control points joined either **linearly** or by a **monotone
cubic** (Fritsch-Carlson tangents, flat at the ends — which makes a two-point
smooth curve exactly a smoothstep, so every LumaKey written before the curve
existed keys identically). `app/src/curveui.rs` is the shared editor and its
`eval()` mirrors the WGSL `curve_eval` line for line, so the drawn response
IS the rendered one.

Editing, the same everywhere including the keyframe editor: **shift-click**
adds, **alt-click** removes, a marquee selects, and two or more selected
points raise a scale box whose handles scale the selection about the opposite
edge. The view **pans and zooms with decoupled axes** — scroll zooms about
the cursor, shift-scroll takes x alone, alt-scroll y alone, ⌘-drag or
middle-drag pans, double-click reframes.

Nothing is clamped. A curve **extrapolates** along its end segments, so a
default Curve (a straight line from (0,0) to (1,1)) is a no-op at 6.0 as much
as at 0.6 — which matters in a scene-linear comp, where clamping at the last
point crushes every highlight above it. The **high point** control is the
shortcut for that: type the input value that should map to 1.0 and press
`set`, and the curve becomes a straight line from the origin to there,
discarding whatever shape it had.

Curve's **split rgb** gives each channel its own curve (seeded from the
ganged one) with the other two drawn as ghosts. LumaKey keeps `lo`/`hi` as
its two end handles — they remain the animated pair and still ctrl-drag out
to a viewer HUD — and any point added between them shapes the response;
coverage there IS 0..1, so that one editor does clamp y.

Packing: interpolation and the split flag are UNIFORMS (toggling them never
recompiles); only the point COUNTS move the slot span, exactly like Spline.
`--test-curveedit` drives the gestures (add, marquee, group drag, remove).

## Node status

Every node is labelled **production**, **beta** or **alpha**, and the
label is read off the node type in one place (`editor::status_of`), so the
three places it shows cannot disagree:

- the add menus (right-click and Tab search) colour the name itself: red
  for alpha, lime for beta, plain white for production — no word (hover an
  entry for it);
- the graph marks beta and alpha nodes with a small dot below left in the
  same colours, away from the pin markers, and marks production nodes with
  nothing at all, so a finished comp is plain.

**nodes:** in the menubar is the project's threshold — *all (alpha)*,
*beta and up* or *production only* — and hides everything below it from
the add menus, plugins, subwobs and controls included. It is saved with the
setup rather than in prefs, follows you into a subwob or noise body and back
out, and never touches the render cache or a subwob's identity. Nodes already
in the graph are never hidden by it.

The policy today: everything is alpha except Transform and CornerPin,
which are beta; nothing is production yet. Graduating a node is a
deliberate edit to `status_of` and to the test that pins the policy.

## ColorCompress

Four knobs and one straight line per channel. **black** and **white** are
the input levels you care about; **lift** and **mult** are where those two
land. So `black → lift` and `white → mult`, exactly, and everything between
rides the line between them — the classic blackpoint/whitepoint/lift/gain, on
its own node.

Narrow the output pair inside the input pair and the range **compresses**:
a full-swing plate tucked into a milky window to sit under a soft comp.
Widen it and it **expands**: a flat plate pulled back out to contrast.
Neither direction is privileged, and the panel reads in that order — the
pair you measure, a rule, then the pair they land on.

**Nothing is clipped at either end.** Outside the two points the line keeps
going, so a highlight at 8.0 is still a highlight afterwards — in a
scene-linear comp a clip is a decision for a view transform or a Render, not
one a colour node makes on your behalf. A white point BELOW the black point
inverts, on purpose; a zero-width range is held to the narrowest one that
still has a slope rather than exploding.

Every knob is an rgb triple on the same slider the Grade uses (ganged,
relative, with the S/D/F colour axes), each lane keys and patches, and the
node takes a matte on its second input like any other colour node. What
reaches the shader is the solved LINE, not the four knobs: `pack` does the
division once per frame, the way the Transform's inverse affine and the
CornerPin's homography already do, so the fused pass costs one multiply-add
per pixel and no control on the node can ever recompile it.

## Expression

A pixel written as arithmetic, the classic Expression node: one field per output
channel — **r**, **g**, **b**, **a** — each an expression over the input
pixel. An empty field passes its channel through.

```text
r == 0 ? 1 : 0          a matte where red is exactly zero
isinf(r) ? 0 : r        infinities out of a render
r == inf ? 1 : r        +inf only (r == -inf for the other way)
k = r * 2; k > 1 ? 1 : k    a line (or a `;`) can name a value
```

It is the patch bay's expression language — the same operators, functions,
`name = …` lines and `#` comments — plus what a test on a pixel needs:
comparisons `== != < > <= >=`, logic `&& || !`, `c ? a : b`, and `inf`,
`nan`, `isinf()`, `isnan()`. Variables: `r g b a`, `x y` (the pixel's centre
in pixels from the top left), `width height`, `frame` (`t`, `$F`); constants
`pi tau e inf nan`. A comparison is 1 or 0 and anything non-zero is true
(NaN included). `x y width height` are full-resolution pixels, so an
expression means the same thing with proxy on. Hover the help line in the
panel for the function list.

The expressions are compiled into the comp's one fused pass — as fast as a
built-in node — so changing one recompiles; a field is applied when you
press Enter or leave it (Escape walks away from an edit). A field that does
not parse passes its channel through, says where it broke in red, and the
comp keeps playing. **Infinity and NaN are tested exactly**: comparisons,
`isinf`, `isnan` and "is it true" are done on the float's bits, because a
GPU's fast math is allowed to assume NaN never happens (Metal really does
fold `x == x` to true). Masked like a Grade.

## HueRotate

Turns every colour about the neutral axis by an **angle** (positive turns
red toward yellow, as every hue control here runs), holding each pixel's
**luminance exactly** — in the working space's own luma weights, so a
rotation never brightens or darkens anything and greys never move. That is
also why a red turned to yellow comes out olive: it keeps red's
brightness. 360° is the identity; the angle keys, patches and drags out to a
HUD; nothing recompiles. The panel shows the hue circle before and after,
so you can read what lands where. Masked like a Grade.

## LUT

Applies a lookup table from a file: **file** is typed (a leading `~` works)
or picked with **…**, and can be an `s3://` path like a Read's. Formats:

- **`.cube`** (Resolve / Adobe) — 1D, 3D, or a 1D shaper followed by a cube
  in one file, with `DOMAIN_MIN`/`DOMAIN_MAX` and Resolve's
  `LUT_1D_INPUT_RANGE`/`LUT_3D_INPUT_RANGE` honoured;
- **`.3dl`** (Autodesk / Lustre / Nuke) — the output depth from a `Mesh`
  header or the largest value (10, 12, 14 or 16 bit);
- **`.spi1d`** and **`.spi3d`** (Sony Imageworks, OCIO's own).

Not read: CLF / CTF, `.csp` and OCIO configs. A cube is interpolated
tetrahedrally and a 1D table linearly, as OCIO does; input outside the
table's range clamps.

The comp is scene-linear and nearly every LUT is not, so the node shapes
the picture into what the table expects first: **expects** is that space
(sRGB by default, or LogC3, S-Log3, ACEScct, …), and **shaper** either uses
that space's own curve or a **log2** allocation about 18% grey over a
**stops** range (for LUTs built on an `lg2` shaper). **output** says what
comes back: **as it went in** undoes the shaping, so a look LUT stays a
look in the working space; a named space decodes a camera-to-display LUT's
result; **raw** leaves the table's numbers as they are (display values —
view them with the view transform off). **mix** fades it in; a mask mixes
back to the input.

It fuses into the main pass: the table is a texture, so picking another
file never recompiles, and an edited file is picked up on its own (and is
a new render-cache key). A file that does not load passes the picture
through, with the reason in red in the panel.

### Inverse View

Colour ▸ **Inverse View** is the same node with its table SOLVED instead of
read: the viewer's display transform (ACES 1.3 or 1.2, AgX, Narkowicz or Standard,
with its display gamut, curve and look) run backwards, so a display-referred
picture — an AI-generated clip, a graded Rec.709 deliverable — comes into the
comp as scene-linear light a blur, glow or grade can treat as light. Shown
through the SAME view transform, the plate reads back the codes it came from
(within an 8-bit code value). It cannot invent what the display clipped: a
code of 1.0 lands at the transform's own white (~16 for ACES 1.3), and the
shape above it is Highlight Rebuild's, HDRCNN's or LTX HDR's job. Set
**plate as** to what the Load decoded the file as (Rec.709 for a movie, by
default). **dither** (8-bit steps) hides the banding an 8-bit source shows
once its highlights are stretched; **limit** is a soft ceiling on the result.
Both are uniforms. The table (33³, stored as ACEScct) is built once per pick,
in well under a second.

## UV Warp

The standard UV warp. Input 1 is a UV picture — **red is U, green is V**, both
0..1 over the source's format — and every output pixel is the source
(input 0) sampled where its UV pixel points: a lens distortion exported from
a solver, a refraction pass, any warp someone computed as an image. The standard
convention exactly: V runs **up** from the bottom edge and coordinates name
pixel centres at `(x + 0.5) / w`, so an identity map gives the source back
pixel for pixel; untick **v up** for maps written top-down. Sampling is
bilinear, or **nearest** for ids and data. Alpha comes from the source; the
map's alpha is not read. The output takes the map's format (else the
source's). It fuses — the source's subtree is evaluated at the looked-up
position, the way a Transform is — so it adds no pass, and both controls
are uniforms. A mask on input 2 mixes back to the unwarped source.

## Lens Distortion

The Brown-Conrady model — radial **k1 k2 k3**, tangential **p1 p2**, the
numbers OpenCV, 3DEqualizer's classic model and most solvers export — either
way round. **undistort** takes a plate and gives back the rectilinear
picture; **distort** bends a rectilinear one (CG, a patch) the way the lens
bent the plate. The same values both ways are exact inverses, so undistort →
work → distort lands back on the plate. The model is normalised to half the
plate's diagonal (r = 1 at the corners, so a k reads the same at any
resolution), about the **centre** (normalised, drag it in the viewer), with
x desqueezed first by the anamorphic **squeeze**; **zoom** scales the
undistorted picture. **overscan** grows the format undistorting (a barrel
plate's corners stretch past the frame) and shrinks it distorting — set the
same value on both ends. **chromatic aberration** reads red and blue at their
own magnification (three lookups instead of one). **stmap** outputs the
lookup instead of the picture, in UV Warp's convention, so a UV Warp fed it
reproduces the node and the warp can be exported (green's map: CA is not in
it). Where the inverse cannot be found, or past the radius where the model
folds over, the pixel is transparent rather than wrong. Selected, the viewer
draws a grid that is straight in the undistorted picture and the source
frame's edge, both where they land: straight lines in the plate should
follow the grid. It fuses like UV Warp: every number is a uniform, only
mode, output and the CA switch recompile. Solving the coefficients from a
grid or from lines is not built yet.

## GridWarp

Two lattices of control points over the node's format: a **source** grid
and a **destination** grid with the same shape (4 x 4 cells to start, up to
16 x 16). Whatever sat under a source point lands on its destination point,
and between the points the warp follows the lattice — a smooth Catmull-Rom
patch net by default, so lines bend through the points, or straight
bilinear cells with **smooth** off. Both grids start as the same regular
lattice, which is no warp at all.

Select the node and the lattice is drawn on the viewer, curved where it
bends. **edit** in the panel picks which grid the handles move:
**destination** (where the picture goes — the usual one), **source** (what
gets picked up), or **both** (re-place a point without changing the warp
there). Click a point to pick it, S-click to add one, drag a box on the
background to pick what it covers, and drag a picked point to move the
whole pick; Z X C V slow the drag. Two or more picked points wear a box:
drag inside it to move them, a corner or edge handle to scale them about
the opposite side (alt: about the pivot), ctrl on an edge handle to skew,
and the knob on its stalk to turn them about the pivot — the crosshair,
which starts at the box's centre and drags anywhere. The panel shows the first picked point's
source and destination as numbers. Every point of both grids is an
animatable channel (`pt7 dest x`…), keyed from the viewer like any gizmo,
and **amount** blends from no warp (0) to the full one (1).

Changing **grid** re-samples both lattices at the new size, so the warp
stays where it is (a point's animation does not survive a re-grid: the
points are renumbered). **no warp** puts the destination back on the
source, **reset** both grids back on the regular lattice, **swap** trades
them, which turns the warp round.

It fuses: per pixel the shader inverts the destination lattice (Newton's
method, seeded at the identity and again from the nearest destination
point when that does not converge) and samples the input where the source
lattice says, the way a Transform does — no pass, no texture, and moving a
point is a uniform write. Only the grid size is structure. Outside the
destination lattice the warp carries on linearly past its edge. Where a
destination grid is folded over itself and no inverse can be found the
pixel is **transparent**, so a fold shows instead of hiding. A mask on
input 1 mixes back to the unwarped source.

## Warp

Curves instead of a lattice: each **curve** is a pair, a source side and a
destination side with the same points, and whatever lay along the source
side is moved onto the destination side, point for point. The picture
between and around the curves follows smoothly and fades back to untouched
past **falloff** (pixels; at that distance from a long curve the pull has
about halved). Curves bend smoothly through their points and can be
**closed** into a loop round a shape. Up to eight curves per node.

Select the node and S-click on the picture to draw: each S-click adds a
point to the current curve (the chip in the panel; **+** starts another),
both sides together, so drawing never moves the picture. D-click a point
to remove it. **edit** picks which side the handles move — **destination**
(the warp), **source** (what gets picked up) or **both** — and points are
picked and moved exactly like GridWarp's: click, S-click to add to the
pick, drag a box, drag to move, and two or more picked get the same affine
box (move, scale, ctrl to skew, rotate about a draggable pivot). Every
point of both sides is an animatable channel (`curve1 pt2 dest x`…), and so
are **falloff** and **amount**. **no warp** puts every destination back on
its source, **swap** turns the warp round.

It fuses: each curve is sampled to the same 24 points whatever its point
count, and per pixel the shader weighs every destination segment's
displacement by its length over the squared distance, so a pixel on a
destination curve samples exactly its source point, then samples the input
there — no pass. Only the number of curves is structure; dragging, adding
or keying points is a uniform write. A mask on input 1 mixes back to the
unwarped source.

## ScreenKey

A green / blue screen keyer, the Keylight shape of it: **one node pulls the
matte, despills the foreground and hands on premultiplied RGBA**, so a
Blend over the background is the next node. It is pointwise and fuses with
its neighbours — cyan in the graph, a few dozen ALU per pixel.

**Sample the screen**: hold `E` over the **screen** row and click (or drag a
box on) the screen in the viewer. The sampled colour's largest channel is
the channel being keyed, so the same node keys green, blue or red screens.
The matte is a **colour-difference key normalised to that sample and to
the pixel's own level**: how far the screen channel stands above the other
two, as a fraction of how far the sampled screen's does at the same
brightness. A lit patch of screen and a shadowed corner both read as
screen; a grey — any grey — reads as foreground; black is foreground.

- **balance** — how the two non-screen channels combine into the level
  the screen channel is measured against: 1 = their max (conservative:
  only a pixel greener than BOTH others is screen, so a red coat is safe),
  0 = their min (aggressive: more of the edge goes). 0.5 is the mean.
- **gain** — multiplies the screen fraction: above 1 calls more of the
  edge screen.
- **clip black / clip white** — remap the matte: alpha at or below black
  becomes 0 (cleans the screen), at or above white becomes 1 (solidifies
  the core), the soft edge between them is kept.
- **holdout** (input 2) forces alpha 1 where it covers — the green shirt
  you want to keep; **garbage** (input 3) forces 0 — the light stands.

**Despill** limits the screen channel to the level the other two say it
should have (the same balance), which takes exactly the screen bounce off
skin and hair. Taking light darkens, so **restore** puts the luminance the
despill took back as neutral light, in the working space's luma weights: a
despilled edge keeps its brightness and loses only its cast. Both default
to 1.

**output**: premultiplied (despilled × matte, matte in alpha — what a Blend
wants), straight, the matte (in rgb, alpha 1 — it reads plainly in the
viewer and feeds any mask port), or the despill alone with the input's
alpha. Everything on the node is a uniform: sampling, dragging and
switching the output never recompile. Every value keys, patches and drags
out to a HUD.

## Despill

The despill on its own, for a foreground keyed somewhere else (a Keyer, a
CorridorKey, a roto) or for a plate that only needs its cast taken off.
**screen** is green or blue; the screen channel is **limited** to what the
other two say it should be — **average** of them (the everyday despill),
**max** (gentle: yellows and cyans survive) or **min** (aggressive).

- **strength** — how much of the spill comes out, 0..1.
- **bias** — scales the limit: above 1 lets more of the screen colour
  through, below 1 takes more out.
- **restore** — puts back the luminance the despill took, as neutral light
  in the working space's luma weights: 1 keeps a despilled edge exactly as
  bright, 0 lets it go darker.

Pointwise and fused (cyan in the graph); screen and method are uniforms, so
nothing on it ever recompiles. Alpha is untouched. Masked like a Grade. The
panel shows a skin-to-screen ramp before and after.

## Lightwrap

Lets the background's light bleed round the foreground's edges, the step
that stops a keyed element looking pasted on. Input 0 is the foreground
(premultiplied, with its alpha), input 1 the background it will sit over,
input 2 the mask. The background, blurred by **bg blur** (px), is laid into
a band just inside the foreground's edge — `fg.a × (1 − blur(fg.a))`, the
blur being **width** (px) — times **intensity**, then **added** to the
foreground (or **screened**, or output alone with the band as alpha:
**wrap only**). The result keeps the foreground's alpha, so Blend it over
the same background as before.

A slow path (red in the graph): three passes, one separable blur that
carries the background in rgb and the foreground alpha in a with their own
sigmas. With no background wired there is nothing to wrap — it costs no
passes, passes the foreground through and the panel says so.

## Clipped highlights

Nothing recovers detail a sensor never recorded. What can be done comes in
three tiers, cheapest and most honest first:

### HighlightRebuild — real data only

Filter ▸ **HighlightRebuild**. Where only one or two channels hit the clip
(a red neon whose red maxed out while green and blue did not), the clipped
channels are rebuilt from the unclipped ones at the colour of the unclipped
surface around the clip — the raw developers' "reconstruct colour". **clip**
is the value the plate clipped at (scene-linear, after its input transform;
no auto-detect, set it by eye or with the pixel readout), **softness** how
far under it the rebuild fades in, **radius** how far out to look for
unclipped colour (it has to reach the middle of the clipped region from its
edge). A pixel with no channel near the clip comes out bit-exact; one with
all three clipped has nothing left to rebuild from and is left as it is.
Four passes, red in the graph like any filter.

### HDRCNN — a CNN's reconstruction, in our own WGSL

AI / ML ▸ **HDRCNN**: Eilertsen et al.'s
[HDR reconstruction CNN](https://github.com/gabrieleilertsen/hdrcnn)
(SIGGRAPH Asia 2017, BSD-3), ported to wobble's WGSL network runtime — no
Python. The download gate fetches the weights (177 MB, from the authors'
site). **render** renders the input over its range and reconstructs each
frame on the GPU (~3 s a 2K frame on an M4 Pro); the result plays back like
a bake. **clip** is the level the plate clipped at (linear): the net
reconstructs only what is within 5% of it — the method's own blend — and
every other pixel is the plate exactly. **video** weights (the default) are
trained for temporal stability and JPEG robustness; **stills** are the
paper's original parameters. A new clip is a new render.

### LTX HDR — generative, steady over a clip

AI / ML ▸ **LTX HDR**: Lightricks'
[LTX-2.5 SDR-to-HDR IC-LoRA](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-SDR-To-HDR)
makes up what is past the clip, for the whole clip in one generation so it
holds still over time. It is a MODEL'S GUESS, not recovered data.

- **install…** fetches the pinned [LTX-2](https://github.com/Lightricks/LTX-2)
  checkout into `~/.wobble/LTX-2` and `uv sync`s it. **fetch weights…**
  downloads ~45 GB (the 22B distilled transformer, its video VAE, the HDR
  LoRA and its scene embedding) at pinned revisions. Both repositories are
  GATED: accept the LTX-2 Community Licence on both model pages and log in
  once (`hf auth login`, or `HF_TOKEN`) first. Nothing downloads without a
  click.
- **run** renders the input (ACEScg EXRs), pads the clip to 8k+1 frames by
  holding the last one, runs the tool, and plays the result (scene-linear
  ACEScg) back through the node.
- The result is composited back **only where the plate nears the clip**
  (**clip**, **softness** — uniforms, no re-run): the model sends every
  pixel through its VAE, which shifts them all a little, and a pixel the
  plate recorded properly must not pay for that. **raw** shows the result
  everywhere.
- **size** runs the model on a 1/2 or 1/4 copy; the result is stretched back
  over the plate. On CUDA the transformer is cast to fp8 (~24 GB of VRAM at
  HD; **offload** cpu/disk for less). On a Mac it runs in bf16 and the
  transformer alone is 42 GB: on 48 GB it does not fit at HD — use a proxy
  and offload disk, and expect it to be very slow or not to fit.
- Licence: the **LTX-2 Community Licence** — free commercial use under $10M
  annual revenue, with use restrictions and a provenance clause. The node is
  tagged (the NC badge) and a Write below it warns before it renders; a bake
  does not clear that.

## CorridorKey — Corridor Digital's neural keyer

**CorridorKey** (Mattes) runs [CorridorKey](https://github.com/nikopueringer/CorridorKey),
the open-source neural green / blue screen keyer Niko Pueringer released in
2026, on the node's input and plays the result back like a bake. Where a
colour-difference key (ScreenKey) measures how green a pixel is, the network
solves the **unmixing** problem: it reconstructs the straight foreground
colour and a linear alpha through motion blur, defocus and hair, given the
plate and a *coarse* alpha hint.

- **image** (input 1): the plate, in the comp's working space; it is
  rendered to disk as linear Rec.709, the tool's own convention.
- **hint** (input 2, optional): a rough matte, white where the subject is.
  A roto shape, a ScreenKey matte, anything. With nothing wired a coarse
  colour-difference key of the plate stands in, which is usually enough
  for a clean screen and usually not for a busy one.

**key** renders the plate and the hint over the plate's frame range, runs
the tool over every frame (a process, its progress in the panel, cancel at
any time), and the node then samples the keyed frames through its reader
slot: **premultiplied** (despilled foreground × the cleaned matte, matte in
alpha — a Blend over the background is the next node) or the **matte**
alone. Switching output is a uniform; **screen** (auto / green / blue —
two models), **despill**, **refiner** and **despeckle** are the tool's own
settings and change the recipe. The run is keyed on everything upstream
and every setting, so an edit shows as **STALE** and nothing re-runs behind
your back; a finished key is adopted when the setup opens, like a bake.
The node is cyan once keyed — the pixels come off disk — and a loaded key
clears the slow taint above it, as a Bake does.

**Installing.** The tool is Python: press **install…** in the panel and
wobble clones it into `~/.wobble/CorridorKey` (or
`$WOBBLE_CORRIDORKEY_DIR`) and runs `uv sync` there (torch: several GB;
`uv` itself is `brew install uv`). On its first key the tool downloads its
~300 MB weights from Hugging Face. Nothing is fetched until you click.
CUDA, Apple Silicon (MLX, then Metal) and CPU all work; a 4K frame is a few
seconds on a big GPU.

**Licence.** The tool is CC BY-NC-SA 4.0 with an explicit grant: keyed
frames may be used "as part of a commercial project". The node is
therefore not tagged non-commercial; what the licence forbids is
repackaging or reselling the tool and paid inference services, which is
why wobble does not bundle it and never runs it as a service. The panel
links the licence.

## Upscale — neural super-resolution

AI / ML ▸ **Upscale**. 4x (or 2x) super-resolution from three published
models, run in wobble's own WGSL: no Python, no ONNX runtime, weights read
straight from the upstream `.pth` files. Like Flow, it is **click to
render**: nothing runs on an edit. `render` renders the node's input over
its frame range, upscales every frame on the GPU and plays the result back
like a bake.

| model | weights | licence | per input pixel | 1080p → 4K, M4 Pro |
|---|---|---|---|---|
| SPAN (fast) | `4xNomosUni_span_multijpg` (Philip Hofmann) | CC-BY-4.0 — credit him | 0.4 M MAC | ~1.3 s |
| Real-ESRGAN compact (fast) | `realesr-general-x4v3` | BSD-3-Clause | 1.2 M MAC | ~3.5 s |
| Real-ESRGAN x4plus (quality) | `RealESRGAN_x4plus` | BSD-3-Clause | 18 M MAC | ~60 s |

- **Nothing downloads without a click.** The panel's gate shows the picked
  model's size and licence and fetches it; `render` waits for it.
- **Unrendered, the node passes its input through** at the input's size.
  Rendered, its format is the input's times the scale. **2x** runs the
  model at its native 4x and halves the result with a [1 3 3 1] decimation
  in linear light.
- **The result is keyed on everything upstream** (and the model, the
  scale, the range): an edit shows **STALE** and keeps playing the old
  frames until you render again. A loaded result is frames on disk, so it
  clears the red above it like a Bake.
- **Colour.** The models were trained on display-referred sRGB in 0..1, so
  that is what they are given: the picture in linear Rec.709, clamped to
  0..1 and sRGB-encoded. What lies outside — highlights past 1,
  out-of-gamut negatives — is upscaled bilinearly beside the model and
  added back, so a 4.0 highlight stays 4.0 (its edge is the bilinear one).
  Colour is unpremultiplied on the way in; alpha is upscaled bilinearly and
  the result premultiplied again.
- The frame is cut into overlapping tiles (the halo is the network's
  receptive field, so tiles join exactly), sized to the card, and submitted
  in pieces small enough that Windows' GPU watchdog never fires.
- `cargo run --release -p engine --example upscale_bench [w h] [x4] [model]`
  times it on your machine.

## SeedVR2 — diffusion restoration and upscaling

AI / ML ▸ **SeedVR2**: ByteDance-Seed's one-step diffusion video restorer
(Apache-2.0, code and weights), run as a process through the community
`numz/ComfyUI-SeedVR2_VideoUpscaler` CLI (Apache-2.0) on a click — the
CorridorKey pattern. It hallucinates plausible detail where Upscale only
sharpens what is there; use it on footage that is genuinely soft or damaged.

- **Where it runs**: an NVIDIA GPU (CUDA) or **Apple Silicon (MPS)**.
  Measured on an M4 Pro (48 GB): the 3B model, 960×540 → 1920×1080, ~32 s
  a frame once loaded, 28 GB of unified memory at the peak (the driver
  tiles the VAE; untiled it was 50 GB). The 7B wants a
  big card: a 16 GB NVIDIA card (the driver picks the fp8 builds under
  20 GB), 24 GB+ for fp16.
- `install…` fetches the CLI at a pinned commit into `~/.wobble/SeedVR2`
  and runs `uv sync` there; the first run downloads the weights from
  Hugging Face (~7 GB for 3B, ~17 GB for 7B). Nothing happens until you
  press the buttons.
- **model** 3B / 7B / 7B sharp, **scale** (of the short side), **seed**,
  **batch** (frames it sees at once — steadier across time, more memory;
  it takes 4n + 1) and **colour** (how it pulls the result's colour back to
  the input's).
- The input is rendered at its own format, linear Rec.709; highlights and
  alpha go around the model exactly as in Upscale. The result is keyed on
  everything upstream and every setting — STALE after an edit, never re-run
  behind your back — and clears the red above it once loaded.

## Depth — steady depth for a clip

AI / ML ▸ **Depth**: ByteDance's
[Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything)
(Apache-2.0 code) run as a process on a click, played back like a bake. It
solves the whole clip in overlapping 32-frame windows, each aligned to the
last, so the depth holds steady over time where a per-frame model flickers.

- **Where it runs**: Apple Silicon (MPS), an NVIDIA GPU (CUDA) or, slowly,
  the CPU. Measured on an M4 Pro: hcr080 (100 frames, 2048×2048), the small
  model at input 518 — 89 ms a frame of inference, 22 s for the whole run
  including reading and writing the frames. Memory does not grow with the
  clip: frames are read a window at a time.
- `install…` fetches the tool at a pinned commit into
  `~/.wobble/VideoDepthAnything` and runs `uv sync` there. The **weights**
  come from the download button the panel shows above it (the models
  catalog, checked by hash), never from the tool.
- **model**: small (Apache-2.0, 116 MB, the default), small metric
  (Apache-2.0, metres), large and large metric (CC-BY-NC-4.0, 1.5 GB —
  research use only: the node is tagged and a Write below it warns, and a
  bake does not clear that). **size**: the network's input short side
  (multiple of 14); larger is finer and slower.
- **output** is the **data** pipe (one value, broadcast to rgb), so it
  wires straight into a Defocus's Z port. Pick **depth** (relative: the
  nearest thing in the clip reads 1, further reads more; metres with a
  metric model), **disparity** (1/depth — turn on the Defocus's *inverse
  depth*) or **near 1 .. far 0**. All three come from the one stored value
  and the clip's statistics, so switching never re-runs. To take the depth
  out of wobble, wire it into a **Render**: it writes one float32 `Z`
  channel per frame. Relative depth is right up to a scale and shift the network
  cannot know: pick the focus off the map, do not type a distance.
- Unrun, the node outputs 0 and does no work. The result is keyed on
  everything upstream, the model and the size — STALE after an edit, never
  re-run behind your back — and clears the red above it once loaded.

## Sammie — click-to-roto with SAM 2

**Sammie** (AI) rotos by clicking. It runs the models
[Sammie-Roto 2](https://github.com/Zarxrax/Sammie-Roto-2) ships — Meta's
SAM 2.1 video segmenter (or EfficientTAM, a faster SAM 2), and optionally
MatAnyone 2 for soft edges — on the node's input, and plays the matte back
like a bake.

- **Click the subject.** With the node selected, **S-click** on the subject
  in the viewer to include it, **shift+S-click** to exclude something,
  **D-click** a point to remove it, drag a point to move it. Points belong
  to the frame they were clicked on (other frames' points show as faint
  rings); clicks on several frames are corrections where the track drifted.
- **auto preview** (on by default): each click shows its segment over the
  viewer straight away (green fill, white outline; grey while a newer
  click is being worked out). The model stays loaded between clicks, so
  after the first one a click is about half a second. Turn it off and the
  **preview** button does the same on demand.
- **track** covers the plate's own frame range, cut to the script range
  when one is set (no handles).
- **track** renders the plate over its frame range, segments the subject
  from your clicks and tracks it forward and backward through the shot (a
  process, progress in the panel, cancel any time), then the node samples
  the matte through its reader slot.
- **output**: a **cutout** (plate × matte, matte in alpha, on the image
  pipe — a Blend's foreground) or the **matte** itself on the mask pipe.
  Before anything is tracked a cutout passes the plate through, so you can
  view the node while you click.
- **model**: SAM 2.1 large (default, the best open tracker), base+
  (lighter) or EfficientTAM (about twice as fast). **matting**: off (a hard segment) or MatAnyone 2
  (soft, hair-aware edges).

The run is keyed on the plate, every click and every setting, so a new
click shows the loaded matte as **STALE** and nothing re-runs behind your
back; a finished matte is adopted when the setup opens. A loaded matte on
the mask pipe clears the slow taint above it, as a Bake does (a cutout
still reads the plate).

**Installing.** Press **install…** in the panel: wobble clones Sammie-Roto 2
into `~/.wobble/Sammie-Roto-2` (or `$WOBBLE_SAMMIE_DIR`) and runs `uv sync`
there with the torch build for this machine (Apple's on a Mac, CUDA where
`nvidia-smi` answers, else CPU; `$WOBBLE_SAMMIE_EXTRA` overrides). The
picked model's weights download on the first track. Nothing is fetched
until you click. wobble drives the models with its own script
(`tools/wobble_sammie.py`); Sammie's own app is never started.

**Licence.** Sammie-Roto 2 is GPL-3.0: you install it, wobble runs its
models as a separate process and ships none of it. SAM 2 and EfficientTAM
are Apache-2.0, so a plain segment is free to deliver. **MatAnyone 2 is the
NTU S-Lab License 1.0 — non-commercial.** With matting on, the node and
everything downstream are tagged, and a matte that was made with MatAnyone
stays tagged after matting is switched off, until it is re-tracked without
it.

## Tinter

The Tinter is Resolve's ColorSlice: a secondary for "the reds", "the
skin", "the sky", without drawing a key. (Internally, and in setup files,
it is still `ColorSlice`.) **Seven vectors** — red, skin, yellow, green,
cyan, blue, magenta — each with **Center** (where the vector sits), **Hue**
(turn a red shirt magenta), **Saturation** and **Density**; and five
**globals**: Saturation, Density, **Sat Depth** and **Den Depth** (how much
of a saturation or density change reaches the brighter areas) and **Sat
Balance** (how much a saturation change may move the luminance of
medium-saturation colours).

The panel reads like Resolve's palette: **one row per vector** — its
colour, its name, its **highlight** eye — with the four controls across it
as sliders whose **rails are pictures of what they do**: Center's rail is
the hues the vector can sit on; Hue, Saturation and Density show what the
node, with everything else as it is, does to that vector's typical colour
at each point along the rail. The globals sit under the grid, a ⟲ at the end
of a row resets that vector, and **reset all** resets everything. Click a
vector's name to **highlight** it: the viewer keeps that vector's pixels
and sinks the rest to grey, so you see exactly what it is moving.

**It is subtractive.** More saturation means more colour without more
light, the way dye in film behaves: the brightest channel holds and the
others are pulled down, as a power law on each channel's ratio to it, so a
colour deepens instead of lighting up and no channel is ever pushed
negative. **Density** darkens a colour in proportion to how colourful it
is — one stop at full saturation per stop set — and never touches a grey.

The six colour vectors are the vertices of the working space's own RGB hue
circle and **blend rather than stack**: an orange is half red, half yellow,
and setting all six to the same saturation is exactly the global one. Skin
is the exception, as in Resolve: an overlapping region between red and
yellow, centred on where the ColorChecker's light-skin patch actually lands
in the working space (about 15° in linear Rec.709, 22° in ACEScg), laid on
top of the other six rather than carved out of them. Hue turns in Oklab, so
a hue shift holds perceived lightness; vectors fade out toward the neutral
axis, where hue is noise rather than a fact.

Resolve documents the controls, not the arithmetic, so this is its own
model built to those descriptions: it will not match Resolve's numbers.
Every control is a uniform — the vector count is fixed — so nothing on the
node recompiles, it fuses into the one pass, and every field keys, patches
and ctrl-drags out to a HUD. The highlight follows the colour keyer's
contract: it acts only while the node is the one being viewed, so it can
never reach a render or a node downstream.

## Grade zones

A classic colour-correct has shadows, midtones and highlights; the Grade has
**zones**, which are that with the count unfixed. The panel is tabbed:
**master** is the grade you always had (wheels or sliders, gang, saturation,
the viewer HUD), **ranges** cuts the luma axis into bands, and every band
gets a tab of its own with the same instrument bound to that zone's lift /
gamma / gain / offset / saturation. A fresh Grade has no zones and costs what
it always cost; **add zones** on the ranges tab gives you the classic three
(shadows fall off over 0..0.09, highlights rise over 0.5..1.0, in
scene-linear luma), and from there **+** splits the selected band and **−**
dissolves it into its neighbour, so two zones or seven are the same
gesture. Splitting a zone copies its correction into both halves, so the
picture does not change until you diverge them.

The ranges page IS the LumaKey's band editor — the same locked partition
(one boundary belongs to the bands on both sides of it, weights always sum
to 1), the same linear/smooth crossfade, the same curve editor with the luma
ramp behind it, the same shift-drag to carry a crossfade, and the same
freedom to add a bias point in a band's plateau. Name a band there and its
tab, its chips and its curves take the name.

**Preview** sits in the selected band's row under the curve, beside its
name (where a LumaKey band has its **view** button), and it answers the
question setting
a range actually asks: *which pixels are these?* **matte** shows the
selected zone's coverage — white where its correction lands in full, black
where it does not land at all — which is how you read a boundary, and it is
the same matte a LumaKey band would hand you. **highlight** holds that
zone's pixels and sinks everything else toward grey, which is how you read
the same coverage against the shot: whose faces, which sky. **zones** shows
every zone at once, each in its own colour — the colours of the plot above
it, so the plot is the legend — mixed by the zones' own weights: a plateau
is its zone's flat colour, a crossfade is the blend of its two neighbours
(as wide as the crossfade is), and a band whose coverage dips (a bias point
in its plateau) shows darker, because the weights are summed and never
normalised. **hard** is the same map with the crossfades cut: each pixel
is the flat colour of whichever zone owns the most of it, so every boundary
is a line where its two weights cross, and black is a pixel no zone covers.
**zones** shows how wide the boundaries are; **hard** shows where they are.
The colours are picked for how they look through the viewer, not on paper:
three zones are green / blue / red, and every further zone is the one
farthest (as the default ACES view shows it) from all the colours before
it — with no two neighbours whose crossfade would pass through an orange.
Off is the graded picture. The matte and highlight follow the band you have
selected, so they move from chip to chip with your hand.

A preview only ever replaces the picture of the node being **viewed** — the
panel says so when it is armed on a node you are not viewing — and that is
not a convenience, it is what makes it safe: it is applied to the transient
graph alone, so it can never reach a render, a bake, or a node downstream,
exactly like the colour keyer's own source preview. Leaving the ranges tab
puts it away. In the shader it is two spare uniform lanes (`gamma.w`, which
holds the mode, and `gain.w`, the zone) that `grade4` never reads, so
arming one is a uniform write: no recompile, and a Grade with no zones pays
nothing for the feature at all.

Underneath, a Grade with zones is exactly a LumaKey with N bands feeding N
masked Grades in series, fused into the one pass: the zone weight is the
band's matte as the key would output it (luma of the grade's INPUT, before
the master), and each zone's correction is `mix(so far, grade(so far),
weight)` over the running result, master first. `engine/tests/grade_zones.rs`
pins that equivalence pixel for pixel against the hand-built graph. The zone
COUNT is topology (each zone is a fixed run of uniform slots); everything
inside one — its five adjustments, its band's edges, its bias points — is a
uniform, and every zone channel keys, patches and HUD-drags like the
master's (`ZoneGainR:1` is zone 1's gain, red). A plain Grade compiles to
the byte-identical shader it did before zones existed.

## Subwobs

A subwob is a subnet: nodes collapsed into ONE node, with its own inputs
and knobs.

- **Make one.** Select the nodes, hold `` ` `` over the graph and release on
  **subwob** (or press alt+G). They leave the comp and one subwob node takes
  their place, wired where they were: the wires that crossed into the
  selection feed its inputs, whatever read the selection reads it, and the
  viewer follows. The picture does not change. A value cable that crossed
  in becomes a knob on the subwob, still driving what it drove. Refused,
  with the reason above the graph: a Load (the subwob takes its picture as
  an input), a harp, a value cable leaving the selection, or a selection
  with two outputs.
- **Open it.** Select it and press ⏎ (or **open subwob ⏎**); **← comp**
  steps back out. An edit inside changes every instance.
- **Inputs are nodes.** Inside, each input is an input node, one type per
  pipe: **Image Input**, **Matte Input**, **Data Input** and **Fat Pipe
  Input**. An input node has no input socket, because it is a portal to the
  parent graph, and each one is one socket on the subwob node, typed as its
  pipe. A collapse makes them; add more from the **input** menu, which is
  only there while you are inside one. Not yet: a node that reads footage
  itself (a ChannelSelect off the fat pipe, a Cryptomatte, a harp) inside a
  subwob — the subwob then passes its input through, and its panel says
  so.
- **Knobs are control nodes.** A Slider, RGB Slider, Colour Wheel or Label
  inside, cabled into what it drives, is a knob on every instance. **panel**
  arranges them (drag one anywhere on its body; it snaps to the grid when
  you let go). Ctrl-drag a row, or a Grade wheel, out of an inner node's
  properties onto the panel to make its knob in one move.
- **Where it lives.** A collapsed subwob is defined IN the comp: it saves
  with the setup and undoes like any edit, so a comp never depends on a
  file being installed. The top bar's **subwobs** button saves a selection
  as a library subwob instead (`~/.wobble/plugins/<id>/tool.json`) that any
  comp can add from the menus.

## Pipes: image, matte, data, fat cable

Every wire is in a PIPE, and the pipe decides what it can plug into. There
are few of them on purpose, and each means something the others do not:

| pipe | carries | colour |
|---|---|---|
| **image** | rgba, colour, in the working space | cost (cyan / red) |
| **matte** | ONE channel of coverage, always 0–1 | lime |
| **data** | ONE channel of measurement at full depth — depth, disparity, an id | violet |
| **fat cable** | every layer of a file, lazily, until something picks one | teal strands |

Matte and data are the same shape with opposite contracts. A matte is
coverage: a value outside 0–1 is a bug, so it can be kept cheaply. A depth
is a measurement: clamp it at 1 or quantise it and the defocus is wrong.
So **a matte may go into a data port** (it loses nothing), **data may not
go into a matte port** (which depth is white? there is no answer that is
right for every shot), and **an image dropped on a data port gets a
ToData** in the gap — pick a channel, no clamp, no re-range. A Defocus's
**Z** input is a data port; setups saved before it was load with the
ToData already spliced in, carrying the channel the Defocus used to pick,
and render exactly as before. Normals, positions and motion vectors stay
on the image pipe for now.

**Setting focus.** A Defocus's `focus` is the value IN the depth pass that
lands sharp — the number a pixel of the pass reads, unbounded, so a
world-space pass at 2400 is focused with 2400 (with 1/Z it is the 1/Z
value). **pick depth** arms the viewer: click, or drag for a live focus
pull, and the node's own Z texture under the pointer becomes the focus
(keyed when the channel is animated). **focal plane** views the node as its
sharp input tinted green in focus, red in front, blue behind, deeper with
the CoC — only while it is the view target, never in a render. **1 unit =**
says what a depth unit is in metres (m / cm / mm), because the lens is in
millimetres; it scales the blur, never where the plane is.

**Modulate.** A Defocus's last port (`~`) is a matte that scales the
circle of confusion per pixel, on top of whatever the depth pass says:
white is the blur the lens asks for, black is sharp. Depth stays correct
and the matte is for hacks — a hand-drawn tilt-shift, a pulse, a noise
field. It can only take blur away (it is clamped to 0–1 under `radius`),
and without depth it scales the uniform disc. Unwired, it costs nothing.

**One plate, two layers.** A Z-pass defocus has a problem no maths removes:
a foreground that goes soft should reveal the background behind it, and a
single plate never had those pixels. What the node does about it, both
modes: a **CoC map** is derived from Z once per frame (the signed CoC and an
8×8 tile max dilated by the max radius — "the widest blur that could reach
here"), and **edge** takes the NEAREST depth within n px so an antialiased
rim travels with its object instead of haloing. **Fast** mode is then two
layers: blurry pixels in front of focus are the NEAR layer, gathered at the
dilated radius so a foreground bleeds OUT past its own edge; everything
else is the FAR layer, gathered at its own CoC with the foreground
EXCLUDED, so under a soft edge it is filled from the background around it
— the best guess for the pixels that were never there. Near is composited
over far by its blurred coverage. A small object thinner than its CoC goes
see-through at its centre, as it does on a lens; deep inside a big one
there is nothing to fill from and the foreground stands. **Accurate** mode
searches each pixel's own reach off the same map (it used to spread every
pixel's taps over the max radius and land a handful), rotates its spiral
per pixel so under-sampling is grain rather than a fingerprint, samples
bilinearly, scales its tap count with the disc (`samples` is the count at
max CoC, floor 32), and **clamp** soft-limits a tap's brightness before it
spreads, for HDR highlights that came out as a scatter of dots.

**The real fix is layers.** When you have the 3D scene, render the
foreground and the background as separate passes (a Blender view layer
with the foreground hidden, or a Cryptomatte split), run a Defocus on each
with the same lens, and comp them with a Layer or Blend: then the
background behind a soft edge is real pixels, not an estimate. The node is
built so the same settings give the same blur on every layer.

## Mattes

A matte is a typed PORT, not a channel. Every connection is either an image
or a mask: `mask_input()` declares input 2 as the matte slot on Grade,
Defocus, Glow, Mult, Mask and Keyer (input 3 on Blend; a Spline's only
input is a matte too), and `output_kind()` reports Mask
for Spline, LumaKey, Keyer and ToMatte. The editor type-checks on connect, so
a mask socket will not take an image wire — mask sockets are drawn green and
rest horizontally, leaning toward the nearest mask output.

Underneath there is nothing special about the data: a mask node writes its
coverage into rgb AND alpha (rgb so the shape reads on its own, alpha so the
viewer's `A` toggle — and anything that composites it — sees a real matte
rather than a solid 1), and consumers read `.x` and crossfade
`mix(original, effect, mask)`. An unwired or dangling matte means coverage 1
— the effect applies everywhere, not nowhere — and so does a BYPASSED mask
source, so switching a matte off turns the effect on everywhere rather than
killing it. (The `Mask` node is the odd one out: it multiplies by coverage
instead of crossfading.)

**Mask vs modulate.** A matte can reach a node two ways. The **mask**
mixes the finished effect back against the input: at 0.5 you see half the
sharp picture and half the full blur. The **modulate** input (`mod_input()`:
Blur, Dilate, Transform, CornerPin) scales the effect ITSELF per pixel: at
0.5 you get a blur of half the size, a dilate of half the distance. Both
take a matte wire; the modulate socket wears a `~` (and, on a node that has
both, sits on the side opposite the mask). A **Blur's one matte port is the
modulate** — a mix-back mask is never what a blur wants. Unwired, a modulate port costs nothing — the node runs the pass it
always did, with no subtree rendered for the matte. Blur's modulated path is
separable with a per-pixel sigma: exact wherever the matte is flat, with
faint vertical streaks where it has a hard edge. Dilate's is exact.

**A modulated Transform** is the UV map + UV Warp trick without the UV
map or the UV Warp: the matte on its `~` port says how much of the move
applies at each pixel, and since a Transform only moves the POSITION the
nodes above it are sampled at, that is one blended position in the fused
pass — no extra pass, no resample, still cyan. Two blends: **params**
(default) applies the transform at partial strength — move × m, rotate × m,
scale to the power m, pin corners m of the way — so a rotation sweeps arcs;
**coords** mixes the positions, exactly that trick, where a rotation cuts
chords and shrinks toward the pivot. The matte is read where pixels LAND
(output space), as usual. The gizmo, the layer pick and the soft-crop
conform still reason about the full-strength transform. **CornerPin**
takes the same `~` port and the same two blends: params moves each corner m
of the way from the frame's own, coords mixes the positions.

**MatteCombine** is the algebra: two matte inputs, one matte output, and an
op — `union` (max), `add` (clamped sum), `replace`, `subtract` (A punched by
B), `intersect` (A×B), `intersect (min)` and `difference`. A key, a spline,
and a luma matte are all the same kind of stream, so a chain
of these is how a garbage matte bounds a key, how two shapes make a hole, or
how you SEE where two mattes disagree (difference, wired to the viewer).

Both soft and hard forms of union and intersect are offered because coverage
is FRACTIONAL and they genuinely differ on every soft edge: `a*b` is right
when the two mattes are independent coverage, `min(a,b)` when one is only
there to bound the other, and `max` versus `add` is the same choice for the
other side — `add` accumulates, so two half-covered edges make a solid pixel,
which is what you want for independent coverage and what seams when it is
the same edge twice.

Two conventions worth knowing, both the opposite of the mask-input rule
above: an unwired operand is EMPTY coverage (so a combine with one input
wired is that input, rather than a solid frame), and `mix` at 0 bypasses to
A. The op joins the topology hash — a different formula is a different
shader, exactly as with a Blend — while `mix` is a uniform. The same
`MatteOp` enum drives a Spline's own combine port, so the vocabulary is one
vocabulary: what `subtract` means on a Spline is what it means here.

**MatteIn** goes the other way: it copies the matte on input 2 into the
stream's ALPHA, colour untouched — what you reach for before a Premult, or
before writing a matte out alongside the plate. An unwired matte reads opaque
there too. With ToMatte it round-trips.

**ToMatte** is the bridge from any image into that world: it picks a channel
— r, g, b, a or luma — and relabels it as coverage, with an invert and a
lo/hi range for sources that are not already 0..1. Luma is the default
because most things you want as a matte (noise, a pass, a grade) carry alpha
1 everywhere, and a flat matte silently does nothing, which is the worst way
to be wrong.

You rarely have to reach for it by hand: **drag an image output onto a matte
input and the editor inserts one for you**, halfway along the wire you just
drew, wired to both ends. The gesture you tried is the gesture that works.
Compatible pairs still connect directly — nothing is invented where nothing
is needed — and a mask at an image input is still refused, because there is
no obvious answer for what it should become.

## Spline masks

A fresh Spline is EMPTY — click in the viewer to lay control points down.
Shift-click the outline inserts a point into that segment, right-click
removes one, drag moves it, and alt-drag sets THAT point's feather (the soft
edge interpolates along the outline, drawn as an offset ring). Four shapes:
linear, catmull-rom, bezier (per-point handles, mirrored unless alt) and
b-spline; switching is a uniform write, never a recompile. Points, handles
and per-point feather all keyframe (channels `SplinePtX/Y/F`).

**Tension** (catmull-rom): each point has its own. Hold **K** and drag a
point — right tightens toward a corner, left loosens — or pick points and
use the *tension* slider in the panel. 0 is the ordinary smooth curve, 1 a
corner, below 0 rounder. The curve still passes through every point; only
how it turns there changes. It is a param, never a recompile, and the same
control is on every spline: the comp Spline, a noise graph's Roto, and a
2dini's Spline and Stroke. Like the Bezier handles it is authored per point,
not a keyed channel.

**Matte controls.** `opacity` gains the coverage, `invert` takes the
complement (an EMPTY inverted shape is a full matte, which makes it a
garbage switch rather than a no-op), `expand` slides the outline along its
own normal in output pixels — the soft edge rides with it instead of being
eaten by it — and `falloff` bends the ramp from linear to smoothstep.
Falloff fixes 0.5 either way, so the viewer's two feather rings mean the
same thing at every setting. All four are uniforms: scrubbing them on a
40-point shape does not recompile.

**A shape transform** (`translate`/`rotate`/`scale` about a `centre`) sits
between the authored points and what is rasterized. A shape that merely
moves is then five curves instead of two per control point, and those five
are patch jacks like any other — cable a Tracker's `TrackX`/`TrackY` through
an `add` value node that subtracts the track's start position (the tracker
reports a normalized POSITION, `translate` wants an OFFSET) and the shape
follows the plate. It is applied to the CONTROL points at
pack time, which is exact because every mode is an affine combination of
them, so it costs the shader nothing. In the viewer a point still grabs
where it is DRAWN and follows the cursor: drags are pulled back through the
inverse, so moving a point on a rotated shape moves that point, not the
rotation. "bake transform" folds it into the points and resets it.

**Several shapes are a chain.** A Spline has one optional matte input and a
`combine` op — union, replace, subtract, intersect, difference — so a roto
is one node per shape, wired up, with no second editor to learn: subtract
punches a hole, intersect trims one shape to another. Unwired, the input
reads as empty coverage, so the default union is exactly the bare shape (and
every setup written before the input existed renders identically). The op
picks the emitted expression, so — like a Blend's — it is part of the
topology hash; everything else about a Spline is a uniform.

The chain is bounded by the uniform block, not by anything else: a shape
spends the whole sample budget whatever its point count (~146 vec4 slots),
so about **seven** shapes fit in one chain before the topology is rejected.
`a_chain_of_shapes_fits_the_uniform` pins that so it cannot quietly shrink.

## Paint

Strokes are vector data on the node — a polyline, a radius, a softness, a
colour, a mode — resampled into instanced soft ellipses and splatted into
persistent per-node layer textures. A layer is a MEMOIZED rasterization,
re-recorded only when its stamp set changes, so painting never recompiles a
shader and playback over static strokes pays nothing. The fused pass reads
the layers like any other source.

Select the node and left-drag in the viewer. The brush cursor is drawn where
the paint will land — the ellipse at its aspect and angle, an inner ring
where the falloff starts — through everything downstream of the node, so it
squashes with a transform below it exactly as the stroke will.

**The brush by hand.** `[` and `]` resize it (multiplicatively — a 4px brush
and a 200px brush both want ~20% steps), `shift` with them takes softness
and `alt` takes opacity. Blender's F does it by MOTION: hold `F` and move —
x sets size, y softness; hold `shift+F` for opacity (x) and angle (y);
`ctrl+F` for the ellipse's aspect. The hand painting never leaves the
tablet, and nothing paints while F is held. All of it is live only while a
Paint node is selected, and remappable in the keymap. (Not the Photoshop
1-9 opacity digits: `1` and `2` are the viewer's A/B slot keys here.)

**Six modes**, grouped by what they cost — which is the thing that actually
matters in a fused pass, and which the panel says in amber:

| mode | what it does | what it costs |
|---|---|---|
| **colour** | composites the brush colour | nothing: the layer is `over`'d |
| **dodge** / **burn** | exposure up / down by `stops`, premultiplied by opacity and ADDED across strokes, so a dodge brush builds up | nothing: pointwise on the pixel already there |
| **clone** | the PAINTED picture at an offset — hold alt (or cmd) and move: the brush stays put while the source follows the hand; let go and every stroke keeps that offset (aligned) | the input read once more, per pixel |
| **smear** | the painted picture from `drag px` back along the stroke, per stamp — a smudge | the input read once more, per pixel |
| **blur** | a ring of `PAINT_BLUR_TAPS` (8) reads at the mean `blur px` under the pixel, plus the centre | the input read 8 more times, per pixel — the heavy one; the tap count is fixed, so a wider blur is sparser, not slower |

Each mode has the opacity every brush has plus at most one more knob
(`strength`: blur px, drag px, or stops), kept per mode in the brush so
switching never hands a dodge the blur's radius.

A clone (or smear) samples what is UNDER its source, strokes included: the
input there, dodged or burned as it is there, with the colour strokes over
it — so a colour stroke you already laid down can be cloned. Not the blur
(its taps would multiply) and not another clone (a clone of a clone is a
loop). It is live, like everything here: a colour stroke painted over the
source later changes the clone.

**Blend modes.** A colour stroke lands **normal**, **erase** (takes earlier
paint back off, never the plate), **add**, **subtract**, **multiply** or
**screen**, picked on the panel's brush tab. Order holds across modes: a
multiply laid after a stroke darkens it, one laid before is covered by it.
Every mode is an affine map per channel (`x -> x*m + b`), and a chain of
those is one, so the colour layer stores the composed map — `B` (the old
premultiplied layer) plus, once a stroke uses add/subtract/multiply/screen,
a multiplier texture `M` — and the fused pass applies `below * M + B` once.
The layer never reads the plate, so it stays memoized however the plate
moves; normal and erase alone keep the one texture they always had.

**Pen pressure.** Wacom tablets and Apple Pencil over Sidecar on macOS (read
off the tablet events winit drops, `app/src/pen.rs`), pen pressure on
Windows, and `PointerEvent.pressure` in a browser (an iPad one day). Linux
has none yet (winit). Each stroke keeps a pressure per point and the map it
was painted with (the **pen** tab: how far pressure moves size and opacity,
and a curve), so changing the response later never reshapes old strokes; a
mouse stroke records none and paints exactly as before. The brush also
takes EVERY cursor sample between frames, not just the last, so a quick
stroke is a curve rather than a polygon.

**Brushes.** The brush tab's preset row (pencil, ink, airbrush, matte,
eraser, dust, dodge to start) is app-wide, in prefs: click to use, `+` to
keep the current brush, shift-click to overwrite one, alt-click to remove
it. A preset is mode, size, edge, shape, opacity, blend and pressure
response — never a colour (the picker's palette has those).

**Bake.** A Paint node that clones, smears or blurs is slow, and says so. If
that is fine — it usually is — the panel's **bake** button puts a `GraphBake`
on the node's output, rewires everything downstream to read the cache, and
starts the bake: the red clears, and the strokes are still yours to edit (the
bake goes stale, never rebakes behind your back).

**The key legend** sits in the viewer's corner while a Paint node is
selected, and the line for whatever you are holding lights up — the cheap way
to see that F is being heard.

Under the hood there are three layers per node, each allocated on a class's
first stroke and dropped with its last: **colour** (premultiplied rgba,
`over`), **source** (clone + smear: `rg` = where to fetch the input from in
output px, premultiplied by coverage so overlapping strokes AVERAGE their
offsets; the fused pass divides it back out), and **effect** (blur, dodge and
burn: `r` = stops, `g` = blur px, `b` = blur coverage, `a` = coverage,
blended ADDITIVELY because these are amounts). Each class is a bit in the
topology tag, so a shader carries exactly the extra reads its strokes need
— the first stroke of a class recompiles once, every stroke after it is just
more stamps — and a node that has only ever painted colour costs what it did
before any of this existed. `--test-paint` and `engine/tests/paint.rs`
pin the read counts per mode.

**Output.** `rgb` (the default) is the input with the strokes applied.
`matte` puts the strokes' COVERAGE on the output — every mode counts,
unioned with whatever alpha came in on the input, so a keyer's matte can be
touched up by hand — and turns the socket into a mask. A matte never
re-reads the input, whatever modes the strokes are.

**Lifetimes.** Every stroke carries one: all frames, a single frame, or a
range. Only "this frame" follows the playhead — the in/out of a range are
frame numbers you typed, so painting somewhere else keeps them. Set the
node's default to "this frame" and it is a per-frame dust bust; leave it on
"all frames" and it is a matte painting. The lifetime is resolved when the
layer is rasterized, not in the shader, so scrubbing a node with nothing
timed on it re-records exactly never — `stamps_for` returns identical bytes
on every frame. The panel counts what is live on the playhead, drops just
those strokes ("clear frame"), and re-times the newest one either way.

The frame a lifetime is judged against is the EVALUATED tick — the one whose
pixels are on screen — not the clock a user shader reports as `P.frame`,
which keeps counting through a playback loop.

## Colour keyer

The secondary-grading keyer: isolate *a colour* — skin, sky, foliage, a
particular red — as a soft mask, and grade through it. Not a green-screen
keyer; nothing here assumes a backing colour.

**Selection is a union of soft ellipsoids in [Oklab](https://bottosson.github.io/posts/oklab/).**
Each sample is an ellipsoid whose axes are the three things an artist names —
**lightness**, **saturation** and **hue** — measured on a frame aligned to
that sample's own hue. Working perceptually rather than in HSL or YUV (which
is what Flame's 3D Keyer and Resolve's 3D qualifier do) buys three things:

- **One softness knob means the same thing everywhere.** Hue degrees are
  worth wildly different amounts at different saturations; Oklab distances
  are not.
- **Hue stops mattering as a pixel desaturates**, automatically, because hue
  is measured as a distance in the a/b plane rather than as an angle. That
  single property removes the speckle an HSL qualifier gets in shadows and
  neutrals, where hue is noise and the qualifier believes it.
- **Scene-linear values above 1.0 stay meaningful.** Oklab is a cube root,
  not a transfer function with a domain. The comp is scene-linear, so the
  keyer has to be.

Samples union as `1 - Π(1 - mᵢ)`: order independent and creaseless, where a
`max()` union leaves a ridge that shows up in the grade. Negative-weight
samples subtract by the same formula, so "this, but not that" is one gesture.
On top of that: a **tilt** that leans the ellipsoid so saturation tolerance
tracks lightness (real subjects do this — skin loses chroma into highlight),
matte **levels** and an **edge** exponent, **invert**, and a garbage-matte
input on input 1.

It is entirely pointwise: a handful of ALU fused into the same shader as its
neighbours, no graph split, no extra pass. Adding a sample recompiles once
(the slot span grows, like a Spline's points); moving one never does. Every
centre, tolerance and softness is an animatable channel.

**Sampling.** With the keyer as the view target, drag over the image: the
pixels under the cursor come back through a small async readback and an
ellipsoid is fitted to their Oklab mean and per-axis spread, with one
outlier-rejection pass so a drag that clips an edge doesn't inflate the key.
Alt-drag fits one that subtracts; shift-drag re-fits the selected sample
instead of adding one. The first drag on a fresh node replaces its
placeholder rather than joining it.

That gesture is why the node shows its own INPUT (matte in alpha — press `A`)
while it is the view target: the pixels on screen are the pixels being
judged. That preview is transient, applied to the engine graph only for the
node being viewed, so downstream consumers and renders always get the matte.

**The panel** is two projections of the ellipsoid, each painted with the
colours it represents, each carrying a scatter of the image's own pixels and
each shading the live matte by knocking back what is *not* selected:

- **hue/chroma** — the a/b plane at the sample's lightness. Out-of-gamut
  regions are darkened rather than clipped, so the display boundary reads as
  a shape instead of a wall of primaries.
- **lightness/chroma** — a slice through the sample's own hue, where the
  tilt is visible as a lean.

Drag the pucks: centre moves the colour, the rim pucks are the saturation and
hue tolerances, the opposite puck is the softness ring, and the lightness
puck also shears the ellipsoid sideways. Under them, three **range brackets**
— one per axis, drawn over the colour ramp that axis sweeps — give precision:
grab the middle to slide the window, an inner edge for tolerance, a shoulder
for softness. The **falloff ramp** is the matte's own response curve with the
two level stops on its floor and a puck in the middle that bends the
roll-off.

Wire the output into any node's mask input (Grade above all). For a spatially
softer key, put a Blur between the keyer and its consumer — the keyer itself
stays pointwise on purpose.

## Cryptomatte

Id mattes the way renderers write them ([the Psyop spec](https://github.com/Psyop/Cryptomatte)):
a `CryptoObject` / `CryptoMaterial` / `CryptoAsset` layer is pairs of
(id, coverage) ranked by coverage — `CryptoObject00.RG` is rank 0, `.BA`
rank 1, `CryptoObject01` ranks 2 and 3, and so on — and the id is a float32
whose **bits** are a MurmurHash3 of the object name, with the name → id
manifest in the EXR header.

**Wire it:** set the Load to all-channels output and drop the fat cable on
the Cryptomatte's one input. Pick the cryptomatte (`CryptoObject`,
`CryptoMaterial`…) in the panel. The node owns its readers — the file's
rgba plus one per rank (six, the renderer default) — each a reader of the
Load's sequence on a virtual layer named `CryptoObject00#0`,
`CryptoObject00#1`, `CryptoObject01#0`…, exactly the way a ChannelSelect is
a reader of one layer: same file, own reader slot, same cache flags, and the
`#rank` suffix is what tells the decoder to read that pair **as ids**. The
cable is a lazy reference, so nothing but those pairs decodes or caches on
the node's account.

**Why the decoder has to know:** reader textures are f16, and an id survives
nothing — not a half conversion, not a filtered sample. A rank layer is
decoded into three 11-bit integer chunks (every integer up to 2048 is exact
in f16) plus the coverage, one rank per texture, and the fused pass loads
the texel **unfiltered** and reassembles the u32 before comparing. Picks are
uniforms (two 16-bit halves each, up to 32 per node): picking never
recompiles.

**Pick objects** three ways: type a name (hashed, or taken from the manifest
when it knows the name), tick it in the panel's manifest list, or view the
node and use the usual hands on the image — **ctrl-click adds** an object
(cmd on a Mac), **ctrl+shift-click removes** it, and a plain click is not a
pick. While a Cryptomatte is viewed, ctrl+LMB pan and the Tab scrub stand
down (MMB, alt+LMB and the wheel still navigate). The click reads the pixel
on screen, so the node has to be the view target; viewed, it shows its
**ids picture**: every object its own flat colour, picked ones lit bright yellow (the usual colour),
the matte in alpha (press **A**). That picture is rank 0's chunks scaled
into [0, 1) (picked: red and green pushed above 1, blue quartered), exact
through the f16 output either way, which is what makes a click — and a
removing click on a lit object — decodable. The picture is transient,
Keyer-style: only the node being viewed renders it, and a downstream
MatteIn or Blend always gets the matte.

The output is a mask: coverage summed over the picked ids across all six
ranks, so a soft edge shared between two picked objects sums back to 1.

**It bakes itself.** Seven f32 layers a frame is what this node costs, so
nobody downstream pays it twice. The matte is the node's own eighth reader:
its decode computes the matte from the id pairs on the decode thread, and
the NVMe tier keeps the result like any decoded frame — so **every frame
that is ever requested is baked**, and the next request is a plain cache
read of one matte. The cache key is the selection itself (layer, sorted
ids, invert), so a pick edit starts a fresh set and can never replay
another selection's frames; the old set ages out of the tier's budget.
**bake all** in the panel warms the remaining frames on background threads
(the player is not involved; cancel loses nothing) and the row shows
`baked n/m` for the current selection. The id layers are in the rig only
while the node is the view target, where picking needs them; hop the view
downstream and they leave it. With no NVMe cache directory set the panel
says so: the matte is then recomputed per request.

`gen_exr … crypto` writes a test plate (three named discs with a manifest);
`--test-crypto` drives the whole loop in the app.

## Caching tiers

Three tiers, each optional and observable in the stats HUD:

Scrubbing and playback take different paths through the scheduler and are
tuned separately. Playback pipelines decodes ahead of the playhead; a drag
re-targets it every ~16ms, faster than any real plate decodes, so the scrub
path instead keeps every decode it has paid for, stops speculating ahead of
a moving target, caps in-flight work so landed frames have somewhere to sit,
and shows the nearest decoded frame while the exact one is still coming.
`--test-scrubperf` measures it: a continuous 60Hz drag over two 2K reads
displays ~60 fps, and over two 4K reads ~17 fps (decode-bound). Turning a
Load's **RAM cache** on makes a second pass over the same range immediate.

- **RAM frame cache** (`--cache-gb N`, default = RAM/4 clamped 4–24 GiB;
  loop-aware pinning instead of plain LRU): first loop decodes, after that
  playback and scrubbing are memcpy + upload (~120 sets/s at 4K).
- **NVMe frame cache** (cache… in the menubar sets the directory + byte
  budget, persisted to `~/.wobble/prefs.json`; `--cache-dir` overrides):
  decoded frames persist as raw f16 `.gcf` files in the staging layout, so
  a disk hit is one sequential read (~10–20 ms at 4K) instead of a 100–400 ms
  EXR decode. Keys hash path+size+mtime (an overwritten plate misses instead
  of serving stale pixels); eviction is oldest-first under the budget.
  Every Load node has **RAM cache / NVMe cache** toggles and a **frame
  range** (in/out) in its properties.
- **A Load's own proxy** (its panel's **proxy** menu: 1, 0.75, 0.5, 0.25;
  the panel shows the size kept, e.g. `3072x1620 cached`). The frames are
  area-averaged once, right after the decode, so RAM, NVMe, the staging ring
  and the GPU all hold the small frame: a 6K plate at 0.5 takes a quarter of
  the space and loads that much faster, and the rest of the comp keeps its
  resolution. The Load stays the size it was: its format, the canvas, and
  every Transform, roto shape and tracker in pixels are unchanged, only
  softer. It multiplies with the comp-wide proxy, each scale has its own cache
  entries, and it is saved with the comp. **Renders use it too**, so set it
  back to 1 before a final Write that needs the plate's full detail.

**A Load's range can run past its source, so you can watch a render arrive.**
Set OUT beyond the frames that exist and the timeline covers them anyway:
what is not on disk yet reads as black, playback runs straight through it,
and each frame appears on its own as the renderer writes it — no reload, no
rescan button. Nothing about an absent frame is cached (in either tier), so
it cannot get stuck as black once the file lands. The panel says how many
frames are on disk and how many are still pending.
- **Viewer render cache** (timeline, "cache viewer" — a deliberate action,
  never automatic): bakes the viewed comp tick-by-tick into the NVMe cache,
  then plays the baked frames straight into the output texture — the comp
  passes stop running entirely. Two storage modes: **RAM** (baked frames
  fill the RAM budget; the read caches drain to their NVMe tier while the
  viewer holds the RAM) and **NVMe** (frames stream from disk every loop —
  measured: a 4K comp holds a 60 fps target streaming ~4.2 GB/s, RAM
  untouched). Any comp edit (params, wiring, loads) invalidates it;
  identical comps resume a partial bake. Baked sequences are keyed by a
  content hash of topology + params + sources + output size.

The bench runs with all caching disabled so decode numbers stay honest. The
app decodes per-file parallel (lower scrub latency); `bench` keeps
`--parallel-decode` as a flag.

Measuring NVMe honestly: any working set that fits in RAM is served by the
OS page cache after one loop, so "disk" numbers lie. `WOBBLE_NOCACHE=w`
bakes the tier with uncached writes (nothing resident) while keeping
buffered reads + readahead — the realistic beyond-RAM cold case;
`WOBBLE_NOCACHE=1` also forces direct reads (QD1, no readahead — the
floor). macOS only (F_NOCACHE); `--fps N` sets the playback target from the
CLI for paced measurements. Measured on real 4K-square plates: cold NVMe
sustains 24 fps with no RAM cache (see RESULTS.md).

## Properties tiles + HUD sliders

**Ratio locks.** Values that go together carry a small padlock between
their fields: a format's width × height (every generator, Reformat, the
script format), a Transform's scale (the lock is its saved gang, closed by
default), a roto shape's scale, a Grid's spacing, a 3D card's scale and a
camera's filmback. Closed, editing one moves the others by the same factor,
so the ratio holds; on an animated channel the followers are keyed too.

## Frame numbers

**A frame is a number, and it is the same number everywhere.** The playhead,
the timeline readout, keyframes, in/out points, handles, a Load's range, a
Render's range and every bake all speak one language: the comp frame. There
is no separate "tick" that counts from zero.

**Every noodle carries a frame range.** Hover a wire and it says its first
and last frame. A Load's range is its own **first / last** (below); a
generator runs for the **script range** unless its own `range` row says
otherwise; everything else inherits what feeds it — the union of its inputs,
moved by any Slip or Speed on the way. Any node can pin its own range.

- **Load first / last.** Explicit fields in the Load panel, filled from the
  files when the sequence loads (a render numbered 1001..1069 reads 1001 /
  1069) and editable. A frame inside the range with no file on disk is
  **black and never cached** — push `last` past the end to watch a render
  arrive; the frames fill in as they are written. `auto` puts them back on
  the files. The Load's `start` places the clip on the timeline (auto: the
  file numbers) and moves first/last with it.
- **Before / after.** What a Load answers for a frame OUTSIDE its range —
  hold (default), loop, bounce or black, per end. Channels picked off a fat
  cable and Cryptomatte ranks answer the same way as their Load.
- **The viewer follows what you are viewing.** The timeline shows the range
  of the noodle you view; switch the view and the bar follows, and if the
  playhead is left outside the new range it moves onto the first frame. The
  transport's `range` pins it by hand — frames it then asks for beyond a
  source's own are that source's before/after. A hairline under the track
  marks the viewed clip's own frames when the pin reaches past them.
- **The script range** (menubar `script`) is what generators run for and
  what a shot delivers to the edit. Unset, it spans the loaded plates.
- **Playback loops** the in/out stretch, the cut or the handles — the
  transport's `loop` buttons — footage or not.
- **Animation evaluates at the frame on screen.**

The transport is two rows: the track, the full width of the window, then
every control under it — frame buttons and readout, autokey, the viewer
range, loop, handles and in/out, fps, resolution, the viewer cache.

Self-tests: `--test-frames` (a Load placed at 1011, viewed directly),
`--test-range` (noodle ranges downstream of a placed plate, the viewer and
playhead following them, a generator's script range and its own, black /
hold / loop past the end, first/last filled from the files and black —
uncached — where there is no file), `--test-bake` (a GraphBake scrubs frame
for frame with its live subtree, placed or not).

## Animation

Any scalar param can be keyframed (transform, all grade channels, blur /
defocus size, luma-key thresholds, corner-pin corners). Curves are authored
data on the node — setups, undo/redo, the A/B viewer slots and the
render-cache key all carry them automatically — and playback evaluates them
transiently at the DISPLAYED tick into the engine graph; the engine itself
never learns what a keyframe is, and evaluated values are never written back.

Authoring: the ● button on each properties row sets/removes a key at the
current frame (right-click clears the channel). "auto" on the timeline arms
autokey: gizmo drags and property edits key themselves. (A full curve
editor is planned; the interim per-channel track editor was removed.)
Red ticks over the scrubber mark the selected nodes' keyframes, with ◀●/●▶
navigation. Gizmos draw from and drag against the EVALUATED values (through
animated downstream transforms too) and write keys on animated channels.
Interpolation is per key — hold, linear, smooth or bezier — with held ends;
keyframes live in timeline frames.
Viewer-cache bakes evaluate per tick, so cached playback bakes real motion.
The shot has its own **frame range**, and it does not need footage to have
one. With plates loaded the range is a window into them; with none — a comp
that is purely generative, a Spline or an Fbm with no Load anywhere — the
range *is* the timeline, defaulting to 100 frames. Either way the clock runs,
the playhead shows, scrubbing works and animation evaluates against it, so
you can key a shape with nothing loaded at all.

`Y` opens the **curve editor**: a value-over-time graph of the animated
channels on the selected nodes (only animated ones — static parameters never
appear; with nothing selected it shows everything animated in the comp).
Keys drag in frame and value, marquee or shift-click selects several to move
together, shift-click on a curve inserts a key and ctrl-click (or Delete)
removes. Its two axes are decoupled: ⌘/middle-drag pans, scroll zooms about
the cursor, shift+scroll stretches time alone, alt+scroll value alone,
⌘|alt + right-drag dollies both independently, and `R` frames what's visible.
The **channel list** on the left groups channels under their node: a click
isolates one (shift-click adds or removes; a node row does its whole group;
`all` / `none` sit in the toolbar), because a whole animated comp is far too
many curves to read at once. The **pin** at a row's right end keeps that
channel in the editor after you select something else, so one node's curve
can be lined up against another's. Select several keys and a box appears around them:
its edge and corner handles scale the group in time and value about the
opposite edge, and dragging a key with **shift** locks the move to whichever
axis you pull; hold **Z** to move in time only, **X** for value only (also
for handles and the scale box). **Frame snap** (`shift+S`, or `snap` in the
toolbar) keeps moves, scales and inserts on whole frames; turn it off and
keys can sit between frames (whole-frame keys still save as integers).
Select keys and the toolbar sets their interpolation on **both sides** — the
segment arriving and the one leaving (shift-click sets only the leaving
one): **hold** (step), **linear**, **smooth** (auto-tangents, flattened at
extrema and curve ends so eases never overshoot) or **bezier** (handles
seeded from the current slope, then dragged freely). Hold `` ` `` over the
plot for the **curve-ops pie**: the four modes, flatten (level bezier
handles), snap, frame all and select all. The hotkey strip across the top
lists all of it. Key glyphs show the mode — square stepped,
diamond curved, dot linear. Old setups still load: keys used to be plain
`[frame, value]` pairs and deserialize as linear.
**Motion paths.** The transport's **path** chip draws the path an animated
element follows, in the viewer HUD: the trajectory over the viewer's frame
range, its keyframes marked, and a filled dot where the playhead is. It
follows whatever moves the element — a translate, a pivot centre, a tracked
point — and a Transform's pivot and a Tracker's point have one today (a node
with no position never draws one, and neither does an element whose curves
leave it standing still). It is off by default and remembered in prefs: it is
another layer over the picture, and the picture comes first.

**Grab the keys on it.** A key on the path is a handle: drag it and that
keyframe moves, at ITS OWN frame, wherever the playhead happens to be — so
you can shape a move by dragging its keys around the picture instead of
scrubbing to each one. The dot under the cursor swells and names its frame.
The edit goes through the same write-back as every other manipulator, so a
locked or cable-driven channel refuses it exactly as it would anywhere else,
and the key's frame never moves — only where the element is when it gets
there. A key sitting under the pivot handle loses the tie deliberately: the
handle you are looking at wins.

**Locking a channel against keyframing.****Locking a channel against keyframing.** Right-click any parameter field —
or a channel's chip in the curve editor — and lock it: that channel takes no
keyframes from anything, while the ones beside it keep animating. Lock
`translate.x` and a gizmo drag, a HUD slider, autokey, a tracker bake and the
curve editor all leave x exactly where it is and key y as usual. A locked
field wears a bar underneath it, its key dot goes grey, and its curve is
drawn in the lock colour with its keys untouchable (they are still visible —
locking freezes animation, it does not hide it). Right-click again to unlock;
a lock on a channel with no keys leaves nothing behind in the setup.

Self-tests: `--test-anim` (engine-side interpolation over playback),Self-tests: `--test-anim` (engine-side interpolation over playback),
`--test-gizmokey` (paused mid-curve pivot drag becomes a key; then with
`translate.x` LOCKED the same drag keys y only; then the motion path draws
once switched on, and dragging one of its keys moves that key alone),
`--test-curves` (a keyframe drag lands on the exact frame and value).

## Alpha feedback + crash reports

Crashes report themselves (`app/src/report.rs`):

- **Panics** on any thread write a report to `~/.wobble/crash/` — message,
  backtrace trimmed to our frames, log tail, build + OS + GPU, the comp that
  was open — and the **next** launch uploads it in the background. No
  network in the dying process; nothing lost offline.
- **Native crashes on Windows** (an access violation in a GPU driver, …):
  an unhandled-exception filter (`app/src/report_win.rs`) writes the
  exception, the faulting module, the stack symbolised from `wobble.pdb`
  (shipped beside the exe) and a minidump. Reports name the GPU driver
  version.
- **Everything else that kills a run** — aborts, stack overflows, hangs,
  kills — never reaches a hook. Each run holds a lock on
  `~/.wobble/sessions/<id>.lock`; a launch that can take a dead session's
  lock reports it, with that session's log (`~/.wobble/logs/`) and the OS's
  own record of the pid: the Windows event log (Application Error /
  Application Hang) or macOS's `.ips` crash report.
- **Autosave**: the comp is snapshotted to `~/.wobble/autosave/` every 15 s
  while it differs from what is on disk; after a crash the next launch
  offers it back (*recover your comp?*).
- **Feedback**: the **α** button on the top bar (or help → feedback…) — bug
  or feature request, with the log and the comp attached for bugs.
- **Stats**: per session, how long, clean or not, which node kinds.
- **New builds**: a **new build ↗** chip appears when a newer alpha is
  published.

First launch shows an alpha notice (name, invite code, and switches for
crash reports and usage stats); nothing uploads before it is dismissed.
Headless runs and self-test suites (`--test-*`, `--dump-after`) disable
the whole path. Backend, invites, releases and triage:
[tools/feedback-worker/README.md](tools/feedback-worker/README.md); publish
with `tools/release_alpha.sh "notes"`.

```sh
WOBBLE_REPORT_SELFTEST=1 wobble    # test report + flush outbox + latest-build query, PASS/FAIL
WOBBLE_CRASH_TEST=1 wobble         # deliberate panic before a window exists
WOBBLE_CRASH_TEST=abort wobble     # deliberate native abort (unclean-exit path)
WOBBLE_CRASH_TEST=segv wobble      # deliberate access violation (Windows exception filter)
tools/crash_smoke.sh target/release/wobble   # all of the above, checked, isolated outbox
```

## Scripting (Python, any language, AI agents)

One protocol, many clients. Start the app with `--api` (default port 7979,
or `--api PORT`; `WOBBLE_API_PORT` sets the default for app and clients):

```bash
./target/release/wobble data/2k_zip_bg data/2k_zip_bg --api
```

It serves newline-delimited JSON-RPC 2.0 on **127.0.0.1 only**, and is off
unless asked for. The method table lives in `api/src/lib.rs`; `api_schema`
returns it as JSON Schema, and every client builds itself from that, so a
new method reaches Python and agents with no client release.

**Python** (stdlib only, 3.9+): `pip install -e clients/python`, then

```python
import wobble
w = wobble.connect()
plate = w.graph_get()["view"]
blur = w.node_create(type="Blur", inputs=[plate])["id"]
w.node_set(id=blur, values={"BlurSize": 12})
w.node_key(id=blur, keys={"BlurSize": [[0, 0], [24, 40]]})
w.view_set(id=blur)
open("look.png", "wb").write(w.capture())
```

A pipeline script — a publish, a farm job, an AYON or ShotGrid action — is
the same client, and needs no Python inside wobble:

```python
w = wobble.connect()
w.setup_load(path="~/shots/sh010/comp_v003.wbl")
with w.batch() as b:                      # one undo step, all or nothing
    g = b.node_create(type="Grade", inputs=[plate])
    out = b.node_create(type="Render", inputs=[g])
    b.node_patch(id=out, kind={"Write": {"path": "~/renders/sh010.####.exr"}})
w.expr_set(id=b.results[0]["id"], channels=["GainR"], expr="1 + sin($F / 12) * 0.1")
w.run(b.results[1]["id"])                 # renders; raises if the Render fails
print(w.sample()["mean"])                 # scene-linear mean of the viewed frame
```

**AI agents (MCP):** `wobble-mcp` bridges MCP on stdio to the same socket,
one tool per method. For Claude Code:

```bash
claude mcp add wobble -- /path/to/target/release/wobble-mcp
```

**Anything else:** open a socket, write one JSON request per line.

What the protocol promises: it edits authored data (values, keys, wires,
node records — `node_patch` merges into the same JSON a setup stores), and
**every mutating call is one undo step**. Locked and cable-driven channels
are refused, as the gizmos refuse them. Calls run between frames, never per
frame or per pixel. `viewer_capture` waits for the edit to render before it
returns a PNG, so a script (or an agent) can look at what it did.
`cost_get` shows the slow paths and non-commercial licences the graph's
colours do.

What it reaches, beyond nodes, wires, values and keys:

| | methods |
|---|---|
| values driving values | `expr_set` (the text you would type in a field), `patch_get` / `patch_link` / `patch_unlink`, `patch_mod_create` / `_set` / `_delete` |
| keys | `node_key` (`[frame, value, interp]`), `node_key_delete`, `node_unkey`, `node_lock` |
| graphs inside nodes | `dive_enter` / `dive_exit` (subwob, Noise, 2dini, SDF), `subwob_collapse` |
| work that takes time | `node_run` (a Render's render, a bake, a track, a planar track, a CorridorKey key, a Sammie track, a GenAI quote; `cancel`), `jobs_get` |
| looking | `viewer_capture` (display PNG), `pixel_sample` (scene-linear numbers), `viewer_set` (exposure, gamma, one channel, proxy), `view_set`, `time_set` |
| clips | `record` (the screen recorder: `start` the window or the viewer, `stop`, `save` to the clips folder, `status` with the saved paths), `graph_op` (`tidy` / `frame` the graph, as the ` pie does), `ui_set` (the lower pane: properties or curves; pin nodes' viewer HUDs) |
| the comp | `comp_set` (script range, fps, play/stop), `selection_set`, `setup_load` / `setup_save`, `undo` / `redo` |
| many at once | `batch`: one undo step, all or nothing, `"$N"` = the id call N made |

Three things a script cannot do, on purpose: **pay** for a generation
(`node_run` quotes; paying is a click in the app), **download** weights or
install a tool, and enable an addon.

Each method says what it is: `read_only` (safe to call unasked),
`mutates` (one undo step) or neither (it moves the viewer, writes a file
or starts work), plus `destructive` for the ones that can throw work away.
The MCP bridge passes those on as tool annotations, so an agent's client
can auto-approve the reads and ask about the rest.

Not yet: a headless `wobble --serve` (it needs the document model split
out of the UI crate), events pushed to clients (poll `jobs_get`), and roto
/ paint authoring beyond `node_patch` on the raw record.

### Addons

Wobble's equivalent of a Blender addon. It's a folder in `plugins/` or
`~/.wobble/plugins` holding an `addon.json`:

```json
{
  "id": "hello_addon", "label": "Hello Addon", "version": "0.1",
  "run": ["python3", "main.py"],
  "commands": [{ "id": "add_blur", "label": "add blur" }],
  "panels": [{ "id": "main", "label": "Blur", "rows": [
    { "kind": "slider", "id": "size", "label": "size", "min": 0, "max": 100, "default": 12 },
    { "kind": "button", "id": "add", "label": "add blur" }
  ]}]
}
```

- **The manifest declares the UI; the app draws it.** Commands show up
  in the top bar's **addons** menu and in the **addons…** window. Panels
  are floating windows built from `button`, `slider`, `toggle`, `text`,
  `field` and `choice` (a dropdown whose options the process can replace)
  rows.
- **`run` names a program.** It starts in the addon's folder with
  `WOBBLE_API_PORT` and `WOBBLE_ADDON` set, and talks to the app through
  the scripting API above. `addon_events` waits for clicks and slider
  moves, and the addon answers with ordinary calls. `addon_panel_set` and
  `addon_status` write back into its panel. The program can be in any
  language. It can crash without taking the app with it, and nothing it
  does runs per frame.
- **Nothing runs until you enable it** in **addons…**. Enabling an addon
  that has `run` starts a program on your machine. The enabled list is
  saved in prefs, and if the app wasn't started with `--api`, it opens
  the API on a free port for that addon.
- Nodes an addon needs ship next to it as ordinary `plugin.json` (node)
  or `tool.json` (subwob) folders.
- `addon_list` and `addon_command` let a script or an agent see addons
  and run their commands.

In Python, `wobble.Addon` handles the event loop:

```python
import wobble
addon = wobble.Addon()

@addon.command("add_blur")
def add_blur(w):
    blur = w.node_create(type="Blur", inputs=[w.app_info()["view"]])["id"]
    w.view_set(id=blur)

@addon.row("main", "size")
def size(w, value): ...

addon.run()
```

`plugins/hello_addon` is the complete example; `plugins/organisicator`
(project navigation for organisicator-managed shows) is a real one. Not yet: addon hotkeys
(the keymap is still a fixed table), viewer overlays and gizmos, or
commands in the Tab search.

## Build & run

```sh
cargo build --release

# generate test sequences (procedural, RGBA half, scanline)
target/release/gen_exr data/4k_zip_fg 4096 2160 24 zip fg
target/release/gen_exr data/4k_zip_bg 4096 2160 24 zip bg
# compressions: none | rle | zips | zip | piz   (DWAA: decode-only in exr crate)

# headless throughput bench (no window, unpaced)
target/release/bench data/4k_zip_fg data/4k_zip_bg --frames 200 [--slots N] [--ahead N] [--parallel-decode]

# interactive app (sequence dirs optional — they prefill Load A/B;
# otherwise point Loads at directories via the properties bin)
target/release/wobble [data/4k_zip_fg] [data/4k_zip_bg] [--no-vsync] \
    [--cache-gb N] [--cache-dir /path/to/nvme/cache] \
    [--view none|filmic|opendrt] [setup.wbl]

# with OpenDRT (GPLv3 — see below). Note the package-qualified feature:
# a bare `--features opendrt` at the workspace root silently does nothing.
cargo build --release -p app --features app/opendrt
```

## Model weights

ML nodes need weights that are far too big to commit, so Wobble ships a
registry and fetches them on request. **models…** in the toolbar lists every
model with its size, licence and publisher, and one button; downloads resume
if interrupted, are checked against a pinned SHA-256, and land in
`~/.wobble/models` (override with `$WOBBLE_MODELS_DIR`). Nothing is ever
fetched on its own — a missing model leaves its node inert with a labelled
button, never a surprise 200 MB on someone's connection.

The same registry drives a CLI, so CI or a setup script can warm the cache
without opening a window:

```sh
target/release/models list          # what exists, and what is already cached
target/release/models get tapir-causal
target/release/models hash <id|path>   # what goes in CATALOG's sha256
target/release/models inspect bootstapir-causal   # named tensors + shapes
target/release/models rm <id>
```

A downloaded checkpoint is read **natively**: `models inspect <id>` parses a
JAX/numpy `.npy` into named tensors in
[engine/src/npy.rs](engine/src/npy.rs), so nobody needs python, numpy or
torch to use a model. It is not a general unpickler and must not become one:
a small stack machine over the opcode subset numpy emits, with exactly two
callables allowed and every other `GLOBAL` refused by name. PyTorch
checkpoints (a `torch.save` zip — RIFE's `flownet.pkl`) are read the same
way by [engine/src/torchpt.rs](engine/src/torchpt.rs): stored zip entries
only, and a pickle machine that allows `OrderedDict`, the tensor rebuild
and the float storage types, nothing else.

Adding a model is one row in `CATALOG` in
[engine/src/models.rs](engine/src/models.rs) — url, size, hash, licence,
source — and nothing else in the app knows a URL.

## Render

**Render** (stored as `Write` in setups) renders the comp above it to an EXR sequence. The image passes
through untouched, so a Render sits inline in a chain rather than terminating
it, and rendering is a BUTTON — nothing lands on disk while you scrub, and
the frames are never written behind your back (rule 7).

The path takes a `####` run for the frame number at whatever width you write
(`out/comp.####.exr`); a bare `.exr` gets the number inserted before the
extension, and anything else is treated as a directory. The panel prints the
resolved first filename, because a pattern is easy to get subtly wrong.
`start` is the number the first rendered frame gets (1001 by default), so a
range renders to a sequence numbered from wherever you want it to begin.

**frames** picks which comp frames a render covers: **from input** (the
default — the range the noodle into the node carries, so the clip decides how
long the render is), **range** (a start and end set on the node), or
**global** (the script's range). The line under it says what that resolved
to, e.g. `1001 to 1100 (from input)`. A setup saved before the modes existed
keeps rendering what it did: a window if it had one, else its input.

It runs on the same per-frame machinery as the GraphBake node — the same
time-boxed render, readback and abort-if-upstream-changed check — with the
destination swapped for yours and nothing adopted at the end, since the files
are the point. Output is RGBA half.

Fed the **data** pipe (a Depth node, a ToData), a Render writes the data as
data: one float32 channel named `Z` per frame, with no colour transform and
no colour tag — the numbers the comp holds, exactly. The panel says so in
place of the output space.

## Channels

**Reorder** (stored as `Shuffle` in setups) rewires them: each of R, G, B, A picks any input channel, or a
constant 0 or 1. Reordering to BGR, copying alpha into the colour to look at
the matte, throwing a channel away — all the same node, and all one 4×4
one-hot matrix in the uniform block, so re-pointing a channel is a uniform
write and never a recompile. The panel reads down the outputs (`r ← b`), with
buttons for the wirings you actually reach for (reset, show the matte, force
opaque).

**A Load picks its EXR layer.** Renderers write passes as prefixed channels
(`diffuse.R`, `N.X`, `depth.Z`); the Load groups them by prefix and offers
each group by name, with the un-prefixed channels as the base `rgba`. The
layer list is probed once from the first frame at load — a header read on the
decode path would cost throughput for nothing, since a sequence is one kind of
file throughout. Mapping is by convention: `R/G/B` straight through, `X/Y/Z`
into RGB for vector passes, and a lone channel (a depth, a mask) spread over
RGB so it reads as grey rather than as red-only. A layer without its own alpha
comes in OPAQUE, so a pass is never invisible.

The layer is part of both cache keys. The NVMe tier hashes it into the frame
file name, so `diffuse` and `rgba` from the same EXR can never be served for
one another, and RAM residency is per reader, which is already per layer.
Multi-part EXRs are listed but only part 0 decodes today: choosing another
part is refused with an error naming the limitation rather than quietly
handing back a black frame (rule 8).

Selecting a layer costs nothing extra to render — it changes which channels
the reader uploads, not the shader. Pair it with a Reorder when a pass needs
its channels moved around after arrival.

## Regions

A region is a named, tinted wash under a stretch of the graph — one per
render layer is the intended grain (FG, ENV, CHAR). Unlike a backdrop it
owns things: its members, and a registry of **common sources** — the crypto
Load, the util Load — that the **`cmd+\`` pie** offers while the view centre
stands inside the wash (the panel corner names the active region).
**⌥`** is the same instrument for the comp's GLOBAL roster: the sources
every region borrows, offered anywhere in the graph, region or no region.
Either way, picking a slice spawns a **source chip**: a Dot hidden-wired to
that source, so the plumbing exists without a noodle crossing the comp.
Hold the key, flick to a disc, let go. With exactly one node selected the
pie flips to ADD mode — up registers that node in the roster under its own
name, down names it first — and a node's right-click menu carries the same
two rosters. Make a
region with `M` over a selection (Blender's move-to-collection habit — the
popup also adds to an existing one), from the right-click menu, or rename
and colour it from the right-click menu on its name chip; register a
source from a member node's own right-click menu.

Hidden wires are ordinary edges with a presentation flag, so the engine,
the cost/licence taints, cycle checks and every cache treat them exactly
like drawn ones. They are never a secret: a hidden-fed node wears an amber
bead, hovering it draws the ghost noodle back to the origin, and
**node-patch mode** (shift+`) locks the ordinary wiring grey and shows the
whole hidden layer in amber for editing — wires dragged there are created
hidden, arbitrary node to node. Pulling the wire out of a badged input in
normal mode surfaces it.

## Breakout / Recombine (the harp)

The AOV comb as one instrument instead of a select-and-Add ladder (that
ladder remains: the Load layer picker's "rebuild from AOVs…" builds either
shape). A **Breakout** is one wide node on the fat cable with an output
nodule per ticked layer — nodules wear their layer names, an evenly
pitched comb with a spacing slider in the panel. Its paired **Recombine**
sums the strings back together in ONE fused expression (no Blend ladder,
nothing slow) and rides x-locked below its Breakout: drag it vertically to
set the working gap, and the noodles between them hang straight like harp
strings. Splice a Grade into a string and that AOV is graded; cut a
string and it stays cut.

The bars wear BOOKENDS rather than strings at their ends: the Breakout
takes the fat cable on its left porch, and the receiver takes the original
rgba on its left and leaves with the rebuilt beauty on its right. A `+` on
the Breakout bar adds a layer straight from the upstream list, and the
S / R drag treats the pair as one instrument: no rotation, and
scaling scales the LAYOUT (string pitch and bar gap) rather than the bars
themselves. The AOV dialog offers guesses too: all / none /
invert, plus `lights` (layers sharing a name prefix) and `aovs` (the
loners, rgba excluded).

**The rebuild is rgba.** Summing AOVs rebuilds COLOUR — the passes carry
the renderer's per-pass coverage in their own alphas, and adding those
together means nothing — so the receiver takes rgb from the sum and
**alpha from the original rgba** by default, off the left bookend the harp
wires for it from the same cable the strings come from. The panel's
`alpha` row switches that to the summed alpha or to opaque, and wiring the
bookend by hand overrides the source (that wire always wins).

**Remainder.** Tick `remainder` on the Breakout and the bar grows one
more string: the base rgba MINUS every chosen layer — whatever the
renderer failed to distribute into its AOVs, surfaced as a pass of its
own. It is signed (mis-distribution can overshoot), and the receiver's
sum takes it back in exactly, so beauty − AOVs + remainder is the beauty
to the bit.

Behind the chrome each nodule is an ordinary hidden ChannelSelect the node
manages (the remainder adds a base select and one signed Recombine), so
per-layer decode, both cache tiers, instancing and laziness are exactly
the select ladder's — only ticked layers ever decode, and the engine
never learns the concept (`app/src/breakout.rs` lowers it away, the
lumabands trick one more time).

## Long names swing up

A node's name sits level beside it until it would run into another node or
another name — then it swings UP around the node's own centre, 15° at a
time, until it clears (up because wires leave downward, so above is where
there is room). A row of Loads with long render names fans up in step
instead of printing through each other.

## Specks twinkle

A node scaled down — or zoomed away from — until its shape is a couple of
pixels across stops being drawn as a shape and becomes a **twinkle**: a hot
core that never blinks out, four short spikes and a small halo carrying a
hint of its type colour. Every node twinkles on its own beat, so a corner of
the graph you have shrunk to nothing reads as a field of stars rather than
vanishing into the grid.

## Reading cost off the graph

Every node either **fuses into the main pass** or **takes a pass of its own**.
That distinction is the single biggest thing about a comp's speed, and it used
to be invisible until you opened a panel. Now it is the colour of the wires:

- **Cyan** — everything upstream fuses. The whole chain is one dispatch.
- **Red** — something upstream takes its own pass, and everything below it is
  part of that chain. The node responsible wears a red ring; the noodles
  carry it downstream.
- **Red that GLOWS** — something upstream is *really* slow: a pass chain
  that dominates the frame rather than a pass. The node's ring goes red hot,
  a halo breathes around it, and every noodle below it is lit the whole way
  down, so the branch that dominates the frame is what your eye lands on. Today that is
  a Defocus in accurate mode (the spiral gather), a Loop iterating eight
  times or more, a user Shader of three passes or more, and **a Cryptomatte
  until its matte is baked** — that one costs six extra layers pulled out
  of the EXR on every frame, which is not a GPU pass at all but is the most
  expensive thing in a comp that has one. Bake the matte (the node's own
  bake row) and it goes cyan, because the frames then come off the NVMe
  tier. A Blur, a Glow or a fast Defocus are slow but do not glow.
- **A loaded Bake turns it cyan again.** That is the point of the colour, not
  a side effect of it: put the expensive branch behind a `GraphBake`, press
  bake, and everything below really is cheap, because the pixels now come off
  disk instead of re-evaluating the branch. An *unbaked* Bake stays red — it
  is still evaluating live, and saying otherwise would make the button look
  like it had already been pressed. The glow goes out at the same bake.

Bypassing a slow node clears the taint too, because a bypassed node does no
work. Two Loads into a Blend, one through a Blur: only that side runs red, and
the Blend inherits it.

Slow is not a warning. Wobble ships a diffraction convolve, a spiral-gather
defocus and four-pass user shaders precisely because they are worth their
cost — the rule is that you can always SEE what you are spending
(`CLAUDE.md`, manifesto rule 8).

## Non-commercial models

Some of the best models in this field ship under research-only licences.
Wobble's policy is to **support them, tag them and warn about them — never
refuse them**:

- The licence is shown in the Models panel *before* a byte is downloaded.
- A node whose weights are research-only wears a violet pip.
- Anything downstream of one is flagged: an **NC** badge beside the setup
  name whenever the current view is made with such weights, and a warning in
  the **Render** panel, because rendering is the moment output leaves.
- Nothing is ever blocked. The render runs.

**A Bake does not clear that flag**, and this is the one asymmetry worth
remembering: baking changes what pixels COST, not what they ARE. Speed taint
and licence taint propagate the same way and clear differently.

Today every registered model is Apache-2.0, so nothing is tagged — the
machinery is there so that adding a research-only model is a data change
(`commercial: false` in the registry) rather than a thing somebody has to
remember to wire up.

## GenAI — generative models as a node

**One node per model.** The AI menu lists every remote model as its own
node — *FLUX.1 [schnell]*, *FLUX.1 Kontext [pro]*, *Nano Banana Pro*,
*GPT Image 2 inpaint*, … — named for the model, with the model's inputs
(image, matte, prompt port) and its own fields: a prompt box, seed, size,
extra JSON. No model menu: to try another model, add that node. A String
wired into the prompt port wins over the typed prompt. Chain them in the
comp (a FLUX text-to-image into a Kontext edit into an upscale) and
collapse the chain into a subwob (alt+G) to reuse it. Each node runs on its
own click; a node whose upstream model has no take yet refuses to run
rather than send the raw plate in its place. Billing, caching and takes
are as below.

**Geometry and relighting from one picture.** *MoGe-3 / MoGe-2 Normals*,
*Marigold Depth / Normals / Albedo*, *Depth Anything v2*, *MiDaS Depth /
Normals* take the node's image and hand back depth or normals as RAW
values: grey for depth (relative, near is bright, 8-bit), `n * 0.5 + 0.5`
in rgb for normals. Those takes skip the sRGB input transform, so the
numbers are what the model wrote, ready for a Refract or a relight.
*Marigold Albedo* is a picture (the lighting taken out). *IC-Light v2*
relights from a prompt (extra `{"initial_latent": "Left"}` picks the
light's side); *Relighting* relights to a named style the prompt picks
(natural, studio, golden hour, rim light, neon…). Depth Anything v2 and
MiDaS have no price from fal before the run.

A **GenAI Pipeline** node is the free-form version: a small generative
pipeline — a prompt,
the node's own input as a reference, one or more model calls, an Output —
and plays the result back like a bake. Its steps are NODES: Enter on the
pipeline opens it like a Noise or a 2dini (U steps back out), and inside
you add, wire and delete step nodes with the ordinary graph gestures — the
add menu there lists the steps below, one node per model. Select a step
and its fields are in the panel; run… is on the pipeline node itself.

**Inputs are a bar.** The node starts with one image input and one matte
input, drawn as a bar like a Layer's: pictures stand on the left, mattes
on the right. Drop a noodle on the left half of the bar for another
picture, the right half for another matte (or use **+ image** / **+ matte**
in the panel; `×` removes one, never the last picture). Inside, each input
is its own **Input** node — one per input, made and removed with it, never
deleted or added in there. An Input can be switched to **clip**: the input
over a range from the playhead, for a video model. Prompts live inside the
pipeline: the node has no prompt input.

| step | what it is |
|---|---|
| prompt | text |
| image in / matte in | one of the node's inputs, rendered at the playhead when you run (or as a clip) |
| text to image / edit / inpaint / upscale / cutout | one call to a model from the catalog |
| image to video | a clip from a start frame and a prompt (Seedance); needs a movie build |
| ComfyUI workflow | a workflow on your own ComfyUI (free, your GPU, any checkpoint) |
| out | the node's picture |

**Prompts as nodes.** A **Prompt** (String) node (Sources) holds text and
nothing else; its rose wire is the text pipe (a Text node's words). A
GenAI node used to take one on a prompt input; a comp saved that way
loads with the String's text copied into the pipeline's prompt.

**Nothing runs, and nothing is billed, without a click, and the panel
says which click is which.** Step 1, **get price**, renders the inputs at
the playhead and asks the service what it will cost: free, nothing is sent
to a model. Step 2, **pay $X & run**, is the request and the charge. A
model node inside a pipeline has the same two steps for itself, an **info**
button (what it takes, what it gives back, its licence), and a gallery of
every result it has made: click one to make it the step's result, **pop
out** for a Result node of it inside the network (wire it like any
picture; copy it and paste it in the comp outside and it is a Load of the
same file). **gallery** opens the results in a window: one large picture
with a strip to move through (Left / Right), or a contact sheet; either
way a result can be selected or popped out from there.
A **prompt group** (add menu, inside a pipeline) runs one prompt on
several models: select the model nodes, add it, and a speech-bubble
outline is drawn round them. Whatever stands inside the bubble is in the
group: drag nodes in or out, drag the bubble's lower-right corner to
resize it, drag the group node to move the lot. Wire a prompt, up to two images and a mask into the group node
and every model in the bubble gets them in the same order (a model with
one image input takes image 1 and ignores the rest). One **get price for
all**, one **pay & run all**, and the group's panel shows the results side
by side as a contact sheet; click one to view that model.
A pipeline with nothing to bill — all cached, or ComfyUI only — runs
straight through. There is no per-frame evaluation and no auto-run on open.

**Results are cached by what made them.** Every step's result is keyed on
its model, seed, options and its inputs' keys (the reference's actual
pixels included), so re-running an unchanged pipeline, reopening a comp or
clicking twice is free and gives the same take. Change the seed ("new
seed") for a variation; that is a new request and is billed.

**Takes.** Each run that lands is a take, kept in the node's history as a
thumbnail and undoable as one step; pick one to show it, `×` to forget it
(the file stays). **settings** puts back the prompt, seed and model that
made the picked take, so a variation starts from the one you liked (it
runs nothing). **load** makes a Load node of the take beside the node, and
**feed** wires one into the node's image input, so an earlier result is
the next run's reference. Results live in `~/.wobble/genai` (`$WOBBLE_GENAI_DIR`); a comp
opened on another machine names its takes but needs those files, and the
panel says so.

**What comes back lines up.** The picture is stretched over the node's
input format (the script's, unwired), so an edit or a cutout sits on the
plate it was made from. Models work in display-referred sRGB: the reference
leaves the comp as sRGB at up to 2 MP and values above 1.0 clip; the result
comes in through the sRGB input transform, on straight colour so a cutout's
soft edge survives.

**Ready-made edits.** **Outpaint** grows the canvas around the picture
(`canvas x`, up to 3) and paints the new border; the picture keeps its size
in the middle and the node takes the new canvas as its format. **Background
Replace** keeps the subject and paints a new background from the prompt.
**Colour Transfer** gives the picture the palette and grade of a second
one, wired into the node's second picture input; it is a
repaint at up to 2 MP in 8 bits, so it is for finding a look, not for a
final grade. *Nano Banana Pro edit, two pictures* is the general form: the
prompt says what to take from the second picture. Inside a GenAI Pipeline
it is a second picture on the bar, and its **image 2** Input node.

**Image to 3D.** *SAM 3D Objects* lifts one object out of the picture, the
one the prompt names ("chair", "blue car"), as a 3D Gaussian splat. A splat
is not a picture, so the take lands as a **Splat** node beside the GenAI
node (same undo step): wire a Camera into it and it renders like any
capture. The GenAI node itself keeps passing its input through; its history
shows the take as `3D`, and **load** makes another Splat node of it. There
is no video-to-3D model: fal lists none that answers today (a clip's camera
is CameraTrack's job, and its depth the Depth node's).

**Fan-out: one prompt, several models.** Above `run…`, **also run on…**
lists the other models that do the same job as the node's own (text to
image with text to image, edit with edit). Tick some and `run…` sends the
same prompt, seed and inputs through each: one price for the lot, one
confirm, one take per model. They land side by side in the history;
**sheet** opens them large in a window, each under the model that made it,
and a click shows that one. A model already run is a cache hit and is not
billed again. Ticking a research-only model tags the node straight away.

**Batch: lists of prompts, seeds, frames.** Under the fan-out row,
**batch…** runs the node over lists: **prompts**, one per line, each in
place of the node's own; **more seeds**, that many after the authored one;
and **frames** (`1001, 1010-1050x10`), which captures the node's inputs at
each of those frames instead of the playhead. Frames are how a pipeline
runs over a list of images: a Load of ten stills is ten frames. The lists
multiply each other and the fan-out, the line beneath says how many takes
`run…` will make (64 at most in one go), and it is still one price and one
confirm. Every take records its frame; hover it in the history.

**Prompt tools.** Three steps inside a GenAI Pipeline make TEXT, and feed
another step's prompt. **Prompt Enhancer** writes a short prompt out in
full with a small language model. **Describe** looks at the picture and
writes it up as a prompt (its own prompt says what to dwell on). **Text
Template** is text with variables: `{a}` and `{b}` are whatever is wired
into its two inputs (a String node through Prompt In, a description, a
typed style), `{frame}` the frame. After a run the panel shows what each
wrote, which is the prompt the model was really sent. The templates
*Enhanced concept* and *Describe and reimagine* show them wired.

**App mode: a pipeline as a form.** Next to templates, **app…** turns a
GenAI Pipeline into a simple form. Tick the fields worth showing (a
prompt, a seed, a scale, megapixels, a clip's length), name each, give the
form a title, and press **app mode**: the panel becomes that form and the
run button, and the steps are locked (Enter no longer opens them).
**suggest** picks every prompt and the last model's seed. **edit** on the
form unlocks it again. A template saved from a pipeline in app mode opens
as that form on whoever's machine it lands on.

**Templates.** A GenAI Pipeline's panel has a **templates…** menu: whole
pipelines, ready-made (Clean plate, Restyle the plate, New background
upscaled, Outpaint the plate, Match a look, Relight, Concept to clip,
Remove from the clip, and more). Picking one replaces the steps, keeps the
takes and runs nothing. Each is also in the add menu under its own name.
**save as** keeps the pipeline as it stands as a template of your own: a
small JSON file in `~/.wobble/genai-templates`. To share one, send the
file; dropped into that folder on another machine it is on the shelf. A
template names models, never keys or URLs.

**Upscalers make the node bigger.** A node whose picture comes out of an
upscaler (Real-ESRGAN, Topaz Upscale / High Fidelity / Recovery / Redefine,
Clarity, SeedVR2, Recraft Crisp, and Topaz or SeedVR2 on a clip) takes the
RESULT's size as its format instead of stretching it back over its
input's, and its reference goes up whole to about 4 MP rather than 2.
Topaz's face enhancement is off unless the step's extra turns it on:
`{"face_enhancement": true}`. A clip is still sent at 720 px on its short
side, so a video upscale starts from there.

**Who pays** is the machine's business, never the comp's: **wobble
credits** (one account, no provider keys) or **my fal key** (straight to
fal, nothing through us). Keys live in `~/.wobble/genai.json`, readable only
by you; `FAL_KEY`, `WOBBLE_RELAY_TOKEN`, `WOBBLE_COMFY_URL` override it. A
setup names catalog models, never a URL or a key.

**Cost and licence on the graph.** A node whose pipeline bills wears a
green `$` off its lower-left corner. It is fast to render — a reader like a
Load. Each remote model carries its licence in the catalog; a pipeline that
names a research-only model, or a take made by one, is tagged
non-commercial like any other (see above), and that tag survives picking a
different model later, because the pixels on screen were made the way they
were made.

**Models.** FLUX (schnell, dev, Kontext, Fill), Real-ESRGAN and BiRefNet,
plus ByteDance's through fal: **Seedream 5 Pro** (text to image, 1–2.3 MP —
held under 1536² so the per-image price is the lower tier — and an edit
that takes the reference as `image_urls`) and **Seedance** image to video
(2.0 fast: 4–15 s at 480p/720p; 2.5: up to 30 s and 1080p). Seedream and
Seedance take no seed; the seed still keys the cache, so "new seed" is how
you ask for a variation.

**Video takes.** An image-to-video step turns the input at the playhead
into an mp4 take. It is quoted per second (or, where fal prices by tokens,
by ByteDance's `h × w × seconds × 24 / 1024`, shown as an estimate), asks
for no audio, and plays back as a movie PLACED at the tick it was made at:
its first frame lands where the reference was captured, and the node holds
the first and last frames outside the clip. Only a build that decodes
movies (`--features app/movie`, a system ffmpeg) offers video steps; open a
comp with a video take in one that does not and the panel says so in red
while the node passes its input through.

## Shader — write WGSL in the app

**Shader** is a node whose behaviour is text you type into Wobble. It is the
Matchbox-shaped node: unlike a Tier 0 plugin (one expression fused into the
main pass), a Shader node gets **its own passes and its own textures**. Every
image input is resolved to a real texture first, so the shader can sample any
input at any coordinate, in a loop, as many times as it likes — and it can
run up to four passes that read each other's results.

That also makes it a **slow path** by construction: resolving each input
costs a full-canvas pass, the node never fuses with its neighbours, and every
pass keeps a full-canvas result alive for the whole chain (four passes at 4K
is ~283 MB). The node says so in its properties, per the rule that a slow
path is never quiet.

Metadata rides in `//!` pragmas, so a shader is one file with no sidecar XML:

```wgsl
//! name: Radial Blur
//! input: front
//! passes: 1
//! param: float amount = 0.15 [0, 1] "Amount"
//! param: vec2 centre = (0.5, 0.5)
//! param: int steps = 24 [2, 64]

fn pass1(uv: vec2f, frag: vec2f) -> vec4f {
    let dir = uv - P.centre;
    var sum = vec4f(0.0);
    for (var i = 0; i < P.steps; i = i + 1) {
        let t = f32(i) / f32(P.steps);
        sum = sum + tex(front, uv - dir * t * P.amount);
    }
    return sum / f32(P.steps);
}
```

| | |
|---|---|
| `input: <name>` | an image port, up to 4; the first is port 0 |
| `passes: <1..4>` | pass `i` is `fn pass{i}(uv, frag) -> vec4f` |
| `param: <ty> <name> = <v> [lo, hi] "Label"` | `float`, `int`, `bool`, `vec2`, `color` |
| `group:` / `name:` | properties grouping, node label |
| `P.<name>` | your params, by name, at their WGSL type |
| `P.res` `P.time` `P.frame` `P.pass_index` `P.aspect` | built-ins |
| `result1..resultN` | an earlier pass's output |
| `tex` `tex_b` `tex_px` `texel` `tex_size` `luma` | sampling helpers |

Params are **uniforms**: moving a slider never recompiles, and every one is a
full animation channel — keyframes, curve editor, patch cables, HUD drag-out,
like any built-in param. Editing the TEXT is what recompiles, so the editor
is a buffer with an explicit **apply** (⌘↵).

The source is parsed and naga-validated before it can become topology. A
shader that does not compile leaves the node a **passthrough** with its
errors listed under the editor — the comp keeps playing while you type, which
is the only way a live text editor over a live comp is usable. Ports come
from the pragmas, not the compile, so a typo mid-edit never drops your wires.

### GLSL — paste a Matchbox shader

**GLSL** (Utility) is the same node reading Flame Matchbox shaders as they
are. Paste the `.glsl` into the editor and it works: every `uniform sampler2D`
becomes an input port and every other `uniform` a param. Paste the `.xml`
anywhere in the same text to get the preset's labels, ranges, defaults, popups,
groups and input order. **open…** on any one file of a shader brings its other
passes (`name.1.glsl`, `name.2.glsl`…) and its `.xml` along; by hand, a line
reading `//! pass` separates passes. Pasting GLSL into a plain Shader node works
too, because the dialect is detected from the text.

| Matchbox | here |
|---|---|
| `adsk_result_w/h`, `adsk_result_frameratio`, `adsk_<input>_w/h` | supplied |
| `adsk_time` | the comp frame |
| `adsk_results_passN` | that pass's result; a pass reading itself sees its previous execution |
| `NbExecutions`, `adsk_result_execution` | the pass really repeats |
| `NoInput="White"` | an unwired input reads white |
| `InputType="Matte"` | reads the wire's alpha into every channel |
| `GL_TEXTURE_WRAP_*`, `GL_NEAREST` | honoured on direct reads of an input |
| `dFdx` `dFdy` `fwidth` | real differences, by running the shader three times per pixel |
| `texture2DLod` | approximated by a 16-tap box; inputs have no mips |
| `MatteProvider` | alpha comes from the shader; otherwise from the first input |
| `CommercialUsePermitted="False"` | tags the node and warns at a Write, like a research-only model |

Not available: frame-to-frame feedback (`adsk_accum_texture` reads black) and
Flame-only functions (`adsk_getBlendedValue`, `adskEvalDynCurves`), which are
reported as errors. `adsk_scene2log`, `adsk_log2scene`, `adsk_shadows` and
`adsk_highlights` are close approximations, and the panel says so when a shader
uses one. Passes run on up to 64 dispatches, and a result nothing will read
again gives its texture to the next pass. **The comp is scene-linear (ACEScg)**,
so a shader tuned for 0..1 video will not land where its author expected until
the plate is converted first.

The shader lives in the setup, not in a sidecar file, so a comp carries its
own effects; **open…** / **save as…** in the editor are for sharing.

### Wrapping a shader in a subwob

**make controls** in the Shader panel gives every declared param its own
control node, already cabled. Inside a subwob those controls ARE its
knobs — arrange them in the panel composer, step out, and the shader is a
node with a real interface that nobody has to hand-wire. In a plain comp the
same button just gives you sliders in the graph driving the shader.

The mapping is plain: `float`/`int` and `bool` become sliders (bool as 0..1 —
there is no toggle control yet), `color` becomes one RGB slider driving three
lanes, and `vec2` becomes two sliders. An RGB Slider control has a
**multiplier** setting beside its `lo`/`hi`: set it for a triple that scales
(a gain, a gamma) and the row takes the slider's scale mode — colour axes in
log space, lanes that never cross zero — as the Grade's own gain row does. Controls start at the value the shader
is currently AT, not its authored default, so pressing the button never moves
the picture, and a param that already has a cable is left alone — press it
again after adding a param and it tops up rather than duplicating.

## Format, geometry, colour and time nodes

**Grid** generates a ruled grid at its own format — a source with no inputs,
so like a Load it can drive the comp's resolution. **Reformat** redefines the
format, fitting the input into it by distort/fit/fill/none plus an extra
scale; the fit is a coordinate map on the source side, so it stays one
sample rather than a resampling pass.

**Saturation** is the whole-image control, and it is more interesting than
the one line it is built on. Every tool's saturation is a lerp between the
pixel and its own luminance — what varies is everything around that line.
The **domain** is the biggest one, and the one nothing else exposes:
Resolve does not saturate scene-linear values, it saturates whatever its
timeline space holds, which in a managed timeline is a log encoding. That
is most of why its slider feels the way it does, so it is a control here.
Write each channel as its ratio to the neutral and the two are `r' = 1 +
(r − 1)·s` against `r' = rˢ` — affine against a power law. Both are
exposure-invariant, so neither favours highlights over midtones (a common
claim, and wrong); the difference is that the affine one crosses zero and
drives channels negative, out of gamut, while the power law cannot. Hence
linear, **log** (ACEScct, the default), or Oklab chroma, which holds
perceived lightness and hue but will not match anyone else's numbers. The **luma**
dropdown is the usual luminance math (Rec.709 / average / maximum, where
maximum makes it exactly an HSV saturation) plus the working space's own
weights, which are the only correct ones and the default. **Colour boost**
is Resolve's Color Boost: extra saturation weighted toward the pixels that
have least, so a sky comes up without the reds clipping. **Shad · mid ·
high** is a three-zone stand-in for the Sat-vs-Lum curve, over a pivot and
a span in stops. **Keep luma** puts the original luminance back — the
working-space lerp preserves it for free, log and Oklab do not — and
**gamut** pulls colours that left the gamut back to its edge rather than
letting the display clip them into a hue shift. Domain and luma pick the
generated code, so those two recompile; every other control is a uniform.

**Mult** is the one-number version of a Grade: multiply by a constant,
through a matte on input 2. Colour is multiplied and alpha is not unless you
ask — the default is for making something HOT (which deliberately breaks
premultiplication), while multiplying both is a fade. Like every masked node
here it crossfades against the untouched input, so the matte's PEAK is the
ceiling: a matte topping out at 0.6 turns a multiply of 8 into 5.2.

**Unpremult / Premult** divide colour by alpha and
multiply it back; colour operations here assume premultiplied input, so the
sandwich is how you grade the opaque part of a soft edge (a fully transparent
pixel has no colour to recover and stays black rather than dividing toward
infinity).

**Tracker** follows a small region and hands back its motion as data. The
track is an explicit bake — decoding the sequence per playhead move is
exactly the quiet slow path this project refuses — and it writes a keyframe
per frame onto the node's `TrackX`/`TrackY` channels, so a patch cable from
the tracker drives anything in the comp with the plate's motion. It bakes
from the frame you are LOOKING AT to the end of the sequence: you park the
playhead where the feature is clear, draw the box on it there, and that is
the frame the search is seeded from (frames before it hold the first key). The image
passes through untouched. It is a thresholded luma centroid inside a search
window that follows the previous frame, with no sub-pixel refinement and no
rotation: the cheap, predictable tracker, and a frame where the blob is lost
holds the last position and is counted in the panel rather than snapping
somewhere plausible.

**Slip**, **Speed** and **TimeRemap** are not shader effects at all: they
remap which frame each reader upstream of them decodes. Slip shifts by whole
frames; a positive offset delays the clip (Nuke's TimeOffset way), so its
range moves by the same amount. Setups saved before that flip stored the
opposite sign and are negated on load. Speed is a constant rate around an anchor frame — the anchor stays
put while everything else stretches away from it, and a negative rate runs
backwards. **TimeRemap is a single animatable number**: which source frame
this timeline frame shows. Key it and its curve *is* the retime, shaped in
the curve editor (`Y`) with the same interpolation modes as everything else —
a flat span freezes, a falling span reverses, and the slope is the rate. Left
unanimated it parks the shot on one frame. Setups written before the split
(TimeWarp's speed + start) migrate exactly: a pure shift becomes a Slip,
anything else a Speed about the frame that stayed put.

Speed and TimeRemap can land between frames, so both offer **nearest** or
**blend**. Nearest snaps to the closest source frame. Blend dissolves between
the two frames the sub-frame time falls between: the branch above it is
instanced twice — once floored, once ceiled — so both plates are decoded and
live at once, and the fused pass mixes them by the fraction. It costs an
extra read and an extra subtree evaluation per blended branch, which is the
honest price of having two frames on screen. Slip has no such switch: it
shifts by whole frames, so there is never a fraction to dissolve. Neither
mode estimates motion.

They emit no code beyond that mix, so snapping costs nothing to render, and
the timeline itself never moves — a slowed reader simply doesn't reach its
last frame before the loop comes round.

**Freeze** holds its input on one comp frame: every tick reads the input at
**frame** (a speed-0 time map, so it costs one decode and no pass). It
starts on the frame under the playhead (**hold this frame** moves it
there). The frame is keyable: keyed, each key holds a frame — stepped keys
jump between holds, smooth ones slide — and it is held flat past the first
and last key. Moving it is a time-map change, never a recompile.

**Frame Fix** rebuilds bad frames from their good neighbours and passes
every other frame through untouched. **frames** takes comp frames and
ranges (`1012, 1040-1042`; **+ this frame** marks the playhead's). A run of
bad frames is rebuilt from the good frame before it and the one after,
each bad frame at its fraction between them. **blend** is live: the input
sits on two time branches (the good frame before, the good frame after),
dissolved per tick — exact passthrough on good frames, the neighbours'
average on a single bad one, and a second decode of the input a frame.
**optical flow** makes the bad frames with RIFE on a click (the Rifetime
runner and the `rife-4.6` weights): **render** renders the input over its
range, makes each bad frame along the motion from its good neighbours,
links the good frames across and plays the result through the node's
reader like a bake (cyan, the input no longer evaluated). Until it is
rendered it shows the blend; editing the list or anything upstream shows
STALE. A bad frame at the edge of the input holds its one neighbour.

**Animation upstream of a time node runs on that node's clock**, sampled at
the remap's unrounded time: a curve keyed at frames 0 and 10 under a half
speed node reads its frame-2.5 value at timeline frame 5, even though the
plate below it can only show whole frames. (TimeRemap's own curve is sampled
into a lookup table at quarter frames when the graph is built, because time
has to be answerable at every tick a reader prefetches, not just the one on
screen.) `Graph::node_time_maps()` reports
the time reaching every node and the transient evaluation uses it. One
limitation: a node reached through two different time paths keeps the first
(view target first) for its animated params, because the graph holds one set
of parameters per node — readers have no such limit, they instance per path.

**A source can be live at several frames at once.** Time flows *down* the
graph the way coordinates do: every path carries the chain of time nodes
between it and the viewer, and a reader-backed node is instanced per distinct
chain. Wire one Load into a merge twice — once straight, once through a
TimeOffset — and it becomes two readers holding two frames, both decoded,
both sampled by the same fused shader. Anything non-pointwise upstream of
the split (a Blur, a Defocus) is instanced too, so it filters each frame it
is asked for rather than one they fight over. Nothing is silently resolved
and nothing warns you off; the cost is simply another reader, which the time
node's panel reports.

Two consequences worth knowing. The instancing is keyed on which time NODES a
path crosses, never on their values, so dragging an offset (even through
zero) moves frames without recompiling; adding or rewiring a time node is
what changes the shape. And time nodes move IMAGE time, not animation time —
a keyframed Grade above a TimeOffset still evaluates at the playhead.

## Planar tracker (Mocha-style)

**PlanarTrack** follows a flat surface — a poster, a screen, a wall, a
tabletop — through the shot the way Mocha does: draw a quad on it in one
frame, press track, and get the plane's motion back as a homography per
frame. Two paradigms come out of the one solve:

- **Planar** — the region's four corners carried through the sequence: a
  corner pin. The `model` says what the plane may do: `translation` (2
  DOF), `similarity` (+ rotation and uniform scale), `affine` (+ shear) or
  `perspective` (the full 8-DOF corner pin). A constrained model is not a
  weaker solve, it is a stronger prior: pick it when you know the shot.
- **Mesh** — a triangle mesh over the region, every vertex refined off the
  plane by its own local alignment and smoothed against its neighbours:
  the deformation a plane cannot express (a bending page, cloth, a face).
  PowerMesh, roughly.

It is a DIRECT dense tracker (`engine/src/planar.rs`, CPU): every pixel of
the region votes, which is what holds on soft, low-texture surfaces where
a point tracker slides. The solve is ESM (Gauss-Newton with the mean of the
template and warped-image gradients) coarse to fine over a pyramid, on
zero-mean unit-variance pixels (a lighting change is not motion), with
the residual CENTRED on its local mean (shading that moves with the view —
a sheen sliding over cloth, a fold's shadow — is a broad smooth residual
that would otherwise read as motion, and as an occluder) and Tukey
reweighting on that residual BLURRED over the region (an occluder is a
blob of bad pixels, a misaligned edge is a line — per-pixel tests cannot
tell them apart and end up rejecting the edges that carry position), plus
a variance gate that drops windows where the picture has no structure at
all while the template has some (a flat grey ball, a hand). A
constant-velocity prediction seeds each frame, an exhaustive translation
search at the coarsest level catches a jump the pyramid cannot reach, and
below the `re-anchor` score the template is re-cut from the current frame
(appearance drifted: perspective, blur, light) — 0 never re-anchors (no
drift, fails on appearance change), 1 tracks frame to frame (drifts).
Trust is earned per frame: a confident fit is taken as it is, an
unconfident one only where the velocity expected it, and otherwise the
frame is LOST — the plane coasts on its velocity for up to twelve frames
while the coarse search looks for it (then holds still), and a lost frame
never becomes the template. That is what keeps a hand crossing a brick
wall from sliding the track one row over. A re-anchor cuts its new
template through the solve's own weights, so whatever is in front of the
plane at that moment is left out rather than baked in.

Using it: add a **PlanarTrack** on a Load with footage, select it, and drag the
quad onto the surface in the viewer (the corners are free — it is a quad,
not a box — and the centre moves the whole thing). Park the playhead on
the frame you drew it on, then `track ⏵` to the end of the clip or
`⏴ track` back to its start. The viewer follows the bake, frame by frame,
overlay and all: the tracked quad, red on a frame where the plane was
lost (the plane coasts), and the mesh as thin lines when it is on. The
panel reports the frame's match score. The results are DATA on the node —
a corner pin per frame and, with the mesh on, a vertex set per frame —
saved with the setup; `clear` drops them and makes the region editable
again.

**Driving other nodes.** The bake also keys twelve channels on the node,
so the track is a patch source like any Tracker's: `planar dx / dy / rot /
scale` are the region's motion since the reference frame as a similarity
(pixels, pixels, degrees, a factor — cable them into a Transform's
translate / rotate / scale, with the Transform's pivot on the region's
reference centre), and `planar pin 0-3` are the frame's corners under the
frame's homography, the full perspective. The panel's **drive a
Transform** cables the pin channels into the selected Transform (or a new
one beside the tracker): wire your element into that Transform and it
rides the plane, perspective and all, while the Transform's own translate,
rotate and scale still place the element on it (its pin is applied first).
**drive a CornerPin** does the same into a CornerPin. Both assume the driven
node works in the plate's format — the pin is normalised there.
**stabilise** is the inverse: it cables the same pin into a CornerPin's
FROM corners and leaves its TO corners on the frame, so the warp undoes the
tracked motion — wire the PLATE into it and the region holds still where
it sat on the reference frame (that frame passes through untouched). The
multi-point Tracker's panel has the same three buttons on its solve.

Measured on a synthetic 960x540 shot (48 frames, 19° of rotation, 1.3x
scale, growing perspective, noise), against the truth it was rendered
with: worst corner error 0.21 px, mean 0.12 px; the same shot with a disc
of foreign texture drifting across the region, 0.23 px worst; the mesh on
that plane within 0.6 px at every vertex; on a surface bulging 8 px off
the plane, the mesh follows to 0.7 px mean (a few vertices reach ~6 px —
the known weak spot). On a Blender-rendered brick poster (Eevee, 960x540,
48 frames: a hand-held-style move with perspective and roll, a light
sweeping across, a ball crossing half the region) the worst corner error
is 0.74 px, mean 0.23 px, with the two frames the ball covers most of the
region coasted and the track re-acquired from there. On a real 2K CG shot
(a beaded necklace on folded cloth, the camera swinging round and up, the
sheen moving with it — `data/hcr080`, 100 frames) the middle tracks end
to end with no lost frame, three re-anchors, and scores between 0.8 and
0.98. The solve is parallel (rayon): 25–40 ms a frame at 960x540 on an
M4 Pro, about 120 ms a frame on that 2K shot, a lost frame under 100 ms,
and a long track costs the UI nothing — the results sit behind a shared
pointer and stay out of the per-frame cache keys. In the app the job reads
frames the viewer has already shown from the NVMe tier rather than
decoding them again. `--test-planar` runs the planar and mesh passes
through the app and drags a corner handle.

The harness runs the same code without the app:

```bash
target/release/planar synth data/planar_synth --frames 48 [--bulge 6] [--occluder]
target/release/planar track data/planar_synth --truth --mesh 12 --overlay /tmp/ov
target/release/planar track data/hcr080 --box 0.35 0.35 0.65 0.65   # any clip, a region in 0..1
WOBBLE_PLANAR_TRACE=1 target/release/planar track ...                # where each frame's time went
```

`synth` writes 16-bit PNGs plus `truth.txt`; `track --truth` scores every
frame against it (corners and mesh vertices) and `--overlay` draws the
result on each frame.

## View transform

The fused pass leaves the output texture scene-linear; `engine/src/blit.rs`
turns that into pixels, applying viewer exposure, a display rendering
transform and the inverse display EOTF in that order. Keeping it at the blit
rather than in the fused tail is what lets the NVMe render cache, any bake and
any future export stay scene-referred, and makes auditioning a transform cost
one pipeline rebuild instead of a re-render. Transform parameters are uniform
writes: only the transform itself, the display gamut or the EOTF regenerate
the shader.

Transforms (menubar "view…", or `--view` for a reproducible headless run):

- **None** — clamp to 0..1. Useful for seeing what the comp actually holds.
- **Filmic** — the Narkowicz ACES fit that used to be hardcoded in the shader
  tail. The default, and the fallback when OpenDRT is not compiled in.
- **OpenDRT v1.1.0** — Jed Smith's display rendering transform, ported to
  WGSL in `engine/src/viewtf/opendrt.rs`. The full SDR path: rendering-space
  desaturation, tonescale norm and RGB ratios, brilliance, low/high contrast,
  hyperbolic compression, hue contrast, RGB/CMY hue shift, purity limits,
  mid-range purity, display gamut and creative white, post brilliance and
  purity softclip, with all eight look presets. Not ported: PQ/HLG, the
  DCI-P3 / X'Y'Z' containers, the log input transfer functions (Wobble reads
  scene-linear EXR), and the tonescale-overlay debug draw.

  Upstream solves the tonescale constraint constants per pixel because DCTL
  cannot precompute; here they are solved once on the CPU and uploaded, so the
  shader only does per-pixel work.

  > **Licence:** OpenDRT is **GPLv3**, the same licence as Wobble, so the
  > port is GPLv3 too. It lives behind the non-default `opendrt` cargo
  > feature, so a stock build contains none of it.

- **ACES 1.3 Output Transform** — the RRT + SDR ODT, ported from the CTL
  (`engine/src/viewtf/aces.rs`). Look: **Reference Gamut Compression**, the
  Academy's LMT (Blender's "ACES 1.3 - Reference Gamut Compression").
- **ACES 1.2 Output Transform** — the same RRT + SDR ODT with no look, for
  matching a pipeline on an ACES 1.2 OCIO config. The viewer menu's
  **sRGB (ACES)** and **Rec.709 (ACES)** are that config's `sRGB` and
  `Rec.709` views (D65, dim surround; sRGB curve and BT.1886 2.4). A Read's
  input and a Write's output offer **Output - Rec.709 (ACES 1.2)** and
  **Output - sRGB (ACES 1.2)**, its `Output - *` colour spaces: as an input
  the picture runs the inverse ODT and inverse RRT on the GPU, in the fused
  pass, back to scene-linear. `engine/tests/aces12.rs` holds both directions
  to OpenColorIO's built-in ACES transforms (`tools/aces12_reference.py`).
  The config itself bakes these into 3D LUTs, so it differs a little from
  the exact maths, most where a channel clips. Like the CTL, the inverse only
  reaches what the display can show, and saturated colours come back up to
  0.007 of a code value off (OCIO's own inverse does the same).
- **AgX** — Blender's, read from the Blender installed on the machine
  (`engine/src/viewtf/agx.rs`): its cubes, its chain, so the picture matches
  Blender's by construction and nothing of Blender's ships with wobble.
  Looks: all nine of Blender's AgX looks (Punchy, Greyscale, Very High to
  Very Low Contrast), the config's numbers through OCIO's grading maths and
  Blender's luminance-compensation cube. `engine/tests/agx_blender.rs` holds
  every look to Blender's own output.

Looks are picked under the transform in "view…" or in the colour menu; a
look is a uniform, never a recompile, and a look the shown transform does
not offer is off.

Overlays that read the comp texture (the adaptive ink, the HUD luma probes)
apply a fixed filmic *estimate* — `viewtf::display_luma` and its WGSL twin —
rather than the live transform, so a light/dark pick stays stable while you
audition transforms and costs no uniforms.

## Frame pipeline

```
rayon decode pool (issued in display order, slot-gated)
  -> mapped staging ring (decode writes f16 straight into mapped wgpu buffers)
  -> copy_buffer_to_texture into per-reader rgba16float textures
  -> single fused compute pass (write-only rgba16float storage texture)
  -> blit/letterbox to swapchain: exposure, view transform, inverse EOTF
     (+ egui on top)
```

Everything up to and including the fused pass is **scene-linear**. Only the
blit is display-referred.

Timing: GPU timestamp queries around upload / fused / blit. On Metal,
`TIMESTAMP_QUERY_INSIDE_ENCODERS` resolves garbage (Apple GPUs sample at
stage boundaries only) so upload is measured via a sentinel no-op pass before
the copies; the feature path is used on other backends.

## Findings on Apple M4 Pro (macOS is the secondary target)

- Fused pass ~0.8–0.9 ms at 4K DCI half-float; blit ~0.2 ms; upload of two
  4K half plates ~0.9 ms (UMA bandwidth-bound, matches theory).
- Steady state is decode-bound, as the brief predicted.
- Codegen + naga validation of the fused shader: < 0.3 ms. Metal PSO create:
  ~0.2–2 ms warm (on-disk kernel cache), **~100–150 ms for novel WGSL** —
  hence the async compile+swap in the app; expect single-digit ms on
  Vulkan/DX12 (the primary target) either way.
- wgpu 29 upload note: mapped-write memory is `WriteOnly` (may be
  write-combined); the decode path writes each pixel exactly once and never
  reads, which is exactly the contract, via one documented unsafe cast.

See `bench_results_4k.txt` / the stats HUD for current numbers.

## Scrub — the timeline editor

Scrub is its own program: the `scrub` binary built from `scrub/app`
(`cargo build --release -p scrub_app`; see `scrub/README.md`). The menubar's
**scrub ↗** button starts it as a separate process, looking for `scrub`
beside the wobble executable first and then on `PATH`; the comp in wobble
stays as it is. Wobble's old in-process edit mode (`--edit`, `.wedit`
projects, `app/src/edit/`) is gone.

## Dual view (two viewers)

`dual` in the menubar splits the viewer in two. The number keys aim them:

| keys | viewer |
|---|---|
| 1 / 2 | the first pane's two slots (as before) |
| 3 / 4 | the second pane's two slots |

Within a pane the pair works exactly as 1/2 always have: the key assigns the
node under the cursor (or the single selected node) to that slot and shows
it; pressing it with nothing targeted just flips to that slot.

**Only the pane you last aimed at renders.** The other holds the last frame
it had, copied once at the moment the live pane changes. Two live comps would
double the per-frame GPU cost, which at 4K is exactly what you cannot afford;
holding is what a second viewer is usually for anyway — a reference to compare
against. The held pane goes through the same view transform and exposure as
the live one, so the comparison is honest.

Slots 3 and 4 are saved with the comp (`view_c` / `view_d`), so a dual-view
arrangement survives a reload, and assigning them in single-view mode is
fine — turning `dual` on then shows what you already picked.
