# Wobble clips (experimental)

Apply a Wobble comp to any clip on the timeline: the clip's source goes into
Wobble as a locked Load, the comp's locked output comes back as what the clip
shows, and Scrub keeps the Wobble setup per clip with versions.

Experimental: behind the `wobble` cargo feature (off in the default build)
AND a runtime switch (sequence > experimental > Wobble clips). Without both,
nothing here is compiled or offered, and a project with Wobble clips still
opens (the clip plays its last render, or its source when there is none).

## What Wobble already has (so the integration needs little of it)

- `wobble --worker PORT setup.wbl`: a hidden render helper (no window in
  front, no session) serving the scripting API on `PORT` — the network
  farm's worker (`apiserve::farm`).
- The scripting API (`api` crate; JSON-RPC on loopback): `node_create`
  (a Load with a `path`), `node_connect`, `node_patch` (a Write's path,
  range, start), `node_run` / `jobs_get` (render and wait), `setup_save`,
  `graph_get`, `viewer_capture`.
- `wobble --api PORT setup.wbl`: the interactive app on a setup, drivable by
  the same API.
- Same wgpu 30 / egui 0.36 as Scrub.

## Integration route

Three options were weighed:

1. **Link Wobble's engine crate in process.** Fastest (GPU-resident, no
   files), but Wobble's engine is large (models, process-backed nodes, its
   own player and caches), pins imfram by rev, and every Wobble change would
   rebuild Scrub. It also makes Wobble's slow paths Scrub's frame-time.
2. **Drive the Wobble app over its API.** Nothing to link; Wobble stays a
   separate program; renders land as files Scrub already plays.
3. **Both** (edit in the app, render in process). The best end state, but
   needs Wobble's document model split out first (its own "direction already
   decided": a `doc` crate the server links without the UI).

**Decision: (2) now, (3) later.** Editing happens in the real Wobble app
(`--api`), rendering in a hidden Wobble worker (`--worker`) that Scrub starts,
drives and stops. Results come back as an EXR sequence per version, which
Scrub decodes, caches and plays like any media (VideoToolbox does not apply;
EXR decode is multi-threaded and the NVMe-backed frame cache holds it).
In-process rendering is a follow-up once Wobble has a headless render entry.

## How the source enters

- **Nothing is exported.** The locked Load (SCRUB SOURCE) reads the
  ORIGINAL file: a movie through Wobble's FFmpeg reader — so Wobble must be
  built with `cargo build --release -p app --features app/movie` (the
  worker and the editing app are the same binary) — and an image sequence
  by its pattern (`dir/name.####.exr`).
- It is placed with `start: 0`, so comp frame k is source frame k: Scrub's
  own numbering (0-based for a movie, the k-th file of a sequence). The
  render's range is the frames the clip uses plus **handles** (1 s each
  side, clamped to the media); a trim or slip that needs more makes a new
  version of the same setup rendered wider (no export to redo).
- **Colour**: the Load's input is pinned as Scrub shows the media — Rec.709
  primaries with the sRGB curve for movies and display-referred images,
  linear for EXR — and the output is linear Rec.709 EXR, which Scrub shows
  through the same sRGB curve. So an empty comp's render IS the source
  picture (`--test-wobble` checks it), whatever the comp's working space.
- **Before or after?** Wobble sees the *source*, untouched: no speed, no
  reframe. Scrub applies speed, reframe, dissolves and trims to Wobble's
  OUTPUT, exactly as it does to footage, so none of them needs a re-render.
- **No movie support** (a stock Wobble): the Load does not read; the
  version is refused and the HUD gives the build command.

## How the output comes back

- The setup has a **locked output**: a Write/Render node named `SCRUB OUT`
  whose path, range and format Scrub sets (EXR, half float, linear Rec.709)
  into `v###/render/out.####.exr`. Its input is whatever the user wires.
- **Render cache per version**: each version renders once into its own
  folder; switching versions is instant (the frames are on disk and enter
  Scrub's RAM/NVMe frame cache by path like any media). A version is
  rendered when it is made (saved from Wobble) and on demand otherwise.
- **Invalidation**: a render is valid for (setup bytes hash, source
  range). Either changing makes a new render; nothing is overwritten.
- **Proxies**: later — the worker can render at half size into
  `render-half/` for scrubbing; the full render stays the delivery source.
- **GPU-resident**: not in route (2) — frames come through the decoder and
  the frame cache. In route (3) the comp's output texture would feed the
  record viewer directly.
- While a version renders, the clip plays the previous version (or the
  source, marked) and the timeline shows the clip striped with progress.

## Version support

Each Wobble clip keeps a **version history** of its setup:

```
<project>.wobble/<clip-uid>/
  v001/setup.wbl     the setup as saved from Wobble
  v001/render/       its output frames
  v002/...
  clip.json          versions, notes, which is active, the source range
```

- The clip stores `wobble: { uid, active: 2, versions: [1, 2] }` in the
  Scrub project (format bump); the folders live **beside the project** (not
  inside the .scrub file: renders are large). An unsaved project keeps them
  under the autosave folder until it is saved, then they move.
- The record viewer/timeline show the active version (`v002`) on the clip;
  the clip's right-click menu lists versions to pick, re-render, open, or
  delete (with confirmation).
- A duplicated clip (alt-drag) shares the history; "make independent" forks
  it.

## Editing

- **cmd+shift+W** on a selected clip (or right-click > open in Wobble):
  first time: make v001 = Load(the original file) -> SCRUB OUT
  through Wobble's API, save it. Then launch the Wobble app on a **working
  copy** of the active version's setup with `--api`, and watch it.
- Each **save in Wobble** (cmd+S) becomes a **new version** (`v003`): Scrub
  copies the saved setup into a new folder, renders it in the worker, and
  makes it active when the render lands (the HUD says so). Closing Wobble
  ends the session; nothing is lost if Scrub is closed first (the working
  copy is picked up next time).
- **Locked nodes**: the Load and SCRUB OUT must not be deleted or retargeted.
  Wobble has no node lock today; the minimal change is a `locked` flag on a
  node (refuse delete / param edits / path changes in the panel and the API,
  draw a padlock) on a local Wobble branch. Until then Scrub **re-asserts**
  them on every save (finds them by name, restores the Load's path/range and
  the output's path; refuses a version without them, saying why).

## Performance and visible slow paths (Wobble manifesto rule 8)

- A Wobble clip costs a decode per frame like any EXR footage; nothing of
  Wobble runs during Scrub playback. The cost moved to the render, and it is
  shown: the timeline draws a Wobble clip with a W badge, **red** while its
  active version is not rendered (playing the source or an older version)
  and plain once it is.
- Renders run in a separate process, never on the UI thread; the HUD shows
  progress; renders of the whole project (cmd+M) wait for (or start) any
  missing Wobble renders first and say so.
- 4K EXR half at 24 fps is ~400 MB/s of reads; the NVMe/RAM frame cache and
  multi-threaded EXR decode keep playback at speed on the target machines;
  proxies are the answer on smaller ones.

## Open questions for you

1. (Settled: no export; Wobble reads the original through FFmpeg.)
2. **Handles**: 1 s each side by default — enough?
3. **Where versions live**: beside the project (`name.wobble/`), or inside
   an organisicator post's folder when the project is a post's edit?
4. **Locked nodes in Wobble**: OK to add a small `locked` node flag to Wobble
   (on a local branch for you to merge), or keep Scrub re-asserting them?
5. **One Wobble per clip, or per sequence span** (a comp over several clips,
   e.g. a title over a cut)? This design is per clip.
6. **Render on save automatically**, or only when asked?

## Status (built)

`cargo build --release -p app --features wobble`, a Wobble built with
`--features app/movie`, then sequence > experimental > Wobble clips:

- cmd+shift+W on a picture clip: a hidden Wobble worker builds v001
  (SCRUB SOURCE reading the original file -> SCRUB OUT), renders the used
  range + handles to EXR in linear Rec.709 and saves the setup; the clip
  then plays the render. A Wobble without FFmpeg is refused with the build
  command.
- cmd+shift+W again opens the Wobble app on `work.wbl` (`--api`); every save
  there becomes the next version, rendered in a worker (both locked nodes
  re-asserted; a setup without either is refused in words), and plays when
  it lands. A clip trimmed past its range gets a wider render as a new
  version when opened again.
- The "wobble clip" window lists versions, picks the active one, and opens
  Wobble. The timeline labels the clip `W v002`, cyan when rendered, red
  while it plays its source.
- Renders (cmd+M) need nothing special: the clip's media IS the render.
- No Wobble changes were needed (its `--worker`, API and movie reader do it).
- `--test-wobble` (feature build) runs it against the real Wobble: v001 from
  the movie matches the source picture with nothing exported, a save in
  Wobble (a Mult ×0.25 added to the working copy) becomes v002 and plays
  darker, v001 can be picked back, the sequence renders, and with
  `EDIFY_WOBBLE_STOCK` a Wobble without FFmpeg is refused with the command.

Not yet: moving an unsaved project's comps when it is first saved, proxies,
a lock flag inside Wobble, deleting versions, render progress drawn on the
clip.
