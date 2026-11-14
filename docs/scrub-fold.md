# Scrub in this workspace, and the `fold` feature

One workspace, two binaries. `wobble` is unchanged. `scrub` (scrub/)
builds on its own and links no wobble code; with the `fold`
feature (experimental) it hosts wobble in process.

```bash
cargo build --release -p scrub_app                      # plain Scrub
cargo test --release -p media -p model -p colour -p scrub_api -p scrub_app
```

```bash
# FFmpeg (the one Scrub builds against) must be found by pkg-config.
cargo build --release -p scrub_app --features fold      # target/release/scrub
target/release/scrub some_movie.mov
```

Sibling checkouts as Scrub needs them: `../imfram`, `../seq_manip_imfram`,
`../sequitur`, `../sequence_manipulator`. Cargo reads the manifest of every
path dependency in the workspace, optional or not, so until that migration
lands a clone of this repo needs `../seq_manip_imfram` beside it to build
anything (wobble included).

Scrub's headless suites (`--test-tools` among them) read media from
`scrub/testdata/`, which is gitignored: run `scrub/testdata/gen.sh`.

## Using it

1. Put a clip on the timeline and select it.
2. **cmd shift W** (or Enter on a clip that already is a wob). The clip
   becomes a wob clip: a wobble comp, `SCRUB SOURCE -> SCRUB OUT`. Wobble
   takes the WHOLE window (the first time it takes a moment to start).
3. Build between the two nodes. Scrub shows the **SCRUB OUT** node,
   whatever wobble's viewer is pointed at.
4. **cmd shift W** again, or "back to scrub" in the bar, returns to Scrub.
   "own window" (the bar, or the wob menu) puts wobble in a window of its
   own instead, for a second monitor. Wobble is never side by side with
   Scrub in one window.
5. **Controls.** In wobble, add control nodes (slider, rgb, wheel, label)
   to the comp's top level, cable them to what they drive, and arrange them
   with the panel composer, exactly as for a subwob. They appear in Scrub's
   **wob controls** panel for the wob clip under the playhead, drawn by
   wobble's own panel widgets, and work there with wobble closed.
   A two-value row (scale, translate) dropped on the composer comes in as
   it is on its node: two drag fields and a padlock. For a node's WHOLE
   panel, select the node and press "+ selected node" in the composer: it
   shows under the controls in Scrub, drawn by wobble's own properties rack.
   The playheads follow each other both ways while wobble is in its own
   window.
6. Park or scrub on another wob clip and wobble (and the controls panel)
   swap to that clip's comp, saving the one it had. Saving the Scrub
   project saves the open comp too.
7. cmd alt W gives the clip its footage back.

Comps are `.wbl` files in `<project>.wobs/` beside a saved project (in the
temp folder for an unsaved one). The clip stores the path.

## How it works

- `app` is a library (app/src/lib.rs) with a thin `main.rs`; Scrub links it as `wobble_app`.
- `app/src/hosted.rs`: a hosted `Gfx`. Built on the host's GPU device with a
  window that is never shown; `redraw` draws the whole frame into a texture
  instead of a swapchain and takes egui events from the host. The scripting
  API is callable in process (`Hosted::call`).
- `scrub/app/src/fold.rs`: starts it on demand, forwards input over the
  panel, runs one wobble frame per Scrub frame while the panel or a wob clip
  is showing, and hands the output node's texture (same device, no copy) to
  the record viewer, converted from the comp's working space to Scrub's.
- During plain forward playback wobble plays the comp itself and is put
  back in step when it drifts; paused or scrubbing, it is sent each frame.
- Scrub does not decode the footage under a live wob clip.

`SCRUB_TEST_FOLD=<folder> target/release/scrub movie.mov --shot x.png` runs
it headless: makes a wob, edits it through the API, clicks inside the
panel, plays through it, writes a.png / b.png / c.png and prints timings.

## Left alone on purpose

- No baked tier: playback renders the wob live. Wobble's own movie decode
  is the limit (4K ProRes HQ was at the edge of real time on Linux).
- Export, stills and the source viewer show the footage, not the wob.
- A dissolve shows the wob at full strength on its outgoing side only.
- One comp open at a time: a cut between two wob clips swaps comps there.
- Speed changes and reverse on a wob clip seek per frame (they trail).
- The output node is found by its name (`SCRUB OUT`); a comp without one
  shows what wobble's viewer shows. While the viewer is on another node the
  output is a second render (as dual view's other pane is).
- The controls shown are those of the comp under the PLAYHEAD, not of the
  selected clip, and their values are saved in the comp file, not in the
  Scrub project (no undo in Scrub for a knob).
- Wobble in its own window is the dock's torn-off window: untested here.
- Torn-off wobble tabs, drag and drop into the panel, IME and the hosted
  window's own cursor capture are untested.
- Scrub's crates are simply copied in (`scrub_app`, `scrub_api` renamed);
  its `colour` crate, recorder and decoder pool still duplicate wobble's.
