# Scrub notes

## Build and run

Needs a system FFmpeg with its headers and pkg-config files (`brew install
ffmpeg pkg-config` on macOS; `libavcodec-dev libavformat-dev
libswscale-dev` on Debian/Ubuntu). Tested against Homebrew's FFmpeg 9.0.2.
imfram is a path dependency: `../imfram` must be checked out next to this.

```bash
cargo build --release
target/release/scrub testdata/prores_hq_4k.mov       # a movie
target/release/scrub testdata/exr/shot.1001.exr      # a sequence, from any frame
target/release/scrub                                 # then cmd O
```

macOS only for now (VideoToolbox, `~/Library` paths).

## Hotkeys (all)

Single letters act on the record side unless the bin has focus. The key
hints over the viewers list most of these (H hides them).

| keys | what |
|---|---|
| **panels (imfram dock)** | |
| Q | the panel under the pointer alone (Q / Esc: back) |
| W | close the panel under the pointer (view menu or shift W brings it back where it was) |
| shift W | every panel back |
| drag a grip (⠿) | move the panel: the leaves glide to preview, drop to commit; out of the window: a window of its own |
| H | key hints over the viewers on / off |
| shift H | the viewers' readouts (frame, decode, cache…) on / off; also view menu |
| drag a divider | resize panels; the layout (and torn-off windows) is saved in `ui.json`; window > reset layout |
| shift 1 / 2 / 3 | focus bin / source / record |
| ` (backtick) | edit / selects mode (also the menubar's edit · selects) |
| **selects mode** (see "Selects mode") | |
| click a clip (list) | view it in the one viewer |
| S / R / U | select / reject (again: clear) / clear the verdict |
| 1 2 3 4 | list all / selects / unmarked / rejects |
| up / down | previous / next clip in the list |
| M | the source in..out into the selects bin as a subclip (repeatable per clip) |
| F | fit the viewer (R is reject here) |
| **transport** | |
| J K L | shuttle back / stop / forward (again: faster) |
| space | play / stop |
| left / right | frame step (the record playhead goes past the last clip) |
| home / end | start / end of the material |
| up / down | previous / next edit point |
| **marks and edits** | |
| I / O | mark in / out (focused side) |
| shift I / O | go to in / out |
| alt I / O / X | clear in / out / both |
| , / . | insert / overwrite (four marks set: fit to fill, said on the HUD) |
| shift . | fit to fill |
| 1 2 3 | target V1 V2 V3 (again: off); the sound target follows to A1 A2 A3 |
| 4 5 6 7 | target A1 A2 A3 M (again: off) |
| **tools** (strip at the timeline's top left) | |
| V | select: click, drag to move, drag edges to trim, drag a lane for a marquee |
| M | append: the source in..out (whole clip with no marks) after the last clip, targeted tracks + linked sound, one undo; playhead to the new end. With the bin focused: its chosen clips / subclips, in bin order |
| X | razor: click cuts (with the clip's sound), shift-click cuts every track (was R) |
| R (over the timeline) | fit the timeline to the whole sequence |
| N | roll: drag a cut, one clip longer and the other shorter |
| (no key; sequence > tools) | rate stretch tool: drag an edge, the speed changes to fill the new length (hold shift A / shift D does it with any tool) |
| Y | slip: drag a clip to show other source in the same place |
| U | slide: drag a clip; its neighbours (or the gap) give and take |
| cmd K / cmd shift K | cut at the playhead on the targeted tracks / every track |
| **hold keys** (held, then press anywhere on a clip and drag) | |
| A / D | trim the clip's in / out point (plain: the cut moves) |
| Z / C | ripple trim its in / out point |
| S | slip it |
| shift A / shift D | rate stretch from its in / out (the other end stays, the speed changes) |
| shift S | its speed, scaled about the playhead (live speed on the HUD) |
| | a held key that dragged does nothing when let go; D tapped alone still selects; the label by the pointer names the op while the key is down |
| **clips and edit points** | |
| D / shift D (tapped) | select the clip under the playhead / next track's |
| del / shift del | lift (gap) / ripple delete |
| G | select the edit point (again: out / in / roll) |
| alt left / right | trim the edit (shift: 10) |
| B | ripple / plain trims (plain by default, every session; the status line says which) |
| E | extend the edit to the playhead |
| alt , / . | slip (shift: 10) |
| cmd R | clip speed |
| alt R | conform the clip on / off: one source frame per sequence frame (no drops or repeats at another rate; length follows, sound muted) |
| cmd D | dissolve: on the selected edit, between two selected clips that meet, or the cut nearest the playhead |
| cmd shift D | remove the dissolve |
| alt = / alt - | longer / shorter dissolve |
| T | snapping |
| Esc | cancel a drag / clear the selection |
| **reframe** | |
| F / alt F | fit / fill (reset) |
| [ ] (shift: fine) | zoom |
| cmd arrows (shift: 10%) | move |
| **viewers** (pointer over one) | |
| wheel / pinch / ctrl + two fingers | zoom about the pointer |
| two fingers, MMB drag, alt+LMB drag | pan |
| ctrl+MMB drag, cmd+RMB drag | drag-zoom |
| R | fit that viewer (over the timeline: fit the timeline) |
| drag the picture | scrub, RV style: relative, 5 pt a frame, either viewer (a drag on a title still moves it) |
| I O, shift / alt I O, alt X | marks on the side under the pointer (source viewer, or record viewer / timeline), else the focused one |
| shift R | the other viewer the same size, zoom and pan |
| × in/out (title bar) | clear that side's in and out |
| **timeline view** | |
| = / - / \ | zoom / fit |
| **mouse** | |
| alt drag | duplicate the clip(s) |
| cmd drag | insert move |
| shift-drag the source picture, or its title bar's "drag clip to timeline" | onto the timeline: overwrite on that row (cmd at the drop: insert) |
| × clear in/out (red, title bar / timeline ruler) | clear the marks; shown whenever a side has one |
| drag a bin item | the same, the whole clip |
| drag "cross dissolve" (bin) | onto a cut |
| right-click a cut / clip | add / remove dissolve, select edit, razor, speed, delete |
| **titles** | |
| cmd shift T | new text clip at the playhead (3 s, above the targeted track), text window open |
| Enter / double-click (text clip) | edit its words and style |
| cmd arrows (text clip selected) | move its STYLE (every clip of the style); shift: 10% |
| drag a text in the record viewer | move its style |
| drag "T text" (bin) | onto a video track: a text clip there |
| **project** | |
| cmd S / cmd shift S / cmd O | save / save as / open |
| cmd I | import (browse) |
| cmd M | render |
| cmd T | optimise the source |
| cmd Z / cmd shift Z | undo / redo |
| cmd Q | quit |

## Selects mode

`app/src/selects.rs`. ` (backtick) or the menubar's "edit · selects"
switches. The main window drops the dock for its own layout: the bin's
clips listed down the left (verdict badge, "n sub" / "in/out" on the
right), the source viewer big, and the SELECTS bin of subclips along the
bottom as hover-scrub tiles. Torn-off dock windows stay as they are, and
the dock layout comes back untouched in edit mode.

- A single click on a clip views it. S selects, R rejects (again: clears),
  U clears; 1-4 filter the list; up / down walk it (and load).
- I / O then M: a subclip of that range into the selects bin (no marks:
  the whole clip). As many per clip as you like; the exact same range twice
  is refused. A clip with no verdict becomes a select with its first
  subclip. Subclips are named `clip #n`.
- Every clip remembers its own in / out (both modes): leaving a clip keeps
  its marks on its bin item, loading it puts them back. Marks put on by a
  drop, an append or a subclip load are not remembered as the clip's own
  (`SourceClip::set_marks`).
- Edit mode: the selects bin is the bin's first group. A subclip tile shows
  its in frame and scrubs only its in..out (long-GOP H.264 scrubs from the
  240-frame strip, so a short subclip of a long H.264 is coarse until the
  exact frames arrive; intra clips are exact). Double-click / Enter loads it
  into the source with its range as the marks (then , . or M). A drag lays
  it down (several chosen: one after another, `Payload::Clips`). M with
  the bin focused (shift 1) appends the chosen items in bin order.
- Keys that mean something else in edit mode: R (fit viewer -> reject; F
  fits instead), M (append -> add subclip), S (slip hold, timeline only),
  1-4 (targets -> filters).
- Saved with the project: format 5 (`BinEntry.range / rating / marks`); an
  older Scrub refuses a format 5 file instead of reading subclips as whole
  clips.

## Clip browser (the load-clip UI)

`app/src/clipbrowser.rs` embeds seq_manip on imfram
(`~/src/devl/seq_manip_imfram`, its `seq` library, branch `edify-embed`):
the same grid / list, thumbnails, path bar, filters, subfolder search,
type-ahead, staging verbs and toasts as the standalone app, on its prefs.
It is THE way clips come in, presented as wobble presents its browser: a
big window (wobble's fills the app; this one fills everything above the
timeline so clips drag straight down onto it), no fade-in.

- Opened by cmd+I, file > import and the bin's "browse…" button; at the
  bin's folder when that changed (a post's rushes) since it last opened.
  Closed by × or Esc.
- Enter / double-click on a sequence or movie: into the bin AND the source
  viewer, window closes (wobble's single pick). "add to bin (n)": the
  selection joins the bin, window stays (wobble's collect into bin).
- Drag rows out onto the timeline (overwrite, cmd: insert): every dragged
  clip, laid down in order (`Payload::Files`). Onto the bin: added. Inside
  the window a drag still moves files into a folder.
- Layout (`seq::embed`, branch `edify-embed`): seq_manip's dock flattened —
  Favourites + the folder tree down the left (as its Folders tab), the
  browser, Info / Pending / Delete queue down the right; both sides
  resize, widths kept in seq_manip's prefs. No Player (Scrub's source
  viewer plays the pick). Its keys over the window: B, cmd+A, cmd+F,
  cmd+↑, F2, cmd+Enter (no Delete key, no Esc: Esc closes).
- TO IMPORT tray at the foot of the left side (seq_manip's bin, where
  wobble keeps its collect-into-bin tray): B / "add to import" / a drag
  collects across folders with thumbnails and frames; Backspace / Remove,
  ⏶ ⏷ reorder, Clear; "Import (n)" puts them in the bin in that order
  (hand-added bin items now keep their arrival order) and closes; dragged
  from the tray to the timeline they lay down in that order.
- Organisicator shortcuts (project / shot / its folders) sit at the top of
  the window inside a project. The bin is now just the bin (the old
  imfram-filebrowser browse tab is gone); a post's rushes bins still fill.
- Over the window Scrub's keys stand down.
- `clips` feature, ON by default (needs the sibling checkouts
  seq_manip_imfram, sequitur, sequence_manipulator); without it import is
  a file dialog.
- Listing is synchronous, as wobble's is (testdata: a few ms; the test
  fails over 500 ms); thumbnails and metadata come off-thread.

## Screen recorder (wobble's)

`app/src/recorder.rs` + `recorder/` (encode_mac, encode_win, overlay, shot)
are wobble's, ported WITHOUT publishing: no share sheet, compose pages or
blog (`share.rs` is now only the save folder, names and "show in
Finder"). The menubar's rec button (right-click: window or record viewer,
cursor, "made with Scrub" mark, GIF, folders) and cmd+shift+R start / stop
a take; the camera button saves a PNG.

- Capture is the app's own frames, never an OS grab: imfram's
  `UiWindow::draw_with` (added for this, local imfram commit 81018e3, not
  pushed) hands main.rs the presented frame, which copies it to the
  recorder's readbacks; the surface carries COPY_SRC only while recording
  (`set_copy_src`). Frames: LZ4 on disk, review window to trim, then the
  OS's H.264 (AVAssetWriter / Media Foundation; else an installed ffmpeg)
  and an optional GIF.
- Clips land in ~/Movies/Edify, screenshots in ~/Pictures/Edify (wobble's
  defaults under its own name; `EDIFY_RECORD_DIR` wins). wobble's recorder
  writes no sidecar and does not use organisicator, so neither does this.
- Torn-off dock windows are not recorded (the main window only).
- `--test-record`: a 2.5 s take while playing (frames copied from the
  offscreen frame, as the draw loop copies the presented one), ~69 frames,
  saved to MP4 + GIF and read back with ffprobe. encode_win.rs is not built
  on this Mac (as in wobble: CI's verdict).

## Colour management (wobble's)

`colour/` is wobble's `color.rs`, `viewtf` (ACES 1.3, AgX from Blender's
cubes, the filmic fit, None; OpenDRT is GPL upstream, its stub only) and
`lut.rs`, copied with their 50 tests. Keep it in step with wobble.

- **Viewer**: the convert pass decodes to code values and runs the
  file's INPUT transform (`color::Transform::wgsl`, one pipeline per plane
  kind x transform) into the WORKING space, scene-linear rgba16f; the
  display pass is wobble's blit tail: exposure, `view_transform`, gamma,
  encode (`ViewSettings::wgsl` / `fill_uniform`, AgX cube bound at 4).
  Dissolves mix the two pictures after the view, as displayed. No CPU
  colour pass.
- **Settings** live in the project (`Sequence.colour`, model
  `ColourSettings`): working space, view (transform, display, curve, look,
  exposure, gamma), per-file input spaces. Menubar > colour; a timeline
  clip's right-click > input colour; colour > source clip read as.
- **Defaults keep the old picture** (screenshot within 1 code value,
  51 dB): working Linear Rec.709, view None to sRGB, movies and stills as
  Rec.709 primaries with the sRGB curve (what Scrub and the Wobble clip
  round trip always showed), EXR as the working space, DPX as Cineon
  (wobble's guess). Pick ACEScg + ACES 1.3 to look like wobble.
- **Matches wobble exactly**: `--test-colour` shows
  `testdata/colour/patches.1001.exr` (24 flat patches, some past 1.0) in
  an ACEScg project through ACES 1.3 and compares it to
  `testdata/colour/wobble_aces13.png` (wobble's `viewer_capture`, default
  prefs, run with a throwaway HOME): 0/255 worst difference. It also checks
  exposure and an input transform.
- **Wobble clips**: SCRUB SOURCE reads the clip's own input space, SCRUB
  OUT writes the project's working space (how Scrub reads an EXR); the
  empty comp is still the source picture (`--test-wobble`).
- **Export and stills** (`render.rs`, `viewer::Offscreen`): the same
  passes on a device of the render's own: each picture through its input
  transform and the view into an RGBA8 frame of the output size (placement
  and dissolves in the shader), read back, then only packed to BT.709
  limited 4:2:0 for the encoder; text is laid over as before. Output
  transform (`render::output_view`, every preset is a Rec.709 upload): the
  project's view to a Rec.709 display with the sRGB curve, no viewer
  exposure / gamma. The old swscale path remains only for a machine with no
  GPU. `--test-colour` holds an export still to the viewer (worst 1/255
  through 4:2:0); 240 frames of 4K ProRes render to 1080p at 48 fps.

## Image sequences on network storage

`media/src/netio.rs`, `fetch.rs`, `netcache.rs`, `dpx.rs`; the frame server
passes its wanted list to `Source::prefetch`; readouts + warning in the
viewers; sequence > cache shot locally; file > media caches….

- **Read-ahead** (`fetch::Fetcher`, one per sequence): 12 reader threads
  on a share (8 locally), a raw-bytes cache budgeted in bytes (2 GB on a
  share, 768 MB locally; `EDIFY_RAW_MB`, `EDIFY_READERS`), reading on in
  the play direction past the decoded prefetch as deep as the budget goes
  (≤ 400 frames). A scrub replaces the queue (cancel); running reads land.
  Reads bypass the page cache on macOS (`F_NOCACHE`).
- **Network detection**: the volume's fs type (smbfs / nfs / afpfs /
  webdav / cifs). **Simulator**: `EDIFY_NETSIM=1g` (110 MB/s, 3 ms),
  `10g` (1.1 GB/s, 2 ms) or `MB/s:ms`: per-read latency, one shared link.
  No share was mounted here; all network numbers are simulated.
- **Local disk cache** (`netcache`): frames read off a share are copied to
  `~/Library/Caches/Edify/network-frames` (key = path + size + mtime, LRU,
  `EDIFY_NETCACHE_GB`, default 50, 0 off), written off-thread; read from
  there next time. "cache shot locally" copies whole sequences (8 threads,
  progress in the record readouts). Shown and cleared in the media caches
  window.
- **Decoders from memory**: EXR via the exr crate on the bytes (half output
  is one memcpy now), DWAA/DWAB fall back to FFmpeg's EXR decoder (the
  exr crate has no DWA), DPX by our own parallel decoder (8/10/16-bit,
  methods A/B, both byte orders): 4.4 ms vs FFmpeg's 35.7 ms at 4K.
- **Never slow silently**: readouts show network MB/s, fps managed,
  frames ahead, local-copy hits; when playing and the read-ahead has run
  dry under the needed rate, a red banner over the picture says
  "network: x of y fps — frames will drop" and points at cache shot
  locally. Drops are counted as before.
- Measured (`cargo run --release -p media --example netbench`, 7 decode
  workers, M4 Pro): link-bound at 1 GbE (2K ZIP 16.5, 2K DPX 12.4, 4K ZIP
  4.2 fps) and at 10 GbE for big frames (4K none 20, 4K DPX 31, 4K ZIP 40);
  PIZ is decode-bound (4K 13 fps even locally; the exr crate beat FFmpeg on
  PIZ / ZIP / none). In the app: 1080p ZIP EXR at 10 GbE 22.8 pictures/s,
  0 dropped; at 1 GbE 12.8 with drops shown; the second pass from the local
  copies 23.3, 0 dropped.
- Not yet: proxies / reduced-resolution playback when the link cannot keep
  up, DPX unpacked on the GPU (the CPU unpack is 4 ms at 4K), timeline
  cache-fill bars for the record side.

## Panels on the dock

`panes.rs` + the end of `editor.rs`: bin, source, record and timeline are
tabs of imfram's `dock` (wobble's and graav's): a split tree, a grip on
each panel's top row (bin header, viewer title bars; the timeline's floats
top right), drag to rearrange with the gliding preview, drag out = a real
OS window (or into another Scrub window), right-click a grip to detach or
move. The main window is the dock's EXTERNAL host: `Editor::ui` runs the
per-frame work and the menubar, then `Docks::draw_external` in the room
under it; torn-off windows are imfram's, routed by window id
(`dock_event`), made and closed between frames (`dock_between`), each
given its own `viewer::Viewers` GPU state (a viewer re-uploads its frame on
the first draw in a new window).

- Q = the dock's solo key; W closes a panel and remembers its window's
  tree, so reopening (view menu / shift+W) puts it back exactly where it
  was when nothing else moved (wobble's props_tree trick); else at home.
- shift+R evens the split between the viewers (`panes::match_viewers`).
- Layout saved in ui.json (`dock`), restored at the first `about_to_wait`;
  a pre-dock ui.json seeds the first tree from bin_w / timeline_h / split.
  window > reset layout closes torn-off windows and puts the default back.
- Keys work in torn-off windows too (`Panels::begin` runs `keys`); the
  floating windows (clip browser, text, Wobble clip, export…) stay egui
  windows of the main window.
- `--test-mouse` section 9/10: W, shift+W back in place, Q, the divider,
  tear the record viewer off and play it there (0 dropped, pictures/s
  within 15% of docked; measured 15 docked / 17 torn on that 12-frame-cut
  sequence), reset layout.

## Bin tiles + hover-scrub

The bin's list / tiles switch and tile-size slider (kept in ui.json). A tile
shows the clip's poster; moving the pointer across it scrubs the clip with a
thin playhead line and a frame counter; name + duration underneath. Click,
cmd click, shift click select; several drag onto the timeline in bin order;
double-click / Enter loads into the source.

- `app/src/binthumbs.rs`: three workers with decoders of their OWN (never
  the viewers' frame servers), frames boxed to 192x108 RGBA on the CPU
  (`media::thumb`, ~1.5 ms a frame from the native planes), textures under
  a 96 MB budget (LRU). Priority: the hovered frame, posters of tiles on
  screen, then one STRIP pass at a time.
- Strip: on first hover, ONE forward decode pass keeping every frame of a
  clip up to 240 frames (every `n/240`-th past that). Long-GOP H.264 seeks
  at ~0.5 s a frame at 4K but decodes forward fast, so it scrubs from the
  strip only; intra clips (ProRes, EXR) also ask for the exact frame (one
  slot, a newer position replaces it).
- Measured (`--test-tools`, 4K testdata): ProRes exact frame cold ~30 ms,
  EXR ~15-30 ms, H.264 cold ~0.9 s (until the strip reaches it); strips fill
  in 0.5-5 s; after that every position is exact on the redraw the pointer
  moved in (~0.3 ms). Posters for the visible tiles ~2 s.
- Bin items are files, so text clips never appear there; a Wobble clip's
  render is an EXR sequence and tiles like any.
- Hand-added bin items keep arrival order (imports land in tray order).

## Wobble clips (experimental)

Design and status: docs/wobble-clips.md. From a clean state:

```bash
cd ~/src/devl/wobble && cargo build --release -p app --features app/movie   # Wobble with FFmpeg (reads the original movie)
cd ~/src/devl/scrub && cargo build --release -p app --features wobble
target/release/scrub
```

Then sequence > experimental > Wobble clips (on once, saved), select a
picture clip (V1-V3, not text) and press cmd+shift+W.

- **Finding Wobble**, first found of: `$EDIFY_WOBBLE` (or
  `$EDIFY_WOBBLE_BIN`), a checkout beside Edify's
  (`<scrub>/../wobble/target/release/wobble`), ~/src/devl/wobble's release
  build, Wobble.app in /Applications or ~/Applications, `wobble` on PATH.
  None: the HUD says how to fix it.
- **Wobble needs `--features app/movie`** (FFmpeg): the comp's source node
  reads the ORIGINAL movie (image sequences by their pattern); nothing is
  exported. A Wobble without it is refused with that command on the HUD.
  (Your current ~/src/devl/wobble build already links FFmpeg.)
- **First cmd+shift+W** renders v001 (the used frames + 1 s handles) in a
  hidden Wobble worker: a few seconds for short clips, longer for 4K (the
  timeline draws the clip red until it lands). A clip later trimmed past
  that range is rendered wider as a new version when opened again. Then cmd+shift+W
  again (or "open in Wobble" in the wobble clip window) opens the Wobble app
  on the clip's comp.
- **In Wobble**, wire anything between SCRUB SOURCE and SCRUB OUT and save
  (cmd+S): each save is a new version, rendered and then playing in Scrub.
  Do not delete or rename those two nodes (a version without them is
  refused).
- **Where comps live**: `<project>.scrub.wobble/<clip>/` beside a saved
  project; an unsaved project's go under ~/Library/Application
  Support/Edify/wobble (save the project before starting, for now: they do
  not move yet).
- `--test-wobble` (feature build) is the end-to-end check. Project format 4
  (a clip's `wobble` versions); a default build plays such a clip's last
  render like any media.

## Titles and subtitles (text clips)

- **Text clips** live on video tracks; their media is a generated "text"
  item (no file, endless handles). Each carries its WORDS and the index of
  a named STYLE (`model::TextStyle`: font, face index, size as a fraction
  of the frame height, colour, alignment, line spacing, outline, shadow,
  box, and the position as fractions of the frame with an anchor).
- **Style vs content**: retyping a clip never moves it; editing the style
  (the text window's lower half, cmd+arrows, a drag in the viewer)
  restyles and moves EVERY clip using it. The subtitle workflow: alt-drag a
  subtitle along, Enter, retype. "new style from this" forks a style.
- **Layering**: `picture_at` looks through text clips; `text_at` lists the
  text to draw over the topmost picture, bottom to top. A dissolve on a
  text clip's cut fades it (text to text crossfades).
- **Drawing**: `app/src/text.rs` lays out and rasterises once with ab_glyph
  into a premultiplied RGBA block (box, soft shadow, outline by dilation,
  fill). The viewer caches it as a texture made at up to 1080 lines and
  warms the next second's text; the renderer makes it at the output's size
  and blends it into the YUV frame (`render::draw_text`), so viewer,
  render and `viewer_capture` agree. 40 cuts + 40 subtitles: 0 dropped;
  render 61 fps (unchanged).
- **Fonts**: the built-in face (egui's Ubuntu, nothing to install), any
  font in /System/Library/Fonts, /Library/Fonts or ~/Library/Fonts (the
  picker lists them by file, with a search), or any TTF / OTF / TTC file.
  A font that will not load falls back to the built-in face.
- **Places**: centred, title top, lower third, subtitle, top left
  (title-safe 10%, subtitles 5% from the bottom).
- **SRT**: file > import subtitles (text clips on the targeted video track
  in the subtitle style) / export subtitles (the track with the most).
- **Project format 3** (styles, clip text); older projects load with none.
- **MCP**: `text_create`, `text_set`, `style_list`, `style_set` (fields, or
  create from another), `srt_import`, `srt_export`; `timeline_get` shows
  each clip's text and style.
- **Tests**: model (layering, crossfade, shared style, SRT round trip,
  documents), `text::tests` (placement, scaling, outline / box / lines, a
  system font, fallback), `--test-tools` (cmd shift T, typing, Enter, Esc,
  viewer drag, alt-drag copy + retype, cmd arrows, bin drop), `--test-api`
  (capture with and without text, style edits reaching every clip, SRT out
  and in, a dissolve between subtitles, a render). `--subs N` puts N
  subtitles on V3 for playback / render checks.

## Hold-key drags and the key remap

While a hold key is down a press ANYWHERE on a clip does its op — the key
decides, not the edge: A / D trim in / out, Z / C ripple in / out, S slip,
shift A / shift D rate stretch from in / out, shift S the speed about the
playhead (`model::Sequence::speed_about`: the clip scales about that frame,
which keeps showing the same source). Same limits, snapping, trim monitor,
handle band and one undo step as the edge drags (`mouse::HoldOp`). They
are quasimodes: a key that dragged fires no tap; D's tap (select the clip)
fires on release only when nothing dragged, and only for a plain D (cmd+D
is still the dissolve). Edges without a key work as before.

The remap that made room: razor C -> R (over the timeline; R over a viewer
still frames it), snapping S -> T, select edit point T -> G, the rate
stretch TOOL lost its key (shift A / D do it; sequence > tools keeps it).
C and S have no tap action.

## UI batch: readouts, messages, glide, nav, framing

- **Readouts** (frame, decode, cache, marks over each picture): shift+H or
  view > viewer readouts, apart from H (key hints). Saved in `ui.json`.
- **Messages** about what was just done show in the TIMELINE panel, top
  right, fading after 4 s; the record strip keeps the status line (tool,
  trim mode, targets, undo). Timeline cursor labels were already there.
- **Panels** are on imfram's dock now (see "Panels on the dock").
- **Nav**: the record timeline takes imfram's `nav` table like the source
  scrub bar and wobble: wheel / pinch zooms about the pointer, two fingers
  and MMB pan, ctrl+MMB and cmd+RMB drag-zoom, alt+LMB pans from empty
  space (on a clip it is Premiere's duplicate).
- **Viewers zoom and pan** (`viewnav.rs`): wheel / pinch about the pointer,
  pan by two fingers, MMB or alt+LMB (a plain LMB drag stays the source's
  drag-onto-the-timeline). The title bar shows the zoom when not fitted.
- **Keys by pane**: R / shift+R over a viewer frame it / match the other
  viewer to it (size, zoom, pan); over the timeline R is still the rate
  stretch tool.
- **Clear in/out buttons**: "× in/out" in each viewer's title bar and on the
  ruler at the end of the marked range, beside alt+X.
- **Collapsed pane strips** read their name turned a quarter round (it was
  letter by letter); Scrub's panes are its own (`panes.rs`), so nothing in
  imfram changed.
- **Esc** no longer steals the key from an open menu (the right-click cut
  menu could not be closed with it), and buttons laid over the timeline
  sense drags so the panel underneath cannot take their press.

## Tools, audio tracks, drops (the editing batch)

- **Audio tracks**: V1-V3 carry their sound on A1-A3 (a clip edited onto V2
  gets its sound on A2; moving it to V3 takes the sound to A3) and M is a
  music track nothing links to (target it with 7). Every audio track is
  mixed in playback and in a render by one `audio::Mixer`; each header has
  M (mute), S (solo: only soloed tracks are heard) and a gain in dB (drag;
  double-click: 0 dB), all saved and undoable. Project format 2; a format 1
  project loads with A2, A3 and M empty and its audio target mapped.
- **Razor**: C, click; shift-click cuts every track; cmd K at the playhead.
  The cut snaps to the playhead and cuts (snapping on).
- **Past the end**: the record playhead can be clicked, dragged and stepped
  up to 10 minutes past the last clip; playback still stops at the end and
  End goes to it. An edit there leaves a gap.
- **Fit to fill**: with in and out on both sides, , and . stretch the
  source to the record range and say so ("fit to fill: 12 source frames
  over 6 record frames, speed 2.000x"). Equal lengths edit normally.
- **Rate stretch** is R (Premiere's key), so the ripple toggle moved to B.
- **Plain trims by default.** Ripple used to default ON (and was restored
  from the project), so a plain left-edge drag kept the clip's start and
  pulled everything after it along: it looked like a slip. Now every
  session starts on plain trims; a drag on an edge trims that side (the cut
  moves), ripple only with B or cmd, roll only with the roll tool (N; a
  selected roll edit point no longer turns drags into rolls). At a butt
  cut the side of the cut a press lands on picks the edge (it used to be
  the outgoing clip's out a few pixels into the incoming clip).
- **Trimming by drag** shows a blue band past the clip's edges: the source
  it has left there, with the frame count ("no handles" when none).
- **Slipping** (S held + drag, the Y tool, alt+, / alt+.) shows the same
  band for the slipped clip, live, with "no handle" on a side at its limit,
  and the viewers become a two-up: the clip's first frame left, its last
  right (for 1.5 s after a slip key, while that clip stays selected).
- **Dissolves**: cmd D, sequence > add dissolve, right-click a cut, or drag
  "× cross dissolve" from the bin onto a cut. Selecting two clips that meet
  on one track puts it on their cut; clips that don't meet (or sit on two
  tracks) are refused on the HUD. Drawn as a bright band with the fade's
  ramp and its length.
- **Drops**: shift-drag the source picture, or its title-bar handle (its marked range) or a bin file (the
  whole clip) onto a row: overwrite there, cmd insert. A video row takes
  the picture and its sound on the paired track, an audio row the sound
  alone. Record marks, targets and the playhead are left alone.
- **Model** (`model/src/edit.rs`): `razor`, `razor_tracks`, `slide`,
  `stretch`, `duplicate_clips`, `insert_fit`, `move_clips` across audio
  tracks, each unit tested.
- **Test**: `--test-tools` drives all of it through key and pointer events
  (and the timeline's own drop path), with or without sound in the file.
- **Font**: Scrub and wobble both use egui's built-in fonts (Ubuntu-Light,
  Hack) at egui's default sizes; neither loads a font file. Same look.

## Scripting API and MCP (Claude / agents drive Scrub)

The same design as wobble's (`api/` there): ONE protocol, newline JSON-RPC
2.0 on a loopback socket, described by one method table
(`api/src/lib.rs`, `api::methods()`). The app serves it; every client —
the MCP bridge, `--test-api`, a script — learns the methods from
`api_schema`, so nothing drifts. Scrub listens on **127.0.0.1:7980**
(`SCRUB_API_PORT` (or `EDIFY_API_PORT`), or `--api PORT`) whenever it runs with a window;
`--no-api` turns it off. Loopback only, and an HTTP request on the port is
hung up on (a web page cannot drive it).

- **Calls run between frames** on the UI thread (drained at the top of
  each pass, `app/src/apiserve.rs`), through the same model edits the keys
  and the mouse use.
- **Undo**: a mutating call is ONE undo step; `batch` ([{method, params}])
  is one step for all of it, all or nothing (the first error puts the
  sequence back and names the call). A failed call leaves no step.
- **Names**: tracks "V1" … "M"; a clip is {track, start} (its record start
  frame, from `timeline_get`); a cut is `at`.
- **Tools** (40): `app_info`, `timeline_get`, `bin_list`, `bin_add`,
  `media_open`, `project_open/new/save`, `post_open`, `sequence_set`,
  `marks_set`, `playhead_set`, `transport`, `targets_set`, `track_set`
  (mute / solo / gain), `edit` (insert / overwrite; four marks fit),
  `fit_to_fill`, `select`, `tool_set`, `trim` (out / in / roll, ripple),
  `slip`, `slide`, `razor`, `clips_move` (rows, insert, duplicate, alone),
  `clip_delete`, `clip_speed`, `clip_stretch` (rate stretch to a length),
  `dissolve_add/set/remove`, `reframe_set`, `viewer_capture` (PNG of the
  record or source picture, composed by the renderer's own code on a
  worker), `render_presets`, `render_start`, `render_status`,
  `render_cancel`, `undo`, `redo`, `batch`, `api_schema`.
- **Hints for agents**: each tool carries readOnly / destructive
  annotations (`ACTS`, `DESTRUCTIVE` in the table): reads are safe to
  auto-approve; opening, saving, rendering and playhead moves are not.

### Registering it

Build once (`cargo build --release` makes `target/release/scrub-mcp`),
then:

**Claude Code** (any project):

```bash
claude mcp add scrub -- ~/src/devl/scrub/target/release/scrub-mcp
```

**Claude desktop app**: add to
`~/Library/Application Support/Claude/claude_desktop_config.json`, beside
wobble's entry, and restart the app:

```json
"mcpServers": {
  "scrub": { "command": "<absolute path to>/scrub/target/release/scrub-mcp", "args": [] }
}
```

Start Scrub first (it listens by default); the bridge dials lazily and
redials, so Scrub can also be started (or restarted) after the agent. A
different port: `edify-mcp --port N`, or `SCRUB_API_PORT` (or `EDIFY_API_PORT`) on both sides.

### Tests

`cargo test -p api` (table invariants, validation, socket round trip,
the MCP handshake and image blocks); `apiserve::tests` (every method has an
implementation; one call = one undo step, a failed batch none);
`--test-api` drives the whole surface through the real socket while the
app draws — media, a batched edit undone in one step, dissolves, razor,
trims, slip / slide, speed, stretch, reframe, moves and copies, tracks,
captures (PNG), four-mark fit, a failed batch, save / reopen, a real render
— and then the MCP bridge listing and calling the same tools.

## Step 2: two-up editing

The window: the bin on the left (browse a folder, image sequences
collapsed to one item; enter or double-click loads into the source), the
source and record viewers side by side, the timeline under them (V3 V2 V1
A1). Focus is shown by a lit title bar.

| keys | what |
|---|---|
| shift 1 / 2 / 3 (or click) | focus bin / source / record |
| J K L | shuttle the last-focused viewer (L again: 2x, 4x … 16x) |
| space | play / stop |
| left / right | frame step |
| home / end | first / last frame |
| I / O | mark in / out on the focused side |
| shift I / shift O | go to in / out |
| alt I / alt O / alt X | clear in / out / both |
| , | insert (ripples every track) |
| . | overwrite |
| up / down | previous / next edit point (record) |
| 1 / 2 / 3 | target V1 / V2 / V3 (again: no video) |
| 4 | A1 on / off (now 4-7: A1 A2 A3 M; see Hotkeys) |
| = / - / \ | zoom the timeline / fit |
| cmd Z, cmd shift Z | undo, redo |
| cmd O | browse |
| cmd T | optimise the source clip |
| bin: up / down, enter, backspace, tab | select, load / open folder, up a folder or remove from bin, bin <-> browse |

Edits follow the usual three-point rules (`model::resolve`): the length
comes from the source marks, else the record marks; a lone out back-times
(past the sequence start the head is trimmed); four marks take the source
length; missing ins are the playheads. After an edit the record playhead
sits at its end and the record marks are cleared. The first edit sets the
sequence rate; sources at other rates are mapped by time.

The record viewer plays the sequence through the same frame servers
(`app/src/program.rs`): each redraw the prefetch window is laid over the
sequence and every clip's server gets the frames of its own stretch, so a
cut ahead is decoding before it arrives. The cache is keyed by FILE, so the
source viewer and the timeline share decoded frames. Starting playback
pre-rolls (the clock holds until the next 6 frames are in, at most 0.5 s),
which is what took the last start-up drops out.

Model: `model/` (no UI): `Sequence` with V1-V3 + A1, clips = media + source
in + record start + length, exact rational rates, insert / overwrite / lift,
edit points, topmost-video resolution; undo is imfram's `History` over
snapshots. 16 unit tests.

## Step 3: trim, timing, dissolves, sound

| keys (record or source focus) | what |
|---|---|
| D / shift D | select the clip under the playhead (targeted track first) / the next track's |
| delete / shift delete | lift the selected clip (and its linked audio) / ripple delete |
| T (again) | select the nearest edit point: roll, then out, then in |
| alt left / right (shift: 10) | trim the selected edit by frames |
| R | ripple trims (yellow bracket) or plain (red) — now B; R is the rate stretch tool |
| E | extend the selected edit to the playhead |
| alt , / alt . (shift: 10) | slip the selected clip |
| cmd R | clip speed (a float; negative plays backwards) |
| alt R | conform on / off (sequence settings: conform new edits at other rates) |
| shift . | fit to fill (source in/out stretched over record in/out) |
| cmd D, cmd shift D | dissolve at the selected (or nearest) cut, 24 frames / remove |
| alt = / alt - | dissolve 2 frames longer / shorter |
| esc | clear the selection |

While an edit point is selected (and not playing) the two viewers are a
trim monitor: the outgoing clip's last frame left, the incoming clip's
first right, both prefetched.

Model (`model/src/edit.rs`): every clip edit applies to the linked group
(picture and its sound) and CLAMPS to what the handles and neighbours
allow, so a held trim key stops at the limit with a notice. Plain trims
never overwrite a neighbour; ripple trims move what follows on every track
and are refused (in words) when another track has a clip across the cut.
Ripple delete is refused when another track has material under the clip.
Speed keeps the source a clip covers (2x is half as long), never grows a
clip into the next one, and reverse starts from the last frame it showed;
frames are the nearest-earlier source frame, no blending. Dissolves sit on
a cut, centred, clamped to the handles on each side (the notice says when),
and go when an edit breaks their cut; a roll carries them. 30 model tests.

Playback through speed changes needs nothing special: the prefetch window
is laid over SEQUENCE frames and each one resolves to the source frame its
speed samples, so a 2x clip asks for every other source frame. Through a
dissolve both pictures are resolved, prefetched and converted (the record
viewer's second slot), and one GPU pass mixes them.

Sound (`app/src/audio.rs`, `media/src/audio.rs`): A1 of the sequence, or
the source clip's own sound, through the default device (cpal / CoreAudio).
FFmpeg decodes and resamples to stereo f32 at the device rate; a feeder
thread renders 0.3 s ahead; the device callback counts what it played, and
that count is the clock: while sound plays, the playhead is placed from it.
Only 1x forward is heard: shuttle, reverse and speed-changed clips are
silent (not resampled). No crossfade on A1 through a dissolve. If the device
stops taking samples for 300 ms the picture carries on by the wall clock,
silent, with a notice. `EDIFY_MUTE=1` keeps the clock but sends silence
(the headless checks use it).

### Step 3 numbers

`--cuts 40 --fx`: 40 twelve-frame cuts on V1, every third clip at 2x, 0.5x
or backwards, an 8-frame dissolve on every other cut. Headless, 60 Hz pace,
8 s.

| clip | 1x | -1x | 2x |
|---|---|---|---|
| ProRes 422 HQ 4K, cuts + speed + dissolves | 0 dropped | 0 dropped | 0 dropped |
| H.264 4K 50 Mbit/s, same | 0 dropped | | 0 dropped |
| HEVC 4K 10-bit, same | 0 dropped | | 0 dropped |
| ProRes 4K + 48 kHz sound, 40 cuts, audio clock | 0 dropped, clock 7.66 s / 8 s | | |
| same with speed + dissolves, audio clock | 0 dropped, clock 7.70 s / 8 s | | 0 dropped (silent) |

Pictures per second read 21.6-23 at 1x: gaps (a 2x clip leaves one) show
black, which the meter does not count, and the pre-roll is inside the 8 s.
The audio clock reads ~0.3 s under the wall time for the same reason.
1-9 redraws a run go over 16.7 ms (17-20 ms): a dissolve frame uploads two
4K pictures in one redraw (~4 ms each on the CPU side).

## Step 4: projects and Organisicator

| keys | what |
|---|---|
| cmd S / cmd shift S | save / save as (`.scrub`; `.edify` still opens) |
| cmd O | open a project (media browse moved to **cmd I**) |
| file > open recent, relink media…, sequence settings… | |

The project file (`app/src/project.rs`) is versioned pretty JSON: bin,
sequence (clips, speeds, dissolves, rate, raster), marks, playheads,
targeting, the optimised copies in use, and every media path both absolute
and relative to the project. Moved media is found by its relative path;
media found by neither is offline (black) and the relink window locates it
one file at a time or searches a folder by name. A file from a newer
format is refused in words. **Saves never lose the last good file**: a
temporary file beside it, flushed, then renamed over it (a test makes the
folder read-only and checks the old file survives). The title shows a dot
while there are unsaved changes; quitting with unsaved changes asks (save /
don't save / cancel). Unsaved work is autosaved every 30 s to
`~/Library/Application Support/Scrub/autosave.scrub`. A clean quit removes
it, so finding one at start-up offers recovery; "discard" keeps it aside as
`autosave.scrub.discarded`. Recent projects are in `recent.json` beside it.
Headless runs never autosave, recover or prompt.

The sequence RASTER (from the first clip, or set in sequence settings with
presets for 1080p, UHD, 9:16 and 1:1) frames the record viewer: every
picture is fitted into it. The rate can be changed only while the sequence
is empty.

Organisicator: see docs/organisicator.md. Scrub reads organisicator's
project manifest (`.organisicator/project.json`, never written): it knows
the project and shot of any file, saving an untitled project suggests the
shot's Scrub folder and a versioned name (`sh0010_edit_v001.scrub`), and the
browser shows the project / shot with jump buttons to its folders.
Organisicator can launch Scrub with an editor rule (app = scrub, args = {}).

## Mouse editing on the timeline

The keys all still work; the mouse is added on top (`app/src/mouse.rs`).

| on the timeline | what |
|---|---|
| click a clip | select it (shift / cmd: add or remove) |
| click a clip edge | select that edit point (the trim monitor shows) |
| click an empty lane / the ruler | move the playhead there |
| drag a clip | move it in time and across tracks; its linked sound follows to the paired track (V2 -> A2). alt at the press: duplicate; alt+shift: the clip alone. Drop overwrites; cmd held inserts (every track opens a gap) |
| drag a clip edge | trim that side: the cut moves and a gap opens or closes; ripple when the trim mode is ripple (B) or cmd is held; rolling is the roll tool (N). The cursor and a label by the pointer say which: trim in / trim out / ripple (yellow) / roll / slip |
| drag a dissolve's edge | that side's length |
| S | snapping on / off |
| Esc during a drag | cancel it |

- **Snapping**: the dragged edge or edges jump to the playhead, the in and
  out marks, and every cut within 8 pixels; a yellow line shows where.
- **Live preview**: while a clip moves, its old place shows as an outline
  and the record viewer shows the frame at its head.
- **Undo**: every drag is computed from a snapshot of the sequence taken at
  the press, with the model's own edits (`move_clips`, `trim`,
  `set_dissolve_sides`, each unit tested). The release is ONE undo step;
  Esc restores the snapshot.
- **Decoding**: the preview only asks for the frame under the dragged head
  (plus the paused neighbours), so a drag never sets off a decode storm, and
  the cache stays valid because it is keyed by file and frame.
- **Test**: `--test-mouse` drives real pointer events through a move (sound
  follows), a move up two tracks, an edge trim, a snap to the playhead, an
  Esc cancel, a cmd insert-move, a dissolve edge and a roll, checking one
  undo per drag.

## Step 5: marketing posts and rendering

| keys / menu | what |
|---|---|
| file > open marketing post… | open (or start) an advanced organisicator post's edit |
| cmd M (file > render…) | render H.264 + AAC MP4: YouTube 1080p / 4K, Reels 9:16, feed 4:5, square |
| file > cancel render | |

A post's project is `posts/<slug>/<slug>.scrub`, beside `post.json` (read,
never written). A new one takes the post's frame shape as the raster, its
sources and `in/` files into the bin, and `rushes/` into the browser. A
render inside a post goes to `out/<slug>_<preset>_v###.mp4` and, if
ticked, a copy goes to `result/<slug>_v###.mp4`. Outside a post a save
dialog asks. Nothing is overwritten (temp name, rename, next free
name), nothing is copied into `rushes/`, nothing is published. See
docs/organisicator.md.

The renderer (`app/src/render.rs`) walks the record frames: each picture is
cropped (fill) or placed (fit) and scaled into the output in one swscale
call, converted to Rec.709 video range; a dissolve renders both pictures
and blends; speed changes resolve exactly as in playback; A1's sound comes
from the same `audio::Mixer` playback uses, encoded as AAC 48 kHz.
H.264 is VideoToolbox (`EDIFY_H264=x264` for libx264), bitrate per preset,
2 s GOP, faststart MP4, BT.709 tags.

### Frame shape, warnings, reframe

- **Post frame shape**: in a post, sequence settings has a "post frame
  shape" row (16:9, 9:16, 1:1, 4:5). It sets the raster and writes the
  post's `frame` into `post.json`, that field only:
  - read-modify-write with unknown fields and key order kept,
  - a hidden `.post.json.scrub-backup` of the file as it was,
  - temp file + rename.

  Nothing else in `post.json` is ever written.
- **Renders go to `out/` only.** Copying into `result/` (and the publish
  record) is organisicator's publish step.
- **Render warnings** (they don't block):
  - the preset's shape differs from the post's frame shape;
  - Reels over 90 s, or over 60 fps (YouTube and Instagram);
  - Instagram under 3 s or over 60 min; YouTube over 12 h.
- **Per-clip reframe** (`model::Reframe` on each clip: fit or fill, zoom,
  x/y offset as a fraction of the raster):
  - **F** fit / fill, **alt F** reset;
  - **[ ]** zoom 5% (with shift: 1%);
  - **cmd arrows** move 1% (with shift: 10%).

  It acts on the selected clip (D), or the clip under the playhead on the
  targeted track, with its linked group. Held keys make one undo step once
  they settle. The record viewer draws every picture (both sides of a
  dissolve) placed by its clip's reframe and cut to the raster, through a
  placement transform in the display shaders. The render places it with
  the same `Reframe::place` and clips to the raster's area of the output,
  so a reframed 16:9 shot in 9:16 renders exactly as it shows. It's saved
  in the project; it isn't keyframed yet.
- `--reframe-test` (with `--cuts`): a 9:16 sequence, every clip filled and
  moved 20% right, for shots and renders. `--test-keys` checks the reframe
  keys.

### Launched from Organisicator

`scrub --post <posts/slug> [--rushes <dir>]...` (how Organisicator's "edit in
Scrub" starts it) opens the post's edit with the post's own rushes in a
first bin and every rushes folder listed off-thread into bins by folder
(5,050 items in 32 ms; folded past 300). Renders are hidden `.partial` files
until complete. See docs/organisicator.md and, in Organisicator's
`edify-integration` branch, docs/edify-integration.md.

### Render speed

Sequential decode, single-threaded scaling, M4 Pro:

| source sequence (480 frames, 40 cuts + speed changes + dissolves) | preset | speed |
|---|---|---|
| ProRes 422 HQ 4K + sound | YouTube 1080p, VideoToolbox | 57 fps (8.4 s) |
| same | Reels 9:16 fill, VideoToolbox | 65 fps |
| same | YouTube 4K, VideoToolbox | 51 fps |
| same | YouTube 1080p, libx264 | 45 fps |
| H.264 1080p 25 fps + sound (120 frames) | YouTube 1080p / Reels fill | 183 / 216 fps |

`--test-render` checks the YouTube (fit) and Reels (fill) files with
ffprobe: H.264 at the preset's size, AAC present, frame count and duration
equal to the sequence's. `--test-post` checks the marketing round trip.

Headless checks (no window shown):

```bash
target/release/scrub FILE --shot out.png [--frame N]   # the window as drawn
target/release/scrub FILE --play 5 [--speed 4]         # playback, measured
target/release/scrub FILE --optimise [--dest beside|cache] [--profile lt|standard|hq]
target/release/scrub FILE --test-optimise              # in-app optimise + switch, PASS/FAIL
target/release/scrub FILE --test-keys                  # three-point editing through key events, PASS/FAIL
target/release/scrub FILE --test-mouse                 # timeline drags through pointer events, PASS/FAIL
target/release/scrub FILE --test-tools                 # tools, audio tracks, drops, dissolves: PASS/FAIL
target/release/scrub FILE --test-api                   # the scripting API + MCP bridge end to end: PASS/FAIL
target/release/scrub FILE --test-project               # save, reopen, move the folder, reopen: PASS/FAIL
target/release/scrub project.scrub                     # open a project
target/release/scrub FILE --cuts 40 --fx --render youtube-1080 out.mp4 [--fill]   # render, no window
target/release/scrub FILE --test-render                # render YouTube + Reels, check with ffprobe: PASS/FAIL
target/release/scrub FILE --test-post                  # marketing post round trip: PASS/FAIL
target/release/scrub FILE --cuts 40 --play 8 [--speed -1 --frame 479]   # timeline playback across cuts
target/release/scrub FILE --cuts 40 --fx --play 8      # ... with speed changes and dissolves
EDIFY_MUTE=1 ...                                       # sound clock without sound
cargo run --release -p media --example bench -- FILE   # decode benchmarks
testdata/gen.sh                                        # regenerate test media
```

Tuning knobs (env): `EDIFY_CACHE_GB` (default a quarter of RAM, max 16),
`EDIFY_WORKERS` (decode threads), `EDIFY_FF_THREADS=slice:N|frame:N`,
`EDIFY_HW=0|h264|all` (VideoToolbox decode; default HEVC only),
`EDIFY_PRORES=ks` (software ProRes encoder instead of VideoToolbox).

## Optimised media

cmd T (or file > optimise…, or `--optimise`) transcodes the open long-GOP
clip to ProRes 422 (LT / 422 / HQ) in the background, with progress in
the HUD; when it lands the viewer switches to it at the same frame.
**Originals are never written, moved or deleted**: the copy is written
under a temporary name and renamed when complete. Two destinations, picked
per transcode, the default in the dialog or the cache window
(`~/Library/Application Support/Edify/optimise.json`):

- **beside the original**: `name.scrub.mov` (older `name.edify.mov` copies are still used); the source's size + mtime are
  in its metadata, so an edited source no longer matches it.
- **managed cache**: `~/Library/Caches/Edify/optimised` (settable), files
  named by a hash of path + size + mtime, `index.json` with sizes and last
  use; optional GB cap with least-recently-used eviction; purge all /
  purge this clip (only the cache's own files).

Opening a file uses its matching optimised copy, beside the original
first. Audio streams are copied, frame rate, size, colour tags and
timecode carry over. Encoder: VideoToolbox ProRes (159 fps for 4K from the
50 Mbit/s H.264; `prores_ks` managed 10 fps).

## How playback is fast

- **Frames stay in their native planes.** FFmpeg's YCbCr frames are handed
  on as the decoder's own refcounted AVFrame (no copy, no CPU colour
  conversion) and converted to RGB on the GPU in one pass. A 4K ProRes HQ
  frame is 33 MB as 10-bit 4:2:2 against 66 MB as RGBA half.
- **Upload and convert happen inside egui's paint callback**, so the frame a
  scrub picks this redraw is on screen this redraw.
- **Intra codecs (ProRes, image files) decode in parallel**, any frame on
  any worker, each FFmpeg session SLICE-threaded. Frame threading delays the
  first picture by one frame per thread, which every seek paid: 700 ms
  seeks became 12 ms. And a warm intra session never "catches up" forward
  (it decoded frames other workers were already decoding: 20 fps became
  135).
- **Long-GOP (H.264) decodes as one forward stream** per playhead: the
  worker keeps its warm decoder and takes the next wanted frame; frames
  just ahead of a running stream are reserved so no second worker seeks
  into the same GOP; pictures decoded on the way to a target are cached
  when wanted (what makes reverse play affordable).
- **Cancellation is the queue.** The wanted list is rebuilt from the
  playhead every redraw; queued frames no longer wanted are dropped. A
  paused seek decodes its frame alone before neighbours start; while
  scrubbing only the frame under the hand and two past it are wanted.
- **The picture never runs ahead of the hand**: when the playhead's frame is
  not in, the viewer shows the cached frame nearest it between the current
  picture and the playhead (Wobble's scrub rule).
- RAM cache: LRU by bytes, shared by every server (two viewers on one clip
  will hold it once), frames around the playhead touched every redraw.

## Benchmarks, step 2: playback across cuts

`--cuts 40`: the clip cut into 40 twelve-frame pieces, each from elsewhere
in it, alternating V1 / V2 (a cut every half second), then played 8 s on
the record side, headless, paced at 60 Hz, waiting on the GPU every redraw.

| clip | 1x | -1x | 2x |
|---|---|---|---|
| ProRes 422 HQ 4K | 0 dropped (3 runs) | 0 dropped | 0 dropped, 47 pictures/s |
| H.264 4K 50 Mbit/s | 0 dropped | 0 dropped | 0 dropped, 47.5 pictures/s |
| HEVC 4K 10-bit (VideoToolbox) | 0 dropped | 0 dropped | 0 dropped, 47.4 pictures/s |

Before the pre-roll, ProRes dropped 1 frame in 2 runs of 3, always in the
first 4 frames after pressing play. Pictures per second read 23.5-23.9 at
1x because the pre-roll is inside the 8 s. 1-3 redraws a run went over
16.7 ms, the first redraw (pipeline set-up) among them.
Upload (CPU side of the GPU copy) 4 ms a 4K frame.

## Benchmarks, step 1b: VideoToolbox, scrub preview, optimised media

Same machine and bench; `jump` = the hand lands mid-GOP on an uncached
frame: time to a picture near it (the keyframe the decode passes) and to
the exact frame.

| clip | seek median | scrub exact | jump: first picture / exact |
|---|---|---|---|
| H.264 4K 50 Mbit/s (software) | 110 ms | 54% → 75% | 26 / 92 ms |
| H.264 4K ~550 Mbit/s (software) | 490 ms | 0% → 6% | 97 / 414 ms |
| HEVC 4K 10-bit 60 Mbit/s, software | 278 ms | 1% | — |
| HEVC 4K 10-bit 60 Mbit/s, VideoToolbox | 114 ms | 85% | 8 / 158 ms |
| optimised (ProRes 422) of the 50 Mbit/s H.264 | 10 ms | 100% | 17 / 17 ms |
| optimised of the ~550 Mbit/s H.264 | 9 ms | 100% | 19 / 19 ms |
| optimised of the HEVC | 9 ms | 93% | 20 / 20 ms |

VideoToolbox through FFmpeg's hwaccel decodes one picture at a time. It
wins on HEVC (seek 114 vs 278 ms, playback 166 vs 95 fps) and LOSES on
H.264 against 14 software threads (seek 222 vs 108 ms, playback 81 vs
266 fps), so it is on for HEVC only. Transcode speed (VideoToolbox ProRes
422): 50 Mbit/s H.264 4K 159 fps, grainy H.264 59 fps (decode-bound),
HEVC 79 fps, 1080p H.264 396 fps. Optimised files are ~9x the size of a
50 Mbit/s source (566 MB for 10 s of 4K).

## Benchmarks, step 1

M4 Pro (14 cores), 48 GB, macOS, FFmpeg 9.0.2, release build, files in the
page cache. Clips are 240 frames (10 s) of `testsrc2` with grain, made by
`testdata/gen.sh`. `bench` columns: raw = one thread, one warm decoder, in
order; play = through the frame server as fast as it delivers; seek =
random clicks 150 ms apart; scrub = a 60 Hz drag moving 2 frames a redraw.

| clip | raw | raw seek | play fwd | play rev | seek median / p90 | scrub exact |
|---|---|---|---|---|---|---|
| ProRes 422 HQ 4K (yuv422p10) | 137 fps | 7.3 ms | 136 fps | 140 fps | 12.5 / 13.8 ms | 85% |
| H.264 4K 50 Mbit/s, GOP 48 | 262 fps | 169 ms | 266 fps | 151 fps | 108 / 173 ms | 54% |
| H.264 4K ~550 Mbit/s (grain), GOP 48 | 67 fps | 677 ms | 63 fps | 28 fps | 451 / 739 ms | 0% |
| EXR 1080p half, zip1 | 126 fps | 6.6 ms | 269 fps | 269 fps | 15.8 / 17.4 ms | 100% |
| DPX 1080p 10-bit | 90 fps | 9.6 ms | 551 fps | 599 fps | 22.5 / 23.3 ms | 100% |
| PNG 1080p 8-bit | 31 fps | 31.5 ms | 203 fps | 206 fps | 45 / 46 ms | 0% (lag 1.7 fr) |

End to end through the app (`--play 5`, paced at 60 Hz, decode + upload +
GPU convert + draw, waiting on the GPU every redraw):

| clip | 1x | 4x | -1x | upload (CPU) |
|---|---|---|---|---|
| ProRes 422 HQ 4K | 24.0 pictures/s, 0 dropped | 59 pictures/s | 23.8, 1 dropped | 3.5 ms |
| H.264 4K 50 Mbit/s | 24.0, 0 dropped | 59 pictures/s | 22.6, 7 dropped | 1.1 ms |
| H.264 4K ~550 Mbit/s | 24.0, 0 dropped | 6 pictures/s | 11, 65 dropped | 1.0 ms |

No redraw went over 16.7 ms. (At 4x the source runs at 96 fps against a
60 Hz display, so some due frames are skipped by design and counted.)

## Known issues and caveats

- **Long-GOP scrubbing is still the weak spot without optimising.** A
  random H.264 frame costs a decode from its keyframe; the keyframe preview
  puts a near picture up in ~26 ms, but the exact frame takes 90-400 ms.
  Optimising is the answer for heavy scrubbing.
- **VideoToolbox frames are read back to system memory** (one copy, NV12 /
  P010, part of the decode time), not handed to Metal as IOSurfaces. A
  zero-copy path (CVPixelBuffer -> IOSurface -> MTLTexture -> wgpu through
  wgpu-hal) is possible but not done; the readback is not what limits the
  numbers above.
- A transcode decodes and encodes in one thread pipeline; the grainy
  clip's 59 fps is software H.264 decode.
- A server seek is ~5-10 ms slower than a back-to-back raw seek; likely
  FFmpeg's and the EXR reader's thread pools waking from idle. Not chased.
- EXR is treated as scene-linear and shown through a plain sRGB curve; no
  OCIO / view transform yet. The test EXRs (written by FFmpeg) hold display
  values, so they look lifted.
- Alpha is ignored in the viewer (no checkerboard yet).
- Projects: one sequence per project; no bins inside bins; the relink
  search matches by file name only; the save dialog and recovery prompt
  are not covered by a headless test (they need a person or a window).
- Organisicator: manifest and marketing folders only; its HTTP API (asset
  ids, ingest) is on an unmerged branch and not used. `post.json` is read,
  never written, so organisicator's `published` map does not know about a
  result Scrub copied (docs/organisicator.md "Still open").
- Render: decode and scale run on one thread (57 fps 4K -> 1080p); a
  parallel decode/scale pipeline is the obvious speed-up. No progress bar
  in the dialog (HUD line only).
- Reframe: static per clip (no keyframes / moves over time yet); no safe
  area guides in the viewer for platform UI overlays.
- Sound: 1x forward only; speed-changed clips are silent (not resampled);
  no scrub audio; no audio crossfade through a dissolve; track gain but no
  clip levels, pan or meters. In one headless run in four the device took no samples at
  first (the test blocks the main thread while CoreAudio starts); the stall
  watchdog now plays on silently instead of freezing the picture.
- Tools: no ripple / roll tools (B toggles ripple trims; a selected roll
  edit point rolls); the cut menu is right-click only; drops onto an empty
  lane far past the end land where dropped (a gap is left). Slip and slide
  have no four-up monitor yet (the record viewer shows the clip's head).
- Dissolve frames take 17-20 ms of CPU+GPU per redraw at 4K (two uploads):
  no drops at 24 fps, but no margin for 60p 4K dissolves yet. Uploading
  from a persistent mapped buffer, or decoding straight into GPU memory,
  is the next performance step.
- Mouse: no marquee selection, no alt-drag duplicate, no slip or slide by
  dragging yet; dissolves are video-only (centred when added by key, either
  side adjustable by dragging).
- Two servers on one file (the source viewer and a timeline clip) share
  cached frames but can decode the same frame at once.
- Cache keys are by path: a file replaced on disk under the same name shows
  its old frames until restart.
- After `brew upgrade ffmpeg` the build fails to link (the Cellar path
  moves): `cargo clean --release -p ffmpeg-sys-next`.
- FFmpeg is linked dynamically from Homebrew. A shipped build needs a
  bundled FFmpeg (LGPL, with the VideoToolbox decoders and the ProRes
  encoder); not done.
- DPX goes through FFmpeg's DPX decoder (code values, no log curve).
- Missing sequence frames are an error per frame (not cached, so a render
  arriving is picked up); the viewer keeps the last picture.

## UI batch 2026-11-13

- Plain left-drag on a viewer's picture scrubs it like RV (relative, 5 pt a
  frame, `SCRUB_PX`); pan stays MMB / alt / two fingers. The source clip
  goes to the timeline by shift-drag on the picture or the "drag clip to
  timeline" handle in its title bar (cmd at the drop still inserts).
- I / O / shift / alt I O, alt X act on the side under the pointer.
- Clear in/out is a red "× clear in/out" button in each viewer's title bar
  and on the timeline ruler, whenever marks are set.
- Key hints wrap to the viewer's width in as many rows as needed; all
  plates and caps first, the hovered row's words last (they fan down), and
  the readouts sit under however many rows there are.
- R over the timeline fits it; the razor moved to X (V X N Y U).

## Playback clock, append, key target (2026-11-13)

- **Slow motion, root cause**: the sound's clock (which the picture
  follows whenever sound plays) counted only samples actually PLAYED. When
  the feeder fell behind the device (a long file decoded at every cut, a
  file on Dropbox / network storage, a busy ring lock) the device played
  silence and the clock stood still for that time, so the picture ran slow
  instead of dropping. The clock now counts every period the device asks
  for; sound that arrives late is owed and skipped (`audio::pull`), and a
  feeder behind the clock renders from where the clock is. A playhead
  moved under playing sound (a seek while playing) now restarts the sound
  there (it used to snap back to the sound's position).
- `--test-realtime`: playhead advance against the wall clock over 5 s
  (after pre-roll) within 1% for ProRes with and without sound, H.264 25 and
  60 fps with sound, HEVC, a sequence of 60 fps cuts with sound, and the same
  with the sound feeder starved (`EDIFY_AUDIO_STARVE_MS`): all 0.997-1.000x.
  `--play` prints the same ratio ("steady: playhead 0.998x real time").
- **M** appends (`Editor::append`: an overwrite whose record in is the end).
- **Key target**: the panel the keys go to (under the pointer, else
  focused) wears a thin accent edge (`key_target`); marks use the same rule,
  in the main window and in torn-off windows (both tested).
- A narrow timeline (docked beside a viewer) no longer panics on the
  ruler's clear in/out chip.

## Renamed: Edify -> Scrub (2026-11-13)

The app is Scrub (to merge with Wobble's scrub feature later; separate
repos for now). The binary is `scrub`, the MCP bridge `scrub-mcp` (server
name "scrub"), the repo `~/src/devl/scrub`. Kept working from before:

- Projects save as `.scrub` (header key `scrub`); `.edify` files (key
  `edify`) open, and an opened `.edify` saves in place as itself. A post's
  project is `<slug>.scrub`, or an existing `<slug>.edify`.
- Settings: `~/Library/Application Support/Scrub` (prefs ui.json, recent,
  autosave, Wobble comps), COPIED once from `.../Edify` (left as it was).
  Caches: `~/Library/Caches/Scrub` (optimised copies, network frames), MOVED
  once from `.../Caches/Edify`. Clips `~/Movies/Scrub`, screenshots
  `~/Pictures/Scrub` (old Edify folders untouched).
- Optimised copies: `name.scrub.mov`; old `name.edify.mov` copies still
  found (stamp `com.scrub.source` or `com.edify.source`).
- Wobble comps: `<project>.scrub.wobble`, or an existing `.edify.wobble`.
- Env: `SCRUB_*` first, `EDIFY_*` still read (`media::sys::var`).
- Organisicator folders named Scrub or Edify, file types scrub or edify.
- Branch names `edify-embed` (seq_manip_imfram) and `edify-integration`
  (organisicator) are unchanged.
