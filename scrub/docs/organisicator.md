# Scrub and Organisicator

What Organisicator offers other apps today, what Scrub uses, and the
questions that decide the "marketing features".

Organisicator wasn't modified. Everything below was read from
`~/src/devl/organisicator` (main, 36ca38d, plus the `claude/move-in-location`
branch) and the apps that already integrate with it (wobble's
`app/src/assets.rs`, its `plugins/organisicator` addon, and Organisicator's
own `blender_addon`). `sequitur` is a library Organisicator uses internally
(sequence grouping, safe file operations). Other apps don't talk to it.

## The integration surfaces that exist

| surface | where | on main? | who uses it |
|---|---|---|---|
| **Project manifest** `<project>/.organisicator/project.json`: name, code, anatomy folders (`path`, `type`, `exts`, `hint`), shot/sequence instances (`id`, `parent`, `decl`, `value`, `rel`), all relative to the project root | `src-tauri/src/manifest.rs` | yes | Blender addon, wobble's addon |
| **Local HTTP API** on 127.0.0.1 (`~/.organisicator/api.json` = port + token; header `x-organisicator-token`): `GET /v1/projects`, `/v1/projects/{id}`, `/v1/projects/{id}/assets`, `/v1/assets/{id}`, `POST /v1/ingest` (copy media into a shot, never move or overwrite) | `src-tauri/src/api.rs` | **no** (branch `claude/move-in-location` only) | wobble (asset picker, Collect) |
| **Record target** `~/Library/Application Support/ing.wobble.organisicator/record-target.json` (Scrub also accepts any `*.organisicator` folder there) = the marketing project + `rushes` folder that apps write recordings into (`rushes/YYYY-MM-DD/`, `.partial` then rename, optional `<clip>.rush.json` sidecar) | `src-tauri/src/rushes.rs` | yes | wobble / graav screen recorders |
| **Editor rules** (front end, `localStorage` `org.editors`): `{name, app, args with "{}", exts, batch, primary}`: launch an app on a file from the folder bar | front end | yes | any app with a CLI |
| **Marketing folder**: `rushes/`, `posts/<slug>/post.json` (title, kind simple/advanced, clip in/out/speed, sound, frame 16:9/9:16/1:1/4:5, sources, distribution, published), advanced posts get `scripts/ in/ out/` ("edited in other software; renders are published from out"), `result/<slug>_v###` | `src-tauri/src/marketing.rs` | yes | Organisicator itself |

There's no URL scheme, no socket protocol, no MCP surface and no "send
to" endpoint. Instagram publishing (`instagram.rs`) is internal to
Organisicator.

## What Scrub implements (step 4)

Everything goes through the **manifest**, the one surface that's on main,
works offline, and is what the other addons read (`app/src/organisicator.rs`,
which mirrors `blender_addon/organisicator/core.py`):

- **Project awareness.** From any clip or project file, Scrub finds the
  enclosing Organisicator project and the deepest shot/sequence instance.
  It never writes into `.organisicator/`.
- **Project-aware save location.** Saving an untitled project whose media
  sits in an Organisicator shot suggests that shot's Scrub folder, with a
  versioned name: `<shot>_edit_v001.scrub`, then v002 and so on. The Scrub
  folder is, in order:
  1. a folder named `Scrub`,
  2. one whose file types include `scrub` (or `edify`),
  3. the shot's software/work folder, or its `Scrub` child.

  If the shot has none of these, Scrub walks up through the enclosing
  instances to the project. This is the Blender addon's `blender_dir` rule
  with Edify in Blender's place, so `software/Edify/` sits beside
  `software/Blender/`.
- **Browser jumps.** When the bin's browser is inside an Organisicator
  project, it shows "project / sequence / shot" plus a button for the project
  root, the shot, and each of the shot's anatomy folders (source/plates,
  renders, …).
- **Launching from Organisicator** works today with no code on either side.
  Add an editor rule in Organisicator with app = the `scrub` binary,
  args = `{}`, and exts = `scrub edify mov mp4 mxf` (plus `exr dpx png jpg` for
  sequences). `scrub <file>` opens a project or loads media into the source
  viewer.
- Tests: `organisicator::tests` (a fixture laid out like Organisicator's
  "VFX Show" demo, with its manifest as Organisicator writes it).

Not implemented, on purpose:

- **The HTTP API.** It isn't on Organisicator's main, and it wasn't
  running here (`api.json` is stale). When it lands, wobble's `assets.rs`
  shows the pattern: pick media by asset id so renames and project moves
  keep working, and `POST /v1/ingest` to copy outside media into a shot. A
  "send to Scrub" endpoint doesn't exist on either side.
- **Exports into `renders/` / `publish/` / `result/`.** Scrub has no render
  or export yet. `Project::render_dir` is ready for when it does.
- **Anything "marketing".** See the questions below.

## Marketing posts (step 5, after the answers)

What the user decided: Scrub is dual purpose, a standalone editor and part
of Organisicator, and marketing features come first. Scrub cuts ADVANCED
posts, renders them itself (YouTube- and Instagram-friendly MP4 for now),
and never publishes. Writing `post.json` isn't confirmed, so Scrub only
reads it. The `/v1` API branch may be merged later. Until then nothing
depends on it.

What Scrub does (`app/src/marketing.rs`, `app/src/render.rs`):

- **file > open marketing post…** finds the marketing folder. It looks at the
  current project or media first, then Organisicator's `record-target.json`;
  you can also pick it. It lists `posts/<slug>/post.json` read-only. Advanced
  posts open; simple ones are shown greyed ("cut in Organisicator").
- **The post's Scrub project lives beside `post.json`**:
  `posts/<slug>/<slug>.scrub`.
  - Why there: the post folder is the post's unit, and a simple post
    keeps its cut in `post.json` in that same folder.
  - `scripts/`, `in/` and `out/` keep the meaning Organisicator gives them.
    `in/` is material brought in and `out/` is what gets published.
  - So a post's edit sits where its cut would be, and `out/` holds only
    finished renders.
- **Opening a post with no project yet** starts one:
  - the sequence raster comes from the post's `frame`: 16:9 is 1920x1080,
    9:16 is 1080x1920, 1:1 is 1080x1080, 4:5 is 1080x1350, and "original"
    means the first clip decides;
  - the bin gets the post's `sources` and the files in `in/`;
  - the browser opens on `rushes/`;
  - cmd S saves it beside `post.json`.
- **file > render… (cmd M)** renders H.264 + AAC MP4. H.264 is encoded by
  VideoToolbox where present, else libx264.
  - Presets:
    - YouTube 1080p 16:9 (12 Mbit/s)
    - YouTube 4K (45 Mbit/s)
    - Instagram Reels/Stories 9:16, 1080x1920 (10 Mbit/s)
    - feed 4:5, 1080x1350 (8 Mbit/s)
    - square, 1080x1080 (8 Mbit/s)
  - Fit (bars) or fill (centred crop). The whole sequence or in..out.
  - Dissolves, speed changes and A1's sound come out exactly as playback
    shows them.
  - Inside a post, the render goes to `out/<slug>_<preset>_v001.mp4` (the
    next free version), and ONLY there. Publishing it into `result/` is
    Organisicator's step (the user's call).
  - Outside a post, a save dialog asks where.
  - Files are written under a temporary name and renamed. Nothing is ever
    overwritten. No copy goes into `rushes/`. Nothing is published.
  - The dialog warns (doesn't refuse) when the preset's shape differs from
    the post's frame shape, and about Instagram / YouTube limits.
- **Frame shape is the one field Scrub writes.** The user OK'd editing a
  post's frame shape. Sequence settings sets it, and Scrub rewrites only
  `frame` in `post.json`: other fields and key order are kept, a hidden
  `.post.json.scrub-backup` is written first, then temp file + rename.
- **Per-clip reframe** places 16:9 shots in 9:16 / 4:5 / 1:1 (fit or fill,
  zoom, move, by keys). It shows in the record viewer and renders the same.
- **Tests**: `marketing::tests` (listing, frame shapes, out/result versions,
  `post.json` byte-identical after reading), and `scrub FILE --test-post`
  (a marketing folder in a temp dir: open an advanced 9:16 post, cut, save
  beside `post.json`, render to `out/` and `result/`, check that an existing
  result and `post.json` are untouched).

## Launched from Organisicator (the integration)

Organisicator's branch `edify-integration` (worktree
`~/src/devl/organisicator-edify`, based on main 245a4b2) adds **edit in
Scrub** to advanced posts, and "open in Scrub" when making one. The
contract is written up for Organisicator's maintainer in that repo's
`docs/scrub-integration.md` (was edify-integration.md); in short:

- **Launch**: `scrub --post <marketing>/posts/<slug> --rushes <dir>...`. Each
  `--rushes` is a rushes folder of the post's project. A post folder or its
  `post.json` given as a plain argument also works.
- **Finding Scrub**, first hit wins:
  1. the user's setting in Organisicator (Scrub…);
  2. `$EDIFY_BIN`;
  3. `/Applications/Edify.app`, then `~/Applications/Edify.app`;
  4. `~/src/devl/scrub/target/release/scrub`;
  5. PATH.
- **Scrub opens or starts** `posts/<slug>/<slug>.scrub`.
- **Bins**:
  - The post's own rushes and `in/` files go first, in "post: <title>".
  - Every rushes folder is listed off the UI thread into bins mirroring its
    folders, newest first, image sequences collapsed, numbered movies kept
    separate (`bin::scan_dir`; imfram's scan folds any numbered run).
  - Past 300 items the bins open folded except the newest.
- **Back**: renders go to `out/` as a hidden `.<name>.partial`, renamed when
  done. Organisicator's posts watcher picks them up and lists them on the
  post. Publishing into `result/` stays in Organisicator.
- **Writes**: the project, renders in `out/`, and `post.json`'s `frame` only.

## Still open

Answered by the user:
- **Writing `post.json`**: only the frame shape.
- **Publishing**: leave it to Organisicator.
- **Frame mismatch**: warn.
- **Platform limits**: warn.

Still open:
1. **The API.** When `claude/move-in-location` is merged, Scrub can add
   asset-id references and ingest. It's optional, with plain paths as the
   fallback; nothing is built yet.
2. **Exports into `rushes/`** weren't answered, so Scrub doesn't do them.

## The questions as first asked (answered above)


1. **Is Scrub meant to make marketing posts?** For example: an "advanced
   post" opened in Scrub, cut from `rushes/`, with renders written to the
   post's `out/` and then published to `result/<slug>_v###`. Or is Scrub for
   shot/editorial work, with marketing staying inside Organisicator's simple
   posts?
2. **Where should an Scrub project live in a post?** In `posts/<slug>/`
   next to `post.json` (and should `post.json` point at it), or in the
   post's `scripts/`?
3. **Should Scrub read and write `post.json`'s `clip {rush, in, out, speed}`
   and `frame`** (16:9, 9:16, 1:1, 4:5)? Scrub could open a simple post as a
   one-clip sequence at that raster, and write in/out/speed back. That makes
   simple posts editable in Scrub, but it means writing into a file
   Organisicator owns.
4. **Exports.** Which formats and sizes per destination (YouTube H.264/AAC
   as `render.rs` does now, vertical 9:16 for Reels/TikTok, ProRes masters)?
   Should Scrub render them itself, or hand a cut list to Organisicator's
   ffmpeg renderer?
5. **Publishing.** `distribution`/`published` are written by
   Organisicator's social integrations. Should Scrub only ever produce files,
   or should it trigger publishing (which needs a new endpoint in
   Organisicator)?
6. **Recordings.** Should Scrub honour `record-target.json` (for example,
   exports of a marketing cut landing in `rushes/` with a `.rush.json`
   sidecar), as wobble's recorder does?
7. **The API.** Will `claude/move-in-location` (the `/v1` API) be merged?
   Asset-id references and ingest only make sense once it is.
