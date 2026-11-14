# Scrub plan (draft, awaiting sign-off)

## Goal
A Premiere-style editor: fast playback and scrubbing of video files first,
image sequences second. Cross-platform Rust. Hotkey editing, not dragging.
Small feature set, performance above all. Organisikator integration later.

## Formats
- Movies through FFmpeg (`ffmpeg-next`): ProRes, H.264 / MP4.
- Sequences: EXR (`exr` crate, as Wobble), DPX, PNG, JPG.

## Reuse
- **imfram**: shell, dock, theme, hudkeys, keymap, filebrowser, timeline, numfield.
- **Wobble engine** `movie.rs`: warm FFmpeg decoder session pool (port, not depend).
- **Wobble** `decode.rs` / still readers and the UI styling.
- **Not** Wobble's edit mode (`app/src/edit/`): semi-abandoned, take nothing from it.

## Steps
1. **Skeleton and playback core.** Cargo workspace (`app`, `media`), imfram
   window with Wobble's theme, decode pipeline (FFmpeg pool plus sequence
   readers, off-thread, prefetch ahead of the playhead), GPU upload, RAM
   frame cache, one viewer with J/K/L, frame step and scrub. Benchmarks for
   4K ProRes and H.264 playback and scrub latency from day one.
2. **Two-up editing.** Source and record viewers, custom file browser and
   bin, source in/out, timeline in/out, insert and overwrite. Timeline model
   with 3 video layers and 1 audio track.
3. **Trim, timing and transitions.** Slip, cursor-based edits of clip
   in/out points, speed as a float multiplier or stretch-to-fit, dissolves on
   the GPU, audio playback in sync.
4. **Project files and Organisikator.** Save/load, then the integration
   points for Organisikator.
