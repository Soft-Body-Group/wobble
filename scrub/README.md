# Scrub

A fast, hotkey-driven video editor in Rust. A traditional two-up editor
(source and record viewers, three-point editing) built on
[imfram](../imfram), with styling and parts taken from [Wobble](../wobble).

**Priority: playback and scrubbing speed over features.**

See [PLAN.md](PLAN.md).

Crates: `media` (decode, cache, frame server, optimise), `model` (the edit), `app` (the `scrub` binary). See NOTES.md for build, keys and numbers.

## Clone and build

Scrub builds against sibling checkouts by relative path, so clone them side
by side in one folder (any folder; `~/src/devl` here):

```
devl/
  scrub/                  this repo (main)
  imfram/                 Soft-Body-Group/imfram (main)
  seq_manip_imfram/       Soft-Body-Group/seq_manip_imfram, branch edify-embed  (the clip browser)
  sequitur/               Soft-Body-Group/sequitur (main)                  (used by seq_manip_imfram)
  sequence_manipulator/   Soft-Body-Group/sequence_manipulator, branch scrub (used by seq_manip_imfram)
  wobble/                 optional: Soft-Body-Group/wobble, for Wobble clips
```

```bash
mkdir -p ~/src/devl && cd ~/src/devl
git clone https://github.com/Soft-Body-Group/scrub.git
git clone https://github.com/Soft-Body-Group/imfram.git
git clone -b edify-embed https://github.com/Soft-Body-Group/seq_manip_imfram.git
git clone https://github.com/Soft-Body-Group/sequitur.git
git clone -b scrub https://github.com/Soft-Body-Group/sequence_manipulator.git
cd scrub && cargo build --release          # target/release/scrub, target/release/scrub-mcp
```

(Private org repos clone over https once `gh auth login` has signed in
as an account with access; `gh auth setup-git` makes git use it.)

Without the clip-browser siblings (`seq_manip_imfram`, `sequitur`,
`sequence_manipulator`): `cargo build --release -p app --no-default-features`
(the bin keeps a file dialog for importing). With Wobble clips (needs
`wobble/` beside it, built with `--features app/movie`):
`cargo build --release -p app --features wobble`.

Prerequisites:

- **Rust** stable (rustup).
- **macOS**: `brew install ffmpeg pkg-config`. This is the platform Scrub
  is developed and tested on.
- **Linux**: FFmpeg's development libraries and pkg-config, e.g. Debian /
  Ubuntu `sudo apt install pkg-config libavcodec-dev libavformat-dev
  libavutil-dev libswscale-dev libswresample-dev libavfilter-dev
  libavdevice-dev clang`, plus ALSA headers for sound (`libasound2-dev`).
  Needs FFmpeg 5.1 or newer (Ubuntu 22.04 / Pop!_OS 22.04 ship 4.4, which
  fails to link: `swr_alloc_set_opts2`). There, build FFmpeg into a prefix
  (`./configure --prefix=$HOME/.local/ffmpeg --enable-shared --enable-gpl
  --enable-libx264 && make -j && make install`) and build Scrub with
  `PKG_CONFIG_PATH=$HOME/.local/ffmpeg/lib/pkgconfig RUSTFLAGS="-L
  native=$HOME/.local/ffmpeg/lib -C link-arg=-Wl,-rpath,$HOME/.local/ffmpeg/lib"`
  (the `-L` puts the new libraries ahead of the system 4.4 ones).
  Tested on Pop!_OS 22.04 (Ryzen 9 5950X, RTX 5080, Wayland and X11).
- **Windows**: FFmpeg development libraries (headers + import libs, e.g. a
  "shared" build, with `FFMPEG_DIR` pointing at it) and LLVM/clang for the
  bindings. Untested.

Test media is not in the repo: `testdata/gen.sh` makes it (needs the ffmpeg
CLI); `testdata/colour/` (the colour chart and wobble's reference capture)
is committed.
