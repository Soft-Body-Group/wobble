#!/bin/sh
# The Linux twin of release_public.sh: one build for everyone, built with
# --no-default-features (the `commerce` feature never ships), movie support
# through an LGPL-only FFmpeg whose shared libraries sit in lib/ beside the
# binary (dynamic, so a user can swap them, which LGPL asks for). The binary
# carries RUNPATH=$ORIGIN/lib and each library RUNPATH=$ORIGIN, so the folder
# runs from anywhere with no LD_LIBRARY_PATH. The licences (and FFMPEG.txt)
# sit in licenses/, so the top level is the app and what it needs to run.
# Scrub, the timeline editor (scrub/app), ships beside it as ./scrub, on the
# same lib/ (wobble's scrub button starts the binary beside it).
#
#   tools/release_public_linux.sh          # dist/public-linux/wobble-<ver>-<git>-linux-x86_64.tar.gz
#   WOBBLE_MOVIE=0 tools/release_public_linux.sh   # ...-nomovie, no FFmpeg
#
# FFmpeg comes from tools/build_ffmpeg_lgpl_linux.sh; bindgen needs libclang
# (apt: clang libclang-dev).
# The scripting API is always compiled in (not a feature): it opens on
# 127.0.0.1:7979 at launch unless prefs → "scripting API" turns it off.
set -e
cd "$(dirname "$0")/.."
GIT=$(git rev-parse --short=12 HEAD)
[ -n "$(git status --porcelain --untracked-files=no)" ] && GIT="$GIT-dirty"
VERSION=$(grep -m1 '^version' app/Cargo.toml | cut -d'"' -f2)
ARCH=$(uname -m)
SUFFIX=
FEATURES=
OUT=dist/public-linux
FF=target/ffmpeg-lgpl
if [ "${WOBBLE_MOVIE:-1}" = 0 ]; then
  SUFFIX=-nomovie; OUT=dist/public-linux-nomovie
else
  sh tools/build_ffmpeg_lgpl_linux.sh "$FF"
  FF=$(cd "$FF" && pwd)
  export FFMPEG_DIR="$FF" PKG_CONFIG_PATH="$FF/lib/pkgconfig"
  FEATURES=app/movie
fi
NAME="wobble-$VERSION-$GIT-linux-$ARCH$SUFFIX"
# No build machine paths in the binary: home dir and checkout are remapped
# (panic locations, file!()), and C sources get the same through CFLAGS.
# The feedback ingest key is compiled in (app/src/report.rs): from the
# environment, else tools/feedback-worker/.secrets (gitignored, a line
# INGEST_KEY=<hex>). Without it the build cannot upload reports.
: "${WOBBLE_INGEST_KEY:=$(sed -n 's/^INGEST_KEY=//p' tools/feedback-worker/.secrets 2>/dev/null)}"
if [ -n "$WOBBLE_INGEST_KEY" ]; then export WOBBLE_INGEST_KEY
else echo "warning: no WOBBLE_INGEST_KEY; this build will not upload reports" >&2; fi
CARGO_TARGET_DIR=target/public \
CARGO_PROFILE_RELEASE_DEBUG=false \
RUSTFLAGS='-C link-arg=-Wl,-rpath,$ORIGIN/lib'" --remap-path-prefix=$HOME=~ --remap-path-prefix=$PWD=wobble" \
CFLAGS="-ffile-prefix-map=$HOME=~" \
  cargo build --release -p app --no-default-features ${FEATURES:+--features $FEATURES}
# Scrub with its own defaults (the clip browser); same env, same target dir.
CARGO_TARGET_DIR=target/public \
CARGO_PROFILE_RELEASE_DEBUG=false \
RUSTFLAGS='-C link-arg=-Wl,-rpath,$ORIGIN/lib'" --remap-path-prefix=$HOME=~ --remap-path-prefix=$PWD=wobble" \
CFLAGS="-ffile-prefix-map=$HOME=~" \
  cargo build --release -p scrub_app

rm -rf "$OUT"
ST="$OUT/$NAME"
mkdir -p "$ST/licenses" "$ST/plugins"
cp target/public/release/wobble target/public/release/scrub "$ST/"
strip "$ST/wobble" "$ST/scrub"
cp LICENSE NOTICE.md THIRD_PARTY_LICENSES.md licenses/LGPL-2.1.txt "$ST/licenses/"
# One flat folder, so LICENSING.md's link to licenses/LGPL-2.1.txt is a sibling.
sed 's#licenses/LGPL-2\.1\.txt#LGPL-2.1.txt#g' LICENSING.md > "$ST/licenses/LICENSING.md"
cp -r plugins/invert plugins/vignette "$ST/plugins/"
mkdir -p "$ST/icons"
cp app/icons/wobble-256.png app/icons/wobble-512.png "$ST/icons/"
cp app/icons/install-desktop-entry.sh "$ST/"
MOVIE_LINE="This build has no movie (ProRes/H.264) support: read EXR, DPX, PNG or JPEG sequences."
if [ -n "$FEATURES" ]; then
  mkdir -p "$ST/lib"
  # The soname links are what the loader asks for; copy them as real files.
  for l in "$FF"/lib/lib*.so.[0-9]*; do
    case "$l" in *.so.[0-9]*.[0-9]*) continue ;; esac
    cp -L "$l" "$ST/lib/"
  done
  MOVIE_LINE="Movies (ProRes, H.264 and friends) read through FFmpeg; see licenses/FFMPEG.txt."
  cat > "$ST/licenses/FFMPEG.txt" <<TXT
Wobble uses FFmpeg (https://ffmpeg.org) under the GNU Lesser General Public
License 2.1 or later (LGPL-2.1.txt, beside this file). Only FFmpeg's LGPL
parts are built: no --enable-gpl, no --enable-nonfree, no x264, x265 or any
other external codec library. It is linked dynamically: the libraries are
lib/libav*.so.* and lib/libsw*.so.* beside the wobble binary, and you may
replace them with your own build of the same version.

Version: FFmpeg 8.1, unmodified
Source:  https://download.wobble.ing/ffmpeg-8.1.tar.bz2
         (identical to https://ffmpeg.org/releases/ffmpeg-8.1.tar.bz2)
SHA-256: c07039598df7d64d3c8b42c4e25b1959fc908621c6f6c2946881133f3b27eda2
Built with GCC on Linux x86_64, then make install:
  $(cat "$FF/CONFIGURE.txt")
The build script is tools/build_ffmpeg_lgpl_linux.sh in Wobble's source.
TXT
fi
cat > "$ST/README.txt" <<TXT
Wobble $VERSION (Linux, $ARCH)

Unpack anywhere, cd into the folder and run ./wobble (the bundled plugins load
from ./plugins, relative to where you start it). Needs a GPU with Vulkan (NVIDIA's
proprietary driver, or Mesa for AMD and Intel).
$MOVIE_LINE
Plugins in the plugins folder load automatically; your own go in ~/.wobble/plugins.
./install-desktop-entry.sh adds it to your app launcher, with its icon.
./scrub is Scrub, the timeline editor (also started by wobble's scrub button).

The first launch asks whether to join alpha testing and to be told about new
builds; nothing is sent until you answer. Change either in prefs any time.

Wobble is free software under the GPL-3.0-or-later; see the licenses folder
(LICENSING.md says what each file covers).
Built from $GIT.
TXT
python3 tools/scrub_paths.py "$ST"
# Owned by root in the archive, not by whoever built it.
tar --owner=0 --group=0 --numeric-owner -C "$OUT" -czf "$OUT/$NAME.tar.gz" "$NAME"
echo "== $OUT/$NAME.tar.gz"
