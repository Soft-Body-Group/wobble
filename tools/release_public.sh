#!/bin/sh
# A macOS build to hand out: movie support, and FFmpeg's dylibs carried
# inside the bundle so it runs on a Mac without Homebrew. There is one build
# for everyone: alpha testing and the new-build check are switches the user
# answers on first run (report.rs), off until they do. Built with
# --no-default-features: the `commerce` feature (accounts, paid wobbins,
# GenAI credits — anything involving money) never ships.
#
#   tools/release_public.sh        # writes dist/public/wobble-<ver>-<git>-macos-<arch>.zip
#                                  # and the same zip as scrub-<ver>-<git>-macos-<arch>.zip
#   WOBBLE_MOVIE=0 tools/release_public.sh   # ...without movie support
#
# Scrub, the timeline editor (scrub/app), ships beside it in the same zip as
# Scrub.app, an app of its own (tools/bundle_scrub.sh) with its own copy of
# the bundled FFmpeg: it opens from Finder, the Dock or Spotlight without
# wobble, and wobble's scrub button finds it beside Wobble.app or in
# /Applications.
#
# FFmpeg is built here by tools/build_ffmpeg_lgpl.sh: LGPL only, decoders
# only, macOS 12+, as dylibs in Contents/Frameworks (dynamic, so a user can
# swap them, which LGPL asks for). The zip holds Wobble.app and Scrub.app and
# nothing else: the licences and an FFMPEG.txt naming the exact source and configure
# line ride inside each, in Contents/Resources/licenses (Blender's layout).
#
# Ad-hoc signed unless WOBBLE_SIGN names a Developer ID identity; without
# one, recipients open each the first time from System Settings → Privacy &
# Security → Open Anyway (or `xattr -cr Wobble.app Scrub.app`).
# The scripting API is always compiled in (not a feature): it opens on
# 127.0.0.1:7979 at launch unless prefs → "scripting API" turns it off.
set -e
cd "$(dirname "$0")/.."
GIT=$(git rev-parse --short=12 HEAD)
[ -n "$(git status --porcelain)" ] && GIT="$GIT-dirty"
VERSION=$(grep -m1 '^version' app/Cargo.toml | cut -d'"' -f2)
OUT=dist/public
FEATURES=app/movie
SUFFIX=
TARGET=target/public
FF=target/ffmpeg-lgpl
if [ "${WOBBLE_MOVIE:-1}" = 0 ]; then
  FEATURES=; SUFFIX=-nomovie; OUT=dist/public-nomovie
else
  sh tools/build_ffmpeg_lgpl.sh "$FF"
  FF=$(cd "$FF" && pwd)
  export FFMPEG_DIR="$FF" PKG_CONFIG_PATH="$FF/lib/pkgconfig" MACOSX_DEPLOYMENT_TARGET=12.0
  export WOBBLE_DYLIB_PATH="$FF/lib"
fi
# No build machine paths in the binary: home dir and checkout are remapped
# (panic locations, file!()), and C sources get the same through CFLAGS.
# The feedback ingest key is compiled in (app/src/report.rs): from the
# environment, else tools/feedback-worker/.secrets (gitignored, a line
# INGEST_KEY=<hex>). Without it the build cannot upload reports.
: "${WOBBLE_INGEST_KEY:=$(sed -n 's/^INGEST_KEY=//p' tools/feedback-worker/.secrets 2>/dev/null)}"
if [ -n "$WOBBLE_INGEST_KEY" ]; then export WOBBLE_INGEST_KEY
else echo "warning: no WOBBLE_INGEST_KEY; this build will not upload reports" >&2; fi
CARGO_TARGET_DIR=$TARGET \
RUSTFLAGS="--remap-path-prefix=$HOME=~ --remap-path-prefix=$PWD=wobble" \
CFLAGS="-ffile-prefix-map=$HOME=~" \
CARGO_PROFILE_RELEASE_DEBUG=line-tables-only \
CARGO_PROFILE_RELEASE_SPLIT_DEBUGINFO=packed \
  cargo build --release -p app --no-default-features ${FEATURES:+--features $FEATURES}
# Scrub with its own defaults (the clip browser) plus `fold` (clips can be
# Wobble comps, wobble hosted in process); same env, same target dir.
CARGO_TARGET_DIR=$TARGET \
RUSTFLAGS="--remap-path-prefix=$HOME=~ --remap-path-prefix=$PWD=wobble" \
CFLAGS="-ffile-prefix-map=$HOME=~" \
CARGO_PROFILE_RELEASE_DEBUG=line-tables-only \
CARGO_PROFILE_RELEASE_SPLIT_DEBUGINFO=packed \
  cargo build --release -p scrub_app --features fold
rm -rf "$OUT"
mkdir -p "$OUT"
WOBBLE_BIN=$TARGET/release/wobble WOBBLE_APP_OUT="$OUT/Wobble.app" sh tools/bundle_app.sh
SCRUB_BIN=$TARGET/release/scrub SCRUB_APP_OUT="$OUT/Scrub.app" sh tools/bundle_scrub.sh
for APP in Wobble Scrub; do
  python3 tools/bundle_dylibs.py "$OUT/$APP.app"
  # The newest minimum any bundled library declares is the app's minimum.
  MINOS=$(for f in "$OUT/$APP.app"/Contents/Frameworks/*.dylib; do
    [ -e "$f" ] && otool -l "$f" | awk '/LC_BUILD_VERSION/{b=1} b&&/minos/{print $2; b=0}'
  done | sort -V | tail -1)
  [ -n "$MINOS" ] && plutil -replace LSMinimumSystemVersion -string "$MINOS" "$OUT/$APP.app/Contents/Info.plist"
done
# Debug symbols stay here (dist/symbols/<git>/) for reading crash dumps;
# they are 3/4 of the app and no use to anyone running it.
if [ -d "$OUT/Wobble.app/Contents/MacOS/wobble.dSYM" ]; then
  mkdir -p "dist/symbols/$GIT"
  rm -rf "dist/symbols/${GIT:?}/wobble.dSYM"
  mv "$OUT/Wobble.app/Contents/MacOS/wobble.dSYM" "dist/symbols/$GIT/"
fi
if [ -d "$OUT/Scrub.app/Contents/MacOS/scrub.dSYM" ]; then
  mkdir -p "dist/symbols/$GIT"
  rm -rf "dist/symbols/${GIT:?}/scrub.dSYM"
  mv "$OUT/Scrub.app/Contents/MacOS/scrub.dSYM" "dist/symbols/$GIT/"
fi
# Every build carries the licences (LICENSING.md says what each covers),
# inside the bundle so the zip's top level is just the app; a movie build adds
# FFMPEG.txt, the exact FFmpeg it links (LGPL 2.1 §6). In before the signing,
# so the seal covers them.
for APP in Wobble Scrub; do
LIC="$OUT/$APP.app/Contents/Resources/licenses"
mkdir -p "$LIC"
cp LICENSE NOTICE.md THIRD_PARTY_LICENSES.md licenses/LGPL-2.1.txt "$LIC/"
# One flat folder, so LICENSING.md's link to licenses/LGPL-2.1.txt is a sibling.
sed 's#licenses/LGPL-2\.1\.txt#LGPL-2.1.txt#g' LICENSING.md > "$LIC/LICENSING.md"
if [ -n "$WOBBLE_DYLIB_PATH" ]; then
  cat > "$LIC/FFMPEG.txt" <<TXT
$APP uses FFmpeg (https://ffmpeg.org) under the GNU Lesser General Public
License 2.1 or later (LGPL-2.1.txt, beside this file). Only FFmpeg's LGPL
parts are built: no --enable-gpl, no --enable-nonfree, no x264, x265 or any
other external codec library. It is linked dynamically: the libraries are
$APP.app/Contents/Frameworks/libav*.dylib and libsw*.dylib, and you may
replace them with your own build of the same version.

Version: FFmpeg 8.1, unmodified
Source:  https://download.wobble.ing/ffmpeg-8.1.tar.bz2
         (identical to https://ffmpeg.org/releases/ffmpeg-8.1.tar.bz2)
SHA-256: c07039598df7d64d3c8b42c4e25b1959fc908621c6f6c2946881133f3b27eda2
Built with (macOS 12 SDK target, then make install):
  $(cat "$FF/CONFIGURE.txt")
The build script is tools/build_ffmpeg_lgpl.sh in Wobble's source.
TXT
fi
done
# The few home paths the remap misses (inlined panic locations, FFmpeg's
# configure line), blanked in place before the signature covers them.
python3 tools/scrub_paths.py "$OUT/Wobble.app" "$OUT/Scrub.app"
for APP in Wobble Scrub; do
  codesign --force --deep --sign "${WOBBLE_SIGN:--}" "$OUT/$APP.app"
  codesign --verify --deep --strict "$OUT/$APP.app"
done
NAME="wobble-$VERSION-$GIT-macos-$(uname -m)$SUFFIX.zip"
# Both apps at the zip's top level. ditto zips one source, so stage them in a
# folder and zip its contents (no --keepParent).
ZS="$OUT/zip-stage"
rm -rf "$ZS" && mkdir -p "$ZS"
ditto "$OUT/Wobble.app" "$ZS/Wobble.app"
ditto "$OUT/Scrub.app" "$ZS/Scrub.app"
ditto -c -k --norsrc --noextattr "$ZS" "$OUT/$NAME"
rm -rf "$ZS"
echo "== $OUT/$NAME"
# One build, two names: the same zip as scrub-<...>, which scrub.ing links to
# (download.scrub.ing serves scrub-*, download.wobble.ing wobble-*). Not for a
# -nomovie build: Scrub is a video editor.
if [ -z "$SUFFIX" ]; then
  SNAME="scrub-$VERSION-$GIT-macos-$(uname -m).zip"
  cp "$OUT/$NAME" "$OUT/$SNAME"
  echo "== $OUT/$SNAME"
fi
# A clean tree's release is tagged v<version>, locally; pushing it is a choice.
case "$GIT" in *-dirty) ;; *)
  if ! git rev-parse -q --verify "refs/tags/v$VERSION" >/dev/null; then
    git tag "v$VERSION" && echo "== tagged v$VERSION (local)"
  fi ;;
esac
