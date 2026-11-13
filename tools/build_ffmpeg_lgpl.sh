#!/bin/sh
# Build the FFmpeg a public macOS release ships: LGPL only (no --enable-gpl,
# no x264/x265 or any external codec library), decode side only (no
# encoders, muxers, filters, devices or network), shared libraries with
# @rpath install names, for macOS 12+. Read nodes only decode; the movie
# encodes in edit/cache.rs run a SYSTEM ffmpeg binary, not these libraries.
#
#   tools/build_ffmpeg_lgpl.sh [PREFIX]   # default target/ffmpeg-lgpl
#
# The source is the release tarball, checked against the same SHA-256
# Blender pins. The tarball is kept beside the prefix: it is the source a
# release must offer (LGPL 2.1 section 6), and release_public.sh points to it.
set -e
cd "$(dirname "$0")/.."
VER=8.1
SHA=c07039598df7d64d3c8b42c4e25b1959fc908621c6f6c2946881133f3b27eda2
PREFIX=${1:-target/ffmpeg-lgpl}
mkdir -p "$PREFIX"
PREFIX=$(cd "$PREFIX" && pwd)
TAR="$PREFIX/ffmpeg-$VER.tar.bz2"
[ -f "$PREFIX/lib/libavcodec.dylib" ] && [ -f "$TAR" ] && { echo "ffmpeg $VER (LGPL) in $PREFIX"; exit 0; }
[ -f "$TAR" ] || curl -sSfL -o "$TAR" "https://ffmpeg.org/releases/ffmpeg-$VER.tar.bz2"
echo "$SHA  $TAR" | shasum -a 256 -c - >/dev/null
SRC="$PREFIX/src"
rm -rf "$SRC"; mkdir -p "$SRC"
tar xjf "$TAR" -C "$SRC"
cd "$SRC/ffmpeg-$VER"
export MACOSX_DEPLOYMENT_TARGET=12.0
# One argument per line: the flags carry spaces, which a flat string split
# on whitespace would cut in two. The prefix map keeps the build directory
# (a home path) out of the dylibs' assert strings.
set -- --enable-shared --disable-static --disable-programs --disable-doc \
  --disable-network --disable-avdevice --disable-avfilter --disable-encoders \
  --disable-muxers --disable-autodetect --enable-videotoolbox \
  --enable-audiotoolbox --enable-zlib --enable-bzlib \
  "--extra-cflags=-mmacosx-version-min=12.0 -ffile-prefix-map=$PREFIX/src=" \
  "--extra-ldflags=-mmacosx-version-min=12.0 -Wl,-headerpad_max_install_names" \
  --install-name-dir=@rpath
ARGS="$*"
./configure --prefix="$PREFIX" "$@" > "$PREFIX/configure.log"
echo "./configure $ARGS" > "$PREFIX/CONFIGURE.txt"
grep -q '^License: LGPL' "$PREFIX/configure.log" || { echo "build_ffmpeg_lgpl: configure did not report an LGPL licence" >&2; exit 1; }
make -j"$(sysctl -n hw.ncpu)" >/dev/null
make install >/dev/null
cp COPYING.LGPLv2.1 "$PREFIX/COPYING.LGPLv2.1"
echo "ffmpeg $VER (LGPL) in $PREFIX"
