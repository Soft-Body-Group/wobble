#!/bin/sh
# The Linux twin of build_ffmpeg_lgpl.sh: the same FFmpeg 8.1 tarball and the
# same LGPL-only, decode-only configure, built into shared libraries
# (lib/libav*.so.N) that the movie build links and the release carries in
# lib/ beside the binary. Each library finds its siblings through
# RUNPATH=$ORIGIN, so the folder can move anywhere.
#
#   tools/build_ffmpeg_lgpl_linux.sh [PREFIX]   # default target/ffmpeg-lgpl
#
# Needs nasm, plus zlib1g-dev and libbz2-dev. Differences from the macOS
# configure: no VideoToolbox/AudioToolbox (macOS only).
set -e
cd "$(dirname "$0")/.."
VER=8.1
SHA=c07039598df7d64d3c8b42c4e25b1959fc908621c6f6c2946881133f3b27eda2
PREFIX=${1:-target/ffmpeg-lgpl}
mkdir -p "$PREFIX"
PREFIX=$(cd "$PREFIX" && pwd)
TAR="$PREFIX/ffmpeg-$VER.tar.bz2"
[ -f "$PREFIX/lib/libavcodec.so" ] && [ -f "$TAR" ] && { echo "ffmpeg $VER (LGPL) in $PREFIX"; exit 0; }
[ -f "$TAR" ] || curl -sSfL -o "$TAR" "https://ffmpeg.org/releases/ffmpeg-$VER.tar.bz2"
echo "$SHA  $TAR" | sha256sum -c - >/dev/null
SRC="$PREFIX/src"
rm -rf "$SRC"; mkdir -p "$SRC"
tar xjf "$TAR" -C "$SRC"
cd "$SRC/ffmpeg-$VER"
ARGS="--enable-shared --disable-static --disable-programs --disable-doc \
--disable-network --disable-avdevice --disable-avfilter --disable-encoders \
--disable-muxers --disable-autodetect --enable-zlib --enable-bzlib"
# configure evals the flag once and make expands LDFLAGS twice (through
# library.mak's $(eval)); this lands in config.mak as '$$$$ORIGIN', which
# comes out as a literal RUNPATH=$ORIGIN on every library.
LDF="--extra-ldflags=-Wl,-rpath,'\\\$\\\$\\\$\\\$ORIGIN'"
# shellcheck disable=SC2086
# The prefix map keeps the build directory (a home path) out of asserts.
./configure --prefix="$PREFIX" $ARGS "$LDF" "--extra-cflags=-ffile-prefix-map=$PREFIX/src=" > "$PREFIX/configure.log"
echo "./configure $ARGS --extra-ldflags=-Wl,-rpath,'\$ORIGIN'" > "$PREFIX/CONFIGURE.txt"
grep -q '^License: LGPL' "$PREFIX/configure.log" || { echo "build_ffmpeg_lgpl_linux: configure did not report an LGPL licence" >&2; exit 1; }
make -j"$(nproc)" >/dev/null
make install >/dev/null
cp COPYING.LGPLv2.1 "$PREFIX/COPYING.LGPLv2.1"
echo "ffmpeg $VER (LGPL) in $PREFIX"
