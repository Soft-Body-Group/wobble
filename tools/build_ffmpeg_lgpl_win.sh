#!/bin/sh
# The Windows twin of build_ffmpeg_lgpl.sh: the same FFmpeg 8.1 tarball and
# the same LGPL-only, decode-only configure, built with MSVC into DLLs
# (bin/av*.dll + lib/av*.lib import libraries) that the movie build links
# and the zip carries beside wobble.exe.
#
#   tools/build_ffmpeg_lgpl_win.cmd [PREFIX]   # default target/ffmpeg-lgpl
#
# Run through the .cmd, which sets up the MSVC environment and runs this in
# MSYS2's bash (make, nasm, diffutils, pkgconf from pacman). Differences
# from the macOS configure: no VideoToolbox/AudioToolbox (macOS only) and no
# zlib/bzlib (Windows has no system copy; nothing Wobble reads needs them).
set -e
cd "$(dirname "$0")/.."
VER=8.1
SHA=c07039598df7d64d3c8b42c4e25b1959fc908621c6f6c2946881133f3b27eda2
PREFIX=${1:-target/ffmpeg-lgpl}
mkdir -p "$PREFIX"
PREFIX=$(cd "$PREFIX" && pwd)
TAR="$PREFIX/ffmpeg-$VER.tar.bz2"
[ -f "$PREFIX/lib/avcodec.lib" ] && [ -f "$PREFIX/bin/avcodec-62.dll" ] && [ -f "$TAR" ] && { echo "ffmpeg $VER (LGPL) in $PREFIX"; exit 0; }
[ -f "$TAR" ] || curl -sSfL -o "$TAR" "https://ffmpeg.org/releases/ffmpeg-$VER.tar.bz2"
echo "$SHA  $TAR" | sha256sum -c - >/dev/null
SRC="$PREFIX/src"
rm -rf "$SRC"; mkdir -p "$SRC"
tar xjf "$TAR" -C "$SRC"
cd "$SRC/ffmpeg-$VER"
ARGS="--toolchain=msvc --target-os=win64 --arch=x86_64 \
--enable-shared --disable-static --disable-programs --disable-doc \
--disable-network --disable-avdevice --disable-avfilter --disable-encoders \
--disable-muxers --disable-autodetect"
# shellcheck disable=SC2086
./configure --prefix="$PREFIX" $ARGS > "$PREFIX/configure.log"
echo "./configure $ARGS" > "$PREFIX/CONFIGURE.txt"
grep -q '^License: LGPL' "$PREFIX/configure.log" || { echo "build_ffmpeg_lgpl_win: configure did not report an LGPL licence" >&2; exit 1; }
make -j"$(nproc)" >/dev/null
make install >/dev/null
# MSVC's import libraries install beside the DLLs; ffmpeg-sys-next links
# from lib/.
mv "$PREFIX"/bin/*.lib "$PREFIX/lib/"
cp COPYING.LGPLv2.1 "$PREFIX/COPYING.LGPLv2.1"
rm -rf "$SRC"
echo "ffmpeg $VER (LGPL) in $PREFIX"
