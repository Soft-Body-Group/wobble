#!/bin/sh
# Build, then run ONLY if the build succeeded — otherwise the old binary
# starts and a compile error looks like "my change didn't land".
# Movies (ProRes/H.264) via a system FFmpeg are on; WOBBLE_NO_MOVIE=1 builds
# without them.
set -e
if [ -n "$WOBBLE_NO_MOVIE" ]; then
  cargo build --release
else
  cargo build --release --features app/movie
fi
exec ./target/release/wobble "$@"
