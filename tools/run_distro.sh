#!/bin/sh
# Run what a user downloads, as a brand-new user: the distribution's
# features (no `commerce`) and every first-launch window (welcome, testing
# opt-in, SSD cache, quick start). Real prefs are neither read nor written.
#
#   tools/run_distro.sh                 # build + launch
#   tools/run_distro.sh data/2k_zip_bg  # ...with a plate
#   WOBBLE_MOVIE=1 tools/run_distro.sh  # with movie support (system ffmpeg)
#
# Its own target dir, so target/release is left alone. Opting into testing
# here is real: reports go to the feedback worker like anyone's.
set -e
cd "$(dirname "$0")/.."
FEATURES=
[ "${WOBBLE_MOVIE:-0}" = 1 ] && FEATURES=app/movie
CARGO_TARGET_DIR=target/distro cargo build --release -p app --no-default-features ${FEATURES:+--features $FEATURES}
exec target/distro/release/wobble --first-run "$@"
