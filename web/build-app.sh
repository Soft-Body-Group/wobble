#!/bin/sh
# Build the FULL wobble app for the browser into web/www/app, then serve
# web/www (python3 -m http.server 8765 --directory web/www) and open /app/.
# WOBBLE_WASM_OPT=1 ./web/build-app.sh for a smaller, publishable module.
# Needs: rustup target add wasm32-unknown-unknown, and wasm-bindgen-cli at
# the version Cargo.lock pins (wasm-pack installs one; this finds it).
set -e
cd "$(dirname "$0")/.."
# No DWARF in the page: the workspace's release profile keeps debug info.
CARGO_PROFILE_RELEASE_DEBUG=false \
RUSTFLAGS="--remap-path-prefix=$HOME=~ --remap-path-prefix=$PWD=wobble" \
    cargo build --release -p app --bin wobble --target wasm32-unknown-unknown "$@"
VER=$(grep -A1 '^name = "wasm-bindgen"$' Cargo.lock | sed -n 's/version = "\(.*\)"/\1/p')
BG=$(command -v wasm-bindgen || true)
if [ -z "$BG" ] || ! "$BG" --version | grep -q "$VER"; then
    BG="$HOME/Library/Caches/.wasm-pack/wasm-bindgen-cargo-install-$VER/wasm-bindgen"
fi
[ -x "$BG" ] || { echo "need wasm-bindgen $VER: cargo install wasm-bindgen-cli --version $VER"; exit 1; }
"$BG" --target web --no-typescript --out-dir web/www/app \
    target/wasm32-unknown-unknown/release/wobble.wasm
# WOBBLE_WASM_OPT=1: shrink for publishing (~22 -> ~16 MB, ~40 s). Uses a
# wasm-opt on PATH, else the one wasm-pack caches.
if [ -n "$WOBBLE_WASM_OPT" ]; then
    OPT=$(command -v wasm-opt || ls "$HOME"/Library/Caches/.wasm-pack/wasm-opt-*/bin/wasm-opt 2>/dev/null | head -1)
    [ -x "$OPT" ] || { echo "WOBBLE_WASM_OPT set but no wasm-opt found"; exit 1; }
    "$OPT" -O2 --enable-bulk-memory --enable-nontrapping-float-to-int --enable-sign-ext \
        --enable-mutable-globals --enable-reference-types --enable-multivalue \
        web/www/app/wobble_bg.wasm -o web/www/app/wobble_bg.wasm
fi
ls -la web/www/app/wobble_bg.wasm
# No build machine paths in the published page.
python3 tools/scrub_paths.py web/www/app/wobble_bg.wasm
