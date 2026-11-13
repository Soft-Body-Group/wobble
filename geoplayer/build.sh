#!/bin/sh
# Build the live-export player module into geoplayer/geoplayer.wasm, which
# the app embeds (include_bytes!) for "export web page" with play forever.
# Re-run after changing anything the 2dini evaluator touches, then rebuild
# the app. Needs: rustup target add wasm32-unknown-unknown. Uses a wasm-opt
# on PATH (or wasm-pack's cached one) to shrink it when found.
set -e
cd "$(dirname "$0")/.."
CARGO_PROFILE_RELEASE_DEBUG=false \
    cargo build --release -p geoplayer --target wasm32-unknown-unknown
# wasm-bindgen strips the describe exports wgpu's web bindings drag in
# (~2.3 -> 1.9 MB); the page stubs the two imports it leaves, which the
# evaluator never calls. Same pinned version web/build-app.sh finds.
VER=$(grep -A1 '^name = "wasm-bindgen"$' Cargo.lock | sed -n 's/version = "\(.*\)"/\1/p')
BG=$(command -v wasm-bindgen || true)
if [ -z "$BG" ] || ! "$BG" --version | grep -q "$VER"; then
    BG="$HOME/Library/Caches/.wasm-pack/wasm-bindgen-cargo-install-$VER/wasm-bindgen"
fi
[ -x "$BG" ] || { echo "need wasm-bindgen $VER: cargo install wasm-bindgen-cli --version $VER"; exit 1; }
OUT=$(mktemp -d)
"$BG" --target web --no-typescript --out-dir "$OUT" target/wasm32-unknown-unknown/release/geoplayer.wasm
OPT=$(command -v wasm-opt || ls "$HOME"/Library/Caches/.wasm-pack/wasm-opt-*/bin/wasm-opt 2>/dev/null | head -1)
if [ -x "$OPT" ]; then
    "$OPT" -Oz --enable-bulk-memory --enable-nontrapping-float-to-int --enable-sign-ext \
        --enable-mutable-globals --enable-reference-types --enable-multivalue \
        "$OUT/geoplayer_bg.wasm" -o geoplayer/geoplayer.wasm
else
    cp "$OUT/geoplayer_bg.wasm" geoplayer/geoplayer.wasm
fi
rm -rf "$OUT"
ls -la geoplayer/geoplayer.wasm
# It is committed and ships inside every build: no build machine paths.
python3 tools/scrub_paths.py geoplayer/geoplayer.wasm
