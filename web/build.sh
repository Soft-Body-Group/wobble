#!/bin/sh
# Build the engine-only page (web/www/core) into web/www/pkg. Serve web/www:
#   python3 -m http.server 8765 --directory web/www
# Needs: rustup target add wasm32-unknown-unknown, and wasm-pack.
set -e
cd "$(dirname "$0")"
wasm-pack build --release --target web --out-dir www/pkg --no-typescript "$@"

