#!/bin/sh
# Regenerates THIRD_PARTY_LICENSES.md: every crate a public build links, with
# its licence text. Run after a dependency change; the release scripts copy
# the file (with LICENSE and LICENSING.md) into each archive.
#   cargo install cargo-about --locked --features cli
set -e
cd "$(dirname "$0")/.."
cargo about generate -m app/Cargo.toml --features "public movie" \
  about.hbs -o THIRD_PARTY_LICENSES.md
