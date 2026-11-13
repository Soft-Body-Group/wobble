#!/bin/sh
# Build and publish an alpha by hand: zip → R2 → the builds table. Testers
# download it from /get with their invite code; running copies show a
# "new build" button on their next launch.
#
# The usual way is CI, not this script: push an `alpha-*` tag and
# .github/workflows/alpha.yml builds macOS + Windows, crash-tests both and
# publishes them (see tools/feedback-worker/README.md). This is the by-hand
# path and the packager CI calls.
#
#   tools/release_alpha.sh "notes"                  # macOS: build here, publish
#   tools/release_alpha.sh --dry-run "notes"        # macOS: build + zip only
#   tools/release_alpha.sh --package-only DIR       # macOS: build + zip + build.json
#                                                   #   into DIR, publish nothing (CI)
#   tools/release_alpha.sh --windows "notes"        # Windows: publish the newest
#                                                   #   successful alpha.yml run's build
#   tools/release_alpha.sh --windows --run ID "notes"   # ...or that run's
#
# Windows builds always come from CI: built, crash-tested and zipped with
# wobble.pdb on a real Windows runner.
#
# macOS builds happen in their own target dir (target/alpha) with
# line-tables debug info + a packed dSYM, so tester panics carry file:line
# without shipping full debug info, and target/release is never disturbed.
#
# Symbols (dSYM / PDB) go to R2 (symbols/) with the build, and a copy is kept
# in dist/symbols/<git>-<stamp>/, for symbolicating crash records later.
#
# A macOS build refuses a dirty tree: a report must name a build someone can
# check out. WOBBLE_ALLOW_DIRTY=1 overrides (the build then says "-dirty").
set -e
cd "$(dirname "$0")/.."
ROOT=$(pwd)
DRY=0
WINDOWS=0
PKG=
RUN=
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --windows) WINDOWS=1 ;;
    --package-only) shift; PKG=${1:?--package-only needs a directory} ;;
    --run) shift; RUN=${1:?--run needs a run id} ;;
    *) break ;;
  esac
  shift
done
NOTES=${1:-}

# publish DIR — hand DIR's build.json + zip to the one publisher.
publish() {
  if [ "$DRY" = 1 ]; then
    echo "dry run: nothing uploaded"
    return
  fi
  sh "$ROOT/tools/publish_build.sh" "$1" "$NOTES"
}

if [ "$WINDOWS" = 1 ]; then
  command -v gh >/dev/null || { echo "release_alpha: needs the gh CLI" >&2; exit 1; }
  if [ -z "$RUN" ]; then
    RUN=$(gh run list --workflow alpha.yml --status success -L 1 --json databaseId -q '.[0].databaseId')
    [ -n "$RUN" ] || { echo "release_alpha: no successful alpha.yml run — gh workflow run alpha.yml" >&2; exit 1; }
  fi
  OUT=dist/alpha-windows
  rm -rf "$OUT"
  echo "== downloading run $RUN"
  gh run download "$RUN" -n alpha-windows-x86_64 -D "$OUT"
  field() { sed -n "s/.*\"$1\":\"\{0,1\}\([^,\"}]*\).*/\1/p" "$OUT/build.json"; }
  GIT=$(field git) STAMP=$(field stamp)
  mkdir -p "dist/symbols/$GIT-$STAMP"
  unzip -o -j -q "$OUT/$(field file)" "wobble/wobble.pdb" -d "dist/symbols/$GIT-$STAMP" || true
  publish "$OUT"
  exit 0
fi

if [ -n "$(git status --porcelain)" ] && [ -z "$WOBBLE_ALLOW_DIRTY" ]; then
  echo "release_alpha: working tree is dirty — commit first, or WOBBLE_ALLOW_DIRTY=1" >&2
  exit 1
fi
case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) PLATFORM=macos-aarch64 ;;
  Darwin-x86_64) PLATFORM=macos-x86_64 ;;
  *) echo "release_alpha: builds are packaged on macOS here; Windows comes from CI (--windows)" >&2; exit 1 ;;
esac

GIT=$(git rev-parse --short=12 HEAD)
[ -n "$(git status --porcelain)" ] && GIT="$GIT-dirty"
# CI hands the stamp in, so every platform of one release shares it.
STAMP=${WOBBLE_BUILD_TS:-$(date +%s)}
VERSION=$(grep -m1 '^version' app/Cargo.toml | cut -d'"' -f2)
BASE="wobble-alpha-$VERSION-$GIT-$STAMP-$PLATFORM"
NAME="$BASE.zip"
SYMS="$BASE-symbols.zip"
OUT=${PKG:-dist/alpha}

echo "== building $GIT ($PLATFORM), stamp $STAMP"
export WOBBLE_BUILD_TS=$STAMP
# The feedback ingest key is compiled in (app/src/report.rs): from the
# environment, else tools/feedback-worker/.secrets (gitignored, a line
# INGEST_KEY=<hex>). Without it the build cannot upload reports.
: "${WOBBLE_INGEST_KEY:=$(sed -n 's/^INGEST_KEY=//p' tools/feedback-worker/.secrets 2>/dev/null)}"
if [ -n "$WOBBLE_INGEST_KEY" ]; then export WOBBLE_INGEST_KEY
else echo "warning: no WOBBLE_INGEST_KEY; this build will not upload reports" >&2; fi
CARGO_TARGET_DIR=target/alpha \
CARGO_PROFILE_RELEASE_DEBUG=line-tables-only \
CARGO_PROFILE_RELEASE_SPLIT_DEBUGINFO=packed \
  cargo build --release -p app

echo "== crash capture smoke test"
sh tools/crash_smoke.sh target/alpha/release/wobble

rm -rf "$OUT"
mkdir -p "$OUT"
WOBBLE_BIN=target/alpha/release/wobble WOBBLE_APP_OUT="$OUT/Wobble.app" sh tools/bundle_app.sh
# --strict is what Gatekeeper applies on a downloaded copy; without it a
# broken bundle (e.g. a dangling symlink) verifies here and is "damaged"
# on the tester's Mac.
if ! codesign --verify --deep --strict "$OUT/Wobble.app" 2>/dev/null; then
  echo "release_alpha: the bundle's ad-hoc signature does not verify" >&2
  exit 1
fi
ditto -c -k --keepParent "$OUT/Wobble.app" "$OUT/$NAME"
rm -rf "$OUT/Wobble.app"

# Symbols, zipped for R2. -L: cargo's wobble.dSYM is a symlink into deps/.
SYMS_FIELD=
if [ -d target/alpha/release/wobble.dSYM ]; then
  mkdir -p "$OUT/syms"
  cp -RL target/alpha/release/wobble.dSYM "$OUT/syms/"
  ditto -c -k "$OUT/syms" "$OUT/$SYMS"
  rm -rf "$OUT/syms"
  SYMS_FIELD=",\"symbols\":\"$SYMS\""
  mkdir -p "dist/symbols/$GIT-$STAMP"
  cp -RL target/alpha/release/wobble.dSYM "dist/symbols/$GIT-$STAMP/"
fi
printf '{"git":"%s","stamp":%s,"version":"%s","platform":"%s","file":"%s"%s}\n' \
  "$GIT" "$STAMP" "$VERSION" "$PLATFORM" "$NAME" "$SYMS_FIELD" > "$OUT/build.json"
echo "== $NAME packaged in $OUT"

[ -n "$PKG" ] && exit 0
publish "$OUT"
