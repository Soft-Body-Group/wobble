#!/bin/sh
# Publish one packaged alpha build: its zip → R2, a row in the builds table,
# and its symbols → R2 for symbolicating crash reports later. The one place
# that does this, so a build published by hand (tools/release_alpha.sh) and
# one published by CI (.github/workflows/alpha.yml) cannot drift.
#
#   tools/publish_build.sh DIR "notes for testers"
#
# DIR holds build.json — {"git","stamp","version","platform","file"[,"symbols"]},
# flat — next to the zip it names (and the symbols zip, when there is one).
#
# Needs wrangler credentials: a `wrangler login` here, or in CI the
# CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID environment.
# WRANGLER overrides how wrangler is run (CI pins a major: "npx --yes wrangler@4").
set -e
DIR=${1:?usage: publish_build.sh DIR "notes"}
NOTES=${2:-}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DIR=$(cd "$DIR" && pwd)
WRANGLER=${WRANGLER:-npx wrangler}

[ -f "$DIR/build.json" ] || { echo "publish_build: no build.json in $DIR" >&2; exit 1; }
# Flat JSON written by our own packaging, so sed will do (no jq on a stock Mac).
field() { sed -n "s/.*\"$1\":\"\{0,1\}\([^,\"}]*\).*/\1/p" "$DIR/build.json"; }
GIT=$(field git) STAMP=$(field stamp) VERSION=$(field version)
PLATFORM=$(field platform) FILE=$(field file) SYMBOLS=$(field symbols)
for v in GIT STAMP VERSION PLATFORM FILE; do
  eval "[ -n \"\$$v\" ]" || { echo "publish_build: build.json has no $v" >&2; exit 1; }
done
case "$STAMP" in *[!0-9]*) echo "publish_build: stamp '$STAMP' is not a number" >&2; exit 1 ;; esac
ZIP="$DIR/$FILE"
[ -f "$ZIP" ] || { echo "publish_build: $ZIP missing" >&2; exit 1; }
SIZE=$(wc -c <"$ZIP" | tr -d ' ')
echo "== $FILE ($((SIZE / 1048576)) MB)"

cd "$ROOT/tools/feedback-worker"
if [ -n "$SYMBOLS" ] && [ -f "$DIR/$SYMBOLS" ]; then
  echo "== uploading symbols"
  $WRANGLER r2 object put "wobble-builds/symbols/$SYMBOLS" --file "$DIR/$SYMBOLS" \
    --content-type application/zip --remote
fi
echo "== uploading to R2"
$WRANGLER r2 object put "wobble-builds/alpha/$FILE" --file "$ZIP" \
  --content-type application/zip --remote
# SQL string literals: double any single quote. The other fields come from
# our own packaging, but a stray quote in one must fail, not inject.
q() { printf "%s" "$1" | sed "s/'/''/g"; }
echo "== publishing"
$WRANGLER d1 execute wobble-feedback --remote --command \
  "INSERT INTO builds (git, stamp, version, platform, file, size, notes, created) VALUES \
   ('$(q "$GIT")', $STAMP, '$(q "$VERSION")', '$(q "$PLATFORM")', 'alpha/$(q "$FILE")', $SIZE, \
    '$(q "$NOTES")', $(date +%s)000)"
echo "published $FILE — testers get it from https://feedback.wobble.ing/get"
