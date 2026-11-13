#!/bin/sh
# Bump THE version (app/Cargo.toml) and the lockfile with it.
#
#   tools/bump_version.sh             # 0.1.0-alpha.1 -> 0.1.0-alpha.2
#   tools/bump_version.sh beta        # 0.1.0-alpha.3 -> 0.1.0-beta.1
#   tools/bump_version.sh release     # 0.1.0-beta.2  -> 0.1.0
#   tools/bump_version.sh 0.2.0-alpha.1   # set it outright
#
# Commits nothing; tools/release_public.sh tags v<version> when it builds
# from a clean tree.
set -e
cd "$(dirname "$0")/.."
CUR=$(grep -m1 '^version' app/Cargo.toml | cut -d'"' -f2)
CORE=${CUR%%-*}
PRE=; [ "$CORE" != "$CUR" ] && PRE=${CUR#*-}
TAG=${PRE%%.*}; N=${PRE##*.}
case "${1:-next}" in
  next)
    [ -n "$PRE" ] || { echo "$CUR is a release: name the next one, e.g. $0 0.2.0-alpha.1" >&2; exit 1; }
    NEW="$CORE-$TAG.$((N + 1))" ;;
  alpha|beta|rc) NEW="$CORE-$1.1" ;;
  release) NEW=$CORE ;;
  *) NEW=$1 ;;
esac
sed -i '' "s/^version = \"$CUR\"/version = \"$NEW\"/" app/Cargo.toml
cargo metadata --format-version 1 >/dev/null   # rewrites Cargo.lock's app entry
echo "$CUR -> $NEW"
