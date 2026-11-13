#!/bin/sh
# Ship an alpha to testers through CI: tag the current commit and push the
# tag. .github/workflows/alpha.yml then runs the full CI, builds macOS +
# Windows, crash-tests them and publishes both.
#
#   tools/ship_alpha.sh "notes for testers"
#
# The tag is alpha-<version>-<UTC date-time>, so shipping twice in a day
# never collides, and its message is the tester notes.
set -e
cd "$(dirname "$0")/.."
NOTES=${1:?usage: ship_alpha.sh "notes for testers"}

if [ -n "$(git status --porcelain)" ]; then
  echo "ship_alpha: working tree is dirty — commit (and push) first" >&2
  exit 1
fi
git fetch -q origin
# CI builds what is on GitHub: a tag on a commit that is not pushed would
# ship code nobody can check out.
if [ -z "$(git branch -r --contains HEAD)" ]; then
  echo "ship_alpha: HEAD is not on origin yet — push it first" >&2
  exit 1
fi

VERSION=$(grep -m1 '^version' app/Cargo.toml | cut -d'"' -f2)
TAG="alpha-$VERSION-$(date -u +%Y%m%d-%H%M)"
git tag -a "$TAG" -m "$NOTES"
git push origin "$TAG"
echo "== $TAG pushed: CI is building and will publish when it passes"
echo "   watch: gh run watch \$(gh run list --workflow alpha.yml -L 1 --json databaseId -q '.[0].databaseId')"
