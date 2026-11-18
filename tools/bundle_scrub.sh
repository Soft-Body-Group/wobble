#!/bin/sh
# Package the scrub binary as Scrub.app, so Scrub is an app of its own on
# macOS: its own icon and name in the Dock, Launchpad and Spotlight, opened
# from Finder like anything else (wobble's twin is tools/bundle_app.sh).
#
#   tools/bundle_scrub.sh            # builds, writes dist/Scrub.app
#   tools/bundle_scrub.sh --install  # ...and copies it to /Applications
#
# tools/release_public.sh reuses this with SCRUB_BIN (a binary it already
# built, skips the build) and SCRUB_APP_OUT (bundle elsewhere), then makes it
# self-contained with tools/bundle_dylibs.py. A local build links Homebrew's
# FFmpeg in place, so it runs on this Mac only.
set -e
cd "$(dirname "$0")/.."
BIN=${SCRUB_BIN:-target/release/scrub}
[ -n "$SCRUB_BIN" ] || cargo build --release -p scrub_app
APP=${SCRUB_APP_OUT:-dist/Scrub.app}
VERSION=$(grep -m1 '^version' app/Cargo.toml | cut -d'"' -f2)
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN" "$APP/Contents/MacOS/scrub"
cp scrub/app/icons/scrub.icns "$APP/Contents/Resources/scrub.icns"
# -L: cargo's scrub.dSYM is a symlink into deps/ (see bundle_app.sh).
[ -d "$BIN.dSYM" ] && cp -RL "$BIN.dSYM" "$APP/Contents/MacOS/scrub.dSYM"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Scrub</string>
  <key>CFBundleDisplayName</key><string>Scrub</string>
  <key>CFBundleIdentifier</key><string>ing.wobble.scrub</string>
  <key>CFBundleExecutable</key><string>scrub</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleIconFile</key><string>scrub</string>
  <key>CFBundleVersion</key><string>${VERSION%%-*}</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>LSApplicationCategoryType</key><string>public.app-category.video</string>
</dict>
</plist>
PLIST
# Ad-hoc sign so Gatekeeper lets a local build launch.
codesign --force --deep --sign - "$APP" >/dev/null 2>&1 || true
if [ -n "$SCRUB_APP_OUT" ]; then
  echo "wrote $APP"
elif [ "$1" = "--install" ]; then
  rm -rf /Applications/Scrub.app
  cp -R "$APP" /Applications/
  echo "installed /Applications/Scrub.app"
else
  echo "wrote $APP"
fi
