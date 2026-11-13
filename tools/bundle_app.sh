#!/bin/sh
# Package the release binary as Wobble.app so macOS routes wobble:// links
# (the plugin store's install buttons) to it. A bare binary cannot own a URL
# scheme; LaunchServices learns it from CFBundleURLTypes in this Info.plist.
#
#   tools/bundle_app.sh            # builds, writes dist/Wobble.app
#   tools/bundle_app.sh --install  # ...and copies it to /Applications
#
# tools/release_alpha.sh reuses this with WOBBLE_BIN (a binary it already
# built, skips the build) and WOBBLE_APP_OUT (bundle elsewhere, and leave
# LaunchServices alone so the alpha copy never steals wobble:// from yours).
set -e
cd "$(dirname "$0")/.."
BIN=${WOBBLE_BIN:-target/release/wobble}
[ -n "$WOBBLE_BIN" ] || cargo build --release -p app
APP=${WOBBLE_APP_OUT:-dist/Wobble.app}
VERSION=$(grep -m1 '^version' app/Cargo.toml | cut -d'"' -f2)
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN" "$APP/Contents/MacOS/wobble"
cp app/icons/wobble.icns "$APP/Contents/Resources/wobble.icns"
# Beside the binary is where Rust's backtrace looks for DWARF, so panic
# reports from this bundle carry file:line.
# -L: cargo makes `wobble.dSYM` a symlink into deps/. Copied as a link it
# dangles inside the bundle, which breaks the signature and macOS calls
# the app "damaged".
[ -d "$BIN.dSYM" ] && cp -RL "$BIN.dSYM" "$APP/Contents/MacOS/wobble.dSYM"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Wobble</string>
  <key>CFBundleDisplayName</key><string>Wobble</string>
  <key>CFBundleIdentifier</key><string>ing.wobble.app</string>
  <key>CFBundleExecutable</key><string>wobble</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleIconFile</key><string>wobble</string>
  <key>CFBundleVersion</key><string>${VERSION%%-*}</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>CFBundleURLTypes</key>
  <array>
    <dict>
      <key>CFBundleURLName</key><string>Wobble plugin link</string>
      <key>CFBundleURLSchemes</key><array><string>wobble</string></array>
    </dict>
  </array>
</dict>
</plist>
PLIST
# Ad-hoc sign so Gatekeeper lets a local build launch.
codesign --force --deep --sign - "$APP" >/dev/null 2>&1 || true
LSREG=/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister
if [ -n "$WOBBLE_APP_OUT" ]; then
  echo "wrote $APP"
elif [ "$1" = "--install" ]; then
  rm -rf /Applications/Wobble.app
  cp -R "$APP" /Applications/
  "$LSREG" -f /Applications/Wobble.app
  echo "installed /Applications/Wobble.app"
else
  "$LSREG" -f "$APP"
  echo "wrote $APP (registered wobble://)"
fi
