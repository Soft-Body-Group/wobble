#!/bin/sh
# Render the app icon (app/src/appicon.rs) at every size into app/icons/:
# wobble-<px>.png, wobble.ico (Windows, embedded by app/build.rs) and
# wobble.icns (macOS, copied into the bundle by tools/bundle_app.sh).
# Needs a GPU. Commit the results; builds never render the icon.
set -e
cd "$(dirname "$0")/.."
cargo test -p app --release appicon::tests::app_icon_render -- --ignored --exact
iconutil -c icns app/icons/wobble.iconset -o app/icons/wobble.icns
rm -rf app/icons/wobble.iconset
echo "wrote app/icons/"
