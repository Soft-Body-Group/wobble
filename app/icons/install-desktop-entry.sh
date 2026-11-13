#!/bin/sh
# Linux: add Wobble to your app launcher with its icon. Run from the unpacked
# folder; it points the entry at this folder's ./wobble.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=${XDG_DATA_HOME:-$HOME/.local/share}
mkdir -p "$DATA/applications" "$DATA/icons/hicolor/256x256/apps" "$DATA/icons/hicolor/512x512/apps"
cp "$HERE/icons/wobble-256.png" "$DATA/icons/hicolor/256x256/apps/wobble.png"
cp "$HERE/icons/wobble-512.png" "$DATA/icons/hicolor/512x512/apps/wobble.png"
cat > "$DATA/applications/wobble.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Wobble
Comment=GPU compositor
Exec="$HERE/wobble"
Path=$HERE
Icon=wobble
Terminal=false
Categories=Graphics;Video;AudioVideo;
DESKTOP
echo "added $DATA/applications/wobble.desktop"
