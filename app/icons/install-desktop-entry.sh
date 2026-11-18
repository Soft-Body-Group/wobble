#!/bin/sh
# Linux: add Wobble and Scrub to your app launcher, each with its icon. Run
# from the unpacked folder; the entries point at this folder's ./wobble and
# ./scrub (an entry is skipped if its binary is not here).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=${XDG_DATA_HOME:-$HOME/.local/share}
mkdir -p "$DATA/applications" "$DATA/icons/hicolor/256x256/apps" "$DATA/icons/hicolor/512x512/apps"
# entry <binary> <Name> <Comment> <Categories> [<Exec args>]
entry() {
  [ -x "$HERE/$1" ] || { echo "skipped $2: no $HERE/$1"; return 0; }
  for px in 256 512; do
    cp "$HERE/icons/$1-$px.png" "$DATA/icons/hicolor/${px}x${px}/apps/$1.png"
  done
  cat > "$DATA/applications/$1.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=$2
Comment=$3
Exec="$HERE/$1"${5:+ $5}
Path=$HERE
Icon=$1
Terminal=false
Categories=$4
DESKTOP
  echo "added $DATA/applications/$1.desktop"
}
entry wobble Wobble "GPU compositor" "Graphics;Video;AudioVideo;"
entry scrub Scrub "Video editor" "AudioVideo;Video;AudioVideoEditing;" %f
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$DATA/applications" || true
