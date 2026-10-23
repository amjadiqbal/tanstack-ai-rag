#!/bin/zsh
# render.sh <art dir>: SVG -> PNG with headless Chrome (transparent background).
set -e
D="$1"; CH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
P=$(mktemp -d)
r() { "$CH" --headless=new --disable-gpu --hide-scrollbars --user-data-dir="$P" \
  --default-background-color=00000000 --force-device-scale-factor=1 \
  --window-size=$2,$3 --screenshot="$D/${1%.svg}.png" "file://$D/$1" >/dev/null 2>&1; }
r icon.svg 512 512
r banner-dark.svg 1200 300
r banner-light.svg 1200 300
r social-preview.svg 1280 640
rm -rf "$P"
for f in "$D"/*.png; do magick identify -format '%f %wx%h %[channels]\n' "$f"; done
