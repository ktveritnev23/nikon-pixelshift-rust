#!/usr/bin/env bash
# Rebuild the README screenshots from the built-in demo series.
# Every picture is one headless-ish window run of "pixelshift gui --shot", so
# the set is reproducible: re-running this script overwrites docs/img/ with the
# same five files. See "pixelshift gui --help" for the shot flags.
#
# Usage: tools/screenshots.sh
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=docs/img
BIN=target/release/pixelshift
LIMIT=921600 # ~900 KiB per JPEG

echo "== release build"
cargo build --release 2>&1 | tail -2
test -x "$BIN" || { echo "binary not found: $BIN" >&2; exit 1; }

mkdir -p "$OUT"

# shot <name> [gui flags...]: capture one window into docs/img/<name>.jpg
shot() {
  local name=$1
  shift
  echo "== $name"
  "$BIN" gui --demo --shot "$OUT/$name.jpg" "$@"
}

# 1, 2: the two themes and languages in the fit-to-window view.
shot gui-dark-ru --theme dark --lang ru
shot gui-light-en --theme light --lang en
# 3: a 1:1 crop of the middle of the frame at 200%, where the rings and the
# grid show the detail the merge reconstructs.
shot gui-detail --theme dark --lang en --shot-zoom 2.0 --shot-crop 80,52,480,320
# 4: the same window in Russian, log open, after saving the merged result: the
#   control labels are translated, the log lines themselves stay English.
shot gui-save --theme dark --lang ru --shot-save --shot-log
# 5: a preset lens correction; the straight lines of the grid bend.
shot gui-distort --theme dark --lang ru --distort -0.040

echo
echo "== screenshots"
fail=0
for name in gui-dark-ru gui-light-en gui-detail gui-save gui-distort; do
  path="$OUT/$name.jpg"
  bytes=$(wc -c <"$path" | tr -d ' ')
  if command -v sips >/dev/null 2>&1; then
    dims=$(sips -g pixelWidth -g pixelHeight "$path" |
      awk '/pixelWidth/ {w=$2} /pixelHeight/ {h=$2} END {printf "%sx%s", w, h}')
  else
    dims=unknown
  fi
  printf '%-22s %8s bytes  %s  %s\n' "$path" "$bytes" "$dims" "$(du -h "$path" | cut -f1)"
  if [ "$bytes" -gt "$LIMIT" ]; then
    echo "error: $path is larger than ~900 KiB" >&2
    fail=1
  fi
done
if [ "$fail" -ne 0 ]; then
  exit 1
fi
