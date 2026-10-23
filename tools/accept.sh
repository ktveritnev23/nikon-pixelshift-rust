#!/bin/sh
# Full acceptance run: release build, merge all four series, validate against NX Studio.
# Usage: tools/accept.sh [outdir]   (default: out)
set -e
cd "$(dirname "$0")/.."
OUT="${1:-out}"
mkdir -p "$OUT" .scratch/accept
export CARGO_HOME="$PWD/.cargo-home"
export CARGO_TARGET_DIR="$PWD/.cargo-target"

echo "== release build"
cargo build --release 2>&1 | tail -2
BIN="${CARGO_TARGET_DIR:-$PWD/target}/release/pixelshift"
test -x "$BIN" || { echo "binary not found: $BIN"; exit 1; }

for s in 104NZ6_3 105NZ6_3 106NZ6_3 107NZ6_3; do
  ref=$(ls "$s"/*_merged.NEFX | head -1)
  single=$(ls "$s"/*.NEF | head -1)
  echo "== merge $s"
  /usr/bin/time -l "$BIN" merge "$s" --out "$OUT/$s.dng" --quiet 2> ".scratch/accept/$s.time" || {
    echo "merge failed for $s"; cat ".scratch/accept/$s.time"; exit 1; }
  grep -E "maximum resident set size|real" ".scratch/accept/$s.time" | sed 's/^ *//' || true
  echo "== validate $s"
  PYTHONDONTWRITEBYTECODE=1 python3 tools/validate.py "$OUT/$s.dng" --ref "$ref" \
      --report ".scratch/accept/$s.validate.json" | tail -n +1 | python3 -c "
import json,sys
d=json.load(sys.stdin)
print('  size', d['out_shape'], 'luma_corr %.5f' % d['luma_corr'], 'align', d['alignment']['dx'], d['alignment']['dy'])
print('  channels', {k:(round(v['corr'],4), round(v['psnr_db'],2)) for k,v in d['channels'].items()})
print('  detail ratio %.3f' % d['detail_laplacian']['ratio'])
"
  case "$s" in
    106*|107*) echo "== resolution $s"
      PYTHONDONTWRITEBYTECODE=1 python3 tools/resolution.py "$OUT/$s.dng" --ref "$ref" --single "$single" \
        | grep -E 'gain_vs_single|ratio_vs_reference|laplacian' ;;
  esac
done
echo "== done: $OUT"