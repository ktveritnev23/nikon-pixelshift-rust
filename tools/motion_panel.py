#!/usr/bin/env python3
"""Comparison panels for motion compensation experiments.

Renders 1:1 crops of the same window from four sources side by side:
  1. our merge without motion compensation,
  2. our merge with motion compensation,
  3. the NX Studio reference (NEFX) of the same series,
  4. the single reference frame (demosaiced by LibRaw),
plus a difference map |plain - masked| that highlights where the mask acted.

Usage:
  python3 tools/motion_panel.py PLAIN.dng MOTION.dng [--ref NEFX] [--single NEF] \
      [--x X] [--y Y] [--size N] [--out panel.png] [--gamma]

Coordinates are in pixels of OUR output (2x grid for 32-frame series); the
NEFX and the single frame are 1x and get upscaled 2x to match. The crops are
rendered through the same neutral pipeline for all sources: LibRaw linear
decode (dcraw_emu -4 -o 0), per-panel normalisation, optional gamma.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def decode(path: Path, extra: list[str]) -> np.ndarray:
    """Decode a raw/DNG with LibRaw into a 16-bit RGB array (PPM path, the
    same one tools/validate.py uses)."""
    cmd = ["dcraw_emu", "-4", "-o", "0", "-W", *extra, str(path)]
    raw = subprocess.run(cmd, capture_output=True)
    produced = Path(str(path) + ".ppm")
    if raw.returncode != 0 or not produced.exists():
        sys.exit(f"dcraw_emu failed on {path}:\n{raw.stderr.decode()[-500:]}")
    blob = produced.read_bytes()
    produced.unlink()
    if not blob.startswith(b"P6"):
        sys.exit(f"{path}: expected a binary PPM")
    fields = []
    pos = 2
    while len(fields) < 3:
        while pos < len(blob) and blob[pos : pos + 1].isspace():
            pos += 1
        if blob[pos : pos + 1] == b"#":
            while blob[pos : pos + 1] != b"\n":
                pos += 1
            continue
        start = pos
        while pos < len(blob) and not blob[pos : pos + 1].isspace():
            pos += 1
        fields.append(int(blob[start:pos]))
    pos += 1
    w, h, _max = fields
    img = (
        np.frombuffer(blob[pos : pos + w * h * 6], dtype=">u2")
        .astype(np.uint16)
        .reshape(h, w, 3)
    )
    return img


def crop(img: np.ndarray, x: int, y: int, size: int, scale: int) -> np.ndarray:
    """Crop a window at OUR-output coordinates; upscale 1x sources."""
    h, w = img.shape[:2]
    x0, y0 = x // scale, y // scale
    s = size // scale
    x0 = max(0, min(x0, w - s))
    y0 = max(0, min(y0, h - s))
    win = img[y0 : y0 + s, x0 : x0 + s]
    if scale == 1:
        win = np.repeat(np.repeat(win, 2, axis=0), 2, axis=1)
    return win


def normalise(win: np.ndarray, gamma: bool) -> np.ndarray:
    v = win.astype(np.float64)
    lo, hi = np.percentile(v, 0.5), np.percentile(v, 99.5)
    if hi <= lo:
        hi = lo + 1.0
    v = (v - lo) / (hi - lo)
    if gamma:
        v = np.clip(v, 0.0, 1.0) ** (1.0 / 2.2)
    return np.clip(v * 255, 0, 255).astype(np.uint8)


def save_png(strips, titles, out: Path) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        sys.exit("Pillow is required for the panel output")
    h = max(s.shape[0] for s in strips)
    w = strips[0].shape[1]
    bar = 22
    canvas = Image.new("RGB", (w * len(strips), h + bar), (16, 16, 18))
    draw = ImageDraw.Draw(canvas)
    for i, (strip, title) in enumerate(zip(strips, titles)):
        canvas.paste(Image.fromarray(strip), (i * w, bar))
        draw.text((i * w + 6, 4), title, fill=(220, 220, 225))
    canvas.save(out)
    print(f"wrote {out} ({canvas.width}x{canvas.height})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("plain", type=Path)
    ap.add_argument("motion", type=Path)
    ap.add_argument("--ref", type=Path, default=None, help="NX Studio NEFX")
    ap.add_argument("--single", type=Path, default=None, help="single reference NEF")
    ap.add_argument("--x", type=int, default=0)
    ap.add_argument("--y", type=int, default=0)
    ap.add_argument("--size", type=int, default=512, help="window edge in our pixels")
    ap.add_argument("--out", type=Path, default=Path("motion_panel.png"))
    ap.add_argument("--gamma", action="store_true", help="display gamma for viewing")
    args = ap.parse_args()

    plain = decode(args.plain, [])
    motion = decode(args.motion, [])
    sources = [("plain", plain, 2), ("motion", motion, 2)]
    if args.ref:
        sources.append(("NX reference", decode(args.ref, []), 1))
    if args.single:
        sources.append(("single frame", decode(args.single, ["-q", "3"]), 1))

    strips, titles = [], []
    windows = {}
    for name, img, scale in sources:
        win = crop(img, args.x, args.y, args.size, scale)
        windows[name] = win
        strips.append(normalise(win, args.gamma))
        titles.append(name)

    # Difference map: where the mask changed the result.
    diff = np.abs(windows["plain"].astype(np.float64) - windows["motion"].astype(np.float64))
    diff = diff.max(axis=2)
    lo, hi = 0.0, max(np.percentile(diff, 99.9), 1.0)
    vis = np.clip(diff / hi, 0.0, 1.0)
    vis = (vis * 255).astype(np.uint8)
    vis = np.repeat(vis[:, :, None], 3, axis=2)
    strips.append(vis)
    titles.append("|plain - motion|")

    save_png(strips, titles, args.out)


if __name__ == "__main__":
    main()
