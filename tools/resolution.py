
#!/usr/bin/env python3
"""Show the resolution gain of a merge against a single frame and the NX reference.

Usage:
  tools/resolution.py OUT.dng --ref SERIES/_DSCnnnn_merged.NEFX --single SERIES/_DSCnnnn.NEF
                          [--work DIR] [--panel 512]
Writes <work>/resolution_panels.png with three 1:1 crops:
  single frame (LibRaw demosaic) upscaled 2x | our merge | NX Studio reference.
"""
import argparse, os, shutil, subprocess, json
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def read_ppm(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"P6"
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = map(int, line.split())
        maxv = int(f.readline())
        off = f.tell()
    assert maxv == 65535
    return np.memmap(path, dtype=">u2", mode="r", offset=off, shape=(h, w, 3))


def decode_raw(path, work):
    produced = path + ".ppm"
    if os.path.exists(produced):
        os.remove(produced)
    cmd = ["dcraw_emu", "-4", "-o", "0", "-W", "-q", "11", path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(produced):
        raise SystemExit("dcraw_emu failed for %s / %s / %s" % (path, r.stdout.strip(), r.stderr.strip()))
    dst = os.path.join(work, os.path.basename(path) + ".ppm")
    shutil.move(produced, dst)
    return dst


def luma(a):
    return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114


def lap(c):
    l = np.asarray(c, dtype=np.float32).mean(axis=2)
    return float(np.abs(4 * l[1:-1, 1:-1] - l[:-2, 1:-1] - l[2:, 1:-1] - l[1:-1, :-2] - l[1:-1, 2:]).mean())


def sharp_window(a, size, step=48):
    g = np.asarray(luma(a), dtype=np.float32)
    dense = g[::4, ::4]
    gy, gx = np.gradient(dense)
    energy = np.abs(gx) + np.abs(gy)
    ii = np.pad(energy.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    bs = max(4, size // 4)
    best = (0.0, 0, 0)
    for y in range(0, max(1, dense.shape[0] - bs), max(1, step // 4)):
        for x in range(0, max(1, dense.shape[1] - bs), max(1, step // 4)):
            s = ii[y + bs, x + bs] - ii[y, x + bs] - ii[y + bs, x] + ii[y, x]
            if s > best[0]:
                best = (float(s), x * 4, y * 4)
    _, x0, y0 = best
    x0 = int(min(max(0, x0), a.shape[1] - size - 1))
    y0 = int(min(max(0, y0), a.shape[0] - size - 1))
    return x0, y0


def to8(c):
    p = float(np.percentile(c, 99.5))
    return (np.clip(np.asarray(c, dtype=np.float32) / max(p, 1e-9), 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--single", required=True)
    ap.add_argument("--work", default=".scratch/resolution")
    ap.add_argument("--panel", type=int, default=512)
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)

    p_out = decode_raw(os.path.abspath(args.out), args.work)
    p_ref = decode_raw(os.path.abspath(args.ref), args.work)
    p_one = decode_raw(os.path.abspath(args.single), args.work)
    out = read_ppm(p_out)
    ref = read_ppm(p_ref)
    one = read_ppm(p_one)
    size = args.panel
    x0, y0 = sharp_window(ref, size)
    crop_ref = np.asarray(ref[y0:y0 + size, x0:x0 + size], dtype=np.float32)
    crop_out = np.asarray(out[y0:y0 + size, x0:x0 + size], dtype=np.float32)
    # single frame is 1x: take half-size window and interpolate to the 2x grid
    from scipy.ndimage import zoom
    half = size // 2
    x1, y1 = x0 // 2, y0 // 2
    one_crop = np.asarray(one[y1:y1 + half, x1:x1 + half], dtype=np.float32)
    one_up = zoom(one_crop, (2.0, 2.0, 1.0), order=3)

    def norm(c, g):
        return c * (g / max(float(np.asarray(c)[..., 1].mean()), 1e-6))

    g_ref = float(crop_ref[..., 1].mean())
    a = norm(one_up, g_ref)
    b = norm(crop_out, g_ref)
    c = crop_ref
    res = {
        "window": [int(x0), int(y0)],
        "panel": size,
        "laplacian_single_upscaled": lap(a),
        "laplacian_merge": lap(b),
        "laplacian_reference": lap(c),
        "gain_vs_single": lap(b) / max(lap(a), 1e-9),
        "ratio_vs_reference": lap(b) / max(lap(c), 1e-9),
    }
    print(json.dumps(res, indent=2))
    sep = np.full((size, 8, 3), 255, np.uint8)
    panel = np.concatenate([to8(a), sep, to8(b), sep, to8(c)], axis=1)
    png = os.path.join(args.work, "resolution_panels.png")
    Image.fromarray(panel).save(png)
    print("saved", png)
    for p in (p_out, p_ref, p_one):
        try:
            os.remove(p)
        except OSError:
            pass


if __name__ == "__main__":
    main()
