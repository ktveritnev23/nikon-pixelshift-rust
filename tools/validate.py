#!/usr/bin/env python3
"""Quantitative validation of a pixelshift result against the NX Studio reference.

Usage:
  tools/validate.py OUT.dng --ref SERIES/_DSCnnnn_merged.NEFX [--work DIR] [--keep]

Both files are decoded with LibRaw (dcraw_emu, linear 16-bit, raw colour space,
no brightening) and compared on a common grid: alignment, per-channel
correlation/gain, PSNR and a 1:1 detail crop.
"""
import argparse, json, os, shutil, subprocess, sys
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def read_ppm(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"P6", "not a P6 ppm: " + path
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = map(int, line.split())
        maxv = int(f.readline())
        off = f.tell()
    if maxv != 65535:
        raise SystemExit("expected 16-bit ppm, got maxval=%d (%s)" % (maxv, path))
    return np.memmap(path, dtype=">u2", mode="r", offset=off, shape=(h, w, 3))


def decode(path, work):
    """Decode a raw/DNG file with LibRaw into a 16-bit ppm inside work."""
    out_name = os.path.basename(path) + ".ppm"
    produced = path + ".ppm"
    if os.path.exists(produced):
        os.remove(produced)
    cmd = ["dcraw_emu", "-4", "-o", "0", "-W", path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(produced):
        raise SystemExit("dcraw_emu failed for %s\n%s\n%s" % (path, r.stdout, r.stderr))
    dst = os.path.join(work, out_name)
    shutil.move(produced, dst)
    return dst


def luma(a):
    return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114


def best_shift(a, b, max_shift=6, step=8):
    """Integer shift (dx, dy) maximising correlation of subsampled luma."""
    A = np.asarray(luma(a[::step, ::step]), dtype=np.float32)
    B = np.asarray(luma(b[::step, ::step]), dtype=np.float32)
    m = 16
    best = (-2.0, 0, 0)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            aa = A[m + dy:A.shape[0] - m + dy, m + dx:A.shape[1] - m + dx]
            bb = B[m:A.shape[0] - m, m:A.shape[1] - m]
            if aa.shape != bb.shape:
                continue
            av = aa - aa.mean()
            bv = bb - bb.mean()
            den = np.sqrt((av * av).sum() * (bv * bv).sum())
            if den <= 0:
                continue
            r = float((av * bv).sum() / den)
            if r > best[0]:
                best = (r, dx, dy)
    return best


def crop_pair(a, b, dx, dy, size):
    """Crop the same content from both images (b shifted by dx,dy).

    The block is chosen by gradient energy (not by standard deviation), so the
    comparison always lands on sharp, in-focus detail instead of a smooth
    background or an out of focus foreground.
    """
    h, w = b.shape[:2]
    step = max(size, 128)
    g = np.asarray(luma(b), dtype=np.float32)
    dense = g[::4, ::4]
    gy, gx = np.gradient(dense)
    energy = np.abs(gx) + np.abs(gy)
    bs = max(4, size // 4)
    # integral image for a fast block sum of the gradient energy
    ii = energy.cumsum(0).cumsum(1)
    ii = np.pad(ii, ((1, 0), (1, 0)))
    best = (0.0, 0, 0)
    for y in range(0, max(1, dense.shape[0] - bs), max(1, step // 4)):
        for x in range(0, max(1, dense.shape[1] - bs), max(1, step // 4)):
            s = ii[y + bs, x + bs] - ii[y, x + bs] - ii[y + bs, x] + ii[y, x]
            if s > best[0]:
                best = (float(s), x * 4, y * 4)
    _, x0, y0 = best
    x0 = int(min(max(0, x0), w - size - 1))
    y0 = int(min(max(0, y0), h - size - 1))
    cb = np.asarray(b[y0:y0 + size, x0:x0 + size], dtype=np.float32)
    xa, ya = x0 + dx, y0 + dy
    ca = np.asarray(a[ya:ya + size, xa:xa + size], dtype=np.float32)
    return ca, cb, (x0, y0)


def to8(c):
    p = float(np.percentile(c, 99.5))
    c = np.clip(c / max(p, 1e-6), 0, 1) ** (1 / 2.2)
    return (c * 255).astype(np.uint8)


def lap_energy(c):
    l = c.mean(axis=2)
    return float(np.abs(4 * l[1:-1, 1:-1] - l[:-2, 1:-1] - l[2:, 1:-1] - l[1:-1, :-2] - l[1:-1, 2:]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--work", default=".scratch/validate")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    os.makedirs(args.work, exist_ok=True)
    rep = {"out": args.out, "ref": args.ref}

    p_out = decode(os.path.abspath(args.out), args.work)
    p_ref = decode(os.path.abspath(args.ref), args.work)
    a = read_ppm(p_out)
    b = read_ppm(p_ref)
    rep["out_shape"] = list(a.shape)
    rep["ref_shape"] = list(b.shape)
    if a.shape != b.shape:
        raise SystemExit("shape mismatch: %s vs %s" % (a.shape, b.shape))

    r, dx, dy = best_shift(a, b)
    rep["alignment"] = {"corr": r, "dx": dx, "dy": dy}

    h, w = b.shape[:2]
    m = 8
    aa = np.asarray(a[m + dy:h - m + dy, m + dx:w - m + dx], dtype=np.float32)
    bb = np.asarray(b[m:h - m, m:w - m], dtype=np.float32)
    lum_a = luma(aa)
    lum_b = luma(bb)
    mask = (lum_a > 300) & (lum_a < 15000) & (lum_b > 300) & (lum_b < 15000)
    rep["mask_fraction"] = float(mask.mean())
    chans = {}
    mse_sum = 0.0
    for i, name in enumerate("RGB"):
        x = aa[..., i][mask].astype(np.float64)
        y = bb[..., i][mask].astype(np.float64)
        gain = float(x.mean() / max(y.mean(), 1e-9))
        corr = float(np.corrcoef(x, y)[0, 1])
        resid = x - gain * y
        rmse = float(np.sqrt((resid * resid).mean()))
        peak = float(max(x.mean(), 1.0))
        chans[name] = {"corr": corr, "gain": gain, "rmse": rmse, "psnr_db": float(20 * np.log10(peak / max(rmse, 1e-9))), "mean_out": float(x.mean()), "mean_ref": float(y.mean())}
        mse_sum += rmse * rmse
    rep["channels"] = chans
    rep["luma_corr"] = float(np.corrcoef(luma(aa)[mask], luma(bb)[mask])[0, 1])

    ca, cb, origin = crop_pair(a, b, dx, dy, 512)
    rep["crop_origin"] = list(origin)
    rep["detail_laplacian"] = {"out": lap_energy(ca), "ref": lap_energy(cb), "ratio": lap_energy(ca) / max(lap_energy(cb), 1e-9)}
    side = np.concatenate([to8(ca), np.full((512, 8, 3), 255, np.uint8), to8(cb)], axis=1)
    png = os.path.join(args.work, "detail_out_vs_ref.png")
    Image.fromarray(side).save(png)
    rep["detail_png"] = png

    print(json.dumps(rep, indent=2, ensure_ascii=False))
    if args.report:
        with open(args.report, "w") as f:
            json.dump(rep, f, indent=2)
    if not args.keep:
        for p in (p_out, p_ref):
            try:
                os.remove(p)
            except OSError:
                pass


if __name__ == "__main__":
    main()
