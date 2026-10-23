import numpy as np, os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def read_ppm(path):
    f = open(path, 'rb'); assert f.readline().strip() == b'P6'
    line = f.readline()
    while line.startswith(b'#'): line = f.readline()
    w, h = map(int, line.split()); maxv = int(f.readline()); off = f.tell(); f.close()
    return np.memmap(path, dtype='>u2', mode='r', offset=off, shape=(h, w, 3))

a = read_ppm('.scratch/distort/104NZ6_3.dng.ppm')
H, W = a.shape[:2]
g = np.asarray(a[...,1], dtype=np.float32)
print('image', W, H)

# find the strongest long horizontal edge: row-wise mean |vertical gradient| over a wide strip
gy = np.abs(g[1:, :] - g[:-1, :])
strip = gy[:, 600:W-600]
row_energy = strip.mean(axis=1)
order = np.argsort(row_energy)[::-1]
cands = []
for r in order:
    if all(abs(r - c) > 40 for c in cands):
        cands.append(int(r))
    if len(cands) >= 6: break
print('candidate rows:', cands, 'energies', [round(float(row_energy[r]),1) for r in cands])

cx, cy = W/2.0, H/2.0
rnorm = float(np.hypot(cx, cy))
for r0 in cands[:3]:
    ys, xs = [], []
    for x in range(400, W-400, 40):
        col = gy[max(0,r0-6):r0+7, x-2:x+3].mean(axis=1)
        i = int(np.argmax(col))
        if i <= 0 or i >= len(col)-1: continue
        cm, c0, cp = col[i-1], col[i], col[i+1]
        den = cm - 2*c0 + cp
        sub = 0.0 if abs(den) < 1e-9 else 0.5*(cm - cp)/den
        y = max(0, r0-6) + i + sub
        ys.append(y); xs.append(float(x))
    xs = np.array(xs); ys = np.array(ys)
    if len(xs) < 40: continue
    A = np.vstack([np.ones_like(xs), xs - cx, (xs - cx)**2]).T
    sol, *_ = np.linalg.lstsq(A, ys, rcond=None)
    resid = A @ sol - ys
    Y = sol[0] - cy
    k1 = sol[2]*rnorm*rnorm/max(abs(Y), 1.0)
    print('edge near row %d: y(x) = %.1f %+.5f*(x-cx) %+.3e*(x-cx)^2, rms %.2f px; Y=%.0f -> k1 approx %+.5f' % (
          r0, sol[0], sol[1], sol[2], float(np.sqrt((resid*resid).mean())), Y, k1))