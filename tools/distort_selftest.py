import numpy as np, os, subprocess, shutil
from PIL import Image
from scipy.ndimage import shift as ndshift
Image.MAX_IMAGE_PIXELS = None

def decode(path, work):
    out = path + '.ppm'
    if os.path.exists(out): os.remove(out)
    r = subprocess.run(['dcraw_emu','-4','-o','0','-W',path], capture_output=True, text=True)
    assert r.returncode == 0 and os.path.exists(out), r.stderr
    dst = os.path.join(work, os.path.basename(path) + '.ppm'); shutil.move(out, dst); return dst

def read_ppm(path):
    f = open(path,'rb'); assert f.readline().strip() == b'P6'
    line = f.readline()
    while line.startswith(b'#'): line = f.readline()
    w,h = map(int, line.split()); maxv = int(f.readline()); off = f.tell(); f.close()
    return np.memmap(path, dtype='>u2', mode='r', offset=off, shape=(h,w,3))

work = '.scratch/distort'; os.makedirs(work, exist_ok=True)
ours = read_ppm(decode('out/104NZ6_3.dng', work))
ref  = read_ppm(decode('104NZ6_3/_DSC0522_merged.NEFX', work))
H, W = ref.shape[:2]

def luma(a, y0, x0, h, w):
    c = np.asarray(a[y0:y0+h, x0:x0+w], dtype=np.float32)
    return c[...,0]*0.299 + c[...,1]*0.587 + c[...,2]*0.114

def match(R, P, maxs):
    # R: reference patch (rh,rw); P: moving patch of (rh+2maxs, rw+2maxs) centred on the same content
    rh, rw = R.shape
    Rc = R - R.mean(); Rn = np.sqrt((Rc*Rc).sum())
    surf = np.zeros((2*maxs+1, 2*maxs+1))
    for iy in range(surf.shape[0]):
        for ix in range(surf.shape[1]):
            Q = P[iy:iy+rh, ix:ix+rw]
            Qc = Q - Q.mean(); Qn = np.sqrt((Qc*Qc).sum())
            surf[iy, ix] = float((Rc*Qc).sum()/max(Rn*Qn, 1e-9))
    iy, ix = np.unravel_index(np.argmax(surf), surf.shape)
    def sub(c0, cm, cp):
        d = cm - 2*c0 + cp
        return 0.0 if abs(d) < 1e-12 else 0.5*(cm - cp)/d
    dx = float(ix - maxs); dy = float(iy - maxs)
    if 0 < iy < surf.shape[0]-1: dy += sub(surf[iy,ix], surf[iy-1,ix], surf[iy+1,ix])
    if 0 < ix < surf.shape[1]-1: dx += sub(surf[iy,ix], surf[iy,ix-1], surf[iy,ix+1])
    return dx, dy, float(surf[iy, ix])

s0, maxs = 256, 8
y0t, x0t = 1800, 2200
R = luma(ref, y0t, x0t, s0, s0)
P0 = luma(ours, y0t - maxs, x0t - maxs, s0 + 2*maxs, s0 + 2*maxs)
print('selftest on real data (expect d ~ 0):', match(R, P0, maxs))
base = luma(ref, y0t - maxs - 10, x0t - maxs - 10, s0 + 2*maxs + 20, s0 + 2*maxs + 20)
for (tx, ty) in ((3.4, -2.2), (-5.0, 4.6)):
    M = ndshift(base, (ty, tx), order=3, mode='nearest')[10:10+s0+2*maxs, 10:10+s0+2*maxs]
    dx, dy, ncc = match(R, M, maxs)
    print('selftest content shift (%+.1f,%+.1f) -> d=(%+.2f,%+.2f) ncc=%.4f' % (tx, ty, dx, dy, ncc))