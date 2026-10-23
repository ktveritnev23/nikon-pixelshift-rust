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

s0, maxs = 256, 10
cx, cy = W/2.0, H/2.0
rnorm = float(np.hypot(cx, cy))
pts, meas = [], []
for gy_ in range(4):
    for gx_ in range(6):
        x0 = int((gx_ + 0.5)*W/6 - s0/2); y0 = int((gy_ + 0.5)*H/4 - s0/2)
        x0 = max(maxs, min(W - s0 - maxs, x0)); y0 = max(maxs, min(H - s0 - maxs, y0))
        R = luma(ref, y0, x0, s0, s0)
        P = luma(ours, y0 - maxs, x0 - maxs, s0 + 2*maxs, s0 + 2*maxs)
        dx, dy, ncc = match(R, P, maxs)
        px, py = x0 + s0/2.0, y0 + s0/2.0
        pts.append((px, py)); meas.append((dx, dy))
        print('block (%4d,%4d) d=(%+.2f,%+.2f) ncc=%.3f' % (px, py, dx, dy, ncc), flush=True)
P = np.array(pts); D = np.array(meas)
X = P[:,0] - cx; Y = P[:,1] - cy
u = X/rnorm; v = Y/rnorm; r2 = u*u + v*v
A = np.zeros((2*len(P), 7)); b = np.zeros(2*len(P))
for i in range(len(P)):
    A[2*i,   0] = 1.0; A[2*i, 1] = X[i]; A[2*i, 2] = Y[i]; A[2*i, 6] = X[i]*r2[i]
    A[2*i+1, 3] = 1.0; A[2*i+1, 4] = X[i]; A[2*i+1, 5] = Y[i]; A[2*i+1, 6] = Y[i]*r2[i]
    b[2*i] = D[i,0]; b[2*i+1] = D[i,1]
sol, *_ = np.linalg.lstsq(A, b, rcond=None)
res = A @ sol - b
print('fit: global shift (%.2f,%.2f) px, scale (%+.5f,%+.5f), shear (%+.5f,%+.5f), k1 = %+.6f' % (sol[0], sol[3], sol[1], sol[4], sol[2], sol[5], sol[6]))
print('residual rms %.2f px, max %.2f px' % (np.sqrt((res*res).mean()), np.abs(res).max()))
print('predicted corner displacement from k1 alone: %.1f px' % (rnorm*sol[6]))
print('mean |d| at the four corners:')
for (qx, qy) in ((0,0),(W-1,0),(0,H-1),(W-1,H-1)):
    uu = (qx-cx)/rnorm; vv = (qy-cy)/rnorm
    print('  corner (%5d,%5d): d=(%+.2f,%+.2f) px' % (qx, qy, (qx-cx)*sol[6]*(uu*uu+vv*vv), (qy-cy)*sol[6]*(uu*uu+vv*vv)))