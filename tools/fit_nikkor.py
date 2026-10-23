import numpy as np, os, subprocess, shutil
from PIL import Image
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

work = '.scratch/nikkor'; os.makedirs(work, exist_ok=True)
ours = read_ppm(decode('.scratch/nikkor/ours.dng', work))
ref  = read_ppm(decode('108NZ6_3/_DSC0582_merged.NEFX', work))
H, W = ref.shape[:2]
print('shapes', ours.shape, ref.shape)

S = 8
def box(a, s):
    h, w = a.shape[:2]; h2, w2 = h//s, w//s
    b = np.asarray(a[:h2*s, :w2*s], dtype=np.float32)
    return b.reshape(h2, s, w2, s).mean(axis=(1,3))

def luma_small(a):
    return box(a[...,0]*0.299 + a[...,1]*0.587 + a[...,2]*0.114, S)

Lo = luma_small(ours); Lr = luma_small(ref)
print('small shapes', Lo.shape, Lr.shape)

def match(R, P, maxs):
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

s0, maxs = 192, 26
Hs, Ws = Lr.shape
cx, cy = Ws/2.0, Hs/2.0
pts, meas, qual = [], [], []
for gy_ in range(3):
    for gx_ in range(4):
        x0 = int((gx_ + 0.5)*Ws/4 - s0/2); y0 = int((gy_ + 0.5)*Hs/3 - s0/2)
        x0 = max(maxs, min(Ws - s0 - maxs, x0)); y0 = max(maxs, min(Hs - s0 - maxs, y0))
        R = Lr[y0:y0+s0, x0:x0+s0]
        P = Lo[y0-maxs:y0+s0+maxs, x0-maxs:x0+s0+maxs]
        dx, dy, ncc = match(R, P, maxs)
        px, py = (x0 + s0/2.0), (y0 + s0/2.0)
        pts.append((px*S, py*S)); meas.append((dx*S, dy*S)); qual.append(ncc)
        print('block (%5d,%5d) d=(%+7.1f,%+7.1f) full-res, ncc=%.3f' % (px*S, py*S, dx*S, dy*S, ncc), flush=True)
P_ = np.array(pts); D = np.array(meas)
rnorm = float(np.hypot(W/2.0, H/2.0))
X = P_[:,0] - W/2.0; Y = P_[:,1] - H/2.0
u = X/rnorm; v = Y/rnorm; r2 = u*u + v*v
A = np.zeros((2*len(P_), 8)); b = np.zeros(2*len(P_))
for i in range(len(P_)):
    A[2*i,   0] = 1.0; A[2*i, 1] = X[i]; A[2*i, 2] = Y[i]; A[2*i, 6] = X[i]*r2[i]; A[2*i, 7] = X[i]*r2[i]*r2[i]
    A[2*i+1, 3] = 1.0; A[2*i+1, 4] = X[i]; A[2*i+1, 5] = Y[i]; A[2*i+1, 6] = Y[i]*r2[i]; A[2*i+1, 7] = Y[i]*r2[i]*r2[i]
    b[2*i] = D[i,0]; b[2*i+1] = D[i,1]
sol, *_ = np.linalg.lstsq(A, b, rcond=None)
res = A @ sol - b
print()
print('fit: shift (%.2f,%.2f), scale (%+.5f,%+.5f), shear (%+.5f,%+.5f)' % (sol[0], sol[3], sol[1], sol[4], sol[2], sol[5]))
print('radial (backward/correction map in types.rs terms): k1 = %+.5f, k2 = %+.5f' % (sol[6], sol[7]))
print('corner displacement: %+.1f px (of half diagonal %.0f px)' % (rnorm*(sol[6]+sol[7]), rnorm))
print('residual rms %.2f px, max %.2f px' % (np.sqrt((res*res).mean()), np.abs(res).max()))