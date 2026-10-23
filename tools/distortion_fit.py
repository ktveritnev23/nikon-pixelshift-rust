import numpy as np, os, subprocess, shutil
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def decode(path, work):
    out = path + '.ppm'
    if os.path.exists(out): os.remove(out)
    r = subprocess.run(['dcraw_emu', '-4', '-o', '0', '-W', path], capture_output=True, text=True)
    assert r.returncode == 0 and os.path.exists(out), r.stderr
    dst = os.path.join(work, os.path.basename(path) + '.ppm')
    shutil.move(out, dst)
    return dst

def read_ppm(path):
    f = open(path, 'rb')
    assert f.readline().strip() == b'P6'
    line = f.readline()
    while line.startswith(b'#'): line = f.readline()
    w, h = map(int, line.split()); maxv = int(f.readline()); off = f.tell(); f.close()
    return np.memmap(path, dtype='>u2', mode='r', offset=off, shape=(h, w, 3))

work = '.scratch/distort'
os.makedirs(work, exist_ok=True)
ours = read_ppm(decode('out/104NZ6_3.dng', work))
ref = read_ppm(decode('104NZ6_3/_DSC0522_merged.NEFX', work))
H, W = ref.shape[:2]

def luma_block(a, y0, x0, h, w):
    c = np.asarray(a[y0:y0+h, x0:x0+w], dtype=np.float32)
    return c[...,0]*0.299 + c[...,1]*0.587 + c[...,2]*0.114

def peak_shift(ref_b, mov_b):
    # normalized cross correlation via FFT: mov(x + d) ~ ref(x)
    R = ref_b - ref_b.mean(); M = mov_b - mov_b.mean()
    H2, W2 = M.shape
    Rp = np.zeros_like(M); y0 = (H2 - R.shape[0]) // 2; x0 = (W2 - R.shape[1]) // 2
    Rp[y0:y0+R.shape[0], x0:x0+R.shape[1]] = R
    Fr = np.fft.rfft2(Rp); Fm = np.fft.rfft2(M)
    cc = np.fft.irfft2(Fr * np.conj(Fm))
    h, w = M.shape
    p = np.unravel_index(np.argmax(cc), cc.shape)
    py, px = p
    def refine(cm, c0, cp):
        d = cm - 2*c0 + cp
        return 0.0 if abs(d) < 1e-12 else 0.5*(cm - cp)/d
    dy = refine(cc[(py-1) % h, px], cc[py, px], cc[(py+1) % h, px])
    dx = refine(cc[py, (px-1) % w], cc[py, px], cc[py, (px+1) % w])
    sy = py + dy; sx = px + dx
    if sy > h/2: sy -= h
    if sx > w/2: sx -= w
    # the padded reference sits at the centre of M, so subtract its origin
    return sx - x0, sy - y0, float(cc[py, px])

s, m = 384, 48
cx, cy = W/2.0, H/2.0
rnorm = float(np.hypot(cx, cy))
pts, meas = [], []
for gy_ in range(4):
    for gx_ in range(6):
        y0 = int((gy_ + 0.5)*H/4 - s/2); x0 = int((gx_ + 0.5)*W/6 - s/2)
        x0 = max(0, min(W - s - 2*m, x0)); y0 = max(0, min(H - s - 2*m, y0))
        R = luma_block(ref, y0 + m, x0 + m, s, s)
        M = luma_block(ours, y0, x0, s + 2*m, s + 2*m)
        dx, dy, peak = peak_shift(R, M)
        # mov window origin is offset by m relative to ref window, so the content
        # displacement (ours - ref) in sensor pixels is (dx - m, dy - m)
        px = x0 + m + s/2.0; py = y0 + m + s/2.0
        pts.append((px, py)); meas.append((dx - m, dy - m))
        print('block (%4d,%4d) d=(%+.2f,%+.2f) peak=%.3f' % (px, py, dx-m, dy-m, peak), flush=True)

P = np.array(pts); D = np.array(meas)
X = (P[:,0] - cx); Y = (P[:,1] - cy)
u = X / rnorm; v = Y / rnorm; r2 = u*u + v*v
A = np.zeros((2*len(P), 7)); b = np.zeros(2*len(P))
for i in range(len(P)):
    A[2*i,   0] = 1.0; A[2*i, 1] = X[i]; A[2*i, 2] = Y[i]; A[2*i, 6] = X[i]*r2[i]
    A[2*i+1, 3] = 1.0; A[2*i+1, 4] = X[i]; A[2*i+1, 5] = Y[i]; A[2*i+1, 6] = Y[i]*r2[i]
    b[2*i] = D[i,0]; b[2*i+1] = D[i,1]
sol, *_ = np.linalg.lstsq(A, b, rcond=None)
resid = A @ sol - b
print('fit: shift=(%.2f,%.2f) px  scale=(%.5f,%.5f)  shear=(%.5f,%.5f)  k1=%.6f' % (sol[0], sol[3], sol[1], sol[4], sol[2], sol[5], sol[6]))
print('residual rms %.2f px, max %.2f px' % (np.sqrt((resid**2).mean()), np.abs(resid).max()))
pts_r = np.hypot(u, v)
print('radial displacement predicted at corners: %.2f px' % (sol[6]*1.0*np.hypot(cx, cy),))
np.save(os.path.join(work, 'fit.npy'), np.array([sol]), allow_pickle=True)