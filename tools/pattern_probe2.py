
import numpy as np, os, subprocess, shutil, sys
from PIL import Image
from scipy.ndimage import shift as ndshift, gaussian_filter
Image.MAX_IMAGE_PIXELS = None
BLACK, W, H = 1008, 6048, 4032
INSET = (8, 4)

def decode(nef, dst):
    subprocess.run(['unprocessed_raw', '-q', '-T', nef], check=True)
    src = nef + '.tiff'
    if not os.path.exists(src):
        src = os.path.splitext(nef)[0] + '.tiff'
    shutil.move(src, dst)

def load_active(path):
    a = np.array(Image.open(path)).astype(np.float32)
    y0, x0 = INSET[1], INSET[0]
    return a[y0:y0+H, x0:x0+W]

def Lplane(act):
    a = np.clip(act - BLACK, 0, None)
    h, w = a.shape
    h2, w2 = h//2, w//2
    b = a[:2*h2, :2*w2]
    return 0.25*(b[0::2,0::2] + b[1::2,0::2] + b[0::2,1::2] + b[1::2,1::2])

def phase_shift(ref, mov, crop=1024):
    h, w = ref.shape
    y0, x0 = h//2 - crop//2, w//2 - crop//2
    R = ref[y0:y0+crop, x0:x0+crop].astype(np.float32)
    M = mov[y0:y0+crop, x0:x0+crop].astype(np.float32)
    win = np.outer(np.hanning(crop), np.hanning(crop)).astype(np.float32)
    R = (R - R.mean())*win; M = (M - M.mean())*win
    FR = np.fft.rfft2(R); FM = np.fft.rfft2(M)
    cc = np.fft.irfft2(FR*np.conj(FM), s=(crop, crop))
    py, px = np.unravel_index(np.argmax(cc), cc.shape)
    def refine(cm, c0, cp):
        den = cm - 2*c0 + cp
        return 0.0 if abs(den) < 1e-20 else float(0.5*(cm - cp)/den)
    dy = refine(cc[(py-1) % crop, px], cc[py, px], cc[(py+1) % crop, px])
    dx = refine(cc[py, (px-1) % crop], cc[py, px], cc[py, (px+1) % crop])
    sx = px + dx; sy = py + dy
    if sx > crop/2: sx -= crop
    if sy > crop/2: sy -= crop
    return -sx, -sy, float(cc[py, px])

series = sys.argv[1]; first = sys.argv[2]; count = int(sys.argv[3])
work = '.scratch/pattern'
os.makedirs(work, exist_ok=True)
paths = []
for i in range(count):
    num = int(first[4:]) + i
    p = os.path.join(work, '%s_%d.tiff' % (series, i))
    if not os.path.exists(p):
        decode('%sNZ6_3/_DSC%04d.NEF' % (series, num), p)
    paths.append(p)
Ls = [Lplane(load_active(p)) for p in paths]
if series == '106':
    for true in [(0.25, 0.0), (-0.5, 0.25), (0.75, -0.5)]:
        M = ndshift(gaussian_filter(Ls[0], 1.0), (true[1], true[0]), order=3, mode='nearest')
        est = phase_shift(gaussian_filter(Ls[0], 1.0), M, 1024)
        print('selftest true_L=(%.2f,%.2f) -> est_L=(%.3f,%.3f) sensor=(%.2f,%.2f)' % (true[0], true[1], est[0], est[1], 2*est[0], 2*est[1]))
print('=== series', series)
for i in range(1, count):
    sx, sy, peak = phase_shift(Ls[0], Ls[i])
    print('  frame %2d: s_sensor=(%+.3f, %+.3f)  (L: %+.3f, %+.3f) peak=%.4g' % (i, 2*sx, 2*sy, sx, sy, peak))
    os.remove(paths[i])
