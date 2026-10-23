
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def read_ppm(path):
    with open(path, 'rb') as f:
        assert f.readline().strip() == b'P6'
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()
        w, h = map(int, line.split())
        maxv = int(f.readline()); off = f.tell()
    assert maxv == 65535
    return np.memmap(path, dtype='>u2', mode='r', offset=off, shape=(h, w, 3))

def luma(a):
    return a[...,0]*0.299 + a[...,1]*0.587 + a[...,2]*0.114

a4 = read_ppm('.scratch/verify/nefx_4.ppm')    # NX 4-shot, 1x
a16 = read_ppm('.scratch/verify/nefx_16.ppm')  # NX 16-shot, 2x

# coarse alignment of the two captures on a 1/8 grid of luma
A = np.asarray(luma(a4[::8, ::8]), dtype=np.float32)
B = np.asarray(luma(a16[::16, ::16]), dtype=np.float32)   # both end up 6048/8 x 4032/8
m = 12
best = (-2.0, 0, 0)
for dy in range(-10, 11):
    for dx in range(-10, 11):
        aa = A[m+dy:A.shape[0]-m+dy, m+dx:A.shape[1]-m+dx]
        bb = B[m:B.shape[0]-m, m:B.shape[1]-m]
        if aa.shape != bb.shape: continue
        av = aa - aa.mean(); bv = bb - bb.mean()
        den = np.sqrt((av*av).sum()*(bv*bv).sum())
        r = float((av*bv).sum()/max(den,1e-9))
        if r > best[0]: best = (r, dx, dy)
print('coarse alignment: r=%.4f dx=%d dy=%d (units: 8 px of the 1x grid = 16 px of the 2x grid)' % best)
_, ddx, ddy = best
sx, sy = 8*ddx, 16*ddy   # offset in 2x coordinates

# pick the sharpest 512x512 window of the 1x reference
g = np.asarray(luma(a4), dtype=np.float32)
gy, gx = np.gradient(g[::4, ::4])
energy = np.abs(gx) + np.abs(gy)
ii = np.pad(energy.cumsum(0).cumsum(1), ((1,0),(1,0)))
bs = 128   # 512/4
bestw = (0.0, 0, 0)
for y in range(0, energy.shape[0]-bs, 32):
    for x in range(0, energy.shape[1]-bs, 32):
        s = ii[y+bs, x+bs] - ii[y, x+bs] - ii[y+bs, x] + ii[y, x]
        if s > bestw[0]: bestw = (float(s), x*4, y*4)
_, x0, y0 = bestw
print('sharpest 1x window at', x0, y0)

crop1 = np.asarray(a4[y0:y0+512, x0:x0+512], dtype=np.float32)
X0 = x0*2 + sx; Y0 = y0*2 + sy
X0 = max(0, min(X0, a16.shape[1]-1024)); Y0 = max(0, min(Y0, a16.shape[0]-1024))
crop2 = np.asarray(a16[Y0:Y0+1024, X0:X0+1024], dtype=np.float32)
print('crop1', crop1.shape, 'crop2', crop2.shape, 'window in 2x coords', X0, Y0)

from scipy.ndimage import zoom
up = zoom(crop1, (2.0, 2.0, 1.0), order=3)
def lap(c):
    l = c.mean(axis=2)
    return float(np.abs(4*l[1:-1,1:-1]-l[:-2,1:-1]-l[2:,1:-1]-l[1:-1,:-2]-l[1:-1,2:]).mean())
g1 = crop1[...,1].mean(); g2 = crop2[...,1].mean()
print('green means: 1x=%.0f 2x=%.0f ratio=%.4f' % (g1, g2, g2/g1))
c1 = crop1*(g2/g1)
print('laplacian (raw ADU): 1x-upscaled=%.2f   NX 2x native=%.2f  ratio=%.2f' % (lap(up), lap(crop2), lap(crop2)/max(lap(up),1e-9)))

def to8(c):
    p = np.percentile(c, 99.5)
    return (np.clip(c/max(p,1e-9),0,1)**(1/2.2)*255).astype(np.uint8)
side = np.concatenate([to8(up), np.full((1024,8,3),255,np.uint8), to8(crop2)], axis=1)
Image.fromarray(side).save('.scratch/verify/nx_1x_up_vs_nx_2x.png')
print('saved .scratch/verify/nx_1x_up_vs_nx_2x.png')
