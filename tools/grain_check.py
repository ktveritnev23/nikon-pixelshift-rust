import numpy as np, os, subprocess, shutil
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def read_ppm(path):
    f = open(path, 'rb')
    assert f.readline().strip() == b'P6'
    line = f.readline()
    while line.startswith(b'#'):
        line = f.readline()
    w, h = map(int, line.split())
    maxv = int(f.readline())
    off = f.tell()
    f.close()
    assert maxv == 65535
    return np.memmap(path, dtype='>u2', mode='r', offset=off, shape=(h, w, 3))

def decode(path, work):
    out = path + '.ppm'
    if os.path.exists(out):
        os.remove(out)
    r = subprocess.run(['dcraw_emu', '-4', '-o', '0', '-W', path], capture_output=True, text=True)
    assert r.returncode == 0 and os.path.exists(out), r.stderr
    dst = os.path.join(work, os.path.basename(path) + '.ppm')
    shutil.move(out, dst)
    return dst

def lap2(x):
    return 4*x[1:-1,1:-1] - x[:-2,1:-1] - x[2:,1:-1] - x[1:-1,:-2] - x[1:-1,2:]

work = '.scratch/grain'
os.makedirs(work, exist_ok=True)
ours = read_ppm(decode('out/104NZ6_3.dng', work))
ref = read_ppm(decode('104NZ6_3/_DSC0522_merged.NEFX', work))
print('shapes', ours.shape, ref.shape)
luma = lambda a: a[...,0]*0.299 + a[...,1]*0.587 + a[...,2]*0.114
g = np.asarray(luma(ref), dtype=np.float32)
step = 128
best = (1e18, 0, 0)
for y in range(0, g.shape[0]-step, 64):
    for x in range(0, g.shape[1]-step, 64):
        blk = g[y:y+step, x:x+step]
        if blk.mean() < 800 or blk.mean() > 12000:
            continue
        s = blk.std()
        if s < best[0]:
            best = (float(s), x, y)
print('flattest patch in reference: std', round(best[0],1), 'at', best[1], best[2])
x0, y0 = best[1], best[2]
size = 256
o = np.asarray(ours[y0:y0+size, x0:x0+size], dtype=np.float32)
r = np.asarray(ref[y0:y0+size, x0:x0+size], dtype=np.float32)
def stats(a, name):
    lum = luma(a)
    print(name, 'mean RGB', [round(float(v)) for v in (a[...,0].mean(), a[...,1].mean(), a[...,2].mean())],
          'std RGB', [round(float(v),1) for v in (a[...,0].std(), a[...,1].std(), a[...,2].std())],
          'luma std', round(float(lum.std()),1))
    rg = a[...,0]-a[...,1]
    bg = a[...,2]-a[...,1]
    print('   hf chroma noise: R-G', round(float(lap2(rg).std()),2), 'B-G', round(float(lap2(bg).std()),2),
          'luma', round(float(lap2(lum).std()),2))
stats(r, 'NX reference')
stats(o, 'our merge   ')
def to8(c):
    p = float(np.percentile(c, 99.5))
    return (np.clip(c/max(p,1e-9), 0, 1)**(1/2.2)*255).astype(np.uint8)
up = np.array(Image.fromarray(to8(o)).resize((size*2, size*2), Image.NEAREST))
up2 = np.array(Image.fromarray(to8(r)).resize((size*2, size*2), Image.NEAREST))
sep = np.full((size*2, 8, 3), 255, np.uint8)
Image.fromarray(np.concatenate([up, sep, up2], axis=1)).save(os.path.join(work, 'grain_1to1.png'))
print('saved', os.path.join(work, 'grain_1to1.png'))