import numpy as np, os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def load16(path):
    im = Image.open(path)
    a = np.array(im)
    return a, im.mode

io_img, mode = load16('.scratch/grain/io_104.png')
print('ImageIO render:', io_img.shape, io_img.dtype, mode)

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
which = sorted([p for p in os.listdir('.scratch/grain') if p.endswith('.ppm')])
print('ppms:', which)
ours = read_ppm('.scratch/grain/104NZ6_3.dng.ppm')
ref = read_ppm('.scratch/grain/_DSC0522_merged.NEFX.ppm')

x0, y0, s = 3400, 160, 320
def norm8(a):
    a = np.asarray(a, dtype=np.float32)
    p = float(np.percentile(a, 99.0))
    return (np.clip(a/max(p,1e-9), 0, 1)**(1/2.2)*255).astype(np.uint8)
panels = []
panels.append(norm8(io_img[y0:y0+s, x0:x0+s]))
panels.append(norm8(np.asarray(ours[y0:y0+s, x0:x0+s], dtype=np.float32)))
panels.append(norm8(np.asarray(ref[y0:y0+s, x0:x0+s], dtype=np.float32)))
sep = np.full((s, 6, 3), 255, np.uint8)
img = np.concatenate([panels[0], sep, panels[1], sep, panels[2]], axis=1)
big = np.array(Image.fromarray(img).resize((img.shape[1]*2, img.shape[0]*2), Image.NEAREST))
Image.fromarray(big).save('.scratch/grain/panel_io_vs_libraw_vs_nx.png')
print('saved .scratch/grain/panel_io_vs_libraw_vs_nx.png', big.shape)