import numpy as np, os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
x0, y0, s = 3400, 160, 320
def norm8(a):
    a = np.asarray(a, dtype=np.float32)
    p = float(np.percentile(a, 99.0))
    return (np.clip(a/max(p,1e-9), 0, 1)**(1/2.2)*255).astype(np.uint8)
panels = []
names = []
for p, label in (('.scratch/grain/io_104.png', 'ImageIO LinearRaw'),
                 ('.scratch/grain/pi2.png', 'ImageIO PI=2'),
                 ('.scratch/grain/104NZ6_3.dng.ppm', 'LibRaw LinearRaw')):
    if p.endswith('.ppm'):
        f = open(p, 'rb'); assert f.readline().strip() == b'P6'; line = f.readline()
        while line.startswith(b'#'): line = f.readline()
        w, h = map(int, line.split()); maxv = int(f.readline()); off = f.tell(); f.close()
        a = np.memmap(p, dtype='>u2', mode='r', offset=off, shape=(h, w, 3))
        crop = np.asarray(a[y0:y0+s, x0:x0+s], dtype=np.float32)
    else:
        img = np.array(Image.open(p))
        crop = img[y0:y0+s, x0:x0+s]
    print(label, crop.shape, crop.dtype, 'mean', [round(float(v)) for v in crop.reshape(-1,3).mean(0)])
    panels.append(norm8(crop))
    names.append(label)
sep = np.full((s, 6, 3), 255, np.uint8)
img = np.concatenate([panels[0], sep, panels[1], sep, panels[2]], axis=1)
big = np.array(Image.fromarray(img).resize((img.shape[1]*2, img.shape[0]*2), Image.NEAREST))
Image.fromarray(big).save('.scratch/grain/panel_pi2.png')
print('saved', big.shape, names)