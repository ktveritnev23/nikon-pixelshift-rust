import numpy as np, os
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

ours = read_ppm('.scratch/grain/104NZ6_3.dng.ppm')
ref = read_ppm('.scratch/grain/_DSC0522_merged.NEFX.ppm')

def spectrum_stats(a, name):
    h, w = a.shape[:2]
    y0, x0 = h//2 - 1024, w//2 - 1024
    win = np.asarray(a[y0:y0+2048, x0:x0+2048], dtype=np.float32)
    lum = win[...,0]*0.299 + win[...,1]*0.587 + win[...,2]*0.114
    rg = win[...,0] - win[...,1]
    for arr, label in ((lum, 'luma'), (rg, 'R-G')):
        arr = arr - arr.mean()
        hann = np.outer(np.hanning(arr.shape[0]), np.hanning(arr.shape[1]))
        F = np.abs(np.fft.rfft2(arr*hann))**2
        tot = F.sum()
        nyq = F[:, -1].sum()/tot
        hf = F[:, int(F.shape[1]*0.9):].sum()/tot
        print(f'{name} {label}: nyq_column_share={nyq:.4f} hf_column_share={hf:.4f}')
    yy, xx = np.mgrid[0:2048, 0:2048]
    chk = ((yy + xx) & 1) * 2.0 - 1.0
    for arr, label in ((lum, 'luma'), (rg, 'R-G')):
        c = float(np.mean((arr - arr.mean()) * chk))
        print(f'{name} {label}: checkerboard component = {c:.2f} ADU (mean {arr.mean():.0f})')

spectrum_stats(ref, 'NX  ')
spectrum_stats(ours, 'ours')