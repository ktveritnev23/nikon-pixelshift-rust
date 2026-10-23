import numpy as np, os, struct
from scipy.ndimage import map_coordinates
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def read_ppm(path):
    f = open(path, 'rb'); assert f.readline().strip() == b'P6'
    line = f.readline()
    while line.startswith(b'#'): line = f.readline()
    w, h = map(int, line.split()); maxv = int(f.readline()); off = f.tell(); f.close()
    assert maxv == 65535
    return np.memmap(path, dtype='>u2', mode='r', offset=off, shape=(h, w, 3))

def write_tiff16_rgb(path, arr):
    h, w = arr.shape[:2]
    data = np.ascontiguousarray(arr.astype('<u2'))
    entries = []
    extra = b''
    def add(tag, typ, count, value_bytes=None, value=None):
        entries.append((tag, typ, count, value_bytes, value))
    # offsets are patched after we know the layout
    n = 12
    ifd_off = 8
    ifd_size = 2 + n * 12 + 4
    bps_off = ifd_off + ifd_size
    bps = struct.pack('<3H', 16, 16, 16)
    data_off = bps_off + len(bps)
    add(256, 4, 1, value=w)
    add(257, 4, 1, value=h)
    add(258, 3, 3, value_bytes=struct.pack('<L', bps_off))
    add(259, 3, 1, value=1)
    add(262, 3, 1, value=2)
    add(273, 4, 1, value=data_off)
    add(277, 3, 1, value=3)
    add(278, 4, 1, value=h)
    add(279, 4, 1, value=data.nbytes)
    add(284, 3, 1, value=1)
    add(339, 3, 1, value=1)
    add(305, 2, 0, value=0)
    buf = bytearray()
    buf += b'II' + struct.pack('<HL', 42, ifd_off)
    val = bytearray()
    val += struct.pack('<H', n)
    for tag, typ, count, value_bytes, value in entries:
        val += struct.pack('<HHL', tag, typ, count)
        if value_bytes is not None:
            val += value_bytes
        else:
            val += struct.pack('<L', value)
    val += struct.pack('<L', 0)
    out = bytes(buf) + bytes(val) + bps + data.tobytes()
    open(path, 'wb').write(out)

def distort_to_reference(distorted, k1, k2, k3, rnorm=None):
    h, w = distorted.shape[:2]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    if rnorm is None:
        rnorm = np.hypot(cx, cy)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dy = (yy - cy); dx = (xx - cx)
    r2 = (dx * dx + dy * dy) / (rnorm * rnorm)
    scale = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    src_y = cy + dy * scale
    src_x = cx + dx * scale
    out = np.empty_like(distorted)
    for c in range(3):
        out[..., c] = map_coordinates(np.asarray(distorted[..., c], dtype=np.float32), [src_y, src_x], order=3, mode='nearest')
    return np.clip(out, 0, 65535).astype(np.uint16)

src = '.scratch/fit/ours_1x.ppm'
os.makedirs('.scratch/fit', exist_ok=True)
if not os.path.exists(src):
    raise SystemExit('run: dcraw_emu -4 -o 0 -W out/108NZ6_3.dng -> ' + src)
d = read_ppm(src)
print('source', d.shape)
ref = distort_to_reference(d, -0.040, -0.004, 0.0006)
write_tiff16_rgb('.scratch/fit/reference_k1-0040.tif', ref)
print('wrote .scratch/fit/reference_k1-0040.tif', ref.shape)