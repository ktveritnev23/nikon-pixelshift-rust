
import numpy as np
from collections import Counter

# CFA: RGGB, colors[(y&1)][(x&1)]
CFA = {(0,0):'R', (0,1):'G', (1,0):'G', (1,1):'B'}

def sim(offsets, scale, sensor_w=16, sensor_h=16):
    acc = {}
    for (sx, sy) in offsets:
        for y in range(sensor_h):
            for x in range(sensor_w):
                X = int(round(scale * (x + sx)))
                Y = int(round(scale * (y + sy)))
                if abs(scale*(x+sx) - X) > 1e-9 or abs(scale*(y+sy) - Y) > 1e-9:
                    continue
                acc.setdefault((X, Y), []).append(CFA[(y & 1, x & 1)])
    # count colours per pixel in the interior
    W = scale * sensor_w; H = scale * sensor_h
    stat = Counter()
    for Y in range(scale*4, H - scale*4):
        for X in range(scale*4, W - scale*4):
            cols = acc.get((X, Y), [])
            stat[tuple(sorted(cols))] += 1
    return stat

grid16 = [(sx, sy) for sx in (0, 0.5, 1, 1.5) for sy in (0, 0.5, 1, 1.5)]
print("16 frames, scale 2, offsets {0,.5,1,1.5}^2:", sim(grid16, 2))
grid4 = [(0,0),(0,1),(1,0),(1,1)]
print("4 frames, scale 1, offsets {0,1}^2:", sim(grid4, 1))
grid8 = grid4 + [(0,0),(0,1),(1,0),(1,1)]
print("8 frames (duplicated), scale 1:", sim(grid8, 1))
grid32 = grid16 + grid16
print("32 frames (duplicated), scale 2:", sim(grid32, 2))
# hypothesis: 16 frames walking over a larger area, e.g. {0,0.5,1,1.5} in x but y only {0,1}
grid_alt = [(sx, sy) for sx in (0, 0.5, 1, 1.5) for sy in (0, 1, 2, 3)]
print("alt: sx in {0,.5,1,1.5}, sy in {0,1,2,3}, scale 2:", sim(grid_alt, 2))
