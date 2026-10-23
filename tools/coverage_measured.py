
import numpy as np
from collections import Counter

# positions measured by tools/pattern_probe2.py for series 106 (frame 0 = 0,0)
pos = [(0.0,0.0),(0.600,2.084),(2.048,1.340),(3.222,1.425),(2.797,2.230),(2.135,2.221),
       (2.314,1.387),(3.433,1.355),(3.480,1.786),(2.426,1.774),(2.337,0.751),(3.463,0.759),
       (3.966,1.833),(2.921,1.795),(2.873,0.803),(3.880,0.882)]
CFA = {(0,0):'R', (0,1):'G', (1,0):'G', (1,1):'B'}

def coverage(pos, scale, n=48):
    acc = {}
    for (dx, dy) in pos:
        for y in range(n):
            for x in range(n):
                X = scale*(x + dx); Y = scale*(y + dy)
                x0 = int(np.floor(X)); y0 = int(np.floor(Y))
                fx = X - x0; fy = Y - y0
                c = CFA[(y & 1, x & 1)]
                for ox, oy, w in ((0,0,(1-fx)*(1-fy)), (1,0,fx*(1-fy)), (0,1,(1-fx)*fy), (1,1,fx*fy)):
                    if w <= 0: continue
                    acc[(x0+ox, y0+oy, c)] = acc.get((x0+ox, y0+oy, c), 0.0) + w
    # interior stats
    lo, hi = scale*4, scale*(n-4)
    percol = {'R':[], 'G':[], 'B':[]}
    tot = []
    for Y in range(lo, hi):
        for X in range(lo, hi):
            s = 0.0
            for c in ('R','G','B'):
                w = acc.get((X, Y, c), 0.0)
                percol[c].append(w)
                s += w
            tot.append(s)
    return {c: (float(np.mean(v)), float(np.min(v)), float(np.percentile(v,5))) for c, v in percol.items()}, (float(np.mean(tot)), float(np.min(tot)), float(np.percentile(tot,5)))

print("measured 106 positions, scale 2:")
per, tot = coverage(pos, 2)
print("  per-colour weight: mean/min/p5 =", {c: tuple(round(x,3) for x in v) for c, v in per.items()})
print("  total per pixel: mean/min/p5 =", tuple(round(x,3) for x in tot))

ideal = [(sx, sy) for sx in (0,0.5,1,1.5) for sy in (0,0.5,1,1.5)]
print("ideal 0.5 lattice, scale 2:")
per, tot = coverage(ideal, 2)
print("  per-colour weight: mean/min/p5 =", {c: tuple(round(x,3) for x in v) for c, v in per.items()})
print("  total per pixel: mean/min/p5 =", tuple(round(x,3) for x in tot))

# coverage if we pretend the pattern is 4x4 at 1 px steps (what the measurements look like, minus jitter)
raster = [(float(i), float(j)) for j in range(4) for i in range(4)]
print("4x4 raster at 1 px, scale 2:")
per, tot = coverage(raster, 2)
print("  per-colour weight: mean/min/p5 =", {c: tuple(round(x,3) for x in v) for c, v in per.items()})
print("  total per pixel: mean/min/p5 =", tuple(round(x,3) for x in tot))
