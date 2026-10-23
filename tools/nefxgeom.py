
import struct, sys, os
def val_of(f, typ, cnt, vo, raw):
    sizes={1:1,2:1,3:2,4:4,5:8,6:1,7:1,8:2,9:4,10:8,11:4,12:8}
    sz = sizes.get(typ,1)*cnt
    if sz<=4: buf = raw[:sz]
    else:
        f.seek(vo); buf = f.read(sz)
    if typ==3: return struct.unpack('<%dH'%cnt, buf)
    if typ==4: return struct.unpack('<%dL'%cnt, buf)
    if typ==2: return buf.split(b'\x00')[0].decode('latin1')
    return buf.hex()[:64]

for path in sys.argv[1:]:
    total = os.path.getsize(path)
    f = open(path,'rb'); f.read(8)
    off0 = struct.unpack('<L', open(path,'rb').read(8)[4:8])[0]
    f.seek(off0); n = struct.unpack('<H', f.read(2))[0]
    ents={}
    for i in range(n):
        e=f.read(12); tag,typ,cnt=struct.unpack('<HHL', e[:8]); vo=struct.unpack('<L',e[8:12])[0]
        ents[tag]=(typ,cnt,vo)
    subs=[]
    if 0x014a in ents:
        typ,cnt,vo=ents[0x014a]; f.seek(vo); subs=list(struct.unpack('<%dL'%cnt, f.read(4*cnt)))
    print("==", os.path.basename(path), "size", total)
    for s in subs:
        f.seek(s); n2=struct.unpack('<H', f.read(2))[0]
        d={}
        for i in range(n2):
            e=f.read(12); tag,typ,cnt=struct.unpack('<HHL', e[:8]); vo=struct.unpack('<L',e[8:12])[0]
            d[tag]=val_of(f, typ, cnt, vo, e[8:12]); f.seek(s+2+i*12+12)
        w=d.get(0x0100); h=d.get(0x0101); bps=d.get(0x0102); spp=d.get(0x0115)
        if w and h and (w[0] if isinstance(w,tuple) else w)>1000:
            print("  FULL IFD @%d: %s x %s bits=%s spp=%s comp=%s photometric=%s stripbytes=%s cfa=%s" % (
                s, w, h, bps, spp, d.get(0x0103), d.get(0x0106), d.get(0x0117), d.get(0x828e)))
        elif w and h:
            print("  preview @%d: %sx%s bits=%s spp=%s" % (s, w, h, bps, spp))
