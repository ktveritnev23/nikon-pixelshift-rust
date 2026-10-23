
import struct, sys, os

def dump(path, max_ifd=8):
    f = open(path, 'rb'); hdr = f.read(8)
    endian = '<' if hdr[:2] == b'II' else '>'
    magic, off = struct.unpack(endian+'HL', hdr[2:8])
    sizes = {1:1,2:1,3:2,4:4,5:8,6:1,7:1,8:2,9:4,10:8,11:4,12:8,13:4}
    names = {0x00fe:'NewSubfileType',0x0100:'ImageWidth',0x0101:'ImageLength',0x0102:'BitsPerSample',0x0103:'Compression',
             0x0106:'PhotometricInterpretation',0x010f:'Make',0x0110:'Model',0x0111:'StripOffsets',0x0112:'Orientation',
             0x0115:'SamplesPerPixel',0x0116:'RowsPerStrip',0x0117:'StripByteCounts',0x011a:'XResolution',0x011b:'YResolution',
             0x011c:'PlanarConfiguration',0x0128:'ResolutionUnit',0x0131:'Software',0x0132:'DateTime',0x013b:'Artist',
             0x014a:'SubIFDs',0x0153:'SampleFormat',0x0201:'JPEGInterchangeFormat',0x0202:'JPEGInterchangeFormatLength',
             0x02bc:'XMP',0x8298:'Copyright',0x8769:'ExifIFD',0x8825:'GPSIFD',0x9003:'DateTimeOriginal',0x9216:'TIFF-EP',
             0x9c9b:'XPTitle',0xc612:'DNGVersion',0xc613:'DNGBackwardVersion',0xc614:'UniqueCameraModel',0xc615:'LocalizedCameraModel',
             0xc61a:'BlackLevel',0xc61d:'WhiteLevel',0xc61f:'DefaultCropOrigin',0xc620:'DefaultCropSize',0xc621:'ColorMatrix1',
             0xc622:'ColorMatrix2',0xc628:'AsShotNeutral',0xc62a:'BaselineExposure',0xc65a:'CalibrationIlluminant1',
             0xc65b:'CalibrationIlluminant2',0xc68d:'ActiveArea',0xc6f3:'CameraCalibration1',0xc6f4:'CameraCalibration2',
             0x829a:'ExposureTime',0x829d:'FNumber',0x8827:'ISO',0x920a:'FocalLength',0x9214:'SubjectArea',0xa405:'FocalLengthIn35mm',
             0x010e:'ImageDescription',0x011d:'PageName',0x011f:'TileOffsets',0x0142:'TileWidth',0x0143:'TileLength',
             0x0144:'TileByteCounts',0xc6f6:'OriginalRawFileName',0xc68e:'MaskedAreas',0xc616:'CFARepeatPatternDim',
             0xc617:'CFAPattern',0xc619:'BlackLevelRepeatDim',0xc61c:'BlackLevelDeltaH',0xc61e:'BlackLevelDeltaV',
             0xc634:'DNGPrivateData',0xc6f8:'OriginalRawFileData',0xc627:'AnalogBalance',0xc62b:'ExposureIndex'}
    print('==', os.path.basename(path), os.path.getsize(path), 'bytes')
    seen = set()
    queue = [(off, 'IFD0')]
    while queue:
        o, tag = queue.pop(0)
        if o in seen or o == 0: continue
        seen.add(o)
        f.seek(o); n = struct.unpack(endian+'H', f.read(2))[0]
        print('--', tag, 'at', o, 'entries', n)
        for i in range(n):
            e = f.read(12)
            t, typ, cnt = struct.unpack(endian+'HHL', e[:8])
            raw = e[8:12]
            sz = sizes.get(typ,1)*cnt
            if sz <= 4: buf = raw[:sz]
            else:
                vo = struct.unpack(endian+'L', raw)[0]
                if tag.startswith('IFD0') or tag.startswith('Sub'):
                    f.seek(vo); buf = f.read(min(sz, 64)); f.seek(o+2+i*12+12)
            if typ == 3: val = struct.unpack(endian+'%dH'%min(cnt,8), buf[:2*min(cnt,8)])
            elif typ == 4: val = struct.unpack(endian+'%dL'%min(cnt,4), buf[:4*min(cnt,4)])
            elif typ == 1: val = buf[:min(cnt,8)].hex()
            elif typ == 2: val = repr(buf.split(b'\x00')[0][:32])
            elif typ == 5: val = 'rational'
            else: val = buf[:8].hex()
            if t in (0x014a,):
                f.seek(struct.unpack(endian+'L', raw)[0]); arr = struct.unpack(endian+'%dL'%cnt, f.read(4*cnt))
                print('   %-26s %s' % (names.get(t, hex(t)), arr))
                for k, a in enumerate(arr): queue.append((a, 'SubIFD%d'%k))
                f.seek(o+2+i*12+12)
                continue
            if t in (0x8769, 0x8825):
                v = struct.unpack(endian+'L', raw)[0]
                queue.append((v, 'ExifIFD' if t == 0x8769 else 'GPS'))
            print('   %-26s  type=%-2d cnt=%-3d %s' % (names.get(t, '0x%04x' % t), typ, cnt, val))
        nxt = struct.unpack(endian+'L', f.read(4))[0]
        if nxt and tag == 'IFD0': queue.append((nxt, 'IFD1'))

for p in sys.argv[1:]:
    dump(p)
