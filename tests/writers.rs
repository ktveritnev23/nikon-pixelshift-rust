//! WP1 acceptance: the DNG and 16-bit TIFF writers are checked with external
//! tools (LibRaw's raw-identify / dcraw_emu, libtiff's tiffinfo, Pillow,
//! ImageMagick, macOS ImageIO) so that a green test means "other software reads
//! this file", not merely "we wrote some bytes".

use pixelshift::types::{DngMeta, MergeResult, MergeStats, WriteOpts};
use pixelshift::write::{
    write_dng, write_dng_with_flavor, write_tiff16, write_tiff16_rendered, DngFlavor,
};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

const WIDTH: usize = 512;
const HEIGHT: usize = 384;
const WHITE: u16 = 15375;
const PREVIEW: u32 = 64;

/// Independent TIFF decoder (python + numpy only): structure, per channel
/// statistics and a checksum, and zlib inflation of Deflate strips.
const PY_TIFF: &str = r#"
import sys, json, struct, zlib
import numpy as np

TYPE_SIZE = {1:1,2:1,3:2,4:4,5:8,6:1,7:1,8:2,9:4,10:8,11:8,12:8}

def parse_ifd(data, off):
    n = struct.unpack_from('<H', data, off)[0]
    tags = {}
    for i in range(n):
        e = off + 2 + 12 * i
        tag, typ, cnt = struct.unpack_from('<HHI', data, e)
        size = TYPE_SIZE[typ] * cnt
        if size <= 4:
            raw = data[e + 8:e + 8 + size]
        else:
            p = struct.unpack_from('<I', data, e + 8)[0]
            raw = data[p:p + size]
        if typ == 3:
            val = list(struct.unpack('<%dH' % cnt, raw))
        elif typ == 4:
            val = list(struct.unpack('<%dI' % cnt, raw))
        else:
            val = list(raw)
        tags[tag] = (typ, cnt, val)
    nxt = struct.unpack_from('<I', data, off + 2 + 12 * n)[0]
    return tags, nxt

def image(path, mode):
    data = open(path, 'rb').read()
    assert data[:4] == b'II*\x00', 'not a little endian classic tiff'
    off = struct.unpack_from('<I', data, 4)[0]
    tags, nxt = parse_ifd(data, off)
    if mode == 'next':
        assert nxt != 0, 'no next IFD'
        off = nxt
        tags, nxt = parse_ifd(data, off)
    elif mode == 'jpeg':
        assert nxt != 0, 'no preview IFD'
        off = nxt
        tags, nxt = parse_ifd(data, off)
        assert nxt != 0, 'no JPEG preview IFD'
        off = nxt
        tags, nxt = parse_ifd(data, off)
    w = tags[256][2][0]; h = tags[257][2][0]
    bps = tags[258][2]; spp = tags[277][2][0]
    comp = tags[259][2][0]; rps = tags[278][2][0]
    offsets = tags[273][2]; counts = tags[279][2]
    if comp == 7:
        # JPEG preview: the strip is a complete JFIF stream, not raw samples.
        arr = np.zeros((1, 1, spp), dtype=np.uint8)
        return {'width': w, 'height': h, 'samples': spp, 'bits': bps[0],
                'photometric': tags[262][2][0], 'new_subfile_type': tags[254][2][0],
                'compression': comp, 'rows_per_strip': rps,
                'strips': len(offsets), 'next_ifd': nxt,
                'sub_ifds': tags[330][2] if 330 in tags else None,
                'cfa_repeat': None, 'cfa_pattern': None, 'cfa_plane_color': None,
                'cfa_layout': None, 'black_level_count': None,
                'white_level_count': None, 'ifd_offset': off,
                'bytes': len(data)}, arr
    chunks = []
    for o, c in zip(offsets, counts):
        blob = data[o:o + c]
        if comp == 8 or comp == 32946:
            blob = zlib.decompress(blob)
        elif comp != 1:
            raise SystemExit('unsupported compression %d' % comp)
        chunks.append(blob)
    raw = b''.join(chunks)
    dtype = '<u2' if bps[0] == 16 else np.uint8
    arr = np.frombuffer(raw, dtype=dtype)[: w * h * spp].reshape(h, w, spp)
    return {'width': w, 'height': h, 'samples': spp, 'bits': bps[0],
            'photometric': tags[262][2][0], 'new_subfile_type': tags[254][2][0],
            'compression': comp, 'rows_per_strip': rps,
            'strips': len(offsets), 'next_ifd': nxt,
            'sub_ifds': tags[330][2] if 330 in tags else None,
            'cfa_repeat': tags[33421][2] if 33421 in tags else None,
            'cfa_pattern': tags[33422][2] if 33422 in tags else None,
            'cfa_plane_color': tags[50710][2] if 50710 in tags else None,
            'cfa_layout': tags[50711][2] if 50711 in tags else None,
            'black_level_count': tags[50714][1] if 50714 in tags else None,
            'white_level_count': tags[50717][1] if 50717 in tags else None,
            'ifd_offset': off, 'bytes': len(data)}, arr

def stats(arr):
    h, w, c = arr.shape
    out = {}
    out['min'] = [int(arr[:, :, i].min()) for i in range(c)]
    out['max'] = [int(arr[:, :, i].max()) for i in range(c)]
    out['mean'] = [float(arr[:, :, i].astype(np.float64).mean()) for i in range(c)]
    out['sum'] = [int(arr[:, :, i].astype(np.uint64).sum()) for i in range(c)]
    idx = np.arange(1, h * w + 1, dtype=np.uint64)
    out['checksum'] = [int((arr[:, :, i].reshape(-1).astype(np.uint64) * idx).sum())
                       for i in range(c)]
    out['total_sum'] = int(arr.reshape(-1).astype(np.uint64).sum())
    out['unique'] = int(np.unique(arr).size)
    if c == 1:
        out['quad'] = [[int(arr[qy::2, qx::2].astype(np.uint64).sum()) for qx in range(2)]
                       for qy in range(2)]
    else:
        out['quad'] = None
    return out

info, arr = image(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'ifd0')
print(json.dumps({'info': info, 'stats': stats(arr)}))
"#;

fn python_json(path: &Path) -> serde_json::Value {
    python_json_mode(path, None)
}

/// `mode = Some("next")` makes the reader follow the next-IFD pointer, which is
/// where the DNG keeps the 8-bit preview; the main image is IFD0.
fn python_json_mode(path: &Path, mode: Option<&str>) -> serde_json::Value {
    let mut command = Command::new("python3");
    command.arg("-c").arg(PY_TIFF).arg(path);
    if let Some(mode) = mode {
        command.arg(mode);
    }
    let out = command.output().expect("python3 must be installed");
    assert!(
        out.status.success(),
        "python failed on {}:\n{}",
        path.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).expect("python produced invalid json")
}

fn tool(name: &str, args: &[&str]) -> (bool, String) {
    let out = Command::new(name)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("cannot run {name}: {e}"));
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    (out.status.success(), text)
}

fn out_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".scratch/wp1/test-out");
    std::fs::create_dir_all(&dir).expect("create output directory");
    dir
}

/// Gradients plus flat patches: every channel has its own mean and range, so a
/// channel swap or a wrong sample layout cannot slip through.
fn synthetic() -> MergeResult {
    let mut planes = [
        vec![0u16; WIDTH * HEIGHT],
        vec![0u16; WIDTH * HEIGHT],
        vec![0u16; WIDTH * HEIGHT],
    ];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let i = y * WIDTH + x;
            let mut rgb = [
                (x as f32 / (WIDTH - 1) as f32 * WHITE as f32) as u16,
                (0.2 * WHITE as f32) as u16,
                (0.8 * WHITE as f32) as u16,
            ];
            if x < 32 && y >= HEIGHT - 32 {
                rgb = [100, 200, 300];
            } else if x >= WIDTH - 32 && y < 32 {
                rgb = [4000, 8000, 12000];
            }
            for c in 0..3 {
                planes[c][i] = rgb[c];
            }
        }
    }
    MergeResult {
        width: WIDTH,
        height: HEIGHT,
        scale: 1,
        planes,
        white: WHITE,
        shifts: Vec::new(),
        stats: MergeStats::default(),
    }
}

fn meta() -> DngMeta {
    DngMeta {
        make: "Nikon".to_string(),
        model: "Z6_3".to_string(),
        timestamp: "2026:10:23 18:58:30".to_string(),
        software: "pixelshift test".to_string(),
        // Neutral multipliers of one keep the dcraw round trip predictable.
        as_shot_neutral: [1.0, 1.0, 1.0],
        color_matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        calib_illuminant: 21,
        baseline_exposure: 0.0,
        iso: 2000.0,
        shutter: 1.0 / 40.0,
        aperture: 4.0,
        focal: 135.0,
        orientation: 1,
        wb_applied: false,
        lens_make: "VILTROX".to_string(),
        lens_model: "VILTROX 135mm F1.8 Z".to_string(),
        focal_35mm: 135,
        max_aperture: 1.8,
        lens_min_focal: 135.0,
        lens_max_focal: 135.0,
        lens_min_fnum: 1.8,
        lens_max_fnum: 1.8,
        lens_serial: String::new(),
        body_serial: String::new(),
        makernote: Vec::new(),
    }
}

fn opts() -> WriteOpts {
    WriteOpts {
        compress: false,
        preview_max: PREVIEW,
    }
}

/// (sum, checksum, min, max) of one plane, computed the same way as the python
/// reference decoder.
fn plane_stats(v: &[u16]) -> (u64, u64, u16, u16) {
    let mut sum = 0u64;
    let mut checksum = 0u64;
    let mut min = u16::MAX;
    let mut max = 0u16;
    for (i, value) in v.iter().enumerate() {
        sum += *value as u64;
        checksum += (*value as u64).wrapping_mul(i as u64 + 1);
        min = min.min(*value);
        max = max.max(*value);
    }
    (sum, checksum, min, max)
}

fn compare_planes_exactly(img: &MergeResult, json: &serde_json::Value) {
    let stats = &json["stats"];
    for c in 0..3 {
        let (sum, checksum, min, max) = plane_stats(&img.planes[c]);
        assert_eq!(
            stats["sum"][c].as_u64().unwrap(),
            sum,
            "channel {c}: sum of decoded samples differs"
        );
        assert_eq!(
            stats["checksum"][c].as_u64().unwrap(),
            checksum,
            "channel {c}: decoded samples differ from the merged image"
        );
        assert_eq!(
            stats["min"][c].as_u64().unwrap(),
            min as u64,
            "channel {c} min"
        );
        assert_eq!(
            stats["max"][c].as_u64().unwrap(),
            max as u64,
            "channel {c} max"
        );
    }
}

/// Which plane feeds a mosaic pixel under the RGGB pattern written by the CFA
/// flavour.
fn mosaic_plane(x: usize, y: usize) -> usize {
    match (y & 1, x & 1) {
        (0, 0) => 0,
        (1, 1) => 2,
        _ => 1,
    }
}

fn compare_mosaic_exactly(img: &MergeResult, json: &serde_json::Value) {
    let stats = &json["stats"];
    let mut total = 0u64;
    let mut checksum = 0u64;
    let mut quad = [[0u64; 2]; 2];
    let mut i = 0usize;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let value = img.planes[mosaic_plane(x, y)][y * WIDTH + x] as u64;
            total += value;
            checksum += value.wrapping_mul(i as u64 + 1);
            quad[y & 1][x & 1] += value;
            i += 1;
        }
    }
    assert_eq!(stats["total_sum"].as_u64().unwrap(), total, "mosaic sum");
    assert_eq!(
        stats["checksum"][0].as_u64().unwrap(),
        checksum,
        "mosaic samples differ"
    );
    for y in 0..2 {
        for x in 0..2 {
            assert_eq!(
                stats["quad"][y][x].as_u64().unwrap(),
                quad[y][x],
                "mosaic quadrant ({y},{x}) comes from the wrong plane"
            );
        }
    }
}

/// IFD0 must carry the full resolution image and IFD1 the 8-bit preview.
fn assert_layout(path: &Path, flavor: DngFlavor, main_samples: u64) {
    let main = python_json_mode(path, None);
    let info = &main["info"];
    assert_eq!(info["width"].as_u64(), Some(WIDTH as u64));
    assert_eq!(info["height"].as_u64(), Some(HEIGHT as u64));
    assert_eq!(info["bits"].as_u64(), Some(16));
    assert_eq!(info["samples"].as_u64(), Some(main_samples));
    assert_eq!(
        info["photometric"].as_u64(),
        Some(flavor.photometric() as u64),
        "IFD0 photometric"
    );
    assert_eq!(info["new_subfile_type"].as_u64(), Some(0));
    assert_eq!(info["rows_per_strip"].as_u64(), Some(16));
    assert_eq!(
        info["black_level_count"].as_u64(),
        Some(main_samples),
        "BlackLevel count must equal SamplesPerPixel"
    );
    assert_eq!(
        info["white_level_count"].as_u64(),
        Some(main_samples),
        "WhiteLevel count must equal SamplesPerPixel"
    );
    assert_eq!(
        info["strips"].as_u64(),
        Some((HEIGHT as u64).div_ceil(16)),
        "strip count"
    );
    assert!(
        info["sub_ifds"].is_null(),
        "no SubIFD must be used: {}",
        info["sub_ifds"]
    );
    assert_ne!(info["next_ifd"].as_u64(), Some(0), "preview IFD missing");

    let preview = python_json_mode(path, Some("next"));
    let info = &preview["info"];
    let expected_preview_w = WIDTH as u64 / (WIDTH as u64).div_ceil(PREVIEW as u64);
    let expected_preview_h = HEIGHT as u64 / (WIDTH as u64).div_ceil(PREVIEW as u64);
    assert_eq!(info["width"].as_u64(), Some(expected_preview_w));
    assert_eq!(info["height"].as_u64(), Some(expected_preview_h));
    assert_eq!(info["bits"].as_u64(), Some(8));
    assert_eq!(info["samples"].as_u64(), Some(3));
    assert_eq!(info["photometric"].as_u64(), Some(2), "preview PI");
    assert_eq!(info["new_subfile_type"].as_u64(), Some(1), "preview type");
    assert!(info["sub_ifds"].is_null());

    // IFD2: JPEG preview (Compression 7, YCbCr), long side JPEG_PREVIEW_MAX.
    let jpeg = python_json_mode(path, Some("jpeg"));
    let info = &jpeg["info"];
    let jpeg_step = (WIDTH as u64).div_ceil(1600).max(1);
    assert_eq!(info["width"].as_u64(), Some(WIDTH as u64 / jpeg_step));
    assert_eq!(info["height"].as_u64(), Some(HEIGHT as u64 / jpeg_step));
    assert_eq!(info["bits"].as_u64(), Some(8));
    assert_eq!(info["samples"].as_u64(), Some(3));
    assert_eq!(info["photometric"].as_u64(), Some(6), "JPEG preview PI");
    assert_eq!(info["compression"].as_u64(), Some(7), "JPEG preview compression");
    assert_eq!(info["new_subfile_type"].as_u64(), Some(1));
    assert_eq!(info["strips"].as_u64(), Some(1));
}

#[test]
fn dng_main_image_is_ifd0_and_preview_is_ifd1_for_every_flavor() {
    let img = synthetic();
    for (flavor, samples) in [
        (DngFlavor::LinearRaw, 3u64),
        (DngFlavor::Rgb, 3),
        (DngFlavor::Cfa, 1),
    ] {
        let path = out_dir().join(format!("wp1_layout_{}.dng", flavor_label(flavor)));
        write_dng_with_flavor(&path, &img, &meta(), &opts(), flavor).expect("write dng");
        assert_layout(&path, flavor, samples);
    }
}

fn flavor_label(flavor: DngFlavor) -> &'static str {
    match flavor {
        DngFlavor::LinearRaw => "linearraw",
        DngFlavor::Rgb => "rgb",
        DngFlavor::Cfa => "cfa",
    }
}

#[test]
fn writes_dng_that_libraw_decodes() {
    let img = synthetic();
    let path = out_dir().join("wp1_linear.dng");
    write_dng(&path, &img, &meta(), &opts()).expect("write dng");
    eprintln!("dng: {} bytes", std::fs::metadata(&path).unwrap().len());

    // 1. LibRaw recognises the file as a raw image with the right geometry.
    let (ok, identify) = tool("raw-identify", &["-v", path.to_str().unwrap()]);
    assert!(ok, "raw-identify failed:\n{identify}");
    eprintln!("--- raw-identify ---\n{}", identify.trim());
    assert!(
        identify.contains("DNG Version: 1.4.0.0"),
        "missing DNGVersion:\n{identify}"
    );
    assert!(
        identify.contains("Raw colors: 3"),
        "LibRaw sees no RGB raw:\n{identify}"
    );
    assert!(
        identify.contains("Full size:") && identify.contains("512 x 384"),
        "unexpected full size:\n{identify}"
    );
    assert!(
        identify.contains("DNG Illuminant 1: D65"),
        "no DNG color metadata:\n{identify}"
    );

    // 2. libtiff sees the LinearRaw main image in IFD0 and no SubIFD.
    let (ok, tiffinfo) = tool("tiffinfo", &["-d", path.to_str().unwrap()]);
    assert!(ok, "tiffinfo failed:\n{tiffinfo}");
    eprintln!("--- tiffinfo ---\n{}", tiffinfo.trim());
    assert!(
        !tiffinfo.contains("SubIFD"),
        "SubIFD must not be used:\n{tiffinfo}"
    );
    assert!(
        tiffinfo.contains("Photometric Interpretation: 34892"),
        "main IFD is not LinearRaw:\n{tiffinfo}"
    );
    assert!(
        tiffinfo.contains("Image Width: 512 Image Length: 384"),
        "wrong main size"
    );
    assert!(
        tiffinfo.contains("Rows/Strip: 16"),
        "strips are not 16 rows"
    );
    assert!(tiffinfo.contains("WhiteLevel: 15375"), "wrong white level");
    assert!(
        tiffinfo.contains("DNGVersion: 1,4,0,0"),
        "wrong DNG version"
    );
    // The preview must be a real 8-bit RGB image in the second directory.
    assert!(
        tiffinfo.contains("Photometric Interpretation: RGB color")
            && tiffinfo.contains("Bits/Sample: 8"),
        "preview is not an 8 bit RGB image:\n{tiffinfo}"
    );
    assert_layout(&path, DngFlavor::LinearRaw, 3);

    // 3. dcraw_emu (LibRaw) decodes the raw data without errors.
    // -w: use the camera (here: AsShotNeutral) white balance, so that the
    // decode is a plain per-channel scaling of what we stored.
    let (ok, decode) = tool(
        "dcraw_emu",
        &["-T", "-4", "-o", "0", "-w", path.to_str().unwrap()],
    );
    eprintln!("--- dcraw_emu ---\n{}", decode.trim());
    assert!(ok, "dcraw_emu failed:\n{decode}");
    assert!(
        !decode.contains("Cannot"),
        "dcraw_emu reported an error:\n{decode}"
    );

    let decoded = PathBuf::from(format!("{}.tiff", path.display()));
    assert!(decoded.exists(), "dcraw_emu produced no tiff");

    let json = python_json(&decoded);
    let info = &json["info"];
    assert_eq!(info["width"].as_u64(), Some(WIDTH as u64));
    assert_eq!(info["height"].as_u64(), Some(HEIGHT as u64));
    assert_eq!(info["samples"].as_u64(), Some(3));
    assert_eq!(info["bits"].as_u64(), Some(16));

    // LibRaw applies its own scaling (65535/white) and the identity matrix, so
    // the channels must all be scaled by the same factor and stay ordered.
    let mut ratios = [0.0f64; 3];
    for c in 0..3 {
        let source_mean =
            img.planes[c].iter().map(|v| *v as f64).sum::<f64>() / (WIDTH * HEIGHT) as f64;
        let decoded_mean = json["stats"]["mean"][c].as_f64().unwrap();
        ratios[c] = decoded_mean / source_mean;
        eprintln!(
            "channel {c}: source mean {source_mean:.1} -> decoded mean {decoded_mean:.1} (x{:.4})",
            ratios[c]
        );
    }
    for c in 1..3 {
        assert!(
            (ratios[c] - ratios[0]).abs() / ratios[0] < 0.01,
            "channels are scaled differently, decoded data is wrong: {ratios:?}"
        );
    }
    assert!(
        ratios[0] > 1.0 && ratios[0] < 8.0,
        "unexpected dcraw scaling {:.3}",
        ratios[0]
    );
}

#[test]
fn cfa_flavor_is_a_valid_bayer_raw() {
    let img = synthetic();
    let path = out_dir().join("wp1_cfa.dng");
    write_dng_with_flavor(&path, &img, &meta(), &opts(), DngFlavor::Cfa).expect("write cfa dng");

    let (ok, tiffinfo) = tool("tiffinfo", &["-d", path.to_str().unwrap()]);
    assert!(ok, "tiffinfo failed:\n{tiffinfo}");
    eprintln!("--- cfa tiffinfo ---\n{}", tiffinfo.trim());
    assert!(tiffinfo.contains("Photometric Interpretation: 32803"));
    assert!(tiffinfo.contains("Samples/Pixel: 1"));
    // libtiff marks the DNG-only CFA tags of unknown type with an "EP " prefix.
    assert!(
        tiffinfo.contains("CFARepeatPatternDim: 2,2"),
        "CFARepeatPatternDim missing:\n{tiffinfo}"
    );
    assert!(
        tiffinfo.contains("CFAPattern: 0,1,1,2"),
        "CFAPattern missing:\n{tiffinfo}"
    );

    let json = python_json(&path);
    let info = &json["info"];
    assert_eq!(info["samples"].as_u64(), Some(1));
    assert_eq!(info["bits"].as_u64(), Some(16));
    assert_eq!(
        info["cfa_repeat"].as_array().map(|v| v.len()),
        Some(2),
        "CFARepeatPatternDim"
    );
    assert_eq!(
        info["cfa_pattern"].as_array().map(|v| v.len()),
        Some(4),
        "CFAPattern"
    );
    for (i, expected) in [0, 1, 1, 2].iter().enumerate() {
        assert_eq!(
            info["cfa_pattern"][i].as_u64(),
            Some(*expected),
            "CFAPattern[{i}]"
        );
    }
    assert_eq!(info["cfa_plane_color"].as_array().map(|v| v.len()), Some(3));
    assert_eq!(info["cfa_layout"][0].as_u64(), Some(1), "CFALayout");
    compare_mosaic_exactly(&img, &json);

    // LibRaw sees a one channel raw with the right size and decodes it.
    let (ok, identify) = tool("raw-identify", &["-v", path.to_str().unwrap()]);
    assert!(ok, "raw-identify failed:\n{identify}");
    eprintln!("--- cfa raw-identify ---\n{}", identify.trim());
    assert!(
        identify.contains("Filter pattern: RGGB"),
        "LibRaw does not see the RGGB mosaic:\n{identify}"
    );
    assert!(identify.contains("Full size:") && identify.contains("512 x 384"), "{identify}");
    let (ok, decode) = tool(
        "dcraw_emu",
        &["-T", "-4", "-o", "0", "-w", path.to_str().unwrap()],
    );
    assert!(ok && !decode.contains("Cannot"), "dcraw_emu failed:\n{decode}");
    let decoded = PathBuf::from(format!("{}.tiff", path.display()));
    let json = python_json(&decoded);
    assert_eq!(json["info"]["width"].as_u64(), Some(WIDTH as u64));
    assert_eq!(json["info"]["samples"].as_u64(), Some(3));
    assert!(
        json["stats"]["unique"].as_u64().unwrap() > 100,
        "LibRaw output is flat"
    );

    // ImageMagick opens the CFA master without a codec complaint.
    let (ok, magick) = tool("magick", &["identify", path.to_str().unwrap()]);
    assert!(ok, "ImageMagick failed:\n{magick}");
    assert!(magick.contains("512x384"), "{magick}");
}

#[test]
fn rgb_flavor_is_plain_16_bit_rgb() {
    let img = synthetic();
    let path = out_dir().join("wp1_rgb.dng");
    write_dng_with_flavor(&path, &img, &meta(), &opts(), DngFlavor::Rgb).expect("write rgb dng");

    let (ok, tiffinfo) = tool("tiffinfo", &["-d", path.to_str().unwrap()]);
    assert!(ok, "tiffinfo failed:\n{tiffinfo}");
    assert!(tiffinfo.contains("Photometric Interpretation: RGB color"));
    assert!(tiffinfo.contains("Bits/Sample: 16"));
    assert!(tiffinfo.contains("Samples/Pixel: 3"));

    let json = python_json(&path);
    compare_planes_exactly(&img, &json);

    // ImageMagick routes every .dng through its LibRaw delegate, which refuses a
    // non-raw DNG; forcing the TIFF coder proves the payload itself is a valid
    // 16-bit RGB TIFF (and that is what ImageIO reads).
    let (ok, forced) = tool(
        "magick",
        &["identify", &format!("TIFF:{}", path.display())],
    );
    assert!(ok, "ImageMagick TIFF coder failed:\n{forced}");
    assert!(forced.contains("512x384"), "{forced}");
    let (ok, default_coder) = tool("magick", &["identify", path.to_str().unwrap()]);
    assert!(
        !ok && default_coder.contains("not RAW"),
        "expected the LibRaw delegate to reject the .dng extension:\n{default_coder}"
    );
}

/// ImageIO is what Preview / Finder / Pages use; sips is its command line front
/// end. It decodes a 1-channel CFA DNG (demosaicing it itself) but not a
/// 3-channel LinearRaw or plain RGB DNG, which is exactly why the cfa flavour
/// exists.
#[cfg(target_os = "macos")]
#[test]
fn imageio_reads_the_cfa_flavor() {
    let img = synthetic();
    let Some(real) = real_frame_meta() else {
        eprintln!("skipping: the workspace carries no NEF sample");
        return;
    };

    let path = out_dir().join("wp1_sips_cfa.dng");
    write_dng_with_flavor(&path, &img, &real, &opts(), DngFlavor::Cfa).expect("write dng");
    let (ok, size) = tool(
        "sips",
        &[
            "-g",
            "pixelWidth",
            "-g",
            "pixelHeight",
            path.to_str().unwrap(),
        ],
    );
    eprintln!("--- sips cfa ---\n{}", size.trim());
    assert!(ok, "sips failed on the cfa DNG:\n{size}");
    assert!(
        size.contains("pixelWidth: 512") && size.contains("pixelHeight: 384"),
        "ImageIO geometry for cfa:\n{size}"
    );
    let png = out_dir().join("wp1_sips_cfa.png");
    let _ = std::fs::remove_file(&png);
    let (ok, out) = tool(
        "sips",
        &[
            "-s",
            "format",
            "png",
            path.to_str().unwrap(),
            "--out",
            png.to_str().unwrap(),
        ],
    );
    assert!(ok && png.exists(), "sips could not render cfa:\n{out}");
    let (ok, pil) = tool(
        "python3",
        &[
            "-c",
            "import sys; from PIL import Image; im = Image.open(sys.argv[1]); \
             px = im.convert('RGB').resize((8, 8)).getdata(); \
             print(im.size, im.getextrema())",
            png.to_str().unwrap(),
        ],
    );
    assert!(ok, "Pillow cannot read the sips png:\n{pil}");
    eprintln!("--- sips png cfa ---\n{}", pil.trim());

    // ImageIO does not decode the plain RGB flavour: it reports no dimensions.
    // The rgb DNG stays useful for readers that treat a DNG as a TIFF container.
    let path = out_dir().join("wp1_sips_rgb.dng");
    write_dng_with_flavor(&path, &img, &real, &opts(), DngFlavor::Rgb).expect("write dng");
    let (_, size) = tool(
        "sips",
        &[
            "-g",
            "pixelWidth",
            "-g",
            "pixelHeight",
            path.to_str().unwrap(),
        ],
    );
    eprintln!("--- sips rgb ---\n{}", size.trim());
    assert!(
        size.contains("<nil>"),
        "ImageIO unexpectedly decodes the rgb DNG now: {size}"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn imageio_reads_the_rendered_tiff() {
    let img = synthetic();
    let path = out_dir().join("wp1_rendered_sips.tiff");
    write_tiff16_rendered(&path, &img, &meta()).expect("write tiff");
    let (ok, size) = tool(
        "sips",
        &[
            "-g",
            "pixelWidth",
            "-g",
            "pixelHeight",
            path.to_str().unwrap(),
        ],
    );
    assert!(ok, "sips failed:\n{size}");
    assert!(size.contains("pixelWidth: 512"), "{size}");
}

/// Camera metadata of the reference NEF (LibRaw cam_xyz / as-shot neutral of a
/// Z6 III), i.e. what the application writes in real use.
/// First raw sample that is present in the workspace; the series are private
/// data and a checkout may carry none of them.
fn available_nef() -> Option<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&root)
        .ok()?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect();
    dirs.sort();
    for dir in dirs {
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
            .ok()?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension()
                    .map(|ext| ext.eq_ignore_ascii_case("nef"))
                    .unwrap_or(false)
            })
            .collect();
        files.sort();
        if let Some(first) = files.into_iter().next() {
            return Some(first);
        }
    }
    None
}

fn real_frame_meta() -> Option<DngMeta> {
    use pixelshift::rawio;

    let nef = available_nef()?;
    let frame = rawio::open(&nef).expect("decode reference NEF");

    let mut meta = meta();
    meta.make = frame.make.clone();
    meta.model = frame.model.clone();
    meta.timestamp = frame.timestamp.clone();
    meta.iso = frame.iso;
    meta.shutter = frame.shutter;
    meta.aperture = frame.aperture;
    meta.focal = frame.focal;
    meta.color_matrix = [
        [
            frame.cam_xyz[0][0],
            frame.cam_xyz[0][1],
            frame.cam_xyz[0][2],
        ],
        [
            frame.cam_xyz[1][0],
            frame.cam_xyz[1][1],
            frame.cam_xyz[1][2],
        ],
        [
            frame.cam_xyz[2][0],
            frame.cam_xyz[2][1],
            frame.cam_xyz[2][2],
        ],
    ];
    meta.as_shot_neutral = [1.0 / frame.wb[0], 1.0, 1.0 / frame.wb[2]];
    Some(meta)
}

/// The embedded JPEG preview must be a decodable JFIF stream: that is what
/// RapidRAW, Finder and libraw's preview extractor use.

/// Lens identification must reach IFD0: raw editors (Lightroom, Capture One)
/// pick their own lens profile from it, and without the tags no editor can
/// compensate the distortion of a third party lens.
#[test]
fn writes_lens_metadata_into_the_main_ifd() {
    let img = synthetic();
    let path = out_dir().join("wp1_lens.dng");
    write_dng(&path, &img, &meta(), &opts()).expect("write dng");

    let (ok, text) = tool(
        "python3",
        &[
            "-c",
            r#"
import struct, sys
data = open(sys.argv[1], 'rb').read()
def ifd(o):
    n = struct.unpack_from('<H', data, o)[0]
    tags = {}
    for i in range(n):
        e = o + 2 + 12 * i
        tag, typ, cnt = struct.unpack_from('<HHI', data, e)
        size = {1:1,2:1,3:2,4:4,5:8,6:1,7:1,8:2,9:4,10:8,11:8,12:8}[typ] * cnt
        raw = data[e+8:e+8+size] if size <= 4 else data[struct.unpack_from('<I', data, e+8)[0]:][:size]
        tags[tag] = (typ, cnt, raw)
    return tags
main = ifd(struct.unpack_from('<I', data, 4)[0])
assert 0x8769 in main, 'IFD0 has no Exif IFD pointer'
t = ifd(struct.unpack('<I', main[0x8769][2][:4])[0])
def text_tag(tag):
    return t[tag][2].rstrip(b'\x00').decode('ascii')
def rationals(tag):
    typ, cnt, raw = t[tag]
    return [struct.unpack_from('<II', raw, 8*i) for i in range(cnt)]
assert text_tag(0xA433) == 'VILTROX', text_tag(0xA433)
assert text_tag(0xA434) == 'VILTROX 135mm F1.8 Z', text_tag(0xA434)
assert t[0xA405][1] == 1 and t[0xA405][2][0] == 135, t[0xA405]
focal = rationals(0x920A)[0]
assert abs(focal[0] / focal[1] - 135.0) < 0.01, focal
apex = rationals(0x9205)[0]
value = apex[0] / apex[1]
assert abs(value - 2.0 * __import__('math').log2(1.8)) < 0.01, apex
spec = rationals(0xA432)
print('lens', text_tag(0xA433), '|', text_tag(0xA434))
print('focal', focal[0]/focal[1], 'focal35', t[0xA405][2][0])
print('maxaperture_apex %.3f (f/%0.1f)' % (value, 2 ** (value / 2.0)))
print('lensspec', [(n / d) for n, d in spec])
"#,
            path.to_str().unwrap(),
        ],
    );
    assert!(ok, "lens tags missing or wrong:
{text}");
    eprintln!("--- lens tags ---
{}", text.trim());

    // tiffinfo: libtiff reads the same values. It prints the EXIF tags without
    // their names outside an ExifIFD, so the check is on the tag numbers.
    let (ok, text) = tool("tiffinfo", &[path.to_str().unwrap()]);
    assert!(ok, "tiffinfo failed:
{text}");
    for needle in [
        "EXIF directory within directory 0",
        "LensMake: VILTROX",
        "LensModel: VILTROX 135mm F1.8 Z",
        "FocalLengthIn35mmFilm: 135",
        "LensSpecification: 135.000000,135.000000,1.800000,1.800000",
        "MaxApertureValue: 1.696000",
        "FocalLength: 135.000000",
    ] {
        assert!(text.contains(needle), "tiffinfo does not show {needle}:
{text}");
    }
    let lens_lines: Vec<&str> = text
        .lines()
        .filter(|line| line.contains("Lens") || line.contains("Aperture") || line.contains("Focal"))
        .collect();
    eprintln!("--- tiffinfo ---
{}", lens_lines.join("\n"));

    // raw-identify reads the DNG through LibRaw, i.e. the path Capture One
    // style importers take.
    let (ok, text) = tool("raw-identify", &["-v", path.to_str().unwrap()]);
    if ok {
        let lens_lines: Vec<&str> = text
            .lines()
            .filter(|line| {
                let lowered = line.to_ascii_lowercase();
                lowered.contains("lens") || lowered.contains("aperture") || lowered.contains("focal")
            })
            .collect();
        eprintln!("--- raw-identify ---
{}", lens_lines.join("\n"));
        assert!(
            lens_lines.iter().any(|line| line.contains("135")),
            "raw-identify does not mention the lens focal length:
{text}"
        );
    }
}

/// The full EXIF path a raw editor takes: a real Exif IFD (0x8769) inside IFD0,
/// the four values of a zoom LensSpecification, the serial numbers and the
/// maker note copied byte for byte from the source file.
#[test]
fn writes_exif_ifd_with_a_zoom_specification_and_the_maker_note() {
    let img = synthetic();
    let mut meta = meta();
    meta.focal = 24.0;
    meta.aperture = 4.0;
    meta.max_aperture = 4.0;
    meta.focal_35mm = 24;
    meta.lens_make = "NIKON".to_string();
    meta.lens_model = "NIKKOR Z 24-120mm f/4 S".to_string();
    meta.lens_min_focal = 24.0;
    meta.lens_max_focal = 120.0;
    meta.lens_min_fnum = 4.0;
    meta.lens_max_fnum = 4.0;
    meta.body_serial = "7016932".to_string();
    meta.lens_serial = "20312694".to_string();
    // The writer treats the maker note as opaque bytes; 5000 of them are an
    // out of line value and must come back unchanged.
    meta.makernote = (0..5000u32).map(|i| (i % 251) as u8).collect();
    let path = out_dir().join("wp1_zoom_exif.dng");
    write_dng(&path, &img, &meta, &opts()).expect("write dng");

    let (ok, text) = tool(
        "python3",
        &[
            "-c",
            r#"
import struct, sys
data = open(sys.argv[1], 'rb').read()
def ifd(o):
    n = struct.unpack_from('<H', data, o)[0]
    tags = {}
    for i in range(n):
        e = o + 2 + 12 * i
        tag, typ, cnt = struct.unpack_from('<HHI', data, e)
        size = {1:1,2:1,3:2,4:4,5:8,6:1,7:1,8:2,9:4,10:8,11:8,12:8}[typ] * cnt
        raw = data[e+8:e+8+size] if size <= 4 else data[struct.unpack_from('<I', data, e+8)[0]:][:size]
        tags[tag] = (typ, cnt, raw)
    return tags
main = ifd(struct.unpack_from('<I', data, 4)[0])
assert 0x8769 in main, 'IFD0 has no Exif IFD pointer'
for tag in (0x829A, 0x829D, 0x8827, 0x9003, 0x920A):
    assert tag not in main, 'EXIF tag %04X is still in IFD0' % tag
exif = ifd(struct.unpack('<I', main[0x8769][2][:4])[0])
spec = [struct.unpack_from('<II', exif[0xA432][2], 8 * i) for i in range(4)]
values = [n / float(d) for n, d in spec]
assert values == [24.0, 120.0, 4.0, 4.0], values
typ, cnt, raw = exif[0x927C]
assert typ == 7, 'maker note type %d' % typ
assert cnt == 5000, cnt
expected = bytes([i % 251 for i in range(5000)])
assert raw == expected, 'maker note bytes changed'
assert exif[0xA431][2].rstrip(b'\x00').decode() == '7016932'
assert exif[0xA435][2].rstrip(b'\x00').decode() == '20312694'
assert exif[0x9003][2].rstrip(b'\x00').decode() == '2026:10:23 18:58:30'
assert exif[0x9004][2].rstrip(b'\x00').decode() == '2026:10:23 18:58:30'
print('exif ok', values, cnt)
"#,
            path.to_str().unwrap(),
        ],
    );
    assert!(ok, "EXIF layout is wrong:\n{text}");
    eprintln!("--- exif ---\n{}", text.trim());
}
#[test]
fn embedded_jpeg_preview_decodes() {
    let img = synthetic();
    let path = out_dir().join("wp1_jpeg_preview.dng");
    write_dng(&path, &img, &meta(), &opts()).expect("write dng");

    let (ok, text) = tool(
        "python3",
        &[
            "-c",
            r#"
import struct, sys
data = open(sys.argv[1], 'rb').read()
def ifd(o):
    n = struct.unpack_from('<H', data, o)[0]
    tags = {}
    for i in range(n):
        e = o + 2 + 12 * i
        tag, typ, cnt = struct.unpack_from('<HHI', data, e)
        size = {1:1,2:1,3:2,4:4,5:8,6:1,7:1,8:2,9:4,10:8,11:8,12:8}[typ] * cnt
        raw = data[e+8:e+8+size] if size <= 4 else data[struct.unpack_from('<I', data, e+8)[0]:][:size]
        tags[tag] = (typ, cnt, raw)
    return tags, struct.unpack_from('<I', data, o + 2 + 12 * n)[0]
o = struct.unpack_from('<I', data, 4)[0]
tags, nxt = ifd(o)
tags, nxt = ifd(nxt)
tags, nxt = ifd(nxt)
off = struct.unpack('<I', tags[273][2][:4])[0]
count = struct.unpack('<I', tags[279][2][:4])[0]
blob = data[off:off+count]
assert blob[:2] == bytes([0xff, 0xd8]) and blob[-2:] == bytes([0xff, 0xd9]), 'not a JFIF stream'
open(sys.argv[2], 'wb').write(blob)
from PIL import Image
im = Image.open(sys.argv[2])
im.load()
px = im.convert('RGB').getdata()
print(im.size, im.mode, max(max(p) for p in px), min(min(p) for p in px))
"#,
            path.to_str().unwrap(),
            out_dir().join("wp1_jpeg_preview.jpg").to_str().unwrap(),
        ],
    );
    assert!(ok, "JPEG preview could not be decoded:
{text}");
    eprintln!("--- embedded jpeg ---
{}", text.trim());
    assert!(text.contains("(512, 384)"), "JPEG preview size:
{text}");
    assert!(text.contains("RGB"), "JPEG preview mode:
{text}");
}

/// The real camera metadata (LibRaw cam_xyz / as-shot neutral of a Z6 III) must
/// survive the DNG writer and be understood by LibRaw as a colour matrix.
#[test]
fn writes_dng_with_real_camera_metadata() {
    use pixelshift::rawio;

    let Some(nef) = available_nef() else {
        eprintln!("skipping: the workspace carries no NEF sample");
        return;
    };
    let frame = rawio::open(&nef).expect("decode reference NEF");

    let mut img = synthetic();
    img.width = 64;
    img.height = 48;
    for (c, plane) in img.planes.iter_mut().enumerate() {
        plane.clear();
        plane.resize(64 * 48, 0);
        for (i, value) in plane.iter_mut().enumerate() {
            *value = (1000 + c * 500 + (i % 64) * 10) as u16;
        }
    }
    img.white = frame.white_span();

    let Some(meta) = real_frame_meta() else {
        eprintln!("skipping: the workspace carries no NEF sample");
        return;
    };

    let path = out_dir().join("wp1_real.dng");
    write_dng(
        &path,
        &img,
        &meta,
        &WriteOpts {
            compress: false,
            preview_max: 64,
        },
    )
    .expect("write dng with real metadata");

    let (ok, tiffinfo) = tool("tiffinfo", &["-d", path.to_str().unwrap()]);
    assert!(ok, "tiffinfo failed:\n{tiffinfo}");
    let matrix = meta.color_matrix[0];
    let expected_matrix = format!(
        "ColorMatrix1: {:.6},{:.6},{:.6}",
        matrix[0], matrix[1], matrix[2]
    );
    assert!(
        tiffinfo.contains(&expected_matrix),
        "camera matrix was not written:\n{tiffinfo}\nexpected {expected_matrix}"
    );
    let neutral_line = tiffinfo
        .lines()
        .find(|line| line.trim_start().starts_with("AsShotNeutral:"))
        .unwrap_or_else(|| panic!("as-shot neutral was not written:\n{tiffinfo}"));
    let parsed: Vec<f32> = neutral_line
        .split(':')
        .nth(1)
        .unwrap_or_default()
        .split(',')
        .filter_map(|value| value.trim().parse().ok())
        .collect();
    assert_eq!(parsed.len(), 3, "unexpected neutral line: {neutral_line}");
    for (got, want) in parsed.iter().zip(meta.as_shot_neutral) {
        assert!(
            (got - want).abs() <= 1e-3,
            "neutral {got} != {want} in {neutral_line}"
        );
    }
    assert!(
        tiffinfo.contains("UniqueCameraModel: NIKON Z6_3"),
        "UniqueCameraModel must not repeat the vendor:\n{tiffinfo}"
    );
    assert!(tiffinfo.contains("Make: NIKON CORPORATION"));
    assert!(tiffinfo.contains("Model: NIKON Z6_3"));
    // Count = SamplesPerPixel on both level tags; rawler asserts exactly that.
    assert!(
        tiffinfo.contains("BlackLevel: 0.000000,0.000000,0.000000"),
        "BlackLevel must have one value per channel:\n{tiffinfo}"
    );
    assert!(
        tiffinfo.contains("WhiteLevel: 15375,15375,15375"),
        "WhiteLevel must have one value per channel:\n{tiffinfo}"
    );
    assert!(tiffinfo.contains("WhiteLevel: 15375"));

    let (ok, identify) = tool("raw-identify", &["-v", path.to_str().unwrap()]);
    assert!(ok, "raw-identify failed:\n{identify}");
    assert!(identify.contains("DNG color matrix 1:"));
    assert!(
        identify.contains("1.1206"),
        "LibRaw did not read the matrix:\n{identify}"
    );

    let (ok, decode) = tool(
        "dcraw_emu",
        &["-T", "-4", "-o", "0", "-w", path.to_str().unwrap()],
    );
    assert!(ok && !decode.contains("Cannot"), "dcraw_emu failed:\n{decode}");
}

#[test]
fn dng_deflate_strips_are_valid_zlib() {
    let img = synthetic();
    let path = out_dir().join("wp1_deflate.dng");
    let opts = WriteOpts {
        compress: true,
        preview_max: 200,
    };
    write_dng(&path, &img, &meta(), &opts).expect("write compressed dng");

    let (ok, tiffinfo) = tool("tiffinfo", &["-d", path.to_str().unwrap()]);
    assert!(ok, "tiffinfo failed:\n{tiffinfo}");
    assert!(
        tiffinfo.contains("Compression Scheme: AdobeDeflate"),
        "deflate flag not honoured:\n{tiffinfo}"
    );
    assert!(tiffinfo.contains("Rows/Strip: 16"));

    // The python reader inflates the strips with zlib and must get the exact
    // merged samples back from the main image in IFD0.
    let json = python_json_mode(&path, None);
    assert_eq!(json["info"]["compression"].as_u64(), Some(8));
    assert_eq!(
        json["info"]["strips"].as_u64(),
        Some((HEIGHT as u64).div_ceil(16))
    );
    compare_planes_exactly(&img, &json);
    eprintln!(
        "deflate dng: {} bytes for {} raw bytes",
        std::fs::metadata(&path).unwrap().len(),
        WIDTH * HEIGHT * 6
    );
}

/// Not part of the default suite: writes a full 96 MP DNG (12096x8064) to show
/// that the writer streams strips and never holds the file in memory.
/// Run manually: `cargo test --test writers -- --ignored writes_96mp`.
#[test]
#[ignore = "writes a 558 MB file, run manually for the memory check"]
fn writes_96mp_dng_by_streaming_strips() {
    const BIG_W: usize = 12096;
    const BIG_H: usize = 8064;
    let mut planes = [
        vec![0u16; BIG_W * BIG_H],
        vec![0u16; BIG_W * BIG_H],
        vec![0u16; BIG_W * BIG_H],
    ];
    for y in 0..BIG_H {
        for x in 0..BIG_W {
            let i = y * BIG_W + x;
            planes[0][i] = (x % 15000) as u16;
            planes[1][i] = (y % 15000) as u16;
            planes[2][i] = ((x + y) % 15000) as u16;
        }
    }
    let img = MergeResult {
        width: BIG_W,
        height: BIG_H,
        scale: 2,
        planes,
        white: WHITE,
        shifts: Vec::new(),
        stats: MergeStats::default(),
    };
    let path = out_dir().join("wp1_96mp.dng");
    let started = Instant::now();
    write_dng(
        &path,
        &img,
        &meta(),
        &WriteOpts {
            compress: false,
            preview_max: 0,
        },
    )
    .expect("write 96 mp dng");
    let bytes = std::fs::metadata(&path).unwrap().len();
    eprintln!(
        "96 MP dng: {bytes} bytes in {:.2}s",
        started.elapsed().as_secs_f64()
    );
    // Uncompressed payload plus the preview plus a few KB of directories.
    assert!(
        bytes >= (BIG_W * BIG_H * 6) as u64,
        "96 MP file is too small: {bytes}"
    );
    let (ok, info) = tool("tiffinfo", &[path.to_str().unwrap()]);
    assert!(ok, "tiffinfo failed on the 96 MP file");
    assert!(info.contains("Image Width: 12096 Image Length: 8064"));
    std::fs::remove_file(&path).expect("remove the 558 MB probe file");
}

#[test]
fn writes_tiff16_that_python_and_imagemagick_read() {
    let img = synthetic();
    let path = out_dir().join("wp1_linear.tiff");
    write_tiff16(&path, &img, &meta()).expect("write tiff");

    let (ok, tiffinfo) = tool("tiffinfo", &[path.to_str().unwrap()]);
    assert!(ok, "tiffinfo failed:\n{tiffinfo}");
    eprintln!("--- tiffinfo ---\n{}", tiffinfo.trim());
    assert!(tiffinfo.contains("Image Width: 512 Image Length: 384"));
    assert!(tiffinfo.contains("Bits/Sample: 16"));
    assert!(tiffinfo.contains("Photometric Interpretation: RGB color"));
    assert!(tiffinfo.contains("Samples/Pixel: 3"));
    assert!(tiffinfo.contains("Rows/Strip: 16"));
    assert!(tiffinfo.contains("Compression Scheme: None"));

    // Pillow opens it with the right geometry, ImageMagick reports the same
    // channel statistics as the merged image.
    let (ok, pil) = tool(
        "python3",
        &[
            "-c",
            "from PIL import Image; import sys; im = Image.open(sys.argv[1]); \
             print(im.size, im.mode, im.format)",
            path.to_str().unwrap(),
        ],
    );
    assert!(ok, "Pillow failed:\n{pil}");
    eprintln!("--- Pillow ---\n{}", pil.trim());
    assert!(pil.contains("(512, 384)"), "Pillow geometry: {pil}");

    let json = python_json(&path);
    assert_eq!(json["info"]["bits"].as_u64(), Some(16));
    compare_planes_exactly(&img, &json);

    let (ok, magick) = tool("magick", &["identify", "-verbose", path.to_str().unwrap()]);
    assert!(ok, "ImageMagick failed:\n{magick}");
    assert!(
        magick.contains("Geometry: 512x384"),
        "ImageMagick geometry:\n{magick}"
    );
    assert!(magick.contains("Depth: 16-bit"), "depth");
    assert!(magick.contains("Endianness: LSB"), "endianness");
    for c in 0..3 {
        let (sum, _, _, _) = plane_stats(&img.planes[c]);
        let mean = sum as f64 / (WIDTH * HEIGHT) as f64;
        // ImageMagick lists Red, Green, Blue in order; compare its raw mean.
        // ImageMagick prints the raw 16 bit mean, e.g. "mean: 3085.7 (...)".
        let needle = format!("mean: {}", mean as u64);
        assert!(
            magick.contains(&needle),
            "ImageMagick channel {c} mean is not {mean:.0}:\n{magick}"
        );
    }
}

#[test]
fn rendered_tiff_looks_like_a_normal_srgb_picture() {
    let img = synthetic();
    let path = out_dir().join("wp1_rendered.tiff");
    write_tiff16_rendered(&path, &img, &meta()).expect("write rendered tiff");

    let (ok, tiffinfo) = tool("tiffinfo", &[path.to_str().unwrap()]);
    assert!(ok, "tiffinfo failed:\n{tiffinfo}");
    assert!(tiffinfo.contains("Bits/Sample: 16"));
    assert!(tiffinfo.contains("Photometric Interpretation: RGB color"));
    assert!(tiffinfo.contains("Samples/Pixel: 3"));

    let json = python_json(&path);
    let stats = &json["stats"];
    let means: Vec<f64> = (0..3)
        .map(|c| stats["mean"][c].as_f64().unwrap())
        .collect();
    let mins: Vec<u64> = (0..3).map(|c| stats["min"][c].as_u64().unwrap()).collect();
    let maxs: Vec<u64> = (0..3).map(|c| stats["max"][c].as_u64().unwrap()).collect();
    eprintln!("rendered means {means:?} min {mins:?} max {maxs:?}");
    // A real tone curve: the data must not sit in one bin and must use most of
    // the 16 bit range.
    assert!(
        stats["unique"].as_u64().unwrap() > 100,
        "histogram is degenerate: {stats}"
    );
    for c in 0..3 {
        assert!(
            maxs[c] > 30_000 && maxs[c] <= 65_535,
            "channel {c} max {}: the render is not gamma encoded",
            maxs[c]
        );
        assert!(means[c] > 1_000.0, "channel {c} is black");
    }
    // The tone ordering of the synthetic image must survive: B (0.8 white) is
    // brighter than G (0.2 white) in the source.
    assert!(means[2] > means[1], "B should be brighter than G: {means:?}");

    // Pillow and ImageMagick open it as an ordinary picture.
    let (ok, pil) = tool(
        "python3",
        &[
            "-c",
            "from PIL import Image; import sys; im = Image.open(sys.argv[1]); \
             print(im.size, im.mode); im.load(); print('loaded')",
            path.to_str().unwrap(),
        ],
    );
    assert!(ok, "Pillow failed:\n{pil}");
    assert!(pil.contains("(512, 384)"), "{pil}");
    assert!(pil.contains("loaded"), "{pil}");

    let (ok, magick) = tool("magick", &["identify", "-verbose", path.to_str().unwrap()]);
    assert!(ok, "ImageMagick failed:\n{magick}");
    assert!(magick.contains("Geometry: 512x384"), "{magick}");
    assert!(magick.contains("Depth: 16-bit"));
}
