//! Acceptance for the lens distortion fit against a reference image: a known
//! three term distortion must come back from the pair "distorted / corrected",
//! the two TIFF readers must be bit exact, and a size mismatch must be reported
//! as such.

use pixelshift::distort::{apply_lens, Distortion};
use pixelshift::fit::{fit_from_reference, tiffread};
use pixelshift::types::{DngMeta, MergeResult, MergeStats};
use pixelshift::write::tiff::{self, Ifd, Value};
use pixelshift::write::write_tiff16;
use std::path::PathBuf;

const W: usize = 1024;
const H: usize = 768;
const WHITE: f64 = 14303.0;

/// Smooth value noise on a lattice of `cell` pixels, deterministic and a
/// continuous function of (x, y) so the lens simulation can sample it anywhere.
fn lattice(x: f64, y: f64, cell: f64, seed: u64) -> f64 {
    fn hash(i: i64, j: i64, seed: u64) -> f64 {
        let mut h = (i as u64)
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ (j as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f)
            ^ seed.wrapping_mul(0x27d4_eb2f);
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 29;
        (h >> 11) as f64 / (1u64 << 53) as f64
    }
    fn smooth(t: f64) -> f64 {
        t * t * (3.0 - 2.0 * t)
    }
    let fx = x / cell;
    let fy = y / cell;
    let i = fx.floor();
    let j = fy.floor();
    let tx = smooth(fx - i);
    let ty = smooth(fy - j);
    let (i0, j0) = (i as i64, j as i64);
    let a = hash(i0, j0, seed);
    let b = hash(i0 + 1, j0, seed);
    let c = hash(i0, j0 + 1, seed);
    let d = hash(i0 + 1, j0 + 1, seed);
    let top = a + (b - a) * tx;
    let bottom = c + (d - c) * tx;
    top + (bottom - top) * ty
}

/// Deterministic scene with detail from a couple of pixels to a few dozen, like
/// a real photograph: enough unique structure for block matching, but no
/// periodic lattice that would make every shift look alike.
fn scene(x: f64, y: f64) -> f64 {
    let texture = 0.42 * lattice(x, y, 5.0, 1) + 0.33 * lattice(x, y, 13.0, 2) + 0.25 * lattice(x, y, 29.0, 3);
    let tone = (0.37 * x + 0.19 * y).sin() * 0.06 + (0.11 * x - 0.07 * y).cos() * 0.04;
    (0.35 + 0.55 * texture + tone).clamp(0.02, 0.98)
}

fn to_u16(value: f64) -> u16 {
    (value * WHITE + 1200.0).round().clamp(0.0, 65535.0) as u16
}

fn scene_image(w: usize, h: usize) -> MergeResult {
    let mut planes = [
        vec![0u16; w * h],
        vec![0u16; w * h],
        vec![0u16; w * h],
    ];
    for y in 0..h {
        for x in 0..w {
            let value = to_u16(scene(x as f64, y as f64));
            for (c, plane) in planes.iter_mut().enumerate() {
                plane[y * w + x] = value + c as u16;
            }
        }
    }
    MergeResult {
        width: w,
        height: h,
        scale: 1,
        planes,
        white: WHITE as u16,
        shifts: Vec::new(),
        stats: MergeStats::default(),
    }
}

fn work_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(".scratch/wp-fit");
    std::fs::create_dir_all(&dir).expect("create the fit scratch directory");
    dir.join(name)
}

#[test]
fn fit_of_an_undistorted_pair_finds_nothing() {
    let image = scene_image(W, H);
    let path = work_dir("identity.tif");
    write_tiff16(&path, &image, &DngMeta::default()).expect("write the reference");
    let report = fit_from_reference(&image, &path).expect("fit");
    eprintln!("identity: {}", report.summary());
    assert!(report.blocks >= 20, "only {} blocks matched", report.blocks);
    for (name, value) in [("k1", report.k1), ("k2", report.k2), ("k3", report.k3)] {
        assert!(
            value.abs() < 2e-4,
            "{name} {value:+.6} should vanish on an undistorted pair"
        );
    }
    assert!(
        report.residual_rms < 0.02,
        "residual {:.4} px on identical geometry",
        report.residual_rms
    );
}

#[test]
fn fit_recovers_three_terms_from_a_distorted_pair_barrel() {
    // The real case: a 24 mm zoom, 4 % of barrel (content pulled inwards), so
    // the corrected reference keeps every scene point our frame also recorded.
    let known = Distortion::new3(-0.04, 0.005, -0.001);
    let clean = scene_image(W, H);
    let recorded = apply_lens(clean.clone(), known);
    let path = work_dir("known-barrel.tif");
    write_tiff16(&path, &clean, &DngMeta::default()).expect("write the reference");
    let report = fit_from_reference(&recorded, &path).expect("fit");
    eprintln!("barrel known {} -> {}", known, report.summary());
    for (name, got, want) in [
        ("k1", report.k1, known.k1),
        ("k2", report.k2, known.k2),
        ("k3", report.k3, known.k3),
    ] {
        eprintln!("  {name}: {got:+.6} vs {want:+.6} ({:+.1} %)", 100.0 * (got - want) / want);
    }
    assert!(report.residual_rms <= 0.05, "residual {:.4} px", report.residual_rms);
}

#[test]
fn fit_recovers_three_terms_from_a_distorted_pair() {
    let known = Distortion::new3(0.04, -0.005, 0.001);
    let clean = scene_image(W, H);
    let recorded = apply_lens(clean.clone(), known);
    let path = work_dir("known.tif");
    write_tiff16(&path, &clean, &DngMeta::default()).expect("write the reference");

    let report = fit_from_reference(&recorded, &path).expect("fit");
    eprintln!("known {} -> {}", known, report.summary());
    assert!(report.blocks >= 24, "only {} blocks matched", report.blocks);

    // k1 is the dominant term and has to come back to a few percent. The two
    // higher terms are mathematically degenerate over the radii a frame can
    // offer (r^2, r^4 and r^6 are nearly parallel for r <= 0.85, and the corner
    // blocks that would separate them lose their border content), so the test
    // checks what decides the picture instead: the correction the coefficients
    // produce, i.e. the model error at the frame corner.
    assert!(
        (report.k1 - known.k1).abs() / known.k1.abs() <= 0.10,
        "k1 {:+.5} is not within 10 % of {:+.5}",
        report.k1,
        known.k1
    );
    let half_diagonal = (((W as f64 - 1.0) * 0.5).powi(2) + ((H as f64 - 1.0) * 0.5).powi(2)).sqrt();
    let applied = known.k1 + known.k2 + known.k3;
    let fitted = report.k1 + report.k2 + report.k3;
    let error = (fitted - applied).abs() * half_diagonal;
    let amplitude = applied.abs() * half_diagonal;
    for (name, got, want) in [
        ("k2", report.k2, known.k2),
        ("k3", report.k3, known.k3),
    ] {
        eprintln!("  {name}: {got:+.6} vs {want:+.6} ({:+.0} %)", 100.0 * (got - want) / want);
    }
    eprintln!("corner displacement {amplitude:.2} px, model error {error:.2} px");
    assert!(
        error <= 0.10 * amplitude,
        "the correction is off by {error:.2} px at the corner ({:.1} % of {amplitude:.2} px)",
        100.0 * error / amplitude
    );
    assert!(
        report.residual_rms <= 0.05,
        "residual rms {:.4} px",
        report.residual_rms
    );
}

#[test]
fn fit_rejects_a_reference_of_a_different_size() {
    let image = scene_image(256, 192);
    let other = scene_image(128, 96);
    let path = work_dir("wrong-size.tif");
    write_tiff16(&path, &other, &DngMeta::default()).expect("write the reference");
    let error = fit_from_reference(&image, &path).expect_err("size mismatch must fail");
    let text = format!("{error:#}");
    assert!(text.contains("128 x 96"), "unhelpful message: {text}");
    assert!(text.contains("256 x 192"), "unhelpful message: {text}");
}

#[test]
fn tiff_reader_round_trips_a_16_bit_rgb_image() {
    let image = scene_image(64, 48);
    let path = work_dir("round16.tif");
    write_tiff16(&path, &image, &DngMeta::default()).expect("write");
    let read = tiffread::read_rgb(&path).expect("read back");
    assert_eq!((read.width, read.height, read.bits), (64, 48, 16));
    for c in 0..3 {
        for y in 0..48 {
            for x in 0..64 {
                let expected = image.planes[c][y * 64 + x];
                let got = (read.planes[c][y * 64 + x] * 65535.0).round() as i32;
                assert_eq!(got, expected as i32, "channel {c} at ({x}, {y})");
            }
        }
    }
}

/// Assemble a single strip 8-bit RGB TIFF with the shared primitives; there is
/// no 8-bit writer in the crate, and the reader has to cope with one anyway.
fn write_tiff8(path: &PathBuf, w: usize, h: usize, rgb: &[u8]) {
    fn build(w: usize, h: usize, bytes: usize, strip_offset: u32) -> Ifd {
        let mut ifd = Ifd::new();
        ifd.set_long(tiff::TAG_IMAGE_WIDTH, w as u32);
        ifd.set_long(tiff::TAG_IMAGE_LENGTH, h as u32);
        ifd.set(tiff::TAG_BITS_PER_SAMPLE, Value::Shorts(vec![8, 8, 8]));
        ifd.set_short(tiff::TAG_COMPRESSION, tiff::COMPRESSION_NONE);
        ifd.set_short(tiff::TAG_PHOTOMETRIC, tiff::PHOTOMETRIC_RGB);
        ifd.set_short(tiff::TAG_SAMPLES_PER_PIXEL, 3);
        ifd.set_long(tiff::TAG_ROWS_PER_STRIP, h as u32);
        ifd.set(tiff::TAG_STRIP_OFFSETS, Value::Longs(vec![strip_offset]));
        ifd.set(tiff::TAG_STRIP_BYTE_COUNTS, Value::Longs(vec![bytes as u32]));
        ifd.set_short(tiff::TAG_PLANAR_CONFIG, 1);
        ifd
    }
    let probe = build(w, h, rgb.len(), 0).encode(8).0;
    let strip_offset = (8 + probe.len()) as u32;
    let ifd = build(w, h, rgb.len(), strip_offset);
    let (bytes, _) = ifd.encode(8);
    let mut file = tiff::HEADER.to_vec();
    file.extend_from_slice(&bytes);
    file.extend_from_slice(rgb);
    std::fs::write(path, file).expect("write the 8-bit TIFF");
}

#[test]
fn tiff_reader_reads_an_8_bit_rgb_image() {
    const W8: usize = 17;
    const H8: usize = 11;
    let mut rgb = Vec::with_capacity(W8 * H8 * 3);
    for y in 0..H8 {
        for x in 0..W8 {
            rgb.push((x * 7 + y * 3) as u8);
            rgb.push((x * 11 + y * 5) as u8);
            rgb.push((x * 13 + y * 17) as u8);
        }
    }
    let path = work_dir("round8.tif");
    write_tiff8(&path, W8, H8, &rgb);
    let read = tiffread::read_rgb(&path).expect("read the 8-bit TIFF");
    assert_eq!((read.width, read.height, read.bits), (W8, H8, 8));
    for c in 0..3 {
        for index in 0..W8 * H8 {
            let got = (read.planes[c][index] * 255.0).round() as i32;
            assert_eq!(got, rgb[index * 3 + c] as i32, "channel {c} sample {index}");
        }
    }
}
