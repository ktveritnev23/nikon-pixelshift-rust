//! Acceptance for the lens distortion correction: exactness of the identity,
//! recovery of a known distortion, absence of folds at the largest accepted
//! coefficient and the parse/label behaviour of the coefficients.

use pixelshift::distort::{apply_lens, rectify, Distortion, MAX_K1};
use pixelshift::types::{MergeResult, MergeStats};

const W: usize = 512;
const H: usize = 384;

/// Deterministic scene with detail down to a few pixels: the interpolation of
/// the correction is measured against this, not against a resampled copy.
fn scene(x: f64, y: f64) -> f64 {
    let s = (0.71 * x + 0.23 * y).sin() * 0.5
        + (0.31 * x - 0.53 * y).sin() * 0.35
        + (1.13 * x + 0.07 * y).cos() * 0.15;
    0.5 + 0.25 * s
}

fn to_u16(value: f64) -> u16 {
    (value * 12000.0 + 1200.0).round().clamp(0.0, 65535.0) as u16
}

/// Ideal (undistorted) image sampled on the pixel grid.
fn clean_image() -> MergeResult {
    let mut planes = [
        vec![0u16; W * H],
        vec![0u16; W * H],
        vec![0u16; W * H],
    ];
    for y in 0..H {
        for x in 0..W {
            for plane in planes.iter_mut() {
                plane[y * W + x] = to_u16(scene(x as f64, y as f64));
            }
        }
    }
    MergeResult {
        width: W,
        height: H,
        scale: 1,
        planes,
        white: 14303,
        shifts: Vec::new(),
        stats: MergeStats::default(),
    }
}

/// Real radius of an ideal radius under the forward lens map r' = f(r),
/// solved with a fixed point iteration (f stays monotonic up to |k1| = 1/3,
/// so every coefficient MAX_K1 accepts converges).
fn inverse_radius(d: Distortion, r: f64) -> f64 {
    let mut u = r;
    for _ in 0..40 {
        u = r / d.radial_scale(u * u);
    }
    u
}

/// Simulate the lens: the recorded pixel at radius r shows the ideal point at
/// radius f^-1(r). Sampled from the analytic scene, so it carries no
/// interpolation error of its own.
fn distorted_image(d: Distortion) -> MergeResult {
    let cx = (W as f64 - 1.0) * 0.5;
    let cy = (H as f64 - 1.0) * 0.5;
    let half_diag = (cx * cx + cy * cy).sqrt();
    let mut img = clean_image();
    for y in 0..H {
        for x in 0..W {
            let dx = x as f64 - cx;
            let dy = y as f64 - cy;
            let r = (dx * dx + dy * dy).sqrt() / half_diag;
            let scale = if r > 0.0 {
                inverse_radius(d, r) / r
            } else {
                1.0
            };
            let value = to_u16(scene(cx + dx * scale, cy + dy * scale));
            for plane in img.planes.iter_mut() {
                plane[y * W + x] = value;
            }
        }
    }
    img
}

/// RMS of the difference to the ideal image over the region where the
/// correction does not have to clamp at the border.
fn rms_against_ideal(img: &MergeResult) -> f64 {
    let ideal = clean_image();
    let cx = (W as f64 - 1.0) * 0.5;
    let cy = (H as f64 - 1.0) * 0.5;
    let limit = 0.6 * (cx * cx + cy * cy).sqrt();
    let mut sum = 0.0f64;
    let mut count = 0.0f64;
    for y in 0..H {
        for x in 0..W {
            let dx = x as f64 - cx;
            let dy = y as f64 - cy;
            if (dx * dx + dy * dy).sqrt() > limit {
                continue;
            }
            for c in 0..3 {
                let a = img.planes[c][y * W + x] as f64;
                let b = ideal.planes[c][y * W + x] as f64;
                sum += (a - b) * (a - b);
                count += 1.0;
            }
        }
    }
    (sum / count.max(1.0)).sqrt()
}

#[test]
fn identity_is_bit_exact() {
    let img = clean_image();
    let before = img.planes.clone();
    let out = rectify(img, Distortion::IDENTITY);
    assert_eq!(out.planes, before);
    assert_eq!((out.width, out.height), (W, H));

    // Same for coefficients that are exactly zero through the constructor.
    let img = clean_image();
    let before = img.planes.clone();
    let out = rectify(img, Distortion::new(0.0, 0.0));
    assert_eq!(out.planes, before);
}

#[test]
fn k2_and_k3_are_zero_by_default() {
    assert_eq!((Distortion::default().k2, Distortion::default().k3), (0.0, 0.0));
    let parsed = Distortion::parse("0.008").expect("parse k1");
    assert_eq!((parsed.k1, parsed.k2, parsed.k3), (0.008, 0.0, 0.0));
    let parsed = Distortion::parse("0.008,0.0002").expect("parse k1,k2");
    assert_eq!((parsed.k1, parsed.k2, parsed.k3), (0.008, 0.0002, 0.0));
    let parsed = Distortion::parse("0.008,0.0002,-0.00001").expect("parse k1,k2,k3");
    assert_eq!((parsed.k1, parsed.k2, parsed.k3), (0.008, 0.0002, -0.00001));
    assert!(Distortion::parse("0.5").is_err(), "a coefficient must stay bounded");
    assert!(Distortion::parse("0.001,0.5").is_err(), "k2 must stay bounded too");
    assert!(Distortion::parse("x").is_err());
    assert!(
        Distortion::parse("0.001,0.002,0.003,0.004").is_err(),
        "a fourth term does not exist"
    );
}

#[test]
fn apply_lens_is_the_inverse_of_rectify() {
    let d = Distortion::new3(-0.04, 0.005, -0.001);
    let ideal = clean_image();
    let recorded = apply_lens(ideal.clone(), d);
    assert!(recorded.planes != ideal.planes, "the lens must change pixels");
    let corrected = rectify(recorded, d);
    let error = rms_against_ideal(&corrected);
    eprintln!("lens {d} -> back: rms {error:.1} ADU");
    assert!(error < 40.0, "round trip rms {error:.1} ADU");
    // Identity is a no-op in both directions.
    let ideal = clean_image();
    let before = ideal.planes.clone();
    assert_eq!(apply_lens(ideal, Distortion::IDENTITY).planes, before);
}

#[test]
fn rectify_removes_a_known_distortion() {
    let k1 = 0.01;
    let d = Distortion::new(k1, 0.0);
    let distorted = distorted_image(d);
    let error_before = rms_against_ideal(&distorted);
    let corrected = rectify(distorted, d);
    let error_after = rms_against_ideal(&corrected);
    eprintln!(
        "k1 {k1}: rms {:.1} -> {:.1} ADU (factor {:.1})",
        error_before,
        error_after,
        error_before / error_after.max(1e-9)
    );
    assert!(
        error_before > 100.0,
        "the simulated distortion must actually hurt: {error_before:.1}"
    );
    assert!(
        error_after * 5.0 < error_before,
        "correction must cut the error at least 5x: {error_before:.1} -> {error_after:.1}"
    );

    // Re-fitting k1 on the corrected image must find essentially nothing.
    let distorted = distorted_image(d);
    let mut best = (f64::INFINITY, 0.0);
    for step in -10..=10 {
        let delta = k1 * 0.005 * step as f64;
        let candidate = rectify(distorted.clone(), Distortion::new(k1 + delta, 0.0));
        let error = rms_against_ideal(&candidate);
        if error < best.0 {
            best = (error, delta);
        }
    }
    eprintln!("re-fit residual k1 offset {:.5} (of {k1})", best.1);
    assert!(
        best.1.abs() <= 0.05 * k1,
        "residual k1 {:.5} is more than 5 % of the applied {k1}",
        best.1
    );
}

#[test]
fn no_folds_up_to_the_largest_accepted_coefficient() {
    let mut cases = vec![
        Distortion::new(-MAX_K1, 0.0),
        Distortion::new(-0.01, 0.0),
        Distortion::new(0.01, 0.0),
        Distortion::new(MAX_K1, 0.0),
        Distortion::new3(-0.04, 0.005, -0.001),
        Distortion::new3(0.04, -0.005, 0.001),
    ];
    for d in cases.drain(..) {
        let k1 = d.k1;
        assert!(d.is_monotonic(), "k1 {k1} folds the radial map");

        // A horizontal ramp must stay monotone after the resampling.
        let mut img = clean_image();
        for y in 0..H {
            for x in 0..W {
                let value = (x * 20) as u16;
                for plane in img.planes.iter_mut() {
                    plane[y * W + x] = value;
                }
            }
        }
        let out = rectify(img, d);
        for y in [0, H / 2, H - 1] {
            let mut previous = 0u16;
            for x in 0..W {
                let value = out.planes[0][y * W + x];
                assert!(
                    value + 1 >= previous,
                    "k1 {k1}: row {y} folds at x {x}: {previous} -> {value}"
                );
                previous = value;
            }
        }
    }
}

#[test]
fn border_is_filled_by_clamping_not_by_holes() {
    let mut img = clean_image();
    for plane in img.planes.iter_mut() {
        plane.fill(1234);
    }
    let out = rectify(img, Distortion::new(MAX_K1, 0.0));
    assert_eq!(out.planes[0][0], 1234);
    assert_eq!(out.planes[1][W * H - 1], 1234);
    assert!(
        out.planes[0].iter().all(|value| *value == 1234),
        "constant image must stay constant"
    );
}
/// Timing of the same operations on the largest output the application
/// produces (12096 x 8064 = 97.5 MP). Ignored by default because it needs
/// ~1.2 GB; run with: cargo test --test distort --ignored --nocapture
#[test]
#[ignore = "needs ~1.2 GB and a few seconds"]
fn timings_at_97_megapixels() {
    use std::time::Instant;
    const W2: usize = 12096;
    const H2: usize = 8064;
    let mut img = MergeResult {
        width: W2,
        height: H2,
        scale: 2,
        planes: [
            vec![0u16; W2 * H2],
            vec![0u16; W2 * H2],
            vec![0u16; W2 * H2],
        ],
        white: 14303,
        shifts: Vec::new(),
        stats: MergeStats::default(),
    };
    for (c, plane) in img.planes.iter_mut().enumerate() {
        for (i, value) in plane.iter_mut().enumerate() {
            *value = ((i.wrapping_mul(2654435761) >> 13) % 12000) as u16 + c as u16;
        }
    }
    let started = Instant::now();
    let preview = pixelshift::merge::preview(&img, 2048, [1.8, 1.0, 1.4]);
    println!(
        "preview 2048 on 97 MP: {:.0} ms ({} x {})",
        started.elapsed().as_secs_f64() * 1000.0,
        preview.width,
        preview.height
    );
    let started = Instant::now();
    let crop = pixelshift::merge::preview_crop(&img, (5000, 3000, 2000, 1200), [1.8, 1.0, 1.4]);
    println!(
        "preview_crop 2000x1200 on 97 MP: {:.0} ms ({} x {})",
        started.elapsed().as_secs_f64() * 1000.0,
        crop.width,
        crop.height
    );
    let started = Instant::now();
    let rectified = rectify(img.clone(), Distortion::new(0.006, 0.0));
    let elapsed = started.elapsed().as_secs_f64() * 1000.0;
    println!("rectify 97 MP k1 +0.006: {elapsed:.0} ms");
    assert_eq!(rectified.planes[0].len(), W2 * H2);
}
