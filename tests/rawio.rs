//! WP1 acceptance: Nikon NEF decoding through LibRaw and series detection.

use pixelshift::rawio;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn workspace() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// First raw file that is actually present in the workspace. The sample
/// series are private data, so the decoder tests must also run in a checkout
/// that carries only one of them (or none at all).
fn available_nef() -> Option<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(workspace())
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

fn assert_close(what: &str, got: f32, want: f32, tol: f32) {
    assert!(
        (got - want).abs() <= tol,
        "{what}: got {got}, want {want} +/- {tol}"
    );
}

#[test]
fn shim_reports_a_libraw_version() {
    let version = rawio::shim::version().expect("LibRaw version string");
    eprintln!("LibRaw {version}");
    assert!(rawio::shim::is_available(), "shim/libraw not available");
    assert!(
        version.starts_with("0.2"),
        "unexpected LibRaw version {version}"
    );
}

#[test]
fn decodes_reference_nef() {
    let Some(path) = available_nef() else {
        eprintln!("skipping: the workspace carries no NEF sample");
        return;
    };
    let frame = rawio::open(&path).expect("decode the reference NEF");

    assert_eq!((frame.width, frame.height), (6048, 4032));
    assert_eq!(frame.data.len(), 6048 * 4032);
    assert_eq!(frame.cfa.pattern_str(), "RGGB");

    assert_eq!(frame.black, 1008, "black level");
    assert_eq!(frame.cblack, [0, 0, 0, 0]);
    assert!(
        frame.white == 16383 || frame.white == 15311,
        "saturation level {} is not what LibRaw reports for this camera",
        frame.white
    );

    // As-shot WB, green normalised (the exact ratios depend on the scene, the
    // neutral green convention does not).
    assert_close("wb.g", frame.wb[1], 1.0, 1e-6);
    assert!(
        frame.wb[0] > 0.5 && frame.wb[2] > 0.5,
        "implausible as-shot multipliers {:?}",
        frame.wb
    );

    // XYZ -> camera matrix (LibRaw cam_xyz), and camera -> sRGB.
    assert_close("cam_xyz[0][0]", frame.cam_xyz[0][0], 1.1206, 0.01);
    assert_close("cam_xyz[1][1]", frame.cam_xyz[1][1], 1.2847, 0.01);
    assert!(
        frame.cam_xyz[0..3].iter().flatten().all(|v| v.abs() > 1e-3),
        "cam_xyz first rows look empty: {:?}",
        frame.cam_xyz
    );
    assert!(
        frame.rgb_cam[0][0].abs() > 1.0 && frame.rgb_cam[1][1].abs() > 1.0,
        "rgb_cam looks empty: {:?}",
        frame.rgb_cam
    );

    assert!(frame.iso > 0.0, "no ISO recorded");
    assert!(frame.shutter > 0.0, "no shutter speed recorded");
    assert!(frame.aperture > 0.0, "no aperture recorded");
    assert!(frame.focal > 0.0, "no focal length recorded");

    // Exactly as written in the NEF EXIF (LibRaw would normalise these to
    // "Nikon"/"Z6_3", but DNG consumers match the camera on the raw strings).
    assert_eq!(frame.make, "NIKON CORPORATION");
    assert_eq!(frame.model, "NIKON Z6_3");
    assert_eq!(frame.index, 0);
    assert!(
        frame.timestamp.len() >= 10 && frame.timestamp.contains(':'),
        "no EXIF timestamp: {:?}",
        frame.timestamp
    );

    // Active area content: 14 bit data above the black level, not flat.
    let min = *frame.data.iter().min().unwrap();
    let max = *frame.data.iter().max().unwrap();
    eprintln!(
        "{} {}x{} black={} white={} wb={:?} iso={} shutter=1/{:.1} ts={:?} min={} max={}",
        path.display(),
        frame.width,
        frame.height,
        frame.black,
        frame.white,
        frame.wb,
        frame.iso,
        1.0 / frame.shutter,
        frame.timestamp,
        min,
        max
    );
    assert!(min >= 100, "unexpectedly dark minimum {min}");
    assert!(max > 4000, "unexpectedly low maximum {max}");
    assert!(
        max <= frame.white,
        "sample {max} above white {}",
        frame.white
    );
}

#[test]
fn decodes_series_in_parallel() {
    let mut series = rawio::find_series(&workspace()).expect("scan workspace");
    if series.is_empty() {
        eprintln!("skipping: the workspace carries no pixel shift series");
        return;
    }
    series.sort_by_key(|s| s.mode);
    for s in &series {
        assert!(
            matches!(s.mode, 4 | 8 | 16 | 32),
            "unexpected mode {} for {:?}",
            s.mode,
            s.dir
        );
        assert_eq!(s.files.len(), s.mode as usize);
    }
    for s in &series {
        let started = Instant::now();
        let frames = rawio::open_series(&s.files).expect("decode series");
        let elapsed = started.elapsed();
        assert_eq!(frames.len(), s.mode as usize);
        for (i, frame) in frames.iter().enumerate() {
            assert_eq!(frame.index, i, "index must follow the input order");
            assert_eq!(&frame.path, &s.files[i]);
            assert_eq!((frame.width, frame.height), (6048, 4032));
            assert_eq!(frame.cfa.pattern_str(), "RGGB");
            assert_eq!(frame.black, 1008);
            assert_eq!(frame.white, frames[0].white);
            assert_eq!(frame.cam_xyz, frames[0].cam_xyz);
            assert_eq!(frame.rgb_cam, frames[0].rgb_cam);
            assert_eq!(frame.iso, frames[0].iso);
        }
        eprintln!(
            "series mode={} frames={} decoded in {:.2}s ({:.0} ms/frame wall)",
            s.mode,
            frames.len(),
            elapsed.as_secs_f64(),
            elapsed.as_secs_f64() * 1000.0 / frames.len() as f64
        );
        assert!(
            elapsed.as_secs() < 120,
            "decoding {} frames took {:?}",
            s.mode,
            elapsed
        );
    }
}

#[test]
fn find_series_ignores_non_frames() {
    let series = rawio::find_series(&workspace()).expect("scan workspace");
    for s in &series {
        for f in &s.files {
            let name = f.file_name().unwrap().to_string_lossy().to_string();
            assert!(
                name.to_ascii_lowercase().ends_with(".nef"),
                "unexpected file {name}"
            );
            assert!(!name.contains("_merged"), "merged result leaked: {name}");
            assert!(!name.starts_with('.'), "hidden file leaked: {name}");
        }
        assert_eq!(s.files.len(), s.mode as usize);
        assert_eq!(s.scale, if s.mode >= 16 { 2 } else { 1 });
        assert_eq!(s.dir, Path::new(&s.files[0]).parent().unwrap());
    }
}
/// The zoom series records its real focal range, its serial numbers and the
/// vendor maker note: exactly the three things a raw editor needs to match a
/// lens profile, and the reason our DNG was imported without one.
#[test]
fn zoom_series_reports_its_focal_range_serials_and_maker_note() {
    let path = workspace().join("108NZ6_3/_DSC0582.NEF");
    if !path.exists() {
        eprintln!("skipping: the zoom series is not in the workspace");
        return;
    }
    let frame = rawio::open(&path).expect("decode the zoom series");
    assert_close("lens min focal", frame.lens_min_focal, 24.0, 0.01);
    assert_close("lens max focal", frame.lens_max_focal, 120.0, 0.01);
    assert_close("f number at min focal", frame.lens_min_fnum, 4.0, 0.01);
    assert_close("f number at max focal", frame.lens_max_fnum, 4.0, 0.01);
    assert_eq!(frame.lens_model, "NIKKOR Z 24-120mm f/4 S");
    assert_eq!(frame.body_serial, "7016932");
    assert_eq!(frame.lens_serial, "20312694");

    let note = rawio::makernote(&path).expect("the NEF carries a maker note");
    assert!(
        note.starts_with(b"Nikon\0"),
        "not a Nikon maker note: {:?}",
        &note[..8.min(note.len())]
    );
    assert_eq!(note.len(), 231314, "maker note length");
}

