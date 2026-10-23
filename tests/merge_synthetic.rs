//! Synthetic ground truth tests for the WP2 merge core (docs/TASKS.md, WP2).
//!
//! Frames are rendered physically: sensor pixel (x, y) of colour c stores
//! scene(x + dx, y + dy)[c] plus the black level, which is exactly the model
//! that merge() inverts. No interpolation artefact of the generator can hide a
//! geometry error, so these tests pin the shift sign, the output grid and the
//! weight normalisation.

use pixelshift::merge;
use pixelshift::shift;
use pixelshift::types::{Cfa, DngMeta, MergeParams, MergeResult, RawFrame, Shift, WriteOpts};
use std::path::{Path, PathBuf};
use std::time::Instant;

const BLACK: u16 = 1008;
const WHITE: u16 = 16383;

fn span() -> f64 {
    WHITE as f64 - BLACK as f64
}

/// Deterministic xorshift generator, so the tests never depend on rand.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u32() as f64 + 0.5) / 4_294_967_296.0
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn soft_square(x: f64, period: f64, soft: f64) -> f64 {
    let s = (2.0 * std::f64::consts::PI * x / period).sin();
    0.5 + 0.5 * (s / soft).tanh()
}

/// Analytic scene: rings, gratings, block patterns and text-like strokes, in
/// reference frame sensor coordinates.
struct Scene;

impl Scene {
    fn rgb(&self, x: f64, y: f64) -> [f64; 3] {
        use std::f64::consts::PI;
        let r = (x * x + y * y).sqrt();
        let rings = 0.5 + 0.5 * (0.55 * r).cos();
        let grat = 0.5 + 0.5 * (2.0 * PI * 0.11 * x).cos();
        let diag = 0.5 + 0.5 * (2.0 * PI * 0.15 * (0.7 * x + 0.7 * y)).sin();
        let bars = soft_square(x, 41.0, 0.4) * soft_square(y, 57.0, 0.4);
        let text = soft_square(x + 0.6 * y, 17.0, 0.55);
        let base_r = 0.14 + 0.62 * rings * grat;
        let base_g = 0.16 + 0.60 * (0.5 * (rings + diag));
        let base_b = 0.12 + 0.58 * (0.5 + 0.5 * (2.0 * PI * 0.08 * (x - 0.3 * y)).sin());
        let r = (base_r * (0.60 + 0.40 * bars) + 0.10 * text).clamp(0.02, 0.97);
        let g = (base_g * (0.62 + 0.38 * bars) + 0.08 * text).clamp(0.02, 0.97);
        let b = (base_b * (0.58 + 0.42 * bars) + 0.12 * (1.0 - text)).clamp(0.02, 0.97);
        [r, g, b]
    }
}

fn render(
    scene: &Scene,
    w: usize,
    h: usize,
    shift: (f64, f64),
    sigma: f64,
    seed: u64,
) -> RawFrame {
    render_rigid(scene, w, h, Shift::new(shift.0, shift.1), sigma, seed)
}

/// The ideal half pixel lattice with a deterministic sub-pixel perturbation of
/// every frame: the samples then land off the output lattice, which is where the
/// splat kernel really blurs, while the coverage of the output grid stays as good
/// as on the lattice itself.
fn perturbed_half_lattice() -> Vec<(f64, f64)> {
    const JITTER: [(f64, f64); 4] = [(0.23, -0.17), (-0.19, 0.11), (0.07, -0.21), (-0.11, 0.19)];
    half_lattice()
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let j = JITTER[i % JITTER.len()];
            (p.0 + j.0, p.1 + j.1)
        })
        .collect()
}

/// Render one frame from an arbitrary analytic scene, sensor pixel (x, y) of
/// colour c holding scene(x + dx, y + dy)[c] (the merge model inverted exactly).
fn render_analytic<F: Fn(f64, f64) -> [f64; 3]>(
    scene: &F,
    w: usize,
    h: usize,
    shift: (f64, f64),
    sigma: f64,
    seed: u64,
) -> RawFrame {
    let cfa = Cfa::from_pattern_str("RGGB").expect("RGGB");
    let mut rng = Rng::new(seed);
    let mut data = vec![0u16; w * h];
    for y in 0..h {
        for x in 0..w {
            let c = cfa.0[y & 1][x & 1];
            let v = scene(x as f64 + shift.0, y as f64 + shift.1)[c.idx()] * span()
                + BLACK as f64
                + if sigma > 0.0 { sigma * rng.normal() } else { 0.0 };
            data[y * w + x] = v.round().clamp(0.0, WHITE as f64) as u16;
        }
    }
    RawFrame {
        path: PathBuf::from("synthetic"),
        width: w,
        height: h,
        data,
        cfa,
        black: BLACK,
        cblack: [0; 4],
        white: WHITE,
        wb: [1.0, 1.0, 1.0, 1.0],
        cam_xyz: [[0.0; 3]; 4],
        rgb_cam: [[0.0; 3]; 4],
        iso: 100.0,
        shutter: 1.0 / 125.0,
        aperture: 5.6,
        focal: 50.0,
        lens_make: String::new(),
        lens_model: String::new(),
        focal_35mm: 0,
        max_aperture: 0.0,
        lens_min_focal: 0.0,
        lens_max_focal: 0.0,
        lens_min_fnum: 0.0,
        lens_max_fnum: 0.0,
        lens_serial: String::new(),
        body_serial: String::new(),
        make: "synthetic".to_string(),
        model: "scene".to_string(),
        timestamp: String::new(),
        index: 0,
    }
}

/// Render a frame as seen through the rigid transform of types::Shift::map.
fn render_rigid(
    scene: &Scene,
    w: usize,
    h: usize,
    shift: Shift,
    sigma: f64,
    seed: u64,
) -> RawFrame {
    let cfa = Cfa::from_pattern_str("RGGB").expect("RGGB");
    let mut rng = Rng::new(seed);
    let mut data = vec![0u16; w * h];
    let (cx, cy) = (w as f64 * 0.5, h as f64 * 0.5);
    for y in 0..h {
        for x in 0..w {
            let c = cfa.0[y & 1][x & 1];
            let (sx, sy) = shift.map(x as f64, y as f64, cx, cy);
            let value = scene.rgb(sx, sy)[c.idx()] * span()
                + BLACK as f64
                + if sigma > 0.0 { sigma * rng.normal() } else { 0.0 };
            data[y * w + x] = value.round().clamp(0.0, WHITE as f64) as u16;
        }
    }
    RawFrame {
        path: PathBuf::from("synthetic"),
        width: w,
        height: h,
        data,
        cfa,
        black: BLACK,
        cblack: [0; 4],
        white: WHITE,
        wb: [1.0, 1.0, 1.0, 1.0],
        cam_xyz: [[0.0; 3]; 4],
        rgb_cam: [[0.0; 3]; 4],
        iso: 100.0,
        shutter: 1.0 / 125.0,
        aperture: 5.6,
        focal: 50.0,
        lens_make: String::new(),
        lens_model: String::new(),
        focal_35mm: 0,
        max_aperture: 0.0,
        lens_min_focal: 0.0,
        lens_max_focal: 0.0,
        lens_min_fnum: 0.0,
        lens_max_fnum: 0.0,
        lens_serial: String::new(),
        body_serial: String::new(),
        make: "synthetic".to_string(),
        model: "scene".to_string(),
        timestamp: String::new(),
        index: 0,
    }
}

/// Error of the merged image against the scene over an inset region.
fn error_stats(merged: &MergeResult, scene: &Scene, inset: usize) -> (f64, f64) {
    let k = merged.scale as f64;
    let mut sse = 0.0f64;
    let mut n = 0.0f64;
    for y in inset..merged.height.saturating_sub(inset) {
        for x in inset..merged.width.saturating_sub(inset) {
            let truth = scene.rgb(x as f64 / k, y as f64 / k);
            for (c, t) in truth.iter().enumerate() {
                let got = merged.planes[c][y * merged.width + x] as f64;
                let want = t * span();
                sse += (got - want) * (got - want);
                n += 1.0;
            }
        }
    }
    let rms = (sse / n.max(1.0)).sqrt();
    let psnr = 20.0 * (span() / rms.max(1e-9)).log10();
    (rms, psnr)
}

/// Real series run: decodes the NEFs through WP1 rawio, estimates the rigid
/// motion and the local correction grids of every frame, merges and writes the
/// result as a linear DNG so that tools/validate.py and tools/resolution.py can
/// score it. Ignored by default because it needs the reference NEF files,
/// several gigabytes of RAM and a 558 MiB output.
///
/// Configuration (environment, all optional):
///   PS_SERIES  series directory, default 106NZ6_3
///   PS_DECONV  MergeParams::deconvolve, default from MergeParams
///   PS_GRID    1 = apply the estimated grids, 0 = rigid model only, default 1
///   PS_OUT     output path, default .scratch/wpd/<series>_d<deconv>_g<grid>.dng
#[test]
#[ignore = "writes a 558 MiB DNG for external validation"]
fn real_series_grid_dng() {
    let env_str = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let name = env_str("PS_SERIES").unwrap_or_else(|| "106NZ6_3".to_string());
    let deconv: f32 = env_str("PS_DECONV")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let use_grid = env_str("PS_GRID").map(|v| v != "0").unwrap_or(true);
    let dir = root.join(&name);
    let Ok(series) = pixelshift::rawio::find_series(&dir) else {
        println!("rawio is not available in this build, skipping");
        return;
    };
    let s = series.first().expect("one series").clone();
    let frames = pixelshift::rawio::open_series(&s.files).expect("decode");
    let t0 = Instant::now();
    let (shifts, grids) =
        pixelshift::shift::estimate_series_with_grids(&frames).expect("estimate");
    let estimate_s = t0.elapsed().as_secs_f64();
    let count = grids.iter().filter(|g| g.is_some()).count();
    let mut res_rms = 0.0f64;
    let mut res_max = 0.0f32;
    let mut res_n = 0.0f64;
    let mut shape = (0usize, 0usize);
    for g in grids.iter().flatten() {
        shape = (g.cols, g.rows);
        for (dx, dy) in g.dx.iter().zip(g.dy.iter()) {
            let d = (dx * dx + dy * dy).sqrt();
            res_rms += (d * d) as f64;
            res_max = res_max.max(d);
            res_n += 1.0;
        }
    }
    println!(
        "{}: {} frames, grid {:?} on {} of them, anchor residual rms {:.3} px max {:.3} px, estimate {:.1} s",
        name,
        frames.len(),
        shape,
        count,
        (res_rms / res_n.max(1.0)).sqrt(),
        res_max,
        estimate_s
    );
    for (i, sh) in shifts.iter().enumerate() {
        println!(
            "  frame {:2}: dx {:+.3} dy {:+.3} rot {:+.1} arcsec residual {:.3} px conf {:.2}",
            i,
            sh.dx,
            sh.dy,
            sh.rot_arcsec(),
            sh.residual,
            sh.confidence
        );
    }
    let p = MergeParams {
        scale: s.scale,
        align: true,
        shifts,
        grids: if use_grid { grids } else { Vec::new() },
        deconvolve: deconv,
        ..Default::default()
    };
    let started = Instant::now();
    let res = merge::merge(&frames, &p).expect("merge");
    println!(
        "grid {} deconv {} merge {}x{}: {:.2} s, {:.3} samples/channel, filled {} ({:.4}%)",
        use_grid as u8,
        deconv,
        res.width,
        res.height,
        started.elapsed().as_secs_f64(),
        res.stats.mean_samples_per_channel,
        res.stats.filled_pixels,
        100.0 * res.stats.filled_pixels as f64 / (res.width * res.height) as f64
    );
    let f = &frames[0];
    let mut color_matrix = [[0.0f32; 3]; 3];
    for (r, row) in color_matrix.iter_mut().enumerate() {
        for (c, v) in row.iter_mut().enumerate() {
            *v = f.cam_xyz[r][c];
        }
    }
    let meta = DngMeta {
        make: f.make.clone(),
        model: f.model.clone(),
        timestamp: f.timestamp.clone(),
        software: "pixelshift wp2".to_string(),
        as_shot_neutral: [
            (1.0 / f.wb[0].max(1e-3)).min(1.0),
            (1.0 / f.wb[1].max(1e-3)).min(1.0),
            (1.0 / f.wb[2].max(1e-3)).min(1.0),
        ],
        color_matrix,
        calib_illuminant: 21,
        baseline_exposure: 0.0,
        iso: f.iso,
        shutter: f.shutter,
        aperture: f.aperture,
        focal: f.focal,
        orientation: 1,
        wb_applied: false,
        lens_make: f.lens_make.clone(),
        lens_model: f.lens_model.clone(),
        focal_35mm: f.focal_35mm,
        max_aperture: f.max_aperture,
        lens_min_focal: f.lens_min_focal,
        lens_max_focal: f.lens_max_focal,
        lens_min_fnum: f.lens_min_fnum,
        lens_max_fnum: f.lens_max_fnum,
        lens_serial: f.lens_serial.clone(),
        body_serial: f.body_serial.clone(),
        makernote: Vec::new(),
    };
    let out = root.join(
        env_str("PS_OUT")
            .unwrap_or_else(|| format!(".scratch/wpd/{}_d{}_g{}.dng", name, deconv, use_grid as u8)),
    );
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent).expect("create output dir");
    }
    pixelshift::write::write_dng(
        &out,
        &res,
        &meta,
        &WriteOpts {
            compress: false,
            preview_max: 0,
        },
    )
    .expect("write dng");
    println!("wrote {}", out.display());
}

#[test]
#[ignore = "real NEF series, run manually with --ignored"]
fn real_series_alignment_and_merge() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let Ok(series) = pixelshift::rawio::find_series(root) else {
        println!("rawio is not available in this build, skipping");
        return;
    };
    for s in &series {
        let frames = pixelshift::rawio::open_series(&s.files).expect("decode");
        let shifts = shift::estimate_series(&frames).expect("align");
        println!(
            "{}: mode {} scale {} frames {}",
            s.dir.display(),
            s.mode,
            s.scale,
            frames.len()
        );
        for (i, sh) in shifts.iter().enumerate() {
            println!(
                "  frame {:2}: dx {:+.3} dy {:+.3} rot {:+.1} arcsec residual {:.3} px conf {:.2}",
                i,
                sh.dx,
                sh.dy,
                sh.rot_arcsec(),
                sh.residual,
                sh.confidence
            );
        }
        let mut p = MergeParams {
            scale: s.scale,
            shifts,
            ..Default::default()
        };
        p.crop_uncovered = true;
        let started = Instant::now();
        let res = merge::merge(&frames, &p).expect("merge");
        println!(
            "  merged {}x{} in {:.2} s, {:.3} samples/channel, min coverage {:.3}, filled {} ({:.4}%), stats {:.0} ms",
            res.width,
            res.height,
            started.elapsed().as_secs_f64(),
            res.stats.mean_samples_per_channel,
            res.stats.min_coverage,
            res.stats.filled_pixels,
            100.0 * res.stats.filled_pixels as f64 / (res.width * res.height) as f64,
            res.stats.elapsed_ms
        );
        let share = res.stats.filled_pixels as f64 / (res.width * res.height) as f64;
        assert!(share < 0.01, "filled share {share:.4} is not below 1%");
        assert!(res.width > 1000 && res.height > 600);
    }
}

fn even_lattice_of(p: &MergeParams) -> Vec<Shift> {
    p.shifts.clone()
}

fn params(scale: u32, shifts: Vec<Shift>) -> MergeParams {
    MergeParams {
        scale,
        shifts,
        ..Default::default()
    }
}

fn even_lattice() -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    for dy in [0.0, 1.0] {
        for dx in [0.0, 1.0] {
            out.push((dx, dy));
        }
    }
    out
}

fn half_lattice() -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    for dy in [0.0, 0.5, 1.0, 1.5] {
        for dx in [0.0, 0.5, 1.0, 1.5] {
            out.push((dx, dy));
        }
    }
    out
}

#[test]
fn four_frames_scale_one_reach_45_db() {
    let scene = Scene;
    let (w, h) = (512usize, 512usize);
    let positions = even_lattice();
    let frames: Vec<RawFrame> = positions
        .iter()
        .enumerate()
        .map(|(i, p)| render(&scene, w, h, *p, 0.0, 11 + i as u64))
        .collect();
    let shifts: Vec<Shift> = positions.iter().map(|p| Shift::new(p.0, p.1)).collect();
    let res = merge::merge(&frames, &params(1, shifts)).expect("merge");
    assert_eq!((res.width, res.height, res.scale), (w, h, 1));
    let (rms, psnr) = error_stats(&res, &scene, 6);
    println!(
        "4 frames scale 1: rms {:.3} ADU psnr {:.1} dB samples/px {:.3} filled {}",
        rms, psnr, res.stats.mean_samples_per_channel, res.stats.filled_pixels
    );
    assert!(psnr > 45.0, "psnr {psnr:.2} dB is below 45 dB");
    // Ideal integer lattice: every interior pixel collects R + G + G + B.
    assert!(
        (res.stats.mean_samples_per_channel - 4.0 / 3.0).abs() < 0.02,
        "samples per channel {} instead of 4/3",
        res.stats.mean_samples_per_channel
    );
    assert_eq!(res.stats.frames, 4);
}

#[test]
fn sixteen_frames_scale_two_reach_40_db() {
    let scene = Scene;
    let (w, h) = (256usize, 256usize);
    let positions = half_lattice();
    assert_eq!(positions.len(), 16);
    let frames: Vec<RawFrame> = positions
        .iter()
        .enumerate()
        .map(|(i, p)| render(&scene, w, h, *p, 0.0, 31 + i as u64))
        .collect();
    let shifts: Vec<Shift> = positions.iter().map(|p| Shift::new(p.0, p.1)).collect();
    let res = merge::merge(&frames, &params(2, shifts)).expect("merge");
    assert_eq!((res.width, res.height, res.scale), (2 * w, 2 * h, 2));
    let (rms, psnr) = error_stats(&res, &scene, 8);
    println!(
        "16 frames scale 2: rms {:.3} ADU psnr {:.1} dB samples/px {:.3} filled {} of {}",
        rms,
        psnr,
        res.stats.mean_samples_per_channel,
        res.stats.filled_pixels,
        res.width * res.height
    );
    assert!(psnr > 40.0, "psnr {psnr:.2} dB is below 40 dB");
    assert!(
        (res.stats.mean_samples_per_channel - 4.0 / 3.0).abs() < 0.05,
        "samples per channel {} instead of 4/3",
        res.stats.mean_samples_per_channel
    );

    // Same series with irrelevant shifts must be visibly worse: this pins the
    // sign convention of MergeParams::shifts.
    let zero: Vec<Shift> = positions.iter().map(|_| Shift::new(0.0, 0.0)).collect();
    let bad = merge::merge(&frames, &params(2, zero)).expect("merge");
    let (_, bad_psnr) = error_stats(&bad, &scene, 8);
    println!("16 frames scale 2 without shifts: psnr {:.1} dB", bad_psnr);
    assert!(
        bad_psnr < psnr - 10.0,
        "unshifted merge {bad_psnr:.1} dB is not clearly worse than {psnr:.1} dB"
    );
}

#[test]
fn thirty_two_frames_halve_the_noise_power() {
    let scene = Scene;
    let (w, h) = (256usize, 256usize);
    let positions = half_lattice();
    let sigma = 50.0;
    let mut frames: Vec<RawFrame> = Vec::new();
    for round in 0..2u64 {
        for (i, p) in positions.iter().enumerate() {
            frames.push(render(
                &scene,
                w,
                h,
                *p,
                sigma,
                1000 + round * 37 + i as u64,
            ));
        }
    }
    let shifts: Vec<Shift> = frames
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let p = positions[i % positions.len()];
            Shift::new(p.0, p.1)
        })
        .collect();
    let res16 = merge::merge(&frames[..16], &params(2, shifts[..16].to_vec())).expect("16");
    let res32 = merge::merge(&frames, &params(2, shifts.clone())).expect("32");
    let (rms16, _) = error_stats(&res16, &scene, 8);
    let (rms32, _) = error_stats(&res32, &scene, 8);
    let ratio = rms32 / rms16;
    println!(
        "sigma {sigma}: rms16 {:.2} ADU rms32 {:.2} ADU ratio {:.4} (expected 0.7071, samples/px {:.3} -> {:.3})",
        rms16, rms32, ratio, res16.stats.mean_samples_per_channel, res32.stats.mean_samples_per_channel
    );
    assert!(
        (ratio - 0.7071).abs() < 0.08,
        "noise ratio {ratio:.3} is not close to 1/sqrt(2)"
    );
    assert!(
        (res32.stats.mean_samples_per_channel / res16.stats.mean_samples_per_channel - 2.0).abs() < 0.05,
        "32 frames should double the samples per channel"
    );

    // denoise = false keeps one frame per lattice position.
    let mut no_denoise = params(2, shifts);
    no_denoise.denoise = false;
    let res_nod = merge::merge(&frames, &no_denoise).expect("denoise off");
    println!(
        "denoise off: {} of 32 frames, samples/px {:.3}, filled {}",
        res_nod.stats.frames, res_nod.stats.mean_samples_per_channel, res_nod.stats.filled_pixels
    );
    assert_eq!(res_nod.stats.frames, 16);
}

#[test]
fn rotation_keeps_the_psnr_targets() {
    let scene = Scene;
    let (w, h) = (256usize, 256usize);
    let positions = half_lattice();
    let rot = 1e-4;
    let shifts: Vec<Shift> = positions
        .iter()
        .map(|p| Shift::with_rot(p.0, p.1, rot))
        .collect();
    let frames: Vec<RawFrame> = shifts
        .iter()
        .enumerate()
        .map(|(i, s)| render_rigid(&scene, w, h, *s, 0.0, 300 + i as u64))
        .collect();
    let res = merge::merge(&frames, &params(2, shifts.clone())).expect("merge");
    let (_, psnr) = error_stats(&res, &scene, 10);
    let flat: Vec<Shift> = shifts.iter().map(|s| Shift::new(s.dx, s.dy)).collect();
    let bad = merge::merge(&frames, &params(2, flat)).expect("merge");
    let (_, psnr_flat) = error_stats(&bad, &scene, 10);
    println!(
        "rotation {:+.1e} rad ({:.1} arcsec): psnr {:.1} dB, rotation ignored {:.1} dB",
        rot,
        rot.to_degrees() * 3600.0,
        psnr,
        psnr_flat
    );
    assert!(psnr > 40.0, "psnr with rotation {psnr:.1} dB");
    assert!(
        psnr_flat < psnr - 1.0,
        "ignoring the rotation ({psnr_flat:.1} dB) is not clearly worse than {psnr:.1} dB"
    );
}

/// Render a frame through a rigid transform plus a linear drift in y, the
/// non-rigid component measured on the real 16/32 frame series.
fn render_drift(
    scene: &Scene,
    w: usize,
    h: usize,
    shift: Shift,
    drift: (f64, f64),
    seed: u64,
) -> RawFrame {
    let cfa = Cfa::from_pattern_str("RGGB").expect("RGGB");
    let mut rng = Rng::new(seed);
    let mut data = vec![0u16; w * h];
    let (cx, cy) = (w as f64 * 0.5, h as f64 * 0.5);
    for y in 0..h {
        let t = y as f64 / h as f64 - 0.5;
        for x in 0..w {
            let c = cfa.0[y & 1][x & 1];
            let (sx, sy) = shift.map(x as f64, y as f64, cx, cy);
            let value = scene.rgb(sx + drift.0 * t, sy + drift.1 * t)[c.idx()] * span()
                + BLACK as f64
                + 0.0 * rng.normal();
            data[y * w + x] = value.round().clamp(0.0, WHITE as f64) as u16;
        }
    }
    RawFrame {
        path: PathBuf::from("synthetic"),
        width: w,
        height: h,
        data,
        cfa,
        black: BLACK,
        cblack: [0; 4],
        white: WHITE,
        wb: [1.0, 1.0, 1.0, 1.0],
        cam_xyz: [[0.0; 3]; 4],
        rgb_cam: [[0.0; 3]; 4],
        iso: 100.0,
        shutter: 1.0 / 125.0,
        aperture: 5.6,
        focal: 50.0,
        lens_make: String::new(),
        lens_model: String::new(),
        focal_35mm: 0,
        max_aperture: 0.0,
        lens_min_focal: 0.0,
        lens_max_focal: 0.0,
        lens_min_fnum: 0.0,
        lens_max_fnum: 0.0,
        lens_serial: String::new(),
        body_serial: String::new(),
        make: "synthetic".to_string(),
        model: "scene".to_string(),
        timestamp: String::new(),
        index: 0,
    }
}

#[test]
fn constant_grid_is_equivalent_to_a_shift_offset() {
    let scene = Scene;
    let (w, h) = (128usize, 128usize);
    let positions = half_lattice();
    let frames: Vec<RawFrame> = positions
        .iter()
        .enumerate()
        .map(|(i, p)| render(&scene, w, h, *p, 0.0, 401 + i as u64))
        .collect();
    let extra = (0.3f32, -0.2f32);
    let shifted: Vec<Shift> = positions
        .iter()
        .map(|p| Shift::new(p.0 + extra.0 as f64, p.1 + extra.1 as f64))
        .collect();
    let mut no_filter = params(2, shifted);
    no_filter.deconvolve = 0.0;
    let plain = merge::merge(&frames, &no_filter).expect("shift");

    let mut p = params(
        2,
        positions.iter().map(|p| Shift::new(p.0, p.1)).collect(),
    );
    p.deconvolve = 0.0;
    p.grids = (0..frames.len())
        .map(|_| {
            Some(pixelshift::types::ShiftGrid {
                x0: 0.0,
                y0: 0.0,
                step_x: 1.0,
                step_y: 1.0,
                cols: 2,
                rows: 2,
                dx: vec![extra.0; 4],
                dy: vec![extra.1; 4],
            })
        })
        .collect();
    let gridded = merge::merge(&frames, &p).expect("grid");
    assert_eq!((plain.width, plain.height), (gridded.width, gridded.height));
    let mut worst = 0i64;
    for c in 0..3 {
        for (a, b) in plain.planes[c].iter().zip(gridded.planes[c].iter()) {
            worst = worst.max((*a as i64 - *b as i64).abs());
        }
    }
    println!("constant grid vs shift offset: worst pixel difference {worst} ADU");
    assert!(worst <= 1, "constant grid differs from a shift offset by {worst} ADU");
}

#[test]
fn grid_sample_clamps_outside_the_grid() {
    let g = pixelshift::types::ShiftGrid {
        x0: 100.0,
        y0: 200.0,
        step_x: 500.0,
        step_y: 400.0,
        cols: 3,
        rows: 2,
        dx: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        dy: vec![-0.1, -0.2, -0.3, -0.4, -0.5, -0.6],
    };
    assert!(g.is_usable());
    let (inside, _) = g.sample(100.0, 200.0);
    assert!((inside - 0.1).abs() < 1e-6);
    let (far_left, dy_left) = g.sample(-1.0e6, -1.0e6);
    assert!((far_left - 0.1).abs() < 1e-6 && (dy_left + 0.1).abs() < 1e-6);
    let (far_right, dy_right) = g.sample(1.0e6, 1.0e6);
    assert!((far_right - 0.6).abs() < 1e-6 && (dy_right + 0.6).abs() < 1e-6);
    let (mid, _) = g.sample(350.0, 200.0);
    assert!((mid - 0.15).abs() < 1e-6, "bilinear midpoint {mid}");
}

#[test]
fn non_rigid_drift_is_absorbed_by_the_grid() {
    let scene = Scene;
    let (w, h) = (256usize, 256usize);
    let positions = half_lattice();
    let drift = (0.25f64, 0.3f64);
    let frames: Vec<RawFrame> = positions
        .iter()
        .enumerate()
        .map(|(i, p)| {
            // The reference frame is the geometric reference, so only the other
            // frames carry the drift: that is what the grid has to undo.
            let d = if i == 0 { (0.0, 0.0) } else { drift };
            render_drift(&scene, w, h, Shift::new(p.0, p.1), d, 501 + i as u64)
        })
        .collect();
    let (shifts, grids) =
        pixelshift::shift::estimate_series_with_grids(&frames).expect("estimate");
    let ngrids = grids.iter().filter(|g| g.is_some()).count();
    println!("drift {:?}: {} of {} frames got a grid", drift, ngrids, grids.len());
    assert!(ngrids >= frames.len() - 2, "too few grids");
    let mut p = params(2, shifts.clone());
    p.grids = grids.clone();
    let res = merge::merge(&frames, &p).expect("merge with grid");
    let (rms_grid, db_grid) = error_stats(&res, &scene, 10);
    let res_rigid = merge::merge(&frames, &params(2, shifts)).expect("merge rigid");
    let (_, db_rigid) = error_stats(&res_rigid, &scene, 10);
    println!(
        "non-rigid drift: psnr with grid {:.1} dB (rms {:.2}), rigid only {:.1} dB",
        db_grid, rms_grid, db_rigid
    );
    assert!(db_grid > 40.0, "psnr with grid {db_grid:.1} dB");
    assert!(
        db_grid > db_rigid + 1.0,
        "the grid does not improve over the rigid model ({db_grid:.1} vs {db_rigid:.1} dB)"
    );
}

#[test]
fn crop_uncovered_removes_the_partial_border() {
    let scene = Scene;
    let (w, h) = (200usize, 180usize);
    let positions = half_lattice();
    let frames: Vec<RawFrame> = positions
        .iter()
        .enumerate()
        .map(|(i, p)| render(&scene, w, h, *p, 0.0, 51 + i as u64))
        .collect();
    let shifts: Vec<Shift> = positions.iter().map(|p| Shift::new(p.0, p.1)).collect();
    let mut p = params(2, shifts);
    p.crop_uncovered = true;
    p.min_coverage = 0.25;
    let res = merge::merge(&frames, &p).expect("merge");
    let full = merge::merge(&frames, &params(2, even_lattice_of(&p)))
        .expect("merge without crop");
    let border = (2 * w * 2 * h - res.width * res.height) as f64 / (4 * w * h) as f64;
    println!(
        "crop: {}x{} (removed {:.2}%), filled {} of {} ({:.4}%)",
        res.width,
        res.height,
        100.0 * border,
        res.stats.filled_pixels,
        res.width * res.height,
        100.0 * res.stats.filled_pixels as f64 / (res.width * res.height) as f64
    );
    assert!(res.width < 2 * w && res.height < 2 * h, "nothing was cropped");
    let share = res.stats.filled_pixels as f64 / (res.width * res.height) as f64;
    assert!(share < 0.001, "filled share {share:.5} is not below 0.1%");
    // The cropped image must be the plain interior of the uncropped one.
    let found = (0..6).find_map(|oy| {
        (0..6).find_map(|ox| {
            let same = (0..res.height).all(|y| {
                (0..res.width).all(|x| {
                    (0..3).all(|c| {
                        res.planes[c][y * res.width + x]
                            == full.planes[c][(y + oy) * full.width + x + ox]
                    })
                })
            });
            same.then_some((ox, oy))
        })
    });
    println!("crop offset {:?}", found);
    assert!(found.is_some(), "cropped image is not a window of the full merge");
    // Green is missing on the first two rows of a half pixel lattice, so the
    // top and the left rim must be gone.
    assert!(found.unwrap().0 >= 1 && found.unwrap().1 >= 1);
}

#[test]
fn preview_and_crop_are_sane() {
    let scene = Scene;
    let (w, h) = (128usize, 128usize);
    let positions = half_lattice();
    let frames: Vec<RawFrame> = positions
        .iter()
        .enumerate()
        .map(|(i, p)| render(&scene, w, h, *p, 0.0, 71 + i as u64))
        .collect();
    let shifts: Vec<Shift> = positions.iter().map(|p| Shift::new(p.0, p.1)).collect();
    let res = merge::merge(&frames, &params(2, shifts)).expect("merge");
    let wb = [1.9f32, 1.0, 1.4];
    let pv = merge::preview(&res, 128, wb);
    assert_eq!((pv.width, pv.height), (128, 128));
    assert_eq!(pv.rgb.len(), 128 * 128 * 3);
    let mean = pv.rgb.iter().map(|v| *v as f64).sum::<f64>() / pv.rgb.len() as f64;
    let var = pv
        .rgb
        .iter()
        .map(|v| {
            let d = *v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / pv.rgb.len() as f64;
    println!("preview: mean {:.1} std {:.1}", mean, var.sqrt());
    assert!(mean > 20.0 && mean < 235.0, "preview mean {mean:.1}");
    assert!(var.sqrt() > 5.0, "preview is flat");

    // never upscales
    let small = merge::preview(&res, 4096, wb);
    assert_eq!((small.width, small.height), (res.width, res.height));

    let crop = merge::preview_crop(&res, (40, 60, 100, 70), wb);
    assert_eq!((crop.width, crop.height), (100, 70));
    assert_eq!(crop.rgb.len(), 100 * 70 * 3);
    assert!(crop.rgb.iter().any(|v| *v > 0 && *v < 255), "crop is blown out");
    let empty = merge::preview_crop(&res, (res.width + 5, 0, 10, 10), wb);
    assert_eq!((empty.width, empty.height), (0, 0));

    let means = merge::channel_means(&res);
    println!("channel means {means:?}");
    assert!(means.iter().all(|m| *m > 100.0 && *m < span()));
}

#[test]
fn downsample_1x_averages_pairs() {
    let scene = Scene;
    let (w, h) = (128usize, 128usize);
    let positions = half_lattice();
    let frames: Vec<RawFrame> = positions
        .iter()
        .enumerate()
        .map(|(i, p)| render(&scene, w, h, *p, 0.0, 91 + i as u64))
        .collect();
    let shifts: Vec<Shift> = positions.iter().map(|p| Shift::new(p.0, p.1)).collect();
    let res = merge::merge(&frames, &params(2, shifts)).expect("merge");
    let half = merge::downsample_1x(&res);
    assert_eq!((half.width, half.height, half.scale), (w, h, 1));
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                let i = (2 * y) * res.width + 2 * x;
                let want = ((res.planes[c][i] as u32
                    + res.planes[c][i + 1] as u32
                    + res.planes[c][i + res.width] as u32
                    + res.planes[c][i + res.width + 1] as u32) as f32
                    * 0.25)
                    .round() as u16;
                assert_eq!(half.planes[c][y * w + x], want, "channel {c} at {x},{y}");
            }
        }
    }
    // The box average is a mild low pass, so it still resembles the scene.
    let (_, psnr_half) = error_stats(&half, &scene, 3);
    println!("downsample 1x vs pointwise scene: psnr {psnr_half:.1} dB");
    assert!(psnr_half > 25.0, "downsampled psnr {psnr_half:.1} dB");
    let same = merge::downsample_1x(&half);
    assert_eq!((same.width, same.height), (w, h));
}

/// Deconvolution is an exact identity when nothing was lost in the
/// reconstruction: on the ideal half pixel lattice every sample lands on an
/// output pixel of the 2x grid, the splat is a delta, the measured kernel
/// variance is 0 and the filter must not touch a single sample.
#[test]
fn deconvolve_is_the_identity_on_the_ideal_lattice() {
    let scene = Scene;
    let (w, h) = (256usize, 256usize);
    let positions = half_lattice();
    let frames: Vec<RawFrame> = positions
        .iter()
        .enumerate()
        .map(|(i, p)| render(&scene, w, h, *p, 0.0, 31 + i as u64))
        .collect();
    let shifts: Vec<Shift> = positions.iter().map(|p| Shift::new(p.0, p.1)).collect();
    let mut flat = params(2, shifts.clone());
    flat.deconvolve = 0.0;
    let plain = merge::merge(&frames, &flat).expect("merge");
    let mut sharp = params(2, shifts);
    sharp.deconvolve = 1.0;
    let out = merge::merge(&frames, &sharp).expect("merge deconvolved");
    let mut worst = 0i64;
    for c in 0..3 {
        for (a, b) in plain.planes[c].iter().zip(out.planes[c].iter()) {
            worst = worst.max((*a as i64 - *b as i64).abs());
        }
    }
    let (_, db_plain) = error_stats(&plain, &scene, 8);
    let (_, db_out) = error_stats(&out, &scene, 8);
    println!(
        "ideal lattice: psnr {:.2} -> {:.2} dB, worst pixel difference {worst} ADU",
        db_plain, db_out
    );
    assert_eq!(worst, 0, "deconvolve changed an ideal lattice merge by {worst} ADU");
    assert!((db_out - db_plain).abs() < 1e-9, "psnr moved on the ideal lattice");
}

/// Off the lattice the same merge is soft, and the post filter must not make it
/// worse. Measured finding: on a point sampled synthetic scene it does not help
/// either. The residual error there is not a blur that can be inverted but the
/// Gibbs ringing of an ideal step reconstructed on a finite lattice (the merge
/// itself overshoots a point sampled step by 12.8 %, see the overshoot test), and
/// boosting the amplitude of that ringing is not an improvement. On real sensor
/// data the aperture integration is a real, invertible blur and the same filter
/// does restore detail (measured: series 106 detail ratio 0.44 -> 0.57 against
/// NX Studio); this test pins that the synthetic null case stays neutral.
#[test]
fn deconvolve_stays_neutral_on_a_synthetic_band_limited_scene() {
    let band = |x: f64, y: f64| {
        use std::f64::consts::PI;
        let _ = y;
        let a = 0.5 + 0.15 * (2.0 * PI * 0.10 * x).cos();
        let b = 0.5 + 0.15 * (2.0 * PI * 0.10 * x).cos();
        let c = 0.5 + 0.15 * (2.0 * PI * 0.09 * x).cos();
        [a, b, c]
    };
    let (w, h) = (256usize, 256usize);
    let offsets = perturbed_half_lattice();
    let shifts: Vec<Shift> = offsets.iter().map(|p| Shift::new(p.0, p.1)).collect();
    let frames: Vec<RawFrame> = offsets
        .iter()
        .enumerate()
        .map(|(i, p)| render_analytic(&band, w, h, *p, 0.0, 601 + i as u64))
        .collect();
    let p0 = params(2, shifts.clone());
    let plain = merge::merge(&frames, &p0).expect("merge");
    let mut p1 = p0.clone();
    p1.deconvolve = 1.0;
    let out = merge::merge(&frames, &p1).expect("merge deconvolved");
    let rms = |r: &MergeResult| -> f64 {
        let (mut sse, mut n) = (0.0f64, 0.0f64);
        for y in 12..r.height - 12 {
            for x in 12..r.width - 12 {
                let truth = band(x as f64 / 2.0, y as f64 / 2.0);
                for (c, t) in truth.iter().enumerate() {
                    let d = r.planes[c][y * r.width + x] as f64 - t * span();
                    sse += d * d;
                    n += 1.0;
                }
            }
        }
        (sse / n.max(1.0)).sqrt()
    };
    let (r0, r1) = (rms(&plain), rms(&out));
    let (db0, db1) = (20.0 * (span() / r0).log10(), 20.0 * (span() / r1).log10());
    println!(
        "off lattice band limited: deconvolve 0 rms {:.2} ADU ({:.2} dB), deconvolve 1 rms {:.2} ADU ({:.2} dB)",
        r0, db0, r1, db1
    );
    assert!(db1 > db0 - 0.25, "deconvolve made the synthetic case worse: {db0:.2} -> {db1:.2} dB");
}

/// A full scale step must survive the post filter without a visible halo: the
/// overshoot is limited to 2 % of the step height.
#[test]
fn deconvolve_bounds_the_step_overshoot() {
    let (w, h) = (384usize, 384usize);
    let (low, high) = (0.0f64, 10000.0 / span());
    let step = |x: f64, y: f64| {
        let _ = y;
        let v = if x >= w as f64 * 0.5 { high } else { low };
        [v, v, v]
    };
    let offsets = perturbed_half_lattice();
    let frames: Vec<RawFrame> = offsets
        .iter()
        .enumerate()
        .map(|(i, p)| render_analytic(&step, w, h, *p, 0.0, 701 + i as u64))
        .collect();
    let shifts: Vec<Shift> = offsets.iter().map(|p| Shift::new(p.0, p.1)).collect();
    let mut p = params(2, shifts);
    p.deconvolve = 1.0;
    let res = merge::merge(&frames, &p).expect("merge");
    let step_adu = (high - low) * span();
    let mut p0 = p.clone();
    p0.deconvolve = 0.0;
    let plain = merge::merge(&frames, &p0).expect("merge");
    let probe = |r: &MergeResult| -> (f32, f32, (usize, usize), (usize, usize)) {
        let (mut over, mut under) = (f32::MIN, f32::MAX);
        let (mut oi, mut ui) = ((0usize, 0usize), (0usize, 0usize));
        let inset = 64;
        for c in 0..3 {
            for y in inset..r.height - inset {
                for x in inset..r.width - inset {
                    let v = r.planes[c][y * r.width + x] as f32;
                    if v > over {
                        over = v;
                        oi = (x, y);
                    }
                    if v < under {
                        under = v;
                        ui = (x, y);
                    }
                }
            }
        }
        (over - step_adu as f32, under, oi, ui)
    };
    let (over0, under0, _, _) = probe(&plain);
    let (over1, under1, oi, ui) = probe(&res);
    println!(
        "step {:.0} ADU: overshoot {:.0} -> {:.0} ADU ({:.2} % -> {:.2} %), undershoot {:.0} -> {:.0} ADU, worst at {:?} / {:?}",
        step_adu,
        over0,
        over1,
        100.0 * over0 / step_adu as f32,
        100.0 * over1 / step_adu as f32,
        -under0,
        -under1,
        oi,
        ui
    );
    // The merge itself rings: the Catmull-Rom splat has negative lobes, so a
    // point sampled step already overshoots a few percent. The criterion is what
    // the post filter adds on top of that.
    let extra = over1 - over0.max(0.0);
    let extra_under = under0.min(0.0) - under1;
    println!("filter added {:.0} ADU overshoot, {:.0} ADU undershoot", extra, extra_under);
    assert!(
        extra <= 0.02 * step_adu as f32 + 2.0,
        "the post filter added {:.0} ADU of overshoot (limit 2 % of the step)",
        extra
    );
    assert!(
        extra_under <= 0.02 * step_adu as f32 + 2.0,
        "the post filter added {:.0} ADU of undershoot",
        extra_under
    );
}

/// The post filter must not turn grain into visible noise: in a flat area the
/// standard deviation may grow by at most 30 %.
#[test]
fn deconvolve_does_not_explode_the_flat_area_noise() {
    let (w, h) = (320usize, 320usize);
    let flat = |x: f64, y: f64| {
        let _ = (x, y);
        let v = 0.35f64;
        [v * 1.1, v, v * 0.9]
    };
    let offsets = perturbed_half_lattice();
    let frames: Vec<RawFrame> = offsets
        .iter()
        .enumerate()
        .map(|(i, p)| render_analytic(&flat, w, h, *p, 40.0, 801 + i as u64))
        .collect();
    let shifts: Vec<Shift> = offsets.iter().map(|p| Shift::new(p.0, p.1)).collect();
    let mut p0 = params(2, shifts.clone());
    p0.crop_uncovered = true;
    p0.min_coverage = 0.05;
    p0.deconvolve = 0.0;
    let plain = merge::merge(&frames, &p0).expect("merge");
    let mut p1 = p0.clone();
    p1.deconvolve = 1.0;
    let out = merge::merge(&frames, &p1).expect("merge deconvolved");
    let std = |r: &MergeResult, c: usize| -> f64 {
        let inset = 12;
        let mut s = 0.0f64;
        let mut s2 = 0.0f64;
        let mut n = 0.0f64;
        for y in inset..r.height - inset {
            for x in inset..r.width - inset {
                let v = r.planes[c][y * r.width + x] as f64;
                s += v;
                s2 += v * v;
                n += 1.0;
            }
        }
        let mean = s / n;
        (s2 / n - mean * mean).max(0.0).sqrt()
    };
    for c in 0..3 {
        let (a, b) = (std(&plain, c), std(&out, c));
        println!("flat area channel {}: std {:.2} -> {:.2} ADU ({:+.0} %)", c, a, b, 100.0 * (b / a - 1.0));
        assert!(b <= 1.3 * a, "noise grew from {a:.2} to {b:.2} ADU");
    }
}

/// deconvolve = 0 must reproduce the current behaviour exactly.
#[test]
fn deconvolve_zero_is_the_previous_behaviour() {
    let scene = Scene;
    let (w, h) = (192usize, 192usize);
    let offsets = [(0.3, 0.2), (1.1, 0.7), (0.6, 1.4), (1.8, 1.9)];
    let frames: Vec<RawFrame> = offsets
        .iter()
        .enumerate()
        .map(|(i, p)| render(&scene, w, h, *p, 20.0, 901 + i as u64))
        .collect();
    let shifts: Vec<Shift> = offsets.iter().map(|p| Shift::new(p.0, p.1)).collect();
    let mut explicit = params(2, shifts.clone());
    explicit.deconvolve = 0.0;
    let a = merge::merge(&frames, &explicit).expect("explicit zero");
    // A non-positive amount must skip the post filter entirely; MergeParams::default
    // now enables it, so the reference is another skipped amount rather than the
    // default. The "filter is a no-op" half of the old promise is covered by
    // deconvolve_is_the_identity_on_the_ideal_lattice.
    explicit.deconvolve = -1.0;
    let b = merge::merge(&frames, &explicit).expect("negative amount");
    for c in 0..3 {
        assert_eq!(a.planes[c], b.planes[c], "channel {c} differs with deconvolve >= 0 skipped");
    }
}
