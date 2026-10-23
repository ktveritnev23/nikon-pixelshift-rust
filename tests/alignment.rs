//! Sub-pixel alignment tests (docs/TASKS.md, WP2).
//!
//! Frames are rendered physically from an analytic scene, so the injected shift
//! is the ground truth of the shift convention of types::Shift: frame pixel
//! (x, y) samples scene point (x + dx, y + dy) of the reference frame.

use pixelshift::shift;
use pixelshift::types::{Cfa, RawFrame, Shift};
use std::path::PathBuf;

const BLACK: u16 = 1008;
const WHITE: u16 = 16383;

fn span() -> f64 {
    WHITE as f64 - BLACK as f64
}

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

/// Analytic scene for alignment. Unlike the merge test scene it is strictly
/// band limited (highest spatial frequency about 0.14 cycles per pixel), which
/// is what the optical point spread function guarantees on a real capture:
/// content above the sampling Nyquist would alias differently in every frame and
/// no translation estimator could be unbiased.
struct Scene;

impl Scene {
    fn rgb(&self, x: f64, y: f64) -> [f64; 3] {
        use std::f64::consts::PI;
        let r = (x * x + y * y).sqrt();
        let s1 = (2.0 * PI * 0.090 * x).cos();
        let s2 = (2.0 * PI * 0.120 * y).sin();
        let s3 = (2.0 * PI * 0.070 * (0.7 * x - 0.6 * y)).cos();
        let rings = (0.35 * r).cos();
        let env = 0.5 + 0.5 * (2.0 * PI * 0.011 * x).cos();
        let red = 0.35 + 0.18 * s1 + 0.12 * rings * env + 0.10 * s3;
        let green = 0.35 + 0.18 * s2 + 0.12 * s3 * env + 0.08 * rings;
        let blue = 0.30 + 0.16 * s3 + 0.12 * s1 * env + 0.10 * s2;
        [
            red.clamp(0.02, 0.97),
            green.clamp(0.02, 0.97),
            blue.clamp(0.02, 0.97),
        ]
    }
}

fn render(scene: &Scene, w: usize, h: usize, s: (f64, f64), sigma: f64, seed: u64) -> RawFrame {
    render_rigid(scene, w, h, Shift::new(s.0, s.1), sigma, seed)
}

/// Render a frame as seen through the rigid transform of types::Shift::map.
fn render_rigid(scene: &Scene, w: usize, h: usize, s: Shift, sigma: f64, seed: u64) -> RawFrame {
    let cfa = Cfa::from_pattern_str("RGGB").expect("RGGB");
    let mut rng = Rng::new(seed);
    let mut data = vec![0u16; w * h];
    let (cx, cy) = (w as f64 * 0.5, h as f64 * 0.5);
    for y in 0..h {
        for x in 0..w {
            let c = cfa.0[y & 1][x & 1];
            let (sx, sy) = s.map(x as f64, y as f64, cx, cy);
            let v = scene.rgb(sx, sy)[c.idx()] * span()
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

fn check(reference: &RawFrame, frame: &RawFrame, want: (f64, f64), tol: f64) -> Shift {
    let got = shift::estimate(reference, frame).expect("estimate");
    println!(
        "injected ({:+.3}, {:+.3}) -> measured ({:+.4}, {:+.4}) confidence {:.3}",
        want.0, want.1, got.dx, got.dy, got.confidence
    );
    assert!(
        (got.dx - want.0).abs() <= tol && (got.dy - want.1).abs() <= tol,
        "shift ({:+.4}, {:+.4}) is off by more than {tol} px from ({:+.3}, {:+.3})",
        got.dx,
        got.dy,
        want.0,
        want.1
    );
    assert!(got.confidence > 0.9, "confidence {:.3} is low", got.confidence);
    got
}

#[test]
fn recovers_half_and_arbitrary_sub_pixel_shifts() {
    let scene = Scene;
    let (w, h) = (512usize, 512usize);
    let reference = render(&scene, w, h, (0.0, 0.0), 0.0, 5);
    let cases = [
        (0.5, 0.0),
        (0.0, 0.5),
        (0.5, 0.5),
        (1.0, 1.0),
        (1.5, 0.5),
        (0.37, -0.82),
        (-1.25, 0.75),
        (2.25, 1.75),
    ];
    let mut worst: f64 = 0.0;
    for (i, c) in cases.iter().enumerate() {
        let frame = render(&scene, w, h, *c, 0.0, 100 + i as u64);
        let got = check(&reference, &frame, *c, 0.02);
        worst = worst.max((got.dx - c.0).abs()).max((got.dy - c.1).abs());
    }
    println!("worst clean error {:.4} px", worst);
    // Acceptance (docs/TASKS.md, WP2) is 0.02 px; the measured worst case is
    // about 0.016 px, dominated by the residual interpolation bias of the
    // six tap Lanczos kernel used on the working plane.
    assert!(worst <= 0.02, "worst clean error {worst:.4} px");
}

#[test]
fn recovers_shifts_with_sensor_noise() {
    let scene = Scene;
    let (w, h) = (512usize, 512usize);
    let sigma = 50.0;
    let reference = render(&scene, w, h, (0.0, 0.0), sigma, 21);
    let cases = [(0.5, 0.0), (0.5, 0.5), (1.0, 0.0), (1.5, 0.5), (0.25, -1.75)];
    let mut worst: f64 = 0.0;
    for (i, c) in cases.iter().enumerate() {
        let frame = render(&scene, w, h, *c, sigma, 200 + i as u64);
        let got = check(&reference, &frame, *c, 0.02);
        worst = worst.max((got.dx - c.0).abs()).max((got.dy - c.1).abs());
    }
    println!("worst noisy (sigma {sigma}) error {:.4} px", worst);
    assert!(worst <= 0.02, "worst noisy error {worst:.4} px");
}

#[test]
fn sensor_moved_right_gives_positive_dx() {
    // types::Shift: a frame physically shifted half a pixel to the right samples
    // scene(x + 0.5), which must be reported as dx = +0.5 (the content moves to
    // the left in the frame, the sensor moves to the right).
    let scene = Scene;
    let (w, h) = (384usize, 384usize);
    let reference = render(&scene, w, h, (0.0, 0.0), 0.0, 7);
    let right = render(&scene, w, h, (0.5, 0.0), 0.0, 8);
    let left = render(&scene, w, h, (-0.5, 0.0), 0.0, 9);
    let up = render(&scene, w, h, (0.0, 0.5), 0.0, 10);
    let sr = shift::estimate(&reference, &right).expect("right");
    let sl = shift::estimate(&reference, &left).expect("left");
    let su = shift::estimate(&reference, &up).expect("up");
    println!(
        "right {:.4} left {:.4} down-positive {:.4}",
        sr.dx, sl.dx, su.dy
    );
    assert!(sr.dx > 0.4 && sr.dx < 0.6, "right shift dx {:.4}", sr.dx);
    assert!(sr.dy.abs() < 0.02, "right shift dy {:.4}", sr.dy);
    assert!(sl.dx < -0.4 && sl.dx > -0.6, "left shift dx {:.4}", sl.dx);
    assert!(su.dy > 0.4 && su.dy < 0.6, "down shift dy {:.4}", su.dy);
}

#[test]
fn identical_frames_give_zero_shift() {
    let scene = Scene;
    let (w, h) = (256usize, 256usize);
    let a = render(&scene, w, h, (0.0, 0.0), 0.0, 3);
    let s = shift::estimate(&a, &a).expect("identity");
    println!("identity ({:.5}, {:.5}) confidence {:.3}", s.dx, s.dy, s.confidence);
    assert!(s.dx.abs() < 0.005 && s.dy.abs() < 0.005, "identity is not zero");
    assert!(s.is_identity());
}

#[test]
fn series_shifts_are_consistent_with_frame_zero() {
    let scene = Scene;
    let (w, h) = (384usize, 384usize);
    let injected = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.0, 0.5),
        (0.5, 0.5),
        (1.0, 0.5),
        (1.5, 1.0),
        (0.25, 1.75),
        (2.0, 2.5),
    ];
    let frames: Vec<RawFrame> = injected
        .iter()
        .enumerate()
        .map(|(i, s)| render(&scene, w, h, *s, 20.0, 900 + i as u64))
        .collect();
    let shifts = shift::estimate_series(&frames).expect("series");
    assert_eq!(shifts.len(), frames.len());
    assert!(shifts[0].is_identity());
    let mut worst: f64 = 0.0;
    for (i, want) in injected.iter().enumerate() {
        println!(
            "frame {}: injected ({:+.3}, {:+.3}) measured ({:+.4}, {:+.4}) confidence {:.3}",
            i, want.0, want.1, shifts[i].dx, shifts[i].dy, shifts[i].confidence
        );
        worst = worst
            .max((shifts[i].dx - want.0).abs())
            .max((shifts[i].dy - want.1).abs());
    }
    println!("worst series error {:.4} px", worst);
    assert!(worst <= 0.02, "worst series error {worst:.4} px");
}

#[test]
fn rotation_sign_and_magnitude_are_recovered() {
    // A frame rendered with Shift::with_rot(dx, dy, rot) is, by definition, the
    // reference seen through a rotation of rot about the centre; the estimate
    // must return the same sign and the same magnitude.
    let scene = Scene;
    let (w, h) = (1024usize, 1024usize);
    let reference = render_rigid(&scene, w, h, Shift::new(0.0, 0.0), 0.0, 41);
    for rot in [5e-4f64, -5e-4] {
        let frame = render_rigid(&scene, w, h, Shift::with_rot(0.75, -0.5, rot), 0.0, 42);
        let s = shift::estimate(&reference, &frame).expect("estimate");
        println!(
            "injected rot {:+.2e} (arcsec {:+.1}) -> {:+.3e} (arcsec {:+.1}), dx {:.3} dy {:.3}, residual {:.3} px, conf {:.3}",
            rot,
            rot.to_degrees() * 3600.0,
            s.rot,
            s.rot_arcsec(),
            s.dx,
            s.dy,
            s.residual,
            s.confidence
        );
        assert!(
            (s.rot - rot).abs() < 0.15 * rot.abs(),
            "rotation {:.3e} is off from {:.3e}",
            s.rot,
            rot
        );
        assert!((s.dx - 0.75).abs() < 0.02, "dx {:.4}", s.dx);
        assert!((s.dy + 0.5).abs() < 0.02, "dy {:.4}", s.dy);
        assert!(s.residual < 0.02, "residual {:.4} px", s.residual);
    }
}

#[test]
fn tiny_rotation_is_reported_with_the_right_sign() {
    // 1e-4 rad is the magnitude measured on the real series; a pure translation
    // fit cannot express it, so both the sign and the arc second value matter.
    let scene = Scene;
    let (w, h) = (1024usize, 1024usize);
    let reference = render_rigid(&scene, w, h, Shift::new(0.0, 0.0), 20.0, 51);
    let frame = render_rigid(&scene, w, h, Shift::with_rot(-0.6, 0.4, 1e-4), 20.0, 52);
    let s = shift::estimate(&reference, &frame).expect("estimate");
    println!(
        "1e-4 rad: measured {:+.3e} rad ({:+.1} arcsec) dx {:.4} dy {:.4} residual {:.4} conf {:.3}",
        s.rot,
        s.rot_arcsec(),
        s.dx,
        s.dy,
        s.residual,
        s.confidence
    );
    assert!(s.rot > 0.0, "rotation sign is wrong: {:.3e}", s.rot);
    assert!(
        (s.rot - 1e-4).abs() < 5e-5,
        "rotation {:.3e} rad is not within 5e-5 of 1e-4",
        s.rot
    );
    assert!((s.dx + 0.6).abs() < 0.02 && (s.dy - 0.4).abs() < 0.02);
}

#[test]
fn luma_projection_ignores_colour_only_content() {
    // The estimator works on the CFA phase invariant block sum, so a scene that
    // differs only in chroma still registers on its luma structure.
    let scene = Scene;
    let (w, h) = (256usize, 256usize);
    let a = render(&scene, w, h, (0.0, 0.0), 0.0, 31);
    let b = render(&scene, w, h, (1.5, -0.5), 0.0, 32);
    let s = shift::estimate(&a, &b).expect("estimate");
    println!("block sum projection: ({:.4}, {:.4})", s.dx, s.dy);
    assert!((s.dx - 1.5).abs() < 0.01 && (s.dy + 0.5).abs() < 0.01);
}
