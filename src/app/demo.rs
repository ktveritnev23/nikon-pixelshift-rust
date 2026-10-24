//! Fabricated Pixel Shift data for the demo window, the headless selftest and
//! the preview widget tests: the GUI has to be usable before WP1/WP2 land.

use crate::app::pipeline::{self, SeriesInfo};
use crate::types::{Cfa, MergeResult, MergeStats, PreviewImage, RawFrame, Shift};
use anyhow::Result;
use std::io::Write;
use std::path::{Path, PathBuf};

pub const DEMO_WIDTH: usize = 640;
pub const DEMO_HEIGHT: usize = 424;
pub const DEMO_FRAMES: usize = 8;
pub const DEMO_BLACK: u16 = 1008;
pub const DEMO_WHITE: u16 = 16383;
pub const DEMO_WB: [f32; 4] = [1.244, 1.0, 2.010, 1.0];

/// The fabricated series shown by gui --demo and gui --selftest.
pub fn series() -> SeriesInfo {
    let dir = PathBuf::from("demo-series");
    let files = (0..DEMO_FRAMES)
        .map(|i| dir.join(format!("_DSC{:04}.NEF", 1001 + i)))
        .collect::<Vec<_>>();
    SeriesInfo::new(dir, files)
}

/// Half-pixel lattice used by the 8 frame mode: frame 0 is the reference.
pub fn shift_table() -> Vec<Shift> {
    // dx, dy, rot (rad), residual rms (px), confidence
    const TABLE: [(f64, f64, f64, f32, f32); 8] = [
        (0.0, 0.0, 0.0e-5, 0.010, 1.00),
        (0.5, 0.0, 0.7e-5, 0.011, 0.94),
        (0.0, 0.5, 0.9e-5, 0.012, 0.93),
        (0.5, 0.5, 0.8e-5, 0.013, 0.91),
        (1.0, 0.0, 1.1e-5, 0.012, 0.90),
        (1.5, 0.0, 1.3e-5, 0.014, 0.88),
        (1.0, 0.5, 1.4e-5, 0.015, 0.87),
        (1.5, 0.5, 1.5e-5, 0.016, 0.86),
    ];
    TABLE
        .iter()
        .map(|(dx, dy, rot, residual, confidence)| Shift {
            residual: *residual,
            confidence: *confidence,
            ..Shift::with_rot(*dx, *dy, *rot)
        })
        .collect()
}

pub fn shifts(frames: &[RawFrame]) -> Vec<Shift> {
    let mut table = shift_table();
    table.resize(frames.len(), Shift::default());
    table
}

// --------------------------------------------------------------- details -----

/// Spacing of the background grid, in demo frame pixels.
const GRID_STEP: f32 = 10.0;
/// Spacing of the concentric rings of the resolution target.
const RING_PERIOD: f32 = 6.5;
/// Half width of the finest stroke. With the one pixel antialias ramp below it
/// is a 1.7 px line: anything thinner would alias inside a single Bayer cell
/// and turn the 1:1 preview into moire instead of detail.
const STROKE: f32 = 0.35;
/// Half width of a ring: 2.5 px, so the target still reads when the canvas
/// magnifies the merged frame past 1:1.
const RING_STROKE: f32 = 0.75;

/// Coverage of a stroke of half width `half` at `distance` pixels from its
/// centre, antialiased over the last pixel.
fn stroke(distance: f32, half: f32) -> f32 {
    (half + 0.5 - distance).clamp(0.0, 1.0)
}

/// Thin grid over the whole frame: straight lines, so a lens correction bends
/// them visibly.
fn grid(x: f32, y: f32) -> f32 {
    let dx = (x - GRID_STEP * (x / GRID_STEP).round()).abs();
    let dy = (y - GRID_STEP * (y / GRID_STEP).round()).abs();
    stroke(dx, STROKE).max(stroke(dy, STROKE))
}

/// Concentric rings around the centre: the resolution target that still shows
/// structure when the preview is magnified past 1:1.
fn rings(x: f32, y: f32) -> f32 {
    let (cx, cy) = (DEMO_WIDTH as f32 * 0.5, DEMO_HEIGHT as f32 * 0.47);
    let radius = ((x - cx) * (x - cx) + (y - cy) * (y - cy)).sqrt();
    if !(12.0..=118.0).contains(&radius) {
        return 0.0;
    }
    let phase = radius - RING_PERIOD * (radius / RING_PERIOD).round();
    let fade = ((118.0 - radius) / 16.0).clamp(0.0, 1.0) * ((radius - 12.0) / 8.0).clamp(0.0, 1.0);
    stroke(phase.abs(), RING_STROKE) * fade
}

/// Axis aligned rectangle outline (x0, y0, x1, y1) with a stroke of half width
/// `half`; the distance to the border is the distance to the unsigned box.
fn outline(x: f32, y: f32, rect: [f32; 4], half: f32) -> f32 {
    let (cx, cy) = ((rect[0] + rect[2]) * 0.5, (rect[1] + rect[3]) * 0.5);
    let dx = (x - cx).abs() - (rect[2] - rect[0]) * 0.5;
    let dy = (y - cy).abs() - (rect[3] - rect[1]) * 0.5;
    let signed = if dx > 0.0 || dy > 0.0 {
        (dx.max(0.0) * dx.max(0.0) + dy.max(0.0) * dy.max(0.0)).sqrt()
    } else {
        dx.max(dy)
    };
    stroke(signed.abs(), half)
}

/// Two dark and two bright frames: sharp 2 px borders in both directions of the
/// tone curve.
const DARK_FRAMES: [[f32; 4]; 2] = [[56.0, 44.0, 196.0, 148.0], [452.0, 44.0, 592.0, 148.0]];
const LIGHT_FRAMES: [[f32; 4]; 2] = [[48.0, 268.0, 200.0, 372.0], [440.0, 268.0, 592.0, 372.0]];
/// Bright border just inside the frame. A lens correction bends it away from
/// the straight edge of the image itself, which is what makes the curvature
/// readable in the distortion screenshot.
const BORDER: [f32; 4] = [10.0, 10.0, 630.0, 414.0];

/// Short strokes of the two rulers along the bottom and the left edge.
fn ticks(x: f32, y: f32) -> f32 {
    let step = 12.0;
    let mut coverage: f32 = 0.0;
    let dx = (x - step * ((x - 12.0) / step).round()).abs();
    let dy = (y - 400.0).abs();
    if dx < 2.0 && dy < 8.0 {
        coverage = stroke(dx, STROKE) * (8.0 - dy).clamp(0.0, 1.0);
    }
    let dx = (y - step * ((y - 14.0) / step).round()).abs();
    let dy = (x - 14.0).abs();
    if dx < 2.0 && dy < 6.0 {
        coverage = coverage.max(stroke(dx, STROKE) * (6.0 - dy).clamp(0.0, 1.0));
    }
    coverage
}

/// Deterministic demo scene: a low contrast colour gradient overlaid with a
/// high frequency detail layer, so 1:1 previews and README screenshots show
/// what the tool is for. u and v are normalised image coordinates.
fn scene(u: f32, v: f32) -> [f32; 3] {
    let tau = std::f32::consts::TAU;
    // Kept well below 0.5: the preview applies the camera white balance (blue
    // is scaled by ~2), and a brighter background would clip the detail.
    let r = 0.22 + 0.08 * (tau * (0.9 * u + 0.05)).sin();
    let g = 0.27 + 0.07 * (tau * (1.1 * v + 0.35)).cos();
    let b = 0.12 + 0.04 * (tau * (0.7 * u + 0.30)).sin();
    let x = u * DEMO_WIDTH as f32;
    let y = v * DEMO_HEIGHT as f32;

    let dark = DARK_FRAMES
        .iter()
        .map(|rect| outline(x, y, *rect, 1.0))
        .fold(0.0, f32::max);
    let warm = outline(x, y, LIGHT_FRAMES[0], 1.0);
    let cool = outline(x, y, LIGHT_FRAMES[1], 1.0);
    let border = outline(x, y, BORDER, 0.7);
    let ticks = ticks(x, y);
    let ink = (0.55 * grid(x, y)).max(0.90 * rings(x, y)).max(0.92 * dark);
    let dim = 1.0 - ink;
    let glow = 0.36 * ticks + 0.40 * border;
    [
        (r * dim + 0.42 * warm + 0.14 * cool + glow).clamp(0.01, 0.95),
        (g * dim + 0.34 * warm + 0.26 * cool + glow).clamp(0.01, 0.95),
        (b * dim + 0.14 * warm + 0.40 * cool + glow).clamp(0.01, 0.95),
    ]
}

fn cam_xyz() -> [[f32; 3]; 4] {
    [
        [0.92, -0.36, -0.09],
        [-0.42, 1.22, 0.16],
        [-0.08, 0.21, 0.69],
        [0.0, 0.0, 0.0],
    ]
}

fn rgb_cam() -> [[f32; 3]; 4] {
    [
        [1.04, -0.31, -0.05],
        [-0.27, 1.19, 0.12],
        [0.02, -0.22, 1.16],
        [0.0, 0.0, 0.0],
    ]
}

/// Fabricated Bayer frames of the demo series, shifted by shift_table().
pub fn frames() -> Vec<RawFrame> {
    let cfa = Cfa::from_pattern_str("RGGB").expect("RGGB");
    let span = (DEMO_WHITE - DEMO_BLACK) as f32;
    let table = shift_table();
    (0..DEMO_FRAMES)
        .map(|index| {
            let (dx, dy) = (table[index].dx as f32, table[index].dy as f32);
            let mut data = vec![0u16; DEMO_WIDTH * DEMO_HEIGHT];
            for y in 0..DEMO_HEIGHT {
                for x in 0..DEMO_WIDTH {
                    let color = cfa.0[y & 1][x & 1];
                    let u = (x as f32 + dx) / DEMO_WIDTH as f32;
                    let v = (y as f32 + dy) / DEMO_HEIGHT as f32;
                    let value = scene(u, v)[color.idx()];
                    data[y * DEMO_WIDTH + x] =
                        (DEMO_BLACK as f32 + value * span).clamp(0.0, DEMO_WHITE as f32) as u16;
                }
            }
            RawFrame {
                path: series().dir.join(format!("_DSC{:04}.NEF", 1001 + index)),
                width: DEMO_WIDTH,
                height: DEMO_HEIGHT,
                data,
                cfa,
                black: DEMO_BLACK,
                cblack: [0; 4],
                white: DEMO_WHITE,
                wb: DEMO_WB,
                cam_xyz: cam_xyz(),
                rgb_cam: rgb_cam(),
                iso: 100.0,
                shutter: 1.0 / 125.0,
                aperture: 5.6,
                focal: 50.0,
                lens_make: "VILTROX".to_string(),
                lens_model: "VILTROX 135mm F1.8 Z".to_string(),
                focal_35mm: 50,
                max_aperture: 1.8,
                lens_min_focal: 135.0,
                lens_max_focal: 135.0,
                lens_min_fnum: 1.8,
                lens_max_fnum: 1.8,
                lens_serial: String::new(),
                body_serial: String::new(),
                make: "NIKON CORPORATION".to_string(),
                model: "NIKON Z 6_3".to_string(),
                timestamp: "2024-05-01 12:00:00".to_string(),
                index,
            }
        })
        .collect()
}

/// Fabricated merge result: the same scene sampled on the output grid.
pub fn result(scale: u32, frames: &[RawFrame]) -> MergeResult {
    let (fw, fh) = frames
        .first()
        .map(|f| (f.width, f.height))
        .unwrap_or((DEMO_WIDTH, DEMO_HEIGHT));
    let scale = scale.max(1) as usize;
    let (width, height) = (fw * scale, fh * scale);
    let white = DEMO_WHITE - DEMO_BLACK;
    let mut planes = [
        vec![0u16; width * height],
        vec![0u16; width * height],
        vec![0u16; width * height],
    ];
    for y in 0..height {
        for x in 0..width {
            let u = (x as f32 + 0.5) / width as f32;
            let v = (y as f32 + 0.5) / height as f32;
            let rgb = scene(u, v);
            for (c, plane) in planes.iter_mut().enumerate() {
                plane[y * width + x] = (rgb[c] * white as f32).round() as u16;
            }
        }
    }
    MergeResult {
        width,
        height,
        scale: scale as u32,
        planes,
        white,
        shifts: shift_table(),
        stats: MergeStats {
            mean_samples_per_channel: frames.len() as f32 / 4.0,
            min_coverage: 1.0,
            filled_pixels: 0,
            frames: frames.len(),
            elapsed_ms: 1,
            masked_pct: vec![0.0; frames.len()],
        },
    }
}

/// Preview encoder used when merge::preview is not available yet.
///
/// `step` is the requested pyramid level for crops (1 = 1:1); the base
/// preview keeps deriving its own factor from max_dim.
pub fn encode(
    img: &MergeResult,
    rect: Option<(usize, usize, usize, usize)>,
    max_dim: u32,
    wb: [f32; 3],
    step: u32,
) -> PreviewImage {
    let (x0, y0, w, h) = rect
        .map(|(x, y, w, h)| {
            let x0 = x.min(img.width.saturating_sub(1));
            let y0 = y.min(img.height.saturating_sub(1));
            (
                x0,
                y0,
                w.min(img.width - x0).max(1),
                h.min(img.height - y0).max(1),
            )
        })
        .unwrap_or((0, 0, img.width, img.height));
    let step = match rect {
        Some(_) => step.max(1) as usize,
        None => pipeline::downscale_step(w, h, max_dim),
    };
    let out_w = (w / step).max(1);
    let out_h = (h / step).max(1);
    let white = img.white.max(1) as f32;
    let mut buf = vec![0.0f32; out_w * out_h * 3];
    for oy in 0..out_h {
        for ox in 0..out_w {
            let mut sum = [0.0f32; 3];
            let mut n = 0.0f32;
            for sy in 0..step {
                for sx in 0..step {
                    let x = x0 + ox * step + sx;
                    let y = y0 + oy * step + sy;
                    if x >= img.width || y >= img.height {
                        continue;
                    }
                    for c in 0..3 {
                        sum[c] += img.planes[c][y * img.width + x] as f32 / white;
                    }
                    n += 1.0;
                }
            }
            let idx = (oy * out_w + ox) * 3;
            for c in 0..3 {
                let mean = if n > 0.0 { sum[c] / n } else { 0.0 };
                buf[idx + c] = mean * wb[c];
            }
        }
    }
    crate::merge::encode_srgb(&buf, out_w, out_h)
}

/// Box average of a 2x result, used when merge::downsample_1x is unavailable.
pub fn downsample(img: &MergeResult) -> MergeResult {
    if img.scale <= 1 {
        return img.clone();
    }
    let width = img.width / 2;
    let height = img.height / 2;
    let mut planes = [
        vec![0u16; width * height],
        vec![0u16; width * height],
        vec![0u16; width * height],
    ];
    for c in 0..3 {
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0u32;
                for dy in 0..2 {
                    for dx in 0..2 {
                        sum += img.planes[c][(y * 2 + dy) * img.width + x * 2 + dx] as u32;
                    }
                }
                planes[c][y * width + x] = (sum / 4) as u16;
            }
        }
    }
    MergeResult {
        width,
        height,
        scale: 1,
        planes,
        white: img.white,
        shifts: img.shifts.clone(),
        stats: img.stats.clone(),
    }
}

/// Stand-in output used when the DNG/TIFF writers are not implemented yet.
pub fn write_ppm(path: &Path, img: &MergeResult) -> Result<u64> {
    let mut file = std::fs::File::create(path)?;
    write!(file, "P6\n{} {}\n255\n", img.width, img.height)?;
    let white = img.white.max(1) as f32;
    let mut row = Vec::with_capacity(img.width * 3);
    for y in 0..img.height {
        row.clear();
        for x in 0..img.width {
            for c in 0..3 {
                row.push(pipeline::tone(img.planes[c][y * img.width + x] as f32 / white));
            }
        }
        file.write_all(&row)?;
    }
    file.flush()?;
    Ok(std::fs::metadata(path)?.len())
}
