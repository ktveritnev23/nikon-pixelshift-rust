//! Measurement of the lens distortion from a reference image.
//!
//! A raw editor refuses to apply its lens profile to a resampled DNG (Capture
//! One only profiles a handful of Z lenses and falls back to what the maker
//! note holds), so the correction has to be baked in here. Guessing the
//! coefficients is what the slider is for; this module measures them: the user
//! exports one frame of the series from the editor with the profile enabled and
//! we fit the radial model to the measured displacement.
//!
//! # Sign convention
//!
//! The merged frame is the *recorded*, distorted picture; the reference is the
//! *corrected* one. [`crate::distort::rectify`] samples the recorded radius
//! r * (1 + k1 r^2 + k2 r^4 + k3 r^6) for an ideal pixel at normalised radius r,
//! so a scene feature that sits at radius r in the reference sits at that
//! stretched radius in the merged frame. The displacement measured by block
//! matching, d = position in the merged frame - position in the reference, is
//! therefore
//!
//! \`\`\`text
//! d(x, y) = shift + (x - cx) * f(r2)
//! f(r2)   = k1 r2 + k2 r2^2 + k3 r2^3
//! r2      = ((x - cx)^2 + (y - cy)^2) / diag2     (diag2 = half diagonal^2)
//! \`\`\`
//!
//! and the fit returns exactly the coefficients rectify() consumes: a 4 %
//! barrel lens measures k1 = -0.04. Because r is normalised to the half
//! diagonal, the coefficients are independent of the output scale, so a fit on
//! the 1x grid applies unchanged to a 2x merge.
//!
//! # Method
//!
//! The frame is split into a COLS x ROWS grid of blocks. Each block takes a
//! square template from the reference and locates it in the merged frame by
//! zero-mean normalised cross correlation, coarsely on a 1/8 downsample, then
//! on a 1/2 downsample and at full resolution; the integer offset is refined to
//! a fraction of a pixel by Gauss-Newton on the SSD with a per-block gain and
//! offset, so the tone curve of the export does not bias the result. Blocks
//! without structure or without a convincing peak are dropped, the rest go
//! through a linear least squares fit of the five unknowns (shift x, shift y,
//! k1, k2, k3).

use crate::distort::Distortion;
use crate::types::MergeResult;
use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use std::path::Path;

pub mod tiffread;

/// Blocks across and down the frame. More than the 6 x 4 of the first sketch:
/// three radial terms need radii spread over the whole range, and the blocks at
/// the corners carry most of the k3 leverage.
const COLS: usize = 10;
const ROWS: usize = 8;
const MIN_TEMPLATE: usize = 8;
const MAX_TEMPLATE: usize = 128;
/// Template half-size cap for the full resolution search and the refinement.
const MAX_TEMPLATE_REFINE: usize = 48;
/// Smallest correlation a block must reach to be used.
const MIN_NCC: f64 = 0.5;
/// Smallest luminance standard deviation (0..=1 scale) of the reference
/// template: a flat patch has no locatable peak.
const MIN_STD: f32 = 1.5e-3;

/// Outcome of a fit, ready for the CLI report and the GUI label.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FitReport {
    pub k1: f64,
    pub k2: f64,
    pub k3: f64,
    /// Global translation between the two images in 1x pixels (reference ->
    /// merged frame), which the radial terms cannot absorb.
    pub shift: (f64, f64),
    /// RMS of the difference between the measured displacements and the model,
    /// over both axes, in 1x pixels.
    pub residual_rms: f64,
    /// Blocks that produced a usable displacement.
    pub blocks: usize,
    /// Distance of the farthest used block from the image centre, in 1x pixels.
    pub max_radius_px: f64,
}

impl FitReport {
    pub fn distortion(&self) -> Distortion {
        Distortion::new3(self.k1, self.k2, self.k3)
    }

    /// One line for the CLI and the GUI log.
    pub fn summary(&self) -> String {
        format!(
            "k1 {:+.5} k2 {:+.5} k3 {:+.5}, shift {:+.2}/{:+.2} px, residual rms {:.3} px, {} blocks, max radius {:.0} px",
            self.k1,
            self.k2,
            self.k3,
            self.shift.0,
            self.shift.1,
            self.residual_rms,
            self.blocks,
            self.max_radius_px
        )
    }
}

/// Monochrome work image on the 1x pixel grid.
struct Gray {
    w: usize,
    h: usize,
    lum: Vec<f32>,
}

impl Gray {
    fn at(&self, x: isize, y: isize) -> f32 {
        let xi = x.clamp(0, self.w as isize - 1) as usize;
        let yi = y.clamp(0, self.h as isize - 1) as usize;
        self.lum[yi * self.w + xi]
    }

    /// Bilinear sample with its gradient; the clamp only guards the last row
    /// and column, the callers keep the position inside the frame.
    fn sample_with_grad(&self, x: f64, y: f64) -> (f64, f64, f64) {
        let xf = x.clamp(0.0, (self.w - 1) as f64);
        let yf = y.clamp(0.0, (self.h - 1) as f64);
        let x0 = xf.floor();
        let y0 = yf.floor();
        let xi = x0 as usize;
        let yi = y0 as usize;
        let x1 = (xi + 1).min(self.w - 1);
        let y1 = (yi + 1).min(self.h - 1);
        let tx = xf - x0;
        let ty = yf - y0;
        let s00 = self.lum[yi * self.w + xi] as f64;
        let s10 = self.lum[yi * self.w + x1] as f64;
        let s01 = self.lum[y1 * self.w + xi] as f64;
        let s11 = self.lum[y1 * self.w + x1] as f64;
        let top = s00 + (s10 - s00) * tx;
        let bottom = s01 + (s11 - s01) * tx;
        let value = top + (bottom - top) * ty;
        let gx = (s10 - s00) * (1.0 - ty) + (s11 - s01) * ty;
        let gy = bottom - top;
        (value, gx, gy)
    }

    /// Binomial low pass (1 4 6 4 1)/16 with clamped borders. Downsampling
    /// without it aliases the fine detail into a lattice of false correlation
    /// peaks, which is exactly how a coarse pyramid picks the wrong shift.
    fn low_pass(&self) -> Gray {
        const KERNEL: [f32; 5] = [0.0625, 0.25, 0.375, 0.25, 0.0625];
        let mut rows = vec![0f32; self.w * self.h];
        for y in 0..self.h {
            let base = y * self.w;
            for x in 0..self.w {
                let mut sum = 0.0f32;
                for (k, weight) in KERNEL.iter().enumerate() {
                    let sx = (x as isize + k as isize - 2).clamp(0, self.w as isize - 1) as usize;
                    sum += self.lum[base + sx] * weight;
                }
                rows[base + x] = sum;
            }
        }
        let mut lum = vec![0f32; self.w * self.h];
        for y in 0..self.h {
            for x in 0..self.w {
                let mut sum = 0.0f32;
                for (k, weight) in KERNEL.iter().enumerate() {
                    let sy = (y as isize + k as isize - 2).clamp(0, self.h as isize - 1) as usize;
                    sum += rows[sy * self.w + x] * weight;
                }
                lum[y * self.w + x] = sum;
            }
        }
        Gray {
            w: self.w,
            h: self.h,
            lum,
        }
    }

    /// Low pass, then drop every second sample; odd trailing rows and columns
    /// disappear.
    fn downsample(&self) -> Gray {
        let blurred = self.low_pass();
        let w = self.w / 2;
        let h = self.h / 2;
        let mut lum = vec![0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                lum[y * w + x] = blurred.lum[(2 * y) * self.w + 2 * x];
            }
        }
        Gray { w, h, lum }
    }
}

/// One measured displacement at a block centre (reference coordinates).
#[derive(Clone, Copy, Debug)]
struct Sample {
    x: f64,
    y: f64,
    dx: f64,
    dy: f64,
}

/// Measure the distortion of `img` against `reference`.
pub fn fit_from_reference(img: &MergeResult, reference: &Path) -> Result<FitReport> {
    if img.width == 0 || img.height == 0 {
        bail!("the merged image is empty ({} x {})", img.width, img.height);
    }
    let loaded = load_reference(reference)?;
    let ours = gray_from_merge(img);
    if ours.w != loaded.w || ours.h != loaded.h {
        bail!(
            "reference {} is {} x {} but the merged frame is {} x {} on the 1x grid: export the reference at the output resolution of the series (a 2x merge is averaged down to 1x for the fit)",
            reference.display(),
            loaded.w,
            loaded.h,
            ours.w,
            ours.h
        );
    }
    fit_gray(&ours, &loaded)
}

/// Weighted luminance of the merged frame, always on the 1x grid.
fn gray_from_merge(img: &MergeResult) -> Gray {
    let (w, h) = (img.width, img.height);
    let scale = 1.0 / img.white.max(1) as f32;
    let mut lum = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let index = y * w + x;
            let mut value = 0.0f32;
            for (c, weight) in [0.299f32, 0.587, 0.114].iter().enumerate() {
                value += img.planes[c][index] as f32 * weight;
            }
            lum[index] = value * scale;
        }
    }
    let gray = Gray { w, h, lum };
    if img.scale >= 2 {
        gray.downsample()
    } else {
        gray
    }
}

fn gray_from_rgb(width: usize, height: usize, planes: &[Vec<f32>; 3]) -> Gray {
    let mut lum = vec![0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            lum[index] =
                0.299 * planes[0][index] + 0.587 * planes[1][index] + 0.114 * planes[2][index];
        }
    }
    Gray {
        w: width,
        h: height,
        lum,
    }
}

/// Read the reference image: a plain RGB TIFF first (what the editor exports),
/// any raw file LibRaw can demosaic as the fallback.
fn load_reference(path: &Path) -> Result<Gray> {
    let tiff_error = match tiffread::read_rgb(path) {
        Ok(rgb) => return Ok(gray_from_rgb(rgb.width, rgb.height, &rgb.planes)),
        Err(error) => format!("{error:#}"),
    };
    match crate::rawio::shim::render_rgb(path) {
        Ok(rendered) => {
            let count = rendered.width * rendered.height;
            let mut planes = [vec![0f32; count], vec![0f32; count], vec![0f32; count]];
            for index in 0..count {
                for c in 0..3 {
                    planes[c][index] = rendered.rgb[index * 3 + c] as f32 / 255.0;
                }
            }
            Ok(gray_from_rgb(rendered.width, rendered.height, &planes))
        }
        Err(raw) => Err(raw).context(format!(
            "cannot read the reference {}: not a usable RGB TIFF ({tiff_error}), and LibRaw refused it too",
            path.display()
        )),
    }
}

fn fit_gray(ours: &Gray, refs: &Gray) -> Result<FitReport> {
    let (w, h) = (ours.w, ours.h);
    if w < 4 * MIN_TEMPLATE || h < 4 * MIN_TEMPLATE {
        bail!("the frame is too small for a block fit ({w} x {h})");
    }
    let coarse = ours.downsample();
    let coarse_ref = refs.downsample();
    let far = coarse.downsample().downsample();
    let far_ref = coarse_ref.downsample().downsample();

    let block_w = (w / COLS).max(1);
    let block_h = (h / ROWS).max(1);
    let tpl = (block_w.min(block_h) / 2).clamp(MIN_TEMPLATE, MAX_TEMPLATE);
    let centre = ((w as f64 - 1.0) * 0.5, (h as f64 - 1.0) * 0.5);
    let half_far = (tpl / 8).clamp(4, 8) as isize;
    let half_coarse = (tpl / 2).clamp(6, MAX_TEMPLATE_REFINE) as isize;
    let half_full = tpl.min(MAX_TEMPLATE_REFINE) as isize;

    let bounds = [half_full, w as isize - 1 - half_full, half_full, h as isize - 1 - half_full];
    if bounds[1] <= bounds[0] || bounds[3] <= bounds[2] {
        bail!("the frame is too small for the block grid ({w} x {h})");
    }

    let jobs: Vec<(usize, usize)> = (0..ROWS)
        .flat_map(|row| (0..COLS).map(move |col| (col, row)))
        .collect();
    let samples: Vec<Sample> = jobs
        .par_iter()
        .filter_map(|&(col, row)| {
            let centre_block = (
                ((col as f64 + 0.5) * block_w as f64).round() as isize,
                ((row as f64 + 0.5) * block_h as f64).round() as isize,
            );
            let levels = [&far, &far_ref, &coarse, &coarse_ref, ours, refs];
            match_one(
                &levels,
                centre_block,
                centre,
                [half_far, half_coarse, half_full],
                bounds,
            )
        })
        .collect();

    let report = solve_model(&samples, w, h)?;
    if report.blocks < 8 {
        bail!(
            "only {} of {} blocks matched: the reference must show the same frame, sharp, with the lens profile applied",
            report.blocks,
            COLS * ROWS
        );
    }
    Ok(report)
}

/// Locate one block: coarse pyramid search, then full resolution, then a
/// sub-pixel refinement.
fn match_one(
    levels: &[&Gray; 6],
    block: (isize, isize),
    image_centre: (f64, f64),
    halves: [isize; 3],
    bounds: [isize; 4],
) -> Option<Sample> {
    let [far, far_ref, coarse, coarse_ref, full, full_ref] = *levels;
    let [half_far, half_coarse, half_full] = halves;
    let [lo_x, hi_x, lo_y, hi_y] = bounds;
    let bx = block.0.clamp(lo_x, hi_x);
    let by = block.1.clamp(lo_y, hi_y);

    let radius = {
        let dx = bx as f64 - image_centre.0;
        let dy = by as f64 - image_centre.1;
        (dx * dx + dy * dy).sqrt()
    };
    // A 10 % radial displacement plus a global shift; the corner of a 24 mm
    // frame needs 300 px and more, so the search has to scale with the radius.
    let reach = (0.10 * radius + 24.0).ceil();

    // Every level clamps its own patch centre so the corners stay usable; a
    // clamped centre moves the measured position along the smooth field by a
    // hundredth of a pixel at most, which the residual absorbs.
    let far_centre = (
        (bx / 8).clamp(half_far, far.w as isize - 1 - half_far),
        (by / 8).clamp(half_far, far.h as isize - 1 - half_far),
    );
    let coarse_centre = (
        (bx / 2).clamp(half_coarse, coarse.w as isize - 1 - half_coarse),
        (by / 2).clamp(half_coarse, coarse.h as isize - 1 - half_coarse),
    );
    let (_, far_x, far_y) = search(
        far,
        far_ref,
        far_centre,
        half_far,
        (0, 0),
        (reach / 8.0) as isize + 1,
    )?;
    let (_, coarse_x, coarse_y) = search(
        coarse,
        coarse_ref,
        coarse_centre,
        half_coarse,
        (4 * far_x, 4 * far_y),
        6,
    )?;
    let (score, ix, iy) = search(
        full,
        full_ref,
        (bx, by),
        half_full,
        (2 * coarse_x, 2 * coarse_y),
        3,
    )?;
    if score < MIN_NCC {
        return None;
    }

    // Structure check on the reference template: a flat patch can reach a high
    // correlation by accident and would drag the fit.
    let mut sum = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut count = 0.0f64;
    for dy in -half_full..=half_full {
        for dx in -half_full..=half_full {
            let value = full_ref.at(bx + dx, by + dy) as f64;
            sum += value;
            sum2 += value * value;
            count += 1.0;
        }
    }
    let variance = (sum2 / count - (sum / count).powi(2)).max(0.0);
    if (variance as f32).sqrt() < MIN_STD {
        return None;
    }

    let (u, v) = refine(full, full_ref, (bx, by), (ix, iy), half_full)?;
    Some(Sample {
        x: bx as f64,
        y: by as f64,
        dx: ix as f64 + u,
        dy: iy as f64 + v,
    })
}

/// Best integer shift (in the level's pixels) for the reference patch at
/// `centre`, searched around `base`.
fn search(
    ours: &Gray,
    refs: &Gray,
    centre: (isize, isize),
    half: isize,
    base: (isize, isize),
    range: isize,
) -> Option<(f64, isize, isize)> {
    let mut best: Option<(f64, isize, isize)> = None;
    for dy in -range..=range {
        for dx in -range..=range {
            let shift = (base.0 + dx, base.1 + dy);
            let Some(score) = ncc(ours, refs, centre, shift, half) else {
                continue;
            };
            let better = match best {
                None => true,
                Some((best_score, best_x, best_y)) => {
                    score > best_score + 1e-6
                        || ((score - best_score).abs() <= 1e-6
                            && shift.0.abs() + shift.1.abs() < best_x.abs() + best_y.abs())
                }
            };
            if better {
                best = Some((score, shift.0, shift.1));
            }
        }
    }
    best
}

/// Zero-mean normalised cross correlation of the reference patch at `centre`
/// with the merged frame displaced by `shift`.
fn ncc(
    ours: &Gray,
    refs: &Gray,
    centre: (isize, isize),
    shift: (isize, isize),
    half: isize,
) -> Option<f64> {
    let (rx, ry) = centre;
    let (ox, oy) = (rx + shift.0, ry + shift.1);
    if rx - half < 0
        || ry - half < 0
        || rx + half >= refs.w as isize
        || ry + half >= refs.h as isize
        || ox - half < 0
        || oy - half < 0
        || ox + half >= ours.w as isize
        || oy + half >= ours.h as isize
    {
        return None;
    }
    let width = (2 * half + 1) as usize;
    let count = (width * width) as f64;
    let mut sum_r = 0.0f64;
    let mut sum_o = 0.0f64;
    for dy in 0..width {
        let start_r = (ry - half + dy as isize) as usize * refs.w + (rx - half) as usize;
        let start_o = (oy - half + dy as isize) as usize * ours.w + (ox - half) as usize;
        for dx in 0..width {
            sum_r += refs.lum[start_r + dx] as f64;
            sum_o += ours.lum[start_o + dx] as f64;
        }
    }
    let mean_r = sum_r / count;
    let mean_o = sum_o / count;
    let mut num = 0.0f64;
    let mut var_r = 0.0f64;
    let mut var_o = 0.0f64;
    for dy in 0..width {
        let start_r = (ry - half + dy as isize) as usize * refs.w + (rx - half) as usize;
        let start_o = (oy - half + dy as isize) as usize * ours.w + (ox - half) as usize;
        for dx in 0..width {
            let a = refs.lum[start_r + dx] as f64 - mean_r;
            let b = ours.lum[start_o + dx] as f64 - mean_o;
            num += a * b;
            var_r += a * a;
            var_o += b * b;
        }
    }
    let denom = (var_r * var_o).sqrt();
    if denom <= 1e-12 {
        None
    } else {
        Some(num / denom)
    }
}

/// Gauss-Newton refinement of an integer shift to a fraction of a pixel.
///
/// Six parameters: the translation of the block centre plus the local linear
/// term of the radial map. A translation-only fit absorbs the linear term but
/// keeps the curvature across the template as a bias, which on a 4 % lens is
/// several hundredths of a pixel and, through the ill-conditioned third radial
/// term, percent level errors in k2 and k3.
///
/// The intensity relation between the two images is a per-block gain and
/// offset, refitted every iteration: the reference is a rendered sRGB export
/// while the merged frame is linear, so a plain SSD would be biased by the tone
/// curve. Returns the offset relative to `shift`.
fn refine(
    ours: &Gray,
    refs: &Gray,
    centre: (isize, isize),
    shift: (isize, isize),
    half: isize,
) -> Option<(f64, f64)> {
    let (rx, ry) = centre;
    if rx - half < 2
        || ry - half < 2
        || rx + half >= refs.w as isize - 2
        || ry + half >= refs.h as isize - 2
    {
        return None;
    }
    // p = [u, v, dX/dx, dX/dy, dY/dx, dY/dy] around the shifted block centre.
    let mut p = [shift.0 as f64, shift.1 as f64, 0.0, 0.0, 0.0, 0.0];
    let count = ((2 * half + 1) * (2 * half + 1)) as f64;
    let position = |dx: isize, dy: isize, p: &[f64; 6]| {
        (
            (rx + dx) as f64 + p[0] + p[2] * dx as f64 + p[3] * dy as f64,
            (ry + dy) as f64 + p[1] + p[4] * dx as f64 + p[5] * dy as f64,
        )
    };

    for _ in 0..10 {
        let mut sum_r = 0.0f64;
        let mut sum_s = 0.0f64;
        for dy in -half..=half {
            for dx in -half..=half {
                sum_r += refs.at(rx + dx, ry + dy) as f64;
                let (x, y) = position(dx, dy, &p);
                sum_s += ours.sample_with_grad(x, y).0;
            }
        }
        let mean_r = sum_r / count;
        let mean_s = sum_s / count;
        let mut cov = 0.0f64;
        let mut var = 0.0f64;
        for dy in -half..=half {
            for dx in -half..=half {
                let a = refs.at(rx + dx, ry + dy) as f64 - mean_r;
                let (x, y) = position(dx, dy, &p);
                let b = ours.sample_with_grad(x, y).0 - mean_s;
                cov += a * b;
                var += a * a;
            }
        }
        if var <= 1e-12 {
            return None;
        }
        let gain = cov / var;
        let offset = mean_s - gain * mean_r;

        let mut h = [[0.0f64; 6]; 6];
        let mut g = [0.0f64; 6];
        for dy in -half..=half {
            for dx in -half..=half {
                let r = refs.at(rx + dx, ry + dy) as f64;
                let (x, y) = position(dx, dy, &p);
                let (s, sx, sy) = ours.sample_with_grad(x, y);
                let e = s - (gain * r + offset);
                let j = [
                    sx,
                    sy,
                    sx * dx as f64,
                    sx * dy as f64,
                    sy * dx as f64,
                    sy * dy as f64,
                ];
                for i in 0..6 {
                    g[i] += e * j[i];
                    for k in 0..6 {
                        h[i][k] += j[i] * j[k];
                    }
                }
            }
        }
        // A tiny ridge keeps the linear terms from running away where the
        // template is one dimensional (a straight edge).
        for i in 0..6 {
            h[i][i] *= 1.0 + 1e-6;
        }
        let Some(step) = solve6(h, g) else {
            break;
        };
        let mut moved = 0.0f64;
        for i in 0..6 {
            let delta = (-step[i]).clamp(-1.0, 1.0);
            p[i] += delta;
            moved = moved.max(delta.abs());
        }
        if moved < 1e-4 {
            break;
        }
    }
    Some((p[0] - shift.0 as f64, p[1] - shift.1 as f64))
}

/// Gaussian elimination with partial pivoting, used by the per-block
/// refinement.
fn solve6(mut a: [[f64; 6]; 6], mut b: [f64; 6]) -> Option<[f64; 6]> {
    for col in 0..6 {
        let mut pivot = col;
        for row in col + 1..6 {
            if a[row][col].abs() > a[pivot][col].abs() {
                pivot = row;
            }
        }
        if a[pivot][col].abs() <= 1e-18 {
            return None;
        }
        if pivot != col {
            a.swap(pivot, col);
            b.swap(pivot, col);
        }
        for row in col + 1..6 {
            let factor = a[row][col] / a[col][col];
            for k in col..6 {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = [0.0f64; 6];
    for row in (0..6).rev() {
        let mut sum = b[row];
        for k in row + 1..6 {
            sum -= a[row][k] * x[k];
        }
        x[row] = sum / a[row][row];
    }
    if x.iter().any(|value| !value.is_finite()) {
        return None;
    }
    Some(x)
}

/// One least squares pass, then a second pass without the blocks whose
/// displacement the model could not explain (a mismatched block is an outlier,
/// not a lens property).
fn solve_model(samples: &[Sample], w: usize, h: usize) -> Result<FitReport> {
    let (first, errors) = fit_once(samples, w, h)?;
    // Median based, not RMS based: a handful of grossly mismatched blocks would
    // inflate an RMS threshold until nothing is rejected at all.
    let mut sorted: Vec<f64> = errors.iter().map(|error| error.abs()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted.get(sorted.len() / 2).copied().unwrap_or(0.0);
    let threshold = (3.0 * median).max(0.10);
    let kept: Vec<Sample> = samples
        .iter()
        .zip(&errors)
        .filter(|(_, error)| error.abs() <= threshold)
        .map(|(sample, _)| *sample)
        .collect();
    if kept.len() >= 10 && kept.len() < samples.len() {
        let (second, _) = fit_once(&kept, w, h)?;
        return Ok(second);
    }
    Ok(first)
}

/// Least squares fit of shift x, shift y, k1, k2, k3 over the block samples.
fn fit_once(samples: &[Sample], w: usize, h: usize) -> Result<(FitReport, Vec<f64>)> {
    if samples.len() < 5 {
        bail!("only {} blocks matched", samples.len());
    }
    let cx = (w as f64 - 1.0) * 0.5;
    let cy = (h as f64 - 1.0) * 0.5;
    let diag2 = (cx * cx + cy * cy).max(1e-12);

    let mut ata = [[0.0f64; 5]; 5];
    let mut atb = [0.0f64; 5];
    for sample in samples {
        let dx = sample.x - cx;
        let dy = sample.y - cy;
        let r2 = (dx * dx + dy * dy) / diag2;
        let row_x = [1.0, 0.0, dx * r2, dx * r2 * r2, dx * r2 * r2 * r2];
        let row_y = [0.0, 1.0, dy * r2, dy * r2 * r2, dy * r2 * r2 * r2];
        accumulate(&mut ata, &mut atb, row_x, sample.dx);
        accumulate(&mut ata, &mut atb, row_y, sample.dy);
    }
    let Some(solution) = solve5(ata, atb) else {
        bail!("the block displacements do not determine a distortion model: the reference is too uniform");
    };
    let model = Distortion::new3(solution[2], solution[3], solution[4]);

    let mut sum = 0.0f64;
    let mut max_radius: f64 = 0.0;
    let mut errors = Vec::with_capacity(samples.len());
    for sample in samples {
        let dx = sample.x - cx;
        let dy = sample.y - cy;
        let r2 = (dx * dx + dy * dy) / diag2;
        let f = model.k1 * r2 + model.k2 * r2 * r2 + model.k3 * r2 * r2 * r2;
        let ex = sample.dx - (solution[0] + dx * f);
        let ey = sample.dy - (solution[1] + dy * f);
        sum += ex * ex + ey * ey;
        errors.push((ex * ex + ey * ey).sqrt());
        max_radius = max_radius.max((dx * dx + dy * dy).sqrt());
    }
    let residual_rms = (sum / (2.0 * samples.len() as f64)).sqrt();

    Ok((
        FitReport {
            k1: solution[2],
            k2: solution[3],
            k3: solution[4],
            shift: (solution[0], solution[1]),
            residual_rms,
            blocks: samples.len(),
            max_radius_px: max_radius,
        },
        errors,
    ))
}

fn accumulate(ata: &mut [[f64; 5]; 5], atb: &mut [f64; 5], row: [f64; 5], value: f64) {
    for i in 0..5 {
        atb[i] += row[i] * value;
        for j in 0..5 {
            ata[i][j] += row[i] * row[j];
        }
    }
}

/// Gaussian elimination with partial pivoting on the 5x5 normal equations.
fn solve5(mut a: [[f64; 5]; 5], mut b: [f64; 5]) -> Option<[f64; 5]> {
    for col in 0..5 {
        let mut pivot = col;
        for row in col + 1..5 {
            if a[row][col].abs() > a[pivot][col].abs() {
                pivot = row;
            }
        }
        if a[pivot][col].abs() <= 1e-18 {
            return None;
        }
        if pivot != col {
            a.swap(pivot, col);
            b.swap(pivot, col);
        }
        for row in col + 1..5 {
            let factor = a[row][col] / a[col][col];
            for k in col..5 {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = [0.0f64; 5];
    for row in (0..5).rev() {
        let mut sum = b[row];
        for k in row + 1..5 {
            sum -= a[row][k] * x[k];
        }
        x[row] = sum / a[row][row];
    }
    if x.iter().any(|value| !value.is_finite()) {
        return None;
    }
    Some(x)
}
