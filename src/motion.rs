//! Motion compensation for Pixel Shift merging: detect where a frame
//! disagrees with the reference after the rigid alignment and mask those
//! regions out of the merge.
//!
//! The detector works on the CFA phase invariant lattice of [`crate::shift`]
//! (2x2 block sums, low passed), so the comparison never depends on the Bayer
//! phase of the two frames. Every non-reference frame is resampled through
//! its rigid [`Shift`] onto the reference lattice, the difference is
//! accumulated per block and compared against a locally median block energy:
//! motion occupies a minority of blocks, so the median estimates the
//! noise + interpolation floor of the neighbourhood, and a block whose energy
//! exceeds a multiple of that floor is moving. Masks are cleaned of isolated
//! noise spikes and dilated by one block.
//!
//! The reference frame (index 0) is never masked: inside heavy motion the
//! merged result then falls back to its own samples - a sharp, ghost free,
//! single frame view of the moving region.

use crate::shift::{luma_plane, sample_taps, taps, Plane};
use crate::types::{MotionMask, MotionParams, RawFrame, Rect, Shift};
use rayon::prelude::*;

/// Block edge in lattice pixels (2 sensor px per lattice step): 32 sensor
/// pixels. Small enough to outline a distant person, coarse enough that one
/// block holds hundreds of difference samples for a stable energy estimate.
pub const BLOCK_LATTICE: usize = 16;
/// A block counts as moving when its difference energy exceeds this multiple
/// of the local median block energy. The median tracks the noise +
/// interpolation floor of the neighbourhood; the factor leaves headroom for
/// the heavy tail of the difference of two noisy blocks.
pub const THRESHOLD: f32 = 6.0;
/// Half window (in blocks) of the local median floor.
const MEDIAN_RADIUS: usize = 3;
/// Difference samples are evaluated on this sub-grid of the lattice: a block
/// still holds (BLOCK_LATTICE / SAMPLE_STRIDE)^2 = 64 samples and the pass
/// runs four times faster.
const SAMPLE_STRIDE: usize = 2;
/// Blocks with fewer masked 8-neighbours than this are dropped as noise
/// spikes (before dilation).
const MIN_NEIGHBOURS: usize = 1;
/// Safety factor on the alignment residual when converting it into an
/// expected static disagreement: the residual is a fit RMS over windows, the
/// local error at the frame corners is larger.
const RESIDUAL_MARGIN: f64 = 2.0;

/// Absolute floor of the local median energy, so that a perfectly flat
/// neighbourhood (median ~ 0) cannot flag every nonzero block above it. In
/// block-sum ADU^2: noise of 4 ADU per lattice sample difference.
const ENERGY_FLOOR: f32 = 16.0;

/// Detect per frame motion masks for a series. `shifts` must be aligned with
/// `frames` (shifts[0] is the identity). The reference frame gets `None`;
/// frames whose mask stays empty get `None` too, so the merge skips the mask
/// lookup for them.
pub fn masks(
    frames: &[RawFrame],
    shifts: &[Shift],
    params: &MotionParams,
) -> Vec<Option<MotionMask>> {
    let n = frames.len().min(shifts.len());
    let Some(reference) = frames.first() else {
        return Vec::new();
    };
    let (lw, lh) = (reference.width / 2, reference.height / 2);
    if n < 2 || lw < 2 * BLOCK_LATTICE || lh < 2 * BLOCK_LATTICE {
        return vec![None; n];
    }
    let reference_plane = luma_plane(reference, Rect::new(0, 0, lw, lh));
    let threshold = THRESHOLD / params.sensitivity.max(0.05);
    // Temporal pair distances: a quarter of the series and the full span.
    let mut distances: Vec<usize> = [n / 4, n - 1]
        .into_iter()
        .filter(|&k| k >= 2 && k < n)
        .collect();
    distances.sort_unstable();
    distances.dedup();

    (0..n)
        .into_par_iter()
        .map(|i| {
            if i == 0 {
                return None;
            }
            let pairs: Vec<(&RawFrame, &Shift)> = distances
                .iter()
                .filter_map(|&k| {
                    let j = i + k;
                    if j < n {
                        Some((&frames[j], &shifts[j]))
                    } else {
                        None
                    }
                })
                .collect();
            detect_one(&reference_plane, &frames[i], &shifts[i], threshold, &pairs)
        })
        .collect()
}

/// Detect the mask of one frame against the reference lattice.
fn detect_one(
    reference: &Plane,
    frame: &RawFrame,
    shift: &Shift,
    threshold: f32,
    pairs: &[(&RawFrame, &Shift)],
) -> Option<MotionMask> {
    let (lw, lh) = (reference.w, reference.h);
    let frame_plane = luma_plane(frame, Rect::new(0, 0, frame.width / 2, frame.height / 2));
    // Block sums are linear only below saturation: a block holding clipped
    // samples disagrees between frames in a way that says nothing about
    // motion, and it does so with energies that dwarf any threshold. The
    // outer 2 % of the block-sum range is treated as no data.
    let sat_limit = 0.98 * 4.0 * frame.white_span().max(1) as f32;

    // Lattice-to-lattice affine map. A lattice sample sits at the centre of
    // its 2x2 sensor block, i.e. at sensor (2x + 1, 2y + 1); composing that
    // with Shift::inverse_map and halving gives
    //   fx = cos * x + sin * y + c0
    //   fy = -sin * x + cos * y + c1
    // with cos/sin of the frame rotation and c0/c1 from dx, dy and centre.
    let (sin, cos) = shift.rot.sin_cos();
    let (cx, cy) = (frame.width as f64 * 0.5, frame.height as f64 * 0.5);
    let c0 = 0.5 * (cx + cos * (1.0 - cx - shift.dx) + sin * (1.0 - cy - shift.dy) - 1.0);
    let c1 = 0.5 * (cy - sin * (1.0 - cx - shift.dx) + cos * (1.0 - cy - shift.dy) - 1.0);

    let cols = lw.div_ceil(BLOCK_LATTICE);
    let rows = lh.div_ceil(BLOCK_LATTICE);
    let mut energies = vec![0.0f32; cols * rows];
    let mut grad_rms = vec![0.0f32; cols * rows];
    // Registration error the rigid model leaves on this frame: a static but
    // busy block disagrees by about |gradient| * residual, which the pure
    // noise floor would flag as motion.
    let residual = (shift.residual.max(0.0) as f64 * RESIDUAL_MARGIN) as f32;

    for by in 0..rows {
        let y0 = by * BLOCK_LATTICE;
        let y1 = (y0 + BLOCK_LATTICE).min(lh);
        for bx in 0..cols {
            let x0 = bx * BLOCK_LATTICE;
            let x1 = (x0 + BLOCK_LATTICE).min(lw);
            let mut acc = 0.0f64;
            let mut grad2 = 0.0f64;
            let mut count = 0usize;
            let mut y = y0;
            while y < y1 {
                let row_base_fx = sin * y as f64 + c0;
                let row_base_fy = cos * y as f64 + c1;
                let ref_row = y * lw;
                let mut x = x0;
                while x < x1 {
                    let fx = row_base_fx + cos * x as f64;
                    let fy = row_base_fy - sin * x as f64;
                    // The six-tap kernel reads base-2..base+3, so samples
                    // closer than 3 lattice px to the frame border would
                    // compare clamped edge content; skipping them turns the
                    // border into "no data" instead of fake motion (the
                    // merge already handles the geometric border).
                    let in_frame = fx >= 3.0
                        && fx <= frame_plane.w as f64 - 4.0
                        && fy >= 3.0
                        && fy <= frame_plane.h as f64 - 4.0;
                    if in_frame {
                        // Lanczos-3: the scene holds content near the lattice
                        // Nyquist frequency, where a bilinear resample of a
                        // sub-pixel shifted frame leaves errors far above the
                        // noise. Same argument as the alignment sampler.
                        let tx = taps(fx.fract());
                        let ty = taps(fy.fract());
                        let v = sample_taps(&frame_plane, fx.floor() as i64, fy.floor() as i64, &tx, &ty);
                        let r = reference.data[ref_row + x];
                        if x > 0 && x + 1 < lw && y > 0 && y + 1 < lh {
                            let gx = 0.5
                                * (reference.data[ref_row + x + 1] - reference.data[ref_row + x - 1])
                                as f64;
                            let gy = 0.5
                                * (reference.data[(y + 1) * lw + x] - reference.data[(y - 1) * lw + x])
                                as f64;
                            grad2 += gx * gx + gy * gy;
                        }
                        if r < sat_limit && v < sat_limit {
                            let diff = v - r;
                            acc += diff as f64 * diff as f64;
                        }
                        count += 1;
                    }
                    x += SAMPLE_STRIDE;
                }
                y += SAMPLE_STRIDE;
            }
            // Blocks that lost more than half of their nominal samples to
            // the frame border are border artefacts, not motion: their
            // shorter average has a much heavier noise tail.
            let nominal = (BLOCK_LATTICE / SAMPLE_STRIDE) * (BLOCK_LATTICE / SAMPLE_STRIDE);
            if count * 2 >= nominal {
                energies[by * cols + bx] = (acc / count.max(1) as f64) as f32;
                // Expected static disagreement of a busy block under the
                // residual registration error: |gradient| * residual.
                grad_rms[by * cols + bx] =
                    ((grad2 / count.max(1) as f64).sqrt() * residual as f64) as f32;
            }
        }
    }

    // Fold in the temporal pair energies (same block grid): slow motion
    // displaces less than a block per frame and hides in the reference
    // difference, but K captures apart it has crossed a block.
    for (other, other_shift) in pairs {
        let other_plane = luma_plane(other, Rect::new(0, 0, other.width / 2, other.height / 2));
        let (ca, sa, a0, a1) = lattice_affine(shift, frame.width, frame.height);
        let (cb, sb, b0, b1) = lattice_affine(other_shift, other.width, other.height);
        // Composed affine lattice_i -> lattice_j = A_j * A_i^-1.
        let m00 = cb * ca + sb * sa;
        let m01 = cb * sa - sb * ca;
        let m10 = -sb * ca + cb * sa;
        let m11 = -sb * sa - cb * ca;
        let o0 = b0 - (m00 * a0 + m01 * a1);
        let o1 = b1 - (m10 * a0 + m11 * a1);
        for by in 0..rows {
            let y0 = by * BLOCK_LATTICE;
            let y1 = (y0 + BLOCK_LATTICE).min(lh);
            for bx in 0..cols {
                let x0 = bx * BLOCK_LATTICE;
                let x1 = (x0 + BLOCK_LATTICE).min(lw);
                let mut acc = 0.0f64;
                let mut count = 0usize;
                let mut y = y0;
                while y < y1 {
                    let row = y * lw;
                    let mut x = x0;
                    while x < x1 {
                        let fx = m00 * x as f64 + m01 * y as f64 + o0;
                        let fy = m10 * x as f64 + m11 * y as f64 + o1;
                        if fx >= 3.0
                            && fx <= other_plane.w as f64 - 4.0
                            && fy >= 3.0
                            && fy <= other_plane.h as f64 - 4.0
                        {
                            let tx = taps(fx.fract());
                            let ty = taps(fy.fract());
                            let v = sample_taps(
                                &other_plane,
                                fx.floor() as i64,
                                fy.floor() as i64,
                                &tx,
                                &ty,
                            );
                            let r = frame_plane.data[row + x];
                            if r < sat_limit && v < sat_limit {
                                let diff = v - r;
                                acc += diff as f64 * diff as f64;
                            }
                            count += 1;
                        }
                        x += SAMPLE_STRIDE;
                    }
                    y += SAMPLE_STRIDE;
                }
                let nominal =
                    (BLOCK_LATTICE / SAMPLE_STRIDE) * (BLOCK_LATTICE / SAMPLE_STRIDE);
                if count * 2 >= nominal {
                    let e = (acc / count.max(1) as f64) as f32;
                    let i = by * cols + bx;
                    if e > energies[i] {
                        energies[i] = e;
                        // The static gradient expectation no longer applies
                        // where the pair says the block really moved.
                        grad_rms[i] = 0.0;
                    }
                }
            }
        }
    }

    // Local median floor: motion is a minority of blocks, so the median of
    // the neighbourhood estimates what pure noise + interpolation error
    // looks like there.
    let mut flagged = vec![false; cols * rows];
    let mut window = Vec::with_capacity((2 * MEDIAN_RADIUS + 1).pow(2));
    for by in 0..rows {
        for bx in 0..cols {
            window.clear();
            let y0 = by.saturating_sub(MEDIAN_RADIUS);
            let y1 = (by + MEDIAN_RADIUS).min(rows - 1);
            let x0 = bx.saturating_sub(MEDIAN_RADIUS);
            let x1 = (bx + MEDIAN_RADIUS).min(cols - 1);
            for wy in y0..=y1 {
                for wx in x0..=x1 {
                    window.push(energies[wy * cols + wx]);
                }
            }
            window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = window[window.len() / 2];
            let misregistration = grad_rms[by * cols + bx].powi(2);
            let floor = median.max(misregistration).max(ENERGY_FLOOR);
            flagged[by * cols + bx] = energies[by * cols + bx] > threshold * floor;
        }
    }

    // Drop isolated noise spikes, then dilate by one block so the mask
    // covers the motion boundary too.
    let keep = drop_isolated(&flagged, cols, rows);
    let dilated = dilate(&keep, cols, rows);

    let mask = MotionMask {
        step: 2 * BLOCK_LATTICE,
        cols,
        rows,
        blocked: dilated.into_iter().map(|v| v as u8).collect(),
    };
    if mask.is_empty() {
        None
    } else {
        if std::env::var_os("PIXELSHIFT_DEBUG_MOTION").is_some() {
            let max_energy = energies.iter().cloned().fold(0.0f32, f32::max);
            let mut sorted = energies.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            eprintln!(
                "motion: frame mask {:.2} % (max block energy {:.0}, median {:.1}, p99 {:.0})",
                mask.masked_fraction() * 100.0,
                max_energy,
                sorted[sorted.len() / 2],
                sorted[sorted.len() * 99 / 100]
            );
        }
        Some(mask)
    }
}

/// Zero every flagged cell that has fewer than MIN_NEIGHBOURS flagged
/// 8-neighbours.
fn drop_isolated(mask: &[bool], cols: usize, rows: usize) -> Vec<bool> {
    let neighbours = |x: usize, y: usize| -> usize {
        let mut n = 0;
        for dy in -1i64..=1 {
            for dx in -1i64..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x as i64 + dx;
                let ny = y as i64 + dy;
                if nx >= 0 && ny >= 0 && (nx as usize) < cols && (ny as usize) < rows
                    && mask[ny as usize * cols + nx as usize]
                {
                    n += 1;
                }
            }
        }
        n
    };
    (0..cols * rows)
        .map(|i| {
            let (x, y) = (i % cols, i / cols);
            mask[i] && neighbours(x, y) >= MIN_NEIGHBOURS
        })
        .collect()
}

/// Grow the mask by one cell in Chebyshev distance.
fn dilate(mask: &[bool], cols: usize, rows: usize) -> Vec<bool> {
    (0..cols * rows)
        .map(|i| {
            let (x, y) = (i % cols, i / cols);
            if mask[i] {
                return true;
            }
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let nx = x as i64 + dx;
                    let ny = y as i64 + dy;
                    if nx >= 0 && ny >= 0 && (nx as usize) < cols && (ny as usize) < rows
                        && mask[ny as usize * cols + nx as usize]
                    {
                        return true;
                    }
                }
            }
            false
        })
        .collect()
}

/// Lattice-to-lattice affine constants of a rigid shift, mapping a reference
/// lattice point to the frame lattice (the same constants detect_one builds
/// inline): fx = cos*x + sin*y + c0, fy = -sin*x + cos*y + c1.
fn lattice_affine(shift: &Shift, width: usize, height: usize) -> (f64, f64, f64, f64) {
    let (sin, cos) = shift.rot.sin_cos();
    let (cx, cy) = (width as f64 * 0.5, height as f64 * 0.5);
    let c0 = 0.5 * (cx + cos * (1.0 - cx - shift.dx) + sin * (1.0 - cy - shift.dy) - 1.0);
    let c1 = 0.5 * (cy - sin * (1.0 - cx - shift.dx) + cos * (1.0 - cy - shift.dy) - 1.0);
    (cos, sin, c0, c1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Cfa;

    const W: usize = 640;
    const H: usize = 424;

    /// Deterministic noisy fill of a frame with a moving bright square that
    /// travels between frames; everything else is a static gradient.
    fn frame_with(
        seed: u64,
        shift: (f64, f64),
        square: Option<(usize, usize)>,
        sigma: f32,
    ) -> RawFrame {
        let cfa = Cfa::from_pattern_str("RGGB").expect("RGGB");
        let black = 1008u16;
        let white = 16383u16;
        let span = (white - black) as f32;
        let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        let mut rnd = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 33) as f32 / u32::MAX as f32
        };
        let mut data = vec![0u16; W * H];
        for y in 0..H {
            for x in 0..W {
                let sx = x as f64 - shift.0;
                let sy = y as f64 - shift.1;
                let mut value =
                    0.25f32 + 0.3 * (sx / W as f64) as f32 + 0.15 * (sy / H as f64) as f32;
                if let Some((ox, oy)) = square {
                    let dx = (sx as i64 - ox as i64).abs();
                    let dy = (sy as i64 - oy as i64).abs();
                    if dx < 40 && dy < 40 {
                        value = 0.85;
                    }
                }
                let noise = (rnd() * 2.0 - 1.0) * sigma;
                data[y * W + x] =
                    (black as f32 + (value * span + noise).clamp(0.0, span)) as u16;
            }
        }
        RawFrame {
            path: std::path::PathBuf::new(),
            width: W,
            height: H,
            data,
            cfa,
            black,
            cblack: [0; 4],
            white,
            wb: [1.0, 1.0, 1.0, 1.0],
            cam_xyz: [[0f32; 3]; 4],
            rgb_cam: [[0f32; 3]; 4],
            iso: 100.0,
            shutter: 1.0 / 125.0,
            aperture: 5.6,
            focal: 50.0,
            lens_make: String::new(),
            lens_model: String::new(),
            focal_35mm: 50,
            max_aperture: 1.8,
            lens_min_focal: 50.0,
            lens_max_focal: 50.0,
            lens_min_fnum: 1.8,
            lens_max_fnum: 1.8,
            lens_serial: String::new(),
            body_serial: String::new(),
            make: String::new(),
            model: String::new(),
            timestamp: String::new(),
            index: 0,
        }
    }

    #[test]
    fn static_frames_get_no_mask() {
        let reference = frame_with(1, (0.0, 0.0), None, 6.0);
        let frames = vec![reference.clone(), frame_with(2, (0.5, 0.0), None, 6.0)];
        let shifts = vec![Shift::default(), Shift::new(0.5, 0.0)];
        let out = masks(&frames, &shifts, &MotionParams::default());
        assert!(out[0].is_none(), "reference is never masked");
        assert!(out[1].is_none(), "a static frame must not be masked");
    }

    #[test]
    fn moving_square_is_masked() {
        let reference = frame_with(1, (0.0, 0.0), Some((100, 100)), 6.0);
        // The square moved far between the frames.
        let frames = vec![reference, frame_with(2, (0.0, 0.0), Some((180, 120)), 6.0)];
        let shifts = vec![Shift::default(), Shift::default()];
        let out = masks(&frames, &shifts, &MotionParams::default());
        let mask = out[1].clone().expect("the moving square must produce a mask");
        let fraction = mask.masked_fraction();
        assert!(fraction > 0.001, "nothing was masked");
        assert!(fraction < 0.20, "mask covers the whole frame: {fraction}");
        // Both the old and the new square position must be flagged (the
        // reference disagrees with the frame in both places).
        assert!(mask.blocked_at(100, 100), "old square position masked");
        assert!(mask.blocked_at(180, 120), "new square position masked");
        assert!(!mask.blocked_at(500, 300), "static background stays clean");
    }

    #[test]
    fn subnoise_motion_is_not_detected() {
        // A contrast far below the per-sample noise (0.02 % of the span) is
        // invisible in the block averages and must not raise a mask.
        let frames = vec![
            frame_with(1, (0.0, 0.0), None, 6.0),
            frame_with_faint(2, (180, 120)),
        ];
        let shifts = vec![Shift::default(), Shift::default()];
        let out = masks(&frames, &shifts, &MotionParams::default());
        assert!(out[1].is_none(), "sub-noise motion produced a mask");
    }

    /// Like [`frame_with`] with a dithered +1 ADU square at (ox, oy): the
    /// lattice averages see a +0.5 ADU per sensor pixel step, below the
    /// noise floor of the blurred block sums.
    fn frame_with_faint(seed: u64, pos: (usize, usize)) -> RawFrame {
        let mut frame = frame_with(seed, (0.0, 0.0), None, 6.0);
        for y in 0..frame.height {
            for x in 0..frame.width {
                let dx = (x as i64 - pos.0 as i64).abs();
                let dy = (y as i64 - pos.1 as i64).abs();
                if dx < 40 && dy < 40 && (x + y) % 2 == 0 {
                    let i = y * frame.width + x;
                    let v = frame.data[i].saturating_add(1);
                    frame.data[i] = v;
                }
            }
        }
        frame
    }
}
