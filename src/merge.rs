//! Reconstruction of the high resolution linear image from aligned frames.
//!
//! Owned by work package WP2 (alignment + merge core).
//!
//! Normalised per-colour splatting (docs/ARCHITECTURE.md): frame px (x, y) of
//! colour c is deposited at scale * (x + dx), scale * (y + dy) of the output
//! grid with bilinear weights into a value accumulator and a weight
//! accumulator; out[c] = sum[c] / weight[c]. Ideal half pixel shifts therefore
//! give a delta kernel (no blur) and repeated positions of the 8/32 frame modes
//! average automatically.
//!
//! Accumulators are u32 fixed point sums (1/16 sample unit) plus f32 weights.
//! A weight quirk check: one sample contributes 1.0 of weight at most, the
//! worst realistic case is 32 frames x 1 sample x 15375 x 16 = 7.9e6, four
//! orders of magnitude below the u32 limit, and such sums are exact in f32
//! (24 bit mantissa), so the u16 output is limited by rounding only. u32 sums
//! with f32 weights cost 8 bytes per pixel and channel, the same as f32 pairs,
//! and never overflow.
//!
//! Coverage is reported relative to the best coverage achieved in the same
//! channel, so min_coverage means "a fraction of a fully covered pixel" and does
//! not depend on the frame count or on the scale.

use crate::types::{
    Cfa, Color, MergeParams, MergeResult, MergeStats, MotionMask, PreviewImage, RawFrame, Shift,
    ShiftGrid, CHANNELS,
};
use anyhow::{bail, Result};
use rayon::prelude::*;
use std::time::Instant;

/// Fixed point factor of the i32 sample accumulators (1/16 sample unit).
const FIXED: f32 = 16.0;
/// Reconstruction kernel half width: the sample is scattered to 4x4 output
/// pixels with Catmull-Rom weights. A bilinear splat is the adjoint of bilinear
/// interpolation and costs about half a stop of contrast at the 2x grid
/// frequency; the cubic kernel keeps the ideal lattice exact (a delta) and
/// leaves the fractional positions sharper, at the price of a small ringing.
const RECON_RADIUS: i64 = 1;
/// Per channel weight below which a pixel carries no usable measurement.
const HOLE_WEIGHT: f32 = 0.05;
/// Output rows accumulated by one rayon task.
const BAND_ROWS: usize = 128;
/// Weight resolution stored per output pixel and channel (1/8 sample unit).
const COV_SCALE: f32 = 8.0;
const MAX_FILL_ITERS: usize = 64;
/// Angular frequency (radians per output pixel) at which the inverse is
/// calibrated: half of the output Nyquist frequency, where both the splat and
/// the aperture still have real signal to restore.
const DECONV_OMEGA: f64 = std::f64::consts::FRAC_PI_2;
/// Gain limit of the reconstruction-kernel inverse at the output Nyquist
/// frequency (the filter reaches 1 + 4k there, so 0.25 caps it at 2.0). Beyond
/// that the sensor aperture has already zeroed the signal near the pixel pitch
/// and only noise would be amplified.
const DECONV_K_MAX: f64 = 0.25;
/// Correction amplitudes up to this many noise sigma count as grain and are
/// shrunk: the deconvolution must not amplify the noise of flat areas.
const DECONV_NOISE_GATE: f32 = 2.5;
/// Lower quantile of the high pass that measures its noise level: unlike the
/// mean or the median it is not inflated by the textured part of the frame.
const DECONV_NOISE_PCT: f64 = 0.10;
/// Quantile of |N(0, 1)| at DECONV_NOISE_PCT, i.e. p10 = RAYLEIGH_P10 * sigma.
const RAYLEIGH_P10: f32 = 0.459;
/// Overshoot the post filter may add on top of the local 3x3 contrast: 2 % of
/// it, so a full scale step is reproduced without a visible halo.
const DECONV_CLAMP_FRAC: f32 = 0.02;
/// Absolute floor of that allowance, relative to the white span (0.25 %): it
/// lets fine low contrast texture keep its sharpening.
const DECONV_CLAMP_ABS: f32 = 0.0025;

/// One frame reduced to what the splatting loop needs.
struct FrameView<'a> {
    data: &'a [u16],
    w: usize,
    h: usize,
    cfa: Cfa,
    /// Output grid coordinates of a sample are
    /// X = ox + ax * (x - cx) + bx * (y - cy), Y = oy + ay * (x - cx) + by * (y - cy):
    /// the rigid map of types::Shift, scaled by the output grid density. The
    /// centre is the frame centre, the same one shift::estimate fits about.
    cx: f64,
    cy: f64,
    ox: f64,
    oy: f64,
    ax: f64,
    bx: f64,
    ay: f64,
    by: f64,
    /// Output pixels of extra row range a band needs because of the rotation.
    rot_margin: f64,
    /// Output grid density, needed to add the local residual grid before scaling.
    scale: f64,
    /// Local correction on top of the rigid model, in sensor pixels.
    grid: Option<ShiftGrid>,
    /// Motion mask: samples in blocked cells are not merged (None for the
    /// reference frame and unmasked frames).
    mask: Option<&'a MotionMask>,
    black: [f32; CHANNELS],
}

struct BandOut {
    planes: [Vec<u16>; CHANNELS],
    cov: [Vec<u8>; CHANNELS],
    weight_sum: [f64; CHANNELS],
}

/// Reconstruct the output image described by params.
/// params.shifts must be aligned with frames (shifts[0] == 0,0).
pub fn merge(frames: &[RawFrame], params: &MergeParams) -> Result<MergeResult> {
    let started = Instant::now();
    let Some(first) = frames.first() else {
        bail!("merge: empty frame list");
    };
    if let Err(e) = first_pixel_guard(frames) {
        return Err(e);
    }
    let k = params.scale.max(1) as usize;
    let (sw, sh) = (first.width, first.height);
    let (ow, oh) = (sw * k, sh * k);
    let span = first.white_span();
    let span_f = span as f32;

    let shifts: Vec<Shift> = applied_shifts(frames, params)?;
    let used = used_frames(params, &shifts);
    if used.is_empty() {
        bail!("merge: no frame left after de-duplication");
    }
    let kf = k as f64;
    let views: Vec<FrameView> = used
        .iter()
        .map(|&i| {
            let f = &frames[i];
            let s = shifts[i];
            let (sin, cos) = s.rot.sin_cos();
            let cx = f.width as f64 * 0.5;
            let cy = f.height as f64 * 0.5;
            FrameView {
                data: &f.data,
                w: f.width,
                h: f.height,
                cfa: f.cfa,
                cx,
                cy,
                ox: kf * (cx + s.dx),
                oy: kf * (cy + s.dy),
                ax: kf * cos,
                bx: -kf * sin,
                ay: kf * sin,
                by: kf * cos,
                rot_margin: kf * s.rot.abs() * f.width.max(f.height) as f64 + 3.0,
                scale: kf,
                grid: if params.align {
                    params.grids.get(i).cloned().flatten()
                } else {
                    None
                },
                // The reference frame is never masked: in fully masked
                // regions the result falls back to its own samples.
                mask: if i == 0 { None } else { params.masks.get(i).and_then(|m| m.as_ref()) },
                black: [
                    f.black_for(Color::R) as f32,
                    f.black_for(Color::G) as f32,
                    f.black_for(Color::B) as f32,
                ],
            }
        })
        .collect();

    // Per input frame masked share, aligned with params.shifts (0 for the
    // reference frame, unused frames included for stable indexing).
    let stats_masked: Vec<f32> = if params.masks.is_empty() {
        Vec::new()
    } else {
        (0..shifts.len())
            .map(|i| {
                params
                    .masks
                    .get(i)
                    .and_then(|m| m.as_ref())
                    .map(|m| m.masked_fraction())
                    .unwrap_or(0.0)
            })
            .collect()
    };

    let band_count = oh.div_ceil(BAND_ROWS);
    let bands: Vec<BandOut> = (0..band_count)
        .into_par_iter()
        .map(|b| {
            let y0 = b * BAND_ROWS;
            let rows = BAND_ROWS.min(oh - y0);
            run_band(&views, ow, oh, y0, rows, span_f)
        })
        .collect();

    let mut planes: [Vec<u16>; CHANNELS] = [
        vec![0; ow * oh],
        vec![0; ow * oh],
        vec![0; oh * ow],
    ];
    let mut cov: [Vec<u8>; CHANNELS] = [vec![0; ow * oh], vec![0; ow * oh], vec![0; oh * ow]];
    let mut weight_sum = [0.0f64; CHANNELS];
    for (b, band) in bands.iter().enumerate() {
        let y0 = b * BAND_ROWS;
        let rows = BAND_ROWS.min(oh - y0);
        for c in 0..CHANNELS {
            let dst = y0 * ow..(y0 + rows) * ow;
            planes[c][dst.clone()].copy_from_slice(&band.planes[c]);
            cov[c][dst].copy_from_slice(&band.cov[c]);
            weight_sum[c] += band.weight_sum[c];
        }
    }
    drop(bands);

    let mut max_cov = [0u8; CHANNELS];
    for (c, mc) in max_cov.iter_mut().enumerate() {
        *mc = cov[c].iter().copied().max().unwrap_or(0);
    }

    // Output crop. Two sources trim the rectangle: electronic stabilization
    // (the sensor-pixel overlap of every frame, scaled to the output grid) and
    // the coverage trim, which drops lines that no column or row can repair -
    // a line is removed when every pixel on it is uncovered in some channel,
    // which is exactly the unreachable rim of the shift lattice (for instance
    // the first row of a half pixel lattice never receives green or blue).
    // The planes are cropped once, after both.
    let (mut x0, mut y0, mut x1, mut y1) = (0usize, 0usize, ow, oh);
    if params.stabilize {
        let (kx0, ky0, kx1, ky1) = stabilized_rect(&shifts, &used, sw, sh);
        x0 = x0.max(kx0 * k);
        y0 = y0.max(ky0 * k);
        x1 = x1.min(kx1 * k);
        y1 = y1.min(ky1 * k);
    }
    if params.crop_uncovered {
        let complete = |i: usize| -> bool {
            (0..CHANNELS).all(|c| cov[c][i] as f32 >= 0.5 * max_cov[c] as f32)
        };
        loop {
            let mut changed = false;
            while y0 + 2 < y1 && (x0..x1).all(|x| !complete(y0 * ow + x)) {
                y0 += 1;
                changed = true;
            }
            while y1 > y0 + 2 && (x0..x1).all(|x| !complete((y1 - 1) * ow + x)) {
                y1 -= 1;
                changed = true;
            }
            while x0 + 2 < x1 && (y0..y1).all(|y| !complete(y * ow + x0)) {
                x0 += 1;
                changed = true;
            }
            while x1 > x0 + 2 && (y0..y1).all(|y| !complete(y * ow + (x1 - 1))) {
                x1 -= 1;
                changed = true;
            }
            if !changed {
                break;
            }
        }
    }
    if (x0, y0, x1, y1) != (0, 0, ow, oh) {
        for c in 0..CHANNELS {
            planes[c] = crop_plane(&planes[c], ow, x0, y0, x1 - x0, y1 - y0);
            cov[c] = crop_bytes(&cov[c], ow, x0, y0, x1 - x0, y1 - y0);
        }
        for (c, mc) in max_cov.iter_mut().enumerate() {
            *mc = cov[c].iter().copied().max().unwrap_or(0);
        }
    }
    let (nw, nh) = (x1 - x0, y1 - y0);
    if (nw, nh) != (ow, oh) {
        // Weight sums refer to the uncropped area; scale down proportionally.
        let frac = (nw * nh) as f64 / (ow * oh) as f64;
        for c in 0..CHANNELS {
            weight_sum[c] *= frac;
        }
    }

    // A pixel is a hole (and is counted in filled_pixels) when its TOTAL weight
    // over the three channels is below min_coverage, which is what the contract
    // asks for. A single channel with almost no weight is normal geometry (the
    // splat deposits one sample into several output pixels) and is rebuilt
    // silently, otherwise real measurements would be replaced by interpolation
    // and the result would be soft.
    let mut needs_rebuild = vec![0u8; nw * nh];
    let mut min_cov = f32::INFINITY;
    for i in 0..nw * nh {
        let total: f32 = (0..CHANNELS).map(|c| cov[c][i] as f32 / COV_SCALE).sum();
        if total < params.min_coverage {
            needs_rebuild[i] = 1;
        }
        for c in 0..CHANNELS {
            if max_cov[c] > 0 {
                min_cov = min_cov.min(cov[c][i] as f32 / COV_SCALE);
            }
        }
    }
    let mut hole_masks: [Vec<u8>; CHANNELS] = [Vec::new(), Vec::new(), Vec::new()];
    for c in 0..CHANNELS {
        let mut mask = vec![0u8; nw * nh];
        if max_cov[c] > 0 {
            for (i, m) in mask.iter_mut().enumerate() {
                let wgt = cov[c][i] as f32 / COV_SCALE;
                if wgt < HOLE_WEIGHT || needs_rebuild[i] != 0 {
                    *m = 1;
                }
            }
        }
        hole_masks[c] = mask;
    }
    drop(cov);
    if !min_cov.is_finite() {
        min_cov = 0.0;
    }

    let filled_pixels = needs_rebuild.iter().filter(|v| **v != 0).count();
    for c in 0..CHANNELS {
        fill_channel(&mut planes[c], &mut hole_masks[c], nw, nh);
    }
    drop(hole_masks);

    let deficit = splat_deficit(&shifts, &used, k as f64);
    let coeff = deconv_coeff(params.deconvolve, k as f64, deficit);
    if std::env::var_os("PIXELSHIFT_DEBUG_DECONV").is_some() {
        eprintln!(
            "deconvolve: amount {:.2}, splat deficit {:.4} (gain {:.4} at w {:.3}), coefficient {:.4}",
            params.deconvolve,
            deficit,
            (-deficit).exp(),
            DECONV_OMEGA,
            coeff
        );
    }
    if coeff > 0.0 {
        let gate = deconv_gate(params, &used);
        if let (Some(g), true) = (&gate, std::env::var_os("PIXELSHIFT_DEBUG_DECONV").is_some()) {
            let mean = g.frac.iter().sum::<f32>() / g.frac.len().max(1) as f32;
            let worst = g.frac.iter().cloned().fold(0.0f32, f32::max);
            eprintln!(
                "deconv gate: mean {:.3}, worst block {:.3}",
                mean, worst
            );
        }
        deconvolve_planes(
            &mut planes,
            nw,
            nh,
            coeff as f32,
            span_f,
            gate.as_ref().map(|g| (g, x0, y0, k)),
        );
    }

    let pixels = (nw * nh).max(1) as f64;
    let mean_weight = (weight_sum[0] + weight_sum[1] + weight_sum[2]) / (3.0 * pixels);
    let stats = MergeStats {
        mean_samples_per_channel: mean_weight as f32,
        min_coverage: min_cov,
        filled_pixels,
        frames: used.len(),
        elapsed_ms: started.elapsed().as_millis(),
        masked_pct: stats_masked.clone(),
    };
    let out_shifts: Vec<Shift> = shifts
        .iter()
        .enumerate()
        .map(|(i, s)| Shift {
            dx: if params.align { s.dx } else { 0.0 },
            dy: if params.align { s.dy } else { 0.0 },
            rot: if params.align { s.rot } else { 0.0 },
            residual: s.residual,
            confidence: s.confidence,
            masked_pct: stats_masked.get(i).copied().unwrap_or(0.0),
        })
        .collect();
    Ok(MergeResult {
        width: nw,
        height: nh,
        scale: params.scale.max(1),
        planes,
        white: span,
        shifts: out_shifts,
        stats,
    })
}

fn first_pixel_guard(frames: &[RawFrame]) -> Result<()> {
    let (w, h) = (frames[0].width, frames[0].height);
    for (i, f) in frames.iter().enumerate() {
        if f.width != w || f.height != h {
            bail!(
                "merge: frame {} is {}x{}, expected {}x{}",
                i,
                f.width,
                f.height,
                w,
                h
            );
        }
        if f.data.len() < w * h {
            bail!("merge: frame {} has a short sample buffer", i);
        }
    }
    Ok(())
}

pub(crate) fn applied_shifts(frames: &[RawFrame], params: &MergeParams) -> Result<Vec<Shift>> {
    // Without alignment every frame lands on the same lattice, two of the three
    // channels get no samples at all and nearly every pixel is reported as
    // filled: that is the expected degenerate result, not a bug.
    if !params.align {
        return Ok(vec![Shift::default(); frames.len()]);
    }
    if params.shifts.is_empty() {
        return Ok(vec![Shift::default(); frames.len()]);
    }
    if params.shifts.len() != frames.len() {
        bail!(
            "merge: {} shifts for {} frames",
            params.shifts.len(),
            frames.len()
        );
    }
    Ok(params.shifts.clone())
}

/// Index of the frames that take part. With denoise = false only the first
/// frame of every distinct sub-pixel position is kept (quantised to 1/4 px, so
/// that measurement noise of a few hundredths of a pixel still groups).
fn used_frames(params: &MergeParams, shifts: &[Shift]) -> Vec<usize> {
    if params.denoise {
        return (0..shifts.len()).collect();
    }
    let mut seen: Vec<(i64, i64, i64)> = Vec::new();
    let mut keep = Vec::new();
    for (i, s) in shifts.iter().enumerate() {
        let key = (
            (s.dx * 4.0).round() as i64,
            (s.dy * 4.0).round() as i64,
            (s.rot * 1e5).round() as i64,
        );
        if !seen.contains(&key) {
            seen.push(key);
            keep.push(i);
        }
    }
    keep
}

fn run_band(
    views: &[FrameView],
    ow: usize,
    oh: usize,
    y0: usize,
    rows: usize,
    span: f32,
) -> BandOut {
    let n = ow * rows;
    let mut sum: [Vec<i32>; CHANNELS] = [vec![0i32; n], vec![0i32; n], vec![0i32; n]];
    let mut wt: [Vec<f32>; CHANNELS] = [vec![0f32; n], vec![0f32; n], vec![0f32; n]];
    let y_end = y0 + rows;

    for f in views {
        // Source rows whose image can reach this band, rotation included.
        let lo_out = y0 as f64 - 1.0 - f.rot_margin;
        let hi_out = y_end as f64 + 1.0 + f.rot_margin;
        let ylo = (((lo_out - f.oy) / f.by + f.cy) - 2.0).floor().max(0.0) as usize;
        let yhi = (((hi_out - f.oy) / f.by + f.cy) + 2.0).ceil().min(f.h as f64) as usize;
        for y in ylo..yhi.min(f.h) {
            let ry = y as f64 - f.cy;
            let base_x = f.ox + f.bx * ry;
            let base_y = f.oy + f.by * ry;
            let c_even = f.cfa.0[y & 1][0];
            let c_odd = f.cfa.0[y & 1][1];
            let base = y * f.w;
            for x in 0..f.w {
                if let Some(m) = f.mask {
                    if m.blocked_at(x, y) {
                        continue;
                    }
                }
                let odd = x & 1 == 1;
                let c = if odd { c_odd } else { c_even };
                let ci = c.idx();
                let b = f.black[ci];
                let v = ((f.data[base + x] as f32 - b).clamp(0.0, span)) * FIXED;
                let rx = x as f64 - f.cx;
                let mut xx = base_x + f.ax * rx;
                let mut yy = base_y + f.ay * rx;
                if let Some(g) = &f.grid {
                    let (gdx, gdy) = g.sample(x as f64, y as f64);
                    xx += f.scale * gdx;
                    yy += f.scale * gdy;
                }
                let yb = yy.floor();
                let fy = (yy - yb) as f32;
                let yb_i = yb as i64;
                let first_row = yb_i - RECON_RADIUS;
                if first_row + 3 < y0 as i64 || first_row >= y_end as i64 {
                    continue;
                }
                let wy = cr_weights(fy);
                let xb = xx.floor();
                let fx = (xx - xb) as f32;
                let xb_i = xb as i64;
                let first_col = xb_i - RECON_RADIUS;
                if first_col + 3 < 0 || first_col >= ow as i64 {
                    continue;
                }
                let wx = cr_weights(fx);
                let s = &mut sum[ci];
                let w = &mut wt[ci];
                for (k, &wyv) in wy.iter().enumerate() {
                    let r = first_row + k as i64;
                    if r < y0 as i64 || r >= y_end as i64 {
                        continue;
                    }
                    let row = (r - y0 as i64) as usize * ow;
                    for (j, &wxv) in wx.iter().enumerate() {
                        let c0 = first_col + j as i64;
                        if c0 < 0 || c0 >= ow as i64 {
                            continue;
                        }
                        let i = row + c0 as usize;
                        s[i] += (v * wxv * wyv) as i32;
                        w[i] += wxv * wyv;
                    }
                }
            }
        }
    }

    let mut planes: [Vec<u16>; CHANNELS] = [vec![0u16; n], vec![0u16; n], vec![0u16; n]];
    let mut cov: [Vec<u8>; CHANNELS] = [vec![0u8; n], vec![0u8; n], vec![0u8; n]];
    let mut weight_sum = [0.0f64; CHANNELS];
    for c in 0..CHANNELS {
        let (s, w) = (&sum[c], &wt[c]);
        let (p, cv) = (&mut planes[c], &mut cov[c]);
        for i in 0..n {
            let weight = w[i];
            if weight > 0.0 {
                let value = s[i] as f32 / (weight * FIXED);
                p[i] = value.round().clamp(0.0, span) as u16;
                cv[i] = (weight * COV_SCALE).round().min(255.0) as u8;
            }
        }
        for &weight in w.iter() {
            weight_sum[c] += weight as f64;
        }
        let _ = oh;
    }
    BandOut {
        planes,
        cov,
        weight_sum,
    }
}

/// Magnitude of the one axis Catmull-Rom response at angular frequency w
/// (radians per output pixel) for a splat at fractional position f.
///
/// The splat kernel reproduces linear ramps exactly, so its second moment is
/// identically zero and says nothing about sharpness; what it really does is
/// lose amplitude at high frequencies (0.88 at half the Nyquist frequency for a
/// splat at f = 0.5, 1.0 at f = 0). That deficit is what the post filter undoes.
fn cr_response(f: f64, w: f64) -> f64 {
    let wts = cr_weights((f - f.floor()) as f32);
    let (mut re, mut im) = (0.0f64, 0.0f64);
    for (i, wv) in wts.iter().enumerate() {
        let p = i as f64 - 1.0;
        re += *wv as f64 * (w * p).cos();
        im -= *wv as f64 * (w * p).sin();
    }
    (re * re + im * im).sqrt().clamp(1e-3, 4.0)
}

/// Log amplitude deficit of the splat kernel at DECONV_OMEGA, averaged over the
/// frames that build the image, in output pixels. It is exactly 0 on an ideal
/// lattice, where every sample lands on an output pixel and the splat is a
/// delta; on the measured bursts the positions are 0.2 .. 0.35 px off the
/// lattice, which is the softness that is left after the merge.
fn splat_deficit(shifts: &[Shift], used: &[usize], k: f64) -> f64 {
    let (mut acc, mut n) = (0.0f64, 0.0f64);
    for &i in used {
        if let Some(s) = shifts.get(i) {
            let gain = cr_response(k * s.dx, DECONV_OMEGA) * cr_response(k * s.dy, DECONV_OMEGA);
            acc += -gain.ln();
            n += 1.0;
        }
    }
    if n > 0.0 {
        acc / n
    } else {
        0.0
    }
}

/// Coefficient of the separable inverse: the filter [ -k, 1 + 2k, -k ] has the
/// response 1 + 2k(1 - cos w), so k = (1 / gain - 1) / (2 (1 - cos w)) restores
/// the frequency DECONV_OMEGA exactly. The response rises monotonically with w,
/// which is why the gain at DECONV_OMEGA is a conservative target: the output
/// Nyquist frequency is the one that would explode the noise.
///
/// amount <= 1 scales the compensation of the reconstruction kernel itself.
/// amount > 1 additionally inverts the sensor aperture (the pixel integration,
/// response sinc(w * scale / 2)), which the reconstruction cannot recover but
/// which is the remaining part of the softness measured against NX Studio.
/// Per sensor-block share of frames the motion mask removed: 0 = the merge
/// used everything (deconvolution fully applies), 1 = the block fell back to
/// the reference frame alone (single-frame Bayer content must not be
/// sharpened - the demosaic pattern and chromatic fringes would only get
/// louder).
struct DeconvGate {
    step: usize,
    cols: usize,
    rows: usize,
    frac: Vec<f32>,
}

fn deconv_gate(params: &MergeParams, used: &[usize]) -> Option<DeconvGate> {
    if params.masks.is_empty() {
        return None;
    }
    let contributors: Vec<usize> = used.iter().copied().filter(|&i| i != 0).collect();
    let first = contributors.first()?;
    let template = params.masks.get(*first).and_then(|m| m.as_ref())?;
    let (cols, rows, step) = (template.cols, template.rows, template.step);
    if cols == 0 || rows == 0 || step == 0 {
        return None;
    }
    let mut frac = vec![0.0f32; cols * rows];
    for &i in &contributors {
        if let Some(mask) = params.masks.get(i).and_then(|m| m.as_ref()) {
            if mask.cols == cols && mask.rows == rows {
                for (f, &b) in frac.iter_mut().zip(mask.blocked.iter()) {
                    *f += if b != 0 { 1.0 } else { 0.0 };
                }
            }
        }
    }
    let n = contributors.len() as f32;
    for f in frac.iter_mut() {
        *f /= n;
    }
    Some(DeconvGate {
        step,
        cols,
        rows,
        frac,
    })
}

fn deconv_coeff(amount: f32, scale: f64, splat_def: f64) -> f64 {
    let a = amount as f64;
    if !(a > 0.0) || !splat_def.is_finite() || !(scale > 0.0) {
        return 0.0;
    }
    let u = DECONV_OMEGA * scale * 0.5;
    let aperture = if u.abs() > 1e-9 { (u.sin() / u).abs().max(1e-3) } else { 1.0 };
    let deficit = a.min(1.0) * splat_def.max(0.0) + (a - 1.0).max(0.0) * -aperture.ln();
    let gain = (-deficit).exp();
    ((1.0 / gain - 1.0) / (2.0 * (1.0 - DECONV_OMEGA.cos()))).clamp(0.0, DECONV_K_MAX)
}

/// Noise level of a high pass from its DECONV_NOISE_PCT quantile.
fn high_pass_noise(hp: &[f32]) -> f32 {
    let mut max = 0.0f32;
    for v in hp {
        let a = v.abs();
        if a > max {
            max = a;
        }
    }
    if !(max > 0.0) {
        return 0.0;
    }
    const BINS: usize = 2048;
    let scale = BINS as f32 / max;
    let mut hist = vec![0u32; BINS];
    for v in hp {
        hist[((v.abs() * scale) as usize).min(BINS - 1)] += 1;
    }
    let target = ((hp.len() as f64) * DECONV_NOISE_PCT) as u64;
    let mut acc = 0u64;
    for (b, &c) in hist.iter().enumerate() {
        acc += c as u64;
        if acc >= target.max(1) {
            return (b as f32 + 1.0) / scale / RAYLEIGH_P10;
        }
    }
    max / RAYLEIGH_P10
}

/// Compensate the reconstruction kernel on every plane separately, in linear
/// data, before the 16 bit conversion. The inverse is the tensor product of the
/// one axis filter [-k, 1 + 2k, -k], so the splat blur along x and along y are
/// both undone; a step edge is constant along y, which makes the second axis of
/// the inverse an identity there and keeps the overshoot one dimensional.
fn deconvolve_planes(
    planes: &mut [Vec<u16>; CHANNELS],
    w: usize,
    h: usize,
    k: f32,
    span: f32,
    motion_gate: Option<(&DeconvGate, usize, usize, usize)>,
) {
    for plane in planes.iter_mut() {
        deconvolve_plane(plane, w, h, k, span, motion_gate);
    }
}

fn deconvolve_plane(
    plane: &mut [u16],
    w: usize,
    h: usize,
    k: f32,
    span: f32,
    motion_gate: Option<(&DeconvGate, usize, usize, usize)>,
) {
    let n = w * h;
    if n == 0 || w < 2 || h < 2 || !(k > 0.0) {
        return;
    }
    let cur: Vec<f32> = plane.iter().map(|&v| v as f32).collect();
    let row_step = BAND_ROWS * w;
    let mut hp = vec![0.0f32; n];
    hp.par_chunks_mut(row_step).enumerate().for_each(|(b, dst)| {
        let y0 = b * BAND_ROWS;
        let rows = dst.len() / w;
        // Horizontal difference of the source rows y0 - 1 ..= y0 + rows.
        let mut tmp = vec![0.0f32; (rows + 2) * w];
        for r in 0..rows + 2 {
            let sy = (y0 + r).saturating_sub(1).min(h - 1);
            let src = &cur[sy * w..sy * w + w];
            let out = &mut tmp[r * w..r * w + w];
            for x in 0..w {
                out[x] = 2.0 * src[x] - src[x.saturating_sub(1)] - src[(x + 1).min(w - 1)];
            }
        }
        for r in 0..rows {
            let up = &tmp[r * w..r * w + w];
            let mid = &tmp[(r + 1) * w..(r + 1) * w + w];
            let dn = &tmp[(r + 2) * w..(r + 2) * w + w];
            for x in 0..w {
                dst[r * w + x] = 2.0 * mid[x] - up[x] - dn[x];
            }
        }
    });

    let gate = DECONV_NOISE_GATE * high_pass_noise(&hp);
    let clamp_abs = DECONV_CLAMP_ABS * span;
    if std::env::var_os("PIXELSHIFT_DEBUG_DECONV").is_some() {
        eprintln!("  plane: k {:.4}, noise gate {:.1} ADU, clamp floor {:.1} ADU", k, gate, clamp_abs);
    }
    let gate2 = gate * gate;
    plane
        .par_chunks_mut(row_step)
        .enumerate()
        .for_each(|(b, dst)| {
            let y0 = b * BAND_ROWS;
            let rows = dst.len() / w;
            for r in 0..rows {
                let y = y0 + r;
                let y_lo = y.saturating_sub(1);
                let y_hi = (y + 1).min(h - 1);
                // Motion gate: the crop origin (x0, y0) and the output grid
                // density map this row onto a sensor block row.
                let block_row = motion_gate.map(|(g, _x0, y0c, kden)| {
                    let sensor_y = (y + y0c) / kden.max(1);
                    (sensor_y / g.step).min(g.rows - 1) * g.cols
                });
                let (g_step, g_x0, g_kden, g_cols, g_frac): (usize, usize, usize, usize, &[f32]) =
                    match motion_gate {
                        Some((g, x0c, _y0c, kden)) => {
                            (g.step, x0c, kden, g.cols, &g.frac[..])
                        }
                        None => (0, 0, 1, 0, &[][..]),
                    };
                for x in 0..w {
                    let i = y * w + x;
                    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
                    for yy in y_lo..=y_hi {
                        let row = &cur[yy * w..yy * w + w];
                        for xx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                            let v = row[xx];
                            if v < lo {
                                lo = v;
                            }
                            if v > hi {
                                hi = v;
                            }
                        }
                    }
                    let h = hp[i];
                    let mut corr = k * h * if gate2 > 0.0 { h * h / (h * h + gate2) } else { 1.0 };
                    if let (Some(base), true) = (block_row, g_cols > 0) {
                        let sensor_x = (x + g_x0) / g_kden.max(1);
                        let col = (sensor_x / g_step).min(g_cols - 1);
                        corr *= 1.0 - g_frac[base + col];
                    }
                    let lim = (DECONV_CLAMP_FRAC * (hi - lo)).max(clamp_abs);
                    let v = (cur[i] + corr).clamp(lo - lim, hi + lim).clamp(0.0, span);
                    dst[r * w + x] = v.round() as u16;
                }
            }
        });
}

/// Catmull-Rom basis at fractional position f for taps -1, 0, 1, 2.
#[inline]
fn cr_weights(f: f32) -> [f32; 4] {
    let f2 = f * f;
    let f3 = f2 * f;
    [
        -0.5 * f3 + f2 - 0.5 * f,
        1.5 * f3 - 2.5 * f2 + 1.0,
        -1.5 * f3 + 2.0 * f2 + 0.5 * f,
        0.5 * f3 - 0.5 * f2,
    ]
}

/// Sensor-pixel rectangle covered by every used frame after the rigid
/// alignment: the intersection of the axis aligned boxes of the mapped frame
/// rectangles. The reference frame (identity) bounds the result; the box of a
/// rotated frame overestimates its true area, which only makes the crop a
/// little more conservative.
pub fn stabilized_rect_pub(
    shifts: &[Shift],
    used: &[usize],
    w: usize,
    h: usize,
) -> (usize, usize, usize, usize) {
    stabilized_rect(shifts, used, w, h)
}

fn stabilized_rect(shifts: &[Shift], used: &[usize], w: usize, h: usize) -> (usize, usize, usize, usize) {
    let (mut x0, mut y0, mut x1, mut y1) = (0usize, 0usize, w, h);
    for &i in used {
        let s = shifts.get(i).copied().unwrap_or_default();
        let (cx, cy) = (w as f64 * 0.5, h as f64 * 0.5);
        let corners = [
            (0.0, 0.0),
            (w as f64, 0.0),
            (0.0, h as f64),
            (w as f64, h as f64),
        ];
        let (mut fx0, mut fy0, mut fx1, mut fy1) =
            (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for (px, py) in corners {
            let (mx, my) = s.map(px, py, cx, cy);
            fx0 = fx0.min(mx);
            fy0 = fy0.min(my);
            fx1 = fx1.max(mx);
            fy1 = fy1.max(my);
        }
        x0 = x0.max(fx0.ceil().max(0.0) as usize);
        y0 = y0.max(fy0.ceil().max(0.0) as usize);
        x1 = x1.min(fx1.floor().min(w as f64).max(0.0) as usize);
        y1 = y1.min(fy1.floor().min(h as f64).max(0.0) as usize);
    }
    // One sample of margin on each side keeps the splat footprint covered.
    (x0.min(w), y0.min(h), x1.max(x0 + 1), y1.max(y0 + 1))
}

fn crop_plane(src: &[u16], stride: usize, x: usize, y: usize, w: usize, h: usize) -> Vec<u16> {
    let mut out = Vec::with_capacity(w * h);
    for row in 0..h {
        let start = (y + row) * stride + x;
        out.extend_from_slice(&src[start..start + w]);
    }
    out
}

fn crop_bytes(src: &[u8], stride: usize, x: usize, y: usize, w: usize, h: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(w * h);
    for row in 0..h {
        let start = (y + row) * stride + x;
        out.extend_from_slice(&src[start..start + w]);
    }
    out
}

/// Rebuild uncovered pixels by iterated neighbour averaging inside one channel.
/// Returns the number of pixels that were rebuilt.
fn fill_channel(plane: &mut [u16], mask: &mut [u8], w: usize, h: usize) -> usize {
    let mut holes: Vec<u32> = Vec::new();
    for (i, m) in mask.iter().enumerate() {
        if *m != 0 {
            holes.push(i as u32);
        }
    }
    let total = holes.len();
    if total == 0 {
        return 0;
    }
    let mut remaining = holes;
    for _ in 0..MAX_FILL_ITERS {
        if remaining.is_empty() {
            return total;
        }
        let mut next: Vec<u32> = Vec::with_capacity(remaining.len());
        for &idx in &remaining {
            let i = idx as usize;
            if mask[i] == 0 {
                continue;
            }
            let x = i % w;
            let y = i / w;
            let mut acc = 0u32;
            let mut cnt = 0u32;
            if x > 0 && mask[i - 1] == 0 {
                acc += plane[i - 1] as u32;
                cnt += 1;
            }
            if x + 1 < w && mask[i + 1] == 0 {
                acc += plane[i + 1] as u32;
                cnt += 1;
            }
            if y > 0 && mask[i - w] == 0 {
                acc += plane[i - w] as u32;
                cnt += 1;
            }
            if y + 1 < h && mask[i + w] == 0 {
                acc += plane[i + w] as u32;
                cnt += 1;
            }
            if cnt > 0 {
                plane[i] = ((acc as f32) / cnt as f32).round() as u16;
                mask[i] = 0;
            } else {
                next.push(idx);
            }
        }
        remaining = next;
    }
    total
}

/// Box-average a 2x result down to sensor resolution. Returns the input
/// unchanged when scale == 1.
pub fn downsample_1x(img: &MergeResult) -> MergeResult {
    if img.scale <= 1 {
        return img.clone();
    }
    let (ow, oh) = (img.width / 2, img.height / 2);
    let mut out = MergeResult {
        width: ow,
        height: oh,
        scale: 1,
        planes: [Vec::with_capacity(ow * oh), Vec::with_capacity(ow * oh), Vec::with_capacity(ow * oh)],
        white: img.white,
        shifts: img.shifts.clone(),
        stats: img.stats.clone(),
    };
    for c in 0..CHANNELS {
        let src = &img.planes[c];
        let dst = &mut out.planes[c];
        for y in 0..oh {
            for x in 0..ow {
                let i = (2 * y) * img.width + 2 * x;
                let s = src[i] as u32
                    + src[i + 1] as u32
                    + src[i + img.width] as u32
                    + src[i + img.width + 1] as u32;
                dst.push(((s as f32) * 0.25).round() as u16);
            }
        }
    }
    out
}

/// Build an sRGB preview for the GUI, scaled so that the longest side is at
/// most max_dim (never upscales). White balance is applied, then the 99.5
/// percentile of the mean channel maps to white.
pub fn preview(img: &MergeResult, max_dim: u32, wb: [f32; 3]) -> PreviewImage {
    let longest = img.width.max(img.height);
    let factor = if max_dim == 0 || longest as u32 <= max_dim {
        1
    } else {
        (longest as u32).div_ceil(max_dim) as usize
    };
    let sw = img.width.div_ceil(factor);
    let sh = img.height.div_ceil(factor);
    let mut buf = vec![0.0f32; sw * sh * 3];
    for y in 0..sh {
        for x in 0..sw {
            let y0 = y * factor;
            let x0 = x * factor;
            let y1 = (y0 + factor).min(img.height);
            let x1 = (x0 + factor).min(img.width);
            let mut acc = [0.0f64; 3];
            let mut n = 0.0f64;
            for yy in y0..y1 {
                for xx in x0..x1 {
                    let i = yy * img.width + xx;
                    for c in 0..3 {
                        acc[c] += img.planes[c][i] as f64;
                    }
                    n += 1.0;
                }
            }
            let o = (y * sw + x) * 3;
            for c in 0..3 {
                buf[o + c] = (acc[c] / n.max(1.0)) as f32 * wb[c];
            }
        }
    }
    encode_srgb(&buf, sw, sh)
}

/// Crop an axis aligned window out of the merged image and encode it as sRGB
/// (white balance applied, white point from the window itself).
pub fn preview_crop(
    img: &MergeResult,
    rect: (usize, usize, usize, usize),
    wb: [f32; 3],
) -> PreviewImage {
    let x0 = rect.0.min(img.width);
    let y0 = rect.1.min(img.height);
    let w = rect.2.min(img.width.saturating_sub(x0));
    let h = rect.3.min(img.height.saturating_sub(y0));
    if w == 0 || h == 0 {
        return PreviewImage {
            width: 0,
            height: 0,
            rgb: Vec::new(),
        };
    }
    let mut buf = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y0 + y) * img.width + x0 + x;
            let o = (y * w + x) * 3;
            for c in 0..3 {
                buf[o + c] = img.planes[c][i] as f32 * wb[c];
            }
        }
    }
    encode_srgb(&buf, w, h)
}

/// Crop a window at a reduced sampling step: one output sample per step x
/// step block, box averaged (f64 like the base preview), so intermediate zoom
/// levels get a texture sized to the viewport instead of jumping straight
/// between the 2048 px base preview and a full resolution 1:1 crop. The
/// rectangle is aligned outward to block boundaries; blocks clipped by the
/// image border contribute only their real samples. `step = 1` is exactly
/// `preview_crop`.
pub fn preview_crop_at(
    img: &MergeResult,
    rect: (usize, usize, usize, usize),
    step: u32,
    wb: [f32; 3],
) -> PreviewImage {
    if step <= 1 {
        return preview_crop(img, rect, wb);
    }
    let s = step as usize;
    let x0 = rect.0.min(img.width);
    let y0 = rect.1.min(img.height);
    let x1 = (rect.0 + rect.2).min(img.width).max(x0);
    let y1 = (rect.1 + rect.3).min(img.height).max(y0);
    if x1 == x0 || y1 == y0 {
        return PreviewImage {
            width: 0,
            height: 0,
            rgb: Vec::new(),
        };
    }
    let bx0 = x0 - x0 % s;
    let by0 = y0 - y0 % s;
    let out_w = (x1 - bx0).div_ceil(s);
    let out_h = (y1 - by0).div_ceil(s);
    let mut buf = vec![0.0f32; out_w * out_h * 3];
    for oy in 0..out_h {
        let ry0 = by0 + oy * s;
        let ry1 = (ry0 + s).min(img.height);
        for ox in 0..out_w {
            let rx0 = bx0 + ox * s;
            let rx1 = (rx0 + s).min(img.width);
            let mut acc = [0.0f64; 3];
            let mut n = 0.0f64;
            for yy in ry0..ry1 {
                let row = yy * img.width;
                for xx in rx0..rx1 {
                    let i = row + xx;
                    for c in 0..3 {
                        acc[c] += img.planes[c][i] as f64;
                    }
                    n += 1.0;
                }
            }
            let o = (oy * out_w + ox) * 3;
            for c in 0..3 {
                buf[o + c] = (acc[c] / n.max(1.0)) as f32 * wb[c];
            }
        }
    }
    encode_srgb(&buf, out_w, out_h)
}

/// Shared preview finisher for every GUI view: the 99.5 percentile of the
/// window luminance maps to white, then the samples go through the sRGB
/// curve. Input is white-balanced linear f32 in 0..~1 scale; using it for
/// merged and reference views alike keeps them tonally consistent.
pub(crate) fn encode_srgb(buf: &[f32], w: usize, h: usize) -> PreviewImage {
    if buf.is_empty() {
        return PreviewImage {
            width: 0,
            height: 0,
            rgb: Vec::new(),
        };
    }
    let mut lum: Vec<f32> = buf
        .chunks_exact(3)
        .map(|p| (p[0] + p[1] + p[2]) / 3.0)
        .filter(|v| v.is_finite())
        .collect();
    let white = if lum.is_empty() {
        1.0
    } else {
        let idx = ((lum.len() as f64 * 0.995) as usize).min(lum.len() - 1);
        lum.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        lum[idx].max(1.0)
    };
    let mut rgb = Vec::with_capacity(w * h * 3);
    for p in buf.chunks_exact(3) {
        for v in p {
            rgb.push(srgb_u8((v / white).clamp(0.0, 1.0)));
        }
    }
    PreviewImage {
        width: w,
        height: h,
        rgb,
    }
}

fn srgb_u8(v: f32) -> u8 {
    let e = if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (e * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

/// Per-channel sample mean, used by CLI inspect and tests.
pub fn channel_means(img: &MergeResult) -> [f64; 3] {
    let mut out = [0.0f64; 3];
    for c in [Color::R, Color::G, Color::B] {
        let p = img.plane(c);
        let sum: u64 = p.iter().map(|v| *v as u64).sum();
        out[c.idx()] = sum as f64 / p.len().max(1) as f64;
    }
    out
}
