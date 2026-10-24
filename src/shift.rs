//! Sub-pixel alignment of the frames of one Pixel Shift series.
//!
//! Owned by work package WP2 (alignment + merge core).
//!
//! Geometry (docs/ARCHITECTURE.md, types::Shift): pixel (x, y) of a frame sees
//! the scene point map(x, y, cx, cy) of the reference frame, where the transform
//! is a rotation about the image centre plus a translation. Nine windows across
//! the field are measured independently and a least squares rigid model
//! (dx, dy, rot) is fitted to them; the residual of that fit is reported.
//!
//! Estimation runs on a CFA phase invariant projection: every 2x2 Bayer block
//! holds exactly R + G + G + B, therefore the block sums on the 2 px lattice
//! describe the box filtered scene and do not depend on the sampling phase of
//! the mosaic. Interpolating the raw green plane instead biases the result by up
//! to a few tenths of a pixel when the CFA phase changes between frames.
//! A shift of s sensor pixels is a shift of s / 2 block pixels, so the working
//! estimate is doubled when it is reported.

use crate::types::{Color, RawFrame, Rect, Shift, ShiftGrid};
use anyhow::{bail, Result};
use rayon::prelude::*;

/// Integer search radius on the 2 px lattice around the global estimate (8
/// sensor pixels).
const SEARCH_RADIUS: i32 = 4;
/// Edge of one working window in lattice pixels (1024 sensor pixels) for the
/// small frame fallback layout.
const CROP: usize = 512;
/// Window centres as a fraction of the frame: a 3x3 layout for small frames.
const WINDOWS3: [f64; 3] = [0.2, 0.5, 0.8];
/// Edge of one working window for the 5x5 layout: 2048 sensor pixels, the size
/// at which a window still measures the residual of the rigid model (a 1024 px
/// window is dominated by estimate noise of the same order as the correction).
const CROP5: usize = 1024;
/// Window centres of the 5x5 layout. A rigid model leaves a drift of up to
/// 0.13 px RMS by window and 1.5 px at the corners of the 16 frame series, which
/// needs finer sampling than 3x3 to be corrected.
const WINDOWS5: [f64; 5] = [0.1, 0.3, 0.5, 0.7, 0.9];
/// Lattice size from which the 5x5 layout is used (a 4000x2800 sensor frame).
const GRID5_MIN: (usize, usize) = (2000, 1400);
/// Pixel stride of the Gauss-Newton normal equations for large windows: every
/// second sample keeps the estimate unbiased (both planes are low passed before)
/// and makes 25 windows affordable.
const REFINE_STRIDE: usize = 2;
/// Context kept around a window for the search range and interpolation kernels.
const PAD: usize = 6;
const BLUR_RADIUS: i32 = 3;
/// Low pass applied to both working planes; it pushes the content well below
/// the lattice Nyquist frequency, which keeps the cubic interpolation bias far
/// below the 0.01 px target.
const BLUR_SIGMA: f32 = 0.8;
const MAX_ITER: usize = 24;
const ITER_TOL: f64 = 1e-4;
/// Windows disagreeing with the rigid model by more than this are dropped.
const OUTLIER_PX: f64 = 0.12;
/// Physical limit of the pixel shift mechanism (about 400 arc seconds). Measured
/// series reach 200 .. 320 arc seconds: warping a frame onto the reference with
/// the rotation halves the residual on real pixels (mean 23.9 -> 12.3 ADU on
/// 106 frame 1, corner window 46.9 -> 20.0), so the rotation must be kept. A fit
/// beyond this limit is degenerate and falls back to a pure translation.
const MAX_ROT: f64 = 2.0e-3;
/// A rotation is only accepted when it lowers the residual to this fraction.
const ROT_GAIN: f64 = 0.8;
/// Windows below this fraction of the best texture energy are ignored by the
/// fit: defocused parts of the frame measure as noise only.
const MIN_ENERGY: f64 = 0.05;

/// Single channel f32 plane with clamped Catmull-Rom sampling.
pub(crate) struct Plane {
    pub(crate) w: usize,
    pub(crate) h: usize,
    pub(crate) data: Vec<f32>,
}

impl Plane {
    #[inline]
    pub(crate) fn at(&self, x: i64, y: i64) -> f32 {
        let xi = x.clamp(0, self.w as i64 - 1) as usize;
        let yi = y.clamp(0, self.h as i64 - 1) as usize;
        self.data[yi * self.w + xi]
    }

}

/// Six tap Lanczos-3 kernel for one axis. The fractional offset of a window is
/// the same for every pixel of that window during one Gauss-Newton iteration, so
/// the weights are built once per iteration instead of per sample. Cubic
/// (Catmull-Rom) interpolation would leave a 0.01 .. 0.02 px interpolation bias,
/// six taps bring it below 0.002 px.
pub(crate) struct Taps {
    off: [i64; 6],
    w: [f32; 6],
}

pub(crate) fn taps(frac: f64) -> Taps {
    let mut w = [0.0f32; 6];
    let mut sum = 0.0f64;
    for (k, wv) in w.iter_mut().enumerate() {
        let t = frac - (k as f64 - 2.0);
        let v = lanczos3(t);
        *wv = v as f32;
        sum += v;
    }
    if sum.abs() > 1e-12 {
        for wv in w.iter_mut() {
            *wv = (*wv as f64 / sum) as f32;
        }
    }
    Taps {
        off: [-2, -1, 0, 1, 2, 3],
        w,
    }
}

fn lanczos3(t: f64) -> f64 {
    let a = t.abs();
    if a < 1e-9 {
        return 1.0;
    }
    if a >= 3.0 {
        return 0.0;
    }
    let x = std::f64::consts::PI * t;
    let sinc = x.sin() / x;
    let y = x / 3.0;
    sinc * (y.sin() / y)
}

#[inline]
pub(crate) fn sample_taps(p: &Plane, base_x: i64, base_y: i64, tx: &Taps, ty: &Taps) -> f32 {
    let mut acc = 0.0f32;
    for j in 0..6 {
        let y = base_y + ty.off[j];
        let mut row = 0.0f32;
        for i in 0..6 {
            row += tx.w[i] * p.at(base_x + tx.off[i], y);
        }
        acc += ty.w[j] * row;
    }
    acc
}

/// Estimate the rigid transform of frame relative to reference, in sensor
/// pixels (translation) and radians (rotation about the frame centre).
pub fn estimate(reference: &RawFrame, frame: &RawFrame) -> Result<Shift> {
    check_pair(reference, frame)?;
    let preps = build_preps(reference);
    Ok(measure(&preps, frame, reference.width, reference.height).0)
}

/// Rigid transform, the local residual grid of one frame, and the correlation
/// of its measurement windows.
type Measurement = (Shift, Option<ShiftGrid>, f32);

/// Below this window correlation a frame is re-registered through its
/// predecessor instead of against the reference frame.
const DIRECT_NCC: f32 = 0.5;

/// Estimate all shifts of a series relative to its first frame.
///
/// Frames are registered against frame 0 in parallel. A frame whose correlation
/// is too weak to register directly is chained through its predecessor
/// (shift[i] = shift[i-1] o shift(frame[i-1], frame[i])), which keeps the result
/// consistent with the reference frame.
pub fn estimate_series(frames: &[RawFrame]) -> Result<Vec<Shift>> {
    Ok(estimate_series_with_grids(frames)?.0)
}

/// Estimate all shifts of a series and, per frame, the grid of local residuals
/// that the rigid model leaves over the field (one anchor per measurement window
/// centre: 5x5 of 2048 px windows on a camera frame, 3x3 on small images).
/// Frames whose rigid fit is unreliable get no grid, and nothing is applied for
/// them.
pub fn estimate_series_with_grids(
    frames: &[RawFrame],
) -> Result<(Vec<Shift>, Vec<Option<ShiftGrid>>)> {
    let Some(first) = frames.first() else {
        bail!("shift::estimate_series: empty frame list");
    };
    for f in frames {
        check_pair(first, f)?;
    }
    let mut shifts = vec![Shift::default(); frames.len()];
    let mut grids: Vec<Option<ShiftGrid>> = vec![None; frames.len()];
    if frames.len() == 1 {
        return Ok((shifts, grids));
    }
    let preps = build_preps(first);
    let direct: Vec<Measurement> = frames[1..]
        .par_iter()
        .map(|f| measure(&preps, f, first.width, first.height))
        .collect();
    let mut weak = vec![false; frames.len()];
    for (i, (s, g, ncc)) in direct.into_iter().enumerate() {
        shifts[i + 1] = s;
        grids[i + 1] = g;
        weak[i + 1] = ncc < DIRECT_NCC;
    }

    for i in 1..frames.len() {
        // A well correlated frame is kept even when a rigid model does not
        // describe it exactly: the residual of a real burst is up to 0.4 px rms
        // and pushing such a frame through its predecessor would drop the grid
        // that corrects exactly that residual.
        if !weak[i] {
            continue;
        }
        let prev = shifts[i - 1];
        let Ok(step) = estimate(&frames[i - 1], &frames[i]) else {
            continue;
        };
        shifts[i] = compose(prev, step);
        grids[i] = None;
    }
    Ok((shifts, grids))
}

/// Compose two rigid transforms: apply step (frame i -> frame i-1) and then
/// prev (frame i-1 -> frame 0).
fn compose(prev: Shift, step: Shift) -> Shift {
    let (sin_p, cos_p) = prev.rot.sin_cos();
    let (sin_s, cos_s) = step.rot.sin_cos();
    let tx = prev.dx + cos_p * step.dx - sin_p * step.dy;
    let ty = prev.dy + sin_p * step.dx + cos_p * step.dy;
    let rot = (sin_p * cos_s + cos_p * sin_s).atan2(cos_p * cos_s - sin_p * sin_s);
    Shift {
        dx: tx,
        dy: ty,
        rot,
        residual: prev.residual.max(step.residual),
        confidence: 0.5 * step.confidence * prev.confidence.max(0.5),
        masked_pct: 0.0,
    }
}

fn check_pair(reference: &RawFrame, frame: &RawFrame) -> Result<()> {
    if (reference.width, reference.height) != (frame.width, frame.height) {
        bail!(
            "shift::estimate: frame size {}x{} differs from reference {}x{}",
            frame.width,
            frame.height,
            reference.width,
            reference.height
        );
    }
    if reference.width < 64 || reference.height < 64 {
        bail!(
            "shift::estimate: frame {}x{} too small for alignment",
            reference.width,
            reference.height
        );
    }
    Ok(())
}

fn build_preps(reference: &RawFrame) -> Vec<CropPrep> {
    window_rects(reference.width / 2, reference.height / 2)
        .into_iter()
        .map(|r| CropPrep::build(reference, r))
        .collect()
}

/// Window rectangles in lattice coordinates, clamped and de-duplicated. Large
/// frames get the 5x5 layout with 2048 sensor pixel windows, small ones the
/// 3x3 layout with windows of about a quarter of the frame.
fn window_rects(wl: usize, hl: usize) -> Vec<Rect> {
    let wide = wl >= GRID5_MIN.0 && hl >= GRID5_MIN.1;
    let (fracs, crop) = if wide {
        (&WINDOWS5[..], CROP5)
    } else {
        (&WINDOWS3[..], CROP)
    };
    let (cw, ch) = if wide {
        ((wl - 2).min(crop), (hl - 2).min(crop))
    } else {
        (
            (wl / 4).clamp(32, crop).min(wl - 2),
            (hl / 4).clamp(32, crop).min(hl - 2),
        )
    };
    let mut out: Vec<Rect> = Vec::with_capacity(fracs.len() * fracs.len());
    for fy in fracs {
        for fx in fracs {
            let cx = (wl as f64 * fx) as usize;
            let cy = (hl as f64 * fy) as usize;
            let x = cx.saturating_sub(cw / 2).min(wl - cw - 1);
            let y = cy.saturating_sub(ch / 2).min(hl - ch - 1);
            let r = Rect::new(x, y, cw, ch);
            if !out.contains(&r) {
                out.push(r);
            }
        }
    }
    out
}

/// Reference side of one window: the working plane, its gradient and the region
/// used for the normal equations. Built once per series.
struct CropPrep {
    rect: Rect,
    plane: Plane,
    gx: Plane,
    gy: Plane,
    /// Accumulation region in plane coordinates.
    inner: Rect,
    /// Mean squared gradient over the accumulation region.
    energy: f64,
}

impl CropPrep {
    fn build(reference: &RawFrame, rect: Rect) -> Self {
        let wl = reference.width / 2;
        let hl = reference.height / 2;
        let x0 = rect.x.saturating_sub(PAD).max(1);
        let y0 = rect.y.saturating_sub(PAD).max(1);
        let x1 = (rect.x + rect.w + PAD).min(wl - 1);
        let y1 = (rect.y + rect.h + PAD).min(hl - 1);
        let plane_rect = Rect::new(x0, y0, x1 - x0, y1 - y0);
        let m = SEARCH_RADIUS as usize + 2;
        let ix0 = rect.x.max(plane_rect.x + m);
        let iy0 = rect.y.max(plane_rect.y + m);
        let ix1 = (rect.x + rect.w).min(plane_rect.x + plane_rect.w - m);
        let iy1 = (rect.y + rect.h).min(plane_rect.y + plane_rect.h - m);
        let inner = Rect::new(
            ix0.saturating_sub(x0),
            iy0.saturating_sub(y0),
            ix1.saturating_sub(ix0).max(1),
            iy1.saturating_sub(iy0).max(1),
        );
        let plane = luma_plane(reference, plane_rect);
        let (gx, gy) = gradients(&plane);
        let mut energy = 0.0f64;
        for y in inner.y..inner.y + inner.h {
            for x in inner.x..inner.x + inner.w {
                let i = y * gx.w + x;
                energy += (gx.data[i] as f64).powi(2) + (gy.data[i] as f64).powi(2);
            }
        }
        energy /= (inner.w * inner.h).max(1) as f64;
        Self {
            rect: plane_rect,
            plane,
            gx,
            gy,
            inner,
            energy,
        }
    }

    /// Window centre in sensor coordinates.
    fn center(&self) -> (f64, f64) {
        (
            (self.rect.x + self.rect.w / 2) as f64 * 2.0,
            (self.rect.y + self.rect.h / 2) as f64 * 2.0,
        )
    }
}

/// One window measurement, in sensor pixels.
#[derive(Clone, Copy, Default)]
struct WindowEst {
    x: f64,
    y: f64,
    dx: f64,
    dy: f64,
    ncc: f32,
    /// Mean squared gradient of the reference window; defocused windows carry
    /// orders of magnitude less and must not steer the rotation fit.
    energy: f64,
}

/// Two stage measurement: the window nearest to the centre gives a global
/// translation, which then initialises every window (a defocused corner can
/// otherwise pull the Gauss-Newton iteration into a wrong minimum).
fn measure(preps: &[CropPrep], frame: &RawFrame, width: usize, height: usize) -> Measurement {
    if preps.is_empty() {
        return (
            Shift {
                dx: 0.0,
                dy: 0.0,
                rot: 0.0,
                residual: 0.0,
                confidence: 0.0,
                masked_pct: 0.0,
            },
            None,
            0.0,
        );
    }
    let (cx, cy) = (width as f64 * 0.5, height as f64 * 0.5);
    let ci = preps
        .iter()
        .enumerate()
        .min_by_key(|(_, p)| {
            let (x, y) = p.center();
            let dx = (x - cx).abs() as i64;
            let dy = (y - cy).abs() as i64;
            dx * dx + dy * dy
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    let center_est = estimate_window(&preps[ci], frame, (0, 0), None);
    let init = (center_est.dx * 0.5, center_est.dy * 0.5);
    let search = (init.0.round() as i32, init.1.round() as i32);
    let ests: Vec<WindowEst> = preps
        .par_iter()
        .map(|p| estimate_window(p, frame, search, Some(init)))
        .collect();
    fit_rigid(&ests, cx, cy)
}

fn estimate_window(
    prep: &CropPrep,
    frame: &RawFrame,
    search_center: (i32, i32),
    init: Option<(f64, f64)>,
) -> WindowEst {
    let tgt = luma_plane(frame, prep.rect);
    let (ix, iy, ncc) = coarse_shift(&prep.plane, &tgt, search_center);
    let s0 = init.unwrap_or((ix as f64, iy as f64));
    let (dx, dy) = refine(prep, &tgt, s0);
    let (x, y) = prep.center();
    WindowEst {
        x,
        y,
        dx: 2.0 * dx,
        dy: 2.0 * dy,
        ncc,
        energy: prep.energy,
    }
}

/// Texture weighted least squares rigid model of the window measurements:
/// dx(x, y) = dx0 - rot * (y - cy), dy(x, y) = dy0 + rot * (x - cx).
///
/// The rotation is only kept when it is physically plausible and when it really
/// explains the measurements: nine noisy windows otherwise produce a huge
/// spurious rotation (a few low texture windows in the defocused part of the
/// frame are enough), which shifts the corners by pixels and destroys the merge.
fn fit_rigid(ests: &[WindowEst], cx: f64, cy: f64) -> Measurement {
    let max_energy = ests.iter().map(|e| e.energy).fold(0.0f64, f64::max);
    let mut w: Vec<f64> = ests
        .iter()
        .map(|e| {
            if max_energy > 0.0 {
                (e.energy / max_energy).min(1.0)
            } else {
                1.0
            }
        })
        .collect();
    let mut keep: Vec<usize> = (0..ests.len()).filter(|&i| w[i] >= MIN_ENERGY).collect();
    if keep.len() < 3 {
        keep = (0..ests.len()).collect();
        for wi in w.iter_mut() {
            *wi = 1.0;
        }
    }

    let mut model = (0.0, 0.0, 0.0);
    for pass in 0..3 {
        model = solve_rigid(ests, &keep, &w, cx, cy);
        let mut worst = (0usize, 0.0f64);
        for (k, &i) in keep.iter().enumerate() {
            let (ex, ey) = expected(model, ests[i].x, ests[i].y, cx, cy);
            let d = ((ests[i].dx - ex).powi(2) + (ests[i].dy - ey).powi(2)).sqrt();
            if d > worst.1 {
                worst = (k, d);
            }
        }
        if pass > 0 || keep.len() < 4 || worst.1 <= OUTLIER_PX.max(3.0 * rms_of(ests, &keep, &w, model, cx, cy)) {
            break;
        }
        keep.remove(worst.0);
    }

    let total_w: f64 = keep.iter().map(|&i| w[i]).sum();
    let (mut tx, mut ty) = (0.0f64, 0.0f64);
    for &i in &keep {
        tx += w[i] * ests[i].dx;
        ty += w[i] * ests[i].dy;
    }
    tx /= total_w.max(1e-9);
    ty /= total_w.max(1e-9);

    let rms_shift = rms_of(ests, &keep, &w, (tx, ty, 0.0), cx, cy);
    let rms_rigid = rms_of(ests, &keep, &w, model, cx, cy);
    if std::env::var_os("PIXELSHIFT_DEBUG_SHIFT").is_some() {
        for (i, e) in ests.iter().enumerate() {
            eprintln!(
                "  window {:2} ({:6.0},{:6.0}) dx {:+7.3} dy {:+7.3} ncc {:.3} energy {:10.1} w {:.3}{}",
                i,
                e.x,
                e.y,
                e.dx,
                e.dy,
                e.ncc,
                e.energy,
                if max_energy > 0.0 { e.energy / max_energy } else { 1.0 },
                if keep.contains(&i) { "" } else { "  dropped" }
            );
        }
        eprintln!(
            "  fit rigid ({:+.3},{:+.3}, {:+.1} arcsec) rms {:.3} | translation ({:+.3},{:+.3}) rms {:.3}",
            model.0,
            model.1,
            model.2 * 180.0 / std::f64::consts::PI * 3600.0,
            rms_rigid,
            tx,
            ty,
            rms_shift
        );
    }
    let (dx, dy, mut rot, rms) = if model.2.abs() <= MAX_ROT && rms_rigid <= ROT_GAIN * rms_shift
    {
        (model.0, model.1, model.2, rms_rigid)
    } else {
        (tx, ty, 0.0, rms_shift)
    };
    rot = if rot.abs() <= MAX_ROT { rot } else { 0.0 };

    let mut ncc = 0.0f32;
    for &i in &keep {
        ncc += w[i] as f32 * ests[i].ncc.max(0.0);
    }
    let ncc = ncc / total_w.max(1e-9) as f32;
    let conf = (ncc * ((-rms / 0.3).exp()) as f32).clamp(0.0, 1.0);
    let shift = Shift {
        dx,
        dy,
        rot,
        residual: rms as f32,
        confidence: conf,
        masked_pct: 0.0,
    };
    (shift, build_grid(ests, &keep, model, cx, cy), ncc)
}

/// Residual field of the rigid model on the measurement anchors, sampled
/// bilinearly by types::ShiftGrid. Frames with fewer than three usable windows
/// get no grid: the rigid model alone is safer than an interpolated guess.
fn build_grid(
    ests: &[WindowEst],
    keep: &[usize],
    model: (f64, f64, f64),
    cx: f64,
    cy: f64,
) -> Option<ShiftGrid> {
    if keep.len() < 3 {
        return None;
    }
    let mut xs: Vec<f64> = ests.iter().map(|e| e.x).collect();
    let mut ys: Vec<f64> = ests.iter().map(|e| e.y).collect();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    xs.dedup_by(|a, b| (*a - *b).abs() < 1.0);
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ys.dedup_by(|a, b| (*a - *b).abs() < 1.0);
    if xs.len() < 2 || ys.len() < 2 {
        return None;
    }
    let (cols, rows) = (xs.len(), ys.len());
    let mut dx = vec![0.0f32; cols * rows];
    let mut dy = vec![0.0f32; cols * rows];
    let anchor = |est: &WindowEst| -> Option<(usize, usize)> {
        let c = xs.iter().position(|v| (v - est.x).abs() < 1.0)?;
        let r = ys.iter().position(|v| (v - est.y).abs() < 1.0)?;
        Some((c, r))
    };
    for &i in keep {
        if let Some((c, r)) = anchor(&ests[i]) {
            let (ex, ey) = expected(model, ests[i].x, ests[i].y, cx, cy);
            dx[c + r * cols] = (ests[i].dx - ex) as f32;
            dy[c + r * cols] = (ests[i].dy - ey) as f32;
        }
    }
    let known: Vec<bool> = (0..cols * rows).map(|_| false).collect();
    let mut known = known;
    for &i in keep {
        if let Some((c, r)) = anchor(&ests[i]) {
            known[c + r * cols] = true;
        }
    }
    for r in 0..rows {
        for c in 0..cols {
            let i = c + r * cols;
            if known[i] {
                continue;
            }
            let (mut sw, mut sx, mut sy) = (0.0f64, 0.0f64, 0.0f64);
            for &k in keep {
                if let Some((cc, rr)) = anchor(&ests[k]) {
                    let d2 = ((cc as f64 - c as f64).powi(2) + (rr as f64 - r as f64).powi(2)).max(0.25);
                    let wgt = 1.0 / d2;
                    sx += wgt * dx[cc + rr * cols] as f64;
                    sy += wgt * dy[cc + rr * cols] as f64;
                    sw += wgt;
                }
            }
            if sw > 0.0 {
                dx[i] = (sx / sw) as f32;
                dy[i] = (sy / sw) as f32;
            }
        }
    }
    // The residual field of the rigid model is smooth: the drift of the pixel
    // shift mechanism varies over the frame, not pixel by pixel. A binomial
    // smoothing keeps a linear drift exact and cuts the estimate noise of the
    // anchors, which is what made a 3x3 / 1024 px grid useless.
    if cols >= 4 && rows >= 4 {
        smooth_anchors(&mut dx, &mut dy, cols, rows);
    }
    let grid = ShiftGrid {
        x0: xs[0],
        y0: ys[0],
        step_x: (xs[cols - 1] - xs[0]) / (cols - 1) as f64,
        step_y: (ys[rows - 1] - ys[0]) / (rows - 1) as f64,
        cols,
        rows,
        dx,
        dy,
    };
    grid.is_usable().then_some(grid)
}

/// 3x3 binomial [1 2 1] x [1 2 1] / 16 smoothing with clamped borders.
fn smooth_anchors(dx: &mut [f32], dy: &mut [f32], cols: usize, rows: usize) {
    let (src_x, src_y) = (dx.to_vec(), dy.to_vec());
    let at = |r: usize, c: usize| r * cols + c;
    for r in 0..rows {
        for c in 0..cols {
            let (mut sx, mut sy) = (0.0f32, 0.0f32);
            for (dr, wr) in [(-1i64, 1.0f32), (0, 2.0), (1, 1.0)] {
                let rr = (r as i64 + dr).clamp(0, rows as i64 - 1) as usize;
                for (dc, wc) in [(-1i64, 1.0f32), (0, 2.0), (1, 1.0)] {
                    let cc = (c as i64 + dc).clamp(0, cols as i64 - 1) as usize;
                    let w = wr * wc / 16.0;
                    sx += w * src_x[at(rr, cc)];
                    sy += w * src_y[at(rr, cc)];
                }
            }
            dx[at(r, c)] = sx;
            dy[at(r, c)] = sy;
        }
    }
}

fn rms_of(
    ests: &[WindowEst],
    keep: &[usize],
    w: &[f64],
    model: (f64, f64, f64),
    cx: f64,
    cy: f64,
) -> f64 {
    let mut acc = 0.0f64;
    let mut total = 0.0f64;
    for &i in keep {
        let (ex, ey) = expected(model, ests[i].x, ests[i].y, cx, cy);
        let d2 = (ests[i].dx - ex).powi(2) + (ests[i].dy - ey).powi(2);
        acc += w[i] * d2;
        total += w[i];
    }
    (acc / total.max(1e-9)).sqrt()
}

/// Local shift predicted by the model at window centre (x, y), exact form.
fn expected(model: (f64, f64, f64), x: f64, y: f64, cx: f64, cy: f64) -> (f64, f64) {
    let (sin, cos) = model.2.sin_cos();
    let rx = x - cx;
    let ry = y - cy;
    (
        model.0 + rx * (cos - 1.0) - ry * sin,
        model.1 + rx * sin + ry * (cos - 1.0),
    )
}

fn solve_rigid(
    ests: &[WindowEst],
    keep: &[usize],
    w: &[f64],
    cx: f64,
    cy: f64,
) -> (f64, f64, f64) {
    let mut n = 0.0f64;
    if keep.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let (mut sa, mut sb) = (0.0f64, 0.0f64);
    let (mut su, mut sv) = (0.0f64, 0.0f64);
    let (mut au, mut bv) = (0.0f64, 0.0f64);
    let mut suu_vv = 0.0f64;
    for &i in keep {
        let e = ests[i];
        let wi = w[i];
        let u = e.y - cy;
        let v = e.x - cx;
        n += wi;
        sa += wi * e.dx;
        sb += wi * e.dy;
        su += wi * u;
        sv += wi * v;
        au += wi * e.dx * u;
        bv += wi * e.dy * v;
        suu_vv += wi * (u * u + v * v);
    }
    if n < 1.0 || suu_vv < 1e-6 {
        return (sa / n.max(1e-9), sb / n.max(1e-9), 0.0);
    }
    let m = [[n, 0.0, -su], [0.0, n, sv], [-su, sv, suu_vv]];
    let rhs = [sa, sb, bv - au];
    match solve3(m, rhs) {
        Some(s) => (s[0], s[1], s[2]),
        None => (sa / n, sb / n, 0.0),
    }
}

fn solve3(m: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-9 {
        return None;
    }
    let mut out = [0.0f64; 3];
    for col in 0..3 {
        let mut c = m;
        for row in 0..3 {
            c[row][col] = b[row];
        }
        let d = c[0][0] * (c[1][1] * c[2][2] - c[1][2] * c[2][1])
            - c[0][1] * (c[1][0] * c[2][2] - c[1][2] * c[2][0])
            + c[0][2] * (c[1][0] * c[2][1] - c[1][1] * c[2][0]);
        out[col] = d / det;
    }
    Some(out)
}

/// CFA phase invariant working plane: each value is the sum of one 2x2 sensor
/// block, black subtracted and low passed. The rectangle is in lattice
/// coordinates.
pub(crate) fn luma_plane(frame: &RawFrame, rect: Rect) -> Plane {
    let w = frame.width;
    let black = frame.black_for(Color::R) as f32
        + 2.0 * frame.black_for(Color::G) as f32
        + frame.black_for(Color::B) as f32;
    let (pw, ph) = (rect.w, rect.h);
    let mut data = vec![0.0f32; pw * ph];
    for yy in 0..ph {
        let sy = (rect.y + yy) * 2;
        for xx in 0..pw {
            let sx = (rect.x + xx) * 2;
            let i = sy * w + sx;
            let s = frame.data[i] as f32
                + frame.data[i + 1] as f32
                + frame.data[i + w] as f32
                + frame.data[i + w + 1] as f32;
            data[yy * pw + xx] = s - black;
        }
    }
    blur(&mut data, pw, ph);
    Plane { w: pw, h: ph, data }
}

fn blur(data: &mut [f32], w: usize, h: usize) {
    let r = BLUR_RADIUS;
    let mut k = Vec::with_capacity((2 * r + 1) as usize);
    for i in -r..=r {
        k.push((-(i * i) as f32 / (2.0 * BLUR_SIGMA * BLUR_SIGMA)).exp());
    }
    let norm: f32 = k.iter().sum();
    for v in &mut k {
        *v /= norm;
    }
    let mut tmp = vec![0.0f32; data.len()];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (i, kv) in k.iter().enumerate() {
                let sx = (x as i64 + i as i64 - r as i64).clamp(0, w as i64 - 1) as usize;
                acc += kv * data[y * w + sx];
            }
            tmp[y * w + x] = acc;
        }
    }
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (i, kv) in k.iter().enumerate() {
                let sy = (y as i64 + i as i64 - r as i64).clamp(0, h as i64 - 1) as usize;
                acc += kv * tmp[sy * w + x];
            }
            data[y * w + x] = acc;
        }
    }
}

fn gradients(p: &Plane) -> (Plane, Plane) {
    let mut gx = Plane {
        w: p.w,
        h: p.h,
        data: vec![0.0; p.w * p.h],
    };
    let mut gy = Plane {
        w: p.w,
        h: p.h,
        data: vec![0.0; p.w * p.h],
    };
    for y in 0..p.h {
        for x in 0..p.w {
            let i = y * p.w + x;
            gx.data[i] = 0.5 * (p.at(x as i64 + 1, y as i64) - p.at(x as i64 - 1, y as i64));
            gy.data[i] = 0.5 * (p.at(x as i64, y as i64 + 1) - p.at(x as i64, y as i64 - 1));
        }
    }
    (gx, gy)
}

/// Integer search by normalised cross correlation on a 2x decimated subset,
/// around the supplied centre.
fn coarse_shift(reference: &Plane, target: &Plane, center: (i32, i32)) -> (i32, i32, f32) {
    let r = SEARCH_RADIUS;
    let stride = 2usize;
    let margin = r as usize + BLUR_RADIUS as usize + 1;
    if reference.w <= 2 * margin + 8 || reference.h <= 2 * margin + 8 {
        return (center.0, center.1, 0.0);
    }
    let (mut sb, mut sbb, mut n) = (0.0f64, 0.0f64, 0.0f64);
    let mut y = margin;
    while y < target.h - margin {
        let mut x = margin;
        while x < target.w - margin {
            let b = target.data[y * target.w + x] as f64;
            sb += b;
            sbb += b * b;
            n += 1.0;
            x += stride;
        }
        y += stride;
    }
    if n < 64.0 {
        return (center.0, center.1, 0.0);
    }
    let var_b = (sbb - sb * sb / n).max(1e-6);
    let mut best = (center.0, center.1, f32::NEG_INFINITY);
    for sy in -r..=r {
        for sx in -r..=r {
            let (mut sa, mut saa, mut sab) = (0.0f64, 0.0f64, 0.0f64);
            let mut yy = margin;
            while yy < target.h - margin {
                let mut xx = margin;
                while xx < target.w - margin {
                    let b = target.data[yy * target.w + xx] as f64;
                    let a = reference.at(
                        xx as i64 + sx as i64 + center.0 as i64,
                        yy as i64 + sy as i64 + center.1 as i64,
                    ) as f64;
                    sa += a;
                    saa += a * a;
                    sab += a * b;
                    xx += stride;
                }
                yy += stride;
            }
            let var_a = (saa - sa * sa / n).max(1e-6);
            let cov = sab - sa * sb / n;
            let ncc = (cov / (var_a * var_b).sqrt()) as f32;
            if ncc > best.2 {
                best = (center.0 + sx, center.1 + sy, ncc);
            }
        }
    }
    best
}

/// Gauss-Newton (Lucas-Kanade) refinement of a pure translation, in lattice
/// pixels. The reference is evaluated at (x + s), the frame stays on the grid.
fn refine(prep: &CropPrep, target: &Plane, s0: (f64, f64)) -> (f64, f64) {
    let mut s = s0;
    let inner = prep.inner;
    // Only the 5x5 layout has windows that are large enough for the stride to be
    // free: small windows are already cheap and lose accuracy with it.
    let stride = if inner.w.min(inner.h) >= 256 {
        REFINE_STRIDE
    } else {
        1
    };
    let mut scale = 0.0f64;
    for _ in 0..MAX_ITER {
        let fx0 = s.0.floor();
        let fy0 = s.1.floor();
        let tx = taps(s.0 - fx0);
        let ty = taps(s.1 - fy0);
        let (bx, by) = (fx0 as i64, fy0 as i64);
        let (mut a11, mut a12, mut a22, mut b1, mut b2) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
        let mut ss = 0.0f64;
        let mut n = 0.0f64;
        for py in (inner.y..inner.y + inner.h).step_by(stride) {
            let base_y = py as i64 + by;
            for px in (inner.x..inner.x + inner.w).step_by(stride) {
                let base_x = px as i64 + bx;
                let a = sample_taps(&prep.plane, base_x, base_y, &tx, &ty) as f64;
                let t = target.data[py * target.w + px] as f64;
                let gx = sample_taps(&prep.gx, base_x, base_y, &tx, &ty) as f64;
                let gy = sample_taps(&prep.gy, base_x, base_y, &tx, &ty) as f64;
                let res = a - t;
                let wgt = if scale > 0.0 {
                    1.0 / (1.0 + (res / scale) * (res / scale))
                } else {
                    1.0
                };
                a11 += wgt * gx * gx;
                a12 += wgt * gx * gy;
                a22 += wgt * gy * gy;
                b1 -= wgt * gx * res;
                b2 -= wgt * gy * res;
                ss += res * res;
                n += 1.0;
            }
        }
        if n < 64.0 {
            break;
        }
        scale = 2.0 * (ss / n).sqrt();
        let det = a11 * a22 - a12 * a12;
        if det.abs() < 1e-6 {
            break;
        }
        let ddx = (a22 * b1 - a12 * b2) / det;
        let ddy = (a11 * b2 - a12 * b1) / det;
        s.0 += ddx;
        s.1 += ddy;
        if ddx.abs() < ITER_TOL && ddy.abs() < ITER_TOL {
            break;
        }
        if !s.0.is_finite() || !s.1.is_finite() {
            return s0;
        }
    }
    s
}
