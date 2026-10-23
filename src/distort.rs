//! Lens distortion correction for the merged linear image.
//!
//! Brown radial model with three terms: a lens maps the ideal (rectilinear)
//! radius r, normalised so that r = 1 in the frame corner, onto the recorded
//! radius r * (1 + k1 r^2 + k2 r^4 + k3 r^6). Correcting a picture is therefore
//! one resampling pass that samples every ideal pixel at its recorded radius:
//! positive k1 pulls the corners inwards and removes pincushion, negative k1
//! removes barrel. Samples that land outside the frame are clamped to it, so
//! the border stays free of holes.
//!
//! The merged result carries no lens profile (a Nikon NEF of a third party lens
//! has no DNG opcodes either), so the coefficients are a user decision; this
//! module only has to be exact and reversible enough to judge that decision.
//! One term is not enough for a wide angle (the residual after a pure k1 fit of
//! a 24 mm frame is a few pixels in the corner), which is why the model carries
//! k2 and k3 too; [`crate::fit`] measures all three from a reference image.

use crate::types::{MergeResult, CHANNELS};
use rayon::prelude::*;
use std::fmt;

/// Largest |k| accepted per term. A 24 mm zoom is around 3-4 % of barrel
/// distortion, so the limit has to reach past it; the radial map stays
/// monotonic up to |k1| = 1/3, well beyond this. The higher terms are naturally
/// an order of magnitude smaller than k1 (r^4 and r^6 fall off fast).
pub const MAX_K1: f64 = 0.06;

/// Radial distortion coefficients of the backward (correction) map.
///
/// The mapping means the same thing in both directions, so the tests use the
/// same struct to simulate a lens: inverting the radial map turns the
/// correction into the distortion.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Distortion {
    pub k1: f64,
    pub k2: f64,
    pub k3: f64,
}

impl Distortion {
    pub const IDENTITY: Self = Self {
        k1: 0.0,
        k2: 0.0,
        k3: 0.0,
    };

    /// Two-term coefficients (k3 = 0).
    pub fn new(k1: f64, k2: f64) -> Self {
        Self {
            k1,
            k2,
            k3: 0.0,
        }
    }

    pub fn new3(k1: f64, k2: f64, k3: f64) -> Self {
        Self { k1, k2, k3 }
    }

    /// True when correcting would be a no-op, in which case rectify() does not
    /// touch the pixels at all (bit exact).
    pub fn is_identity(&self) -> bool {
        self.k1 == 0.0 && self.k2 == 0.0 && self.k3 == 0.0
    }

    pub fn is_finite(&self) -> bool {
        self.k1.is_finite() && self.k2.is_finite() && self.k3.is_finite()
    }

    /// Radius multiplier of the sampling radius at squared normalised radius r2.
    pub fn radial_scale(&self, r2: f64) -> f64 {
        1.0 + self.k1 * r2 + self.k2 * r2 * r2 + self.k3 * r2 * r2 * r2
    }

    /// Radius (normalised, 1 = corner) that an ideal pixel at radius r samples.
    pub fn map_radius(&self, r: f64) -> f64 {
        r * self.radial_scale(r * r)
    }

    /// True when the radial map is increasing on 0..=1, i.e. the correction
    /// cannot fold the image over itself.
    pub fn is_monotonic(&self) -> bool {
        let steps = 1000;
        let mut previous = 0.0;
        for i in 0..=steps {
            let r = i as f64 / steps as f64;
            let mapped = self.map_radius(r);
            if mapped < previous {
                return false;
            }
            previous = mapped;
        }
        true
    }

    /// Parse "K1", "K1,K2" or "K1,K2,K3" as accepted by the --distort option.
    pub fn parse(text: &str) -> Result<Self, String> {
        let parts: Vec<&str> = text
            .split([',', ';'])
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .collect();
        if parts.is_empty() || parts.len() > 3 {
            return Err(format!("{text}: expected K1, K1,K2 or K1,K2,K3"));
        }
        let mut values = [0.0f64; 3];
        for (slot, part) in values.iter_mut().zip(&parts) {
            let value: f64 = part
                .parse()
                .map_err(|_| format!("{part} is not a number"))?;
            if !value.is_finite() || value.abs() > MAX_K1 {
                return Err(format!(
                    "{part} is outside -{MAX_K1}..{MAX_K1} (a lens needs fractions of a percent)"
                ));
            }
            *slot = value;
        }
        Ok(Self::new3(values[0], values[1], values[2]))
    }

    /// Coefficients measured by [`crate::fit`], rejected when they are not the
    /// small fractions a lens can produce.
    pub fn from_measured(k1: f64, k2: f64, k3: f64) -> Result<Self, String> {
        Self::parse(&format!("{k1},{k2},{k3}"))
    }
}

impl fmt::Display for Distortion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "k1 {:+.4} k2 {:+.4} k3 {:+.4}",
            self.k1, self.k2, self.k3
        )
    }
}
/// Remove the distortion described by d from a merged image.
///
/// The frame size is unchanged and the image is sampled with a bicubic
/// (Catmull-Rom) kernel; identity coefficients return the input untouched.
pub fn rectify(img: MergeResult, d: Distortion) -> MergeResult {
    if d.is_identity() || img.width == 0 || img.height == 0 {
        return img;
    }
    let (w, h) = (img.width, img.height);
    let cx = (w as f64 - 1.0) * 0.5;
    let cy = (h as f64 - 1.0) * 0.5;
    let norm = 1.0 / (cx * cx + cy * cy).max(1e-12);

    let mut out = img;
    let planes: [Vec<u16>; CHANNELS] = std::array::from_fn(|c| {
        let src = &out.planes[c];
        let mut dst = vec![0u16; w * h];
        dst.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            let dy = y as f64 - cy;
            for (x, pixel) in row.iter_mut().enumerate() {
                let dx = x as f64 - cx;
                let scale = d.radial_scale((dx * dx + dy * dy) * norm);
                *pixel = sample_bicubic(src, w, h, cx + dx * scale, cy + dy * scale);
            }
        });
        dst
    });
    out.planes = planes;
    out
}

/// Apply the lens instead of removing it: the counterpart of [`rectify`],
/// used to synthesise a distorted reference in the tests and in the GUI
/// selftest.
///
/// A recorded pixel at normalised radius u samples the ideal frame at the
/// radius r with r * (1 + k1 r^2 + k2 r^4 + k3 r^6) = u, solved by fixed point
/// iteration on r^2 = u^2 / scale(r^2)^2. Only plain resampling, so the result
/// is exactly what the same coefficients remove again.
pub fn apply_lens(img: MergeResult, d: Distortion) -> MergeResult {
    if d.is_identity() || img.width == 0 || img.height == 0 {
        return img;
    }
    let (w, h) = (img.width, img.height);
    let cx = (w as f64 - 1.0) * 0.5;
    let cy = (h as f64 - 1.0) * 0.5;
    let norm = 1.0 / (cx * cx + cy * cy).max(1e-12);

    let mut out = img;
    let planes: [Vec<u16>; CHANNELS] = std::array::from_fn(|c| {
        let src = &out.planes[c];
        let mut dst = vec![0u16; w * h];
        dst.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            let dy = y as f64 - cy;
            for (x, pixel) in row.iter_mut().enumerate() {
                let dx = x as f64 - cx;
                let u2 = (dx * dx + dy * dy) * norm;
                let mut r2 = u2;
                for _ in 0..30 {
                    let scale = d.radial_scale(r2);
                    let next = u2 / (scale * scale);
                    if (next - r2).abs() < 1e-13 {
                        r2 = next;
                        break;
                    }
                    r2 = next;
                }
                let scale = if u2 > 0.0 { (r2 / u2).sqrt() } else { 1.0 };
                *pixel = sample_bicubic(src, w, h, cx + dx * scale, cy + dy * scale);
            }
        });
        dst
    });
    out.planes = planes;
    out
}

/// Catmull-Rom weights of the four taps around a fractional position.
#[inline]
fn cubic_weights(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    ]
}

/// Bicubic sample with the neighbouring samples clamped to the frame, so a
/// pixel whose source lands outside the image repeats the border instead of
/// turning into a hole.
#[inline]
pub fn sample_bicubic(src: &[u16], w: usize, h: usize, x: f64, y: f64) -> u16 {
    let x = x.clamp(0.0, (w - 1) as f64);
    let y = y.clamp(0.0, (h - 1) as f64);
    let x0 = x.floor();
    let y0 = y.floor();
    let wx = cubic_weights(x - x0);
    let wy = cubic_weights(y - y0);
    let xi = x0 as isize;
    let yi = y0 as isize;
    let last_x = w as isize - 1;
    let last_y = h as isize - 1;
    let mut acc = 0.0f64;
    for (j, weight_y) in wy.iter().enumerate() {
        if *weight_y == 0.0 {
            continue;
        }
        let yy = (yi + j as isize - 1).clamp(0, last_y) as usize;
        let row = &src[yy * w..yy * w + w];
        let mut row_acc = 0.0f64;
        for (i, weight_x) in wx.iter().enumerate() {
            if *weight_x == 0.0 {
                continue;
            }
            let xx = (xi + i as isize - 1).clamp(0, last_x) as usize;
            row_acc += row[xx] as f64 * weight_x;
        }
        acc += row_acc * weight_y;
    }
    acc.round().clamp(0.0, u16::MAX as f64) as u16
}
