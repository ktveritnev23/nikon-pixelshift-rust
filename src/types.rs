//! Shared data contract used by raw I/O, the alignment/merge core, the writers
//! and the GUI. Extension by any work package must keep these signatures
//! stable; add new fields with defaults instead of changing existing ones.

use std::path::PathBuf;

/// Number of colour channels in the linear output.
pub const CHANNELS: usize = 3;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Color {
    R = 0,
    G = 1,
    B = 2,
}

impl Color {
    pub fn idx(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

impl Rect {
    pub fn new(x: usize, y: usize, w: usize, h: usize) -> Self {
        Self { x, y, w, h }
    }
}

/// 2x2 colour filter array of the active area: colors[y & 1][x & 1].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cfa(pub [[Color; 2]; 2]);

impl Cfa {
    /// RGGB-style helper: Cfa::from_pattern_str("RGGB").
    pub fn from_pattern_str(pattern: &str) -> Option<Self> {
        let chars: Vec<char> = pattern.chars().filter(|c| !c.is_whitespace()).collect();
        if chars.len() != 4 {
            return None;
        }
        let mut out = [[Color::R; 2]; 2];
        for (i, ch) in chars.iter().enumerate() {
            out[i / 2][i % 2] = match ch.to_ascii_uppercase() {
                'R' => Color::R,
                'G' => Color::G,
                'B' => Color::B,
                _ => return None,
            };
        }
        Some(Cfa(out))
    }

    pub fn pattern_str(self) -> String {
        let mut s = String::with_capacity(4);
        for y in 0..2 {
            for x in 0..2 {
                s.push(match self.0[y][x] {
                    Color::R => 'R',
                    Color::G => 'G',
                    Color::B => 'B',
                });
            }
        }
        s
    }
}

/// One decoded raw frame, already cropped to the camera active (visible)
/// pixel area: data is width * height samples in camera native units,
/// (0, 0) being the top-left active pixel and cfa being the pattern at that
/// pixel.
#[derive(Clone)]
pub struct RawFrame {
    pub path: PathBuf,
    pub width: usize,
    pub height: usize,
    pub data: Vec<u16>,
    pub cfa: Cfa,
    /// Global black level (subtracted from every sample).
    pub black: u16,
    /// Per-channel black offsets added on top of black (LibRaw cblack).
    pub cblack: [u32; 4],
    /// Saturation level in the same units as data.
    pub white: u16,
    /// As-shot white balance multipliers in camera native units (r, g, b, g).
    pub wb: [f32; 4],
    /// LibRaw color.cam_xyz: rows R,G,B; columns X,Y,Z (XYZ -> camera).
    pub cam_xyz: [[f32; 3]; 4],
    /// LibRaw color.rgb_cam: rows R,G,B; columns R,G,B (camera -> sRGB D65).
    pub rgb_cam: [[f32; 3]; 4],
    pub iso: f32,
    pub shutter: f32,
    pub aperture: f32,
    pub focal: f32,
    /// Lens identification as recorded in the file ("" when unknown); raw
    /// editors use it to pick their own lens profile.
    pub lens_make: String,
    pub lens_model: String,
    /// Focal length in 35 mm terms, 0 when unknown.
    pub focal_35mm: u16,
    /// Maximum aperture as an f number, 0 when unknown.
    pub max_aperture: f32,
    /// Lens focal range and the widest aperture at each end as the file records
    /// them (LibRaw MinFocal/MaxFocal/MaxAp4MinFocal/MaxAp4MaxFocal); 0 when
    /// unknown. A prime lens has min == max, a zoom reports both ends, which is
    /// what an EXIF LensSpecification needs.
    pub lens_min_focal: f32,
    pub lens_max_focal: f32,
    pub lens_min_fnum: f32,
    pub lens_max_fnum: f32,
    /// Lens and camera body serial numbers as recorded, "" when unknown.
    pub lens_serial: String,
    pub body_serial: String,
    pub make: String,
    pub model: String,
    pub timestamp: String,
    /// Position of the file inside the series (0 = first).
    pub index: usize,
}

impl RawFrame {
    #[inline]
    pub fn sample(&self, x: usize, y: usize) -> u16 {
        self.data[y * self.width + x]
    }

    #[inline]
    pub fn color_at(&self, x: usize, y: usize) -> Color {
        self.cfa.0[y & 1][x & 1]
    }

    /// Effective black level of one colour channel (clamped below white).
    pub fn black_for(&self, c: Color) -> u16 {
        let extra = self.cblack.get(c.idx()).copied().unwrap_or(0) as u32;
        ((self.black as u32).saturating_add(extra)).min(self.white as u32) as u16
    }

    pub fn pixel_count(&self) -> usize {
        self.width * self.height
    }

    /// Saturation in black-subtracted units (range of the merged output).
    pub fn white_span(&self) -> u16 {
        self.white.saturating_sub(self.black)
    }
}

/// Rigid transform of a frame relative to the reference frame.
///
/// Sign convention: pixel (x, y) of the frame samples the scene point
/// `map(x, y, cx, cy)` expressed in reference frame pixel coordinates, where
/// `(cx, cy)` is the image centre the rotation is taken about. A frame
/// physically shifted half a pixel to the right therefore has dx = +0.5 and a
/// frame rotated by a tiny angle has a non-zero `rot`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Shift {
    pub dx: f64,
    pub dy: f64,
    /// Rotation about the image centre in radians; small values (1e-4) are
    /// typical for a pixel shift series and must not be dropped, they move the
    /// image corners by up to ~0.6 px on a 6048 px wide sensor.
    pub rot: f64,
    /// Residual RMS of the fit in sensor pixels, 0 when not measured.
    pub residual: f32,
    /// Fitting quality 0..1 (higher is better).
    pub confidence: f32,
}

impl Shift {
    pub fn new(dx: f64, dy: f64) -> Self {
        Self {
            dx,
            dy,
            rot: 0.0,
            residual: 0.0,
            confidence: 1.0,
        }
    }

    pub fn with_rot(dx: f64, dy: f64, rot: f64) -> Self {
        Self {
            dx,
            dy,
            rot,
            residual: 0.0,
            confidence: 1.0,
        }
    }

    /// Reference frame coordinates of the scene point seen by frame pixel
    /// (x, y) when the rotation is taken about (cx, cy).
    pub fn map(&self, x: f64, y: f64, cx: f64, cy: f64) -> (f64, f64) {
        let (sin, cos) = self.rot.sin_cos();
        let rx = (x - cx) * cos - (y - cy) * sin;
        let ry = (x - cx) * sin + (y - cy) * cos;
        (cx + rx + self.dx, cy + ry + self.dy)
    }

    pub fn is_identity(&self) -> bool {
        self.dx.abs() < 1e-9 && self.dy.abs() < 1e-9 && self.rot.abs() < 1e-12
    }

    /// Rotation expressed in arc seconds, for display.
    pub fn rot_arcsec(&self) -> f64 {
        self.rot * 180.0 / std::f64::consts::PI * 3600.0
    }
}

/// Coarse residual correction field over the image: one anchor per grid node,
/// values are added to the rigid transform and expressed in sensor pixels.
///
/// Real pixel shift bursts are not perfectly rigid: on series 106/107 the
/// measured displacement deviates from a translation + rotation model by up to
/// 1.5 px at the corners (drift of the mechanism during the burst), while the
/// model fits the centre to a few hundredths of a pixel.
#[derive(Clone, Debug, Default)]
pub struct ShiftGrid {
    pub x0: f64,
    pub y0: f64,
    pub step_x: f64,
    pub step_y: f64,
    pub cols: usize,
    pub rows: usize,
    /// Residual dx per anchor (index = col + row * cols), sensor px.
    pub dx: Vec<f32>,
    /// Residual dy per anchor (index = col + row * cols), sensor px.
    pub dy: Vec<f32>,
}

impl ShiftGrid {
    pub fn is_usable(&self) -> bool {
        self.cols >= 2
            && self.rows >= 2
            && self.dx.len() >= self.cols * self.rows
            && self.dy.len() >= self.cols * self.rows
            && self.step_x > 0.0
            && self.step_y > 0.0
    }

    /// Bilinearly interpolated residual at (x, y); clamped to the border
    /// anchors outside the grid.
    pub fn sample(&self, x: f64, y: f64) -> (f64, f64) {
        if !self.is_usable() {
            return (0.0, 0.0);
        }
        let fx = ((x - self.x0) / self.step_x).clamp(0.0, (self.cols - 1) as f64);
        let fy = ((y - self.y0) / self.step_y).clamp(0.0, (self.rows - 1) as f64);
        let c0 = fx.floor() as usize;
        let r0 = fy.floor() as usize;
        let c1 = (c0 + 1).min(self.cols - 1);
        let r1 = (r0 + 1).min(self.rows - 1);
        let tx = fx - c0 as f64;
        let ty = fy - r0 as f64;
        let at = |r: usize, c: usize| r * self.cols + c;
        let w00 = (1.0 - tx) * (1.0 - ty);
        let w10 = tx * (1.0 - ty);
        let w01 = (1.0 - tx) * ty;
        let w11 = tx * ty;
        let dx = w00 * self.dx[at(r0, c0)] as f64
            + w10 * self.dx[at(r0, c1)] as f64
            + w01 * self.dx[at(r1, c0)] as f64
            + w11 * self.dx[at(r1, c1)] as f64;
        let dy = w00 * self.dy[at(r0, c0)] as f64
            + w10 * self.dy[at(r0, c1)] as f64
            + w01 * self.dy[at(r1, c0)] as f64
            + w11 * self.dy[at(r1, c1)] as f64;
        (dx, dy)
    }
}

#[derive(Clone, Debug)]
pub struct MergeParams {
    /// Output grid density: 1 = sensor resolution, 2 = double resolution on
    /// both axes (four times the pixel count).
    pub scale: u32,
    /// Apply measured sub-pixel shifts. When false, shifts are treated as 0.
    pub align: bool,
    /// Frame shifts consumed by merge (shifts[0] == 0,0).
    pub shifts: Vec<Shift>,
    /// Output pixels whose coverage is below this weight are rebuilt from
    /// neighbours instead of being left as holes.
    pub min_coverage: f32,
    /// Throw away the outermost pixels whose colour coverage is incomplete.
    pub crop_uncovered: bool,
    /// Keep all redundant samples (8/32 frame modes) and average them.
    pub denoise: bool,
    /// Optional residual correction grid per frame (same order as `shifts`).
    pub grids: Vec<Option<ShiftGrid>>,
    /// Strength of the reconstruction-kernel compensation (deconvolution) applied
    /// to the merged planes, 0.0 = pure normalised splatting (soft), 1.0 = full
    /// compensation of the splat kernel. Compensating keeps the linear data
    /// honest (it only undoes the known blur of the reconstruction), while a
    /// sharpening filter would not be invertible.
    pub deconvolve: f32,
}

impl Default for MergeParams {
    fn default() -> Self {
        Self {
            scale: 1,
            align: true,
            shifts: Vec::new(),
            min_coverage: 0.25,
            crop_uncovered: false,
            denoise: true,
            grids: Vec::new(),
            // Measured on the four reference series: compensating the
            // reconstruction kernel raises the high frequency energy from 0.44
            // to 0.57 of the NX Studio result on series 106 (0.73 -> 0.99 on
            // 107) for +15 % noise in flat areas. Higher values keep sharpening
            // past the point where the data supports it.
            deconvolve: 1.0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MergeStats {
    /// Mean number of samples contributed per colour channel per output pixel.
    pub mean_samples_per_channel: f32,
    /// Smallest per-channel coverage over the whole frame.
    pub min_coverage: f32,
    /// Output pixels that had to be rebuilt from neighbours.
    pub filled_pixels: usize,
    pub frames: usize,
    pub elapsed_ms: u128,
}

/// Linear RGB result of the merge, in camera native units (black subtracted,
/// white balance NOT applied), 16 bit.
#[derive(Clone)]
pub struct MergeResult {
    pub width: usize,
    pub height: usize,
    pub scale: u32,
    /// Planar RGB: planes[0] = R, planes[1] = G, planes[2] = B.
    pub planes: [Vec<u16>; CHANNELS],
    /// Saturation level of the stored samples.
    pub white: u16,
    /// Shifts that were applied (frame 0 is the reference, shift 0,0).
    pub shifts: Vec<Shift>,
    pub stats: MergeStats,
}

impl MergeResult {
    pub fn pixel_count(&self) -> usize {
        self.width * self.height
    }

    pub fn plane(&self, c: Color) -> &[u16] {
        &self.planes[c.idx()]
    }

    pub fn plane_mut(&mut self, c: Color) -> &mut Vec<u16> {
        &mut self.planes[c.idx()]
    }

    pub fn color_at(&self, c: Color, x: usize, y: usize) -> u16 {
        self.planes[c.idx()][y * self.width + x]
    }
}

/// Metadata needed by the writers.
#[derive(Clone, Debug)]
pub struct DngMeta {
    pub make: String,
    pub model: String,
    pub timestamp: String,
    pub software: String,
    /// Camera-native RGB triple that renders as neutral (DNG AsShotNeutral).
    pub as_shot_neutral: [f32; 3],
    /// XYZ (D50) -> camera native matrix for the stored data, row major.
    pub color_matrix: [[f32; 3]; 3],
    /// DNG calibration illuminant code (21 = D65, 17 = Standard light A).
    pub calib_illuminant: u16,
    pub baseline_exposure: f32,
    pub iso: f32,
    pub shutter: f32,
    pub aperture: f32,
    pub focal: f32,
    pub orientation: u16,
    /// True when the stored pixels already have the white balance applied.
    pub wb_applied: bool,
    /// Lens identification, copied from the NEF so that raw editors can match
    /// their own lens profile (distortion / vignetting correction).
    pub lens_make: String,
    pub lens_model: String,
    /// Focal length in 35 mm terms, 0 when unknown.
    pub focal_35mm: u16,
    /// Maximum aperture as an f number, 0 when unknown.
    pub max_aperture: f32,
    /// Focal range and the widest aperture at each end, 0 when unknown; the
    /// four values of EXIF LensSpecification.
    pub lens_min_focal: f32,
    pub lens_max_focal: f32,
    pub lens_min_fnum: f32,
    pub lens_max_fnum: f32,
    /// Serial numbers, "" when the source file has none (never made up).
    pub lens_serial: String,
    pub body_serial: String,
    /// Maker note (EXIF 0x927C) copied verbatim from the reference file; raw
    /// editors read the lens identification of the camera vendor from it.
    /// Empty means "do not write the tag".
    pub makernote: Vec<u8>,
}

impl Default for DngMeta {
    fn default() -> Self {
        Self {
            make: String::new(),
            model: String::new(),
            timestamp: String::new(),
            software: String::new(),
            as_shot_neutral: [1.0, 1.0, 1.0],
            color_matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            calib_illuminant: 21,
            baseline_exposure: 0.0,
            iso: 0.0,
            shutter: 0.0,
            aperture: 0.0,
            focal: 0.0,
            orientation: 1,
            wb_applied: false,
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
            makernote: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct WriteOpts {
    /// Compress the image data losslessly: the DNG writer stores lossless
    /// JPEG (Compression = 7, the process NEFs themselves use), the TIFF
    /// writer stores AdobeDeflate with the horizontal predictor.
    pub compress: bool,
    /// Longest side of the embedded preview / thumbnail (0 = none).
    pub preview_max: u32,
}

impl Default for WriteOpts {
    fn default() -> Self {
        // Uncompressed by default: the exact lossless bytes of the merged
        // data, no codec in the reading path.
        Self {
            compress: false,
            preview_max: 0,
        }
    }
}

/// 8-bit preview image for the GUI.
#[derive(Clone)]
pub struct PreviewImage {
    pub width: usize,
    pub height: usize,
    /// width * height * 3 bytes, sRGB encoded.
    pub rgb: Vec<u8>,
}
