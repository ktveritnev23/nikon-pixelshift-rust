//! Pipeline operations shared by the CLI and the GUI worker.
//!
//! Every call into a module owned by another work package (rawio, shift, merge,
//! write) goes through [guarded]: while those modules are still todo!() stubs,
//! the stub panic is turned into an ordinary error with a readable message, so
//! neither the headless CLI nor the GUI aborts mid-run.

use crate::app::model::{Compression, OutFormat};
use crate::types::{DngMeta, MergeParams, MergeResult, PreviewImage, RawFrame, Shift, WriteOpts};
use crate::write::DngFlavor;
use anyhow::{anyhow, bail, Context, Result};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------- guards -----

/// Serialises the global panic hook installed by [guarded].
static HOOK_LOCK: Mutex<()> = Mutex::new(());

/// Run one call into a work package module and turn a panic into an error.
///
/// todo!() stubs, an unfinished module and ordinary bugs all end up as a plain
/// error instead of aborting the process; the GUI also uses this so a missing
/// module shows up in the interface instead of killing the window. The panic
/// hook is global, so the calls are serialised by HOOK_LOCK; never nest them.
pub fn guarded<T>(module: &str, f: impl FnOnce() -> Result<T>) -> Result<T> {
    let _serial = HOOK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let captured: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let slot = captured.clone();
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let message = info
            .payload()
            .downcast_ref::<&str>()
            .map(|s| (*s).to_string())
            .or_else(|| info.payload().downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "panic".to_string());
        let location = info
            .location()
            .map(|l| format!(" at {}:{}", l.file(), l.line()))
            .unwrap_or_default();
        *slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(format!("{message}{location}"));
    }));
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    std::panic::set_hook(previous);
    match outcome {
        Ok(result) => result,
        Err(_) => {
            let message = captured
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take()
                .unwrap_or_default();
            Err(unavailable(module, &message))
        }
    }
}

fn unavailable(module: &str, message: &str) -> anyhow::Error {
    let detail = message.trim();
    if detail.contains("not yet implemented") {
        anyhow!("{module}: not implemented yet ({detail})")
    } else if detail.is_empty() {
        anyhow!("{module}: panicked")
    } else {
        anyhow!("{module}: panicked ({detail})")
    }
}

/// True when the error only says that a work package is not finished.
pub fn is_unavailable(err: &anyhow::Error) -> bool {
    format!("{err:#}").contains("not implemented yet")
}

pub fn apply_threads(threads: Option<usize>) {
    if let Some(n) = threads.filter(|n| *n > 0) {
        let _ = rayon::ThreadPoolBuilder::new().num_threads(n).build_global();
    }
}

// ---------------------------------------------------------------- series -----

/// A Pixel Shift capture that the CLI and the GUI can work with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeriesInfo {
    pub dir: PathBuf,
    pub files: Vec<PathBuf>,
    /// Detected capture mode (4, 8, 16, 32) or 0 for an unexpected frame count.
    pub mode: u32,
    pub suggested_scale: u32,
}

impl SeriesInfo {
    pub fn new(dir: PathBuf, files: Vec<PathBuf>) -> Self {
        let mode = mode_for_count(files.len());
        Self {
            dir,
            files,
            mode,
            suggested_scale: suggested_scale(mode),
        }
    }

    pub fn frame_count(&self) -> usize {
        self.files.len()
    }

    pub fn name(&self) -> String {
        self.dir
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| self.dir.display().to_string())
    }

    /// Short description used by both front ends, e.g.
    /// "104NZ6_3 - mode 4 - 4 frames - scale 1".
    pub fn label(&self) -> String {
        let mode = if self.mode == 0 {
            "mode unknown".to_string()
        } else {
            format!("mode {}", self.mode)
        };
        format!(
            "{} - {} - {} frames - scale {}",
            self.name(),
            mode,
            self.files.len(),
            self.suggested_scale
        )
    }

    /// True when the frame count is not one of the four Pixel Shift modes.
    pub fn unexpected_mode(&self) -> bool {
        self.mode == 0
    }
}

fn mode_for_count(count: usize) -> u32 {
    match count {
        4 => 4,
        8 => 8,
        16 => 16,
        32 => 32,
        _ => 0,
    }
}

fn suggested_scale(mode: u32) -> u32 {
    match mode {
        16 | 32 => 2,
        _ => 1,
    }
}

/// Build one series out of an explicit list of files.
pub fn series_from_files(files: &[PathBuf]) -> Result<SeriesInfo> {
    if files.is_empty() {
        bail!("no input files");
    }
    let dir = files[0]
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    Ok(SeriesInfo::new(dir, files.to_vec()))
}

/// Resolve CLI/GUI inputs: directories are scanned with rawio::find_series,
/// explicit files are taken as one series in the given order.
pub fn resolve_inputs(inputs: &[PathBuf]) -> Result<Vec<SeriesInfo>> {
    if inputs.is_empty() {
        bail!("no inputs given");
    }
    let mut out = Vec::new();
    let mut files = Vec::new();
    for input in inputs {
        if input.is_dir() {
            out.extend(scan_dir(input)?);
        } else if input.is_file() {
            files.push(input.clone());
        } else {
            bail!("input does not exist: {}", input.display());
        }
    }
    if !files.is_empty() {
        out.push(series_from_files(&files)?);
    }
    Ok(out)
}

/// Series below dir, via the raw I/O work package.
pub fn scan_dir(dir: &Path) -> Result<Vec<SeriesInfo>> {
    let found = guarded("rawio::find_series (WP1)", || crate::rawio::find_series(dir))
        .with_context(|| format!("scanning {}", dir.display()))?;
    Ok(found
        .into_iter()
        .map(|s| SeriesInfo {
            dir: s.dir,
            files: s.files,
            mode: s.mode,
            suggested_scale: s.scale,
        })
        .collect())
}

// ---------------------------------------------------------------- output -----

/// Dimensions, pixel count and size of the uncompressed output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResultInfo {
    pub width: usize,
    pub height: usize,
    pub scale: u32,
    pub white: u16,
}

impl ResultInfo {
    pub fn megapixels(&self) -> f64 {
        (self.width as f64 * self.height as f64) / 1e6
    }

    /// Uncompressed 16-bit RGB size of the stored image data.
    pub fn bytes(&self) -> u64 {
        (self.width as u64)
            .saturating_mul(self.height as u64)
            .saturating_mul(crate::types::CHANNELS as u64)
            .saturating_mul(2)
    }

    pub fn mib(&self) -> f64 {
        self.bytes() as f64 / (1024.0 * 1024.0)
    }
}

pub fn expected_output(width: usize, height: usize, scale: u32) -> ResultInfo {
    let scale = scale.max(1) as usize;
    ResultInfo {
        width: width * scale,
        height: height * scale,
        scale: scale as u32,
        white: 0,
    }
}

/// df -k free space of the volume holding path, in bytes.
pub fn free_space(path: &Path) -> Option<u64> {
    let dir = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or(Path::new("."))
    };
    let output = std::process::Command::new("df")
        .arg("-k")
        .arg(dir)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let line = text.lines().nth(1)?;
    let available_kib: u64 = line.split_whitespace().nth(3)?.parse().ok()?;
    Some(available_kib.saturating_mul(1024))
}

/// Default output path next to the first input:
/// <first-stem>_pixelshift_<scale>x.<ext>.
pub fn default_output_path(first_input: &Path, scale: u32, ext: &str) -> PathBuf {
    let stem = first_input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "pixelshift".to_string());
    let dir = first_input
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    dir.join(format!("{stem}_pixelshift_{scale}x.{ext}"))
}

/// Path with the scale token replaced by 1x:
/// _DSC0522_pixelshift_2x.dng -> _DSC0522_pixelshift_1x.dng.
pub fn sibling_1x_path(path: &Path) -> PathBuf {
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let extension = path.extension().map(|e| e.to_string_lossy().into_owned());
    let new_stem = match stem.rfind("_2x") {
        Some(position) => format!("{}_1x{}", &stem[..position], &stem[position + 3..]),
        None => format!("{stem}_1x"),
    };
    let mut out = path.with_file_name(new_stem);
    if let Some(extension) = extension {
        out.set_extension(extension);
    }
    out
}

pub fn fmt_bytes(bytes: u64) -> String {
    let mib = bytes as f64 / (1024.0 * 1024.0);
    if mib >= 1024.0 {
        format!("{:.2} GiB", mib / 1024.0)
    } else {
        format!("{:.1} MiB", mib)
    }
}

/// One decoded frame as the GUI shows it.
pub fn frame_info(frame: &RawFrame) -> crate::app::model::FrameInfo {
    crate::app::model::FrameInfo {
        path: frame.path.clone(),
        width: frame.width,
        height: frame.height,
        cfa: frame.cfa.pattern_str(),
        black: frame.black,
        white: frame.white,
        wb: [frame.wb[0], frame.wb[1], frame.wb[2]],
        iso: frame.iso,
        shutter: frame.shutter,
        aperture: frame.aperture,
        focal: frame.focal,
        camera: format!("{} {}", frame.make.trim(), frame.model.trim())
            .trim()
            .to_string(),
    }
}

// --------------------------------------------------------------- decode ------

pub fn decode(files: &[PathBuf]) -> Result<Vec<RawFrame>> {
    guarded("rawio::open_series (WP1)", || crate::rawio::open_series(files))
        .with_context(|| format!("decoding {} frames", files.len()))
}

/// Shift per frame; all zero when alignment is disabled.
pub fn estimate_shifts(frames: &[RawFrame], align: bool) -> Result<Vec<Shift>> {
    if !align || frames.len() < 2 {
        return Ok(vec![Shift::default(); frames.len()]);
    }
    let shifts = guarded("shift::estimate_series (WP2)", || {
        crate::shift::estimate_series(frames)
    })
    .context("estimating frame shifts")?;
    if shifts.len() != frames.len() {
        bail!(
            "shift::estimate_series returned {} shifts for {} frames",
            shifts.len(),
            frames.len()
        );
    }
    Ok(shifts)
}

pub fn merge_frames(frames: &[RawFrame], params: &MergeParams) -> Result<MergeResult> {
    // Motion masks are derived from the aligned frames unless the caller
    // supplied them (tests do). Computed here, both the CLI and the GUI
    // paths get them.
    let mut params = params.clone();
    if params.motion.enabled && params.align && params.masks.is_empty() && frames.len() > 1 {
        let shifts = crate::merge::applied_shifts(frames, &params)?;
        params.masks = crate::motion::masks(frames, &shifts, &params.motion);
    }
    guarded("merge::merge (WP2)", || crate::merge::merge(frames, &params)).context("merging frames")
}

/// Apply the lens distortion correction of the current options.
pub fn rectify(img: MergeResult, d: crate::distort::Distortion) -> Result<MergeResult> {
    Ok(crate::distort::rectify(img, d))
}

/// Displacement of the frame corner produced by a correction, in pixels of an
/// image of the given size. Useful to judge a k1 without rendering it.
pub fn corner_shift(width: usize, height: usize, d: crate::distort::Distortion) -> f64 {
    if width == 0 || height == 0 {
        return 0.0;
    }
    let cx = (width as f64 - 1.0) * 0.5;
    let cy = (height as f64 - 1.0) * 0.5;
    let half_diagonal = (cx * cx + cy * cy).sqrt();
    (d.map_radius(1.0) - 1.0).abs() * half_diagonal
}

pub fn downsample_1x(img: &MergeResult) -> Result<MergeResult> {
    guarded("merge::downsample_1x (WP2)", || {
        Ok(crate::merge::downsample_1x(img))
    })
    .context("downsampling to 1x")
}

/// White balance multipliers normalised to green (the preview expects [r, g, b]).
pub fn preview_wb(frame: &RawFrame) -> [f32; 3] {
    let g = frame.wb[1].max(1e-3);
    [frame.wb[0] / g, 1.0, frame.wb[2] / g]
}

pub fn preview(img: &MergeResult, max_dim: u32, wb: [f32; 3]) -> Result<PreviewImage> {
    guarded("merge::preview (WP2)", || {
        Ok(crate::merge::preview(img, max_dim, wb))
    })
    .context("building the preview")
}

pub fn crop_preview(
    img: &MergeResult,
    rect: (usize, usize, usize, usize),
    step: u32,
    wb: [f32; 3],
) -> Result<PreviewImage> {
    guarded("merge::preview_crop_at (WP2)", || {
        Ok(crate::merge::preview_crop_at(img, rect, step, wb))
    })
    .context("building the crop")
}

/// Cheap colour preview of a single raw frame, used by the reference view.
///
/// With `rect = None` the whole frame is rendered at `max_dim` on the long
/// side. With a rect the window is rendered at reduced detail: `step` CFA
/// blocks per output sample, the rectangle aligned outward to whole blocks
/// (2 * step native pixels), matching the merged view's pyramid level. The
/// per-channel black level is subtracted, the white balance is applied and
/// the result goes through the same sRGB encoder as the merged view (99.5
/// percentile white point, sRGB curve). This never touches the merge work
/// package, so the reference view works as soon as decoding works.
pub fn reference_preview(
    frame: &RawFrame,
    rect: Option<(usize, usize, usize, usize)>,
    max_dim: u32,
    wb: [f32; 3],
    step: u32,
) -> PreviewImage {
    let Some((rx, ry, rw, rh)) = rect else {
        return reference_base_preview(frame, max_dim, wb);
    };
    let s = step.max(1) as usize;
    let a = 2 * s;
    let x0 = (rx.min(frame.width.saturating_sub(a)) / a) * a;
    let y0 = (ry.min(frame.height.saturating_sub(a)) / a) * a;
    let x1 = ((rx + rw).min(frame.width).max(x0 + a) + a - 1) / a * a;
    let y1 = ((ry + rh).min(frame.height).max(y0 + a) + a - 1) / a * a;
    let out_w = ((x1 - x0) / a).max(1);
    let out_h = ((y1 - y0) / a).max(1);
    let white = frame.white_span().max(1) as f32;
    let mut buf = vec![0.0f32; out_w * out_h * 3];
    for oy in 0..out_h {
        for ox in 0..out_w {
            let mut acc = [0.0f32; 3];
            let mut n = [0.0f32; 3];
            for by in 0..s {
                for bx in 0..s {
                    let px = x0 + (ox * s + bx) * 2;
                    let py = y0 + (oy * s + by) * 2;
                    if px + 1 >= frame.width || py + 1 >= frame.height {
                        continue;
                    }
                    for (c, v) in block_channels(frame, px, py) {
                        acc[c] += v as f32 / white;
                        n[c] += 1.0;
                    }
                }
            }
            let idx = (oy * out_w + ox) * 3;
            for c in 0..3 {
                let mean = if n[c] > 0.0 { acc[c] / n[c] } else { 0.0 };
                buf[idx + c] = mean * wb[c];
            }
        }
    }
    crate::merge::encode_srgb(&buf, out_w, out_h)
}

/// Whole frame reference preview: every 2x2 CFA block is one pixel, box
/// averaged down to max_dim on the long side.
fn reference_base_preview(frame: &RawFrame, max_dim: u32, wb: [f32; 3]) -> PreviewImage {
    let x0 = 0;
    let y0 = 0;
    let w = frame.width & !1;
    let h = frame.height & !1;
    let (w, h) = (w.max(2), h.max(2));
    let bw = w / 2;
    let bh = h / 2;
    let step = downscale_step(bw, bh, max_dim);
    let out_w = (bw / step).max(1);
    let out_h = (bh / step).max(1);
    let white = frame.white_span().max(1) as f32;
    let mut buf = vec![0.0f32; out_w * out_h * 3];
    for oy in 0..out_h {
        for ox in 0..out_w {
            let mut acc = [0.0f32; 3];
            let mut n = [0.0f32; 3];
            for sy in 0..step {
                for sx in 0..step {
                    let bx = x0 + (ox * step + sx) * 2;
                    let by = y0 + (oy * step + sy) * 2;
                    if bx + 1 >= frame.width || by + 1 >= frame.height {
                        continue;
                    }
                    for (c, v) in block_channels(frame, bx, by) {
                        acc[c] += v as f32 / white;
                        n[c] += 1.0;
                    }
                }
            }
            let idx = (oy * out_w + ox) * 3;
            for c in 0..3 {
                let mean = if n[c] > 0.0 { acc[c] / n[c] } else { 0.0 };
                buf[idx + c] = mean * wb[c];
            }
        }
    }
    crate::merge::encode_srgb(&buf, out_w, out_h)
}

fn block_channels(frame: &RawFrame, x: usize, y: usize) -> [(usize, u16); 3] {
    let mut r = 0u16;
    let mut b = 0u16;
    let mut g_sum = 0u32;
    let mut g_n = 0u32;
    for dy in 0..2 {
        for dx in 0..2 {
            let color = frame.color_at(x + dx, y + dy);
            let v = frame
                .sample(x + dx, y + dy)
                .saturating_sub(frame.black_for(color));
            match color {
                crate::types::Color::R => r = v,
                crate::types::Color::B => b = v,
                crate::types::Color::G => {
                    g_sum += v as u32;
                    g_n += 1;
                }
            }
        }
    }
    [(0, r), (1, (g_sum / g_n.max(1)) as u16), (2, b)]
}

pub(crate) fn downscale_step(w: usize, h: usize, max_dim: u32) -> usize {
    let max = max_dim.max(1) as usize;
    let mut step = 1;
    while w / step > max || h / step > max {
        step += 1;
    }
    step
}

pub(crate) fn tone(linear: f32) -> u8 {
    let v = linear.clamp(0.0, 1.0).powf(1.0 / 2.2);
    (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

// ---------------------------------------------------------------- write ------

/// Metadata for the writers, derived from the reference frame.
pub fn dng_meta(frame: &RawFrame, software: &str) -> DngMeta {
    let neutral = [1.0 / frame.wb[0].max(1e-3), 1.0, 1.0 / frame.wb[2].max(1e-3)];
    let peak = neutral.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
    let mut color_matrix = [[0.0f32; 3]; 3];
    for (row, out) in color_matrix.iter_mut().enumerate() {
        for (col, value) in out.iter_mut().enumerate() {
            *value = frame.cam_xyz[row][col];
        }
    }
    DngMeta {
        make: frame.make.clone(),
        model: frame.model.clone(),
        timestamp: frame.timestamp.clone(),
        software: software.to_string(),
        as_shot_neutral: [neutral[0] / peak, neutral[1] / peak, neutral[2] / peak],
        color_matrix,
        calib_illuminant: 21,
        baseline_exposure: 0.0,
        iso: frame.iso,
        shutter: frame.shutter,
        aperture: frame.aperture,
        focal: frame.focal,
        orientation: 1,
        wb_applied: false,
        lens_make: frame.lens_make.clone(),
        lens_model: frame.lens_model.clone(),
        focal_35mm: frame.focal_35mm,
        max_aperture: frame.max_aperture,
        lens_min_focal: frame.lens_min_focal,
        lens_max_focal: frame.lens_max_focal,
        lens_min_fnum: frame.lens_min_fnum,
        lens_max_fnum: frame.lens_max_fnum,
        lens_serial: frame.lens_serial.clone(),
        body_serial: frame.body_serial.clone(),
        // The vendor maker note is what a raw editor identifies a first party
        // lens by; it is copied verbatim from the reference frame.
        makernote: crate::rawio::makernote(&frame.path).unwrap_or_default(),
    }
}

/// Write img and return the size of the resulting file.
///
/// `preview_max = 0` asks the DNG writer for its default preview (1024 px on
/// the long side), which is what viewers and importers actually use.
pub fn write_output(
    path: &Path,
    img: &MergeResult,
    meta: &DngMeta,
    format: OutFormat,
    compression: Compression,
    flavor: DngFlavor,
    tiff_render: bool,
) -> Result<u64> {
    let compress = format.use_compression(compression);
    let opts = WriteOpts {
        compress,
        preview_max: 0,
    };
    guarded("output writer (WP1)", || match format {
        OutFormat::Dng => crate::write::write_dng_with_flavor(path, img, meta, &opts, flavor),
        OutFormat::Tiff if tiff_render => {
            crate::write::write_tiff16_rendered(path, img, meta, compress)
        }
        OutFormat::Tiff => crate::write::write_tiff16(path, img, meta, compress),
    })
    .with_context(|| format!("writing {}", path.display()))?;
    Ok(std::fs::metadata(path).map(|m| m.len()).unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Cfa;

    /// Uniform frame whose channels are a neutral grey after the white
    /// balance: native value = black + K / wb[c], so the WB-applied signal is
    /// K / span in every channel. A black-subtracting preview must render it
    /// channel-neutral; scaling the raw pedestal by wb[c] instead (the old
    /// behaviour) tints it.
    fn neutral_frame(wb: [f32; 3]) -> RawFrame {
        let cfa = Cfa::from_pattern_str("RGGB").expect("RGGB");
        let (black, white) = (1008u16, 16383u16);
        let span = (white - black) as f32;
        let k = 0.25 * span;
        let mut data = vec![0u16; 64 * 64];
        for y in 0..64 {
            for x in 0..64 {
                let c = cfa.0[y & 1][x & 1];
                let value = black as f32 + k / wb[c.idx()].max(1e-3);
                data[y * 64 + x] = value.min(white as f32) as u16;
            }
        }
        RawFrame {
            path: std::path::PathBuf::new(),
            width: 64,
            height: 64,
            data,
            cfa,
            black,
            cblack: [0; 4],
            white,
            wb: [wb[0], wb[1], wb[2], wb[1]],
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
    fn reference_preview_renders_wb_matched_grey_neutral() {
        let wb = [1.244f32, 1.0, 2.010];
        let frame = neutral_frame(wb);
        let image = reference_preview(&frame, None, 2048, wb, 1);
        assert!(image.width > 0 && image.height > 0);
        for p in image.rgb.chunks_exact(3) {
            assert_eq!(p[0], p[1], "red must match green");
            assert_eq!(p[1], p[2], "green must match blue");
        }
        // The patch sits at a quarter of the span, well above black.
        assert!(image.rgb.iter().all(|&v| v > 16));
    }

    #[test]
    fn reference_preview_subtracts_black() {
        // Pedestal-only frame: with the black level subtracted every channel
        // is zero, so the preview must be pure black regardless of wb.
        let mut frame = neutral_frame([1.0, 1.0, 1.0]);
        for v in frame.data.iter_mut() {
            *v = frame.black;
        }
        let image = reference_preview(&frame, None, 2048, [1.244, 1.0, 2.010], 1);
        assert!(image.rgb.iter().all(|&v| v == 0));
    }
}
