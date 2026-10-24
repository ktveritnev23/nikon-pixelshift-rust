//! Minimal linear DNG writer.
//!
//! Owned by work package WP1. Layout (see docs/ARCHITECTURE.md, "Output format
//! decisions"): IFD0 holds the full-resolution image with the camera metadata
//! and the DNG tags, IFD1 (reached through the next-IFD pointer) holds an 8-bit
//! sRGB preview. Both are written strip by strip / block by block, so a 96 MP
//! frame never has to live in memory as a whole.
//!
//! The main image is a LinearRaw (PhotometricInterpretation 34892) by default.
//! [`DngFlavor::Rgb`] writes it as plain 16-bit RGB (PhotometricInterpretation
//! 2) for readers that reject a raw main image; LibRaw refuses that, which is
//! exactly why linearraw stays the default.
//!
//! `WriteOpts::compress` selects lossless JPEG (Compression = 7, the same
//! process the DNG specification and Nikon's own NEFs use), so the compressed
//! file stays readable by LibRaw; every strip is an independent JPEG stream.

use crate::types::{DngMeta, MergeResult, WriteOpts};
use crate::write::color::{camera_to_srgb, encode_srgb_u8, mul3};
use crate::write::jpeg;
use crate::write::ljpeg;
use crate::write::tiff::{self, Ifd, Value};
use anyhow::{bail, Context, Result};
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

/// Rows per strip: small enough to keep the write buffer around 1 MB for a
/// 96 MP image, large enough to keep the strip tables short.
pub const ROWS_PER_STRIP: usize = 16;

// EXIF tags. Raw editors (Capture One in particular) look for the standard
// Exif IFD first and only then read the lens block and the vendor maker note,
// so these live in a directory of their own linked from IFD0.
const TAG_EXIF_IFD: u16 = 0x8769;
const TAG_EXIF_VERSION: u16 = 0x9000;
const TAG_DATETIME_DIGITIZED: u16 = 0x9004;
const TAG_MAX_APERTURE_VALUE: u16 = 0x9205;
const TAG_MAKERNOTE: u16 = 0x927C;
const TAG_BODY_SERIAL: u16 = 0xA431;
const TAG_LENS_SPECIFICATION: u16 = 0xA432;
const TAG_LENS_MAKE: u16 = 0xA433;
const TAG_LENS_MODEL: u16 = 0xA434;
const TAG_LENS_SERIAL: u16 = 0xA435;
const TAG_FOCAL_LENGTH_IN_35MM: u16 = 0xA405;

/// ExifVersion 2.32, i.e. "0232" as four UNDEFINED bytes.
const EXIF_VERSION: [u8; 4] = *b"0232";

/// Preview size used when `WriteOpts::preview_max == 0`. 1024 px on the long
/// side is what viewers, Capture One's importer and the Finder thumbnails need
/// to show the image without decoding the raw data.
pub const DEFAULT_PREVIEW_MAX: u32 = 1024;

/// Long side of the JPEG preview. Readers that do not want to decode the raw
/// image (RapidRAW, Finder, preview extractors in general) use this one.
pub const JPEG_PREVIEW_MAX: u32 = 1600;

/// DNGVersion written to the main directory.
pub const DNG_VERSION: [u8; 4] = [1, 4, 0, 0];
/// DNGBackwardVersion written to the main directory.
pub const DNG_BACKWARD_VERSION: [u8; 4] = [1, 1, 0, 0];

/// Photometric interpretation of the full resolution image in IFD0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DngFlavor {
    /// LinearRaw (34892): what every raw converter expects from a linear DNG.
    #[default]
    LinearRaw,
    /// Plain 16-bit RGB (2): for readers that reject a raw-looking main image.
    Rgb,
    /// 16-bit RGGB mosaic (32803 CFA): readers that only accept a Bayer raw
    /// (rawler based applications, ImageIO, Capture One) demosaic it themselves.
    Cfa,
}

impl DngFlavor {
    pub const ALL: [DngFlavor; 3] = [DngFlavor::LinearRaw, DngFlavor::Rgb, DngFlavor::Cfa];

    pub fn photometric(self) -> u16 {
        match self {
            DngFlavor::LinearRaw => tiff::PHOTOMETRIC_LINEAR_RAW,
            DngFlavor::Rgb => tiff::PHOTOMETRIC_RGB,
            DngFlavor::Cfa => tiff::PHOTOMETRIC_CFA,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            DngFlavor::LinearRaw => "linearraw",
            DngFlavor::Rgb => "rgb",
            DngFlavor::Cfa => "cfa",
        }
    }

    fn is_cfa(self) -> bool {
        matches!(self, DngFlavor::Cfa)
    }
}

/// Write img as a linear DNG (LinearRaw main image).
pub fn write_dng(path: &Path, img: &MergeResult, meta: &DngMeta, opts: &WriteOpts) -> Result<()> {
    write_dng_with_flavor(path, img, meta, opts, DngFlavor::LinearRaw)
}

/// Write img as a DNG with the requested main image flavour.
pub fn write_dng_with_flavor(
    path: &Path,
    img: &MergeResult,
    meta: &DngMeta,
    opts: &WriteOpts,
    flavor: DngFlavor,
) -> Result<()> {
    validate(img)?;
    let file = File::create(path).with_context(|| format!("cannot create {}", path.display()))?;
    let mut writer = BufWriter::with_capacity(1 << 20, file);
    write_inner(&mut writer, img, meta, opts, flavor)?;
    writer
        .flush()
        .with_context(|| format!("cannot write {}", path.display()))?;
    Ok(())
}

fn validate(img: &MergeResult) -> Result<()> {
    if img.width == 0 || img.height == 0 {
        bail!("cannot write an empty image ({}x{})", img.width, img.height);
    }
    let expected = img.width * img.height;
    for (c, plane) in img.planes.iter().enumerate() {
        if plane.len() != expected {
            bail!(
                "plane {} has {} samples, expected {} for {}x{}",
                c,
                plane.len(),
                expected,
                img.width,
                img.height
            );
        }
    }
    Ok(())
}

fn write_inner<W: Write + Seek>(
    w: &mut W,
    img: &MergeResult,
    meta: &DngMeta,
    opts: &WriteOpts,
    flavor: DngFlavor,
) -> Result<()> {
    let width = img.width;
    let height = img.height;
    // Lossless JPEG is written as one strip spanning the image: LibRaw's
    // DNG loader cannot walk a multi-strip lossless JPEG layout.
    let rows_per_strip = if opts.compress { height } else { ROWS_PER_STRIP };
    let strips = height.div_ceil(rows_per_strip);
    let (preview_w, preview_h, preview) = build_preview(img, meta, opts.preview_max);
    let (jpeg_w, jpeg_h, jpeg_rgb) = build_preview(img, meta, JPEG_PREVIEW_MAX);
    let jpeg = jpeg::encode_rgb(&jpeg_rgb, jpeg_w, jpeg_h);

    // First pass: sizes of the directories (they do not depend on the values
    // that are only known once the layout is fixed).
    let mut main_ifd = Ifd::new();
    fill_main_ifd(
        &mut main_ifd,
        img,
        meta,
        opts,
        flavor,
        strips,
        rows_per_strip,
        0,
    );
    let main_size = main_ifd.size();
    // The Exif IFD sits right after IFD0; it is reached through tag 0x8769 and
    // is deliberately not part of the next-IFD chain.
    let exif_ifd_off = tiff::align2(8 + main_size);
    let mut exif_ifd = Ifd::new();
    fill_exif_ifd(&mut exif_ifd, meta);
    let exif_ifd_size = exif_ifd.size();
    let preview_ifd_off = tiff::align2(exif_ifd_off + exif_ifd_size);

    let mut preview_ifd = Ifd::new();
    fill_preview_ifd(&mut preview_ifd, meta, preview_w, preview_h, 0, preview.len());
    let preview_ifd_size = preview_ifd.size();
    let jpeg_ifd_off = tiff::align2(preview_ifd_off + preview_ifd_size);

    let mut jpeg_ifd = Ifd::new();
    fill_jpeg_ifd(&mut jpeg_ifd, meta, jpeg_w, jpeg_h, 0, jpeg.len());
    let jpeg_ifd_size = jpeg_ifd.size();
    let preview_off = tiff::align2(jpeg_ifd_off + jpeg_ifd_size);
    let jpeg_off = tiff::align2(preview_off + preview.len() as u64);
    let strips_off = tiff::align2(jpeg_off + jpeg.len() as u64);

    if strips_off + (height * width * 6) as u64 >= u32::MAX as u64 {
        bail!("output would exceed the 4 GiB limit of classic TIFF");
    }

    // Second pass with the real offsets.
    let mut preview_ifd = Ifd::new();
    fill_preview_ifd(
        &mut preview_ifd,
        meta,
        preview_w,
        preview_h,
        preview_off,
        preview.len(),
    );
    if preview_ifd.size() != preview_ifd_size {
        bail!("internal error: preview IFD size changed between passes");
    }
    let mut jpeg_ifd = Ifd::new();
    fill_jpeg_ifd(&mut jpeg_ifd, meta, jpeg_w, jpeg_h, jpeg_off, jpeg.len());
    if jpeg_ifd.size() != jpeg_ifd_size {
        bail!("internal error: JPEG IFD size changed between passes");
    }
    let mut main_ifd = Ifd::new();
    fill_main_ifd(
        &mut main_ifd,
        img,
        meta,
        opts,
        flavor,
        strips,
        rows_per_strip,
        exif_ifd_off as u32,
    );
    if main_ifd.size() != main_size {
        bail!("internal error: main IFD size changed between passes");
    }
    let (preview_ifd_bytes, _) = preview_ifd.encode_with_next(preview_ifd_off, jpeg_ifd_off as u32);
    let (jpeg_ifd_bytes, _) = jpeg_ifd.encode(jpeg_ifd_off);
    let (main_bytes, main_positions) = main_ifd.encode_with_next(8, preview_ifd_off as u32);
    let (exif_ifd_bytes, _) = exif_ifd.encode(exif_ifd_off);

    w.write_all(&tiff::HEADER)?;
    w.write_all(&main_bytes)?;
    w.seek(SeekFrom::Start(exif_ifd_off))?;
    w.write_all(&exif_ifd_bytes)?;
    w.seek(SeekFrom::Start(preview_ifd_off))?;
    w.write_all(&preview_ifd_bytes)?;
    w.seek(SeekFrom::Start(jpeg_ifd_off))?;
    w.write_all(&jpeg_ifd_bytes)?;
    w.seek(SeekFrom::Start(preview_off))?;
    w.write_all(&preview)?;
    w.seek(SeekFrom::Start(jpeg_off))?;
    w.write_all(&jpeg)?;
    if w.stream_position()? < strips_off {
        w.write_all(&[0u8])?;
    }
    if w.stream_position()? != strips_off {
        bail!("internal error: wrong strip start offset");
    }

    let mut offsets: Vec<u32> = Vec::with_capacity(strips);
    let mut counts: Vec<u32> = Vec::with_capacity(strips);
    let mut raw: Vec<u8> = Vec::new();
    let mut written = 0u64;

    if opts.compress {
        // One strip, streamed: pass one collects the difference histogram,
        // pass two writes the entropy data row by row.
        let components = if flavor.is_cfa() { 1 } else { 3 };
        let mut row = vec![0u16; width * components];
        let mut encoder = ljpeg::StripEncoder::new(width, components);
        for y in 0..height {
            pack_row(&img.planes, width, y, flavor.is_cfa(), &mut row);
            encoder.add_row(&row);
        }
        let bytes = encoder.write_to(&mut *w, |y, row| {
            pack_row(&img.planes, width, y, flavor.is_cfa(), row)
        })?;
        offsets.push(strips_off as u32);
        counts.push(bytes as u32);
    } else {
        for strip in 0..strips {
            let y0 = strip * ROWS_PER_STRIP;
            let y1 = (y0 + ROWS_PER_STRIP).min(height);
            if flavor.is_cfa() {
                tiff::mosaic_rows(&img.planes, width, y0, y1, &mut raw);
            } else {
                tiff::interleave_rows(&img.planes, width, y0, y1, &mut raw);
            }
            offsets.push((strips_off + written) as u32);
            w.write_all(&raw)?;
            counts.push(raw.len() as u32);
            written += raw.len() as u64;
        }
    }

    patch_u32_values(w, main_positions.get(&tiff::TAG_STRIP_OFFSETS), &offsets)?;
    patch_u32_values(w, main_positions.get(&tiff::TAG_STRIP_BYTE_COUNTS), &counts)?;
    Ok(())
}

/// Pack one row of the merged image into interleaved u16 samples: RGB triples
/// for the colour flavours, the Bayer mosaic for the CFA flavour.
fn pack_row(
    planes: &[Vec<u16>; 3],
    width: usize,
    y: usize,
    cfa: bool,
    out: &mut [u16],
) {
    let base = y * width;
    if cfa {
        // RGGB: the CFA position decides which plane each sample comes from.
        for x in 0..width {
            let plane = match (x & 1, y & 1) {
                (0, 0) => 0,
                (1, 1) => 2,
                _ => 1,
            };
            out[x] = planes[plane][base + x];
        }
    } else {
        for x in 0..width {
            let i = base + x;
            out[x * 3] = planes[0][i];
            out[x * 3 + 1] = planes[1][i];
            out[x * 3 + 2] = planes[2][i];
        }
    }
}

fn patch_u32_values<W: Write + Seek>(w: &mut W, pos: Option<&u64>, values: &[u32]) -> Result<()> {
    let Some(pos) = pos else {
        bail!("internal error: strip table tag not found");
    };
    w.seek(SeekFrom::Start(*pos))?;
    for value in values {
        w.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

/// IFD1: reduced resolution preview, 8 bit RGB, one strip.
fn fill_preview_ifd(
    ifd: &mut Ifd,
    meta: &DngMeta,
    width: usize,
    height: usize,
    strip_offset: u64,
    strip_len: usize,
) {
    ifd.set_long(tiff::TAG_NEW_SUBFILE_TYPE, 1); // reduced resolution
    ifd.set_long(tiff::TAG_IMAGE_WIDTH, width as u32);
    ifd.set_long(tiff::TAG_IMAGE_LENGTH, height as u32);
    ifd.set(tiff::TAG_BITS_PER_SAMPLE, Value::Shorts(vec![8, 8, 8]));
    ifd.set_short(tiff::TAG_COMPRESSION, tiff::COMPRESSION_NONE);
    ifd.set_short(tiff::TAG_PHOTOMETRIC, tiff::PHOTOMETRIC_RGB);
    ifd.set_short(tiff::TAG_ORIENTATION, meta.orientation.max(1));
    ifd.set_short(tiff::TAG_SAMPLES_PER_PIXEL, 3);
    ifd.set(
        tiff::TAG_STRIP_OFFSETS,
        Value::Longs(vec![strip_offset as u32]),
    );
    ifd.set_long(tiff::TAG_ROWS_PER_STRIP, height as u32);
    ifd.set(
        tiff::TAG_STRIP_BYTE_COUNTS,
        Value::Longs(vec![strip_len as u32]),
    );
    ifd.set_short(tiff::TAG_PLANAR_CONFIG, 1);
    preview_identity(ifd, meta);
}

/// IFD2: JPEG preview (Compression 7, YCbCr) in one strip.
fn fill_jpeg_ifd(
    ifd: &mut Ifd,
    meta: &DngMeta,
    width: usize,
    height: usize,
    strip_offset: u64,
    strip_len: usize,
) {
    ifd.set_long(tiff::TAG_NEW_SUBFILE_TYPE, 1);
    ifd.set_long(tiff::TAG_IMAGE_WIDTH, width as u32);
    ifd.set_long(tiff::TAG_IMAGE_LENGTH, height as u32);
    ifd.set(tiff::TAG_BITS_PER_SAMPLE, Value::Shorts(vec![8, 8, 8]));
    ifd.set_short(tiff::TAG_COMPRESSION, tiff::COMPRESSION_JPEG);
    ifd.set_short(tiff::TAG_PHOTOMETRIC, tiff::PHOTOMETRIC_YCBCR);
    ifd.set_short(tiff::TAG_ORIENTATION, meta.orientation.max(1));
    ifd.set_short(tiff::TAG_SAMPLES_PER_PIXEL, 3);
    ifd.set(
        tiff::TAG_STRIP_OFFSETS,
        Value::Longs(vec![strip_offset as u32]),
    );
    ifd.set_long(tiff::TAG_ROWS_PER_STRIP, height as u32);
    ifd.set(
        tiff::TAG_STRIP_BYTE_COUNTS,
        Value::Longs(vec![strip_len as u32]),
    );
    ifd.set_short(tiff::TAG_PLANAR_CONFIG, 1);
    ifd.set(tiff::TAG_YCBCR_SUB_SAMPLING, Value::Shorts(vec![1, 1]));
    ifd.set_short(tiff::TAG_YCBCR_POSITIONING, 1);
    preview_identity(ifd, meta);
}

/// IFD0: the full resolution image of the selected flavour plus every DNG tag.
fn fill_main_ifd(
    ifd: &mut Ifd,
    img: &MergeResult,
    meta: &DngMeta,
    opts: &WriteOpts,
    flavor: DngFlavor,
    strips: usize,
    rows_per_strip: usize,
    exif_ifd_off: u32,
) {
    ifd.set_long(tiff::TAG_NEW_SUBFILE_TYPE, 0);
    ifd.set_long(tiff::TAG_IMAGE_WIDTH, img.width as u32);
    ifd.set_long(tiff::TAG_IMAGE_LENGTH, img.height as u32);
    let samples = if flavor.is_cfa() { 1 } else { 3 };
    ifd.set(
        tiff::TAG_BITS_PER_SAMPLE,
        Value::Shorts(vec![16; samples]),
    );
    ifd.set_short(
        tiff::TAG_COMPRESSION,
        if opts.compress {
            tiff::COMPRESSION_JPEG
        } else {
            tiff::COMPRESSION_NONE
        },
    );
    ifd.set_short(tiff::TAG_PHOTOMETRIC, flavor.photometric());
    ifd.set_short(tiff::TAG_ORIENTATION, meta.orientation.max(1));
    ifd.set_short(tiff::TAG_SAMPLES_PER_PIXEL, samples as u16);
    ifd.set(tiff::TAG_STRIP_OFFSETS, Value::Longs(vec![0; strips]));
    ifd.set_long(tiff::TAG_ROWS_PER_STRIP, rows_per_strip as u32);
    ifd.set(tiff::TAG_STRIP_BYTE_COUNTS, Value::Longs(vec![0; strips]));
    ifd.set(tiff::TAG_SAMPLE_FORMAT, Value::Shorts(vec![1; samples]));
    if flavor.is_cfa() {
        ifd.set(tiff::TAG_CFA_REPEAT_PATTERN_DIM, Value::Shorts(vec![2, 2]));
        ifd.set(tiff::TAG_CFA_PATTERN, Value::Bytes(tiff::CFA_RGGB.to_vec()));
        ifd.set(tiff::TAG_CFA_PLANE_COLOR, Value::Bytes(vec![0, 1, 2]));
        ifd.set_short(tiff::TAG_CFA_LAYOUT, 1);
    } else {
        ifd.set_short(tiff::TAG_PLANAR_CONFIG, 1);
    }

    camera_identity(ifd, meta);
    dng_color(ifd, meta);
    // The DNG specification puts count = SamplesPerPixel on both tags, and
    // rawler asserts exactly that: a single BlackLevel next to
    // SamplesPerPixel = 3 is what made RapidRAW refuse the file.
    ifd.set(
        tiff::TAG_BLACK_LEVEL,
        Value::Rationals(vec![(0, 1); samples]),
    );
    ifd.set(
        tiff::TAG_WHITE_LEVEL,
        Value::Longs(vec![img.white as u32; samples]),
    );
    ifd.set(tiff::TAG_DEFAULT_CROP_ORIGIN, Value::Longs(vec![0, 0]));
    ifd.set(
        tiff::TAG_DEFAULT_CROP_SIZE,
        Value::Longs(vec![img.width as u32, img.height as u32]),
    );
    ifd.set(
        tiff::TAG_ACTIVE_AREA,
        Value::Longs(vec![0, 0, img.height as u32, img.width as u32]),
    );
    exif_pointer(ifd, exif_ifd_off);
}

/// Link IFD0 to the Exif IFD. Zero means "not built yet", used by the sizing
/// pass; the value is always four bytes, so the size never depends on it.
fn exif_pointer(ifd: &mut Ifd, exif_ifd_off: u32) {
    ifd.set_long(TAG_EXIF_IFD, exif_ifd_off);
}

/// Tags that identify the camera and the file in the main directory.
fn camera_identity(ifd: &mut Ifd, meta: &DngMeta) {
    ifd.set(tiff::TAG_MAKE, Value::Ascii(meta.make.clone()));
    ifd.set(tiff::TAG_MODEL, Value::Ascii(meta.model.clone()));
    ifd.set(tiff::TAG_SOFTWARE, Value::Ascii(software(meta)));
    ifd.set(
        tiff::TAG_DATETIME,
        Value::Ascii(normalise_datetime(&meta.timestamp)),
    );
    ifd.set(tiff::TAG_DNG_VERSION, Value::Bytes(DNG_VERSION.to_vec()));
    ifd.set(
        tiff::TAG_DNG_BACKWARD_VERSION,
        Value::Bytes(DNG_BACKWARD_VERSION.to_vec()),
    );
    ifd.set(
        tiff::TAG_UNIQUE_CAMERA_MODEL,
        Value::Ascii(unique_camera_model(&meta.make, &meta.model)),
    );
    resolution(ifd);
}

/// Camera identity without the DNG tags: a preview is not a DNG main image.
fn preview_identity(ifd: &mut Ifd, meta: &DngMeta) {
    ifd.set(tiff::TAG_MAKE, Value::Ascii(meta.make.clone()));
    ifd.set(tiff::TAG_MODEL, Value::Ascii(meta.model.clone()));
    ifd.set(tiff::TAG_SOFTWARE, Value::Ascii(software(meta)));
    ifd.set(
        tiff::TAG_DATETIME,
        Value::Ascii(normalise_datetime(&meta.timestamp)),
    );
    resolution(ifd);
}

/// "NIKON CORPORATION" + "NIKON Z6_3" -> "NIKON Z6_3".
///
/// The camera stores the vendor on the make line and repeats it in the model
/// line; UniqueCameraModel is a single identifier, so the duplication is
/// dropped. Capture One writes the same value for this camera.
fn unique_camera_model(make: &str, model: &str) -> String {
    let vendor = make.split_whitespace().next().unwrap_or(make).trim();
    let model = model.trim();
    let repeated =
        model.len() >= vendor.len() && model[..vendor.len()].eq_ignore_ascii_case(vendor);
    let rest = if repeated {
        model[vendor.len()..].trim()
    } else {
        model
    };
    if rest.is_empty() {
        vendor.to_string()
    } else {
        format!("{vendor} {rest}")
    }
}

fn resolution(ifd: &mut Ifd) {
    ifd.set(tiff::TAG_X_RESOLUTION, Value::Rationals(vec![(300, 1)]));
    ifd.set(tiff::TAG_Y_RESOLUTION, Value::Rationals(vec![(300, 1)]));
    ifd.set_short(tiff::TAG_RESOLUTION_UNIT, 2);
}

fn dng_color(ifd: &mut Ifd, meta: &DngMeta) {
    let matrix: Vec<(i32, i32)> = meta
        .color_matrix
        .iter()
        .flatten()
        .map(|v| ((v * 10000.0).round() as i32, 10000))
        .collect();
    ifd.set(tiff::TAG_COLOR_MATRIX1, Value::Srationals(matrix));
    ifd.set_short(tiff::TAG_CALIBRATION_ILLUMINANT1, meta.calib_illuminant);
    let neutral = if meta.wb_applied {
        [1.0, 1.0, 1.0]
    } else {
        meta.as_shot_neutral
    };
    ifd.set(
        tiff::TAG_AS_SHOT_NEUTRAL,
        Value::Rationals(
            neutral
                .iter()
                .map(|v| ((v.max(0.0) * 1_000_000.0).round() as u32, 1_000_000))
                .collect(),
        ),
    );
    ifd.set(
        tiff::TAG_BASELINE_EXPOSURE,
        Value::Srationals(vec![(
            (meta.baseline_exposure * 1000.0).round() as i32,
            1000,
        )]),
    );
}

/// Exif IFD: the standard capture values plus the lens block and the maker
/// note of the source file. IFD0 keeps only the tags that belong there.
fn fill_exif_ifd(ifd: &mut Ifd, meta: &DngMeta) {
    ifd.set(TAG_EXIF_VERSION, Value::Undefined(EXIF_VERSION.to_vec()));
    if !meta.timestamp.is_empty() {
        let dt = normalise_datetime(&meta.timestamp);
        ifd.set(tiff::TAG_DATETIME_ORIGINAL, Value::Ascii(dt.clone()));
        ifd.set(TAG_DATETIME_DIGITIZED, Value::Ascii(dt));
    }
    if meta.iso > 0.0 {
        ifd.set_short(tiff::TAG_ISO, meta.iso.round().clamp(1.0, 65535.0) as u16);
    }
    if meta.shutter > 0.0 {
        let (n, d) = if meta.shutter < 1.0 {
            (1u32, (1.0 / meta.shutter).round().max(1.0) as u32)
        } else {
            (meta.shutter.round().max(1.0) as u32, 1)
        };
        ifd.set(tiff::TAG_EXPOSURE_TIME, Value::Rationals(vec![(n, d)]));
    }
    if meta.aperture > 0.0 {
        ifd.set(
            tiff::TAG_FNUMBER,
            Value::Rationals(vec![((meta.aperture * 10.0).round() as u32, 10)]),
        );
    }
    if meta.focal > 0.0 {
        ifd.set(
            tiff::TAG_FOCAL_LENGTH,
            Value::Rationals(vec![((meta.focal * 100.0).round() as u32, 100)]),
        );
    }
    lens_basics(ifd, meta);
    if !meta.makernote.is_empty() {
        ifd.set(TAG_MAKERNOTE, Value::Undefined(meta.makernote.clone()));
    }
}

/// Lens identification plus the EXIF values derived from it.
///
/// Only written when the file actually carries them; a made up lens string is
/// worse than none because the editor would apply a wrong profile.
fn lens_basics(ifd: &mut Ifd, meta: &DngMeta) {
    let lens_make = meta.lens_make.trim();
    if !lens_make.is_empty() {
        ifd.set(TAG_LENS_MAKE, Value::Ascii(lens_make.to_string()));
    }
    let lens_model = meta.lens_model.trim();
    if !lens_model.is_empty() {
        ifd.set(TAG_LENS_MODEL, Value::Ascii(lens_model.to_string()));
    }
    if meta.focal_35mm > 0 {
        ifd.set_short(TAG_FOCAL_LENGTH_IN_35MM, meta.focal_35mm);
    }
    let max_ap = if meta.max_aperture > 0.0 {
        meta.max_aperture
    } else {
        meta.aperture
    };
    if max_ap > 0.0 {
        // MaxApertureValue is the widest aperture of the lens at the current
        // focal length, in APEX: A = 2 log2(N).
        let apex = 2.0 * (max_ap as f64).log2();
        ifd.set(
            TAG_MAX_APERTURE_VALUE,
            Value::Rationals(vec![((apex * 1000.0).round().max(0.0) as u32, 1000)]),
        );
    }
    // LensSpecification is (min focal, max focal, f number at min, f number at
    // max). A zoom written as two copies of the current focal length is what
    // made Capture One fall back to an empty profile on a 24-120 mm lens: the
    // profile key includes the lens range, so it has to come from the file.
    let min_focal = if meta.lens_min_focal > 0.0 {
        meta.lens_min_focal
    } else {
        meta.focal
    };
    let max_focal = if meta.lens_max_focal > 0.0 {
        meta.lens_max_focal.max(min_focal)
    } else {
        min_focal
    };
    // Unlike MaxApertureValue above, this must not fall back to the aperture
    // the picture was taken with: that is the shooting aperture, not the widest
    // one, and a wrong specification is worse than no tag at all.
    let min_fnum = if meta.lens_min_fnum > 0.0 {
        meta.lens_min_fnum
    } else {
        meta.max_aperture
    };
    let max_fnum = if meta.lens_max_fnum > 0.0 {
        meta.lens_max_fnum.max(min_fnum)
    } else {
        min_fnum
    };
    if min_focal > 0.0 && min_fnum > 0.0 {
        let focal = |v: f32| ((v * 100.0).round().max(1.0) as u32, 100);
        let fnum = |v: f32| ((v * 100.0).round().max(1.0) as u32, 100);
        ifd.set(
            TAG_LENS_SPECIFICATION,
            Value::Rationals(vec![
                focal(min_focal),
                focal(max_focal),
                fnum(min_fnum),
                fnum(max_fnum),
            ]),
        );
    }
    let lens_serial = meta.lens_serial.trim();
    if !lens_serial.is_empty() {
        ifd.set(TAG_LENS_SERIAL, Value::Ascii(lens_serial.to_string()));
    }
    let body_serial = meta.body_serial.trim();
    if !body_serial.is_empty() {
        ifd.set(TAG_BODY_SERIAL, Value::Ascii(body_serial.to_string()));
    }
}

fn software(meta: &DngMeta) -> String {
    if meta.software.is_empty() {
        crate::version()
    } else {
        meta.software.clone()
    }
}

/// EXIF wants "YYYY:MM:DD HH:MM:SS"; anything else is passed through so that a
/// caller supplied timestamp is not silently mangled.
fn normalise_datetime(raw: &str) -> String {
    if raw.len() >= 19 && raw.as_bytes()[4] == b'-' {
        let mut s = raw[..19].to_string();
        s.replace_range(4..5, ":");
        s.replace_range(7..8, ":");
        s
    } else if raw.is_empty() {
        "1970:01:01 00:00:00".to_string()
    } else {
        raw.to_string()
    }
}

/// Box-filter the merged planes into an 8 bit sRGB preview.
fn build_preview(img: &MergeResult, meta: &DngMeta, max_side: u32) -> (usize, usize, Vec<u8>) {
    let target = if max_side == 0 {
        DEFAULT_PREVIEW_MAX as usize
    } else {
        max_side as usize
    };
    let longest = img.width.max(img.height);
    let step = longest.div_ceil(target.max(1)).max(1);
    let out_w = (img.width / step).max(1);
    let out_h = (img.height / step).max(1);

    let cam_to_srgb = camera_to_srgb(&meta.color_matrix);
    let neutral = if meta.wb_applied {
        [1.0f32, 1.0, 1.0]
    } else {
        meta.as_shot_neutral
    };
    let scale = 1.0 / img.white.max(1) as f32;

    let mut rgb = Vec::with_capacity(out_w * out_h * 3);
    for py in 0..out_h {
        let y0 = py * step;
        let y1 = (y0 + step).min(img.height);
        for px in 0..out_w {
            let x0 = px * step;
            let x1 = (x0 + step).min(img.width);
            let mut acc = [0.0f32; 3];
            let mut n = 0.0f32;
            for y in y0..y1 {
                let row = y * img.width;
                for x in x0..x1 {
                    for c in 0..3 {
                        acc[c] += img.planes[c][row + x] as f32;
                    }
                    n += 1.0;
                }
            }
            if n > 0.0 {
                for c in 0..3 {
                    acc[c] /= n;
                }
            }
            let mut cam = [0.0f32; 3];
            for c in 0..3 {
                cam[c] = acc[c] * scale / neutral[c].max(1e-6);
            }
            let srgb = mul3(&cam_to_srgb, &cam);
            for value in srgb {
                rgb.push(encode_srgb_u8(value));
            }
        }
    }
    (out_w, out_h, rgb)
}
