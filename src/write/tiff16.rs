//! Plain 16-bit RGB TIFF writer (universal fallback output).
//!
//! Owned by work package WP1. Baseline little-endian TIFF, one directory,
//! strips; readable by every TIFF consumer (Pillow, ImageMagick, libtiff)
//! without any raw/DNG support. With `compress = true` the strips are stored
//! as AdobeDeflate (Compression = 8) with the horizontal-differencing
//! predictor (Predictor = 2), which roughly halves linear raw data. Two
//! variants:
//!
//! - [`write_tiff16`] stores the merged data as is (camera native linear, no
//!   white balance), which is what a raw pipeline wants but what looks dark and
//!   tinted in a normal editor;
//! - [`write_tiff16_rendered`] applies the as-shot white balance, the camera to
//!   sRGB matrix, the white-level normalisation and the sRGB transfer curve, so
//!   the file displays like a finished picture everywhere.

use crate::types::{DngMeta, MergeResult};
use crate::write::color::{camera_to_srgb, encode_srgb_u16, mul3};
use crate::write::tiff::{self, Ifd, Value};
use anyhow::{bail, Context, Result};
use flate2::write::ZlibEncoder;
use flate2::Compression;
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

pub const ROWS_PER_STRIP: usize = 16;

/// Write the merged data as plain 16-bit RGB, camera native linear.
pub fn write_tiff16(path: &Path, img: &MergeResult, meta: &DngMeta, compress: bool) -> Result<()> {
    write_inner(path, img, meta, false, compress)
}

/// Write the merged data as 16-bit sRGB: white balance, colour matrix, sRGB
/// transfer curve and clipping, i.e. what an editor displays without guessing.
pub fn write_tiff16_rendered(
    path: &Path,
    img: &MergeResult,
    meta: &DngMeta,
    compress: bool,
) -> Result<()> {
    write_inner(path, img, meta, true, compress)
}

fn write_inner(
    path: &Path,
    img: &MergeResult,
    meta: &DngMeta,
    rendered: bool,
    compress: bool,
) -> Result<()> {
    if img.width == 0 || img.height == 0 {
        bail!("cannot write an empty image ({}x{})", img.width, img.height);
    }
    let expected = img.width * img.height;
    for (c, plane) in img.planes.iter().enumerate() {
        if plane.len() != expected {
            bail!(
                "plane {} has {} samples, expected {}",
                c,
                plane.len(),
                expected
            );
        }
    }

    let file = File::create(path).with_context(|| format!("cannot create {}", path.display()))?;
    let mut w = BufWriter::with_capacity(1 << 20, file);
    let height = img.height;
    let width = img.width;
    let strips = height.div_ceil(ROWS_PER_STRIP);
    let render = Render::new(img, meta);

    let ifd = make_ifd(img, meta, strips, compress);
    let ifd_size = ifd.size();
    let strips_off = tiff::align2(8 + ifd_size);
    let (ifd_bytes, positions) = ifd.encode(8);

    w.write_all(&tiff::HEADER)?;
    w.write_all(&ifd_bytes)?;
    if w.stream_position()? != strips_off {
        bail!("internal error: wrong strip start offset");
    }

    let mut offsets = Vec::with_capacity(strips);
    let mut counts = Vec::with_capacity(strips);
    let mut raw = Vec::new();
    let mut packed = Vec::new();
    let mut written = 0u64;
    for strip in 0..strips {
        let y0 = strip * ROWS_PER_STRIP;
        let y1 = (y0 + ROWS_PER_STRIP).min(height);
        if rendered {
            pack_rendered_rows(&img.planes, width, y0, y1, &render, &mut raw);
        } else {
            tiff::interleave_rows(&img.planes, width, y0, y1, &mut raw);
        }
        offsets.push((strips_off + written) as u32);
        if compress {
            // Predictor 2: store differences to the previous sample of each
            // row, which roughly halves the entropy of linear raw data
            // before zlib sees it.
            apply_horizontal_predictor(&mut raw, width * 3, 3);
            packed.clear();
            let mut encoder = ZlibEncoder::new(&mut packed, Compression::default());
            encoder.write_all(&raw)?;
            encoder.finish()?;
            w.write_all(&packed)?;
            counts.push(packed.len() as u32);
            written += packed.len() as u64;
        } else {
            w.write_all(&raw)?;
            counts.push(raw.len() as u32);
            written += raw.len() as u64;
        }
    }

    patch(&mut w, positions.get(&tiff::TAG_STRIP_OFFSETS), &offsets)?;
    patch(&mut w, positions.get(&tiff::TAG_STRIP_BYTE_COUNTS), &counts)?;
    w.flush()?;
    Ok(())
}

/// White balance + colour matrix + normalisation used by the rendered variant.
struct Render {
    neutral: [f32; 3],
    matrix: [[f32; 3]; 3],
    scale: f32,
}

impl Render {
    fn new(img: &MergeResult, meta: &DngMeta) -> Self {
        let neutral = if meta.wb_applied {
            [1.0f32, 1.0, 1.0]
        } else {
            meta.as_shot_neutral
        };
        Self {
            neutral,
            matrix: camera_to_srgb(&meta.color_matrix),
            scale: 1.0 / img.white.max(1) as f32,
        }
    }
}

fn pack_rendered_rows(
    planes: &[Vec<u16>; 3],
    width: usize,
    y0: usize,
    y1: usize,
    render: &Render,
    out: &mut Vec<u8>,
) {
    out.clear();
    out.reserve((y1 - y0) * width * 6);
    for y in y0..y1 {
        let base = y * width;
        for x in 0..width {
            let index = base + x;
            let mut cam = [0.0f32; 3];
            for (c, value) in cam.iter_mut().enumerate() {
                *value = planes[c][index] as f32 * render.scale / render.neutral[c].max(1e-6);
            }
            let srgb = mul3(&render.matrix, &cam);
            for value in srgb {
                out.extend_from_slice(&encode_srgb_u16(value).to_le_bytes());
            }
        }
    }
}

/// In place horizontal differencing (TIFF Predictor 2) over little endian
/// u16 samples: per component, each pixel stores its difference to the same
/// component of the previous pixel of the row; the first pixel keeps its
/// value. Wrapping subtraction.
fn apply_horizontal_predictor(bytes: &mut [u8], samples_per_row: usize, components: usize) {
    for row in bytes.chunks_exact_mut(samples_per_row * 2) {
        let mut previous = [0u16; 4];
        for pixel in row.chunks_exact_mut(2 * components) {
            for (c, pair) in pixel.chunks_exact_mut(2).enumerate() {
                let value = u16::from_le_bytes([pair[0], pair[1]]);
                pair.copy_from_slice(&value.wrapping_sub(previous[c]).to_le_bytes());
                previous[c] = value;
            }
        }
    }
}

fn make_ifd(img: &MergeResult, meta: &DngMeta, strips: usize, compress: bool) -> Ifd {
    let mut ifd = Ifd::new();
    ifd.set_long(tiff::TAG_NEW_SUBFILE_TYPE, 0);
    ifd.set_long(tiff::TAG_IMAGE_WIDTH, img.width as u32);
    ifd.set_long(tiff::TAG_IMAGE_LENGTH, img.height as u32);
    ifd.set(tiff::TAG_BITS_PER_SAMPLE, Value::Shorts(vec![16, 16, 16]));
    ifd.set_short(
        tiff::TAG_COMPRESSION,
        if compress {
            tiff::COMPRESSION_DEFLATE
        } else {
            tiff::COMPRESSION_NONE
        },
    );
    if compress {
        ifd.set_short(tiff::TAG_PREDICTOR, 2);
    }
    ifd.set_short(tiff::TAG_PHOTOMETRIC, tiff::PHOTOMETRIC_RGB);
    ifd.set_short(tiff::TAG_ORIENTATION, meta.orientation.max(1));
    ifd.set_short(tiff::TAG_SAMPLES_PER_PIXEL, 3);
    ifd.set(tiff::TAG_STRIP_OFFSETS, Value::Longs(vec![0; strips]));
    ifd.set_long(tiff::TAG_ROWS_PER_STRIP, ROWS_PER_STRIP as u32);
    ifd.set(tiff::TAG_STRIP_BYTE_COUNTS, Value::Longs(vec![0; strips]));
    ifd.set_short(tiff::TAG_PLANAR_CONFIG, 1);
    ifd.set(tiff::TAG_SAMPLE_FORMAT, Value::Shorts(vec![1, 1, 1]));
    ifd.set(tiff::TAG_X_RESOLUTION, Value::Rationals(vec![(300, 1)]));
    ifd.set(tiff::TAG_Y_RESOLUTION, Value::Rationals(vec![(300, 1)]));
    ifd.set_short(tiff::TAG_RESOLUTION_UNIT, 2);
    ifd.set_ascii(tiff::TAG_MAKE, &meta.make);
    ifd.set_ascii(tiff::TAG_MODEL, &meta.model);
    let software = if meta.software.is_empty() {
        crate::version()
    } else {
        meta.software.clone()
    };
    ifd.set_ascii(tiff::TAG_SOFTWARE, &software);
    if !meta.timestamp.is_empty() {
        ifd.set_ascii(tiff::TAG_DATETIME, &meta.timestamp);
    }
    ifd
}

fn patch<W: Write + Seek>(w: &mut W, pos: Option<&u64>, values: &[u32]) -> Result<()> {
    let Some(pos) = pos else {
        bail!("internal error: strip table tag not found");
    };
    w.seek(SeekFrom::Start(*pos))?;
    for value in values {
        w.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}
