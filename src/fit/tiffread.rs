//! Minimal RGB TIFF reader for reference images.
//!
//! Only what a reference export needs: classic (32 bit) TIFF, first directory,
//! photometric RGB, 8 or 16 bits per sample, chunky (interleaved) layout,
//! strips (not tiles), compression none, Deflate or the old-style Deflate tag,
//! with or without horizontal differencing (predictor 2). Anything else is
//! refused with a message that names the unsupported feature, so the user can
//! re-export instead of guessing why the fit failed.

use anyhow::{bail, Context, Result};
use flate2::read::{DeflateDecoder, ZlibDecoder};
use std::io::Read;
use std::path::Path;

/// An RGB reference image, planar and normalised to 0..=1.
pub struct Rgb16 {
    pub width: usize,
    pub height: usize,
    pub bits: u16,
    pub planes: [Vec<f32>; 3],
}

const TAG_IMAGE_WIDTH: u16 = 256;
const TAG_IMAGE_LENGTH: u16 = 257;
const TAG_BITS_PER_SAMPLE: u16 = 258;
const TAG_COMPRESSION: u16 = 259;
const TAG_PHOTOMETRIC: u16 = 262;
const TAG_STRIP_OFFSETS: u16 = 273;
const TAG_SAMPLES_PER_PIXEL: u16 = 277;
const TAG_ROWS_PER_STRIP: u16 = 278;
const TAG_STRIP_BYTE_COUNTS: u16 = 279;
const TAG_PLANAR_CONFIG: u16 = 284;
const TAG_PREDICTOR: u16 = 317;
const TAG_TILE_WIDTH: u16 = 322;

/// One TIFF field of the first directory.
#[derive(Clone, Copy)]
struct Entry {
    typ: u16,
    count: u32,
    /// File offset of the value field (inline values live there too).
    field: usize,
}

struct Reader<'a> {
    data: &'a [u8],
    big: bool,
}

impl<'a> Reader<'a> {
    fn u16(&self, off: usize) -> Result<u16> {
        let bytes = self
            .data
            .get(off..off + 2)
            .with_context(|| format!("truncated TIFF at offset {off}"))?;
        Ok(if self.big {
            u16::from_be_bytes([bytes[0], bytes[1]])
        } else {
            u16::from_le_bytes([bytes[0], bytes[1]])
        })
    }

    fn u32(&self, off: usize) -> Result<u32> {
        let bytes = self
            .data
            .get(off..off + 4)
            .with_context(|| format!("truncated TIFF at offset {off}"))?;
        Ok(if self.big {
            u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
        } else {
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
        })
    }

    fn find(&self, ifd: usize, count: usize, tag: u16) -> Result<Option<Entry>> {
        for index in 0..count {
            let base = ifd + 2 + index * 12;
            if self.u16(base)? == tag {
                return Ok(Some(Entry {
                    typ: self.u16(base + 2)?,
                    count: self.u32(base + 4)?,
                    field: base + 8,
                }));
            }
        }
        Ok(None)
    }

    fn values(&self, entry: Entry) -> Result<Vec<u32>> {
        let size = match entry.typ {
            1 | 6 => 1,
            3 | 8 => 2,
            4 | 9 => 4,
            other => bail!("unsupported TIFF field type {other}"),
        };
        let total = size * entry.count as usize;
        let base = if total <= 4 {
            entry.field
        } else {
            self.u32(entry.field)? as usize
        };
        let mut out = Vec::with_capacity(entry.count as usize);
        for index in 0..entry.count as usize {
            let at = base + index * size;
            out.push(match size {
                1 => self.data.get(at).copied().context("truncated TIFF")? as u32,
                2 => self.u16(at)? as u32,
                _ => self.u32(at)?,
            });
        }
        Ok(out)
    }
}

fn first_u32(reader: &Reader, ifd: usize, count: usize, tag: u16) -> Result<Option<u32>> {
    match reader.find(ifd, count, tag)? {
        Some(entry) => Ok(reader.values(entry)?.first().copied()),
        None => Ok(None),
    }
}

fn inflate(raw: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    if ZlibDecoder::new(raw).read_to_end(&mut out).is_ok() {
        return Ok(out);
    }
    out.clear();
    DeflateDecoder::new(raw)
        .read_to_end(&mut out)
        .context("the Deflate strip cannot be decoded")?;
    Ok(out)
}

/// Read a whole RGB TIFF into planar normalised floats.
pub fn read_rgb(path: &Path) -> Result<Rgb16> {
    let data =
        std::fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
    if data.len() < 8 {
        bail!("{} is too short to be a TIFF", path.display());
    }
    let big = match &data[..2] {
        b"II" => false,
        b"MM" => true,
        _ => bail!("{} is not a TIFF file", path.display()),
    };
    let reader = Reader { data: &data, big };
    if reader.u16(2)? != 42 {
        bail!(
            "{}: only classic TIFF is supported (BigTIFF and JPEG-in-TIFF are not)",
            path.display()
        );
    }
    let ifd = reader.u32(4)? as usize;
    let count = reader.u16(ifd)? as usize;

    let width = first_u32(&reader, ifd, count, TAG_IMAGE_WIDTH)?
        .context("the TIFF has no ImageWidth")? as usize;
    let height = first_u32(&reader, ifd, count, TAG_IMAGE_LENGTH)?
        .context("the TIFF has no ImageLength")? as usize;
    if width == 0 || height == 0 {
        bail!("the TIFF declares an empty image ({width} x {height})");
    }
    if reader.find(ifd, count, TAG_TILE_WIDTH)?.is_some() {
        bail!(
            "{}: tiled TIFF is not supported; export strips, uncompressed or Deflate",
            path.display()
        );
    }
    let bits: Vec<u16> = match reader.find(ifd, count, TAG_BITS_PER_SAMPLE)? {
        Some(entry) => reader
            .values(entry)?
            .iter()
            .map(|value| *value as u16)
            .collect(),
        None => vec![1],
    };
    if bits.iter().any(|value| *value != bits[0]) || !(bits[0] == 8 || bits[0] == 16) {
        bail!(
            "{}: samples of {bits:?} bits are not supported (8 or 16 bit RGB)",
            path.display()
        );
    }
    let bits = bits[0];
    let spp = first_u32(&reader, ifd, count, TAG_SAMPLES_PER_PIXEL)?.unwrap_or(3) as usize;
    if spp != 3 && spp != 4 {
        bail!(
            "{}: {spp} samples per pixel; the reference must be RGB",
            path.display()
        );
    }
    let photometric = first_u32(&reader, ifd, count, TAG_PHOTOMETRIC)?.unwrap_or(2);
    if photometric != 2 {
        bail!(
            "{}: photometric interpretation {photometric} is not RGB (2)",
            path.display()
        );
    }
    if first_u32(&reader, ifd, count, TAG_PLANAR_CONFIG)?.unwrap_or(1) != 1 {
        bail!("{}: planar (separate channel) TIFF is not supported", path.display());
    }
    let compression = first_u32(&reader, ifd, count, TAG_COMPRESSION)?.unwrap_or(1);
    if !matches!(compression, 1 | 8 | 32946) {
        bail!(
            "{}: compression {compression} is not supported; export uncompressed or Deflate \
             (LZW and JPEG TIFFs cannot be read)",
            path.display()
        );
    }
    let predictor = first_u32(&reader, ifd, count, TAG_PREDICTOR)?.unwrap_or(1);
    let offsets = match reader.find(ifd, count, TAG_STRIP_OFFSETS)? {
        Some(entry) => reader.values(entry)?,
        None => bail!("the TIFF has no StripOffsets"),
    };
    let counts = match reader.find(ifd, count, TAG_STRIP_BYTE_COUNTS)? {
        Some(entry) => reader.values(entry)?,
        None => bail!("the TIFF has no StripByteCounts"),
    };
    let rows_per_strip =
        first_u32(&reader, ifd, count, TAG_ROWS_PER_STRIP)?.unwrap_or(height as u32) as usize;

    let bps = (bits / 8) as usize;
    let max = if bits == 16 { 65535.0f32 } else { 255.0f32 };
    let mask: u32 = if bits == 16 { 0xffff } else { 0xff };
    let row_bytes = width * spp * bps;
    let mut planes: [Vec<f32>; 3] = [
        vec![0.0; width * height],
        vec![0.0; width * height],
        vec![0.0; width * height],
    ];

    let mut y0 = 0usize;
    for (strip, offset) in offsets.iter().enumerate() {
        let rows = rows_per_strip.min(height - y0);
        if rows == 0 {
            break;
        }
        let used = counts.get(strip).copied().unwrap_or(0) as usize;
        let start = *offset as usize;
        let end = start + used;
        if end > data.len() {
            bail!(
                "strip {strip} runs past the end of {} ({end} > {})",
                path.display(),
                data.len()
            );
        }
        let raw = &data[start..end];
        let inflated;
        let samples: &[u8] = if compression == 1 {
            raw
        } else {
            inflated = inflate(raw)?;
            &inflated
        };
        if samples.len() < rows * row_bytes {
            bail!(
                "strip {strip} holds {} bytes but {rows} rows of {width} px need {}",
                samples.len(),
                rows * row_bytes
            );
        }
        for row in 0..rows {
            let mut previous = [0u32; 4];
            let base = row * row_bytes;
            for x in 0..width {
                for c in 0..spp {
                    let at = base + (x * spp + c) * bps;
                    let mut value = if bps == 1 {
                        samples[at] as u32
                    } else if big {
                        u16::from_be_bytes([samples[at], samples[at + 1]]) as u32
                    } else {
                        u16::from_le_bytes([samples[at], samples[at + 1]]) as u32
                    };
                    if predictor == 2 {
                        value = value.wrapping_add(previous[c]) & mask;
                        previous[c] = value;
                    }
                    if c < 3 {
                        planes[c][(y0 + row) * width + x] = value as f32 / max;
                    }
                }
            }
        }
        y0 += rows;
    }
    if y0 < height {
        bail!(
            "the strips cover {y0} of {height} rows of {}",
            path.display()
        );
    }

    Ok(Rgb16 {
        width,
        height,
        bits,
        planes,
    })
}
