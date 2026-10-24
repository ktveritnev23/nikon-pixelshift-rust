//! Little-endian TIFF primitives shared by the DNG and the 16-bit TIFF writer.
//!
//! Only the subset needed by the two writers is implemented: classic (32 bit
//! offset) TIFF, one or more directories, values inlined when they fit into
//! the four byte value field.

use std::collections::HashMap;

pub const TYPE_BYTE: u16 = 1;
pub const TYPE_ASCII: u16 = 2;
pub const TYPE_SHORT: u16 = 3;
pub const TYPE_LONG: u16 = 4;
pub const TYPE_RATIONAL: u16 = 5;
pub const TYPE_SBYTE: u16 = 6;
/// EXIF UNDEFINED: opaque bytes whose interpretation depends on the tag (the
/// vendor maker note is the one this writer needs).
pub const TYPE_UNDEFINED: u16 = 7;
pub const TYPE_SRATIONAL: u16 = 10;

pub const TAG_NEW_SUBFILE_TYPE: u16 = 254;
pub const TAG_IMAGE_WIDTH: u16 = 256;
pub const TAG_IMAGE_LENGTH: u16 = 257;
pub const TAG_BITS_PER_SAMPLE: u16 = 258;
pub const TAG_COMPRESSION: u16 = 259;
pub const TAG_PHOTOMETRIC: u16 = 262;
pub const TAG_MAKE: u16 = 271;
pub const TAG_MODEL: u16 = 272;
pub const TAG_STRIP_OFFSETS: u16 = 273;
pub const TAG_ORIENTATION: u16 = 274;
pub const TAG_SAMPLES_PER_PIXEL: u16 = 277;
pub const TAG_ROWS_PER_STRIP: u16 = 278;
pub const TAG_STRIP_BYTE_COUNTS: u16 = 279;
pub const TAG_YCBCR_SUB_SAMPLING: u16 = 530;
pub const TAG_YCBCR_POSITIONING: u16 = 531;
pub const TAG_X_RESOLUTION: u16 = 282;
pub const TAG_Y_RESOLUTION: u16 = 283;
pub const TAG_PREDICTOR: u16 = 317;
pub const TAG_PLANAR_CONFIG: u16 = 284;
pub const TAG_RESOLUTION_UNIT: u16 = 296;
pub const TAG_SOFTWARE: u16 = 305;
pub const TAG_DATETIME: u16 = 306;
pub const TAG_SUB_IFDS: u16 = 330;
pub const TAG_SAMPLE_FORMAT: u16 = 339;
pub const TAG_CFA_REPEAT_PATTERN_DIM: u16 = 33421;
pub const TAG_CFA_PATTERN: u16 = 33422;
pub const TAG_EXPOSURE_TIME: u16 = 33434;
pub const TAG_FNUMBER: u16 = 33437;
pub const TAG_ISO: u16 = 34855;
pub const TAG_DATETIME_ORIGINAL: u16 = 36867;
pub const TAG_FOCAL_LENGTH: u16 = 37386;

pub const TAG_DNG_VERSION: u16 = 50706;
pub const TAG_DNG_BACKWARD_VERSION: u16 = 50707;
pub const TAG_UNIQUE_CAMERA_MODEL: u16 = 50708;
pub const TAG_BLACK_LEVEL: u16 = 50714;
pub const TAG_WHITE_LEVEL: u16 = 50717;
pub const TAG_DEFAULT_CROP_ORIGIN: u16 = 50719;
pub const TAG_DEFAULT_CROP_SIZE: u16 = 50720;
pub const TAG_COLOR_MATRIX1: u16 = 50721;
pub const TAG_AS_SHOT_NEUTRAL: u16 = 50728;
pub const TAG_BASELINE_EXPOSURE: u16 = 50730;
pub const TAG_CALIBRATION_ILLUMINANT1: u16 = 50778;
pub const TAG_ACTIVE_AREA: u16 = 50829;
pub const TAG_CFA_PLANE_COLOR: u16 = 50710;
pub const TAG_CFA_LAYOUT: u16 = 50711;

pub const COMPRESSION_NONE: u16 = 1;
pub const COMPRESSION_JPEG: u16 = 7;
pub const COMPRESSION_DEFLATE: u16 = 8;
pub const PHOTOMETRIC_RGB: u16 = 2;
pub const PHOTOMETRIC_YCBCR: u16 = 6;
pub const PHOTOMETRIC_CFA: u16 = 32803;
pub const PHOTOMETRIC_LINEAR_RAW: u16 = 34892;
/// RGGB in DNG colour indices (0 = red, 1 = green, 2 = blue).
pub const CFA_RGGB: [u8; 4] = [0, 1, 1, 2];

/// One TIFF tag value.
#[derive(Clone, Debug)]
pub enum Value {
    Bytes(Vec<u8>),
    Undefined(Vec<u8>),
    Ascii(String),
    Shorts(Vec<u16>),
    Longs(Vec<u32>),
    Rationals(Vec<(u32, u32)>),
    Srationals(Vec<(i32, i32)>),
}

impl Value {
    pub fn type_id(&self) -> u16 {
        match self {
            Value::Bytes(_) => TYPE_BYTE,
            Value::Undefined(_) => TYPE_UNDEFINED,
            Value::Ascii(_) => TYPE_ASCII,
            Value::Shorts(_) => TYPE_SHORT,
            Value::Longs(_) => TYPE_LONG,
            Value::Rationals(_) => TYPE_RATIONAL,
            Value::Srationals(_) => TYPE_SRATIONAL,
        }
    }

    /// Number of values (ASCII counts its terminating NUL).
    pub fn count(&self) -> u32 {
        match self {
            Value::Bytes(v) => v.len() as u32,
            Value::Undefined(v) => v.len() as u32,
            Value::Ascii(s) => s.len() as u32 + 1,
            Value::Shorts(v) => v.len() as u32,
            Value::Longs(v) => v.len() as u32,
            Value::Rationals(v) => v.len() as u32,
            Value::Srationals(v) => v.len() as u32,
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        match self {
            Value::Bytes(v) => v.clone(),
            Value::Undefined(v) => v.clone(),
            Value::Ascii(s) => {
                let mut b = s.as_bytes().to_vec();
                b.push(0);
                b
            }
            Value::Shorts(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Value::Longs(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Value::Rationals(v) => v
                .iter()
                .flat_map(|(n, d)| [n.to_le_bytes(), d.to_le_bytes()])
                .flatten()
                .collect(),
            Value::Srationals(v) => v
                .iter()
                .flat_map(|(n, d)| [n.to_le_bytes(), d.to_le_bytes()])
                .flatten()
                .collect(),
        }
    }
}

/// One TIFF directory, serialised into file order.
#[derive(Clone, Debug, Default)]
pub struct Ifd {
    tags: Vec<(u16, Value)>,
}

impl Ifd {
    pub fn new() -> Self {
        Self { tags: Vec::new() }
    }

    pub fn set(&mut self, tag: u16, value: Value) {
        self.tags.push((tag, value));
    }

    pub fn set_short(&mut self, tag: u16, value: u16) {
        self.set(tag, Value::Shorts(vec![value]));
    }

    pub fn set_long(&mut self, tag: u16, value: u32) {
        self.set(tag, Value::Longs(vec![value]));
    }

    pub fn set_ascii(&mut self, tag: u16, value: &str) {
        if !value.is_empty() {
            self.set(tag, Value::Ascii(value.to_string()));
        }
    }

    /// Size of the encoded directory including its out-of-line values.
    pub fn size(&self) -> u64 {
        self.encode(0).0.len() as u64
    }

    /// Encode the directory for the absolute file offset `ifd_off`. The second
    /// return value maps every tag to the absolute position of its encoded
    /// value bytes, so callers can patch values (strip offsets) after the image
    /// data has been written.
    pub fn encode(&self, ifd_off: u64) -> (Vec<u8>, HashMap<u16, u64>) {
        self.encode_with_next(ifd_off, 0)
    }

    /// Encode the directory and link it to the next one (0 = last directory).
    pub fn encode_with_next(
        &self,
        ifd_off: u64,
        next_ifd: u32,
    ) -> (Vec<u8>, HashMap<u16, u64>) {
        let mut tags: Vec<&(u16, Value)> = self.tags.iter().collect();
        tags.sort_by_key(|(tag, _)| *tag);

        let count = tags.len() as u64;
        let blob_base = ifd_off + 2 + 12 * count + 4;

        let mut out = Vec::new();
        out.extend_from_slice(&(count as u16).to_le_bytes());

        let mut entries: Vec<Vec<u8>> = Vec::with_capacity(tags.len());
        let mut blob: Vec<u8> = Vec::new();
        let mut positions = HashMap::new();

        for (tag, value) in &tags {
            let data = value.encode();
            let value_field_pos = ifd_off + 2 + 12 * entries.len() as u64 + 8;
            let field: [u8; 4] = if data.len() <= 4 {
                let mut f = [0u8; 4];
                f[..data.len()].copy_from_slice(&data);
                positions.insert(*tag, value_field_pos);
                f
            } else {
                let offset = blob_base + blob.len() as u64;
                positions.insert(*tag, offset);
                blob.extend_from_slice(&data);
                if blob.len() % 2 == 1 {
                    blob.push(0);
                }
                (offset as u32).to_le_bytes()
            };

            let mut entry = Vec::with_capacity(12);
            entry.extend_from_slice(&tag.to_le_bytes());
            entry.extend_from_slice(&value.type_id().to_le_bytes());
            entry.extend_from_slice(&value.count().to_le_bytes());
            entry.extend_from_slice(&field);
            entries.push(entry);
        }

        for entry in entries {
            out.extend_from_slice(&entry);
        }
        out.extend_from_slice(&next_ifd.to_le_bytes());
        out.extend_from_slice(&blob);
        (out, positions)
    }
}

/// TIFF header: little endian, classic magic, first IFD at offset 8.
pub const HEADER: [u8; 8] = [b'I', b'I', 0x2a, 0x00, 0x08, 0x00, 0x00, 0x00];

/// Round a file position up to an even offset.
pub fn align2(pos: u64) -> u64 {
    pos + (pos & 1)
}

/// Encode the rows `y0..y1` of an RGGB mosaic: one little endian 16 bit sample
/// per pixel, taken from the plane of that pixel's colour.
pub fn mosaic_rows(
    planes: &[Vec<u16>; 3],
    width: usize,
    y0: usize,
    y1: usize,
    out: &mut Vec<u8>,
) {
    out.clear();
    out.reserve((y1 - y0) * width * 2);
    for y in y0..y1 {
        let base = y * width;
        let row = (y & 1) * 2;
        for x in 0..width {
            let color = CFA_RGGB[row + (x & 1)] as usize;
            out.extend_from_slice(&planes[color][base + x].to_le_bytes());
        }
    }
}

/// Encode interleaved RGB planes (planar input) as chunky little endian 16 bit
/// samples for the rows `y0..y1`.
pub fn interleave_rows(
    planes: &[Vec<u16>; 3],
    width: usize,
    y0: usize,
    y1: usize,
    out: &mut Vec<u8>,
) {
    out.clear();
    out.reserve((y1 - y0) * width * 6);
    for y in y0..y1 {
        let base = y * width;
        for x in 0..width {
            for plane in planes.iter() {
                out.extend_from_slice(&plane[base + x].to_le_bytes());
            }
        }
    }
}
