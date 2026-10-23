//! Minimal baseline JPEG encoder for the embedded DNG preview.
//!
//! Cargo.toml is frozen and no JPEG encoder is available as a dependency, so
//! this module writes a plain 8-bit 4:4:4 baseline stream with standard quant
//! tables and one canonical Huffman table per class, derived from the symbol
//! histogram of the image (the all-ones code is left unused, as the standard
//! requires). Every decoder reads such a stream: ImageIO, libjpeg, PIL and the
//! preview extractors in LibRaw and rawler.

/// Standard luminance quantisation table, JPEG Annex K.
const QUANT_LUMA: [u8; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69,
    56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81,
    104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

/// Quality of the embedded preview (1..100).
pub const QUALITY: u8 = 80;

/// Standard zig-zag order: position k in the stream -> natural index.
const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27,
    20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Encode interleaved 8-bit RGB as a baseline JFIF stream.
pub fn encode_rgb(rgb: &[u8], width: usize, height: usize) -> Vec<u8> {
    assert!(width > 0 && height > 0, "empty jpeg preview");
    assert_eq!(rgb.len(), width * height * 3, "rgb buffer size");

    let quant = scaled_quant(QUALITY);
    let blocks_x = width.div_ceil(8);
    let blocks_y = height.div_ceil(8);
    let coeffs = coefficients(rgb, width, height, &quant, blocks_x, blocks_y);

    let (dc_table, ac_table) = tables(&coeffs);

    let mut out = Vec::with_capacity(width * height / 2);
    marker(&mut out, 0xd8); // SOI
    app0(&mut out);
    dqt(&mut out, &quant);
    sof0(&mut out, width, height);
    out.extend_from_slice(&dc_table.segment(0x00));
    out.extend_from_slice(&ac_table.segment(0x10));
    sos(&mut out);

    let mut bits = BitWriter::new(&mut out);
    let mut pred = [0i32; 3];
    for (index, block) in coeffs.iter().enumerate() {
        let component = index % 3;
        encode_block(&mut bits, block, component, &dc_table, &ac_table, &mut pred);
    }
    bits.finish();
    marker(&mut out, 0xd9); // EOI
    out
}

fn marker(out: &mut Vec<u8>, code: u8) {
    out.push(0xff);
    out.push(code);
}

fn segment(out: &mut Vec<u8>, code: u8, body: &[u8]) {
    marker(out, code);
    out.extend_from_slice(&((body.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(body);
}

fn app0(out: &mut Vec<u8>) {
    let mut body = Vec::with_capacity(14);
    body.extend_from_slice(b"JFIF\0");
    body.extend_from_slice(&[1, 1, 0]); // version 1.1, no units
    body.extend_from_slice(&[0, 1, 0, 1]); // 1x1 density
    body.extend_from_slice(&[0, 0]); // no thumbnail
    segment(out, 0xe0, &body);
}

/// IJG scaling: quality 50 keeps the table, lower values enlarge the steps.
fn scaled_quant(quality: u8) -> [u16; 64] {
    let q = quality.clamp(1, 100) as u32;
    let scale = if q < 50 { 5000 / q } else { 200 - 2 * q };
    let mut table = [0u16; 64];
    for (slot, base) in table.iter_mut().zip(QUANT_LUMA) {
        *slot = (((base as u32 * scale + 50) / 100).clamp(1, 255)) as u16;
    }
    table
}

fn dqt(out: &mut Vec<u8>, quant: &[u16; 64]) {
    let mut body = Vec::with_capacity(65);
    body.push(0x00); // 8-bit values, table 0
    for k in 0..64 {
        body.push(quant[ZIGZAG[k]] as u8);
    }
    segment(out, 0xdb, &body);
}

fn sof0(out: &mut Vec<u8>, width: usize, height: usize) {
    let mut body = Vec::with_capacity(15);
    body.push(8);
    body.extend_from_slice(&(height as u16).to_be_bytes());
    body.extend_from_slice(&(width as u16).to_be_bytes());
    body.push(3);
    for id in 1..=3u8 {
        body.push(id);
        body.push(0x11); // 1x1 sampling, 4:4:4
        body.push(0x00); // quantisation table 0
    }
    segment(out, 0xc0, &body);
}

fn sos(out: &mut Vec<u8>) {
    let mut body = Vec::with_capacity(10);
    body.push(3);
    for id in 1..=3u8 {
        body.push(id);
        body.push(0x00); // DC table 0, AC table 0
    }
    body.extend_from_slice(&[0, 63, 0]);
    segment(out, 0xda, &body);
}

/// Quantised coefficients of every block, in MCU order (block, then component).
fn coefficients(
    rgb: &[u8],
    width: usize,
    height: usize,
    quant: &[u16; 64],
    blocks_x: usize,
    blocks_y: usize,
) -> Vec<[i16; 64]> {
    let mut out = Vec::with_capacity(blocks_x * blocks_y * 3);
    let mut plane = [[0f32; 64]; 3];
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            for (c, block) in plane.iter_mut().enumerate() {
                for y in 0..8 {
                    let sy = (by * 8 + y).min(height - 1);
                    for x in 0..8 {
                        let sx = (bx * 8 + x).min(width - 1);
                        let i = (sy * width + sx) * 3;
                        let (r, g, b) = (rgb[i] as f32, rgb[i + 1] as f32, rgb[i + 2] as f32);
                        block[y * 8 + x] = match c {
                            0 => 0.299 * r + 0.587 * g + 0.114 * b,
                            1 => -0.168_736 * r - 0.331_264 * g + 0.5 * b + 128.0,
                            _ => 0.5 * r - 0.418_688 * g - 0.081_312 * b + 128.0,
                        } - 128.0;
                    }
                }
            }
            for block in plane.iter_mut() {
                fdct(block);
                out.push(quantise(block, quant));
            }
        }
    }
    out
}

fn fdct(block: &mut [f32; 64]) {
    const SCALE: [f32; 8] = [
        0.353_553_38,
        0.490_392_48,
        0.461_939_76,
        0.415_734_74,
        0.353_553_38,
        0.277_785_12,
        0.191_341_73,
        0.097_545_16,
    ];
    let mut tmp = [0f32; 64];
    for y in 0..8 {
        for u in 0..8 {
            let mut sum = 0.0;
            for x in 0..8 {
                let angle = std::f32::consts::PI / 16.0 * ((2 * x + 1) * u) as f32;
                sum += block[y * 8 + x] * angle.cos();
            }
            tmp[y * 8 + u] = sum * SCALE[u];
        }
    }
    for x in 0..8 {
        for v in 0..8 {
            let mut sum = 0.0;
            for y in 0..8 {
                let angle = std::f32::consts::PI / 16.0 * ((2 * y + 1) * v) as f32;
                sum += tmp[y * 8 + x] * angle.cos();
            }
            block[v * 8 + x] = sum * SCALE[v];
        }
    }
}

fn quantise(block: &[f32; 64], quant: &[u16; 64]) -> [i16; 64] {
    let mut out = [0i16; 64];
    for k in 0..64 {
        let q = quant[k] as f32;
        out[k] = ((block[ZIGZAG[k]] / q).round() as i32).clamp(-32_000, 32_000) as i16;
    }
    out
}

fn amplitude_bits(value: i32) -> u8 {
    if value == 0 {
        0
    } else {
        (32 - value.unsigned_abs().leading_zeros()) as u8
    }
}

/// Count the DC differences and AC symbols of the whole image.
fn tables(coeffs: &[[i16; 64]]) -> (HuffmanTable, HuffmanTable) {
    let mut dc = [0u32; 256];
    let mut ac = [0u32; 256];
    let mut pred = [0i32; 3];
    for (index, block) in coeffs.iter().enumerate() {
        let component = index % 3;
        let dc_value = block[0] as i32;
        dc[amplitude_bits(dc_value - pred[component]) as usize] += 1;
        pred[component] = dc_value;

        let mut run = 0usize;
        for value in block.iter().skip(1) {
            let value = *value as i32;
            if value == 0 {
                run += 1;
                continue;
            }
            while run > 15 {
                ac[0xf0] += 1;
                run -= 16;
            }
            ac[(((run as usize) << 4) | amplitude_bits(value) as usize) & 0xff] += 1;
            run = 0;
        }
        if run > 0 {
            ac[0] += 1;
        }
    }
    (HuffmanTable::build(&dc), HuffmanTable::build(&ac))
}

/// Canonical Huffman table: optimal lengths for the used symbols, with the
/// all-ones code left unused.
struct HuffmanTable {
    counts: [u8; 16],
    symbols: Vec<u8>,
    codes: Vec<(u16, u8)>,
}

impl HuffmanTable {
    fn build(freq: &[u32; 256]) -> Self {
        let mut used: Vec<(u32, u8)> = (0..256u32)
            .filter(|symbol| freq[*symbol as usize] > 0)
            .map(|symbol| (freq[symbol as usize], symbol as u8))
            .collect();
        if used.is_empty() {
            used.push((1, 0));
        }
        // Frequent symbols get the shorter codes; the tie-break on the symbol
        // value keeps the table deterministic.
        used.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        let n = used.len();
        let mut lengths = vec![0u8; n];
        if n == 1 {
            lengths[0] = 1;
        } else {
            let long = 32 - (n as u32 - 1).leading_zeros(); // ceil(log2 n)
            let capacity = 1usize << long;
            let short = capacity - n;
            for (i, length) in lengths.iter_mut().enumerate() {
                *length = if i < short {
                    (long - 1) as u8
                } else {
                    long as u8
                };
            }
            // Reserve the all-ones code: the least frequent symbol moves one
            // level deeper, which also keeps the table incomplete as required.
            lengths[n - 1] = long as u8 + 1;
        }

        let mut counts = [0u8; 16];
        for length in &lengths {
            counts[*length as usize - 1] += 1;
        }
        let mut symbols = Vec::with_capacity(n);
        for length in 1..=16u8 {
            for (i, symbol) in used.iter().enumerate() {
                if lengths[i] == length {
                    symbols.push(symbol.1);
                }
            }
        }
        let mut codes = vec![(0u16, 0u8); 256];
        let mut code = 0u32;
        let mut k = 0usize;
        for length in 1..=16u8 {
            for _ in 0..counts[length as usize - 1] {
                codes[symbols[k] as usize] = (code as u16, length);
                code += 1;
                k += 1;
            }
            code <<= 1;
        }
        Self {
            counts,
            symbols,
            codes,
        }
    }

    fn segment(&self, class_id: u8) -> Vec<u8> {
        let mut body = Vec::with_capacity(2 + 16 + self.symbols.len());
        body.push(class_id);
        body.extend_from_slice(&self.counts);
        body.extend_from_slice(&self.symbols);
        let mut out = Vec::new();
        segment(&mut out, 0xc4, &body);
        out
    }

    fn code(&self, symbol: u8) -> (u16, u8) {
        self.codes[symbol as usize]
    }
}

fn encode_block(
    bits: &mut BitWriter,
    block: &[i16; 64],
    component: usize,
    dc: &HuffmanTable,
    ac: &HuffmanTable,
    pred: &mut [i32; 3],
) {
    let value = block[0] as i32;
    let diff = value - pred[component];
    pred[component] = value;
    let size = amplitude_bits(diff);
    let (code, code_len) = dc.code(size);
    bits.write(code as u32, code_len as u32);
    bits.write_amplitude(diff, size);

    let mut run = 0usize;
    for value in block.iter().skip(1) {
        let value = *value as i32;
        if value == 0 {
            run += 1;
            continue;
        }
        while run > 15 {
            let (code, code_len) = ac.code(0xf0);
            bits.write(code as u32, code_len as u32);
            run -= 16;
        }
        let size = amplitude_bits(value);
        let symbol = (((run as u32) << 4) | size as u32) as u8;
        let (code, code_len) = ac.code(symbol);
        bits.write(code as u32, code_len as u32);
        bits.write_amplitude(value, size);
        run = 0;
    }
    if run > 0 {
        let (code, code_len) = ac.code(0);
        bits.write(code as u32, code_len as u32);
    }
}

struct BitWriter<'a> {
    out: &'a mut Vec<u8>,
    acc: u32,
    used: u32,
}

impl<'a> BitWriter<'a> {
    fn new(out: &'a mut Vec<u8>) -> Self {
        Self {
            out,
            acc: 0,
            used: 0,
        }
    }

    fn write(&mut self, value: u32, bits: u32) {
        if bits == 0 {
            return;
        }
        self.acc = (self.acc << bits) | (value & ((1u32 << bits) - 1));
        self.used += bits;
        while self.used >= 8 {
            let byte = ((self.acc >> (self.used - 8)) & 0xff) as u8;
            self.out.push(byte);
            if byte == 0xff {
                self.out.push(0x00);
            }
            self.used -= 8;
        }
        // Drop the emitted bits so the next value starts right after them.
        self.acc &= (1u32 << self.used) - 1;
    }

    fn write_amplitude(&mut self, value: i32, size: u8) {
        if size == 0 {
            return;
        }
        let mask = (1i32 << size) - 1;
        let encoded = if value < 0 { (value - 1) & mask } else { value & mask };
        self.write(encoded as u32, size as u32);
    }

    fn finish(&mut self) {
        if self.used > 0 {
            let pad = 8 - self.used;
            self.write((1u32 << pad) - 1, pad);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_a_valid_looking_stream() {
        let width = 37;
        let height = 23;
        let mut rgb = Vec::new();
        for y in 0..height {
            for x in 0..width {
                rgb.push((x * 255 / width) as u8);
                rgb.push((y * 255 / height) as u8);
                rgb.push(128);
            }
        }
        let jpeg = encode_rgb(&rgb, width, height);
        assert_eq!(&jpeg[..2], &[0xff, 0xd8]);
        assert_eq!(&jpeg[jpeg.len() - 2..], &[0xff, 0xd9]);
        assert!(jpeg.windows(2).any(|w| w == [0xff, 0xc0]));
        assert!(jpeg.windows(2).any(|w| w == [0xff, 0xc4]));
        assert!(jpeg.windows(2).any(|w| w == [0xff, 0xda]));
    }

    #[test]
    fn huffman_tables_are_incomplete_and_leave_the_all_ones_code_free() {
        let table = HuffmanTable::build(&[1; 256].map(|v| v as u32).to_vec().try_into().unwrap());
        assert!(table.counts.iter().map(|c| *c as u32).sum::<u32>() > 0);
        // The largest code must not be all ones for its length.
        let mut worst = (0u16, 0u8);
        for (code, len) in &table.codes {
            if *len > worst.1 || (*len == worst.1 && *code > worst.0) {
                worst = (*code, *len);
            }
        }
        assert!(worst.0 != (1u16 << worst.1) - 1, "all-ones code used");
    }
}
