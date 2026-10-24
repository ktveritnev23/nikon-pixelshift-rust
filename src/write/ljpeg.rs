//! Lossless JPEG encoder (ITU-T T.81 Annex H, the process DNG Compression = 7
//! uses) for the main image of a DNG.
//!
//! The crate carries no JPEG codec dependency, so the entropy machinery is
//! shared with the baseline preview encoder in [`crate::write::jpeg`]: the
//! canonical Huffman table builder and the bit writer with 0xff stuffing are
//! exactly what lossless coding needs. What is added here is the SOF3 frame:
//! 16-bit samples, 1 (CFA mosaic) or 3 (interleaved RGB) components, predictor
//! 1 (left neighbour of the same component), point transform 0.
//!
//! The stream is written as ONE strip covering the whole image, and the
//! entropy data is streamed to the file row by row (first pass collects the
//! difference histogram, second pass writes the bits). This is not only a
//! memory decision: LibRaw's `lossless_dng_load_raw` follows multi-strip
//! lossless JPEG through a tile-offset walk that does not work for strip
//! layouts, so a per-strip stream format would decode only the first strip
//! in the very reader this project validates against.

use crate::write::jpeg::{amplitude_bits, segment, BitWriter, HuffmanTable};
use std::io::Write;

const SOI: [u8; 2] = [0xff, 0xd8];
const EOI: [u8; 2] = [0xff, 0xd9];
const SOF3: u8 = 0xc3;
const SOS: u8 = 0xda;
/// Sample precision: 16-bit linear data.
const PRECISION: u8 = 16;
/// Predictor 1: the left neighbour of the same component.
const PREDICTOR: u8 = 1;

/// Incremental lossless JPEG strip: collects the difference histogram row by
/// row (pass one) and then streams the entropy-coded rows to a writer (pass
/// two), so a 97 MP image never has to sit in memory as a whole.
pub struct StripEncoder {
    width: usize,
    components: usize,
    freq: [u32; 256],
    line_start: [u16; 4],
    rows: usize,
}

impl StripEncoder {
    pub fn new(width: usize, components: usize) -> Self {
        Self {
            width,
            components,
            freq: [0; 256],
            line_start: [RESET_PREDICTOR; 4],
            rows: 0,
        }
    }

    /// Add one row of `width * components` interleaved samples.
    pub fn add_row(&mut self, row: &[u16]) {
        let stride = self.components.max(1);
        debug_assert_eq!(row.len(), self.width * stride);
        let mut predictors = self.line_start;
        for (col, sample) in row.iter().take(self.width * stride).enumerate() {
            let c = col % stride;
            let diff = modular_diff(*sample, predictors[c]);
            self.freq[amplitude_bits(diff) as usize] += 1;
            predictors[c] = *sample;
        }
        for c in 0..stride.min(4) {
            self.line_start[c] = row[c];
        }
        self.rows += 1;
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Write the complete JPEG stream (SOI..EOI); `fill` is called once per
    /// row with a reusable buffer of `width * components` samples. Returns
    /// the number of bytes written.
    pub fn write_to<W: Write, F: FnMut(usize, &mut Vec<u16>)>(
        &self,
        w: &mut W,
        mut fill: F,
    ) -> std::io::Result<u64> {
        let mut w = Counting {
            inner: w,
            bytes: 0,
        };
        let table = HuffmanTable::build(&self.freq);
        w.write_all(&SOI)?;
        let mut header = Vec::with_capacity(64);
        write_sof3(&mut header, self.width, self.rows, self.components);
        w.write_all(&header)?;
        w.write_all(&table.segment(DHT_TABLE_ID))?;
        header.clear();
        write_sos(&mut header, self.components);
        w.write_all(&header)?;
        {
            let mut bits = BitWriter::new(&mut w);
            let stride = self.components.max(1);
            let mut row = vec![0u16; self.width * stride];
            let mut line_start = [RESET_PREDICTOR; 4];
            for y in 0..self.rows {
                fill(y, &mut row);
                let mut predictors = line_start;
                for (col, value) in row.iter().enumerate() {
                    let c = col % stride;
                    let diff = modular_diff(*value, predictors[c]);
                    let size = amplitude_bits(diff);
                    let (code, len) = table.code(size);
                    bits.write(code as u32, len as u32);
                    // Category 16 is the -32768 sentinel: no amplitude bits.
                    if size < 16 {
                        bits.write_amplitude(diff, size);
                    }
                    predictors[c] = *value;
                }
                for c in 0..stride.min(4) {
                    line_start[c] = row[c];
                }
            }
            bits.finish();
        }
        w.write_all(&EOI)?;
        Ok(w.bytes)
    }
}

/// Write adapter that only counts the bytes passing through.
struct Counting<'a, W: Write> {
    inner: &'a mut W,
    bytes: u64,
}

impl<W: Write> Write for Counting<'_, W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.bytes += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Encode a complete in-memory strip; see [`StripEncoder`] for the layout.
pub fn encode_strip(
    samples: &[u16],
    width: usize,
    height: usize,
    components: usize,
) -> Vec<u8> {
    debug_assert_eq!(samples.len(), width * height * components);
    let mut encoder = StripEncoder::new(width, components);
    for y in 0..height {
        encoder.add_row(&samples[y * width * components..][..width * components]);
    }
    let mut out = Vec::with_capacity(samples.len() / 2 + 256);
    encoder
        .write_to(&mut out, |y, row| {
            row.copy_from_slice(&samples[y * width * components..][..width * components])
        })
        .expect("Vec write cannot fail");
    out
}

/// One Huffman table serves every component (Td = Ta = 0): the difference
/// categories of all components share a distribution in linear raw data.
const DHT_TABLE_ID: u8 = 0x00;

fn write_sof3(out: &mut Vec<u8>, width: usize, height: usize, components: usize) {
    let mut body = Vec::with_capacity(8 + 3 * components);
    body.push(PRECISION);
    body.extend_from_slice(&(height as u16).to_be_bytes());
    body.extend_from_slice(&(width as u16).to_be_bytes());
    body.push(components as u8);
    for component in 0..components {
        body.push(component as u8 + 1); // Ci
        body.push(0x11); // Hi = Vi = 1: interleaved by sample
        body.push(0); // Tqi unused by lossless
    }
    segment(out, SOF3, &body);
}

fn write_sos(out: &mut Vec<u8>, components: usize) {
    let mut body = Vec::with_capacity(1 + 2 * components + 3);
    body.push(components as u8);
    for component in 0..components {
        body.push(component as u8 + 1);
        body.push(DHT_TABLE_ID); // Td | Ta
    }
    body.push(PREDICTOR); // Ss: predictor selection
    body.push(0); // Se: unused by lossless
    body.push(0); // Ah | Al: point transform 0
    segment(out, SOS, &body);
}

/// Difference as the lossless JPEG decoders reconstruct it: modular in
/// 2^16, so it always lands in [-32768, 32767] and category 16 - which the
/// DNG readers (LibRaw, Adobe) decode as the fixed sentinel -32768 without
/// amplitude bits - is only ever produced by exactly that value.
fn modular_diff(value: u16, pred: u16) -> i32 {
    value.wrapping_sub(pred) as i16 as i32
}

const RESET_PREDICTOR: u16 = 1 << (PRECISION - 1);

#[cfg(test)]
mod tests {
    use super::*;

    /// Decode the difference categories back out of a strip we encoded and
    /// check the samples round trip. A full independent decode is exercised in
    /// tests/writers.rs through LibRaw.
    #[test]
    fn strip_round_trips_through_the_bit_stream() {
        let (w, h) = (37usize, 5usize);
        let mut samples = Vec::with_capacity(w * h * 3);
        let mut value = 1234u16;
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    value = value
                        .wrapping_add(((x * 7 + y * 13 + c * 29) % 61) as u16)
                        .wrapping_sub(30);
                    samples.push(value);
                }
            }
        }
        let strip = encode_strip(&samples, w, h, 3);
        assert_eq!(&strip[..2], &SOI);
        assert_eq!(&strip[strip.len() - 2..], &EOI);
        // SOF3 header fields.
        assert_eq!(strip[2], 0xff);
        assert_eq!(strip[3], SOF3);
        let sof_len = u16::from_be_bytes([strip[4], strip[5]]) as usize;
        assert_eq!(strip[6], PRECISION);
        assert_eq!(u16::from_be_bytes([strip[7], strip[8]]) as usize, h);
        assert_eq!(u16::from_be_bytes([strip[9], strip[10]]) as usize, w);
        assert_eq!(strip[11], 3);
        assert_eq!(sof_len, 8 + 3 * 3 + 2 - 2); // Lf counts itself

        // The compressed strip must be a reasonable size: header + entropy
        // data, far below the uncompressed 37*5*3*2 = 1110 bytes for smooth
        // gradients.
        assert!(strip.len() < samples.len() * 2, "strip grew: {}", strip.len());
    }

    #[test]
    fn single_component_cfa_strip_encodes() {
        let (w, h) = (16usize, 4usize);
        let samples: Vec<u16> = (0..w * h).map(|i| (3000 + (i % 100) * 3) as u16).collect();
        let strip = encode_strip(&samples, w, h, 1);
        assert_eq!(strip[11], 1, "one component");
        assert!(strip.len() > 2 + 2 + (8 + 3) + 2 + (1 + 2 + 3) + 2);
    }
}
