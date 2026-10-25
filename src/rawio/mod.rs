//! Raw file decoding (Nikon NEF) through a thin C shim over LibRaw.
//!
//! Owned by work package WP1 (raw I/O + writers). Keep the signatures below
//! stable.

pub mod shim;

use crate::types::RawFrame;
use anyhow::{Context, Result};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// Decode a single raw file into a RawFrame (active area only).
pub fn open(path: &Path) -> Result<RawFrame> {
    open_indexed(path, 0)
}

/// Decode several frames in parallel, preserving order and setting index.
pub fn open_series(paths: &[PathBuf]) -> Result<Vec<RawFrame>> {
    let frames: Vec<Result<RawFrame>> = paths
        .par_iter()
        .enumerate()
        .map(|(index, path)| open_indexed(path, index))
        .collect();
    frames.into_iter().collect()
}

fn open_indexed(path: &Path, index: usize) -> Result<RawFrame> {
    let mut decoded = shim::decode(path)?;
    // LibRaw normalises "NIKON CORPORATION"/"NIKON Z6_3" into "Nikon"/"Z6_3",
    // but DNG consumers (Capture One, rawler, ImageIO) match the camera on the
    // exact strings the camera wrote, so prefer the file's own EXIF.
    if let Some((make, model)) = exif_make_model(path) {
        decoded.make = make;
        decoded.model = model;
    }
    Ok(RawFrame {
        path: path.to_path_buf(),
        width: decoded.width,
        height: decoded.height,
        data: decoded.data,
        cfa: decoded.cfa,
        black: decoded.black,
        cblack: decoded.cblack,
        white: decoded.white,
        wb: decoded.wb,
        cam_xyz: decoded.cam_xyz,
        rgb_cam: decoded.rgb_cam,
        iso: decoded.iso,
        shutter: decoded.shutter,
        aperture: decoded.aperture,
        focal: decoded.focal,
        lens_make: decoded.lens_make,
        lens_model: decoded.lens_model,
        lens_min_focal: decoded.lens_min_focal,
        lens_max_focal: decoded.lens_max_focal,
        lens_min_fnum: decoded.lens_min_fnum,
        lens_max_fnum: decoded.lens_max_fnum,
        lens_serial: decoded.lens_serial,
        body_serial: decoded.body_serial,
        focal_35mm: decoded.focal_35mm,
        max_aperture: decoded.max_aperture,
        make: decoded.make,
        model: decoded.model,
        timestamp: decoded.timestamp,
        index,
    })
}

/// Largest maker note copied into the output; a Nikon maker note is ~230 kB.
const MAX_MAKERNOTE: usize = 8 << 20;

const TAG_EXIF_IFD: u16 = 0x8769;
const TAG_MAKERNOTE: u16 = 0x927C;
const TYPE_UNDEFINED: u16 = 7;
const TYPE_LONG: u16 = 4;

/// One 12 byte IFD entry: tag, type, count and the raw value field.
struct Entry {
    tag: u16,
    kind: u16,
    count: u32,
    field: [u8; 4],
}

impl Entry {
    /// The entry as a single LONG (an IFD pointer). None for anything else.
    fn as_long(&self, little: bool) -> Option<u32> {
        if self.kind != TYPE_LONG || self.count != 1 {
            return None;
        }
        Some(if little {
            u32::from_le_bytes(self.field)
        } else {
            u32::from_be_bytes(self.field)
        })
    }
}

/// Every entry of the directory at the given offset.
fn ifd_entries(file: &mut fs::File, offset: u64, little: bool) -> Option<Vec<Entry>> {
    let mut count = [0u8; 2];
    file.seek(SeekFrom::Start(offset)).ok()?;
    file.read_exact(&mut count).ok()?;
    let entries = if little {
        u16::from_le_bytes(count)
    } else {
        u16::from_be_bytes(count)
    } as usize;
    if entries == 0 || entries > 1024 {
        return None;
    }
    let mut table = vec![0u8; entries * 12];
    file.read_exact(&mut table).ok()?;
    let u16_at = |b: &[u8]| {
        if little {
            u16::from_le_bytes([b[0], b[1]])
        } else {
            u16::from_be_bytes([b[0], b[1]])
        }
    };
    let u32_at = |b: &[u8]| {
        if little {
            u32::from_le_bytes([b[0], b[1], b[2], b[3]])
        } else {
            u32::from_be_bytes([b[0], b[1], b[2], b[3]])
        }
    };
    Some(
        table
            .chunks_exact(12)
            .map(|entry| {
                let mut field = [0u8; 4];
                field.copy_from_slice(&entry[8..12]);
                Entry {
                    tag: u16_at(&entry[0..2]),
                    kind: u16_at(&entry[2..4]),
                    count: u32_at(&entry[4..8]),
                    field,
                }
            })
            .collect(),
    )
}

/// Maker note (EXIF tag 0x927C) of a TIFF based raw file, copied byte for byte.
///
/// Raw editors identify the lens of a first party lens from the vendor maker
/// note rather than from LensModel, so an output without it gets an empty lens
/// profile. The Nikon block is self contained (its offsets are relative to its
/// own TIFF header), which is exactly why a verbatim copy needs no relocation.
///
/// Returns None when the file has no maker note or it cannot be parsed; the
/// writer then emits no tag, which is better than a broken one.
pub fn makernote(path: &Path) -> Option<Vec<u8>> {
    let mut file = fs::File::open(path).ok()?;
    let mut header = [0u8; 8];
    file.read_exact(&mut header).ok()?;
    let little = match &header[..2] {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let first = if little {
        u32::from_le_bytes([header[4], header[5], header[6], header[7]])
    } else {
        u32::from_be_bytes([header[4], header[5], header[6], header[7]])
    };
    let exif = ifd_entries(&mut file, first as u64, little)?
        .into_iter()
        .find(|entry| entry.tag == TAG_EXIF_IFD)?
        .as_long(little)?;
    let note = ifd_entries(&mut file, exif as u64, little)?
        .into_iter()
        .find(|entry| entry.tag == TAG_MAKERNOTE)?;
    if note.kind != TYPE_UNDEFINED || note.count == 0 || note.count as usize > MAX_MAKERNOTE {
        return None;
    }
    let mut bytes = vec![0u8; note.count as usize];
    if note.count <= 4 {
        bytes.copy_from_slice(&note.field[..note.count as usize]);
    } else {
        // An UNDEFINED value stores the offset in the value field, so it cannot
        // go through as_long (which insists on type LONG).
        let offset = if little {
            u32::from_le_bytes(note.field)
        } else {
            u32::from_be_bytes(note.field)
        };
        file.seek(SeekFrom::Start(offset as u64)).ok()?;
        file.read_exact(&mut bytes).ok()?;
    }
    Some(bytes)
}

/// Shutter count of one Nikon raw file: the Nikon3 maker note is an embedded
/// TIFF after the "Nikon\0" + version header, and its tag 0x00A7 carries the
/// counter. None when the note is absent, not embedded-TIFF style, or the
/// tag is missing.
pub fn shutter_count(path: &Path) -> Option<u64> {
    shutter_count_from_note(&makernote(path)?)
}

fn shutter_count_from_note(note: &[u8]) -> Option<u64> {
    // "Nikon\0" + 4 version bytes + TIFF header (II/MM, magic, IFD offset
    // relative to the header start).
    if note.len() < 18 || &note[..6] != b"Nikon\0" {
        return None;
    }
    let little = match &note[10..12] {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let rd_u16 = |o: usize| -> Option<u16> {
        note.get(o..o + 2).map(|b| {
            if little {
                u16::from_le_bytes([b[0], b[1]])
            } else {
                u16::from_be_bytes([b[0], b[1]])
            }
        })
    };
    let rd_u32 = |o: usize| -> Option<u32> {
        note.get(o..o + 4).map(|b| {
            if little {
                u32::from_le_bytes([b[0], b[1], b[2], b[3]])
            } else {
                u32::from_be_bytes([b[0], b[1], b[2], b[3]])
            }
        })
    };
    let ifd = 10 + rd_u32(14)? as usize;
    let entries = rd_u16(ifd)? as usize;
    for e in 0..entries {
        let base = ifd + 2 + 12 * e;
        let tag = rd_u16(base)?;
        if tag == 0x00a7 {
            let kind = rd_u16(base + 2)?;
            let count = rd_u32(base + 4)? as usize;
            if count != 1 {
                return None;
            }
            return match kind {
                4 => rd_u32(base + 8).map(|v| v as u64),
                3 => rd_u16(base + 8).map(|v| v as u64),
                1 => note.get(base + 8).map(|&v| v as u64),
                _ => None,
            };
        }
    }
    None
}

/// Make and Model exactly as written in IFD0 of a TIFF based raw file.
///
/// Only the 12 bytes of each matching IFD entry and the string itself are read,
/// never the whole file. Returns None for anything that is not a readable
/// classic TIFF, in which case the caller keeps LibRaw's normalised strings.
fn exif_make_model(path: &Path) -> Option<(String, String)> {
    let mut file = fs::File::open(path).ok()?;
    let mut header = [0u8; 8];
    file.read_exact(&mut header).ok()?;
    let little = match &header[..2] {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let u16_at = |bytes: &[u8]| {
        let pair = [bytes[0], bytes[1]];
        if little {
            u16::from_le_bytes(pair)
        } else {
            u16::from_be_bytes(pair)
        }
    };
    let u32_at = |bytes: &[u8]| {
        let quad = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if little {
            u32::from_le_bytes(quad)
        } else {
            u32::from_be_bytes(quad)
        }
    };
    if u16_at(&header[2..4]) != 42 {
        return None;
    }
    file.seek(SeekFrom::Start(u32_at(&header[4..8]) as u64))
        .ok()?;
    let mut count = [0u8; 2];
    file.read_exact(&mut count).ok()?;
    let entries = u16_at(&count) as usize;
    if entries == 0 || entries > 1024 {
        return None;
    }
    let mut table = vec![0u8; entries * 12];
    file.read_exact(&mut table).ok()?;

    let mut make = None;
    let mut model = None;
    for entry in table.chunks_exact(12) {
        let tag = u16_at(&entry[0..2]);
        let kind = u16_at(&entry[2..4]);
        let count = u32_at(&entry[4..8]) as usize;
        if kind != 2 || (tag != 271 && tag != 272) || count == 0 || count > 256 {
            continue;
        }
        let mut value = vec![0u8; count];
        if count <= 4 {
            value.copy_from_slice(&entry[8..8 + count]);
        } else {
            let offset = u32_at(&entry[8..12]) as u64;
            if file.seek(SeekFrom::Start(offset)).is_err() || file.read_exact(&mut value).is_err() {
                continue;
            }
        }
        let text = String::from_utf8_lossy(&value)
            .trim_end_matches('\0')
            .trim()
            .to_string();
        if tag == 271 {
            make = Some(text);
        } else {
            model = Some(text);
        }
    }
    match (make, model) {
        (Some(make), Some(model)) if !make.is_empty() && !model.is_empty() => Some((make, model)),
        _ => None,
    }
}

/// A Pixel Shift capture found on disk.
#[derive(Clone, Debug)]
pub struct Series {
    pub dir: PathBuf,
    pub files: Vec<PathBuf>,
    /// Detected capture mode (4, 8, 16, 32) when the file count matches.
    pub mode: u32,
    /// Suggested output scale for that mode.
    pub scale: u32,
}

/// Supported Pixel Shift capture sizes and the output scale they imply.
pub const MODES: [u32; 4] = [4, 8, 16, 32];

/// Suggested output grid density for a capture mode.
pub fn scale_for_mode(mode: u32) -> u32 {
    if mode >= 16 {
        2
    } else {
        1
    }
}

const MAX_SCAN_DEPTH: usize = 3;

/// Find Pixel Shift series below dir: groups of consecutive _DSCnnnn.NEF
/// files inside one directory, ignoring *_merged.NEFX, NC_FLLST.DAT and
/// anything else.
pub fn find_series(dir: &Path) -> Result<Vec<Series>> {
    let mut dirs = Vec::new();
    collect_dirs(dir, 0, &mut dirs);
    let mut out = Vec::new();
    for d in &dirs {
        out.extend(series_in_dir(d)?);
    }
    out.sort_by(|a: &Series, b: &Series| a.files.first().cmp(&b.files.first()));
    Ok(out)
}

fn collect_dirs(root: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    if let Ok(meta) = fs::metadata(root) {
        if meta.is_file() {
            if let Some(parent) = root.parent() {
                return collect_dirs(parent, depth, out);
            }
            return;
        }
    }
    out.push(root.to_path_buf());
    if depth >= MAX_SCAN_DEPTH {
        return;
    }
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    let mut subdirs: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .map(|e| e.path())
        .filter(|p| {
            !p.file_name()
                .map(|n| n.to_string_lossy().starts_with('.'))
                .unwrap_or(false)
        })
        .collect();
    subdirs.sort();
    for sub in subdirs {
        collect_dirs(&sub, depth + 1, out);
    }
}

/// Split "_DSC0522" into ("_DSC", 522); None when the stem has no trailing
/// digits.
fn split_trailing_number(stem: &str) -> Option<(&str, u32)> {
    let digits = stem.len() - stem.trim_end_matches(|c: char| c.is_ascii_digit()).len();
    if digits == 0 || digits == stem.len() {
        return None;
    }
    let number = stem[stem.len() - digits..].parse().ok()?;
    Some((&stem[..stem.len() - digits], number))
}

fn series_in_dir(dir: &Path) -> Result<Vec<Series>> {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(err) => return Err(err).with_context(|| format!("cannot read {}", dir.display())),
    };

    let mut groups: BTreeMap<String, Vec<(u32, PathBuf)>> = BTreeMap::new();
    for entry in entries.filter_map(|e| e.ok()) {
        let path = entry.path();
        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        let lower = name.to_ascii_lowercase();
        if !lower.ends_with(".nef") || lower.contains("_merged") {
            continue;
        }
        let stem = &name[..name.len() - 4];
        if let Some((prefix, number)) = split_trailing_number(stem) {
            groups
                .entry(prefix.to_ascii_lowercase())
                .or_default()
                .push((number, path));
        }
    }

    let mut out = Vec::new();
    for (_, mut files) in groups {
        files.sort_by(|a, b| a.0.cmp(&b.0));
        files.dedup_by(|a, b| a.0 == b.0);

        let mut run: Vec<PathBuf> = Vec::new();
        let mut previous: Option<u32> = None;
        for (number, path) in files {
            if let Some(prev) = previous {
                if number != prev.wrapping_add(1) {
                    push_run(dir, &mut run, &mut out);
                }
            }
            run.push(path);
            previous = Some(number);
        }
        push_run(dir, &mut run, &mut out);
    }
    Ok(out)
}

/// Cut a run of consecutive frame numbers into whole capture modes, largest
/// first (a 20 frame run becomes 16 + 4). Trailing frames that cannot form a
/// supported mode are dropped.
fn push_run(dir: &Path, run: &mut Vec<PathBuf>, out: &mut Vec<Series>) {
    let mut start = 0usize;
    let mut remaining = run.len();
    while remaining >= 4 {
        let Some(mode) = MODES
            .iter()
            .rev()
            .copied()
            .find(|m| (*m as usize) <= remaining)
        else {
            break;
        };
        let n = mode as usize;
        out.push(Series {
            dir: dir.to_path_buf(),
            files: run[start..start + n].to_vec(),
            mode,
            scale: scale_for_mode(mode),
        });
        start += n;
        remaining -= n;
    }
    run.clear();
}

#[cfg(test)]
mod tests {
    use super::shutter_count_from_note;

    /// Nikon3 style maker note: "Nikon\0" + version + an embedded little
    /// endian TIFF whose IFD carries tag 0x00A7.
    fn note(count: u32) -> Vec<u8> {
        let mut n = b"Nikon\0".to_vec();
        n.extend_from_slice(b"0100");
        n.extend_from_slice(b"II\x2a\x00");
        n.extend_from_slice(&8u32.to_le_bytes());
        let ifd = n.len();
        n.extend_from_slice(&1u16.to_le_bytes());
        n.extend_from_slice(&0x00a7u16.to_le_bytes());
        n.extend_from_slice(&4u16.to_le_bytes()); // LONG
        n.extend_from_slice(&1u32.to_le_bytes());
        n.extend_from_slice(&count.to_le_bytes());
        n.extend_from_slice(&0u32.to_le_bytes());
        let _ = ifd;
        n
    }

    #[test]
    fn shutter_count_is_read_from_the_nikon_note() {
        assert_eq!(shutter_count_from_note(&note(123_456)), Some(123_456));
        assert_eq!(shutter_count_from_note(&note(1)), Some(1));
    }

    #[test]
    fn foreign_notes_are_rejected() {
        assert_eq!(shutter_count_from_note(b"\x00\x01\x02"), None);
        assert_eq!(shutter_count_from_note(&note(5)[..14]), None);
        let mut other = note(5);
        other[10] = b'M';
        other[11] = b'M';
        assert_eq!(shutter_count_from_note(&other), None);
    }
}
