//! Thin FFI layer over LibRaw (shim/libraw_shim.c).
//!
//! Owned by work package WP1. Only the fields the pipeline needs cross the
//! boundary; the pixel buffer is handed over by LibRaw's C shim as a plain
//! `uint16_t` array of the active area.

use crate::types::Cfa;
use anyhow::{bail, Context, Result};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::path::Path;

/// Mirrors `ps_frame` from shim/libraw_shim.h.
#[repr(C)]
struct PsFrame {
    width: c_int,
    height: c_int,
    raw_width: c_int,
    raw_height: c_int,
    active_left: c_int,
    active_top: c_int,
    left_margin: c_int,
    top_margin: c_int,
    iwidth: c_int,
    iheight: c_int,
    colors: c_int,
    filters: u32,
    cfa: [c_int; 4],
    black: c_int,
    cblack: [u32; 4],
    maximum: c_int,
    linear_max: [u32; 4],
    cam_mul: [f32; 4],
    cam_xyz: [f32; 12],
    rgb_cam: [f32; 12],
    iso_speed: f32,
    shutter: f32,
    aperture: f32,
    focal_len: f32,
    lens_make: [c_char; 64],
    lens_model: [c_char; 128],
    lens_serial: [c_char; 128],
    body_serial: [c_char; 64],
    focal_35mm: c_int,
    max_ap: f32,
    min_focal: f32,
    max_focal: f32,
    max_ap4_min_focal: f32,
    max_ap4_max_focal: f32,
    timestamp: i64,
    make: [c_char; 64],
    model: [c_char; 64],
    software: [c_char; 64],
    datetime: [c_char; 24],
    pixels: *mut u16,
    error: [c_char; PS_ERROR_LEN],
}

const PS_ERROR_LEN: usize = 512;

/// Mirrors `ps_rgb` from shim/libraw_shim.h.
#[repr(C)]
struct PsRgb {
    width: c_int,
    height: c_int,
    colors: c_int,
    bits: c_int,
    data: *mut u8,
    error: [c_char; PS_ERROR_LEN],
}

extern "C" {
    fn ps_decode(path: *const c_char, out: *mut PsFrame) -> c_int;
    fn ps_free(f: *mut PsFrame);
    fn ps_render_rgb(path: *const c_char, out: *mut PsRgb) -> c_int;
    fn ps_free_rgb(r: *mut PsRgb);
    fn ps_version() -> *const c_char;
}

/// Everything the shim reports about one decoded file.
#[derive(Clone, Debug)]
pub struct Decoded {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u16>,
    pub cfa: Cfa,
    pub black: u16,
    pub cblack: [u32; 4],
    pub white: u16,
    pub wb: [f32; 4],
    pub cam_xyz: [[f32; 3]; 4],
    pub rgb_cam: [[f32; 3]; 4],
    pub iso: f32,
    pub shutter: f32,
    pub aperture: f32,
    pub focal: f32,
    pub lens_make: String,
    pub lens_model: String,
    pub lens_serial: String,
    pub body_serial: String,
    pub focal_35mm: u16,
    pub max_aperture: f32,
    pub lens_min_focal: f32,
    pub lens_max_focal: f32,
    pub lens_min_fnum: f32,
    pub lens_max_fnum: f32,
    pub make: String,
    pub model: String,
    pub software: String,
    pub timestamp: String,
    /// Active-area origin inside the raw image (for diagnostics only).
    pub active_left: usize,
    pub active_top: usize,
    pub raw_width: usize,
    pub raw_height: usize,
}

/// LibRaw version string reported by the linked shim, e.g. "0.22.2".
pub fn version() -> Option<String> {
    // SAFETY: ps_version returns a static NUL terminated C string.
    let ptr = unsafe { ps_version() };
    if ptr.is_null() {
        return None;
    }
    Some(
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned(),
    )
}

/// True when the native shim was linked and LibRaw reports a usable version.
pub fn is_available() -> bool {
    match version() {
        Some(v) => {
            let mut parts = v.split('.');
            let major: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
            let minor: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
            major > 0 || minor >= 18
        }
        None => false,
    }
}

/// LibRaw reports an unknown lens number as 0 (or a denormal); treat anything
/// that small as unknown so the writer never emits a nonsense rational.
fn positive(value: f32) -> f32 {
    if value.is_finite() && value > 0.01 {
        value
    } else {
        0.0
    }
}

fn c_string(buf: &[c_char]) -> String {
    // SAFETY: the buffer is NUL terminated by the shim (snprintf/strncpy).
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// One demosaiced 8-bit sRGB rendering of a raw file.
pub struct Rendered {
    pub width: usize,
    pub height: usize,
    /// width * height * 3 bytes.
    pub rgb: Vec<u8>,
}

/// Render a raw file (NEF/NEFX/DNG ...) through LibRaw's demosaic pipeline.
///
/// The result is 8-bit sRGB with the camera orientation ignored, which is all
/// the distortion fit needs: it compares structure, not colour. This is the
/// fallback reference reader for files that are not plain RGB TIFFs.
pub fn render_rgb(path: &Path) -> Result<Rendered> {
    let cpath = CString::new(path.as_os_str().as_encoded_bytes())
        .with_context(|| format!("path contains a NUL byte: {}", path.display()))?;
    let mut out = PsRgb {
        width: 0,
        height: 0,
        colors: 0,
        bits: 0,
        data: std::ptr::null_mut(),
        error: [0; PS_ERROR_LEN],
    };
    // SAFETY: out is a valid, writable ps_rgb; the shim owns the buffer.
    let rc = unsafe { ps_render_rgb(cpath.as_ptr(), &mut out) };
    if rc != 0 {
        let msg = c_string(&out.error);
        // SAFETY: ps_free_rgb tolerates a null buffer.
        unsafe { ps_free_rgb(&mut out) };
        bail!("LibRaw failed to render {}: {}", path.display(), msg);
    }
    let width = out.width.max(0) as usize;
    let height = out.height.max(0) as usize;
    let colors = out.colors.max(0) as usize;
    if width == 0 || height == 0 || colors < 3 || out.data.is_null() {
        unsafe { ps_free_rgb(&mut out) };
        bail!("LibRaw returned an empty rendering for {}", path.display());
    }
    let count = width * height;
    let rgb = if colors == 3 {
        // SAFETY: the shim allocated width * height * colors bytes.
        unsafe { std::slice::from_raw_parts(out.data, count * 3) }.to_vec()
    } else {
        // SAFETY: same buffer, more channels; drop everything past RGB.
        let all = unsafe { std::slice::from_raw_parts(out.data, count * colors) };
        let mut rgb = Vec::with_capacity(count * 3);
        for pixel in all.chunks_exact(colors) {
            rgb.extend_from_slice(&pixel[..3]);
        }
        rgb
    };
    // SAFETY: releases the buffer allocated by ps_render_rgb.
    unsafe { ps_free_rgb(&mut out) };
    Ok(Rendered { width, height, rgb })
}

/// Decode one raw file with LibRaw and copy out its active pixel area.
pub fn decode(path: &Path) -> Result<Decoded> {
    let cpath = CString::new(path.as_os_str().as_encoded_bytes())
        .with_context(|| format!("path contains a NUL byte: {}", path.display()))?;

    let mut frame = PsFrame {
        width: 0,
        height: 0,
        active_left: 0,
        active_top: 0,
        raw_width: 0,
        raw_height: 0,
        left_margin: 0,
        top_margin: 0,
        iwidth: 0,
        iheight: 0,
        colors: 0,
        filters: 0,
        cfa: [0; 4],
        black: 0,
        cblack: [0; 4],
        maximum: 0,
        linear_max: [0; 4],
        cam_mul: [0.0; 4],
        cam_xyz: [0.0; 12],
        rgb_cam: [0.0; 12],
        iso_speed: 0.0,
        shutter: 0.0,
        aperture: 0.0,
        focal_len: 0.0,
        lens_make: [0; 64],
        lens_model: [0; 128],
        lens_serial: [0; 128],
        body_serial: [0; 64],
        focal_35mm: 0,
        max_ap: 0.0,
        min_focal: 0.0,
        max_focal: 0.0,
        max_ap4_min_focal: 0.0,
        max_ap4_max_focal: 0.0,
        timestamp: 0,
        make: [0; 64],
        model: [0; 64],
        software: [0; 64],
        datetime: [0; 24],
        pixels: std::ptr::null_mut(),
        error: [0; PS_ERROR_LEN],
    };

    // SAFETY: frame is a valid, writable ps_frame; the shim fills it in and
    // owns the pixel buffer until ps_free.
    let rc = unsafe { ps_decode(cpath.as_ptr(), &mut frame) };
    if rc != 0 {
        let msg = c_string(&frame.error);
        bail!("LibRaw failed to decode {}: {}", path.display(), msg);
    }

    let width = frame.width.max(0) as usize;
    let height = frame.height.max(0) as usize;
    if width == 0 || height == 0 || frame.pixels.is_null() {
        // SAFETY: ps_free tolerates a null pixel buffer.
        unsafe { ps_free(&mut frame) };
        bail!("LibRaw returned an empty image for {}", path.display());
    }

    // SAFETY: the shim allocated width * height samples.
    let data = unsafe { std::slice::from_raw_parts(frame.pixels, width * height) }.to_vec();
    // SAFETY: releases the buffer allocated by ps_decode.
    unsafe { ps_free(&mut frame) };

    let mut cfa = [[crate::types::Color::G; 2]; 2];
    for (i, slot) in cfa.iter_mut().flatten().enumerate() {
        *slot = match frame.cfa[i] {
            0 => crate::types::Color::R,
            2 => crate::types::Color::B,
            _ => crate::types::Color::G,
        };
    }

    let mut cam_xyz = [[0.0f32; 3]; 4];
    let mut rgb_cam = [[0.0f32; 3]; 4];
    for i in 0..4 {
        for j in 0..3 {
            cam_xyz[i][j] = frame.cam_xyz[i * 3 + j];
            rgb_cam[i][j] = frame.rgb_cam[i * 3 + j];
        }
    }

    // LibRaw's as-shot multipliers are usually already green-normalised; make
    // that explicit because merge and the DNG writer assume wb[1] == 1.
    let mut wb = frame.cam_mul;
    if wb[1] > 0.0 {
        let g = wb[1];
        for v in wb.iter_mut() {
            *v /= g;
        }
    } else {
        wb = [1.0, 1.0, 1.0, 1.0];
    }
    for v in wb.iter_mut() {
        if !v.is_finite() || *v <= 0.0 {
            *v = 1.0;
        }
    }

    Ok(Decoded {
        width,
        height,
        data,
        cfa: Cfa(cfa),
        black: frame.black.clamp(0, u16::MAX as c_int) as u16,
        cblack: frame.cblack,
        white: frame.maximum.clamp(0, u16::MAX as c_int) as u16,
        wb,
        cam_xyz,
        rgb_cam,
        iso: frame.iso_speed,
        shutter: frame.shutter,
        aperture: frame.aperture,
        focal: frame.focal_len,
        lens_make: c_string(&frame.lens_make).trim().to_string(),
        lens_model: c_string(&frame.lens_model).trim().to_string(),
        lens_serial: c_string(&frame.lens_serial).trim().to_string(),
        body_serial: c_string(&frame.body_serial).trim().to_string(),
        focal_35mm: frame.focal_35mm.clamp(0, u16::MAX as c_int) as u16,
        max_aperture: positive(frame.max_ap),
        lens_min_focal: positive(frame.min_focal),
        lens_max_focal: positive(frame.max_focal),
        lens_min_fnum: positive(frame.max_ap4_min_focal),
        lens_max_fnum: positive(frame.max_ap4_max_focal),
        make: c_string(&frame.make),
        model: c_string(&frame.model),
        software: c_string(&frame.software),
        timestamp: c_string(&frame.datetime),
        active_left: frame.active_left.max(0) as usize,
        active_top: frame.active_top.max(0) as usize,
        raw_width: frame.raw_width.max(0) as usize,
        raw_height: frame.raw_height.max(0) as usize,
    })
}
