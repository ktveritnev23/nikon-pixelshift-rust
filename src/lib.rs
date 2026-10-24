//! Pixel Shift merging for Nikon raw files.
//!
//! Pipeline: rawio (decode NEF via LibRaw) -> shift (sub-pixel alignment)
//! -> merge (per-colour reconstruction on a 1x or 2x grid) -> write
//! (linear DNG / 16-bit TIFF).

pub mod types;
pub mod rawio;
pub mod shift;
pub mod merge;
pub mod motion;
pub mod distort;
pub mod fit;
pub mod write;
pub mod cli;
pub mod app;

pub use types::*;

/// Short version banner used by CLI and GUI.
pub fn version() -> String {
    format!("pixelshift {}", env!("CARGO_PKG_VERSION"))
}
