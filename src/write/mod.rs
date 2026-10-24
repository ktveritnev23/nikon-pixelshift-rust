//! Output writers: linear DNG (dng) and 16-bit RGB TIFF (tiff16).
//!
//! Owned by work package WP1.

mod color;
pub mod dng;
pub mod jpeg;
pub mod ljpeg;
pub mod tiff;
pub mod tiff16;

pub use dng::{write_dng, write_dng_with_flavor, DngFlavor};
pub use tiff16::{write_tiff16, write_tiff16_rendered};
