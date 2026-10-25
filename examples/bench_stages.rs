//! Stage-by-stage timing of one 32 frame series. Run from the repo root:
//!   cargo run --release --example bench_stages
use anyhow::Result;
use pixelshift::app::pipeline;
use pixelshift::types::*;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<()> {
    let paths: Vec<PathBuf> = (710..=741)
        .map(|i| PathBuf::from(format!("109NZ6_3/_DSC0{i}.NEF")))
        .collect();

    let t = Instant::now();
    let frames = pipeline::decode(&paths)?;
    println!("decode         {:7.2} s  ({} frames)", t.elapsed().as_secs_f32(), frames.len());

    let t = Instant::now();
    let (shifts, grids) = pixelshift::shift::estimate_series_with_grids(&frames)?;
    println!("align          {:7.2} s", t.elapsed().as_secs_f32());

    let t = Instant::now();
    let motion = MotionParams { enabled: true, sensitivity: 1.0 };
    let masks = pixelshift::motion::masks(&frames, &shifts, &motion);
    println!("motion masks   {:7.2} s", t.elapsed().as_secs_f32());

    for deconv in [0.0f32, 1.0] {
        let params = MergeParams {
            scale: 2,
            align: true,
            shifts: shifts.clone(),
            grids: grids.clone(),
            motion,
            masks: masks.clone(),
            deconvolve: deconv,
            ..Default::default()
        };
        let t = Instant::now();
        let img = pixelshift::merge::merge(&frames, &params)?;
        println!(
            "merge deconv={:<3} {:7.2} s  ({}x{}, filled {})",
            deconv,
            t.elapsed().as_secs_f32(),
            img.width,
            img.height,
            img.stats.filled_pixels
        );

        let meta = pipeline::dng_meta(&frames[0], "bench");
        let opts = WriteOpts { compress: false, preview_max: 0 };
        let out = PathBuf::from("/tmp/ps_bench_stages.dng");
        let t = Instant::now();
        pixelshift::write::write_dng_with_flavor(&out, &img, &meta, &opts, pixelshift::write::DngFlavor::LinearRaw)?;
        println!(
            "write dng      {:7.2} s  ({} bytes)",
            t.elapsed().as_secs_f32(),
            std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0)
        );
        std::fs::remove_file(&out).ok();
    }
    Ok(())
}
