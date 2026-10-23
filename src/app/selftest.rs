//! Headless GUI selftest: drives the same worker + GuiState state machine the
//! window uses (load -> align -> merge -> preview -> 1:1 crop -> save) without
//! opening a window, and prints the timings.

use crate::app::model::{
    lod_level, visible_rect, GuiState, LodLevel, Stage, ViewSource, MAX_PREVIEW_DIM,
};
use crate::distort::Distortion;
use crate::app::pipeline;
use crate::app::worker::{Command, Event, Mode, SaveTarget, Worker};
use crate::app::GuiOptions;
use crate::types::PreviewImage;
use anyhow::{anyhow, bail, Result};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const STEP_TIMEOUT: Duration = Duration::from_secs(600);
const SELFTEST_VIEWPORT: [f32; 2] = [1200.0, 800.0];

pub fn run(options: &GuiOptions) -> Result<()> {
    println!("{} - headless GUI selftest", crate::version());
    let out_dir = PathBuf::from(".scratch/wp3/selftest");
    std::fs::create_dir_all(&out_dir)?;
    println!("output directory: {}", out_dir.display());
    println!();

    let report = drive(Mode::Demo, &[], &out_dir)?;
    report.print();
    if !report.ok() {
        bail!("GUI selftest failed");
    }

    if !options.inputs.is_empty() {
        println!();
        let listed = options
            .inputs
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(" ");
        println!("real data: {listed}");
        match drive(Mode::Real, &options.inputs, &out_dir) {
            Ok(report) => {
                report.print();
                if !report.ok() {
                    bail!("real pipeline selftest failed");
                }
            }
            Err(error) if pipeline::is_unavailable(&error) => {
                println!("[SKIP] real pipeline: {error:#}");
                println!("       WP1/WP2 are still stubs; re-run once they are implemented");
            }
            Err(error) => bail!("real pipeline failed: {error:#}"),
        }
    }
    Ok(())
}

/// Fit step: apply a known three term lens to a synthetic frame, write that
/// frame as the reference the user would export, and measure it back. This is
/// the same code path as the button, without depending on a raw editor.
fn fit_from_synthetic_reference(out_dir: &Path) -> Result<Check> {
    use crate::distort::{apply_lens, Distortion};
    use crate::types::{DngMeta, MergeResult, MergeStats};

    fn noise(x: f64, y: f64, cell: f64, seed: u64) -> f64 {
        let hash = |i: i64, j: i64| {
            let mut h = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
                ^ (j as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f)
                ^ seed.wrapping_mul(0x27d4_eb2f);
            h ^= h >> 33;
            h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
            h ^= h >> 29;
            (h >> 11) as f64 / (1u64 << 53) as f64
        };
        let (fx, fy) = (x / cell, y / cell);
        let (i, j) = (fx.floor(), fy.floor());
        let smooth = |t: f64| t * t * (3.0 - 2.0 * t);
        let (tx, ty) = (smooth(fx - i), smooth(fy - j));
        let top = hash(i as i64, j as i64) * (1.0 - tx) + hash(i as i64 + 1, j as i64) * tx;
        let bottom =
            hash(i as i64, j as i64 + 1) * (1.0 - tx) + hash(i as i64 + 1, j as i64 + 1) * tx;
        top * (1.0 - ty) + bottom * ty
    }

    const FIT_W: usize = 640;
    const FIT_H: usize = 480;
    let white = 12000.0;
    let mut planes = [vec![0u16; FIT_W * FIT_H], vec![0u16; FIT_W * FIT_H], vec![0u16; FIT_W * FIT_H]];
    for y in 0..FIT_H {
        for x in 0..FIT_W {
            let value = 0.25 + 0.55 * noise(x as f64, y as f64, 5.0, 1)
                + 0.15 * noise(x as f64, y as f64, 17.0, 2);
            let sample = (value * white).round().clamp(0.0, 65535.0) as u16;
            for (c, plane) in planes.iter_mut().enumerate() {
                plane[y * FIT_W + x] = sample + c as u16;
            }
        }
    }
    let clean = MergeResult {
        width: FIT_W,
        height: FIT_H,
        scale: 1,
        planes,
        white: white as u16,
        shifts: Vec::new(),
        stats: MergeStats::default(),
    };
    let known = Distortion::new3(-0.04, 0.005, -0.001);
    let recorded = apply_lens(clean.clone(), known);
    let path = out_dir.join("fit-reference.tif");
    crate::write::write_tiff16(&path, &clean, &DngMeta::default())?;
    let report = crate::fit::fit_from_reference(&recorded, &path)?;
    println!("  fit       {}", report.summary());
    let applied = known.k1 + known.k2 + known.k3;
    let fitted = report.k1 + report.k2 + report.k3;
    let ok = (report.k1 - known.k1).abs() / known.k1.abs() <= 0.10
        && (fitted - applied).abs() <= 0.10 * applied.abs()
        && report.residual_rms <= 0.05
        && report.blocks >= 8;
    Ok(Check::new(
        "fit: synthetic reference",
        ok,
        format!(
            "known {} -> k1 {:+.5} k2 {:+.5} k3 {:+.5}, residual {:.3} px, {} blocks",
            known, report.k1, report.k2, report.k3, report.residual_rms, report.blocks
        ),
    ))
}

fn drive(mode: Mode, inputs: &[PathBuf], out_dir: &Path) -> Result<Report> {
    let worker = Worker::spawn(mode);
    let mut state = GuiState::new(mode);
    state.auto_run = false;
    let started = Instant::now();

    worker.send(Command::Scan(inputs.to_vec()));
    pump_until(&worker, &mut state, Duration::from_secs(120), |state| {
        !state.series.is_empty()
    })?;
    let series = state
        .series
        .first()
        .map(|series| series.info.clone())
        .ok_or_else(|| anyhow!("scan returned no series"))?;
    let scale = state
        .options
        .scale
        .resolve(series.frame_count(), series.suggested_scale);
    let save_path = out_dir.join(format!(
        "_selftest_{}_{}x.{}",
        if mode == Mode::Demo { "demo" } else { "real" },
        scale,
        state.options.format.ext()
    ));
    let _ = std::fs::remove_file(&save_path);
    worker.send(Command::Load {
        index: 0,
        options: state.options.clone(),
        save: Some(SaveTarget {
            path: save_path,
            format: state.options.format,
            flavor: state.options.flavor,
            tiff_render: state.options.tiff_render,
            also_1x: state.options.also_1x,
        }),
    });
    pump_finished(&worker, &mut state)?;
    let full_preview = state.previews[0].clone();
    let dims = state
        .selected_series()
        .and_then(|series| series.result)
        .map(|info| (info.width, info.height))
        .unwrap_or((0, 0));

    // Zoom step: ask for the same 1:1 window the canvas asks for once the base
    // preview is magnified past 1:1. The fabricated demo frame is smaller than
    // the base preview, so the level decision is checked on a real frame size.
    let zoom = 1.0;
    let lod_decision = lod_level(zoom, SELFTEST_VIEWPORT, (6048, 4032), MAX_PREVIEW_DIM);
    let crop_rect = visible_rect(
        [dims.0 as f32 * 0.5, dims.1 as f32 * 0.5],
        SELFTEST_VIEWPORT,
        zoom,
        dims,
        0.25,
    )
    .unwrap_or((dims.0 / 4, dims.1 / 4, (dims.0 / 2).max(2), (dims.1 / 2).max(2)));
    worker.send(Command::View {
        source: ViewSource::Merged,
        rect: Some(crop_rect),
    });
    pump_until(&worker, &mut state, STEP_TIMEOUT, |state| {
        state.crop_for(Some(crop_rect)).is_some()
    })?;
    let detail_crop = state.crop_for(Some(crop_rect)).cloned();

    // Lens correction step: a k1 change must rebuild the preview without
    // touching the UI thread; the export path uses the same coefficients.
    let before = state.previews[0].as_ref().cloned();
    state.options.distort = Distortion::new(0.005, 0.0);
    let now = Instant::now();
    state.poll_distort(now);
    if state.poll_distort(now + crate::app::model::DISTORT_DEBOUNCE) {
        worker.send(Command::Distort {
            distort: state.options.distort,
        });
    }
    pump_until(&worker, &mut state, STEP_TIMEOUT, |state| {
        state
            .previews[0]
            .as_ref()
            .map(|slot| slot.generation)
            .unwrap_or(0)
            != before.as_ref().map(|slot| slot.generation).unwrap_or(0)
    })?;
    let distorted = state.previews[0].clone();
    let total = started.elapsed();

    let mut checks = Vec::new();
    checks.push(fit_from_synthetic_reference(out_dir)?);
    let series = state
        .selected_series()
        .ok_or_else(|| anyhow!("no series loaded"))?;
    checks.push(Check::new(
        "load: frames decoded",
        series.frames.len() == series.info.frame_count() && !series.frames.is_empty(),
        format!(
            "{} frames, {} x {}, CFA {}, black {}, white {}",
            series.frames.len(),
            series.frames.first().map(|f| f.width).unwrap_or(0),
            series.frames.first().map(|f| f.height).unwrap_or(0),
            series.frames.first().map(|f| f.cfa.clone()).unwrap_or_default(),
            series.frames.first().map(|f| f.black).unwrap_or(0),
            series.frames.first().map(|f| f.white).unwrap_or(0)
        ),
    ));
    let worst = series
        .shifts
        .iter()
        .map(|shift| shift.dx.abs().max(shift.dy.abs()))
        .fold(0.0f64, f64::max);
    let worst_rot = series
        .shifts
        .iter()
        .map(|shift| shift.rot_arcsec().abs())
        .fold(0.0f64, f64::max);
    checks.push(Check::new(
        "align: shifts estimated",
        series.shifts.len() == series.frames.len(),
        format!(
            "{} transforms, max |shift| {:.3} px, max |rot| {:.1}\"",
            series.shifts.len(),
            worst,
            worst_rot
        ),
    ));
    let result = series.result;
    checks.push(Check::new(
        "merge: result produced",
        result.map(|info| info.width > 0 && info.height > 0).unwrap_or(false),
        result
            .map(|info| {
                format!(
                    "{} x {} scale {}, {:.2} MP, white {}",
                    info.width,
                    info.height,
                    info.scale,
                    info.megapixels(),
                    info.white
                )
            })
            .unwrap_or_else(|| "no result".to_string()),
    ));
    checks.push(Check::new(
        "preview: full image",
        full_preview
            .as_ref()
            .map(|slot| slot.rect.is_none() && slot.image.width > 0 && !is_flat(&slot.image))
            .unwrap_or(false),
        full_preview
            .as_ref()
            .map(|slot| format!("{} x {} px", slot.image.width, slot.image.height))
            .unwrap_or_else(|| "no preview".to_string()),
    ));
    checks.push(Check::new(
        "zoom: lod decision",
        lod_decision == LodLevel::Detail,
        format!(
            "zoom {zoom:.2} of a 6048 x 4032 frame -> {lod_decision:?}, base preview {MAX_PREVIEW_DIM} px"
        ),
    ));
    let base_pixels = full_preview
        .as_ref()
        .map(|slot| slot.image.width * slot.image.height)
        .unwrap_or(0);
    let crop_pixels = crop_rect.2 * crop_rect.3;
    let crop_ok = detail_crop
        .as_ref()
        .map(|slot| {
            slot.rect == Some(crop_rect)
                && slot.image.width == crop_rect.2
                && slot.image.height == crop_rect.3
        })
        .unwrap_or(false);
    checks.push(Check::new(
        "preview: zoom -> detail crop",
        crop_ok && crop_pixels < base_pixels.max(1),
        detail_crop
            .as_ref()
            .map(|slot| {
                format!(
                    "{} x {} px for the {} x {} window (base preview {} px)",
                    slot.image.width, slot.image.height, crop_rect.2, crop_rect.3, base_pixels
                )
            })
            .unwrap_or_else(|| "no crop".to_string()),
    ));
    let distorted_changed = before
        .as_ref()
        .zip(distorted.as_ref())
        .map(|(a, b)| a.image.rgb != b.image.rgb)
        .unwrap_or(false);
    checks.push(Check::new(
        "preview: distortion k1 live",
        distorted_changed && state.options.distort.k1 == 0.005,
        format!(
            "k1 {:+.4}: preview rebuilt ({} -> generation {})",
            state.options.distort.k1,
            before.as_ref().map(|slot| slot.generation).unwrap_or(0),
            distorted
                .as_ref()
                .map(|slot| slot.generation)
                .unwrap_or(0)
        ),
    ));
    let saved = series.saved.last().cloned();
    checks.push(Check::new(
        "save: file written",
        saved
            .as_ref()
            .map(|(path, bytes)| *bytes > 0 && path.exists())
            .unwrap_or(false),
        saved
            .as_ref()
            .map(|(path, bytes)| format!("{} ({})", path.display(), pipeline::fmt_bytes(*bytes)))
            .unwrap_or_else(|| "nothing written".to_string()),
    ));

    Ok(Report {
        mode,
        checks,
        timings: state.stage_log.clone(),
        total,
    })
}

fn pump_until(
    worker: &Worker,
    state: &mut GuiState,
    timeout: Duration,
    done: impl Fn(&GuiState) -> bool,
) -> Result<()> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(event) = worker.wait(Duration::from_millis(250)) {
            state.apply(event);
        }
        if let Some(error) = state.error.clone() {
            return Err(anyhow!("{error}"));
        }
        if done(state) {
            return Ok(());
        }
        if Instant::now() > deadline {
            bail!("timed out after {:?}", timeout);
        }
    }
}

fn pump_finished(worker: &Worker, state: &mut GuiState) -> Result<()> {
    let deadline = Instant::now() + STEP_TIMEOUT;
    loop {
        match worker.wait(Duration::from_millis(250)) {
            Some(event) => {
                let finished = matches!(event, Event::Finished);
                state.apply(event);
                if let Some(error) = state.error.clone() {
                    return Err(anyhow!("{error}"));
                }
                if finished {
                    return Ok(());
                }
            }
            None if Instant::now() > deadline => bail!("timed out waiting for the merge"),
            None => {}
        }
    }
}

struct Check {
    name: &'static str,
    ok: bool,
    detail: String,
}

impl Check {
    fn new(name: &'static str, ok: bool, detail: String) -> Self {
        Self { name, ok, detail }
    }
}

pub struct Report {
    mode: Mode,
    checks: Vec<Check>,
    timings: Vec<(Stage, Duration)>,
    total: Duration,
}

impl Report {
    pub fn ok(&self) -> bool {
        self.checks.iter().all(|check| check.ok)
    }

    pub fn print(&self) {
        let title = match self.mode {
            Mode::Demo => "demo state machine (fabricated data)",
            Mode::Real => "real state machine (decoded NEF files)",
        };
        println!("{title}");
        for check in &self.checks {
            let mark = if check.ok { "ok  " } else { "FAIL" };
            let time = self.stage_for(&check.name);
            println!(
                "  [{mark}] {:<26} {:<58}{}",
                check.name, check.detail, time
            );
        }
        let mut line = String::new();
        for (stage, elapsed) in &self.timings {
            if !line.is_empty() {
                line.push_str(" | ");
            }
            line.push_str(&format!("{} {:.0} ms", stage.label(), ms(*elapsed)));
        }
        if !line.is_empty() {
            println!("  stages: {line}");
        }
        println!(
            "  total {:.2} s: {}",
            self.total.as_secs_f64(),
            if self.ok() { "PASS" } else { "FAIL" }
        );
    }

    fn stage_for(&self, name: &str) -> String {
        let stage = match name.split(':').next().unwrap_or("") {
            "load" => Some(Stage::Decoding),
            "align" => Some(Stage::Aligning),
            "merge" => Some(Stage::Merging),
            "preview" => Some(Stage::Previewing),
            "save" => Some(Stage::Saving),
            _ => None,
        };
        stage
            .and_then(|stage| {
                self.timings
                    .iter()
                    .find(|(s, _)| *s == stage)
                    .map(|(_, d)| format!("{:.0} ms", ms(*d)))
            })
            .unwrap_or_default()
    }
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn is_flat(image: &PreviewImage) -> bool {
    let mut min = 255u8;
    let mut max = 0u8;
    for value in image.rgb.iter().step_by(97) {
        min = min.min(*value);
        max = max.max(*value);
    }
    max.saturating_sub(min) < 8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::demo;

    #[test]
    fn preview_flatness_check_works_on_fabricated_data() {
        let frames = demo::frames();
        let result = demo::result(1, &frames);
        let image = demo::encode(&result, None, 512, [1.244, 1.0, 2.010]);
        assert!(!is_flat(&image));
        let flat = PreviewImage {
            width: 4,
            height: 4,
            rgb: vec![10; 4 * 4 * 3],
        };
        assert!(is_flat(&flat));
    }
}
