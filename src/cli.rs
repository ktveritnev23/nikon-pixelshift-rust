//! Command line interface (headless merging and inspection).
//!
//! Required surface:
//!   pixelshift merge <input...> [--out FILE|DIR] [--scale auto|1|2]
//!                   [--format dng|tiff] [--compression none|lossless] [--motion]
//!                   [--also-1x] [--no-align] [--threads N] [--quiet]
//!   pixelshift inspect <input...>
//!   pixelshift gui [--selftest] [--demo] [--lang auto|en|ru]
//!                   [--theme auto|dark|light] [--font system|builtin]
//!                   [--distort K1[,K2[,K3]]] [--shot PATH] [--shot-zoom Z]
//!                   [--shot-crop X,Y,W,H] [--shot-source merged|reference]
//!                   [--shot-save] [--shot-timeout SECS] [--shot-log] [input...]
//!
//! Progress goes to stderr, the summary to stdout; any error exits with code 1.

use crate::app::i18n::Lang;
use crate::app::model::{self, Compression, OutFormat, ScaleMode, ViewSource};
use crate::app::pipeline::{self, SeriesInfo};
use crate::app::prefs::{FontMode, ThemeMode};
use crate::app::{GuiOptions, PrefsOverrides};
use crate::types::MergeParams;
use crate::write::DngFlavor;
use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "pixelshift",
    version,
    about = "Nikon Pixel Shift merger: linear DNG / 16-bit TIFF output",
    long_about = "Merges a Nikon Pixel Shift series (4/8/16/32 frames) into a high \
                  resolution linear RGB image stored as a linear DNG (default) or a \
                  16-bit TIFF. Inputs are directories (series auto-detected) or \
                  explicit NEF files."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Merge one or more Pixel Shift series.
    Merge(MergeArgs),
    /// Print camera metadata, detected mode, measured shifts and output size.
    Inspect(InspectArgs),
    /// Start the graphical interface.
    Gui(GuiArgs),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum ScaleArg {
    Auto,
    #[value(name = "1")]
    X1,
    #[value(name = "2")]
    X2,
}

impl ScaleArg {
    pub fn mode(self) -> ScaleMode {
        match self {
            ScaleArg::Auto => ScaleMode::Auto,
            ScaleArg::X1 => ScaleMode::One,
            ScaleArg::X2 => ScaleMode::Two,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum FormatArg {
    Dng,
    Tiff,
}

impl FormatArg {
    pub fn format(self) -> OutFormat {
        match self {
            FormatArg::Dng => OutFormat::Dng,
            FormatArg::Tiff => OutFormat::Tiff,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum FlavorArg {
    /// LinearRaw (34892) main image: full colour resolution raw DNG.
    #[value(name = "linearraw")]
    LinearRaw,
    /// 16-bit RGB (2) main image: for readers that reject a raw main image.
    #[value(name = "rgb")]
    Rgb,
    /// RGGB mosaic (32803 CFA): for readers that only accept a Bayer raw.
    #[value(name = "cfa")]
    Cfa,
}

impl FlavorArg {
    pub fn flavor(self) -> DngFlavor {
        match self {
            FlavorArg::LinearRaw => DngFlavor::LinearRaw,
            FlavorArg::Rgb => DngFlavor::Rgb,
            FlavorArg::Cfa => DngFlavor::Cfa,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum CompressionArg {
    None,
    /// DNG: lossless JPEG (Compression = 7); TIFF: AdobeDeflate + predictor.
    Lossless,
    /// Compatibility alias of `lossless` from the pre-1.0 CLI.
    Deflate,
}

impl CompressionArg {
    pub fn compression(self) -> Compression {
        match self {
            CompressionArg::None => Compression::None,
            CompressionArg::Lossless | CompressionArg::Deflate => Compression::Lossless,
        }
    }
}

#[derive(Args, Debug)]
pub struct MergeArgs {
    /// Input directory (series auto-detected) or explicit NEF files.
    #[arg(value_name = "INPUT", required = true)]
    pub inputs: Vec<PathBuf>,
    /// Output file (one series) or directory (several series).
    #[arg(long, value_name = "PATH")]
    pub out: Option<PathBuf>,
    /// Output grid density; auto = 1 for 4/8 frames and 2 for 16/32 frames.
    #[arg(long, value_enum, default_value_t = ScaleArg::Auto)]
    pub scale: ScaleArg,
    /// Output container.
    #[arg(long, value_enum, default_value_t = FormatArg::Dng)]
    pub format: FormatArg,
    /// DNG main image flavour: linearraw (full colour resolution), rgb (plain
    /// 16-bit RGB) or cfa (RGGB mosaic, needed by rawler/ImageIO based readers).
    #[arg(long, value_enum, default_value_t = FlavorArg::LinearRaw)]
    pub dng_flavor: FlavorArg,
    /// Render the 16-bit TIFF as sRGB (white balance, colour matrix, gamma).
    /// This is the default for --format tiff.
    #[arg(long, conflicts_with = "tiff_linear")]
    pub tiff_render: bool,
    /// Keep the 16-bit TIFF camera native and linear (no white balance, no gamma).
    #[arg(long)]
    pub tiff_linear: bool,
    /// Merge deconvolution strength, 0..2 (0 = pure normalised splatting).
    #[arg(long, value_name = "0..2", value_parser = parse_deconvolve)]
    pub deconvolve: Option<f32>,
    /// Mask scene motion out of the merge (ghost suppression): areas where a
    /// frame disagrees with the reference after alignment fall back to the
    /// reference samples instead of averaging into ghosts.
    #[arg(long)]
    pub motion: bool,
    /// Sensitivity of the motion mask, 0.25..4 (higher masks more).
    #[arg(long, value_name = "F", default_value_t = 1.0)]
    pub motion_sensitivity: f32,
    /// Lens distortion correction, K1[,K2[,K3]] of the Brown model; 0 keeps
    /// the merged geometry untouched. Positive k1 removes pincushion.
    #[arg(long, value_name = "K1[,K2[,K3]]", default_value = "0", allow_hyphen_values = true, value_parser = parse_distort)]
    pub distort: crate::distort::Distortion,
    /// Measure the distortion instead of guessing it: the merged frame is
    /// matched block by block against a reference image exported from a raw
    /// editor with its lens profile applied (16-bit TIFF, or any raw file) and
    /// the resulting k1,k2,k3 are applied. Conflicts with --distort.
    #[arg(long, value_name = "PATH")]
    pub distort_fit: Option<PathBuf>,
    /// Lossless compression: lossless JPEG inside the DNG, AdobeDeflate with
    /// a horizontal predictor inside the TIFF.
    #[arg(long, value_enum, default_value_t = CompressionArg::None)]
    pub compression: CompressionArg,
    /// Compatibility alias for --compression none.
    #[arg(long, hide = true)]
    pub no_compress: bool,
    /// Also write a 1x downsample when the merged result is 2x.
    #[arg(long)]
    pub also_1x: bool,
    /// Do not estimate shifts; merge as if every frame were aligned.
    #[arg(long)]
    pub no_align: bool,
    /// Electronic stabilization: crop the output to the rectangle every frame
    /// covers after alignment (the common overlap). A hand-held burst then
    /// yields a slightly smaller, correctly merged image.
    #[arg(long)]
    pub stabilize: bool,
    /// Do not copy the vendor maker note (EXIF 0x927C) of the reference file
    /// into the DNG. Copying is the default because raw editors identify a
    /// first party lens, and therefore its profile, from the maker note.
    #[arg(long)]
    pub no_makernote: bool,
    /// Worker threads for decoding and merging.
    #[arg(long, value_name = "N")]
    pub threads: Option<usize>,
    /// Suppress progress lines on stderr.
    #[arg(long)]
    pub quiet: bool,
}

impl MergeArgs {
    pub fn options(&self) -> model::EditOptions {
        model::EditOptions {
            scale: self.scale.mode(),
            format: self.format.format(),
            compression: if self.no_compress {
                Compression::None
            } else {
                self.compression.compression()
            },
            flavor: self.dng_flavor.flavor(),
            tiff_render: self.tiff_render || !self.tiff_linear,
            deconvolve: self
                .deconvolve
                .unwrap_or_else(|| MergeParams::default().deconvolve),
            stabilize: self.stabilize,
            motion_enabled: self.motion,
            motion_sensitivity: self.motion_sensitivity,
            also_1x: self.also_1x,
            align: !self.no_align,
            threads: self.threads,
            distort: self.distort,
        }
    }
}

#[derive(Args, Debug)]
pub struct InspectArgs {
    /// Input directory (series auto-detected) or explicit NEF files.
    #[arg(value_name = "INPUT", required = true)]
    pub inputs: Vec<PathBuf>,
    /// Do not estimate shifts (skips decoding the remaining frames).
    #[arg(long)]
    pub no_align: bool,
    /// Report the crop electronic stabilization would apply.
    #[arg(long)]
    pub stabilize: bool,
    /// Also run the motion detector and report the masked share per frame.
    #[arg(long)]
    pub motion: bool,
    /// Sensitivity of the motion mask, 0.25..4 (higher masks more).
    #[arg(long, value_name = "F", default_value_t = 1.0)]
    pub motion_sensitivity: f32,
    /// Lens distortion correction that a merge with these flags would apply.
    #[arg(long, value_name = "K1[,K2[,K3]]", default_value = "0", allow_hyphen_values = true, value_parser = parse_distort)]
    pub distort: crate::distort::Distortion,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum LangArg {
    /// Follow the system locale (ru-* -> Russian, otherwise English).
    Auto,
    En,
    Ru,
}

impl LangArg {
    pub fn lang(self) -> Lang {
        match self {
            LangArg::Auto => Lang::detect(),
            LangArg::En => Lang::En,
            LangArg::Ru => Lang::Ru,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum ThemeArg {
    /// Follow the system theme.
    Auto,
    Dark,
    Light,
}

impl ThemeArg {
    pub fn theme(self) -> ThemeMode {
        match self {
            ThemeArg::Auto => ThemeMode::System,
            ThemeArg::Dark => ThemeMode::Dark,
            ThemeArg::Light => ThemeMode::Light,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum FontArg {
    /// The system UI font, with the built-in font as a fallback.
    System,
    /// Only the fonts bundled with egui.
    Builtin,
}

impl FontArg {
    pub fn font(self) -> FontMode {
        match self {
            FontArg::System => FontMode::System,
            FontArg::Builtin => FontMode::Builtin,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum ShotSourceArg {
    /// The merged high resolution result.
    Merged,
    /// The first frame of the series, as decoded.
    Reference,
}

impl ShotSourceArg {
    pub fn source(self) -> ViewSource {
        match self {
            ShotSourceArg::Merged => ViewSource::Merged,
            ShotSourceArg::Reference => ViewSource::Reference,
        }
    }
}

#[derive(Args, Debug)]
pub struct GuiArgs {
    /// Run the GUI state machine without opening a window and print a report.
    #[arg(long)]
    pub selftest: bool,
    /// Open the window with fabricated data instead of real NEF files.
    #[arg(long)]
    pub demo: bool,
    /// Interface language for this run; auto follows the system locale.
    #[arg(long, value_enum)]
    pub lang: Option<LangArg>,
    /// Colour theme for this run; auto follows the system theme.
    #[arg(long, value_enum)]
    pub theme: Option<ThemeArg>,
    /// UI font for this run; omitted keeps the stored choice.
    #[arg(long, value_enum)]
    pub font: Option<FontArg>,
    /// Lens distortion for this run: K1[,K2[,K3]] of the Brown model, positive
    /// k1 removes pincushion; never stored.
    #[arg(long, value_name = "K1[,K2[,K3]]", allow_hyphen_values = true, value_parser = parse_distort)]
    pub distort: Option<crate::distort::Distortion>,
    /// Capture the window to a JPEG once the merge is done, then exit; for the
    /// reproducible README screenshots.
    #[arg(long, value_name = "PATH")]
    pub shot: Option<PathBuf>,
    /// Canvas zoom of --shot; omitted keeps the fit-to-window view.
    #[arg(long, value_name = "FACTOR")]
    pub shot_zoom: Option<f32>,
    /// 1:1 crop of the merged image shown by --shot: X,Y,W,H in output pixels.
    #[arg(long, value_name = "X,Y,W,H", value_parser = parse_shot_crop)]
    pub shot_crop: Option<(usize, usize, usize, usize)>,
    /// Preview --shot captures: the merged result or the reference frame.
    #[arg(long, value_enum, default_value_t = ShotSourceArg::Merged)]
    pub shot_source: ShotSourceArg,
    /// Save the merged result to a temporary file before --shot captures the
    /// window, so the log line about it is in the picture.
    #[arg(long)]
    pub shot_save: bool,
    /// Seconds --shot waits for the merge and the frame.
    #[arg(long, value_name = "SECS", default_value_t = 20.0)]
    pub shot_timeout: f32,
    /// Start --shot with the log panel expanded.
    #[arg(long)]
    pub shot_log: bool,
    /// Optional inputs to preload (directories or NEF files).
    #[arg(value_name = "INPUT")]
    pub inputs: Vec<PathBuf>,
}

impl GuiArgs {
    /// Screenshot mode as the application sees it; None for a normal run.
    pub fn shot(&self) -> Option<crate::app::ShotRequest> {
        let path = self.shot.clone()?;
        Some(crate::app::ShotRequest {
            path,
            zoom: self.shot_zoom,
            crop: self.shot_crop,
            source: self.shot_source.source(),
            save: self.shot_save,
            log: self.shot_log,
            timeout: std::time::Duration::from_secs_f32(self.shot_timeout.max(0.1)),
        })
    }
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Merge(args) => run_merge(args),
        Command::Inspect(args) => run_inspect(args),
        Command::Gui(args) => {
            let shot = args.shot();
            crate::app::run_with(GuiOptions {
                selftest: args.selftest,
                demo: args.demo,
                inputs: args.inputs,
                prefs: PrefsOverrides {
                    lang: args.lang.map(LangArg::lang),
                    theme: args.theme.map(ThemeArg::theme),
                    font: args.font.map(FontArg::font),
                },
                distort: args.distort,
                shot,
            })
        }
    }
}

fn parse_distort(text: &str) -> std::result::Result<crate::distort::Distortion, String> {
    crate::distort::Distortion::parse(text)
}

/// "X,Y,W,H" of a preview crop, in output pixels.
fn parse_shot_crop(
    text: &str,
) -> std::result::Result<(usize, usize, usize, usize), String> {
    let parts: Vec<&str> = text.split([',', ';']).map(str::trim).collect();
    if parts.len() != 4 {
        return Err(format!("{text}: expected X,Y,W,H"));
    }
    let mut values = [0usize; 4];
    for (slot, part) in values.iter_mut().zip(&parts) {
        *slot = part
            .parse()
            .map_err(|_| format!("{part} is not a pixel count"))?;
    }
    if values[2] == 0 || values[3] == 0 {
        return Err(format!("{text}: the crop needs a width and a height"));
    }
    Ok((values[0], values[1], values[2], values[3]))
}

fn parse_deconvolve(text: &str) -> std::result::Result<f32, String> {
    let value: f32 = text
        .parse()
        .map_err(|_| format!("{text} is not a number"))?;
    if (0.0..=2.0).contains(&value) {
        Ok(value)
    } else {
        Err(format!("{text} is outside 0..2"))
    }
}

fn run_merge(args: MergeArgs) -> Result<()> {
    pipeline::apply_threads(args.threads);
    let series = pipeline::resolve_inputs(&args.inputs)?;
    if series.is_empty() {
        bail!(
            "no Pixel Shift series found in {}",
            args.inputs
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    let options = args.options();
    if args.distort_fit.is_some() && !args.distort.is_identity() {
        bail!("--distort and --distort-fit are mutually exclusive: the fit measures the coefficients itself");
    }
    if let Some(note) = options.compression_note() {
        if !args.quiet {
            eprintln!("warning: {note}");
        }
    }
    let mut failures = 0usize;
    for (index, item) in series.iter().enumerate() {
        if let Err(error) = merge_one(&args, &options, item, index, series.len()) {
            eprintln!("error: {error:#}");
            failures += 1;
        }
    }
    if failures > 0 {
        bail!("{failures} of {} series failed", series.len());
    }
    Ok(())
}

fn merge_one(
    args: &MergeArgs,
    options: &model::EditOptions,
    series: &SeriesInfo,
    index: usize,
    total: usize,
) -> Result<()> {
    let scale = options
        .scale
        .resolve(series.frame_count(), series.suggested_scale);
    let out = output_path(
        args.out.as_deref(),
        series,
        index,
        total,
        scale,
        options.format.ext(),
    )?;
    let prefix = format!("[{}/{}] {}", index + 1, total, series.name());
    let started = Instant::now();

    let say = |text: String| {
        if !args.quiet {
            eprintln!("{prefix}: {text}");
        }
    };

    say(format!("decoding {} frames", series.frame_count()));
    let decode_started = Instant::now();
    let frames = pipeline::decode(&series.files)?;
    if frames.is_empty() {
        bail!("{}: decoder returned no frames", series.name());
    }
    say(format!(
        "decoded in {:.2} s",
        decode_started.elapsed().as_secs_f64()
    ));

    let first = &frames[0];
    let expected = pipeline::expected_output(first.width, first.height, scale);
    if let Some(free) = pipeline::free_space(&out) {
        if free < expected.bytes() {
            say(format!(
                "warning: {} need {} but only {} is free",
                out.display(),
                pipeline::fmt_bytes(expected.bytes()),
                pipeline::fmt_bytes(free)
            ));
        }
    }

    let align = !args.no_align;
    if align {
        say(format!("aligning {} frames", frames.len()));
    } else {
        say("alignment disabled (--no-align)".to_string());
    }
    let shifts = pipeline::estimate_shifts(&frames, align)?;

    say(format!(
        "merging at scale {} ({} x {}, {:.2} MP)",
        scale, expected.width, expected.height, expected.megapixels()
    ));
    let params = options.merge_params(scale, shifts.clone());
    let merged = pipeline::merge_frames(&frames, &params)?;
    let mut correction = options.distort;
    if let Some(reference) = args.distort_fit.as_deref() {
        say(format!("fitting the lens distortion from {}", reference.display()));
        let report = crate::fit::fit_from_reference(&merged, reference)
            .with_context(|| format!("fitting the distortion from {}", reference.display()))?;
        say(format!("fit: {}", report.summary()));
        correction = crate::distort::Distortion::from_measured(report.k1, report.k2, report.k3)
            .map_err(|error| anyhow::anyhow!("the fit is unusable: {error}"))?;
        println!("  fit       {}", report.summary());
    }
    if !correction.is_identity() {
        say(format!(
            "correcting lens distortion ({}), corner moves {:.1} px",
            correction,
            pipeline::corner_shift(merged.width, merged.height, correction)
        ));
    }
    let merged = pipeline::rectify(merged, correction)?;

    let mut meta = pipeline::dng_meta(first, &crate::version());
    if args.no_makernote {
        meta.makernote.clear();
        say("maker note not copied (--no-makernote)".to_string());
    } else if !meta.makernote.is_empty() {
        say(format!(
            "copying the {} byte maker note of the reference file",
            meta.makernote.len()
        ));
    }
    say(format!("writing {}", out.display()));
    if let Some(dir) = out.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir)?;
        }
    }
    let bytes = pipeline::write_output(
        &out,
        &merged,
        &meta,
        options.format,
        options.compression,
        options.flavor,
        options.tiff_render,
    )?;

    let mut written = vec![(out.clone(), bytes)];
    if options.also_1x && merged.scale > 1 {
        let one_x_path = pipeline::sibling_1x_path(&out);
        say(format!("writing {}", one_x_path.display()));
        let reduced = pipeline::downsample_1x(&merged)?;
        let bytes = pipeline::write_output(
            &one_x_path,
            &reduced,
            &meta,
            options.format,
            options.compression,
            options.flavor,
            options.tiff_render,
        )?;
        written.push((one_x_path, bytes));
    }

    println!("{}", series.label());
    println!(
        "  input     {} frames, {} x {}, CFA {}, black {}, white {}",
        frames.len(),
        first.width,
        first.height,
        first.cfa.pattern_str(),
        first.black,
        first.white
    );
    println!(
        "  camera    {} {}  ISO {:.0}  {}  f/{:.1}  {:.0} mm",
        first.make.trim(),
        first.model.trim(),
        first.iso,
        model::shutter_label(first.shutter),
        first.aperture,
        first.focal
    );
    println!("  shifts    {}", shift_summary(&shifts));
    println!(
        "  distort   {}  (corner {:.1} px, half diagonal {:.0} px)",
        if correction.is_identity() {
            "off".to_string()
        } else {
            correction.to_string()
        },
        pipeline::corner_shift(merged.width, merged.height, correction),
        (((merged.width as f64 - 1.0) * 0.5).powi(2)
            + ((merged.height as f64 - 1.0) * 0.5).powi(2))
        .sqrt()
    );
    println!(
        "  output    {} x {}  scale {}  {:.2} MP  {} uncompressed",
        merged.width,
        merged.height,
        merged.scale,
        merged.pixel_count() as f64 / 1e6,
        pipeline::fmt_bytes(merged_bytes(&merged))
    );
    println!(
        "  merged    {:.2} s, {:.2} samples/channel, min coverage {:.2}, filled {} px ({:.3}%)",
        merged.stats.elapsed_ms as f64 / 1000.0,
        merged.stats.mean_samples_per_channel,
        merged.stats.min_coverage,
        merged.stats.filled_pixels,
        100.0 * merged.stats.filled_pixels as f64 / merged.pixel_count().max(1) as f64
    );
    if !merged.stats.masked_pct.is_empty() {
        let worst = merged.stats.masked_pct.iter().cloned().fold(0.0f32, f32::max);
        let mean = merged.stats.masked_pct.iter().sum::<f32>()
            / merged.stats.masked_pct.len().max(1) as f32;
        println!(
            "  motion    masked {:.1} % mean, {:.1} % worst frame",
            mean * 100.0,
            worst * 100.0
        );
    }
    for (path, bytes) in &written {
        println!("  wrote     {} ({})", path.display(), pipeline::fmt_bytes(*bytes));
    }
    println!(
        "  elapsed   {:.2} s",
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn shift_summary(shifts: &[crate::types::Shift]) -> String {
    let worst = shifts
        .iter()
        .map(|shift| shift.dx.abs().max(shift.dy.abs()))
        .fold(0.0f64, f64::max);
    let worst_rot = shifts
        .iter()
        .map(|shift| shift.rot_arcsec().abs())
        .fold(0.0f64, f64::max);
    let worst_rms = shifts
        .iter()
        .map(|shift| shift.residual as f64)
        .fold(0.0f64, f64::max);
    let low_confidence = shifts
        .iter()
        .filter(|shift| shift.confidence < 0.5)
        .count();
    format!(
        "{} frames, max |shift| {:.3} px, max |rot| {:.1}\" , max rms {:.3} px, {} low-confidence",
        shifts.len(),
        worst,
        worst_rot,
        worst_rms,
        low_confidence
    )
}

fn run_inspect(args: InspectArgs) -> Result<()> {
    let series = pipeline::resolve_inputs(&args.inputs)?;
    if series.is_empty() {
        bail!(
            "no Pixel Shift series found in {}",
            args.inputs
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    println!("{} inspect", crate::version());
    let mut failures = 0usize;
    for item in &series {
        match inspect_one(
            item,
            !args.no_align,
            args.distort,
            if args.motion {
                Some(crate::types::MotionParams {
                    enabled: true,
                    sensitivity: args.motion_sensitivity,
                })
            } else {
                None
            },
            args.stabilize,
        ) {
            Ok(()) => {}
            Err(error) => {
                eprintln!("error: {error:#}");
                failures += 1;
            }
        }
    }
    if failures > 0 {
        bail!("{failures} of {} series could not be inspected", series.len());
    }
    Ok(())
}

fn inspect_one(
    series: &SeriesInfo,
    align: bool,
    distort: crate::distort::Distortion,
    motion: Option<crate::types::MotionParams>,
    stabilize: bool,
) -> Result<()> {
    let frames = pipeline::decode(&series.files)?;
    let first = frames
        .first()
        .ok_or_else(|| anyhow::anyhow!("{}: no frames", series.name()))?;
    let shifts = pipeline::estimate_shifts(&frames, align)?;
    if stabilize {
        let used: Vec<usize> = (0..frames.len()).collect();
        let (x0, y0, x1, y1) = crate::merge::stabilized_rect_pub(
            &shifts,
            &used,
            first.width,
            first.height,
        );
        println!(
            "  stabilize  crop to {} x {} of {} x {} (edges {}, {}, {}, {})",
            x1 - x0,
            y1 - y0,
            first.width,
            first.height,
            x0,
            y0,
            first.width - x1,
            first.height - y1
        );
    }
    let masked = motion.map(|params| {
        let masks = crate::motion::masks(&frames, &shifts, &params);
        masks
            .into_iter()
            .map(|m| m.map(|mask| mask.masked_fraction()).unwrap_or(0.0))
            .collect::<Vec<f32>>()
    });
    let scale = ScaleMode::Auto.resolve(frames.len(), series.suggested_scale);
    let expected = pipeline::expected_output(first.width, first.height, scale);

    println!();
    println!("{}", series.dir.display());
    println!(
        "  series    {} ({} frames)",
        series.label(),
        frames.len()
    );
    println!(
        "  camera    {} {}",
        first.make.trim(),
        first.model.trim()
    );
    println!(
        "  frame     {} x {}  CFA {}  black {}  white {}  (stored range 0..{})",
        first.width,
        first.height,
        first.cfa.pattern_str(),
        first.black,
        first.white,
        first.white_span()
    );
    println!(
        "  WB        {:.3} {:.3} {:.3} (camera native, not applied to the output)",
        first.wb[0], first.wb[1], first.wb[2]
    );
    println!(
        "  exposure  ISO {:.0}  {}  f/{:.1}  {:.0} mm",
        first.iso,
        model::shutter_label(first.shutter),
        first.aperture,
        first.focal
    );
    println!("  timestamp {}", first.timestamp);
    println!(
        "  lens      {} ({})  {} mm equiv  max f/{:.1}",
        if first.lens_model.is_empty() {
            "-".to_string()
        } else {
            first.lens_model.clone()
        },
        if first.lens_make.is_empty() {
            "-".to_string()
        } else {
            first.lens_make.clone()
        },
        if first.focal_35mm == 0 {
            "-".to_string()
        } else {
            first.focal_35mm.to_string()
        },
        first.max_aperture
    );
    println!(
        "  distort   {}  (corner {:.1} px)",
        if distort.is_identity() {
            "off".to_string()
        } else {
            distort.to_string()
        },
        pipeline::corner_shift(expected.width, expected.height, distort)
    );
    if series.unexpected_mode() {
        println!(
            "  mode      unexpected ({} frames) -> scale {} chosen from the frame count",
            frames.len(),
            scale
        );
    } else {
        println!(
            "  mode      {} ({} frames) -> scale {}",
            series.mode,
            frames.len(),
            scale
        );
    }
    for (index, shift) in shifts.iter().enumerate() {
        let motion_note = match &masked {
            Some(values) => format!("  mask={:.1}%", values.get(index).copied().unwrap_or(0.0) * 100.0),
            None => String::new(),
        };
        println!(
            "  frame {:2}  dx={:+.3}  dy={:+.3}  rot={:+.1}\"  residual={:.3}  conf={:.2}{}",
            index,
            shift.dx,
            shift.dy,
            shift.rot_arcsec(),
            shift.residual,
            shift.confidence,
            motion_note
        );
    }
    println!(
        "  output    {} x {}  scale {}  {:.2} MP  {} uncompressed 16-bit RGB",
        expected.width,
        expected.height,
        expected.scale,
        expected.megapixels(),
        pipeline::fmt_bytes(expected.bytes())
    );
    if let Some(free) = pipeline::free_space(&series.files[0]) {
        if free < expected.bytes() {
            println!(
                "  warning   only {} free on the output volume: the result needs {}",
                pipeline::fmt_bytes(free),
                pipeline::fmt_bytes(expected.bytes())
            );
        } else {
            println!("  free      {}", pipeline::fmt_bytes(free));
        }
    }
    Ok(())
}

fn output_path(
    out: Option<&Path>,
    series: &SeriesInfo,
    index: usize,
    total: usize,
    scale: u32,
    ext: &str,
) -> Result<PathBuf> {
    let first = series
        .files
        .first()
        .cloned()
        .unwrap_or_else(|| PathBuf::from("input.nef"));
    let default = pipeline::default_output_path(&first, scale, ext);
    match out {
        None => Ok(default),
        Some(path) if total == 1 => Ok(path.to_path_buf()),
        Some(path) => {
            if path.extension().is_some() {
                bail!(
                    "--out must be a directory when several series are merged ({} series)",
                    total
                );
            }
            std::fs::create_dir_all(path)?;
            let name = default
                .file_name()
                .map(|name| name.to_os_string())
                .unwrap_or_default();
            let _ = index;
            Ok(path.join(name))
        }
    }
}

fn merged_bytes(img: &crate::types::MergeResult) -> u64 {
    (img.width as u64)
        .saturating_mul(img.height as u64)
        .saturating_mul(crate::types::CHANNELS as u64)
        .saturating_mul(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_parses_three_coefficients_and_a_distortion_fit() {
        let cli = Cli::try_parse_from(["pixelshift", "merge", "series", "--distort", "0.01,-0.002,0.0001"])
            .expect("parse three coefficients");
        match cli.command {
            Command::Merge(args) => {
                assert_eq!((args.distort.k1, args.distort.k2, args.distort.k3), (0.01, -0.002, 0.0001));
            }
            _ => panic!("wrong subcommand"),
        }
        let cli = Cli::try_parse_from(["pixelshift", "merge", "series", "--distort-fit", "ref.tif"])
            .expect("the default --distort must not conflict with --distort-fit");
        match cli.command {
            Command::Merge(args) => {
                assert_eq!(args.distort_fit.as_deref(), Some(Path::new("ref.tif")));
                assert!(args.distort.is_identity());
            }
            _ => panic!("wrong subcommand"),
        }
    }

    #[test]
    fn cli_parses_the_screenshot_flags() {
        let cli = Cli::try_parse_from([
            "pixelshift",
            "gui",
            "--demo",
            "--distort",
            "-0.040",
            "--shot",
            "out.jpg",
            "--shot-zoom",
            "2.0",
            "--shot-crop",
            "80,52,480,320",
            "--shot-source",
            "reference",
            "--shot-save",
            "--shot-log",
            "--shot-timeout",
            "5",
        ])
        .expect("parse the shot flags");
        match cli.command {
            Command::Gui(args) => {
                let shot = args.shot().expect("--shot builds a request");
                assert_eq!(shot.path, PathBuf::from("out.jpg"));
                assert_eq!(shot.zoom, Some(2.0));
                assert_eq!(shot.crop, Some((80, 52, 480, 320)));
                assert_eq!(shot.source, ViewSource::Reference);
                assert!(shot.save && shot.log);
                assert_eq!(shot.timeout, std::time::Duration::from_secs(5));
                assert_eq!(args.distort.map(|d| d.k1), Some(-0.040));
            }
            _ => panic!("wrong subcommand"),
        }

        let plain = Cli::try_parse_from(["pixelshift", "gui"]).expect("parse a plain gui");
        match plain.command {
            Command::Gui(args) => {
                assert!(args.shot().is_none());
                assert!(args.distort.is_none());
            }
            _ => panic!("wrong subcommand"),
        }

        assert!(Cli::try_parse_from(["pixelshift", "gui", "--shot", "x.jpg", "--shot-crop", "1,2,3"])
            .is_err());
    }

    #[test]
    fn cli_parses_a_plain_merge() {
        let cli = Cli::try_parse_from(["pixelshift", "merge", "series"]).expect("parse");
        match cli.command {
            Command::Merge(args) => {
                assert_eq!(args.scale, ScaleArg::Auto);
                assert_eq!(args.format, FormatArg::Dng);
                assert_eq!(args.compression, CompressionArg::None);
                assert!(!args.no_compress);
                assert_eq!(args.options().compression, Compression::None);
            }
            _ => panic!("wrong subcommand"),
        }
    }

    #[test]
    fn cli_parses_the_gui_appearance_flags() {
        let cli = Cli::try_parse_from([
            "pixelshift",
            "gui",
            "--demo",
            "--lang",
            "ru",
            "--theme",
            "light",
            "--font",
            "builtin",
        ])
        .expect("parse the gui flags");
        match cli.command {
            Command::Gui(args) => {
                assert_eq!(args.lang.map(LangArg::lang), Some(Lang::Ru));
                assert_eq!(args.theme.map(ThemeArg::theme), Some(ThemeMode::Light));
                assert_eq!(args.font.map(FontArg::font), Some(FontMode::Builtin));
            }
            _ => panic!("wrong subcommand"),
        }

        let cli = Cli::try_parse_from(["pixelshift", "gui"]).expect("parse a plain gui");
        match cli.command {
            Command::Gui(args) => {
                assert_eq!(args.lang, None);
                assert_eq!(args.theme, None);
                assert_eq!(args.font, None);
            }
            _ => panic!("wrong subcommand"),
        }

        let cli = Cli::try_parse_from(["pixelshift", "gui", "--lang", "auto", "--theme", "auto"])
            .expect("parse the auto flags");
        match cli.command {
            Command::Gui(args) => {
                assert_eq!(args.lang.map(LangArg::lang), Some(Lang::detect()));
                assert_eq!(args.theme.map(ThemeArg::theme), Some(ThemeMode::System));
            }
            _ => panic!("wrong subcommand"),
        }
    }
}
