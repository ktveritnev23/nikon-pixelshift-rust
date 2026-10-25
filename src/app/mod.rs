//! egui/eframe GUI.

pub mod demo;
pub mod i18n;
pub mod model;
pub mod pipeline;
pub mod prefs;
pub mod selftest;
pub mod style;
pub mod ui;
pub mod worker;

use crate::app::i18n::Lang;
use crate::app::model::{GuiState, Request, Stage, ViewSource, MAX_ZOOM, MIN_ZOOM};
use crate::app::prefs::{FontMode, ThemeMode, UiPrefs};
use crate::app::worker::{Command, Event, Mode, SaveTarget, Worker};
use anyhow::{anyhow, Context, Result};
use eframe::egui;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Per-run overrides of the stored preferences (pixelshift gui --lang ...).
#[derive(Clone, Copy, Debug, Default)]
pub struct PrefsOverrides {
    pub lang: Option<Lang>,
    pub theme: Option<ThemeMode>,
    pub font: Option<FontMode>,
}

/// Options of the GUI entry point, filled in from pixelshift gui ... .
#[derive(Clone, Debug, Default)]
pub struct GuiOptions {
    pub selftest: bool,
    pub demo: bool,
    pub inputs: Vec<PathBuf>,
    pub prefs: PrefsOverrides,
    /// Lens distortion applied to this run only (pixelshift gui --distort).
    pub distort: Option<crate::distort::Distortion>,
    /// Reproducible screenshot mode (pixelshift gui --shot).
    pub shot: Option<ShotRequest>,
}

/// What the screenshot mode captures once the series is merged.
#[derive(Clone, Debug)]
pub struct ShotRequest {
    /// JPEG file to write; the run exits with 0 afterwards.
    pub path: PathBuf,
    /// Canvas zoom; None keeps the fit-to-window view.
    pub zoom: Option<f32>,
    /// Region of the merged image to fetch and show as a 1:1 crop.
    pub crop: Option<(usize, usize, usize, usize)>,
    /// Preview the window shows.
    pub source: ViewSource,
    /// Write the merged result to a temporary file first, so the log shows it.
    pub save: bool,
    /// Start with the log panel expanded.
    pub log: bool,
    /// Deadline for the whole sequence.
    pub timeout: Duration,
}

/// Start the GUI with default options (pixelshift without arguments).
pub fn run() -> Result<()> {
    run_with(GuiOptions::default())
}

pub fn run_with(options: GuiOptions) -> Result<()> {
    if options.selftest {
        return selftest::run(&options);
    }
    let mode = if options.demo { Mode::Demo } else { Mode::Real };
    let native = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("pixelshift")
            .with_inner_size([1400.0, 900.0])
            .with_min_inner_size([900.0, 600.0]),
        ..Default::default()
    };
    let inputs = options.inputs.clone();
    let overrides = options.prefs;
    let distort = options.distort;
    let shot = options.shot.clone();
    eframe::run_native(
        "pixelshift",
        native,
        Box::new(move |cc| {
            Ok(Box::new(PixelShiftApp::new(
                cc, mode, inputs, overrides, distort, shot,
            )))
        }),
    )
    .map_err(|error| anyhow!("could not start the GUI: {error}"))
}

struct PixelShiftApp {
    state: GuiState,
    cache: ui::PreviewCache,
    worker: Worker,
    last_selected: usize,
    /// Paths of the last processed drag-and-drop, to ignore repeat reports.
    last_dropped: Vec<PathBuf>,
    prefs: UiPrefs,
    saved_prefs: UiPrefs,
    overrides: PrefsOverrides,
    prefs_stored: bool,
    /// Set in --shot mode: the capture state machine.
    shot: Option<Shot>,
}

impl PixelShiftApp {
    fn new(
        cc: &eframe::CreationContext<'_>,
        mode: Mode,
        inputs: Vec<PathBuf>,
        overrides: PrefsOverrides,
        distort: Option<crate::distort::Distortion>,
        shot: Option<ShotRequest>,
    ) -> Self {
        let mut state = GuiState::new(mode);
        // --distort overrides the slider for this run: the worker receives it
        // with the load command and nothing is written to the preferences.
        if let Some(distort) = distort {
            state.options.distort = distort;
        }
        if shot.as_ref().map(|shot| shot.log).unwrap_or(false) {
            state.show_log = true;
        }
        let saved_prefs = load_prefs(cc.storage, &mut state);
        let mut prefs = saved_prefs.clone();
        if let Some(lang) = overrides.lang {
            prefs.lang = lang;
        }
        if let Some(theme) = overrides.theme {
            prefs.theme = theme;
        }
        if let Some(font) = overrides.font {
            prefs.font = font;
        }
        // Materialise the editable style files on first launch, then apply
        // the theme/fonts/style. Both before the first egui pass: no light
        // flash when the stored or requested theme is dark.
        if let Some(dir) = style::styles_dir() {
            let _ = std::fs::create_dir_all(&dir);
            let written = style::ensure_builtin_files(&dir);
            for path in written {
                state.push_log(format!("style file created: {}", path.display()));
            }
        }
        prefs.install(&cc.egui_ctx);
        if prefs.style_file.is_none() {
            state.push_log(format!(
                "styles directory: {}",
                style::styles_dir()
                    .map(|d| d.display().to_string())
                    .unwrap_or_else(|| "unavailable".to_string())
            ));
        }
        if let Some(warning) = prefs::take_font_warning() {
            state.push_log(warning);
        }
        let worker = Worker::spawn(mode);
        match mode {
            Mode::Demo => worker.send(Command::Scan(Vec::new())),
            Mode::Real => {
                if !inputs.is_empty() {
                    state.enter_stage(Stage::Scanning);
                    worker.send(Command::Scan(inputs));
                }
            }
        }
        Self {
            state,
            cache: ui::PreviewCache::default(),
            worker,
            last_selected: 0,
            last_dropped: Vec::new(),
            prefs,
            saved_prefs,
            overrides,
            prefs_stored: false,
            shot: shot.map(Shot::new),
        }
    }

    /// Preferences as they should be written back: a value that only came from
    /// a command line flag is replaced by the stored one, so flags never stick.
    fn persisted_prefs(&self) -> UiPrefs {
        let mut prefs = self.prefs.clone();
        if let Some(lang) = self.overrides.lang {
            if prefs.lang == lang {
                prefs.lang = self.saved_prefs.lang;
            }
        }
        if let Some(theme) = self.overrides.theme {
            if prefs.theme == theme {
                prefs.theme = self.saved_prefs.theme;
            }
        }
        if let Some(font) = self.overrides.font {
            if prefs.font == font {
                prefs.font = self.saved_prefs.font;
            }
        }
        prefs
    }

    fn store_prefs(&self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, prefs::STORAGE_KEY, &self.persisted_prefs());
        storage.flush();
    }

    fn poll_worker(&mut self) -> bool {
        let mut finished = false;
        for event in self.worker.poll() {
            if matches!(event, Event::Finished) {
                finished = true;
            }
            self.state.apply(event);
        }
        finished
    }

    fn refresh_free_space(&mut self) {
        let dir = self
            .state
            .selected_series()
            .and_then(|series| series.info.files.first().cloned())
            .map(|path| {
                path.parent()
                    .map(|dir| dir.to_path_buf())
                    .unwrap_or_else(|| PathBuf::from("."))
            })
            .unwrap_or_else(|| PathBuf::from("."));
        self.state.free_space = pipeline::free_space(&dir);
    }

    fn load_selected(&mut self, index: usize, save: Option<SaveTarget>) {
        if self.state.series.is_empty() {
            return;
        }
        pipeline::apply_threads(self.state.options.threads);
        self.state.error = None;
        self.state.enter_stage(Stage::Decoding);
        let options = self.state.series_options(index);
        self.worker.send(Command::Load {
            index,
            options,
            save,
        });
        self.refresh_free_space();
    }

    fn advance_batch(&mut self) {
        if !self.state.batch_running {
            return;
        }
        let next = if self.state.queue.is_empty() {
            None
        } else {
            Some(self.state.queue.remove(0))
        };
        let Some(index) = next else {
            self.state.batch_running = false;
            self.state.push_log("batch finished".to_string());
            return;
        };
        let Some(series) = self.state.series.get(index) else {
            self.state.batch_running = false;
            return;
        };
        if !series.queued {
            // A series unchecked while queued: skip it instead of merging
            // against the user's intent.
            self.advance_batch();
            return;
        }
        let info = series.info.clone();
        let options = self.state.series_options(index);
        let scale = options
            .scale
            .resolve(info.frame_count(), info.suggested_scale);
        let Some(first) = info.files.first() else {
            self.state.batch_running = false;
            return;
        };
        let path = pipeline::default_output_path(first, scale, options.format.ext());
        let save = Some(SaveTarget {
            path,
            format: options.format,
            flavor: options.flavor,
            tiff_render: options.tiff_render,
            also_1x: options.also_1x,
        });
        self.state.selected = index;
        self.state.selected_frame = 0;
        self.last_selected = index;
        self.state.fit_view();
        self.load_selected(index, save);
    }

    fn save_dialog(&mut self, downsample: bool) {
        let format = self.state.options.format;
        let scale = self
            .state
            .selected_series()
            .map(|series| {
                self.state
                    .options
                    .scale
                    .resolve(series.info.frame_count(), series.info.suggested_scale)
            })
            .unwrap_or(1);
        let name = self
            .state
            .default_save_path(if downsample { 1 } else { scale })
            .and_then(|path| path.file_name().map(|n| n.to_string_lossy().into_owned()))
            .unwrap_or_else(|| "pixelshift.dng".to_string());
        let mut dialog = rfd::FileDialog::new()
            .add_filter(format.label(), &[format.ext()])
            .set_file_name(name);
        if let Some(dir) = self
            .state
            .selected_series()
            .and_then(|series| series.info.files.first().cloned())
            .and_then(|path| path.parent().map(|dir| dir.to_path_buf()))
        {
            dialog = dialog.set_directory(dir);
        }
        if let Some(path) = dialog.save_file() {
            self.state
                .requests
                .push(Request::SaveAs { path, downsample });
        }
    }

    fn handle_requests(&mut self, ctx: &egui::Context) {
        for request in self.state.take_requests() {
            match request {
                Request::PickFolder => {
                    if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                        self.state.enter_stage(Stage::Scanning);
                        self.worker.send(Command::Scan(vec![dir]));
                    }
                }
                Request::PickFiles => {
                    if let Some(files) = rfd::FileDialog::new()
                        .add_filter("Nikon raw", &["nef", "NEF", "nrw", "NRW"])
                        .pick_files()
                    {
                        self.state.enter_stage(Stage::Scanning);
                        self.worker.send(Command::Scan(files));
                    }
                }
                Request::Scan(inputs) => {
                    self.state.enter_stage(Stage::Scanning);
                    self.worker.send(Command::Scan(inputs));
                }
                Request::PreviewFirst { index, path } => {
                    self.worker.send(Command::Preview { index, path });
                }
                Request::Load => {
                    let index = self.state.selected;
                    self.load_selected(index, None);
                }
                Request::View { source, rect, step } => {
                    self.worker.send(Command::View { source, rect, step });
                }
                Request::Distort { distort } => {
                    self.worker.send(Command::Distort { distort });
                }
                Request::SetPinned { index, pinned } => {
                    if let Some(series) = self.state.series.get_mut(index) {
                        series.pinned = if pinned {
                            Some(self.state.options.clone())
                        } else {
                            None
                        };
                    }
                }
                Request::PickDistortReference => {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter(
                            "reference image",
                            &["tif", "tiff", "nef", "nefx", "dng", "cr3", "arw"],
                        )
                        .pick_file()
                    {
                        self.worker.send(Command::FitDistort { reference: path });
                    }
                }
                Request::SaveDialog { downsample } => self.save_dialog(downsample),
                Request::SaveAs { path, downsample } => {
                    let format = self.state.options.format;
                    self.worker.send(Command::SaveAs {
                        index: self.state.selected,
                        path,
                        format,
                        flavor: self.state.options.flavor,
                        tiff_render: self.state.options.tiff_render,
                        downsample,
                        also_1x: self.state.options.also_1x,
                        compression: self.state.options.compression,
                    });
                }
                Request::StartBatch => {
                    if self.state.queue.is_empty() {
                        self.state.queue = (0..self.state.series.len())
                            .filter(|&i| self.state.series[i].queued)
                            .collect();
                    }
                    self.state.batch_running = true;
                    self.advance_batch();
                }
                Request::StopBatch => {
                    self.state.batch_running = false;
                    self.state.push_log("batch stopped".to_string());
                }
                Request::Quit => ctx.send_viewport_cmd(egui::ViewportCommand::Close),
            }
        }
    }
}

impl eframe::App for PixelShiftApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let finished = self.poll_worker();
        if self.state.selected != self.last_selected {
            self.last_selected = self.state.selected;
            self.refresh_free_space();
        }
        let mut save_needed = !self.prefs_stored;
        if ui::draw(&mut self.state, &mut self.cache, &mut self.prefs, ui) {
            self.prefs.install(ui.ctx());
            if let Some(warning) = prefs::take_font_warning() {
                self.state.push_log(warning);
            }
            save_needed = true;
        }
        if save_needed {
            if let Some(storage) = frame.storage_mut() {
                self.store_prefs(storage);
                self.prefs_stored = true;
            }
        }
        // Drag-and-drop: folders and raw files dropped anywhere on the window
        // load like the "Open" buttons. egui reports the drop for one frame;
        // the guard keeps a repeated report from re-scanning.
        let dropped = ui.ctx().input(|i| {
            i.raw
                .dropped_files
                .iter()
                .map(|file| file.path().to_path_buf())
                .collect::<Vec<_>>()
        });
        if !dropped.is_empty() && self.last_dropped != dropped {
            self.last_dropped = dropped.clone();
            let (valid, skipped) = split_dropped_paths(&dropped);
            for path in skipped {
                self.state
                    .push_log(format!("skipped {}: not a folder or a raw file", path.display()));
            }
            if !valid.is_empty() {
                self.state.requests.push(crate::app::model::Request::Scan(valid));
            }
        }
        self.handle_requests(ui.ctx());
        if let Some(shot) = self.shot.as_mut() {
            shot.step(&mut self.state, ui.ctx());
        }
        if finished && self.state.batch_running {
            self.advance_batch();
        }
        if self.state.needs_tick() {
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(80));
        }
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, prefs::STORAGE_KEY, &self.persisted_prefs());
    }
}

/// Load the stored preferences; a damaged entry is reported, never fatal.
fn load_prefs(storage: Option<&dyn eframe::Storage>, state: &mut GuiState) -> UiPrefs {
    let Some(storage) = storage else {
        return UiPrefs::default();
    };
    match eframe::get_value::<UiPrefs>(storage, prefs::STORAGE_KEY) {
        Some(prefs) => prefs,
        None => {
            if storage.get_string(prefs::STORAGE_KEY).is_some() {
                state.push_log(
                    "stored ui_prefs could not be decoded; using the defaults".to_string(),
                );
            }
            UiPrefs::default()
        }
    }
}

/// Split dropped paths into loadable inputs (folders and Nikon raw files, the
/// same set the "Open files" dialog accepts) and everything else, preserving
/// order.
fn split_dropped_paths(paths: &[std::path::PathBuf]) -> (Vec<std::path::PathBuf>, Vec<std::path::PathBuf>) {
    let mut valid = Vec::new();
    let mut skipped = Vec::new();
    for path in paths {
        let is_raw = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| matches!(ext.to_ascii_lowercase().as_str(), "nef" | "nrw"))
            .unwrap_or(false);
        if path.is_dir() || (path.is_file() && is_raw) {
            valid.push(path.clone());
        } else {
            skipped.push(path.clone());
        }
    }
    (valid, skipped)
}

#[cfg(test)]
mod tests {
    use super::split_dropped_paths;
    use std::path::PathBuf;

    #[test]
    fn dropped_paths_split_into_folders_raws_and_the_rest() {
        let dir = std::env::temp_dir();
        let raw = dir.join("drop-test.nef");
        std::fs::write(&raw, b"x").expect("write temp nef");
        let text = dir.join("drop-test.txt");
        std::fs::write(&text, b"x").expect("write temp txt");
        let missing_nef = PathBuf::from("/definitely/missing/series.NEF");
        let (valid, skipped) = split_dropped_paths(&[
            dir.clone(),
            raw.clone(),
            text.clone(),
            missing_nef.clone(),
        ]);
        assert_eq!(valid, vec![dir, raw]);
        // A raw-sounding path that does not exist is not loadable either.
        assert_eq!(skipped, vec![text, missing_nef]);
    }
}
// ---------------------------------------------------------------- shot ------

/// Phases of the screenshot mode; each one waits for the application to reach a
/// state the previous step asked for, so nothing depends on a fixed delay.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ShotPhase {
    /// Waiting for the merge to finish and the base preview to arrive.
    Merging,
    /// Waiting for the requested view, crop and save to settle.
    Settling,
    /// Waiting for the compositor to hand back the captured frame.
    Capturing,
}

impl ShotPhase {
    fn label(self) -> &'static str {
        match self {
            ShotPhase::Merging => "waiting for the merge",
            ShotPhase::Settling => "waiting for the view to settle",
            ShotPhase::Capturing => "waiting for the screenshot",
        }
    }
}

/// Drives the screenshot mode from the frame callback: the same state machine
/// the window uses, plus a view request, an optional save and the framebuffer
/// grab.
struct Shot {
    request: ShotRequest,
    started: Instant,
    phase: ShotPhase,
    /// Consecutive frames the canvas has been idle: two mean it settled.
    settled: u32,
    configured: Instant,
    /// Crop window handed to the worker; the shot waits until it is cached.
    crop: Option<(usize, usize, usize, usize)>,
    reference_requested: bool,
    save_pending: bool,
    capture: Arc<Mutex<Option<Arc<egui::ColorImage>>>>,
    capture_requested: bool,
}

/// The canvas debounces crop requests; the shot has to outlast that debounce
/// before it may call the view settled.
const SETTLE_DELAY: Duration = Duration::from_millis(400);
/// Repaint cadence while a shot waits for the worker or the framebuffer.
const SHOT_TICK: Duration = Duration::from_millis(40);

impl Shot {
    fn new(request: ShotRequest) -> Self {
        Self {
            request,
            started: Instant::now(),
            phase: ShotPhase::Merging,
            settled: 0,
            configured: Instant::now(),
            crop: None,
            reference_requested: false,
            save_pending: false,
            capture: Arc::new(Mutex::new(None)),
            capture_requested: false,
        }
    }

    /// One step per frame, after the widgets ran and the worker commands were
    /// dispatched, so a view change here is painted by the next frame.
    fn step(&mut self, state: &mut GuiState, ctx: &egui::Context) {
        if self.started.elapsed() > self.request.timeout {
            eprintln!(
                "pixelshift: --shot timed out after {:.1} s ({}): no JPEG was written to {}",
                self.request.timeout.as_secs_f64(),
                self.phase.label(),
                self.request.path.display()
            );
            std::process::exit(1);
        }
        // Keep painting even when the window ends up behind another one: without
        // this the merge would stall (eframe runs no pass for an occluded
        // window) and the shot would only hit the timeout.
        ctx.send_viewport_cmd(egui::ViewportCommand::RequestPaintWhileHidden);
        match self.phase {
            ShotPhase::Merging => self.wait_merged(state),
            ShotPhase::Settling => self.wait_settled(state, ctx),
            ShotPhase::Capturing => self.wait_frame(ctx),
        }
    }

    fn wait_merged(&mut self, state: &mut GuiState) {
        if let Some(error) = &state.error {
            // A failed load cannot produce the picture; report it instead of
            // waiting for the timeout.
            eprintln!(
                "pixelshift: --shot stopped, the merge failed: {error} (no JPEG was written)"
            );
            std::process::exit(1);
        }
        let merged = state
            .selected_series()
            .and_then(|series| series.result)
            .is_some();
        if !merged || state.stage != Stage::Done {
            return;
        }
        if self.request.source == ViewSource::Reference && !self.reference_requested {
            self.reference_requested = true;
            state.requests.push(Request::View {
                source: ViewSource::Reference,
                rect: None,
                step: 1,
            });
        }
        let ready = state.previews[self.request.source.index()]
            .as_ref()
            .map(|slot| slot.image.width > 0 && slot.image.height > 0)
            .unwrap_or(false);
        if !ready {
            return;
        }
        self.configure(state);
        self.phase = ShotPhase::Settling;
    }

    /// Leave the fit-to-window view and ask for the crop and the save the shot
    /// asked for; the widget code picks the new view up on the next frame.
    fn configure(&mut self, state: &mut GuiState) {
        let dims = state.image_size().unwrap_or((0, 0));
        state.view.source = self.request.source;
        state.view.fitted = false;
        state.view.crop = self.request.crop.is_some();
        match self.request.crop {
            Some((x, y, w, h)) => {
                let zoom = self.request.zoom.unwrap_or(1.0);
                state.view.zoom = zoom.clamp(MIN_ZOOM, MAX_ZOOM);
                state.view.center = [x as f32 + w as f32 * 0.5, y as f32 + h as f32 * 0.5];
                self.crop = Some((x, y, w, h));
                state.requests.push(Request::View {
                    source: self.request.source,
                    rect: Some((x, y, w, h)),
                    step: 1,
                });
            }
            None => match self.request.zoom {
                Some(zoom) => {
                    state.view.zoom = zoom.clamp(MIN_ZOOM, MAX_ZOOM);
                    state.view.center = [dims.0 as f32 * 0.5, dims.1 as f32 * 0.5];
                }
                None => state.fit_view(),
            },
        }
        if self.request.save {
            let path = std::env::temp_dir().join(format!(
                "pixelshift-shot-{}.{}",
                std::process::id(),
                state.options.format.ext()
            ));
            state.requests.push(Request::SaveAs {
                path,
                downsample: false,
            });
            self.save_pending = true;
        }
        self.configured = Instant::now();
        self.settled = 0;
    }

    fn wait_settled(&mut self, state: &mut GuiState, ctx: &egui::Context) {
        let crop_ok = match self.crop {
            Some(rect) => state
                .crop_for(Some(crate::app::model::CropWant { rect, step: 1 }))
                .is_some(),
            None => true,
        };
        let save_ok = !self.save_pending
            || state
                .selected_series()
                .map(|series| !series.saved.is_empty())
                .unwrap_or(false);
        let idle = !state.is_busy() && !state.crop_pending;
        let warm = self.configured.elapsed() >= SETTLE_DELAY;
        if crop_ok && save_ok && idle && warm {
            self.settled += 1;
        } else {
            self.settled = 0;
        }
        if self.settled >= 2 {
            self.phase = ShotPhase::Capturing;
            ctx.request_repaint();
        } else {
            ctx.request_repaint_after(SHOT_TICK);
        }
    }

    fn wait_frame(&mut self, ctx: &egui::Context) {
        if let Some(image) = self.captured() {
            match self.write(&image, ctx) {
                Ok(line) => {
                    println!("{line}");
                    std::process::exit(0);
                }
                Err(error) => {
                    eprintln!(
                        "pixelshift: --shot could not write {}: {error:#}",
                        self.request.path.display()
                    );
                    std::process::exit(1);
                }
            }
        }
        if !self.capture_requested {
            let slot = Arc::clone(&self.capture);
            let repaint = ctx.clone();
            ctx.request_screenshot(move |image| {
                if let Ok(mut guard) = slot.lock() {
                    *guard = Some(image);
                }
                repaint.request_repaint();
            });
            self.capture_requested = true;
        }
        ctx.request_repaint_after(SHOT_TICK);
    }

    fn captured(&self) -> Option<Arc<egui::ColorImage>> {
        self.capture.lock().ok().and_then(|slot| slot.clone())
    }

    fn write(&self, image: &egui::ColorImage, ctx: &egui::Context) -> Result<String> {
        let (width, height, rgb) = shrink(image, ctx.pixels_per_point());
        if width == 0 || height == 0 {
            return Err(anyhow!("the captured frame is empty"));
        }
        let bytes = crate::write::jpeg::encode_rgb(&rgb, width, height);
        if let Some(dir) = self.request.path.parent() {
            if !dir.as_os_str().is_empty() {
                std::fs::create_dir_all(dir)
                    .with_context(|| format!("creating {}", dir.display()))?;
            }
        }
        std::fs::write(&self.request.path, &bytes)
            .with_context(|| format!("writing {}", self.request.path.display()))?;
        Ok(format!(
            "shot: {} ({} x {} px, {})",
            self.request.path.display(),
            width,
            height,
            pipeline::fmt_bytes(bytes.len() as u64)
        ))
    }
}

/// Physical screenshot pixels to logical pixels: the framebuffer has
/// pixels_per_point times as many pixels as the window has points, so the frame
/// is box filtered back to the size the window reports. Each output pixel
/// averages exactly its share of the source, which keeps Retina (2.0) and
/// fractional scale factors alike.
fn shrink(image: &egui::ColorImage, pixels_per_point: f32) -> (usize, usize, Vec<u8>) {
    let (width, height) = (image.size[0], image.size[1]);
    if width == 0 || height == 0 {
        return (0, 0, Vec::new());
    }
    let ppp = if pixels_per_point.is_finite() && pixels_per_point > 0.0 {
        pixels_per_point
    } else {
        1.0
    };
    let out_w = ((width as f32 / ppp).round() as usize).clamp(1, width);
    let out_h = ((height as f32 / ppp).round() as usize).clamp(1, height);
    let step_x = width as f32 / out_w as f32;
    let step_y = height as f32 / out_h as f32;
    let mut rgb = vec![0u8; out_w * out_h * 3];
    for oy in 0..out_h {
        let y0 = oy as f32 * step_y;
        let y1 = y0 + step_y;
        for ox in 0..out_w {
            let x0 = ox as f32 * step_x;
            let x1 = x0 + step_x;
            let mut sum = [0.0f32; 3];
            let mut weight = 0.0f32;
            let mut y = y0.floor().max(0.0) as usize;
            while (y as f32) < y1 && y < height {
                let wy = ((y as f32 + 1.0).min(y1) - (y as f32).max(y0)).max(0.0);
                let mut x = x0.floor().max(0.0) as usize;
                while wy > 0.0 && (x as f32) < x1 && x < width {
                    let wx = ((x as f32 + 1.0).min(x1) - (x as f32).max(x0)).max(0.0);
                    if wx > 0.0 {
                        let color = image.pixels[y * width + x];
                        let tap = wx * wy;
                        sum[0] += f32::from(color.r()) * tap;
                        sum[1] += f32::from(color.g()) * tap;
                        sum[2] += f32::from(color.b()) * tap;
                        weight += tap;
                    }
                    x += 1;
                }
                y += 1;
            }
            let index = (oy * out_w + ox) * 3;
            if weight > 0.0 {
                for channel in 0..3 {
                    rgb[index + channel] = (sum[channel] / weight).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    (out_w, out_h, rgb)
}
