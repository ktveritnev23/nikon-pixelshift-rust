//! GUI state, view geometry and the reducer that turns worker events into
//! widget visible state. No drawing code lives here, so the whole state machine
//! can be driven headlessly by the selftest and by unit tests.

use crate::app::pipeline::{self, ResultInfo, SeriesInfo};
use crate::app::worker::{Event, Mode};
use crate::types::{MergeParams, MergeStats, MotionParams, PreviewImage, Shift};
use crate::write::DngFlavor;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Longest side of the cached base preview. 2048 px keeps the whole frame sharp
/// up to roughly 1:3 of the image resolution and is what the detail crop then
/// has to beat.
pub const MAX_PREVIEW_DIM: u32 = 2048;
/// Number of detail crops kept around; panning reuses them instead of asking
/// the worker again.
pub const CROP_CACHE_LEN: usize = 4;
/// Delay between the last zoom/pan and the crop request, so dragging does not
/// queue a render per frame.
pub const CROP_DEBOUNCE: Duration = Duration::from_millis(160);
/// Delay between the last slider move and the distortion recompute.
pub const DISTORT_DEBOUNCE: Duration = Duration::from_millis(200);
pub const MIN_ZOOM: f32 = 0.02;
pub const MAX_ZOOM: f32 = 16.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleMode {
    Auto,
    One,
    Two,
}

impl ScaleMode {
    pub const ALL: [ScaleMode; 3] = [ScaleMode::Auto, ScaleMode::One, ScaleMode::Two];

    pub fn label(self) -> &'static str {
        match self {
            ScaleMode::Auto => "auto",
            ScaleMode::One => "1x",
            ScaleMode::Two => "2x",
        }
    }

    /// Resolve the output grid density for a series of frame_count frames.
    pub fn resolve(self, frame_count: usize, suggested: u32) -> u32 {
        match self {
            ScaleMode::One => 1,
            ScaleMode::Two => 2,
            ScaleMode::Auto => {
                if suggested >= 1 {
                    suggested
                } else if frame_count >= 16 {
                    2
                } else {
                    1
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutFormat {
    Dng,
    Tiff,
}

impl OutFormat {
    pub const ALL: [OutFormat; 2] = [OutFormat::Dng, OutFormat::Tiff];

    pub fn ext(self) -> &'static str {
        match self {
            OutFormat::Dng => "dng",
            OutFormat::Tiff => "tiff",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            OutFormat::Dng => "DNG (linear)",
            OutFormat::Tiff => "TIFF 16-bit",
        }
    }

    /// Whether the writer compresses the image data: lossless JPEG for the
    /// DNG (Compression = 7, what LibRaw and the Adobe readers decode), Adobe
    /// Deflate with the horizontal predictor for the TIFF.
    pub fn use_compression(self, compression: Compression) -> bool {
        compression == Compression::Lossless
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compression {
    None,
    /// DNG: lossless JPEG (Compression = 7); TIFF: AdobeDeflate + Predictor 2.
    Lossless,
}

impl Compression {
    pub const ALL: [Compression; 2] = [Compression::None, Compression::Lossless];

    pub fn label(self) -> &'static str {
        match self {
            Compression::None => "none",
            Compression::Lossless => "lossless",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EditOptions {
    pub scale: ScaleMode,
    pub format: OutFormat,
    pub compression: Compression,
    /// Photometric interpretation of the DNG main image.
    pub flavor: DngFlavor,
    /// Render the 16-bit TIFF as sRGB instead of storing linear camera data.
    pub tiff_render: bool,
    /// Strength of the merge deconvolution, see `MergeParams::deconvolve`.
    pub deconvolve: f32,
    /// Mask scene motion out of the merge, see `MergeParams::motion`.
    pub motion_enabled: bool,
    /// Motion detection sensitivity (1.0 = the tuned default).
    pub motion_sensitivity: f32,
    /// Lens distortion correction applied to the merged result (preview and
    /// export alike).
    pub distort: crate::distort::Distortion,
    pub also_1x: bool,
    pub align: bool,
    pub threads: Option<usize>,
}

impl Default for EditOptions {
    fn default() -> Self {
        Self {
            scale: ScaleMode::Auto,
            format: OutFormat::Dng,
            compression: Compression::None,
            flavor: DngFlavor::LinearRaw,
            tiff_render: true,
            deconvolve: MergeParams::default().deconvolve,
            motion_enabled: false,
            motion_sensitivity: 1.0,
            distort: crate::distort::Distortion::IDENTITY,
            also_1x: false,
            align: true,
            threads: None,
        }
    }
}

impl EditOptions {
    pub fn merge_params(&self, scale: u32, shifts: Vec<Shift>) -> MergeParams {
        MergeParams {
            scale,
            align: self.align,
            shifts,
            deconvolve: self.deconvolve,
            motion: MotionParams {
                enabled: self.motion_enabled,
                sensitivity: self.motion_sensitivity,
            },
            ..MergeParams::default()
        }
    }

    /// Note shown when the requested compression cannot be applied as asked.
    pub fn compression_note(&self) -> Option<&'static str> {
        match (self.format, self.compression) {
            (_, Compression::None) => None,
            (OutFormat::Dng, Compression::Lossless) => Some(
                "lossless JPEG: the size estimate above assumes no compression, expect roughly half",
            ),
            (OutFormat::Tiff, Compression::Lossless) => Some(
                "AdobeDeflate + predictor: the size estimate above assumes no compression, expect roughly half",
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Idle,
    Scanning,
    Decoding,
    Aligning,
    Merging,
    Previewing,
    Saving,
    Done,
}

impl Stage {
    pub fn label(self) -> &'static str {
        match self {
            Stage::Idle => "idle",
            Stage::Scanning => "scanning",
            Stage::Decoding => "decoding",
            Stage::Aligning => "aligning",
            Stage::Merging => "merging",
            Stage::Previewing => "preview",
            Stage::Saving => "saving",
            Stage::Done => "done",
        }
    }

    pub fn is_busy(self) -> bool {
        matches!(
            self,
            Stage::Scanning | Stage::Decoding | Stage::Aligning | Stage::Merging | Stage::Saving
        )
    }
}

/// Which image the canvas shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewSource {
    Merged,
    Reference,
}

impl ViewSource {
    pub fn label(self) -> &'static str {
        match self {
            ViewSource::Merged => "merged",
            ViewSource::Reference => "reference",
        }
    }

    pub fn index(self) -> usize {
        match self {
            ViewSource::Merged => 0,
            ViewSource::Reference => 1,
        }
    }
}

/// One decoded frame as the GUI shows it.
#[derive(Clone, Debug, PartialEq)]
pub struct FrameInfo {
    pub path: PathBuf,
    pub width: usize,
    pub height: usize,
    pub cfa: String,
    pub black: u16,
    pub white: u16,
    pub wb: [f32; 3],
    pub iso: f32,
    pub shutter: f32,
    pub aperture: f32,
    pub focal: f32,
    pub camera: String,
}

impl FrameInfo {
    pub fn file_name(&self) -> String {
        self.path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| self.path.display().to_string())
    }

    pub fn exposure_line(&self) -> String {
        format!(
            "ISO {:.0}  {}  f/{:.1}  {:.0} mm",
            self.iso,
            shutter_label(self.shutter),
            self.aperture,
            self.focal
        )
    }
}

pub fn shutter_label(shutter: f32) -> String {
    if shutter <= 0.0 {
        "-".to_string()
    } else if shutter >= 1.0 {
        format!("{:.1} s", shutter)
    } else {
        format!("1/{:.0} s", 1.0 / shutter)
    }
}

/// One series as shown in the left panel.
#[derive(Clone, Debug)]
pub struct SeriesView {
    pub info: SeriesInfo,
    pub frames: Vec<FrameInfo>,
    pub shifts: Vec<Shift>,
    pub result: Option<ResultInfo>,
    pub stats: Option<MergeStats>,
    pub loaded: bool,
    pub saved: Vec<(PathBuf, u64)>,
}

impl SeriesView {
    pub fn new(info: SeriesInfo) -> Self {
        Self {
            info,
            frames: Vec::new(),
            shifts: Vec::new(),
            result: None,
            stats: None,
            loaded: false,
            saved: Vec::new(),
        }
    }

    pub fn shift_for(&self, index: usize) -> Shift {
        self.shifts.get(index).copied().unwrap_or_default()
    }
}

/// One cached preview image plus the region of the full resolution image it
/// covers (None means the whole image), the view it belongs to and the
/// pyramid step it was rendered at (1 = full resolution samples).
#[derive(Clone)]
pub struct PreviewSlot {
    pub image: PreviewImage,
    pub source: ViewSource,
    pub rect: Option<(usize, usize, usize, usize)>,
    pub step: u32,
    pub generation: u64,
}

/// A crop the canvas wants: image rectangle (full resolution coordinates,
/// before step alignment) plus the pyramid step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CropWant {
    pub rect: (usize, usize, usize, usize),
    pub step: u32,
}

/// Native-pixel alignment factor of a crop request. Merged crops tile the
/// output grid at `step` pixels per block; reference crops tile the CFA block
/// grid, which is 2 native pixels per block.
pub fn crop_alignment(source: ViewSource, step: u32) -> usize {
    match source {
        ViewSource::Merged => step.max(1) as usize,
        ViewSource::Reference => 2 * step.max(1) as usize,
    }
}

/// Expand the rectangle outward to whole `factor`-pixel blocks so reduced
/// step crops cover complete samples; blocks clipped by the image border stay
/// partial and the renderers average only the real samples.
pub fn align_rect(
    rect: (usize, usize, usize, usize),
    factor: usize,
    dims: (usize, usize),
) -> (usize, usize, usize, usize) {
    if factor <= 1 || dims.0 == 0 || dims.1 == 0 {
        return rect;
    }
    let x0 = rect.0.min(dims.0.saturating_sub(1)) / factor * factor;
    let y0 = rect.1.min(dims.1.saturating_sub(1)) / factor * factor;
    let x1 = (((rect.0 + rect.2).min(dims.0)).max(x0 + 1) + factor - 1) / factor * factor;
    let y1 = (((rect.1 + rect.3).min(dims.1)).max(y0 + 1) + factor - 1) / factor * factor;
    (x0, y0, x1.min(dims.0).max(x0 + 1) - x0, y1.min(dims.1).max(y0 + 1) - y0)
}

/// Which tab of the left panel is open. Session state (not persisted): the
/// tab returns to "Series" on the next launch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LeftTab {
    #[default]
    Series,
    Params,
    Settings,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CanvasView {
    pub zoom: f32,
    /// Image coordinates of the point shown at the centre of the canvas.
    pub center: [f32; 2],
    /// Show a 1:1 crop of the merged result instead of the fitted preview.
    pub crop: bool,
    pub source: ViewSource,
    /// Overlay a straight grid on the image, for manual lens alignment.
    pub grid: bool,
    /// Zoom and centre still follow the fit-to-window rule.
    pub fitted: bool,
}

impl Default for CanvasView {
    fn default() -> Self {
        Self {
            zoom: 0.0,
            center: [0.0, 0.0],
            crop: false,
            source: ViewSource::Merged,
            grid: false,
            fitted: true,
        }
    }
}

/// Something the widgets want the application to do (dialogs, worker commands).
#[derive(Clone, Debug, PartialEq)]
pub enum Request {
    PickFolder,
    PickFiles,
    Scan(Vec<PathBuf>),
    Load,
    View {
        source: ViewSource,
        rect: Option<(usize, usize, usize, usize)>,
        step: u32,
    },
    Distort {
        distort: crate::distort::Distortion,
    },
    PickDistortReference,
    SaveDialog {
        downsample: bool,
    },
    SaveAs {
        path: PathBuf,
        downsample: bool,
    },
    StartBatch,
    StopBatch,
    Quit,
}

pub struct GuiState {
    pub mode: Mode,
    pub options: EditOptions,
    /// Which tab of the left panel is open.
    pub left_tab: LeftTab,
    pub series: Vec<SeriesView>,
    pub selected: usize,
    pub selected_frame: usize,
    pub stage: Stage,
    pub progress: f32,
    pub status: String,
    pub log: Vec<String>,
    pub error: Option<String>,
    pub previews: [Option<PreviewSlot>; 2],
    /// Most recent detail crops first (LRU, CROP_CACHE_LEN entries).
    pub crops: Vec<PreviewSlot>,
    pub view: CanvasView,
    pub crop_pending: bool,
    pub requests: Vec<Request>,
    pub queue: Vec<usize>,
    pub batch_running: bool,
    pub stage_log: Vec<(Stage, Duration)>,
    pub auto_run: bool,
    pub show_log: bool,
    pub free_space: Option<u64>,
    stage_started: Option<Instant>,
    generation: u64,
    /// Wanted crop and the instant the debounce expires.
    pending_crop: Option<(Option<CropWant>, Instant)>,
    /// Crop request already handed to the worker and not answered yet.
    crop_inflight: Option<Option<CropWant>>,
    /// Coefficients the worker currently uses.
    distort_committed: crate::distort::Distortion,
    /// Last measured fit, shown next to the distortion controls.
    pub last_fit: Option<crate::fit::FitReport>,
    /// Last slider move, for the distortion debounce.
    distort_edited: Option<Instant>,
}

impl GuiState {
    pub fn new(mode: Mode) -> Self {
        Self {
            mode,
            options: EditOptions::default(),
            left_tab: LeftTab::default(),
            series: Vec::new(),
            selected: 0,
            selected_frame: 0,
            stage: Stage::Idle,
            progress: 0.0,
            status: "open a folder with a Pixel Shift series".to_string(),
            log: Vec::new(),
            error: None,
            previews: [None, None],
            crops: Vec::new(),
            view: CanvasView::default(),
            crop_pending: false,
            pending_crop: None,
            crop_inflight: None,
            distort_committed: crate::distort::Distortion::IDENTITY,
            last_fit: None,
            distort_edited: None,
            requests: Vec::new(),
            queue: Vec::new(),
            batch_running: false,
            stage_log: Vec::new(),
            auto_run: mode == Mode::Demo,
            show_log: false,
            free_space: None,
            stage_started: None,
            generation: 0,
        }
    }

    pub fn selected_series(&self) -> Option<&SeriesView> {
        self.series.get(self.selected)
    }

    pub fn selected_series_mut(&mut self) -> Option<&mut SeriesView> {
        self.series.get_mut(self.selected)
    }

    pub fn is_busy(&self) -> bool {
        self.stage.is_busy() || self.crop_pending
    }

    /// True while a debounce timer has to run, so the UI must keep repainting
    /// even though the worker is idle.
    pub fn needs_tick(&self) -> bool {
        self.is_busy() || self.pending_crop.is_some() || self.distort_edited.is_some()
    }

    /// Drop the cached detail crops: the image they were cut from changed (new
    /// merge, new lens coefficients, new flavor).
    pub fn invalidate_crops(&mut self) {
        self.crops.clear();
        self.pending_crop = None;
        self.crop_inflight = None;
    }

    /// Align a wanted crop onto the block grid of the active source, so the
    /// request, the cache entry and the coverage check all describe the same
    /// rectangle.
    fn aligned_want(&self, want: CropWant) -> CropWant {
        let factor = crop_alignment(self.view.source, want.step);
        match self.image_size() {
            Some(dims) => CropWant {
                rect: align_rect(want.rect, factor, dims),
                step: want.step,
            },
            None => want,
        }
    }

    /// Cached crop that covers the wanted window, smallest one first. Only
    /// crops of the active view qualify, and only when they are at least as
    /// detailed as the wanted step (a finer cached crop is a valid stand-in).
    pub fn crop_for(&self, wanted: Option<CropWant>) -> Option<&PreviewSlot> {
        let wanted = wanted.map(|w| self.aligned_want(w))?;
        self.crops
            .iter()
            .filter(|slot| {
                slot.source == self.view.source
                    && slot.step <= wanted.step
                    && slot
                        .rect
                        .map(|rect| covers(rect, wanted.rect))
                        .unwrap_or(false)
            })
            .min_by_key(|slot| slot.rect.map(|rect| rect.2 * rect.3).unwrap_or(usize::MAX))
    }

    /// Ask the worker for a crop of the visible window, after the
    /// debounce and only when the cached crops do not already cover it.
    /// Returns true while a crop is on its way or being computed.
    pub fn poll_lod(&mut self, wanted: Option<CropWant>, now: Instant) -> bool {
        let wanted = wanted.map(|w| self.aligned_want(w));
        let covered = wanted.is_none() || self.crop_for(wanted).is_some();
        if covered {
            self.pending_crop = None;
            if self.crop_inflight.is_none() {
                self.crop_pending = false;
            }
            return self.crop_pending;
        }
        if self.crop_inflight == Some(wanted) {
            return true;
        }
        match self.pending_crop {
            Some((pending, deadline)) if pending == wanted => {
                if now >= deadline {
                    self.pending_crop = None;
                    self.crop_inflight = Some(wanted);
                    self.crop_pending = true;
                    if let Some(want) = wanted {
                        self.requests.push(Request::View {
                            source: self.view.source,
                            rect: Some(want.rect),
                            step: want.step,
                        });
                    }
                }
            }
            _ => {
                self.pending_crop = Some((wanted, now + CROP_DEBOUNCE));
                self.crop_pending = false;
            }
        }
        self.crop_pending
    }

    /// True while a distortion slider is being dragged (between the first
    /// move and the debounce commit): the canvas shows the alignment grid.
    pub fn distortion_being_edited(&self) -> bool {
        self.distort_edited.is_some()
    }

    /// True when the live distortion preview has to be recomputed now.
    pub fn poll_distort(&mut self, now: Instant) -> bool {
        if self.options.distort == self.distort_committed {
            self.distort_edited = None;
            return false;
        }
        match self.distort_edited {
            Some(edited) if now.duration_since(edited) >= DISTORT_DEBOUNCE => {
                self.distort_edited = None;
                self.distort_committed = self.options.distort;
                self.invalidate_crops();
                self.crop_pending = true;
                true
            }
            Some(_) => false,
            None => {
                self.distort_edited = Some(now);
                false
            }
        }
    }

    pub fn take_requests(&mut self) -> Vec<Request> {
        std::mem::take(&mut self.requests)
    }

    pub fn push_log(&mut self, message: impl Into<String>) {
        self.log.push(message.into());
        if self.log.len() > 500 {
            self.log.drain(0..100);
        }
    }

    pub fn fail(&mut self, message: impl Into<String>) {
        let message = message.into();
        self.push_log(format!("error: {message}"));
        self.error = Some(message);
        self.batch_running = false;
        self.enter_stage(Stage::Done);
    }

    /// Switch the state machine to a new stage, closing the previous one into
    /// the timing log.
    pub fn enter_stage(&mut self, stage: Stage) {
        if self.stage == stage {
            return;
        }
        if let Some(started) = self.stage_started.take() {
            if self.stage != Stage::Idle {
                self.stage_log.push((self.stage, started.elapsed()));
            }
        }
        self.stage = stage;
        self.progress = if matches!(stage, Stage::Idle | Stage::Done) {
            1.0
        } else {
            0.0
        };
        self.stage_started = if stage == Stage::Done {
            None
        } else {
            Some(Instant::now())
        };
        self.status = stage.label().to_string();
    }

    pub fn stage_time(&self, stage: Stage) -> Option<Duration> {
        let mut total = Duration::ZERO;
        let mut seen = false;
        for (s, d) in &self.stage_log {
            if *s == stage {
                total += *d;
                seen = true;
            }
        }
        seen.then_some(total)
    }

    pub fn total_time(&self) -> Duration {
        self.stage_log.iter().map(|(_, d)| *d).sum()
    }

    pub fn apply(&mut self, event: Event) {
        match event {
            Event::Stage(stage) => self.enter_stage(stage),
            Event::Progress(fraction) => self.progress = fraction.clamp(0.0, 1.0),
            Event::Log(message) => self.push_log(message),
            Event::Scanned(list) => {
                self.series = list.into_iter().map(SeriesView::new).collect();
                self.selected = self.selected.min(self.series.len().saturating_sub(1));
                self.selected_frame = 0;
                self.previews = [None, None];
                self.invalidate_crops();
                self.error = None;
                self.progress = 0.0;
                self.enter_stage(Stage::Idle);
                self.push_log(format!("found {} series", self.series.len()));
                if self.auto_run && !self.series.is_empty() {
                    self.auto_run = false;
                    self.requests.push(Request::Load);
                }
            }
            Event::Decoded { index, frames } => {
                let note = self
                    .series
                    .get(index)
                    .map(|series| format!("decoded {} frames of {}", frames.len(), series.info.name()));
                if let Some(note) = note {
                    self.push_log(note);
                }
                if let Some(series) = self.series.get_mut(index) {
                    series.frames = frames;
                }
            }
            Event::Aligned { index, shifts } => {
                let worst = shifts
                    .iter()
                    .map(|s| s.dx.abs().max(s.dy.abs()))
                    .fold(0.0f64, f64::max);
                self.push_log(format!(
                    "aligned {} frames (max |shift| {:.3} px)",
                    shifts.len(),
                    worst
                ));
                if let Some(series) = self.series.get_mut(index) {
                    series.shifts = shifts;
                }
            }
            Event::Merged { index, info, stats } => {
                let motion = if stats.masked_pct.is_empty() {
                    String::new()
                } else {
                    let worst = stats.masked_pct.iter().cloned().fold(0.0f32, f32::max);
                    format!(", motion masked up to {:.1} %", worst * 100.0)
                };
                self.push_log(format!(
                    "merged {}x{} at scale {}, {:.2} MP, {:.2} samples/channel{}",
                    info.width,
                    info.height,
                    info.scale,
                    info.megapixels(),
                    stats.mean_samples_per_channel,
                    motion
                ));
                if let Some(series) = self.series.get_mut(index) {
                    series.result = Some(info);
                    if stats.masked_pct.len() == series.shifts.len() {
                        for (shift, masked) in series.shifts.iter_mut().zip(stats.masked_pct.iter())
                        {
                            shift.masked_pct = *masked;
                        }
                    }
                    series.stats = Some(stats);
                    series.loaded = true;
                }
                self.view.fitted = true;
                self.view.zoom = 0.0;
                self.invalidate_crops();
            }
            Event::Preview {
                source,
                rect,
                step,
                image,
            } => {
                self.generation += 1;
                let slot = PreviewSlot {
                    image,
                    source,
                    rect,
                    step,
                    generation: self.generation,
                };
                match rect {
                    None => self.previews[source.index()] = Some(slot),
                    Some(rect) => {
                        // Newest first; cached copies of the same window that
                        // are coarser or equal are superseded, finer ones stay.
                        self.crops.retain(|cached| {
                            !(cached.source == source
                                && cached.rect == Some(rect)
                                && cached.step >= step)
                        });
                        self.crops.insert(0, slot);
                        self.crops.truncate(CROP_CACHE_LEN);
                    }
                }
                if source == self.view.source {
                    self.crop_pending = false;
                    self.crop_inflight = None;
                }
            }
            Event::Saved { index, path, bytes } => {
                if let Some(series) = self.series.get_mut(index) {
                    series.saved.push((path.clone(), bytes));
                    self.push_log(format!(
                        "wrote {} ({})",
                        path.display(),
                        pipeline::fmt_bytes(bytes)
                    ));
                }
            }
            Event::Finished => {
                self.enter_stage(Stage::Done);
                self.status = "done".to_string();
                self.progress = 1.0;
            }
            Event::Fitted(report) => {
                self.push_log(format!("distortion fit: {}", report.summary()));
                self.last_fit = Some(report);
                self.options.distort = report.distortion();
            }
            Event::Error(message) => self.fail(message),
        }
    }

    /// Dimensions of the full resolution image of the active source.
    pub fn image_size(&self) -> Option<(usize, usize)> {
        let series = self.selected_series()?;
        match self.view.source {
            ViewSource::Merged => series.result.map(|r| (r.width, r.height)),
            ViewSource::Reference => series.frames.first().map(|f| (f.width, f.height)),
        }
    }

    /// Sampling grid of the active source: output pixels for the merged view,
    /// 2x2 CFA blocks for the reference view. The canvas picks the pyramid
    /// step against this grid so both views hand the worker a similarly sized
    /// texture.
    pub fn sample_grid(&self) -> Option<(usize, usize)> {
        let dims = self.image_size()?;
        Some(match self.view.source {
            ViewSource::Merged => dims,
            ViewSource::Reference => (dims.0.div_ceil(2), dims.1.div_ceil(2)),
        })
    }

    pub fn output_estimate(&self) -> Option<ResultInfo> {
        let series = self.selected_series()?;
        let frame = series.frames.first()?;
        let scale = self
            .options
            .scale
            .resolve(series.info.frame_count(), series.info.suggested_scale);
        let mut info = pipeline::expected_output(frame.width, frame.height, scale);
        info.white = series.result.map(|r| r.white).unwrap_or(0);
        Some(info)
    }

    /// (needed bytes, free bytes) when the output would not fit on disk.
    pub fn space_shortfall(&self) -> Option<(u64, u64)> {
        let needed = self.output_estimate()?.bytes();
        let free = self.free_space?;
        if free < needed.saturating_add(512 * 1024 * 1024) {
            Some((needed, free))
        } else {
            None
        }
    }

    pub fn default_save_path(&self, scale: u32) -> Option<PathBuf> {
        let series = self.selected_series()?;
        let first = series.info.files.first()?;
        Some(pipeline::default_output_path(
            first,
            scale,
            self.options.format.ext(),
        ))
    }

    /// Let the canvas recompute the fit-to-window zoom on the next frame.
    pub fn fit_view(&mut self) {
        self.view.fitted = true;
        self.view.zoom = 0.0;
    }
}

/// Zoom at which the whole image fits into the viewport.
pub fn fit_zoom(image: (usize, usize), viewport: [f32; 2]) -> f32 {
    if image.0 == 0 || image.1 == 0 || viewport[0] <= 1.0 || viewport[1] <= 1.0 {
        return 1.0;
    }
    let zx = viewport[0] / image.0 as f32;
    let zy = viewport[1] / image.1 as f32;
    zx.min(zy).clamp(MIN_ZOOM, MAX_ZOOM)
}

/// Region of the image visible in the canvas at the current zoom, expanded by
/// margin and clamped to the image. None when the whole image is visible, so a
/// crop request is only issued when it actually buys detail.
pub fn visible_rect(
    center: [f32; 2],
    viewport: [f32; 2],
    zoom: f32,
    image: (usize, usize),
    margin: f32,
) -> Option<(usize, usize, usize, usize)> {
    if zoom <= 0.0 || image.0 == 0 || image.1 == 0 {
        return None;
    }
    let half_w = viewport[0] / zoom * 0.5 * (1.0 + margin);
    let half_h = viewport[1] / zoom * 0.5 * (1.0 + margin);
    let x0 = (center[0] - half_w).floor().clamp(0.0, image.0 as f32) as usize;
    let y0 = (center[1] - half_h).floor().clamp(0.0, image.1 as f32) as usize;
    let x1 = (center[0] + half_w).ceil().clamp(0.0, image.0 as f32) as usize;
    let y1 = (center[1] + half_h).ceil().clamp(0.0, image.1 as f32) as usize;
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    if x0 == 0 && y0 == 0 && x1 >= image.0 && y1 >= image.1 {
        return None;
    }
    Some((x0, y0, x1 - x0, y1 - y0))
}

/// True when the first rectangle contains the second one.
pub fn covers(
    outer: (usize, usize, usize, usize),
    inner: (usize, usize, usize, usize),
) -> bool {
    outer.0 <= inner.0
        && outer.1 <= inner.1
        && outer.0 + outer.2 >= inner.0 + inner.2
        && outer.1 + outer.3 >= inner.1 + inner.3
}

/// Which pyramid level the canvas has to ask for: None when the base preview
/// is enough, otherwise the downsampling step of the crop (a power of two,
/// 1 = full resolution). The step is the smallest power of two that brings
/// the visible window down to `base_dim` samples on its long side, so every
/// zoom level gets a texture sized to the viewport instead of jumping between
/// the base preview and a full resolution crop.
pub fn lod_step(
    zoom: f32,
    viewport: [f32; 2],
    image: (usize, usize),
    base_dim: u32,
) -> Option<u32> {
    if zoom <= 0.0 || image.0 == 0 || image.1 == 0 || base_dim == 0 {
        return None;
    }
    let longest = image.0.max(image.1) as f32;
    let base_ratio = base_dim as f32 / longest;
    if zoom <= base_ratio {
        return None;
    }
    let window = viewport[0].max(viewport[1]).max(1.0) / zoom;
    let mut step = 1u32;
    while window / step as f32 > base_dim as f32 {
        step *= 2;
    }
    Some(step)
}

/// Keep the view centre inside the image for the current zoom.
pub fn clamp_center(
    center: [f32; 2],
    zoom: f32,
    viewport: [f32; 2],
    image: (usize, usize),
) -> [f32; 2] {
    if zoom <= 0.0 {
        return center;
    }
    let half_w = viewport[0] / zoom * 0.5;
    let half_h = viewport[1] / zoom * 0.5;
    let (w, h) = (image.0 as f32, image.1 as f32);
    let cx = if half_w * 2.0 >= w {
        w * 0.5
    } else {
        center[0].clamp(half_w, w - half_w)
    };
    let cy = if half_h * 2.0 >= h {
        h * 0.5
    } else {
        center[1].clamp(half_h, h - half_h)
    };
    [cx, cy]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_auto_follows_the_capture_mode() {
        assert_eq!(ScaleMode::Auto.resolve(4, 1), 1);
        assert_eq!(ScaleMode::Auto.resolve(8, 1), 1);
        assert_eq!(ScaleMode::Auto.resolve(16, 2), 2);
        assert_eq!(ScaleMode::Auto.resolve(32, 2), 2);
        assert_eq!(ScaleMode::One.resolve(16, 2), 1);
        assert_eq!(ScaleMode::Two.resolve(4, 1), 2);
        assert_eq!(ScaleMode::Auto.resolve(7, 0), 1);
        assert_eq!(ScaleMode::Auto.resolve(20, 0), 2);
    }

    #[test]
    fn crop_cache_is_keyed_by_source_and_step() {
        let frames = crate::app::demo::frames();
        let result = crate::app::demo::result(1, &frames);
        let mut series = SeriesView::new(crate::app::demo::series());
        series.result = Some(ResultInfo {
            width: result.width,
            height: result.height,
            scale: 1,
            white: result.white,
        });
        let mut state = GuiState::new(Mode::Demo);
        state.series = vec![series];

        let want = CropWant {
            rect: (16, 16, 320, 208),
            step: 1,
        };
        assert_eq!(
            state.aligned_want(want).rect,
            want.rect,
            "step 1 alignment is the identity"
        );

        // A merged crop of that window arrives.
        let tiny = PreviewImage {
            width: 2,
            height: 2,
            rgb: vec![0; 12],
        };
        state.apply(Event::Preview {
            source: ViewSource::Merged,
            rect: Some(want.rect),
            step: 1,
            image: tiny.clone(),
        });
        assert!(state.crop_for(Some(want)).is_some());

        // The same window in the reference view is NOT covered by it.
        state.view.source = ViewSource::Reference;
        assert!(state.crop_for(Some(want)).is_none());

        // A coarser reference crop covers the step 2 want but not step 1.
        state.apply(Event::Preview {
            source: ViewSource::Reference,
            rect: Some(want.rect),
            step: 2,
            image: tiny,
        });
        assert!(state.crop_for(Some(CropWant { step: 2, ..want })).is_some());
        assert!(state.crop_for(Some(want)).is_none());

        // And a finer new arrival evicts the coarser copy of the same window.
        state.apply(Event::Preview {
            source: ViewSource::Reference,
            rect: Some(want.rect),
            step: 1,
            image: crate::app::demo::encode(&result, Some(want.rect), 1, [1.0, 1.0, 1.0], 1),
        });
        let cached = state.crop_for(Some(want)).expect("step 1 reference crop cached");
        assert_eq!(cached.step, 1);
        assert_eq!(
            state
                .crops
                .iter()
                .filter(|c| c.source == ViewSource::Reference && c.rect == Some(want.rect))
                .count(),
            1,
            "the superseded coarser copy was dropped"
        );
    }

    #[test]
    fn visible_rect_is_only_returned_when_it_buys_detail() {
        let image = (6048, 4032);
        assert_eq!(
            visible_rect([3024.0, 2016.0], [1200.0, 800.0], 0.1, image, 0.0),
            None
        );
        let rect = visible_rect([3024.0, 2016.0], [1200.0, 800.0], 2.0, image, 0.0).unwrap();
        assert_eq!(rect, (2724, 1816, 600, 400));
        let corner = visible_rect([0.0, 0.0], [1200.0, 800.0], 2.0, image, 0.0).unwrap();
        assert_eq!((corner.0, corner.1), (0, 0));
    }

    #[test]
    fn lod_switches_to_a_step_ladder_when_the_base_preview_is_magnified() {
        let image = (6048, 4032);
        let viewport = [1200.0, 800.0];
        // Whole frame below the preview ratio: the base preview is enough.
        assert_eq!(lod_step(0.1, viewport, image, MAX_PREVIEW_DIM), None);
        // Magnified past 1:1: the step is the smallest power of two that fits
        // the visible window into the base dimension, full resolution once
        // the window itself is small enough.
        assert_eq!(lod_step(0.35, viewport, image, MAX_PREVIEW_DIM), Some(2));
        assert_eq!(lod_step(0.6, viewport, image, MAX_PREVIEW_DIM), Some(1));
        assert_eq!(lod_step(1.0, viewport, image, MAX_PREVIEW_DIM), Some(1));
        assert_eq!(lod_step(8.0, viewport, image, MAX_PREVIEW_DIM), Some(1));
        // Barely past the base ratio the window is large: a coarse step first.
        assert_eq!(lod_step(0.34, viewport, image, MAX_PREVIEW_DIM), Some(2));
        let two_x = (12096, 8064);
        assert_eq!(lod_step(0.35, viewport, two_x, MAX_PREVIEW_DIM), Some(2));
        assert_eq!(lod_step(0.7, viewport, two_x, MAX_PREVIEW_DIM), Some(1));
        // Degenerate inputs never ask for a crop.
        assert_eq!(lod_step(0.0, viewport, image, MAX_PREVIEW_DIM), None);
        assert_eq!(lod_step(2.0, viewport, (0, 0), MAX_PREVIEW_DIM), None);
    }

    #[test]
    fn crop_wants_are_aligned_to_the_block_grid_of_their_source() {
        let dims = (12096, 8064);
        let rect = (1001, 2003, 1500, 1000);
        assert_eq!(align_rect(rect, 4, dims), (1000, 2000, 1504, 1004));
        // Alignment is idempotent.
        let aligned = align_rect(rect, 4, dims);
        assert_eq!(align_rect(aligned, 4, dims), aligned);
        // Factor 1 and degenerate dims pass the rect through.
        assert_eq!(align_rect(rect, 1, dims), rect);
        assert_eq!(align_rect(rect, 4, (0, 0)), rect);
        // The right/bottom border clamps to the image instead of overshooting.
        let edge = align_rect((12090, 8060, 6, 4), 4, dims);
        assert!(edge.0 + edge.2 <= dims.0 && edge.1 + edge.3 <= dims.1);
        assert!(covers(edge, (12090, 8060, 6, 4)) || edge.0 + edge.2 == dims.0);
    }

    #[test]
    fn lod_rect_is_the_visible_window_with_a_margin() {
        let image = (6048, 4032);
        let viewport = [1200.0, 800.0];
        let center = [3024.0, 2016.0];
        let rect = visible_rect(center, viewport, 1.0, image, 0.25).unwrap();
        // 1:1 window is viewport-sized, plus 25 % of margin on every side.
        assert_eq!(rect, (2274, 1516, 1500, 1000));
        assert!(covers(rect, visible_rect(center, viewport, 1.0, image, 0.0).unwrap()));
        let cached = crate::types::Rect::new(0, 0, 6048, 4032);
        let _ = cached;
    }

    #[test]
    fn clamp_center_keeps_the_window_inside_the_image() {
        let image = (1000, 500);
        assert_eq!(
            clamp_center([-50.0, -50.0], 1.0, [200.0, 100.0], image),
            [100.0, 50.0]
        );
        assert_eq!(
            clamp_center([5000.0, 5000.0], 1.0, [200.0, 100.0], image),
            [900.0, 450.0]
        );
        assert_eq!(
            clamp_center([10.0, 10.0], 0.1, [200.0, 100.0], image),
            [500.0, 250.0]
        );
    }

    #[test]
    fn output_estimate_matches_the_documented_sizes() {
        let one = pipeline::expected_output(6048, 4032, 1);
        assert_eq!((one.width, one.height), (6048, 4032));
        assert!((one.mib() - 139.5).abs() < 0.5, "{}", one.mib());
        let two = pipeline::expected_output(6048, 4032, 2);
        assert_eq!((two.width, two.height), (12096, 8064));
        assert!((two.mib() - 558.0).abs() < 1.0, "{}", two.mib());
    }
}
