//! egui/eframe GUI.

pub mod demo;
pub mod model;
pub mod pipeline;
pub mod selftest;
pub mod ui;
pub mod worker;

use crate::app::model::{GuiState, Request, Stage};
use crate::app::worker::{Command, Event, Mode, SaveTarget, Worker};
use anyhow::{anyhow, Result};
use eframe::egui;
use std::path::PathBuf;

/// Options of the GUI entry point, filled in from pixelshift gui ... .
#[derive(Clone, Debug, Default)]
pub struct GuiOptions {
    pub selftest: bool,
    pub demo: bool,
    pub inputs: Vec<PathBuf>,
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
    eframe::run_native(
        "pixelshift",
        native,
        Box::new(move |cc| Ok(Box::new(PixelShiftApp::new(cc, mode, inputs)))),
    )
    .map_err(|error| anyhow!("could not start the GUI: {error}"))
}

struct PixelShiftApp {
    state: GuiState,
    cache: ui::PreviewCache,
    worker: Worker,
    last_selected: usize,
}

impl PixelShiftApp {
    fn new(_cc: &eframe::CreationContext<'_>, mode: Mode, inputs: Vec<PathBuf>) -> Self {
        let mut state = GuiState::new(mode);
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
        }
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
        let options = self.state.options.clone();
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
        let info = series.info.clone();
        let scale = self
            .state
            .options
            .scale
            .resolve(info.frame_count(), info.suggested_scale);
        let Some(first) = info.files.first() else {
            self.state.batch_running = false;
            return;
        };
        let path = pipeline::default_output_path(first, scale, self.state.options.format.ext());
        let save = Some(SaveTarget {
            path,
            format: self.state.options.format,
            flavor: self.state.options.flavor,
            tiff_render: self.state.options.tiff_render,
            also_1x: self.state.options.also_1x,
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
            self.state.requests.push(Request::SaveAs { path, downsample });
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
                Request::Load => {
                    let index = self.state.selected;
                    self.load_selected(index, None);
                }
                Request::View { source, rect } => {
                    self.worker.send(Command::View { source, rect });
                }
                Request::Distort { distort } => {
                    self.worker.send(Command::Distort { distort });
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
                    });
                }
                Request::StartBatch => {
                    if self.state.queue.is_empty() {
                        self.state.queue = (0..self.state.series.len()).collect();
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
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let finished = self.poll_worker();
        if self.state.selected != self.last_selected {
            self.last_selected = self.state.selected;
            self.refresh_free_space();
        }
        ui::draw(&mut self.state, &mut self.cache, ui);
        self.handle_requests(ui.ctx());
        if finished && self.state.batch_running {
            self.advance_batch();
        }
        if self.state.needs_tick() {
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(80));
        }
    }
}
