//! Background worker thread: owns the decoded frames and the merged result so
//! the GUI thread only ever moves small preview images. Progress and results
//! travel back over an mpsc channel, the GUI stays responsive.

use crate::app::demo;
use crate::app::model::{EditOptions, FrameInfo, OutFormat, Stage, ViewSource, MAX_PREVIEW_DIM};
use crate::app::pipeline::{self, ResultInfo, SeriesInfo};
use crate::types::{MergeResult, MergeStats, PreviewImage, RawFrame, Shift};
use crate::write::DngFlavor;
use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::JoinHandle;
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Real NEF files through rawio/shift/merge/write.
    Real,
    /// Fabricated data, usable while WP1/WP2 are still stubs.
    Demo,
}

#[derive(Clone, Debug)]
pub struct SaveTarget {
    pub path: PathBuf,
    pub format: OutFormat,
    pub flavor: DngFlavor,
    pub tiff_render: bool,
    pub also_1x: bool,
}

#[derive(Clone, Debug)]
pub enum Command {
    /// Directory scan or explicit file list; empty input means "demo series".
    Scan(Vec<PathBuf>),
    Load {
        index: usize,
        options: EditOptions,
        save: Option<SaveTarget>,
    },
    View {
        source: ViewSource,
        rect: Option<(usize, usize, usize, usize)>,
    },
    /// Re-rectify the merged result with new lens distortion coefficients and
    /// send a fresh preview; the export uses the same values.
    Distort {
        distort: crate::distort::Distortion,
    },
    /// Measure the lens distortion of the merged result against a reference
    /// image and report it; the GUI then applies it like a slider move.
    FitDistort {
        reference: PathBuf,
    },
    SaveAs {
        index: usize,
        path: PathBuf,
        format: OutFormat,
        flavor: DngFlavor,
        tiff_render: bool,
        downsample: bool,
        also_1x: bool,
    },
    Shutdown,
}

pub enum Event {
    Stage(Stage),
    Progress(f32),
    Log(String),
    Scanned(Vec<SeriesInfo>),
    Decoded {
        index: usize,
        frames: Vec<FrameInfo>,
    },
    Aligned {
        index: usize,
        shifts: Vec<Shift>,
    },
    Merged {
        index: usize,
        info: ResultInfo,
        stats: MergeStats,
    },
    Preview {
        source: ViewSource,
        rect: Option<(usize, usize, usize, usize)>,
        image: PreviewImage,
    },
    Saved {
        index: usize,
        path: PathBuf,
        bytes: u64,
    },
    Fitted(crate::fit::FitReport),
    Finished,
    Error(String),
}

pub struct Worker {
    tx: Sender<Command>,
    rx: Receiver<Event>,
    handle: Option<JoinHandle<()>>,
}

impl Worker {
    pub fn spawn(mode: Mode) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel();
        let (event_tx, event_rx) = mpsc::channel();
        let handle = std::thread::Builder::new()
            .name("pixelshift-pipeline".to_string())
            .spawn(move || run(mode, cmd_rx, event_tx))
            .expect("spawn the pipeline worker");
        Self {
            tx: cmd_tx,
            rx: event_rx,
            handle: Some(handle),
        }
    }

    pub fn send(&self, command: Command) {
        let _ = self.tx.send(command);
    }

    pub fn poll(&self) -> Vec<Event> {
        let mut out = Vec::new();
        while let Ok(event) = self.rx.try_recv() {
            out.push(event);
        }
        out
    }

    pub fn wait(&self, timeout: Duration) -> Option<Event> {
        self.rx.recv_timeout(timeout).ok()
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.tx.send(Command::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn run(mode: Mode, rx: Receiver<Command>, tx: Sender<Event>) {
    let mut session = Session::new(mode, tx);
    loop {
        let command = match rx.recv() {
            Ok(command) => command,
            Err(_) => break,
        };
        if matches!(command, Command::Shutdown) {
            break;
        }
        if let Err(error) = session.handle(command) {
            let _ = session.tx.send(Event::Error(format!("{error:#}")));
        }
    }
}

struct Session {
    mode: Mode,
    tx: Sender<Event>,
    scanned: Vec<SeriesInfo>,
    frames: Vec<RawFrame>,
    /// Merge output before the lens correction.
    result: Option<MergeResult>,
    /// Rectified copy of [Session::result] for the current coefficients; None
    /// when the correction is the identity, so the cache only exists when it
    /// changes the pixels.
    rectified: Option<MergeResult>,
    distort: crate::distort::Distortion,
    wb: [f32; 3],
    scale: u32,
}

impl Session {
    fn new(mode: Mode, tx: Sender<Event>) -> Self {
        Self {
            mode,
            tx,
            scanned: Vec::new(),
            frames: Vec::new(),
            result: None,
            rectified: None,
            distort: crate::distort::Distortion::IDENTITY,
            wb: [1.0, 1.0, 1.0],
            scale: 1,
        }
    }

    fn handle(&mut self, command: Command) -> Result<()> {
        match command {
            Command::Shutdown => Ok(()),
            Command::Scan(inputs) => self.scan(&inputs),
            Command::Load {
                index,
                options,
                save,
            } => self.load(index, &options, save),
            Command::View { source, rect } => self.view(source, rect),
            Command::Distort { distort } => self.distort(distort),
            Command::FitDistort { reference } => self.fit_distort(&reference),
            Command::SaveAs {
                index,
                path,
                format,
                flavor,
                tiff_render,
                downsample,
                also_1x,
            } => {
                self.stage(Stage::Saving);
                self.write(index, &path, format, flavor, tiff_render, downsample)?;
                if also_1x && !downsample && self.scale > 1 {
                    let one_x = pipeline::sibling_1x_path(&path);
                    self.write(index, &one_x, format, flavor, tiff_render, true)?;
                }
                self.stage(Stage::Done);
                self.send(Event::Finished);
                Ok(())
            }
        }
    }

    fn send(&self, event: Event) {
        let _ = self.tx.send(event);
    }

    /// Image the GUI shows and the writer stores: the rectified copy when the
    /// correction is not the identity, the merge output otherwise.
    fn image(&self) -> Result<&MergeResult> {
        self.rectified
            .as_ref()
            .or(self.result.as_ref())
            .ok_or_else(|| anyhow!("nothing merged yet"))
    }

    fn rebuild_rectified(&mut self) -> Result<()> {
        self.rectified = None;
        if self.distort.is_identity() {
            return Ok(());
        }
        let started = std::time::Instant::now();
        let Some(result) = self.result.as_ref() else {
            return Ok(());
        };
        let image = pipeline::rectify(result.clone(), self.distort)?;
        self.rectified = Some(image);
        self.send(Event::Log(format!(
            "lens correction {} applied in {:.0} ms",
            self.distort,
            started.elapsed().as_secs_f64() * 1000.0
        )));
        Ok(())
    }

    fn distort(&mut self, distort: crate::distort::Distortion) -> Result<()> {
        if distort == self.distort {
            return Ok(());
        }
        self.distort = distort;
        self.rebuild_rectified()?;
        self.view(ViewSource::Merged, None)
    }

    /// Fit the coefficients of the merged result against an exported reference
    /// frame; the caller applies the report exactly as a slider move.
    fn fit_distort(&mut self, reference: &Path) -> Result<()> {
        let result = self
            .result
            .clone()
            .ok_or_else(|| anyhow!("merge a series before fitting the distortion"))?;
        self.send(Event::Log(format!(
            "fitting the lens distortion from {}",
            reference.display()
        )));
        let report = crate::fit::fit_from_reference(&result, reference)?;
        self.send(Event::Log(format!("distortion fit: {}", report.summary())));
        self.send(Event::Fitted(report));
        Ok(())
    }

    fn stage(&self, stage: Stage) {
        self.send(Event::Stage(stage));
    }

    fn scan(&mut self, inputs: &[PathBuf]) -> Result<()> {
        self.stage(Stage::Scanning);
        let list = if self.mode == Mode::Demo {
            if !inputs.is_empty() {
                self.send(Event::Log(
                    "demo mode: the given inputs are ignored, using fabricated frames".to_string(),
                ));
            }
            vec![demo::series()]
        } else {
            pipeline::resolve_inputs(inputs)?
        };
        self.send(Event::Log(format!(
            "scan: {} series found ({} mode)",
            list.len(),
            if self.mode == Mode::Demo { "demo" } else { "real" }
        )));
        self.scanned = list.clone();
        self.send(Event::Scanned(list));
        Ok(())
    }

    fn load(
        &mut self,
        index: usize,
        options: &EditOptions,
        save: Option<SaveTarget>,
    ) -> Result<()> {
        let series = self
            .scanned
            .get(index)
            .cloned()
            .ok_or_else(|| anyhow!("series index {index} out of range"))?;
        self.scale = options
            .scale
            .resolve(series.frame_count(), series.suggested_scale);

        self.stage(Stage::Decoding);
        self.send(Event::Progress(0.1));
        let frames = if self.mode == Mode::Demo {
            demo::frames()
        } else {
            pipeline::decode(&series.files)?
        };
        let infos: Vec<FrameInfo> = frames.iter().map(pipeline::frame_info).collect();
        self.send(Event::Decoded {
            index,
            frames: infos,
        });
        self.wb = frames
            .first()
            .map(pipeline::preview_wb)
            .unwrap_or([1.0, 1.0, 1.0]);

        self.stage(Stage::Aligning);
        self.send(Event::Progress(0.4));
        let shifts = if self.mode == Mode::Demo {
            demo::shifts(&frames)
        } else {
            pipeline::estimate_shifts(&frames, options.align)?
        };
        let shifts = if options.align {
            shifts
        } else {
            vec![Shift::default(); frames.len()]
        };
        self.send(Event::Aligned {
            index,
            shifts: shifts.clone(),
        });

        self.stage(Stage::Merging);
        self.send(Event::Progress(0.6));
        let result = if self.mode == Mode::Demo {
            demo::result(self.scale, &frames)
        } else {
            let params = options.merge_params(self.scale, shifts);
            pipeline::merge_frames(&frames, &params)?
        };
        let info = ResultInfo {
            width: result.width,
            height: result.height,
            scale: result.scale,
            white: result.white,
        };
        self.send(Event::Merged {
            index,
            info,
            stats: result.stats.clone(),
        });
        self.result = Some(result);
        self.distort = options.distort;
        self.rectified = None;
        self.rebuild_rectified()?;
        self.frames = frames;

        self.stage(Stage::Previewing);
        self.send(Event::Progress(0.85));
        self.view(ViewSource::Merged, None)?;

        if let Some(target) = save {
            self.stage(Stage::Saving);
            self.write(
                index,
                &target.path,
                target.format,
                target.flavor,
                target.tiff_render,
                false,
            )?;
            if target.also_1x && self.scale > 1 {
                let path = pipeline::sibling_1x_path(&target.path);
                self.write(
                    index,
                    &path,
                    target.format,
                    target.flavor,
                    target.tiff_render,
                    true,
                )?;
            }
        }
        self.stage(Stage::Done);
        self.send(Event::Finished);
        Ok(())
    }

    fn view(
        &mut self,
        source: ViewSource,
        rect: Option<(usize, usize, usize, usize)>,
    ) -> Result<()> {
        let image = match source {
            ViewSource::Merged => {
                let result = self.image()?;
                self.render_merged(result, rect)?
            }
            ViewSource::Reference => {
                let frame = self
                    .frames
                    .first()
                    .ok_or_else(|| anyhow!("no reference frame decoded yet"))?;
                pipeline::reference_preview(frame, rect, MAX_PREVIEW_DIM, self.wb)
            }
        };
        self.send(Event::Preview {
            source,
            rect,
            image,
        });
        Ok(())
    }

    fn render_merged(
        &self,
        result: &MergeResult,
        rect: Option<(usize, usize, usize, usize)>,
    ) -> Result<PreviewImage> {
        let attempt = match rect {
            None => pipeline::preview(result, MAX_PREVIEW_DIM, self.wb),
            Some(rect) => pipeline::crop_preview(result, rect, self.wb),
        };
        match attempt {
            Ok(image) => Ok(image),
            Err(error) if self.mode == Mode::Demo && pipeline::is_unavailable(&error) => {
                Ok(demo::encode(result, rect, MAX_PREVIEW_DIM, self.wb))
            }
            Err(error) => Err(error),
        }
    }

    fn write(
        &mut self,
        index: usize,
        path: &Path,
        format: OutFormat,
        flavor: DngFlavor,
        tiff_render: bool,
        downsample: bool,
    ) -> Result<()> {
        let result = self.image()?;
        let reference = self
            .frames
            .first()
            .ok_or_else(|| anyhow!("no reference frame decoded yet"))?;
        let meta = pipeline::dng_meta(reference, &crate::version());
        let reduced;
        let image: &MergeResult = if downsample {
            reduced = match pipeline::downsample_1x(result) {
                Ok(img) => img,
                Err(error) if self.mode == Mode::Demo && pipeline::is_unavailable(&error) => {
                    demo::downsample(result)
                }
                Err(error) => return Err(error),
            };
            &reduced
        } else {
            result
        };
        self.send(Event::Progress(0.95));
        let outcome = pipeline::write_output(
            path,
            image,
            &meta,
            format,
            crate::app::model::Compression::None,
            flavor,
            tiff_render,
        );
        let (final_path, bytes) = match outcome {
            Ok(bytes) => (path.to_path_buf(), bytes),
            Err(error) if self.mode == Mode::Demo && pipeline::is_unavailable(&error) => {
                let alternative = path.with_extension("ppm");
                let bytes = demo::write_ppm(&alternative, image)?;
                self.send(Event::Log(format!(
                    "output writer (WP1) is not ready: demo wrote {} instead",
                    alternative.display()
                )));
                (alternative, bytes)
            }
            Err(error) => return Err(error),
        };
        self.send(Event::Saved {
            index,
            path: final_path,
            bytes,
        });
        Ok(())
    }
}
