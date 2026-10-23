//! egui drawing code: panels, canvas and the preview texture cache.

use crate::app::model::{
    clamp_center, fit_zoom, lod_level, visible_rect, Compression, GuiState, LodLevel, OutFormat,
    PreviewSlot, Request, ScaleMode, ViewSource, MAX_PREVIEW_DIM, MAX_ZOOM, MIN_ZOOM,
};
use crate::app::pipeline::{self, ResultInfo};
use crate::app::worker::Mode;
use crate::write::DngFlavor;
use eframe::egui;
use egui::{Color32, Rect, Sense, Vec2};

/// Texture cache for the preview images the worker sends back.
#[derive(Default)]
pub struct PreviewCache {
    handle: Option<egui::TextureHandle>,
    key: u64,
    size: (usize, usize),
}

impl PreviewCache {
    fn texture(&mut self, ctx: &egui::Context, slot: &PreviewSlot) -> &egui::TextureHandle {
        if self.key != slot.generation || self.handle.is_none() {
            let image = egui::ColorImage::from_rgb(
                [slot.image.width, slot.image.height],
                &slot.image.rgb,
            );
            self.handle = Some(ctx.load_texture(
                "pixelshift-preview",
                image,
                egui::TextureOptions::LINEAR,
            ));
            self.key = slot.generation;
            self.size = (slot.image.width, slot.image.height);
        }
        self.handle.as_ref().expect("preview texture")
    }
}

pub fn draw(state: &mut GuiState, cache: &mut PreviewCache, ui: &mut egui::Ui) {
    let has_result = state
        .selected_series()
        .map(|series| series.result.is_some())
        .unwrap_or(false);
    if has_result && state.poll_distort(std::time::Instant::now()) {
        state.requests.push(Request::Distort {
            distort: state.options.distort,
        });
    }
    let _ = egui::Panel::top("pixelshift-top").show(ui, |ui| top_bar(state, ui));
    let _ = egui::Panel::left("pixelshift-left")
        .resizable(true)
        .default_size(370.0)
        .show(ui, |ui| left_panel(state, ui));
    let _ = egui::Panel::bottom("pixelshift-bottom")
        .resizable(true)
        .default_size(150.0)
        .show(ui, |ui| bottom_panel(state, ui));
    let _ = egui::CentralPanel::default_margins().show(ui, |ui| canvas(state, cache, ui));
}

fn top_bar(state: &mut GuiState, ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        if ui.button("Open folder...").clicked() {
            state.requests.push(Request::PickFolder);
        }
        if ui.button("Open files...").clicked() {
            state.requests.push(Request::PickFiles);
        }
        ui.separator();
        let has_series = !state.series.is_empty();
        let busy = state.stage.is_busy();
        if ui
            .add_enabled(has_series && !busy, egui::Button::new("Merge"))
            .clicked()
        {
            state.requests.push(Request::Load);
        }
        if ui
            .add_enabled(!state.queue.is_empty() && !state.batch_running, egui::Button::new("Start batch"))
            .clicked()
        {
            state.requests.push(Request::StartBatch);
        }
        if ui
            .add_enabled(state.batch_running, egui::Button::new("Stop batch"))
            .clicked()
        {
            state.requests.push(Request::StopBatch);
        }
        ui.separator();
        let has_result = state
            .selected_series()
            .map(|s| s.result.is_some())
            .unwrap_or(false);
        if ui
            .add_enabled(has_result, egui::Button::new("Save..."))
            .clicked()
        {
            state.requests.push(Request::SaveDialog { downsample: false });
        }
        let can_1x = has_result
            && state
                .selected_series()
                .and_then(|s| s.result)
                .map(|r| r.scale > 1)
                .unwrap_or(false);
        if ui
            .add_enabled(can_1x, egui::Button::new("Save 1x..."))
            .clicked()
        {
            state.requests.push(Request::SaveDialog { downsample: true });
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button("Quit").clicked() {
                state.requests.push(Request::Quit);
            }
            ui.checkbox(&mut state.show_log, "log");
            ui.label(egui::RichText::new(crate::version()).weak());
        });
    });
}

fn left_panel(state: &mut GuiState, ui: &mut egui::Ui) {
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.heading("Series");
            if state.mode == Mode::Demo {
                ui.label(egui::RichText::new("demo data: fabricated frames").color(Color32::YELLOW));
            }
            let mut select = None;
            for (index, series) in state.series.iter().enumerate() {
                let text = format!(
                    "{}  [{}]",
                    series.info.label(),
                    if series.loaded { "merged" } else { "idle" }
                );
                if ui.selectable_label(index == state.selected, text).clicked() {
                    select = Some(index);
                }
            }
            if state.series.is_empty() {
                ui.label("no series loaded");
            }
            if let Some(index) = select {
                state.selected = index;
                state.selected_frame = 0;
                state.fit_view();
                state.requests.push(Request::View {
                    source: state.view.source,
                    rect: None,
                });
            }
            let queued = state.queue.len();
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(!state.series.is_empty(), egui::Button::new("queue all"))
                    .clicked()
                {
                    state.queue = (0..state.series.len()).collect();
                }
                if ui
                    .add_enabled(queued > 0, egui::Button::new("clear queue"))
                    .clicked()
                {
                    state.queue.clear();
                }
                ui.label(format!("{queued} queued"));
            });

            ui.separator();
            frames_panel(state, ui);
            ui.separator();
            options_panel(state, ui);
        });
}

fn frames_panel(state: &mut GuiState, ui: &mut egui::Ui) {
    ui.heading("Frames");
    let selected_frame = state.selected_frame;
    let (frames, shifts) = match state.selected_series() {
        Some(series) => (series.frames.clone(), series.shifts.clone()),
        None => (Vec::new(), Vec::new()),
    };
    if frames.is_empty() {
        ui.label("not decoded yet");
        return;
    }
    let mut select = None;
    egui::Grid::new("pixelshift-frames")
        .striped(true)
        .num_columns(7)
        .show(ui, |ui| {
            ui.label(egui::RichText::new("#").strong());
            ui.label(egui::RichText::new("file").strong());
            ui.label(egui::RichText::new("dx").strong());
            ui.label(egui::RichText::new("dy").strong());
            ui.label(egui::RichText::new("rot \"").strong());
            ui.label(egui::RichText::new("rms").strong());
            ui.label(egui::RichText::new("conf").strong());
            ui.end_row();
            for (index, frame) in frames.iter().enumerate() {
                let shift = shifts.get(index).copied().unwrap_or_default();
                if ui
                    .selectable_label(index == selected_frame, format!("{index}"))
                    .clicked()
                {
                    select = Some(index);
                }
                ui.label(frame.file_name());
                ui.label(format!("{:+.3}", shift.dx));
                ui.label(format!("{:+.3}", shift.dy));
                ui.label(format!("{:+.1}", shift.rot_arcsec()));
                ui.label(if shift.residual > 0.0 {
                    format!("{:.3}", shift.residual)
                } else {
                    "-".to_string()
                });
                ui.label(format!("{:.2}", shift.confidence));
                ui.end_row();
            }
        });
    if let Some(index) = select {
        state.selected_frame = index;
    }
    let selected_frame = state.selected_frame;
    if let Some(frame) = frames.get(selected_frame) {
        let info = format!(
            "{} x {}  CFA {}  black {}  white {}",
            frame.width, frame.height, frame.cfa, frame.black, frame.white
        );
        ui.label(egui::RichText::new(info).weak());
        ui.label(egui::RichText::new(frame.exposure_line()).weak());
        ui.label(
            egui::RichText::new(format!(
                "WB {:.3} {:.3} {:.3}",
                frame.wb[0], frame.wb[1], frame.wb[2]
            ))
            .weak(),
        );
        ui.label(egui::RichText::new(&frame.camera).weak());
    }
}

fn options_panel(state: &mut GuiState, ui: &mut egui::Ui) {
    ui.heading("Options");
    ui.horizontal(|ui| {
        ui.label("scale");
        egui::ComboBox::from_id_salt("pixelshift-scale")
            .selected_text(state.options.scale.label())
            .show_ui(ui, |ui| {
                for mode in ScaleMode::ALL {
                    ui.selectable_value(&mut state.options.scale, mode, mode.label());
                }
            });
        ui.label("format");
        egui::ComboBox::from_id_salt("pixelshift-format")
            .selected_text(state.options.format.label())
            .show_ui(ui, |ui| {
                for format in OutFormat::ALL {
                    ui.selectable_value(&mut state.options.format, format, format.label());
                }
            });
    });
    ui.horizontal(|ui| {
        ui.label("DNG main image");
        let mut flavor = state.options.flavor;
        egui::ComboBox::from_id_salt("pixelshift-flavor")
            .selected_text(flavor.label())
            .show_ui(ui, |ui| {
                for candidate in DngFlavor::ALL {
                    ui.selectable_value(&mut flavor, candidate, candidate.label());
                }
            });
        if flavor != state.options.flavor {
            state.options.flavor = flavor;
            // The cached crops belong to the previous flavor's pipeline.
            state.invalidate_crops();
        }
        if state.options.format == OutFormat::Tiff {
            ui.checkbox(&mut state.options.tiff_render, "render sRGB");
        }
    });
    ui.label(
        egui::RichText::new(
            "linearraw: full colour resolution; rgb: plain 16-bit RGB;              cfa: Bayer mosaic for readers that need a raw sensor pattern",
        )
        .small()
        .weak(),
    );
    ui.horizontal(|ui| {
        ui.label("deconvolve");
        ui.add(egui::Slider::new(&mut state.options.deconvolve, 0.0..=2.0));
    });
    ui.horizontal(|ui| {
        ui.label("Distortion k1");
        let range = -crate::distort::MAX_K1..=crate::distort::MAX_K1;
        ui.add(
            egui::Slider::new(&mut state.options.distort.k1, range)
                .step_by(0.0001)
                .fixed_decimals(4),
        );
        if ui.button("reset 0").clicked() {
            state.options.distort = crate::distort::Distortion::IDENTITY;
        }
    });
    ui.horizontal(|ui| {
        let range = -crate::distort::MAX_K1..=crate::distort::MAX_K1;
        ui.label("k2");
        ui.add(
            egui::Slider::new(&mut state.options.distort.k2, range.clone())
                .step_by(0.00001)
                .fixed_decimals(5),
        );
        ui.label("k3");
        ui.add(
            egui::Slider::new(&mut state.options.distort.k3, range.clone())
                .step_by(0.00001)
                .fixed_decimals(5),
        );
    });
    if ui
        .button("Fit distortion from reference...")
        .on_hover_text(
            "Export one frame of the series from Capture One (or Lightroom) with \
             its lens profile enabled, then measure k1, k2, k3 from the difference",
        )
        .clicked()
    {
        state.requests.push(Request::PickDistortReference);
    }
    if let Some(fit) = state.last_fit {
        ui.label(
            egui::RichText::new(format!(
                "measured k1 {:+.5} k2 {:+.5} k3 {:+.5}, residual {:.3} px, {} blocks, max radius {:.0} px",
                fit.k1, fit.k2, fit.k3, fit.residual_rms, fit.blocks, fit.max_radius_px
            ))
            .small()
            .weak(),
        );
    }
    if state.options.distort.is_identity() {
        ui.label(
            egui::RichText::new(
                "positive k1 removes pincushion (VILTROX needs +0.002..+0.01); a 24 mm zoom needs about -0.03",
            )
                .small()
                .weak(),
        );
    } else {
        let corner = state
            .image_size()
            .map(|(w, h)| crate::app::pipeline::corner_shift(w, h, state.options.distort))
            .unwrap_or(0.0);
        ui.label(
            egui::RichText::new(format!(
                "positive k1 removes pincushion: corner moves {corner:.1} px, k2 {:+.4}, k3 {:+.5}",
                state.options.distort.k2,
                state.options.distort.k3
            ))
            .small()
            .weak(),
        );
    }
    ui.horizontal(|ui| {
        ui.label("compression");
        egui::ComboBox::from_id_salt("pixelshift-compression")
            .selected_text(state.options.compression.label())
            .show_ui(ui, |ui| {
                for compression in Compression::ALL {
                    ui.selectable_value(
                        &mut state.options.compression,
                        compression,
                        compression.label(),
                    );
                }
            });
    });
    if let Some(note) = state.options.compression_note() {
        ui.label(
            egui::RichText::new(note)
                .small()
                .color(ui.visuals().warn_fg_color),
        );
    }
    ui.checkbox(&mut state.options.also_1x, "also write 1x");
    ui.checkbox(&mut state.options.align, "align: translation + rotation");
    ui.label(
        egui::RichText::new("sub-pixel shift and rotation estimation, per frame")
            .small()
            .weak(),
    );
    let mut limited = state.options.threads.is_some();
    if ui.checkbox(&mut limited, "limit threads").changed() {
        state.options.threads = if limited {
            Some(state.options.threads.unwrap_or(8).max(1))
        } else {
            None
        };
    }
    if let Some(threads) = state.options.threads.as_mut() {
        ui.add(egui::DragValue::new(threads).range(1..=128).speed(1.0));
    }

    ui.separator();
    ui.heading("Output");
    if let Some(series) = state.selected_series() {
        let mode = if series.info.unexpected_mode() {
            format!("{} frames (unexpected)", series.info.frame_count())
        } else {
            format!("mode {}", series.info.mode)
        };
        ui.label(format!("series: {mode} -> scale {}", series.info.suggested_scale));
        if series.info.unexpected_mode() {
            ui.label(
                egui::RichText::new(
                    "frame count is not 4/8/16/32: pick the scale by hand",
                )
                .small()
                .color(ui.visuals().warn_fg_color),
            );
        }
    }
    if let Some(estimate) = state.output_estimate() {
        ui.label(format!(
            "{} x {}  scale {}  {:.2} MP",
            estimate.width,
            estimate.height,
            estimate.scale,
            estimate.megapixels()
        ));
        ui.label(output_size_line(state, &estimate));
        if let Some((needed, free)) = state.space_shortfall() {
            let _ = needed;
            ui.label(
                egui::RichText::new(format!(
                    "only {} free on the output volume",
                    pipeline::fmt_bytes(free)
                ))
                .small()
                .color(Color32::from_rgb(230, 90, 90)),
            );
        }
    }
    ui.label(
        egui::RichText::new(
            "linear camera-native RGB after black subtraction (0..white-black), \
             white balance NOT applied; AsShotNeutral and the colour matrix stay in the DNG metadata",
        )
        .small()
        .weak(),
    );
}

fn output_size_line(state: &GuiState, estimate: &ResultInfo) -> String {
    let uncompressed = pipeline::fmt_bytes(estimate.bytes());
    match (state.options.format, state.options.compression) {
        (OutFormat::Tiff, Compression::Deflate) => format!(
            "{} uncompressed 16-bit; deflate expected ~1.5-2x smaller",
            uncompressed
        ),
        _ => format!("{} uncompressed 16-bit RGB", uncompressed),
    }
}

fn bottom_panel(state: &mut GuiState, ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(state.stage.label()).strong());
        ui.add(
            egui::ProgressBar::new(state.progress)
                .desired_width(240.0)
                .show_percentage(),
        );
        ui.label(format!("{:.0} ms", state.total_time().as_secs_f64() * 1000.0));
        if state.crop_pending {
            ui.spinner();
        }
        if let Some(series) = state.selected_series() {
            if let Some(stats) = &series.stats {
                ui.separator();
                ui.label(format!(
                    "frames {}  samples/channel {:.2}  min coverage {:.2}  filled {}",
                    stats.frames,
                    stats.mean_samples_per_channel,
                    stats.min_coverage,
                    stats.filled_pixels
                ));
            }
        }
    });
    if let Some(error) = state.error.clone() {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("error").color(Color32::RED).strong());
            ui.label(error);
            if ui.button("dismiss").clicked() {
                state.error = None;
            }
        });
    }
    if let Some(series) = state.selected_series() {
        if let Some(result) = series.result {
            ui.label(format!(
                "result {} x {}  scale {}  {:.2} MP  white {}",
                result.width,
                result.height,
                result.scale,
                result.megapixels(),
                result.white
            ));
        }
        for (path, bytes) in &series.saved {
            ui.label(format!(
                "saved {} ({})",
                path.display(),
                pipeline::fmt_bytes(*bytes)
            ));
        }
    }
    if state.show_log {
        egui::ScrollArea::vertical()
            .max_height(70.0)
            .auto_shrink([false, true])
            .show(ui, |ui| {
                for line in state.log.iter().rev().take(40) {
                    ui.label(egui::RichText::new(line).small().monospace());
                }
            });
    }
}

fn canvas(state: &mut GuiState, cache: &mut PreviewCache, ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        let previous = state.view.source;
        ui.selectable_value(&mut state.view.source, ViewSource::Merged, "Merged");
        ui.selectable_value(&mut state.view.source, ViewSource::Reference, "Reference");
        if state.view.source != previous {
            state.view.fitted = true;
            state.view.zoom = 0.0;
            state.requests.push(Request::View {
                source: state.view.source,
                rect: None,
            });
        }
        ui.checkbox(&mut state.view.crop, "1:1 crop");
        if ui.button("fit").clicked() {
            state.view.fitted = true;
        }
        if ui.button("100%").clicked() {
            state.view.zoom = 1.0;
            state.view.fitted = false;
        }
        ui.label(format!("{:.0}%", state.view.zoom * 100.0));
    });

    let viewport = ui.available_size();
    let (response, painter) = ui.allocate_painter(viewport, Sense::click_and_drag());
    let rect = response.rect;
    painter.rect_filled(rect, 0.0, Color32::from_rgb(18, 18, 20));

    let Some(dims) = state.image_size() else {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "no merged image yet: open a folder and press Merge",
            egui::FontId::proportional(16.0),
            Color32::GRAY,
        );
        return;
    };
    let viewport = [rect.width().max(1.0), rect.height().max(1.0)];

    if state.view.fitted || state.view.zoom <= 0.0 {
        state.view.zoom = fit_zoom(dims, viewport);
        state.view.center = [dims.0 as f32 * 0.5, dims.1 as f32 * 0.5];
    }

    if response.dragged() {
        let delta = response.drag_delta();
        state.view.center[0] -= delta.x / state.view.zoom;
        state.view.center[1] -= delta.y / state.view.zoom;
        state.view.fitted = false;
    }
    if response.hovered() {
        let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
        let factor = pinch * (scroll * 0.002).exp();
        if (factor - 1.0).abs() > 1e-4 {
            state.view.zoom = (state.view.zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
            state.view.fitted = false;
        }
    }
    state.view.center = clamp_center(state.view.center, state.view.zoom, viewport, dims);

    // Level of detail: the fitted preview until it is magnified past 1:1, a 1:1
    // crop of the merged image afterwards (the "1:1 crop" box forces it).
    let level = if state.view.crop && state.view.zoom >= 1.0 {
        LodLevel::Detail
    } else {
        lod_level(state.view.zoom, viewport, dims, MAX_PREVIEW_DIM)
    };
    let wanted = match level {
        LodLevel::Detail => visible_rect(state.view.center, viewport, state.view.zoom, dims, 0.25),
        LodLevel::Base => None,
    };
    let loading = state.poll_lod(wanted, std::time::Instant::now());

    let zoom = state.view.zoom;
    let center = state.view.center;
    let source = state.view.source.index();
    if let Some(slot) = state.previews[source].as_ref() {
        let texture = cache.texture(ui.ctx(), slot);
        paint_slot(
            &painter,
            texture.id(),
            (0, 0, dims.0, dims.1),
            center,
            zoom,
            rect,
            false,
        );
    }
    if let Some(slot) = state.crop_for(wanted) {
        let texture = cache.texture(ui.ctx(), slot);
        let crop_rect = slot.rect.unwrap_or((0, 0, dims.0, dims.1));
        paint_slot(&painter, texture.id(), crop_rect, center, zoom, rect, true);
    }
    if loading {
        painter.text(
            rect.right_top() + Vec2::new(-8.0, 8.0),
            egui::Align2::RIGHT_TOP,
            "loading detail",
            egui::FontId::monospace(12.0),
            Color32::from_rgb(230, 200, 120),
        );
    }

    let label = format!(
        "{}  {} x {}  {:.0}%{}  {}",
        state.view.source.label(),
        dims.0,
        dims.1,
        zoom * 100.0,
        if state.view.crop { "  crop" } else { "" },
        match level {
            LodLevel::Base => "base",
            LodLevel::Detail => "detail 1:1",
        }
    );
    painter.text(
        rect.left_bottom() + Vec2::new(8.0, -8.0),
        egui::Align2::LEFT_BOTTOM,
        label,
        egui::FontId::monospace(12.0),
        Color32::from_rgb(180, 180, 190),
    );
}

/// Draw one cached preview at the canvas position of its region.
fn paint_slot(
    painter: &egui::Painter,
    texture: egui::TextureId,
    source_rect: (usize, usize, usize, usize),
    center: [f32; 2],
    zoom: f32,
    canvas: Rect,
    outline: bool,
) {
    let top_left = canvas.center()
        + Vec2::new(
            (source_rect.0 as f32 - center[0]) * zoom,
            (source_rect.1 as f32 - center[1]) * zoom,
        );
    let size = Vec2::new(source_rect.2 as f32 * zoom, source_rect.3 as f32 * zoom);
    let target = Rect::from_min_size(top_left, size);
    painter.image(
        texture,
        target,
        Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
        Color32::WHITE,
    );
    if outline {
        painter.rect_stroke(
            target,
            0.0,
            egui::Stroke::new(1.0, Color32::from_rgb(90, 90, 120)),
            egui::StrokeKind::Inside,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::demo;
    use crate::app::model::{EditOptions, FrameInfo, PreviewSlot, SeriesView, Stage, MAX_PREVIEW_DIM};
    use crate::app::pipeline::ResultInfo;

    fn demo_state() -> GuiState {
        let frames = demo::frames();
        let result = demo::result(1, &frames);
        let mut series = SeriesView::new(demo::series());
        series.frames = frames
            .iter()
            .map(pipeline::frame_info)
            .collect::<Vec<FrameInfo>>();
        series.shifts = demo::shift_table();
        series.result = Some(ResultInfo {
            width: result.width,
            height: result.height,
            scale: 1,
            white: result.white,
        });
        series.stats = Some(result.stats.clone());
        series.loaded = true;
        let image = demo::encode(&result, None, MAX_PREVIEW_DIM, [1.244, 1.0, 2.010]);
        let mut state = GuiState::new(Mode::Demo);
        state.series = vec![series];
        state.previews[0] = Some(PreviewSlot {
            image,
            rect: None,
            generation: 1,
        });
        state.stage = Stage::Done;
        state
    }

    fn run_frame(state: &mut GuiState, cache: &mut PreviewCache) -> usize {
        let ctx = egui::Context::default();
        let mut input = egui::RawInput::default();
        input.screen_rect = Some(Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        ));
        let mut output = ctx.run_ui(input, |ui| draw(state, cache, ui));
        output.textures_delta.clear();
        output.shapes.len()
    }

    #[test]
    fn fabricated_result_drives_the_preview_widgets() {
        let mut state = demo_state();
        let mut cache = PreviewCache::default();
        for _ in 0..3 {
            let shapes = run_frame(&mut state, &mut cache);
            assert!(shapes > 0, "egui produced no shapes");
        }
        assert!(cache.handle.is_some(), "preview texture was not uploaded");
        assert_eq!(cache.size, (640, 424));
        assert!(state.view.zoom > 0.0 && state.view.zoom < 2.0);
    }

    #[test]
    fn options_widgets_edit_the_state() {
        let mut state = demo_state();
        state.options = EditOptions::default();
        state.options.scale = ScaleMode::Two;
        state.options.format = OutFormat::Tiff;
        state.options.compression = Compression::Deflate;
        let mut cache = PreviewCache::default();
        run_frame(&mut state, &mut cache);
        assert_eq!(
            state.options.compression_note(),
            Some("16-bit TIFF: deflate would shrink the file ~1.5-2x, but this build writes TIFF uncompressed")
        );
        let estimate = state.output_estimate().unwrap();
        assert_eq!((estimate.width, estimate.height), (1280, 848));
    }
}
