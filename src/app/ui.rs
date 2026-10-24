//! egui drawing code: panels, canvas and the preview texture cache.

use crate::app::i18n::Lang;
use crate::app::model::{
    clamp_center, fit_zoom, lod_step, visible_rect, Compression, CropWant, GuiState, LeftTab,
    OutFormat, PreviewSlot, Request, ScaleMode, Stage, ViewSource, MAX_PREVIEW_DIM, MAX_ZOOM,
    MIN_ZOOM,
};
use crate::app::pipeline::{self, ResultInfo};
use crate::app::prefs::UiPrefs;
use crate::app::worker::Mode;
use crate::write::DngFlavor;
use eframe::egui;
use egui::{Color32, Rect, Sense, Vec2};

/// Texture cache for the preview images the worker sends back. One handle per
/// preview generation, so the base image and the current crop can be on
/// screen at the same time; handles beyond the cap are dropped oldest first
/// (egui frees the GPU texture when the last handle goes away).
pub struct PreviewCache {
    textures: Vec<(u64, egui::TextureHandle)>,
}

impl Default for PreviewCache {
    fn default() -> Self {
        Self {
            textures: Vec::new(),
        }
    }
}

impl PreviewCache {
    const MAX_TEXTURES: usize = 8;

    fn texture(&mut self, ctx: &egui::Context, slot: &PreviewSlot) -> &egui::TextureHandle {
        if let Some(pos) = self
            .textures
            .iter()
            .position(|(key, _)| *key == slot.generation)
        {
            return &self.textures[pos].1;
        }
        let image =
            egui::ColorImage::from_rgb([slot.image.width, slot.image.height], &slot.image.rgb);
        let handle = ctx.load_texture(
            format!("pixelshift-preview-{}", slot.generation),
            image,
            egui::TextureOptions::LINEAR,
        );
        self.textures.push((slot.generation, handle));
        while self.textures.len() > Self::MAX_TEXTURES {
            // Dropping the handle frees the GPU texture.
            drop(self.textures.remove(0));
        }
        let last = self.textures.len() - 1;
        &self.textures[last].1
    }
}

/// Draw one frame; true when a preference changed and must be re-applied.
pub fn draw(
    state: &mut GuiState,
    cache: &mut PreviewCache,
    prefs: &mut UiPrefs,
    ui: &mut egui::Ui,
) -> bool {
    let has_result = state
        .selected_series()
        .map(|series| series.result.is_some())
        .unwrap_or(false);
    if has_result && state.poll_distort(std::time::Instant::now()) {
        state.requests.push(Request::Distort {
            distort: state.options.distort,
        });
    }
    let lang = prefs.lang;
    let mut changed = false;
    let _ = egui::Panel::top("pixelshift-top").show(ui, |ui| {
        top_bar(state, lang, ui);
    });
    let _ = egui::Panel::left("pixelshift-left")
        .resizable(true)
        .default_size(370.0)
        .show(ui, |ui| {
            changed |= left_panel(state, prefs, lang, ui);
        });
    let _ = egui::Panel::bottom("pixelshift-bottom")
        .resizable(true)
        .default_size(150.0)
        .show(ui, |ui| bottom_panel(state, lang, ui));
    let _ = egui::CentralPanel::default_margins().show(ui, |ui| canvas(state, cache, lang, ui));
    changed
}

fn top_bar(state: &mut GuiState, lang: Lang, ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        if ui
            .button(lang.t("Open folder...", "Открыть папку..."))
            .clicked()
        {
            state.requests.push(Request::PickFolder);
        }
        if ui
            .button(lang.t("Open files...", "Открыть файлы..."))
            .clicked()
        {
            state.requests.push(Request::PickFiles);
        }
        ui.separator();
        let has_series = !state.series.is_empty();
        let busy = state.stage.is_busy();
        if ui
            .add_enabled(
                has_series && !busy,
                egui::Button::new(lang.t("Merge", "Объединить")),
            )
            .clicked()
        {
            state.requests.push(Request::Load);
        }
        if ui
            .add_enabled(
                !state.queue.is_empty() && !state.batch_running,
                egui::Button::new(lang.t("Start batch", "Запустить очередь")),
            )
            .clicked()
        {
            state.requests.push(Request::StartBatch);
        }
        if ui
            .add_enabled(
                state.batch_running,
                egui::Button::new(lang.t("Stop batch", "Остановить очередь")),
            )
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
            .add_enabled(
                has_result,
                egui::Button::new(lang.t("Save...", "Сохранить...")),
            )
            .clicked()
        {
            state
                .requests
                .push(Request::SaveDialog { downsample: false });
        }
        let can_1x = has_result
            && state
                .selected_series()
                .and_then(|s| s.result)
                .map(|r| r.scale > 1)
                .unwrap_or(false);
        if ui
            .add_enabled(
                can_1x,
                egui::Button::new(lang.t("Save 1x...", "Сохранить 1x...")),
            )
            .clicked()
        {
            state
                .requests
                .push(Request::SaveDialog { downsample: true });
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button(lang.t("Quit", "Выход")).clicked() {
                state.requests.push(Request::Quit);
            }
            ui.checkbox(&mut state.show_log, lang.t("log", "лог"));
            ui.label(egui::RichText::new(crate::version()).weak());
        });
    });
}

/// Warning colour that follows the active theme: the bright warning colour in
/// the dark theme, the same hue darkened to amber on the light panel.
fn demo_warning_color(ui: &egui::Ui) -> Color32 {
    let color = ui.visuals().warn_fg_color;
    if ui.visuals().dark_mode {
        color
    } else {
        Color32::from_rgb(
            (color.r() as f32 * 0.72).round() as u8,
            (color.g() as f32 * 0.72).round() as u8,
            (color.b() as f32 * 0.72).round() as u8,
        )
    }
}

/// Left panel with three tabs: the series list and frame table, the merge
/// parameters, and the application settings. Returns true when a preference
/// changed and must be re-applied.
fn left_panel(state: &mut GuiState, prefs: &mut UiPrefs, lang: Lang, ui: &mut egui::Ui) -> bool {
    let mut prefs_changed = false;
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut state.left_tab, LeftTab::Series, lang.t("Series", "Серии"));
                ui.selectable_value(&mut state.left_tab, LeftTab::Params, lang.t("Params", "Параметры"));
                ui.selectable_value(
                    &mut state.left_tab,
                    LeftTab::Settings,
                    lang.t("Settings", "Настройки"),
                );
            });
            ui.separator();
            match state.left_tab {
                LeftTab::Series => series_tab(state, lang, ui),
                LeftTab::Params => options_panel(state, lang, ui),
                LeftTab::Settings => prefs_changed |= prefs.settings_body(ui),
            }
        });
    prefs_changed
}

fn series_tab(state: &mut GuiState, lang: Lang, ui: &mut egui::Ui) {
    if state.mode == Mode::Demo {
        ui.label(
            egui::RichText::new("demo data: fabricated frames")
                .color(demo_warning_color(ui)),
        );
    }
    let mut select = None;
    for (index, series) in state.series.iter().enumerate() {
        let status = if series.loaded {
            lang.t("merged", "объединено")
        } else {
            lang.t("idle", "ожидает")
        };
        let text = format!("{}  [{status}]", series.info.label());
        if ui.selectable_label(index == state.selected, text).clicked() {
            select = Some(index);
        }
    }
    if state.series.is_empty() {
        ui.label(lang.t("no series loaded", "серии не загружены"));
    }
    if let Some(index) = select {
        state.selected = index;
        state.selected_frame = 0;
        state.fit_view();
        state.requests.push(Request::View {
            source: state.view.source,
            rect: None,
            step: 1,
        });
    }
    let queued = state.queue.len();
    ui.horizontal(|ui| {
        if ui
            .add_enabled(
                !state.series.is_empty(),
                egui::Button::new(lang.t("queue all", "в очередь все")),
            )
            .clicked()
        {
            state.queue = (0..state.series.len()).collect();
        }
        if ui
            .add_enabled(
                queued > 0,
                egui::Button::new(lang.t("clear queue", "очистить очередь")),
            )
            .clicked()
        {
            state.queue.clear();
        }
        ui.label(format!("{queued} {}", lang.t("queued", "в очереди")));
    });

    ui.separator();
    frames_panel(state, lang, ui);
}

fn frames_panel(state: &mut GuiState, lang: Lang, ui: &mut egui::Ui) {
    ui.heading(lang.t("Frames", "Кадры"));
    let selected_frame = state.selected_frame;
    let (frames, shifts) = match state.selected_series() {
        Some(series) => (series.frames.clone(), series.shifts.clone()),
        None => (Vec::new(), Vec::new()),
    };
    if frames.is_empty() {
        ui.label(lang.t("not decoded yet", "ещё не декодировано"));
        return;
    }
    let mut select = None;
    egui::Grid::new("pixelshift-frames")
        .striped(true)
        .num_columns(8)
        .show(ui, |ui| {
            ui.label(egui::RichText::new("#").strong());
            ui.label(egui::RichText::new(lang.t("file", "файл")).strong());
            ui.label(egui::RichText::new("dx").strong());
            ui.label(egui::RichText::new("dy").strong());
            ui.label(egui::RichText::new("rot \"").strong());
            ui.label(egui::RichText::new("rms").strong());
            ui.label(egui::RichText::new("conf").strong());
            ui.label(egui::RichText::new(lang.t("mask", "маска")).strong());
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
                ui.label(if shift.masked_pct > 0.0 {
                    format!("{:.1}%", shift.masked_pct * 100.0)
                } else {
                    "-".to_string()
                });
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

fn options_panel(state: &mut GuiState, lang: Lang, ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        ui.label(lang.t("scale", "масштаб"));
        egui::ComboBox::from_id_salt("pixelshift-scale")
            .selected_text(scale_label(lang, state.options.scale))
            .show_ui(ui, |ui| {
                for mode in ScaleMode::ALL {
                    ui.selectable_value(&mut state.options.scale, mode, scale_label(lang, mode));
                }
            });
        ui.label(lang.t("format", "формат"));
        egui::ComboBox::from_id_salt("pixelshift-format")
            .selected_text(format_label(lang, state.options.format))
            .show_ui(ui, |ui| {
                for format in OutFormat::ALL {
                    ui.selectable_value(
                        &mut state.options.format,
                        format,
                        format_label(lang, format),
                    );
                }
            });
    });
    ui.horizontal(|ui| {
        ui.label(lang.t("DNG main image", "основное изображение DNG"));
        let mut flavor = state.options.flavor;
        egui::ComboBox::from_id_salt("pixelshift-flavor")
            .selected_text(flavor_label(lang, flavor))
            .show_ui(ui, |ui| {
                for candidate in DngFlavor::ALL {
                    ui.selectable_value(&mut flavor, candidate, flavor_label(lang, candidate));
                }
            });
        if flavor != state.options.flavor {
            state.options.flavor = flavor;
            // The cached crops belong to the previous flavor's pipeline.
            state.invalidate_crops();
        }
        if state.options.format == OutFormat::Tiff {
            ui.checkbox(
                &mut state.options.tiff_render,
                lang.t("render sRGB", "рендерить sRGB"),
            );
        }
    });
    ui.label(
        egui::RichText::new(lang.t(
            "linearraw: full colour resolution; rgb: plain 16-bit RGB; cfa: Bayer mosaic for readers that need a raw sensor pattern",
            "linearraw: полное цветовое разрешение; rgb: обычный 16-bit RGB; cfa: мозаика Байера для программ, которым нужен сырой шаблон сенсора",
        ))
        .small()
        .weak(),
    );
    ui.horizontal(|ui| {
        ui.label(lang.t("deconvolve", "деконволюция"));
        ui.add(egui::Slider::new(&mut state.options.deconvolve, 0.0..=2.0));
    });
    ui.horizontal(|ui| {
        ui.checkbox(
            &mut state.options.motion_enabled,
            lang.t("motion compensation", "компенсация движения"),
        );
        if state.options.motion_enabled {
            ui.label(lang.t("sensitivity", "чувствительность"));
            ui.add(
                egui::Slider::new(&mut state.options.motion_sensitivity, 0.25..=4.0).fixed_decimals(2),
            );
        }
    });
    ui.horizontal(|ui| {
        ui.label(lang.t("Distortion k1", "Дисторсия k1"));
        let range = -crate::distort::MAX_K1..=crate::distort::MAX_K1;
        ui.add(
            egui::Slider::new(&mut state.options.distort.k1, range)
                .step_by(0.0001)
                .fixed_decimals(4),
        );
        if ui.button(lang.t("reset 0", "сброс 0")).clicked() {
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
        .button(lang.t(
            "Fit distortion from reference...",
            "Измерить дисторсию по эталону...",
        ))
        .on_hover_text(lang.t(
            "Export one frame of the series from Capture One (or Lightroom) with its lens profile enabled, then measure k1, k2, k3 from the difference",
            "Экспортируйте один кадр серии из Capture One (или Lightroom) с включённым профилем объектива, затем измерьте k1, k2, k3 по разнице",
        ))
        .clicked()
    {
        state.requests.push(Request::PickDistortReference);
    }
    if let Some(fit) = state.last_fit {
        ui.label(
            egui::RichText::new(format!(
                "{} k1 {:+.5} k2 {:+.5} k3 {:+.5}, {} {:.3} px, {} {}, {} {:.0} px",
                lang.t("measured", "измерено"),
                fit.k1,
                fit.k2,
                fit.k3,
                lang.t("residual", "остаток"),
                fit.residual_rms,
                fit.blocks,
                lang.t("blocks", "блоков"),
                lang.t("max radius", "максимальный радиус"),
                fit.max_radius_px
            ))
            .small()
            .weak(),
        );
    }
    if state.options.distort.is_identity() {
        ui.label(
            egui::RichText::new(lang.t(
                "positive k1 removes pincushion (VILTROX needs +0.002..+0.01); a 24 mm zoom needs about -0.03",
                "положительный k1 убирает подушкообразную дисторсию (VILTROX нужно +0.002..+0.01); зуму 24 мм нужно около -0.03",
            ))
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
                "{}: {} {corner:.1} px, k2 {:+.4}, k3 {:+.5}",
                lang.t(
                    "positive k1 removes pincushion",
                    "положительный k1 убирает подушкообразную дисторсию"
                ),
                lang.t("corner moves", "угол смещается на"),
                state.options.distort.k2,
                state.options.distort.k3
            ))
            .small()
            .weak(),
        );
    }
    ui.horizontal(|ui| {
        ui.label(lang.t("compression", "сжатие"));
        egui::ComboBox::from_id_salt("pixelshift-compression")
            .selected_text(compression_label(lang, state.options.compression))
            .show_ui(ui, |ui| {
                for compression in Compression::ALL {
                    ui.selectable_value(
                        &mut state.options.compression,
                        compression,
                        compression_label(lang, compression),
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
    ui.checkbox(
        &mut state.options.also_1x,
        lang.t("also write 1x", "также писать 1x"),
    );
    ui.checkbox(
        &mut state.options.align,
        lang.t(
            "align: translation + rotation",
            "выравнивание: сдвиг + поворот",
        ),
    );
    ui.label(
        egui::RichText::new(lang.t(
            "sub-pixel shift and rotation estimation, per frame",
            "оценка субпиксельного сдвига и поворота для каждого кадра",
        ))
        .small()
        .weak(),
    );
    let mut limited = state.options.threads.is_some();
    if ui
        .checkbox(&mut limited, lang.t("limit threads", "ограничить потоки"))
        .changed()
    {
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
    ui.heading(lang.t("Output", "Вывод"));
    if let Some(series) = state.selected_series() {
        let mode = if series.info.unexpected_mode() {
            format!(
                "{} {}",
                series.info.frame_count(),
                lang.t("frames (unexpected)", "кадров (неожиданно)")
            )
        } else {
            format!("{} {}", lang.t("mode", "режим"), series.info.mode)
        };
        ui.label(format!(
            "{}: {mode} -> {} {}",
            lang.t("series", "серия"),
            lang.t("scale", "масштаб"),
            series.info.suggested_scale
        ));
        if series.info.unexpected_mode() {
            ui.label(
                egui::RichText::new(lang.t(
                    "frame count is not 4/8/16/32: pick the scale by hand",
                    "число кадров не 4/8/16/32: выберите масштаб вручную",
                ))
                .small()
                .color(ui.visuals().warn_fg_color),
            );
        }
    }
    if let Some(estimate) = state.output_estimate() {
        ui.label(format!(
            "{} x {}  {} {}  {:.2} MP",
            estimate.width,
            estimate.height,
            lang.t("scale", "масштаб"),
            estimate.scale,
            estimate.megapixels()
        ));
        ui.label(output_size_line(state, lang, &estimate));
        if let Some((needed, free)) = state.space_shortfall() {
            let _ = needed;
            ui.label(
                egui::RichText::new(format!(
                    "{} {}",
                    lang.t("free on the output volume:", "свободно на целевом томе:"),
                    pipeline::fmt_bytes(free)
                ))
                .small()
                .color(Color32::from_rgb(230, 90, 90)),
            );
        }
    }
    ui.label(
        egui::RichText::new(lang.t(
            "linear camera-native RGB after black subtraction (0..white-black), white balance NOT applied; AsShotNeutral and the colour matrix stay in the DNG metadata",
            "линейный RGB в родном для камеры пространстве после вычитания чёрного (0..white-black), баланс белого НЕ применяется; AsShotNeutral и цветовая матрица остаются в метаданных DNG",
        ))
        .small()
        .weak(),
    );
}

fn scale_label(lang: Lang, mode: ScaleMode) -> &'static str {
    match mode {
        ScaleMode::Auto => lang.t("auto", "авто"),
        ScaleMode::One => "1x",
        ScaleMode::Two => "2x",
    }
}

fn format_label(lang: Lang, format: OutFormat) -> &'static str {
    match format {
        OutFormat::Dng => lang.t("DNG (linear)", "DNG (линейный)"),
        OutFormat::Tiff => "TIFF 16-bit",
    }
}

fn flavor_label(lang: Lang, flavor: DngFlavor) -> &'static str {
    match flavor {
        DngFlavor::LinearRaw => lang.t("linearraw (full colour)", "linearraw (полный цвет)"),
        DngFlavor::Rgb => "rgb (16-bit RGB)",
        DngFlavor::Cfa => lang.t("cfa (Bayer mosaic)", "cfa (мозаика Байера)"),
    }
}

fn compression_label(lang: Lang, compression: Compression) -> &'static str {
    match compression {
        Compression::None => lang.t("none", "без сжатия"),
        Compression::Lossless => lang.t("lossless", "без потерь"),
    }
}

fn stage_label(lang: Lang, stage: Stage) -> &'static str {
    match stage {
        Stage::Idle => lang.t("idle", "ожидание"),
        Stage::Scanning => lang.t("scanning", "поиск серий"),
        Stage::Decoding => lang.t("decoding", "декодирование"),
        Stage::Aligning => lang.t("aligning", "выравнивание"),
        Stage::Merging => lang.t("merging", "объединение"),
        Stage::Previewing => lang.t("preview", "предпросмотр"),
        Stage::Saving => lang.t("saving", "сохранение"),
        Stage::Done => lang.t("done", "готово"),
    }
}

fn source_label(lang: Lang, source: ViewSource) -> &'static str {
    match source {
        ViewSource::Merged => lang.t("Merged", "Объединённое"),
        ViewSource::Reference => lang.t("Reference", "Эталон"),
    }
}

fn output_size_line(state: &GuiState, lang: Lang, estimate: &ResultInfo) -> String {
    let uncompressed = pipeline::fmt_bytes(estimate.bytes());
    match (state.options.format, state.options.compression) {
        (_, Compression::Lossless) => format!(
            "{} {}",
            uncompressed,
            lang.t(
                "uncompressed estimate; lossless is typically ~2x smaller",
                "оценка без сжатия; lossless обычно даёт файл вдвое меньше",
            )
        ),
        _ => format!(
            "{} {}",
            uncompressed,
            lang.t("uncompressed 16-bit RGB", "без сжатия, 16-bit RGB")
        ),
    }
}

fn bottom_panel(state: &mut GuiState, lang: Lang, ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(stage_label(lang, state.stage)).strong());
        ui.add(
            egui::ProgressBar::new(state.progress)
                .desired_width(240.0)
                .show_percentage(),
        );
        ui.label(format!(
            "{:.0} ms",
            state.total_time().as_secs_f64() * 1000.0
        ));
        if state.crop_pending {
            ui.spinner();
        }
        if let Some(series) = state.selected_series() {
            if let Some(stats) = &series.stats {
                ui.separator();
                ui.label(format!(
                    "{} {}  {} {:.2}  {} {:.2}  {} {}",
                    lang.t("frames", "кадров"),
                    stats.frames,
                    lang.t("samples/channel", "отсчётов/канал"),
                    stats.mean_samples_per_channel,
                    lang.t("min coverage", "мин. покрытие"),
                    stats.min_coverage,
                    lang.t("filled", "заполнено"),
                    stats.filled_pixels
                ));
            }
        }
    });
    if let Some(error) = state.error.clone() {
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new(lang.t("error", "ошибка"))
                    .color(Color32::RED)
                    .strong(),
            );
            ui.label(error);
            if ui.button(lang.t("dismiss", "скрыть")).clicked() {
                state.error = None;
            }
        });
    }
    if let Some(series) = state.selected_series() {
        if let Some(result) = series.result {
            ui.label(format!(
                "{} {} x {}  {} {}  {:.2} MP  white {}",
                lang.t("result", "результат"),
                result.width,
                result.height,
                lang.t("scale", "масштаб"),
                result.scale,
                result.megapixels(),
                result.white
            ));
        }
        for (path, bytes) in &series.saved {
            ui.label(format!(
                "{} {} ({})",
                lang.t("saved", "сохранено"),
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

fn canvas(state: &mut GuiState, cache: &mut PreviewCache, lang: Lang, ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        let previous = state.view.source;
        ui.selectable_value(
            &mut state.view.source,
            ViewSource::Merged,
            source_label(lang, ViewSource::Merged),
        );
        ui.selectable_value(
            &mut state.view.source,
            ViewSource::Reference,
            source_label(lang, ViewSource::Reference),
        );
        if state.view.source != previous {
            state.view.fitted = true;
            state.view.zoom = 0.0;
            state.requests.push(Request::View {
                source: state.view.source,
                rect: None,
                step: 1,
            });
        }
        ui.checkbox(&mut state.view.crop, lang.t("1:1 crop", "кроп 1:1"));
        ui.checkbox(&mut state.view.grid, lang.t("grid", "сетка"));
        if ui.button(lang.t("fit", "вписать")).clicked() {
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
            lang.t(
                "no merged image yet: open a folder and press Merge",
                "объединённого изображения ещё нет: откройте папку и нажмите «Объединить»",
            ),
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

    // Level of detail: the fitted preview until it is magnified past its own
    // resolution, then a crop at the pyramid step that fits the visible window
    // into one viewport-sized texture (the "1:1 crop" box forces full
    // resolution samples).
    let step = if state.view.crop && state.view.zoom >= 1.0 {
        Some(1u32)
    } else {
        state
            .sample_grid()
            .and_then(|grid| lod_step(state.view.zoom, viewport, grid, MAX_PREVIEW_DIM))
    };
    let wanted = step.and_then(|step| {
        visible_rect(state.view.center, viewport, state.view.zoom, dims, 0.25)
            .map(|rect| CropWant { rect, step })
    });
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
    // Straight grid for manual lens alignment: pinned to the image, shown
    // while a distortion slider is being dragged even when toggled off.
    if state.view.grid || state.distortion_being_edited() {
        draw_grid(&painter, rect, center, zoom, dims, ui.visuals().dark_mode);
    }
    // Drag-and-drop feedback: files hovering over the window promise to load.
    if ui.ctx().input(|i| !i.raw.hovered_files.is_empty()) {
        painter.rect_filled(rect, 4.0, Color32::from_black_alpha(150));
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            lang.t(
                "drop a folder or raw files to open",
                "отпустите папку или raw-файлы, чтобы открыть",
            ),
            egui::FontId::proportional(18.0),
            Color32::WHITE,
        );
    }
    if loading {
        painter.text(
            rect.right_top() + Vec2::new(-8.0, 8.0),
            egui::Align2::RIGHT_TOP,
            lang.t("loading detail", "загрузка деталей"),
            egui::FontId::monospace(12.0),
            Color32::from_rgb(230, 200, 120),
        );
    }

    let label = format!(
        "{}  {} x {}  {:.0}%{}  {}",
        source_label(lang, state.view.source),
        dims.0,
        dims.1,
        zoom * 100.0,
        if state.view.crop {
            lang.t("  crop", "  кроп")
        } else {
            ""
        },
        match step {
            None => lang.t("base", "база").to_owned(),
            Some(1) => lang.t("detail 1:1", "детали 1:1").to_owned(),
            Some(s) => format!("{} 1/{s}", lang.t("detail", "детали")),
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

/// Straight-line grid over the image extent, for manual distortion
/// alignment: scene lines that should be straight are bent against it. The
/// cell size is a power of two in image pixels, chosen so one cell spans
/// 80..160 screen pixels, and only lines inside the visible canvas are
/// emitted.
fn draw_grid(
    painter: &egui::Painter,
    canvas: Rect,
    center: [f32; 2],
    zoom: f32,
    dims: (usize, usize),
    dark: bool,
) {
    if zoom <= 0.0 || dims.0 == 0 || dims.1 == 0 {
        return;
    }
    let to_screen = |x: f32, y: f32| -> egui::Pos2 {
        canvas.center() + Vec2::new((x - center[0]) * zoom, (y - center[1]) * zoom)
    };
    let (weak, strong) = if dark {
        (
            Color32::from_white_alpha(60),
            Color32::from_white_alpha(110),
        )
    } else {
        (
            Color32::from_black_alpha(45),
            Color32::from_black_alpha(80),
        )
    };
    let mut step = 1.0f32;
    while step * zoom < 80.0 {
        step *= 2.0;
    }
    // Visible image range: the canvas window intersected with the image.
    let top_left = canvas.center() + Vec2::new(-center[0] * zoom, -center[1] * zoom);
    let x_start = ((canvas.left() - top_left.x) / zoom).floor().clamp(0.0, dims.0 as f32);
    let x_end = ((canvas.right() - top_left.x) / zoom)
        .ceil()
        .clamp(0.0, dims.0 as f32);
    let y_start = ((canvas.top() - top_left.y) / zoom).floor().clamp(0.0, dims.1 as f32);
    let y_end = ((canvas.bottom() - top_left.y) / zoom)
        .ceil()
        .clamp(0.0, dims.1 as f32);
    let center_x = dims.0 as f32 * 0.5;
    let center_y = dims.1 as f32 * 0.5;
    let mut x = (x_start / step).ceil() * step;
    while x <= x_end {
        let color = if (x - center_x).abs() < step * 0.25 { strong } else { weak };
        painter.line_segment(
            [to_screen(x, y_start), to_screen(x, y_end)],
            egui::Stroke::new(1.0, color),
        );
        x += step;
    }
    let mut y = (y_start / step).ceil() * step;
    while y <= y_end {
        let color = if (y - center_y).abs() < step * 0.25 { strong } else { weak };
        painter.line_segment(
            [to_screen(x_start, y), to_screen(x_end, y)],
            egui::Stroke::new(1.0, color),
        );
        y += step;
    }
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
    use crate::app::model::{
        EditOptions, FrameInfo, PreviewSlot, SeriesView, Stage, MAX_PREVIEW_DIM,
    };
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
        let image = demo::encode(&result, None, MAX_PREVIEW_DIM, [1.244, 1.0, 2.010], 1);
        let mut state = GuiState::new(Mode::Demo);
        state.series = vec![series];
        state.previews[0] = Some(PreviewSlot {
            image,
            source: ViewSource::Merged,
            rect: None,
            step: 1,
            generation: 1,
        });
        state.stage = Stage::Done;
        state
    }

    fn run_frame(state: &mut GuiState, cache: &mut PreviewCache) -> usize {
        let ctx = egui::Context::default();
        let input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1280.0, 800.0),
            )),
            ..Default::default()
        };
        let mut prefs = UiPrefs::default();
        let mut output = ctx.run_ui(input, |ui| {
            let _ = draw(state, cache, &mut prefs, ui);
        });
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
        assert!(!cache.textures.is_empty(), "preview texture was not uploaded");
        assert_eq!(cache.textures[0].1.size_vec2(), Vec2::new(640.0, 424.0));
        assert!(state.view.zoom > 0.0 && state.view.zoom < 2.0);
    }

    #[test]
    fn left_panel_tabs_render_their_content() {
        let mut state = demo_state();
        let mut cache = PreviewCache::default();
        for tab in [LeftTab::Series, LeftTab::Params, LeftTab::Settings] {
            state.left_tab = tab;
            let shapes = run_frame(&mut state, &mut cache);
            assert!(shapes > 0, "{tab:?} tab produced no shapes");
        }
    }

    #[test]
    fn alignment_grid_draws_and_follows_the_distortion_sliders() {
        let mut state = demo_state();
        state.view.grid = true;
        state.view.fitted = false;
        state.view.zoom = 1.0;
        let mut cache = PreviewCache::default();
        assert!(run_frame(&mut state, &mut cache) > 0, "grid frame drew nothing");

        // Toggled off, the grid still appears while a slider is being dragged.
        state.view.grid = false;
        state.options.distort = crate::distort::Distortion::new(0.001, 0.0);
        state.poll_distort(std::time::Instant::now());
        assert!(state.distortion_being_edited());
        assert!(run_frame(&mut state, &mut cache) > 0);
        // Once the debounce commits the slider move, it hides again.
        let later = std::time::Instant::now() + crate::app::model::DISTORT_DEBOUNCE;
        assert!(state.poll_distort(later));
        assert!(!state.distortion_being_edited());
    }

    #[test]
    fn options_widgets_edit_the_state() {
        let mut state = demo_state();
        state.options = EditOptions::default();
        state.options.scale = ScaleMode::Two;
        state.options.format = OutFormat::Tiff;
        state.options.compression = Compression::Lossless;
        let mut cache = PreviewCache::default();
        run_frame(&mut state, &mut cache);
        assert_eq!(
            state.options.compression_note(),
            Some("AdobeDeflate + predictor: the size estimate above assumes no compression, expect roughly half")
        );
        let estimate = state.output_estimate().unwrap();
        assert_eq!((estimate.width, estimate.height), (1280, 848));
    }
}
