# План: UI-улучшения, многоуровневые превью, сетка дисторсии, lossless-сжатие, фикс цветов референса

## WP0 — Фикс бага: референс с искажёнными цветами (src/app/pipeline.rs, src/merge.rs)

Корень найден: `reference_preview`/`block_channels` (pipeline.rs:412-481) не вычитает чёрный уровень (`frame.black_for(color)`, пер-канальный cblack тоже игнорируется), делит на `white_span()` и умножает **пьедестал** на WB-множители ([1.24, 1, 2.01] → синий сдвиг), плюс gamma 1/2.2 вместо sRGB и нет перцентильной точки белого — референс выглядит темнее/другой контраст.

- В `block_channels` вычитать чёрный уровень каждого канала, клампить в [0, span].
- Вынести из `merge::encode_srgb` (merge.rs:867-898) общий финальный энкодер «WB-применённые f32 → u8 sRGB» (точка белого = 99.5-перцентиль окна + `srgb_u8`) и прогнать через него reference-путь вместо `tone()`.
- Тест: синтетический RawFrame с пьедесталом (black=1008) и значениями, нейтральными после WB; `reference_preview` обязан дать R=G=B (сейчас — сдвиг).

## WP1 — Многоступенчатые превью (model.rs, worker.rs, merge.rs, pipeline.rs, ui.rs)

Сейчас ровно два уровня: база ≤2048 px и полноразмерный кроп видимой области. Плюс два найденных скрытых бага: кропы не ключуются по `ViewSource` (кроп merged может показаться поверх reference) и одна текстурная ручка на базу+кроп (затирают друг друга).

- `Request::View`/`Command::View` получают `step: u32` (степень 2). Уровень выбирается в `lod_level`: step = минимальная степень 2, при которой длинная сторона видимой области / step ≤ MAX_PREVIEW_DIM (2048); лестница 1,2,4,8…; step ≥ full/2048 вырождается в текущий базовый уровень.
- Worker: `merge::preview_leveled(img, rect, step, wb)` — rect выравнивается по границам step, box-усреднение step×step блоков (f64, как в `merge::preview`), затем общий энкодер из WP0; при step=1 совпадает с текущим `preview_crop`. `reference_preview` обобщается тем же step (у него уже есть `downscale_step`, привязать к степени 2).
- Кэш кропов: ключ `(source, rect, step)` вместо голого rect — правит и баг подмены источников; LRU (`CROP_CACHE_LEN`) и `poll_lod`/`crop_inflight` учитывают полный ключ.
- `PreviewCache`: отдельные `TextureHandle` на базовые слоты (merged/reference) и на кропы; кроп уровня step рисуется размером `rect.wh × zoom` (экранный масштаб zoom/step, LINEAR).
- Тесты: unit на leveled crop (средние блоков на синтетике); ui-тест, что кропы разных step/source сосуществуют и не подменяются.

## WP2 — Три вкладки в левой панели (model.rs, ui.rs, prefs.rs)

- `GuiState.left_tab: LeftTab` enum `{Series, Params, Settings}` (default Series, не персистится).
- `left_panel`: горизонтальный ряд вкладок-кнопок + separator. «Серии»: список серий, очередь, `frames_panel`. «Параметры»: весь `options_panel` (ui.rs:308-569). «Настройки»: тело `UiPrefs::settings_ui` (prefs.rs:97-149) выносится в новый метод `settings_body(ui)` (те же ряды Language/Theme/Font); menu_button удаляется.
- `top_bar` больше не рисует меню «Настройки»; сигнал «настройки изменились» возвращается из `left_panel` → `ui::draw` (переставить `changed |=`).
- Обновить тесты prefs.rs (они гоняют menu_button) и ui-тесты; перегенерировать скриншоты (`tools/screenshots.sh` — левая панель на всех снимках меняется).
- Новые строки через `lang.t`: «Параметры»/"Params" и подписи к вкладкам.

## WP3 — Drag-and-drop папок и файлов (mod.rs, ui.rs)

- В `App::ui` после отрисовки: `ctx.input(|i| i.raw.dropped_files)` → собрать пути; фильтр: директории и `.nef/.nrw` (пропущенное — строка лога); непустой список → готовый хук `Request::Scan(paths)` (model.rs:346, обработчик уже существует mod.rs:330-333; `pipeline::resolve_inputs` уже принимает смешанные списки папок/файлов). Дедупликация повторных кадров события.
- Hover-фидбек: при `hovered_files` непустом — полупрозрачная рамка + подпись «Отпустите, чтобы открыть» поверх канваса.
- Тест: unit на фильтрацию путей; событие проверяется вручную (headless не эмулируется).

## WP4 — Сетка для ручной коррекции дисторсии (model.rs, ui.rs)

- `CanvasView.grid: bool`; чекбокс «Сетка»/"Grid" в тулбаре канваса рядом с «1:1 crop».
- Показ: `grid || движется слайдер k1..k3` (пока `state.distort_edited` активен — сетка всплывает автоматически, чтобы править, не отпуская слайдер).
- Рисование поверх изображения (после `paint_slot`, в пределах прямоугольника изображения): линии по полю изображения, шаг в пикселях изображения подбирается под зум (~100–200 экранных px на клетку), полупрозрачный цвет с контрастом к теме. Маппинг image→screen тот же, что в `paint_slot` — сетка прибита к изображению, пользователь выравнивает линии сцены по прямым сетки, подстраивая k1..k3.
- Тест: ui-тест на переключение чекбокса и условие авто-показа при правке слайдера.

## WP5 — Lossless-сжатие: Lossless JPEG для DNG, Deflate+predictor для TIFF

Решения подтверждены: LJ (Compression=7) для DNG — читается LibRaw/dcraw_emu, пройдёт приёмочный прогон; Deflate (Compression=8) + Predictor=2 для TIFF.

- **`src/write/ljpeg.rs` (новый кодер SOF3)**: P=16, Nf=1 (CFA) / 3 (LinearRaw/RGB, чередование по строке), предиктор 2, point transform 0; канонические Huffman-таблицы по гистограммам символов (двухпроходно на полосу) — переиспользовать `HuffmanTable::build` и `BitWriter` из `write/jpeg.rs`, сделав их pub(crate). Каждая полоса (16 строк) — самостоятельный JPEG-поток SOI..EOI; если dcraw не примет per-strip потоки — добавить общий тег JPEGTables (347).
- **dng.rs**: полоса → `ljpeg::encode_strip` вместо Deflate-ветки (Deflate-код DNG удалить — мёртвый путь), TAG_COMPRESSION=7.
- **tiff16.rs**: параметр сжатия в `write_tiff16*`; полоса: горизонтальные разности u16 (predictor 2) → `ZlibEncoder` (flate2 уже зависимость), теги Compression=8, Predictor=2.
- **Разводка опций**: `Compression` enum {None, Lossless}; CLI `--compression none|lossless` (старое значение `deflate` принять как синоним lossless); `WriteOpts` вместо `compress: bool` несёт выбор метода по формату (DNG→LJ, TIFF→Deflate); `OutFormat::use_compression()` возвращает честное значение; **убрать hardcoded `Compression::None` при сохранении из GUI** (worker.rs:491-497 — сейчас комбобокс вообще не влияет на файл).
- **Тесты writers.rs**: LJ DNG — tiffinfo/raw-identify + dcraw_emu декодирует + ключевая проверка: PPM от сжатого DNG побитово равен PPM от несжатого того же изображения; TIFF deflate — дополнить встроенный PY_TIFF декодером predictor 2, round-trip Pillow/ImageMagick, tiffinfo показывает AdobeDeflate+Predictor. Замерить коэффициенты сжатия на реальной серии.
- **Доки**: README (таблица флагов, раздел «Ограничения» — снять «пишется без сжатия», оценка размеров), ARCHITECTURE.md (обе секции вывода), COMPATIBILITY.md — новая таблица «сжатие × читатели» с результатами эмпирической проверки; `compression_note()` переписать.

## Порядок и общая валидация

WP0 → WP1 → WP2 → WP3 → WP4 → WP5 (WP5 независим по коду от UI, но самый объёмный). После каждого WP: `cargo test --lib` зелёный, в конце — полный `cargo test`, при наличии серий `tools/accept.sh`, перегенерация скриншотов. Все новые строки UI — пары EN/RU через `lang.t`.