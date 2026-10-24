//! Saved UI preferences (language, theme, font) and their application to egui.

use crate::app::i18n::Lang;
use eframe::egui;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex, OnceLock};

/// The one key the preferences are stored under.
pub const STORAGE_KEY: &str = "ui_prefs";

const SYSTEM_SANS: &str = "system-sans-serif";
const SYSTEM_MONO: &str = "system-monospace";

/// egui's light style inherits the dark weak-text alpha (0.6), which leaves
/// secondary text nearly invisible on the near-white panel fill.
const LIGHT_WEAK_TEXT_ALPHA: f32 = 0.85;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThemeMode {
    #[default]
    System,
    Dark,
    Light,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FontMode {
    #[default]
    System,
    Builtin,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct UiPrefs {
    pub lang: Lang,
    pub theme: ThemeMode,
    pub font: FontMode,
}

impl Default for UiPrefs {
    fn default() -> Self {
        Self {
            lang: Lang::detect(),
            theme: ThemeMode::System,
            font: FontMode::System,
        }
    }
}

impl UiPrefs {
    /// Apply the theme and the fonts to `ctx`. Safe to call on every change.
    pub fn install(&self, ctx: &egui::Context) {
        ctx.set_theme(match self.theme {
            ThemeMode::System => egui::ThemePreference::System,
            ThemeMode::Dark => egui::ThemePreference::Dark,
            ThemeMode::Light => egui::ThemePreference::Light,
        });

        // Touch both stored styles, not just the active one: when the system
        // theme switches at runtime egui picks the other style, and the raised
        // contrast must survive that switch. Dark keeps egui's default.
        ctx.all_styles_mut(|style| {
            if !style.visuals.dark_mode {
                style.visuals.weak_text_alpha =
                    style.visuals.weak_text_alpha.max(LIGHT_WEAK_TEXT_ALPHA);
            }
        });

        let mut definitions = egui::FontDefinitions::default();
        if self.font == FontMode::System {
            let mut installed = false;
            if let Some(fonts) = system_fonts() {
                installed |= insert_system_font(
                    &mut definitions,
                    fonts.sans.as_ref(),
                    SYSTEM_SANS,
                    egui::FontFamily::Proportional,
                );
                installed |= insert_system_font(
                    &mut definitions,
                    fonts.mono.as_ref(),
                    SYSTEM_MONO,
                    egui::FontFamily::Monospace,
                );
            }
            if !installed {
                warn("system font unavailable: falling back to the built-in Ubuntu-Light and Hack");
            }
        }
        ctx.set_fonts(definitions);
    }

    /// Settings rows shown on the settings tab of the left panel; true when
    /// one of the values changed.
    pub fn settings_body(&mut self, ui: &mut egui::Ui) -> bool {
        let lang = self.lang;
        let mut changed = false;
        ui.heading(lang.t("Settings", "Настройки"));
        ui.horizontal(|ui| {
            ui.label(lang.t("Language", "Язык"));
            changed |= ui
                .selectable_value(&mut self.lang, Lang::En, "English")
                .changed();
            changed |= ui
                .selectable_value(&mut self.lang, Lang::Ru, "Русский")
                .changed();
        });
        ui.horizontal(|ui| {
            ui.label(lang.t("Theme", "Тема"));
            changed |= ui
                .selectable_value(&mut self.theme, ThemeMode::System, lang.t("System", "Системная"))
                .changed();
            changed |= ui
                .selectable_value(&mut self.theme, ThemeMode::Dark, lang.t("Dark", "Тёмная"))
                .changed();
            changed |= ui
                .selectable_value(&mut self.theme, ThemeMode::Light, lang.t("Light", "Светлая"))
                .changed();
        });
        ui.horizontal(|ui| {
            ui.label(lang.t("Font", "Шрифт"));
            changed |= ui
                .selectable_value(&mut self.font, FontMode::System, lang.t("System", "Системный"))
                .changed();
            changed |= ui
                .selectable_value(&mut self.font, FontMode::Builtin, lang.t("Built-in", "Встроенный"))
                .changed();
        });
        changed
    }
}

/// One message about a font fallback, to be written into the GUI log.
pub fn take_font_warning() -> Option<String> {
    warning().lock().ok().and_then(|mut slot| slot.take())
}

fn warn(message: &str) {
    if let Ok(mut slot) = warning().lock() {
        slot.get_or_insert_with(|| message.to_owned());
    }
}

fn warning() -> &'static Mutex<Option<String>> {
    static WARNING: OnceLock<Mutex<Option<String>>> = OnceLock::new();
    WARNING.get_or_init(|| Mutex::new(None))
}

struct SystemFont {
    name: String,
    path: Option<String>,
    blob: fontique::Blob<u8>,
    index: u32,
    covers_cyrillic: bool,
}

struct SystemFonts {
    sans: Option<SystemFont>,
    mono: Option<SystemFont>,
}

fn system_fonts() -> Option<&'static SystemFonts> {
    static FONTS: OnceLock<Option<SystemFonts>> = OnceLock::new();
    FONTS.get_or_init(load_system_fonts).as_ref()
}

/// The system sans-serif font, `family (path)`, when one was found.
pub fn system_font_description() -> Option<String> {
    system_fonts()
        .and_then(|fonts| fonts.sans.as_ref())
        .map(|font| {
            let mut text = match &font.path {
                Some(path) => format!("{} ({path})", font.name),
                None => font.name.clone(),
            };
            text.push_str(if font.covers_cyrillic {
                ", cyrillic: yes"
            } else {
                ", cyrillic: no"
            });
            text
        })
}

pub fn system_font_available() -> bool {
    system_fonts().is_some_and(|fonts| fonts.sans.is_some())
}

pub fn system_font_covers_cyrillic() -> bool {
    system_fonts().is_some_and(|fonts| fonts.sans.as_ref().is_some_and(|font| font.covers_cyrillic))
}

fn load_system_fonts() -> Option<SystemFonts> {
    let mut collection = fontique::Collection::new(fontique::CollectionOptions {
        shared: false,
        system_fonts: true,
    });
    let mut cache = fontique::SourceCache::new(fontique::SourceCacheOptions::default());
    let sans = query_generic(
        &mut collection,
        &mut cache,
        fontique::GenericFamily::SansSerif,
    );
    let mono = query_generic(
        &mut collection,
        &mut cache,
        fontique::GenericFamily::Monospace,
    );
    if sans.is_none() && mono.is_none() {
        return None;
    }
    Some(SystemFonts { sans, mono })
}

fn query_generic(
    collection: &mut fontique::Collection,
    cache: &mut fontique::SourceCache,
    generic: fontique::GenericFamily,
) -> Option<SystemFont> {
    let (family, index, blob, covers_cyrillic) = {
        let mut query = collection.query(cache);
        query.set_families([fontique::QueryFamily::Generic(generic)]);
        let mut found = None;
        query.matches_with(|font| {
            let covers = font
                .charmap()
                .map(|charmap| charmap.map('Ж').is_some())
                .unwrap_or(false);
            found = Some((font.family, font.index, font.blob.clone(), covers));
            fontique::QueryStatus::Stop
        });
        found?
    };
    let name = collection
        .family_name(family.0)
        .unwrap_or("unknown")
        .to_owned();
    let family_info = collection.family(family.0);
    let path = family_info.as_ref().and_then(|info| {
        let font = info.fonts().get(family.1)?;
        match font.source().kind() {
            fontique::SourceKind::Path(path) => Some(path.display().to_string()),
            fontique::SourceKind::Memory(_) => None,
        }
    });
    Some(SystemFont {
        name,
        path,
        blob,
        index,
        covers_cyrillic,
    })
}

fn insert_system_font(
    definitions: &mut egui::FontDefinitions,
    font: Option<&SystemFont>,
    name: &str,
    family: egui::FontFamily,
) -> bool {
    let Some(font) = font else {
        return false;
    };
    let (blob, _id) = font.blob.clone().into_raw_parts();
    definitions.font_data.insert(
        name.to_owned(),
        Arc::new(egui::FontData::from_blob(blob, font.index)),
    );
    definitions
        .families
        .entry(family)
        .or_default()
        .insert(0, name.to_owned());
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefs_round_trip_through_ron() {
        let prefs = UiPrefs {
            lang: Lang::Ru,
            theme: ThemeMode::Dark,
            font: FontMode::Builtin,
        };
        let text = ron::to_string(&prefs).unwrap();
        assert!(text.contains("ru") && text.contains("dark") && text.contains("builtin"));
        assert_eq!(ron::from_str::<UiPrefs>(&text).unwrap(), prefs);
    }

    #[test]
    fn unknown_values_and_fields_are_errors() {
        let unknown_value = "(lang:de,theme:dark,font:system)";
        assert!(ron::from_str::<UiPrefs>(unknown_value).is_err());
        let unknown_field = "(lang:ru,theme:dark,font:system,extra:1)";
        assert!(ron::from_str::<UiPrefs>(unknown_field).is_err());
        assert!(ron::from_str::<UiPrefs>("this is not ron").is_err());
    }

    #[test]
    fn missing_fields_fall_back_to_defaults() {
        let partial: UiPrefs = ron::from_str("(theme:light)").unwrap();
        assert_eq!(partial.theme, ThemeMode::Light);
        assert_eq!(partial.font, FontMode::System);
    }

    #[test]
    fn theme_and_font_serialize_as_names() {
        assert_eq!(ron::to_string(&ThemeMode::Light).unwrap(), "light");
        assert_eq!(ron::to_string(&FontMode::Builtin).unwrap(), "builtin");
        assert!(ron::from_str::<ThemeMode>("sepia").is_err());
    }

    fn layout_cyrillic(ctx: &egui::Context) -> (f32, usize) {
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(800.0, 600.0),
            )),
            ..Default::default()
        };
        let mut output = ctx.run_ui(input, |_| {});
        output.textures_delta.clear();
        ctx.fonts_mut(|fonts| {
            let galley = fonts.layout_no_wrap(
                "Ж".to_owned(),
                egui::FontId::proportional(14.0),
                egui::Color32::WHITE,
            );
            let glyphs = galley
                .rows
                .iter()
                .map(|row| row.row.glyphs.len())
                .sum::<usize>();
            (galley.size().x, glyphs)
        })
    }

    #[test]
    fn system_and_builtin_fonts_render_cyrillic() {
        let builtin = UiPrefs {
            font: FontMode::Builtin,
            ..UiPrefs::default()
        };
        let builtin_ctx = egui::Context::default();
        builtin.install(&builtin_ctx);
        let (builtin_width, builtin_glyphs) = layout_cyrillic(&builtin_ctx);

        let system = UiPrefs {
            font: FontMode::System,
            ..UiPrefs::default()
        };
        let system_ctx = egui::Context::default();
        system.install(&system_ctx);
        let (system_width, system_glyphs) = layout_cyrillic(&system_ctx);

        println!(
            "system font: {:?}, width {system_width:.2}, glyphs {system_glyphs}",
            system_font_description()
        );
        println!("builtin font: width {builtin_width:.2}, glyphs {builtin_glyphs}");

        assert!(system_width > 0.0 && system_glyphs > 0);
        assert!(builtin_width > 0.0 && builtin_glyphs > 0);
        if system_font_available() {
            assert!(
                system_font_covers_cyrillic(),
                "the selected system font has no Cyrillic glyphs"
            );
            assert!(
                (system_width - builtin_width).abs() > 0.01,
                "the system font was selected but the layout equals the built-in one"
            );
        }
    }

    #[test]
    fn install_applies_the_requested_theme() {
        let ctx = egui::Context::default();
        UiPrefs {
            theme: ThemeMode::Dark,
            ..UiPrefs::default()
        }
        .install(&ctx);
        assert_eq!(ctx.theme(), egui::Theme::Dark);
        UiPrefs {
            theme: ThemeMode::Light,
            ..UiPrefs::default()
        }
        .install(&ctx);
        assert_eq!(ctx.theme(), egui::Theme::Light);
    }

    #[test]
    fn light_style_gets_readable_weak_text_and_dark_stays_default() {
        let ctx = egui::Context::default();
        UiPrefs::default().install(&ctx);
        let light = ctx.style_of(egui::Theme::Light).visuals.weak_text_alpha;
        let dark = ctx.style_of(egui::Theme::Dark).visuals.weak_text_alpha;
        assert!(light >= LIGHT_WEAK_TEXT_ALPHA);
        assert!(dark >= 0.6);
        assert!(light > dark);
        // A later system-theme switch must not undo the stored light style.
        ctx.set_theme(egui::ThemePreference::System);
        assert!(ctx.style_of(egui::Theme::Light).visuals.weak_text_alpha >= LIGHT_WEAK_TEXT_ALPHA);
    }

    #[test]
    fn install_is_idempotent() {
        let prefs = UiPrefs {
            font: FontMode::System,
            ..UiPrefs::default()
        };
        let ctx = egui::Context::default();
        prefs.install(&ctx);
        prefs.install(&ctx);
        let (width, glyphs) = layout_cyrillic(&ctx);
        assert!(width > 0.0 && glyphs > 0);
    }
}
