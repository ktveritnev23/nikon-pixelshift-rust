//! Style files: the whole egui look (visuals) plus the canvas colours of the
//! GUI, stored as hand-editable JSON next to the preferences
//! (`<storage dir>/styles/*.json`).
//!
//! The built-in dark and light styles are the exact previous hardcoded look;
//! they are written into the styles directory on first launch so a user can
//! copy, edit and re-load them. `install` in prefs.rs applies the active
//! style after the theme, so a file overrides the built-in it is based on.

use anyhow::{Context, Result};
use eframe::egui;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, RwLock};

/// The canvas colours that used to be hardcoded in ui.rs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CanvasColors {
    /// Canvas background behind the image.
    pub background: [u8; 3],
    /// Empty-state drag-and-drop field fill.
    pub drop_fill: [u8; 3],
    /// Drag-and-drop field stroke when no file hovers.
    pub drop_stroke: [u8; 3],
    /// Drag-and-drop field stroke while a file hovers (also the title colour).
    pub drop_accent: [u8; 3],
    /// Empty-state title colour.
    pub drop_title: [u8; 3],
    /// "loading detail" hint.
    pub loading_text: [u8; 3],
    /// Bottom-left canvas status line.
    pub status_text: [u8; 3],
    /// Stroke around the displayed detail crop.
    pub crop_outline: [u8; 3],
    /// Alignment grid lines.
    pub grid_line: [u8; 4],
    /// Brighter centre lines of the alignment grid.
    pub grid_center: [u8; 4],
}

/// One style: a display name, whether it is a dark look (drives semantic
/// choices like the grid colour pair) and the full egui visuals.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StyleFile {
    pub name: String,
    pub dark: bool,
    pub visuals: egui::Visuals,
    pub canvas: CanvasColors,
}

impl StyleFile {
    pub fn file_stem(&self) -> String {
        // A filesystem-safe name derived from the display name.
        self.name
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == ' ' {
                    c
                } else {
                    '-'
                }
            })
            .collect::<String>()
            .trim()
            .replace(' ', "_")
            .to_lowercase()
    }

    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).context("serializing the style")
    }

    pub fn from_json(text: &str) -> Result<Self> {
        serde_json::from_str(text).context("parsing the style file")
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        std::fs::write(path, self.to_json()?)
            .with_context(|| format!("writing {}", path.display()))
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        Self::from_json(&text)
    }

    /// The previous hardcoded dark look.
    pub fn builtin_dark() -> Self {
        Self {
            name: "Dark (built-in)".to_string(),
            dark: true,
            visuals: egui::Visuals::dark(),
            canvas: CanvasColors {
                background: [18, 18, 20],
                drop_fill: [26, 26, 30],
                drop_stroke: [90, 90, 110],
                drop_accent: [120, 180, 250],
                drop_title: [200, 200, 210],
                loading_text: [230, 200, 120],
                status_text: [180, 180, 190],
                crop_outline: [90, 90, 120],
                grid_line: [255, 255, 255, 60],
                grid_center: [255, 255, 255, 110],
            },
        }
    }

    /// The previous hardcoded light look (the raised weak-text contrast is
    /// applied on top by prefs::install, like for every light look).
    pub fn builtin_light() -> Self {
        Self {
            name: "Light (built-in)".to_string(),
            dark: false,
            visuals: egui::Visuals::light(),
            canvas: CanvasColors {
                background: [18, 18, 20],
                drop_fill: [26, 26, 30],
                drop_stroke: [90, 90, 110],
                drop_accent: [120, 180, 250],
                drop_title: [200, 200, 210],
                loading_text: [230, 200, 120],
                status_text: [180, 180, 190],
                crop_outline: [90, 90, 120],
                grid_line: [0, 0, 0, 45],
                grid_center: [0, 0, 0, 80],
            },
        }
    }
}

/// Directory of the style files: `<eframe storage dir>/styles`.
pub fn styles_dir() -> Option<PathBuf> {
    eframe::storage_dir("pixelshift").map(|dir| dir.join("styles"))
}

/// Load every *.json in the directory, best effort; missing directory = none.
pub fn load_dir(dir: &Path) -> Vec<StyleFile> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut paths: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()).map(|e| e == "json").unwrap_or(false))
        .collect();
    paths.sort();
    for path in paths {
        match StyleFile::load(&path) {
            Ok(style) => out.push(style),
            Err(error) => {
                // Not fatal: a broken user file must not keep the app from
                // starting; the error surfaces in the log line below.
                eprintln!("style file {} could not be loaded: {error:#}", path.display());
            }
        }
    }
    out
}

/// Write the built-in styles into an empty directory (first launch), so the
/// current look is right there in editable files.
pub fn ensure_builtin_files(dir: &Path) -> Vec<PathBuf> {
    let mut written = Vec::new();
    let entries = std::fs::read_dir(dir).map(|d| d.count()).unwrap_or(0);
    if entries == 0 {
        for builtin in [StyleFile::builtin_dark(), StyleFile::builtin_light()] {
            let path = dir.join(format!("{}.json", builtin.file_stem()));
            if builtin.save(&path).is_ok() {
                written.push(path);
            }
        }
    }
    written
}

fn active_slot() -> &'static RwLock<Arc<StyleFile>> {
    static SLOT: OnceLock<RwLock<Arc<StyleFile>>> = OnceLock::new();
    SLOT.get_or_init(|| RwLock::new(Arc::new(StyleFile::builtin_dark())))
}

/// The style the canvas colours are read from; set by prefs::install before
/// the first frame and on every preference change.
pub fn active() -> Arc<StyleFile> {
    active_slot()
        .read()
        .map(|s| s.clone())
        .unwrap_or_else(|_| Arc::new(StyleFile::builtin_dark()))
}

pub fn set_active(style: StyleFile) {
    if let Ok(mut slot) = active_slot().write() {
        *slot = Arc::new(style);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn style_file_round_trips_through_json() {
        for builtin in [StyleFile::builtin_dark(), StyleFile::builtin_light()] {
            let text = builtin.to_json().unwrap();
            let back = StyleFile::from_json(&text).unwrap();
            assert_eq!(back, builtin);
        }
    }

    #[test]
    fn json_is_editable_and_carries_key_colors() {
        let text = StyleFile::builtin_dark().to_json().unwrap();
        assert!(text.contains("\"name\": \"Dark (built-in)\""));
        assert!(text.contains("\"dark_mode\": true"), "full visuals must be in the file");
        assert!(
            text.contains("\"background\"") && text.contains("18,"),
            "canvas background must be in the file"
        );
        // A hand-edited value survives the round trip.
        let edited = text.replace("\"name\": \"Dark (built-in)\"", "\"name\": \"My dark\"");
        let back = StyleFile::from_json(&edited).unwrap();
        assert_eq!(back.name, "My dark");
    }

    #[test]
    fn broken_json_is_reported_not_fatal() {
        assert!(StyleFile::from_json("{ not json").is_err());
    }

    #[test]
    fn file_stem_is_filesystem_safe() {
        let mut style = StyleFile::builtin_dark();
        style.name = "Тёмная тема / v2!".to_string();
        assert!(!style.file_stem().is_empty());
        assert!(!style.file_stem().contains('/'));
    }
}
