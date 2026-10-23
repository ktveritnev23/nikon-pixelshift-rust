//! Interface language and the translations of the control labels.
//!
//! Only widget labels are translated. Log lines, errors, file names, DNG
//! flavour names, CLI flags and units stay in English.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Lang {
    #[default]
    En,
    Ru,
}

impl Lang {
    /// The system locale (ru-* -> Russian, anything else -> English).
    pub fn detect() -> Lang {
        Self::from_locale(sys_locale::get_locale().as_deref())
    }

    /// Two letter code used in the config file and on the command line.
    pub fn code(self) -> &'static str {
        match self {
            Self::En => "en",
            Self::Ru => "ru",
        }
    }

    /// Parse `ru`, `ru-RU`, `ru_RU`, `en_US` and so on; unknown -> None.
    pub fn from_code(code: &str) -> Option<Lang> {
        let primary = code.split(['-', '_']).next().unwrap_or_default();
        match primary.to_ascii_lowercase().as_str() {
            "en" => Some(Self::En),
            "ru" => Some(Self::Ru),
            _ => None,
        }
    }

    /// Pick one of the two strings written next to each other at the use site.
    pub fn t<'a>(self, en: &'a str, ru: &'a str) -> &'a str {
        match self {
            Self::En => en,
            Self::Ru => ru,
        }
    }

    fn from_locale(locale: Option<&str>) -> Lang {
        locale.and_then(Lang::from_code).unwrap_or(Lang::En)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locale_codes_map_to_the_supported_languages() {
        assert_eq!(Lang::from_code("ru"), Some(Lang::Ru));
        assert_eq!(Lang::from_code("ru-RU"), Some(Lang::Ru));
        assert_eq!(Lang::from_code("ru_RU.UTF-8"), Some(Lang::Ru));
        assert_eq!(Lang::from_code("en_US"), Some(Lang::En));
        assert_eq!(Lang::from_code("en"), Some(Lang::En));
        assert_eq!(Lang::from_code(""), None);
        assert_eq!(Lang::from_code("de-DE"), None);
        assert_eq!(Lang::from_code("russian"), None);
        assert_eq!(Lang::from_code("  "), None);
    }

    #[test]
    fn detection_falls_back_to_english() {
        assert_eq!(Lang::from_locale(Some("ru-RU")), Lang::Ru);
        assert_eq!(Lang::from_locale(Some("ru")), Lang::Ru);
        assert_eq!(Lang::from_locale(Some("en_GB")), Lang::En);
        assert_eq!(Lang::from_locale(Some("de-DE")), Lang::En);
        assert_eq!(Lang::from_locale(None), Lang::En);
        assert!(matches!(Lang::detect(), Lang::En | Lang::Ru));
    }

    #[test]
    fn translation_picks_the_requested_language() {
        assert_eq!(Lang::En.t("Merge", "Объединить"), "Merge");
        assert_eq!(Lang::Ru.t("Merge", "Объединить"), "Объединить");
        assert_eq!(Lang::En.code(), "en");
        assert_eq!(Lang::Ru.code(), "ru");
    }

    #[test]
    fn serde_uses_the_language_code() {
        assert_eq!(ron::to_string(&Lang::Ru).unwrap(), "ru");
        assert_eq!(ron::from_str::<Lang>("en").unwrap(), Lang::En);
        assert!(ron::from_str::<Lang>("de").is_err());
        assert!(ron::from_str::<Lang>("42").is_err());
    }
}
