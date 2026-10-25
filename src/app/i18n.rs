//! Interface language and the translations of the control labels.
//!
//! Only widget labels are translated. Log lines, errors, file names, DNG
//! flavour names, CLI flags and units stay in English.
//!
//! English and Russian are written side by side at the use site
//! (`lang.t("Merge", "Объединить")`). Chinese (simplified) and Japanese live
//! in the catalogs below, keyed by the English string; a string missing from
//! a catalog falls back to English, so adding a widget never breaks the
//! other languages.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Lang {
    #[default]
    En,
    Ru,
    Zh,
    Ja,
}

impl Lang {
    /// The system locale (ru-* -> Russian, zh-* -> Chinese, ja-* -> Japanese,
    /// anything else -> English).
    pub fn detect() -> Lang {
        Self::from_locale(sys_locale::get_locale().as_deref())
    }

    /// Two letter code used in the config file and on the command line.
    pub fn code(self) -> &'static str {
        match self {
            Self::En => "en",
            Self::Ru => "ru",
            Self::Zh => "zh",
            Self::Ja => "ja",
        }
    }

    /// Label shown in the language selector itself (always in its own language).
    pub fn label(self) -> &'static str {
        match self {
            Self::En => "English",
            Self::Ru => "Русский",
            Self::Zh => "简体中文",
            Self::Ja => "日本語",
        }
    }

    /// All languages, in selector order.
    pub const ALL: [Lang; 4] = [Lang::En, Lang::Ru, Lang::Zh, Lang::Ja];

    /// Parse `ru`, `ru-RU`, `ru_RU`, `en_US`, `zh-CN`, `ja_JP` and so on;
    /// unknown -> None.
    pub fn from_code(code: &str) -> Option<Lang> {
        let primary = code.split(['-', '_']).next().unwrap_or_default();
        match primary.to_ascii_lowercase().as_str() {
            "en" => Some(Self::En),
            "ru" => Some(Self::Ru),
            "zh" => Some(Self::Zh),
            "ja" => Some(Self::Ja),
            _ => None,
        }
    }

    /// Pick one of the two strings written next to each other at the use site.
    /// For Chinese and Japanese the English text is looked up in the catalog.
    pub fn t<'a>(self, en: &'a str, ru: &'a str) -> &'a str {
        match self {
            Self::En => en,
            Self::Ru => ru,
            Self::Zh => zh(en).unwrap_or(en),
            Self::Ja => ja(en).unwrap_or(en),
        }
    }

    fn from_locale(locale: Option<&str>) -> Lang {
        locale.and_then(Lang::from_code).unwrap_or(Lang::En)
    }
}

/// Simplified Chinese catalog.
fn zh(en: &str) -> Option<&'static str> {
    Some(match en {
        "  crop" => "  裁剪",
        "1:1 crop" => "1:1 裁剪",
        "align: translation + rotation" => "对齐：平移 + 旋转",
        "aligning" => "对齐中",
        "also write 1x" => "同时输出 1x",
        "auto" => "自动",
        "base" => "基础",
        "blocks" => "块",
        "Built-in" => "内置",
        "Camera" => "相机",
        "cfa (Bayer mosaic)" => "cfa（拜耳阵列）",
        "clear queue" => "清空队列",
        "compression" => "压缩",
        "corner moves" => "角部位移",
        "crop to the frames' overlap" => "裁剪到各帧重叠区域",
        "Dark" => "深色",
        "decoding" => "解码中",
        "deconvolve" => "反卷积",
        "detail" => "细节",
        "detail 1:1" => "细节 1:1",
        "dismiss" => "关闭",
        "Distortion k1" => "畸变 k1",
        "DNG (linear)" => "DNG（线性）",
        "DNG main image" => "DNG 主图像",
        "done" => "完成",
        "electronic stabilization" => "电子防抖",
        "error" => "错误",
        "Exposure" => "曝光",
        "file" => "文件",
        "filled" => "补填",
        "fit" => "适应",
        "fit to window" => "适应窗口",
        "Font" => "字体",
        "format" => "格式",
        "Frame" => "帧",
        "Frames" => "帧",
        "frames" => "帧",
        "frames (unexpected)" => "帧（数目异常）",
        "frame count is not 4/8/16/32: pick the scale by hand" => {
            "帧数不是 4/8/16/32：请手动选择倍率"
        }
        "free on the output volume:" => "输出卷剩余空间：",
        "grid" => "网格",
        "idle" => "等待",
        "Language" => "语言",
        "Lens" => "镜头",
        "Light" => "浅色",
        "limit threads" => "限制线程",
        "linearraw (full colour)" => "linearraw（全色彩）",
        "linearraw: full colour resolution; rgb: plain 16-bit RGB; cfa: Bayer mosaic for readers that need a raw sensor pattern" => {
            "linearraw：完整色彩分辨率；rgb：普通 16 位 RGB；cfa：拜耳阵列，供需要原始传感器图案的软件使用"
        }
        "loading detail" => "加载细节",
        "log" => "日志",
        "lossless" => "无损",
        "mask" => "遮罩",
        "max radius" => "最大半径",
        "measured" => "已测量",
        "Merge" => "合并",
        "merged" => "已合并",
        "Merged" => "合并结果",
        "merging" => "合并中",
        "metadata" => "元数据",
        "Metadata" => "元数据",
        "min coverage" => "最小覆盖率",
        "mode" => "模式",
        "motion compensation" => "运动补偿",
        "no series loaded" => "未加载系列",
        "none" => "无",
        "not decoded yet" => "尚未解码",
        "Open files..." => "打开文件…",
        "Open folder..." => "打开文件夹…",
        "Output" => "输出",
        "own merge settings" => "独立合并设置",
        "positive k1 removes pincushion" => "正 k1 消除枕形畸变",
        "Params" => "参数",
        "preview" => "预览",
        "queue all" => "全部入队",
        "queued" => "已排队",
        "Quit" => "退出",
        "re-snapshot current settings" => "采用当前设置",
        "Reference" => "参考帧",
        "render sRGB" => "渲染为 sRGB",
        "reset 0" => "归零",
        "residual" => "残差",
        "result" => "结果",
        "Result" => "结果",
        "samples/channel" => "样本/通道",
        "Save 1x..." => "另存 1x…",
        "Save..." => "保存…",
        "saved" => "已保存",
        "saving" => "保存中",
        "scale" => "倍率",
        "scanning" => "扫描中",
        "sensitivity" => "灵敏度",
        "series" => "系列",
        "Series" => "系列",
        "Settings" => "设置",
        "Shutter count" => "快门计数",
        "Start batch" => "开始批处理",
        "Stop batch" => "停止批处理",
        "sub-pixel shift and rotation estimation, per frame" => "逐帧估计亚像素平移与旋转",
        "System" => "跟随系统",
        "Theme" => "主题",
        "Time" => "时间",
        "take part in the batch queue" => "参与批处理",
        "uncompressed 16-bit RGB" => "未压缩 16 位 RGB",
        _ => return None,
    })
}

/// Japanese catalog.
fn ja(en: &str) -> Option<&'static str> {
    Some(match en {
        "  crop" => "  クロップ",
        "1:1 crop" => "1:1 クロップ",
        "align: translation + rotation" => "整列：平行移動 + 回転",
        "aligning" => "整列中",
        "also write 1x" => "1x も出力",
        "auto" => "自動",
        "base" => "ベース",
        "blocks" => "ブロック",
        "Built-in" => "内蔵",
        "Camera" => "カメラ",
        "cfa (Bayer mosaic)" => "cfa（ベイヤー配列）",
        "clear queue" => "キューを空に",
        "compression" => "圧縮",
        "corner moves" => "四隅の移動量",
        "crop to the frames' overlap" => "全フレームの重なり部分に切り取り",
        "Dark" => "ダーク",
        "decoding" => "デコード中",
        "deconvolve" => "デコンボリューション",
        "detail" => "詳細",
        "detail 1:1" => "詳細 1:1",
        "dismiss" => "閉じる",
        "Distortion k1" => "歪曲収差 k1",
        "DNG (linear)" => "DNG（リニア）",
        "DNG main image" => "DNG メイン画像",
        "done" => "完了",
        "electronic stabilization" => "電子手ブレ補正",
        "error" => "エラー",
        "Exposure" => "露出",
        "file" => "ファイル",
        "filled" => "補完",
        "fit" => "フィット",
        "fit to window" => "ウィンドウに合わせる",
        "Font" => "フォント",
        "format" => "形式",
        "Frame" => "フレーム",
        "Frames" => "フレーム",
        "frames" => "フレーム",
        "frames (unexpected)" => "フレーム（想定外）",
        "frame count is not 4/8/16/32: pick the scale by hand" => {
            "フレーム数が 4/8/16/32 ではありません：倍率を手動で選んでください"
        }
        "free on the output volume:" => "出力ボリュームの空き容量：",
        "grid" => "グリッド",
        "idle" => "待機",
        "Language" => "言語",
        "Lens" => "レンズ",
        "Light" => "ライト",
        "limit threads" => "スレッド数を制限",
        "linearraw (full colour)" => "linearraw（フルカラー）",
        "linearraw: full colour resolution; rgb: plain 16-bit RGB; cfa: Bayer mosaic for readers that need a raw sensor pattern" => {
            "linearraw：フルカラー解像度。rgb：通常の 16 ビット RGB。cfa：生センサー配列を必要とするソフト向けベイヤー配列"
        }
        "loading detail" => "詳細を読み込み中",
        "log" => "ログ",
        "lossless" => "ロスレス",
        "mask" => "マスク",
        "max radius" => "最大半径",
        "measured" => "測定済み",
        "Merge" => "合成",
        "merged" => "合成済み",
        "Merged" => "合成結果",
        "merging" => "合成中",
        "metadata" => "メタデータ",
        "Metadata" => "メタデータ",
        "min coverage" => "最小カバレッジ",
        "mode" => "モード",
        "motion compensation" => "ブレ補正",
        "no series loaded" => "シリーズ未読み込み",
        "none" => "なし",
        "not decoded yet" => "未デコード",
        "Open files..." => "ファイルを開く…",
        "Open folder..." => "フォルダを開く…",
        "Output" => "出力",
        "own merge settings" => "固有の合成設定",
        "positive k1 removes pincushion" => "正の k1 で糸巻き歪みを除去",
        "Params" => "パラメータ",
        "preview" => "プレビュー",
        "queue all" => "すべてキューへ",
        "queued" => "キュー中",
        "Quit" => "終了",
        "re-snapshot current settings" => "現在の設定を取り込む",
        "Reference" => "参照フレーム",
        "render sRGB" => "sRGB で描画",
        "reset 0" => "0 に戻す",
        "residual" => "残差",
        "result" => "結果",
        "Result" => "結果",
        "samples/channel" => "サンプル/チャンネル",
        "Save 1x..." => "1x で保存…",
        "Save..." => "保存…",
        "saved" => "保存済み",
        "saving" => "保存中",
        "scale" => "倍率",
        "scanning" => "検索中",
        "sensitivity" => "感度",
        "series" => "シリーズ",
        "Series" => "シリーズ",
        "Settings" => "設定",
        "Shutter count" => "シャッター数",
        "Start batch" => "バッチ開始",
        "Stop batch" => "バッチ停止",
        "sub-pixel shift and rotation estimation, per frame" => "フレームごとのサブピクセル平行移動と回転の推定",
        "System" => "システムに従う",
        "Theme" => "テーマ",
        "Time" => "日時",
        "take part in the batch queue" => "バッチ処理に参加",
        "uncompressed 16-bit RGB" => "非圧縮 16 ビット RGB",
        _ => return None,
    })
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
        assert_eq!(Lang::from_code("zh"), Some(Lang::Zh));
        assert_eq!(Lang::from_code("zh-CN"), Some(Lang::Zh));
        assert_eq!(Lang::from_code("zh_TW"), Some(Lang::Zh));
        assert_eq!(Lang::from_code("ja"), Some(Lang::Ja));
        assert_eq!(Lang::from_code("ja-JP"), Some(Lang::Ja));
        assert_eq!(Lang::from_code(""), None);
        assert_eq!(Lang::from_code("de-DE"), None);
        assert_eq!(Lang::from_code("russian"), None);
        assert_eq!(Lang::from_code("  "), None);
    }

    #[test]
    fn detection_falls_back_to_english() {
        assert_eq!(Lang::from_locale(Some("ru-RU")), Lang::Ru);
        assert_eq!(Lang::from_locale(Some("zh-CN")), Lang::Zh);
        assert_eq!(Lang::from_locale(Some("ja")), Lang::Ja);
        assert_eq!(Lang::from_locale(Some("en_GB")), Lang::En);
        assert_eq!(Lang::from_locale(Some("de-DE")), Lang::En);
        assert_eq!(Lang::from_locale(None), Lang::En);
        assert!(matches!(
            Lang::detect(),
            Lang::En | Lang::Ru | Lang::Zh | Lang::Ja
        ));
    }

    #[test]
    fn translation_picks_the_requested_language() {
        assert_eq!(Lang::En.t("Merge", "Объединить"), "Merge");
        assert_eq!(Lang::Ru.t("Merge", "Объединить"), "Объединить");
        assert_eq!(Lang::En.code(), "en");
        assert_eq!(Lang::Ru.code(), "ru");
        assert_eq!(Lang::Zh.code(), "zh");
        assert_eq!(Lang::Ja.code(), "ja");
    }

    #[test]
    fn catalogs_translate_and_fall_back() {
        // Catalog hits.
        assert_eq!(Lang::Zh.t("Merge", "Объединить"), "合并");
        assert_eq!(Lang::Ja.t("Merge", "Объединить"), "合成");
        assert_eq!(Lang::Zh.t("Start batch", "x"), "开始批处理");
        assert_eq!(Lang::Ja.t("electronic stabilization", "x"), "電子手ブレ補正");
        // Missing entries fall back to English, never panic.
        assert_eq!(Lang::Zh.t("definitely not in the catalog", "x"), "definitely not in the catalog");
        assert_eq!(Lang::Ja.t("also missing", "x"), "also missing");
    }

    #[test]
    fn serde_uses_the_language_code() {
        assert_eq!(ron::to_string(&Lang::Ru).unwrap(), "ru");
        assert_eq!(ron::to_string(&Lang::Zh).unwrap(), "zh");
        assert_eq!(ron::to_string(&Lang::Ja).unwrap(), "ja");
        assert_eq!(ron::from_str::<Lang>("en").unwrap(), Lang::En);
        assert_eq!(ron::from_str::<Lang>("zh").unwrap(), Lang::Zh);
        assert!(ron::from_str::<Lang>("de").is_err());
        assert!(ron::from_str::<Lang>("42").is_err());
    }
}
