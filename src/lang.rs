// SPDX-License-Identifier: GPL-3.0-only
//! Language codes, from what the daemon sends to what the model was trained on.
//!
//! The daemon sends a BCP-47 tag (`en`, `zh-CN`, …) narrowed to the model's
//! declared `supported_languages`, or the reserved [`AUTO`]. Qwen3-ASR names its
//! languages in English instead — `English`, `Chinese` — and a forced language
//! reaches it as the words `language English` at the start of its answer. This
//! module is the join between the two, and the reason `backend.toml` can
//! declare ordinary language codes rather than leaking the model's vocabulary
//! into the settings UI.
//!
//! The Python backend this replaces handed the code to the model's wrapper
//! as it was, which capitalizes it to `En`, finds no such language, and fails
//! the request — so a transcription with any language set never succeeded
//! there. Mapping the code first is the fix.
//!
//! Regional subtags are dropped: the model has one Chinese and one Portuguese,
//! so `zh-CN` and `zh-TW` both resolve to `Chinese`. Cantonese is its own
//! language to the model, and is reached by its own code, `yue`.

/// The thirty languages Qwen3-ASR transcribes, as `(BCP-47 primary subtag,
/// model name)`, in the order the checkpoints' `support_languages` lists them.
///
/// The names are the model's own spelling and the prompt uses them verbatim.
/// `backend.toml` declares the same thirty codes for every model, and a test
/// below holds the two lists together.
pub const LANGUAGES: &[(&str, &str)] = &[
    ("zh", "Chinese"),
    ("en", "English"),
    ("yue", "Cantonese"),
    ("ar", "Arabic"),
    ("de", "German"),
    ("fr", "French"),
    ("es", "Spanish"),
    ("pt", "Portuguese"),
    ("id", "Indonesian"),
    ("it", "Italian"),
    ("ko", "Korean"),
    ("ru", "Russian"),
    ("th", "Thai"),
    ("vi", "Vietnamese"),
    ("ja", "Japanese"),
    ("tr", "Turkish"),
    ("hi", "Hindi"),
    ("ms", "Malay"),
    ("nl", "Dutch"),
    ("sv", "Swedish"),
    ("da", "Danish"),
    ("fi", "Finnish"),
    ("pl", "Polish"),
    ("cs", "Czech"),
    ("fil", "Filipino"),
    ("fa", "Persian"),
    ("el", "Greek"),
    ("ro", "Romanian"),
    ("hu", "Hungarian"),
    ("mk", "Macedonian"),
];

/// The reserved tag that asks for detection.
///
/// The contract says a backend must accept it and never answer it with
/// `unsupported_language`. Qwen3-ASR detects on its own when the prompt names
/// no language — it then begins its answer with `language X<asr_text>` — so
/// this maps to naming none.
pub const AUTO: &str = "auto";

/// What a request's `language` asks the model for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    /// Let the model detect it.
    Detect,
    /// Force this one, by its model name.
    Forced(&'static str),
}

/// Resolve a request's `language`, or `None` for a tag the model does not
/// transcribe, which the caller reports as `unsupported_language`.
///
/// An absent or blank field detects, as the Python backend did. Matching is on
/// the primary subtag alone and is case-insensitive, so `EN`, `en` and `en-GB`
/// are one language.
#[must_use]
pub fn resolve(tag: Option<&str>) -> Option<Language> {
    let Some(tag) = tag.map(str::trim).filter(|t| !t.is_empty()) else {
        return Some(Language::Detect);
    };
    let primary = tag.split(['-', '_']).next()?.trim();
    if primary.eq_ignore_ascii_case(AUTO) {
        return Some(Language::Detect);
    }
    LANGUAGES
        .iter()
        .find(|(code, _)| primary.eq_ignore_ascii_case(code))
        .map(|(_, name)| Language::Forced(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference's `SUPPORTED_LANGUAGES`, which is also every checkpoint's
    /// `support_languages`: a name here that the model does not know would be
    /// a prompt it was never trained on.
    const MODEL_NAMES: [&str; 30] = [
        "Chinese",
        "English",
        "Cantonese",
        "Arabic",
        "German",
        "French",
        "Spanish",
        "Portuguese",
        "Indonesian",
        "Italian",
        "Korean",
        "Russian",
        "Thai",
        "Vietnamese",
        "Japanese",
        "Turkish",
        "Hindi",
        "Malay",
        "Dutch",
        "Swedish",
        "Danish",
        "Finnish",
        "Polish",
        "Czech",
        "Filipino",
        "Persian",
        "Greek",
        "Romanian",
        "Hungarian",
        "Macedonian",
    ];

    #[test]
    fn every_name_is_one_the_model_knows() {
        let names: Vec<&str> = LANGUAGES.iter().map(|(_, name)| *name).collect();
        assert_eq!(names, MODEL_NAMES);
    }

    /// Every model in the manifest declares exactly the codes of the table,
    /// and its `primary_language` among them.
    #[test]
    fn the_manifest_declares_the_table() {
        let manifest = include_str!("../backend.toml");
        let mut codes: Vec<&str> = LANGUAGES.iter().map(|(code, _)| *code).collect();
        codes.sort_unstable();
        let models = crate::manifest_probe::model_blocks(manifest);
        assert_eq!(models.len(), 2, "two models in backend.toml");
        for block in models {
            let mut declared = crate::manifest_probe::string_list(&block, "supported_languages")
                .expect("supported_languages");
            declared.sort_unstable();
            assert_eq!(declared, codes);
            let primary = crate::manifest_probe::string_field(&block, "primary_language")
                .expect("primary_language");
            assert!(codes.contains(&primary));
        }
    }

    #[test]
    fn codes_resolve_to_names() {
        assert_eq!(resolve(Some("en")), Some(Language::Forced("English")));
        assert_eq!(resolve(Some("EN-gb")), Some(Language::Forced("English")));
        assert_eq!(resolve(Some("zh_TW")), Some(Language::Forced("Chinese")));
        assert_eq!(resolve(Some("yue")), Some(Language::Forced("Cantonese")));
        assert_eq!(resolve(Some("fil")), Some(Language::Forced("Filipino")));
    }

    #[test]
    fn auto_and_nothing_detect() {
        assert_eq!(resolve(Some("auto")), Some(Language::Detect));
        assert_eq!(resolve(Some("AUTO")), Some(Language::Detect));
        assert_eq!(resolve(None), Some(Language::Detect));
        assert_eq!(resolve(Some("  ")), Some(Language::Detect));
    }

    #[test]
    fn an_unknown_code_is_unsupported() {
        assert_eq!(resolve(Some("xx")), None);
        assert_eq!(resolve(Some("he")), None);
        // A name is not a code.
        assert_eq!(resolve(Some("English")), None);
    }
}
