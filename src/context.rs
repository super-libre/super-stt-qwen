// SPDX-License-Identifier: GPL-3.0-only
//! The user's dictation context, as the text Qwen3-ASR is prompted with.
//!
//! The manifest declares `[capabilities] context = true`, so the daemon hands
//! this backend the dictation context in force on every `/v1` request, as two
//! JSON headers: `x-voice-prompt`, a string saying what the user is dictating,
//! and `x-voice-vocabulary`, an array of terms spelled the way the user wants
//! them. Each is left out when its half is empty. A Super STT daemon sends the
//! same headers as `x-stt-prompt` and `x-stt-vocabulary`; a Super Voice daemon
//! sends both spellings.
//!
//! Qwen3-ASR was trained to be steered by such text. The reference's
//! `transcribe(context=...)` puts it, as it is, in the system turn of the chat
//! template — the turn this backend otherwise leaves empty (see
//! [`crate::prompt`]) — and the model card describes it as anything from a
//! list of keywords to a paragraph. So the two halves are folded into one text
//! for that turn: the terms one to a line, then the prompt.
//!
//! The terms come first because they are the half an ASR model can use most
//! directly, as spellings to prefer, and [`crate::prompt::fit_context`] keeps
//! the start of a context too long for the budget.

use axum::http::HeaderMap;

/// The most tokens of context a transcription is prompted with.
///
/// The context is prefilled on every request, and the daemon sends a request
/// for each preview of a dictation in progress, so every token here costs as
/// much as a token of audio, about every second. 512 tokens is about forty
/// seconds of audio's worth: invisible on a GPU, and a bound on what a long
/// context adds on a CPU. It holds a paragraph of prompt and a hundred or more
/// terms; the daemon's own limits allow about four times that.
pub const MAX_TOKENS: usize = 512;

/// The prompt's header, then its Super STT spelling.
const PROMPT: [&str; 2] = ["x-voice-prompt", "x-stt-prompt"];
/// The vocabulary's header, then its Super STT spelling.
const VOCABULARY: [&str; 2] = ["x-voice-vocabulary", "x-stt-vocabulary"];

/// The context the request carries, folded into the text for the system turn
/// (see [`fold`]). Empty when the request carries none.
///
/// A header that does not decode is logged and left out rather than failing
/// the request: a malformed context is the daemon's bug, and the user's audio
/// should still be transcribed.
#[must_use]
pub fn from_headers(headers: &HeaderMap) -> String {
    let prompt: Option<String> = decode(headers, PROMPT);
    let terms: Vec<String> = decode(headers, VOCABULARY).unwrap_or_default();
    fold(prompt.as_deref(), &terms)
}

/// The first of `names` the request carries, decoded from its JSON.
fn decode<T: serde::de::DeserializeOwned>(headers: &HeaderMap, names: [&str; 2]) -> Option<T> {
    let (name, value) = names
        .iter()
        .find_map(|&name| headers.get(name).map(|value| (name, value)))?;
    match value
        .to_str()
        .map_err(|e| e.to_string())
        .and_then(|raw| serde_json::from_str(raw).map_err(|e| e.to_string()))
    {
        Ok(decoded) => Some(decoded),
        Err(e) => {
            log::warn!("ignoring the {name} header, which does not decode: {e}");
            None
        }
    }
}

/// The terms, one to a line, then a blank line and the prompt. Blank terms and
/// a blank prompt are left out, so no context at all is the empty string.
///
/// A line apiece keeps a term with a comma in it, `Menjivar, Jorge`, one term.
#[must_use]
pub fn fold(prompt: Option<&str>, terms: &[String]) -> String {
    let terms: Vec<&str> = terms
        .iter()
        .map(|t| t.trim())
        .filter(|t| !t.is_empty())
        .collect();
    let prompt = prompt.map(str::trim).filter(|p| !p.is_empty());
    let mut parts = Vec::with_capacity(2);
    if !terms.is_empty() {
        parts.push(terms.join("\n"));
    }
    if let Some(prompt) = prompt {
        parts.push(prompt.to_string());
    }
    parts.join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for &(name, value) in pairs {
            map.insert(name, value.parse().unwrap());
        }
        map
    }

    #[test]
    fn no_context_is_empty() {
        assert_eq!(from_headers(&HeaderMap::new()), "");
        assert_eq!(fold(Some("  "), &[" ".into(), String::new()]), "");
    }

    #[test]
    fn the_terms_lead_one_to_a_line_then_the_prompt() {
        let map = headers(&[
            (
                "x-voice-prompt",
                r#""An email to my team about the release.""#,
            ),
            (
                "x-voice-vocabulary",
                r#"["Super Voice", "Menjivar, Jorge", " ", "Qwen"]"#,
            ),
        ]);
        assert_eq!(
            from_headers(&map),
            "Super Voice\nMenjivar, Jorge\nQwen\n\nAn email to my team about the release."
        );
    }

    #[test]
    fn either_half_stands_alone() {
        // The daemon escapes a header to ASCII; the JSON decode undoes it.
        let prompt = headers(&[("x-voice-prompt", r#""Notas de la reuni\u00f3n""#)]);
        assert_eq!(from_headers(&prompt), "Notas de la reunión");
        let vocabulary = headers(&[("x-voice-vocabulary", r#"["CubeCL", "Burn"]"#)]);
        assert_eq!(from_headers(&vocabulary), "CubeCL\nBurn");
    }

    /// A Super STT daemon sends only the old names; a Super Voice one sends
    /// both, with the same values, and the new name is the one read.
    #[test]
    fn the_super_stt_names_are_read_when_the_new_ones_are_absent() {
        let old = headers(&[
            ("x-stt-prompt", r#""Dictating code""#),
            ("x-stt-vocabulary", r#"["axum"]"#),
        ]);
        assert_eq!(from_headers(&old), "axum\n\nDictating code");
        let both = headers(&[
            ("x-voice-vocabulary", r#"["new"]"#),
            ("x-stt-vocabulary", r#"["old"]"#),
        ]);
        assert_eq!(from_headers(&both), "new");
    }

    #[test]
    fn a_header_that_does_not_decode_is_left_out() {
        let map = headers(&[
            ("x-voice-prompt", "not json"),
            ("x-voice-vocabulary", r#"["kept"]"#),
        ]);
        assert_eq!(from_headers(&map), "kept");
        let map = headers(&[
            ("x-voice-prompt", r#""kept""#),
            ("x-voice-vocabulary", r#""a string, not a list""#),
        ]);
        assert_eq!(from_headers(&map), "kept");
        // Unescaped bytes past ASCII are not what the daemon sends.
        let mut map = HeaderMap::new();
        map.insert(
            "x-voice-prompt",
            axum::http::HeaderValue::from_bytes("\"reunión\"".as_bytes()).unwrap(),
        );
        assert_eq!(from_headers(&map), "");
    }
}
