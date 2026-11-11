// SPDX-License-Identifier: GPL-3.0-only
//! Text in and text out: the tokenizer, the chat template a transcription is
//! asked in, and reading the model's answer back.
//!
//! # The tokenizer
//!
//! The checkpoints ship the Qwen2 byte-level BPE as `vocab.json` and
//! `merges.txt`, with the added tokens in `tokenizer_config.json`, and no
//! assembled `tokenizer.json`. [`load_tokenizer`] assembles one — the same
//! normalizer, pre-tokenizer, model and decoder `transformers`' Qwen2 converter
//! builds — so the manifest fetches the files the checkpoint actually has.
//!
//! One reference detail is deliberately not copied. Loaded from a local
//! directory with `fix_mistral_regex=True`, as the Python backend did,
//! `transformers` 4.57 swaps the Qwen2 pre-tokenizer pattern for one written
//! for Mistral tokenizers, because the checkpoint's `transformers_version`
//! falls in the range its detection treats as suspect. The model was trained
//! with the Qwen2 pattern, so that is what is built here. Every string this
//! backend encodes — the template and `language X` — tokenizes the same under
//! both, and the test at the bottom pins those ids.
//!
//! # The template
//!
//! ```text
//! <|im_start|>system\n{context}<|im_end|>\n
//! <|im_start|>user\n<|audio_start|>{<|audio_pad|> × N}<|audio_end|><|im_end|>\n
//! <|im_start|>assistant\n[language {Name}<asr_text>]
//! ```
//!
//! The context is the user's dictation context, folded into one text by
//! [`crate::context`] and cut to its budget by [`fit_context`], and empty when
//! they have none: the reference's default, and all the Python backend ever
//! sent. The bracketed tail is added when the request forces a language; the
//! model then answers with the text alone. Without it, the model begins its
//! answer with `language {Name}<asr_text>` itself.

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};
use tokenizers::Tokenizer;

use crate::lang::Language;

/// The Qwen2 pre-tokenizer pattern, as `transformers`' `Qwen2Converter` writes
/// it.
const QWEN2_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The tag that separates the model's detected language from its text.
pub const ASR_TEXT_TAG: &str = "<asr_text>";

/// Assemble the tokenizer from the checkpoint's `vocab.json`, `merges.txt`
/// and `tokenizer_config.json`.
///
/// # Errors
/// If a file is missing or not what a Qwen2 tokenizer ships.
pub fn load_tokenizer(model_dir: &Path) -> Result<Tokenizer> {
    let read = |name: &str| {
        let path = model_dir.join(name);
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))
    };
    let vocab: Value = serde_json::from_str(&read("vocab.json")?).context("parsing vocab.json")?;
    // One merge per line after the `#version` header, as two space-separated
    // symbols.
    let merges: Vec<Value> = read("merges.txt")?
        .lines()
        .filter(|l| !l.starts_with("#version") && !l.trim().is_empty())
        .map(|l| {
            let (a, b) = l
                .split_once(' ')
                .ok_or_else(|| anyhow!("merges.txt has a line that is not a pair: {l:?}"))?;
            Ok(json!([a, b]))
        })
        .collect::<Result<_>>()?;
    let config: Value = serde_json::from_str(&read("tokenizer_config.json")?)
        .context("parsing tokenizer_config.json")?;
    let decoder = config
        .get("added_tokens_decoder")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow!("tokenizer_config.json declares no added_tokens_decoder"))?;
    let mut added: Vec<(u32, Value)> = decoder
        .iter()
        .map(|(id, token)| {
            let id: u32 = id
                .parse()
                .with_context(|| format!("added token id {id:?}"))?;
            let field = |key: &str| token.get(key).cloned().unwrap_or(json!(false));
            Ok((
                id,
                json!({
                    "id": id,
                    "content": token.get("content").cloned().unwrap_or_default(),
                    "single_word": field("single_word"),
                    "lstrip": field("lstrip"),
                    "rstrip": field("rstrip"),
                    "normalized": field("normalized"),
                    "special": field("special"),
                }),
            ))
        })
        .collect::<Result<_>>()?;
    added.sort_by_key(|(id, _)| *id);

    let assembled = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added.into_iter().map(|(_, t)| t).collect::<Vec<_>>(),
        "normalizer": { "type": "NFC" },
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": { "Regex": QWEN2_PATTERN },
                    "behavior": "Isolated",
                    "invert": false
                },
                { "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false }
            ]
        },
        "post_processor": { "type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": true },
        "decoder": { "type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true },
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": vocab,
            "merges": merges
        }
    });
    Tokenizer::from_bytes(serde_json::to_vec(&assembled)?)
        .map_err(|e| anyhow!("assembling the tokenizer: {e}"))
}

/// The template's text either side of the audio placeholders, with `context`
/// as the system turn.
///
/// The whole of it is tokenized as one string, as the reference tokenizes the
/// chat template's output, so a context is split into tokens the way the
/// model saw contexts split in training.
#[must_use]
pub fn template(context: &str, language: Language) -> (String, String) {
    let before =
        format!("<|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n<|audio_start|>");
    let mut after = "<|audio_end|><|im_end|>\n<|im_start|>assistant\n".to_string();
    if let Language::Forced(name) = language {
        after.push_str("language ");
        after.push_str(name);
        after.push_str(ASR_TEXT_TAG);
    }
    (before, after)
}

/// `context`, trimmed, and cut to about `max_tokens` tokens if it runs longer:
/// the start of it is kept, which [`crate::context::fold`] fills with the
/// terms, and the cut goes back to the last whitespace before the limit, so
/// that a term or a word is kept whole or not at all.
///
/// About, because the text kept is tokenized again inside the template; a cut
/// at a token's start re-tokenizes to the same tokens or near enough.
///
/// # Errors
/// If the tokenizer fails on the context.
pub fn fit_context<'a>(
    tokenizer: &Tokenizer,
    context: &'a str,
    max_tokens: usize,
) -> Result<&'a str> {
    let context = context.trim();
    let encoding = tokenizer
        .encode(context, false)
        .map_err(|e| anyhow!("tokenizing the dictation context: {e}"))?;
    let offsets = encoding.get_offsets();
    let Some(&(first_dropped, _)) = offsets.get(max_tokens) else {
        return Ok(context);
    };
    let kept = cut_before(context, first_dropped);
    log::info!(
        "the dictation context is {} tokens, over the {max_tokens} a transcription is prompted \
         with; keeping its first {} of {} characters",
        offsets.len(),
        kept.chars().count(),
        context.chars().count(),
    );
    Ok(kept)
}

/// `text` up to byte `end`, cut back to the last whitespace before it when
/// there is one, and trimmed. A text with no whitespace before `end`, as
/// Chinese or Japanese is written, is cut at `end` itself.
fn cut_before(text: &str, end: usize) -> &str {
    let mut end = end.min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    let head = &text[..end];
    let at_space = text[end..].starts_with(char::is_whitespace);
    let head = match head.rfind(char::is_whitespace) {
        Some(space) if !at_space => &head[..space],
        _ => head,
    };
    head.trim_end()
}

/// The text of a finished answer, the reference's `parse_asr_output`.
///
/// Repetition is cut back first (see [`fix_repetitions`]). A forced language
/// means the answer is text alone. Otherwise the text is what follows
/// [`ASR_TEXT_TAG`] — or the whole answer, if the model left the tag out — and
/// an answer detecting `language None` is silence, whose text is usually
/// empty.
#[must_use]
pub fn parse_answer(raw: &str, language: Language) -> String {
    let s = raw.trim();
    if s.is_empty() {
        return String::new();
    }
    let s = fix_repetitions(s, 20);
    if let Language::Forced(_) = language {
        return s;
    }
    match s.split_once(ASR_TEXT_TAG) {
        Some((_, text)) => text.trim().to_string(),
        None => s.trim().to_string(),
    }
}

/// What of an answer still being written can be shown: the text so far, once
/// the model is past its language tag, less a trailing partial character.
///
/// Byte-level BPE can split a character across tokens, and a character cut
/// in half decodes as U+FFFD until its other half arrives.
#[must_use]
pub fn preview(raw: &str, language: Language) -> Option<String> {
    let text = match language {
        Language::Forced(_) => raw,
        Language::Detect => raw.split_once(ASR_TEXT_TAG)?.1,
    };
    let text = text.trim_end_matches('\u{FFFD}').trim();
    Some(text.to_string())
}

/// The reference's `detect_and_fix_repetitions`: a run of one character longer
/// than `threshold` becomes that character once, and a pattern of up to 20
/// characters repeated `threshold` times or more becomes the pattern once.
///
/// Greedy decoding can fall into a loop and repeat a phrase until the token
/// limit; this is what keeps such an answer from reaching the user as
/// hundreds of copies. Ported operation for operation, on characters as
/// Python indexes a string.
#[must_use]
pub fn fix_repetitions(text: &str, threshold: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    let chars = fix_char_repeats(&chars, threshold);
    fix_pattern_repeats(&chars, threshold, 20)
        .into_iter()
        .collect()
}

fn fix_char_repeats(s: &[char], threshold: usize) -> Vec<char> {
    let mut out = Vec::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        let mut count = 1;
        while i + count < s.len() && s[i + count] == s[i] {
            count += 1;
        }
        if count > threshold {
            out.push(s[i]);
        } else {
            out.extend_from_slice(&s[i..i + count]);
        }
        i += count;
    }
    out
}

fn fix_pattern_repeats(s: &[char], threshold: usize, max_len: usize) -> Vec<char> {
    let n = s.len();
    let min_repeat_chars = threshold * 2;
    if n < min_repeat_chars {
        return s.to_vec();
    }
    let mut out = Vec::with_capacity(n);
    let mut i = 0;
    while i <= n - min_repeat_chars {
        for k in 1..=max_len {
            if i + k * threshold > n {
                break;
            }
            let pattern = &s[i..i + k];
            let repeated = (1..threshold).all(|rep| {
                let start = i + rep * k;
                &s[start..(start + k).min(n)] == pattern
            });
            if repeated {
                let mut end = i + threshold * k;
                while end + k <= n && &s[end..end + k] == pattern {
                    end += k;
                }
                out.extend_from_slice(pattern);
                out.extend(fix_pattern_repeats(&s[end..], threshold, max_len));
                return out;
            }
        }
        out.push(s[i]);
        i += 1;
    }
    out.extend_from_slice(&s[i..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The directory holding the tokenizer files, when present: the checkpoint
    /// downloaded for the parity check, or what `just fetch-tokenizer` puts in
    /// `target/test-backend`.
    fn tokenizer_dir() -> Option<std::path::PathBuf> {
        let candidates = [
            std::env::var_os("QWEN3_ASR_MODEL_DIR").map(std::path::PathBuf::from),
            Some(
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("target/test-backend/models/qwen3-asr-0.6b"),
            ),
        ];
        candidates
            .into_iter()
            .flatten()
            .find(|dir| dir.join("vocab.json").is_file() && dir.join("merges.txt").is_file())
    }

    /// Ids from the reference tokenizer, `Qwen2TokenizerFast` loaded from the
    /// 0.6B checkpoint by `transformers` 4.57.6 with the Qwen2 pattern, for
    /// the strings this backend encodes and for a few where the Qwen2 and the
    /// Mistral patterns would part ways.
    #[test]
    fn tokenization_matches_the_reference() {
        let Some(dir) = tokenizer_dir() else {
            eprintln!("skipping: no tokenizer files; run `just fetch-tokenizer`");
            return;
        };
        let tokenizer = load_tokenizer(&dir).unwrap();
        let encode = |s: &str| tokenizer.encode(s, false).unwrap().get_ids().to_vec();

        let (before, after) = template("", Language::Detect);
        assert_eq!(
            encode(&before),
            [151_644, 8948, 198, 151_645, 198, 151_644, 872, 198, 151_669]
        );
        assert_eq!(encode(&after), [151_670, 151_645, 198, 151_644, 77091, 198]);
        let (_, forced) = template("", Language::Forced("English"));
        assert_eq!(
            encode(&forced),
            [
                151_670, 151_645, 198, 151_644, 77091, 198, 11528, 6364, 151_704
            ]
        );
        let (_, forced) = template("", Language::Forced("Cantonese"));
        assert_eq!(
            encode(&forced),
            [
                151_670, 151_645, 198, 151_644, 77091, 198, 11528, 72366, 2367, 151_704
            ]
        );
        assert_eq!(
            encode("They'll say it's GPT-4o's JSONParser, isn't it?"),
            [
                6865, 3278, 1977, 432, 594, 479, 2828, 12, 19, 78, 594, 4718, 6570, 11, 4436, 944,
                432, 30
            ]
        );
        assert_eq!(
            encode("北京欢迎你 ¿Qué tal? 1234"),
            [
                68990, 100_437, 56568, 28286, 65706, 8210, 30, 220, 16, 17, 18, 19
            ]
        );

        // Decoding skips the special tokens but keeps `<asr_text>`, which the
        // checkpoint does not mark special.
        let decoded = tokenizer
            .decode(
                &[
                    11528, 6364, 151_704, 9707, 11, 94305, 245, 46553, 0, 151_645,
                ],
                true,
            )
            .unwrap();
        assert_eq!(decoded, "language English<asr_text>Hello, 北京!");
    }

    /// A context sits between `system\n` and `<|im_end|>` and tokenizes as
    /// it would alone: the newline before it is a token of its own, and the
    /// special token after it is split out first.
    #[test]
    fn a_context_fills_the_system_turn() {
        let Some(dir) = tokenizer_dir() else {
            eprintln!("skipping: no tokenizer files; run `just fetch-tokenizer`");
            return;
        };
        let tokenizer = load_tokenizer(&dir).unwrap();
        let encode = |s: &str| tokenizer.encode(s, false).unwrap().get_ids().to_vec();

        let context = "Super Voice\nMenjivar, Jorge\n\nAn email about the release.";
        let (before, _) = template(context, Language::Detect);
        assert!(before.starts_with("<|im_start|>system\nSuper Voice\n"));
        let (empty, _) = template("", Language::Detect);
        let empty = encode(&empty);
        let expected: Vec<u32> = [&empty[..3], &encode(context)[..], &empty[3..]].concat();
        assert_eq!(encode(&before), expected);
    }

    #[test]
    fn a_long_context_keeps_its_start_in_whole_words() {
        let Some(dir) = tokenizer_dir() else {
            eprintln!("skipping: no tokenizer files; run `just fetch-tokenizer`");
            return;
        };
        let tokenizer = load_tokenizer(&dir).unwrap();
        let count = |s: &str| tokenizer.encode(s, false).unwrap().get_ids().len();

        let short = "  Super Voice\nQwen  ";
        assert_eq!(fit_context(&tokenizer, short, 512).unwrap(), short.trim());

        let terms: Vec<String> = (0..400).map(|i| format!("Term{i}")).collect();
        let long = terms.join("\n");
        let kept = fit_context(&tokenizer, &long, 512).unwrap();
        assert!(long.starts_with(kept));
        assert!(count(kept) <= 512, "{}", count(kept));
        // Whole terms only: the last line kept is one of them.
        let last = kept.lines().last().unwrap();
        assert!(terms.iter().any(|t| t == last), "{last:?}");
        assert!(count(kept) > 400, "cut back too far: {}", count(kept));
    }

    #[test]
    fn a_cut_goes_back_to_whitespace() {
        // Mid-word: back to the space before it.
        assert_eq!(cut_before("alpha beta gamma", 8), "alpha");
        // At a space: the word before it is whole.
        assert_eq!(cut_before("alpha beta gamma", 10), "alpha beta");
        assert_eq!(cut_before("alpha\nbeta", 7), "alpha");
        // No whitespace to go back to: cut where it falls, on a character.
        assert_eq!(cut_before("北京欢迎你", 7), "北京");
        assert_eq!(cut_before("北京欢迎你", 6), "北京");
        assert_eq!(cut_before("short", 99), "short");
    }

    #[test]
    fn a_detected_answer_is_the_text_after_its_tag() {
        let raw = "language English<asr_text>And so, my fellow Americans.";
        assert_eq!(
            parse_answer(raw, Language::Detect),
            "And so, my fellow Americans."
        );
        assert_eq!(parse_answer("  just text ", Language::Detect), "just text");
        assert_eq!(
            parse_answer("language None<asr_text>", Language::Detect),
            ""
        );
        assert_eq!(parse_answer("", Language::Detect), "");
    }

    #[test]
    fn a_forced_answer_is_all_text() {
        let raw = " Hola, ¿qué tal? ";
        assert_eq!(
            parse_answer(raw, Language::Forced("Spanish")),
            "Hola, ¿qué tal?"
        );
    }

    #[test]
    fn previews_wait_for_the_tag_and_drop_half_characters() {
        assert_eq!(preview("language Eng", Language::Detect), None);
        assert_eq!(
            preview("language English<asr_text>Hel", Language::Detect).as_deref(),
            Some("Hel")
        );
        assert_eq!(
            preview("北\u{FFFD}", Language::Forced("Chinese")).as_deref(),
            Some("北")
        );
    }

    /// Inputs run through the reference's `detect_and_fix_repetitions`, and
    /// what it returned.
    #[test]
    fn repetitions_are_cut_back_as_the_reference_cuts_them() {
        assert_eq!(fix_repetitions(&"ab".repeat(30), 20), "ab");
        let almost = format!("{}zzz", "xy".repeat(19));
        assert_eq!(fix_repetitions(&almost, 20), almost);
        // A run of 21 of one character is one character; 20 is left alone.
        let long = format!("a{}b", "x".repeat(21));
        assert_eq!(fix_repetitions(&long, 20), "axb");
        let short = format!("a{}b", "x".repeat(20));
        assert_eq!(fix_repetitions(&short, 20), short);
        // A phrase looping 25 times is the phrase once, with what follows.
        let looped = format!("start {}end", "la ".repeat(25));
        assert_eq!(fix_repetitions(&looped, 20), "start la end");
        // Nineteen repetitions are not a loop.
        let nineteen = format!("start {}end", "la ".repeat(19));
        assert_eq!(fix_repetitions(&nineteen, 20), nineteen);
        assert_eq!(fix_repetitions("short", 20), "short");
    }
}
