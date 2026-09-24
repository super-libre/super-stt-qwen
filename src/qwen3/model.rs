// SPDX-License-Identifier: GPL-3.0-only
//! The Qwen3-ASR thinker: the audio tower and the text decoder joined, and
//! greedy decoding over the result.
//!
//! A transcription is one prompt — the chat template with one
//! `<|audio_pad|>` per encoded audio frame — prefilled in a single pass, then
//! decoded a token at a time until an end-of-text token. The placeholders are
//! never embedded: the prompt is built as the embeddings of the text before
//! them, the encoded audio, and the embeddings of the text after, which is
//! what the reference's `masked_scatter` computes.
//!
//! Decoding a token is one pass of the decoder on one row, a few hundred small
//! kernels, and launching them one at a time is most of what it costs on a
//! GPU. So the step is captured as a graph and replayed: the token and its
//! position are read from device buffers the host rewrites in place, the
//! key/value caches are preallocated and written at the position the buffer
//! holds, and the step ends by writing its greedy choice back into the token
//! buffer — so the host's only round trip per token is reading that choice to
//! see whether it ended the text. On a device without graph support the same
//! closure simply runs again.
//!
//! Every shape here is bucketed, because `CubeCL` compiles and tunes per
//! shape: the prompt is padded at its end to a power of two for the prefill,
//! and a step attends over a power-of-two window of the caches — one captured
//! step per window, all sharing the same caches. See [`Decoder`].

// `slice_assign` takes one range per dimension, in an array: one for a one-dimensional tensor.
#![allow(clippy::single_range_in_vec_init)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use burn::nn::{Embedding, EmbeddingConfig, Linear};
use burn::prelude::*;
use burn::tensor::{DType, Graph, capture};
use burn_store::{FloatCastAdapter, KeyRemapper, ModuleAdapter, ModuleSnapshot, SafetensorsStore};

use crate::qwen3::Taps;
use crate::qwen3::config::{Config, TextConfig};
use crate::qwen3::encoder::{AudioEncoder, AudioTower};
use crate::qwen3::transformer::{Transformer, TransformerState};

#[derive(Module, Debug)]
struct TextModel {
    embed_tokens: Embedding,
    transformer: Transformer,
}

#[derive(Module, Debug)]
struct Thinker {
    audio_tower: AudioEncoder,
    model: TextModel,
    lm_head: Linear,
}

/// The root of a Qwen3-ASR checkpoint.
#[derive(Module, Debug)]
struct Model {
    thinker: Thinker,
}

impl Model {
    fn init(cfg: &Config, device: &Device) -> Self {
        let text = &cfg.thinker_config.text_config;
        Self {
            thinker: Thinker {
                audio_tower: AudioEncoder::init(&cfg.thinker_config.audio_config, device),
                model: TextModel {
                    embed_tokens: EmbeddingConfig::new(text.vocab_size, text.hidden_size)
                        .init(device),
                    transformer: Transformer::init(text, device),
                },
                lm_head: crate::qwen3::linear_config(text.hidden_size, text.vocab_size, device)
                    .with_bias(false)
                    .init(device),
            },
        }
    }
}

/// Maps the checkpoint's names onto the module tree above: the decoder's
/// layers and final norm sit next to its embedding table in the checkpoint and
/// one level down, inside [`Transformer`], here.
fn remapper() -> Result<KeyRemapper, String> {
    KeyRemapper::from_patterns(vec![(
        r"^thinker\.model\.(layers|norm)\.".to_string(),
        "thinker.model.transformer.$1.".to_string(),
    )])
    .map_err(|err| format!("invalid remapping: {err}"))
}

/// A prompt around the encoded audio: the token ids before the audio and
/// after it. The `<|audio_pad|>` placeholders between them are implied, one per
/// encoded frame.
#[derive(Debug, Clone, Copy)]
pub struct Prompt<'a> {
    /// Up to and including `<|audio_start|>`.
    pub prefix: &'a [u32],
    /// From `<|audio_end|>` to the end of the prompt.
    pub suffix: &'a [u32],
}

/// What the captured decode steps read and write besides the model, shared by
/// all of them.
struct StepState {
    transformer: TransformerState,
    /// The token a step feeds, (1, 1), overwritten by the step with the token
    /// it chooses.
    token: Tensor<2, Int>,
    /// The position of that token, (1).
    pos: Tensor<1, Int>,
}

/// A captured decode step.
type Step = Graph<Tensor<1, Int>, Box<dyn FnMut() -> Tensor<1, Int>>>;

/// The decode steps: one captured per attention window, all over the same
/// caches.
///
/// A step's shape is fixed by the window it attends over, not by the
/// position it writes, so a handful of windows — powers of two from
/// [`MIN_WINDOW`] — serve every position, and the warm-up compiles and tunes
/// each of them before the first request. The caches are sized for the
/// largest window used so far and grow, rarely, by doubling; a grown cache has
/// moved, and every step captured over the old one is captured again, lazily.
struct Decoder {
    state: Rc<RefCell<StepState>>,
    steps: HashMap<usize, Step>,
}

/// The smallest attention window a decode step uses.
const MIN_WINDOW: usize = 256;

/// The window a step at `pos` attends over: the smallest power of two that
/// holds its position, and at least [`MIN_WINDOW`].
fn window_for(pos: usize) -> usize {
    (pos + 1).next_power_of_two().max(MIN_WINDOW)
}

/// The shortest prompt the prefill runs on.
const MIN_PREFILL: usize = 64;

/// The length a prompt of `len` tokens is padded to for the prefill: a power
/// of two, so that the prefill's shapes are a handful the warm-up covers
/// rather than one per request. The padding goes at the end, where causal
/// attention never sees it and the decode steps overwrite or mask it.
fn prefill_len(len: usize) -> usize {
    len.next_power_of_two().max(MIN_PREFILL)
}

/// A loaded Qwen3-ASR checkpoint.
pub struct Qwen3Asr {
    audio: AudioTower,
    text: TextModel,
    lm_head: Linear,
    /// Built on the first transcription and kept for the next, whose prefill
    /// simply overwrites the caches from position 0.
    decoder: Option<Decoder>,
    text_config: TextConfig,
    device: Device,
    dtype: DType,
}

// The graph and the modules have nothing printable; the configuration and the
// placement are what a log wants.
impl std::fmt::Debug for Qwen3Asr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3Asr")
            .field("text_config", &self.text_config)
            .field(
                "decode_capacity",
                &self
                    .decoder
                    .as_ref()
                    .map(|d| d.state.borrow().transformer.capacity()),
            )
            .field("device", &self.device)
            .field("dtype", &self.dtype)
            .finish_non_exhaustive()
    }
}

impl Qwen3Asr {
    /// Loads a checkpoint from its safetensors files — one, or the shards the
    /// 1.7B checkpoint is split into — casting the weights to `dtype`. `read`
    /// counts the checkpoint's bytes as they are read.
    ///
    /// # Errors
    /// If the configuration describes a model this port does not implement,
    /// a file cannot be read, or the files do not fill every parameter.
    pub fn load(
        cfg: &Config,
        weights: &[PathBuf],
        dtype: DType,
        device: &Device,
        read: &Arc<AtomicU64>,
    ) -> Result<Self, String> {
        cfg.validate()?;
        let mut model = Model::init(cfg, device);
        let mut results = Vec::with_capacity(weights.len());
        for file in weights {
            let mut store = SafetensorsStore::from_file(file)
                .with_from_adapter(
                    crate::qwen3::ReadCounter(Arc::clone(read))
                        .chain(crate::qwen3::CheckpointAdapter::for_device(device))
                        .chain(crate::qwen3::HalfCast { target: dtype })
                        .chain(FloatCastAdapter::to(dtype)),
                )
                .remap(remapper()?)
                .allow_partial(true);
            results.push(
                model
                    .load_from(&mut store)
                    .map_err(|err| format!("loading {}: {err}", file.display()))?,
            );
        }
        crate::qwen3::check_shards(&results)?;

        let Model {
            thinker:
                Thinker {
                    audio_tower,
                    model: text,
                    lm_head,
                },
        } = model;
        Ok(Self {
            audio: AudioTower::new(audio_tower, &cfg.thinker_config.audio_config, dtype, device),
            text,
            lm_head,
            decoder: None,
            text_config: cfg.thinker_config.text_config.clone(),
            device: device.clone(),
            dtype,
        })
    }

    /// Encodes a log-mel spectrogram of `frames` frames into the (N, hidden)
    /// embeddings the prompt's placeholders stand for, recording every stage
    /// in `taps` (see [`AudioTower::forward`]).
    pub fn encode_audio(&self, mel: &[f32], frames: usize, taps: &mut Taps) -> Tensor<2> {
        self.audio.forward(mel, frames, &self.device, taps)
    }

    /// See [`AudioTower::full_attention`].
    #[cfg(test)]
    pub(crate) fn set_full_attention(&mut self, full: bool) {
        self.audio.full_attention = full;
    }

    /// How many embeddings [`Self::encode_audio`] makes of `frames` frames.
    #[cfg(test)]
    pub fn audio_len(&self, frames: usize) -> usize {
        self.audio.output_len(frames)
    }

    fn embed(&self, ids: &[u32]) -> Tensor<3> {
        let len = ids.len();
        let ids: Vec<i64> = ids.iter().map(|&id| i64::from(id)).collect();
        let ids = Tensor::<2, Int>::from_data(TensorData::new(ids, [1, len]), &self.device);
        self.text.embed_tokens.forward(ids)
    }

    /// The prompt's embeddings, (1, L, hidden): the text before the audio,
    /// the audio, and the text after it.
    fn splice(&self, prompt: Prompt<'_>, audio: Tensor<2>) -> Tensor<3> {
        let [n, hidden] = audio.dims();
        Tensor::cat(
            vec![
                self.embed(prompt.prefix),
                audio.cast(self.dtype).reshape([1, n, hidden]),
                self.embed(prompt.suffix),
            ],
            1,
        )
    }

    /// Decodes greedily after `prompt` and `audio`, (N, hidden) from
    /// [`Self::encode_audio`], until one of `eos` or `max_new_tokens` tokens.
    ///
    /// `on_token` sees every token as it is chosen, the ending one excluded,
    /// and [`ControlFlow::Break`] ends the decoding after it — how a cancel
    /// stops paying for the rest of a transcription. Returns the tokens chosen,
    /// without the ending one.
    pub fn generate(
        &mut self,
        prompt: Prompt<'_>,
        audio: Tensor<2>,
        max_new_tokens: usize,
        eos: &[u32],
        on_token: impl FnMut(u32) -> ControlFlow<()>,
    ) -> Vec<u32> {
        let embeds = self.splice(prompt, audio);
        self.generate_captured(embeds, max_new_tokens, eos, on_token)
    }

    /// [`Self::generate`] one operation at a time, recording in `taps` the
    /// spliced prompt (`dec.embeds`), the prefill's layers (see
    /// [`Transformer::forward`]) and the logits of every step (`logits.{n}`).
    ///
    /// With `forced`, the decoding is teacher-forced: step `n` is fed
    /// `forced[n - 1]` whatever it chose, for `forced.len()` steps, so that
    /// every step's logits can be compared with a reference's even past a
    /// near-tie that would otherwise send the two down different paths. The
    /// return is then the model's own choice at every step, the ending token
    /// included.
    #[cfg(test)]
    pub fn generate_tapped(
        &self,
        prompt: Prompt<'_>,
        audio: Tensor<2>,
        max_new_tokens: usize,
        eos: &[u32],
        forced: Option<&[u32]>,
        taps: &mut Taps,
    ) -> Vec<u32> {
        let embeds = self.splice(prompt, audio);
        taps.record("dec.embeds", &embeds);
        let prefix_len = embeds.dims()[1];
        let steps = forced.map_or(max_new_tokens + 1, <[u32]>::len);
        let mut state = TransformerState::new(
            &self.text_config,
            prefix_len + steps,
            self.dtype,
            &self.device,
        );
        let hidden = self.text.transformer.forward(embeds, 0, &mut state, taps);
        let mut logits = self.lm_head.forward(hidden.narrow(1, prefix_len - 1, 1));
        let mut chosen = Vec::new();
        for n in 0..steps {
            taps.record_with(|| format!("logits.{n}"), &logits);
            let token = read_token(&logits.clone().argmax(2).reshape([1]));
            chosen.push(token);
            let feed = match forced {
                Some(forced) => forced[n],
                None if eos.contains(&token) => {
                    chosen.pop();
                    break;
                }
                None => token,
            };
            if n + 1 == steps {
                break;
            }
            let hidden = self.text.transformer.forward(
                self.embed(&[feed]),
                prefix_len + n,
                &mut state,
                &mut Taps::off(),
            );
            logits = self.lm_head.forward(hidden);
        }
        chosen
    }

    fn generate_captured(
        &mut self,
        embeds: Tensor<3>,
        max_new_tokens: usize,
        eos: &[u32],
        mut on_token: impl FnMut(u32) -> ControlFlow<()>,
    ) -> Vec<u32> {
        let [_, len, hidden] = embeds.dims();
        let padded = prefill_len(len);
        let embeds = if padded > len {
            let padding = Tensor::zeros([1, padded - len, hidden], (&self.device, self.dtype));
            Tensor::cat(vec![embeds, padding], 1)
        } else {
            embeds
        };
        // The prefill writes every padded position, and the first step
        // attends over its window.
        self.ensure_capacity(padded.max(window_for(len)));

        let first = {
            let decoder = self.decoder.as_ref().expect("just ensured");
            let state = &mut *decoder.state.borrow_mut();
            let hidden =
                self.text
                    .transformer
                    .forward(embeds, 0, &mut state.transformer, &mut Taps::off());
            let last = hidden.narrow(1, len - 1, 1);
            let token = self.lm_head.forward(last).argmax(2).reshape([1, 1]);
            state
                .token
                .inplace(|buffer| buffer.slice_assign([0..1, 0..1], token));
            read_token(&state.token.clone().reshape([1]))
        };

        let mut tokens = Vec::new();
        let mut token = first;
        while !eos.contains(&token) {
            tokens.push(token);
            if on_token(token).is_break() || tokens.len() >= max_new_tokens {
                break;
            }
            // The token just chosen is fed at the position after the prompt
            // and every token before it.
            token = self.step(len + tokens.len() - 1, token);
        }
        tokens
    }

    /// Makes sure the caches hold at least `needed` positions, growing them —
    /// and dropping every captured step, which replays against the old ones —
    /// when they do not. Returns whether they grew.
    fn ensure_capacity(&mut self, needed: usize) -> bool {
        let capacity = needed.next_power_of_two();
        match &mut self.decoder {
            Some(decoder) if decoder.state.borrow().transformer.capacity() >= needed => false,
            Some(decoder) => {
                // The steps replay against the old caches, so they go before
                // the caches move.
                decoder.steps.clear();
                decoder.state.borrow_mut().transformer.grow(capacity);
                true
            }
            None => {
                self.decoder = Some(Decoder {
                    state: Rc::new(RefCell::new(StepState {
                        transformer: TransformerState::new(
                            &self.text_config,
                            capacity,
                            self.dtype,
                            &self.device,
                        ),
                        token: Tensor::zeros([1, 1], &self.device),
                        pos: Tensor::zeros([1], &self.device),
                    })),
                    steps: HashMap::new(),
                });
                true
            }
        }
    }

    /// Captures the decode step attending over `window` positions.
    fn capture_step(&self, state: &Rc<RefCell<StepState>>, window: usize) -> Step {
        // The capture's warm-up runs write the cache at the position the
        // buffer holds: the last one, which lies outside every smaller window,
        // stays masked in its own until a transcription reaches it, and is
        // overwritten before it is read.
        {
            let state = &mut *state.borrow_mut();
            let last = state.transformer.capacity() - 1;
            write_position(&mut state.pos, last, &self.device);
        }
        let text = self.text.clone();
        let lm_head = self.lm_head.clone();
        let step_state = Rc::clone(state);
        let mut run: Box<dyn FnMut() -> Tensor<1, Int>> = Box::new(move || {
            let state = &mut *step_state.borrow_mut();
            let xs = text.embed_tokens.forward(state.token.clone());
            let hidden =
                text.transformer
                    .forward_at(xs, &state.pos, &mut state.transformer, window);
            let next = lm_head.forward(hidden).argmax(2).reshape([1, 1]);
            state
                .token
                .inplace(|buffer| buffer.slice_assign([0..1, 0..1], next.clone()));
            next.reshape([1])
        });
        // Compiled and autotuned outside the capture: tuning allocates the
        // buffers of every candidate it benchmarks, which a capture would keep.
        let _ = run().into_data();
        capture(&self.device, run)
    }

    /// Feeds `token`, which the step buffer holds, at `pos` and returns the
    /// next one.
    fn step(&mut self, pos: usize, token: u32) -> u32 {
        let window = window_for(pos);
        let mut disturbed = self.ensure_capacity(window);
        let decoder = self.decoder.as_ref().expect("just ensured");
        if !decoder.steps.contains_key(&window) {
            let state = Rc::clone(&decoder.state);
            let step = self.capture_step(&state, window);
            let decoder = self.decoder.as_mut().expect("just ensured");
            decoder.steps.insert(window, step);
            disturbed = true;
        }
        let decoder = self.decoder.as_mut().expect("just ensured");
        if disturbed {
            // A capture's warm-up runs overwrote the token buffer with tokens
            // of their own. The value comes from the host, where the caller
            // already has it.
            let token = Tensor::<2, Int>::from_data([[i64::from(token)]], &self.device);
            decoder
                .state
                .borrow_mut()
                .token
                .inplace(|buffer| buffer.slice_assign([0..1, 0..1], token));
        }
        write_position(&mut decoder.state.borrow_mut().pos, pos, &self.device);
        let step = decoder.steps.get_mut(&window).expect("just captured");
        // Safety: every buffer the step reads or writes is kept alive by the
        // shared state or by the modules its closure owns, the writes above
        // and the read below go through the same device client and stream as
        // the replay, and nothing else touches those buffers meanwhile.
        let next = unsafe { step.replay() }.clone();
        read_token(&next)
    }
}

/// Writes `pos` into the one-element `buffer` in place: the buffer is what the
/// captured step reads its position from, so it must stay where it is.
fn write_position(buffer: &mut Tensor<1, Int>, pos: usize, device: &Device) {
    let pos = i64::try_from(pos).expect("a position fits in an i64");
    let at = Tensor::<1, Int>::from_data([pos], device);
    buffer.inplace(|buffer| buffer.slice_assign([0..1], at));
}

/// The one round trip of a decode step.
fn read_token(token: &Tensor<1, Int>) -> u32 {
    let value = token
        .clone()
        .into_data()
        .iter::<i64>()
        .next()
        .expect("a token tensor holds one value");
    u32::try_from(value).expect("a token id is a vocabulary index")
}
