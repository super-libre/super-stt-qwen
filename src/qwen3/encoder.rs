// SPDX-License-Identifier: GPL-3.0-only
//! The Qwen3-ASR audio tower: log-mel frames in, one decoder-sized embedding
//! per 80 ms of audio out.
//!
//! The spectrogram is cut into one-second chunks of 100 frames, and each chunk
//! goes through three 3x3, stride-2 convolutions on its own — 100 frames become
//! 13 — with the tail chunk zero-padded to a full one first, as the reference
//! pads it. Sinusoidal positions are added per chunk, so they restart at every
//! second. The chunks' outputs are then laid end to end, the tail's padding
//! dropped, and the transformer layers attend within windows of eight chunks:
//! every frame sees the frames of its own eight seconds, and nothing else.
//!
//! That locality is what keeps a long recording cheap — the encoder's cost
//! grows linearly with the audio — and it is also what lets the windows run as
//! one batch: every full window is the same shape, and only the last one,
//! shorter, runs on its own.

use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, PaddingConfig2d};
use burn::prelude::*;
use burn::tensor::DType;
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;

use crate::qwen3::Taps;
use crate::qwen3::config::{Activation, AudioEncoderConfig, conv_out_len};

/// Multi-head self-attention over one window, with biases on every
/// projection. No mask: a frame attends to its whole window, both ways.
#[derive(Module, Debug)]
struct EncoderAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl EncoderAttention {
    fn init(cfg: &AudioEncoderConfig, device: &Device) -> Self {
        let linear = || crate::qwen3::linear_config(cfg.d_model, cfg.d_model, device).init(device);
        Self {
            q_proj: linear(),
            k_proj: linear(),
            v_proj: linear(),
            out_proj: linear(),
            num_heads: cfg.encoder_attention_heads,
            head_dim: cfg.d_model / cfg.encoder_attention_heads,
        }
    }

    /// `xs` is (windows, frames, `d_model`); `mask` (windows, heads, frames,
    /// frames) is `true` where a frame must not attend.
    #[allow(clippy::many_single_char_names)]
    fn forward(&self, xs: Tensor<3>, mask: &Tensor<4, Bool>) -> Tensor<3> {
        let [w, l, d] = xs.dims();
        let split = |xs: Tensor<3>| {
            xs.reshape([w, l, self.num_heads, self.head_dim])
                .swap_dims(1, 2)
        };
        let q = split(self.q_proj.forward(xs.clone()));
        let k = split(self.k_proj.forward(xs.clone()));
        let v = split(self.v_proj.forward(xs));
        let options = AttentionModuleOptions {
            scale: None,
            softcap: None,
            is_causal: false,
        };
        let out = attention(q, k, v, Some(mask.clone()), None, options)
            .swap_dims(1, 2)
            .reshape([w, l, d]);
        self.out_proj.forward(out)
    }
}

/// A pre-norm transformer layer with a GELU MLP, Whisper's shape.
#[derive(Module, Debug)]
struct EncoderLayer {
    self_attn: EncoderAttention,
    self_attn_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
    #[module(skip)]
    activation: Activation,
}

impl EncoderLayer {
    fn init(cfg: &AudioEncoderConfig, device: &Device) -> Self {
        // `nn.LayerNorm`'s default epsilon, which is also Burn's.
        let norm = || LayerNormConfig::new(cfg.d_model).init(device);
        Self {
            self_attn: EncoderAttention::init(cfg, device),
            self_attn_layer_norm: norm(),
            fc1: crate::qwen3::linear_config(cfg.d_model, cfg.encoder_ffn_dim, device).init(device),
            fc2: crate::qwen3::linear_config(cfg.encoder_ffn_dim, cfg.d_model, device).init(device),
            final_layer_norm: norm(),
            activation: cfg.activation_function,
        }
    }

    fn forward(&self, xs: Tensor<3>, mask: &Tensor<4, Bool>) -> Tensor<3> {
        let hidden = self
            .self_attn
            .forward(self.self_attn_layer_norm.forward(xs.clone()), mask);
        let xs = xs + hidden;
        let hidden = self.final_layer_norm.forward(xs.clone());
        let hidden = self
            .fc2
            .forward(self.activation.forward(self.fc1.forward(hidden)));
        xs + hidden
    }
}

/// The audio tower's parameters, named as the checkpoint names them under
/// `thinker.audio_tower`.
#[derive(Module, Debug)]
pub struct AudioEncoder {
    conv2d1: Conv2d,
    conv2d2: Conv2d,
    conv2d3: Conv2d,
    /// Flattens the convolutions' channels and remaining frequency rows into
    /// `d_model`. No bias.
    conv_out: Linear,
    layers: Vec<EncoderLayer>,
    ln_post: LayerNorm,
    proj1: Linear,
    proj2: Linear,
    #[module(skip)]
    activation: Activation,
}

impl AudioEncoder {
    /// The attention heads of every layer.
    fn heads(&self) -> usize {
        self.layers
            .first()
            .map_or(1, |layer| layer.self_attn.num_heads)
    }

    pub fn init(cfg: &AudioEncoderConfig, device: &Device) -> Self {
        let channels = cfg.downsample_hidden_size;
        let conv = |c_in| {
            Conv2dConfig::new([c_in, channels], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device)
        };
        Self {
            conv2d1: conv(1),
            conv2d2: conv(channels),
            conv2d3: conv(channels),
            conv_out: crate::qwen3::linear_config(
                channels * cfg.downsampled_mel_bins(),
                cfg.d_model,
                device,
            )
            .with_bias(false)
            .init(device),
            layers: (0..cfg.encoder_layers)
                .map(|_| EncoderLayer::init(cfg, device))
                .collect(),
            ln_post: LayerNormConfig::new(cfg.d_model).init(device),
            proj1: crate::qwen3::linear_config(cfg.d_model, cfg.d_model, device).init(device),
            proj2: crate::qwen3::linear_config(cfg.d_model, cfg.output_dim, device).init(device),
            activation: cfg.activation_function,
        }
    }
}

/// How many embeddings the encoder produces for `frames` mel frames: 13 for
/// every whole chunk, and what the three convolutions leave of the tail.
///
/// The reference computes the same from the frame count alone
/// (`_get_feat_extract_output_lengths`), and the prompt carries one
/// `<|audio_pad|>` per embedding, so the two have to agree exactly.
pub fn output_len(cfg: &AudioEncoderConfig, frames: usize) -> usize {
    let chunk = cfg.chunk_frames();
    let per_chunk = conv_out_len(conv_out_len(conv_out_len(chunk)));
    let tail = frames % chunk;
    let tail_len = if tail == 0 {
        0
    } else {
        conv_out_len(conv_out_len(conv_out_len(tail)))
    };
    frames / chunk * per_chunk + tail_len
}

/// The audio tower with the tables it reads besides its parameters.
#[derive(Debug)]
pub struct AudioTower {
    pub encoder: AudioEncoder,
    /// The sinusoidal positions of one chunk's frames, (13, `d_model`). Not a
    /// parameter: the checkpoint does not store it, the reference computes it.
    positions: Tensor<2>,
    cfg: AudioEncoderConfig,
    dtype: DType,
    /// Attend across the whole clip instead of within windows — what the
    /// reference's `sdpa` path does by omission, reproduced for the parity
    /// check to measure against. Never set outside it.
    #[cfg(test)]
    pub(crate) full_attention: bool,
}

impl AudioTower {
    pub fn new(
        encoder: AudioEncoder,
        cfg: &AudioEncoderConfig,
        dtype: DType,
        device: &Device,
    ) -> Self {
        let per_chunk = conv_out_len(conv_out_len(conv_out_len(cfg.chunk_frames())));
        let rows = per_chunk.min(cfg.max_source_positions);
        let positions = Tensor::<2>::from_data(
            TensorData::new(sinusoids(rows, cfg.d_model), [rows, cfg.d_model]),
            device,
        )
        .cast(dtype);
        Self {
            encoder,
            positions,
            cfg: cfg.clone(),
            dtype,
            #[cfg(test)]
            full_attention: false,
        }
    }

    /// Encodes the log-mel spectrogram `mel`, `num_mel_bins` rows of `frames`
    /// values, into (N, `output_dim`) embeddings, N being [`output_len`].
    ///
    /// `taps` records the output of every stage, under the names of the
    /// reference module that produces it: `enc.conv{1,2,3}` before their GELU,
    /// `enc.conv_out` before the positions are added, `enc.layers.{i}`,
    /// `enc.ln_post`, `enc.proj1` before its GELU, and `enc.proj2`.
    ///
    /// # Panics
    /// If `mel` does not hold `num_mel_bins * frames` values or `frames` is 0.
    pub fn forward(
        &self,
        mel: &[f32],
        frames: usize,
        device: &Device,
        taps: &mut Taps,
    ) -> Tensor<2> {
        let cfg = &self.cfg;
        let bins = cfg.num_mel_bins;
        assert_eq!(mel.len(), bins * frames, "a spectrogram of {frames} frames");
        assert!(frames > 0, "an empty spectrogram has nothing to encode");
        let chunk = cfg.chunk_frames();
        let chunks = frames.div_ceil(chunk);
        let len = output_len(cfg, frames);
        let per_chunk = conv_out_len(conv_out_len(conv_out_len(chunk)));
        let window = per_chunk * cfg.chunks_per_window();

        // Every shape below follows the number of windows, and CubeCL
        // compiles and tunes per shape: left to the audio, every utterance
        // length would pay for its own kernels on its first request — up to
        // half a minute, measured on a fresh process. So the chunks are padded
        // to whole windows and the windows to a power of two, a handful of
        // shapes the warm-up covers. Neither changes a real frame: chunks are
        // convolved independently, and the padded frames are masked out of the
        // last real window's attention.
        let windows = len.div_ceil(window).next_power_of_two();
        let padded_chunks = windows * cfg.chunks_per_window();

        // (chunks, 1, bins, chunk): each chunk's frames side by side, zero
        // beyond the audio — laid out on the host, where it is a copy per row.
        let mut padded = vec![0f32; padded_chunks * bins * chunk];
        for c in 0..chunks {
            let start = c * chunk;
            let len = chunk.min(frames - start);
            for m in 0..bins {
                let from = m * frames + start;
                let to = (c * bins + m) * chunk;
                padded[to..to + len].copy_from_slice(&mel[from..from + len]);
            }
        }
        let padded = Tensor::<4>::from_data(
            TensorData::new(padded, [padded_chunks, 1, bins, chunk]),
            device,
        )
        .cast(self.dtype);

        let encoder = &self.encoder;
        let act = self.activation();
        let embedded = self.convolve(&padded, chunks, taps);
        if taps.is_on() {
            taps.record("enc.conv_out", &embedded.clone().narrow(0, 0, chunks));
        }
        let d_model = embedded.dims()[2];
        let embedded = embedded + self.positions.clone().unsqueeze_dim::<3>(0);

        // The tail chunk is the last real one and its valid frames are its
        // first, so the real frames are a prefix: `len` rows, then padding.
        // The windows are consecutive runs of `window` rows.
        #[allow(unused_mut)]
        let (mut count, mut span) = (windows, window);
        #[cfg(test)]
        if self.full_attention {
            (count, span) = (1, windows * window);
        }
        let mut xs = embedded.reshape([count, span, d_model]);
        let mask = key_mask(count, span, len, self.encoder.heads(), device);
        let real = |xs: &Tensor<3>| {
            xs.clone()
                .reshape([count * span, d_model])
                .narrow(0, 0, len)
        };
        for (i, layer) in encoder.layers.iter().enumerate() {
            xs = layer.forward(xs, &mask);
            if taps.is_on() {
                taps.record_with(|| format!("enc.layers.{i}"), &real(&xs));
            }
        }
        let hidden = xs.reshape([count * span, d_model]);

        // Row by row from here, so the padding rides along in the shape the
        // kernels were built for and is dropped at the end.
        let hidden = encoder.ln_post.forward(hidden);
        if taps.is_on() {
            taps.record("enc.ln_post", &hidden.clone().narrow(0, 0, len));
        }
        let hidden = encoder.proj1.forward(hidden);
        if taps.is_on() {
            taps.record("enc.proj1", &hidden.clone().narrow(0, 0, len));
        }
        let hidden = encoder.proj2.forward(act.forward(hidden)).narrow(0, 0, len);
        taps.record("enc.proj2", &hidden);
        hidden
    }

    /// The three convolutions and `conv_out` over `padded`,
    /// (chunks, 1, bins, `chunk_frames`), into (chunks, 13, `d_model`), of
    /// which the first `real` chunks are audio.
    ///
    /// Chunks go through a group at a time, which only bounds memory: the
    /// first convolution's output is 3 MB per second of audio in bf16. The
    /// group is a power of two, so that it divides a padded chunk count and
    /// every group has the same shape.
    #[allow(clippy::many_single_char_names)]
    fn convolve(&self, padded: &Tensor<4>, real: usize, taps: &mut Taps) -> Tensor<3> {
        let encoder = &self.encoder;
        let act = self.activation();
        let chunks = padded.dims()[0];
        let most = self.cfg.conv_chunksize.max(1);
        let group = 1usize << most.ilog2();
        // Only a recording needs the groups' stages whole; production keeps
        // nothing but each group's output.
        let mut stages: [Vec<Tensor<4>>; 3] = Default::default();
        let embedded: Vec<Tensor<3>> = (0..chunks)
            .step_by(group)
            .map(|start| {
                let n = group.min(chunks - start);
                let mut xs = padded.clone().narrow(0, start, n);
                for (stage, conv) in [&encoder.conv2d1, &encoder.conv2d2, &encoder.conv2d3]
                    .into_iter()
                    .enumerate()
                {
                    let out = conv.forward(xs);
                    if taps.is_on() {
                        stages[stage].push(out.clone());
                    }
                    xs = act.forward(out);
                }
                // (n, channels, freq, time) -> (n, time, channels * freq), the
                // order `conv_out` was trained on.
                let [n, c, f, t] = xs.dims();
                let xs = xs.permute([0, 3, 1, 2]).reshape([n, t, c * f]);
                encoder.conv_out.forward(xs)
            })
            .collect();
        if taps.is_on() {
            for (stage, outputs) in stages.into_iter().enumerate() {
                taps.record_with(
                    || format!("enc.conv{}", stage + 1),
                    &Tensor::cat(outputs, 0).narrow(0, 0, real),
                );
            }
        }
        Tensor::cat(embedded, 0)
    }

    /// How many embeddings [`Self::forward`] makes of `frames` frames.
    #[cfg(test)]
    pub fn output_len(&self, frames: usize) -> usize {
        output_len(&self.cfg, frames)
    }

    fn activation(&self) -> Activation {
        self.encoder.activation
    }
}

/// The attention mask of `count` windows of `span` frames, of which the first
/// `len` are audio: `true` on the keys a frame must not attend to, broadcast
/// over `heads` and every query, (count, heads, span, span).
///
/// Only the window the audio ends in is masked. A window wholly of padding is
/// left open — a query with every key masked has a softmax of nothing, which
/// is NaN — and its frames are dropped anyway.
fn key_mask(
    count: usize,
    span: usize,
    len: usize,
    heads: usize,
    device: &Device,
) -> Tensor<4, Bool> {
    let masked: Vec<bool> = (0..count * span)
        .map(|at| at >= len && at / span * span < len)
        .collect();
    Tensor::<4, Bool>::from_data(TensorData::new(masked, [count, 1, 1, span]), device)
        .expand([count, heads, span, span])
}

/// Whisper's sinusoidal position table, `(length, channels)` row-major: the
/// sines of every timescale, then the cosines.
///
/// In f32, as the reference computes it, so that the handful of rows read
/// match to the last bit that survives the cast to bf16.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn sinusoids(length: usize, channels: usize) -> Vec<f32> {
    let half = channels / 2;
    let increment = (10_000f64.ln() / (half as f64 - 1.0)) as f32;
    let inv_timescales: Vec<f32> = (0..half).map(|i| (-increment * i as f32).exp()).collect();
    let mut table = Vec::with_capacity(length * channels);
    for t in 0..length {
        let scaled: Vec<f32> = inv_timescales.iter().map(|inv| t as f32 * inv).collect();
        table.extend(scaled.iter().map(|x| x.sin()));
        table.extend(scaled.iter().map(|x| x.cos()));
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> AudioEncoderConfig {
        serde_json::from_str(
            r#"{"num_mel_bins": 128, "encoder_layers": 1, "encoder_attention_heads": 2,
                "encoder_ffn_dim": 8, "d_model": 4, "activation_function": "gelu",
                "max_source_positions": 1500, "n_window": 50, "n_window_infer": 800,
                "conv_chunksize": 500, "downsample_hidden_size": 2, "output_dim": 6}"#,
        )
        .unwrap()
    }

    /// The reference's `_get_feat_extract_output_lengths`, verbatim, including
    /// Python's flooring division of negative numbers.
    fn reference_output_len(frames: i64) -> i64 {
        let floor_div = |a: i64, b: i64| a.div_euclid(b);
        let leave = frames % 100;
        let feat = floor_div(leave - 1, 2) + 1;
        floor_div(floor_div(feat - 1, 2) + 1 - 1, 2) + 1 + (frames / 100) * 13
    }

    #[test]
    #[allow(clippy::cast_possible_wrap)]
    fn the_embedding_count_matches_the_reference() {
        let cfg = cfg();
        for frames in 1..2_000 {
            assert_eq!(
                output_len(&cfg, frames) as i64,
                reference_output_len(frames as i64),
                "{frames} frames"
            );
        }
        // The fixture clip: eleven seconds, 1100 frames, 143 placeholders.
        assert_eq!(output_len(&cfg, 1100), 143);
    }

    #[test]
    fn the_position_table_starts_with_sines_then_cosines() {
        let table = sinusoids(3, 4);
        // Row 0: sin(0) = 0 twice, cos(0) = 1 twice.
        assert_eq!(&table[..4], &[0.0, 0.0, 1.0, 1.0]);
        // Row 1: the first timescale is 1, the last 1/10000.
        assert!((table[4] - 1f32.sin()).abs() < 1e-7);
        assert!((table[5] - 1e-4f32.sin()).abs() < 1e-7);
        assert!((table[6] - 1f32.cos()).abs() < 1e-7);
    }

    /// Frames in different windows must not see each other: the output for
    /// the first window is the same whatever follows it.
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn windows_do_not_attend_across_each_other() {
        let device = crate::qwen3::test_device();
        let cfg = cfg();
        let tower = AudioTower::new(AudioEncoder::init(&cfg, &device), &cfg, DType::F32, &device);
        let frames_one_window = 800;
        let signal = |frames: usize, seed: f32| -> Vec<f32> {
            (0..128 * frames)
                .map(|i| ((i as f32) * 0.37 + seed).sin())
                .collect()
        };
        // The same first eight seconds, followed by two different tails. The
        // mel layout is row-major per bin, so the shared prefix is rebuilt row
        // by row.
        let first = signal(frames_one_window, 0.0);
        let build = |tail_seed: f32| {
            let tail = signal(300, tail_seed);
            let frames = frames_one_window + 300;
            let mut mel = Vec::with_capacity(128 * frames);
            for m in 0..128 {
                mel.extend_from_slice(&first[m * frames_one_window..(m + 1) * frames_one_window]);
                mel.extend_from_slice(&tail[m * 300..(m + 1) * 300]);
            }
            (mel, frames)
        };
        let (a, frames) = build(1.0);
        let (b, _) = build(2.0);
        let window = 104;
        let ea = tower
            .forward(&a, frames, &device, &mut Taps::off())
            .narrow(0, 0, window);
        let eb = tower
            .forward(&b, frames, &device, &mut Taps::off())
            .narrow(0, 0, window);
        let diff = (ea - eb).abs().max().into_scalar::<f32>();
        assert!(diff < 1e-5, "the first window moved by {diff}");
    }

    /// The encoder as it would run without padding: only the real chunks
    /// through the convolutions, the full windows as one batch and the last,
    /// shorter one on its own, neither masked. What the padded [`forward`]
    /// must reproduce.
    ///
    /// [`forward`]: AudioTower::forward
    #[allow(clippy::many_single_char_names)]
    fn unpadded(tower: &AudioTower, mel: &[f32], frames: usize, device: &Device) -> Tensor<2> {
        let cfg = &tower.cfg;
        let (bins, chunk) = (cfg.num_mel_bins, cfg.chunk_frames());
        let chunks = frames.div_ceil(chunk);
        let mut padded = vec![0f32; chunks * bins * chunk];
        for c in 0..chunks {
            let start = c * chunk;
            let len = chunk.min(frames - start);
            for m in 0..bins {
                let to = (c * bins + m) * chunk;
                padded[to..to + len].copy_from_slice(&mel[m * frames + start..][..len]);
            }
        }
        let padded =
            Tensor::<4>::from_data(TensorData::new(padded, [chunks, 1, bins, chunk]), device);
        let embedded = tower.convolve(&padded, chunks, &mut Taps::off())
            + tower.positions.clone().unsqueeze_dim::<3>(0);
        let [_, per_chunk, d] = embedded.dims();
        let len = output_len(cfg, frames);
        let hidden = embedded.reshape([chunks * per_chunk, d]).narrow(0, 0, len);
        let window = per_chunk * cfg.chunks_per_window();
        let heads = tower.encoder.heads();
        let run = |xs: Tensor<3>| {
            let [w, l, _] = xs.dims();
            let open = Tensor::<4, Bool>::from_data(
                TensorData::new(vec![false; w * heads * l * l], [w, heads, l, l]),
                device,
            );
            let xs = tower
                .encoder
                .layers
                .iter()
                .fold(xs, |xs, layer| layer.forward(xs, &open));
            xs.reshape([w * l, d])
        };
        let (full, rest) = (len / window, len % window);
        let mut parts = Vec::new();
        if full > 0 {
            parts.push(run(hidden
                .clone()
                .narrow(0, 0, full * window)
                .reshape([full, window, d])));
        }
        if rest > 0 {
            parts.push(run(hidden
                .narrow(0, full * window, rest)
                .reshape([1, rest, d])));
        }
        let hidden = tower.encoder.ln_post.forward(Tensor::cat(parts, 0));
        let act = tower.activation();
        tower
            .encoder
            .proj2
            .forward(act.forward(tower.encoder.proj1.forward(hidden)))
    }

    /// Padding the audio to whole windows, and the windows to a power of
    /// two, changes no real frame: the cases are a clip under one chunk, a
    /// partial window, exactly one and exactly two windows, and three windows
    /// padded to four.
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn padding_to_a_bucket_changes_no_frame() {
        let device = crate::qwen3::test_device();
        let cfg = cfg();
        let tower = AudioTower::new(AudioEncoder::init(&cfg, &device), &cfg, DType::F32, &device);
        for frames in [37, 530, 800, 1600, 2150] {
            let mel: Vec<f32> = (0..128 * frames)
                .map(|i| ((i as f32) * 0.013).sin())
                .collect();
            let padded = tower.forward(&mel, frames, &device, &mut Taps::off());
            let reference = unpadded(&tower, &mel, frames, &device);
            assert_eq!(padded.dims(), reference.dims(), "{frames} frames");
            let diff = (padded - reference).abs().max().into_scalar::<f32>();
            assert!(
                diff < 1e-5,
                "{frames} frames: the padding moved a frame by {diff}"
            );
        }
    }
}
