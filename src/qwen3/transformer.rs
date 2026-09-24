// SPDX-License-Identifier: GPL-3.0-only
//! The Qwen3 text decoder: pre-norm, `SwiGLU` MLP, rotary positions,
//! grouped-query attention with per-head RMS normalization of the queries and
//! keys.
//!
//! The checkpoints configure multimodal `RoPE` (`mrope`). Speech recognition only
//! ever feeds the decoder text and audio positions, never an image grid, so the
//! three position streams are the same sequence and the result is exactly the
//! standard 1D rotary embedding implemented here; [`Config::validate`] refuses a
//! checkpoint that would scale it.
//!
//! The parameters live in [`Transformer`], everything that changes while
//! decoding (the rotary tables and the per-layer key/value caches) lives in
//! [`TransformerState`], so the stack is shared while each transcription keeps
//! its own cache.
//!
//! Adapted from `examples/qwen3-tts/src/transformer.rs` of `jorge-menjivar/burn`
//! at the revision `Cargo.toml` pins, less the layer scales and sliding windows
//! only the TTS stacks use.
//!
//! [`Config::validate`]: crate::qwen3::config::Config::validate

use burn::nn::{Linear, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{DType, IndexingUpdateOp};

use crate::qwen3::Taps;
use crate::qwen3::config::{Activation, TextConfig};

/// Precomputed rotary tables, `(positions, head_dim)`.
///
/// The rotation pairs channel `i` with `i + D / 2`, the convention of the
/// reference implementation: `out = x * cos + rotate_half(x) * sin` where
/// `rotate_half` swaps the two halves and negates the first. The tables are
/// stored full width, with the sign of the rotated half folded into `sin`, so
/// that the rotation is one gather along the channels followed by arithmetic
/// the fusion folds into the surrounding kernels.
#[derive(Debug, Clone)]
struct RotaryEmbedding {
    cos: Tensor<2>,
    sin: Tensor<2>,
    /// The channel each output channel is rotated with: `i + D / 2` for the
    /// first half, `i - D / 2` for the second.
    rotate: Tensor<1, Int>,
}

impl RotaryEmbedding {
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap
    )]
    fn new(dim: usize, theta: f64, positions: usize, dtype: DType, device: &Device) -> Self {
        let half = dim / 2;
        // In f32, as the reference computes its inverse frequencies and angles.
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1f32 / (theta as f32).powf((2 * i) as f32 / dim as f32))
            .collect();
        let mut cos = Vec::with_capacity(positions * dim);
        let mut sin = Vec::with_capacity(positions * dim);
        for pos in 0..positions {
            let angles: Vec<f32> = inv_freq.iter().map(|freq| pos as f32 * freq).collect();
            // Both halves see the same angle; the first half subtracts its
            // rotated partner.
            cos.extend(angles.iter().map(|theta| theta.cos()));
            cos.extend(angles.iter().map(|theta| theta.cos()));
            sin.extend(angles.iter().map(|theta| -theta.sin()));
            sin.extend(angles.iter().map(|theta| theta.sin()));
        }
        let rotate: Vec<i64> = (half..dim).chain(0..half).map(|i| i as i64).collect();
        let shape = [positions, dim];
        Self {
            cos: Tensor::<2>::from_data(TensorData::new(cos, shape), device).cast(dtype),
            sin: Tensor::<2>::from_data(TensorData::new(sin, shape), device).cast(dtype),
            rotate: Tensor::<1, Int>::from_data(TensorData::new(rotate, [dim]), device),
        }
    }

    /// The rows of the tables for `seq_len` positions starting at `offset`,
    /// broadcastable over (B, H, L, D). Sliced once per forward and shared by
    /// every layer.
    fn slice(&self, offset: usize, seq_len: usize) -> RotarySlice {
        let dim = self.cos.dims()[1];
        RotarySlice {
            cos: self
                .cos
                .clone()
                .narrow(0, offset, seq_len)
                .reshape([1, 1, seq_len, dim]),
            sin: self
                .sin
                .clone()
                .narrow(0, offset, seq_len)
                .reshape([1, 1, seq_len, dim]),
            rotate: self.rotate.clone(),
        }
    }

    /// The rows of the tables for the single position held by `pos`, looked up
    /// on the device so that the pass does not depend on the position's value.
    fn gather(&self, pos: &Tensor<1, Int>) -> RotarySlice {
        let dim = self.cos.dims()[1];
        RotarySlice {
            cos: self
                .cos
                .clone()
                .select(0, pos.clone())
                .reshape([1, 1, 1, dim]),
            sin: self
                .sin
                .clone()
                .select(0, pos.clone())
                .reshape([1, 1, 1, dim]),
            rotate: self.rotate.clone(),
        }
    }
}

/// The rotary tables narrowed to the positions of one forward pass.
#[derive(Debug, Clone)]
struct RotarySlice {
    cos: Tensor<4>,
    sin: Tensor<4>,
    rotate: Tensor<1, Int>,
}

impl RotarySlice {
    /// Applies `RoPE` to `xs` (B, H, L, D).
    fn apply(&self, xs: Tensor<4>) -> Tensor<4> {
        let rotated = xs.clone().select(3, self.rotate.clone());
        xs * self.cos.clone() + rotated * self.sin.clone()
    }
}

/// Keys and values of one attention layer, preallocated to a fixed capacity,
/// (B, Hkv, capacity, D), and written in place at the position of each call,
/// so that the buffers of a forward pass stay where they are: what a captured
/// graph replays against.
#[derive(Debug, Clone)]
struct KvCache {
    k: Tensor<4>,
    v: Tensor<4>,
}

impl KvCache {
    /// Stores the keys and values `k` and `v` (B, Hkv, L, D) of the tokens at
    /// positions `pos..pos + L` and returns everything cached so far,
    /// (B, Hkv, pos + L, D).
    #[allow(clippy::many_single_char_names)]
    fn append(&mut self, k: Tensor<4>, v: Tensor<4>, pos: usize) -> (Tensor<4>, Tensor<4>) {
        let [b, h, capacity, d] = self.k.dims();
        let len = k.dims()[2];
        assert!(
            pos + len <= capacity,
            "a cache of {capacity} positions cannot hold positions {pos}..{}",
            pos + len
        );
        // Nothing else refers to the cache buffers at this point, so the
        // assignments write into them rather than into copies, and the buffers
        // never move.
        let ranges = [0..b, 0..h, pos..pos + len, 0..d];
        self.k
            .inplace(|cache| cache.slice_assign(ranges.clone(), k));
        self.v.inplace(|cache| cache.slice_assign(ranges, v));
        (
            self.k.clone().narrow(2, 0, pos + len),
            self.v.clone().narrow(2, 0, pos + len),
        )
    }

    /// Stores the keys and values `k` and `v` (B, Hkv, 1, D) of one token at
    /// the position held by `pos` and returns the first `window` positions of
    /// the cache, (B, Hkv, window, D), for the caller to mask.
    fn append_at(
        &mut self,
        k: Tensor<4>,
        v: Tensor<4>,
        pos: &Tensor<1, Int>,
        window: usize,
    ) -> (Tensor<4>, Tensor<4>) {
        self.k
            .inplace(|cache| cache.select_assign(2, pos.clone(), k, IndexingUpdateOp::Assign));
        self.v
            .inplace(|cache| cache.select_assign(2, pos.clone(), v, IndexingUpdateOp::Assign));
        (
            self.k.clone().narrow(2, 0, window),
            self.v.clone().narrow(2, 0, window),
        )
    }
}

/// Where a forward pass puts its tokens in the cache and how far it attends.
#[derive(Clone, Copy)]
enum Step<'a> {
    /// Tokens at `offset..offset + L`, decided when the pass is built.
    Static { offset: usize },
    /// One token at the position held by `pos`, attending to the first
    /// `window` positions of the cache through the additive `mask`
    /// (1, 1, 1, window), which hides the positions after the token.
    Dynamic {
        pos: &'a Tensor<1, Int>,
        mask: &'a Tensor<4>,
        window: usize,
    },
}

/// Everything a [`Transformer`] needs on top of its parameters to decode.
#[derive(Debug, Clone)]
pub struct TransformerState {
    rotary_emb: RotaryEmbedding,
    caches: Vec<KvCache>,
    /// `0..capacity`: what a token's position is compared with to mask the
    /// cache positions after it.
    positions: Tensor<1, Int>,
    head_dim: usize,
    rope_theta: f64,
    dtype: DType,
    device: Device,
}

impl TransformerState {
    /// Caches for one sequence of at most `capacity` positions.
    pub fn new(cfg: &TextConfig, capacity: usize, dtype: DType, device: &Device) -> Self {
        let shape = [1, cfg.num_key_value_heads, capacity, cfg.head_dim];
        Self {
            rotary_emb: RotaryEmbedding::new(cfg.head_dim, cfg.rope_theta, capacity, dtype, device),
            caches: (0..cfg.num_hidden_layers)
                .map(|_| KvCache {
                    k: Tensor::zeros(shape, (device, dtype)),
                    v: Tensor::zeros(shape, (device, dtype)),
                })
                .collect(),
            positions: positions(capacity, device),
            head_dim: cfg.head_dim,
            rope_theta: cfg.rope_theta,
            dtype,
            device: device.clone(),
        }
    }

    /// The number of positions the caches hold.
    pub fn capacity(&self) -> usize {
        self.positions.dims()[0]
    }

    /// Moves the caches to buffers of `capacity` positions, keeping their
    /// content.
    pub fn grow(&mut self, capacity: usize) {
        let old = self.capacity();
        assert!(
            capacity >= old,
            "a cache cannot shrink from {old} to {capacity} positions"
        );
        let grow = |cache: Tensor<4>| {
            let [b, h, _, d] = cache.dims();
            let (device, dtype) = (cache.device(), cache.dtype());
            Tensor::zeros([b, h, capacity, d], (&device, dtype))
                .slice_assign([0..b, 0..h, 0..old, 0..d], cache)
        };
        for cache in &mut self.caches {
            cache.k = grow(cache.k.clone());
            cache.v = grow(cache.v.clone());
        }
        self.rotary_emb = RotaryEmbedding::new(
            self.head_dim,
            self.rope_theta,
            capacity,
            self.dtype,
            &self.device,
        );
        self.positions = positions(capacity, &self.device);
    }

    /// Additive attention mask (1, 1, 1, window) for one query at the
    /// position held by `pos` attending to the first `window` positions of the
    /// cache: `-1e9` after the query.
    fn mask(&self, pos: &Tensor<1, Int>, window: usize) -> Tensor<4> {
        (self.positions.clone().narrow(0, 0, window) - pos.clone())
            .greater_elem(0)
            .float()
            .mul_scalar(-1e9)
            .reshape([1, 1, 1, window])
    }
}

/// `0..capacity` on the device.
fn positions(capacity: usize, device: &Device) -> Tensor<1, Int> {
    let end = i64::try_from(capacity).expect("a capacity fits in an i64");
    Tensor::arange(0..end, device)
}

#[derive(Module, Debug)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    #[module(skip)]
    act_fn: Activation,
}

impl Mlp {
    fn init(cfg: &TextConfig, device: &Device) -> Self {
        let linear = |d_in, d_out| {
            crate::qwen3::linear_config(d_in, d_out, device)
                .with_bias(false)
                .init(device)
        };
        Self {
            gate_proj: linear(cfg.hidden_size, cfg.intermediate_size),
            up_proj: linear(cfg.hidden_size, cfg.intermediate_size),
            down_proj: linear(cfg.intermediate_size, cfg.hidden_size),
            act_fn: cfg.hidden_act,
        }
    }

    fn forward(&self, xs: Tensor<3>) -> Tensor<3> {
        let lhs = self.act_fn.forward(self.gate_proj.forward(xs.clone()));
        let rhs = self.up_proj.forward(xs);
        self.down_proj.forward(lhs * rhs)
    }
}

/// Repeats each key/value head `n_rep` times, the grouped-query attention
/// expansion.
fn repeat_kv(xs: Tensor<4>, n_rep: usize) -> Tensor<4> {
    if n_rep == 1 {
        return xs;
    }
    let [b, h, l, d] = xs.dims();
    xs.unsqueeze_dim::<5>(2)
        .expand([b, h, n_rep, l, d])
        .reshape([b, h * n_rep, l, d])
}

#[derive(Module, Debug)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
}

impl Attention {
    fn init(cfg: &TextConfig, device: &Device) -> Self {
        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let linear = |d_in, d_out| {
            crate::qwen3::linear_config(d_in, d_out, device)
                .with_bias(cfg.attention_bias)
                .init(device)
        };
        let norm = || {
            RmsNormConfig::new(head_dim)
                .with_epsilon(cfg.rms_norm_eps)
                .init(device)
        };
        Self {
            q_proj: linear(cfg.hidden_size, num_heads * head_dim),
            k_proj: linear(cfg.hidden_size, num_kv_heads * head_dim),
            v_proj: linear(cfg.hidden_size, num_kv_heads * head_dim),
            o_proj: linear(num_heads * head_dim, cfg.hidden_size),
            q_norm: norm(),
            k_norm: norm(),
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
        }
    }

    #[allow(clippy::many_single_char_names)]
    fn forward(
        &self,
        xs: Tensor<3>,
        rotary: &RotarySlice,
        cache: &mut KvCache,
        step: Step<'_>,
    ) -> Tensor<3> {
        let [b, l, _] = xs.dims();

        // (B, L, H * D) -> (B, H, L, D)
        let split =
            |xs: Tensor<3>, heads: usize| xs.reshape([b, l, heads, self.head_dim]).swap_dims(1, 2);
        let q = split(self.q_proj.forward(xs.clone()), self.num_heads);
        let k = split(self.k_proj.forward(xs.clone()), self.num_kv_heads);
        let v = split(self.v_proj.forward(xs), self.num_kv_heads);

        // Qwen3 normalizes every head of the queries and keys, over their last
        // dimension, before the rotation.
        let q = rotary.apply(self.q_norm.forward(q));
        let k = rotary.apply(self.k_norm.forward(k));

        let out = match step {
            Step::Dynamic { pos, mask, window } => {
                let (k, v) = cache.append_at(k, v, pos, window);
                self.decode(q, k, v, Some(mask))
            }
            Step::Static { offset } => {
                let (k, v) = cache.append(k, v, offset);
                if l == 1 {
                    self.decode(q, k, v, None)
                } else {
                    let k = repeat_kv(k, self.num_kv_groups);
                    let v = repeat_kv(v, self.num_kv_groups);
                    // One fused kernel rather than a matmul/softmax/matmul
                    // chain. Leaving `scale` unset keeps the default
                    // `1/sqrt(head_dim)` and the flash-attention path; the
                    // causal flag aligns on the bottom right corner, which is
                    // what queries attending to a whole cache need.
                    let options = AttentionModuleOptions {
                        scale: None,
                        softcap: None,
                        is_causal: true,
                    };
                    attention(q, k, v, None, None, options)
                        .swap_dims(1, 2)
                        .reshape([b, l, self.num_heads * self.head_dim])
                }
            }
        };
        self.o_proj.forward(out)
    }

    /// Attention of a single query (B, H, 1, D) over the cached keys and
    /// values (B, Hkv, L, D).
    ///
    /// The attention op has no kernel for one query and falls back to a dozen
    /// small ones on top of the copies expanding the keys and values to every
    /// head. Grouping the query heads by the key/value head they share turns
    /// the expansion into a reshape: `(B, Hkv, G, D) @ (B, Hkv, D, L)` scores
    /// every head at once, and the result comes out in the head order. `mask`
    /// is added to the scores, (1, 1, 1, L) in f32.
    #[allow(clippy::cast_precision_loss)]
    fn decode(
        &self,
        q: Tensor<4>,
        k: Tensor<4>,
        v: Tensor<4>,
        mask: Option<&Tensor<4>>,
    ) -> Tensor<3> {
        let [b, _, _, _] = q.dims();
        let dtype = q.dtype();
        let q = q.reshape([b, self.num_kv_heads, self.num_kv_groups, self.head_dim]);
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        // The softmax runs in f32, as the attention kernels and the reference
        // do.
        let scores = q.matmul(k.transpose()).mul_scalar(scale).cast(DType::F32);
        let scores = match mask {
            Some(mask) => scores + mask.clone(),
            None => scores,
        };
        let probs = burn::tensor::activation::softmax(scores, 3).cast(dtype);
        probs
            .matmul(v)
            .reshape([b, 1, self.num_heads * self.head_dim])
    }
}

#[derive(Module, Debug)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn init(cfg: &TextConfig, device: &Device) -> Self {
        let norm = || {
            RmsNormConfig::new(cfg.hidden_size)
                .with_epsilon(cfg.rms_norm_eps)
                .init(device)
        };
        Self {
            self_attn: Attention::init(cfg, device),
            mlp: Mlp::init(cfg, device),
            input_layernorm: norm(),
            post_attention_layernorm: norm(),
        }
    }

    fn forward(
        &self,
        xs: Tensor<3>,
        rotary: &RotarySlice,
        cache: &mut KvCache,
        step: Step<'_>,
    ) -> Tensor<3> {
        let hidden = self.self_attn.forward(
            self.input_layernorm.forward(xs.clone()),
            rotary,
            cache,
            step,
        );
        let xs = xs + hidden;
        let hidden = self
            .mlp
            .forward(self.post_attention_layernorm.forward(xs.clone()));
        xs + hidden
    }
}

/// A stack of decoder layers followed by the final RMS normalization.
///
/// The stack works on input embeddings rather than token ids: the prompt is
/// token embeddings with the encoded audio spliced in.
#[derive(Module, Debug)]
pub struct Transformer {
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
}

impl Transformer {
    pub fn init(cfg: &TextConfig, device: &Device) -> Self {
        Self {
            layers: (0..cfg.num_hidden_layers)
                .map(|_| DecoderLayer::init(cfg, device))
                .collect(),
            norm: RmsNormConfig::new(cfg.hidden_size)
                .with_epsilon(cfg.rms_norm_eps)
                .init(device),
        }
    }

    /// Runs the stack on `xs` (1, L, hidden) whose first token sits at position
    /// `offset` and returns the normalized hidden states (1, L, hidden). Keys
    /// and values are written to the caches of `state`, so consecutive calls
    /// must use increasing offsets.
    ///
    /// `taps` records every layer's output as `dec.layers.{i}` and the final
    /// norm's as `dec.norm`.
    pub fn forward(
        &self,
        xs: Tensor<3>,
        offset: usize,
        state: &mut TransformerState,
        taps: &mut Taps,
    ) -> Tensor<3> {
        let [_b, seq_len, _] = xs.dims();
        let rotary = state.rotary_emb.slice(offset, seq_len);
        let step = Step::Static { offset };
        let mut xs = xs;
        for (i, (layer, cache)) in self.layers.iter().zip(state.caches.iter_mut()).enumerate() {
            xs = layer.forward(xs, &rotary, cache, step);
            taps.record_with(|| format!("dec.layers.{i}"), &xs);
        }
        let xs = self.norm.forward(xs);
        taps.record("dec.norm", &xs);
        xs
    }

    /// Runs the stack on one token `xs` (1, 1, hidden) at the position held by
    /// `pos`, which is not read on the host: the pass is the same for every
    /// position, so it can be captured once and replayed. The token is written
    /// at its position and attends to the first `window` positions of the
    /// cache, masked after its own; the position must lie inside the window.
    ///
    /// The window is what keeps a short transcription cheap on a cache sized
    /// for a long one: a step reads every key and value it attends to, and at
    /// 4096 positions that is about 470 MB a token for the 0.6B checkpoint.
    pub fn forward_at(
        &self,
        xs: Tensor<3>,
        pos: &Tensor<1, Int>,
        state: &mut TransformerState,
        window: usize,
    ) -> Tensor<3> {
        assert!(
            window <= state.capacity(),
            "a window of {window} positions over a cache of {}",
            state.capacity()
        );
        let rotary = state.rotary_emb.gather(pos);
        let mask = state.mask(pos, window);
        let step = Step::Dynamic {
            pos,
            mask: &mask,
            window,
        };
        let mut xs = xs;
        for (layer, cache) in self.layers.iter().zip(state.caches.iter_mut()) {
            xs = layer.forward(xs, &rotary, cache, step);
        }
        self.norm.forward(xs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen3::test_device;

    fn tiny_config() -> TextConfig {
        TextConfig {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 24,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            hidden_act: Activation::Silu,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            attention_bias: false,
            rope_scaling: None,
        }
    }

    fn max_abs_diff(a: Tensor<3>, b: Tensor<3>) -> f32 {
        (a - b).abs().max().into_scalar::<f32>()
    }

    /// Decoding one token at a time through [`Transformer::forward_at`] must
    /// agree with one pass over the whole sequence: the captured decode step
    /// relies on it, and it is the one property a mask or rotary lookup off by
    /// one position would break.
    #[test]
    fn stepping_one_token_at_a_time_matches_one_pass() {
        let device = test_device();
        let cfg = tiny_config();
        let stack = Transformer::init(&cfg, &device);
        let xs = Tensor::<3>::random(
            [1, 6, cfg.hidden_size],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let mut whole = TransformerState::new(&cfg, 16, DType::F32, &device);
        let expected = stack.forward(xs.clone(), 0, &mut whole, &mut Taps::off());

        // Two tokens as a prefill, then one at a time at a device-held
        // position, the way a transcription runs.
        let mut stepped = TransformerState::new(&cfg, 16, DType::F32, &device);
        let mut outputs = vec![stack.forward(
            xs.clone().narrow(1, 0, 2),
            0,
            &mut stepped,
            &mut Taps::off(),
        )];
        for pos in 2..6 {
            let at = Tensor::<1, Int>::from_data([i64::try_from(pos).unwrap()], &device);
            outputs.push(stack.forward_at(xs.clone().narrow(1, pos, 1), &at, &mut stepped, 16));
        }
        let got = Tensor::cat(outputs, 1);
        let diff = max_abs_diff(expected, got);
        assert!(diff < 1e-4, "stepwise decoding differs by {diff}");
    }

    /// Growing the caches mid-sequence keeps what they held.
    #[test]
    fn a_grown_cache_decodes_as_a_large_one() {
        let device = test_device();
        let cfg = tiny_config();
        let stack = Transformer::init(&cfg, &device);
        let xs = Tensor::<3>::random(
            [1, 5, cfg.hidden_size],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let step = |state: &mut TransformerState, pos: usize| {
            let at = Tensor::<1, Int>::from_data([i64::try_from(pos).unwrap()], &device);
            stack.forward_at(xs.clone().narrow(1, pos, 1), &at, state, 8)
        };

        let mut large = TransformerState::new(&cfg, 8, DType::F32, &device);
        let _ = stack.forward(xs.clone().narrow(1, 0, 4), 0, &mut large, &mut Taps::off());
        let expected = step(&mut large, 4);

        let mut small = TransformerState::new(&cfg, 4, DType::F32, &device);
        let _ = stack.forward(xs.clone().narrow(1, 0, 4), 0, &mut small, &mut Taps::off());
        small.grow(8);
        let got = step(&mut small, 4);
        let diff = max_abs_diff(expected, got);
        assert!(diff < 1e-5, "a grown cache differs by {diff}");
    }

    /// A step attending to a window of the cache sees exactly what one
    /// attending to all of it sees, as long as its position is inside: what
    /// lies past it is masked either way.
    #[test]
    fn a_window_of_the_cache_decodes_as_the_whole() {
        let device = test_device();
        let cfg = tiny_config();
        let stack = Transformer::init(&cfg, &device);
        let xs = Tensor::<3>::random(
            [1, 6, cfg.hidden_size],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let run = |window: usize| {
            let mut state = TransformerState::new(&cfg, 32, DType::F32, &device);
            let _ = stack.forward(xs.clone().narrow(1, 0, 5), 0, &mut state, &mut Taps::off());
            let at = Tensor::<1, Int>::from_data([5i64], &device);
            stack.forward_at(xs.clone().narrow(1, 5, 1), &at, &mut state, window)
        };
        let diff = max_abs_diff(run(8), run(32));
        assert!(diff < 1e-5, "the window changed the step by {diff}");
    }

    /// Padding a prefill at its end changes nothing before the padding, and a
    /// step written where the padding was decodes as if it had never been
    /// there: causal attention never looks ahead, and the step masks what lies
    /// past it. This is what lets the prompt be padded to a length whose
    /// kernels are already compiled.
    #[test]
    fn a_prefill_padded_at_its_end_decodes_as_an_unpadded_one() {
        let device = test_device();
        let cfg = tiny_config();
        let stack = Transformer::init(&cfg, &device);
        let hidden = cfg.hidden_size;
        let xs = Tensor::<3>::random(
            [1, 6, hidden],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let prompt = xs.clone().narrow(1, 0, 5);
        let next = xs.narrow(1, 5, 1);
        let at = Tensor::<1, Int>::from_data([5i64], &device);

        let mut plain = TransformerState::new(&cfg, 16, DType::F32, &device);
        let plain_prefill = stack.forward(prompt.clone(), 0, &mut plain, &mut Taps::off());
        let plain_step = stack.forward_at(next.clone(), &at, &mut plain, 16);

        let padding = Tensor::<3>::ones([1, 3, hidden], &device);
        let mut padded = TransformerState::new(&cfg, 16, DType::F32, &device);
        let padded_prefill = stack.forward(
            Tensor::cat(vec![prompt, padding], 1),
            0,
            &mut padded,
            &mut Taps::off(),
        );
        let padded_step = stack.forward_at(next, &at, &mut padded, 16);

        let diff = max_abs_diff(plain_prefill, padded_prefill.narrow(1, 0, 5));
        assert!(diff < 1e-5, "the padding moved the prompt by {diff}");
        let diff = max_abs_diff(plain_step, padded_step);
        assert!(diff < 1e-5, "the padding moved the next step by {diff}");
    }
}
