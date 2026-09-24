// SPDX-License-Identifier: GPL-3.0-only
//! Qwen3-ASR, Qwen's speech recognition model, ported to Burn.
//!
//! The checkpoint is a *thinker* in two pieces, each a module here:
//!
//! - [`encoder`]: the audio tower. Three stride-2 convolutions over one-second
//!   chunks of a 128-bin log-mel spectrogram, sinusoidal positions restarting
//!   at every chunk, then pre-norm transformer layers that attend within
//!   eight-second windows, and a projection to the decoder's width;
//! - [`transformer`]: the text decoder, a Qwen3 stack (RMS norm, per-head query
//!   and key norms, rotary positions, grouped-query attention, `SwiGLU`).
//!
//! [`model`] joins them — the encoded audio replaces the `<|audio_pad|>`
//! placeholders of the chat prompt — and decodes greedily over the result.
//! [`audio`] turns 16 kHz samples into the spectrogram the encoder reads.
//!
//! # Provenance
//!
//! Ported from the `transformers` implementation that ships in the `qwen-asr`
//! package (`qwen_asr/core/transformers_backend/modeling_qwen3_asr.py`, version
//! 0.0.6), which is what this backend ran before, and checked against it on the
//! same clip layer by layer (see `parity`, and [`Taps`]). The decoder stack and the
//! checkpoint adapter below are adapted from the `qwen3-tts` example of
//! `jorge-menjivar/burn` at `9cd83cc49`, with what Qwen3-TTS needs and
//! Qwen3-ASR does not (layer scales, sliding windows) taken out.

pub mod audio;
pub mod config;
pub mod encoder;
pub mod model;
#[cfg(test)]
mod parity;
pub mod transformer;

use burn::nn::{LinearConfig, LinearLayout};
use burn::prelude::*;
use burn::tensor::DType;
use burn_store::burn_pack::Tensor as PackTensor;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use burn_store::{ApplyResult, ModuleAdapter, ModuleContext, PyTorchToBurnAdapter, bridge};

/// Intermediate outputs, recorded by name for the layer-by-layer comparison
/// with the reference (see `parity`).
///
/// Off in production, where recording is one branch per tap and no tensor is
/// touched. On, every tap is read back to the host in f32, which synchronizes
/// the device at every layer — a test-only cost.
#[derive(Debug, Default)]
pub struct Taps {
    records: Option<Vec<Tap>>,
}

/// One recorded output: its name, shape and values. Only the parity check
/// reads them.
#[derive(Debug)]
#[cfg_attr(not(test), allow(dead_code))]
pub struct Tap {
    pub name: String,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

impl Taps {
    /// Taps that record nothing.
    #[must_use]
    pub fn off() -> Self {
        Self::default()
    }

    /// Taps that record everything, in f32.
    #[cfg(test)]
    pub fn on() -> Self {
        Self {
            records: Some(Vec::new()),
        }
    }

    /// Whether anything is being recorded.
    #[must_use]
    pub fn is_on(&self) -> bool {
        self.records.is_some()
    }

    /// Record `tensor` under `name`.
    pub fn record<const D: usize>(&mut self, name: &str, tensor: &Tensor<D>) {
        self.record_with(|| name.to_string(), tensor);
    }

    /// Record `tensor` under the name `name` builds, which is only called when
    /// recording — so a per-layer name costs nothing in production.
    pub fn record_with<const D: usize>(
        &mut self,
        name: impl FnOnce() -> String,
        tensor: &Tensor<D>,
    ) {
        if let Some(records) = &mut self.records {
            let data = tensor.clone().cast(DType::F32).into_data();
            let values = data
                .try_to_vec::<f32>()
                .expect("a tensor cast to f32 reads back as f32");
            records.push(Tap {
                name: name(),
                shape: data.shape.to_vec(),
                values,
            });
        }
    }

    /// What was recorded, in order.
    #[cfg(test)]
    pub fn into_records(self) -> Vec<Tap> {
        self.records.unwrap_or_default()
    }
}

/// The configuration of every linear layer here, on `device`.
///
/// On a GPU the weight keeps the column-major, `[d_output, d_input]` layout of
/// the checkpoints. That is not only about loading without a transpose.
/// Decoding a token runs every linear layer of the decoder on a single row, and
/// the matmul kernels for that product are very sensitive to the layout of the
/// matrix: the Burn port of Qwen3-TTS measured ~55 µs with a row-major weight
/// against ~12 µs, the memory-bandwidth limit, with a column-major one on an
/// RTX 3090.
///
/// Burn's flex CPU backend is the other way round. It multiplies by a
/// column-major matrix through a strided path, and on one thread below 192³
/// multiply-adds, which is every projection of a decode step. On the 0.6B
/// model's shapes that came to 1.48 s a token, against 0.04 s row-major, on a
/// Ryzen 9 5900X. So on flex the weights are row-major, `[d_input, d_output]`,
/// and transposed once as they are loaded (see [`CheckpointAdapter`]).
pub(crate) fn linear_config(d_input: usize, d_output: usize, device: &Device) -> LinearConfig {
    LinearConfig::new(d_input, d_output).with_layout(linear_layout(device))
}

/// The layout [`linear_config`] gives a weight on `device`.
fn linear_layout(device: &Device) -> LinearLayout {
    if is_flex(device) {
        LinearLayout::Row
    } else {
        LinearLayout::Col
    }
}

/// Whether `device` is Burn's flex CPU backend.
#[cfg(feature = "flex")]
fn is_flex(device: &Device) -> bool {
    *device == Device::flex()
}

/// Whether `device` is Burn's flex CPU backend, which a build without the
/// `flex` feature does not have.
#[cfg(not(feature = "flex"))]
fn is_flex(_: &Device) -> bool {
    false
}

/// Counts the checkpoint's bytes into `read` as each tensor's are drawn, which
/// is how far a load has got. First in the chain, so it counts what the file
/// holds rather than what a cast turns it into.
///
/// Adapted from the Voxtral backend's adapter of the same name.
#[derive(Debug, Clone)]
pub(crate) struct ReadCounter(pub Arc<AtomicU64>);

impl ModuleAdapter for ReadCounter {
    fn adapt(&self, tensor: PackTensor, _ctx: ModuleContext<'_>) -> PackTensor {
        let read = Arc::clone(&self.0);
        let bytes = tensor.byte_len() as u64;
        let (name, dtype, shape) = (tensor.name.clone(), tensor.dtype, tensor.shape.clone());
        bridge::map_data(tensor, name, dtype, shape, move |data| {
            read.fetch_add(bytes, Ordering::Relaxed);
            data
        })
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

/// Loads the PyTorch checkpoints into modules built with [`linear_config`].
///
/// Renames the parameters of the normalization layers — `weight` and `bias` in
/// PyTorch, `gamma` and `beta` in Burn. A column-major linear layer stores its
/// weight as `[d_output, d_input]`, which is exactly the PyTorch layout, so on
/// a GPU the linear weights are left alone. A row-major one, on flex, needs
/// them transposed, which is what `burn-store`'s own PyTorch adapter does on
/// top of the renaming, so there the work is handed to it.
#[derive(Debug, Clone)]
pub(crate) struct CheckpointAdapter {
    transpose: bool,
}

impl CheckpointAdapter {
    /// The adapter for the modules [`linear_config`] builds on `device`.
    pub(crate) fn for_device(device: &Device) -> Self {
        Self {
            transpose: matches!(linear_layout(device), LinearLayout::Row),
        }
    }

    fn is_normalization_layer(module_type: &str) -> bool {
        matches!(
            module_type,
            "Struct:BatchNorm" | "Struct:LayerNorm" | "Struct:GroupNorm" | "Struct:RmsNorm"
        )
    }
}

impl ModuleAdapter for CheckpointAdapter {
    fn adapt(&self, mut tensor: PackTensor, ctx: ModuleContext<'_>) -> PackTensor {
        if self.transpose {
            return PyTorchToBurnAdapter.adapt(tensor, ctx);
        }
        let Some(module_type) = ctx.module_type() else {
            return tensor;
        };
        if !Self::is_normalization_layer(module_type) {
            return tensor;
        }
        let start = tensor.name.rfind('.').map_or(0, |dot| dot + 1);
        let renamed = match &tensor.name[start..] {
            "weight" => "gamma",
            "bias" => "beta",
            _ => return tensor,
        };
        tensor.name.truncate(start);
        tensor.name.push_str(renamed);
        tensor
    }

    /// The store looks the parameters up under their Burn names, which the
    /// checkpoint does not use for the normalization layers.
    fn get_alternative_param_name(&self, param_name: &str, module_type: &str) -> Option<String> {
        if !Self::is_normalization_layer(module_type) {
            return None;
        }
        match param_name {
            "gamma" => Some("weight".to_string()),
            "beta" => Some("bias".to_string()),
            _ => None,
        }
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

/// Folds the outcome of loading every shard of a checkpoint into one verdict.
///
/// A sharded checkpoint — the 1.7B one is two files — is loaded one file at a
/// time, each allowed to be partial, so a parameter one shard does not carry is
/// reported missing by that shard even when another fills it. What is really
/// missing is what *every* shard reported missing. Tensors no parameter claims
/// are an error too: a checkpoint carrying weights this port never reads is one
/// it does not implement, and running it anyway would be quietly wrong.
pub(crate) fn check_shards(results: &[ApplyResult]) -> Result<(), String> {
    let errors: Vec<String> = results
        .iter()
        .flat_map(|r| r.errors.iter().map(ToString::to_string))
        .collect();
    if !errors.is_empty() {
        return Err(errors.join(", "));
    }
    let Some((first, rest)) = results.split_first() else {
        return Err("the checkpoint has no shards".to_string());
    };
    let missing: Vec<&str> = first
        .missing
        .iter()
        .map(|(path, _)| path.as_str())
        .filter(|path| {
            rest.iter()
                .all(|r| r.missing.iter().any(|(p, _)| p == path))
        })
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "{} parameters were not found in any shard, e.g. {:?}",
            missing.len(),
            &missing[..missing.len().min(5)]
        ));
    }
    let unused: Vec<&str> = results
        .iter()
        .flat_map(|r| r.unused.iter().map(String::as_str))
        .collect();
    if !unused.is_empty() {
        return Err(format!(
            "{} tensors of the checkpoint belong to no parameter, e.g. {:?}",
            unused.len(),
            &unused[..unused.len().min(5)]
        ));
    }
    Ok(())
}

/// The device the unit tests run on.
#[cfg(test)]
pub(crate) fn test_device() -> burn::prelude::Device {
    crate::model::select_device(None).0
}
