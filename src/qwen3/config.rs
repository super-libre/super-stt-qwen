// SPDX-License-Identifier: GPL-3.0-only
//! Configuration of the Qwen3-ASR checkpoints, as shipped in their
//! `config.json`: an audio encoder and a Qwen3 text decoder, both under
//! `thinker_config`.
//!
//! Only what the port reads is declared; the files carry every field of the
//! `transformers` config classes they were saved from, and serde ignores the
//! rest.

use burn::prelude::*;
use burn::tensor::activation::{gelu, silu};
use serde::Deserialize;

/// Activation of an MLP. The audio encoder uses `gelu`, the decoder `silu`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    Silu,
    /// The exact, erf-based GELU — PyTorch's default and Burn's `gelu`.
    Gelu,
}

impl Activation {
    pub fn forward<const D: usize>(self, xs: Tensor<D>) -> Tensor<D> {
        match self {
            Self::Silu => silu(xs),
            Self::Gelu => gelu(xs),
        }
    }
}

/// The audio encoder, `thinker_config.audio_config`.
#[derive(Debug, Clone, Deserialize)]
pub struct AudioEncoderConfig {
    pub num_mel_bins: usize,
    pub encoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub d_model: usize,
    pub activation_function: Activation,
    /// Rows of the sinusoidal position table. Positions restart at every
    /// chunk, so only the first few are ever read.
    pub max_source_positions: usize,
    /// Half the mel frames of one convolution chunk: 50, so a chunk is 100
    /// frames, one second.
    pub n_window: usize,
    /// Mel frames per attention window: 800, eight chunks.
    pub n_window_infer: usize,
    /// Chunks run through the convolutions at once. Only bounds memory.
    pub conv_chunksize: usize,
    /// Channels of the three downsampling convolutions.
    pub downsample_hidden_size: usize,
    /// The width of what the encoder hands the decoder: its hidden size.
    pub output_dim: usize,
    #[serde(default)]
    pub scale_embedding: bool,
}

impl AudioEncoderConfig {
    /// Mel frames per convolution chunk.
    pub fn chunk_frames(&self) -> usize {
        self.n_window * 2
    }

    /// Chunks per attention window.
    pub fn chunks_per_window(&self) -> usize {
        self.n_window_infer / self.chunk_frames()
    }

    /// The frequency rows left after the three stride-2 convolutions, which
    /// is what the flattening in front of `conv_out` multiplies the channels by.
    pub fn downsampled_mel_bins(&self) -> usize {
        conv_out_len(conv_out_len(conv_out_len(self.num_mel_bins)))
    }
}

/// The length a 3-wide, stride-2, 1-padded convolution leaves of `len`.
pub fn conv_out_len(len: usize) -> usize {
    len.div_ceil(2)
}

/// How the rotary embedding of the decoder is scaled, `rope_scaling`.
///
/// The checkpoints ship multimodal `RoPE` with `rope_type = "default"`. Speech
/// recognition feeds the decoder text and audio positions only — never an image
/// grid — so the three position streams are the same sequence, and `MRoPE` with
/// identical streams is exactly the standard 1D rotary embedding. Anything but
/// `default` would scale the frequencies, which this port does not implement,
/// so a checkpoint asking for it is refused at load.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    #[serde(default = "default_rope_type")]
    pub rope_type: String,
}

fn default_rope_type() -> String {
    "default".to_string()
}

/// The Qwen3 text decoder, `thinker_config.text_config`.
#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub hidden_act: Activation,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ThinkerConfig {
    pub audio_config: AudioEncoderConfig,
    pub text_config: TextConfig,
    /// `<|audio_pad|>`, the placeholder the encoded audio replaces.
    pub audio_token_id: u32,
    /// `<|audio_start|>`.
    pub audio_start_token_id: u32,
    /// `<|audio_end|>`.
    pub audio_end_token_id: u32,
}

/// The `config.json` of a Qwen3-ASR checkpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub thinker_config: ThinkerConfig,
    /// The languages the checkpoint names, in the English the prompt uses.
    #[serde(default)]
    pub support_languages: Vec<String>,
}

impl Config {
    /// Refuse a checkpoint whose shape this port does not implement, rather
    /// than run it and transcribe nonsense.
    ///
    /// # Errors
    /// Names the first property that is out of reach.
    pub fn validate(&self) -> Result<(), String> {
        let audio = &self.thinker_config.audio_config;
        let text = &self.thinker_config.text_config;
        if let Some(scaling) = &text.rope_scaling
            && scaling.rope_type != "default"
        {
            return Err(format!(
                "rope_type {:?} is not implemented, only \"default\"",
                scaling.rope_type
            ));
        }
        if audio.output_dim != text.hidden_size {
            return Err(format!(
                "the audio encoder outputs {} values per frame, the decoder reads {}",
                audio.output_dim, text.hidden_size
            ));
        }
        if audio.chunk_frames() == 0 || !audio.n_window_infer.is_multiple_of(audio.chunk_frames()) {
            return Err(format!(
                "an attention window of {} frames is not a whole number of {}-frame chunks",
                audio.n_window_infer,
                audio.chunk_frames()
            ));
        }
        if audio.scale_embedding {
            return Err("scale_embedding is not implemented".to_string());
        }
        if !audio.d_model.is_multiple_of(audio.encoder_attention_heads) {
            return Err(format!(
                "{} encoder heads do not divide d_model {}",
                audio.encoder_attention_heads, audio.d_model
            ));
        }
        if !text
            .num_attention_heads
            .is_multiple_of(text.num_key_value_heads)
        {
            return Err(format!(
                "{} key/value heads do not divide {} query heads",
                text.num_key_value_heads, text.num_attention_heads
            ));
        }
        Ok(())
    }
}

/// The `generation_config.json` of a Qwen3-ASR checkpoint. Only the ids that
/// end a transcription are read: the checkpoints decode greedily.
#[derive(Debug, Clone, Deserialize)]
pub struct GenerationConfig {
    #[serde(deserialize_with = "one_or_many")]
    pub eos_token_id: Vec<u32>,
}

fn one_or_many<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<u32>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum OneOrMany {
        One(u32),
        Many(Vec<u32>),
    }
    Ok(match OneOrMany::deserialize(d)? {
        OneOrMany::One(id) => vec![id],
        OneOrMany::Many(ids) => ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shape of the 0.6B checkpoint's config, trimmed to what is read.
    const CONFIG_0_6B: &str = r#"{
      "support_languages": ["Chinese", "English"],
      "thinker_config": {
        "audio_end_token_id": 151670, "audio_start_token_id": 151669, "audio_token_id": 151676,
        "audio_config": {
          "activation_function": "gelu", "conv_chunksize": 500, "d_model": 896,
          "downsample_hidden_size": 480, "encoder_attention_heads": 14, "encoder_ffn_dim": 3584,
          "encoder_layers": 18, "max_source_positions": 1500, "n_window": 50,
          "n_window_infer": 800, "num_mel_bins": 128, "output_dim": 1024, "scale_embedding": false
        },
        "text_config": {
          "attention_bias": false, "head_dim": 128, "hidden_act": "silu", "hidden_size": 1024,
          "intermediate_size": 3072, "max_position_embeddings": 65536, "num_attention_heads": 16,
          "num_hidden_layers": 28, "num_key_value_heads": 8, "rms_norm_eps": 1e-06,
          "rope_scaling": {"interleaved": true, "mrope_interleaved": true,
                           "mrope_section": [24, 20, 20], "rope_type": "default", "type": "default"},
          "rope_theta": 1000000, "vocab_size": 151936
        }
      }
    }"#;

    #[test]
    fn the_shipped_shape_parses_and_validates() {
        let config: Config = serde_json::from_str(CONFIG_0_6B).unwrap();
        config.validate().unwrap();
        let audio = &config.thinker_config.audio_config;
        assert_eq!(audio.chunk_frames(), 100);
        assert_eq!(audio.chunks_per_window(), 8);
        // 128 mel bins → 64 → 32 → 16, which with 480 channels is the 7680
        // inputs of the checkpoint's `conv_out`.
        assert_eq!(
            audio.downsampled_mel_bins() * audio.downsample_hidden_size,
            7680
        );
    }

    #[test]
    fn a_scaled_rotary_embedding_is_refused() {
        let config = CONFIG_0_6B.replace(r#""rope_type": "default""#, r#""rope_type": "yarn""#);
        let config: Config = serde_json::from_str(&config).unwrap();
        assert!(config.validate().unwrap_err().contains("yarn"));
    }

    #[test]
    fn a_rope_scaling_without_a_type_is_the_default() {
        let config = CONFIG_0_6B.replace(r#""rope_type": "default""#, r#""x": 1"#);
        let config: Config = serde_json::from_str(&config).unwrap();
        let scaling = config.thinker_config.text_config.rope_scaling.as_ref();
        assert_eq!(scaling.unwrap().rope_type, "default");
        config.validate().unwrap();
    }

    /// Every shape this port does not implement is refused, by name.
    #[test]
    fn shapes_out_of_reach_are_refused() {
        for (from, to, named) in [
            (
                r#""output_dim": 1024"#,
                r#""output_dim": 512"#,
                "outputs 512",
            ),
            (
                r#""n_window_infer": 800"#,
                r#""n_window_infer": 750"#,
                "750",
            ),
            (r#""n_window": 50"#, r#""n_window": 0"#, "0-frame"),
            (
                r#""scale_embedding": false"#,
                r#""scale_embedding": true"#,
                "scale_embedding",
            ),
            (
                r#""encoder_attention_heads": 14"#,
                r#""encoder_attention_heads": 5"#,
                "5 encoder heads",
            ),
            (
                r#""num_attention_heads": 16"#,
                r#""num_attention_heads": 12"#,
                "do not divide 12",
            ),
        ] {
            assert!(CONFIG_0_6B.contains(from), "{from}");
            let config: Config = serde_json::from_str(&CONFIG_0_6B.replace(from, to)).unwrap();
            let err = config.validate().unwrap_err();
            assert!(err.contains(named), "{to}: {err}");
        }
    }

    #[test]
    fn the_end_of_text_ids_read_as_one_or_many() {
        let many: GenerationConfig =
            serde_json::from_str(r#"{"eos_token_id": [151643, 151645]}"#).unwrap();
        assert_eq!(many.eos_token_id, vec![151_643, 151_645]);
        let one: GenerationConfig = serde_json::from_str(r#"{"eos_token_id": 151645}"#).unwrap();
        assert_eq!(one.eos_token_id, vec![151_645]);
    }

    #[test]
    fn the_convolutions_halve_rounding_up() {
        // The reference computes `(len - 1) // 2 + 1` per stage.
        for len in 1..300 {
            assert_eq!(conv_out_len(len), (len - 1) / 2 + 1);
        }
    }
}
