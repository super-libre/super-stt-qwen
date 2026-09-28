// SPDX-License-Identifier: GPL-3.0-only
//! A tiny Qwen3-ASR checkpoint, written on the fly for the tests that load,
//! warm up and transcribe: CI has the tokenizer of the real one at most, never
//! its weights.
//!
//! The checkpoint has the real one's layout — its tensor names, PyTorch's
//! `[d_output, d_input]` weights, bf16 — at a size that loads in milliseconds,
//! with seeded random weights, so what it transcribes is noise, but the same
//! noise every run. Its tokenizer is byte-level BPE with no merges: the 256
//! byte symbols, then the special tokens the prompt template names.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use burn::tensor::bf16;
use serde_json::{Value, json};

/// The model the fixture is installed as: the manifest's first.
pub const MODEL: &str = "qwen3-asr-0.6b";

/// The special tokens, from id 256 on.
const SPECIAL: [(&str, bool); 7] = [
    ("<|endoftext|>", true),
    ("<|im_start|>", true),
    ("<|im_end|>", true),
    ("<|audio_start|>", true),
    ("<|audio_end|>", true),
    ("<|audio_pad|>", true),
    ("<asr_text>", false),
];
const ENDOFTEXT: u32 = 256;
const IM_END: u32 = 258;
pub const AUDIO_START: u32 = 259;
const AUDIO_END: u32 = 260;
const AUDIO_PAD: u32 = 261;
const VOCAB: usize = 256 + SPECIAL.len();

const MEL_BINS: usize = 128;
const CHANNELS: usize = 4;
const D_MODEL: usize = 8;
const ENCODER_FFN: usize = 16;
const HIDDEN: usize = 16;
const INTERMEDIATE: usize = 32;
const HEADS: usize = 2;
const KV_HEADS: usize = 1;
const HEAD_DIM: usize = 8;
const LAYERS: usize = 2;

/// A backend directory holding the fixture as [`MODEL`], removed when dropped.
pub struct Backend {
    pub dir: PathBuf,
}

impl Backend {
    /// The model's own directory.
    pub fn model_dir(&self) -> PathBuf {
        self.dir.join("models").join(MODEL)
    }

    /// Replace one of the model's JSON files with what `edit` makes of it.
    pub fn edit_json(&self, file: &str, edit: impl FnOnce(&mut Value)) {
        let path = self.model_dir().join(file);
        let mut value: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        edit(&mut value);
        std::fs::write(path, value.to_string()).unwrap();
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// How the fixture's weights are laid out on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// One `model.safetensors`, as the 0.6B checkpoint ships.
    Single,
    /// Two shards and the index naming them, as the 1.7B one does.
    Sharded,
}

/// A fresh backend directory with the fixture in it.
pub fn backend(layout: Layout) -> Backend {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "qwen3-asr-fixture-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let backend = Backend { dir };
    let model = backend.model_dir();
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(model.join("config.json"), config().to_string()).unwrap();
    std::fs::write(
        model.join("generation_config.json"),
        json!({ "eos_token_id": [ENDOFTEXT, IM_END] }).to_string(),
    )
    .unwrap();
    write_tokenizer(&model);
    let tensors = tensors();
    match layout {
        Layout::Single => write_safetensors(&model.join("model.safetensors"), &tensors),
        Layout::Sharded => {
            let (first, second) = tensors.split_at(tensors.len() / 2);
            let mut map = serde_json::Map::new();
            for (file, shard) in [
                ("model-00001-of-00002.safetensors", first),
                ("model-00002-of-00002.safetensors", second),
            ] {
                write_safetensors(&model.join(file), shard);
                for (name, _, _) in shard {
                    map.insert(name.clone(), json!(file));
                }
            }
            std::fs::write(
                model.join("model.safetensors.index.json"),
                json!({ "weight_map": map }).to_string(),
            )
            .unwrap();
        }
    }
    backend
}

/// The fixture's `config.json`, shaped like the real one's.
pub fn config() -> Value {
    json!({
        "support_languages": ["Chinese", "English"],
        "thinker_config": {
            "audio_start_token_id": AUDIO_START,
            "audio_end_token_id": AUDIO_END,
            "audio_token_id": AUDIO_PAD,
            "audio_config": {
                "activation_function": "gelu",
                "conv_chunksize": 500,
                "d_model": D_MODEL,
                "downsample_hidden_size": CHANNELS,
                "encoder_attention_heads": 2,
                "encoder_ffn_dim": ENCODER_FFN,
                "encoder_layers": 1,
                "max_source_positions": 1500,
                "n_window": 50,
                "n_window_infer": 800,
                "num_mel_bins": MEL_BINS,
                "output_dim": HIDDEN,
                "scale_embedding": false
            },
            "text_config": {
                "attention_bias": false,
                "head_dim": HEAD_DIM,
                "hidden_act": "silu",
                "hidden_size": HIDDEN,
                "intermediate_size": INTERMEDIATE,
                "num_attention_heads": HEADS,
                "num_hidden_layers": LAYERS,
                "num_key_value_heads": KV_HEADS,
                "rms_norm_eps": 1e-6,
                "rope_scaling": { "rope_type": "default" },
                "rope_theta": 1_000_000.0,
                "vocab_size": VOCAB
            }
        }
    })
}

/// Every tensor of the checkpoint: its name, shape and values.
type Tensors = Vec<(String, Vec<usize>, Vec<f32>)>;

fn tensors() -> Tensors {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut out: Tensors = Vec::new();
    let mut random = |name: String, shape: &[usize]| {
        let len = shape.iter().product();
        // Scaled by the fan-in, as an initializer would, so the activations
        // stay in range through the layers.
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 / (shape[1..].iter().product::<usize>().max(1) as f32).sqrt();
        let values = (0..len).map(|_| rng.uniform() * scale).collect();
        out.push((name, shape.to_vec(), values));
    };
    let a = "thinker.audio_tower";
    random(format!("{a}.conv2d1.weight"), &[CHANNELS, 1, 3, 3]);
    random(format!("{a}.conv2d1.bias"), &[CHANNELS]);
    for conv in ["conv2d2", "conv2d3"] {
        random(format!("{a}.{conv}.weight"), &[CHANNELS, CHANNELS, 3, 3]);
        random(format!("{a}.{conv}.bias"), &[CHANNELS]);
    }
    random(format!("{a}.conv_out.weight"), &[D_MODEL, CHANNELS * 16]);
    let l = format!("{a}.layers.0");
    for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
        random(format!("{l}.self_attn.{proj}.weight"), &[D_MODEL, D_MODEL]);
        random(format!("{l}.self_attn.{proj}.bias"), &[D_MODEL]);
    }
    random(format!("{l}.fc1.weight"), &[ENCODER_FFN, D_MODEL]);
    random(format!("{l}.fc1.bias"), &[ENCODER_FFN]);
    random(format!("{l}.fc2.weight"), &[D_MODEL, ENCODER_FFN]);
    random(format!("{l}.fc2.bias"), &[D_MODEL]);
    random(format!("{a}.proj1.weight"), &[D_MODEL, D_MODEL]);
    random(format!("{a}.proj1.bias"), &[D_MODEL]);
    random(format!("{a}.proj2.weight"), &[HIDDEN, D_MODEL]);
    random(format!("{a}.proj2.bias"), &[HIDDEN]);
    random("thinker.model.embed_tokens.weight".into(), &[VOCAB, HIDDEN]);
    random("thinker.lm_head.weight".into(), &[VOCAB, HIDDEN]);
    for i in 0..LAYERS {
        let l = format!("thinker.model.layers.{i}");
        random(
            format!("{l}.self_attn.q_proj.weight"),
            &[HEADS * HEAD_DIM, HIDDEN],
        );
        random(
            format!("{l}.self_attn.k_proj.weight"),
            &[KV_HEADS * HEAD_DIM, HIDDEN],
        );
        random(
            format!("{l}.self_attn.v_proj.weight"),
            &[KV_HEADS * HEAD_DIM, HIDDEN],
        );
        random(
            format!("{l}.self_attn.o_proj.weight"),
            &[HIDDEN, HEADS * HEAD_DIM],
        );
        random(format!("{l}.mlp.gate_proj.weight"), &[INTERMEDIATE, HIDDEN]);
        random(format!("{l}.mlp.up_proj.weight"), &[INTERMEDIATE, HIDDEN]);
        random(format!("{l}.mlp.down_proj.weight"), &[HIDDEN, INTERMEDIATE]);
    }
    // The norms at one and zero, as they start out.
    let ones = |n: usize| vec![1.0; n];
    let mut norm = |name: String, n: usize, bias: bool| {
        out.push((format!("{name}.weight"), vec![n], ones(n)));
        if bias {
            out.push((format!("{name}.bias"), vec![n], vec![0.0; n]));
        }
    };
    norm(format!("{l}.self_attn_layer_norm"), D_MODEL, true);
    norm(format!("{l}.final_layer_norm"), D_MODEL, true);
    norm(format!("{a}.ln_post"), D_MODEL, true);
    for i in 0..LAYERS {
        let l = format!("thinker.model.layers.{i}");
        norm(format!("{l}.input_layernorm"), HIDDEN, false);
        norm(format!("{l}.post_attention_layernorm"), HIDDEN, false);
        norm(format!("{l}.self_attn.q_norm"), HEAD_DIM, false);
        norm(format!("{l}.self_attn.k_norm"), HEAD_DIM, false);
    }
    norm("thinker.model.norm".into(), HIDDEN, false);
    out
}

/// Writes `tensors` as a bf16 safetensors file.
fn write_safetensors(path: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape, values) in tensors {
        let start = data.len();
        for v in values {
            data.extend_from_slice(&bf16::from_f32(*v).to_le_bytes());
        }
        header.insert(
            name.clone(),
            json!({ "dtype": "BF16", "shape": shape, "data_offsets": [start, data.len()] }),
        );
    }
    let mut header = Value::Object(header).to_string().into_bytes();
    // The format asks for the data to start 8-byte aligned.
    header.resize(header.len().next_multiple_of(8), b' ');
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend(header);
    file.extend(data);
    std::fs::write(path, file).unwrap();
}

/// A byte-level BPE with no merges: every byte its own token.
fn write_tokenizer(dir: &Path) {
    let vocab: serde_json::Map<String, Value> = byte_symbols()
        .into_iter()
        .enumerate()
        .map(|(id, symbol)| (symbol.to_string(), json!(id)))
        .collect();
    std::fs::write(dir.join("vocab.json"), Value::Object(vocab).to_string()).unwrap();
    std::fs::write(dir.join("merges.txt"), "#version: 0.2\n").unwrap();
    let added: serde_json::Map<String, Value> = SPECIAL
        .iter()
        .enumerate()
        .map(|(i, (content, special))| {
            (
                (256 + i).to_string(),
                json!({ "content": content, "special": special }),
            )
        })
        .collect();
    std::fs::write(
        dir.join("tokenizer_config.json"),
        json!({ "added_tokens_decoder": added }).to_string(),
    )
    .unwrap();
}

/// GPT-2's byte-to-symbol table, which byte-level BPE vocabularies are
/// written in: the printable bytes stand for themselves, the rest are moved
/// up past 255.
fn byte_symbols() -> Vec<char> {
    let printable = |b: u32| {
        (u32::from(b'!')..=u32::from(b'~')).contains(&b)
            || (0xa1..=0xac).contains(&b)
            || (0xae..=0xff).contains(&b)
    };
    let mut shifted = 0;
    (0..256u32)
        .map(|b| {
            let c = if printable(b) {
                b
            } else {
                shifted += 1;
                255 + shifted
            };
            char::from_u32(c).unwrap()
        })
        .collect()
}

/// A seeded xorshift, for weights that are the same every run.
struct Rng(u64);

impl Rng {
    /// Uniform in [-1, 1).
    fn uniform(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        #[allow(clippy::cast_precision_loss)]
        let unit = (self.0 >> 40) as f32 / (1u64 << 24) as f32;
        unit * 2.0 - 1.0
    }
}
