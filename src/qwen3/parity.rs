// SPDX-License-Identifier: GPL-3.0-only
//! Layer-by-layer comparison with the `transformers` implementation, which is
//! what this backend ran before the port.
//!
//! `scripts/parity_dump.py` runs the reference over one clip and writes every
//! layer's output to a safetensors file; this runs the port over the same
//! inputs and compares each against it. Gated, since it needs the weights and
//! the dump:
//!
//! ```sh
//! just parity                       # dumps with PyTorch, then runs this
//! SUPER_STT_PARITY_REF=<dump.safetensors> SUPER_STT_BACKEND_DIR=<dir> \
//!   cargo test --release parity -- --nocapture
//! ```
//!
//! `SUPER_STT_PARITY_DTYPE` picks the port's dtype, `f32` by default, and
//! `SUPER_STT_PARITY_MODEL` the checkpoint, `qwen3-asr-0.6b` by default. In
//! f32 against an f32 dump the two are the same arithmetic in a different
//! order, and every layer is held to that; in bf16 or f16 the table is the
//! measure of what the narrower type costs, and only the transcript is held.
//!
//! The features are compared on their own first; the model is then fed the
//! reference's, so the comparison starts from identical input. From there each
//! layer reads the port's own output of the layer before, so the error at a
//! layer is what it adds plus everything carried in — the table shows how it
//! grows with depth. The decoding steps are teacher-forced with the reference's
//! tokens, so every step's logits are compared even past a near-tie.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use burn::tensor::DType;
use safetensors::{Dtype, SafeTensors};

use crate::qwen3::audio::{MelFilters, frame_count, log_mel_spectrogram};
use crate::qwen3::config::{Config, GenerationConfig};
use crate::qwen3::model::{Prompt, Qwen3Asr};
use crate::qwen3::{Tap, Taps};

/// How far one tensor is from another.
#[derive(Debug, Clone, Copy)]
struct Diff {
    max_abs: f64,
    /// `‖ours − theirs‖ / ‖theirs‖`.
    rel_l2: f64,
    cosine: f64,
    /// `max |theirs|`, for reading `max_abs` against.
    scale: f64,
}

fn diff(ours: &[f32], theirs: &[f32]) -> Diff {
    assert_eq!(ours.len(), theirs.len());
    let (mut max_abs, mut sq_err, mut sq_ref, mut sq_ours, mut dot, mut scale) =
        (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
    for (&a, &b) in ours.iter().zip(theirs) {
        let (a, b) = (f64::from(a), f64::from(b));
        max_abs = max_abs.max((a - b).abs());
        sq_err += (a - b) * (a - b);
        sq_ref += b * b;
        sq_ours += a * a;
        dot += a * b;
        scale = scale.max(b.abs());
    }
    Diff {
        max_abs,
        rel_l2: (sq_err / sq_ref.max(f64::MIN_POSITIVE)).sqrt(),
        cosine: dot / (sq_ref.sqrt() * sq_ours.sqrt()).max(f64::MIN_POSITIVE),
        scale,
    }
}

fn f32s(st: &SafeTensors<'_>, name: &str) -> (Vec<usize>, Vec<f32>) {
    let view = st.tensor(name).unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(view.dtype(), Dtype::F32, "{name} is not f32");
    let values = view
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    (view.shape().to_vec(), values)
}

fn u32s(st: &SafeTensors<'_>, name: &str) -> Vec<u32> {
    let view = st.tensor(name).unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(view.dtype(), Dtype::U32, "{name} is not u32");
    view.data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect()
}

fn model_dir() -> Option<PathBuf> {
    let name = std::env::var("SUPER_STT_PARITY_MODEL").unwrap_or_else(|_| "qwen3-asr-0.6b".into());
    std::env::var_os("SUPER_STT_PARITY_MODEL_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("SUPER_STT_BACKEND_DIR")
                .map(|d| Path::new(&d).join("models").join(&name))
        })
}

/// The safetensors files of a checkpoint directory, sharded or not.
fn weight_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("the model directory")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    files.sort();
    files
}

#[test]
#[allow(clippy::too_many_lines)]
fn layers_match_the_reference() {
    let Some(reference) = std::env::var_os("SUPER_STT_PARITY_REF") else {
        return; // no reference dump provisioned
    };
    let model_dir = model_dir().expect("SUPER_STT_BACKEND_DIR or SUPER_STT_PARITY_MODEL_DIR");
    let dtype = match std::env::var("SUPER_STT_PARITY_DTYPE").as_deref() {
        Ok("bf16") => DType::BF16,
        Ok("f16") => DType::F16,
        Ok("f32") | Err(_) => DType::F32,
        Ok(other) => panic!("SUPER_STT_PARITY_DTYPE takes f32, bf16 or f16, not {other}"),
    };
    let bytes = std::fs::read(&reference).expect("reading the reference dump");
    let st = SafeTensors::deserialize(&bytes).expect("parsing the reference dump");
    let (_, header) = SafeTensors::read_metadata(&bytes).expect("the dump's header");
    let meta = |key: &str| header.metadata().as_ref().and_then(|m| m.get(key).cloned());
    let reference_dtype = meta("dtype").unwrap_or_else(|| "float32".to_string());
    // Windowed is what the port computes; a `--full-attention` dump is the
    // reference's sdpa path, reproduced here to measure against.
    let full_attention = meta("encoder_attention").as_deref() == Some("full");
    // Held layer by layer only when both sides computed in f32; otherwise the
    // table measures a narrower type, not the port. On a GPU "f32" is not
    // quite that: the matmuls run on tensor cores at TF32, a 10-bit mantissa,
    // which puts every layer near 1e-3 by itself. The CPU is held to what f32
    // arithmetic in a different order gives, about 1e-5.
    let strict = dtype == DType::F32 && reference_dtype == "float32";
    let bound = if crate::model::BUILT_FOR == "cpu" {
        1e-4
    } else {
        1e-2
    };

    // The features, from the same samples the reference was given.
    let (_, samples) = f32s(&st, "audio");
    let filters = MelFilters::slaney(128);
    let ours = log_mel_spectrogram(&samples, &filters);
    let frames = frame_count(samples.len());
    let (mel_shape, mel) = f32s(&st, "mel");
    assert_eq!(mel_shape, [128, frames], "the frame count");
    let features = diff(&ours, &mel);
    eprintln!(
        "features: {frames} frames, max |Δ| {:.3e} of max |ref| {:.3e}, rel L2 {:.3e}",
        features.max_abs, features.scale, features.rel_l2
    );
    // A bf16 reference rounds its features to bf16 before the encoder reads
    // them — and the model below is fed those, rounded as they are.
    let feature_bound = if reference_dtype == "float32" {
        1e-5
    } else {
        1e-2
    };
    assert!(
        features.rel_l2 < feature_bound,
        "the features differ from the reference's by rel L2 {:.3e}",
        features.rel_l2
    );

    // The model, from the reference's features, prompt and tokens.
    let config: Config =
        serde_json::from_slice(&std::fs::read(model_dir.join("config.json")).unwrap()).unwrap();
    let generation: GenerationConfig =
        serde_json::from_slice(&std::fs::read(model_dir.join("generation_config.json")).unwrap())
            .unwrap();
    let device = crate::qwen3::test_device();
    let started = std::time::Instant::now();
    let mut model = Qwen3Asr::load(
        &config,
        &weight_files(&model_dir),
        dtype,
        &device,
        &Arc::default(),
    )
    .expect("loading the weights");
    model.set_full_attention(full_attention);
    eprintln!(
        "loaded {} on {device:?} in {dtype:?} in {:.1?}, against a {reference_dtype} reference \
         with {} encoder attention",
        model_dir.display(),
        started.elapsed(),
        if full_attention { "full" } else { "windowed" }
    );

    let input_ids = u32s(&st, "input_ids");
    let tokens = u32s(&st, "tokens");
    let pad = config.thinker_config.audio_token_id;
    let first = input_ids
        .iter()
        .position(|&id| id == pad)
        .expect("audio placeholders");
    let last = input_ids.iter().rposition(|&id| id == pad).unwrap();
    assert_eq!(
        last + 1 - first,
        model.audio_len(frames),
        "the placeholder count"
    );
    let prompt = Prompt {
        prefix: &input_ids[..first],
        suffix: &input_ids[last + 1..],
    };

    let mut taps = Taps::on();
    let started = std::time::Instant::now();
    let audio = model.encode_audio(&mel, frames, &mut taps);
    let ours = model.generate_tapped(
        prompt,
        audio,
        0,
        &generation.eos_token_id,
        Some(&tokens),
        &mut taps,
    );
    eprintln!("ran in {:.1?}", started.elapsed());

    eprintln!(
        "\n{:<14} {:>20} {:>11} {:>11} {:>11} {:>11}",
        "tap", "shape", "max |Δ|", "max |ref|", "rel L2", "1 − cos"
    );
    let mut worst = (0f64, String::new());
    let mut compared = 0;
    for Tap {
        name,
        shape: ours_shape,
        values,
    } in taps.into_records()
    {
        let (shape, theirs) = f32s(&st, &name);
        assert_eq!(ours_shape, shape, "{name}: the shapes differ");
        let d = diff(&values, &theirs);
        eprintln!(
            "{:<14} {:>20} {:>11.3e} {:>11.3e} {:>11.3e} {:>11.3e}",
            name,
            format!("{shape:?}"),
            d.max_abs,
            d.scale,
            d.rel_l2,
            1.0 - d.cosine
        );
        if d.rel_l2 > worst.0 {
            worst = (d.rel_l2, name);
        }
        compared += 1;
    }
    let agree = ours.iter().zip(&tokens).filter(|(a, b)| a == b).count();
    eprintln!(
        "\n{compared} taps; worst rel L2 {:.3e} at {}; greedy tokens agree at {agree} of {} steps",
        worst.0,
        worst.1,
        tokens.len()
    );
    // Three convolutions, conv_out, the encoder layers, ln_post and the two
    // projections; the spliced embeddings, the decoder layers and norm; and
    // the logits of every step.
    let expected = 3
        + 1
        + config.thinker_config.audio_config.encoder_layers
        + 3
        + 1
        + config.thinker_config.text_config.num_hidden_layers
        + 1
        + tokens.len();
    assert_eq!(compared, expected, "a tap is missing from one side");
    assert_eq!(
        ours, tokens,
        "greedy decoding diverged from the reference's"
    );
    if strict {
        assert!(
            worst.0 < bound,
            "in f32 every layer should be within {bound:e} of the reference's, {} is {:.3e}",
            worst.1,
            worst.0
        );
    }
}
