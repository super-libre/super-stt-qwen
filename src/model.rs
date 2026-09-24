// SPDX-License-Identifier: GPL-3.0-only
//! The Qwen3-ASR session: which device to run on, how a `/v1/transcribe`
//! request becomes a prompt, and how the answer becomes text.
//!
//! The model itself is in [`crate::qwen3`]. Everything here is the part that
//! faces the daemon, and it follows what the Python backend it replaces did
//! through the `qwen-asr` package, step for step: audio normalized and cut as
//! [`crate::audio`] describes, the chat template of [`crate::prompt`] with an
//! empty context, greedy decoding, and the answer parsed back to its text.
//!
//! Everything here is synchronous and compute-bound, and the model is not
//! `Send`, so it lives on [`crate::model_thread`] and the handlers send it
//! jobs.

use std::ops::ControlFlow;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use burn::prelude::Device;
use burn::tensor::DType;
use tokenizers::Tokenizer;

use crate::audio;
use crate::lang::Language;
use crate::prompt;
use crate::qwen3::audio::{MelFilters, SAMPLE_RATE, frame_count, log_mel_spectrogram};
use crate::qwen3::config::{Config, GenerationConfig};
use crate::qwen3::model::{Prompt, Qwen3Asr};

/// The models `backend.toml` declares, by wire name. A load naming another is
/// `400 invalid_model`, and a test holds this list to the manifest's.
pub const MODELS: &[&str] = &["qwen3-asr-0.6b", "qwen3-asr-1.7b"];

/// The fewest tokens a transcription may run to.
///
/// What the Python backend allowed every transcription, whatever its length.
/// It is a floor here rather than the whole rule, because at a flat 256 the
/// Python backend cut off anything past a minute and a half of speech.
const MIN_NEW_TOKENS: usize = 256;

/// Tokens allowed per second of audio above [`MIN_NEW_TOKENS`].
///
/// Well past what speech produces — fast English is four tokens a second and
/// fast Mandarin about six — so it never truncates a real transcription. What
/// it bounds is a decoder that has fallen into a loop and would otherwise run
/// until the cache could hold no more.
const NEW_TOKENS_PER_SECOND: f64 = 12.0;

/// The token budget of a piece of audio `samples` long.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn max_new_tokens(samples: usize) -> usize {
    let seconds = samples as f64 / f64::from(SAMPLE_RATE);
    MIN_NEW_TOKENS.max((seconds * NEW_TOKENS_PER_SECOND).ceil() as usize)
}

/// Lengths of audio, in seconds, the warm-up transcribes on a GPU.
///
/// `CubeCL` compiles and tunes a kernel per shape, and the model buckets its
/// shapes so that few exist: the encoder runs a power of two of eight-second
/// windows, the prefill a power of two of prompt positions from 64, and each
/// decode step a power of two of cache positions from 256 (see
/// [`crate::qwen3::model`]). These rungs land on every bucket up to sixteen
/// windows and 2048 positions — a little over two minutes of audio:
///
/// | audio | windows | prefill | decode window |
/// | ----: | ------: | ------: | ------------: |
/// |   1 s |       1 |      64 |           256 |
/// |   4 s |       1 |     128 |           256 |
/// |  10 s |       2 |     256 |           256 |
/// |  20 s |       4 |     512 |           512 |
/// |  40 s |       8 |    1024 |          1024 |
/// |  90 s |      16 |    2048 |          2048 |
///
/// A longer recording meets a bucket the warm-up did not, and pays for its
/// kernels once, on its own request.
const WARM_UP_SECONDS: [f64; 6] = [1.0, 4.0, 10.0, 20.0, 40.0, 90.0];

/// The one rung a CPU build warms, which compiles nothing: it only proves the
/// whole path runs before `ready` says so.
const CPU_WARM_UP_SECONDS: f64 = 2.0;

/// Tokens each warm-up rung decodes. Enough to replay the captured step, and
/// no more: the step's shape does not change from one token to the next.
const WARM_UP_TOKENS: usize = 4;

/// Audio for the warm-up to transcribe.
///
/// Two tones under a slow tremolo with a little noise rather than silence:
/// silence gives the mel filterbank a log of zero, and the point is only to
/// put a signal with energy across the band through the encoder, not to sound
/// like anything.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn warm_up_audio(seconds: f64) -> Vec<f32> {
    let rate = f64::from(SAMPLE_RATE);
    let samples = (seconds * rate) as usize;
    let mut noise = 0x2545_f491_4f6c_dd1d_u64;
    (0..samples)
        .map(|i| {
            let t = i as f64 / rate;
            // xorshift64, for a little broadband energy.
            noise ^= noise << 13;
            noise ^= noise >> 7;
            noise ^= noise << 17;
            let white = (noise >> 40) as f64 / f64::from(1u32 << 24) - 0.5;
            let tremolo = 0.5 + 0.5 * (std::f64::consts::TAU * 3.0 * t).sin();
            let tones = (std::f64::consts::TAU * 220.0 * t).sin()
                + 0.5 * (std::f64::consts::TAU * 1310.0 * t).sin();
            (0.2 * tremolo * tones + 0.02 * white) as f32
        })
        .collect()
}

/// The accelerator this build was compiled for, as the contract names it.
///
/// One build serves one accelerator: Burn's backends are cargo features, and
/// the asset that carries this binary declares the matching `accel`. The
/// constant is what `GET /v1/status` reports and what the mismatch warning in
/// [`select_device`] compares against. The contract has no word for `wgpu`,
/// which is Vulkan underneath on Linux, the one platform this ships for.
pub const BUILT_FOR: &str = if cfg!(feature = "cuda") {
    "cuda"
} else if cfg!(feature = "rocm") {
    "rocm"
} else if cfg!(any(feature = "vulkan", feature = "wgpu")) {
    "vulkan"
} else if cfg!(feature = "metal") {
    "metal"
} else {
    "cpu"
};

/// Whether [`BUILT_FOR`] is a GPU, which decides the dtype and the warm-up.
const ON_GPU: bool = cfg!(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal",
    feature = "wgpu"
));

/// The type the model computes in on `device`.
///
/// A half-width type halves the weights and the bandwidth on a GPU. On the CPU
/// it is slower than f32 rather than faster.
///
/// bf16 is what the reference ran in and what CUDA and ROCm get, where the
/// device computes in it. Never on Vulkan, whatever the device reports: the
/// SPIR-V extension for bf16 allows no arithmetic on it, yet CubeCL emits some,
/// and NVIDIA's driver crashes compiling it. Vulkan gets f16 instead. It
/// holds this model's activations, which peak near 1.1e4 in the 1.7B decoder,
/// and is the closer of the two to f32: for the 0.6B model, 2.3e-2 worst
/// relative error against the reference on Vulkan and 3.8e-2 on CUDA, where
/// bf16 is 6.5e-2, with the same transcript. It is also 4.6 to 6.2 times as
/// fast as f32 on an RTX 3090's Vulkan driver.
///
/// Metal gets f16 too. Every Metal GPU computes in it natively, where bf16
/// arrived with Apple's M-series GPUs and a recent Metal, and CubeCL's bf16
/// has only ever run here on CUDA. A device that computes in neither type gets
/// f32.
fn compute_dtype(device: &Device) -> DType {
    let half = if matches!(BUILT_FOR, "vulkan" | "metal") {
        DType::F16
    } else {
        DType::BF16
    };
    if ON_GPU && device.supports_dtype(half) {
        half
    } else {
        DType::F32
    }
}

/// The device this build runs on, and the name to report for it.
///
/// The daemon sends the accelerator the *installed asset* targets, and it is
/// absent when the daemon has no record of one — an install from a local
/// directory, for instance. Either way the answer is the same: a build has
/// exactly one backend compiled into it, so the request is a cross-check
/// rather than a choice, and a mismatch is worth a line in the log because it
/// means the wrong asset was installed for the host.
#[must_use]
pub fn select_device(requested: Option<&str>) -> (Device, &'static str) {
    if let Some(d) = requested.map(str::trim).filter(|s| !s.is_empty())
        && !d.eq_ignore_ascii_case(BUILT_FOR)
    {
        log::warn!(
            "the daemon asked for {d:?} but this build only has {BUILT_FOR}; using {BUILT_FOR}"
        );
    }
    // One arm per backend, in the order a build that somehow enabled several
    // would prefer them. Cargo features are additive, so the arms have to
    // exclude each other by hand.
    #[cfg(feature = "cuda")]
    return (Device::cuda(0), BUILT_FOR);
    #[cfg(all(not(feature = "cuda"), feature = "rocm"))]
    return (Device::rocm(0), BUILT_FOR);
    #[cfg(all(not(feature = "cuda"), not(feature = "rocm"), feature = "vulkan"))]
    return (
        Device::vulkan(burn::prelude::DeviceKind::DefaultDevice),
        BUILT_FOR,
    );
    #[cfg(all(
        not(feature = "cuda"),
        not(feature = "rocm"),
        not(feature = "vulkan"),
        feature = "metal"
    ))]
    return (
        Device::metal(burn::prelude::DeviceKind::DefaultDevice),
        BUILT_FOR,
    );
    #[cfg(all(
        not(feature = "cuda"),
        not(feature = "rocm"),
        not(feature = "vulkan"),
        not(feature = "metal"),
        feature = "wgpu"
    ))]
    return (
        Device::wgpu(burn::prelude::DeviceKind::DefaultDevice),
        BUILT_FOR,
    );
    // Both CPU backends report `cpu`: they are one accelerator as far as the
    // manifest and the daemon are concerned.
    #[cfg(all(
        not(feature = "cuda"),
        not(feature = "rocm"),
        not(feature = "vulkan"),
        not(feature = "metal"),
        not(feature = "wgpu"),
        feature = "cpu"
    ))]
    return (Device::cpu(), BUILT_FOR);
    #[cfg(all(
        not(feature = "cuda"),
        not(feature = "rocm"),
        not(feature = "vulkan"),
        not(feature = "metal"),
        not(feature = "wgpu"),
        not(feature = "cpu"),
        feature = "flex"
    ))]
    return (Device::flex(), BUILT_FOR);
    #[cfg(all(
        not(feature = "cuda"),
        not(feature = "rocm"),
        not(feature = "vulkan"),
        not(feature = "metal"),
        not(feature = "wgpu"),
        not(feature = "cpu"),
        not(feature = "flex")
    ))]
    (Device::default(), BUILT_FOR)
}

/// Point `CubeCL`'s kernel cache at `cache_dir`.
///
/// `CubeCL` compiles every kernel it meets at runtime and keeps them on disk so
/// only the first run of a build pays. Left to itself it would write under
/// `$HOME`, which the sandbox mounts read-only, so it would recompile on every
/// load.
///
/// Must run before the first device is created, because the configuration is
/// frozen the first time anything reads it.
pub fn configure_kernel_cache(cache_dir: &Path) {
    use burn::cubecl::config::cache::CacheConfig;
    use burn::cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};

    let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
    config.compilation.cache = true;
    config.environment.path = CacheConfig::Directory(cache_dir.to_path_buf());
    // `false` means something already read the configuration and this call is
    // too late to matter. Nothing in this backend touches a device before
    // `main` calls this, so it is a guard rather than a case to handle.
    if !CubeClRuntimeConfig::try_set(config) {
        log::warn!("the CubeCL configuration was already read; the kernel cache keeps its default");
    }
}

/// The pre-warmed autotune cache, shipped in the release tarball beside the
/// binary rather than downloaded.
///
/// One file for every GPU and every model: nothing in the cache is keyed by
/// model — the namespaces are keyed by `CubeCL` version, device and kernel
/// family — so entries a machine cannot use are simply never looked up.
const KERNEL_BUNDLE: &str = "kernels/autotune.bundle";

/// Seed the kernel cache from the bundle in this build, if there is one.
///
/// `CubeCL` compiles every kernel it meets at runtime and then *autotunes*
/// several candidates per operation per shape, and the tuning is the slow half
/// of a cold load. A bundle is that tuning, done once per GPU and shipped.
///
/// Best-effort by construction: every failure here is a slow load rather than
/// a broken one, which is why this warns and returns instead of failing. Must
/// run after [`configure_kernel_cache`] and before anything touches a device.
pub fn import_kernel_bundle(backend_dir: &Path) {
    let path = backend_dir.join(KERNEL_BUNDLE);
    if !path.is_file() {
        log::info!(
            "no {KERNEL_BUNDLE} in this build; every operation is tuned on this machine, \
             which is the slow part of a first load"
        );
        return;
    }
    let started = std::time::Instant::now();
    let bundle = match burn::cubecl::bundle::open(&path) {
        Ok(bundle) => bundle,
        Err(err) => {
            log::warn!("ignoring {}: {err}", path.display());
            return;
        }
    };
    let report = burn::cubecl::bundle::import(bundle.as_ref());
    // `skipped` counts keys the cache already held: the import is insert-only
    // and idempotent.
    log::info!(
        "imported {} kernel-cache entries from {KERNEL_BUNDLE} in {:.1?} \
         ({} namespaces, {} already present, {} refused)",
        report.imported,
        started.elapsed(),
        report.namespaces.len(),
        report.skipped,
        report.failed
    );
}

/// Write the current kernel cache out as a bundle, for shipping.
///
/// The other half of [`import_kernel_bundle`]: there is no way to produce a
/// bundle except by running the work on the hardware it is for. Call it after
/// a load, which warms the cache by running the same ladder a first request
/// would. Autotune results only unless `everything`: they are the expensive
/// half to produce and the durable one — a compiled kernel dies with the next
/// build, a tuning result is keyed by operation and shape and does not.
///
/// # Errors
/// Returns an error if the cache cannot be read or `out` cannot be written.
pub fn export_kernel_bundle(out: &Path, name: &str, everything: bool) -> Result<()> {
    use burn::cubecl::bundle::{BundleFormat, ExportOptions};

    let namespaces = if everything {
        Vec::new()
    } else {
        vec!["autotune".to_string(), "throughput".to_string()]
    };
    let options = ExportOptions {
        name: name.to_string(),
        format: BundleFormat::Sqlite,
        namespaces,
        ..Default::default()
    };
    let manifest =
        burn::cubecl::bundle::export(&[burn::cubecl::environment::path()], out, &options)
            .map_err(|err| anyhow!("exporting the kernel cache to {}: {err}", out.display()))?;
    let tenths = std::fs::metadata(out).map_or(0, |m| m.len() * 10 / (1024 * 1024));
    log::info!(
        "wrote {} ({}.{} MB) for CubeCL {}",
        out.display(),
        tenths / 10,
        tenths % 10,
        manifest.cubecl_version
    );
    Ok(())
}

/// Why a load failed, in the two words `GET /v1/status` has for it.
#[derive(Debug)]
pub enum LoadError {
    /// The build's accelerator could not be initialized: no driver, no device.
    DeviceUnavailable(String),
    /// Anything else: a missing file, a checkpoint this port cannot read.
    Failed(anyhow::Error),
}

impl LoadError {
    /// The contract's machine-readable `reason`.
    #[must_use]
    pub fn reason(&self) -> &'static str {
        match self {
            Self::DeviceUnavailable(_) => "device_unavailable",
            Self::Failed(_) => "load_failed",
        }
    }
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeviceUnavailable(m) => write!(f, "the {BUILT_FOR} device is unavailable: {m}"),
            Self::Failed(e) => write!(f, "{e:#}"),
        }
    }
}

impl From<anyhow::Error> for LoadError {
    fn from(e: anyhow::Error) -> Self {
        Self::Failed(e)
    }
}

/// How a transcription ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// The model finished; the text is the whole transcription.
    Finished(String),
    /// The caller asked to stop; the text is what was transcribed by then.
    Stopped(String),
}

/// A loaded Qwen3-ASR checkpoint, its tokenizer and its feature extractor.
pub struct QwenAsr {
    model: Qwen3Asr,
    tokenizer: Tokenizer,
    filters: MelFilters,
    /// The tokens that end an answer, from `generation_config.json`.
    eos: Vec<u32>,
    device_name: &'static str,
}

// The tokenizer has no `Debug` and the model nothing worth printing.
impl std::fmt::Debug for QwenAsr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QwenAsr")
            .field("model", &self.model)
            .field("eos", &self.eos)
            .field("device_name", &self.device_name)
            .finish_non_exhaustive()
    }
}

/// The one file layout this backend and its manifest agree on.
fn model_file(model_dir: &Path, relative: &str) -> Result<PathBuf> {
    let path = model_dir.join(relative);
    if !path.is_file() {
        bail!(
            "{} is missing; the daemon downloads it from `[[models.files]]` before loading",
            path.display()
        );
    }
    Ok(path)
}

/// The safetensors files of a checkpoint: `model.safetensors`, or the shards
/// its `model.safetensors.index.json` names.
fn weight_files(model_dir: &Path) -> Result<Vec<PathBuf>> {
    let single = model_dir.join("model.safetensors");
    if single.is_file() {
        return Ok(vec![single]);
    }
    let index = model_file(model_dir, "model.safetensors.index.json")?;
    let index: serde_json::Value = serde_json::from_slice(&std::fs::read(&index)?)
        .with_context(|| format!("parsing {}", index.display()))?;
    let map = index
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| anyhow!("the safetensors index has no weight_map"))?;
    let mut shards: Vec<&str> = map.values().filter_map(serde_json::Value::as_str).collect();
    shards.sort_unstable();
    shards.dedup();
    shards
        .into_iter()
        .map(|shard| model_file(model_dir, shard))
        .collect()
}

/// Hold the tokenizer and the template to the checkpoint's own ids.
///
/// The template names the audio tokens by their text and the configuration by
/// their ids; a checkpoint whose tokenizer disagrees would splice the audio in
/// the wrong place and transcribe nothing, quietly. And a language the table in
/// [`crate::lang`] forces but the checkpoint does not list would prompt it with
/// a name it was never trained on — worth a line in the log, not a refusal,
/// since the other twenty-nine still work.
fn check_vocabulary(config: &Config, tokenizer: &Tokenizer) -> Result<()> {
    let thinker = &config.thinker_config;
    for (token, id) in [
        ("<|audio_start|>", thinker.audio_start_token_id),
        ("<|audio_pad|>", thinker.audio_token_id),
        ("<|audio_end|>", thinker.audio_end_token_id),
    ] {
        let found = tokenizer.token_to_id(token);
        if found != Some(id) {
            bail!("the tokenizer has {token} as {found:?}, the configuration as {id}");
        }
    }
    if !config.support_languages.is_empty() {
        for (_, name) in crate::lang::LANGUAGES {
            if !config.support_languages.iter().any(|l| l == name) {
                log::warn!("this checkpoint does not list {name}; forcing it may not work");
            }
        }
    }
    Ok(())
}

/// Run `f` and turn a panic into an error: `CubeCL` panics rather than returns
/// when a runtime cannot find its driver or device.
pub(crate) fn catching<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(|panic| {
        panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_string()))
            .unwrap_or_else(|| "panicked".to_string())
    })
}

impl QwenAsr {
    /// Load `model_name` from a backend directory, reporting progress in
    /// `0.0..=1.0` as it goes.
    ///
    /// # Errors
    /// [`LoadError::DeviceUnavailable`] if the accelerator cannot be
    /// initialized, [`LoadError::Failed`] for anything else.
    pub fn load(
        backend_dir: &Path,
        model_name: &str,
        device: Option<&str>,
        progress: &dyn Fn(f32),
    ) -> Result<Self, LoadError> {
        let dir = backend_dir.join("models").join(model_name);
        let config_file = model_file(&dir, "config.json")?;
        let generation_file = model_file(&dir, "generation_config.json")?;
        let weights = weight_files(&dir)?;

        let config: Config = serde_json::from_slice(
            &std::fs::read(&config_file)
                .with_context(|| format!("reading {}", config_file.display()))?,
        )
        .with_context(|| format!("parsing {}", config_file.display()))?;
        let generation: GenerationConfig = serde_json::from_slice(
            &std::fs::read(&generation_file)
                .with_context(|| format!("reading {}", generation_file.display()))?,
        )
        .with_context(|| format!("parsing {}", generation_file.display()))?;
        let tokenizer = prompt::load_tokenizer(&dir)?;
        check_vocabulary(&config, &tokenizer)?;

        let (device, device_name) = select_device(device);
        // One tensor through the device before a gigabyte of weights: the
        // runtime initializes on first use, and a host without the driver or
        // the device fails here, where it can be named for what it is.
        catching(|| {
            let probe = burn::prelude::Tensor::<1>::zeros([1], &device);
            let _ = probe.into_data();
        })
        .map_err(LoadError::DeviceUnavailable)?;

        let dtype = compute_dtype(&device);
        log::info!(
            "loading {model_name} on {device_name} ({dtype:?}) from {}",
            dir.display()
        );
        let started = std::time::Instant::now();
        progress(0.05);
        let model = Qwen3Asr::load(&config, &weights, dtype, &device)
            .map_err(|e| anyhow!("building the model from {}: {e}", dir.display()))?;
        log::info!("mapped the weights in {:.1?}", started.elapsed());
        progress(0.5);

        let mut model = Self {
            model,
            tokenizer,
            filters: MelFilters::slaney(config.thinker_config.audio_config.num_mel_bins),
            eos: generation.eos_token_id,
            device_name,
        };
        model.warm_up(progress);
        Ok(model)
    }

    /// The device the model is actually running on, as `GET /v1/status`
    /// reports it.
    #[must_use]
    pub fn device_name(&self) -> &'static str {
        self.device_name
    }

    /// Compile and tune the kernels, so the first request does not.
    ///
    /// `CubeCL` compiles every GPU kernel the first time it meets one and tunes
    /// each operation against its candidates, and on a cold cache that is
    /// minutes — which would otherwise land on whoever speaks first, after
    /// `GET /v1/status` has already said `ready`. See [`WARM_UP_SECONDS`] for
    /// why the clips are several lengths rather than one.
    ///
    /// A failure here is logged and swallowed: the model is loaded and usable,
    /// and refusing the load over a warm-up would turn a slow first request
    /// into no service at all.
    fn warm_up(&mut self, progress: &dyn Fn(f32)) {
        let ladder: &[f64] = if ON_GPU {
            &WARM_UP_SECONDS
        } else {
            &[CPU_WARM_UP_SECONDS]
        };
        let started = std::time::Instant::now();
        #[allow(clippy::cast_precision_loss)]
        for (rung, &seconds) in ladder.iter().enumerate() {
            let samples = warm_up_audio(seconds);
            // Both prompts a request can make: a language to detect and one
            // named. They differ in length, and a fresh machine that warmed
            // only one compiled the other's kernels inside its first request —
            // two seconds of it, measured, against 0.17 once compiled.
            for language in [Language::Detect, Language::Forced("English")] {
                let mut tokens = 0;
                let keep_going = || {
                    tokens += 1;
                    tokens <= WARM_UP_TOKENS
                };
                let result = catching(|| self.transcribe(&samples, language, keep_going, |_| {}));
                match result {
                    Ok(Ok(_)) => {}
                    Ok(Err(e)) => {
                        log::warn!("the warm-up failed at {seconds}s: {e:#}");
                        return;
                    }
                    Err(panic) => {
                        log::warn!("the warm-up panicked at {seconds}s: {panic}");
                        return;
                    }
                }
            }
            progress(0.5 + 0.5 * (rung + 1) as f32 / ladder.len() as f32);
        }
        log::info!(
            "warmed {} audio lengths up in {:.1?}",
            ladder.len(),
            started.elapsed()
        );
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| anyhow!("tokenizing the prompt: {e}"))
    }

    fn decode(tokenizer: &Tokenizer, ids: &[u32]) -> String {
        tokenizer.decode(ids, true).unwrap_or_default()
    }

    /// Transcribe `samples`, mono 16 kHz in `[-1, 1]` (see
    /// [`crate::audio::normalize`]).
    ///
    /// `should_continue` is checked once per decoded token, so a cancel is
    /// noticed within a token. `on_preview` sees the transcription so far each
    /// time it grows; a recording long enough to be cut into pieces previews
    /// the finished pieces' text followed by the current one's.
    ///
    /// # Errors
    /// If the prompt cannot be tokenized.
    pub fn transcribe(
        &mut self,
        samples: &[f32],
        language: Language,
        mut should_continue: impl FnMut() -> bool,
        mut on_preview: impl FnMut(&str),
    ) -> Result<Outcome> {
        let (before, after) = prompt::template(language);
        let prefix = self.encode(&before)?;
        let suffix = self.encode(&after)?;

        let mut pieces: Vec<String> = Vec::new();
        for range in audio::split_points(samples) {
            let piece = audio::pad_short(&samples[range]);
            let frames = frame_count(piece.len());
            let mel = log_mel_spectrogram(&piece, &self.filters);
            let encoded = self
                .model
                .encode_audio(&mel, frames, &mut crate::qwen3::Taps::off());

            let Self {
                model, tokenizer, ..
            } = self;
            let done = pieces.concat();
            let mut stopped = false;
            let mut so_far: Vec<u32> = Vec::new();
            let mut shown = String::new();
            let tokens = model.generate(
                Prompt {
                    prefix: &prefix,
                    suffix: &suffix,
                },
                encoded,
                max_new_tokens(piece.len()),
                &self.eos,
                |token| {
                    if !should_continue() {
                        stopped = true;
                        return ControlFlow::Break(());
                    }
                    so_far.push(token);
                    if let Some(text) = prompt::preview(&Self::decode(tokenizer, &so_far), language)
                        && text != shown
                    {
                        on_preview(&format!("{done}{text}"));
                        shown = text;
                    }
                    ControlFlow::Continue(())
                },
            );
            let raw = Self::decode(tokenizer, &tokens);
            if stopped {
                // An answer cut off may not have reached its text yet: before
                // `<asr_text>` it is the model naming the language, and
                // `parse_answer` would read a tagless answer as all text. What
                // a preview would show is what was transcribed.
                pieces.push(prompt::preview(&raw, language).unwrap_or_default());
                return Ok(Outcome::Stopped(pieces.concat()));
            }
            pieces.push(prompt::parse_answer(&raw, language));
        }
        // The reference joins the pieces' texts with nothing between them,
        // which reads right for Chinese and runs two words together in
        // English — only past twenty minutes of audio, where a cut is made.
        Ok(Outcome::Finished(pieces.concat()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_model_list_is_the_manifests() {
        let manifest = include_str!("../backend.toml");
        let declared: Vec<&str> = crate::manifest_probe::model_blocks(manifest)
            .iter()
            .filter_map(|block| crate::manifest_probe::string_field(block, "name"))
            .collect();
        assert_eq!(declared, MODELS);
    }

    #[test]
    fn the_token_budget_grows_with_the_audio() {
        let rate = SAMPLE_RATE as usize;
        assert_eq!(max_new_tokens(rate), MIN_NEW_TOKENS);
        assert_eq!(max_new_tokens(21 * rate), MIN_NEW_TOKENS);
        assert_eq!(max_new_tokens(60 * rate), 720);
    }

    #[test]
    fn the_warm_up_audio_has_energy_and_stays_in_range() {
        let audio = warm_up_audio(1.0);
        assert_eq!(audio.len(), 16_000);
        assert!(audio.iter().all(|s| s.abs() <= 1.0));
        let energy: f32 = audio.iter().map(|s| s * s).sum::<f32>() / 16_000.0;
        assert!(energy > 1e-3, "energy {energy}");
    }

    #[test]
    fn a_missing_checkpoint_fails_the_load_with_its_reason() {
        let err =
            QwenAsr::load(Path::new("/nonexistent"), "qwen3-asr-0.6b", None, &|_| {}).unwrap_err();
        assert_eq!(err.reason(), "load_failed");
        assert!(err.to_string().contains("config.json"), "{err}");
    }

    /// Transcribes the JFK clip end to end through the session — the audio
    /// path, the prompt, the decoding and the parsing — when a checkpoint is
    /// at hand: `QWEN3_ASR_BACKEND_DIR` names a backend directory holding
    /// `models/qwen3-asr-0.6b`.
    #[test]
    fn transcribes_the_fixture_when_weights_are_present() {
        let Some(dir) = std::env::var_os("QWEN3_ASR_BACKEND_DIR") else {
            eprintln!("skipping: QWEN3_ASR_BACKEND_DIR is not set");
            return;
        };
        let samples = crate::audio::tests::read_wav(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/jfk.wav"),
        );
        let samples = crate::audio::normalize(samples, SAMPLE_RATE);
        let mut model = QwenAsr::load(Path::new(&dir), "qwen3-asr-0.6b", None, &|_| {}).unwrap();
        let expected = "And so, my fellow Americans, ask not what your country can do for you; \
                        ask what you can do for your country.";
        for language in [Language::Detect, Language::Forced("English")] {
            let mut previews = Vec::new();
            let outcome = model
                .transcribe(
                    &samples,
                    language,
                    || true,
                    |p| previews.push(p.to_string()),
                )
                .unwrap();
            assert_eq!(outcome, Outcome::Finished(expected.to_string()));
            assert!(previews.len() > 5, "{previews:?}");
            assert_eq!(previews.last().map(String::as_str), Some(expected));
            assert!(expected.starts_with(&previews[0]), "{previews:?}");
        }
        // A stop after five tokens keeps what those five said.
        let mut tokens = 0;
        let outcome = model
            .transcribe(
                &samples,
                Language::Forced("English"),
                || {
                    tokens += 1;
                    tokens <= 5
                },
                |_| {},
            )
            .unwrap();
        let Outcome::Stopped(text) = outcome else {
            panic!("the transcription was not stopped: {outcome:?}");
        };
        assert!(!text.is_empty() && expected.starts_with(&text), "{text:?}");
        // Detecting, the first tokens name the language: a stop there has
        // transcribed nothing yet, and must not answer with the word
        // `language`.
        let mut tokens = 0;
        let outcome = model
            .transcribe(
                &samples,
                Language::Detect,
                || {
                    tokens += 1;
                    tokens <= 1
                },
                |_| {},
            )
            .unwrap();
        assert_eq!(outcome, Outcome::Stopped(String::new()));
    }
}
