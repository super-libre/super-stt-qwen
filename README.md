# Super STT — Qwen3-ASR backend

[![coverage](https://img.shields.io/endpoint?url=https://super-libre.github.io/super-stt-qwen/coverage.json)](https://super-libre.github.io/super-stt-qwen/)

Qwen's [Qwen3-ASR](https://huggingface.co/Qwen) speech recognition models as a
subprocess backend for [Super STT](https://github.com/jorge-menjivar/super-stt).
Thirty languages, detected or named, on Linux and macOS, on the CPU or on any
GPU that CUDA, ROCm, Vulkan or Metal can drive.

## What it is

The Super STT daemon does not compile model inference in-tree. It discovers
backends on disk and drives each one over a `/v1` HTTP contract on a Unix
socket. This is one such backend: a single Rust binary that runs the models
through [Burn](https://github.com/tracel-ai/burn) and answers
`POST /v1/transcribe`.

It is a port of
[super-stt-qwen-asr](https://github.com/jorge-menjivar/super-stt-qwen-asr), the
Python backend that runs the same models through PyTorch and the `qwen-asr`
package, bundled with a relocatable CPython. That one stays as the reference for
driving a `transformers` model from Super STT; this one is what to install.
Against it:

- **One small binary per accelerator** instead of a multi-gigabyte bundle — the
  CUDA one used to ship in two parts to get under GitHub's 2 GiB asset limit.
- **ROCm, Vulkan, ARM and macOS builds**, which a PyTorch wheel could not
  provide in one bundle: Vulkan on x86_64 and ARM Linux, and Metal and the CPU
  on Apple Silicon and Intel Macs. Burn compiles every kernel at runtime
  through CubeCL, so one build per accelerator covers every GPU generation its
  driver can compile for.
- **Streamed previews.** A streamed transcription sends `preview` frames as the
  text is decoded; the Python backend sent nothing until `done`.
- **`POST /v1/cancel` stops a transcription**, within a token. The Python
  backend could only acknowledge it.
- **Forcing a language works.** The Python backend handed the daemon's code
  (`en`) to the model's wrapper, which capitalized it to `En`, found no such
  language and failed the request — so any transcription with a language set
  failed with `inference_failed`. Codes are now mapped to the model's names.

## How a transcription runs

```
samples ─▶ 16 kHz, range-normalized ─▶ log-mel (128 bins, 100 frames/s)
        ─▶ audio encoder: 1 s chunks through 3 convolutions (100 → 13 frames),
                          transformer layers attending within 8 s windows
        ─▶ decoder: the chat prompt with one embedding per 80 ms of audio,
                    decoded greedily until an end-of-text token
        ─▶ "language English<asr_text>…" ─▶ text
```

The decoder's per-token step is captured as a graph and replayed, so a token
costs one dispatch rather than a few hundred kernel launches. A recording
longer than twenty minutes is cut at its quietest point near the limit and the
pieces are transcribed in turn, as the reference does.

### Shapes come in buckets

CubeCL compiles and tunes a kernel per shape, and every shape in this model
follows the length of the audio. Left alone, each new utterance length paid for
its own kernels on its first request — measured on a fresh process, a 5.3 s
clip took 28.5 s where its second run took 0.09. So the shapes are bucketed,
without changing a single real value:

- the encoder runs a power of two of eight-second windows, the padded frames
  masked out of the last real window's attention;
- the prefill runs a power of two of prompt positions, the padding at the end,
  where causal attention never looks;
- each decode step attends over a power-of-two window of one shared cache,
  from 256 positions, so a short transcription does not read a cache sized for
  a long one.

The load's warm-up transcribes one clip per bucket, up to a little over two
minutes of audio, so every request up to that length finds its kernels ready.
Unit tests hold each padding to the unpadded computation, and the parity check
below runs through it.

## Models

Two checkpoints, one loaded at a time. Only the selected model's files are
downloaded.

| Model (`name`)   | Upstream model                                                    | Weights (bf16) | ~VRAM   |
| ---------------- | ----------------------------------------------------------------- | -------------- | ------- |
| `qwen3-asr-0.6b` | [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | 1.9 GB         | ~2.5 GB |
| `qwen3-asr-1.7b` | [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | 4.7 GB         | ~6 GB   |

Both transcribe Chinese, English, Cantonese, Arabic, German, French, Spanish,
Portuguese, Indonesian, Italian, Korean, Russian, Thai, Vietnamese, Japanese,
Turkish, Hindi, Malay, Dutch, Swedish, Danish, Finnish, Polish, Czech,
Filipino, Persian, Greek, Romanian, Hungarian and Macedonian. A request names
one by its BCP-47 code (`src/lang.rs` holds the join), or asks for `auto` — or
nothing — and the model detects it.

### The encoder's attention window

The audio encoder is meant to attend within eight-second windows: that is what
Qwen's vLLM backend computes, and what `transformers` computes under flash
attention. Under its default `sdpa` attention, `transformers` never builds the
equivalent mask — the method that would, `_prepare_attention_mask`, exists and
is not called — so every frame attends to the whole clip. The Python backend
ran that path.

This backend attends within windows, as the model's own serving path does,
which also keeps the encoder's cost linear in the length of the recording.
Clips under eight seconds are unaffected, since they are one window either
way. `scripts/parity_dump.py --full-attention` reproduces the old behaviour for
comparison, and the port matches either to f32 precision (below).

## Numerical parity

The port is checked against the reference layer by layer, not only by its
output. `scripts/parity_dump.py` runs the `transformers` implementation over
`tests/data/jfk.wav` and records every stage — the spectrogram, each
convolution, each of the encoder's 18 or 24 layers and its projections, the
spliced prompt, each of the decoder's 28 layers and its norm, and the logits
of every decoding step — and `src/qwen3/parity.rs` runs the port over the same
input and compares each, teacher-forcing the reference's tokens so every
step's logits are compared.

| Port vs reference                         | Taps | Worst rel. L2 | Greedy tokens |
| ----------------------------------------- | ---: | ------------: | ------------: |
| 0.6B, CPU f32 vs PyTorch f32              |   85 |       1.2e-5  |         30/30 |
| 1.7B, CPU f32 vs PyTorch f32              |   91 |       7.0e-6  |         30/30 |
| 0.6B, CUDA f32 (TF32) vs PyTorch f32      |   85 |       3.6e-3  |         30/30 |
| 0.6B, CUDA bf16 vs PyTorch f32            |   85 |       6.5e-2  |         30/30 |
| 0.6B, Vulkan f16 vs PyTorch f32           |   85 |       2.3e-2  |         30/30 |
| 1.7B, Vulkan f16 vs PyTorch f32           |   91 |       9.7e-3  |         30/30 |

In f32 on the CPU the two are the same arithmetic in a different order. On a
GPU the matmuls run on tensor cores at TF32, which puts every layer near 1e-3
by itself. In bf16 the error is what the narrower type costs: at the decoder's
final norm Burn's bf16 is 6.5e-2 from f32 where PyTorch's own bf16 is 1.5e-1.
f16 keeps three more bits of mantissa than bf16 in a narrower range, which this
model's activations fit: they peak near 1.1e4, and f16's largest value is 6.5e4.

Run it with `just parity [model] [f32|bf16|f16] [float32|bfloat16] [cargo args]`;
it provisions the Python reference in `target/parity-venv` on first use.

## Requirements and performance

A GPU is not required, but the 0.6B model is the one to try without one.
Measured through the socket, the eleven-second fixture clip, request JSON
included, on an RTX 3090 and, for the CPU build, a Ryzen 9 5900X:

| Build        | Model | Load, warm kernel cache | Transcription |
| ------------ | ----- | ----------------------: | ------------: |
| CUDA (bf16)  | 0.6B  |                   4.9 s |        0.21 s |
| CUDA (bf16)  | 1.7B  |                   6.1 s |        0.25 s |
| Vulkan (f16) | 0.6B  |                   5.9 s |        0.19 s |
| Vulkan (f16) | 1.7B  |                   9.9 s |        0.30 s |
| CPU (f32)    | 0.6B  |                   8.1 s |         4.9 s |

The CPU build has no kernel cache; its load is mapping the weights, transposing
them to the row-major layout Burn's CPU backend multiplies fastest, and a short
warm-up.

Vulkan computes in f16, never bf16. The SPIR-V extension for bf16 allows no
arithmetic on it, yet CubeCL emits some, and NVIDIA's driver crashes compiling
it. f16 is what the RTX 3090 computes fastest there: 0.19 s for the clip
against 0.88 s in f32, and 0.89 s against 5.6 s for a 66-second one. A device
that computes in neither half-width type gets f32.

Metal computes in f16 too: every Metal GPU does so natively, and f16 is the
half-width type measured on this model. The Metal and macOS builds are built
and linted in CI, and the CPU one tested there, but no Metal run has been
measured yet.

ROCm computes in f16 as well. On an RDNA1 card (gfx1013, ROCm 7.2) CubeCL
reports bf16 supported and then fails to compile every kernel that uses it;
the Voxtral backend met this and gets CUDA's transcripts word for word in f16
there. This build has not been run on AMD hardware. A load whose warm-up cannot
transcribe at all fails with `load_failed` rather than reporting `ready`.

A clip of a length not seen before costs the same as one that was — 0.12 to
0.31 s on a fresh process for clips of 5 to 18 seconds — because of the
bucketing above.

### Against the Python backend

The same socket client sent both backends the same requests: the fixture clip,
cut or repeated to length, one request after the load and five more timed. The
Python backend is super-stt-qwen-asr's released code on `transformers` 4.57.6
and PyTorch 2.14 (CUDA 13.0), in bf16 as it shipped. Medians:

| Clip  | 0.6B, Python | 0.6B, Burn | 1.7B, Python | 1.7B, Burn |
| ----- | -----------: | ---------: | -----------: | ---------: |
| 5 s   |       0.33 s |     0.08 s |       0.30 s |     0.12 s |
| 11 s  |       0.66 s |     0.14 s |       0.62 s |     0.24 s |
| 33 s  |       1.68 s |     0.38 s |       1.73 s |     0.63 s |
| 66 s  |       2.77 s |     0.64 s |       2.91 s |     1.23 s |
| 121 s |       4.75 s |     1.34 s |       5.15 s |     2.04 s |

`transformers` takes as long for the 1.7B model as for the 0.6B one: it is held
back by the cost of each decoding step on the CPU, not by the GPU, which is
what the captured step removes. Its first request after a load also took 1.2
to 2.4 s, where this backend's warm-up leaves it at 0.15 to 0.22 s, and it
streamed no previews.

What this backend costs in exchange is GPU memory, which it holds from the load
on rather than growing into: 4.2 GiB for the 0.6B model, where PyTorch went
from 2.0 GiB to 4.3 GiB over the long clips, and 7.6 GiB for the 1.7B one,
against PyTorch's 5.8 GiB at most. And the first load of a model builds its
kernels, which takes about six minutes on CUDA and two on Vulkan, against six or
seven seconds for `transformers`; loads after it take four to nine seconds.

On the CPU, the 0.6B model and the eleven-second clip: 4.9 s here, 4.2 to
5.1 s for the Python backend in bf16 as it shipped, and 3.1 to 3.5 s for
`transformers` in f32 (two runs each).

Weights are downloaded by the daemon before the first load. The process has no
network at all — it runs with `PrivateNetwork=yes` and a read-only backend
directory — so it can neither fetch nor write a model file.

### The kernel cache

CubeCL compiles every GPU kernel it meets at runtime and autotunes each
operation against its candidates, keyed by the shape of the problem. So the
load does not stop at mapping the weights: it transcribes synthetic audio at
six lengths — one to ninety seconds, one per bucket — with and without a
language named, and throws the text away, which walks the encoder, the prefill
and the captured decode step at the shapes real requests use. `ready` then
means ready.

That warm-up compiles and tunes every kernel the first time a model loads with
a build, which on an RTX 3090 takes about six minutes on CUDA (361 s for the
0.6B model, 375 s for the 1.7B) and under two on Vulkan (107 s and 112 s).
Nothing ships pre-warmed: every machine builds its own cache, keyed by its own
GPU and driver.

The kernels are kept in `SUPER_STT_BACKEND_CACHE_DIR`, the writable directory
the daemon grants for keeping things between runs, so only that first load
pays: a load after it takes 4 to 6 seconds on CUDA and 5 to 9 on Vulkan. Without the directory granted,
the backend keeps them in its private `/tmp`, which dies with the process, and
every load is a first load.

### Load progress

While a load runs, `GET /v1/status` says what it is doing, for the app to show:

- `phase`: `initial_setup` on the first load of a model with this build, when
  the kernels are built; `loading` after. A marker in the cache directory,
  named by model, version and accelerator, records a warm-up that ran to the
  end, so clearing the cache also brings the setup back. A CPU build compiles
  nothing and always reports `loading`.
- `step`: `loading_weights`, then `building_kernels` on an initial setup or
  `warming_up` after.
- `progress`: how much of the step is done, below 1 until it ends. The
  weights by the bytes of the checkpoint read. Building kernels by the entries
  CubeCL writes to its cache, compiled kernels and tuning results both,
  against how many a cold load wrote on an RTX 3090; past 90% of that
  estimate it closes on 1 without reaching it, so a GPU that writes more slows
  the bar rather than stopping it. Warming up by the warm-up's transcriptions
  done.

The daemon fails a load whose step and progress stand still for two minutes.
On the cold loads above the longest such stretch was 6 s on CUDA and 10 s on
Vulkan, both while building kernels.

## Building

```sh
just build-release          # the pure-Rust CPU backend
just build-cuda             # needs the CUDA headers — no GPU, no compute capability
just build-rocm             # needs the ROCm headers
just build-vulkan           # needs nothing; the loader is found at runtime
just build-metal            # on macOS, with Xcode's command line tools
```

Each build carries exactly one accelerator, which is why the recipes pass
`--no-default-features`: Cargo features are additive, so `--features cuda` on
its own would keep the default CPU backend too.

**Burn comes from a fork.** `Cargo.toml` pins `jorge-menjivar/burn` at
`1e9de733` — the revision the Voxtral backend pins — which is upstream Burn
plus fixes to its fusion crates, to reading a tensor back to the host and to
freeing memory under fusion, and repeats the fork's CubeCL patch (`46f7861b`),
since a `[patch]` in a dependency does not reach the crate being built.

## Testing

```sh
just ci                     # format, lint, build, test — no weights needed
just test-tokenizer         # also fetches the tokenizer files for the reference ids
just test-e2e               # transcribes the fixture through the session (fetches 0.6B)
just parity                 # the layer-by-layer comparison above
```

The unit suite needs no weights: everything that would is behind a model load
and skips without one.

## Installing it

Tag a release and the workflow publishes the tarballs and `backend.toml`; the
daemon installs it from the registry like any other backend. For a local build:

```sh
just stage
```

which puts the binary in the repo root under the name `backend.toml` declares,
making the directory installable with the daemon's import-from-directory path.

## Layout

| Path | What it holds |
|---|---|
| `src/main.rs` | Socket setup and the kernel cache's location. |
| `src/server.rs` | The `/v1` routes and the error codes the contract names. |
| `src/model.rs` | Device and dtype selection, loading, the warm-up, a transcription end to end. |
| `src/progress.rs` | What a load reports in `GET /v1/status` while it runs. |
| `src/model_thread.rs` | The thread the model lives on, since its captured decode step is not `Send`. |
| `src/qwen3/` | The model: feature extraction, encoder, decoder, the thinker joining them, and the parity check. |
| `src/prompt.rs` | The tokenizer, the chat template, and reading the answer back. |
| `src/audio.rs` | Resampling, range normalization, and cutting long recordings. |
| `src/lang.rs` | BCP-47 codes to the names the model prompts with. |
| `scripts/parity_dump.py` | The reference side of the parity check. |
| `backend.toml` | The manifest: models, files and their hashes, release assets. |

## License

GPL-3.0-only. See `LICENSE`, and `NOTICE` for the third-party work this
derives from and the Apache-2.0 weights the daemon fetches at runtime.
