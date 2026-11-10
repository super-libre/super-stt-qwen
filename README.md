# Super STT — Qwen3-ASR backend

[![coverage](https://img.shields.io/endpoint?url=https://super-libre.github.io/super-voice-qwen-asr/coverage.json)](https://super-libre.github.io/super-voice-qwen-asr/)

Qwen's [Qwen3-ASR](https://huggingface.co/Qwen) speech recognition models as a
subprocess backend for [Super Voice](https://github.com/jorge-menjivar/super-voice).
Thirty languages, detected or named, on Linux and macOS, on the CPU or on any
GPU that CUDA, ROCm, Vulkan or Metal can drive.

## What it is

The Super Voice daemon does not compile model inference in-tree. It discovers
backends on disk and drives each one over a `/v1` HTTP contract on a Unix
socket. This is one such backend: a single Rust binary that runs the models
through [Burn](https://github.com/tracel-ai/burn) and answers
`POST /v1/transcribe`.

## License

GPL-3.0-only. See `LICENSE`, and `NOTICE` for the third-party work this
derives from and the Apache-2.0 weights the daemon fetches at runtime.
