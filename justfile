# SPDX-License-Identifier: GPL-3.0-only
# Task runner for the standalone Qwen3-ASR backend. Mirrors the recipe names
# used by the main super-stt repo and by the other out-of-tree backends, so
# `just check` means the same thing everywhere.
#
# Burn comes from a fork, pinned by revision in Cargo.toml. The first build
# fetches and compiles it, which is slow; nothing else about the build is
# unusual and there is no C toolchain to install.
#
# Every recipe that builds for a GPU passes `--no-default-features` with one
# accelerator named, matching the release workflow: features are additive, so
# a build that kept the default `flex` alongside `cuda` would carry two
# backends.

hf := "https://huggingface.co/Qwen"
test_backend := justfile_directory() / "target/test-backend"
parity_venv := justfile_directory() / "target/parity-venv"

# Default: build release
default: build-release

# Compiles with debug profile. Usage: just build-debug [args]
build-debug *args:
    cargo build {{ args }}

# Compiles with release profile — the pure-Rust CPU backend.
# Usage: just build-release [args]
build-release *args:
    cargo build --release --locked {{ args }}

# Build with CUDA. Needs the CUDA toolkit headers on the host; no GPU and no
# compute capability are needed, since CubeCL compiles the kernels at runtime.
build-cuda *args:
    cargo build --release --locked --no-default-features --features cuda {{ args }}

# Build with ROCm. Needs the ROCm headers `cubecl-hip-sys` binds against.
build-rocm *args:
    cargo build --release --locked --no-default-features --features rocm {{ args }}

# Build with Vulkan — the vendor-neutral GPU path. Needs no SDK to build; the
# loader is found at runtime.
build-vulkan *args:
    cargo build --release --locked --no-default-features --features vulkan {{ args }}

# Build with Metal, on macOS. Needs Xcode's command line tools, for the SDK
# the bundled SQLite compiles against; the kernels are compiled at runtime.
build-metal *args:
    cargo build --release --locked --no-default-features --features metal {{ args }}

# Cargo already names the artifact as `backend.toml`'s entrypoint, and the
# release workflow tarballs it under the same name, so a local install and a
# published one stage the same bytes.
#
# Build and stage the binary for Import-from-dir. Usage: just stage [args]
stage *args: (build-release args)
    cp target/release/super-stt-backend-qwen super-stt-backend-qwen
    @echo "staged super-stt-backend-qwen — this directory is now installable with Import from dir"

# Remove build output.
clean:
    cargo clean

# Remove build output and every generated artifact in the tree.
clean-all: clean
    rm -f super-stt-backend-qwen lcov.info *.profraw *.profdata

# Runs a clippy check — mirrors super-stt's lint.
check *args:
    cargo clippy --all-targets {{ args }} -- -W clippy::pedantic -D warnings -D unused_must_use

# Runs a clippy check with JSON message format (consumed by clippy-sarif in CI)
check-json: (check '--message-format=json')

# Apply rustfmt to the whole crate
fmt:
    cargo fmt --all

# Check formatting without modifying files
fmt-check:
    cargo fmt --all -- --check

# The suite needs no weights: everything that would need them is behind the
# model load and skips without it. One test does want the tokenizer files;
# see `test-tokenizer`.
#
# Run the test suite. Usage: just test [--verbose]
test *args:
    cargo test --locked {{ args }}

# The daemon fetches these in production; they are 4.5 MB and deliberately
# not in the repository.
#
# Fetch the tokenizer files so the reference tokenization test runs instead of
# skipping.
fetch-tokenizer:
    #!/usr/bin/env bash
    set -euo pipefail
    dir="{{ test_backend }}/models/qwen3-asr-0.6b"
    mkdir -p "$dir"
    for f in vocab.json merges.txt tokenizer_config.json; do
        [ -f "$dir/$f" ] || curl -fL --retry 3 -o "$dir/$f" "{{ hf }}/Qwen3-ASR-0.6B/resolve/main/$f"
    done
    echo "tokenizer files in $dir"

# Run the whole suite including the reference tokenization fixture.
test-tokenizer *args: fetch-tokenizer
    cargo test --locked {{ args }}

# Everything `backend.toml` declares for a model, laid out as the daemon lays
# it out, so `target/test-backend` works as `SUPER_STT_BACKEND_DIR`.
#
# Fetch a model's files. Usage: just fetch-model [qwen3-asr-0.6b|qwen3-asr-1.7b]
fetch-model model="qwen3-asr-0.6b":
    #!/usr/bin/env bash
    set -euo pipefail
    grep -o 'url = "[^"]*", destination = "models/{{ model }}/[^"]*"' backend.toml |
    while read -r line; do
        url=$(echo "$line" | sed 's/url = "\([^"]*\)".*/\1/')
        dest="{{ test_backend }}/$(echo "$line" | sed 's/.*destination = "\([^"]*\)"/\1/')"
        mkdir -p "$(dirname "$dest")"
        [ -f "$dest" ] || curl -fL --retry 3 -o "$dest" "$url"
    done
    echo "{{ model }} in {{ test_backend }}/models/{{ model }}"

# Transcribe the fixture clip end to end through the session, on whatever
# accelerator the features name. Usage: just test-e2e [--no-default-features --features cuda]
test-e2e *args: (fetch-model "qwen3-asr-0.6b")
    QWEN3_ASR_BACKEND_DIR={{ test_backend }} cargo test --release --locked {{ args }} transcribes_the_fixture -- --nocapture

# The reference is `qwen-asr` on `transformers`, installed without its
# dependencies: the forced aligner's tokenizers do not build everywhere and
# transcription never reaches them.
#
# Provision the Python reference the parity check compares against.
parity-env:
    #!/usr/bin/env bash
    set -euo pipefail
    [ -x "{{ parity_venv }}/bin/python" ] && exit 0
    uv venv -q -p 3.11 "{{ parity_venv }}"
    VIRTUAL_ENV="{{ parity_venv }}" uv pip install -q --torch-backend cpu \
        torch "transformers==4.57.6" "accelerate==1.12.0" librosa soundfile numpy safetensors
    VIRTUAL_ENV="{{ parity_venv }}" uv pip install -q --no-deps "qwen-asr==0.0.6"

# Dumps every layer of the reference over the fixture clip, then runs the
# port over the same input and prints the per-layer error table. The port runs
# on whatever accelerator the extra args name; `dtype` is the port's, and the
# reference computes in f32 on the CPU unless `ref_dtype` says bfloat16.
#
# Compare every layer with the reference. Usage: just parity [model] [f32|bf16|f16] [float32|bfloat16] [args]
parity model="qwen3-asr-0.6b" dtype="f32" ref_dtype="float32" *args: parity-env (fetch-model model)
    #!/usr/bin/env bash
    set -euo pipefail
    dump="{{ justfile_directory() }}/target/parity/{{ model }}-{{ ref_dtype }}.safetensors"
    mkdir -p "$(dirname "$dump")"
    # The reference's processor reads two files the backend does not, so the
    # manifest does not list them.
    dir="{{ test_backend }}/models/{{ model }}"
    base=$(grep -o 'url = "[^"]*/config.json", destination = "models/{{ model }}/config.json"' backend.toml |
        sed 's|url = "\([^"]*\)/config.json".*|\1|')
    for f in preprocessor_config.json chat_template.json; do
        [ -f "$dir/$f" ] || curl -fsSL --retry 3 -o "$dir/$f" "$base/$f"
    done
    [ -f "$dump" ] || "{{ parity_venv }}/bin/python" scripts/parity_dump.py \
        "{{ test_backend }}/models/{{ model }}" tests/data/jfk.wav "$dump" --dtype {{ ref_dtype }}
    SUPER_STT_PARITY_REF="$dump" SUPER_STT_BACKEND_DIR="{{ test_backend }}" \
    SUPER_STT_PARITY_MODEL={{ model }} SUPER_STT_PARITY_DTYPE={{ dtype }} \
        cargo test --release --locked {{ args }} layers_match_the_reference -- --nocapture

# cross-rs builds inside a container, so no local CUDA or C toolchain is needed.
#
# Cross-compile for a target. Usage: just cross-build <target>
cross-build target="x86_64-unknown-linux-gnu":
    cross build --release --locked --target {{ target }}

# --remap-path-prefix keeps report paths relative (src/...), and test code is
# excluded so only product code is counted: tests/, and the parity harness,
# which runs only against the reference dump `just parity` writes. CI's badge
# step repeats this pattern.
coverage_ignore := 'tests/|qwen3/parity\.rs'

# Measure coverage, requires cargo-llvm-cov. Usage: just coverage [--html]
coverage *args:
    cargo llvm-cov --locked --remap-path-prefix --ignore-filename-regex '{{ coverage_ignore }}' {{ args }}

# Coverage for CI: write lcov.info and print a summary.
coverage-lcov:
    cargo llvm-cov --locked --remap-path-prefix --ignore-filename-regex '{{ coverage_ignore }}' --lcov --output-path lcov.info
    cargo llvm-cov report --summary-only --ignore-filename-regex '{{ coverage_ignore }}'

# No doctests: this is a binary-only crate, so `cargo test --doc` has no lib
# target.
#
# Full local CI gate: format, lint, build, test.
ci: fmt-check check build-release test
