# SPDX-License-Identifier: GPL-3.0-only
"""Dump every layer of the reference Qwen3-ASR over one clip, for the Burn port
to be compared against (see src/qwen3/parity.rs).

    parity_dump.py <model-dir> <clip.wav> <out.safetensors> [--dtype float32|bfloat16] [--language English]

The reference is what this backend ran before the port: the `transformers`
implementation in the `qwen-asr` package, driven through its own
`Qwen3ASRModel.transcribe`, so the prompt, the features and the decoding are
exactly what that backend computed. Forward hooks record each module's output
on the way through, under the names the port's taps use:

    audio, mel                       the samples as the model saw them, and their spectrogram
    enc.conv{1,2,3}                  each convolution, before its GELU
    enc.conv_out                     the flattening projection, before the positions
    enc.layers.{i}                   each encoder layer
    enc.ln_post, enc.proj1, enc.proj2
    dec.embeds                       the prompt with the audio spliced in
    dec.layers.{i}, dec.norm         the decoder over the prompt (the prefill)
    logits.{n}                       the logits of decoding step n
    input_ids, tokens                the prompt, and the greedy tokens with their end

By default the encoder's attention is windowed, as the flash-attention and
vLLM paths compute it; `--full-attention` leaves it as `transformers`' sdpa
path runs it, across the whole clip (see below).

`just parity` provisions an environment with `qwen-asr` installed without its
dependencies — the forced aligner's tokenizers are stubbed out below, since
transcription never reaches them.
"""

import argparse
import sys
import types

# The forced aligner's imports, which plain transcription never touches, and
# whose packages do not build everywhere.
for _name in ("nagisa", "soynlp", "soynlp.tokenizer", "qwen_omni_utils"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["soynlp.tokenizer"].LTokenizer = object

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from qwen_asr import Qwen3ASRModel  # noqa: E402
from qwen_asr.inference.utils import normalize_audios  # noqa: E402
from safetensors.numpy import save_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("clip")
    parser.add_argument("out")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--language", default=None)
    parser.add_argument(
        "--full-attention",
        action="store_true",
        help="leave the encoder attending across its windows, as the sdpa path does",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    dtype = getattr(torch, args.dtype)
    wav, sr = sf.read(args.clip, dtype="float32")
    asr = Qwen3ASRModel.from_pretrained(
        args.model_dir, dtype=dtype, device_map="cpu", max_new_tokens=512
    )
    thinker = asr.model.thinker
    tower = thinker.audio_tower
    text = thinker.model

    # The encoder's layers are meant to attend within windows of eight chunks:
    # the flash-attention path and the vLLM backend both pass `cu_seqlens` to
    # a varlen kernel that does exactly that. Under `sdpa` and `eager` the
    # encoder never builds the equivalent mask — `_prepare_attention_mask`
    # exists and is not called — so every frame attends to the whole clip.
    # The window is restored here unless asked not to, by handing each layer
    # the block-diagonal mask `_prepare_attention_mask` builds.
    if not args.full_attention:

        def windowed(layer, layer_args, kwargs):
            hidden, cu_seqlens = layer_args[0], layer_args[1]
            kwargs["attention_mask"] = tower._prepare_attention_mask(hidden, cu_seqlens)
            return layer_args, kwargs

        for layer in tower.layers:
            layer.register_forward_pre_hook(windowed, with_kwargs=True)

    out: dict[str, np.ndarray] = {}

    def keep(name, tensor, only_first=True):
        if only_first and name in out:
            return
        out[name] = tensor.detach().to(torch.float32).cpu().numpy().copy()

    def hook(name, pick=lambda o: o, only_first=True):
        return lambda _m, _i, o: keep(name, pick(o), only_first)

    handles = []
    for i, conv in enumerate((tower.conv2d1, tower.conv2d2, tower.conv2d3), 1):
        handles.append(conv.register_forward_hook(hook(f"enc.conv{i}")))
    handles.append(tower.conv_out.register_forward_hook(hook("enc.conv_out")))
    for i, layer in enumerate(tower.layers):
        handles.append(layer.register_forward_hook(hook(f"enc.layers.{i}", lambda o: o[0])))
    handles.append(tower.ln_post.register_forward_hook(hook("enc.ln_post")))
    handles.append(tower.proj1.register_forward_hook(hook("enc.proj1")))
    handles.append(tower.proj2.register_forward_hook(hook("enc.proj2")))

    # The decoder's input is a keyword argument; its first call is the prefill.
    def embeds(_m, _args, kwargs):
        if "dec.embeds" not in out:
            keep("dec.embeds", kwargs["inputs_embeds"])

    handles.append(text.register_forward_pre_hook(embeds, with_kwargs=True))
    for i, layer in enumerate(text.layers):
        handles.append(layer.register_forward_hook(hook(f"dec.layers.{i}")))
    handles.append(text.norm.register_forward_hook(hook("dec.norm")))

    steps = []

    def logits(_m, _i, o):
        steps.append(o[:, -1:, :].detach().to(torch.float32).cpu().numpy().copy())

    handles.append(thinker.lm_head.register_forward_hook(logits))

    features = {}
    audio_features = thinker.get_audio_features

    def capture_features(input_features, feature_attention_mask=None, **kw):
        n = int(feature_attention_mask.sum())
        features["mel"] = input_features[0, :, :n]
        return audio_features(input_features, feature_attention_mask=feature_attention_mask, **kw)

    thinker.get_audio_features = capture_features

    generate = asr.model.generate
    prompt = {}

    def capture_generate(**kw):
        result = generate(**kw)
        prompt["input_ids"] = kw["input_ids"][0]
        prompt["tokens"] = result.sequences[0, kw["input_ids"].shape[1] :]
        return result

    asr.model.generate = capture_generate

    # What `transcribe` hands the processor: mono 16 kHz, range-normalized.
    (samples,) = normalize_audios([(wav, sr)])
    (result,) = asr.transcribe(audio=(samples, 16000), language=args.language)
    for h in handles:
        h.remove()

    out["audio"] = np.asarray(samples, dtype=np.float32)
    keep("mel", features["mel"])
    out["input_ids"] = prompt["input_ids"].cpu().numpy().astype(np.uint32)
    out["tokens"] = prompt["tokens"].cpu().numpy().astype(np.uint32)
    assert len(steps) == len(out["tokens"]), (len(steps), len(out["tokens"]))
    for n, step in enumerate(steps):
        out[f"logits.{n}"] = step

    save_file(
        out,
        args.out,
        metadata={
            "dtype": args.dtype,
            "encoder_attention": "full" if args.full_attention else "windowed",
            "language": args.language or "",
            "text": result.text,
            "detected": result.language,
        },
    )
    print(
        f"{len(out)} tensors, {len(steps)} steps -> {args.out}\n"
        f"{result.language}: {result.text!r}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
