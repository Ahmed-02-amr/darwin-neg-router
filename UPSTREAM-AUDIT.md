# Darwin-9B-NEG public-artifact audit

Audited on 2026-09-01 from the Hugging Face model repository, all twelve
commits in its Git history, the public demo Space, and the linked Darwin Family
paper. This is a reproducibility audit, not a claim about the authors' intent.

## What is actually embedded

- The four main language-model shards in `FINAL-Bench/Darwin-9B-NEG` have the
  same sizes and SHA-256 LFS object IDs as the four shards in
  `FINAL-Bench/Darwin-9B-Opus`.
- The learned NEG head and gate are a separate 16.8 MB file named
  `neg_modules.safetensors`. They are not tensors in the main checkpoint
  shards and are not present in community GGUF conversions.
- The supplied `Darwin-9B-NEG.mmproj-Q8_0.gguf` is a 624 MB multimodal
  projector. It is not the 9B language model or the NEG head.
- `config.json` names the standard `Qwen3_5ForConditionalGeneration`
  architecture and has no `auto_map`. Consequently, `trust_remote_code=True`
  does not import or attach `modeling_darwin_neg.py` automatically.
- The authors' separate `FINAL-Bench/Darwin-9B-MFP4` repository is a ModelOpt
  mixed-FP4 quantization of `Darwin-9B-Opus`. It contains neither the NEG module
  file nor an evaluator and is not an Ollama/GGUF artifact.

The shard match is exact, not a filename comparison:

| Shard | Bytes | Shared SHA-256 LFS object ID |
|---|---:|---|
| `model-00001-of-00004.safetensors` | 5,276,436,216 | `8bbd456f1367d1d9d7273b0a5735a57ce73f6a56f5a09c3b99a8a607a0a5f65a` |
| `model-00002-of-00004.safetensors` | 5,335,161,512 | `048129af3b6acde304c92fc262b12db11e25bc23a5187c42a60b0a6ee16749fb` |
| `model-00003-of-00004.safetensors` | 5,368,717,440 | `7283cf97c0bc17a351e1b08ba6b6f3d4c2920704a6d1b1cacfb6ae6510c45730` |
| `model-00004-of-00004.safetensors` | 3,325,988,568 | `53161974a653473c3829f77974fda95d7bbaabc62a9ee309925a663625fdf0ee` |

The released sidecar is 16,785,908 bytes with SHA-256
`8fcc1a5a9f7cdeaf2462af9f6de87ecf7626be8a96287e95bb2a20d63cbcb71a`.
These values were checked from the Hub metadata and the local sidecar on
2026-09-02.

The two community imatrix Q6_K containers do not have matching whole-file
hashes. The Opus-labelled file is 7,359,260,992 bytes with SHA-256
`6ab52e0e34b4c6fe9583e88a0b8e53e18b0e0f2d6652d033fc3508059286a851`;
the NEG-labelled file is 7,359,260,576 bytes with SHA-256
`3304a4913eec467e6775cba66f563199ef4b14e8a1976232e2ec82b7dcaa49bc`.
The project's local Q6_K matches the latter exactly. A GGUF header or metadata
difference is enough to change the container hash, so this audit relies on the
identical BF16 source shards to establish backbone identity and does not claim
bit-identical quantized tensors without a tensor-by-tensor comparison.

## Released NEG helper

The separate helper does contain a two-layer entropy head and a top-20 gate,
but the public package has three mechanical defects:

1. `neg_modules.safetensors` contains `gate.temp_scale`; the published
   `NEGGate` class defines only `threshold`. Its strict `load_state_dict` call
   therefore rejects the released state dictionary.
2. The gate computes `logits * (1 - active) + masked * active`. For a mixed
   active/inactive batch, the inactive rows can receive NaNs from `-inf * 0`.
3. On an active row, the gate preserves the original top 20 logits and masks
   only lower-ranked logits. This cannot change `argmax(logits)`, so it cannot
   alter greedy decoding as implemented.

This third point is also acknowledged by FINAL-Bench itself in the deprecated
`Darwin-28B-NEG` model card: its failure analysis says the top-k gate is
mathematically ineffective in greedy decoding. That repository references a
more detailed `reference_darwin_v8_neg.md`, but the file is absent from both
the present repository and all three commits in its Git history.

The released values are:

- entropy threshold: `1.175187349319458`
- unused `temp_scale`: `0.5983633399009705`
- head: `4096 -> 1024 -> 1`

## 84.34% ensemble protocol

The model card describes:

- four choice-order permutations with majority voting;
- temperature-sampled candidates;
- a second-opinion re-query;
- approximately 20 inference calls in total.

It does not specify the prompt, permutation seed/order, number of samples per
temperature, temperatures, vote weighting, tie breaking, second-opinion
prompt, or final selection rule. The card calls the production protocol
"internal" while saying reproduction scripts are released separately, but no
link is supplied.

The complete model Git history contains no evaluator script. The only paths
ever committed are the checkpoint/tokenizer files, the NEG helper, the model
card, a removed MRI report, and a one-line evaluation-result registration.
That registration points back to the model card as its source and describes
84.34 as "standard inference, Pass@1", while the model card describes it as an
approximately 20-call ensemble. This is a provenance/metadata inconsistency,
not evidence about the authors' intent.
The public `FINAL-Bench/Darwin-9B-NEG` Space performs one ordinary API call and
contains no NEG or ensemble evaluator. The linked Darwin Family paper discusses
the evolutionary merge framework and does not discuss Darwin-9B-NEG or Native
Entropy Gating. Hugging Face displays the 84.34 result as self-reported.

A related repository, `FINAL-Bench/Darwin-9B-NEG-x-Negentropy-V8`, was also
checked across its complete two-commit history. It contains no evaluator. Its
card says the NEG modules were preserved, but its file tree has no
`neg_modules.safetensors` and its checkpoint index has no NEG-head or NEG-gate
tensors; the included helper consequently tries to fetch a file that is absent
from that repository.

The exact 84.34% result should therefore be treated as a reported result that
cannot presently be reproduced from the public artifacts alone. It is not
valid to infer from that fact alone that the result is fabricated.

## Local native reconstruction

Ollama cannot attach the separate hidden-state head or inject a custom logits
processor from a Modelfile. This project therefore patches the exact
Ollama-pinned llama.cpp runner to expose the normalized final hidden state,
load the separately released NEG head, predict entropy for every generated
token, and apply the released gate before sampling. Ollama remains the pinned
source/runtime dependency; the model is served by this companion runner rather
than by stock `ollama serve`.

The general router sits above that native path. Its explicit 20-call profile
uses fifteen role-diverse candidates, three specialist reviewers, one blinded
evaluator, and one refiner. This is a transparent new implementation designed
for general agentic work. It uses the same approximate compute budget as the
upstream claim but is not presented as the unpublished FINAL-Bench evaluator
or as a reproduction of the claimed 84.34% score.

## Primary sources

- Model repository: https://huggingface.co/FINAL-Bench/Darwin-9B-NEG
- Base checkpoint: https://huggingface.co/FINAL-Bench/Darwin-9B-Opus
- Qwen parent: https://huggingface.co/Qwen/Qwen3.5-9B
- Reasoning-distilled parent: https://huggingface.co/Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled
- Opus imatrix GGUF: https://huggingface.co/mradermacher/Darwin-9B-Opus-i1-GGUF
- NEG imatrix GGUF: https://huggingface.co/mradermacher/Darwin-9B-NEG-i1-GGUF
- Public demo Space: https://huggingface.co/spaces/FINAL-Bench/Darwin-9B-NEG
- Related V8 repository: https://huggingface.co/FINAL-Bench/Darwin-9B-NEG-x-Negentropy-V8
- First-party NEG failure analysis: https://huggingface.co/FINAL-Bench/Darwin-28B-NEG
- Linked paper: https://arxiv.org/abs/2605.14386
- Ollama native API schema: https://github.com/ollama/ollama/blob/main/docs/openapi.yaml
