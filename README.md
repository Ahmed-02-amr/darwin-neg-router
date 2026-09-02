# Darwin-9B-NEG native inference stack

This project serves the imatrix `Q6_K` Darwin-9B-NEG GGUF with its separately
released Darwin NEG hidden-state head attached inside a patched, Ollama-pinned
`llama-server`. A dual OpenAI Chat Completions and Anthropic Messages routing
layer adds selective candidate generation, task-aware voting, specialist
review, guarded refinement, and an optional exact 20-call profile for
CodePilot and Claude Code clients.

```text
CodePilot / Claude Code
  -> OpenAI + Anthropic router :11435
       -> darwin-neg-agent      (1 call normally; 5 for complex/uncertain steps)
       -> darwin-neg-agent20    (15 candidates + 3 reviews + evaluator + refiner)
       -> native llama-server :11436
            -> Q6_K language model on RTX 5070
            -> released FP32 NEG head on each generated hidden state
```

CodePilot remains responsible for repository tools, checkpoints, compaction,
and the long-horizon task loop. This stack owns inference-time NEG, uncertainty
routing, candidate evaluation, and response refinement.

## What the base model actually is

The upstream Darwin-9B-Opus lineage description and the released bytes tell
the same story:

```text
Qwen/Qwen3.5-9B
        +
Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled
        |
        +-- DARE-TIES / MRI-guided evolutionary merge --> Darwin-9B-Opus
                                                            |
                                      frozen-backbone NEG training
                                                            |
                                      Darwin-9B-NEG = same LM backbone
                                                      + separate NEG head/gate
```

According to that upstream description, Darwin-9B-Opus is a
Qwen3.5-9B-family merge, not Claude Opus weights
and not an Anthropic model runtime. One parent is Qwen's released 9B model; the
other is a Qwen3.5-9B fine-tune trained on reasoning trajectories attributed to
Claude Opus datasets. The `Opus` name describes that distilled training
influence. Upstream then froze the resulting Darwin-9B-Opus language model and
trained an approximately four-million-parameter entropy predictor and scalar
gate alongside it.

That distinction is not based only on model-card prose. The four BF16 language
shards in `Darwin-9B-Opus` and `Darwin-9B-NEG` have identical byte sizes and
identical Hugging Face LFS SHA-256 object IDs:

| Language shard | Bytes | Opus SHA-256 | NEG SHA-256 |
|---|---:|---|---|
| `model-00001-of-00004.safetensors` | 5,276,436,216 | `8bbd456f1367d1d9d7273b0a5735a57ce73f6a56f5a09c3b99a8a607a0a5f65a` | same |
| `model-00002-of-00004.safetensors` | 5,335,161,512 | `048129af3b6acde304c92fc262b12db11e25bc23a5187c42a60b0a6ee16749fb` | same |
| `model-00003-of-00004.safetensors` | 5,368,717,440 | `7283cf97c0bc17a351e1b08ba6b6f3d4c2920704a6d1b1cacfb6ae6510c45730` | same |
| `model-00004-of-00004.safetensors` | 3,325,988,568 | `53161974a653473c3829f77974fda95d7bbaabc62a9ee309925a663625fdf0ee` | same |

The NEG repository adds `neg_modules.safetensors` (16,785,908 bytes,
SHA-256 `8fcc1a5a9f7cdeaf2462af9f6de87ecf7626be8a96287e95bb2a20d63cbcb71a`).
It is a sidecar, not part of those language shards.

### Which Q6_K is this project serving?

The whole-file GGUF hashes do **not** match each other:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `mradermacher/Darwin-9B-Opus-i1-GGUF` `Darwin-9B-Opus.i1-Q6_K.gguf` | 7,359,260,992 | `6ab52e0e34b4c6fe9583e88a0b8e53e18b0e0f2d6652d033fc3508059286a851` |
| `mradermacher/Darwin-9B-NEG-i1-GGUF` `Darwin-9B-NEG.i1-Q6_K.gguf` | 7,359,260,576 | `3304a4913eec467e6775cba66f563199ef4b14e8a1976232e2ec82b7dcaa49bc` |
| local `models/gguf/Darwin-9B-NEG.i1-Q6_K.gguf` | 7,359,260,576 | `3304a4913eec467e6775cba66f563199ef4b14e8a1976232e2ec82b7dcaa49bc` |

The local file is thus a byte-for-byte match for the released NEG-labelled
imatrix Q6_K, not the Opus-labelled GGUF. The 416-byte container-size
difference and whole-file hashes do not prove different quantized language
tensors; GGUF metadata alone can change both. The identical upstream BF16
shards are the stronger evidence that the underlying language backbone is the
same. This project makes no stronger tensor-level claim without a full GGUF
tensor comparison.

### Base weights versus extended capability

| Layer | Runs where | What it adds | What it does not add |
|---|---|---|---|
| Darwin-9B-Opus backbone | patched `llama-server` | the Qwen-family language, reasoning, and coding behavior | the separate NEG head or an agent loop |
| Native NEG sidecar | once per generated token | predicted-entropy telemetry and gated top-k/temperature guidance for sampling | new factual knowledge, tools, or a changed greedy argmax |
| Darwin router | once or selectively 5/20 calls per request | task classification, diverse candidates, evidence scoring, reliability priors, specialist review, refinement, truncation recovery, and tool-loop guards | weight training or persistent project memory |
| CodePilot / Claude Code | client side | skills, MCP tools, repository actions, compaction, checkpoints, and the long-horizon control loop | native NEG token processing |

The router is general-purpose rather than a GPQA wrapper. Its exact, coding,
investigation, creative, and general policies change candidate roles,
temperatures, sampling widths, and small post-verifier reliability priors. Tool
schemas and tool results remain part of the original client contract. The
benchmark harness merely selects the `exact` policy so the same production
router can be measured under a fixed, auditable configuration.

## Windows controller and installer

`Darwin NEG Control` is the desktop start/stop application for the complete
stack. It launches the patched native runner, waits for the Q6_K model and
released NEG head to become healthy, then exposes the CodePilot-compatible
router. The live dashboard reports aggregate request/call/token counts, routed
request rate, real NEG activation rate, generated-token throughput, GPU memory,
GPU load/temperature, recent request metadata, and service logs. Prompts and
generated text are never retained in telemetry.

The controller records the exact PID, executable path, and Windows process
creation time for both managed services. If the UI is force-closed while the
stack remains alive, the next launch safely adopts only those matching Darwin
processes. A surviving half-stack appears as **Partial** with working **Stop**
and **Restart** actions; Restart first removes the verified orphan before
bringing both services back online. Unrelated processes on the configured ports
are never terminated automatically.

![Darwin NEG Control dashboard](docs/design/darwin-neg-control-implementation.png)

The **CodePilot setup** button opens an offline configuration guide inside the
app. It shows the correct Anthropic base URL (without `/v1`), the separate
OpenAI-compatible URL, API-key behavior, model IDs, context/output limits, and
the distinction between model serving and CodePilot's web-search MCP tools.

The installer and portable build bundle the patched native runtime and the
released 16.8 MB FP32 NEG head. The 6.85 GiB Q6_K GGUF remains an external model
file; select it once in the app and the path is saved under
`%LOCALAPPDATA%\DarwinNEGControl\config.json`. Ollama must be installed because
the native runner reuses its CUDA 12 backend.

Install the desktop build dependency and build the unsigned Windows artifacts
with:

```powershell
python -m pip install -e ".[desktop]"
& .\scripts\build-windows-release.ps1
```

This produces an x64 portable ZIP, a per-user Inno Setup installer, and SHA-256
checksums under `dist`. The binaries are intentionally unsigned, so Windows may
show a SmartScreen warning until a code-signing certificate is applied.

## What is genuinely native NEG here

The runner retrieves Qwen3.5's normalized final hidden state immediately before
the LM head for every generated token. It executes the released sidecar exactly:

1. `Linear(4096 -> 1024)`
2. exact GELU
3. `Linear(1024 -> 1)`
4. softplus predicted entropy
5. activate above the released threshold `1.1751873493`
6. on activation, scale logits by `1 / 0.5983633399` and retain the top 20

The API returns per-request `neg` telemetry: steps, activations, activation
rate, mean/max predicted entropy, guided steps, gate parameters, and head time.
The router uses activation rate to decide when extra inference is warranted.

One upstream limitation is preserved rather than hidden: top-20 masking and a
positive temperature scale cannot change a greedy argmax. Consequently,
token-level NEG changes sampled ensemble candidates, while greedy calls benefit
through real-head telemetry and request-level routing. This is materially
different from the former top-logprob entropy surrogate.

The language model is the imatrix `Q6_K` GGUF and remains fully GPU-resident on
the 12 GB RTX 5070. The 16.8 MB NEG head is FP32 and runs as a vectorized CPU
side-head; it does not cause language-model CPU offload. Stock Ollama cannot
attach an external head or custom per-token processor, so the runnable model is
a patched companion server built against Ollama's exact llama.cpp revision,
while reusing Ollama's installed CUDA backend.

## Pinned components

- Ollama source: `b7871fc0d1d82fe109536efa3e0e8e411c766c75` (v0.32.15)
- llama.cpp source: `9d77fa17254e1dee4b9e92504c91611a60b1359f` (b10488)
- Native patch: `native/llama-b10488-neg.patch`
- Released NEG sidecar SHA-256: `8fcc1a5a9f7cdeaf2462af9f6de87ecf7626be8a96287e95bb2a20d63cbcb71a`
- NEG binary SHA-256: `c2ca7f61897ab3071afd022bc3bf7c0efa84c25b080d727bf4be4e2a36ff1e2a`
- Current runner SHA-256: `411fe728880087baad30d0e40e2ed0ad6c1e828f697be7428880d21378a8cf57`

The build script checks both source commits before applying the patch. The
native head's deterministic output matches PyTorch within `2.4e-7` absolute
error; the optimized build was also checked separately.

## Build

The required model and released sidecar currently live at:

```text
models/gguf/Darwin-9B-NEG.i1-Q6_K.gguf
models/Darwin-9B-NEG-BF16/neg_modules.safetensors
```

Regenerate the native sidecar when needed:

```powershell
python .\scripts\convert-neg-head.py `
  .\models\Darwin-9B-NEG-BF16\neg_modules.safetensors `
  .\models\neg-head.fp32.bin
```

Build the pinned runner and package its DLLs under `runtime/native-neg`:

```powershell
& .\scripts\build-native-neg.ps1
```

The script uses a short build path under `%LOCALAPPDATA%\DarwinNEG` to avoid
Windows path-length failures. It requires Git, CMake, a C++ compiler, and
Ollama 0.32.15's installed CUDA 12 backend.

## Start the CodePilot stack

Start both layers with one command:

```powershell
& .\scripts\start-codepilot-stack.ps1
```

The native server binds only to `127.0.0.1:11436`; the router binds to
`127.0.0.1:11435`. The combined script runs the native helper hidden, keeps the
router in the foreground, and stops the helper when the foreground process
exits. To run them separately:

```powershell
& .\scripts\start-native-neg.ps1
& .\scripts\start-native-router.ps1
```

Health and model discovery:

```powershell
Invoke-RestMethod http://127.0.0.1:11436/health
Invoke-RestMethod http://127.0.0.1:11435/health
Invoke-RestMethod http://127.0.0.1:11435/v1/models
```

## CodePilot configuration

For Claude Code runtime, Anthropic third-party provider, and maintained web
search setup, follow [CODEPILOT-CLAUDE-SETUP.md](CODEPILOT-CLAUDE-SETUP.md).

The Anthropic base URL is `http://127.0.0.1:11435` and accepts
`POST /v1/messages` plus `POST /v1/messages/count_tokens`. It translates system
blocks, thinking blocks, tool schemas, `tool_use`, `tool_result`, tool choice,
parallel-tool control, stop sequences, and Anthropic SSE events onto the same
Darwin NEG router used by the OpenAI endpoint. `x-api-key` and Bearer
authentication are both accepted when `DARWIN_API_KEY` is configured.

### CodePilot native/OpenAI runtime

In CodePilot, open **Settings -> Providers -> Add Provider -> Custom API
(OpenAI-compatible)** and use:

| Field | Automatic profile | Full profile |
|---|---|---|
| Base URL | `http://127.0.0.1:11435/v1` | same |
| API key | `EMPTY` | same |
| Model Name | `darwin-neg-agent` | `darwin-neg-agent20` |
| Context | `163840` | `163840` |
| Output allowance | `43008` | task appropriate, up to `43008` by default |
| Temperature | `0` | candidates are routed internally |

`darwin-neg-agent` preserves CodePilot `tools`, `tool_choice`,
`parallel_tool_calls`, stop sequences, seeds, presence/frequency penalties, and
the llama.cpp `repeat_penalty` extension. Thinking stays enabled by default.
Streaming requests are accepted; because routing needs complete candidates,
the final result is emitted as a buffered SSE chunk. `Darwin-NEG` is also
advertised as a compatibility alias for clients that preserve a display-name
model ID. Streamed tool calls include the required numeric `index` field used by
CodePilot's OpenAI event validator.

### Tool-loop safety without reducing long-form output

The gateway keeps the configured 42K maximum for reasoning, code, and final
answers. Internal specialist reviews and evaluator verdicts receive a 3K
allowance so thinking can complete before the required structured verdict;
override it with `DARWIN_REVIEW_MAX_TOKENS`. A tool-enabled request first
receives a 4K action-selection budget; if
it reaches that boundary without a tool call, the request is automatically
retried with the complete output allowance. This prevents runaway tool syntax
from consuming the entire context while preserving long-form generation.

If any ordinary generation still reaches its output boundary without a tool
call, the router performs one deterministic continuation with a 2K budget. It
receives a bounded tail of the interrupted draft and must immediately emit the
pending tool action or concise final response without restarting its analysis.
Visible partial output is retained, usage is combined, and recovery status is
recorded in telemetry. Configure the allowance with
`DARWIN_TRUNCATION_RECOVERY_TOKENS`; recovery never repeats recursively. The
continuation remains part of the same candidate rather than becoming another
ensemble vote, and it sees no other candidates. If its saved state is
insufficient, it abstains and the verifier selects another candidate. Recovery
inferences are reported separately from the requested ensemble budget as well
as in the actual inference-call total.

Within one response, exact duplicate actions are collapsed by canonical
function name and JSON arguments. Distinct parallel actions remain ordered and
available—up to 32 by default—or one when the caller explicitly disables
parallel tools. Across turns, an action is considered stalled only after the
same function and arguments have produced the same result twice since the last
real user message. Changing polling results, deliberate retries after a new
user instruction, different arguments, and other tools remain available. A
stalled action gets one recovery inference that must use existing evidence,
answer, or choose a materially different action.

The limits are configurable with `DARWIN_TOOL_PHASE_MAX_TOKENS`,
`DARWIN_MAX_PARALLEL_TOOL_CALLS`, and
`DARWIN_UNCHANGED_TOOL_RESULT_LIMIT`. Aggregate and per-request guard activity
is exposed through `/telemetry` and in the controller's request table.

## Routing and refinement

The automatic profile begins with one deterministic response. It expands to
three role-diverse candidates plus one evaluator and one final refiner when any
of these are true:

- a new repository task matches at least three complexity concerns;
- released-head activation rate is at least 5% across at least 16 active steps;
- the answer expresses explicit uncertainty;
- the proposed tool is high-impact; or
- the caller sets `ensemble` explicitly.

### Adaptive voter policy

Before generating alternatives, the general router assigns an explainable task
profile without spending another model call:

| Profile | Candidate temperatures | Selection emphasis |
|---|---|---|
| `exact` | greedy through low | recomputation, definitions, units, bounds, exact format |
| `coding` | greedy through moderate | repository evidence, valid tools, compatibility, tests |
| `investigation` | low through diverse | information gain, falsifiable hypotheses, source quality |
| `creative` | moderate through high | constraints, originality, audience fit, feasibility |
| `general` | low through moderately diverse | correctness, instruction compliance, completeness |

Every profile also supplies relevant candidate roles rather than applying
software-engineering roles to math, research, or creative work. The evaluator
sees candidate content and tool actions but is deliberately blinded to sampling
temperature. It returns a 0–100 evidence score for every available candidate.
Only after that independent judgment does the router add a small, profile-specific
reliability prior. The maximum prior is 2.5 points, so it can resolve close calls
but cannot rescue a candidate with materially weaker evidence. Duplicate answers
gain no majority advantage because selection is score-based, and failed
truncation recoveries remain unavailable.

The response's `darwin.routing` metadata includes the detected profile and
signals, candidate roles, temperatures, sampling widths, blinded evidence
scores, priors, adjusted scores, and whether adaptive weighting changed the
verifier's categorical winner. `/telemetry` aggregates task-profile counts and
adaptive selection changes without retaining prompts or response text.

For controlled evaluation, an OpenAI-compatible request may set
`{"darwin":{"routing_profile":"exact"}}`. Anthropic Messages requests may set
`{"metadata":{"darwin_routing_profile":"exact"}}`. Valid overrides are
`exact`, `coding`, `investigation`, `creative`, and `general`; normal CodePilot
traffic uses automatic detection.

Valid selected tool calls are immutable across refinement unless the refiner
emits the same function names and parsed arguments. This prevents an evaluator
from turning a correct tool action into prose. The 20-call profile spends
exactly 15 candidate calls, three specialist reviews, one evaluator, and one
refiner. Use it for architecture, difficult debugging, migration plans, or
final verification—not every tool round.

For a custom one-off budget:

```json
{
  "model": "darwin-neg-agent",
  "messages": [{"role": "user", "content": "Investigate this failure."}],
  "ensemble": 8,
  "max_tokens": 43008
}
```

### Long-context GPU profile

The native launch paths allocate one `163840`-token slot, use `q8_0` for both
K and V cache tensors, and force Flash Attention on. This keeps the Q6_K model
fully GPU-resident on a 12 GB card while reserving up to `43008` tokens for
thinking and output. The context size is the total sequence limit, so a request
that reserves the full output allowance can retain at most `120832` prompt
tokens. Q8 applies only to the temporary KV cache; model weights remain Q6_K.
The single-slot configuration is intentional because additional parallel slots
divide or multiply the available context-memory budget.

The served GPQA profile uses a separate `6144`-token allowance for every
solver, critic, juror, and arbiter call. This is large enough to reduce
thinking-related answer truncation without allowing each member of a 20-call
ensemble to consume the general router's full 42K generation budget. Override
these independently with `DARWIN_GPQA_SOLVER_TOKENS` and
`DARWIN_GPQA_REVIEW_TOKENS`. When a member still reaches the limit after stating
an explicit conclusion but before printing the strict `FINAL: X` suffix, the
router recovers that conclusion locally. This does not spend or hide a 21st
ensemble inference.

The separate GPQA reconstruction uses choice-order permutations, critics, and
arbiters. Its adaptive 3–1 path now uses guarded consensus: disagreeing late
reviewers cannot erase three independent permutation votes, while two reviewers
that agree on one alternative may resolve a tied reviewed panel.

## Validation evidence

Current local results on Ryzen 5 7600X + RTX 5070 12 GB:

| Check | Result |
|---|---|
| Native/PyTorch head parity | absolute error <= `2.4e-7` |
| Unit/integration tests | `63 passed` |
| Duplicate tool burst regression | `780` identical calls collapsed to `1` |
| Complex parallel tool validation | `3/3` distinct live calls preserved; `24/24` synthetic |
| Stalled-result live recovery | third identical `403` fetch blocked; recovery completed in `2` calls |
| Executable coding micro-suite | `3/3` |
| Forced exact-schema tool calls | `3/3` |
| Native-head throughput case | `74.96 tok/s` (512 generated tokens) |
| Controlled head/no-head comparison | `69.43` vs `72.51 tok/s` (~4.3% cost) |
| Side-head compute, controlled run | `0.633 ms/token` |
| Live automatic refinement | 5 calls completed in `72.68 s` at a deliberately small 1K cap |
| GPQA guarded-evaluator regression | `1/1`, six calls, `110.1 s` |

The coding/tool/performance record is in
`benchmarks/results/native-agent-validation.json`; the GPQA record is in
`benchmarks/results/native-neg-smoke-adaptive20-guarded.jsonl`.

The GPQA result is a one-question regression check, not an estimate of
GPQA-Diamond accuracy and not evidence for the upstream 84.34% claim. A full
198-question adaptive or fixed-20 run is required before making any aggregate
claim. The exact FINAL-Bench evaluator remains unpublished; this project's
protocol is intentionally transparent and versioned.

Run all local tests and the validation suite with:

```powershell
python -m pytest -q
python .\benchmarks\native_agent_validation.py `
  --output .\benchmarks\results\native-agent-validation.json

python .\benchmarks\gpqa_reconstruction.py `
  --backend router `
  --model darwin-neg-gpqa `
  --mode adaptive20 `
  --limit 1 `
  --output .\benchmarks\results\router-gpqa-smoke.jsonl
```

The public GPQA dataset is pinned to revision
`68be7564497676e07a77a042fdb587deb88c51c3` and all 198 rows validate. Reference
answers are read only after inference for scoring and are never placed in
solver, critic, juror, or arbiter prompts.

### Pause and resume a full GPQA run

Each completed item is flushed as one JSONL record. The general-router launcher
always passes `--resume`, so it skips indices already present in the output:

```powershell
& .\scripts\run-gpqa-general-router20.ps1
```

If a worker finishes the next item while shutting down, reset the active file
to a precise boundary without losing that record:

```powershell
& .\scripts\checkpoint-gpqa.ps1 `
  -Results .\benchmarks\results\gpqa-diamond-v4-general-router20-adaptive-voting-full.jsonl `
  -ResumeAt 70
```

Records from Q70 onward are moved to a timestamped local checkpoint, then the
active JSONL is atomically replaced. Raw full-run outputs and checkpoints are
ignored by Git because they include dataset text and model reasoning. The
current local handoff is summarized in
[`BENCHMARK-STATUS.md`](BENCHMARK-STATUS.md).

## Upstream audit

`UPSTREAM-AUDIT.md` records the released checkpoint layout, exact provenance
hashes, and the limitations
of the upstream model/evaluator claims. In particular, the public language
checkpoint and GGUF do not embed the NEG side-head, the upstream greedy top-k
gate cannot change argmax, and the exact claimed 84.34% evaluation procedure is
not included. This implementation attaches the head that is actually released
and reports only locally reproducible evidence.
