# HLD — The Trajectory Refinery

Status: **draft for review** · Diagram: [`diagrams/refinery-hld.png`](./diagrams/refinery-hld.png) ·
Editable source: `diagrams/refinery-hld.excalidraw`

## 1. Purpose and scope

Build a production-grade **agent-data flywheel**: take verifiable tasks, run a frontier-model
agent over them many times, keep only trajectories that survive an independent verifier, compile
them into an SFT dataset, train a LoRA on an open 8B model, and evaluate that model against the
teacher and a mid-tier API model on cost/quality frontier.

**In scope (v1):** Python repos + pytest-executable tasks only. Single teacher model for
generation. LoRA (not full FT). One GPU. Local vLLM serving for eval.

**Out of scope (v1):** non-executable tasks (no LLM-judged correctness), multi-turn live-API
tasks, RL/GRPO, full-parameter fine-tuning, distributed training.

**Design stance:** the *pipeline* is the contribution. The verifier gate, the anti-hack filters,
the deterministic compiler, and the cost accounting are the deliverable. The LoRA is the proof
that the pipeline works.

## 2. System context

| Dependency | Role | Failure mode if unavailable |
|---|---|---|
| SWE-Gym task set | task pool seed | blocks everything; fallback is SWE-smith-only synth |
| SWE-smith | task synthesis at scale | pool too small for eval split power |
| Frontier API model (teacher) | generates trajectories | farm cost explodes; degrade to mid-tier teacher, log it |
| Repo container registry / base images | fresh verification env | verification runs in dirty tree → entire project invalid |
| Modal / RunPod GPU | LoRA training + vLLM eval | training blocked; can still ship dataset + verifier |
| HF Hub | dataset + model distribution | no artifact |

Nothing in the loop requires a GPU except the trainer and the eval server. Stages 1–4 are
CPU-only and cheap — that is deliberate, so a failed GPU rental does not lose the expensive
stage (the farm).

## 3. Component architecture

Six subsystems, each independently runnable and resumable. Every stage reads from and writes to
an append-only JSONL store keyed by content hash — no database, no state server, no coordination.

| # | Component | Interface | Determinism requirement |
|---|---|---|---|
| 1 | `taskpool` | `build()` → task manifest; `split()` → train/eval by hash wall | fully deterministic given seed + repo corpus digest |
| 2 | `farm` | `run(tasks, k, teacher, caps)` → run records | non-deterministic by design (temp 0.8); records everything |
| 3 | `verifier` | `verify(run) → verdict` | deterministic; never reads the agent's working tree |
| 4 | `compiler` | `compile(verdicts) → SFT samples`; `render(trajectory) → token sequence` | **bit-exact** re-render (trap #2) |
| 5 | `trainer` | `train(dataset, lora_cfg)` → adapter | seeded; single GPU |
| 6 | `evalfarm` | `eval(models, harness, tasks)` → frontier | same harness + same prompts for all contenders |

### 3.1 Task pool

- Sources: SWE-Gym tasks; SWE-smith-style synthetic tasks (mutate a real function → bug; keep
  tasks whose reference patch flips fail→pass).
- **Identity:** `task_id = sha256(repo_url @ base_commit @ test_spec @ prompt_text)`. All task
  identity is content-derived, never sequential — that is what makes the hash wall possible.
- **Splits:** partition on `sha256(task_id + split_salt)` mod 100 → 0–79 train, 80–99 eval.
  Because the key is the content hash, *any* task that collides across splits is impossible by
  construction, including near-duplicate forks of the same upstream PR.
- Additional guard: **base-model zero-shot pass rate on eval**. If the untrained base model
  solves more than ~10% of the eval set, assume pretraining contamination and rebuild the eval
  split (raise the salt, regenerate).
- Store: `tasks/manifest.jsonl` — one record per task with `repo_url`, `base_commit`,
  `env_image`, `install_cmds`, `test_cmd`, `FAIL_TO_PASS`, `PASS_TO_PASS`, `task_id`, `split`.

### 3.2 Generation farm

- k = 8 samples/task, temp 0.8, one teacher model. **Hard caps per run: 30 steps, $0.40.**
- Runs are embarrassingly parallel; shard by `task_id` into N worker containers. Each worker
  writes its own JSONL shard — no shared writer, no coordination.
- **Record everything** per run: full ordered message list (system/user/assistant/tool), every
  tool call with arguments and raw result, the final diff, raw test output, per-step token
  counts, cumulative cost, wall time, exit reason (`solved`, `step_cap`, `dollar_cap`,
  `crash`, `timeout`).
- Cost controls: system-prompt prefix caching; easy-task tier on a cheap mid-tier model first,
  escalation to the teacher only on failure (this is the ~60% cut — measure it, don't assume).
- Budget control: a global spend ledger with a hard daily ceiling; workers poll it before
  launching a run. The farm must never be able to surprise you on the invoice.

### 3.3 Verifier gate — the trust boundary

Success requires **all** of:

1. **Fresh container.** Rebuild from the task's base image + task's base commit, apply the
   agent's final diff, install, run `test_cmd`. Never reuse the agent's dirty working tree.
2. **All tests pass** (`FAIL_TO_PASS` and `PASS_TO_PASS`).
3. **Diff-path allowlist.** No changes under test paths, CI config, or dependency manifests that
   exist solely to alter test behavior.
4. **Non-trivial diff.** ≥ 1 changed functional line in a source path; reject pure renames,
   whitespace, and comment-only changes.
5. **No state destruction.** Reject if the diff (or the agent's shell history) contains
   `git checkout/reset/clean --hard`, `git stash`, test deletion, or `pytest.skip` /
   `@pytest.mark.skip` additions.
6. **No reward-hack residue.** Reject diffs that write to stdout only (print-the-answer),
   monkeypatch the test runner, or forge exit codes.

Every rejection is logged with a **reason code** — the histogram is a publishable artifact.

```
VERIFIED_OK | TEST_FAIL | PATH_VIOLATION | TRIVIAL_DIFF | STATE_DESTRUCTION
SKIP_INJECTED | EXIT_FORGERY | FRESH_ENV_ERROR | AGENT_CRASH | DOLLAR_CAP | STEP_CAP
```

`FRESH_ENV_ERROR` is separated from `TEST_FAIL` on purpose: infrastructure noise must not be
reported as agent failure, and it must not pollute the training set.

### 3.4 Dataset compiler — the make-or-break component

Two decisions carry this project.

**(a) Format fidelity.** The training text must be *exactly* what the harness produces at
inference time: same system prompt string, same tool schemas, same message envelope, same
serialization. Therefore the compiler never renders from ad-hoc string building — it imports the
harness's own prompt renderer and calls it. `render(trajectory) → tokens` is deterministic and
idempotent, and the repo contains a test that decodes one training sample and diffs it against a
live inference prompt **byte-for-byte**. This is trap #2 and it is ~60% of the failure probability.

**(b) Lossless windowing.** Many trajectories exceed the 32k window. The policy:

- keep the first plan and *all* tool arguments verbatim, always;
- collapse *old* tool outputs (older than the active window) to a structured digest:
  `{files_touched[], tests_run[], exit_code, lines_changed, stderr_head}`;
- never truncate the final 3 steps — the fix window is what is being taught;
- record `windowing_strategy` per sample so the ablation can slice on it.

Ablation axis: `full_context` | `windowed` (this policy) | `naive_truncated` (last-N tokens).
Expected result: naive truncation tanks success because the plan is gone; windowed holds.

**Output format:** mirror the JetBrains agent-trajectories HF schema so the dataset is drop-in
usable, with loss masks marking assistant and tool-call tokens.

### 3.5 Trainer

- LoRA r = 16–32, alpha = 2r, dropout 0.05, all linear projections; bf16; single GPU.
- Sequences packed with loss masked to assistant/tool-call spans only (never the tool results, or
  the model learns to predict environment output).
- **Curriculum:** stage 1 = trajectories ≤ 8 steps, tool-call-only supervision; stage 2 = mixed,
  full-length. Small models frequently cannot emit syntactically valid tool calls at all without
  this; report the ablation (curriculum vs mixed-from-scratch).
- 3–5k samples is enough to show transfer; 10k+ if the farm cooperates.
- Artifacts: adapter, loss curves, training log, exact git commit of the dataset compiler.

### 3.6 Eval + frontier

- **Identical harness, identical prompts, identical caps** for all three contenders — otherwise
  the frontier is meaningless.
- Contenders: (a) distilled model served locally (vLLM on rented GPU), (b) frontier API teacher,
  (c) mid-tier API model.
- Metrics: pass@1, pass@k (for completeness), **cost per resolved task** (API $ + amortized GPU
  rental ÷ resolved tasks), p50/p95 latency, steps-to-success, and **valid-tool-call rate**
  (the metric that exposes the "can't emit tool calls at all" failure mode).
- Headline: *"distilled 8B resolves X% of what the frontier model does at Y% of the cost."*

## 4. Data model (core records)

```jsonc
// Task (tasks/manifest.jsonl)
{"task_id":"sha256:…","repo_url":"…","base_commit":"…","env_image":"…",
 "install_cmds":["…"],"test_cmd":"…","fail_to_pass":["…"],"pass_to_pass":["…"],
 "prompt":"…","origin":"swe-gym|swe-smith","split":"train|eval"}

// Run (runs/shard-XX.jsonl)
{"run_id":"…","task_id":"…","sample_idx":3,"seed":null,"teacher":"…",
 "started_at":"…","wall_s":412,"steps":27,"cost_usd":0.31,"exit":"step_cap",
 "messages":[…],"tool_calls":[…],"diffs":[{"path":"…","patch":"…"}],
 "token_usage":{"in":0,"out":0},"trajectory_ref":"sha256:…","storage":"cold|hot"}

// Verdict (verdicts.jsonl)
{"run_id":"…","verdict":"VERIFIED_OK","reason_code":null,
 "checks":{"fresh_env":true,"tests_pass":true,"paths_ok":true,
           "nontrivial":true,"no_state_destruction":true,"no_hack":true},
 "tests":{"f2p_pass":4,"f2p_total":4,"p2p_pass":118,"p2p_total":118},
 "rejected_at":"verifier","wall_s":96,"cost_usd":0.0}

// Training sample (dataset/train.jsonl, HF agent-trajectory schema)
{"messages":[…],"tools":[…],"loss_mask":[…],
 "meta":{"task_id":"…","run_id":"…","origin":"…","n_steps":27,
         "windowing":"windowed","prompt_hash":"sha256:…","compiler_version":"…"}}

// Eval result (eval/results.jsonl)
{"contender":"distilled-8b|r1|tier-B","model":"…","served":"vllm:0.6",
 "task_id":"…","passed":true,"cost_usd":0.03,"gpu_amortized_usd":0.004,
 "latency_s":83,"steps":14,"valid_tool_calls":true}
```

The `prompt_hash` on every training sample is the join key that makes trap #2 detectable in the
field: at eval time, hash the live prompt and compare distributions. Any drift shows up as a
hash mismatch before it shows up as a quality regression.

## 5. Storage and scale

| Store | Volume (2,400 runs) | Notes |
|---|---|---|
| `tasks/manifest.jsonl` | ~5 MB | repo + test specs |
| container images / repo clones | 20–60 GB | cache by `repo@commit`, prune to eval + last N train |
| `runs/*.jsonl` (full traces) | 3–10 GB | hot for 7 days, then content-addressed into cold storage |
| `verdicts.jsonl` | < 5 MB | the histogram |
| `dataset/train.jsonl` | 200 MB – 1 GB | packed SFT samples |
| adapter + eval logs | ~1 GB | |
| Project 2 snapshots (deltas) | +2–5 GB | content-addressed dedup; full snapshots are a trap |

Storage discipline: traces are the expensive artifact and the least compressible. Reject early
(no point keeping a `DOLLAR_CAP` run's full trace beyond the summary), and content-address them
so identical tool outputs are stored once.

## 6. Cost model (the bill, pre-registered)

| Line item | Assumption | Estimate |
|---|---|---|
| Farm — 2,400 runs | $0.10–0.30/run, avg ~$0.19 | **$250–700** |
| Farm — with tiered routing | ~40% of tasks resolved by cheap model | −~60% → $100–280 |
| Verifier | CPU containers, ~2 min/run | $20–60 |
| LoRA training | 3–8 h @ $1–3/hr | $3–24 |
| Eval — 3 contenders × 150 held-out tasks × 3 seeds | frontier $ dominates | $60–200 |
| Eval — vLLM GPU amortization | rented, amortized over resolved tasks | $10–40 |
| **Total (tiered routing)** | | **$200–600** |

The bill goes in the README, itemized, with the actual numbers replacing the estimates.

## 7. Failure modes and mitigations

| # | Failure | Detection | Mitigation |
|---|---|---|---|
| 1 | Contamination | base model zero-shot on eval scores implausibly high | hash wall + re-salting; report base zero-shot as a table row |
| 2 | Format drift | `prompt_hash` mismatch / byte-diff test fails | single renderer imported by both harness and compiler; CI test |
| 3 | Reward hacks | anti-hack checks + rejection histogram | publish the histogram; treat `SKIP_INJECTED` / `EXIT_FORGERY` as headline numbers |
| 4 | Farm cost bleed | daily spend ledger + per-run caps | hard ceiling in code; workers poll before launching |
| 5 | No valid tool calls | `valid_tool_call_rate` in eval | curriculum: short tool-call-only stage first |
| 6 | "So you SFT'd a model" | — | README leads with pipeline numbers table; LoRA is the footnote |
| 7 | Replay/nondeterminism (P2 handoff) | variance across seeds | n=3 replays; prefer local vLLM for deterministic conditions |
| 8 | Snapshots explode (P2 handoff) | storage growth | content-addressed deltas, never full-context-per-step |

## 8. Deployment topology

| Where | What | Why |
|---|---|---|
| Phone (Termux/proot) | editing, `gsync up`, spec/doc work | the only always-on machine |
| Local CPU (proot) | task pool, compiler, verifier orchestration, dataset stats | CPU-only, cheap |
| Container farm (Modal or RunPod CPU) | generation workers + fresh verification containers | needs real containers and isolation |
| Modal/RunPod GPU | LoRA training, vLLM eval server | ephemeral, paid by the hour |
| Colab/Kaggle | ablation sweeps where a notebook is faster than a job | zero setup |

Everything is driven by shell scripts and JSONL on object storage, so any stage can be re-run
from a cold clone on a different backend.

## 9. Milestones

| Week | Milestone | Exit criterion |
|---|---|---|
| 1 | Task pool + one end-to-end smoke run | 1 task, 8 samples, 1 verified trajectory, dataset row rendered |
| 2 | Farm at scale + verifier hardened | 2,400 runs, histogram printed, hacking attempts logged |
| 3 | Dataset compiler + trainer | 5k samples, adapter trained, loss curves sane, curriculum ablation run |
| 4 | Eval + frontier + write-up | three contenders, frontier plot, README first screen complete |

**Order matters:** never spend on stage 2 before stage 1 produces one verified trajectory. One
verified trajectory end-to-end de-risks every subsequent week.

## 10. Repo layout

```
refinery/
  taskpool/     manifest build, hash-wall split, decontamination checks
  farm/         agent harness, caps, spend ledger, JSONL writers
  verifier/     fresh-container runner, anti-hack checks, reason codes
  compiler/     renderer (imported from harness), windowing, packing, masks
  trainer/      LoRA config, curriculum, launch scripts
  common/       schemas, hashing, storage paths
eval/           harness, contenders, metrics, frontier plot
docs/           this HLD, ablations, bill
diagrams/       refinery-hld.excalidraw + .png
```
