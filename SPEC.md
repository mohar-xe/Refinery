# SPEC — The Trajectory Refinery (flagship)

> This is the flagship's spec, inlined so this repo is self-contained.
> The cross-project portfolio spec (all six projects, publishing ops, interlock story) lives in
> `/root/Projects/portfolio/SPEC.md`.

## Reading stack

| # | Read | Take from it |
|---|---|---|
| 1 | **STaR** (Zelikman et al.) | the ancestor: bootstrap reasoning from self-generated *verified* solutions. This project is STaR applied to agentic tool use. |
| 2 | **Rejection-sampling FT (Llama 2/3)** | the standard recipe: sample k, keep verified-correct, SFT. Here: RFT for agent trajectories. |
| 3 | **SWE-Gym** (arXiv 2412.21139) | the environment: real repos + executable tasks + verifiable tests. Full-scale task pool. |
| 4 | **SWE-smith** (2025) | how to synthesize tasks at scale from repos. Full-scale expansion strategy + contamination story. |
| 5 | **AgentTuning / FireAct** | earlier SFT-for-agents work; why they underperformed (synthetic tasks, no verification). |
| 6 | **JetBrains agent-trajectories (HF)** | the established trajectory-SFT schema; mirror it so the dataset is drop-in usable. |
| 7 | **Unsloth + TRL** | LoRA SFT on one GPU; Qwen3-8B / Llama-3.1-8B as full-scale base. |

---

# PROJECT 1: The Trajectory Refinery (flagship)

## The core loop

```
verifiable tasks → agent runs (k samples/task) → outcome verification →
filtered trajectories → SFT dataset → LoRA on open 7-8B model →
same eval harness → cost/quality frontier vs frontier API
```

## The reading stack (in order, with what to take from each)

1. **STaR** (Zelikman et al.) — the intellectual ancestor: bootstrap reasoning from
   self-generated, verified solutions. This project is STaR applied to agentic tool use.
   Name-drop it correctly in interviews.
2. **Rejection-sampling finetuning as used in Llama 2/3** — the industry-standard
   filtering recipe: sample *k*, keep verified-correct, SFT. Here: RFT for agent
   trajectories.
3. **SWE-Gym** (arXiv 2412.21139) — the environment: real Python repos + executable
   tasks + verifiable tests. Task-pool source.
4. **SWE-smith** (2025) — how to *synthesize* tasks at scale from repos when the public
   pool is too small. Expansion strategy; directly relevant to the contamination story.
5. **AgentTuning / FireAct** — earlier work on SFT for agents. Know why they
   underperformed (synthetic tasks, no verification) and why verified outcome filtering
   is the upgrade.
6. **JetBrains agent-trajectories on HF** — the established data schema for trajectory
   SFT. Mirror its format so the dataset is drop-in usable.
7. **Unsloth + TRL docs** — LoRA SFT on a single GPU; Qwen3-8B or Llama-3.1-8B-Instruct
   as base.

## Architecture — six subsystems

**1. Task pool (week 1).** Start from SWE-Gym's task set (~1–2k executable tasks). Add
self-generated tasks via the SWE-smith recipe: take repos, mutate real functions to
inject bugs, require a failing-test→passing-test diff. Pin every task by content hash.
Split train/eval with a hard hash wall (trap #1).

**2. Generation farm (week 1–2).** The agent harness (own harness or OpenHands for
speed) runs each task *k = 8* times at temp 0.8 with one frontier model. Record
**everything** per run as JSONL: full message list, every tool call + result, every diff,
test output, token counts, wall time. Budget-cap each run (e.g. 30 steps / $0.40) — the
farm is where costs explode without caps. Target scale: 300 tasks × 8 samples = 2,400
runs; at ~$0.10–0.30/run that is a **$250–700** exercise. Cut ~60% by starting with cheap
models for easy tasks. This spend is legitimate — put the exact bill in the write-up;
founders *love* seeing cost engineering.

**3. Verifier gate (week 2).** Success = all unit tests pass on the patched repo **plus
anti-hack filters**:

- diff must not touch test files (diff-path allowlist)
- diff must be non-trivial (≥ 1 functional line changed in source)
- the agent must not have `git checkout`-ed away state or commented out tests
- test run happens in a fresh container from the task's base image (never in the agent's
  dirty working tree)

Log every rejection with its reason — the rejection histogram is itself a publishable
artifact ("14% of 'passing' trajectories were reward hacks").

**4. Dataset compiler (week 2–3).** The make-or-break component. Two decisions separate
the 0.1% from the slop here:

- **Format fidelity** — train on *exactly* the token sequence the harness produces at
  inference: same system prompt, same tool schemas, same message structure. A single
  mismatch between training format and inference format destroys transfer. The compiler
  must re-render any stored trajectory into the training format deterministically.
- **Context windowing** — trajectories exceed 32k tokens; compress losslessly:
  (a) keep the full first plan + all tool *arguments* verbatim,
  (b) collapse old tool *outputs* to structured digests (file list, test names, exit codes),
  (c) never truncate the final 3 steps (the fix window is what is being taught).

  Windowing ablation is a section of the write-up: full-context vs windowed vs
  naive-truncated — show naive truncation tanks success and windowing holds it.

Include failure trajectories as contrastive data (DPO pairs: winning vs losing branch from
the same task) if time allows — stretch goal, not core.

**5. Trainer (week 3).** LoRA (r = 16–32, all linear layers) on Qwen3-8B / Llama-3.1-8B,
bf16, 1 GPU (Modal/RunPod, ~$1–3/hr, 3–8 hrs). Pack sequences, mask loss to
assistant/tool-call tokens. 3–5k trajectory samples is enough to show transfer; 10k+ if
the farm cooperates. Keep a training log (loss curves, W&B screenshots) — data-engineering
credibility lives in these details.

**6. Eval + frontier (week 4).** Identical harness, held-out tasks, three contenders:
distilled model (local via vLLM/Ollama), a frontier API model, a mid-tier API model. Report
the frontier: pass@1, cost per resolved task (amortize GPU rental), p50/p95 latency,
steps-to-success. The headline number writes itself: *"distilled 8B resolves X% of what
the frontier model does at Y% of the cost."* Even X = 55–70% at Y = 5–10% is a strong
result — publish whatever is honest.

## Trap list (where this project dies)

1. **Contamination leakage** — eval tasks appear in train (or in base-model
   pretraining). Fix: hash wall between splits + run the base model zero-shot on the eval
   set; a suspiciously high score on "unseen" tasks means leakage — regenerate.
2. **Format drift** — 60% of failure probability. Fix: deterministic re-renderer in the
   compiler; verify by decoding one training sample and diffing against a live inference
   prompt byte-for-byte.
3. **Reward hacks inflating pass rate** — the test-editing class above. The anti-hack
   filters *are* a result; report them.
4. **The farm bleeds money** — cap steps AND dollars per run; run easy-task tiers on
   cheap models; cache the system-prompt prefix.
5. **Small model can't emit valid tool calls at all** — curriculum: train first on short
   trajectories (≤ 8 steps, tool-call-only samples), then full. Report the curriculum
   ablation.
6. **"So you SFT'd a model" dismissal** — preempt in the README: the contribution is the
   *pipeline* (verification gate, anti-hack filters, windowing, cost accounting), not the
   LoRA. Numbers table up top.

## Artifact packaging

- Repo: `refinery/` (farm, verifier, compiler, trainer) + `eval/` + a README whose first
  screen is the numbers table and the frontier plot.
- HF: dataset card (filtering criteria, rejection histogram, decontamination statement) +
  model card.
- The post: "I ran the agent-data flywheel end to end for $[bill]" — sections: the loop,
  the farm economics, what verification rejected, windowing ablation, the frontier, what
  broke, what I'd do at 100x scale. That last section is the interview for the next job.

**Outreach mapping:** data/post-training teams at every agent company; distillation plays;
edge/small-model inference companies; the Prime Intellect ecosystem (Environments Hub) for
credibility adjacency. Opener: the cost-per-task plot, not the repo.

---
