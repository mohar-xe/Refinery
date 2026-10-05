# LLD — Low-Level Design / Decision Log

> The HLD says *what the system is*. This file says **what I chose, and why**, at the level of
> modules, formats, and algorithms — including every place the toy-scale run forced a
> substitution, and what that substitution costs in claim strength.
>
> Convention: decisions are append-only, numbered `D-0NN`, never rewritten. If a decision is
> reversed, add a new entry that supersedes the old one and mark it `SUPERSEDED`.
>
> Status: `v0.1` · toy scale (`configs/toy.json`) · full scale (`configs/full.json`) unshipped

---

## 0. The core stance

**One codebase, two scales.** Nothing in `refinery/` knows whether it is running at toy or full
scale. Scale enters only through `refinery/config.py` (task source, model, caps, k, split sizes).
The pipeline stages are identical in both cases. This is the whole reason the toy run produces a
transferable claim: the thing being demonstrated is the *pipeline*, and the pipeline is the same
object in both configurations.

The corollary rule: **a substitution is only allowed if it changes scale, not structure.** If a
substitution would remove a stage, weaken a verifier, or skip a filter, it must be recorded here
as a cost, and it must be visible in the README.

---

## 1. Scale substitutions (read this before quoting any toy number)

| Full-scale (spec) | Toy-scale (shipped) | Structure kept? | Claim cost |
|---|---|---|---|
| SWE-Gym / SWE-smith repo tasks, pytest | SNLI + procedural adversarial NLI tasks, label check | ✅ verifier is still independent of the generator | weaker: labels are cheaper to verify than test suites, so the anti-hack surface is narrower — we say so and add synthetic traps to compensate |
| Teacher = frontier API, ~$0.10–0.30/run | Teacher = `stealth/space-bunny-alpha` (free, 1M ctx) | ✅ same agent loop, same trace format, same caps | free teacher hides real cost engineering → we substitute **requests, tokens, wall-time and rejection-rate accounting** for dollars, and report `$0.00` honestly rather than inventing a price |
| Student = Qwen3-8B / Llama-3.1-8B + **LoRA** r16–32 | Student = ~1M-param transformer trained from scratch | ❌ **LoRA is meaningless at 1M params** — there is nothing to adapt | the "distillation" claim is weaker: we cannot claim LoRA transfer, only small-model transfer. Recorded as `D-006` |
| Container farm, fresh image per verification | Local subprocess, fresh temp dir per verification | ⚠️ partial — isolation is weaker | anti-hack checks still run on a *pristine copy* of the input, so the "never trust the agent's working tree" property is preserved; the "container" is a temp dir, not a namespace. Recorded as `D-008` |
| Frontier plot vs frontier API | Same plot, teacher vs student vs a cheap majority-vote baseline | ✅ | the mid-tier contender is weaker than a real mid-tier API model |

---

## 2. Decisions

### D-001 — Scale is configuration, never a code branch
**Context.** The flagship is specified at ~300 tasks × 8 samples = 2,400 API runs. The phone can
afford ~1,000 free teacher requests/day and trains a 1M-param model on CPU in minutes.
**Decision.** Every stage takes a `Config` object. `configs/toy.json` and `configs/full.json`
differ only in values: `n_tasks`, `samples_per_task`, `caps`, `model`, `task_source`,
`student_init`. No `if toy:` anywhere in `refinery/`.
**Why.** A toy run is only evidence about the real system if it exercises the real code. A
scale-conditional branch would make the demo a different program.
**Rejected.** Separate `toy/` package (duplicated logic, and reviewers cannot tell which is real);
env vars (untyped, invisible in review).
**Consequence.** Toy runs are fast enough to iterate on a phone; the full config is a JSON edit.
**Revisit if** a stage genuinely cannot scale down without becoming a different stage.

### D-002 — Task = natural-language inference (entailed / not entailed), not repo repair
**Context.** Repo-fixing tasks need Docker-in-Docker, per-repo installs, and hours per run. Not
possible on a phone; also not possible inside the 1,000 req/day teacher budget.
**Decision.** Use SNLI (public, ~550k rows, verifiable by label) as the task pool, plus a
procedural adversarial generator that injects *deliberate shortcuts*.
**Why.** Entailment keeps every property the pipeline depends on: an objective ground truth, a
multi-step reasoning trace worth supervising, and a cheap independent verifier. It also keeps the
"generate k, keep verified, SFT" shape of rejection-sampling fine-tuning intact.
**Rejected.** Purely synthetic data (this is exactly the AgentTuning mistake the spec calls out —
synthetic tasks with no independent verification); classification-only (no reasoning trace, so the
trace format, windowing, and tool-call curriculum would all be vacuous).
**Consequence.** We get a real published dataset instead of self-generated slop.
**Revisit if** the full-scale run gets container budget (the config already swaps `task_source`).

### D-003 — The toy agent gets exactly one tool: `lookup(evidence_id)`
**Context.** The flagship's transferable claim is about *agentic tool use*. A pure classifier
trajectory has no tool calls, so the tool-call curriculum, the loss mask, and the windowing policy
would all be untested.
**Decision.** Each task carries a small evidence set. The agent may call `lookup` up to 3 times,
receives one evidence span per call, then must emit `ANSWER: entailed|not_entailed` plus the
evidence id it relied on.
**Why.** One tool is enough to exercise every structural feature (multi-step trace, tool-call
syntax, windowing of tool *outputs*, "cite your evidence" anti-hack check) while keeping the
prompt small enough for a 1M-context free teacher and a 1M-param student.
**Rejected.** A shell/filesystem tool (not reproducible for replay — see Project 2's
nondeterminism trap); several tools (curriculum signal gets noisy at this sample count).
**Consequence.** `valid_tool_call_rate` is a real metric, not a synthetic one.
**Revisit if** sample counts grow past ~2k, where multi-tool credit assignment becomes measurable.

### D-004 — Verifier is a *separate process*, and it re-derives its input
**Context.** The classic failure of this whole genre is verifying the generator's own claim in the
generator's own process.
**Decision.** `verifier/gate.py` takes `(task, trajectory)` as plain JSON, reads nothing from the
farm's memory, and re-reads the evidence set from the task manifest. Its checks are pure functions.
**Why.** It is the cheapest possible structural defence against the "so you SFT'd a model"
objection, and it is exactly the property the flagship spec demands: *never verify in the agent's
dirty working tree*.
**Rejected.** In-process assertion (trivially coupled to generator bugs); pytest on the farm's
artifacts (too slow to iterate on a phone).
**Consequence.** A bug in the farm cannot make a trajectory look verified.
**Revisit if** verification ever needs the executed side effects (it will at full scale — then it
becomes the fresh-container runner in `HLD §3.3`).

### D-005 — Every rejection carries a reason code from a closed enum
**Context.** "The rejection histogram is itself a publishable artifact" (spec trap #3). A histogram
is only publishable if the categories are stable and mutually exclusive.
**Decision.** Closed enum, first-match-wins evaluation order, one code per trajectory:
`VERIFIED_OK | TEST_FAIL | LABEL_LEAK | DUPLICATE_ACROSS_SPLIT | UNSUPPORTED_ANSWER | TRIVIAL_OUTPUT
| FORMAT_INVALID | NO_TOOL_CALL | STEP_CAP | REQUEST_CAP | TEACHER_ERROR`.
**Why.** First-match-wins makes the histogram a partition (counts sum to the number of runs), so
percentages are meaningful and two runs are comparable. A free-text reason is not.
**Rejected.** Multiple independent booleans (double-counting makes every percentage a lie);
free-text reasons (unqueryable, non-comparable across runs).
**Consequence.** `reports/rejection_histogram.md` is generated, never hand-written.
**Revisit if** a new failure mode appears — add a code, bump `SCHEMA_VERSION`, never reuse one.

### D-006 — Student is trained from scratch; LoRA is dropped and that is stated
**Context.** The spec's student is Qwen3-8B/Llama-3.1-8B with LoRA r16–32. The toy student is
~1M params.
**Decision.** A small decoder-only transformer trained from scratch, full-parameter, on CPU.
The trainer exposes the same knobs (loss mask, curriculum, packing) but has no adapter path.
**Why.** LoRA on a 1M-param model is not a scaled-down LoRA, it is a different (and pointless)
method. Pretrained small models are either too big for the CPU budget or (e.g. SmolLM-135M)
still 100× our target and would muddy the "extremely small" claim. Reporting the substitution is
worth more than performing the method.
**Rejected.** SmolLM-135M + LoRA (defensible but 135M ≠ 1M, and the CPU training cost is real);
training from scratch on 8B's LoRA (impossible here).
**Consequence.** The claim becomes "a 1M-param model trained on verified trajectories reaches Y% of
the teacher at ~0 marginal cost" — not "LoRA distillation works". Stated in the README's
limitations section.
**Revisit if** GPU budget appears: the full config swaps `student_init` to a base model + LoRA and
the trainer grows an adapter path behind the same interface.

### D-007 — Determinism: one renderer, imported by both the harness and the compiler
**Context.** Spec trap #2: "format drift is 60% of failure probability". The mitigation is that the
compiler must not be able to render differently from the harness.
**Decision.** `compiler/render.py` owns `system_prompt()`, `tool_schema()`, `render_messages()`.
`farm/harness.py` imports them to build the live prompt; `compiler/dataset.py` imports them to
build training text. A unit test asserts `render(train_sample) == render(live_run)` byte-for-byte
for a golden fixture.
**Why.** Two independent string builders would drift on the first refactor. One builder makes the
bug structurally impossible, which is stronger than a test that catches it.
**Rejected.** Snapshotting prompts into the dataset (then the dataset is only as correct as the
snapshot, and drift becomes invisible); a parity test alone (catches drift, doesn't prevent it).
**Consequence.** `prompt_hash` is computed by one function and recorded on every sample, so drift
is detectable in the field, not just in CI.

### D-008 — "Fresh environment" at toy scale = a pristine copy of the input, re-read from disk
**Context.** The flagship verifies in a fresh container. No containers here.
**Decision.** The verifier copies the task's evidence set and the candidate answer into a fresh
temp directory and re-reads it there; nothing is reused from the farm process.
**Why.** It preserves the property that matters (verification input is not the generator's
mutable state) while staying honest about the weaker isolation.
**Rejected.** Verifying in memory (reintroduces exactly the coupling `D-004` removes).
**Consequence.** Documented as ⚠️ partial in §1. Full-scale swaps this function for the container
runner; the interface does not change.

### D-009 — Append-only JSONL everywhere, content-addressed, no database
**Context.** No services, no systemd, and Android may kill the process at any time.
**Decision.** Every stage reads/writes JSONL under a run directory keyed by content hash. Stages
are resumable by "skip records whose id already exists".
**Why.** A phone process that dies at step 3 of 6 must not lose steps 1–2. JSONL + hash keys make
every stage idempotent and restartable, and the artifacts stay diffable and greppable.
**Rejected.** SQLite (fine, but a lock file that a killed process can leave behind is a real
hazard on Android; and it hides the data model); a single JSON blob (one bad write loses
everything).
**Consequence.** `runs/<run_id>/` is self-describing; `gsync`-able; diffable in review.

### D-010 — Windowing keeps arguments verbatim, digests outputs, never touches the last 3 steps
**Context.** Spec: trajectories exceed the context window and naive truncation destroys them.
**Decision.** (a) system prompt, tool schema, first plan, and **all tool call arguments** are kept
verbatim, always; (b) tool *results* older than the active window are replaced by a structured
digest `{evidence_id, n_chars, head}`; (c) the final 3 messages are never touched.
**Why.** The model is being taught to *fix* something; the fix window is the supervised signal.
Arguments are what the model must learn to emit; outputs are what it must learn to compress.
Truncating either one destroys the corresponding skill.
**Rejected.** Plain last-N truncation (the ablation arm that we expect to lose); summarizing
arguments too (then the model is trained to emit lossy tool calls — actively harmful).
**Consequence.** `windowing` is recorded per sample so the ablation is a `jq` filter, not a rerun.

### D-011 — Curriculum: short tool-call-only first, then full
**Context.** Spec trap #5: small models cannot emit valid tool calls at all without it.
**Decision.** Stage 1 trains only on trajectories with ≤ 3 tool calls and supervises **only** the
tool-call tokens. Stage 2 mixes in full trajectories and supervises assistant + tool-call tokens.
**Why.** The failure mode is syntactic, not semantic — the model needs many examples of the call
format before it can spend capacity on reasoning. Masking to tool-call tokens makes stage 1 a pure
format-learning problem, which is measurable independently.
**Rejected.** Mixed-from-scratch (that's the ablation arm); curriculum by dataset size (confounds
format with content).
**Consequence.** We report the ablation, and `valid_tool_call_rate` is the metric that shows whether
it worked.

### D-012 — Loss mask excludes tool results and system prompt
**Context.** If you supervise tool outputs, the model learns to predict the environment.
**Decision.** Loss is computed only on assistant-authored tokens: reasoning text, the `lookup(...)`
call, and the `ANSWER:` line. Everything else is masked to `-100`.
**Why.** Standard SFT practice for tool-use models, and it is the difference between a model that
*drives* tools and one that hallucinates their output.
**Consequence.** The mask is stored on the sample (`loss_mask`) so a different masking policy is a
compiler flag, not a data regeneration.

### D-013 — Cost accounting survives a free teacher
**Context.** The flagship's publishable artifact is the bill. Our teacher is free, so the bill is
`$0.00` and the headline cost story evaporates.
**Decision.** The ledger records requests, prompt/completion tokens, wall-clock seconds, and
teacher-error rate *always*, and dollars when a price is known. Reports print `$0.00` with the
real quantities beside it.
**Why.** The interesting engineering is throughput, tokens-per-verified-trajectory, and
waste-per-verified-trajectory — all of which survive a free teacher. Inventing a price would be
the single easiest way to lose credibility on this project.
**Rejected.** Omitting cost entirely (loses the artifact); quoting a list price for a free model
(misleading).
**Consequence.** The toy run's headline is **verified-trajectories per 1,000 teacher requests** and
**local cost per resolved task ≈ $0.00 (CPU electricity)** — a different metric with the same
shape as the full-scale one.

### D-014 — Python 3.12 in a uv-managed venv, because torch has no 3.14 wheels
**Context.** System Python is 3.14.4; torch does not publish 3.14 wheels, and the trainer needs
torch. Compiling torch from source on a phone is not happening.
**Decision.** `uv`-managed CPython 3.12 in `.venv/`; torch CPU wheels from
`download.pytorch.org/whl/cpu`. Pipeline stages 1–4 have **no torch dependency** and run on
system Python.
**Why.** Keeps the expensive-to-install dependency isolated to the one stage that needs it, and
stages 1–4 stay runnable on a bare interpreter (important on a 1 GB-free-RAM phone).
**Rejected.** System-wide torch install (couples every stage to a 300 MB import);
pure-Python autograd for the student (correct but ~100× slower per step; we would be training the
*framework* instead of the pipeline).
**Consequence.** `scripts/setup.sh` provisions the venv; `scripts/run_toy.sh` picks the right
interpreter per stage.

### D-015 — ~1M params means: 6 layers, d_model 256, 4 heads, vocab 8k (BPE-free word level)
**Context.** Target ≈ 1M non-embedding parameters, CPU-trainable, and small enough that the whole
thing trains in minutes on 8 phone cores.
**Decision.** Decoder-only transformer, 6 layers, d_model 256, 4 heads, d_ff 1024, context 512,
tied embeddings, word-level vocabulary built from the training split only.
**Why.** Word-level (not BPE) because the tokenizer must not become the confound: with a word-level
vocab the `lookup(evidence_id)` call is a handful of tokens, so tool-call syntax is learnable in
stage 1 of the curriculum at all. Tied embeddings because at 1M params the embedding matrix is a
large fraction of the budget.
**Consequence.** Non-embedding params ≈ 6 × (4·256² + 2·256·1024) ≈ 4.7M... **too big** — see D-015a.
**Revisit if** training loss plateaus above chance.

### D-015a — Corrected parameter budget: 4 layers, d_model 192, d_ff 768 → ≈ 1.0M
**Context.** D-015's arithmetic does not reach 1M: 6 layers × (4·256² + 2·256·1024) ≈ 4.7M
non-embedding parameters, an order of magnitude over target.
**Decision.** 4 layers, d_model 192, 4 heads (head dim 48), d_ff 768, context 512, tied embeddings.
Non-embedding params = 4 × (4·192² + 2·192·768) = 4 × (147,456 + 294,912) = **1.77M**... still
over, because attention projections dominate at small d_model.
Going to 3 layers, d_model 160: 3 × (4·160² + 2·160·640) = 3 × (102,400 + 204,800) = **0.92M**. ✅
**Decision (final):** 3 layers, d_model 160, 4 heads, d_ff 640, context 512, tied embeddings,
vocab ≤ 8,000 → **≈ 0.92M non-embedding parameters + ~2.6M embedding rows** (tied, and reported
separately so the number is not flattered).
**Why.** Recording the failed arithmetic is the point of this file: "1M params" is a claim about
non-embedding params, and the embedding table is a different cost centre that must be disclosed.
**Consequence.** README reports both numbers, and the frontier plot labels the student honestly.

---

## 3. Trap → code map

Where each spec trap is actually defended. Every row must point at a function that exists.

| Trap | Defence | Location |
|---|---|---|
| 1 contamination | split key = `sha256(task_id + salt)`; identity = `sha256(repo/premise+hypothesis+evidence)` | `refinery/common/hashing.py`, `taskpool/build.py` |
| 2 format drift | single renderer imported by harness *and* compiler; parity test | `compiler/render.py`, `tests/test_format_parity.py` |
| 3 reward hacks | closed reason-code enum + unsupported-answer + leak checks | `verifier/gate.py`, `verifier/antihack.py` |
| 4 farm cost bleed | hard caps in steps and requests, enforced *before* each call; ledger | `farm/harness.py`, `farm/ledger.py` |
| 5 no valid tool calls | curriculum stage 1, tool-call-only loss; `valid_tool_call_rate` metric | `trainer/train.py`, `evalkit/metrics.py` |
| 6 "you just SFT'd" | pipeline-first README; ablation + histogram above the model card | `README.md`, `reports/` |

---

## 4. Open decisions (deliberately not yet made)

- **Verifier strength at scale.** Label equality is a weak verifier compared to a test suite. At
  full scale, replace with pytest-in-container; keep the reason-code enum so the histogram stays
  comparable across the migration (`D-005`).
- **Evidence-set size.** 3 candidate spans per task is arbitrary; the right value depends on how
  often the teacher's cited evidence is the distractor. Deferred to the first histogram.
- **Whether the majority-vote baseline is a fair "mid-tier contender".** It is cheap and free but it
  is not a model. Decide after the first frontier plot; if it is embarrassing, drop the row rather
  than defend it.
