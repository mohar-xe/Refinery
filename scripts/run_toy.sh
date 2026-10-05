#!/usr/bin/env bash
# End-to-end toy-scale run: stages 1-6 plus both ablations, writing reports/.
#
# Every stage is resumable, so re-running after an interruption is cheap. Pass
# arguments to forward them to `refinery eval`, e.g.
#
#   ./scripts/run_toy.sh --seeds 1            # quick pass
#   ./scripts/run_toy.sh --no-teacher         # skip the API contender (no quota)
#
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"
CONFIG="${CONFIG:-configs/toy.json}"

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"   # stages 1-4 need no torch

STEP() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
NO_TEACHER=0
for arg in "$@"; do
  [ "$arg" = "--no-teacher" ] && NO_TEACHER=1
done

# Teacher: the real model if a key is present, otherwise the offline heuristic.
# The pipeline must be runnable with no key at all (LLD.md D-003 rationale), so a
# missing key downgrades the run instead of failing it.
if [ -n "${OPENROUTER_API_KEY:-}" ]; then
  TEACHER=api
else
  TEACHER=heuristic
  echo "!! OPENROUTER_API_KEY is not set — generating with the offline heuristic."
  echo "!! The dataset will still be built; the 'teacher' contender will be absent."
fi

STEP "1/6  task pool"
$PY -m refinery.cli --config "$CONFIG" taskpool

STEP "2/6  generation farm (k samples/task)"
$PY -m refinery.cli --config "$CONFIG" farm --teacher "$TEACHER" --split train

STEP "3/6  verifier gate (independent check + anti-hack filters)"
$PY -m refinery.cli --config "$CONFIG" verify --fresh

STEP "4/6  dataset compiler (deterministic render + windowing)"
$PY -m refinery.cli --config "$CONFIG" compile --strategies windowed,naive_truncated

STEP "5/6  train student (curriculum) + ablation arm (no curriculum)"
$PY -m refinery.cli --config "$CONFIG" train --name student
$PY -m refinery.cli --config "$CONFIG" train --name student_no_curriculum --no-curriculum

STEP "6/6  eval — identical harness, identical caps"
CONTENDERS="student,student_no_curriculum,heuristic"
[ "$NO_TEACHER" -eq 0 ] && [ -n "${OPENROUTER_API_KEY:-}" ] && CONTENDERS="teacher,$CONTENDERS"
$PY -m refinery.cli --config "$CONFIG" eval --contenders "$CONTENDERS" --seeds "${SEEDS:-3}" \
  --split eval "${EXTRA_EVAL_ARGS:-}"

# Windowing ablation: train on the naive arm and evaluate it through the same code.
# At toy scale `windowed` and `full_context` compile to identical bytes (tool
# results are shorter than the digest envelope), so this arm isolates the part of
# the policy that does bite here — losing the tool-call supervision entirely.
if [ "${SKIP_WINDOW_ABLATION:-0}" != "1" ]; then
  STEP "ablation: naive-truncated windowing"
  $PY -m refinery.cli --config "$CONFIG" train --name win_naive_truncated --strategy naive_truncated
  $PY -m refinery.cli --config "$CONFIG" train --name win_full_context --strategy full_context
  $PY -m refinery.cli --config "$CONFIG" eval --contenders win_naive_truncated,win_full_context \
    --seeds "${SEEDS:-3}" --split eval
fi

STEP "reports"
$PY -m refinery.cli --config "$CONFIG" report

echo
echo "done. numbers: reports/frontier.md  reports/rejection_histogram.md  reports/bill.md"
