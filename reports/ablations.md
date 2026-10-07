# Ablations

One decision, run both ways, numbers on both sides.

## Windowing strategy (data compiler)

| Arm | pass@1 | valid tool calls | mean tokens |
|---|---|---|---|
| `full_context` | 0.000 | 0.595 | 362.0 |
| `naive_truncated` | 0.000 | 0.973 | 132.0 |

`full_context` is the upper bound where trajectories fit the student
context; `windowed` is the shipping policy (arguments verbatim, old tool
results digested, last 3 steps untouched); `naive_truncated` keeps the head of
the conversation and drops the tail, which should lose the fix window.

## Curriculum (short tool-call-only stage first)

| Arm | pass@1 | valid tool calls |
|---|---|---|
| curriculum | 0.000 | 0.540 |
| mixed from scratch | 0.000 | 0.180 |

The metric to watch is `valid tool calls`: spec trap #5 is a *syntactic*
failure, so it shows up there before it shows up in accuracy.
