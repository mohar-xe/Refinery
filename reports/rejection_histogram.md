# Rejection histogram

Generated from `/root/Projects/Refinery/runs/toy/verdicts.jsonl` — 328 runs, schema `codes.ORDER` (first match wins, so the codes partition the runs).

| Reason code | Count | Share | Class |
|---|---|---|---|
| `VERIFIED_OK` | 217 | 0.6616 | accept |
| `TEST_FAIL` | 76 | 0.2317 | reject |
| `LABEL_LEAK` | 0 | 0.0000 | hack |
| `DUPLICATE_ACROSS_SPLIT` | 0 | 0.0000 | reject |
| `UNSUPPORTED_ANSWER` | 35 | 0.1067 | hack |
| `TRIVIAL_OUTPUT` | 0 | 0.0000 | hack |
| `FORMAT_INVALID` | 0 | 0.0000 | reject |
| `NO_TOOL_CALL` | 0 | 0.0000 | reject |
| `STEP_CAP` | 0 | 0.0000 | reject |
| `REQUEST_CAP` | 0 | 0.0000 | reject |
| `TEACHER_ERROR` | 0 | 0.0000 | reject |

**Verified:** 217/328 (0.6616)

## What the label check alone would have accepted

- Runs whose predicted label matched gold: **217**
- Of those, rejected by an anti-hack filter: **0**
- Reward-hack rate among label-passes: **0.0000**

> This ratio is the toy-scale stand-in for the flagship's "14% of 'passing'
> trajectories were reward hacks". It is a *floor*: our filters only catch
> provable defects (see `verifier/antihack.py`), because a stricter filter would
> silently delete hard examples rather than catch more hacks.

## Rejections by adversarial variant

| Variant | TEST_FAIL | UNSUPPORTED_ANSWER | VERIFIED_OK |
|---|---|---|---|
| base | 69 | 6 | 137 |
| distractor | 7 | 0 | 33 |
| negation_flip | 0 | 6 | 38 |
| premise_negation | 0 | 23 | 9 |
