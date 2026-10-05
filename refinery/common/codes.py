"""The closed reason-code enum (LLD.md D-005).

The rejection histogram is a *publishable artifact*, which means the categories
must be stable, mutually exclusive, and exhaustive. Two rules follow:

  1. Evaluation is first-match-wins in `ORDER`, so the codes form a partition:
     counts sum to the number of runs and percentages are meaningful.
  2. A code is never reused or repurposed. New failure mode -> new code +
     SCHEMA_VERSION bump, so histograms from different runs stay comparable.

`TEST_FAIL` keeps its name from the full-scale design (pytest in a container) even
though the toy verifier checks a label: the enum is the contract across the
migration, and renaming it would break comparability of published histograms.
"""

from __future__ import annotations

from enum import StrEnum


class Reason(StrEnum):
    """Why a trajectory was accepted or rejected."""

    VERIFIED_OK = "VERIFIED_OK"

    # --- the objective check failed -------------------------------------------
    TEST_FAIL = "TEST_FAIL"

    # --- anti-hack / trust-boundary violations -------------------------------
    LABEL_LEAK = "LABEL_LEAK"
    DUPLICATE_ACROSS_SPLIT = "DUPLICATE_ACROSS_SPLIT"
    UNSUPPORTED_ANSWER = "UNSUPPORTED_ANSWER"
    TRIVIAL_OUTPUT = "TRIVIAL_OUTPUT"

    # --- the trajectory is not usable for SFT --------------------------------
    FORMAT_INVALID = "FORMAT_INVALID"
    NO_TOOL_CALL = "NO_TOOL_CALL"

    # --- the run was cut short by a cap or by the provider -------------------
    STEP_CAP = "STEP_CAP"
    REQUEST_CAP = "REQUEST_CAP"
    TEACHER_ERROR = "TEACHER_ERROR"


#: First-match-wins evaluation order — the order `verifier/gate.py` applies.
#: Operational codes come first: if the provider failed or a cap fired, there is
#: nothing else to evaluate, and reporting that as `FORMAT_INVALID` would
#: misattribute infrastructure noise to the agent.
#:
#: Note the middle of the list: anti-hack checks deliberately precede the
#: objective check. A trajectory whose cited evidence contradicts its answer is
#: rejected even when the label happens to be right, because training on it
#: teaches the model to reach the right answer with broken reasoning. That
#: precedence is the core anti-reward-hack move.
ORDER: tuple[Reason, ...] = (
    Reason.TEACHER_ERROR,
    Reason.STEP_CAP,
    Reason.REQUEST_CAP,
    Reason.FORMAT_INVALID,
    Reason.NO_TOOL_CALL,
    Reason.LABEL_LEAK,
    Reason.DUPLICATE_ACROSS_SPLIT,
    Reason.UNSUPPORTED_ANSWER,
    Reason.TRIVIAL_OUTPUT,
    Reason.TEST_FAIL,
    Reason.VERIFIED_OK,
)

REASON_CODES: tuple[str, ...] = tuple(r.value for r in Reason)

#: Codes that mean "the agent tried to win the metric instead of the task".
REWARD_HACK_CLASSES: frozenset[Reason] = frozenset(
    {Reason.LABEL_LEAK, Reason.UNSUPPORTED_ANSWER, Reason.TRIVIAL_OUTPUT}
)


def is_valid(code: str) -> bool:
    return code in REASON_CODES
