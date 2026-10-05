"""Verifier tests: the histogram must be a partition, and the filters sound.

Two distinct claims are tested here:

  1. **Partition.** Every run gets exactly one reason code, and the counts sum to
     the number of runs. Without this the rejection histogram is not a
     distribution and the headline number is unfalsifiable.
  2. **Soundness.** Each anti-hack filter fires on a provable defect and *not* on
     a merely-weak-looking trajectory. A filter that rejects valid reasoning
     silently deletes the hard examples the dataset exists to collect.
"""

from __future__ import annotations

from refinery.common.codes import ORDER, REASON_CODES, Reason
from refinery.common.protocol import parse_assistant
from refinery.verifier import antihack
from refinery.verifier.gate import histogram, verify

TASK = {
    "task_id": "sha256:t1",
    "hypothesis": "A boy is playing a guitar.",
    "label": "entailed",
    "split": "train",
    "prompt_variant": "base",
    "evidence": [
        {"evidence_id": "e0", "index": 0, "text": "A boy sits on a stool."},
        {"evidence_id": "e1", "index": 1, "text": "He strums a guitar."},
    ],
}

TRAP_TASK = {
    **TASK,
    "task_id": "sha256:t2",
    "prompt_variant": "premise_negation",
    "hypothesis": "It is not the case that He strums a guitar.",
    "label": "not_entailed",
    "evidence": [
        {"evidence_id": "e0", "index": 0, "text": "A boy sits on a stool."},
        {"evidence_id": "e1", "index": 1, "text": "He strums a guitar."},
    ],
}


def _run(answer="entailed", cite=(0, 1), *, retrieved=(0, 1), plan="PLAN: read both.", exit="answered"):
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": f"{plan}\nlookup(0)"},
    ]
    for idx in retrieved:
        messages.append(
            {"role": "tool", "content": TASK["evidence"][idx]["text"], "evidence_index": idx,
             "tool_ok": True}
        )
    cite_s = ", ".join(str(i) for i in cite)
    messages.append({"role": "assistant", "content": f"PLAN: compare.\nCITE: {cite_s}\nANSWER: {answer}"})
    return {"run_id": f"{TASK['task_id']}:0", "task_id": TASK["task_id"], "exit": exit,
            "messages": messages, "steps": 2, "requests": 2, "tokens": {"total": 100},
            "n_lookups": len(retrieved), "trajectory_hash": "sha256:x"}


def test_codes_are_a_partition():
    assert set(ORDER) == set(Reason)
    assert len(set(ORDER)) == len(ORDER)
    assert set(REASON_CODES) == {r.value for r in Reason}


def test_histogram_sums_to_total():
    runs = [
        (_run(), TASK),
        (_run(answer="not_entailed"), TASK),
        (_run(cite=(0, 1), retrieved=(0,)), TASK),
        (_run(retrieved=(), exit="answered"), TASK),
        (_run(exit="step_cap"), TASK),
    ]
    verdicts = [verify(task, run) for run, task in runs]
    hist = histogram(verdicts)
    assert sum(hist["counts"].values()) == hist["total"] == len(verdicts)
    assert hist["counts"]["VERIFIED_OK"] == 1


def test_correct_trajectory_verifies():
    assert verify(TASK, _run())["reason_code"] == "VERIFIED_OK"


def test_wrong_label_is_test_fail():
    assert verify(TASK, _run(answer="not_entailed"))["reason_code"] == "TEST_FAIL"


def test_hallucinated_citation_is_unsupported():
    verdict = verify(TASK, _run(cite=(0, 1, 2)))
    assert verdict["reason_code"] == "UNSUPPORTED_ANSWER"
    assert verdict["detail"]["why"] == "citation_not_retrieved"


def test_negation_trap_rejects_confident_wrong_reasoning():
    """The headline anti-hack case: right-ish surface, broken reasoning.

    Claiming `entailed` for a hypothesis that negates the cited sentence is a
    logical contradiction, so it is rejected even though... it also gets the label
    wrong here. The point is the *reason code* is UNSUPPORTED, not TEST_FAIL, so
    the histogram shows a reasoning failure rather than a knowledge failure.
    """
    run = _run(answer="entailed", cite=(1,))
    verdict = verify(TRAP_TASK, run)
    assert verdict["reason_code"] == "UNSUPPORTED_ANSWER"
    assert verdict["detail"]["why"] == "negation_conflict"


def test_no_tool_call_is_rejected():
    run = _run()
    run["messages"] = [m for m in run["messages"] if m.get("role") != "tool"]
    assert verify(TASK, run)["reason_code"] == "NO_TOOL_CALL"


def test_trivial_output_is_rejected():
    run = _run(plan="ok.")
    assert verify(TASK, run)["reason_code"] == "TRIVIAL_OUTPUT"


def test_missing_answer_is_format_invalid():
    run = _run()
    run["messages"][-1]["content"] = "PLAN: I am not sure."
    assert verify(TASK, run)["reason_code"] == "FORMAT_INVALID"


def test_caps_short_circuit_before_content_checks():
    """Infrastructure noise must not be reported as an agent failure."""
    run = _run(exit="step_cap")
    run["messages"] = [{"role": "assistant", "content": "junk"}]
    assert verify(TASK, run)["reason_code"] == "STEP_CAP"


def test_label_leak_detected_on_task():
    leaky = {**TASK, "hypothesis": "The statement is entailed, plainly."}
    assert antihack.label_leak(leaky)
    assert not antihack.label_leak(TASK)


def test_support_conflict_does_not_fire_on_weak_overlap():
    """Soundness: unrelated evidence must NOT be treated as a contradiction."""
    assert not antihack.support_conflict("A boy is playing a guitar.", ["A woman sells fish."], "entailed")


def test_support_conflict_only_judges_entailed_claims():
    """A `not_entailed` claim is never second-guessed by the negation heuristic:
    we cannot prove support for a negative claim without a model, and guessing
    would delete legitimate rejections."""
    assert not antihack.support_conflict(
        "It is not the case that He strums a guitar.", ["He strums a guitar."], "not_entailed"
    )


def test_duplicate_across_split_tripwire():
    verdict = verify(TASK, _run(), split_index={TASK["task_id"]: {"train", "eval"}})
    assert verdict["reason_code"] == "DUPLICATE_ACROSS_SPLIT"


def test_parser_takes_the_last_answer_not_the_first():
    """A stream of self-corrections must not launder a guess into a pass."""
    text = "PLAN: hmm.\nANSWER: not_entailed\nANSWER: entailed"
    assert parse_assistant(text).answer == "entailed"
