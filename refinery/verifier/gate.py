"""The verifier gate — the trust boundary.

Invariants (HLD §3.3, LLD.md D-004):

  * **Pure.** `verify()` takes a task and a run as plain JSON and touches no
    process state. A bug in the farm cannot manufacture a passing trajectory.
  * **First-match-wins over `codes.ORDER`.** The codes form a partition, so the
    rejection histogram is a real distribution: counts sum to the number of runs.
  * **Independent input.** The task is re-read from the manifest, and
    `verify_manifest()` additionally writes a pristine copy of the task to a temp
    directory and verifies against *that* (LLD.md D-008). At full scale this is
    where the fresh container gets built; the interface does not change.
  * **Sound before strict.** Anti-hack filters reject only provable defects.
    See `antihack.py`.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from refinery.common.codes import REASON_CODES, REWARD_HACK_CLASSES, Reason
from refinery.common.jsonl import read_jsonl
from refinery.common.protocol import parse_assistant
from refinery.verifier import antihack

__all__ = ["verify", "verify_manifest", "histogram"]


def _retrieved_indices(messages: list[dict]) -> set[int]:
    return {
        msg["evidence_index"]
        for msg in messages
        if msg.get("role") == "tool" and msg.get("tool_ok") and "evidence_index" in msg
    }


def _cited_texts(task: dict, messages: list[dict], cite: list[int]) -> list[str]:
    by_index = {e["index"]: e["text"] for e in task.get("evidence", [])}
    return [by_index[i] for i in cite if i in by_index]


def _final_assistant(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            return msg["content"]
    return ""


def verify(task: dict, run: dict, *, split_index: dict[str, set[str]] | None = None) -> dict:
    """Return a verdict record for one run. Never raises on bad input."""
    messages = run.get("messages") or []
    exit_reason = run.get("exit", "")
    retrieved = _retrieved_indices(messages)
    parsed = parse_assistant(_final_assistant(messages))

    reason: Reason = Reason.VERIFIED_OK
    detail: dict[str, Any] = {}

    # ---- operational short-circuits ----------------------------------------
    if exit_reason == "teacher_error":
        reason = Reason.TEACHER_ERROR
        detail = {"error": run.get("messages", [{}])[-1].get("error") if messages else None}
    elif exit_reason == "step_cap":
        reason, detail = Reason.STEP_CAP, {"steps": run.get("steps")}
    elif exit_reason == "request_cap":
        reason, detail = Reason.REQUEST_CAP, {"requests": run.get("requests")}

    # ---- trajectory must be parseable --------------------------------------
    elif not parsed.has_answer or not parsed.cite_present:
        reason = Reason.FORMAT_INVALID
        detail = {"parse_error": parsed.error, "cited": parsed.cite, "answer": parsed.answer}

    # ---- must have retrieved something -------------------------------------
    elif not antihack.has_successful_tool_call(messages):
        reason = Reason.NO_TOOL_CALL
        detail = {"retrieved": sorted(retrieved)}

    # ---- task-level defects ------------------------------------------------
    elif antihack.label_leak(task):
        reason = Reason.LABEL_LEAK
        detail = {"label": task.get("label")}

    elif split_index is not None and len(split_index.get(task.get("task_id", ""), set())) > 1:
        # Tripwire: impossible by construction (HLD §3.1). If this ever fires,
        # the identity function is broken and every other guarantee is void.
        reason = Reason.DUPLICATE_ACROSS_SPLIT
        detail = {"task_id": task.get("task_id")}

    # ---- anti-hack: the reasoning must hold up ------------------------------
    elif not antihack.citation_integrity(parsed.cite, retrieved):
        reason = Reason.UNSUPPORTED_ANSWER
        detail = {"cited": parsed.cite, "retrieved": sorted(retrieved), "why": "citation_not_retrieved"}

    elif antihack.support_conflict(
        task.get("hypothesis", ""), _cited_texts(task, messages, parsed.cite), parsed.answer or ""
    ):
        reason = Reason.UNSUPPORTED_ANSWER
        detail = {"cited": parsed.cite, "why": "negation_conflict"}

    elif antihack.is_trivial(messages):
        reason = Reason.TRIVIAL_OUTPUT
        detail = {"why": "no_reasoning_before_first_tool_call"}

    # ---- the objective check, last -----------------------------------------
    elif parsed.answer != task.get("label"):
        reason = Reason.TEST_FAIL
        detail = {"predicted": parsed.answer, "gold": task.get("label")}

    return {
        "run_id": run.get("run_id"),
        "task_id": task.get("task_id"),
        "sample_idx": run.get("sample_idx"),
        "reason_code": reason.value,
        "detail": detail,
        "checks": {
            "parsed": reason not in (Reason.FORMAT_INVALID,),
            "tool_called": bool(retrieved),
            "no_leak": reason is not Reason.LABEL_LEAK,
            "citation_ok": reason is not Reason.UNSUPPORTED_ANSWER,
            "nontrivial": reason is not Reason.TRIVIAL_OUTPUT,
            "label_match": reason is not Reason.TEST_FAIL,
        },
        "predicted": parsed.answer,
        "gold": task.get("label"),
        "variant": task.get("prompt_variant"),
        "steps": run.get("steps"),
        "requests": run.get("requests"),
        "tokens": (run.get("tokens") or {}).get("total"),
        "trajectory_hash": run.get("trajectory_hash"),
    }


def verify_manifest(cfg: Any, *, shards: list[Path] | None = None) -> dict:
    """Verify every run on disk against a freshly-read manifest.

    The task used for verification is re-materialised into a temp directory
    (LLD.md D-008): verification input is never the generator's live state, only
    a pristine copy re-read from the manifest of record.
    """
    tasks = {t["task_id"]: t for t in read_jsonl(cfg.manifest_path)}
    split_index: dict[str, set[str]] = {}
    for task in tasks.values():
        split_index.setdefault(task["task_id"], set()).add(task.get("split", "?"))

    shard_paths = shards or sorted(cfg.runs_dir.glob("shard-*.jsonl"))
    runs: list[dict] = []
    for path in shard_paths:
        runs.extend(read_jsonl(path))

    verdicts: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="refinery-verify-") as tmp:
        for run in runs:
            task = tasks.get(run.get("task_id", ""))
            if task is None:
                verdicts.append(
                    {
                        "run_id": run.get("run_id"),
                        "task_id": run.get("task_id"),
                        "reason_code": Reason.FORMAT_INVALID.value,
                        "detail": {"why": "task_not_in_manifest"},
                    }
                )
                continue
            pristine = Path(tmp) / f"{task['task_id']}.json"
            pristine.write_text(json.dumps(task, sort_keys=True), encoding="utf-8")
            verdicts.append(verify(json.loads(pristine.read_text()), run, split_index=split_index))

    from refinery.common.jsonl import append_unique, write_json

    written = append_unique(cfg.verdicts_path, verdicts, key="run_id")
    hist = histogram(verdicts)
    write_json(cfg.root / "verdicts_summary.json", hist)
    return {"runs_seen": len(runs), "verdicts_written": written, "histogram": hist}


def histogram(verdicts: list[dict]) -> dict:
    """Counts and shares per reason code, plus the reward-hack subtotal.

    `REWARD_HACK_CLASSES` is the headline number in the write-up: what fraction of
    trajectories that *looked* solved were reaching for the metric instead of the
    task. It is computed over all runs, and separately over runs the objective
    check alone would have accepted — the second number is the honest "what did
    the label check miss" figure.
    """
    total = len(verdicts)
    counts = {code: 0 for code in REASON_CODES}
    for v in verdicts:
        counts[v["reason_code"]] = counts.get(v["reason_code"], 0) + 1

    label_ok = [v for v in verdicts if v.get("predicted") == v.get("gold") and v.get("predicted")]
    hacks = [v for v in label_ok if v["reason_code"] in {r.value for r in REWARD_HACK_CLASSES}]

    return {
        "total": total,
        "counts": counts,
        "shares": {c: (round(n / total, 4) if total else 0.0) for c, n in counts.items()},
        "verified": counts.get(Reason.VERIFIED_OK.value, 0),
        "verify_rate": round(counts.get(Reason.VERIFIED_OK.value, 0) / total, 4) if total else 0.0,
        "label_check_would_accept": len(label_ok),
        "rejected_despite_correct_label": len(hacks),
        "reward_hack_rate_of_label_passes": round(len(hacks) / len(label_ok), 4) if label_ok else 0.0,
    }


def clear_verdicts(path: Path) -> None:
    """Remove verdicts so a re-verification is not skipped as 'already done'."""
    if path.exists():
        shutil.copy2(path, path.with_suffix(".jsonl.bak"))
        path.unlink()
