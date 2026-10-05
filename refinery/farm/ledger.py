"""The farm's spend ledger.

Every request appends one line. This is the artifact that replaces the dollar
bill when the teacher is free (LLD.md D-013): a `$0.00` bill with no quantities
beside it is a useless artifact, whereas requests + tokens + wall time + waste per
verified trajectory is the same engineering story with the price removed.

The interesting farm metrics are derived, not the total:
  * tokens per *verified* trajectory (efficiency of the filter)
  * requests wasted on runs the verifier rejected (efficiency of the caps)
  * teacher error rate (is the bottleneck the model or the provider?)
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

from refinery.common.jsonl import append_jsonl, read_jsonl

__all__ = ["record_request", "summarize", "ledger_rows"]


def record_request(
    path: Any,
    *,
    run_id: str,
    task_id: str,
    teacher: str,
    model: str,
    attempt: int,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
    latency_s: float,
    error: str | None = None,
) -> dict:
    entry = {
        "ts": round(time.time(), 3),
        "run_id": run_id,
        "task_id": task_id,
        "teacher": teacher,
        "model": model,
        "attempt": attempt,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tokens": prompt_tokens + completion_tokens,
        "cost_usd": round(cost_usd, 6),
        "latency_s": round(latency_s, 3),
        "error": error,
    }
    append_jsonl(path, [entry])
    return entry


def ledger_rows(path: Any) -> list[dict]:
    return list(read_jsonl(path))


def summarize(path: Any, *, verdicts_path: Any = None) -> dict:
    """Totals plus the derived efficiency numbers that make the ledger useful."""
    rows = ledger_rows(path)
    if not rows:
        return {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "tokens": 0,
                "cost_usd": 0.0, "latency_s": 0.0, "errors": 0, "error_rate": 0.0}

    total_tokens = sum(r.get("tokens", 0) for r in rows)
    errors = sum(1 for r in rows if r.get("error"))
    ok_rows = [r for r in rows if not r.get("error")]

    out = {
        "requests": len(rows),
        "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in rows),
        "completion_tokens": sum(r.get("completion_tokens", 0) for r in rows),
        "tokens": total_tokens,
        "cost_usd": round(sum(r.get("cost_usd", 0.0) for r in rows), 4),
        "latency_s": round(sum(r.get("latency_s", 0.0) for r in rows), 1),
        "errors": errors,
        "error_rate": round(errors / len(rows), 4),
        "mean_latency_s": round(sum(r.get("latency_s", 0.0) for r in ok_rows) / max(len(ok_rows), 1), 2),
    }

    if verdicts_path is not None:
        verified = [v for v in read_jsonl(verdicts_path) if v.get("reason_code") == "VERIFIED_OK"]
        rejected_runs = {v["run_id"] for v in read_jsonl(verdicts_path)} - {
            v["run_id"] for v in verified
        }
        wasted = sum(
            r.get("tokens", 0) for r in rows if r.get("run_id") in rejected_runs and not r.get("error")
        )
        out.update(
            {
                "verified_trajectories": len(verified),
                "tokens_per_verified": round(total_tokens / len(verified), 1) if verified else None,
                "requests_per_verified": round(len(ok_rows) / len(verified), 2) if verified else None,
                "tokens_wasted_on_rejected": wasted,
                "waste_share": round(wasted / total_tokens, 4) if total_tokens else None,
            }
        )
    return out


def iter_by_run(rows: Iterable[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["run_id"], []).append(row)
    return grouped
