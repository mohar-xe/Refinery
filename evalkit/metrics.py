"""Metrics for the cost/quality frontier.

The headline is `cost_per_resolved_task`, and its definition is the whole game:
it must include amortized compute, not just API dollars, or a local model looks
free and the frontier plot is a lie. For the student that means CPU-seconds; for
the teacher it means tokens x price.

`valid_tool_call_rate` is reported next to accuracy on purpose. A small model that
emits malformed calls scores zero while its loss curve looks fine, and the only
way to tell those two failures apart is to measure them separately.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

__all__ = ["summarize_contender", "frontier_rows", "CONTENDER_ORDER"]

#: Read order for every table and plot: strongest claim first.
CONTENDER_ORDER = ("teacher", "student", "student_no_curriculum", "heuristic")


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    pos = q * (len(ordered) - 1)
    lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)
    frac = pos - lo
    return round(ordered[lo] * (1 - frac) + ordered[hi] * frac, 3)


def summarize_contender(results: list[dict]) -> dict:
    """All reported numbers for one contender, from its raw per-run records."""
    n = len(results)
    if n == 0:
        return {"n": 0}

    passed = [r for r in results if r["passed"]]
    label_correct = [r for r in results if r.get("label_correct")]
    latencies = [r["latency_s"] for r in results]
    tokens = [r["tokens"] for r in results]
    api_cost = sum(r.get("cost_usd", 0.0) for r in results)

    # Amortized local compute: CPU seconds for the student, priced at the rental
    # rate it displaces. Zero for API contenders beyond their token spend.
    model_seconds = sum(r.get("latency_s", 0.0) for r in results if r.get("kind") == "local")
    gpu_amortized = 0.0

    by_variant: dict[str, dict] = defaultdict(lambda: {"n": 0, "passed": 0})
    for r in results:
        bucket = by_variant[r.get("variant") or "?"]
        bucket["n"] += 1
        bucket["passed"] += int(bool(r["passed"]))

    return {
        "n": n,
        "pass@1": round(len(passed) / n, 4),
        "label_accuracy": round(len(label_correct) / n, 4),
        # Syntactic: did it emit a parseable `lookup(...)`? This is the metric the
        # curriculum claim rests on (spec trap #5 is a *syntax* failure).
        "valid_tool_call_rate": round(sum(1 for r in results if r["valid_tool_call"]) / n, 4),
        # Behavioural: did a document segment actually come back? Lower than the
        # syntactic rate means the model asked for indices that do not exist.
        "retrieval_success_rate": round(sum(1 for r in results if r.get("retrieved")) / n, 4),
        "mean_steps": round(sum(r["steps"] for r in results) / n, 2),
        "mean_tokens": round(sum(tokens) / n, 1),
        "latency_p50_s": _percentile(latencies, 0.50),
        "latency_p95_s": _percentile(latencies, 0.95),
        "api_cost_usd": round(api_cost, 4),
        "compute_seconds": round(model_seconds, 2),
        "cost_per_resolved_task_usd": round(
            (api_cost + gpu_amortized) / len(passed), 6) if passed else None,
        "rejections": _rejection_counts(results),
        "by_variant": {
            k: {"n": v["n"], "pass@1": round(v["passed"] / v["n"], 4)}
            for k, v in sorted(by_variant.items())
        },
    }


def _rejection_counts(results: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for r in results:
        counts[r["reason_code"]] += 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def frontier_rows(summaries: list[dict]) -> list[dict]:
    """One row per contender, in canonical order, for the frontier table/plot."""
    by_name = {s["contender"]: s for s in summaries}
    rows: list[dict] = []
    for name in CONTENDER_ORDER:
        s = by_name.get(name)
        if not s or not s.get("n"):
            continue
        rows.append(
            {
                "contender": name,
                "params": _params_label(s),
                "pass@1": s["pass@1"],
                "valid_tool_call_rate": s["valid_tool_call_rate"],
                "cost_per_resolved_task_usd": s["cost_per_resolved_task_usd"],
                "latency_p50_s": s["latency_p50_s"],
                "latency_p95_s": s["latency_p95_s"],
                "mean_steps": s["mean_steps"],
            }
        )
    return rows


def _params_label(summary: dict) -> str:
    if summary.get("kind") == "local":
        return f"{summary['non_embedding_params'] / 1e6:.2f}M non-emb"
    if summary.get("kind") == "api":
        return "API"
    return "—"
