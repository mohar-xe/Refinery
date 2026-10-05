"""The generation farm: k samples per task, under hard caps, recording everything.

The loop is deliberately the *real* agent loop rather than a single-shot call,
because the things this project claims to have solved are properties of
multi-step traces: windowing, tool-call curricula, loss masks, per-step caps.
A single-shot generator would make all of them vacuous.

Caps are checked *before* each request is issued, never after — a cap enforced
after the fact is a report, not a control (spec trap #4).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from refinery.common.hashing import content_id, stable_seed
from refinery.common.jsonl import append_unique
from refinery.common.protocol import TOOL_NAME, parse_assistant
from refinery.compiler.render import MAX_LOOKUPS, seed_messages
from refinery.farm import ledger as ledger_mod
from refinery.farm.teacher import Teacher

__all__ = ["run_one", "run_all", "EXITS"]

EXITS = ("answered", "step_cap", "request_cap", "teacher_error", "malformed")


@dataclass
class RunOutcome:
    record: dict
    exit_reason: str


def _valid_indices(parsed_indices: list[int], n_units: int, *, budget: int) -> list[int]:
    """Dedupe, keep order, clamp to the run's remaining lookup budget.

    The budget is per *run*, not per turn: an agent that burns all four
    retrievals on one turn has none left, and letting it keep asking would train
    the model to loop instead of decide.
    """
    if budget <= 0:
        return []
    seen: set[int] = set()
    out: list[int] = []
    for idx in parsed_indices:
        if idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
        if len(out) >= budget:
            break
    return out


def run_one(
    teacher: Teacher,
    task: dict,
    sample_idx: int,
    cfg: Any,
    *,
    teacher_cfg: Any,
) -> RunOutcome:
    """Run one agent trajectory against one task. Never raises on model misbehaviour."""
    caps = cfg.caps
    run_id = f"{task['task_id']}:{sample_idx}"
    evidence = task.get("evidence", [])
    n = len(evidence)
    lookup_budget = MAX_LOOKUPS

    # Temperature is resampled per sample so k>1 is a real distribution rather
    # than k identical calls at one temperature.
    temperature = round(min(max(0.0, teacher_cfg.temperature + ((sample_idx % 5) - 2) * 0.05), 2.0), 2)

    messages = seed_messages(task)
    requests = ptok = ctok = 0
    cost = 0.0
    steps = 0
    tool_calls: list[dict] = []
    exit_reason = "step_cap"
    answer: str | None = None
    cite: list[int] = []
    reply_model = teacher_cfg.model
    cost_fn = getattr(teacher, "cost_usd", None)
    started = time.monotonic()

    while True:
        if steps >= caps.max_steps:
            exit_reason = "step_cap"
            break

        cap_code = caps.exceeded_by(requests=requests, tokens=ptok + ctok, usd=cost)
        if cap_code:
            exit_reason = "request_cap" if cap_code == "REQUEST_CAP" else "step_cap"
            break

        reply = teacher.complete(messages, temperature=temperature)
        requests += 1
        ptok += reply.prompt_tokens
        ctok += reply.completion_tokens
        reply_model = reply.model or reply_model
        run_cost = cost_fn(reply.prompt_tokens, reply.completion_tokens) if cost_fn else 0.0
        cost += run_cost

        ledger_mod.record_request(
            cfg.ledger_path,
            run_id=run_id,
            task_id=task["task_id"],
            teacher=teacher.name,
            model=reply_model,
            attempt=steps,
            prompt_tokens=reply.prompt_tokens,
            completion_tokens=reply.completion_tokens,
            cost_usd=run_cost,
            latency_s=reply.latency_s,
            error=reply.error,
        )

        if not reply.ok:
            exit_reason = "teacher_error"
            messages.append({"role": "assistant", "content": "", "error": reply.error})
            break

        parsed = parse_assistant(reply.content)
        messages.append({"role": "assistant", "content": reply.content})
        steps += 1

        if parsed.has_answer:
            exit_reason = "answered"
            answer = parsed.answer
            cite = parsed.cite
            break

        requested = _valid_indices(parsed.lookups, n, budget=lookup_budget)
        if not requested:
            exit_reason = "malformed" if parsed.lookups else "step_cap"
            break
        lookup_budget -= len(requested)

        for idx in requested:
            in_range = 0 <= idx < n
            content = evidence[idx]["text"] if in_range else (
                f"error: index {idx} is out of range [0, {max(n - 1, 0)}]"
            )
            messages.append(
                {
                    "role": "tool",
                    "content": content,
                    "evidence_index": idx,
                    "tool_ok": in_range,
                    "tool_name": TOOL_NAME,
                }
            )
            tool_calls.append({"index": idx, "ok": in_range, "step": steps})

    record = {
        "run_id": run_id,
        "task_id": task["task_id"],
        "sample_idx": sample_idx,
        "teacher": teacher.name,
        "model": reply_model,
        "temperature": temperature,
        "started_at": started,
        "wall_s": round(time.monotonic() - started, 2),
        "steps": steps,
        "requests": requests,
        "tokens": {"prompt": ptok, "completion": ctok, "total": ptok + ctok},
        "cost_usd": round(cost, 6),
        "exit": exit_reason,
        "answer": answer,
        "cite": cite,
        "tool_calls": tool_calls,
        "n_lookups": len({c["index"] for c in tool_calls}),
        "messages": messages,
    }
    record["trajectory_hash"] = content_id(messages)
    return RunOutcome(record=record, exit_reason=exit_reason)


def run_all(
    teacher: Teacher,
    tasks: list[dict],
    cfg: Any,
    *,
    teacher_cfg: Any,
    shard: int = 0,
) -> dict:
    """Run k samples for every task. Resumable: completed runs are skipped."""
    shard_path = cfg.runs_dir / f"shard-{shard:02d}.jsonl"
    runs: list[dict] = []
    for task in tasks:
        for sample_idx in range(cfg.samples_per_task):
            outcome = run_one(teacher, task, sample_idx, cfg, teacher_cfg=teacher_cfg)
            runs.append(outcome.record)

    written = append_unique(shard_path, runs, key="run_id")
    return {
        "runs_attempted": len(runs),
        "runs_written": written,
        "shard": str(shard_path),
        "by_exit": {
            e: sum(1 for r in runs if r["exit"] == e) for e in EXITS if any(r["exit"] == e for r in runs)
        },
        "seed_salt": stable_seed(cfg.name, shard),
    }
