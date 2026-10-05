"""Dataset compiler: verified trajectories -> SFT samples.

Consumes *only* verified trajectories, and emits the segments produced by the
single renderer (LLD.md D-007) plus the `kind` labels the trainer needs for
curriculum stage 1 (LLD.md D-011).

Two outputs that are easy to get wrong and are checked here rather than trusted:

  * `prompt_hash` — hash of the rendered conversation, so format drift between
    training data and live eval prompts is detectable in the field.
  * `windowing` — recorded per sample, which turns the windowing ablation into a
    filter over the dataset rather than a re-run of the farm (the expensive part).

Splits follow the *task* split, never the run: a trajectory from a train task can
never land in the eval set, because the task hash already decided.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from refinery.common.jsonl import read_jsonl, write_json
from refinery.compiler.render import (
    SYSTEM_PROMPT_VERSION,
    Segment,
    prompt_hash,
    segments,
)
from refinery.compiler.window import apply_windowing
from refinery.common.codes import Reason
from refinery.config import WindowCfg

__all__ = ["build_dataset", "sample_stats"]


def _seg_to_dict(seg: Segment) -> dict:
    return {"text": seg.text, "trainable": seg.trainable, "role": seg.role, "kind": seg.kind}


def build_dataset(cfg: Any, *, strategies: list[str] | None = None) -> dict:
    """Build SFT datasets for one or more windowing strategies.

    Writes `dataset/<split>__<strategy>.jsonl` plus `dataset/summary.json`.
    """
    strategies = strategies or [cfg.window.strategy]
    base_window = cfg.window

    tasks = {t["task_id"]: t for t in read_jsonl(cfg.manifest_path)}
    runs: dict[str, dict] = {}
    for shard in sorted(cfg.runs_dir.glob("shard-*.jsonl")):
        for run in read_jsonl(shard):
            runs[run["run_id"]] = run

    verdicts = [v for v in read_jsonl(cfg.verdicts_path) if v["reason_code"] == Reason.VERIFIED_OK.value]

    cfg.dataset_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"strategies": {}, "n_verified": len(verdicts), "compiler_version": "1.0.0"}

    for strategy in strategies:
        window = replace(base_window, strategy=strategy)
        by_split: dict[str, list[dict]] = {"train": [], "eval": []}
        skipped: dict[str, int] = {}

        for verdict in verdicts:
            run = runs.get(verdict["run_id"])
            task = tasks.get(verdict["task_id"])
            if run is None or task is None:
                skipped["missing_run_or_task"] = skipped.get("missing_run_or_task", 0) + 1
                continue

            windowed = apply_windowing(run.get("messages", []), window)
            segs = segments(windowed)
            n_assistant = sum(1 for m in windowed if m.get("role") == "assistant")

            sample = {
                "sample_id": f"{run['run_id']}|{strategy}",
                "task_id": task["task_id"],
                "run_id": run["run_id"],
                "split": task["split"],
                "windowing": strategy,
                "n_steps": n_assistant,
                "n_tool_calls": run.get("n_lookups", 0),
                "variant": task.get("prompt_variant"),
                "label": task.get("label"),
                "system_prompt_version": SYSTEM_PROMPT_VERSION,
                "prompt_hash": prompt_hash(windowed),
                "segments": [_seg_to_dict(s) for s in segs],
                "chars": sum(len(s.text) for s in segs),
                "trainable_chars": sum(len(s.text) for s in segs if s.trainable),
                "tool_call_chars": sum(len(s.text) for s in segs if s.kind == "tool_call"),
            }
            by_split.setdefault(task["split"], []).append(sample)

        for split, samples in by_split.items():
            path = cfg.dataset_dir / f"{split}__{strategy}.jsonl"
            path.write_text(
                "".join(json.dumps(s, sort_keys=True, ensure_ascii=False) + "\n" for s in samples),
                encoding="utf-8",
            )

        summary["strategies"][strategy] = {
            "n_train": len(by_split["train"]),
            "n_eval": len(by_split["eval"]),
            "skipped": skipped,
            "stats": sample_stats([s for samples in by_split.values() for s in samples]),
            "files": {
                split: str(cfg.dataset_dir / f"{split}__{strategy}.jsonl") for split in by_split
            },
        }

    write_json(cfg.dataset_dir / "summary.json", summary)
    return summary


def sample_stats(samples: list[dict]) -> dict:
    """Descriptive stats that predict training behaviour before training starts."""
    if not samples:
        return {}
    n = len(samples)
    return {
        "n": n,
        "avg_steps": round(sum(s["n_steps"] for s in samples) / n, 2),
        "max_steps": max(s["n_steps"] for s in samples),
        "avg_chars": round(sum(s["chars"] for s in samples) / n, 1),
        "avg_trainable_chars": round(sum(s["trainable_chars"] for s in samples) / n, 1),
        "avg_tool_call_chars": round(sum(s["tool_call_chars"] for s in samples) / n, 1),
        "by_variant": {
            v: sum(1 for s in samples if s["variant"] == v)
            for v in sorted({s["variant"] for s in samples if s.get("variant")})
        },
        "unique_prompt_hashes": len({s["prompt_hash"] for s in samples}),
    }
