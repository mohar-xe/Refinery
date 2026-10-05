"""Generated reports — the only source of numbers for the README.

Every file here is written by `scripts/run_toy.sh`. Nothing in `reports/` is
hand-edited, and the README table is transcribed from `frontier.md`. That rule is
what makes "publish whatever is honest" enforceable rather than aspirational.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from refinery.common.codes import REASON_CODES, REWARD_HACK_CLASSES
from refinery.common.jsonl import read_json, read_jsonl
from refinery.farm import ledger as ledger_mod

__all__ = ["write_all"]


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_all(cfg: Any, *, summaries: list[dict] | None = None, out_dir: Path | None = None) -> dict:
    out_dir = out_dir or (cfg.repo_root / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    written["histogram"] = _histogram(cfg, out_dir)
    written["bill"] = _bill(cfg, out_dir)
    if summaries:
        from evalkit.frontier import write_frontier

        written.update(write_frontier(summaries, out_dir))
        written["ablations"] = _ablations(summaries, out_dir)
    return written


def _histogram(cfg: Any, out_dir: Path) -> str:
    hist = read_json(cfg.root / "verdicts_summary.json", {}) or {}
    if not hist:
        return ""

    verdicts = list(read_jsonl(cfg.verdicts_path))
    by_variant: dict[str, dict[str, int]] = {}
    for v in verdicts:
        bucket = by_variant.setdefault(v.get("variant") or "?", {})
        bucket[v["reason_code"]] = bucket.get(v["reason_code"], 0) + 1

    total = hist["total"]
    lines = [
        "# Rejection histogram",
        "",
        f"Generated from `{cfg.verdicts_path}` — {total} runs, schema "
        f"`codes.ORDER` (first match wins, so the codes partition the runs).",
        "",
        "| Reason code | Count | Share | Class |",
        "|---|---|---|---|",
    ]
    for code in REASON_CODES:
        count = hist["counts"].get(code, 0)
        if not count and code == "VERIFIED_OK":
            count = hist["counts"].get(code, 0)
        klass = "hack" if code in {r.value for r in REWARD_HACK_CLASSES} else (
            "accept" if code == "VERIFIED_OK" else "reject"
        )
        lines.append(f"| `{code}` | {count} | {_fmt(hist['shares'].get(code, 0.0), 4)} | {klass} |")

    lines += [
        "",
        f"**Verified:** {hist['verified']}/{total} ({_fmt(hist['verify_rate'], 4)})",
        "",
        "## What the label check alone would have accepted",
        "",
        f"- Runs whose predicted label matched gold: **{hist['label_check_would_accept']}**",
        f"- Of those, rejected by an anti-hack filter: **{hist['rejected_despite_correct_label']}**",
        f"- Reward-hack rate among label-passes: **{_fmt(hist['reward_hack_rate_of_label_passes'], 4)}**",
        "",
        "> This ratio is the toy-scale stand-in for the flagship's \"14% of 'passing'",
        "> trajectories were reward hacks\". It is a *floor*: our filters only catch",
        "> provable defects (see `verifier/antihack.py`), because a stricter filter would",
        "> silently delete hard examples rather than catch more hacks.",
        "",
        "## Rejections by adversarial variant",
        "",
        "| Variant | " + " | ".join(sorted({c for b in by_variant.values() for c in b})) + " |",
        "|---" * (1 + len({c for b in by_variant.values() for c in b})) + "|",
    ]
    all_codes = sorted({c for b in by_variant.values() for c in b})
    for variant, counts in sorted(by_variant.items()):
        lines.append(f"| {variant} | " + " | ".join(str(counts.get(c, 0)) for c in all_codes) + " |")

    path = out_dir / "rejection_histogram.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _bill(cfg: Any, out_dir: Path) -> str:
    summary = ledger_mod.summarize(cfg.ledger_path, verdicts_path=cfg.verdicts_path)
    lines = [
        "# The bill",
        "",
        "The teacher is a free model, so the dollar figure is `$0.00` and the real",
        "quantities are printed beside it rather than replaced by an invented price",
        "(`LLD.md` D-013). The interesting numbers are efficiency: what it costs to",
        "produce one *verified* trajectory, and how much is wasted on rejected runs.",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| API cost | **$0.00** (free tier) |",
        f"| Teacher requests | {summary.get('requests')} |",
        f"| Prompt tokens | {summary.get('prompt_tokens')} |",
        f"| Completion tokens | {summary.get('completion_tokens')} |",
        f"| Total tokens | {summary.get('tokens')} |",
        f"| Teacher wall time | {summary.get('latency_s')}s |",
        f"| Mean latency / request | {_fmt(summary.get('mean_latency_s'), 2)}s |",
        f"| Provider errors | {summary.get('errors')} ({_fmt(summary.get('error_rate'), 4)}) |",
        f"| Verified trajectories | {summary.get('verified_trajectories', '—')} |",
        f"| Tokens per verified trajectory | {_fmt(summary.get('tokens_per_verified'), 1)} |",
        f"| Requests per verified trajectory | {_fmt(summary.get('requests_per_verified'), 2)} |",
        f"| Tokens spent on rejected runs | {summary.get('tokens_wasted_on_rejected', '—')} |",
        f"| Waste share of all tokens | {_fmt(summary.get('waste_share'), 4)} |",
        "",
        "## Per-run caps (enforced before each request)",
        "",
        f"- max assistant steps: {cfg.caps.max_steps}",
        f"- max requests: {cfg.caps.max_requests}",
        f"- max tokens: {cfg.caps.max_tokens}",
    ]
    path = out_dir / "bill.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _ablations(summaries: list[dict], out_dir: Path) -> str:
    by_name = {s["contender"]: s for s in summaries}

    lines = ["# Ablations", "", "One decision, run both ways, numbers on both sides.", ""]

    # --- windowing -----------------------------------------------------------
    win = {k: v for k, v in by_name.items() if k.startswith("win_")}
    if win:
        lines += [
            "## Windowing strategy (data compiler)",
            "",
            "| Arm | pass@1 | valid tool calls | mean tokens |",
            "|---|---|---|---|",
        ]
        for name, s in sorted(win.items()):
            lines.append(
                f"| `{name.replace('win_', '')}` | {_fmt(s['pass@1'])} | "
                f"{_fmt(s['valid_tool_call_rate'])} | {_fmt(s['mean_tokens'], 1)} |"
            )
        lines += [
            "",
            "`full_context` is the upper bound where trajectories fit the student",
            "context; `windowed` is the shipping policy (arguments verbatim, old tool",
            "results digested, last 3 steps untouched); `naive_truncated` keeps the head of",
            "the conversation and drops the tail, which should lose the fix window.",
            "",
        ]

    # --- curriculum ----------------------------------------------------------
    stu, no_cur = by_name.get("student"), by_name.get("student_no_curriculum")
    if stu and no_cur:
        lines += [
            "## Curriculum (short tool-call-only stage first)",
            "",
            "| Arm | pass@1 | valid tool calls |",
            "|---|---|---|",
            f"| curriculum | {_fmt(stu['pass@1'])} | {_fmt(stu['valid_tool_call_rate'])} |",
            f"| mixed from scratch | {_fmt(no_cur['pass@1'])} | {_fmt(no_cur['valid_tool_call_rate'])} |",
            "",
            "The metric to watch is `valid tool calls`: spec trap #5 is a *syntactic*",
            "failure, so it shows up there before it shows up in accuracy.",
            "",
        ]

    path = out_dir / "ablations.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)
