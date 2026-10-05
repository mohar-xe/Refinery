"""Frontier report: markdown table + a dependency-free SVG plot.

No matplotlib. A 40-line SVG writer keeps the repo installable on a bare
interpreter and keeps `reports/` diffable in review — the plot is regenerated from
`eval/results.jsonl` on every run, and a binary PNG would hide that.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from evalkit.metrics import frontier_rows

__all__ = ["write_frontier", "render_svg"]

_COLORS = {
    "teacher": "#e8590c",
    "student": "#1971c2",
    "student_no_curriculum": "#9c36b5",
    "heuristic": "#868e96",
}


def write_frontier(summaries: list[dict], out_dir: Path, *, extra: dict[str, Any] | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = frontier_rows(summaries)

    header = (
        "| Contender | Params | pass@1 | Valid tool calls | Cost / resolved | p50 latency | p95 latency | Mean steps |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for r in rows:
        cost = r["cost_per_resolved_task_usd"]
        lines.append(
            f"| {r['contender']} | {r['params']} | {r['pass@1']:.3f} | "
            f"{r['valid_tool_call_rate']:.3f} | "
            f"{'$' + format(cost, '.6f') if cost is not None else '—'} | "
            f"{r['latency_p50_s']}s | {r['latency_p95_s']}s | {r['mean_steps']} |"
        )

    body = [header, "\n".join(lines), ""]
    if extra:
        body.append("## Notes\n")
        for k, v in extra.items():
            body.append(f"- **{k}**: {v}")
        body.append("")

    by_variant = _variant_block(summaries)
    if by_variant:
        body.append(by_variant)

    (out_dir / "frontier.md").write_text("\n".join(body), encoding="utf-8")
    (out_dir / "frontier.svg").write_text(render_svg(rows), encoding="utf-8")
    return {"frontier_md": str(out_dir / "frontier.md"), "frontier_svg": str(out_dir / "frontier.svg")}


def _variant_block(summaries: list[dict]) -> str:
    lines = ["## pass@1 by adversarial variant", "", "| Contender | "]
    variants: list[str] = []
    for s in summaries:
        for v in (s.get("by_variant") or {}):
            if v not in variants:
                variants.append(v)
    lines[0] += " | ".join(variants) + " |"
    lines.append("|---" * (len(variants) + 1) + "|")
    for s in summaries:
        cells = []
        for v in variants:
            cell = (s.get("by_variant") or {}).get(v)
            cells.append(f"{cell['pass@1']:.3f}" if cell else "—")
        lines.append(f"| {s['contender']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_svg(rows: list[dict], *, width: int = 760, height: int = 420) -> str:
    """pass@1 vs p50 latency, one dot per contender. Hand-rolled on purpose."""
    if not rows:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"></svg>'

    pad_l, pad_r, pad_t, pad_b = 70, 30, 40, 70
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    xs = [r["latency_p50_s"] or 0.0 for r in rows]
    ys = [r["pass@1"] for r in rows]
    x_min, x_max = 0.0, max(xs) * 1.25 or 1.0
    y_min, y_max = 0.0, min(1.0, max(ys) * 1.25 + 0.05)

    def sx(v: float) -> float:
        return pad_l + (v - x_min) / (x_max - x_min) * plot_w

    def sy(v: float) -> float:
        return pad_t + plot_h - (v - y_min) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Helvetica,Arial,sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{pad_l}" y="24" font-size="16" font-weight="600" fill="#1e1e1e">'
        "pass@1 vs p50 latency — cost/quality frontier</text>",
    ]

    for i in range(5):
        gy = pad_t + plot_h * i / 4
        val = y_max - (y_max - y_min) * i / 4
        parts.append(
            f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{pad_l + plot_w}" y2="{gy:.1f}" '
            'stroke="#e9ecef" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_l - 10}" y="{gy + 4:.1f}" font-size="11" fill="#868e96" '
            f'text-anchor="end">{val:.2f}</text>'
        )

    parts.append(
        f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" y2="{pad_t + plot_h}" '
        'stroke="#1e1e1e" stroke-width="1.5"/>'
    )
    parts.append(
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" '
        'stroke="#1e1e1e" stroke-width="1.5"/>'
    )
    parts.append(
        f'<text x="{pad_l + plot_w / 2:.0f}" y="{height - 22}" font-size="12" fill="#495057" '
        'text-anchor="middle">p50 latency per task (seconds)</text>'
    )
    parts.append(
        f'<text x="18" y="{pad_t + plot_h / 2:.0f}" font-size="12" fill="#495057" '
        f'text-anchor="middle" transform="rotate(-90 18 {pad_t + plot_h / 2:.0f})">pass@1</text>'
    )

    for r in rows:
        x, y = sx(r["latency_p50_s"] or 0.0), sy(r["pass@1"])
        color = _COLORS.get(r["contender"], "#495057")
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}" fill-opacity="0.85"/>')
        parts.append(
            f'<text x="{x + 12:.1f}" y="{y + 4:.1f}" font-size="12" fill="#1e1e1e">'
            f"{html.escape(r['contender'])}</text>"
        )

    parts.append("</svg>")
    return "\n".join(parts)
