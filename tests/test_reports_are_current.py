"""No stale numbers in the repository.

The failure this guards against is one of my own making: `reports/` is committed
(it is the published artifact), so a report generated for a superseded
configuration stays in the repo and describes a run directory the active config no
longer uses. That happened — the committed histogram described `runs/toy` while the
active task was legal — and a reader would reasonably believe it.

The rule: any committed report must name the run directory of the configuration
currently marked active. No reports is the correct state before the first run, so
an empty directory passes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
REPORTS = REPO / "reports"
#: The config a reader is expected to reproduce. Changing the active task means
#: changing this line, and regenerating every report.
ACTIVE_CONFIG = REPO / "configs" / "legal.json"

_SOURCE_RE = re.compile(r"`(/[^`]*?/runs/[a-z]+)/(?:verdicts|results)[^`]*`")


def _committed_reports() -> list[Path]:
    if not REPORTS.is_dir():
        return []
    return sorted(p for p in REPORTS.glob("*.md") if p.name != "README.md")


def test_no_stale_reports_are_committed():
    cfg = json.loads(ACTIVE_CONFIG.read_text(encoding="utf-8"))
    expected_root = cfg["paths"]["root"]

    stale: list[str] = []
    for path in _committed_reports():
        text = path.read_text(encoding="utf-8")
        for found in _SOURCE_RE.findall(text):
            if not found.endswith(expected_root):
                stale.append(f"{path.name}: references {found}, active config root is {expected_root}")
    assert not stale, "stale reports committed:\n  " + "\n  ".join(stale)


def test_reports_directory_is_not_empty_of_guidance():
    """reports/README.md must survive even when every generated report is absent."""
    assert (REPORTS / "README.md").exists(), "reports/README.md was deleted"
