"""The wire protocol between agent, tools, and verifier.

Parsing lives here rather than in `compiler/render.py` on purpose: rendering has a
single owner (LLD.md D-007) because *drift* is the danger, and parsing is a
separate concern with a separate owner. The two are coupled by
`tests/test_protocol_render_contract.py`, which asserts the parser accepts
exactly what the renderer produces.

Any change to the assistant's output grammar must change both sides, and the
harness must never hand-roll its own regex — a second parser is how a run gets
recorded in a format the verifier cannot read, which shows up as mystery
`FORMAT_INVALID` rejections.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = ["ANSWER_LABELS", "ParsedOutput", "parse_assistant", "format_answer", "TOOL_NAME"]

TOOL_NAME = "lookup"
ANSWER_LABELS: tuple[str, ...] = ("entailed", "not_entailed")

_LOOKUP_RE = re.compile(rf"\b{TOOL_NAME}\s*\(\s*(-?\d+)\s*\)")
_CITE_RE = re.compile(r"^\s*CITE\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)
_ANSWER_RE = re.compile(r"^\s*ANSWER\s*:\s*(\S+)\s*$", re.MULTILINE | re.IGNORECASE)


@dataclass
class ParsedOutput:
    """Structured view of one assistant turn."""

    lookups: list[int] = field(default_factory=list)
    cite: list[int] = field(default_factory=list)
    answer: str | None = None
    cite_present: bool = False
    answer_present: bool = False
    error: str | None = None

    @property
    def has_answer(self) -> bool:
        return self.answer in ANSWER_LABELS

    @property
    def ok(self) -> bool:
        return self.has_answer and self.cite_present and self.error is None


def parse_assistant(text: str) -> ParsedOutput:
    """Parse one assistant turn. Never raises — malformed output is data, not a crash.

    The final answer must be the *last* `ANSWER:` line: an agent that writes
    "ANSWER: not_entailed ... ANSWER: entailed" is malformed, and accepting the
    first match would let a stream of self-corrections launder a guess into a
    verified trajectory.
    """
    out = ParsedOutput()
    if not text or not text.strip():
        out.error = "empty_assistant_turn"
        return out

    out.lookups = [int(m) for m in _LOOKUP_RE.findall(text)]

    cite_matches = _CITE_RE.findall(text)
    if cite_matches:
        out.cite_present = True
        raw = cite_matches[-1]
        items: list[int] = []
        for token in raw.replace(";", ",").split(","):
            token = token.strip().strip("[]()")
            if token.isdigit():
                items.append(int(token))
        out.cite = items

    answer_matches = _ANSWER_RE.findall(text)
    if answer_matches:
        out.answer_present = True
        candidate = answer_matches[-1].strip().strip(".").lower()
        out.answer = candidate if candidate in ANSWER_LABELS else candidate

    if not out.answer_present:
        out.error = "missing_ANSWER_line"
    elif not out.has_answer:
        out.error = f"unknown_label:{out.answer}"
    elif not out.cite_present:
        out.error = "missing_CITE_line"
    elif not out.cite:
        out.error = "empty_CITE"

    return out


def format_answer(label: str, cite: list[int]) -> str:
    """Canonical final-answer block (used by the harness when recording)."""
    return f"CITE: {', '.join(str(i) for i in cite)}\nANSWER: {label}"
