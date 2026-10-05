"""Format-fidelity tests — the guard for spec trap #2.

The claim under test: *the training text is byte-identical to what the harness
sends at inference time*. If these fail, every downstream number is meaningless
and the failure mode is silent (the model just gets worse), so they are the first
tests to run in CI.
"""

from __future__ import annotations

import json

import pytest

from refinery.common.protocol import parse_assistant
from refinery.compiler.render import (
    render_messages,
    seed_messages,
    segments,
    task_prompt,
)

TASK = {
    "task_id": "sha256:test",
    "hypothesis": "A boy is playing a guitar.",
    "label": "entailed",
    "split": "train",
    "prompt_variant": "base",
    "evidence": [
        {"evidence_id": "e0", "index": 0, "text": "A boy sits on a stool."},
        {"evidence_id": "e1", "index": 1, "text": "He strums a guitar."},
    ],
}


def _trajectory() -> list[dict]:
    msgs = seed_messages(TASK)
    msgs.append({"role": "assistant", "content": "PLAN: read both segments.\nlookup(0) lookup(1)"})
    msgs.append(
        {"role": "tool", "content": "A boy sits on a stool.", "evidence_index": 0, "tool_ok": True}
    )
    msgs.append(
        {"role": "tool", "content": "He strums a guitar.", "evidence_index": 1, "tool_ok": True}
    )
    msgs.append(
        {"role": "assistant", "content": "PLAN: compare with the hypothesis.\nCITE: 0, 1\nANSWER: entailed"}
    )
    return msgs


def test_segments_concatenate_to_the_rendered_text():
    """Lossless: segment texts rejoin to exactly the rendered conversation."""
    msgs = _trajectory()
    segs = segments(msgs)
    assert "".join(s.text for s in segs) == render_messages(msgs)


def test_renderer_is_deterministic():
    msgs = _trajectory()
    assert render_messages(msgs) == render_messages(list(msgs))


def test_renderer_does_not_mutate_its_input():
    msgs = _trajectory()
    before = json.dumps(msgs, sort_keys=True)
    render_messages(msgs)
    assert json.dumps(msgs, sort_keys=True) == before


def test_document_is_absent_from_the_prompt():
    """If the document leaked into the prompt, the tool would be decorative and
    every tool-call metric would be meaningless."""
    prompt = task_prompt(TASK)
    for unit in TASK["evidence"]:
        assert unit["text"] not in prompt
    assert TASK["hypothesis"] in prompt


def test_assistant_segments_are_classified_by_kind():
    msgs = _trajectory()
    segs = segments(msgs)
    kinds = {s.kind for s in segs}
    assert {"tool_call", "prose", "answer", "context"} <= kinds
    # Exactly the lookup line is a tool_call segment.
    tool_segs = [s for s in segs if s.kind == "tool_call"]
    assert all("lookup(" in s.text for s in tool_segs)
    assert sum(len(s.text) for s in tool_segs) == len("lookup(0) lookup(1)")


def test_only_assistant_content_is_trainable():
    msgs = _trajectory()
    segs = segments(msgs)
    for seg in segs:
        if seg.trainable:
            assert seg.role == "assistant"
        else:
            assert seg.role != "assistant" or seg.kind == "context"


def test_parser_accepts_exactly_what_the_renderer_emits():
    """The renderer/parser contract. If the assistant grammar changes on one side
    only, this is what notices (it is how the stale `sentences`/`segments`
    mismatch was caught)."""
    msgs = _trajectory()
    for msg in msgs:
        if msg["role"] != "assistant":
            continue
        parsed = parse_assistant(msg["content"])
        if "CITE:" in msg["content"]:
            assert parsed.cite_present
            assert parsed.cite == [0, 1]
        if "ANSWER:" in msg["content"]:
            assert parsed.answer == "entailed"


def test_prompt_hash_changes_when_the_prompt_changes():
    from refinery.compiler.render import prompt_hash

    msgs = _trajectory()
    altered = [dict(m) for m in msgs]
    altered[1] = {**altered[1], "content": altered[1]["content"] + " (edited)"}
    assert prompt_hash(msgs) != prompt_hash(altered)
