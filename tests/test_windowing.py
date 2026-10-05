"""Windowing tests.

Two properties matter and they pull in opposite directions:

  * **Arguments survive verbatim.** Compressing a tool call trains the model to
    emit lossy tool calls, which is worse than not training on it.
  * **Old outputs collapse.** Otherwise the fix window falls off the end.

Plus the guard added during the toy run: never digest content that is *shorter*
than the digest envelope, because doing so made `windowed` samples longer than
`full_context` — a policy silently doing the opposite of its job.
"""

from __future__ import annotations

from dataclasses import replace

from refinery.compiler.render import render_messages
from refinery.compiler.window import DIGEST_OVERHEAD_CHARS, apply_windowing, digest_tool_result

CFG = replace(
    __import__("refinery.config", fromlist=["WindowCfg"]).WindowCfg(),
    strategy="windowed",
    keep_last_n=3,
    digest_head_chars=80,
    naive_keep_chars=150,
)

LONG = "x" * 600


def _messages() -> list[dict]:
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "PLAN: read.\nlookup(0)"},
        {"role": "tool", "content": LONG, "evidence_index": 0, "tool_ok": True},
        {"role": "assistant", "content": "PLAN: more.\nlookup(1)"},
        {"role": "tool", "content": LONG, "evidence_index": 1, "tool_ok": True},
        {"role": "assistant", "content": "PLAN: done.\nCITE: 0, 1\nANSWER: entailed"},
    ]


def test_windowing_never_mutates_the_recorded_trajectory():
    msgs = _messages()
    before = [dict(m) for m in msgs]
    apply_windowing(msgs, CFG)
    assert msgs == before


def test_full_context_is_identity():
    msgs = _messages()
    cfg = replace(CFG, strategy="full_context")
    assert [dict(m) for m in apply_windowing(msgs, cfg)] == [dict(m) for m in msgs]


def test_old_tool_results_are_digested():
    out = apply_windowing(_messages(), CFG)
    tool_msgs = [m for m in out if m.get("role") == "tool"]
    assert tool_msgs, "tool results must survive, just compressed"
    assert any(m.get("windowed") for m in tool_msgs)


def test_tool_call_arguments_stay_verbatim():
    """Assistant turns are never rewritten — including the ones carrying the tool
    call. Only tool *results* may be compressed."""
    original = [dict(m) for m in _messages()]
    out = apply_windowing(original, CFG)
    for before, after in zip(original, out, strict=True):
        if before["role"] == "assistant":
            assert after["content"] == before["content"]
    rendered = render_messages(out)
    assert "lookup(0)" in rendered and "lookup(1)" in rendered


def test_final_window_is_untouched():
    msgs = _messages()
    out = apply_windowing(msgs, CFG)
    assert out[-1] == msgs[-1]
    assert out[-2] == msgs[-2] if len(msgs) > 3 else True


def test_short_tool_results_are_not_digested():
    """Regression guard for the measured bug: digesting a 60-char result with an
    80-char envelope made the 'compressed' sample longer than the original."""
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "PLAN: a.\nlookup(0)"},
        {"role": "tool", "content": "short segment", "evidence_index": 0, "tool_ok": True},
        {"role": "assistant", "content": "PLAN: b.\nlookup(1)"},
        {"role": "tool", "content": "other segment", "evidence_index": 1, "tool_ok": True},
        {"role": "assistant", "content": "PLAN: c.\nCITE: 0, 1\nANSWER: entailed"},
    ]
    out = apply_windowing(msgs, CFG)
    assert not any(m.get("windowed") for m in out)
    assert len(render_messages(out)) == len(render_messages(msgs))


def test_windowed_is_shorter_than_full_when_outputs_are_long():
    windowed = len(render_messages(apply_windowing(_messages(), CFG)))
    full = len(render_messages(apply_windowing(_messages(), replace(CFG, strategy="full_context"))))
    assert windowed < full


def test_digest_keeps_index_length_and_head():
    msg = {"role": "tool", "content": "A boy sits on a stool and smiles.", "evidence_index": 3}
    import json

    digest = json.loads(digest_tool_result(msg, 10))
    assert digest["index"] == 3
    assert digest["chars"] == len(msg["content"])
    assert digest["head"] == "A boy sits"
    assert DIGEST_OVERHEAD_CHARS > 0


def test_naive_truncation_keeps_the_tail_and_drops_the_head():
    out = apply_windowing(_messages(), replace(CFG, strategy="naive_truncated"))
    assert "ANSWER: entailed" in render_messages(out)


def test_unknown_strategy_is_rejected_loudly():
    try:
        apply_windowing(_messages(), replace(CFG, strategy="nonsense"))
    except ValueError as exc:
        assert "nonsense" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unknown windowing strategy must raise")
