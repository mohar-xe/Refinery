"""Context windowing (LLD.md D-010).

Trajectories must fit a 512-token student context. Naive last-N truncation is the
obvious approach and it destroys the signal: the plan is at the front, and the
fix window — the last three steps — is what the model is actually being taught.

Policy for `windowed`:
  (a) system prompt, tool schema, first plan, and ALL tool-call *arguments* stay
      verbatim, always;
  (b) tool *results* older than the active window collapse to a structured
      digest {index, chars, head};
  (c) the final `keep_last_n` messages are never touched.

Arguments are what the model must learn to emit; results are what it must learn
to compress. Truncating either one destroys the corresponding skill — and
compressing the arguments would train the model to emit lossy tool calls, which
is worse than not training on it at all.

`windowing` is recorded per sample so the ablation is a filter over the dataset,
not a re-run of the farm.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = ["apply_windowing", "digest_tool_result", "STRATEGIES"]

STRATEGIES = ("full_context", "windowed", "naive_truncated")

#: Approximate character cost of the digest envelope (JSON keys + quoting).
DIGEST_OVERHEAD_CHARS = 80


def digest_tool_result(msg: dict, head_chars: int) -> str:
    """Structured, lossless-where-it-matters digest of a tool result.

    Note what survives: the index (so the citation still resolves), the length
    (so the model can tell 'long file' from 'short file'), and the head (so
    negation near the start of a sentence is not erased).
    """
    content = msg.get("content", "")
    return json.dumps(
        {
            "digest": "tool_result",
            "index": msg.get("evidence_index"),
            "chars": len(content),
            "head": content[:head_chars],
        },
        sort_keys=True,
    )


def _is_tool_result(msg: dict) -> bool:
    return msg.get("role") == "tool" or "evidence_index" in msg


def apply_windowing(messages: list[dict], cfg: Any) -> list[dict]:
    """Return a possibly-reduced copy of `messages` under the configured policy.

    Never mutates the input: the full-fidelity trajectory is the artifact of
    record, and windowing is a derived view used for training and for the
    ablation arms.
    """
    strategy = getattr(cfg, "strategy", "windowed")
    if strategy == "full_context":
        return list(messages)

    if strategy == "naive_truncated":
        return _naive_truncated(messages, getattr(cfg, "naive_keep_chars", 1200))

    if strategy != "windowed":
        raise ValueError(f"unknown windowing strategy: {strategy!r} (expected one of {STRATEGIES})")

    keep_last = int(getattr(cfg, "keep_last_n", 3))
    head_chars = int(getattr(cfg, "digest_head_chars", 80))
    cut = max(len(messages) - keep_last, 0)

    # Only digest when the digest is actually shorter. Measured on the toy
    # corpus: a retrieved segment averages ~60 chars while the digest envelope
    # costs ~140, so digesting unconditionally made `windowed` samples *longer*
    # than `full_context` — a policy that silently does the opposite of its job.
    # Rule: digest only content longer than the head plus the envelope overhead.
    min_digestable = head_chars + DIGEST_OVERHEAD_CHARS

    out: list[dict] = []
    for i, msg in enumerate(messages):
        # (a) arguments verbatim: assistant turns and the opening prefix are
        #     never rewritten, in-window or not.
        # (c) the final keep_last messages are untouched.
        if i >= cut or not _is_tool_result(msg):
            out.append(dict(msg))
            continue
        if len(msg.get("content", "")) <= min_digestable:
            out.append(dict(msg))  # nothing to gain; compressing would inflate it
            continue
        # (b) old tool results collapse.
        collapsed = dict(msg)
        collapsed["content"] = digest_tool_result(msg, head_chars)
        collapsed["windowed"] = True
        out.append(collapsed)
    return out


def _naive_truncated(messages: list[dict], keep_chars: int) -> list[dict]:
    """The ablation arm: keep the *tail* of the conversation, drop the head.

    This is what a pipeline does when it hits the context limit and has no policy:
    slice the rendered text to the last N characters. Measured on the toy corpus
    this arm is not a no-op — it deletes the system prompt, the task, and every
    retrieved segment, leaving the final answer supervised with no question in
    context. That is the failure the shipping policy exists to avoid, and the
    ablation number to look for is `valid_tool_call_rate` collapsing.

    The budget is a *stated convention*, not a tuned value: it is set to roughly a
    third of the mean rendered trajectory length so the arm is under real pressure.
    Left at the flagship's 24000 the arm is a literal no-op on this corpus, which
    would make the ablation vacuous.

    Note the measured counterpart: `windowed` vs `full_context` *is* a no-op here,
    because toy tool outputs average ~60 chars and the digest only fires above
    `head_chars + 80`. That null result is reported rather than engineered away.
    """
    if len(messages) <= 2:
        return list(messages)

    kept: list[dict] = []
    budget = keep_chars
    for msg in reversed(messages[2:]):
        cost = len(msg.get("content", "")) + 24  # role markers + im_end
        if kept and budget - cost < 0:
            break
        kept.append(msg)
        budget -= cost
    kept.reverse()
    return list(messages[:2]) + kept
