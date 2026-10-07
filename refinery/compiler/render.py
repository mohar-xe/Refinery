"""Dataset compiler: deterministic rendering of prompts and conversations.

**Single owner of the format** (LLD.md D-007). The farm imports `render_messages`
to build the prompt it actually sends; the compiler imports the *same function*
to build training text. A single builder makes format drift structurally
impossible, which is stronger than a parity test that catches drift after the
fact — and drift is ~60% of this project's failure probability (spec trap #2).

Invariants:
  * Pure function of its inputs. No timestamps, no randomness, no dict ordering.
  * Loss masking falls out of the segmentation: assistant content and the
    terminating `<|im_end|>` are trainable; everything else is context.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from refinery.common.hashing import content_id
from refinery.common.protocol import TOOL_NAME, answers_are_freeform

__all__ = [
    "SYSTEM_PROMPT",
    "URL_SYSTEM_PROMPT",
    "LEGAL_SYSTEM_PROMPT",
    "TOOL_SCHEMA",
    "TOOL_SCHEMA_JSON",
    "MAX_LOOKUPS",
    "Segment",
    "task_prompt",
    "seed_messages",
    "segments",
    "render_messages",
    "prompt_hash",
    "prompt_version",
]

MAX_LOOKUPS = 4  # total segments an agent may retrieve per run

#: Verbatim. Changing this string changes every prompt in the corpus, so it is a
#: versioned constant: `SYSTEM_PROMPT_VERSION` is recorded on every training
#: sample and mixed versions in one dataset are a bug, not a detail.
SYSTEM_PROMPT_VERSION = "v3"

#: Kept deliberately short. v2 was ~300 tokens — 62% of every training sequence —
#: and it is *masked out of the loss*, so all of it was compute spent on a constant
#: prefix with no learning signal. Same protocol, ~40% of the tokens. Any change
#: here changes every prompt in the corpus, hence the version bump.
SYSTEM_PROMPT = f"""You are a reading agent. Decide whether a hypothesis follows from a document.

The document is hidden and split into numbered segments. Retrieve segments with lookup(index); you may write several lookup() calls in one turn. At most {MAX_LOOKUPS} lookups per task.

End your reply with exactly:
CITE: <indices you used>
ANSWER: entailed | not_entailed

entailed = must be true given the document. Contradicted, merely possible, or unrelated = not_entailed.
Word overlap alone proves nothing; negation matters."""

#: URL-reconstruction prompt. Separate version string because the two task
#: families have genuinely different prompts, and mixing them in one dataset
#: would make `system_prompt_version` useless for detecting drift.
URL_SYSTEM_PROMPT_VERSION = "v1"

URL_SYSTEM_PROMPT = f"""You are a URL repair agent. You reconstruct a damaged URL.

Protocol:
- A hidden list of candidate URLs exists. You cannot see it; retrieve it with the tool.
- Retrieve a candidate by writing lookup(index), where index is an integer in [0, N).
  You may write several lookup() calls in one turn.
- You may retrieve at most {MAX_LOOKUPS} candidates in the whole task.
- Damage includes dropped schemes, lost percent-encoding, HTML-escaped characters,
  sentence punctuation, reordered query parameters, and truncation.
- End your reply with exactly these two lines and nothing after them:
    CITE: <comma-separated candidate indices you used>
    ANSWER: <the reconstructed URL>

Copy the URL exactly as it should be, not the damaged form. Do not add a scheme, a www prefix,
or a trailing slash that the candidates do not have."""

#: Legal-claim verification prompt.
LEGAL_SYSTEM_PROMPT_VERSION = "v1"

LEGAL_SYSTEM_PROMPT = f"""You are a legal verification analyst. You decide whether a claim is entailed by the propositions in a judgment.

Protocol:
- The propositions are hidden. You cannot see them; retrieve them with the tool.
- Retrieve a proposition by writing lookup(index), where index is an integer in [0, N).
  You may write several lookup() calls in one turn.
- You may retrieve at most {MAX_LOOKUPS} propositions in the whole task.
- End your reply with exactly these two lines and nothing after them:
    CITE: <comma-separated proposition indices you relied on>
    ANSWER: entailed | not_entailed

entailed = the propositions you retrieved actually establish the claim.
Report not_entailed when a proposition is about a different court, a different bench, a
different year or a different party; when it asserts the opposite; when it holds only for
one bench and the claim generalises to all courts; or when a claim needs a proposition you
could not find. Do not answer entailed from memory of the case - only from what you retrieved."""

TOOL_SCHEMA: dict = {
    "name": "lookup",
    "description": "Reveal one numbered sentence of the hidden document.",
    "parameters": {
        "type": "object",
        "properties": {
            "index": {
                "type": "integer",
                "minimum": 0,
                "description": "Zero-based sentence index.",
            }
        },
        "required": ["index"],
    },
}

#: Compact (no indentation): this string is re-sent on every request and re-tokenised
#: in every training sample, so whitespace is not free.
TOOL_SCHEMA_JSON = json.dumps(TOOL_SCHEMA, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class Segment:
    """One contiguous piece of the rendered conversation.

    `kind` exists so the trainer can supervise *different parts* of an assistant
    turn differently. The curriculum's stage 1 supervises `tool_call` segments
    only (LLD.md D-011), which requires the split to happen here — the single
    owner of the format — rather than in the trainer with a second regex.
    """

    text: str
    trainable: bool
    role: str
    kind: str = "context"


_TOOL_CALL_LINE = re.compile(rf"\b{TOOL_NAME}\s*\(", re.IGNORECASE)
_FINAL_ANSWER_LINE = re.compile(r"^\s*(CITE|ANSWER)\s*:", re.IGNORECASE)


def _assistant_segments(content: str) -> list[Segment]:
    """Split one assistant turn into prose / tool-call / answer segments.

    Lossless: the concatenation of the segment texts is byte-identical to
    `content`. `tests/test_format_parity.py` asserts that.
    """
    segs: list[Segment] = []
    if not content:
        return [Segment("", True, "assistant", "prose")]
    for line in content.splitlines(keepends=True):
        if _TOOL_CALL_LINE.search(line):
            kind = "tool_call"
        elif _FINAL_ANSWER_LINE.match(line):
            kind = "answer"
        else:
            kind = "prose"
        segs.append(Segment(line, True, "assistant", kind))
    return segs


def segments(messages: list[dict]) -> list[Segment]:
    """Render a conversation into segments, marking which are trainable.

    Masking policy (LLD.md D-012): loss on assistant-authored tokens only —
    reasoning, the `lookup(...)` call, and the CITE/ANSWER block. Tool results
    and the system prompt are context. Supervising tool output teaches the model
    to hallucinate the environment, which is the single most common way tool-use
    SFT goes wrong.
    """
    segs: list[Segment] = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        if role == "assistant":
            segs.append(Segment("<|im_start|>assistant\n", False, role, "context"))
            segs.extend(_assistant_segments(content))
            segs.append(Segment("<|im_end|>", True, role, "answer"))
            segs.append(Segment("\n", False, role, "context"))
        else:
            segs.append(Segment(f"<|im_start|>{role}\n{content}<|im_end|>\n", False, role, "context"))
    return segs


#: task kind -> (system prompt, version). One table, so the harness and the compiler
#: can never disagree about which prompt family a task belongs to.
PROMPTS = {
    "url": (None, URL_SYSTEM_PROMPT_VERSION),
    "legal": (None, LEGAL_SYSTEM_PROMPT_VERSION),
    "entailment": (None, SYSTEM_PROMPT_VERSION),
}


def prompt_version(task: dict) -> str:
    """Which prompt generation a task belongs to."""
    return PROMPTS.get(task.get("kind", "entailment"), PROMPTS["entailment"])[1]


def system_prompt_for(task: dict) -> str:
    kind = task.get("kind", "entailment")
    if kind == "url":
        return URL_SYSTEM_PROMPT
    if kind == "legal":
        return LEGAL_SYSTEM_PROMPT
    return SYSTEM_PROMPT


def task_prompt(task: dict) -> str:
    """The user turn for a task.

    Note what is *absent*: the document. Only the input and the candidate count
    are visible, so retrieval through `lookup` is semantically required rather
    than decorative (LLD.md D-003). A tool the model could route around would
    make the tool-call curriculum and the windowing policy untestable.

    Dispatch on task kind happens *here*, so there is still exactly one function
    that decides what an agent sees (LLD.md D-007).
    """
    n = len(task.get("evidence", []))
    valid = f"0-{max(n - 1, 0)}"
    if task.get("kind") == "legal":
        return (
            f"CLAIM: {task['claim']}\n"
            f"PROPOSITIONS: {n} from one judgment, hidden. Valid indices {valid}.\n"
            f"Decide whether the retrieved propositions entail the claim."
        )
    if answers_are_freeform(task):
        return (
            f"DAMAGED URL: {task['broken']}\n"
            f"CANDIDATES: {n} candidate URLs, hidden. Valid indices {valid}.\n"
            f"Reconstruct the original URL."
        )
    return (
        f"HYPOTHESIS: {task['hypothesis']}\n"
        f"DOCUMENT: {n} segments, hidden. Valid indices {valid}.\n"
        f"Decide whether the hypothesis must be true given the document."
    )


def _system_turn(task: dict) -> dict:
    return {
        "role": "system",
        "content": f"{system_prompt_for(task)}\n\nTOOLS:\n{TOOL_SCHEMA_JSON}",
        "system_prompt_version": prompt_version(task),
    }


def seed_messages(task: dict) -> list[dict]:
    """The prefix every trajectory starts from, in both farm and compiler.

    Built by the renderer so the farm cannot invent its own opening move.
    """
    return [_system_turn(task), {"role": "user", "content": task_prompt(task)}]


def render_messages(messages: list[dict]) -> str:
    """Full deterministic rendering. Identical inputs -> identical string, always."""
    return "".join(seg.text for seg in segments(messages))


def prompt_hash(messages: list[dict]) -> str:
    """Hash of the rendered conversation.

    Recorded on every training sample so that format drift is detectable *in the
    field*: hash the live eval prompt and compare distributions against the
    training set before blaming the model for a quality regression.
    """
    return content_id(render_messages(messages))
