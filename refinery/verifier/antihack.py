"""Anti-hack filters: the checks that make "pass rate" mean something.

Everything here is a **filter on the trajectory**, not on the model. A verifier
that only checks the final answer is a rubber stamp: an agent that edits the
tests, skips the suite, or prints the expected output passes it. These are the
toy-scale equivalents of the full-scale filters (diff-path allowlist, no
`git checkout`, no `pytest.skip`, no exit-code forgery), adapted to a task with
a label instead of a test suite.

Design constraint that keeps the dataset honest: **every filter must be sound,
not merely strict.** A check that rejects valid reasoning silently biases the
training set toward easy items. So `support_conflict` only fires on a *provable*
contradiction (negation parity), never on weak lexical overlap — the asymmetry
matters, and it is why this module has few heuristics rather than many.
"""

from __future__ import annotations

import re

__all__ = [
    "label_leak",
    "citation_integrity",
    "support_conflict",
    "is_trivial",
    "has_successful_tool_call",
    "NEGATION_MARKERS",
]

_NEGATION_MARKERS = ("not", "no", "never", "none", "nothing", "cannot", "without", "neither")
_WORD = re.compile(r"[a-z']+")
_LABEL_TOKEN = re.compile(r"\b(entailed|not[_ ]entailed)\b", re.IGNORECASE)
_LOOKUP_CALL = re.compile(r"\blookup\s*\([^)]*\)", re.IGNORECASE)

_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "being", "been", "of", "to",
    "in", "on", "at", "for", "with", "and", "or", "that", "this", "it", "its", "as",
    "by", "from", "there", "his", "her", "their", "who", "which", "as", "s", "t",
}


#: Negation words are excluded from the token comparison: the question is whether
#: two texts assert the *same content*, with negation compared separately. Leaving
#: "not" in the denominator made a negated hypothesis look like low overlap.
_NEGATION_SET = frozenset(_NEGATION_MARKERS) | {"case", "true", "fact", "there"}


def _content_tokens(text: str) -> set[str]:
    return {
        w
        for w in _WORD.findall(text.lower())
        if w not in _STOP and w not in _NEGATION_SET and len(w) > 2
    }


def _overlap(a: set[str], b: set[str]) -> float:
    if not a:
        return 0.0
    return len(a & b) / len(a)


def has_negation(text: str) -> bool:
    return any(w in _NEGATION_MARKERS for w in _WORD.findall(text.lower()))


def label_leak(task: dict) -> bool:
    """Does the gold label appear verbatim in text the agent can see?

    A task-level defect, not an agent behaviour: if the answer is printed in the
    prompt, every trajectory "solves" it and the task contributes noise. Catching
    it here is what makes the rejection histogram trustworthy — otherwise a leaky
    task would show up as 100% pass rate and quietly inflate the eval set.
    """
    gold = task.get("label", "")
    visible = [task.get("hypothesis", "")]
    visible += [e.get("text", "") for e in task.get("evidence", [])]
    gold_norm = gold.replace("_", " ")
    for text in visible:
        for match in _LABEL_TOKEN.finditer(text):
            if match.group(1).lower().replace("_", " ") == gold_norm:
                return True
    return False


def citation_integrity(cite: list[int], retrieved: set[int]) -> bool:
    """Is every cited index one the agent actually retrieved?

    Catching hallucinated citations is the cheap half of the reward-hack story:
    an agent that cites sentence 4 without ever calling `lookup(4)` has not read
    anything, it has guessed with a plausible-looking trace.
    """
    if not cite:
        return False
    return all(idx in retrieved for idx in cite)


def support_conflict(hypothesis: str, cited_texts: list[str], answer: str) -> bool:
    """Provable contradiction between the cited evidence and an `entailed` claim.

    Deliberately narrow. It fires only when all three hold:

      1. the hypothesis and the cited text share most content words (>= 0.6),
      2. exactly one of them is negated, and
      3. the answer is `entailed`.

    Case 2 with 1 and 3 is a logical contradiction, not a judgement call, so
    rejecting it cannot remove a legitimate trajectory. Everything short of that
    — low overlap, neutral-vs-entailed confusion, missing inference — is left to
    the label check, because a heuristic that guesses here would quietly delete
    the hard examples the dataset exists to collect.
    """
    if answer != "entailed" or not cited_texts:
        return False
    hyp_tokens = _content_tokens(hypothesis)
    cited = " ".join(cited_texts)
    cit_tokens = _content_tokens(cited)
    if not hyp_tokens or not cit_tokens:
        return False

    shared = hyp_tokens & cit_tokens
    # Floor of 2 shared content words: a single shared word ("guitar") between a
    # short hypothesis and a long sentence is not evidence of a contradiction.
    if len(shared) < 2:
        return False
    # Normalise by the *shorter* side. Our negation trap puts a negation frame
    # around the cited sentence, so the hypothesis is always the longer text;
    # normalising by the hypothesis alone made the frame count against us.
    if len(shared) / min(len(hyp_tokens), len(cit_tokens)) < 0.6:
        return False

    return has_negation(hypothesis) != has_negation(cited)


def is_trivial(messages: list[dict]) -> bool:
    """A trajectory with no reasoning: the first turn is a bare tool call.

    The full-scale analogue is a diff that only touches whitespace or comments.
    We keep a token of reasoning per trajectory because supervision on
    plan-free traces teaches the model to act without thinking, which then fails
    on any task that needs two steps.
    """
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        without_calls = _LOOKUP_CALL.sub("", content)
        prose = re.sub(r"^\s*(CITE|ANSWER)\s*:.*$", "", without_calls, flags=re.MULTILINE | re.IGNORECASE)
        return len(prose.strip()) < 10
    return True


def has_successful_tool_call(messages: list[dict]) -> bool:
    """Did the agent actually retrieve anything?

    With the document hidden by construction, a trajectory with no successful
    `lookup` cannot have grounded its answer — it either guessed or leaked.
    """
    return any(msg.get("role") == "tool" and msg.get("tool_ok") for msg in messages)
