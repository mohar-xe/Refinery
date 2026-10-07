"""URL-reconstruction checks.

Same reason codes as the entailment pool, so the rejection histogram stays
comparable across task families (LLD.md D-005), but the substance is different and
stronger: the gold is the string that was damaged, so correctness is exact match
after a shared normalization rather than a judgement call.

The checks that carry the weight:

  * **Location support.** An answer's `scheme://host[:port]/path` must match some
    *cited* candidate. This is a necessity, not a heuristic: the agent cannot
    produce a location it never read. It also catches the failure that a
    citation-integrity check alone misses — an agent that retrieves candidates and
    cites them, then writes a URL assembled from parts nobody showed it.
  * **Echo rejection.** An answer that normalizes to the damaged input is rejected
    as TRIVIAL_OUTPUT even when the label check would pass. Construction already
    refuses such tasks, so this is the tripwire that says the constructor's guard
    broke.
  * **Whole-answer support** for `pick` tasks, where the gold is a single
    candidate, so citing the right candidate must fully justify the answer.
"""

from __future__ import annotations

from urllib.parse import urlsplit

from refinery.common.urls import normalize

__all__ = ["answer_is_supported", "is_echo", "location_of", "is_correct"]


def location_of(url: str) -> str:
    """`scheme://host[:port]/path` — the part of a URL that identifies a resource.

    Uses the *normalized* form so that `HTTP://Example.COM/a/` and
    `http://example.com/a` yield the same location, and drops the query and
    fragment because for a `compose` task those come from a different candidate.
    """
    norm = normalize(url)
    if not norm:
        return ""
    parts = urlsplit(norm)
    path = parts.path or "/"
    return f"{parts.scheme}://{parts.netloc}{path}"


def query_of(url: str) -> str:
    return urlsplit(normalize(url)).query


def is_correct(predicted: str | None, gold: str) -> bool:
    """Exact match after normalization. No fuzzy credit, no partial URL."""
    if not predicted:
        return False
    return normalize(predicted) == normalize(gold)


def is_echo(predicted: str | None, damaged: str) -> bool:
    """Did the agent simply return its input?"""
    if not predicted:
        return False
    return normalize(predicted) == normalize(damaged)


def answer_is_supported(
    predicted: str | None,
    cited_texts: list[str],
    variant: str,
) -> tuple[bool, str]:
    """Is the answer justified by the candidates the agent cited?

    Returns `(supported, why)`. Never raises — `predicted` is arbitrary agent text.
    """
    if not predicted:
        return False, "no_answer"
    if not cited_texts:
        return False, "nothing_cited"

    answer_loc = location_of(predicted)
    if not answer_loc:
        return False, "unparseable_answer"

    # (1) The resource identity must have been read. Necessity, not plausibility.
    locations = [location_of(c) for c in cited_texts]
    if answer_loc not in locations:
        return False, "location_not_cited"

    # (2) For a single-candidate task the whole answer must match a cited candidate:
    #     no room for an invented query string or fragment.
    if variant in ("pick", "pick_hard"):
        answer_norm = normalize(predicted)
        if answer_norm not in {normalize(c) for c in cited_texts}:
            return False, "answer_not_a_cited_candidate"

    # (3) For a composed answer the query must also have come from somewhere read.
    if variant == "compose":
        answer_query = query_of(predicted)
        if answer_query and answer_query not in {query_of(c) for c in cited_texts}:
            return False, "query_not_cited"

    return True, "supported"
