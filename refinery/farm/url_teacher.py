"""URL-repair teacher: a stochastic, multi-strategy stand-in for a real teacher.

Why this exists and what it is not
----------------------------------
The NLI pool's offline teacher was *deterministic*: 60 unique prompts across 217
trajectories, which is why the 1M student learned nothing (pass@1 = 0.00 on every
arm). A pipeline demo needs a dataset with variance, and at toy scale the variance
has to come from somewhere.

This teacher provides it: varied retrieval strategies, a ranked search over
candidates, and — crucially — an explicit **error profile**. Real frontier models
get tasks wrong, and a dataset built only from successes teaches the verifier
nothing: the rejection histogram would be all zeros and the anti-hack filters
would be untested.

**It is not a model and must never be reported as one.** Its error rates are
*parameters chosen to exercise the filter*, not measurements. Every report that
includes it says so. The frontier comparison uses the real API teacher; this thing
exists so stages 1–6 are runnable and reproducible with no key and no quota.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlsplit

from refinery.common.protocol import TOOL_NAME
from refinery.common.urls import compose_url, normalize
from refinery.farm.teacher import TeacherReply

__all__ = ["UrlRepairTeacher", "PLANS", "TeacherProfile"]


@dataclass(frozen=True)
class TeacherProfile:
    """Error rates. Parameters, not measurements — see the module docstring."""

    #: Chance of picking the runner-up candidate instead of the top-ranked one.
    slip: float = 0.18
    #: Chance of emitting a correct answer while citing an index it never retrieved.
    ungrounded_citation: float = 0.05
    #: Chance of echoing the damaged input back as the answer.
    echo: float = 0.04
    #: Chance of retrieving only the single best candidate on a compose task,
    #: which guarantees a wrong answer (it cannot know the query donor).
    compose_myopia: float = 0.12


PLANS = (
    "PLAN: compare the damaged string against the candidate list.",
    "PLAN: match scheme, host and path before deciding.",
    "PLAN: the query parameters are the tell here; check them first.",
    "PLAN: narrow by host, then confirm the path.",
    "PLAN: look for the closest surviving fragment.",
    "PLAN: check whether a scheme was dropped before anything else.",
    "PLAN: work from the most distinctive surviving part.",
    "PLAN: verify each field against the candidates in turn.",
)


def _tokens(url: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", url.lower()))


def _score(damaged: str, candidate: str) -> float:
    """How well a candidate explains the damaged string.

    Weighted toward the host and the surviving path fragments, because those are
    what survive most corruptions (a dropped scheme or an escaped slash leaves the
    host intact), while query keys survive truncation and query order does not.
    """
    dmg, cand = normalize(damaged), normalize(candidate)
    if not dmg or not cand:
        return 0.0
    d_host = urlsplit(dmg).hostname or ""
    c_host = urlsplit(cand).hostname or ""
    d_path = set(re.findall(r"[a-z0-9]+", urlsplit(dmg).path))
    c_path = set(re.findall(r"[a-z0-9]+", urlsplit(cand).path))
    d_q = {k for k, _v in parse_qsl(urlsplit(dmg).query)}
    c_q = {k for k, _v in parse_qsl(urlsplit(cand).query)}

    score = 0.0
    if d_host and c_host:
        score += 2.0 if d_host == c_host else 0.4 * _jaccard(d_host, c_host)
    if d_path and c_path:
        score += 1.5 * _jaccard(d_path, c_path)
    if d_q and c_q:
        score += 1.2 * _jaccard(d_q, c_q)
    # Length agreement: truncation leaves the candidate strictly longer.
    score += 0.5 * (1.0 - min(abs(len(dmg) - len(cand)), 60) / 60.0)
    return score


def _jaccard(a, b) -> float:
    """Over a set of tokens, or over characters when handed raw strings (hosts).

    Hosts are compared character-wise because a damaged host is often truncated
    mid-word, and tokenising it would report zero overlap for a prefix that is
    still highly informative.
    """
    if isinstance(a, str):
        a, b = set(a), set(b)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class UrlRepairTeacher:
    """Retrieves a ranked subset of candidates, then answers — with error injection.

    Implements the same `complete()` surface as the API teachers, so the farm
    harness, the eval harness and the ledger are unchanged.
    """

    name = "url-repair-synthetic"

    def __init__(
        self,
        profile: TeacherProfile | None = None,
        *,
        seed: int = 0,
        temperature: float = 0.8,
    ) -> None:
        self.profile = profile or TeacherProfile()
        self.rng = random.Random(seed)
        self.temperature = temperature
        self.calls = 0

    # -- error-profile knobs, all temperature-sensitive ---------------------
    def _p(self, base: float) -> float:
        """Scale a probability with temperature: hotter sampling explores more."""
        return max(0.0, min(0.95, base * (self.temperature / 0.8)))

    def complete(self, messages: list[dict], *, temperature: float | None = None) -> TeacherReply:
        if temperature is not None:
            self.temperature = temperature
        self.calls += 1

        task = _task_from_messages(messages)
        if task is None:
            return TeacherReply(content="", error="no_task_in_prompt", model=self.name)

        n = task["n_candidates"]
        retrieved = sorted({m["evidence_index"] for m in messages if "evidence_index" in m})
        budget = self._remaining_budget(messages)

        if not retrieved:
            return self._first_turn(task, n, budget)

        candidates: list[str] = [e["text"] for e in task["evidence"]]

        # Final answer: choose among what was actually retrieved.
        texts = [candidates[i] for i in retrieved if i < len(candidates)]
        return self._final_turn(task, retrieved, texts)

    # -- turns ---------------------------------------------------------------
    def _first_turn(self, task: dict, n: int, budget: int) -> TeacherReply:
        """Fetch the top hits.

        The teacher is under the same information restriction as a real API model:
        it sees the damaged URL and the candidate *count*, and nothing else. So its
        retrieval policy is the learnable one — take the top of the ranked list,
        escalate if the top hit does not explain the damage.
        """
        k = min(n, max(1, budget), self.rng.choice((1, 2, 2, 3, 3)))
        wanted = list(range(k))
        plan = self.rng.choice(PLANS)
        calls = " ".join(f"{TOOL_NAME}({i})" for i in wanted)
        return TeacherReply(
            content=f"{plan}\n{calls}",
            completion_tokens=len(wanted) + 12,
            model=self.name,
        )

    def _final_turn(self, task: dict, retrieved: list[int], texts: list[str]) -> TeacherReply:
        broken = task["broken"]
        answer = self._decide(broken, texts)

        if self.rng.random() < self._p(self.profile.echo):
            answer = broken

        # An ungrounded citation: right answer, an index it never retrieved. This
        # is the trajectory class the citation filter exists to catch.
        if self.rng.random() < self._p(self.profile.ungrounded_citation) and len(retrieved) < 4:
            extra = [i for i in range(4) if i not in retrieved]
            if extra:
                retrieved = sorted(set(retrieved) | {self.rng.choice(extra)})

        plan = self.rng.choice(PLANS)
        cite = ", ".join(str(i) for i in retrieved)
        return TeacherReply(
            content=f"{plan}\nCITE: {cite}\nANSWER: {answer}",
            completion_tokens=24,
            model=self.name,
        )

    def _decide(self, broken: str, texts: list[str]) -> str:
        """Pick the base candidate, then decide whether to splice in a query donor.

        The compose/pick decision is made from the damaged string alone — exactly
        the decision the student has to make. Handing the teacher the task's real
        variant would leak the answer into its own reasoning and inflate whatever
        the student learns from its trajectories.
        """
        if not texts:
            return broken

        # Base: best location match. Falls back to the overall score so a heavily
        # truncated URL still gets its best available answer.
        def location_score(cand: str) -> float:
            parts = urlsplit(normalize(broken))
            return (
                2.0 * _jaccard(parts.hostname or "", (urlsplit(normalize(cand)).hostname or ""))
                + 1.5 * _jaccard(
                    set(re.findall(r"[a-z0-9]+", parts.path)),
                    set(re.findall(r"[a-z0-9]+", urlsplit(normalize(cand)).path)),
                )
            )

        base = max(texts, key=location_score)

        # Query donor: a candidate whose query keys explain the damage better than
        # the base's own query does.
        dmg_keys = {k for k, _v in parse_qsl(urlsplit(normalize(broken)).query)}
        if not dmg_keys:
            return base

        def query_keys(cand: str) -> set[str]:
            return {k for k, _v in parse_qsl(urlsplit(normalize(cand)).query)}

        base_q = query_keys(base)
        # Compare *how well* each candidate's query explains the damage rather than
        # testing for any shared key. An intersection test said "the base already
        # mentions a query key, so no splicing", which is wrong whenever the base
        # happens to carry an unrelated query — that bug made every compose task
        # fail verification.
        base_explains = _jaccard(dmg_keys, base_q)

        donors = [c for c in texts if c != base and query_keys(c)]
        if not donors:
            return base
        donor = max(donors, key=lambda c: _jaccard(dmg_keys, query_keys(c)))
        if _jaccard(dmg_keys, query_keys(donor)) <= base_explains:
            return base
        return compose_url(_parts(base), _parts(donor))

    @staticmethod
    def _remaining_budget(messages: list[dict]) -> int:
        from refinery.compiler.render import MAX_LOOKUPS

        used = len({m["evidence_index"] for m in messages if "evidence_index" in m})
        return max(1, MAX_LOOKUPS - used)


def _parts(url: str) -> dict:
    parts = urlsplit(normalize(url))
    return {
        "scheme": parts.scheme,
        "host": parts.netloc,
        "port": "",
        "path": parts.path or "/",
        "query": f"?{parts.query}" if parts.query else "",
        "fragment": "",
    }


def _task_from_messages(messages: list[dict]) -> dict | None:
    """Recover what an agent can legitimately see from the rendered prompt.

    Deliberately lossy: the damaged URL and the candidate count, plus whatever has
    been retrieved so far. The teacher gets no access to the task record, so it is
    under exactly the information restriction the student will face at eval time.
    """
    damaged = n = None
    for msg in messages:
        if msg.get("role") != "user":
            continue
        m = re.search(r"DAMAGED URL:\s*(.+)", msg.get("content", ""))
        c = re.search(r"CANDIDATES:\s*(\d+)", msg.get("content", ""))
        if m:
            damaged = m.group(1).strip()
        if c:
            n = int(c.group(1))
    if damaged is None or n is None:
        return None

    retrieved_texts: list[str] = [""] * n
    for msg in messages:
        idx = msg.get("evidence_index")
        if msg.get("role") == "tool" and msg.get("tool_ok") and isinstance(idx, int) and idx < n:
            retrieved_texts[idx] = msg.get("content", "")

    return {
        "broken": damaged,
        "n_candidates": n,
        "evidence": [{"text": t} for t in retrieved_texts],
    }
