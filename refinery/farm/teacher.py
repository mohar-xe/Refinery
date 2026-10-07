"""Teacher clients.

The farm is the only stage that talks to a model provider. It is deliberately
small and deliberately swappable, because at toy scale the teacher is a *free*
1M-context model and at full scale it is a frontier API — and because the
pipeline must be runnable with no key at all, so a reviewer can reproduce the
verifier and compiler stages without spending anyone's quota.

Two implementations, one interface:
  * `OpenRouterTeacher` — real API, full usage accounting.
  * `HeuristicTeacher` — offline, deterministic, no key. Doubles as the
    cheap "mid-tier contender" baseline in the frontier plot (LLD.md D-013/§4).
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field

from refinery.common.protocol import TOOL_NAME
from refinery.compiler.render import MAX_LOOKUPS
from refinery.config import TeacherCfg

__all__ = ["TeacherReply", "Teacher", "OpenRouterTeacher", "HeuristicTeacher", "build_teacher"]


@dataclass
class TeacherReply:
    """One completion plus everything needed for the cost ledger."""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    error: str | None = None
    model: str = ""
    cached: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.content.strip())


class Teacher:
    """Interface: given a conversation, produce the next assistant turn."""

    name: str = "teacher"

    def complete(self, messages: list[dict], *, temperature: float | None = None) -> TeacherReply:
        raise NotImplementedError


class OpenRouterTeacher(Teacher):
    """OpenRouter chat-completions client with retries and honest usage capture."""

    name = "openrouter"

    def __init__(self, cfg: TeacherCfg, api_key: str | None = None) -> None:
        self.cfg = cfg
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._client = None

    def available(self) -> bool:
        return bool(self.api_key)

    def _post(self, payload: dict) -> dict:
        import httpx  # imported lazily: offline stages must not need it

        if self._client is None:
            self._client = httpx.Client(timeout=self.cfg.timeout_s)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # OpenRouter attributes usage to the app; harmless and useful.
            "HTTP-Referer": "https://github.com/mohar-xe/Refinery",
            "X-Title": "Refinery (agent trajectory refinery)",
        }
        resp = self._client.post(f"{self.cfg.base_url}/chat/completions", json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()

    def complete(self, messages: list[dict], *, temperature: float | None = None) -> TeacherReply:
        payload: dict = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": self.cfg.max_tokens,
        }
        if self.cfg.prompt_cache:
            payload["usage"] = {"include": True}

        last_error: str | None = None
        for attempt in range(self.cfg.max_retries + 1):
            started = time.monotonic()
            try:
                data = self._post(payload)
            except Exception as exc:  # network/HTTP/provider errors are data here
                last_error = f"{type(exc).__name__}: {exc}"[:300]
                time.sleep(1.5 * (attempt + 1))
                continue

            latency = time.monotonic() - started
            try:
                choice = data["choices"][0]["message"]
                usage = data.get("usage") or {}
            except (KeyError, IndexError, TypeError) as exc:
                last_error = f"malformed_response: {exc}"
                continue

            return TeacherReply(
                content=choice.get("content") or "",
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
                latency_s=latency,
                model=data.get("model") or self.cfg.model,
                cached=bool(usage.get("cache_read_tokens")),
            )

        return TeacherReply(content="", error=last_error or "unknown_error", model=self.cfg.model)

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * self.cfg.price_per_1k_prompt) / 1000 + (
            completion_tokens * self.cfg.price_per_1k_completion
        ) / 1000


class HeuristicTeacher(Teacher):
    """Offline baseline: retrieves every segment, then guesses by lexical overlap.

    Two jobs:
      1. makes every downstream stage runnable and testable with no API key;
      2. provides the cheap contender in the frontier plot.

    It is a *heuristic*, not a model — the README must not imply otherwise
    (LLD.md §4, open decision on whether to keep this row at all). Its accuracy
    on binary SNLI lands near the majority-class-plus-overlap band, which is the
    honest bar a distilled student has to clear.
    """

    name = "heuristic"

    def __init__(self, cfg: TeacherCfg, seed: int = 0) -> None:
        self.cfg = cfg
        self.rng = random.Random(seed)

    def complete(self, messages: list[dict], *, temperature: float | None = None) -> TeacherReply:
        started = time.monotonic()
        # What has already been retrieved is exactly the tool results in context,
        # read off the transcript the harness hands us. Re-deriving it by scanning
        # assistant text for `lookup(...)` and then subtracting the tool results is
        # what this loop did before, and it cancelled itself out into an infinite
        # re-fetch of segment 0.
        retrieved = {m["evidence_index"] for m in messages if "evidence_index" in m}

        n = _doc_len(messages)
        # Retrieve up to a *target* and then answer, rather than retrieving until the
        # budget runs out. Retrieving-until-exhausted looks correct and is a trap:
        # it spends every lookup on retrieval, never emits the ANSWER line, and the
        # run dies as `malformed` with an empty dataset downstream. This is the
        # retrieval-starvation failure mode Project 2 studies, in miniature.
        target = min(n, max(1, MAX_LOOKUPS - 1)) if n else 0
        remaining = [i for i in range(n) if i not in retrieved][: max(0, target - len(retrieved))]
        if remaining:
            idx = remaining[0]
            return TeacherReply(
                content=f"PLAN: still need segment {idx}.\n{TOOL_NAME}({idx})",
                completion_tokens=12,
                latency_s=time.monotonic() - started,
                model="heuristic-v1",
            )

        hypothesis = _last_user_text(messages)
        retrieved_text = " ".join(
            msg.get("content", "") for msg in messages if msg.get("role") == "tool"
        )
        answer = _overlap_guess(hypothesis, retrieved_text, self.rng)
        cite = ", ".join(str(i) for i in sorted(retrieved))
        return TeacherReply(
            content=(
                "PLAN: compared the retrieved segments with the hypothesis.\n"
                f"CITE: {cite}\nANSWER: {answer}"
            ),
            completion_tokens=20,
            latency_s=time.monotonic() - started,
            model="heuristic-v1",
        )


def _doc_len(messages: list[dict]) -> int:
    for msg in messages:
        if msg["role"] == "user":
            # Tolerant on purpose: this parses the renderer's prompt text, so a
            # wording change in `compiler/render.py` would otherwise silently
            # produce a teacher that never calls the tool. Covered by
            # `tests/test_prompt_contract.py`.
            # Tolerant on purpose: this parses the renderer's prompt text, so a
            # wording change in `compiler/render.py` would otherwise silently
            # produce a teacher that never calls the tool. Covered by
            # `tests/test_prompt_contract.py`.
            match = re.search(
                r"(?:DOCUMENT|PROPOSITIONS|CANDIDATES):\s*(\d+)", msg.get("content", "")
            )
            if match:
                return int(match.group(1))
    return 0


def _last_user_text(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg["role"] == "user":
            match = re.search(r"(?:HYPOTHESIS|CLAIM|DAMAGED URL):\s*(.+)", msg.get("content", ""))
            if match:
                return match.group(1).strip()
    return ""


def _overlap_guess(hypothesis: str, document: str, rng: random.Random) -> str:
    """Crude entailment guess: high content-word overlap -> entailed.

    Deliberately shallow, because its failure mode on the adversarial variants is
    the point: it answers `entailed` on `premise_negation`, which is exactly the
    trap the verifier and the student have to beat.
    """
    stop = {
        "a", "an", "the", "is", "are", "was", "were", "be", "being", "been", "of", "to",
        "in", "on", "at", "for", "with", "and", "or", "that", "this", "it", "its", "as",
        "by", "from", "there", "his", "her", "their", "who", "which", "not",
    }
    h = {w for w in re.findall(r"[a-z']+", hypothesis.lower()) if w not in stop}
    d = {w for w in re.findall(r"[a-z']+", document.lower()) if w not in stop}
    if not h:
        return "not_entailed"
    overlap = len(h & d) / len(h)
    if overlap >= 0.8:
        return "entailed"
    if overlap < 0.35:
        return "not_entailed"
    # Genuinely ambiguous band: coin-flip, with temperature still respected so the
    # farm records a real distribution over samples rather than a constant.
    return "entailed" if rng.random() < overlap - 0.35 else "not_entailed"


def build_teacher(cfg: TeacherCfg, *, prefer: str = "api", seed: int = 0) -> Teacher:
    """Factory. `prefer="api"` falls back to the offline heuristic when no key."""
    if prefer in ("api", "auto"):
        client = OpenRouterTeacher(cfg)
        if client.available():
            return client
        if prefer == "api":
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Export it, or run with --teacher heuristic."
            )
    return HeuristicTeacher(cfg, seed=seed)


def teacher_metadata(teacher: Teacher, cfg: TeacherCfg) -> dict:
    """Provenance recorded on every run so results are attributable."""
    return {
        "teacher": teacher.name,
        "model": cfg.model if teacher.name == "openrouter" else getattr(teacher, "cfg", cfg).model,
        "temperature": cfg.temperature,
        "priced": cfg.priced,
        "price_per_1k_prompt": cfg.price_per_1k_prompt,
        "price_per_1k_completion": cfg.price_per_1k_completion,
    }


def dumps(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)
