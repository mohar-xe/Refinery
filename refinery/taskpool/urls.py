"""URL-reconstruction task pool.

A different task from the NLI pool that preceded it, chosen for one specific
reason: **the gold label is the string that was damaged**. Verification is exact
match after a normalization both the corruptor and the verifier import from the
same module, so there is no label noise, no judge model, and no interpretation
gap. That is the strongest verifier this pipeline can have, and it is worth more
for the thesis than a task with a subtler notion of correctness.

Why this is still a *hard* task, and not a lookup exercise (LLD D-002 argued
self-generated tasks without independent verification are worthless — this design
answers that objection rather than dodging it):

  * The document is hidden. Candidates must be retrieved with the tool, so
    retrieval is semantically required, not decorative.
  * Candidates are near-misses (same host, sibling paths; correct path, wrong
    query), so the agent has to compare against the damaged string rather than
    pattern-match.
  * `compose` tasks have a gold that appears in **no** candidate list and must be
    spliced from two retrieved facts. Citation checking is therefore not trivially
    satisfiable.
  * The corruption catalog is real damage (HTML unescaping, sentence punctuation,
    truncated log lines), applied in stacks, with difficulty recorded per task.

The URL corpus is procedurally generated from a documented component catalog
rather than harvested, and that is a real limitation: distribution shift from real
traffic is unmeasured. `real_urls_path` in the config exists so a harvested corpus
can be dropped in without touching code.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from refinery.common.hashing import split_of, stable_seed
from refinery.common.urls import (
    CORRUPTIONS,
    compose_url,
    corrupt,
    make_url,
    normalize,
)

__all__ = ["build_url_manifest", "make_url_task", "VARIANTS"]

VARIANTS = ("pick", "compose", "pick_hard")

#: Difficulty is a property of the corruption stack, and it is recorded on the task
#: so the report can break pass@1 down by it instead of averaging over a mix.
_DIFFICULTY = {name: spec[1] for name, spec in CORRUPTIONS.items()}


#: Corruptions that keep the query string legible. A `compose` task splices a query
#: onto a base path, so a corruption that deletes the query (`truncate`,
#: `truncate_tail`) makes the task unsolvable from the damaged string — the gold's
#: distinguishing information is simply gone. Measured: with these allowed, compose
#: scored 0% verified across 60 runs; restricted, the same runs verify.
QUERY_PRESERVING = frozenset({
    "drop_scheme", "drop_www", "sentence_punctuation", "markdown_wrapper", "whitespace",
    "newlines", "http_typo", "uppercase_host", "double_slash", "reorder_query",
    "drop_fragment", "fragment_moved", "double_encode", "html_entities", "lose_fragment",
})


def _sample_kinds(
    rng: random.Random, *, hard: bool, k_max: int = 3, query_must_survive: bool = False
) -> list[str]:
    if query_must_survive:
        pool = sorted(QUERY_PRESERVING)
    else:
        pool = [name for name, spec in CORRUPTIONS.items() if (spec[1] == 3 if hard else spec[1] >= 1)]
    k = rng.randint(1, k_max)
    return rng.sample(pool, min(k, len(pool)))


def _near_misses(gold: dict, rng: random.Random, n: int) -> list[str]:
    """Distractors that share the gold's surface — the reason picking is not trivial."""
    out: list[str] = []
    guard = 0
    while len(out) < n and guard < 40:
        guard += 1
        other = make_url(rng)
        style = rng.random()
        if style < 0.4:
            # Same host, different path — tests path discrimination.
            other["host"] = gold["host"]
            other["port"] = gold["port"]
            other["scheme"] = gold["scheme"]
        elif style < 0.7:
            # Same path, different host — tests host discrimination.
            other["path"] = gold["path"]
            other["fragment"] = gold["fragment"]
        url = other["url"]
        if url not in out and url != gold["url"]:
            out.append(url)
    return out


def make_url_task(rng: random.Random, variant: str, *, max_attempts: int = 12) -> dict | None:
    """Assemble one URL task, or None if no non-trivial corruption stack was found.

    Resamples the corruption stack rather than accepting the first one, because
    roughly a third of random stacks are *cosmetic* — case, whitespace, sentence
    punctuation, query order, doubled slashes — and `normalize()` folds all of
    them. A cosmetic-damage task is a leak, not a task: the agent can echo its own
    input and score. That guard is the whole reason this function can return None.
    """
    gold = make_url(rng)
    kinds: list[str] = []
    broken = gold["url"]
    landed: list[str] = []
    for _attempt in range(max_attempts):
        kinds = _sample_kinds(
            rng,
            hard=(variant == "pick_hard"),
            query_must_survive=(variant == "compose"),
        )
        broken = corrupt(gold["url"], kinds, rng)
        if normalize(broken) != normalize(gold["url"]):
            landed = [k for k in kinds if _lands(gold["url"], gold["url"], k)]
            break
    else:
        return None

    if variant == "compose":
        # The donor must actually carry a query, or `compose_url` falls back to a
        # fixed "?ref=combined" and the gold ends up nowhere in the candidate list
        # being *derivable* — the repair below would then paste the gold in as
        # candidate 0 and silently turn a compose task into a pick task.
        donor = make_url(rng)
        for _attempt in range(10):
            if donor.get("query"):
                break
            donor = make_url(rng)
        # Strip the base's own query and fragment. If the base carried a query,
        # the gold (which takes the donor's query) and the base would disagree
        # about two things at once, and "did the agent splice or not?" stops
        # having a single defensible answer.
        gold["query"] = ""
        gold["fragment"] = ""
        gold_url = compose_url(gold, donor)
        candidates = [gold["url"], donor["url"]]
        candidates += _near_misses(gold, rng, 2)
        if gold_url in candidates:
            return None
    else:
        gold_url = gold["url"]
        candidates = [gold_url] + _near_misses(gold, rng, 3)

    # Candidates are ordered by *relevance to the damaged string*, then the top
    # few are permuted. This is the single most consequential design choice in the
    # task, and it is not cosmetic:
    #
    # With a shuffled list, choosing which index to retrieve is unlearnable — the
    # prompt names no candidate, so the best a small model can do is guess, and the
    # gold's position carries all the signal. With a relevance-ordered list (which
    # is what a real retriever returns), the learnable policy is "fetch the top
    # hits, verify each field, escalate if none fits" — the actual skill.
    #
    # The top-3 permutation keeps the gold from being trivially pinned at index 0,
    # and the near-misses ranked above it are the cases where the agent must
    # *reject* the top hit rather than trust it.
    candidates = list(dict.fromkeys(candidates))
    candidates = _order_candidates(candidates, broken, rng)
    # Guarantee the gold is present for `pick`. Deliberately NOT done for
    # `compose`: there the whole point is that the gold appears in no candidate
    # and must be spliced from two retrieved facts, so pasting it in would quietly
    # convert a compose task into a pick task.
    if variant != "compose" and gold_url not in candidates:
        candidates[0] = gold_url

    difficulty = max((_DIFFICULTY[k] for k in landed), default=1)

    task = {
        "kind": "url",
        "origin": "url_synth",
        "prompt_variant": variant,
        "hypothesis": broken,  # reused field: the renderer reads it as the input
        "broken": broken,
        "gold_url": gold_url,
        "label": gold_url,
        "corruptions": landed,
        "difficulty": difficulty,
        "n_units": len(candidates),
        "evidence": [
            {"evidence_id": f"e{i}", "index": i, "text": c} for i, c in enumerate(candidates)
        ],
        "source_id": str(stable_seed("url", broken, gold_url) % 10**9),
    }

    task["task_id"] = _task_id(task)
    return task


def _lands(broken: str, gold: str, kind: str) -> bool:
    """Did this corruption actually change anything? (`corrupt` skips no-ops.)"""
    fn, _d = CORRUPTIONS[kind]
    return fn(gold, random.Random(0)) != gold


def _order_candidates(candidates: list[str], broken: str, rng: random.Random) -> list[str]:
    """Relevance order, then a partial shuffle of the head."""
    from refinery.farm.url_teacher import _score

    scored = sorted(candidates, key=lambda c: (-_score(broken, c), c))
    head, tail = scored[:3], scored[3:]
    rng.shuffle(head)
    return head + tail


def _task_id(task: dict) -> str:
    from refinery.common.hashing import content_id

    return content_id({k: task[k] for k in ("kind", "broken", "gold_url", "evidence") if k in task})


def _load_real_hosts(cfg: Any) -> list[str]:
    """Optional anchor corpus. Absent by default; documented as a limitation."""
    path = cfg.taskpool.get("real_urls_path")
    if not path:
        return []
    p = Path(path)
    if not p.is_absolute():
        p = cfg.repo_root / p
    if not p.exists():
        return []
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def build_url_manifest(cfg: Any) -> dict:
    """Build `tasks/manifest.jsonl` for the URL task source. Deterministic and idempotent."""
    n_tasks = int(cfg.taskpool.get("n_tasks", 2000))
    salt = cfg.split_salt
    seed_salt = cfg.taskpool.get("seed_salt", "url-v1")
    synth_fraction = float(cfg.taskpool.get("synth_fraction", 0.34))

    n_synth = int(round(n_tasks * synth_fraction))
    counts = {"pick": n_tasks - n_synth, "compose": n_synth // 2}
    counts["pick_hard"] = n_tasks - sum(counts.values())

    tasks: list[dict] = []
    for variant, want in counts.items():
        for i in range(want):
            rng = random.Random(stable_seed(seed_salt, variant, i))
            task = make_url_task(rng, variant)
            if task is None:
                continue  # construction refused it; the next seed gets its own URL
            task["split"] = split_of(task["task_id"], salt, cfg.eval_pct)
            tasks.append(task)

    deduped: dict[str, dict] = {}
    for task in tasks:
        deduped.setdefault(task["task_id"], task)
    duplicates = len(tasks) - len(deduped)
    tasks = list(deduped.values())
    tasks.sort(key=lambda t: (t["split"], t["task_id"]))

    cfg.ensure_dirs()
    import json

    cfg.manifest_path.write_text(
        "".join(json.dumps(t, sort_keys=True, ensure_ascii=False) + "\n" for t in tasks),
        encoding="utf-8",
    )

    train_ids = {t["task_id"] for t in tasks if t["split"] == "train"}
    eval_ids = {t["task_id"] for t in tasks if t["split"] == "eval"}
    summary = {
        "source": "url",
        "n_tasks": len(tasks),
        "n_train": len(train_ids),
        "n_eval": len(eval_ids),
        "split_overlap": len(train_ids & eval_ids),
        "duplicates_dropped": duplicates,
        "by_variant": {v: sum(1 for t in tasks if t["prompt_variant"] == v) for v in VARIANTS},
        "by_difficulty": {
            d: sum(1 for t in tasks if t["difficulty"] == d) for d in (1, 2, 3)
        },
        "by_corruptions": _top(tasks),
        "real_urls_loaded": len(_load_real_hosts(cfg)),
        "manifest": str(cfg.manifest_path),
    }
    from refinery.common.jsonl import write_json

    write_json(cfg.tasks_dir / "summary.json", summary)
    return summary


def _top(tasks: list[dict], n: int = 12) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in tasks:
        for kind in task.get("corruptions", []):
            counts[kind] = counts.get(kind, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1])[:n])
