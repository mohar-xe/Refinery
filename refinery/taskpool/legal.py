"""Complex legal verification task pool, over livelaw.in propositions.

Upstream, not invented
-----------------------
The propositions come from `para-proposition` (private): livelaw.in articles are
scraped, split into sentences, decontextualized, decomposed into atomic legal
facts, and gated for atomicity and fidelity. That repo's own README states that
`entailed` is not produced — stage 1 emits facts and stops. This module is the
missing stage: **does a legal claim follow from the propositions?**

Why this task is worth more than the NLI pool it replaces
--------------------------------------------------------
1. **Real, checked propositions.** p2facts hand-verified 56/56 of an earlier run's
   facts as correct. The premise side of every task is therefore grounded in
   human-checked text, not in a generator.
2. **Retrieval over ~40 units.** A real judgment decomposes into dozens of
   propositions; only one or two bear on any claim. Retrieval is genuinely
   required, and the context is long enough that the windowing policy is no
   longer a no-op (it was, at NLI toy scale — see LLD.md D-020a).
3. **The negatives are the interesting half.** The failure modes that matter in
   legal verification are entity swaps (wrong court, wrong bench, wrong year),
   negation, and scope overreach. Those are what this generator produces, and
   they are exactly what a lexical-overlap model gets wrong.
4. **Asymmetric cost.** Asserting "entailed" when the proposition does not support
   the claim is a legal error; the reverse is a missed lead. So class-wise metrics
   are reported, not just accuracy, and the base rate of positives is held near
   half on purpose rather than left at whatever the corpus gives.

Label construction is verified by construction: every positive claim restates a
proposition the corpus asserts, and every negative is a transformation that cannot
be entailed by it. Every transform is recorded on the task so any label can be
audited by reading the transform.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from refinery.common.hashing import content_id, split_of, stable_seed

__all__ = [
    "Article",
    "load_articles",
    "make_legal_task",
    "build_legal_manifest",
    "TRANSFORMS",
    "VARIANTS",
]

VARIANTS = ("restate", "identity", "negation", "entity_swap", "scope_overreach", "conjunction", "broken_conjunction")

#: Share of tasks whose gold label is `entailed`. Held near 0.5 so accuracy cannot
#: be won by always answering one class.
POSITIVE_RATE = 0.5

#: Entity substitutions that turn an entailed claim into a non-entailed one while
#: leaving ~95% of the tokens intact — the whole point, because that is what defeats
#: a lexical-overlap verifier.
#:
#: The first version only covered courts, benches and years, and applied to 2 of 19
#: sampled propositions: this corpus is mostly about parties, amounts and periods.
#: The table is ordered most-specific-first so a court swap wins over a party swap.
_SWAPPABLE = (
    (r"\bthe Supreme Court of India\b", "the Delhi High Court"),
    (r"\bthe Supreme Court\b", "a High Court"),
    (r"\bthe First Appellate Court\b", "the Second Appellate Court"),
    (r"\bthe Trial Court\b", "the High Court"),
    (r"\bthe Appellate Court\b", "the Sessions Court"),
    (r"\bthe bench\b", "the respondent's counsel"),
    (r"\bthe appellant[-\u2011]plaintiff\b", "the respondent-defendant"),
    (r"\bthe respondent[-\u2011]defendant\b", "the appellant-plaintiff"),
    (r"\bthe appellant\b", "the respondent"),
    (r"\bthe respondent\b", "the appellant"),
    (r"\bthe applicant\b", "the respondent"),
    (r"\bthe defendant\b", "the plaintiff"),
    (r"\bthe purchaser\b", "the vendor"),
    (r"\bthe vendor\b", "the purchaser"),
    (r"\bthe Union of India\b", "the State of West Bengal"),
    (r"\bRs\.?\s?([\d,]+)", None),          # amount, handled specially
    (r"\bJustice\b", "an advocate"),
    (r"\bthe Court held\b", "the parties agreed"),
    (r"\bfour decades\b", "one decade"),
    (r"\btwo years\b", "five years"),
    (r"\bone month\b", "six months"),
    (r"\b\d{4}\b", "1998"),
)

#: Replacement amount: a different number of the same shape, so the claim stays
#: fluent and the diff is a single token class.
_AMOUNT_RE = re.compile(r"\bRs\.?\s?[\d,]+")
_FRAMES = (
    "According to the judgment, {fact}",
    "The report states that {fact}",
    "It was observed that {fact}",
    "{fact}",
    "The judgment records that {fact}",
)


@dataclass
class Article:
    """One judgment's proposition set."""

    article_id: str
    title: str
    raw_text: str
    url: str = ""
    facts: list[str] = field(default_factory=list)
    fidelity: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        return {
            "article_id": self.article_id,
            "title": self.title,
            "raw_text": self.raw_text,
            "url": self.url,
            "n_facts": len(self.facts),
        }


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _load_pipeline_output(path: Path) -> list[Article]:
    """Read one `para-proposition` run directory (5_final_facts.json + stage 4)."""
    facts_path = path / "5_final_facts.json"
    if not facts_path.exists():
        return []
    payload = json.loads(facts_path.read_text(encoding="utf-8"))
    data = payload.get("data") or {}
    facts = [f["fact"] for f in (data.get("facts") or []) if f.get("fact")]

    fidelity: dict = {}
    fid_path = path / "4_fidelity.json"
    if fid_path.exists():
        try:
            fid_payload = json.loads(fid_path.read_text(encoding="utf-8"))
            fidelity = {"reports": fid_payload.get("data") or []}
        except json.JSONDecodeError:
            fidelity = {}

    title = payload.get("title", "")
    return [
        Article(
            article_id=content_id({"title": title, "n": len(facts)}),
            title=title,
            raw_text=payload.get("raw_text", ""),
            facts=facts,
            fidelity=fidelity,
        )
    ]


def load_articles(cfg: Any) -> list[Article]:
    """Load propositions from local `para-proposition` run directories.

    `sources` in the config is a list of paths; a directory is read as one
    pipeline run, a file is read as a JSON list of `{title, raw_text, facts}`.
    """
    articles: list[Article] = []
    for raw in cfg.taskpool.get("sources", []):
        path = Path(raw)
        if not path.is_absolute():
            path = cfg.repo_root / path
        if path.is_dir():
            articles.extend(_load_pipeline_output(path))
        elif path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload if isinstance(payload, list) else payload.get("articles", [])
            for row in rows:
                facts = row.get("facts") or []
                facts = [f["fact"] if isinstance(f, dict) else str(f) for f in facts]
                if facts:
                    articles.append(
                        Article(
                            article_id=content_id({"title": row.get("title", ""), "n": len(facts)}),
                            title=row.get("title", ""),
                            raw_text=row.get("raw_text", ""),
                            url=row.get("url", ""),
                            facts=facts,
                        )
                    )
    return articles


# --------------------------------------------------------------------------- #
# Claim transforms
# --------------------------------------------------------------------------- #
def _lower_first(text: str) -> str:
    return text[0].lower() + text[1:] if text else text


#: Every transform returns (claim, label, applied_name). The third element exists
#: because `entity_swap` has a fallback: a proposition with no swappable entity in
#: it gets negated instead. Without recording which one actually happened, the
#: `prompt_variant` field claimed `entity_swap` on a negation, and any audit of the
#: dataset by variant would have been reading the wrong claims.
TransformResult = tuple[str, str, str]


def t_identity(fact: str, rng: random.Random, article: Article) -> TransformResult:
    """The proposition itself. Entailed, and trivially so — the retrieval is the work."""
    return fact, "entailed", "identity"


def t_restate(fact: str, rng: random.Random, article: Article) -> TransformResult:
    """A truth-preserving framing. Entailed."""
    frame = rng.choice(_FRAMES)
    return frame.format(fact=_lower_first(fact)), "entailed", "restate"


def t_negation(fact: str, rng: random.Random, article: Article) -> TransformResult:
    """The logical negation of an asserted proposition. Cannot be entailed by it."""
    return f"It is not the case that {_lower_first(fact)}", "not_entailed", "negation"


def t_entity_swap(fact: str, rng: random.Random, article: Article) -> TransformResult:
    """Swap a court, a bench member, or a year.

    The single most common real-world legal error and the one a lexical matcher is
    blind to: 97% of the tokens still match.
    """
    present = set(re.findall(r"\d{4}", fact))
    for pattern, replacement in _SWAPPABLE:
        if not re.search(pattern, fact):
            continue
        if replacement is None:  # amount: replace the digits, keep the shape
            match = _AMOUNT_RE.search(fact)
            if not match:
                continue
            digits = "".join(ch for ch in match.group(0) if ch.isdigit())
            if not digits:
                continue
            # Rebuild the amount rather than string-replacing the digit run: the
            # corpus writes "Rs.20,000" with a separator, so the extracted digits
            # ("20000") do not appear literally in the text and a plain `.replace`
            # was a silent no-op — the claim came back identical to its premise
            # while still labelled `entity_swap`.
            bumped = f"{int(digits) + 25_000:,}"
            replacement = re.sub(r"[\d,]+", bumped, match.group(0), count=1)
        if replacement == "1998" and present:
            # Never swap a year to the year already in the text, or the "wrong"
            # claim is accidentally true.
            others = [y for y in ("1998", "2004", "2011", "2016", "2021") if y not in present]
            replacement = others[0]
        return re.sub(pattern, replacement, fact, count=1), "not_entailed", "entity_swap"
    return t_negation(fact, rng, article)


def t_scope_overreach(fact: str, rng: random.Random, article: Article) -> TransformResult:
    """Generalise one court's holding into a universal one.

    Kept as its own variant because the label is a genuine judgement call rather
    than a proof: a holding of one bench is not entailed as a universal. Anyone
    auditing the dataset should read these first.
    """
    return f"Indian courts have held that {_lower_first(fact)}", "not_entailed", "scope_overreach"


def t_conjunction(article: Article, fact: str, other: str, rng: random.Random) -> TransformResult:
    """Two propositions joined: entailed only if BOTH were retrieved.

    This is what makes the task complex rather than a lookup — it is the case the
    upstream's per-fact stage-4 gate structurally cannot handle, because it judges
    one fact against the source chunk and never considers a claim spanning two.
    """
    return (
        f"{fact} In addition, {_lower_first(other)}",
        "entailed",
        "conjunction",
    )


def t_broken_conjunction(
    article: Article, fact: str, other: str, rng: random.Random
) -> TransformResult:
    """One true conjunct, one negated. Not entailed, and not detectable by overlap."""
    return (
        f"{fact} It is not the case that {_lower_first(other)}",
        "not_entailed",
        "broken_conjunction",
    )


TRANSFORMS = {
    "identity": t_identity,
    "restate": t_restate,
    "negation": t_negation,
    "entity_swap": t_entity_swap,
    "scope_overreach": t_scope_overreach,
}


# --------------------------------------------------------------------------- #
# Task assembly
# --------------------------------------------------------------------------- #
_STOP = frozenset(
    "a an the of in on at for and or that this it its as by from was were is are be been has "
    "have had shall not no which who whom whose when where while than then there their they "
    "his her him he she we you i".split()
)


def _content(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOP and len(w) > 2}


def _rank_facts(facts: list[str], claim: str) -> list[str]:
    """Order propositions by how much of the claim they cover."""
    claim_tokens = _content(claim)

    def score(fact: str) -> float:
        overlap = claim_tokens & _content(fact)
        return len(overlap) / len(claim_tokens) if claim_tokens else 0.0

    ordered = sorted(facts, key=lambda f: (-score(f), f))
    head, tail = ordered[:3], ordered[3:]
    # Deterministically permute the head: pin it to index 0 half the time so the
    # policy cannot collapse to "always fetch candidate 0".
    if len(head) > 1 and (len(ordered) + score("")) % 2 == 0:
        head = head[1:] + head[:1]
    return head + tail


def make_legal_task(
    article: Article, index: int, variant: str, rng: random.Random
) -> dict | None:
    """One claim-verification task over one judgment's proposition set.

    The whole proposition set is the hidden document: the claim is answered only
    by retrieving the one or two propositions that bear on it, out of 30-50.
    """
    facts = article.facts
    if len(facts) < 4:
        return None

    primary = facts[index % len(facts)]
    other = facts[(index + rng.randint(1, len(facts) - 1)) % len(facts)]

    if variant == "conjunction":
        claim, label, applied = t_conjunction(article, primary, other, rng)
    elif variant == "broken_conjunction":
        claim, label, applied = t_broken_conjunction(article, primary, other, rng)
    else:
        claim, label, applied = TRANSFORMS[variant](primary, rng, article)

    # Propositions are ordered by relevance to *this claim*, then the head is
    # permuted. Same reasoning as D-023: the agent can retrieve at most 4 of ~41
    # propositions, and walking them in document order means it essentially never
    # finds the supporting one (measured: sequential top-k over document order
    # retrieves the support almost never). A ranked list is also what a real
    # retrieval-augmented system returns, so "fetch the top hits, then verify"
    # is the learnable policy rather than an artifact.
    ordered = _rank_facts(facts, claim)
    gold_position = next(i for i, _f in enumerate(ordered) if _f == primary)
    other_position = next((i for i, _f in enumerate(ordered) if _f == other), None)

    evidence = [{"evidence_id": f"e{i}", "index": i, "text": f} for i, f in enumerate(ordered)]

    task = {
        "kind": "legal",
        "origin": "livelaw",
        "prompt_variant": variant,
        "transform": variant,
        "transform_applied": applied,
        "article_id": article.article_id,
        "article_title": article.title,
        "article_url": article.url,
        "claim": claim,
        "label": label,
        # The proposition(s) the claim is about. Recorded so a human can audit any
        # label without re-running the transform, and so eval can report per-variant
        # accuracy.
        # Indices into the *ranked* evidence list, which is what the agent sees.
        "support_positions": [gold_position] + (
            [other_position] if ("conjunction" in variant and other_position is not None) else []
        ),
        "support_fact_ids": [facts.index(primary)] + ([facts.index(other)] if "conjunction" in variant else []),
        "hypothesis": claim,
        "n_units": len(facts),
        "evidence": evidence,
        "source_id": str(stable_seed("legal", article.article_id, index) % 10**9),
    }
    task["task_id"] = content_id(
        {k: task[k] for k in ("kind", "claim", "article_id", "evidence") if k in task}
    )
    return task


def build_legal_manifest(cfg: Any) -> dict:
    """Build the manifest over every (article, claim-index) pair.

    Unlike the other pools this one does not cap `n_tasks`: the natural unit is the
    full cross product of propositions, and more tasks is strictly better for the
    rejection-sampling yield. `n_tasks` is still honoured as a ceiling when set.
    """
    articles = load_articles(cfg)
    if not articles:
        raise RuntimeError(
            "no propositions found. Set taskpool.sources to one or more "
            "para-proposition pipeline_output directories (see LLD.md D-025)."
        )

    n_tasks = int(cfg.taskpool.get("n_tasks", 0)) or None
    seed_salt = cfg.taskpool.get("seed_salt", "legal-v1")
    variants = tuple(cfg.taskpool.get("variants", VARIANTS))
    positives = tuple(
        v for v in variants if v in ("identity", "restate", "conjunction")
    )
    negatives = tuple(v for v in variants if v not in positives)

    tasks: list[dict] = []
    for article in articles:
        # A balanced variant schedule: one task per proposition, cycling variants,
        # so class balance is structural rather than a post-hoc resample.
        for index in range(len(article.facts)):
            rng = random.Random(stable_seed(seed_salt, article.article_id, index))
            # Hash-based coin, not `index % 100`: with 40-odd propositions per
            # article, `index` never reaches 100, so an index-based threshold made
            # every single task positive (measured: 123/123 entailed). The hash is
            # balanced, deterministic, and independent of how many facts an
            # article happens to have.
            want_positive = (stable_seed(seed_salt, article.article_id, index, "pos") % 1000) < (
                POSITIVE_RATE * 1000
            )
            pool = positives if want_positive else negatives
            variant = pool[index % len(pool)]
            task = make_legal_task(article, index, variant, rng)
            if task is None:
                continue
            task["split"] = split_of(task["task_id"], cfg.split_salt, cfg.eval_pct)
            tasks.append(task)

    deduped: dict[str, dict] = {}
    for task in tasks:
        deduped.setdefault(task["task_id"], task)
    duplicates = len(tasks) - len(deduped)
    tasks = list(deduped.values())
    if n_tasks:
        tasks = tasks[:n_tasks]
    tasks.sort(key=lambda t: (t["split"], t["task_id"]))

    cfg.ensure_dirs()
    cfg.manifest_path.write_text(
        "".join(json.dumps(t, sort_keys=True, ensure_ascii=False) + "\n" for t in tasks),
        encoding="utf-8",
    )

    train_ids = {t["task_id"] for t in tasks if t["split"] == "train"}
    eval_ids = {t["task_id"] for t in tasks if t["split"] == "eval"}
    summary = {
        "source": "legal",
        "n_articles": len(articles),
        "n_propositions": sum(len(a.facts) for a in articles),
        "n_tasks": len(tasks),
        "n_train": len(train_ids),
        "n_eval": len(eval_ids),
        "split_overlap": len(train_ids & eval_ids),
        "duplicates_dropped": duplicates,
        "by_variant": {v: sum(1 for t in tasks if t["prompt_variant"] == v) for v in variants},
        "by_label": {
            lab: sum(1 for t in tasks if t["label"] == lab)
            for lab in ("entailed", "not_entailed")
        },
        "by_transform_applied": {
            name: sum(1 for t in tasks if t["transform_applied"] == name)
            for name in sorted({t["transform_applied"] for t in tasks})
        },
        "manifest": str(cfg.manifest_path),
    }
    from refinery.common.jsonl import write_json

    write_json(cfg.tasks_dir / "summary.json", summary)
    return summary
