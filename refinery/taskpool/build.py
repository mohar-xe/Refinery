"""Task pool: real tasks, content-hashed, with a hard wall between train and eval.

Two sources, mirroring the flagship design (HLD §3.1):

  * **SNLI** — a published, human-written inference corpus. Real data, because
    AgentTuning's failure was training on synthetic tasks with no independent
    verification; generating our own premises would repeat that mistake.
  * **Procedural adversarial variants** — the SWE-smith analogue. We mutate
    *gold* tasks to inject a specific retrieval/reasoning shortcut, so the traps
    are deliberate and documented rather than accidental.

Binary labelling: SNLI's `entailed` stays `entailed`; `neutral` and
`not_entailed` both collapse to `not_entailed`. This is the standard SNLI-binary
setup and it is stated in the dataset card, because collapsing neutral into
negative is a real (and frequently hidden) label-noise source.

Every task's identity is the hash of its content, and the train/eval split is a
salted hash of that identity — so cross-split duplication is impossible by
construction, not by checking (LLD.md, `common/hashing.py`).
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from refinery.common.hashing import split_of, stable_seed, task_identity
from refinery.compiler.render import seed_messages

__all__ = ["build_manifest", "load_snli", "split_units", "make_task", "VARIANTS"]

#: Primary source: HF's datasets-server rows API returns JSON, so stage 1 needs no
#: parquet reader and no dataframe library at all (LLD.md D-014). The response is
#: cached on disk, so this costs one request per split per machine, ever.
ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=stanfordnlp/snli&config=plain_text&split={split}&offset={offset}&length={length}"
)
#: Fallback: the parquet shard, read with pyarrow if it happens to be installed.
SNLI_URL = (
    "https://huggingface.co/datasets/stanfordnlp/snli/resolve/main/plain_text/"
    "{split}-00000-of-00001.parquet"
)
#: datasets-server label ids, per the SNLI ClassLabel: 0 entailment, 1 neutral,
#: 2 contradiction.
_LABEL_BY_ID = {0: "entailed", 1: "not_entailed", 2: "not_entailed"}
_LABEL_MAP = {"entailed": "entailed", "not_entailed": "not_entailed", "neutral": "not_entailed"}

VARIANTS = ("base", "negation_flip", "premise_negation", "distractor")

#: Document units are clause-ish spans, not sentences. SNLI premises are almost
#: always a *single* sentence (2118 of 3700 test rows have no internal clause
#: boundary), so a sentence-level document would make retrieval a no-op for most
#: of the corpus and collapse the tool-call curriculum. Splitting on clause
#: boundaries keeps the document hidden and multi-part without changing the
#: entailment semantics at all: the units concatenate back to the premise.
MAX_UNITS = 4
#: Adversarial documents are held to this many units so the whole document is
#: retrievable within MAX_LOOKUPS — the trap must be about comprehension, not
#: about the agent failing to find the evidence.
TRAP_MAX_UNITS = 3

_UNIT_SPLIT = re.compile(
    r"(?<=[,;:])\s+|\s+(?:and|but|while|although|though|because|so|which|who)\s+",
    re.IGNORECASE,
)
_WORD = re.compile(r"[a-z']+")
_NEG_PREFIX = "It is not the case that "


def split_units(text: str, *, max_units: int = MAX_UNITS) -> list[str]:
    """Split a premise into retrievable units, deterministically.

    Deliberately simple and regex-based: a fancier segmenter is a dependency, and
    any drift between the segmenter used to build tasks and the one used anywhere
    else would be a silent train/inference skew.
    """
    parts = [p.strip() for p in _UNIT_SPLIT.split(text.strip()) if p.strip()]
    if len(parts) < 2:
        return parts or [text.strip()]
    if len(parts) > max_units:
        # Keep the head and the tail: the tail of an SNLI premise carries the
        # predicate that the hypothesis usually talks about.
        parts = parts[: max_units - 1] + [parts[-1]]
    return parts


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


#: The rows API caps `length` at 100 per request (422 above that), so rows are
#: pulled in pages. This is a provider quirk, discovered the hard way.
ROWS_PAGE = 100


def _fetch_rows_api(split: str, length: int) -> list[dict] | None:
    """Pull rows as JSON from the datasets-server, paged. None if unavailable."""
    rows: list[dict] = []
    offset = 0
    while offset < length:
        page = min(ROWS_PAGE, length - offset)
        url = ROWS_URL.format(split=split, offset=offset, length=page)
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return rows or None

        items = payload.get("rows", [])
        if not items:
            break
        for item in items:
            row = item.get("row") or {}
            label = row.get("label")
            rows.append(
                {
                    "sentence1": row.get("premise", ""),
                    "sentence2": row.get("hypothesis", ""),
                    "label": _LABEL_BY_ID.get(label) if isinstance(label, int) else label,
                }
            )
        offset += page
        if len(items) < page:
            break
    return rows or None


def _fetch_parquet(split: str, cache_dir: Path) -> list[dict] | None:
    """Fallback path: download the parquet shard and read it with pyarrow."""
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"snli_{split}.parquet"
    if not cached.exists():
        url = SNLI_URL.format(split=split)
        try:
            with urllib.request.urlopen(url, timeout=120) as resp, cached.open("wb") as fh:
                fh.write(resp.read())
        except (urllib.error.URLError, TimeoutError, OSError):
            return None
    table = pq.read_table(cached, columns=["sentence1", "sentence2", "label"])
    return table.to_pylist()


def load_snli(cache_dir: Path, split: str = "test", *, length: int = 6000) -> list[dict]:
    """Fetch (and cache) SNLI, return raw rows. Prefers the JSON rows API.

    Caching means the manifest can be rebuilt offline and byte-identically: the
    cache file is the provenance record for the task pool.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_json = cache_dir / f"snli_{split}.json"
    if cached_json.exists():
        return json.loads(cached_json.read_text(encoding="utf-8"))

    rows = _fetch_rows_api(split, length) or _fetch_parquet(split, cache_dir)
    if not rows:
        raise RuntimeError(
            "could not obtain SNLI rows: the datasets-server API and the parquet "
            "shard are both unreachable. Offline copies can be dropped into "
            f"{cache_dir}/snli_{split}.json as a list of "
            "{sentence1, sentence2, label} objects."
        )
    cached_json.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def make_task(
    premise: str,
    hypothesis: str,
    label: str,
    *,
    origin: str,
    variant: str,
    source_id: str = "",
) -> dict:
    """Assemble a task record. Refuses degenerate tasks rather than shipping them."""
    units = split_units(premise)
    if len(units) < 2:
        raise ValueError("premise must split into >= 2 units so that retrieval is required")
    task = {
        "origin": origin,
        "prompt_variant": variant,
        "source_id": source_id,
        "hypothesis": hypothesis.strip(),
        "label": label,
        "n_units": len(units),
        "evidence": [
            {"evidence_id": f"e{i}", "index": i, "text": u} for i, u in enumerate(units)
        ],
    }
    task["task_id"] = task_identity(task)
    return task


def _binary_rows(rows: list[dict], limit: int, seed_salt: str) -> list[dict]:
    """Filter to usable binary rows, deterministically shuffled, de-duplicated."""
    # Dedupe on (premise, hypothesis), not on premise: SNLI gives several
    # hypotheses per premise with *different* labels, and throwing those away
    # would discard most of the label diversity for no benefit.
    seen_pair: set[tuple[str, str]] = set()
    out: list[dict] = []
    for row in rows:
        label = _LABEL_MAP.get(row.get("label") or "")
        premise = (row.get("sentence1") or "").strip()
        hypothesis = (row.get("sentence2") or "").strip()
        if label is None or not premise or not hypothesis:
            continue
        if len(split_units(premise)) < 2:
            continue
        if (premise, hypothesis) in seen_pair:
            continue
        seen_pair.add((premise, hypothesis))
        out.append({"premise": premise, "hypothesis": hypothesis, "label": label})
        if len(out) >= limit * 4:  # over-fetch: variants consume additional rows
            break
    out.sort(key=lambda r: stable_seed(seed_salt, r["premise"], r["hypothesis"]))
    return out[: limit * 4]


def _negate(text: str) -> str:
    return _NEG_PREFIX + text[0].lower() + text[1:] if text else text


def _adversarial_variants(
    base: list[dict], donors: list[dict], n: int, seed_salt: str
) -> Iterator[dict]:
    """Yield deliberately trapped tasks.

    Each variant is a *logical* consequence of the base task, so the gold label
    stays sound without a model in the loop:

      `negation_flip`    negation of an entailed hypothesis -> not_entailed, with
                         near-total lexical overlap (surface matchers break).
      `premise_negation` negation of a sentence the document asserts -> not_entailed,
                         ~100% overlap. The strongest retrieval trap.
      `distractor`       a lexically similar sentence that does *not* support the
                         hypothesis appended to the document; the gold label is
                         unchanged, so retrieving the distractor and believing it
                         produces a confidently wrong answer.
    """
    made = 0
    for row in base:
        if made >= n:
            return
        kind = ("negation_flip", "premise_negation", "distractor")[made % 3]

        if kind == "negation_flip":
            if row["label"] != "entailed":
                continue
            yield make_task(
                row["premise"],
                _negate(row["hypothesis"]),
                "not_entailed",
                origin="synth",
                variant=kind,
                source_id=stable_seed(seed_salt, row["premise"]),
            )
        elif kind == "premise_negation":
            if len(split_units(row["premise"])) > TRAP_MAX_UNITS:
                continue
            victim = split_units(row["premise"])[-1]
            yield make_task(
                row["premise"],
                _negate(victim),
                "not_entailed",
                origin="synth",
                variant=kind,
                source_id=stable_seed(seed_salt, victim),
            )
        else:
            hyp_tokens = _tokens(row["hypothesis"])
            best, best_score = None, 0.0
            if len(split_units(row["premise"])) > TRAP_MAX_UNITS - 1:
                continue
            for donor in donors:
                if donor["premise"] == row["premise"] or donor["label"] == "entailed":
                    continue
                cand = split_units(donor["premise"])[-1]
                cand_tokens = _tokens(cand)
                if not cand_tokens:
                    continue
                score = len(hyp_tokens & cand_tokens) / len(hyp_tokens | cand_tokens)
                if score > best_score:
                    best, best_score = cand, score
            if best is None or best_score < 0.15:
                continue
            yield make_task(
                f"{row['premise']} {best}",
                row["hypothesis"],
                row["label"],
                origin="synth",
                variant=kind,
                source_id=stable_seed(seed_salt, best),
            )
        made += 1


def build_manifest(cfg: Any) -> dict:
    """Build `tasks/manifest.jsonl` for a config. Idempotent and deterministic.

    Returns a summary dict (counts by split/variant) for the run report.
    """
    tp = cfg.taskpool
    n_tasks = int(tp.get("n_tasks", 120))
    salt = cfg.split_salt
    synth_frac = float(tp.get("synth_fraction", 0.34))
    seed_salt = tp.get("seed_salt", "taskpool-v1")

    data_dir = Path(tp.get("data_dir", "data")) / "raw"
    if not data_dir.is_absolute():
        data_dir = cfg.repo_root / data_dir

    rows = _binary_rows(load_snli(data_dir, tp.get("snli_split", "test")), n_tasks, seed_salt)

    n_synth = int(round(n_tasks * synth_frac))
    n_base = n_tasks - n_synth

    tasks: list[dict] = []
    for row in rows[:n_base]:
        try:
            tasks.append(
                make_task(
                    row["premise"],
                    row["hypothesis"],
                    row["label"],
                    origin="snli",
                    variant="base",
                    source_id=str(stable_seed(seed_salt, row["premise"]) % 10**8),
                )
            )
        except ValueError:
            continue

    donors = [r for r in rows if r not in rows[:n_base]] or rows
    tasks.extend(_adversarial_variants(rows[n_base:], donors, n_synth, seed_salt))

    # Content identity first, then the salted split. Identity ignores `origin`
    # and `split`, so a task cannot smuggle itself across the wall by being
    # rebuilt from another source.
    for task in tasks:
        task["split"] = split_of(task["task_id"], salt, cfg.eval_pct)
        task["prompt_hash"] = None  # filled by the compiler; kept for schema parity

    # Dedup by content hash. Two SNLI rows can hash identically (duplicate
    # premise+hypothesis pairs that survived the earlier filter); keeping both
    # would silently double-weight one task in the dataset and make the manifest
    # count a lie.
    deduped: dict[str, dict] = {}
    for task in tasks:
        deduped.setdefault(task["task_id"], task)
    duplicates = len(tasks) - len(deduped)
    tasks = list(deduped.values())

    tasks.sort(key=lambda t: (t["split"], t["task_id"]))
    cfg.ensure_dirs()
    from refinery.common.jsonl import write_json  # local import: keeps module import cheap

    manifest = cfg.manifest_path
    manifest.write_text(
        "".join(json.dumps(t, sort_keys=True, ensure_ascii=False) + "\n" for t in tasks),
        encoding="utf-8",
    )

    train_ids = {t["task_id"] for t in tasks if t["split"] == "train"}
    eval_ids = {t["task_id"] for t in tasks if t["split"] == "eval"}
    summary = {
        "n_tasks": len(tasks),
        "n_train": len(train_ids),
        "n_eval": len(eval_ids),
        "split_overlap": len(train_ids & eval_ids),
        "duplicates_dropped": duplicates,
        "by_variant": {
            v: sum(1 for t in tasks if t["prompt_variant"] == v) for v in VARIANTS
        },
        "by_origin": {
            o: sum(1 for t in tasks if t["origin"] == o)
            for o in {t["origin"] for t in tasks}
        },
        "manifest": str(manifest),
    }
    write_json(cfg.tasks_dir / "summary.json", summary)
    return summary


def iter_tasks(cfg: Any, split: str | None = None) -> Iterator[dict]:
    """Stream tasks from the manifest, optionally filtered by split."""
    from refinery.common.jsonl import read_jsonl

    for task in read_jsonl(cfg.manifest_path):
        if split is None or task.get("split") == split:
            yield task


def seed_messages_for(task: dict) -> list[dict]:
    """Convenience re-export so callers need not import the renderer directly."""
    return seed_messages(task)
