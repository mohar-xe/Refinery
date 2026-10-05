"""Content addressing.

Task identity is derived from task *content*, never from a sequence number or a
filename. This is the mechanism behind trap #1 (contamination leakage): if two
tasks are "the same task", they have the same id, so a hash-walled split cannot
leak between them — not "we checked and found no duplicates", but "it is
impossible by construction".

The split key is salted so that an adversary who knows the task pool cannot
predict which tasks land in eval, and so we can re-salting to build a fresh eval
split when the base model scores suspiciously high zero-shot.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = [
    "canonical_bytes",
    "content_id",
    "task_identity",
    "split_of",
    "stable_seed",
]

# Fields that define a task's meaning. Two records agreeing on all of these are
# the same task, regardless of provenance, ordering, or filename.
IDENTITY_FIELDS = ("premise", "hypothesis", "evidence", "prompt_variant")


def canonical_bytes(obj: Any) -> bytes:
    """Deterministic JSON encoding: sorted keys, no incidental whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_id(obj: Any) -> str:
    """Content id of any JSON-serializable object, prefixed with its algorithm."""
    return f"sha256:{_sha256(canonical_bytes(obj))}"


def task_identity(record: dict) -> str:
    """Identity of a task = hash of its semantic content only.

    Deliberately excludes `origin`, `split`, and any bookkeeping field: adding a
    task to the eval split must not change what the task *is*, or the hash wall
    would be trivially bypassed.
    """
    return content_id({k: record[k] for k in IDENTITY_FIELDS if k in record})


def split_of(task_id: str, salt: str, eval_pct: int = 20) -> str:
    """Assign a task to train or eval by hashing (task_id, salt).

    Deterministic, uniform, and independent of insertion order — so the split is
    reproducible from the manifest alone, with no split file to drift.
    """
    digest = _sha256(f"{task_id}|{salt}".encode())
    bucket = int(digest[:8], 16) % 100
    return "eval" if bucket < eval_pct else "train"


def stable_seed(*parts: Any) -> int:
    """A reproducible 63-bit seed from arbitrary parts.

    Used wherever "random" must be reproducible across processes and machines —
    task construction, sample selection, curriculum shuffling.
    """
    return int(_sha256(canonical_bytes(list(parts)))[:16], 16) >> 1
