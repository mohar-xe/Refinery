"""Contamination tests — the guard for spec trap #1.

The property is stronger than "we checked for duplicates": identity is a hash of
task content, and the split is a hash of that identity, so cross-split duplication
is *impossible by construction*. These tests assert the construction holds, so a
refactor that breaks it fails here rather than silently inflating the eval set.
"""

from __future__ import annotations

from refinery.common.hashing import canonical_bytes, content_id, split_of, stable_seed, task_identity

TASK = {
    "task_id": "sha256:whatever",
    "origin": "snli",
    "split": "train",
    "prompt_variant": "base",
    "premise": "A boy sits. He strums a guitar.",
    "hypothesis": "A boy plays a guitar.",
    "evidence": [
        {"evidence_id": "e0", "index": 0, "text": "A boy sits."},
        {"evidence_id": "e1", "index": 1, "text": "He strums a guitar."},
    ],
}


def test_identity_ignores_bookkeeping_fields():
    """Moving a task between splits or relabelling its origin must not change
    what the task *is* — otherwise the hash wall is trivially bypassed."""
    moved = {**TASK, "split": "eval", "origin": "swe-smith", "task_id": "sha256:different"}
    assert task_identity(moved) == task_identity(TASK)


def test_identity_changes_with_content():
    edited = {**TASK, "hypothesis": "A boy plays a drum."}
    assert task_identity(edited) != task_identity(TASK)


def test_split_is_deterministic():
    tid = task_identity(TASK)
    assert split_of(tid, "salt-a", 25) == split_of(tid, "salt-a", 25)


def test_salt_changes_the_split():
    """Re-salting must be able to build a fresh eval split — that is the
    remediation for a base model that scores suspiciously high zero-shot."""
    tid = task_identity(TASK)
    salts = {split_of(tid, f"salt-{i}", 25) for i in range(40)}
    assert salts == {"train", "eval"}


def test_split_is_roughly_proportional():
    train = sum(
        1 for i in range(2000) if split_of(f"sha256:task-{i}", "salt", 20) == "train"
    )
    assert 0.74 < train / 2000 < 0.86  # target 80%, wide band for n=2000


def test_canonical_bytes_is_key_order_independent():
    assert canonical_bytes({"b": 1, "a": 2}) == canonical_bytes({"a": 2, "b": 1})


def test_content_id_is_prefixed():
    assert content_id(TASK).startswith("sha256:")


def test_stable_seed_is_reproducible_and_bounded():
    a = stable_seed("farm-v1", 3)
    assert a == stable_seed("farm-v1", 3)
    assert a != stable_seed("farm-v1", 4)
    assert 0 <= a < 2**63
