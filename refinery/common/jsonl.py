"""Append-only JSONL stores with resumable writes (LLD.md D-009).

Every stage writes one JSON object per line and never rewrites history. Two
properties matter on a phone, where Android can kill the process at any moment:

  1. A partial write can only damage the *last* line, and a damaged last line is
     detected and skipped on read rather than crashing the stage.
  2. Stages are idempotent: `append_unique` skips records whose id is already
     present, so re-running stage N after stage N-1 completed does nothing.

No database, no lock files: a lock left behind by a killed process is a hazard
that JSONL simply does not have.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

__all__ = ["read_jsonl", "append_jsonl", "append_unique", "write_json", "read_json", "count_lines"]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: str | Path, *, skip_broken: bool = True) -> Iterator[dict]:
    """Yield records from a JSONL file. Missing file yields nothing."""
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # A torn final line is the expected shape of "the process died
                # mid-append". Do not lose the rest of the file over it.
                if not skip_broken:
                    raise
                print(f"[jsonl] skipping unparseable line {p}:{lineno}")


def _id_of(record: dict) -> str | None:
    for key in ("task_id", "run_id", "sample_id", "id"):
        if key in record:
            return str(record[key])
    return None


def append_jsonl(path: str | Path, records: Iterable[dict]) -> int:
    """Append records verbatim. Returns the number written."""
    p = Path(path)
    _ensure_parent(p)
    written = 0
    with p.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            written += 1
        fh.flush()
        os.fsync(fh.fileno())
    return written


def append_unique(path: str | Path, records: Iterable[dict], *, key: str | None = None) -> int:
    """Append only records whose id is not already present. Makes stages resumable.

    `key` overrides id-field detection; otherwise the first of
    task_id/run_id/sample_id/id present on the record is used. Records without an
    identifiable id are always appended (we would rather duplicate than drop data).
    """
    p = Path(path)
    seen: set[str] = set()
    for existing in read_jsonl(p):
        ident = key and existing.get(key) or _id_of(existing)
        if ident is not None:
            seen.add(ident)

    fresh: list[dict] = []
    for record in records:
        ident = key and record.get(key) or _id_of(record)
        if ident is not None and ident in seen:
            continue
        if ident is not None:
            seen.add(ident)
        fresh.append(record)

    return append_jsonl(p, fresh) if fresh else 0


def write_json(path: str | Path, obj: Any) -> Path:
    """Atomically write a single JSON document (write to temp, then rename)."""
    p = Path(path)
    _ensure_parent(p)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    tmp.replace(p)
    return p


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def count_lines(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    with p.open("rb") as fh:
        return sum(1 for _ in fh)
