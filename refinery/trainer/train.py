"""Trainer: curriculum, loss on assistant/tool-call tokens only, CPU-friendly.

Two policies live here and both are switches, not forks:

  * `kinds_allow` — which assistant segment kinds get loss (LLD.md D-011/D-012).
    Stage 1 = `{"tool_call"}` only; stage 2 = everything the renderer marked
    trainable. The mask comes from the renderer's own segmentation, so no second
    regex exists to drift.
  * `--no-curriculum` — the ablation arm. Same data, same steps, mixed from
    scratch; the difference in `valid_tool_call_rate` is the whole point of
    reporting the ablation.

Training data is the *verified* set only. There is no code path that trains on
unverified trajectories, which is the property that makes the dataset an artifact
rather than a pile of logs.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

#: Phone CPU, 8 cores. Without this, torch picks a default that is usually 1
#: thread on aarch64 and a 0.9M model trains ~4x slower than it should.
torch.set_num_threads(int(os.environ.get("REFINERY_THREADS", "8")))

from refinery.common.jsonl import read_jsonl
from refinery.config import ArchCfg, OptimCfg
from refinery.trainer.model import build_model
from refinery.trainer.tokenizer import WordTokenizer

__all__ = ["train", "TrainResult"]


@dataclass
class TrainResult:
    adapter_path: Path
    tokenizer_path: Path
    log_path: Path
    stats: dict


def _load_samples(cfg: Any, strategy: str) -> list[dict]:
    return list(read_jsonl(cfg.dataset_dir / f"train__{strategy}.jsonl"))


def _encode(
    tok: WordTokenizer, sample: dict, kinds: set[str] | None, context: int
) -> tuple[list[int], list[int]]:
    """Tokenize a sample, narrowing the loss mask to `kinds` when given.

    Truncation to `context` is the hard cap that makes the windowing policy
    load-bearing: without it a long trajectory would simply be cut from the tail,
    which is exactly the naive-truncation failure the ablation is meant to
    expose.
    """
    segments = []
    for seg in sample["segments"]:
        enriched = dict(seg)
        if kinds is not None:
            enriched["kinds_allow"] = sorted(kinds)
        segments.append(enriched)
    ids, mask = tok.encode_segments(segments)
    return ids[:context], mask[:context]


def _batches(items: list[tuple[list[int], list[int]]], batch_size: int, *, pad: int) -> list:
    """Pad into batches. Padded positions get mask 0 and label -100, so the
    shorter sequences in a batch do not contribute gradient from pad tokens."""
    out = []
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        width = max(len(ids) for ids, _ in chunk)
        ids_b, mask_b, labels_b = [], [], []
        for ids, mask in chunk:
            pad_len = width - len(ids)
            ids_b.append(ids + [pad] * pad_len)
            mask_b.append(mask + [0] * pad_len)
            labels_b.append([i if m else -100 for i, m in zip(ids, mask, strict=True)]
                            + [-100] * pad_len)
        out.append(
            (
                torch.tensor(ids_b, dtype=torch.long),
                torch.tensor(mask_b, dtype=torch.bool),
                torch.tensor(labels_b, dtype=torch.long),
            )
        )
    return out


def train(
    cfg: Any,
    *,
    strategy: str | None = None,
    curriculum: bool = True,
    out_name: str | None = None,
) -> TrainResult:
    """Train the student. Returns paths + a stats dict (the training log)."""
    if cfg.trainer.student_init != "scratch":
        raise NotImplementedError(
            f"student_init={cfg.trainer.student_init!r} is the flagship path (base model + "
            "LoRA) and is not implemented yet; see LLD.md D-006. Use student_init='scratch'."
        )

    strategy = strategy or cfg.window.strategy
    arch: ArchCfg = cfg.trainer.arch
    optim: OptimCfg = cfg.trainer.optim
    torch.manual_seed(optim.seed)

    samples = _load_samples(cfg, strategy)
    if not samples:
        raise RuntimeError(
            f"no training samples at {cfg.dataset_dir / f'train__{strategy}.jsonl'}; "
            "run stages 1-3 first"
        )

    # Vocab from train text only (tokenizer docstring: eval text would leak).
    tokenizer = WordTokenizer.build(
        ["".join(s["text"] for s in smp["segments"]) for smp in samples],
        max_vocab=arch.max_vocab,
    )

    # Curriculum stage 1 = the shortest trajectories, tool-call supervision only.
    short = _stage1_subset(samples, cfg.trainer.stage1_fraction, cfg.trainer.stage1_max_steps)
    log: list[dict] = []
    model = build_model(arch, tokenizer.vocab_size)
    model.train()

    stage_defs: list[tuple[str, list[dict], set[str] | None, int]] = []
    if curriculum and short:
        stage_defs.append(("stage1_tool_call_only", short, {"tool_call"}, cfg.trainer.stage1_epochs))
    stage_defs.append(("stage2_full", samples, None, optim.epochs))
    if not curriculum:
        stage_defs = [("mixed_from_scratch", samples, None, optim.epochs)]

    total_steps = _count_steps(stage_defs, tokenizer, arch.context, optim.batch_size, tokenizer.pad_id)
    opt = torch.optim.AdamW(
        model.parameters(), lr=optim.lr, weight_decay=optim.weight_decay, betas=(0.9, 0.95)
    )
    global_step = 0

    for stage_name, data, kinds, epochs in stage_defs:
        encoded = [
            (i, m) for i, m in (_encode(tokenizer, s, kinds, arch.context) for s in data) if len(i) > 1
        ]
        if not encoded:
            continue
        batches = _batches(encoded, optim.batch_size, pad=tokenizer.pad_id)
        t_stage = time.monotonic()

        for epoch in range(epochs):
            epoch_loss, seen = 0.0, 0
            for ids, mask, labels in batches:
                lr = _lr_at(global_step, total_steps, optim)
                for group in opt.param_groups:
                    group["lr"] = lr

                logits = model(ids)
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), optim.grad_clip)
                opt.step()

                supervised = int((labels[:, 1:] != -100).sum())
                epoch_loss += float(loss) * supervised
                seen += supervised
                global_step += 1

                log.append(
                    {
                        "stage": stage_name,
                        "epoch": epoch,
                        "step": global_step,
                        "loss": round(float(loss), 4),
                        "lr": round(lr, 6),
                        "supervised_tokens": supervised,
                    }
                )

            if seen:
                mean_loss = round(epoch_loss / seen, 4)
                log.append({"stage": stage_name, "epoch": epoch, "epoch_mean_loss": mean_loss})
                print(
                    f"[train] {stage_name} epoch {epoch + 1}/{epochs} "
                    f"loss={mean_loss} ({time.monotonic() - t_stage:.0f}s)",
                    file=sys.stderr,
                    flush=True,
                )

    name = out_name or ("student" if curriculum else "student_no_curriculum")
    model_dir = cfg.model_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = model_dir / "student.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "arch": vars(arch),
            "vocab_size": tokenizer.vocab_size,
            "strategy": strategy,
            "curriculum": curriculum,
            "system_prompt_version": "v1",
        },
        adapter_path,
    )
    tokenizer_path = tokenizer.save(model_dir / "tokenizer.json")
    log_path = model_dir / "training_log.jsonl"
    log_path.write_text(
        "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in log), encoding="utf-8"
    )

    stats = {
        "name": name,
        "strategy": strategy,
        "curriculum": curriculum,
        "n_samples": len(samples),
        "n_stage1_samples": len(short),
        "stage1_fraction": cfg.trainer.stage1_fraction,
        "stage1_max_n_steps": max((s["n_steps"] for s in short), default=None),
        "vocab_size": tokenizer.vocab_size,
        "non_embedding_params": model.non_embedding_params(),
        "total_params": model.total_params(),
        "total_steps": global_step,
        "final_loss": next(
            (e["loss"] for e in reversed(log) if "loss" in e),
            None,
        ),
        "mean_loss_last_epoch": _mean_last(log),
    }
    return TrainResult(
        adapter_path=adapter_path,
        tokenizer_path=tokenizer_path,
        log_path=log_path,
        stats=stats,
    )


def _lr_at(step: int, total: int, optim: OptimCfg) -> float:
    if step < optim.warmup_steps:
        return optim.lr * (step + 1) / max(optim.warmup_steps, 1)
    progress = (step - optim.warmup_steps) / max(total - optim.warmup_steps, 1)
    return 0.1 * optim.lr + 0.9 * optim.lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def _mean_last(log: list[dict]) -> float | None:
    losses = [e["loss"] for e in log if e.get("stage") and "loss" in e]
    if not losses:
        return None
    tail = losses[-min(len(losses), 50) :]
    return round(sum(tail) / len(tail), 4)


def _stage1_subset(samples: list[dict], fraction: float, max_steps: int | None) -> list[dict]:
    """Shortest `fraction` of trajectories, optionally capped in steps.

    Selection is by *rank* on n_steps rather than a threshold, so it cannot
    silently select nothing when the corpus shifts. Ties are broken by sample_id
    to keep the subset deterministic.
    """
    if fraction <= 0:
        return []
    ordered = sorted(samples, key=lambda s: (s["n_steps"], s["sample_id"]))
    keep = max(1, int(round(len(ordered) * min(fraction, 1.0))))
    subset = ordered[:keep]
    if max_steps is not None:
        bounded = [s for s in subset if s["n_steps"] <= max_steps]
        subset = bounded or subset
    return subset


def _count_steps(stage_defs, tokenizer: WordTokenizer, context: int, batch_size: int, pad: int) -> int:
    """Total optimizer steps, needed up front for the cosine schedule."""
    total = 0
    for _, data, kinds, epochs in stage_defs:
        encoded = [(i, m) for i, m in (_encode(tokenizer, s, kinds, context) for s in data) if len(i) > 1]
        if encoded:
            total += max(1, math.ceil(len(encoded) / batch_size)) * epochs
    return max(total, 1)


def eval_loss(
    model: Any,
    tokenizer: WordTokenizer,
    samples: list[dict],
    pad: int,
    context: int,
    batch_size: int = 16,
) -> float | None:
    """Mean loss on held-out verified trajectories (sanity metric, not the eval)."""
    encoded = [(i, m) for i, m in (_encode(tokenizer, s, None, context) for s in samples) if len(i) > 1]
    if not encoded:
        return None
    model.eval()
    total, seen = 0.0, 0
    with torch.no_grad():
        for ids, _mask, labels in _batches(encoded, batch_size, pad=pad):
            logits = model(ids)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total += float(loss)
            seen += int((labels[:, 1:] != -100).sum())
    model.train()
    return round(total / max(seen, 1), 4)
