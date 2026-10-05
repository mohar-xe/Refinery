"""Eval harness: identical loop, identical caps, different completion backend.

The important structural decision here is that **the eval harness is the farm's
harness**. `run_one()` from `farm.harness` is reused verbatim; only the object
that produces assistant turns changes. That means a contender cannot accidentally
get a friendlier evaluation than the teacher:

  * same system prompt, same tool schema, same message rendering
  * same per-run caps (steps, requests, tokens)
  * same trajectory record, scored by the same verifier

A frontier plot built from anything else is a plot of two different experiments.
"""

from __future__ import annotations

import re
import time
from typing import Any

import torch

from refinery.common.protocol import parse_assistant
from refinery.compiler.render import render_messages
from refinery.farm.harness import run_one
from refinery.farm.teacher import HeuristicTeacher, OpenRouterTeacher, Teacher, TeacherReply

__all__ = ["StudentTeacher", "build_contender", "evaluate"]


class StudentTeacher(Teacher):
    """The distilled student, wrapped as a `Teacher`.

    Sampling with temperature (not greedy) because the eval reports pass@1 over
    seeds; a greedy decode would report pass@1 for a deterministic system and hide
    the variance that the frontier claim depends on.
    """

    name = "student"

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        max_new_tokens: int = 48,
        temperature: float = 0.8,
        top_p: float = 0.95,
        context_override: int | None = None,
    ) -> None:
        self.model = model.eval()
        self.tok = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        #: Shrinks the student's effective context. The API contenders cannot be
        #: shrunk, so this is only ever used for a *within-student* stress test,
        #: never to claim the student beat the teacher under a handicap.
        self.context_override = context_override
        self.calls = 0
        self.generated_tokens = 0
        self.model_seconds = 0.0
        self.truncated_turns = 0

    @torch.no_grad()
    def complete(self, messages: list[dict], *, temperature: float | None = None) -> TeacherReply:
        temp = self.temperature if temperature is None else temperature
        prompt = render_messages(messages)
        ids = self.tok.encode(prompt)
        context = self.context_override or self.model.arch.context
        # Truncate from the left: the instruction block is fixed, so the tail is
        # what carries the current state. Same policy at train and eval time.
        ids = ids[-context:]

        started = time.monotonic()
        out: list[int] = []
        stop = False
        for _ in range(self.max_new_tokens):
            logits = self.model(torch.tensor([ids], dtype=torch.long))[0, -1]
            next_id = self._sample(logits, temp)
            if next_id == self.tok.stoi["<|im_end|>"]:
                break
            if len(ids) >= context:
                stop = True
                break
            ids.append(next_id)
            out.append(next_id)
            # Stop as soon as the turn is functionally complete rather than
            # waiting for the cap. Without this a student that never emits
            # <|im_end|> costs 48 forwards per call *and* is indistinguishable
            # from one that merely rambles — early stop makes the failure cheap to
            # observe instead of merely slow.
            if _turn_complete(self.tok.decode(out)):
                stop = True
                break
        self.truncated_turns += int(stop and not _turn_complete(self.tok.decode(out)))

        elapsed = time.monotonic() - started
        self.calls += 1
        self.generated_tokens += len(out)
        self.model_seconds += elapsed

        return TeacherReply(
            content=self.tok.decode(out),
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(out),
            latency_s=elapsed,
            model=f"student-{self.model.non_embedding_params()/1e6:.2f}M",
        )

    def _sample(self, logits: torch.Tensor, temperature: float) -> int:
        if temperature <= 0:
            return int(torch.argmax(logits))
        probs = torch.softmax(logits / temperature, dim=-1)
        # nucleus sampling: the student is undertrained and a long tail produces
        # syntactically valid but semantically empty tool calls.
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        cutoff = int((cumulative < self.top_p).sum()) + 1
        choice = torch.multinomial(sorted_probs[:cutoff], 1)
        return int(sorted_idx[choice])


#: The turn is done once a complete CITE + ANSWER block has been emitted.
_DONE_RE = re.compile(r"CITE\s*:\s*\d+[^\n]*\n\s*ANSWER\s*:\s*(?:not[_ ])?entailed\s*$", re.I)


def _turn_complete(text: str) -> bool:
    return bool(_DONE_RE.search(text.strip()))


def build_contender(
    name: str, cfg: Any, *, seed: int = 0, context_override: int | None = None
) -> tuple[Teacher, dict]:
    """Return (teacher-like object, metadata) for a contender id.

    `student` and `student_no_curriculum` load whichever adapter the trainer wrote,
    so the curriculum ablation is evaluated through exactly the same code path.
    """
    teacher_cfg = cfg.teacher

    if name in ("heuristic", "majority"):
        return HeuristicTeacher(teacher_cfg, seed=seed), {"contender": name, "kind": "heuristic"}

    if name == "teacher":
        client = OpenRouterTeacher(teacher_cfg)
        if not client.available():
            raise RuntimeError("contender 'teacher' needs OPENROUTER_API_KEY")
        return client, {"contender": name, "kind": "api", "model": teacher_cfg.model}

    # Any trained adapter directory is a valid contender, not just ones whose name
    # starts with "student" — the windowing ablation arms are named win_*, and a
    # name-prefix whitelist silently made them unevaluable.
    if name.startswith("student") or (cfg.model_dir / name / "student.pt").exists():
        from refinery.trainer.model import build_model
        from refinery.trainer.tokenizer import WordTokenizer

        model_dir = cfg.model_dir / name
        ckpt = torch.load(model_dir / "student.pt", map_location="cpu", weights_only=False)
        model = build_model(_arch_from(ckpt["arch"]), ckpt["vocab_size"])
        model.load_state_dict(ckpt["state_dict"])
        tokenizer = WordTokenizer.load(model_dir / "tokenizer.json")
        return (
            StudentTeacher(model, tokenizer, context_override=context_override),
            {
                "contender": name,
                "context": context_override or arch_context(cfg),
                "kind": "local",
                "non_embedding_params": model.non_embedding_params(),
                "total_params": model.total_params(),
                "curriculum": ckpt.get("curriculum"),
                "strategy": ckpt.get("strategy"),
            },
        )

    raise ValueError(f"unknown contender: {name!r}")


def _arch_from(data: dict):
    from refinery.config import ArchCfg

    return ArchCfg(**data)


def arch_context(cfg: Any) -> int:
    return int(cfg.trainer.arch.context)


def evaluate(
    contender: str,
    tasks: list[dict],
    cfg: Any,
    *,
    seeds: int = 3,
    split: str = "eval",
    out_path: Any = None,
    context_override: int | None = None,
) -> dict:
    """Run one contender over held-out tasks x seeds, then score with the verifier."""
    from refinery.common.jsonl import append_unique
    from refinery.verifier.gate import verify

    backend, meta = build_contender(contender, cfg, context_override=context_override)
    teacher_cfg = cfg.teacher
    results: list[dict] = []

    for task in tasks:
        for seed in range(seeds):
            outcome = run_one(backend, task, seed, cfg, teacher_cfg=teacher_cfg)
            verdict = verify(task, outcome.record)
            record = {
                "result_id": f"{contender}:{seed}:{outcome.record['run_id']}",
                "contender": contender,
                "seed": seed,
                "task_id": task["task_id"],
                "run_id": outcome.record["run_id"],
                "variant": task.get("prompt_variant"),
                "gold": task.get("label"),
                "predicted": verdict.get("predicted"),
                "passed": verdict["reason_code"] == "VERIFIED_OK",
                "label_correct": verdict.get("predicted") == task.get("label"),
                "reason_code": verdict["reason_code"],
                "valid_tool_call": _valid_tool_call(outcome.record),
                "steps": outcome.record["steps"],
                "requests": outcome.record["requests"],
                "latency_s": outcome.record["wall_s"],
                "cost_usd": outcome.record["cost_usd"],
                "tokens": outcome.record["tokens"]["total"],
                **meta,
            }
            results.append(record)

    if out_path is not None:
        append_unique(out_path, results, key="result_id")

    from evalkit.metrics import summarize_contender

    summary = summarize_contender(results)
    summary["contender"] = contender
    summary["seeds"] = seeds
    summary["n_tasks"] = len(tasks)
    summary.update({k: v for k, v in meta.items() if k != "contender"})
    if isinstance(backend, StudentTeacher):
        summary["model_seconds"] = round(backend.model_seconds, 2)
        summary["local_inference_calls"] = backend.calls
        summary["generated_tokens"] = backend.generated_tokens
        summary["truncated_turns"] = backend.truncated_turns
    return summary


def _valid_tool_call(run: dict) -> bool:
    """Did the trajectory contain at least one syntactically valid tool call?

    Reported separately from correctness because the two fail differently: a model
    that never emits a valid call scores ~0 while looking perfectly calibrated
    (spec trap #5).
    """
    for msg in run.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        parsed = parse_assistant(msg.get("content", ""))
        if parsed.lookups or parsed.answer:
            return True
    return False
