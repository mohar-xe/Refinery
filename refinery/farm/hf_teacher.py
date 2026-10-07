"""HuggingFace teacher — an open model on our own GPU, behind the same interface.

Why this exists
---------------
The OpenRouter free tier allows **50 requests/day** on the account (`/api/v1/auth/key`
reports `free_model_daily_requests: {limit: 50}`). At ~2 requests per agent run that
is ~25 runs/day, so a frontier API teacher cannot be the data engine — the full
farm would take 11 days. Space Bunny Alpha used to allow 1000/day and no longer has
any endpoints.

The response is not to run smaller. It is to move generation onto GPU we already
have: 2x T4 / 30h per week on Kaggle versus 50 API calls per day. And the
verification gate is exactly what makes distilling from an *open* model legitimate:
unverified self-generated data is the AgentTuning failure mode, verified
self-generated data is STaR with a small teacher.

Design
------
This class implements the same `complete()` surface as the API teachers, so
`farm.harness.run_one` drives it unchanged. Two decisions carry the transfer:

  * **The teacher is prompted with our renderer, not its own chat template.** The
    student is trained on exactly the byte sequence `render_messages` produces, so
    a teacher that reads the same bytes generates the surface form the student will
    be supervised on. That requires a ChatML-compatible model (Qwen3 and friends);
    `use_native_template=True` flips it, and the difference is worth measuring
    rather than assuming.
  * **The teacher sees only the conversation**, never the task record, so it is
    under the same information restriction the student will face at eval time.
"""

from __future__ import annotations

import re
import time
from typing import Any

from refinery.common.protocol import parse_assistant
from refinery.compiler.render import render_messages
from refinery.farm.teacher import TeacherReply

__all__ = ["HFTeacher"]

#: Leading artefacts seen from the first GPU run, each a real failure mode.
_ROLE_ECHO = re.compile(r"^\s*(?:<\|[^|]*\|>\s*)?(?:assistant|model)\b\s*[:\n]", re.IGNORECASE)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_OPEN_THINK = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)


def _clean_completion(text: str) -> str:
    """Strip what the model emits around the answer rather than around the protocol.

    Measured on the first 4B run, 267 trajectories:
      * every completion began with a literal ``assistant\n`` role echo, because the
        prompt's trailing ``<|im_start|>assistant`` block was not presented the way
        Qwen's template presents it, so the model repaired the transcript by writing
        the role label as text;
      * reasoning models emit ``<think>`` blocks, which consumed the whole token
        budget on 86 of 267 runs (STEP_CAP) before ever reaching the answer.

    Both are stripped here rather than in the parser, because they are properties of
    *this* backend. `parse_assistant` stays strict: it should reject a malformed
    trajectory, not a well-formed one wrapped in model boilerplate.
    """
    text = _THINK.sub("", text)
    text = _OPEN_THINK.sub("", text)
    text = _ROLE_ECHO.sub("", text)
    return text.strip()


#: Candidates tried in order by `resolve_model_id`. Qwen3 is first because its
#: tokenizer and chat format are the ChatML this project renders, and 4B fits a
#: T4 in fp16 with room for KV cache.
DEFAULT_MODEL_CANDIDATES = (
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-1.7B",
)


def resolve_model_id(candidates: tuple[str, ...] = DEFAULT_MODEL_CANDIDATES) -> str:
    """First candidate that exists on the Hub. Raises with the full list if none do."""
    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
        raise RuntimeError("huggingface_hub is required to resolve a teacher model") from exc

    api = HfApi()
    for model_id in candidates:
        try:
            api.model_info(model_id)
            return model_id
        except Exception:  # noqa: BLE001 - any hub error means "not available"
            continue
    raise RuntimeError(
        f"none of the candidate teacher models resolve on the Hub: {list(candidates)}"
    )


class HFTeacher:
    """A local open-weight model behind the farm's `Teacher` interface."""

    name = "hf"

    def __init__(
        self,
        model_id: str | None = None,
        *,
        dtype: str = "fp16",
        device: str = "cuda",
        max_new_tokens: int = 320,
        enable_thinking: bool = False,
        temperature: float = 0.8,
        top_p: float = 0.95,
        use_native_template: bool = False,
        seed: int = 0,
    ) -> None:
        self.model_id = model_id
        self.dtype = dtype
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.use_native_template = use_native_template
        self.enable_thinking = enable_thinking
        self.seed = seed

        self._tok: Any = None
        self._model: Any = None
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.generate_seconds = 0.0

    # -- lifecycle -----------------------------------------------------------
    def load(self) -> "HFTeacher":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.model_id is None:
            self.model_id = resolve_model_id()

        torch_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[self.dtype]

        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map=self.device,
        )
        self._model.eval()
        return self

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def stats(self) -> dict:
        return {
            "teacher": self.name,
            "model": self.model_id,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "generate_seconds": round(self.generate_seconds, 2),
            "native_template": self.use_native_template,
        }

    # -- the Teacher interface ------------------------------------------------
    def complete(self, messages: list[dict], *, temperature: float | None = None) -> TeacherReply:
        if not self.loaded:
            self.load()

        import torch

        temp = self.temperature if temperature is None else temperature
        prompt = self._render(messages)
        encoded = self._tok(prompt, return_tensors="pt").to(self._model.device)

        started = time.monotonic()
        with torch.no_grad():
            output = self._model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=temp > 0,
                temperature=max(temp, 1e-5),
                top_p=self.top_p,
                pad_token_id=self._tok.pad_token_id or self._tok.eos_token_id,
            )
        elapsed = time.monotonic() - started

        new_tokens = output[0][encoded["input_ids"].shape[1]:]
        content = _clean_completion(self._tok.decode(new_tokens, skip_special_tokens=True))
        self.calls += 1
        self.prompt_tokens += int(encoded["input_ids"].shape[1])
        self.completion_tokens += int(new_tokens.shape[0])
        self.generate_seconds += elapsed

        return TeacherReply(
            content=content.strip(),
            prompt_tokens=int(encoded["input_ids"].shape[1]),
            completion_tokens=int(new_tokens.shape[0]),
            latency_s=elapsed,
            model=self.model_id or "hf",
        )

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Local GPU is not billed per token. The dollar figure is 0 and the tokens
        are still recorded, because the frontier's interesting axis is
        tokens-per-verified-trajectory, not price (LLD.md D-013)."""
        return 0.0

    # -- internals ------------------------------------------------------------
    def _render(self, messages: list[dict]) -> str:
        """Our renderer by default; the model's own chat template on request.

        The default is deliberate and it is the whole transfer story: the student is
        supervised on `render_messages(...)`, so the teacher must read the same
        bytes or it is generating a surface form the student never saw.
        """
        if not self.use_native_template:
            return render_messages(messages)

        payload = [
            {"role": m["role"], "content": m.get("content", "")}
            for m in messages
            if m["role"] in ("system", "user", "assistant")
        ]
        # Tool results are carried as a user turn for native templates, which have
        # no tool role of their own in this path.
        merged: list[dict] = []
        for item in payload:
            if item["role"] == "user" and merged and merged[-1]["role"] == "user":
                merged[-1]["content"] += "\n\n" + item["content"]
            else:
                merged.append(item)
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            return self._tok.apply_chat_template(merged, enable_thinking=self.enable_thinking, **kwargs)
        except TypeError:
            # Older templates have no thinking switch; strip the block downstream.
            return self._tok.apply_chat_template(merged, **kwargs)
