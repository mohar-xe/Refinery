"""Single source of truth for scale (LLD.md D-001).

There is no `if toy:` anywhere in `refinery/`. Toy and full scale differ only in
the values loaded here. A stage that needs to branch on scale is a stage that has
silently become two programs, and the toy run stops being evidence about the real
system.

`configs/toy.json` and `configs/full.json` are the only two files that should ever
need to change to move between scales.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Config", "TeacherCfg", "Caps", "WindowCfg", "ArchCfg", "OptimCfg", "TrainerCfg", "load"]

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class TeacherCfg:
    provider: str = "openrouter"
    model: str = "stealth/space-bunny-alpha"
    base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens: int = 320
    timeout_s: float = 90.0
    #: OpenRouter sends these so providers can cache the shared prefix. With a
    #: stable system prompt this is the cheapest farm cost control available.
    prompt_cache: bool = True
    #: Dollars per 1k tokens. 0.0 for the free tier — reported honestly rather
    #: than replaced by a list price (LLD.md D-013).
    price_per_1k_prompt: float = 0.0
    price_per_1k_completion: float = 0.0
    max_retries: int = 2

    @property
    def priced(self) -> bool:
        return self.price_per_1k_prompt > 0 or self.price_per_1k_completion > 0


@dataclass(frozen=True)
class Caps:
    """Hard per-run limits, enforced *before* the next call is issued."""

    max_steps: int = 6
    max_requests: int = 4
    max_tokens: int = 1200
    max_usd: float = 0.0

    def exceeded_by(self, *, requests: int, tokens: int, usd: float) -> str | None:
        if requests >= self.max_requests:
            return "REQUEST_CAP"
        if tokens >= self.max_tokens:
            return "STEP_CAP"
        if self.max_usd > 0 and usd >= self.max_usd:
            return "STEP_CAP"
        return None


@dataclass(frozen=True)
class WindowCfg:
    strategy: str = "windowed"  # full_context | windowed | naive_truncated
    keep_last_n: int = 3
    digest_head_chars: int = 80
    naive_keep_chars: int = 1200


@dataclass(frozen=True)
class ArchCfg:
    n_layer: int = 3
    d_model: int = 160
    n_head: int = 4
    d_ff: int = 640
    context: int = 512
    max_vocab: int = 8000
    tied_embeddings: bool = True
    dropout: float = 0.0

    def non_embedding_params(self) -> int:
        """Params excluding the embedding matrix — the number we report as '1M'.

        At this scale the embedding table is a large fraction of the total, so
        quoting the total would flatter the model (LLD.md D-015a).
        """
        attn = 4 * self.d_model * self.d_model
        ffn = 2 * self.d_model * self.d_ff
        return self.n_layer * (attn + ffn)


@dataclass(frozen=True)
class OptimCfg:
    epochs: int = 12
    batch_size: int = 32
    lr: float = 3e-3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 7


@dataclass(frozen=True)
class TrainerCfg:
    arch: ArchCfg = field(default_factory=ArchCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    #: Curriculum stage 1 selects the shortest `stage1_fraction` of trajectories
    #: rather than everything under an absolute step count. The absolute version
    #: silently selected *nothing* on the toy corpus (min n_steps was 3 against a
    #: threshold of 2), so stage 1 never ran and the curriculum ablation was
    #: measuring nothing. A fraction cannot silently no-op (LLD.md D-011).
    stage1_fraction: float = 0.4
    stage1_max_steps: int | None = None
    stage1_epochs: int = 4
    #: "scratch" trains `arch` from scratch. "base+lora" is the flagship path and
    #: is NOT implemented yet (LLD.md D-006) — the trainer refuses it loudly rather
    #: than silently training the wrong thing.
    student_init: str = "scratch"


@dataclass(frozen=True)
class Config:
    name: str
    root: Path
    repo_root: Path = REPO_ROOT
    taskpool: dict[str, Any] = field(default_factory=dict)
    farm: dict[str, Any] = field(default_factory=dict)
    compiler: dict[str, Any] = field(default_factory=dict)
    trainer: TrainerCfg = field(default_factory=TrainerCfg)
    evalkit: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    # ---- convenience accessors used across stages ---------------------------
    @property
    def teacher(self) -> TeacherCfg:
        return TeacherCfg(**self.farm.get("teacher", {}))

    @property
    def caps(self) -> Caps:
        return Caps(**self.farm.get("caps", {}))

    @property
    def window(self) -> WindowCfg:
        return WindowCfg(**self.compiler.get("window", {}))

    @property
    def split_salt(self) -> str:
        return self.taskpool.get("split_salt", "default")

    @property
    def eval_pct(self) -> int:
        return int(self.taskpool.get("eval_pct", 20))

    @property
    def samples_per_task(self) -> int:
        return int(self.farm.get("samples_per_task", 4))

    # ---- derived directories -------------------------------------------------
    @property
    def tasks_dir(self) -> Path:
        return self.root / "tasks"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def dataset_dir(self) -> Path:
        return self.root / "dataset"

    @property
    def model_dir(self) -> Path:
        return self.root / "model"

    @property
    def eval_dir(self) -> Path:
        return self.root / "eval"

    @property
    def manifest_path(self) -> Path:
        return self.tasks_dir / "manifest.jsonl"

    @property
    def verdicts_path(self) -> Path:
        return self.root / "verdicts.jsonl"

    @property
    def ledger_path(self) -> Path:
        return self.root / "ledger.jsonl"

    def ensure_dirs(self) -> None:
        for d in (
            self.tasks_dir,
            self.runs_dir,
            self.dataset_dir,
            self.model_dir,
            self.eval_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


def _merge_dataclass(cls, data: dict):
    """Instantiate a dataclass from a dict, ignoring unknown keys loudly."""
    known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown config keys {sorted(unknown)}")
    return cls(**{k: v for k, v in data.items() if k in known})


def load(path: str | Path) -> Config:
    """Load a scale config (JSON). Unknown keys are an error, not a shrug."""
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))

    trainer_raw = raw.get("trainer", {})
    trainer = TrainerCfg(
        student_init=trainer_raw.get("student_init", "scratch"),
        arch=_merge_dataclass(ArchCfg, trainer_raw.get("arch", {})),
        optim=_merge_dataclass(OptimCfg, trainer_raw.get("optim", {})),
        stage1_fraction=float(trainer_raw.get("stage1_fraction", 0.4)),
        stage1_max_steps=(
            int(trainer_raw["stage1_max_steps"]) if trainer_raw.get("stage1_max_steps") else None
        ),
        stage1_epochs=int(trainer_raw.get("stage1_epochs", 4)),
    )

    root = Path(raw.get("paths", {}).get("root", f"runs/{raw.get('name', 'unnamed')}"))
    if not root.is_absolute():
        root = (REPO_ROOT / root).resolve()

    return Config(
        name=raw.get("name", p.stem),
        root=root,
        taskpool=raw.get("taskpool", {}),
        farm=raw.get("farm", {}),
        compiler=raw.get("compiler", {}),
        trainer=trainer,
        evalkit=raw.get("evalkit", {}),
        raw=raw,
    )
