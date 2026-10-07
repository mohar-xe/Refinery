"""Kaggle kernel: the whole Refinery pipeline on a GPU, start to finish.

Why a kernel rather than the phone, or the API
----------------------------------------------
The OpenRouter free tier allows 50 requests/day, so a frontier API teacher cannot
be the data engine: the legal farm is ~534 requests, i.e. 11 days. Kaggle gives
2x T4 and 30 GPU-hours a week. So generation *and* training both happen here.

The important property is that this script imports the same `refinery` modules the
phone runs. Nothing about the pipeline is reimplemented here — the kernel only
supplies a different completion backend (`HFTeacher`) and a different scale
config. If a stage behaved differently on Kaggle than locally, that would be a bug
in the stage, which is the point of keeping them identical.

Stages, in order, each resumable:
    taskpool -> farm -> verify -> compile -> train -> eval -> report

Artifacts written to /kaggle/working/runs/<name>/ and /kaggle/working/reports/, so
`kaggle kernels output` brings them back whole.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/kaggle/working/Refinery")
sys.path.insert(0, str(REPO))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HOME", "/kaggle/working/hf-cache")
# T4 is Ampere-era: no native bf16, so fp16 is the correct dtype here even though
# the phone trains in fp32.
os.environ.setdefault("REFINERY_THREADS", "4")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def clone_repo() -> Path:
    if REPO.exists():
        log("repo already present")
        return REPO
    REPO.parent.mkdir(parents=True, exist_ok=True)
    log("cloning Refinery")
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/mohar-xe/Refinery.git", str(REPO)],
        check=True,
    )
    return REPO


def install_deps() -> None:
    log("installing transformers")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "transformers", "accelerate",
         "huggingface_hub", "safetensors"],
        check=True,
    )


def probe_gpu() -> dict:
    """`nvidia-smi` is not on PATH inside Kaggle kernels; ask torch instead."""
    import torch

    info = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    if info["cuda_available"]:
        props = torch.cuda.get_device_properties(0)
        info["total_memory_gb"] = round(props.total_memory / 1024**3, 2)
        info["capability"] = f"{props.major}.{props.minor}"
    log(f"gpu probe: {info}")
    return info


def build_teacher(model_id: str | None, native_template: bool):
    from refinery.farm.hf_teacher import HFTeacher, resolve_model_id

    resolved = model_id or resolve_model_id()
    log(f"loading teacher {resolved}")
    started = time.monotonic()
    teacher = HFTeacher(resolved, dtype="fp16", device="cuda",
                        use_native_template=native_template).load()
    log(f"teacher ready in {time.monotonic() - started:.1f}s")
    return teacher


def run_farm(cfg, teacher, tasks_limit: int | None, k: int) -> dict:
    from refinery.farm.harness import run_one
    from refinery.taskpool.build import iter_tasks

    tasks = list(iter_tasks(cfg, split="train"))
    if tasks_limit:
        tasks = tasks[:tasks_limit]
    log(f"farm: {len(tasks)} tasks x k={k}")

    records, failures = [], []
    started = time.monotonic()
    for i, task in enumerate(tasks, 1):
        for sample in range(k):
            outcome = run_one(teacher, task, sample, cfg, teacher_cfg=cfg.teacher)
            records.append(outcome.record)
        if i % 5 == 0 or i == len(tasks):
            rate = i / max(time.monotonic() - started, 1e-6)
            log(f"  {i}/{len(tasks)} tasks, {len(records)} runs, {rate:.2f} task/s")
    if not records:
        raise SystemExit("farm produced no runs")

    from refinery.common.jsonl import append_unique

    shard = cfg.runs_dir / "shard-00.jsonl"
    written = append_unique(shard, records, key="run_id")
    stats = teacher.stats()
    (cfg.root / "teacher_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log(f"farm wrote {written} runs; teacher: {stats}")
    return {"runs": written, **stats}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO / "configs" / "legal.json"))
    parser.add_argument("--model", default=None, help="teacher model id")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--tasks", type=int, default=None, help="cap tasks (smoke test)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--native-template", action="store_true",
                        help="teacher uses its own chat template instead of ours (ablation)")
    parser.add_argument("--eval-seeds", type=int, default=2)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    clone_repo()
    install_deps()

    import torch

    torch.set_num_threads(int(os.environ.get("REFINERY_THREADS", "4")))
    probe_gpu()

    from evalkit.contenders import evaluate
    from evalkit.report import write_all
    from refinery import config as config_mod
    from refinery.compiler.dataset import build_dataset
    from refinery.taskpool.build import iter_tasks
    from refinery.taskpool.legal import build_legal_manifest
    from refinery.verifier.gate import verify_manifest

    cfg = config_mod.load(args.config)
    if args.epochs:
        cfg = config_mod.Config(
            name=cfg.name, root=cfg.root, taskpool=cfg.taskpool, farm=cfg.farm,
            compiler=cfg.compiler, evalkit=cfg.evalkit, raw=cfg.raw,
            trainer=type(cfg.trainer)(
                arch=cfg.trainer.arch,
                optim=type(cfg.trainer.optim)(**{**cfg.trainer.optim.__dict__, "epochs": args.epochs}),
                stage1_fraction=cfg.trainer.stage1_fraction,
                stage1_max_steps=cfg.trainer.stage1_max_steps,
                stage1_epochs=cfg.trainer.stage1_epochs,
                student_init=cfg.trainer.student_init,
            ),
        )
    cfg.ensure_dirs()

    log("=== 1/6 taskpool ===")
    log(json.dumps(build_legal_manifest(cfg)))

    log("=== 2/6 farm ===")
    teacher = build_teacher(args.model, args.native_template)
    farm_stats = run_farm(cfg, teacher, args.tasks, args.k)

    log("=== 3/6 verify ===")
    verdict = verify_manifest(cfg)
    log(f"verdicts: {verdict['histogram']['counts']}")

    log("=== 4/6 compile ===")
    summary = build_dataset(cfg, strategies=["windowed", "full_context"])
    log(f"dataset: {summary['strategies']['windowed']['n_train']} train samples")

    summaries = []
    if not args.skip_train:
        from refinery.trainer.train import train

        for name, curriculum in (("student", True), ("student_no_curriculum", False)):
            log(f"=== 5/6 train {name} ===")
            result = train(cfg, strategy="windowed", curriculum=curriculum, out_name=name)
            log(json.dumps(result.stats))

    log("=== 6/6 eval ===")
    for contender in ("student", "student_no_curriculum", "heuristic"):
        tasks = list(iter_tasks(cfg, split="eval"))
        s = evaluate(contender, tasks, cfg, seeds=args.eval_seeds,
                     out_path=cfg.eval_dir / "results.jsonl")
        summaries.append(s)
        log(f"{contender}: pass@1={s['pass@1']} tool_calls={s['valid_tool_call_rate']} "
            f"retrieval={s['retrieval_success_rate']}")
    (cfg.eval_dir / "summaries.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    log("=== report ===")
    reports = Path("/kaggle/working/reports")
    written = write_all(cfg, summaries=summaries, out_dir=reports)
    log(f"reports: {written}")

    (REPO / "kaggle" / "run_summary.json").write_text(
        json.dumps({"farm": farm_stats, "verdicts": verdict["histogram"],
                    "dataset": summary, "summaries": summaries,
                    "gpu": probe_gpu()}, indent=2),
        encoding="utf-8",
    )
    log("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
