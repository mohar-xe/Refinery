"""Command-line entrypoint. One subcommand per pipeline stage.

Stage order is the dependency order, and nothing is hidden:

    taskpool -> farm -> verify -> compile -> train -> eval -> report

`run_toy.sh` calls these in sequence. Running them by hand is supported and
expected — every stage is resumable, so re-running a stage is cheap.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from refinery import config as config_mod

__all__ = ["main"]


def _cfg(args: argparse.Namespace):
    cfg = config_mod.load(args.config)
    cfg.ensure_dirs()
    return cfg


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


# --- stages -----------------------------------------------------------------
def cmd_taskpool(args) -> int:
    from refinery.taskpool.build import build_manifest

    cfg = _cfg(args)
    _print(build_manifest(cfg))
    return 0


def cmd_farm(args) -> int:
    from refinery.farm.harness import run_all
    from refinery.farm.teacher import build_teacher
    from refinery.taskpool.build import iter_tasks

    cfg = _cfg(args)
    tasks = list(iter_tasks(cfg, split=args.split))
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        print("no tasks in manifest — run `taskpool` first", file=sys.stderr)
        return 1

    teacher = build_teacher(cfg.teacher, prefer=args.teacher, seed=args.seed)
    _print(run_all(teacher, tasks, cfg, teacher_cfg=cfg.teacher, shard=args.shard))
    return 0


def cmd_verify(args) -> int:
    from refinery.verifier.gate import clear_verdicts, verify_manifest

    cfg = _cfg(args)
    if args.fresh:
        clear_verdicts(cfg.verdicts_path)
    _print(verify_manifest(cfg))
    return 0


def cmd_compile(args) -> int:
    from refinery.compiler.dataset import build_dataset

    cfg = _cfg(args)
    strategies = [s for s in (args.strategies or "").split(",") if s] or None
    _print(build_dataset(cfg, strategies=strategies))
    return 0


def cmd_train(args) -> int:
    from refinery.trainer.train import train

    cfg = _cfg(args)
    result = train(
        cfg,
        strategy=args.strategy,
        curriculum=not args.no_curriculum,
        out_name=args.name,
    )
    _print(
        {
            **result.stats,
            "adapter": str(result.adapter_path),
            "tokenizer": str(result.tokenizer_path),
            "log": str(result.log_path),
        }
    )
    return 0


def cmd_eval(args) -> int:
    from evalkit.contenders import evaluate
    from refinery.taskpool.build import iter_tasks

    cfg = _cfg(args)
    tasks = list(iter_tasks(cfg, split=args.split))
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        print("no eval tasks — run `taskpool` first", file=sys.stderr)
        return 1

    summaries = []
    for contender in [c for c in args.contenders.split(",") if c]:
        summary = evaluate(
            contender,
            tasks,
            cfg,
            seeds=args.seeds,
            out_path=cfg.eval_dir / "results.jsonl",
            context_override=args.context,
        )
        summaries.append(summary)
        print(f"[eval] {contender}: pass@1={summary['pass@1']} "
              f"tool_calls={summary['valid_tool_call_rate']}", file=sys.stderr)

    out = cfg.eval_dir / "summaries.json"
    out.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    _print({"summaries": str(out), "contenders": [s["contender"] for s in summaries]})
    return 0


def cmd_report(args) -> int:
    from evalkit.report import write_all

    cfg = _cfg(args)
    summaries = []
    summaries_path = cfg.eval_dir / "summaries.json"
    if summaries_path.exists():
        summaries = json.loads(summaries_path.read_text(encoding="utf-8"))
    _print(write_all(cfg, summaries=summaries))
    return 0


def cmd_info(args) -> int:
    cfg = _cfg(args)
    arch = cfg.trainer.arch
    _print(
        {
            "config": cfg.name,
            "root": str(cfg.root),
            "n_tasks": cfg.taskpool.get("n_tasks"),
            "samples_per_task": cfg.samples_per_task,
            "caps": vars(cfg.caps),
            "window": vars(cfg.window),
            "student_non_embedding_params": arch.non_embedding_params(),
            "student_total_params_incl_embeddings": None,
            "teacher": vars(cfg.teacher),
        }
    )
    return 0


# --- parser -----------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="refinery", description=__doc__)
    parser.add_argument("--config", default="configs/toy.json", help="scale config JSON")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("taskpool", help="build the task manifest + hash-walled split").set_defaults(
        func=cmd_taskpool
    )

    p_farm = sub.add_parser("farm", help="run k samples/task against the teacher")
    p_farm.add_argument("--teacher", default="api", choices=["api", "heuristic"])
    p_farm.add_argument("--split", default="train", choices=["train", "eval", "all"])
    p_farm.add_argument("--limit", type=int, default=None)
    p_farm.add_argument("--shard", type=int, default=0)
    p_farm.add_argument("--seed", type=int, default=0)
    p_farm.set_defaults(func=cmd_farm)

    p_verify = sub.add_parser("verify", help="independent verification + anti-hack filters")
    p_verify.add_argument("--fresh", action="store_true", help="discard existing verdicts first")
    p_verify.set_defaults(func=cmd_verify)

    p_compile = sub.add_parser("compile", help="verified trajectories -> SFT dataset")
    p_compile.add_argument("--strategies", default=None, help="comma list, e.g. windowed,naive_truncated")
    p_compile.set_defaults(func=cmd_compile)

    p_train = sub.add_parser("train", help="train the student")
    p_train.add_argument("--strategy", default=None)
    p_train.add_argument("--name", default=None, help="adapter subdirectory / contender id")
    p_train.add_argument("--no-curriculum", action="store_true", help="ablation arm")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="run contenders through the identical harness")
    p_eval.add_argument("--contenders", default="student,heuristic")
    p_eval.add_argument("--seeds", type=int, default=3)
    p_eval.add_argument("--split", default="eval", choices=["train", "eval"])
    p_eval.add_argument("--limit", type=int, default=None)
    p_eval.add_argument(
        "--context",
        type=int,
        default=None,
        help="override the student's context window (stress test for the windowing policy)",
    )
    p_eval.set_defaults(func=cmd_eval)

    sub.add_parser("report", help="write reports/ from generated artifacts").set_defaults(
        func=cmd_report
    )
    sub.add_parser("info", help="print the resolved config").set_defaults(func=cmd_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
