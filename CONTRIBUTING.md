# Contributing

Solo project, but the conventions are real because the repo doubles as a portfolio artifact:
a reviewer should be able to read the history and see *why* the system looks the way it does.

## Branches

- `main` — always runnable; the reports in `README.md` correspond to what is on `main`.
- `feat/<slug>` — one feature or one pipeline stage at a time.
- `exp/<slug>` — anything whose result is an ablation arm (these may not be "clean").

## Commits — Conventional Commits

```
feat(verifier): add unsupported-answer anti-hack check
fix(compiler): keep tool arguments verbatim when windowing
docs(lld): record D-015a parameter-budget correction
exp(trainer): mixed-from-scratch curriculum ablation arm
```

Scopes in use: `taskpool`, `farm`, `verifier`, `compiler`, `trainer`, `evalkit`, `docs`, `infra`.

## Rules that are not stylistic preferences

1. **No results in a commit message that are not in `reports/`.** Every number in the README comes
   from a generated file. If you cannot regenerate it, you may not claim it.
2. **A new failure mode gets a new reason code and a `SCHEMA_VERSION` bump.** Never reuse or
   repurpose an existing code in `refinery/verifier/codes.py`; the histogram is a partition.
3. **Decisions get logged in `LLD.md`** before the code that depends on them lands, including the
   options rejected. Reversing a decision is a new entry, and the old one is marked `SUPERSEDED`.
4. **The renderer is single-owner.** `compiler/render.py` is the only place that formats a prompt
   or a message. If you need a second format, you need a second decision entry.
5. **No secrets, ever.** The teacher key comes from `$OPENROUTER_API_KEY`. `.env` is ignored.
   This repo is public.
6. **No torch outside `refinery/trainer/` and `evalkit/`.** Stages 1–4 must run on a bare
   interpreter so the pipeline can be inspected and debugged cheaply.

## Before you push

```bash
.venv/bin/ruff check .        # lint
.venv/bin/pytest              # includes the byte-for-byte format-parity test
./scripts/run_toy.sh          # regenerates reports/ (the only source of README numbers)
```

## Reporting a bug

Open an issue with the failing stage, the `run_id`, and the reason code. If it reached the
verifier, the reason code is already in `runs/<run_id>/verdicts.jsonl` — quote it rather than
describing the symptom.
