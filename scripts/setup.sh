#!/usr/bin/env bash
# Provision the toolchain. Idempotent.
#
# Why uv and a 3.12 venv instead of system pip (LLD.md D-014): this box's system
# Python is 3.14, and there are no aarch64 cp314 wheels for torch — pip would
# fall back to a source build and hang. uv fetches a CPython 3.12 for which the
# CPU torch wheel exists. Verified working on aarch64/proot.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "==> installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d .venv ]; then
  echo "==> creating .venv (CPython 3.12)"
  uv venv --python 3.12 .venv
fi

echo "==> installing dependencies"
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python httpx pyarrow numpy pytest ruff

echo "==> versions"
.venv/bin/python - <<'PY'
import sys, torch, pyarrow, httpx
print("python  ", sys.version.split()[0])
print("torch   ", torch.__version__)
print("pyarrow ", pyarrow.__version__)
print("httpx   ", httpx.__version__)
PY

echo "==> ok. Next: export OPENROUTER_API_KEY=… && ./scripts/run_toy.sh"
