#!/bin/bash
# The quick start, from a clone of the repository with the release
# downloaded into it: build this machine's environment and evaluate
# Prospector-4B on two questions per difficulty tier, printing the same table
# as the paper's runs.
#
#   ./quickstart.sh
#
# On a Mac with Apple silicon it builds .venv-mac and evaluates through the
# mlx: driver; it needs uv. On Linux with an NVIDIA GPU it runs
# launch/setup.sh, which needs sudo for CUDA's compiler, and evaluates through
# the vllm: driver. Every command is printed before it runs. Without the
# served adapters under models/ it stops before building anything. The first
# run fetches the base model from the Hub (about 9 GB) and merges the three
# adapters into one model (about 8 GB, inside the last adapter's directory
# under models/), which takes a few minutes; allow about 27 GB of disk in all.
set -euo pipefail
cd "$(dirname "$0")"

usage() {
  sed -n '2,16s/^# \{0,1\}//p' "$0"
}
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then usage; exit 0; fi

run() {
  printf '+ %s\n' "$*"
  "$@"
}

STACK=models/traj_sft_4b_flash20g2/final,models/traj_grpo_4b_flash20d1/checkpoint-120,models/traj_grpo_4b_flash20d2/checkpoint-120

for adapter in ${STACK//,/ }; do
  if [ ! -f "$adapter/adapter_config.json" ]; then
    echo "quickstart.sh needs the release: $adapter is missing. Download it first:" >&2
    echo '  uvx --from huggingface_hub hf download lamm-mit/Prospector --local-dir . --include "models/*" --include "data/*"' >&2
    exit 2
  fi
done

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64)
    VENV=.venv-mac
    DRIVER=mlx
    if ! command -v uv >/dev/null 2>&1; then
      echo "quickstart.sh needs uv: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
      exit 2
    fi
    [ -x "$VENV/bin/python" ] || run uv venv --python 3.12 "$VENV"
    run uv pip install --python "$VENV/bin/python" -e ".[dev,train,mlx,app]" -c launch/constraints-mac.txt
    ;;
  Linux-*)
    if ! command -v nvidia-smi >/dev/null 2>&1; then
      echo "quickstart.sh runs on a Mac with Apple silicon or on Linux with an NVIDIA GPU" >&2
      exit 2
    fi
    VENV=.venv
    DRIVER=vllm
    run launch/setup.sh
    # What every job on the GPU machine runs with (launch/README.md, section 4).
    export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONUNBUFFERED=1
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    ;;
  *)
    echo "quickstart.sh runs on a Mac with Apple silicon or on Linux with an NVIDIA GPU" >&2
    exit 2
    ;;
esac

# The activate script reads variables it may not set, so not under set -u.
set +u
# shellcheck disable=SC1090
source "$VENV/bin/activate"
set -u

run python scripts/run_agent_eval.py --graph tsmc --model "$DRIVER:Qwen/Qwen3.5-4B" --adapter "$STACK" \
  --turn-max-tokens 1024 --limit 2 --no-latency --name quickstart

echo
echo "Done. To run more commands, activate the environment first:"
echo "  source $VENV/bin/activate"
echo "The app then serves the same model: python -m uvicorn app.server:app --port 8321"
