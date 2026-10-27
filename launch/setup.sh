#!/bin/bash
# Set up a fresh GPU machine (Ubuntu, a CUDA card of 80-96 GB): launch/README.md, section 3
# as one idempotent script. Run it on that machine from a clone of the
# repository, in the background:
#   mkdir -p logs && (nohup launch/setup.sh > logs/bootstrap.log 2>&1 < /dev/null &)
# Anywhere but Linux with an NVIDIA card it prints this usage and stops.
# Steps:
#   1. uv                                  ~/.local/bin/uv
#   2. .venv (the trainers, vLLM, the app, the tests) uv pip install -e ".[dev,train,rl,app]" against
#      launch/constraints-gpu.txt. The kernels compile against torch's ABI, so the pins
#      come before step 4.
#   3. CUDA 13.0 nvcc from NVIDIA's apt repo for this machine's architecture (sbsa on aarch64,
#      x86_64 otherwise): torch is built against 13.0, and causal-conv1d compiles against
#      the same version
#   4. flash-linear-attention (a wheel) + causal-conv1d; without both, transformers
#      silently falls back to a much slower torch implementation. No causal-conv1d wheel
#      exists for the pinned torch and CUDA on either architecture, so it is built from
#      source (about eight minutes; the package hard-codes its nine target architectures
#      and its nvcc thread count, so no variable shortens it). A wheel built on an
#      identical stack and kept under launch/wheels/ (launch/README.md, section 8) is installed
#      instead.
#   5. an import check: the stack, vLLM and the kernels import; torch sees the card
#   6. the tests' tokenizer into the HF cache (lifts the suite's tokenizer skips)
cd "$(dirname "$0")/.." || exit 1
if [ "$(uname -s)" != Linux ] || ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "usage (on a Linux GPU machine, from a clone of the repository): mkdir -p logs && (nohup launch/setup.sh > logs/bootstrap.log 2>&1 < /dev/null &)"; exit 2
fi
mkdir -p logs
step() { echo "=== BOOTSTRAP $1 $(date '+%m-%d %H:%M:%S')"; }
export PATH="$HOME/.local/bin:$PATH"
PINS=(-c launch/constraints-gpu.txt)
PY=.venv/bin/python
case "$(uname -m)" in aarch64) CUDA_ARCH=sbsa ;; *) CUDA_ARCH=x86_64 ;; esac

step "1 uv"
[ -x "$HOME/.local/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version

step "2 .venv"
[ -x $PY ] || uv venv --python 3.12 .venv
uv pip install --python $PY -e ".[dev,train,rl,app]" "${PINS[@]}" 2>&1 | tail -2

step "3 cuda 13.0 nvcc"
if [ ! -x /usr/local/cuda-13.0/bin/nvcc ]; then
  curl -fsSL -o /tmp/cuda-keyring.deb "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/$CUDA_ARCH/cuda-keyring_1.1-1_all.deb"
  sudo dpkg -i /tmp/cuda-keyring.deb && sudo apt-get update -qq && sudo apt-get install -y -qq cuda-nvcc-13-0 cuda-cudart-dev-13-0
fi
/usr/local/cuda-13.0/bin/nvcc --version | grep release

step "4 fast kernels"
# causal-conv1d compiles against torch's C++ ABI. A build made against a different torch
# is installed but fails to import (an undefined c10::cuda symbol), and uv would
# reinstall the same cached wheel. So a broken causal_conv1d is uninstalled, its cache
# cleaned, and the package rebuilt against the venv's torch. The build is from source
# on both architectures, since no wheel matches the pins; aarch64 says --no-binary
# explicitly because PyPI carries aarch64 wheels for other torch versions that uv would
# otherwise consider.
kernels() {
  if $PY -c "import causal_conv1d" 2>/dev/null; then return; fi
  if $PY -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('causal_conv1d') else 1)" 2>/dev/null; then
    echo "causal_conv1d present but broken (ABI) - rebuilding from source"
    uv pip uninstall --python $PY causal-conv1d 2>&1 | tail -1; uv cache clean causal-conv1d 2>&1 | tail -1
  fi
  # --no-build-isolation is required: uv otherwise builds the sdist in a sandbox that
  # installs the sdist's own build requirements, which name torch unpinned, so the
  # extension compiles against the newest torch on PyPI whatever the venv holds. The
  # venv therefore needs the build dependencies itself.
  uv pip install --python $PY setuptools wheel ninja packaging 2>&1 | tail -1
  CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH \
    uv pip install --python $PY "${PINS[@]}" flash-linear-attention 2>&1 | tail -1
  # A wheel kept from a previous machine with the same torch, CUDA, Python and
  # architecture installs in a second; one built against another torch fails the
  # import and is replaced by a build.
  KEPT=$(ls launch/wheels/causal_conv1d-*-cp312-cp312-linux_$(uname -m).whl 2>/dev/null | head -1)
  if [ -n "$KEPT" ]; then
    uv pip install --python $PY "${PINS[@]}" "$KEPT" 2>&1 | tail -1
    if $PY -c "import causal_conv1d" 2>/dev/null; then
      echo "causal-conv1d from the kept wheel $KEPT"; return
    fi
    echo "the kept wheel does not import against this torch - building from source"
    uv pip uninstall --python $PY causal-conv1d 2>&1 | tail -1
  fi
  NOBIN=(); [ "$CUDA_ARCH" = sbsa ] && NOBIN=(--no-binary causal-conv1d)
  CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH \
    uv pip install --python $PY "${PINS[@]}" --no-build-isolation "${NOBIN[@]}" causal-conv1d 2>&1 | tail -2
}
kernels

step "5 proof"
$PY - <<'PY' 2>&1 | grep -vE "PyTorch was not found|outdated"
import importlib, torch
mods = {m: getattr(importlib.import_module(m), "__version__", "ok") for m in ("torch", "transformers", "trl", "peft", "vllm", "fla", "causal_conv1d")}
print(mods, "cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
PY
# The pipe's status is grep's; the interpreter's own is the one that counts.
[ "${PIPESTATUS[0]}" = 0 ] || { echo "!!! BOOTSTRAP INCOMPLETE: .venv does not import the stack (see the traceback above)"; exit 1; }

step "6 test tokenizer"
# The suite's tokenizer tests run on Qwen/Qwen3.5-0.8B's tokenizer (tests/conftest.py
# shipped_tokenizer) and skip until it is in the HF cache, where prospector looks too.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
$PY -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B'); print('test tokenizer cached')" 2>&1 | tail -1
echo "=== BOOTSTRAP COMPLETE $(date '+%m-%d %H:%M:%S')"
