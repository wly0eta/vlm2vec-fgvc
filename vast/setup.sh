#!/bin/bash
# One-time (and safe-to-rerun) setup ON THE VAST NODE, after vast/push.sh has copied files:
#
#   bash /workspace/vlm2vec/vast/setup.sh
#
# 1. reports GPUs / disk, 2. installs Python deps (keeps the image's torch if it already works
# with CUDA), 3. rewrites the Unity paths baked into checkpoints and datasets to this node's
# layout, 4. runs the unit tests.
#   CUDA_TAG=cu121 bash vast/setup.sh     # force a torch build (default: keep existing, else cu124)
#   SKIP_TESTS=1 bash vast/setup.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
export PIP_NO_CACHE_DIR=1   # the pip cache would take GBs of a small instance disk

echo "== hardware"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || echo "no GPU visible!"
df -h "$WORKSPACE" | tail -1
echo "GPUs: $NGPUS"

echo "== python packages"
if $PYTHON -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "keeping preinstalled torch: $($PYTHON -c 'import torch; print(torch.__version__, "cuda", torch.version.cuda)')"
else
  $PYTHON -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG:-cu124}"
fi
# transformers pinned to the version every Unity result was produced with
$PYTHON -m pip install -r requirements.txt "transformers==${TRANSFORMERS_VERSION:-5.14.1}"
$PYTHON -m pip uninstall -y torchao torchaudio 2>/dev/null || true   # unused; avoids import clashes
# flash-attn: models then load with attention "auto" -> flash_attention_2 (else sdpa, same maths).
# Uses a prebuilt wheel when one matches torch/CUDA/python, otherwise compiles (can take ~1 h).
# FLASH_ATTN=0 skips it; a failure is not fatal.
if [ "${FLASH_ATTN:-1}" = 1 ] && ! $PYTHON -c "import flash_attn" 2>/dev/null; then
  # 2.7.4.post1: its prebuilt wheels match the torch 2.6 ABI (2.8.3 picked a wheel that fails to import)
  MAX_JOBS=${MAX_JOBS:-8} $PYTHON -m pip install "flash-attn==${FLASH_ATTN_VERSION:-2.7.4.post1}" --no-build-isolation \
    || echo "(flash-attn install failed; models will use sdpa)"
fi
$PYTHON -c "import flash_attn; print('flash-attn', flash_attn.__version__)" 2>/dev/null || echo "flash-attn: not installed (sdpa)"
command -v tmux >/dev/null || { apt-get update -qq && apt-get install -y -qq tmux; } || echo "(tmux not installed; use DETACH=0 or nohup)"

echo "== relocating Unity paths"
$PYTHON tools/relocate_paths.py --dir runs --dir configs \
  --map /home/obicakci_umass_edu/train_v2/runs="$VLM2VEC_ROOT/runs" \
  --map /home/obicakci_umass_edu/fgvc_sft_ckpt="$FGVC_CKPT" \
  --map /datasets/ai/inaturalist="$INAT_ROOT"

echo "== checks"
[ -d "$FGVC_CKPT" ] || echo "WARNING: $FGVC_CKPT missing -- mentor_* runs cannot load"
[ -d images/cub200 ] || echo "WARNING: images/ missing -- push data first (vast/push.sh)"
[ -d "$INAT_ROOT/val" ] || echo "note: no iNat images yet -- iNat eval/training needs: bash vast/get_inat.sh"
$PYTHON -c "import torch, transformers, peft; print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__, '| cuda', torch.cuda.is_available())"
if [ "${SKIP_TESTS:-0}" != 1 ]; then
  $PYTHON -m pytest -q tests
fi
echo "setup done. Next: bash vast/train.sh ... / bash vast/eval.sh ...  (see vast/README.md)"
