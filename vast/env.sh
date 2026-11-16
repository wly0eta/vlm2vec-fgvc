# Shared settings for every node-side script (sourced, not run). Replaces slurm/env.sh.
#
# Layout on the Vast node (everything under /workspace, the instance's persistent disk):
#   /workspace/vlm2vec/          this repo (code, experiments/, configs/*.yaml + *.jsonl, images/, runs/)
#   /workspace/fgvc_sft_ckpt/    Logan's FGVC-SFT adapter (base of every mentor_* run)
#   /workspace/data/inaturalist/ iNat-2021 images (val/, optionally train/), see vast/get_inat.sh
#   /workspace/hf_cache/         Hugging Face downloads (Qwen weights)
export WORKSPACE=${WORKSPACE:-/workspace}
export VLM2VEC_ROOT=${VLM2VEC_ROOT:-$WORKSPACE/vlm2vec}
export FGVC_CKPT=${FGVC_CKPT:-$WORKSPACE/fgvc_sft_ckpt}
export INAT_ROOT=${INAT_ROOT:-$WORKSPACE/data/inaturalist}

# Per-node overrides that are not code (e.g. HF_HOME on /dev/shm when the disk is small): one
# file on the node itself, never pushed
[ -f "$WORKSPACE/node.env" ] && source "$WORKSPACE/node.env"
export HF_HOME=${HF_HOME:-$WORKSPACE/hf_cache}
export HF_HUB_DISABLE_XET=1          # xet download threads crash at interpreter exit
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false  # we parallelize with dataloader workers instead
export PYTHONUNBUFFERED=1            # logs appear as they happen
export PYTHONIOENCODING=utf-8         # the node's locale is ASCII: printing a question with "°" crashed
export LANG=${LANG:-C.UTF-8} LC_ALL=${LC_ALL:-C.UTF-8}
# dataloader workers hand batches over as shared-memory tensors, one file descriptor each: the
# container's soft limit (1024) ran out at step 0 with batch 256 ("Too many open files")
ulimit -n "$(ulimit -Hn)" 2>/dev/null || ulimit -n 65536 2>/dev/null || true

# Vast images keep torch in /venv/main, activated only in interactive logins: put it on PATH so
# tmux sessions and `ssh vast <command>` use the same python/torchrun/pip
[ -d /venv/main/bin ] && case ":$PATH:" in *:/venv/main/bin:*) ;; *) export PATH=/venv/main/bin:$PATH ;; esac
PYTHON=${PYTHON:-python3}
# GPU numbers are nvidia-smi's (PCI order). GPU_IDS (node.env), e.g. "0 2 3", leaves out a broken
# card: whole-node jobs (torchrun) then see only those GPUs, and per-GPU jobs map their slot
# 0, 1, 2... to a real GPU with gpu_id. A caller's own CUDA_VISIBLE_DEVICES is never overridden.
# CUDA_VISIBLE_DEVICES gets UUIDs, not numbers: CUDA drops a failed card from its own numbering
# (GPU 1 here), so CUDA's "3" stopped existing and nvidia-smi's 2 became CUDA's 1.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
gpu_uuid() {   # gpu_uuid <nvidia-smi index | UUID> -> UUID
  case $1 in GPU-*) echo "$1" ;; *) nvidia-smi -i "$1" --query-gpu=uuid --format=csv,noheader 2>/dev/null || echo "$1" ;; esac
}
if [ -n "${GPU_IDS:-}" ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  CUDA_VISIBLE_DEVICES=$(for g in $GPU_IDS; do gpu_uuid "$g"; done | paste -sd, -)
  export CUDA_VISIBLE_DEVICES
fi
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  NGPUS=${NGPUS:-$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | grep -c .)}
else
  NGPUS=${NGPUS:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}
fi
gpu_id() {   # gpu_id <slot> -> the real GPU for per-GPU job number <slot>
  local ids
  read -ra ids <<< "${GPU_IDS:-$(seq -s ' ' 0 $(( $(nvidia-smi -L 2>/dev/null | wc -l) - 1 )))}"
  gpu_uuid "${ids[$(( $1 % ${#ids[@]} ))]}"
}

# the calling script, as an absolute path: scripts that relaunch themselves in tmux ("bash $SELF")
# keep working after the cd below, wherever they were started from
SELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
# a tmux server keeps the environment of whoever started it: a CUDA_VISIBLE_DEVICES left there
# (e.g. by an old 2-GPU job) silently overrode GPU_IDS in every later session. Launchers pass
# their own GPU choice explicitly, so drop the stale one.
[ -z "${TMUX:-}" ] && tmux set-environment -g -u CUDA_VISIBLE_DEVICES 2>/dev/null

cd "$VLM2VEC_ROOT"
mkdir -p runs logs
