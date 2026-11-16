#!/bin/bash
# Copy results and new checkpoints FROM the Vast node back to this Mac. Safe to rerun; do it
# before destroying an instance (its disk is deleted with it).
#
#   bash vast/pull.sh                     # results, logs, plots, adapters of every run
#   FULL=1 bash vast/pull.sh              # + every checkpoint-N incl. optimizer state (large)
#
# Lands in $DEST/<run>/ (default ~/Downloads/vlm2vec_checkpoints, next to the Unity backup).
# Pulled vlm2vec_config.json files keep the node's /workspace paths; vast/setup.sh (or
# tools/relocate_paths.py) rewrites them wherever they are used next.
set -euo pipefail
HOST=${HOST:-vast}          # the Host alias in ~/.ssh/config (address, port, key live there)
PORT=${PORT:-}
KEY=${KEY:-}
WORKSPACE=${WORKSPACE:-/workspace}
DEST=${DEST:-$HOME/Downloads/vlm2vec_checkpoints}
SSH="ssh ${PORT:+-p $PORT} ${KEY:+-i $KEY} -o ServerAliveInterval=30"

if [ "${FULL:-0}" = 1 ]; then
  rsync -a --partial -e "$SSH" "$HOST:$WORKSPACE/vlm2vec/runs/" "$DEST/"
else
  # everything except optimizer / scheduler / RNG state (what you need to evaluate or build on)
  rsync -a --partial -e "$SSH" --exclude=optimizer.pt --exclude=scheduler.pt --exclude='rng_state*.pth' \
    "$HOST:$WORKSPACE/vlm2vec/runs/" "$DEST/"
fi
echo "pulled to $DEST"
python3 "$(dirname "$0")/../tools/collect_results.py" $(ls -d "$DEST"/*/ "$DEST"/*/checkpoint-*/ 2>/dev/null | grep -vE '/(data|unity_code|analysis|fgvc_sft_ckpt)/') 2>/dev/null | sed "s#$DEST/##g" || true
# results database of everything pulled (tools/results_db.py): query it with
#   python tools/results_db.py --runs-dir "$DEST" table --experiment 07
python3 "$(dirname "$0")/../tools/results_db.py" --runs-dir "$DEST" build || true
