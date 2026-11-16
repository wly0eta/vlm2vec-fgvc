#!/bin/bash
# Copy code, data and checkpoints FROM THIS MAC to the Vast node. Safe to rerun.
#
#   bash vast/push.sh                       # code + data + Logan's adapter + all runs
#   WHAT=code bash vast/push.sh             # only the code (after editing it locally)
#   RUNS="q2b_mixed q2b_mixed__cub_full" bash vast/push.sh     # only these runs
#   PROGRESS=1 bash vast/push.sh            # show each file as it transfers
#   SLIM=1 bash vast/push.sh                # small disk: no iNat images / iNat train JSONLs, and
#                                           # runs + Logan's adapter without optimizer state or
#                                           # intermediate checkpoint-N dirs (enough for CUB / evals)
#
# Connection: the `vast` alias in ~/.ssh/config -- update it there when you rent a new node
# (or HOST=root@1.2.3.4 PORT=12345 KEY=~/.ssh/unity bash vast/push.sh).
# Source of checkpoints/data: BACKUP=~/Downloads/vlm2vec_checkpoints (made by backup_from_unity.sh)
#
# Code is mirrored (local edits overwrite the node). Data and runs are copied with
# --ignore-existing: files already on the node are never overwritten, so relocated paths and
# checkpoints written by training on the node are safe. Interrupted files wait in .rsync-partial/
# (not under their real name), so a rerun finishes them instead of skipping a truncated copy.
#   REPAIR=1 bash vast/push.sh      # re-copy any node file whose size differs from the Mac's
#                                   # (fixes truncated copies; do NOT use after setup.sh has run,
#                                   #  it would undo the path relocation)
#
# The whole script is one { ... } block, so bash reads it completely before running:
# editing this file while a push is running can't corrupt the running push.
{
set -euo pipefail
cd "$(dirname "$0")/.."
HOST=${HOST:-vast}          # the Host alias in ~/.ssh/config (address, port, key live there)
PORT=${PORT:-}
KEY=${KEY:-}
BACKUP=${BACKUP:-$HOME/Downloads/vlm2vec_checkpoints}
WORKSPACE=${WORKSPACE:-/workspace}
WHAT=${WHAT:-all}
SSH="ssh ${PORT:+-p $PORT} ${KEY:+-i $KEY} -o ServerAliveInterval=30"
# Every connection prints the node's login banner (Vast's welcome text): that is normal, and
# each banner means the login succeeded. rsync itself is silent unless PROGRESS=1.
P=(--partial-dir=.rsync-partial); [ "${PROGRESS:-0}" = 1 ] && P+=(--progress)
MODE=(--ignore-existing); [ "${REPAIR:-0}" = 1 ] && MODE=(--size-only)
SLIM_DATA=() SLIM_CKPT=()
if [ "${SLIM:-0}" = 1 ]; then
  SLIM_DATA=(--exclude='inaturalist/' --exclude='inat_train*.jsonl' --exclude='inaturalist_train.jsonl'
             --exclude='inat_aves_train.jsonl')
  SLIM_CKPT=(--exclude='checkpoint-*/' --exclude=optimizer.pt --exclude=scheduler.pt --exclude='rng_state*.pth')
fi
REMOTE=$WORKSPACE/vlm2vec

$SSH "$HOST" "mkdir -p $REMOTE/runs $REMOTE/configs $REMOTE/images $WORKSPACE/fgvc_sft_ckpt"

echo "== code -> $HOST:$REMOTE"
rsync -a "${P[@]}" -e "$SSH" --exclude='__pycache__/' --exclude='.DS_Store' --exclude='.git/' --exclude='runs/' \
  --exclude='images/' --exclude='*.jsonl' --exclude='*.offsets.npz' ./ "$HOST:$REMOTE/"
[ "$WHAT" = code ] && { echo "code pushed"; exit 0; }

echo "== datasets (JSONL) and images"
rsync -a "${P[@]}" "${MODE[@]}" -e "$SSH" --exclude='*.offsets.npz' ${SLIM_DATA[@]+"${SLIM_DATA[@]}"} "$BACKUP/data/configs/" "$HOST:$REMOTE/configs/"
rsync -a "${P[@]}" "${MODE[@]}" -e "$SSH" ${SLIM_DATA[@]+"${SLIM_DATA[@]}"} "$BACKUP/data/images/" "$HOST:$REMOTE/images/"

echo "== Logan's FGVC-SFT adapter (base of every mentor_* run)"
rsync -a "${P[@]}" "${MODE[@]}" -e "$SSH" ${SLIM_CKPT[@]+"${SLIM_CKPT[@]}"} "$BACKUP/fgvc_sft_ckpt/" "$HOST:$WORKSPACE/fgvc_sft_ckpt/"

echo "== runs"
if [ -z "${RUNS:-}" ]; then
  RUNS=$(cd "$BACKUP" && ls -d */ | tr -d / | grep -vE '^(data|unity_code|analysis|fgvc_sft_ckpt)$|__checkpoint-')
fi
for run in $RUNS; do
  echo "   $run ($(du -sh "$BACKUP/$run" | cut -f1))"
  rsync -a "${P[@]}" "${MODE[@]}" -e "$SSH" ${SLIM_CKPT[@]+"${SLIM_CKPT[@]}"} "$BACKUP/$run/" "$HOST:$REMOTE/runs/$run/"
done
echo "pushed. On the node:  bash $REMOTE/vast/setup.sh"
exit 0
}
