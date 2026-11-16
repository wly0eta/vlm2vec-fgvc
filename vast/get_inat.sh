#!/bin/bash
# Download iNaturalist-2021 images ON THE VAST NODE (on Unity they lived on /datasets).
#
#   bash vast/get_inat.sh                  # val only: ~9 GB, enough for every iNat eval
#   bash vast/get_inat.sh train_mini       # + train_mini: 500K images (50/species), ~45 GB
#   bash vast/get_inat.sh train            # + full train: 2.7M images, ~250 GB (needs a big disk)
#
# Archives are streamed straight into tar (never stored), so a split needs only its extracted
# size in free disk -- but an interrupted download restarts that split from scratch.
# Extracts to $INAT_ROOT/<split>/<class folder>/<image>.jpg, the layout the iNat JSONLs point at
# after vast/setup.sh relocates /datasets/ai/inaturalist to $INAT_ROOT. For a train split it also
# fetches <split>.json and builds configs/inat_<split>.jsonl (train_mini: see configs/inat_mini.yaml).
set -euo pipefail
source "$(dirname "$0")/env.sh"
URL=https://ml-inat-competition-datasets.s3.amazonaws.com/2021
mkdir -p "$INAT_ROOT"

for split in val "$@"; do
  if [ -f "$INAT_ROOT/$split/.complete" ]; then
    echo "$INAT_ROOT/$split already present"
  else
    echo "== downloading + extracting $split ($(df -h "$INAT_ROOT" | tail -1 | awk '{print $4}') free)"
    rm -rf "${INAT_ROOT:?}/$split"
    wget -q --show-progress -O - "$URL/$split.tar.gz" | tar xzf - -C "$INAT_ROOT"
    touch "$INAT_ROOT/$split/.complete"
  fi
  [ "$split" = val ] && continue
  [ -f "$INAT_ROOT/$split.json" ] || wget -q -O - "$URL/$split.json.tar.gz" | tar xzf - -C "$INAT_ROOT"
  OUT=configs/inat_$split.jsonl
  if [ ! -f "$OUT" ]; then
    echo "== building $OUT"
    $PYTHON build_dataset.py inat-local --root "$INAT_ROOT" --annotations "$split.json" --out "$OUT"
  fi
done
echo "iNat images in $INAT_ROOT:"; ls "$INAT_ROOT"; df -h "$INAT_ROOT" | tail -1
