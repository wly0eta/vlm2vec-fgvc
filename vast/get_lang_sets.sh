#!/bin/bash
# English-vs-Latin eval pairs (tools/make_lang_sets.py), ON THE NODE:   bash vast/get_lang_sets.sh
# Needs iNat-2021 val: its images (~9 GB, vast/get_inat.sh) and val.json, the category list with
# English common names. Waits for configs/zs_flowers.jsonl from vast/get_zeroshot.sh.
set -euo pipefail
source "$(dirname "$0")/env.sh"
bash vast/get_inat.sh                     # val images only
[ -f "$INAT_ROOT/val.json" ] || wget -q -O - \
  https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz | tar xzf - -C "$INAT_ROOT"
until [ -s configs/zs_flowers.jsonl ]; do
  pgrep -f "vast/get_zerosho[t].sh" > /dev/null || { echo "no configs/zs_flowers.jsonl: run vast/get_zeroshot.sh"; exit 1; }
  sleep 30
done
$PYTHON tools/make_lang_sets.py --inat-annotations "$INAT_ROOT/val.json"
df -h / | tail -1
