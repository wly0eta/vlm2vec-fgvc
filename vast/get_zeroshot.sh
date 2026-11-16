#!/bin/bash
# Build the zero-shot FGVC benchmark suite ON THE NODE (test splits only; nothing here is ever
# trained on):   bash vast/get_zeroshot.sh
#
#   configs/zs_cars.jsonl       Stanford Cars        196 models     8,041  (tanganke/stanford_cars)
#   configs/zs_aircraft.jsonl   FGVC-Aircraft        100 variants   3,333  (Multimodal-Fatima/FGVC_Aircraft_test)
#   configs/zs_food.jsonl       Food-101             101 dishes    10,100  (ethz/food101, 100 per class of 250)
#   configs/zs_pets.jsonl       Oxford-IIIT Pets      37 breeds     3,669  (timm/oxford-iiit-pet)
#   configs/zs_dogs.jsonl       Stanford Dogs        120 breeds     6,174  (amaye15/stanford-dogs)
#   configs/zs_flowers.jsonl    Oxford Flowers-102   102 species    6,149  (dpdl-benchmark/oxford_flowers102)
#   configs/cub200_val_unseen20.jsonl, cub200_val_latin_unseen20.jsonl
#       CUB test images of the 20 species NOT among iNat-2021's 10,000 classes ("absent" in
#       configs/cub200_latin_aliases.tsv): class-disjoint from every iNat-trained model. Score them
#       against the full 200-name pool (vast/run_zeroshot.sh does) so they are not easier.
# Labels are the datasets' standard class names (what CLIP-style zero-shot papers use), with a
# task instruction per domain. Overlap with iNat: none for cars / aircraft / food / pets / dogs
# (iNat has wild species, not breeds or products); flowers and CUB share domains with it.
set -euo pipefail
source "$(dirname "$0")/env.sh"

build() {   # build <name> <dataset> <split> <instruction> [extra build_dataset.py args]
  local name=$1 ds=$2 split=$3 instruction=$4; shift 4
  [ -s "configs/zs_$name.jsonl" ] && { echo "configs/zs_$name.jsonl exists"; return; }
  echo "== $name <- $ds ($split)"
  $PYTHON build_dataset.py classification --dataset "$ds" --split "$split" --instruction "$instruction" \
    --out "configs/zs_$name.jsonl" --image-dir "images/zs_$name" "$@"
}
build cars     tanganke/stanford_cars               test       "Identify the car model in the image."
build aircraft Multimodal-Fatima/FGVC_Aircraft_test test       "Identify the aircraft model in the image." \
  --keep-raw-labels                                  # variant names like 737-300: no prefix stripping
build food     ethz/food101                         validation "Identify the food dish in the image." \
  --per-class-cap 100
build pets     timm/oxford-iiit-pet                 test       "Identify the cat or dog breed in the image."
build dogs     amaye15/stanford-dogs                test       "Identify the dog breed in the image." \
  --image-field pixel_values
build flowers  dpdl-benchmark/oxford_flowers102     test       "Identify the flower species in the image."

$PYTHON - <<'EOF'
import json
absent = {l.split("\t")[0] for l in open("configs/cub200_latin_aliases.tsv")
          if not l.startswith("#") and l.rstrip("\n").split("\t")[2:3] == ["absent"]}
en = [json.loads(l) for l in open("configs/cub200_val.jsonl")]
la = [json.loads(l) for l in open("configs/cub200_val_latin.jsonl")]
keep = [i for i, r in enumerate(en) if r["positive"]["text"] in absent]
assert len(absent) == 20 and all(en[i]["query"]["image"] == la[i]["query"]["image"] for i in keep)
for rows, out in ((en, "configs/cub200_val_unseen20.jsonl"), (la, "configs/cub200_val_latin_unseen20.jsonl")):
    with open(out, "w") as f:
        f.writelines(json.dumps(rows[i], ensure_ascii=False) + "\n" for i in keep)
print(f"{len(keep)} CUB test images of the 20 species absent from iNat-2021 -> configs/cub200_val*_unseen20.jsonl")
EOF
for f in configs/zs_*.jsonl; do
  echo "$f: $(wc -l < "$f") rows, $($PYTHON -c "import json,sys; print(len({json.loads(l)['positive']['text'] for l in open(sys.argv[1])}))" "$f") classes"
done
du -sh images/zs_* | tr '\n' ' '; echo; df -h / | tail -1
