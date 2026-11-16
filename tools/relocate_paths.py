"""Rewrite the absolute Unity paths baked into checkpoints and datasets after moving machines.

Two kinds of files remember where things lived:
  * vlm2vec_config.json -> "base_model" (e.g. /home/obicakci_umass_edu/train_v2/runs/q2b_mixed,
    /home/obicakci_umass_edu/fgvc_sft_ckpt): needed to load stacked adapters and mentor runs.
  * dataset JSONLs -> image paths (iNat rows use /datasets/ai/inaturalist/...).
HF model ids (Qwen/...) and relative paths (images/...) need no change.

    python tools/relocate_paths.py --dir runs --dir configs \\
        --map /home/obicakci_umass_edu/train_v2/runs=/root/vlm2vec/runs \\
        --map /home/obicakci_umass_edu/fgvc_sft_ckpt=/root/fgvc_sft_ckpt \\
        --map /datasets/ai/inaturalist=/root/data/inaturalist
    (add --dry-run first to see what would change)

Every vlm2vec_config.json under the --dir trees is updated, and every *.jsonl is rewritten
line by line (atomically, via a temp file). Only exact prefix matches are replaced.
"""
import argparse
import json
import os


def remap(value: str, maps: list[tuple[str, str]]) -> str:
    for old, new in maps:
        if value == old or value.startswith(old.rstrip("/") + "/"):
            return new.rstrip("/") + value[len(old.rstrip("/")):]
    return value


def fix_contract(path: str, maps, dry: bool) -> int:
    data = json.load(open(path))
    new = remap(data.get("base_model", ""), maps)
    if new == data.get("base_model"):
        return 0
    print(f"{path}: base_model {data['base_model']} -> {new}")
    if not dry:
        data["base_model"] = new
        json.dump(data, open(path, "w"), indent=2)
    return 1


def fix_jsonl(path: str, maps, dry: bool) -> int:
    changed, out = 0, []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            for side in ("query", "positive"):
                item = row.get(side) or {}
                if item.get("image"):
                    new = remap(item["image"], maps)
                    changed += new != item["image"]
                    item["image"] = new
            for neg in row.get("negatives", []):
                if neg.get("image"):
                    new = remap(neg["image"], maps)
                    changed += new != neg["image"]
                    neg["image"] = new
            out.append(json.dumps(row))
    if changed:
        print(f"{path}: {changed} image paths remapped")
        if not dry:
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                f.write("\n".join(out) + "\n")
            os.replace(tmp, path)
            stale = path + ".offsets.npz"          # byte offsets no longer match
            if os.path.exists(stale):
                os.remove(stale)
    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", action="append", required=True, help="tree to scan (repeatable)")
    ap.add_argument("--map", action="append", required=True, help="OLD_PREFIX=NEW_PREFIX (repeatable)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    maps = [tuple(m.split("=", 1)) for m in args.map]
    contracts = rows = 0
    for top in args.dir:
        for root, _, files in os.walk(top):
            for name in files:
                path = os.path.join(root, name)
                if name == "vlm2vec_config.json":
                    contracts += fix_contract(path, maps, args.dry_run)
                elif name.endswith(".jsonl"):
                    rows += fix_jsonl(path, maps, args.dry_run)
    print(f"{'would change' if args.dry_run else 'changed'}: {contracts} checkpoint configs, {rows} image paths")


if __name__ == "__main__":
    main()
