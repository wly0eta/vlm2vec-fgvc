"""Raw datasets -> the shared JSONL row format.

Each source module yields `common.Record`s; `common.build()` filters, caps, samples
negatives, splits, and writes. `build_dataset.py` at the repo root is the CLI.
"""
