"""VLM2Vec: turn a generative vision-language model into a contrastive embedding model.

Package layout (each module owns one concern):

    config.py        Every tunable knob, as dataclasses parsed from the command line.
    modeling.py      Backbone loading, LoRA, last-token pooling -> VLM2VecModel.encode().
    checkpoint.py    Saving/loading adapters (with load verification) + the embedding contract.
    data.py          JSONL rows, the weighted task mixture, and the collator that tokenizes.
    losses.py        InfoNCE with same-label (false-negative) masking.
    gradcache.py     Two-pass GradCache: big contrastive batches at one-chunk memory cost.
    distributed.py   The few collectives multi-GPU training needs.
    trainer.py       HF Trainer subclass wiring the pieces into one optimizer step.
    callbacks.py     Time-based checkpoints, graceful stop-for-requeue, live metrics.
    plotting.py      Training-curve rendering (pure matplotlib, no training logic).
    evaluation.py    Embedding-based metrics: multiple choice, classification, retrieval.
    sources/         Raw datasets -> the shared JSONL row format (local iNat, HF Hub sets).

Entry points live at the repository root: train.py, evaluate.py, build_dataset.py.
"""
