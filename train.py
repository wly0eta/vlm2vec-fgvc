"""Train a VLM2Vec embedding model (contrastive LoRA fine-tuning of a vision-language model).

    # one GPU
    python train.py --model_name Qwen/Qwen3-VL-2B-Instruct \\
        --dataset_config configs/inat_1m.yaml --output_dir runs/inat_1m --num_train_epochs 1

    # several GPUs on one node: identical arguments
    torchrun --standalone --nproc_per_node 4 train.py <same arguments>

Every option: `python train.py --help` (model/data options from vlm2vec/config.py, plus all
standard HF TrainingArguments). Re-running with the same --output_dir resumes from the
newest checkpoint. A finished run writes <output_dir>/TRAINING_COMPLETE. If the stop file
appears (see StopFileCallback) the run saves and exits early with code 3 -- reported as 1
under torchrun, which is why slurm/train.sbatch checks the marker file, not the exit code.
"""
import dataclasses
import functools
import json
import logging
import os
import sys

import torch
from transformers import HfArgumentParser

from vlm2vec.callbacks import (LiveMetricsCallback, PruneCheckpointsCallback, StopFileCallback,
                               TimedCheckpointCallback, sample_rows)
from vlm2vec.checkpoint import EmbeddingContract, newest_step_checkpoint
from vlm2vec.config import DataConfig, ModelConfig, TrainConfig
from vlm2vec.data import (ContrastiveCollator, JsonlRows, MixedDataset, ReseedMixtureCallback,
                          image_tokens_for, load_processor, positive_item)
from vlm2vec.distributed import replicas_in_sync
from vlm2vec.evaluation import Embedder, classification, multiple_choice
from vlm2vec.modeling import VLM2VecModel
from vlm2vec.trainer import VLM2VecTrainer

EXIT_STOPPED_FOR_REQUEUE = 3
COMPLETE_MARKER = "TRAINING_COMPLETE"
log = logging.getLogger("train")


def allow_resume() -> None:
    """Let a resume read back the optimizer/RNG files this script itself wrote.

    1. HF loads rng_state.pth with torch.load(weights_only=True), which rejects the numpy
       arrays inside it (numpy's RNG state). We allow-list exactly numpy's array/dtype types
       rather than turning weights_only off. Without this, every preempted job crashed on
       resume instead of continuing.
    2. transformers refuses torch.load() entirely on torch < 2.6 (CVE-2025-32434); Unity's
       CUDA 12.1 wheels stop at torch 2.5, so there we skip that version check too.
    """
    import numpy as np
    import numpy.dtypes
    numpy_core = getattr(np, "_core", None) or np.core   # numpy 2 renamed core -> _core
    torch.serialization.add_safe_globals(
        [numpy_core.multiarray._reconstruct, np.ndarray, np.dtype]
        + [getattr(numpy.dtypes, name) for name in dir(numpy.dtypes) if name.endswith("DType")])
    major, minor = (int(x) for x in torch.__version__.split(".")[:2])
    if (major, minor) >= (2, 6):
        return
    for module in ("transformers.trainer", "transformers.modeling_utils", "transformers.utils.import_utils"):
        mod = sys.modules.get(module) or __import__(module, fromlist=["_"])
        if hasattr(mod, "check_torch_load_is_safe"):
            mod.check_torch_load_is_safe = lambda: None


def live_eval_sets(cfg: TrainConfig) -> dict:
    """Small seeded subsets for in-training eval. Classification keeps the FULL label set
    of the file (scoring a 300-row subset against only its own labels inflates accuracy)."""
    sets = {}
    if cfg.live_eval_mc:
        rows = list(JsonlRows(cfg.live_eval_mc))
        sets[os.path.basename(cfg.live_eval_mc)] = (multiple_choice, sample_rows(rows, cfg.live_eval_rows))
    if cfg.live_eval_classify:
        rows = list(JsonlRows(cfg.live_eval_classify))
        labels = sorted({positive_item(r).text for r in rows})
        sets[os.path.basename(cfg.live_eval_classify)] = (
            functools.partial(classification, class_pool=labels), sample_rows(rows, cfg.live_eval_rows))
    return sets


def main() -> None:
    model_cfg, data_cfg, train_cfg = HfArgumentParser((ModelConfig, DataConfig, TrainConfig)).parse_args_into_dataclasses()
    is_main = train_cfg.process_index == 0   # also initializes torch.distributed under torchrun
    logging.basicConfig(level=logging.INFO if is_main else logging.WARNING,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    allow_resume()
    if is_main and os.path.exists(train_cfg.stop_file):
        os.remove(train_cfg.stop_file)   # left over from the chunk before a requeue

    dtype = torch.bfloat16 if train_cfg.bf16 else torch.float16 if train_cfg.fp16 else torch.float32
    model = VLM2VecModel.for_training(model_cfg, dtype)
    processor = load_processor(model_cfg.model_name, data_cfg.max_pixels)
    try:
        log.info("image budget: max_pixels=%d -> %d vision tokens for a 1024x1024 image",
                 data_cfg.max_pixels, image_tokens_for(processor))
    except Exception as err:   # informational only
        log.info("image budget: max_pixels=%d (token count unavailable: %s)", data_cfg.max_pixels, err)

    dataset = MixedDataset.from_yaml(data_cfg.dataset_config, seed=train_cfg.seed)
    log.info("training mixture: %s", dataset.describe())
    if is_main:   # how this run was trained, for tools/results_db.py (first launch only: resumes keep it)
        os.makedirs(train_cfg.output_dir, exist_ok=True)
        cfg_path = os.path.join(train_cfg.output_dir, "run_config.json")
        if not os.path.exists(cfg_path):
            with open(cfg_path, "w") as f:
                json.dump({"argv": sys.argv[1:], "world_size": train_cfg.world_size,
                           "global_batch": train_cfg.per_device_train_batch_size * train_cfg.world_size
                           * train_cfg.gradient_accumulation_steps,
                           "mixture": dataset.describe(),
                           **{k: v for cfg in (model_cfg, data_cfg) for k, v in dataclasses.asdict(cfg).items()},
                           **{k: v for k, v in train_cfg.to_dict().items() if isinstance(v, (int, float, str, bool, type(None)))}},
                          f, indent=1, default=str)

    contract = EmbeddingContract(base_model=model_cfg.model_name, max_pixels=data_cfg.max_pixels,
                                 max_len=data_cfg.max_len, temperature=model_cfg.temperature)
    stop = StopFileCallback(train_cfg.stop_file)
    embedder = lambda m: Embedder(m, processor, data_cfg.max_len, batch_size=train_cfg.gc_chunk_size)
    callbacks = [
        ReseedMixtureCallback(dataset),
        TimedCheckpointCallback(train_cfg.save_every_minutes),
        PruneCheckpointsCallback(),
        stop,
        LiveMetricsCallback(train_cfg.output_dir, embedder, live_eval_sets(train_cfg) if is_main else {},
                            train_cfg.live_eval_every_steps),
    ]
    trainer = VLM2VecTrainer(
        model=model, args=train_cfg, train_dataset=dataset,
        data_collator=ContrastiveCollator(processor, train_cfg.gc_chunk_size, data_cfg.max_len),
        callbacks=callbacks, contract=contract, processor=processor,
    )

    resume = train_cfg.resume_from_checkpoint or newest_step_checkpoint(train_cfg.output_dir)
    if resume:
        log.info("resuming from %s", resume)
    trainer.train(resume_from_checkpoint=resume)

    if not replicas_in_sync(model.parameters()):
        raise RuntimeError("GPU replicas diverged: trainable weights differ across ranks")
    if stop.stopped:
        sys.exit(EXIT_STOPPED_FOR_REQUEUE)
    trainer.save_model(train_cfg.output_dir)   # final adapter + contract at the run's top level
    if is_main:
        # The SLURM script keys off this file, not the exit code: torchrun reports any
        # non-zero worker exit (including our 3) as 1.
        with open(os.path.join(train_cfg.output_dir, COMPLETE_MARKER), "w") as f:
            json.dump({"global_step": trainer.state.global_step, "epoch": trainer.state.epoch}, f)
    log.info("finished: %s", train_cfg.output_dir)


if __name__ == "__main__":
    try:
        main()
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
