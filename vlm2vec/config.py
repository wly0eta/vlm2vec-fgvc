"""All configuration, as dataclasses.

`train.py` parses these from the command line with HfArgumentParser, so every field
here is also a `--flag` whose `--help` text is the `help` metadata below. That replaces
the old pile of environment variables: a run is now fully described by its command
line, which SLURM logs verbatim.
"""
from dataclasses import dataclass, field, fields

from transformers import TrainingArguments

# Qwen-VL vision tokens are 28x28-pixel patches after merging, so this is a 256-token
# image budget. Larger images are downscaled to fit; smaller ones are left alone.
DEFAULT_MAX_PIXELS = 256 * 28 * 28

# The projections LoRA is attached to. Matching by name also reaches the vision tower's
# MLP (Qwen-VL names it gate/up/down_proj too), so both towers adapt.
DEFAULT_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

# How a query is phrased to the model. Candidates (labels, answer options) are embedded as
# their bare text. These strings are part of what a checkpoint learned: changing them
# changes the embeddings, so they are recorded in every checkpoint's contract file.
QUERY_TEMPLATE = "Instruct: {instruction}\nQuery: {text}"
DEFAULT_INSTRUCTION = "Represent the query for retrieval."


@dataclass
class ModelConfig:
    model_name: str = field(
        default="Qwen/Qwen3-VL-2B-Instruct",
        metadata={"help": "HF model id, a local full-model dir, or a local LoRA-adapter dir. "
                          "An adapter dir is merged into its own base first, then a fresh adapter "
                          "is trained on top (how we continue from an earlier stage)."},
    )
    attn_implementation: str = field(
        default="auto",
        metadata={"help": "auto (flash_attention_2 if installed, else sdpa) | flash_attention_2 | sdpa | eager"},
    )
    lora_r: int = field(default=16, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA scaling numerator (effective scale alpha/r)."})
    lora_dropout: float = field(default=0.05, metadata={"help": "Dropout on the LoRA branch."})
    lora_target_modules: list[str] = field(
        default_factory=lambda: list(DEFAULT_LORA_TARGETS),
        metadata={"help": "Module names that receive LoRA adapters."},
    )
    temperature: float = field(
        default=0.02,
        metadata={"help": "InfoNCE temperature. Fixed (not learned), as in VLM2Vec."},
    )


@dataclass
class DataConfig:
    dataset_config: str = field(
        default="configs/aokvqa.yaml",
        metadata={"help": "YAML mapping task name -> {data_path, weight}. Weights are each "
                          "task's share of every epoch (see data.MixedDataset)."},
    )
    max_pixels: int = field(
        default=DEFAULT_MAX_PIXELS,
        metadata={"help": "Per-image pixel budget. Lower = fewer vision tokens = faster/less memory. "
                          "Saved into the checkpoint so evaluation uses the same value."},
    )
    max_len: int = field(
        default=1024,
        metadata={"help": "Token cap for text-only inputs. Inputs with an image are never "
                          "truncated (cutting image tokens corrupts the sample); over-long ones "
                          "are logged instead."},
    )
    # The per-epoch mixture draw uses HF's --seed (TrainConfig.seed), so one seed governs the run.


@dataclass
class TrainConfig(TrainingArguments):
    """HF TrainingArguments plus the knobs specific to contrastive GradCache training.

    Defaults below are overridden from HF's to match how this project trains.
    """

    # --- contrastive / GradCache --------------------------------------------------------
    gc_chunk_size: int = field(
        default=8,
        metadata={"help": "Rows encoded per forward pass inside a GradCache step. Sets peak "
                          "activation memory; does not change the maths."},
    )
    mask_false_negatives: bool = field(
        default=True,
        metadata={"help": "Drop candidates whose label equals the query's own label from that "
                          "query's negatives (e.g. two images of the same species in one batch)."},
    )
    cross_device_negatives: bool = field(
        default=True,
        metadata={"help": "With multiple GPUs, score each query against every GPU's candidates, "
                          "so negatives scale with the global batch rather than the per-GPU batch."},
    )

    # --- checkpointing / cluster ----------------------------------------------------------
    save_every_minutes: float = field(
        default=30.0,
        metadata={"help": "Save a checkpoint every N minutes of wall time (0 disables). Replaces "
                          "per-job save_steps tuning: the same setting is right for a fast 2B and a "
                          "slow 7B run."},
    )
    stop_file: str = field(
        default="",
        metadata={"help": "If this file appears, save and exit with code 3 (the SLURM script "
                          "creates it shortly before the wall-time limit, then requeues). "
                          "Default: <output_dir>/STOP."},
    )

    # --- live evaluation during training (rank 0 only) --------------------------------
    live_eval_mc: str = field(default="", metadata={"help": "Multiple-choice JSONL to track during training."})
    live_eval_classify: str = field(default="", metadata={"help": "Classification JSONL to track during training."})
    live_eval_every_steps: int = field(default=0, metadata={"help": "Run the live eval every N steps (0 = off)."})
    live_eval_rows: int = field(default=300, metadata={"help": "Random (seeded) subset size for live eval."})

    warmup_fraction: float = field(
        default=0.03,
        metadata={"help": "Warmup as a fraction of total steps (used unless --warmup_steps is given). "
                          "Separate field because transformers v5 moved this between versions."},
    )

    # --- HF defaults changed for this project -----------------------------------------
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    bf16: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = 5
    save_strategy: str = "no"             # checkpoints come from save_every_minutes / the stop file;
                                          # pass --save_strategy epoch to also keep one per epoch
    save_total_limit: int | None = None   # retention is PruneCheckpointsCallback's job (keeps epochs)
    remove_unused_columns: bool = False   # our batches are custom dicts, not model kwargs
    dataloader_drop_last: bool = True     # equal batches on every GPU; no tiny final batch
    dataloader_num_workers: int = 4       # tokenization + image decoding happen in workers
    report_to: str = "none"

    def __post_init__(self):
        if not self.stop_file:
            self.stop_file = f"{self.output_dir}/STOP"
        if self.gradient_checkpointing_kwargs is None:
            # Non-reentrant checkpointing is required with our manual multi-backward step.
            self.gradient_checkpointing_kwargs = {"use_reentrant": False}
        if not self.warmup_steps and self.warmup_fraction > 0:
            if "warmup_ratio" in {f.name for f in fields(TrainingArguments)}:
                self.warmup_ratio = self.warmup_fraction      # older v5: separate ratio field
            else:
                self.warmup_steps = self.warmup_fraction      # newer v5: float < 1 means a ratio
        super().__post_init__()
        if self.gradient_accumulation_steps != 1:
            raise ValueError(
                "gradient_accumulation_steps must be 1: GradCache already gives large batches at "
                "constant memory. Raise per_device_train_batch_size (and use more GPUs) instead."
            )
