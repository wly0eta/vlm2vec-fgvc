"""Saving and loading trained adapters.

A checkpoint directory holds:
    adapter_model.safetensors   LoRA weights (PEFT format)
    adapter_config.json         LoRA shape/targets + the HF base it was applied to
    vlm2vec_config.json         the embedding contract (below)
    processor / tokenizer files

Two things here exist because of failures we actually hit:

* `load_adapter_weights` verifies that weights landed. Some PEFT versions (0.20.0)
  silently skip tensors whose keys don't line up, leaving LoRA at its zero init: no
  error, and the "trained" model is just the base model. That would have invalidated
  every continued-training and evaluation result without anyone noticing.

* The embedding contract records what the weights alone don't: the true base model
  (adapter_config.json only names the HF model a merged backbone came from, so a run
  trained on top of the mentor's FGVC checkpoint looked like plain Qwen2.5-VL-7B),
  the image budget, and the pooling/prompt format. Evaluation reads it, so eval can no
  longer silently use the wrong base or a different resolution than training did.
"""
import json
import os
import re
from dataclasses import asdict, dataclass, fields

from .config import DEFAULT_MAX_PIXELS, QUERY_TEMPLATE

ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
CONTRACT_FILE = "vlm2vec_config.json"


@dataclass
class EmbeddingContract:
    """Everything besides the weights needed to reproduce a checkpoint's embeddings."""

    base_model: str           # what to load before applying this adapter (HF id, model dir, or adapter dir)
    max_pixels: int = DEFAULT_MAX_PIXELS
    max_len: int = 1024
    temperature: float = 0.02
    pooling: str = "last_token"
    normalize: bool = True
    query_template: str = QUERY_TEMPLATE
    format_version: int = 1

    def save(self, directory: str) -> None:
        with open(os.path.join(directory, CONTRACT_FILE), "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, directory: str) -> "EmbeddingContract | None":
        path = os.path.join(directory, CONTRACT_FILE)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = json.load(f)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# --- what kind of directory is this? ------------------------------------------------------

def is_adapter_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, ADAPTER_CONFIG))


def base_of(adapter_dir: str) -> str:
    """The model this adapter must be applied on top of.

    Prefers the contract (correct for runs that continued from another adapter) and
    falls back to adapter_config.json (correct for adapters trained on an HF model).
    """
    contract = EmbeddingContract.load(adapter_dir)
    if contract is not None:
        return contract.base_model
    with open(os.path.join(adapter_dir, ADAPTER_CONFIG)) as f:
        return json.load(f)["base_model_name_or_path"]


def hf_model_of(name: str) -> str:
    """Follow a chain of adapter dirs down to the underlying HF model (used for the
    processor, which only exists for the real base)."""
    seen = set()
    while is_adapter_dir(name):
        if name in seen:
            raise ValueError(f"adapter base chain loops at {name}")
        seen.add(name)
        name = base_of(name)
    return name


# --- finding checkpoints in a run directory ------------------------------------------------

def newest_step_checkpoint(run_dir: str) -> str | None:
    """`run_dir/checkpoint-N` with the largest N, or None (used to resume training)."""
    if not os.path.isdir(run_dir):
        return None
    steps = [int(m.group(1)) for d in os.listdir(run_dir) if (m := re.fullmatch(r"checkpoint-(\d+)", d))]
    return os.path.join(run_dir, f"checkpoint-{max(steps)}") if steps else None


def resolve_trained_checkpoint(path: str) -> str:
    """For evaluation: a run dir's final adapter if training finished, else its newest
    checkpoint-N, else `path` itself if it is a checkpoint."""
    if os.path.exists(os.path.join(path, ADAPTER_WEIGHTS)):
        return path
    newest = newest_step_checkpoint(path)
    if newest is None:
        raise FileNotFoundError(f"no adapter or checkpoint-N under {path}")
    return newest


# --- weights in and out ----------------------------------------------------------------------

def load_adapter_weights(peft_model, adapter_dir: str, adapter_name: str = "default") -> None:
    """Load `adapter_dir`'s weights into `peft_model` and verify they actually landed.

    Raises if any saved LoRA tensor matched no module, or if every LoRA-B matrix is still
    zero afterwards (the signature of the silent no-op load described above).
    """
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    state = load_file(os.path.join(adapter_dir, ADAPTER_WEIGHTS))
    if not any("lora_" in k for k in state):
        raise RuntimeError(f"{adapter_dir}/{ADAPTER_WEIGHTS} contains no LoRA tensors")
    result = set_peft_model_state_dict(peft_model, state, adapter_name=adapter_name)
    unexpected = [k for k in (getattr(result, "unexpected_keys", None) or []) if "lora_" in k]
    if unexpected:
        raise RuntimeError(f"{len(unexpected)} LoRA tensors from {adapter_dir} matched no module "
                           f"(e.g. {unexpected[:2]}); the adapter does not fit this base model")
    lora_b = [p for n, p in peft_model.named_parameters() if "lora_B" in n and f".{adapter_name}." in n]
    if lora_b and all(float(p.detach().abs().max()) == 0.0 for p in lora_b):
        raise RuntimeError(f"adapter from {adapter_dir} loaded as all-zero LoRA-B (a no-op); "
                           "its weights did not apply")


def save_adapter(peft_backbone, processor, contract: EmbeddingContract, directory: str) -> None:
    os.makedirs(directory, exist_ok=True)
    peft_backbone.save_pretrained(directory)
    if processor is not None:
        processor.save_pretrained(directory)
    contract.save(directory)
