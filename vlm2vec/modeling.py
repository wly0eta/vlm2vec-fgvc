"""Building the model and the embedding forward pass."""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForImageTextToText

from .checkpoint import (EmbeddingContract, base_of, is_adapter_dir, load_adapter_weights,
                         resolve_trained_checkpoint)
from .config import ModelConfig

log = logging.getLogger(__name__)


def resolve_attn_implementation(attn_implementation: str, dtype: torch.dtype) -> str:
    """"auto" -> flash_attention_2 when it can run (flash-attn installed, a CUDA GPU, a 16-bit
    dtype), else sdpa. Both compute exact attention; FA2 is faster and lighter on memory, and is
    what upstream VLM2Vec uses for the Qwen-VL backbones."""
    if attn_implementation != "auto":
        return attn_implementation
    try:
        import flash_attn  # noqa: F401
        usable = torch.cuda.is_available() and dtype in (torch.bfloat16, torch.float16)
    except ImportError:
        usable = False
    choice = "flash_attention_2" if usable else "sdpa"
    log.info("attention: %s", choice)
    return choice


def load_backbone(model_name: str, dtype: torch.dtype, attn_implementation: str = "auto"):
    """Return a plain HF vision-language model with any adapter chain merged in.

    If `model_name` is a LoRA-adapter directory (e.g. the mentor's FGVC-SFT checkpoint, or
    one of our own runs), we load its base -- recursively, so chains work -- apply the
    adapter with verification, and merge it into the weights. A fresh LoRA can then be
    trained on top, which is how training continues from an earlier stage.
    """
    attn_implementation = resolve_attn_implementation(attn_implementation, dtype)
    if not is_adapter_dir(model_name):
        return AutoModelForImageTextToText.from_pretrained(
            model_name, dtype=dtype, attn_implementation=attn_implementation)
    base = load_backbone(base_of(model_name), dtype, attn_implementation)
    adapted = PeftModel.from_pretrained(base, model_name)
    load_adapter_weights(adapted, model_name)
    log.info("merged adapter %s into its base", model_name)
    return adapted.merge_and_unload()


def last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Hidden state at each sequence's last real (unpadded) token.

    Correct for left *and* right padding: we locate the last 1 in the mask instead of
    assuming right padding (the old `mask.sum() - 1` silently picks the wrong token under
    left padding -- which is the Qwen tokenizer's default).
    """
    seq_len = attention_mask.shape[1]
    last = seq_len - 1 - attention_mask.flip(dims=[1]).int().argmax(dim=1)
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]


class VLM2VecModel(nn.Module):
    """A vision-language model used as an embedding function.

    `encode(inputs)` runs the decoder stack -- skipping the LM head, which an embedding
    model never needs -- takes the hidden state of the last real token, and L2-normalizes
    it in float32. In a causal model the last token is the only position that has
    attended to the whole image + text input, so it serves as the summary vector.
    """

    def __init__(self, backbone, temperature: float):
        super().__init__()
        self.backbone = backbone
        self.temperature = temperature

    # --- construction --------------------------------------------------------------------

    @classmethod
    def for_training(cls, cfg: ModelConfig, dtype: torch.dtype) -> "VLM2VecModel":
        """Base (with any adapter chain merged) + a fresh trainable LoRA."""
        backbone = load_backbone(cfg.model_name, dtype, cfg.attn_implementation)
        lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                          target_modules=list(cfg.lora_target_modules), bias="none",
                          task_type="FEATURE_EXTRACTION")
        model = cls(get_peft_model(backbone, lora), cfg.temperature)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        log.info("trainable params: %s / %s (%.2f%%)", f"{trainable:,}", f"{total:,}", 100 * trainable / total)
        return model

    @classmethod
    def for_inference(cls, checkpoint: str | None, base_model: str | None = None,
                      dtype: torch.dtype = torch.bfloat16,
                      attn_implementation: str = "auto") -> tuple["VLM2VecModel", EmbeddingContract]:
        """Load a trained run (or, with checkpoint=None, the untrained base) for embedding.

        `checkpoint` may be a run dir: it resolves to the final adapter, or the newest
        checkpoint-N while training is still going. The base model comes from the
        checkpoint's contract; `base_model` is only needed for pre-refactor checkpoints,
        which have no contract (pass the model_name they were trained from).
        """
        contract = None
        if checkpoint:
            checkpoint = resolve_trained_checkpoint(checkpoint)
            contract = EmbeddingContract.load(checkpoint)
        if base_model and contract and base_model != contract.base_model:
            log.warning("--base-model %s overrides the checkpoint's recorded base %s",
                        base_model, contract.base_model)
        base = base_model or (contract.base_model if contract else None)
        if base is None:
            raise ValueError(f"{checkpoint} has no vlm2vec_config.json (pre-refactor checkpoint): pass "
                             "--base-model with the model it was trained from (for runs on the mentor "
                             "checkpoint, that is the mentor's adapter dir, not the bare 7B)")
        backbone = load_backbone(base, dtype, attn_implementation)
        if checkpoint:
            # from_pretrained builds LoRA from the checkpoint's own adapter_config, so rank /
            # targets always match what was trained (the old eval rebuilt them from defaults).
            backbone = PeftModel.from_pretrained(backbone, checkpoint)
            load_adapter_weights(backbone, checkpoint)
        contract = contract or EmbeddingContract(base_model=base)
        if base_model:
            contract.base_model = base_model
        return cls(backbone, contract.temperature).eval(), contract

    # --- forward -------------------------------------------------------------------------

    @property
    def decoder(self) -> nn.Module:
        """The image+text transformer stack without the LM head (HF's `.model` submodule
        of every *ForConditionalGeneration VLM)."""
        base = self.backbone.get_base_model() if isinstance(self.backbone, PeftModel) else self.backbone
        decoder = getattr(base, "model", None)
        if decoder is None:
            raise TypeError(f"{type(base).__name__} has no `.model` submodule; expected an HF "
                            "*ForConditionalGeneration vision-language model")
        return decoder

    def encode(self, inputs) -> torch.Tensor:
        out = self.decoder(**inputs, use_cache=False, return_dict=True)
        pooled = last_token_pool(out.last_hidden_state, inputs["attention_mask"])
        return F.normalize(pooled.float(), dim=-1)

    forward = encode

    # --- plumbing the HF Trainer expects -----------------------------------------------------

    @property
    def config(self):
        return self.backbone.config

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None, **kwargs):
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs or {"use_reentrant": False}, **kwargs)
