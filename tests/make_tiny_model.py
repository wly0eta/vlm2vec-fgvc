"""Write a tiny random Qwen2-VL (real tokenizer + processor, toy weights, ~40M params) for
CPU tests:   python tests/make_tiny_model.py /tmp/tiny-qwen2vl
then:        VLM2VEC_TINY_MODEL=/tmp/tiny-qwen2vl pytest
"""
import sys

import torch
from transformers import AutoConfig, AutoProcessor, Qwen2VLForConditionalGeneration

SOURCE = "Qwen/Qwen2-VL-2B-Instruct"


def main(out: str) -> None:
    cfg = AutoConfig.from_pretrained(SOURCE)
    text, vision = cfg.text_config, cfg.vision_config
    text.hidden_size, text.intermediate_size, text.num_hidden_layers = 256, 256, 2
    text.num_attention_heads, text.num_key_value_heads = 2, 1   # head_dim 128 keeps mrope sections valid
    if getattr(text, "layer_types", None):
        text.layer_types = text.layer_types[:2]                 # must match num_hidden_layers
    vision.embed_dim, vision.depth, vision.num_heads, vision.hidden_size = 64, 1, 2, 256
    torch.manual_seed(0)
    Qwen2VLForConditionalGeneration(cfg).save_pretrained(out)
    AutoProcessor.from_pretrained(SOURCE).save_pretrained(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1])
