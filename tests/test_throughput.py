import torch
import torch.nn as nn

from vlm2vec.trainer import count_params, count_work


def test_count_work_sums_real_tokens_and_patches():
    chunk = {"attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]),
             "image_grid_thw": torch.tensor([[1, 4, 6], [1, 2, 2]])}
    text = {"attention_mask": torch.tensor([[1, 1, 0]])}
    assert count_work({"query": [chunk], "positive": [text]}) == (7 + 2, 24 + 4)


def test_count_params_splits_vision_and_skips_embeddings():
    m = nn.Module()
    m.visual = nn.Linear(2, 3)            # 9 params
    m.layer = nn.Linear(4, 4)             # 20
    m.embed_tokens = nn.Embedding(10, 4)  # skipped
    m.lm_head = nn.Linear(4, 10)          # skipped
    assert count_params(m) == (20, 9)
