"""Shared fixtures.

Most tests are pure torch and run anywhere. Tests marked with the `tiny_model` fixture
need a small Qwen2-VL checkpoint (real processor, toy weights) at $VLM2VEC_TINY_MODEL;
they are skipped otherwise. Build one with: python tests/make_tiny_model.py <dir>
"""
import os

import pytest
import torch
import torch.nn.functional as F


def unit_vectors(n: int, dim: int = 8, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, dim, generator=g), dim=-1)


@pytest.fixture
def tiny_model() -> str:
    path = os.environ.get("VLM2VEC_TINY_MODEL")
    if not path or not os.path.isdir(path):
        pytest.skip("set VLM2VEC_TINY_MODEL to a tiny Qwen2-VL dir (tests/make_tiny_model.py)")
    return path
