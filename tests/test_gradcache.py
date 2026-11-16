"""GradCache must produce exactly the gradient of the un-chunked loss -- including with
dropout, which is what RandContext is for."""
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from vlm2vec.losses import contrastive_loss
from vlm2vec.trainer import contrastive_step

CPU = torch.device("cpu")


class TinyEncoder(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.proj = nn.Linear(6, 8)
        self.drop = nn.Dropout(dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.drop(self.proj(x)), dim=-1)


def _batch(rows=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, 6, generator=g), torch.randn(rows, 6, generator=g), torch.randn(2 * rows, 6, generator=g)


def _chunk(x, size):
    return list(torch.split(x, size))


def reference_grads(model, q, p, n, chunk, seed):
    """Plain autograd through the whole batch, encoding chunks in the same order GradCache
    does, so dropout draws the same masks."""
    model.zero_grad()
    torch.manual_seed(seed)
    enc = lambda xs: torch.cat([model.encode(c) for c in _chunk(xs, chunk)])
    loss, _ = contrastive_loss(enc(q), enc(p), enc(n), 0.05)
    loss.backward()
    return loss.detach(), [x.grad.clone() for x in model.parameters()]


def gradcache_grads(model, q, p, n, chunk, seed):
    model.zero_grad()
    torch.manual_seed(seed)
    chunks = {"query": _chunk(q, chunk), "positive": _chunk(p, chunk), "negative": _chunk(n, chunk)}
    result = contrastive_step(model.encode, chunks, [], [], temperature=0.05, device=CPU,
                              mask_false_negatives=False)
    return result.loss, [x.grad.clone() for x in model.parameters()]


def test_matches_full_batch_gradient_with_dropout():
    torch.manual_seed(0)
    model = TinyEncoder(dropout=0.3).train()
    q, p, n = _batch()
    ref_loss, ref = reference_grads(model, q, p, n, chunk=2, seed=7)
    gc_loss, gc = gradcache_grads(model, q, p, n, chunk=2, seed=7)
    assert torch.allclose(ref_loss, gc_loss)
    for a, b in zip(ref, gc):
        assert torch.allclose(a, b, atol=1e-6)


def test_without_rng_replay_the_gradient_is_wrong(monkeypatch):
    """Documents the bug the old trainer had: replaying without restoring the RNG
    gives dropout masks different from the cached pass, and a different gradient."""
    import vlm2vec.gradcache as gc_module
    monkeypatch.setattr(gc_module.RandContext, "__enter__", lambda self: None)
    monkeypatch.setattr(gc_module.RandContext, "__exit__", lambda self, *a: None)
    torch.manual_seed(0)
    model = TinyEncoder(dropout=0.3).train()
    q, p, n = _batch()
    _, ref = reference_grads(model, q, p, n, chunk=2, seed=7)
    _, broken = gradcache_grads(model, q, p, n, chunk=2, seed=7)
    assert not all(torch.allclose(a, b, atol=1e-6) for a, b in zip(ref, broken))


def test_chunk_size_does_not_change_the_gradient_without_dropout():
    torch.manual_seed(0)
    model = TinyEncoder(dropout=0.0)
    q, p, n = _batch()
    _, by_one = gradcache_grads(model, q, p, n, chunk=1, seed=0)
    _, by_six = gradcache_grads(model, q, p, n, chunk=6, seed=0)
    for a, b in zip(by_one, by_six):
        assert torch.allclose(a, b, atol=1e-6)
