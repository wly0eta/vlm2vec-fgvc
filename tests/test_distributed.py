"""Multi-GPU maths, checked on CPU with two real processes (gloo backend).

Splitting a batch across 2 ranks with cross-device negatives + a summed all-reduce must
give exactly the gradient (and loss) of the whole batch on one process -- including
false-negative masking when the duplicate label sits on the *other* rank.
"""
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from vlm2vec.distributed import all_reduce_gradients
from vlm2vec.trainer import contrastive_step

CPU = torch.device("cpu")
LABELS = ["robin", "cardinal", "jay", "robin"]          # "robin" on rank 0 and rank 1
NEGATIVE_LABELS = ["jay", "wren", "robin", "finch"]      # one distractor per row


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(6, 8)

    def encode(self, x):
        return F.normalize(self.proj(x), dim=-1)


def _data():
    g = torch.Generator().manual_seed(1)
    return torch.randn(4, 6, generator=g), torch.randn(4, 6, generator=g), torch.randn(4, 6, generator=g)


def _model():
    torch.manual_seed(0)
    return TinyEncoder()


def _step(model, q, p, n, labels, negative_labels, global_loss):
    chunks = {"query": list(torch.split(q, 1)), "positive": list(torch.split(p, 1)),
              "negative": list(torch.split(n, 1))}
    return contrastive_step(model.encode, chunks, labels, negative_labels, temperature=0.05, device=CPU,
                            mask_false_negatives=True, global_loss=global_loss)


def _worker(rank: int, world: int, init_file: str, out_file: str):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    q, p, n = _data()
    rows = slice(rank * 2, rank * 2 + 2)
    model = _model()
    result = _step(model, q[rows], p[rows], n[rows], LABELS[rows], NEGATIVE_LABELS[rows], global_loss=True)
    all_reduce_gradients(model.parameters(), average=False)
    if rank == 0:
        torch.save({"loss": result.loss, "grads": [x.grad for x in model.parameters()],
                    "candidates": result.num_candidates}, out_file)
    dist.barrier()
    dist.destroy_process_group()


def test_two_ranks_match_one_process(tmp_path):
    q, p, n = _data()
    model = _model()
    single = _step(model, q, p, n, LABELS, NEGATIVE_LABELS, global_loss=False)
    single_grads = [x.grad for x in model.parameters()]

    out_file = str(tmp_path / "rank0.pt")
    mp.spawn(_worker, args=(2, str(tmp_path / "init"), out_file), nprocs=2, join=True)
    multi = torch.load(out_file)

    assert multi["candidates"] == 8   # 4 positives + 4 distractors, gathered from both ranks
    assert torch.allclose(multi["loss"], single.loss, atol=1e-6)
    for a, b in zip(single_grads, multi["grads"]):
        assert torch.allclose(a, b, atol=1e-6)
