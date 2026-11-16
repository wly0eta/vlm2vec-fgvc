"""The handful of collectives multi-GPU training needs. All are no-ops on one process.

Why we don't let DDP sync gradients for us: DDP all-reduces inside backward(), once per
backward. A GradCache step runs one backward per chunk (plus one on the loss), so DDP
would either sync many times per step or, with `no_sync`, trip over parameters that a
text-only chunk never touches (the vision tower). Instead the trainer bypasses the DDP
wrapper, accumulates local gradients over the whole step, and calls
`all_reduce_gradients` exactly once before the optimizer step.
"""
import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return rank() == 0


def _collective_device() -> torch.device:
    """NCCL collectives need CUDA tensors; gloo (CPU tests) needs CPU tensors."""
    if is_distributed() and dist.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def all_gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    """Concatenate `tensor` from every rank along dim 0, in rank order.

    Ranks may hold different numbers of rows. The returned tensor keeps autograd only
    for this rank's own rows (other ranks' rows are constants here); gradients for those
    rows are computed on their own ranks, and `all_reduce_gradients` combines the results.
    """
    if not is_distributed():
        return tensor
    sizes = all_gather_objects(tensor.shape[0])
    padded = tensor.detach()
    if padded.shape[0] < max(sizes):
        pad = padded.new_zeros((max(sizes) - padded.shape[0], *padded.shape[1:]))
        padded = torch.cat([padded, pad])
    buffers = [torch.empty_like(padded) for _ in sizes]
    dist.all_gather(buffers, padded.contiguous())
    parts = [buf[:n] for buf, n in zip(buffers, sizes)]
    parts[rank()] = tensor  # the original, still attached to the autograd graph
    return torch.cat(parts)


def all_gather_objects(obj) -> list:
    """Gather one picklable object from every rank (used for label strings and sizes)."""
    if not is_distributed():
        return [obj]
    out = [None] * world_size()
    dist.all_gather_object(out, obj)
    return out


def all_reduce_gradients(parameters, average: bool) -> None:
    """Combine `.grad` across ranks in one collective per dtype.

    `average=False` sums (correct when every rank computed the gradient of the same
    global loss w.r.t. its own rows); `average=True` averages (correct when each rank
    computed its own local loss). Parameters without a gradient on this rank contribute
    zeros, so every rank always passes identically shaped buffers.
    """
    if not is_distributed():
        return
    params = [p for p in parameters if p.requires_grad]
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    by_dtype: dict[torch.dtype, list[torch.Tensor]] = {}
    for p in params:
        by_dtype.setdefault(p.grad.dtype, []).append(p.grad)
    op = dist.ReduceOp.AVG if average and dist.get_backend() == "nccl" else dist.ReduceOp.SUM
    for grads in by_dtype.values():
        flat = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flat, op=op)
        if average and op == dist.ReduceOp.SUM:   # gloo has no AVG
            flat /= world_size()
        for grad, synced in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
            grad.copy_(synced)


def replicas_in_sync(parameters, rtol: float = 1e-5) -> bool:
    """True if every rank holds the same trainable weights (compared via a checksum).

    Since we sync gradients ourselves instead of through DDP's hooks, this is the cheap
    end-of-run proof that the replicas never drifted apart."""
    if not is_distributed():
        return True
    checksum = float(sum(p.detach().double().sum() for p in parameters if p.requires_grad))
    sums = all_gather_objects(checksum)
    return max(sums) - min(sums) <= rtol * max(1.0, abs(sums[0]))


def broadcast_flag(flag: bool) -> bool:
    """Make every rank adopt rank 0's decision (e.g. 'save now'), so all ranks enter
    checkpointing together instead of deadlocking on a collective only some reached."""
    if not is_distributed():
        return flag
    t = torch.tensor([1 if flag else 0], device=_collective_device())
    dist.broadcast(t, src=0)
    return bool(t.item())
