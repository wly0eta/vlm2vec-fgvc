"""GradCache: contrastive training with a large batch at the memory cost of one chunk.

Contrastive quality grows with the number of negatives, i.e. with batch size, but the
activations for a 32-row multimodal batch don't fit next to a 7B model. GradCache
(Gao et al., 2021) splits the step in two passes over chunks of the batch:

  1. `encode_without_grad`: embed every chunk with autograd off. Embeddings are tiny,
     so the whole batch's embeddings fit even though its activations don't.
  2. The caller computes the loss on detached copies of those embeddings and backprops
     to get d(loss)/d(embedding) for every row.
  3. `replay_with_grad`: re-embed each chunk with autograd on and backprop
     sum(embedding * d(loss)/d(embedding)). By the chain rule this deposits exactly the
     parameter gradient the full-batch loss would have produced, one chunk at a time.

Step 3 is only exact if the replayed forward is *identical* to the cached one. Dropout
(LoRA dropout here) draws fresh random masks on every call, so we snapshot the RNG
before each cached chunk and restore it for that chunk's replay (RandContext). The
previous trainer skipped this, so its gradients came from embeddings with different
dropout masks than the ones the loss was computed on.
"""
import torch


class RandContext:
    """Captures RNG state at construction; `with ctx:` replays from that state, then
    restores whatever state was current before entering (so replays don't disturb the
    global random stream)."""

    def __init__(self, device: torch.device):
        self.device = device
        self.snapshot = _get_rng(device)

    def __enter__(self):
        self._saved = _get_rng(self.device)
        _set_rng(self.device, self.snapshot)

    def __exit__(self, *exc):
        _set_rng(self.device, self._saved)


def _get_rng(device: torch.device) -> dict:
    state = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    elif device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def _set_rng(device: torch.device, state: dict) -> None:
    torch.set_rng_state(state["cpu"])
    if "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device)
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def encode_without_grad(encode, chunks, device) -> tuple[torch.Tensor, list[RandContext]]:
    """Pass 1. Returns all embeddings (concatenated) and one RNG snapshot per chunk."""
    embeddings, rngs = [], []
    for chunk in chunks:
        rngs.append(RandContext(device))
        with torch.no_grad():
            embeddings.append(encode(chunk))
    return torch.cat(embeddings), rngs


def replay_with_grad(encode, chunks, rngs: list[RandContext], embedding_grad: torch.Tensor) -> None:
    """Pass 3. Accumulates parameter gradients into `.grad`, one chunk at a time."""
    offset = 0
    for chunk, rng in zip(chunks, rngs):
        with rng:
            embeddings = encode(chunk)
        n = embeddings.shape[0]
        surrogate = (embeddings * embedding_grad[offset:offset + n].to(embeddings.dtype)).sum()
        surrogate.backward()
        offset += n
    assert offset == embedding_grad.shape[0], "chunks and cached gradient are out of sync"
