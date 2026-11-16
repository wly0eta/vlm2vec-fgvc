"""InfoNCE contrastive loss with same-label (false-negative) masking.

Layout. For a batch of B rows, candidates are [positives (B) ; negatives (N)]. Query i's
correct candidate is column i, its own positive; every other column is a negative for it:
the other rows' positives ("in-batch negatives") plus every row's explicit distractors.

The false-negative problem. If two rows share a label (two images of the same species,
two questions whose answer is "yes"), row j's positive is a *correct* match for query i,
yet plain InfoNCE pushes query i away from it. Identical label text also means identical
candidate embeddings, so the loss is asking for something impossible. We set those
logits to -inf so they drop out of the softmax entirely. Distinct-but-similar labels
(two warbler species) stay as negatives: separating those is the point of FGVC.
"""
import torch
import torch.nn.functional as F


def false_negative_mask(query_labels: list[str], candidate_labels: list[str]) -> torch.Tensor:
    """Bool [Q, C]: True where candidate c carries query q's label but is not column q."""
    ids: dict[str, int] = {}
    q = torch.tensor([ids.setdefault(label, len(ids)) for label in query_labels])
    c = torch.tensor([ids.setdefault(label, len(ids)) for label in candidate_labels])
    mask = q[:, None] == c[None, :]
    own = torch.arange(len(query_labels))
    mask[own, own] = False  # never mask a query's own positive
    return mask


def contrastive_loss(
    query: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor | None,
    temperature: float,
    query_labels: list[str] | None = None,
    negative_labels: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mean InfoNCE loss, in-batch top-1 accuracy).

    Embeddings are expected L2-normalized, so dot products are cosine similarities.
    Similarities are computed in float32: at temperature 0.02 logits are cosine x 50, and
    bf16's ~3 significant digits would put noise of order 0.1 on every logit.

    Pass `query_labels` (each row's positive label) and `negative_labels` to enable
    false-negative masking; omit them to get plain InfoNCE.
    """
    candidates = positive if negative is None or negative.numel() == 0 else torch.cat([positive, negative])
    logits = query.float() @ candidates.float().T / temperature
    if query_labels is not None:
        candidate_labels = list(query_labels) + list(negative_labels or [])
        logits = logits.masked_fill(false_negative_mask(query_labels, candidate_labels).to(logits.device),
                                    float("-inf"))
    target = torch.arange(query.shape[0], device=logits.device)
    loss = F.cross_entropy(logits, target)
    accuracy = (logits.argmax(dim=1) == target).float().mean()
    return loss, accuracy.detach()
