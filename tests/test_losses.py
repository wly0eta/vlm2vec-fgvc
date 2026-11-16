import torch
import torch.nn.functional as F

from conftest import unit_vectors
from vlm2vec.losses import contrastive_loss, false_negative_mask


def test_mask_flags_same_label_except_own_column():
    mask = false_negative_mask(["a", "b", "a"], ["a", "b", "a", "b", "c"])
    T, F_ = True, False
    assert mask.tolist() == [
        [F_, F_, T, F_, F_],   # query 0 ("a"): row 2's positive is also "a"
        [F_, F_, F_, T, F_],   # query 1 ("b"): a distractor is also "b"
        [T, F_, F_, F_, F_],   # query 2 ("a"): row 0's positive is "a"
    ]


def test_unmasked_loss_is_standard_infonce():
    q, p, n = unit_vectors(4), unit_vectors(4, seed=1), unit_vectors(6, seed=2)
    loss, _ = contrastive_loss(q, p, n, temperature=0.05)
    logits = q @ torch.cat([p, n]).T / 0.05
    assert torch.allclose(loss, F.cross_entropy(logits, torch.arange(4)))


def test_masked_loss_equals_infonce_without_those_candidates():
    q, p, n = unit_vectors(3), unit_vectors(3, seed=1), unit_vectors(3, seed=2)
    labels, negative_labels = ["x", "y", "x"], ["y", "z", "x"]
    loss, _ = contrastive_loss(q, p, n, 0.05, labels, negative_labels)

    candidates, candidate_labels = torch.cat([p, n]), labels + negative_labels
    per_query = []
    for i in range(3):
        keep = [c for c in range(6) if c == i or candidate_labels[c] != labels[i]]
        logits = (q[i] @ candidates[keep].T / 0.05)[None]
        per_query.append(F.cross_entropy(logits, torch.tensor([keep.index(i)])))
    assert torch.allclose(loss, torch.stack(per_query).mean())


def test_same_species_twice_is_no_longer_penalized():
    """Two rows with the same label and identical embeddings: plain InfoNCE is stuck at
    log 2 (it cannot tell the copies apart); masked InfoNCE goes to ~0."""
    q = unit_vectors(1).repeat(2, 1)
    p = q.clone()
    plain, _ = contrastive_loss(q, p, None, 0.02)
    masked, _ = contrastive_loss(q, p, None, 0.02, ["cardinal", "cardinal"], [])
    assert plain > 0.69 and masked < 1e-4


def test_bf16_inputs_are_scored_in_float32():
    q, p = unit_vectors(4).bfloat16(), unit_vectors(4, seed=1).bfloat16()
    loss, _ = contrastive_loss(q, p, None, 0.02)
    assert loss.dtype == torch.float32
