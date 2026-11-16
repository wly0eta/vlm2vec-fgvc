"""Model, pooling and checkpoint tests against a tiny real Qwen2-VL (see conftest)."""
import pytest
import torch
from PIL import Image

from vlm2vec.checkpoint import EmbeddingContract, save_adapter
from vlm2vec.config import ModelConfig
from vlm2vec.data import ContrastiveCollator, Item, build_inputs, image_tokens_for, load_processor
from vlm2vec.modeling import VLM2VecModel, last_token_pool
from vlm2vec.trainer import contrastive_step

CPU = torch.device("cpu")


def test_last_token_pool_is_padding_side_agnostic():
    hidden = torch.arange(2 * 5 * 3, dtype=torch.float).reshape(2, 5, 3)
    right = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    left = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    assert torch.equal(last_token_pool(hidden, right)[0], hidden[0, 2])
    assert torch.equal(last_token_pool(hidden, left)[0], hidden[0, 4])


@pytest.fixture
def images(tmp_path):
    paths = []
    for i, color in enumerate([(200, 30, 30), (30, 200, 30), (30, 30, 200)]):
        path = tmp_path / f"img{i}.jpg"
        Image.new("RGB", (64 + 32 * i, 64), color).save(path)
        paths.append(str(path))
    return paths


def _trained_like(model):
    """Give the fresh LoRA non-zero B matrices, as training would."""
    torch.manual_seed(0)
    for name, p in model.backbone.named_parameters():
        if "lora_B" in name:
            p.data.normal_(0, 0.02)
    return model


def test_embeddings_identical_under_left_and_right_padding(tiny_model, images):
    model = VLM2VecModel.for_training(ModelConfig(model_name=tiny_model), torch.float32).eval()
    processor = load_processor(tiny_model, max_pixels=64 * 64)
    items = [Item("short", images[0], ""), Item("a much longer piece of query text here", images[1], "")]
    right = build_inputs(processor, items, max_len=512)
    processor.tokenizer.padding_side = "left"
    left = processor(text=[processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": i.prompt()}]}],
        tokenize=False) for i in items], images=[Image.open(p).convert("RGB") for p in images[:2]],
        padding=True, padding_side="left", return_tensors="pt")
    with torch.no_grad():
        assert torch.allclose(model.encode(right), model.encode(left), atol=1e-4)


def test_checkpoint_round_trip_reproduces_embeddings(tiny_model, images, tmp_path):
    model = _trained_like(VLM2VecModel.for_training(ModelConfig(model_name=tiny_model), torch.float32)).eval()
    processor = load_processor(tiny_model, max_pixels=64 * 64)
    save_adapter(model.backbone, processor, EmbeddingContract(base_model=tiny_model, max_pixels=64 * 64),
                 str(tmp_path / "run"))
    loaded, contract = VLM2VecModel.for_inference(str(tmp_path / "run"), dtype=torch.float32)
    assert contract.base_model == tiny_model and contract.max_pixels == 64 * 64
    batch = build_inputs(processor, [Item("q", images[0], ""), Item("label text")], max_len=512)
    with torch.no_grad():
        assert torch.allclose(model.encode(batch), loaded.encode(batch), atol=1e-5)


def test_no_op_adapter_is_refused(tiny_model, tmp_path):
    """An adapter whose LoRA-B is all zero (e.g. silently failed load) must raise, not
    quietly evaluate the base model."""
    model = VLM2VecModel.for_training(ModelConfig(model_name=tiny_model), torch.float32)   # B = 0 at init
    save_adapter(model.backbone, None, EmbeddingContract(base_model=tiny_model), str(tmp_path / "zero"))
    with pytest.raises(RuntimeError, match="all-zero"):
        VLM2VecModel.for_inference(str(tmp_path / "zero"), dtype=torch.float32)


def test_continuing_from_an_adapter_merges_it_first(tiny_model, images, tmp_path):
    """model_name = an adapter dir: the new run's frozen backbone must equal base+adapter."""
    stage1 = _trained_like(VLM2VecModel.for_training(ModelConfig(model_name=tiny_model), torch.float32)).eval()
    processor = load_processor(tiny_model, max_pixels=64 * 64)
    save_adapter(stage1.backbone, processor, EmbeddingContract(base_model=tiny_model), str(tmp_path / "stage1"))
    stage2 = VLM2VecModel.for_training(ModelConfig(model_name=str(tmp_path / "stage1")), torch.float32).eval()
    batch = build_inputs(processor, [Item("q", images[2], "")], max_len=512)
    with torch.no_grad():   # stage2's fresh LoRA is a no-op at init, so outputs must match stage1
        assert torch.allclose(stage1.encode(batch), stage2.encode(batch), atol=1e-4)


def test_contrastive_step_runs_on_real_model(tiny_model, images):
    model = VLM2VecModel.for_training(ModelConfig(model_name=tiny_model), torch.float32).train()
    processor = load_processor(tiny_model, max_pixels=64 * 64)
    rows = [{"query": {"text": "", "image": images[i]}, "instruction": "Identify the species in the image.",
             "positive": {"text": f"species {i}"}, "negatives": [{"text": f"species {(i + 1) % 3}"}]}
            for i in range(3)]
    batch = ContrastiveCollator(processor, chunk_size=2, max_len=512)(rows)
    assert [len(batch[s]) for s in ("query", "positive", "negative")] == [2, 2, 2]
    result = contrastive_step(model.encode, batch, batch["positive_keys"], batch["negative_keys"],
                              temperature=0.02, device=CPU)
    assert torch.isfinite(result.loss)
    grads = [p.grad for n, p in model.named_parameters() if "lora_" in n]
    assert grads and all(g is not None for g in grads)


def test_image_budget_is_applied(tiny_model):
    assert image_tokens_for(load_processor(tiny_model, max_pixels=100352)) < \
           image_tokens_for(load_processor(tiny_model, max_pixels=200704))


def test_auto_attention_falls_back_to_sdpa_without_cuda_or_16bit():
    from vlm2vec.modeling import resolve_attn_implementation
    assert resolve_attn_implementation("auto", torch.float32) == "sdpa"
    assert resolve_attn_implementation("eager", torch.bfloat16) == "eager"
