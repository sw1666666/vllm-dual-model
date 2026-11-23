import pytest
import torch
from safetensors.torch import save_file

from vllm_dual.grpo_router import GrpoRouter


def test_policy_uses_current_hidden_and_per_request_prior_large_positions(tmp_path):
    embedding = torch.eye(2)
    save_file({"model.embed_tokens.weight": embedding}, tmp_path / "model.safetensors")
    state = {
        "backbone.model.embed_tokens.weight": embedding,
        "backbone.lm_head.weight": embedding,
        "usage_pos_emb.weight": torch.tensor([[3., 0.], [-3., 0.]]),
        "policy_head.weight": torch.tensor([[0., 0., 0., 0.], [1., 0., 1., 0.]]),
        "policy_head.bias": torch.zeros(2),
    }
    checkpoint = tmp_path / "router.pt"
    torch.save(state, checkpoint)
    router = GrpoRouter(checkpoint, tmp_path, 2, "cpu")
    hidden = torch.tensor([[1., 0.], [-1., 0.]])
    assert router.decide(["a", "b"], hidden) == [1, 0]
    router.commit("a", 0, 1)
    router.commit("b", 0, 0)
    both_negative = hidden[[1, 1]]
    assert router.decide(["b", "a"], both_negative) == [0, 1]
    router.commit("a", 2, 1)  # Outside the learned position table.
    assert router.decide(["a"], both_negative[:1]) == [1]
    router.discard("a")
    assert router.decide(["a"], both_negative[:1]) == [0]
    state["backbone.model.embed_tokens.weight"] = embedding + 1
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="backbone differs"):
        GrpoRouter(checkpoint, tmp_path, 2, "cpu")


@pytest.mark.parametrize(("threshold", "expected"), [
    (0.1, [1, 1, 1]), (0.4, [1, 1, 1]), (0.5, [0, 0, 1]),
    (0.6, [0, 0, 0]), (0.9, [0, 0, 0]),
])
def test_large_threshold(tmp_path, threshold, expected):
    embedding = torch.eye(1)
    save_file({"model.embed_tokens.weight": embedding}, tmp_path / "model.safetensors")
    checkpoint = tmp_path / "router.pt"
    torch.save({
        "backbone.model.embed_tokens.weight": embedding,
        "usage_pos_emb.weight": embedding,
        "policy_head.weight": torch.tensor([[0., 0.], [1., 0.]]),
        "policy_head.bias": torch.zeros(2),
    }, checkpoint)
    router = GrpoRouter(checkpoint, tmp_path, 1, "cpu", large_threshold=threshold)
    # P(large) ~= 0.45, 0.5, 0.55; the middle row exercises the old argmax tie.
    assert router.decide(["a", "b", "c"], torch.tensor([[-0.2], [0.0], [0.2]])) == expected
