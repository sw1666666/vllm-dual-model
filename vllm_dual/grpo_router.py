"""Inference head for a GRPO switch policy with a frozen small backbone."""

import math
from pathlib import Path

import torch
from safetensors import safe_open
from torch.nn import functional as F

from .config import DEFAULT_LARGE_THRESHOLD, validate_large_threshold


def verify_backbone(state, model_dir):
    """The worker shares hidden states only if all backbone weights match in FP16."""
    expected = {k.removeprefix("backbone."): v for k, v in state.items()
                if k.startswith("backbone.")}
    if not expected:
        raise ValueError("GRPO checkpoint has no frozen backbone")
    # Tied LM heads are sometimes omitted by HF serialization.
    if "lm_head.weight" in expected and torch.equal(
        expected["lm_head.weight"], expected["model.embed_tokens.weight"]
    ):
        del expected["lm_head.weight"]
    for path in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as weights:
            for name in weights.keys():
                if name == "lm_head.weight" and name not in expected:
                    continue
                reference = expected.pop(name, None)
                if reference is None or not torch.equal(
                    weights.get_tensor(name).half(), reference.half()
                ):
                    raise ValueError(f"GRPO frozen backbone differs at {name}")
    if expected:
        raise ValueError(f"Small model is missing GRPO weights: {list(expected)[:3]}")


class GrpoRouter:
    def __init__(self, checkpoint, model_dir, hidden_size, device, *, large_threshold=DEFAULT_LARGE_THRESHOLD):
        validate_large_threshold(large_threshold)
        # P(large) > t iff large_logit - small_logit > log(t / (1-t)).
        self.log_odds_threshold = math.log(large_threshold) - math.log1p(-large_threshold)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        verify_backbone(state, model_dir)
        self.positions = state["usage_pos_emb.weight"].to(device, torch.float32)
        self.weight = state["policy_head.weight"].to(device, torch.float32)
        self.bias = state["policy_head.bias"].to(device, torch.float32)
        if (self.positions.ndim != 2 or self.positions.shape[1] != hidden_size
                or self.weight.shape != (2, 2 * hidden_size) or self.bias.shape != (2,)):
            raise ValueError("GRPO policy shapes do not match the small hidden size")
        self.zero = self.positions.new_zeros(hidden_size)
        self.usage = {}

    def decide(self, req_ids, hidden):
        return self.decide_device(req_ids, hidden).tolist()

    def decide_device(self, req_ids, hidden):
        """One real route per row, on the same device as the small hidden states."""
        if len(req_ids) != hidden.shape[0]:
            raise ValueError("GRPO needs one last-token hidden state per request")
        usage = torch.stack([self.usage.get(rid, self.zero) for rid in req_ids])
        features = torch.cat((hidden.float(), usage), dim=1)
        logits = F.linear(features, self.weight, self.bias)
        if self.log_odds_threshold == 0:
            return logits.argmax(dim=1)
        large = logits[:, 1] - logits[:, 0] > self.log_odds_threshold
        return large.long()

    def commit(self, req_id, output_index, route):
        if route == 1 and output_index < len(self.positions):
            self.usage[req_id] = self.usage.get(req_id, self.zero) + self.positions[output_index]

    def discard(self, req_id):
        self.usage.pop(req_id, None)
