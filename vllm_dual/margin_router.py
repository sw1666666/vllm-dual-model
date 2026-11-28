"""Probability-margin routing used by Margin V4."""
from .fused_margin import fused_device_actions


class MarginRouter:
    def __init__(self):
        self.committed = [0, 0]

    def decide(self, logits):
        return fused_device_actions(logits, 0.2)

    def commit(self, req_id, output_index, action):
        self.committed[action] += 1

    def discard(self, req_id):
        pass
