"""Runtime descriptors used while staging the dual-model worker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DualRuntime:
    name: str
    model: Any
    model_config: Any
    static_forward_context: Any = None
    attention_metadata_builder: Any = None
    kv_namespace: str = ""


def namespaced_layer_name(runtime_name: str, layer_name: str) -> str:
    return f"{runtime_name}::{layer_name}"
