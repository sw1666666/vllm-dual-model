from vllm_dual.model_runner import namespaced_layer_name


def test_layer_namespaces_are_disjoint() -> None:
    layer = "model.layers.0.self_attn"
    assert namespaced_layer_name("small", layer) == f"small::{layer}"
    assert namespaced_layer_name("large", layer) == f"large::{layer}"
    assert namespaced_layer_name("small", layer) != namespaced_layer_name(
        "large", layer
    )
