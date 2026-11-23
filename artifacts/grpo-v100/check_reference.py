"""Independently evaluate actual accepted prefixes with HF and the saved policy."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs = json.loads((args.out / "grpo.json").read_text())
    thresholds = {r.get("router_large_threshold", 0.5) for r in runs}
    if len(thresholds) != 1:
        raise ValueError("Expected one GRPO threshold for the run")
    threshold = thresholds.pop()
    requests = []
    for run in runs:
        if run["repeat"] != 0 or run["batch"] not in (1, 32):
            continue
        for index in ((0,) if run["batch"] == 1 else (0, 15, 31)):
            requests.append(run["requests"][index])

    checkpoint = torch.load(args.checkpoint, mmap=True, map_location="cpu", weights_only=True)
    position_weights = checkpoint["usage_pos_emb.weight"].to("cuda", torch.float32)
    weight = checkpoint["policy_head.weight"].to("cuda", torch.float32)
    bias = checkpoint["policy_head.bias"].to("cuda", torch.float32)
    results = []
    for action, size in enumerate(("0.6B", "8B")):
        model = AutoModelForCausalLM.from_pretrained(
            args.models / f"Qwen3-{size}", dtype=torch.float16, attn_implementation="eager"
        ).eval().cuda()
        for req in requests:
            ids = req["prompt"] + req["tokens"][:-1]
            tokens = torch.tensor([ids], device="cuda")
            hidden = model.model(input_ids=tokens, use_cache=False).last_hidden_state
            hidden = hidden[0, len(req["prompt"])-1:]
            used = torch.tensor(req["actions"], device="cuda")
            if action == 0:
                # Build the full strictly causal usage mask independently.
                prior = torch.tril(torch.ones(32, 32, device="cuda"), diagonal=-1)
                summed_usage = (used.float()[None, :] * prior) @ position_weights[:32]
                logits = torch.nn.functional.linear(
                    torch.cat([hidden.float(), summed_usage], dim=-1), weight, bias)
                large_probability = logits.double().softmax(dim=-1)[:, 1]
                correct = (large_probability > threshold) == used
                results.append(dict(request=req["id"], check="router", n=32,
                                    matched=int(correct.sum()), min_margin=float(
                                        (logits[:, 1] - logits[:, 0]).abs().min())))
            selected = used == action
            logits = model.lm_head(hidden[selected])
            ref = torch.tensor(req["tokens"], device="cuda")[selected]
            correct = logits.argmax(dim=-1) == ref
            results.append(dict(request=req["id"], check=f"token-{size}", n=ref.numel(),
                                matched=int(correct.sum())))
        del model
        torch.cuda.empty_cache()
    (args.out / "reference.json").write_text(json.dumps(results, indent=2) + "\n")
    for name in ("router", "token-0.6B", "token-8B"):
        group = [r for r in results if r["check"] == name]
        print(name, sum(r["matched"] for r in group), "/", sum(r["n"] for r in group))


if __name__ == "__main__":
    main()
