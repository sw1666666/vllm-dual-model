#!/usr/bin/env python3
"""Transformers reference for sync, lazy, and replay dual-model decoding."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from vllm_dual.router import LARGE, ROUTE_NAMES, SMALL, DeterministicRouter
from vllm_dual.state import DualRequestState, StepPlan
from vllm_dual.trace import JsonlTraceWriter


@dataclass
class Runtime:
    name: str
    model: Any
    past_key_values: Any = None

    @torch.inference_mode()
    def cached_forward(self, token_ids: list[int], start: int, end: int) -> torch.Tensor:
        if not 0 <= start < end <= len(token_ids):
            raise AssertionError(f"invalid {self.name} input slice [{start}:{end}]")
        inputs = torch.tensor([token_ids[start:end]], device="cuda", dtype=torch.long)
        output = self.model(
            input_ids=inputs,
            past_key_values=self.past_key_values,
            use_cache=True,
        )
        self.past_key_values = output.past_key_values
        return output.logits[0, -1].float()

    @torch.inference_mode()
    def replay_forward(self, token_ids: list[int]) -> torch.Tensor:
        inputs = torch.tensor([token_ids], device="cuda", dtype=torch.long)
        return self.model(input_ids=inputs, use_cache=False).logits[0, -1].float()


def parse_forced_routes(value: str | None) -> list[int] | None:
    if value is None:
        return None
    routes = []
    for item in value.replace(",", " ").split():
        normalized = item.strip().lower()
        if normalized in {"s", "small", "0"}:
            routes.append(SMALL)
        elif normalized in {"l", "large", "1"}:
            routes.append(LARGE)
        else:
            raise ValueError(f"unknown route value: {item}")
    return routes


def assert_compatible_models(small_model: str, large_model: str) -> Any:
    small_config = AutoConfig.from_pretrained(small_model)
    large_config = AutoConfig.from_pretrained(large_model)
    for field in ("vocab_size", "bos_token_id", "eos_token_id"):
        small_value = getattr(small_config, field)
        large_value = getattr(large_config, field)
        if small_value != large_value:
            raise ValueError(f"incompatible {field}: {small_value} != {large_value}")
    small_tokenizer = AutoTokenizer.from_pretrained(small_model)
    large_tokenizer = AutoTokenizer.from_pretrained(large_model)
    if small_tokenizer.get_vocab() != large_tokenizer.get_vocab():
        raise ValueError("small and large tokenizers have different vocabularies")
    return small_tokenizer


def load_runtime(name: str, model_path: str) -> Runtime:
    print(f"loading {name}: {model_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()
    return Runtime(name=name, model=model)


def greedy_argmax(logits: torch.Tensor) -> int:
    if not torch.isfinite(logits).all():
        raise AssertionError("logits contain NaN or Inf")
    return int(torch.argmax(logits).item())


def trace_row(
    *,
    request_id: str,
    plan: StepPlan,
    state: DualRequestState,
    small_argmax: int | None,
    large_argmax: int | None,
    accepted_token: int,
) -> dict[str, Any]:
    selected_argmax = large_argmax if plan.route == LARGE else small_argmax
    assert selected_argmax is not None
    assert accepted_token == selected_argmax
    return {
        "request_id": request_id,
        "output_index": state.output_index,
        "route": ROUTE_NAMES[plan.route],
        "history_len_before": plan.history_len_before,
        "small_before": plan.small_before,
        "small_delta": plan.small_delta,
        "small_after": state.small_num_computed_tokens,
        "large_before": plan.large_before,
        "large_delta": plan.large_delta,
        "large_after": state.large_num_computed_tokens,
        "small_argmax": small_argmax,
        "large_argmax": large_argmax,
        "selected_argmax": selected_argmax,
        "accepted_token": accepted_token,
    }


def generate(
    *,
    small: Runtime,
    large: Runtime,
    prompt_token_ids: list[int],
    request_id: str,
    max_tokens: int,
    mode: str,
    router: DeterministicRouter,
    trace_writer: JsonlTraceWriter,
) -> tuple[list[int], list[dict[str, Any]]]:
    public_token_ids = list(prompt_token_ids)
    state = DualRequestState()
    rows: list[dict[str, Any]] = []

    for _ in range(max_tokens):
        route = router.route(request_id, state.output_index)
        history_len = len(public_token_ids)
        sync_large = mode == "sync"
        plan = state.plan(history_len, route, sync_large=sync_large)
        small_logits = large_logits = None

        if mode in {"sync", "lazy"}:
            small_logits = small.cached_forward(
                public_token_ids, plan.small_before, history_len
            )
            if sync_large or route == LARGE:
                large_logits = large.cached_forward(
                    public_token_ids, plan.large_before, history_len
                )
        elif mode == "replay":
            if route == SMALL:
                small_logits = small.replay_forward(public_token_ids)
            else:
                large_logits = large.replay_forward(public_token_ids)
        else:
            raise ValueError(f"unknown mode: {mode}")

        state.commit_forwards(plan, sync_large=sync_large)
        small_argmax = greedy_argmax(small_logits) if small_logits is not None else None
        large_argmax = greedy_argmax(large_logits) if large_logits is not None else None
        selected_logits = large_logits if route == LARGE else small_logits
        assert selected_logits is not None
        accepted_token = greedy_argmax(selected_logits)
        row = trace_row(
            request_id=request_id,
            plan=plan,
            state=state,
            small_argmax=small_argmax,
            large_argmax=large_argmax,
            accepted_token=accepted_token,
        )
        rows.append(row)
        trace_writer.write(row)

        public_token_ids.append(accepted_token)
        state.commit_sample()

    return public_token_ids[len(prompt_token_ids) :], rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--small-model", required=True)
    parser.add_argument("--large-model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--request-id", default="r0")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--mode", choices=("sync", "lazy", "replay"), required=True)
    parser.add_argument("--router-seed", type=int, default=17)
    parser.add_argument("--p-large", type=float, default=0.10)
    parser.add_argument("--forced-routes")
    parser.add_argument("--trace-path", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    forced_routes = parse_forced_routes(args.forced_routes)
    if forced_routes is not None and len(forced_routes) < args.max_tokens:
        raise ValueError("forced route list must cover every output position")
    tokenizer = assert_compatible_models(args.small_model, args.large_model)
    prompt_token_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    small = load_runtime("small", args.small_model)
    large = load_runtime("large", args.large_model)
    router = DeterministicRouter(
        seed=args.router_seed,
        p_large=args.p_large,
        forced_routes=forced_routes,
    )
    output_token_ids, rows = generate(
        small=small,
        large=large,
        prompt_token_ids=prompt_token_ids,
        request_id=args.request_id,
        max_tokens=args.max_tokens,
        mode=args.mode,
        router=router,
        trace_writer=JsonlTraceWriter(args.trace_path),
    )
    result = {
        "mode": args.mode,
        "small_model": args.small_model,
        "large_model": args.large_model,
        "request_id": args.request_id,
        "prompt": args.prompt,
        "prompt_token_ids": prompt_token_ids,
        "output_token_ids": output_token_ids,
        "text": tokenizer.decode(output_token_ids),
        "routes": [row["route"] for row in rows],
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
