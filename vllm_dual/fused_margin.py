"""Top-two probability gap without materializing vocabulary-sized probabilities."""
import torch
import triton
import triton.language as tl


@triton.jit
def _tile_stats(X, Stats, stride_row, stride_col, vocab: tl.constexpr, tiles: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    col = tile * block + tl.arange(0, block)
    x = tl.load(X + row * stride_row + col * stride_col, mask=col < vocab, other=-float("inf")).to(tl.float32)
    high = tl.max(x, 0)
    tied = tl.sum((x == high).to(tl.int32), 0) > 1
    second = tl.where(tied, high, tl.max(tl.where(x < high, x, -float("inf")), 0))
    pivot = tl.where(high == -float("inf"), 0., high)
    z = tl.sum(tl.exp(x - pivot), 0)
    base = Stats + row * (3 * tiles) + tile
    tl.store(base, high)
    tl.store(base + tiles, second)
    tl.store(base + 2 * tiles, z)


@triton.jit
def _merge(Stats, Out, tiles: tl.constexpr, size: tl.constexpr):
    row = tl.program_id(0)
    idx = tl.arange(0, size)
    base = Stats + row * (3 * tiles) + idx
    highs = tl.load(base, mask=idx < tiles, other=-float("inf"))
    seconds = tl.load(base + tiles, mask=idx < tiles, other=-float("inf"))
    sums = tl.load(base + 2 * tiles, mask=idx < tiles, other=0.)
    high = tl.max(highs, 0)
    tied = tl.sum((highs == high).to(tl.int32), 0) > 1
    second = tl.where(tied, high, tl.max(tl.where(highs == high, seconds, highs), 0))
    z = tl.sum(sums * tl.exp(highs - high), 0)
    tl.store(Out + row, (1. - tl.exp(second - high)) / z)


def fused_margin(logits):
    rows, vocab = logits.shape
    block = 2048
    tiles = triton.cdiv(vocab, block)
    stats = torch.empty((rows, 3, tiles), device=logits.device, dtype=torch.float32)
    margin = torch.empty(rows, device=logits.device, dtype=torch.float32)
    _tile_stats[(rows, tiles)](logits, stats, *logits.stride(), vocab, tiles, block)
    _merge[(rows,)](stats, margin, tiles, triton.next_power_of_2(tiles))
    return margin


def fused_device_actions(logits, delta):
    return (fused_margin(logits) < delta).long()


def fused_actions(logits, delta):
    return fused_device_actions(logits, delta).tolist()
