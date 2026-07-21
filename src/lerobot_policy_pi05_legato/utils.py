"""
LEGATO weight-schedule utilities (PyTorch).

A LEGATO training example draws two integers and builds a per-position weight
curve over the action chunk:

    warmup  in [warmup_min, warmup_max]   -> number of leading positions kept
                                             fully anchored to the previous chunk
    ramp    in [ramp_min,   ramp_max]     -> length of the linear hand-over from
                                             anchored (w=0) to fresh (w=1)

    w[i] = 0.0                     for i < warmup            (previous chunk)
         = (i - warmup + 1) / ramp for warmup <= i < warmup + ramp   (ramp)
         = 1.0                     for i >= warmup + ramp    (freshly generated)

`ramp <= 1` collapses to the hard step used by the reference Kinetix sim
(`Legato-kinetix/src/model_legato.py`); a longer ramp is the smoothing the
paper recommends for real-robot deployment.

Both `warmup` and `ramp` are sampled per example (independently) from one of
three schemes over their inclusive integer range:
    "bell"    beta curve peaked at the low end (short values favoured)
    "exp"     exponential decay favouring smaller values
    "uniform" uniform over the range
"""

from __future__ import annotations

import torch

__all__ = [
    "sample_bell",
    "sample_exp",
    "sample_length",
    "build_weight_curve",
]


def sample_bell(
    low: int,
    high: int,
    size: tuple[int, ...] = (),
    peak: int | None = None,
    *,
    concentration: float = 3.0,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Sample integers in [low, high] from a beta ("bell") curve peaked at `peak`
    (default `low`). Returns a LongTensor of shape `size`.
    """
    range_size = high - low
    if range_size == 0:
        return torch.full(size, low, dtype=torch.long, device=device)

    if peak is None:
        peak = low
    normalized_peak = (peak - low) / max(range_size, 1)

    # asymmetric alpha/beta so the mode sits at `normalized_peak`
    if normalized_peak < 0.5:
        alpha = concentration * normalized_peak + 1.0
        beta = concentration * (1.0 - normalized_peak) + 1.0
    else:
        alpha = concentration * (1.0 - normalized_peak) + 1.0
        beta = concentration * normalized_peak + 1.0

    # torch.distributions.Beta ignores an external generator; sample via two
    # Gamma draws so `generator` (reproducibility) is respected.
    a = torch.full(size, float(alpha), device=device)
    b = torch.full(size, float(beta), device=device)
    ga = torch._standard_gamma(a, generator=generator)
    gb = torch._standard_gamma(b, generator=generator)
    beta_sample = ga / (ga + gb)

    continuous = low + beta_sample * (range_size + 1)
    return torch.floor(continuous).long().clamp_(low, high)


def sample_exp(
    low: int,
    high: int,
    size: tuple[int, ...] = (),
    *,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Sample integers in [low, high] with exponentially decaying probability
    (smaller values favoured). Returns a LongTensor of shape `size`.
    """
    range_size = high - low
    if range_size == 0:
        return torch.full(size, low, dtype=torch.long, device=device)

    candidates = torch.arange(low, high + 1, device=device)
    weights = torch.exp(-(candidates - low).float())
    probs = weights / weights.sum()

    n = 1
    for s in size:
        n *= s
    idx = torch.multinomial(probs, num_samples=n, replacement=True, generator=generator)
    return candidates[idx].reshape(size)


def sample_length(
    low: int,
    high: int,
    size: tuple[int, ...] = (),
    sampling: str = "bell",
    *,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Dispatch to the requested sampler ("bell" | "exp" | "uniform"). Generic
    integer-range sampler shared by both the warmup and ramp draws. Returns a
    LongTensor of shape `size` in [low, high].
    """
    if sampling == "bell":
        return sample_bell(low, high, size, device=device, generator=generator)
    if sampling == "exp":
        return sample_exp(low, high, size, device=device, generator=generator)
    if sampling == "uniform":
        return torch.randint(low, high + 1, size, device=device, generator=generator)
    raise ValueError(f"Unknown sampling: {sampling!r} (expected bell|exp|uniform)")


def build_weight_curve(
    action_chunk_size: int,
    warmup: torch.Tensor | int,
    ramp: torch.Tensor | int = 1,
    shape: str = "cosine",
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build the per-position weight curve. Let f = clip((i - warmup + 1)/ramp,
    0, 1) be the normalized ramp progress; then

        shape="step":   w = 1.0 if i >= warmup else 0.0   (ramp ignored)
        shape="linear": w = f
        shape="cosine": w = 0.5 * (1 - cos(pi * f))        (ease-in-out; the
                        previous-chunk weight (1 - w) is a cosine decay)

    So w is 0 for the first `warmup` positions, rises over the next `ramp`
    positions, then is 1. `ramp <= 1` collapses the linear/cosine forms to the
    hard step.

    `warmup` and `ramp` may be Python ints / 0-d tensors -> returns shape
    (action_chunk_size,); or broadcastable tensors of shape (B,) -> returns
    (B, action_chunk_size).
    """
    warmup = torch.as_tensor(warmup, device=device)
    ramp = torch.as_tensor(ramp, device=device)
    warmup, ramp = torch.broadcast_tensors(warmup, ramp)

    indices = torch.arange(action_chunk_size, device=warmup.device)
    pos = indices - warmup.unsqueeze(-1)  # (*S, chunk)

    if shape == "step":
        return (pos >= 0).to(dtype)

    r = ramp.clamp(min=1).unsqueeze(-1).to(dtype)  # avoid div-by-zero; <=1 -> step
    f = ((pos + 1).to(dtype) / r).clamp(0.0, 1.0)

    if shape == "linear":
        return f
    if shape == "cosine":
        return 0.5 * (1.0 - torch.cos(torch.pi * f))
    raise ValueError(f"Unknown weight_shape: {shape!r} (expected step|linear|cosine)")
