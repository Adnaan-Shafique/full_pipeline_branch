"""Will this model fit on this card?

One arithmetic question, given its own module because it is the one that
decides whether a model can be benchmarked at all — and because the answer for
Molmo-72B is "no", which is not obvious until you do the multiplication:

    72B parameters x 2 bytes (bf16) = 144 GB of weights
    H200 NVL usable                 = 141 GB

Over the card before a single byte of KV cache. Prose in a runbook goes stale
when someone swaps a GPU or adds a model; this does not.

Deliberately approximate. It models weights only, which is the term that
decides feasibility, and reports the KV headroom that is left over. It does not
model activation memory, CUDA graphs, the vision tower's working set, or
fragmentation — so a model that "fits" here with a few GB to spare may still
OOM in practice. Treat a pass as "worth trying" and the VRAM reading from
`/health` after loading as the real answer.

Pure stdlib.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Bytes per parameter, by dtype.
BYTES_PER_PARAM = {
    "bf16": 2.0, "fp16": 2.0, "float16": 2.0, "bfloat16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "awq": 0.5, "gptq": 0.5, "int4": 0.5,
}

# H200 NVL, per gpu_api_server_v6's own header.
H200_NVL_GB = 141.0

# Below this much KV cache a vision model will hold barely one request, so the
# concurrency numbers a benchmark produces describe the cache, not the model.
MIN_USEFUL_KV_GB = 8.0


@dataclass
class Fit:
    model: str
    params_b: float
    dtype: str
    weights_gb: float
    per_gpu_budget_gb: float
    kv_gb: float
    tensor_parallel: int
    fits: bool
    reason: str

    @property
    def usable(self) -> bool:
        """Fits AND has enough KV left to be worth measuring."""
        return self.fits and self.kv_gb >= MIN_USEFUL_KV_GB


def fit_on_gpu(model: str, params_b: float, dtype: str = "bf16",
               gpu_gb: float = H200_NVL_GB, gpu_memory_utilization: float = 0.85,
               tensor_parallel: int = 1,
               already_resident_gb: float = 0.0) -> Fit:
    """Can `model` load, and what is left for KV cache?

    `already_resident_gb` is anything else holding VRAM on the same card -
    mistral's ~21 GB at gpu_memory_utilization 0.15, typically. It is subtracted
    from the budget rather than ignored, because on this box that is the
    difference between a 38B model loading and not.
    """
    per_param = BYTES_PER_PARAM.get(dtype.lower())
    if per_param is None:
        raise ValueError(f"unknown dtype {dtype!r}; known: {sorted(BYTES_PER_PARAM)}")

    weights_gb = params_b * per_param                       # 1e9 params x bytes = GB
    shard_gb = weights_gb / max(1, tensor_parallel)
    budget_gb = gpu_gb * gpu_memory_utilization - already_resident_gb
    kv_gb = budget_gb - shard_gb

    if shard_gb >= gpu_gb:
        reason = (f"weights alone are {shard_gb:.0f} GB per GPU against a "
                  f"{gpu_gb:.0f} GB card - it cannot load at any "
                  f"gpu_memory_utilization. Quantise it, or raise "
                  f"tensor_parallel_size")
        fits = False
    elif kv_gb <= 0:
        reason = (f"weights are {shard_gb:.0f} GB but the budget is only "
                  f"{budget_gb:.0f} GB - raise gpu_memory_utilization, unload "
                  f"what else is resident ({already_resident_gb:.0f} GB), or "
                  f"shard wider")
        fits = False
    elif kv_gb < MIN_USEFUL_KV_GB:
        reason = (f"loads with only {kv_gb:.0f} GB of KV cache - it will hold "
                  f"barely one request, so any concurrency figure describes the "
                  f"cache rather than the model")
        fits = True
    else:
        reason = f"{shard_gb:.0f} GB of weights, {kv_gb:.0f} GB left for KV cache"
        fits = True

    return Fit(model=model, params_b=params_b, dtype=dtype, weights_gb=weights_gb,
               per_gpu_budget_gb=budget_gb, kv_gb=kv_gb,
               tensor_parallel=tensor_parallel, fits=fits, reason=reason)


def smallest_working_dtype(model: str, params_b: float, **kw) -> Optional[str]:
    """The least aggressive quantisation that makes this model usable, or None.

    Ordered best-quality first: a benchmark should quantise only as far as it
    must, because every step down is a different numerical path and stops the
    model's answer quality being comparable with the ones that did not.
    """
    for dtype in ("bf16", "fp8", "int8", "awq"):
        if fit_on_gpu(model, params_b, dtype=dtype, **kw).usable:
            return dtype
    return None
