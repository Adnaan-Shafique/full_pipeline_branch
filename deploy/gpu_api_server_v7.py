from __future__ import annotations
# gpu_api_server_v7.py
#
# CHANGES FROM v6 — all of them driven by benchmarking four VLMs on ONE GPU at
# a time. See BENCHMARK_RUNBOOK.md in the pipeline repo.
#
#   [A] TWO MORE VLMs: pixtral (12B) and molmo (72B), with the prompt builders
#       they need. v6 had neither in MODEL_CONFIGS, and _build_engine_input
#       knew only three prompt styles, so no amount of /models/x/load would
#       have worked.
#
#   [B] MOLMO IS FP8, NOT bf16, AND NOT BY PREFERENCE. 72B x 2 bytes = 144 GB
#       of weights against a 141 GB H200 — over the card before a single byte
#       of KV cache, at any gpu_memory_utilization. FP8 halves it to ~72 GB and
#       leaves ~55 GB of KV at 0.90. Consequence to carry into any report: FP8
#       is a different numerical path, so molmo's answer quality is not
#       strictly comparable with the bf16 models, and its latency is flattered.
#
#   [C] PER-MODEL ENV OVERRIDES for tensor_parallel_size, gpu_memory_utilization,
#       max_model_len and max_concurrent. Benchmarking needs internvl at TP=1
#       and qwen3-vl at a different memory budget; production needs them back.
#       In v6 that was editing this file twice and hoping the revert was clean.
#       Now it is VLM_TP_INTERNVL=1 in the benchmark shell and nothing at all
#       afterwards — the file on disk is always the production configuration.
#
#   [D] EAGER_LOAD FROM THE ENVIRONMENT. Same reason: a benchmark wants an
#       empty GPU at boot, production wants qwen3-vl and mistral resident.
#
#   [E] SERVER-SIDE QUEUE WAIT. v6 timed inference from AFTER the semaphore was
#       acquired, so the wait for a slot was invisible unless you went through
#       llm_proxy_v3 and subtracted. With max_concurrent=2 on the VLMs that
#       wait IS the saturation signal, so /infer now measures and returns it as
#       queue_wait_s — on the direct path too.
#
#   [F] /metrics/reset, because /metrics is cumulative and a benchmark wants
#       per-run counters rather than everything since the process started.
#
#   [G] BUG FIX in /debug/prompt: its else-branch called the InternVL builder
#       for ANY non-qwen model, so a pixtral request would have been rendered
#       with InternVL's template and looked fine. That endpoint is the one
#       thing standing between a wrong template and a run of confident
#       garbage, so it now dispatches explicitly and refuses an unknown style.
#
# ─────────────────────────────────────────────────────────────────────────────
#
# gpu_api_server_v6.py
#
# 2x NVIDIA H200 NVL (~141 GB usable each = ~282 GB total)
#
# KEY CHANGES FROM v5:
#   [1] REMOVED: defog (sqlcoder-7b-2) and nvidia-ai (Nemotron Mamba hybrid).
#       ("cadastral" was never a registry entry in v5 — nothing to remove.)
#       The mamba-ssm / causal-conv1d dependency is no longer needed.
#   [2] ADDED two vision-language models (VLMs), both lazy:
#         qwen3-vl  → Qwen/Qwen3-VL-30B-A3B-Instruct  (/data01/llm_models/Qwen_VLM/)
#         internvl  → OpenGVLab/InternVL3_5-38B       (/data01/llm_models/InternVL/)
#   [3] NEW multimodal request path: InferRequest now accepts `images`
#       (http(s) URL, data: URI, base64 blob, or local file path). Images are
#       decoded to PIL and passed to vLLM as `multi_modal_data`.
#   [4] NEW per-model prompt builders. VLMs will NOT work with a raw prompt
#       string — each family needs its own chat template + image placeholder:
#         Qwen3-VL : AutoProcessor.apply_chat_template →
#                    <|vision_start|><|image_pad|><|vision_end|>
#         InternVL : AutoTokenizer.apply_chat_template with a literal
#                    "<image>\n" prefix per image (vLLM expands it to IMG_CONTEXT)
#   [5] NEW eviction groups (EVICT_GROUPS). The two VLMs are in the same group,
#       so loading one automatically unloads the other. Unlike EXCLUSIVE_MODELS
#       this does NOT evict mistral, which stays resident for Falcon.
#   [6] RE-BUDGETED VRAM. This matters — read it. In v5 mistral held
#       gpu_memory_utilization=0.80, which left ~28 GB free on GPU 0 and made a
#       30B/38B VLM impossible to load. mistral is now 0.15 (~21 GB, still far
#       more KV cache than a 7B at 16k context can use).
#
# VRAM allocation plan (~141 GB per GPU):
#   mistral    7B   TP=1   0.15   ~21 GB          ← eager, primary Falcon model
#   qwen3     32B   TP=2   0.30   ~42 GB / GPU    ← lazy, text
#   codestral 22B   TP=1   0.40   ~56 GB          ← lazy, code
#   qwen3-vl  30B   TP=2   0.35   ~49 GB / GPU    ← lazy, VLM  [group "vlm"]
#   internvl  38B   TP=2   0.40   ~56 GB / GPU    ← lazy, VLM  [group "vlm"]
#
#   Note on placement: in a single process, a TP=1 engine always lands on GPU 0.
#   Both VLMs are therefore TP=2 so they use the full 282 GB pool instead of
#   fighting mistral for GPU 0. InternVL3.5-38B is ~76 GB in bf16 and officially
#   needs 2x A100 — TP=2 is the right call regardless. Qwen3-VL-30B-A3B is a MoE
#   (~61 GB bf16, 3B active) and would fit TP=1, but only if GPU 0 is otherwise
#   empty; TP=2 keeps it predictable.
#
# Start (single process, all models):
#   uvicorn gpu_api_server_v7:app --host 0.0.0.0 --port 5432 --workers 1
#
# Required installs:
#   pip install "vllm>=0.11.0" fastapi "uvicorn[standard]" pydantic pillow
#   pip install "transformers>=4.57.0"     # Qwen3-VL support
#   pip install qwen-vl-utils==0.0.14      # optional; only needed for video
#
#   Qwen3-VL needs vLLM >= 0.11.0. InternVL3.5 needs transformers >= 4.52.1 and
#   trust_remote_code=True (the OpenGVLab repo ships custom modelling code).
#
# Quick multimodal smoke test:
#   curl -s localhost:5432/infer -H 'Content-Type: application/json' -d '{
#     "model": "qwen3-vl",
#     "prompt": "Describe this image in one sentence.",
#     "images": ["/data01/samples/plot.png"],
#     "max_new_tokens": 256, "temperature": 0.2
#   }' | jq .
#
# NOTE on TP + multiprocessing: the Qwen3-VL card sets
# VLLM_WORKER_MULTIPROC_METHOD=spawn. This server has been running TP=2 with the
# default fork method, so it is NOT set here — changing it would affect the
# already-working qwen3 engine too. If a VLM worker dies during load with a CUDA
# re-init error, export VLLM_WORKER_MULTIPROC_METHOD=spawn in the tmux pane
# before starting uvicorn.
#

import asyncio
import base64
import binascii
import gc
import io
import logging
import os
import time
import urllib.request
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ─────────────────────────────── Logging ────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("gpu_api")

# ─────────────────────────────── Constants ───────────────────────────────────

# How long a request will wait in the semaphore queue before giving up
QUEUE_TIMEOUT_S = 60.0

# Models loaded eagerly at startup (everything else is lazy).
#
# [D] Read from the environment so a benchmark window does not need this file
# edited and then reverted. The DEFAULT is the production configuration, which
# means a plain restart always comes back up the way it should:
#
#     EAGER_LOAD=""            # benchmark: boot with an empty GPU
#     (unset)                  # production: qwen3-vl + mistral, as before
EAGER_LOAD = [
    m.strip() for m in os.environ.get("EAGER_LOAD", "qwen3-vl,mistral").split(",")
    if m.strip()
]

# Models that demand the whole box — loading one unloads *everything* else.
# Nothing qualifies right now; the VLMs use the softer EVICT_GROUPS below.
EXCLUSIVE_MODELS: set[str] = set()

# Softer mutual exclusion: models sharing a group evict each other on load,
# but leave models outside the group (e.g. mistral) alone.
# Both VLMs are TP=2 and large — only one can be resident at a time.
# [A] All four VLMs share the group. This is not optional: on a one-GPU plan a
# VLM that does not evict its siblings means the second load OOMs, and the
# error arrives from CUDA rather than from anything that names the cause.
EVICT_GROUPS: dict[str, str] = {
    "qwen3-vl": "vlm",
    "internvl": "vlm",
    "pixtral":  "vlm",
    "molmo":    "vlm",
}

# Hard ceiling on decoded image size before it reaches the vision tower.
# Oversized images blow up the vision token count (and therefore KV cache),
# which is the #1 cause of OOM on VLM serving.
MAX_IMAGE_SIDE_PX = 2048
MAX_IMAGES_PER_REQUEST = 8
IMAGE_FETCH_TIMEOUT_S = 20

# ─────────────────────────────── Model registry ──────────────────────────────
#
#  modality               – "text" | "vision"  (vision models accept `images`)
#  prompt_style           – how the chat prompt is assembled:
#                             "raw"      → prompt string passed through untouched
#                             "qwen_vl"  → AutoProcessor chat template (Qwen3-VL)
#                             "internvl" → AutoTokenizer chat template + <image>
#  tensor_parallel_size   – GPUs to shard across (2 = both H200s)
#  gpu_memory_utilization – fraction of each GPU's VRAM vLLM may use
#  quantization           – None | "compressed-tensors" (W4A16 Neural Magic)
#  max_model_len          – context window cap (tokens); governs KV cache budget
#  max_concurrent         – semaphore depth: parallel generations allowed
#  limit_mm_per_prompt    – vision only: caps mm items, preallocates encoder mem
#  mm_processor_kwargs    – vision only: shrinks the processed image
#  stop_token_ids         – extra EOS ids appended to every SamplingParams
#  lazy                   – if True, not loaded at startup

MODEL_CONFIGS: dict[str, dict] = {

    # ── Primary Falcon model — always resident, GPU 0 ────────────────────────
    "mistral": {
        "path":                   "/data01/llm_models/Mistral-7B-Instruct",
        "modality":               "text",
        "prompt_style":           "raw",
        "dtype":                  "bfloat16",
        "quantization":           None,
        "tensor_parallel_size":   1,
        "gpu_memory_utilization": 0.15,   # ~21 GB — was 0.80 in v5; see header
        "max_model_len":          16000,  # Falcon prompts need the room
        "trust_remote_code":      False,
        "max_concurrent":         4,
        "lazy":                   False,  # eager load
    },

    # ── Large text model — TP=2 ──────────────────────────────────────────────
    "qwen3": {
        "path":                   "/data01/llm_models/Qwen3-32B/Qwen3-32B",
        "modality":               "text",
        "prompt_style":           "raw",
        "dtype":                  "bfloat16",
        "quantization":           None,
        "tensor_parallel_size":   2,
        "gpu_memory_utilization": 0.30,   # ~42 GB per GPU
        "max_model_len":          32768,
        "trust_remote_code":      True,
        "max_concurrent":         2,
        "lazy":                   True,
    },

    # ── Long-context code model — TP=1, GPU 0 ────────────────────────────────
    "codestral": {
        "path":                   "/data01/llm_models/CodeStral/CodeStral/",
        "modality":               "text",
        "prompt_style":           "raw",
        "dtype":                  "bfloat16",
        "quantization":           None,
        "tensor_parallel_size":   1,
        "gpu_memory_utilization": 0.40,   # ~56 GB on GPU 0 (mistral only takes 21)
        "max_model_len":          32768,
        "trust_remote_code":      True,
        "max_concurrent":         2,
        "lazy":                   True,
    },

    # ── VLM #1 — Qwen3-VL-30B-A3B-Instruct (MoE, 3B active) ──────────────────
    # Requires vLLM >= 0.11.0 and transformers >= 4.57.
    # Native context is 256K; capped here so the KV cache stays sane while
    # sharing the box. Raise max_model_len only if you also raise the budget.
    "qwen3-vl": {
        "path":                   "/data01/llm_models/Qwen_VLM/",
        "modality":               "vision",
        "prompt_style":           "qwen_vl",
        "dtype":                  "bfloat16",
        "quantization":           None,
        "tensor_parallel_size":   1,
        "gpu_memory_utilization": 0.60,   # ~49 GB per GPU (~98 GB total)
        "max_model_len":          32768,
        "trust_remote_code":      True,
        "max_concurrent":         2,
        "limit_mm_per_prompt":    {"image": 4, "video": 0},
        "mm_processor_kwargs":    {
            # Default max is 1280*28*28. Lowering it is the cheapest way to cut
            # vision tokens per image; raise if you need fine OCR detail.
            "min_pixels": 4 * 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        },
        "stop_token_ids":         [],
        "lazy":                   True,
    },

    # ── VLM #2 — InternVL3_5-38B (dense) ─────────────────────────────────────
    # OpenGVLab ships custom modelling code → trust_remote_code is mandatory.
    # 38B bf16 is ~76 GB of weights; the model card states 2 GPUs, hence TP=2.
    # max_dynamic_patch is the InternVL equivalent of Qwen's max_pixels: each
    # patch is a full 448x448 tile through the vision tower, so 12 (the default)
    # can mean 3k+ vision tokens for one image.
    "internvl": {
        "path":                   "/data01/llm_models/InternVL/",
        "modality":               "vision",
        "prompt_style":           "internvl",
        "dtype":                  "bfloat16",
        "quantization":           None,
        "tensor_parallel_size":   2,
        "gpu_memory_utilization": 0.40,   # ~56 GB per GPU (~112 GB total)
        "max_model_len":          16384,
        "trust_remote_code":      True,
        "max_concurrent":         2,
        "limit_mm_per_prompt":    {"image": 4, "video": 0},
        "mm_processor_kwargs":    {"max_dynamic_patch": 6},   # default 12
        # InternVL's chat template does not always emit a clean stop; these are
        # the ids the OpenGVLab / vLLM examples use.
        "stop_strings":           ["<|im_end|>", "<|endoftext|>", "<|end|>"],
        "stop_token_ids":         [],
        "lazy":                   True,
    },

    # ── VLM #3 — Pixtral-12B ─────────────────────────────────────────────────
    # 12B bf16 is ~24 GB of weights: the one VLM here that would still fit
    # alongside mistral if it had to. TP=1 because the benchmark compares every
    # model on ONE GPU; gpu_memory_utilization assumes mistral is unloaded, so
    # subtract its 0.15 if it is not.
    "pixtral": {
        "path":                   "/data01/llm_models/Pixtral_12B/",
        "modality":               "vision",
        "prompt_style":           "pixtral",
        "dtype":                  "bfloat16",
        "quantization":           None,
        "tensor_parallel_size":   1,
        "gpu_memory_utilization": 0.85,   # ~120 GB: 24 for weights, rest KV
        "max_model_len":          16384,
        "trust_remote_code":      True,
        "max_concurrent":         2,
        "limit_mm_per_prompt":    {"image": 4, "video": 0},
        "stop_token_ids":         [],
        "lazy":                   True,
    },

    # ── VLM #4 — Molmo-72B ───────────────────────────────────────────────────
    # [B] FP8, and not by preference. 72B x 2 bytes is 144 GB of weights
    # against a 141 GB card: bf16 CANNOT be loaded here at any
    # gpu_memory_utilization. FP8 halves the weights to ~72 GB and leaves ~55 GB
    # of KV cache at 0.90.
    #
    # What that costs, and what belongs in any report built on it: FP8 is a
    # different numerical path from the bf16 models, so molmo's ANSWER QUALITY
    # is not strictly comparable with theirs, and its LATENCY is flattered
    # because FP8 arithmetic is faster.
    #
    # If this vLLM build rejects quantization="fp8", the remaining options are a
    # pre-quantised AWQ/GPTQ checkpoint (~40 GB) or tensor_parallel_size=2 —
    # which uses both H200s and leaves the one-GPU comparison behind.
    "molmo": {
        "path":                   "/data01/llm_models/Molmo_72B/",
        "modality":               "vision",
        "prompt_style":           "molmo",
        "dtype":                  "bfloat16",
        "quantization":           "fp8",   # REQUIRED — see above
        "tensor_parallel_size":   1,
        "gpu_memory_utilization": 0.90,
        "max_model_len":          8192,    # KV is the tight resource at this size
        "trust_remote_code":      True,    # allenai ships custom modelling code
        "max_concurrent":         1,       # 72B on one card; do not oversubscribe
        "limit_mm_per_prompt":    {"image": 1, "video": 0},
        "stop_token_ids":         [],
        "lazy":                   True,
    },
}


# ─────────────────────────────── Per-model env overrides ─────────────────────
#
# [C] Benchmarking needs internvl at TP=1 (it ships TP=2) and a different memory
# budget for qwen3-vl; production needs both back exactly as they were. Doing
# that by editing MODEL_CONFIGS means editing this file twice and trusting the
# revert — and the revert is the step people skip.
#
# So the table above is ALWAYS the production configuration, and a benchmark
# window overrides it from the shell:
#
#     export VLM_TP_INTERNVL=1  VLM_GMU_INTERNVL=0.85
#     export VLM_GMU_QWEN3_VL=0.85
#     export EAGER_LOAD=""
#
# Unset them and restart to restore. The override is logged on every load, so a
# forgotten variable shows up in the tmux pane rather than silently changing
# what a later benchmark measures.
#
# Name shape: VLM_<FIELD>_<MODEL>, model name upper-cased with - and . as _.

_ENV_OVERRIDABLE = {
    "tensor_parallel_size":   ("VLM_TP",  int),
    "gpu_memory_utilization": ("VLM_GMU", float),
    "max_model_len":          ("VLM_LEN", int),
    "max_concurrent":         ("VLM_CONC", int),
    "quantization":           ("VLM_QUANT", str),
}


def _env_key(prefix: str, model: str) -> str:
    return f"{prefix}_{model.upper().replace('-', '_').replace('.', '_')}"


def _effective_config(name: str) -> dict:
    """MODEL_CONFIGS[name] with any VLM_* environment overrides applied.

    Returns a COPY: the table itself is never mutated, so unsetting a variable
    and restarting genuinely restores the original rather than leaving the
    process holding an edited dict.
    """
    cfg = dict(MODEL_CONFIGS[name])
    for field, (prefix, cast) in _ENV_OVERRIDABLE.items():
        raw = os.environ.get(_env_key(prefix, name))
        if raw is None or not raw.strip():
            continue
        try:
            # "none" is how you switch quantization OFF from the shell, which
            # an empty string cannot express — empty means "not set".
            if cast is str and raw.strip().lower() in ("none", "null"):
                value = None
            else:
                value = cast(raw.strip())
        except (TypeError, ValueError):
            log.warning("ignoring %s=%r — not a valid %s",
                        _env_key(prefix, name), raw, cast.__name__)
            continue
        if cfg.get(field) != value:
            log.warning("'%s': %s overridden by environment: %s -> %s",
                        name, field, cfg.get(field), value)
        cfg[field] = value
    return cfg


# ─────────────────────────────── Runtime state ───────────────────────────────

# AsyncLLMEngine instances (replaces sync vllm.LLM)
MODELS:      dict[str, object]            = {}
LOAD_ERRORS: dict[str, str]               = {}
SEMAPHORES:  dict[str, asyncio.Semaphore] = {}

# The config each resident model was ACTUALLY loaded with, after environment
# overrides. /models reports from here when a model is loaded, so a benchmark
# records the configuration that served it rather than the one in the file.
EFFECTIVE_CONFIGS: dict[str, dict] = {}

# Seconds the last load of each model took. A cold 72B load is tens of seconds
# and it is a real operational cost — it is what the first photograph of a demo
# pays — so it is recorded rather than left for a client to infer.
LOAD_SECONDS: dict[str, float] = {}

# HF processors / tokenizers used to build chat prompts for the VLMs.
# Cached per model name so we pay the from_pretrained cost only once.
PROCESSORS:  dict[str, Any] = {}

# Per-model metrics (thread-safe enough for our single-process use)
_metrics: dict[str, dict] = defaultdict(lambda: {
    "requests_total":    0,
    "requests_active":   0,
    "requests_queued":   0,
    "tokens_generated":  0,
    "images_processed":  0,
    "total_elapsed_s":   0.0,
    "errors":            0,
})

# Per-client metrics (thread-safe for single-process use)
_client_metrics: dict[str, dict] = defaultdict(lambda: {
    "requests_total":   0,
    "tokens_generated": 0,
    "errors":           0,
    "total_elapsed_s":  0.0,
})


# ─────────────────────────────── VRAM helper ─────────────────────────────────

def _vram_stats() -> list[dict]:
    stats = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        stats.append({
            "gpu":      i,
            "name":     torch.cuda.get_device_name(i),
            "used_gb":  round((total - free) / 1e9, 2),
            "free_gb":  round(free            / 1e9, 2),
            "total_gb": round(total           / 1e9, 2),
            "util_pct": round((total - free) / total * 100, 1),
        })
    return stats


# ─────────────────────────────── Image handling ──────────────────────────────

def _decode_one_image(src: str):
    """
    Turn one image reference into a PIL.Image (RGB).

    Accepted forms:
      - "data:image/png;base64,iVBOR..."   (data URI, what browsers send)
      - "iVBORw0KGgo..."                   (bare base64 blob)
      - "http://..." / "https://..."       (fetched server-side)
      - "/data01/samples/page1.png"        (path on the GPU box)
    """
    from PIL import Image

    if not isinstance(src, str) or not src.strip():
        raise HTTPException(400, "Empty image entry")

    src = src.strip()
    raw: Optional[bytes] = None

    try:
        if src.startswith("data:"):
            # data:[<mediatype>][;base64],<data>
            header, _, payload = src.partition(",")
            if "base64" not in header:
                raise HTTPException(400, "Only base64 data URIs are supported")
            raw = base64.b64decode(payload, validate=False)

        elif src.startswith(("http://", "https://")):
            with urllib.request.urlopen(src, timeout=IMAGE_FETCH_TIMEOUT_S) as resp:
                raw = resp.read()

        elif os.path.exists(src):
            with open(src, "rb") as fh:
                raw = fh.read()

        else:
            # Last resort: assume a bare base64 payload
            raw = base64.b64decode(src, validate=True)

    except HTTPException:
        raise
    except (binascii.Error, ValueError):
        raise HTTPException(400, f"Image is neither a readable path, URL, nor valid base64: {src[:80]}")
    except OSError as exc:
        raise HTTPException(400, f"Could not read image '{src[:80]}': {exc}")

    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
        img = img.convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Not a decodable image: {exc}")

    # Downscale very large images — the vision tower cost is quadratic-ish in
    # pixel count and this is where VLM OOMs come from.
    w, h = img.size
    if max(w, h) > MAX_IMAGE_SIDE_PX:
        scale = MAX_IMAGE_SIDE_PX / max(w, h)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        log.info("Resizing image %sx%s → %sx%s", w, h, *new_size)
        img = img.resize(new_size)

    return img


def _decode_images(srcs: list[str]) -> list:
    if len(srcs) > MAX_IMAGES_PER_REQUEST:
        raise HTTPException(
            400, f"Too many images ({len(srcs)}); max is {MAX_IMAGES_PER_REQUEST}"
        )
    return [_decode_one_image(s) for s in srcs]


# ─────────────────────────────── Prompt building ─────────────────────────────

def _get_processor(name: str):
    """
    Lazily load and cache the HF processor/tokenizer used to apply the chat
    template. Loaded from the same local path as the weights — no network.
    """
    if name in PROCESSORS:
        return PROCESSORS[name]

    cfg   = MODEL_CONFIGS[name]
    path  = cfg["path"]
    style = cfg["prompt_style"]

    if style in ("qwen_vl", "pixtral", "molmo"):
        # All three carry an AutoProcessor with a chat template. InternVL is the
        # odd one out below because the OpenGVLab (non "-HF") repo ships no
        # processor — its tokenizer holds the template instead.
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    elif style == "internvl":
        # The OpenGVLab (non "-HF") repo has no AutoProcessor; the tokenizer
        # carries the chat template and vLLM does the image preprocessing.
        from transformers import AutoTokenizer
        proc = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    else:
        from transformers import AutoTokenizer
        proc = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    PROCESSORS[name] = proc
    log.info("Loaded HF processor for '%s' (%s)", name, type(proc).__name__)
    return proc


def _build_qwen_vl_prompt(name: str, text: str, n_images: int, system: Optional[str]) -> str:
    """
    Qwen3-VL chat format. apply_chat_template expands each {"type": "image"}
    entry into <|vision_start|><|image_pad|><|vision_end|>, which is exactly
    what vLLM's Qwen3-VL processor looks for when it splices in the pixels.
    Order matters: image blocks first, then the question — same as the model card.
    """
    processor = _get_processor(name)

    content: list[dict] = [{"type": "image"} for _ in range(n_images)]
    content.append({"type": "text", "text": text})

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
    messages.append({"role": "user", "content": content})

    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _build_internvl_prompt(name: str, text: str, n_images: int, system: Optional[str]) -> str:
    """
    InternVL chat format. vLLM expects a literal "<image>" placeholder per image
    in the user turn; it is expanded internally into the IMG_CONTEXT token run.
    Numbering the images (Image-1, Image-2 ...) is what the OpenGVLab card
    recommends for multi-image conversations.
    """
    tokenizer = _get_processor(name)

    if n_images == 1:
        prefix = "<image>\n"
    elif n_images > 1:
        prefix = "".join(f"Image-{i + 1}: <image>\n" for i in range(n_images))
    else:
        prefix = ""

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prefix + text})

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _build_pixtral_prompt(name: str, text: str, n_images: int, system: Optional[str]) -> str:
    """Pixtral chat format (HF-format checkpoint).

    The HF conversion carries a chat template whose {"type": "image"} entries
    expand to the [IMG] placeholder vLLM's Pixtral processor splices pixels
    into — the same shape as Qwen3-VL, which is why this mirrors that builder.

    VERIFY WITH /debug/prompt: the rendered string must contain [IMG] once per
    image. If it does not, the template did not expand, and the model will
    answer fluently about an image it never received.

    If the checkpoint is the OFFICIAL Mistral format (params.json +
    consolidated.safetensors + tekken.json) rather than the HF one, this will
    not work and no builder can fix it: that path needs tokenizer_mode="mistral"
    and structured chat messages rather than a rendered string, which is a
    different shape from the one _build_engine_input is built around. Get the
    HF conversion instead.
    """
    processor = _get_processor(name)

    content: list[dict] = [{"type": "image"} for _ in range(n_images)]
    content.append({"type": "text", "text": text})

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
    messages.append({"role": "user", "content": content})

    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _build_molmo_prompt(name: str, text: str, n_images: int, system: Optional[str]) -> str:
    """Molmo chat format.

    Molmo's vLLM path inserts image tokens itself from multi_modal_data, so —
    unlike Qwen3-VL and InternVL — the prompt string carries NO image
    placeholder. That is why this builder looks suspiciously plain, and it is
    why `n_images` is accepted and deliberately unused. Do not "fix" it by
    adding <image>.

    Molmo has no system role, so a system prompt is folded into the user turn
    rather than dropped. That matters here: every question in the field-ops
    pipeline carries its own system prompt, and silently losing it makes the
    model answer unframed — which reads as a model-quality problem rather than
    the wiring difference it actually is.

    VERIFY WITH /debug/prompt before spending GPU time on this model.
    """
    processor = _get_processor(name)
    body = f"{system}\n\n{text}" if system else text

    template = getattr(processor, "apply_chat_template", None)
    if template is None:
        return f"User: {body} Assistant:"
    try:
        return template(
            [{"role": "user", "content": body}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception as exc:
        # Custom modelling code; if it exposes no usable template this is the
        # documented plain form. Logged rather than silent, because a template
        # that has stopped working is worth knowing about before a benchmark.
        log.warning("molmo: chat template unusable (%s) — falling back to the "
                    "plain User/Assistant form", exc)
        return f"User: {body} Assistant:"


# One place that maps a prompt_style to its builder, so /infer and
# /debug/prompt cannot disagree about what a model's prompt looks like. In v6
# they could: debug_prompt's else-branch rendered ANY non-qwen model with
# InternVL's template, which would have made a wrong pixtral template look
# correct in the one check that exists to catch it.
_PROMPT_BUILDERS = {
    "qwen_vl":  _build_qwen_vl_prompt,
    "internvl": _build_internvl_prompt,
    "pixtral":  _build_pixtral_prompt,
    "molmo":    _build_molmo_prompt,
}


def _render_prompt(model: str, text: str, n_images: int, system: Optional[str]) -> str:
    style = MODEL_CONFIGS[model]["prompt_style"]
    builder = _PROMPT_BUILDERS.get(style)
    if builder is None:
        raise HTTPException(500, f"No prompt builder for style '{style}'")
    return builder(model, text, n_images, system)


def _build_engine_input(req: "InferRequest") -> Any:
    """
    Produce whatever vLLM's generate() should receive:
      - text models      → the prompt string, untouched (v5 behaviour preserved)
      - vision models    → {"prompt": <templated str>,
                            "multi_modal_data": {"image": [PIL, ...]}}
    A vision model with no images still goes through its chat template, so it
    can be used for text-only questions too.
    """
    cfg    = MODEL_CONFIGS[req.model]
    images = req.images or []

    if cfg["modality"] != "vision":
        if images:
            raise HTTPException(
                400,
                f"Model '{req.model}' is text-only. Use a vision model: "
                f"{[n for n, c in MODEL_CONFIGS.items() if c['modality'] == 'vision']}",
            )
        return req.prompt

    cap = cfg.get("limit_mm_per_prompt", {}).get("image", MAX_IMAGES_PER_REQUEST)
    if len(images) > cap:
        raise HTTPException(
            400,
            f"'{req.model}' is configured for at most {cap} image(s) per prompt "
            f"(limit_mm_per_prompt). Got {len(images)}.",
        )

    pil_images = _decode_images(images)

    prompt = _render_prompt(req.model, req.prompt, len(pil_images), req.system)

    engine_input: dict = {"prompt": prompt}
    if pil_images:
        engine_input["multi_modal_data"] = {"image": pil_images}

    _metrics[req.model]["images_processed"] += len(pil_images)
    return engine_input


# ─────────────────────────────── Load / unload ───────────────────────────────

async def _load_one_async(name: str) -> None:
    """
    Load a model as an AsyncLLMEngine.
    Must be called from an async context so the engine's event loop is correct.
    - EXCLUSIVE models evict every other resident model.
    - Models in an EVICT_GROUPS group evict only their group siblings
      (so loading internvl frees qwen3-vl but leaves mistral serving).
    """
    if name in MODELS:
        return

    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model '{name}'")

    # [C] The effective config, not the table: environment overrides are applied
    # here so every consumer below — engine args, semaphore depth, the load log
    # — sees the same values, and /models reports what is actually serving.
    cfg = _effective_config(name)
    EFFECTIVE_CONFIGS[name] = cfg

    # Exclusive models need full VRAM — evict everything else first
    if name in EXCLUSIVE_MODELS and MODELS:
        log.warning(
            "'%s' is exclusive — unloading all resident models: %s",
            name, list(MODELS.keys()),
        )
        for loaded in list(MODELS.keys()):
            _unload_one(loaded)

    # Group eviction — only siblings in the same group
    group = EVICT_GROUPS.get(name)
    if group:
        for loaded in list(MODELS.keys()):
            if loaded != name and EVICT_GROUPS.get(loaded) == group:
                log.warning(
                    "'%s' and '%s' share eviction group '%s' — unloading '%s' first",
                    name, loaded, group, loaded,
                )
                _unload_one(loaded)

    log.info("Loading '%s' (AsyncLLMEngine) from %s ...", name, cfg["path"])
    t0 = time.time()

    try:
        import inspect
        from vllm import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        # ── Version-safe kwargs ───────────────────────────────────────────────
        # Different vLLM releases accept different AsyncEngineArgs params.
        # We probe the constructor signature at runtime and only pass what
        # this installation actually supports — no more unexpected-kwarg crashes.
        _valid_params = set(inspect.signature(AsyncEngineArgs.__init__).parameters)
        log.info("vLLM AsyncEngineArgs supports: %s", sorted(_valid_params))

        # Base kwargs — always valid across all vLLM versions
        engine_kwargs: dict = {
            "model":                   cfg["path"],
            "dtype":                   cfg["dtype"],
            "tensor_parallel_size":    cfg["tensor_parallel_size"],
            "gpu_memory_utilization":  cfg["gpu_memory_utilization"],
            "max_model_len":           cfg["max_model_len"],
            "trust_remote_code":       cfg["trust_remote_code"],
        }

        # quantization: param name changed across versions
        if "quantization" in _valid_params and cfg["quantization"]:
            engine_kwargs["quantization"] = cfg["quantization"]

        # disable_log_requests: present in older vLLM (< 0.4), removed later
        if "disable_log_requests" in _valid_params:
            engine_kwargs["disable_log_requests"] = False

        # enable_prefix_caching: added in vLLM ~0.4.2
        # Huge win for Falcon — schema/system prompt prefix is identical across requests
        if "enable_prefix_caching" in _valid_params:
            engine_kwargs["enable_prefix_caching"] = True
            log.info("'%s': prefix caching ENABLED", name)
        else:
            log.info("'%s': prefix caching not available in this vLLM version", name)

        # max_num_seqs: cap concurrent sequences in the engine scheduler
        # Prevents OOM on KV cache under high concurrency
        if "max_num_seqs" in _valid_params:
            engine_kwargs["max_num_seqs"] = cfg["max_concurrent"] * 4

        # ── Multimodal kwargs (VLMs only) ────────────────────────────────────
        # limit_mm_per_prompt tells vLLM how much encoder memory to preallocate;
        # setting video=0 on an image-only workload frees real VRAM.
        if cfg["modality"] == "vision":
            if "limit_mm_per_prompt" in _valid_params and cfg.get("limit_mm_per_prompt"):
                engine_kwargs["limit_mm_per_prompt"] = cfg["limit_mm_per_prompt"]
                log.info("'%s': limit_mm_per_prompt=%s", name, cfg["limit_mm_per_prompt"])
            if "mm_processor_kwargs" in _valid_params and cfg.get("mm_processor_kwargs"):
                engine_kwargs["mm_processor_kwargs"] = cfg["mm_processor_kwargs"]
                log.info("'%s': mm_processor_kwargs=%s", name, cfg["mm_processor_kwargs"])

        engine_args = AsyncEngineArgs(**engine_kwargs)
        engine      = AsyncLLMEngine.from_engine_args(engine_args)

        MODELS[name]     = engine
        SEMAPHORES[name] = asyncio.Semaphore(cfg["max_concurrent"])
        LOAD_ERRORS.pop(name, None)

        # Warm the chat template up front so the first VLM request doesn't pay
        # a from_pretrained stall inside the semaphore.
        if cfg["modality"] == "vision":
            try:
                _get_processor(name)
            except Exception as exc:
                log.warning("'%s': processor preload failed (%s) — will retry per request", name, exc)

        LOAD_SECONDS[name] = round(time.time() - t0, 2)
        log.info("'%s' ready in %.1f s.", name, time.time() - t0)
        for g in _vram_stats():
            log.info(
                "  GPU %d  used=%.1f GB  free=%.1f GB  (%.1f%%)",
                g["gpu"], g["used_gb"], g["free_gb"], g["util_pct"],
            )

    except Exception as exc:
        LOAD_ERRORS[name] = str(exc)
        log.error("Failed to load '%s': %s", name, exc)
        raise


def _unload_one(name: str) -> None:
    """Synchronously destroy an engine and release VRAM."""
    if name not in MODELS:
        return
    log.info("Unloading '%s' ...", name)
    engine = MODELS.pop(name)
    SEMAPHORES.pop(name, None)

    # Shutdown method name changed across vLLM versions — try each in order
    for shutdown_method in ("shutdown", "shutdown_background_loop", "abort_all_requests"):
        fn = getattr(engine, shutdown_method, None)
        if fn is not None:
            try:
                fn()
                log.info("'%s' shut down via engine.%s()", name, shutdown_method)
            except Exception as exc:
                log.warning("engine.%s() raised: %s (continuing)", shutdown_method, exc)
            break

    del engine
    gc.collect()
    torch.cuda.empty_cache()
    log.info("'%s' unloaded. VRAM after: %s", name, _vram_stats())


# ─────────────────────────────── Lifespan ────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=== GPU API Server v7 startup ===")
    try:
        import vllm
        log.info("vLLM version: %s", vllm.__version__)
    except Exception:
        log.warning("Could not determine vLLM version")
    try:
        import transformers
        log.info("transformers version: %s", transformers.__version__)
    except Exception:
        log.warning("Could not determine transformers version")
    log.info("VRAM at boot: %s", _vram_stats())

    # Eager-load only the primary model(s)
    for name in EAGER_LOAD:
        try:
            await _load_one_async(name)
        except Exception as exc:
            log.error("Eager load of '%s' failed: %s", name, exc)

    log.info("Startup complete — eager models: %s", list(MODELS.keys()))
    log.info("Lazy models (load on first request): %s",
             [n for n in MODEL_CONFIGS if MODEL_CONFIGS[n]["lazy"]])
    log.info("Vision models: %s",
             [n for n, c in MODEL_CONFIGS.items() if c["modality"] == "vision"])

    yield

    # Graceful shutdown — drain in-flight before destroying engines
    log.info("Shutting down — waiting for in-flight requests ...")
    for name, sem in list(SEMAPHORES.items()):
        max_c = MODEL_CONFIGS[name]["max_concurrent"]
        for _ in range(max_c):
            try:
                await asyncio.wait_for(sem.acquire(), timeout=30.0)
            except asyncio.TimeoutError:
                log.warning("Shutdown timeout waiting for '%s' to drain", name)
                break

    for name in list(MODELS.keys()):
        _unload_one(name)

    log.info("=== GPU API Server v7 shutdown complete ===")


# ─────────────────────────────── App ─────────────────────────────────────────

app = FastAPI(title="GPU Inference API", version="7.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────── Schemas ─────────────────────────────────────

class InferRequest(BaseModel):
    model:              str
    prompt:             str
    # Vision models only. Each entry may be a data URI, bare base64, an
    # http(s) URL, or a path on the GPU box.
    images:             Optional[list[str]] = None
    # Optional system turn; only applied to models that use a chat template.
    system:             Optional[str] = None
    max_new_tokens:     int   = Field(1024, ge=1,   le=32768)
    temperature:        float = Field(0.7,  ge=0.0, le=2.0)
    top_p:              float = Field(0.9,  ge=0.0, le=1.0)
    top_k:              int   = Field(50,   ge=0)
    repetition_penalty: float = Field(1.1,  ge=1.0, le=2.0)
    stop_sequences:     Optional[list[str]] = None
    # Tracing fields echoed back in the response
    request_id:         Optional[str] = None
    client_id:          Optional[str] = None


class InferResponse(BaseModel):
    text:           str
    model:          str
    prompt_tokens:  int
    new_tokens:     int
    images:         int = 0
    elapsed_s:      float
    tokens_per_sec: float
    # [E] Seconds spent waiting for a concurrency slot before inference began.
    # elapsed_s covers inference ONLY, so the two add up to the server-side
    # cost and their ratio says which of the two a slow request was.
    queue_wait_s:   float = 0.0
    request_id:     Optional[str] = None
    client_id:      Optional[str] = None


class ModelInfo(BaseModel):
    name:                   str
    loaded:                 bool
    lazy:                   bool
    modality:               str
    tensor_parallel_size:   int
    max_model_len:          int
    max_concurrent:         int
    gpu_memory_utilization: float
    max_images:             Optional[int] = None
    quantization:           Optional[str] = None
    error:                  Optional[str] = None
    # Which template a client should expect in /debug/prompt.
    prompt_style:           Optional[str] = None
    # Seconds the last load took. A benchmark records this as the cold-start
    # cost instead of inferring it from a suspiciously slow first request.
    load_seconds:           Optional[float] = None
    # True when the values above came from environment overrides rather than
    # from MODEL_CONFIGS — so a result file can never claim a configuration the
    # server was not actually running. See _effective_config.
    env_overridden:         bool = False


# ─────────────────────────────── Internal inference ──────────────────────────

_load_lock = asyncio.Lock()


async def _ensure_loaded(name: str) -> None:
    """Lazy-load a model if not yet in MODELS. Serialised via an asyncio lock."""
    if name not in MODELS:
        # Use a module-level lock so two concurrent first requests
        # don't both try to load the same model simultaneously
        async with _load_lock:
            if name not in MODELS:   # double-check after acquiring lock
                await _load_one_async(name)
                if name in LOAD_ERRORS:
                    raise HTTPException(503, f"Model load failed: {LOAD_ERRORS[name]}")


def _sampling_params(req: InferRequest):
    from vllm import SamplingParams

    cfg  = MODEL_CONFIGS[req.model]
    stop = list(req.stop_sequences or [])
    stop.extend(cfg.get("stop_strings", []))

    kwargs: dict = {
        "max_tokens":         req.max_new_tokens,
        "temperature":        req.temperature,
        "top_p":              req.top_p,
        "top_k":              req.top_k if req.top_k > 0 else -1,
        "repetition_penalty": req.repetition_penalty,
        "stop":               sorted(set(stop)),
    }
    if cfg.get("stop_token_ids"):
        kwargs["stop_token_ids"] = cfg["stop_token_ids"]
    return SamplingParams(**kwargs)


async def _run_inference(req: InferRequest, queue_wait_s: float = 0.0) -> InferResponse:
    """
    Core async inference path using AsyncLLMEngine.
    The engine queues the request internally and batches it with any other
    concurrent requests hitting the same engine — this is where continuous
    batching actually happens.
    """
    rid = req.request_id or str(uuid.uuid4())
    m   = _metrics[req.model]

    params       = _sampling_params(req)
    engine_input = _build_engine_input(req)   # str, or dict with multi_modal_data

    t0 = time.time()
    m["requests_active"] += 1
    m["requests_total"]  += 1

    try:
        engine       = MODELS[req.model]
        results_gen  = engine.generate(engine_input, params, request_id=rid)
        final_output = None

        # Iterate the async generator — vLLM yields incremental outputs;
        # we only need the final completed one for a non-streaming response.
        async for output in results_gen:
            final_output = output

        if final_output is None:
            raise RuntimeError("AsyncLLMEngine returned no output")

        result  = final_output.outputs[0]
        n_new   = len(result.token_ids)
        elapsed = time.time() - t0

        m["tokens_generated"] += n_new
        m["total_elapsed_s"]  += elapsed

        # Track per-client stats for observability dashboard
        if req.client_id:
            cm = _client_metrics[req.client_id]
            cm["requests_total"]   += 1
            cm["tokens_generated"] += n_new
            cm["total_elapsed_s"]  += elapsed

        return InferResponse(
            text           = result.text.strip() or "I'm not confident enough to answer that.",
            model          = req.model,
            prompt_tokens  = len(final_output.prompt_token_ids or []),
            new_tokens     = n_new,
            images         = len(req.images or []),
            elapsed_s      = round(elapsed, 3),
            # [E] Time this request spent waiting for one of max_concurrent
            # slots, measured on the server rather than inferred by a client
            # subtracting two numbers. With max_concurrent=2 on the VLMs this
            # is what separates "the model is slow" from "the model is busy",
            # and a 503 only arrives once it exceeds QUEUE_TIMEOUT_S.
            queue_wait_s   = round(queue_wait_s, 3),
            tokens_per_sec = round(n_new / elapsed, 1) if elapsed > 0 else 0.0,
            request_id     = req.request_id,
            client_id      = req.client_id,
        )

    except Exception as exc:
        m["errors"] += 1
        raise exc

    finally:
        m["requests_active"] -= 1


async def _stream_inference(req: InferRequest) -> AsyncIterator[str]:
    """
    SSE streaming path — yields tokens as they are generated.
    Useful for long extraction / explanation generations where the client
    wants to start parsing before the full response is ready.
    """
    rid    = req.request_id or str(uuid.uuid4())
    m      = _metrics[req.model]
    params = _sampling_params(req)

    m["requests_active"] += 1
    m["requests_total"]  += 1
    prev_len = 0

    try:
        engine_input = _build_engine_input(req)
        engine       = MODELS[req.model]
        results_gen  = engine.generate(engine_input, params, request_id=rid)

        async for output in results_gen:
            text     = output.outputs[0].text
            delta    = text[prev_len:]
            prev_len = len(text)
            if delta:
                yield f"data: {delta}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as exc:
        m["errors"] += 1
        yield f"data: [ERROR] {exc}\n\n"

    finally:
        m["requests_active"] -= 1


# ─────────────────────────────── Routes ──────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":         "ok",
        "cuda_available":  torch.cuda.is_available(),
        "gpu_count":       torch.cuda.device_count(),
        "vram":            _vram_stats(),
        "loaded_models":   list(MODELS.keys()),
        "vision_models":   [n for n, c in MODEL_CONFIGS.items() if c["modality"] == "vision"],
        "failed_models":   LOAD_ERRORS,
        "load_seconds":    LOAD_SECONDS,
        # Every VLM_* override in effect right now. A benchmark variable left
        # set after the window is the quietest way to make a later measurement
        # wrong, so it is surfaced here rather than only in the boot log.
        "env_overrides":   {
            k: v for k, v in os.environ.items()
            if k.startswith(("VLM_TP_", "VLM_GMU_", "VLM_LEN_", "VLM_CONC_",
                             "VLM_QUANT_", "EAGER_LOAD"))
        },
        "eager_load":      EAGER_LOAD,
    }


@app.get("/models", response_model=list[ModelInfo])
async def list_models():
    """The registry, reporting what is ACTUALLY serving.

    For a loaded model these values come from EFFECTIVE_CONFIGS — the config it
    was really loaded with, after environment overrides — not from the table in
    this file. A benchmark that records "internvl at TP=2" because that is what
    the source says, while the process is serving it at TP=1 from
    VLM_TP_INTERNVL, has recorded a configuration that never existed.
    """
    out = []
    for name in MODEL_CONFIGS:
        table = MODEL_CONFIGS[name]
        cfg = EFFECTIVE_CONFIGS.get(name) or _effective_config(name)
        out.append(ModelInfo(
            name=name,
            loaded=(name in MODELS),
            lazy=table["lazy"],
            modality=cfg["modality"],
            tensor_parallel_size=cfg["tensor_parallel_size"],
            max_model_len=cfg["max_model_len"],
            max_concurrent=cfg["max_concurrent"],
            gpu_memory_utilization=cfg["gpu_memory_utilization"],
            max_images=cfg.get("limit_mm_per_prompt", {}).get("image"),
            quantization=cfg["quantization"],
            error=LOAD_ERRORS.get(name),
            prompt_style=cfg.get("prompt_style"),
            load_seconds=LOAD_SECONDS.get(name),
            env_overridden=any(
                cfg.get(field) != table.get(field) for field in _ENV_OVERRIDABLE),
        ))
    return out


@app.post("/models/{name}/load")
async def load_model(name: str):
    if name not in MODEL_CONFIGS:
        raise HTTPException(404, f"Unknown model '{name}'")
    if name in MODELS:
        return {"status": "already_loaded", "model": name, "vram": _vram_stats()}
    async with _load_lock:
        await _load_one_async(name)
    if name in LOAD_ERRORS:
        raise HTTPException(500, LOAD_ERRORS[name])
    return {"status": "loaded", "model": name, "vram": _vram_stats()}


@app.post("/models/{name}/unload")
async def unload_model(name: str):
    if name not in MODEL_CONFIGS:
        raise HTTPException(404, f"Unknown model '{name}'")
    _unload_one(name)
    PROCESSORS.pop(name, None)
    return {"status": "unloaded", "model": name, "vram": _vram_stats()}


@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    """
    Primary inference endpoint (text and vision).
    - Lazy-loads the model on first request
    - Acquires a per-model semaphore slot (queue depth = max_concurrent)
    - Times out after QUEUE_TIMEOUT_S seconds if all slots are busy
    - Delegates to AsyncLLMEngine which batches concurrent requests
    """
    if req.model not in MODEL_CONFIGS:
        raise HTTPException(400, f"Unknown model '{req.model}'. Available: {list(MODEL_CONFIGS)}")

    # Fail fast on a text model + images before paying a lazy load
    if req.images and MODEL_CONFIGS[req.model]["modality"] != "vision":
        raise HTTPException(
            400,
            f"Model '{req.model}' is text-only. Vision models: "
            f"{[n for n, c in MODEL_CONFIGS.items() if c['modality'] == 'vision']}",
        )

    # Lazy load under lock
    await _ensure_loaded(req.model)

    if req.request_id:
        log.info("[%s] client=%s model=%s tokens=%d images=%d",
                 req.request_id, req.client_id, req.model,
                 req.max_new_tokens, len(req.images or []))

    sem = SEMAPHORES[req.model]
    _metrics[req.model]["requests_queued"] += 1

    # Wait for a semaphore slot — reject with 503 if queue is backed up too long
    _queue_t0 = time.time()
    try:
        await asyncio.wait_for(sem.acquire(), timeout=QUEUE_TIMEOUT_S)
    except asyncio.TimeoutError:
        _metrics[req.model]["requests_queued"] -= 1
        _metrics[req.model]["errors"]          += 1
        raise HTTPException(
            503,
            f"Model '{req.model}' is busy — all {MODEL_CONFIGS[req.model]['max_concurrent']} "
            f"slots occupied for >{QUEUE_TIMEOUT_S}s. Try again shortly."
        )

    _metrics[req.model]["requests_queued"] -= 1
    _queue_wait = time.time() - _queue_t0

    try:
        return await _run_inference(req, queue_wait_s=_queue_wait)
    except HTTPException:
        raise
    except Exception as exc:
        log.error("[%s] Inference error on '%s': %s", req.request_id, req.model, exc)
        raise HTTPException(500, f"Inference failed: {exc}")
    finally:
        sem.release()


@app.post("/infer/stream")
async def infer_stream(req: InferRequest):
    """
    Streaming SSE endpoint — returns tokens as they are generated.
    Client receives Server-Sent Events: `data: <token_delta>\\n\\n`
    Terminated with `data: [DONE]\\n\\n`
    Accepts `images` for vision models, same as /infer.
    """
    if req.model not in MODEL_CONFIGS:
        raise HTTPException(400, f"Unknown model '{req.model}'")

    if req.images and MODEL_CONFIGS[req.model]["modality"] != "vision":
        raise HTTPException(400, f"Model '{req.model}' is text-only.")

    await _ensure_loaded(req.model)

    sem = SEMAPHORES[req.model]
    try:
        await asyncio.wait_for(sem.acquire(), timeout=QUEUE_TIMEOUT_S)
    except asyncio.TimeoutError:
        raise HTTPException(503, f"Model '{req.model}' is busy. Try again shortly.")

    async def _guarded_stream():
        try:
            async for chunk in _stream_inference(req):
                yield chunk
        finally:
            sem.release()

    return StreamingResponse(_guarded_stream(), media_type="text/event-stream")


@app.post("/debug/prompt")
async def debug_prompt(req: InferRequest):
    """
    Renders the final templated prompt without running the GPU.
    Invaluable when a VLM answers as if it never saw the image — check that the
    vision placeholders (<|image_pad|> or <image>) are actually in the string.
    Loads the HF processor only; no engine required.
    """
    if req.model not in MODEL_CONFIGS:
        raise HTTPException(400, f"Unknown model '{req.model}'")

    cfg      = MODEL_CONFIGS[req.model]
    n_images = len(req.images or [])

    if cfg["modality"] != "vision":
        return {"model": req.model, "prompt": req.prompt, "templated": False}

    # [G] v6 fell through to the InternVL builder for ANY non-qwen model, so a
    # pixtral or molmo request rendered here would have come back looking
    # perfectly plausible while using the wrong template entirely — defeating
    # the one check that stands between a wrong template and a run of confident
    # garbage. Dispatch is explicit now, and an unknown style raises.
    prompt = _render_prompt(req.model, req.prompt, n_images, req.system)

    return {
        "model": req.model, "prompt": prompt, "images": n_images,
        "templated": True, "prompt_style": cfg["prompt_style"],
        # What to look for, so the caller does not have to remember which
        # placeholder belongs to which family.
        "expect_placeholder": {
            "qwen_vl":  "<|vision_start|><|image_pad|><|vision_end|>",
            "internvl": "<image>",
            "pixtral":  "[IMG]",
            "molmo":    "(none — molmo inserts image tokens from multi_modal_data; "
                        "check instead that your system prompt survived into the "
                        "user turn)",
        }.get(cfg["prompt_style"], "unknown style"),
    }


@app.get("/metrics")
async def metrics():
    """
    Per-model request statistics.
    Use this to decide when you need to scale or adjust max_concurrent.
    """
    result = {}
    for name, m in _metrics.items():
        total   = m["requests_total"]
        elapsed = m["total_elapsed_s"]
        tokens  = m["tokens_generated"]
        result[name] = {
            "requests_total":       total,
            "requests_active":      m["requests_active"],
            "requests_queued":      m["requests_queued"],
            "errors":               m["errors"],
            "tokens_generated":     tokens,
            "images_processed":     m["images_processed"],
            "avg_tokens_per_req":   round(tokens / total, 1) if total else 0,
            "avg_latency_s":        round(elapsed / total, 3) if total else 0,
            "overall_tokens_per_s": round(tokens / elapsed, 1) if elapsed else 0,
            "loaded":               name in MODELS,
        }
    return {"models": result, "vram": _vram_stats()}


@app.post("/metrics/reset")
async def reset_metrics():
    """Zero the counters. [F]

    /metrics is cumulative since process start, which is the wrong window for a
    benchmark: a run's own throughput and error count are what matter, and
    subtracting two snapshots by hand is how an off-by-one enters a report.
    Call this immediately before a run.

    Returns what was cleared, so the reset itself is auditable rather than
    silently losing numbers somebody wanted.
    """
    previous = {name: dict(m) for name, m in _metrics.items()}
    previous_clients = {cid: dict(cm) for cid, cm in _client_metrics.items()}
    _metrics.clear()
    _client_metrics.clear()
    log.info("metrics reset — cleared %d model and %d client counters",
             len(previous), len(previous_clients))
    return {"status": "reset", "cleared_models": previous,
            "cleared_clients": previous_clients}


@app.get("/client-metrics")
async def client_metrics():
    """Per-client/project request and token statistics."""
    result = {}
    for cid, cm in _client_metrics.items():
        total   = cm["requests_total"]
        elapsed = cm["total_elapsed_s"]
        tokens  = cm["tokens_generated"]
        result[cid] = {
            "requests_total":       total,
            "tokens_generated":     tokens,
            "errors":               cm["errors"],
            "avg_latency_s":        round(elapsed / total, 3) if total else 0,
            "avg_tokens_per_req":   round(tokens  / total, 1) if total else 0,
            "overall_tokens_per_s": round(tokens  / elapsed, 1) if elapsed else 0,
        }
    return result


# ─────────────────────────────── Entry point ─────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print("""
╔══════════════════════════════════════════════════════════════════╗
║  GPU Inference API  v7  —  2x H200 NVL  (text + vision)          ║
╠══════════════════════════════════════════════════════════════════╣
║  AsyncLLMEngine  │  Continuous batching  │  Prefix cache         ║
║  Per-model semaphore queuing  │  SSE streaming  │  VLM images    ║
╠══════════════════════════════════════════════════════════════════╣
║  Endpoints:                                                      ║
║    POST /infer             — blocking inference (text + images)  ║
║    POST /infer/stream      — SSE streaming                       ║
║    POST /debug/prompt      — render templated prompt, no GPU     ║
║    GET  /health            — VRAM + loaded models                ║
║    GET  /models            — registry with config                ║
║    POST /models/{n}/load   — manual lazy load                    ║
║    POST /models/{n}/unload — free VRAM                           ║
║    GET  /metrics           — per-model throughput stats          ║
╠══════════════════════════════════════════════════════════════════╣
║  Eager load:  $EAGER_LOAD (default qwen3-vl, mistral)            ║
║  Lazy text:   qwen3, codestral                                   ║
║  Lazy vision: qwen3-vl, internvl, pixtral, molmo                 ║
║               (mutually exclusive — one VLM resident at a time)  ║
║  molmo is FP8: 72B bf16 is 144 GB vs a 141 GB card               ║
╚══════════════════════════════════════════════════════════════════╝
""")

    uvicorn.run(
        "gpu_api_server_v7:app",
        host="0.0.0.0",
        port=5432,
        workers=1,        # MUST be 1 — AsyncLLMEngine is not fork-safe
        log_level="info",
    )
