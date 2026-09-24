"""Registry entries and prompt builders to add to gpu_api_server_v6.py, so
pixtral and molmo can be benchmarked alongside qwen3-vl and internvl.

THIS FILE DOES NOT RUN. It lives here so the change to the GPU server is
version-controlled and reviewable next to the benchmark that depends on it.
Copy the marked blocks into `gpu_api_server_v6.py` on 10.66.98.137, then follow
`BENCHMARK_RUNBOOK.md`.

── Read this before pasting ─────────────────────────────────────────────────

**Molmo-72B does not fit on one GPU in bf16.** 72B parameters at 2 bytes each
is 144 GB of weights against a 141 GB H200 NVL — over the card before a single
byte of KV cache. The entry below therefore uses FP8, which halves the weights
to ~72 GB and leaves ~55 GB for KV at gpu_memory_utilization 0.90. That is the
only way to honour "one GPU, one at a time" for this model, and it is not free:
FP8 is a different numerical path from the bf16 models, so molmo's ANSWER
QUALITY figures are not strictly comparable with theirs. Its latency figures
are also flattering by comparison, because FP8 is faster. Both facts belong in
any write-up that puts them in the same table.

The alternatives, for the record: TP=2 (breaks the one-GPU rule and uses both
H200s), a pre-quantised AWQ/GPTQ checkpoint (~40 GB, needs downloading), or
dropping molmo from the comparison.

**Check the checkpoint format before pasting.** On the GPU box:

    ls /data01/llm_models/Pixtral_12B/ /data01/llm_models/Molmo_72B/

  · `config.json` + `model-*.safetensors` + `preprocessor_config.json`
    → HF format. Use the entries below as written.
  · `params.json` + `consolidated.safetensors` + `tekken.json`
    → official Mistral format. See PIXTRAL_MISTRAL_FORMAT at the bottom; this
    server is built around `apply_chat_template` producing a prompt STRING, and
    the Mistral format does not fit that shape. Getting the HF-format
    checkpoint is far less work than rewriting the prompt path.

**Verify the prompt before spending GPU time.** The failure mode for a wrong
VLM template is not an error — the model answers fluently as if it never saw
the image. `POST /debug/prompt` renders the templated string without loading an
engine. Check the image placeholder is actually in it. The runbook has the
exact curl.
"""

# ════════════════════════════════════════════════════════════════════════════
# BLOCK 1 — add to MODEL_CONFIGS
# ════════════════════════════════════════════════════════════════════════════
#
# Paste these two entries inside MODEL_CONFIGS, after "internvl".
#
# gpu_memory_utilization is set for a box where MISTRAL HAS BEEN UNLOADED, per
# the benchmark plan. Leave mistral resident and subtract its ~21 GB (0.15)
# from every budget below, which internvl and molmo cannot spare.

BLOCK_1_MODEL_CONFIGS = '''
    # ── VLM #3 — Pixtral-12B ─────────────────────────────────────────────────
    # 12B bf16 is ~24 GB of weights, so this is the one model here that fits
    # comfortably next to mistral if you ever need it to. TP=1 by design: the
    # benchmark compares models on ONE GPU.
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
    # FP8, NOT bf16, and not by preference: 72B x 2 bytes = 144 GB of weights
    # against a 141 GB card. bf16 cannot be loaded on one GPU at all. FP8 gives
    # ~72 GB of weights and ~55 GB of KV at 0.90.
    #
    # Consequences to carry into the report: FP8 is a different numerical path,
    # so molmo's answer quality is not strictly comparable with the bf16
    # models, and its latency is flattered because FP8 is faster.
    #
    # If vLLM rejects quantization="fp8" on this build, the fallbacks are a
    # pre-quantised AWQ/GPTQ checkpoint, or TP=2 - which uses both GPUs and
    # leaves the one-GPU comparison behind.
    "molmo": {
        "path":                   "/data01/llm_models/Molmo_72B/",
        "modality":               "vision",
        "prompt_style":           "molmo",
        "dtype":                  "bfloat16",
        "quantization":           "fp8",   # REQUIRED - see above
        "tensor_parallel_size":   1,
        "gpu_memory_utilization": 0.90,
        "max_model_len":          8192,    # modest: KV is the tight resource here
        "trust_remote_code":      True,    # allenai ships custom modelling code
        "max_concurrent":         1,       # 72B on one card; do not oversubscribe
        "limit_mm_per_prompt":    {"image": 1, "video": 0},
        "stop_token_ids":         [],
        "lazy":                   True,
    },
'''

# ════════════════════════════════════════════════════════════════════════════
# BLOCK 2 — extend EVICT_GROUPS
# ════════════════════════════════════════════════════════════════════════════
#
# REPLACE the existing EVICT_GROUPS with this. Without it, loading molmo does
# NOT unload qwen3-vl, and the second load OOMs — on a one-GPU plan every VLM
# must evict every other VLM.

BLOCK_2_EVICT_GROUPS = '''
EVICT_GROUPS: dict[str, str] = {
    "qwen3-vl": "vlm",
    "internvl": "vlm",
    "pixtral":  "vlm",
    "molmo":    "vlm",
}
'''

# ════════════════════════════════════════════════════════════════════════════
# BLOCK 3 — internvl onto one GPU
# ════════════════════════════════════════════════════════════════════════════
#
# internvl currently ships TP=2 @ 0.40, which spans both H200s. For a one-GPU
# comparison it needs TP=1, and then gpu_memory_utilization must cover 38B bf16
# = ~76 GB of weights on a single card: 0.40 (56 GB) is NOT ENOUGH and the load
# will fail. Change BOTH lines together.
#
#     "tensor_parallel_size":   1,      # was 2
#     "gpu_memory_utilization": 0.85,   # was 0.40 - 76 GB weights + ~44 GB KV
#
# qwen3-vl is already TP=1, but at 0.60. Raise it to 0.85 for the benchmark so
# every model gets the same total budget; note this is NOT its production value
# and its KV cache, and therefore its concurrency behaviour, differs from what
# you ship.

# ════════════════════════════════════════════════════════════════════════════
# BLOCK 4 — prompt builders
# ════════════════════════════════════════════════════════════════════════════
#
# Paste these two functions after _build_internvl_prompt, then add their
# branches to _build_engine_input and to debug_prompt (BLOCK 5).
#
# BOTH ARE STARTING POINTS THAT MUST BE VERIFIED with POST /debug/prompt before
# any benchmark run. A wrong VLM template does not raise - the model answers
# fluently as if blind, and every number you then collect is worthless. The
# runbook step that checks this is not optional.

BLOCK_4_PROMPT_BUILDERS = '''
def _build_pixtral_prompt(name: str, text: str, n_images: int, system) -> str:
    """Pixtral chat format (HF-format checkpoint).

    The HF conversion carries a chat template whose {"type": "image"} entries
    expand to the [IMG] placeholder vLLM looks for. Same shape as the Qwen3-VL
    builder, which is why it reuses AutoProcessor.

    VERIFY WITH /debug/prompt: the rendered string must contain [IMG] once per
    image. If it does not, the template did not expand and the model will
    answer blind.
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


def _build_molmo_prompt(name: str, text: str, n_images: int, system) -> str:
    """Molmo chat format.

    Molmo's vLLM path inserts the image tokens itself from multi_modal_data, so
    unlike Qwen3-VL and InternVL the prompt string carries NO image placeholder.
    That is the documented behaviour for Molmo in vLLM and it is why this
    builder looks suspiciously plain - do not "fix" it by adding <image>.

    Molmo has no system role in its template, so a system prompt is folded into
    the user turn. That matters here: every question in this pipeline carries a
    per-question system prompt, and silently dropping it would make molmo answer
    unframed - which reads as a model-quality problem rather than the wiring
    difference it is.

    VERIFY WITH /debug/prompt before running anything.
    """
    processor = _get_processor(name)
    body = f"{system}\\n\\n{text}" if system else text

    template = getattr(processor, "apply_chat_template", None)
    if template is None:
        return f"User: {body} Assistant:"
    try:
        return template(
            [{"role": "user", "content": body}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception as exc:
        # Molmo's processor is custom code; if it has no usable template this
        # is the documented plain form. Logged rather than silent, because a
        # template that stopped working is worth knowing about.
        log.warning("molmo: chat template unavailable (%s) - using plain form", exc)
        return f"User: {body} Assistant:"
'''

# ════════════════════════════════════════════════════════════════════════════
# BLOCK 5 — wire the builders in
# ════════════════════════════════════════════════════════════════════════════
#
# In _build_engine_input, REPLACE the prompt_style branch with:
#
#     if cfg["prompt_style"] == "qwen_vl":
#         prompt = _build_qwen_vl_prompt(req.model, req.prompt, len(pil_images), req.system)
#     elif cfg["prompt_style"] == "internvl":
#         prompt = _build_internvl_prompt(req.model, req.prompt, len(pil_images), req.system)
#     elif cfg["prompt_style"] == "pixtral":
#         prompt = _build_pixtral_prompt(req.model, req.prompt, len(pil_images), req.system)
#     elif cfg["prompt_style"] == "molmo":
#         prompt = _build_molmo_prompt(req.model, req.prompt, len(pil_images), req.system)
#     else:
#         raise HTTPException(500, f"No prompt builder for style '{cfg['prompt_style']}'")
#
# In _get_processor, the "qwen_vl" branch loads AutoProcessor. Extend its
# condition so the two new styles use it too:
#
#     if style in ("qwen_vl", "pixtral", "molmo"):
#         from transformers import AutoProcessor
#         proc = AutoProcessor.from_pretrained(path, trust_remote_code=True)
#
# In debug_prompt, the else-branch assumes internvl. Make it explicit, or
# /debug/prompt will render a pixtral request with InternVL's template and the
# check you are relying on will pass while lying:
#
#     style = cfg["prompt_style"]
#     if style == "qwen_vl":
#         prompt = _build_qwen_vl_prompt(req.model, req.prompt, n_images, req.system)
#     elif style == "internvl":
#         prompt = _build_internvl_prompt(req.model, req.prompt, n_images, req.system)
#     elif style == "pixtral":
#         prompt = _build_pixtral_prompt(req.model, req.prompt, n_images, req.system)
#     elif style == "molmo":
#         prompt = _build_molmo_prompt(req.model, req.prompt, n_images, req.system)
#     else:
#         raise HTTPException(500, f"No prompt builder for style '{style}'")

# ════════════════════════════════════════════════════════════════════════════
# BLOCK 6 — stop eager-loading qwen3-vl
# ════════════════════════════════════════════════════════════════════════════
#
# EAGER_LOAD is currently ["qwen3-vl", "mistral"]. During the benchmark that
# loads a VLM you may not want at every restart, and on a one-GPU plan it
# occupies the card you are about to need. For the benchmark window:
#
#     EAGER_LOAD = []           # load every model explicitly, on purpose
#
# RESTORE IT AFTERWARDS to ["qwen3-vl", "mistral"], along with qwen3-vl's
# gpu_memory_utilization of 0.60 and internvl's TP=2 @ 0.40. The runbook's last
# step is that restore, and it is the step people skip.

# ════════════════════════════════════════════════════════════════════════════
# BLOCK 7 — the proxy
# ════════════════════════════════════════════════════════════════════════════
#
# llm_proxy_v3 rejects the new models TWICE, and neither refusal mentions the
# proxy, so both look like GPU-server faults:
#
#   1. AVAILABLE_MODELS does not list them -> pydantic 422 before any network
#      call happens.
#   2. VISION_MODELS is derived from MODEL_IMAGE_CAPS, so an unlisted model is
#      treated as text-only -> 400 "Model 'pixtral' is text-only" the moment
#      images are attached.
#
# Both are environment variables; no code change is needed. Restart the proxy
# with:
#
#     export AVAILABLE_MODELS="mistral,qwen3,codestral,qwen3-vl,internvl,pixtral,molmo"
#     export PIXTRAL_IMAGE_CAP=4
#     export MOLMO_IMAGE_CAP=1
#
# MODEL_IMAGE_CAPS in the proxy only reads QWEN3_VL_IMAGE_CAP and
# INTERNVL_IMAGE_CAP from the environment, so the two new caps need three lines
# of code as well:
#
#     MODEL_IMAGE_CAPS: dict[str, int] = {
#         "qwen3-vl": int(os.environ.get("QWEN3_VL_IMAGE_CAP", "4")),
#         "internvl": int(os.environ.get("INTERNVL_IMAGE_CAP", "4")),
#         "pixtral":  int(os.environ.get("PIXTRAL_IMAGE_CAP",  "4")),
#         "molmo":    int(os.environ.get("MOLMO_IMAGE_CAP",    "1")),
#     }
#
# Keep the caps in step with limit_mm_per_prompt on the GPU server. The
# pipeline only ever sends 1 image (2 in full+crop mode), so these are
# headroom rather than a constraint - but a cap of 0 would reject every
# request with a message about image counts, which reads as a client bug.

# ════════════════════════════════════════════════════════════════════════════
# PIXTRAL_MISTRAL_FORMAT — only if the checkpoint is NOT HF format
# ════════════════════════════════════════════════════════════════════════════
#
# If /data01/llm_models/Pixtral_12B/ holds params.json + consolidated.safetensors
# + tekken.json, it is the official Mistral format. vLLM can serve it, but it
# needs engine args this server does not currently pass:
#
#     tokenizer_mode="mistral", config_format="mistral", load_format="mistral"
#
# and the prompt cannot be built with apply_chat_template - the Mistral path
# expects structured chat messages, not a rendered string, which is the shape
# _build_engine_input is built around.
#
# Adding that is a genuine change to the server's prompt path, not a new
# builder. Obtaining the HF-format conversion instead is much less work and
# keeps every model on one code path. Decide before pasting BLOCK 1.
