# Benchmark runbook — four VLMs, one GPU at a time

Step by step, for the actual boxes: GPU server `10.66.98.137:5432`, proxy
`10.19.71.246:8071`, benchmark client on whatever can reach the proxy.

`BENCHMARKS.md` explains what is measured and why. This is the procedure.

---

## Read this first — three things that will stop you

**1. Pixtral and Molmo are not in the server's registry.** There is no load
command that will work today. `MODEL_CONFIGS` in `gpu_api_server_v6.py` holds
only `mistral`, `qwen3`, `codestral`, `qwen3-vl`, `internvl`. They need new
entries *and* new prompt builders — `_build_engine_input` only knows `raw`,
`qwen_vl` and `internvl`. `deploy/gpu_models_bench.py` has every block to paste.

**2. Molmo-72B cannot run on one GPU in bf16.**

```
72B params x 2 bytes = 144 GB of weights
H200 NVL usable      = 141 GB
```

Over the card before a single byte of KV cache. Check any model against any
card yourself — this is executable, not prose, so it stays true when the
hardware or the model list changes:

```bash
python -c "
import sys; sys.path.insert(0,'backend')
from bench.capacity import fit_on_gpu, smallest_working_dtype
f = fit_on_gpu('molmo-72b', 72, dtype='bf16', gpu_memory_utilization=0.90)
print(f.fits, '-', f.reason)
print('smallest dtype that works:', smallest_working_dtype('molmo-72b', 72))
"
```

The registry entry therefore
uses **FP8** (~72 GB of weights, ~55 GB of KV at 0.90), which is the only way
to honour "one GPU". It is not free: FP8 is a different numerical path, so
molmo's **answer quality is not strictly comparable** with the bf16 models, and
its **latency is flattered** because FP8 is faster. Both belong in any table
that puts them side by side. If that trade is unacceptable, the choices are
TP=2 (both GPUs), a pre-quantised AWQ/GPTQ checkpoint, or dropping molmo.

**3. `internvl` currently spans both GPUs.** It ships TP=2 @ 0.40. For a
one-GPU run it needs TP=1, and then 0.40 is **not enough** — 38B bf16 is ~76 GB
and 0.40 of 141 GB is 56 GB, so the load fails. Change both values together.

---

## Step 0 — check the checkpoint format

On the GPU box. This decides whether Pixtral needs a different code path
entirely, so do it before editing anything.

```bash
ls -la /data01/llm_models/Pixtral_12B/
ls -la /data01/llm_models/Molmo_72B/
```

| What you see | Meaning |
|---|---|
| `config.json`, `model-0000*.safetensors`, `preprocessor_config.json` | **HF format.** Use `deploy/gpu_models_bench.py` as written. |
| `params.json`, `consolidated.safetensors`, `tekken.json` | **Official Mistral format.** See `PIXTRAL_MISTRAL_FORMAT` in that file — this server is built around a rendered prompt string and that format is not. Getting the HF conversion is far less work. |

Also confirm the model actually is what the folder name says:

```bash
python -c "import json;c=json.load(open('/data01/llm_models/Molmo_72B/config.json'));print(c.get('architectures'), c.get('torch_dtype'))"
```

---

## Step 1 — edit the GPU server

On `10.66.98.137`, back it up first:

```bash
cd /path/to/server
cp gpu_api_server_v6.py gpu_api_server_v6.py.pre-bench
```

Apply the blocks from `deploy/gpu_models_bench.py`:

| Block | What |
|---|---|
| 1 | `pixtral` and `molmo` entries in `MODEL_CONFIGS` |
| 2 | `EVICT_GROUPS` — all four VLMs in group `vlm`, so each evicts the others |
| 3 | `internvl` to TP=1 @ 0.85; `qwen3-vl` to 0.85 |
| 4 | `_build_pixtral_prompt` and `_build_molmo_prompt` |
| 5 | wire them into `_build_engine_input`, `_get_processor` and `debug_prompt` |
| 6 | `EAGER_LOAD = []` for the window |

Block 2 is the one that bites if skipped: without it, loading molmo does not
unload qwen3-vl, and on one GPU the second load OOMs.

Restart:

```bash
# in the tmux pane where it runs
uvicorn gpu_api_server_v6:app --host 0.0.0.0 --port 5432 --workers 1
```

Confirm the new models are registered:

```bash
curl -s localhost:5432/models | python -m json.tool | grep -A2 '"name"'
```

---

## Step 2 — the proxy

Your client can only reach the proxy, and it will reject the new models twice
over — neither refusal mentions the proxy, so both look like GPU faults. Apply
the `MODEL_IMAGE_CAPS` lines from Block 7, then restart with:

```bash
export AVAILABLE_MODELS="mistral,qwen3,codestral,qwen3-vl,internvl,pixtral,molmo"
export PIXTRAL_IMAGE_CAP=4
export MOLMO_IMAGE_CAP=1
export API_KEYS="bench:secret-bench,..."      # keep your existing keys
uvicorn llm_proxy_v3:app --host 0.0.0.0 --port 8071
```

Check from the **client machine**, not the GPU box:

```bash
curl -s http://10.19.71.246:8071/v1/models | python -m json.tool
curl -s http://10.19.71.246:8071/v1/gpu-models | python -m json.tool
```

The first is the proxy's allowlist; the second is the GPU server's real
registry passed through. Both must list all four VLMs.

---

## Step 3 — verify the prompt before spending GPU time

**Do not skip this.** A wrong VLM template does not raise. The model answers
fluently *as if it never saw the image*, and every number you then collect is
worthless. `/debug/prompt` renders the templated string without loading an
engine, so this costs seconds.

On the GPU box, for each new model:

```bash
curl -s localhost:5432/debug/prompt -H 'Content-Type: application/json' -d '{
  "model": "pixtral",
  "prompt": "Is a hazardous-warning sign visible?",
  "system": "You are a site-safety inspector. Respond with JSON only.",
  "images": ["/data01/samples/plot.png"]
}' | python -m json.tool
```

What to look for:

| Model | The prompt MUST contain |
|---|---|
| `qwen3-vl` | `<\|vision_start\|><\|image_pad\|><\|vision_end\|>` |
| `internvl` | `<image>` |
| `pixtral` | `[IMG]` |
| `molmo` | **no image placeholder** — Molmo's vLLM path inserts them from `multi_modal_data`. Check instead that your system prompt survived into the user turn, since Molmo has no system role. |

If a placeholder is missing, the chat template did not expand. Fix that before
going further.

---

## Step 4 — the benchmark loop, one model at a time

Repeat this block per model. Keep **the same `--run-id`** throughout so the
files merge at the end.

### 4a. Free the GPU

```bash
# on the GPU box
curl -s -X POST localhost:5432/models/mistral/unload | python -m json.tool
curl -s -X POST localhost:5432/models/qwen3-vl/unload | python -m json.tool   # whatever is resident
curl -s localhost:5432/health | python -m json.tool     # loaded_models should be []
```

Mistral is down from here until Step 5. That is the couple of minutes you said
you can spare — but note it is *per model*, so plan the window around the whole
sweep, not one load.

### 4b. Load the one you want

```bash
time curl -s -X POST localhost:5432/models/internvl/load | python -m json.tool
```

Read the VRAM in the response. If `free_gb` on GPU 0 is under ~5 GB, the KV
cache is starved and your concurrency numbers will be about that, not about the
model — lower `gpu_memory_utilization` and reload.

### 4c. Run it

From the client machine:

```bash
python tools/run_bench.py \
    --model internvl \
    --photos <folder> --limit 8 \
    --transport proxy \
    --gpu-url http://10.19.71.246:8071 \
    --api-key secret-bench \
    --timeout 600 \
    --scenario batch,continuous,parallel \
    --modes \
    --run-id 20260924
```

`--timeout 600` matters: the proxy's own read timeout is 600 s precisely
because a cold VLM load exceeds the old 120 s. A shorter client timeout records
a successful cold load as a failure.

The run aborts if the server is serving a different model than you named. On a
one-model-at-a-time box that is the failure that would otherwise hand you a
complete, plausible, entirely wrong file.

### 4d. The saturation ramp, separately

```bash
python tools/run_bench.py --model internvl --photos <folder> \
    --transport proxy --gpu-url http://10.19.71.246:8071 --api-key secret-bench \
    --timeout 600 --scenario ramp --ramp-levels 1,2,4,8 --run-id 20260924 --yes
```

**Set expectations:** `max_concurrent` is **2** for the VLMs and
`QUEUE_TIMEOUT_S` is **60**. So beyond 2 in flight, requests queue rather than
fail, and only turn into 503s once the wait exceeds 60 s. The ramp will
therefore show latency climbing steeply from level 4 and 503s appearing at 8 —
that is the semaphore, not the model. The report reads
`queue_wait_s` from the proxy and says so when waiting dominates inference.

Molmo is configured `max_concurrent: 1`; a ramp on it saturates at 2.

### 4e. Move on

Unload, load the next, repeat. Order suggestion — cheapest first, so a mistake
in the setup surfaces on a 30-second load rather than a five-minute one:

```
pixtral  →  qwen3-vl  →  internvl  →  molmo
```

---

## Step 5 — restore production

**This is the step people skip.** Put the box back the way you found it:

```bash
# on the GPU box
cp gpu_api_server_v6.py.pre-bench gpu_api_server_v6.py   # or revert blocks 3 and 6 by hand
```

What must go back:

| Setting | Benchmark value | Restore to |
|---|---|---|
| `EAGER_LOAD` | `[]` | `["qwen3-vl", "mistral"]` |
| `qwen3-vl` `gpu_memory_utilization` | `0.85` | `0.60` |
| `internvl` `tensor_parallel_size` | `1` | `2` |
| `internvl` `gpu_memory_utilization` | `0.85` | `0.40` |

The `pixtral` and `molmo` entries can stay — they are lazy, so they cost
nothing until requested, and leaving them means the next benchmark needs no
edit. `EVICT_GROUPS` can stay too; it only acts when a VLM is loaded.

Restart, then confirm the original state:

```bash
curl -s localhost:5432/health | python -m json.tool
# loaded_models should be ["qwen3-vl", "mistral"]
```

Leave the proxy's `AVAILABLE_MODELS` extended — an allowlist entry for a model
nobody requests has no cost, and removing it is how the next run starts with a
confusing 422.

---

## Step 6 — merge and read

```bash
python tools/bench_report.py bench_runs/20260924
```

Read in this order:

1. **Contract compliance.** A model flagged `UNUSABLE` or `fragile` is not a
   candidate however fast it is. `parse_vlm_answer` is tolerant in four tiers,
   so a model that never emits valid JSON still produces answers and looks
   fine — while being one prompt edit from producing nothing.
2. **Cold load.** What the first photograph costs after a switch. For a 72B
   this may dominate everything else.
3. **batch p50/p95**, with nothing else running.
4. **Queue wait vs server time.** If p95 queue wait exceeds median server time,
   you measured the semaphore, not the model.
5. **Disagreements.** The shortlist to look at by eye. With no ground truth,
   agreement is weak evidence of correctness but disagreement is strong
   evidence someone is wrong.

The report flags models run under different settings rather than tabulating
them silently — which will fire here, because molmo is FP8 and the rest are
bf16. That flag is correct. Keep it in the write-up.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `422 Unknown model 'pixtral'` | proxy `AVAILABLE_MODELS` — Step 2 |
| `400 Model 'pixtral' is text-only` | proxy `MODEL_IMAGE_CAPS` — Block 7 |
| `503 Model 'x' is busy` during a ramp | working as intended: `max_concurrent` slots full for >60 s |
| `504` on the first request | cold load exceeded a timeout. Raise `--timeout`; the proxy itself allows 600 s |
| Model answers fluently but ignores the image | wrong prompt template. Step 3 |
| CUDA OOM on load | `gpu_memory_utilization` too high for what is already resident. Unload first, check `/health` |
| Run aborts: "server answered as X" | the model was never switched. That is the guard doing its job |
| Numbers look great and identical across models | check `resident_models` in the result file — you probably benchmarked one model four times |

Keep the tmux pane with the GPU server visible while running. `_load_one_async`
logs VRAM per GPU after every load, and that is the fastest way to see whether
a model fitted comfortably or scraped in.
