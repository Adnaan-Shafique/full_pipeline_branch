# Benchmark runbook — four VLMs, one GPU at a time

Step by step, for the actual boxes: GPU server `10.66.98.137:5432`, proxy
`10.19.71.246:8071`, benchmark client on whatever can reach the proxy.

`BENCHMARKS.md` explains what is measured and why. This is the procedure.

**Two files replace the editing this runbook used to ask for.**
`deploy/gpu_api_server_v7.py` and `deploy/llm_proxy_v4.py` are complete drop-in
replacements for the v6 server and v3 proxy — copy them over, do not paste
blocks into the running ones. `deploy/gpu_models_bench.py` is kept only as the
annotated diff, for reading what changed and why; it is no longer the procedure.

Both carry a `CHANGES FROM` header listing every difference with its reason.
The one that matters most here: **the benchmark settings are environment
variables, not edits.** `MODEL_CONFIGS` on disk is always the production
configuration, so Step 5's restore is unsetting three variables rather than
remembering four numbers.

---

## Read this first — two things that will stop you

**1. Molmo-72B cannot run on one GPU in bf16.**

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

v7's `molmo` entry therefore uses **FP8** (~72 GB of weights, ~55 GB of KV at
0.90), which is the only way to honour "one GPU". It is not free: FP8 is a
different numerical path, so molmo's **answer quality is not strictly
comparable** with the bf16 models, and its **latency is flattered** because FP8
is faster. Both belong in any table that puts them side by side. If that trade
is unacceptable, the choices are TP=2 (both GPUs), a pre-quantised AWQ/GPTQ
checkpoint, or dropping molmo.

**2. `internvl` ships TP=2 @ 0.40, spanning both GPUs.** For a one-GPU run it
needs TP=1, and then 0.40 is **not enough** — 38B bf16 is ~76 GB and 0.40 of
141 GB is 56 GB, so the load fails. Both must change together, which is why
Step 4 sets them as a pair:

```bash
export VLM_TP_INTERNVL=1
export VLM_GMU_INTERNVL=0.85
```

v7 logs every override it applies at load time and reports them on `/models`,
so a run can record the configuration that actually served it rather than the
one in the file.

---

## Step 0 — check the checkpoint format

On the GPU box. This decides whether Pixtral needs a different code path
entirely, so do it before copying anything.

```bash
ls -la /data01/llm_models/Pixtral_12B/
ls -la /data01/llm_models/Molmo_72B/
```

| What you see | Meaning |
|---|---|
| `config.json`, `model-0000*.safetensors`, `preprocessor_config.json` | **HF format.** v7 works as shipped. |
| `params.json`, `consolidated.safetensors`, `tekken.json` | **Official Mistral format.** See `PIXTRAL_MISTRAL_FORMAT` in `deploy/gpu_models_bench.py` — this server is built around a rendered prompt string and that format is not. Getting the HF conversion is far less work. |

Also confirm the model actually is what the folder name says:

```bash
python -c "import json;c=json.load(open('/data01/llm_models/Molmo_72B/config.json'));print(c.get('architectures'), c.get('torch_dtype'))"
```

---

## Step 1 — the GPU server

On `10.66.98.137`. Keep the old file; you are not editing it, and it is the
thing to fall back to.

```bash
cd /path/to/server
cp gpu_api_server_v6.py gpu_api_server_v6.py.keep
# copy deploy/gpu_api_server_v7.py from this repo to the same folder
python -c "import ast;ast.parse(open('gpu_api_server_v7.py').read())"   # parses?
```

What v7 adds over v6, in one table — the full reasoning is in its own header:

| | |
|---|---|
| `pixtral` and `molmo` in `MODEL_CONFIGS` | with their own prompt builders |
| all four VLMs in `EVICT_GROUPS` | loading one unloads the others, so the second load on one GPU does not OOM |
| `VLM_TP_*` / `VLM_GMU_*` / `VLM_LEN_*` / `VLM_CONC_*` / `VLM_QUANT_*` | benchmark settings as environment, so the file stays production |
| `EAGER_LOAD` from the environment | `EAGER_LOAD=""` for the window, without an edit |
| `queue_wait_s` measured at the semaphore | v3's proxy could only approximate it; see Step 6 |
| `POST /metrics/reset` | `/metrics` is cumulative since process start, which is the wrong window for a run |
| `/debug/prompt` **bug fix** | v6's else-branch rendered the InternVL template for *any* non-qwen model, so the check in Step 3 would have passed pixtral and molmo while showing the wrong prompt |

Start it with the benchmark window's settings:

```bash
# in the tmux pane where it runs
export EAGER_LOAD=""              # nothing resident at startup; we load by hand
export VLM_TP_INTERNVL=1
export VLM_GMU_INTERNVL=0.85
export VLM_GMU_QWEN3_VL=0.85
uvicorn gpu_api_server_v7:app --host 0.0.0.0 --port 5432 --workers 1
```

Confirm the registry, and that the overrides took:

```bash
curl -s localhost:5432/models | python -m json.tool | grep -E '"name"|env_overridden|tensor_parallel|gpu_memory'
curl -s localhost:5432/health | python -m json.tool | grep -A6 env_overrides
```

All four VLMs must be listed, and `internvl` must read TP 1 @ 0.85.

---

## Step 2 — the proxy

On the proxy box. Your client can only reach the proxy, and v3 would reject the
new models twice over — a 422 from the allowlist and a 400 "is text-only" from
the image caps — and **neither refusal mentions the proxy**, so both read as GPU
faults. v4's defaults already cover all four VLMs, so there is nothing to set
unless you want to narrow them.

```bash
cd /path/to/proxy
cp llm_proxy_v3.py llm_proxy_v3.py.keep
# copy deploy/llm_proxy_v4.py from this repo to the same folder
export API_KEYS="bench:secret-bench,..."      # keep your existing keys
export GPU_HOST=10.66.98.137
uvicorn llm_proxy_v4:app --host 0.0.0.0 --port 8071
```

v4 also fixes a silent drop: v3's `InferRequest` declared neither
`repetition_penalty` nor `stop_sequences`, and **pydantic discards undeclared
fields rather than rejecting them** — so a client setting either had it removed
at the edge while the server's own default applied, and a direct-vs-proxied
comparison silently ran two different sampling configurations.

Check from the **client machine**, not the GPU box:

```bash
curl -s http://10.19.71.246:8071/v1/models | python -m json.tool
curl -s http://10.19.71.246:8071/v1/gpu-models | python -m json.tool
```

The first is the proxy's allowlist and its image caps; the second is the GPU
server's real registry passed through. Both must list all four VLMs, and the
caps must match each model's `limit_mm_per_prompt` — molmo is **1**, the rest 4.

---

## Step 2b — the development VM (the benchmark client)

Everything from here runs on the machine you launch the benchmark from, not on
either server. It needs to reach **the proxy only** — `10.19.71.246:8071`.

### Get the code and a clean interpreter

```bash
git clone -b claude/vibrant-einstein-4y89hn <this repo> fieldops && cd fieldops
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements-demo.txt
```

`torch` is **not** needed. `--modes` runs with `use_model=False`, which reads
annotation `.txt` sidecars rather than the trained detector, so the whole
three-mode pipeline runs without it. `rembg` + `onnxruntime` are needed — they
are stage 1, and without them every photograph fails the quality gate for the
wrong reason. `rapidocr` is needed only if you benchmark one of the four OCR
questions, and its absence is a skip with a line saying so, not a crash.

Check the interpreter before the GPU is involved:

```bash
python tools/preflight.py --skip-gpu
for t in tests/test_*.py; do python "$t" >/dev/null || echo "FAILED $t"; done
```

### Point it at the proxy

Set it once in the shell rather than passing four flags to every command. These
are read by `pipeline.config`; explicit flags still win over them.

```bash
export FIELDOPS_VLM_TRANSPORT=proxy
export FIELDOPS_GPU_URL=http://10.19.71.246:8071
export FIELDOPS_VLM_API_KEY=secret-bench
```

**The failure to expect here** is `gpu_url` pointing at the proxy with
transport still `direct`: `/infer` does not exist on the proxy, so it 404s and
reads exactly like a dead GPU server. Both variables or neither.

Now the same preflight with the GPU reachable:

```bash
python tools/preflight.py --transport proxy
```

It must reach `/v1/health`, list all four VLMs, and report no registry
warnings. Warnings about `torch`, `ultralytics` and `dash` are expected on a
benchmark client and cost nothing.

### Photographs, and their annotations

```bash
python tools/run_bench.py --model qwen3-vl --photos <folder> --mock --modes
```

A mock run calls no model at all — every sample is tagged `MOCK` and the
timings appear only under `harness_only_timing_NOT_latency`. It is here to
prove the plumbing: the photographs decode, the question resolves, the three
modes run, and the report writes. Do this before spending GPU time.

If mode 1 or 2 reports every photograph stopped at detection, the annotation
sidecars are missing. The detector looks beside the photographs first and falls
back to `data/labels/`; **"the detector found nothing" and "nothing was ever
trained to find this" look identical on a card**, and only one of them is
evidence.

### Pick ONE question and keep it

```bash
python tools/smoke_ocr.py --list x      # which questions use OCR, and their rules
```

All four models must be run with the same `--question`, or `bench_report.py`
refuses to tabulate them side by side — correctly, since a model answering an
easier question is not a faster model. `hazard_warning` is the default and the
safest choice: Site Safety, no OCR, and the two trained classes are the only
ones a detector box can be scored against.

Choosing one of the four OCR questions (`spd_class_b_installed`,
`spd_class_c_installed`, `temp_within_limit`, `earthing_value_egb`) makes the
run more interesting and less clean: modes 1 and 2 get OCR evidence in the
prompt and mode 3 does not, by design, so the mode-3 column is measuring
something different from the other two. That asymmetry is the comparison, not a
bug — but say so wherever the numbers are quoted.

### One live call before the sweep

```bash
python tools/smoke_stage3.py <folder> --limit 1 --transport proxy
```

One photograph, one answer, end to end. If this works the benchmark will run;
if it 404s, 422s or 400s, re-read Step 2 before blaming the GPU.
---

## Step 3 — verify the prompt before spending GPU time

**Do not skip this.** A wrong VLM template does not raise. The model answers
fluently *as if it never saw the image*, and every number you then collect is
worthless. `/debug/prompt` renders the templated string without loading an
engine, so this costs seconds.

v6's `/debug/prompt` could not have caught it: its else-branch called the
InternVL builder for **any** non-qwen model, so pixtral and molmo would have
rendered an InternVL prompt here and a correct one at inference — the check
disagreeing with the thing it checks. v7 renders through the same
`_render_prompt()` the inference path uses, and returns `expect_placeholder`
so the answer below comes from the server rather than from this table.

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
| `molmo` | **no image placeholder** — Molmo's vLLM path inserts them from `multi_modal_data`. Check instead that your system prompt survived into the user turn, since Molmo has no system role, and v7 folds it in rather than dropping it. |

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

With `EAGER_LOAD=""` a fresh start is already empty; this is for the second and
later models in the sweep. `EVICT_GROUPS` also unloads the other VLMs on its
own when you load one, so the explicit unload is belt and braces rather than
the mechanism.

Zero the server's counters, so `/metrics` describes this run and not the
afternoon:

```bash
curl -s -X POST localhost:5432/metrics/reset | python -m json.tool
# or, from the client machine, which could not do this at all before v4:
curl -s -X POST http://10.19.71.246:8071/v1/metrics/reset \
     -H 'X-API-Key: secret-bench' | python -m json.tool
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
    --question hazard_warning \
    --timeout 600 \
    --scenario batch,continuous,parallel \
    --modes \
    --run-id 20260924
```

Transport, URL and key come from the environment exported in Step 2b; pass
`--transport proxy --gpu-url ... --api-key ...` instead if you would rather be
explicit. Keep `--question`, `--photos` and `--limit` **identical across all
four models** — the report checks each one and refuses to tabulate runs that
differ.

`--timeout 600` matters: the pipeline's own default is **180 s**, and the
proxy allows 600 precisely because a cold VLM load exceeds the old 120. A
client timeout shorter than the load records a successful cold load as a
failure, which is the one number in the report that is *supposed* to be large.

Roughly twenty minutes per model with these settings — batch is
`--repeats 2` over 8 photographs, continuous is 60 s at 0.5 rps, parallel is 20
requests at concurrency 4, and `--modes` adds three more passes over the same 8.
Drop `--scenario continuous` first if the window is tight; it is the only
open-loop pattern, so it is also the only one that shows a queue forming.

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
that is the semaphore, not the model.

`queue_wait_s` is what separates the two, and against v7 it is **measured**:
the server times the wait around the semaphore itself. v3 could only infer it
as proxy time minus server time, which is queueing *plus* network *plus* proxy
— on a fast, unloaded server almost entirely transport, so reading it as queue
wait reported saturation where nothing was queueing. The harness records which
kind it got, the report says so, and v4 also reports the leftover separately as
`transport_overhead_s`.

Molmo is configured `max_concurrent: 1`; a ramp on it saturates at 2.

### 4e. Move on

Unload, load the next, repeat. Order suggestion — cheapest first, so a mistake
in the setup surfaces on a 30-second load rather than a five-minute one:

```
pixtral  →  qwen3-vl  →  internvl  →  molmo
```

You can run `bench_report.py` after each one rather than waiting for all four.
It reports on however many models are done, so a sweep interrupted after two
still produces something readable — and a comparability warning appearing after
model two is much cheaper to act on than after model four.

---

## Step 5 — restore production

**This is the step people skip**, which is why v7 made it impossible to get
wrong by hand. The benchmark settings were never in the file, so restoring them
is unsetting the variables:

```bash
# on the GPU box, in the pane where it runs
unset EAGER_LOAD VLM_TP_INTERNVL VLM_GMU_INTERNVL VLM_GMU_QWEN3_VL
uvicorn gpu_api_server_v7:app --host 0.0.0.0 --port 5432 --workers 1
```

`MODEL_CONFIGS` on disk already holds the production values, and `EAGER_LOAD`
defaults to `qwen3-vl,mistral`. Nothing to revert, nothing to remember:

| Setting | Benchmark window | Back to, on its own |
|---|---|---|
| `EAGER_LOAD` | `""` | `qwen3-vl,mistral` |
| `qwen3-vl` `gpu_memory_utilization` | `0.85` | `0.60` |
| `internvl` `tensor_parallel_size` | `1` | `2` |
| `internvl` `gpu_memory_utilization` | `0.85` | `0.40` |

If you would rather go back to v6 entirely, `gpu_api_server_v6.py.keep` from
Step 1 is untouched. Staying on v7 costs nothing: `pixtral` and `molmo` are
lazy, so they hold no VRAM until requested, and `EVICT_GROUPS` only acts when a
VLM is loaded.

Restart, then confirm the original state:

```bash
curl -s localhost:5432/health | python -m json.tool
# loaded_models should be ["qwen3-vl", "mistral"]
```

Leave the proxy on v4. An allowlist entry for a model nobody requests has no
cost, and reverting it is how the next run starts with a confusing 422 that
does not mention the proxy. Its other changes — the forwarded sampling fields,
the passed-through queue wait — are strictly better for production traffic too.

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
| `404` on every call, server looks dead | `FIELDOPS_GPU_URL` points at the proxy while transport is still `direct`. `/infer` exists only on the GPU server — Step 2b |
| every photograph stops at detection in modes 1 and 2 | no annotation sidecars beside the photographs or in `data/labels/` — Step 2b |
| the report refuses to tabulate two models | they were run with a different question, photograph set, sampling or transport. Keep the flags identical across the sweep |
| `422 Unknown model 'pixtral'` | the proxy is still v3, or `AVAILABLE_MODELS` is set and narrow — Step 2 |
| `400 Model 'pixtral' is text-only` | the proxy's image caps have no entry for it, so its cap is 0 — Step 2 |
| `repetition_penalty` seems to have no effect | the proxy is still v3, which discards undeclared fields silently — Step 2 |
| queue wait looks enormous on an idle server | v3's approximation includes network and proxy time. The report flags this; v4 measures it — Step 4d |
| `503 Model 'x' is busy` during a ramp | working as intended: `max_concurrent` slots full for >60 s |
| `504` on the first request | cold load exceeded a timeout. Raise `--timeout`; the proxy itself allows 600 s |
| Model answers fluently but ignores the image | wrong prompt template. Step 3 |
| CUDA OOM on load | `gpu_memory_utilization` too high for what is already resident. Unload first, check `/health` |
| Run aborts: "server answered as X" | the model was never switched. That is the guard doing its job |
| Numbers look great and identical across models | check `resident_models` in the result file — you probably benchmarked one model four times |

Keep the tmux pane with the GPU server visible while running. `_load_one_async`
logs VRAM per GPU after every load, and that is the fastest way to see whether
a model fitted comfortably or scraped in. v7 also logs every environment
override as it applies it, so a run that behaves unlike the last one names the
reason in its own startup lines.
