# Benchmarking the pipeline

Two questions, and they need different machinery:

1. **Which of the four VLMs should we ship?** — qwen3-vl, pixtral-12b,
   internvl-38b, molmo-72b.
2. **What load does this survive, and where do the seconds go?**

Everything below is shaped by one hard constraint: **the GPU server holds one
model at a time.** A four-model comparison is therefore four separate runs with
a manual switch between them, possibly days apart. The tooling is built around
that rather than against it — one file per model, merged by a separate report
step, so a half-finished comparison is still readable.

---

## The plan

### Phase 1 — one model, end to end

For each model in turn: load it on the server, then

```bash
python tools/run_bench.py --model qwen3-vl \
    --photos <folder> --limit 8 \
    --scenario batch,continuous,parallel \
    --modes --run-id 20260924
```

Writes `bench_runs/20260924/qwen3-vl.json`. Switch the server to the next model
and repeat **with the same `--run-id`**, so the files merge.

### Phase 2 — find the saturation point

Separately, because it deliberately overloads the server:

```bash
python tools/run_bench.py --model qwen3-vl --photos <folder> \
    --scenario ramp --run-id 20260924 --yes
```

`--yes` is required. While this runs, anyone else using that GPU gets 503s.

### Phase 3 — merge

```bash
python tools/bench_report.py bench_runs/20260924
```

Writes `report.md` and `comparison.csv`. Run it whenever you like — it reports
on however many models are done so far.

---

## What gets measured

### The three arrival patterns

"Streaming" here means how requests **arrive**, not how tokens leave — the
pipeline calls `/infer`, which returns a complete answer.

**Correction, from reading the server:** the GPU server *does* expose
`POST /infer/stream` as SSE. It is the pipeline client that never uses it, not
the server that lacks it. So time-to-first-token is measurable with client
work only — no server change — and that is now a real option rather than the
impossibility this file first claimed. See "What is not measured" below.

| Scenario | Shape | The question it answers |
|---|---|---|
| `batch` | N requests back to back, one at a time | How long does a folder of photographs take? Cleanest per-request latency. |
| `continuous` | fixed rate for a fixed duration, **open-loop** | Can it keep up with a crew uploading steadily? The only pattern that shows a queue forming. |
| `parallel` | K in flight at once, **closed-loop** | What happens with K users? Reports throughput as well as latency. |
| `ramp` | parallel at K = 1, 2, 4, 8 … until it breaks | What is the safe concurrency limit? |

**Open-loop vs closed-loop is the important distinction.** Closed-loop load
(`parallel`) cannot overload a server: if it slows down, the client sends less,
and the system quietly finds an equilibrium that hides the problem. Open-loop
(`continuous`) keeps offering work regardless, so a queue builds and the latency
includes the waiting — which is what a real user experiences. A benchmark that
only does closed-loop reports a system as healthy right up until it collapses.

The catch is that an open-loop generator can fall behind itself. Every sample
records `schedule_lag_ms` and the result warns when it grows, because past that
point the harness is not offering the rate it claims.

### Per model

| Measure | Why it is there |
|---|---|
| **queue wait** | With `max_concurrent` of 2 on the VLMs, the third caller waits for a semaphore slot rather than being refused — so a rising queue wait against flat server time is saturation, long before any 503. `gpu_api_server_v7` times that wait **at the semaphore** and `llm_proxy_v4` passes the figure through. `llm_proxy_v3` could only approximate it as its own elapsed minus the GPU's, which is queueing plus network plus proxy; the harness records which kind it got and the report refuses to tabulate the two as one. |
| **output tokens / sec** | A model that writes longer answers takes longer. Comparing raw latency across models compares verbosity; tokens per second is what survives that. |
| **contract compliance** | Every prompt ends with a JSON contract, and `parse_vlm_answer` is tolerant in four descending tiers. A model that never emits valid JSON still produces answers and **looks fine** — it is one prompt edit from producing nothing. This is reported first, before any latency. |
| **unparseable rate** | Replies the parser gave up on arrive as `unknown`, indistinguishable from an honest `unknown`. Only the tier tells them apart. |
| **unknown rate** | A model that is unknown on everything has told you nothing. One that is never unknown is probably guessing. |
| **cold load** | The first call after a model switch loads the weights — tens of seconds for a 72B. Measured deliberately, reported on its own, and **excluded from every latency figure**. |
| **client vs server latency** | The response carries the server's own `elapsed_s`. The gap against client wall-clock is transport plus queueing — the number that answers "is the proxy expensive". |
| **grounding** | Mode 3 asks for coordinates. A model that cannot produce usable ones cannot do mode 3, whatever its speed. |

### Per photograph

`--modes` runs the full three-mode pipeline and records where the time goes
across quality, detection, OCR and the model calls. Those stage timings are new
— `QualityStageResult.elapsed_ms` and `DetectionStageResult.elapsed_ms` — and a
stage reporting nothing is listed as **not measured**, never as instant.

Remember modes 1 and 2 share one model call and mode 3 makes three, so compare
per-photograph wall time, not per-call latency, across modes.

---

## What the harness refuses to do

A benchmark's failure mode is not crashing. It is producing a confident number
that is wrong, which nobody catches because it looks like every other number.

- **A failed request is not a fast request.** Errors, 503s and timeouts never
  enter the latency distribution. Counting a 4 ms connection refusal as a 4 ms
  response makes a saturated server look quick. They are counted separately and
  the result says so.
- **A 503 is load, not breakage.** It gets its own outcome, because "the server
  was busy" and "the server is broken" are opposite findings.
- **A mock is not a measurement.** `--mock` runs the harness with no GPU. Every
  sample is tagged, the timings appear only under
  `harness_only_timing_NOT_latency`, and the report excludes those files from
  the ranking entirely.
- **Percentiles are nearest-rank**, so the number printed actually occurred.
  Under 100 samples a caveat is attached to p95 and above; under 20, it says
  percentiles above p50 are not meaningful.
- **Models measured under different settings are not tabulated silently.** The
  report checks question, sampling, photograph set and transport, and says so.

### The guard that matters most

The server serves one model at a time. The failure that costs you the whole
exercise is forgetting to switch it: you run `--model molmo-72b`, the server is
still on qwen3-vl, and you get a complete, plausible, entirely wrong file.
Nothing in the numbers would ever tell you.

So every run does two checks before measuring anything — the registry must
**list** the model, and the warm-up response must come back **attributed** to
it. A mismatch aborts. A server that lists four models but has one resident
passes the first check and fails the second, which is exactly this setup.

---

## Reading the results

Take them in this order:

1. **Contract compliance and unparseable rate.** A model that is `UNUSABLE` or
   `fragile` here is not a candidate, however fast. Stop.
2. **Cold load.** This is what the first photograph of a demo costs. For a 72B
   after a switch it may dominate everything else.
3. **batch p50 / p95.** The honest per-photograph cost with nothing else going on.
4. **Saturation point.** How many concurrent users before 503s.
5. **Disagreements.** With no ground truth, agreement is weak evidence of
   correctness but disagreement is strong evidence that someone is wrong. The
   split list is a shortlist to look at by eye — that, not any percentage, is
   the real product of a label-free comparison.

**None of these numbers say a model is right.** There is no ground truth here.
They say whether a model is usable, self-consistent and comparable — which for
choosing what to ship is usually the decisive question, but it is a different
one, and the difference should be stated whenever these results are quoted.

---

## Before you run it against a shared GPU

- `parallel` and `ramp` deliberately saturate the server. Anyone demoing at the
  time will see 503s. `ramp`, and any concurrency above 8, require `--yes`.
- **Contention invalidates everything.** If someone else is using that GPU, the
  numbers describe a machine under unknown load. Check first.
- Raise `--timeout` for a cold load of a large model — the default will time
  out on a 72B's first call and record it as a failure.
- A full sweep is not cheap. Four models × (batch + continuous + parallel +
  ramp + three modes), plus four cold loads, plus the switching between them.
  Budget an afternoon, and run `ramp` when nobody needs the box.

---

## What is not measured, and why

| Not measured | Why |
|---|---|
| **Accuracy** | No ground truth. Add a labels file and this becomes possible; until then, agreement and compliance only. |
| **Time to first token** | The server **does** offer `POST /infer/stream` (SSE); the pipeline client does not use it, so the harness does not either. Adding it is client-side work only: consume the event stream, record the first `data:` chunk into a new `Sample.first_token_ms`. Worth doing if perceived responsiveness matters more than total latency — on a 72B it is the difference between "slow" and "frozen". |
| **Transport overhead, against an older proxy** | `transport_overhead_s` — the round trip minus the server's own time minus the queue wait — needs a *measured* queue wait to subtract, so `llm_proxy_v4` reports it and nothing else can. Against `llm_proxy_v3` the two quantities are fused into one approximation and cannot be separated after the fact. Queue wait itself is no longer proxy-only: `gpu_api_server_v7` returns it on the direct path too. |
| **VRAM, GPU utilisation, cost** | Not visible from an HTTP client. Read them on the server during a run. |
| **Quality of the reasoning text** | Only its parseability and length. Judging the prose needs a human or a second model, and both are separate exercises. |
| **Detector accuracy** | `tools/smoke_yolox.py` covers that; this is about the model leg. |

---

## Files

| Path | Role |
|---|---|
| `tools/run_bench.py` | one model, one run, one file |
| `tools/bench_report.py` | merge the files into a comparison |
| `backend/bench/metrics.py` | samples, percentiles, and what may enter them |
| `backend/bench/scenarios.py` | the arrival patterns and the ramp |
| `backend/bench/harness.py` | the real client, the model guard, cold-load warm-up |
| `backend/bench/quality.py` | compliance, decisiveness, agreement |
| `backend/bench/capacity.py` | will this model fit on this card, and what is left for KV |
| `tests/test_bench.py` | assertions, mostly about refusing to mislead |
| `BENCHMARK_RUNBOOK.md` | **the actual procedure** for these boxes, model by model |
| `deploy/gpu_api_server_v7.py` | drop-in replacement for the v6 GPU server: pixtral and molmo, environment-overridable settings, measured queue wait, `/metrics/reset` |
| `deploy/llm_proxy_v4.py` | drop-in replacement for the v3 proxy: all four VLMs allowlisted, image caps as data, forwarded sampling fields, queue wait passed through |
| `deploy/gpu_models_bench.py` | the annotated diff behind v7, kept for reading what changed and why — no longer the procedure |

Output lands in `bench_runs/<run_id>/` — one `<model>.json` per model, plus
`report.md` and `comparison.csv` once merged. It is gitignored: these are
measurements of one machine on one afternoon, not source.
