# Changelog

What changed, when, and against which commit. Dates are commit dates.

Versions are milestones in the demo's development, not released artefacts —
nothing here is published or installed from a package index. `FROZEN.md`
records the one commit that is pinned as a fallback.

Reasoning lives in `DECISIONS.md`; this file is the record of *what* and
*when*. Assertion counts are the whole suite after that change.

---

## v1.1 — Benchmark runbook for the real boxes · 2026-09-24

| Change |
|---|
| `BENCHMARK_RUNBOOK.md` — step by step against 10.66.98.137 and the proxy |
| `deploy/gpu_models_bench.py` — registry entries and prompt builders to paste into the GPU server |
| `backend/bench/capacity.py` — will a model fit on a card |
| the harness records `queue_wait_s`, output tokens, and the serving config |

Written after reading `gpu_api_server_v6.py` and `llm_proxy_v3.py`. Three
findings changed the plan.

**Pixtral and Molmo are not in the server's registry**, so no load command
works today: they need new `MODEL_CONFIGS` entries *and* new prompt builders,
since `_build_engine_input` knows only `raw`, `qwen_vl` and `internvl`. The
proxy rejects them twice more — `AVAILABLE_MODELS` (422) and `MODEL_IMAGE_CAPS`
deriving `VISION_MODELS` (400 "is text-only").

**Molmo-72B cannot run on one H200 in bf16.** 72B x 2 bytes is 144 GB of
weights against a 141 GB card, over the limit before any KV cache. The entry
uses FP8, which is the only way to honour one-GPU — and makes its answer
quality not strictly comparable with the bf16 models, and its latency
flattering. `capacity.py` makes that arithmetic executable so it survives a
hardware change.

**`internvl` ships TP=2 @ 0.40**, spanning both GPUs. On one GPU it needs TP=1,
and 0.40 is then too small for 76 GB of weights — both values change together
or the load fails.

Two corrections to v1.0: the GPU server **does** expose `POST /infer/stream`
(SSE), so time-to-first-token is client-side work rather than impossible; and
model switching is scriptable through `/models/{name}/load` and `/unload`,
though not through the proxy.

The harness now records `queue_wait_s` — the proxy's own saturation signal, and
with `max_concurrent` of 2 the thing that separates "slow" from "busy" — plus
output token counts, so latency comparisons are not just comparing verbosity,
and the serving config (TP, gpu_memory_utilization, resident models) beside
every number.

**947 assertions, fourteen suites** (953 with every dependency, 769 bare).

---

## v1.0 — Benchmarking · 2026-09-24

| Change |
|---|
| `backend/bench/` + `tools/run_bench.py` + `tools/bench_report.py` |
| `BENCHMARKS.md` — the plan, and how to read the results |
| per-stage timings on the quality and detection results |
| `parse_vlm_answer_tiered()` — which tier of the parser rescued a reply |

Built for comparing qwen3-vl, pixtral-12b, internvl-38b and molmo-72b, and for
load-testing the pipeline. Shaped throughout by the constraint that **the GPU
holds one model at a time**: one run per model, one file per model, merged by a
separate report step, so a half-finished comparison is still readable.

**Three arrival patterns**, since there is no token streaming anywhere —
`/infer` returns a complete answer. `batch` (sequential), `continuous`
(open-loop at a fixed rate, the only one that can show a queue forming), and
`parallel` (closed-loop at K workers), plus a `ramp` that raises concurrency
until latency or 503s cross a ceiling and then stops.

**Most of the package is about refusing to produce a misleading number.**
Failed requests never enter a latency distribution — counting a 4 ms connection
refusal as a 4 ms response makes a saturated server look quick. A 503 is its own
outcome, because busy and broken are opposite findings. Mock runs are tagged and
excluded from the ranking. Percentiles are nearest-rank and carry a caveat below
100 samples. Models run under different questions, settings or photograph sets
are flagged rather than tabulated silently.

**The guard that matters most:** on a one-model-at-a-time server, forgetting to
switch produces a complete, plausible, entirely wrong file. Every run checks the
registry lists the model AND that the warm-up reply is attributed to it, and
aborts otherwise. Cold model load — tens of seconds for a 72B — is measured
deliberately and excluded from every latency figure.

**Contract compliance is reported before speed.** `parse_vlm_answer` is tolerant
in four tiers, so a model that never emits valid JSON still produces answers and
looks fine; it is one prompt edit from producing nothing. The parser now reports
which tier rescued each reply, so that is visible.

Verified against four synthetic models with deliberately different characters
driven through the real scenario machinery: the fast compliant one, one rescued
by parser tolerance, one that saturates at concurrency 4, and one that 503s ten
of twelve parallel requests. Every figure matched the profile it was given.

**933 assertions, fourteen suites** (939 with every dependency, 755 bare).

---

## v0.9 — Overlay control, and a frontend/backend split · 2026-09-23

| Change |
|---|
| a three-way overlay control for mode 3's claimed boxes, switchable without re-running |
| `app/` split into `backend/` (the pipeline) and `frontend/` (the four Dash UIs) |
| `WALKTHROUGH.md` — how to read this repository |

**The overlay control.** A claimed box's caption is long enough to cover the
object it points at — on a device reading it sits over the digits, which
defeats the point of pointing. The overlay is now box + label, box only, or
off. All three renderings are written during the run, so switching is a
re-render and never three more model calls; the control sits with the mode
selector rather than in the sidebar, where everything forces a re-run. A render
that *fails* is now distinguished from a record that predates the feature,
because falling back to the plain photograph silently reads as "the model
claimed nothing".

**The split.** `backend/` holds the pipeline, `frontend/` the UIs, and the
arrow points one way: nothing under `backend/` imports anything from
`frontend/`. Verified by running the full pipeline — three modes, OCR included
— from a script with `dash` and `gradio` hard-blocked at import. A parsed (not
grepped) guard in `test_phase1.py` keeps it that way; grepping failed on a
docstring that merely *names* a UI, which is documentation and welcome.

`backend/` and `frontend/` sit at the same depth `app/` did, so every
`__file__`-derived path — `config.py`'s `parents[2]`,
`foreground_segmentation.py`'s `parent.parent` — resolves unchanged. That was
the reason for choosing this layout over `src/`. No compatibility shims: the
systemd units, tools, tests and every document moved in the same commit.

**816 assertions, thirteen suites** (822 with every dependency installed, 645
bare).

---

## v0.8 — Mode 3 points at things · 2026-09-23

| Commit | Change |
|---|---|
| (this) | mode 3's presence and answer legs are asked for coordinates, and what comes back is drawn and scored |

**Reverses "mode 3 draws no bounding boxes"** (`MODES.md`, v0.6). The old
reasoning — Qwen3-VL's grounding is well below YOLOX's, and a visibly wrong box
is worse than an honest "presence, not geometry" — is still true. What changed
is that the weakness can now be *measured on screen* instead of asserted in a
document: modes 1 and 2 have already run the detector over the same photograph
in the same frame, so each claimed box carries an IoU against it.

Three conditions keep it honest. The boxes are drawn **dashed** in alternating
white and black, a visual language nothing else in the demo uses, and captioned
`model says: …`. They carry their IoU, or say *"no trained detector box to
compare against"* — which is not the same as scoring zero, and is the case for
every Infra question today. And they live in `claimed_boxes`, never in
`detections`, so a claimed region cannot reach a prompt and have the model cite
itself as evidence.

**The coordinate trap.** `array_to_data_uri()` downscales to 2048px, so the
model never sees the original frame and an absolute pixel reply is in the
*resized* one — drawing it on a 4000×3000 photo is out by ~2× with nothing
raising. The prompts ask for normalized 0–1000 (immune to the resize, and the
convention Qwen-VL was trained on); `vlm_grounding.interpret_box()` accepts
fractions and absolute pixels too, scales them from the encoded frame, and
prints which reading it used so a systematic misread is visible.

Boxes that cannot be placed confidently are **refused with a reason** — a
whole-frame box especially, which is the model declining to localise while
appearing to comply.

**Not grounded:** the quality leg (sharpness and exposure are whole-frame
properties, and there would be nothing to score a box against), and mocks,
which claim no geometry at all.

Also: a `vlm_grounding` config flag and a UI toggle on 7873; an evidence-region
panel separating "where the subject is" from "what I looked at to decide";
three new CSV columns (`vlm_boxes`, `vlm_box_convention`, `vlm_box_best_iou`).

**798 assertions, thirteen suites** (804 with every dependency installed, 627
bare). One existing assertion changed — `test_modes`'s "states no boxes are
drawn" — replaced by assertions covering both the box and the no-box case.

---

## v0.7 — Domains, OCR and the plugin layer · 2026-09-22

| Commit | Change |
|---|---|
| `3b4761a` | questions, domains and detector classes move from Python to `config/` |
| `0a10ce0` | OCR added as stage 2b, between detection and the model |
| `a5ac5df` | `frontend/demo_dash_pipeline.py` on port 7873; mode 3's OCR mis-framing fixed |

**Scope.** Two questions became sixteen, in two domains. The fourteen new ones
are the site-infrastructure checklist: lightning arrestor, enclosure condition
inside and out, Roxtec sealing, Class B and C SPD installed and active,
rectifier modules, temperature sensor placement and reading, earth pit and
earthing value, DCDB cable tagging.

**The plugin layer.** `config/domains.yaml`, `config/classes.yaml` and
`config/questions/*.yaml`, read by the new `backend/pipeline/registry.py`. Adding
or removing a question, domain or object class is a YAML edit —
`PLUGINS.md` is the procedure, and the **Reload config/** button on 7873 picks
it up without a restart. `backend/pipeline/questions.py` remains the import surface,
so every existing import still resolves. `yolox_class_names` is now derived
from the YAML's explicit `yolox_index` values.

**Stage 2b — OCR.** PP-OCRv6 tiny through RapidOCR, over ONNX files committed
under `models/ocr/`. Runs for the four questions whose YAML sets
`ocr.enabled` — the two SPD "installed" questions and the two device readings —
and reads the relevant detected boxes, falling back to the whole frame when
detection finds nothing. Its text and any threshold check reach the model as
*advisory evidence*; the pipeline never answers from them.

**Mode 3 runs no OCR**, deliberately: on a device reading its answer is the
model reading the display unaided, which is the comparison the new screen
exists to show. Running the app surfaced a real bug here — all four OCR
questions were telling mode 3 that OCR text "may be supplied", which in that
mode it never is. Questions now carry a `system_prompt_no_ocr` for that leg.

**New UI.** `frontend/demo_dash_pipeline.py` on **7873**: domain → question → one
photograph → all three modes side by side, with an OCR panel and a Prompts tab
showing exactly what each mode sent. 7872 stays current as the batch screen;
7870 stays frozen.

**Also.** `tools/smoke_ocr.py`; `preflight.py` gains config-tree and OCR-model
sections and an `--offline-check` that builds the OCR engines with the network
denied; `deploy/fieldops-demo-pipeline.service`; `rapidocr` added to
`requirements-demo.txt` as a soft dependency, and `pyyaml` reclassified as
required.

**711 assertions, twelve suites** on a host without the heavy dependencies
(up from 459 across nine); 717 with everything installed, and 540 on a bare
interpreter where the pyyaml-dependent sections skip themselves rather than
crash. Two existing
assertions changed, both pinning facts this work deliberately changes:
`test_phase1`'s "two questions registered", and `test_yolox`'s ban on a second
definition of the class names — replaced by a stricter check that pins the
YAML, `config.py`'s fallback and `registry.py`'s fallback against each other.

---

## v0.6 — Deployment · 2026-09-18

| Commit | Change |
|---|---|
| `aa1146a` | systemd unit and `SERVICE.md` for the modes demo on port 7872 |

- `deploy/fieldops-demo-modes.service` — starts at boot, restarts on failure,
  logs to journald.
- `deploy/fieldops-demo.env.example` — the proxy credentials, kept out of the
  world-readable unit file. `.gitignore` refuses a filled-in copy.
- `SERVICE.md` — install, the no-root user-service path, firewall, update loop,
  and what the service does not survive (`demo_runs/` growth, ownership of the
  project directory, prompts read once at import).

No application code changed. **459 assertions, nine suites.**

---

## v0.5 — The field-ops VM · 2026-09-16 → 2026-09-17

Moved the demo to `10.19.75.122`, which cannot reach the GPU server at all.

| Commit | Date | Change |
|---|---|---|
| `48caf4f` | 09-17 | Uploads accumulate instead of replacing |
| `dc2135f` | 09-17 | Dropzone stops silently discarding dropped photographs |
| `419282c` | 09-17 | Two tests stop depending on the machine they run on |
| `f591bef` | 09-16 | Reach the VLM through `llm_proxy_v3` |

### Added
- **Second transport.** `vlm_transport` (`direct` | `proxy`), `vlm_api_key`,
  `FIELDOPS_*` environment overrides, a UI radio and key field, `--transport`
  on `preflight.py` and `smoke_stage3.py`. New `PROXY.md`.
- **Upload accumulation.** Staged-to-disk photographs in a `dcc.Store`, a
  "Clear photos" button, per-drop reporting of what was ignored and why.
- `tests/test_transport.py` — 54 assertions with `requests` stubbed by a
  recorder, so the assertions are about what is *sent*.

### Fixed
- **Dropzone lost most dropped files.** `accept="image/*"` filters on the MIME
  type the browser reports; a file from a network share arrives with none and
  react-dropzone discarded it without a word. Reproduced in Chromium — three
  dropped, one accepted. `accept` now lists extensions too, from the same set
  `collect_images()` uses. Affected `demo_dash_modes.py` and
  `demo_dash_yolox.py`; the frozen `demo_dash.py` never set `accept` and so
  never had it.
- **Two host-dependent tests.** One read `FIELDOPS_*` through
  `default_config()`, so on a proxy-configured host "a proxy with no API key"
  became a proxy *with* one. The other popped `yolox.models` from `sys.modules`,
  which clears the cache but does not stop a re-import — it passed only where
  the package was genuinely absent.
- A filename carrying a path traversal is flattened to its basename before it
  reaches disk. A staged photograph that disappears before Run is reported
  rather than vanishing from the batch.

### Changed
- `_get` raises the same `HTTP <code> from <path>` shape as `_post`, so a 401
  on the registry read gets the same explanation a POST would.
- Proxy statuses carry their meaning: 502 is "the proxy is up and the GPU
  behind it is not", 503 is load rather than breakage, 404 is almost always
  the wrong transport.

**459 assertions, nine suites** (was 379 across eight).

---

## v0.4 — Three modes · 2026-09-15 → 2026-09-16

| Commit | Date | Change |
|---|---|---|
| `61d5df5` | 09-16 | Mode 3 renders images; per-leg system prompts |
| `bf89b6d` | 09-16 | Three-mode version of the demo UI |
| `477ba5e` | 09-15 | Open-source inventory |

### Added
- `frontend/demo_dash_modes.py` on **port 7872** — mode 1 (quality gate → detector →
  model), mode 2 (quality **OR** detector → model), mode 3 (everything by the
  model). One run fills all three; switching mode re-reads results rather than
  re-running. A Compare tab puts one row per photograph against one column per
  mode.
- `backend/pipeline/modes.py` — the shared-work runner. One quality pass, one
  detection pass, one VLM answer when modes 1 and 2 both proceed.
- `backend/pipeline/prompts.py` — `PromptStore`, persisting mode-3 overrides to
  `config/prompts.yaml`.
- Mode 3 split into three calls with editable per-leg system prompts, a fixed
  user prompt shown read-only, and a "Re-run mode 3 only" button that reuses
  modes 1 and 2's results.
- Mode 3 renders the plain EXIF-corrected photograph — deliberately no
  foreground box, since u2netp never runs there.
- `MODES.md`, `OPEN_SOURCE.md`.

**379 assertions, eight suites.**

---

## v0.3 — Trained detector · 2026-09-10

| Commit | Change |
|---|---|
| `820ebc4` | Image upload in the YOLOX UI |
| `39728ce` | `smoke_yolox` stops overriding the configured class names |
| `674113c` | Class order corrected from `classes.json` |
| `c6e9338` | `preflight --port` |
| `860f7cd`, `1ce5613` | torch install: proxy TLS, and dropping `--extra-index-url` |
| `0d29e85` | `.gitignore` anchored so vendored source survives |
| `d1b9834` | Default checkpoint path; `COPY_FROM_AISERVER.md` |
| `6d96dab` | YOLOX-S detector as a second Dash version |
| `d613e80` | Checkpoint inspector |

### Added
- `frontend/demo_dash_yolox.py` on **port 7871** — the trained YOLOX-S checkpoint in
  place of annotation sidecars.
- `backend/pipeline/yolox_runtime.py` — letterbox `preproc`, `build_model`,
  `YoloxPredictor`, all mirroring the training exp.
- `backend/vendor/yolox/` — utils under Apache 2.0. `yolox/models/` is *not*
  tracked: the upstream repo's bare `models/` gitignore pattern excluded it.
- `tools/inspect_ckpt.py` — reads class count, width and depth out of a
  checkpoint and says what they imply.
- `YOLOX_SETUP.md`, `COPY_FROM_AISERVER.md`.

### Fixed
- **Class order.** Reported as the detector swapping classes; the config was
  right and `smoke_yolox.py`'s `--classes` default was overriding it. Two guard
  tests added.
- **Missing upload widget** in the YOLOX UI — the annotation version's sidecar
  constraint had been carried over, but a trained detector needs no sidecar.

**Verified on FALCONPRD:** 2.4–6.7s per photograph on CPU, boxes matching the
human annotations at 0.89–0.94 confidence.

---

## v0.2 — Dash UI, frozen · 2026-09-09

| Commit | Change |
|---|---|
| `cf23a9a` | Record the `dash-ui-v1` freeze |
| `f978b7d` | **Rebuild the UI in Dash, styled to Design System V.01** |

`frontend/demo_dash.py` on **port 7870**, with `frontend/assets/demo.css`. Verified end
to end on FALCONPRD against the live GPU server, and **frozen** at `f978b7d` as
the known-good fallback — see `FROZEN.md`. Later versions are separate files.

A local git tag `dash-ui-v1` marks it; the tag is not on the remote (the
environment's proxy rejects tag pushes with 403), so the SHA is the durable
reference.

**229 assertions, six suites.**

---

## v0.1 — The pipeline · 2026-09-09

| Commit | Change |
|---|---|
| `cd5214d` | Phase 4: orchestrator and the integrated demo UI |
| `9623abb` | Phase 3: VLM client, prompt wiring, tolerant parser, mock fallback |
| `72c13b1`, `4aa7be2`, `06d819e` | Label resolution: beside the photo, per question, at any depth |
| `a08c823` | Phase 2: annotation-file detector, shared box drawing, label tooling |
| `8491b13` | rembg 2.0.69 compatibility |
| `eb17da8` | Stage 1 smoke test |
| `c5a6764` | Vendor `quality_check` + `foreground_segmentation`; setup and requirements |
| `9e5da32` | preflight: gate the offline check on rembg |
| `c4c7ee4` | Phase 1: schemas, config, question registry, quality stage |

### Added
- `backend/pipeline/` — `schemas.py` (stage contracts), `config.py`, `questions.py`
  (per-question system prompts and class names), the three stages, and
  `orchestrator.py`.
- `tools/preflight.py` — checks a demo host before the demo, not a dev box.
- `SETUP.md`.

### Fixed
- **rembg 2.0.69** builds its own `SessionOptions` and passes it positionally,
  so passing `sess_opts` — even `None` — raised `got multiple values for
  argument 'sess_opts'`. A thread cap now becomes `OMP_NUM_THREADS`.
- **Sidecar labels not found.** `resolve_annotation_dir` used non-recursive
  `glob` while images used `rglob`. Labels now resolve per image, beside the
  photograph first.
- **Two single-class CVAT exports both numbering class 0** — resolved with
  per-question `default_class_names`.

**Verified on FALCONPRD:** stage 1 ~2s per photograph, segmentation 3/3; stage 3
0.21–0.48s server-side with reasoning citing real sign text.

---

## Before this work

Commits before `c4c7ee4` (2026-07-21 and earlier) are unrelated Instant Graph
MCP server work — `instant_graph_mcp_server_v2_4_1.py` at the repo root.
