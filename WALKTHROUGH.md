# Reading this repository

A guided path through the code, in the order that makes it make sense. Written
for three readers: whoever comes back to this months later, someone new to the
project, and anyone who only wants to change a question and not touch code.

If you only want to add or remove a **question, domain or object class**, you
do not need this file. Go to **`PLUGINS.md`** — that is a YAML edit, and no
Python changes.

---

## The one-paragraph version

A photograph and an inspection question go in; `YES` / `NO` / `UNKNOWN` plus
reasoning comes out. Between them are four stages — a quality gate, an object
detector, OCR, and a vision model — and **three modes** that argue about how
much to trust each stage. The interesting part of the demo is not any one
answer; it is the same photograph answered three ways and the places where the
three disagree.

```
photograph ─▶ 1. Quality ─▶ 2. Detection ─▶ 2b. OCR ─▶ 3. Answer ─▶ YES/NO/UNKNOWN
              MM-IQA        YOLOX-S or      PP-OCRv6   Qwen3-VL      + reasoning
              + u2netp      annotations     (4 qs)
```

---

## Layout

```
config/      the questions, domains and object classes — YAML, no code
backend/     the pipeline. Runs with no UI installed.
frontend/    four Dash UIs. Import backend/; backend never imports them.
tools/       preflight + one smoke script per stage
tests/       thirteen standalone suites — no pytest, no build step
models/      u2netp.onnx and ocr/*.onnx (committed); best_ckpt.pth (copied in)
```

**The arrow points one way.** `frontend/` imports `backend/`, never the
reverse. That is what lets the whole pipeline run from a script on a machine
with no Dash — which is how the tests, the smoke tools and any edge-device port
consume it. `tests/test_phase1.py` enforces it by parsing imports.

---

## Read these six files, in this order

About 1,900 lines. After them the rest of the repo is detail.

### 1. `backend/pipeline/schemas.py` (395 lines) — start here

Every dataclass each stage fills and every renderer reads. Read it first
because it is the contract: change a field here and you touch every stage and
all four UIs.

Four results hang off one `PipelineRecord`: `quality`, `detection`, `ocr`,
`vlm`, plus `stopped_at` saying where the photograph stopped.

Watch for the fields that exist to stop a *silent* lie:

- `QualityStageResult.resolution_ok` / `.fail_kind` — an image can score 82
  against a threshold of 65 and still FAIL, purely on a resolution floor.
  Without these the UI renders `FAIL 82.1 / 65`, which reads as a scoring bug.
- `DetectionStageResult.is_stub` — set by the detector, never inferred. An
  earlier version looked for `"model"` inside `model_name`, which is a
  substring of `"no model loaded"`, so stub results were reported as real.
- `OCRStageResult` — three different nothings, deliberately distinguishable:
  the stage *could not run*, it *ran and read nothing*, it *was never asked*.
  Only the first is a problem with the host.
- `DetectionStageResult.claimed_boxes` — mode 3's boxes, kept **out** of
  `detections` so they can never be fed back to the model as evidence.

### 2. `config/questions/infra.yaml` + `backend/pipeline/registry.py` (566)

Where the questions actually live, and the loader that reads them. Skim one
question entry, then read the registry's docstring.

Two rules in the loader carry real scars:

- **Loading never raises.** A malformed file must not take the demo down five
  minutes before it runs, so every survivable problem becomes a warning the UI
  prints and the bad entry is dropped.
- **`yolox_index` is written per class, not inferred from list order.** YOLOX
  stores no class names in a checkpoint, so `config/classes.yaml` is the only
  record of them, and a wrong *order* does not error — it puts a confident
  wrong label on screen and into the model's prompt as evidence. It has already
  happened here once.

`backend/pipeline/questions.py` (269) is the stable import surface over all of
this. Import from it, not from `registry.py` or `question_types.py`.

### 3. `backend/pipeline/modes.py` (476) — the spine

`run_all_modes()` is the function to follow. One loop over photographs, and
inside it the whole pipeline in order. Follow these lines:

| Line | What happens |
|---|---|
| `load_image_bgr(path)` | decoded **once**, EXIF-corrected, and passed as an array everywhere after |
| `score_image(...)` | stage 1 |
| `detector.detect(...)` | stage 2 |
| `select_relevant(...)` | which detections this question cares about |
| `stage2b_ocr.run(...)` | stage 2b — only for questions whose YAML enables it |
| `client.ask(...)` | modes 1 and 2's answer, **computed once and shared** |
| `client.ask_vlm_only(...)` | mode 3, three separate calls |

The thing to notice is how little is repeated. One quality pass, one detection
pass, one OCR pass; modes 1 and 2 differ only in what they *do* with the same
results, and when both gates open the model is asked once. That sharing is why
running all three costs about four model calls per photograph rather than
nine.

### 4. `backend/pipeline/stage3_vlm.py` (847) — the model boundary

The biggest file, and the one with the most hard-won detail. Read in this
order:

1. `ENDPOINTS` — the *only* place that knows whether we are talking to the GPU
   server directly or through the proxy. Every path difference lives here.
2. `array_to_data_uri()` — and note it **downscales to 2048px**. The model
   never sees your original frame. This one fact is behind the whole of
   `vlm_grounding.py`.
3. `build_payload()` — refuses an empty registry and a blank system prompt,
   because both fail *silently* at the server and look like model-quality
   problems.
4. `ask()` — modes 1 and 2.
5. `ask_vlm_only()` — mode 3's three legs.
6. `mock_answer()` / `mock_vlm_only()` — what happens when the server is
   unreachable. Always `unknown`, always labelled. **Never let a mock assert.**

### 5. `backend/pipeline/vlm_grounding.py` (363) — mode 3's boxes

Turns coordinates the model wrote into a rectangle worth drawing. Most of the
file is about refusing, not drawing: a box covering the whole frame is the
model declining to localise while appearing to comply, and a wrong box still
looks authoritative on a photograph.

`interpret_box()` is the core. It exists because of the 2048px resize above —
absolute pixel coordinates are in the frame the model *saw*, and drawing them
on a 4000×3000 original is out by ~2× with nothing raising.

### 6. `frontend/demo_dash_pipeline.py` (895) — the current UI

Read `_panel()` and `detail_card()` and you have the whole render path. The
rest is controls.

The renderers are **imported** from `demo_dash.py` and `demo_dash_modes.py`
rather than copied, on purpose: a real detection and a human annotation must
look identical on screen, or the audience learns to read the styling instead of
the provenance banner.

---

## Following one photograph all the way through

Pick the temperature question, because it touches every stage including OCR.

1. **Upload.** `on_upload()` writes the file to disk immediately and stores the
   path. The original filename's stem is captured here and never re-derived —
   it is the join key across every stage, and both of the older renaming paths
   in this codebase destroyed it.
2. **Run.** `execute_run()` builds a `PipelineConfig` from the controls and
   calls `run_all_modes()`. Only the Run button reaches here; changing a tab,
   a mode or the overlay re-renders from results already computed.
3. **Decode once.** `load_image_bgr()` applies EXIF transposition. Everything
   downstream gets *that array*. Re-reading the file with `cv2.imread` would
   give a different orientation and every box would be silently wrong.
4. **Stage 1.** u2netp finds the foreground, MM-IQA cues score it. Mode 1
   treats a FAIL as fatal; mode 2 treats it as advisory.
5. **Stage 2.** No trained weights exist for the Infra classes, so the
   annotation-sidecar backend runs and labels itself a stub. "The detector
   found nothing" and "nothing was ever trained to find this" look identical on
   a card — only one is evidence, which is why the question panel names the
   untrained classes.
6. **Stage 2b.** OCR reads the relevant boxes, or the whole frame when there
   are none. The numeric rule matches `31.2` and checks it against 35 °C. That
   result is **evidence in the prompt, never the answer** — the rule cannot see
   a range multiplier, tell a set-point from a measurement, or tell °F from °C.
7. **Stage 3.** Modes 1 and 2 get one call with the detection block and the OCR
   block. Mode 3 gets three calls and **no OCR at all** — it reads the display
   itself, and the gap between the two is the most useful thing on the screen.
8. **Render.** Four panels per mode, plus the three-mode comparison row.
9. **Write.** `demo_runs/<run_id>/<mode>/pipeline_results.csv` and
   `results.json`.

---

## Where each kind of change goes

| You want to… | Go to |
|---|---|
| add/remove a question, domain, object class | `config/*.yaml` — see `PLUGINS.md` |
| change what a stage *computes* | `backend/pipeline/stage*.py` |
| change what a stage *returns* | `schemas.py` first, then every reader of it |
| change what the screen *shows* | `frontend/demo_dash_pipeline.py` |
| change how the model is reached | `stage3_vlm.py`'s `ENDPOINTS` — nothing else |
| change a threshold or a path | `config.py`, or a `FIELDOPS_*` env var |
| understand why something is odd | `DECISIONS.md` |

---

## Conventions worth knowing before you edit

**Comments explain *why*, and several carry the symptom of a failure that
actually happened.** They are not noise; they are the bug report. If a comment
says something reads as a dead server or a model-quality problem, someone lost
an afternoon to exactly that.

**Nothing asserts on no evidence.** Mocks return `unknown`. A quality verdict
with no numeric score says so rather than printing `0.0`. A claimed box with
nothing to compare against says "no trained detector box to compare against",
which is *not* the same as scoring zero. If you find yourself rendering a
confident-looking default, that is the bug.

**The tests are standalone scripts**, not pytest. Each prints `N passed,
M failed` and exits non-zero. A section needing a real dependency must skip
cleanly and say so — a traceback breaks that contract.

```bash
for t in tests/test_*.py; do python "$t" >/dev/null || echo "FAILED $t"; done
python tests/test_modes.py          # one suite, with its output
python tools/preflight.py --skip-gpu
```

---

## Running it

```bash
python tools/preflight.py --port 7873        # check the host first
python frontend/demo_dash_pipeline.py        # http://<host>:7873
```

Then: pick a domain, pick a question, drop a photograph, press **Run all three
modes**. With no GPU server reachable you still get a full run — every answer
comes back a clearly labelled `unknown` mock, which is the point.

---

## The map of the other documents

| File | For |
|---|---|
| `README.md` | the front door — what this is, the four ports, quick start |
| `PLUGINS.md` | **adding or removing a question, domain or class** |
| `MODES.md` | what the three modes argue, what they cost, mode 3's boxes |
| `DECISIONS.md` | why things are the way they are, and what each choice costs |
| `CHANGELOG.md` | what changed, when |
| `CLAUDE.md` | orientation for a Claude Code session; has the trap list |
| `SETUP.md` / `SERVICE.md` / `PROXY.md` | installing, running as a service, the proxy |
| `FROZEN.md` | the pinned fallback UI and how to restore it |
| `OPEN_SOURCE.md` | every component, version and licence |

`CLAUDE.md`'s **"Things that will bite you"** is the highest-value page in the
repository for anyone changing code. Read it before your first edit, not after.
