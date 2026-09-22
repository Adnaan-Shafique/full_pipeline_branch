# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A field-inspection demo pipeline that answers a yes/no question about a
photograph:

1. **Quality gate** — MM-IQA cues + u2netp foreground segmentation
2. **Detection** — trained YOLOX-S, or human annotation `.txt` sidecars
2b. **OCR** — PP-OCRv6 via RapidOCR, for the questions that turn on text
3. **Answer** — Qwen3-VL on a remote GPU server, reached directly or via a proxy

Questions, domains and detector classes are **YAML under `config/`**, not
Python. `PLUGINS.md` is the procedure for adding or removing one; read it
before editing `backend/pipeline/questions.py`, which is now a compatibility
surface over `registry.py` rather than where questions live.

It integrates three previously independent tools. The bias throughout is
"works on stage" over architectural purity — this is demo software with a
date, and several deliberate compromises below exist for that reason.

`instant_graph_mcp_server_v2_4_1.py` at the repo root is unrelated earlier work
(see commits before 2026-09-09). Leave it alone.

## Commands

No build step, no linter, no pytest. Suites are standalone scripts that print
`N passed, M failed` and exit non-zero on failure.

```bash
# Every suite (816 assertions across thirteen files)
for t in tests/test_*.py; do python "$t" >/dev/null || echo "FAILED $t"; done

# One suite, with its output
python tests/test_modes.py

# There is no single-test runner. Suites are linear scripts; to isolate a
# section, comment out the ones above it or read the `print("\n...")` banners
# that separate them.

# Before running anything on a demo host
python tools/preflight.py --port 7872 --transport proxy

# Smoke one stage against real data
python tools/smoke_stage1.py <photos>
python tools/smoke_yolox.py <photos> --ckpt models/best_ckpt.pth --limit 3
python tools/smoke_stage3.py <photos> --limit 2 --transport proxy
python tools/smoke_ocr.py <photos> --question temp_within_limit
python tools/smoke_ocr.py --list x          # which questions use OCR, and their rules
python tools/inspect_ckpt.py models/best_ckpt.pth

# The UIs (each on its own port, all runnable at once)
python frontend/demo_dash.py           # 7870 — FROZEN, annotation sidecars
python frontend/demo_dash_yolox.py     # 7871 — trained detector
python frontend/demo_dash_modes.py     # 7872 — three modes over a FOLDER of photos
python frontend/demo_dash_pipeline.py  # 7873 — domain → question → ONE photo → 3 modes
```

7872 and 7873 are both current and answer different questions. 7872 is the
batch screen: run twenty photographs, find the two where the modes disagree.
7873 is the per-photograph screen, and the one that has the domain selector and
the OCR panel.

Tests stub `cv2`, `dash` and `requests`. Measured on three interpreters:

| Interpreter | Assertions | Suites failing |
|---|---|---|
| everything installed | **822** | 0 |
| no `cv2`, `dash`, `torch`, `rembg`, `pandas`, `rapidocr`, `onnxruntime` | **816** | 0 |
| bare — nothing installed at all, `requests` and `pyyaml` included | **645** | 2, both pre-existing |

Keep that middle row at zero — a suite that needs torch cannot run where it is
most needed.

**A section that needs a real dependency must skip cleanly and say so**, never
crash. Suites print `N passed, M failed`; a traceback breaks that contract and
tells the reader nothing. `test_ocr.py`'s live-engine section needs `rapidocr`
and a real `cv2`; four suites skip their config-dependent sections without
`pyyaml`. Note `test_ocr.py` stubs `cv2` **conditionally**, only when the real
one is absent — an unconditional stub wins for the whole process and silently
skips that section.

**`pyyaml` is effectively required now, not soft.** It was only
`PromptStore.save` and the `dataset.yaml` parser (the two suites still failing
in the bare row above); it now also carries the entire question registry.
Without it the registry falls back to two built-in Site Safety questions and
names the reason in every UI's status line — so the demo starts with 2
questions instead of 16 and no Infra domain at all, and the suites that assert
on Infra questions skip rather than fail.

`tests/test_pipeline_app.py` reuses `test_dash_ui.py`'s dash stub by **exec'ing
that file with `sys.exit` neutralised**, not by importing it. Importing it
raises its closing `SystemExit` during the import, which also drops the
half-built module from `sys.modules` — so an import/catch/re-import loop runs
it twice and then exits the *calling* suite with 0 before its first assertion.
It did exactly that, silently, until the assertion count gave it away.

## Architecture

### The frontend/backend split

```
backend/    the pipeline. Importable and testable with no UI installed.
frontend/   the four Dash UIs. Imports backend/ as a library.
```

**The arrow points one way.** `frontend/` imports `backend/`; nothing under
`backend/` imports anything from `frontend/`. That is not tidiness — it is what
lets the whole pipeline run from a script, a test or a smoke tool on a machine
with no Dash installed, which is how most of the suite runs. If you ever find
yourself importing a renderer into a pipeline module, the thing you want is a
field on the dataclass instead.

Each UI puts `backend/` on `sys.path` itself (`BACKEND_DIR`), so the packages
import as `pipeline.*` exactly as before the split. Tests add `ROOT / "backend"`;
the UI suites also load the frontend files by path.

**Depth is load-bearing.** Every `__file__`-derived path — `config.py`'s
`parents[2]`, `registry.py`, `ocr_engine.py`, and
`foreground_segmentation.py`'s `parent.parent` — assumes it sits exactly one or
two levels below the project root. `backend/` and `frontend/` are at the same
depth `app/` was, which is why the move needed no arithmetic changes. Introduce
a `src/` above them and every one of those has to be re-derived.

### Stage contracts

`backend/pipeline/schemas.py` defines the dataclasses every stage fills and the UI
reads. A `PipelineRecord` holds one photograph's `quality`, `detection`, `ocr`
and `vlm` results plus `stopped_at`. Changing a field here touches every stage
and all the renderers — read it first.

`OCRStageResult` distinguishes three states that must never be conflated,
because only the first is a problem with the demo host: `error` set (the stage
could not run — no rapidocr, no models, an engine that raised), `error` unset
with no lines (it ran and read nothing legible), and `scope_used == "skipped"`
(nobody asked it to run). `build_ocr_block()` renders the first and third as
**no prompt block at all** — "OCR failed" is information for the operator,
never a piece of visual evidence for the model.

### Configuration

`backend/pipeline/config.py` is a single `PipelineConfig` dataclass with
`default_config(**overrides)`. Every path derives from `project_root`, which
derives from `__file__`, so the tree relocates without edits. `validate()`
returns human-readable warnings rather than raising — the UI shows them.

`FIELDOPS_*` environment variables override config fields (see `ENV_OVERRIDES`).
Explicit keyword arguments beat the environment. **This makes
`default_config()` environment-sensitive**, which has already broken a test
that asserted defaults — see `tests/test_transport.py`, which clears those
variables before its first call and restores them at the end.

### Two transports, one GPU

Stage 3 reaches the same model either directly (`/infer`) or through
`llm_proxy_v3` (`/v1/infer` + `X-API-Key`). Every path difference lives in the
`ENDPOINTS` table in `stage3_vlm.py`; nothing else knows which route is in use.
`PROXY.md` documents the four behaviours that genuinely differ. The failure to
expect: `gpu_url` pointing at the proxy while transport is still `direct`,
which is a 404 that reads as a dead server.

### Three modes

`backend/pipeline/modes.py` runs all three over one set of photographs, sharing
work: one quality pass, one detection pass, **one OCR pass**, and one VLM
answer when modes 1 and 2 both proceed. Mode 3 replaces the classical stages
with the model itself, as three separate calls with per-leg editable system
prompts. `MODES.md` has the reasoning.

### OCR (stage 2b)

`backend/pipeline/stage2b_ocr.py` drives `ocr_engine.py` (PP-OCRv6 through
RapidOCR, over the ONNX files committed in `models/ocr/`). It runs for the
questions whose YAML sets `ocr.enabled` — four today — and **never in mode 3**,
where the model reads the text itself. That asymmetry is the comparison
`demo_dash_pipeline.py` exists to show, so do not "fix" it by feeding mode 3
the OCR output.

### Mode 3's claimed boxes (grounding)

`backend/pipeline/vlm_grounding.py` turns coordinates the model wrote into boxes
worth drawing. Mode 3's presence and answer legs are asked to point (the
quality leg is not — it judges a whole-frame property). `MODES.md` has the full
reasoning; the three rules that must not be relaxed are in the traps below.

### Questions, domains and classes — the plugin layer

The data is **YAML under `config/`**; `backend/pipeline/registry.py` reads it.
`PLUGINS.md` is the procedure. In short:

| File | Holds |
|---|---|
| `config/domains.yaml` | the domain dropdown |
| `config/classes.yaml` | every object class; `trained: true` entries derive `yolox_class_names` |
| `config/questions/*.yaml` | the questions, one file per domain |

`backend/pipeline/question_types.py` holds the `Question` dataclass and the pure
prompt helpers, split out so `registry.py` can build questions without
importing the module the registry populates. `backend/pipeline/questions.py` is the
**stable import surface** over all of it — `from pipeline.questions import
get_question, select_relevant, QUESTIONS` still resolves exactly as before, and
that is the import every module and test should use.

Loading never raises. Every problem it can survive becomes a warning on
`REGISTRY.warnings` that the UIs print and `preflight.py` lists; the entry that
caused it is dropped. The three files stand or fall independently.

Mode 3's three legs have their own defaults in `questions.py`; `prompts.py`
persists UI overrides to `config/prompts.yaml` (gitignored, per-machine).

### Vendored code

`backend/vendor/yolox/` holds YOLOX utils under its Apache licence. **`yolox/models/`
is deliberately absent from git** — the upstream repo's bare `models/` gitignore
pattern excluded it — and must be copied in per `YOLOX_SETUP.md`. `build_model()`
raises an ImportError naming that cause when it is missing.

`backend/quality_check.py` and `backend/foreground_segmentation.py` are vendored from
the original toolset. `foreground_segmentation.py` computes `MODELS_DIR` from
its own location, so it must stay directly in `backend/` — one level below
the project root, which is what makes `models/` resolve. `app/` was also one
level below the root, which is why the move preserved it.

## Things that will bite you

**`frontend/demo_dash.py` is frozen** (`FROZEN.md`, commit `f978b7d`). It is the
fallback that is known to work. `demo_dash_modes.py` imports renderers from it
and re-implements only what mode 3 needs; `demo_dash_pipeline.py` imports from
both and adds only the OCR panel. Do not "fix" the frozen file to avoid a
re-implementation.

**Arrays, never paths.** `quality_check.load_image_bgr()` applies EXIF
transposition; re-reading the file with `cv2.imread` does not. A box computed
on one and drawn on the other is silently wrong. Stage 3 only accepts the BGR
array the pipeline is already holding.

**The registry bootstrap.** `build_payload()` refuses an empty registry rather
than defaulting an unknown model to text-only — without that, every image
request fails with `'qwen3-vl' is text-only`, which looks like a broken server.
Call `refresh_registry()`/`ensure_registry()` first.

**A blank system prompt fails silently.** The server skips a falsy system turn,
so the model answers unframed and it reads as a model-quality problem.
`Question.__post_init__` and `build_payload()` both refuse one.

**Class names come from config, not the checkpoint.** YOLOX stores no names. A
wrong order does not error — it puts a confident wrong label on screen *and*
into the VLM prompt as evidence. The source of truth is now
`config/classes.yaml`, whose trained entries carry an **explicit
`yolox_index`** — position in the file means nothing, so reordering it for
readability cannot relabel every detection. Index 0 is `GPS Antenna`, from the
training run's `classes.json`. `config.py` and `registry.py` keep literal
copies as the no-pyyaml fallback, and `tests/test_yolox.py` pins all three
against each other: a drifted fallback relabels everything with nothing
raising.

**Untrained is not absent.** Only two classes have weights. Every Infra
question runs on annotation sidecars, and "the detector found nothing" looks
identical on a card to "nothing was ever trained to find this" — only one of
which is evidence. The question panel on 7873 names the untrained classes for
exactly this reason; keep that, and keep `is_stub` honest.

**OCR produces evidence, not answers.** The numeric rule takes the *first*
number its pattern matches. It cannot see a range multiplier on a rotary
switch, tell a set-point from a measurement, or tell °F from °C — on
`SET 22 C ACT 38.0 C` it takes the set-point, and `tests/test_ocr.py` pins that
as a known limit rather than describing it in a comment. So the rule's output
goes to the model as one advisory line, the card shows *which token* was
matched, and it is never styled as an answer chip. The confidence floor runs
**before** both the prompt and the rule, so a 0.11-confidence misread cannot
become a fabricated measurement with a verdict attached.

**A question whose prompt mentions OCR needs `system_prompt_no_ocr`.** Mode 3's
answer leg reuses `system_prompt`, and mode 3 runs no OCR — so without the
variant, the one mode meant to show what the model reads unaided is told to
expect evidence that never arrives. This was a real bug on all four OCR
questions. `preflight.py` warns about any that lack it.

**A claimed box is not a detection, and must never become one.** Mode 3's
boxes live in `DetectionStageResult.claimed_boxes`, and `detections` stays
EMPTY. Everything that reads `detections` — `build_detection_block()`,
`select_relevant()`, the full+crop chooser — treats its contents as detector
output, so a claimed region put there would end up in a prompt as evidence and
the model would be citing itself. The detections passed to
`compare_to_detections()` arrive after the legs have answered and never reach a
prompt.

**Absolute coordinates are in the ENCODED frame, not the original.**
`array_to_data_uri()` downscales to `MAX_UPLOAD_SIDE_PX` (2048), so the model
never sees the original. `encoded_size()` mirrors that arithmetic and
`interpret_box()` scales from it; if the resize in `array_to_data_uri()` ever
changes, `encoded_size()` must change with it or every absolute box lands at
the wrong scale silently. The prompt asks for normalized 0-1000 precisely to
avoid depending on this.

**"Unscored" is not "scored zero".** A claimed box with no trained class to
compare against carries `iou is None` and the wording *"no trained detector box
to compare against"*. Rendering that as 0.00, or as a blank, states the
opposite of what was found. Every Infra question is in this case today.

**`yolox_input_size` is `(640, 480)`, not the stock square.** For portrait
photographs the ratio is identical either way; only landscape diverges (by 4/3).

**Never let a mock assert.** Mock and parse-failure paths return `unknown`
(and `poor` for quality), always labelled. Asserting yes/no when no model
looked is the one thing this demo must not do on screen.

**Tests must not depend on their host.** Two have already failed only on the
demo VM: one read `FIELDOPS_*` from the environment, and one popped
`yolox.models` from `sys.modules` (which clears the cache but does not prevent a
re-import, so it passed only where the package was genuinely absent).

## Conventions

Comments explain *why*, especially where a line guards against a specific
failure that has actually happened. Several carry the symptom so the next
reader recognises it. Do not strip them as noise.

Commit messages are prose explaining the reasoning and what was verified, not
bullet lists.

Two companion logs are kept up to date by hand: `DECISIONS.md` (why things are
the way they are) and `CHANGELOG.md` (what changed, when). Add to them when a
change is worth explaining later.

`README.md` is the front door for a human arriving cold — what the pipeline
does, the four UIs and ports, a quick start, and a map of the other documents.
This file is the one for a Claude session; keep the split, and do not duplicate
architecture into the README. `PLUGINS.md` is for whoever edits the questions,
who may not be an engineer at all — keep it procedural and keep the *reasons*
for its rules in it, since those rules look arbitrary without them.
