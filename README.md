# Field Ops inspection pipeline — demo

Answers a yes/no inspection question about a site photograph, and shows its
working: whether the photo was good enough to judge, whether the subject is
actually in frame, and what the model concluded.

Three tools that were built separately — a quality gate, an object detector and
a vision-model Q&A leg — run as one pipeline here.

```
photograph ─▶ 1. Quality gate ─▶ 2. Detection ─▶ 2b. OCR ─▶ 3. Answer ─▶ YES / NO / UNKNOWN
              MM-IQA + u2netp    YOLOX-S         PP-OCRv6    Qwen3-VL     + reasoning
                                                 (4 questions)
```

A photograph that fails the quality gate stops there — an inspection decision
made from an unusable photo is worse than no decision. Modes 2 and 3 argue with
that premise; see below.

## The questions

Sixteen, in two domains. Each carries its own system prompt, answer semantics
and relevant detector classes.

| Domain | Questions | Trained detector classes |
|---|---|---|
| **Site Safety** | 2 — hazard signage, GPS antenna sky view | both |
| **Infra** | 14 — lightning arrestor, enclosure condition inside and out, Roxtec sealing, Class B/C SPD installed and active, rectifier modules, temperature sensor placement and reading, earth pit and earthing value, DCDB cable tagging | none yet |

Four of the Infra questions carry an **OCR stage**: the two SPD "installed"
questions, where the class marking printed on the module body is what
distinguishes a Class B module from a Class C one, and the two device readings
(temperature ≤ 35 °C, earthing < 2 Ω), where the answer is a number on a
display.

None of the Infra object classes has trained YOLOX weights yet, so stage 2 runs
on annotation sidecars for them and says so on the card. That matters: "the
detector found nothing" and "nothing was ever trained to find this" look
identical, and only one is evidence of absence.

### Adding or removing one

Questions, domains and object classes are **YAML under `config/`** — no Python.
`PLUGINS.md` is the procedure; the **Reload config/** button on 7873 picks up
an edit without a restart.

```
config/domains.yaml       the domain dropdown
config/classes.yaml       every object class, trained or not
config/questions/*.yaml   the questions, one file per domain
```

## Four UIs, four ports

All are Dash apps and all can run at once.

| App | Port | What it is for | Status |
|---|---|---|---|
| `frontend/demo_dash.py` | 7870 | annotation sidecars, one mode | **frozen** — the known-good fallback |
| `frontend/demo_dash_yolox.py` | 7871 | trained detector, one mode | superseded |
| `frontend/demo_dash_modes.py` | 7872 | three modes over a **folder** of photographs | current — the batch screen |
| `frontend/demo_dash_pipeline.py` | 7873 | domain → question → **one photograph** → three modes | current — the per-photograph screen |

7872 and 7873 are both current and answer different questions. Running twenty
photographs to find the two where the modes disagree is a different job from
examining one closely, and only 7873 has the domain selector and the OCR panel.

### The three modes

| | Mode 1 | Mode 2 | Mode 3 |
|---|---|---|---|
| Quality | hard gate | advisory | judged by the model |
| Detection | YOLOX-S boxes | YOLOX-S boxes | presence in words, no boxes |
| OCR | PP-OCRv6 | PP-OCRv6 | **none — the model reads the text itself** |
| Boxes | YOLOX-S, measured | YOLOX-S, measured | the model's own **claim**, dashed, scored against the detector |
| A blurry photo whose sign is clearly detected | **dropped** | **answered** | answered if the model calls it usable |

Mode 3 not running OCR is the point, not an omission. On a device reading, its
answer is the model reading a seven-segment display unaided, next to two modes
handed PP-OCRv6's transcription — which is the most direct measure of the
model's competence this demo can produce.

Mode 3 also **points at what it is talking about**. Its boxes are drawn dashed
so they cannot be read as detections, and where a trained class exists they
carry an IoU against the detector's own box on the same photograph — so "the
model's grounding is worse than YOLOX's" is a number on screen rather than a
claim in a document. Where no trained class exists, the card says so instead of
showing a score. `MODES.md` has the reasoning and the coordinate trap behind
it.

One run fills all three; switching mode re-reads results rather than
re-running, so the same photographs can be argued three ways in front of an
audience. A Compare tab puts one row per photograph against one column per
mode. `MODES.md` has the reasoning and the cost.

## Quick start

Run on Python 3.11 and 3.12. Every module uses `from __future__ import
annotations`, so nothing here needs a specific minor — but those are the two
that have actually been exercised.

The interpreter must have a working `_ctypes`. One built without `libffi`
breaks pandas, rembg and gradio at once, with an error that names none of them:

```bash
python -c "import _ctypes; print('ok')"     # do this before pip install
```

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-demo.txt

# Check the machine before trusting it. Prints how many questions loaded and
# from where, which OCR models are present, and every config warning.
python tools/preflight.py --port 7873

# Every test (798 assertions, thirteen suites; 804 with every dependency present)
for t in tests/test_*.py; do python "$t" >/dev/null || echo "FAILED $t"; done

python frontend/demo_dash_pipeline.py   # http://<host>:7873
```

Two things are **not** in this repository and must be copied in — the trained
checkpoint (`models/best_ckpt.pth`, ~70 MB) and the YOLOX network definition
(`backend/vendor/yolox/models/`, which upstream's `.gitignore` excluded). See
`COPY_FROM_AISERVER.md` and `YOLOX_SETUP.md`. The demo runs without either, on
annotation sidecars, which is how the Infra questions run regardless.

The OCR models **are** committed (`models/ocr/`, ~6 MB). RapidOCR answers a
missing model path by downloading one, so on an offline host an absent file is
a hang rather than an error. Two are missing from the set — the angle
classifier and the `small` variant; `models/ocr/README.md` records what that
costs and how to drop them in. Confirm the models load with the network down:

```bash
python tools/preflight.py --offline-check --skip-gpu
```

Then press **Load / check model** in the UI. It reports in one line whether
u2netp is present, whether the detector loaded with the right class names, and
whether the model server is reachable — three things worth knowing before an
audience is watching, not during.

## Reaching the model

Stage 3 talks to a Qwen3-VL server. Hosts that cannot see it go through
`llm_proxy_v3` instead:

```bash
export FIELDOPS_VLM_TRANSPORT=proxy
export FIELDOPS_GPU_URL=http://<proxy-host>:8071
export FIELDOPS_VLM_API_KEY=<key>
```

Same model, different paths and an auth header. `PROXY.md` documents the four
behaviours that genuinely differ, and the failure to expect — pointing the URL
at the proxy while leaving the transport on `direct`, which is a 404 that reads
as a dead server.

When the server cannot be reached the pipeline does not fail: it returns a
**clearly labelled mock**, always `unknown`. Asserting yes or no when no model
looked is the one thing this demo will not do.

## Where the documentation lives

| File | For |
|---|---|
| `SETUP.md` | installing on a fresh host |
| `YOLOX_SETUP.md`, `COPY_FROM_AISERVER.md` | the detector, and the two files git does not carry |
| `PROXY.md` | reaching the model through the proxy |
| `MODES.md` | what the three modes argue, and what they cost |
| `PLUGINS.md` | **adding or removing a question, domain or object class** |
| `SERVICE.md` | running it under systemd |
| `FROZEN.md` | the pinned fallback, and how to restore it |
| `OPEN_SOURCE.md` | every open-source component, version and licence |
| `CHANGELOG.md` / `DECISIONS.md` | what changed when / why it is the way it is |
| `CLAUDE.md` | orientation for Claude Code sessions |

## Repository layout

`backend/` is the pipeline and `frontend/` is the UI, and the arrow between
them points one way: the UIs import the pipeline as a library, and **nothing
under `backend/` imports anything from `frontend/`**. That is what lets the
whole pipeline be imported, tested and driven from a script on a machine with
no Dash installed — which is most of the test suite.

```
config/                 THE QUESTIONS - domains, classes, questions (see PLUGINS.md)
backend/                the pipeline, importable with no UI installed
  pipeline/             schemas, config, the registry, the stages, modes
    registry.py         reads config/ into domains, classes and questions
    question_types.py   the Question dataclass and the pure prompt helpers
    stage2b_ocr.py      the OCR stage; ocr_engine.py drives PP-OCRv6
    vlm_grounding.py    mode 3's claimed boxes: read, refuse, place, score
  vendor/yolox/         vendored YOLOX utils (Apache 2.0)
  quality_check.py      vendored from the original quality tool
  foreground_segmentation.py
frontend/               the four Dash UIs - nothing here is imported by backend/
  demo_dash*.py
  assets/demo.css
tools/                  preflight + one smoke script per stage
tests/                  thirteen standalone suites - no pytest, no build step
deploy/                 systemd units and an env template
models/                 u2netp.onnx and ocr/*.onnx (committed); best_ckpt.pth (copied in)
```

Run outputs land under `demo_runs/<run_id>/`:

```
quality/          u2netp foreground boxes, one per photograph
detection/        detector boxes, one per photograph with a detection
ocr/              magenta text-line boxes, for the questions that use OCR
<mode>/           pipeline_results.csv + results.json, one folder per mode
vlm_only/images/  mode 3's plain EXIF-corrected copies (no boxes - u2netp
                  never ran in that mode, so drawing one would be a lie)
```

Nothing prunes them, and a rehearsal is measured in gigabytes; check `du -sh
demo_runs/` before a demo.

## Status

Demo software with a date, not a product. The bias throughout is "works on
stage" over architectural purity, and several compromises are deliberate —
`DECISIONS.md` records which, and what each one costs.

Verified end to end on the demo hosts: stage 1 ~2s per photograph, YOLOX-S
2.4–6.7s on CPU with boxes matching the human annotations at 0.89–0.94, and the
model answering in 0.21–0.48s server-side with reasoning that cites real sign
text. Stage 2b adds roughly 60 ms per detected box or ~340 ms for a whole
frame — the gap is `Det.limit_type="max"` on crops, and losing that setting
turns a small crop into a ~3.5 s read.

`instant_graph_mcp_server_v2_4_1.py` at the repo root is unrelated earlier
work.
