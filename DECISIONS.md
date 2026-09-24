# Decisions

Why things are the way they are. Each entry records the choice, the reason, and
what it costs — so a later reader can tell a deliberate trade from an accident,
and knows what to re-examine if the constraint changes.

Newest first. Dates are when the decision was made, not when it was written up.

---

## 2026-09-24 · Ship the operator whole servers, and put the benchmark's settings in the environment

**Decision.** `deploy/gpu_api_server_v7.py` and `deploy/llm_proxy_v4.py` are
complete replacements for the operator's running v6 and v3, not blocks to paste
into them. The benchmark's own settings — internvl at TP=1, the raised
`gpu_memory_utilization`, an empty `EAGER_LOAD` — are environment variables
(`VLM_TP_*`, `VLM_GMU_*`, …) rather than values written into `MODEL_CONFIGS`,
so the file on disk is always the production configuration.
`deploy/gpu_models_bench.py` is kept as the annotated diff and demoted out of
the procedure.

**Why.** Two different failures, with the same shape: a change that half-lands
and says nothing.

Pasting seven blocks into a running server is an instruction that works until
someone misses one. The symptom surfaces three steps later — a 400 that reads
as a GPU fault, a prompt template that produces fluent answers about an image
the model never saw — and nothing points back at the block. Copying a file has
one failure mode, and `ast.parse` catches it in a second.

Editing `MODEL_CONFIGS` for a benchmark is worse, because the damage is
deferred. The run finishes, the numbers are fine, and the file still says
internvl is TP=1 at 0.85 — which is wrong for production and wrong in a way
nobody sees until mistral and qwen3-vl will not both fit on the card, on a
morning when someone is demoing. The runbook's Step 5 already said "this is the
step people skip"; that is a sentence admitting a design problem rather than
fixing it. With the settings in the environment, restoring is `unset`, the
default is right, and `tests/test_bench.py` fails if a benchmark value is ever
committed into the file.

**What it costs.** Two more files in `deploy/`, and the operator's v6 and v3
are now history that this repo does not track — if they patch their running
copy, our v7 will silently not contain it. The mitigation is that both files
carry a `CHANGES FROM` header listing every difference with its reason, so a
merge is readable rather than a diff hunt. The environment indirection also
means a misconfigured shell can serve a run under settings nobody intended,
which is why v7 logs every override as it applies it and reports them on
`/models`, and why the harness records the serving configuration beside the
numbers.

---

## 2026-09-24 · A queue wait that is measured, and one that is not, are different numbers

**Decision.** `gpu_api_server_v7` measures the semaphore wait itself and returns
it; `llm_proxy_v4` passes that figure through and reports what is left of the
round trip separately as `transport_overhead_s`. When only the old
approximation is available, it is still reported — but flagged, and any single
approximated sample flags the whole distribution.

**Why.** `llm_proxy_v3` computed `queue_wait_s` as its own elapsed minus the
GPU's, and `BENCHMARKS.md` quoted it as the saturation signal: rising queue
wait against flat server time means requests are waiting for one of the two
concurrency slots. The arithmetic is right about what it contains and wrong
about what it is. That difference is queueing *plus* network *plus* JSON
encoding of a multi-megabyte data URI *plus* the proxy — and on a fast,
unloaded server it is almost entirely the last three. Read as queue wait, an
idle server looks saturated. The finding it was meant to produce is the one it
would most reliably fabricate.

Measuring at the semaphore is a few lines on the server and leaves nothing to
infer. The subtraction is still useful once the real wait is known: what
remains is transport, and "the proxy is expensive" and "the GPU is busy" are
opposite findings with opposite fixes.

**What it costs.** A run against an older proxy now carries a warning it did
not before, and its queue figures are labelled an upper bound rather than a
measurement — which is less convenient and more honest. Mixing such a run into
a comparison is refused rather than smoothed over, so a half-upgraded pair of
boxes produces a table with a caveat on it instead of a clean one that is
quietly wrong.

---

## 2026-09-24 · A benchmark that refuses more than it reports

**Decision.** The benchmark harness excludes failed requests from latency
figures, gives 503 its own outcome, tags mock runs out of the ranking, uses
nearest-rank percentiles with a caveat below 100 samples, and aborts entirely
when the server is serving a different model than the run claims.

**Why.** A benchmark's failure mode is not crashing. It is producing a
confident number that is wrong, which nobody catches because it looks like
every other number. Two examples decided the design. A connection refusal
returns in 4 ms; counted as a sample, it drags the mean down and makes a
saturated server look fast — so only requests that actually answered are timed,
and the failures are counted separately where they read as what they are. And
on a server that holds one model at a time, forgetting to switch gives you a
full result file attributed to the wrong model, with nothing in the numbers to
betray it — so the registry listing AND the reply's own attribution are both
checked before anything is measured.

**What it costs.** The harness reports fewer numbers than it could, and says
"not measured" more often than a benchmark usually does. A mock run produces no
latency at all, only a separately-named harness timing, which makes it useless
for anything but proving the plumbing works — deliberately.

---

## 2026-09-24 · Contract compliance is reported before speed

**Decision.** The per-model table leads with the share of replies that honoured
the JSON contract outright, and the share the parser gave up on. Latency comes
after.

**Why.** `parse_vlm_answer` is tolerant in four descending tiers, and that
tolerance is a safety net rather than a licence. A model that never emits valid
JSON still produces answers through tiers 2 and 3 and looks completely healthy
in the results — while being one prompt edit away from producing nothing at
all. Worse, replies the parser gives up on arrive as "unknown", which is
indistinguishable from a model honestly declining to guess. Only the tier tells
them apart, so the parser now reports it.

**What it costs.** `parse_vlm_answer_tiered()` is a second public entry point
on a function that had one. It was split rather than re-implemented in the
benchmark, because two parsers meant to agree eventually will not, and the one
being reported on would stop being the one the pipeline runs.

---

## 2026-09-24 · Open-loop as well as closed-loop load

**Decision.** `continuous` offers requests on a clock regardless of whether
previous ones have returned; `parallel` waits for each worker's previous call.
Both ship.

**Why.** Closed-loop load cannot overload a server. If it slows, the client
sends less, and the system finds an equilibrium that hides the problem — a
benchmark built only that way reports a system as healthy right up until it
collapses. Open-loop keeps offering work, so a queue builds and the measured
latency includes the waiting, which is what a user actually experiences.

**What it costs.** An open-loop generator can fall behind its own schedule and
then be describing a load it never produced. Every sample records
`schedule_lag_ms` and the result warns when it grows. There is also a
`max_inflight` valve so an overloaded shared GPU does not turn the scenario
into an unbounded fork bomb; when it engages the scenario says so, because at
that moment the offered rate stopped being the configured one.

---

## 2026-09-23 · backend/ and frontend/, with the arrow pointing one way

**Decision.** `app/` becomes `backend/` (pipeline, vendored modules, YOLOX
utils) and `frontend/` (the four Dash UIs, the stylesheet). `frontend/` imports
`backend/`; nothing under `backend/` imports anything from `frontend/`. No
compatibility shims at the old paths.

**Why.** The pipeline was already the importable part and the UIs already the
leaves, but nothing enforced it and the layout did not say so. The property
worth protecting is that the whole pipeline runs on a machine with no Dash
installed — that is how most of the test suite runs, how the smoke tools run,
and how anyone porting this to the edge device will consume it. A backend
module importing a renderer would break that silently: it would only fail on
the machine that has no dash, which is the machine that matters.

**Why not `src/`.** Every `__file__`-derived path in the repo assumes a fixed
depth below the project root — `config.py`'s `parents[2]`, `registry.py`,
`ocr_engine.py`, and `foreground_segmentation.py`'s `parent.parent`, which is
what makes `models/` resolve and what `SETUP.md` warns about. `backend/` and
`frontend/` sit exactly where `app/` sat, so the move needed no arithmetic
changes at all. A `src/` above them would have required re-deriving every one,
for a tidier root.

**What it costs.** Anything scripted against `app/` outside this repository
breaks once. Shims were considered and rejected: two paths to every file is a
worse long-term cost than one update, and a shim that nobody removes becomes
the layout. The guard against the arrow reversing lives in `test_phase1.py` and
parses imports rather than grepping — the grep version failed on a docstring
that merely names a UI, which is documentation and should stay.

---

## 2026-09-23 · Mode 3's overlay is a view control, rendered ahead of time

**Decision.** Mode 3's claimed boxes can be shown as box + label, box only, or
off. All three renderings are written during the run; the UI control picks a
file rather than redrawing.

**Why.** The caption is long — "model says: a temperature display or device
reading (IoU 0.62 vs detector)" — and on a tight box around a small object it
covers the object. On a device reading that is the digits, which is the one
thing the box exists to point at.

Rendering ahead of time rather than on the callback is the part worth
recording. Redrawing would mean either holding every decoded photograph in
memory for the session, or re-reading it from disk — and re-reading is exactly
what the "arrays, never paths" rule exists to prevent, because the re-read
frame is not EXIF-corrected and the boxes would land wrong. Three JPEGs per
photograph is the cheaper and safer trade, and it makes the control free:
switching never costs a model call, which is why it sits with the mode selector
rather than among the settings that do force a re-run.

**What it costs.** Two extra files per photograph that has a claim, and a
`demo_runs/` folder that grows faster. A failed variant render now has to be
distinguished from a record that predates the feature — falling back to the
plain photograph in both cases would show an image with no boxes beside a card
listing a claim, which reads as "the model claimed nothing".

---

## 2026-09-23 · Mode 3 draws boxes after all, scored against the detector

**Decision.** Reverses "mode 3 draws no bounding boxes". The presence and
answer legs are asked for coordinates; what comes back is drawn dashed,
captioned `model says: …`, and — where a trained class exists — scored against
the detector's own box on the same photograph, with the IoU printed.

**Why the original decision was right, and why it still changed.** The original
reasoning was that Qwen3-VL's grounding is well below YOLOX's and a visibly
wrong box is worse than an honest "presence, not geometry". Both halves are
still true. What was wrong was the conclusion: the weakness was being asserted
in a document nobody reads during a demo, while the question it was avoiding —
*where does the model think it is?* — is the first thing anyone asks of a
vision model, and mode 3 exists precisely to expose what the model can do
alone.

Drawing it became defensible once the box could arrive with its own error bar.
Modes 1 and 2 have already run the detector over that exact photograph in the
same frame, so the IoU is free, and the claim becomes measurable on screen
rather than argued here. Three conditions make it honest: dashed boxes that
cannot be confused with the three solid visual languages already in use; an
IoU in the caption; and a refusal path for anything that cannot be placed
confidently — a whole-frame box especially, which is the model declining to
localise while appearing to comply.

**The claim is never an input.** The boxes live in `claimed_boxes`, not in
`detections`, because everything that reads `detections` treats it as detector
output and a claimed region fed back into a prompt would have the model citing
itself. The detections used for scoring arrive after the legs have answered.

**What it costs.** A wrong box can now appear on screen — that risk was real
and has been mitigated, not eliminated. Two prompts per grounded leg to keep in
step, a second visual language to learn, and a `vlm_grounding=False` escape
hatch to remember when the boxes are noise on a given photo set. And where no
trained class exists — every Infra question today — the box is unscored, so it
carries "no trained detector box to compare against" rather than a number.
That wording is load-bearing: it is not the same as scoring zero.

---

## 2026-09-23 · Normalized 0-1000 coordinates, and refusing what cannot be placed

**Decision.** The grounded prompts ask for integers normalized 0-1000.
`vlm_grounding.interpret_box()` also accepts fractions and absolute pixels,
scales the latter from the ENCODED frame, and reports which reading it used.
Anything it cannot place confidently is refused with a reason rather than
drawn.

**Why.** `array_to_data_uri()` downscales to 2048px before sending, so the
model never sees the original frame. An absolute pixel reply is in the resized
frame, and drawing it on a 4000x3000 original is out by about 2x with nothing
raising — the same class of bug as computing a box on an EXIF-corrected array
and drawing it on a re-read file. Normalized coordinates are immune to the
resize, and are the convention the Qwen-VL family was trained to emit, so that
is what the prompt asks for.

Accepting the other conventions anyway, and naming which one was used, is the
part that makes a systematic misread visible: it shows up as every box on every
photograph arriving as `absolute_px_encoded_frame`, instead of as boxes that
are quietly wrong.

**What it costs.** A model that returns absolute coordinates on a small image
cannot be told apart from one returning normalized ones — both are under 1000.
The demo accepts that ambiguity rather than adding a round-trip to resolve it;
the convention is printed, so a human can see it.

---

## 2026-09-22 · The questions are YAML, and the loader never raises

**Decision.** Questions, domains and object classes move out of
`backend/pipeline/questions.py` into `config/domains.yaml`, `config/classes.yaml`
and `config/questions/*.yaml`, read by `backend/pipeline/registry.py`. Every
problem the loader can survive becomes a warning the UI prints, and the entry
that caused it is dropped. `questions.py` stays as the import surface, so every
existing `from pipeline.questions import ...` still resolves.

**Why.** Adding a question used to mean editing four places in one file — the
`Question` literal, `SUBJECTS`, `QUESTION_TEXT`, and the class names it matches
on. Forgetting either of the middle two did not error; it silently downgraded
mode 3 to generic fallback wording. At two questions that was survivable. At
sixteen it is not, and the people most likely to add a seventeenth are not
engineers.

Loading is tolerant because a malformed file must not take the demo down five
minutes before it runs. The two things it will not survive — a tree with no
questions at all, and a question with a blank system prompt — fall back to
built-ins loudly, because the first leaves nothing to demo and the second fails
*silently at the server*, which is the failure this repo keeps legislating
against.

**What it costs.** `pyyaml` stops being a soft dependency: without it the
registry falls back to two built-in questions and the Infra domain does not
exist. The demo still starts, and says why in every status line, but it is a
much smaller demo. `registry.py` and `config.py` also now hold literal copies
of the trained class names as that fallback, which is a second source of truth
— pinned against the YAML by `tests/test_yolox.py`, because a drifted fallback
relabels every detection with nothing raising.

---

## 2026-09-22 · `yolox_index` is written per class, not inferred from list order

**Decision.** A trained entry in `config/classes.yaml` carries an explicit
`yolox_index`. `PipelineConfig.yolox_class_names` is derived by sorting on it.
Duplicates, gaps and a `trained: true` with no index each demote the class to
untrained, with a warning.

**Why.** YOLOX stores no class names in a checkpoint, so that file is the only
record of them, and a wrong ORDER does not error — it puts a confident wrong
label on screen *and* into the VLM prompt as evidence. That has already
happened here once: the detector was localising hazard signs correctly and
calling them GPS antennas. Position in a YAML list is far too easy to change by
accident; reordering two entries for readability would have silently relabelled
every detection.

**What it costs.** Two fields to keep in step instead of one, and a rule that
looks like ceremony until you have hit the failure it prevents. `PLUGINS.md`
carries the reason next to the rule for that reason.

---

## 2026-09-22 · OCR produces evidence; the model still answers

**Decision.** Stage 2b reads text with PP-OCRv6 and puts it into the prompt
behind a hedge telling the model to prefer the image. Where a question carries
a numeric rule, the threshold check goes in the same way. **The pipeline never
answers a question from the OCR output**, even when the output is a number and
the question is a threshold.

**Why.** It is tempting to let `33.5 <= 35` settle "is the temperature within
35 °C". It must not, because the rule takes the first number its pattern
matches and cannot see three things that change the answer:

- **the range multiplier** — `1.85` on a 20 Ω range is 1.85 ohms; the same
  digits under a kΩ indicator are not,
- **a set-point versus a measurement** — on `SET 22 C ACT 38.0 C` the rule
  takes 22, and the real reading fails the limit,
- **°F versus °C**, which OCR reports identically when the unit is
  silk-screened on the bezel where no camera sees it.

A confident wrong number *with a verdict attached* is worse than no number. So
the rule's output is one advisory line, the card shows **which token** was
matched rather than only the value, and the verdict is never styled as an
answer chip. Two of those three failure modes are pinned as passing assertions
in `tests/test_ocr.py` — as known limits, so they stay visible instead of being
rediscovered live.

The confidence floor runs **before** both the prompt and the rule, so a
0.11-confidence misread of `1B5` cannot become "185 ohms, outside the limit".

**What it costs.** On a clean display where OCR is right and the model is
wrong, the pipeline answers wrongly and the correct number is sitting on the
card. That is the accepted trade: the demo's one non-negotiable is that it does
not assert something no model concluded.

---

## 2026-09-22 · Mode 3 runs no OCR, and gets its own system prompt

**Decision.** Stage 2b runs in modes 1 and 2 and never in mode 3. A question
that uses OCR carries a second system prompt, `system_prompt_no_ocr`, which
mode 3's answer leg sends instead of the usual one.

**Why.** Mode 3's argument is that one model seeing the whole photograph judges
more coherently than components that each see a slice. Handing it PP-OCRv6's
transcription would make it answer better and tell us nothing — and "can this
model read a seven-segment display on an edge device" is the question anyone
evaluating this pipeline is actually asking. The asymmetry *is* the measurement.

The second prompt exists because the first one lies in that mode. All four OCR
questions tell the model that OCR text "may be supplied to you as advisory
evidence", and in mode 3 nothing ever supplies it. Found by running the app and
reading the Prompts tab, not by a test — the mode had been mis-framed since the
stage was added.

**What it costs.** Two prompts to keep in step per OCR question, and the
temptation to fix a mode-3 wrong answer by feeding it the OCR block. Preflight
warns about any OCR question missing the variant; `tests/test_ocr.py` asserts
the two prompts differ and that mode 3's never promises evidence.

---

## 2026-09-22 · Commit the OCR models, and omit the classifier rather than point at it

**Decision.** `models/ocr/*.onnx` are committed. The angle classifier did not
ship with them, and `_get_engine()` omits the `Cls.model_path` key entirely
rather than passing a path to a file that does not exist.

**Why.** The same reason `u2netp.onnx` is committed, with more force: RapidOCR
answers a missing model path by **downloading** one. On a demo host with no
route out that is a hang in front of an audience, not an error anyone can read.
Omitting the key is the only way to say "there is no classifier" without
triggering that. `preflight.py --offline-check` builds both engines with the
network denied, which is the only test that actually proves it.

**What it costs.** ~6 MB in git, and text rotated 180° reads worse. Upright
text is unaffected. `models/ocr/README.md` records the hashes and what dropping
the missing files in would restore.

---

## 2026-09-18 · Run the demo under systemd, on Werkzeug

**Decision.** A systemd unit (`deploy/fieldops-demo-modes.service`) running
Dash's built-in server, not gunicorn.

**Why.** The demo had been started by hand on every host, so it died with the
SSH session. Werkzeug is a development server, but for a closed network and a
known audience it is the better trade: no extra dependency, no extra failure
mode, and it is threaded, so a multi-second pipeline run does not block the
page.

**Cost.** Not suitable if this outlives the demo. `SERVICE.md` has the gunicorn
line, pinned to `--workers 1` — results live in a module-level dict, so a
second worker would answer half the requests from an empty set.

**Related.** The API key lives in `/etc/fieldops-demo.env` at `chmod 600`, not
in the unit: unit files are world-readable and `systemctl cat` prints them on
request.

---

## 2026-09-17 · Uploads accumulate instead of replacing

**Decision.** Photographs stage to disk on arrival and accumulate in a
`dcc.Store`; a second drop adds to the set rather than replacing it.

**Why.** Six files selected in Explorer, dragged in one gesture, arrived as
one. Six correctly-typed JPEGs dropped in a real browser against this exact
layout all arrive, so the desktop handed the browser a single-file
`DataTransfer` — not something Python can fix. What *can* be fixed is the UI
depending on that gesture succeeding.

**Cost.** More moving parts than reading `contents` off the component: a Store,
a Clear button, and staged files that outlive a page reload.

**Also decided here.** The note reports what is *staged*, not what the last drop
carried, so the number on screen is the number that will run. Clearing the
component's `contents` after each merge releases browser memory *and* lets the
same file be dropped twice — an unchanged prop would not fire the callback
again.

---

## 2026-09-17 · `accept` lists extensions, not just `image/*`

**Decision.** The dropzone accepts `image/*` **and** every extension
`collect_images()` takes, from one `IMAGE_EXTENSIONS` set.

**Why.** `image/*` filters on the MIME type the *browser* reports. A file
dragged from a network share or a mapped drive arrives with no type, and
react-dropzone discards it silently — no message, no count, nothing in the
server log. Reproduced in Chromium: three files dropped, one accepted.

**Cost.** None found. Non-images are still refused, now server-side as well —
a filter that runs only in the browser is one that can be bypassed.

---

## 2026-09-16 · A second transport rather than a second client

**Decision.** `vlm_transport` selects between the GPU server's paths and
`llm_proxy_v3`'s. One `ENDPOINTS` table holds every difference.

**Why.** The field-ops VM has no route to the GPU box at all; it reaches
FALCONPRD, which forwards. That is a different HTTP contract, not just a
different address. A second client class would have duplicated the payload
building, the registry bootstrap and the parsers — the parts most expensive to
get wrong twice.

**Cost.** Every call site passes a transport. Worth it: nothing outside
`stage3_vlm.py` knows which route is in use.

**Deliberate.** `build_payload()` strips `repetition_penalty`, `request_id` and
`client_id` for the proxy. Pydantic *ignores* undeclared fields, so they would
be dropped in silence; dropping them here means a recorded run cannot claim a
setting that was never honoured. At the demo default of 1.0 — the server's own
floor — the behaviour is identical either way, and `validate()` warns if it is
raised.

---

## 2026-09-16 · Mode 3 is three calls, not one

**Decision.** Mode 3 asks quality, presence and the answer as three separate
calls, each with its own editable system prompt.

**Why.** One call keeps the three judgements mutually consistent but makes a
wording change to one criterion move the other two with it. Three calls let
each be tuned — and blamed — independently.

**Cost.** Three calls per photograph instead of one; roughly 6.5s per photo for
all three modes against 5.5s for one. The image is encoded once and reused, so
the cost is tokens, not another decode.

**Deliberate.** Each leg's *user* prompt is shown but fixed: it carries the JSON
contract the parser keys on, and an edit there would silently turn every answer
into `unknown`.

---

## 2026-09-16 · Mode 3 renders the plain photograph, with no box

**Decision.** Mode 3 writes an unannotated EXIF-corrected copy and renders it;
no green u2netp foreground box.

**Why.** u2netp does not run in that mode. Drawing its output would credit a
component for work it did not do. For the same reason the quality panel is
re-implemented rather than reused from the frozen one, which would print
`whole frame 0.0` — a number the model never produced.

---

## 2026-09-16 · Tuned prompts are per-machine, and absence is the default

**Decision.** `config/prompts.yaml` holds only the legs that differ from the
built-ins. Text equal to a default clears the override; an empty store deletes
the file. It is gitignored.

**Why.** The repo default should be the *absence* of a file, not a stale copy
of the defaults that drifts from the code. Committing one machine's wording
would silently change what every other machine sends.

**Cost.** Tuning does not travel between hosts. Deliberate.

---

## 2026-09-15 · A fourth UI rather than a fourth mode of one UI

**Decision.** `demo_dash_modes.py` is a separate file that imports renderers
from the frozen `demo_dash.py`.

**Why.** `demo_dash.py` is the fallback that is known to work end to end on
FALCONPRD against the live GPU server. Keeping it runnable is worth the
duplication of two renderers.

**Cost.** `quality_column` and `detection_column` exist twice, with the copies
handling mode 3's cases. `FROZEN.md` records the SHA to restore from.

---

## 2026-09-10 · Class names live in config, read from the data

**Decision.** `yolox_class_names` is pinned in `config.py`, index 0 =
`GPS Antenna`, taken from the training run's own `classes.json`.

**Why.** YOLOX stores no class names in a checkpoint. A wrong order does not
error — it puts a confident wrong label on screen *and* into the VLM prompt as
evidence the model is told to weigh.

**How it went wrong.** The detector was reported as swapping the two classes.
The config was correct; `smoke_yolox.py` carried
`--classes default="hazard_sign,gps_antenna"`, overriding it. Two guard tests
now assert no `--classes` string default exists anywhere and that no pipeline
module besides `config.py`/`questions.py` defines the names.

**Lesson recorded.** Trace to the source before flipping a constant. Flipping
it would have "fixed" the smoke tool and broken the UI.

---

## 2026-09-10 · `yolox_input_size` is `(640, 480)`

**Decision.** Pinned to the training run's exp table, not the stock 640×640.

**Why.** `test_size` is not stored in a checkpoint; it has to come from the
training config.

**Correction.** An earlier comment here claimed a wrong size "shifts every
box". The test proved otherwise: for a portrait photograph the height limits
the scale either way, so the boxes land identically. Only landscape diverges,
by 4/3. The comment and the test were corrected.

---

## 2026-09-09 · Vendor the HTTP contract, do not import it

**Decision.** `stage3_vlm.py` is self-contained rather than importing from
`vlm_test_client_v2.py`.

**Why.** That client does not run on the demo host, so there is nothing to
import. The contract is lifted, and the three traps in it are guarded
explicitly: the registry bootstrap, arrays-never-paths, and the silent blank
system prompt.

---

## 2026-09-09 · One question registry, per-question class names

**Decision.** Each `Question` carries its own `default_class_names`.

**Why.** Each CVAT export is a separate single-class task, so every one numbers
its only class `0`. The same id means a hazard sign in one export and a GPS
antenna in another. A single shared `classes.txt` cannot express that, and
guessing injects a false label into the VLM prompt.

---

## Standing rules

**Never let a mock assert.** Mock and parse-failure paths return `unknown`, and
`poor` for quality, always labelled as mock. A partial answer that looks
complete is worse than an obvious mock.

**A silent failure gets a named error.** Where a wrong configuration would
produce plausible-looking output instead of an error, the code refuses and the
message names the cause — the empty registry, the blank system prompt, the
missing `yolox/models`, the proxy 404 that means the wrong transport.

**Tests run without heavy dependencies.** `cv2`, `dash` and `requests` are
stubbed; `torch`, `rembg`, `rapidocr` and `pandas` are never imported. Verified
by running every suite with all of them blocked. `pyyaml` is the exception —
two suites assert saving and parsing that genuinely need it, and the registry
falls back to built-ins without it.

A suite may carry a section that needs a real dependency, provided it
**announces itself when skipped** and the stub is conditional. `test_ocr.py`
stubs `cv2` only when a real one is absent, so its live-engine section can run
against the committed ONNX files where they can be loaded, and is skipped with
a printed line where they cannot.

**A claim is never rendered as a measurement.** Anything the model asserts
about geometry is drawn in a visual language of its own, captioned as a claim,
and scored against something measured where that is possible. "We could not
measure" and "it scored zero" are opposite findings and never share a
rendering.

**Three different nothings are never conflated.** A stage that could not run,
a stage that ran and found nothing, and a stage nobody asked to run are three
separate states with three separate renderings — only the first is a problem
with the host. The same rule that separates "the detector found nothing" from
"nothing was ever trained to find this".

**Tests must not depend on their host.** Clear the environment you read;
force the absence you assert.
