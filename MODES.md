# Three pipeline modes

`frontend/demo_dash_modes.py` on **port 7872**. One click runs all three modes over
the same photographs; the mode selector then switches between three sets of
results without re-running anything.

| | Mode 1 | Mode 2 | Mode 3 |
|---|---|---|---|
| Name | Quality gate → Detector → OCR → Model | (Quality **OR** Detector) → OCR → Model | Everything by the model |
| Quality | MM-IQA + u2netp, **hard gate** | MM-IQA + u2netp, advisory | the model judges it |
| Detection | YOLOX-S, boxes | YOLOX-S, boxes | the model reports presence, **no boxes** |
| OCR | PP-OCRv6 | PP-OCRv6 | **none** — the model reads the text |
| Question | Qwen3-VL | Qwen3-VL | Qwen3-VL, a third call of its own |
| A blurry photo whose sign is clearly detected | **dropped** | **answered** | answered if the model calls it usable |

## What each mode is arguing

**Mode 1** — the pipeline as built. Quality decides first, and a photograph
that fails is never looked at again. Cheap, and defensible: you don't want an
inspection decision made from an unusable photograph.

**Mode 2** — the quality score is advisory, not final. A photo proceeds if the
gate passes **or** the detector found the subject. The argument: a soft-focus
shot where the warning sign is plainly detected at 0.94 is not a photo you
should throw away. Every card in this mode states *why* it was let through.

**Mode 3** — no u2netp, no YOLOX, **and no OCR**. The vision model makes all
three judgements itself, as **three separate calls** over the same photograph:
is it usable, is the subject there, and what is the answer. The argument: one
model that sees the whole photograph may judge those more coherently than three
components that each see a slice.

They are three calls rather than one so that each leg has its **own system
prompt**, editable in the UI, and so that a wording change to one judgement
cannot drag the other two with it. The image is encoded once and reused across
all three, so the extra calls cost tokens, not another decode.

Mode 2 is **strictly more permissive** than mode 1 — never less. The only case
where they differ is the interesting one: quality failed, the detector found
the subject anyway.

## Comparing them

The **Compare modes** tab puts one row per photograph against one column per
mode, and counts how often all three agree. A row where they disagree is the
one worth talking about: same photograph, three ways of deciding, different
conclusions.

## Cost

Running all three does **not** cost three times as much. Each photograph is
loaded once, quality-scored once and detected once; modes 1 and 2 read the same
results and differ only in what they do with them. When both gates open, the
model sees identical inputs, so that answer is computed once and shared.

Modes 1 and 2 share a single model call when both gates open; a second is only
needed when mode 1 stops a photo that mode 2 lets through. Mode 3 adds three
calls of its own — one per leg. Typical cost is therefore **four model calls per
photograph** for all three modes.

OCR is shared the same way: one pass, read by both modes 1 and 2, and skipped
entirely unless at least one of them is going to ask the model something. On a
question that does not use OCR it costs nothing at all.

On current timings (~2s quality, ~3s detection, ~0.06s per detected box or
~0.34s for a whole frame of OCR, ~0.4s per model call) that is roughly **6.5s
per photograph** for all three modes, against ~5.5s for one.

## Mode 3 and OCR

Four questions carry an OCR stage: the two SPD "installed" questions, decided
by the class marking printed on the module body, and the two device readings,
where the answer is a number on a display. Modes 1 and 2 run PP-OCRv6 on them
and hand the model the text. **Mode 3 does not**, and that is the single most
interesting thing on the screen.

Handing mode 3 the transcription would make it answer *better* and tell you
*nothing*. Left alone, its answer is the model reading a seven-segment display
or a weathered module label unaided — which is exactly the question anyone
evaluating this pipeline for an edge device is actually asking. Where mode 3
disagrees with modes 1 and 2 on a reading, the Stage detail tab on 7873 shows
what PP-OCRv6 read next to what the model says it saw, and the disagreement is
usually legible in one glance.

This asymmetry also means mode 3 needs its own wording. A question with an OCR
stage tells the model in its system prompt that OCR text "may be supplied to
you as advisory evidence" — and in mode 3 it never is. Promising evidence that
does not arrive is the worst available framing for the one mode meant to
measure unaided reading, so those four questions carry a `system_prompt_no_ocr`
that mode 3's answer leg uses instead, telling the model to read the text
itself and say what digits it saw. See `PLUGINS.md`.

The OCR output is **advisory even in modes 1 and 2**. It arrives behind a hedge
telling the model to prefer the image, and where a question carries a numeric
rule, the threshold check arrives the same way — as a computed line of
evidence, never as the answer. `DECISIONS.md` has the three failure modes that
make that non-negotiable.

## Mode 3's claimed boxes

**This reverses an earlier decision, and the reasoning is worth keeping.** Mode
3 used to draw no boxes at all. The argument was that Qwen3-VL's grounding
accuracy is well below YOLOX's and a visibly wrong box is worse than an honest
"presence, not geometry" — which is true, but was being *asserted* rather than
shown, and it left the most-asked question about this pipeline ("where does the
model think it is?") unanswerable.

Mode 3 now asks the presence and answer legs for coordinates and draws what
comes back, under three conditions that between them make it honest:

1. **It cannot be mistaken for a detection.** Claimed boxes are drawn
   **dashed**, in alternating white and black so they stay legible on sky and
   on dark equipment alike, and captioned `model says: …`. Nothing else in this
   demo is dashed — stage 1 is a solid green foreground box, stage 2 solid
   per-class colours, stage 2b magenta text polygons. A viewer who has learned
   those three reads a dashed box as a different kind of claim before reading
   the caption.
2. **It arrives with its own error bar.** Modes 1 and 2 have already run the
   detector over the same photograph in the same frame, so where a trained
   class exists the claimed box is scored against the detector's box and the
   **IoU is printed in the caption and on the card**. The old objection becomes
   a number on screen instead of a sentence in this file. Where no trained
   class exists — every Infra question today — the card says *"no trained
   detector box to compare against"*, which is not the same as scoring zero and
   must never be rendered as if it were.
3. **It is a claim, never an input.** The boxes live in `claimed_boxes`, not in
   `detections`. Everything that reads `detections` — the VLM prompt's
   detection block, `select_relevant()`, the full+crop chooser — would treat
   them as detector output, and a claimed region fed back into a prompt would
   have the model citing itself as evidence. The detections used for scoring
   arrive *after* the three legs have answered and never reach a prompt.

A box that cannot be placed confidently is **refused**, and the card says the
model claimed a region that could not be placed. Refusals include a box
covering the whole frame (that is the model declining to localise while
appearing to comply), a degenerate or negative box, and absolute coordinates
that cannot be scaled. `tests/test_grounding.py` pins each one.

Turn it off with the *"mode 3: presence in words only"* option on 7873, or
`vlm_grounding=False`, if the boxes turn out to be noise on a given photo set.

### The coordinate trap

`array_to_data_uri()` **downscales the photograph to 2048px** before sending
it, so the model never sees the original frame. An absolute pixel coordinate it
replies with is in the *resized* frame, and drawing it on a 4000×3000 original
puts every box out by about 2× with nothing raising — the same shape of bug as
this repo's "arrays, never paths" rule, one stage later.

Two defences: the prompt asks for **integers normalized 0–1000**, the
convention the Qwen-VL family was trained on and the only one immune to the
resize; and `vlm_grounding.interpret_box()` accepts the other conventions
anyway, scales them from the encoded frame, and **reports which reading it
used** on the card. A systematic misread then shows up as every box on every
photograph arriving as `absolute_px_encoded_frame`, rather than as boxes that
are quietly wrong.

### The quality leg is not grounded

Sharpness, exposure and framing are properties of the whole frame. A box drawn
around "the blurry part" would be fabricated precision, and unlike the other
two legs there would be nothing to score it against. Only the presence and
answer legs are asked to point.

### Presence box vs evidence box

They are different claims and are kept apart. The presence leg is asked *where
the subject is*; the answer leg is asked *what it looked at to decide*. On the
OCR questions the gap between them is the interesting part — mode 3 pointing at
the display it read can be put straight next to the region PP-OCRv6 read from
in modes 1 and 2.

## What mode 3 still does NOT do

Mode 3 has no numeric quality score. The card says "judged by the model,
no numeric score" rather than printing a fabricated number next to PASS, and the
quality panel omits the MM-IQA line entirely rather than showing `0.0`.

It renders the photograph as it was judged: the EXIF-corrected original, with
**no green foreground box and no detector boxes** — neither u2netp nor YOLOX ran
in this mode, so drawing either would credit a component that did not run. The
only thing that may appear on it is mode 3's own dashed claim.

## Tuning mode 3's prompts

The **Mode 3 prompts** card in the sidebar holds one editable system prompt per
leg. The *user* prompt below each is shown but fixed: it carries the JSON
contract the parser keys on, and an edit there would silently turn every answer
into `unknown`.

Edits are per question — switching the question dropdown switches the whole set
— and **Save** writes only the legs you actually changed to `config/prompts.yaml`.
Text equal to the built-in default clears the override; when nothing is
overridden the file is deleted, so the repo default is the absence of a file, not
a copy of it. A malformed file falls back to the built-ins and says so in the
status line rather than failing the run.

**Re-run mode 3 only** re-runs the three legs over the photographs already
loaded, reusing modes 1 and 2's existing results. Quality scoring and detection
are the expensive stages and a prompt change cannot affect them, so re-running
everything would spend them to produce identical output. The comparison tab
still lines up row for row afterwards.

## Reaching the model

All three modes call the same GPU server, either directly or through
`llm_proxy_v3` on FALCONPRD when the host cannot see the GPU box. Mode 3 is the
most affected by the choice: it makes three calls per photograph where modes 1
and 2 share one. See `PROXY.md`.

## Files

| Path | Role |
|---|---|
| `backend/pipeline/modes.py` | The three modes, the shared-work runner, comparison and agreement |
| `backend/pipeline/questions.py` | The three legs, their default system prompts and their fixed user prompts |
| `backend/pipeline/prompts.py` | `PromptStore` — reads and writes `config/prompts.yaml` |
| `backend/pipeline/stage2b_ocr.py` | Stage 2b — scope, the confidence floor, the numeric rule |
| `backend/pipeline/vlm_grounding.py` | Reading, refusing and scoring the model's coordinates |
| `backend/pipeline/stage3_vlm.py` | `ask_leg()`, `ask_vlm_only()`, `parse_quality()`, `parse_presence()` |
| `frontend/demo_dash_modes.py` | The batch UI, 7872 |
| `frontend/demo_dash_pipeline.py` | The per-photograph UI, 7873 |
| `config/questions/*.yaml` | The questions themselves — see `PLUGINS.md` |
| `tests/test_modes.py` | 116 assertions |
| `tests/test_ocr.py` | 85 assertions over stage 2b |
| `tests/test_grounding.py` | 79 assertions over mode 3's claimed boxes |

Each mode writes its own `pipeline_results.csv` and `results.json` under
`demo_runs/<run_id>/<mode>/`. The CSV carries nine OCR columns, including
`ocr_numeric_value` and `ocr_numeric_passes` — both blank when no reading was
matched, which is not the same as a failed check and must not be read as one.
