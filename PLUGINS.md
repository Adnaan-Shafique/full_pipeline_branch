# Adding and removing questions, domains and object classes

Everything the pipeline asks about lives in three YAML files. Editing them is
the whole procedure — no Python changes, no restart if you press **Reload
config/** in the UI on 7873.

```
config/
  domains.yaml          the domain dropdown
  classes.yaml          every object class, trained or not
  questions/
    site_safety.yaml    the two original questions
    infra.yaml          the site-infrastructure checklist
```

Files under `config/questions/` are read in filename order and merged, so a new
domain can have a file of its own. A question's position **within** its file is
the order it appears in the dropdown, which is the order the checklist is
walked in.

## Add a question

The minimum is eight fields:

```yaml
  - id: battery_bank_condition          # unique across every file
    domain: infra                       # must exist in domains.yaml
    label: Is the battery bank in good condition?      # the dropdown text
    subject: a battery bank             # mode 3: "is {subject} visible?"
    question_text: is the battery bank in good condition?   # mode 3's answer leg
    yes_means: the cells are clean, upright and free of leakage or bulging.
    no_means: cells are leaking, bulging, corroded at the terminals, or displaced.
    classes: [battery_bank]             # ids from classes.yaml
    system_prompt: >-
      You are a telecom site inspector judging ... Respond with JSON only.
```

Everything else is derived:

| Derived | From |
|---|---|
| `answer_semantics` | `YES = {yes_means}  NO = {no_means}` |
| `user_template` | built with `{detection_block}`, `{ocr_block}` and `{output_contract}` in the right places |
| mode 3's three legs | `subject`, `question_text` and `system_prompt` |
| what detections are surfaced | the names and aliases of the classes you listed |

Write an explicit `user_template` only when the builder cannot express what you
need. The two Site Safety questions carry one because every recorded timing and
answer for this demo was produced with that exact wording, and re-deriving it
would silently invalidate all of it.

### Two rules the loader enforces at import

Both guard failures that are **silent at runtime**, which is why they are
errors rather than warnings:

- **`system_prompt` must not be blank.** The GPU server skips a falsy system
  turn, so the model answers with no framing at all — which reads as a
  model-quality problem rather than the wiring bug it is.
- **A template must contain `{detection_block}` and `{output_contract}`**, and
  `{ocr_block}` too if the question uses OCR. Without the last one, OCR runs,
  costs its seconds, and its text is discarded on the way to the prompt.

A question that trips either is **dropped with a warning** — it does not appear
in the dropdown. Press **Reload config/** and read the warnings, or run
`python tools/preflight.py --skip-gpu`, which prints every one.

## Give a question an OCR stage

```yaml
    ocr:
      enabled: true
      scope: boxes_then_image     # boxes | image | boxes_then_image
      min_confidence: 0.30        # lines below this are discarded
```

| scope | behaviour |
|---|---|
| `boxes_then_image` | OCR the relevant detected boxes; if they yield nothing, read the whole frame. The right default while a class has no trained weights. |
| `boxes` | boxes only. Reports nothing when there are none. |
| `image` | always the whole frame, boxes ignored. |

For a reading checked against a limit, add a numeric rule:

```yaml
      numeric:
        label: Temperature
        unit: "°C"
        comparator: "<="          # < | <= | > | >=
        limit: 35.0
        pattern: '(?<![\w.])(-?\d{1,3}(?:[.,]\d{1,2})?)\s*[Cc]?(?![\w])'
```

The pattern's **first capturing group** is the number. The lookarounds are not
decoration: without the leading one, the `55` of `IP55` on an enclosure in the
same frame is read as a temperature.

**The rule produces evidence, never the answer.** It takes the first number its
pattern matches, and it cannot see a range multiplier on a rotary switch, tell
a set-point from a measurement, or tell °F from °C. On `SET 22 C ACT 38.0 C` it
matches the set-point. That is why its output goes to the model as one more
advisory line, why the card shows *which token* it matched, and why the model
still answers. See `DECISIONS.md`.

### `system_prompt_no_ocr` — required whenever OCR is on

Mode 3 runs no OCR, by design. If your `system_prompt` mentions OCR evidence,
mode 3's answer leg would promise the model something it never receives — in
the one mode meant to show what the model reads unaided. So write a second
prompt for it:

```yaml
    system_prompt_no_ocr: >-
      ... No OCR output is provided here: read the text yourself, directly from
      the image, and say in your reasoning what you could make out. ...
```

Modes 1 and 2 keep `system_prompt`. Preflight warns about any OCR question
missing this.

### Mode 3 and claimed boxes

Nothing to configure per question — mode 3 asks its presence and answer legs
for coordinates whenever `vlm_grounding` is on, whatever the question. Two
things are worth knowing when you add one:

- **A question with no trained class gets unscored boxes.** The claim is drawn,
  but there is no detector box to compare it against, so the card says *"no
  trained detector box to compare against"* rather than a number. That is
  every Infra question today. Give a class trained weights and the IoU appears
  with no further edit.
- **`subject` is what the presence box gets labelled with.** It is already the
  wording of the presence leg's question, so a vague subject produces both a
  vague question and a vague caption.

## Add or remove an object class

```yaml
  - id: battery_bank
    name: Battery bank              # what appears on a box and in the prompt
    aliases: [battery, cells, vrla] # widen matching only; never narrow it
    trained: false
```

`aliases` exist because real labels are written for humans: a checkpoint says
`GPS Antenna`, an annotation folder says `gps_antenna`, a CVAT export says
something else again. Matching folds case, punctuation and separators and
compares substrings in both directions, so short aliases are deliberately
broad. The id is a match term too.

### Trained classes and `yolox_index`

```yaml
  - id: gps_antenna
    name: GPS Antenna
    trained: true
    yolox_index: 0
```

`PipelineConfig.yolox_class_names` is derived from the `trained: true` entries
**ordered by `yolox_index`** — not by position in the file. That is deliberate.
YOLOX stores no names in a checkpoint, so this file is the only record of them,
and a wrong order does not error: it puts a confident wrong label on screen
*and* into the VLM prompt as evidence. It has already happened here once, with
the detector localising hazard signs correctly and calling them GPS antennas.
Reordering the file for readability must not be able to cause that.

Set `trained: true` only after a retrain that actually produced that index,
never in anticipation. The loader demotes to untrained, with a warning, any
class that is `trained: true` with no index, that duplicates another's index,
or that sits after a gap in the 0..N range.

**Removing a class** that a question still names is safe: the question loads, a
warning says the class is unknown, and detections of it are no longer surfaced.
Removing a *trained* class means retraining — the checkpoint still has that
head output.

## Add a domain

```yaml
  - id: power
    label: Power
    order: 3
    blurb: >-
      One or two sentences shown under the dropdown.
```

A domain with **no questions is dropped from the dropdown** rather than shown
empty: an empty domain on screen reads as "the questions failed to load", which
is the wrong story when the truth is that nobody has written them yet.

## Removing things

Delete the entry. There is no tombstone and no migration. A run already written
under `demo_runs/` keeps the question id in its CSV, which is why ids should be
retired rather than reused for something else.

## Checking your edit

```bash
python tools/preflight.py --skip-gpu     # prints every warning, and the totals
python tests/test_registry.py            # 65 assertions over the loader
python tools/smoke_ocr.py --list x       # the questions that use OCR, and their rules
```

Then press **Reload config/** on 7873. The status line names how many questions
loaded, from where, and every problem found.

## What a broken file costs

Nothing takes the demo down. Each of the three files stands or falls on its
own, every problem the loader can survive becomes a warning the UI prints, and
the entry that caused it is dropped:

| Broken | Result |
|---|---|
| one question entry | that question is absent; the rest load |
| `questions/` yields nothing | the two built-in Site Safety questions fill in; **the classes and domains you did load are kept** |
| `classes.yaml` unparseable | the two built-in trained classes; questions still load |
| `domains.yaml` missing | the built-in `Site Safety` domain only |
| pyyaml not installed | the two built-in questions, and a line saying so in every status bar |

The last one is why `backend/pipeline/registry.py` keeps `BUILTIN_QUESTIONS` and
`BUILTIN_CLASSES` as Python. `tests/test_yolox.py` pins those copies against
the YAML so a fallback that has drifted — which would relabel every detection
with nothing raising — fails a suite instead.
