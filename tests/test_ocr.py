"""Stage 2b - OCR: scope selection, the confidence floor, the numeric rule, the
prompt block, and every way the stage is allowed to produce nothing.

Runs with rapidocr absent, which is the normal case on a test host: the engine
is faked so that the stage's own decisions are what get tested rather than
PP-OCRv6's accuracy. The section at the end that needs the real engine
announces itself and is skipped when it is not installed.
"""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

# The other suites stub cv2 unconditionally. This one must not: the live-engine
# section at the end needs a REAL cv2 to draw its test image, and a stub
# installed here wins over the real module for the rest of the process. So the
# stub goes in only when cv2 is genuinely absent - which keeps the "runs with
# nothing installed" property while letting the live section run where it can.
try:
    import cv2  # noqa: F401
except ImportError:
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok   {name}")
    else:
        failed += 1
        print(f"  FAIL {name}" + (f"\n         {detail}" if detail else ""))


from pipeline import ocr_engine, stage2b_ocr                          # noqa: E402
from pipeline.question_types import (NumericRule, OCRSpec, Question,  # noqa: E402
                                     build_ocr_block, build_user_template,
                                     render_user_prompt)
from pipeline.questions import get_question                           # noqa: E402

try:
    import yaml  # noqa: F401
    HAVE_YAML = True
except ImportError:
    # Without pyyaml the registry serves two built-in Site Safety questions and
    # none of the four that use OCR exists. Sections that name them by id would
    # raise KeyError - a crash rather than a reported failure, which breaks the
    # "every suite prints N passed, M failed" convention this repo runs on.
    HAVE_YAML = False
from pipeline.schemas import (Detection, OCRLine, OCRStageResult,     # noqa: E402
                              OCR_FROM_BOXES, OCR_FROM_IMAGE, OCR_SKIPPED,
                              SOURCE_ANNOTATION)


def question(ocr=None, **kw):
    spec = ocr or OCRSpec()
    return Question(
        id=kw.pop("id", "q"), label="Q?", system_prompt="S. Respond with JSON only.",
        user_template=build_user_template("is it?", "yes.", "no.", with_ocr=spec.enabled),
        answer_semantics="", relevant_classes=["Device reading"], ocr=spec, **kw)


def det(label="Device reading", conf=0.9, box=(10, 10, 100, 60)):
    return Detection(label=label, confidence=conf, box=list(box), source=SOURCE_ANNOTATION)


class FakeEngine:
    """Stands in for ocr_engine. Records what it was asked to read so the
    stage's scope decisions are observable, not inferred from the output."""

    def __init__(self, image_lines=None, box_texts=None, raises=False):
        self.image_lines = image_lines if image_lines is not None else []
        self.box_texts = box_texts or {}
        self.raises = raises
        self.image_calls = 0
        self.box_calls = []

    def read_image(self, image_bgr, variant=None, model_dir=None):
        if self.raises:
            raise RuntimeError("onnxruntime fell over")
        self.image_calls += 1
        return list(self.image_lines), 120.0

    def read_box(self, image_bgr, box, variant=None, model_dir=None):
        if self.raises:
            raise RuntimeError("onnxruntime fell over")
        self.box_calls.append(list(box))
        text, conf = self.box_texts.get(tuple(box), ("", 0.0))
        return {"text": text, "text_confidence": conf, "ocr_ms": 30.0}

    default_variant = staticmethod(lambda model_dir=None: "tiny")
    describe = staticmethod(lambda model_dir=None: "FakeOCR")
    # The stage reads this to derive a model dir when no cfg is passed.
    OCR_MODEL_DIR = ROOT / "models" / "ocr"


def with_fake(fake, fn, cfg=None):
    """Run fn() with the stage's engine and availability check faked out.

    The stage does `from . import ocr_engine` INSIDE its functions, which
    resolves the attribute on the already-imported `pipeline` package rather
    than going back to sys.modules. Both have to be swapped, or the fake is
    built, never called, and every assertion below reads the real engine's
    "rapidocr is not installed" instead.
    """
    import pipeline

    real_module = sys.modules.get("pipeline.ocr_engine")
    real_attr = getattr(pipeline, "ocr_engine", None)
    real_available = stage2b_ocr.available
    sys.modules["pipeline.ocr_engine"] = fake
    pipeline.ocr_engine = fake
    stage2b_ocr.available = lambda cfg=None: (True, "FakeOCR")
    try:
        return fn()
    finally:
        sys.modules["pipeline.ocr_engine"] = real_module
        pipeline.ocr_engine = real_attr
        stage2b_ocr.available = real_available


def line(text, conf=0.9, box=(0, 0, 10, 10)):
    return {"text": text, "text_confidence": conf, "box": list(box),
            "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]}


print("\nthe stage runs only for the questions that asked for it")
r = stage2b_ocr.run(None, question(), [])
check("a question with ocr disabled is skipped", r.scope_used == OCR_SKIPPED)
check("and costs nothing", r.elapsed_ms == 0.0)
check("and is not an error", r.error is None)
check("the card says so rather than showing an empty read",
      r.headline == "not run for this question", r.headline)
check("skipped() returns a fresh object each time, not a shared singleton",
      stage2b_ocr.skipped("a") is not stage2b_ocr.skipped("b"))
a, b = stage2b_ocr.skipped("a"), stage2b_ocr.skipped("b")
check("so one record's note cannot leak into another's", a.note != b.note)

print("\nscope: boxes_then_image")
spec = OCRSpec(enabled=True, scope="boxes_then_image")
fake = FakeEngine(box_texts={(10, 10, 100, 60): ("33.5 C", 0.94)})
r = with_fake(fake, lambda: stage2b_ocr.run(object(), question(ocr=spec), [det()]))
check("a detection is read as a box", r.scope_used == OCR_FROM_BOXES)
check("the whole frame is not touched when a box yields text", fake.image_calls == 0)
check("the line carries the detection's own label",
      [ln.from_label for ln in r.lines] == ["Device reading"])
check("the line's box is the detection's, in full-frame coordinates",
      r.lines[0].box == [10.0, 10.0, 100.0, 60.0])

fake = FakeEngine(image_lines=[line("33.5 C", 0.94)])
r = with_fake(fake, lambda: stage2b_ocr.run(object(), question(ocr=spec), []))
check("no detections falls through to the whole frame", r.scope_used == OCR_FROM_IMAGE)
check("and says why on the card", "no relevant detections" in r.note, r.note)

fake = FakeEngine(image_lines=[line("33.5 C", 0.94)],
                  box_texts={(10, 10, 100, 60): ("", 0.0)})
r = with_fake(fake, lambda: stage2b_ocr.run(object(), question(ocr=spec), [det()]))
check("a box with no text ALSO falls through to the whole frame",
      r.scope_used == OCR_FROM_IMAGE)
check("the fallback is explained, not silent",
      "read the whole frame instead" in r.note, r.note)
check("and the text is the frame's", r.text == "33.5 C")

print("\nscope: boxes, and scope: image")
boxes_only = OCRSpec(enabled=True, scope="boxes")
fake = FakeEngine(image_lines=[line("should never be read")])
r = with_fake(fake, lambda: stage2b_ocr.run(object(), question(ocr=boxes_only), []))
check("scope 'boxes' with no detections reads nothing at all", fake.image_calls == 0)
check("it is reported as a box scope with zero boxes",
      r.scope_used == OCR_FROM_BOXES and r.boxes_read == 0)
check("no legible text is not an error", r.error is None)
check("the headline distinguishes it from a failure",
      "no legible text" in r.headline or r.lines == [], r.headline)

image_only = OCRSpec(enabled=True, scope="image")
fake = FakeEngine(image_lines=[line("whole frame")],
                  box_texts={(10, 10, 100, 60): ("box text", 0.99)})
r = with_fake(fake, lambda: stage2b_ocr.run(object(), question(ocr=image_only), [det()]))
check("scope 'image' ignores the boxes even when there are some",
      r.scope_used == OCR_FROM_IMAGE and fake.box_calls == [])
check("and says the boxes were deliberately unused", "not used" in r.note, r.note)

print("\nboxes are read highest-confidence first, and only the relevant ones")
fake = FakeEngine(box_texts={(0, 0, 5, 5): ("low", 0.9), (9, 9, 20, 20): ("high", 0.9)})
r = with_fake(fake, lambda: stage2b_ocr.run(
    object(), question(ocr=spec),
    [det(conf=0.4, box=(0, 0, 5, 5)), det(conf=0.95, box=(9, 9, 20, 20))]))
check("the most confident detection is read first",
      fake.box_calls[0] == [9, 9, 20, 20], str(fake.box_calls))
check("every relevant detection is read", r.boxes_read == 2)
check("the text is joined in that order", r.text == "high low", r.text)

print("\nthe confidence floor runs before the prompt and before the rule")
floor = OCRSpec(enabled=True, scope="image", min_confidence=0.5,
                numeric=NumericRule(label="Reading", unit="V", comparator="<",
                                    limit=10.0, pattern=r"(\d+(?:\.\d+)?)"))
fake = FakeEngine(image_lines=[line("185", 0.11), line("1.85", 0.97)])
r = with_fake(fake, lambda: stage2b_ocr.run(object(), question(ocr=floor), []))
check("a line below the floor is discarded", r.text == "1.85", r.text)
check("the discard is reported, not silent",
      "confidence floor" in r.note, r.note)
# The worst output this stage can produce is a fabricated measurement with a
# verdict attached, so the floor must run BEFORE the rule sees the text.
check("the rule never sees the discarded line", r.numeric["value"] == 1.85)
check("and reaches the opposite verdict to the noise it dropped",
      r.numeric["passes"] is True)

print("\nthe numeric rule produces evidence, and its absence is not a failure")
nrule = OCRSpec(enabled=True, scope="image",
                numeric=NumericRule(label="Temperature", unit="C", comparator="<=",
                                    limit=35.0, pattern=r"(\d+(?:\.\d+)?)"))
fake = FakeEngine(image_lines=[line("display is off", 0.9)])
r = with_fake(fake, lambda: stage2b_ocr.run(object(), question(ocr=nrule), []))
check("text with no number yields no numeric result", r.numeric is None)
check("which is reported as no reading, not a failed check",
      "could be matched" in r.note, r.note)
check("and is still not an error", r.error is None)

fake = FakeEngine(image_lines=[])
r = with_fake(fake, lambda: stage2b_ocr.run(object(), question(ocr=nrule), []))
check("no lines at all means the rule is never run", r.numeric is None)

print("\nevery way the stage is allowed to fail")
r = with_fake(FakeEngine(raises=True),
              lambda: stage2b_ocr.run(object(), question(ocr=spec), [det()]))
check("an engine that raises becomes an error, not an exception",
      r.error is not None and "OCR failed" in r.error, str(r.error))
check("a failed read produces no lines", r.lines == [])
check("the headline leads with ERROR", r.headline.startswith("ERROR"), r.headline)

real_available = stage2b_ocr.available
stage2b_ocr.available = lambda cfg=None: (False, "rapidocr is not installed")
try:
    r = stage2b_ocr.run(object(), question(ocr=spec), [det()])
finally:
    stage2b_ocr.available = real_available
check("a missing dependency is an error the operator can act on",
      r.error == "rapidocr is not installed", str(r.error))
check("it does not pretend the stage was merely skipped", r.ran is False)

print("\nan error must never become a line of prompt")
for bad in (OCRStageResult(lines=[], engine="", scope_used=OCR_SKIPPED,
                           error="rapidocr is not installed"),
            OCRStageResult(lines=[], engine="e", scope_used=OCR_FROM_IMAGE),
            None,
            stage2b_ocr.skipped("not used")):
    check(f"renders as no block at all ({bad.error if bad else 'None'})",
          build_ocr_block(bad) == "")
# The reverse: a real read must reach the model, with its hedge and its numbers.
good = OCRStageResult(
    lines=[OCRLine(text="33.5 C", text_confidence=0.94, box=[0, 0, 1, 1])],
    engine="e", scope_used=OCR_FROM_IMAGE,
    numeric={"value": 33.5, "passes": True, "matched": "33.5",
             "sentence": "Temperature read as 33.5 C, which is within the limit."})
block = build_ocr_block(good)
check("a real read carries the text", '"33.5 C"' in block, block)
check("with its per-line confidence", "0.94" in block)
check("behind the advisory hedge", "trust the image over this text" in block)
check("and the threshold check is labelled as computed",
      "Threshold check computed from that text" in block)

print("\nthe prompt the model actually receives")
q = question(ocr=spec)
prompt = render_user_prompt(q, [det()], good)
check("carries the detection block", "Object detector output" in prompt)
check("carries the OCR block", "Text read from the image" in prompt)
check("carries the output contract", '"answer"' in prompt)
check("in that order",
      prompt.index("Object detector output") < prompt.index("Text read from the image")
      < prompt.index('"answer"'))
check("and has no stray blank lines", "\n\n\n" not in prompt)
empty = render_user_prompt(q, [], None)
check("an absent OCR read collapses cleanly",
      "Text read from the image" not in empty and "\n\n\n" not in empty)

print("\nmode 3 never receives an OCR block, by design")
import pipeline.questions as pq  # noqa: E402
if not HAVE_YAML:
    print("  (pyyaml absent - the questions that use OCR do not exist; skipped)")
if HAVE_YAML:
  for qid in ("temp_within_limit", "spd_class_b_installed"):
      leg = pq.render_leg_user(pq.get_question(qid), "answer")
      check(f"{qid}: the mode-3 answer leg has no OCR block",
            "Text read from the image" not in leg and "{ocr_block}" not in leg)
  check("the mode-3 legs take no ocr argument at all",
        "ocr" not in pq.render_leg_user.__code__.co_varnames)

  # A question whose system prompt promises OCR evidence must not send that same
  # prompt in the one mode where the evidence never arrives. This was a real bug:
  # all four OCR questions told mode 3 that "text read from the image by an OCR
  # engine may be supplied to you", and nothing ever supplied it.
  for qid in ("temp_within_limit", "earthing_value_egb",
              "spd_class_b_installed", "spd_class_c_installed"):
      q = pq.get_question(qid)
      m3 = pq.default_leg_system(q, "answer")
      check(f"{qid}: modes 1-2 and mode 3 get DIFFERENT system prompts",
            q.system_prompt != m3)
      check(f"{qid}: modes 1-2 are told OCR text may be supplied",
            "may be supplied" in q.system_prompt)
      check(f"{qid}: mode 3 is NOT promised evidence it never gets",
            "may be supplied" not in m3, m3[:120])
      check(f"{qid}: mode 3 is told to read the text itself",
            "No OCR output" in m3, m3[:120])
      check(f"{qid}: mode 3's prompt is not blank", bool(m3.strip()))
  for qid in ("gps_antenna", "hazard_warning", "earth_pit_condition"):
      q = pq.get_question(qid)
      check(f"{qid}: no OCR stage, so one prompt still serves both modes",
            q.system_prompt == pq.default_leg_system(q, "answer"))
  check("every question that carries OCR also carries the mode-3 variant",
        all(q.system_prompt_no_ocr.strip()
            for q in pq.QUESTIONS.values() if q.ocr.enabled),
        str([q.id for q in pq.QUESTIONS.values()
             if q.ocr.enabled and not q.system_prompt_no_ocr.strip()]))
  # Blank-but-present is trap 9 again: the server skips a falsy system turn.
  from pipeline.question_types import Question as _Q  # noqa: E402
  try:
      _Q(id="t", label="t", system_prompt="S",
         user_template="{detection_block}{output_contract}",
         answer_semantics="", relevant_classes=[], system_prompt_no_ocr="   ")
      raised = False
  except ValueError:
      raised = True
  check("a whitespace-only mode-3 prompt is rejected at construction", raised)

print("\nthe engine module reports what is actually on disk")
present = ocr_engine.available_variants()
check("the committed tiny variant is found", "tiny" in present, str(present))
check("the default variant is one that exists",
      ocr_engine.default_variant() in present)
check("describe() names the angle classifier's absence",
      "angle classifier" in ocr_engine.describe(), ocr_engine.describe())
check("a directory with no models yields no variants",
      ocr_engine.available_variants(ROOT / "nope") == [])
check("and default_variant() returns None rather than a bad path",
      ocr_engine.default_variant(ROOT / "nope") is None)
check("describe() says so too",
      "no OCR models found" in ocr_engine.describe(ROOT / "nope"))

try:
    import rapidocr  # noqa: F401
    import cv2 as _cv2
    import numpy as _np
    HAVE_ENGINE = hasattr(_cv2, "putText")
except Exception:
    HAVE_ENGINE = False

if HAVE_ENGINE and HAVE_YAML:
    print("\nthe real engine, against the committed ONNX files")
    from pipeline.config import default_config  # noqa: E402

    cfg = default_config()
    ok, description = stage2b_ocr.available(cfg)
    check("the stage reports itself available", ok, description)
    img = _np.full((400, 700, 3), 40, _np.uint8)
    _cv2.rectangle(img, (150, 120), (560, 300), (15, 15, 15), -1)
    _cv2.putText(img, "33.5 C", (185, 240), _cv2.FONT_HERSHEY_SIMPLEX, 2.4,
                 (80, 255, 120), 6)
    real_q = get_question("temp_within_limit") if HAVE_YAML else None
    r = stage2b_ocr.run(img, real_q, [], cfg=cfg)
    check("the whole-frame read finds the display", bool(r.lines), r.headline)
    check("the reading is parsed", r.numeric is not None and r.numeric["value"] == 33.5,
          str(r.numeric))
    check("and checked against the question's own limit",
          r.numeric["passes"] is True)
    box_r = stage2b_ocr.run(img, real_q, [det(box=(150, 120, 560, 300))], cfg=cfg)
    check("the box-scoped read finds it too", bool(box_r.lines), box_r.headline)
    check("box and frame agree on the number",
          box_r.numeric["value"] == r.numeric["value"])
else:
    print("\n(rapidocr, a real cv2 or pyyaml is absent - the live-engine section "
          "is skipped)")

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
