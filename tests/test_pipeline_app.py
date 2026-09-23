"""frontend/demo_dash_pipeline.py — the domain → question → photograph → 3 modes UI.

Reuses tests/test_dash_ui.py's dash stub, so this runs with dash absent and
never starts a server. What it exercises is this screen's own decisions: the
domain/question cascade, the four different nothings the OCR panel has to
distinguish, and the honesty requirements that carry over from the frozen UI.
"""
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "tests"))
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


# The dash stub lives in tests/test_dash_ui.py and installing it is a side
# effect of running that file. It cannot simply be imported: it prints its own
# 38 assertions, and it ENDS WITH sys.exit(). A SystemExit raised during an
# import also drops the half-built module from sys.modules, so an import,
# catch, re-import loop runs the file a second time undirected and then exits
# THIS suite with 0 before its first check - silently asserting nothing.
#
# So the file is exec'd with its stdout captured and sys.exit neutralised, and
# its exit code is asserted below rather than discarded. The dash stub it
# installs into sys.modules survives either way; Node comes out of the namespace.
import contextlib  # noqa: E402
import io  # noqa: E402

_stub_ns = {"__name__": "test_dash_ui", "__file__": str(ROOT / "tests" / "test_dash_ui.py")}
_stub_out = io.StringIO()
_stub_exit = 0


def _record_exit(code=0):
    global _stub_exit
    _stub_exit = code or 0


_real_exit = sys.exit
sys.exit = _record_exit
try:
    with contextlib.redirect_stdout(_stub_out):
        exec(compile((ROOT / "tests" / "test_dash_ui.py").read_text(),
                     "test_dash_ui.py", "exec"), _stub_ns)
finally:
    sys.exit = _real_exit

Node = _stub_ns["Node"]

# demo_dash_modes builds a dcc.Textarea for mode 3's editable prompts, which
# test_dash_ui's stub does not define because the frozen UI never uses one.
# Added here rather than in test_dash_ui, which is passing and not this
# change's to edit.
for _name in ("Textarea", "Slider", "Markdown", "Graph"):
    if not hasattr(sys.modules["dash.dcc"], _name):
        setattr(sys.modules["dash.dcc"], _name, Node)
for _name in ("H2", "H3", "Table", "Tr", "Td", "Th", "Hr", "Small", "I"):
    if not hasattr(sys.modules["dash.html"], _name):
        setattr(sys.modules["dash.html"], _name, Node)


def text_of(node) -> str:
    """Flatten a rendered tree to searchable text, lists included."""
    if isinstance(node, Node):
        return node.text()
    if isinstance(node, (list, tuple)):
        return " ".join(text_of(n) for n in node)
    return str(node)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# demo_dash_modes is imported by the app for its two mode-3-aware renderers.
load("demo_dash", ROOT / "frontend" / "demo_dash.py")
load("demo_dash_modes", ROOT / "frontend" / "demo_dash_modes.py")
APP = load("demo_dash_pipeline", ROOT / "frontend" / "demo_dash_pipeline.py")

import pipeline.questions as pq                                        # noqa: E402
from pipeline.modes import MODE_CLASSIC, MODE_ORDER, MODE_VLM_ONLY     # noqa: E402
from pipeline.schemas import (Detection, DetectionStageResult,         # noqa: E402
                              OCRLine, OCRStageResult, OCR_FROM_BOXES,
                              OCR_FROM_IMAGE, OCR_SKIPPED, PipelineRecord,
                              QualityStageResult, SOURCE_ANNOTATION,
                              STOPPED_COMPLETE, VLMAnswer)


def quality(passed_=True):
    return QualityStageResult(
        passed=passed_, score=72.0, threshold=65.0, resolution_ok=True,
        ignore_resolution_used=False, failure_reasons=[], retake_instructions=[],
        foreground_box=None, segmentation_used=False, whole_frame_score=72.0,
        width=1600, height=1200)


def record(ocr=None, answer="yes", question_id="temp_within_limit"):
    return PipelineRecord(
        filename="site.jpg", source_path="/tmp/site.jpg", stem="site",
        question_id=question_id, quality=quality(),
        detection=DetectionStageResult(detections=[], annotated_path=None,
                                       model_name="human annotation", is_stub=True),
        ocr=ocr,
        vlm=VLMAnswer(answer=answer, reasoning="Because the display reads 33.5.",
                      raw_text="{}", model="qwen3-vl", elapsed_s=0.4),
        stopped_at=STOPPED_COMPLETE)


def ocr_result(**kw):
    base = dict(
        lines=[OCRLine(text="33.5 C", text_confidence=0.94, box=[1, 2, 3, 4],
                       from_label="Device reading")],
        engine="PP-OCRv6 tiny", scope_used=OCR_FROM_IMAGE, elapsed_ms=120.0)
    base.update(kw)
    return OCRStageResult(**base)


try:
    import yaml  # noqa: F401
    HAVE_YAML = True
except ImportError:
    # No pyyaml means no Infra domain and none of the OCR questions, so every
    # assertion below would raise KeyError rather than report a failure.
    HAVE_YAML = False

if not HAVE_YAML:
    print("\n(pyyaml absent - config/ cannot be read, so this screen has only "
          "the two\n built-in questions and no OCR question to render. "
          "Skipping.)")
    print("\n0 passed, 0 failed")
    sys.exit(0)

TEMP = pq.get_question("temp_within_limit")
ANTENNA = pq.get_question("gps_antenna")

print("\nthe dash stub loaded from test_dash_ui")
check("test_dash_ui's own assertions passed while installing the stub",
      _stub_exit == 0, f"test_dash_ui exited {_stub_exit}")
check("its output was captured rather than mixed into this suite's",
      "passed, 0 failed" in _stub_out.getvalue())
check("and the stub really is installed", sys.modules["dash"].Dash is not None)

print("\nthe domain → question cascade")
options, value, blurb = APP.on_domain("infra")
check("choosing Infra lists its questions", len(options) == 14, str(len(options)))
check("the value is reset to that domain's first question",
      value == options[0]["value"], str(value))
check("the domain's blurb is shown", "site-infrastructure" in blurb, blurb[:60])
options, value, _ = APP.on_domain("site_safety")
check("switching back lists Site Safety's two",
      [o["value"] for o in options] == ["hazard_warning", "gps_antenna"])
# A stale id from the previous domain would look selected and resolve to
# nothing on Run, so the reset is the point of that callback.
check("and resets the value again, never leaving a stale id",
      value == "hazard_warning")
check("an unknown domain yields no questions rather than raising",
      APP.on_domain("no_such_domain")[0] == [])
check("every domain option is reachable from the registry",
      {o["value"] for o in APP._domain_options()} == {d.id for d in pq.REGISTRY.ordered_domains()})

print("\nthe question panel says what has weights and what does not")
t = text_of(APP.question_detail("temp_within_limit"))
check("it names the untrained classes", "No trained weights" in t)
check("and says an empty box list is not evidence of absence",
      "not evidence the object is absent" in t, t[:200])
check("it says OCR runs for this question", "OCR runs for this question" in t)
check("it names the threshold that will be checked",
      "Temperature <= 35" in t, t[:400])
check("it says mode 3 reads the text itself instead",
      "Mode 3 reads the text with the model instead" in t or
      "reads the text with the model instead" in t, t[:400])

t = text_of(APP.question_detail("gps_antenna"))
check("a trained question shows no untrained warning", "No trained weights" not in t)
check("and names its trained class", "GPS Antenna" in t)
check("and says nothing about OCR", "OCR runs" not in t)
check("an unknown question id does not raise",
      "No question selected" in text_of(APP.question_detail("nope")))

print("\nthe OCR panel distinguishes four different nothings")
# 1. Ran and read something.
t = text_of(APP.ocr_column(record(ocr=ocr_result(
    numeric={"value": 33.5, "passes": True, "matched": "33.5",
             "sentence": "Temperature read as 33.5 C, within the limit."})), TEMP))
check("a real read shows the text", "33.5 C" in t)
check("with its confidence", "0.94" in t)
check("and which detection it came from", "Device reading" in t)
check("the threshold line is labelled EVIDENCE, not an answer",
      "evidence, not the answer" in t, t[:300])
check("and shows WHICH token was matched",
      "Matched the text" in t and '"33.5"' in t.replace("'", '"'), t[:400])
check("and repeats that the model may overrule it",
      "prefer the image" in t)
# The rule's verdict must not be styled as a yes/no chip - it is not the answer.
check("the threshold verdict is not rendered as an answer chip",
      "chip-yes" not in t and "chip-no" not in t, t[:300])

# 2. Ran, read nothing.
t = text_of(APP.ocr_column(record(ocr=ocr_result(lines=[])), TEMP))
check("a read that found nothing says so", "No legible text" in t)
check("and is NOT presented as an error", "could not run" not in t)

# 3. Could not run.
t = text_of(APP.ocr_column(record(ocr=OCRStageResult(
    lines=[], engine="", scope_used=OCR_SKIPPED,
    error="rapidocr is not installed")), TEMP))
check("a stage that could not run names the missing piece",
      "rapidocr is not installed" in t)
check("and says the question was still answered from the image",
      "from the image alone" in t, t[:300])

# 4. Skipped — and the two reasons for skipping read differently.
t = text_of(APP.ocr_column(record(ocr=OCRStageResult(
    lines=[], engine="", scope_used=OCR_SKIPPED,
    note="mode 3 runs no OCR - the model read the image itself")), TEMP))
check("mode 3's skip says the model read it itself",
      "the model read the image itself" in t)
check("and invites the comparison with modes 1 and 2",
      "how well the model reads the text unaided" in t, t[:300])

t = text_of(APP.ocr_column(record(ocr=OCRStageResult(
    lines=[], engine="", scope_used=OCR_SKIPPED,
    note="this question does not use OCR"), question_id="gps_antenna"), ANTENNA))
check("a non-OCR question's skip is quiet", "does not use OCR" in t)
check("and does not invite a comparison that does not apply",
      "unaided" not in t)

t = text_of(APP.ocr_column(record(ocr=None), TEMP))
check("a stage never reached says an earlier one stopped it",
      "an earlier stage stopped" in t.lower() or "earlier stage" in t, t[:200])

print("\na numeric question whose text carried no number")
t = text_of(APP.ocr_column(record(ocr=ocr_result(
    lines=[OCRLine(text="AC ROOM", text_confidence=0.9, box=[0, 0, 1, 1])],
    numeric=None)), TEMP))
check("says no reading was matched", "No temperature reading could be matched" in t
      or "reading could be matched" in t, t[:300])
check("and shows the text that WAS read", "AC ROOM" in t)

print("\nthe three-mode column reads across")
per_mode = {MODE_CLASSIC: record(ocr=ocr_result(), answer="yes"),
            MODE_ORDER[1]: record(ocr=ocr_result(), answer="yes"),
            MODE_VLM_ONLY: record(ocr=OCRStageResult(
                lines=[], engine="", scope_used=OCR_SKIPPED,
                note="mode 3 runs no OCR"), answer="no")}
t = text_of(APP.side_by_side("site", per_mode, TEMP))
check("a disagreement is called out, not left to be spotted",
      "The three modes disagree" in t, t[:300])
check("and names each mode's answer", "YES" in t and "NO" in t)
check("modes that ran OCR show what it read", "33.5 C" in t)
check("mode 3 states it had none", "the model read the image itself" in t
      or "No OCR" in t, t[:600])

agreed = {m: record(ocr=ocr_result(), answer="yes") for m in MODE_ORDER}
t = text_of(APP.side_by_side("site", agreed, TEMP))
check("unanimity is stated plainly", "All three modes agree" in t)
check("and is not dressed as a warning", "disagree" not in t)

print("\na mock answer is never allowed to look real")
mock = record(ocr=ocr_result())
mock.vlm = VLMAnswer(answer="unknown", reasoning="", raw_text="", model="qwen3-vl",
                     elapsed_s=0.0, is_mock=True)
t = text_of(APP.side_by_side("site", {m: mock for m in MODE_ORDER}, TEMP))
check("a mock says so on the three-mode view", "MOCK" in t)
check("and says no model looked", "no model looked" in t.lower())

print("\nthe stage detail card carries all four stages")
t = text_of(APP.detail_card(record(ocr=ocr_result()), TEMP, MODE_CLASSIC))
for stage in ("1 · Quality gate", "2b · Text read (OCR)", "3 · Model answer"):
    check(f"it shows {stage}", stage in t, t[:200])
check("and names the mode it is showing", "Quality gate" in t)

print("\nthe panel before anything has run")
APP._STATE.update({"results": {}, "question_id": None})
t = text_of(APP._panel("compare", MODE_CLASSIC))
check("invites the three steps in order",
      "domain" in t and "question" in t and "photograph" in t, t[:200])

print("\nthe transport help names the failure to expect")
check("proxy names the header", "X-API-Key" in APP.on_transport("proxy"))
check("direct names the 404 that reads as a dead server",
      "404" in APP.on_transport("direct"))

print("\nthe run button is the only thing that runs the pipeline")
# Switching tab or mode must re-read, never re-run: that is the entire reason
# all three modes are computed in one pass.
src = (ROOT / "frontend" / "demo_dash_pipeline.py").read_text()
check("on_run checks which control fired before running",
      'triggered != "run"' in src)
check("and the run body is a plain function the tests can drive",
      "def execute_run(" in src)
check("execute_run is not itself a callback",
      "@app.callback" not in src.split("def execute_run(")[0].rsplit("\n\n", 1)[-1])

print("\nthe resource registry is warmed before the server takes requests")
# Dash fills app.registered_paths only while rendering the index. A browser
# holding a cached page - any tab open across a restart - never re-requests the
# index, so its component chunks 500 on an empty registry and the UPLOAD
# DROPZONE silently fails to render. The symptom is a missing control, not an
# error on screen, which is why this is warmed at startup rather than noted in
# a document.
import ui_common  # noqa: E402

for _name in ("demo_dash_pipeline.py", "demo_dash_modes.py", "demo_dash_yolox.py"):
    _src = (ROOT / "frontend" / _name).read_text()
    check(f"{_name} warms the registry before app.run",
          "warm_resource_registry(app)" in _src
          and _src.index("warm_resource_registry(app)") < _src.index("app.run("),
          _name)
# demo_dash.py is frozen (FROZEN.md) and must not have grown the call.
check("the frozen UI was left alone",
      "warm_resource_registry" not in (ROOT / "frontend" / "demo_dash.py").read_text())


class _FakeApp:
    """An app whose index render populates the registry, like the real one."""

    def __init__(self, explode=False):
        self.registered_paths = {}
        self.explode = explode
        outer = self

        class _Server:
            def test_client(self):
                return outer

        self.server = _Server()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, path):
        if self.explode:
            raise RuntimeError("index render failed")
        self.registered_paths["dash"] = {"x.js"}


_app = _FakeApp()
check("warming reports success and fills the registry",
      ui_common.warm_resource_registry(_app) is True and "dash" in _app.registered_paths)
# A UI that cannot warm up must still start - the first real page load fills
# the registry exactly as it always did.
check("a failed warm-up returns False rather than raising",
      ui_common.warm_resource_registry(_FakeApp(explode=True)) is False)
# ui_common must not drag the pipeline into the UI-only layer. Parsed, not
# grepped: its docstring says the words "no pipeline imports", and a substring
# check fails on the very sentence promising the property.
import ast as _ast  # noqa: E402

_imports = set()
for _node in _ast.walk(_ast.parse((ROOT / "frontend" / "ui_common.py").read_text())):
    if isinstance(_node, _ast.Import):
        _imports.update(a.name.split(".")[0] for a in _node.names)
    elif isinstance(_node, _ast.ImportFrom):
        _imports.add((_node.module or "").split(".")[0])
check("ui_common imports nothing from the backend",
      not (_imports & {"pipeline", "quality_check", "foreground_segmentation"}),
      str(sorted(_imports)))

print("\nit imports the shared renderers rather than copying them")
check("quality_column and detection_column come from demo_dash_modes",
      "from demo_dash_modes import" in src
      and "quality_column" in src.split("from demo_dash_modes import")[1][:200]
      and "detection_column" in src.split("from demo_dash_modes import")[1][:200])
check("vlm_column comes from the frozen UI",
      "from demo_dash import" in src and "vlm_column" in src)
check("no class names are hard-coded in the UI",
      "GPS Antenna" not in src and "Warning sign" not in src)
check("no question ids are hard-coded into the layout",
      "hazard_warning" not in src)
check("it runs on its own port, leaving 7872 alone",
      '"7873"' in src and "7872" not in src.split("def main()")[1])

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
