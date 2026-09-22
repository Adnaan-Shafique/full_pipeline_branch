"""The plugin layer: config/domains.yaml, classes.yaml and questions/*.yaml.

Runs with nothing installed. The sections that need pyyaml announce themselves
and are skipped without it - the fallback behaviour they would otherwise test is
covered by the degradation section, which runs either way.
"""
import sys
import textwrap
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
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


from pipeline.question_types import (OCRSpec, Question,             # noqa: E402
                                     build_user_template)
from pipeline.registry import (BUILTIN_QUESTIONS, Domain, DetectorClass,  # noqa: E402
                               builtin_registry, load_registry)

try:
    import yaml  # noqa: F401
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False


def write_tree(root: Path, domains=None, classes=None, questions=None):
    """A minimal config/ tree. Any part omitted is simply not written, so the
    'file missing' paths are reachable."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "questions").mkdir(exist_ok=True)
    if domains is not None:
        (root / "domains.yaml").write_text(textwrap.dedent(domains))
    if classes is not None:
        (root / "classes.yaml").write_text(textwrap.dedent(classes))
    if questions is not None:
        (root / "questions" / "test.yaml").write_text(textwrap.dedent(questions))
    return root


MINIMAL_CLASSES = """
    classes:
      - {id: widget, name: Widget, aliases: [widgets, thing], trained: true, yolox_index: 0}
      - {id: gauge, name: Gauge, trained: false}
"""
MINIMAL_DOMAINS = """
    domains:
      - {id: alpha, label: Alpha, order: 2}
      - {id: beta, label: Beta, order: 1}
"""
MINIMAL_QUESTIONS = """
    questions:
      - id: widget_present
        domain: alpha
        label: Is the widget present?
        subject: a widget
        question_text: is a widget visible?
        yes_means: a widget is visible.
        no_means: no widget is visible.
        classes: [widget]
        system_prompt: You are a widget inspector. Respond with JSON only.
      - id: gauge_reading
        domain: beta
        label: Is the gauge under 10?
        subject: a gauge
        question_text: does the gauge read under 10?
        yes_means: the gauge reads under 10.
        no_means: the gauge reads 10 or more.
        classes: [gauge]
        ocr:
          enabled: true
          scope: boxes_then_image
          numeric: {label: Reading, unit: bar, comparator: "<", limit: 10.0,
                    pattern: '(\\d+(?:\\.\\d+)?)'}
        system_prompt: You are a gauge reader. Respond with JSON only.
"""

# Everything from here to the degradation section asserts on the repository's
# OWN questions, which only exist when config/ can be read. Without pyyaml the
# registry serves two built-in questions and `real.questions["temp_within_limit"]`
# is a KeyError - a crash, not a reported failure, which breaks this repo's
# "every suite prints N passed, M failed" convention.
if not HAVE_YAML:
    print("\n(pyyaml absent - every section that asserts on config/ is skipped;\n"
          " the degradation and dataclass sections below still run)")

if HAVE_YAML:
  print("\nthe repository's own config/ tree loads cleanly")
  real = load_registry(ROOT / "config")
  check("config/ is the source, not the built-ins",
        "built-in" not in real.source, real.source)
  check("it loads with no warnings at all", not real.warnings,
        "; ".join(real.warnings))
  check("both domains are present and populated",
        [d.id for d in real.ordered_domains()] == ["site_safety", "infra"],
        str([d.id for d in real.ordered_domains()]))
  check("the two Site Safety questions survived the move to YAML",
        {q.id for q in real.questions_in("site_safety")} == {"hazard_warning", "gps_antenna"})
  check("every question resolves at least one detector class",
        all(q.class_ids for q in real.questions.values()),
        str([q.id for q in real.questions.values() if not q.class_ids]))
  check("every class a question names exists in classes.yaml",
        all(cid in real.classes for q in real.questions.values() for cid in q.class_ids))
  check("exactly four questions carry an OCR stage",
        sorted(q.id for q in real.questions.values() if q.ocr.enabled) ==
        ["earthing_value_egb", "spd_class_b_installed", "spd_class_c_installed",
         "temp_within_limit"],
        str(sorted(q.id for q in real.questions.values() if q.ocr.enabled)))
  check("both device-reading questions carry a numeric rule",
        all(real.questions[q].ocr.numeric is not None
            for q in ("temp_within_limit", "earthing_value_egb")))
  check("no question that carries OCR is missing {ocr_block}",
        all("{ocr_block}" in q.user_template
            for q in real.questions.values() if q.ocr.enabled))
  check("no question WITHOUT OCR carries a stray {ocr_block}",
        all("{ocr_block}" not in q.user_template
            for q in real.questions.values() if not q.ocr.enabled))

  print("\nthe numeric rules resist the readings that are not measurements")
  temp = real.questions["temp_within_limit"].ocr.numeric
  ohm = real.questions["earthing_value_egb"].ocr.numeric
  # The failure that motivated the word-boundary guard: an enclosure rating in
  # the same frame as the display, read as a temperature.
  check("IP55 is not read as 55 degrees",
        (temp.evaluate("IP55 CABINET 33.5 C") or {}).get("value") == 33.5,
        str(temp.evaluate("IP55 CABINET 33.5 C")))
  check("IP65 alone yields no reading at all",
        temp.evaluate("IP65 ONLY") is None, str(temp.evaluate("IP65 ONLY")))
  check("a comma decimal is read as a decimal",
        (temp.evaluate("34,2 C") or {}).get("value") == 34.2)
  check("at the limit exactly, <= passes",
        (temp.evaluate("35 C") or {}).get("passes") is True)
  check("one tenth over the limit fails",
        (temp.evaluate("35.1 C") or {}).get("passes") is False)
  check("no digits means no reading, never a failed check",
        temp.evaluate("display is off") is None)
  check("ohms below the limit pass", (ohm.evaluate("1.85 ohm") or {}).get("passes") is True)
  check("a decimal point lost to glare fails, as it should",
        (ohm.evaluate("185") or {}).get("passes") is False)
  check("2.0 exactly fails the strict < comparator",
        (ohm.evaluate("2.0") or {}).get("passes") is False)
  # The rule is evidence, not the answer, precisely because of these two. Both
  # are pinned so the limitation stays visible rather than being discovered live.
  check("a set-point ahead of the measurement is what gets matched (known limit)",
        (temp.evaluate("SET 22 C ACT 38.0 C") or {}).get("value") == 22.0)
  check("a range-switch label is matched as if it were a reading (known limit)",
        (ohm.evaluate("RANGE 20") or {}).get("value") == 20.0)

  print("\nthe mode-3 legs derive from the question, not from a second dict")
  import pipeline.questions as pq  # noqa: E402
  check("SUBJECTS covers every registered question",
        set(pq.SUBJECTS) == set(pq.QUESTIONS))
  check("QUESTION_TEXT covers every registered question",
        set(pq.QUESTION_TEXT) == set(pq.QUESTIONS))
  check("no question falls through to the generic subject wording",
        all("the subject of the inspection" not in s for s in pq.SUBJECTS.values()),
        str([k for k, v in pq.SUBJECTS.items() if "the subject" in v]))
  for qid in ("temp_within_limit", "hazard_warning"):
      leg = pq.render_leg_user(pq.get_question(qid), "presence")
      check(f"{qid}: the presence leg names its own subject",
            pq.get_question(qid).effective_subject in leg)

  if HAVE_YAML:
      import tempfile

      print("\na hand-written tree: ordering, derivation and the OCR spec")
      with tempfile.TemporaryDirectory() as tmp:
          reg = load_registry(write_tree(Path(tmp) / "config", MINIMAL_DOMAINS,
                                         MINIMAL_CLASSES, MINIMAL_QUESTIONS))
          check("no warnings on a well-formed tree", not reg.warnings, str(reg.warnings))
          check("domains sort by `order`, not by file position",
                [d.id for d in reg.ordered_domains()] == ["beta", "alpha"])
          check("yolox_class_names comes from the trained entries only",
                reg.yolox_class_names() == ("Widget",), str(reg.yolox_class_names()))
          check("an untrained class is still selectable by a question",
                "gauge" in reg.questions["gauge_reading"].class_ids)
          check("untrained_classes_for names what has no weights",
                [c.id for c in reg.untrained_classes_for("gauge_reading")] == ["gauge"])
          # The id is a match term too - annotation folders label boxes with it
          # far more often than with the display name - but it is deduped against
          # the name case-insensitively, because select_relevant() normalises both
          # sides anyway and a duplicate term only costs a comparison.
          check("name, id and aliases all become match terms, deduped by case",
                reg.classes["widget"].match_terms == ["Widget", "widgets", "thing"],
                str(reg.classes["widget"].match_terms))
          check("an id that differs from the name is kept as its own term",
                "spd_class_b" in real.classes["spd_class_b"].match_terms,
                str(real.classes["spd_class_b"].match_terms))
          q = reg.questions["gauge_reading"]
          check("answer_semantics is derived from yes_means/no_means",
                q.answer_semantics == "YES = the gauge reads under 10.  NO = the gauge reads 10 or more.",
                q.answer_semantics)
          check("the generated template carries all three placeholders",
                all(p in q.user_template
                    for p in ("{detection_block}", "{ocr_block}", "{output_contract}")))
          check("a question without OCR gets no {ocr_block}",
                "{ocr_block}" not in reg.questions["widget_present"].user_template)
          check("the numeric rule round-trips through YAML",
                (q.ocr.numeric.evaluate("7.5 bar") or {}).get("passes") is True)

      print("\na broken tree costs the broken entry, never the whole registry")
      with tempfile.TemporaryDirectory() as tmp:
          reg = load_registry(write_tree(Path(tmp) / "config", MINIMAL_DOMAINS,
              MINIMAL_CLASSES, """
              questions:
                - id: good
                  domain: alpha
                  label: A good question?
                  yes_means: yes it is.
                  no_means: no it is not.
                  classes: [widget]
                  system_prompt: A real prompt. Respond with JSON only.
                - id: blank_system
                  domain: alpha
                  label: A question with no system prompt?
                  yes_means: yes.
                  no_means: no.
                  system_prompt: "   "
                - id: no_semantics
                  domain: alpha
                  label: A question with nothing to build a template from?
                  system_prompt: A real prompt. Respond with JSON only.
                - id: unknown_domain
                  domain: nowhere
                  label: A question filed under a domain that does not exist?
                  yes_means: yes.
                  no_means: no.
                  classes: [widget, not_a_class]
                  system_prompt: A real prompt. Respond with JSON only.
              """))
          check("the good question survives its broken neighbours",
                "good" in reg.questions)
          check("a blank system prompt is dropped, not shipped",
                "blank_system" not in reg.questions, str(sorted(reg.questions)))
          check("a question with no way to build a template is dropped",
                "no_semantics" not in reg.questions)
          check("an unknown domain is re-filed rather than dropped",
                "unknown_domain" in reg.questions)
          check("an unknown class is dropped but its question is not",
                reg.questions["unknown_domain"].class_ids == ["widget"],
                str(reg.questions["unknown_domain"].class_ids))
          check("every one of those problems is reported",
                len(reg.warnings) >= 4, str(reg.warnings))

      print("\neach config file stands or falls on its own")
      with tempfile.TemporaryDirectory() as tmp:
          # A questions folder that yields nothing must NOT reset the classes:
          # a typo in one question file silently reverting the detector's class
          # list to the built-in two is the failure this pins.
          reg = load_registry(write_tree(Path(tmp) / "config", MINIMAL_DOMAINS,
                                         MINIMAL_CLASSES, "questions: []"))
          check("classes.yaml survives a questions folder that yields nothing",
                reg.yolox_class_names() == ("Widget",), str(reg.yolox_class_names()))
          check("the built-in questions fill in",
                set(reg.questions) == set(BUILTIN_QUESTIONS))
          check("their domain is added so they stay reachable",
                all(q.domain in reg.domains for q in reg.questions.values()))
          check("the substitution is reported", any("built-in" in w for w in reg.warnings))

      with tempfile.TemporaryDirectory() as tmp:
          root = Path(tmp) / "config"
          write_tree(root, MINIMAL_DOMAINS, MINIMAL_CLASSES, MINIMAL_QUESTIONS)
          (root / "classes.yaml").write_text("classes: [ this is not: valid: yaml")
          reg = load_registry(root)
          check("an unparseable classes.yaml falls back without raising",
                reg.yolox_class_names() != (), str(reg.yolox_class_names()))
          check("the parse failure is named", any("classes.yaml" in w for w in reg.warnings),
                str(reg.warnings))
  else:
      print("\n(pyyaml absent - the hand-written-tree sections are skipped)")

print("\ndegradation: no config tree at all")
missing = load_registry(ROOT / "does_not_exist")
check("a missing config dir yields the built-ins", "built-in" in missing.source)
check("the built-ins are still a usable demo", len(missing.questions) == 2)
check("and it says why", bool(missing.warnings))
for qid, q in builtin_registry("x").questions.items():
    check(f"built-in {qid} satisfies trap 9", bool(q.system_prompt.strip()))
    check(f"built-in {qid} has a subject for mode 3", bool(q.subject))

print("\nQuestion rejects what fails silently at runtime")
for name, kwargs, want in [
    ("a blank system prompt",
     dict(system_prompt="  ", user_template="{detection_block}{output_contract}"), True),
    ("a template with no {output_contract}",
     dict(system_prompt="S", user_template="{detection_block}"), True),
    ("a template with no {detection_block}",
     dict(system_prompt="S", user_template="{output_contract}"), True),
    ("ocr enabled with no {ocr_block}",
     dict(system_prompt="S", user_template="{detection_block}{output_contract}",
          ocr=OCRSpec(enabled=True)), True),
    ("ocr enabled WITH {ocr_block}",
     dict(system_prompt="S",
          user_template="{detection_block}{ocr_block}{output_contract}",
          ocr=OCRSpec(enabled=True)), False),
]:
    try:
        Question(id="t", label="t", answer_semantics="", relevant_classes=[], **kwargs)
        raised = False
    except ValueError:
        raised = True
    check(f"{name}: {'rejected' if want else 'accepted'}", raised is want)

try:
    OCRSpec(enabled=True, scope="somewhere_else")
    raised = False
except ValueError:
    raised = True
check("an unknown OCR scope is rejected at construction", raised)

print("\nbuild_user_template")
t = build_user_template("is it there", "it is visible.", "it is not.", with_ocr=True)
check("the question text is capitalised and ends in a question mark",
      t.startswith("Is it there?"), t.splitlines()[0])
check("with_ocr=False omits the block",
      "{ocr_block}" not in build_user_template("q", "y", "n", with_ocr=False))
check("the blocks are ordered detection, then ocr, then contract",
      t.index("{detection_block}") < t.index("{ocr_block}") < t.index("{output_contract}"))

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
