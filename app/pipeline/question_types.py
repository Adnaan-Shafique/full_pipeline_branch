"""The Question dataclass, its OCR spec, and the pure helpers that render a
prompt from one.

Split out of questions.py so that pipeline/registry.py can build Question
objects from YAML without importing the module that is itself populated from
the registry. questions.py re-exports everything here, so
`from pipeline.questions import Question` keeps working and remains the import
every other module and test should use — this module is an implementation
detail of that split.

Pure stdlib. No yaml, no cv2, no torch: a Question must be constructible and
assertable on a machine with nothing installed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

# ─────────────────────────────── Shared contract ─────────────────────────────
# Appended to every question so the parser only ever sees one shape.
OUTPUT_CONTRACT = (
    'Respond with JSON only, no markdown fence:\n'
    '{"answer": "yes" | "no" | "unknown", '
    '"reasoning": "<2-3 sentences citing the specific visual evidence in the image>"}'
)

# Injected only when detections exist, otherwise the empty string. The
# "advisory only" wording is load-bearing: a confident-looking 0.91 presented as
# ground truth will anchor the model's answer, and in stub mode that number is a
# deterministic hash, not a measurement.
DETECTION_BLOCK_PREFIX = (
    "Object detector output (advisory only - it may be incomplete or wrong; "
    "trust the image over this list): "
)

# The same hedge, for the same reason, one stage later. PP-OCRv6 on a
# seven-segment display or a weathered module label is materially less reliable
# than it looks: it will return a confident 0.97 for "1.85" read off something
# that says "1,B5". The model is explicitly told it may overrule this, and the
# per-line confidence is included so it can see which lines are shaky.
OCR_BLOCK_PREFIX = (
    "Text read from the image by an OCR engine (advisory only - OCR misreads "
    "digital displays and worn labels; trust the image over this text): "
)

# Appended to the OCR block when a question carries a numeric rule and a number
# was actually found. Phrased as a computed check rather than a verdict,
# because it is evidence for the model, not a substitute for its answer.
OCR_NUMERIC_PREFIX = " Threshold check computed from that text: "

ANSWER_YES = "yes"
ANSWER_NO = "no"
ANSWER_UNKNOWN = "unknown"
VALID_ANSWERS = (ANSWER_YES, ANSWER_NO, ANSWER_UNKNOWN)

# OCRSpec.scope values.
OCR_SCOPE_BOXES = "boxes"
OCR_SCOPE_IMAGE = "image"
OCR_SCOPE_BOXES_THEN_IMAGE = "boxes_then_image"
OCR_SCOPES = (OCR_SCOPE_BOXES, OCR_SCOPE_IMAGE, OCR_SCOPE_BOXES_THEN_IMAGE)

COMPARATORS = {
    "<":  lambda v, limit: v < limit,
    "<=": lambda v, limit: v <= limit,
    ">":  lambda v, limit: v > limit,
    ">=": lambda v, limit: v >= limit,
}


@dataclass(frozen=True)
class NumericRule:
    """Turn a reading OCR'd off a display into a checked threshold.

    This produces EVIDENCE, never the answer. The rule cannot see the range
    multiplier on a rotary switch, cannot tell a set-point from a measurement,
    and cannot tell °F from °C — all three of which are exactly the mistakes
    that make a confident wrong number worse than no number. So its output is
    handed to the model as one more line of advisory input, and the model still
    answers. See DECISIONS.md.
    """

    label: str                 # "Temperature", "Earth resistance"
    unit: str                  # "°C", "Ω" — display only, never parsed
    comparator: str            # one of COMPARATORS
    limit: float
    pattern: str               # regex whose FIRST group is the number

    def __post_init__(self) -> None:
        if self.comparator not in COMPARATORS:
            raise ValueError(
                f"NumericRule comparator {self.comparator!r} is not one of "
                f"{sorted(COMPARATORS)}")
        try:
            compiled = re.compile(self.pattern)
        except re.error as exc:
            raise ValueError(f"NumericRule pattern does not compile: {exc}") from None
        if compiled.groups < 1:
            raise ValueError(
                "NumericRule pattern must have a capturing group around the "
                f"number; {self.pattern!r} has none")

    def evaluate(self, text: str) -> Optional[dict]:
        """First number in `text` that the pattern matches, checked against the
        limit. None when nothing matched — which is the common case on a photo
        where the display is unlit or illegible, and must read as "no reading",
        never as a failed check.

        Returns {"value", "passes", "sentence", "matched"}. `sentence` is what
        goes into the prompt and onto the card, worded as a measurement against
        a limit rather than as a yes/no.
        """
        match = re.search(self.pattern, text or "")
        if not match:
            return None
        raw = match.group(1)
        try:
            # Field displays and OCR both produce "1,85" for "1.85".
            value = float(raw.replace(",", "."))
        except ValueError:
            return None
        passes = COMPARATORS[self.comparator](value, self.limit)
        return {
            "value": value,
            "passes": passes,
            "matched": raw,
            "sentence": (f"{self.label} read as {value:g} {self.unit}, which is "
                         f"{'within' if passes else 'outside'} the limit of "
                         f"{self.comparator} {self.limit:g} {self.unit}."),
        }


@dataclass(frozen=True)
class OCRSpec:
    """Whether, and how, the OCR stage runs for one question.

    Disabled by default: OCR costs real time per photograph and returns noise on
    an image with no text in it, and noise injected into the prompt as evidence
    is worse than an absent block. A question opts in by setting
    `ocr.enabled: true` in its YAML.
    """

    enabled: bool = False
    scope: str = OCR_SCOPE_BOXES_THEN_IMAGE
    numeric: Optional[NumericRule] = None
    min_confidence: float = 0.30   # drop lines below this before building the block

    def __post_init__(self) -> None:
        if self.scope not in OCR_SCOPES:
            raise ValueError(f"OCRSpec scope {self.scope!r} is not one of {list(OCR_SCOPES)}")


@dataclass(frozen=True)
class Question:
    id: str
    label: str                        # shown in the question dropdown
    system_prompt: str                # PER QUESTION - sent as payload["system"]
    user_template: str                # PER QUESTION - .format(**ctx)
    answer_semantics: str             # human-readable yes/no meaning, shown in the UI
    relevant_classes: list[str]       # detector labels to surface / crop to
    # Class names for THIS question's annotation export, indexed by class_id.
    #
    # Each CVAT export is a separate single-class task, so every one of them
    # numbers its only class 0. "0" therefore means a hazard sign in the HV
    # export and a GPS antenna in the antenna export - the same id, two
    # meanings. A single shared classes.txt cannot express that, and guessing
    # wrong injects a false label into {detection_block}, which the prompt
    # explicitly tells the model to weigh as evidence.
    #
    # A real classes.txt or dataset.yaml in the label folder always wins; this
    # is the fallback that makes a bare single-class export readable.
    default_class_names: list[str] = field(default_factory=list)
    sampling: dict = field(default_factory=dict)   # optional per-question overrides
    # ── Added with the domain/OCR expansion ──────────────────────────────────
    domain: str = "site_safety"       # groups questions in the first dropdown
    # Mode 3's presence leg asks "is {subject} visible in this photograph?" and
    # its answer leg opens with {question_text}. Both used to live in the
    # SUBJECTS and QUESTION_TEXT dicts keyed by question id, which meant adding
    # a question required remembering to add it to two more places or silently
    # getting the generic fallback wording in mode 3.
    subject: str = ""
    question_text: str = ""
    ocr: OCRSpec = field(default_factory=OCRSpec)
    # Class ids from config/classes.yaml, kept for the UI's "what does this
    # question look for" panel. relevant_classes above is the resolved,
    # match-ready list of names and aliases.
    class_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Trap 9: _build_payload only sets payload["system"] when the string is
        # non-empty, and both of the server's prompt builders guard on
        # `if system:`. A blank or mis-keyed system prompt therefore fails
        # SILENTLY - the model answers with no framing at all, which looks like a
        # model-quality problem rather than the wiring bug it is. Fail loudly
        # here instead, at import time.
        if not (self.system_prompt or "").strip():
            raise ValueError(f"Question {self.id!r} has an empty system_prompt")
        if "{output_contract}" not in self.user_template:
            raise ValueError(f"Question {self.id!r} template is missing {{output_contract}}")
        if "{detection_block}" not in self.user_template:
            raise ValueError(f"Question {self.id!r} template is missing {{detection_block}}")
        # The same failure one stage later: OCR would run, cost its seconds, and
        # its text would never reach the model, which reads as "the OCR stage
        # found nothing" rather than "the template forgot to include it".
        if self.ocr.enabled and "{ocr_block}" not in self.user_template:
            raise ValueError(
                f"Question {self.id!r} has ocr.enabled but its template is "
                f"missing {{ocr_block}} - the OCR text would be read and then "
                f"silently discarded")

    @property
    def effective_subject(self) -> str:
        return self.subject or "the subject of the inspection"

    @property
    def effective_question_text(self) -> str:
        return self.question_text or self.label


# ─────────────────────────────── Template builder ────────────────────────────

def build_user_template(question_text: str, yes_means: str, no_means: str,
                        with_ocr: bool = False) -> str:
    """The user prompt for a question that does not author its own.

    Keeping this in one place is what makes "add a question" a five-line YAML
    entry rather than a template someone has to get the three placeholders right
    in. The two Site Safety questions still carry explicit templates, because
    their exact wording is what every recorded result for this demo was produced
    with.
    """
    text = (question_text or "").strip()
    if text:
        text = text[0].upper() + text[1:]
    if not text.endswith("?"):
        text += "?"
    lines = [text,
             f"yes = {(yes_means or '').strip()}",
             f"no  = {(no_means or '').strip()}",
             "{detection_block}"]
    if with_ocr:
        lines.append("{ocr_block}")
    lines.append("{output_contract}")
    return "\n".join(lines)


# ─────────────────────────────── Matching ────────────────────────────────────

def normalise(label: str) -> str:
    """Fold a class name to comparable words.

    Real labels are written for humans, not for matching: the trained detector's
    own classes.json spells its classes in title case with punctuation, while
    annotation folders use lower_snake_case for the same object. Exact string
    equality would miss every one of those, so punctuation, case and separators
    are flattened. The names themselves are in config/classes.yaml; this module
    deliberately does not repeat them.
    """
    return re.sub(r"[^a-z0-9]+", " ", (label or "").lower()).strip()


# Kept under the old private name too: stage2_detect and the tests import it.
_normalise = normalise


def select_relevant(detections: Iterable, question: Question) -> list:
    """Detections worth surfacing for this question.

    Matches on normalised substrings in either direction, so a detector label
    in title case matches a question whose class list carries the lower-case id,
    and a long parenthesised label matches on any of its aliases. The names and
    aliases come from config/classes.yaml via the registry.

    Still deliberately TOLERANT: if nothing matches, return ALL detections
    rather than none. A label this function does not recognise silently
    emptying both the detection block and the crop would look exactly like
    "the detector found nothing" - the wrong story to tell on stage.
    """
    dets = list(detections)
    if not dets:
        return []
    wanted = [normalise(c) for c in question.relevant_classes if c and c.strip()]
    wanted = [w for w in wanted if w]
    matched = []
    for d in dets:
        label = normalise(str(getattr(d, "label", "")))
        if label and any(w in label or label in w for w in wanted):
            matched.append(d)
    return matched or dets


# ─────────────────────────────── Prompt blocks ───────────────────────────────

def build_detection_block(detections: Iterable) -> str:
    """The {detection_block} substitution. Empty string when there is nothing to
    report, so the template collapses cleanly rather than saying "detector found:"
    followed by nothing."""
    dets = list(detections)
    if not dets:
        return ""
    parts = []
    for d in dets:
        box = [int(round(v)) for v in d.box]
        parts.append(f"{d.label} (confidence {d.confidence:.2f}) at {box}")
    return DETECTION_BLOCK_PREFIX + "; ".join(parts) + "."


def build_ocr_block(ocr_result, question: Optional[Question] = None) -> str:
    """The {ocr_block} substitution, from an OCRStageResult.

    Empty string when OCR did not run, found nothing, or errored — in every one
    of those cases the template collapses and the model is simply not told about
    an OCR stage, which is the honest rendering. An error in particular must not
    become a line of prompt: "OCR failed" is information for the operator on the
    card, not a piece of visual evidence for the model.
    """
    if ocr_result is None or getattr(ocr_result, "error", None):
        return ""
    lines = [ln for ln in getattr(ocr_result, "lines", []) if (ln.text or "").strip()]
    if not lines:
        return ""
    parts = [f'"{ln.text.strip()}" (confidence {ln.text_confidence:.2f})' for ln in lines]
    block = OCR_BLOCK_PREFIX + "; ".join(parts) + "."
    numeric = getattr(ocr_result, "numeric", None)
    if numeric and numeric.get("sentence"):
        block += OCR_NUMERIC_PREFIX + numeric["sentence"]
    return block


def render_user_prompt(question: Question, detections: Optional[Iterable] = None,
                       ocr_result=None) -> str:
    """Format a question's user_template with its detection block, its OCR block
    and the shared output contract. Collapses the blank lines left behind when a
    block is empty, so the prompt never carries a stray empty line."""
    text = question.user_template.format(
        detection_block=build_detection_block(detections or []),
        ocr_block=build_ocr_block(ocr_result, question),
        output_contract=OUTPUT_CONTRACT,
    )
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()
