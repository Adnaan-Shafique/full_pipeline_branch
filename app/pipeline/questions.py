"""The demo's question registry - each question owns its own prompt pair.

There is deliberately no single global system prompt. "Is a warning sign
present" is a presence/legibility judgement; "is this antenna open to the sky"
is a spatial-occlusion judgement about the scene ABOVE the object. One generic
persona does both worse than two specific ones, so swapping the dropdown swaps
the entire prompt pair with it.

── Where the questions now live ─────────────────────────────────────────────
They used to be Question(...) literals in this file. They are now YAML, under
config/questions/, loaded by pipeline/registry.py:

    config/domains.yaml          the domain dropdown
    config/classes.yaml          every detector class, trained or not
    config/questions/*.yaml      the questions

Adding a question is a YAML entry; no Python changes. Removing one is deleting
that entry. Adding or removing a detector class is an entry in classes.yaml,
from which PipelineConfig.yolox_class_names is derived. This module is now the
stable import surface over that registry, so every existing import of it -
`from pipeline.questions import get_question, select_relevant, QUESTIONS` -
still resolves exactly as before.

If pyyaml is missing or config/ has been deleted, the registry falls back to
two built-in Site Safety questions and says why in `REGISTRY.warnings`, which
the UI prints. The demo degrades; it does not fail to import.

The dataclass and the pure prompt helpers live in question_types.py, so the
registry can build Questions without importing this module back. They are
re-exported here; import them from here.

Pure stdlib apart from the registry's optional `import yaml`.
"""
from __future__ import annotations

from .question_types import (  # noqa: F401  - re-exported, this is the public surface
    ANSWER_NO, ANSWER_UNKNOWN, ANSWER_YES, COMPARATORS, DETECTION_BLOCK_PREFIX,
    OCR_BLOCK_PREFIX, OCR_NUMERIC_PREFIX, OCR_SCOPES, OCR_SCOPE_BOXES,
    OCR_SCOPE_BOXES_THEN_IMAGE, OCR_SCOPE_IMAGE, OUTPUT_CONTRACT, NumericRule,
    OCRSpec, Question, VALID_ANSWERS, _normalise, build_detection_block,
    build_ocr_block, build_user_template, normalise, render_user_prompt,
    select_relevant)
from .registry import get_registry, load_registry  # noqa: F401

# ─────────────────────────────── The registry ────────────────────────────────
# Module-level names kept for compatibility: QUESTIONS and QUESTION_CHOICES were
# the public surface before the YAML move and are imported directly in several
# places. They are snapshots taken at import; use refresh() after an edit, which
# is what the UI's Reload button calls.

REGISTRY = get_registry()
QUESTIONS: dict = dict(REGISTRY.questions)
QUESTION_CHOICES = [(q.label, q.id) for q in QUESTIONS.values()]


def refresh() -> None:
    """Re-read config/ and rebind the module-level snapshots.

    The snapshots are rebound rather than mutated in place so that a caller
    holding the old dict keeps a consistent view instead of watching questions
    appear and vanish mid-render.
    """
    global REGISTRY, QUESTIONS, QUESTION_CHOICES, SUBJECTS, QUESTION_TEXT
    REGISTRY = get_registry(reload=True)
    QUESTIONS = dict(REGISTRY.questions)
    QUESTION_CHOICES = [(q.label, q.id) for q in QUESTIONS.values()]
    SUBJECTS = {q.id: q.effective_subject for q in QUESTIONS.values()}
    QUESTION_TEXT = {q.id: q.effective_question_text for q in QUESTIONS.values()}


def get_question(question_id: str) -> Question:
    try:
        return QUESTIONS[question_id]
    except KeyError:
        raise KeyError(
            f"Unknown question id {question_id!r}. Known: {sorted(QUESTIONS)}"
        ) from None


def sampling_for(question: Question, cfg) -> dict:
    """Per-question sampling overrides layered over the config defaults."""
    base = {
        "max_new_tokens": cfg.max_new_tokens,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "top_k": cfg.top_k,
        "repetition_penalty": cfg.repetition_penalty,
    }
    base.update(question.sampling or {})
    return base


# ─────────────────── Mode 3: three legs, three system prompts ────────────────
# Mode 3 replaces the classical quality gate, the trained detector AND the OCR
# stage with the model. Each of the three judgements is its own call with its
# own system prompt, so each can be tuned - and blamed - independently. The
# alternative, one call returning all three, keeps them mutually consistent but
# makes a wording change to one criterion move the other two answers with it.
#
# Mode 3 deliberately does NOT get the OCR block. On the four questions that
# carry an OCR stage, the model reads the text itself, and the gap between what
# PP-OCRv6 read in modes 1-2 and what the model read here is the thing worth
# looking at - feeding it the OCR output would erase exactly that comparison.
# See MODES.md and DECISIONS.md.
#
# These are DEFAULTS. pipeline/prompts.py loads any saved overrides from
# config/prompts.yaml and the UI edits them live.

LEG_QUALITY = "quality"
LEG_PRESENCE = "presence"
LEG_ANSWER = "answer"
LEGS = (LEG_QUALITY, LEG_PRESENCE, LEG_ANSWER)

LEG_LABELS = {
    LEG_QUALITY: "1 · Image quality",
    LEG_PRESENCE: "2 · Subject present",
    LEG_ANSWER: "3 · Inspection question",
}

# What each question is looking for, in words the model can match against the
# image, and the question as mode 3's answer leg asks it.
#
# These were hand-maintained dicts keyed by question id, which meant adding a
# question required remembering to add it to two more places - and forgetting
# silently downgraded mode 3 to the generic fallback wording, with no error.
# They are now derived from the questions' own `subject` and `question_text`
# fields, and kept as dicts only because both names are imported elsewhere.
SUBJECTS = {q.id: q.effective_subject for q in QUESTIONS.values()}
QUESTION_TEXT = {q.id: q.effective_question_text for q in QUESTIONS.values()}

QUALITY_SYSTEM_DEFAULT = (
    "You are reviewing photographs taken during telecom site inspections and "
    "judging whether each one is of usable quality for an inspection decision. "
    "Consider sharpness, exposure, haze, and whether the subject is large "
    "enough and fully enough in frame to be judged. Do not judge what the "
    "photograph shows - only whether it can be judged from. A photograph can be "
    "imperfect and still perfectly usable; reserve \"poor\" for photographs that "
    "would genuinely prevent a decision. Respond with JSON only."
)

PRESENCE_SYSTEM_DEFAULT = (
    "You are reviewing photographs from telecom site inspections and "
    "determining whether a specific piece of equipment or signage is visible in "
    "the frame. Judge only presence and identifiability - not condition, not "
    "compliance, not whether it is correctly installed. If something is "
    "partially visible but cannot be confidently identified, answer "
    "\"unknown\" rather than guessing. Respond with JSON only."
)

QUALITY_USER_TEMPLATE = """Is this photograph of usable quality for an inspection decision?
good = sharp enough, exposed well enough and framed well enough to judge from.
poor = blurred, too dark or too bright, hazy, or the subject is too small or cut off.

Respond with JSON only, no markdown fence:
{"quality": "good" | "poor", "reasoning": "<1-2 sentences on sharpness, exposure and framing>"}"""

PRESENCE_USER_TEMPLATE = """Is {subject} visible in this photograph?
yes     = it is visible and identifiable.
no      = it is not in this photograph.
unknown = something may be there but it cannot be confidently identified.

Respond with JSON only, no markdown fence:
{{"present": "yes" | "no" | "unknown", "reasoning": "<1-2 sentences: is it visible, and where>"}}"""

ANSWER_USER_TEMPLATE = """{question_text}
{semantics}

Respond with JSON only, no markdown fence:
{{"answer": "yes" | "no" | "unknown", "reasoning": "<2-3 sentences citing the specific visual evidence in the image>"}}"""

# ── Grounded variants: the same legs, asked to point ─────────────────────────
# Used when cfg.vlm_grounding is on. Three things about the wording are
# load-bearing and should not be "tidied":
#
#   · "integers from 0 to 1000, normalized" is the convention the Qwen-VL family
#     was trained to emit. It is also the only one immune to array_to_data_uri()
#     downscaling the photograph to 2048px before sending it - an absolute pixel
#     reply is in THAT frame, and drawing it on a 4000px original is out by ~2x.
#     vlm_grounding.interpret_box() handles the other conventions anyway, but
#     this is the one we want back.
#   · "tightly, around the object itself - not the whole image" exists because
#     the cheapest way to comply with a box request is to return the frame.
#     That is the model declining to localise while appearing to obey, and
#     interpret_box() rejects it.
#   · "omit it entirely" gives a way to decline that is not a wrong box. A model
#     with no way to say "I can't" will invent coordinates.
GROUNDING_RULE = (
    'Coordinates are integers from 0 to 1000, normalized to the image width and '
    'height - [0, 0] is the top-left corner and [1000, 1000] the bottom-right - '
    'written as [x1, y1, x2, y2]. Box it tightly, around the object itself, not '
    'the whole image. If you cannot locate it confidently, omit the box entirely '
    'rather than guessing.'
)

PRESENCE_USER_TEMPLATE_GROUNDED = """Is {subject} visible in this photograph?
yes     = it is visible and identifiable.
no      = it is not in this photograph.
unknown = something may be there but it cannot be confidently identified.

If it is visible, give one bounding box around it. {grounding_rule}

Respond with JSON only, no markdown fence:
{{"present": "yes" | "no" | "unknown", "box": [x1, y1, x2, y2], "reasoning": "<1-2 sentences: is it visible, and where>"}}"""

ANSWER_USER_TEMPLATE_GROUNDED = """{question_text}
{semantics}

Also point to the region of the image you based your answer on - the thing you
actually looked at to decide. {grounding_rule}

Respond with JSON only, no markdown fence:
{{"answer": "yes" | "no" | "unknown", "evidence_box": [x1, y1, x2, y2], "reasoning": "<2-3 sentences citing the specific visual evidence in the image>"}}"""

# The quality leg is deliberately NOT grounded. Sharpness, exposure and framing
# are properties of the whole frame; a box drawn around "the blurry part" would
# be fabricated precision, and unlike the other two legs there is nothing to
# score it against. See DECISIONS.md.
GROUNDED_LEGS = (LEG_PRESENCE, LEG_ANSWER)


def default_leg_system(question: Question, leg: str) -> str:
    """The built-in system prompt for one leg of mode 3.

    The answer leg reuses the question's OWN system prompt, so there is still
    exactly one place where each question's framing is authored - editing it in
    the UI overrides that copy for mode 3 only, leaving modes 1 and 2 on the
    original.

    The exception is a question with an OCR stage. Its system prompt tells the
    model that OCR text "may be supplied to you as advisory evidence", and in
    mode 3 it never is - so those four questions carry a `system_prompt_no_ocr`
    in their YAML that says the opposite, and `answer_leg_system` picks it. A
    prompt promising evidence that does not arrive is the worst framing
    available for the one mode meant to show what the model reads unaided.
    """
    if leg == LEG_QUALITY:
        return QUALITY_SYSTEM_DEFAULT
    if leg == LEG_PRESENCE:
        return PRESENCE_SYSTEM_DEFAULT
    if leg == LEG_ANSWER:
        return question.answer_leg_system
    raise KeyError(f"Unknown leg {leg!r}; expected one of {LEGS}")


def render_leg_user(question: Question, leg: str, grounding: bool = False) -> str:
    """The user prompt for one leg. Not editable: it carries the JSON contract
    the parser depends on, and a demo is not the place to discover that someone
    removed it.

    `grounding` swaps the presence and answer legs for variants that also ask
    for coordinates. It defaults to False so that every existing caller - the
    prompt viewer on 7872, the tests, tools - keeps getting the prompt it
    always got; only the mode-3 runner passes cfg.vlm_grounding through.
    """
    if leg == LEG_QUALITY:
        return QUALITY_USER_TEMPLATE
    if leg == LEG_PRESENCE:
        template = (PRESENCE_USER_TEMPLATE_GROUNDED if grounding
                    else PRESENCE_USER_TEMPLATE)
        return template.format(subject=question.effective_subject,
                               grounding_rule=GROUNDING_RULE)
    if leg == LEG_ANSWER:
        template = (ANSWER_USER_TEMPLATE_GROUNDED if grounding
                    else ANSWER_USER_TEMPLATE)
        return template.format(
            question_text=question.effective_question_text,
            semantics=question.answer_semantics,
            grounding_rule=GROUNDING_RULE)
    raise KeyError(f"Unknown leg {leg!r}; expected one of {LEGS}")
