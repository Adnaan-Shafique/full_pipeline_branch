"""The plugin layer: domains, detector classes and questions, loaded from YAML.

    config/domains.yaml          the first dropdown
    config/classes.yaml          every detector class, trained or not
    config/questions/*.yaml      the questions, grouped into files by domain

Editing one of those three files is the whole procedure for adding or removing
a question, a domain or an object class. Nothing in app/ needs to change, and
no file lists the questions a second time — the dropdowns, the mode 3 legs, the
detector's class names and the relevance matching are all derived from here.

── Why loading is tolerant and reports rather than raises ────────────────────
A malformed config file must not take the demo down five minutes before it
runs. So every problem this module can survive becomes a WARNING on
`Registry.warnings`, which the UI prints in its status line, and the entry that
caused it is dropped. The two things it will not survive are a config tree that
yields no questions at all, and a question whose system prompt is blank — the
first leaves nothing to demo, and the second fails silently at the server (see
Question.__post_init__). Both fall back to the built-ins with a loud warning.

── The built-in fallback ────────────────────────────────────────────────────
`BUILTIN_QUESTIONS` holds the two Site Safety questions as Python, identical to
the YAML. It is what you get when pyyaml is missing, when config/ has been
deleted, or when every YAML file failed to parse. It exists so that
`import pipeline.questions` works on a machine with nothing installed, which is
the property the whole test suite depends on.

Pure stdlib apart from an optional `import yaml`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .question_types import (NumericRule, OCRSpec, Question, build_user_template)

# app/pipeline/registry.py -> app/pipeline -> app -> <project root>
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"

# Override for tests and for running against a second config tree without
# editing code. Mirrors the FIELDOPS_* convention in config.py.
CONFIG_DIR_ENV = "FIELDOPS_CONFIG_DIR"

DEFAULT_DOMAIN_ID = "site_safety"
DEFAULT_DOMAIN_LABEL = "Site Safety"

REQUIRED_QUESTION_FIELDS = ("id", "label", "system_prompt")


@dataclass(frozen=True)
class Domain:
    id: str
    label: str
    order: int = 999
    blurb: str = ""


@dataclass(frozen=True)
class DetectorClass:
    """One object class. `name` is what appears on a box, in the detection block
    and in the UI; `aliases` only ever widen matching."""

    id: str
    name: str
    aliases: list[str] = field(default_factory=list)
    trained: bool = False
    yolox_index: Optional[int] = None

    @property
    def match_terms(self) -> list[str]:
        """Everything select_relevant() should match this class on. The id is
        included because annotation folders label boxes with it ("gps_antenna")
        far more often than with the display name."""
        terms = [self.name, self.id] + list(self.aliases)
        seen, out = set(), []
        for t in terms:
            t = (t or "").strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)
        return out


@dataclass
class Registry:
    domains: dict            # id -> Domain
    classes: dict            # id -> DetectorClass
    questions: dict          # id -> Question
    warnings: list           # human-readable, shown in the UI status line
    source: str              # "config/" or "built-in (…reason…)"

    # ── Domains ──────────────────────────────────────────────────────────────
    def ordered_domains(self) -> list:
        """Domains that actually have at least one question, in `order`.

        A domain with no questions is dropped rather than shown empty: an empty
        dropdown on screen reads as "the questions failed to load", which is the
        wrong story when the truth is "nobody has written those questions yet".
        """
        populated = {q.domain for q in self.questions.values()}
        chosen = [d for d in self.domains.values() if d.id in populated]
        return sorted(chosen, key=lambda d: (d.order, d.label))

    def domain_choices(self) -> list:
        """[(label, id), ...] for the domain dropdown."""
        return [(d.label, d.id) for d in self.ordered_domains()]

    def questions_in(self, domain_id: str) -> list:
        """Questions of one domain, in the order their YAML file lists them —
        which is the order the checklist itself is walked in, and therefore the
        order an inspector expects to find them in."""
        return [q for q in self.questions.values() if q.domain == domain_id]

    def question_choices(self, domain_id: str) -> list:
        return [(q.label, q.id) for q in self.questions_in(domain_id)]

    def domain_of(self, question_id: str) -> Optional[Domain]:
        q = self.questions.get(question_id)
        return self.domains.get(q.domain) if q else None

    # ── Classes ──────────────────────────────────────────────────────────────
    def trained_classes(self) -> list:
        """Trained classes in yolox_index order — the checkpoint's own order.

        This is where PipelineConfig.yolox_class_names comes from. Sorting by
        the explicit index rather than by position in the YAML file is the whole
        point: reordering classes.yaml for readability must not relabel every
        detection. See the header of config/classes.yaml.
        """
        trained = [c for c in self.classes.values() if c.trained and c.yolox_index is not None]
        return sorted(trained, key=lambda c: c.yolox_index)

    def yolox_class_names(self) -> tuple:
        return tuple(c.name for c in self.trained_classes())

    def untrained_classes_for(self, question_id: str) -> list:
        """The question's classes that have no trained weights. The UI names
        these on the card so nobody reads "detector found nothing" as evidence
        of absence when the truth is that nothing was ever trained to find it."""
        q = self.questions.get(question_id)
        if not q:
            return []
        return [self.classes[cid] for cid in q.class_ids
                if cid in self.classes and not self.classes[cid].trained]


# ─────────────────────────────── Built-in fallback ───────────────────────────
# Identical to config/questions/site_safety.yaml. Kept as Python so that the
# registry can always produce a working demo, including on a machine with no
# pyyaml — see the module docstring.

_HAZARD_SYSTEM = (
    "You are a site-safety inspector reviewing photographs of telecom equipment "
    "installations. Your task is to determine whether a hazardous-warning sign, "
    "label, or placard is visible in the photograph - for example high-voltage "
    "warnings, electrical-hazard symbols, RF-radiation warnings, danger/caution "
    "placards, or equivalent safety signage. Judge only from what is actually "
    "visible; do not infer that signage exists because the equipment type would "
    "normally require it. If the image is too unclear, cropped, or distant to "
    'tell, answer "unknown". Respond with JSON only.'
)

_ANTENNA_SYSTEM = (
    "You are a telecom installation inspector reviewing photographs of GPS "
    "antennas mounted at cell sites. A GPS antenna functions correctly only when "
    "it has a clear, unobstructed view of the open sky above it. Your task is to "
    "judge, from the photograph, whether the antenna's upward view is clear. Pay "
    "attention to what sits directly above and around the antenna: roof "
    "overhangs, canopies, walls, tree canopy, cable trays, other antennas or "
    "equipment, or an indoor/sheltered location. Do not judge the antenna's "
    "cabling, cosmetic condition, or brand. If the region above the antenna is "
    'not visible in the frame, answer "unknown". Respond with JSON only.'
)

_HAZARD_TEMPLATE = """Does this photograph contain a hazardous-warning sign or label?
yes = at least one hazard/warning/danger sign or safety placard is visible.
no  = no such signage is visible anywhere in the frame.
{detection_block}
{output_contract}"""

_ANTENNA_TEMPLATE = """Is the GPS antenna in this photograph properly mounted with an open view of the sky?
yes = the antenna is open to the sky; nothing significant obstructs its upward view.
no  = the antenna's view of the sky is blocked or partially blocked (overhang, canopy,
      wall, foliage, other equipment, or an indoor/sheltered mounting).
{detection_block}
{output_contract}"""

BUILTIN_CLASSES = {
    "gps_antenna": DetectorClass(
        id="gps_antenna", name="GPS Antenna", aliases=["gps antenna", "gps", "antenna"],
        trained=True, yolox_index=0),
    "hazard_sign": DetectorClass(
        id="hazard_sign", name="Warning sign (HV / RF radiation)",
        aliases=["hazard_sign", "warning_sign", "warning sign", "hazard", "sign",
                 "placard"],
        trained=True, yolox_index=1),
}

BUILTIN_DOMAINS = {
    DEFAULT_DOMAIN_ID: Domain(id=DEFAULT_DOMAIN_ID, label=DEFAULT_DOMAIN_LABEL, order=1),
}

BUILTIN_QUESTIONS = {
    "hazard_warning": Question(
        id="hazard_warning",
        label="Is a hazardous-warning sign present?",
        system_prompt=_HAZARD_SYSTEM,
        user_template=_HAZARD_TEMPLATE,
        answer_semantics=(
            "YES = a hazard/warning/danger sign or safety placard is visible.  "
            "NO = no such signage anywhere in the frame."),
        relevant_classes=["Warning sign (HV / RF radiation)", "hazard_sign",
                          "warning_sign", "hazard", "sign", "placard"],
        default_class_names=["Warning sign (HV / RF radiation)"],
        domain=DEFAULT_DOMAIN_ID,
        subject="a hazard, warning or danger sign, label or placard",
        question_text="does this photograph contain a hazardous-warning sign or label?",
        class_ids=["hazard_sign"],
    ),
    "gps_antenna": Question(
        id="gps_antenna",
        label="Is the GPS antenna open to the sky?",
        system_prompt=_ANTENNA_SYSTEM,
        user_template=_ANTENNA_TEMPLATE,
        answer_semantics=(
            "YES = the antenna's upward view is clear.  "
            "NO = its view of the sky is blocked or partially blocked."),
        relevant_classes=["GPS Antenna", "gps_antenna", "gps", "antenna"],
        default_class_names=["GPS Antenna"],
        domain=DEFAULT_DOMAIN_ID,
        subject="a GPS antenna",
        question_text=("is the GPS antenna properly mounted with an open view of "
                       "the sky?"),
        class_ids=["gps_antenna"],
    ),
}


def builtin_registry(reason: str) -> Registry:
    return Registry(
        domains=dict(BUILTIN_DOMAINS), classes=dict(BUILTIN_CLASSES),
        questions=dict(BUILTIN_QUESTIONS), warnings=[reason] if reason else [],
        source=f"built-in ({reason})" if reason else "built-in")


# ─────────────────────────────── Loading ─────────────────────────────────────

def config_dir() -> Path:
    raw = os.environ.get(CONFIG_DIR_ENV)
    return Path(raw) if raw and raw.strip() else DEFAULT_CONFIG_DIR


def _read_yaml(path: Path, warnings: list):
    """One YAML document, or None with a warning. Never raises: a single bad
    file costs its own entries, not the whole registry."""
    try:
        import yaml
    except ImportError:
        warnings.append(
            "pyyaml is not installed, so config/ cannot be read - falling back "
            "to the two built-in questions. pip install pyyaml")
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.append(f"{path.name}: could not be parsed ({exc}) - skipped")
        return None
    if data is None:
        warnings.append(f"{path.name}: is empty - skipped")
        return None
    if not isinstance(data, dict):
        warnings.append(f"{path.name}: top level must be a mapping, got "
                        f"{type(data).__name__} - skipped")
        return None
    return data


def _load_domains(root: Path, warnings: list) -> dict:
    path = root / "domains.yaml"
    if not path.exists():
        warnings.append(f"{path} not found - using the built-in "
                        f"{DEFAULT_DOMAIN_LABEL!r} domain only")
        return dict(BUILTIN_DOMAINS)
    data = _read_yaml(path, warnings)
    if not data:
        return dict(BUILTIN_DOMAINS)
    out = {}
    for index, entry in enumerate(data.get("domains") or []):
        if not isinstance(entry, dict) or not entry.get("id"):
            warnings.append(f"domains.yaml: entry {index} has no id - skipped")
            continue
        did = str(entry["id"]).strip()
        if did in out:
            warnings.append(f"domains.yaml: duplicate domain id {did!r} - "
                            f"the later entry is ignored")
            continue
        out[did] = Domain(
            id=did, label=str(entry.get("label") or did),
            # Missing order falls back to file position, so appending a domain
            # without an order still lands where it was written.
            order=int(entry.get("order", index + 1)),
            blurb=str(entry.get("blurb") or "").strip())
    if not out:
        warnings.append("domains.yaml: no usable domains - using the built-in one")
        return dict(BUILTIN_DOMAINS)
    return out


def _load_classes(root: Path, warnings: list) -> dict:
    path = root / "classes.yaml"
    if not path.exists():
        warnings.append(f"{path} not found - using the two built-in trained classes")
        return dict(BUILTIN_CLASSES)
    data = _read_yaml(path, warnings)
    if not data:
        return dict(BUILTIN_CLASSES)
    out, by_index = {}, {}
    for index, entry in enumerate(data.get("classes") or []):
        if not isinstance(entry, dict) or not entry.get("id"):
            warnings.append(f"classes.yaml: entry {index} has no id - skipped")
            continue
        cid = str(entry["id"]).strip()
        if cid in out:
            warnings.append(f"classes.yaml: duplicate class id {cid!r} - "
                            f"the later entry is ignored")
            continue
        trained = bool(entry.get("trained", False))
        raw_index = entry.get("yolox_index")
        yolox_index = None
        if trained:
            if raw_index is None:
                warnings.append(
                    f"classes.yaml: {cid!r} is trained: true but has no "
                    f"yolox_index - it cannot be placed in the checkpoint's "
                    f"class order and is treated as untrained")
                trained = False
            else:
                try:
                    yolox_index = int(raw_index)
                except (TypeError, ValueError):
                    warnings.append(f"classes.yaml: {cid!r} yolox_index "
                                    f"{raw_index!r} is not an integer - treated "
                                    f"as untrained")
                    trained = False
                else:
                    if yolox_index in by_index:
                        # Two names for one head output. Whichever wins, half
                        # the boxes get the wrong label - and a wrong label is
                        # evidence in the VLM prompt, not just a caption.
                        warnings.append(
                            f"classes.yaml: yolox_index {yolox_index} is claimed "
                            f"by both {by_index[yolox_index]!r} and {cid!r} - "
                            f"{cid!r} is treated as untrained. Every detection "
                            f"at that index would otherwise carry an ambiguous "
                            f"label")
                        trained, yolox_index = False, None
                    else:
                        by_index[yolox_index] = cid
        aliases = [str(a) for a in (entry.get("aliases") or []) if str(a).strip()]
        out[cid] = DetectorClass(
            id=cid, name=str(entry.get("name") or cid), aliases=aliases,
            trained=trained, yolox_index=yolox_index)

    # A gap means the head has an output nothing names, so every class after the
    # gap is off by one - the exact failure classes.yaml's header describes.
    indices = sorted(by_index)
    if indices and indices != list(range(len(indices))):
        warnings.append(
            f"classes.yaml: trained yolox_index values are {indices}, which is "
            f"not a gapless 0..N range. The checkpoint's head has one output per "
            f"index, so a gap shifts every label after it. Detection labels "
            f"cannot be trusted until this is fixed")
    if not out:
        warnings.append("classes.yaml: no usable classes - using the built-ins")
        return dict(BUILTIN_CLASSES)
    return out


def _ocr_from(entry: dict, qid: str, warnings: list) -> OCRSpec:
    raw = entry.get("ocr")
    if not isinstance(raw, dict) or not raw.get("enabled"):
        return OCRSpec()
    numeric = None
    raw_numeric = raw.get("numeric")
    if isinstance(raw_numeric, dict):
        try:
            numeric = NumericRule(
                label=str(raw_numeric.get("label") or "Reading"),
                unit=str(raw_numeric.get("unit") or ""),
                comparator=str(raw_numeric.get("comparator") or "<="),
                limit=float(raw_numeric["limit"]),
                pattern=str(raw_numeric["pattern"]))
        except (KeyError, TypeError, ValueError) as exc:
            # The OCR stage still runs and still shows its text; only the
            # threshold line is lost, and the model was never relying on it.
            warnings.append(f"{qid}: numeric rule is unusable ({exc}) - OCR will "
                            f"run but no threshold will be checked")
            numeric = None
    try:
        return OCRSpec(enabled=True, scope=str(raw.get("scope") or "boxes_then_image"),
                       numeric=numeric,
                       min_confidence=float(raw.get("min_confidence", 0.30)))
    except ValueError as exc:
        warnings.append(f"{qid}: {exc} - OCR disabled for this question")
        return OCRSpec()


def _question_from(entry: dict, classes: dict, domains: dict, warnings: list):
    missing = [f for f in REQUIRED_QUESTION_FIELDS if not str(entry.get(f) or "").strip()]
    if missing:
        warnings.append(f"question {entry.get('id') or '<no id>'!r}: missing "
                        f"{', '.join(missing)} - skipped")
        return None
    qid = str(entry["id"]).strip()

    domain = str(entry.get("domain") or DEFAULT_DOMAIN_ID).strip()
    if domain not in domains:
        warnings.append(f"{qid}: domain {domain!r} is not in domains.yaml - "
                        f"filed under {DEFAULT_DOMAIN_ID!r}")
        domain = DEFAULT_DOMAIN_ID if DEFAULT_DOMAIN_ID in domains else next(iter(domains))

    # Resolve class ids into the names and aliases select_relevant() matches on.
    class_ids, relevant, default_names = [], [], []
    for cid in (entry.get("classes") or []):
        cid = str(cid).strip()
        cls = classes.get(cid)
        if cls is None:
            warnings.append(f"{qid}: class {cid!r} is not in classes.yaml - "
                            f"ignored. Detections of it will not be surfaced")
            continue
        class_ids.append(cid)
        relevant.extend(cls.match_terms)
        default_names.append(cls.name)
    if not class_ids:
        warnings.append(f"{qid}: no usable classes. select_relevant() falls back "
                        f"to showing every detection for this question")

    ocr = _ocr_from(entry, qid, warnings)

    yes_means = str(entry.get("yes_means") or "").strip()
    no_means = str(entry.get("no_means") or "").strip()
    semantics = str(entry.get("answer_semantics") or "").strip()
    if not semantics and (yes_means or no_means):
        semantics = f"YES = {yes_means}  NO = {no_means}"

    template = str(entry.get("user_template") or "").strip()
    if not template:
        if not (yes_means and no_means):
            warnings.append(f"{qid}: needs either user_template, or both "
                            f"yes_means and no_means - skipped")
            return None
        template = build_user_template(
            str(entry.get("question_text") or entry["label"]),
            yes_means, no_means, with_ocr=ocr.enabled)

    try:
        return Question(
            id=qid, label=str(entry["label"]).strip(),
            system_prompt=str(entry["system_prompt"]).strip(),
            user_template=template,
            answer_semantics=semantics,
            relevant_classes=relevant,
            default_class_names=default_names,
            sampling=dict(entry.get("sampling") or {}),
            domain=domain,
            subject=str(entry.get("subject") or "").strip(),
            question_text=str(entry.get("question_text") or "").strip(),
            ocr=ocr, class_ids=class_ids,
            system_prompt_no_ocr=str(entry.get("system_prompt_no_ocr") or "").strip())
    except ValueError as exc:
        # Question.__post_init__ rejected it — a blank system prompt or a
        # template missing a placeholder. Both are silent failures at runtime,
        # so the entry is dropped rather than shipped.
        warnings.append(f"{qid}: {exc} - skipped")
        return None


def _load_questions(root: Path, classes: dict, domains: dict, warnings: list) -> dict:
    folder = root / "questions"
    if not folder.is_dir():
        warnings.append(f"{folder} not found - no questions could be loaded")
        return {}
    out = {}
    # Sorted so the question order is the same on every machine; within a file,
    # the YAML order is preserved, which is the checklist's own order.
    for path in sorted(folder.glob("*.yaml")) + sorted(folder.glob("*.yml")):
        data = _read_yaml(path, warnings)
        if not data:
            continue
        entries = data.get("questions")
        if not isinstance(entries, list):
            warnings.append(f"{path.name}: has no `questions:` list - skipped")
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                warnings.append(f"{path.name}: a question entry is not a mapping "
                                f"- skipped")
                continue
            question = _question_from(entry, classes, domains, warnings)
            if question is None:
                continue
            if question.id in out:
                warnings.append(f"duplicate question id {question.id!r} in "
                                f"{path.name} - the later entry is ignored")
                continue
            out[question.id] = question
    return out


def load_registry(root=None) -> Registry:
    """Read config/ into a Registry. Never raises."""
    root = Path(root) if root is not None else config_dir()
    warnings: list = []
    if not root.is_dir():
        return builtin_registry(f"{root} does not exist")
    try:
        import yaml  # noqa: F401
    except ImportError:
        return builtin_registry(
            "pyyaml is not installed, so config/ cannot be read. The two "
            "built-in Site Safety questions are available; pip install pyyaml "
            "to get the rest")

    domains = _load_domains(root, warnings)
    classes = _load_classes(root, warnings)
    questions = _load_questions(root, classes, domains, warnings)
    if not questions:
        # Fall back on the QUESTIONS only. An earlier version returned the whole
        # built-in registry here, which threw away a classes.yaml that had
        # parsed perfectly well - so a typo in one question file silently reset
        # the detector's class list to the two originals. Each of the three
        # files stands or falls on its own.
        warnings.append(
            "no questions could be loaded from config/questions/ - falling back "
            "to the two built-in Site Safety questions. Domains and classes that "
            "did load are kept")
        questions = dict(BUILTIN_QUESTIONS)
        # Those questions name a domain, and a question in a domain the registry
        # has never heard of is unreachable from the dropdown.
        for question in questions.values():
            domains.setdefault(question.domain,
                               BUILTIN_DOMAINS.get(question.domain)
                               or Domain(id=question.domain, label=question.domain))
        return Registry(domains=domains, classes=classes, questions=questions,
                        warnings=warnings,
                        source=f"{root} (classes/domains) + built-in questions")
    return Registry(domains=domains, classes=classes, questions=questions,
                    warnings=warnings, source=str(root))


_CACHE: Optional[Registry] = None


def get_registry(reload: bool = False) -> Registry:
    """The process-wide registry. Loaded once — the UI reads it on every
    callback, and re-parsing twenty YAML files per keystroke is waste.

    `reload=True` re-reads from disk, which is what the UI's Reload button
    calls so a question can be edited and picked up without a restart.
    """
    global _CACHE
    if _CACHE is None or reload:
        _CACHE = load_registry()
    return _CACHE
