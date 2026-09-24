"""Data contracts passed between pipeline stages.

Pure stdlib on purpose - importing this module must never pull in cv2, torch,
onnxruntime or gradio, so the orchestrator's shapes can be unit-tested on a
machine with none of the heavy dependencies installed.

Two fields here are NOT in the original integration plan and are load-bearing:

  QualityStageResult.resolution_ok
  QualityStageResult.ignore_resolution_used
      quality_check.assess_quality() computes
          passed = resolution_ok and score >= threshold
      so an image can score 82 against a threshold of 65 and still FAIL, purely
      on the resolution floor. Without these fields the UI renders
      "FAIL 82.1 / 65", which reads to an audience as a scoring bug rather than
      a resolution rejection. See .fail_kind below.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# Verdict strings, matching run_quality_batch_foreground.py's CSV vocabulary
# exactly so a pipeline row and a CLI row can be compared without translation.
PASS = "PASS"
FAIL = "FAIL"
ERROR = "ERROR"

# PipelineRecord.stopped_at values.
STOPPED_QUALITY = "quality"
STOPPED_DETECTION = "detection"
STOPPED_OCR = "ocr"
STOPPED_VLM = "vlm"
STOPPED_COMPLETE = "complete"

# Detection.source values. The stub path must be distinguishable from the real
# one everywhere downstream - see the provenance requirement in plan section 3.3.
SOURCE_MODEL = "model"
SOURCE_ANNOTATION = "annotation_file"


@dataclass
class QualityStageResult:
    """Stage 1 output. Mirrors what run_quality_batch_foreground._process_one()
    computes per image, minus the file-copying and CSV-row concerns."""

    passed: bool
    score: float                 # the score the verdict was made on (cropped if segmented)
    threshold: float
    resolution_ok: bool
    ignore_resolution_used: bool
    failure_reasons: list[str]
    retake_instructions: list[str]
    foreground_box: Optional[tuple[int, int, int, int]]   # x1, y1, x2, y2
    segmentation_used: bool
    whole_frame_score: float
    width: int
    height: int
    foreground_score: Optional[float] = None   # None when the segmenter found no plausible box
    mask_area_frac: Optional[float] = None
    mask_mean_alpha: Optional[float] = None
    annotated_path: Optional[str] = None       # green foreground box drawn, for the UI
    error: Optional[str] = None                # set when scoring itself raised
    # "classical" = the MM-IQA cue fusion in quality_check.py.
    # "vlm"       = mode 3, where the model judged the photo instead. There is
    #               no 0-100 score in that case, so `score` stays 0.0 and the
    #               headline must not pretend otherwise.
    assessed_by: str = "classical"
    # Wall-clock cost of this stage, for the benchmark's per-photograph
    # breakdown. Informational only - nothing gates on it, and 0.0 means "not
    # measured" rather than "instant", which is why the benchmark reports it as
    # absent instead of as zero.
    elapsed_ms: float = 0.0

    @property
    def verdict(self) -> str:
        if self.error:
            return ERROR
        return PASS if self.passed else FAIL

    @property
    def fail_kind(self) -> Optional[str]:
        """Why this image failed, so the UI can say the right thing.

        "resolution" - below the min_width/min_height floor, regardless of score.
        "score"      - scored under the threshold.
        "both"       - both of the above.
        None         - it passed, or it errored.
        """
        if self.error or self.passed:
            return None
        below_floor = not self.resolution_ok and not self.ignore_resolution_used
        below_score = self.score < self.threshold
        if below_floor and below_score:
            return "both"
        if below_floor:
            return "resolution"
        return "score"

    @property
    def headline(self) -> str:
        """One line for the UI card, honest about which gate rejected the image."""
        if self.error:
            return f"{ERROR} - {self.error}"
        if self.assessed_by == "vlm":
            # No numeric score exists here. Showing "PASS 0.0 / 65" would be a
            # lie dressed as precision.
            return f"{self.verdict} - judged by the model, no numeric score"
        base = f"{self.verdict} {self.score:.1f} / {self.threshold:.0f}"
        if self.fail_kind == "resolution":
            return f"{base}  (score is fine; failed the {self.width}x{self.height} resolution floor)"
        if self.fail_kind == "both":
            return f"{base}  (also below the resolution floor)"
        return base


@dataclass
class Detection:
    """One detected object. Absolute pixel coordinates, in the same EXIF-corrected
    frame every other stage sees - see the orchestrator's load-once contract."""

    label: str
    confidence: float
    box: list[float]        # x1, y1, x2, y2 absolute px
    source: str             # SOURCE_MODEL | SOURCE_ANNOTATION

    @classmethod
    def from_model_dict(cls, d: dict, source: str = SOURCE_MODEL) -> "Detection":
        """inference.run_inference() returns {"class", "confidence", "box"}; this
        dataclass calls it `label`. Doing the rename here, in one place, is what
        keeps the real and stub detector backends interchangeable."""
        return cls(
            label=str(d["class"]),
            confidence=float(d["confidence"]),
            box=[float(v) for v in d["box"]],
            source=source,
        )

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class DetectionStageResult:
    """Stage 2 output. `model_name` is what the UI prints as provenance and must
    never imply a model ran when one did not."""

    detections: list[Detection]
    annotated_path: Optional[str]
    model_name: str          # e.g. "human annotation (no model loaded)"
    note: str = ""           # e.g. "no annotation file found for IMG_0042.txt"

    # Explicit, never inferred. An earlier version derived this by looking for
    # "model" inside model_name, which is a substring of "no model loaded" and
    # therefore reported stub results as real ones. Provenance is an honesty
    # requirement (plan section 3.3), so it is set by the detector, not guessed.
    is_stub: bool = False
    # Mode 3 only: the model reports whether the subject is visible but draws no
    # boxes, so presence is carried as text rather than geometry. None means
    # this result came from a real detector and the box list is the answer.
    presence: Optional[str] = None          # "yes" | "no" | "unknown"
    presence_reasoning: str = ""
    # Mode 3 only: regions the MODEL said the subject occupies, as
    # vlm_grounding.GroundingBox. Kept in a field of their own rather than in
    # `detections` because they are a different kind of thing - a claim, not a
    # measurement - and anything that reads `detections` (the VLM prompt's
    # detection block, select_relevant, the crop chooser) must not pick them up
    # and start treating them as detector output.
    claimed_boxes: list = field(default_factory=list)
    grounding_note: str = ""
    elapsed_ms: float = 0.0   # see QualityStageResult.elapsed_ms
    # {overlay variant: path} for mode 3 - "box_label", "box", "off". Written
    # once during the run so the UI can switch overlays without re-running
    # anything. Absent keys simply mean that rendering was not produced (no
    # claim to draw, or a failed write); the UI falls back rather than assuming.
    overlay_paths: dict = field(default_factory=dict)

    def top(self) -> Optional[Detection]:
        return max(self.detections, key=lambda d: d.confidence, default=None)


# OCRStageResult.scope_used values. Which one it is decides how the boxes in
# `lines` should be read: box-scoped coordinates are in the FULL frame (the
# stage translates them back), image-scoped ones already are.
OCR_FROM_BOXES = "boxes"
OCR_FROM_IMAGE = "image"
OCR_SKIPPED = "skipped"


@dataclass
class OCRLine:
    """One recognised line of text, in the same EXIF-corrected frame every other
    stage sees - the load-once contract applies to OCR as much as to detection.

    `polygon` is the text detector's own quadrilateral, which is not
    axis-aligned on angled labels; `box` is its bounding rectangle. Both are in
    FULL-FRAME pixels even when the read came from a crop, because a card that
    draws them has only the full frame to draw on.
    """

    text: str
    text_confidence: float
    box: list           # x1, y1, x2, y2 absolute px in the full frame
    polygon: list = field(default_factory=list)
    # Which detection this line was read out of, when the read was box-scoped.
    # None for a whole-frame read. Shown on the card so an operator can tell
    # "read off the SPD label" from "read off something else in the photo".
    from_label: Optional[str] = None


@dataclass
class OCRStageResult:
    """Stage 2b output. Runs between detection and the model, for the questions
    whose YAML sets `ocr.enabled: true` - and never in mode 3, where the model
    reads the text itself.

    `error` is set when OCR could not run at all (no models, rapidocr missing,
    the engine raised). It is NOT set when OCR ran and found no text: that is a
    successful read of a photograph with nothing legible in it, and the two must
    stay distinguishable, because only the first is a problem with the demo host.
    """

    lines: list                      # list[OCRLine], reading order
    engine: str                      # e.g. "PP-OCRv6 tiny (no angle classifier...)"
    scope_used: str = OCR_SKIPPED    # OCR_FROM_BOXES | OCR_FROM_IMAGE | OCR_SKIPPED
    elapsed_ms: float = 0.0
    boxes_read: int = 0              # how many detections were cropped and read
    annotated_path: Optional[str] = None   # magenta line boxes, for the UI
    note: str = ""                   # e.g. "no detections; read the whole frame"
    error: Optional[str] = None
    # The numeric rule's output, when the question carries one AND a number was
    # matched. Shape: {"value", "passes", "matched", "sentence"}. None means no
    # rule, or a rule that matched nothing - never a failed check. See
    # question_types.NumericRule.
    numeric: Optional[dict] = None

    @property
    def ran(self) -> bool:
        return self.scope_used != OCR_SKIPPED and not self.error

    @property
    def text(self) -> str:
        """Every line joined in reading order - what the numeric rule is run
        over and what the card shows as one block."""
        return " ".join(ln.text.strip() for ln in self.lines if (ln.text or "").strip())

    @property
    def mean_confidence(self) -> float:
        scores = [ln.text_confidence for ln in self.lines]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def headline(self) -> str:
        """One line for the UI card, honest about the three different nothings:
        an OCR stage that could not run, one that ran and read nothing, and one
        that was never asked to run for this question."""
        if self.error:
            return f"{ERROR} - {self.error}"
        if self.scope_used == OCR_SKIPPED:
            return "not run for this question"
        where = ("whole frame" if self.scope_used == OCR_FROM_IMAGE
                 else f"{self.boxes_read} detected box{'es' if self.boxes_read != 1 else ''}")
        if not self.lines:
            return f"no legible text found ({where}, {self.elapsed_ms:.0f} ms)"
        return (f"{len(self.lines)} line{'s' if len(self.lines) != 1 else ''} read "
                f"from the {where}, mean confidence {self.mean_confidence:.2f} "
                f"({self.elapsed_ms:.0f} ms)")


@dataclass
class VLMAnswer:
    """Stage 3 output. `answer` is always one of yes / no / unknown - the parser
    guarantees it, so the UI chip never has to handle a surprise value."""

    answer: str              # "yes" | "no" | "unknown"
    reasoning: str
    raw_text: str
    model: str
    elapsed_s: float
    error: Optional[str] = None
    is_mock: bool = False
    # Mode 3 only: the region the model said it drew its answer from. Distinct
    # from the presence box - "where the subject is" and "what I looked at to
    # decide" are not the same claim, and on the OCR questions the difference
    # is the whole point (which display did you actually read?).
    evidence_boxes: list = field(default_factory=list)

    @property
    def chip(self) -> str:
        return self.answer.upper()

    @property
    def provenance(self) -> str:
        if self.is_mock:
            return "MOCK - GPU server unreachable"
        return self.model


@dataclass
class PipelineRecord:
    """One image's journey through all three legs.

    `stem` is the join key across every leg and is captured at upload time from
    the ORIGINAL filename. Both of the renaming paths in the existing code -
    run_quality_batch.unique_destination() ("name__1.jpg", an integer counter)
    and batch_ui._stage_uploaded_files() ("name__a1b2c3d4.jpg", a random uuid)
    - destroy it, and the uuid one is not even stable across two runs of the
    same file. Never re-derive this from an output path.
    """

    filename: str
    source_path: str
    stem: str
    question_id: str
    quality: QualityStageResult
    detection: Optional[DetectionStageResult] = None
    ocr: Optional[OCRStageResult] = None
    vlm: Optional[VLMAnswer] = None
    stopped_at: str = STOPPED_COMPLETE
    extra: dict[str, Any] = field(default_factory=dict)

    def to_flat_row(self) -> dict:
        """One flat CSV row per image, per plan section 5.

        Deliberately NOT the same schema as quality_review_foreground.csv - that
        file's columns are read by batch_ui.py and must stay byte-identical, so
        this is a separate artefact with a superset of the interesting fields.
        """
        q = self.quality
        d = self.detection
        o = self.ocr
        v = self.vlm
        return {
            "filename": self.filename,
            "stem": self.stem,
            "question": self.question_id,
            "quality_verdict": q.verdict,
            "quality_score": round(q.score, 1),
            "quality_threshold": q.threshold,
            "quality_fail_kind": q.fail_kind or "",
            "quality_elapsed_ms": round(q.elapsed_ms, 1) if q.elapsed_ms else None,
            "resolution_ok": q.resolution_ok,
            "width": q.width,
            "height": q.height,
            "whole_frame_score": round(q.whole_frame_score, 1),
            "foreground_score": round(q.foreground_score, 1) if q.foreground_score is not None else None,
            "segmentation_used": q.segmentation_used,
            "failure_reasons": "; ".join(q.failure_reasons),
            "retake_instructions": " | ".join(q.retake_instructions),
            "detection_source": (d.model_name if d else ""),
            "detections": "; ".join(f"{x.label}:{x.confidence:.2f}" for x in d.detections) if d else "",
            "detection_note": (d.note if d else ""),
            "detection_elapsed_ms": (round(d.elapsed_ms, 1) if d and d.elapsed_ms else None),
            "ocr_engine": (o.engine if o and o.ran else ""),
            "ocr_scope": (o.scope_used if o else ""),
            "ocr_text": (o.text if o else ""),
            "ocr_confidence": (round(o.mean_confidence, 3) if o and o.lines else None),
            "ocr_lines": (len(o.lines) if o else 0),
            "ocr_elapsed_ms": (round(o.elapsed_ms, 1) if o else None),
            # The rule's reading and verdict, kept as separate columns so a
            # spreadsheet can sort on the number. Blank when no rule matched -
            # which is not the same as a failed check, and must not be read as
            # one, so ocr_numeric_passes stays None rather than False.
            "ocr_numeric_value": (o.numeric["value"] if o and o.numeric else None),
            "ocr_numeric_passes": (o.numeric["passes"] if o and o.numeric else None),
            "ocr_error": ((o.error or "") if o else ""),
            # Mode 3's claimed geometry. Blank for modes 1 and 2, which draw
            # real detector boxes and have nothing to claim.
            "vlm_boxes": "; ".join(
                f"[{int(b.box[0])},{int(b.box[1])},{int(b.box[2])},{int(b.box[3])}]"
                for b in (d.claimed_boxes if d else [])),
            "vlm_box_convention": "; ".join(
                sorted({b.convention for b in (d.claimed_boxes if d else [])})),
            # None, not 0.0, when there was no detector box to compare against -
            # "we did not measure" and "it scored zero" are opposite findings.
            "vlm_box_best_iou": (
                max((b.iou for b in d.claimed_boxes if b.iou is not None), default=None)
                if d and d.claimed_boxes else None),
            "vlm_answer": (v.answer if v else ""),
            "vlm_reasoning": (v.reasoning if v else ""),
            "vlm_model": (v.provenance if v else ""),
            "vlm_elapsed_s": (round(v.elapsed_s, 2) if v else None),
            "vlm_error": (v.error or "" if v else ""),
            "stopped_at": self.stopped_at,
            "source_path": self.source_path,
        }


FLAT_ROW_COLUMNS = [
    "filename", "stem", "question",
    "quality_verdict", "quality_score", "quality_threshold", "quality_fail_kind",
    "quality_elapsed_ms",
    "resolution_ok", "width", "height",
    "whole_frame_score", "foreground_score", "segmentation_used",
    "failure_reasons", "retake_instructions",
    "detection_source", "detections", "detection_note", "detection_elapsed_ms",
    "ocr_engine", "ocr_scope", "ocr_text", "ocr_confidence", "ocr_lines",
    "ocr_elapsed_ms", "ocr_numeric_value", "ocr_numeric_passes", "ocr_error",
    "vlm_boxes", "vlm_box_convention", "vlm_box_best_iou",
    "vlm_answer", "vlm_reasoning", "vlm_model", "vlm_elapsed_s", "vlm_error",
    "stopped_at", "source_path",
]
