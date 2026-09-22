"""Stage 2b - OCR, between detection and the model.

Runs only for the questions whose YAML sets `ocr.enabled: true`. Four do today:
the two SPD "installed" questions, where the class marking printed on the module
body is what tells a Class B module from a Class C one, and the two device
readings, where the answer IS a number on a display.

It never runs in mode 3. That mode's whole argument is that one model looking at
the whole photograph judges it more coherently than components that each see a
slice, and handing it PP-OCRv6's transcription would quietly settle the most
interesting comparison this demo can make - whether the model reads a
seven-segment display as well as a purpose-built OCR engine does. See MODES.md.

── What it produces, and what it does not ───────────────────────────────────
It produces EVIDENCE. The text goes into the prompt behind a hedge that tells
the model to prefer the image, and a numeric rule's threshold check goes in the
same way. The pipeline does not answer the question from the OCR output, even
when the output is a number and the question is a threshold - see DECISIONS.md
for the three failure modes (range multiplier, set-point vs measurement, °F vs
°C) that make a confident wrong number worse than no number at all.

Degrades like every other stage: no rapidocr, no models, or an engine that
raises all produce an OCRStageResult carrying `error`, which the UI prints and
which build_ocr_block() deliberately renders as NO prompt block at all. "OCR
failed" is information for the operator, never a piece of visual evidence.

No rapidocr or cv2 import at module scope.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from .question_types import (OCR_SCOPE_BOXES, OCR_SCOPE_BOXES_THEN_IMAGE,
                             OCR_SCOPE_IMAGE)
from .schemas import (OCR_FROM_BOXES, OCR_FROM_IMAGE, OCR_SKIPPED, OCRLine,
                      OCRStageResult)

# A whole-frame read on a 4000x3000 photograph is the expensive path; the det
# model's own limit_side_len caps what it actually sees, but the encode is still
# real. Nothing here resizes - read_image() hands the frame straight to
# RapidOCR, whose defaults are tuned for exactly this.
SKIPPED = OCRStageResult(lines=[], engine="", scope_used=OCR_SKIPPED)


def skipped(reason: str = "") -> OCRStageResult:
    """The result for a question that did not ask for OCR. A distinct object
    each time: OCRStageResult is mutable and a shared singleton would let one
    record's note leak into every other record's card."""
    return OCRStageResult(lines=[], engine="", scope_used=OCR_SKIPPED, note=reason)


def available(cfg=None) -> tuple:
    """(is_available, description). The description is shown in the UI's status
    line whether or not OCR can run, so a missing piece is named before a demo
    rather than during one."""
    from . import ocr_engine

    model_dir = _model_dir(cfg)
    variant = ocr_engine.default_variant(model_dir)
    if variant is None:
        return False, (f"no PP-OCRv6 ONNX files in {model_dir} - the OCR stage "
                       f"will be skipped and its questions answered from the "
                       f"image alone")
    try:
        import rapidocr  # noqa: F401
    except ImportError:
        return False, ("rapidocr is not installed - the OCR stage will be "
                       "skipped. pip install rapidocr onnxruntime")
    return True, ocr_engine.describe(model_dir)


def _model_dir(cfg) -> Path:
    from . import ocr_engine

    if cfg is None:
        return ocr_engine.OCR_MODEL_DIR
    # Same derivation as every other model path: from the config's root, never
    # recomputed from a module's own __file__.
    return Path(cfg.models_dir) / "ocr"


def preload(cfg=None) -> Optional[str]:
    """Build the engines before the first photograph. Returns an error string
    rather than raising, so a preflight can report it and carry on."""
    from . import ocr_engine

    ok, description = available(cfg)
    if not ok:
        return description
    try:
        ocr_engine.preload(model_dir=_model_dir(cfg))
    except Exception as exc:
        return f"the OCR engine failed to load: {exc}"
    return None


def run(image_bgr, question, detections=None, cfg=None, dest_path=None) -> OCRStageResult:
    """OCR one photograph for one question.

    `detections` should already be the RELEVANT ones - select_relevant() has run
    - because reading every box on a photograph that happens to contain a
    warning sign would put that sign's text into a prompt about an SPD label.
    """
    spec = getattr(question, "ocr", None)
    if spec is None or not spec.enabled:
        return skipped("this question does not use OCR")

    ok, description = available(cfg)
    if not ok:
        return OCRStageResult(lines=[], engine="", scope_used=OCR_SKIPPED,
                              error=description)

    from . import ocr_engine

    model_dir = _model_dir(cfg)
    dets = list(detections or [])
    started = time.perf_counter()

    try:
        if spec.scope == OCR_SCOPE_IMAGE:
            result = _read_whole(image_bgr, model_dir, description,
                                 note="scope is 'image'; the detector's boxes were not used")
        elif dets:
            result = _read_boxes(image_bgr, dets, model_dir, description)
            # A box-scoped read that found nothing is a real answer for
            # scope: boxes. For boxes_then_image it is not the end - the point
            # of the fallback is that these classes have no trained weights, so
            # "the detector found a box but no text in it" is far more often a
            # bad box than an absence of text.
            if not result.lines and spec.scope == OCR_SCOPE_BOXES_THEN_IMAGE:
                result = _read_whole(
                    image_bgr, model_dir, description,
                    note=(f"no text in the {len(dets)} detected box"
                          f"{'es' if len(dets) != 1 else ''}; read the whole frame instead"))
        elif spec.scope == OCR_SCOPE_BOXES:
            result = OCRStageResult(
                lines=[], engine=description, scope_used=OCR_FROM_BOXES, boxes_read=0,
                note="no relevant detections, and this question's scope is 'boxes' only")
        else:
            result = _read_whole(
                image_bgr, model_dir, description,
                note="no relevant detections; read the whole frame instead")
    except Exception as exc:
        # Anything RapidOCR or onnxruntime raises. One photograph's OCR failing
        # must not end a batch, and the question is still answerable from the
        # image - the model simply gets no OCR block.
        return OCRStageResult(lines=[], engine=description, scope_used=OCR_SKIPPED,
                              elapsed_ms=(time.perf_counter() - started) * 1000,
                              error=f"OCR failed on this image: {exc}")

    # Drop the lines the engine itself is unsure of, before they reach either
    # the prompt or the numeric rule. A 0.11-confidence read of "1B5" that the
    # rule then turns into "185 ohms, outside the limit" is the worst output
    # this stage can produce: a fabricated measurement with a verdict on it.
    kept = [ln for ln in result.lines if ln.text_confidence >= spec.min_confidence]
    dropped = len(result.lines) - len(kept)
    result.lines = kept
    if dropped:
        result.note = (result.note + "; " if result.note else "") + (
            f"{dropped} line{'s' if dropped != 1 else ''} below the "
            f"{spec.min_confidence:.2f} confidence floor discarded")

    if spec.numeric is not None and result.lines:
        result.numeric = spec.numeric.evaluate(result.text)
        if result.numeric is None:
            result.note = (result.note + "; " if result.note else "") + (
                f"no {spec.numeric.label.lower()} reading could be matched in "
                f"the text that was read")

    if dest_path is not None and result.lines:
        result.annotated_path = _annotate(image_bgr, result.lines, Path(dest_path))
    return result


def _read_whole(image_bgr, model_dir, description, note="") -> OCRStageResult:
    from . import ocr_engine

    raw, ocr_ms = ocr_engine.read_image(image_bgr, model_dir=model_dir)
    lines = [OCRLine(text=d["text"], text_confidence=d["text_confidence"],
                     box=d["box"], polygon=d["polygon"]) for d in raw]
    return OCRStageResult(lines=lines, engine=description, scope_used=OCR_FROM_IMAGE,
                          elapsed_ms=ocr_ms, boxes_read=0, note=note)


def _read_boxes(image_bgr, detections, model_dir, description) -> OCRStageResult:
    """One read per detection, highest confidence first.

    read_box() returns the crop's text joined into one string with no geometry,
    so each box contributes at most ONE OCRLine whose box is the detection's own
    - in full-frame coordinates, which is what the card draws on and what the
    load-once contract requires. Per-line geometry inside a crop is available
    but deliberately not used: translating it back through the pad-and-upscale
    in crop_box() is arithmetic with a silent off-by-a-scale-factor in it, and
    the boxes it would produce say nothing the detection's box does not.
    """
    from . import ocr_engine

    lines, total_ms, read = [], 0.0, 0
    for det in sorted(detections, key=lambda d: -d.confidence):
        out = ocr_engine.read_box(image_bgr, det.box, model_dir=model_dir)
        total_ms += out["ocr_ms"]
        read += 1
        if (out["text"] or "").strip():
            lines.append(OCRLine(
                text=out["text"], text_confidence=out["text_confidence"],
                box=[float(v) for v in det.box], polygon=[],
                from_label=det.label))
    return OCRStageResult(lines=lines, engine=description, scope_used=OCR_FROM_BOXES,
                          elapsed_ms=total_ms, boxes_read=read)


def _annotate(image_bgr, lines, dest: Path) -> Optional[str]:
    """Magenta outlines with the text above each - the same visual language the
    standalone OCR tool uses, and distinct from stage 1's green foreground box
    and stage 2's per-class detection colours. Three annotations on one screen
    must not be confusable."""
    from . import ocr_engine

    try:
        import cv2

        dest.parent.mkdir(parents=True, exist_ok=True)
        drawn = ocr_engine.draw_lines(image_bgr, lines)
        return str(dest) if cv2.imwrite(str(dest), drawn) else None
    except Exception:
        # A failed annotation costs a thumbnail, never the read itself.
        return None
