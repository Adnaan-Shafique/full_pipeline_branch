"""OCR on detected bounding boxes and whole images - PP-OCRv6 (small or tiny det +
rec) + the shared PP-OCR angle classifier, run through RapidOCR/onnxruntime from
the ONNX files in models/ocr/.

The exact same ONNX files are what gets handed to the Android team, so what this
demo reads is what the edge device will read. Fully offline: explicit model
paths, so RapidOCR never reaches for its downloader.

── Differences from the standalone ocr.py this came from ────────────────────
Three, all of them about surviving a host that is missing something:

  1. `rapidocr` is imported inside _get_engine(), never at module scope, and
     every entry point returns an empty read with a named error rather than
     raising. The rest of the pipeline already works this way - a missing
     dependency must degrade one stage, not take the import down and with it
     the whole test suite. available_variants() is the honest check.
  2. The angle classifier is OPTIONAL. Only the det and rec ONNX files for the
     tiny variant shipped; ch_ppocr_mobile_v2.0_cls_mobile.onnx did not. Rather
     than pass a path that does not exist - which makes RapidOCR fall back to
     its downloader, and on a demo host with no network that is a hang, not an
     error - the Cls path is simply omitted when the file is absent. Text
     rotated 180 degrees reads worse without it; nothing else changes.
  3. The model directory is a parameter, defaulting to the same derived path,
     so it can be pointed at a config's models_dir like every other stage.
"""
from __future__ import annotations

import time
from pathlib import Path

# app/pipeline/ocr_engine.py -> app/pipeline -> app -> <project root>
OCR_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "ocr"
CLS_MODEL = "ch_ppocr_mobile_v2.0_cls_mobile.onnx"  # shared by every variant
OCR_VARIANTS = {  # insertion order = display order
    "small": {"label": "PP-OCRv6 small", "det": "PP-OCRv6_det_small.onnx", "rec": "PP-OCRv6_rec_small.onnx"},
    "tiny": {"label": "PP-OCRv6 tiny", "det": "PP-OCRv6_det_tiny.onnx", "rec": "PP-OCRv6_rec_tiny.onnx"},
}
# "tiny" rather than "small": only the tiny pair is committed under models/ocr/.
# default_variant() below falls through to whatever is actually present, so
# dropping the small pair in makes it available without a code change - but the
# DEFAULT stays what the repo can actually run today.
DEFAULT_VARIANT = "tiny"

CROP_PAD_FRAC = 0.08   # context around the box - detector boxes are often a hair tight on text
MIN_CROP_SIDE = 48     # upscale tiny crops so the text detector has something to find
CROP_DET_MAX_SIDE = 960  # detector input cap for crops (see _get_engine)

_ENGINES = {}


def available_variants(model_dir=None):
    """Variant keys whose ONNX files are actually present in models/ocr/."""
    model_dir = Path(model_dir or OCR_MODEL_DIR)
    return [k for k, v in OCR_VARIANTS.items()
            if (model_dir / v["det"]).exists() and (model_dir / v["rec"]).exists()]


def default_variant(model_dir=None):
    """DEFAULT_VARIANT when its files are present, else the first variant that
    is, else None. Returning None is what lets the stage say "no OCR models
    found" instead of failing inside RapidOCR with a path error."""
    present = available_variants(model_dir)
    if DEFAULT_VARIANT in present:
        return DEFAULT_VARIANT
    return present[0] if present else None


def angle_classifier_available(model_dir=None) -> bool:
    return (Path(model_dir or OCR_MODEL_DIR) / CLS_MODEL).exists()


def describe(model_dir=None) -> str:
    """One line for the UI's status row: which variant runs, and whether the
    angle classifier is behind it."""
    model_dir = Path(model_dir or OCR_MODEL_DIR)
    variant = default_variant(model_dir)
    if variant is None:
        return f"no OCR models found in {model_dir}"
    label = OCR_VARIANTS[variant]["label"]
    if angle_classifier_available(model_dir):
        return f"{label} + angle classifier"
    return f"{label} (no angle classifier - rotated text reads worse)"


def _get_engine(variant, for_crop=False, model_dir=None):
    """One engine per (variant, mode) over the same ONNX files. Box crops use
    Det.limit_type="max": RapidOCR's default ("min", 736) upscales any image whose short
    side is under 736px, which turns a ~90px-high crop into a ~3200x736 detector input -
    measured ~3.5 s/crop vs ~77 ms with "max" (and 4/4 vs 3/4 exact reads on a small
    synthetic set). Whole images keep the defaults. The Android port must use the same
    detector setting for crops."""
    model_dir = Path(model_dir or OCR_MODEL_DIR)
    key = (str(model_dir), variant, for_crop)
    if key not in _ENGINES:
        from rapidocr import RapidOCR
        spec = OCR_VARIANTS[variant]
        params = {
            "Det.model_path": str(model_dir / spec["det"]),
            "Rec.model_path": str(model_dir / spec["rec"]),
            "Global.log_level": "warning",
        }
        # Absent on this checkout. Passing a non-existent path is worse than
        # omitting the key: RapidOCR treats it as "go and fetch one", and on an
        # offline demo host that is a hang rather than an error.
        if angle_classifier_available(model_dir):
            params["Cls.model_path"] = str(model_dir / CLS_MODEL)
        if for_crop:
            params.update({"Det.limit_type": "max", "Det.limit_side_len": CROP_DET_MAX_SIDE})
        _ENGINES[key] = RapidOCR(params=params)
    return _ENGINES[key]


def crop_box(image_bgr, box, pad_frac=CROP_PAD_FRAC):
    """box: [x0, y0, x1, y1] in image pixels. Returns the padded, bounds-clamped crop,
    or None if it has no area."""
    import cv2

    h, w = image_bgr.shape[:2]
    x0, y0, x1, y1 = box
    px, py = (x1 - x0) * pad_frac, (y1 - y0) * pad_frac
    x0, y0 = max(0, int(x0 - px)), max(0, int(y0 - py))
    x1, y1 = min(w, int(x1 + px)), min(h, int(y1 + py))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    crop = image_bgr[y0:y1, x0:x1]
    short = min(crop.shape[:2])
    if short < MIN_CROP_SIDE:
        scale = MIN_CROP_SIDE / short
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop


def read_image(image_bgr, variant=None, model_dir=None):
    """OCR the whole image, independent of any detector. Returns (lines, ocr_ms): one dict
    per text line, top-to-bottom: {"text", "text_confidence", "box": [x0, y0, x1, y1],
    "polygon": [[x, y]*4]}, and the wall-clock OCR time in milliseconds (engine load
    excluded, so a first call isn't inflated by one-off model loading)."""
    variant = variant or default_variant(model_dir)
    engine = _get_engine(variant, model_dir=model_dir)
    t0 = time.perf_counter()
    result = engine(image_bgr)
    ocr_ms = (time.perf_counter() - t0) * 1000
    if result.txts is None or len(result.txts) == 0:
        return [], ocr_ms
    lines = []
    for poly, text, score in zip(result.boxes, result.txts, result.scores):
        xs, ys = poly[:, 0], poly[:, 1]
        lines.append({
            "text": text,
            "text_confidence": float(score),
            "box": [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
            "polygon": poly.astype(int).tolist(),
        })
    return lines, ocr_ms


def draw_lines(image_bgr, lines):
    """Copy of image_bgr with each recognised line outlined and its text drawn above it
    (ASCII only - cv2.putText limit; the full strings are in `lines`)."""
    import cv2
    import numpy as np

    out = image_bgr.copy()
    for line in lines:
        polygon = line["polygon"] if isinstance(line, dict) else line.polygon
        text_raw = line["text"] if isinstance(line, dict) else line.text
        box = line["box"] if isinstance(line, dict) else line.box
        if not polygon:
            # A box-scoped read carries a rectangle but no detector polygon.
            x0, y0, x1, y1 = box
            polygon = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, (255, 0, 255), 2)
        text = text_raw.encode("ascii", "replace").decode()
        org = (max(0, int(box[0])), max(14, int(box[1]) - 6))
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def read_box(image_bgr, box, variant=None, model_dir=None):
    """OCR one detected box. Returns {"text": str, "text_confidence": float, "ocr_ms": float};
    empty text and 0.0 when nothing legible was found. Multiple text lines are joined
    top-to-bottom with a space; confidence is the mean over lines. ocr_ms is the wall-clock
    cost of crop + upscale + OCR for this box (engine load excluded)."""
    variant = variant or default_variant(model_dir)
    engine = _get_engine(variant, for_crop=True, model_dir=model_dir)
    t0 = time.perf_counter()
    crop = crop_box(image_bgr, box)
    if crop is None:
        return {"text": "", "text_confidence": 0.0, "ocr_ms": (time.perf_counter() - t0) * 1000}
    result = engine(crop)
    ocr_ms = (time.perf_counter() - t0) * 1000
    if result.txts is None or len(result.txts) == 0:
        return {"text": "", "text_confidence": 0.0, "ocr_ms": ocr_ms}
    scores = [float(s) for s in result.scores]
    return {"text": " ".join(result.txts), "text_confidence": sum(scores) / len(scores),
            "ocr_ms": ocr_ms}


def preload(variant=None, model_dir=None) -> None:
    """Build both engines now, so the first photograph of a demo does not pay
    the one-off model load in front of an audience. Mirrors
    stage1_quality.preload_segmenter()."""
    variant = variant or default_variant(model_dir)
    if variant is None:
        return
    _get_engine(variant, for_crop=False, model_dir=model_dir)
    _get_engine(variant, for_crop=True, model_dir=model_dir)
