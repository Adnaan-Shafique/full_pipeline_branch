"""Turning a coordinate the model wrote into a box we are willing to draw.

Mode 3 asks Qwen3-VL where things are. This module is everything between that
answer and a rectangle on screen, and it exists because the gap is wider than
it looks.

── The trap this module was written for ─────────────────────────────────────
`array_to_data_uri()` DOWNSCALES the photograph to MAX_UPLOAD_SIDE_PX (2048)
before sending it. The model therefore never sees the original frame. If it
replies with absolute pixel coordinates, they are in the RESIZED frame — and
drawing them on the 4000x3000 original puts every box out by the scale factor,
about 2x, with nothing raising. That is the same class of bug as this repo's
"arrays, never paths" rule, one stage further on.

Two defences, in this order:

  1. **Ask for normalized 0-1000.** It is the convention the Qwen-VL family was
     trained to emit, and being resolution-independent it sidesteps the resize
     entirely. This is the path we expect to take.
  2. **Accept the other conventions anyway, and say which one was used.** A
     model that ignores the instruction should not silently produce garbage
     geometry. `interpret_box()` reports the convention it chose, the UI prints
     it, and a systematic misread is then visible as "every box on every photo
     came back `absolute_px`" rather than as boxes that are quietly wrong.

── What this module will not do ─────────────────────────────────────────────
It will not guess. A box it cannot place confidently is rejected with a reason,
and the card says the model claimed a region that could not be placed — which
is honest, where a plausible-looking rectangle in the wrong place is not.

Pure stdlib. No cv2, no numpy.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

# The convention we ask for, and the one we hope to see. Qwen-VL was trained on
# integers in this range; it is independent of the resize, so a box in this
# convention is correct no matter what MAX_UPLOAD_SIDE_PX is set to.
NORMALIZED_SCALE = 1000

CONV_NORMALIZED_1000 = "normalized_1000"
CONV_FRACTION = "fraction_0_1"
CONV_ABSOLUTE_ENCODED = "absolute_px_encoded_frame"
CONV_ABSOLUTE_ORIGINAL = "absolute_px_original_frame"

# A box covering essentially the whole frame is not a localisation — it is the
# model declining to localise while appearing to comply. Drawing it implies a
# precision that was never claimed, so it is rejected with that reason.
MAX_AREA_FRAC = 0.98
# Below this it is almost certainly a coordinate-order mistake or a stray token
# rather than an object in a site photograph.
MIN_AREA_FRAC = 0.0002


@dataclass
class GroundingBox:
    """A region the MODEL says is relevant. Never a measurement.

    `box` is always in ORIGINAL-frame pixels, whatever convention arrived, so
    every renderer downstream can treat it like any other box. `convention` and
    `raw` are kept so a human can audit how it got there — which is the whole
    point of drawing it at all.
    """

    box: list                     # x1, y1, x2, y2 in the ORIGINAL frame
    label: str = ""               # what the model said is there
    convention: str = ""          # which reading of the numbers was used
    raw: list = field(default_factory=list)   # the numbers as the model wrote them
    leg: str = ""                 # "presence" | "answer"
    # Set by compare_to_detections(): how well this agrees with the trained
    # detector on the SAME photograph. None when nothing was available to
    # compare against, which is the normal case for the Infra questions.
    iou: Optional[float] = None
    matched_label: Optional[str] = None

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def agreement(self) -> str:
        """How the IoU should be described on a card, in words rather than as a
        bare number. The bands are deliberately coarse: this is a demo claim
        about grounding quality, not a benchmark, and three digits of IoU would
        imply a precision the comparison does not have."""
        if self.iou is None:
            return "no trained detector box to compare against"
        if self.iou >= 0.7:
            return f"closely matches the detector (IoU {self.iou:.2f})"
        if self.iou >= 0.4:
            return f"overlaps the detector (IoU {self.iou:.2f})"
        if self.iou > 0.0:
            return f"barely overlaps the detector (IoU {self.iou:.2f})"
        return "does not overlap the detector's box at all (IoU 0.00)"


@dataclass
class GroundingResult:
    """Every box one leg claimed, plus why any were thrown away."""

    boxes: list = field(default_factory=list)
    note: str = ""            # shown on the card when something was rejected
    attempted: bool = False   # did we actually ask this leg for coordinates?

    @property
    def ok(self) -> bool:
        return bool(self.boxes)


# ─────────────────────────────── Extraction ──────────────────────────────────

# Matches a bare 4-number list anywhere in the text, as a fallback for a model
# that wrote prose around its JSON. Deliberately does NOT match 2- or 8-number
# lists: a polygon or a point is not a box, and coercing one into a box is how
# a wrong rectangle gets on screen.
_QUAD_RE = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
    r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")


def _loads(text: str) -> Optional[dict]:
    """The same tolerant JSON read the other parsers use: a fenced block, or the
    first {...} span in a reply that wrapped prose around it."""
    raw = (text or "").strip()
    if not raw:
        return None
    fenced = re.search(r"```(?:json)?\s*(.+?)```", raw, re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(raw[start:end + 1])
            return data if isinstance(data, dict) else None
        except Exception:
            return None
    return None


def extract_raw_boxes(text: str) -> list:
    """Every 4-number coordinate list in a leg's reply, with its label if it had
    one. Returns [(numbers, label), ...] — still in whatever convention the
    model used; interpret_box() decides what they mean.

    Reads the JSON contract first (`box`, or `boxes` as a list of objects or of
    bare lists), and only then falls back to scraping. The scrape exists because
    a model that adds a sentence before its JSON is common and recoverable; it
    is not a licence to invent structure that was not there.
    """
    out = []
    data = _loads(text)
    if data:
        for key in ("box", "bbox", "bbox_2d", "region", "evidence_box"):
            value = data.get(key)
            if isinstance(value, (list, tuple)) and len(value) == 4:
                out.append(([_num(v) for v in value], str(data.get("label", "")).strip()))
                break
        for key in ("boxes", "bboxes", "regions"):
            value = data.get(key)
            if not isinstance(value, (list, tuple)):
                continue
            for entry in value:
                if isinstance(entry, dict):
                    for bkey in ("box", "bbox", "bbox_2d", "region"):
                        inner = entry.get(bkey)
                        if isinstance(inner, (list, tuple)) and len(inner) == 4:
                            out.append(([_num(v) for v in inner],
                                        str(entry.get("label", "")).strip()))
                            break
                elif isinstance(entry, (list, tuple)) and len(entry) == 4:
                    out.append(([_num(v) for v in entry], ""))
            break
    if out:
        return [(nums, label) for nums, label in out if all(n is not None for n in nums)]

    return [([float(g) for g in m.groups()], "") for m in _QUAD_RE.finditer(text or "")]


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ─────────────────────────────── Interpretation ──────────────────────────────

def interpret_box(numbers, original_wh, encoded_wh=None):
    """Place four numbers in the ORIGINAL frame, or refuse and say why.

    Returns (box, convention, None) or (None, None, reason).

    `encoded_wh` is the size the model actually saw — the downscaled frame from
    array_to_data_uri(). It is what absolute coordinates would be relative to,
    and omitting it means an absolute reply cannot be scaled and is rejected
    rather than drawn in the wrong place.
    """
    if not numbers or len(numbers) != 4 or any(n is None for n in numbers):
        return None, None, "not four numbers"
    x1, y1, x2, y2 = (float(n) for n in numbers)

    # A model that swaps corners is writing a valid box badly, not a different
    # box — normalise rather than reject. A model that writes a zero-area box
    # is saying nothing, and that IS a rejection.
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    if x2 <= x1 or y2 <= y1:
        return None, None, "degenerate box (zero width or height)"
    if min(x1, y1) < 0:
        return None, None, "negative coordinate"

    width, height = float(original_wh[0]), float(original_wh[1])
    if width <= 0 or height <= 0:
        return None, None, "unknown frame size"
    largest = max(x2, y2)

    if largest <= 1.0:
        # Fractions of the frame. Unambiguous: no real pixel box on a site
        # photograph has its far corner inside the first pixel.
        scaled = [x1 * width, y1 * height, x2 * width, y2 * height]
        convention = CONV_FRACTION
    elif largest <= NORMALIZED_SCALE:
        # The convention we asked for. Note this range also covers an absolute
        # box on a small image, which is why the two cannot be told apart by the
        # numbers alone — see the note below on why that is acceptable.
        scaled = [x1 / NORMALIZED_SCALE * width, y1 / NORMALIZED_SCALE * height,
                  x2 / NORMALIZED_SCALE * width, y2 / NORMALIZED_SCALE * height]
        convention = CONV_NORMALIZED_1000
    else:
        # Over 1000, so it can only be absolute pixels. Which frame? The model
        # saw the ENCODED one, so that is what we scale from. Without knowing
        # that size we would be guessing at a ~2x error, so refuse instead.
        if not encoded_wh or not encoded_wh[0] or not encoded_wh[1]:
            return None, None, ("absolute pixel coordinates, but the size of the "
                                "image the model saw is unknown, so they cannot "
                                "be placed in the original frame")
        enc_w, enc_h = float(encoded_wh[0]), float(encoded_wh[1])
        if largest > max(enc_w, enc_h) * 1.02:
            # Beyond even the encoded frame. Possibly the original frame's
            # coordinates from a model that somehow knew them, but far more
            # likely a hallucinated number - and the two are indistinguishable.
            if largest <= max(width, height) * 1.02:
                scaled = [x1, y1, x2, y2]
                convention = CONV_ABSOLUTE_ORIGINAL
            else:
                return None, None, (f"coordinates run past the frame the model "
                                    f"saw ({int(enc_w)}x{int(enc_h)})")
        else:
            scaled = [x1 / enc_w * width, y1 / enc_h * height,
                      x2 / enc_w * width, y2 / enc_h * height]
            convention = CONV_ABSOLUTE_ENCODED

    # Clamp to the frame. A box that runs a little past the edge is a model
    # rounding outward, not a wrong box; one that starts outside entirely was
    # already rejected above.
    box = [max(0.0, min(scaled[0], width - 1)), max(0.0, min(scaled[1], height - 1)),
           max(0.0, min(scaled[2], width)), max(0.0, min(scaled[3], height))]
    if box[2] <= box[0] or box[3] <= box[1]:
        return None, None, "box falls outside the frame once placed"

    frac = ((box[2] - box[0]) * (box[3] - box[1])) / (width * height)
    if frac >= MAX_AREA_FRAC:
        return None, None, (f"covers {frac * 100:.0f}% of the frame - that is the "
                            f"model declining to localise, not a location")
    if frac < MIN_AREA_FRAC:
        return None, None, f"covers only {frac * 100:.3f}% of the frame - too small to be real"
    return box, convention, None


def boxes_from_text(text, original_wh, encoded_wh=None, leg="", default_label=""):
    """A leg's whole reply -> GroundingResult. Never raises."""
    result = GroundingResult(attempted=True)
    raw_boxes = extract_raw_boxes(text)
    if not raw_boxes:
        result.note = "the model returned no coordinates"
        return result

    rejected = []
    for numbers, label in raw_boxes:
        box, convention, reason = interpret_box(numbers, original_wh, encoded_wh)
        if box is None:
            rejected.append(reason)
            continue
        result.boxes.append(GroundingBox(
            box=box, label=(label or default_label), convention=convention,
            raw=[round(float(n), 2) for n in numbers], leg=leg))

    if rejected and not result.boxes:
        result.note = ("the model claimed a region that could not be placed: "
                       + "; ".join(dict.fromkeys(rejected)))
    elif rejected:
        result.note = (f"{len(rejected)} of {len(raw_boxes)} claimed regions were "
                       f"discarded: " + "; ".join(dict.fromkeys(rejected)))
    return result


# ─────────────────────────────── Scoring ─────────────────────────────────────

def iou(box_a, box_b) -> float:
    """Intersection over union of two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compare_to_detections(result: GroundingResult, detections) -> GroundingResult:
    """Score each claimed box against the trained detector's boxes, in place.

    This is what makes drawing mode 3's boxes defensible rather than reckless.
    The original objection to them was that the model's grounding is well below
    YOLOX's — true, and previously asserted rather than shown. Modes 1 and 2
    have ALREADY run the detector over this exact photograph in the same frame,
    so the number is free, and the claim becomes measurable.

    Mode 3 does NOT consume this. The detections arrive after its three legs
    have answered, are used only to score what the model already said, and
    never reach a prompt. If that ever stops being true, mode 3 stops being
    "everything by the model" and the comparison stops meaning anything.
    """
    dets = [d for d in (detections or []) if getattr(d, "box", None)]
    if not dets:
        return result
    for claimed in result.boxes:
        best, best_label = 0.0, None
        for det in dets:
            score = iou(claimed.box, det.box)
            if score > best:
                best, best_label = score, det.label
        claimed.iou = best
        claimed.matched_label = best_label
    return result


def summarise(result: GroundingResult) -> str:
    """One line for a card or a CSV: how many regions, and how well they agree."""
    if not result.attempted:
        return "grounding was not requested"
    if not result.boxes:
        return result.note or "the model returned no coordinates"
    scored = [b for b in result.boxes if b.iou is not None]
    head = f"{len(result.boxes)} region{'s' if len(result.boxes) != 1 else ''} claimed"
    if not scored:
        return head + " (no trained detector box to compare against)"
    best = max(b.iou for b in scored)
    return f"{head}, best agreement with the detector IoU {best:.2f}"
