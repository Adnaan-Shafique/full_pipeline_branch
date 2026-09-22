"""Shared box drawing for both detector backends.

The real path draws boxes via Ultralytics' results.plot() or YOLOX's visual();
the stub has to produce output a demo audience cannot tell apart, so both go
through this one function. If the visual language diverges, the stub starts
looking like a different system - which is exactly the impression to avoid.
"""
from __future__ import annotations

import hashlib
from typing import Iterable

# BGR. Distinct at a glance, readable on both bright sky and dark equipment,
# and none of them the green that stage 1 uses for the foreground box - the two
# annotations must never be confused for each other.
_PALETTE = [
    (56, 56, 255),    # red
    (255, 157, 51),   # blue
    (0, 204, 255),    # amber
    (255, 112, 209),  # violet
    (51, 219, 255),   # yellow
    (204, 102, 255),  # pink
]


def color_for(label: str) -> tuple[int, int, int]:
    """Stable colour per class name, so the same label is the same colour on
    every image and across re-runs."""
    digest = hashlib.md5(label.encode("utf-8")).hexdigest()
    return _PALETTE[int(digest[:8], 16) % len(_PALETTE)]


# Mode 3's claimed boxes are drawn DASHED and in two tones, and nothing else in
# this demo is. That is the whole design: stage 1 draws a solid green foreground
# box, stage 2 solid per-class colours, stage 2b magenta text polygons, and a
# viewer who has learned those three reads a dashed box as "different kind of
# thing" before reading the caption. It has to, because the claim behind it is
# different - a model said this is where it is, and nothing measured it.
#
# Two tones rather than one colour because a claimed box lands anywhere: white
# vanishes on a bright sky and black on dark equipment, while alternating dashes
# stay legible on both.
CLAIMED_DASH_PX = 18
CLAIMED_TONE_A = (255, 255, 255)   # white
CLAIMED_TONE_B = (20, 20, 20)      # near-black


def _dashed_line(out, p1, p2, thickness, dash=CLAIMED_DASH_PX):
    """A two-tone dashed segment from p1 to p2."""
    import cv2
    import math

    (x1, y1), (x2, y2) = p1, p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1:
        return
    steps = max(1, int(length // dash))
    for i in range(steps + 1):
        a = i / (steps + 1)
        b = min(1.0, (i + 1) / (steps + 1))
        start = (int(x1 + (x2 - x1) * a), int(y1 + (y2 - y1) * a))
        end = (int(x1 + (x2 - x1) * b), int(y1 + (y2 - y1) * b))
        colour = CLAIMED_TONE_A if i % 2 == 0 else CLAIMED_TONE_B
        cv2.line(out, start, end, colour, thickness, cv2.LINE_AA)


def draw_claimed_boxes(image_bgr, claimed: Iterable, thickness: int | None = None,
                       labels: bool = True):
    """Boxes the MODEL claimed, drawn so they cannot be mistaken for detections.

    Each carries a caption that says who is claiming it and, where the trained
    detector ran on the same photograph, how well the two agree. That second
    part is what makes drawing these defensible: the box arrives with its own
    error bar rather than as an unqualified assertion.

    `labels=False` draws the dashed rectangles alone. The captions are long -
    "model says: a temperature display or device reading (IoU 0.62 vs
    detector)" - and on a tight box around a small object they cover the thing
    being pointed at, which defeats the point of pointing at it. The caption
    text is still on the card either way, so nothing is lost by hiding it here;
    the UI offers box+label, box-only and off.
    """
    import cv2

    out = image_bgr.copy()
    h, w = out.shape[:2]
    t = thickness or max(2, round(min(h, w) / 400))
    font_scale = max(0.5, min(h, w) / 1200)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Caption plates already drawn, so later ones can be pushed clear of them.
    # The presence box and the answer's evidence box usually land on the SAME
    # object - that is the normal, good case - so without this the second
    # caption paints over the first and one of the two claims becomes invisible.
    occupied = []

    for claim in claimed:
        x1, y1, x2, y2 = (int(round(v)) for v in claim.box)
        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
        for p1, p2 in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
                       ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
            _dashed_line(out, p1, p2, t)

        if not labels:
            continue

        caption = f"model says: {claim.label or 'here'}"
        if claim.iou is not None:
            caption += f"  (IoU {claim.iou:.2f} vs detector)"
        (tw, th), baseline = cv2.getTextSize(caption, font, font_scale, max(1, t // 2))
        # INSIDE the box, below its top edge - not above it like every other
        # caption in this demo. draw_detections() puts its class captions above
        # their boxes, and mode 3's claimed box usually sits near the detector's
        # on the same object, so a caption above would land on top of the one
        # naming the detection and hide the provenance both are there to show.
        cap_y = y1 + th + baseline + 2
        if cap_y + baseline > h:
            cap_y = max(th + baseline, y1 - baseline - 2)
        cap_x = min(x1, max(0, w - tw - 6))

        def _overlaps(top, bottom, left, right):
            return any(not (bottom < o_top or top > o_bottom
                            or right < o_left or left > o_right)
                       for o_top, o_bottom, o_left, o_right in occupied)

        step = th + baseline + 4
        for _ in range(6):   # bounded: give up rather than march off the image
            top, bottom = cap_y - th - baseline, cap_y + baseline
            if not _overlaps(top, bottom, cap_x, cap_x + tw + 4) or bottom + step > h:
                break
            cap_y += step
        occupied.append((cap_y - th - baseline, cap_y + baseline,
                         cap_x, cap_x + tw + 4))
        # Solid dark plate behind white text: the caption must stay readable
        # over whatever the box happens to sit on.
        cv2.rectangle(out, (cap_x, cap_y - th - baseline),
                      (cap_x + tw + 4, cap_y + baseline), CLAIMED_TONE_B, -1)
        cv2.putText(out, caption, (cap_x + 2, cap_y), font, font_scale,
                    CLAIMED_TONE_A, max(1, t // 2), cv2.LINE_AA)
    return out


def draw_detections(image_bgr, detections: Iterable, thickness: int | None = None):
    """Rectangle plus a filled '<label> <conf>' caption, matching what the real
    detector's own plotting produces. Returns a copy; never mutates the input,
    which every other stage is still holding."""
    import cv2

    out = image_bgr.copy()
    h, w = out.shape[:2]
    # Scale with the image so a 4000px photo does not get hairline boxes and a
    # 640px one does not get boxes that swallow the subject.
    t = thickness or max(2, round(min(h, w) / 400))
    font_scale = max(0.5, min(h, w) / 1200)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det.box)
        # Clamp: an annotation file can legitimately describe a box that runs
        # off the frame, and cv2 will happily draw off-canvas but the caption
        # would vanish.
        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
        colour = color_for(det.label)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, t)

        caption = f"{det.label} {det.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(caption, font, font_scale, max(1, t // 2))
        # Caption sits above the box, or inside it when the box touches the top.
        cap_y = y1 - baseline - 2
        if cap_y - th < 0:
            cap_y = y1 + th + baseline + 2
        # The real class names are long ("Warning sign (HV / RF radiation)"), so
        # a box near the right edge would push its caption off the frame. Shift
        # it left instead of letting it disappear.
        cap_x = min(x1, max(0, w - tw - 6))
        cv2.rectangle(out, (cap_x, cap_y - th - baseline),
                      (cap_x + tw + 4, cap_y + baseline), colour, -1)
        cv2.putText(out, caption, (cap_x + 2, cap_y), font, font_scale, (255, 255, 255),
                    max(1, t // 2), cv2.LINE_AA)
    return out
