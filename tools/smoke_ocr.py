#!/usr/bin/env python3
"""Stage 2b smoke test - runs OCR and writes the annotated image.

    python tools/smoke_ocr.py /path/to/photos --question temp_within_limit --limit 3

The output image is the point, for the same reason as smoke_stage2.py: the OCR
detector's polygons are drawn on the EXIF-corrected array from
quality_check.load_image_bgr(), and a line box in the wrong place means
something upstream disagrees about rotation.

What to look at, in order:

  1. Did it read the RIGHT text? A whole-frame read on a site photograph picks
     up every label in shot - an IP55 sticker, a manufacturer's plate, a
     warning sign - and only some of that is about the question being asked.
     What ends up in the prompt is everything above the confidence floor.

  2. If the question carries a numeric rule, WHICH TOKEN did it match? The rule
     takes the first number its pattern finds. On a display showing both a
     set-point and a measurement, that is the set-point. On an instrument with
     a range switch in frame, it can be the range. The tool prints the matched
     substring for exactly this reason - see DECISIONS.md on why the rule is
     evidence rather than the answer.

  3. Compare the box-scoped and whole-frame timings. Box crops run with
     Det.limit_type="max" (see ocr_engine._get_engine); without it a ~90px-high
     crop becomes a ~3200x736 detector input and takes ~3.5s instead of ~77ms.
     If the two timings are close on a small crop, that setting has been lost.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "app"))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect(target: Path, limit: int):
    if target.is_file():
        return [target]
    paths = sorted(p for p in target.rglob("*")
                   if p.suffix.lower() in IMAGE_EXTENSIONS)
    return paths[:limit]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", type=Path)
    ap.add_argument("--question", default="temp_within_limit",
                    help="which question's OCR spec to use. Only the questions "
                         "with ocr.enabled do anything; --list shows them")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "demo_runs" / "smoke_ocr")
    ap.add_argument("--scope", choices=("boxes", "image", "boxes_then_image"),
                    default=None, help="override the question's own scope")
    ap.add_argument("--list", action="store_true",
                    help="list the questions that use OCR, and exit")
    args = ap.parse_args()

    from pipeline.config import default_config
    from pipeline.question_types import OCRSpec
    from pipeline.questions import QUESTIONS, get_question, select_relevant
    from pipeline import stage2b_ocr

    if args.list:
        for question in QUESTIONS.values():
            if not question.ocr.enabled:
                continue
            rule = question.ocr.numeric
            extra = (f"  [{rule.label} {rule.comparator} {rule.limit:g} {rule.unit}]"
                     if rule else "")
            print(f"  {question.id:26} scope={question.ocr.scope:18}{extra}")
        return 0

    cfg = default_config()
    available, description = stage2b_ocr.available(cfg)
    print(f"OCR engine: {description}")
    if not available:
        print("\nNothing to smoke - install rapidocr, or put the ONNX files in "
              "models/ocr/.")
        return 1

    question = get_question(args.question)
    if not question.ocr.enabled:
        print(f"\n{question.id!r} does not use OCR. Run with --list to see the "
              f"ones that do.")
        return 1
    if args.scope:
        question = type(question)(**{**question.__dict__,
                                     "ocr": OCRSpec(enabled=True, scope=args.scope,
                                                    numeric=question.ocr.numeric,
                                                    min_confidence=question.ocr.min_confidence)})

    paths = collect(args.target, args.limit)
    if not paths:
        print(f"No images under {args.target}")
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    from quality_check import load_image_bgr
    from pipeline.stage2_detect import get_detector

    detector = get_detector(cfg, question=question)
    print(f"Question:   {question.id} (scope {question.ocr.scope})")
    if question.ocr.numeric:
        rule = question.ocr.numeric
        print(f"Rule:       {rule.label} {rule.comparator} {rule.limit:g} {rule.unit}")
    print(f"Detector:   {detector.name}\n")

    n_read = 0
    for path in paths:
        print(f"{path.name}")
        try:
            image_bgr = load_image_bgr(path)
        except Exception as exc:
            print(f"    could not read the file: {exc}\n")
            continue

        detection = detector.detect(image_bgr, path.stem, image_path=path)
        relevant = select_relevant(detection.detections, question)
        print(f"    detections: {len(relevant)} relevant "
              f"of {len(detection.detections)}")

        result = stage2b_ocr.run(image_bgr, question, relevant, cfg=cfg,
                                 dest_path=args.out / f"{path.stem}.jpg")
        if result.error:
            print(f"    ERROR {result.error}\n")
            continue

        print(f"    {result.headline}")
        for line in result.lines:
            source = f"  (from {line.from_label})" if line.from_label else ""
            print(f"      {line.text_confidence:.2f}  {line.text!r}{source}")
        if result.lines:
            n_read += 1
        if result.numeric:
            # The matched substring, not just the value: on a display showing a
            # set-point and a measurement, seeing WHICH token was taken is the
            # whole reason to run this tool rather than trust the number.
            print(f"    rule matched {result.numeric['matched']!r} -> "
                  f"{result.numeric['value']:g} "
                  f"({'within' if result.numeric['passes'] else 'OUTSIDE'} the limit)")
        elif question.ocr.numeric:
            print(f"    rule matched nothing in that text")
        if result.note:
            print(f"    note: {result.note}")
        if result.annotated_path:
            print(f"    -> {result.annotated_path}")
        print()

    print(f"{n_read}/{len(paths)} image(s) yielded text.")
    if n_read:
        print(f"\nNow OPEN the images in {args.out}. Confirm each magenta box sits "
              f"on the text it claims, and that the text driving the answer is "
              f"the text the QUESTION is about - a whole-frame read picks up "
              f"every label in shot, and all of it above the confidence floor "
              f"goes into the prompt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
