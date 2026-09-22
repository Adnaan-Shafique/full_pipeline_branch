# `models/ocr/` — PP-OCRv6 for stage 2b

Committed on purpose, for a stronger version of the reason `u2netp.onnx` is:
RapidOCR's response to a model path that does not exist is to **download** one.
On a demo host with no route out, that is a hang in front of an audience rather
than an error anyone can read. `ocr_engine._get_engine()` therefore passes
explicit paths and never lets the downloader be reached.

These are the same ONNX files handed to the Android team, so what this demo
reads is what the edge device will read.

## What is here

| File | Bytes | sha256 |
|---|---|---|
| `PP-OCRv6_det_tiny.onnx` | 1,829,618 | `f42c0fbd294d95eac1a550e131b277dac97462c8025fa4b6c3cec1b7894bd3d5` |
| `PP-OCRv6_rec_tiny.onnx` | 4,489,813 | `e16e242de5937ad92609223f19bc2aff3727ee40b095f996907c24749bad251b` |

## What is NOT here, and what that costs

**`ch_ppocr_mobile_v2.0_cls_mobile.onnx` — the angle classifier.** Shared by
every variant, and absent from the archive these files came in.
`ocr_engine._get_engine()` omits the `Cls.model_path` key entirely rather than
passing a path that does not exist, because a bad path is what wakes the
downloader. The cost is text rotated 180° reading worse; upright text is
unaffected. Drop the file in and it is picked up with no code change —
`angle_classifier_available()` checks for it on every engine build, and
`describe()` stops printing the caveat.

**The `small` variant** (`PP-OCRv6_det_small.onnx`, `PP-OCRv6_rec_small.onnx`).
`OCR_VARIANTS` still lists it, `available_variants()` reports it as absent, and
`DEFAULT_VARIANT` is `"tiny"` because tiny is what the repo can actually run.
Drop both files in and `small` becomes selectable; change `DEFAULT_VARIANT` to
make it the default.

## Checking them on the demo host

```bash
sha256sum models/ocr/*.onnx
python -c "import sys; sys.path.insert(0,'app'); from pipeline.ocr_engine import describe; print(describe())"
```

The second prints the variant that will actually run and whether the angle
classifier is behind it. `tools/preflight.py` runs the same check.
