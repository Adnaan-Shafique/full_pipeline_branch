"""Mode 3's claimed boxes: reading coordinates, refusing bad ones, scoring them.

The coordinate handling is the whole risk here. A box that is merely WRONG
still looks authoritative on a photograph, so most of what follows is about the
cases where this module must refuse rather than draw.
"""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.modules.setdefault("cv2", types.ModuleType("cv2"))

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok   {name}")
    else:
        failed += 1
        print(f"  FAIL {name}" + (f"\n         {detail}" if detail else ""))


from pipeline import vlm_grounding as G                                # noqa: E402
from pipeline.schemas import Detection, SOURCE_MODEL                   # noqa: E402

# A 4000x3000 original, encoded down to 2048x1536 before the model ever saw it.
# Every number below is chosen so the three conventions describe the SAME
# region, which is what makes the comparisons meaningful.
ORIG = (4000, 3000)
ENC = (2048, 1536)


def box_of(text, orig=ORIG, enc=ENC):
    r = G.boxes_from_text(text, orig, enc, leg="presence")
    return (r.boxes[0] if r.boxes else None), r


print("\nthe three conventions all land on the same region")
same = {
    "normalized 0-1000": '{"box": [250, 300, 500, 600]}',
    "fraction 0-1": '{"box": [0.25, 0.30, 0.50, 0.60]}',
    "absolute px in the ENCODED frame": '{"box": [512, 460.8, 1024, 921.6]}',
}
placed = {}
for name, text in same.items():
    b, _ = box_of(text)
    check(f"{name} is read", b is not None, text)
    if b:
        placed[name] = [round(v) for v in b.box]
check("all three resolve to the same original-frame box",
      len({tuple(v) for v in placed.values()}) == 1, str(placed))
check("and that box is where it should be",
      placed.get("normalized 0-1000") == [1000, 900, 2000, 1800],
      str(placed.get("normalized 0-1000")))
b, _ = box_of(same["normalized 0-1000"])
check("the convention used is reported, not hidden",
      b.convention == G.CONV_NORMALIZED_1000, b.convention)
check("and the raw numbers are kept for audit", b.raw == [250.0, 300.0, 500.0, 600.0],
      str(b.raw))

print("\nthe resize trap: absolute coordinates are in the frame the MODEL saw")
# This is the bug the module exists for. array_to_data_uri downscales to 2048,
# so [512,460,1024,921] means the middle of a 2048-wide image. Read naively as
# original-frame pixels it would land in the top-left QUARTER of a 4000px photo.
b, _ = box_of('{"box": [512, 460.8, 1024, 921.6]}')
check("an absolute box is scaled up from the encoded frame",
      b is not None and abs(b.box[0] - 1000) < 2, str(b.box if b else None))
check("and is labelled as having been read that way",
      b.convention == G.CONV_ABSOLUTE_ENCODED, b.convention)
naive = [512, 460.8, 1024, 921.6]
check("which is nowhere near the naive reading - hence the module",
      abs(b.box[0] - naive[0]) > 400, f"{b.box[0]} vs {naive[0]}")
# Without knowing the encoded size, an absolute reply CANNOT be placed. Refuse.
b, r = box_of('{"box": [512, 460, 1024, 921]}', enc=None)
check("with no encoded size, an absolute box is refused, not guessed",
      b is None, str(b.box if b else None))
check("and the refusal names the reason",
      "cannot be placed" in r.note or "unknown" in r.note, r.note)

print("\nrefusing what must not be drawn")
for name, text, fragment in [
    ("the whole frame", '{"box": [0, 0, 1000, 1000]}', "declining to localise"),
    ("a near-whole frame", '{"box": [1, 1, 999, 999]}', "declining to localise"),
    ("zero height", '{"box": [100, 100, 400, 100]}', "degenerate"),
    ("zero width", '{"box": [100, 100, 100, 400]}', "degenerate"),
    ("a negative corner", '{"box": [-5, 10, 100, 200]}', "negative"),
    ("a speck", '{"box": [500, 500, 501, 501]}', "too small"),
    ("three numbers", '{"box": [1, 2, 3]}', "no coordinates"),
    ("no coordinates at all", '{"present": "yes"}', "no coordinates"),
    ("empty text", "", "no coordinates"),
]:
    b, r = box_of(text)
    check(f"{name} is refused", b is None, str(b.box if b else None))
    check(f"  ...and says why: {fragment}", fragment in r.note, r.note)

print("\nrecovering what is merely written badly")
b, _ = box_of('{"box": [500, 600, 250, 300]}')
check("swapped corners are normalised, not rejected - it is a valid region "
      "written badly", b is not None and b.box[0] < b.box[2], str(b.box if b else None))
b, _ = box_of('Certainly! Here is the answer: {"present":"yes","box":[250,300,500,600]}')
check("prose wrapped around the JSON still parses", b is not None)
b, _ = box_of('```json\n{"box": [250, 300, 500, 600]}\n```')
check("a markdown fence still parses", b is not None)
b, _ = box_of('The sign sits at [250, 300, 500, 600] in the frame.')
check("a bare coordinate list is scraped as a fallback", b is not None)
b, _ = box_of('{"box": [1010, 100, 1200, 400]}')
check("just over 1000 is treated as encoded pixels, not normalized",
      b is not None and b.convention == G.CONV_ABSOLUTE_ENCODED,
      b.convention if b else "refused")
b, _ = box_of('{"box": [9000, 9000, 9500, 9500]}')
check("coordinates past even the original frame are refused", b is None)

print("\na polygon or a point is not a box")
# Coercing either into a rectangle is how a confident wrong box reaches a card.
_, r = box_of('{"box": [100, 200]}')
check("a two-number point yields nothing", not r.boxes)
_, r = box_of('{"polygon": [10, 20, 30, 40, 50, 60, 70, 80]}')
check("an eight-number polygon under an unknown key yields nothing", not r.boxes)

print("\nmultiple regions, and partial rejection")
r = G.boxes_from_text(
    '{"boxes": [{"box": [100, 100, 300, 300], "label": "sign"},'
    ' {"box": [0, 0, 1000, 1000], "label": "everything"}]}', ORIG, ENC)
check("the good one survives", len(r.boxes) == 1 and r.boxes[0].label == "sign",
      str([b.label for b in r.boxes]))
check("and the discard is reported rather than silent",
      "discarded" in r.note, r.note)
r = G.boxes_from_text('{"boxes": [[100, 100, 300, 300], [400, 400, 600, 600]]}',
                      ORIG, ENC)
check("a bare list of lists is read too", len(r.boxes) == 2)

print("\nscoring against the trained detector")
det = [Detection(label="GPS Antenna", confidence=0.9,
                 box=[1000, 900, 2000, 1800], source=SOURCE_MODEL)]
r = G.boxes_from_text('{"box": [250, 300, 500, 600]}', ORIG, ENC)
G.compare_to_detections(r, det)
check("an exact match scores 1.0", abs(r.boxes[0].iou - 1.0) < 1e-6,
      str(r.boxes[0].iou))
check("and is described in words, not just a number",
      "closely matches" in r.boxes[0].agreement, r.boxes[0].agreement)
check("and names which detection it matched",
      r.boxes[0].matched_label == "GPS Antenna")

r = G.boxes_from_text('{"box": [600, 600, 800, 800]}', ORIG, ENC)
G.compare_to_detections(r, det)
check("a disjoint box scores 0.0", r.boxes[0].iou == 0.0, str(r.boxes[0].iou))
check("and says so plainly rather than staying quiet",
      "does not overlap" in r.boxes[0].agreement, r.boxes[0].agreement)

# The distinction that matters most on a card: "we could not measure" is not
# "it scored zero". Every Infra question is in the first case today.
r = G.boxes_from_text('{"box": [250, 300, 500, 600]}', ORIG, ENC)
G.compare_to_detections(r, [])
check("with no detector boxes, IoU stays None - never 0.0",
      r.boxes[0].iou is None, str(r.boxes[0].iou))
check("and the wording says there was nothing to compare against",
      "no trained detector box" in r.boxes[0].agreement, r.boxes[0].agreement)

print("\nIoU itself")
check("identical boxes", abs(G.iou([0, 0, 10, 10], [0, 0, 10, 10]) - 1.0) < 1e-9)
check("disjoint boxes", G.iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0)
check("touching edges do not overlap", G.iou([0, 0, 10, 10], [10, 0, 20, 10]) == 0.0)
check("half overlap", abs(G.iou([0, 0, 10, 10], [5, 0, 15, 10]) - (50 / 150)) < 1e-9)
check("a degenerate box scores 0 rather than dividing by zero",
      G.iou([0, 0, 0, 0], [0, 0, 10, 10]) == 0.0)

print("\nthe summary line")
r = G.GroundingResult()
check("nothing requested says so", "not requested" in G.summarise(r))
r = G.GroundingResult(attempted=True, note="the model returned no coordinates")
check("asked but empty repeats the note", "no coordinates" in G.summarise(r))
r = G.boxes_from_text('{"box": [250, 300, 500, 600]}', ORIG, ENC)
check("unscored boxes say there was nothing to compare with",
      "no trained detector" in G.summarise(r), G.summarise(r))
G.compare_to_detections(r, det)
check("scored boxes quote the best agreement", "IoU 1.00" in G.summarise(r),
      G.summarise(r))

print("\nthe prompts that ask for all this")
import pipeline.questions as pq  # noqa: E402

q = pq.get_question("gps_antenna")
plain_p = pq.render_leg_user(q, pq.LEG_PRESENCE)
ground_p = pq.render_leg_user(q, pq.LEG_PRESENCE, grounding=True)
check("grounding is OFF by default, so every existing caller is unaffected",
      plain_p == pq.PRESENCE_USER_TEMPLATE.format(subject=q.effective_subject,
                                                  grounding_rule=pq.GROUNDING_RULE))
check("the ungrounded presence leg asks for no coordinates",
      "box" not in plain_p.lower().replace("boxes", ""), plain_p)
check("the grounded one asks for the 0-1000 convention",
      "0 to 1000" in ground_p, ground_p[:200])
# The cheapest way to comply with "give me a box" is to return the frame, and
# the cheapest way to avoid admitting ignorance is to invent one. Both are
# addressed in the prompt, not only in the parser.
check("it tells the model to box tightly, not the whole image",
      "not the whole image" in ground_p)
check("and gives it a way to decline", "omit the box entirely" in ground_p)
check("the grounded answer leg asks for an evidence region",
      "evidence_box" in pq.render_leg_user(q, pq.LEG_ANSWER, grounding=True))
check("the QUALITY leg is never grounded - it judges a whole-frame property",
      pq.render_leg_user(q, pq.LEG_QUALITY) ==
      pq.render_leg_user(q, pq.LEG_QUALITY, grounding=True))
check("and it is not in the grounded set", pq.LEG_QUALITY not in pq.GROUNDED_LEGS)
check("both other legs are", set(pq.GROUNDED_LEGS) == {pq.LEG_PRESENCE, pq.LEG_ANSWER})

print("\nthe encoded size this all depends on")
from pipeline.stage3_vlm import MAX_UPLOAD_SIDE_PX, encoded_size  # noqa: E402


class FakeArr:
    def __init__(self, h, w):
        self.shape = (h, w, 3)


check("a large photo is reported at the size it will be sent",
      encoded_size(FakeArr(3000, 4000)) == (2048, 1536),
      str(encoded_size(FakeArr(3000, 4000))))
check("a portrait photo too", encoded_size(FakeArr(4000, 3000)) == (1536, 2048),
      str(encoded_size(FakeArr(4000, 3000))))
check("a small photo is sent unchanged", encoded_size(FakeArr(600, 800)) == (800, 600))
check("the cap matches what array_to_data_uri uses", MAX_UPLOAD_SIDE_PX == 2048)

print("\ndrawing: the visual language must not read as a detection")
draw_src = (ROOT / "app" / "pipeline" / "detect_draw.py").read_text()
check("claimed boxes are drawn dashed", "_dashed_line" in draw_src)
check("and detections are not", "_dashed_line" not in
      draw_src.split("def draw_detections(")[1])
check("the caption says who is claiming",
      'f"model says: {claim.label' in draw_src)
check("and carries the IoU when there is one",
      'IoU {claim.iou:.2f} vs detector' in draw_src)
# Two claims on the same object is the NORMAL good case - the presence box and
# the evidence box agreeing - so their captions must not paint over each other.
check("colliding captions are pushed clear rather than overwritten",
      "occupied" in draw_src and "_overlaps" in draw_src)
check("and the search is bounded, not a while-loop off the image",
      "for _ in range(6)" in draw_src)

print("\na mock claims no geometry")
from pipeline.stage3_vlm import mock_vlm_only  # noqa: E402

m = mock_vlm_only(q, "qwen3-vl")
check("a mock carries the grounding keys", "presence_boxes" in m and "answer_boxes" in m)
# A rectangle is the most assertive thing this UI renders. One drawn when no
# model looked would be the worst output the demo could produce.
check("but no boxes at all", not m["presence_boxes"].boxes and not m["answer_boxes"].boxes)
check("and does not claim grounding ran", m["grounding"] is False)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
