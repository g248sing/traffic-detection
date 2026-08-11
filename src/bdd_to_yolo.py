"""
SESSION 1 -- Convert BDD100K detection labels into YOLO format.

This is the same shape of problem as Penn-Fudan, one level harder:
  Penn-Fudan  ->  one class, labels stored as image masks
  BDD100K     ->  ten categories, labels stored as JSON, and messy

Your job is the three TODOs below. Everything else is written.
Check your work with:  python tests/test_bdd_conversion.py

------------------------------------------------------------------
What a BDD100K label entry actually looks like (one per image):

{
  "name": "0000f77c-6257be58.jpg",
  "attributes": {"weather": "clear", "scene": "city street",
                 "timeofday": "daytime"},
  "labels": [
    {"id": 0,
     "category": "traffic sign",
     "attributes": {"occluded": false, "truncated": false},
     "box2d": {"x1": 1000.7, "y1": 281.9, "x2": 1040.6, "y2": 326.9}},
    {"id": 1,
     "category": "drivable area",
     "poly2d": [...]}            <- no box2d! not a detection label
  ]
}

Things that will bite you, all of which the tests check:
  * some labels have poly2d instead of box2d (lanes, drivable area)
  * "labels" can be null for an empty image
  * a few boxes have x2 < x1
  * a few boxes run outside the frame
  * a lot of boxes are 2-3 px (distant signs) and are unlearnable noise

BDD100K images are all 1280x720, so we can hardcode the size instead of
opening 70,000 files to ask. (Confirm it once on a real sample rather than
trusting me -- that's a habit worth having.)
------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

IMG_W, IMG_H = 1280, 720


# ============================================================== TODO 1 ======
# Decide your classes, then fill in both structures below.
#
# BDD100K gives you 10 detection categories:
#     pedestrian, rider, car, truck, bus, bike, motor,
#     traffic light, traffic sign, train
#
# Merging classes is a MODELLING decision, not a formatting one. Things worth
# thinking about before you write anything:
#
#   - The brief says "vehicles, pedestrians, and traffic signs". Nothing forces
#     you to keep car/truck/bus separate. What do you gain by splitting them,
#     and what do you pay? (Hint: what happens to a class with 2,000 instances
#     next to a class with 700,000?)
#   - A "rider" is a person on a bike. Is that a person, or a two-wheeler, or
#     both? There is no right answer -- but there is a defensible one, and
#     you'll need to defend it in the write-up.
#   - "train" has roughly 150 instances in the entire dataset. What can a model
#     possibly learn from that? What does it do to your mean-over-classes mAP?
#
# CATEGORY_MAP maps BDD's category strings -> your YOLO class ids.
# Several BDD strings may point at the same id. That's the merging.
# CLASS_NAMES is indexed BY class id, so its order must match.
#
# The tests pin down a few properties (contiguous ids, train excluded,
# signs != lights) but the rest is yours.

CATEGORY_MAP: dict[str, int] = {
    "pedestrian": 0,        # person
    "rider": 0,             # BDD boxes the human separately from their bike/motor
                             # -- "rider" is a person, not a vehicle
    "car": 1,
    "truck": 2,
    "bus": 3,
    "bike": 4,              # two_wheeler = the vehicle itself
    "motor": 4,
    "bicycle": 4,           # BDD renamed bike/motor -> bicycle/motorcycle in
    "motorcycle": 4,        # the 2020 (det_v2/det_20) label release
    "traffic light": 5,
    "traffic sign": 6,
    # Deliberately excluded, same reasoning as "train" (~150 instances):
    # too rare to learn anything from, and would drag down mean-over-classes
    # mAP for free. "other vehicle" (804) and "other person" (210) are also
    # catch-all buckets with no consistent visual identity -- folding them
    # into car/person would mean labelling things that don't look like a
    # car/person as a car/person, which is worse than dropping them.
    #   "train": ~150, "other vehicle": 804, "other person": 210,
    #   "trailer": 71
}

CLASS_NAMES: list[str] = [
    "person",         # 0
    "car",            # 1
    "truck",          # 2
    "bus",            # 3
    "two_wheeler",    # 4  (bike, motor, rider merged)
    "traffic_light",  # 5
    "traffic_sign",   # 6
]


# ============================================================== TODO 2 ======
def box2d_to_yolo(box: dict, img_w: int = IMG_W, img_h: int = IMG_H):
    """Convert one BDD box2d dict to a YOLO tuple.

    In:   {"x1": .., "y1": .., "x2": .., "y2": ..}   pixel corners
    Out:  (cx, cy, w, h) normalised to 0-1, or None if the box is unusable

    Recall the YOLO convention from the Penn-Fudan project: centre point plus
    size, not corners, and every value divided by the image dimension.

    Handle, in this order:
      1. corners possibly reversed (x2 < x1, or y2 < y1)
      2. corners possibly outside the frame -- clip to [0, img_w] / [0, img_h]
      3. zero-area boxes after clipping -> return None

    Order matters. Think about why clipping before the reversal check would
    give you a different answer.
    """
    x1, x2 = min(box["x1"], box["x2"]), max(box["x1"], box["x2"])
    y1, y2 = min(box["y1"], box["y2"]), max(box["y1"], box["y2"])

    x1, x2 = max(0.0, min(x1, img_w)), max(0.0, min(x2, img_w))
    y1, y2 = max(0.0, min(y1, img_h)), max(0.0, min(y2, img_h))

    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None

    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    return (cx, cy, w / img_w, h / img_h)


# ============================================================== TODO 3 ======
def convert_entry(entry: dict, min_box_px: float = 4.0) -> list[str]:
    """Convert one image's JSON entry into a list of YOLO label lines.

    Each line: "<class_id> <cx> <cy> <w> <h>"  -- format the floats to 6dp
    so the files stay small and diffable.

    Skip a label when:
      - its category isn't in your CATEGORY_MAP
      - it has no "box2d" key (polygon annotations)
      - box2d_to_yolo returned None
      - the box is smaller than min_box_px in EITHER dimension, in PIXELS
        (careful: box2d_to_yolo hands you normalised values)

    An image with no usable labels returns [] -- that's legal and normal.
    YOLO expects an empty .txt file for it, and those empty files are useful:
    they teach the model what "no object here" looks like.

    Why 4 px? It's a judgement call. Have a look at what fraction of traffic
    signs you're throwing away at 4 vs 8 vs 16 -- that number is worth knowing
    before you wonder why sign recall is low.
    """
    lines = []
    for label in entry.get("labels") or []:
        category = label.get("category")
        if category not in CATEGORY_MAP:
            continue

        box = label.get("box2d")
        if box is None:
            continue

        yolo = box2d_to_yolo(box)
        if yolo is None:
            continue

        cx, cy, w, h = yolo
        if w * IMG_W < min_box_px or h * IMG_H < min_box_px:
            continue

        class_id = CATEGORY_MAP[category]
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines


# ---------------------------------------------------------------------------
# Already written for you -- this walks the whole JSON file and writes labels.
# Read it, don't change it yet.
# ---------------------------------------------------------------------------
def convert_split(json_path: Path, out_dir: Path, min_box_px: float = 4.0) -> dict:
    """Convert a BDD100K label JSON into one .txt per image. Returns a summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = json.loads(Path(json_path).read_text())

    per_class = {i: 0 for i in range(len(CLASS_NAMES))}
    n_empty = 0

    for entry in entries:
        lines = convert_entry(entry, min_box_px=min_box_px)
        stem = Path(entry["name"]).stem
        (out_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        if not lines:
            n_empty += 1
        for line in lines:
            per_class[int(line.split()[0])] += 1

    return {
        "images": len(entries),
        "empty_images": n_empty,
        "instances": sum(per_class.values()),
        "per_class": {CLASS_NAMES[i]: n for i, n in per_class.items()},
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="BDD100K JSON -> YOLO labels")
    ap.add_argument("--json", required=True, help="bdd100k_labels_images_train.json")
    ap.add_argument("--out", required=True, help="output label directory")
    ap.add_argument("--min-box-px", type=float, default=4.0)
    args = ap.parse_args()

    summary = convert_split(Path(args.json), Path(args.out), args.min_box_px)
    print(f"images:    {summary['images']:,}  ({summary['empty_images']:,} empty)")
    print(f"instances: {summary['instances']:,}")
    print("\nper class:")
    width = max((len(k) for k in summary["per_class"]), default=0)
    for name, n in sorted(summary["per_class"].items(), key=lambda kv: -kv[1]):
        share = 100 * n / max(1, summary["instances"])
        print(f"  {name:<{width}}  {n:>8,}  {share:5.1f}%")
    print("\nLook hard at that distribution before you train on it.")
