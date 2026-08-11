"""
Session 1 tests. Run these to check your work:

    python tests/test_bdd_conversion.py

They need no dataset, no GPU, and finish in a second. Work until all of them
pass. Each failure message tells you which TODO it's about.

Read the test bodies -- they ARE the spec. If a test looks wrong to you, say so;
you might be right.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bdd_to_yolo import CATEGORY_MAP, CLASS_NAMES, box2d_to_yolo, convert_entry  # noqa

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
    except AssertionError as e:
        FAIL.append((name, str(e)))
    except Exception as e:
        FAIL.append((name, f"{type(e).__name__}: {e}"))


# ---------------------------------------------------------------- TODO 1 ----
def test_map_covers_bdd_categories():
    """Every BDD100K detection category you care about must be mapped."""
    for cat in ["pedestrian", "rider", "car", "truck", "bus",
                "bike", "motor", "traffic light", "traffic sign"]:
        assert cat in CATEGORY_MAP, f"'{cat}' is missing from CATEGORY_MAP"


def test_map_indices_are_contiguous():
    """YOLO class ids must be 0..N-1 with no gaps, matching CLASS_NAMES."""
    ids = sorted(set(CATEGORY_MAP.values()))
    assert ids == list(range(len(CLASS_NAMES))), (
        f"class ids {ids} don't line up with {len(CLASS_NAMES)} CLASS_NAMES"
    )


def test_train_is_excluded():
    """'train' has ~150 instances in all of BDD100K. Too rare to learn."""
    assert "train" not in CATEGORY_MAP, "drop 'train' -- see the session notes"


def test_person_classes_merge():
    assert CATEGORY_MAP["pedestrian"] == CATEGORY_MAP["rider"], \
        "pedestrian and rider should share a class id"


def test_signs_and_lights_are_separate():
    assert CATEGORY_MAP["traffic sign"] != CATEGORY_MAP["traffic light"], \
        "signs and lights are different objects -- keep them apart"


# ---------------------------------------------------------------- TODO 2 ----
def test_box_centre_and_size():
    """A 1280x720 image; box from (100,200) to (300,400).
    centre (200,300), size (200,200) -> normalise by width/height."""
    out = box2d_to_yolo({"x1": 100, "y1": 200, "x2": 300, "y2": 400}, 1280, 720)
    assert out is not None, "returned None for a perfectly valid box"
    cx, cy, w, h = out
    assert abs(cx - 200 / 1280) < 1e-6, f"cx wrong: {cx}"
    assert abs(cy - 300 / 720) < 1e-6, f"cy wrong: {cy}"
    assert abs(w - 200 / 1280) < 1e-6, f"w wrong: {w}"
    assert abs(h - 200 / 720) < 1e-6, f"h wrong: {h}"


def test_box_is_normalised():
    """Full-frame box -> centre (0.5, 0.5), size (1.0, 1.0)."""
    cx, cy, w, h = box2d_to_yolo({"x1": 0, "y1": 0, "x2": 1280, "y2": 720}, 1280, 720)
    for v, want in ((cx, .5), (cy, .5), (w, 1.), (h, 1.)):
        assert abs(v - want) < 1e-6, f"expected {want}, got {v}"


def test_box_handles_reversed_corners():
    """Some BDD entries have x2 < x1. Don't emit a negative width."""
    out = box2d_to_yolo({"x1": 300, "y1": 400, "x2": 100, "y2": 200}, 1280, 720)
    assert out is not None, "reversed corners should still produce a box"
    _, _, w, h = out
    assert w > 0 and h > 0, f"negative size from reversed corners: w={w} h={h}"


def test_box_clipped_to_frame():
    """Boxes can run past the edge. Clip them -- YOLO rejects coords outside 0-1."""
    out = box2d_to_yolo({"x1": -50, "y1": -30, "x2": 1400, "y2": 800}, 1280, 720)
    cx, cy, w, h = out
    for v in (cx, cy, w, h):
        assert 0.0 <= v <= 1.0, f"coordinate outside [0,1]: {v}"
    assert abs(w - 1.0) < 1e-6, "a box covering the frame should have w == 1.0"


def test_degenerate_box_returns_none():
    """Zero-area boxes exist in the wild. Reject, don't crash."""
    assert box2d_to_yolo({"x1": 100, "y1": 100, "x2": 100, "y2": 300}, 1280, 720) is None
    assert box2d_to_yolo({"x1": 100, "y1": 100, "x2": 300, "y2": 100}, 1280, 720) is None


# ---------------------------------------------------------------- TODO 3 ----
def _entry(labels):
    return {"name": "x.jpg", "attributes": {}, "labels": labels}


def test_convert_basic():
    lines = convert_entry(_entry([
        {"category": "car", "box2d": {"x1": 100, "y1": 200, "x2": 300, "y2": 400}},
        {"category": "traffic sign", "box2d": {"x1": 10, "y1": 20, "x2": 60, "y2": 90}},
    ]))
    assert len(lines) == 2, f"expected 2 lines, got {len(lines)}"
    for line in lines:
        parts = line.split()
        assert len(parts) == 5, f"a YOLO line needs 5 fields, got {len(parts)}: {line}"
        int(parts[0])                       # class id must parse as int
        for v in parts[1:]:
            assert 0.0 <= float(v) <= 1.0


def test_convert_skips_unmapped_categories():
    lines = convert_entry(_entry([
        {"category": "train", "box2d": {"x1": 0, "y1": 0, "x2": 100, "y2": 100}},
        {"category": "drivable area", "box2d": {"x1": 0, "y1": 0, "x2": 100, "y2": 100}},
    ]))
    assert lines == [], f"unmapped categories should be dropped, got {lines}"


def test_convert_skips_polygon_only_labels():
    """Lane and drivable-area entries have poly2d and NO box2d. Skip them."""
    lines = convert_entry(_entry([
        {"category": "car", "poly2d": [[0, 0], [1, 1]]},
        {"category": "car", "box2d": {"x1": 100, "y1": 200, "x2": 300, "y2": 400}},
    ]))
    assert len(lines) == 1, f"expected 1 line (poly2d one skipped), got {len(lines)}"


def test_convert_handles_missing_labels_key():
    """Images with nothing in them have labels: null. Must not crash."""
    assert convert_entry({"name": "x.jpg"}) == []
    assert convert_entry({"name": "x.jpg", "labels": None}) == []


def test_convert_drops_tiny_boxes():
    """A 2x2 px traffic sign is unlearnable noise. Default floor is 4 px."""
    lines = convert_entry(_entry([
        {"category": "traffic sign", "box2d": {"x1": 100, "y1": 100, "x2": 102, "y2": 102}},
    ]))
    assert lines == [], f"2px box should be dropped, got {lines}"


def test_convert_class_id_matches_map():
    lines = convert_entry(_entry([
        {"category": "traffic sign", "box2d": {"x1": 10, "y1": 20, "x2": 60, "y2": 90}},
    ]))
    assert int(lines[0].split()[0]) == CATEGORY_MAP["traffic sign"]


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        check(name, fn)

    for name, err in FAIL:
        print(f"FAIL  {name}\n      {err}")
    print(f"\n{len(PASS)}/{len(tests)} passing")
    if not FAIL:
        print("\nAll green. Session 1 done -- ping me and we'll move on.")
    sys.exit(1 if FAIL else 0)
