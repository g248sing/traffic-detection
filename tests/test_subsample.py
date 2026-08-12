"""
Session 2 tests. Run these to check your work:

    python tests/test_subsample.py

No dataset needed -- everything here is synthetic or a temp directory.
Each failure message tells you which TODO it's about.

Read the test bodies -- they ARE the spec. If a test looks wrong to you, say
so; you might be right.
"""

import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subsample import (  # noqa
    RARE_CLASS_IDS,
    TARGET_IMAGES,
    count_labels,
    select_subset,
    subset_class_table,
)
from bdd_to_yolo import CLASS_NAMES  # noqa

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
def test_car_is_not_protected():
    car_id = CLASS_NAMES.index("car")
    assert car_id not in RARE_CLASS_IDS, (
        "car is 55% of the dataset -- protecting it defeats the point"
    )


def test_rare_class_ids_is_nonempty():
    assert len(RARE_CLASS_IDS) > 0, "pick at least one class to protect"


def test_target_images_is_a_real_subset():
    full_train_images = 69_863  # BDD100K train split, from Session 1
    assert 0 < TARGET_IMAGES < full_train_images, (
        f"TARGET_IMAGES={TARGET_IMAGES} isn't a meaningful subset of "
        f"{full_train_images} train images"
    )


# ---------------------------------------------------------------- TODO 2 ----
def test_count_labels_basic():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "a.txt").write_text(
            "1 0.5 0.5 0.1 0.1\n3 0.2 0.2 0.05 0.05\n1 0.1 0.1 0.1 0.1\n"
        )
        (d / "b.txt").write_text("4 0.5 0.5 0.2 0.2\n")
        counts = count_labels(d)
        assert set(counts) == {"a", "b"}, f"expected stems a, b -- got {set(counts)}"
        assert counts["a"] == Counter({1: 2, 3: 1}), f"wrong counts for a: {counts['a']}"
        assert counts["b"] == Counter({4: 1}), f"wrong counts for b: {counts['b']}"


def test_count_labels_keeps_empty_files():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "empty.txt").write_text("")
        counts = count_labels(d)
        assert "empty" in counts, (
            "empty label files must still appear -- they're negative examples"
        )
        assert sum(counts["empty"].values()) == 0, "empty file should have zero instances"


# ---------------------------------------------------------------- TODO 3 ----
def _synthetic_counts():
    return {
        "a": Counter({1: 3}),   # car only
        "b": Counter({1: 2}),   # car only
        "c": Counter({1: 1}),   # car only
        "d": Counter({3: 1}),   # bus -- rare
        "e": Counter({4: 2}),   # two_wheeler -- rare
    }


def test_select_subset_always_keeps_rare_images():
    counts = _synthetic_counts()
    subset = select_subset(counts, target_images=3, rare_class_ids={3, 4}, seed=0)
    assert {"d", "e"} <= subset, f"rare images d, e must always be kept, got {subset}"


def test_select_subset_fills_to_target_when_possible():
    counts = _synthetic_counts()
    subset = select_subset(counts, target_images=3, rare_class_ids={3, 4}, seed=0)
    assert len(subset) == 3, f"expected exactly 3 images, got {len(subset)}: {subset}"
    assert subset - {"d", "e"} <= {"a", "b", "c"}, f"unexpected extra images: {subset}"


def test_select_subset_protects_beyond_budget():
    counts = _synthetic_counts()
    subset = select_subset(counts, target_images=1, rare_class_ids={3, 4}, seed=0)
    assert subset == {"d", "e"}, (
        f"budget of 1 is smaller than the 2 protected images -- rare "
        f"coverage should win, got {subset}"
    )


def test_select_subset_returns_everything_if_not_enough():
    counts = _synthetic_counts()
    subset = select_subset(counts, target_images=100, rare_class_ids={3, 4}, seed=0)
    assert subset == set(counts), f"only 5 images exist total, expected all of them, got {subset}"


def test_select_subset_is_reproducible():
    counts = _synthetic_counts()
    first = select_subset(counts, target_images=3, rare_class_ids={3, 4}, seed=7)
    second = select_subset(counts, target_images=3, rare_class_ids={3, 4}, seed=7)
    assert first == second, "same seed must give the same subset"


def test_subset_class_table_sums_correctly():
    counts = _synthetic_counts()
    subset = {"a", "d", "e"}  # car x3, bus x1, two_wheeler x2
    table = subset_class_table(counts, subset, CLASS_NAMES)
    assert table["car"] == 3, table
    assert table["bus"] == 1, table
    assert table["two_wheeler"] == 2, table
    assert table["person"] == 0, "classes with zero instances must still appear as 0"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        check(name, fn)

    for name, err in FAIL:
        print(f"FAIL  {name}\n      {err}")
    print(f"\n{len(PASS)}/{len(tests)} passing")
    if not FAIL:
        print("\nAll green. Session 2 done -- ping me and we'll move on.")
    sys.exit(1 if FAIL else 0)
