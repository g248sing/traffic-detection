"""
Session 5 tests. Run these to check your work:

    python tests/test_evaluate.py

No dataset needed. Each failure message tells you which TODO it's about.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate import (  # noqa
    PR_GAP_THRESHOLD,
    LOC_GAP_THRESHOLD,
    parse_class_table,
    flag_anomalies,
)

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
    except AssertionError as e:
        FAIL.append((name, str(e)))
    except Exception as e:
        FAIL.append((name, f"{type(e).__name__}: {e}"))


# Real Session 4 output, pasted verbatim -- header, "all" row, and all.
REAL_TABLE_TEXT = """
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all      10000     185681      0.681      0.552      0.599      0.343
                person       3443      14077       0.78      0.509      0.608      0.301
                   car       9882     102799      0.837      0.671      0.763      0.479
                 truck       2733       4243      0.734      0.437      0.559        0.4
                   bus       1299       1660      0.436      0.645      0.555      0.434
           two_wheeler        876       1497      0.542        0.5      0.485      0.245
         traffic_light       5648      26754      0.707      0.557      0.602      0.221
          traffic_sign       8211      34651      0.732      0.547      0.619      0.321
"""


# ---------------------------------------------------------------- TODO 1 ----
def test_pr_gap_threshold_is_sane():
    assert 0.02 <= PR_GAP_THRESHOLD <= 0.5, (
        f"PR_GAP_THRESHOLD={PR_GAP_THRESHOLD} should be a real, calibrated "
        f"gap size, not 0 or something wildly loose"
    )


def test_loc_gap_threshold_is_sane():
    assert 0.05 <= LOC_GAP_THRESHOLD <= 0.5, (
        f"LOC_GAP_THRESHOLD={LOC_GAP_THRESHOLD} should be a real, calibrated "
        f"gap size, not 0 or something wildly loose"
    )


# ---------------------------------------------------------------- TODO 2 ----
def test_parse_class_table_row_count():
    rows = parse_class_table(REAL_TABLE_TEXT)
    assert len(rows) == 7, (
        f"expected 7 classes (all/header excluded), got {len(rows)}: "
        f"{[r['class'] for r in rows]}"
    )


def test_parse_class_table_skips_all_row():
    rows = parse_class_table(REAL_TABLE_TEXT)
    assert "all" not in {r["class"] for r in rows}, "the 'all' summary row is not a class"


def test_parse_class_table_values():
    rows = {r["class"]: r for r in parse_class_table(REAL_TABLE_TEXT)}
    assert "car" in rows, rows
    car = rows["car"]
    assert car["images"] == 9882, car
    assert car["instances"] == 102799, car
    assert abs(car["P"] - 0.837) < 1e-9, car
    assert abs(car["R"] - 0.671) < 1e-9, car
    assert abs(car["mAP50"] - 0.763) < 1e-9, car
    assert abs(car["mAP50_95"] - 0.479) < 1e-9, car


# ---------------------------------------------------------------- TODO 3 ----
def test_flag_anomalies_catches_pr_inversion():
    rows = [{"class": "x", "images": 1, "instances": 1,
             "P": 0.2, "R": 0.9, "mAP50": 0.5, "mAP50_95": 0.49}]
    flags = flag_anomalies(rows)
    assert any("x" in f for f in flags), (
        f"a class with R=0.9, P=0.2 should be flagged for PR inversion, got {flags}"
    )


def test_flag_anomalies_ignores_normal_pr():
    rows = [{"class": "y", "images": 1, "instances": 1,
             "P": 0.8, "R": 0.75, "mAP50": 0.5, "mAP50_95": 0.49}]
    flags = flag_anomalies(rows)
    assert not any("y" in f for f in flags), (
        f"P and R nearly equal (and P > R) shouldn't be flagged, got {flags}"
    )


def test_flag_anomalies_catches_localization_gap():
    rows = [{"class": "z", "images": 1, "instances": 1,
             "P": 0.7, "R": 0.7, "mAP50": 0.9, "mAP50_95": 0.1}]
    flags = flag_anomalies(rows)
    assert any("z" in f for f in flags), (
        f"a huge mAP50/mAP50-95 gap should be flagged, got {flags}"
    )


def test_flag_anomalies_ignores_tight_localization():
    rows = [{"class": "w", "images": 1, "instances": 1,
             "P": 0.7, "R": 0.7, "mAP50": 0.5, "mAP50_95": 0.48}]
    flags = flag_anomalies(rows)
    assert not any("w" in f for f in flags), (
        f"a 0.02 gap is tight localization, shouldn't be flagged, got {flags}"
    )


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        check(name, fn)

    for name, err in FAIL:
        print(f"FAIL  {name}\n      {err}")
    print(f"\n{len(PASS)}/{len(tests)} passing")
    if not FAIL:
        print("\nAll green. Session 5 done -- ping me and we'll move on.")
    sys.exit(1 if FAIL else 0)
