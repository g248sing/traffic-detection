"""
Session 6 tests. Run these to check your work:

    python tests/test_compare.py

No dataset needed. Each failure message tells you which TODO it's about.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from compare import compare_runs, format_comparison_table  # noqa
from evaluate import parse_class_table  # noqa

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
    except AssertionError as e:
        FAIL.append((name, str(e)))
    except Exception as e:
        FAIL.append((name, f"{type(e).__name__}: {e}"))


# Real Session 4 baseline.
BEFORE_TEXT = """
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

# Synthetic "after" -- traffic_light improved the most (targeted by imgsz
# increase), bus improved the least (untouched by this session's change).
AFTER_TEXT = """
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all      10000     185681      0.695      0.565      0.617      0.364
                person       3443      14077       0.79      0.515      0.617      0.305
                   car       9882     102799       0.84      0.675       0.77      0.482
                 truck       2733       4243       0.74      0.445       0.57      0.405
                   bus       1299       1660       0.44       0.64      0.558      0.436
           two_wheeler        876       1497      0.545      0.505       0.49      0.248
         traffic_light       5648      26754       0.72       0.59       0.63       0.29
          traffic_sign       8211      34651      0.735       0.55      0.632      0.325
"""


# ---------------------------------------------------------------- TODO 1 ----
def test_compare_runs_matches_by_class():
    before = parse_class_table(BEFORE_TEXT)
    after = parse_class_table(AFTER_TEXT)
    result = compare_runs(before, after)
    assert {r["class"] for r in result} == {r["class"] for r in before}, (
        f"expected all 7 classes matched, got {[r['class'] for r in result]}"
    )


def test_compare_runs_computes_delta_direction():
    before = parse_class_table(BEFORE_TEXT)
    after = parse_class_table(AFTER_TEXT)
    result = {r["class"]: r for r in compare_runs(before, after)}
    tl = result["traffic_light"]
    assert abs(tl["mAP50_95_delta"] - (0.290 - 0.221)) < 1e-6, tl
    assert tl["mAP50_95_delta"] > 0, "traffic_light should show improvement (positive delta)"


def test_compare_runs_drops_unmatched_classes():
    before = [
        {"class": "a", "P": 0.5, "R": 0.5, "mAP50": 0.5, "mAP50_95": 0.5, "images": 1, "instances": 1},
        {"class": "b", "P": 0.5, "R": 0.5, "mAP50": 0.5, "mAP50_95": 0.5, "images": 1, "instances": 1},
    ]
    after = [
        {"class": "a", "P": 0.6, "R": 0.6, "mAP50": 0.6, "mAP50_95": 0.6, "images": 1, "instances": 1},
    ]
    result = compare_runs(before, after)
    assert {r["class"] for r in result} == {"a"}, (
        f"class 'b' only exists in 'before' -- can't compare it, should be dropped: {result}"
    )


# ---------------------------------------------------------------- TODO 2 ----
def test_format_comparison_table_sorted_worst_first():
    before = parse_class_table(BEFORE_TEXT)
    after = parse_class_table(AFTER_TEXT)
    rows = compare_runs(before, after)
    table = format_comparison_table(rows)
    data_lines = [
        line for line in table.splitlines()
        if line.startswith("|") and "---" not in line and "Class" not in line
    ]
    assert "bus" in data_lines[0], (
        f"bus has the smallest mAP50 improvement (+0.003) -- should sort first, got: {data_lines[0]}"
    )


def test_format_comparison_table_shows_signed_deltas():
    before = parse_class_table(BEFORE_TEXT)
    after = parse_class_table(AFTER_TEXT)
    rows = compare_runs(before, after)
    table = format_comparison_table(rows)
    assert "+0." in table, f"expected an explicit '+' sign on a positive delta somewhere: {table}"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        check(name, fn)

    for name, err in FAIL:
        print(f"FAIL  {name}\n      {err}")
    print(f"\n{len(PASS)}/{len(tests)} passing")
    if not FAIL:
        print("\nAll green. Session 6 done -- ping me and we'll move on.")
    sys.exit(1 if FAIL else 0)
