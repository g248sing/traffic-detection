"""
Session 3 tests. Run these to check your work:

    python tests/test_dataset_config.py

No dataset needed. Each failure message tells you which TODO it's about.

Read the test bodies -- they ARE the spec. If a test looks wrong to you, say
so; you might be right.
"""

import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset_config import AUGMENTATION_POLICY, build_data_yaml, write_yaml  # noqa

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
    except AssertionError as e:
        FAIL.append((name, str(e)))
    except Exception as e:
        FAIL.append((name, f"{type(e).__name__}: {e}"))


REQUIRED_KEYS = {
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
    "perspective", "flipud", "fliplr", "mosaic", "mixup", "copy_paste",
}


# ---------------------------------------------------------------- TODO 1 ----
def test_policy_has_all_required_keys():
    missing = REQUIRED_KEYS - set(AUGMENTATION_POLICY)
    assert not missing, f"AUGMENTATION_POLICY is missing: {missing}"


def test_policy_values_are_in_valid_ranges():
    zero_to_one = {"hsv_h", "hsv_s", "hsv_v", "flipud", "fliplr",
                    "mosaic", "mixup", "copy_paste"}
    for key in zero_to_one & set(AUGMENTATION_POLICY):
        v = AUGMENTATION_POLICY[key]
        assert 0.0 <= v <= 1.0, f"{key}={v} must be in [0, 1]"
    assert AUGMENTATION_POLICY["degrees"] >= 0, "degrees can't be negative"
    assert AUGMENTATION_POLICY["scale"] >= 0, "scale can't be negative"
    assert AUGMENTATION_POLICY["perspective"] >= 0, "perspective can't be negative"


def test_flipud_is_disabled():
    assert AUGMENTATION_POLICY["flipud"] == 0, (
        "an upside-down dashcam frame doesn't happen in real driving -- "
        "flipud should be 0"
    )


def test_degrees_and_perspective_are_conservative():
    assert AUGMENTATION_POLICY["degrees"] <= 10, (
        "a windshield-mounted camera doesn't rotate more than a few "
        "degrees -- keep this small"
    )
    assert AUGMENTATION_POLICY["perspective"] <= 0.001, (
        "Ultralytics' own valid range for perspective tops out at 0.001 "
        "-- the images already have real perspective baked in"
    )


# ---------------------------------------------------------------- TODO 2 ----
def test_build_data_yaml_shape():
    d = build_data_yaml(["a", "b", "c"], "/kaggle/working", "train.txt", "val/")
    assert d["path"] == "/kaggle/working", d
    assert d["train"] == "train.txt", d
    assert d["val"] == "val/", d
    assert d["nc"] == 3, d
    assert d["names"] == ["a", "b", "c"], d


def test_build_data_yaml_preserves_order():
    names = ["z", "a", "m"]
    d = build_data_yaml(names, "/x", "t", "v")
    assert d["names"] == names, "names order must match the input, not be sorted"


# ---------------------------------------------------------------- TODO 3 ----
def test_write_yaml_round_trips():
    data = {"path": "/x", "train": "t.txt", "val": "v/", "nc": 2, "names": ["a", "b"]}
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "data.yaml"
        write_yaml(data, out)
        assert out.exists(), "write_yaml did not create the file"
        loaded = yaml.safe_load(out.read_text())
        assert loaded == data, f"round-tripped yaml doesn't match: {loaded}"


def test_write_yaml_creates_parent_dirs():
    data = {"a": 1}
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "nested" / "dir" / "hyp.yaml"
        write_yaml(data, out)
        assert out.exists(), "write_yaml must create missing parent directories"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        check(name, fn)

    for name, err in FAIL:
        print(f"FAIL  {name}\n      {err}")
    print(f"\n{len(PASS)}/{len(tests)} passing")
    if not FAIL:
        print("\nAll green. Session 3 done -- ping me and we'll move on.")
    sys.exit(1 if FAIL else 0)
