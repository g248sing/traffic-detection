"""
Session 4 tests. Run these to check your work:

    python tests/test_train_config.py

No dataset needed. Each failure message tells you which TODO it's about.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_config import stems_to_image_paths, write_image_list  # noqa

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
def test_stems_to_image_paths_basic():
    with tempfile.TemporaryDirectory() as d:
        manifest = Path(d) / "manifest.txt"
        manifest.write_text("b\na\nc\n")
        paths = stems_to_image_paths(manifest, Path("/kaggle/working/dataset/images/100k/train"))
        assert paths == [
            "/kaggle/working/dataset/images/100k/train/a.jpg",
            "/kaggle/working/dataset/images/100k/train/b.jpg",
            "/kaggle/working/dataset/images/100k/train/c.jpg",
        ], paths


def test_stems_to_image_paths_skips_blank_lines():
    with tempfile.TemporaryDirectory() as d:
        manifest = Path(d) / "manifest.txt"
        manifest.write_text("a\n\nb\n\n")
        paths = stems_to_image_paths(manifest, Path("/x"))
        assert len(paths) == 2, f"blank lines should be skipped, got {paths}"


def test_stems_to_image_paths_respects_ext():
    with tempfile.TemporaryDirectory() as d:
        manifest = Path(d) / "manifest.txt"
        manifest.write_text("a\n")
        paths = stems_to_image_paths(manifest, Path("/x"), ext=".png")
        assert paths == ["/x/a.png"], paths


def test_stems_to_image_paths_uses_forward_slashes():
    with tempfile.TemporaryDirectory() as d:
        manifest = Path(d) / "manifest.txt"
        manifest.write_text("a\n")
        paths = stems_to_image_paths(manifest, Path("/x/y"))
        assert "\\" not in paths[0], (
            f"path must use forward slashes for Linux/Kaggle, got {paths[0]!r}"
        )


# ---------------------------------------------------------------- TODO 2 ----
def test_write_image_list_basic():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "train_images.txt"
        write_image_list(["/a/1.jpg", "/a/2.jpg"], out)
        assert out.read_text() == "/a/1.jpg\n/a/2.jpg\n", out.read_text()


def test_write_image_list_creates_parent_dirs():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "nested" / "dir" / "train_images.txt"
        write_image_list(["/a/1.jpg"], out)
        assert out.exists(), "write_image_list must create missing parent directories"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        check(name, fn)

    for name, err in FAIL:
        print(f"FAIL  {name}\n      {err}")
    print(f"\n{len(PASS)}/{len(tests)} passing")
    if not FAIL:
        print("\nAll green. Session 4 done -- ping me and we'll move on.")
    sys.exit(1 if FAIL else 0)
