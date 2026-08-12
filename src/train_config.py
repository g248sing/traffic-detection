"""
SESSION 4 -- Wire the real Kaggle paths together and run a baseline training.

Sessions 1-3 gave you converted labels, a class-balanced subset, and a
defended dataset/augmentation config. One gap is left before model.train()
actually works: Ultralytics finds each image's label by taking its path and
swapping "images" for "labels" in it (same depth, same filename, .txt
instead of .jpg). Your labels live in /kaggle/working/labels_yolo/, which
doesn't match that convention against BDD100K's real image path on Kaggle
-- and /kaggle/input is read-only anyway, so you can't fix it by writing
labels next to the images.

The fix (a Kaggle notebook cell, not Python -- see SESSION_04.md): symlink
the real images and your converted labels under a shared writable root in
/kaggle/working, so images/.../train/X.jpg and labels/.../train/X.txt sit
where Ultralytics expects them, without copying 70,000 images.

This file's job is turning Session 2's subset manifest (image STEMS) into
the exact list of full image PATHS Ultralytics needs for `train:` in
data.yaml, once those symlinks exist.

Check your work with:
    python tests/test_train_config.py
"""

from __future__ import annotations

from pathlib import Path


# ============================================================== TODO 1 ======
def stems_to_image_paths(manifest_path: Path, image_dir: Path, ext: str = ".jpg") -> list[str]:
    """Read a manifest of image stems (one per line -- Session 2's output
    format) and turn each into a full image path under image_dir.

    Returns a sorted list of path strings (str, not Path -- these get
    written straight into a text file Ultralytics reads on Linux, so use
    forward slashes regardless of what OS you're running this on).

    Blank lines in the manifest (e.g. a trailing newline) must be skipped,
    not turned into a bogus path.
    """
    stems = [line.strip() for line in Path(manifest_path).read_text().splitlines() if line.strip()]
    paths = [(Path(image_dir) / f"{stem}{ext}").as_posix() for stem in stems]
    return sorted(paths)


# ============================================================== TODO 2 ======
def write_image_list(paths: list[str], out_path: Path) -> None:
    """Write `paths`, one per line, to out_path. Create parent directories
    if they don't exist. This becomes the `train:`/`val:` value in
    data.yaml when pointing at an explicit list rather than a directory.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(paths) + ("\n" if paths else ""))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Build a full-path image list from a Session 2 stem manifest"
    )
    ap.add_argument("--manifest", required=True, help="Session 2's subset_train.txt (stems)")
    ap.add_argument("--image-dir", required=True,
                     help="symlinked images dir Ultralytics will actually read from")
    ap.add_argument("--out", required=True, help="output .txt of full image paths")
    args = ap.parse_args()

    paths = stems_to_image_paths(Path(args.manifest), Path(args.image_dir))
    write_image_list(paths, Path(args.out))
    print(f"wrote {len(paths):,} image paths to {args.out}")
