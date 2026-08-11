"""
Where is everything? Answered once, for both machines.

The same repo has to run in two places:
  local    -> data/ next to the code, you control the layout
  Kaggle   -> /kaggle/input/<whatever-the-uploader-called-it>/..., read-only,
              and the layout is whatever THEY chose

Different Kaggle uploads of BDD100K nest things differently. Hardcoding a path
means editing code every time you switch datasets, and it fails with a
FileNotFoundError three frames deep instead of saying what's wrong. So we
discover the layout by searching for the two things we actually need:

  * a detection label JSON  (det_train.json / bdd100k_labels_images_train.json)
  * a directory of .jpg images

Run it directly to see what it found:  python src/paths.py
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

ON_KAGGLE = Path("/kaggle/input").exists()
ON_COLAB = "COLAB_GPU" in os.environ

# Filenames BDD has used across releases, newest first.
LABEL_PATTERNS = {
    "train": ["det_train.json", "bdd100k_labels_images_train.json",
              "det_v2_train_release.json"],
    "val": ["det_val.json", "bdd100k_labels_images_val.json",
            "det_v2_val_release.json"],
}


def search_roots() -> list[Path]:
    """Directories worth searching, in priority order."""
    roots = []
    if ON_KAGGLE:
        roots += sorted(Path("/kaggle/input").glob("*"))
        roots.append(Path("/kaggle/working"))
    roots.append(REPO / "data")
    return [r for r in roots if r.exists()]


def find_label_json(split: str, roots: list[Path] | None = None) -> Path | None:
    """First matching detection label JSON, or None."""
    if split not in LABEL_PATTERNS:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    for root in roots if roots is not None else search_roots():
        for name in LABEL_PATTERNS[split]:
            hits = sorted(root.rglob(name))
            if hits:
                return hits[0]
    return None


def find_image_dir(split: str, roots: list[Path] | None = None) -> Path | None:
    """Directory holding this split's .jpg images, or None.

    Looks for a directory literally named after the split that contains jpgs,
    which is how every BDD layout we've seen organises it.
    """
    for root in roots if roots is not None else search_roots():
        candidates = [d for d in root.rglob(split) if d.is_dir()]
        # prefer the one with the most images -- guards against stray
        # empty dirs named 'train' elsewhere in the tree
        best, best_n = None, 0
        for d in candidates:
            n = sum(1 for _ in d.glob("*.jpg"))
            if n > best_n:
                best, best_n = d, n
        if best is not None:
            return best
    return None


def output_root() -> Path:
    """Where WE may write. /kaggle/input is read-only."""
    if ON_KAGGLE:
        return Path("/kaggle/working")
    return REPO


def describe() -> str:
    where = "Kaggle" if ON_KAGGLE else ("Colab" if ON_COLAB else "local")
    lines = [f"environment: {where}", f"writable root: {output_root()}"]
    for split in ("train", "val"):
        j = find_label_json(split)
        d = find_image_dir(split)
        lines.append(f"{split} labels: {j if j else 'NOT FOUND'}")
        if d:
            n = sum(1 for _ in d.glob("*.jpg"))
            lines.append(f"{split} images: {d}  ({n:,} jpgs)")
        else:
            lines.append(f"{split} images: NOT FOUND")
    if not find_label_json("train"):
        lines.append("")
        lines.append("No labels found. Searched: "
                     + ", ".join(str(r) for r in search_roots()))
        lines.append("On Kaggle: did you attach the dataset to the notebook?")
        lines.append("If the filename differs, add it to LABEL_PATTERNS.")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
