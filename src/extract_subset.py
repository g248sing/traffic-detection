"""
Disk-saver: pull ONLY the images you need straight out of the BDD100K zip.

The naive route costs you 10.6 GB at peak:
    bdd100k_images_100k.zip   5.3 GB
  + fully extracted images    5.3 GB
  = 10.6 GB, before you've trained anything

This script never fully extracts. Python's zipfile can read a single member
out of an archive without unpacking the rest, so we:
    1. read the label JSON (small -- 50 MB, already on disk)
    2. choose the image names we actually want
    3. pull just those out of the zip
    4. you delete the zip

Peak becomes 5.3 GB + ~0.5 GB, and steady state is ~0.5 GB.

Order matters: you must CHOOSE before you EXTRACT. That's why Session 1
(reading the labels) comes before the images finish downloading -- your
converter is what tells this script which files are worth keeping.

Usage:
  # see what it would do, without writing anything
  python src/extract_subset.py --zip bdd100k_images_100k.zip \
      --json data/bdd100k/labels/det_20/det_train.json \
      --out data/bdd100k/images/train --n 8000 --dry-run

  # do it
  python src/extract_subset.py --zip bdd100k_images_100k.zip \
      --json data/bdd100k/labels/det_20/det_train.json \
      --out data/bdd100k/images/train --n 8000

NOTE: --n picks images at RANDOM with a fixed seed. That is deliberately the
dumb version. In Session 2 you'll replace the selection with something that
preserves your rare classes, because uniform random sampling of a long-tailed
dataset throws away exactly the examples you can least afford to lose.
"""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from pathlib import Path


def choose_names(json_path: Path, n: int | None, seed: int = 0) -> list[str]:
    """Image file names to keep, in a stable order."""
    entries = json.loads(json_path.read_text())
    names = [e["name"] for e in entries]
    if n is None or n >= len(names):
        return sorted(names)
    rng = random.Random(seed)
    return sorted(rng.sample(names, n))


def extract(zip_path: Path, names: list[str], out_dir: Path,
            dry_run: bool = False) -> dict:
    """Pull the named images out of the archive, ignoring its folder layout."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(names)

    with zipfile.ZipFile(zip_path) as zf:
        # The zip stores paths like "bdd100k/images/100k/train/abc.jpg".
        # Index by basename so we don't care how the release nests things.
        index: dict[str, str] = {}
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            base = member.rsplit("/", 1)[-1]
            if base in wanted:
                index[base] = member

        missing = sorted(wanted - set(index))
        found = sorted(index)
        total_bytes = sum(zf.getinfo(index[b]).file_size for b in found)

        if dry_run:
            return {"found": len(found), "missing": len(missing),
                    "bytes": total_bytes, "missing_sample": missing[:5]}

        for i, base in enumerate(found, 1):
            with zf.open(index[base]) as src:
                (out_dir / base).write_bytes(src.read())
            if i % 500 == 0 or i == len(found):
                print(f"  {i:,}/{len(found):,}", end="\r", flush=True)
        print()

    return {"found": len(found), "missing": len(missing),
            "bytes": total_bytes, "missing_sample": missing[:5]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, type=Path, help="bdd100k images zip")
    ap.add_argument("--json", required=True, type=Path, help="det_*.json for this split")
    ap.add_argument("--out", required=True, type=Path, help="output image directory")
    ap.add_argument("--n", type=int, default=None, help="how many images to keep")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    names = choose_names(args.json, args.n, args.seed)
    print(f"selected {len(names):,} image names from {args.json.name}")

    result = extract(args.zip, names, args.out, args.dry_run)
    print(f"found in zip: {result['found']:,}")
    if result["missing"]:
        print(f"MISSING:      {result['missing']:,}  e.g. {result['missing_sample']}")
        print("  (are you pointing at the zip for the right split?)")
    print(f"size:         {result['bytes'] / 1e9:.2f} GB")

    if args.dry_run:
        print("\ndry run -- nothing written")
    else:
        print(f"\nwritten to {args.out}")
        print("Once you've done this for BOTH train and val, delete the zip.")


if __name__ == "__main__":
    main()
