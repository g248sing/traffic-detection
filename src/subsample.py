"""
SESSION 2 -- Build a training subset that fits a Kaggle session and doesn't
drown out the rare classes.

Session 1 converted BDD100K into labels_yolo/{train,val}/*.txt and gave you
this shape for train (69,863 images, 1,270,020 instances):

    car             55.1%
    traffic_sign    18.7%
    traffic_light   14.6%
    person           7.6%
    truck            2.2%
    bus              0.9%
    two_wheeler      0.8%

Two separate problems, and this file solves both:

  SIZE    -- training on all 69,863 images won't finish in a 12-hour Kaggle
             session at a useful epoch count. You need a subset.
  BALANCE -- car outnumbers bus/two_wheeler by 60-70x. A *random* subset of
             any size reproduces that same ratio -- the model would see so
             few bus/two_wheeler examples per epoch it barely learns them.
             The subset has to be deliberately biased toward images that
             contain the rare classes.

Your job is the three TODOs below. Check your work with:
    python tests/test_subsample.py
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path


# ============================================================== TODO 1 ======
# Two judgement calls, same spirit as Session 1's class design:
#
#   - RARE_CLASS_IDS: which classes get "protected" -- every image
#     containing at least one instance of a protected class is guaranteed
#     into the subset, no matter what the size budget says. Look at the
#     table above. bus (0.9%) and two_wheeler (0.8%) are the obvious picks.
#     Is truck (2.2%) rare enough to protect too? There's no free answer:
#     protecting more classes means more of your budget is spent on
#     guaranteed images, leaving less room for random (mostly car) coverage.
#
#   - TARGET_IMAGES: how many images fit a ~12-hour Kaggle session. Back of
#     envelope: pick a model size (yolo11n/s/m), estimate images/sec for a
#     training step on your GPU, work out how many epochs over N images
#     fits in 12 hours. Don't trust a number I hand you here -- run a short
#     smoke-training (a few hundred steps) on Kaggle and time it, the same
#     way Session 1 told you to confirm image size on real data instead of
#     trusting the docstring.
#
# The tests only pin loose sanity bounds (car isn't protected, the subset
# is meaningfully smaller than the full dataset). The actual numbers, and a
# one-line comment on why, are yours.

RARE_CLASS_IDS: set[int] = {3, 4}  # bus, two_wheeler -- both under 1% of
                                     # instances, a random sample would
                                     # barely see them. truck (2.2%) left
                                     # unprotected -- common enough that
                                     # random fill should cover it; revisit
                                     # if the real subset table says otherwise.

# Back-of-envelope: yolo11s on 2xT4 @ 640px, ~50 img/s combined (unverified
# -- benchmark on Kaggle before trusting this). 100 epochs over 15,000
# images / 50 img/s =~ 8.3h, leaving headroom in a 12h session for
# validation + checkpointing.
TARGET_IMAGES: int = 15000


# ============================================================== TODO 2 ======
def count_labels(label_dir: Path) -> dict[str, Counter]:
    """Count class instances per image in a directory of YOLO .txt files.

    Returns {image_stem: Counter({class_id: count, ...})}.

    Every .txt file in label_dir must produce an entry, even ones with no
    lines -- an empty file means "zero objects", not "skip this image".
    Session 1 is why those files exist; don't throw them away here.

    A YOLO line looks like "<class_id> <cx> <cy> <w> <h>" -- you only need
    the first field.
    """
    result = {}
    for f in sorted(Path(label_dir).glob("*.txt")):
        c = Counter()
        text = f.read_text().strip()
        if text:
            for line in text.splitlines():
                c[int(line.split()[0])] += 1
        result[f.stem] = c
    return result


# ============================================================== TODO 3 ======
def select_subset(
    counts: dict[str, Counter],
    target_images: int,
    rare_class_ids: set[int],
    seed: int = 0,
) -> set[str]:
    """Pick a subset of image stems, biased toward rare_class_ids.

    Strategy:
      1. "Protected" = every image whose Counter has at least one class id
         in rare_class_ids. All protected images are always included --
         even if that makes the result BIGGER than target_images. Rare-class
         coverage wins over the size budget, not the other way round.
      2. Fill the rest of the budget (target_images - len(protected)) with
         a reproducible random sample of the remaining images. Sort the
         remaining stems before sampling so the same seed gives the same
         answer regardless of dict ordering, and use
         random.Random(seed).sample(...) rather than the global random
         module -- that's what makes it reproducible.
      3. If there aren't enough remaining images to fill the budget, just
         return everything you've got.

    Images with an empty Counter (no usable labels) are ordinary images for
    step 2 -- don't exclude them, YOLO needs negative examples too.
    """
    protected = {stem for stem, c in counts.items() if set(c) & rare_class_ids}
    remaining = sorted(set(counts) - protected)

    budget = target_images - len(protected)
    if budget <= 0:
        return set(protected)

    if budget >= len(remaining):
        fill = remaining
    else:
        fill = random.Random(seed).sample(remaining, budget)

    return protected | set(fill)


# ---------------------------------------------------------------------------
# Already written -- sums instances for whatever subset you picked, and
# wires the three pieces together into a manifest file.
# ---------------------------------------------------------------------------
def subset_class_table(
    counts: dict[str, Counter], subset: set[str], class_names: list[str]
) -> dict[str, int]:
    """Total instances per class, summed over only the images in `subset`."""
    totals = {i: 0 for i in range(len(class_names))}
    for stem in subset:
        for class_id, n in counts[stem].items():
            totals[class_id] += n
    return {class_names[i]: n for i, n in totals.items()}


def build_subset(
    label_dir: Path,
    out_manifest: Path,
    target_images: int,
    rare_class_ids: set[int],
    class_names: list[str],
    seed: int = 0,
) -> dict:
    """Run count_labels -> select_subset -> subset_class_table, and write
    the chosen image stems to out_manifest, one per line.

    Why a manifest of stems, not a copy of the images: Ultralytics' YOLO
    accepts a .txt file of image paths as `train:`/`val:` in its dataset
    config instead of a directory -- so this list is directly usable in
    Session 3/4 without duplicating tens of thousands of images on disk
    (and Kaggle's /kaggle/input is read-only anyway).
    """
    counts = count_labels(label_dir)
    subset = select_subset(counts, target_images, rare_class_ids, seed=seed)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text("\n".join(sorted(subset)) + "\n")
    table = subset_class_table(counts, subset, class_names)
    return {
        "images": len(subset),
        "instances": sum(table.values()),
        "per_class": table,
    }


if __name__ == "__main__":
    import argparse

    from bdd_to_yolo import CLASS_NAMES

    ap = argparse.ArgumentParser(description="Build a class-balanced training subset")
    ap.add_argument("--labels", required=True, help="labels_yolo/train directory")
    ap.add_argument("--out", required=True, help="output manifest .txt path")
    ap.add_argument("--target-images", type=int, default=TARGET_IMAGES)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    summary = build_subset(
        Path(args.labels), Path(args.out), args.target_images,
        RARE_CLASS_IDS, CLASS_NAMES, seed=args.seed,
    )
    print(f"images:    {summary['images']:,}")
    print(f"instances: {summary['instances']:,}")
    print("\nper class:")
    width = max((len(k) for k in summary["per_class"]), default=0)
    for name, n in sorted(summary["per_class"].items(), key=lambda kv: -kv[1]):
        share = 100 * n / max(1, summary["instances"])
        print(f"  {name:<{width}}  {n:>8,}  {share:5.1f}%")
