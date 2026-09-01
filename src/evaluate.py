"""
SESSION 5 -- Evaluation: read what the model actually did wrong.

Session 4 trained a baseline and gave you a per-class table pasted straight
from Ultralytics' console output. Two numbers per class tell two different
stories, and it's easy to eyeball them once but tedious to keep doing by
hand every time you retrain:

  - Precision vs Recall: a class where R is much higher than P usually
    means the model is OVER-predicting that class -- often because it's
    confusing it with something that looks similar (e.g. bus vs truck).
  - mAP50 vs mAP50-95: a class where this gap is large means the model
    finds the object (clears the loose 50%-overlap bar) but doesn't box
    it tightly (fails the stricter 55-95% bars) -- usually a small-object
    localization problem, not a classification one.

This file turns that manual reading into reusable code: parse Ultralytics'
console output, flag classes worth a closer look, and format a clean table
for the Session 7 write-up.

Check your work with:
    python tests/test_evaluate.py
"""

from __future__ import annotations

from pathlib import Path


# ============================================================== TODO 1 ======
# Two thresholds, both judgement calls -- how big a gap is worth flagging?
# Too low and every class gets flagged (noise); too high and you miss real
# problems. Look at Session 4's actual numbers to calibrate:
#
#   bus:           P=0.436  R=0.645   (gap 0.209)
#   traffic_light: mAP50=0.602  mAP50-95=0.221   (gap 0.381)
#
# Those two are the clearest cases in this run -- pick thresholds that would
# catch them without flagging every class with a small, unremarkable gap.

PR_GAP_THRESHOLD: float = 0.1     # catches bus (0.209), ignores routine gaps
LOC_GAP_THRESHOLD: float = 0.35   # catches traffic_light (0.381) specifically


# ============================================================== TODO 2 ======
def parse_class_table(text: str) -> list[dict]:
    """Parse Ultralytics' per-class validation table (pasted straight from
    the console) into a list of dicts, one per class.

    A data row looks like:
        car       9882     102799      0.837      0.671      0.763      0.479
    Columns, in order: class, images, instances, P, R, mAP50, mAP50-95

    Returns [{"class": "car", "images": 9882, "instances": 102799,
              "P": 0.837, "R": 0.671, "mAP50": 0.763, "mAP50_95": 0.479}, ...]

    Skip the "all" summary row (it's not a class) and the header row.
    A line that doesn't parse cleanly as one class row should just be
    skipped, not raise -- pasted console output always has extra lines
    (headers, blanks, progress-bar text) mixed in.
    """
    rows = []
    for line in text.strip().splitlines():
        parts = line.split()
        if len(parts) != 7:
            continue
        name = parts[0]
        if name == "all":
            continue
        try:
            images, instances = int(parts[1]), int(parts[2])
            p, r, map50, map5095 = (float(x) for x in parts[3:7])
        except ValueError:
            continue
        rows.append({"class": name, "images": images, "instances": instances,
                     "P": p, "R": r, "mAP50": map50, "mAP50_95": map5095})
    return rows


# ============================================================== TODO 3 ======
def flag_anomalies(rows: list[dict]) -> list[str]:
    """Check every row against PR_GAP_THRESHOLD and LOC_GAP_THRESHOLD,
    returning one human-readable string per issue found.

    For each row:
      - if (R - P) > PR_GAP_THRESHOLD: flag a precision/recall inversion
      - if (mAP50 - mAP50_95) > LOC_GAP_THRESHOLD: flag a localization gap

    A class can produce zero, one, or two flags. Message content isn't
    pinned by the tests beyond containing the class name -- write whatever
    you'd actually want to read in a write-up.
    """
    flags = []
    for row in rows:
        pr_gap = row["R"] - row["P"]
        if pr_gap > PR_GAP_THRESHOLD:
            flags.append(
                f"{row['class']}: R ({row['R']:.3f}) exceeds P ({row['P']:.3f}) "
                f"by {pr_gap:.3f} -- likely over-predicted, possibly confused "
                f"with a visually similar class"
            )
        loc_gap = row["mAP50"] - row["mAP50_95"]
        if loc_gap > LOC_GAP_THRESHOLD:
            flags.append(
                f"{row['class']}: mAP50-mAP50_95 gap is {loc_gap:.3f} -- "
                f"found but not tightly localized"
            )
    return flags


# ---------------------------------------------------------------------------
# Already written -- formats parsed rows as a markdown table, weakest class
# (lowest mAP50) first, since that's the one worth discussing.
# ---------------------------------------------------------------------------
def format_markdown_table(rows: list[dict]) -> str:
    ordered = sorted(rows, key=lambda r: r["mAP50"])
    lines = [
        "| Class | Images | Instances | P | R | mAP50 | mAP50-95 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in ordered:
        lines.append(
            f"| {r['class']} | {r['images']:,} | {r['instances']:,} | "
            f"{r['P']:.3f} | {r['R']:.3f} | {r['mAP50']:.3f} | {r['mAP50_95']:.3f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Turn a pasted Ultralytics per-class table into flags + a markdown table"
    )
    ap.add_argument("--input", required=True,
                     help="text file containing the pasted per-class validation table")
    args = ap.parse_args()

    text = Path(args.input).read_text()
    rows = parse_class_table(text)
    print(f"parsed {len(rows)} classes\n")

    flags = flag_anomalies(rows)
    print("Flags:")
    for f in flags:
        print(" -", f)
    if not flags:
        print(" (none)")

    print("\nMarkdown table:\n")
    print(format_markdown_table(rows))
