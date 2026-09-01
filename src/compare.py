"""
SESSION 6 -- Before/after: did the one change you made actually help?

Session 5 gave you evaluate.parse_class_table to turn Ultralytics' console
output into structured rows, and flag_anomalies to call out the two
specific problems in the baseline: bus's precision/recall inversion, and
traffic_light's localization gap (mAP50 vs mAP50-95).

Session 6 picks ONE change targeting one of those findings, reruns
training, and this file answers the only question that matters afterward:
did the specific class you targeted actually improve, or did the number
just move around?

Check your work with:
    python tests/test_compare.py
"""

from __future__ import annotations

from evaluate import parse_class_table  # noqa -- reused, not reimplemented


# ============================================================== TODO 1 ======
def compare_runs(before: list[dict], after: list[dict]) -> list[dict]:
    """Match rows from two runs by class name and compute deltas.

    Returns one dict per class present in BOTH before and after (a class
    that only exists in one run can't be compared, so it's dropped):
        {"class": "bus",
         "P_before": .., "P_after": .., "P_delta": ..,
         "R_before": .., "R_after": .., "R_delta": ..,
         "mAP50_before": .., "mAP50_after": .., "mAP50_delta": ..,
         "mAP50_95_before": .., "mAP50_95_after": .., "mAP50_95_delta": ..}

    delta = after - before for every metric, so a positive delta always
    means "got better."
    """
    after_by_class = {r["class"]: r for r in after}
    result = []
    for b in before:
        a = after_by_class.get(b["class"])
        if a is None:
            continue
        row = {"class": b["class"]}
        for metric in ("P", "R", "mAP50", "mAP50_95"):
            row[f"{metric}_before"] = b[metric]
            row[f"{metric}_after"] = a[metric]
            row[f"{metric}_delta"] = a[metric] - b[metric]
        result.append(row)
    return result


# ============================================================== TODO 2 ======
def format_comparison_table(rows: list[dict]) -> str:
    """Render compare_runs' output as a markdown table, sorted by
    mAP50_delta ascending -- classes that got WORSE (or improved least)
    float to the top, since a regression is more urgent to notice than
    a class that was already fine and stayed fine.

    Columns: Class | mAP50 before | mAP50 after | Δ mAP50 | Δ mAP50-95
    Format the delta with an explicit sign (+0.012 / -0.004) so a
    reader can tell direction at a glance without doing subtraction
    in their head.
    """
    ordered = sorted(rows, key=lambda r: r["mAP50_delta"])
    lines = [
        "| Class | mAP50 before | mAP50 after | Δ mAP50 | Δ mAP50-95 |",
        "|---|---|---|---|---|",
    ]
    for r in ordered:
        lines.append(
            f"| {r['class']} | {r['mAP50_before']:.3f} | {r['mAP50_after']:.3f} | "
            f"{r['mAP50_delta']:+.3f} | {r['mAP50_95_delta']:+.3f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(
        description="Compare two Ultralytics per-class tables (before/after an improvement)"
    )
    ap.add_argument("--before", required=True, help="text file, baseline per-class table")
    ap.add_argument("--after", required=True, help="text file, new run's per-class table")
    args = ap.parse_args()

    before_rows = parse_class_table(Path(args.before).read_text())
    after_rows = parse_class_table(Path(args.after).read_text())
    comparison = compare_runs(before_rows, after_rows)
    print(format_comparison_table(comparison))
