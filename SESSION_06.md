# Session 6 — One improvement iteration

**Time:** TODOs ~20 min locally; the retrain itself is another Kaggle
session (~6-13h depending on how `imgsz` affects your throughput).

---

## Where Session 5 left you

Two concrete, evidence-backed findings from the baseline:
- `bus` has an inverted P/R (0.436 / 0.645) — likely confused with `truck`.
- `traffic_light` has the largest mAP50→mAP50-95 gap (0.602 → 0.221) —
  found, not tightly localized.

## The change: `imgsz` 640 → 960

Targets the localization-gap finding specifically. Small objects (traffic
lights, distant signs) get more actual pixels to work with at higher input
resolution, which should narrow the mAP50/mAP50-95 gap on exactly the
classes where it's currently worst — without touching class balance at all,
so it's a clean, single-variable test of one hypothesis.

**Why not just increase `TARGET_IMAGES`**, even though there's GPU-hour
budget left over (5.7h used of 12h)? Session 2's rare-class protection is
an absolute count of images, not a percentage — growing the subset only
grows the random-fill portion, which would *dilute* bus/two_wheeler's
share back toward their natural (<1%) baseline. That's the opposite of
what Session 2 was for. Worth knowing this trap exists before reaching for
"more data" as a default fix.

**Trade-off**: `imgsz=960` costs roughly (960/640)² ≈ 2.25x more compute
per image than the baseline. Budget for a longer run.

---

## Today's job (local, no Kaggle needed for this part)

Open `src/compare.py`. Two TODOs, both mechanical — reuses
`evaluate.parse_class_table` from Session 5 rather than re-parsing
anything. Then:

```powershell
python tests/test_compare.py
```

5 tests, currently 0 passing.

### TODO 1 — `compare_runs`

Match before/after rows by class name, compute `after - before` for P, R,
mAP50, mAP50-95. A class that only appears in one run (shouldn't happen in
practice, since both runs use the same fixed val set, but worth handling)
gets dropped rather than crashing.

### TODO 2 — `format_comparison_table`

Markdown table, sorted so the class that improved *least* (or regressed)
appears first — that's the one worth a second look, not the one that was
already fine. Explicit `+`/`-` signs on the deltas.

---

## On Kaggle: rerun with the one change

Same pipeline as Session 4 (re-clone, re-convert labels, rebuild subset,
symlinks, manifests, data.yaml — all of it gets wiped on session
restart, same as before). The only difference in Cell D:

```python
results = model.train(
    data="/kaggle/working/configs/data.yaml",
    epochs=100,
    batch=32,
    imgsz=960,      # was 640
    device="0,1",
    **hyp,
)
```

Everything else — the subset, the augmentation policy, the class map —
stays identical to Session 4. That's what makes this a controlled
comparison instead of a pile of confounded changes.

---

## When training finishes

Save both per-class tables to text files (Session 4's original output,
and this run's), then:

```powershell
python src/compare.py --before session4_output.txt --after session6_output.txt
```

**Paste that comparison table back to me.** Specifically look at whether
`traffic_light`'s mAP50-95 delta is meaningfully positive (the hypothesis
being tested) and whether anything regressed that wasn't supposed to move
at all — a real before/after check should surprise you at least a little,
or you didn't look hard enough.

---

## Rules for this session

- Don't ask me for the answer before you've had a real go. Ask me *why*
  your attempt fails — better question, better answer.
- Stuck more than 15 minutes on one TODO? Ping me. Stuck isn't learning.
- Paste the test output when you want a review, even if it's all green.
