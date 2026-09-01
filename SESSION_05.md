# Session 5 — Evaluation: per-class P/R, mAP, PR curves

**Time:** ~30–40 min. **You need:** nothing downloaded — this runs
entirely locally on the per-class table Session 4 already gave you. No
Kaggle, no GPU.

---

## Where Session 4 left you

A baseline `yolo11s` run, 100 epochs, 5.712 hours. Overall: P=0.681,
R=0.552, mAP50=0.599, mAP50-95=0.343. Per-class:

| class | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| car | 0.837 | 0.671 | 0.763 | 0.479 |
| traffic_sign | 0.732 | 0.547 | 0.619 | 0.321 |
| person | 0.780 | 0.509 | 0.608 | 0.301 |
| traffic_light | 0.707 | 0.557 | 0.602 | 0.221 |
| truck | 0.734 | 0.437 | 0.559 | 0.400 |
| bus | 0.436 | 0.645 | 0.555 | 0.434 |
| two_wheeler | 0.542 | 0.500 | 0.485 | 0.245 |

We already read two things out of this by hand: `bus` has an inverted
P/R (low precision, high recall — likely confused with `truck`), and
`traffic_light` has the biggest mAP50→mAP50-95 drop (found, not tightly
boxed). Doing that by eye works once. This session makes it repeatable —
useful again in Session 6 when you check whether an improvement actually
fixed either problem, not just moved the numbers around.

---

## Today's job

Open `src/evaluate.py`. Three TODOs. Then:

```powershell
python tests/test_evaluate.py
```

9 tests, currently 0 passing.

### TODO 1 — `PR_GAP_THRESHOLD` and `LOC_GAP_THRESHOLD` (~10 min)

Two calibration numbers. Too low and everything gets flagged (noise); too
high and you miss real problems. Use Session 4's actual numbers as your
calibration set — `bus`'s P/R gap is 0.209, `traffic_light`'s mAP gap is
0.381. Pick thresholds that would catch those two without flagging every
class with an unremarkable gap.

### TODO 2 — `parse_class_table` (~10–15 min)

Turn Ultralytics' pasted console table into structured data. Mechanical,
with one real gotcha: the pasted text always has extra junk mixed in — a
header row, the `all` summary row, sometimes stray whitespace or
progress-bar fragments if you copy sloppily. A line that doesn't cleanly
parse as `class images instances P R mAP50 mAP50-95` should just be
skipped, not crash the whole parse.

### TODO 3 — `flag_anomalies` (~10 min)

Apply the two thresholds from TODO 1 to every row, using the two rules laid
out in the docstring. Straightforward once TODO 1 and TODO 2 are done —
this is where the two judgement calls from TODO 1 actually get used.

---

## When the tests go green

Run it on Session 4's real output — save the per-class table (the block
you already pasted me) to a text file and:

```powershell
python src/evaluate.py --input session4_output.txt
```

You should see `bus` and `traffic_light` flagged, and a markdown table
sorted weakest-class-first. That table is what goes straight into the
Session 7 write-up.

**One more thing worth doing while you're in here**: Ultralytics already
saved a confusion matrix to your Session 4 run directory
(`runs/detect/train/confusion_matrix.png`) — download it and take a look.
The `bus` P/R inversion is a strong hint the model is mixing up `bus` and
`truck`; the confusion matrix will confirm or deny that directly. Tell me
what you see.

---

## Rules for this session

- Don't ask me for the answer before you've had a real go. Ask me *why*
  your attempt fails — better question, better answer.
- Stuck more than 15 minutes on one TODO? Ping me. Stuck isn't learning.
- Paste the test output when you want a review, even if it's all green.
