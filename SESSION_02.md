# Session 2 — Class balance & subsampling

**Time:** ~45–60 min. **You need:** nothing downloaded — this only touches
label counts, not images, so it runs fine locally with synthetic data.
Building the real subset happens on Kaggle at the end, once tests pass.

---

## Where Session 1 left you

Real BDD100K train split, converted:

| class | instances | share |
|---|---|---|
| car | 700,212 | 55.1% |
| traffic_sign | 237,518 | 18.7% |
| traffic_light | 185,660 | 14.6% |
| person | 96,632 | 7.6% |
| truck | 27,884 | 2.2% |
| bus | 11,972 | 0.9% |
| two_wheeler | 10,142 | 0.8% |

69,863 images total.

## The problem, in two parts

**Size.** Training on all 69,863 images won't finish in a 12-hour Kaggle
session at a useful epoch count. You need a subset.

**Balance.** `car` outnumbers `bus`/`two_wheeler` by roughly 60–70x. A
*random* subset of any size reproduces that exact ratio — a 15,000-image
random sample still gives the model only ~120 bus examples to learn from
across however many epochs you run. The subset has to be deliberately
biased toward images containing the rare classes, or you're training a
car/sign/light detector that happens to also emit two other classes.

---

## Today's job

Open `src/subsample.py`. Three TODOs. Then:

```powershell
python tests/test_subsample.py
```

11 tests, currently 2 passing. Work until all 11 are green.

**Read the test bodies.** Same as Session 1 — they're the spec.

### TODO 1 — `RARE_CLASS_IDS` and `TARGET_IMAGES` (~15–20 min)

Two judgement calls:

- **Which classes get "protected"?** A protected class means every image
  containing it is guaranteed into the subset, regardless of the size
  budget. `bus` and `two_wheeler` are the obvious picks given the table
  above. Is `truck` (2.2%) rare enough to protect too? Protecting more
  classes spends more of your budget on guaranteed images, leaving less
  room for everything else.
- **What's `TARGET_IMAGES`?** Work it out from a real constraint: pick a
  model size, estimate a training step's speed on your GPU, and figure out
  how many images × epochs fits in ~12 hours. Don't take my word for
  throughput numbers — benchmark a few hundred steps on Kaggle first, same
  as Session 1 told you to confirm image size instead of trusting the
  docstring.

Tests only pin that `car` isn't protected and that `TARGET_IMAGES` is an
actual subset of the 69,863 train images — not a specific number.

### TODO 2 — `count_labels` (~15 min)

Read every `.txt` file in a label directory, count instances per class id.
Mechanical — the one thing to get right is that an *empty* file still needs
an entry in the result (empty Counter), not to be skipped. Those are the
negative examples Session 1's `convert_entry` deliberately produced.

### TODO 3 — `select_subset` (~20–25 min)

The actual algorithm. Docstring lays out the three steps: protect images
with a rare class, fill the rest of the budget with a reproducible random
sample of what's left, return everything if there isn't enough to fill the
budget. The subtle part is using `random.Random(seed)` (not the global
`random` module) over a *sorted* list — that's what makes two runs with the
same seed give you the same answer.

---

## When the tests go green

Push, then on Kaggle, using the `labels_yolo/train` directory Session 1's
cell 5 already wrote:

```python
!python src/subsample.py --labels /kaggle/working/labels_yolo/train \
    --out /kaggle/working/subset_train.txt --target-images <your TARGET_IMAGES>
```

**Paste that printed table back to me**, next to Session 1's full-dataset
table. Look specifically at what happened to `bus` and `two_wheeler`'s
*share* (not just their count) — that comparison is what tells you whether
the subsampling actually did its job, and it's the input to Session 3.

---

## Rules for this session

- Don't ask me for the answer before you've had a real go. Ask me *why*
  your attempt fails — better question, better answer.
- Stuck more than 15 minutes on one TODO? Ping me. Stuck isn't learning.
- Paste the test output when you want a review, even if it's all green.
