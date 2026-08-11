# Session 1 — Labels: BDD100K JSON → YOLO

**Time:** ~60–90 min. **You need:** no GPU, no dataset, nothing downloaded.

---

## The setup we settled on

**Code on GitHub → data on Kaggle → nothing on your laptop.**

You write `src/` files locally, push, and a six-cell Kaggle notebook clones and
runs them. See `notebooks/KAGGLE_SETUP.md`.

Two things this buys you beyond solving the disk problem:

- **16 GB VRAM instead of 4.** Everything I designed for the 1650 — batch 8,
  yolo11n only, no room for anything bigger — stops binding. You can train
  yolo11s or yolo11m at batch 16–32.
- **A repo, not a notebook.** "I have a Kaggle notebook" and "I have a tested
  repo that runs on two different machines" read very differently to whoever's
  hiring you.

### Why BDD100K

| Dataset | Vehicles | Pedestrians | Traffic **signs** |
|---|---|---|---|
| **BDD100K** | ✅ car/truck/bus | ✅ pedestrian/rider | ✅ **yes** |
| Udacity (Roboflow) | ✅ | ✅ | ❌ lights only |
| KITTI | ✅ | ✅ | ❌ none |

Only BDD has all three groups your CV bullet claims. It's also real dashcam
footage across day/night and rain/clear, which makes the metrics interesting
instead of "0.95, done".

### Do this before you start (5 min)

1. `git init` this folder, push to a **public** GitHub repo.
2. Kaggle account → **verify your phone number** (no verification, no GPU).
3. New Notebook → *Add Input* → search `bdd100k` → attach a version with the
   **raw `det_*.json` labels**, not a pre-converted YOLO one. You want to write
   the converter yourself; that's the part that teaches you how detection data
   works, and the part you can point at in an interview.

You do **not** need to download anything. Kaggle datasets mount read-only at
`/kaggle/input` and don't touch your disk.

---

## The plan (7 sessions)

| # | Session | Output |
|---|---|---|
| **1** | **Labels: JSON → YOLO** | **converter + passing tests** |
| 2 | Class balance & subsampling | a training set that fits a 12-hour session |
| 3 | Preprocessing & augmentation for road scenes | dataset config + a defended policy |
| 4 | Baseline training run | first real numbers |
| 5 | Evaluation: per-class P/R, mAP, PR curves | metrics table for the write-up |
| 6 | One improvement iteration | before/after comparison |
| 7 | Inference demo on HF Spaces + README | a public URL |

---

## Today's job

Open `src/bdd_to_yolo.py`. Three TODOs. Then:

```powershell
python tests/test_bdd_conversion.py
```

16 tests, currently 2 passing. Work until all 16 are green. This runs fine on
your laptop with no data — that's the point of testing the conversion logic
separately from the conversion run.

**Read the test bodies.** They're the spec, and more precise than my prose.

### TODO 1 — Design your classes (~20 min, mostly thinking)

BDD gives you 10 categories. You decide what they become.

The only decision today that isn't mechanical, and the one an interviewer will
actually ask about. Two questions to settle:

- **Split car / truck / bus, or merge them?** Splitting gives a more useful
  model. Merging gives better per-class numbers. Which is honest?
- **What is a "rider"** — a person on a bike? Person class, two-wheeler class,
  or its own? All defensible; none free.

Tests pin three properties: ids contiguous from 0, `train` excluded, signs ≠
lights. The rest is yours. Write a one-line comment saying *why* — you'll want
it in Session 7.

> On `train`: ~150 instances in 100,000 images. Ask yourself what mAP does when
> one class is a coin flip and you average over classes.

### TODO 2 — `box2d_to_yolo` (~20 min)

Corners → centre/size, normalised. Same convention as Penn-Fudan, dirtier data:
reversed corners, boxes past the frame edge, zero-area boxes.

The docstring gives the order of operations. That order matters — work out why
before you write it.

### TODO 3 — `convert_entry` (~20 min)

Walk one image's labels, skip what can't be used, emit YOLO lines.

Four skip conditions in the docstring. The subtle one: `min_box_px` is in
**pixels**, but `box2d_to_yolo` hands you **normalised** values.

---

## When the tests go green

Push, then run cells 1–5 of `notebooks/KAGGLE_SETUP.md`. Cell 5 converts the
real labels and prints a per-class instance count.

**Paste that table to me.** It's the input to Session 2, and its shape will
probably surprise you.

---

## What the other two files are for

- `src/paths.py` — finds the data whether you're local or on Kaggle. Every
  Kaggle upload of BDD nests folders differently, so it searches instead of
  hardcoding. Run `python src/paths.py` on Kaggle to see what it found.
- `src/extract_subset.py` — pulls a small image subset out of a zip. You no
  longer need it for disk reasons, but you'll want a ~200-image local subset
  for debugging, because waiting on a Kaggle session to discover a typo is a
  miserable loop.

---

## Rules for this session

- Don't ask me for the answer before you've had a real go. Ask me *why* your
  attempt fails — better question, better answer.
- Stuck more than 15 minutes on one TODO? Ping me. Stuck isn't learning.
- Paste the test output when you want a review, even if it's all green. There
  are ways to pass these tests that I'd still push back on.
