# Traffic Object Detection (BDD100K / YOLO11s)

A YOLO11s detector trained on real dashcam footage (BDD100K) to find
**people, vehicles, and traffic infrastructure** — built as a full
pipeline, not a notebook: label conversion, class-balanced subsampling,
a defended augmentation policy, training, evaluation tooling, and a
measured before/after improvement iteration, each with its own tests.

**Live demo:** _[add your Streamlit Community Cloud link here after deploying]_

---

## Results

Final model (`yolo11s`, `imgsz=960`, 100 epochs, class-balanced subset):

| metric | value |
|---|---|
| mAP50 | 0.657 |
| mAP50-95 | 0.385 |
| Precision | 0.698 |
| Recall | 0.599 |

Per class:

| class | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| car | 0.855 | 0.704 | 0.811 | 0.516 |
| traffic_sign | 0.746 | 0.628 | 0.696 | 0.378 |
| person | 0.791 | 0.574 | 0.675 | 0.349 |
| traffic_light | 0.732 | 0.595 | 0.664 | 0.260 |
| truck | 0.737 | 0.480 | 0.608 | 0.441 |
| bus | 0.448 | 0.679 | 0.604 | 0.474 |
| two_wheeler | 0.577 | 0.530 | 0.541 | 0.277 |

## What changed from the baseline

The first trained model (`imgsz=640`) scored mAP50 0.599 / mAP50-95 0.343.
Raising input resolution to `imgsz=960` improved **every single class**:

| class | mAP50 (640) | mAP50 (960) | Δ |
|---|---|---|---|
| car | 0.763 | 0.811 | +0.048 |
| truck | 0.559 | 0.608 | +0.049 |
| bus | 0.555 | 0.604 | +0.049 |
| two_wheeler | 0.485 | 0.541 | +0.056 |
| traffic_light | 0.602 | 0.664 | +0.062 |
| person | 0.608 | 0.675 | +0.067 |
| traffic_sign | 0.619 | 0.696 | +0.077 |

This was a targeted change: evaluation on the baseline flagged
`traffic_light` as having the largest gap between mAP50 and mAP50-95
(0.602 vs 0.221) — found, but not tightly localized, a classic small-object
symptom. The gain ended up broader than that single hypothesis predicted
(`traffic_sign` and `person` improved more than `traffic_light`), which is
itself worth noting rather than glossing over.

**Known open issue:** `bus` has an inverted precision/recall
(0.448 / 0.679) both before and after the resolution change, suggesting
confusion with `truck` (a classification problem, not a localization one —
consistent with resolution not fixing it).

---

## Why BDD100K

The brief called for vehicles, pedestrians, and traffic signs. BDD100K is
the only common driving dataset with all three as first-class categories,
and it's real dashcam footage across day/night and weather conditions —
which makes the metrics meaningful instead of trivially easy.

## Classes

BDD100K's 10 raw categories were mapped to 7 target classes:

| class id | name | source categories |
|---|---|---|
| 0 | person | pedestrian, rider |
| 1 | car | car |
| 2 | truck | truck |
| 3 | bus | bus |
| 4 | two_wheeler | bicycle/bike, motorcycle/motor |
| 5 | traffic_light | traffic light |
| 6 | traffic_sign | traffic sign |

`rider` merges into `person` rather than `two_wheeler`: BDD boxes the
human separately from the vehicle they're riding, so `rider` is
semantically a person, not a vehicle. `train`, `other vehicle`,
`other person`, and `trailer` are excluded — each has under 1,000
instances across the full 100k-image dataset, too rare to learn from and
liable to silently drag down class-averaged mAP.

## The pipeline

| step | file | what it does |
|---|---|---|
| 1. Label conversion | `src/bdd_to_yolo.py` | BDD100K JSON → YOLO `.txt` labels |
| 2. Class-balanced subsampling | `src/subsample.py` | Builds a 15,000-image training subset that guarantees every image containing a rare class (`bus`, `two_wheeler`) is included |
| 3. Dataset config + augmentation | `src/dataset_config.py` | Writes `data.yaml` and a defended, road-scene-specific augmentation policy |
| 4. Path wiring | `src/train_config.py` | Turns image-stem manifests into the full paths Ultralytics needs, working around Kaggle's read-only input mount |
| 5. Evaluation tooling | `src/evaluate.py` | Parses Ultralytics' per-class output, auto-flags precision/recall inversions and localization gaps |
| 6. Before/after comparison | `src/compare.py` | Diffs two evaluation runs class-by-class |

Each numbered `SESSION_0N.md` in this repo documents the reasoning behind
that step in more depth than this summary does.

### The class-imbalance problem

In the full dataset, `car` outnumbers `bus` by **~58x** and `two_wheeler`
by **~69x**. A random training subset of any size reproduces that same
ratio. `subsample.py` fixes this by protecting every image containing a
rare class — a fixed, absolute guarantee, not a percentage — then filling
the rest of the budget randomly. Result: `bus` went from 0.9% → 3.6% of
the training subset, `two_wheeler` from 0.8% → 3.0%, and both ended up
performing close to `truck` rather than trailing far behind it.

### The augmentation policy

Standard YOLO augmentation defaults are tuned for general photography, not
dashcam footage. A few deliberate departures:

- `flipud = 0` — a dashcam frame flipped upside down never happens in real
  driving.
- `degrees = 3.0`, `perspective = 0.0` — a windshield-mounted camera is
  near-level; the images already have real dashcam perspective baked in.
- `scale = 0.3` (below the 0.5 default) — Session 1 drops boxes under 4px
  as unlearnable noise; aggressive scale augmentation would push more
  `traffic_sign` boxes below that floor.
- `mosaic = 1.0` — known to help small/rare-object recall specifically,
  which is exactly the problem the class-balancing step was fighting.

---

## Running this yourself

```bash
git clone https://github.com/g248sing/traffic-detection.git
cd traffic-detection
pip install -r requirements.txt
python tests/test_bdd_conversion.py
python tests/test_subsample.py
python tests/test_dataset_config.py
python tests/test_train_config.py
python tests/test_evaluate.py
python tests/test_compare.py
```

All six test suites need no dataset or GPU — they run against synthetic
fixtures and finish in seconds. Training itself needs the BDD100K dataset
(see `notebooks/KAGGLE_SETUP.md` for the Kaggle setup this project was
built against) and a GPU.

## Running the demo locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires `best.pt` (the trained checkpoint) in the repo root — download it
from your training run's output directory. Unlike the rest of the repo,
`best.pt` **is** committed here (`.gitignore` has a `!best.pt` exception):
Streamlit Community Cloud deploys straight from this GitHub repo with no
separate weights-only artifact store, so the checkpoint has to live in
git for the hosted demo to work.
