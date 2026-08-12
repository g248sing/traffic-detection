# Session 3 — Preprocessing & augmentation for road scenes

**Time:** ~45–60 min, mostly thinking. **You need:** nothing downloaded —
this is pure config, no dataset or GPU required. Runs entirely locally.

---

## Where Sessions 1–2 left you

- `src/bdd_to_yolo.py` — converts BDD100K JSON into YOLO `.txt` labels for
  7 classes (person, car, truck, bus, two_wheeler, traffic_light,
  traffic_sign).
- `src/subsample.py` — picks a 15,000-image training subset that protects
  every image containing a `bus` or `two_wheeler` instance, since those
  are each under 1% of the full dataset.

Both of those are *what data goes in*. This session is *how it gets used*
during training: the dataset config Ultralytics reads, and the
augmentation policy applied to every image on the fly.

## Why this session is different from 1 and 2

Sessions 1–2 were mostly mechanical once the design calls were made — a
box format has a right answer, a subsampling algorithm either protects
rare images or it doesn't. Augmentation doesn't work like that. Every knob
is a judgement call about what kinds of image variation are *realistic*
for a dashcam, and getting it wrong doesn't crash anything — it just quietly
trains a worse model, or teaches it to expect scenes that never happen in
real driving. This is the "defended policy" the original plan calls for:
you need an actual reason for each number, not a default you didn't look at.

---

## Today's job

Open `src/dataset_config.py`. Three TODOs. Then:

```powershell
python tests/test_dataset_config.py
```

8 tests, currently 0 passing.

### TODO 1 — `AUGMENTATION_POLICY` (~25–30 min, mostly thinking)

Thirteen Ultralytics augmentation hyperparameters, all required. The
docstring in the file walks through six of them with road-scene-specific
questions — read those before you fill anything in. The short version of
what's at stake:

- A vertical flip (`flipud`) makes a dashcam frame upside down. That
  never happens in real driving, so the test pins this one exactly: it
  must be `0`.
- Rotation (`degrees`) and perspective warp are both bounded loosely by
  the tests (a windshield camera doesn't tumble, and the images already
  have real perspective in them) — but the exact numbers are your call.
- `scale` interacts directly with Session 1's `min_box_px = 4` decision:
  shrink the image further at training time and a 6px sign can drop below
  what's learnable. Worth thinking about, not pinned by a test.
- `mosaic` is a real lever for small/rare-object recall in YOLO
  specifically — is that worth it here, given your class table?

### TODO 2 — `build_data_yaml` (~10 min)

Pure assembly: map `class_names`/`path`/`train`/`val` onto the dict shape
Ultralytics expects (`path`, `train`, `val`, `nc`, `names`). No judgement
calls, just get the keys and structure right.

### TODO 3 — `write_yaml` (~10 min)

Write a dict to a YAML file, creating parent directories if they don't
exist. `yaml.safe_dump(data, sort_keys=False)` — the `sort_keys=False`
matters for readability (path/train/val/nc/names in that order, not
alphabetical), even though the tests only check the content round-trips,
not the key order.

---

## When the tests go green

Run the file's `__main__` locally — it doesn't need Kaggle or the dataset:

```powershell
python src/dataset_config.py
```

That writes `configs/data.yaml` and `configs/hyp_road_scene.yaml` using
placeholder paths (`/kaggle/working`, etc.). Take a look at both files and
sanity-check they read the way you'd expect.

**Real path wiring is Session 4's job**, not today's — you'll only know
the exact resolved Kaggle paths (from `paths.py`) once you're actually
there running training. Today's output is the config *shape* plus a
defended set of numbers, both of which are portable regardless of the
exact paths.

Push your changes, then paste me:
1. The test output (should be 8/8)
2. Your filled-in `AUGMENTATION_POLICY` values
3. A one-or-two-line reason for the two or three choices you think are
   least obvious — that's the part worth reviewing before Session 4
   actually trains on it.

---

## Rules for this session

- Don't ask me for the answer before you've had a real go. Ask me *why*
  your attempt fails — better question, better answer.
- Stuck more than 15 minutes on one TODO? Ping me. Stuck isn't learning.
- Paste the test output when you want a review, even if it's all green.
