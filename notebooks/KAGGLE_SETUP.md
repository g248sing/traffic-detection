# Running this repo on Kaggle

The idea: **code lives in GitHub, data lives in Kaggle, nothing lives on your
laptop.** The notebook is a thin launcher — six cells, no logic. All the real
code stays in `src/` where it's reviewable, testable, and shows up on your
GitHub profile.

That last part matters. "I have a notebook" and "I have a repo with tests that
runs on two different machines" read very differently to whoever's hiring you.

---

## One-time setup

1. **Push this repo to GitHub** (public, so Kaggle can clone it without a token).
2. **Kaggle account** → verify your phone number. Without that, no GPU.
3. **New Notebook** → right sidebar:
   - *Add Input* → search `bdd100k` → add a dataset with the **raw JSON labels**
     (the `det_*.json` files, not a pre-converted YOLO one)
   - *Accelerator* → **GPU T4 x2** or **P100**
   - *Internet* → **On** (needed to clone and to pull pretrained weights)

Free quota is roughly 30 GPU-hours/week and sessions stop at ~12 hours. Plan
training runs that finish inside a session, and save weights to
`/kaggle/working` — that's what survives as notebook output.

---

## The six cells

### 1 — Clone

```python
!rm -rf /kaggle/working/traffic-detection
!git clone -q https://github.com/YOURNAME/traffic-detection.git /kaggle/working/traffic-detection
%cd /kaggle/working/traffic-detection
```

Re-run this cell after every push. It's how you get your edits onto Kaggle —
you edit locally, push, re-clone. Never edit code in the notebook.

### 2 — Check what the machine has

```python
!nvidia-smi --query-gpu=name,memory.total --format=csv
!python src/paths.py
```

`paths.py` searches `/kaggle/input` for the label JSON and the image folders,
because every Kaggle upload of BDD nests things differently. If it prints
`NOT FOUND`, the dataset isn't attached, or its filenames differ — paste the
output to me and we'll add the pattern.

### 3 — Install

```python
!pip install -q ultralytics
```

torch is already on Kaggle images, and it's a CUDA build. Don't reinstall it.

### 4 — Run your tests

```python
!python tests/test_bdd_conversion.py
```

Yes, on Kaggle too. If the tests don't pass here, nothing downstream is
trustworthy — and this catches the "works on my machine" class of bug for free.

### 5 — Convert labels

```python
import sys; sys.path.insert(0, "src")
from paths import find_label_json, output_root
from bdd_to_yolo import convert_split

for split in ("train", "val"):
    summary = convert_split(
        find_label_json(split),
        output_root() / "labels_yolo" / split,
    )
    print(split, summary["images"], "images,", summary["instances"], "instances")
```

### 6 — Train

Comes in Session 4. Don't skip ahead — training on data you haven't looked at
is how you burn six GPU-hours learning nothing.

---

## Things that will catch you out

| Symptom | Cause |
|---|---|
| `Read-only file system` | You wrote to `/kaggle/input`. Use `output_root()`. |
| `paths.py` says NOT FOUND | Dataset not attached, or different filenames |
| No GPU in `nvidia-smi` | Accelerator not set, or phone not verified |
| `git clone` fails | Internet toggle is off, or the repo is private |
| Your fix didn't apply | You edited in the notebook instead of pushing. Re-run cell 1. |
| Session died, weights gone | Only `/kaggle/working` is kept, and only via *Save Version* |

## Local still matters

Keep a ~200-image subset on your laptop for debugging. Waiting on a Kaggle
session to find out you had a typo is a miserable loop. Local for "does it
run", Kaggle for "does it learn".

`src/paths.py` is what lets the identical code do both — it detects the
environment and finds the data wherever it is. Nothing in `src/` should ever
contain a `/kaggle/` path.
