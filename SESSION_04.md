# Session 4 — Baseline training run

**Time:** ~1–2 hrs active work, then a training run that eats most of your
12-hour Kaggle session budget in the background. **You need:** Kaggle, GPU
on, everything from Sessions 1–3 pushed.

---

## Where Sessions 1–3 left you

- `src/bdd_to_yolo.py` — labels converted, 7 classes.
- `src/subsample.py` — a 15,000-image training subset, rare classes protected.
- `src/dataset_config.py` — `configs/data.yaml` + `configs/hyp_road_scene.yaml`,
  with placeholder paths (`/kaggle/working`, `subset_train.txt`) since
  Session 3 didn't have real Kaggle paths to work with yet.

This session closes that last gap and actually trains something.

## The problem: Ultralytics can't find your labels

Ultralytics discovers a label file by taking an image's path and swapping
`images` for `labels` in it — same directory depth, same filename, `.txt`
instead of `.jpg`. Your labels live at `/kaggle/working/labels_yolo/train/`.
BDD's real images live somewhere under `/kaggle/input/...` — read-only, and
nowhere near that shape. Ultralytics' auto-pairing physically cannot work
against that layout, and you can't write a `labels/` folder inside
`/kaggle/input` to fix it.

The fix: build a **parallel directory under `/kaggle/working`** where
`images/` is a symlink to the real Kaggle input folder and `labels/` is a
symlink to your converted labels — sitting side by side, so the
images→labels swap resolves correctly. No copying: a symlink to a whole
70,000-image folder is one syscall, not 70,000.

This is a one-time Kaggle notebook operation, not something to unit test
locally (Windows doesn't handle symlinks the same way, and there's no real
Kaggle filesystem to test against off-platform). The part that *is*
testable locally is turning Session 2's subset manifest into the exact
image paths Ultralytics needs once those symlinks exist.

---

## Today's job (local, no Kaggle needed for this part)

Open `src/train_config.py`. Two TODOs, both mechanical — no design calls
left, Sessions 1–3 already made those. Then:

```powershell
python tests/test_train_config.py
```

6 tests, currently 0 passing.

### TODO 1 — `stems_to_image_paths`

Turn each stem in Session 2's manifest into a full image path under
whatever `image_dir` you're given. One gotcha: build the path with
forward slashes regardless of what OS you run this on locally — the file
gets read by Ultralytics on Kaggle (Linux), and a Windows-style path with
backslashes would silently fail to resolve there. `Path.as_posix()` is
the tool for this.

### TODO 2 — `write_image_list`

Write the path list to a file, one per line, creating parent directories
if needed. Same shape as `write_yaml` from Session 3, just for plain text
instead of YAML.

---

## When the tests go green: on Kaggle

Re-run cell 1 (clone), then walk through these in order.

### Cell A — symlink the writable dataset layout

```python
import sys; sys.path.insert(0, "src")
from pathlib import Path
from paths import find_image_dir, output_root

root = output_root() / "dataset"
for split in ("train", "val"):
    img_link = root / "images" / "100k" / split
    lbl_link = root / "labels" / "100k" / split
    img_link.parent.mkdir(parents=True, exist_ok=True)
    lbl_link.parent.mkdir(parents=True, exist_ok=True)
    if not img_link.exists():
        img_link.symlink_to(find_image_dir(split), target_is_directory=True)
    if not lbl_link.exists():
        lbl_link.symlink_to(output_root() / "labels_yolo" / split, target_is_directory=True)
print("linked under:", root)
```

### Cell B — build the real train image-path list

```python
!python src/train_config.py \
    --manifest /kaggle/working/subset_train.txt \
    --image-dir /kaggle/working/dataset/images/100k/train \
    --out /kaggle/working/dataset/train_images.txt
```

(`val` doesn't need this — Session 2 only subsampled `train`. Point `val:`
straight at the symlinked `images/100k/val` directory; you want the full,
honest validation set, not a rebalanced one.)

### Cell C — regenerate data.yaml with the real paths

Reuses Session 3's script directly:

```python
!python src/dataset_config.py \
    --path /kaggle/working/dataset \
    --train train_images.txt \
    --val images/100k/val \
    --out-dir /kaggle/working/configs
```

Print `/kaggle/working/configs/data.yaml` and sanity-check it before
moving on — this is exactly the kind of thing worth eyeballing rather than
trusting blindly.

### Cell D — train

```python
from ultralytics import YOLO
import yaml

hyp = yaml.safe_load(open("configs/hyp_road_scene.yaml"))
model = YOLO("yolo11s.pt")

results = model.train(
    data="/kaggle/working/configs/data.yaml",
    epochs=100,
    batch=32,
    imgsz=640,
    device="0,1",
    **hyp,
)
```

Watch the first few epochs complete before walking away — if something in
the path wiring is wrong, it usually shows up immediately as "0 images
found" or similar, not after an hour.

---

## Paste back when it's done

- Final epoch's overall `mAP50` and `mAP50-95`
- Per-class `mAP50` if the training log shows it (it should, under
  `results.box`) — this is the number that tells you whether Session 2's
  subsampling actually helped `bus`/`two_wheeler`, or just moved the
  problem around
- How long the run actually took, versus the ~8.3h estimate from Session 2

That table is the input to Session 5 (evaluation: per-class P/R, mAP, PR
curves) — first real numbers, first real read on whether any of this
worked.

---

## Rules for this session

- Don't ask me for the answer before you've had a real go. Ask me *why*
  your attempt fails — better question, better answer.
- Stuck more than 15 minutes on one TODO? Ping me. Stuck isn't learning.
- Paste the test output when you want a review, even if it's all green.
