"""
SESSION 3 -- Dataset config + a defended augmentation policy for road scenes.

Sessions 1-2 gave you converted YOLO labels and a class-balanced training
subset. This session turns that into two YAML files Ultralytics YOLO
actually reads when you call model.train() in Session 4:

  configs/data.yaml           -- what the classes are, where images/labels live
  configs/hyp_road_scene.yaml -- how aggressively to augment during training

Your job is the three TODOs below. Check your work with:
    python tests/test_dataset_config.py

Wiring the exact real Kaggle paths into data.yaml happens in Session 4, once
you're actually on Kaggle and can see what paths.py resolved. Today is about
getting the config *shape* right, and making (and defending) the
augmentation choices.
"""

from __future__ import annotations

from pathlib import Path

import yaml


# ============================================================== TODO 1 ======
# Ultralytics YOLO augments every training image on the fly using these
# knobs -- the same names you'll pass to model.train() in Session 4.
# Defaults are tuned for general object detection (COCO-style photos from
# all kinds of angles). Dashcam footage is not that.
#
# Think about each of these for road scenes specifically, not in the
# abstract:
#   - flipud (vertical flip probability): a dashcam frame flipped upside
#     down puts the sky at the bottom. Does that ever happen in real
#     driving? What would it teach a traffic_light/traffic_sign detector,
#     whose context (sky above, road below) the network learns from
#     whether you intend it to or not?
#   - degrees (rotation): how far off-level can a windshield-mounted camera
#     realistically be?
#   - perspective: the images already have real dashcam perspective baked
#     in. What does adding synthetic perspective warp on top do to a class
#     like traffic_sign, which is partly a SHAPE (rectangle/diamond/circle)?
#   - scale: Session 1 dropped boxes under 4px because they're unlearnable
#     noise. If training-time scale augmentation shrinks images further,
#     what happens to a traffic_sign box that started at, say, 6px?
#   - mosaic (stitches 4 images into 1): known to help small-object recall
#     in YOLO specifically, because it changes an object's relative size in
#     the frame. Given how rare/small your traffic_sign and two_wheeler
#     instances are, is this worth keeping on?
#   - hsv_v (brightness/value jitter): BDD100K is explicitly day AND night,
#     clear AND rainy. Does that argue for MORE brightness jitter than a
#     dataset shot in consistent lighting, or less?
#
# Fill in every key below (all are required -- see the tests). Tests only
# check ranges/sanity (e.g. flipud must be exactly 0), not that you picked
# the "right" number -- but you'll defend these in the Session 7 write-up,
# so leave yourself a comment on why each departs from (or matches) the
# Ultralytics default.

AUGMENTATION_POLICY: dict[str, float] = {
    # Colour/lighting: BDD100K spans day/night and clear/rain, so extra
    # brightness jitter helps (hsv_v raised above the 0.4 default). Hue is
    # kept low -- overly aggressive hue shifts risk turning a real red
    # traffic light into something that reads as green in pixel space,
    # which is a confusing signal to train on even though this project
    # doesn't classify light state.
    "hsv_h": 0.010,
    "hsv_s": 0.6,
    "hsv_v": 0.5,

    # Geometry: a windshield-mounted camera is close to level and doesn't
    # tumble, so rotation/shear stay small. Perspective stays at Ultralytics'
    # minimum -- the images already contain real dashcam perspective, adding
    # synthetic warp on top risks distorting traffic_sign's shape, which is
    # part of what identifies it.
    "degrees": 3.0,
    "translate": 0.1,
    "shear": 1.0,
    "perspective": 0.0,

    # Scale: kept moderate, not aggressive. Session 1 dropped boxes under
    # 4px as unlearnable noise -- shrinking images further at training time
    # would push more traffic_sign boxes below that floor. 0.3 (vs the 0.5
    # default) trades some scale-invariance robustness for keeping more
    # small signs learnable.
    "scale": 0.3,

    # Flips: flipud MUST be 0 -- an upside-down dashcam frame with the sky
    # at the bottom never happens in real driving, and would teach the
    # network a scene layout that actively misleads it. fliplr stays at
    # the default -- roads are left/right symmetric enough that mirroring
    # is realistic (a car doesn't stop looking like a car mirrored).
    "flipud": 0.0,
    "fliplr": 0.5,

    # Mosaic/mixup: mosaic stays on (and high) specifically because it's
    # known to help small/rare-object recall in YOLO -- exactly the
    # problem Session 2 was fighting for two_wheeler and bus. mixup
    # (blending two whole images) is turned off: it can create physically
    # nonsensical overlapping scenes, which is a bigger risk for already-rare
    # classes than the regularisation benefit is worth. copy_paste is a
    # segmentation-oriented augmentation and doesn't apply to box-only data.
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
}


# ============================================================== TODO 2 ======
def build_data_yaml(class_names: list[str], path: str, train: str, val: str) -> dict:
    """Build the dict that becomes configs/data.yaml.

    Ultralytics' dataset YAML shape:
        path:  <dataset root>          -- train/val resolve relative to this
        train: <images dir, or a .txt list of image paths>
        val:   <same, for validation>
        nc:    <number of classes>
        names: [<class 0 name>, <class 1 name>, ...]   -- order matters,
               must match the class ids your labels actually use

    `class_names`, `path`, `train`, `val` map directly onto that shape --
    this function has no logic to speak of, just assembly.
    """
    return {
        "path": path,
        "train": train,
        "val": val,
        "nc": len(class_names),
        "names": list(class_names),
    }


# ============================================================== TODO 3 ======
def write_yaml(data: dict, out_path: Path) -> None:
    """Write `data` to `out_path` as YAML, creating parent directories.

    Use yaml.safe_dump. Pass sort_keys=False -- these files get read by
    humans (including you, debugging a training run at 2am), and
    path/train/val/nc/names in a sensible order is easier to scan than
    alphabetical.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(data, sort_keys=False))


if __name__ == "__main__":
    import argparse

    from bdd_to_yolo import CLASS_NAMES

    ap = argparse.ArgumentParser(
        description="Write configs/data.yaml and configs/hyp_road_scene.yaml"
    )
    ap.add_argument("--path", default="/kaggle/working",
                     help="dataset root Ultralytics resolves train/val against")
    ap.add_argument("--train", default="subset_train.txt",
                     help="train images path/manifest, relative to --path")
    ap.add_argument("--val", default="images/100k/val",
                     help="val images dir, relative to --path")
    ap.add_argument("--out-dir", default="configs", help="where to write the yaml files")
    args = ap.parse_args()

    data = build_data_yaml(CLASS_NAMES, args.path, args.train, args.val)
    write_yaml(data, Path(args.out_dir) / "data.yaml")
    write_yaml(AUGMENTATION_POLICY, Path(args.out_dir) / "hyp_road_scene.yaml")
    print(f"wrote {args.out_dir}/data.yaml and {args.out_dir}/hyp_road_scene.yaml")
