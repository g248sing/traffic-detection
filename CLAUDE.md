# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A YOLO11s object detector trained on BDD100K dashcam footage, built as a
full pipeline rather than a notebook. The project is organized as seven
sequential sessions (`SESSION_01.md` through `SESSION_07.md`), each
documenting the reasoning behind one pipeline stage. Read the relevant
`SESSION_0N.md` before touching the corresponding `src/` file — the
design decisions (why a class was merged a certain way, why an
augmentation default was overridden) live there, not in code comments.

Three platforms, deliberately separated:
- **GitHub** (this repo) — code, tests, docs, and (unusually) the demo's
  trained weights (`best.pt`) — see "The demo" below for why.
- **Kaggle** — dataset storage and all training/GPU work.
- **Streamlit Community Cloud** — the public inference demo, deployed
  directly from this repo (`app.py`).

## Commands

Run each test suite directly with `python` (not pytest — these are
self-contained scripts with a hand-rolled pass/fail harness). All are
local-only: no dataset, no GPU, finish in seconds.

```
python tests/test_bdd_conversion.py
python tests/test_subsample.py
python tests/test_dataset_config.py
python tests/test_train_config.py
python tests/test_evaluate.py
python tests/test_compare.py
```

Each prints `N/M passing` and exits 1 on any failure. There is no
aggregate runner — run the file(s) relevant to what you changed.

Run the inference demo locally:
```
pip install -r requirements.txt
streamlit run app.py
```
Requires `best.pt` in the repo root — download it from the Kaggle
training run's output. Unlike other trained artifacts, this one is
committed to git (`.gitignore` carves out a `!best.pt` exception) because
Streamlit Community Cloud deploys straight from this repo with no
separate weights-only store.

There is no build step, linter, or CI config in this repo.

## Architecture

### The `src/` pipeline

Each file corresponds to one session and is designed to run identically
locally (using synthetic fixtures, no GPU/dataset) and on Kaggle (real
data/GPU), via `src/paths.py`'s environment detection
(`ON_KAGGLE = Path("/kaggle/input").exists()`). This split — test the
logic locally, run the heavy step on Kaggle — is the recurring pattern
across the whole pipeline.

1. **`bdd_to_yolo.py`** — Converts BDD100K JSON labels to YOLO `.txt`
   format. `CATEGORY_MAP`/`CLASS_NAMES` define the 7-class taxonomy
   (`person, car, truck, bus, two_wheeler, traffic_light, traffic_sign`)
   merged from BDD's 10 raw categories. Notably, `rider` merges into
   `person`, not `two_wheeler` — BDD boxes the human separately from the
   vehicle they're riding, so `rider` is semantically a person.
2. **`subsample.py`** — Builds a class-balanced training subset.
   `RARE_CLASS_IDS` are protected as an absolute guarantee (every image
   containing one is included), not a percentage — inflating the overall
   subset size does *not* dilute this guarantee the way it would with a
   percentage-based scheme.
3. **`dataset_config.py`** — Writes `configs/data.yaml` and
   `configs/hyp_road_scene.yaml` (the `AUGMENTATION_POLICY`). These two
   YAML files are generated artifacts — regenerate via this script if the
   class list or augmentation policy changes; don't hand-edit them.
4. **`train_config.py`** — Converts Session 2's stem manifests into
   explicit full image-path lists. This works around a real Ultralytics
   behavior: given a *directory* (not an explicit file list), it resolves
   a symlinked images directory down to its real (read-only, Kaggle
   input-mounted) target before doing the images→labels path
   substitution, which breaks label discovery. Explicit path lists avoid
   the resolution step entirely.
5. **`evaluate.py`** — Parses Ultralytics' pasted per-class console
   output into structured rows, then flags precision/recall inversions
   and localization gaps (`mAP50` vs `mAP50-95`) against calibrated
   thresholds (`PR_GAP_THRESHOLD`, `LOC_GAP_THRESHOLD`).
6. **`compare.py`** — Diffs two `evaluate.py` outputs (before/after a
   change) class by class, for controlled improvement iterations.

`paths.py` and `extract_subset.py` are shared utilities, not pipeline
stages: `paths.py` is the environment-detection shim every other script
and notebook cell calls into (Kaggle mounts data at `/kaggle/input` with
inconsistent nesting per-upload, so paths are discovered, not
hardcoded); `extract_subset.py` pulls a small local image subset out of
a zip for offline debugging.

### Training happens only on Kaggle

Training itself never runs locally. It follows `notebooks/KAGGLE_SETUP.md`'s
six-cell pattern in a Kaggle notebook. The notebook re-clones from GitHub
at the start of every session rather than being edited in place, because
`/kaggle/working` is wiped on every session restart or accelerator change.

Because Ultralytics can't discover labels against Kaggle's read-only
`/kaggle/input` mount directly (see `train_config.py` above), the Kaggle
notebook wires a parallel writable directory tree under `/kaggle/working`
with symlinked `images/` and `labels/` subdirectories, and
`train_config.py` builds the explicit image-path lists that point into
that symlinked tree.

### Test fixtures encode real project history

Several test suites (`test_compare.py`, `test_evaluate.py`) use real
per-class metric tables from actual training runs as fixtures rather than
purely synthetic data — they double as regression checks against the
project's own recorded results, not just unit tests of parsing logic.

### The demo

`app.py` is a Streamlit app deployed on Streamlit Community Cloud,
connected directly to this GitHub repo. It was originally built with
Gradio for Hugging Face Spaces, but HF now requires a PRO subscription
just to create a new Gradio/Docker Space (the CPU-basic compute itself is
still free once one exists — creation is what's gated). Streamlit
Community Cloud remains genuinely free for public apps deployed from
GitHub, at the cost of no separate artifact store: `best.pt` has to be
committed to this repo rather than living apart from it the way a
Hugging Face Space would hold it. If revisiting this later, verify
current pricing before assuming either platform's free tier still works
the same way.

`.plot()`'s channel order was verified empirically (feed it RGB, get RGB
back, for both the base image and drawn boxes) rather than assumed —
worth re-checking if the detection overlay's colors ever look wrong
after an ultralytics version bump.

## Frontend aesthetics

<frontend_aesthetics>
You tend to converge toward generic, "on distribution" outputs. In frontend design, this creates what users call the "AI slop" aesthetic. Avoid this: make creative, distinctive frontends that surprise and delight. Focus on:

Typography: Choose fonts that are beautiful, unique, and interesting. Avoid generic fonts like Arial and Inter; opt instead for distinctive choices that elevate the frontend's aesthetics.

Color & Theme: Commit to a cohesive aesthetic. Use CSS variables for consistency. Dominant colors with sharp accents outperform timid, evenly-distributed palettes. Draw from IDE themes and cultural aesthetics for inspiration.

Motion: Use animations for effects and micro-interactions. Prioritize CSS-only solutions for HTML. Use Motion library for React when available. Focus on high-impact moments: one well-orchestrated page load with staggered reveals (animation-delay) creates more delight than scattered micro-interactions.

Backgrounds: Create atmosphere and depth rather than defaulting to solid colors. Layer CSS gradients, use geometric patterns, or add contextual effects that match the overall aesthetic.

Avoid generic AI-generated aesthetics:
- Overused font families (Inter, Roboto, Arial, system fonts)
- Clichéd color schemes (particularly purple gradients on white backgrounds)
- Predictable layouts and component patterns
- Cookie-cutter design that lacks context-specific character

Interpret creatively and make unexpected choices that feel genuinely designed for the context. Vary between light and dark themes, different fonts, different aesthetics. You still tend to converge on common choices (Space Grotesk, for example) across generations. Avoid this: it is critical that you think outside the box!
</frontend_aesthetics>
