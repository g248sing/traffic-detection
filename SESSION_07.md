# Session 7 — Inference demo + README

**Time:** ~30–45 min, mostly clicking through Kaggle/Streamlit UI, not
writing code.

---

## Why this session looks different

Sessions 1–6 were all testable Python logic with a TODO+tests structure.
This one isn't — the actual work is downloading a checkpoint, deploying a
small app, and writing documentation. Forcing a fake test suite onto that
would just be busywork, so this session is a straightforward checklist
instead.

`app.py`, `requirements.txt`, and `README.md` are already written for you
in the repo root. Your job is deploying and reviewing them, not building
them from scratch.

## A platform change worth knowing about

This was originally planned around Hugging Face Spaces (Gradio SDK). As of
this session, Hugging Face now requires a PRO subscription ($9/mo) just to
*create* a new Gradio or Docker Space — the CPU-basic compute itself is
still free once a Space exists, but creating one isn't, anymore. Rather
than pay for a portfolio demo, `app.py` was rewritten from Gradio to
**Streamlit**, deployed on **Streamlit Community Cloud**, which is still
genuinely free for public apps deployed from a GitHub repo. Worth knowing
if you're following this pattern later and platform pricing has changed
again — check before assuming either one is still free.

The UI itself ended up more deliberately designed than a typical demo:
"Overpass" (highway-signage typography) and "JetBrains Mono", a dark
asphalt-toned background instead of a default light theme, and a single
accent color rather than a generic gradient. `CLAUDE.md`'s
`frontend_aesthetics` section captures the reasoning, if you want to
apply the same approach elsewhere.

---

## Step 1 — Get the trained weights off Kaggle

Your Session 6 run (the `imgsz=960` one) saved its best checkpoint to:

```
/kaggle/working/traffic-detection/runs/detect/train/weights/best.pt
```

Download it from the notebook's **Output** tab (`kaggle kernels output
gurveer10sj/notebook62ebde5635 -p <dest>`, or click download in the
browser). Note: Kaggle's browser download sometimes names the file
`best.zip` even though it's the actual `.pt` checkpoint (PyTorch's format
is internally a zip archive) — if that happens, just rename it back to
`.pt`, don't extract it.

Place `best.pt` in the repo root, next to `app.py`. Unlike most trained
artifacts, this one **is** meant to be committed to git — Streamlit
Community Cloud deploys straight from the GitHub repo with no separate
weights-only store, unlike a Hugging Face Space. `.gitignore` has a
`!best.pt` exception carved out of the general `*.pt` rule for exactly
this reason.

## Step 2 — Try the demo locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens a local Streamlit UI. Upload a road-scene image (grab one from
BDD100K, or any street/dashcam photo) and confirm you get boxes back with
sensible labels. Worth doing before deploying — catches "wrong model
path" or "wrong class names" mistakes on your own machine, and it's also
where you'll actually notice things a validation-set mAP number hides:
real photos with a camera angle the model never trained on, or a specific
object type getting confused with a vehicle class. Both of those showed
up during this project's own testing and are documented in the README's
limitations, worth reading if you want the pattern for your own writeup.

## Step 3 — Commit `best.pt` and push

```bash
git add -A
git commit -m "Session 7: inference demo, README, best.pt for deployment"
git push
```

## Step 4 — Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
   with GitHub (sign up first if you don't have an account).
2. **New app** → pick this repo, the branch (`main`), and the entry-point
   file (`app.py`).
3. Deploy. The build log installs `requirements.txt` then runs `app.py` —
   watch for either the app loading or a red error. `ultralytics`/`torch`
   are heavy installs, so the first build takes a few minutes.
4. You get a public URL (`https://<something>.streamlit.app`) once it's
   live.

## Step 5 — Update the README

Put the live URL into `README.md`'s "Live demo" line (currently a
placeholder), and the "Running the demo locally" section already reflects
`streamlit run app.py`. Read the rest of the README end to end too — it's
drafted with your real numbers from Sessions 1–6, but you should correct
anything that doesn't sound like you, or that undersells/oversells a
decision you'd explain differently.

## Step 6 — Push the README update

```bash
git add -A
git commit -m "Session 7: add live demo link"
git push
```

Streamlit Community Cloud auto-redeploys on every push to the connected
branch, so this also confirms the deploy pipeline actually works
end-to-end for future changes.

---

## When this is done

You'll have: a GitHub repo with tested code and documented reasoning for
every decision, plus a live public URL where anyone can drop in an image
and see the model work. That combination — not just a notebook, not just
a demo, but both plus the reasoning connecting them — is the actual
deliverable for whoever's evaluating this.
