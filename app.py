"""
SESSION 7 -- Public inference demo (Streamlit / Streamlit Community Cloud).

Loads the trained yolo11s checkpoint (best.pt, from Session 6's imgsz=960
run) and exposes a Streamlit interface: upload a road-scene image, get
back the same image with detected boxes drawn on it.

Run locally:    streamlit run app.py
Deploy:         Streamlit Community Cloud, connected directly to this
                GitHub repo (unlike Hugging Face Spaces, there's no
                separate weights-only artifact store here -- best.pt has
                to be committed to the repo for this deployment target;
                see the note in .gitignore).

UI design note: "Overpass" (the display/heading font below) was designed
for U.S. Interstate highway signage -- an intentional choice for a
road-scene detector, not a default. See CLAUDE.md's frontend_aesthetics
section for the guidance this was built against.

Channel-order note: ultralytics' Results.plot() returns an array in
whatever channel order it was given (verified empirically -- feed it
RGB, get RGB back, for both the base image and drawn boxes). Images here
go in as RGB (PIL .convert("RGB") + np.array()) and come out the same
way, so no BGR/RGB conversion is needed anywhere in this file.
"""

import numpy as np
from PIL import Image
import streamlit as st
from streamlit_paste_button import paste_image_button

CLASS_NAMES = ["person", "car", "truck", "bus", "two_wheeler", "traffic_light", "traffic_sign"]

st.set_page_config(page_title="Traffic Object Detection", layout="wide")


@st.cache_resource
def load_model():
    # Imported lazily, inside the cached loader, on purpose: `import
    # ultralytics` alone costs ~5.5s (the YOLO("best.pt") load itself is
    # only ~0.1s). At module scope that delay lands on first page paint,
    # since Streamlit runs the script on the first visit. Here it moves to
    # first inference instead, so the UI appears immediately.
    from ultralytics import YOLO

    return YOLO("best.pt")


# NOTE: injected with st.html(), not st.markdown(unsafe_allow_html=True).
# st.markdown runs content through a Markdown parser first, which turns any
# 4-space-indented line into a literal code block -- that renders the CSS as
# visible page text instead of applying it. Keep this string left-flush and
# keep using st.html().
HEAD_AND_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Overpass:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
@keyframes reveal {
from { opacity: 0; transform: translateY(14px); }
to { opacity: 1; transform: translateY(0); }
}
.stApp {
background:
repeating-linear-gradient(0deg, rgba(255,255,255,0.015) 0px, rgba(255,255,255,0.015) 1px, transparent 1px, transparent 3px),
radial-gradient(ellipse 900px 500px at 50% -10%, rgba(255,176,32,0.12), transparent 60%),
linear-gradient(160deg, #05070a 0%, #0b0e14 55%, #05070a 100%) !important;
}
.stApp, .stApp p, .stApp span, .stApp label { font-family: 'Overpass', sans-serif; }
.block-container { max-width: 1080px; padding-top: 3rem; padding-bottom: 4rem; }
.hud-header { animation: reveal 0.6s cubic-bezier(0.16, 1, 0.3, 1) both; animation-delay: 0.05s; }
[data-testid="stColumn"] { animation: reveal 0.6s cubic-bezier(0.16, 1, 0.3, 1) both; }
[data-testid="stColumn"]:nth-of-type(1) { animation-delay: 0.2s; }
[data-testid="stColumn"]:nth-of-type(2) { animation-delay: 0.35s; }
.hud-title {
font-family: 'Overpass', sans-serif;
font-weight: 800;
font-size: 3rem;
letter-spacing: 0.02em;
color: #f4f1ea;
text-transform: uppercase;
line-height: 1.1;
}
.hud-title span { color: #ffb020; text-shadow: 0 0 18px rgba(255,176,32,0.55); }
.hud-subtitle { font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; color: #8b93a1; margin-top: 6px; }
.hud-classes {
font-family: 'JetBrains Mono', monospace;
font-size: 0.72rem;
letter-spacing: 0.08em;
color: #4fd1c5;
margin-top: 14px;
padding-top: 14px;
border-top: 1px solid rgba(255,255,255,0.08);
}
.hud-link {
display: inline-block;
font-family: 'JetBrains Mono', monospace;
font-size: 0.78rem;
letter-spacing: 0.05em;
color: #ffb020 !important;
margin-top: 16px;
text-decoration: none;
border-bottom: 1px solid rgba(255,176,32,0.4);
padding-bottom: 2px;
}
.hud-link:hover { color: #ffcc66 !important; border-color: #ffcc66; }
[data-testid="stWidgetLabel"] p {
font-family: 'JetBrains Mono', monospace !important;
font-size: 0.72rem !important;
letter-spacing: 0.1em !important;
text-transform: uppercase !important;
color: #4fd1c5 !important;
}
.stButton button {
font-family: 'JetBrains Mono', monospace !important;
font-weight: 600 !important;
letter-spacing: 0.12em !important;
text-transform: uppercase !important;
background: #ffb020 !important;
color: #0b0e14 !important;
border: none !important;
box-shadow: 0 0 24px rgba(255,176,32,0.35) !important;
transition: box-shadow 0.2s ease, transform 0.15s ease !important;
}
.stButton button:hover { box-shadow: 0 0 32px rgba(255,176,32,0.55) !important; transform: translateY(-1px) !important; }
[data-testid="stButtonGroup"] label {
font-family: 'JetBrains Mono', monospace !important;
font-size: 0.78rem !important;
letter-spacing: 0.05em !important;
}
[data-testid="stButtonGroup"] label[aria-checked="true"] {
background: rgba(255,176,32,0.18) !important;
border-color: #ffb020 !important;
color: #ffb020 !important;
}
</style>
"""

HEADER = """
<div class="hud-header">
<div class="hud-title">TRAFFIC<span>//</span>DETECT</div>
<div class="hud-subtitle">YOLO11s &middot; trained on a class-balanced subset of BDD100K</div>
<div class="hud-classes">PERSON &nbsp;CAR &nbsp;TRUCK &nbsp;BUS &nbsp;TWO_WHEELER &nbsp;TRAFFIC_LIGHT &nbsp;TRAFFIC_SIGN</div>
<a class="hud-link" href="https://github.com/g248sing/traffic-detection" target="_blank">SOURCE + DESIGN REASONING &rarr;</a>
</div>
"""

st.html(HEAD_AND_CSS)
st.html(HEADER)

col1, col2 = st.columns(2)

with col1:
    # st.tabs() renders every tab's content into the DOM on every run --
    # it's a display-only (CSS show/hide) mechanism, not conditional
    # execution. That meant st.camera_input() ran (and requested camera
    # permission) immediately on page load regardless of which tab was
    # visually selected. st.segmented_control is a plain widget, not a
    # container, so only the matching if-branch below actually runs its
    # widget -- camera access is now requested only when "Webcam" is
    # chosen.
    mode = st.segmented_control("Image source", ["Upload", "Webcam", "Paste"], default="Upload")

    image = None
    # Type checkers flag paste_image_button's .image_data as the PIL.Image
    # *module* rather than an image instance -- that's a bad annotation in
    # streamlit-paste-button (it does `from PIL import Image` then
    # annotates `image_data: Image`, where it meant `Image.Image`). At
    # runtime it's a real PngImageFile; verified. Ignore the warning rather
    # than "fixing" it.
    if mode == "Upload":
        uploaded = st.file_uploader("Road scene image", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            image = Image.open(uploaded)
    elif mode == "Webcam":
        captured = st.camera_input("Capture a frame")
        if captured is not None:
            image = Image.open(captured)
    elif mode == "Paste":
        pasted = paste_image_button(
            "Paste from clipboard",
            text_color="#0b0e14",
            background_color="#ffb020",
            hover_background_color="#ffcc66",
        )
        if pasted.image_data is not None:
            image = pasted.image_data

    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    run_clicked = st.button("Detect")

    if image is not None:
        st.image(image, use_container_width=True)

with col2:
    if image is not None and run_clicked:
        # load_model() is deliberately called here rather than at script
        # level -- see its docstring comment. Calling it at module scope
        # would pay the ~5.5s ultralytics import on every page load.
        with st.spinner("Loading model and running detection"):
            model = load_model()
            results = model.predict(np.array(image.convert("RGB")), conf=confidence, verbose=False)
        st.image(results[0].plot(), use_container_width=True)
