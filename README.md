# Panacea: Surgical Mastery

AI-guided surgical training simulator that generates structured syllabi and walks trainees through instrument identification using real-time video.

---

## What It Does

1. **Pick a module** — Choose from _Laparoscopic Appendectomy_, _Basic Suturing 101_, or _Cataract Tray Setup_.
2. **Gemini generates a syllabus** — A 3-step training plan with target instruments, tasks, and educational pro-tips.
3. **Camera-guided identification** — The live video feed simulates YOLO-based tool detection with an animated lock-on bounding box every ~5 seconds.
4. **Auto-advancing checklist** — When the correct tool is detected, the syllabus step turns green, the next step activates, and the Tutor's Message displays a clinical pro-tip.

---

## Tech Stack

| Layer                   | Technology                                 |
| ----------------------- | ------------------------------------------ |
| Frontend                | Streamlit (wide layout, custom dark CSS)   |
| Video                   | streamlit-webrtc, OpenCV, PyAV             |
| AI Syllabus             | Gemini 1.5 Flash (with hardcoded fallback) |
| Validation              | Pydantic v2                                |
| Detection (placeholder) | YOLOv11 + MediaPipe (TODO)                 |

---

## Project Structure

```
panacea/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .gitignore
├── README.md
└── styles/
    ├── __init__.py         # load_css() helper
    └── surgical_theme.css  # Dark surgical dashboard theme
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- A webcam (for the WebRTC feed)

### Installation

```bash
git clone https://github.com/anle0429/Panacea.git
cd Panacea
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. Select a training module from the sidebar, click **Initialize Training**, then grant camera access and click **START** on the WebRTC widget.

---

## Gemini API (Optional)

The app works fully offline with hardcoded syllabi. To enable live Gemini generation:

1. Create `.streamlit/secrets.toml`:
   ```toml
   GEMINI_API_KEY = "your-api-key-here"
   ```
2. Restart the app. The syllabus will be generated dynamically by Gemini 1.5 Flash.

---

## How the Training Loop Works

```
IDLE ──► TRAINING ──► COMPLETE
  ▲                      │
  └──────────────────────┘
         (Reset)
```

**IDLE** — Welcome screen. Sidebar shows module dropdown + Initialize button.

**TRAINING** — 70/30 column layout:

- **Left (70%):** WebRTC video feed with lock-on detection animation. Below it, a _Tutor's Message_ box that shows the current step's `pro_tip` on completion.
- **Right (30%):** Live syllabus with 3 step cards. The active step pulses cyan; completed steps turn green with a checkmark; pending steps are dimmed.

**COMPLETE** — Trophy banner. All steps verified. Click _Reset Module_ to try another.

### Mock Detection

The video callback simulates a detection every ~5 seconds (150 frames at 30 fps). It draws an animated bounding box that "locks on" to the center of the frame, then pushes the current step's `target_tool` through a thread-safe `queue.Queue` to the main Streamlit thread, which advances the syllabus.

---

## Pre-Op / Intra-Op / Post-Op flow (current)

The app can run in demo mode: **Pre-Op** (checklist from webcam + YOLO) → **Intra-Op** (phase + rules) → **Post-Op** (analytics from event log).

- **Run:** `streamlit run app.py` → http://localhost:8501  
- **Requires:** `models/best.pt`, `recipes/trauma_room.json`. Webcam optional.
- **Config:** Sidebar sets confidence, inference size, frame skip. Recipe sets `stable_seconds` and intra-op rules.

---

## Development & debugging

Code is split so you can develop and debug by layer:

| Where | What to change / debug |
|-------|-------------------------|
| **`app.py`** | Flow, UI, and wiring. Config in sidebar; webcam loop; Pre-Op / Intra-Op / Post-Op sections. Use constants from `src.constants` instead of magic strings. |
| **`src/constants.py`** | Paths, config defaults, session state key names. Change once here instead of searching the app. |
| **`src/state.py`** | Mode (PRE_OP / INTRA_OP / POST_OP) and phase. All mode/phase access goes through `get_mode`, `set_mode`, `get_phase`, `set_phase`. |
| **`src/detector.py`** | YOLO load (cached), inference, count, draw. Add logging or try/except here if inference fails or classes don’t match. |
| **`src/rules.py`** | Intra-op rule logic and debounce. Rule names and tool names use normalized form (lower, spaces → `_`). |
| **`src/logger.py`** | Events go to `logs/events.jsonl`. Use `read_events()` in Post-Op or in a script to inspect what was logged. |
| **`src/utils.py`** | Recipe load and `ToolPresenceSmoother`. Checklist “present” = seen in ≥1 of last N samples. |
| **`recipes/trauma_room.json`** | Required tools and intra-op rules. Tool names must match YOLO class names when normalized (e.g. `needle_holder` or `Needle Holder`). |

**Tips:**

- Session state keys are in `src.constants` (e.g. `KEY_LAST_COUNTS`, `KEY_ALERTS_LOG`). Use them when reading/writing so renames are in one place.
- If the checklist never passes, check that recipe `preop_required[].tool` matches your model’s class names (after normalizing: lower, spaces → `_`).
- To test without a camera, open **Post-Op** and rely on `logs/events.jsonl` (you can append test events manually).

---

## Replacing the Mock with Real YOLO (legacy flow)

Inside `video_frame_callback` there are TODO blocks ready for integration:

```python
# from ultralytics import YOLO
# model = YOLO("yolov11_surgical_tools.pt")
# results = model(img, conf=0.5)
```

Swap the mock detection block for real inference, match `model.names[cls]` against `target_tool`, and push to `tool_queue`.

---

## Team

Built for the hackathon by the Panacea team.

---
