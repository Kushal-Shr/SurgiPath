"""
Panacea: Surgical Mastery — AI-Guided Training Simulator
Syllabus-driven workflow with mock vision detection and Gemini integration.
"""

import json
import math
import queue
import threading
import time
from datetime import datetime

import av
import cv2
import numpy as np
import streamlit as st
from pydantic import BaseModel, ValidationError
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from styles import load_css

# ──────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────

class SyllabusStep(BaseModel):
    step_number: int
    task: str
    target_tool: str
    pro_tip: str


class TrainingSyllabus(BaseModel):
    steps: list[SyllabusStep]


# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Panacea: Surgical Mastery",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css()

# ──────────────────────────────────────────────
# Session state defaults
# ──────────────────────────────────────────────

_DEFAULTS: dict = {
    "app_state": "IDLE",             # IDLE → TRAINING → COMPLETE
    "module_name": "",
    "syllabus": None,                # TrainingSyllabus dict
    "current_step_index": 0,
    "completed_steps": set(),
    "latest_pro_tip": "",
    "training_start_time": None,
}

for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ──────────────────────────────────────────────
# Thread-safe shared state
# ──────────────────────────────────────────────

tool_queue: queue.Queue = queue.Queue(maxsize=64)

_shared_lock = threading.Lock()
_shared_target_tool: list[str] = [""]     # current step's target tool
_shared_all_tools: list[str] = []         # all target tools in the syllabus
_frame_counter: list[int] = [0]


def set_shared_target(target: str, all_tools: list[str]) -> None:
    with _shared_lock:
        _shared_target_tool[0] = target
        _shared_all_tools.clear()
        _shared_all_tools.extend(all_tools)
        _frame_counter[0] = 0


def get_shared_target() -> tuple[str, list[str]]:
    with _shared_lock:
        return _shared_target_tool[0], list(_shared_all_tools)


# ──────────────────────────────────────────────
# Training modules
# ──────────────────────────────────────────────

TRAINING_MODULES = [
    "Laparoscopic Appendectomy",
    "Basic Suturing 101",
    "Cataract Tray Setup",
]

# ──────────────────────────────────────────────
# Gemini syllabus generation
# ──────────────────────────────────────────────

GEMINI_SYLLABUS_PROMPT = (
    "You are a surgical educator. Generate a 3-step training syllabus "
    "for {procedure}. Return a JSON object with:\n"
    '{{"steps": [{{"step_number": 1, "task": "Identify X", '
    '"target_tool": "tool_name", "pro_tip": "Educational fact"}}]}}\n'
    "Return ONLY valid JSON. Be specific and educational."
)


def generate_syllabus(procedure: str) -> TrainingSyllabus:
    """Generate a training syllabus via Gemini 1.5 Flash or hardcoded mock."""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        api_key = ""

    if api_key:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = GEMINI_SYLLABUS_PROMPT.format(procedure=procedure)
        response = model.generate_content(prompt)
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        data = json.loads(raw)
    else:
        _MOCK_SYLLABI: dict[str, dict] = {
            "Laparoscopic Appendectomy": {
                "steps": [
                    {
                        "step_number": 1,
                        "task": "Identify and present the Laparoscopic Trocar",
                        "target_tool": "Laparoscopic Trocar",
                        "pro_tip": (
                            "The 12 mm umbilical trocar is inserted first to "
                            "establish pneumoperitoneum. Always confirm intra-"
                            "abdominal placement with a 0° scope before "
                            "inserting secondary ports."
                        ),
                    },
                    {
                        "step_number": 2,
                        "task": "Locate the Grasper Forceps for mesoappendix retraction",
                        "target_tool": "Grasper Forceps",
                        "pro_tip": (
                            "Atraumatic graspers should be used on the appendix "
                            "tip to avoid perforation. Apply gentle traction "
                            "towards the anterior abdominal wall to create a "
                            "'critical view' of the mesoappendix."
                        ),
                    },
                    {
                        "step_number": 3,
                        "task": "Present the Endoscopic Stapler for base ligation",
                        "target_tool": "Endoscopic Stapler",
                        "pro_tip": (
                            "Fire the stapler across the appendiceal base with "
                            "at least 3 mm of healthy cecal tissue. A single "
                            "fire is preferred — double-stapling increases "
                            "tissue necrosis risk."
                        ),
                    },
                ],
            },
            "Basic Suturing 101": {
                "steps": [
                    {
                        "step_number": 1,
                        "task": "Identify and pick up the Needle Driver",
                        "target_tool": "Needle Driver",
                        "pro_tip": (
                            "Grip the needle driver two-thirds of the way back "
                            "from the tip. The needle should be loaded at the "
                            "junction of the middle and distal thirds for "
                            "optimal control and arc."
                        ),
                    },
                    {
                        "step_number": 2,
                        "task": "Locate the Tissue Forceps for wound edge eversion",
                        "target_tool": "Tissue Forceps",
                        "pro_tip": (
                            "Adson forceps with teeth provide the best grip on "
                            "skin without crushing tissue. Always evert wound "
                            "edges slightly to promote first-intention healing."
                        ),
                    },
                    {
                        "step_number": 3,
                        "task": "Present the Suture material for wound closure",
                        "target_tool": "Suture (3-0 Vicryl)",
                        "pro_tip": (
                            "3-0 Vicryl is a braided, absorbable suture ideal "
                            "for subcutaneous closure. It maintains ~75% tensile "
                            "strength at 2 weeks and absorbs fully by 70 days."
                        ),
                    },
                ],
            },
            "Cataract Tray Setup": {
                "steps": [
                    {
                        "step_number": 1,
                        "task": "Identify the Phaco Handpiece on the tray",
                        "target_tool": "Phaco Handpiece",
                        "pro_tip": (
                            "The phacoemulsification handpiece uses ultrasonic "
                            "vibrations at 28–40 kHz to emulsify the lens "
                            "nucleus. Verify the irrigation/aspiration lines "
                            "are primed and bubble-free before use."
                        ),
                    },
                    {
                        "step_number": 2,
                        "task": "Locate the Capsulorhexis Forceps",
                        "target_tool": "Capsulorhexis Forceps",
                        "pro_tip": (
                            "Utrata forceps are the gold standard for continuous "
                            "curvilinear capsulorhexis (CCC). Maintain constant "
                            "anterior chamber depth with viscoelastic to prevent "
                            "the rhexis from running out peripherally."
                        ),
                    },
                    {
                        "step_number": 3,
                        "task": "Present the IOL Injector for lens implantation",
                        "target_tool": "IOL Injector",
                        "pro_tip": (
                            "Load the foldable IOL with viscoelastic coating the "
                            "cartridge. Advance the plunger slowly and steadily — "
                            "a controlled injection unfolds the lens within the "
                            "capsular bag, reducing endothelial cell damage."
                        ),
                    },
                ],
            },
        }
        data = _MOCK_SYLLABI.get(procedure, _MOCK_SYLLABI["Laparoscopic Appendectomy"])
        time.sleep(2)

    return TrainingSyllabus(**data)


# ──────────────────────────────────────────────
# WebRTC video callback
# ──────────────────────────────────────────────

_CYAN = (0, 229, 255)
_GREEN = (0, 230, 118)
_DARK = (0, 0, 0)
_YELLOW = (0, 171, 255)

# Detection fires every ~150 frames (~5 seconds at 30 fps)
_DETECTION_INTERVAL = 150


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    h, w, _ = img.shape

    target, all_tools = get_shared_target()
    if not target:
        cv2.putText(img, "AWAITING TARGET", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, _YELLOW, 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    _frame_counter[0] += 1
    fc = _frame_counter[0]

    # TODO: Replace with real YOLO inference
    # ─────────────────────────────────────────
    # from ultralytics import YOLO
    # model = YOLO("yolov11_surgical_tools.pt")
    # results = model(img, conf=0.5)
    # ─────────────────────────────────────────

    # ── HUD: current target ──
    cv2.putText(img, f"TARGET: {target}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, _CYAN, 2)

    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    cv2.putText(img, ts, (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _CYAN, 1)

    # Detect every ~5 seconds: draw a bounding box and push to queue
    is_detecting = (fc % _DETECTION_INTERVAL) >= (_DETECTION_INTERVAL - 60)

    if is_detecting:
        # Animated box that "locks on"
        progress = ((fc % _DETECTION_INTERVAL) - (_DETECTION_INTERVAL - 60)) / 60.0
        margin = int(40 * (1 - progress))

        cx, cy = w // 2, h // 2
        box_w, box_h = int(w * 0.40), int(h * 0.45)
        x1 = cx - box_w // 2 + margin
        y1 = cy - box_h // 2 + margin
        x2 = cx + box_w // 2 - margin
        y2 = cy + box_h // 2 - margin

        color = _GREEN if progress > 0.8 else _CYAN

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Corner brackets for lock-on effect
        bracket = 20
        for (bx, by, dx, dy) in [
            (x1, y1, 1, 1), (x2, y1, -1, 1),
            (x1, y2, 1, -1), (x2, y2, -1, -1),
        ]:
            cv2.line(img, (bx, by), (bx + dx * bracket, by), color, 3)
            cv2.line(img, (bx, by), (bx, by + dy * bracket), color, 3)

        conf = 0.75 + 0.23 * progress
        label = f"{target}  [{conf:.2f}]"
        (tw, th_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1 - th_text - 12), (x1 + tw + 10, y1), color, -1)
        cv2.putText(img, label, (x1 + 5, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _DARK, 2)

        if progress > 0.8:
            cv2.putText(img, "LOCKED", (x1 + 5, y2 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, _GREEN, 2)

    # At the end of a detection cycle, push the tool name
    if fc > 0 and fc % _DETECTION_INTERVAL == 0:
        try:
            tool_queue.put_nowait(target)
        except queue.Full:
            pass

    # Scanning animation bar at bottom
    scan_x = int((fc % 120) / 120.0 * w)
    cv2.line(img, (scan_x, h - 4), (min(scan_x + 60, w), h - 4), _CYAN, 3)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


# ──────────────────────────────────────────────
# Sidebar — module selector & training controls
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div class="dashboard-header">'
        "<h1>🩺 Panacea</h1>"
        "<p>Surgical Mastery Trainer</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    state = st.session_state.app_state

    # Status badge
    badge_map = {
        "IDLE": ("● Awaiting Module", "status-idle"),
        "TRAINING": ("● Training Active", "status-active"),
        "COMPLETE": ("● Module Complete", "status-ready"),
    }
    badge_text, badge_cls = badge_map.get(state, badge_map["IDLE"])
    st.markdown(
        f'<span class="status-badge {badge_cls}">{badge_text}</span>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### Select Your Training Module")

    selected_module = st.selectbox(
        "Training module",
        TRAINING_MODULES,
        label_visibility="collapsed",
        disabled=state == "TRAINING",
    )

    if state == "IDLE":
        init_clicked = st.button(
            "🧠  Initialize Training",
            use_container_width=True,
        )

        if init_clicked:
            with st.status("Gemini is building your syllabus…", expanded=True) as status:
                st.write(f"**Module:** {selected_module}")
                st.write("Querying Gemini 1.5 Flash for structured syllabus…")

                try:
                    syllabus = generate_syllabus(selected_module)
                except (json.JSONDecodeError, ValidationError) as exc:
                    status.update(label="Failed to generate syllabus", state="error")
                    st.error(f"Gemini response error: {exc}")
                    st.stop()

                st.write(f"Received **{len(syllabus.steps)}** training steps.")
                for step in syllabus.steps:
                    st.write(f"  Step {step.step_number}: {step.task}")
                status.update(label="Syllabus ready!", state="complete")

            st.session_state.module_name = selected_module
            st.session_state.syllabus = syllabus.model_dump()
            st.session_state.current_step_index = 0
            st.session_state.completed_steps = set()
            st.session_state.latest_pro_tip = ""
            st.session_state.training_start_time = datetime.now().isoformat()
            st.session_state.app_state = "TRAINING"
            st.rerun()

    elif state in ("TRAINING", "COMPLETE"):
        if st.button("🔄  Reset Module", use_container_width=True):
            st.session_state.app_state = "IDLE"
            st.session_state.syllabus = None
            st.session_state.completed_steps = set()
            st.session_state.current_step_index = 0
            st.session_state.latest_pro_tip = ""
            set_shared_target("", [])
            st.rerun()

    # Sidebar metrics when training
    if state in ("TRAINING", "COMPLETE") and st.session_state.syllabus:
        syllabus = st.session_state.syllabus
        n_steps = len(syllabus["steps"])
        n_done = len(st.session_state.completed_steps)
        pct = int(n_done / n_steps * 100) if n_steps else 0

        st.markdown("---")
        st.markdown(
            f'<div class="progress-ring">'
            f'<div class="pct">{pct}%</div>'
            f'<div class="pct-label">Steps Complete ({n_done}/{n_steps})</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("### System")
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(
                '<div class="metric-card">'
                '<div class="value">30</div>'
                '<div class="label">FPS</div>'
                "</div>",
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                '<div class="metric-card">'
                '<div class="value">~5s</div>'
                '<div class="label">Detect Cycle</div>'
                "</div>",
                unsafe_allow_html=True,
            )


# ──────────────────────────────────────────────
# Main area — header
# ──────────────────────────────────────────────

st.markdown(
    '<div class="dashboard-header">'
    "<h1>PANACEA: SURGICAL MASTERY</h1>"
    "<p>AI-Guided Training Simulator • Gemini 1.5 Flash • YOLOv11</p>"
    "</div>",
    unsafe_allow_html=True,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IDLE state — welcome screen
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if st.session_state.app_state == "IDLE":
    st.markdown("")
    st.markdown("")
    cols = st.columns([1, 2, 1])
    with cols[1]:
        st.markdown(
            '<div class="dashboard-header" style="text-align:center;padding:3rem 2rem;">'
            "<h2>Welcome to Surgical Mastery</h2>"
            '<p style="font-size:1rem;margin-top:0.8rem;">'
            "Select a training module from the sidebar and click "
            "<strong>Initialize Training</strong> to begin.<br><br>"
            "Gemini will generate a structured syllabus, then the camera "
            "will guide you through identifying each surgical instrument."
            "</p></div>",
            unsafe_allow_html=True,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAINING / COMPLETE — 70/30 layout
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

elif st.session_state.app_state in ("TRAINING", "COMPLETE"):
    syllabus = st.session_state.syllabus
    steps = syllabus["steps"]
    n_steps = len(steps)
    current_idx = st.session_state.current_step_index

    # ── Drain the tool_queue and advance steps ──
    while not tool_queue.empty():
        try:
            detected = tool_queue.get_nowait()
            if current_idx < n_steps:
                current_step = steps[current_idx]
                if detected == current_step["target_tool"]:
                    st.session_state.completed_steps.add(current_idx)
                    st.session_state.latest_pro_tip = current_step["pro_tip"]
                    st.session_state.current_step_index = current_idx + 1
                    current_idx = st.session_state.current_step_index

                    if current_idx >= n_steps:
                        st.session_state.app_state = "COMPLETE"
        except queue.Empty:
            break

    # Update shared target for the video callback thread
    if current_idx < n_steps:
        target_tool = steps[current_idx]["target_tool"]
    else:
        target_tool = ""
    all_tools = [s["target_tool"] for s in steps]
    set_shared_target(target_tool, all_tools)

    # ── 70/30 columns ──
    col_video, col_syllabus = st.columns([7, 3])

    # ── LEFT: Video feed + tutor message ──
    with col_video:
        st.markdown(f"### 🎥 Live Feed — {st.session_state.module_name}")
        st.markdown('<div class="video-container">', unsafe_allow_html=True)

        ctx = webrtc_streamer(
            key="panacea-training-feed",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Tutor's Message box ──
        pro_tip = st.session_state.latest_pro_tip
        if pro_tip:
            st.markdown(
                f'<div class="tutor-box has-tip">'
                f'<div class="tutor-label">🎓 Tutor\'s Message</div>'
                f'<div class="tutor-text">{pro_tip}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            if current_idx < n_steps:
                hint = f"Present the <strong>{steps[current_idx]['target_tool']}</strong> to the camera to proceed."
            else:
                hint = "All steps completed!"
            st.markdown(
                f'<div class="tutor-box">'
                f'<div class="tutor-label">🎓 Tutor\'s Message</div>'
                f'<div class="tutor-text">{hint}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── RIGHT: Live syllabus ──
    with col_syllabus:
        st.markdown("### 📋 Live Syllabus")

        for i, step in enumerate(steps):
            if i in st.session_state.completed_steps:
                css_cls = "step-completed"
                icon = "✓"
            elif i == current_idx:
                css_cls = "step-active"
                icon = str(step["step_number"])
            else:
                css_cls = "step-pending"
                icon = str(step["step_number"])

            st.markdown(
                f'<div class="syllabus-step {css_cls}">'
                f'  <div class="step-header">'
                f'    <div class="step-num">{icon}</div>'
                f'    <div class="step-task">{step["task"]}</div>'
                f"  </div>"
                f'  <div class="step-tool">Target: <span>{step["target_tool"]}</span></div>'
                f"</div>",
                unsafe_allow_html=True,
            )

        # Completion banner
        if st.session_state.app_state == "COMPLETE":
            st.markdown(
                '<div class="completion-banner">'
                "<h2>🏆 Module Complete!</h2>"
                "<p>All instruments identified successfully. "
                "Reset from the sidebar to try another module.</p>"
                "</div>",
                unsafe_allow_html=True,
            )


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────

st.markdown("---")
ft1, ft2, ft3 = st.columns(3)
with ft1:
    st.caption("Panacea: Surgical Mastery v1.0.0")
with ft2:
    st.caption("Gemini 1.5 Flash • YOLOv11 • WebRTC")
with ft3:
    st.caption(f"Session: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
