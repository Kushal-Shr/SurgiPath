"""
SurgiPath — AI-Guided Training Simulator
UI and video pipeline. All AI logic lives in brain.py (Clarity AI).
"""

import os
import queue
import random
import threading
import time
from datetime import datetime

import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from brain import (
    STANDARD_TOOL_KEYS,
    SKIP_PENALTY,
    TIMER_PENALTY,
    ActionSuccess,
    LiveProctor,
    SyllabusError,
    check_student_action,
    generate_dynamic_syllabus,
    generate_final_critique,
    generate_learning_resources,
    generate_session_report,
    generate_skip_warning,
    get_live_proctor,
    set_live_proctor,
)
from styles import load_css

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[assignment,misc]

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None  # type: ignore[assignment,misc]

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="SurgiPath",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css()

# ──────────────────────────────────────────────
# Session state defaults
# ──────────────────────────────────────────────

_DEFAULTS: dict = {
    "app_state": "IDLE",
    "module_name": "",
    "syllabus": None,
    "current_step_index": 0,
    "completed_steps": set(),
    "latest_rationale": "",
    "proctor_log": [],
    "training_start_time": None,
    "demo_mode": False,
    "demo_injection": None,
    "mastery_score": 100,
    "skipped_steps": [],
    "session_report": "",
    "action_log": [],
    "event_log": [],
    "clarity_feedback": [],
    "step_start_time": 0.0,
    "final_critique": "",
    "learning_resources": "",
}

for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ──────────────────────────────────────────────
# Thread-safe shared state (for WebRTC mode)
# ──────────────────────────────────────────────

tool_queue: queue.Queue = queue.Queue(maxsize=64)

_shared_lock = threading.Lock()
_shared_target_tool: list[str] = [""]
_shared_all_tools: list[str] = []
_shared_wrong_tools: list[str] = []


def set_shared_target(target: str, all_tools: list[str], wrong_tools: list[str] | None = None) -> None:
    with _shared_lock:
        _shared_target_tool[0] = target
        _shared_all_tools.clear()
        _shared_all_tools.extend(all_tools)
        _shared_wrong_tools.clear()
        if wrong_tools:
            _shared_wrong_tools.extend(wrong_tools)


def get_shared_target() -> tuple[str, list[str], list[str]]:
    with _shared_lock:
        return _shared_target_tool[0], list(_shared_all_tools), list(_shared_wrong_tools)


# ──────────────────────────────────────────────
# YOLO model (loaded once at module level)
# ──────────────────────────────────────────────

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "best.pt")

if YOLO is not None and os.path.isfile(_MODEL_PATH):
    _yolo_model = YOLO(_MODEL_PATH)
else:
    _yolo_model = None

_debug_detections: list[str] = []


def _set_debug_detections(names: list[str]) -> None:
    with _shared_lock:
        _debug_detections.clear()
        _debug_detections.extend(names)


def _get_debug_detections() -> list[str]:
    with _shared_lock:
        return list(_debug_detections)


# ──────────────────────────────────────────────
# Live API frame capture (1 fps)
# ──────────────────────────────────────────────

_LIVE_FRAME_INTERVAL = 30  # ~1fps at 30fps
_live_frame_counter: list[int] = [0]


# ──────────────────────────────────────────────
# Proctor log helper
# ──────────────────────────────────────────────

def log_proctor(entry_type: str, text: str) -> None:
    st.session_state.proctor_log.insert(0, {
        "type": entry_type,
        "time": datetime.now().strftime("%H:%M:%S"),
        "text": text,
    })
    st.session_state.proctor_log = st.session_state.proctor_log[:30]


# ──────────────────────────────────────────────
# Proctoring logic (shared by both modes)
# ──────────────────────────────────────────────

def process_detection(detected: str, steps: list[dict], current_idx: int) -> int:
    """Run a single detected tool through brain.check_student_action.
    Returns the (possibly advanced) current_idx."""
    if current_idx >= len(steps):
        return current_idx

    current_step = steps[current_idx]
    result = check_student_action(
        detected_tools=[detected],
        current_target_tool=current_step["target_tool_key"],
        current_instruction=current_step["instruction"],
        current_safety_tip=current_step.get("critical_safety_tip", ""),
    )

    ts = datetime.now().strftime("%H:%M:%S")

    if isinstance(result, ActionSuccess):
        st.session_state.completed_steps.add(current_idx)
        st.session_state.latest_rationale = result.message
        log_proctor("correct", f"✓ {result.tool} — {result.message}")
        st.toast(f"✅ {result.tool} — correct!", icon="🎯")
        st.session_state.action_log.append({
            "time": ts, "type": "match", "tool": result.tool,
            "detail": f"Correctly identified at step {current_idx + 1}",
        })
        st.session_state.event_log.append({
            "time": ts, "type": "detection_match", "tool": result.tool,
            "detail": f"Correctly identified at step {current_idx + 1}",
        })

        st.session_state.current_step_index = current_idx + 1
        st.session_state.step_start_time = time.time()
        current_idx = st.session_state.current_step_index

        if current_idx >= len(steps):
            st.session_state.app_state = "COMPLETE"
            log_proctor("correct", "All training steps completed successfully.")
            st.toast("🏆 Module complete!", icon="🏆")
    else:
        log_proctor("correction", f"✗ Picked {result.wrong_tool} → {result.message}")
        st.toast(f"❌ Wrong tool: {result.wrong_tool}", icon="⚠️")
        st.session_state.action_log.append({
            "time": ts, "type": "wrong_tool", "tool": result.wrong_tool,
            "detail": f"Expected {result.target_tool}, got {result.wrong_tool}",
        })
        st.session_state.event_log.append({
            "time": ts, "type": "wrong_tool", "tool": result.wrong_tool,
            "detail": f"Expected {result.target_tool}, got {result.wrong_tool}",
        })

    return current_idx


# ──────────────────────────────────────────────
# WebRTC video callback
# ──────────────────────────────────────────────

_CYAN = (0, 229, 255)
_GREEN = (0, 230, 118)
_RED = (68, 23, 255)
_DARK = (0, 0, 0)
_YELLOW = (0, 171, 255)

_YOLO_CONF = 0.40


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    h, w, _ = img.shape

    target, _, _ = get_shared_target()

    if target:
        cv2.putText(img, f"TARGET: {target}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, _CYAN, 2)
    else:
        cv2.putText(img, "AWAITING TARGET", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, _YELLOW, 2)

    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    cv2.putText(img, ts, (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _CYAN, 1)

    if _yolo_model is None:
        cv2.putText(img, "YOLO NOT LOADED", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _RED, 2)
        _set_debug_detections([])
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    results = _yolo_model(img, conf=_YOLO_CONF, verbose=False)

    detected_names: list[str] = []

    if results and results[0].boxes is not None and len(results[0].boxes):
        boxes = results[0].boxes
        names_map = results[0].names

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = names_map[cls_id]
            detected_names.append(class_name)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            is_match = class_name == target
            color = _GREEN if is_match else _CYAN

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            bk = 15
            for (bx, by, dx, dy) in [
                (x1, y1, 1, 1), (x2, y1, -1, 1),
                (x1, y2, 1, -1), (x2, y2, -1, -1),
            ]:
                cv2.line(img, (bx, by), (bx + dx * bk, by), color, 3)
                cv2.line(img, (bx, by), (bx, by + dy * bk), color, 3)

            label = f"{class_name} {conf:.2f}"
            (tw, th_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(img, (x1, y1 - th_t - 10), (x1 + tw + 8, y1), color, -1)
            cv2.putText(img, label, (x1 + 4, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, _DARK, 2)

            if is_match:
                cv2.putText(img, "MATCH", (x1 + 4, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, _GREEN, 2)

    _set_debug_detections(detected_names)

    pushed: set[str] = set()
    for name in detected_names:
        if name not in pushed:
            pushed.add(name)
            try:
                tool_queue.put_nowait(name)
            except queue.Full:
                pass

    n_det = len(detected_names)
    det_color = _GREEN if n_det > 0 else _YELLOW
    cv2.putText(img, f"Detections: {n_det}", (15, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, det_color, 1)

    # ── Live API: send 1 frame/sec to Clarity ──
    _live_frame_counter[0] += 1
    if _live_frame_counter[0] % _LIVE_FRAME_INTERVAL == 0:
        proctor = get_live_proctor()
        if proctor is not None and proctor.active:
            small = cv2.resize(img, (768, 768))
            _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 50])
            proctor.send_frame(buf.tobytes())

    return av.VideoFrame.from_ndarray(img, format="bgr24")


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div class="dashboard-header">'
        "<h1>SurgiPath</h1>"
        "<p>AI-Guided Surgical Training</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    state = st.session_state.app_state

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

    # ── Mastery Score (visible during training / complete) ──
    if state in ("TRAINING", "COMPLETE"):
        score = st.session_state.mastery_score
        if score > 80:
            score_color = "#34C759"
        elif score >= 50:
            score_color = "#FF9500"
        else:
            score_color = "#FF3B30"
        n_skips = len(st.session_state.skipped_steps)
        delta_str = f"-{n_skips * SKIP_PENALTY} (skips)" if n_skips else "No deductions"
        st.markdown(
            f'<div style="text-align:center;padding:0.8rem 0 0.3rem;">'
            f'<div style="font-size:0.65rem;color:#666;letter-spacing:0.8px;'
            f'text-transform:uppercase;font-weight:400;">Mastery Score</div>'
            f'<div style="font-size:2.2rem;font-weight:600;color:{score_color};'
            f'margin:0.15rem 0;">{score}</div>'
            f'<div style="font-size:0.65rem;color:#666;font-weight:300;">'
            f'{delta_str}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.session_state.demo_mode = st.toggle(
        "🧪 Demo Mode (no camera)",
        value=st.session_state.demo_mode,
        help="Use a virtual instrument tray instead of the live camera feed.",
    )
    if st.session_state.demo_mode:
        st.caption("Click instruments from the virtual tray to simulate detections.")

    st.markdown("---")
    st.markdown("### Enter Surgery / Procedure Name")

    procedure_input = st.text_input(
        "Procedure",
        placeholder="e.g. Laparoscopic Cholecystectomy, Central Line Insertion…",
        label_visibility="collapsed",
        disabled=state == "TRAINING",
    )

    if state == "IDLE":
        init_clicked = st.button(
            "🧠  Start Training",
            use_container_width=True,
            disabled=not procedure_input.strip(),
        )

        if init_clicked and procedure_input.strip():
            with st.status("Clarity is reasoning…", expanded=True) as status:
                st.write(f"**Procedure:** {procedure_input.strip()}")
                st.write("Analyzing WHO Surgical Safety protocols…")

                result = generate_dynamic_syllabus(procedure_input.strip())

                if isinstance(result, SyllabusError):
                    status.update(label="Could not generate syllabus", state="error")
                    st.error(result.error)
                    st.stop()

                syllabus = result
                st.write(f"Generated **{len(syllabus.steps)}** training steps:")
                for step in syllabus.steps:
                    st.write(
                        f"  • **{step.step_name}** → `{step.target_tool_key}` "
                        f"({step.time_limit_seconds}s)"
                    )
                status.update(label="Syllabus ready!", state="complete")

            st.session_state.module_name = procedure_input.strip()
            st.session_state.syllabus = syllabus.model_dump()
            st.session_state.current_step_index = 0
            st.session_state.completed_steps = set()
            st.session_state.latest_rationale = ""
            st.session_state.proctor_log = []
            st.session_state.training_start_time = datetime.now().isoformat()
            st.session_state.mastery_score = 100
            st.session_state.skipped_steps = []
            st.session_state.session_report = ""
            st.session_state.action_log = []
            st.session_state.event_log = []
            st.session_state.clarity_feedback = []
            st.session_state.step_start_time = time.time()
            st.session_state.final_critique = ""
            st.session_state.learning_resources = ""
            _live_frame_counter[0] = 0
            proctor = LiveProctor()
            set_live_proctor(proctor)
            proctor.start(procedure_input.strip())
            st.session_state.app_state = "TRAINING"
            st.rerun()

    elif state in ("TRAINING", "COMPLETE"):
        if st.button("🔄  Reset Module", use_container_width=True):
            st.session_state.app_state = "IDLE"
            st.session_state.syllabus = None
            st.session_state.completed_steps = set()
            st.session_state.current_step_index = 0
            st.session_state.latest_rationale = ""
            st.session_state.proctor_log = []
            st.session_state.demo_injection = None
            st.session_state.mastery_score = 100
            st.session_state.skipped_steps = []
            st.session_state.session_report = ""
            st.session_state.action_log = []
            st.session_state.event_log = []
            st.session_state.clarity_feedback = []
            st.session_state.step_start_time = 0.0
            st.session_state.final_critique = ""
            st.session_state.learning_resources = ""
            _live_frame_counter[0] = 0
            old_proctor = get_live_proctor()
            if old_proctor is not None:
                old_proctor.stop()
            set_live_proctor(None)
            set_shared_target("", [], [])
            st.rerun()

    # ── Dynamic syllabus in sidebar ──
    if state in ("TRAINING", "COMPLETE") and st.session_state.syllabus:
        syllabus_data = st.session_state.syllabus
        n_steps = len(syllabus_data["steps"])
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

        st.markdown("### Syllabus")
        current_idx_sidebar = st.session_state.current_step_index
        _skipped_indices = {s["step_idx"] for s in st.session_state.skipped_steps}
        for i, step in enumerate(syllabus_data["steps"]):
            _tl_s = step.get("time_limit_seconds", 60)
            if i in st.session_state.completed_steps and i in _skipped_indices:
                st.markdown(
                    f'<span style="color:#FF3B30;font-weight:700;">✕</span> '
                    f'<del style="color:#9E9E9E;">{step["step_name"]}</del> '
                    f'— <code>{step["target_tool_key"]}</code>',
                    unsafe_allow_html=True,
                )
            elif i in st.session_state.completed_steps:
                st.markdown(
                    f'<span style="color:#34C759;font-weight:700;">✓</span> '
                    f'{step["step_name"]} '
                    f'— <code>{step["target_tool_key"]}</code>',
                    unsafe_allow_html=True,
                )
            elif i == current_idx_sidebar:
                st.markdown(
                    f'<span style="color:#007AFF;font-weight:700;">▶</span> '
                    f'<strong>{step["step_name"]}</strong> '
                    f'— <code>{step["target_tool_key"]}</code> '
                    f'<span style="color:#9E9E9E;">{_tl_s}s</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<span style="color:#666;">○</span> '
                    f'<span style="color:#666;">{step["step_name"]} '
                    f'— <code>{step["target_tool_key"]}</code> ({_tl_s}s)</span>',
                    unsafe_allow_html=True,
                )

        mode_label = "Demo" if st.session_state.demo_mode else "Camera"
        st.markdown("### System")
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="value">{mode_label}</div>'
                f'<div class="label">Mode</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
        with m2:
            det_label = "Live" if not st.session_state.demo_mode else "Click"
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="value">{det_label}</div>'
                f'<div class="label">Detection</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

        # ── Clarity Live Feed ──
        st.markdown("---")
        n_feedback = len(st.session_state.clarity_feedback)
        n_events = len(st.session_state.event_log)

        proctor = get_live_proctor()
        if state == "TRAINING" and proctor is not None and proctor.active:
            st.markdown(
                '<div style="padding:0.5rem 0.7rem;background:#1E1E1E;'
                'border:1px solid rgba(0,122,255,0.3);border-radius:6px;">'
                '<span style="color:#007AFF;font-weight:500;">●</span> '
                '<span style="color:#007AFF;font-size:0.8rem;font-weight:500;'
                'letter-spacing:0.5px;">CLARITY LIVE</span><br>'
                f'<span style="color:#666;font-size:0.7rem;font-weight:300;">'
                f'{n_feedback} observations &middot; {n_events} events</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        elif state == "TRAINING":
            st.markdown(
                '<div style="padding:0.5rem 0.7rem;background:#1E1E1E;'
                'border:1px solid #333;border-radius:6px;">'
                '<span style="color:#9E9E9E;font-weight:500;">○</span> '
                '<span style="color:#9E9E9E;font-size:0.8rem;font-weight:500;'
                'letter-spacing:0.5px;">CLARITY STANDBY</span><br>'
                f'<span style="color:#666;font-size:0.7rem;font-weight:300;">'
                f'{n_events} events logged</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        elif state == "COMPLETE":
            st.markdown(
                '<div style="padding:0.5rem 0.7rem;background:#1E1E1E;'
                'border:1px solid #333;border-radius:6px;">'
                '<span style="color:#007AFF;font-weight:500;">■</span> '
                '<span style="color:#007AFF;font-size:0.8rem;font-weight:500;'
                'letter-spacing:0.5px;">SESSION CLOSED</span><br>'
                f'<span style="color:#666;font-size:0.7rem;font-weight:300;">'
                f'{n_feedback} observations &middot; {n_events} events &middot; '
                f'Ready for grading</span>'
                '</div>',
                unsafe_allow_html=True,
            )

        # ── CLINICAL NARRATIVE — Clarity observations ──
        st.markdown(
            '<div style="font-size:0.65rem;font-weight:500;color:#666;'
            'letter-spacing:0.8px;text-transform:uppercase;'
            'margin-bottom:0.4rem;">Clinical Narrative</div>',
            unsafe_allow_html=True,
        )
        feedback = st.session_state.clarity_feedback

        if feedback:
            narrative_html = (
                '<div style="max-height:240px;overflow-y:auto;'
                'background:#1E1E1E;border:1px solid #333;border-left:3px solid #007AFF;'
                'border-radius:6px;padding:0.5rem 0.7rem;">'
            )
            for entry in reversed(feedback[-15:]):
                narrative_html += (
                    f'<div style="padding:3px 0;border-bottom:1px solid #2A2A2A;'
                    f'font-family:\'Crimson Pro\',Georgia,serif;font-size:0.78rem;'
                    f'line-height:1.5;color:#9E9E9E;">{entry}</div>'
                )
            narrative_html += '</div>'
            st.markdown(narrative_html, unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="background:#1E1E1E;border:1px solid #333;'
                'border-left:3px solid #007AFF;border-radius:6px;'
                'padding:0.6rem 0.7rem;font-family:\'Consolas\',monospace;'
                'font-size:0.68rem;color:#666;">'
                'Clarity will provide observations when active.</div>',
                unsafe_allow_html=True,
            )
        st.caption(f"Observations: {n_feedback} | Events: {n_events}")

        # ── Sidebar footer ──
        st.markdown("---")
        st.markdown(
            "*SurgiPath v1.0 | Developed by Kushal and An | HackSLU 2026*"
        )


# ──────────────────────────────────────────────
# Main area — header
# ──────────────────────────────────────────────

st.markdown(
    '<div class="dashboard-header">'
    "<h1>SurgiPath</h1>"
    "<p>AI-Guided Training &mdash; Clarity AI &middot; Gemini Live &middot; YOLOv11</p>"
    "</div>",
    unsafe_allow_html=True,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IDLE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if st.session_state.app_state == "IDLE":
    st.markdown("")
    cols = st.columns([1, 2, 1])
    with cols[1]:
        st.markdown(
            '<div class="dashboard-header" style="text-align:center;padding:3rem 2rem;">'
            "<h2>Welcome to SurgiPath</h2>"
            '<p style="font-size:1rem;margin-top:0.8rem;font-weight:300;">'
            "Type <strong>any</strong> medical or surgical procedure in the "
            "sidebar and click <strong>Start Training</strong>.<br><br>"
            "Clarity will reason through the procedure, validate it against "
            "WHO Surgical Safety protocols, and generate a structured "
            "training syllabus.<br><br>"
            "Use <strong>Demo Mode</strong> to practise without a camera, "
            "or connect a webcam for real-time Clarity analysis."
            "</p></div>",
            unsafe_allow_html=True,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAINING / COMPLETE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

elif st.session_state.app_state in ("TRAINING", "COMPLETE"):
    syllabus_data = st.session_state.syllabus
    steps = syllabus_data["steps"]
    n_steps = len(steps)
    current_idx = st.session_state.current_step_index
    is_demo = st.session_state.demo_mode

    # ── 1a. Process demo injection (set by tray button on previous rerun) ──
    demo_detected = st.session_state.pop("demo_injection", None)
    if demo_detected is not None:
        current_idx = process_detection(demo_detected, steps, current_idx)

    # ── 1b. Process manual override (set by skip button on previous rerun) ──
    skip_req = st.session_state.pop("_skip_requested", None)
    if skip_req is not None and current_idx < n_steps:
        skipped_tool = steps[current_idx]["target_tool_key"]
        skipped_instruction = steps[current_idx]["instruction"]
        skip_reason = skip_req.get("reason", "")

        warning = generate_skip_warning(
            target_tool=skipped_tool,
            instruction=skipped_instruction,
            reason=skip_reason,
        )

        st.session_state.mastery_score = max(
            0, st.session_state.mastery_score - SKIP_PENALTY
        )
        st.session_state.skipped_steps.append({
            "step_idx": current_idx,
            "tool": skipped_tool,
            "reason": skip_reason or "No reason given",
            "warning": warning,
        })
        st.session_state.completed_steps.add(current_idx)
        st.session_state.latest_rationale = f"⚠️ SKIPPED: {warning}"
        log_proctor(
            "skip",
            f"⚠️ Manual override — skipped {skipped_tool} "
            f"(−{SKIP_PENALTY} pts): {warning}",
        )
        st.session_state.action_log.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "type": "override",
            "tool": skipped_tool,
            "detail": f"Skipped (−{SKIP_PENALTY} pts): {skip_reason or 'No reason given'}",
        })
        st.session_state.event_log.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "type": "override",
            "tool": skipped_tool,
            "detail": f"Manual advance — {skip_reason or 'No reason given'}",
        })
        st.toast(
            f"⚠️ Skipped {skipped_tool} — {SKIP_PENALTY} pts deducted",
            icon="⚠️",
        )

        st.session_state.current_step_index = current_idx + 1
        st.session_state.step_start_time = time.time()
        current_idx = st.session_state.current_step_index
        if current_idx >= n_steps:
            st.session_state.app_state = "COMPLETE"
            log_proctor("correct", "All training steps completed.")

    # ── 1c. Timer expiry — auto-deduction if time runs out ──
    if (
        current_idx < n_steps
        and st.session_state.app_state == "TRAINING"
        and st.session_state.step_start_time > 0
    ):
        step_time_limit = steps[current_idx].get("time_limit_seconds", 60)
        elapsed = time.time() - st.session_state.step_start_time
        if elapsed >= step_time_limit:
            expired_step = steps[current_idx]
            expired_tool = expired_step["target_tool_key"]

            st.session_state.mastery_score = max(
                0, st.session_state.mastery_score - TIMER_PENALTY
            )
            st.session_state.completed_steps.add(current_idx)
            st.session_state.latest_rationale = (
                f"⏰ TIMEOUT: {expired_step.get('critical_safety_tip', '')}"
            )
            st.session_state.skipped_steps.append({
                "step_idx": current_idx,
                "tool": expired_tool,
                "reason": "Timer expired",
                "warning": f"TIMEOUT: {expired_step.get('critical_safety_tip', '')}",
            })
            log_proctor(
                "timeout",
                f"⏰ Time expired for {expired_step['step_name']} "
                f"(−{TIMER_PENALTY} pts)",
            )
            st.session_state.clarity_feedback.append(
                f"[Clarity] TIMEOUT: {expired_step['step_name']} — "
                f"safety protocol bypassed."
            )
            st.session_state.event_log.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "type": "timeout",
                "tool": expired_tool,
                "detail": f"Time expired — auto-advanced (−{TIMER_PENALTY} pts)",
            })
            st.toast(
                f"⏰ Time expired for {expired_step['step_name']}!",
                icon="⏰",
            )

            st.session_state.current_step_index = current_idx + 1
            st.session_state.step_start_time = time.time()
            current_idx = st.session_state.current_step_index
            if current_idx >= n_steps:
                st.session_state.app_state = "COMPLETE"
                log_proctor("correct", "All training steps completed.")
            st.rerun()

    # ── 2. Drain tool_queue from YOLO callback (camera mode) ──
    if not is_demo:
        while not tool_queue.empty():
            try:
                detected = tool_queue.get_nowait()
            except queue.Empty:
                break
            prev_idx = current_idx
            current_idx = process_detection(detected, steps, current_idx)
            if current_idx > prev_idx:
                # Flush remaining queue to prevent ghost detections skipping steps
                while not tool_queue.empty():
                    try:
                        tool_queue.get_nowait()
                    except queue.Empty:
                        break
                break

    # ── 2b. Drain Clarity feedback ──
    proctor = get_live_proctor()
    if proctor is not None:
        new_fb = proctor.drain_all_feedback()
        if new_fb:
            st.session_state.clarity_feedback.extend(new_fb)

    # Auto-refresh every 2s — drives timer countdown + YOLO queue drain
    if st_autorefresh is not None and st.session_state.app_state == "TRAINING":
        st_autorefresh(interval=2000, key="timer_poll")

    # ── 3. Update shared target for the video callback thread ──
    if current_idx < n_steps:
        target_tool_key = steps[current_idx]["target_tool_key"]
    else:
        target_tool_key = ""

    syllabus_keys = {s["target_tool_key"] for s in steps}
    wrong_pool = [k for k in STANDARD_TOOL_KEYS if k not in syllabus_keys]
    all_tools = [s["target_tool_key"] for s in steps]
    set_shared_target(target_tool_key, all_tools, wrong_pool)

    # ── 70 / 30 layout ──
    module = st.session_state.module_name
    col_video, col_syllabus = st.columns([7, 3])

    # ── LEFT COLUMN: Camera feed OR Demo Tray ──
    with col_video:
        if is_demo:
            # ─────────────────────────────────────
            # DEMO MODE — Virtual Instrument Tray
            # ─────────────────────────────────────
            st.markdown(f"### 🧪 Virtual Instrument Tray — {module}")

            if current_idx < n_steps:
                st.info(
                    f"**Current objective:** {steps[current_idx]['instruction']}  \n"
                    f"Pick the correct tool: **`{steps[current_idx]['target_tool_key']}`**"
                )
            else:
                st.success("All steps complete! Reset to try another procedure.")

            # Build tray: syllabus tools + distractors, shuffled
            tray_tools = list(syllabus_keys)
            distractors = [k for k in STANDARD_TOOL_KEYS if k not in syllabus_keys]
            random.seed(42)
            tray_tools += random.sample(distractors, min(5, len(distractors)))
            tray_tools.sort()

            training_active = st.session_state.app_state == "TRAINING"

            n_cols = 4
            for row_start in range(0, len(tray_tools), n_cols):
                row_tools = tray_tools[row_start:row_start + n_cols]
                cols_tray = st.columns(n_cols)
                for col_idx, tool_key in enumerate(row_tools):
                    with cols_tray[col_idx]:
                        is_target = (current_idx < n_steps and tool_key == steps[current_idx]["target_tool_key"])
                        already_done = tool_key in {steps[i]["target_tool_key"] for i in st.session_state.completed_steps}

                        btn_type = "primary" if is_target else "secondary"
                        icon = "✅" if already_done else "🔧"

                        if st.button(
                            f"{icon} {tool_key}",
                            key=f"demo_{tool_key}",
                            use_container_width=True,
                            type=btn_type,
                            disabled=not training_active,
                        ):
                            st.session_state.demo_injection = tool_key
                            st.rerun()
        else:
            # ─────────────────────────────────────
            # CAMERA MODE — WebRTC
            # ─────────────────────────────────────
            st.markdown(f"### 🎥 Live Feed — {module}")

            if st.session_state.app_state == "TRAINING":
                st.markdown('<div class="video-container">', unsafe_allow_html=True)

                ctx = webrtc_streamer(
                    key="surgipath-training-feed",
                    mode=WebRtcMode.SENDRECV,
                    video_frame_callback=video_frame_callback,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                )

                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info(
                    "📷 Camera feed stopped — session complete. "
                    "Click **Finalize & Grade** below for your performance analysis."
                )

        # ── Countdown Timer Bar ──
        if (
            current_idx < n_steps
            and st.session_state.app_state == "TRAINING"
            and st.session_state.step_start_time > 0
        ):
            _tl = steps[current_idx].get("time_limit_seconds", 60)
            _elapsed = time.time() - st.session_state.step_start_time
            _left = max(0.0, _tl - _elapsed)
            _pct = max(0.0, min(1.0, _left / _tl)) * 100

            if _left < 10:
                _bar_color = "#FF3B30"
                _timer_label = f"{int(_left)}s remaining"
                _border = "border:1px solid rgba(255,59,48,0.3);"
            elif _left < 20:
                _bar_color = "#FF9500"
                _timer_label = f"{int(_left)}s / {_tl}s"
                _border = "border:1px solid rgba(255,149,0,0.2);"
            else:
                _bar_color = "#007AFF"
                _timer_label = f"{int(_left)}s / {_tl}s"
                _border = "border:1px solid #333;"

            st.markdown(
                f'<div style="background:#1E1E1E;border-radius:6px;padding:0.4rem 0.6rem;'
                f'margin:0.5rem 0;{_border}">'
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:center;margin-bottom:4px;">'
                f'<span style="color:{_bar_color};font-weight:500;'
                f'font-size:0.8rem;">{_timer_label}</span>'
                f'<span style="color:#666;font-size:0.7rem;font-weight:300;">'
                f'{steps[current_idx]["step_name"]}</span></div>'
                f'<div style="background:#121212;border-radius:3px;height:4px;">'
                f'<div style="width:{_pct:.1f}%;height:100%;background:{_bar_color};'
                f'border-radius:3px;transition:width 1s linear;"></div></div></div>',
                unsafe_allow_html=True,
            )

        # ── Clarity rationale box (both modes) ──
        rationale = st.session_state.latest_rationale
        if rationale:
            st.markdown(
                f'<div class="tutor-box has-tip">'
                f'<div class="tutor-label">Clarity — Rationale</div>'
                f'<div class="tutor-text">{rationale}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            if current_idx < n_steps:
                action = "Click" if is_demo else "Present"
                hint = (
                    f"{action} <strong>{steps[current_idx]['target_tool_key']}</strong> "
                    f"to proceed."
                )
            else:
                hint = "All steps completed!"
            st.markdown(
                f'<div class="tutor-box">'
                f'<div class="tutor-label">Clarity — Rationale</div>'
                f'<div class="tutor-text">{hint}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── RIGHT COLUMN: Syllabus Timeline ──
    with col_syllabus:
        st.markdown("### 📋 Syllabus Timeline")

        _skipped_set = {s["step_idx"] for s in st.session_state.skipped_steps}
        for i, step in enumerate(steps):
            is_failed = i in _skipped_set
            if i in st.session_state.completed_steps and is_failed:
                css_cls = "step-completed"
                icon = "✕"
                task_html = f'<del style="color:#9E9E9E;">{step["step_name"]}</del>'
                num_style = (
                    'border-color:#FF3B30;color:#fff;background:#FF3B30;'
                )
            elif i in st.session_state.completed_steps:
                css_cls = "step-completed"
                icon = "✓"
                task_html = step["step_name"]
                num_style = ""
            elif i == current_idx:
                css_cls = "step-active"
                icon = str(i + 1)
                task_html = step["step_name"]
                num_style = ""
            else:
                css_cls = "step-pending"
                icon = str(i + 1)
                task_html = step["step_name"]
                num_style = ""

            _step_time = step.get("time_limit_seconds", 60)
            num_extra = f' style="{num_style}"' if num_style else ""
            st.markdown(
                f'<div class="syllabus-step {css_cls}">'
                f'  <div class="step-header">'
                f'    <div class="step-num"{num_extra}>{icon}</div>'
                f'    <div class="step-task">{task_html}</div>'
                f"  </div>"
                f'  <div class="step-tool">{step["instruction"]}</div>'
                f'  <div class="step-tool">Target: <code>{step["target_tool_key"]}</code>'
                f' &middot; <span style="color:#9E9E9E;">{_step_time}s</span></div>'
                f"</div>",
                unsafe_allow_html=True,
            )

        if st.session_state.app_state == "COMPLETE":
            st.markdown(
                '<div class="completion-banner">'
                "<h2>🏆 Module Complete!</h2>"
                "<p>All instruments identified. Reset to try another procedure.</p>"
                "</div>",
                unsafe_allow_html=True,
            )

    # ── Proctor Feedback panel ──
    st.markdown("### Clarity — Clinical Evaluation")

    if st.session_state.proctor_log:
        feed_html = '<div class="proctor-panel">'
        feed_html += '<div class="panel-title">Clarity Reasoning Log</div>'
        for entry in st.session_state.proctor_log:
            cls = "entry-correct" if entry["type"] == "correct" else "entry-correction"
            if entry["type"] == "skip":
                cls = "entry-correction"
            feed_html += (
                f'<div class="proctor-entry {cls}">'
                f'<div class="entry-time">{entry["time"]}</div>'
                f'{entry["text"]}'
                f"</div>"
            )
        feed_html += "</div>"
        st.markdown(feed_html, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="proctor-panel">'
            '<div class="panel-title">Clarity Reasoning Log</div>'
            '<div class="proctor-entry entry-thinking">'
            '<div class="entry-time">--:--:--</div>'
            "Awaiting first detection to begin evaluation…"
            "</div></div>",
            unsafe_allow_html=True,
        )

    # ── Manual Override / Skip Tool ──
    if st.session_state.app_state == "TRAINING" and current_idx < n_steps:
        st.markdown("---")
        with st.expander("⚠️ Tool Unavailable / Manual Advance", expanded=False):
            skip_reason = st.text_input(
                "Reason for skipping",
                placeholder="e.g. instrument not in kit, broken, unavailable…",
                key="skip_reason_input",
            )
            if st.button(
                f"⏭️ Skip {steps[current_idx]['target_tool_key']} (−{SKIP_PENALTY} pts)",
                use_container_width=True,
                type="secondary",
                key="manual_advance_btn",
            ):
                st.session_state["_skip_requested"] = {"reason": skip_reason}
                st.rerun()

    # ── Post-Op Report (COMPLETE state) ──
    if st.session_state.app_state == "COMPLETE":
        _done_proctor = get_live_proctor()
        if _done_proctor is not None and _done_proctor.active:
            _done_proctor.stop()

        st.markdown("---")
        st.markdown("### Post-Op Session Report")

        if not st.session_state.session_report:
            with st.spinner("Clarity is generating your session report…"):
                report = generate_session_report(
                    procedure=st.session_state.module_name,
                    total_steps=n_steps,
                    verified_count=n_steps - len(st.session_state.skipped_steps),
                    skipped_steps=st.session_state.skipped_steps,
                    mastery_score=st.session_state.mastery_score,
                )
            st.session_state.session_report = report

        score = st.session_state.mastery_score
        skipped_count = len(st.session_state.skipped_steps)
        verified_count = n_steps - skipped_count

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            if score > 80:
                score_color = "#34C759"
            elif score >= 50:
                score_color = "#FF9500"
            else:
                score_color = "#FF3B30"
            st.markdown(
                f'<div style="text-align:center;padding:1rem;'
                f'background:#1E1E1E;border:1px solid #333;border-radius:8px;">'
                f'<div style="font-size:2rem;font-weight:600;'
                f'color:{score_color};">{score}/100</div>'
                f'<div style="color:#666;font-size:0.65rem;text-transform:uppercase;'
                f'letter-spacing:0.8px;font-weight:400;">Mastery Score</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with rc2:
            st.markdown(
                f'<div style="text-align:center;padding:1rem;'
                f'background:#1E1E1E;border:1px solid #333;border-radius:8px;">'
                f'<div style="font-size:2rem;font-weight:600;'
                f'color:#fff;">{verified_count}</div>'
                f'<div style="color:#666;font-size:0.65rem;text-transform:uppercase;'
                f'letter-spacing:0.8px;font-weight:400;">Vision Verified</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with rc3:
            skip_color = "#FF3B30" if skipped_count > 0 else "#34C759"
            st.markdown(
                f'<div style="text-align:center;padding:1rem;'
                f'background:#1E1E1E;border:1px solid #333;border-radius:8px;">'
                f'<div style="font-size:2rem;font-weight:600;'
                f'color:{skip_color};">{skipped_count}</div>'
                f'<div style="color:#666;font-size:0.65rem;text-transform:uppercase;'
                f'letter-spacing:0.8px;font-weight:400;">Manual Overrides</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("")
        st.markdown(st.session_state.session_report)

        # ── Learning Resources ──
        st.markdown("---")
        st.markdown("### Recommended Resources")

        if not st.session_state.learning_resources:
            with st.spinner("Clarity is curating learning resources…"):
                resources = generate_learning_resources(
                    procedure=st.session_state.module_name,
                )
            st.session_state.learning_resources = resources

        st.markdown(
            '<div style="background:#1E1E1E;border:1px solid #333;'
            'border-left:3px solid #007AFF;'
            'border-radius:8px;padding:1.2rem 1.5rem;margin-top:0.5rem;">'
            '<div style="color:#007AFF;font-size:0.65rem;font-weight:500;'
            'text-transform:uppercase;letter-spacing:0.8px;'
            'margin-bottom:0.6rem;">Study Materials</div>',
            unsafe_allow_html=True,
        )
        st.markdown(st.session_state.learning_resources)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Finalize & Grade — Clarity Performance Note ──
        st.markdown("---")

        if not st.session_state.final_critique:
            n_obs = len(st.session_state.clarity_feedback)
            if n_obs > 0:
                st.caption(
                    f"Clarity recorded **{n_obs}** live observations "
                    f"during this session."
                )
            else:
                st.caption(
                    "No live observations recorded. Clarity will analyze "
                    "the event log."
                )

            if st.button(
                "Finalize & Grade",
                use_container_width=True,
                type="primary",
            ):
                with st.status(
                    "Clarity is analyzing your session…", expanded=True
                ) as status:
                    st.write(
                        f"Processing **{n_obs}** observations…"
                    )
                    st.write(
                        "Generating Surgical Performance Note…"
                    )
                    critique = generate_final_critique(
                        procedure=st.session_state.module_name,
                        clarity_feedback=st.session_state.clarity_feedback,
                        event_log=st.session_state.event_log,
                        mastery_score=st.session_state.mastery_score,
                    )
                    status.update(
                        label="Analysis complete!", state="complete"
                    )
                st.session_state.final_critique = critique
                st.rerun()

        if st.session_state.final_critique:
            st.markdown(
                '<div style="background:#1E1E1E;border:1px solid #333;'
                'border-left:3px solid #007AFF;'
                'border-radius:8px;padding:1.5rem 2rem;margin-top:0.5rem;">'
                '<h3 style="color:#fff;margin:0 0 0.3rem;font-weight:500;">'
                'Clarity — Performance Note</h3>'
                '<hr style="border-color:#333;margin:0.3rem 0 1rem;">',
                unsafe_allow_html=True,
            )
            st.markdown(st.session_state.final_critique)
            st.markdown('</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────

st.markdown("---")
ft1, ft2, ft3 = st.columns(3)
with ft1:
    st.caption("SurgiPath v1.0")
with ft2:
    st.caption("Clarity AI · Gemini Live · YOLOv11")
with ft3:
    st.caption(f"Session: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
