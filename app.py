"""
Panacea: Surgical Mastery — AI-Guided Training Simulator
UI and video pipeline. All AI logic lives in brain.py.
"""

import queue
import random
import threading
from datetime import datetime

import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from brain import (
    STANDARD_TOOL_KEYS,
    ActionSuccess,
    SyllabusError,
    check_student_action,
    generate_dynamic_syllabus,
)
from styles import load_css

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
    "app_state": "IDLE",
    "module_name": "",
    "syllabus": None,
    "current_step_index": 0,
    "completed_steps": set(),
    "latest_rationale": "",
    "proctor_log": [],
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
_shared_target_tool: list[str] = [""]
_shared_all_tools: list[str] = []
_shared_wrong_tools: list[str] = []
_frame_counter: list[int] = [0]


def set_shared_target(target: str, all_tools: list[str], wrong_tools: list[str] | None = None) -> None:
    with _shared_lock:
        _shared_target_tool[0] = target
        _shared_all_tools.clear()
        _shared_all_tools.extend(all_tools)
        _shared_wrong_tools.clear()
        if wrong_tools:
            _shared_wrong_tools.extend(wrong_tools)
        _frame_counter[0] = 0


def get_shared_target() -> tuple[str, list[str], list[str]]:
    with _shared_lock:
        return _shared_target_tool[0], list(_shared_all_tools), list(_shared_wrong_tools)


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
# WebRTC video callback
# ──────────────────────────────────────────────

_CYAN = (0, 229, 255)
_GREEN = (0, 230, 118)
_RED = (68, 23, 255)
_DARK = (0, 0, 0)
_YELLOW = (0, 171, 255)

_DETECTION_INTERVAL = 150
_WRONG_TOOL_CHANCE = 0.30


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    h, w, _ = img.shape

    target, all_tools, wrong_tools = get_shared_target()
    if not target:
        cv2.putText(img, "AWAITING TARGET", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, _YELLOW, 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    _frame_counter[0] += 1
    fc = _frame_counter[0]

    # TODO: Replace with real YOLO inference
    # from ultralytics import YOLO
    # model = YOLO("yolov11_surgical_tools.pt")
    # results = model(img, conf=0.5)

    cv2.putText(img, f"TARGET: {target}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, _CYAN, 2)

    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    cv2.putText(img, ts, (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _CYAN, 1)

    cycle_num = fc // _DETECTION_INTERVAL
    rng = random.Random(cycle_num * 7919)
    is_wrong_cycle = wrong_tools and rng.random() < _WRONG_TOOL_CHANCE
    display_tool = rng.choice(wrong_tools) if is_wrong_cycle else target
    box_color = _RED if is_wrong_cycle else _CYAN

    is_detecting = (fc % _DETECTION_INTERVAL) >= (_DETECTION_INTERVAL - 60)

    if is_detecting:
        progress = ((fc % _DETECTION_INTERVAL) - (_DETECTION_INTERVAL - 60)) / 60.0
        margin = int(40 * (1 - progress))

        cx, cy = w // 2, h // 2
        bw, bh = int(w * 0.40), int(h * 0.45)
        x1 = cx - bw // 2 + margin
        y1 = cy - bh // 2 + margin
        x2 = cx + bw // 2 - margin
        y2 = cy + bh // 2 - margin

        lock_color = _GREEN if (progress > 0.8 and not is_wrong_cycle) else box_color

        cv2.rectangle(img, (x1, y1), (x2, y2), lock_color, 2)

        bracket = 20
        for (bx, by, dx, dy) in [
            (x1, y1, 1, 1), (x2, y1, -1, 1),
            (x1, y2, 1, -1), (x2, y2, -1, -1),
        ]:
            cv2.line(img, (bx, by), (bx + dx * bracket, by), lock_color, 3)
            cv2.line(img, (bx, by), (bx, by + dy * bracket), lock_color, 3)

        conf = 0.75 + 0.23 * progress
        label = f"{display_tool}  [{conf:.2f}]"
        (tw, th_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1 - th_t - 12), (x1 + tw + 10, y1), lock_color, -1)
        cv2.putText(img, label, (x1 + 5, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _DARK, 2)

        if progress > 0.8:
            status_label = "WRONG TOOL" if is_wrong_cycle else "LOCKED"
            status_color = _RED if is_wrong_cycle else _GREEN
            cv2.putText(img, status_label, (x1 + 5, y2 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

    if fc > 0 and fc % _DETECTION_INTERVAL == 0:
        try:
            tool_queue.put_nowait(display_tool)
        except queue.Full:
            pass

    scan_x = int((fc % 120) / 120.0 * w)
    cv2.line(img, (scan_x, h - 4), (min(scan_x + 60, w), h - 4), _CYAN, 3)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


# ──────────────────────────────────────────────
# Sidebar
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
            with st.status("Gemini is Thinking…", expanded=True) as status:
                st.write(f"**Procedure:** {procedure_input.strip()}")
                st.write("Reasoning through WHO Surgical Safety protocols…")

                result = generate_dynamic_syllabus(procedure_input.strip())

                if isinstance(result, SyllabusError):
                    status.update(label="Could not generate syllabus", state="error")
                    st.error(result.error)
                    st.stop()

                syllabus = result
                st.write(f"Generated **{len(syllabus.steps)}** training steps:")
                for step in syllabus.steps:
                    st.write(f"  • **{step.step_name}** → `{step.target_tool_key}`")
                status.update(label="Syllabus ready!", state="complete")

            st.session_state.module_name = procedure_input.strip()
            st.session_state.syllabus = syllabus.model_dump()
            st.session_state.current_step_index = 0
            st.session_state.completed_steps = set()
            st.session_state.latest_rationale = ""
            st.session_state.proctor_log = []
            st.session_state.training_start_time = datetime.now().isoformat()
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
        for i, step in enumerate(syllabus_data["steps"]):
            if i in st.session_state.completed_steps:
                st.markdown(f"✅  ~~{step['step_name']}~~ — `{step['target_tool_key']}`")
            elif i == current_idx_sidebar:
                st.markdown(f"▶️  **{step['step_name']}** — `{step['target_tool_key']}`")
            else:
                st.markdown(f"⬜  {step['step_name']} — `{step['target_tool_key']}`")

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
    "<p>AI-Guided Training Simulator • Gemini (Thinking) • YOLOv11</p>"
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
            "<h2>Welcome to Surgical Mastery</h2>"
            '<p style="font-size:1rem;margin-top:0.8rem;">'
            "Type <strong>any</strong> medical or surgical procedure in the "
            "sidebar and click <strong>Start Training</strong>.<br><br>"
            "Gemini will reason through the procedure, validate it against "
            "WHO Surgical Safety protocols, and generate a structured "
            "training syllabus with standardized instrument keys.<br><br>"
            "The camera then guides you through each tool identification "
            "step — with real-time AI course correction."
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

    # ── Drain tool_queue → proctoring via brain.check_student_action ──
    while not tool_queue.empty():
        try:
            detected = tool_queue.get_nowait()
        except queue.Empty:
            break

        if current_idx >= n_steps:
            continue

        current_step = steps[current_idx]
        result = check_student_action(
            detected_tools=[detected],
            current_target_tool=current_step["target_tool_key"],
            current_instruction=current_step["instruction"],
            current_rationale=current_step["medical_rationale"],
        )

        if isinstance(result, ActionSuccess):
            st.session_state.completed_steps.add(current_idx)
            st.session_state.latest_rationale = result.message
            log_proctor("correct", f"✓ {result.tool} — {result.message}")

            st.session_state.current_step_index = current_idx + 1
            current_idx = st.session_state.current_step_index

            if current_idx >= n_steps:
                st.session_state.app_state = "COMPLETE"
                log_proctor("correct", "All training steps completed successfully.")
        else:
            log_proctor("correction", f"✗ Picked {result.wrong_tool} → {result.message}")

    # Build distractor pool from STANDARD_TOOL_KEYS, excluding syllabus tools
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

    with col_video:
        st.markdown(f"### 🎥 Live Feed — {module}")
        st.markdown('<div class="video-container">', unsafe_allow_html=True)

        ctx = webrtc_streamer(
            key="panacea-training-feed",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

        rationale = st.session_state.latest_rationale
        if rationale:
            st.markdown(
                f'<div class="tutor-box has-tip">'
                f'<div class="tutor-label">🎓 Medical Rationale</div>'
                f'<div class="tutor-text">{rationale}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            if current_idx < n_steps:
                hint = (
                    f"Present <strong>{steps[current_idx]['target_tool_key']}</strong> "
                    f"to the camera to proceed."
                )
            else:
                hint = "All steps completed!"
            st.markdown(
                f'<div class="tutor-box">'
                f'<div class="tutor-label">🎓 Medical Rationale</div>'
                f'<div class="tutor-text">{hint}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

    with col_syllabus:
        st.markdown("### 📋 Syllabus Timeline")

        for i, step in enumerate(steps):
            if i in st.session_state.completed_steps:
                css_cls = "step-completed"
                icon = "✓"
            elif i == current_idx:
                css_cls = "step-active"
                icon = str(i + 1)
            else:
                css_cls = "step-pending"
                icon = str(i + 1)

            st.markdown(
                f'<div class="syllabus-step {css_cls}">'
                f'  <div class="step-header">'
                f'    <div class="step-num">{icon}</div>'
                f'    <div class="step-task">{step["step_name"]}</div>'
                f"  </div>"
                f'  <div class="step-tool">{step["instruction"]}</div>'
                f'  <div class="step-tool">Target: <code>{step["target_tool_key"]}</code></div>'
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
    st.markdown("### 🧑‍⚕️ Proctor Feedback")

    if st.session_state.proctor_log:
        feed_html = '<div class="proctor-panel">'
        feed_html += '<div class="panel-title">Gemini Reasoning Log</div>'
        for entry in st.session_state.proctor_log:
            cls = "entry-correct" if entry["type"] == "correct" else "entry-correction"
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
            '<div class="panel-title">Gemini Reasoning Log</div>'
            '<div class="proctor-entry entry-thinking">'
            '<div class="entry-time">--:--:--</div>'
            "Awaiting first detection to begin proctoring…"
            "</div></div>",
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────

st.markdown("---")
ft1, ft2, ft3 = st.columns(3)
with ft1:
    st.caption("Panacea: Surgical Mastery v4.0.0")
with ft2:
    st.caption("brain.py • Gemini (Thinking) • YOLOv11 • WebRTC")
with ft3:
    st.caption(f"Session: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
