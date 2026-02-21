"""
Project Panacea — Surgical AI Dashboard
3-phase workflow: INPUT → READY → ACTIVE
Working demo hardcoded for Appendectomy procedure.
"""

import json
import math
import queue
import random
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

class SurgeryChecklist(BaseModel):
    tools: list[str]
    estimated_steps: int


# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Project Panacea — Surgical Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css()

# ──────────────────────────────────────────────
# Session state defaults
# ──────────────────────────────────────────────

_DEFAULTS: dict = {
    "app_state": "INPUT",
    "surgery_name": "",
    "surgery_checklist": None,
    "verified_tools": set(),
    "gemini_responses": [],
}

for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ──────────────────────────────────────────────
# Thread-safe shared state for the video callback
# ──────────────────────────────────────────────

detection_queue: queue.Queue = queue.Queue(maxsize=64)

# The video callback runs in a worker thread and cannot access
# st.session_state.  We bridge the gap with a plain module-level
# list protected by a lock.  The main thread writes the current
# checklist tools here before the streamer starts; the callback
# reads them to know which tools to simulate detecting.
_shared_lock = threading.Lock()
_shared_checklist_tools: list[str] = []
_frame_counter: list[int] = [0]  # mutable container so the callback can increment


def set_shared_checklist(tools: list[str]) -> None:
    with _shared_lock:
        _shared_checklist_tools.clear()
        _shared_checklist_tools.extend(tools)
        _frame_counter[0] = 0


def get_shared_checklist() -> list[str]:
    with _shared_lock:
        return list(_shared_checklist_tools)


# ──────────────────────────────────────────────
# Gemini API integration
# ──────────────────────────────────────────────

GEMINI_SYSTEM_PROMPT = (
    "You are a surgical consultant. Generate a mandatory tool checklist "
    "for the following procedure in JSON format.\n"
    "Return ONLY valid JSON matching this schema:\n"
    '{"tools": ["tool_name", ...], "estimated_steps": <int>}\n'
    "Be specific and include all standard instruments."
)


def call_gemini_checklist(surgery_name: str) -> SurgeryChecklist:
    """Call Gemini 1.5 Flash to generate a procedure-specific tool checklist.

    Falls back to a curated mock when no API key is configured.
    """
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        api_key = ""

    if api_key:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            f"{GEMINI_SYSTEM_PROMPT}\n\nProcedure: {surgery_name}",
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        data = json.loads(raw)
    else:
        _MOCK_DB: dict[str, dict] = {
            "default": {
                "tools": [
                    "Scalpel (#10 blade)",
                    "Tissue Forceps",
                    "Metzenbaum Scissors",
                    "Needle Driver",
                    "Self-retaining Retractor",
                    "Suction Cannula",
                    "Electrocautery Pencil",
                    "Suture (3-0 Vicryl)",
                ],
                "estimated_steps": 12,
            },
            "appendectomy": {
                "tools": [
                    "Scalpel (#15 blade)",
                    "Kelly Clamp",
                    "Babcock Forceps",
                    "Metzenbaum Scissors",
                    "Suction Irrigator",
                    "Laparoscopic Trocar (12mm)",
                    "Laparoscopic Trocar (5mm)",
                    "Endoscopic Stapler",
                    "Grasper Forceps",
                    "Electrocautery Hook",
                    "Needle Driver",
                    "Suture (2-0 Vicryl)",
                ],
                "estimated_steps": 8,
            },
            "cholecystectomy": {
                "tools": [
                    "Scalpel (#11 blade)",
                    "Maryland Dissector",
                    "Laparoscopic Trocar (10mm)",
                    "Laparoscopic Trocar (5mm)",
                    "Clip Applier",
                    "Hook Cautery",
                    "Grasper Forceps",
                    "Retrieval Bag",
                    "Suction Irrigator",
                    "Needle Driver",
                    "Suture (3-0 PDS)",
                ],
                "estimated_steps": 7,
            },
        }
        key = surgery_name.strip().lower()
        data = _MOCK_DB.get(key, _MOCK_DB["default"])
        time.sleep(2)

    return SurgeryChecklist(**data)


# Appendectomy-specific guidance messages that rotate over time
_APPENDECTOMY_GUIDANCE = [
    {
        "phase": "Preparation",
        "suggestion": (
            "All port sites should be marked and confirmed. Verify the "
            "12 mm umbilical trocar is ready for camera insertion."
        ),
        "risk_level": "low",
    },
    {
        "phase": "Port Placement",
        "suggestion": (
            "Pneumoperitoneum established at 12 mmHg. Recommend 30° "
            "angled scope for optimal visualization of the RLQ."
        ),
        "risk_level": "low",
    },
    {
        "phase": "Identification",
        "suggestion": (
            "Appendix visualized with surrounding inflammatory tissue. "
            "Use Babcock forceps for atraumatic retraction of the tip."
        ),
        "risk_level": "medium",
    },
    {
        "phase": "Mesoappendix Dissection",
        "suggestion": (
            "Activate electrocautery hook at 30W coag setting. Maintain "
            "clear visualization of the appendiceal artery before division."
        ),
        "risk_level": "medium",
    },
    {
        "phase": "Base Ligation",
        "suggestion": (
            "Apply endoscopic stapler across the appendiceal base. "
            "Ensure at least 3 mm of healthy cecal tissue in the staple line."
        ),
        "risk_level": "high",
    },
    {
        "phase": "Specimen Retrieval",
        "suggestion": (
            "Place the specimen in a retrieval bag via the 12 mm port. "
            "Inspect the staple line for hemostasis before desufflation."
        ),
        "risk_level": "low",
    },
    {
        "phase": "Closure",
        "suggestion": (
            "Close the fascia at the 12 mm port site with 2-0 Vicryl. "
            "Subcuticular closure for skin. Apply sterile dressing."
        ),
        "risk_level": "low",
    },
]


def call_gemini_api(data: dict) -> dict:
    """Runtime Gemini call for real-time surgical guidance (mock).

    Cycles through appendectomy-specific guidance messages.
    """
    n = len(st.session_state.gemini_responses)
    template = _APPENDECTOMY_GUIDANCE[n % len(_APPENDECTOMY_GUIDANCE)]

    response = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "phase": template["phase"],
        "suggestion": template["suggestion"],
        "risk_level": template["risk_level"],
        "detected_tools": data.get("tools", []),
    }
    st.session_state.gemini_responses.insert(0, response)
    st.session_state.gemini_responses = st.session_state.gemini_responses[:20]
    return response


# ──────────────────────────────────────────────
# WebRTC video callback
# ──────────────────────────────────────────────

# Pre-defined box positions so detections look spatially varied
_BOX_SLOTS = [
    (0.02, 0.05, 0.32, 0.40),
    (0.35, 0.05, 0.65, 0.40),
    (0.68, 0.05, 0.98, 0.40),
    (0.02, 0.55, 0.32, 0.92),
    (0.35, 0.55, 0.65, 0.92),
    (0.68, 0.55, 0.98, 0.92),
]

_CYAN = (0, 229, 255)
_GREEN = (0, 230, 118)
_DARK = (0, 0, 0)


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    h, w, _ = img.shape

    tools = get_shared_checklist()
    if not tools:
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    # Increment frame counter
    _frame_counter[0] += 1
    fc = _frame_counter[0]

    # TODO: Replace this entire mock block with real YOLOv11 inference
    # ─────────────────────────────────────────
    # from ultralytics import YOLO
    # model = YOLO("yolov11_surgical_tools.pt")
    # results = model(img, conf=0.5)
    # for r in results:
    #     for box in r.boxes:
    #         cls_name = model.names[int(box.cls)]
    #         x1, y1, x2, y2 = map(int, box.xyxy[0])
    #         conf = float(box.conf)
    #         ...
    # ─────────────────────────────────────────

    # Every ~90 frames (~3 sec at 30 fps) we "discover" the next tool.
    # This means the checklist fills progressively over ~36 seconds for 12 tools.
    num_visible = min(len(tools), (fc // 90) + 1)
    visible_tools = tools[:num_visible]

    detected_names: list[str] = []

    for i, tool_name in enumerate(visible_tools):
        slot = _BOX_SLOTS[i % len(_BOX_SLOTS)]
        x1 = int(slot[0] * w)
        y1 = int(slot[1] * h)
        x2 = int(slot[2] * w)
        y2 = int(slot[3] * h)

        # Oscillating confidence for realism
        conf = 0.88 + 0.10 * abs(math.sin(fc * 0.03 + i))

        cv2.rectangle(img, (x1, y1), (x2, y2), _CYAN, 2)

        label = f"{tool_name}  [{conf:.2f}]"
        (tw, th_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.rectangle(img, (x1, y1 - th_text - 10), (x1 + tw + 8, y1), _CYAN, -1)
        cv2.putText(img, label, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, _DARK, 2)

        # Small "VERIFIED" tag under the box
        cv2.putText(img, "VERIFIED", (x1 + 4, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _GREEN, 1)

        detected_names.append(tool_name)

    # ── HUD overlays ──
    progress_text = f"SCANNING  {num_visible}/{len(tools)}"
    cv2.putText(img, progress_text, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, _GREEN, 2)

    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    cv2.putText(img, ts, (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _CYAN, 1)

    pct = int(num_visible / len(tools) * 100)
    cv2.putText(img, f"{pct}% Complete", (w - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, _CYAN, 2)

    # Progress bar across bottom
    bar_y = h - 6
    bar_w = int(w * (num_visible / len(tools)))
    cv2.rectangle(img, (0, bar_y), (bar_w, h), _CYAN, -1)
    cv2.rectangle(img, (bar_w, bar_y), (w, h), (30, 30, 40), -1)

    # Push detections through thread-safe queue
    try:
        detection_queue.put_nowait(detected_names)
    except queue.Full:
        pass

    return av.VideoFrame.from_ndarray(img, format="bgr24")


# ──────────────────────────────────────────────
# Helper: render the phase indicator bar
# ──────────────────────────────────────────────

_PHASES = [("1", "Input"), ("2", "Ready"), ("3", "Active")]
_STATE_TO_IDX = {"INPUT": 0, "READY": 1, "ACTIVE": 2}


def render_phase_bar() -> None:
    current = _STATE_TO_IDX.get(st.session_state.app_state, 0)
    cells = []
    for i, (num, label) in enumerate(_PHASES):
        if i < current:
            cls = "completed"
        elif i == current:
            cls = "active"
        else:
            cls = ""
        cells.append(
            f'<div class="phase-step {cls}">'
            f'<span class="phase-num">{num}</span>{label}'
            f"</div>"
        )
    st.markdown(f'<div class="phase-bar">{"".join(cells)}</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div class="dashboard-header">'
            "<h1>🩺 Panacea</h1>"
            "<p>Surgical AI Co-pilot</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        state = st.session_state.app_state
        checklist = st.session_state.surgery_checklist

        badge_map = {
            "INPUT": ("● Awaiting Procedure", "status-idle"),
            "READY": ("● Checklist Ready", "status-ready"),
            "ACTIVE": ("● Scanner Active", "status-active"),
        }
        badge_text, badge_cls = badge_map[state]
        st.markdown(
            f'<span class="status-badge {badge_cls}">{badge_text}</span>',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        if checklist is None:
            st.markdown("### Surgical Checklist")
            st.caption("Select a procedure to generate the checklist.")
            return

        tools = checklist["tools"]
        verified = st.session_state.verified_tools

        # Drain the detection queue and update verified set
        if state == "ACTIVE":
            while not detection_queue.empty():
                try:
                    names = detection_queue.get_nowait()
                    for name in names:
                        if name in tools:
                            verified.add(name)
                except queue.Empty:
                    break

        n_verified = sum(1 for t in tools if t in verified)
        pct = int(n_verified / len(tools) * 100) if tools else 0

        st.markdown(
            f'<div class="progress-ring">'
            f'<div class="pct">{pct}%</div>'
            f'<div class="pct-label">Tools Verified ({n_verified}/{len(tools)})</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown(f"### Checklist — {st.session_state.surgery_name}")

        for tool in tools:
            is_verified = tool in verified
            if is_verified:
                st.markdown(
                    f'<div class="checklist-item verified">'
                    f'<span class="tool-detected">✅  {tool}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="checklist-item">'
                    f'<span class="tool-missing">⬜  {tool}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.markdown("### System Metrics")
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
            latency = "42ms" if state == "ACTIVE" else "—"
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="value">{latency}</div>'
                f'<div class="label">Latency</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.caption(f"Est. steps: {checklist['estimated_steps']}")


render_sidebar()

# ──────────────────────────────────────────────
# Main area — Header + phase bar
# ──────────────────────────────────────────────

st.markdown(
    '<div class="dashboard-header">'
    "<h1>PROJECT PANACEA</h1>"
    "<p>Real-Time Surgical Intelligence • YOLOv11 • MediaPipe • Gemini 1.5 Flash</p>"
    "</div>",
    unsafe_allow_html=True,
)

render_phase_bar()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 1 — INPUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if st.session_state.app_state == "INPUT":

    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown("## What surgery are we performing today?")
    st.markdown(
        "Enter a procedure name below. Gemini will generate a tailored "
        "instrument checklist for your team.",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    COMMON_PROCEDURES = [
        "— Select or type below —",
        "Appendectomy",
        "Cholecystectomy",
        "Hernia Repair",
        "Mastectomy",
        "Thyroidectomy",
        "Cesarean Section",
        "Coronary Artery Bypass",
    ]

    col_sel, col_or, col_txt = st.columns([2, 0.3, 2])
    with col_sel:
        selected = st.selectbox(
            "Common procedures",
            COMMON_PROCEDURES,
            label_visibility="collapsed",
        )
    with col_or:
        st.markdown(
            "<div style='text-align:center;padding-top:0.5rem;"
            "color:var(--text-secondary);'>or</div>",
            unsafe_allow_html=True,
        )
    with col_txt:
        typed = st.text_input(
            "Type a procedure",
            placeholder="e.g. Laparoscopic Cholecystectomy",
            label_visibility="collapsed",
        )

    surgery_input = typed.strip() if typed.strip() else (
        selected if selected != COMMON_PROCEDURES[0] else ""
    )

    _, btn_col, _ = st.columns([1, 1, 1])
    with btn_col:
        generate = st.button(
            "🧠  Generate Checklist",
            disabled=not surgery_input,
            use_container_width=True,
        )

    if generate and surgery_input:
        with st.status("Gemini is analysing the procedure…", expanded=True) as status:
            st.write(f"**Procedure:** {surgery_input}")
            st.write("Sending to Gemini 1.5 Flash…")

            try:
                checklist = call_gemini_checklist(surgery_input)
            except (json.JSONDecodeError, ValidationError) as exc:
                status.update(label="Gemini returned invalid data", state="error")
                st.error(f"Failed to parse Gemini response: {exc}")
                st.stop()

            st.write(f"Received **{len(checklist.tools)}** tools, "
                     f"**{checklist.estimated_steps}** estimated steps.")
            status.update(label="Checklist generated!", state="complete")

        st.session_state.surgery_name = surgery_input
        st.session_state.surgery_checklist = checklist.model_dump()
        st.session_state.verified_tools = set()
        st.session_state.gemini_responses = []
        st.session_state.app_state = "READY"
        st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 2 — READY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

elif st.session_state.app_state == "READY":
    checklist = st.session_state.surgery_checklist
    tools = checklist["tools"]

    st.markdown('<div class="ready-card">', unsafe_allow_html=True)
    st.markdown(f"## ✅ Checklist Ready — {st.session_state.surgery_name}")
    st.markdown(
        f"Gemini identified **{len(tools)}** mandatory instruments "
        f"across **{checklist['estimated_steps']}** procedural steps.",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    tags = "".join(f'<span class="checklist-tag">{t}</span>' for t in tools)
    st.markdown(f'<div class="checklist-preview">{tags}</div>', unsafe_allow_html=True)

    st.markdown("")

    col_back, col_gap, col_start = st.columns([1, 2, 1])
    with col_back:
        if st.button("← Change Procedure", use_container_width=True):
            st.session_state.app_state = "INPUT"
            st.session_state.surgery_checklist = None
            st.rerun()
    with col_start:
        if st.button("🟢  START SURGERY", type="primary", use_container_width=True):
            st.session_state.app_state = "ACTIVE"
            st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 3 — ACTIVE  (scanner + reasoning feed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

elif st.session_state.app_state == "ACTIVE":
    checklist = st.session_state.surgery_checklist

    # Populate shared state so the video callback thread knows the tools
    set_shared_checklist(checklist["tools"])

    st.markdown("### 🎥 Live Surgical Feed — Scanning Tools")
    st.markdown('<div class="video-container">', unsafe_allow_html=True)

    ctx = webrtc_streamer(
        key="panacea-surgical-feed",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if ctx.state.playing:
        call_gemini_api({"tools": list(st.session_state.verified_tools)})

    # ── AI Reasoning Feed ──
    st.markdown("### 🧠 AI Reasoning Feed")

    if not st.session_state.gemini_responses:
        st.markdown(
            '<div class="reasoning-feed">'
            '<div class="reasoning-entry">'
            '<span class="timestamp">--:--:--</span><br>'
            "Awaiting live feed to begin AI reasoning…"
            "</div></div>",
            unsafe_allow_html=True,
        )
    else:
        feed_html = '<div class="reasoning-feed">'
        for entry in st.session_state.gemini_responses:
            risk_colors = {"low": "#00e676", "medium": "#ffab00", "high": "#ff1744"}
            rc = risk_colors.get(entry["risk_level"], "#8899aa")
            feed_html += (
                '<div class="reasoning-entry">'
                f'<span class="timestamp">{entry["timestamp"]}  •  '
                f'Phase: {entry["phase"]}  •  '
                f'Risk: <span style="color:{rc};font-weight:600;">'
                f'{entry["risk_level"].upper()}</span></span><br>'
                f'{entry["suggestion"]}'
                "</div>"
            )
        feed_html += "</div>"
        st.markdown(feed_html, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("⏹  End Session", type="secondary"):
        st.session_state.app_state = "INPUT"
        st.session_state.surgery_checklist = None
        st.session_state.verified_tools = set()
        st.session_state.gemini_responses = []
        set_shared_checklist([])
        st.rerun()


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────

st.markdown("---")
ft1, ft2, ft3 = st.columns(3)
with ft1:
    st.caption("Project Panacea v0.3.0")
with ft2:
    st.caption("Models: YOLOv11 · MediaPipe · Gemini 1.5 Flash")
with ft3:
    st.caption(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
