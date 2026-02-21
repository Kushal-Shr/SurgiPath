"""
MedLab Coach AI — AI-Guided Skill Assessment for Medical Training Labs

================================================================================
HOW THIS SCRIPT WORKS (for studying)
================================================================================

This is the main Streamlit entry point. Streamlit re-runs top to bottom on
every user interaction. State persists via st.session_state.

  FLOW:
  1. Page config, CSS
  2. Session state init (mode, smoother, streak, video source, etc.)
  3. Load recipe (required tools, params, rules, expected_time_seconds)
  4. SIDEBAR: branding, video source selector, nav radio, settings
  5. MAIN AREA:
     - Header
     - VIDEO FEED: a single @st.fragment(run_every=1s) grabs camera frames
       and renders them as base64 data-URIs (avoids Streamlit media-file
       storage race conditions). A slower st_autorefresh(4s) triggers full-
       page reruns so the Setup checklist picks up smoother data.
     - SETUP tab: tool checklist, readiness %, "Begin Lab Session" button
     - PRACTICE tab: phase dropdown, alerts panel, streak counter, "End Session"
     - REPORT tab: rubric radar chart (Plotly), metrics, export JSON/CSV

  RUN:
    streamlit run app.py  →  http://localhost:8501
"""

import base64
import time
import json
import random
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

import cv2
import numpy as np
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from styles import load_css
from src.constants import (
    MODEL_PATH, RECIPE_PATH, PHASES, PHASE_LABELS, SAMPLE_VIDEO_PATH,
    CONF_MIN_DEFAULT, IMGSZ_DEFAULT, FRAME_SKIP_DEFAULT,
    SMOOTH_WINDOW_SIZE, FEED_LOOP_FRAMES,
    KEY_NAV, KEY_PREOP_SMOOTHER, KEY_PREOP_STABLE_START,
    KEY_ALERTS_LOG, KEY_LAST_DETECTIONS, KEY_LAST_COUNTS,
    KEY_FRAME_INDEX, KEY_CONFIG_CONF_MIN, KEY_CONFIG_IMGSZ,
    KEY_CONFIG_FRAME_SKIP, KEY_CAMERA,
    KEY_VIDEO_SOURCE, KEY_UPLOADED_FILE,
    KEY_STREAK_SECONDS, KEY_STREAK_BEST,
    KEY_SESSION_START, KEY_HAND_STABILITY,
    KEY_DEMO_MODE, KEY_DEMO_TICK,
)
from src.state import init_state, get_mode, set_mode, get_phase, set_phase
from src.detector import get_model, infer_tools, count_tools, draw_detections
from src.logger import log_event, read_events
from src.rules import evaluate_rules
from src.utils import load_recipe, ToolPresenceSmoother
from src.scoring import compute_rubric, rubric_to_json, events_to_csv

# =============================================================================
# Page config & CSS
# =============================================================================

st.set_page_config(
    page_title="MedLab Coach AI",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_css()

# =============================================================================
# Helpers
# =============================================================================


def get_config() -> dict:
    if KEY_CONFIG_CONF_MIN not in st.session_state:
        st.session_state[KEY_CONFIG_CONF_MIN] = CONF_MIN_DEFAULT
    if KEY_CONFIG_IMGSZ not in st.session_state:
        st.session_state[KEY_CONFIG_IMGSZ] = IMGSZ_DEFAULT
    if KEY_CONFIG_FRAME_SKIP not in st.session_state:
        st.session_state[KEY_CONFIG_FRAME_SKIP] = FRAME_SKIP_DEFAULT
    return {
        "conf_min": st.session_state[KEY_CONFIG_CONF_MIN],
        "imgsz": st.session_state[KEY_CONFIG_IMGSZ],
        "frame_skip": max(1, st.session_state[KEY_CONFIG_FRAME_SKIP]),
    }


def init_session_state() -> None:
    init_state()
    defaults = {
        KEY_PREOP_SMOOTHER: lambda: ToolPresenceSmoother(window_size=SMOOTH_WINDOW_SIZE),
        KEY_PREOP_STABLE_START: lambda: None,
        KEY_ALERTS_LOG: list,
        KEY_LAST_DETECTIONS: list,
        KEY_LAST_COUNTS: dict,
        KEY_FRAME_INDEX: lambda: 0,
        KEY_NAV: lambda: "Setup",
        KEY_VIDEO_SOURCE: lambda: "Live Webcam",
        KEY_UPLOADED_FILE: lambda: None,
        KEY_STREAK_SECONDS: lambda: 0.0,
        KEY_STREAK_BEST: lambda: 0.0,
        KEY_SESSION_START: lambda: None,
        KEY_HAND_STABILITY: list,
        KEY_DEMO_MODE: lambda: False,
        KEY_DEMO_TICK: lambda: 0,
    }
    for key, factory in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = factory()


def norm_tool(name: str) -> str:
    return (name or "").strip().lower().replace(" ", "_")


def counts_for_recipe(counts_norm: dict[str, int], required_tools: list[dict]) -> dict[str, int]:
    return {r["tool"]: counts_norm.get(norm_tool(r["tool"]), 0) for r in required_tools if r.get("tool")}


def open_camera() -> cv2.VideoCapture | None:
    for api in ((cv2.CAP_DSHOW, cv2.CAP_ANY) if hasattr(cv2, "CAP_DSHOW") else (cv2.CAP_ANY,)):
        try:
            cap = cv2.VideoCapture(0, api)
            if cap is not None and cap.isOpened():
                return cap
            if cap is not None:
                cap.release()
        except Exception:
            pass
    return None


def get_video_capture() -> cv2.VideoCapture | None:
    """Return a VideoCapture. All sources are cached in session_state so video
    files advance frame-by-frame across reruns rather than restarting."""
    source = st.session_state.get(KEY_VIDEO_SOURCE, "Live Webcam")

    existing = st.session_state.get(KEY_CAMERA)
    if existing is not None:
        try:
            if existing.isOpened():
                return existing
        except Exception:
            pass
        st.session_state[KEY_CAMERA] = None

    cap = None
    if source == "Live Webcam":
        cap = open_camera()
    elif source == "Upload Video":
        data = st.session_state.get(KEY_UPLOADED_FILE)
        if data is None:
            return None
        tmp_path = st.session_state.get("_upload_tmp_path")
        if tmp_path is None or not Path(tmp_path).exists():
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(data)
            tmp.flush()
            tmp_path = tmp.name
            st.session_state["_upload_tmp_path"] = tmp_path
        cap = cv2.VideoCapture(tmp_path)
    elif source == "Sample Video":
        p = Path(SAMPLE_VIDEO_PATH)
        if not p.exists():
            return None
        cap = cv2.VideoCapture(str(p))

    if cap is not None and cap.isOpened():
        st.session_state[KEY_CAMERA] = cap
        return cap
    if cap is not None:
        cap.release()
    return None


@st.cache_resource
def cached_model():
    return get_model(MODEL_PATH)


def generate_demo_detections(tick: int, required_tools: list[dict], mode: str) -> list[dict]:
    """
    Generate synthetic detections for demo mode.
    In SETUP (PRE_OP): progressively reveal tools over ticks so the checklist fills up.
    In PRACTICE (INTRA_OP): show most tools with occasional random drops to trigger rules.
    """
    h, w = 480, 640
    all_tools = [r["tool"] for r in required_tools]
    if not all_tools:
        return []

    if mode == "PRE_OP":
        # Reveal one more tool each tick (~1.5s) until all are shown
        n_visible = min(len(all_tools), tick + 1)
        visible = all_tools[:n_visible]
    else:
        # Show all tools but randomly drop 0-2 to create occasional alerts
        drop = random.randint(0, min(2, len(all_tools) - 1)) if tick % 5 == 0 else 0
        visible = all_tools if drop == 0 else random.sample(all_tools, max(1, len(all_tools) - drop))

    detections = []
    for i, tool in enumerate(visible):
        row = i // 4
        col = i % 4
        x1 = 30 + col * 150
        y1 = 30 + row * 120
        x2 = x1 + 120
        y2 = y1 + 90
        detections.append({
            "name": tool,
            "conf": round(random.uniform(0.70, 0.98), 2),
            "xyxy": [float(x1), float(y1), float(x2), float(y2)],
        })
    return detections


def render_demo_frame(detections: list[dict]) -> np.ndarray:
    """Render a dark placeholder frame with synthetic bounding boxes for demo mode."""
    frame = np.full((480, 640, 3), (23, 17, 10), dtype=np.uint8)
    cv2.putText(frame, "DEMO MODE", (220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 229, 255), 2)
    cv2.putText(frame, "Simulated detections - no camera needed", (120, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (136, 153, 170), 1)
    return draw_detections(frame, detections)


def display_frame(frame_bgr: np.ndarray) -> None:
    """Embed a BGR frame as a base64 JPEG data-URI, bypassing Streamlit media storage."""
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    st.markdown(
        f'<img src="data:image/jpeg;base64,{b64}" style="width:100%;border-radius:8px;">',
        unsafe_allow_html=True,
    )


# =============================================================================
# Recipe & model
# =============================================================================


def load_recipe_safe() -> dict:
    try:
        return load_recipe(RECIPE_PATH)
    except Exception:
        return {"preop_required": [], "params": {"conf_min": CONF_MIN_DEFAULT, "stable_seconds": 2.0}, "intraop_rules": []}


recipe = load_recipe_safe()
preop_required = recipe.get("preop_required", [])
params = recipe.get("params", {})
stable_seconds = float(params.get("stable_seconds", 2.0))
intraop_rules = recipe.get("intraop_rules", [])

# =============================================================================
# Session state
# =============================================================================

init_session_state()

# =============================================================================
# Sidebar
# =============================================================================

NAV_ITEMS = ["Setup", "Practice", "Report"]
MODE_MAP = {"Setup": "PRE_OP", "Practice": "INTRA_OP", "Report": "POST_OP"}

with st.sidebar:
    st.markdown(
        '<div class="dashboard-header">'
        "<h1>🧪 MedLab Coach</h1>"
        "<p>AI Lab Coach • Skill Assessment</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    mode = get_mode()
    badge_map = {
        "PRE_OP": ("● Setup", "status-idle"),
        "INTRA_OP": ("● Practicing", "status-active"),
        "POST_OP": ("● Report Ready", "status-ready"),
    }
    badge_text, badge_cls = badge_map.get(mode, badge_map["PRE_OP"])
    st.markdown(f'<span class="status-badge {badge_cls}">{badge_text}</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Video Source")
    source = st.radio(
        "Feed",
        ["Live Webcam", "Upload Video", "Sample Video"],
        index=["Live Webcam", "Upload Video", "Sample Video"].index(
            st.session_state.get(KEY_VIDEO_SOURCE, "Live Webcam")
        ),
        label_visibility="collapsed",
        key="source_radio",
    )
    st.session_state[KEY_VIDEO_SOURCE] = source
    if source == "Upload Video":
        uploaded = st.file_uploader("Upload a lab recording", type=["mp4", "avi", "mov"], key="vid_upload")
        if uploaded is not None:
            st.session_state[KEY_UPLOADED_FILE] = uploaded.read()

    st.markdown("---")
    demo = st.toggle("Demo Mode (no camera needed)", value=st.session_state.get(KEY_DEMO_MODE, False), key="demo_toggle")
    st.session_state[KEY_DEMO_MODE] = demo
    if demo:
        st.caption("Synthetic detections — walk through the full flow without tools or camera.")

    st.markdown("---")
    st.markdown("### Navigation")
    nav = st.radio(
        "Section",
        NAV_ITEMS,
        index=NAV_ITEMS.index(st.session_state[KEY_NAV]) if st.session_state[KEY_NAV] in NAV_ITEMS else 0,
        label_visibility="collapsed",
        key="nav_radio",
    )
    st.session_state[KEY_NAV] = nav

    st.markdown("---")
    with st.expander("Settings", expanded=False):
        st.session_state[KEY_CONFIG_CONF_MIN] = st.slider(
            "Confidence", 0.2, 0.9,
            st.session_state.get(KEY_CONFIG_CONF_MIN, CONF_MIN_DEFAULT), 0.05,
        )
        with st.expander("Advanced / Dev", expanded=False):
            imgsz_opts = [320, 416, 640, 832]
            cur = st.session_state.get(KEY_CONFIG_IMGSZ, IMGSZ_DEFAULT)
            idx = imgsz_opts.index(cur) if cur in imgsz_opts else 2
            st.session_state[KEY_CONFIG_IMGSZ] = st.selectbox("Inference size", imgsz_opts, index=idx)
            st.session_state[KEY_CONFIG_FRAME_SKIP] = st.number_input(
                "Frame skip", min_value=1, max_value=10, value=FRAME_SKIP_DEFAULT, step=1,
            )
            cfg = get_config()
            st.caption(f"conf={cfg['conf_min']} imgsz={cfg['imgsz']} skip={cfg['frame_skip']}")

# =============================================================================
# Main header
# =============================================================================

st.markdown(
    '<div class="dashboard-header">'
    "<h1>MEDLAB COACH AI</h1>"
    "<p>AI-Guided Skill Assessment for Medical Training Labs</p>"
    "</div>",
    unsafe_allow_html=True,
)

# =============================================================================
# Video feed — fragment for fast video, autorefresh for checklist sync
# =============================================================================
#
# Design: a single stable @st.fragment(run_every=1s) grabs frames and renders
# them as base64 data-URIs (bypasses Streamlit media storage → no missing-file
# errors). A slower st_autorefresh(4s) triggers full-page reruns so the Setup
# checklist / Practice alerts pick up the latest smoother data written by the
# fragment.

is_demo = st.session_state.get(KEY_DEMO_MODE, False)
run_feed = nav in ("Setup", "Practice") and mode in ("PRE_OP", "INTRA_OP")

# Release camera when leaving feed views or switching to demo
if (not run_feed or is_demo) and KEY_CAMERA in st.session_state and st.session_state[KEY_CAMERA] is not None:
    try:
        st.session_state[KEY_CAMERA].release()
    except Exception:
        pass
    st.session_state[KEY_CAMERA] = None

# Periodic full rerun keeps the checklist in sync with smoother data
if run_feed:
    st_autorefresh(interval=4000, limit=None, key="checklist_sync")


@st.fragment(run_every=timedelta(seconds=1.0))
def video_feed_fragment():
    """High-frequency video feed using base64 images (no media-file storage)."""
    current_nav = st.session_state.get(KEY_NAV, "")
    if current_nav not in ("Setup", "Practice"):
        return
    current_mode = get_mode()
    if current_mode not in ("PRE_OP", "INTRA_OP"):
        return

    demo = st.session_state.get(KEY_DEMO_MODE, False)

    if demo:
        tick = st.session_state.get(KEY_DEMO_TICK, 0)
        st.session_state[KEY_DEMO_TICK] = tick + 1
        detections = generate_demo_detections(tick, preop_required, current_mode)
        counts_norm = {norm_tool(k): v for k, v in count_tools(detections).items()}
        st.session_state[KEY_LAST_DETECTIONS] = detections
        st.session_state[KEY_LAST_COUNTS] = counts_norm
        st.session_state[KEY_PREOP_SMOOTHER].update(
            counts_for_recipe(counts_norm, preop_required), preop_required,
        )
        display_frame(render_demo_frame(detections))
    else:
        cap = get_video_capture()
        if cap is None:
            src = st.session_state.get(KEY_VIDEO_SOURCE, "Live Webcam")
            if src == "Live Webcam":
                st.warning("Camera not available. Connect a webcam and refresh, or enable Demo Mode.")
            elif src == "Upload Video":
                st.info("Upload a video file using the sidebar to begin.")
            else:
                st.info(f"Sample video not found at {SAMPLE_VIDEO_PATH}.")
            return
        try:
            model = cached_model()
        except FileNotFoundError:
            st.error(f"Model not found: {MODEL_PATH}")
            return
        cfg = get_config()
        smoother = st.session_state[KEY_PREOP_SMOOTHER]
        last_frame = None
        for _ in range(max(cfg["frame_skip"], 2)):
            ret, f = cap.read()
            if ret:
                last_frame = f
        if last_frame is not None:
            detections = infer_tools(
                last_frame, conf=cfg["conf_min"],
                imgsz=cfg["imgsz"], model=model,
            )
            counts_norm = {norm_tool(k): v for k, v in count_tools(detections).items()}
            st.session_state[KEY_LAST_DETECTIONS] = detections
            st.session_state[KEY_LAST_COUNTS] = counts_norm
            smoother.update(
                counts_for_recipe(counts_norm, preop_required), preop_required,
            )
            display_frame(draw_detections(last_frame, detections))
        else:
            try:
                cap.release()
            except Exception:
                pass
            st.session_state[KEY_CAMERA] = None
            st.warning("No frames — camera may have disconnected.")


# Always call so the fragment ID stays stable across full reruns
video_feed_fragment()

# =============================================================================
# SETUP tab (was Pre-Op)
# =============================================================================

if nav == "Setup":
    st.subheader("Lab Setup Checklist")
    st.caption("Ensure all required tools are visible to the camera before starting your lab session.")
    smoother = st.session_state[KEY_PREOP_SMOOTHER]
    readiness = smoother.readiness_counts(preop_required) if preop_required else {}
    all_present = smoother.all_present(preop_required)
    now_ts = time.time()

    if all_present:
        if st.session_state[KEY_PREOP_STABLE_START] is None:
            st.session_state[KEY_PREOP_STABLE_START] = now_ts
    else:
        st.session_state[KEY_PREOP_STABLE_START] = None

    stable_start = st.session_state[KEY_PREOP_STABLE_START]
    stable_elapsed = (now_ts - stable_start) if stable_start else 0
    # In demo mode, skip the stable_seconds wait — pass immediately when all tools detected
    effective_stable = 0.0 if st.session_state.get(KEY_DEMO_MODE, False) else stable_seconds
    checklist_pass = all_present and (stable_elapsed >= effective_stable)

    n_required = len(preop_required)
    n_ok = sum(1 for r in preop_required if readiness.get(r["tool"], (0, False))[1])
    readiness_pct = int(100 * n_ok / n_required) if n_required else 100
    st.progress(readiness_pct / 100.0)
    status_text = (
        "All tools detected — ready!" if checklist_pass
        else "Almost there — hold steady..." if all_present
        else "Some tools still missing"
    )
    st.caption(f"Readiness: {n_ok}/{n_required} tools ({readiness_pct}%) — {status_text}")

    if preop_required:
        header = ("Tool", "Required", "Seen (window)", "Status")
        rows = []
        for r in preop_required:
            tool = r["tool"]
            detected, is_ok = readiness.get(tool, (0, False))
            rows.append((str(tool), str(r.get("min_count", 1)), str(detected), "Ready" if is_ok else "Missing"))
        st.table([header] + rows)

    if checklist_pass:
        if st.button("Begin Lab Session", type="primary", width="stretch"):
            set_mode("INTRA_OP")
            st.session_state[KEY_NAV] = "Practice"
            st.session_state[KEY_SESSION_START] = datetime.now().isoformat()
            st.session_state[KEY_STREAK_SECONDS] = 0.0
            st.session_state[KEY_STREAK_BEST] = 0.0
            st.session_state[KEY_ALERTS_LOG] = []
            log_event("STATE_CHANGE", {"from": "PRE_OP", "to": "INTRA_OP"}, mode="INTRA_OP")
            log_event("CHECKLIST_STATUS", {"status": "PASS", "readiness_pct": readiness_pct}, mode="INTRA_OP")
            st.rerun()
    else:
        st.button("Begin Lab Session", disabled=True, width="stretch", help="All tools must be detected and held steady.")

# =============================================================================
# PRACTICE tab (was Intra-Op)
# =============================================================================

if nav == "Practice":
    st.subheader("Practice Session")
    phase = st.selectbox(
        "Current Exercise",
        PHASES,
        index=PHASES.index(get_phase()) if get_phase() in PHASES else 0,
        format_func=lambda p: PHASE_LABELS.get(p, p),
        key="phase_select",
    )
    prev_phase = get_phase()
    set_phase(phase)
    if phase != prev_phase:
        log_event("PHASE_CHANGE", {"phase": phase}, mode="INTRA_OP", phase=phase)

    counts = st.session_state.get(KEY_LAST_COUNTS, {})
    dt_seconds = (FEED_LOOP_FRAMES * 0.033) / max(1, get_config()["frame_skip"])
    alerts = evaluate_rules(phase, counts, intraop_rules, dt_seconds, st.session_state)

    # Streak logic: consecutive seconds without new alerts
    if alerts:
        st.session_state[KEY_STREAK_SECONDS] = 0.0
    else:
        st.session_state[KEY_STREAK_SECONDS] = st.session_state.get(KEY_STREAK_SECONDS, 0) + dt_seconds
    current_streak = st.session_state[KEY_STREAK_SECONDS]
    if current_streak > st.session_state.get(KEY_STREAK_BEST, 0):
        st.session_state[KEY_STREAK_BEST] = current_streak

    for a in alerts:
        st.session_state[KEY_ALERTS_LOG].append({
            "ts": datetime.now().isoformat(),
            "phase": a.get("phase", phase),
            "message": a.get("message", ""),
            "rule_id": a.get("rule_id", ""),
        })
        log_event("ALERT", {"rule_id": a.get("rule_id"), "message": a.get("message")}, mode="INTRA_OP", phase=phase)

    # Streak display
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Current Streak", f"{int(current_streak)}s", help="Consecutive seconds without coaching alerts")
    with col_s2:
        st.metric("Best Streak", f"{int(st.session_state.get(KEY_STREAK_BEST, 0))}s")

    # Alerts panel
    alerts_log = st.session_state.get(KEY_ALERTS_LOG, [])[-10:]
    st.markdown("**Areas for Improvement** (last 10)")
    if alerts_log:
        for e in reversed(alerts_log):
            st.markdown(f"- [{e.get('ts', '')}] **{PHASE_LABELS.get(e.get('phase', ''), e.get('phase', ''))}**: {e.get('message', '')}")
    else:
        st.success("Great work! No coaching notes so far.")

    if st.button("End Lab Session", type="primary", width="stretch"):
        set_mode("POST_OP")
        st.session_state[KEY_NAV] = "Report"
        log_event("STATE_CHANGE", {"from": "INTRA_OP", "to": "POST_OP"}, mode="POST_OP")
        st.rerun()

# =============================================================================
# REPORT tab (was Post-Op)
# =============================================================================

if nav == "Report":
    st.subheader("Lab Session Report")

    rubric = compute_rubric(
        recipe,
        best_streak=st.session_state.get(KEY_STREAK_BEST, 0),
        hand_stability_samples=st.session_state.get(KEY_HAND_STABILITY, []),
    )

    # --- Radar chart (Plotly) ---
    try:
        import plotly.graph_objects as go

        categories = ["Setup\nCompleteness", "Technique\nSafety", "Efficiency", "Streak\nBonus"]
        values = [
            rubric["setup_completeness"],
            rubric["technique_safety"],
            rubric["efficiency"],
            rubric["streak_bonus"],
        ]
        fig = go.Figure(data=go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(0,229,255,0.15)",
            line=dict(color="#00e5ff", width=2),
            marker=dict(size=6),
        ))
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10, color="#8899aa")),
                angularaxis=dict(tickfont=dict(size=11, color="#e0e6ed")),
            ),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=40, l=60, r=60),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.info("Install plotly (`pip install plotly`) for the radar chart visualization.")

    # --- Metric cards ---
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Overall", f"{rubric['total_weighted']}%")
    with c2:
        st.metric("Setup", f"{rubric['setup_completeness']}%")
    with c3:
        st.metric("Safety", f"{rubric['technique_safety']}%")
    with c4:
        st.metric("Efficiency", f"{rubric['efficiency']}%")
    with c5:
        st.metric("Streak", f"{rubric['streak_bonus']}%")

    details = rubric.get("details", {})
    st.caption(
        f"Alerts fired: {details.get('n_alerts', 0)} · "
        f"Duration: {int(details.get('actual_seconds', 0))}s / "
        f"{int(details.get('expected_seconds', 0))}s expected · "
        f"Best streak: {int(details.get('best_streak_seconds', 0))}s"
    )

    # --- Export buttons ---
    st.markdown("---")
    st.markdown("### Export Your Results")
    exp1, exp2 = st.columns(2)
    with exp1:
        st.download_button(
            "Download Coach Report (JSON)",
            data=rubric_to_json(rubric),
            file_name="medlab_coach_report.json",
            mime="application/json",
        )
    with exp2:
        csv_data = events_to_csv()
        st.download_button(
            "Download CSV Transcript",
            data=csv_data if csv_data else "No events recorded yet.",
            file_name="medlab_session_transcript.csv",
            mime="text/csv",
        )

    # --- Timeline ---
    with st.expander("Full Event Timeline", expanded=False):
        events = read_events()
        if events:
            for e in events[-50:]:
                st.caption(
                    f"{e.get('timestamp', '')} | {e.get('type', '')} | "
                    f"mode={e.get('mode', '')} | phase={e.get('phase', '')}"
                )
        else:
            st.caption("No events logged yet.")

# =============================================================================
# Footer
# =============================================================================

st.markdown("---")
st.caption(f"MedLab Coach AI • {datetime.now().strftime('%Y-%m-%d %H:%M')}")
