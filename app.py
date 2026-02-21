"""
Panacea: Surgical Mastery — AI-Guided Training Simulator

================================================================================
HOW THIS SCRIPT WORKS (for studying)
================================================================================

This is the main Streamlit entry point. It runs top to bottom on every user
action (click, slider change, etc.); Streamlit re-runs the whole script and
updates the page.

  FLOW:
  1. Page config & CSS
  2. Define config helpers and session state keys (constants come from src.constants)
  3. Initialize session state (mode, phase, smoother, alerts, last detections, etc.)
  4. Load recipe from recipes/trauma_room.json (required tools, params, intraop_rules)
  5. SIDEBAR: mode badge, config sliders (conf, imgsz, frame_skip), navigation radio
  6. MAIN AREA:
     - Header
     - WEBCAM: only when nav is Pre-Op or Intra-Op and camera works.
       Opens VideoCapture(0), runs a short loop (FEED_LOOP_FRAMES), runs YOLO
       every frame_skip frames, updates last_detections/last_counts and the
       preop smoother, then shows the frame with boxes. Model is loaded once
       via @st.cache_resource.
     - PRE-OP SECTION (if nav == "Pre-Op"): checklist table from smoother,
       readiness %, stable_seconds logic, "Start Surgery" → set mode INTRA_OP,
       log events, rerun.
     - INTRA-OP SECTION (if nav == "Intra-Op"): phase dropdown, evaluate_rules()
       with current counts, append alerts to session_state, show last 10 alerts,
       "End Surgery" → set mode POST_OP, log, rerun.
     - POST-OP SECTION (if nav == "Post-Op"): read_events() from logs/events.jsonl,
       show metrics (total alerts, by phase, checklist pass/fail) and timeline.
  7. Footer

  KEY CONCEPTS:
  - Session state (st.session_state) keeps data between reruns. We init it once
    and use keys from src.constants (KEY_NAV, KEY_LAST_COUNTS, etc.).
  - Mode (PRE_OP / INTRA_OP / POST_OP) is in state; nav (Pre-Op / Intra-Op / Post-Op)
    is which tab the user is viewing. "Start Surgery" and "End Surgery" change
    mode and sync nav.
  - Tool names: we normalize to lowercase + underscores so recipe tools
    (e.g. "needle_holder") match YOLO class names (e.g. "Needle Holder") via
    norm_tool() and counts_for_recipe().

  RUN:
    streamlit run app.py  →  http://localhost:8501
  REQUIRES: models/best.pt, recipes/trauma_room.json. Webcam optional.
"""

import time
from pathlib import Path
from datetime import datetime

import cv2
import streamlit as st

from styles import load_css
from src.constants import (
    MODEL_PATH,
    RECIPE_PATH,
    PHASES,
    CONF_MIN_DEFAULT,
    IMGSZ_DEFAULT,
    FRAME_SKIP_DEFAULT,
    SMOOTH_WINDOW_SIZE,
    FEED_LOOP_FRAMES,
    KEY_NAV,
    KEY_PREOP_SMOOTHER,
    KEY_PREOP_STABLE_START,
    KEY_ALERTS_LOG,
    KEY_LAST_DETECTIONS,
    KEY_LAST_COUNTS,
    KEY_FRAME_INDEX,
    KEY_CONFIG_CONF_MIN,
    KEY_CONFIG_IMGSZ,
    KEY_CONFIG_FRAME_SKIP,
)
from src.state import init_state, get_mode, set_mode, get_phase, set_phase
from src.detector import get_model, infer_tools, count_tools, draw_detections
from src.logger import log_event, read_events
from src.rules import evaluate_rules
from src.utils import load_recipe, ToolPresenceSmoother

# -----------------------------------------------------------------------------
# Page config & CSS
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Panacea: Surgical Mastery",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_css()

# -----------------------------------------------------------------------------
# Config: confidence, inference size, frame skip (sidebar uses these keys)
# -----------------------------------------------------------------------------


def get_config() -> dict:
    """Return current config from session state, with defaults applied."""
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
    """Ensure all session state keys we use exist with safe defaults."""
    init_state()
    if KEY_PREOP_SMOOTHER not in st.session_state:
        st.session_state[KEY_PREOP_SMOOTHER] = ToolPresenceSmoother(
            window_size=SMOOTH_WINDOW_SIZE
        )
    if KEY_PREOP_STABLE_START not in st.session_state:
        st.session_state[KEY_PREOP_STABLE_START] = None
    if KEY_ALERTS_LOG not in st.session_state:
        st.session_state[KEY_ALERTS_LOG] = []
    if KEY_LAST_DETECTIONS not in st.session_state:
        st.session_state[KEY_LAST_DETECTIONS] = []
    if KEY_LAST_COUNTS not in st.session_state:
        st.session_state[KEY_LAST_COUNTS] = {}
    if KEY_FRAME_INDEX not in st.session_state:
        st.session_state[KEY_FRAME_INDEX] = 0
    if KEY_NAV not in st.session_state:
        st.session_state[KEY_NAV] = "Pre-Op"


def norm_tool(name: str) -> str:
    """Normalize tool name for matching: lower case, spaces → underscores."""
    return (name or "").strip().lower().replace(" ", "_")


def counts_for_recipe(
    counts_normalized: dict[str, int], required_tools: list[dict]
) -> dict[str, int]:
    """Map normalized detection counts to recipe tool names for the smoother."""
    return {
        r["tool"]: counts_normalized.get(norm_tool(r["tool"]), 0)
        for r in required_tools
        if r.get("tool")
    }


# -----------------------------------------------------------------------------
# Load recipe (once per run) and cached model
# -----------------------------------------------------------------------------


def load_recipe_safe() -> dict:
    """Load recipe JSON; return empty structure if file missing or invalid."""
    try:
        return load_recipe(RECIPE_PATH)
    except Exception:
        return {
            "preop_required": [],
            "params": {"conf_min": CONF_MIN_DEFAULT, "stable_seconds": 2.0},
            "intraop_rules": [],
        }


@st.cache_resource
def cached_model():
    """Load YOLO model once; cached for the session."""
    return get_model(MODEL_PATH)


recipe = load_recipe_safe()
preop_required = recipe.get("preop_required", [])
params = recipe.get("params", {})
stable_seconds = float(params.get("stable_seconds", 2.0))
intraop_rules = recipe.get("intraop_rules", [])

# -----------------------------------------------------------------------------
# Sidebar: mode badge, config sliders, navigation
# -----------------------------------------------------------------------------

init_session_state()

with st.sidebar:
    st.markdown(
        '<div class="dashboard-header">'
        "<h1>🩺 Panacea</h1>"
        "<p>Pre-Op → Intra-Op → Post-Op</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    mode = get_mode()
    badge_map = {
        "PRE_OP": ("● Pre-Op", "status-idle"),
        "INTRA_OP": ("● Intra-Op", "status-active"),
        "POST_OP": ("● Post-Op", "status-ready"),
    }
    badge_text, badge_cls = badge_map.get(mode, badge_map["PRE_OP"])
    st.markdown(
        f'<span class="status-badge {badge_cls}">{badge_text}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("### Config")
    st.session_state[KEY_CONFIG_CONF_MIN] = st.slider(
        "Confidence min", 0.2, 0.9,
        st.session_state.get(KEY_CONFIG_CONF_MIN, CONF_MIN_DEFAULT), 0.05
    )
    imgsz_opts = [320, 416, 640, 832]
    current_imgsz = st.session_state.get(KEY_CONFIG_IMGSZ, IMGSZ_DEFAULT)
    idx = imgsz_opts.index(current_imgsz) if current_imgsz in imgsz_opts else 2
    st.session_state[KEY_CONFIG_IMGSZ] = st.selectbox(
        "Inference size", imgsz_opts, index=idx
    )
    st.session_state[KEY_CONFIG_FRAME_SKIP] = st.number_input(
        "Frame skip (inference every N frames)",
        min_value=1, max_value=10, value=FRAME_SKIP_DEFAULT, step=1
    )
    cfg = get_config()
    st.caption(f"conf={cfg['conf_min']} imgsz={cfg['imgsz']} skip={cfg['frame_skip']}")
    st.markdown("---")
    st.markdown("### Navigation")
    nav = st.radio(
        "Section",
        ["Pre-Op", "Intra-Op", "Post-Op"],
        index=["Pre-Op", "Intra-Op", "Post-Op"].index(st.session_state[KEY_NAV]),
        label_visibility="collapsed",
        key="nav_radio",
    )
    st.session_state[KEY_NAV] = nav

# -----------------------------------------------------------------------------
# Main header
# -----------------------------------------------------------------------------

st.markdown(
    '<div class="dashboard-header">'
    "<h1>PANACEA: SURGICAL MASTERY</h1>"
    "<p>Pre-Op Checklist • Intra-Op Monitoring • Post-Op Analytics • YOLO Tool Detection</p>"
    "</div>",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Webcam + YOLO inference (only when Pre-Op or Intra-Op and camera available)
# -----------------------------------------------------------------------------

def open_camera() -> cv2.VideoCapture | None:
    """Open default webcam; return None if unavailable."""
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            return cap
        cap.release()
    except Exception:
        pass
    return None


video_placeholder = st.empty()
run_feed = nav in ("Pre-Op", "Intra-Op") and mode in ("PRE_OP", "INTRA_OP")
cap = open_camera() if run_feed else None

if run_feed and cap is not None:
    try:
        model = cached_model()
    except FileNotFoundError:
        model = None
        video_placeholder.error(
            f"Model not found: {MODEL_PATH}. Place weights at models/best.pt"
        )

    if model is not None:
        cfg = get_config()
        frame_index = st.session_state[KEY_FRAME_INDEX]
        smoother = st.session_state[KEY_PREOP_SMOOTHER]

        for _ in range(FEED_LOOP_FRAMES):
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1
            run_inference = (frame_index % cfg["frame_skip"]) == 0

            if run_inference:
                detections = infer_tools(
                    frame,
                    conf=cfg["conf_min"],
                    imgsz=cfg["imgsz"],
                    model=model,
                )
                counts_raw = count_tools(detections)
                counts_norm = {norm_tool(k): v for k, v in counts_raw.items()}
                st.session_state[KEY_LAST_DETECTIONS] = detections
                st.session_state[KEY_LAST_COUNTS] = counts_norm
                counts_for_smoother = counts_for_recipe(counts_norm, preop_required)
                smoother.update(counts_for_smoother, preop_required)
            else:
                detections = st.session_state.get(KEY_LAST_DETECTIONS, [])

            frame = draw_detections(frame, detections)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            time.sleep(0.033)

        st.session_state[KEY_FRAME_INDEX] = frame_index

    if cap is not None:
        cap.release()

elif run_feed:
    video_placeholder.warning(
        "Camera not available. Connect a webcam and refresh, or use another device."
    )

# -----------------------------------------------------------------------------
# Pre-Op: checklist table, readiness %, Start Surgery button
# -----------------------------------------------------------------------------

if nav == "Pre-Op":
    st.subheader("Pre-Op Checklist")
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
    checklist_pass = all_present and (stable_elapsed >= stable_seconds)

    n_required = len(preop_required)
    n_ok = sum(1 for r in preop_required if readiness.get(r["tool"], (0, False))[1])
    readiness_pct = int(100 * n_ok / n_required) if n_required else 100
    st.progress(readiness_pct / 100.0)
    status_text = (
        "PASS (stable)" if checklist_pass
        else "Not yet stable" if all_present
        else "Missing tools"
    )
    st.caption(
        f"Readiness: {n_ok}/{n_required} tools ({readiness_pct}%) — {status_text}"
    )

    if preop_required:
        table_header = ("Tool", "Min", "Detected (window)", "Status")
        table_rows = []
        for r in preop_required:
            tool = r["tool"]
            min_count = r.get("min_count", 1)
            detected, is_ok = readiness.get(tool, (0, False))
            status = "OK" if is_ok else "MISSING"
            table_rows.append((tool, min_count, detected, status))
        st.table([table_header] + table_rows)

    if checklist_pass:
        if st.button("Start Surgery", type="primary", use_container_width=True):
            set_mode("INTRA_OP")
            st.session_state[KEY_NAV] = "Intra-Op"
            log_event("STATE_CHANGE", {"from": "PRE_OP", "to": "INTRA_OP"}, mode="INTRA_OP")
            log_event("CHECKLIST_STATUS", {"status": "PASS", "readiness_pct": readiness_pct}, mode="INTRA_OP")
            st.rerun()
    else:
        st.button(
            "Start Surgery",
            disabled=True,
            use_container_width=True,
            help="Complete checklist and hold stable for required time.",
        )

# -----------------------------------------------------------------------------
# Intra-Op: phase dropdown, rule evaluation, alerts panel, End Surgery button
# -----------------------------------------------------------------------------

if nav == "Intra-Op":
    st.subheader("Intra-Op Monitoring")
    phase = st.selectbox(
        "Phase",
        PHASES,
        index=PHASES.index(get_phase()) if get_phase() in PHASES else 0,
        key="phase_select",
    )
    prev_phase = get_phase()
    set_phase(phase)
    if phase != prev_phase:
        log_event("PHASE_CHANGE", {"phase": phase}, mode="INTRA_OP", phase=phase)

    counts = st.session_state.get(KEY_LAST_COUNTS, {})
    dt_seconds = (FEED_LOOP_FRAMES * 0.033) / max(1, get_config()["frame_skip"])
    alerts = evaluate_rules(phase, counts, intraop_rules, dt_seconds, st.session_state)

    for a in alerts:
        st.session_state[KEY_ALERTS_LOG].append({
            "ts": datetime.now().isoformat(),
            "phase": a.get("phase", phase),
            "message": a.get("message", ""),
            "rule_id": a.get("rule_id", ""),
        })
        log_event(
            "ALERT",
            {"rule_id": a.get("rule_id"), "message": a.get("message")},
            mode="INTRA_OP",
            phase=phase,
        )

    alerts_log = st.session_state.get(KEY_ALERTS_LOG, [])[-10:]
    st.markdown("**Last 10 alerts**")
    if alerts_log:
        for e in reversed(alerts_log):
            st.markdown(f"- [{e.get('ts', '')}] **{e.get('phase', '')}**: {e.get('message', '')}")
    else:
        st.caption("No alerts yet.")

    if st.button("End Surgery", type="primary", use_container_width=True):
        set_mode("POST_OP")
        st.session_state[KEY_NAV] = "Post-Op"
        log_event("STATE_CHANGE", {"from": "INTRA_OP", "to": "POST_OP"}, mode="POST_OP")
        st.rerun()

# -----------------------------------------------------------------------------
# Post-Op: event log summary (alerts, checklist, timeline)
# -----------------------------------------------------------------------------

if nav == "Post-Op":
    st.subheader("Post-Op Analytics")
    events = read_events()
    alerts_ev = [e for e in events if e.get("type") == "ALERT"]
    checklist_ev = [e for e in events if e.get("type") == "CHECKLIST_STATUS"]
    by_phase = {}
    for e in alerts_ev:
        p = e.get("phase", "unknown")
        by_phase[p] = by_phase.get(p, 0) + 1

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total alerts", len(alerts_ev))
    with col2:
        st.metric("Phases with alerts", len(by_phase))
    with col3:
        if not checklist_ev:
            pass_fail = "N/A"
        elif any(e.get("status") == "PASS" for e in checklist_ev):
            pass_fail = "PASS"
        else:
            pass_fail = "FAIL"
        st.metric("Checklist", pass_fail)

    st.markdown("**Alerts by phase**")
    if by_phase:
        st.json(by_phase)
    else:
        st.caption("None")

    st.markdown("**Timeline (last 50 events)**")
    for e in events[-50:]:
        st.caption(
            f"{e.get('timestamp', '')} | {e.get('type', '')} | "
            f"mode={e.get('mode', '')} | phase={e.get('phase', '')}"
        )

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------

st.markdown("---")
st.caption(f"Panacea • {datetime.now().strftime('%Y-%m-%d %H:%M')}")
