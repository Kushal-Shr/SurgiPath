"""
SurgiPath — AI-Guided Skill Assessment for Medical Training Labs

================================================================================
HOW THIS SCRIPT WORKS (for studying)
================================================================================

Streamlit re-runs top to bottom on every interaction. State in st.session_state.

  FLOW:
  1. Page config, CSS
  2. Session state init (mode, smoother, streak, video source, evidence, etc.)
  3. Load recipe
  4. SIDEBAR: branding, video source, demo toggle, nav, settings
  5. MAIN AREA:
     - Header + real-time / stream-time clock
     - VIDEO FEED (hybrid):
         Live Webcam → WebRTC (15-30 FPS, callback thread + queue sync)
         Upload/Sample/Demo → @st.fragment(run_every=1s) + base64 images
       Both paths run YOLO + MediaPipe Hands per frame.
     - SETUP tab: enhanced calibration (coverage, angle, obstruction) + checklist
     - PRACTICE tab: evidence-gated Coach Prompt Cards with hand-context rules,
       user override, streak counter, phase selector
     - REPORT tab: radar chart, trust & reliability section, export
  6. TTS: edge-tts (natural voice, browser) → gTTS (browser) → pyttsx3 (offline)

  RUN:
    streamlit run app.py --server.headless true
"""

import base64
import io
import os
import queue
import time
import random
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from styles import load_css
from src.constants import (
    MODEL_PATH, RECIPE_PATH, PHASES, PHASE_LABELS, SAMPLE_VIDEO_PATH,
    CONF_MIN_DEFAULT, IMGSZ_DEFAULT, FRAME_SKIP_DEFAULT,
    SMOOTH_WINDOW_SIZE, FEED_LOOP_FRAMES, EVIDENCE_WINDOW, MISSING_GRACE_SEC,
    AUTO_TRANSITION_ENABLED, COACH_PROMPT_COOLDOWN_SEC,
    WEBRTC_QUEUE_MAXSIZE, PERF_WINDOW_SIZE,
    KEY_NAV, KEY_PREOP_SMOOTHER, KEY_PREOP_STABLE_START,
    KEY_ALERTS_LOG, KEY_LAST_DETECTIONS, KEY_LAST_COUNTS,
    KEY_FRAME_INDEX, KEY_CONFIG_CONF_MIN, KEY_CONFIG_IMGSZ,
    KEY_CONFIG_FRAME_SKIP, KEY_CAMERA,
    KEY_VIDEO_SOURCE, KEY_UPLOADED_FILE,
    KEY_STREAK_SECONDS, KEY_STREAK_BEST,
    KEY_SESSION_START, KEY_HAND_STABILITY,
    KEY_DEMO_MODE, KEY_DEMO_TICK,
    KEY_EVIDENCE, KEY_COACH_PROMPTS, KEY_OVERRIDES,
    KEY_PROMPT_COUNTER, KEY_LAST_PROMPT_TS, KEY_AUTO_TRANSITION,
    KEY_CALIBRATION_DONE, KEY_STREAM_START_TS,
    KEY_PREV_HANDS, KEY_HELD_TOOLS, KEY_TECHNIQUE, KEY_BIMANUAL_HISTORY,
    KEY_WEBRTC_QUEUE, KEY_WEBRTC_ACTIVE, KEY_WEBRTC_ENABLED,
    KEY_PERF_SAMPLES, KEY_LAST_DISPLAY_TS,
    KEY_TIP_HISTORY, KEY_WRIST_PATH, KEY_JERK_DATA, KEY_ECONOMY_DATA,
)
from src.state import init_state, get_mode, set_mode, get_phase, set_phase
from src.detector import get_model, infer_tools, count_tools, draw_detections
from src.logger import log_event
from src.rules import evaluate_rules
from src.utils import load_recipe, ToolPresenceSmoother
from src.evidence import EvidenceState
from src.hands import (
    detect_hands, compute_hand_jitter, draw_hands, get_held_tools,
    compute_technique_summary,
    get_technique_feedback, IDEAL_GRIPS_BY_PHASE, GRIP_DISPLAY_NAMES,
    compute_jerk_smoothness, compute_motion_economy,
    INDEX_TIP, WRIST,
)

try:
    from brain import (
        generate_final_critique,
        generate_learning_resources,
        generate_dynamic_syllabus,
        SyllabusError,
    )
    BRAIN_AVAILABLE = True
    BRAIN_IMPORT_ERROR = ""
except Exception:
    generate_final_critique = None
    generate_learning_resources = None
    generate_dynamic_syllabus = None
    SyllabusError = None
    BRAIN_AVAILABLE = False
    import traceback
    BRAIN_IMPORT_ERROR = (
        "brain.py import failed: "
        + traceback.format_exc().splitlines()[-1]
    )

# =============================================================================
# Page config & CSS
# =============================================================================

st.set_page_config(
    page_title="SurgiPath",
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
        KEY_EVIDENCE: lambda: EvidenceState(window=EVIDENCE_WINDOW, missing_grace_sec=MISSING_GRACE_SEC),
        KEY_COACH_PROMPTS: list,
        KEY_OVERRIDES: list,
        KEY_PROMPT_COUNTER: lambda: 0,
        KEY_LAST_PROMPT_TS: dict,
        KEY_AUTO_TRANSITION: lambda: AUTO_TRANSITION_ENABLED,
        KEY_CALIBRATION_DONE: lambda: False,
        KEY_STREAM_START_TS: lambda: None,
        KEY_PREV_HANDS: list,
        KEY_HELD_TOOLS: set,
        KEY_TECHNIQUE: dict,
        KEY_BIMANUAL_HISTORY: list,
        KEY_WEBRTC_QUEUE: lambda: queue.Queue(maxsize=WEBRTC_QUEUE_MAXSIZE),
        KEY_WEBRTC_ACTIVE: lambda: False,
        KEY_WEBRTC_ENABLED: lambda: True,
        KEY_PERF_SAMPLES: list,
        KEY_LAST_DISPLAY_TS: lambda: 0.0,
        "_tts_queue": list,
        "_tts_busy_until": lambda: 0.0,
        KEY_TIP_HISTORY: list,
        KEY_WRIST_PATH: list,
        KEY_JERK_DATA: dict,
        KEY_ECONOMY_DATA: dict,
        "_brain_summary": lambda: "",
        "_brain_resources": lambda: "",
        "_gemini_key": lambda: os.getenv("GOOGLE_API_KEY", ""),
        "_ai_reasoning_enabled": lambda: True,
        "_procedure_text": lambda: "",
        "_procedure_name": lambda: "",
        "_procedure_steps": list,
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
        if not data:
            return None
        tmp_path = st.session_state.get("_upload_tmp_path")
        if tmp_path is None or not Path(tmp_path).exists():
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(data)
            tmp.flush()
            tmp.close()
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


def next_prompt_id() -> str:
    c = st.session_state.get(KEY_PROMPT_COUNTER, 0) + 1
    st.session_state[KEY_PROMPT_COUNTER] = c
    return f"P{c:04d}"


def severity_rank(tier: str) -> int:
    """Lower value means higher urgency in UI ordering."""
    return {"high": 0, "medium": 1, "low": 2}.get((tier or "").lower(), 3)


def format_coach_message(message: str, tier: str) -> str:
    """Convert raw rule text into concise, action-first coaching microcopy."""
    msg = (message or "").strip()
    if not msg:
        msg = "Please re-check your current step and tool setup."
    if msg[-1:] not in ".!?":
        msg += "."
    t = (tier or "").lower()
    if t == "high":
        return f"Action needed now: {msg}"
    if t == "medium":
        return f"Please review this step: {msg}"
    return f"Coaching tip: {msg}"


def prompt_sort_key(prompt: dict) -> tuple[int, float]:
    """Sort by severity first, then newest timestamp first."""
    ts_iso = prompt.get("ts", "")
    try:
        ts_epoch = datetime.fromisoformat(ts_iso).timestamp()
    except Exception:
        ts_epoch = 0.0
    return (severity_rank(prompt.get("risk_tier", "high")), -ts_epoch)


def record_perf_sample(sample: dict) -> None:
    """Append one telemetry sample into a rolling in-memory window."""
    samples = st.session_state.setdefault(KEY_PERF_SAMPLES, [])
    samples.append(sample)
    if len(samples) > PERF_WINDOW_SIZE:
        del samples[:-PERF_WINDOW_SIZE]


def compute_perf_summary() -> dict:
    """Compute capture/inference/display FPS and latency percentiles."""
    samples = st.session_state.get(KEY_PERF_SAMPLES, [])
    if not samples:
        return {}

    capture_fps_vals = [s.get("capture_fps", 0.0) for s in samples if s.get("capture_fps", 0.0) > 0]
    infer_fps_vals = [
        (1000.0 / s["infer_ms"]) for s in samples
        if s.get("infer_ms", 0.0) > 0
    ]
    display_fps_vals = [s.get("display_fps", 0.0) for s in samples if s.get("display_fps", 0.0) > 0]
    e2e_latency_vals = [s.get("e2e_ms", 0.0) for s in samples if s.get("e2e_ms", 0.0) > 0]

    def _avg(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    def _pct(vals: list[float], q: float) -> float:
        if not vals:
            return 0.0
        ordered = sorted(vals)
        idx = int(max(0, min(len(ordered) - 1, round(q * (len(ordered) - 1)))))
        return ordered[idx]

    return {
        "capture_fps": _avg(capture_fps_vals),
        "infer_fps": _avg(infer_fps_vals),
        "display_fps": _avg(display_fps_vals),
        "e2e_p50_ms": _pct(e2e_latency_vals, 0.5),
        "e2e_p95_ms": _pct(e2e_latency_vals, 0.95),
        "samples": len(samples),
    }


def build_brain_event_log(prompts: list[dict], overrides: list[dict]) -> list[dict]:
    """Convert session prompts/overrides to the event format expected by brain.py."""
    out: list[dict] = []
    for p in prompts:
        out.append({
            "time": p.get("error_time", ""),
            "type": "coach_prompt",
            "tool": p.get("rule_id", ""),
            "detail": p.get("message", ""),
        })
    for o in overrides:
        out.append({
            "time": (o.get("ts", "")[11:19] if isinstance(o.get("ts", ""), str) and len(o.get("ts", "")) >= 19 else ""),
            "type": "override",
            "tool": o.get("prompt_id", ""),
            "detail": o.get("decision", ""),
        })
    return out


def _phase_from_step_text(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ("incis", "cut", "scalpel")):
        return "incision"
    if any(k in t for k in ("sutur", "needle", "close", "stitch")):
        return "suturing"
    if any(k in t for k in ("irrig", "flush", "wash", "suction")):
        return "irrigation"
    return "closing"


def _procedure_phase_from_elapsed(steps: list[dict], elapsed_s: float) -> tuple[int, str]:
    if not steps:
        return 0, get_phase() if get_phase() in PHASES else PHASES[0]
    cum = 0.0
    idx = len(steps) - 1
    for i, s in enumerate(steps):
        step_t = max(15, int(s.get("time_limit_seconds", 60)))
        cum += step_t
        if elapsed_s <= cum:
            idx = i
            break
    step_text = f"{steps[idx].get('step_name', '')} {steps[idx].get('instruction', '')}"
    return idx, _phase_from_step_text(step_text)


def generate_demo_detections(tick: int, required_tools: list[dict], mode: str) -> list[dict]:
    all_tools = [r["tool"] for r in required_tools]
    if not all_tools:
        return []
    if mode == "PRE_OP":
        n_visible = min(len(all_tools), tick + 1)
        visible = all_tools[:n_visible]
    else:
        drop = random.randint(0, min(2, len(all_tools) - 1)) if tick % 5 == 0 else 0
        visible = all_tools if drop == 0 else random.sample(all_tools, max(1, len(all_tools) - drop))

    detections = []
    for i, tool in enumerate(visible):
        row, col = i // 4, i % 4
        x1, y1 = 30 + col * 150, 30 + row * 120
        detections.append({
            "name": tool,
            "conf": round(random.uniform(0.70, 0.98), 2),
            "xyxy": [float(x1), float(y1), float(x1 + 120), float(y1 + 90)],
        })
    return detections


def render_demo_frame(detections: list[dict]) -> np.ndarray:
    frame = np.full((480, 640, 3), (245, 242, 240), dtype=np.uint8)
    cv2.putText(frame, "DEMO MODE", (220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (196, 115, 26), 2)
    cv2.putText(frame, "Simulated detections - no camera needed", (120, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 110, 100), 1)
    return draw_detections(frame, detections)


def display_frame(frame_bgr: np.ndarray) -> None:
    """Embed a BGR frame as a base64 JPEG, bypassing Streamlit media storage."""
    now_str = datetime.now().strftime("%H:%M:%S")
    stream_start = st.session_state.get(KEY_STREAM_START_TS)
    if stream_start:
        elapsed = int(time.time() - stream_start)
        mins, secs = divmod(elapsed, 60)
        elapsed_str = f"{mins:02d}:{secs:02d}"
    else:
        elapsed_str = "00:00"

    # Overlay clock + stream timer on the frame
    overlay = frame_bgr.copy()
    cv2.putText(overlay, now_str, (frame_bgr.shape[1] - 130, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (196, 115, 26), 1, cv2.LINE_AA)
    cv2.putText(overlay, f"Stream {elapsed_str}", (frame_bgr.shape[1] - 155, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (126, 106, 90), 1, cv2.LINE_AA)

    _, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    st.markdown(
        f'<img src="data:image/jpeg;base64,{b64}" style="width:100%;border-radius:8px;">',
        unsafe_allow_html=True,
    )


def _autoplay_audio_b64(audio_bytes: bytes, mime: str = "audio/mp3") -> None:
    """Embed audio as a base64 data URI with autoplay."""
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    st.markdown(
        f'<audio autoplay><source src="data:{mime};base64,{b64}" type="{mime}"></audio>',
        unsafe_allow_html=True,
    )


def speak_prompt(text: str) -> bool:
    """Generate TTS audio for a coach prompt.
    Cascade: edge-tts (natural voice, browser) → gTTS (browser) → pyttsx3 (server)."""
    if not text or not text.strip():
        return False

    # 1) edge-tts — natural Microsoft Edge voices, free, no API key
    try:
        import asyncio
        import edge_tts

        async def _generate():
            communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
            buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            return buf.getvalue()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                audio_data = pool.submit(lambda: asyncio.run(_generate())).result(timeout=15)
        else:
            audio_data = asyncio.run(_generate())

        if audio_data and len(audio_data) > 100:
            _autoplay_audio_b64(audio_data, "audio/mp3")
            return True
    except Exception:
        pass

    # 2) gTTS — Google TTS, needs internet, decent quality
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang="en", slow=False)
        audio_buf = io.BytesIO()
        tts.write_to_fp(audio_buf)
        audio_bytes = audio_buf.getvalue()
        if audio_bytes and len(audio_bytes) > 100:
            _autoplay_audio_b64(audio_bytes, "audio/mp3")
            return True
    except Exception:
        pass

    # 3) pyttsx3 — offline, server-side (only works when browser == server machine)
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        return True
    except Exception:
        return False
    return False


def _estimate_tts_seconds(text: str) -> float:
    words = max(1, len((text or "").split()))
    # Conservative speech rate + small buffer.
    return max(2.5, words / 2.2 + 0.7)


def queue_tts(text: str) -> None:
    """Queue TTS text to avoid interrupting previous utterances."""
    msg = (text or "").strip()
    if not msg:
        return
    q = st.session_state.setdefault("_tts_queue", [])
    if not q or q[-1] != msg:
        q.append(msg)
        # Keep queue bounded.
        if len(q) > 6:
            del q[:-6]


def flush_tts_queue() -> None:
    """Play one queued utterance only when prior one should be done."""
    busy_until = float(st.session_state.get("_tts_busy_until", 0.0))
    now = time.time()
    if now < busy_until:
        return
    q = st.session_state.setdefault("_tts_queue", [])
    if not q:
        return
    msg = q.pop(0)
    ok = speak_prompt(msg)
    if ok:
        st.session_state["_tts_busy_until"] = now + _estimate_tts_seconds(msg)


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
known_tools = [r["tool"] for r in preop_required]
_alias_map: dict[str, str] = recipe.get("aliases", {})


def alias_tool(name: str) -> str:
    """Normalize a model class name and resolve it through the recipe alias map."""
    n = norm_tool(name)
    return _alias_map.get(n, n)


def remap_detections(detections: list[dict]) -> list[dict]:
    """Remap detection names through the alias map so labels and counts match the recipe."""
    # Ignore placeholder/background classes from external models.
    ignore_labels = {"", "empty", "background", "none", "null"}
    mapped: list[dict] = []
    for d in detections:
        mapped_name = alias_tool(d.get("name", ""))
        if mapped_name in ignore_labels:
            continue
        item = dict(d)
        item["name"] = mapped_name
        mapped.append(item)
    return mapped

# =============================================================================
# Session state
# =============================================================================

init_session_state()

# =============================================================================
# Sidebar
# =============================================================================

NAV_ITEMS = ["Setup", "Practice", "Report"]

with st.sidebar:
    st.markdown(
        '<div class="dashboard-header">'
        "<h1>SurgiPath</h1>"
        "<p>AI Surgical Training &nbsp;·&nbsp; Skill Assessment</p>"
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
    ai_enabled = st.session_state.get("_ai_reasoning_enabled", True)
    eye_label = "👁 AI Reasoning ON" if ai_enabled else "🙈 AI Reasoning OFF"
    if st.button(eye_label, width="stretch", key="ai_eye_toggle_btn"):
        st.session_state["_ai_reasoning_enabled"] = not ai_enabled
        st.rerun()

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
            data = uploaded.getvalue()
            if data:
                st.session_state[KEY_UPLOADED_FILE] = data
                if st.session_state.get("_upload_tmp_path"):
                    old = Path(st.session_state["_upload_tmp_path"])
                    if old.exists():
                        try:
                            old.unlink()
                        except OSError:
                            pass
                    st.session_state["_upload_tmp_path"] = None

    st.markdown("---")
    demo = st.toggle("Demo Mode (no camera needed)", value=st.session_state.get(KEY_DEMO_MODE, False), key="demo_toggle")
    # Keep demo constrained to Setup for predictable judging in Practice.
    if get_mode() != "PRE_OP":
        st.session_state[KEY_DEMO_MODE] = False
    else:
        st.session_state[KEY_DEMO_MODE] = demo
    if st.session_state.get(KEY_DEMO_MODE, False):
        st.caption("Synthetic detections for Setup only. Practice always uses real input.")

    webrtc_on = st.toggle(
        "WebRTC Mode (higher FPS for live webcam)",
        value=st.session_state.get(KEY_WEBRTC_ENABLED, True),
        key="webrtc_toggle",
    )
    st.session_state[KEY_WEBRTC_ENABLED] = webrtc_on
    if webrtc_on:
        st.caption("WebRTC streams video in browser at 15-30 FPS. Click START in the player to begin.")

    if BRAIN_AVAILABLE:
        st.caption("AI reasoning: ready")
    else:
        st.caption("AI reasoning: unavailable (install brain dependencies)")

    auto_trans = st.toggle(
        "Auto-start Practice when ready",
        value=st.session_state.get(KEY_AUTO_TRANSITION, AUTO_TRANSITION_ENABLED),
        key="auto_trans_toggle",
    )
    st.session_state[KEY_AUTO_TRANSITION] = auto_trans

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
            "Min Accuracy", 0.2, 0.9,
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
        with st.expander("AI (optional)", expanded=False):
            key_val = st.text_input(
                "Gemini API key",
                value=st.session_state.get("_gemini_key", ""),
                type="password",
                help="Only stored in memory for this run; not written to files.",
            )
            st.session_state["_gemini_key"] = key_val
            if key_val:
                os.environ["GOOGLE_API_KEY"] = key_val
                st.caption("Gemini connected for AI summary features.")

# =============================================================================
# Main header
# =============================================================================

st.markdown(
    '<div class="dashboard-header">'
    "<h1>SurgiPath</h1>"
    "<p>AI-Guided Skill Assessment &nbsp;·&nbsp; Medical Training Lab System</p>"
    "</div>",
    unsafe_allow_html=True,
)

# =============================================================================
# Video feed — WebRTC for Live Webcam, fragment for upload/sample/demo
# =============================================================================

is_demo = st.session_state.get(KEY_DEMO_MODE, False)
run_feed = nav in ("Setup", "Practice") and mode in ("PRE_OP", "INTRA_OP")
use_webrtc = (
    run_feed
    and not is_demo
    and st.session_state.get(KEY_VIDEO_SOURCE) == "Live Webcam"
    and st.session_state.get(KEY_WEBRTC_ENABLED, True)
)

if (not run_feed or is_demo or use_webrtc) and KEY_CAMERA in st.session_state and st.session_state[KEY_CAMERA] is not None:
    try:
        st.session_state[KEY_CAMERA].release()
    except Exception:
        pass
    st.session_state[KEY_CAMERA] = None

if run_feed:
    if st.session_state.get(KEY_STREAM_START_TS) is None:
        st.session_state[KEY_STREAM_START_TS] = time.time()
    st_autorefresh(interval=2000, limit=None, key="checklist_sync")


def _process_frame_common(
    frame_bgr: np.ndarray,
    detections: list[dict],
    evidence: EvidenceState,
    smoother: ToolPresenceSmoother,
) -> tuple[np.ndarray, list[dict], set[str]]:
    """Shared processing: hand tracking, technique analysis, state updates.
    Returns annotated frame, hand results, and held_tools set."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    hand_results = detect_hands(frame_rgb)

    prev_hands = st.session_state.get(KEY_PREV_HANDS, [])
    jitter = compute_hand_jitter(hand_results, prev_hands)
    st.session_state[KEY_PREV_HANDS] = hand_results
    st.session_state[KEY_HAND_STABILITY].append(jitter)

    held = get_held_tools(hand_results, detections)
    st.session_state[KEY_HELD_TOOLS] = held

    jitter_samples = st.session_state.get(KEY_HAND_STABILITY, [])
    tech = compute_technique_summary(hand_results, jitter_samples, detections)
    st.session_state[KEY_TECHNIQUE] = tech
    bimanual = tech.get("bimanual", {})
    if bimanual.get("detected"):
        bh = st.session_state.get(KEY_BIMANUAL_HISTORY, [])
        bh.append(bimanual["inter_hand_dist"])
        if len(bh) > 60:
            bh = bh[-60:]
        st.session_state[KEY_BIMANUAL_HISTORY] = bh

    # ── B: Jerk Smoothness / C: Economy of Motion ──────────────────────────
    if hand_results:
        first = hand_results[0]
        lms = first.get("landmarks", [])
        if len(lms) > INDEX_TIP:
            tip_h = st.session_state.get(KEY_TIP_HISTORY, [])
            tip_h.append((lms[INDEX_TIP][0], lms[INDEX_TIP][1]))
            if len(tip_h) > 50:
                tip_h = tip_h[-50:]
            st.session_state[KEY_TIP_HISTORY] = tip_h
            st.session_state[KEY_JERK_DATA] = compute_jerk_smoothness(tip_h)
        if len(lms) > WRIST:
            wrist_h = st.session_state.get(KEY_WRIST_PATH, [])
            wrist_h.append((lms[WRIST][0], lms[WRIST][1]))
            if len(wrist_h) > 300:
                wrist_h = wrist_h[-300:]
            st.session_state[KEY_WRIST_PATH] = wrist_h
            st.session_state[KEY_ECONOMY_DATA] = compute_motion_economy(wrist_h)

    counts_norm = {norm_tool(k): v for k, v in count_tools(detections).items()}
    st.session_state[KEY_LAST_DETECTIONS] = detections
    st.session_state[KEY_LAST_COUNTS] = counts_norm
    smoother.update(counts_for_recipe(counts_norm, preop_required), preop_required)
    evidence.update(detections, frame_bgr=frame_bgr, known_tools=known_tools)

    annotated = draw_detections(frame_bgr, detections)
    annotated = draw_hands(annotated, hand_results)
    return annotated, hand_results, held


# --------------- WebRTC callback (runs in a separate thread) ----------------

def _make_webrtc_callback(result_q: queue.Queue, model, cfg: dict):
    """Return a video_frame_callback closure for webrtc_streamer."""
    def callback(frame: av.VideoFrame) -> av.VideoFrame:
        capture_ts = time.time()
        prev_capture_ts = getattr(callback, "_prev_capture_ts", 0.0)
        if prev_capture_ts > 0:
            capture_dt = max(capture_ts - prev_capture_ts, 1e-6)
            capture_fps = 1.0 / capture_dt
        else:
            capture_fps = 0.0
        callback._prev_capture_ts = capture_ts

        img = frame.to_ndarray(format="bgr24")
        t0 = time.perf_counter()
        detections = remap_detections(infer_tools(img, conf=cfg["conf_min"], imgsz=cfg["imgsz"], model=model))
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hand_results = detect_hands(frame_rgb)
        held = get_held_tools(hand_results, detections)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        try:
            result_q.put_nowait({
                "detections": detections,
                "hands": hand_results,
                "held_tools": held,
                "frame_bgr": img,
                "capture_ts": capture_ts,
                "infer_ms": infer_ms,
                "capture_fps": capture_fps,
            })
        except queue.Full:
            # Drop oldest sample to keep latency low, then enqueue newest frame.
            try:
                _ = result_q.get_nowait()
            except queue.Empty:
                pass
            try:
                result_q.put_nowait({
                    "detections": detections,
                    "hands": hand_results,
                    "held_tools": held,
                    "frame_bgr": img,
                    "capture_ts": capture_ts,
                    "infer_ms": infer_ms,
                    "capture_fps": capture_fps,
                })
            except queue.Full:
                pass

        annotated = draw_detections(img, detections)
        annotated = draw_hands(annotated, hand_results)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")
    return callback


def _drain_webrtc_queue(result_q: queue.Queue):
    """Drain the WebRTC result queue and update session state from the latest result."""
    latest = None
    while True:
        try:
            latest = result_q.get_nowait()
        except queue.Empty:
            break
    if latest is None:
        return
    evidence: EvidenceState = st.session_state[KEY_EVIDENCE]
    smoother = st.session_state[KEY_PREOP_SMOOTHER]

    detections = latest["detections"]
    hand_results = latest["hands"]
    held = latest["held_tools"]

    prev_hands = st.session_state.get(KEY_PREV_HANDS, [])
    jitter = compute_hand_jitter(hand_results, prev_hands)
    st.session_state[KEY_PREV_HANDS] = hand_results
    st.session_state[KEY_HAND_STABILITY].append(jitter)
    st.session_state[KEY_HELD_TOOLS] = held

    jitter_samples = st.session_state.get(KEY_HAND_STABILITY, [])
    tech = compute_technique_summary(hand_results, jitter_samples, detections)
    st.session_state[KEY_TECHNIQUE] = tech
    bimanual = tech.get("bimanual", {})
    if bimanual.get("detected"):
        bh = st.session_state.get(KEY_BIMANUAL_HISTORY, [])
        bh.append(bimanual["inter_hand_dist"])
        if len(bh) > 60:
            bh = bh[-60:]
        st.session_state[KEY_BIMANUAL_HISTORY] = bh

    counts_norm = {norm_tool(k): v for k, v in count_tools(detections).items()}
    st.session_state[KEY_LAST_DETECTIONS] = detections
    st.session_state[KEY_LAST_COUNTS] = counts_norm
    smoother.update(counts_for_recipe(counts_norm, preop_required), preop_required)
    evidence.update(detections, frame_bgr=latest.get("frame_bgr"), known_tools=known_tools)

    # End-to-end telemetry: capture/infer/display FPS + latency.
    now_ts = time.time()
    prev_display_ts = float(st.session_state.get(KEY_LAST_DISPLAY_TS, 0.0))
    display_fps = (1.0 / max(now_ts - prev_display_ts, 1e-6)) if prev_display_ts > 0 else 0.0
    st.session_state[KEY_LAST_DISPLAY_TS] = now_ts

    capture_ts = float(latest.get("capture_ts", 0.0))
    infer_ms = float(latest.get("infer_ms", 0.0))
    e2e_ms = (now_ts - capture_ts) * 1000.0 if capture_ts > 0 else 0.0
    record_perf_sample({
        "capture_fps": float(latest.get("capture_fps", 0.0)),
        "infer_ms": infer_ms,
        "display_fps": display_fps,
        "e2e_ms": e2e_ms,
        "ts": now_ts,
    })


# --------------- Render video section ----------------------------------------

RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
}

MEDIA_CONSTRAINTS = {
    "video": {
        "width": {"ideal": 640},
        "height": {"ideal": 480},
        "frameRate": {"ideal": 15, "max": 30},
    },
    "audio": False,
}

if use_webrtc:
    try:
        model = cached_model()
    except FileNotFoundError:
        st.error(f"Model not found: {MODEL_PATH}")
        model = None

    if model is not None:
        cfg = get_config()
        result_q: queue.Queue = st.session_state[KEY_WEBRTC_QUEUE]
        ctx = webrtc_streamer(
            key="surgipath-live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_frame_callback=_make_webrtc_callback(result_q, model, cfg),
            media_stream_constraints=MEDIA_CONSTRAINTS,
            async_processing=True,
        )
        st.session_state[KEY_WEBRTC_ACTIVE] = ctx.state.playing if ctx else False
        _drain_webrtc_queue(result_q)

elif run_feed:
    @st.fragment(run_every=timedelta(seconds=1.0))
    def video_feed_fragment():
        current_nav = st.session_state.get(KEY_NAV, "")
        if current_nav not in ("Setup", "Practice"):
            return
        current_mode = get_mode()
        if current_mode not in ("PRE_OP", "INTRA_OP"):
            return

        evidence: EvidenceState = st.session_state[KEY_EVIDENCE]
        smoother = st.session_state[KEY_PREOP_SMOOTHER]
        demo_on = st.session_state.get(KEY_DEMO_MODE, False)

        if demo_on:
            tick = st.session_state.get(KEY_DEMO_TICK, 0)
            st.session_state[KEY_DEMO_TICK] = tick + 1
            detections = generate_demo_detections(tick, preop_required, current_mode)
            counts_norm = {norm_tool(k): v for k, v in count_tools(detections).items()}
            st.session_state[KEY_LAST_DETECTIONS] = detections
            st.session_state[KEY_LAST_COUNTS] = counts_norm
            smoother.update(
                counts_for_recipe(counts_norm, preop_required), preop_required,
            )
            evidence.update(detections, frame_bgr=None, known_tools=known_tools)
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
            last_frame = None
            for _ in range(max(cfg["frame_skip"], 2)):
                ret, f = cap.read()
                if ret:
                    last_frame = f
            if last_frame is not None:
                detections = remap_detections(infer_tools(
                    last_frame, conf=cfg["conf_min"],
                    imgsz=cfg["imgsz"], model=model,
                ))
                annotated, _, _ = _process_frame_common(
                    last_frame, detections, evidence, smoother,
                )
                display_frame(annotated)
            else:
                try:
                    cap.release()
                except Exception:
                    pass
                st.session_state[KEY_CAMERA] = None
                st.warning("No frames — camera may have disconnected.")

    video_feed_fragment()

# =============================================================================
# SETUP tab — calibration check + checklist + auto-transition
# =============================================================================

if nav == "Setup":
    st.markdown(
        '<div class="section-header">'
        '<span class="section-num">01 / SETUP</span>'
        '<span class="section-title">Lab Setup Checklist</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.get("_ai_reasoning_enabled", True):
        st.markdown("### Upload Your Procedure")
        mode_opt = st.radio(
            "Procedure mode",
            ["AI generate procedure", "Manual note (no AI)"],
            horizontal=True,
            key="procedure_mode_radio",
        )

        if mode_opt == "AI generate procedure":
            proc_text = st.text_input(
                "Procedure text (optional)",
                value=st.session_state.get("_procedure_text", ""),
                placeholder="Type a procedure name, or leave empty to let AI infer from session.",
            )
            st.session_state["_procedure_text"] = proc_text
            if not BRAIN_AVAILABLE or generate_dynamic_syllabus is None:
                st.warning("AI reasoning unavailable. Install brain dependencies first.")
            elif st.button("AI Generate Procedure", width="stretch"):
                if proc_text.strip():
                    prompt = proc_text.strip()
                    proc_name = proc_text.strip()
                else:
                    dets = st.session_state.get(KEY_LAST_DETECTIONS, [])
                    tools = sorted({d.get("name", "") for d in dets if d.get("name", "")})
                    prompt = (
                        "Infer a likely surgical or medical lab procedure based on observed tools: "
                        + ", ".join(tools)
                    ) if tools else "Infer a likely basic surgical training procedure from current session context."
                    proc_name = "AI inferred procedure"
                with st.spinner("Generating procedure steps..."):
                    res = generate_dynamic_syllabus(prompt)
                if SyllabusError is not None and isinstance(res, SyllabusError):
                    st.warning(res.error)
                else:
                    steps = [s.model_dump() for s in res.steps] if hasattr(res, "steps") else []
                    st.session_state["_procedure_name"] = proc_name
                    st.session_state["_procedure_steps"] = steps
        else:
            note = st.text_area(
                "Procedure note",
                value=st.session_state.get("_procedure_text", ""),
                placeholder="Write step notes, one line per step...",
                height=120,
                key="manual_procedure_note",
            )
            st.session_state["_procedure_text"] = note
            if st.button("Use Manual Note", width="stretch"):
                lines = [ln.strip(" -•\t") for ln in (note or "").splitlines() if ln.strip()]
                steps = [
                    {"step_name": f"Step {i+1}", "instruction": ln, "time_limit_seconds": 60}
                    for i, ln in enumerate(lines[:12])
                ]
                st.session_state["_procedure_name"] = "Manual procedure note"
                st.session_state["_procedure_steps"] = steps

        proc_steps = st.session_state.get("_procedure_steps", [])
        if proc_steps:
            st.caption(f"Procedure plan: {st.session_state.get('_procedure_name', 'Session plan')}")
            st.table(
                [("Step", "Instruction", "Time (s)")] + [
                    (str(i + 1), s.get("instruction", ""), str(s.get("time_limit_seconds", 60)))
                    for i, s in enumerate(proc_steps[:8])
                ]
            )

    # --- Calibration panel ---
    evidence: EvidenceState = st.session_state[KEY_EVIDENCE]
    cal = evidence.calibration_status()
    with st.expander("Camera Calibration", expanded=not cal["ok"]):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Brightness", f"{cal['brightness']:.0f}")
        with c2:
            st.metric("Sharpness", f"{cal['blur_score']:.0f}")
        with c3:
            st.metric("Detection Rate", f"{cal['detection_rate']:.0%}")
        c4, c5 = st.columns(2)
        with c4:
            st.metric("Workspace Coverage", f"{cal.get('workspace_coverage', 0):.0%}")
        with c5:
            st.metric("Centroid Spread", f"{cal.get('centroid_spread', 0):.2f}")
        if cal.get("obstruction_warning"):
            st.error("Possible obstruction detected — check that nothing is blocking the camera.")
        if cal["ok"]:
            st.success("Calibration OK — camera conditions are good.")
            st.session_state[KEY_CALIBRATION_DONE] = True
        else:
            for issue in cal["issues"]:
                st.warning(issue)
            if is_demo:
                st.session_state[KEY_CALIBRATION_DONE] = True

    st.caption("Ensure all required tools are visible to the camera before starting.")
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
        header = ("Tool", "Required", "Seen (window)", "Accuracy", "Status")
        rows = []
        for r in preop_required:
            tool = r["tool"]
            detected, is_ok = readiness.get(tool, (0, False))
            ts = evidence.tool_state(tool)
            conf_str = f"{ts['avg_conf']:.0%}" if ts["avg_conf"] > 0 else "—"
            rows.append((
                str(tool), str(r.get("min_count", 1)), str(detected),
                conf_str, "Ready" if is_ok else "Missing",
            ))
        st.table([header] + rows)

    def _begin_lab_session():
        """Transition to Practice — used by both button and auto-transition."""
        if get_mode() != "PRE_OP":
            return
        set_mode("INTRA_OP")
        st.session_state[KEY_NAV] = "Practice"
        st.session_state[KEY_SESSION_START] = datetime.now().isoformat()
        st.session_state[KEY_STREAK_SECONDS] = 0.0
        st.session_state[KEY_STREAK_BEST] = 0.0
        st.session_state[KEY_ALERTS_LOG] = []
        st.session_state[KEY_COACH_PROMPTS] = []
        st.session_state[KEY_OVERRIDES] = []
        st.session_state[KEY_PROMPT_COUNTER] = 0
        st.session_state[KEY_LAST_PROMPT_TS] = {}
        st.session_state[KEY_PERF_SAMPLES] = []
        st.session_state[KEY_LAST_DISPLAY_TS] = 0.0
        st.session_state["_tts_queue"] = []
        st.session_state["_tts_busy_until"] = 0.0
        st.session_state["_brain_summary"] = ""
        st.session_state["_brain_resources"] = ""
        # Demo is only for setup walkthrough; switch to real camera in Practice.
        if st.session_state.get(KEY_DEMO_MODE, False):
            st.session_state[KEY_DEMO_MODE] = False
            st.session_state[KEY_VIDEO_SOURCE] = "Live Webcam"
            # For seamless transition after demo, default to OpenCV camera path.
            st.session_state[KEY_WEBRTC_ENABLED] = False
            st.session_state["source_radio"] = "Live Webcam"
            st.session_state["webrtc_toggle"] = False
            st.session_state["demo_toggle"] = False
        log_event("STATE_CHANGE", {"from": "PRE_OP", "to": "INTRA_OP"}, mode="INTRA_OP")
        log_event("CHECKLIST_STATUS", {"status": "PASS", "readiness_pct": readiness_pct}, mode="INTRA_OP")

    if checklist_pass:
        # Auto-transition: if toggle is on, go straight to Practice
        if st.session_state.get(KEY_AUTO_TRANSITION, False):
            _begin_lab_session()
            st.rerun()
        elif st.button("Begin Lab Session", type="primary", width="stretch"):
            _begin_lab_session()
            st.rerun()
    else:
        st.button("Begin Lab Session", disabled=True, width="stretch",
                   help="All tools must be detected and held steady.")

# =============================================================================
# PRACTICE tab — Coach Prompt Cards + user override + streak
# =============================================================================

if nav == "Practice":
    st.markdown(
        '<div class="section-header">'
        '<span class="section-num">02 / PRACTICE</span>'
        '<span class="section-title">Practice Session</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    proc_steps = st.session_state.get("_procedure_steps", [])
    phase = get_phase() if get_phase() in PHASES else PHASES[0]
    if st.session_state.get("_ai_reasoning_enabled", True) and proc_steps:
        sess_start = st.session_state.get(KEY_SESSION_START)
        if sess_start:
            try:
                elapsed_s = max(0.0, (datetime.now() - datetime.fromisoformat(sess_start)).total_seconds())
            except Exception:
                elapsed_s = 0.0
        else:
            elapsed_s = 0.0
        step_idx, fixed_phase = _procedure_phase_from_elapsed(proc_steps, elapsed_s)
        phase = fixed_phase
        prev_phase = get_phase()
        set_phase(phase)
        if phase != prev_phase:
            log_event("PHASE_CHANGE", {"phase": phase}, mode="INTRA_OP", phase=phase)
        st.caption(
            f"Operation (fixed by procedure): Step {step_idx + 1}/{len(proc_steps)} · "
            f"{proc_steps[step_idx].get('step_name', 'Procedure step')} · "
            f"{PHASE_LABELS.get(phase, phase)}"
        )
    else:
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

    perf = compute_perf_summary()
    show_perf = use_webrtc or st.session_state.get(KEY_WEBRTC_ACTIVE, False)
    if perf and show_perf:
        pf1, pf2, pf3, pf4, pf5 = st.columns(5)
        with pf1:
            st.metric("Capture FPS", f"{perf['capture_fps']:.1f}")
        with pf2:
            st.metric("Inference FPS", f"{perf['infer_fps']:.1f}")
        with pf3:
            st.metric("Display FPS", f"{perf['display_fps']:.1f}")
        with pf4:
            st.metric("E2E p50", f"{perf['e2e_p50_ms']:.0f} ms")
        with pf5:
            st.metric("E2E p95", f"{perf['e2e_p95_ms']:.0f} ms")
        st.caption(f"WebRTC telemetry window: {perf['samples']} recent frames (drop-oldest queue active).")
    elif show_perf:
        st.caption("Collecting WebRTC telemetry...")

    if (
        st.session_state.get(KEY_VIDEO_SOURCE) == "Live Webcam"
        and st.session_state.get(KEY_WEBRTC_ENABLED, True)
        and not st.session_state.get(KEY_WEBRTC_ACTIVE, False)
    ):
        st.info("WebRTC is enabled. Click START in the video widget to begin camera stream.")
        if st.button("Use camera fallback (no START needed)", key="webrtc_fallback_btn"):
            st.session_state[KEY_WEBRTC_ENABLED] = False
            st.rerun()

    counts = st.session_state.get(KEY_LAST_COUNTS, {})
    evidence_obj: EvidenceState = st.session_state[KEY_EVIDENCE]
    held_tools = st.session_state.get(KEY_HELD_TOOLS, set())
    dt_seconds = (FEED_LOOP_FRAMES * 0.033) / max(1, get_config()["frame_skip"])
    tool_signal = sum(int(v) for v in counts.values()) > 0
    if tool_signal:
        alerts = evaluate_rules(
            phase, counts, intraop_rules, dt_seconds, st.session_state,
            evidence=evidence_obj, held_tools=held_tools,
        )
    else:
        alerts = []
        st.caption("Tool signal is low — coaching is currently based on hand technique only.")

    # --- Technique-based alerts (grip + smoothness) ---
    tech_summary = st.session_state.get(KEY_TECHNIQUE, {})
    tech_timers = st.session_state.setdefault("_tech_timers", {})
    tech_triggered = st.session_state.setdefault("_tech_triggered", set())

    _TECH_RULES = [
        {
            "id": "tech_tremor",
            "phases": ("suturing", "closing"),
            "check": lambda t, _p: t.get("smoothness") == "tremor",
            "hold": 3.0,
            "message": "Hand tremor detected — try resting your wrists on the table for stability.",
        },
        {
            "id": "tech_palmar_scalpel",
            "phases": ("incision",),
            "check": lambda t, _p: any(
                g.get("grip_type") == "palmar_grip" and "scalpel" in g.get("near_tools", [])
                for g in t.get("grips", [])
            ),
            "hold": 2.5,
            "message": "Palmar grip on scalpel — consider a pencil grip for finer incision control.",
        },
        {
            "id": "tech_open_needle",
            "phases": ("suturing",),
            "check": lambda t, _p: any(
                g.get("grip_type") == "open_hand" and "needle_holder" in g.get("near_tools", [])
                for g in t.get("grips", [])
            ),
            "hold": 2.5,
            "message": "Open hand near needle holder — use a palmar or precision grip for secure control.",
        },
        {
            "id": "tech_steep_angle",
            "phases": ("incision",),
            "check": lambda t, _p: any(
                g.get("instrument_angle", 0) > 70 and "scalpel" in g.get("near_tools", [])
                for g in t.get("grips", [])
            ),
            "hold": 3.0,
            "message": "Steep scalpel angle (>70°) — aim for 30-45° for a controlled incision.",
        },
    ]

    for tr in _TECH_RULES:
        if phase not in tr["phases"]:
            tech_timers[tr["id"]] = 0
            tech_triggered.discard(tr["id"])
            continue
        if tr["check"](tech_summary, phase):
            tech_timers[tr["id"]] = tech_timers.get(tr["id"], 0) + dt_seconds
            if tr["id"] not in tech_triggered and tech_timers[tr["id"]] >= tr["hold"]:
                alerts.append({
                    "rule_id": tr["id"],
                    "message": tr["message"],
                    "phase": phase,
                    "risk_tier": "medium",
                    "avg_conf": 0.0,
                    "seen_ratio": 0.0,
                    "last_seen_ts": time.time(),
                    "evidence_tools": {},
                })
                tech_triggered.add(tr["id"])
        else:
            tech_timers[tr["id"]] = 0
            tech_triggered.discard(tr["id"])

    # Streak logic
    if alerts:
        st.session_state[KEY_STREAK_SECONDS] = 0.0
    else:
        st.session_state[KEY_STREAK_SECONDS] = st.session_state.get(KEY_STREAK_SECONDS, 0) + dt_seconds
    current_streak = st.session_state[KEY_STREAK_SECONDS]
    if current_streak > st.session_state.get(KEY_STREAK_BEST, 0):
        st.session_state[KEY_STREAK_BEST] = current_streak

    # Register new alerts as Coach Prompts + TTS
    new_prompt_messages: list[str] = []
    last_prompt_ts = st.session_state.setdefault(KEY_LAST_PROMPT_TS, {})
    now_epoch = time.time()
    for a in alerts:
        rule_id = a.get("rule_id", "") or "unknown_rule"
        previous = float(last_prompt_ts.get(rule_id, 0.0))
        if (now_epoch - previous) < COACH_PROMPT_COOLDOWN_SEC:
            continue
        last_prompt_ts[rule_id] = now_epoch

        tier = a.get("risk_tier", "high")
        coach_message = format_coach_message(a.get("message", ""), tier)
        pid = next_prompt_id()
        error_time = datetime.now().strftime("%H:%M:%S")
        prompt = {
            "prompt_id": pid,
            "ts": datetime.now().isoformat(),
            "error_time": error_time,
            "phase": a.get("phase", phase),
            "rule_id": rule_id,
            "message": coach_message,
            "risk_tier": tier,
            "avg_conf": a.get("avg_conf", 1.0),
            "seen_ratio": a.get("seen_ratio", 1.0),
            "last_seen_ts": a.get("last_seen_ts", 0),
        }
        st.session_state[KEY_COACH_PROMPTS].append(prompt)
        st.session_state[KEY_ALERTS_LOG].append(prompt)
        log_event("COACH_PROMPT", {**prompt}, mode="INTRA_OP", phase=phase)
        if len(new_prompt_messages) < 2:
            new_prompt_messages.append(coach_message)

    # Streak display
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Current Streak", f"{int(current_streak)}s", help="Consecutive seconds without coaching alerts")
    with col_s2:
        st.metric("Best Streak", f"{int(st.session_state.get(KEY_STREAK_BEST, 0))}s")

    # --- Live Technique Monitor ---
    tech = st.session_state.get(KEY_TECHNIQUE, {})
    grips = tech.get("grips", [])
    smoothness = tech.get("smoothness", "steady")
    bimanual = tech.get("bimanual", {})

    # Demo mode: synthesize realistic hand data so monitor is always visible
    is_demo = st.session_state.get(KEY_DEMO_MODE, False)
    _demo_grips = grips
    _demo_smoothness = smoothness
    if is_demo and not grips:
        demo_tick = st.session_state.get(KEY_DEMO_TICK, 0)
        _grip_cycle = ["pencil_grip", "pencil_grip", "pencil_grip", "palmar_grip", "precision_grip"]
        _smooth_cycle = ["steady", "steady", "moderate"]
        _demo_grips = [{
            "grip_type": _grip_cycle[demo_tick % len(_grip_cycle)],
            "instrument_angle": 32 + (demo_tick % 7) * 3,
            "handedness": "Right",
            "near_tools": [],
        }]
        _demo_smoothness = _smooth_cycle[demo_tick % len(_smooth_cycle)]
        # Simulate jerk (B) and economy (C) for demo
        _jerk_cycle = ["fluid", "fluid", "moderate", "fluid", "jerky"]
        _econ_cycle = ["efficient", "efficient", "moderate", "efficient"]
        st.session_state[KEY_JERK_DATA] = {
            "smoothness_score": [0.85, 0.80, 0.55, 0.82, 0.30][demo_tick % 5],
            "mean_jerk": [1.2, 1.5, 4.0, 1.3, 8.5][demo_tick % 5],
            "label": _jerk_cycle[demo_tick % len(_jerk_cycle)],
        }
        st.session_state[KEY_ECONOMY_DATA] = {
            "path_length_px": 850 + demo_tick * 12,
            "directness": [0.78, 0.72, 0.65, 0.81][demo_tick % 4],
            "idle_ratio": [0.20, 0.25, 0.38, 0.18][demo_tick % 4],
            "label": _econ_cycle[demo_tick % len(_econ_cycle)],
        }

    if _demo_grips:
        g = _demo_grips[0]
        fb = get_technique_feedback(
            g["grip_type"], phase, _demo_smoothness, g["instrument_angle"]
        )
        _sc = {"good": "#00E5A0", "warning": "#FFB703", "error": "#FF3D3D"}
        _si = {"good": "✓", "warning": "⚠", "error": "✗"}
        _sl = {"good": "GOOD", "warning": "REVIEW", "error": "CORRECT"}
        sc  = _sc.get(fb["status"], "#6A8CA8")
        si  = _si.get(fb["status"], "—")
        sl  = _sl.get(fb["status"], "—")

        smc = {"steady": "#00E5A0", "moderate": "#FFB703", "tremor": "#FF3D3D"}.get(_demo_smoothness, "#6A8CA8")
        sm_icon = {"steady": "✓", "moderate": "⚠", "tremor": "✗"}.get(_demo_smoothness, "—")

        ang = fb["angle"]
        if ang > 70:
            anc, an_label = "#FF3D3D", "Steep"
        elif 25 <= ang <= 60:
            anc, an_label = "#00E5A0", "Ideal"
        else:
            anc, an_label = "#FFB703", "Low"

        ideal_names = [GRIP_DISPLAY_NAMES.get(g2, g2) for g2 in IDEAL_GRIPS_BY_PHASE.get(phase, [])]
        ideal_str = " / ".join(ideal_names) if ideal_names else "Any"

        hands_count = bimanual.get("hands_count", len(_demo_grips))
        bm_color = "#00E5A0" if hands_count >= 2 else "#5E7D9A"

        # ── B: Jerk Smoothness ──
        jerk_data = st.session_state.get(KEY_JERK_DATA, {})
        jerk_label = jerk_data.get("label", "fluid")
        jerk_score = jerk_data.get("smoothness_score", 1.0)
        jkc = {"fluid": "#00E5A0", "moderate": "#FFB703", "jerky": "#FF3D3D"}.get(jerk_label, "#6A8CA8")
        jki = {"fluid": "✓", "moderate": "~", "jerky": "✗"}.get(jerk_label, "—")
        jk_pct = f"{jerk_score * 100:.0f}%"

        # ── C: Economy of Motion ──
        econ_data = st.session_state.get(KEY_ECONOMY_DATA, {})
        econ_label = econ_data.get("label", "efficient")
        econ_direct = econ_data.get("directness", 1.0)
        ecc = {"efficient": "#00E5A0", "moderate": "#FFB703", "excessive": "#FF3D3D"}.get(econ_label, "#6A8CA8")
        eci = {"efficient": "✓", "moderate": "~", "excessive": "✗"}.get(econ_label, "—")
        ec_pct = f"{econ_direct * 100:.0f}%"

        st.markdown(f"""
<div style="background:var(--bg-card);border:1px solid {sc}44;border-left:3px solid {sc};
            border-radius:8px;padding:1rem 1.25rem 0.9rem;margin-bottom:1rem;
            box-shadow:0 0 16px rgba(0,0,0,0.4);">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">
    <span style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;color:var(--text-3);
                 text-transform:uppercase;letter-spacing:0.12em;">Live Technique Monitor</span>
    <span style="font-family:'IBM Plex Mono',monospace;font-size:0.60rem;font-weight:700;
                 color:{sc};background:{sc}18;border:1px solid {sc}44;border-radius:3px;
                 padding:0.15rem 0.5rem;letter-spacing:0.08em;">{sl}</span>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.6rem;margin-bottom:0.8rem;">
    <div style="background:var(--bg-elevated);border-radius:6px;padding:0.65rem 0.5rem;text-align:center;border:1px solid {sc}30;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;font-weight:700;color:{sc};line-height:1;">{si}</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:0.70rem;font-weight:600;color:{sc};margin-top:0.2rem;">{fb['message']}</div>
      <div style="font-size:0.57rem;color:var(--text-3);margin-top:0.25rem;text-transform:uppercase;letter-spacing:0.08em;">Grip (D)</div>
    </div>
    <div style="background:var(--bg-elevated);border-radius:6px;padding:0.65rem 0.5rem;text-align:center;border:1px solid {smc}30;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;font-weight:700;color:{smc};line-height:1;">{sm_icon}</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:0.70rem;font-weight:600;color:{smc};margin-top:0.2rem;">{_demo_smoothness.title()}</div>
      <div style="font-size:0.57rem;color:var(--text-3);margin-top:0.25rem;text-transform:uppercase;letter-spacing:0.08em;">Stability (A)</div>
    </div>
    <div style="background:var(--bg-elevated);border-radius:6px;padding:0.65rem 0.5rem;text-align:center;border:1px solid {jkc}30;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;font-weight:700;color:{jkc};line-height:1;">{jk_pct}</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:0.70rem;font-weight:600;color:{jkc};margin-top:0.2rem;">{jerk_label.title()}</div>
      <div style="font-size:0.57rem;color:var(--text-3);margin-top:0.25rem;text-transform:uppercase;letter-spacing:0.08em;">Smoothness (B)</div>
    </div>
    <div style="background:var(--bg-elevated);border-radius:6px;padding:0.65rem 0.5rem;text-align:center;border:1px solid {ecc}30;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;font-weight:700;color:{ecc};line-height:1;">{ec_pct}</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:0.70rem;font-weight:600;color:{ecc};margin-top:0.2rem;">{econ_label.title()}</div>
      <div style="font-size:0.57rem;color:var(--text-3);margin-top:0.25rem;text-transform:uppercase;letter-spacing:0.08em;">Economy (C)</div>
    </div>
    <div style="background:var(--bg-elevated);border-radius:6px;padding:0.65rem 0.5rem;text-align:center;border:1px solid {anc}30;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;font-weight:700;color:{anc};line-height:1;">{ang:.0f}°</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:0.70rem;font-weight:600;color:{anc};margin-top:0.2rem;">{an_label}</div>
      <div style="font-size:0.57rem;color:var(--text-3);margin-top:0.25rem;text-transform:uppercase;letter-spacing:0.08em;">Angle</div>
    </div>
    <div style="background:var(--bg-elevated);border-radius:6px;padding:0.65rem 0.5rem;text-align:center;border:1px solid {bm_color}30;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;font-weight:700;color:{bm_color};line-height:1;">{hands_count}</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:0.70rem;font-weight:600;color:{bm_color};margin-top:0.2rem;">{"Bimanual" if hands_count >= 2 else "1 Hand"}</div>
      <div style="font-size:0.57rem;color:var(--text-3);margin-top:0.25rem;text-transform:uppercase;letter-spacing:0.08em;">Hands</div>
    </div>
  </div>
  <div style="display:flex;justify-content:space-between;align-items:center;
              border-top:1px solid var(--border);padding-top:0.6rem;">
    <div style="font-size:0.80rem;color:var(--text-2);font-family:'IBM Plex Sans',sans-serif;">
      <span style="color:{sc};font-weight:700;">{si}</span>&nbsp; {fb['tip']}
    </div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.60rem;color:var(--text-3);
                white-space:nowrap;margin-left:1rem;">Ideal: {ideal_str}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    else:
        st.markdown("""
<div style="background:var(--bg-card);border:1px solid var(--border);border-radius:8px;
            padding:0.85rem 1.25rem;margin-bottom:1rem;display:flex;align-items:center;gap:0.75rem;">
  <span style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;color:var(--text-3);
               text-transform:uppercase;letter-spacing:0.12em;">Live Technique Monitor</span>
  <span style="font-family:'IBM Plex Mono',monospace;font-size:0.76rem;color:var(--text-3);">
    — No hands detected. Position hands in frame.</span>
</div>
""", unsafe_allow_html=True)

    # --- Technique Analysis detail panel ---
    with st.expander("Technique Details", expanded=False):
        smooth_icons = {"steady": "Steady", "moderate": "Moderate", "tremor": "Tremor"}
        smooth_colors = {"steady": "🟢", "moderate": "🟡", "tremor": "🔴"}

        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            st.metric("Smoothness", f"{smooth_colors.get(smoothness, '⚪')} {smooth_icons.get(smoothness, smoothness)}")
        with tc2:
            st.metric("Hands Detected", bimanual.get("hands_count", 0))
        with tc3:
            if bimanual.get("detected"):
                st.metric("Inter-hand Dist", f"{bimanual['inter_hand_dist']:.0f} px")
            else:
                st.metric("Inter-hand Dist", "—")

        if grips:
            grip_labels = {
                "pencil_grip": "Pencil Grip (pen-hold)",
                "palmar_grip": "Palmar Grip (power)",
                "precision_grip": "Precision Grip (3-finger)",
                "open_hand": "Open Hand",
            }
            for g in grips:
                hand_side = g.get("handedness", "?")
                grip_type = g.get("grip_type", "open_hand")
                angle = g.get("instrument_angle", 0)
                near = g.get("near_tools", [])
                tools_str = ", ".join(near) if near else "none"
                st.caption(
                    f"**{hand_side}**: {grip_labels.get(grip_type, grip_type)} · "
                    f"Angle: {angle:.0f}° · Near: {tools_str}"
                )
        else:
            st.caption("No hands detected — technique analysis requires hand visibility.")

        bh = st.session_state.get(KEY_BIMANUAL_HISTORY, [])
        if len(bh) >= 5:
            avg_dist = sum(bh[-20:]) / len(bh[-20:])
            std_dist = (sum((x - avg_dist) ** 2 for x in bh[-20:]) / len(bh[-20:])) ** 0.5
            sync_label = "Stable" if std_dist < 25 else "Variable" if std_dist < 50 else "Unstable"
            st.caption(f"Bimanual coordination: {sync_label} (avg {avg_dist:.0f} px, σ {std_dist:.0f} px)")

    # --- TTS for new prompts ---
    if new_prompt_messages:
        for msg in new_prompt_messages:
            queue_tts(msg)
    flush_tts_queue()

    # --- Coach Prompt Cards ---
    prompts = st.session_state.get(KEY_COACH_PROMPTS, [])
    resolved_ids = {o["prompt_id"] for o in st.session_state.get(KEY_OVERRIDES, [])}
    active_prompts = [p for p in prompts if p["prompt_id"] not in resolved_ids]
    active_prompts_sorted = sorted(
        active_prompts,
        key=prompt_sort_key,
        reverse=False,
    )
    recent = active_prompts_sorted[:8]

    high_n = sum(1 for p in active_prompts if p.get("risk_tier", "") == "high")
    med_n = sum(1 for p in active_prompts if p.get("risk_tier", "") == "medium")
    low_n = sum(1 for p in active_prompts if p.get("risk_tier", "") == "low")

    st.markdown("**Coach Prompts** (priority view)")
    st.caption(
        f"Open notes: 🔴 High {high_n} · 🟡 Medium {med_n} · 🟢 Low {low_n} "
        f"(duplicate prompts are auto-suppressed for {int(COACH_PROMPT_COOLDOWN_SEC)}s)"
    )
    if not recent:
        st.success("Great work! No coaching notes so far.")
    else:
        for p in recent:
            tier = p.get("risk_tier", "high")
            tier_colors = {"high": "🔴", "medium": "🟡", "low": "🟢"}
            tier_icon = tier_colors.get(tier, "⚪")
            acc_pct = f"{p.get('avg_conf', 0):.0%}"
            error_time = p.get("error_time", "")
            if not error_time:
                ts_str = p.get("ts", "")
                try:
                    error_time = datetime.fromisoformat(ts_str).strftime("%H:%M:%S")
                except Exception:
                    error_time = "—"
            phase_label = PHASE_LABELS.get(p.get("phase", ""), p.get("phase", ""))

            title = p.get("message", "")

            with st.container(border=True):
                st.markdown(f'<span class="prompt-tier-{tier}" style="display:none"></span>', unsafe_allow_html=True)
                st.markdown(
                    f"**{tier_icon} {title}**  \n"
                    f"<small>Accuracy: {acc_pct} ({tier}) · "
                    f"Time: {error_time} · Phase: {phase_label}</small>",
                    unsafe_allow_html=True,
                )
                if tier in ("low", "medium"):
                    bc1, bc2 = st.columns(2)
                    with bc1:
                        if st.button("Confirm issue", key=f"confirm_{p['prompt_id']}"):
                            override = {
                                "prompt_id": p["prompt_id"],
                                "decision": "confirm",
                                "ts": datetime.now().isoformat(),
                            }
                            st.session_state[KEY_OVERRIDES].append(override)
                            log_event("USER_OVERRIDE", override, mode="INTRA_OP", phase=phase)
                            st.rerun()
                    with bc2:
                        if st.button("Mark as resolved", key=f"deny_{p['prompt_id']}"):
                            override = {
                                "prompt_id": p["prompt_id"],
                                "decision": "deny",
                                "ts": datetime.now().isoformat(),
                            }
                            st.session_state[KEY_OVERRIDES].append(override)
                            log_event("USER_OVERRIDE", override, mode="INTRA_OP", phase=phase)
                            st.rerun()

    st.markdown("**AI Reasoning**")
    if not BRAIN_AVAILABLE:
        st.caption(BRAIN_IMPORT_ERROR)
    else:
        if st.button("Generate AI reasoning for this session", key="brain_reason_btn", width="stretch"):
            with st.spinner("Generating AI reasoning..."):
                prompts_data = st.session_state.get(KEY_ALERTS_LOG, [])
                overrides_data = st.session_state.get(KEY_OVERRIDES, [])
                event_log = build_brain_event_log(prompts_data, overrides_data)
                summary = generate_final_critique(
                    procedure="Medical lab training session",
                    clarity_feedback=[p.get("message", "") for p in prompts_data][-25:],
                    event_log=event_log,
                    mastery_score=max(0, 100 - len(prompts_data) * 2),
                )
                st.session_state["_brain_summary"] = summary
        ai_now = st.session_state.get("_brain_summary", "")
        if ai_now:
            st.markdown(ai_now)

    if st.button("End Lab Session", type="primary", width="stretch"):
        set_mode("POST_OP")
        st.session_state[KEY_NAV] = "Report"
        st.session_state[KEY_STREAM_START_TS] = None
        log_event("STATE_CHANGE", {"from": "INTRA_OP", "to": "POST_OP"}, mode="POST_OP")
        st.rerun()

# =============================================================================
# REPORT tab — minimal error report for end users
# =============================================================================

if nav == "Report":
    st.markdown(
        '<div class="section-header">'
        '<span class="section-num">03 / REPORT</span>'
        '<span class="section-title">Error Report</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    prompts = st.session_state.get(KEY_ALERTS_LOG, [])
    if not prompts:
        st.success("No coaching errors recorded in this session.")
    else:
        rows = []
        for p in prompts:
            error_time = p.get("error_time", "")
            if not error_time:
                ts_str = p.get("ts", "")
                try:
                    error_time = datetime.fromisoformat(ts_str).strftime("%H:%M:%S")
                except Exception:
                    error_time = "—"
            rows.append({
                "time": error_time,
                "phase": PHASE_LABELS.get(p.get("phase", ""), p.get("phase", "")),
                "severity": (p.get("risk_tier", "high") or "high").upper(),
                "error": p.get("message", ""),
            })

        sev_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        rows.sort(key=lambda r: (sev_rank.get(r["severity"], 3), r["time"]), reverse=False)

        st.metric("Total Errors", len(rows))
        st.caption("Only showing what was wrong and when it happened.")
        st.table([("Time", "Phase", "Severity", "Error")] + [
            (r["time"], r["phase"], r["severity"], r["error"]) for r in rows
        ])

    st.markdown("---")
    st.markdown("### AI Coach Summary")
    if not BRAIN_AVAILABLE:
        st.caption(BRAIN_IMPORT_ERROR)
    else:
        if st.button("Generate AI Summary", width="stretch"):
            with st.spinner("Generating summary..."):
                prompts_data = st.session_state.get(KEY_ALERTS_LOG, [])
                overrides_data = st.session_state.get(KEY_OVERRIDES, [])
                event_log = build_brain_event_log(prompts_data, overrides_data)
                summary = generate_final_critique(
                    procedure="Medical lab training session",
                    clarity_feedback=[p.get("message", "") for p in prompts_data][-25:],
                    event_log=event_log,
                    mastery_score=max(0, 100 - len(prompts_data) * 2),
                )
                resources = generate_learning_resources("basic surgical instrument handling") \
                    if generate_learning_resources is not None else ""
                st.session_state["_brain_summary"] = summary
                st.session_state["_brain_resources"] = resources

        ai_summary = st.session_state.get("_brain_summary", "")
        if ai_summary:
            st.markdown(ai_summary)
            ai_resources = st.session_state.get("_brain_resources", "")
            if ai_resources:
                st.markdown("---")
                st.markdown("**Recommended Resources**")
                st.markdown(ai_resources)

# =============================================================================
# Footer
# =============================================================================

st.markdown("---")
st.caption(f"SurgiPath • {datetime.now().strftime('%Y-%m-%d %H:%M')}")
