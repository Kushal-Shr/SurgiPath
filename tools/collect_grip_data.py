"""
Grip Data Collection Tool for MedLab Coach AI
===============================================

HOW TO USE:
  1. Run:  streamlit run tools/collect_grip_data.py --server.headless true
  2. Select the TOOL you are holding (scalpel, needle_holder, etc.)
  3. Select the GRIP TYPE (pencil, palmar, precision, open_hand)
  4. Hold the tool in front of the camera in that grip
  5. Click "Record" — it captures MediaPipe landmarks for ~3 seconds
  6. Repeat for each tool+grip combination (~50 samples each)
  7. Data is saved to data/grip_samples.csv

HOW IT WORKS:
  - Uses MediaPipe Hands to detect 21 hand landmarks per frame
  - Extracts normalized features: finger curl ratios, angles, distances
  - Saves raw landmarks + computed features + label to CSV
  - Each row = one frame snapshot with label

AFTER COLLECTION:
  Run tools/train_grip_classifier.py to train the classifier.
"""

import csv
import math
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title="Grip Data Collector", page_icon="✋", layout="wide")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
CSV_PATH = DATA_DIR / "grip_samples.csv"

TOOLS = ["scalpel", "needle_holder", "forceps", "scissors", "clamp", "syringe", "none"]
GRIP_TYPES = ["pencil_grip", "palmar_grip", "precision_grip", "open_hand"]

WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_MCP = 13
RING_TIP = 16
PINKY_MCP = 17
PINKY_TIP = 20


def _dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def extract_features(landmarks_px: list[tuple]) -> dict:
    """Extract normalized geometric features from 21 hand landmarks."""
    if len(landmarks_px) < 21:
        return {}

    wrist = landmarks_px[WRIST]
    thumb_tip = landmarks_px[THUMB_TIP]
    index_tip = landmarks_px[INDEX_TIP]
    index_mcp = landmarks_px[INDEX_MCP]
    middle_tip = landmarks_px[MIDDLE_TIP]
    middle_mcp = landmarks_px[MIDDLE_MCP]
    ring_tip = landmarks_px[RING_TIP]
    ring_mcp = landmarks_px[RING_MCP]
    pinky_tip = landmarks_px[PINKY_TIP]
    pinky_mcp = landmarks_px[PINKY_MCP]

    palm_size = _dist(wrist, index_mcp)
    if palm_size < 1:
        return {}

    # Normalized distances (scale-invariant)
    thumb_index_dist = _dist(thumb_tip, index_tip) / palm_size
    thumb_middle_dist = _dist(thumb_tip, middle_tip) / palm_size
    thumb_ring_dist = _dist(thumb_tip, ring_tip) / palm_size
    thumb_pinky_dist = _dist(thumb_tip, pinky_tip) / palm_size

    # Finger curl ratios: tip_to_wrist / mcp_to_wrist (>1 = extended, <1 = curled)
    def curl(tip, mcp):
        return _dist(tip, wrist) / max(_dist(mcp, wrist), 1e-6)

    index_curl = curl(index_tip, index_mcp)
    middle_curl = curl(middle_tip, middle_mcp)
    ring_curl = curl(ring_tip, ring_mcp)
    pinky_curl = curl(pinky_tip, pinky_mcp)

    # Finger spread: distance between adjacent fingertips / palm_size
    index_middle_spread = _dist(index_tip, middle_tip) / palm_size
    middle_ring_spread = _dist(middle_tip, ring_tip) / palm_size
    ring_pinky_spread = _dist(ring_tip, pinky_tip) / palm_size

    # Grip angle: angle at wrist between thumb and index vectors
    v1 = (thumb_tip[0] - wrist[0], thumb_tip[1] - wrist[1])
    v2 = (index_tip[0] - wrist[0], index_tip[1] - wrist[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    if mag1 > 0 and mag2 > 0:
        cos_a = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        grip_angle = math.degrees(math.acos(cos_a))
    else:
        grip_angle = 0.0

    # Instrument angle: wrist to fingertip midpoint vs horizontal
    mid_tip_x = (index_tip[0] + middle_tip[0]) / 2
    mid_tip_y = (index_tip[1] + middle_tip[1]) / 2
    dx = mid_tip_x - wrist[0]
    dy = mid_tip_y - wrist[1]
    instrument_angle = math.degrees(math.atan2(abs(dy), abs(dx))) if (abs(dx) > 1e-6 or abs(dy) > 1e-6) else 0.0

    return {
        "thumb_index_dist": round(thumb_index_dist, 4),
        "thumb_middle_dist": round(thumb_middle_dist, 4),
        "thumb_ring_dist": round(thumb_ring_dist, 4),
        "thumb_pinky_dist": round(thumb_pinky_dist, 4),
        "index_curl": round(index_curl, 4),
        "middle_curl": round(middle_curl, 4),
        "ring_curl": round(ring_curl, 4),
        "pinky_curl": round(pinky_curl, 4),
        "index_middle_spread": round(index_middle_spread, 4),
        "middle_ring_spread": round(middle_ring_spread, 4),
        "ring_pinky_spread": round(ring_pinky_spread, 4),
        "grip_angle": round(grip_angle, 2),
        "instrument_angle": round(instrument_angle, 2),
        "palm_size": round(palm_size, 2),
    }


FEATURE_NAMES = [
    "thumb_index_dist", "thumb_middle_dist", "thumb_ring_dist", "thumb_pinky_dist",
    "index_curl", "middle_curl", "ring_curl", "pinky_curl",
    "index_middle_spread", "middle_ring_spread", "ring_pinky_spread",
    "grip_angle", "instrument_angle", "palm_size",
]


def _init_csv():
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "tool", "grip_type"] + FEATURE_NAMES)


def _append_row(tool: str, grip_type: str, features: dict):
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        row = [time.time(), tool, grip_type] + [features.get(fn, 0) for fn in FEATURE_NAMES]
        writer.writerow(row)


def _count_samples() -> dict:
    if not CSV_PATH.exists():
        return {}
    counts = {}
    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row.get('tool', '?')} / {row.get('grip_type', '?')}"
            counts[key] = counts.get(key, 0) + 1
    return counts


def _detect_hands_mp(frame_rgb):
    """Detect hands using MediaPipe Tasks API (same as src/hands.py)."""
    try:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
        import mediapipe as mp
    except ImportError:
        return []

    model_path = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
    if not model_path.exists():
        return []

    if "mp_detector" not in st.session_state:
        try:
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.5,
            )
            st.session_state["mp_detector"] = HandLandmarker.create_from_options(options)
        except Exception:
            return []

    detector = st.session_state["mp_detector"]
    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = detector.detect(mp_image)
    except Exception:
        return []

    h, w = frame_rgb.shape[:2]
    hands = []
    if not result.hand_landmarks:
        return hands

    for hand_lm in result.hand_landmarks:
        lms = [(lm.x * w, lm.y * h, lm.z) for lm in hand_lm]
        hands.append(lms)
    return hands


# ============================== UI ==============================

_init_csv()

st.title("✋ Grip Data Collector")
st.caption("Record hand grip samples for training a grip classifier. Hold a tool, select the grip type, and record.")

col_left, col_right = st.columns([2, 1])

with col_right:
    tool = st.selectbox("Tool being held", TOOLS, key="tool_select")
    grip = st.selectbox("Grip type", GRIP_TYPES, key="grip_select")

    st.markdown("---")
    recording = st.toggle("Recording", value=False, key="recording_toggle")
    if recording:
        st.warning("RECORDING — hold your grip steady in front of the camera")
    else:
        st.info("Toggle Recording ON, then hold your grip in front of camera")

    st.markdown("---")
    st.markdown("### Sample Counts")
    counts = _count_samples()
    if counts:
        for k, v in sorted(counts.items()):
            color = "🟢" if v >= 50 else "🟡" if v >= 20 else "🔴"
            st.caption(f"{color} {k}: **{v}** samples")
    else:
        st.caption("No samples yet. Start recording!")

    total = sum(counts.values()) if counts else 0
    st.metric("Total Samples", total)

    if st.button("Clear All Data", type="secondary"):
        if CSV_PATH.exists():
            CSV_PATH.unlink()
            _init_csv()
            st.rerun()

with col_left:
    cap = st.session_state.get("_collector_cap")
    if cap is None or not cap.isOpened():
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY)
            st.session_state["_collector_cap"] = cap
        except Exception:
            cap = None

    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands = _detect_hands_mp(frame_rgb)

            if hands:
                # Draw skeleton
                CONNECTIONS = [
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    (0, 9), (9, 10), (10, 11), (11, 12),
                    (0, 13), (13, 14), (14, 15), (15, 16),
                    (0, 17), (17, 18), (18, 19), (19, 20),
                    (5, 9), (9, 13), (13, 17),
                ]
                for lms in hands:
                    pts = [(int(l[0]), int(l[1])) for l in lms]
                    for a, b in CONNECTIONS:
                        if a < len(pts) and b < len(pts):
                            cv2.line(frame, pts[a], pts[b], (0, 255, 128), 2)
                    for pt in pts:
                        cv2.circle(frame, pt, 3, (255, 255, 255), -1)

                features = extract_features(hands[0])

                if recording and features:
                    _append_row(tool, grip, features)
                    cv2.putText(frame, "REC", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    status_placeholder.success(f"Recording: {tool} / {grip}")
                elif features:
                    status_placeholder.info(f"Hand detected — {len(hands)} hand(s). Toggle Recording to save.")
                else:
                    status_placeholder.warning("Hand detected but features could not be extracted.")
            else:
                status_placeholder.warning("No hands detected. Show your hand to the camera.")

            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
    else:
        st.error("Camera not available. Connect a webcam and reload.")

    if cap is not None and cap.isOpened():
        time.sleep(0.1)
        st.rerun()
