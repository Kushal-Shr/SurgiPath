"""
MediaPipe Hands integration — hand detection, technique analysis, tool proximity.

Uses the MediaPipe Tasks API (HandLandmarker), which works on Python 3.13+.
Falls back gracefully: if the model file is missing or MediaPipe fails to
initialize, all functions return empty results and the app continues without
hand tracking.

Provides 21 keypoints per hand (wrist, finger joints, fingertips). Used to:
  - Measure hand stability (jitter between frames)
  - Detect if a hand is near/holding a tool (centroid overlap with tool bbox)
  - Classify grip type (pencil, palmar, precision, open)
  - Estimate instrument angle relative to horizontal
  - Assess motion smoothness over rolling windows
  - Evaluate bimanual coordination (inter-hand distance + sync)
  - Draw hand skeleton overlay on the video frame

The detector is loaded once and reused across frames.
"""

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

HAND_LANDMARKER_MODEL = "models/hand_landmarker.task"
GRIP_CLASSIFIER_MODEL = "models/grip_classifier.joblib"

# ── Ideal grip per procedure phase ──────────────────────────────────────────
IDEAL_GRIPS_BY_PHASE: dict[str, list[str]] = {
    "incision":  ["pencil_grip"],
    "suturing":  ["pencil_grip", "precision_grip"],
    "closing":   ["precision_grip", "pencil_grip"],
    "exposure":  ["precision_grip", "pencil_grip"],
    "prep":      ["pencil_grip", "precision_grip"],
}

GRIP_DISPLAY_NAMES: dict[str, str] = {
    "pencil_grip":    "Pencil",
    "precision_grip": "Precision",
    "palmar_grip":    "Palmar",
    "open_hand":      "Open Hand",
}

# BGR colors used for hand skeleton overlays
_COLOR_GOOD = (80, 210, 60)    # green  — correct grip
_COLOR_WARN = (30, 155, 255)   # orange — suboptimal grip
_COLOR_BAD  = (55, 55, 240)    # red    — wrong grip

_detector: Any = None
_mediapipe_available: bool | None = None
_grip_model: dict | None = None
_grip_model_loaded: bool = False

# Landmark indices (same as the 21-point MediaPipe hand model)
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


def _init_detector() -> bool:
    """Try to create the HandLandmarker. Returns True on success."""
    global _detector, _mediapipe_available
    if _mediapipe_available is False:
        return False
    if _detector is not None:
        return True
    try:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker,
            HandLandmarkerOptions,
            RunningMode,
        )

        model_path = Path(HAND_LANDMARKER_MODEL)
        if not model_path.exists():
            _mediapipe_available = False
            return False

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.4,
        )
        _detector = HandLandmarker.create_from_options(options)
        _mediapipe_available = True
        return True
    except Exception:
        _mediapipe_available = False
        _detector = None
        return False


def detect_hands(frame_rgb: np.ndarray) -> list[dict]:
    """
    Run MediaPipe Hands on an RGB frame.

    Returns list of hand dicts:
      [{"landmarks": [(x, y, z) * 21],
        "handedness": "Left" | "Right",
        "wrist": (x, y),
        "fingertips": [(x, y) * 5],
        "centroid": (x, y)}, ...]

    Coordinates are in pixel space (scaled by frame dimensions).
    Returns empty list if MediaPipe is unavailable.
    """
    if not _init_detector():
        return []
    try:
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = _detector.detect(mp_image)
    except Exception:
        return []

    h, w = frame_rgb.shape[:2]
    hands: list[dict] = []
    if not result.hand_landmarks:
        return hands

    for i, hand_lm in enumerate(result.hand_landmarks):
        lms = [(lm.x * w, lm.y * h, lm.z) for lm in hand_lm]

        handedness = "Right"
        if result.handedness and i < len(result.handedness):
            handedness = result.handedness[i][0].category_name

        wrist = (lms[WRIST][0], lms[WRIST][1])
        fingertips = [
            (lms[THUMB_TIP][0], lms[THUMB_TIP][1]),
            (lms[INDEX_TIP][0], lms[INDEX_TIP][1]),
            (lms[MIDDLE_TIP][0], lms[MIDDLE_TIP][1]),
            (lms[RING_TIP][0], lms[RING_TIP][1]),
            (lms[PINKY_TIP][0], lms[PINKY_TIP][1]),
        ]
        cx = sum(l[0] for l in lms) / len(lms)
        cy = sum(l[1] for l in lms) / len(lms)

        hands.append({
            "landmarks": lms,
            "handedness": handedness,
            "wrist": wrist,
            "fingertips": fingertips,
            "centroid": (cx, cy),
        })
    return hands


def compute_hand_jitter(
    current_hands: list[dict],
    previous_hands: list[dict] | None,
) -> float:
    """
    Compute average wrist displacement (px) between current and previous frame.
    Returns 0.0 if comparison is not possible. Lower = more stable.
    """
    if not current_hands or not previous_hands:
        return 0.0
    total = 0.0
    matched = 0
    for curr in current_hands:
        best_dist = float("inf")
        for prev in previous_hands:
            dx = curr["wrist"][0] - prev["wrist"][0]
            dy = curr["wrist"][1] - prev["wrist"][1]
            d = math.sqrt(dx * dx + dy * dy)
            best_dist = min(best_dist, d)
        if best_dist < float("inf"):
            total += best_dist
            matched += 1
    return total / max(1, matched)


def compute_grip_angle(landmarks: list[tuple]) -> float:
    """
    Compute the angle (degrees) between thumb tip and index tip relative to wrist.
    A narrow angle suggests a pinch/grip, wider angle suggests open hand.
    """
    if len(landmarks) < 21:
        return 0.0
    wx, wy = landmarks[WRIST][0], landmarks[WRIST][1]
    tx, ty = landmarks[THUMB_TIP][0], landmarks[THUMB_TIP][1]
    ix, iy = landmarks[INDEX_TIP][0], landmarks[INDEX_TIP][1]
    v1 = (tx - wx, ty - wy)
    v2 = (ix - wx, iy - wy)
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def hand_near_tool(
    hand: dict,
    tool_bbox: list[float],
    margin: float = 40.0,
) -> bool:
    """
    Check if the hand centroid is inside or near a tool's bounding box.
    tool_bbox: [x1, y1, x2, y2] in pixel coordinates.
    margin: extra pixels around the bbox to count as "near".
    """
    cx, cy = hand["centroid"]
    x1, y1, x2, y2 = tool_bbox
    return (x1 - margin) <= cx <= (x2 + margin) and (y1 - margin) <= cy <= (y2 + margin)


def get_held_tools(
    hands: list[dict],
    detections: list[dict],
    margin: float = 40.0,
) -> set[str]:
    """
    Return set of normalized tool names that have a hand centroid nearby.
    """
    held: set[str] = set()
    for det in detections:
        bbox = det.get("xyxy", [])
        name = (det.get("name", "") or "").strip().lower().replace(" ", "_")
        if len(bbox) != 4 or not name:
            continue
        for hand in hands:
            if hand_near_tool(hand, bbox, margin):
                held.add(name)
                break
    return held


def classify_grip(landmarks: list[tuple]) -> str:
    """
    Classify the grip type from 21 hand landmarks.

    Returns one of: "pencil_grip", "palmar_grip", "precision_grip", "open_hand".

    Heuristics:
      - pencil_grip: thumb-index pinch with other fingers curled (holding like a pen)
      - palmar_grip: all fingers wrapped around an object (power grip)
      - precision_grip: thumb-index-middle pinch (three-finger grip)
      - open_hand: no grip detected — fingers extended
    """
    if len(landmarks) < 21:
        return "open_hand"

    def _dist(a: tuple, b: tuple) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    thumb_tip = landmarks[THUMB_TIP]
    index_tip = landmarks[INDEX_TIP]
    middle_tip = landmarks[MIDDLE_TIP]
    ring_tip = landmarks[RING_TIP]
    pinky_tip = landmarks[PINKY_TIP]
    wrist = landmarks[WRIST]
    index_mcp = landmarks[INDEX_MCP]
    middle_mcp = landmarks[MIDDLE_MCP]
    ring_mcp = landmarks[RING_MCP]
    pinky_mcp = landmarks[PINKY_MCP]

    palm_size = _dist(wrist, (index_mcp[0], index_mcp[1]))
    if palm_size < 1:
        return "open_hand"

    thumb_index_dist = _dist(thumb_tip, index_tip) / palm_size
    thumb_middle_dist = _dist(thumb_tip, middle_tip) / palm_size

    # Finger curl: tip closer to MCP than extended → curled
    def _curl_ratio(tip: tuple, mcp: tuple) -> float:
        tip_wrist = _dist(tip, wrist)
        mcp_wrist = _dist(mcp, wrist)
        return tip_wrist / max(mcp_wrist, 1e-6)

    ring_curl = _curl_ratio(ring_tip, (ring_mcp[0], ring_mcp[1]))
    pinky_curl = _curl_ratio(pinky_tip, (pinky_mcp[0], pinky_mcp[1]))
    middle_curl = _curl_ratio(middle_tip, (middle_mcp[0], middle_mcp[1]))

    pinch = thumb_index_dist < 0.45
    three_finger_pinch = pinch and thumb_middle_dist < 0.55
    others_curled = ring_curl < 1.3 and pinky_curl < 1.3
    all_curled = others_curled and middle_curl < 1.3

    if three_finger_pinch and others_curled:
        return "precision_grip"
    if pinch and others_curled and middle_curl < 1.4:
        return "pencil_grip"
    if all_curled:
        return "palmar_grip"
    return "open_hand"


def compute_instrument_angle(landmarks: list[tuple]) -> float:
    """
    Estimate the angle (degrees) of the held instrument relative to horizontal.

    Uses the vector from wrist to the midpoint of index+middle fingertips
    as a proxy for the instrument axis. 0 = horizontal, 90 = vertical.
    """
    if len(landmarks) < 21:
        return 0.0
    wx, wy = landmarks[WRIST][0], landmarks[WRIST][1]
    ix, iy = landmarks[INDEX_TIP][0], landmarks[INDEX_TIP][1]
    mx, my = landmarks[MIDDLE_TIP][0], landmarks[MIDDLE_TIP][1]
    tip_x = (ix + mx) / 2
    tip_y = (iy + my) / 2
    dx = tip_x - wx
    dy = tip_y - wy
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    angle_rad = math.atan2(abs(dy), abs(dx))
    return math.degrees(angle_rad)


def classify_smoothness(jitter_samples: list[float], window: int = 20) -> str:
    """
    Classify motion smoothness from recent jitter samples.

    Returns: "steady", "moderate", or "tremor".
    Thresholds are in pixels of average wrist displacement per frame.
    """
    if not jitter_samples:
        return "steady"
    recent = jitter_samples[-window:]
    avg = sum(recent) / len(recent)
    if avg < 3.0:
        return "steady"
    if avg < 8.0:
        return "moderate"
    return "tremor"


def compute_bimanual(hands: list[dict]) -> dict:
    """
    Evaluate bimanual coordination when two hands are detected.

    Returns:
      {"detected": bool, "inter_hand_dist": float, "hands_count": int}

    inter_hand_dist is the pixel distance between the two wrist positions.
    When only one or zero hands are detected, inter_hand_dist is 0.
    """
    if len(hands) < 2:
        return {"detected": False, "inter_hand_dist": 0.0, "hands_count": len(hands)}
    w1 = hands[0]["wrist"]
    w2 = hands[1]["wrist"]
    dist = math.sqrt((w1[0] - w2[0]) ** 2 + (w1[1] - w2[1]) ** 2)
    return {"detected": True, "inter_hand_dist": round(dist, 1), "hands_count": 2}


def compute_jerk_smoothness(tip_history: list[tuple[float, float]]) -> dict:
    """
    Compute motion smoothness from index-fingertip trajectory using jerk (3rd derivative).

    tip_history: list of (x, y) pixel positions, newest last.
    Returns:
      {
        "smoothness_score": float 0–1  (1 = perfectly smooth),
        "mean_jerk":        float px/frame³,
        "label":            "fluid" | "moderate" | "jerky",
      }
    Requires at least 4 samples to compute one jerk value.
    """
    n = len(tip_history)
    if n < 4:
        return {"smoothness_score": 1.0, "mean_jerk": 0.0, "label": "fluid"}

    pts = tip_history[-30:]  # cap at last 30 frames

    def _dist(a: tuple, b: tuple) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    # velocity magnitudes
    vel = [_dist(pts[i], pts[i - 1]) for i in range(1, len(pts))]
    # acceleration magnitudes
    acc = [abs(vel[i] - vel[i - 1]) for i in range(1, len(vel))]
    # jerk magnitudes
    jerk = [abs(acc[i] - acc[i - 1]) for i in range(1, len(acc))]

    if not jerk:
        return {"smoothness_score": 1.0, "mean_jerk": 0.0, "label": "fluid"}

    mean_jerk = sum(jerk) / len(jerk)

    # Normalize: 0 px/f³ → score 1.0; ≥15 px/f³ → score 0.0
    score = max(0.0, 1.0 - mean_jerk / 15.0)

    if score >= 0.72:
        label = "fluid"
    elif score >= 0.40:
        label = "moderate"
    else:
        label = "jerky"

    return {
        "smoothness_score": round(score, 3),
        "mean_jerk": round(mean_jerk, 2),
        "label": label,
    }


def compute_motion_economy(
    path_history: list[tuple[float, float]],
    idle_threshold_px: float = 4.0,
) -> dict:
    """
    Compute economy of motion from wrist trajectory over a session window.

    path_history: list of (x, y) wrist positions, newest last (up to ~300 samples).
    idle_threshold_px: movement below this per frame is counted as idle.

    Returns:
      {
        "path_length_px":  float,
        "directness":      float 0–1  (displacement / path_length; 1 = straight line),
        "idle_ratio":      float 0–1  (fraction of frames the hand was idle),
        "label":           "efficient" | "moderate" | "excessive",
      }
    """
    pts = path_history[-300:]
    n = len(pts)
    if n < 2:
        return {
            "path_length_px": 0.0,
            "directness": 1.0,
            "idle_ratio": 0.0,
            "label": "efficient",
        }

    def _dist(a: tuple, b: tuple) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    steps = [_dist(pts[i], pts[i - 1]) for i in range(1, n)]
    path_length = sum(steps)

    displacement = _dist(pts[0], pts[-1])
    directness = displacement / max(path_length, 1e-6)

    idle_frames = sum(1 for s in steps if s < idle_threshold_px)
    idle_ratio = idle_frames / max(len(steps), 1)

    # Economy label based on directness + idle
    if directness >= 0.55 and idle_ratio <= 0.35:
        label = "efficient"
    elif directness >= 0.30 or idle_ratio <= 0.60:
        label = "moderate"
    else:
        label = "excessive"

    return {
        "path_length_px": round(path_length, 1),
        "directness": round(directness, 3),
        "idle_ratio": round(idle_ratio, 3),
        "label": label,
    }


def compute_technique_summary(
    hands: list[dict],
    jitter_samples: list[float],
    detections: list[dict],
    margin: float = 40.0,
) -> dict:
    """
    Compute a full technique summary for the current frame.

    Returns dict with:
      - grips: list of {hand, grip_type, instrument_angle, near_tools}
      - smoothness: "steady" / "moderate" / "tremor"
      - bimanual: bimanual coordination dict
    """
    grips = []
    for hand in hands:
        lms = hand.get("landmarks", [])
        grip = classify_grip(lms)
        angle = compute_instrument_angle(lms)

        near = []
        for det in detections:
            bbox = det.get("xyxy", [])
            name = (det.get("name", "") or "").strip().lower().replace(" ", "_")
            if len(bbox) == 4 and name and hand_near_tool(hand, bbox, margin):
                near.append(name)

        grips.append({
            "handedness": hand.get("handedness", "?"),
            "grip_type": grip,
            "instrument_angle": round(angle, 1),
            "near_tools": near,
        })

    return {
        "grips": grips,
        "smoothness": classify_smoothness(jitter_samples),
        "bimanual": compute_bimanual(hands),
    }


def get_technique_feedback(
    grip: str,
    phase: str,
    smoothness: str,
    angle: float,
) -> dict:
    """
    Return real-time coaching feedback based on grip, phase, smoothness, and angle.

    Returns dict:
      {
        "status":   "good" | "warning" | "error",
        "grip_ok":  bool,
        "message":  str,   # short label, e.g. "Pencil Grip ✓"
        "tip":      str,   # one coaching sentence
      }
    """
    ideal = IDEAL_GRIPS_BY_PHASE.get(phase, ["pencil_grip", "precision_grip"])
    grip_name = GRIP_DISPLAY_NAMES.get(grip, grip.replace("_", " ").title())
    grip_ok = grip in ideal and grip != "open_hand"

    if grip == "open_hand":
        status = "error"
        message = "No Grip Detected"
        ideal_name = GRIP_DISPLAY_NAMES.get(ideal[0], ideal[0])
        tip = f"Hold the instrument — use {ideal_name} grip for {phase}."
    elif not grip_ok:
        status = "warning"
        message = f"{grip_name} Grip"
        ideal_name = GRIP_DISPLAY_NAMES.get(ideal[0], ideal[0])
        tip = f"Try switching to {ideal_name} grip for better {phase} control."
    else:
        status = "good"
        message = f"{grip_name} Grip \u2713"
        tip = f"Good — {grip_name} grip is correct for {phase}."

    # Smoothness can override to a worse status
    if smoothness == "tremor":
        status = "error"
        tip = "Hand tremor detected. Rest your wrists on the table to stabilize."
    elif smoothness == "moderate" and status == "good":
        status = "warning"
        tip += " Slight movement detected — try to keep your wrists steadier."

    # Angle feedback
    if phase == "incision" and angle > 70:
        if status == "good":
            status = "warning"
        tip = f"Instrument angle too steep ({angle:.0f}\u00b0). Aim for 30\u201345\u00b0 for a controlled incision."
    elif phase == "suturing" and angle < 15:
        if status == "good":
            status = "warning"
        tip = f"Instrument nearly horizontal ({angle:.0f}\u00b0). Raise slightly for better needle penetration."

    return {
        "status":   status,
        "grip_ok":  grip_ok,
        "message":  message,
        "tip":      tip,
        "grip":     grip,
        "smoothness": smoothness,
        "angle":    angle,
    }


def _grip_color(grip: str) -> tuple[int, int, int]:
    """Return BGR color based on grip quality."""
    if grip in ("pencil_grip", "precision_grip"):
        return _COLOR_GOOD
    if grip == "palmar_grip":
        return _COLOR_WARN
    return _COLOR_BAD


def draw_hands(frame_bgr: np.ndarray, hands: list[dict]) -> np.ndarray:
    """Draw color-coded hand skeletons on a BGR frame. Returns the annotated frame."""
    if not hands:
        return frame_bgr
    out = frame_bgr.copy()

    CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]

    for hand in hands:
        lms = hand["landmarks"]
        pts = [(int(l[0]), int(l[1])) for l in lms]
        grip = classify_grip(lms)
        angle = compute_instrument_angle(lms)
        color = _grip_color(grip)

        # Skeleton
        for a, b in CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(out, pts[a], pts[b], color, 2, cv2.LINE_AA)

        # Landmarks — larger circles for fingertips
        fingertip_idx = {4, 8, 12, 16, 20}
        for i, pt in enumerate(pts):
            r = 5 if i in fingertip_idx else 3
            cv2.circle(out, pt, r, color, -1, cv2.LINE_AA)
            cv2.circle(out, pt, r + 1, (0, 0, 0), 1, cv2.LINE_AA)

        # Instrument axis line (wrist → index+middle midpoint)
        if len(pts) >= 21:
            tip_x = (pts[8][0] + pts[12][0]) // 2
            tip_y = (pts[8][1] + pts[12][1]) // 2
            cv2.line(out, pts[0], (tip_x, tip_y), color, 1, cv2.LINE_AA)

        # Grip status pill
        side = hand.get("handedness", "?")[0]
        grip_short = GRIP_DISPLAY_NAMES.get(grip, grip)
        label = f"{side}: {grip_short}  {angle:.0f}\u00b0"
        cx, cy = int(hand["centroid"][0]), int(hand["centroid"][1])
        lx = max(4, cx - 50)
        ly = max(18, cy - 20)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        pad = 4
        cv2.rectangle(out, (lx - pad, ly - th - pad), (lx + tw + pad, ly + pad),
                      (10, 12, 16), -1)
        cv2.rectangle(out, (lx - pad, ly - th - pad), (lx + tw + pad, ly + pad),
                      color, 1)
        cv2.putText(out, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

    return out
