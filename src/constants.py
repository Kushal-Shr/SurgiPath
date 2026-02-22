"""
Shared constants for MedLab Coach AI.

Single place to change paths, config defaults, session state key names.
"""

# Paths (relative to project root)
MODEL_PATH = "models/best.pt"
RECIPE_PATH = "recipes/trauma_room.json"
EVENTS_LOG_PATH = "logs/events.jsonl"
SAMPLE_VIDEO_PATH = "assets/sample_lab.mp4"

# Config defaults
CONF_MIN_DEFAULT = 0.45
IMGSZ_DEFAULT = 640
FRAME_SKIP_DEFAULT = 3
SMOOTH_WINDOW_SIZE = 20
FEED_LOOP_FRAMES = 60
STABLE_SECONDS_DEFAULT = 2.0

# --- Coach-mode gating defaults (overridable per-recipe via params) ---
EVIDENCE_WINDOW = 20              # sliding window length for EvidenceState
MISSING_GRACE_SEC = 3.0           # seconds before a missing tool triggers a prompt
CONF_HIGH = 0.75                  # avg_conf >= this → high-confidence prompt
CONF_MEDIUM = 0.50                # avg_conf >= this → medium-confidence prompt
SEEN_RATIO_HIGH = 0.60            # seen_ratio >= this → high tier
SEEN_RATIO_MEDIUM = 0.40          # seen_ratio >= this → medium tier
CALIBRATION_MIN_BRIGHTNESS = 50   # grayscale mean below this → "too dark"
CALIBRATION_MAX_BRIGHTNESS = 230  # grayscale mean above this → "over-exposed"
CALIBRATION_MIN_BLUR = 30         # Laplacian variance below this → "blurry"
CALIBRATION_MIN_DET_RATE = 0.10   # detection_rate below this → "no tools detected"
AUTO_TRANSITION_ENABLED = True    # auto-switch Setup→Practice when checklist passes
COACH_PROMPT_COOLDOWN_SEC = 8.0   # suppress duplicate prompt per rule within cooldown window
WEBRTC_QUEUE_MAXSIZE = 8          # keep queue small to avoid stale-frame latency
PERF_WINDOW_SIZE = 120            # rolling samples for FPS/latency stats

# Session state keys
KEY_NAV = "nav"
KEY_PREOP_SMOOTHER = "preop_smoother"
KEY_PREOP_STABLE_START = "preop_stable_start"
KEY_ALERTS_LOG = "alerts_log"
KEY_LAST_DETECTIONS = "last_detections"
KEY_LAST_COUNTS = "last_counts"
KEY_FRAME_INDEX = "frame_index"
KEY_CONFIG_CONF_MIN = "config_conf_min"
KEY_CONFIG_IMGSZ = "config_imgsz"
KEY_CONFIG_FRAME_SKIP = "config_frame_skip"
KEY_CAMERA = "camera"

# EdTech / video source
KEY_VIDEO_SOURCE = "video_source"
KEY_UPLOADED_FILE = "uploaded_file_bytes"
KEY_STREAK_SECONDS = "streak_seconds"
KEY_STREAK_BEST = "streak_best"
KEY_SESSION_START = "session_start_ts"
KEY_HAND_STABILITY = "hand_stability_samples"
KEY_DEMO_MODE = "demo_mode"
KEY_DEMO_TICK = "demo_tick"

# Coach mode
KEY_EVIDENCE = "evidence_state"
KEY_COACH_PROMPTS = "coach_prompts"         # list of active prompt dicts
KEY_OVERRIDES = "user_overrides"            # list of override dicts
KEY_PROMPT_COUNTER = "prompt_counter"       # monotonic int for unique prompt IDs
KEY_LAST_PROMPT_TS = "last_prompt_ts"       # dict[str, float]: rule_id -> last emitted epoch seconds
KEY_AUTO_TRANSITION = "auto_transition"     # bool (user can toggle)
KEY_CALIBRATION_DONE = "calibration_done"   # bool: calibration passed at least once
KEY_STREAM_START_TS = "stream_start_ts"     # epoch float: when feed started

# Hand tracking
KEY_PREV_HANDS = "prev_hands"              # list[dict]: previous frame hand results
KEY_HELD_TOOLS = "held_tools"              # set[str]: tools currently near a hand
KEY_TECHNIQUE = "technique_summary"        # dict: latest grip/smoothness/bimanual data
KEY_BIMANUAL_HISTORY = "bimanual_history"  # deque[float]: recent inter-hand distances

# WebRTC
KEY_WEBRTC_QUEUE = "webrtc_result_queue"   # queue.Queue for thread → main sync
KEY_WEBRTC_ACTIVE = "webrtc_active"        # bool: WebRTC streamer is running
KEY_WEBRTC_ENABLED = "webrtc_enabled"      # bool: user toggle for WebRTC mode
KEY_PERF_SAMPLES = "perf_samples"          # list[dict]: rolling perf telemetry samples
KEY_LAST_DISPLAY_TS = "last_display_ts"    # float epoch: latest UI render timestamp
KEY_TIP_HISTORY = "tip_history"            # list[tuple]: index-tip positions (jerk B)
KEY_WRIST_PATH = "wrist_path_history"      # list[tuple]: wrist positions (economy C)
KEY_JERK_DATA = "jerk_smoothness_data"     # dict: latest jerk smoothness result
KEY_ECONOMY_DATA = "motion_economy_data"   # dict: latest economy of motion result

# Intra-op phases
PHASES = ["incision", "suturing", "irrigation", "closing"]
PHASE_LABELS = {
    "incision": "Incision Technique",
    "suturing": "Suturing Practice",
    "irrigation": "Irrigation Drill",
    "closing": "Wound Closing",
}
