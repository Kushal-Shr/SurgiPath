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

# New keys for EdTech pivot
KEY_VIDEO_SOURCE = "video_source"          # "Live Webcam" | "Upload Video" | "Sample Video"
KEY_UPLOADED_FILE = "uploaded_file_bytes"
KEY_STREAK_SECONDS = "streak_seconds"      # consecutive seconds without alerts
KEY_STREAK_BEST = "streak_best"            # best streak this session
KEY_SESSION_START = "session_start_ts"     # ISO timestamp when PRACTICE started
KEY_HAND_STABILITY = "hand_stability_samples"  # list of floats for MediaPipe hand jitter
KEY_DEMO_MODE = "demo_mode"                    # bool: generate fake detections without camera/model
KEY_DEMO_TICK = "demo_tick"                    # int: counts up each fragment run to progressively reveal tools

# Intra-op phases (educational wording used in UI, internal keys stay the same)
PHASES = ["incision", "suturing", "irrigation", "closing"]
PHASE_LABELS = {
    "incision": "Incision Technique",
    "suturing": "Suturing Practice",
    "irrigation": "Irrigation Drill",
    "closing": "Wound Closing",
}
