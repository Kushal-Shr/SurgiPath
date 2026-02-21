"""
Shared constants for the Panacea app.

HOW THIS SCRIPT WORKS

This file has no logic—only values. It is the single place to change paths,
default config numbers, and the string keys used for Streamlit session state.

  WHY USE THIS FILE:
  - Changing the model path or recipe path: edit here once instead of
    searching through app.py.
  - Adding a new session state variable: define KEY_XXX here and use KEY_XXX
    in app.py so you never typo the key name and can find all usages easily.
  - Tuning defaults (confidence, frame skip, stable seconds): change the
    *_DEFAULT / FEED_LOOP_FRAMES / etc. here.

  SECTIONS:
  - Paths: where the YOLO weights, recipe JSON, and event log live.
  - Config defaults: confidence threshold, inference size, frame skip,
    smoothing window size, how many frames to grab per run, stable_seconds.
  - Session state keys: the names we use in st.session_state (nav, smoother,
    alerts, last detections/counts, frame index, config sliders). Mode and
    phase are managed in src.state and use their own keys there.
  - PHASES: list of intra-op phase names; order matches the dropdown in app.
"""

# Paths (relative to project root)
MODEL_PATH = "models/best.pt"
RECIPE_PATH = "recipes/trauma_room.json"
EVENTS_LOG_PATH = "logs/events.jsonl"

# Config defaults
CONF_MIN_DEFAULT = 0.45
IMGSZ_DEFAULT = 640
FRAME_SKIP_DEFAULT = 3
SMOOTH_WINDOW_SIZE = 20
FEED_LOOP_FRAMES = 60
STABLE_SECONDS_DEFAULT = 2.0

# Session state keys (use these in app.py instead of raw strings)
# Mode and phase are managed by src.state (get_mode, set_phase, etc.)
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

# Intra-op phases (order matches dropdown)
PHASES = ["incision", "suturing", "irrigation", "closing"]
