"""
MedLab Coach AI backend package.

- constants: paths, config defaults, session state keys, gating thresholds
- detector: YOLO model load (cached), infer_tools, count_tools, draw_detections
- evidence: EvidenceState — per-tool confidence/stability tracker, enhanced calibration
- hands: MediaPipe Hands — detection, jitter, grip angle, tool proximity, drawing
- state: app mode (PRE_OP / INTRA_OP / POST_OP) and phase
- rules: debounced intra-op rule evaluation with evidence + hand-context rules
- logger: log_event, read_events for logs/events.jsonl
- utils: load_recipe, ToolPresenceSmoother
- scoring: override-aware rubric scoring engine, export helpers
"""
