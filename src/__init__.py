"""
MedLab Coach AI backend package.

- constants: paths, config defaults, session state key names, PHASES
- detector: YOLO model load (cached), infer_tools, count_tools, draw_detections
- state: app mode (PRE_OP / INTRA_OP / POST_OP) and phase
- rules: debounced intra-op rule evaluation
- logger: log_event, read_events for logs/events.jsonl
- utils: load_recipe, ToolPresenceSmoother
- scoring: rubric scoring engine, export helpers (JSON report, CSV transcript)
"""
