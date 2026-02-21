"""
Panacea backend package.

================================================================================
WHAT EACH MODULE DOES (for studying)
================================================================================

- detector: YOLO model load (cached), infer_tools(), count_tools(), draw_detections()
- state: app mode (PRE_OP / INTRA_OP / POST_OP) and phase; get_mode, set_mode, get_phase, set_phase
- rules: debounced intra-op rule evaluation; evaluate_rules() returns list of alerts
- logger: log_event(), read_events() for logs/events.jsonl
- utils: load_recipe(), ToolPresenceSmoother (sliding-window tool presence)
- constants: paths, config defaults, session state key names, PHASES list

Read the top-of-file docstring in each module for a "HOW THIS SCRIPT WORKS" section.
"""
