"""
Debounced intra-op rule engine: evaluate phase-specific rules, fire alerts after hold time.

================================================================================
HOW THIS SCRIPT WORKS (for studying)
================================================================================

The recipe defines intraop_rules (phase, if_present, if_missing, hold_seconds, message).
A rule fires only after its condition is true continuously for hold_seconds, and
only once until the condition goes false again (one-shot per condition-true period).
  RULE SHAPE: id, phase, if_present, if_missing, hold_seconds, message.
  EVALUATION: Filter by phase; build present tools from tool_counts; per rule add
  dt_seconds to timer when condition met; when timer >= hold_seconds fire alert.


  SESSION STATE:
  We store two dicts in the session_state dict passed in: RULE_TIMERS_KEY
  (rule_id -> seconds_true) and RULE_TRIGGERED_KEY (rule_id -> True when fired).
  app.py passes st.session_state so these persist across reruns.

  NORMALIZATION:
  Tool names are normalized with _norm() (lowercase, spaces -> underscores) so
  recipe tools like "needle_holder" match detection counts after normalization.
"""
from typing import Any

RULE_TIMERS_KEY = "rules_seconds_true"
RULE_TRIGGERED_KEY = "rules_triggered"


def _norm(name: str) -> str:
    """Normalize tool name: lower case, spaces → underscores (match recipe)."""
    return (name or "").strip().lower().replace(" ", "_")


def _tools_present(counts: dict[str, int], min_count: int = 1) -> set[str]:
    """Set of normalized tool names with count >= min_count."""
    return {_norm(k) for k, v in counts.items() if v >= min_count}


def _condition_met(rule: dict, present: set[str]) -> bool:
    """
    True iff:
    - every tool in if_present is in present, and
    - no tool in if_missing is in present.
    """
    if_present = [_norm(t) for t in rule.get("if_present", [])]
    if_missing = [_norm(t) for t in rule.get("if_missing", [])]
    if any(p not in present for p in if_present):
        return False
    if any(m in present for m in if_missing):
        return False
    return True


def evaluate_rules(
    phase: str,
    tool_counts: dict[str, int],
    rules: list[dict],
    dt_seconds: float,
    session_state: dict,
) -> list[dict]:
    """
    Evaluate rules for the current phase with debouncing.

    - phase: current phase (e.g. "incision").
    - tool_counts: dict tool_name → count (normalized names recommended).
    - rules: list of rule dicts (id, phase, if_present, if_missing, hold_seconds, message).
    - dt_seconds: time elapsed since last evaluation (for debounce timer).
    - session_state: Streamlit session_state (or any mutable dict) for timers.

    Returns list of newly triggered alerts: [{"rule_id", "message", "phase"}, ...].
    """
    if not rules:
        return []

    phase_norm = _norm(phase)
    present = _tools_present(tool_counts)

    timers = session_state.get(RULE_TIMERS_KEY, {})
    triggered = session_state.get(RULE_TRIGGERED_KEY, {})
    if RULE_TIMERS_KEY not in session_state:
        session_state[RULE_TIMERS_KEY] = timers
    if RULE_TRIGGERED_KEY not in session_state:
        session_state[RULE_TRIGGERED_KEY] = triggered

    alerts = []
    for rule in rules:
        if _norm(rule.get("phase", "")) != phase_norm:
            continue
        rule_id = rule.get("id", "")
        hold = float(rule.get("hold_seconds", 1.0))
        met = _condition_met(rule, present)

        if met:
            timers[rule_id] = timers.get(rule_id, 0) + dt_seconds
            if rule_id in triggered:
                continue  # already fired this cycle
            if timers[rule_id] >= hold:
                alerts.append({
                    "rule_id": rule_id,
                    "message": rule.get("message", ""),
                    "phase": phase,
                })
                triggered[rule_id] = True
        else:
            timers[rule_id] = 0
            triggered.pop(rule_id, None)

    session_state[RULE_TIMERS_KEY] = timers
    session_state[RULE_TRIGGERED_KEY] = triggered
    return alerts
