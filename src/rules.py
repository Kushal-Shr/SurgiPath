"""
Debounced intra-op rule engine with evidence-gated Coach Prompts.

Supports two rule types:

1. **Presence rules** (original): `if_present` / `if_missing` — checks whether tools
   are visible in the frame. Fires when condition holds for >= `hold_seconds`.

2. **Hand-context rules** (new): `if_holding` / `if_not_holding` — checks whether a
   hand is near/overlapping a tool's bounding box (from `src/hands.get_held_tools()`).
   Fires when condition holds for >= `hold_seconds`. This is a step beyond
   "tool visible" toward "tool being used."

Each fired alert carries a risk_tier (high/medium/low), the tool's avg_conf,
seen_ratio, and last_seen_ts so the UI can render "Coach Prompt Cards" with
evidence and optional user-override buttons.
"""

import time
from typing import Any

RULE_TIMERS_KEY = "rules_seconds_true"
RULE_TRIGGERED_KEY = "rules_triggered"


def _norm(name: str) -> str:
    return (name or "").strip().lower().replace(" ", "_")


def _tools_present(counts: dict[str, int], min_count: int = 1) -> set[str]:
    return {_norm(k) for k, v in counts.items() if v >= min_count}


def _condition_met(rule: dict, present: set[str]) -> bool:
    if_present = [_norm(t) for t in rule.get("if_present", [])]
    if_missing = [_norm(t) for t in rule.get("if_missing", [])]
    if any(p not in present for p in if_present):
        return False
    if any(m in present for m in if_missing):
        return False
    return True


def _hand_condition_met(rule: dict, held_tools: set[str]) -> bool:
    """Check hand-context conditions: if_holding / if_not_holding."""
    if_holding = [_norm(t) for t in rule.get("if_holding", [])]
    if_not_holding = [_norm(t) for t in rule.get("if_not_holding", [])]
    if not if_holding and not if_not_holding:
        return False
    if any(t not in held_tools for t in if_holding):
        return False
    if any(t in held_tools for t in if_not_holding):
        return False
    return True


def _is_hand_rule(rule: dict) -> bool:
    return bool(rule.get("if_holding") or rule.get("if_not_holding"))


def _enrich_alert(
    alert: dict[str, Any],
    rule: dict,
    evidence: Any | None,
) -> None:
    """Add evidence metadata (risk_tier, avg_conf, etc.) to an alert dict."""
    if evidence is not None:
        relevant_tools = (
            rule.get("if_present", []) + rule.get("if_missing", [])
            + rule.get("if_holding", []) + rule.get("if_not_holding", [])
        )
        ev_tools: dict[str, dict] = {}
        confs: list[float] = []
        ratios: list[float] = []
        for t in relevant_tools:
            ts = evidence.tool_state(t)
            ev_tools[_norm(t)] = ts
            confs.append(ts["avg_conf"])
            ratios.append(ts["seen_ratio"])
        avg_c = sum(confs) / max(1, len(confs))
        avg_r = sum(ratios) / max(1, len(ratios))
        tier = evidence.risk_tier(avg_c, avg_r)
        alert["risk_tier"] = tier
        alert["avg_conf"] = round(avg_c, 3)
        alert["seen_ratio"] = round(avg_r, 3)
        last_ts = max(
            (ts.get("last_seen_ts", 0) for ts in ev_tools.values()),
            default=0,
        )
        alert["last_seen_ts"] = last_ts
        alert["evidence_tools"] = ev_tools
    else:
        alert["risk_tier"] = "high"
        alert["avg_conf"] = 1.0
        alert["seen_ratio"] = 1.0
        alert["last_seen_ts"] = time.time()
        alert["evidence_tools"] = {}


def evaluate_rules(
    phase: str,
    tool_counts: dict[str, int],
    rules: list[dict],
    dt_seconds: float,
    session_state: dict,
    evidence: Any | None = None,
    held_tools: set[str] | None = None,
) -> list[dict]:
    """
    Evaluate rules with debouncing + optional evidence gating.

    Args:
        held_tools: set of normalized tool names currently held (hand near tool bbox).
                    Required for if_holding / if_not_holding rules.

    Returns list of newly triggered alerts:
      [{rule_id, message, phase, risk_tier, avg_conf, seen_ratio, last_seen_ts,
        evidence_tools: {tool: tool_state_dict}}, ...]
    """
    if not rules:
        return []

    phase_norm = _norm(phase)
    present = _tools_present(tool_counts)
    if held_tools is None:
        held_tools = set()

    timers: dict[str, float] = session_state.get(RULE_TIMERS_KEY, {})
    triggered: dict[str, bool] = session_state.get(RULE_TRIGGERED_KEY, {})
    if RULE_TIMERS_KEY not in session_state:
        session_state[RULE_TIMERS_KEY] = timers
    if RULE_TRIGGERED_KEY not in session_state:
        session_state[RULE_TRIGGERED_KEY] = triggered

    alerts: list[dict] = []
    for rule in rules:
        if _norm(rule.get("phase", "")) != phase_norm:
            continue
        rule_id = rule.get("id", "")
        hold = float(rule.get("hold_seconds", 1.0))

        if _is_hand_rule(rule):
            met = _hand_condition_met(rule, held_tools)
        else:
            met = _condition_met(rule, present)

        if met:
            timers[rule_id] = timers.get(rule_id, 0) + dt_seconds
            if rule_id in triggered:
                continue
            if timers[rule_id] >= hold:
                alert: dict[str, Any] = {
                    "rule_id": rule_id,
                    "message": rule.get("message", ""),
                    "phase": phase,
                }
                _enrich_alert(alert, rule, evidence)
                alerts.append(alert)
                triggered[rule_id] = True
        else:
            timers[rule_id] = 0
            triggered.pop(rule_id, None)

    session_state[RULE_TIMERS_KEY] = timers
    session_state[RULE_TRIGGERED_KEY] = triggered
    return alerts
