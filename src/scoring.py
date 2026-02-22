"""
Rubric scoring engine for MedLab Coach AI.

Dimensions (0-100 each):
  1. Setup Completeness  — all required tools found?
  2. Technique Safety     — fewer coaching alerts = higher; overrides reduce penalty
  3. Efficiency           — actual vs expected time
  4. Streak Bonus         — longest error-free streak

Override-aware: if a student denied a low/medium-confidence prompt, that alert
does NOT count as a violation (or counts at reduced weight).
"""

import json
import csv
import io
from datetime import datetime
from typing import Any

from src.logger import read_events


def _count_alerts_with_overrides(events: list[dict]) -> dict[str, Any]:
    """Count effective alerts, accounting for USER_OVERRIDE deny events."""
    prompts: dict[str, dict] = {}
    overrides: dict[str, dict] = {}

    for e in events:
        etype = e.get("type", "")
        if etype == "COACH_PROMPT":
            pid = e.get("prompt_id", "")
            if pid:
                prompts[pid] = e
        elif etype == "USER_OVERRIDE":
            pid = e.get("prompt_id", "")
            if pid:
                overrides[pid] = e

    # Also count legacy ALERT events (no prompt_id)
    legacy_alerts = [e for e in events if e.get("type") == "ALERT"]

    total = len(prompts) + len(legacy_alerts)
    confirmed = 0
    denied = 0
    effective = len(legacy_alerts)  # legacy alerts always count

    for pid, prompt in prompts.items():
        override = overrides.get(pid)
        tier = prompt.get("risk_tier", "high")
        if override and override.get("decision") == "deny":
            denied += 1
            if tier == "high":
                effective += 0.5  # high-confidence deny still counts half
            # low/medium deny → 0 penalty
        else:
            if override and override.get("decision") == "confirm":
                confirmed += 1
            effective += 1  # unresolved or confirmed = full penalty

    tiers = {"high": 0, "medium": 0, "low": 0}
    for p in prompts.values():
        t = p.get("risk_tier", "high")
        tiers[t] = tiers.get(t, 0) + 1

    return {
        "total": total,
        "effective": effective,
        "confirmed": confirmed,
        "denied": denied,
        "tiers": tiers,
        "override_rate": denied / max(1, total),
    }


def compute_rubric(
    recipe: dict,
    best_streak: float = 0,
    hand_stability_samples: list[float] | None = None,
    technique_summary: dict | None = None,
) -> dict[str, Any]:
    events = read_events()

    # --- 1. Setup Completeness ---
    checklist_ev = [e for e in events if e.get("type") == "CHECKLIST_STATUS"]
    if any(e.get("status") == "PASS" for e in checklist_ev):
        pct = max((e.get("readiness_pct", 100) for e in checklist_ev if e.get("status") == "PASS"), default=100)
        setup_score = min(100, int(pct))
    elif checklist_ev:
        setup_score = max((e.get("readiness_pct", 0) for e in checklist_ev), default=0)
    else:
        setup_score = 0

    # --- 2. Technique Safety (override-aware) ---
    alert_info = _count_alerts_with_overrides(events)
    n_effective = alert_info["effective"]
    if n_effective == 0:
        alert_penalty = 0
    else:
        alert_penalty = min(100, int(n_effective * 12))
    safety_from_alerts = max(0, 100 - alert_penalty)

    hand_score = 100
    if hand_stability_samples and len(hand_stability_samples) > 0:
        avg_jitter = sum(hand_stability_samples) / len(hand_stability_samples)
        hand_score = max(0, int(100 - avg_jitter * 500))

    # Grip quality bonus/penalty from technique summary
    grip_score = 100
    if technique_summary and technique_summary.get("grips"):
        grips = technique_summary["grips"]
        good_grips = sum(1 for g in grips if g.get("grip_type") in ("pencil_grip", "precision_grip"))
        total_grips = len(grips)
        if total_grips > 0:
            grip_score = int(100 * good_grips / total_grips)

    technique_safety = int(0.5 * safety_from_alerts + 0.3 * hand_score + 0.2 * grip_score)

    # --- 3. Efficiency ---
    expected = float(recipe.get("expected_time_seconds", 300))
    state_changes = [e for e in events if e.get("type") == "STATE_CHANGE"]
    start_ts = None
    end_ts = None
    for e in state_changes:
        if e.get("to") == "INTRA_OP" or e.get("from") == "PRE_OP":
            start_ts = e.get("timestamp")
        if e.get("to") == "POST_OP" or e.get("from") == "INTRA_OP":
            end_ts = e.get("timestamp")
    if start_ts and end_ts:
        try:
            t0 = datetime.fromisoformat(start_ts)
            t1 = datetime.fromisoformat(end_ts)
            actual = (t1 - t0).total_seconds()
        except Exception:
            actual = expected
    else:
        actual = expected
    if actual <= 0:
        actual = expected
    ratio = expected / actual
    efficiency = min(100, max(0, int(ratio * 100)))

    # --- 4. Streak Bonus ---
    streak_bonus = min(100, int(best_streak * 3.33))

    # --- Weighted total ---
    total = int(
        0.25 * setup_score
        + 0.30 * technique_safety
        + 0.25 * efficiency
        + 0.20 * streak_bonus
    )

    return {
        "setup_completeness": setup_score,
        "technique_safety": technique_safety,
        "efficiency": efficiency,
        "streak_bonus": streak_bonus,
        "total_weighted": total,
        "details": {
            "n_alerts": alert_info["total"],
            "n_effective_alerts": alert_info["effective"],
            "confirmed": alert_info["confirmed"],
            "denied": alert_info["denied"],
            "override_rate": round(alert_info["override_rate"], 3),
            "tiers": alert_info["tiers"],
            "actual_seconds": actual,
            "expected_seconds": expected,
            "best_streak_seconds": best_streak,
            "hand_samples": len(hand_stability_samples or []),
            "grip_score": grip_score,
            "smoothness": technique_summary.get("smoothness", "unknown") if technique_summary else "unknown",
        },
    }


def rubric_to_json(rubric: dict) -> str:
    return json.dumps(rubric, indent=2, ensure_ascii=False)


def events_to_csv() -> str:
    events = read_events()
    if not events:
        return ""
    all_keys = set()
    for e in events:
        all_keys.update(e.keys())
    all_keys = sorted(all_keys)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=all_keys, extrasaction="ignore")
    writer.writeheader()
    for e in events:
        writer.writerow(e)
    return buf.getvalue()
