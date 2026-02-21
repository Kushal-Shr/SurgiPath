"""
Rubric scoring engine for MedLab Coach AI.

Reads the JSONL event log and computes four pedagogical rubric dimensions:
  1. Setup Completeness  — Did the student find all required tools?
  2. Technique Safety     — How few coaching alerts fired? Includes hand-stability if available.
  3. Efficiency           — Actual time vs expected time for the lab session.
  4. Streak Bonus         — Longest error-free streak (gamification reward).

Each dimension is scored 0-100. A weighted total is also returned.
"""
import json
import csv
import io
from datetime import datetime
from typing import Any

from src.logger import read_events


def compute_rubric(
    recipe: dict,
    best_streak: float = 0,
    hand_stability_samples: list[float] | None = None,
) -> dict[str, Any]:
    """
    Compute the full rubric dict from event log + recipe.

    Returns: {
      "setup_completeness": 0-100,
      "technique_safety": 0-100,
      "efficiency": 0-100,
      "streak_bonus": 0-100,
      "total_weighted": 0-100,
      "details": {...},
    }
    """
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

    # --- 2. Technique Safety ---
    alerts_ev = [e for e in events if e.get("type") == "ALERT"]
    n_alerts = len(alerts_ev)
    if n_alerts == 0:
        alert_penalty = 0
    else:
        alert_penalty = min(100, n_alerts * 12)
    safety_from_alerts = max(0, 100 - alert_penalty)

    hand_score = 100
    if hand_stability_samples and len(hand_stability_samples) > 0:
        avg_jitter = sum(hand_stability_samples) / len(hand_stability_samples)
        hand_score = max(0, int(100 - avg_jitter * 500))

    technique_safety = int(0.6 * safety_from_alerts + 0.4 * hand_score)

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
            "n_alerts": n_alerts,
            "actual_seconds": actual,
            "expected_seconds": expected,
            "best_streak_seconds": best_streak,
            "hand_samples": len(hand_stability_samples or []),
        },
    }


def rubric_to_json(rubric: dict) -> str:
    """Serialize rubric dict to pretty JSON for download."""
    return json.dumps(rubric, indent=2, ensure_ascii=False)


def events_to_csv() -> str:
    """Convert events.jsonl to a CSV string for download."""
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
