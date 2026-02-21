"""
Event logging: append events to logs/events.jsonl and read them back.

================================================================================
HOW THIS SCRIPT WORKS (for studying)
================================================================================

All important actions in the app (mode change, phase change, checklist pass,
alert fired) are written as one JSON object per line in logs/events.jsonl.
Post-Op reads this file to show analytics (total alerts, alerts by phase,
checklist pass/fail, timeline). You can also open the file in a text editor
or script to debug or analyse runs.

  EVENT RECORD SHAPE:
  Every record has: timestamp (ISO UTC), type, mode, phase, plus whatever
  extra fields were in the payload (e.g. status, readiness_pct, rule_id, message).
  - type: "STATE_CHANGE" | "PHASE_CHANGE" | "CHECKLIST_STATUS" | "ALERT"
  - mode: "PRE_OP" | "INTRA_OP" | "POST_OP" (or None)
  - phase: e.g. "incision" (or None)

  FUNCTIONS:
  - log_event(event_type, payload, mode=None, phase=None): appends one line to
    logs/events.jsonl. Creates the logs/ directory if needed. payload is merged
    into the record (so payload {"status": "PASS"} becomes a field in the JSON).
  - read_events(limit=None): reads the file, parses each line as JSON, returns
    list of dicts. If limit is set, returns only the last limit lines (useful
    for "last N events" in UI).

  FILE LOCATION:
  Path is logs/events.jsonl relative to the current working directory when the
  app runs (usually the project root). _log_path() and _ensure_log_dir() handle
  creating the directory and resolving the path.
"""
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "events.jsonl"


def _log_path() -> Path:
    p = Path(LOG_FILE)
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


def _ensure_log_dir() -> None:
    p = _log_path().parent
    p.mkdir(parents=True, exist_ok=True)


def log_event(
    event_type: str,
    payload: dict[str, Any],
    mode: str | None = None,
    phase: str | None = None,
) -> None:
    """
    Append one event to logs/events.jsonl.
    payload is merged into the record (timestamp, type, mode, phase added).
    """
    _ensure_log_dir()
    record = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "type": event_type,
        "mode": mode,
        "phase": phase,
        **payload,
    }
    with open(_log_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_events(limit: int | None = None) -> list[dict]:
    """
    Read events from logs/events.jsonl.
    If limit is set, return only the last limit lines.
    """
    p = _log_path()
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").strip().split("\n")
    lines = [ln for ln in lines if ln]
    if limit is not None:
        lines = lines[-limit:]
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out
