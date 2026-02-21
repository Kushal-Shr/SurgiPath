"""
Helpers: load recipe JSON and smooth tool presence over a sliding window.

================================================================================
HOW THIS SCRIPT WORKS (for studying)
================================================================================

  load_recipe(path):
  - Reads a JSON file (e.g. recipes/trauma_room.json) and returns a dict.
  - The recipe defines preop_required (list of {tool, min_count}), params
    (conf_min, stable_seconds), and intraop_rules. Path can be relative to
    cwd or absolute. Used once at app startup (in app.py) to get the checklist
    and rules.

  ToolPresenceSmoother:
  - Problem: raw detection counts flicker frame to frame (tool appears then
    disappears). We don't want the checklist to flip between OK and MISSING
    every second.
  - Idea: keep a sliding window of the last N samples per tool. A tool is
    "present" if it was seen in at least one of those N samples. So brief
    dropouts don't immediately mark the tool missing.
  - update(counts, required_tools): called each time we have new detection
    counts. For each required tool we record whether count >= min_count in a
    deque of length window_size (old samples drop off).
  - is_present(tool): True if any of the last N samples had that tool present.
  - all_present(required_tools): True if every required tool is present (used
    to decide when the checklist can pass).
  - readiness_counts(required_tools): returns {tool: (num_samples_seen, is_present)}
    for building the Pre-Op table (e.g. "Detected (window)" column and "OK/MISSING").

  TOOL NAMES:
  Recipe and YOLO may use different spellings (e.g. "needle_holder" vs
  "Needle Holder"). Normalization (lowercase, spaces → underscores) is done
  in app.py and in src.rules when matching; this module expects counts keyed
  by the same names as in the recipe (app.py builds that mapping with
  counts_for_recipe()).
"""
import json
from pathlib import Path
from collections import deque
from typing import Any


def load_recipe(path: str | Path = "recipes/trauma_room.json") -> dict[str, Any]:
    """
    Load recipe JSON from path (relative to cwd or absolute).
    Raises if file missing or invalid JSON.
    """
    p = Path(path)
    if not p.is_absolute():
        p = Path.cwd() / p
    with open(p, encoding="utf-8") as f:
        return json.load(f)


class ToolPresenceSmoother:
    """
    Sliding window over recent samples. A tool is “present” if it appeared
    in at least one of the last N samples (reduces flicker).

    - update(counts, required_tools): record one sample (counts keyed by recipe tool name).
    - is_present(tool): True if tool was seen in any of last N samples.
    - all_present(required_tools): True if every required tool is present.
    - readiness_counts(required_tools): {tool: (num_samples_seen, is_present)} for display.
    """
    def __init__(self, window_size: int = 20):
        self.window_size = max(1, window_size)
        self._history: dict[str, deque[bool]] = {}

    def update(self, counts: dict[str, int], required_tools: list[dict]) -> None:
        """
        Record one sample. counts: tool name → current count.
        required_tools: list of {"tool": name, "min_count": n}.
        """
        for req in required_tools:
            tool = req.get("tool", "")
            min_count = req.get("min_count", 1)
            if not tool:
                continue
            present = counts.get(tool, 0) >= min_count
            if tool not in self._history:
                self._history[tool] = deque(maxlen=self.window_size)
            self._history[tool].append(present)

    def is_present(self, tool: str) -> bool:
        """True if tool appeared in at least one of the last N samples."""
        if tool not in self._history or len(self._history[tool]) == 0:
            return False
        return any(self._history[tool])

    def all_present(self, required_tools: list[dict]) -> bool:
        """True if every required tool is present (smoothed)."""
        for req in required_tools:
            tool = req.get("tool", "")
            if tool and not self.is_present(tool):
                return False
        return True

    def readiness_counts(
        self, required_tools: list[dict]
    ) -> dict[str, tuple[int, bool]]:
        """
        For each required tool: (number of samples in window where seen, is_present).
        Use for display (e.g. checklist table).
        """
        out = {}
        for req in required_tools:
            tool = req.get("tool", "")
            if not tool:
                continue
            q = self._history.get(tool, deque())
            detected = sum(1 for x in q if x)
            out[tool] = (detected, self.is_present(tool))
        return out
