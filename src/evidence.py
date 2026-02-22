"""
EvidenceState tracker — per-tool confidence, stability, and detection-rate tracking.

Feeds the Coach Prompt system with evidence so alerts are only fired when
detection evidence is stable, and the UI can show confidence tiers + last-seen
timestamps to the student.

Usage (called every frame in the video loop):
    evidence = st.session_state[KEY_EVIDENCE]
    evidence.update(detections, frame_bgr)  # list[dict], np.ndarray | None
    tool_state = evidence.tool_state("scalpel")
    # -> {"last_seen_ts": 1708..., "stable_present": True, "stable_missing": False,
    #     "avg_conf": 0.87, "seen_ratio": 0.9}
    evidence.detection_rate  # float 0-1 over the last N frames
    evidence.brightness      # float 0-255
    evidence.blur_score      # float (higher = sharper)
"""

import time
from collections import deque
from typing import Any

import numpy as np


class EvidenceState:
    """Sliding-window evidence tracker for all detected tools."""

    def __init__(
        self,
        window: int = 20,
        missing_grace_sec: float = 3.0,
    ):
        self.window = max(1, window)
        self.missing_grace_sec = missing_grace_sec

        # per-tool sliding windows: tool -> deque of (present: bool, conf: float)
        self._history: dict[str, deque[tuple[bool, float]]] = {}
        # per-tool last-seen timestamp (epoch seconds)
        self._last_seen: dict[str, float] = {}

        # global: did at least 1 detection exist in frame? (for detection_rate)
        self._det_frames: deque[bool] = deque(maxlen=self.window)

        # calibration metrics (updated per frame)
        self.brightness: float = 128.0
        self.blur_score: float = 100.0

        # enhanced calibration: bbox coverage + detection rate history for drop detect
        self._bbox_union: deque[tuple[float, float, float, float]] = deque(maxlen=self.window)
        self._recent_det_rates: deque[float] = deque(maxlen=10)
        self._frame_size: tuple[int, int] = (640, 480)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections: list[dict],
        frame_bgr: np.ndarray | None = None,
        known_tools: list[str] | None = None,
    ) -> None:
        """Call once per processed frame with raw YOLO detections."""
        now = time.time()

        # Build per-tool max confidence from this frame
        frame_confs: dict[str, float] = {}
        for d in detections:
            name = self._norm(d.get("name", ""))
            conf = float(d.get("conf", 0))
            if name:
                frame_confs[name] = max(frame_confs.get(name, 0), conf)

        # Record detection_rate sample
        self._det_frames.append(len(frame_confs) > 0)

        # Update per-tool history
        all_tools = set(frame_confs.keys())
        if known_tools:
            all_tools |= {self._norm(t) for t in known_tools}
        all_tools |= set(self._history.keys())

        for tool in all_tools:
            if tool not in self._history:
                self._history[tool] = deque(maxlen=self.window)
            present = tool in frame_confs
            conf = frame_confs.get(tool, 0.0)
            self._history[tool].append((present, conf))
            if present:
                self._last_seen[tool] = now

        # Track bounding box union for workspace coverage
        if detections:
            xs = [d["xyxy"][0] for d in detections if "xyxy" in d] + \
                 [d["xyxy"][2] for d in detections if "xyxy" in d]
            ys = [d["xyxy"][1] for d in detections if "xyxy" in d] + \
                 [d["xyxy"][3] for d in detections if "xyxy" in d]
            if xs and ys:
                self._bbox_union.append((min(xs), min(ys), max(xs), max(ys)))

        # Calibration metrics
        if frame_bgr is not None:
            self._frame_size = (frame_bgr.shape[1], frame_bgr.shape[0])
            try:
                gray = np.mean(frame_bgr)
                self.brightness = float(gray)
                gray_img = frame_bgr if len(frame_bgr.shape) == 2 else np.mean(frame_bgr, axis=2).astype(np.uint8)
                import cv2
                self.blur_score = float(cv2.Laplacian(gray_img, cv2.CV_64F).var())
            except Exception:
                pass

        # Track detection rate history for obstruction detection
        if len(self._det_frames) >= 3:
            self._recent_det_rates.append(self.detection_rate)

    def tool_state(self, tool: str) -> dict[str, Any]:
        """Return evidence summary for a single tool."""
        tool = self._norm(tool)
        q = self._history.get(tool, deque())
        n = len(q)
        if n == 0:
            return {
                "last_seen_ts": 0,
                "stable_present": False,
                "stable_missing": True,
                "avg_conf": 0.0,
                "seen_ratio": 0.0,
            }
        seen_count = sum(1 for p, _ in q if p)
        seen_ratio = seen_count / n
        avg_conf = (
            sum(c for p, c in q if p) / max(1, seen_count)
        )
        last_seen = self._last_seen.get(tool, 0)
        now = time.time()
        stable_present = seen_ratio >= 0.5 and avg_conf > 0
        time_since = now - last_seen if last_seen else float("inf")
        stable_missing = (
            seen_ratio < 0.25 and time_since >= self.missing_grace_sec
        )
        return {
            "last_seen_ts": last_seen,
            "stable_present": stable_present,
            "stable_missing": stable_missing,
            "avg_conf": round(avg_conf, 3),
            "seen_ratio": round(seen_ratio, 3),
        }

    @property
    def detection_rate(self) -> float:
        """Fraction of recent frames that had at least one detection."""
        if not self._det_frames:
            return 0.0
        return sum(self._det_frames) / len(self._det_frames)

    def risk_tier(self, avg_conf: float, seen_ratio: float) -> str:
        """Classify evidence into risk tiers: high / medium / low."""
        if avg_conf >= 0.75 and seen_ratio >= 0.6:
            return "high"
        if avg_conf >= 0.5 or seen_ratio >= 0.4:
            return "medium"
        return "low"

    @property
    def workspace_coverage(self) -> float:
        """Fraction of the frame area covered by detected tool bounding boxes (union)."""
        if not self._bbox_union:
            return 0.0
        fw, fh = self._frame_size
        frame_area = fw * fh
        if frame_area == 0:
            return 0.0
        all_x1 = min(b[0] for b in self._bbox_union)
        all_y1 = min(b[1] for b in self._bbox_union)
        all_x2 = max(b[2] for b in self._bbox_union)
        all_y2 = max(b[3] for b in self._bbox_union)
        union_area = max(0, all_x2 - all_x1) * max(0, all_y2 - all_y1)
        return union_area / frame_area

    @property
    def detection_centroid_spread(self) -> float:
        """Fraction of frame area spanned by the centroid of recent detections.
        Low value → detections clustered in a small region (camera may be misaligned)."""
        if len(self._bbox_union) < 2:
            return 0.0
        fw, fh = self._frame_size
        if fw == 0 or fh == 0:
            return 0.0
        cxs = [(b[0] + b[2]) / 2 for b in self._bbox_union]
        cys = [(b[1] + b[3]) / 2 for b in self._bbox_union]
        cx_range = (max(cxs) - min(cxs)) / fw
        cy_range = (max(cys) - min(cys)) / fh
        return cx_range * cy_range

    def _detect_obstruction(self) -> bool:
        """Return True if detection_rate dropped sharply while brightness is OK,
        suggesting something is blocking the camera."""
        if len(self._recent_det_rates) < 4:
            return False
        rates = list(self._recent_det_rates)
        early = sum(rates[:len(rates) // 2]) / max(1, len(rates) // 2)
        late = sum(rates[len(rates) // 2:]) / max(1, len(rates) - len(rates) // 2)
        brightness_ok = 50 <= self.brightness <= 230
        return brightness_ok and early > 0.4 and late < 0.1

    def calibration_status(self) -> dict[str, Any]:
        """Enhanced calibration check: brightness, blur, detection_rate,
        workspace coverage, detection clustering, and obstruction detection."""
        issues: list[str] = []

        if self.brightness < 50:
            issues.append("Too dark — increase lighting")
        elif self.brightness > 230:
            issues.append("Over-exposed — reduce glare")

        if self.blur_score < 30:
            issues.append("Image is blurry — steady the camera or move closer")

        if len(self._det_frames) >= 3 and self.detection_rate < 0.1:
            issues.append("No tools detected — check camera angle and distance")

        coverage = self.workspace_coverage
        if len(self._bbox_union) >= 3:
            if coverage < 0.05:
                issues.append("Tools cover very little of the frame — move camera closer")
            elif coverage > 0.85:
                issues.append("Tools fill most of the frame — move camera further away")

        spread = self.detection_centroid_spread
        if len(self._bbox_union) >= 5 and spread < 0.02 and coverage > 0:
            issues.append("Detections clustered in one area — reposition camera to center workspace")

        if self._detect_obstruction():
            issues.append("Detection rate dropped suddenly — something may be blocking the camera")

        ok = len(issues) == 0
        return {
            "ok": ok,
            "brightness": round(self.brightness, 1),
            "blur_score": round(self.blur_score, 1),
            "detection_rate": round(self.detection_rate, 3),
            "workspace_coverage": round(coverage, 3),
            "centroid_spread": round(spread, 3),
            "obstruction_warning": self._detect_obstruction(),
            "issues": issues,
        }

    @staticmethod
    def _norm(name: str) -> str:
        return (name or "").strip().lower().replace(" ", "_")
