"""
YOLO (Ultralytics) tool detector: load model, run inference, count tools, draw boxes.

HOW THIS SCRIPT WORKS (for studying)

This module wraps the Ultralytics YOLO API so the rest of the app gets a
simple interface: pass a BGR image and get back a list of detections (tool
name, confidence, bounding box). The model is loaded once and cached so we
don't reload weights on every frame.

  DATA FLOW:
  1. app.py (or any caller) gets the model with get_model(path). First call
     loads the .pt file; later calls return the same object (cached).
  2. For each frame (or every Nth frame), caller runs infer_tools(frame, conf, imgsz, model).
     YOLO returns boxes; we convert them to a list of dicts: name, conf, xyxy.
  3. count_tools(detections) turns that list into { "tool_name": count, ... }.
  4. draw_detections(frame, detections) draws rectangles and labels on the frame
     and returns the annotated image for display.

  DETECTION FORMAT:
  Each detection is a dict: {"name": str, "conf": float, "xyxy": [x1, y1, x2, y2]}.
  "name" comes from the model's class names (e.g. model.names[class_id]).
  "xyxy" is the box in pixel coordinates (left, top, right, bottom).

  CACHING:
  get_model() is decorated with @_cache. In Streamlit we use st.cache_resource
  so the model is loaded once per session. Without Streamlit we use
  functools.lru_cache(maxsize=1) so the same path returns the same model.

  WHERE IT'S USED:
  app.py calls get_model (via cached_model()), then infer_tools, count_tools,
  and draw_detections inside the webcam loop. Counts are normalized (lowercase,
  spaces to underscores) in app.py so they match recipe tool names.
"""
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Use Streamlit cache when running in Streamlit; otherwise lru_cache
try:
    import streamlit as st
    _cache = st.cache_resource
except Exception:
    from functools import lru_cache
    _cache = lru_cache(maxsize=1)


def _load_model(path: str) -> Any:
    from ultralytics import YOLO
    return YOLO(path)


@_cache
def get_model(path: str = "models/best.pt") -> Any:
    """
    Load YOLO model once (cached).
    Path can be relative to cwd or absolute.
    Raises FileNotFoundError if path does not exist.
    """
    p = Path(path)
    if not p.is_absolute():
        p = Path.cwd() / p
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}")
    return _load_model(str(p))


def infer_tools(
    frame_bgr: np.ndarray,
    conf: float = 0.5,
    imgsz: int = 640,
    model: Any = None,
    model_path: str = "models/best.pt",
) -> list[dict]:
    """
    Run YOLO inference on a BGR frame.

    Returns list of detections, each:
      {"name": str, "conf": float, "xyxy": [x1, y1, x2, y2]}
    """
    if model is None:
        model = get_model(model_path)
    results = model.predict(frame_bgr, conf=conf, imgsz=imgsz, verbose=False)
    detections = []
    for r in results:
        if r.boxes is None:
            continue
        names = r.names or {}
        for box in r.boxes:
            cls_id = int(box.cls.item())
            name = names.get(cls_id, f"class_{cls_id}")
            detections.append({
                "name": name,
                "conf": float(box.conf.item()),
                "xyxy": [float(x) for x in box.xyxy[0].tolist()],
            })
    return detections


def count_tools(detections: list[dict]) -> dict[str, int]:
    """Return dict mapping class name → count of detections."""
    counts: dict[str, int] = {}
    for d in detections:
        name = d.get("name", "")
        if name:
            counts[name] = counts.get(name, 0) + 1
    return counts


def draw_detections(
    frame_bgr: np.ndarray, detections: list[dict]
) -> np.ndarray:
    """
    Draw bounding boxes and labels on a copy of the frame.
    Returns the annotated frame (BGR).
    """
    out = frame_bgr.copy()
    color_bgr = (0, 229, 255)  # cyan
    for d in detections:
        xyxy = d.get("xyxy", [])
        if len(xyxy) != 4:
            continue
        x1, y1, x2, y2 = map(int, xyxy)
        name = d.get("name", "?")
        conf = d.get("conf", 0)
        label = f"{name} {conf:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color_bgr, -1)
        cv2.putText(
            out, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )
    return out
