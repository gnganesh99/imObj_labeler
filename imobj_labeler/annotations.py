"""Annotation model and YOLO text I/O.

Annotations are plain dicts so they serialise straight to/from the browser::

    {"type": "bbox", "cls": 0, "x1": .., "y1": .., "x2": .., "y2": ..,
     "conf": 0.87 | None, "source": "manual" | "model" | "edited" | "existing"}

Coordinates are pixels in the displayed (top-left origin) image. A future
segmentation mode adds ``{"type": "polygon", "cls": 0, "points": [[x, y], ...]}``.
"""
from __future__ import annotations

from pathlib import Path


def bbox(cls: int, x1: float, y1: float, x2: float, y2: float,
         conf: float | None = None, source: str = "manual") -> dict:
    x1, x2 = sorted((float(x1), float(x2)))
    y1, y2 = sorted((float(y1), float(y2)))
    return {"type": "bbox", "cls": int(cls), "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "conf": conf, "source": source}


def clip_bbox(a: dict, width: int, height: int) -> dict | None:
    """Clip to the image; return None if the box collapses."""
    x1 = min(max(a["x1"], 0.0), width)
    x2 = min(max(a["x2"], 0.0), width)
    y1 = min(max(a["y1"], 0.0), height)
    y2 = min(max(a["y2"], 0.0), height)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return {**a, "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def parse_yolo(text: str, width: int, height: int, source: str = "existing") -> list[dict]:
    """Parse YOLO detection labels. Tolerates tabs, float class ids ("0.0")
    and extra columns (e.g. a trailing confidence)."""
    out = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        conf = float(parts[5]) if len(parts) > 5 else None
        out.append(bbox(cls, (cx - w / 2) * width, (cy - h / 2) * height,
                        (cx + w / 2) * width, (cy + h / 2) * height, conf, source))
    return out


def read_yolo(path: str | Path, width: int, height: int, source: str = "existing") -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    return parse_yolo(path.read_text(encoding="utf-8", errors="replace"), width, height, source)


def format_yolo(annotations: list[dict], width: int, height: int) -> str:
    """Integer class id, space separated, 6 decimals, normalised xywh."""
    lines = []
    for a in annotations:
        if a.get("type", "bbox") != "bbox":
            continue
        a = clip_bbox(a, width, height)
        if a is None:
            continue
        cx = (a["x1"] + a["x2"]) / 2 / width
        cy = (a["y1"] + a["y2"]) / 2 / height
        w = (a["x2"] - a["x1"]) / width
        h = (a["y2"] - a["y1"]) / height
        lines.append(f"{int(a['cls'])} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def write_yolo(path: str | Path, annotations: list[dict], width: int, height: int) -> int:
    text = format_yolo(annotations, width, height)
    Path(path).write_text(text, encoding="utf-8")
    return text.count("\n")
