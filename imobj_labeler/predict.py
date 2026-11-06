"""YOLO inference (ultralytics). Imported lazily so labeling works without torch."""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from . import annotations as ann

_models: dict[str, object] = {}
_lock = threading.Lock()


def find_models(folder: str | Path, max_depth: int = 4) -> list[dict]:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    out = []
    for p in sorted(folder.rglob("*.pt")):
        rel = p.relative_to(folder)
        if len(rel.parts) > max_depth + 1:
            continue
        out.append({"path": str(p), "name": rel.as_posix()})
    return out


def _load(path: str):
    key = str(Path(path).resolve())
    with _lock:
        if key not in _models:
            from ultralytics import YOLO

            _models[key] = YOLO(key)
        return _models[key]


def model_info(path: str) -> dict:
    m = _load(path)
    args = {}
    ckpt = getattr(m, "ckpt", None)
    if isinstance(ckpt, dict):
        args = ckpt.get("train_args", {}) or {}
    imgsz = args.get("imgsz") or getattr(m, "overrides", {}).get("imgsz") or 640
    if isinstance(imgsz, (list, tuple)):
        imgsz = max(imgsz)
    return {"path": path, "task": getattr(m, "task", "detect"),
            "names": {int(k): str(v) for k, v in (m.names or {}).items()},
            "imgsz": int(imgsz)}


def predict(path: str, rgb: np.ndarray, conf: float = 0.25, iou: float = 0.7,
            imgsz: int | None = None, device: str | None = None) -> list[dict]:
    """Run detection on an RGB uint8 image; return bbox annotations in pixels."""
    m = _load(path)
    if imgsz is None:
        imgsz = model_info(path)["imgsz"]
    # ultralytics treats numpy input as BGR (cv2 convention), like its training loader
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    kw = dict(conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    if device:
        kw["device"] = device
    with _lock:
        res = m.predict(bgr, **kw)[0]
    out = []
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return out
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    cf = boxes.conf.cpu().numpy()
    for (x1, y1, x2, y2), c, p in zip(xyxy, cls, cf):
        out.append(ann.bbox(int(c), x1, y1, x2, y2, conf=round(float(p), 4), source="model"))
    return out
