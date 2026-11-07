"""Offline augmentation for YOLO detection data, with exact boxes.

Only transforms that keep axis-aligned boxes exact are offered: 90-degree rotations, plus
Gaussian noise and blur (which don't move anything). Flips, scaling, crops, mosaic and
brightness are better done on the fly by ultralytics during training.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

TRANSFORMS = ("rot90", "rot180", "rot270", "noise", "blur")
_ROT_K = {"rot90": 1, "rot180": 2, "rot270": 3}


def label_path_for(img: Path) -> Path:
    """The label file YOLO uses for an image: last ``/images/`` -> ``/labels/``, ``.txt``."""
    head, sep, tail = Path(img).as_posix().rpartition("/images/")
    if not sep:
        raise ValueError(f"image is not inside an images/ folder: {img}")
    return Path(f"{head}/labels/{tail.rsplit('.', 1)[0]}.txt")


def read_boxes(path: Path) -> list[tuple[int, float, float, float, float]]:
    """YOLO rows (cls, cx, cy, w, h); tolerant of tabs, ``0.0`` classes and junk lines."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            rows.append((int(float(parts[0])), *map(float, parts[1:5])))
        except ValueError:
            continue
    return rows


def write_boxes(path: Path, rows) -> None:
    Path(path).write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
                                  for c, x, y, w, h in rows), encoding="utf-8")


def rotate_box(cx: float, cy: float, w: float, h: float, k: int):
    """Box after ``np.rot90(image, k)`` (k quarter turns counter-clockwise)."""
    k %= 4
    if k == 1:
        return cy, 1 - cx, h, w
    if k == 2:
        return 1 - cx, 1 - cy, w, h
    if k == 3:
        return 1 - cy, cx, h, w
    return cx, cy, w, h


def load_image(path: Path) -> np.ndarray:
    """uint8 array, 2D for grayscale or HxWx3 for colour (kept as it is on disk)."""
    im = Image.open(path)
    if im.mode not in ("L", "RGB"):
        im = im.convert("L" if im.mode in ("I", "I;16", "F", "LA") else "RGB")
    return np.asarray(im)


def apply(arr: np.ndarray, rows, transform: str, rng: np.random.Generator,
          noise_sigma: float = 10.0, blur_radius: float = 1.0):
    """Return (image, boxes) for one transform."""
    if transform in _ROT_K:
        k = _ROT_K[transform]
        return (np.ascontiguousarray(np.rot90(arr, k)),
                [(c, *rotate_box(x, y, w, h, k)) for c, x, y, w, h in rows])
    if transform == "noise":
        noise = rng.normal(0.0, noise_sigma, arr.shape[:2])
        if arr.ndim == 3:
            noise = noise[..., None]  # same noise on every channel, like noise in the data
        return np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8), list(rows)
    if transform == "blur":
        im = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(blur_radius))
        return np.asarray(im), list(rows)
    raise ValueError(f"unknown transform {transform!r}; choose from {TRANSFORMS}")
