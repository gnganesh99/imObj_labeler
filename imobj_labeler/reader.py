"""Read microscopy / image files into 2D arrays and render them to RGB.

Supported inputs
----------------
* ``.sxm`` (Nanonis) via SciFiReaders. Every channel/direction is exposed as a
  separate 2D channel; the first one is used when none is requested.
* Common raster images (png, jpg, tif, bmp). RGB / 8-bit grayscale images are
  passed through unchanged; 16-bit / float images are treated as 2D data.

Orientation
-----------
SciFiReaders returns SXM images with row 0 at the *bottom* of the scan frame.
They are flipped here so that row 0 is the *top* (image convention). Everything
downstream (displayed image, saved PNG, YOLO boxes) uses this top-left frame.
"""
from __future__ import annotations

import re
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

SXM_EXT = {".sxm"}
IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SUPPORTED = SXM_EXT | IMG_EXT

FLATTEN_MODES = ("none", "offset", "line", "plane")
COLORMAPS = ("viridis", "gray", "afmhot", "inferno", "magma", "plasma", "cividis")

CACHE_SIZE = 16


def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def list_files(folder: str | Path) -> list[str]:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    names = [p.name for p in folder.iterdir()
             if p.is_file() and p.suffix.lower() in SUPPORTED]
    return sorted(names, key=natural_key)


@dataclass
class Loaded:
    path: Path
    kind: str                      # "sxm" | "rgb" | "data"
    channels: "OrderedDict[str, np.ndarray]" = field(default_factory=OrderedDict)
    units: dict = field(default_factory=dict)
    rgb: np.ndarray | None = None  # for passthrough images
    meta: dict = field(default_factory=dict)

    @property
    def channel_names(self) -> list[str]:
        return list(self.channels.keys())


_cache: "OrderedDict[str, tuple[float, Loaded]]" = OrderedDict()


def load(path: str | Path) -> Loaded:
    """Load a file (cached, invalidated on modification time)."""
    path = Path(path)
    key = str(path.resolve())
    mtime = path.stat().st_mtime
    hit = _cache.get(key)
    if hit and hit[0] == mtime:
        _cache.move_to_end(key)
        return hit[1]
    ext = path.suffix.lower()
    if ext in SXM_EXT:
        obj = _load_sxm(path)
    elif ext in IMG_EXT:
        obj = _load_image(path)
    else:
        raise ValueError(f"Unsupported file type: {path.name}")
    _cache[key] = (mtime, obj)
    while len(_cache) > CACHE_SIZE:
        _cache.popitem(last=False)
    return obj


def _load_sxm(path: Path) -> Loaded:
    from SciFiReaders import NanonisSXMReader

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        datasets = NanonisSXMReader(str(path)).read()
    obj = Loaded(path=path, kind="sxm")
    meta = {}
    for name, ds in datasets.items():
        arr = np.asarray(ds, dtype=np.float64)
        if arr.ndim != 2:
            continue  # 2D only for now
        obj.channels[name] = np.flipud(arr)  # row 0 -> top of the scan frame
        obj.units[name] = str(getattr(ds, "units", "") or "")
        if not meta:
            om = getattr(ds, "original_metadata", {}) or {}
            for k in ("scan_pixels", "scan_range", "scan_offset", "scan_angle",
                      "scan_dir", "bias", "rec_date", "rec_time"):
                if k in om:
                    v = om[k]
                    meta[k] = v.tolist() if hasattr(v, "tolist") else v
    if not obj.channels:
        raise ValueError(f"No 2D channels found in {path.name}")
    obj.meta = meta
    return obj


def _load_image(path: Path) -> Loaded:
    im = Image.open(path)
    if im.mode in ("RGB", "RGBA", "P", "CMYK", "L", "LA", "1"):
        rgb = np.asarray(im.convert("RGB"))
        return Loaded(path=path, kind="rgb", rgb=rgb)
    arr = np.asarray(im, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[..., 0]
    obj = Loaded(path=path, kind="data")
    obj.channels["intensity"] = arr
    obj.units["intensity"] = ""
    return obj


# ---------------------------------------------------------------- processing
def apply_flatten(img: np.ndarray, mode: str) -> np.ndarray:
    img = np.array(img, dtype=np.float64)
    if mode == "none" or mode is None:
        return img
    if mode == "offset":
        return img - np.nanmedian(img, axis=1, keepdims=True)
    if mode == "line":
        x = np.arange(img.shape[1], dtype=np.float64)
        out = np.empty_like(img)
        for i, row in enumerate(img):
            ok = np.isfinite(row)
            if ok.sum() >= 2:
                p = np.polyfit(x[ok], row[ok], 1)
                out[i] = row - np.polyval(p, x)
            else:
                out[i] = row
        return out
    if mode == "plane":
        ny, nx = img.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        ok = np.isfinite(img)
        A = np.c_[xx[ok], yy[ok], np.ones(ok.sum())]
        coef, *_ = np.linalg.lstsq(A, img[ok], rcond=None)
        return img - (coef[0] * xx + coef[1] * yy + coef[2])
    raise ValueError(f"Unknown flatten mode: {mode}")


def _lut(cmap: str) -> np.ndarray:
    if cmap == "gray":
        g = np.arange(256, dtype=np.uint8)
        return np.stack([g, g, g], axis=1)
    import matplotlib

    cm = matplotlib.colormaps[cmap]
    return (cm(np.linspace(0, 1, 256))[:, :3] * 255).round().astype(np.uint8)


def to_rgb(img: np.ndarray, cmap: str = "viridis", lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """Percentile-clip a 2D array and map it through a colormap -> uint8 RGB."""
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return np.zeros(img.shape + (3,), np.uint8)
    vmin, vmax = np.percentile(finite, [lo, hi])
    if vmax <= vmin:
        vmin, vmax = finite.min(), finite.max()
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = np.clip((img - vmin) / (vmax - vmin), 0, 1)
    idx = np.nan_to_num(norm * 255, nan=0).round().astype(np.uint8)
    return _lut(cmap)[idx]


@dataclass
class RenderParams:
    channel: str | None = None
    flatten: str = "plane"
    cmap: str = "viridis"
    lo: float = 1.0
    hi: float = 99.0

    def as_dict(self) -> dict:
        return dict(channel=self.channel, flatten=self.flatten, cmap=self.cmap,
                    lo=self.lo, hi=self.hi)


def render(path: str | Path, params: RenderParams) -> tuple[np.ndarray, dict]:
    """Return (RGB uint8 image, info). Falls back to the first channel."""
    obj = load(path)
    info = {"kind": obj.kind, "channels": obj.channel_names, "fallback": False,
            "channel": None, "meta": obj.meta}
    if obj.kind == "rgb":
        rgb = obj.rgb
    else:
        ch = params.channel
        if ch not in obj.channels:
            info["fallback"] = ch is not None
            ch = obj.channel_names[0]
        info["channel"] = ch
        info["units"] = obj.units.get(ch, "")
        data = apply_flatten(obj.channels[ch], params.flatten)
        rgb = to_rgb(data, params.cmap, params.lo, params.hi)
    info["height"], info["width"] = rgb.shape[:2]
    return np.ascontiguousarray(rgb), info
