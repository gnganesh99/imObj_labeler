"""Test trained models: training summary per model, and evaluation on the current dataset.

Evaluation runs ultralytics ``model.val`` locally (CPU is fine for a few hundred images) on
image lists built from the current split (see ``Project.eval_lists``):

    val       validation images, clean (no augmented copies)
    val_aug   augmented validation copies only (e.g. noisy): robustness
    train     training images, clean: compare with val to spot over-fitting
    all       every labelled image (when there is no split)
"""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path

EVAL_SETS = {"val": "Validation (clean)", "val_aug": "Validation, augmented copies",
             "train": "Training (clean)", "all": "All labelled images"}


def _read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_args_yaml(p: Path) -> dict:
    try:
        import yaml

        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _results(p: Path) -> list[dict]:
    try:
        with p.open(newline="", encoding="utf-8") as f:
            rows = [{k.strip(): v.strip() for k, v in r.items() if k} for r in csv.DictReader(f)]
    except OSError:
        return []
    out = []
    for r in rows:
        try:
            out.append({"epoch": int(float(r["epoch"])),
                        "map50": float(r.get("metrics/mAP50(B)", "nan")),
                        "map": float(r.get("metrics/mAP50-95(B)", "nan"))})
        except (KeyError, ValueError):
            continue
    return out


def _iso(stamp: str | None) -> str | None:
    """train_yolo.py's "2026-11-09_18-01-54" -> "2026-11-09T18:01:54"."""
    try:
        return datetime.strptime(stamp, "%Y-%m-%d_%H-%M-%S").isoformat()
    except (TypeError, ValueError):
        return stamp


def run_info(model_path: str | Path) -> dict:
    """What is known about how a model was trained, from the run folder around it
    (``<run>/weights/best.pt``): train_yolo.py's training_data.json, ultralytics' args.yaml
    and results.csv. Missing files just leave fields empty."""
    p = Path(model_path)
    info: dict = {"path": str(p), "file": p.name, "size_mb": round(p.stat().st_size / 1e6, 1),
                  "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="minutes")}
    run = p.parent.parent if p.parent.name == "weights" else None
    if run is None:
        return info
    info["run"] = run.name
    args = _read_args_yaml(run / "args.yaml")
    td = _read_json(run / "training_data.json") or {}
    hist = _results(run / "results.csv")
    info.update({"imgsz": args.get("imgsz"), "epochs": args.get("epochs"),
                 "start_model": Path(str(args.get("model", ""))).name or None,
                 "batch": args.get("batch"), "epochs_done": len(hist)})
    if hist:
        best = max(hist, key=lambda h: h["map"] if h["map"] == h["map"] else -1)
        info["best"] = best
        info["last"] = hist[-1]
    if td:
        env = td.get("environment") or {}
        info.update({"trained_start": _iso(td.get("training_start_time")),
                     "trained_end": _iso(td.get("training_end_time")),
                     "gpu": env.get("gpu"), "ultralytics": env.get("ultralytics"),
                     "bundle": Path(str(td.get("bundle", ""))).name or None})
    return info


def evaluate(model_path: str, data_yaml: Path, imgsz: int | None, device: str = "cpu",
             out_dir: Path | None = None) -> dict:
    """``model.val`` on the ``val`` list of ``data_yaml``; precision, recall, mAP50, mAP50-95."""
    from ultralytics import YOLO

    from .predict import model_info

    model = YOLO(str(model_path))  # own instance: val() changes the model (fuse), keep predict's
    if imgsz is None:
        imgsz = model_info(str(model_path))["imgsz"]
    t0 = time.time()
    m = model.val(data=str(data_yaml), split="val", imgsz=int(imgsz), batch=16, device=device,
                  workers=0, plots=False, save_json=False, verbose=False,
                  project=str(out_dir) if out_dir else None, name="val", exist_ok=True)
    names = getattr(m, "names", None) or model.names or {}
    per_class = {}
    try:
        for i, c in enumerate(m.box.ap_class_index):
            p, r, ap50, ap = m.box.class_result(i)
            per_class[str(names.get(int(c), c))] = {"precision": float(p), "recall": float(r),
                                                    "map50": float(ap50), "map": float(ap)}
    except Exception:
        pass
    return {"precision": float(m.box.mp), "recall": float(m.box.mr),
            "map50": float(m.box.map50), "map": float(m.box.map), "per_class": per_class,
            "imgsz": int(imgsz), "seconds": round(time.time() - t0, 1),
            "speed_ms": {k: round(float(v), 1) for k, v in (m.speed or {}).items()}}
