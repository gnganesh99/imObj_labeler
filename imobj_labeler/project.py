"""A data folder and its ``obj_det_labels/`` training output.

Layout written on *Register*::

    <data_folder>/obj_det_labels/
        images/<stem>.png       rendered image the boxes were drawn on
        labels/<stem>.txt       YOLO detection labels
        classes.json            class definitions
        data.yaml               ultralytics dataset file
        register_log.csv        one row per registration (provenance)
"""
from __future__ import annotations

import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from . import annotations as ann
from . import classes as cls_cfg
from . import reader

OUT_DIR = {"detect": "obj_det_labels"}
LOG_FIELDS = ["time", "file", "stem", "n_objects", "source", "model", "channel",
              "flatten", "cmap", "lo", "hi", "width", "height"]


class Project:
    def __init__(self, data_folder: str | Path, task: str = "detect"):
        self.data = Path(data_folder).resolve()
        if not self.data.is_dir():
            raise FileNotFoundError(f"Not a folder: {self.data}")
        self.task = task
        self.out = self.data / OUT_DIR[task]
        self.images = self.out / "images"
        self.labels = self.out / "labels"
        self.classes_path = self.out / "classes.json"
        self.log_path = self.out / "register_log.csv"
        self.yaml_path = self.out / "data.yaml"

    # ------------------------------------------------------------ files
    def files(self) -> list[str]:
        return reader.list_files(self.data)

    def stem_for(self, name: str) -> str:
        """Label stem; files sharing a stem (a.sxm, a.png) get the extension appended."""
        p = Path(name)
        clash = sum(1 for f in self.files() if Path(f).stem == p.stem) > 1
        return f"{p.stem}_{p.suffix.lstrip('.').lower()}" if clash else p.stem

    def path_of(self, name: str) -> Path:
        p = (self.data / name).resolve()
        if p.parent != self.data or not p.is_file():
            raise FileNotFoundError(name)
        return p

    def is_registered(self, name: str) -> bool:
        return (self.labels / f"{self.stem_for(name)}.txt").is_file()

    def status(self) -> dict[str, bool]:
        done = {p.stem for p in self.labels.glob("*.txt")} if self.labels.is_dir() else set()
        return {f: self.stem_for(f) in done for f in self.files()}

    def sibling_labels_dir(self) -> Path | None:
        """If the data folder is a YOLO ``images/`` folder, its ``labels/`` sibling."""
        if self.data.name.lower() == "images":
            d = self.data.parent / "labels"
            return d if d.is_dir() else None
        return None

    def labels_for(self, name: str, width: int, height: int) -> list[dict]:
        stem = self.stem_for(name)
        reg = self.labels / f"{stem}.txt"
        if reg.is_file():
            return ann.read_yolo(reg, width, height, source="registered")
        sib = self.sibling_labels_dir()
        if sib is not None:
            return ann.read_yolo(sib / f"{Path(name).stem}.txt", width, height, source="existing")
        return []

    def registered_params(self, name: str) -> dict | None:
        """Render settings used for the last registration of ``name``."""
        if not self.log_path.is_file():
            return None
        last = None
        with self.log_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("file") == name:
                    last = row
        if not last:
            return None
        return {"channel": last["channel"] or None, "flatten": last["flatten"],
                "cmap": last["cmap"], "lo": float(last["lo"]), "hi": float(last["hi"])}

    # ------------------------------------------------------------ classes
    def classes(self) -> dict:
        return cls_cfg.load(self.classes_path)

    def save_classes(self, cfg: dict) -> dict:
        cfg = cls_cfg.save(self.classes_path, cfg)
        if self.yaml_path.is_file():
            self.write_data_yaml()
        return cfg

    # ------------------------------------------------------------ register
    def register(self, name: str, rgb: np.ndarray, annotations: list[dict],
                 params: reader.RenderParams, source: str = "manual",
                 model: str = "") -> dict:
        h, w = rgb.shape[:2]
        stem = self.stem_for(name)
        self.images.mkdir(parents=True, exist_ok=True)
        self.labels.mkdir(parents=True, exist_ok=True)
        if not self.classes_path.is_file():
            cls_cfg.save(self.classes_path, cls_cfg.DEFAULT)

        Image.fromarray(rgb).save(self.images / f"{stem}.png")
        n = ann.write_yolo(self.labels / f"{stem}.txt", annotations, w, h)

        new_log = not self.log_path.is_file()
        with self.log_path.open("a", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            if new_log:
                wr.writeheader()
            wr.writerow({"time": datetime.now().isoformat(timespec="seconds"),
                         "file": name, "stem": stem, "n_objects": n, "source": source,
                         "model": model, **params.as_dict(), "width": w, "height": h})
        self.write_data_yaml()
        known = set(cls_cfg.names(self.classes()))
        unknown = sorted({int(a["cls"]) for a in annotations} - known)
        return {"stem": stem, "n_objects": n, "unknown_classes": unknown,
                "image": str(self.images / f"{stem}.png"),
                "label": str(self.labels / f"{stem}.txt")}

    def unregister(self, name: str) -> bool:
        stem = self.stem_for(name)
        removed = False
        for p in (self.labels / f"{stem}.txt", self.images / f"{stem}.png"):
            if p.is_file():
                p.unlink()
                removed = True
        return removed

    # ------------------------------------------------------------ data.yaml
    def write_data_yaml(self, val_fraction: float = 0.0, seed: int = 0) -> dict:
        names = cls_cfg.names(self.classes())
        imgs = sorted(self.images.glob("*.png")) if self.images.is_dir() else []
        imgs = [p for p in imgs if (self.labels / f"{p.stem}.txt").is_file()]
        lines = [f"path: {json.dumps(self.out.as_posix())}"]
        n_train = n_val = len(imgs)
        if val_fraction > 0 and len(imgs) >= 2:
            rng = random.Random(seed)
            shuffled = imgs[:]
            rng.shuffle(shuffled)
            n_val = max(1, int(round(len(imgs) * val_fraction)))
            val, train = shuffled[:n_val], shuffled[n_val:]
            n_train = len(train)
            (self.out / "train.txt").write_text(
                "".join(f"./images/{p.name}\n" for p in sorted(train)), encoding="utf-8")
            (self.out / "val.txt").write_text(
                "".join(f"./images/{p.name}\n" for p in sorted(val)), encoding="utf-8")
            lines += ["train: train.txt", "val: val.txt"]
        else:
            lines += ["train: images", "val: images"]
        lines.append("names:")
        lines += [f"  {i}: {json.dumps(n)}" for i, n in sorted(names.items())]
        self.yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {"path": str(self.yaml_path), "n_train": n_train, "n_val": n_val,
                "split": val_fraction > 0 and len(imgs) >= 2}
