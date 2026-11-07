"""A data folder and its ``obj_det_labels/`` training output.

Layout written on *Register*::

    <data_folder>/obj_det_labels/
        images/<stem>.png       rendered image the boxes were drawn on
        labels/<stem>.txt       YOLO detection labels
        classes.json            class definitions
        data.yaml               ultralytics dataset file (train.txt / val.txt image lists)
        split.json              dataset source + split settings used for data.yaml
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
DATASET_SOURCES = ("combined", "registered", "existing")
SPLIT_DEFAULT = {"source": "combined", "val_fraction": 0.0, "seed": 0}
YOLO_IMAGE_EXT = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


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
        self.split_path = self.out / "split.json"  # last split settings, re-applied on register

    # ------------------------------------------------------------ files
    def files(self) -> list[str]:
        return reader.list_files(self.data)

    def stem_for(self, name: str, files: list[str] | None = None) -> str:
        """Label stem; files sharing a stem (a.sxm, a.png) get the extension appended.
        Pass ``files`` when calling in a loop: listing the folder is slow."""
        p = Path(name)
        files = self.files() if files is None else files
        clash = sum(1 for f in files if Path(f).stem == p.stem) > 1
        return f"{p.stem}_{p.suffix.lstrip('.').lower()}" if clash else p.stem

    def path_of(self, name: str) -> Path:
        p = (self.data / name).resolve()
        if p.parent != self.data or not p.is_file():
            raise FileNotFoundError(name)
        return p

    # A file can be registered several times (overwrite off): <stem>, <stem>_1, <stem>_2, ...
    # register_log.csv maps each saved stem back to its source file.
    def _log_rows(self) -> list[dict]:
        if not self.log_path.is_file():
            return []
        with self.log_path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _exists(self, stem: str) -> bool:
        return (self.labels / f"{stem}.txt").is_file()

    def versions(self, name: str, rows: list[dict] | None = None,
                 files: list[str] | None = None) -> list[str]:
        """Saved stems of ``name`` that still exist, oldest registration first."""
        rows = self._log_rows() if rows is None else rows
        out: list[str] = []
        base = self.stem_for(name, files)
        for r in rows:
            s = r.get("stem")
            if r.get("file") == name and s:
                if s in out:
                    out.remove(s)  # re-registered (overwritten): move to the end
                out.append(s)
        if base not in out:
            out.insert(0, base)
        return [s for s in out if self._exists(s)]

    def current_stem(self, name: str) -> str | None:
        """Most recently registered version (shown and edited in the app)."""
        v = self.versions(name)
        return v[-1] if v else None

    def _next_free_stem(self, base: str) -> str:
        if not (self._exists(base) or (self.images / f"{base}.png").is_file()):
            return base
        i = 1
        while self._exists(f"{base}_{i}") or (self.images / f"{base}_{i}.png").is_file():
            i += 1
        return f"{base}_{i}"

    def is_registered(self, name: str) -> bool:
        return bool(self.versions(name))

    def status(self) -> dict[str, bool]:
        rows, files = self._log_rows(), self.files()
        return {f: bool(self.versions(f, rows, files)) for f in files}

    def sibling_labels_dir(self) -> Path | None:
        """If the data folder is a YOLO ``images/`` folder, its ``labels/`` sibling."""
        if self.data.name.lower() == "images":
            d = self.data.parent / "labels"
            return d if d.is_dir() else None
        return None

    def labels_for(self, name: str, width: int, height: int) -> list[dict]:
        stem = self.current_stem(name)
        if stem:
            return ann.read_yolo(self.labels / f"{stem}.txt", width, height, source="registered")
        sib = self.sibling_labels_dir()
        if sib is not None:
            return ann.read_yolo(sib / f"{Path(name).stem}.txt", width, height, source="existing")
        return []

    def registered_params(self, name: str) -> dict | None:
        """Render settings used for the current registered version of ``name``."""
        stem = self.current_stem(name)
        last = None
        for row in self._log_rows():
            if row.get("file") == name and row.get("stem") == stem:
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
                 model: str = "", overwrite: bool = True) -> dict:
        """Save image + labels. ``overwrite`` replaces the current version of this file;
        otherwise a new copy is written as ``<stem>_1``, ``<stem>_2``, ..."""
        h, w = rgb.shape[:2]
        base = self.stem_for(name)
        if overwrite:
            stem = self.current_stem(name) or base
            replaced = self._exists(stem)
        else:
            stem = self._next_free_stem(base)
            replaced = False
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
        return {"stem": stem, "n_objects": n, "unknown_classes": unknown, "replaced": replaced,
                "versions": self.versions(name),
                "image": str(self.images / f"{stem}.png"),
                "label": str(self.labels / f"{stem}.txt")}

    def existing_label_path(self, name: str) -> Path | None:
        """The read-only label file for ``name`` in the sibling ``../labels`` folder."""
        sib = self.sibling_labels_dir()
        p = sib / f"{Path(name).stem}.txt" if sib is not None else None
        return p if p is not None and p.is_file() else None

    def unregister(self, name: str, all_versions: bool = False) -> bool:
        """Remove the current version (an earlier one, if any, becomes current), or with
        ``all_versions`` every registered version, so existing labels are shown again."""
        stems = self.versions(name) if all_versions else [self.current_stem(name)]
        removed = False
        for stem in filter(None, stems):
            for p in (self.labels / f"{stem}.txt", self.images / f"{stem}.png"):
                if p.is_file():
                    p.unlink()
                    removed = True
        return removed

    # ------------------------------------------------------------ data.yaml
    # Training images come from two places:
    #   registered: obj_det_labels/images + labels (written by this app)
    #   existing:   the data folder itself, when it is a YOLO images/ folder with ../labels
    # "combined" uses both; a file that is registered replaces its existing image + labels.
    # The split is done per source file, so all registered versions of one file stay together.
    def split_settings(self) -> dict:
        cfg = dict(SPLIT_DEFAULT)
        try:
            cfg.update(json.loads(self.split_path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
        return cfg

    def dataset_groups(self, source: str = "combined") -> tuple[dict[str, list[Path]], dict]:
        """Images for training, grouped by source file, plus counts per origin."""
        if source not in DATASET_SOURCES:
            raise ValueError(f"source must be one of {DATASET_SOURCES}")
        files, rows = self.files(), self._log_rows()
        reg: dict[str, list[Path]] = {}
        claimed: set[str] = set()
        for name in files:
            stems = self.versions(name, rows, files)
            claimed.update(stems)
            imgs = [self.images / f"{s}.png" for s in stems]
            imgs = [p for p in imgs if p.is_file()]
            if imgs:
                reg[name] = imgs
        if self.labels.is_dir():  # registered images whose source file is gone: own group each
            for lab in sorted(self.labels.glob("*.txt")):
                img = self.images / f"{lab.stem}.png"
                if lab.stem not in claimed and img.is_file():
                    reg[f"[{lab.stem}]"] = [img]
        existing: dict[str, list[Path]] = {}
        sib = self.sibling_labels_dir()
        if sib is not None:
            for name in files:
                p = self.data / name
                if p.suffix.lower() in YOLO_IMAGE_EXT and (sib / f"{p.stem}.txt").is_file():
                    existing[name] = [p]
        overridden = sorted(set(reg) & set(existing))
        if source == "registered":
            groups = reg
        elif source == "existing":
            groups = existing
        else:
            groups = {**existing, **reg}  # registered takes precedence
        stats = {"n_registered": len(reg), "n_existing": len(existing),
                 "n_overridden": len(overridden) if source == "combined" else 0}
        return groups, stats

    def _list_entry(self, img: Path) -> str:
        try:
            return "./" + img.relative_to(self.out).as_posix()
        except ValueError:
            return img.as_posix()  # existing images live outside obj_det_labels

    def write_data_yaml(self, val_fraction: float | None = None, seed: int | None = None,
                        source: str | None = None) -> dict:
        """Write train.txt / val.txt / data.yaml. Called without arguments (e.g. after a
        registration) it re-applies the last saved split settings instead of resetting them."""
        cfg = self.split_settings()
        given = {k: v for k, v in (("val_fraction", val_fraction), ("seed", seed),
                                   ("source", source)) if v is not None}
        cfg.update(given)
        groups, stats = self.dataset_groups(cfg["source"])  # also validates the source
        self.out.mkdir(parents=True, exist_ok=True)
        if given:
            self.split_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        keys = sorted(groups, key=reader.natural_key)
        frac = float(cfg["val_fraction"])
        split = frac > 0 and len(keys) >= 2
        val_keys: set[str] = set()
        if split:
            shuffled = keys[:]
            random.Random(int(cfg["seed"])).shuffle(shuffled)
            n_val = min(len(keys) - 1, max(1, int(round(len(keys) * frac))))
            val_keys = set(shuffled[:n_val])
        train = [img for k in keys if k not in val_keys for img in groups[k]]
        val = [img for k in keys if k in val_keys for img in groups[k]]

        def write_list(fname: str, imgs: list[Path]):
            (self.out / fname).write_text("".join(self._list_entry(p) + "\n" for p in imgs),
                                          encoding="utf-8")

        write_list("train.txt", train)
        if split:
            write_list("val.txt", val)
        elif (self.out / "val.txt").is_file():
            (self.out / "val.txt").unlink()

        names = cls_cfg.names(self.classes())
        lines = [f"# source: {cfg['source']}, val_fraction: {frac}, seed: {cfg['seed']}"
                 " (written by imobj_labeler)",
                 f"path: {json.dumps(self.out.as_posix())}",
                 "train: train.txt", f"val: {'val.txt' if split else 'train.txt'}", "names:"]
        lines += [f"  {i}: {json.dumps(n)}" for i, n in sorted(names.items())]
        self.yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {"path": str(self.yaml_path), "source": cfg["source"], "val_fraction": frac,
                "seed": int(cfg["seed"]), "split": split,
                "n_train": len(train), "n_val": len(val) if split else len(train),
                "groups_train": len(keys) - len(val_keys), "groups_val": len(val_keys), **stats}
