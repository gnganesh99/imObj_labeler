"""Object classes (classes.json).

Format::

    {"task": "detect",
     "classes": [{"id": 0, "name": "CO", "color": "#e6194b"}, ...]}
"""
from __future__ import annotations

import json
from pathlib import Path

PALETTE = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
           "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff"]

DEFAULT = {"task": "detect", "classes": [{"id": 0, "name": "object", "color": PALETTE[0]}]}


def validate(cfg: dict) -> dict:
    if not isinstance(cfg, dict):
        raise ValueError("classes.json must be an object")
    task = cfg.get("task", "detect")
    if task not in ("detect", "segment"):
        raise ValueError(f"Unsupported task '{task}'")
    classes = cfg.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError("'classes' must be a non-empty list")
    seen_ids, seen_names, out = set(), set(), []
    for i, c in enumerate(classes):
        if not isinstance(c, dict) or "name" not in c:
            raise ValueError(f"class #{i} needs a 'name'")
        cid = int(c.get("id", i))
        name = str(c["name"]).strip()
        if not name:
            raise ValueError(f"class #{i} has an empty name")
        if cid < 0 or cid in seen_ids:
            raise ValueError(f"duplicate or negative class id {cid}")
        if name in seen_names:
            raise ValueError(f"duplicate class name '{name}'")
        seen_ids.add(cid)
        seen_names.add(name)
        out.append({"id": cid, "name": name,
                    "color": str(c.get("color") or PALETTE[cid % len(PALETTE)])})
    out.sort(key=lambda c: c["id"])
    return {"task": task, "classes": out}


def load(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        return validate(DEFAULT)
    return validate(json.loads(path.read_text(encoding="utf-8")))


def save(path: str | Path, cfg: dict) -> dict:
    cfg = validate(cfg)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg


def from_names(names: dict[int, str]) -> dict:
    return validate({"task": "detect", "classes": [
        {"id": int(i), "name": n, "color": PALETTE[int(i) % len(PALETTE)]}
        for i, n in sorted(names.items())]})


def names(cfg: dict) -> dict[int, str]:
    return {c["id"]: c["name"] for c in cfg["classes"]}
