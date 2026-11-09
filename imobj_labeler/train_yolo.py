#!/usr/bin/env python
"""Train a YOLO detector on a training bundle exported by imObj Labeler.

Standalone: needs only ultralytics + torch (no imobj_labeler install). Copy it to the GPU
machine once and use it for every bundle:

    python train_yolo.py BUNDLE                 train with BUNDLE/train_config.json
    python train_yolo.py BUNDLE --epochs 5      override any config value from the command line
    python train_yolo.py BUNDLE --check         validate the bundle + environment, don't train
    python train_yolo.py BUNDLE --resume        continue the latest run from its last.pt
    python train_yolo.py --version

Run it detached so a dropped SSH connection doesn't kill it, e.g.
    nohup python train_yolo.py ~/bundles/NAME > ~/bundles/NAME/train.log 2>&1 &

Each run is written to BUNDLE/runs/<name>_<timestamp>/ (ultralytics output: weights/best.pt,
last.pt, results.csv, plots) together with a copy of this script, the config used and
training_data.json (settings, versions, GPU, timing and final metrics).
"""
from __future__ import annotations

import argparse
import datetime
import json
import platform
import shutil
import sys
from pathlib import Path

SCRIPT_VERSION = "1.0.0"
SUPPORTED_FORMATS = {1}  # bundle format_version values this script understands


def now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def fail(msg: str, code: int = 2):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def load_bundle(bundle: Path) -> tuple[dict, dict]:
    if not (bundle / "train_config.json").is_file() or not (bundle / "data.yaml").is_file():
        fail(f"{bundle} is not a training bundle (train_config.json / data.yaml missing)")
    cfg = json.loads((bundle / "train_config.json").read_text(encoding="utf-8"))
    manifest = {}
    if (bundle / "manifest.json").is_file():
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    fmt = cfg.get("format_version", manifest.get("format_version"))
    if fmt not in SUPPORTED_FORMATS:
        fail(f"bundle format {fmt} is not supported by train_yolo.py {SCRIPT_VERSION} "
             f"(supports {sorted(SUPPORTED_FORMATS)}): update this script from the app")
    return cfg, manifest


def pick_device(requested: str, allow_cpu: bool) -> str:
    import torch

    if str(requested).lower() == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return str(requested)
    if allow_cpu:
        print("WARNING: CUDA not available, training on CPU (--allow-cpu)")
        return "cpu"
    fail("CUDA is not available to torch, so training would silently run on CPU. Check "
         "nvidia-smi and that torch was installed with CUDA, or pass --allow-cpu.")


def environment(device: str) -> dict:
    import torch
    import ultralytics

    env = {"python": platform.python_version(), "torch": torch.__version__,
           "ultralytics": ultralytics.__version__, "host": platform.node(), "device": device,
           "train_yolo_version": SCRIPT_VERSION}
    if device != "cpu" and torch.cuda.is_available():
        idx = int(str(device).split(",")[0]) if str(device).split(",")[0].isdigit() else 0
        env["gpu"] = torch.cuda.get_device_name(idx)
    return env


ONLINE_LOGGERS = ("wandb", "comet", "clearml", "mlflow", "neptune", "dvc")


def disable_online_loggers(keep=()):
    """Turn off experiment-tracking integrations for this process only.

    If e.g. wandb is installed, ultralytics starts it automatically; on a detached run it can
    wait for a login, and it rejects absolute project folders. ``dict.__setitem__`` changes
    the in-memory setting without writing ultralytics' global settings file."""
    from ultralytics.utils import SETTINGS

    for key in ONLINE_LOGGERS:
        if key in SETTINGS and key not in keep:
            dict.__setitem__(SETTINGS, key, False)


def latest_last_pt(runs: Path) -> Path | None:
    cands = sorted(runs.glob("*/weights/last.pt"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def check(bundle: Path, data_yaml: Path):
    from ultralytics.data.utils import check_det_dataset

    d = check_det_dataset(str(data_yaml))
    for split in ("train", "val", "test"):
        if d.get(split):
            src = Path(d[split])
            n = len(list(src.iterdir())) if src.is_dir() else len(src.read_text().split())
            print(f"  {split:5s}: {n} images  ({src})")
    print(f"  classes: {d['names']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bundle", nargs="?", help="bundle folder (contains train_config.json)")
    ap.add_argument("--version", action="store_true", help="print the script version and exit")
    ap.add_argument("--check", action="store_true", help="validate bundle + environment only")
    ap.add_argument("--resume", action="store_true", help="resume the latest run (last.pt)")
    ap.add_argument("--allow-cpu", action="store_true", help="train on CPU if CUDA is missing")
    for key, typ in (("model", str), ("epochs", int), ("imgsz", int), ("batch", float),
                     ("patience", int), ("save_period", int), ("device", str),
                     ("workers", int), ("name", str)):
        ap.add_argument(f"--{key}", type=typ, help=f"override train_config.json '{key}'")
    a = ap.parse_args(argv)

    if a.version:
        print(f"train_yolo.py {SCRIPT_VERSION} (bundle formats {sorted(SUPPORTED_FORMATS)})")
        return
    if not a.bundle:
        ap.error("the bundle folder is required")

    bundle = Path(a.bundle).expanduser().resolve()
    cfg, manifest = load_bundle(bundle)
    for key in ("model", "epochs", "imgsz", "batch", "patience", "save_period", "device",
                "workers", "name"):
        if getattr(a, key) is not None:
            cfg[key] = getattr(a, key)
    if cfg.get("batch") is not None and float(cfg["batch"]).is_integer():
        cfg["batch"] = int(cfg["batch"])  # -1 = auto batch; 0.7 = fraction of GPU memory
    data_yaml = bundle / "data.yaml"  # absolute: ultralytics resolves its paths next to it
    runs = bundle / "runs"

    print(f"train_yolo.py {SCRIPT_VERSION} | bundle {bundle}")
    if manifest:
        c = manifest.get("counts", {})
        print(f"  exported {manifest.get('created')} from {manifest.get('data_folder')}: "
              f"{c.get('train')} train / {c.get('val')} val images "
              f"({c.get('train_aug')} / {c.get('val_aug')} augmented)")
    try:
        device = pick_device(cfg.get("device", "0"), a.allow_cpu)
    except SystemExit:
        if not a.check:
            raise
        device, cuda_problem = "cpu", True  # --check: report, but still validate the data
    else:
        cuda_problem = False
    env = environment(device)
    print(f"  device {device} {env.get('gpu', '')} | torch {env['torch']} | "
          f"ultralytics {env['ultralytics']}")
    if manifest.get("ultralytics_at_export") and \
            manifest["ultralytics_at_export"].split(".")[:2] != env["ultralytics"].split(".")[:2]:
        print(f"  NOTE: the app has ultralytics {manifest['ultralytics_at_export']}; a model "
              f"trained with {env['ultralytics']} may not load there. Match the versions.")

    from ultralytics import YOLO

    if a.check:
        check(bundle, data_yaml)
        model = cfg["model"]
        local = bundle / model
        print(f"  model: {local if local.is_file() else model + ' (downloaded by ultralytics)'}")
        if cuda_problem:
            fail("bundle OK, but CUDA is not available (see above)")
        print("OK")
        return

    # train_config.json "loggers": ["wandb"] keeps an integration on (off by default)
    disable_online_loggers(keep=cfg.get("loggers") or ())
    start = now()
    if a.resume:
        last = latest_last_pt(runs)
        if last is None:
            fail(f"nothing to resume: no {runs}/*/weights/last.pt")
        print(f"Resuming {last}")
        model = YOLO(str(last))
        model.train(resume=True, device=device)
    else:
        run_name = f"{cfg.get('name') or 'train'}_{start}"
        save_dir = runs / run_name
        save_dir.mkdir(parents=True)
        # keep exactly what produced this run next to its weights
        shutil.copy2(__file__, save_dir / "train_yolo.py")
        (save_dir / "train_config_used.json").write_text(json.dumps(cfg, indent=2))
        weights = bundle / cfg["model"]
        model = YOLO(str(weights) if weights.is_file() else cfg["model"])
        print(f"Training {cfg['epochs']} epochs, imgsz {cfg['imgsz']}, batch {cfg['batch']}, "
              f"patience {cfg['patience']}, save_period {cfg['save_period']} -> {save_dir}")
        model.train(data=str(data_yaml), epochs=int(cfg["epochs"]), imgsz=int(cfg["imgsz"]),
                    batch=cfg["batch"], patience=int(cfg["patience"]),
                    save_period=int(cfg["save_period"]), device=device,
                    workers=int(cfg.get("workers", 8)), seed=int(cfg.get("seed", 0)),
                    project=str(runs), name=run_name, exist_ok=True, verbose=True,
                    **(cfg.get("args") or {}))

    trainer = model.trainer
    save_dir = Path(trainer.save_dir)
    metrics = getattr(trainer, "metrics", None) or {}
    summary = {
        "bundle": str(bundle), "data_yaml": str(data_yaml), "model": cfg["model"],
        "epochs": cfg["epochs"], "imgsz": cfg["imgsz"], "batch": cfg["batch"],
        "patience": cfg["patience"], "save_period": cfg["save_period"],
        "args": cfg.get("args"), "resumed": a.resume, "training_results": str(save_dir),
        "best": str(save_dir / "weights" / "best.pt"),
        "training_start_time": start, "training_end_time": now(),
        "final_metrics": {k: float(v) for k, v in dict(metrics).items()
                          if isinstance(v, (int, float))},
        "environment": env, "bundle_manifest_created": manifest.get("created"),
    }
    (save_dir / "training_data.json").write_text(json.dumps(summary, indent=4))
    print("Training completed. Results saved in:", save_dir)


if __name__ == "__main__":
    main()
