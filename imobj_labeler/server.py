"""FastAPI backend: serves the browser UI and the JSON/PNG API."""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from . import classes as cls_cfg
from . import folders
from . import reader
from .project import Project

WEB = Path(__file__).parent / "web"
SETTINGS_PATH = Path.home() / ".imobj_labeler" / "settings.json"


class State:
    project: Project | None = None
    settings: dict = {}
    native_dialog: bool = True  # False for remote/headless servers -> in-app picker

    @classmethod
    def load_settings(cls):
        try:
            cls.settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            cls.settings = {}

    @classmethod
    def save_settings(cls, **kw):
        cls.settings.update({k: v for k, v in kw.items() if v is not None})
        try:
            SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            SETTINGS_PATH.write_text(json.dumps(cls.settings, indent=2), encoding="utf-8")
        except OSError:
            pass


State.load_settings()
app = FastAPI(title="imObj Labeler")


def proj() -> Project:
    if State.project is None:
        raise HTTPException(400, "No data folder open")
    return State.project


def _fail(e: Exception, code: int = 400):
    raise HTTPException(code, f"{type(e).__name__}: {e}")


# ------------------------------------------------------------------ models
class Params(BaseModel):
    channel: str | None = None
    flatten: str = "plane"
    cmap: str = "viridis"
    lo: float = 1.0
    hi: float = 99.0

    def rp(self) -> reader.RenderParams:
        if self.flatten not in reader.FLATTEN_MODES:
            raise HTTPException(400, f"flatten must be one of {reader.FLATTEN_MODES}")
        if self.cmap not in reader.COLORMAPS:
            raise HTTPException(400, f"cmap must be one of {reader.COLORMAPS}")
        if not (0 <= self.lo < self.hi <= 100):
            raise HTTPException(400, "need 0 <= lo < hi <= 100")
        return reader.RenderParams(self.channel, self.flatten, self.cmap, self.lo, self.hi)


class OpenReq(BaseModel):
    folder: str


class RegisterReq(BaseModel):
    name: str
    params: Params
    annotations: list[dict[str, Any]]
    source: str = "manual"
    overwrite: bool = True
    model: str = ""


class PredictReq(BaseModel):
    name: str
    params: Params
    model: str
    conf: float = Field(0.25, ge=0.0, le=1.0)
    iou: float = Field(0.7, ge=0.0, le=1.0)
    imgsz: int | None = None


class YamlReq(BaseModel):
    val_fraction: float = Field(0.0, ge=0.0, lt=1.0)
    seed: int = 0
    source: str = "combined"  # combined | registered | existing


# ------------------------------------------------------------------ state / folders
@app.get("/api/state")
def get_state():
    p = State.project
    return {"folder": str(p.data) if p else None,
            "out": str(p.out) if p else None,
            "settings": State.settings,
            "flatten_modes": reader.FLATTEN_MODES,
            "colormaps": reader.COLORMAPS}


def _remember_recent(key: str, path: str, keep: int = 8):
    recent = [p for p in State.settings.get(key, []) if p != path]
    State.save_settings(**{key: [path] + recent[: keep - 1]})


@app.post("/api/open")
def open_folder(req: OpenReq):
    try:
        State.project = Project(Path(req.folder).expanduser())
    except Exception as e:
        _fail(e)
    State.save_settings(last_folder=str(State.project.data))
    _remember_recent("recent_folders", str(State.project.data))
    return get_state()


class PickReq(BaseModel):
    initial: str = ""
    title: str = "Select folder"
    purpose: str = "data"  # "data" or "models": picks which remembered folder to start in


def _start_folder(initial: str, purpose: str) -> str:
    """Where a folder dialog opens: the given folder, else the last one used for this purpose."""
    key = "models_folder" if purpose == "models" else "last_folder"
    for cand in (initial, State.settings.get(key), *State.settings.get("recent_folders", [])):
        if cand and Path(cand).expanduser().is_dir():
            return str(Path(cand).expanduser())
    return ""


@app.post("/api/pick_folder")
def pick_folder(req: PickReq):
    """Native OS folder dialog (shown on the machine running the server)."""
    if not State.native_dialog:
        raise HTTPException(501, "Native folder dialog disabled")
    try:
        path = folders.pick_folder(_start_folder(req.initial, req.purpose), req.title)
    except RuntimeError as e:
        raise HTTPException(409 if "already open" in str(e) else 501, str(e))
    return {"path": path}


@app.get("/api/places")
def places():
    recent = [p for p in State.settings.get("recent_folders", []) if Path(p).is_dir()]
    models = State.settings.get("models_folder")
    return {"places": folders.places(), "recent": recent,
            "models_folder": models if models and Path(models).is_dir() else None}


@app.get("/api/browse")
def browse(path: str = ""):
    p = Path(path).expanduser() if path else Path(State.settings.get("last_folder") or Path.home())
    if not p.is_dir():
        p = Path.home()
    p = p.resolve()
    try:
        dirs = sorted((d.name for d in p.iterdir() if d.is_dir() and not d.name.startswith((".", "$"))),
                      key=reader.natural_key)
    except (PermissionError, OSError):
        dirs = []
    try:
        n_models = sum(1 for _ in p.glob("*.pt"))
    except OSError:
        n_models = 0
    return {"path": str(p), "name": p.name or str(p),
            "parent": str(p.parent) if p.parent != p else None,
            "dirs": dirs, "n_files": len(reader.list_files(p)), "n_models": n_models}


@app.get("/api/files")
def files():
    p = proj()
    st = p.status()
    return {"folder": str(p.data), "files": [{"name": n, "registered": r} for n, r in st.items()]}


# ------------------------------------------------------------------ images
def _params_from_query(channel, flatten, cmap, lo, hi) -> reader.RenderParams:
    return Params(channel=channel or None, flatten=flatten, cmap=cmap, lo=lo, hi=hi).rp()


@app.get("/api/file")
def file_info(name: str, channel: str | None = None, flatten: str = "plane",
              cmap: str = "viridis", lo: float = 1.0, hi: float = 99.0,
              use_registered: bool = True):
    """Render info + stored labels. If registered, its render settings win."""
    p = proj()
    try:
        path = p.path_of(name)
        reg_params = p.registered_params(name) if use_registered else None
        params = Params(**reg_params).rp() if reg_params else _params_from_query(channel, flatten, cmap, lo, hi)
        _, info = reader.render(path, params)
        labels = p.labels_for(name, info["width"], info["height"])
    except HTTPException:
        raise
    except Exception as e:
        _fail(e)
    params.channel = info.get("channel")
    return {"name": name, "info": info, "params": params.as_dict(),
            "registered": p.is_registered(name), "used_registered_params": bool(reg_params),
            "labels": labels, "stem": p.current_stem(name) or p.stem_for(name),
            "versions": p.versions(name), "has_existing": p.existing_label_path(name) is not None}


@app.get("/api/image")
def image(name: str, channel: str | None = None, flatten: str = "plane",
          cmap: str = "viridis", lo: float = 1.0, hi: float = 99.0):
    p = proj()
    try:
        rgb, _ = reader.render(p.path_of(name), _params_from_query(channel, flatten, cmap, lo, hi))
    except HTTPException:
        raise
    except Exception as e:
        _fail(e)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"})


# ------------------------------------------------------------------ labels
@app.post("/api/register")
def register(req: RegisterReq):
    p = proj()
    try:
        params = req.params.rp()
        rgb, info = reader.render(p.path_of(req.name), params)
        params.channel = info.get("channel")
        res = p.register(req.name, rgb, req.annotations, params, req.source, req.model,
                         overwrite=req.overwrite)
    except HTTPException:
        raise
    except Exception as e:
        _fail(e)
    return res


@app.delete("/api/register")
def unregister(name: str, all: bool = False):
    return {"removed": proj().unregister(name, all_versions=all)}


@app.get("/api/classes")
def get_classes():
    p = State.project
    cfg = p.classes() if p else cls_cfg.validate(cls_cfg.DEFAULT)
    return {**cfg, "exists": bool(p and p.classes_path.is_file())}


@app.put("/api/classes")
def put_classes(cfg: dict):
    try:
        return proj().save_classes(cfg)
    except ValueError as e:
        _fail(e)


@app.get("/api/dataset")
def dataset_summary():
    """Saved split settings and how many source files each dataset source would use."""
    p = proj()
    counts = {}
    for src in ("combined", "registered", "existing"):
        groups, stats = p.dataset_groups(src)
        counts[src] = {"files": len(groups), "images": sum(map(len, groups.values())),
                       "overridden": stats["n_overridden"]}
    return {"settings": p.split_settings(), "counts": counts, "aug": p.aug_summary(),
            "has_existing": p.sibling_labels_dir() is not None,
            "yaml_exists": p.yaml_path.is_file(), "out": str(p.out)}


@app.post("/api/data_yaml")
def data_yaml(req: YamlReq):
    try:
        return proj().write_data_yaml(req.val_fraction, req.seed, req.source)
    except ValueError as e:
        _fail(e)


class AugReq(BaseModel):
    split: YamlReq  # the split to augment (written first, so copies match what the dialog shows)
    transforms: list[str]
    noise_sigma: float = Field(10.0, gt=0, le=128)
    blur_radius: float = Field(1.0, gt=0, le=20)
    train: bool = True
    val: bool = True
    rot_noise_p: float = Field(0.0, ge=0.0, le=1.0)  # chance a rotated copy also gets noise


@app.post("/api/augment")
def run_augment(req: AugReq):
    p = proj()
    try:
        p.write_data_yaml(req.split.val_fraction, req.split.seed, req.split.source)
        return p.augment(req.transforms, req.noise_sigma, req.blur_radius, req.train, req.val,
                         req.rot_noise_p)
    except ValueError as e:
        _fail(e)


@app.delete("/api/augment")
def remove_augment():
    return proj().remove_augmentation()


# ------------------------------------------------------------------ training bundle
TRAIN_SCRIPT = Path(__file__).parent / "train_yolo.py"


@app.get("/api/bundle")
def bundle_info():
    """Defaults for the export dialog: what the current split would export."""
    from .project import TRAIN_DEFAULT

    p = proj()
    base = p.data.parent.name if p.data.name.lower() == "images" else p.data.name
    stamp = __import__("datetime").datetime.now().strftime("%Y%m%d-%H%M")
    return {**p.bundle_summary(), "defaults": TRAIN_DEFAULT,
            "suggested_name": f"{base}_{stamp}", "script": str(TRAIN_SCRIPT),
            "last_dest": State.settings.get("bundle_dest")}


class BundleReq(BaseModel):
    dest: str          # parent folder
    name: str
    config: dict[str, Any]


@app.post("/api/bundle")
def export_bundle(req: BundleReq):
    p = proj()
    name = req.name.strip()
    if not name or any(c in name for c in '\\/:*?"<>|') or name in (".", ".."):
        raise HTTPException(400, "Bundle name must be a plain folder name")
    try:
        res = p.export_bundle(Path(req.dest).expanduser() / name, req.config)
    except (ValueError, FileExistsError, OSError) as e:
        _fail(e)
    State.save_settings(bundle_dest=req.dest)
    return {**res, "script": str(TRAIN_SCRIPT)}


# ------------------------------------------------------------------ remote training (SSH)
class ServerReq(BaseModel):
    host: str
    root: str = "~/bundles"
    python: str = "python3"


def _server():
    from . import remote

    cfg = State.settings.get("remote")
    if not cfg or not cfg.get("host"):
        raise HTTPException(400, "Set up the training server first (Train… → Server)")
    try:
        return remote.Server(**cfg)
    except remote.RemoteError as e:
        raise HTTPException(400, str(e))


def _remote_call(fn, *args, **kw):
    from . import remote

    try:
        return fn(*args, **kw)
    except remote.RemoteError as e:
        raise HTTPException(502, str(e))


@app.get("/api/remote/settings")
def remote_settings():
    return State.settings.get("remote") or {"host": "", "root": "~/bundles", "python": "python3"}


@app.post("/api/remote/test")
def remote_test(req: ServerReq):
    from . import remote

    try:
        srv = remote.Server(req.host, req.root, req.python)
    except remote.RemoteError as e:
        raise HTTPException(400, str(e))
    State.save_settings(remote={"host": srv.host, "root": srv.root, "python": srv.python})
    return _remote_call(remote.test, srv)


@app.get("/api/remote/bundles")
def remote_bundles():
    from . import remote

    return {"bundles": _remote_call(remote.list_bundles, _server())}


class SendReq(BaseModel):
    path: str            # local bundle folder
    start: bool = True


@app.post("/api/remote/send")
def remote_send(req: SendReq):
    """Upload the script (if needed) and the bundle, run --check, then start training."""
    from . import remote

    srv = _server()
    steps = [_remote_call(remote.ensure_script, srv)]
    up = _remote_call(remote.upload, srv, Path(req.path))
    steps.append(f"uploaded {up['files']} files ({up['mb']} MB) to {up['remote']}")
    _remote_call(remote.check, srv, up["name"])
    steps.append("train_yolo.py --check passed")
    res = {"name": up["name"], "steps": steps}
    if req.start:
        st = _remote_call(remote.start, srv, up["name"])
        steps.append(f"training started (pid {st['pid']})")
        res["pid"] = st["pid"]
    return res


class RunReq(BaseModel):
    name: str
    resume: bool = False


@app.get("/api/remote/status")
def remote_status(name: str):
    from . import remote

    return _remote_call(remote.status, _server(), name)


@app.post("/api/remote/start")
def remote_start(req: RunReq):
    from . import remote

    return _remote_call(remote.start, _server(), req.name, req.resume)


@app.post("/api/remote/stop")
def remote_stop(req: RunReq):
    from . import remote

    return _remote_call(remote.stop, _server(), req.name)


class FetchReq(BaseModel):
    name: str
    run: str | None = None
    dest: str = ""          # default: the models folder (so Prediction mode finds best.pt)
    checkpoints: bool = False


@app.post("/api/remote/fetch")
def remote_fetch(req: FetchReq):
    from . import remote

    dest = req.dest or State.settings.get("models_folder") or ""
    if not dest:
        dest = str(proj().out / "trained_models")
    res = _remote_call(remote.fetch, _server(), req.name, Path(dest), req.run, req.checkpoints)
    return {**res, "dest": dest}


# ------------------------------------------------------------------ prediction
@app.get("/api/models")
def models(folder: str = ""):
    from .predict import find_models

    folder = folder or State.settings.get("models_folder", "")
    if folder:
        State.save_settings(models_folder=folder)
    return {"folder": folder, "models": find_models(folder) if folder else []}


@app.get("/api/model_info")
def model_info(path: str):
    try:
        from .predict import model_info as mi
        info = mi(path)
    except Exception as e:
        _fail(e)
    p = State.project
    if p:
        ours = cls_cfg.names(p.classes())
        info["class_mismatch"] = {str(k): v for k, v in info["names"].items() if ours.get(k) != v}
        info["classes_file_exists"] = p.classes_path.is_file()
    return info


@app.post("/api/classes/from_model")
def classes_from_model(path: str = Query(...)):
    from .predict import model_info as mi

    try:
        cfg = cls_cfg.from_names(mi(path)["names"])
        return proj().save_classes(cfg)
    except HTTPException:
        raise
    except Exception as e:
        _fail(e)


@app.post("/api/predict")
def run_predict(req: PredictReq):
    p = proj()
    try:
        from .predict import predict

        params = req.params.rp()
        rgb, info = reader.render(p.path_of(req.name), params)
        boxes = predict(req.model, rgb, req.conf, req.iou, req.imgsz)
    except HTTPException:
        raise
    except Exception as e:
        _fail(e)
    return {"annotations": boxes, "width": info["width"], "height": info["height"]}


# ------------------------------------------------------------------ static UI
@app.get("/")
def index():
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-store"})


app.mount("/", StaticFiles(directory=WEB), name="web")


@app.exception_handler(FileNotFoundError)
def _nf(_, e):
    return JSONResponse({"detail": f"Not found: {e}"}, status_code=404)
