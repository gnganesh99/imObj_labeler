"""FastAPI backend: serves the browser UI and the JSON/PNG API."""
from __future__ import annotations

import io
import json
import os
import string
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from . import classes as cls_cfg
from . import reader
from .project import Project

WEB = Path(__file__).parent / "web"
SETTINGS_PATH = Path.home() / ".imobj_labeler" / "settings.json"


class State:
    project: Project | None = None
    settings: dict = {}

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


# ------------------------------------------------------------------ state / folders
@app.get("/api/state")
def get_state():
    p = State.project
    return {"folder": str(p.data) if p else None,
            "out": str(p.out) if p else None,
            "settings": State.settings,
            "flatten_modes": reader.FLATTEN_MODES,
            "colormaps": reader.COLORMAPS}


@app.post("/api/open")
def open_folder(req: OpenReq):
    try:
        State.project = Project(Path(req.folder).expanduser())
    except Exception as e:
        _fail(e)
    State.save_settings(last_folder=str(State.project.data))
    return get_state()


@app.get("/api/browse")
def browse(path: str = ""):
    drives = []
    if os.name == "nt":
        drives = [f"{d}:\\" for d in string.ascii_uppercase if Path(f"{d}:\\").exists()]
    p = Path(path).expanduser() if path else Path(State.settings.get("last_folder") or Path.home())
    if not p.is_dir():
        p = Path.home()
    p = p.resolve()
    try:
        dirs = sorted((d.name for d in p.iterdir() if d.is_dir() and not d.name.startswith(".")),
                      key=reader.natural_key)
    except PermissionError:
        dirs = []
    return {"path": str(p), "parent": str(p.parent) if p.parent != p else None,
            "dirs": dirs, "n_files": len(reader.list_files(p)), "drives": drives}


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
            "labels": labels, "stem": p.stem_for(name)}


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
        res = p.register(req.name, rgb, req.annotations, params, req.source, req.model)
    except HTTPException:
        raise
    except Exception as e:
        _fail(e)
    return res


@app.delete("/api/register")
def unregister(name: str):
    return {"removed": proj().unregister(name)}


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


@app.post("/api/data_yaml")
def data_yaml(req: YamlReq):
    p = proj()
    if not p.out.is_dir():
        raise HTTPException(400, "Nothing registered yet")
    return p.write_data_yaml(req.val_fraction, req.seed)


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
