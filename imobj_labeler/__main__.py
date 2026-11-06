"""Launch the labeler: ``python -m imobj_labeler [data_folder] [--models DIR]``."""
from __future__ import annotations

import argparse
import threading
import webbrowser
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="imobj_labeler", description=__doc__)
    ap.add_argument("folder", nargs="?", help="data folder to open")
    ap.add_argument("--models", help="folder containing YOLO .pt models")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)

    import uvicorn

    from . import server
    from .project import Project

    if a.folder:
        server.State.project = Project(Path(a.folder))
        server.State.save_settings(last_folder=str(server.State.project.data))
    if a.models:
        server.State.save_settings(models_folder=str(Path(a.models).resolve()))

    url = f"http://{a.host}:{a.port}/"
    if not a.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    print(f"imObj Labeler running at {url}  (Ctrl+C to stop)")
    uvicorn.run(server.app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
