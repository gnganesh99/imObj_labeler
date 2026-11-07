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
    ap.add_argument("--no-native-dialog", action="store_true",
                    help="use the in-app folder picker instead of the OS dialog "
                         "(automatic when --host is not a local address)")
    a = ap.parse_args(argv)

    import uvicorn

    from . import server
    from .project import Project

    local = a.host in ("127.0.0.1", "localhost", "::1")
    server.State.native_dialog = local and not a.no_native_dialog

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
