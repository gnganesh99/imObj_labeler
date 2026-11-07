"""Folder selection helpers: native OS folder dialog and quick-access places."""
from __future__ import annotations

import os
import string
import sys
import threading
from pathlib import Path

_dialog_lock = threading.Lock()

# Windows known-folder GUIDs (resolves OneDrive-redirected Desktop/Documents correctly)
_KNOWN = {
    "Desktop": "B4BFCC3A-DB2C-424C-B029-7FE99A87C641",
    "Documents": "FDD39AD0-238F-46AF-ADB4-6C85480369C7",
    "Downloads": "374DE290-123F-4565-9164-39C4925E467B",
    "Pictures": "33E28130-4E1E-4676-835A-98395C3BC3BB",
}


def _known_folder(guid: str) -> str | None:
    import ctypes
    import uuid
    from ctypes import wintypes

    class GUID(ctypes.Structure):
        _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                    ("Data3", wintypes.WORD), ("Data4", ctypes.c_ubyte * 8)]

    g = GUID.from_buffer_copy(uuid.UUID(guid).bytes_le)
    p = ctypes.c_wchar_p()
    if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(g), 0, None, ctypes.byref(p)) != 0:
        return None
    try:
        return p.value
    finally:
        ctypes.windll.ole32.CoTaskMemFree(p)


def places() -> list[dict]:
    """Home, Desktop, Documents, Downloads, Pictures and drives that exist."""
    home = Path.home()
    out = [{"name": "Home", "path": str(home), "kind": "home"}]
    for name, guid in _KNOWN.items():
        path = None
        if sys.platform == "win32":
            try:
                path = _known_folder(guid)
            except Exception:
                path = None
        path = Path(path) if path else home / name
        if path.is_dir():
            out.append({"name": name, "path": str(path), "kind": name.lower()})
    seen = set()
    for var in ("OneDriveCommercial", "OneDrive"):
        od = os.environ.get(var)
        if od and od not in seen and Path(od).is_dir():
            seen.add(od)
            out.append({"name": Path(od).name, "path": od, "kind": "cloud"})
    if sys.platform == "win32":
        for d in string.ascii_uppercase:
            root = f"{d}:\\"
            if os.path.exists(root):
                out.append({"name": f"{d}:", "path": root, "kind": "drive"})
    else:
        out.append({"name": "/", "path": "/", "kind": "drive"})
    return out


def pick_folder(initial: str = "", title: str = "Select folder") -> str | None:
    """Open the native folder dialog on this machine. Returns the path, or None if cancelled.

    Raises RuntimeError when no GUI is available (e.g. headless/remote server)."""
    if not _dialog_lock.acquire(blocking=False):
        raise RuntimeError("A folder dialog is already open")
    try:
        try:
            import tkinter as tk
            from tkinter import filedialog
        except ImportError as e:
            raise RuntimeError("tkinter is not available") from e
        try:
            root = tk.Tk()
        except tk.TclError as e:
            raise RuntimeError(f"No display available for a native dialog: {e}") from e
        try:
            root.withdraw()
            root.attributes("-topmost", True)  # keep the dialog in front of the browser
            root.update()
            init = initial if initial and Path(initial).is_dir() else str(Path.home())
            path = filedialog.askdirectory(parent=root, initialdir=init, title=title, mustexist=True)
        finally:
            root.destroy()
        return str(Path(path)) if path else None
    finally:
        _dialog_lock.release()
