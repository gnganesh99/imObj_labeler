"""Run training bundles on a Linux GPU machine over SSH.

Uses the system ``ssh`` with the user's own keys and ``~/.ssh/config`` (BatchMode: it never
prompts for or stores a password). Everything on the server lives in one folder (``root``,
default ``~/bundles``)::

    ~/bundles/train_yolo.py          the common training script (uploaded/updated by the app)
    ~/bundles/<bundle>/              an uploaded bundle
        train.pid  train.log         process id + console output of the detached run
        runs/<name>_<time>/          ultralytics output (results.csv, weights/best.pt, ...)

Training is started detached (nohup, stdin/stdout closed), so it doesn't depend on the app
or the SSH connection. Progress is read from files the run writes anyway (results.csv,
train.log); reading never touches the training process.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path

SCRIPT = Path(__file__).parent / "train_yolo.py"
# IMOBJ_SSH='["C:/path/ssh.exe", "-o", "BatchMode=yes"]' (a JSON list) replaces the ssh command,
# e.g. to use a different ssh client; tests use it to point at a fake server.
SSH_COMMAND = (json.loads(os.environ["IMOBJ_SSH"]) if os.environ.get("IMOBJ_SSH")
               else ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"])
_HOST_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._@:-]*$")
_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]*$")


class RemoteError(RuntimeError):
    pass


@dataclass
class Server:
    host: str                       # ssh host or alias from ~/.ssh/config
    root: str = "~/bundles"         # folder on the server for bundles + script
    python: str = "python3"         # python with ultralytics + CUDA torch, e.g. a venv's bin/python

    def __post_init__(self):
        self.host, self.root, self.python = self.host.strip(), self.root.strip(), self.python.strip()
        if not _HOST_RE.match(self.host):
            raise RemoteError("Host must be an ssh host name or alias (e.g. myvm or user@host)")
        if not self.root or not self.python:
            raise RemoteError("Remote folder and python must not be empty")

    @property
    def script(self) -> str:
        return f"{rq(self.root)}/train_yolo.py"

    @property
    def py(self) -> str:
        return rq(self.python)


def rq(path: str) -> str:
    """Quote a remote path for sh, keeping a leading ~/ expandable."""
    path = path.strip()
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        return '"$HOME"/' + shlex.quote(path[2:])
    return shlex.quote(path)


def check_name(name: str) -> str:
    if not _NAME_RE.match(name or ""):
        raise RemoteError(f"Not a valid bundle name: {name!r}")
    return name


def ssh(srv: Server, command: str, *, stdin=None, timeout: float | None = 60,
        check: bool = True) -> subprocess.CompletedProcess:
    """Run ``command`` (a sh script) on the server."""
    args = [*SSH_COMMAND, srv.host, command]
    try:
        p = subprocess.run(args, input=stdin, capture_output=True, timeout=timeout)
    except FileNotFoundError as e:
        raise RemoteError("ssh is not installed on this computer") from e
    except subprocess.TimeoutExpired as e:
        raise RemoteError(f"ssh {srv.host} timed out after {timeout:.0f} s") from e
    if check and p.returncode != 0:
        err = (p.stderr or b"").decode(errors="replace").strip()
        if p.returncode == 255:
            raise RemoteError(f"Cannot connect to {srv.host}: {err or 'ssh failed'} "
                              "(the app needs key-based login; it never asks for a password)")
        raise RemoteError(err or (p.stdout or b"").decode(errors="replace").strip()
                          or f"remote command failed ({p.returncode})")
    return p


def out(p: subprocess.CompletedProcess) -> str:
    return (p.stdout or b"").decode(errors="replace")


# ------------------------------------------------------------------ server checks
def local_script_version() -> str:
    m = re.search(r'^SCRIPT_VERSION = "([^"]+)"', SCRIPT.read_text(encoding="utf-8"), re.M)
    return m.group(1) if m else "?"


def test(srv: Server) -> dict:
    """Connection, python, torch/CUDA/GPU, ultralytics and script version on the server."""
    probe = ("import json, platform\n"
             "r = {'python': platform.python_version()}\n"
             "try:\n import torch; r['torch'] = torch.__version__; r['cuda'] = torch.cuda.is_available()\n"
             " r['gpus'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]\n"
             "except Exception as e: r['torch_error'] = str(e)\n"
             "try:\n import ultralytics; r['ultralytics'] = ultralytics.__version__\n"
             "except Exception as e: r['ultralytics_error'] = str(e)\n"
             "print(json.dumps(r))\n")
    cmd = (f"echo; echo @@host; uname -n; echo; echo @@probe; {srv.py} -c {shlex.quote(probe)} 2>&1; "
           f"echo; echo @@script; [ -f {srv.script} ] && {srv.py} {srv.script} --version 2>&1; "
           f"echo; echo @@root; [ -d {rq(srv.root)} ] && echo yes; true")
    sec = sections(out(ssh(srv, cmd, timeout=90)))
    info: dict = {"host": sec.get("host", "").strip()}
    try:
        info.update(json.loads(sec.get("probe", "").strip().splitlines()[-1]))
    except (ValueError, IndexError):
        info["python_error"] = sec.get("probe", "").strip()[-400:] or f"{srv.python} did not run"
    m = re.search(r"train_yolo\.py (\S+)", sec.get("script", ""))
    info["script_version"] = m.group(1) if m else None
    info["local_script_version"] = local_script_version()
    info["root_exists"] = sec.get("root", "").strip() == "yes"
    problems = []
    if "python_error" in info:
        problems.append(f"python not usable: {info['python_error']}")
    if info.get("torch_error") or "torch" not in info and "python_error" not in info:
        problems.append(f"torch missing: {info.get('torch_error', '')}")
    elif "torch" in info and not info.get("cuda"):
        problems.append("torch has no CUDA (CPU-only torch, or no GPU visible)")
    if info.get("ultralytics_error"):
        problems.append(f"ultralytics missing: {info['ultralytics_error']}")
    info["problems"] = problems
    return info


_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


_BAR = re.compile(r"(\d{1,3})%\|")  # tqdm progress bar: " 67%|######6   | 4/6 [...]"


def clean_log(text: str) -> str:
    """Console log as it looked on screen: no colour codes, each progress bar at its last state.

    Written to a file, tqdm puts every update of a bar on its own line (or after a \\r).
    Consecutive updates of the same bar (same first word, e.g. the epoch "2/300" or "Class",
    and a percentage that doesn't go back) are collapsed into the latest one."""
    lines = [_ANSI.sub("", line.split("\r")[-1] if line.strip("\r") else "").rstrip()
             for line in text.replace("\r\n", "\n").split("\n")]
    out: list[str] = []
    prev_key, prev_pct = None, -1
    for line in lines:
        m = _BAR.search(line)
        if m:
            # a bar printed before its first update has no label ("  0%|   | 0/6"): key ""
            key = "" if line.lstrip().startswith(m.group(0)) else line.split(maxsplit=1)[0]
            pct = int(m.group(1))
            if out and prev_key is not None and (prev_key == "" or (key == prev_key and pct >= prev_pct)):
                out[-1] = line  # same bar, further along: replace
            else:
                out.append(line)
            prev_key, prev_pct = key, pct
        else:
            out.append(line)
            prev_key, prev_pct = None, -1
    return "\n".join(out).strip("\n")


def sections(text: str) -> dict[str, str]:
    """Split output made of ``@@name`` marker lines into {name: text}."""
    res, cur = {}, None
    for line in text.splitlines():
        if line.startswith("@@"):
            cur = line[2:].strip()
            res[cur] = ""
        elif cur is not None:
            res[cur] += line + "\n"
    return res


def ensure_script(srv: Server) -> str:
    """Upload train_yolo.py unless the server's copy is byte-identical (checksum)."""
    p = ssh(srv, f"[ -f {srv.script} ] && {srv.py} {srv.script} --version; echo; echo @@sha; "
                 f"sha256sum {srv.script} 2>/dev/null", check=False)
    m = re.search(r"train_yolo\.py (\S+)", out(p))
    have, want = (m.group(1) if m else None), local_script_version()
    remote_sha = (sections(out(p)).get("sha") or "").split()[:1]
    if remote_sha == [hashlib.sha256(SCRIPT.read_bytes()).hexdigest()]:
        return f"train_yolo.py {want} already on the server"
    ssh(srv, f"mkdir -p {rq(srv.root)} && cat > {srv.script}.tmp && mv {srv.script}.tmp {srv.script}",
        stdin=SCRIPT.read_bytes())
    return f"uploaded train_yolo.py {want}" + (f" (replaced {have})" if have else "")


# ------------------------------------------------------------------ bundles
def upload(srv: Server, bundle: Path) -> dict:
    """Copy a local bundle to ``root/<name>`` (one tar stream; local runs/ are skipped)."""
    bundle = Path(bundle)
    name = check_name(bundle.name)
    if not (bundle / "train_config.json").is_file():
        raise RemoteError(f"{bundle} is not a training bundle")
    dest = f"{rq(srv.root)}/{shlex.quote(name)}"
    if out(ssh(srv, f"[ -e {dest} ] && echo exists; true")).strip() == "exists":
        raise RemoteError(f"{srv.root}/{name} already exists on the server; export the bundle "
                          "under a new name")
    buf = io.BytesIO()
    n = 0
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=1) as tar:  # PNGs barely compress
        for f in sorted(bundle.rglob("*")):
            rel = f.relative_to(bundle)
            if rel.parts[0] == "runs" or not f.is_file():
                continue
            tar.add(f, arcname=f"{name}/{rel.as_posix()}", recursive=False)
            n += 1
    data = buf.getvalue()
    ssh(srv, f"mkdir -p {rq(srv.root)} && cd {rq(srv.root)} && tar -xzf -", stdin=data,
        timeout=None)
    return {"name": name, "files": n, "mb": round(len(data) / 1e6, 1),
            "remote": f"{srv.root}/{name}"}


def _bundle_dir(srv: Server, name: str) -> str:
    return f"{rq(srv.root)}/{shlex.quote(check_name(name))}"


def check(srv: Server, name: str) -> str:
    """train_yolo.py --check on the server (data + CUDA); raises with its output on failure."""
    d = _bundle_dir(srv, name)
    p = ssh(srv, f"cd {d} && {srv.py} {srv.script} \"$PWD\" --check 2>&1", check=False,
            timeout=300)
    if p.returncode != 0:
        raise RemoteError("train_yolo.py --check failed:\n" + out(p).strip()[-1500:])
    return out(p).strip()


def _running(d: str) -> str:
    return f'[ -f {d}/train.pid ] && kill -0 "$(cat {d}/train.pid)" 2>/dev/null'


def start(srv: Server, name: str, resume: bool = False, extra: list[str] | None = None) -> dict:
    """Start training detached. It keeps running after the app or SSH connection goes away."""
    d = _bundle_dir(srv, name)
    if ssh(srv, _running(d), check=False).returncode == 0:
        raise RemoteError(f"{name} is already training on the server")
    args = " ".join(shlex.quote(a) for a in (["--resume"] if resume else []) + (extra or []))
    # ';' not '&&' before nohup: "a && b &" would background a subshell, and $! would be its pid
    # leading echo: a stopped run can leave its last progress bar without a newline
    cmd = (f"cd {d} || exit 1; [ -s train.log ] && echo >> train.log; "
           f"echo \"=== $(date '+%Y-%m-%d %H:%M:%S') start {args}\" >> train.log; "
           f"nohup {srv.py} {srv.script} \"$PWD\" {args} >> train.log 2>&1 < /dev/null & "
           "echo $! > train.pid; cat train.pid")
    pid = out(ssh(srv, cmd)).strip()
    return {"name": name, "pid": pid, "resume": resume}


def stop(srv: Server, name: str) -> dict:
    """Stop a run (its last.pt stays, so it can be resumed)."""
    d = _bundle_dir(srv, name)
    cmd = (f"{_running(d)} || {{ echo notrunning; exit 0; }}; pid=$(cat {d}/train.pid); "
           f"pkill -TERM -P $pid 2>/dev/null; kill -TERM $pid; echo stopped")
    return {"name": name, "result": out(ssh(srv, cmd)).strip()}


def status(srv: Server, name: str, log_lines: int = 30) -> dict:
    """Progress of the latest run of a bundle, read from its files."""
    d = _bundle_dir(srv, name)
    cmd = (f"cd {d} 2>/dev/null || {{ echo; echo @@missing; exit 0; }}; "
           f"echo; echo @@state; if {_running(d)}; then echo running; else echo idle; fi; "
           "run=$(ls -td runs/*/ 2>/dev/null | head -n 1); echo; echo @@run; echo \"$run\"; "
           "echo; echo @@config; cat \"${run}train_config_used.json\" 2>/dev/null || cat train_config.json; "
           "echo; echo @@results; cat \"${run}results.csv\" 2>/dev/null; "
           "echo; echo @@summary; cat \"${run}training_data.json\" 2>/dev/null; "
           "echo; echo @@best; [ -f \"${run}weights/best.pt\" ] && echo yes; "
           f"echo; echo @@log; tail -n {int(log_lines)} train.log 2>/dev/null; true")
    sec = sections(out(ssh(srv, cmd)))
    if "missing" in sec:
        raise RemoteError(f"{srv.root}/{name} not found on the server")
    res: dict = {"name": name, "running": sec.get("state", "").strip() == "running",
                 "run": sec.get("run", "").strip().rstrip("/").split("/")[-1] or None,
                 "has_best": sec.get("best", "").strip() == "yes",
                 "log": clean_log(sec.get("log", ""))}
    try:
        res["epochs"] = int(json.loads(sec.get("config") or "{}").get("epochs") or 0)
    except ValueError:
        res["epochs"] = None
    try:
        res["summary"] = json.loads(sec["summary"]) if sec.get("summary", "").strip() else None
    except ValueError:
        res["summary"] = None
    rows = [r for r in csv.DictReader(io.StringIO(sec.get("results", "")))]
    rows = [{k.strip(): v.strip() for k, v in r.items() if k} for r in rows]
    hist = []
    for r in rows:
        try:
            hist.append({"epoch": int(float(r.get("epoch", 0))),
                         "map50": float(r.get("metrics/mAP50(B)", "nan")),
                         "map": float(r.get("metrics/mAP50-95(B)", "nan")),
                         "box_loss": float(r.get("val/box_loss", "nan"))})
        except ValueError:
            continue
    res["history"] = hist
    res["epoch"] = hist[-1]["epoch"] if hist else 0
    res["best"] = max(hist, key=lambda h: h["map"]) if hist else None
    if res["running"]:
        res["state"] = "running"
    elif res["summary"]:
        res["state"] = "finished"
    elif res["run"]:
        res["state"] = "stopped"  # killed, crashed or interrupted: see the log; --resume continues
    else:
        res["state"] = "not started"
    return res


def list_bundles(srv: Server) -> list[dict]:
    """Bundles in root with a quick state each."""
    cmd = (f"cd {rq(srv.root)} 2>/dev/null || exit 0; for d in */; do d=${{d%/}}; "
           "[ -f \"$d/train_config.json\" ] || continue; s=idle; "
           "[ -f \"$d/train.pid\" ] && kill -0 \"$(cat \"$d/train.pid\")\" 2>/dev/null && s=running; "
           "r=$(ls -td \"$d\"/runs/*/ 2>/dev/null | head -n 1); f=no; "
           "[ -n \"$r\" ] && [ -f \"${r}training_data.json\" ] && f=yes; "
           "t=$(stat -c %Y \"$d/train_config.json\"); echo \"$d|$s|$(basename \"$r\" 2>/dev/null)|$f|$t\"; done")
    res = []
    for line in out(ssh(srv, cmd)).splitlines():
        parts = line.split("|")
        if len(parts) != 5:
            continue
        name, s, run, finished, t = parts
        state = "running" if s == "running" else "finished" if finished == "yes" else \
            "stopped" if run else "not started"
        res.append({"name": name, "state": state, "run": run or None,
                    "time": int(t) if t.isdigit() else 0})
    return sorted(res, key=lambda r: -r["time"])


def fetch(srv: Server, name: str, dest_parent: Path, run: str | None = None,
          checkpoints: bool = False) -> dict:
    """Download a run folder to ``dest_parent/<bundle>/<run>`` (epoch*.pt skipped by default)."""
    d = _bundle_dir(srv, name)
    if run is None:
        run = out(ssh(srv, f"cd {d}/runs 2>/dev/null && ls -td */ | head -n 1")).strip().rstrip("/")
    if not run or not _NAME_RE.match(run):
        raise RemoteError(f"No run found for {name}")
    excl = "" if checkpoints else "--exclude='epoch*.pt'"
    p = ssh(srv, f"cd {d}/runs && tar {excl} -czf - {shlex.quote(run)}", timeout=None)
    dest = Path(dest_parent).expanduser() / name
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(p.stdout), mode="r:gz") as tar:
        tar.extractall(dest, filter="data")  # refuses absolute paths / links outside dest
    best = dest / run / "weights" / "best.pt"
    return {"name": name, "run": run, "path": str(dest / run),
            "best": str(best) if best.is_file() else None, "mb": round(len(p.stdout) / 1e6, 1)}
