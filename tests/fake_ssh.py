"""Stand-in for ``ssh`` in tests: runs the remote command in a local bash.

Usage (as remote.SSH_COMMAND): python fake_ssh.py HOST COMMAND. The "server" home folder is
$FAKE_SSH_HOME; any -o options are ignored.
"""
import os
import shutil
import subprocess
import sys

args = [a for a in sys.argv[1:]]
while args and args[0] == "-o":
    args = args[2:]
host, command = args[0], args[-1]
if host == "unreachable":
    sys.stderr.write("ssh: connect to host unreachable port 22: Connection refused\n")
    sys.exit(255)
bash = os.environ.get("FAKE_SSH_BASH") or shutil.which("bash")
env = {**os.environ, "HOME": os.environ["FAKE_SSH_HOME"]}
# a real server has ls, uname, stat... on PATH; with Git Bash they sit next to bash.exe
env["PATH"] = os.path.dirname(bash) + os.pathsep + env.get("PATH", "")
p = subprocess.run([bash, "-c", command], stdin=sys.stdin.buffer, env=env)
sys.exit(p.returncode)
