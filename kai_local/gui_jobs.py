"""Process isolation keeps TensorFlow off the desktop's UI thread."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def console_python():
    executable = Path(sys.executable)
    # pythonw has no console streams; the worker needs python.exe + redirected pipes.
    candidate = executable.with_name("python.exe")
    return str(candidate if executable.name.lower() == "pythonw.exe" and candidate.is_file() else executable)


def command(root, action, category, paths, labels=None, **options):
    args = [console_python(), "-u", "-B", "-X", "faulthandler", str(Path(root) / "app.py"), action, "--category", category, "--data", *map(str, paths)]
    if labels:
        args += ["--labels", *labels]
    for key, value in options.items():
        args += ["--" + key.replace("_", "-"), str(value)]
    return args


def run_audit(root, args):
    result = subprocess.run(args, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"}, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        report = json.loads(result.stdout)
    except ValueError as exc:
        raise ValueError(result.stderr.strip() or result.stdout.strip() or "El intérprete no devolvió una auditoría válida.") from exc
    if result.returncode not in (0, 2) or not isinstance(report, dict) or "ready_to_train" not in report:
        raise ValueError(result.stderr.strip() or "La auditoría terminó de forma inesperada.")
    return report


def epoch_from_line(line):
    match = re.search(r"Epoch (\d+)/(\d+)", line)
    return tuple(map(int, match.groups())) if match else None


def run_training(root, args, emit, log_path):
    """Stream stdout+stderr; never use shell=True or terminate the process to cancel."""
    with Path(log_path).open("x", encoding="utf-8") as log:
        with subprocess.Popen(args, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1, env={**os.environ, "PYTHONIOENCODING": "utf-8"}, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                emit("line", line)
            return process.wait()
