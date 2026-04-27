"""Filesystem, platform, logging, and subprocess utilities for install/."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable

USE_COMFY_ENV_VAR = "USE_COMFY_ENV"


def _rmtree(path) -> None:
    """rmtree that handles read-only files and long paths on Windows."""
    import shutil
    if sys.platform == "win32":
        import subprocess, tempfile
        target = str(Path(path).resolve())
        empty = tempfile.mkdtemp()
        try:
            subprocess.run(
                ["robocopy", empty, target, "/MIR", "/W:0", "/R:0"],
                capture_output=True,
            )
            shutil.rmtree(target, ignore_errors=True)
        finally:
            shutil.rmtree(empty, ignore_errors=True)
    else:
        shutil.rmtree(path)


def _is_comfy_env_enabled() -> bool:
    return os.environ.get(USE_COMFY_ENV_VAR, "1").lower() not in ("0", "false", "no", "off")


def _enable_windows_long_paths(log: Callable[[str], None]) -> None:
    """Enable Windows long path support via registry (requires admin)."""
    if sys.platform != "win32":
        return
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\FileSystem",
            0, winreg.KEY_SET_VALUE
        )
        winreg.SetValueEx(key, "LongPathsEnabled", 0, winreg.REG_DWORD, 1)
        winreg.CloseKey(key)
        log("[comfy-env] Enabled Windows long path support")
    except PermissionError:
        log("[comfy-env] WARNING: Could not enable long paths (needs admin)")
    except Exception:
        pass


def _patch_uv_platform_py(log: Callable[[str], None] = print) -> None:
    """Patch uv-managed Python's platform.py to handle conda-forge version strings.

    conda-forge Python embeds '| packaged by conda-forge |' in sys.version.
    When pixi's uv creates build-isolation venvs it may use a standard CPython
    whose platform.py can't parse that string, crashing setuptools.  Apply the
    same one-line regex fix that conda-forge ships in their own builds.
    """
    if sys.platform != "win32":
        return
    search_dirs = [
        Path.home() / "AppData" / "Roaming" / "uv" / "python",
        Path.home() / "AppData" / "Local" / "rattler" / "cache" / "python",
    ]
    MARKER = r"r'([\w.+]+)\s*'"
    REPLACEMENT = r"r'([\w.+]+)\s*(?:\ \|\ packaged\ by\ conda\-forge\ \|)?\s*'"
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for py_dir in search_dir.iterdir():
            if not py_dir.name.startswith("cpython-"):
                continue
            platform_py = py_dir / "Lib" / "platform.py"
            if not platform_py.exists():
                continue
            content = platform_py.read_text(encoding="utf-8")
            if "packaged by conda" in content:
                continue
            idx = content.find(MARKER)
            if idx == -1:
                continue
            patched = content[:idx] + REPLACEMENT + content[idx + len(MARKER):]
            platform_py.write_text(patched, encoding="utf-8")
            log(f"[comfy-env] Patched {platform_py} for conda-forge compat")


def _find_uv() -> str:
    """Find the uv binary installed alongside comfy-env."""
    import shutil
    exe_dir = Path(sys.executable).parent
    uv_name = "uv.exe" if sys.platform == "win32" else "uv"
    candidate = exe_dir / uv_name
    if candidate.exists():
        return str(candidate)
    if sys.platform == "win32":
        candidate = exe_dir / "Scripts" / uv_name
        if candidate.exists():
            return str(candidate)
    uv = shutil.which("uv")
    if uv:
        return uv
    raise FileNotFoundError("uv binary not found")


def _make_tee_log(log_callback: Callable[[str], None], log_path: Path) -> Callable[[str], None]:
    """Tee logs to both the original callback and a file."""
    import datetime
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "w", encoding="utf-8")
    fh.write(f"# comfy-env install log - {datetime.datetime.now().isoformat()}\n")
    fh.write(f"# Python: {sys.executable} ({sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro})\n")
    fh.write(f"# Platform: {sys.platform}\n\n")
    fh.flush()

    def tee(msg):
        log_callback(msg)
        sys.stdout.flush()
        fh.write(msg + "\n")
        fh.flush()

    tee.file = fh
    tee.close = fh.close
    tee.path = log_path
    return tee


def _log_subprocess(log: Callable, result, label: str = "") -> None:
    """Write subprocess stdout/stderr to the log file (verbose, file-only)."""
    fh = getattr(log, "file", None)
    if fh is None:
        return
    if label:
        fh.write(f"\n--- {label} (exit {result.returncode}) ---\n")
    if result.stdout and result.stdout.strip():
        fh.write(f"[stdout]\n{result.stdout}\n")
    if result.stderr and result.stderr.strip():
        fh.write(f"[stderr]\n{result.stderr}\n")
    fh.flush()


def _run_streaming(cmd, log: Callable, cwd=None, env=None):
    """Run a subprocess, streaming stdout/stderr lines to log in real time."""
    import subprocess
    import threading

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    def _read_stderr():
        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_lines.append(line)
            line_text = line.rstrip("\n")
            if line_text.strip():
                log(f"  {line_text}")

    t = threading.Thread(target=_read_stderr, daemon=True)
    t.start()

    assert proc.stdout is not None
    for line in proc.stdout:
        line_text = line.rstrip("\n")
        stdout_lines.append(line_text)
        if line_text.strip():
            log(f"  {line_text}")

    proc.wait()
    t.join(timeout=5)

    return subprocess.CompletedProcess(
        cmd, proc.returncode,
        "\n".join(stdout_lines),
        "".join(stderr_lines),
    )
