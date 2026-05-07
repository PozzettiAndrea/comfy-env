"""Pixi binary management. Installs to ~/.pixi/bin/ if missing."""

import os
import platform
import stat
import sys
import urllib.request
from pathlib import Path

_name = "pixi.exe" if sys.platform == "win32" else "pixi"
PIXI_HOME = Path.home() / ".pixi"
PIXI = str(PIXI_HOME / "bin" / _name)

_URLS = {
    ("Linux", "x86_64"):   "https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-unknown-linux-musl",
    ("Linux", "aarch64"):  "https://github.com/prefix-dev/pixi/releases/latest/download/pixi-aarch64-unknown-linux-musl",
    ("Darwin", "x86_64"):  "https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-apple-darwin",
    ("Darwin", "arm64"):   "https://github.com/prefix-dev/pixi/releases/latest/download/pixi-aarch64-apple-darwin",
    ("Windows", "AMD64"):  "https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-pc-windows-msvc.exe",
}


def ensure_pixi():
    """Ensure pixi is installed at ~/.pixi/bin/pixi. Downloads if missing."""
    if Path(PIXI).exists():
        return PIXI

    key = (platform.system(), platform.machine())
    url = _URLS.get(key)
    if not url:
        raise RuntimeError(f"No pixi binary for {key[0]}/{key[1]}")

    print(f"[comfy-env] pixi not found, downloading...", file=sys.stderr, flush=True)
    dest = Path(PIXI)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)

    if sys.platform != "win32":
        dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    print(f"[comfy-env] pixi installed: {PIXI}", file=sys.stderr, flush=True)
    return PIXI
