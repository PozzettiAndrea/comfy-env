"""Pixi binary management. Installs to ~/.pixi/bin/ via official installer if missing."""

import os
import subprocess
import sys
from pathlib import Path

_name = "pixi.exe" if sys.platform == "win32" else "pixi"
PIXI_HOME = Path.home() / ".pixi"
PIXI = str(PIXI_HOME / "bin" / _name)


def ensure_pixi():
    """Ensure pixi is installed at ~/.pixi/bin/pixi. Downloads if missing."""
    if Path(PIXI).exists():
        return PIXI
    print("[comfy-env] pixi not found, installing...", file=sys.stderr, flush=True)
    env = os.environ.copy()
    env["PIXI_HOME"] = str(PIXI_HOME)
    env["PIXI_NO_PATH_UPDATE"] = "1"
    subprocess.run(
        ["sh", "-c", "curl -fsSL https://pixi.sh/install.sh | sh"],
        env=env, check=True,
    )
    if not Path(PIXI).exists():
        raise RuntimeError(f"pixi install failed — expected binary at {PIXI}")
    print(f"[comfy-env] pixi installed: {PIXI}", file=sys.stderr, flush=True)
    return PIXI
