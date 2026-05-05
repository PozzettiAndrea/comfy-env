"""Pixi binary location."""

import shutil
import sys
from pathlib import Path

_name = "pixi.exe" if sys.platform == "win32" else "pixi"
_default = Path.home() / ".pixi" / "bin" / _name
PIXI = str(_default) if _default.exists() else shutil.which("pixi")
