"""Pixi binary location. Installed to ~/.pixi/bin by the official installer."""

import sys
from pathlib import Path

PIXI = str(Path.home() / ".pixi" / "bin" / ("pixi.exe" if sys.platform == "win32" else "pixi"))
