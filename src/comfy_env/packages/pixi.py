"""Pixi interface. Pixi is a pip dependency — binary lives next to sys.executable."""

import subprocess
import sys
from pathlib import Path

PIXI = str(Path(sys.executable).parent / ("pixi.exe" if sys.platform == "win32" else "pixi"))
