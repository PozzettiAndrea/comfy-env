"""
macOS (Darwin) platform provider implementation.
"""

import os
import stat
import shutil
from pathlib import Path
from typing import Optional, Tuple

from .base import PlatformProvider, PlatformPaths


class DarwinPlatformProvider(PlatformProvider):
    """Platform provider for macOS systems."""

    @property
    def name(self) -> str:
        return 'darwin'

    @property
    def executable_suffix(self) -> str:
        return ''

    @property
    def shared_lib_extension(self) -> str:
        return '.dylib'

    def get_env_paths(self, env_dir: Path, python_version: str = "3.10") -> PlatformPaths:
        return PlatformPaths(
            python=env_dir / "bin" / "python",
            pip=env_dir / "bin" / "pip",
            site_packages=env_dir / "lib" / f"python{python_version}" / "site-packages",
            bin_dir=env_dir / "bin"
        )

    def check_prerequisites(self) -> Tuple[bool, Optional[str]]:
        # macOS with Apple Silicon can use MPS (Metal Performance Shaders)
        # but CUDA is not available
        return (True, None)

    def is_apple_silicon(self) -> bool:
        """Check if running on Apple Silicon."""
        import platform
        return platform.machine() == 'arm64'

    def make_executable(self, path: Path) -> None:
        current = os.stat(path).st_mode
        os.chmod(path, current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    def rmtree_robust(self, path: Path) -> bool:
        shutil.rmtree(path)
        return True
