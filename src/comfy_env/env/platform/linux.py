"""
Linux platform provider implementation.
"""

import os
import stat
import shutil
from pathlib import Path
from typing import Optional, Tuple

from .base import PlatformProvider, PlatformPaths


class LinuxPlatformProvider(PlatformProvider):
    """Platform provider for Linux systems."""

    @property
    def name(self) -> str:
        return 'linux'

    @property
    def executable_suffix(self) -> str:
        return ''

    @property
    def shared_lib_extension(self) -> str:
        return '.so'

    def get_env_paths(self, env_dir: Path, python_version: str = "3.10") -> PlatformPaths:
        return PlatformPaths(
            python=env_dir / "bin" / "python",
            pip=env_dir / "bin" / "pip",
            site_packages=env_dir / "lib" / f"python{python_version}" / "site-packages",
            bin_dir=env_dir / "bin"
        )

    def check_prerequisites(self) -> Tuple[bool, Optional[str]]:
        # WSL2 with NVIDIA CUDA drivers is supported
        return (True, None)

    def is_wsl(self) -> bool:
        """Detect if running under Windows Subsystem for Linux."""
        # Method 1: Check /proc/sys/kernel/osrelease
        try:
            with open('/proc/sys/kernel/osrelease', 'r') as f:
                kernel_release = f.read().lower()
                if 'microsoft' in kernel_release or 'wsl' in kernel_release:
                    return True
        except (FileNotFoundError, PermissionError):
            pass

        # Method 2: Check for WSLInterop
        if os.path.exists('/proc/sys/fs/binfmt_misc/WSLInterop'):
            return True

        # Method 3: Check environment variable
        if 'WSL_DISTRO_NAME' in os.environ:
            return True

        return False

    def make_executable(self, path: Path) -> None:
        current = os.stat(path).st_mode
        os.chmod(path, current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    def rmtree_robust(self, path: Path) -> bool:
        shutil.rmtree(path)
        return True
