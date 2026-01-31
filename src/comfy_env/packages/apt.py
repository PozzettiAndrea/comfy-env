"""APT package installation (Linux only)."""

import subprocess
import sys
from typing import Callable, List


def apt_install(packages: List[str], log: Callable[[str], None] = print) -> bool:
    """Install system packages via apt-get. No-op on non-Linux."""
    if not packages or sys.platform != "linux":
        return True

    log(f"Installing apt packages: {packages}")

    subprocess.run(["sudo", "apt-get", "update"], capture_output=True, text=True)

    result = subprocess.run(
        ["sudo", "apt-get", "install", "-y"] + packages,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        log(f"Warning: apt-get install failed: {result.stderr[:200]}")
        return False

    return True


def check_apt_packages(packages: List[str]) -> List[str]:
    """Return list of packages NOT installed."""
    if sys.platform != "linux":
        return []

    return [
        pkg for pkg in packages
        if subprocess.run(["dpkg", "-s", pkg], capture_output=True).returncode != 0
    ]
