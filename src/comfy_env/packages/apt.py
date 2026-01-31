"""
APT package installation for Linux systems.

Handles installation of system packages via apt-get.
"""

import subprocess
import sys
from typing import Callable, List


def apt_install(
    packages: List[str],
    log: Callable[[str], None] = print,
) -> bool:
    """
    Install system packages via apt-get.

    Only works on Linux systems. Silently skipped on other platforms.

    Args:
        packages: List of package names to install.
        log: Logging callback.

    Returns:
        True if installation succeeded (or skipped on non-Linux).
    """
    if not packages:
        return True

    if sys.platform != "linux":
        return True

    log(f"Installing apt packages: {packages}")

    # Update package list
    result = subprocess.run(
        ["sudo", "apt-get", "update"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log(f"Warning: apt-get update failed: {result.stderr[:200]}")

    # Install packages
    result = subprocess.run(
        ["sudo", "apt-get", "install", "-y"] + packages,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        log(f"Warning: apt-get install failed: {result.stderr[:200]}")
        return False

    return True


def check_apt_packages(packages: List[str]) -> List[str]:
    """
    Check which apt packages are already installed.

    Args:
        packages: List of package names to check.

    Returns:
        List of packages that are NOT installed.
    """
    if sys.platform != "linux":
        return []

    missing = []
    for package in packages:
        result = subprocess.run(
            ["dpkg", "-s", package],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            missing.append(package)

    return missing
