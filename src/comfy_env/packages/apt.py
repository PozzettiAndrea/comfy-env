"""APT package installation (Linux only)."""

import subprocess
import sys
from typing import Callable, List


def apt_install(packages: List[str], log: Callable[[str], None] = print) -> bool:
    """Install system packages via apt-get. No-op on non-Linux."""
    if not packages or sys.platform != "linux":
        return True

    log(f"[apt] Requested packages: {packages}")

    # Check which packages are missing
    missing = check_apt_packages(packages)
    if not missing:
        log("[apt] All packages already installed")
        return True

    log(f"[apt] Missing packages: {missing}")

    # Run apt-get update with full output
    log("[apt] Running: sudo apt-get update")
    update_result = subprocess.run(
        ["sudo", "apt-get", "update"],
        capture_output=True, text=True
    )
    if update_result.returncode != 0:
        log(f"[apt] WARNING: apt-get update failed (exit {update_result.returncode})")
        log(f"[apt] stderr: {update_result.stderr}")
    else:
        log("[apt] apt-get update succeeded")

    # Install missing packages
    log(f"[apt] Running: sudo apt-get install -y {' '.join(missing)}")
    result = subprocess.run(
        ["sudo", "apt-get", "install", "-y"] + missing,
        capture_output=True, text=True
    )

    if result.returncode != 0:
        log(f"[apt] ERROR: apt-get install failed (exit {result.returncode})")
        log(f"[apt] stdout: {result.stdout}")
        log(f"[apt] stderr: {result.stderr}")
        return False

    log("[apt] Installation succeeded")

    # Verify installation
    still_missing = check_apt_packages(missing)
    if still_missing:
        log(f"[apt] WARNING: These packages still not found: {still_missing}")
        return False

    log("[apt] All packages verified installed")
    return True


def check_apt_packages(packages: List[str]) -> List[str]:
    """Return list of packages NOT installed."""
    if sys.platform != "linux":
        return []

    return [
        pkg for pkg in packages
        if subprocess.run(["dpkg", "-s", pkg], capture_output=True).returncode != 0
    ]
