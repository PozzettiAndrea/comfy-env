"""Homebrew package installation (macOS only)."""

import subprocess
import sys
from typing import Callable, List


def brew_install(packages: List[str], log: Callable[[str], None] = print) -> bool:
    """Install system packages via Homebrew. No-op on non-macOS."""
    if not packages or sys.platform != "darwin":
        return True

    log(f"[brew] Requested packages: {packages}")

    # Check which packages are missing
    missing = check_brew_packages(packages)
    if not missing:
        log("[brew] All packages already installed")
        return True

    log(f"[brew] Missing packages: {missing}")

    # Install packages ONE BY ONE so one failure doesn't block others
    installed = []
    failed = []
    for pkg in missing:
        log(f"[brew] Installing: {pkg}")
        result = subprocess.run(
            ["brew", "install", pkg],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            log(f"[brew] FAILED: {pkg} - {result.stderr.strip()}")
            failed.append(pkg)
        else:
            log(f"[brew] OK: {pkg}")
            installed.append(pkg)

    # Summary
    if installed:
        log(f"[brew] Successfully installed: {installed}")
    if failed:
        log(f"[brew] Failed to install: {failed}")

    # Verify what's actually installed now
    still_missing = check_brew_packages(packages)
    if still_missing:
        log(f"[brew] WARNING: These packages are not available: {still_missing}")

    # Return True if we installed at least something (partial success is OK)
    return len(installed) > 0 or len(missing) == len(failed)


def check_brew_packages(packages: List[str]) -> List[str]:
    """Return list of packages NOT installed."""
    if sys.platform != "darwin":
        return []

    return [
        pkg for pkg in packages
        if subprocess.run(["brew", "list", pkg], capture_output=True).returncode != 0
    ]
