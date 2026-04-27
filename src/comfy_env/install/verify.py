"""Post-install verification helpers."""

from __future__ import annotations

from typing import Callable, List


def verify_installation(packages: List[str], log: Callable[[str], None] = print) -> bool:
    all_ok = True
    for package in packages:
        import_name = package.replace("-", "_").split("[")[0]
        try:
            __import__(import_name)
            log(f"  {package}: OK")
        except ImportError as e:
            log(f"  {package}: FAILED ({e})")
            all_ok = False
    return all_ok
