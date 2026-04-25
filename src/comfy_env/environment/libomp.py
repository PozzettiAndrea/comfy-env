"""macOS: Dedupe libomp.dylib copies in a site-packages to prevent OpenMP runtime conflicts."""

import glob
import os
import sys
from pathlib import Path
from typing import Optional


def dedupe_libomp(site_packages: Optional[Path] = None) -> None:
    """Symlink redundant libomp.dylib copies to torch's, in `site_packages`.

    macOS only. Many pip wheels ship their own bundled libomp.dylib (torch,
    sklearn, pymeshlab, etc.); having multiple copies loaded in one process
    can corrupt OMP runtime state and SIGSEGV inside native filters.

    With no argument: dedupes the parent process's torch site-packages
    (legacy ComfyUI prestartup behavior).
    With `site_packages` set: dedupes that directory's wheels and the
    enclosing env's `lib/libomp.dylib` (conda-forge libomp at env root).

    The canonical libomp is `<sp>/torch/lib/libomp.dylib` — usually present
    because torch is installed in every env via the `comfyui` feature.
    """
    if sys.platform != "darwin":
        return

    if site_packages is None:
        try:
            import torch
        except ImportError:
            return
        sp_dir = os.path.dirname(os.path.dirname(torch.__file__))
    else:
        sp_dir = str(site_packages)

    torch_libomp = os.path.join(sp_dir, "torch", "lib", "libomp.dylib")
    if not os.path.exists(torch_libomp):
        return  # No canonical libomp to point at; bail.

    patterns = [
        os.path.join(sp_dir, "*", "Frameworks", "libomp.dylib"),
        os.path.join(sp_dir, "*", ".dylibs", "libomp.dylib"),
        os.path.join(sp_dir, "*", "lib", "libomp.dylib"),
    ]

    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))

    # Also handle the env-root libomp at <env>/lib/libomp.dylib. Site-packages
    # lives at <env>/lib/python*/site-packages, so the env-root lib dir is two
    # parents up from sp_dir.
    env_lib_dir = Path(sp_dir).parent.parent  # <env>/lib
    env_libomp = env_lib_dir / "libomp.dylib"
    if env_libomp.exists():
        candidates.append(str(env_libomp))

    for libomp in candidates:
        if "torch" in libomp:
            continue
        try:
            if os.path.islink(libomp):
                if os.path.realpath(libomp) == os.path.realpath(torch_libomp):
                    continue
                os.unlink(libomp)
            else:
                os.rename(libomp, libomp + ".bak")
            os.symlink(torch_libomp, libomp)
        except OSError:
            pass
