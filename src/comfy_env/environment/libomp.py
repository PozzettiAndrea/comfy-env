"""
macOS libomp deduplication.

Prevents OpenMP runtime conflicts when multiple packages bundle their own libomp.
"""

import glob
import os
import sys


def dedupe_libomp() -> None:
    """
    macOS: Dedupe libomp.dylib to prevent OpenMP runtime conflicts.

    Torch and other packages (PyMeshLab, etc.) bundle their own libomp.
    Two OpenMP runtimes in same process = crash.

    Fix: symlink all libomp copies to torch's (canonical source).
    """
    if sys.platform != "darwin":
        return

    # Find torch's libomp (canonical source) via introspection
    try:
        import torch
        torch_libomp = os.path.join(os.path.dirname(torch.__file__), 'lib', 'libomp.dylib')
        if not os.path.exists(torch_libomp):
            return
    except ImportError:
        return  # No torch, skip

    # Find site-packages for scanning
    site_packages = os.path.dirname(os.path.dirname(torch.__file__))

    # Find other libomp files and symlink to torch's
    patterns = [
        os.path.join(site_packages, '*', 'Frameworks', 'libomp.dylib'),
        os.path.join(site_packages, '*', '.dylibs', 'libomp.dylib'),
        os.path.join(site_packages, '*', 'lib', 'libomp.dylib'),
    ]

    for pattern in patterns:
        for libomp in glob.glob(pattern):
            # Skip torch's own copy
            if 'torch' in libomp:
                continue
            # Skip if already a symlink pointing to torch
            if os.path.islink(libomp):
                if os.path.realpath(libomp) == os.path.realpath(torch_libomp):
                    continue
            # Replace with symlink to torch's
            try:
                if os.path.islink(libomp):
                    os.unlink(libomp)
                else:
                    os.rename(libomp, libomp + '.bak')
                os.symlink(torch_libomp, libomp)
            except OSError:
                pass  # Permission denied, etc.
