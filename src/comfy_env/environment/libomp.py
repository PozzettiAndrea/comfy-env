"""macOS: Dedupe libomp.dylib to prevent OpenMP runtime conflicts."""

import glob
import os
import sys


def dedupe_libomp() -> None:
    """Symlink all libomp copies to torch's to prevent dual-runtime crashes."""
    if sys.platform != "darwin":
        return

    try:
        import torch
        torch_libomp = os.path.join(os.path.dirname(torch.__file__), 'lib', 'libomp.dylib')
        if not os.path.exists(torch_libomp):
            return
    except ImportError:
        return

    site_packages = os.path.dirname(os.path.dirname(torch.__file__))
    patterns = [
        os.path.join(site_packages, '*', 'Frameworks', 'libomp.dylib'),
        os.path.join(site_packages, '*', '.dylibs', 'libomp.dylib'),
        os.path.join(site_packages, '*', 'lib', 'libomp.dylib'),
    ]

    for pattern in patterns:
        for libomp in glob.glob(pattern):
            if 'torch' in libomp:
                continue
            if os.path.islink(libomp) and os.path.realpath(libomp) == os.path.realpath(torch_libomp):
                continue
            try:
                if os.path.islink(libomp):
                    os.unlink(libomp)
                else:
                    os.rename(libomp, libomp + '.bak')
                os.symlink(torch_libomp, libomp)
            except OSError:
                pass
