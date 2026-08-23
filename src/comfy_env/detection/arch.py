"""CPU architecture detection.

A leaf module on purpose: it imports nothing from comfy_env, so both
`detection` (which sits at the bottom of the layer graph) and `packages`
(which sits above it) can use it without either importing the other.
"""

import platform


def cpu_arch() -> str:
    """Normalized CPU architecture: 'aarch64' or 'x86_64'.

    ARM reports `aarch64` on Linux and `arm64` on macOS; both mean the same
    machine language, and a wheel built for one will not load on the other kind
    of CPU no matter how well it matches the GPU.
    """
    return "aarch64" if platform.machine().lower() in ("aarch64", "arm64") else "x86_64"
