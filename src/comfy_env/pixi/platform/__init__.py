"""Platform-specific providers for comfyui-isolation."""

import sys

from .base import PlatformProvider, PlatformPaths

# Import platform-specific provider
if sys.platform == 'win32':
    from .windows import WindowsPlatformProvider as _Provider
elif sys.platform == 'darwin':
    from .darwin import DarwinPlatformProvider as _Provider
else:
    from .linux import LinuxPlatformProvider as _Provider


def get_platform() -> PlatformProvider:
    """Get the platform provider for the current system."""
    return _Provider()


__all__ = ["PlatformProvider", "PlatformPaths", "get_platform"]
