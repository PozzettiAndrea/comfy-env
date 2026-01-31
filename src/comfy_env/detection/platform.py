"""
Platform detection - OS, architecture, and platform tags.

Pure functions for detecting the current platform configuration.
"""

from __future__ import annotations

import platform as platform_module
import sys
from dataclasses import dataclass


@dataclass
class PlatformInfo:
    """Platform information."""
    os_name: str  # linux, windows, darwin
    arch: str  # x86_64, aarch64, arm64
    platform_tag: str  # linux_x86_64, win_amd64, macosx_11_0_arm64


def detect_platform() -> PlatformInfo:
    """Detect current platform."""
    os_name = _get_os_name()
    arch = platform_module.machine().lower()
    platform_tag = get_platform_tag()

    return PlatformInfo(
        os_name=os_name,
        arch=arch,
        platform_tag=platform_tag,
    )


def _get_os_name() -> str:
    """Get normalized OS name."""
    os_name = sys.platform
    if os_name.startswith('linux'):
        return 'linux'
    elif os_name == 'win32':
        return 'windows'
    elif os_name == 'darwin':
        return 'darwin'
    return os_name


def get_platform_tag() -> str:
    """Get wheel platform tag for current system."""
    machine = platform_module.machine().lower()

    if sys.platform.startswith('linux'):
        if machine in ('x86_64', 'amd64'):
            return 'linux_x86_64'
        elif machine == 'aarch64':
            return 'linux_aarch64'
        return f'linux_{machine}'

    elif sys.platform == 'win32':
        if machine in ('amd64', 'x86_64'):
            return 'win_amd64'
        return 'win32'

    elif sys.platform == 'darwin':
        if machine == 'arm64':
            return 'macosx_11_0_arm64'
        return 'macosx_10_9_x86_64'

    return f'{sys.platform}_{machine}'


def get_pixi_platform() -> str:
    """Get pixi platform identifier (e.g., linux-64, win-64, osx-arm64)."""
    os_name = _get_os_name()
    machine = platform_module.machine().lower()

    if os_name == 'linux':
        if machine in ('x86_64', 'amd64'):
            return 'linux-64'
        elif machine == 'aarch64':
            return 'linux-aarch64'
    elif os_name == 'windows':
        return 'win-64'
    elif os_name == 'darwin':
        if machine == 'arm64':
            return 'osx-arm64'
        return 'osx-64'

    return f'{os_name}-{machine}'


def get_library_extension() -> str:
    """Get shared library extension for current platform."""
    os_name = _get_os_name()
    if os_name == 'windows':
        return '.dll'
    elif os_name == 'darwin':
        return '.dylib'
    return '.so'


def get_executable_suffix() -> str:
    """Get executable suffix for current platform."""
    if _get_os_name() == 'windows':
        return '.exe'
    return ''


def is_linux() -> bool:
    """Check if running on Linux."""
    return _get_os_name() == 'linux'


def is_windows() -> bool:
    """Check if running on Windows."""
    return _get_os_name() == 'windows'


def is_macos() -> bool:
    """Check if running on macOS."""
    return _get_os_name() == 'darwin'
