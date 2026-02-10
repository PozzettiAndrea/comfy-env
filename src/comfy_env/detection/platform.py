"""Platform detection - OS, architecture, platform tags."""

from __future__ import annotations

import platform as platform_module
import sys
from dataclasses import dataclass


@dataclass
class PlatformInfo:
    os_name: str       # linux, windows, darwin
    arch: str          # x86_64, aarch64, arm64
    platform_tag: str  # linux_x86_64, win_amd64, macosx_11_0_arm64


def detect_platform() -> PlatformInfo:
    return PlatformInfo(
        os_name=_get_os_name(),
        arch=platform_module.machine().lower(),
        platform_tag=get_platform_tag(),
    )


def _get_os_name() -> str:
    if sys.platform.startswith('linux'): return 'linux'
    if sys.platform == 'win32': return 'windows'
    if sys.platform == 'darwin': return 'darwin'
    return sys.platform


_PLATFORM_TAGS = {
    ('linux', 'x86_64'): 'linux_x86_64', ('linux', 'amd64'): 'linux_x86_64',
    ('linux', 'aarch64'): 'linux_aarch64',
    ('win32', 'amd64'): 'win_amd64', ('win32', 'x86_64'): 'win_amd64',
    ('darwin', 'arm64'): 'macosx_11_0_arm64',
    ('darwin', 'x86_64'): 'macosx_10_9_x86_64',
}


def get_platform_tag() -> str:
    key = (sys.platform if sys.platform != 'linux' else 'linux', platform_module.machine().lower())
    return _PLATFORM_TAGS.get(key, f'{sys.platform}_{platform_module.machine().lower()}')


_PIXI_PLATFORMS = {
    ('linux', 'x86_64'): 'linux-64', ('linux', 'amd64'): 'linux-64',
    ('linux', 'aarch64'): 'linux-aarch64',
    ('windows', 'amd64'): 'win-64', ('windows', 'x86_64'): 'win-64',
    ('darwin', 'arm64'): 'osx-arm64',
    ('darwin', 'x86_64'): 'osx-64',
}


def get_pixi_platform() -> str:
    key = (_get_os_name(), platform_module.machine().lower())
    return _PIXI_PLATFORMS.get(key, f'{key[0]}-{key[1]}')


def get_library_extension() -> str:
    return {'.dll': 'windows', '.dylib': 'darwin'}.get(_get_os_name(), '.so')


def get_executable_suffix() -> str:
    return '.exe' if _get_os_name() == 'windows' else ''


def is_linux() -> bool: return _get_os_name() == 'linux'
def is_windows() -> bool: return _get_os_name() == 'windows'
def is_macos() -> bool: return _get_os_name() == 'darwin'
