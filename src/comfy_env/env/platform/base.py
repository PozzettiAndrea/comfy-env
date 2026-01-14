"""
Abstract base class for platform-specific operations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple


@dataclass
class PlatformPaths:
    """Platform-specific paths within an environment."""
    python: Path
    pip: Path
    site_packages: Path
    bin_dir: Path


class PlatformProvider(ABC):
    """
    Abstract base class for platform-specific operations.

    Each platform (Linux, Windows, macOS) implements this interface
    to provide consistent behavior across operating systems.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Platform name: 'linux', 'windows', 'darwin'."""
        pass

    @property
    @abstractmethod
    def executable_suffix(self) -> str:
        """Executable suffix: '' for Unix, '.exe' for Windows."""
        pass

    @property
    @abstractmethod
    def shared_lib_extension(self) -> str:
        """Shared library extension: '.so', '.dll', '.dylib'."""
        pass

    @abstractmethod
    def get_env_paths(self, env_dir: Path, python_version: str = "3.10") -> PlatformPaths:
        """
        Get platform-specific paths for an environment.

        Args:
            env_dir: Root directory of the environment
            python_version: Python version (e.g., "3.10")

        Returns:
            PlatformPaths with python, pip, site_packages, bin_dir
        """
        pass

    @abstractmethod
    def check_prerequisites(self) -> Tuple[bool, Optional[str]]:
        """
        Check platform-specific prerequisites.

        Returns:
            Tuple of (is_compatible, error_message)
            error_message is None if compatible
        """
        pass

    @abstractmethod
    def make_executable(self, path: Path) -> None:
        """
        Make a file executable.

        Args:
            path: Path to the file
        """
        pass

    @abstractmethod
    def rmtree_robust(self, path: Path) -> bool:
        """
        Remove directory tree with platform-specific error handling.

        Args:
            path: Directory to remove

        Returns:
            True if successful
        """
        pass

    def get_uv_exe_name(self) -> str:
        """Get uv executable name for this platform."""
        return f"uv{self.executable_suffix}"
