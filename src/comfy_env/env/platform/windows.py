"""
Windows platform provider implementation.
"""

import os
import stat
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from .base import PlatformProvider, PlatformPaths


class WindowsPlatformProvider(PlatformProvider):
    """Platform provider for Windows systems."""

    @property
    def name(self) -> str:
        return 'windows'

    @property
    def executable_suffix(self) -> str:
        return '.exe'

    @property
    def shared_lib_extension(self) -> str:
        return '.dll'

    def get_env_paths(self, env_dir: Path, python_version: str = "3.10") -> PlatformPaths:
        return PlatformPaths(
            python=env_dir / "Scripts" / "python.exe",
            pip=env_dir / "Scripts" / "pip.exe",
            site_packages=env_dir / "Lib" / "site-packages",
            bin_dir=env_dir / "Scripts"
        )

    def check_prerequisites(self) -> Tuple[bool, Optional[str]]:
        # Check for MSYS2/Cygwin/Git Bash
        shell_env = self._detect_shell_environment()
        if shell_env in ('msys2', 'cygwin', 'git-bash'):
            return (False,
                    f"Running in {shell_env.upper()} environment.\n"
                    f"This package requires native Windows Python.\n"
                    f"Please use PowerShell, Command Prompt, or native Windows terminal.")

        # Check Visual C++ Redistributable
        vc_ok, vc_error = self._check_vc_redistributable()
        if not vc_ok:
            return (False, vc_error)

        return (True, None)

    def _detect_shell_environment(self) -> str:
        """Detect if running in MSYS2, Cygwin, Git Bash, or native Windows."""
        msystem = os.environ.get('MSYSTEM', '')
        if msystem:
            if 'MINGW' in msystem:
                return 'git-bash'
            return 'msys2'

        term = os.environ.get('TERM', '')
        if term and 'cygwin' in term:
            return 'cygwin'

        return 'native-windows'

    def _find_vc_dlls(self) -> Dict[str, Optional[Path]]:
        """Find VC++ runtime DLLs in common locations."""
        required_dlls = ['vcruntime140.dll', 'msvcp140.dll']
        found = {}

        # Search locations in order of preference
        search_paths = []

        # 1. Current Python environment (conda/venv)
        if hasattr(sys, 'base_prefix'):
            search_paths.append(Path(sys.base_prefix) / 'Library' / 'bin')
            search_paths.append(Path(sys.base_prefix) / 'DLLs')
        if hasattr(sys, 'prefix'):
            search_paths.append(Path(sys.prefix) / 'Library' / 'bin')
            search_paths.append(Path(sys.prefix) / 'DLLs')

        # 2. System directories
        system_root = os.environ.get('SystemRoot', r'C:\Windows')
        search_paths.append(Path(system_root) / 'System32')

        # 3. Visual Studio redistributable directories
        program_files = os.environ.get('ProgramFiles', r'C:\Program Files')
        vc_redist = Path(program_files) / 'Microsoft Visual Studio' / '2022' / 'Community' / 'VC' / 'Redist' / 'MSVC'
        if vc_redist.exists():
            for version_dir in vc_redist.iterdir():
                search_paths.append(version_dir / 'x64' / 'Microsoft.VC143.CRT')

        for dll_name in required_dlls:
            found[dll_name] = None
            for search_path in search_paths:
                dll_path = search_path / dll_name
                if dll_path.exists():
                    found[dll_name] = dll_path
                    break

        return found

    def bundle_vc_dlls_to_env(self, env_dir: Path) -> Tuple[bool, Optional[str]]:
        """Bundle VC++ runtime DLLs into the isolated environment."""
        required_dlls = ['vcruntime140.dll', 'msvcp140.dll']
        found_dlls = self._find_vc_dlls()

        # Check which DLLs are missing
        missing = [dll for dll, path in found_dlls.items() if path is None]

        if missing:
            return (False,
                f"Could not find VC++ DLLs to bundle: {', '.join(missing)}\n\n"
                f"Please install Visual C++ Redistributable:\n"
                f"  Download: https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
                f"\nAfter installation, delete the environment and try again.")

        # Copy DLLs to the environment's Scripts directory
        scripts_dir = env_dir / 'Scripts'

        copied = []
        for dll_name, source_path in found_dlls.items():
            if source_path:
                try:
                    if scripts_dir.exists():
                        scripts_target = scripts_dir / dll_name
                        if not scripts_target.exists():
                            shutil.copy2(source_path, scripts_target)
                            copied.append(f"{dll_name} -> Scripts/")
                except (OSError, IOError) as e:
                    return (False, f"Failed to copy {dll_name}: {e}")

        return (True, None)

    def _check_vc_redistributable(self) -> Tuple[bool, Optional[str]]:
        """Check if Visual C++ Redistributable DLLs are available."""
        required_dlls = ['vcruntime140.dll', 'msvcp140.dll']
        found_dlls = self._find_vc_dlls()

        missing = [dll for dll, path in found_dlls.items() if path is None]

        if missing:
            error_msg = (
                f"Visual C++ Redistributable DLLs not found!\n"
                f"\nMissing: {', '.join(missing)}\n"
                f"\nPlease install Visual C++ Redistributable for Visual Studio 2015-2022:\n"
                f"\n  Download (64-bit): https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
                f"\nAfter installation, restart your terminal and try again."
            )
            return (False, error_msg)

        return (True, None)

    def make_executable(self, path: Path) -> None:
        # No-op on Windows - executables are determined by extension
        pass

    def rmtree_robust(self, path: Path, max_retries: int = 5, delay: float = 0.5) -> bool:
        """
        Windows-specific rmtree with retry logic for file locking issues.

        Handles Windows file locking, read-only files, and antivirus interference.
        """
        def handle_remove_readonly(func, fpath, exc):
            """Error handler for removing read-only files."""
            if isinstance(exc[1], PermissionError):
                try:
                    os.chmod(fpath, stat.S_IWRITE)
                    func(fpath)
                except Exception:
                    raise exc[1]
            else:
                raise exc[1]

        for attempt in range(max_retries):
            try:
                shutil.rmtree(path, onerror=handle_remove_readonly)
                return True
            except PermissionError:
                if attempt < max_retries - 1:
                    wait_time = delay * (2 ** attempt)
                    time.sleep(wait_time)
                else:
                    raise
            except OSError:
                if attempt < max_retries - 1:
                    wait_time = delay * (2 ** attempt)
                    time.sleep(wait_time)
                else:
                    raise

        return False
