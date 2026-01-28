"""
Windows platform provider implementation.
"""

import os
import stat
import shutil
import subprocess
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
        # Note: VC++ runtime is handled by msvc-runtime package, no system check needed
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

    # =========================================================================
    # Media Foundation Detection and Installation
    # =========================================================================

    def check_media_foundation(self) -> bool:
        """
        Check if Media Foundation DLLs exist on the system.

        These are required by packages like opencv-python for video/media support.
        Missing on Windows N/KN editions and some Windows Server installations.
        """
        system_root = os.environ.get('SystemRoot', r'C:\Windows')
        mf_dlls = ['MFPlat.dll', 'MF.dll', 'MFReadWrite.dll']

        for dll in mf_dlls:
            dll_path = Path(system_root) / 'System32' / dll
            if not dll_path.exists():
                return False
        return True

    def install_media_foundation(self, log_callback=None) -> Tuple[bool, Optional[str]]:
        """
        Install Media Foundation via DISM.

        Requires administrator privileges. Will trigger UAC prompt if needed.

        Returns:
            Tuple of (success, error_message)
        """
        log = log_callback or print

        if self.check_media_foundation():
            return (True, None)

        log("Media Foundation not found. Installing via DISM...")
        log("(This requires administrator privileges - a UAC prompt may appear)")

        # DISM command to install Media Feature Pack
        dism_cmd = [
            "DISM.exe", "/Online", "/Add-Capability",
            "/CapabilityName:Media.MediaFeaturePack~~~~0.0.1.0"
        ]

        try:
            # First try without elevation (in case already running as admin)
            result = subprocess.run(
                dism_cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout for installation
            )

            if result.returncode == 0:
                log("Media Foundation installed successfully!")
                return (True, None)

            # Check if we need elevation
            if "administrator" in result.stderr.lower() or result.returncode == 740:
                log("Requesting administrator privileges...")
                return self._install_media_foundation_elevated(log)

            # Other error
            return (False,
                f"DISM failed with code {result.returncode}:\n{result.stderr}\n\n"
                f"Please install Media Foundation manually:\n"
                f"  1. Open Settings > Apps > Optional Features\n"
                f"  2. Click 'Add a feature'\n"
                f"  3. Search for 'Media Feature Pack' and install it\n"
                f"  4. Restart your computer")

        except subprocess.TimeoutExpired:
            return (False, "DISM timed out. Please try installing manually via Settings.")
        except FileNotFoundError:
            return (False, "DISM.exe not found. Please install Media Feature Pack manually via Settings.")
        except Exception as e:
            return (False, f"Error running DISM: {e}")

    def _install_media_foundation_elevated(self, log_callback=None) -> Tuple[bool, Optional[str]]:
        """
        Install Media Foundation with UAC elevation.

        Uses PowerShell Start-Process -Verb RunAs to trigger UAC prompt.
        """
        log = log_callback or print

        # Create a PowerShell script that runs DISM elevated
        ps_script = '''
        $result = Start-Process -FilePath "DISM.exe" -ArgumentList "/Online", "/Add-Capability", "/CapabilityName:Media.MediaFeaturePack~~~~0.0.1.0" -Verb RunAs -Wait -PassThru
        exit $result.ExitCode
        '''

        try:
            result = subprocess.run(
                ["powershell", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                # Verify installation
                if self.check_media_foundation():
                    log("Media Foundation installed successfully!")
                    return (True, None)
                else:
                    return (False,
                        "Installation completed but Media Foundation DLLs not found.\n"
                        "A system restart may be required.")
            else:
                return (False,
                    f"Installation failed or was cancelled.\n"
                    f"Please install Media Feature Pack manually:\n"
                    f"  Settings > Apps > Optional Features > Add a feature > Media Feature Pack")

        except subprocess.TimeoutExpired:
            return (False, "Installation timed out. Please try installing manually via Settings.")
        except Exception as e:
            return (False, f"Error during elevated installation: {e}")

    def ensure_media_foundation(self, log_callback=None) -> Tuple[bool, Optional[str]]:
        """
        Ensure Media Foundation is installed, installing if necessary.

        This is the main entry point for MF dependency checking.
        """
        if self.check_media_foundation():
            return (True, None)

        return self.install_media_foundation(log_callback)

    # =========================================================================
    # OpenCV DLL Directory Setup
    # =========================================================================

    def setup_opencv_dll_paths(self, env_dir: Path) -> Tuple[bool, Optional[str]]:
        """
        Set up the DLL directory structure that opencv-python expects.

        OpenCV's config.py expects VC++ DLLs at:
          {site-packages}/cv2/../../x64/vc17/bin
        Which resolves to:
          {env_dir}/Lib/x64/vc17/bin

        This copies the VC++ DLLs to that location.
        """
        # Target directory that opencv expects
        target_dir = env_dir / "Lib" / "x64" / "vc17" / "bin"

        # Source: DLLs in Scripts or base env dir (from msvc-runtime package)
        scripts_dir = env_dir / "Scripts"

        vc_dlls = [
            'vcruntime140.dll', 'vcruntime140_1.dll', 'vcruntime140_threads.dll',
            'msvcp140.dll', 'msvcp140_1.dll', 'msvcp140_2.dll',
            'msvcp140_atomic_wait.dll', 'msvcp140_codecvt_ids.dll',
            'concrt140.dll', 'vcomp140.dll', 'vcamp140.dll', 'vccorlib140.dll'
        ]

        try:
            target_dir.mkdir(parents=True, exist_ok=True)

            copied = 0
            for dll_name in vc_dlls:
                # Try Scripts first, then env root
                for source_dir in [scripts_dir, env_dir]:
                    source = source_dir / dll_name
                    if source.exists():
                        target = target_dir / dll_name
                        if not target.exists():
                            shutil.copy2(source, target)
                            copied += 1
                        break

            if copied > 0:
                return (True, f"Copied {copied} VC++ DLLs to opencv path")
            else:
                return (True, "VC++ DLLs already in place or not found in venv")

        except Exception as e:
            return (False, f"Failed to set up opencv DLL paths: {e}")
