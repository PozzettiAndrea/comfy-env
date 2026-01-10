"""
IsolatedEnvManager - Creates and manages isolated Python environments.

Uses uv for fast environment creation and package installation.
"""

import subprocess
import shutil
from pathlib import Path
from typing import Optional, Callable

from .config import IsolatedEnv
from .platform import get_platform, PlatformProvider
from .detection import detect_cuda_version
from .security import (
    normalize_env_name,
    validate_dependency,
    validate_dependencies,
    validate_path_within_root,
    validate_wheel_url,
)


class IsolatedEnvManager:
    """
    Manages isolated Python environments for ComfyUI nodes.

    This class handles:
    - Creating Python virtual environments using uv
    - Installing PyTorch with correct CUDA version
    - Installing packages from requirements or wheel sources
    - Caching environments (same config = reuse existing)
    - Platform-specific handling (Windows DLLs, etc.)

    Example:
        manager = IsolatedEnvManager(base_dir=Path("./"))

        env = IsolatedEnv(
            name="my-node",
            python="3.10",
            cuda="12.8",
            requirements=["torch==2.8.0", "nvdiffrast"],
        )

        # Create environment and install dependencies
        env_path = manager.setup(env)

        # Get Python executable path
        python_exe = manager.get_python(env)
    """

    def __init__(
        self,
        base_dir: Path,
        log_callback: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize environment manager.

        Args:
            base_dir: Directory where environments will be created
            log_callback: Optional callback for logging (default: print)
        """
        self.base_dir = Path(base_dir)
        self.platform: PlatformProvider = get_platform()
        self.log = log_callback or print

        # Check platform compatibility
        is_compatible, error = self.platform.check_prerequisites()
        if not is_compatible:
            raise RuntimeError(f"Platform incompatible: {error}")

    def _find_uv(self) -> Optional[Path]:
        """Find uv executable."""
        uv_path = shutil.which("uv")
        if uv_path:
            return Path(uv_path)
        return None

    def _run_uv(self, args: list, env_dir: Optional[Path] = None, **kwargs) -> subprocess.CompletedProcess:
        """Run uv command."""
        uv = self._find_uv()
        if not uv:
            raise RuntimeError(
                "uv not found. Please install it:\n"
                "  curl -LsSf https://astral.sh/uv/install.sh | sh\n"
                "  # or on Windows: powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\""
            )

        cmd = [str(uv)] + args
        return subprocess.run(cmd, capture_output=True, text=True, **kwargs)

    def get_env_dir(self, env: IsolatedEnv) -> Path:
        """Get the environment directory path for a config."""
        # Validate environment name to prevent directory traversal
        safe_name = normalize_env_name(env.name)
        env_dir = env.get_default_env_dir(self.base_dir)

        # Ensure the resulting path is within base_dir
        validate_path_within_root(env_dir, self.base_dir)

        return env_dir

    def get_python(self, env: IsolatedEnv) -> Path:
        """Get the Python executable path for an environment."""
        env_dir = self.get_env_dir(env)
        return self.platform.get_env_paths(env_dir, env.python).python

    def get_pip(self, env: IsolatedEnv) -> Path:
        """Get the pip executable path for an environment."""
        env_dir = self.get_env_dir(env)
        return self.platform.get_env_paths(env_dir, env.python).pip

    def exists(self, env: IsolatedEnv) -> bool:
        """Check if environment already exists."""
        python_exe = self.get_python(env)
        return python_exe.exists()

    def is_ready(self, env: IsolatedEnv, verify_packages: Optional[list] = None) -> bool:
        """
        Check if environment is ready to use.

        Args:
            env: Environment configuration
            verify_packages: Optional list of packages to verify (e.g., ["torch", "numpy"])

        Returns:
            True if environment exists and packages are importable
        """
        python_exe = self.get_python(env)
        if not python_exe.exists():
            return False

        if verify_packages:
            imports = ", ".join(verify_packages)
            try:
                result = subprocess.run(
                    [str(python_exe), "-c", f"import {imports}"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                return result.returncode == 0
            except (subprocess.SubprocessError, OSError):
                return False

        return True

    def create_venv(self, env: IsolatedEnv) -> Path:
        """
        Create a virtual environment using uv.

        Args:
            env: Environment configuration

        Returns:
            Path to the environment directory
        """
        env_dir = self.get_env_dir(env)

        if env_dir.exists():
            self.log(f"Environment already exists: {env_dir}")
            return env_dir

        self.log(f"Creating environment: {env_dir}")
        self.log(f"  Python: {env.python}")

        result = self._run_uv([
            "venv",
            str(env_dir),
            "--python", env.python,
        ])

        if result.returncode != 0:
            raise RuntimeError(f"Failed to create venv: {result.stderr}")

        self.log(f"Environment created successfully")
        return env_dir

    def install_pytorch(self, env: IsolatedEnv) -> None:
        """
        Install PyTorch with appropriate CUDA version.

        Args:
            env: Environment configuration
        """
        cuda_version = env.cuda
        if cuda_version is None:
            # Auto-detect CUDA version
            cuda_version = detect_cuda_version()
            if cuda_version:
                self.log(f"Auto-detected CUDA version: {cuda_version}")
            else:
                self.log("No GPU detected, installing CPU-only PyTorch")

        python_exe = self.get_python(env)

        # Determine PyTorch index URL
        if cuda_version:
            cuda_short = cuda_version.replace(".", "")
            index_url = f"https://download.pytorch.org/whl/cu{cuda_short}"
            self.log(f"Installing PyTorch with CUDA {cuda_version}")
        else:
            index_url = "https://download.pytorch.org/whl/cpu"
            self.log("Installing PyTorch (CPU)")

        # Build uv pip command
        uv = self._find_uv()
        pip_args = [
            str(uv), "pip", "install",
            "--python", str(python_exe),
            "--index-url", index_url,
        ]

        # Install torch + torchvision together from same index to ensure ABI compatibility
        if env.pytorch_version:
            pip_args.append(f"torch=={env.pytorch_version}")
        else:
            pip_args.append("torch")

        # Always install torchvision from the same index - uv will resolve matching version
        pip_args.append("torchvision")

        result = subprocess.run(pip_args, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to install PyTorch: {result.stderr}")

        self.log("PyTorch + torchvision installed successfully")

    def install_requirements(
        self,
        env: IsolatedEnv,
        extra_args: Optional[list] = None,
    ) -> None:
        """
        Install requirements for an environment.

        Args:
            env: Environment configuration
            extra_args: Extra pip arguments
        """
        python_exe = self.get_python(env)

        if not env.requirements and not env.requirements_file:
            self.log("No requirements to install")
            return

        # Validate requirements for security
        if env.requirements:
            validate_dependencies(env.requirements)

        # Validate wheel sources
        for wheel_source in env.wheel_sources:
            validate_wheel_url(wheel_source)

        # Validate index URLs
        for index_url in env.index_urls:
            validate_wheel_url(index_url)

        uv = self._find_uv()
        pip_args = [str(uv), "pip", "install", "--python", str(python_exe)]

        # Add wheel sources as --find-links
        for wheel_source in env.wheel_sources:
            pip_args.extend(["--find-links", wheel_source])

        # Add extra index URLs
        for index_url in env.index_urls:
            pip_args.extend(["--extra-index-url", index_url])

        # Add extra args
        if extra_args:
            pip_args.extend(extra_args)

        # Install from requirements file
        if env.requirements_file and env.requirements_file.exists():
            self.log(f"Installing from {env.requirements_file}")
            result = subprocess.run(
                pip_args + ["-r", str(env.requirements_file)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to install requirements: {result.stderr}")

        # Install individual requirements
        if env.requirements:
            self.log(f"Installing {len(env.requirements)} packages")
            result = subprocess.run(
                pip_args + env.requirements,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to install packages: {result.stderr}")

        self.log("Requirements installed successfully")

    def _install_comfyui_isolation(self, env: IsolatedEnv) -> None:
        """Install comfyui-isolation package (needed for BaseWorker)."""
        python_exe = self.get_python(env)
        uv = self._find_uv()

        self.log("Installing comfyui-isolation (for worker support)...")
        result = subprocess.run(
            [str(uv), "pip", "install", "--python", str(python_exe),
             "comfyui-isolation @ git+https://github.com/PozzettiAndrea/comfyui-isolation"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            # Non-fatal - might already be installed or network issues
            self.log(f"Warning: Failed to install comfyui-isolation: {result.stderr}")
        else:
            self.log("comfyui-isolation installed")

    def setup(
        self,
        env: IsolatedEnv,
        install_pytorch: bool = True,
        verify_packages: Optional[list] = None,
    ) -> Path:
        """
        Create environment and install all dependencies.

        This is the main entry point for setting up an isolated environment.

        Args:
            env: Environment configuration
            install_pytorch: Whether to install PyTorch
            verify_packages: Packages to verify after installation

        Returns:
            Path to the environment directory
        """
        self.log("=" * 50)
        self.log(f"Setting up isolated environment: {env.name}")
        self.log("=" * 50)

        # Check if already ready
        if self.is_ready(env, verify_packages):
            self.log("Environment already ready, skipping setup")
            return self.get_env_dir(env)

        # Create virtual environment
        env_dir = self.create_venv(env)

        # Install PyTorch
        if install_pytorch and (env.cuda is not None or detect_cuda_version()):
            self.install_pytorch(env)

        # Install comfyui-isolation (needed for BaseWorker in workers)
        self._install_comfyui_isolation(env)

        # Install other requirements
        self.install_requirements(env)

        # Windows: Bundle VC++ DLLs
        if self.platform.name == 'windows':
            self.log("Bundling VC++ DLLs...")
            success, error = self.platform.bundle_vc_dlls_to_env(env_dir)
            if not success:
                self.log(f"Warning: {error}")

        # Verify installation
        if verify_packages:
            if self.is_ready(env, verify_packages):
                self.log("Verification passed!")
            else:
                raise RuntimeError(f"Verification failed: could not import {verify_packages}")

        self.log("=" * 50)
        self.log("Setup complete!")
        self.log(f"Python: {self.get_python(env)}")
        self.log("=" * 50)

        return env_dir

    def delete(self, env: IsolatedEnv) -> bool:
        """
        Delete an environment.

        Args:
            env: Environment configuration

        Returns:
            True if deleted, False if didn't exist
        """
        env_dir = self.get_env_dir(env)

        if not env_dir.exists():
            return False

        self.log(f"Deleting environment: {env_dir}")
        self.platform.rmtree_robust(env_dir)
        self.log("Environment deleted")
        return True

    def repair(
        self,
        env: IsolatedEnv,
        install_pytorch: bool = True,
        verify_packages: Optional[list] = None,
    ) -> Path:
        """
        Delete and recreate an environment.

        Args:
            env: Environment configuration
            install_pytorch: Whether to install PyTorch
            verify_packages: Packages to verify after installation

        Returns:
            Path to the new environment directory
        """
        self.log("Repairing environment (delete + recreate)")
        self.delete(env)
        return self.setup(env, install_pytorch, verify_packages)
