"""
IsolatedEnvManager - Creates and manages isolated Python environments.

Uses uv for fast environment creation and package installation.
"""

import subprocess
import shutil
import sys
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
from ..registry import PACKAGE_REGISTRY, is_registered, get_cuda_short2
from ..resolver import RuntimeEnv, parse_wheel_requirement


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
        """Find uv executable in PATH or current Python environment."""
        # First check system PATH
        uv_path = shutil.which("uv")
        if uv_path:
            return Path(uv_path)

        # Check current Python environment's Scripts/bin folder
        import sys
        if sys.platform == "win32":
            candidates = [
                Path(sys.prefix) / "Scripts" / "uv.exe",
                Path(sys.base_prefix) / "Scripts" / "uv.exe",
            ]
        else:
            candidates = [
                Path(sys.prefix) / "bin" / "uv",
                Path(sys.base_prefix) / "bin" / "uv",
            ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

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

        # Merge platform-specific requirements
        if sys.platform == 'win32':
            platform_reqs = env.windows_requirements
            platform_name = 'Windows'
        elif sys.platform == 'darwin':
            platform_reqs = env.darwin_requirements
            platform_name = 'macOS'
        else:
            platform_reqs = env.linux_requirements
            platform_name = 'Linux'

        all_requirements = list(env.requirements) + list(platform_reqs)

        if platform_reqs:
            self.log(f"Including {len(platform_reqs)} {platform_name}-specific packages")

        if not all_requirements and not env.requirements_file:
            self.log("No requirements to install")
            return

        # Validate requirements for security
        if all_requirements:
            validate_dependencies(all_requirements)

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

        # Install no-deps requirements first (e.g., CUDA extensions with conflicting metadata)
        # For packages in registry, resolve proper wheel URLs
        if env.no_deps_requirements:
            self.log(f"Installing {len(env.no_deps_requirements)} CUDA packages")
            self._install_cuda_packages(env, pip_args)

        # Install individual requirements (including platform-specific)
        if all_requirements:
            self.log(f"Installing {len(all_requirements)} packages")
            result = subprocess.run(
                pip_args + all_requirements,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to install packages: {result.stderr}")

        self.log("Requirements installed successfully")

    def _install_cuda_packages(self, env: IsolatedEnv, pip_args: list) -> None:
        """Install CUDA packages using registry-based wheel resolution."""
        import platform as plat

        # Build RuntimeEnv from IsolatedEnv config
        os_name = plat.system().lower()
        if os_name == 'darwin':
            platform_tag = f"macosx_{plat.mac_ver()[0].replace('.', '_')}_{plat.machine()}"
        elif os_name == 'windows':
            platform_tag = f"win_{plat.machine().lower()}"
        else:
            platform_tag = f"linux_{plat.machine()}"

        cuda_short = env.cuda.replace(".", "") if env.cuda else None
        torch_short = env.pytorch_version.replace(".", "") if env.pytorch_version else None
        torch_mm = "".join(env.pytorch_version.split(".")[:2]) if env.pytorch_version else None

        runtime_env = RuntimeEnv(
            os_name=os_name,
            platform_tag=platform_tag,
            python_version=env.python,
            python_short=env.python.replace(".", ""),
            cuda_version=env.cuda,
            cuda_short=cuda_short,
            torch_version=env.pytorch_version,
            torch_short=torch_short,
            torch_mm=torch_mm,
        )
        vars_dict = runtime_env.as_dict()
        if env.cuda:
            vars_dict["cuda_short2"] = get_cuda_short2(env.cuda)
        # Add py_tag for wheel filename templates (e.g., cp310)
        vars_dict["py_tag"] = f"cp{env.python.replace('.', '')}"

        for req in env.no_deps_requirements:
            package, version = parse_wheel_requirement(req)
            pkg_lower = package.lower()

            if pkg_lower in PACKAGE_REGISTRY:
                config = PACKAGE_REGISTRY[pkg_lower]
                method = config["method"]
                self.log(f"  Installing {package} ({method})...")

                if method == "index":
                    # PEP 503 index - use --extra-index-url
                    index_url = self._substitute_template(config["index_url"], vars_dict)
                    pkg_spec = f"{package}=={version}" if version else package
                    result = subprocess.run(
                        pip_args + ["--extra-index-url", index_url, "--no-deps", pkg_spec],
                        capture_output=True, text=True,
                    )

                elif method in ("github_index", "find_links"):
                    # GitHub Pages or generic find-links
                    index_url = self._substitute_template(config["index_url"], vars_dict)
                    pkg_spec = f"{package}=={version}" if version else package
                    result = subprocess.run(
                        pip_args + ["--find-links", index_url, "--no-deps", pkg_spec],
                        capture_output=True, text=True,
                    )

                elif method == "pypi_variant":
                    # Transform package name based on CUDA version
                    actual_package = self._substitute_template(config["package_template"], vars_dict)
                    pkg_spec = f"{actual_package}=={version}" if version else actual_package
                    self.log(f"    -> {actual_package}")
                    result = subprocess.run(
                        pip_args + ["--no-deps", pkg_spec],
                        capture_output=True, text=True,
                    )

                elif method == "github_release":
                    # Direct wheel URL from GitHub releases
                    release_vars = vars_dict.copy()
                    release_vars["version"] = version or ""
                    self._install_from_github_release(
                        package, version, release_vars, config, pip_args
                    )
                    continue  # Already handled

                else:
                    # Unknown method - try regular install
                    self.log(f"    Unknown method '{method}', trying regular install")
                    pkg_spec = f"{package}=={version}" if version else package
                    result = subprocess.run(
                        pip_args + ["--no-deps", pkg_spec],
                        capture_output=True, text=True,
                    )

                if result.returncode != 0:
                    raise RuntimeError(f"Failed to install {package}: {result.stderr}")
            else:
                # Not in registry - try regular pip install (e.g., spconv-cu126)
                self.log(f"  Installing {package} (PyPI)...")
                pkg_spec = f"{package}=={version}" if version else package
                result = subprocess.run(
                    pip_args + ["--no-deps", pkg_spec],
                    capture_output=True, text=True,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"Failed to install {package}: {result.stderr}")

    def _install_from_github_release(
        self,
        package: str,
        version: str,
        vars_dict: dict,
        config: dict,
        pip_args: list,
    ) -> None:
        """Install package from GitHub release wheels with fallback sources."""
        import platform as plat
        import sys
        # Use consistent platform tags (win_amd64, linux_x86_64, etc.)
        if sys.platform == 'win32':
            machine = plat.machine().lower()
            current_platform = 'win_amd64' if machine in ('amd64', 'x86_64') else f'win_{machine}'
        elif sys.platform == 'darwin':
            current_platform = f"macosx_{plat.machine()}"
        else:
            current_platform = f"linux_{plat.machine()}"

        sources = config.get("sources", [])
        errors = []

        for source in sources:
            # Check platform compatibility
            platforms = source.get("platforms", [])
            if platforms and not any(p in current_platform for p in platforms):
                continue

            url_template = source["url_template"]
            url = self._substitute_template(url_template, vars_dict)

            self.log(f"    Trying {source.get('name', 'unknown')}: {url[:80]}...")
            result = subprocess.run(
                pip_args + ["--no-deps", url],
                capture_output=True, text=True,
            )

            if result.returncode == 0:
                return  # Success!

            errors.append(f"{source.get('name', 'unknown')}: {result.stderr[:100]}")

        # All sources failed
        raise RuntimeError(
            f"Failed to install {package}=={version} from any source:\n"
            + "\n".join(errors)
        )

    def _substitute_template(self, template: str, vars_dict: dict) -> str:
        """Substitute template variables with environment values."""
        result = template
        for key, value in vars_dict.items():
            if value is not None:
                result = result.replace(f"{{{key}}}", str(value))
        return result

    def _install_comfy_env(self, env: IsolatedEnv) -> None:
        """Install comfy-env package (needed for BaseWorker)."""
        python_exe = self.get_python(env)
        uv = self._find_uv()

        self.log("Installing comfy-env (for worker support)...")
        result = subprocess.run(
            [str(uv), "pip", "install", "--python", str(python_exe),
             "--upgrade", "--no-cache",
             "comfy-env @ git+https://github.com/PozzettiAndrea/comfy-env"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            # Non-fatal - might already be installed or network issues
            self.log(f"Warning: Failed to install comfy-env: {result.stderr}")
        else:
            self.log("comfy-env installed")

    def _setup_windows_deps(self, env: IsolatedEnv, env_dir: Path) -> None:
        """
        Set up Windows-specific dependencies (VC++ runtime, DLL paths).

        This:
        1. Installs msvc-runtime package (provides VC++ DLLs)
        2. Sets up the Lib/x64/vc17/bin directory structure for opencv
        """
        python_exe = self.get_python(env)
        uv = self._find_uv()

        # Install msvc-runtime package to get VC++ DLLs
        self.log("Installing VC++ runtime (msvc-runtime)...")
        result = subprocess.run(
            [str(uv), "pip", "install", "--python", str(python_exe), "msvc-runtime"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            self.log(f"Warning: Failed to install msvc-runtime: {result.stderr}")
        else:
            self.log("msvc-runtime installed")

        # Set up opencv DLL paths (copies DLLs to Lib/x64/vc17/bin)
        self.log("Setting up opencv DLL paths...")
        success, msg = self.platform.setup_opencv_dll_paths(env_dir)
        if success:
            if msg:
                self.log(f"  {msg}")
        else:
            self.log(f"Warning: {msg}")

    def ensure_system_deps(self) -> bool:
        """
        Ensure system-level dependencies are installed (Windows only).

        On Windows, this checks for Media Foundation and installs if missing.
        Returns True if all deps are available, False otherwise.
        """
        if self.platform.name != 'windows':
            return True

        # Check and install Media Foundation if needed
        success, error = self.platform.ensure_media_foundation(self.log)
        if not success:
            self.log(f"WARNING: {error}")
            self.log("Some packages (like opencv) may not work correctly.")
            return False

        return True

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

        # Ensure system-level deps (Media Foundation on Windows)
        self.ensure_system_deps()

        # Check if already ready
        if self.is_ready(env, verify_packages):
            self.log("Environment already ready, skipping setup")
            return self.get_env_dir(env)

        # Create virtual environment
        env_dir = self.create_venv(env)

        # Install PyTorch
        if install_pytorch and (env.cuda is not None or detect_cuda_version()):
            self.install_pytorch(env)

        # Install comfy-env (needed for BaseWorker in workers)
        self._install_comfy_env(env)

        # Install other requirements
        self.install_requirements(env)

        # Windows: Install VC++ runtime and set up DLL paths
        if self.platform.name == 'windows':
            self._setup_windows_deps(env, env_dir)

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
