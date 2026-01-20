"""
Installation API for comfy-env.

This module provides the main `install()` function that handles both:
- In-place installation (CUDA wheels into current environment)
- Isolated installation (create separate venv with dependencies)

Example:
    from comfy_env import install

    # In-place install (auto-discovers config)
    install()

    # In-place with explicit config
    install(config="comfy-env.toml")

    # Isolated environment
    install(config="comfy-env.toml", mode="isolated")
"""

import inspect
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Union

from .env.config import IsolatedEnv, LocalConfig, NodeReq, SystemConfig
from .env.config_file import load_config, discover_config
from .env.manager import IsolatedEnvManager
from .errors import CUDANotFoundError, InstallError
from .pixi import pixi_install
from .registry import PACKAGE_REGISTRY, get_cuda_short2
from .resolver import RuntimeEnv, parse_wheel_requirement


def _install_system_packages(
    system_config: SystemConfig,
    log: Callable[[str], None],
    dry_run: bool = False,
) -> bool:
    """
    Install system-level packages (apt, brew, etc.).

    Args:
        system_config: SystemConfig with package lists per OS.
        log: Logging callback.
        dry_run: If True, show what would be installed without installing.

    Returns:
        True if installation succeeded or no packages needed.
    """
    platform = sys.platform

    if platform.startswith("linux"):
        packages = system_config.linux
        if not packages:
            return True

        log(f"Installing {len(packages)} system package(s) via apt...")

        if dry_run:
            log(f"  Would install: {', '.join(packages)}")
            return True

        if not shutil.which("apt-get"):
            log("  Warning: apt-get not found. Cannot install system packages.")
            log(f"  Please install manually: {', '.join(packages)}")
            return True

        sudo_available = shutil.which("sudo") is not None

        try:
            if sudo_available:
                log("  Running apt-get update...")
                subprocess.run(["sudo", "apt-get", "update"], capture_output=True, text=True)

                log(f"  Installing: {', '.join(packages)}")
                install_result = subprocess.run(
                    ["sudo", "apt-get", "install", "-y"] + packages,
                    capture_output=True,
                    text=True,
                )

                if install_result.returncode != 0:
                    log(f"  Warning: apt-get install failed: {install_result.stderr.strip()}")
                    log(f"  Please install manually: sudo apt-get install {' '.join(packages)}")
                else:
                    log("  System packages installed successfully.")
            else:
                log("  Warning: sudo not available.")
                log(f"  Please install manually: sudo apt-get install {' '.join(packages)}")

        except Exception as e:
            log(f"  Warning: Failed to install system packages: {e}")
            log(f"  Please install manually: sudo apt-get install {' '.join(packages)}")

        return True

    elif platform == "darwin":
        packages = system_config.darwin
        if packages:
            log(f"System packages for macOS: {', '.join(packages)}")
            log(f"  Please install manually: brew install {' '.join(packages)}")
        return True

    elif platform == "win32":
        packages = system_config.windows
        if packages:
            log(f"System packages for Windows: {', '.join(packages)}")
            log("  Please install manually.")
        return True

    return True


def _install_node_dependencies(
    node_reqs: List[NodeReq],
    node_dir: Path,
    log: Callable[[str], None],
    dry_run: bool = False,
) -> bool:
    """Install node dependencies (other ComfyUI custom nodes)."""
    from .nodes import install_node_deps

    custom_nodes_dir = node_dir.parent
    log(f"\nInstalling {len(node_reqs)} node dependencies...")

    if dry_run:
        for req in node_reqs:
            node_path = custom_nodes_dir / req.name
            status = "exists" if node_path.exists() else "would clone"
            log(f"  {req.name}: {status}")
        return True

    visited: Set[str] = {node_dir.name}
    install_node_deps(node_reqs, custom_nodes_dir, log, visited)
    return True


def install(
    log_callback: Optional[Callable[[str], None]] = None,
    dry_run: bool = False,
) -> bool:
    """
    Install dependencies from comfy-env.toml, auto-discovered from caller's directory.

    Example:
        from comfy_env import install
        install()

    Args:
        log_callback: Optional callback for logging. Defaults to print.
        dry_run: If True, show what would be installed without installing.

    Returns:
        True if installation succeeded.
    """
    # Auto-discover caller's directory
    frame = inspect.stack()[1]
    caller_file = frame.filename
    node_dir = Path(caller_file).parent.resolve()

    log = log_callback or print

    full_config = _load_full_config(None, node_dir)
    if full_config is None:
        raise FileNotFoundError(
            f"No comfy-env.toml found in {node_dir}. "
            "Create comfy-env.toml to define dependencies."
        )

    if full_config.node_reqs:
        _install_node_dependencies(full_config.node_reqs, node_dir, log, dry_run)

    if full_config.has_system:
        _install_system_packages(full_config.system, log, dry_run)

    env_config = full_config.default_env
    if env_config is None and not full_config.has_local:
        log("No packages to install")
        return True

    if env_config:
        log(f"Found configuration: {env_config.name}")

    if env_config and env_config.uses_conda:
        log("Environment uses conda packages - using pixi backend")
        return pixi_install(env_config, node_dir, log, dry_run)

    # Get user wheel_sources overrides
    user_wheel_sources = full_config.wheel_sources if hasattr(full_config, 'wheel_sources') else {}

    if env_config:
        if env_config.python:
            return _install_isolated(env_config, node_dir, log, dry_run)
        else:
            return _install_inplace(env_config, node_dir, log, dry_run, user_wheel_sources)
    elif full_config.has_local:
        return _install_local(full_config.local, node_dir, log, dry_run, user_wheel_sources)
    else:
        return True


def _load_full_config(config: Optional[Union[str, Path]], node_dir: Path):
    """Load full EnvManagerConfig (includes tools)."""
    if config is not None:
        config_path = Path(config)
        if not config_path.is_absolute():
            config_path = node_dir / config_path
        return load_config(config_path, node_dir)
    return discover_config(node_dir)


def _install_isolated(
    env_config: IsolatedEnv,
    node_dir: Path,
    log: Callable[[str], None],
    dry_run: bool,
) -> bool:
    """Install in isolated mode using IsolatedEnvManager."""
    log(f"Installing in isolated mode: {env_config.name}")

    if dry_run:
        log("Dry run - would create isolated environment:")
        log(f"  Python: {env_config.python}")
        log(f"  CUDA: {env_config.cuda or 'auto-detect'}")
        if env_config.requirements:
            log(f"  Requirements: {len(env_config.requirements)} packages")
        return True

    manager = IsolatedEnvManager(base_dir=node_dir, log_callback=log)
    env_dir = manager.setup(env_config)
    log(f"Isolated environment ready: {env_dir}")
    return True


def _install_inplace(
    env_config: IsolatedEnv,
    node_dir: Path,
    log: Callable[[str], None],
    dry_run: bool,
    user_wheel_sources: Dict[str, str],
) -> bool:
    """Install in-place into current environment."""
    log("Installing in-place mode")

    if sys.platform == "win32":
        log("Installing MSVC runtime for Windows...")
        if not dry_run:
            _pip_install(["msvc-runtime"], no_deps=False, log=log)

    env = RuntimeEnv.detect()
    log(f"Detected environment: {env}")

    if not env.cuda_version:
        cuda_packages = env_config.no_deps_requirements or []
        if cuda_packages:
            raise CUDANotFoundError(package=", ".join(cuda_packages))

    cuda_packages = env_config.no_deps_requirements or []
    regular_packages = env_config.requirements or []

    if dry_run:
        log("\nDry run - would install:")
        for req in cuda_packages:
            package, version = parse_wheel_requirement(req)
            url = _resolve_wheel_url(package, version, env, user_wheel_sources)
            log(f"  {package}: {url[:80]}...")
        if regular_packages:
            log("  Regular packages:")
            for pkg in regular_packages:
                log(f"    {pkg}")
        return True

    if cuda_packages:
        log(f"\nInstalling {len(cuda_packages)} CUDA packages...")
        for req in cuda_packages:
            package, version = parse_wheel_requirement(req)
            _install_cuda_package(package, version, env, user_wheel_sources, log)

    if regular_packages:
        log(f"\nInstalling {len(regular_packages)} regular packages...")
        _pip_install(regular_packages, no_deps=False, log=log)

    log("\nInstallation complete!")
    return True


def _install_local(
    local_config: LocalConfig,
    node_dir: Path,
    log: Callable[[str], None],
    dry_run: bool,
    user_wheel_sources: Dict[str, str],
) -> bool:
    """Install local packages into current environment."""
    log("Installing local packages into host environment")

    if sys.platform == "win32":
        log("Installing MSVC runtime for Windows...")
        if not dry_run:
            _pip_install(["msvc-runtime"], no_deps=False, log=log)

    env = RuntimeEnv.detect()
    log(f"Detected environment: {env}")

    if not env.cuda_version and local_config.cuda_packages:
        raise CUDANotFoundError(package=", ".join(local_config.cuda_packages.keys()))

    cuda_packages = []
    for pkg, ver in local_config.cuda_packages.items():
        if ver:
            cuda_packages.append(f"{pkg}=={ver}")
        else:
            cuda_packages.append(pkg)

    if dry_run:
        log("\nDry run - would install:")
        for pkg in cuda_packages:
            log(f"  {pkg}")
        if local_config.requirements:
            log("  Regular packages:")
            for pkg in local_config.requirements:
                log(f"    {pkg}")
        return True

    if cuda_packages:
        log(f"\nInstalling {len(cuda_packages)} CUDA packages...")
        for req in cuda_packages:
            package, version = parse_wheel_requirement(req)
            _install_cuda_package(package, version, env, user_wheel_sources, log)

    if local_config.requirements:
        log(f"\nInstalling {len(local_config.requirements)} regular packages...")
        _pip_install(local_config.requirements, no_deps=False, log=log)

    log("\nLocal installation complete!")
    return True


def _resolve_wheel_url(
    package: str,
    version: Optional[str],
    env: RuntimeEnv,
    user_wheel_sources: Dict[str, str],
) -> str:
    """
    Resolve wheel URL for a CUDA package.

    Resolution order:
    1. User's [wheel_sources] in comfy-env.toml (highest priority)
    2. Built-in wheel_sources.yml registry
    3. Error if not found
    """
    pkg_lower = package.lower()
    vars_dict = _build_template_vars(env, version)

    # 1. Check user overrides first
    if pkg_lower in user_wheel_sources:
        template = user_wheel_sources[pkg_lower]
        return _substitute_template(template, vars_dict)

    # 2. Check built-in registry
    if pkg_lower in PACKAGE_REGISTRY:
        config = PACKAGE_REGISTRY[pkg_lower]

        # wheel_template: direct URL
        if "wheel_template" in config:
            effective_version = version or config.get("default_version")
            if not effective_version:
                raise InstallError(f"Package {package} requires version (no default in registry)")
            vars_dict["version"] = effective_version
            return _substitute_template(config["wheel_template"], vars_dict)

        # package_name: PyPI variant (e.g., spconv-cu124)
        if "package_name" in config:
            pkg_name = _substitute_template(config["package_name"], vars_dict)
            return f"pypi:{pkg_name}"  # Special marker for PyPI install

    raise InstallError(
        f"Package {package} not found in registry or user wheel_sources.\n"
        f"Add it to [wheel_sources] in your comfy-env.toml:\n\n"
        f"[wheel_sources]\n"
        f'{package} = "https://example.com/{package}-{{version}}+cu{{cuda_short}}-{{py_tag}}-{{platform}}.whl"'
    )


def _install_cuda_package(
    package: str,
    version: Optional[str],
    env: RuntimeEnv,
    user_wheel_sources: Dict[str, str],
    log: Callable[[str], None],
) -> None:
    """
    Install a single CUDA package.

    Uses wheel_template for direct URL or package_name for PyPI variants.
    """
    url_or_marker = _resolve_wheel_url(package, version, env, user_wheel_sources)

    if url_or_marker.startswith("pypi:"):
        # PyPI variant package (e.g., spconv-cu124)
        pkg_name = url_or_marker[5:]  # Strip "pypi:" prefix
        pkg_spec = f"{pkg_name}=={version}" if version else pkg_name
        log(f"  Installing {package} as {pkg_spec} from PyPI...")
        _pip_install([pkg_spec], no_deps=False, log=log)
    else:
        # Direct wheel URL
        log(f"  Installing {package}...")
        log(f"    URL: {url_or_marker}")
        _pip_install([url_or_marker], no_deps=True, log=log)


def _build_template_vars(env: RuntimeEnv, version: Optional[str] = None) -> Dict[str, str]:
    """Build template variables dict from RuntimeEnv."""
    vars_dict = env.as_dict()

    if version:
        vars_dict["version"] = version

    # Add cuda_short2 for spconv (e.g., "124" not "1240")
    if env.cuda_version:
        vars_dict["cuda_short2"] = get_cuda_short2(env.cuda_version)

    return vars_dict


def _substitute_template(template: str, vars_dict: Dict[str, str]) -> str:
    """Substitute {var} placeholders in template with values from vars_dict."""
    result = template
    for key, value in vars_dict.items():
        if value is not None:
            result = result.replace(f"{{{key}}}", str(value))
    return result


def _pip_install(
    packages: List[str],
    no_deps: bool = False,
    log: Callable[[str], None] = print,
) -> None:
    """Install packages using pip (prefers uv if available)."""
    pip_cmd = _get_pip_command()

    args = pip_cmd + ["install"]
    if no_deps:
        args.append("--no-deps")
    args.extend(packages)

    log(f"Running: {' '.join(args[:3])}... ({len(packages)} packages)")

    result = subprocess.run(args, capture_output=True, text=True)

    if result.returncode != 0:
        raise InstallError(
            f"Failed to install packages",
            exit_code=result.returncode,
            stderr=result.stderr,
        )


def _get_pip_command() -> List[str]:
    """Get the pip command to use (prefers uv if available)."""
    uv_path = shutil.which("uv")
    if uv_path:
        return [uv_path, "pip"]
    return [sys.executable, "-m", "pip"]


def verify_installation(
    packages: List[str],
    log: Callable[[str], None] = print,
) -> bool:
    """Verify that packages are importable."""
    all_ok = True
    for package in packages:
        import_name = package.replace("-", "_").split("[")[0]
        try:
            __import__(import_name)
            log(f"  {package}: OK")
        except ImportError as e:
            log(f"  {package}: FAILED ({e})")
            all_ok = False
    return all_ok


