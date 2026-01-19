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
    install(config="comfy-env.toml", mode="inplace")

    # Isolated environment
    install(config="comfy-env.toml", mode="isolated")
"""

import inspect
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

from .env.config import IsolatedEnv, LocalConfig, NodeReq, SystemConfig
from .env.config_file import load_config, discover_config
from .env.manager import IsolatedEnvManager
from .errors import CUDANotFoundError, DependencyError, InstallError, WheelNotFoundError
from .pixi import pixi_install
from .registry import PACKAGE_REGISTRY, get_cuda_short2
from .resolver import RuntimeEnv, WheelResolver, parse_wheel_requirement
from .index_resolver import resolve_wheel_from_index


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

        # Check if apt-get is available
        if not shutil.which("apt-get"):
            log("  Warning: apt-get not found. Cannot install system packages.")
            log(f"  Please install manually: {', '.join(packages)}")
            return True  # Don't fail - just warn

        # Check if we can use sudo
        sudo_available = shutil.which("sudo") is not None

        try:
            if sudo_available:
                # Try with sudo
                log("  Running apt-get update...")
                update_result = subprocess.run(
                    ["sudo", "apt-get", "update"],
                    capture_output=True,
                    text=True,
                )
                if update_result.returncode != 0:
                    log(f"  Warning: apt-get update failed: {update_result.stderr.strip()}")
                    # Continue anyway - packages might already be cached

                log(f"  Installing: {', '.join(packages)}")
                install_cmd = ["sudo", "apt-get", "install", "-y"] + packages
                install_result = subprocess.run(
                    install_cmd,
                    capture_output=True,
                    text=True,
                )

                if install_result.returncode != 0:
                    log(f"  Warning: apt-get install failed: {install_result.stderr.strip()}")
                    log(f"  Please install manually: sudo apt-get install {' '.join(packages)}")
                    return True  # Don't fail - just warn
                else:
                    log("  System packages installed successfully.")
                    return True
            else:
                log("  Warning: sudo not available. Cannot install system packages automatically.")
                log(f"  Please install manually: sudo apt-get install {' '.join(packages)}")
                return True  # Don't fail - just warn

        except Exception as e:
            log(f"  Warning: Failed to install system packages: {e}")
            log(f"  Please install manually: sudo apt-get install {' '.join(packages)}")
            return True  # Don't fail - just warn

    elif platform == "darwin":
        packages = system_config.darwin
        if not packages:
            return True

        log(f"System packages for macOS: {', '.join(packages)}")
        log("  Note: macOS system package installation not yet implemented.")
        log(f"  Please install manually with Homebrew: brew install {' '.join(packages)}")
        return True

    elif platform == "win32":
        packages = system_config.windows
        if not packages:
            return True

        log(f"System packages for Windows: {', '.join(packages)}")
        log("  Note: Windows system package installation not yet implemented.")
        log(f"  Please install manually.")
        return True

    return True


def _install_node_dependencies(
    node_reqs: List[NodeReq],
    node_dir: Path,
    log: Callable[[str], None],
    dry_run: bool = False,
) -> bool:
    """
    Install node dependencies (other ComfyUI custom nodes).

    Args:
        node_reqs: List of NodeReq objects from [node_reqs] config section.
        node_dir: Directory of the current node (used to find custom_nodes/).
        log: Logging callback.
        dry_run: If True, show what would be installed without installing.

    Returns:
        True if installation succeeded or no dependencies needed.
    """
    from .nodes import install_node_deps

    # Detect custom_nodes directory (parent of current node)
    custom_nodes_dir = node_dir.parent

    log(f"\nInstalling {len(node_reqs)} node dependencies...")

    if dry_run:
        for req in node_reqs:
            node_path = custom_nodes_dir / req.name
            status = "exists" if node_path.exists() else "would clone"
            log(f"  {req.name}: {status}")
        return True

    # Track visited nodes to prevent cycles
    # Start with current node's directory name
    visited: Set[str] = {node_dir.name}

    install_node_deps(node_reqs, custom_nodes_dir, log, visited)

    return True


def install(
    config: Optional[Union[str, Path]] = None,
    node_dir: Optional[Path] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    dry_run: bool = False,
    verify_wheels: bool = False,
    **kwargs,  # Accept but ignore deprecated 'mode' parameter
) -> bool:
    """
    Install dependencies from a comfy-env.toml configuration.

    This is the main entry point for installing CUDA dependencies.
    Mode is auto-detected from config:
    - [envname] with python = "X.X" → isolated (creates separate venv)
    - [envname] without python → inplace (installs into current env)
    - [local.cuda] → local (installs into current env)

    Args:
        config: Path to config file. If None, auto-discovers in node_dir.
        node_dir: Directory to search for config. Defaults to current directory.
        log_callback: Optional callback for logging. Defaults to print.
        dry_run: If True, show what would be installed without installing.
        verify_wheels: If True, verify wheel URLs exist before installing.

    Returns:
        True if installation succeeded.

    Raises:
        FileNotFoundError: If config file not found.
        WheelNotFoundError: If required wheel cannot be resolved.
        InstallError: If installation fails.

    Example:
        # Simple usage - auto-discover config
        install()

        # Explicit config file
        install(config="comfy-env.toml")

        # Dry run to see what would be installed
        install(dry_run=True)
    """
    log = log_callback or print
    node_dir = Path(node_dir) if node_dir else Path.cwd()

    # Load full configuration (includes tools)
    full_config = _load_full_config(config, node_dir)
    if full_config is None:
        raise FileNotFoundError(
            "No configuration file found. "
            "Create comfy-env.toml or specify path explicitly."
        )

    # Install node dependencies first (other ComfyUI custom nodes)
    # These may have their own system packages and Python packages
    if full_config.node_reqs:
        _install_node_dependencies(full_config.node_reqs, node_dir, log, dry_run)

    # Install system packages (apt, brew, etc.)
    # These need to be installed before Python packages that depend on them
    if full_config.has_system:
        _install_system_packages(full_config.system, log, dry_run)

    # Get environment config
    env_config = full_config.default_env
    if env_config is None and not full_config.has_local:
        log("No packages to install")
        return True

    if env_config:
        log(f"Found configuration: {env_config.name}")

    # Check if environment uses conda packages (pixi backend)
    if env_config and env_config.uses_conda:
        log(f"Environment uses conda packages - using pixi backend")
        return pixi_install(env_config, node_dir, log, dry_run)

    # Auto-detect mode based on config:
    # - env_config.python set → isolated (creates separate venv)
    # - env_config without python → inplace (installs into current env)
    # - only [local.cuda] → local (installs into current env)
    if env_config:
        if env_config.python:
            return _install_isolated(env_config, node_dir, log, dry_run)
        else:
            return _install_inplace(env_config, node_dir, log, dry_run, verify_wheels)
    elif full_config.has_local:
        # Handle [local.cuda] and [local.packages] without isolated env
        return _install_local(full_config.local, node_dir, log, dry_run)
    else:
        return True


def _load_full_config(config: Optional[Union[str, Path]], node_dir: Path):
    """Load full EnvManagerConfig (includes tools)."""
    from .env.config import EnvManagerConfig
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
    verify_wheels: bool,
) -> bool:
    """Install in-place into current environment using the package registry."""
    log("Installing in-place mode")

    # Install MSVC runtime on Windows (required for CUDA/PyTorch native extensions)
    if sys.platform == "win32":
        log("Installing MSVC runtime for Windows...")
        if not dry_run:
            _pip_install(["msvc-runtime"], no_deps=False, log=log)

    # Detect runtime environment
    env = RuntimeEnv.detect()
    log(f"Detected environment: {env}")

    # Check CUDA requirement
    if not env.cuda_version:
        cuda_packages = _get_cuda_packages(env_config)
        if cuda_packages:
            raise CUDANotFoundError(package=", ".join(cuda_packages))

    # Get packages to install
    cuda_packages = _get_cuda_packages(env_config)
    regular_packages = _get_regular_packages(env_config)

    # Legacy wheel sources from config (for packages not in registry)
    legacy_wheel_sources = env_config.wheel_sources or []

    if dry_run:
        log("\nDry run - would install:")
        for req in cuda_packages:
            package, version = parse_wheel_requirement(req)
            install_info = _get_install_info(package, version, env, legacy_wheel_sources)
            log(f"  {package}: {install_info['description']}")
        if regular_packages:
            log("  Regular packages:")
            for pkg in regular_packages:
                log(f"    {pkg}")
        return True

    # Install CUDA packages using appropriate method per package
    if cuda_packages:
        log(f"\nInstalling {len(cuda_packages)} CUDA packages...")
        for req in cuda_packages:
            package, version = parse_wheel_requirement(req)
            _install_cuda_package(package, version, env, legacy_wheel_sources, log)

    # Install regular packages
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
) -> bool:
    """Install local packages into current environment (no isolated venv)."""
    log("Installing local packages into host environment")

    # Install MSVC runtime on Windows (required for CUDA/PyTorch native extensions)
    if sys.platform == "win32":
        log("Installing MSVC runtime for Windows...")
        if not dry_run:
            _pip_install(["msvc-runtime"], no_deps=False, log=log)

    # Detect runtime environment
    env = RuntimeEnv.detect()
    log(f"Detected environment: {env}")

    # Check CUDA requirement
    if not env.cuda_version and local_config.cuda_packages:
        raise CUDANotFoundError(package=", ".join(local_config.cuda_packages.keys()))

    # Convert cuda_packages dict to list of specs
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

    # Install CUDA packages
    if cuda_packages:
        log(f"\nInstalling {len(cuda_packages)} CUDA packages...")
        for req in cuda_packages:
            package, version = parse_wheel_requirement(req)
            _install_cuda_package(package, version, env, [], log)

    # Install regular packages
    if local_config.requirements:
        log(f"\nInstalling {len(local_config.requirements)} regular packages...")
        _pip_install(local_config.requirements, no_deps=False, log=log)

    log("\nLocal installation complete!")
    return True


def _get_install_info(
    package: str,
    version: Optional[str],
    env: RuntimeEnv,
    legacy_wheel_sources: List[str],
) -> Dict[str, str]:
    """Get installation info for a package (for dry-run output)."""
    pkg_lower = package.lower()

    if pkg_lower in PACKAGE_REGISTRY:
        config = PACKAGE_REGISTRY[pkg_lower]
        method = config["method"]
        index_url = _substitute_template(config.get("index_url", ""), env)

        if method == "index":
            version_info = ""
            if "version_template" in config:
                resolved_version = _substitute_template(config["version_template"], env)
                version_info = f" (version {resolved_version})"
            return {"method": method, "description": f"from index {index_url}{version_info}"}
        elif method == "github_index":
            return {"method": method, "description": f"from {index_url}"}
        elif method == "find_links":
            return {"method": method, "description": f"from {index_url}"}
        elif method == "pypi_variant":
            vars_dict = env.as_dict()
            if env.cuda_version:
                vars_dict["cuda_short2"] = get_cuda_short2(env.cuda_version)
            actual_pkg = _substitute_template(config["package_template"], vars_dict)
            return {"method": method, "description": f"as {actual_pkg} from PyPI"}
        elif method == "github_release":
            sources = config.get("sources", [])
            source_names = [s.get("name", "unknown") for s in sources]
            return {"method": method, "description": f"from GitHub ({', '.join(source_names)})"}
    elif legacy_wheel_sources:
        return {"method": "legacy", "description": f"from config wheel_sources"}
    else:
        return {"method": "pypi", "description": "from PyPI"}


def _install_cuda_package(
    package: str,
    version: Optional[str],
    env: RuntimeEnv,
    legacy_wheel_sources: List[str],
    log: Callable[[str], None],
) -> None:
    """Install a single CUDA package using the appropriate method from registry."""
    pkg_lower = package.lower()

    # Check if package is in registry
    if pkg_lower in PACKAGE_REGISTRY:
        config = PACKAGE_REGISTRY[pkg_lower]
        method = config["method"]

        if method == "index":
            # PEP 503 index - try to resolve exact wheel URL first
            index_url = _substitute_template(config["index_url"], env)
            # Check for version_template (e.g., detectron2 with embedded torch/cuda version)
            if "version_template" in config:
                resolved_version = _substitute_template(config["version_template"], env)
                pkg_spec = f"{package}=={resolved_version}"
            else:
                pkg_spec = f"{package}=={version}" if version else package
            log(f"  Installing {package} (index)...")

            # Resolve version: use provided version, default_version, or None
            effective_version = version if version and version != "*" else config.get("default_version")

            # Check for wheel_template first (direct URL construction, no index parsing)
            if "wheel_template" in config and effective_version:
                vars_dict = env.as_dict()
                vars_dict["version"] = effective_version
                wheel_url = _substitute_template(config["wheel_template"], vars_dict)
                log(f"    Wheel: {wheel_url}")
                _pip_install([wheel_url], no_deps=True, log=log)
            else:
                # Try to resolve exact wheel URL from index
                actual_version = resolved_version if "version_template" in config else version
                vars_dict = env.as_dict()
                wheel_url = resolve_wheel_from_index(index_url, package, vars_dict, actual_version)
                if wheel_url:
                    # Install from resolved URL directly (guarantees we get what we resolved)
                    log(f"    Wheel: {wheel_url}")
                    _pip_install([wheel_url], no_deps=True, log=log)
                else:
                    raise InstallError(
                        f"Failed to resolve wheel URL for {package} from index {index_url}. "
                        "No matching wheel found and PyPI fallback is disabled.",
                    )

        elif method == "github_index":
            # GitHub Pages index - try to resolve exact wheel URL first
            index_url = _substitute_template(config["index_url"], env)
            pkg_spec = f"{package}=={version}" if version else package
            log(f"  Installing {package} (github_index)...")
            # Try to resolve exact wheel URL from find-links page
            vars_dict = env.as_dict()
            wheel_url = resolve_wheel_from_index(index_url, package, vars_dict, version)
            if wheel_url:
                # Install from resolved URL directly (guarantees we get what we resolved)
                log(f"    Wheel: {wheel_url}")
                _pip_install([wheel_url], no_deps=True, log=log)
            else:
                # Fallback to find-links based resolution
                log(f"    Find-links: {index_url}")
                log(f"    Package: {pkg_spec}")
                _pip_install_with_find_links(pkg_spec, index_url, log)

        elif method == "find_links":
            # Generic find-links (e.g., PyG) - try to resolve exact wheel URL first
            index_url = _substitute_template(config["index_url"], env)
            pkg_spec = f"{package}=={version}" if version else package
            log(f"  Installing {package} (find_links)...")
            # Try to resolve exact wheel URL from find-links page
            vars_dict = env.as_dict()
            wheel_url = resolve_wheel_from_index(index_url, package, vars_dict, version)
            if wheel_url:
                # Install from resolved URL directly (guarantees we get what we resolved)
                log(f"    Wheel: {wheel_url}")
                _pip_install([wheel_url], no_deps=True, log=log)
            else:
                # Fallback to find-links based resolution
                log(f"    Find-links: {index_url}")
                log(f"    Package: {pkg_spec}")
                _pip_install_with_find_links(pkg_spec, index_url, log)

        elif method == "pypi_variant":
            # Transform package name based on CUDA version
            vars_dict = env.as_dict()
            if env.cuda_version:
                vars_dict["cuda_short2"] = get_cuda_short2(env.cuda_version)
            actual_package = _substitute_template(config["package_template"], vars_dict)
            pkg_spec = f"{actual_package}=={version}" if version else actual_package
            log(f"  Installing {package} (pypi_variant)...")
            log(f"    PyPI variant: {pkg_spec}")
            _pip_install([pkg_spec], no_deps=False, log=log)

        elif method == "github_release":
            # Direct wheel URL from GitHub releases with fallback sources
            _install_from_github_release(package, version, env, config, log)

    elif legacy_wheel_sources:
        # Fall back to legacy wheel sources from config
        log(f"  Installing {package} from config wheel_sources...")
        resolver = WheelResolver()
        if version:
            try:
                url = resolver.resolve(package, version, env, verify=False)
                _pip_install([url], no_deps=True, log=log)
            except WheelNotFoundError:
                # Try with find-links
                pkg_spec = f"{package}=={version}"
                for source in legacy_wheel_sources:
                    source_url = _substitute_template(source, env)
                    try:
                        _pip_install_with_find_links(pkg_spec, source_url, log)
                        return
                    except InstallError:
                        continue
                raise WheelNotFoundError(
                    package=package,
                    version=version,
                    env=env,
                    tried_urls=legacy_wheel_sources,
                    reason="Not found in any wheel source",
                )
    else:
        # Package not in registry - try regular pip install (e.g., spconv-cu126)
        log(f"  Installing {package} from PyPI...")
        pkg_spec = f"{package}=={version}" if version else package
        _pip_install([pkg_spec], no_deps=False, log=log)


def _substitute_template(template: str, env_or_dict: Union[RuntimeEnv, Dict[str, str]]) -> str:
    """Substitute template variables with runtime environment values."""
    if isinstance(env_or_dict, dict):
        vars_dict = env_or_dict.copy()
    else:
        vars_dict = env_or_dict.as_dict()
        # Add py_minor for pytorch3d URL pattern
        if env_or_dict.python_version:
            vars_dict["py_minor"] = env_or_dict.python_version.split(".")[-1]

    result = template
    for key, value in vars_dict.items():
        if value is not None:
            result = result.replace(f"{{{key}}}", str(value))
    return result


def _pip_install_with_index(
    package: str,
    index_url: str,
    log: Callable[[str], None],
) -> None:
    """Install package using pip with --extra-index-url."""
    pip_cmd = _get_pip_command()
    args = pip_cmd + ["install", "--extra-index-url", index_url, package]

    log(f"    Running: pip install --extra-index-url ... {package}")
    result = subprocess.run(args, capture_output=True, text=True)

    if result.returncode != 0:
        raise InstallError(
            f"Failed to install {package}",
            exit_code=result.returncode,
            stderr=result.stderr,
        )


def _pip_install_with_find_links(
    package: str,
    find_links_url: str,
    log: Callable[[str], None],
) -> None:
    """Install package using pip with --find-links."""
    pip_cmd = _get_pip_command()
    args = pip_cmd + ["install", "--find-links", find_links_url, package]

    log(f"    Running: pip install --find-links ... {package}")
    result = subprocess.run(args, capture_output=True, text=True)

    if result.returncode != 0:
        raise InstallError(
            f"Failed to install {package}",
            exit_code=result.returncode,
            stderr=result.stderr,
        )


def _install_from_github_release(
    package: str,
    version: Optional[str],
    env: RuntimeEnv,
    config: Dict[str, Any],
    log: Callable[[str], None],
) -> None:
    """Install package from GitHub release wheels with fallback sources.

    This method handles packages like flash-attn that have multiple wheel
    sources for different platforms (Linux: Dao-AILab, mjun0812; Windows: bdashore3).
    """
    if not version:
        raise InstallError(
            f"Package {package} requires explicit version for github_release method"
        )

    sources = config.get("sources", [])
    if not sources:
        raise InstallError(f"No sources configured for {package}")

    # Build template variables
    vars_dict = env.as_dict()
    vars_dict["version"] = version

    # Add py_tag (e.g., "cp310")
    vars_dict["py_tag"] = f"cp{env.python_short}"

    # Add cuda_major (e.g., "12") for Dao-AILab URL pattern
    if env.cuda_version:
        vars_dict["cuda_major"] = env.cuda_version.split(".")[0]

    # Filter sources by platform
    current_platform = env.platform_tag
    compatible_sources = [
        s for s in sources
        if current_platform in s.get("platforms", [])
    ]

    if not compatible_sources:
        available = set()
        for s in sources:
            available.update(s.get("platforms", []))
        raise InstallError(
            f"No {package} wheels available for platform {current_platform}. "
            f"Available platforms: {', '.join(sorted(available))}"
        )

    # Try each source in order
    errors = []
    for source in compatible_sources:
        source_name = source.get("name", "unknown")
        url_template = source.get("url_template", "")

        # Substitute template variables
        url = url_template
        for key, value in vars_dict.items():
            if value is not None:
                url = url.replace(f"{{{key}}}", str(value))

        log(f"  Trying {source_name}: {package}=={version}...")
        log(f"    Resolved wheel to: {url}")

        try:
            pip_cmd = _get_pip_command()
            args = pip_cmd + ["install", "--no-deps", url]

            result = subprocess.run(args, capture_output=True, text=True)

            if result.returncode == 0:
                log(f"    Successfully installed from {source_name}")
                return
            else:
                error_msg = result.stderr.strip().split('\n')[-1] if result.stderr else "Unknown error"
                errors.append(f"{source_name}: {error_msg}")
                log(f"    Failed: {error_msg[:80]}...")

        except Exception as e:
            errors.append(f"{source_name}: {str(e)}")
            log(f"    Error: {str(e)[:80]}...")

    # All sources failed
    raise InstallError(
        f"Failed to install {package}=={version} from any source.\n"
        f"Tried sources:\n" + "\n".join(f"  - {e}" for e in errors)
    )


def _get_cuda_packages(env_config: IsolatedEnv) -> List[str]:
    """Extract CUDA packages that need wheel resolution."""
    # For now, treat no_deps_requirements as CUDA packages
    # In future, could parse from [packages.cuda] section
    return env_config.no_deps_requirements or []


def _get_regular_packages(env_config: IsolatedEnv) -> List[str]:
    """Extract regular pip packages."""
    return env_config.requirements or []


def _pip_install(
    packages: List[str],
    no_deps: bool = False,
    log: Callable[[str], None] = print,
) -> None:
    """
    Install packages using pip.

    Args:
        packages: List of packages or URLs to install.
        no_deps: If True, use --no-deps flag.
        log: Logging callback.

    Raises:
        InstallError: If pip install fails.
    """
    # Prefer uv if available for speed
    pip_cmd = _get_pip_command()

    args = pip_cmd + ["install"]
    if no_deps:
        args.append("--no-deps")
    args.extend(packages)

    log(f"Running: {' '.join(args[:3])}... ({len(packages)} packages)")

    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise InstallError(
            f"Failed to install packages",
            exit_code=result.returncode,
            stderr=result.stderr,
        )


def _get_pip_command() -> List[str]:
    """Get the pip command to use (prefers uv if available)."""
    # Check for uv
    uv_path = shutil.which("uv")
    if uv_path:
        return [uv_path, "pip"]

    # Fall back to pip
    return [sys.executable, "-m", "pip"]


def verify_installation(
    packages: List[str],
    log: Callable[[str], None] = print,
) -> bool:
    """
    Verify that packages are importable.

    Args:
        packages: List of package names to verify.
        log: Logging callback.

    Returns:
        True if all packages are importable.
    """
    all_ok = True
    for package in packages:
        # Convert package name to import name
        import_name = package.replace("-", "_").split("[")[0]

        try:
            __import__(import_name)
            log(f"  {package}: OK")
        except ImportError as e:
            log(f"  {package}: FAILED ({e})")
            all_ok = False

    return all_ok


def setup(
    log_callback: Optional[Callable[[str], None]] = None,
    dry_run: bool = False,
) -> bool:
    """
    One-liner setup that auto-discovers config from caller's directory.

    This is the simplest way to install dependencies - just call setup()
    from your install.py and it will find the comfy-env.toml in the same
    directory as the calling script.

    Example:
        # install.py (entire file)
        from comfy_env import setup
        setup()

    Args:
        log_callback: Optional callback for logging. Defaults to print.
        dry_run: If True, show what would be installed without installing.

    Returns:
        True if installation succeeded.

    Raises:
        FileNotFoundError: If no config file found.
        InstallError: If installation fails.
    """
    # Get the caller's directory by inspecting the stack
    frame = inspect.stack()[1]
    caller_file = frame.filename
    caller_dir = Path(caller_file).parent.resolve()

    return install(
        node_dir=caller_dir,
        log_callback=log_callback,
        dry_run=dry_run,
    )
