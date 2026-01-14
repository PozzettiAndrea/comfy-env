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
    install(config="comfyui_env.toml", mode="inplace")

    # Isolated environment
    install(config="comfyui_env.toml", mode="isolated")
"""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .env.config import IsolatedEnv
from .env.config_file import discover_env_config, load_env_from_file
from .env.manager import IsolatedEnvManager
from .errors import CUDANotFoundError, DependencyError, InstallError, WheelNotFoundError
from .registry import PACKAGE_REGISTRY, get_cuda_short2, is_registered
from .resolver import RuntimeEnv, WheelResolver, parse_wheel_requirement


def install(
    config: Optional[Union[str, Path]] = None,
    mode: str = "inplace",
    node_dir: Optional[Path] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    dry_run: bool = False,
    verify_wheels: bool = False,
) -> bool:
    """
    Install dependencies from a comfyui_env.toml configuration.

    This is the main entry point for installing CUDA dependencies.

    Args:
        config: Path to config file. If None, auto-discovers in node_dir.
        mode: Installation mode - "inplace" (current env) or "isolated" (new venv).
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
        install(config="comfyui_env.toml")

        # Isolated mode
        install(mode="isolated")

        # Dry run to see what would be installed
        install(dry_run=True)
    """
    log = log_callback or print
    node_dir = Path(node_dir) if node_dir else Path.cwd()

    # Load configuration
    env_config = _load_config(config, node_dir)
    if env_config is None:
        raise FileNotFoundError(
            "No configuration file found. "
            "Create comfyui_env.toml or specify path explicitly."
        )

    log(f"Found configuration: {env_config.name}")

    if mode == "isolated":
        return _install_isolated(env_config, node_dir, log, dry_run)
    else:
        return _install_inplace(env_config, node_dir, log, dry_run, verify_wheels)


def _load_config(
    config: Optional[Union[str, Path]],
    node_dir: Path,
) -> Optional[IsolatedEnv]:
    """Load configuration from file or auto-discover."""
    if config is not None:
        config_path = Path(config)
        if not config_path.is_absolute():
            config_path = node_dir / config_path
        return load_env_from_file(config_path, node_dir)

    return discover_env_config(node_dir)


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
            return {"method": method, "description": f"from index {index_url}"}
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
            # PEP 503 index - use pip --extra-index-url
            index_url = _substitute_template(config["index_url"], env)
            pkg_spec = f"{package}=={version}" if version else package
            log(f"  Installing {package} from PyG index...")
            _pip_install_with_index(pkg_spec, index_url, log)

        elif method == "github_index":
            # GitHub Pages index - use pip --find-links
            index_url = _substitute_template(config["index_url"], env)
            pkg_spec = f"{package}=={version}" if version else package
            log(f"  Installing {package} from GitHub wheels...")
            _pip_install_with_find_links(pkg_spec, index_url, log)

        elif method == "find_links":
            # Generic find-links (e.g., PyG) - use pip --find-links
            index_url = _substitute_template(config["index_url"], env)
            pkg_spec = f"{package}=={version}" if version else package
            log(f"  Installing {package} from {index_url}...")
            _pip_install_with_find_links(pkg_spec, index_url, log)

        elif method == "pypi_variant":
            # Transform package name based on CUDA version
            vars_dict = env.as_dict()
            if env.cuda_version:
                vars_dict["cuda_short2"] = get_cuda_short2(env.cuda_version)
            actual_package = _substitute_template(config["package_template"], vars_dict)
            pkg_spec = f"{actual_package}=={version}" if version else actual_package
            log(f"  Installing {package} as {actual_package}...")
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
