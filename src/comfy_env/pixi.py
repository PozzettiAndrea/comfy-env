"""
Pixi integration for comfy-env.

Pixi is a fast package manager that supports both conda and pip packages.
When an environment has conda packages defined, we use pixi as the backend
instead of uv.

See: https://pixi.sh/
"""

import os
import platform
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .env.config import IsolatedEnv, CondaConfig


# Pixi download URLs by platform
PIXI_URLS = {
    ("Linux", "x86_64"): "https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-unknown-linux-musl",
    ("Linux", "aarch64"): "https://github.com/prefix-dev/pixi/releases/latest/download/pixi-aarch64-unknown-linux-musl",
    ("Darwin", "x86_64"): "https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-apple-darwin",
    ("Darwin", "arm64"): "https://github.com/prefix-dev/pixi/releases/latest/download/pixi-aarch64-apple-darwin",
    ("Windows", "AMD64"): "https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-pc-windows-msvc.exe",
}


def get_pixi_path() -> Optional[Path]:
    """
    Find the pixi executable.

    Checks:
    1. System PATH
    2. ~/.pixi/bin/pixi
    3. ~/.local/bin/pixi

    Returns:
        Path to pixi executable, or None if not found.
    """
    # Check system PATH
    pixi_cmd = shutil.which("pixi")
    if pixi_cmd:
        return Path(pixi_cmd)

    # Check common install locations
    home = Path.home()
    candidates = [
        home / ".pixi" / "bin" / "pixi",
        home / ".local" / "bin" / "pixi",
    ]

    # Add .exe on Windows
    if sys.platform == "win32":
        candidates = [p.with_suffix(".exe") for p in candidates]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def ensure_pixi(
    install_dir: Optional[Path] = None,
    log: Callable[[str], None] = print,
) -> Path:
    """
    Ensure pixi is installed, downloading if necessary.

    Args:
        install_dir: Directory to install pixi to. Defaults to ~/.local/bin/
        log: Logging callback.

    Returns:
        Path to pixi executable.

    Raises:
        RuntimeError: If pixi cannot be installed.
    """
    # Check if already installed
    existing = get_pixi_path()
    if existing:
        log(f"Found pixi at: {existing}")
        return existing

    log("Pixi not found, downloading...")

    # Determine install location
    if install_dir is None:
        install_dir = Path.home() / ".local" / "bin"
    install_dir.mkdir(parents=True, exist_ok=True)

    # Determine download URL
    system = platform.system()
    machine = platform.machine()

    # Normalize machine name
    if machine in ("x86_64", "AMD64"):
        machine = "x86_64" if system != "Windows" else "AMD64"
    elif machine in ("arm64", "aarch64"):
        machine = "arm64" if system == "Darwin" else "aarch64"

    url_key = (system, machine)
    if url_key not in PIXI_URLS:
        raise RuntimeError(
            f"No pixi download available for {system}/{machine}. "
            f"Available: {list(PIXI_URLS.keys())}"
        )

    url = PIXI_URLS[url_key]
    pixi_path = install_dir / ("pixi.exe" if system == "Windows" else "pixi")

    log(f"Downloading pixi from: {url}")

    # Download using curl or urllib
    try:
        import urllib.request
        urllib.request.urlretrieve(url, pixi_path)
    except Exception as e:
        # Try curl as fallback
        result = subprocess.run(
            ["curl", "-fsSL", "-o", str(pixi_path), url],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to download pixi: {result.stderr}") from e

    # Make executable on Unix
    if system != "Windows":
        pixi_path.chmod(pixi_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    # Verify installation
    result = subprocess.run([str(pixi_path), "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Pixi installation failed: {result.stderr}")

    log(f"Installed pixi {result.stdout.strip()} to: {pixi_path}")
    return pixi_path


def _parse_pypi_requirement(dep: str) -> Tuple[str, Optional[str], List[str]]:
    """
    Parse a pip requirement into (name, version_spec, extras).

    Examples:
        "trimesh[easy]>=4.0.0" -> ("trimesh", ">=4.0.0", ["easy"])
        "numpy>=1.21.0" -> ("numpy", ">=1.21.0", [])
        "torch" -> ("torch", None, [])
        "pkg[a,b]" -> ("pkg", None, ["a", "b"])

    Returns:
        Tuple of (package_name, version_spec_or_None, list_of_extras)
    """
    dep = dep.strip()

    # Match: name[extras]version_spec or name version_spec
    # Package names can contain letters, numbers, underscores, hyphens, and dots
    match = re.match(r'^([a-zA-Z0-9._-]+)(?:\[([^\]]+)\])?(.*)$', dep)
    if not match:
        return dep, None, []

    name = match.group(1)
    extras_str = match.group(2)
    version_spec = match.group(3).strip() if match.group(3) else None

    extras = []
    if extras_str:
        extras = [e.strip() for e in extras_str.split(',')]

    # Return None instead of empty string for version_spec
    if version_spec == "":
        version_spec = None

    return name, version_spec, extras


def create_pixi_toml(
    env_config: IsolatedEnv,
    node_dir: Path,
    log: Callable[[str], None] = print,
) -> Path:
    """
    Generate a pixi.toml file from the environment configuration.

    The generated pixi.toml includes:
    - Project metadata
    - Conda channels
    - Conda dependencies
    - PyPI dependencies (from requirements + no_deps_requirements)

    Args:
        env_config: The isolated environment configuration.
        node_dir: Directory to write pixi.toml to.
        log: Logging callback.

    Returns:
        Path to the generated pixi.toml file.
    """
    # Conda is optional - use defaults if not present
    if env_config.conda:
        conda = env_config.conda
    else:
        from comfy_env.env.config import CondaConfig
        conda = CondaConfig(channels=["conda-forge"], packages=[])
    pixi_toml_path = node_dir / "pixi.toml"

    # Build pixi.toml content
    lines = []

    # Project section
    lines.append("[workspace]")
    lines.append(f'name = "{env_config.name}"')
    lines.append('version = "0.1.0"')

    # Channels
    channels = conda.channels or ["conda-forge"]
    channels_str = ", ".join(f'"{ch}"' for ch in channels)
    lines.append(f"channels = [{channels_str}]")

    # Platforms
    if sys.platform == "linux":
        lines.append('platforms = ["linux-64"]')
    elif sys.platform == "darwin":
        if platform.machine() == "arm64":
            lines.append('platforms = ["osx-arm64"]')
        else:
            lines.append('platforms = ["osx-64"]')
    elif sys.platform == "win32":
        lines.append('platforms = ["win-64"]')

    # System requirements - specify glibc version for proper wheel resolution
    # Ubuntu 22.04+ has glibc 2.35, enabling manylinux_2_35 wheels
    if sys.platform == "linux":
        lines.append("")
        lines.append("[system-requirements]")
        lines.append('libc = { family = "glibc", version = "2.35" }')

    lines.append("")

    # Dependencies section (conda packages)
    lines.append("[dependencies]")
    lines.append(f'python = "{env_config.python}.*"')

    # On Windows, use MKL BLAS to avoid OpenBLAS crashes (numpy blas_fpe_check issue)
    if sys.platform == "win32":
        lines.append('libblas = { version = "*", build = "*mkl" }')

    for pkg in conda.packages:
        # Parse package spec (name=version or name>=version or name<version or just name)
        if ">=" in pkg:
            name, version = pkg.split(">=", 1)
            lines.append(f'{name} = ">={version}"')
        elif "<=" in pkg:
            name, version = pkg.split("<=", 1)
            lines.append(f'{name} = "<={version}"')
        elif "==" in pkg:
            name, version = pkg.split("==", 1)
            lines.append(f'{name} = "=={version}"')
        elif ">" in pkg:
            name, version = pkg.split(">", 1)
            lines.append(f'{name} = ">{version}"')
        elif "<" in pkg:
            name, version = pkg.split("<", 1)
            lines.append(f'{name} = "<{version}"')
        elif "=" in pkg and not pkg.startswith("="):
            # Single = means exact version in conda
            name, version = pkg.split("=", 1)
            lines.append(f'{name} = "=={version}"')
        else:
            # No version, use any
            lines.append(f'{pkg} = "*"')

    lines.append("")

    # PyPI dependencies section
    pypi_deps = []
    special_deps = {}  # For dependencies that need special syntax (path, etc.)

    # Always include comfy-env for worker support
    # Priority: 1. COMFY_LOCAL_WHEELS env var, 2. ~/utils/comfy-env, 3. PyPI
    local_wheels_dir = os.environ.get("COMFY_LOCAL_WHEELS")
    if local_wheels_dir:
        local_wheels = list(Path(local_wheels_dir).glob("comfy_env-*.whl"))
        if local_wheels:
            # Copy wheel to node_dir (next to pixi.toml) for simple relative path
            wheel_name = local_wheels[0].name
            wheel_dest = node_dir / wheel_name
            if not wheel_dest.exists():
                shutil.copy(local_wheels[0], wheel_dest)
            # Reference with simple relative path (forward slashes, no backslash issues)
            special_deps["comfy-env"] = f'{{ path = "./{wheel_name}" }}'
        else:
            pypi_deps.append("comfy-env")
    else:
        # Check for local editable comfy-env at ~/utils/comfy-env
        local_comfy_env = Path.home() / "utils" / "comfy-env"
        if local_comfy_env.exists() and (local_comfy_env / "pyproject.toml").exists():
            # Use forward slashes for TOML compatibility
            path_str = local_comfy_env.as_posix()
            special_deps["comfy-env"] = f'{{ path = "{path_str}", editable = true }}'
        else:
            pypi_deps.append("comfy-env")

    # Add regular requirements
    if env_config.requirements:
        pypi_deps.extend(env_config.requirements)

    # Add CUDA packages with resolved wheel URLs
    if env_config.no_deps_requirements:
        from .registry import PACKAGE_REGISTRY

        # Use fixed CUDA 12.8 / PyTorch 2.8 for pixi environments (modern GPU default)
        # This ensures wheels match what pixi will install, not what the host has
        vars_dict = {
            "cuda_version": "12.8",
            "cuda_short": "128",
            "cuda_short2": "128",
            "cuda_major": "12",
            "torch_version": "2.8.0",
            "torch_short": "280",
            "torch_mm": "28",
            "torch_dotted_mm": "2.8",
        }

        # Platform detection
        if sys.platform == "linux":
            vars_dict["platform"] = "linux_x86_64"
        elif sys.platform == "darwin":
            vars_dict["platform"] = "macosx_arm64" if platform.machine() == "arm64" else "macosx_x86_64"
        elif sys.platform == "win32":
            vars_dict["platform"] = "win_amd64"

        # Python version from pixi env config
        if env_config.python:
            py_parts = env_config.python.split(".")
            py_major = py_parts[0]
            py_minor = py_parts[1] if len(py_parts) > 1 else "0"
            vars_dict["py_version"] = env_config.python
            vars_dict["py_short"] = f"{py_major}{py_minor}"
            vars_dict["py_minor"] = py_minor
            vars_dict["py_tag"] = f"cp{py_major}{py_minor}"

        for req in env_config.no_deps_requirements:
            # Parse requirement (e.g., "cumesh" or "cumesh==0.0.1")
            if "==" in req:
                pkg_name, version = req.split("==", 1)
            else:
                pkg_name = req
                version = None

            pkg_lower = pkg_name.lower()
            if pkg_lower in PACKAGE_REGISTRY:
                config = PACKAGE_REGISTRY[pkg_lower]
                template = config.get("wheel_template")
                if template:
                    # Use version from requirement or default
                    v = version or config.get("default_version")
                    if v:
                        vars_dict["version"] = v

                    # Resolve URL
                    url = template
                    for key, value in vars_dict.items():
                        if value:
                            url = url.replace(f"{{{key}}}", str(value))

                    special_deps[pkg_name] = f'{{ url = "{url}" }}'
                    log(f"  CUDA package {pkg_name}: resolved wheel URL")

    # Add platform-specific requirements
    if sys.platform == "linux" and env_config.linux_requirements:
        pypi_deps.extend(env_config.linux_requirements)
    elif sys.platform == "darwin" and env_config.darwin_requirements:
        pypi_deps.extend(env_config.darwin_requirements)
    elif sys.platform == "win32" and env_config.windows_requirements:
        pypi_deps.extend(env_config.windows_requirements)

    if pypi_deps or special_deps:
        lines.append("[pypi-dependencies]")

        # Add special dependencies first (path-based, etc.)
        for name, value in special_deps.items():
            lines.append(f'{name} = {value}')

        for dep in pypi_deps:
            # Handle git dependencies in two formats:
            # 1. pkg @ git+https://github.com/user/repo.git@commit
            # 2. git+https://github.com/user/repo.git@commit (extract name from URL)
            if "git+" in dep:
                if " @ git+" in dep:
                    # Format: pkg @ git+URL@commit
                    match = re.match(r'^([a-zA-Z0-9._-]+)\s*@\s*git\+(.+?)(?:@([a-f0-9]+))?$', dep)
                    if match:
                        pkg_name = match.group(1)
                        git_url = match.group(2)
                        rev = match.group(3)
                else:
                    # Format: git+URL@commit (extract package name from repo name)
                    match = re.match(r'^git\+(.+?)(?:@([a-f0-9]+))?$', dep)
                    if match:
                        git_url = match.group(1)
                        rev = match.group(2)
                        # Extract package name from URL (repo name without .git)
                        repo_match = re.search(r'/([^/]+?)(?:\.git)?$', git_url)
                        pkg_name = repo_match.group(1) if repo_match else git_url.split('/')[-1].replace('.git', '')

                if match:
                    if rev:
                        lines.append(f'{pkg_name} = {{ git = "{git_url}", rev = "{rev}" }}')
                    else:
                        lines.append(f'{pkg_name} = {{ git = "{git_url}" }}')
                    continue

            # Parse pip requirement format to pixi format
            # Handles extras like trimesh[easy]>=4.0.0
            name, version_spec, extras = _parse_pypi_requirement(dep)

            if extras:
                # Use table syntax for packages with extras
                # e.g., trimesh = { version = ">=4.0.0", extras = ["easy"] }
                extras_json = "[" + ", ".join(f'"{e}"' for e in extras) + "]"
                if version_spec:
                    lines.append(f'{name} = {{ version = "{version_spec}", extras = {extras_json} }}')
                else:
                    lines.append(f'{name} = {{ version = "*", extras = {extras_json} }}')
            else:
                # Simple syntax for packages without extras
                if version_spec:
                    lines.append(f'{name} = "{version_spec}"')
                else:
                    lines.append(f'{name} = "*"')

    content = "\n".join(lines) + "\n"

    # Write the file
    pixi_toml_path.write_text(content)
    log(f"Generated pixi.toml at: {pixi_toml_path}")

    return pixi_toml_path


def clean_pixi_artifacts(
    node_dir: Path,
    env_name: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> None:
    """
    Remove previous pixi installation artifacts.

    This ensures a clean state before generating a new pixi.toml,
    preventing stale lock files or cached environments from causing conflicts.

    Args:
        node_dir: Directory containing the pixi artifacts.
        env_name: Environment name (for removing _env_ symlink).
        log: Logging callback.
    """
    pixi_toml = node_dir / "pixi.toml"
    pixi_lock = node_dir / "pixi.lock"
    pixi_dir = node_dir / ".pixi"

    if pixi_toml.exists():
        pixi_toml.unlink()
        log("  Removed previous pixi.toml")
    if pixi_lock.exists():
        pixi_lock.unlink()
        log("  Removed previous pixi.lock")
    if pixi_dir.exists():
        shutil.rmtree(pixi_dir)
        log("  Removed previous .pixi/ directory")

    # Remove _env_ symlink if it exists
    if env_name:
        symlink_path = node_dir / f"_env_{env_name}"
        if symlink_path.is_symlink():
            symlink_path.unlink()
            log(f"  Removed previous _env_{env_name} symlink")


def pixi_install(
    env_config: IsolatedEnv,
    node_dir: Path,
    log: Callable[[str], None] = print,
    dry_run: bool = False,
) -> bool:
    """
    Install conda and pip packages using pixi.

    This is the main entry point for pixi-based installation. It:
    1. Cleans previous pixi artifacts
    2. Ensures pixi is installed
    3. Generates pixi.toml from the config
    4. Runs `pixi install` to install all dependencies

    Args:
        env_config: The isolated environment configuration.
        node_dir: Directory containing the node (where pixi.toml will be created).
        log: Logging callback.
        dry_run: If True, only show what would be done.

    Returns:
        True if installation succeeded.

    Raises:
        RuntimeError: If installation fails.
    """
    log(f"Installing {env_config.name} with pixi backend...")

    if dry_run:
        log("Dry run - would:")
        log(f"  - Clean previous pixi artifacts")
        log(f"  - Ensure pixi is installed")
        log(f"  - Generate pixi.toml in {node_dir}")
        if env_config.conda:
            log(f"  - Install {len(env_config.conda.packages)} conda packages")
        if env_config.requirements:
            log(f"  - Install {len(env_config.requirements)} pip packages")
        if env_config.no_deps_requirements:
            log(f"  - Install {len(env_config.no_deps_requirements)} CUDA packages: {', '.join(env_config.no_deps_requirements)}")
        return True

    # Clean previous pixi artifacts
    clean_pixi_artifacts(node_dir, env_config.name, log)

    # Ensure pixi is installed
    pixi_path = ensure_pixi(log=log)

    # Generate pixi.toml
    pixi_toml = create_pixi_toml(env_config, node_dir, log)

    # Run pixi install
    log("Running pixi install...")
    result = subprocess.run(
        [str(pixi_path), "install"],
        cwd=node_dir,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        log(f"pixi install failed:")
        log(result.stderr)
        raise RuntimeError(f"pixi install failed: {result.stderr}")

    if result.stdout:
        # Log output, but filter for key info
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                log(f"  {line}")

    log("pixi install completed successfully!")

    # Create _env_{name} link for compatibility with uv backend
    # This ensures code that expects _env_envname/bin/python works with pixi
    symlink_path = node_dir / f"_env_{env_config.name}"
    pixi_env_path = node_dir / ".pixi" / "envs" / "default"

    if pixi_env_path.exists():
        # Remove existing symlink/junction or directory if present
        if symlink_path.is_symlink() or (sys.platform == "win32" and symlink_path.is_dir()):
            # On Windows, junctions appear as directories but can be removed with rmdir
            try:
                symlink_path.unlink()
            except (OSError, PermissionError):
                # Junction on Windows - remove with rmdir (doesn't delete contents)
                subprocess.run(["cmd", "/c", "rmdir", str(symlink_path)], capture_output=True)
        elif symlink_path.exists():
            shutil.rmtree(symlink_path)

        # On Windows, use directory junctions (no admin required) instead of symlinks
        if sys.platform == "win32":
            # mklink /J creates a directory junction (no admin privileges needed)
            result = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(symlink_path), str(pixi_env_path)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                log(f"Created junction: _env_{env_config.name} -> .pixi/envs/default")
            else:
                log(f"Warning: Failed to create junction: {result.stderr}")
        else:
            symlink_path.symlink_to(pixi_env_path)
            log(f"Created symlink: _env_{env_config.name} -> .pixi/envs/default")

    return True


def get_pixi_python(node_dir: Path) -> Optional[Path]:
    """
    Get the path to the Python interpreter in the pixi environment.

    Args:
        node_dir: Directory containing pixi.toml.

    Returns:
        Path to Python executable in the pixi env, or None if not found.
    """
    # Pixi creates .pixi/envs/default/ in the project directory
    env_dir = node_dir / ".pixi" / "envs" / "default"

    if sys.platform == "win32":
        python_path = env_dir / "python.exe"
    else:
        python_path = env_dir / "bin" / "python"

    if python_path.exists():
        return python_path

    return None


def pixi_run(
    command: List[str],
    node_dir: Path,
    log: Callable[[str], None] = print,
) -> subprocess.CompletedProcess:
    """
    Run a command in the pixi environment.

    Args:
        command: Command and arguments to run.
        node_dir: Directory containing pixi.toml.
        log: Logging callback.

    Returns:
        CompletedProcess result.
    """
    pixi_path = get_pixi_path()
    if not pixi_path:
        raise RuntimeError("Pixi not found")

    full_cmd = [str(pixi_path), "run"] + command
    log(f"Running: pixi run {' '.join(command)}")

    return subprocess.run(
        full_cmd,
        cwd=node_dir,
        capture_output=True,
        text=True,
    )
