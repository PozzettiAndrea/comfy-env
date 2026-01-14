"""Load IsolatedEnv configuration from TOML files.

This module provides declarative configuration for isolated environments,
allowing custom nodes to define their requirements in a TOML file instead
of programmatically.

Config file: comfy-env.toml

Simplified config (recommended for CUDA packages only):

    # comfy-env.toml - just list CUDA packages
    [cuda]
    torch-scatter = "2.1.2"
    torch-cluster = "1.6.3"
    spconv = "*"  # latest compatible

Or as a list:

    cuda = [
        "torch-scatter==2.1.2",
        "torch-cluster==1.6.3",
        "spconv",
    ]

Optional overrides:

    [env]
    cuda = "12.4"       # Override auto-detection
    pytorch = "2.5.1"   # Override auto-detection

Full format:

    [env]
    name = "my-node"
    python = "3.10"
    cuda = "auto"

    [packages]
    requirements = ["my-package>=1.0.0"]
    no_deps = ["torch-scatter==2.1.2"]

    [sources]
    wheel_sources = ["https://my-wheels.github.io/"]

Available auto-derived variables:
    - {cuda_version}: Full CUDA version (e.g., "12.8")
    - {cuda_short}: CUDA version without dot (e.g., "128")
    - {pytorch_version}: Full PyTorch version (e.g., "2.9.1")
    - {pytorch_short}: PyTorch version without dots (e.g., "291")
    - {pytorch_mm}: PyTorch major.minor without dot (e.g., "29")
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

# Use built-in tomllib (Python 3.11+) or tomli fallback
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None  # type: ignore

from .config import IsolatedEnv, EnvManagerConfig, LocalConfig, NodeReq
from .detection import detect_cuda_version


# Config file name
CONFIG_FILE_NAMES = [
    "comfy-env.toml",
]


def load_env_from_file(
    path: Path,
    base_dir: Optional[Path] = None,
) -> IsolatedEnv:
    """
    Load IsolatedEnv configuration from a TOML file.

    Args:
        path: Path to the TOML config file
        base_dir: Base directory for resolving relative paths (default: file's parent)

    Returns:
        Configured IsolatedEnv instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
        ImportError: If tomli is not installed (Python < 3.11)

    Example:
        >>> env = load_env_from_file(Path("my_node/comfy-env.toml"))
        >>> print(env.name)
        'my-node'
    """
    if tomllib is None:
        raise ImportError(
            "TOML parsing requires tomli for Python < 3.11. "
            "Install it with: pip install tomli"
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    base_dir = Path(base_dir) if base_dir else path.parent

    with open(path, "rb") as f:
        data = tomllib.load(f)

    return _parse_config(data, base_dir)


def discover_env_config(
    node_dir: Path,
    file_names: Optional[List[str]] = None,
) -> Optional[IsolatedEnv]:
    """
    Auto-discover and load config from a node directory.

    Searches for standard config file names in order of priority.

    Args:
        node_dir: Directory to search for config files
        file_names: Custom list of file names to search (default: CONFIG_FILE_NAMES)

    Returns:
        IsolatedEnv if config found, None otherwise

    Example:
        >>> env = discover_env_config(Path("my_custom_node/"))
        >>> if env:
        ...     print(f"Found config: {env.name}")
        ... else:
        ...     print("No config file found")
    """
    if tomllib is None:
        # Can't parse TOML without the library
        return None

    node_dir = Path(node_dir)
    file_names = file_names or CONFIG_FILE_NAMES

    for name in file_names:
        config_path = node_dir / name
        if config_path.exists():
            return load_env_from_file(config_path, node_dir)

    return None


def _get_default_pytorch_version(cuda_version: Optional[str]) -> str:
    """
    Get default PyTorch version based on CUDA version.

    Args:
        cuda_version: CUDA version (e.g., "12.4", "12.8") or None

    Returns:
        PyTorch version string

    Version Mapping:
        - CUDA 12.4 (Pascal): PyTorch 2.5.1
        - CUDA 12.8 (Turing+): PyTorch 2.8.0
    """
    if cuda_version == "12.4":
        return "2.5.1"  # Legacy: Pascal GPUs
    return "2.8.0"  # Modern: Turing through Blackwell


def _parse_config(data: Dict[str, Any], base_dir: Path) -> IsolatedEnv:
    """
    Parse TOML data into IsolatedEnv.

    Supports both simplified and legacy config formats:

    Simplified (CUDA packages only):
        [packages]
        torch-scatter = "2.1.2"
        torch-cluster = "1.6.3"

    Or as list:
        packages = ["torch-scatter==2.1.2", "torch-cluster==1.6.3"]

    Legacy:
        [env]
        name = "my-node"
        [packages]
        no_deps = ["torch-scatter==2.1.2"]

    Args:
        data: Parsed TOML data
        base_dir: Base directory for resolving relative paths

    Returns:
        Configured IsolatedEnv instance
    """
    env_section = data.get("env", {})
    packages_section = data.get("packages", {})
    sources_section = data.get("sources", {})
    worker_section = data.get("worker", {})
    variables = dict(data.get("variables", {}))  # Copy to avoid mutation

    # Handle CUDA version - default to "auto" if not specified
    cuda = env_section.get("cuda", "auto")
    if cuda == "auto":
        cuda = detect_cuda_version()
    elif cuda == "null" or cuda == "none":
        cuda = None

    # Add auto-derived variables based on CUDA
    if cuda:
        variables.setdefault("cuda_version", cuda)
        variables.setdefault("cuda_short", cuda.replace(".", ""))

    # Handle pytorch version - auto-derive if "auto" or not specified
    pytorch_version = env_section.get("pytorch_version") or env_section.get("pytorch")
    if pytorch_version == "auto" or (pytorch_version is None and cuda):
        pytorch_version = _get_default_pytorch_version(cuda)

    if pytorch_version:
        variables.setdefault("pytorch_version", pytorch_version)
        # Add short version without dots (e.g., "2.9.1" -> "291")
        pytorch_short = pytorch_version.replace(".", "")
        variables.setdefault("pytorch_short", pytorch_short)
        # Add major.minor without dot (e.g., "2.9.1" -> "29") for wheel naming
        parts = pytorch_version.split(".")[:2]
        pytorch_mm = "".join(parts)
        variables.setdefault("pytorch_mm", pytorch_mm)

    # Parse CUDA packages - support multiple formats
    # Priority: [cuda] section > cuda = [...] > legacy [packages] section
    no_deps_requirements = []
    requirements = []

    cuda_section = data.get("cuda", {})

    if cuda_section:
        # New format: [cuda] section or cuda = [...]
        if isinstance(cuda_section, list):
            # Format: cuda = ["torch-scatter==2.1.2", ...]
            no_deps_requirements = [_substitute_vars(req, variables) for req in cuda_section]
        elif isinstance(cuda_section, dict):
            # Format: [cuda] with package = "version" pairs
            for pkg, ver in cuda_section.items():
                if ver == "*" or ver == "":
                    no_deps_requirements.append(pkg)
                else:
                    no_deps_requirements.append(f"{pkg}=={ver}")

    elif isinstance(packages_section, list):
        # Legacy format: packages = ["torch-scatter==2.1.2", ...]
        no_deps_requirements = [_substitute_vars(req, variables) for req in packages_section]

    elif isinstance(packages_section, dict):
        # Check for simplified format: [packages] with key=value pairs
        # vs legacy format: [packages] with requirements/no_deps lists

        has_legacy_keys = any(k in packages_section for k in ["requirements", "no_deps", "requirements_file"])

        if has_legacy_keys:
            # Legacy format
            raw_requirements = packages_section.get("requirements", [])
            requirements = [_substitute_vars(req, variables) for req in raw_requirements]

            raw_no_deps = packages_section.get("no_deps", [])
            no_deps_requirements = [_substitute_vars(req, variables) for req in raw_no_deps]
        else:
            # Simplified format: [packages] with package = "version" pairs
            # All packages are CUDA packages
            for pkg, ver in packages_section.items():
                if ver == "*" or ver == "":
                    no_deps_requirements.append(pkg)
                else:
                    no_deps_requirements.append(f"{pkg}=={ver}")

    # Resolve requirements_file path (relative to base_dir)
    requirements_file = None
    if isinstance(packages_section, dict) and "requirements_file" in packages_section:
        req_file_path = packages_section["requirements_file"]
        requirements_file = base_dir / req_file_path

    # Get wheel sources and index URLs (optional - registry handles most cases now)
    wheel_sources = sources_section.get("wheel_sources", [])
    index_urls = sources_section.get("index_urls", [])

    # Parse worker configuration
    worker_package = worker_section.get("package")
    worker_script = worker_section.get("script")

    return IsolatedEnv(
        name=env_section.get("name", base_dir.name),
        python=env_section.get("python", "3.10"),
        cuda=cuda,
        pytorch_version=pytorch_version,
        requirements=requirements,
        no_deps_requirements=no_deps_requirements,
        requirements_file=requirements_file,
        wheel_sources=wheel_sources,
        index_urls=index_urls,
        worker_package=worker_package,
        worker_script=worker_script,
    )


def _substitute_vars(s: str, variables: Dict[str, str]) -> str:
    """
    Substitute {var_name} placeholders with values from variables dict.

    Args:
        s: String with placeholders like {var_name}
        variables: Dictionary mapping variable names to values

    Returns:
        String with placeholders replaced

    Example:
        >>> _substitute_vars("torch=={pytorch_version}", {"pytorch_version": "2.4.1"})
        'torch==2.4.1'
    """
    for key, value in variables.items():
        s = s.replace(f"{{{key}}}", str(value))
    return s


# =============================================================================
# V2 Schema Parser
# =============================================================================

# Reserved table names that are NOT isolated environments
RESERVED_TABLES = {"local", "node_reqs", "env", "packages", "sources", "cuda", "variables", "worker"}


def load_config(
    path: Path,
    base_dir: Optional[Path] = None,
) -> EnvManagerConfig:
    """
    Load full EnvManagerConfig from a TOML file (v2 schema).

    Supports both v2 schema and legacy format (auto-detected).

    Args:
        path: Path to the TOML config file
        base_dir: Base directory for resolving relative paths

    Returns:
        EnvManagerConfig with local, envs, and node_reqs

    Example:
        >>> config = load_config(Path("comfy-env.toml"))
        >>> if config.has_local:
        ...     print(f"Local CUDA packages: {config.local.cuda_packages}")
        >>> for name, env in config.envs.items():
        ...     print(f"Isolated env: {name}")
    """
    if tomllib is None:
        raise ImportError(
            "TOML parsing requires tomli for Python < 3.11. "
            "Install it with: pip install tomli"
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    base_dir = Path(base_dir) if base_dir else path.parent

    with open(path, "rb") as f:
        data = tomllib.load(f)

    return _parse_config_v2(data, base_dir)


def discover_config(
    node_dir: Path,
    file_names: Optional[List[str]] = None,
) -> Optional[EnvManagerConfig]:
    """
    Auto-discover and load EnvManagerConfig from a node directory.

    Args:
        node_dir: Directory to search for config files
        file_names: Custom list of file names to search

    Returns:
        EnvManagerConfig if found, None otherwise
    """
    if tomllib is None:
        return None

    node_dir = Path(node_dir)
    file_names = file_names or CONFIG_FILE_NAMES

    for name in file_names:
        config_path = node_dir / name
        if config_path.exists():
            return load_config(config_path, node_dir)

    return None


def _parse_config_v2(data: Dict[str, Any], base_dir: Path) -> EnvManagerConfig:
    """
    Parse TOML data into EnvManagerConfig (v2 schema).

    Schema:
        [local.cuda]        - CUDA packages for host
        [local.packages]    - Regular packages for host
        [envname]           - Isolated env (python, cuda, pytorch)
        [envname.cuda]      - CUDA packages for env
        [envname.packages]  - Packages for env
        [node_reqs]         - Node dependencies

    Also supports legacy format for backward compatibility.
    """
    # Detect if this is v2 schema or legacy
    is_v2 = "local" in data or _has_named_env(data)

    if not is_v2:
        # Legacy format - convert to v2 structure
        return _convert_legacy_to_v2(data, base_dir)

    # Parse v2 schema
    local = _parse_local_section(data.get("local", {}))
    envs = _parse_env_sections(data, base_dir)
    node_reqs = _parse_node_reqs(data.get("node_reqs", {}))

    return EnvManagerConfig(
        local=local,
        envs=envs,
        node_reqs=node_reqs,
    )


def _has_named_env(data: Dict[str, Any]) -> bool:
    """Check if data has any named environment tables (not reserved names)."""
    env_indicators = ["python", "cuda", "pytorch", "cuda_version", "pytorch_version"]
    for key in data.keys():
        if key not in RESERVED_TABLES and isinstance(data[key], dict):
            # Check if it looks like an env definition
            section = data[key]
            if any(k in section for k in env_indicators):
                return True
    return False


def _parse_local_section(local_data: Dict[str, Any]) -> LocalConfig:
    """Parse [local] section."""
    cuda_packages = {}
    requirements = []

    # [local.cuda] - CUDA packages
    cuda_section = local_data.get("cuda", {})
    if isinstance(cuda_section, dict):
        for pkg, ver in cuda_section.items():
            cuda_packages[pkg] = ver if ver and ver != "*" else ""

    # [local.packages] - regular packages
    packages_section = local_data.get("packages", {})
    if isinstance(packages_section, dict):
        requirements = packages_section.get("requirements", [])
    elif isinstance(packages_section, list):
        requirements = packages_section

    return LocalConfig(
        cuda_packages=cuda_packages,
        requirements=requirements,
    )


def _parse_env_sections(data: Dict[str, Any], base_dir: Path) -> Dict[str, IsolatedEnv]:
    """Parse named environment sections."""
    envs = {}
    env_indicators = ["python", "cuda", "pytorch", "cuda_version", "pytorch_version"]

    for key, value in data.items():
        if key in RESERVED_TABLES:
            continue
        if not isinstance(value, dict):
            continue

        # Check if this looks like an env definition
        if not any(k in value for k in env_indicators):
            continue

        env = _parse_single_env(key, value, base_dir)
        envs[key] = env

    return envs


def _parse_single_env(name: str, env_data: Dict[str, Any], base_dir: Path) -> IsolatedEnv:
    """Parse a single isolated environment section."""
    # Get basic env config
    python = env_data.get("python", "3.10")
    # Support both "cuda" and "cuda_version" field names (cuda_version avoids conflict with [envname.cuda] table)
    cuda = env_data.get("cuda_version") or env_data.get("cuda", "auto")
    pytorch = env_data.get("pytorch_version") or env_data.get("pytorch", "auto")

    # Handle auto-detection
    if cuda == "auto":
        cuda = detect_cuda_version()
    elif cuda in ("null", "none", None):
        cuda = None

    if pytorch == "auto":
        pytorch = _get_default_pytorch_version(cuda)

    # Parse [envname.cuda] - CUDA packages
    cuda_section = env_data.get("cuda", {})
    no_deps_requirements = []
    if isinstance(cuda_section, dict):
        for pkg, ver in cuda_section.items():
            if ver == "*" or ver == "":
                no_deps_requirements.append(pkg)
            else:
                no_deps_requirements.append(f"{pkg}=={ver}")

    # Parse [envname.packages] - regular packages
    packages_section = env_data.get("packages", {})
    requirements = []
    if isinstance(packages_section, dict):
        requirements = packages_section.get("requirements", [])
    elif isinstance(packages_section, list):
        requirements = packages_section

    return IsolatedEnv(
        name=name,
        python=python,
        cuda=cuda,
        pytorch_version=pytorch,
        requirements=requirements,
        no_deps_requirements=no_deps_requirements,
    )


def _parse_node_reqs(node_reqs_data: Dict[str, Any]) -> List[NodeReq]:
    """Parse [node_reqs] section."""
    reqs = []

    for name, value in node_reqs_data.items():
        if isinstance(value, str):
            # Simple format: VideoHelperSuite = "Kosinkadink/ComfyUI-VideoHelperSuite"
            reqs.append(NodeReq(name=name, repo=value))
        elif isinstance(value, dict):
            # Extended format: VideoHelperSuite = { repo = "..." }
            repo = value.get("repo", "")
            reqs.append(NodeReq(name=name, repo=repo))

    return reqs


def _convert_legacy_to_v2(data: Dict[str, Any], base_dir: Path) -> EnvManagerConfig:
    """Convert legacy config format to v2 EnvManagerConfig."""
    # Parse using legacy parser to get IsolatedEnv
    legacy_env = _parse_config(data, base_dir)

    # Check if this is really just a Type 2 config (local CUDA only, no venv)
    # Type 2 indicators: no [env] section with name, or name matches directory
    env_section = data.get("env", {})
    has_explicit_env = bool(env_section.get("name") or env_section.get("python"))

    if has_explicit_env:
        # This is a Type 1 config (isolated venv)
        return EnvManagerConfig(
            local=LocalConfig(),
            envs={legacy_env.name: legacy_env},
            node_reqs=[],
        )
    else:
        # This is a Type 2 config (local CUDA only)
        cuda_packages = {}
        for req in legacy_env.no_deps_requirements:
            if "==" in req:
                pkg, ver = req.split("==", 1)
                cuda_packages[pkg] = ver
            else:
                cuda_packages[req] = ""

        return EnvManagerConfig(
            local=LocalConfig(
                cuda_packages=cuda_packages,
                requirements=legacy_env.requirements,
            ),
            envs={},
            node_reqs=[],
        )
