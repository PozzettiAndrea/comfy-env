"""Configuration for isolated environments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class LocalConfig:
    """Configuration for local (host environment) installs.

    These packages are installed into ComfyUI's main environment,
    not into an isolated venv.
    """
    cuda_packages: Dict[str, str] = field(default_factory=dict)  # package -> version
    requirements: List[str] = field(default_factory=list)


@dataclass
class NodeReq:
    """A node dependency (another ComfyUI node pack)."""
    name: str
    repo: str  # GitHub repo path, e.g., "Kosinkadink/ComfyUI-VideoHelperSuite"


@dataclass
class EnvManagerConfig:
    """
    Full configuration parsed from comfyui_env.toml.

    Supports the v2 schema:
        [local.cuda]        - CUDA packages for host environment
        [local.packages]    - Regular packages for host environment
        [envname]           - Isolated env definition
        [envname.cuda]      - CUDA packages for isolated env
        [envname.packages]  - Regular packages for isolated env
        [node_reqs]         - Node dependencies
    """
    local: LocalConfig = field(default_factory=LocalConfig)
    envs: Dict[str, "IsolatedEnv"] = field(default_factory=dict)
    node_reqs: List[NodeReq] = field(default_factory=list)

    @property
    def has_local(self) -> bool:
        """Check if there are local packages to install."""
        return bool(self.local.cuda_packages or self.local.requirements)

    @property
    def has_envs(self) -> bool:
        """Check if there are isolated environments defined."""
        return bool(self.envs)

    def get_env(self, name: str) -> Optional["IsolatedEnv"]:
        """Get an isolated environment by name."""
        return self.envs.get(name)

    @property
    def default_env(self) -> Optional["IsolatedEnv"]:
        """Get the first/only isolated environment, or None."""
        if self.envs:
            return next(iter(self.envs.values()))
        return None


@dataclass
class IsolatedEnv:
    """
    Configuration for an isolated Python environment.

    This defines what Python version, CUDA version, and dependencies
    should be installed in the isolated environment.

    Args:
        name: Unique name for this environment (used for caching)
        python: Python version (e.g., "3.10", "3.11")
        cuda: CUDA version (e.g., "12.4", "12.8") or None for CPU-only
        requirements: List of pip requirements (e.g., ["torch==2.8.0", "numpy"])
        requirements_file: Path to requirements.txt file
        wheel_sources: List of URLs for --find-links (custom wheel repos)
        index_urls: List of URLs for --extra-index-url
        env_dir: Custom directory for the venv (default: auto-generated)
        pytorch_version: Specific PyTorch version (auto-detected if None)
        worker_package: Worker package directory (e.g., "worker" -> worker/__main__.py)
        worker_script: Worker script file (e.g., "worker.py")

    Example:
        env = IsolatedEnv(
            name="my-node",
            python="3.10",
            cuda="12.8",
            requirements=["torch==2.8.0", "nvdiffrast"],
            wheel_sources=["https://my-wheels.github.io/"],
        )
    """

    name: str
    python: str = "3.10"
    cuda: Optional[str] = None
    requirements: list[str] = field(default_factory=list)
    no_deps_requirements: list[str] = field(default_factory=list)  # Install with --no-deps
    requirements_file: Optional[Path] = None
    wheel_sources: list[str] = field(default_factory=list)
    index_urls: list[str] = field(default_factory=list)
    env_dir: Optional[Path] = None
    pytorch_version: Optional[str] = None
    # Worker configuration
    worker_package: Optional[str] = None  # e.g., "worker" -> worker/__main__.py
    worker_script: Optional[str] = None   # e.g., "worker.py" -> worker.py

    def __post_init__(self):
        """Validate and normalize configuration."""
        # Normalize paths
        if self.requirements_file is not None:
            self.requirements_file = Path(self.requirements_file)
        if self.env_dir is not None:
            self.env_dir = Path(self.env_dir)

        # Validate Python version
        if not self.python.replace(".", "").isdigit():
            raise ValueError(f"Invalid Python version: {self.python}")

        # Validate CUDA version if specified
        if self.cuda is not None:
            cuda_clean = self.cuda.replace(".", "")
            if not cuda_clean.isdigit():
                raise ValueError(f"Invalid CUDA version: {self.cuda}")

    @property
    def cuda_short(self) -> Optional[str]:
        """Get CUDA version without dots (e.g., '128' for '12.8')."""
        if self.cuda is None:
            return None
        return self.cuda.replace(".", "")

    @property
    def python_short(self) -> str:
        """Get Python version without dots (e.g., '310' for '3.10')."""
        return self.python.replace(".", "")

    def get_default_env_dir(self, base_dir: Path) -> Path:
        """Get the default environment directory path."""
        if self.env_dir is not None:
            return self.env_dir
        return base_dir / f"_env_{self.name}"
