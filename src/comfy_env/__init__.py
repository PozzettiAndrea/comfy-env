"""
comfy-env - Environment management for ComfyUI custom nodes.

Features:
- CUDA wheel resolution (pre-built wheels without compilation)
- Process isolation (run nodes in separate Python environments)
- Local _env_* folders (no central cache, no junctions)
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("comfy-env")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"


# =============================================================================
# Primary API (what most users need)
# =============================================================================

# Install API
from .install import install, verify_installation, USE_COMFY_ENV_VAR

# Prestartup helpers
from .environment.setup import setup_env
from .environment.paths import copy_files

# Isolation
from .isolation import wrap_isolated_nodes, wrap_nodes, register_nodes


# =============================================================================
# Config Layer
# =============================================================================

from .config import (
    ComfyEnvConfig,
    NodeDependency,
    NodeReq,
    load_config,
    discover_config,
    CONFIG_FILE_NAME,
    ROOT_CONFIG_FILE_NAME,
)


# =============================================================================
# Detection Layer
# =============================================================================

from .detection import (
    # CUDA detection
    detect_cuda_version,
    detect_cuda_environment,
    get_recommended_cuda_version,
    # GPU detection
    GPUInfo,
    CUDAEnvironment,
    detect_gpu,
    get_gpu_summary,
    # Platform detection
    detect_platform,
    get_platform_tag,
    # Runtime detection
    RuntimeEnv,
    detect_runtime,
)


# =============================================================================
# Packages Layer
# =============================================================================

from .packages import (
    # Pixi
    ensure_pixi,
    get_pixi_path,
    get_pixi_python,
    pixi_run,
    pixi_clean,
    # CUDA wheels
    CUDA_WHEELS_INDEX,
    get_wheel_url,
    get_cuda_torch_mapping,
)


# =============================================================================
# Environment Layer
# =============================================================================

from .environment import (
    get_cache_dir,
    resolve_env_path,
    CACHE_DIR,
)


# =============================================================================
# Isolation Layer
# =============================================================================

from .isolation import (
    # Workers
    Worker,
    WorkerError,
    SubprocessWorker,
    # Tensor utilities
    TensorKeeper,
    release_tensor,
    release_tensors_recursive,
)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Install API
    "install",
    "verify_installation",
    "USE_COMFY_ENV_VAR",
    # Prestartup
    "setup_env",
    "copy_files",
    # Isolation
    "wrap_isolated_nodes",
    "wrap_nodes",
    "register_nodes",
    # Config
    "ComfyEnvConfig",
    "NodeDependency",
    "NodeReq",
    "load_config",
    "discover_config",
    "CONFIG_FILE_NAME",
    "ROOT_CONFIG_FILE_NAME",
    # Detection
    "detect_cuda_version",
    "detect_cuda_environment",
    "get_recommended_cuda_version",
    "GPUInfo",
    "CUDAEnvironment",
    "detect_gpu",
    "get_gpu_summary",
    "detect_platform",
    "get_platform_tag",
    "RuntimeEnv",
    "detect_runtime",
    # Packages
    "ensure_pixi",
    "get_pixi_path",
    "get_pixi_python",
    "pixi_run",
    "pixi_clean",
    "CUDA_WHEELS_INDEX",
    "get_wheel_url",
    "get_cuda_torch_mapping",
    # Environment
    "get_cache_dir",
    "resolve_env_path",
    "CACHE_DIR",
    # Workers
    "Worker",
    "WorkerError",
    "SubprocessWorker",
    "TensorKeeper",
    "release_tensor",
    "release_tensors_recursive",
]


# =============================================================================
# CUDA package mocking for testing (comfy-test integration)
# =============================================================================

def _mock_cuda_packages():
    """Mock CUDA packages when running under comfy-test.

    When COMFY_TEST_MOCK_PACKAGES is set, creates empty mock modules
    so imports don't fail on CPU-only test machines.
    """
    import os
    import sys
    import types
    import importlib.machinery

    mock_packages = os.environ.get("COMFY_TEST_MOCK_PACKAGES", "")
    if not mock_packages:
        return

    import importlib.util

    for pkg in mock_packages.split(","):
        pkg = pkg.strip()
        if not pkg or pkg in sys.modules:
            continue

        # Skip packages that are already installable (e.g. torch from ComfyUI's requirements)
        if importlib.util.find_spec(pkg) is not None:
            continue

        mock_module = types.ModuleType(pkg)
        mock_module.__spec__ = importlib.machinery.ModuleSpec(pkg, None)
        mock_module.__path__ = []  # Mark as package
        mock_module.__file__ = "<mocked by comfy-env for testing>"
        sys.modules[pkg] = mock_module

_mock_cuda_packages()
