"""
Wheel URL resolver for CUDA-compiled packages.

This module provides deterministic wheel URL construction based on the runtime
environment (CUDA version, PyTorch version, Python version, platform).

Unlike pip's constraint solver, this module constructs exact URLs from templates
and validates that they exist. If a wheel doesn't exist, it fails fast with
a clear error message.

Example:
    from comfy_env.resolver import WheelResolver, RuntimeEnv

    env = RuntimeEnv.detect()
    resolver = WheelResolver()

    url = resolver.resolve("nvdiffrast", version="0.4.0", env=env)
    # Returns: https://github.com/.../nvdiffrast-0.4.0+cu128torch28-cp310-...-linux_x86_64.whl
"""

import platform
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from .env.detection import detect_cuda_version, detect_gpu_info


@dataclass
class RuntimeEnv:
    """
    Detected runtime environment for wheel resolution.

    Contains all variables needed for wheel URL template expansion.
    """
    # OS/Platform
    os_name: str  # linux, windows, darwin
    platform_tag: str  # linux_x86_64, win_amd64, macosx_...

    # Python
    python_version: str  # 3.10, 3.11, 3.12
    python_short: str  # 310, 311, 312

    # CUDA
    cuda_version: Optional[str]  # 12.8, 12.4, None
    cuda_short: Optional[str]  # 128, 124, None

    # PyTorch (detected or configured)
    torch_version: Optional[str]  # 2.8.0, 2.5.1
    torch_short: Optional[str]  # 280, 251
    torch_mm: Optional[str]  # 28, 25 (major.minor without dot)

    # GPU info
    gpu_name: Optional[str] = None
    gpu_compute: Optional[str] = None  # sm_89, sm_100

    @classmethod
    def detect(cls, torch_version: Optional[str] = None) -> "RuntimeEnv":
        """
        Detect runtime environment from current system.

        Args:
            torch_version: Optional PyTorch version override. If not provided,
                          attempts to detect from installed torch.

        Returns:
            RuntimeEnv with detected values.
        """
        # OS detection
        os_name = sys.platform
        if os_name.startswith('linux'):
            os_name = 'linux'
        elif os_name == 'win32':
            os_name = 'windows'
        elif os_name == 'darwin':
            os_name = 'darwin'

        # Platform tag
        platform_tag = _get_platform_tag()

        # Python version
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        py_short = f"{sys.version_info.major}{sys.version_info.minor}"

        # CUDA version
        cuda_version = detect_cuda_version()
        cuda_short = cuda_version.replace(".", "") if cuda_version else None

        # PyTorch version
        if torch_version is None:
            torch_version = _detect_torch_version()

        torch_short = None
        torch_mm = None
        if torch_version:
            torch_short = torch_version.replace(".", "")
            parts = torch_version.split(".")[:2]
            torch_mm = "".join(parts)

        # GPU info
        gpu_name = None
        gpu_compute = None
        try:
            gpu_info = detect_gpu_info()
            if gpu_info:
                gpu_name = gpu_info.get("name")
                gpu_compute = gpu_info.get("compute_capability")
        except Exception:
            pass

        return cls(
            os_name=os_name,
            platform_tag=platform_tag,
            python_version=py_version,
            python_short=py_short,
            cuda_version=cuda_version,
            cuda_short=cuda_short,
            torch_version=torch_version,
            torch_short=torch_short,
            torch_mm=torch_mm,
            gpu_name=gpu_name,
            gpu_compute=gpu_compute,
        )

    def as_dict(self) -> Dict[str, str]:
        """Convert to dict for template substitution."""
        # Extract py_minor from python_version (e.g., "3.10" -> "10")
        py_minor = self.python_version.split(".")[-1] if self.python_version else ""

        result = {
            "os": self.os_name,
            "platform": self.platform_tag,
            "python_version": self.python_version,
            "py_version": self.python_version,
            "py_short": self.python_short,
            "py_minor": py_minor,
        }

        if self.cuda_version:
            result["cuda_version"] = self.cuda_version
            result["cuda_short"] = self.cuda_short

        if self.torch_version:
            result["torch_version"] = self.torch_version
            result["torch_short"] = self.torch_short
            result["torch_mm"] = self.torch_mm
            # torch_dotted_mm: "2.8" format (major.minor with dot) for flash-attn URLs
            parts = self.torch_version.split(".")[:2]
            result["torch_dotted_mm"] = ".".join(parts)

        return result

    def __str__(self) -> str:
        parts = [
            f"Python {self.python_version}",
            f"CUDA {self.cuda_version}" if self.cuda_version else "CPU",
        ]
        if self.torch_version:
            parts.append(f"PyTorch {self.torch_version}")
        if self.gpu_name:
            parts.append(f"GPU: {self.gpu_name}")
        return ", ".join(parts)


def _get_platform_tag() -> str:
    """Get wheel platform tag for current system."""
    machine = platform.machine().lower()

    if sys.platform.startswith('linux'):
        # Use manylinux tag
        if machine in ('x86_64', 'amd64'):
            return 'linux_x86_64'
        elif machine == 'aarch64':
            return 'linux_aarch64'
        return f'linux_{machine}'

    elif sys.platform == 'win32':
        if machine in ('amd64', 'x86_64'):
            return 'win_amd64'
        return 'win32'

    elif sys.platform == 'darwin':
        # macOS - use generic tag
        if machine == 'arm64':
            return 'macosx_11_0_arm64'
        return 'macosx_10_9_x86_64'

    return f'{sys.platform}_{machine}'


def _detect_torch_version() -> Optional[str]:
    """Detect installed PyTorch version."""
    try:
        import torch
        version = torch.__version__
        # Strip CUDA suffix (e.g., "2.8.0+cu128" -> "2.8.0")
        if '+' in version:
            version = version.split('+')[0]
        return version
    except ImportError:
        return None


@dataclass
class WheelSource:
    """Configuration for a wheel source (GitHub releases, custom index, etc.)."""
    name: str
    url_template: str
    packages: List[str] = field(default_factory=list)  # Empty = all packages

    def supports(self, package: str) -> bool:
        """Check if this source provides the given package."""
        if not self.packages:
            return True  # Empty list = supports all
        return package.lower() in [p.lower() for p in self.packages]


# Default wheel sources for common CUDA packages
DEFAULT_WHEEL_SOURCES = [
    WheelSource(
        name="nvdiffrast-wheels",
        url_template="https://github.com/PozzettiAndrea/nvdiffrast-full-wheels/releases/download/v{version}/nvdiffrast-{version}%2Bcu{cuda_short}torch{torch_mm}-cp{py_short}-cp{py_short}-{platform}.whl",
        packages=["nvdiffrast"],
    ),
    WheelSource(
        name="cumesh-wheels",
        url_template="https://github.com/PozzettiAndrea/cumesh-wheels/releases/download/v{version}/{package}-{version}%2Bcu{cuda_short}torch{torch_mm}-cp{py_short}-cp{py_short}-{platform}.whl",
        packages=["pytorch3d", "torch-cluster", "torch-scatter", "torch-sparse"],
    ),
]


class WheelResolver:
    """
    Resolves CUDA wheel URLs from package name and runtime environment.

    Resolution strategy:
    1. Check explicit overrides in config
    2. Try configured wheel sources in order
    3. Fail with actionable error message

    This is NOT a constraint solver. It constructs deterministic URLs
    based on exact version matches.
    """

    def __init__(
        self,
        sources: Optional[List[WheelSource]] = None,
        overrides: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize resolver.

        Args:
            sources: List of WheelSource configurations.
            overrides: Package-specific URL overrides (package -> template).
        """
        self.sources = sources or DEFAULT_WHEEL_SOURCES
        self.overrides = overrides or {}

    def resolve(
        self,
        package: str,
        version: str,
        env: RuntimeEnv,
        verify: bool = False,
    ) -> str:
        """
        Resolve wheel URL for a package.

        Args:
            package: Package name (e.g., "nvdiffrast").
            version: Package version (e.g., "0.4.0").
            env: Runtime environment for template expansion.

        Returns:
            Fully resolved wheel URL.

        Raises:
            WheelNotFoundError: If no wheel URL could be constructed or verified.
        """
        from .errors import WheelNotFoundError

        # Prepare template variables
        variables = env.as_dict()
        variables["package"] = package
        variables["version"] = version

        # 1. Check explicit override
        if package.lower() in self.overrides:
            url = self._substitute(self.overrides[package.lower()], variables)
            if verify and not self._url_exists(url):
                raise WheelNotFoundError(
                    package=package,
                    version=version,
                    env=env,
                    tried_urls=[url],
                    reason="Override URL returned 404",
                )
            return url

        # 2. Try wheel sources
        tried_urls = []
        for source in self.sources:
            if not source.supports(package):
                continue

            url = self._substitute(source.url_template, variables)
            tried_urls.append(url)

            if verify:
                if self._url_exists(url):
                    return url
            else:
                return url

        # 3. Fail with helpful error
        raise WheelNotFoundError(
            package=package,
            version=version,
            env=env,
            tried_urls=tried_urls,
            reason="No wheel source found for package",
        )

    def resolve_all(
        self,
        packages: Dict[str, str],
        env: RuntimeEnv,
        verify: bool = False,
    ) -> Dict[str, str]:
        """
        Resolve URLs for multiple packages.

        Args:
            packages: Dict of package -> version.
            env: Runtime environment.
            verify: Whether to verify URLs exist.

        Returns:
            Dict of package -> resolved URL.

        Raises:
            WheelNotFoundError: If any package cannot be resolved.
        """
        results = {}
        for package, version in packages.items():
            results[package] = self.resolve(package, version, env, verify=verify)
        return results

    def _substitute(self, template: str, variables: Dict[str, str]) -> str:
        """
        Substitute variables into URL template.

        Handles both {var} and {var_name} style placeholders.
        Missing variables are left as-is (caller should validate).
        """
        result = template
        for key, value in variables.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result

    def _url_exists(self, url: str, timeout: float = 10.0) -> bool:
        """
        Check if a URL exists using HTTP HEAD request.

        Returns True if URL returns 200 OK.
        """
        try:
            import urllib.request
            request = urllib.request.Request(url, method='HEAD')
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status == 200
        except Exception:
            return False


def parse_wheel_requirement(req: str) -> Tuple[str, Optional[str]]:
    """
    Parse a wheel requirement string.

    Examples:
        "nvdiffrast==0.4.0" -> ("nvdiffrast", "0.4.0")
        "pytorch3d>=0.7.8" -> ("pytorch3d", "0.7.8")
        "torch-cluster" -> ("torch-cluster", None)

    Returns:
        Tuple of (package_name, version_or_None).
    """
    # Handle version specifiers
    for op in ['==', '>=', '<=', '~=', '!=', '>', '<']:
        if op in req:
            parts = req.split(op, 1)
            return (parts[0].strip(), parts[1].strip())

    return (req.strip(), None)
