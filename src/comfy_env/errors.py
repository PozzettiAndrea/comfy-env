"""
Rich error messages for comfy-env.

This module provides error classes with actionable, user-friendly messages.
Instead of cryptic pip errors, users see exactly what went wrong and what
they can do to fix it.

Example output:

    +------------------------------------------------------------------+
    |  CUDA Wheel Not Found                                            |
    +------------------------------------------------------------------+
    |  Package:   nvdiffrast==0.4.0                                    |
    |  Requested: cu130-torch291-cp312-linux_x86_64                    |
    |                                                                  |
    |  Suggestions:                                                    |
    |    1. Use Python 3.10 instead of 3.12                            |
    |    2. Use CUDA 12.8 (set cuda = "12.8" in config)                |
    |    3. Build wheel locally: comfy-env build nvdiffrast            |
    +------------------------------------------------------------------+
"""

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .resolver import RuntimeEnv


class EnvManagerError(Exception):
    """Base exception for comfy-env errors."""

    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(self.format())

    def format(self) -> str:
        """Format error message for display."""
        if self.details:
            return f"{self.message}\n\n{self.details}"
        return self.message


class ConfigError(EnvManagerError):
    """Error in configuration file."""

    def __init__(self, message: str, file_path: Optional[str] = None, line: Optional[int] = None):
        self.file_path = file_path
        self.line = line

        details = None
        if file_path:
            location = f"in {file_path}"
            if line:
                location += f" at line {line}"
            details = location

        super().__init__(message, details)


@dataclass
class WheelNotFoundError(EnvManagerError):
    """
    Raised when a CUDA wheel cannot be found or resolved.

    Provides actionable suggestions based on the environment and what
    wheels are typically available.
    """
    package: str
    version: Optional[str] = None
    env: Optional["RuntimeEnv"] = None
    tried_urls: List[str] = field(default_factory=list)
    reason: Optional[str] = None
    available_combos: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Build the formatted message
        self.message = self._build_message()
        self.details = self._build_details()
        Exception.__init__(self, self.format())

    def _build_message(self) -> str:
        """Build the main error message."""
        pkg = self.package
        if self.version:
            pkg = f"{self.package}=={self.version}"
        return f"CUDA wheel not found: {pkg}"

    def _build_details(self) -> str:
        """Build detailed error with suggestions."""
        lines = []

        # Box header
        lines.append("+" + "-" * 66 + "+")
        lines.append("|  CUDA Wheel Not Found" + " " * 44 + "|")
        lines.append("+" + "-" * 66 + "+")

        # Package info
        pkg_line = f"  Package:   {self.package}"
        if self.version:
            pkg_line += f"=={self.version}"
        lines.append(f"|{pkg_line:<66}|")

        # Requested configuration
        if self.env:
            requested = self._format_requested()
            lines.append(f"|  Requested: {requested:<54}|")

        lines.append("|" + " " * 66 + "|")

        # Reason if provided
        if self.reason:
            lines.append(f"|  Reason: {self.reason:<57}|")
            lines.append("|" + " " * 66 + "|")

        # Tried URLs
        if self.tried_urls:
            lines.append("|  Tried URLs:" + " " * 53 + "|")
            for url in self.tried_urls[:3]:  # Limit to first 3
                # Truncate long URLs
                if len(url) > 60:
                    url = "..." + url[-57:]
                lines.append(f"|    {url:<62}|")
            lines.append("|" + " " * 66 + "|")

        # Suggestions
        suggestions = self._generate_suggestions()
        if suggestions:
            lines.append("|  Suggestions:" + " " * 52 + "|")
            for i, suggestion in enumerate(suggestions, 1):
                lines.append(f"|    {i}. {suggestion:<60}|")
            lines.append("|" + " " * 66 + "|")

        # Footer
        lines.append("+" + "-" * 66 + "+")

        return "\n".join(lines)

    def _format_requested(self) -> str:
        """Format the requested configuration."""
        if not self.env:
            return "unknown"

        parts = []
        if self.env.cuda_short:
            parts.append(f"cu{self.env.cuda_short}")
        else:
            parts.append("cpu")

        if self.env.torch_mm:
            parts.append(f"torch{self.env.torch_mm}")

        parts.append(f"cp{self.env.python_short}")
        parts.append(self.env.platform_tag)

        return "-".join(parts)

    def _generate_suggestions(self) -> List[str]:
        """Generate actionable suggestions based on the error context."""
        suggestions = []

        if not self.env:
            suggestions.append("Run 'comfy-env info' to see your environment")
            return suggestions

        # Python version suggestion
        if self.env.python_short not in ("310", "311"):
            suggestions.append(
                f"Use Python 3.10 or 3.11 (you have {self.env.python_version})"
            )

        # CUDA version suggestion
        if self.env.cuda_version and self.env.cuda_version not in ("12.4", "12.8"):
            suggestions.append(
                f"Use CUDA 12.4 or 12.8 (you have {self.env.cuda_version})"
            )

        # PyTorch version suggestion
        if self.env.torch_version:
            torch_major_minor = ".".join(self.env.torch_version.split(".")[:2])
            if torch_major_minor not in ("2.5", "2.8"):
                suggestions.append(
                    f"Use PyTorch 2.5 or 2.8 (you have {self.env.torch_version})"
                )

        # General suggestions
        suggestions.append(
            f"Check if wheel exists: comfy-env resolve {self.package}"
        )
        suggestions.append(
            f"Build wheel locally: comfy-env build {self.package}"
        )

        return suggestions[:4]  # Limit to 4 suggestions

    def format(self) -> str:
        """Format the complete error message."""
        return f"{self.message}\n\n{self.details}"


class DependencyError(EnvManagerError):
    """Error resolving or installing dependencies."""

    def __init__(
        self,
        message: str,
        package: Optional[str] = None,
        pip_error: Optional[str] = None,
    ):
        self.package = package
        self.pip_error = pip_error

        details = None
        if pip_error:
            # Extract relevant lines from pip error
            relevant_lines = self._extract_pip_error(pip_error)
            if relevant_lines:
                details = "pip error:\n" + "\n".join(relevant_lines)

        super().__init__(message, details)

    def _extract_pip_error(self, pip_error: str) -> List[str]:
        """Extract the most relevant lines from pip error output."""
        lines = pip_error.strip().split("\n")
        relevant = []

        for line in lines:
            # Skip empty lines and progress bars
            if not line.strip():
                continue
            if line.startswith("  ") and "%" in line:
                continue

            # Keep error lines and important info
            lower = line.lower()
            if any(keyword in lower for keyword in [
                "error", "failed", "could not", "no matching",
                "requirement", "conflict", "incompatible"
            ]):
                relevant.append(line)

        return relevant[:10]  # Limit to 10 lines


class CUDANotFoundError(EnvManagerError):
    """Raised when CUDA is required but not available."""

    def __init__(self, package: Optional[str] = None):
        message = "CUDA is required but not detected"
        details_lines = [
            "This package requires a CUDA-capable GPU.",
            "",
            "To fix this:",
            "  1. Ensure you have an NVIDIA GPU",
            "  2. Install NVIDIA drivers",
            "  3. Install CUDA Toolkit",
            "",
            "Or if you want to run on CPU:",
            "  Set 'fallback_to_cpu = true' in your config",
        ]

        if package:
            message = f"CUDA is required for {package} but not detected"

        super().__init__(message, "\n".join(details_lines))


class InstallError(EnvManagerError):
    """Error during package installation."""

    def __init__(
        self,
        message: str,
        package: Optional[str] = None,
        exit_code: Optional[int] = None,
        stderr: Optional[str] = None,
    ):
        self.package = package
        self.exit_code = exit_code
        self.stderr = stderr

        details_parts = []
        if exit_code is not None:
            details_parts.append(f"Exit code: {exit_code}")
        if stderr:
            # Truncate long stderr
            if len(stderr) > 500:
                stderr = stderr[:500] + "\n... (truncated)"
            details_parts.append(f"Output:\n{stderr}")

        details = "\n".join(details_parts) if details_parts else None
        super().__init__(message, details)
