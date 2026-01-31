"""
Config layer - Configuration parsing and types.

Pure parsing, no side effects.
"""

from .types import (
    ComfyEnvConfig,
    NodeDependency,
    NodeReq,  # Backwards compatibility alias
)
from .parser import (
    CONFIG_FILE_NAME,
    load_config,
    discover_config,
    parse_config,
)

__all__ = [
    # Types
    "ComfyEnvConfig",
    "NodeDependency",
    "NodeReq",  # Backwards compatibility
    # Parsing
    "CONFIG_FILE_NAME",
    "load_config",
    "discover_config",
    "parse_config",
]
