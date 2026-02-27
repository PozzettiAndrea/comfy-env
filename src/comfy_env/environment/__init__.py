"""
Environment layer - Environment management with side effects.

Handles environment caching, path resolution, and runtime setup.
"""

from .cache import (
    CACHE_DIR,
    get_cache_dir,
    get_env_name,
    get_local_env_path,
    resolve_env_path,
)
from .paths import (
    get_site_packages_path,
    get_lib_path,
    copy_files,
)
from .setup import (
    setup_env,
)
from .libomp import (
    dedupe_libomp,
)

__all__ = [
    # Cache management
    "CACHE_DIR",
    "get_cache_dir",
    "get_env_name",
    "get_local_env_path",
    "resolve_env_path",
    # Path resolution
    "get_site_packages_path",
    "get_lib_path",
    "copy_files",
    # Setup helpers
    "setup_env",
    # macOS workaround
    "dedupe_libomp",
]
