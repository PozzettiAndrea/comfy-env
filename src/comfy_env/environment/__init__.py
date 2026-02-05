"""
Environment layer - Environment management with side effects.

Handles environment caching, path resolution, and runtime setup.
"""

from .cache import (
    CACHE_DIR,
    MARKER_FILE,
    JUNCTION_FILE,
    get_cache_dir,
    get_env_name,
    get_env_path,
    cleanup_orphaned_envs,
    write_marker_file,
    read_marker_file,
    write_junction_path,
    read_junction_path,
    create_junction,
    get_junction_name,
)
from .paths import (
    resolve_env_path,
    get_site_packages_path,
    get_lib_path,
    copy_files,
)
from .setup import (
    setup_env,
    load_env_vars,
    inject_site_packages,
)
from .libomp import (
    dedupe_libomp,
)

__all__ = [
    # Cache management
    "CACHE_DIR",
    "MARKER_FILE",
    "JUNCTION_FILE",
    "get_cache_dir",
    "get_env_name",
    "get_env_path",
    "cleanup_orphaned_envs",
    "write_marker_file",
    "read_marker_file",
    "write_junction_path",
    "read_junction_path",
    "create_junction",
    "get_junction_name",
    # Path resolution
    "resolve_env_path",
    "get_site_packages_path",
    "get_lib_path",
    "copy_files",
    # Setup helpers
    "setup_env",
    "load_env_vars",
    "inject_site_packages",
    # macOS workaround
    "dedupe_libomp",
]
