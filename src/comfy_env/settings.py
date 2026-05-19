"""General settings for comfy-env.

Settings can be configured via:
  1. Environment variables (highest priority)
  2. Persistent settings in ~/.comfy-env/settings.env
  3. Defaults

Workers can't import this module (different venv), so they parse env vars directly.
"""

import os
from pathlib import Path

SETTINGS_FILE = Path.home() / ".comfy-env" / "settings.env"

# Load persistent settings (simple KEY=VALUE file) -- env vars always override
if SETTINGS_FILE.exists():
    try:
        for line in SETTINGS_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    except Exception:
        pass


def _is_on(var: str, default: bool = False) -> bool:
    val = os.environ.get(var, "")
    if val == "":
        return default
    return val.lower() in ("1", "true", "yes")


# General settings
GENERAL_SETTINGS = [
    ("COMFY_ENV_INSTALL_ISOLATED", "Install to isolated envs (pixi)"),
    ("COMFY_ENV_ISOLATE", "Run nodes in subprocess workers"),
    ("COMFY_ENV_AUTO_INSTALL", "Auto-materialize missing envs on first node load (blocks startup)"),
    ("COMFY_ENV_POOL_IPC", "Pool IPC (zero-copy GPU tensor transfer)"),
]

GENERAL_DEFAULTS = {
    "COMFY_ENV_INSTALL_ISOLATED": True,
    "COMFY_ENV_ISOLATE": True,
    "COMFY_ENV_AUTO_INSTALL": False,   # OFF by default -- explicit opt-in (multi-min installs)
    "COMFY_ENV_POOL_IPC": False,
}

INSTALL_ISOLATED = _is_on("COMFY_ENV_INSTALL_ISOLATED", GENERAL_DEFAULTS["COMFY_ENV_INSTALL_ISOLATED"])
ISOLATE = _is_on("COMFY_ENV_ISOLATE", GENERAL_DEFAULTS["COMFY_ENV_ISOLATE"])

# Numeric settings
NUMERIC_SETTINGS = [
    ("COMFY_ENV_WORKER_VRAM_BUDGET", "Worker VRAM budget (GB, 0=auto)", 0, "GB"),
]


def get_numeric(var: str, default: float = 0) -> float:
    val = os.environ.get(var, "")
    if val == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


WORKER_VRAM_BUDGET = get_numeric("COMFY_ENV_WORKER_VRAM_BUDGET", 0)

# Patches (monkey-patching ComfyUI internals)
PATCH_SETTINGS = [
    ("COMFY_ENV_PATCH_SHAREABLE_POOL", "CUDA shareable pool (full zero-copy IPC)"),
]
PATCH_DEFAULTS = {
    "COMFY_ENV_PATCH_SHAREABLE_POOL": False,
}

# Mapping from short TOML key names to env var names (for [settings] in comfy-env-root.toml)
SETTINGS_KEY_MAP = {
    "isolate": "COMFY_ENV_ISOLATE",
    "install_isolated": "COMFY_ENV_INSTALL_ISOLATED",
    "auto_install": "COMFY_ENV_AUTO_INSTALL",
    "pool_ipc": "COMFY_ENV_POOL_IPC",
    "worker_vram_budget": "COMFY_ENV_WORKER_VRAM_BUDGET",
}
_ENV_TO_SHORT = {v: k for k, v in SETTINGS_KEY_MAP.items()}


def resolve_bool(var: str, node_settings: dict = None, default: bool = False) -> bool:
    """Resolve a boolean setting with per-node override support."""
    if node_settings:
        short_key = _ENV_TO_SHORT.get(var)
        if short_key and short_key in node_settings:
            return bool(node_settings[short_key])
    return _is_on(var, default)


def resolve_numeric(var: str, node_settings: dict = None, default: float = 0) -> float:
    """Resolve a numeric setting with per-node override support."""
    if node_settings:
        short_key = _ENV_TO_SHORT.get(var)
        if short_key and short_key in node_settings:
            try:
                return float(node_settings[short_key])
            except (ValueError, TypeError):
                pass
    return get_numeric(var, default)
