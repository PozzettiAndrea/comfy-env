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

# Load persistent settings (simple KEY=VALUE file) — env vars always override
if SETTINGS_FILE.exists():
    try:
        for line in SETTINGS_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    except Exception:
        pass

# General settings: (env_var, label)
GENERAL_SETTINGS = [
    ("COMFY_ENV_INSTALL_ISOLATED", "Install to isolated envs (pixi/venv)"),
    ("COMFY_ENV_INSTALL_MAIN", "Install to main Python env (pip)"),
    ("COMFY_ENV_ISOLATE", "Run nodes in subprocess workers"),
    ("COMFY_ENV_POOL_IPC", "Pool IPC (zero-copy GPU tensor transfer)"),
]

# Defaults (True = on when env var is unset)
GENERAL_DEFAULTS = {
    "COMFY_ENV_INSTALL_ISOLATED": True,
    "COMFY_ENV_INSTALL_MAIN": False,
    "COMFY_ENV_ISOLATE": True,
    "COMFY_ENV_POOL_IPC": False,
}


def _is_on(var: str, default: bool = False) -> bool:
    val = os.environ.get(var, "")
    if val == "":
        return default
    return val.lower() in ("1", "true", "yes")


INSTALL_ISOLATED = _is_on("COMFY_ENV_INSTALL_ISOLATED", GENERAL_DEFAULTS["COMFY_ENV_INSTALL_ISOLATED"])
INSTALL_MAIN = _is_on("COMFY_ENV_INSTALL_MAIN", GENERAL_DEFAULTS["COMFY_ENV_INSTALL_MAIN"])
ISOLATE = _is_on("COMFY_ENV_ISOLATE", GENERAL_DEFAULTS["COMFY_ENV_ISOLATE"])

# Numeric settings: (env_var, label, default_value, unit)
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

# Risky patches (monkey-patching ComfyUI internals)
PATCH_SETTINGS = [
    ("COMFY_ENV_PATCH_SHAREABLE_POOL", "CUDA shareable pool (full zero-copy IPC)"),
    ("COMFY_ENV_PATCH_FLASH_ATTENTION", "Auto-activate flash attention"),
    ("COMFY_ENV_PATCH_SAGE_ATTENTION", "Auto-activate sage attention"),
]
PATCH_DEFAULTS = {
    "COMFY_ENV_PATCH_SHAREABLE_POOL": False,
    "COMFY_ENV_PATCH_FLASH_ATTENTION": False,
    "COMFY_ENV_PATCH_SAGE_ATTENTION": False,
}
