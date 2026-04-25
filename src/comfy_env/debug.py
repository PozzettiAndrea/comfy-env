"""Granular debug logging configuration for comfy-env.

Categories can be enabled via:
  1. Environment variables (highest priority)
  2. Persistent settings in ~/.comfy-env/debug.env
  3. Master switch COMFY_ENV_DEBUG=1 enables all categories

Workers can't import this module (different venv), so they parse env vars directly.
The env vars propagate to subprocesses automatically via os.environ.
"""

import os
from pathlib import Path

SETTINGS_FILE = Path.home() / ".comfy-env" / "debug.env"

# Load persistent settings (simple KEY=1 file) -- env vars always override
if SETTINGS_FILE.exists():
    try:
        for line in SETTINGS_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    except Exception:
        pass


def _is_on(var: str) -> bool:
    return os.environ.get(var, "").lower() in ("1", "true", "yes")


# Master switch
DEBUG = _is_on("COMFY_ENV_DEBUG")

# Granular categories -- each is ON if its own var is set OR master is on
SERIALIZE = DEBUG or _is_on("COMFY_ENV_DEBUG_SERIALIZE")
IPC = DEBUG or _is_on("COMFY_ENV_DEBUG_IPC")
WORKER = DEBUG or _is_on("COMFY_ENV_DEBUG_WORKER")
MODELS = DEBUG or _is_on("COMFY_ENV_DEBUG_MODELS")
META = DEBUG or _is_on("COMFY_ENV_DEBUG_META")
INSTALL = DEBUG or _is_on("COMFY_ENV_DEBUG_INSTALL")
STACKTRACE = DEBUG or _is_on("COMFY_ENV_DEBUG_STACKTRACE")
INPUTS_OUTPUTS = DEBUG or _is_on("COMFY_ENV_DEBUG_INPUTS_OUTPUTS")
VRAM = DEBUG or _is_on("COMFY_ENV_DEBUG_VRAM")
WATCHDOG = DEBUG or _is_on("COMFY_ENV_DEBUG_WATCHDOG")

# Ordered list for TUI display
CATEGORIES = [
    ("COMFY_ENV_DEBUG", "All (master switch)"),
    ("COMFY_ENV_DEBUG_INPUTS_OUTPUTS", "Node inputs/outputs (shapes, types, devices)"),
    ("COMFY_ENV_DEBUG_VRAM", "GPU VRAM state before/after node execution"),
    ("COMFY_ENV_DEBUG_SERIALIZE", "Tensor serialization / deserialization"),
    ("COMFY_ENV_DEBUG_IPC", "CUDA IPC (legacy + pool)"),
    ("COMFY_ENV_DEBUG_WORKER", "Worker lifecycle (start/stop/crash)"),
    ("COMFY_ENV_DEBUG_WATCHDOG", "Worker watchdog (thread dumps every 60s)"),
    ("COMFY_ENV_DEBUG_MODELS", "Model registration & VRAM"),
    ("COMFY_ENV_DEBUG_META", "Node metadata scanning"),
    ("COMFY_ENV_DEBUG_INSTALL", "Environment install & build"),
    ("COMFY_ENV_DEBUG_STACKTRACE", "Full stack traces from workers"),
]
