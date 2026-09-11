"""Metadata extraction for isolation nodes via subprocess scan.

Spawns a short-lived subprocess in the isolation env's Python to import node modules
and extract class metadata (INPUT_TYPES, RETURN_TYPES, etc.). The main process never
imports isolation code -- it builds proxy classes from the serialized metadata.
"""

import hashlib
import os
import json
import subprocess
import sys
import tempfile
import time
import uuid

from .. import state_sync
from pathlib import Path

from ..environment.cache import env_label
from typing import Any, Dict, List, Optional

from ..config import DEFAULT_HEALTH_CHECK_TIMEOUT
from ..debug import (META as _DBG_META, INPUTS_OUTPUTS as _DBG_IO,
                     VRAM as _DBG_VRAM, log as _log)
from .subenv import build_isolation_env  # leaf; was a function-body cycle-dodge from .wrap

_DEBUG = _DBG_META  # backward compat -- all metadata debug logging uses META category
_CACHE_VERSION = "18"  # Bump when _METADATA_SCRIPT or cache format changes


def _write_cache_atomic(cache_file, cache_key, payload) -> None:
    """Write the metadata cache via temp-file + os.replace.

    The env dir is machine-global and two ComfyUI processes can start
    together; a plain write_text can be read half-written by the sibling.
    os.replace is atomic on POSIX and Windows within one filesystem."""
    import tempfile
    data = json.dumps({"cache_key": cache_key, "payload": payload},
                      ensure_ascii=False)
    fd, tmp = tempfile.mkstemp(dir=str(cache_file.parent),
                               prefix=cache_file.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, str(cache_file))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _warn_dropped_nodes(package_name: str, payload) -> None:
    """A node in nodes_failed is a USER-VISIBLE disappearance -- its saved
    workflows report 'node type not found'. One loud line per node, at the
    moment the payload is consumed, so the answer to 'where did my node go'
    is in the startup log next to the pack's own scan line."""
    failed = payload.get("nodes_failed") or {}
    for name, reason in failed.items():
        print(f"[comfy-env] WARNING: {package_name}: node {name!r} was DROPPED "
              f"by the metadata scan and will be missing from ComfyUI: {reason}",
              file=sys.stderr, flush=True)
    for w in (payload.get("sanitize_warnings") or [])[:10]:
        print(f"[comfy-env] {package_name}: scan sanitize: {w}",
              file=sys.stderr, flush=True)
    _warn_node_conformance(package_name, payload)


def _warn_node_conformance(package_name: str, payload) -> None:
    """One loud line per node whose contract isolation cannot carry.

    Every defect named here is SILENT at runtime: the node registers, runs,
    and produces a wrong or stale answer with no traceback. The startup log
    next to the pack's own scan line is the only place a user can meet them
    before the symptom.
    """
    for name, meta in (payload.get("nodes") or {}).items():
        if not isinstance(meta, dict):
            continue

        err = meta.get("input_types_error")
        if err:
            print(f"[comfy-env] WARNING: {package_name}: node {name!r} "
                  f"INPUT_TYPES() raised during the scan and is registered "
                  f"with ZERO inputs -- every widget and socket is missing: "
                  f"{err}", file=sys.stderr, flush=True)

        if meta.get("fingerprint_args") is not None:
            print(f"[comfy-env] WARNING: {package_name}: node {name!r} defines "
                  f"IS_CHANGED/fingerprint_inputs, which is NOT forwarded "
                  f"across isolation. ComfyUI will treat this node as never "
                  f"changing and serve its cached output until restart. If it "
                  f"reads a file, a clock or an API, make that an input.",
                  file=sys.stderr, flush=True)




def _describe_value(name: str, v) -> str:
    """Single-line summary of a value for I/O logging."""
    try:
        import torch
        if isinstance(v, torch.Tensor):
            shape = ",".join(str(s) for s in v.shape)
            return f"{name}: {v.dtype} [{shape}] {v.device}"
    except ImportError:
        pass
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            shape = ",".join(str(s) for s in v.shape)
            return f"{name}: {v.dtype} [{shape}]"
    except ImportError:
        pass
    if isinstance(v, (list, tuple)) and len(v) > 0:
        first = v[0]
        try:
            import torch
            if isinstance(first, torch.Tensor):
                shape = ",".join(str(s) for s in first.shape)
                return f"{name}: {len(v)}x {first.dtype} [{shape}] {first.device}"
        except (ImportError, AttributeError):
            pass
        return f"{name}: {type(v).__name__}[{len(v)}]"
    if isinstance(v, (str, int, float, bool)):
        s = repr(v)
        if len(s) > 60:
            s = s[:57] + "..."
        return f"{name}: {s}"
    return f"{name}: {type(v).__name__}"


def _log_vram(label: str) -> None:
    """Log compact GPU memory state."""
    try:
        import comfy.model_management as mm
        dev = mm.get_torch_device()
        if dev.type != "cuda":
            return
        total = mm.get_total_memory(dev) // (1024 * 1024)
        free = mm.get_free_memory(dev) // (1024 * 1024)
        used = total - free
        _log(f"[VRAM] {label}: {used} / {total} MB")
        # Loaded models
        loaded = mm.current_loaded_models
        if loaded:
            parts = []
            for lm in loaded:
                n = lm.model.model.__class__.__name__
                gpu_mb = lm.model_loaded_memory() // (1024 * 1024)
                parts.append(f"{n} ({gpu_mb} MB)")
            _log(f"[VRAM] Loaded: {', '.join(parts)}")
    except ImportError:
        # No comfy -- try raw torch
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                used = (total - free) // (1024 * 1024)
                total_mb = total // (1024 * 1024)
                _log(f"[VRAM] {label}: {used} / {total_mb} MB")
        except Exception:
            pass


# Metadata extraction script (runs in isolation subprocess)

_METADATA_SCRIPT = r'''
import sys
import os
import json
import importlib

# Windows: register DLL directories BEFORE any extension module imports.
# Python 3.8+ doesn't search PATH for DLLs -- os.add_dll_directory() required.
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    _env_root = os.path.dirname(sys.executable)
    os.add_dll_directory(_env_root)
    _lib_bin = os.path.join(_env_root, "Library", "bin")
    if os.path.isdir(_lib_bin):
        os.add_dll_directory(_lib_bin)
    _dlls_dir = os.path.join(_env_root, "DLLs")
    if os.path.isdir(_dlls_dir):
        os.add_dll_directory(_dlls_dir)

# Pre-import torch on Windows so its bundled libiomp5md.dll/fbgemm.dll claim the
# DLL name slots before anything else loads them. The diagnostic probe below and
# any later numpy/MKL import will otherwise pull conda-forge's libiomp5md.dll
# under the same name; torch's own libiomp is then shadowed and fbgemm's import
# table calls into mismatched exports -> WinError 127 on first `import torch`.
# DO NOT reorder.
if sys.platform == "win32":
    try:
        import torch  # noqa: F401
    except ImportError:
        pass

_debug = os.environ.get("COMFY_ENV_DEBUG", "").lower() in ("1", "true", "yes")

working_dir = sys.argv[1]
package_name = sys.argv[2]
output_path = sys.argv[3]

sys.path.insert(0, working_dir)
os.chdir(working_dir)

# Add ComfyUI source dir to sys.path so nodes can import folder_paths, comfy_api etc.
_comfyui_base = os.environ.get("COMFYUI_BASE")
if _comfyui_base and _comfyui_base not in sys.path:
    sys.path.insert(1, _comfyui_base)

# On Desktop app, redirect folder_paths to the user data dir (for input/output/models)
_comfyui_user_dir = os.environ.get("COMFYUI_USER_DIR")
if _comfyui_user_dir:
    try:
        import folder_paths
        folder_paths.base_path = _comfyui_user_dir
        folder_paths.output_directory = os.path.join(_comfyui_user_dir, "output")
        folder_paths.input_directory = os.path.join(_comfyui_user_dir, "input")
        folder_paths.user_directory = os.path.join(_comfyui_user_dir, "user")
    except ImportError:
        pass


# ---------------------------------------------------------------------------

# Redirect stdout to stderr so any prints from imported code (or pixi/torch
# DLL loaders) are captured for debugging but never mix with our protocol --
# the payload is written to a dedicated file, not stdout.
sys.stdout = sys.stderr

if _debug:
    print(f"[meta-scan] importing {package_name} from {working_dir}", file=sys.stderr, flush=True)
module = importlib.import_module(package_name)
if _debug:
    print(f"[meta-scan] import OK", file=sys.stderr, flush=True)

try:
    from comfy_api.internal import _ComfyNodeInternal as _V3Base
except Exception:
    _V3Base = None

# --- Node discovery: V1 dict, else V3 comfy_entrypoint -------------------
# Mirrors ComfyUI's load_custom_node (nodes.py). Upstream prefers the V1 dict
# and only falls through to comfy_entrypoint when it is absent-or-None -- note
# an EMPTY dict still wins there, so a pack exporting {} plus an entrypoint
# registers nothing upstream. We treat empty-and-has-entrypoint as V3 instead,
# because for an isolated pack the empty dict is our own scan result, not the
# author's intent.
_class_map = dict(getattr(module, "NODE_CLASS_MAPPINGS", None) or {})
_display_v3 = {}
_entrypoint = getattr(module, "comfy_entrypoint", None)
_has_entrypoint = _entrypoint is not None
_v3_error = None
_discovery = "v1" if _class_map else "none"

if not _class_map and _has_entrypoint:
    if not callable(_entrypoint):
        _v3_error = "comfy_entrypoint is not callable"
    else:
        try:
            import asyncio as _asyncio
            import inspect as _inspect

            async def _await_if_needed(_v):
                return await _v if _inspect.isawaitable(_v) else _v

            async def _collect_v3(_ep):
                _ext = await _await_if_needed(_ep())
                if _ext is None:
                    raise RuntimeError("comfy_entrypoint returned None")
                _on_load = getattr(_ext, "on_load", None)
                if callable(_on_load):
                    await _await_if_needed(_on_load())
                return await _await_if_needed(_ext.get_node_list())

            _node_list = _asyncio.run(_collect_v3(_entrypoint))
            if not isinstance(_node_list, list):
                raise RuntimeError("get_node_list() did not return a list")
            for _node_cls in _node_list:
                _schema = _node_cls.GET_SCHEMA()
                _class_map[_schema.node_id] = _node_cls
                if getattr(_schema, "display_name", None) is not None:
                    _display_v3[_schema.node_id] = _schema.display_name
            _discovery = "v3"
            print(f"[meta-scan] comfy_entrypoint: {len(_class_map)} node(s)",
                  file=sys.stderr, flush=True)
        except Exception as _e:
            _v3_error = f"{type(_e).__name__}: {_e}"
            print(f"[meta-scan] comfy_entrypoint failed: {_v3_error}",
                  file=sys.stderr, flush=True)

_ACCEL_VOCAB = ("cuda", "rocm", "xpu", "mps")


def _normalize_accel(value, node_name):
    """ACCELERATOR -> sorted list of backends, or None for CPU-capable.

    Accepts a string or a list/tuple: a node that runs on some but not all
    GPU backends says so directly (["cuda", "mps"]). There is no "any GPU"
    sentinel -- spell out the backends the node actually supports.

    An unrecognized value is a hard error. It used to be str()'d into
    something no backend could ever equal, which hid the node on EVERY
    machine, silently, including one with the right hardware.
    """
    if not value:
        return None
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        raise TypeError(
            "%s: ACCELERATOR must be a string or a list of strings, got %s"
            % (node_name, type(value).__name__))
    out = []
    for item in items:
        key = str(item).strip().lower()
        if key not in _ACCEL_VOCAB:
            raise ValueError(
                "%s: ACCELERATOR value %r is not a known backend (%s)"
                % (node_name, item, ", ".join(_ACCEL_VOCAB)))
        if key not in out:
            out.append(key)
    return sorted(out)


nodes = {}
for name, cls in _class_map.items():
    meta = {
        "function": getattr(cls, "FUNCTION", None),
        "category": getattr(cls, "CATEGORY", ""),
        "output_node": getattr(cls, "OUTPUT_NODE", False),
        "return_types": getattr(cls, "RETURN_TYPES", ()),
        "return_names": getattr(cls, "RETURN_NAMES", ()),
        "output_is_list": getattr(cls, "OUTPUT_IS_LIST", None),
        "input_is_list": getattr(cls, "INPUT_IS_LIST", None),
        "module_name": cls.__module__,
        "class_name": cls.__name__,
        # Accelerator declaration (comfy-env convention): a sorted list of
        # "cuda" / "rocm" / "xpu" / "mps", or None. Meaning: the node
        # REQUIRES one of these backends at execution; absent = CPU-capable.
        "accelerator": _normalize_accel(getattr(cls, "ACCELERATOR", None), name),
    }

    # Validation / fingerprint contracts (captured as ARG NAMES, not code).
    # The parent synthesizes named-arg replacements: forwarding either to a
    # worker is banned -- VALIDATE_INPUTS runs at prompt validation and
    # IS_CHANGED once per node per prompt, so a worker call there cold-spawns
    # every env before anything executes.
    def _named_args(fn):
        """(parameter names, declared **kwargs) -- ComfyUI reads BOTH.

        An input escapes the built-in min/max/combo checks if it is NAMED in
        the argspec OR the function has a catch-all (execution.py:889-893,
        applied at :1019). Capturing only the names silently narrows a
        `(cls, **kwargs)` validate -- which asks to exempt everything -- into
        exempting nothing, and a workflow that submits fine natively is then
        rejected once the pack is isolated.
        """
        import inspect as _inspect
        try:
            spec = _inspect.getfullargspec(fn)
        except TypeError:
            return None, False
        args = [a for a in spec.args if a not in ("cls", "self", "s")]
        return args, spec.varkw is not None
    _validate = None
    _validate_varkw = False
    _fingerprint = None
    _has_lazy_cb = False
    if _V3Base is not None and isinstance(cls, type) and issubclass(cls, _V3Base):
        try:
            from comfy_api.internal import first_real_override as _fro
            _v = _fro(cls, "validate_inputs")
            if _v is not None:
                _validate, _validate_varkw = _named_args(_v)
            _f = _fro(cls, "fingerprint_inputs")
            if _f is not None:
                _fingerprint, _ = _named_args(_f)
            # An AUTHOR-PROVIDED check_lazy_status is the whole gate. The
            # inherited default is unreachable: first_real_override breaks at
            # GET_BASE_CLASS(), which for a V3 node IS ComfyNode, so a native
            # V3 node declaring `lazy` without an override also runs with
            # None. Only an explicit one is a regression when we drop it.
            _has_lazy_cb = _fro(cls, "check_lazy_status") is not None
        except Exception:
            pass
    else:
        _v = getattr(cls, "VALIDATE_INPUTS", None)
        if callable(_v):
            _validate, _validate_varkw = _named_args(_v)
        _f = getattr(cls, "IS_CHANGED", None)
        if callable(_f):
            _fingerprint, _ = _named_args(_f)
        # V1: CheckLazyMixin is opt-in, not a base-class default, so a plain
        # getattr is the honest test.
        _has_lazy_cb = callable(getattr(cls, "check_lazy_status", None))
    meta["has_check_lazy"] = bool(_has_lazy_cb)
    meta["validate_args"] = _validate
    meta["validate_varkw"] = _validate_varkw
    meta["fingerprint_args"] = _fingerprint

    # Call INPUT_TYPES classmethod
    if hasattr(cls, "INPUT_TYPES") and callable(cls.INPUT_TYPES):
        try:
            meta["input_types"] = cls.INPUT_TYPES()
        except Exception as e:
            meta["input_types"] = {"required": {}}
            meta["input_types_error"] = str(e)

    # V3 detection + native metadata capture. The real class lives here in the
    # isolation env, so its schema-backed classproperties/GET_NODE_INFO_V1 resolve
    # correctly; we capture the plain-dict results (Schema objects are not serializable).
    is_v3 = _V3Base is not None and isinstance(cls, type) and issubclass(cls, _V3Base)
    meta["is_v3"] = is_v3
    if is_v3:
        try:
            meta["node_info_v1"] = cls.GET_NODE_INFO_V1()
            meta["not_idempotent"] = bool(getattr(cls, "NOT_IDEMPOTENT", False))
            meta["accept_all_inputs"] = bool(getattr(cls, "ACCEPT_ALL_INPUTS", False))
        except Exception as e:
            # degrade gracefully: build the V1 proxy for this node instead
            meta["is_v3"] = False
            print(f"[meta-scan] V3 capture failed for {name}: {e}", file=sys.stderr, flush=True)

    nodes[name] = meta

display = dict(getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", None) or {})
for _k, _v in _display_v3.items():
    display.setdefault(_k, _v)

# Discover API routes declared via ROUTES convention (walk all imported submodules)
routes = list(getattr(module, "ROUTES", []))
for mod_name, mod_obj in list(sys.modules.items()):
    if mod_name == package_name or not mod_name.startswith(package_name + "."):
        continue
    for r in getattr(mod_obj, "ROUTES", []):
        r.setdefault("module", mod_name)
        routes.append(r)
for r in routes:
    r.setdefault("module", package_name)

# Accelerator import-rule check (observed, not predicted): nothing has
# executed during this scan, so if any declared accelerator package is in
# sys.modules NOW, some module imported it at top level -- the pattern that
# makes this whole scan die on machines where the package isn't installed.
# Map import names -> distributions so dist names like "faithc-aot" match
# their actual import name.
_accel_violations = []
_accel_pkgs = [p.strip().lower() for p in
               os.environ.get("COMFY_ENV_ACCEL_PKGS", "").split(",") if p.strip()]
if _accel_pkgs:
    _import_names = set()
    try:
        from importlib.metadata import packages_distributions
        for _imp, _dists in packages_distributions().items():
            for _d in _dists:
                if _d.lower().replace("_", "-") in [p.replace("_", "-") for p in _accel_pkgs]:
                    _import_names.add(_imp)
    except Exception:
        pass
    for _p in _accel_pkgs:  # name-variant fallback for missing metadata
        _import_names.add(_p.replace("-", "_"))
    for _m in list(sys.modules):
        _top = _m.split(".", 1)[0]
        if _top in _import_names:
            _accel_violations.append(_top)
    _accel_violations = sorted(set(_accel_violations))

payload = {"nodes": nodes, "display": display, "routes": routes,
           "accel_import_violations": _accel_violations,
           "discovery": _discovery,
           "has_comfy_entrypoint": _has_entrypoint,
           "v3_entrypoint_error": _v3_error}

# Convert the payload to strict JSON-safe data, tracking the key path.
#
# The predecessor (_sanitize + pickle) had a fallthrough `return obj` that
# let unknown objects ride the pickle into the parent, where they failed
# LATE and cryptically: an Enum default made the parent's unpickle raise
# ModuleNotFoundError outside the caught tuple, silently deleting the whole
# pack; a torch.dtype survived to web.json_response and 500'd the entire
# /object_info. Here every value is decided NOW, in the process where the
# offending type is importable and debuggable, and the failure names the
# node and the exact key path.
#
# Policy (2026-08 review): coerce what has an obvious faithful mapping,
# DROP-AND-WARN the single offending key/element for the rest -- a weird
# widget default must cost that widget's default, never the node, and a
# broken node must never cost the pack.
_COERCE = (str, int, float, bool)
_warnings = []

def _to_json(obj, path):
    if obj is None or type(obj) in (str, int, bool):
        return obj
    if type(obj) is float:
        if obj != obj or obj in (float("inf"), float("-inf")):
            _warnings.append(f"{path}: non-finite float {obj!r} dropped")
            return _DROP
        return obj
    for base in _COERCE:
        if isinstance(obj, base) and type(obj) is not base:
            return base(obj)   # AnyType("*"), IntEnum, np.bool_ subclasses
    if type(obj).__module__ == "numpy" and hasattr(obj, "item"):
        return _to_json(obj.item(), path)   # numpy scalars
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, str):
                key = str(k)   # collapse str subclasses
            else:
                _warnings.append(
                    f"{path}: non-string dict key {k!r} ({type(k).__name__}) "
                    f"dropped (json would silently stringify it)")
                continue
            val = _to_json(v, f"{path}[{key!r}]")
            if val is not _DROP:
                out[key] = val
        return out
    if isinstance(obj, (list, tuple)):
        out = []
        for i, v in enumerate(obj):
            val = _to_json(v, f"{path}[{i}]")
            if val is not _DROP:
                out.append(val)
        return out
    _warnings.append(
        f"{path}: {type(obj).__module__}.{type(obj).__name__} is not "
        f"JSON-serializable, dropped: {repr(obj)[:120]}")
    return _DROP

_DROP = object()

# Per-node containment: one bad node lands in nodes_failed with its reason;
# the other nodes survive. Top-level keys convert without a net -- a failure
# there is a scan bug and must exit non-zero.
_converted_nodes = {}
_nodes_failed = {}
for _name, _meta in payload["nodes"].items():
    try:
        _v = _to_json(_meta, "nodes[" + repr(_name) + "]")
        _converted_nodes[_name] = {} if _v is _DROP else _v
    except Exception as _e:
        _nodes_failed[_name] = repr(_e)[:300]
        print(f"[meta-scan] node {_name} dropped: {_e}", file=sys.stderr, flush=True)
payload["nodes"] = _converted_nodes
rest = {k: _to_json(v, k) for k, v in payload.items() if k != "nodes"}
payload = {"nodes": payload["nodes"], **{k: (None if v is _DROP else v) for k, v in rest.items()}}
if _nodes_failed:
    payload["nodes_failed"] = _nodes_failed
if _warnings:
    payload["sanitize_warnings"] = _warnings
    for _w in _warnings:
        print(f"[meta-scan] WARNING: {_w}", file=sys.stderr, flush=True)

# Serialize FIRST, then write: json.dump straight to the stream could raise
# mid-write and leave a truncated-but-nonempty file for the parent's salvage
# path to trip over. allow_nan=False is a second belt behind the converter;
# ensure_ascii=False + explicit utf-8 because real payloads carry non-ASCII
# and an unmarked Windows default is cp1252.
_data = json.dumps(payload, allow_nan=False, ensure_ascii=False)
with open(output_path, "w", encoding="utf-8") as _f:
    _f.write(_data)
'''


# Metadata fetching

def _warn_empty_v3_scan(package_name: str, payload: dict, node_count: int) -> None:
    """Loud diagnostic when a pack has a V3 entrypoint but the scan found nothing.

    Zero nodes from a pack that ships `comfy_entrypoint` is never a legitimate
    result -- and it is invisible downstream, because ComfyUI's loader takes the
    V1 branch on an empty-but-present NODE_CLASS_MAPPINGS, returns True, and
    never reaches its own "lack of NODE_CLASS_MAPPINGS or comfy_entrypoint"
    warning (nodes.py). So if we stay quiet here, nothing anywhere says a word.
    """
    if node_count > 0 or not payload.get("has_comfy_entrypoint"):
        return
    err = payload.get("v3_entrypoint_error")
    print(
        f"[comfy-env] WARNING: {package_name} declares comfy_entrypoint() but "
        f"the metadata scan registered 0 nodes.",
        file=sys.stderr, flush=True)
    if err:
        print(f"[comfy-env]   the entrypoint raised: {err}",
              file=sys.stderr, flush=True)
    else:
        print("[comfy-env]   the entrypoint returned no nodes.",
              file=sys.stderr, flush=True)
    print(
        "[comfy-env]   This pack's nodes will be MISSING from ComfyUI, and "
        "nothing else will report it.",
        file=sys.stderr, flush=True)


def fetch_metadata(
    env_dir: Path,
    package_name: str,
    working_dir: Path,
    env_vars: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Fetch node metadata by running a subprocess in the isolation env.

    Args:
        env_dir: Path to the materialized env in the global cache
        package_name: Dotted module name (e.g., "nodes.gpu")
        working_dir: Package root for sys.path (e.g., .../ComfyUI-GeometryPack/)
        env_vars: Additional environment variables from comfy-env.toml

    Returns:
        {"nodes": {name: meta_dict, ...}, "display": {name: display_name, ...}}
        Empty dict on failure.
    """
    python = env_dir / ("python.exe" if sys.platform == "win32" else "bin/python")
    if not python.exists():
        print(f"[comfy-env] No Python in {env_dir}, skipping metadata scan")
        return {"nodes": {}, "display": {}}

    # --- Metadata cache ---
    # Invalidate when ANY .py file in the package changes (not just __init__.py).
    # Uses max mtime of all .py files -- fast (stat calls only, no file reads).
    # .json since 0.4.33. The legacy .pkl is deliberately NOT unlinked: the
    # workspace is machine-global (environment/cache.py get_workspace_dir) and
    # installs pinned to older comfy-env versions still read and rewrite it --
    # deleting it here would make the two versions thrash a full rescan at
    # each other on every restart. GC it in a post-barrage release.
    cache_file = env_dir / ".metadata_cache.json"
    pkg_dir = working_dir / package_name.replace(".", "/")
    try:
        py_files = sorted(pkg_dir.rglob("*.py"))
        if py_files:
            mtimes = "|".join(
                f"{f.relative_to(pkg_dir)}:{f.stat().st_mtime_ns}"
                for f in py_files
            )
            pkg_hash = hashlib.sha256(mtimes.encode()).hexdigest()[:16]
        else:
            pkg_hash = "empty"
    except (OSError, FileNotFoundError):
        pkg_hash = "missing"
    cache_key = f"v{_CACHE_VERSION}:{pkg_hash}"

    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached.get("cache_key") == cache_key:
                payload = cached["payload"]
                node_count = len(payload.get("nodes", {}))
                if _DEBUG or node_count > 0:
                    print(f"[comfy-env] Cache hit for {package_name}: {node_count} nodes",
                          file=sys.stderr, flush=True)
                # A zero-node payload is cached like any other, and the cache
                # only invalidates on a .py mtime change. Warning only on the
                # fresh-scan path meant a broken entrypoint screamed once and
                # was silent on every startup after.
                _warn_empty_v3_scan(package_name, payload, node_count)
                _warn_dropped_nodes(package_name, payload)
                return payload
            elif _DEBUG:
                print(f"[comfy-env] Cache stale for {package_name} "
                      f"(key {cached.get('cache_key')} != {cache_key})",
                      file=sys.stderr, flush=True)
        except Exception:
            pass  # Corrupted cache, fall through to scan

    # Build proper subprocess environment (DLL paths, library paths, etc.)
    scan_env = build_isolation_env(python, env_vars)
    # Write script and allocate a dedicated payload file. The worker dumps the
    # JSON payload into `output_file` so the protocol is decoupled from
    # stdout/stderr (which pixi, torch DLL loaders, and other noise can
    # contaminate, especially on Windows).
    script_file = None
    output_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="comfy_meta_", delete=False,
            encoding="utf-8",
        ) as f:
            f.write(_METADATA_SCRIPT)
            script_file = f.name

        out_fd, output_file = tempfile.mkstemp(
            suffix=".json", prefix="comfy_meta_out_",
        )
        os.close(out_fd)

        t0 = time.perf_counter()

        # Route the metadata scan through `pixi run -e <env> --frozen` so pixi
        # handles activation (PATH for delay-loaded DLLs, CONDA_PREFIX,
        # [activation.env] vars like KMP_DUPLICATE_LIB_OK). Hand-rolling the
        # PATH activation worked for delay-load resolution but missed the
        # [activation.env] block — without KMP_DUPLICATE_LIB_OK, torch's OMP
        # guard or MKL init failed mid-scan. The previous attempt also
        # hard-coded the env name as "default", silently scanning under the
        # wrong env's site-packages. `--frozen` avoids re-resolving the
        # lockfile per scan.
        # COMPONENT, not substring: `".pixi" in str(python)` also matched a
        # host interpreter living under ~/.pixi from `pixi global`, and is
        # the same reading-semantics-out-of-a-path mistake as the libomp bug.
        is_pixi = ".pixi" in Path(python).parts
        if is_pixi:
            from ..environment.cache import resolve_pixi_manifest
            from ..pixi import PIXI
            # Per-env layout: python lives at
            #   <workspace>/envs/<name>/.pixi/envs/default/{bin,Scripts}/python
            # so the per-env manifest is at <workspace>/envs/<name>/pixi.toml
            # and the pixi env inside it is always named "default". Each
            # env's manifest is independent -- a parse error in one cannot
            # break this scan.
            env_root = python.parent if sys.platform == "win32" else python.parent.parent
            manifest_path, env_pixi_name = resolve_pixi_manifest(env_root)
            cmd = [
                PIXI, "run", "--as-is",
                "--manifest-path", str(manifest_path),
                "-e", env_pixi_name,
                "python", script_file, str(working_dir), package_name, output_file,
            ]
        else:
            cmd = [str(python), script_file, str(working_dir), package_name, output_file]

        if _DEBUG:
            print(f"[comfy-env] Metadata scan: {' '.join(cmd)}", file=sys.stderr, flush=True)
            path_sep = ";" if sys.platform == "win32" else ":"
            scan_path = scan_env.get("PATH", "")
            print(f"[comfy-env] Scan env PATH for {package_name}:", file=sys.stderr, flush=True)
            for i, p in enumerate(scan_path.split(path_sep)):
                print(f"[comfy-env]   [{i}] {p}", file=sys.stderr, flush=True)

        result = subprocess.run(
            cmd,
            capture_output=True,
            cwd=str(working_dir),
            env=scan_env,
        )

        elapsed = time.perf_counter() - t0

        # Always print stderr from scan subprocess when debug is on
        if _DEBUG:
            scan_stderr = result.stderr.decode("utf-8", errors="replace").strip()
            if scan_stderr:
                print(f"[comfy-env] Metadata scan stderr for {package_name}:", file=sys.stderr, flush=True)
                for line in scan_stderr.splitlines():
                    print(f"[comfy-env]   {line}", file=sys.stderr, flush=True)

        if result.returncode != 0:
            rc = result.returncode
            hex_rc = f" 0x{rc & 0xFFFFFFFF:08X}" if sys.platform == "win32" and rc < 0 else ""
            # A crash during interpreter teardown does not invalidate a payload that
            # was already written. Environments bundling native libraries (bpy /
            # embedded Blender, spconv) can fault on exit -- 0xC0000005 on Windows --
            # after the scan has fully succeeded. Discarding the payload here makes
            # every node in the pack silently vanish from the registry.
            # Trust the file: if it parses and contains nodes, salvage it and
            # warn. A truncated JSON raises JSONDecodeError exactly as a
            # truncated pickle raised -- same fall-through to the failure
            # branch. (The child serializes fully before writing, so a
            # truncated-but-nonempty file means the process died mid-write.)
            salvaged = None
            try:
                if output_file and os.path.getsize(output_file) > 0:
                    candidate = json.loads(
                        Path(output_file).read_text(encoding="utf-8"))
                    if candidate.get("nodes"):
                        salvaged = candidate
            except Exception:
                salvaged = None
            if salvaged is not None:
                print(f"[comfy-env] Metadata scan for {package_name} crashed on exit "
                      f"(exit {rc}{hex_rc}) but the payload was complete -- salvaged "
                      f"{len(salvaged['nodes'])} nodes.", file=sys.stderr, flush=True)
                try:
                    _write_cache_atomic(cache_file, cache_key, salvaged)
                except Exception:
                    pass
                return salvaged
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            print(f"[comfy-env] Metadata scan failed for {package_name} "
                  f"(exit {rc}{hex_rc}, {elapsed:.1f}s):", file=sys.stderr, flush=True)
            for line in stderr.splitlines()[-10:]:
                print(f"[comfy-env]   {line}", file=sys.stderr, flush=True)
            return {"nodes": {}, "display": {}}

        # Read the payload from the dedicated file -- never touches stdout,
        # so pixi/torch/DLL-loader noise can't corrupt the protocol.
        try:
            payload = json.loads(Path(output_file).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError) as e:
            # json.JSONDecodeError is a ValueError subclass. This set is
            # CLOSED and enumerable -- the pickle version let a
            # ModuleNotFoundError from an unpicklable payload escape to the
            # blanket handler below, silently deleting the whole pack.
            stderr_tail = result.stderr.decode("utf-8", errors="replace").strip().splitlines()[-5:]
            print(
                f"[comfy-env] Metadata scan: payload unreadable for {package_name}: "
                f"{type(e).__name__}: {e}",
                file=sys.stderr, flush=True,
            )
            for line in stderr_tail:
                print(f"[comfy-env]   {line}", file=sys.stderr, flush=True)
            return {"nodes": {}, "display": {}}

        node_count = len(payload.get("nodes", {}))
        if _DEBUG or node_count > 0:
            print(f"[comfy-env] Scanned {package_name}: {node_count} nodes ({elapsed:.1f}s)", file=sys.stderr, flush=True)
        _warn_empty_v3_scan(package_name, payload, node_count)
        _warn_dropped_nodes(package_name, payload)

        # --- Write cache (atomic: two ComfyUI processes share this dir) ---
        try:
            _write_cache_atomic(cache_file, cache_key, payload)
        except Exception:
            pass  # Non-fatal

        return payload

    except Exception as e:
        print(f"[comfy-env] Metadata scan error for {package_name}: {e}", file=sys.stderr, flush=True)
        return {"nodes": {}, "display": {}}
    finally:
        for path in (script_file, output_file):
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass


# Dynamic combo refresh (parent-side directory rescan)
#
# Isolated nodes are represented in the main process by a proxy whose
# INPUT_TYPES would otherwise return a snapshot captured once at scan time, so
# combos built from a filesystem scan (e.g. "list the files in input/cad") never
# refresh -- newly uploaded files never appear in the dropdown, even on reload.
#
# A node opts a combo into live refresh by attaching a marker to its options
# dict (via io.Combo.Input(extra_dict=...)). Simple single-directory form:
#     {"comfy_env_dynamic_dir": "cad",
#      "comfy_env_exts": [".step", ".stp", ".iges", ".igs", ".brep"],
#      "comfy_env_placeholder": "(no CAD files found in input/cad)"}
# Richer multi-source form (e.g. a recursive subfolder plus the input root), where
# each source is {"dir": <subdir>, "recursive": bool, "rel_to_input": bool}:
#     {"comfy_env_dynamic_dir": "3d",   # trigger; ignored when sources given
#      "comfy_env_sources": [
#          {"dir": "3d", "recursive": True,  "rel_to_input": True},
#          {"dir": "",   "recursive": False, "rel_to_input": False}],
#      "comfy_env_exts": [...], "comfy_env_placeholder": "..."}
# All dirs are relative to ComfyUI's input directory; rel_to_input controls whether
# returned values are relative to the input root (e.g. "3d/foo.obj") or to the
# scanned dir (e.g. "foo.obj"). The scan is plain os.listdir/os.walk of a ComfyUI
# input folder -- it needs none of the node's isolated dependencies and runs
# cheaply in the parent on every /object_info, keeping the fast read path off the
# (possibly slow/hung) worker.

_DYNAMIC_DIR_KEY = "comfy_env_dynamic_dir"
_DYNAMIC_SOURCES_KEY = "comfy_env_sources"


#: How long the options refresh will wait for a worker's lock before giving
#: up. Short on purpose: this is cosmetic work on the /object_info path, and
#: the answer if we lose the race is "use the cached list", which is what the
#: whole feature degrades to anyway. Never worth stalling ComfyUI for.
_REFRESH_LOCK_TIMEOUT = 0.25


def _refresh_combo_options(env_dir, module_name, class_name):
    """Ask a warm, idle worker for a node's real combo options. Or don't.

    This is the whole of comfy-env's live-dropdown support, and it is a
    ladder with three rungs:

      1. the worker for this env is alive AND idle -- ask it to re-run the
         node's own INPUT_TYPES, and use what comes back. No guessing: it is
         the pack's real function, in the env that owns it, so it sees a
         hand-rolled os.walk exactly as well as a folder_paths lookup.
      2. alive but mid-call -- `send_command_no_spawn` returns "busy" after
         a quarter second and we fall through rather than queueing behind a
         node that may run for minutes.
      3. never started, or dead -- fall through.

    Rungs 2 and 3 both mean "keep the cached options", which is exactly the
    behaviour of a node whose dropdown was never dynamic. So the worst case
    is what every isolated node did before this existed, and the ladder can
    only ever add.

    The rung that matters is deliberately NOT "spawn a worker". /object_info
    enumerates the entire node registry on every page load, so spawning there
    would start every isolated env on the machine to draw a dropdown -- and
    the spawn happens on the event loop, which would freeze ComfyUI's HTTP
    server for the length of it. Live options are worth a socket round trip
    to a process that already exists. They are not worth starting one.

    Never raises. A raise inside INPUT_TYPES makes core omit the node from
    /object_info entirely, and a vanished node is strictly worse than a
    stale dropdown.
    """
    try:
        from .pool import _WORKER_POOL
        entry = _WORKER_POOL.get(str(env_dir))
        if entry is None:
            return None                      # rung 3: nothing to ask
        worker = entry[0]
        resp = worker.send_command_no_spawn(
            "refresh_input_types", lock_timeout=_REFRESH_LOCK_TIMEOUT,
            module=module_name, class_name=class_name)
        # "dead" / "busy" are sentinel strings, not responses -- rungs 3 and 2.
        if not isinstance(resp, dict) or resp.get("status") != "ok":
            return None
        opts = resp.get("options")
        return opts if isinstance(opts, dict) and opts else None
    except Exception:
        return None


def _splice_combo_options(sections, fresh):
    """Overlay fresh option lists onto a cached input-spec snapshot.

    Only the options list is replaced, never the trailing config dict, so
    tooltips, defaults and `image_upload` survive. An input the worker did
    not report is left exactly as captured -- the fresh answer is additive,
    never authoritative about what is absent.
    """
    out = {}
    for section, entries in sections.items():
        new_entries = dict(entries or {})
        for name, opts in (fresh.get(section) or {}).items():
            entry = new_entries.get(name)
            if not isinstance(entry, (list, tuple)) or not entry:
                continue
            if not isinstance(entry[0], (list, tuple)):
                continue                     # not a combo; leave it alone
            if not opts:
                continue                     # empty listing keeps the cache
            new_entries[name] = [list(opts)] + list(entry[1:])
        out[section] = new_entries
    return out


def _combo_input_names(input_types):
    """Every input whose options list could change under us.

    An input spec's first element is a list only for a combo, so this is the
    exact set the refresh can rewrite -- and therefore the exact set whose
    membership check has to be relaxed. Numeric inputs are not in it, so
    min/max clamps keep working; that distinction is the whole reason this
    is a named list and not a **kwargs validate.
    """
    out = []
    for section in ("required", "optional"):
        for name, entry in (input_types.get(section) or {}).items():
            if (isinstance(entry, (list, tuple)) and entry
                    and isinstance(entry[0], (list, tuple))):
                out.append(name)
    return out


def _make_named_validate(names, varkw: bool = False):
    """A classmethod `f(cls, a=None, b=None, ..., **kwargs) -> True` carrying
    EXACTLY the original's exemptions.

    The signature is the whole point: execution.py exempts an input from its
    built-in min/max/combo checks iff the input's name appears in the validate
    function's argspec, OR the function declares a catch-all
    (execution.py:889-893, applied at :1019).

    So `varkw` is reproduced, never invented. Adding `**kwargs` to a validate
    the author wrote with explicit names would exempt every input including
    numeric clamps users rely on -- which is why this is driven by what the
    scan observed rather than by convenience. Omitting it when the author DID
    write one is the mirror error and the worse of the two: it attaches
    nothing at all, ComfyUI re-imposes every check the author deliberately
    waived, and a workflow that submits fine natively is rejected once the
    pack is isolated.

    Names that are not identifiers are skipped -- they could not be exempted
    this way anyhow.
    """
    names = [n for n in names if isinstance(n, str) and n.isidentifier()
             and n not in ("cls", "self", "s")]
    if not names and not varkw:
        return None, []
    sig = ", ".join(p for p in (
        ", ".join(f"{n}=None" for n in names),
        "**kwargs" if varkw else "",
    ) if p)
    ns: dict = {}
    exec(f"def _cev_validate(cls, {sig}):\n    return True\n", ns)
    return classmethod(ns["_cev_validate"]), names


# --- Provider resolution (parent-side re-listing) ---------------------------

# Per-pack private category registry. NEVER written into ComfyUI's global
# folder_names_and_paths: the global feeds /models, the asset seeder, upload
# routing, and is snapshot-pushed wholesale into EVERY worker. Host-defined
# categories always win; recorded registrations are unioned across packs
# (core unions paths for shared category names, so must we).
_PACK_FOLDER_REGISTRY: Dict[str, Dict[str, Any]] = {}


# provider-json -> (names, {dir: mtime_ns}). The mtime map records EVERY
# directory visited (core's own cached_filename_list_ trick), so a change at
# any depth invalidates; a re-check is a handful of stats, not a walk.
_LIVE_CACHE: Dict[str, Any] = {}


# Proxy class builder

#: HiddenHolder attributes comfy-env forwards, lowercase of their sentinel
#: (`Hidden.prompt = "PROMPT"`, _io.py:1592-1606). `dynprompt` is absent on
#: purpose: it is a live DynamicPrompt, not JSON, and nothing ComfyUI ships
#: consumes it.
_V3_HIDDEN_ATTRS = ("prompt", "extra_pnginfo", "unique_id",
                    "auth_token_comfy_org", "api_key_comfy_org",
                    "comfy_usage_source")


def _call_in_worker(*, worker_spec, module_name, class_name, method_name,
                    self_state, kwargs, node_name, hidden=None,
                    state_id=None, state_dict=None):
    """Run one node call in the pack's worker. Shared by the V1 and V3 proxies.

    Keyword-only on purpose: the two closures this replaces took nine and
    eleven positional single-letter arguments in slightly different orders, and
    being unable to mix them up was the only safety the duplication provided.

    self_state: V1 sends the instance __dict__; V3 sends None. It is passed IN
    rather than derived here -- a V3 proxy's bound object is the CLASS, whose
    __dict__ is truthy and full of classmethod objects that json.dumps refuses,
    so deriving it would kill every V3 node call on the wire.

    Callers shape kwargs BEFORE calling (V1 strips hidden keys and nests
    DynamicCombo dotted keys; V3 does neither), so what is logged here is
    exactly what goes out.
    """
    from .pool import (_get_or_create_worker, _remove_worker,
                       _register_new_patchers)
    from .tensor_utils import prepare_for_ipc_recursive
    from .errors import translate_error
    from .workers.base import WorkerError

    env_dir = worker_spec[0]
    if _DBG_IO:
        _log(f"[comfy-env] Running {node_name}: "
             + ", ".join(_describe_value(k, v) for k, v in kwargs.items()))
    if _DBG_VRAM:
        _log_vram(f"Before {node_name}")

    worker, gen = _get_or_create_worker(*worker_spec)
    _t0 = time.perf_counter()
    try:
        kwargs = {k: prepare_for_ipc_recursive(v) for k, v in kwargs.items()}
        # In-flight marker for admission: while this worker computes, its
        # models charge full size in _worker_held_bytes (an aimdo worker can
        # lazily re-fault mid call with no parent-visible signal, so the
        # ceiling must be the supremum). Incremented before any worker Python
        # runs; decremented in the finally STRICTLY AFTER the boundary census
        # applies, so no window opens between the flag clearing and the peak
        # decaying to sampled truth.
        worker.begin_call()
        try:
            result = worker.call_method(
                module_name=module_name,
                class_name=class_name,
                method_name=method_name,
                self_state=self_state,
                kwargs=kwargs,
                hidden=hidden,
                state_id=state_id,
                timeout=600.0,
            )
        finally:
            # Register auto-detected models even when the call RAISED: the
            # weights are on the GPU either way, and a model with no ledger
            # entry can never be evicted.
            try:
                _register_new_patchers(env_dir, worker, gen)
            except Exception as _re:
                _log(f"[comfy-env] patcher registration failed: {_re}")
            # Apply the state return the same way, and for the same reason: a
            # node that mutated self and then raised keeps the mutation, like
            # a non-isolated node would.
            if state_dict is not None:
                try:
                    _sso = getattr(worker, "_last_state_out", None)
                    if _sso is not None:
                        worker._last_state_out = None
                        state_sync.apply_state_out(state_dict, _sso)
                except Exception as _se:
                    _log(f"[comfy-env] state apply failed: {_se}")
            # After the census apply above, never before: the ordering is
            # what closes the mid-call-echo-then-idle window.
            worker.end_call()

        result = prepare_for_ipc_recursive(result)

        if _DBG_IO:
            elapsed = time.perf_counter() - _t0
            out_desc = (", ".join(_describe_value(f"[{i}]", v)
                                  for i, v in enumerate(result))
                        if isinstance(result, tuple)
                        else _describe_value("result", result))
            _log(f"[comfy-env] {node_name} done ({elapsed:.2f}s): {out_desc}")
        if _DBG_VRAM:
            _log_vram(f"After {node_name}")
        return result
    # TimeoutError is an OSError, not a RuntimeError, so it was declining this
    # clause and skipping _remove_worker entirely: the pool entry, the temp
    # dir, the socket and the worker's _WORKER_HELD reserve all survived a
    # worker that had just been killed, and the reserve never shrank because
    # node boundary republishes refuse to lower without a receipt.
    except (RuntimeError, ConnectionError, TimeoutError) as te:
        # Always on: a worker teardown is the single most consequential
        # event in this file and used to be silent. Name the env and the
        # exception class so a user's log shows WHICH worker died and why.
        _log(f"[comfy-env] worker teardown env={env_label(env_dir)} "
             f"node={node_name} cause={type(te).__name__}: {str(te)[:200]}")
        _remove_worker(env_dir)
        raise
    except WorkerError as we:
        # Typed translation frontier. This clause must stay BELOW the
        # (RuntimeError, ConnectionError) teardown: torch.OutOfMemoryError IS
        # a RuntimeError, so raising it from inside the try body would tear
        # down the worker on every OOM right before ComfyUI's recovery runs.
        # Here the WorkerError has already been declined by that clause (it
        # subclasses Exception only), the worker survives, and what we raise
        # from this handler is out of this try's reach.
        translated = translate_error(we)
        if translated is we:
            raise
        stats = getattr(we, "oom_stats", None) or {}
        host_free = None
        try:
            import comfy.model_management as _mm
            host_free = int(_mm.get_free_memory(_mm.get_torch_device()))
        except Exception:
            pass
        _log(f"[comfy-env] worker {we.error_kind} env={env_label(env_dir)} "
             f"node={node_name} call_id={getattr(worker, '_call_counter', '?')} "
             f"worker_allocated={stats.get('allocated')} "
             f"worker_reserved={stats.get('reserved')} "
             f"worker_largest_free={stats.get('largest_free_block')} "
             f"host_free={host_free} -> raising {type(translated).__name__}")
        raise translated


def _build_v3_proxy_class(
    node_name: str,
    meta: Dict[str, Any],
    env_dir: Path,
    package_root: Path,
    sys_path: list,
    env_vars: Dict[str, str],
    health_check_timeout: float = DEFAULT_HEALTH_CHECK_TIMEOUT,
) -> type:
    """Build a V3-native proxy: a genuine io.ComfyNode subclass, so ComfyUI's
    server treats it exactly like the real V3 node it stands in for.

    Why this exists: the V1 proxy needed a compatibility hack that flattened every
    DynamicCombo option's children into dotted `parent.child` optional inputs
    (the V1 execution path drops undeclared dotted inputs). The flattened extras
    were then materialized as widgets by the frontend on node creation, showing
    every backend's parameters at once. As a real V3 class, the server finalizes
    dynamic inputs (`get_finalized_class_inputs`) and nests dotted prompt keys
    into dicts (`build_nested_inputs`) natively -- no flattening, no manual
    re-nesting, no hidden-tuple unwrapping.

    Contract notes (all verified against execution.py/server.py/_io.py):
    - `/object_info` for V3 is served solely by `GET_NODE_INFO_V1()`; we return
      the dict captured verbatim from the real class during the metadata scan.
    - `FUNCTION = "execute"` (a plain string) bypasses the base's
      EXECUTE_NORMALIZED classproperty, so no SCHEMA object is needed even when
      the worker returns an expand graph -- the output stage handles NodeOutput
      and plain dicts class-agnostically.
    - `define_schema` must EXIST as a distinct classmethod (VALIDATE_CLASS checks
      for a real override) but is never called, because every schema-derived
      classproperty is shadowed with plain attrs below.
    - `@final` decorators in comfy_api are typing-only; shadowing is legal at
      runtime.
    """
    from comfy_api.latest import io as _comfy_io

    func_name = meta["function"] or "EXECUTE_NORMALIZED"
    module_name = meta["module_name"]
    class_name = meta["class_name"]
    input_types = {k: dict(v) if isinstance(v, dict) else v
                   for k, v in meta.get("input_types", {"required": {}}).items()}
    node_info = meta["node_info_v1"]

    return_types = tuple(meta.get("return_types", ()) or ())
    output_is_list = meta.get("output_is_list")
    if not output_is_list or len(output_is_list) != len(return_types):
        output_is_list = tuple(bool(x) for x in (output_is_list or ())) \
            + (False,) * (len(return_types) - len(output_is_list or ()))

    # Live options, unconditionally. Every isolated node asks its own worker
    # for fresh combo options on every /object_info -- see _refresh_combo_options
    # for the three rungs and why "spawn one" is not among them. When no worker
    # answers, both methods return exactly the cached snapshot, which is what
    # they returned for every node before this existed.
    @classmethod
    def _input_types(cls, _cached=input_types, _ed=env_dir,
                     _mod=module_name, _cn=class_name):
        fresh = _refresh_combo_options(_ed, _mod, _cn)
        if not fresh:
            return _cached
        result = _splice_combo_options(
            {s: e for s, e in _cached.items() if s in ("required", "optional")}, fresh)
        for s, e in _cached.items():
            if s not in ("required", "optional"):
                result[s] = e
        return result

    @classmethod
    def _get_node_info_v1(cls, _info=node_info, _ed=env_dir,
                          _mod=module_name, _cn=class_name):
        info = dict(_info)
        fresh = _refresh_combo_options(_ed, _mod, _cn)
        if fresh:
            inp = info.get("input") or {}
            sections = {s: e for s, e in inp.items()
                        if s in ("required", "optional")}
            new_inp = dict(inp)
            new_inp.update(_splice_combo_options(sections, fresh))
            info["input"] = new_inp
        # RELATIVE_PYTHON_MODULE is stamped on the registered class by the main
        # process (nodes.py), not the scan env -- re-read it here or the frontend
        # crashes on python_module=None.
        info["python_module"] = getattr(cls, "RELATIVE_PYTHON_MODULE", None) or "nodes"
        return info

    @classmethod
    def _define_schema_stub(cls):
        raise NotImplementedError(
            f"comfy-env V3 proxy for {node_name}: define_schema is a stub -- the real "
            f"schema lives in the isolation env. If this is reached, a code path is "
            f"bypassing the proxy's shadowed classmethods.")

    def _make_v3_proxy(fn, mod, cn, ed, pr, sp, ev, hct, nn):
        def proxy(cls, **kwargs):
            # V3 hidden inputs never travel as kwargs. ComfyUI puts them in
            # v3_data["hidden_inputs"], and PREPARE_CLASS_CLONE hangs them on
            # a per-call class clone (_io.py:2085-2091). This proxy IS a real
            # io.ComfyNode with FUNCTION="execute", so ComfyUI runs that
            # machinery on it like any other node -- meaning `cls` arriving
            # here already carries a populated HiddenHolder. It used to be
            # discarded on the next line; read it instead.
            #
            # getattr-guarded: outside that dispatch `hidden` is the class
            # default None (_io.py:1977), and dynprompt is skipped because it
            # is a live object rather than data.
            _hidden = None
            _holder = getattr(cls, "hidden", None)
            if _holder is not None:
                _hidden = [[_a.upper(), None, _v]
                           for _a in _V3_HIDDEN_ATTRS
                           for _v in (getattr(_holder, _a, None),)
                           if _v is not None] or None
            # self_state is the literal None, never derived from `cls`.
            return _call_in_worker(
                worker_spec=(ed, pr, sp, ev, hct),
                module_name=mod, class_name=cn, method_name=fn,
                self_state=None, kwargs=kwargs, node_name=nn,
                hidden=_hidden,
            )
        return proxy

    attrs = {
        "INPUT_TYPES": _input_types,
        "GET_NODE_INFO_V1": _get_node_info_v1,
        "define_schema": _define_schema_stub,
        "execute": classmethod(_make_v3_proxy(
            func_name, module_name, class_name,
            env_dir, package_root, sys_path, env_vars,
            health_check_timeout, node_name,
        )),
        "FUNCTION": "execute",
        "RETURN_TYPES": return_types,
        "RETURN_NAMES": tuple(meta.get("return_names", ()) or ()),
        "OUTPUT_IS_LIST": tuple(output_is_list),
        "INPUT_IS_LIST": bool(meta.get("input_is_list") or False),
        "OUTPUT_NODE": bool(meta.get("output_node", False)),
        "NOT_IDEMPOTENT": bool(meta.get("not_idempotent", False)),
        "ACCEPT_ALL_INPUTS": bool(meta.get("accept_all_inputs", False)),
        "CATEGORY": meta.get("category", ""),
        # Every schema-backed lazy classproperty on the io.ComfyNode base must be
        # shadowed with a plain value: any that isn't falls through to the base
        # descriptor, which calls GET_SCHEMA() -> our define_schema stub -> raise.
        "DESCRIPTION": node_info.get("description") or "",
        "EXPERIMENTAL": bool(node_info.get("experimental", False)),
        "DEPRECATED": bool(node_info.get("deprecated", False)),
        "DEV_ONLY": bool(node_info.get("dev_only", False)),
        "API_NODE": node_info.get("api_node"),
        "HAS_INTERMEDIATE_OUTPUT": bool(node_info.get("has_intermediate_output", False)),
        "OUTPUT_TOOLTIPS": tuple(node_info["output_tooltips"])
            if node_info.get("output_tooltips") else None,
        "_comfy_env_isolated": True,
        "_comfy_env_module": module_name,
        "_comfy_env_class": class_name,
        "_comfy_env_accelerator": meta.get("accelerator"),
    }

    # check_lazy_status -- forwarded, and ONLY when the author wrote one.
    #
    # Unlike VALIDATE_INPUTS / IS_CHANGED this is safe to forward: upstream
    # calls it at execution.py:504-517 for a node it has ALREADY picked to
    # execute, whose worker is spawning anyway. One round-trip on a warm or
    # imminent worker, not a cold spawn per node per prompt.
    #
    # Built by the same factory as "execute" with the name swapped, so hidden
    # inputs, the per-call class clone and error translation all come along
    # unchanged -- upstream feeds this method the same input_data_all it
    # feeds the real call. Lowercase, and in the proxy's own __dict__: that
    # is what first_real_override (execution.py:504) walks the MRO for.
    #
    # Conditional is load-bearing. Attach unconditionally and every isolated
    # node pays a round-trip to be told "no questions". And a node that
    # declares `lazy` WITHOUT writing this method gets None natively too --
    # ComfyNode's own default is unreachable (first_real_override breaks at
    # GET_BASE_CLASS(), which for a V3 node IS ComfyNode). Synthesising one
    # here would make isolation more correct than upstream, so a pack built
    # only against comfy-env would ship broken for everyone else.
    if meta.get("has_check_lazy"):
        attrs["check_lazy_status"] = classmethod(_make_v3_proxy(
            "check_lazy_status", module_name, class_name,
            env_dir, package_root, sys_path, env_vars,
            health_check_timeout, node_name,
        ))

    # Validation exemption (named-arg, parent-side, NEVER forwarded).
    # V3 branch note: execution.py resolves the V3 validate via
    # first_real_override(cls, "validate_inputs") -- the LOWERCASE name.
    # Attaching VALIDATE_INPUTS here would be dead code.
    _marked = _combo_input_names(input_types)
    _exempt = list(_marked) + [a for a in (meta.get("validate_args") or [])
                               if a not in _marked]
    _validate_cm, _ = _make_named_validate(
        _exempt, varkw=bool(meta.get("validate_varkw")))
    if _validate_cm is not None:
        attrs["validate_inputs"] = _validate_cm

    # No staleness fingerprint. It hashed a dynamic input's value by
    # the mtime of the file it resolved to, so overwriting a mesh in
    # place re-executed instead of serving the cached result. It needed
    # a per-input directory spec to resolve that path, and nothing knows
    # one any more: which inputs are file listings is now the worker's
    # answer at refresh time, not a fact the parent holds at build time.
    # Attaching an mtime hash to every combo instead would change caching
    # for nodes that have nothing to do with files. Recorded as a real
    # loss rather than quietly dropped.

    return type(class_name, (_comfy_io.ComfyNode,), attrs)


# Accelerator availability (ACCELERATOR node declaration)

_MACHINE_BACKEND: Optional[str] = None


def _machine_backend() -> str:
    """Detected torch backend of THIS machine ("cuda"/"rocm"/"mps"/"cpu"...), cached."""
    global _MACHINE_BACKEND
    if _MACHINE_BACKEND is None:
        try:
            from ..detection.backend import detect_backend
            _MACHINE_BACKEND = detect_backend()[0]
        except Exception:
            _MACHINE_BACKEND = "cpu"
    return _MACHINE_BACKEND


def _accelerator_available(accels: Optional[List[str]]) -> bool:
    """Can a node declaring these accelerators execute on this machine?

    None/empty = CPU-capable, always available. Otherwise this machine's
    backend must be one of them. The scan normalizes the declaration to a
    list (_normalize_accel), so there is no scalar case to handle here.
    """
    if not accels:
        return True
    return _machine_backend() in accels


def _build_unavailable_stub(node_name: str, meta: Dict[str, Any]) -> type:
    """Visible-but-unavailable node for machines lacking the declared backend.

    Deliberately NOT hidden: a missing node type breaks workflow load with an
    inscrutable frontend error. The stub registers with the real inputs and
    outputs, badges its description, and raises a named-reason error when
    executed.
    """
    accel = meta.get("accelerator") or []
    backend = _machine_backend()
    names = " or ".join(a.upper() for a in accel)
    reason = (
        f"Node '{node_name}' requires {names}; this machine has "
        f"backend '{backend}'"
        + (" (no NVIDIA GPU detected)" if accel == ["cuda"] and backend == "cpu" else "")
        + ". Use a CPU-capable alternative node or run on a machine with "
        f"{names}."
    )
    input_types = meta.get("input_types", {"required": {}})
    func_name = meta.get("function") or "execute"

    def _raiser(self, **kwargs):
        raise RuntimeError(reason)

    attrs = {
        "RETURN_TYPES": tuple(meta.get("return_types", ())),
        "RETURN_NAMES": tuple(meta.get("return_names", ())),
        "FUNCTION": func_name,
        "CATEGORY": meta.get("category", ""),
        "OUTPUT_NODE": meta.get("output_node", False),
        "INPUT_TYPES": classmethod(lambda cls, _cached=input_types: _cached),
        "DESCRIPTION": f"(requires {names} -- unavailable on this machine)",
        # ADR-0012: hidden from the node picker (ComfyUI hides DEPRECATED
        # nodes from menu/search) but still REGISTERED so shared workflows
        # load and dispatcher node-ids resolve.
        "DEPRECATED": True,
        "_comfy_env_isolated": True,
        "_comfy_env_accelerator": accel,
        "_comfy_env_unavailable": reason,
        func_name: _raiser,
    }
    print(f"[comfy-env] {node_name}: requires {names}, machine backend is "
          f"'{backend}' -- registered but hidden from the node menu",
          file=sys.stderr, flush=True)
    return type(f"ComfyEnvUnavailable_{meta.get('class_name', node_name)}", (), attrs)


def build_proxy_class(
    node_name: str,
    meta: Dict[str, Any],
    env_dir: Path,
    package_root: Path,
    sys_path: list,
    env_vars: Dict[str, str],
    health_check_timeout: float = DEFAULT_HEALTH_CHECK_TIMEOUT,
) -> type:
    """Build a proxy class from metadata that delegates execution to subprocess.

    V3-scanned nodes (is_v3 + node_info_v1 captured) get a V3-native proxy --
    see _build_v3_proxy_class. V1 nodes keep the classic V1 proxy below, with
    its DynamicCombo-flattening/nesting compatibility hacks.

    Nodes declaring an ACCELERATOR the machine lacks get a visible
    unavailable-stub instead of a worker proxy.
    """
    if not _accelerator_available(meta.get("accelerator")):
        return _build_unavailable_stub(node_name, meta)

    if meta.get("is_v3") and meta.get("node_info_v1") is not None:
        try:
            return _build_v3_proxy_class(
                node_name, meta, env_dir, package_root, sys_path,
                env_vars, health_check_timeout)
        except Exception as e:
            print(f"[comfy-env] V3 proxy build failed for {node_name}, "
                  f"falling back to V1 proxy: {e}", file=sys.stderr, flush=True)

    func_name = meta["function"]
    module_name = meta["module_name"]
    class_name = meta["class_name"]
    input_types = meta.get("input_types", {"required": {}})
    input_types = {k: dict(v) for k, v in input_types.items()}  # shallow copy

    # Expand DynamicCombo children for V1 compatibility.
    # ComfyUI only expands DynamicCombo schemas for V3 nodes (subclasses of
    # _ComfyNodeInternal).  Since the proxy is a V1 class, child inputs with
    # dotted names (e.g. "backend.target_edge_length") are silently dropped by
    # get_input_data().  We flatten all option children into "optional" so
    # they survive, then nest them back in the proxy function before sending
    # to the worker.
    dynamic_combo_parents = set()
    for section in ("required", "optional"):
        if section not in input_types:
            continue
        for name, info in list(input_types[section].items()):
            if (isinstance(info, (list, tuple)) and len(info) >= 1
                    and info[0] == "COMFY_DYNAMICCOMBO_V3"):
                dynamic_combo_parents.add(name)
                opts_dict = info[1] if len(info) > 1 and isinstance(info[1], dict) else {}
                for opt in opts_dict.get("options", []):
                    child_inputs = opt.get("inputs", {})
                    for child_section in ("required", "optional"):
                        if child_section in child_inputs:
                            for child_name, child_info in child_inputs[child_section].items():
                                dotted = f"{name}.{child_name}"
                                input_types.setdefault("optional", {})[dotted] = child_info

    # Build class attributes
    attrs = {
        "RETURN_TYPES": tuple(meta.get("return_types", ())),
        "RETURN_NAMES": tuple(meta.get("return_names", ())),
        "FUNCTION": func_name,
        "CATEGORY": meta.get("category", ""),
        "OUTPUT_NODE": meta.get("output_node", False),
        "_comfy_env_isolated": True,
        "_comfy_env_module": module_name,
        "_comfy_env_class": class_name,
        "_comfy_env_accelerator": meta.get("accelerator"),
    }

    # Batch processing attributes (ComfyUI uses these for list iteration)
    if meta.get("output_is_list") is not None:
        attrs["OUTPUT_IS_LIST"] = tuple(meta["output_is_list"])
    if meta.get("input_is_list") is not None:
        attrs["INPUT_IS_LIST"] = meta["input_is_list"]

    # V3 nodes wrap hidden values in tuples, e.g. ("UNIQUE_ID",), but V1
    # hidden processing in execution.py compares bare strings.  Unwrap them
    # so ComfyUI injects hidden values properly for proxy (V1) classes.
    if "hidden" in input_types:
        input_types["hidden"] = {
            k: v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v
            for k, v in input_types["hidden"].items()
        }

    # Combos needing live re-listing: journaled providers first, legacy
    # markers second (journal wins per input) -- same collector and same
    # Live options, same ladder as the V3 proxy. ComfyUI calls INPUT_TYPES on
    # every /object_info, prompt validation and node execution, so the miss
    # path has to be cheap: a dict lookup in the worker pool, and on a hit a
    # socket round trip to a process that is already running.
    @classmethod
    def _input_types(cls, _cached=input_types, _ed=env_dir,
                     _mod=module_name, _cn=class_name):
        fresh = _refresh_combo_options(_ed, _mod, _cn)
        if not fresh:
            return _cached
        sections = {s: e for s, e in _cached.items()
                    if s in ("required", "optional")}
        result = _splice_combo_options(sections, fresh)
        for s, e in _cached.items():
            if s not in ("required", "optional"):
                result[s] = e
        return result
    attrs["INPUT_TYPES"] = _input_types

    # Hidden inputs travel in their own frame field, never as kwargs.
    #
    # ComfyUI matches hidden inputs on the SENTINEL (`h[x] == "PROMPT"`,
    # execution.py:212-224) and delivers under the author's own parameter
    # name. This map records both halves so the worker can put each value
    # back under the name that node actually declared -- whatever it is.
    # The predecessor of this map keyed on the literal spelling `unique_id`,
    # which silently dropped a node declaring `{"node_id": "UNIQUE_ID"}`.
    #
    # DYNPROMPT is excluded: it is a live DynamicPrompt, not JSON, and
    # nothing ComfyUI ships consumes it. A node declaring it sees the same
    # absent kwarg it sees today.
    _hidden_map = {k: v for k, v in input_types.get("hidden", {}).items()
                   if isinstance(v, str) and v != "DYNPROMPT"}

    # Proxy FUNCTION method -- reuses persistent worker across calls
    def _make_proxy(fn, mod, cn, ed, pr, sp, ev, hct, dcp, nn, hmap):
        def proxy(self, **kwargs):
            # Lift hidden inputs out of kwargs into their own channel, keyed
            # by sentinel and carrying the author's parameter name with them.
            _hidden = None
            if hmap:
                _hidden = [[hmap[k], k, kwargs.pop(k)]
                           for k in list(kwargs) if k in hmap]

            # Nest DynamicCombo inputs: flat dotted keys -> nested dicts.
            # e.g. {"backend": "grid", "backend.smooth_normals": "true", ...}
            #   -> {"backend": {"backend": "grid", "smooth_normals": "true"}, ...}
            if dcp:
                nested = {}
                for k, v in kwargs.items():
                    if '.' in k:
                        parent, child = k.split('.', 1)
                        if parent in dcp:
                            nested.setdefault(parent, {})[child] = v
                            continue
                    if k in dcp:
                        nested.setdefault(k, {})[k] = v
                        continue
                    nested[k] = v
                kwargs = nested

            _d = self.__dict__ if hasattr(self, "__dict__") else None
            if _d is None:
                return _call_in_worker(
                    worker_spec=(ed, pr, sp, ev, hct),
                    module_name=mod, class_name=cn, method_name=fn,
                    self_state=None, kwargs=kwargs, node_name=nn,
                    hidden=_hidden,
                )
            # The state id is parent-only bookkeeping: stripped from every
            # outbound state, never written by the worker. It names this
            # instance to the worker, which decides for itself whether
            # __init__ has run there (ComfyUI's sweep drops the instance,
            # the next one gets a new id, so __init__-once maps onto the
            # sweep for free).
            _sid = _d.setdefault(state_sync.STATE_ID_KEY, uuid.uuid4().hex[:12])
            return _call_in_worker(
                worker_spec=(ed, pr, sp, ev, hct),
                module_name=mod, class_name=cn, method_name=fn,
                self_state=state_sync.outbound_state(_d),
                kwargs=kwargs, node_name=nn, hidden=_hidden,
                state_id=_sid, state_dict=_d,
            )
        return proxy

    attrs[func_name] = _make_proxy(
        func_name, module_name, class_name,
        env_dir, package_root, sys_path, env_vars, health_check_timeout,
        dynamic_combo_parents, node_name, _hidden_map,
    )

    # check_lazy_status, forwarded iff the author wrote one -- see the V3
    # builder for why forwarding is sound here and why conditional matters.
    # Upstream's V1 test is getattr(obj, "check_lazy_status") on the
    # INSTANCE (execution.py:506); a plain function in attrs binds as a
    # method, exactly as FUNCTION does. Reusing _make_proxy is the point:
    # the hidden-input lift, DynamicCombo re-nesting and state sync all
    # apply to this call for the same reason they apply to the real one.
    if meta.get("has_check_lazy"):
        attrs["check_lazy_status"] = _make_proxy(
            "check_lazy_status", module_name, class_name,
            env_dir, package_root, sys_path, env_vars, health_check_timeout,
            dynamic_combo_parents, node_name, _hidden_map,
        )

    # Validation exemption (named-arg, parent-side, NEVER forwarded). The V1
    # branch of execution.py looks up the UPPERCASE name.
    _marked = _combo_input_names(input_types)
    _exempt = list(_marked) + [a for a in (meta.get("validate_args") or [])
                               if a not in _marked]
    _validate_cm, _ = _make_named_validate(
        _exempt, varkw=bool(meta.get("validate_varkw")))
    if _validate_cm is not None:
        attrs["VALIDATE_INPUTS"] = _validate_cm

    # No staleness fingerprint. It hashed a dynamic input's value by
    # the mtime of the file it resolved to, so overwriting a mesh in
    # place re-executed instead of serving the cached result. It needed
    # a per-input directory spec to resolve that path, and nothing knows
    # one any more: which inputs are file listings is now the worker's
    # answer at refresh time, not a fact the parent holds at build time.
    # Attaching an mtime hash to every combo instead would change caching
    # for nodes that have nothing to do with files. Recorded as a real
    # loss rather than quietly dropped.

    # Create the class
    proxy_cls = type(class_name, (), attrs)
    return proxy_cls
