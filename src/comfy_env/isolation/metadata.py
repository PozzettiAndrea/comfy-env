"""Metadata extraction for isolation nodes via subprocess scan.

Spawns a short-lived subprocess in the isolation env's Python to import node modules
and extract class metadata (INPUT_TYPES, RETURN_TYPES, etc.). The main process never
imports isolation code -- it builds proxy classes from the serialized metadata.
"""

import base64
import glob
import hashlib
import os
import pickle
import subprocess
import sys
import tempfile
import time
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..config.types import DEFAULT_HEALTH_CHECK_TIMEOUT
from ..debug import META as _DBG_META, INPUTS_OUTPUTS as _DBG_IO, VRAM as _DBG_VRAM

_DEBUG = _DBG_META  # backward compat — all metadata debug logging uses META category
_CACHE_VERSION = "8"  # Bump when _METADATA_SCRIPT or cache format changes


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


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
        # No comfy — try raw torch
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                used = (total - free) // (1024 * 1024)
                total_mb = total // (1024 * 1024)
                _log(f"[VRAM] {label}: {used} / {total_mb} MB")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Metadata extraction script (runs in isolation subprocess)
# ---------------------------------------------------------------------------

_METADATA_SCRIPT = r'''
import sys
import os
import pickle
import base64
import importlib

# Windows: register DLL directories BEFORE any extension module imports.
# Python 3.8+ doesn't search PATH for DLLs — os.add_dll_directory() required.
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    _env_root = os.path.dirname(sys.executable)
    os.add_dll_directory(_env_root)
    _lib_bin = os.path.join(_env_root, "Library", "bin")
    if os.path.isdir(_lib_bin):
        os.add_dll_directory(_lib_bin)
    _dlls_dir = os.path.join(_env_root, "DLLs")
    if os.path.isdir(_dlls_dir):
        os.add_dll_directory(_dlls_dir)
    _host_sp = os.environ.get("_COMFY_ENV_HOST_SP")
    if _host_sp:
        _torch_lib = os.path.join(_host_sp, "torch", "lib")
        if os.path.isdir(_torch_lib):
            os.add_dll_directory(_torch_lib)

# Print environment diagnostics to stderr (survives crashes)
_debug = os.environ.get("COMFY_ENV_DEBUG", "").lower() in ("1", "true", "yes")
if _debug:
    _sep = ";" if sys.platform == "win32" else ":"
    print(f"[meta-scan] python: {sys.executable}", file=sys.stderr, flush=True)
    print(f"[meta-scan] PATH:", file=sys.stderr, flush=True)
    for _i, _p in enumerate(os.environ.get("PATH", "").split(_sep)):
        print(f"[meta-scan]   [{_i}] {_p}", file=sys.stderr, flush=True)
    # List shared libraries in the env's library directory
    _env_root = os.path.dirname(sys.executable)
    if sys.platform == "win32":
        import ctypes
        _lib_dir = os.path.join(_env_root, "Library", "bin")
        _ext = ".dll"
    elif sys.platform == "darwin":
        _lib_dir = os.path.join(_env_root, "..", "lib")
        _ext = ".dylib"
    else:
        _lib_dir = os.path.join(_env_root, "..", "lib")
        _ext = ".so"
    _lib_dir = os.path.normpath(_lib_dir)
    if os.path.isdir(_lib_dir):
        _libs = sorted(f for f in os.listdir(_lib_dir) if _ext in f.lower())
        print(f"[meta-scan] {_lib_dir}: {len(_libs)} shared libs", file=sys.stderr, flush=True)
        # On Windows, probe each DLL with ctypes to detect missing dependencies
        if sys.platform == "win32":
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(_lib_dir)
            for _d in _libs:
                _path = os.path.join(_lib_dir, _d)
                try:
                    ctypes.WinDLL(_path)
                    print(f"[meta-scan]   OK   {_d}", file=sys.stderr, flush=True)
                except OSError as _e:
                    print(f"[meta-scan]   FAIL {_d}: {_e}", file=sys.stderr, flush=True)
        else:
            # Deduplicate versioned symlinks: libfoo.so.1.2.0 -> libfoo.so
            import re
            _seen = set()
            _deduped = []
            for _d in _libs:
                _base = re.sub(r'\.so[\d.]*$', '.so', _d) if '.so' in _d else re.sub(r'\.(\d+\.)*dylib$', '.dylib', _d)
                if _base not in _seen:
                    _seen.add(_base)
                    _deduped.append(_base)
            for _d in _deduped[:40]:
                print(f"[meta-scan]   {_d}", file=sys.stderr, flush=True)
            if len(_deduped) > 40:
                print(f"[meta-scan]   ... and {len(_deduped) - 40} more", file=sys.stderr, flush=True)
    else:
        print(f"[meta-scan] lib dir NOT FOUND: {_lib_dir}", file=sys.stderr, flush=True)
    # Also print LD/DYLD paths on non-Windows
    if sys.platform != "win32":
        _ld = os.environ.get("LD_LIBRARY_PATH") or os.environ.get("DYLD_LIBRARY_PATH")
        if _ld:
            print(f"[meta-scan] LD/DYLD_LIBRARY_PATH: {_ld}", file=sys.stderr, flush=True)

working_dir = sys.argv[1]
package_name = sys.argv[2]

sys.path.insert(0, working_dir)
os.chdir(working_dir)

# Add ComfyUI base to sys.path so nodes can import folder_paths etc.
_comfyui_base = os.environ.get("COMFYUI_BASE")
if _comfyui_base and _comfyui_base not in sys.path:
    sys.path.insert(1, _comfyui_base)

# Add host site-packages for torch inheritance (share_torch)
_host_sp = os.environ.get("_COMFY_ENV_HOST_SP")
if _host_sp and os.path.isdir(_host_sp) and _host_sp not in sys.path:
    if sys.platform == "darwin":
        sys.path.append(_host_sp)
    else:
        sys.path.insert(0, _host_sp)


# Redirect stdout to stderr during import so that any print() calls
# from imported code don't corrupt our base64 payload on stdout.
_real_stdout = sys.stdout
sys.stdout = sys.stderr

if _debug:
    print(f"[meta-scan] importing {package_name} from {working_dir}", file=sys.stderr, flush=True)
module = importlib.import_module(package_name)
if _debug:
    print(f"[meta-scan] import OK", file=sys.stderr, flush=True)

nodes = {}
for name, cls in getattr(module, "NODE_CLASS_MAPPINGS", {}).items():
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
    }
    # Call INPUT_TYPES classmethod
    if hasattr(cls, "INPUT_TYPES") and callable(cls.INPUT_TYPES):
        try:
            meta["input_types"] = cls.INPUT_TYPES()
        except Exception as e:
            meta["input_types"] = {"required": {}}
            meta["input_types_error"] = str(e)

    nodes[name] = meta

display = getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {})

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

payload = {"nodes": nodes, "display": display, "routes": routes}

# Sanitize payload: coerce subclass instances (e.g. AnyType(str)) back to
# plain built-in types so pickle doesn't embed module references that may
# not be importable in the main process.
_COERCE = {str: str, int: int, float: float, bool: bool, bytes: bytes}
def _sanitize(obj):
    if obj is None or type(obj) in (str, int, float, bool, bytes):
        return obj
    for base, ctor in _COERCE.items():
        if isinstance(obj, base) and type(obj) is not base:
            return ctor(obj)
    if isinstance(obj, dict):
        return {_sanitize(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        sanitized = [_sanitize(v) for v in obj]
        return type(obj)(sanitized) if type(obj) in (list, tuple) else list(sanitized)
    return obj

payload = _sanitize(payload)

# Restore real stdout for the payload write
sys.stdout = _real_stdout
sys.stdout.buffer.write(base64.b64encode(pickle.dumps(payload)))
'''


# ---------------------------------------------------------------------------
# Metadata fetching
# ---------------------------------------------------------------------------

def fetch_metadata(
    env_dir: Path,
    node_dir: Path,
    package_name: str,
    working_dir: Path,
    env_vars: Optional[Dict[str, str]] = None,
    host_torch_sp: Optional[Path] = None,
) -> Dict[str, Any]:
    """Fetch node metadata by running a subprocess in the isolation env.

    Args:
        env_dir: Path to _env_* directory
        node_dir: Path to the node subdirectory (e.g., nodes/gpu/)
        package_name: Dotted module name (e.g., "nodes.gpu")
        working_dir: Package root for sys.path (e.g., .../ComfyUI-GeometryPack/)
        env_vars: Additional environment variables from comfy-env.toml
        host_torch_sp: Host site-packages path for torch inheritance (share_torch)

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
    cache_file = env_dir / ".metadata_cache.pkl"
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
            cached = pickle.loads(cache_file.read_bytes())
            if cached.get("cache_key") == cache_key:
                payload = cached["payload"]
                node_count = len(payload.get("nodes", {}))
                if _DEBUG or node_count > 0:
                    print(f"[comfy-env] Cache hit for {package_name}: {node_count} nodes",
                          file=sys.stderr, flush=True)
                return payload
            elif _DEBUG:
                print(f"[comfy-env] Cache stale for {package_name} "
                      f"(key {cached.get('cache_key')} != {cache_key})",
                      file=sys.stderr, flush=True)
        except Exception:
            pass  # Corrupted cache, fall through to scan

    # Build proper subprocess environment (DLL paths, library paths, etc.)
    from .wrap import build_isolation_env
    scan_env = build_isolation_env(python, env_vars)
    if host_torch_sp:
        scan_env["_COMFY_ENV_HOST_SP"] = str(host_torch_sp)

    # Write script to temp file
    script_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="comfy_meta_", delete=False,
            encoding="utf-8",
        ) as f:
            f.write(_METADATA_SCRIPT)
            script_file = f.name

        t0 = time.perf_counter()

        # Launch pixi python directly instead of "pixi run" to avoid pixi
        # reinstalling torch from the lockfile (undoes share_torch uninstall).
        # Set up conda env vars manually for DLL search paths.
        is_pixi = ".pixi" in str(python)
        cmd = [str(python), script_file, str(working_dir), package_name]
        if sys.platform == "win32" and is_pixi:
            # Find pixi env root for PATH setup
            pixi_env_root = python
            while pixi_env_root.name != ".pixi" and pixi_env_root.parent != pixi_env_root:
                pixi_env_root = pixi_env_root.parent
            pixi_env_root = pixi_env_root / "envs" / "default"
            scan_env["CONDA_PREFIX"] = str(pixi_env_root)
            path_sep = ";"
            pixi_paths = [
                str(pixi_env_root),
                str(pixi_env_root / "Library" / "mingw-w64" / "bin"),
                str(pixi_env_root / "Library" / "usr" / "bin"),
                str(pixi_env_root / "Library" / "bin"),
                str(pixi_env_root / "Scripts"),
            ]
            current_path = scan_env.get("PATH", "")
            scan_env["PATH"] = path_sep.join(pixi_paths + [current_path])

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
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            print(f"[comfy-env] Metadata scan failed for {package_name} "
                  f"(exit {rc}{hex_rc}, {elapsed:.1f}s):", file=sys.stderr, flush=True)
            for line in stderr.splitlines()[-10:]:
                print(f"[comfy-env]   {line}", file=sys.stderr, flush=True)
            return {"nodes": {}, "display": {}}

        raw = result.stdout.strip()
        if not raw:
            print(f"[comfy-env] Metadata scan returned empty for {package_name}", file=sys.stderr, flush=True)
            return {"nodes": {}, "display": {}}

        payload = pickle.loads(base64.b64decode(raw))

        node_count = len(payload.get("nodes", {}))
        if _DEBUG or node_count > 0:
            print(f"[comfy-env] Scanned {package_name}: {node_count} nodes ({elapsed:.1f}s)", file=sys.stderr, flush=True)

        # --- Write cache ---
        try:
            cache_file.write_bytes(pickle.dumps({"cache_key": cache_key, "payload": payload}))
        except Exception:
            pass  # Non-fatal

        return payload

    except Exception as e:
        print(f"[comfy-env] Metadata scan error for {package_name}: {e}", file=sys.stderr, flush=True)
        return {"nodes": {}, "display": {}}
    finally:
        if script_file and os.path.exists(script_file):
            try:
                os.unlink(script_file)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Proxy class builder
# ---------------------------------------------------------------------------

def build_proxy_class(
    node_name: str,
    meta: Dict[str, Any],
    env_dir: Path,
    package_root: Path,
    sys_path: list,
    lib_path: Optional[str],
    env_vars: Dict[str, str],
    health_check_timeout: float = DEFAULT_HEALTH_CHECK_TIMEOUT,
) -> type:
    """Build a proxy class from metadata that delegates execution to subprocess.

    The returned class has all the ComfyUI metadata attributes (INPUT_TYPES,
    RETURN_TYPES, FUNCTION, CATEGORY, etc.) but the FUNCTION method spawns
    a SubprocessWorker to run the real code in the isolation env.
    """
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

    # INPUT_TYPES classmethod returning cached metadata
    @classmethod
    def _input_types(cls, _cached=input_types):
        return _cached
    attrs["INPUT_TYPES"] = _input_types

    # Hidden kwargs to strip before sending to worker (V3 execute() won't
    # accept them).  Keep unique_id since isolated nodes may need it.
    _hidden_strip = set(input_types.get("hidden", {}).keys()) - {"unique_id"}

    # Proxy FUNCTION method -- reuses persistent worker across calls
    def _make_proxy(fn, mod, cn, ed, pr, sp, lp, ev, hct, dcp, nn, hsk):
        def proxy(self, **kwargs):
            from .wrap import (_get_or_create_worker, _remove_worker,
                               _load_worker_models, _register_new_patchers)

            # Strip hidden kwargs that V3 execute() doesn't expect
            if hsk:
                kwargs = {k: v for k, v in kwargs.items() if k not in hsk}

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

            # I/O + VRAM logging (before call)
            if _DBG_IO:
                inputs_desc = ", ".join(_describe_value(k, v) for k, v in kwargs.items())
                _log(f"[comfy-env] Running {nn}: {inputs_desc}")
            if _DBG_VRAM:
                _log_vram(f"Before {nn}")

            worker, gen = _get_or_create_worker(ed, pr, sp, lp, ev, hct)
            _t0 = time.perf_counter()
            try:
                # Ensure any previously-registered subprocess models are on GPU
                _load_worker_models(ed)

                try:
                    from .tensor_utils import prepare_for_ipc_recursive
                    kwargs = {k: prepare_for_ipc_recursive(v) for k, v in kwargs.items()}
                except ImportError:
                    pass

                result = worker.call_method(
                    module_name=mod,
                    class_name=cn,
                    method_name=fn,
                    self_state=self.__dict__.copy() if hasattr(self, "__dict__") else None,
                    kwargs=kwargs,
                    timeout=600.0,
                )

                try:
                    from .tensor_utils import prepare_for_ipc_recursive
                    result = prepare_for_ipc_recursive(result)
                except ImportError:
                    pass

                # Create patchers for any models auto-detected during this call
                _register_new_patchers(ed, worker, gen)

                # I/O + VRAM logging (after call)
                if _DBG_IO:
                    elapsed = time.perf_counter() - _t0
                    if isinstance(result, tuple):
                        out_desc = ", ".join(
                            _describe_value(f"[{i}]", v) for i, v in enumerate(result)
                        )
                    else:
                        out_desc = _describe_value("result", result)
                    _log(f"[comfy-env] {nn} done ({elapsed:.2f}s): {out_desc}")
                if _DBG_VRAM:
                    _log_vram(f"After {nn}")

                return result
            except (RuntimeError, ConnectionError):
                _remove_worker(ed)
                raise
        return proxy

    attrs[func_name] = _make_proxy(
        func_name, module_name, class_name,
        env_dir, package_root, sys_path, lib_path, env_vars, health_check_timeout,
        dynamic_combo_parents, node_name, _hidden_strip,
    )

    # Create the class
    proxy_cls = type(class_name, (), attrs)
    return proxy_cls
