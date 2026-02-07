"""Metadata extraction for isolation nodes via subprocess scan.

Spawns a short-lived subprocess in the isolation env's Python to import node modules
and extract class metadata (INPUT_TYPES, RETURN_TYPES, etc.). The main process never
imports isolation code — it builds proxy classes from the serialized metadata.
"""

import base64
import glob
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

_DEBUG = os.environ.get("COMFY_ENV_DEBUG", "").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Metadata extraction script (runs in isolation subprocess)
# ---------------------------------------------------------------------------

_METADATA_SCRIPT = r'''
import sys
import os
import pickle
import base64
import importlib

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
            # Deduplicate versioned symlinks: libfoo.so.1.2.0 → libfoo.so
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

payload = {"nodes": nodes, "display": display}
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
    timeout: float = 30.0,
    env_vars: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Fetch node metadata by running a subprocess in the isolation env.

    Args:
        env_dir: Path to _env_* directory
        node_dir: Path to the node subdirectory (e.g., nodes/gpu/)
        package_name: Dotted module name (e.g., "nodes.gpu")
        working_dir: Package root for sys.path (e.g., .../ComfyUI-GeometryPack/)
        timeout: Max seconds to wait for subprocess
        env_vars: Additional environment variables from comfy-env.toml

    Returns:
        {"nodes": {name: meta_dict, ...}, "display": {name: display_name, ...}}
        Empty dict on failure.
    """
    python = env_dir / ("python.exe" if sys.platform == "win32" else "bin/python")
    if not python.exists():
        print(f"[comfy-env] No Python in {env_dir}, skipping metadata scan")
        return {"nodes": {}, "display": {}}

    # Build proper subprocess environment (DLL paths, library paths, etc.)
    from .wrap import build_isolation_env
    scan_env = build_isolation_env(python, env_vars)

    # Write script to temp file
    script_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="comfy_meta_", delete=False
        ) as f:
            f.write(_METADATA_SCRIPT)
            script_file = f.name

        t0 = time.perf_counter()
        cmd = [str(python), script_file, str(working_dir), package_name]

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
            timeout=timeout,
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

        return payload

    except subprocess.TimeoutExpired:
        print(f"[comfy-env] Metadata scan timed out for {package_name} ({timeout}s)")
        return {"nodes": {}, "display": {}}
    except Exception as e:
        print(f"[comfy-env] Metadata scan error for {package_name}: {e}")
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

    # INPUT_TYPES classmethod returning cached metadata
    @classmethod
    def _input_types(cls, _cached=input_types):
        return _cached
    attrs["INPUT_TYPES"] = _input_types

    # Proxy FUNCTION method — spawns SubprocessWorker for each call
    def _make_proxy(fn, mod, cn, ed, pr, sp, lp, ev, hct):
        def proxy(self, **kwargs):
            from .wrap import _create_worker
            worker = _create_worker(ed, pr, sp, lp, ev, hct)
            try:
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
                return result
            finally:
                worker.shutdown()
        return proxy

    attrs[func_name] = _make_proxy(
        func_name, module_name, class_name,
        env_dir, package_root, sys_path, lib_path, env_vars, health_check_timeout,
    )

    # Create the class
    proxy_cls = type(class_name, (), attrs)
    return proxy_cls
