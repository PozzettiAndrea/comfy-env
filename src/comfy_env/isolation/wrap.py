"""Process isolation for ComfyUI nodes - wraps FUNCTION methods to run in isolated env."""

import inspect
import os
import sys
from pathlib import Path
from typing import Optional

from ..config import DEFAULT_HEALTH_CHECK_TIMEOUT
from ..debug import WORKER as _DBG_WORKER
# Worker pool (state, lifecycle, VRAM/progress callbacks, route proxying,
# stale-patcher invariant) extracted to isolation/pool.py so metadata.py
# imports it downward. wrap.register_nodes uses these two at startup:
from .pool import _cleanup_stale_workers, _register_proxy_routes  # noqa: F401


def _log(msg: str) -> None:
    """Print to stderr with flush -- survives process crashes."""
    print(msg, file=sys.stderr, flush=True)




# Launch-env construction (build_isolation_env, _get_env_paths, platform
# builders) moved to isolation/subenv.py -- a leaf, so subprocess.py and
# metadata.py import it DOWNWARD instead of reaching up into this file.
from .subenv import _get_env_paths

def _find_env_dir(node_dir: Path, config_path: Optional[Path] = None) -> Optional[Path]:
    """Resolve the pixi env dir for a node config in the workspace model.

    Looks up `<comfyui_dir>/.ce/.pixi/envs/<env_name>` based on the node's plugin
    root and config path. Returns None (with a log line) when the env hasn't been
    materialized — that signals to register_nodes() that this node should fall
    back to plain in-process import.
    """
    from ..environment.cache import get_env_name, get_workspace_env_dir

    node_dir = Path(node_dir)

    # Locate plugin root: walk up to the directory whose parent is `custom_nodes/`.
    # abspath, NOT resolve(): a pack installed as a junction/symlink
    # (custom_nodes/Pack -> elsewhere/Pack) must keep its custom_nodes-side
    # identity. resolve() follows the link, the walk then never sees
    # `custom_nodes`, and the env name degrades to the subdir ("nodes") --
    # while install enumerates custom_nodes/ UNRESOLVED and materializes the
    # real name, so the two sides can never agree (#8). abspath absolutizes
    # lexically without following links, on Windows junctions too.
    plugin_dir = node_dir
    for parent in Path(os.path.abspath(node_dir)).parents:
        if parent.parent and parent.parent.name == "custom_nodes":
            plugin_dir = parent
            break
    else:
        _log(
            f"[comfy-env] plugin root not found: no `custom_nodes` ancestor above "
            f"{node_dir}; deriving the env name from {node_dir.name!r} instead. "
            f"If this pack lives under custom_nodes/ (possibly behind a "
            f"junction/symlink), this is a bug -- please report it."
        )

    # Find ComfyUI base
    try:
        from ..environment.cache import find_comfyui_dir_from_node as get_comfyui_dir
        comfyui_dir = get_comfyui_dir(node_dir)
    except Exception:
        comfyui_dir = None

    if comfyui_dir is None:
        _log(
            f"[comfy-env] isolation env not found: ComfyUI base could not be located "
            f"from {node_dir}; using in-process import"
        )
        return None

    if config_path is None:
        # Only comfy-env.toml names an env. The root file was once a
        # fallback candidate here -- an identity for an env that install
        # never creates (2026-08 review wart, removed).
        if (node_dir / "comfy-env.toml").exists():
            config_path = node_dir / "comfy-env.toml"
        if config_path is None:
            _log(
                f"[comfy-env] isolation env not found: no comfy-env.toml in {node_dir}; "
                f"using in-process import"
            )
            return None

    env_name = get_env_name(plugin_dir, config_path)
    env_dir = get_workspace_env_dir(comfyui_dir, env_name)
    if env_dir.exists():
        # Never bind on directory existence alone. The env's worker shares
        # tensors with this process over torch's PRIVATE multiprocessing ABI
        # (reduce_storage / rebuild_cuda_tensor) -- there is no version
        # handshake downstream, so a foreign-stack env fails as DLL-load
        # chaos (ERROR_PROC_NOT_FOUND on shm.dll) or worse, not as a clean
        # error. The stamp written at install time is checked here instead.
        from ..environment.cache import validate_env_stamp
        ok, reason = validate_env_stamp(env_dir.parent.parent.parent)
        if not ok:
            _log(
                f"[comfy-env] REFUSING env `{env_name}` at {env_dir}: {reason}. "
                f"Re-run `comfy-env install` from this ComfyUI to materialize "
                f"one for this stack; falling back to in-process import."
            )
        else:
            return env_dir.resolve() if sys.platform == "win32" else env_dir

    # Envs are materialized ONLY by install() -- there is one builder
    # (install/workspace.py). A lazy second path used to exist behind
    # COMFY_ENV_AUTO_INSTALL; it was removed in 0.4.25 because it diverged
    # from install_workspace in ways no seal could detect (it skipped the
    # macOS libomp dedupe and uv's python-preference pinning, both of which
    # left a permanently-wrong env that every later `comfy-env install`
    # then SKIPPED as identity-matching).
    #
    # The path shown here must work from the user's cwd: bare
    # `comfy-env install` resolves the config from the CURRENT directory,
    # so it fails from the ComfyUI root. --dir is the spelling that works
    # from anywhere.
    _log(
        f"[comfy-env] isolation env not found at {env_dir}: pixi has not "
        f"materialized `{env_name}`; using in-process import. Build it with "
        f"`comfy-env install --dir {plugin_dir}` "
        f"(or `python -m comfy_env.cli install --dir {plugin_dir}` if the "
        f"console script is not on PATH)."
    )
    return None



def _warn_accel_violations(meta: dict, label: str) -> None:
    """Surface top-level accelerator imports observed by the metadata scan.

    A top-level import of a [cuda] package makes the whole scan die on
    machines where the package isn't installed -- every node in the env
    silently vanishes. The scan observes sys.modules AFTER import (nothing
    has executed yet), so presence is proof, not prediction.
    """
    violations = meta.get("accel_import_violations") or []
    if violations:
        _log(
            f"[comfy-env] WARNING: {label}: accelerator package(s) "
            f"{', '.join(violations)} imported at module top level. "
            f"Accelerator packages must be imported lazily inside the nodes "
            f"that declare them (ACCELERATOR=...), or this pack's nodes will "
            f"ALL fail to load on machines without them."
        )


def register_nodes(nodes_package: str = "nodes") -> tuple:
    """Discover and register all nodes -- main-process and isolation.

    Usage in custom node __init__.py:
        from comfy_env import register_nodes
        NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = register_nodes()

    For main-process dirs (no comfy-env.toml): imports normally.
    For isolation dirs (comfy-env.toml + _env_*): subprocess metadata scan + proxy classes.

    Args:
        nodes_package: Name of the nodes subpackage (default: "nodes")

    Returns:
        (NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)
    """
    import importlib
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from .metadata import fetch_metadata, build_proxy_class

    # Log version
    try:
        from importlib.metadata import version as get_version
        _log(f"[comfy-env] Version: {get_version('comfy-env')}")
    except Exception:
        pass

    _cleanup_stale_workers()

    # Get caller info
    frame = inspect.stack()[1]
    caller_module = inspect.getmodule(frame.frame)
    # abspath, not resolve(): keep the caller's custom_nodes-side path when the
    # pack is a junction/symlink -- see the plugin-root walk in _find_env_dir.
    pkg_dir = Path(os.path.abspath(frame.filename)).parent
    caller_pkg_name = caller_module.__name__ if caller_module else None

    _log(f"[comfy-env] register_nodes: {pkg_dir.name}")

    nodes_dir = pkg_dir / nodes_package
    if not nodes_dir.is_dir():
        _log(f"[comfy-env] No '{nodes_package}/' directory in {pkg_dir}")
        return {}, {}

    # Discover isolation configs. Discovery matches the binder EXACTLY:
    # <nodes_package>/comfy-env.toml and <nodes_package>/<subdir>/ only --
    # the two shapes pattern 1/2 below can bind. Deliberately not a
    # recursive glob: a config anywhere else could be scanned but never
    # bound (and rglob walked .git/assets on every boot).
    isolation_envs = {}  # {resolved_dir: env_config}
    config_files = []
    _root_cf = nodes_dir / "comfy-env.toml"
    if _root_cf.is_file():
        config_files.append(_root_cf)
    for _child in sorted(nodes_dir.iterdir()):
        if _child.is_dir() and not _child.name.startswith((".", "_")):
            _cf = _child / "comfy-env.toml"
            if _cf.is_file():
                config_files.append(_cf)

    from ..environment.cache import find_comfyui_source_dir
    comfyui_base = find_comfyui_source_dir(pkg_dir)
    if comfyui_base:
        comfyui_base = str(comfyui_base)
        _log(f"[comfy-env] ComfyUI source dir: {comfyui_base}")
    else:
        _log("[comfy-env] ComfyUI source dir not found")

    # Root config (comfy-env-root.toml): loaded once, used for [types]
    # here and [settings] below.
    # A config that exists but does not parse is a correctness fault, not an
    # availability one, so it fails LOUDLY (ADR-0008 as amended). Failing
    # here fails this pack only: ComfyUI's load_custom_node wraps the import,
    # logs the traceback and boots without us -- "never break ComfyUI
    # startup" is satisfied by the host's own per-pack isolation, not by us
    # swallowing the error. This block used to `except Exception` into
    # root_cfg=None, which silently emptied _custom_sockets below and so
    # disabled the very [types] validation the next comment calls LOUD.
    root_cfg = None
    try:
        from ..config import discover_config
        root_cfg = discover_config(pkg_dir, root=True)
    except (ValueError, OSError) as e:
        raise RuntimeError(
            f"[comfy-env] {pkg_dir.name}: the root config exists but could "
            f"not be read -- {type(e).__name__}: {e}. Fix it (or delete it "
            f"if this pack no longer needs one); comfy-env will not run a "
            f"pack from a half-read config, because [settings] and [types] "
            f"would silently fall back to defaults."
        ) from e

    # [types] (ADR-0015): "custom" sockets require <pack>/serialization.py.
    # Load it parent-side by file path (mangled module name -- no
    # sys.modules collisions between packs) and validate LOUDLY: a
    # declared-custom pack whose serializer file is missing, does not
    # import, or registers nothing is a broken contract, not a warning.
    _serializer_file = None
    _custom_sockets = sorted(
        s for s, mode in (root_cfg.types if root_cfg else {}).items()
        if mode == "custom")
    if _custom_sockets:
        _ser_path = pkg_dir / "serialization.py"
        if not _ser_path.is_file():
            raise ValueError(
                f"{pkg_dir.name}: [types] declares custom socket(s) "
                f"{', '.join(_custom_sockets)} but {_ser_path} does not "
                f"exist.")
        from .workers._ipc_shared import (
            load_serializer_files,
            registration_calls,
        )
        # Count register_serializer CALLS, not distinct registered types. Two
        # packs sharing a wire tag on purpose (ADR-0015 type identity: TRIMESH
        # across 3D-Pack and GeometryPack) register the same class name, so
        # whichever loads second left the key count unchanged and was failed
        # here for "registered no serializers" -- and which one that is depends
        # on ComfyUI's unsorted os.listdir walk of custom_nodes.
        _before = registration_calls()
        _available, _executed = load_serializer_files(str(_ser_path), log=_log)
        if _available and _executed and registration_calls() == _before:
            raise ValueError(
                f"{pkg_dir.name}: [types] declares custom socket(s) "
                f"{', '.join(_custom_sockets)} but serialization.py "
                f"registered no serializers. Either it failed to import "
                f"(top-level imports must be stdlib/numpy/comfy_env only "
                f"-- heavy deps go inside the functions) or it never "
                f"calls register_serializer().")
        if not _available:
            raise ValueError(
                f"{pkg_dir.name}: [types] declares custom socket(s) "
                f"{', '.join(_custom_sockets)} but {_ser_path} failed to "
                f"import -- top-level imports must be stdlib/numpy/comfy_env "
                f"only; heavy deps go inside the functions.")
        _serializer_file = str(_ser_path)

    for cf in config_files:
        env_dir = _find_env_dir(cf.parent, config_path=cf)
        if not env_dir:
            continue
        sp = _get_env_paths(env_dir)
        if not sp:
            continue

        # ONE parser (the config layer) -- this block once tomli.load'ed the
        # file and re-implemented env_vars/options/cuda normalization by
        # hand; duplicated contracts drift (the .log/.txt faulthandler split
        # was the same disease).
        env_vars = {}
        health_check_timeout = DEFAULT_HEALTH_CHECK_TIMEOUT
        try:
            from ..config import load_config
            cfg = load_config(cf)
            env_vars = dict(cfg.env_vars)
            health_check_timeout = cfg.options["health_check_timeout"]
            # Feed the declared accelerator packages to the metadata scan so
            # it can detect top-level accelerator imports
            # (accel_import_violations in the scan payload).
            if cfg.cuda_packages:
                env_vars["COMFY_ENV_ACCEL_PKGS"] = ",".join(
                    str(p) for p in cfg.cuda_packages)
        except (ValueError, OSError) as e:
            # Same rule as the root config above: binding an env from a
            # config we could not read means env_vars silently empty and
            # health_check_timeout silently default -- a worker configured
            # differently from what the author wrote. Fail the pack.
            raise RuntimeError(
                f"[comfy-env] {pkg_dir.name}: {cf} exists but could not be "
                f"read -- {type(e).__name__}: {e}. Fix the config; comfy-env "
                f"will not bind an isolated env from a half-read file."
            ) from e
        # [types] custom serializers (root-level, ADR-0015): workers load
        # the same serialization.py by file path at startup.
        if _serializer_file:
            env_vars["COMFY_ENV_SERIALIZER_FILES"] = _serializer_file
        if comfyui_base:
            env_vars["COMFYUI_BASE"] = str(comfyui_base)
        # On Desktop app, folder_paths needs the user data dir for input/output/models
        try:
            import folder_paths
            user_data = folder_paths.base_path
            if user_data and str(user_data) != str(comfyui_base):
                env_vars["COMFYUI_USER_DIR"] = str(user_data)
        except ImportError:
            pass

        package_root = pkg_dir
        isolation_envs[cf.parent.resolve()] = {
            "dir": cf.parent,
            "env_dir": env_dir,
            "sp": sp,
            "env_vars": env_vars,
            "health_check_timeout": health_check_timeout,
            "package_root": package_root,
        }

    if _DBG_WORKER:
        _log(f"[comfy-env] Found {len(isolation_envs)} isolation env(s)")

    all_mappings = {}
    all_display = {}
    import_failures = []  # (source, formatted traceback)

    # Worker reentry guard: inside an isolation worker, never isolate again.
    enabled = os.environ.get("COMFYUI_ISOLATION_WORKER") != "1"

    # ==================================================================
    # Discover and import node sources
    # ==================================================================
    # Two patterns (mutually exclusive):
    #   1. nodes/ itself is the source (isolation or direct)
    #   2. Subdirectories of nodes/ are individual sources
    # Check root first; fall through to subdirs if root yields nothing.

    root_resolved = nodes_dir.resolve()

    # --- Pattern 1: nodes/ root ---
    if root_resolved in isolation_envs and enabled:
        # Isolation env at root -- subprocess scan
        env = isolation_envs[root_resolved]
        _log(f"[comfy-env] Scanning {nodes_package} metadata (isolation root)...")
        try:
            import time as _time
            _t0 = _time.perf_counter()
            root_meta = fetch_metadata(
                env_dir=env["env_dir"],
                package_name=nodes_package,
                working_dir=pkg_dir,
                env_vars=env["env_vars"],
            )
            root_nodes = root_meta.get("nodes", {})
            _log(f"[comfy-env] Scanned {nodes_package} root: {len(root_nodes)} nodes ({_time.perf_counter()-_t0:.1f}s)")
            _warn_accel_violations(root_meta, nodes_package)
            root_display = root_meta.get("display", {})

            package_root = env["package_root"]
            sys_path_list = [str(env["sp"]), str(package_root)]
            # Don't add host site-packages to sys_path -- torch is symlinked
            # into pixi env by metadata.py/subprocess.py. Adding host sp leaks
            # pip C-extension packages (scipy, numpy) that crash on macOS.

            for name, meta in root_nodes.items():
                all_mappings[name] = build_proxy_class(
                    node_name=name,
                    meta=meta,
                    env_dir=env["env_dir"],
                    package_root=package_root,
                    sys_path=sys_path_list,
                    env_vars=env["env_vars"],
                    health_check_timeout=env["health_check_timeout"],
                )
            all_display.update(root_display)
            # Register proxy routes for isolation API endpoints
            root_routes = root_meta.get("routes", [])
            if root_routes:
                _register_proxy_routes(
                    root_routes, env["env_dir"], package_root,
                    sys_path_list, env["env_vars"],
                    env["health_check_timeout"],
                )
            _log(f"[comfy-env] Imported {nodes_package} root: {len(root_nodes)} nodes (isolation)")
        except Exception as e:
            _log(f"[comfy-env] Failed to scan {nodes_package} root: {e}")

    elif root_resolved not in isolation_envs or not enabled:
        # No isolation at root -- try direct import
        _log(f"[comfy-env] Importing {nodes_package} (root)...")
        try:
            mod = importlib.import_module(f".{nodes_package}", package=caller_pkg_name)
            mappings = getattr(mod, "NODE_CLASS_MAPPINGS", {})
            display = getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", {})
            all_mappings.update(mappings)
            all_display.update(display)
            _log(f"[comfy-env] Imported {nodes_package} root: {len(mappings)} nodes")
        except ModuleNotFoundError as e:
            # The nodes package itself not existing is ABSENCE, not failure
            # (types-only packs have no nodes package at all). A missing
            # module INSIDE an existing package is a real import failure.
            if e.name in (f"{caller_pkg_name}.{nodes_package}", nodes_package):
                _log(f"[comfy-env] No {nodes_package} package to import (root)")
            else:
                import traceback
                import_failures.append((f"{nodes_package} (root)", traceback.format_exc()))
        except Exception:
            import traceback
            import_failures.append((f"{nodes_package} (root)", traceback.format_exc()))

    # --- Pattern 2: subdirectories (only if root yielded nothing) ---
    # Skip if root was an isolation env (even if scan returned 0 nodes) -- subdirs
    # are part of that isolation env and must not be direct-imported.
    if not all_mappings and (root_resolved not in isolation_envs or not enabled):
        main_dirs = []
        isolation_dirs = []

        for subdir in sorted(nodes_dir.iterdir()):
            if not subdir.is_dir():
                continue
            if not (subdir / "__init__.py").exists():
                continue
            if subdir.name.startswith("_") or subdir.name.startswith("."):
                continue

            if subdir.resolve() in isolation_envs and enabled:
                isolation_dirs.append(subdir)
            else:
                main_dirs.append(subdir)

        # Import main-process dirs normally
        for subdir in main_dirs:
            module_path = f".{nodes_package}.{subdir.name}"
            _log(f"[comfy-env] Importing {subdir.name}...")
            try:
                mod = importlib.import_module(module_path, package=caller_pkg_name)
                mappings = getattr(mod, "NODE_CLASS_MAPPINGS", {})
                display = getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", {})
                all_mappings.update(mappings)
                all_display.update(display)
                _log(f"[comfy-env] Imported {subdir.name}: {len(mappings)} nodes")
            except ModuleNotFoundError as e:
                if e.name == f"{caller_pkg_name}{module_path}":
                    _log(f"[comfy-env] No importable package at {module_path}")
                else:
                    import traceback
                    import_failures.append((module_path, traceback.format_exc()))
            except Exception:
                import traceback
                import_failures.append((module_path, traceback.format_exc()))

        # Subprocess-scan isolation dirs (in parallel)
        if enabled and isolation_dirs:
            def _scan_isolation(subdir):
                env = isolation_envs[subdir.resolve()]
                package_name = f"{nodes_package}.{subdir.name}"
                _log(f"[comfy-env] Scanning {subdir.name} metadata...")
                import time
                t0 = time.perf_counter()
                meta = fetch_metadata(
                    env_dir=env["env_dir"],
                    package_name=package_name,
                    working_dir=pkg_dir,
                    env_vars=env["env_vars"],
                )
                n = len(meta.get("nodes", {}))
                _log(f"[comfy-env] Scanned {subdir.name}: {n} nodes ({time.perf_counter()-t0:.1f}s)")
                _warn_accel_violations(meta, package_name)
                return subdir, env, meta

            with ThreadPoolExecutor(max_workers=len(isolation_dirs)) as executor:
                futures = {executor.submit(_scan_isolation, d): d for d in isolation_dirs}
                for future in as_completed(futures):
                    try:
                        subdir, env, metadata = future.result()
                    except Exception as e:
                        subdir = futures[future]
                        _log(f"[comfy-env] Metadata scan failed for {subdir.name}: {e}")
                        continue

                    nodes_meta = metadata.get("nodes", {})
                    display = metadata.get("display", {})

                    package_root = env["package_root"]
                    sys_path_list = [str(env["sp"]), str(package_root)]
                    # Don't add host site-packages -- torch is symlinked into pixi env

                    for name, meta in nodes_meta.items():
                        all_mappings[name] = build_proxy_class(
                            node_name=name,
                            meta=meta,
                            env_dir=env["env_dir"],
                            package_root=package_root,
                            sys_path=sys_path_list,
                            env_vars=env["env_vars"],
                            health_check_timeout=env["health_check_timeout"],
                        )

                    all_display.update(display)
                    # Register proxy routes for isolation API endpoints
                    sub_routes = metadata.get("routes", [])
                    if sub_routes:
                        _register_proxy_routes(
                            sub_routes, env["env_dir"], package_root,
                            sys_path_list, env["env_vars"],
                            env["health_check_timeout"],
                        )
                    if nodes_meta:
                        _log(f"[comfy-env] Registered {len(nodes_meta)} isolation nodes from {subdir.name}")

    # Report skipped isolation dirs (no _env_* installed)
    for cf in config_files:
        if cf.parent.resolve() not in isolation_envs:
            env_dir = _find_env_dir(cf.parent)
            if not env_dir:
                _log(f"[comfy-env] No env for {cf.parent.name} -- run 'comfy-env install'")

    # An in-process import failure means every node in that source silently
    # vanishes -- historically the top "No Nodes Found" ticket class. Print
    # the FULL traceback (a one-line str(e) hides the actual cause), and when
    # NOTHING registered at all, raise: the exception propagates to ComfyUI's
    # load_custom_node, which prints it and marks this pack IMPORT FAILED in
    # the startup summary -- visible by construction instead of a green load
    # with zero nodes.
    if import_failures:
        for _src, _tb in import_failures:
            _log(f"[comfy-env] ERROR: failed to import {_src}:\n{_tb}")
        if not all_mappings:
            raise ImportError(
                f"[comfy-env] all {len(import_failures)} node source(s) failed "
                f"to import; first failure ({import_failures[0][0]}) above"
            )
        _log(
            f"[comfy-env] WARNING: {len(import_failures)} node source(s) failed "
            f"to import ({', '.join(s for s, _ in import_failures)}); their "
            f"nodes are missing from this run"
        )

    _log(f"[comfy-env] Registered {len(all_mappings)} total nodes")

    # ADR-0012 startup summary: accelerator nodes this machine can't serve
    # are registered (workflows load) but hidden from the node menu.
    unavailable = [n for n, c in all_mappings.items()
                   if getattr(c, "_comfy_env_unavailable", None)]
    if unavailable:
        # _comfy_env_accelerator is a LIST -- flatten across nodes so the
        # summary reads "cuda/mps", not "['cuda']/['cuda', 'mps']".
        accels = sorted({a for n in unavailable
                         for a in (getattr(all_mappings[n],
                                           "_comfy_env_accelerator", None) or ["?"])})
        _log(f"[comfy-env] WARNING: {len(unavailable)} node(s) require "
             f"{'/'.join(accels)} but no such accelerator is present on this "
             f"machine -- registered but hidden from the node menu: "
             f"{', '.join(unavailable[:8])}"
             + (" ..." if len(unavailable) > 8 else ""))

    return all_mappings, all_display
