"""Workspace install: per-env pixi manifests, materialized one at a time.

Discovers every comfy-env.toml under custom_nodes, resolves the bootstrap
torch pin and (when a GPU is present) a cuda-wheels combo, generates one
self-contained pixi.toml per env (packages/toml_generator.py), and runs
`pixi install --manifest-path envs/<name>/pixi.toml` per env -- so a broken
manifest cannot poison another env's install (ADR-0007).

CUDA wheels are deliberately NOT inlined into the manifests: pixi cannot
express no-deps installs and the wheels' upstream Requires-Dist metadata is
wrong for our artifacts, so after `pixi install` they are installed
out-of-band via `uv pip install --no-deps` (the wheel pass inside
install_workspace). This puts them outside pixi.lock -- the "two-system
problem" in the docs -- until Requires-Dist curation in the cuda-wheels
farm makes them resolver-safe and the inline path can revive.

Also handles env stamping (ABI + version + torch pin), install-hash skip of
unchanged envs, and the post-install libomp dedupe.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..config import (
    ComfyEnvConfig,
    load_config,
    CONFIG_FILE_NAME,
    ROOT_CONFIG_FILE_NAME,
)
from ..environment.cache import get_env_name
from .helpers import _make_tee_log, _log_subprocess, _run_streaming, _patch_uv_platform_py, _find_uv


_PYTORCH_PACKAGES = {"torch", "torchvision", "torchaudio"}
_INSTALL_HASH_FILE = "install.hash"


# ---------------------------------------------------------------------------
# Bootstrap probing & node discovery
# ---------------------------------------------------------------------------

def _resolve_workspace_torch(
    log: Callable[[str], None],
) -> Tuple[Optional[str], Optional[str], Optional[str], str, Optional[str]]:
    """Decide (torch_index, cuda_version, cuda_major, python_version, torch_version)
    once for the whole workspace.

    `cuda_version` is the full string (e.g. "12.4"), used by `get_wheel_url`.
    `cuda_major` is just the leading digit (e.g. "12"), used in `[system-requirements]`.
    `python_version` is the bootstrap interpreter's MAJOR.MINOR (e.g. "3.10").
    `torch_version` is the bootstrap's torch.__version__ (public part), or None
    if torch isn't importable from bootstrap (then the comfyui feature stays
    `torch = "*"` and the cuda-wheel picker reads the actual version from the
    materialized template env post-install).

    macOS: (None, None, None, py, torch) -- pixi falls through to default PyPI
    which has osx-arm64/osx-64 torch wheels. Linux/Windows + NVIDIA: cu* index
    + version. Linux/Windows without GPU: (cpu_index, None, None, py, torch).
    """
    from ..detection import (
        get_recommended_cuda_version,
        get_gpu_summary,
        get_bootstrap_python_version,
        get_bootstrap_torch_version,
        get_bootstrap_torch_cuda,
    )
    cpu_index = "https://download.pytorch.org/whl/cpu"
    python_version = get_bootstrap_python_version()
    torch_version = get_bootstrap_torch_version()
    bootstrap_cuda = get_bootstrap_torch_cuda()

    # Portable ComfyUI ships torch+cu128 inside python_embeded, so bootstrap_cuda
    # is "12.8" even on a hosted runner with no NVIDIA driver. Treat it as
    # CPU-only when no GPU is actually present -- otherwise pixi installs cu*
    # wheels into every env and `import torch` later dies with WinError 127
    # (or its Linux equivalent) trying to load shm.dll / libtorch_cuda.so.
    from ..detection.cuda import has_nvidia_gpu
    if bootstrap_cuda and not has_nvidia_gpu():
        log(
            f"[comfy-env] Bootstrap torch is cu{bootstrap_cuda.replace('.', '')[:3]} "
            f"but no NVIDIA GPU detected -- ignoring cuda tag, using CPU index"
        )
        bootstrap_cuda = None

    cu_tag_bootstrap = (
        f"cu{bootstrap_cuda.replace('.', '')}" if bootstrap_cuda else "cpu"
    )
    py_short = python_version.replace(".", "") if python_version else "?"
    if torch_version:
        log(
            f"[comfy-env] Bootstrap interpreter has python {python_version} + "
            f"torch {torch_version} ({cu_tag_bootstrap}). Looking for cuda-only "
            f"wheels matching {cu_tag_bootstrap}/torch{torch_version.rsplit('.', 1)[0]}/cp{py_short}."
        )
    else:
        log(
            f"[comfy-env] Bootstrap interpreter has python {python_version}; "
            f"no torch importable, will rely on cuda-wheels resolver to pick a combo."
        )

    if sys.platform == "darwin":
        # PyTorch's `/whl/cpu/` index is Linux+Windows-only — there are no
        # osx-arm64 / osx-64 wheels there. macOS torch lives on regular PyPI
        # (the wheel is implicitly CPU-only since there's no CUDA on Apple
        # Silicon). Returning None for torch_index lets pixi resolve from the
        # default PyPI index instead of erroring "no matching platform tag".
        return None, None, None, python_version, torch_version

    log(f"[comfy-env] GPU: {get_gpu_summary()}")
    cuda_version = bootstrap_cuda or get_recommended_cuda_version()
    if not cuda_version:
        log("[comfy-env] No CUDA detected -- pinning comfyui torch to CPU index")
        return cpu_index, None, None, python_version, torch_version

    cu_tag = "cu" + cuda_version.replace(".", "")[:3]
    torch_index = f"https://download.pytorch.org/whl/{cu_tag}"
    cuda_major = cuda_version.split(".")[0]
    src = "bootstrap torch" if bootstrap_cuda else "GPU driver"
    log(
        f"[comfy-env] Comfyui feature -> torch {torch_version or '*'} from "
        f"{torch_index} (CUDA {cuda_version} via {src})"
    )
    return torch_index, cuda_version, cuda_major, python_version, torch_version


def _bindable_config_paths(plugin_dir: Path) -> List[Path]:
    """The ONLY locations the runtime binder supports: `nodes/comfy-env.toml`
    and `nodes/<subdir>/comfy-env.toml` (isolation/wrap.py pattern 1 and 2).

    Discovery enumerates exactly these shapes -- deliberately NOT a recursive
    glob. A config anywhere else (pack root, deeper nesting, vendored trees)
    could be materialized but never bound, silently wasting gigabytes; by
    matching discovery to binding, that whole class cannot occur.
    """
    nodes_dir = plugin_dir / "nodes"
    if not nodes_dir.is_dir():
        return []
    out = []
    root_cf = nodes_dir / CONFIG_FILE_NAME
    if root_cf.is_file():
        out.append(root_cf)
    for child in sorted(nodes_dir.iterdir()):
        if child.is_dir() and not child.name.startswith((".", "_")):
            cf = child / CONFIG_FILE_NAME
            if cf.is_file():
                out.append(cf)
    return out


def _discover_node_configs(
    comfyui_dir: Path,
    log: Callable[[str], None] = print,
    failures: Optional[List[Tuple[Path, Exception]]] = None,
) -> List[Tuple[str, Path, Path, ComfyEnvConfig]]:
    """Find bindable comfy-env.toml configs under custom_nodes/ and pair
    with (env_name, plugin_dir, config_path, cfg).

    Only the shapes the runtime can bind are discovered (see
    `_bindable_config_paths`). Duplicate env names across configs are a
    hard error: proceeding would make both configs share one env dir and
    overwrite each other's install hash -- a permanent multi-GB rebuild
    loop with no diagnostic (the 2026-08 review's collision-thrash defect).

    Logs the scan loudly so failed parses don't silently produce an empty result.

    A config that does not parse is SKIPPED, not fatal: this sweep covers
    every pack on the machine, and one pack's typo must not abort another
    pack's install (the per-env isolation rule -- see ADR-0007). The
    calling pack's OWN config is loaded unwrapped by `install()`, so an
    author's own mistake still hard-errors where they can see it. Skipped
    configs are appended to `failures` (when provided) so the caller can
    report them instead of finishing with a silent success.
    """
    custom_nodes = comfyui_dir / "custom_nodes"
    if not custom_nodes.is_dir():
        log(f"[comfy-env] _discover: {custom_nodes} is not a directory")
        return []

    log(f"[comfy-env] _discover: scanning {custom_nodes}")
    out: List[Tuple[str, Path, Path, ComfyEnvConfig]] = []
    seen: Dict[str, Path] = {}
    for plugin_dir in sorted(custom_nodes.iterdir()):
        if not plugin_dir.is_dir():
            continue
        if plugin_dir.name.startswith((".", "_")):
            log(f"[comfy-env] _discover: skip {plugin_dir.name} (dot/underscore prefix)")
            continue
        # Quarantine suffix used by _node_guard and other "disable this node
        # without deleting it" tools. Skip so we don't materialize a pixi env
        # for a node that isn't even being loaded by ComfyUI.
        if plugin_dir.name.endswith((".disabled", "._disabled")):
            log(f"[comfy-env] _discover: skip {plugin_dir.name} (quarantine suffix)")
            continue
        toml_files = _bindable_config_paths(plugin_dir)
        if not toml_files:
            log(f"[comfy-env] _discover: {plugin_dir.name}: no bindable {CONFIG_FILE_NAME}")
            continue
        for cf in toml_files:
            try:
                cfg = load_config(cf)
            except (ValueError, OSError) as e:
                # Data errors only (ValueError covers TOMLDecodeError and
                # every schema rejection). Anything else -- AttributeError,
                # KeyError, TypeError -- is a bug in the config layer, not a
                # bad file, and must propagate instead of being relabelled
                # as the user's fault.
                log(
                    f"[comfy-env] WARNING: skipping {cf} -- "
                    f"{type(e).__name__}: {e}"
                )
                log(
                    f"[comfy-env] WARNING: env for {plugin_dir.name} will NOT "
                    f"be materialized; its isolated nodes will fail to load "
                    f"until this config parses."
                )
                if failures is not None:
                    failures.append((cf, e))
                continue
            env_name = get_env_name(plugin_dir, cf)
            if env_name in seen:
                raise ValueError(
                    f"env name '{env_name}' is derived from BOTH "
                    f"{seen[env_name]} and {cf}. Two configs sharing one env "
                    f"name would share one env directory and permanently "
                    f"rebuild over each other. Rename one of the "
                    f"directories so the derived names differ."
                )
            seen[env_name] = cf
            log(f"[comfy-env] _discover: {plugin_dir.name} -> {env_name} ({cf.relative_to(comfyui_dir)})")
            out.append((env_name, plugin_dir, cf, cfg))
    return out


# ---------------------------------------------------------------------------
# Env identity (v2): two-level skip decision.
#
# Level 1 -- FAST KEY (pure local, no network): a hash of the local inputs
# that could change the derivation: this env's config bytes, the bootstrap
# ABI tag, GPU presence/backend, and the cross-env combo inputs. If the fast
# key matches, the env is stamped tier-1 (not "fallback"), and the env is
# materialized, the run skips with zero network -- this preserves the cheap
# all-clean CI path.
#
# Level 2 -- IDENTITY (derivation OUTPUT): sha256 over the canonical
# generated manifest plus the resolved cuda wheel URLs. Computed only when
# the fast key missed (or the env is on a fallback combo, which a
# later-published wheel can upgrade). Identity match = refresh the hash
# file, do NOT rebuild; mismatch = rebuild. Consequences:
#   - comment/[env_vars] edits cost one derivation, never a rebuild
#     (they change the fast key but not the generated output);
#   - a fallback-combo env re-derives every run and upgrades itself the
#     moment its missing wheel is published (torch pin changes -> identity
#     changes);
#   - GPU-presence flips change the fast key -> full derivation -> the
#     cpu/cu index flip changes the manifest -> rebuild;
#   - comfy-env version bumps no longer force rebuilds (identity depends
#     only on output; the version stays in the stamp for diagnostics).
#
# install.hash format (one entry per line):
#   v2:<identity-sha256>
#   fastkey:<fastkey-sha256>
# A single-line legacy (v1) file is grandfathered: the env is accepted
# as-built once (no surprise multi-GB rebuild on upgrade) and the file is
# rewritten in v2 form so drift tracking starts now.
# ---------------------------------------------------------------------------


def _bootstrap_torch_pin(bootstrap_torch: Optional[str]) -> Optional[str]:
    """major.minor wildcard pin (``==2.10.*``), THE pin rule for every
    builder (install_workspace AND auto_install must agree, or the same env
    means different things depending on who built it).

    major.minor, not the exact patch: must match the granularity of the ABI
    tag that names the env directory (environment/cache.py _abi_tag). A
    finer pin than the key means two installs collide on one directory and
    re-solve each other; a coarser pin would let an env silently satisfy a
    stack it was not built for.
    """
    if not bootstrap_torch:
        return None
    return "==" + ".".join(bootstrap_torch.split(".")[:2]) + ".*"


def _fast_key(
    cf: Path,
    discovered: List[Tuple[str, "Path", "Path", "ComfyEnvConfig"]],
) -> str:
    """Local-only change detector (level 1). See block comment above."""
    from ..detection.cuda import has_nvidia_gpu
    from ..environment.cache import _abi_tag

    h = hashlib.sha256()
    h.update(b"abi:")
    h.update(_abi_tag().encode())
    h.update(b"\ngpu:")
    h.update(b"1" if has_nvidia_gpu() else b"0")
    h.update(b"\n")

    combo_inputs = sorted(
        {pkg for _n, _p, _c, cfg in discovered for pkg in cfg.cuda_packages}
    ) + sorted(
        {cfg.python or "host" for _n, _p, _c, cfg in discovered}
    )
    h.update(b"combo-inputs:")
    h.update(",".join(combo_inputs).encode())
    h.update(b"\ntoml:")
    try:
        h.update(cf.read_bytes())
    except OSError:
        pass
    h.update(b"\n")
    return h.hexdigest()


def _env_identity(manifest: Dict[str, Any], wheel_urls: List[str]) -> str:
    """Derivation-output identity (level 2). Canonical JSON so dict insertion
    order between call paths cannot change the hash."""
    import json

    h = hashlib.sha256()
    h.update(json.dumps(manifest, sort_keys=True, default=str).encode())
    h.update(b"\0")
    h.update("\n".join(sorted(wheel_urls)).encode())
    return "v2:" + h.hexdigest()


def _read_hash_file(hp: Path) -> Tuple[Optional[str], Optional[str], bool]:
    """Returns (identity, fastkey, is_legacy_v1)."""
    try:
        lines = [ln.strip() for ln in
                 hp.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except OSError:
        return None, None, False
    if not lines:
        return None, None, False
    identity = next((ln for ln in lines if ln.startswith("v2:")), None)
    fastkey = next((ln[len("fastkey:"):] for ln in lines
                    if ln.startswith("fastkey:")), None)
    if identity is None and fastkey is None:
        return None, None, True  # single-line v1 format
    return identity, fastkey, False


def _write_hash_file(hp: Path, identity: str, fastkey: str,
                     log: Callable[[str], None]) -> None:
    try:
        hp.write_text(f"{identity}\nfastkey:{fastkey}\n", encoding="utf-8")
        log(f"[comfy-env] Recorded env identity {identity[:15]} -> {hp}")
    except OSError as e:
        log(f"[comfy-env] WARNING: could not write {hp}: {e}")


def _stamp_provenance(env_manifest_dir: Path) -> str:
    """Provenance string from the env stamp ('' if unstamped/unreadable)."""
    import json

    from ..environment.cache import _STAMP_FILE
    try:
        stamp = json.loads(
            (env_manifest_dir / _STAMP_FILE).read_text(encoding="utf-8"))
        return str(stamp.get("provenance") or "")
    except (OSError, ValueError):
        return ""


def _dedupe_envs_libomp(
    workspace_dir: Path,
    discovered: List[Tuple[str, Path, Path, ComfyEnvConfig]],
    log: Callable[[str], None],
) -> None:
    """Run `dedupe_libomp` against each env's site-packages (macOS only).

    Pip wheels often bundle their own libomp.dylib (torch in `torch/lib/`,
    sklearn in `.dylibs/`, pymeshlab in `Frameworks/`) and conda-forge installs
    one at the env root `lib/`. Multiple libomps loaded into the same worker
    process can cause OMP runtime corruption and segfaults; the dedupe symlinks
    every redundant copy to torch's libomp so only one binary is in play.
    """
    if sys.platform != "darwin":
        return
    import glob as _glob
    from ..environment.cache import get_workspace_env_dir
    from ..environment.libomp import dedupe_libomp

    for env_name, _plugin, _cf, _cfg in discovered:
        env_dir = get_workspace_env_dir(workspace_dir, env_name)
        sp_matches = _glob.glob(str(env_dir / "lib" / "python*" / "site-packages"))
        if not sp_matches:
            continue
        try:
            dedupe_libomp(Path(sp_matches[0]))
            log(f"[comfy-env] {env_name}: deduped libomp")
        except Exception as e:
            log(f"[comfy-env] {env_name}: libomp dedupe failed: {e}")


# ---------------------------------------------------------------------------
# Cuda-wheel combo resolution
# ---------------------------------------------------------------------------

def _aggregate_cuda_packages(
    discovered: List[Tuple[str, Path, Path, ComfyEnvConfig]],
) -> List[str]:
    """Union of `cuda_packages` across all discovered node configs, minus the
    workspace-global torch family (those come from the comfyui feature, not
    cuda-wheels)."""
    seen: List[str] = []
    for _en, _pl, _cf, cfg in discovered:
        for p in cfg.cuda_packages:
            if p in _PYTORCH_PACKAGES:
                continue
            if p not in seen:
                seen.append(p)
    return seen


def _cuda_packages_by_python(
    discovered: List[Tuple[str, Path, Path, ComfyEnvConfig]],
    bootstrap_python: str,
) -> Dict[str, List[str]]:
    """Group cuda packages by their env's Python version.

    Returns ``{python_version: [packages]}`` where each package list is
    deduplicated but preserves insertion order.
    """
    by_py: Dict[str, List[str]] = {}
    for _en, _pl, _cf, cfg in discovered:
        py = cfg.python or bootstrap_python
        for p in cfg.cuda_packages:
            if p in _PYTORCH_PACKAGES:
                continue
            pkgs = by_py.setdefault(py, [])
            if p not in pkgs:
                pkgs.append(p)
    return by_py


def _resolve_wheel_combo(
    discovered: List[Tuple[str, Path, Path, ComfyEnvConfig]],
    bootstrap_python: str,
    bootstrap_cuda: Optional[str],
    bootstrap_torch: Optional[str],
    log: Callable[[str], None],
) -> Optional[Tuple[str, str, str, str, str]]:
    """Pick the (python, cuda, torch_match, torch_pin, source) combo for the workspace.

    The returned ``python`` field is the bootstrap Python; per-env Python versions
    are handled at URL-resolution time in ``build_workspace_toml``.

    Strategy:
      1. Try the bootstrap combo (`bootstrap_python` / `bootstrap_cuda` / `bootstrap_torch`).
         If every required cuda-wheel is published for it (across all Python versions
         used by envs), use it. Pin torch to ``==<bootstrap_torch>``.
      2. Else try the known-good fallback for this machine's CPU architecture:
         ``(py, "12.8", "2.8")`` on x86_64, ``(py, "13.0", "2.10")`` on linux
         aarch64 -- torch never shipped an aarch64 wheel for the 2.8 line, so
         the x86 combo is unsatisfiable there, and 12.8/12.9 would leave Thor
         without a kernel image (see FALLBACK_COMBO_AARCH64). Both axes differ,
         so the torch pin follows the chosen combo rather than being fixed.
      3. Else raise.

    Returns None when there's nothing to resolve (no cuda-only packages required,
    or running on macOS/CPU). In that case the caller skips wheel-combo logic and
    leaves torch as `*` in the comfyui feature.
    """
    if not bootstrap_cuda or sys.platform == "darwin":
        return None

    from ..detection.cuda import has_nvidia_gpu
    if not has_nvidia_gpu():
        log("[comfy-env] cuda-wheels: skipping (no NVIDIA GPU detected)")
        return None

    packages = _aggregate_cuda_packages(discovered)
    if not packages:
        return None

    pkgs_by_py = _cuda_packages_by_python(discovered, bootstrap_python)

    from ..packages.cuda_wheels import (
        check_all_wheels_available,
        resolve_fallback_combo,
        CUDA_WHEELS_INDEX,
    )

    log(
        f"[comfy-env] cuda-wheels: {len(packages)} package(s) need a "
        f"matched (cuda, torch, python) combo: {packages}"
    )
    if len(pkgs_by_py) > 1:
        log(
            f"[comfy-env] cuda-wheels: checking across Python versions: "
            f"{', '.join(f'cp{py}' for py in pkgs_by_py)}"
        )

    def _check_all_python_versions(
        torch_ver: str, cuda_ver: str, label: str,
    ) -> Optional[str]:
        """Check wheels for every Python version. Returns first missing pkg or None."""
        for py, py_pkgs in pkgs_by_py.items():
            cp = py.replace(".", "")
            log(
                f"[comfy-env] cuda-wheels {label}: probing "
                f"cu{cuda_ver}/torch{torch_ver}/cp{cp}"
            )
            miss = check_all_wheels_available(
                py_pkgs, torch_ver, cuda_ver, py, log=log,
            )
            if miss is not None:
                return miss
        return None

    # Tier 1: bootstrap combo
    if bootstrap_torch:
        torch_short = ".".join(bootstrap_torch.split(".")[:2])
        miss = _check_all_python_versions(
            torch_short, bootstrap_cuda, "tier 1 (bootstrap)",
        )
        if miss is None:
            log(
                f"[comfy-env] cuda-wheels combo: cu{bootstrap_cuda}/torch{torch_short} "
                f"(bootstrap matches; per-node envs will pin to this)"
            )
            return (
                bootstrap_python,
                bootstrap_cuda,
                torch_short,
                # major.minor, matching tier 2 below and the ABI tag in the env
                # directory name. Pinning the exact patch here made the pin
                # FINER than the key: installs on 2.10.0 and 2.10.2 share
                # `...-torch2-10-...` and rewrote each other's manifest on every
                # alternation. The cuda wheels are stamped `torch2.10` anyway.
                f"=={torch_short}.*",
                "bootstrap",
            )
        log(
            f"[comfy-env] cuda-wheels tier 1 incomplete: `{miss}` not built for "
            f"cu{bootstrap_cuda}+torch{torch_short}; falling back"
        )
    else:
        log(
            "[comfy-env] cuda-wheels: bootstrap torch unknown; skipping tier 1, trying fallback"
        )

    # Tier 2: known-good fallback for this CPU arch (cu128/torch2.8 on x86_64,
    # cu130/torch2.10 on linux aarch64 -- see FALLBACK_COMBO_AARCH64 for why).
    fb_cuda, fb_torch = resolve_fallback_combo()
    miss = _check_all_python_versions(
        fb_torch, fb_cuda, "tier 2 (fallback)",
    )
    if miss is None:
        log(
            f"[comfy-env] cuda-wheels combo: cu{fb_cuda}/torch{fb_torch} "
            f"(fallback; per-node cuda envs will override torch to this combo "
            f"while comfyui keeps bootstrap torch)"
        )
        return (
            bootstrap_python,
            fb_cuda,
            fb_torch,
            f"=={fb_torch}.*",
            "fallback",
        )

    py_summary = ", ".join(f"cp{py}" for py in pkgs_by_py)
    raise RuntimeError(
        f"No cuda-wheels combo covers all required packages.\n"
        f"  packages: {packages}\n"
        f"  python versions: {py_summary}\n"
        f"  tier 1 (bootstrap): cu{bootstrap_cuda}/torch{bootstrap_torch}"
        f" -- missing or untried\n"
        f"  tier 2 (fallback):  cu{fb_cuda}/torch{fb_torch}.*"
        f" -- {miss} missing\n"
        f"Check {CUDA_WHEELS_INDEX}{miss}/ and update the cuda-wheels build matrix."
    )


# ---------------------------------------------------------------------------
# Top-level workspace install
# ---------------------------------------------------------------------------

def install_workspace(
    comfyui_dir: Path,
    log: Callable[[str], None] = print,
    dry_run: bool = False,
) -> Optional[Path]:
    """Generate one ``pixi.toml`` per env under ``<workspace>/envs/<name>/`` and
    run ``pixi install --manifest-path <env>/pixi.toml`` for each.

    Per-env layout (v0.4+): each env's manifest is fully isolated. A parse
    error in one env's pixi.toml cannot poison another env's scan or install.

    No backward compatibility with the v0.3.x single-file workspace layout.
    Users upgrading should ``rm -rf <workspace>/.pixi/`` and
    ``<workspace>/pixi.toml`` before the first install; this function
    re-materializes everything in the new layout.

    Returns the workspace directory on success, None if nothing to install.
    """
    from ..pixi import ensure_pixi, PIXI
    ensure_pixi()
    from ..environment.cache import (
        CE_WORKSPACE_DIR, get_workspace_dir, get_env_manifest_dir,
    )
    from ..packages.toml_generator import (
        build_env_toml, write_env_pixi_toml, resolve_env_cuda_wheel_urls,
    )

    comfyui_dir = Path(comfyui_dir).resolve()
    config_failures: List[Tuple[Path, Exception]] = []
    discovered = _discover_node_configs(
        comfyui_dir, log=log, failures=config_failures)
    if config_failures:
        # Not fatal (one pack must not poison another), but the install is
        # not a clean success either -- say so here rather than letting the
        # skip surface days later as an ImportError with no visible cause.
        log(f"[comfy-env] {len(config_failures)} config(s) did not parse and "
            f"were skipped -- their envs are NOT installed:")
        for cf, err in config_failures:
            log(f"[comfy-env]   {cf}: {type(err).__name__}: {err}")
    if not discovered:
        log("[comfy-env] No custom-node comfy-env.toml files found -- skipping workspace install")
        return None

    workspace_dir = get_workspace_dir(comfyui_dir)

    # No backward compatibility with v0.3.x layout. Per-env manifests under
    # <workspace>/envs/<name>/ only. If a workspace still has
    # <workspace>/pixi.toml + <workspace>/.pixi/envs/<name>/ from an older
    # comfy-env, the user is expected to `rm -rf` those manually; this
    # install will re-materialize at the new layout.
    legacy_workspace = comfyui_dir / CE_WORKSPACE_DIR
    if (legacy_workspace / ".pixi").is_dir():
        log(
            f"[comfy-env] Old workspace at {legacy_workspace} -- safe to delete, "
            f"no longer used"
        )

    # Two-level skip decision (see the identity block comment above
    # `_fast_key`). Level 1 here is pure-local: when every env's fast key
    # matches, is materialized, and is not on a fallback combo, the whole
    # run short-circuits before torch/combo resolution -- the cheap
    # N-installs-per-CI-run path. Everything else goes to level-2
    # DERIVATION, where only an actual identity change rebuilds.
    from ..environment.cache import get_workspace_env_dir as _env_dir_of
    fast_keys: Dict[str, str] = {}
    stored_identity: Dict[str, Optional[str]] = {}
    legacy_v1: Dict[str, bool] = {}
    derive: List[str] = []
    for env_name, _plugin, cf, _cfg in discovered:
        fast_keys[env_name] = _fast_key(cf, discovered)
        env_manifest_dir = get_env_manifest_dir(env_name, comfyui_dir)
        identity, fastkey, legacy = _read_hash_file(
            env_manifest_dir / _INSTALL_HASH_FILE)
        stored_identity[env_name] = identity
        legacy_v1[env_name] = legacy
        materialized = _env_dir_of(workspace_dir, env_name).is_dir()
        on_fallback = _stamp_provenance(env_manifest_dir).endswith(":fallback")
        if (identity is not None and fastkey == fast_keys[env_name]
                and materialized and not on_fallback):
            continue  # clean: zero-network skip
        derive.append(env_name)
    if not dry_run and not derive:
        log(
            f"[comfy-env] All {len(discovered)} env(s) unchanged since last "
            f"successful install -- skipping. Delete an env's install.hash to force."
        )
        return workspace_dir
    if derive and len(derive) < len(discovered):
        log(
            f"[comfy-env] {len(derive)}/{len(discovered)} env(s) need a "
            f"derivation check: {', '.join(derive)} (others clean, skipped)"
        )

    log_path = workspace_dir / "install.log"
    tee_log = _make_tee_log(log, log_path)

    try:
        log = tee_log
        log(f"[comfy-env] Workspace: {workspace_dir}")
        log(f"[comfy-env] ComfyUI: {comfyui_dir}")
        log(f"[comfy-env] Found {len(discovered)} node config(s):")
        for env_name, plugin_dir, cf, cfg in discovered:
            try:
                rel = cf.relative_to(comfyui_dir)
            except ValueError:
                rel = cf
            log(f"  - {env_name} <- {rel} (python={cfg.python or 'host'})")

        (
            torch_index, cuda_version, _cuda_major,
            bootstrap_python, bootstrap_torch,
        ) = _resolve_workspace_torch(log)

        # Pre-validate cuda-wheel availability against the v2 index. May downgrade
        # the workspace's torch/cuda to a known-good combo if the bootstrap one
        # has unpublished wheels. Returns None on macOS / CPU / no cuda-only deps.
        combo = _resolve_wheel_combo(
            discovered, bootstrap_python, cuda_version, bootstrap_torch, log,
        )
        torch_pin: Optional[str] = _bootstrap_torch_pin(bootstrap_torch)
        chosen_torch_index: Optional[str] = None
        chosen_torch_pin_for_override: Optional[str] = None
        if combo is not None:
            chosen_python, chosen_cuda, chosen_torch_short, chosen_torch_pin_for_override, combo_src = combo
            chosen_torch_index = (
                f"https://download.pytorch.org/whl/cu"
                f"{chosen_cuda.replace('.', '')[:3]}"
            )
        else:
            chosen_python = bootstrap_python
            chosen_cuda = cuda_version
            chosen_torch_short = (
                ".".join(bootstrap_torch.split(".")[:2]) if bootstrap_torch else None
            )

        # On Desktop app, requirements.txt is in the app bundle (source dir),
        # not the user data dir. Resolve source dir separately for downstream use.
        from ..environment.cache import find_comfyui_source_dir
        source_dir = find_comfyui_source_dir(comfyui_dir / "custom_nodes")
        if source_dir and source_dir != comfyui_dir:
            log(f"[comfy-env] Desktop app detected: source={source_dir}, data={comfyui_dir}")

        # Level-2 derivation for each candidate env: build the manifest
        # in memory + resolve wheel URLs, then compare the OUTPUT identity
        # with the stored one. Only a real identity change rebuilds;
        # matches (comment edits, version bumps, fallback probe that found
        # nothing new) just refresh the hash file. Grandfather: a legacy
        # v1 hash with a materialized env is accepted as-built once.
        _derive_set = set(derive)
        candidates = [d for d in discovered if d[0] in _derive_set]
        to_install = []
        env_identity: Dict[str, str] = {}
        cuda_urls_by_env: Dict[str, Dict[str, str]] = {}
        for env_name, _plugin, _cf, cfg in candidates:
            env_manifest_dir = get_env_manifest_dir(env_name, comfyui_dir)
            manifest = build_env_toml(
                env_name, cfg,
                torch_index=torch_index,
                bootstrap_python=bootstrap_python,
                torch_pin=torch_pin,
                chosen_torch_index=chosen_torch_index,
                chosen_torch_pin=chosen_torch_pin_for_override,
                chosen_cuda=chosen_cuda if combo is not None else None,
                chosen_torch_short=chosen_torch_short if combo is not None else None,
                log=log,
            )
            urls = resolve_env_cuda_wheel_urls(
                env_name=env_name,
                cfg=cfg,
                bootstrap_python=bootstrap_python,
                chosen_cuda=chosen_cuda if combo is not None else None,
                chosen_torch_short=chosen_torch_short if combo is not None else None,
                log=log,
            )
            identity = _env_identity(manifest, list(urls.values()))
            env_identity[env_name] = identity
            materialized = _env_dir_of(workspace_dir, env_name).is_dir()

            if (not dry_run and materialized
                    and identity == stored_identity[env_name]):
                _write_hash_file(env_manifest_dir / _INSTALL_HASH_FILE,
                                 identity, fast_keys[env_name], log)
                log(f"[comfy-env] {env_name}: derivation unchanged -- skipping")
                continue
            if (not dry_run and materialized and legacy_v1[env_name]
                    and stored_identity[env_name] is None):
                _write_hash_file(env_manifest_dir / _INSTALL_HASH_FILE,
                                 identity, fast_keys[env_name], log)
                log(
                    f"[comfy-env] {env_name}: legacy install.hash grandfathered "
                    f"-- env accepted as-built, identity tracking starts now"
                )
                continue

            to_install.append((env_name, _plugin, _cf, cfg))
            if urls:
                cuda_urls_by_env[env_name] = urls
                log(
                    f"[comfy-env] {env_name}: cuda-wheels deferred for post-pixi "
                    f"install ({', '.join(urls.keys())})"
                )

        # Emit one pixi.toml per genuinely-stale env. Clean envs keep their
        # manifest untouched (identical content, but no mtime churn and no
        # clobbering of another install's manifest).
        log(f"[comfy-env] Writing {len(to_install)} per-env manifest(s):")
        for env_name, _plugin, _cf, cfg in to_install:
            env_manifest_dir = get_env_manifest_dir(env_name, comfyui_dir)
            write_env_pixi_toml(
                env_manifest_dir=env_manifest_dir,
                env_name=env_name,
                cfg=cfg,
                torch_index=torch_index,
                bootstrap_python=bootstrap_python,
                torch_pin=torch_pin,
                chosen_torch_index=chosen_torch_index,
                chosen_torch_pin=chosen_torch_pin_for_override,
                chosen_cuda=chosen_cuda if combo is not None else None,
                chosen_torch_short=chosen_torch_short if combo is not None else None,
                log=log,
            )
            log(f"  - {env_name}: {env_manifest_dir / 'pixi.toml'}")

        if dry_run:
            log("[comfy-env] dry_run -- skipping `pixi install`")
            return workspace_dir

        _patch_uv_platform_py(log)

        pixi_env = dict(os.environ)
        pixi_env["UV_PYTHON_INSTALL_DIR"] = str(workspace_dir / "_no_python")
        pixi_env["UV_PYTHON_PREFERENCE"] = "only-system"
        pixi_env["PIXI_NO_PROGRESS"] = "true"

        # Install each stale env independently. One failure doesn't stop the
        # others by default (we collect and raise at the end), so users see all
        # diagnostics from one run instead of having to re-trigger after each fix.
        log(f"[comfy-env] Installing {len(to_install)} environment(s):")
        for env_name, _plugin, _cf, cfg in to_install:
            py = cfg.python or "host"
            deps = list(cfg.pixi_passthrough.get("pypi-dependencies", {}).keys())
            cuda = cfg.cuda_packages
            parts = [f"python={py}"]
            if deps:
                parts.append(f"{len(deps)} pypi deps")
            if cuda:
                parts.append(f"{len(cuda)} cuda wheels")
            log(f"  - {env_name} ({', '.join(parts)})")

        install_failures: List[str] = []
        for env_name, _plugin, _cf, _cfg in to_install:
            env_manifest_dir = get_env_manifest_dir(env_name, comfyui_dir)
            env_manifest = env_manifest_dir / "pixi.toml"
            log(f"[comfy-env] Running `pixi install --manifest-path {env_manifest}` ...")
            result = _run_streaming(
                [PIXI, "install", "--manifest-path", str(env_manifest)],
                log=log, cwd=env_manifest_dir, env=pixi_env,
            )
            _log_subprocess(log, result, f"pixi install ({env_name})")
            if result.returncode != 0:
                install_failures.append(env_name)
                log(
                    f"[comfy-env] {env_name}: pixi install FAILED (exit {result.returncode}). "
                    f"Continuing with remaining envs."
                )

        if install_failures:
            raise RuntimeError(
                f"pixi install failed for {len(install_failures)} env(s): "
                f"{', '.join(install_failures)}"
            )

        # Report envs on disk that no node declares -- DO NOT prune. A user may
        # have multiple ComfyUI installs sharing this workspace, or a node's
        # `comfy-env.toml` may be transiently missing (mid-clone, partial checkout).
        new_envs_root = workspace_dir / "envs"
        if new_envs_root.is_dir():
            # Compare DIRECTORY names, not logical env names: directories carry
            # the ABI tag (`<name>-py313-torch2-10-cu128`), so matching the bare
            # name here would report every live env as orphaned.
            from ..environment.cache import _env_dir_name
            current_names = {
                _env_dir_name(env_name) for env_name, _, _, _ in discovered
            }
            for d in sorted(new_envs_root.iterdir()):
                if not d.is_dir() or d.name in current_names:
                    continue
                log(
                    f"[comfy-env] Note: env `{d.name}` is on disk but no node "
                    f"declares it in this run. Leaving as-is. "
                    f"Remove via `rm -rf {d}` if intended."
                )

        # CUDA-only wheels installed with --no-deps after pixi (pixi's resolver
        # can't suppress their declared dependencies, which are often wrong for
        # custom-built wheels).
        if cuda_urls_by_env:
            uv_path = _find_uv()
            from ..environment.cache import get_workspace_env_dir
            for env_name, pkg_urls in cuda_urls_by_env.items():
                env_dir = get_workspace_env_dir(workspace_dir, env_name)
                if not env_dir.is_dir():
                    log(f"[comfy-env] Warning: env dir {env_dir} not found, skipping cuda-wheels")
                    continue
                wheel_urls = list(pkg_urls.values())
                log(
                    f"[comfy-env] {env_name}: installing cuda-wheels with --no-deps "
                    f"({', '.join(pkg_urls.keys())})"
                )
                env_python = env_dir / "bin" / "python"
                if not env_python.exists():
                    env_python = env_dir / "python.exe"
                if not env_python.exists():
                    env_python = env_dir / "Scripts" / "python.exe"
                uv_result = _run_streaming(
                    [uv_path, "pip", "install", "--no-deps", "--no-cache",
                     "--python", str(env_python)] + wheel_urls,
                    log=log, cwd=workspace_dir, env=pixi_env,
                )
                _log_subprocess(log, uv_result, f"uv pip install --no-deps ({env_name})")
                if uv_result.returncode != 0:
                    raise RuntimeError(
                        f"uv pip install --no-deps failed for {env_name}:\n"
                        f"stderr: {uv_result.stderr}\nstdout: {uv_result.stdout}"
                    )

        # Dedupe libomp.dylib copies in each env's site-packages (macOS only).
        _dedupe_envs_libomp(workspace_dir, to_install, log)

        # Stamp each freshly-built env with what it was built from/against.
        # `_find_env_dir` validates this at bind time: today an env is trusted
        # because its directory exists, which is how a foreign-stack env gets
        # silently loaded into torch's private multiprocessing ABI with no
        # handshake anywhere downstream (see _ipc_parent.py). The stamp turns
        # that into a loud mismatch instead.
        from ..environment.cache import write_env_stamp
        for env_name, _plugin, cf, _cfg in to_install:
            # Provenance carries the combo tier for cuda envs: envs stamped
            # ":fallback" are re-derived on every install run so they upgrade
            # themselves the moment their missing wheel is published.
            if combo is not None and _cfg.cuda_packages:
                prov = f"install_workspace:{combo_src}"
                stamp_pin = chosen_torch_pin_for_override
            else:
                prov = "install_workspace"
                stamp_pin = torch_pin
            write_env_stamp(
                get_env_manifest_dir(env_name, comfyui_dir),
                torch_pin=stamp_pin,
                provenance=prov,
                log=log,
            )

        # Record each env's derivation identity + fast key IN ITS OWN
        # DIRECTORY so subsequent runs skip it individually. Only written
        # after pixi install + post-steps all succeed -- a failed install
        # leaves no hash and forces a retry.
        for env_name, _plugin, cf, _cfg in to_install:
            _write_hash_file(
                get_env_manifest_dir(env_name, comfyui_dir) / _INSTALL_HASH_FILE,
                env_identity[env_name], fast_keys[env_name], log)
        try:
            legacy_hash = workspace_dir / _INSTALL_HASH_FILE
            if legacy_hash.is_file():
                legacy_hash.unlink()
                log(f"[comfy-env] Removed legacy workspace-level {legacy_hash}")
        except OSError:
            pass

        log(f"[comfy-env] Install log: {log_path}")
        return workspace_dir
    finally:
        try:
            tee_log.close()
        except Exception:
            pass
