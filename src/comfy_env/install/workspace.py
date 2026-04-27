"""Workspace install: generate `<comfyui>/.ce/pixi.toml` and run `pixi install --all`.

Handles bootstrap torch resolution, cuda-wheel combo selection, and the post-install
libomp dedupe. The cuda-wheel installation itself is now inlined into `pixi.toml` as
`pypi-dependencies.{url=...}` entries by `packages/toml_generator.py`, so this module
no longer pip-installs them out-of-band — the `_install_cuda_wheels` function below
is dead code retained only as a reference until removed in a follow-up cleanup.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from ..config import (
    ComfyEnvConfig,
    load_config,
    CONFIG_FILE_NAME,
    ROOT_CONFIG_FILE_NAME,
)
from ..environment.cache import get_env_name
from .helpers import _make_tee_log, _log_subprocess, _run_streaming, _patch_uv_platform_py


_PYTORCH_PACKAGES = {"torch", "torchvision", "torchaudio"}


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

    macOS: (cpu_index, None, None, py, torch). Linux/Windows + NVIDIA: cu* index +
    version. Linux/Windows without GPU: (cpu_index, None, None, py, torch).
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
        return cpu_index, None, None, python_version, torch_version

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


def _discover_node_configs(comfyui_dir: Path) -> List[Tuple[str, Path, Path, ComfyEnvConfig]]:
    """Find every comfy-env.toml under custom_nodes/ and pair with (env_name, plugin_dir, config_path, cfg)."""
    custom_nodes = comfyui_dir / "custom_nodes"
    if not custom_nodes.is_dir():
        return []

    out: List[Tuple[str, Path, Path, ComfyEnvConfig]] = []
    for plugin_dir in sorted(custom_nodes.iterdir()):
        if not plugin_dir.is_dir() or plugin_dir.name.startswith((".", "_")):
            continue
        for cf in sorted(plugin_dir.rglob(CONFIG_FILE_NAME)):
            if cf.name == ROOT_CONFIG_FILE_NAME:
                continue
            try:
                cfg = load_config(cf)
            except Exception:
                continue
            env_name = get_env_name(plugin_dir, cf)
            out.append((env_name, plugin_dir, cf, cfg))
    return out


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
    from ..environment.libomp import dedupe_libomp

    for env_name, _plugin, _cf, _cfg in discovered:
        env_dir = workspace_dir / ".pixi" / "envs" / env_name
        sp_matches = _glob.glob(str(env_dir / "lib" / "python*" / "site-packages"))
        if not sp_matches:
            continue
        try:
            dedupe_libomp(Path(sp_matches[0]))
            log(f"[comfy-env] {env_name}: deduped libomp")
        except Exception as e:
            log(f"[comfy-env] {env_name}: libomp dedupe failed: {e}")


# ---------------------------------------------------------------------------
# Torch-version probe via pixi run (used by _install_cuda_wheels validation;
# kept here so workspace.py is self-contained)
# ---------------------------------------------------------------------------

def _read_env_torch_version(
    pixi_path: Path,
    workspace_dir: Path,
    env_name: str,
    log: Optional[Callable[[str], None]] = None,
) -> Optional[str]:
    """Run `pixi run -e <env_name> python -c 'import torch; print(...)'`.

    Returns the public torch version (e.g. "2.11.0", local label stripped), or
    None if torch isn't importable from that env. If `log` is given, the
    failure reason (subprocess returncode/stderr/stdout) is written to it.

    Goes through `pixi run` rather than invoking the env's `python.exe`
    directly so pixi's own activation runs first — that's what puts
    `<env>/Library/bin` on PATH, sets KMP/MKL env vars correctly for the
    conda-forge libs, and matches what an end-user would get if they ran
    `pixi run -e <env> python` themselves.
    """
    import subprocess
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    r = subprocess.run(
        [str(pixi_path), "run", "-e", env_name,
         "python", "-c",
         "import torch, sys; sys.stdout.write(torch.__version__)"],
        cwd=str(workspace_dir),
        capture_output=True, text=True, env=env,
    )
    if r.returncode != 0:
        if log:
            log(
                f"[comfy-env] _read_env_torch_version: pixi run -e {env_name} "
                f"exit={r.returncode}; stderr={(r.stderr or '').strip()!r}; "
                f"stdout={(r.stdout or '').strip()!r}"
            )
        return None
    out = r.stdout.strip()
    if not out:
        if log:
            log(
                f"[comfy-env] _read_env_torch_version: pixi run -e {env_name} "
                f"exit=0 but empty stdout; stderr={(r.stderr or '').strip()!r}"
            )
        return None
    return out.split("+", 1)[0]


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


def _resolve_wheel_combo(
    discovered: List[Tuple[str, Path, Path, ComfyEnvConfig]],
    bootstrap_python: str,
    bootstrap_cuda: Optional[str],
    bootstrap_torch: Optional[str],
    log: Callable[[str], None],
) -> Optional[Tuple[str, str, str, str, str]]:
    """Pick the (python, cuda, torch_match, torch_pin, source) combo for the workspace.

    Strategy:
      1. Try the bootstrap combo (`bootstrap_python` / `bootstrap_cuda` / `bootstrap_torch`).
         If every required cuda-wheel is published for it, use it. Pin torch to
         `==<bootstrap_torch>`.
      2. Else try the known-good fallback `(bootstrap_python, FALLBACK_COMBO)` =
         `(py, "12.8", "2.8")`. Pin torch to `==2.8.*` (a major.minor pin -- pixi
         resolves the latest 2.8.x on the cu128 index).
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

    from ..packages.cuda_wheels import (
        check_all_wheels_available,
        FALLBACK_COMBO,
        CUDA_WHEELS_INDEX,
    )

    cp = bootstrap_python.replace(".", "")
    log(
        f"[comfy-env] cuda-wheels: {len(packages)} package(s) need a "
        f"matched (cuda, torch, python) combo: {packages}"
    )

    # Tier 1: bootstrap combo
    if bootstrap_torch:
        torch_short = ".".join(bootstrap_torch.split(".")[:2])
        log(
            f"[comfy-env] cuda-wheels tier 1 (bootstrap): probing "
            f"cu{bootstrap_cuda}/torch{torch_short}/cp{cp}"
        )
        miss = check_all_wheels_available(
            packages, torch_short, bootstrap_cuda, bootstrap_python, log=log,
        )
        if miss is None:
            log(
                f"[comfy-env] cuda-wheels combo: cu{bootstrap_cuda}/torch{torch_short}/cp{cp} "
                f"(bootstrap matches; per-node envs will pin to this)"
            )
            return (
                bootstrap_python,
                bootstrap_cuda,
                torch_short,
                f"=={bootstrap_torch}",
                "bootstrap",
            )
        log(
            f"[comfy-env] cuda-wheels tier 1 incomplete: `{miss}` not built for "
            f"cu{bootstrap_cuda}+torch{torch_short}+cp{cp}; falling back"
        )
    else:
        log(
            "[comfy-env] cuda-wheels: bootstrap torch unknown; skipping tier 1, trying fallback"
        )

    # Tier 2: known-good fallback (same python, cu128, torch 2.8)
    fb_cuda, fb_torch = FALLBACK_COMBO
    log(
        f"[comfy-env] cuda-wheels tier 2 (fallback): probing "
        f"cu{fb_cuda}/torch{fb_torch}/cp{cp}"
    )
    miss = check_all_wheels_available(
        packages, fb_torch, fb_cuda, bootstrap_python, log=log,
    )
    if miss is None:
        log(
            f"[comfy-env] cuda-wheels combo: cu{fb_cuda}/torch{fb_torch}/cp{cp} "
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

    raise RuntimeError(
        f"No cuda-wheels combo covers all required packages.\n"
        f"  packages: {packages}\n"
        f"  tier 1 (bootstrap): cu{bootstrap_cuda}/torch{bootstrap_torch}"
        f"/cp{bootstrap_python} -- missing or untried\n"
        f"  tier 2 (fallback):  cu{fb_cuda}/torch{fb_torch}.*"
        f"/cp{bootstrap_python} -- {miss} missing\n"
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
    """Generate `<comfyui_dir>/.ce/pixi.toml` and run `pixi install --all`.

    Returns the workspace directory on success, None if nothing to install.
    """
    from ..environment.cache import CE_WORKSPACE_DIR
    from ..packages.pixi import ensure_pixi
    from ..packages.toml_generator import write_workspace_pixi_toml

    comfyui_dir = Path(comfyui_dir).resolve()
    discovered = _discover_node_configs(comfyui_dir)
    if not discovered:
        log("[comfy-env] No custom-node comfy-env.toml files found -- skipping workspace install")
        return None

    workspace_dir = comfyui_dir / CE_WORKSPACE_DIR
    workspace_dir.mkdir(parents=True, exist_ok=True)

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
            torch_index, cuda_version, cuda_major,
            bootstrap_python, bootstrap_torch,
        ) = _resolve_workspace_torch(log)

        # Pre-validate cuda-wheel availability against the v2 index. May downgrade
        # the workspace's torch/cuda to a known-good combo if the bootstrap one
        # has unpublished wheels. Returns None on macOS / CPU / no cuda-only deps.
        combo = _resolve_wheel_combo(
            discovered, bootstrap_python, cuda_version, bootstrap_torch, log,
        )
        # The comfyui feature ALWAYS pins the bootstrap torch — it's what
        # python_embeded ships with on portable, and what the main ComfyUI
        # process actually loads. Per-node envs that have cuda-only wheel
        # requirements get an explicit override toward the chosen combo.
        torch_pin: Optional[str] = (
            f"=={bootstrap_torch}" if bootstrap_torch else None
        )
        chosen_torch_index: Optional[str] = None
        chosen_torch_pin_for_override: Optional[str] = None
        if combo is not None:
            chosen_python, chosen_cuda, chosen_torch_short, chosen_torch_pin_for_override, _src = combo
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

        node_configs = [(env_name, cfg) for env_name, _, _, cfg in discovered]
        write_workspace_pixi_toml(
            workspace_dir, comfyui_dir, torch_index, cuda_major, node_configs,
            bootstrap_python=bootstrap_python,
            torch_pin=torch_pin,
            log=log,
            chosen_torch_index=chosen_torch_index,
            chosen_torch_pin=chosen_torch_pin_for_override,
            chosen_cuda=chosen_cuda if combo is not None else None,
            chosen_torch_short=chosen_torch_short if combo is not None else None,
            chosen_python=chosen_python if combo is not None else None,
        )

        if dry_run:
            log("[comfy-env] dry_run -- skipping `pixi install`")
            return workspace_dir

        pixi_path = ensure_pixi(log=log)
        log(f"[comfy-env] pixi: {pixi_path}")

        _patch_uv_platform_py(log)

        pixi_env = dict(os.environ)
        pixi_env["UV_PYTHON_INSTALL_DIR"] = str(workspace_dir / "_no_python")
        pixi_env["UV_PYTHON_PREFERENCE"] = "only-system"

        log("[comfy-env] Running `pixi install --all`...")
        result = _run_streaming(
            [str(pixi_path), "install", "--all"],
            log=log, cwd=workspace_dir, env=pixi_env,
        )
        _log_subprocess(log, result, "pixi install --all")
        if result.returncode != 0:
            raise RuntimeError(
                f"pixi install --all failed:\nstderr: {result.stderr}\nstdout: {result.stdout}"
            )

        # Report envs that aren't in the current manifest, but DO NOT prune.
        # A user may have multiple ComfyUI installs sharing this `.ce` workspace,
        # or a node's `comfy-env.toml` may be transiently missing (mid-clone,
        # partial checkout). Auto-pruning here would delete real working envs.
        envs_dir = workspace_dir / ".pixi" / "envs"
        if envs_dir.is_dir():
            current_names = {env_name for env_name, _, _, _ in discovered}
            current_names.add("comfyui")  # template env always present
            for d in envs_dir.iterdir():
                if not d.is_dir() or d.name in current_names:
                    continue
                log(
                    f"[comfy-env] Note: env `{d.name}` is on disk but no node "
                    f"declares it in this run. Leaving as-is. "
                    f"Remove via `pixi clean --environment {d.name}` if intended."
                )

        # CUDA-only wheels (cumesh, flash-attn, etc.) are inlined as
        # `pypi-dependencies` URL entries in the per-node features (see
        # `toml_generator.build_workspace_toml`). Pixi installs them as part of
        # `pixi install --all`, no post-step pip pass needed.

        # Dedupe libomp.dylib copies in each env's site-packages (macOS only).
        _dedupe_envs_libomp(workspace_dir, discovered, log)

        log(f"[comfy-env] Install log: {log_path}")
        return workspace_dir
    finally:
        try:
            tee_log.close()
        except Exception:
            pass
