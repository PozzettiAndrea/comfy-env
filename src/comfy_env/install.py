"""Installation API for comfy-env.

In the workspace model, comfy-env owns one pixi workspace per ComfyUI install at
`<comfyui_dir>/.ce/`. Every custom node's `comfy-env.toml` becomes a pixi feature;
every (node, python-version) becomes a pixi environment. The `comfyui` feature
is generated from `<comfyui_dir>/requirements.txt` so torch + ComfyUI baseline
are always present in every env.

Each plugin's `install.py` calls `install()`. Per-plugin work (apt/brew/node_reqs,
main-env pip installs) runs locally; then `install_workspace(comfyui_dir)`
discovers all node tomls under `custom_nodes/` and runs a single `pixi install --all`.
"""

import inspect
import os
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Set, Tuple, Union

from .config import (
    ComfyEnvConfig,
    NodeDependency,
    load_config,
    discover_config,
    CONFIG_FILE_NAME,
    ROOT_CONFIG_FILE_NAME,
)
from .environment.cache import get_env_name

USE_COMFY_ENV_VAR = "USE_COMFY_ENV"


# ---------------------------------------------------------------------------
# Filesystem and platform helpers
# ---------------------------------------------------------------------------

def _rmtree(path) -> None:
    """rmtree that handles read-only files and long paths on Windows."""
    import shutil
    if sys.platform == "win32":
        import subprocess, tempfile
        target = str(Path(path).resolve())
        empty = tempfile.mkdtemp()
        try:
            subprocess.run(
                ["robocopy", empty, target, "/MIR", "/W:0", "/R:0"],
                capture_output=True,
            )
            shutil.rmtree(target, ignore_errors=True)
        finally:
            shutil.rmtree(empty, ignore_errors=True)
    else:
        shutil.rmtree(path)


def _is_comfy_env_enabled() -> bool:
    return os.environ.get(USE_COMFY_ENV_VAR, "1").lower() not in ("0", "false", "no", "off")


def _enable_windows_long_paths(log: Callable[[str], None]) -> None:
    """Enable Windows long path support via registry (requires admin)."""
    if sys.platform != "win32":
        return
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\FileSystem",
            0, winreg.KEY_SET_VALUE
        )
        winreg.SetValueEx(key, "LongPathsEnabled", 0, winreg.REG_DWORD, 1)
        winreg.CloseKey(key)
        log("[comfy-env] Enabled Windows long path support")
    except PermissionError:
        log("[comfy-env] WARNING: Could not enable long paths (needs admin)")
    except Exception:
        pass


def _patch_uv_platform_py(log: Callable[[str], None] = print) -> None:
    """Patch uv-managed Python's platform.py to handle conda-forge version strings.

    conda-forge Python embeds '| packaged by conda-forge |' in sys.version.
    When pixi's uv creates build-isolation venvs it may use a standard CPython
    whose platform.py can't parse that string, crashing setuptools.  Apply the
    same one-line regex fix that conda-forge ships in their own builds.
    """
    if sys.platform != "win32":
        return
    search_dirs = [
        Path.home() / "AppData" / "Roaming" / "uv" / "python",
        Path.home() / "AppData" / "Local" / "rattler" / "cache" / "python",
    ]
    MARKER = r"r'([\w.+]+)\s*'"
    REPLACEMENT = r"r'([\w.+]+)\s*(?:\ \|\ packaged\ by\ conda\-forge\ \|)?\s*'"
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for py_dir in search_dir.iterdir():
            if not py_dir.name.startswith("cpython-"):
                continue
            platform_py = py_dir / "Lib" / "platform.py"
            if not platform_py.exists():
                continue
            content = platform_py.read_text(encoding="utf-8")
            if "packaged by conda" in content:
                continue
            idx = content.find(MARKER)
            if idx == -1:
                continue
            patched = content[:idx] + REPLACEMENT + content[idx + len(MARKER):]
            platform_py.write_text(patched, encoding="utf-8")
            log(f"[comfy-env] Patched {platform_py} for conda-forge compat")


def _find_uv() -> str:
    """Find the uv binary installed alongside comfy-env."""
    import shutil
    exe_dir = Path(sys.executable).parent
    uv_name = "uv.exe" if sys.platform == "win32" else "uv"
    candidate = exe_dir / uv_name
    if candidate.exists():
        return str(candidate)
    if sys.platform == "win32":
        candidate = exe_dir / "Scripts" / uv_name
        if candidate.exists():
            return str(candidate)
    uv = shutil.which("uv")
    if uv:
        return uv
    raise FileNotFoundError("uv binary not found")


def _make_tee_log(log_callback: Callable[[str], None], log_path: Path) -> Callable[[str], None]:
    """Tee logs to both the original callback and a file."""
    import datetime
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "w", encoding="utf-8")
    fh.write(f"# comfy-env install log - {datetime.datetime.now().isoformat()}\n")
    fh.write(f"# Python: {sys.executable} ({sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro})\n")
    fh.write(f"# Platform: {sys.platform}\n\n")
    fh.flush()

    def tee(msg):
        log_callback(msg)
        sys.stdout.flush()
        fh.write(msg + "\n")
        fh.flush()

    tee.file = fh
    tee.close = fh.close
    tee.path = log_path
    return tee


def _log_subprocess(log: Callable, result, label: str = "") -> None:
    """Write subprocess stdout/stderr to the log file (verbose, file-only)."""
    fh = getattr(log, "file", None)
    if fh is None:
        return
    if label:
        fh.write(f"\n--- {label} (exit {result.returncode}) ---\n")
    if result.stdout and result.stdout.strip():
        fh.write(f"[stdout]\n{result.stdout}\n")
    if result.stderr and result.stderr.strip():
        fh.write(f"[stderr]\n{result.stderr}\n")
    fh.flush()


def _run_streaming(cmd, log: Callable, cwd=None, env=None):
    """Run a subprocess, streaming stdout/stderr lines to log in real time."""
    import subprocess
    import threading

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    def _read_stderr():
        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_lines.append(line)

    t = threading.Thread(target=_read_stderr, daemon=True)
    t.start()

    assert proc.stdout is not None
    for line in proc.stdout:
        line_text = line.rstrip("\n")
        stdout_lines.append(line_text)
        if line_text.strip():
            log(f"  {line_text}")

    proc.wait()
    t.join(timeout=5)

    return subprocess.CompletedProcess(
        cmd, proc.returncode,
        "\n".join(stdout_lines),
        "".join(stderr_lines),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def install(
    config: Optional[Union[str, Path]] = None,
    node_dir: Optional[Path] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    dry_run: bool = False,
) -> bool:
    """Install dependencies for the calling plugin and (re)build the workspace.

    Called from a plugin's `install.py` as `from comfy_env import install; install()`.
    Performs per-plugin work (apt/brew/node_reqs/main-env pip), then triggers a
    workspace-wide `pixi install --all` covering every plugin in this ComfyUI install.
    """
    if node_dir is None:
        node_dir = Path(inspect.stack()[1].filename).parent.resolve()

    log = log_callback or print
    _enable_windows_long_paths(log)

    if config is not None:
        config_path = Path(config)
        if not config_path.is_absolute():
            config_path = node_dir / config_path
        cfg = load_config(config_path)
    else:
        cfg = discover_config(node_dir, root=True)

    if cfg is None:
        raise FileNotFoundError(f"No {ROOT_CONFIG_FILE_NAME} or {CONFIG_FILE_NAME} found in {node_dir}")

    if cfg.apt_packages:
        _install_apt_packages(cfg.apt_packages, log, dry_run)
    if cfg.brew_packages:
        _install_brew_packages(cfg.brew_packages, log, dry_run)

    node_req_dirs: List[Path] = []
    if cfg.node_reqs:
        _install_node_dependencies(cfg.node_reqs, node_dir, log, dry_run)
        _reinstall_main_requirements(node_dir, log, dry_run)
        node_req_dirs = _collect_node_req_dirs(cfg.node_reqs, node_dir.parent)
        for nr_dir in node_req_dirs:
            nr_cfg = discover_config(nr_dir, root=True)
            if nr_cfg:
                if nr_cfg.apt_packages:
                    _install_apt_packages(nr_cfg.apt_packages, log, dry_run)
                if nr_cfg.brew_packages:
                    _install_brew_packages(nr_cfg.brew_packages, log, dry_run)

    from .settings import resolve_bool, GENERAL_DEFAULTS
    node_settings = cfg.settings if cfg.settings else None
    install_isolated = resolve_bool(
        "COMFY_ENV_INSTALL_ISOLATED", node_settings,
        GENERAL_DEFAULTS["COMFY_ENV_INSTALL_ISOLATED"],
    )
    install_main = resolve_bool(
        "COMFY_ENV_INSTALL_MAIN", node_settings,
        GENERAL_DEFAULTS["COMFY_ENV_INSTALL_MAIN"],
    )

    if install_isolated:
        # Workspace install -- picks up every plugin's comfy-env.toml under custom_nodes/
        from .environment.paths import get_comfyui_dir
        comfyui_dir = get_comfyui_dir(node_dir)
        if comfyui_dir is None:
            log("[comfy-env] WARNING: Could not locate ComfyUI base; skipping workspace install")
        else:
            install_workspace(comfyui_dir, log=log, dry_run=dry_run)

    if install_main:
        _install_to_main_env(node_dir, log, dry_run, node_req_dirs=node_req_dirs)

    if not install_isolated and not install_main:
        log("\n[comfy-env] Both install targets disabled -- nothing to install")

    log("\nInstallation complete!")
    return True


# ---------------------------------------------------------------------------
# Per-plugin helpers (apt/brew/node_reqs/main env)
# ---------------------------------------------------------------------------

def _install_apt_packages(packages: List[str], log: Callable[[str], None], dry_run: bool) -> None:
    from .packages.apt import apt_install
    import platform
    if platform.system() != "Linux":
        return
    log(f"\n[apt] Installing: {', '.join(packages)}")
    if not dry_run:
        success = apt_install(packages, log)
        if not success:
            log("[apt] WARNING: Some apt packages failed to install. This may cause issues.")


def _install_brew_packages(packages: List[str], log: Callable[[str], None], dry_run: bool) -> None:
    from .packages.brew import brew_install
    import platform
    if platform.system() != "Darwin":
        return
    log(f"\n[brew] Installing: {', '.join(packages)}")
    if not dry_run:
        success = brew_install(packages, log)
        if not success:
            log("[brew] WARNING: Some brew packages failed to install. This may cause issues.")


def _install_node_dependencies(
    node_reqs: List[NodeDependency], node_dir: Path,
    log: Callable[[str], None], dry_run: bool,
) -> None:
    from .packages.node_dependencies import install_node_dependencies
    custom_nodes_dir = node_dir.parent
    log(f"\nInstalling {len(node_reqs)} node dependencies...")
    if dry_run:
        for req in node_reqs:
            log(f"  {req.name}: {'exists' if (custom_nodes_dir / req.name).exists() else 'would clone'}")
        return
    install_node_dependencies(node_reqs, custom_nodes_dir, log, {node_dir.name})


def _reinstall_main_requirements(
    node_dir: Path, log: Callable[[str], None], dry_run: bool,
) -> None:
    """Re-install main package's requirements.txt after node_reqs to restore correct versions."""
    from .packages.node_dependencies import install_requirements
    req_file = node_dir / "requirements.txt"
    if not req_file.exists():
        return
    log(f"\n[requirements] Re-installing main package requirements...")
    if not dry_run:
        install_requirements(node_dir, log)


def _collect_node_req_dirs(
    node_reqs: List[NodeDependency],
    custom_nodes_dir: Path,
    visited: Optional[Set[str]] = None,
) -> List[Path]:
    """Recursively collect all directories of nodes installed via node_reqs."""
    visited = visited or set()
    result = []
    for dep in node_reqs:
        if dep.name in visited:
            continue
        visited.add(dep.name)
        node_path = custom_nodes_dir / dep.name
        if not node_path.exists():
            continue
        result.append(node_path)
        nested_cfg = discover_config(node_path)
        if nested_cfg and nested_cfg.node_reqs:
            result.extend(_collect_node_req_dirs(nested_cfg.node_reqs, custom_nodes_dir, visited))
    return result


def _install_to_main_env(
    node_dir: Path, log: Callable[[str], None], dry_run: bool,
    node_req_dirs: Optional[List[Path]] = None,
) -> None:
    """Install deps from comfy-env.toml files into the main Python env (isolation disabled)."""
    import subprocess

    log("\n[comfy-env] Installing to main env")
    all_pypi: dict = {}
    cuda_packages: List[str] = []
    conda_warnings: List[Tuple[str, List[str]]] = []

    scan_dirs = [node_dir]
    if node_req_dirs:
        scan_dirs.extend(node_req_dirs)
        log(f"  Scanning {len(node_req_dirs)} node_req(s): {', '.join(d.name for d in node_req_dirs)}")

    for scan_dir in scan_dirs:
        for config_file in scan_dir.rglob(CONFIG_FILE_NAME):
            if config_file.parent == scan_dir:
                continue
            cfg = load_config(config_file)
            try:
                rel = config_file.parent.relative_to(scan_dir)
                label = f"{scan_dir.name}/{rel}" if scan_dir != node_dir else str(rel)
            except ValueError:
                label = str(config_file.parent)
            pypi = cfg.pixi_passthrough.get("pypi-dependencies", {})
            conda = cfg.pixi_passthrough.get("dependencies", {})
            if pypi:
                log(f"  [{label}] PyPI: {', '.join(pypi.keys())}")
                all_pypi.update(pypi)
            if cfg.cuda_packages:
                cuda_packages.extend(cfg.cuda_packages)
            if conda:
                conda_warnings.append((label, list(conda.keys())))

    if conda_warnings:
        for rel, pkgs in conda_warnings:
            log(f"  WARNING: [{rel}] has conda deps ({', '.join(pkgs)}) -- cannot install to main env")

    if not all_pypi and not cuda_packages:
        log("  No packages to install")
        return

    pip_args: List[str] = []
    for name, spec in all_pypi.items():
        if isinstance(spec, dict):
            version = spec.get("version", "*")
            extras = spec.get("extras", [])
            pkg = name
            if extras:
                pkg = f"{name}[{','.join(extras)}]"
            if version and version != "*":
                pkg = f"{pkg}{version}"
            pip_args.append(pkg)
        elif isinstance(spec, str) and spec != "*":
            pip_args.append(f"{name}{spec}")
        else:
            pip_args.append(name)

    pytorch_packages = {"torch", "torchvision", "torchaudio"}
    cuda_only = [p for p in cuda_packages if p not in pytorch_packages]
    if cuda_only:
        try:
            from .packages.cuda_wheels import get_wheel_url
            from .detection import get_recommended_cuda_version
            import torch
            torch_ver = ".".join(torch.__version__.split(".")[:2])
            cuda_ver = get_recommended_cuda_version()
            py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
            for package in cuda_only:
                url = get_wheel_url(package, torch_ver, cuda_ver, py_ver)
                if url:
                    pip_args.append(url)
                    log(f"  {package} from {url}")
                else:
                    log(f"  WARNING: No cuda-wheel for {package} (cu{cuda_ver}/torch{torch_ver}/py{py_ver})")
        except Exception as e:
            log(f"  WARNING: Could not resolve cuda-wheels: {e}")

    if pip_args:
        log(f"\n  pip install {' '.join(pip_args)}")
        if not dry_run:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install"] + pip_args,
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                log(f"  WARNING: pip install failed:\n{result.stderr}")
            else:
                log("  Installed successfully")


# ---------------------------------------------------------------------------
# Workspace install -- the heart of the new model
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
    from .detection import (
        get_recommended_cuda_version,
        get_gpu_summary,
        get_bootstrap_python_version,
        get_bootstrap_torch_version,
    )
    cpu_index = "https://download.pytorch.org/whl/cpu"
    python_version = get_bootstrap_python_version()
    torch_version = get_bootstrap_torch_version()

    if torch_version:
        log(f"[comfy-env] Bootstrap python={python_version} torch={torch_version}")
    else:
        log(f"[comfy-env] Bootstrap python={python_version} (no torch importable)")

    if sys.platform == "darwin":
        return cpu_index, None, None, python_version, torch_version

    log(f"[comfy-env] GPU: {get_gpu_summary()}")
    cuda_version = get_recommended_cuda_version()
    if not cuda_version:
        log("[comfy-env] No CUDA detected -- using PyTorch CPU index")
        return cpu_index, None, None, python_version, torch_version

    cu_tag = "cu" + cuda_version.replace(".", "")[:3]
    torch_index = f"https://download.pytorch.org/whl/{cu_tag}"
    cuda_major = cuda_version.split(".")[0]
    log(f"[comfy-env] CUDA {cuda_version} -> torch index {torch_index}")
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
    from .environment.libomp import dedupe_libomp

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


def _read_env_torch_version(env_python: Path) -> Optional[str]:
    """Run `python -c 'import torch; print(torch.__version__)'` in a pixi env.

    Returns the public torch version (e.g. "2.11.0", local label stripped), or
    None if torch isn't importable from that env.
    """
    import subprocess
    if not env_python.exists():
        return None
    r = subprocess.run(
        [str(env_python), "-c",
         "import torch, sys; sys.stdout.write(torch.__version__)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return None
    out = r.stdout.strip()
    if not out:
        return None
    return out.split("+", 1)[0]


_PYTORCH_PACKAGES = {"torch", "torchvision", "torchaudio"}


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

    packages = _aggregate_cuda_packages(discovered)
    if not packages:
        return None

    from .packages.cuda_wheels import (
        check_all_wheels_available,
        FALLBACK_COMBO,
        CUDA_WHEELS_INDEX,
    )

    log(f"[comfy-env] cuda-wheels: probing {len(packages)} package(s) {packages}")

    # Tier 1: bootstrap combo
    if bootstrap_torch:
        torch_short = ".".join(bootstrap_torch.split(".")[:2])
        miss = check_all_wheels_available(
            packages, torch_short, bootstrap_cuda, bootstrap_python, log=log,
        )
        if miss is None:
            log(
                f"[comfy-env] cuda-wheels combo: cu{bootstrap_cuda}/torch{torch_short}"
                f"/cp{bootstrap_python.replace('.', '')} (bootstrap)"
            )
            return (
                bootstrap_python,
                bootstrap_cuda,
                torch_short,
                f"=={bootstrap_torch}",
                "bootstrap",
            )
        log(
            f"[comfy-env] cuda-wheels: {miss} not built for "
            f"cu{bootstrap_cuda}+torch{torch_short}+cp{bootstrap_python.replace('.', '')}; "
            f"trying fallback"
        )
    else:
        log(
            "[comfy-env] cuda-wheels: bootstrap torch unknown; trying fallback combo"
        )

    # Tier 2: known-good fallback (same python, cu128, torch 2.8)
    fb_cuda, fb_torch = FALLBACK_COMBO
    miss = check_all_wheels_available(
        packages, fb_torch, fb_cuda, bootstrap_python, log=log,
    )
    if miss is None:
        log(
            f"[comfy-env] cuda-wheels combo: cu{fb_cuda}/torch{fb_torch}"
            f"/cp{bootstrap_python.replace('.', '')} (fallback)"
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


def _install_cuda_wheels(
    workspace_dir: Path,
    discovered: List[Tuple[str, Path, Path, ComfyEnvConfig]],
    chosen_python: str,
    chosen_cuda: str,
    chosen_torch_short: str,
    log: Callable[[str], None],
) -> None:
    """Install CUDA-only wheel packages (cumesh, flash-attn, etc.) into per-env site-packages.

    The combo `(chosen_python, chosen_cuda, chosen_torch_short)` was pre-validated
    by `_resolve_wheel_combo` against the v2 index and matches what the comfyui
    template env actually has on disk after `pixi install --all`. As a sanity
    check, this function reads the template env's torch version and fails loudly
    if it diverges from `chosen_torch_short`.
    """
    if sys.platform == "darwin":
        return

    import subprocess
    from .packages.cuda_wheels import get_wheel_url, CUDA_WHEELS_INDEX

    py_exe = "python.exe" if sys.platform == "win32" else "bin/python"
    template_python = workspace_dir / ".pixi" / "envs" / "comfyui" / py_exe
    template_torch = _read_env_torch_version(template_python)
    if not template_torch:
        raise RuntimeError(
            f"Cannot read torch version from comfyui template env at "
            f"{template_python}. Either pixi install --all didn't materialize "
            f"the env, or torch isn't in the comfyui feature."
        )
    template_short = ".".join(template_torch.split(".")[:2])
    if template_short != chosen_torch_short:
        raise RuntimeError(
            f"Workspace torch drift: pinned cu{chosen_cuda}/torch{chosen_torch_short} "
            f"but comfyui env materialized torch {template_torch} "
            f"(short {template_short}). Regenerate pixi.toml or check resolver."
        )
    log(f"[comfy-env] cuda-wheels: torch={template_torch} cuda={chosen_cuda}")

    for env_name, _plugin, _cf, cfg in discovered:
        cuda_only = [p for p in cfg.cuda_packages if p not in _PYTORCH_PACKAGES]
        if not cuda_only:
            continue
        env_python = workspace_dir / ".pixi" / "envs" / env_name / py_exe
        if not env_python.exists():
            log(f"[comfy-env] {env_name}: python not found at {env_python}, skipping cuda-wheels")
            continue

        log(f"[comfy-env] {env_name}: installing cuda-wheels {cuda_only}")
        for package in cuda_only:
            url = get_wheel_url(
                package, chosen_torch_short, chosen_cuda, chosen_python, log=log,
            )
            if not url:
                raise RuntimeError(
                    f"cuda-wheel disappeared between probe and install: {package} "
                    f"for cu{chosen_cuda}/torch{chosen_torch_short}/cp{chosen_python}. "
                    f"Check {CUDA_WHEELS_INDEX}{package}/."
                )
            log(f"[comfy-env]   {package} <- {url}")
            cmd = [str(env_python), "-m", "pip", "install", "--no-deps", "--no-cache-dir", url]
            result = subprocess.run(cmd, capture_output=True, text=True)
            _log_subprocess(log, result, f"pip install {package} (env={env_name})")
            if result.returncode != 0:
                raise RuntimeError(
                    f"pip install failed for {package} in {env_name}:\n"
                    f"stderr: {result.stderr}\nstdout: {result.stdout}"
                )


def install_workspace(
    comfyui_dir: Path,
    log: Callable[[str], None] = print,
    dry_run: bool = False,
) -> Optional[Path]:
    """Generate `<comfyui_dir>/.ce/pixi.toml` and run `pixi install --all`.

    Returns the workspace directory on success, None if nothing to install.
    """
    from .environment.cache import CE_WORKSPACE_DIR
    from .packages.pixi import ensure_pixi
    from .packages.toml_generator import write_workspace_pixi_toml

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
        if combo is not None:
            chosen_python, chosen_cuda, chosen_torch_short, chosen_torch_pin, _src = combo
            cuda_version = chosen_cuda
            cuda_major = chosen_cuda.split(".")[0]
            torch_index = f"https://download.pytorch.org/whl/cu{chosen_cuda.replace('.', '')[:3]}"
            torch_pin: Optional[str] = chosen_torch_pin
        else:
            chosen_python = bootstrap_python
            chosen_cuda = cuda_version
            chosen_torch_short = (
                ".".join(bootstrap_torch.split(".")[:2]) if bootstrap_torch else None
            )
            torch_pin = f"=={bootstrap_torch}" if bootstrap_torch else None

        node_configs = [(env_name, cfg) for env_name, _, _, cfg in discovered]
        write_workspace_pixi_toml(
            workspace_dir, comfyui_dir, torch_index, cuda_major, node_configs,
            bootstrap_python=bootstrap_python,
            torch_pin=torch_pin,
            log=log,
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

        # Prune envs no longer in the manifest
        envs_dir = workspace_dir / ".pixi" / "envs"
        if envs_dir.is_dir():
            current_names = {env_name for env_name, _, _, _ in discovered}
            current_names.add("comfyui")  # template env always present
            for d in envs_dir.iterdir():
                if not d.is_dir():
                    continue
                if d.name in current_names:
                    continue
                log(f"[comfy-env] Removing stale env: {d.name}")
                import subprocess
                subprocess.run(
                    [str(pixi_path), "clean", "--environment", d.name],
                    cwd=workspace_dir,
                    capture_output=True, text=True,
                )

        # CUDA-only wheels (cumesh, flash-attn, etc.)
        if combo is not None:
            _install_cuda_wheels(
                workspace_dir, discovered,
                chosen_python=chosen_python,
                chosen_cuda=chosen_cuda,
                chosen_torch_short=chosen_torch_short,
                log=log,
            )

        # Dedupe libomp.dylib copies in each env's site-packages (macOS only).
        # Multiple bundled libomps from pip wheels (torch, sklearn, pymeshlab,
        # ...) coexisting in the same process can SIGSEGV inside native filters
        # -- KMP_DUPLICATE_LIB_OK only suppresses the abort, not the corruption.
        _dedupe_envs_libomp(workspace_dir, discovered, log)

        log(f"[comfy-env] Install log: {log_path}")
        return workspace_dir
    finally:
        try:
            tee_log.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def verify_installation(packages: List[str], log: Callable[[str], None] = print) -> bool:
    all_ok = True
    for package in packages:
        import_name = package.replace("-", "_").split("[")[0]
        try:
            __import__(import_name)
            log(f"  {package}: OK")
        except ImportError as e:
            log(f"  {package}: FAILED ({e})")
            all_ok = False
    return all_ok
