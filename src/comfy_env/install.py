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
        # Workspace install — picks up every plugin's comfy-env.toml under custom_nodes/
        from .environment.paths import get_comfyui_dir
        comfyui_dir = get_comfyui_dir(node_dir)
        if comfyui_dir is None:
            log("[comfy-env] WARNING: Could not locate ComfyUI base; skipping workspace install")
        else:
            install_workspace(comfyui_dir, log=log, dry_run=dry_run)

    if install_main:
        _install_to_main_env(node_dir, log, dry_run, node_req_dirs=node_req_dirs)

    if not install_isolated and not install_main:
        log("\n[comfy-env] Both install targets disabled — nothing to install")

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
            log(f"  WARNING: [{rel}] has conda deps ({', '.join(pkgs)}) — cannot install to main env")

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
# Workspace install — the heart of the new model
# ---------------------------------------------------------------------------

def _resolve_workspace_torch(
    log: Callable[[str], None],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Decide (torch_index, cuda_version, cuda_major) once for the whole workspace.

    `cuda_version` is the full string (e.g. "12.4"), used by `get_wheel_url`.
    `cuda_major` is just the leading digit (e.g. "12"), used in `[system-requirements]`.

    macOS: (cpu_index, None, None). Linux/Windows + NVIDIA: cuXYZ index + version.
    Linux/Windows without GPU: (cpu_index, None, None).
    """
    from .detection import get_recommended_cuda_version, get_gpu_summary
    cpu_index = "https://download.pytorch.org/whl/cpu"

    if sys.platform == "darwin":
        return cpu_index, None, None

    log(f"[comfy-env] GPU: {get_gpu_summary()}")
    cuda_version = get_recommended_cuda_version()
    if not cuda_version:
        log("[comfy-env] No CUDA detected — using PyTorch CPU index")
        return cpu_index, None, None

    cu_tag = "cu" + cuda_version.replace(".", "")[:3]
    torch_index = f"https://download.pytorch.org/whl/{cu_tag}"
    cuda_major = cuda_version.split(".")[0]
    log(f"[comfy-env] CUDA {cuda_version} → torch index {torch_index}")
    return torch_index, cuda_version, cuda_major


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


def _install_cuda_wheels(
    workspace_dir: Path,
    discovered: List[Tuple[str, Path, Path, ComfyEnvConfig]],
    cuda_version: Optional[str],
    log: Callable[[str], None],
) -> None:
    """Install CUDA-only wheel packages (cumesh, flash-attn, etc.) into per-env site-packages.

    Skipped on macOS or without a CUDA host.
    """
    if not cuda_version or sys.platform == "darwin":
        return

    import subprocess
    from .packages.cuda_wheels import get_wheel_url, CUDA_TORCH_MAP

    pytorch_packages = {"torch", "torchvision", "torchaudio"}
    cu_short = ".".join(cuda_version.split(".")[:2])
    torch_ver = CUDA_TORCH_MAP.get(cu_short, "2.8")

    for env_name, _plugin, _cf, cfg in discovered:
        cuda_only = [p for p in cfg.cuda_packages if p not in pytorch_packages]
        if not cuda_only:
            continue
        env_python = workspace_dir / ".pixi" / "envs" / env_name / (
            "python.exe" if sys.platform == "win32" else "bin/python"
        )
        if not env_python.exists():
            log(f"[comfy-env] {env_name}: python not found at {env_python}, skipping cuda-wheels")
            continue

        pyv = subprocess.run(
            [str(env_python), "-c",
             "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            capture_output=True, text=True,
        )
        py_version = pyv.stdout.strip() or f"{sys.version_info.major}.{sys.version_info.minor}"

        log(f"[comfy-env] {env_name}: installing cuda-wheels {cuda_only}")
        for package in cuda_only:
            url = get_wheel_url(package, torch_ver, cuda_version, py_version)
            if not url:
                log(f"[comfy-env]   WARNING: No wheel for {package} "
                    f"(cu{cuda_version}/torch{torch_ver}/py{py_version})")
                continue
            log(f"[comfy-env]   {package} ← {url}")
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
        log("[comfy-env] No custom-node comfy-env.toml files found — skipping workspace install")
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
            log(f"  - {env_name} ← {rel} (python={cfg.python or 'host'})")

        torch_index, cuda_version, cuda_major = _resolve_workspace_torch(log)

        node_configs = [(env_name, cfg) for env_name, _, _, cfg in discovered]
        write_workspace_pixi_toml(
            workspace_dir, comfyui_dir, torch_index, cuda_major, node_configs, log=log,
        )

        if dry_run:
            log("[comfy-env] dry_run — skipping `pixi install`")
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
        _install_cuda_wheels(workspace_dir, discovered, cuda_version, log)

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
