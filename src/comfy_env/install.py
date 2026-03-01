"""Installation API for comfy-env."""

import inspect
import os
import sys
from pathlib import Path
from typing import Callable, List, Optional, Set, Union

from .config import ComfyEnvConfig, NodeDependency, load_config, discover_config, CONFIG_FILE_NAME, ROOT_CONFIG_FILE_NAME
from .environment.cache import get_local_env_path

USE_COMFY_ENV_VAR = "USE_COMFY_ENV"


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


def _find_main_node_dir(node_dir: Path) -> Path:
    """Walk up to find the custom_nodes/<plugin> root."""
    for parent in node_dir.parents:
        if parent.parent and parent.parent.name == "custom_nodes":
            return parent
    return node_dir


def _find_uv() -> str:
    """Find the uv binary installed alongside comfy-env."""
    import shutil
    exe_dir = Path(sys.executable).parent
    uv_name = "uv.exe" if sys.platform == "win32" else "uv"
    # Check next to python executable (venvs on Windows, bin/ on Unix)
    candidate = exe_dir / uv_name
    if candidate.exists():
        return str(candidate)
    # Check Scripts subdirectory (embedded Python on Windows)
    if sys.platform == "win32":
        candidate = exe_dir / "Scripts" / uv_name
        if candidate.exists():
            return str(candidate)
    # Fallback to PATH
    uv = shutil.which("uv")
    if uv:
        return uv
    raise FileNotFoundError("uv binary not found")


def _make_tee_log(log_callback: Callable[[str], None], log_path: Path) -> Callable[[str], None]:
    """Create a log callback that writes to both the original callback and a file.

    The returned callable has a ``.file`` attribute for writing verbose output
    (e.g. subprocess stdout/stderr) that shouldn't go to the console.
    Call ``.close()`` when done.
    """
    import datetime
    fh = open(log_path, "w", encoding="utf-8")
    fh.write(f"# comfy-env install log - {datetime.datetime.now().isoformat()}\n")
    fh.write(f"# Python: {sys.executable} ({sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro})\n")
    fh.write(f"# Platform: {sys.platform}\n\n")
    fh.flush()

    def tee(msg):
        log_callback(msg)
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


def install(
    config: Optional[Union[str, Path]] = None,
    node_dir: Optional[Path] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    dry_run: bool = False,
) -> bool:
    """Install dependencies from comfy-env-root.toml or comfy-env.toml."""
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

    if cfg.apt_packages: _install_apt_packages(cfg.apt_packages, log, dry_run)
    if cfg.brew_packages: _install_brew_packages(cfg.brew_packages, log, dry_run)
    if cfg.node_reqs:
        _install_node_dependencies(cfg.node_reqs, node_dir, log, dry_run)
        _reinstall_main_requirements(node_dir, log, dry_run)

    if _is_comfy_env_enabled():
        _install_isolated_subdirs(node_dir, log, dry_run)
    else:
        log("\n[comfy-env] Isolation disabled (USE_COMFY_ENV=0)")

    log("\nInstallation complete!")
    return True


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


def _install_node_dependencies(node_reqs: List[NodeDependency], node_dir: Path, log: Callable[[str], None], dry_run: bool) -> None:
    from .packages.node_dependencies import install_node_dependencies
    custom_nodes_dir = node_dir.parent
    log(f"\nInstalling {len(node_reqs)} node dependencies...")
    if dry_run:
        for req in node_reqs:
            log(f"  {req.name}: {'exists' if (custom_nodes_dir / req.name).exists() else 'would clone'}")
        return
    install_node_dependencies(node_reqs, custom_nodes_dir, log, {node_dir.name})


def _reinstall_main_requirements(node_dir: Path, log: Callable[[str], None], dry_run: bool) -> None:
    """Re-install main package's requirements.txt after node_reqs to restore correct versions."""
    from .packages.node_dependencies import install_requirements
    req_file = node_dir / "requirements.txt"
    if not req_file.exists():
        return
    log(f"\n[requirements] Re-installing main package requirements...")
    if not dry_run:
        install_requirements(node_dir, log)


def _has_isolated_subdirs(node_dir: Path) -> bool:
    """Check if there are any comfy-env.toml files in subdirectories."""
    for config_file in node_dir.rglob(CONFIG_FILE_NAME):
        if config_file.parent != node_dir:
            return True
    return False


def _save_env_metadata(build_dir: Path, node_dir: Path, config_path: Path) -> None:
    """Save source config metadata alongside the built environment."""
    import json
    try:
        main_dir = _find_main_node_dir(node_dir)
        try:
            subpath = str(node_dir.relative_to(main_dir))
        except ValueError:
            subpath = ""
        node_label = main_dir.name if subpath == "." else f"{main_dir.name}/{subpath}"

        # Parse config for a compact summary instead of dumping raw toml
        summary = {}
        try:
            import tomli
            with open(config_path, "rb") as f:
                toml_data = tomli.load(f)
            if "cuda" in toml_data and "packages" in toml_data["cuda"]:
                summary["cuda"] = toml_data["cuda"]["packages"]
            pypi = toml_data.get("pypi-dependencies", {})
            if pypi:
                summary["pypi_count"] = len(pypi)
            py_ver = toml_data.get("python")
            if py_ver:
                summary["python"] = py_ver
        except Exception:
            pass

        meta = {
            "node_name": node_label,
            "config_file": config_path.name,
            "config_content": config_path.read_text(encoding="utf-8"),
            **summary,
        }
        (build_dir / ".comfy-env-meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
    except Exception:
        pass  # Non-fatal — metadata is optional


_DETECT_SH = r'''#!/usr/bin/env bash
# List all comfy-env environments and their metadata
BASE="$(cd "$(dirname "$0")" && pwd)"
for d in "$BASE"/_env_*/; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    meta="$d/.comfy-env-meta.json"
    done_marker="$d/.done"
    log="$d/install.log"
    status="incomplete"; [ -f "$done_marker" ] && status="ok"
    printf "=== %s [%s] ===\n" "$name" "$status"
    [ -f "$meta" ] && cat "$meta"
    [ -f "$log" ] && printf "  install.log: %s\n" "$log"
    echo
done
'''

_DETECT_BAT = r'''@echo off
setlocal enabledelayedexpansion
REM List all comfy-env environments and their metadata
for /d %%d in (%~dp0\_env_*) do (
    set "STATUS=incomplete"
    if exist "%%d\.done" set "STATUS=ok"
    echo === %%~nxd [!STATUS!] ===
    if exist "%%d\.comfy-env-meta.json" type "%%d\.comfy-env-meta.json"
    if exist "%%d\install.log" echo   install.log: %%d\install.log
    echo.
)
'''


def _ensure_detect_scripts(build_base: Path) -> None:
    """Write detect.sh / detect.bat to the build cache directory.

    Always overwrites so scripts stay up-to-date with comfy-env.
    """
    try:
        sh = build_base / "detect.sh"
        sh.write_text(_DETECT_SH, encoding="utf-8")
        sh.chmod(0o755)
    except Exception:
        pass
    try:
        bat = build_base / "detect.bat"
        bat.write_text(_DETECT_BAT, encoding="utf-8")
    except Exception:
        pass


def _install_via_pixi(cfg: ComfyEnvConfig, node_dir: Path, log: Callable[[str], None], dry_run: bool) -> None:
    """Install dependencies into an isolated pixi environment (for comfy-env.toml subdirs only)."""
    from .packages.pixi import ensure_pixi
    from .packages.toml_generator import write_pixi_toml
    from .packages.cuda_wheels import get_wheel_url, check_all_wheels_available, CUDA_TORCH_MAP, FALLBACK_COMBO
    from .detection import get_recommended_cuda_version, get_gpu_summary
    import shutil, subprocess, tempfile, time

    deps = cfg.pixi_passthrough.get("dependencies", {})
    pypi_deps = cfg.pixi_passthrough.get("pypi-dependencies", {})
    has_pixi_deps = bool(deps or pypi_deps)
    has_cuda = bool(cfg.cuda_packages)
    if not has_pixi_deps and not has_cuda:
        return

    log(f"\nInstalling via pixi:")
    if has_cuda: log(f"  CUDA: {', '.join(cfg.cuda_packages)}")
    if deps: log(f"  Conda: {len(deps)}")
    if pypi_deps: log(f"  PyPI: {len(pypi_deps)}")
    if dry_run: return

    config_path = node_dir / CONFIG_FILE_NAME
    main_node_dir = _find_main_node_dir(node_dir)
    env_path = get_local_env_path(main_node_dir, config_path)

    # Central build dir -- shared across nodes with same config hash
    if sys.platform == "win32":
        build_base = Path("C:/ce")
    else:
        build_base = Path.home() / ".ce"
    build_base.mkdir(parents=True, exist_ok=True)
    _ensure_detect_scripts(build_base)
    build_dir = build_base / env_path.name
    log(f"[comfy-env] build_dir={build_dir}")
    log(f"[comfy-env] env_path={env_path}")

    done_marker = build_dir / ".done"
    lock_dir = build_dir / ".building"

    def _is_link_or_junction(p):
        """Check if path is a symlink or NTFS junction (works on Python 3.10+)."""
        if p.is_symlink():
            return True
        if sys.platform == "win32":
            import stat
            try:
                return bool(os.lstat(str(p)).st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
            except (OSError, AttributeError):
                pass
        return False

    def _link_env():
        """Link env_path -> build_dir/.pixi/envs/default (junction on Windows, symlink on Unix)."""
        target = build_dir / ".pixi" / "envs" / "default"
        if not target.exists():
            return
        if _is_link_or_junction(env_path):
            # unlink for symlinks, rmdir for junctions -- never _rmtree (would follow the link)
            try: env_path.unlink()
            except OSError: env_path.rmdir()
        elif env_path.exists():
            _rmtree(env_path)
        env_path.parent.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            # Junctions don't require Developer Mode; symlinks do
            subprocess.run(["cmd", "/c", "mklink", "/J", str(env_path), str(target)],
                          capture_output=True)
        else:
            env_path.symlink_to(target)
        log(f"Env: {env_path} -> {target}")

    # Fast path: env already built
    if done_marker.exists():
        log(f"[comfy-env] Found existing env for {env_path.name}, skipping install")
        _link_env()
        # Backfill metadata for pre-existing builds
        if not (build_dir / ".comfy-env-meta.json").exists():
            _save_env_metadata(build_dir, node_dir, config_path)
        try: _rmtree(node_dir / ".pixi")
        except OSError: pass
        return

    # Try to acquire build lock (mkdir is atomic)
    try:
        build_dir.mkdir(parents=True, exist_ok=True)
        lock_dir.mkdir(exist_ok=False)
    except FileExistsError:
        # Another process is building -- wait for completion
        log("[comfy-env] Another build in progress, waiting...")
        for _ in range(600):  # 10 min timeout
            if done_marker.exists():
                log("[comfy-env] Build completed by other process, reusing")
                _link_env()
                try: _rmtree(node_dir / ".pixi")
                except OSError: pass
                return
            time.sleep(1)
        # Stale lock from crashed build -- nuke and take over
        log("[comfy-env] Stale lock detected, rebuilding...")
        _rmtree(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)
        lock_dir.mkdir(exist_ok=True)

    # We own the build
    tee_log = None
    try:
        # Tee all output to build_dir/install.log
        build_dir.mkdir(parents=True, exist_ok=True)
        tee_log = _make_tee_log(log, build_dir / "install.log")
        log = tee_log

        pixi_path = ensure_pixi(log=log)
        log(f"[comfy-env] pixi={pixi_path}")

        cuda_version = torch_version = None
        cuda_override = torch_override = None
        force_install_torch = False
        pytorch_packages = {"torch", "torchvision", "torchaudio"}
        cuda_wheels_packages = [p for p in cfg.cuda_packages if p not in pytorch_packages]

        if cfg.has_cuda and sys.platform != "darwin":
            log(f"[comfy-env] GPU: {get_gpu_summary()}")
            cuda_version = get_recommended_cuda_version()
            if cuda_version:
                # Try the host's actual torch version first — if all cuda-wheels
                # are available for it, we can inherit torch from the host (zero-copy).
                host_torch = None
                host_cuda = None
                try:
                    import torch as _torch
                    # e.g. "2.8.0+cu128" → "2.8"
                    host_torch = ".".join(_torch.__version__.split(".")[:2])
                    host_cuda = _torch.version.cuda  # e.g. "12.8"
                except Exception:
                    pass

                py_version = cfg.python or f"{sys.version_info.major}.{sys.version_info.minor}"
                if host_torch:
                    log(f"[comfy-env] Main env: torch {_torch.__version__}, Python {py_version}")
                else:
                    log(f"[comfy-env] Main env: no torch, Python {py_version}")

                if host_torch and cuda_wheels_packages:
                    log(f"[comfy-env] Target: cu{cuda_version}/torch{host_torch}/py{py_version}")
                    log(f"[comfy-env] Checking cuda-wheels for: {', '.join(cuda_wheels_packages)}")
                    missing = check_all_wheels_available(cuda_wheels_packages, host_torch, cuda_version, py_version)
                    if missing is None:
                        torch_version = host_torch
                        log(f"[comfy-env] All cuda-wheels available for "
                            f"cu{cuda_version}/torch{torch_version}/py{py_version} — will share_torch")
                    else:
                        # Fall back to known-good combo and install torch via pixi
                        fb_cuda, fb_torch = FALLBACK_COMBO
                        log(f"[comfy-env] Missing wheel: {missing} "
                            f"(cu{cuda_version}/torch{host_torch}/py{py_version})")
                        log(f"[comfy-env] Falling back to cu{fb_cuda}/torch{fb_torch}")
                        cuda_version, torch_version = fb_cuda, fb_torch
                        cuda_override, torch_override = fb_cuda, fb_torch
                        force_install_torch = True
                else:
                    torch_version = CUDA_TORCH_MAP.get(".".join(cuda_version.split(".")[:2]), "2.8")

                log(f"[comfy-env] Selected: CUDA {cuda_version} + PyTorch {torch_version}")
            else:
                log("[comfy-env] No GPU detected, using CPU")

        write_pixi_toml(cfg, build_dir, log, cuda_override=cuda_override,
                        torch_override=torch_override, force_install_torch=force_install_torch)
        log("Running pixi install...")
        pixi_env = dict(os.environ)
        pixi_env["UV_PYTHON_INSTALL_DIR"] = str(build_dir / "_no_python")
        pixi_env["UV_PYTHON_PREFERENCE"] = "only-system"
        result = subprocess.run([str(pixi_path), "install"], cwd=build_dir, capture_output=True, text=True, env=pixi_env)
        _log_subprocess(log, result, "pixi install")
        if result.returncode != 0:
            raise RuntimeError(f"pixi install failed:\nstderr: {result.stderr}\nstdout: {result.stdout}")

        # Install cuda-wheels packages (nvdiffrast, pytorch3d, etc.) via uv pip.
        # PyTorch packages (torch, torchvision, torchaudio) are handled by pixi
        # via per-package index URLs in the generated pixi.toml.
        if cuda_wheels_packages and cuda_version and sys.platform != "darwin":
            pixi_default = build_dir / ".pixi" / "envs" / "default"
            python_path = pixi_default / ("python.exe" if sys.platform == "win32" else "bin/python")
            if not python_path.exists():
                raise RuntimeError(f"No Python in pixi env: {python_path}")

            result = subprocess.run([str(python_path), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                                   capture_output=True, text=True)
            py_version = result.stdout.strip() if result.returncode == 0 else f"{sys.version_info.major}.{sys.version_info.minor}"

            uv_path = _find_uv()
            log(f"[comfy-env] Installing cuda-wheels packages (uv={uv_path}, python={python_path})")

            for package in cuda_wheels_packages:
                wheel_url = get_wheel_url(package, torch_version, cuda_version, py_version)
                if not wheel_url:
                    raise RuntimeError(f"No wheel for {package} (cu{cuda_version}/torch{torch_version}/py{py_version})")
                log(f"  {package} from {wheel_url}")
                cmd = [uv_path, "pip", "install", "--python", str(python_path), "--no-deps", "--no-cache", wheel_url]
                result = subprocess.run(cmd, capture_output=True, text=True)
                _log_subprocess(log, result, f"pip install {package}")
                if result.returncode != 0:
                    raise RuntimeError(f"Failed to install {package}:\nstderr: {result.stderr}\nstdout: {result.stdout}")

        # Link _env_<hash> directly to .pixi/envs/default.
        # We do NOT move the env -- conda packages have hardcoded RPATHs
        # pointing to .pixi/envs/default/lib/ and moving breaks them.
        _link_env()
        try: _rmtree(node_dir / ".pixi")
        except OSError: pass

        done_marker.touch()
        _save_env_metadata(build_dir, node_dir, config_path)
        log(f"[comfy-env] Install log: {build_dir / 'install.log'}")
    finally:
        if tee_log:
            tee_log.close()
        try: lock_dir.rmdir()
        except OSError: pass


def _install_isolated_subdirs(node_dir: Path, log: Callable[[str], None], dry_run: bool) -> None:
    """Find and install comfy-env.toml in subdirectories (isolated folders only)."""
    for config_file in node_dir.rglob(CONFIG_FILE_NAME):
        if config_file.parent == node_dir: continue  # Skip root
        log(f"\n[isolated] {config_file.parent.relative_to(node_dir)}")
        if not dry_run:
            _install_via_pixi(load_config(config_file), config_file.parent, log, dry_run)


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
