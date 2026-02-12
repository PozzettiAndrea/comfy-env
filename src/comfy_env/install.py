"""Installation API for comfy-env."""

import inspect
import os
import sys
from pathlib import Path
from typing import Callable, List, Optional, Set, Union

from .config import ComfyEnvConfig, NodeDependency, load_config, discover_config, CONFIG_FILE_NAME, ROOT_CONFIG_FILE_NAME
from .environment.cache import get_root_env_path, get_local_env_path

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
    if cfg.env_vars: _set_persistent_env_vars(cfg.env_vars, log, dry_run)
    if cfg.node_reqs:
        _install_node_dependencies(cfg.node_reqs, node_dir, log, dry_run)
        _reinstall_main_requirements(node_dir, log, dry_run)

    if _is_comfy_env_enabled():
        _install_via_pixi(cfg, node_dir, log, dry_run)
        _install_isolated_subdirs(node_dir, log, dry_run)
    else:
        log("\n[comfy-env] Isolation disabled (USE_COMFY_ENV=0)")
        _install_to_host_python(cfg, node_dir, log, dry_run)

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


def _set_persistent_env_vars(env_vars: dict, log: Callable[[str], None], dry_run: bool) -> None:
    import platform, subprocess
    if not env_vars: return

    log(f"\n[env] Setting {len(env_vars)} env var(s)")
    for k, v in env_vars.items(): log(f"  {k}={v}")
    if dry_run: return

    system = platform.system()
    if system == "Windows":
        for k, v in env_vars.items():
            subprocess.run(["setx", k, v], capture_output=True)
    elif system == "Darwin":
        for k, v in env_vars.items():
            subprocess.run(["launchctl", "setenv", k, v], capture_output=True)
        _add_to_shell_profile(env_vars, log)
    else:
        _add_to_shell_profile(env_vars, log)


def _add_to_shell_profile(env_vars: dict, log: Callable[[str], None]) -> None:
    shell = os.environ.get("SHELL", "/bin/bash")
    rc_file = Path.home() / (".zshrc" if "zsh" in shell else ".bashrc")
    profile_file = Path.home() / ".comfy-env-profile"

    with open(profile_file, "w") as f:
        f.write("# Generated by comfy-env\n")
        for k, v in env_vars.items():
            f.write(f'export {k}="{v}"\n')

    source_line = f'source "{profile_file}"'
    existing = rc_file.read_text() if rc_file.exists() else ""
    if source_line not in existing:
        with open(rc_file, "a") as f:
            f.write(f'\n# comfy-env\n{source_line}\n')
    log(f"  [env] Wrote {profile_file}")


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


def _install_via_pixi(cfg: ComfyEnvConfig, node_dir: Path, log: Callable[[str], None], dry_run: bool, is_root: bool = True) -> None:
    from .packages.pixi import ensure_pixi
    from .packages.toml_generator import write_pixi_toml
    from .packages.cuda_wheels import get_wheel_url, CUDA_TORCH_MAP
    from .detection import get_recommended_cuda_version, get_gpu_summary
    import shutil, subprocess, tempfile, time

    deps = cfg.pixi_passthrough.get("dependencies", {})
    pypi_deps = cfg.pixi_passthrough.get("pypi-dependencies", {})
    if not cfg.cuda_packages and not deps and not pypi_deps:
        if not is_root or not _has_isolated_subdirs(node_dir):
            log("No packages to install")
        return

    log(f"\nInstalling via pixi:")
    if cfg.cuda_packages: log(f"  CUDA: {', '.join(cfg.cuda_packages)}")
    if deps: log(f"  Conda: {len(deps)}")
    if pypi_deps: log(f"  PyPI: {len(pypi_deps)}")
    if dry_run: return

    # Root config -> _root_env, subdirectory config -> _env_*
    if is_root:
        env_path = get_root_env_path(node_dir)
    else:
        config_path = node_dir / CONFIG_FILE_NAME
        main_node_dir = _find_main_node_dir(node_dir)
        env_path = get_local_env_path(main_node_dir, config_path)

    # Central build dir -- shared across nodes with same config hash
    if sys.platform == "win32":
        build_base = Path("C:/ce")
    else:
        build_base = Path.home() / ".ce"
    build_base.mkdir(parents=True, exist_ok=True)
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
        """Link env_path -> build_dir/env (junction on Windows, symlink elsewhere)."""
        target = build_dir / "env"
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
        log(f"[comfy-env] Found existing env for {env_path.name}, skipping install ({build_dir / 'env'})")
        _link_env()
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
    try:
        pixi_path = ensure_pixi(log=log)
        log(f"[comfy-env] pixi={pixi_path}")

        cuda_version = torch_version = None
        if cfg.has_cuda and sys.platform != "darwin":
            log(f"[comfy-env] GPU: {get_gpu_summary()}")
            cuda_version = get_recommended_cuda_version()
            if cuda_version:
                torch_version = CUDA_TORCH_MAP.get(".".join(cuda_version.split(".")[:2]), "2.8")
                log(f"[comfy-env] Selected: CUDA {cuda_version} + PyTorch {torch_version}")
            else:
                log("[comfy-env] No GPU detected, using CPU")

        write_pixi_toml(cfg, build_dir, log)
        log("Running pixi install...")
        pixi_env = dict(os.environ)
        pixi_env["UV_PYTHON_INSTALL_DIR"] = str(build_dir / "_no_python")
        pixi_env["UV_PYTHON_PREFERENCE"] = "only-system"
        result = subprocess.run([str(pixi_path), "install"], cwd=build_dir, capture_output=True, text=True, env=pixi_env)
        if result.returncode != 0:
            raise RuntimeError(f"pixi install failed:\nstderr: {result.stderr}\nstdout: {result.stdout}")

        if cfg.cuda_packages and sys.platform != "darwin":
            pixi_default = build_dir / ".pixi" / "envs" / "default"
            python_path = pixi_default / ("python.exe" if sys.platform == "win32" else "bin/python")
            if not python_path.exists():
                raise RuntimeError(f"No Python in pixi env: {python_path}")

            result = subprocess.run([str(python_path), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                                   capture_output=True, text=True)
            py_version = result.stdout.strip() if result.returncode == 0 else f"{sys.version_info.major}.{sys.version_info.minor}"

            uv_path = _find_uv()
            log(f"[comfy-env] uv={uv_path}")
            log(f"[comfy-env] python={python_path} (py{py_version})")

            pytorch_packages = {"torch", "torchvision", "torchaudio"}
            torchvision_map = {"2.8": "0.23", "2.4": "0.19"}

            if cuda_version:
                pytorch_index = f"https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')[:3]}"
                pin_torch_version = torch_version
                log(f"Installing CUDA packages from {pytorch_index}")
            else:
                pytorch_index = "https://download.pytorch.org/whl/cpu"
                pin_torch_version = "2.8"
                log(f"Installing CPU packages from {pytorch_index}")

            for package in cfg.cuda_packages:
                if package in pytorch_packages:
                    if package == "torch":
                        pin_version = pin_torch_version
                    elif package == "torchvision":
                        pin_version = torchvision_map.get(pin_torch_version, "0.23")
                    else:
                        pin_version = pin_torch_version
                    pkg_spec = f"{package}=={pin_version}.*"
                    pip_cmd = [uv_path, "pip", "install", "--python", str(python_path),
                              "--extra-index-url", pytorch_index, "--index-strategy", "unsafe-best-match", pkg_spec]
                    log(f"  {' '.join(pip_cmd)}")
                    result = subprocess.run(pip_cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        raise RuntimeError(f"Failed to install {package}:\nstderr: {result.stderr}\nstdout: {result.stdout}")
                elif cuda_version:
                    wheel_url = get_wheel_url(package, torch_version, cuda_version, py_version)
                    if not wheel_url:
                        raise RuntimeError(f"No wheel for {package}")
                    log(f"  {package} from {wheel_url}")
                    cmd = [uv_path, "pip", "install", "--python", str(python_path), "--no-deps", wheel_url]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        raise RuntimeError(f"Failed to install {package}:\nstderr: {result.stderr}\nstdout: {result.stdout}")
                else:
                    log(f"  {package} (skipped - GPU only)")

        # Move env to build_dir/env, then link env_path -> build_dir/env
        pixi_default = build_dir / ".pixi" / "envs" / "default"
        if pixi_default.exists():
            final_short = build_dir / "env"
            if final_short.exists():
                _rmtree(final_short)
            shutil.move(str(pixi_default), str(final_short))
            _link_env()
            try: _rmtree(node_dir / ".pixi")
            except OSError: pass

        done_marker.touch()
    finally:
        try: lock_dir.rmdir()
        except OSError: pass


def _install_to_host_python(cfg: ComfyEnvConfig, node_dir: Path, log: Callable[[str], None], dry_run: bool) -> None:
    import shutil, subprocess, sys
    from .packages.cuda_wheels import get_wheel_url, CUDA_TORCH_MAP
    from .detection import get_recommended_cuda_version

    pypi_deps = cfg.pixi_passthrough.get("pypi-dependencies", {})
    if not pypi_deps and not cfg.cuda_packages:
        log("No packages to install")
        return

    pip_packages = []
    for pkg, spec in pypi_deps.items():
        if isinstance(spec, str):
            pip_packages.append(pkg if spec == "*" else f"{pkg}{spec}")
        elif isinstance(spec, dict):
            extras = spec.get("extras", [])
            version = spec.get("version", "*")
            name = f"{pkg}[{','.join(extras)}]" if extras else pkg
            pip_packages.append(name if version == "*" else f"{name}{version}")

    log(f"\nInstalling to {sys.executable}")
    if dry_run: return

    uv_path = _find_uv()
    if pip_packages:
        cmd = [uv_path, "pip", "install", "--python", sys.executable] + pip_packages
        subprocess.run(cmd, capture_output=True)

    if cfg.cuda_packages:
        cuda_version = get_recommended_cuda_version()
        if not cuda_version: return
        torch_version = CUDA_TORCH_MAP.get(".".join(cuda_version.split(".")[:2]), "2.8")
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}"

        for package in cfg.cuda_packages:
            wheel_url = get_wheel_url(package, torch_version, cuda_version, py_version)
            if wheel_url:
                cmd = [uv_path, "pip", "install", "--python", sys.executable, "--no-deps", wheel_url]
                subprocess.run(cmd, capture_output=True)


def _install_isolated_subdirs(node_dir: Path, log: Callable[[str], None], dry_run: bool) -> None:
    """Find and install comfy-env.toml in subdirectories (isolated folders only)."""
    for config_file in node_dir.rglob(CONFIG_FILE_NAME):
        if config_file.parent == node_dir: continue  # Skip root
        log(f"\n[isolated] {config_file.parent.relative_to(node_dir)}")
        if not dry_run:
            _install_via_pixi(load_config(config_file), config_file.parent, log, dry_run, is_root=False)


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
