"""Central environment cache at ~/.comfy-env/envs/"""

import glob
import hashlib
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Tuple

import tomli
import tomli_w

try:
    from .. import __version__
except ImportError:
    __version__ = "0.0.0-dev"

CACHE_DIR = Path.home() / ".comfy-env" / "envs"
MARKER_FILE = ".comfy-env-marker.toml"
METADATA_FILE = ".comfy-env-metadata.toml"


def get_cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def compute_config_hash(config_path: Path) -> str:
    return hashlib.sha256(config_path.read_bytes()).hexdigest()[:8]


def sanitize_name(name: str) -> str:
    name = name.lower()
    for prefix in ("comfyui-", "comfyui_"):
        if name.startswith(prefix): name = name[len(prefix):]
    return name.replace("-", "_").replace(" ", "_")


def get_env_name(node_dir: Path, config_path: Path) -> str:
    """Generate env name: <nodename>_<subfolder>_<hash>"""
    node_name = sanitize_name(node_dir.name)
    config_parent = config_path.parent
    if config_parent == node_dir:
        subfolder = ""
    else:
        try:
            subfolder = config_parent.relative_to(node_dir).as_posix().replace("/", "_")
        except ValueError:
            subfolder = sanitize_name(config_parent.name)
    return f"{node_name}_{subfolder}_{compute_config_hash(config_path)}"


def get_env_path(node_dir: Path, config_path: Path) -> Path:
    return get_cache_dir() / get_env_name(node_dir, config_path)

get_central_env_path = get_env_path


def write_marker_file(config_path: Path, env_path: Path) -> None:
    marker_path = config_path.parent / MARKER_FILE
    marker_path.write_text(tomli_w.dumps({
        "env": {"name": env_path.name, "path": str(env_path),
                "config_hash": compute_config_hash(config_path),
                "created": datetime.now().isoformat(), "comfy_env_version": __version__}
    }))

write_marker = write_marker_file


def write_env_metadata(env_path: Path, marker_path: Path) -> None:
    (env_path / METADATA_FILE).write_text(tomli_w.dumps({
        "marker_path": str(marker_path), "created": datetime.now().isoformat()
    }))


def read_marker_file(marker_path: Path) -> Optional[dict]:
    if not marker_path.exists(): return None
    try:
        with open(marker_path, "rb") as f: return tomli.load(f)
    except Exception: return None

read_marker = read_marker_file


def read_env_metadata(env_path: Path) -> Optional[dict]:
    metadata_path = env_path / METADATA_FILE
    if not metadata_path.exists(): return None
    try:
        with open(metadata_path, "rb") as f: return tomli.load(f)
    except Exception: return None


def resolve_env_path(node_dir: Path) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Resolve env with fallback: marker -> _env_<name> -> .pixi -> .venv"""
    # Marker
    marker = read_marker_file(node_dir / MARKER_FILE)
    if marker and "env" in marker:
        env_path = Path(marker["env"]["path"])
        if env_path.exists(): return _get_env_paths(env_path)

    # Local _env_<name>
    local_env = node_dir / f"_env_{sanitize_name(node_dir.name)}"
    if local_env.exists(): return _get_env_paths(local_env)

    # .pixi
    pixi_env = node_dir / ".pixi" / "envs" / "default"
    if pixi_env.exists(): return _get_env_paths(pixi_env)

    # .venv
    venv = node_dir / ".venv"
    if venv.exists(): return _get_env_paths(venv)

    return None, None, None


def _get_env_paths(env_path: Path) -> Tuple[Path, Optional[Path], Optional[Path]]:
    if sys.platform == "win32":
        return env_path, env_path / "Lib" / "site-packages", env_path / "Library" / "bin"
    matches = glob.glob(str(env_path / "lib" / "python*" / "site-packages"))
    return env_path, Path(matches[0]) if matches else None, env_path / "lib"


def cleanup_orphaned_envs(log: Callable[[str], None] = print) -> int:
    """Remove envs whose marker files no longer exist."""
    cache_dir = get_cache_dir()
    if not cache_dir.exists(): return 0

    cleaned = 0
    for env_dir in cache_dir.iterdir():
        if not env_dir.is_dir(): continue
        metadata = read_env_metadata(env_dir)
        if not metadata: continue
        marker_path = metadata.get("marker_path", "")
        if marker_path and not Path(marker_path).exists():
            log(f"[comfy-env] Cleaning: {env_dir.name}")
            try:
                shutil.rmtree(env_dir)
                cleaned += 1
            except Exception: pass
    return cleaned
