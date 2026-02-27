"""Node dependency installation - clone ComfyUI nodes from [node_reqs] section."""

import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import NodeDependency

COMFY_REGISTRY_API = "https://api.comfy.org"


def normalize_repo_url(repo: str) -> str:
    if repo.startswith("http"): return repo
    return f"https://github.com/{repo}"


def clone_node(
    repo: str,
    name: str,
    target_dir: Path,
    log: Callable[[str], None] = print,
    tag: Optional[str] = None,
    branch: Optional[str] = None,
    commit: Optional[str] = None,
) -> Path:
    """Clone a git repo. Supports tag, branch, or commit pinning."""
    node_path = target_dir / name
    url = normalize_repo_url(repo)

    if tag or branch:
        ref = tag or branch
        log(f"  Cloning {name} ({ref})...")
        cmd = ["git", "clone", "--depth", "1", "--branch", ref, url, str(node_path)]
    elif commit:
        log(f"  Cloning {name} ({commit[:8]})...")
        # Can't shallow-clone arbitrary commits, so full clone + checkout
        cmd = ["git", "clone", url, str(node_path)]
    else:
        log(f"  Cloning {name}...")
        cmd = ["git", "clone", "--depth", "1", url, str(node_path)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to clone {url}: {result.stderr.strip()}")

    if commit:
        result = subprocess.run(
            ["git", "-C", str(node_path), "checkout", commit],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to checkout {commit}: {result.stderr.strip()}")

    return node_path


def install_from_registry(
    registry_id: str,
    name: str,
    target_dir: Path,
    log: Callable[[str], None] = print,
    version: Optional[str] = None,
) -> Path:
    """Install a node from the ComfyUI registry (api.comfy.org)."""
    node_path = target_dir / name

    url = f"{COMFY_REGISTRY_API}/nodes/{registry_id}/install"
    if version:
        url += f"?version={version}"
        log(f"  Installing {name} from registry (v{version})...")
    else:
        log(f"  Installing {name} from registry (latest)...")

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"Registry lookup failed for {registry_id}: {e}")

    download_url = data.get("downloadUrl")
    if not download_url:
        raise RuntimeError(f"No downloadUrl in registry response for {registry_id}")

    log(f"  Downloading from {download_url}")
    try:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = tmp.name
            urllib.request.urlretrieve(download_url, tmp_path)

        with zipfile.ZipFile(tmp_path, "r") as zf:
            # Extract to a temp dir first, then move to target
            with tempfile.TemporaryDirectory() as extract_dir:
                zf.extractall(extract_dir)
                # Handle single top-level directory in archive
                entries = list(Path(extract_dir).iterdir())
                if len(entries) == 1 and entries[0].is_dir():
                    shutil.move(str(entries[0]), str(node_path))
                else:
                    shutil.move(extract_dir, str(node_path))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return node_path


def install_requirements(node_dir: Path, log: Callable[[str], None] = print) -> None:
    req_file = node_dir / "requirements.txt"
    if not req_file.exists(): return
    log(f"  Installing requirements for {node_dir.name}...")

    # Filter out comfy-env and sister packages to prevent self-downgrade
    _PROTECTED = {"comfy-env", "comfy_env", "comfy-test", "comfy_test", "comfy-3d-viewers", "comfy_3d_viewers", "comfy-attn", "comfy_attn"}
    lines = req_file.read_text().splitlines()
    filtered = [l for l in lines if not any(l.strip().lower().startswith(p) for p in _PROTECTED)]
    if len(filtered) < len(lines):
        # Write filtered requirements to temp file
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        tmp.write("\n".join(filtered) + "\n")
        tmp.close()
        target = tmp.name
    else:
        target = str(req_file)

    cmd = ["uv", "pip", "install", "-r", target, "--python", sys.executable] if shutil.which("uv") else [sys.executable, "-m", "pip", "install", "-r", target]
    result = subprocess.run(cmd, cwd=node_dir, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"  Warning: requirements failed: {result.stderr.strip()[:200]}")

    if target != str(req_file):
        import os
        os.unlink(target)


def run_install_script(node_dir: Path, log: Callable[[str], None] = print) -> None:
    install_script = node_dir / "install.py"
    if install_script.exists():
        log(f"  Running install.py for {node_dir.name}...")
        result = subprocess.run([sys.executable, str(install_script)], cwd=node_dir, capture_output=True, text=True)
        if result.returncode != 0:
            log(f"  Warning: install.py failed: {result.stderr.strip()[:200]}")


def install_node_dependencies(
    node_deps: "List[NodeDependency]",
    custom_nodes_dir: Path,
    log: Callable[[str], None] = print,
    visited: Set[str] = None,
) -> None:
    """Install node dependencies recursively."""
    from ..config import discover_config

    visited = visited or set()
    for dep in node_deps:
        if dep.name in visited:
            log(f"  {dep.name}: cycle, skipping")
            continue
        visited.add(dep.name)

        node_path = custom_nodes_dir / dep.name
        if node_path.exists():
            log(f"  {dep.name}: exists")
            continue

        try:
            if dep.registry:
                install_from_registry(dep.registry, dep.name, custom_nodes_dir, log, version=dep.version)
            elif dep.github:
                clone_node(dep.github, dep.name, custom_nodes_dir, log,
                           tag=dep.tag, branch=dep.branch, commit=dep.commit)
            else:
                log(f"  Warning: {dep.name} has no github or registry source, skipping")
                continue

            install_requirements(node_path, log)
            run_install_script(node_path, log)

            nested_config = discover_config(node_path)
            if nested_config and nested_config.node_reqs:
                install_node_dependencies(nested_config.node_reqs, custom_nodes_dir, log, visited)
        except Exception as e:
            log(f"  Warning: {dep.name} failed: {e}")
