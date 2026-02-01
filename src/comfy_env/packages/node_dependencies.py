"""Node dependency installation - clone ComfyUI nodes from [node_reqs] section."""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import NodeDependency


def normalize_repo_url(repo: str) -> str:
    if repo.startswith("http"): return repo
    return f"https://github.com/{repo}"


def clone_node(repo: str, name: str, target_dir: Path, log: Callable[[str], None] = print) -> Path:
    node_path = target_dir / name
    url = normalize_repo_url(repo)
    log(f"  Cloning {name}...")
    result = subprocess.run(["git", "clone", "--depth", "1", url, str(node_path)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to clone {url}: {result.stderr.strip()}")
    return node_path


def install_requirements(node_dir: Path, log: Callable[[str], None] = print) -> None:
    req_file = node_dir / "requirements.txt"
    if not req_file.exists(): return
    log(f"  Installing requirements for {node_dir.name}...")
    cmd = ["uv", "pip", "install", "-r", str(req_file), "--python", sys.executable] if shutil.which("uv") else [sys.executable, "-m", "pip", "install", "-r", str(req_file)]
    result = subprocess.run(cmd, cwd=node_dir, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"  Warning: requirements failed: {result.stderr.strip()[:200]}")


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
            clone_node(dep.repo, dep.name, custom_nodes_dir, log)
            install_requirements(node_path, log)
            run_install_script(node_path, log)

            nested_config = discover_config(node_path)
            if nested_config and nested_config.node_reqs:
                install_node_dependencies(nested_config.node_reqs, custom_nodes_dir, log, visited)
        except Exception as e:
            log(f"  Warning: {dep.name} failed: {e}")
