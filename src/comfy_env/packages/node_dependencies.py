"""
Node dependency installation for comfy-env.

Handles installation of dependent ComfyUI custom nodes
specified in the [node_reqs] section of comfy-env.toml.

Example configuration:
    [node_reqs]
    ComfyUI_essentials = "cubiq/ComfyUI_essentials"
    ComfyUI-DepthAnythingV2 = "kijai/ComfyUI-DepthAnythingV2"
"""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import NodeDependency


def normalize_repo_url(repo: str) -> str:
    """
    Convert GitHub shorthand to full URL.

    Args:
        repo: Either 'owner/repo' or full URL like 'https://github.com/owner/repo'

    Returns:
        Full GitHub URL.
    """
    if repo.startswith("http://") or repo.startswith("https://"):
        return repo
    return f"https://github.com/{repo}"


def clone_node(
    repo: str,
    name: str,
    target_dir: Path,
    log: Callable[[str], None] = print,
) -> Path:
    """
    Clone a node repository to target_dir/name.

    Args:
        repo: GitHub repo path (e.g., 'owner/repo') or full URL.
        name: Directory name for the cloned repo.
        target_dir: Parent directory (usually custom_nodes/).
        log: Logging callback.

    Returns:
        Path to the cloned node directory.

    Raises:
        RuntimeError: If git clone fails.
    """
    node_path = target_dir / name
    url = normalize_repo_url(repo)

    log(f"  Cloning {name} from {url}...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", url, str(node_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to clone {url}: {result.stderr.strip()}")

    return node_path


def install_requirements(
    node_dir: Path,
    log: Callable[[str], None] = print,
) -> None:
    """
    Install requirements.txt in a node directory if it exists.

    Uses uv if available, falls back to pip.

    Args:
        node_dir: Path to the node directory.
        log: Logging callback.
    """
    requirements_file = node_dir / "requirements.txt"

    if not requirements_file.exists():
        return

    log(f"  Installing requirements for {node_dir.name}...")

    # Try uv first, fall back to pip
    if shutil.which("uv"):
        cmd = ["uv", "pip", "install", "-r", str(requirements_file), "--python", sys.executable]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]

    result = subprocess.run(cmd, cwd=node_dir, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"  Warning: requirements.txt install failed for {node_dir.name}: {result.stderr.strip()[:200]}")


def run_install_script(
    node_dir: Path,
    log: Callable[[str], None] = print,
) -> None:
    """
    Run install.py in a node directory if it exists.

    Args:
        node_dir: Path to the node directory.
        log: Logging callback.
    """
    install_script = node_dir / "install.py"

    if install_script.exists():
        log(f"  Running install.py for {node_dir.name}...")
        result = subprocess.run(
            [sys.executable, str(install_script)],
            cwd=node_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            log(f"  Warning: install.py failed for {node_dir.name}: {result.stderr.strip()[:200]}")


def install_node_dependencies(
    node_deps: "List[NodeDependency]",
    custom_nodes_dir: Path,
    log: Callable[[str], None] = print,
    visited: Set[str] = None,
) -> None:
    """
    Install node dependencies recursively.

    Args:
        node_deps: List of NodeDependency objects to install.
        custom_nodes_dir: Path to custom_nodes directory.
        log: Logging callback.
        visited: Set of already-processed node names (for cycle detection).
    """
    from ..config import discover_config

    if visited is None:
        visited = set()

    for dep in node_deps:
        # Skip if already visited (cycle detection)
        if dep.name in visited:
            log(f"  {dep.name}: already in dependency chain, skipping")
            continue

        visited.add(dep.name)

        node_path = custom_nodes_dir / dep.name

        # Skip if already installed (directory exists)
        if node_path.exists():
            log(f"  {dep.name}: already installed, skipping")
            continue

        try:
            # Clone the repository
            clone_node(dep.repo, dep.name, custom_nodes_dir, log)

            # Install requirements.txt if present
            install_requirements(node_path, log)

            # Run install.py if present
            run_install_script(node_path, log)

            # Recursively process nested node_reqs
            try:
                nested_config = discover_config(node_path)
                if nested_config and nested_config.node_reqs:
                    log(f"  {dep.name}: found {len(nested_config.node_reqs)} nested dependencies")
                    install_node_dependencies(
                        nested_config.node_reqs,
                        custom_nodes_dir,
                        log,
                        visited,
                    )
            except Exception as e:
                # Don't fail if we can't parse nested config
                log(f"  {dep.name}: could not check for nested deps: {e}")

        except Exception as e:
            log(f"  Warning: Failed to install {dep.name}: {e}")
