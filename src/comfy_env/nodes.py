"""
Node dependency installation for comfy-env.

This module handles installation of dependent ComfyUI custom nodes
specified in the [node_reqs] section of comfy-env.toml.

Example configuration:
    [node_reqs]
    ComfyUI_essentials = "cubiq/ComfyUI_essentials"
    ComfyUI-DepthAnythingV2 = "kijai/ComfyUI-DepthAnythingV2"
"""

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Set

if TYPE_CHECKING:
    from .config.types import NodeReq


def normalize_repo_url(repo: str) -> str:
    """
    Convert GitHub shorthand to full URL.

    Args:
        repo: Either 'owner/repo' or full URL like 'https://github.com/owner/repo'

    Returns:
        Full GitHub URL
    """
    if repo.startswith("http://") or repo.startswith("https://"):
        return repo
    return f"https://github.com/{repo}"


def clone_node(
    repo: str,
    name: str,
    target_dir: Path,
    log: Callable[[str], None],
) -> Path:
    """
    Clone a node repository to target_dir/name.

    Args:
        repo: GitHub repo path (e.g., 'owner/repo') or full URL
        name: Directory name for the cloned repo
        target_dir: Parent directory (usually custom_nodes/)
        log: Logging callback

    Returns:
        Path to the cloned node directory

    Raises:
        RuntimeError: If git clone fails
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
    log: Callable[[str], None],
) -> None:
    """
    Install requirements.txt in a node directory if it exists.

    Args:
        node_dir: Path to the node directory
        log: Logging callback
    """
    requirements_file = node_dir / "requirements.txt"

    if requirements_file.exists():
        log(f"  Installing requirements for {node_dir.name}...")
        result = subprocess.run(
            ["uv", "pip", "install", "-r", str(requirements_file), "--python", sys.executable],
            cwd=node_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            log(f"  Warning: requirements.txt install failed for {node_dir.name}: {result.stderr.strip()[:200]}")


def run_install_script(
    node_dir: Path,
    log: Callable[[str], None],
) -> None:
    """
    Run install.py in a node directory if it exists.

    Args:
        node_dir: Path to the node directory
        log: Logging callback
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


def install_node_deps(
    node_reqs: "List[NodeReq]",
    custom_nodes_dir: Path,
    log: Callable[[str], None],
    visited: Set[str],
) -> None:
    """
    Install node dependencies recursively.

    Args:
        node_reqs: List of NodeReq objects to install
        custom_nodes_dir: Path to custom_nodes directory
        log: Logging callback
        visited: Set of already-processed node names (for cycle detection)
    """
    from .config.parser import discover_config

    for req in node_reqs:
        # Skip if already visited (cycle detection)
        if req.name in visited:
            log(f"  {req.name}: already in dependency chain, skipping")
            continue

        visited.add(req.name)

        node_path = custom_nodes_dir / req.name

        # Skip if already installed (directory exists)
        if node_path.exists():
            log(f"  {req.name}: already installed, skipping")
            continue

        try:
            # Clone the repository
            clone_node(req.repo, req.name, custom_nodes_dir, log)

            # Install requirements.txt if present
            install_requirements(node_path, log)

            # Run install.py if present
            run_install_script(node_path, log)

            # Recursively process nested node_reqs
            try:
                nested_config = discover_config(node_path)
                if nested_config and nested_config.node_reqs:
                    log(f"  {req.name}: found {len(nested_config.node_reqs)} nested dependencies")
                    install_node_deps(
                        nested_config.node_reqs,
                        custom_nodes_dir,
                        log,
                        visited,
                    )
            except Exception as e:
                # Don't fail if we can't parse nested config
                log(f"  {req.name}: could not check for nested deps: {e}")

        except Exception as e:
            log(f"  Warning: Failed to install {req.name}: {e}")
