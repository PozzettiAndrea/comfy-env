"""
Isolation module for wrapping ComfyUI nodes to run in isolated environments.
"""

from .wrap import wrap_isolated_nodes, wrap_nodes

__all__ = [
    "wrap_isolated_nodes",
    "wrap_nodes",
]
