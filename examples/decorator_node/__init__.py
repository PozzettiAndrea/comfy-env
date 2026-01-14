"""
Example ComfyUI nodes using the @isolated decorator.

This demonstrates the simplest way to run node methods in isolated subprocess.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
