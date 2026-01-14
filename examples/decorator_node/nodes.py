"""
Example ComfyUI nodes using the @isolated decorator.

This demonstrates the simplest possible way to run node methods in
isolated subprocess environments. No separate worker file needed!

The @isolated decorator:
1. Extracts your method source code at class decoration time
2. Auto-generates a worker file on first call
3. Creates an isolated venv with your requirements
4. Intercepts method calls and forwards to the subprocess
5. Handles all serialization/deserialization automatically

Compare this to examples/basic_node which uses the traditional approach
with separate worker.py and explicit bridge management.
"""

from comfy_env import isolated


# ===========================================================================
# EXAMPLE 1: Simple inline requirements
# ===========================================================================

@isolated(
    env="decorator-example",
    requirements=["torch", "pillow", "numpy"],
    python="3.10",
    cuda="auto",  # Auto-detect GPU (12.8 for Blackwell, 12.4 for others)
)
class SimpleIsolatedNode:
    """
    Simplest possible isolated node - just add @isolated decorator!

    The process() method body runs in a separate Python subprocess
    with its own torch installation. The main ComfyUI process never
    imports torch from this node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "example/decorator"

    def process(self, image, brightness):
        # Everything below runs in isolated subprocess!
        # These imports happen in the subprocess, not ComfyUI main process
        import torch
        import numpy as np

        # The image arrives already deserialized as a torch tensor
        # (comfyui-isolation handles ComfyUI IMAGE format automatically)

        # Apply brightness adjustment
        result = image * brightness
        result = torch.clamp(result, 0.0, 1.0)

        return (result,)


# ===========================================================================
# EXAMPLE 2: Multiple isolated methods
# ===========================================================================

@isolated(
    env="multi-method-example",
    requirements=["torch", "pillow", "numpy"],
)
class MultiMethodNode:
    """
    Node with multiple methods that all run in the isolated subprocess.

    Use ISOLATED_METHODS to specify which methods should be isolated.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "operation": (["invert", "grayscale", "double"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "example/decorator"

    # List all methods that should run in subprocess
    ISOLATED_METHODS = ["process", "invert_image", "to_grayscale", "double_brightness"]

    def process(self, image, operation):
        import torch

        if operation == "invert":
            return self.invert_image(image)
        elif operation == "grayscale":
            return self.to_grayscale(image)
        else:
            return self.double_brightness(image)

    def invert_image(self, image):
        import torch
        return (1.0 - image,)

    def to_grayscale(self, image):
        import torch
        # Weighted grayscale conversion
        weights = torch.tensor([0.299, 0.587, 0.114])
        gray = (image[..., :3] * weights).sum(dim=-1, keepdim=True)
        gray = gray.expand_as(image[..., :3])
        if image.shape[-1] == 4:  # Has alpha
            gray = torch.cat([gray, image[..., 3:4]], dim=-1)
        return (gray,)

    def double_brightness(self, image):
        import torch
        return (torch.clamp(image * 2.0, 0.0, 1.0),)


# ===========================================================================
# EXAMPLE 3: Using TOML config for complex dependencies
# ===========================================================================

# For complex dependencies with custom wheel sources, use a config file:
#
# @isolated(env="complex-deps", config="isolation_config.toml")
# class ComplexDepsNode:
#     """Node with complex dependencies defined in TOML."""
#
#     FUNCTION = "process"
#
#     def process(self, image):
#         import pytorch3d  # From custom wheel index
#         import nvdiffrast  # From custom wheel index
#         ...
#
# The isolation_config.toml would look like:
#
# [env]
# name = "complex-deps"
# python = "3.10"
# cuda = "auto"
#
# [packages]
# requirements = [
#     "torch=={pytorch_version}",
#     "pytorch3d==0.7.8+5043d15pt{pytorch_version}cu{cuda_short}",
# ]
#
# [sources]
# index_urls = [
#     "https://pozzettiandrea.github.io/sam3dobjects-wheels/",
# ]


# ===========================================================================
# Node registration
# ===========================================================================

NODE_CLASS_MAPPINGS = {
    "SimpleIsolatedExample": SimpleIsolatedNode,
    "MultiMethodIsolatedExample": MultiMethodNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleIsolatedExample": "Simple Isolated (Decorator Example)",
    "MultiMethodIsolatedExample": "Multi-Method Isolated (Decorator Example)",
}
