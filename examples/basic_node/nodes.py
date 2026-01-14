"""
Example ComfyUI node that uses process isolation.

This demonstrates how to use comfyui-isolation in a ComfyUI custom node.

Two approaches are shown:
1. Programmatic configuration (IsolatedEnv directly in code)
2. Declarative configuration (comfy_env_reqs.toml file)
"""

from pathlib import Path
from comfy_env import IsolatedEnv, WorkerBridge, detect_cuda_version

# Get the node's root directory
NODE_ROOT = Path(__file__).parent


# ===========================================================================
# APPROACH 1: Programmatic Configuration (traditional)
# ===========================================================================
# Define the isolated environment configuration in code
ENV_CONFIG = IsolatedEnv(
    name="example-node",
    python="3.10",
    cuda=detect_cuda_version(),  # Auto-detect (12.8 for Blackwell, 12.4 for others)
    requirements=[
        "torch",
        "torchvision",
        "pillow",
        "numpy",
    ],
    # Add custom wheel sources if needed:
    # wheel_sources=["https://my-wheels.github.io/"],
)

# Create bridge singleton
_bridge = None


def get_bridge() -> WorkerBridge:
    """Get or create the worker bridge singleton (programmatic approach)."""
    global _bridge
    if _bridge is None:
        _bridge = WorkerBridge(
            env=ENV_CONFIG,
            worker_script=NODE_ROOT / "worker.py",
            base_dir=NODE_ROOT,
        )
    return _bridge


# ===========================================================================
# APPROACH 2: Declarative Configuration (recommended for new projects)
# ===========================================================================
# Instead of the above, you can use a comfy_env_reqs.toml file:
#
# def get_bridge() -> WorkerBridge:
#     """Get or create the worker bridge singleton (config file approach)."""
#     global _bridge
#     if _bridge is None:
#         _bridge = WorkerBridge.from_config_file(
#             node_dir=NODE_ROOT,
#             worker_script=NODE_ROOT / "worker.py",
#         )
#     return _bridge
#
# This auto-discovers comfy_env_reqs.toml in the node directory.
# See the example file in this directory for the format.


class IsolatedProcessorNode:
    """
    Example node that processes images in an isolated environment.

    This node runs inference in a separate Python process with its own
    dependencies, avoiding conflicts with ComfyUI's environment.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "example/isolation"

    def process(self, image, scale):
        """Process image in isolated environment."""
        import torch
        import numpy as np
        from PIL import Image

        bridge = get_bridge()

        # Ensure environment is set up (first run only)
        bridge.ensure_environment(verify_packages=["torch"])

        # Convert ComfyUI image format to PIL
        # ComfyUI uses (B, H, W, C) float32 tensors
        img_np = (image[0].numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)

        # Call worker method
        result = bridge.call("process_image", image=pil_image, scale=scale)

        # Convert back to ComfyUI format
        result_np = np.array(result).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np).unsqueeze(0)

        return (result_tensor,)


class SetupIsolatedEnvNode:
    """
    Node to set up the isolated environment.

    Run this once to create the environment before using other nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("*",),  # Any input to trigger
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "setup"
    CATEGORY = "example/isolation"

    def setup(self, trigger):
        """Set up the isolated environment."""
        bridge = get_bridge()

        try:
            bridge.ensure_environment(verify_packages=["torch", "numpy"])
            return (f"Environment ready: {bridge.python_exe}",)
        except Exception as e:
            return (f"Setup failed: {e}",)


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "IsolatedProcessorExample": IsolatedProcessorNode,
    "SetupIsolatedEnvExample": SetupIsolatedEnvNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IsolatedProcessorExample": "Isolated Processor (Example)",
    "SetupIsolatedEnvExample": "Setup Isolated Env (Example)",
}
