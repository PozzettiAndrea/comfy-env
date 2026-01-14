"""
Example worker script for isolated environment.

This runs in the isolated Python environment and handles inference requests.
"""

from comfy_env import BaseWorker, register


class ExampleWorker(BaseWorker):
    """Example worker that processes images."""

    def setup(self):
        """Called once when worker starts - load models here."""
        self.log("Loading model...")

        # Import heavy dependencies only in the isolated env
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.log(f"Using device: {self.device}")

        # In a real node, you'd load your model here:
        # self.model = load_my_model().to(self.device)

        self.log("Model loaded!")

    @register("echo")
    def echo(self, message: str):
        """Simple echo for testing."""
        return f"Echo: {message}"

    @register("process_image")
    def process_image(self, image, scale: float = 1.0):
        """
        Process an image.

        Args:
            image: PIL Image or numpy array
            scale: Scale factor

        Returns:
            Processed image
        """
        import numpy as np
        from PIL import Image

        self.log(f"Processing image with scale={scale}")

        # Convert to numpy if needed
        if hasattr(image, 'save'):  # PIL Image
            arr = np.array(image)
        else:
            arr = image

        # Example processing: simple brightness adjustment
        result = np.clip(arr * scale, 0, 255).astype(np.uint8)

        self.log("Processing complete")
        return Image.fromarray(result)

    @register("gpu_info")
    def get_gpu_info(self):
        """Get GPU information from the isolated environment."""
        import torch

        if not torch.cuda.is_available():
            return {"available": False}

        return {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
            "memory_allocated": torch.cuda.memory_allocated(0),
            "memory_cached": torch.cuda.memory_reserved(0),
        }


if __name__ == "__main__":
    ExampleWorker().run()
