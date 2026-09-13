"""Nodes that drive ComfyUI's own ProgressBar with a preview image."""


class Previewer:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"steps": ("INT", {"default": 3})}}
    RETURN_TYPES = ("INT",)
    FUNCTION = "run"

    def run(self, steps=3):
        import comfy.utils
        from PIL import Image
        bar = comfy.utils.ProgressBar(steps)
        for i in range(steps):
            img = Image.new("RGB", (40, 30), (i * 40, 0, 0))
            bar.update_absolute(i + 1, preview=("JPEG", img, 512))
        return (steps,)


class BigPreviewer:
    """One frame of 2048x2048 noise: several MB as PNG, well over the
    forwarded-preview cap at full size, a thumbnail once fitted."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"max_size": ("INT", {"default": 512}),
                             "fmt": ("STRING", {"default": "PNG"})}}
    RETURN_TYPES = ("INT",)
    FUNCTION = "run"

    def run(self, max_size=512, fmt="PNG"):
        import os
        import comfy.utils
        from PIL import Image
        img = Image.frombytes("RGB", (2048, 2048), os.urandom(2048 * 2048 * 3))
        bar = comfy.utils.ProgressBar(1)
        bar.update_absolute(1, preview=(fmt, img, max_size if max_size > 0 else None))
        return (1,)
