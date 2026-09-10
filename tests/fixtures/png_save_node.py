"""A pack-style save node that writes a REAL PNG from its hidden inputs.

Mirrors upstream SaveImage's shape exactly, including the `if x is not None`
guards -- which is why the original bug was silent: with the values stripped
both guards simply skipped, and a perfectly valid image came out with no
metadata in it.
"""
import json

from PIL import Image
from PIL.PngImagePlugin import PngInfo


class PackSaveImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING",)},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}}

    def save(self, path, prompt=None, extra_pnginfo=None):
        md = PngInfo()
        if prompt is not None:
            md.add_text("prompt", json.dumps(prompt))
        if extra_pnginfo is not None:
            for k in extra_pnginfo:
                md.add_text(k, json.dumps(extra_pnginfo[k]))
        Image.new("RGB", (4, 4), (255, 0, 0)).save(path, pnginfo=md)
        return path
