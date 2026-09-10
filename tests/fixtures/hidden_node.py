"""Node classes for the hidden-input contract test.

Deliberately plain: no comfy imports, no torch. The V3 shape is duck-typed --
the worker dispatches on `PREPARE_CLASS_CLONE` being present, which is exactly
what upstream's io.ComfyNode provides, so a stand-in with the same method
exercises the same branch without dragging ComfyUI into the test env.
"""


class V1SaveNode:
    """A V1 node in the shape of SaveImage: hidden values arrive as kwargs,
    under the author's own spelling, defaulting to None when absent."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"images": ("IMAGE",)},
            "hidden": {"prompt": "PROMPT",
                       "extra_pnginfo": "EXTRA_PNGINFO",
                       "node_id": "UNIQUE_ID"},
        }

    def save(self, images, prompt=None, extra_pnginfo=None, node_id=None):
        return {"images": images, "prompt": prompt,
                "extra_pnginfo": extra_pnginfo, "node_id": node_id}


class _HiddenHolder:
    """Stand-in for comfy_api's HiddenHolder: attribute per sentinel, and
    unknown attributes read as None rather than raising."""

    def __init__(self, d):
        self.prompt = d.get("PROMPT")
        self.extra_pnginfo = d.get("EXTRA_PNGINFO")
        self.unique_id = d.get("UNIQUE_ID")

    def __getattr__(self, _k):
        return None


class V3SaveNode:
    """A V3 node: hidden values arrive on the CLASS, not as kwargs. execute()
    would raise TypeError if handed prompt=, which is what the old strip was
    (correctly) protecting against on the V1-fallback path."""

    hidden = None

    @classmethod
    def PREPARE_CLASS_CLONE(cls, v3_data):
        clone = type(f"{cls.__name__}Clone", (cls,), {})
        clone.hidden = _HiddenHolder((v3_data or {}).get("hidden_inputs") or {})
        return clone

    @classmethod
    def execute(cls, images):
        return {"images": images,
                "prompt": cls.hidden.prompt,
                "extra_pnginfo": cls.hidden.extra_pnginfo,
                "unique_id": cls.hidden.unique_id}
