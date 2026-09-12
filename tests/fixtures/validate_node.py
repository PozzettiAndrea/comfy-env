"""Nodes with a VALIDATE_INPUTS body, for the execute-time validation contract."""


class Ranged:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"x": ("INT", {"default": 0}),
                             "mesh": (["a.obj"], {})}}
    RETURN_TYPES = ("INT",)
    FUNCTION = "run"

    @classmethod
    def VALIDATE_INPUTS(cls, x):          # no **kwargs: `mesh` must be filtered out
        if x < 0:
            return f"x must be non-negative, got {x}"
        return True

    def run(self, x=0, mesh=None):
        return (x,)


class Blanket:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"x": ("INT", {"default": 0})}}
    RETURN_TYPES = ("INT",)
    FUNCTION = "run"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):   # sees everything, returns bare False
        return kwargs.get("x", 0) != 13

    def run(self, x=0):
        return (x,)


class Async:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"x": ("INT", {"default": 0})}}
    RETURN_TYPES = ("INT",)
    FUNCTION = "run"

    @classmethod
    async def VALIDATE_INPUTS(cls, x):
        return "async says no" if x == 7 else True

    def run(self, x=0):
        return (x,)


NODE_CLASS_MAPPINGS = {"Ranged": Ranged, "Blanket": Blanket, "Async": Async}
NODE_DISPLAY_NAME_MAPPINGS = {}
