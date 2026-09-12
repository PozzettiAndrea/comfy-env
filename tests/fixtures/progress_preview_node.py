"""A node that drives ComfyUI's own ProgressBar with a preview image."""


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
