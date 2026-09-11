"""The shape of 154 of 493 surveyed packs: `from server import PromptServer`
at the top of the module, before any node is defined, and an import-time
poke at the instance. Without a stand-in this line fails in a lean env with
ModuleNotFoundError (or, with aiohttp present, on PromptServer.instance being
None), before any node is defined.
"""

import threading

from server import BinaryEventTypes, PromptServer

# import-time use, as packs do: guarded the way they guard it
if PromptServer.instance is not None:
    PromptServer.instance.send_sync("server_user.loaded", {"at": "import"})


class ServerUser:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"x": ("INT", {"default": 0})}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"

    def run(self, x=0):
        inst = PromptServer.instance
        inst.send_sync("server_user.dict", {"x": x})                       # JSON
        inst.send_sync("server_user.targeted", {"x": x}, inst.client_id)    # sid = the client
        inst.send_sync(BinaryEventTypes.PREVIEW_IMAGE, b"\x89PNG\x00")      # bytes
        inst.send_progress_text("working", "7")
        try:
            from PIL import Image
            img = Image.new("RGB", (4, 4), (255, 0, 0))
            inst.send_sync(BinaryEventTypes.UNENCODED_PREVIEW_IMAGE, ("JPEG", img, 512))
        except ImportError:
            pass
        return (str(inst.client_id),)

    def from_thread(self, x=0):
        """A background thread calling send_sync: must be dropped, not
        interleaved with the call's own frames, and must not fail the node."""
        done = threading.Event()

        def poke():
            try:
                PromptServer.instance.send_sync("server_user.thread", {"x": x})
            finally:
                done.set()
        threading.Thread(target=poke).start()
        done.wait(5)
        return ("ok",)

    def off_surface(self, x=0):
        return (PromptServer.instance.prompt_queue,)
