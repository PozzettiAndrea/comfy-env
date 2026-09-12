"""A node that holds the main lane for a while, plus a combo and a fingerprint,
for the side-lane contract: cheap questions are answered while it runs."""

import os
import time


class Slow:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"seconds": ("FLOAT", {"default": 1.0}),
                             "mesh": (sorted(os.listdir(os.environ.get("SLOW_NODE_DIR", "."))), {})}}
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "run"

    @classmethod
    def IS_CHANGED(cls, seconds=1.0, mesh=None):
        return f"fp:{mesh}"

    def run(self, seconds=1.0, mesh=None):
        time.sleep(seconds)
        return (seconds,)


class Spinner(Slow):
    """CPU-bound in pure Python: the worst honest case for a side request."""

    def run(self, seconds=1.0, mesh=None):
        end = time.perf_counter() + seconds
        x = 0
        while time.perf_counter() < end:
            x += 1
        return (float(x),)


class Sleeper(Slow):
    """A fingerprint that takes longer than a side reply cap."""

    @classmethod
    def IS_CHANGED(cls, seconds=1.0, mesh=None):
        time.sleep(0.5)
        return "slept"
