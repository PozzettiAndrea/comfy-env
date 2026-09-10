"""A node whose state is the shape that used to kill the call.

`self.device`-style attributes are boilerplate in half the packs in the wild:
picklable, not JSON-native. They passed the shippability gate and then died on
the wire -- every call, forever, because the rollback meant the seed flag never
cleared. The tuple and the int-keyed dict cover the other half of the defect,
where the value DID cross and came back silently retyped.
"""


class StateNode:
    def __init__(self):
        self.calls = 0
        self.tup = (1, 2, 3)
        self.by_int = {1: "a"}
        self.blob = b"\x00\x01"
        self.tags = {"x", "y"}

    def run(self):
        self.calls += 1
        return {
            "calls": self.calls,
            "tup": self.tup,
            "tup_type": type(self.tup).__name__,
            "by_int_keys": sorted(type(k).__name__ for k in self.by_int),
            "blob_type": type(self.blob).__name__,
            "tags_type": type(self.tags).__name__,
        }
