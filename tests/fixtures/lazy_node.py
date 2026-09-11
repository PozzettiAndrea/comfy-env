"""A switch with two lazy inputs -- the case lazy evaluation exists for.

Upstream calls `check_lazy_status` before running the node, with the inputs it
already has and the lazy ones as None, and computes only what comes back. A
proxy without that method never gets asked, so the pruned branches are never
promoted and the node runs with BOTH of them None. This fixture is what the
forwarded call has to reproduce, round for round.
"""


class V1Switch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "select":   ("BOOLEAN",),
            "on_true":  ("*", {"lazy": True}),
            "on_false": ("*", {"lazy": True}),
        }}

    RETURN_TYPES = ("*",)
    FUNCTION = "pick"

    def check_lazy_status(self, select, on_true=None, on_false=None):
        """Ask for exactly the branch that will be taken, and only while it
        is still missing. Round one returns the name; round two, with the
        value present, returns nothing -- which is upstream's signal to run."""
        want = "on_true" if select else "on_false"
        have = on_true if select else on_false
        return [want] if have is None else []

    def pick(self, select, on_true=None, on_false=None):
        return (on_true if select else on_false,)
