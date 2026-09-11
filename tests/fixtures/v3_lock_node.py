"""Real io.ComfyNode subclasses, for the class-lock contract.

Needs a ComfyUI checkout on sys.path (the comfyui-marked tests put it there).
"""

from comfy_api.latest import io


class V3Mutator(io.ComfyNode):
    """Writes to `cls` inside execute: the one thing upstream's locked clone
    forbids. Plain ComfyUI raises AttributeError here on every run."""

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="V3Mutator",
                         inputs=[io.Int.Input("x")],
                         outputs=[io.Int.Output()])

    @classmethod
    def execute(cls, x):
        cls.leak = x
        return io.NodeOutput(x)


class V3HiddenReader(io.ComfyNode):
    """Declares no hidden inputs and reads cls.hidden anyway. Upstream always
    clones, so cls.hidden is a holder of Nones; a worker that skipped the
    clone when nothing was hidden left it as the class default, None itself,
    and this node died on NoneType."""

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="V3HiddenReader",
                         inputs=[io.Int.Input("x")],
                         outputs=[io.String.Output()])

    @classmethod
    def execute(cls, x):
        return io.NodeOutput(repr(cls.hidden.unique_id))
