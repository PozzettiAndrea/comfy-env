"""A V3 node returning an expand graph, for the cls.SCHEMA fill-in contract."""

from comfy_api.latest import io


class Expander(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="Expander", inputs=[io.Int.Input("x")],
                         outputs=[io.Int.Output()], enable_expand=True)

    @classmethod
    def execute(cls, x):
        return io.NodeOutput(expand={"n1": {"class_type": "Nothing", "inputs": {"v": x}}})
