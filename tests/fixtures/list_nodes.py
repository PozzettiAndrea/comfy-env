"""Nodes for the list-batching contract, V1 and V3 shapes.

PerItem takes one value; ComfyUI calls it once per element of a list.
Gather opts out (INPUT_IS_LIST) and receives the whole list in one call.
Fan returns a list and declares it (OUTPUT_IS_LIST) so downstream nodes
see the elements, not one list-valued output.
"""


class PerItem:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"x": ("INT", {})}}
    RETURN_TYPES = ("INT",)
    FUNCTION = "run"

    def run(self, x):
        return (x * 2,)


class Gather:
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"x": ("INT", {})}}
    RETURN_TYPES = ("INT", "STRING")
    FUNCTION = "run"

    def run(self, x):
        assert isinstance(x, list), type(x)
        return (len(x), ",".join(str(v) for v in x))


class Fan:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"n": ("INT", {})}}
    RETURN_TYPES = ("INT",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "run"

    def run(self, n):
        return (list(range(n)),)


try:
    from comfy_api.latest import io
except Exception:  # scanned without ComfyUI on the path: V1 nodes only
    io = None

if io is not None:
    class GatherV3(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id="GatherV3", inputs=[io.Int.Input("x")],
                             outputs=[io.Int.Output(), io.String.Output()],
                             is_input_list=True)

        @classmethod
        def execute(cls, x):
            assert isinstance(x, list), type(x)
            return io.NodeOutput(len(x), ",".join(str(v) for v in x))

    class FanV3(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id="FanV3", inputs=[io.Int.Input("n")],
                             outputs=[io.Int.Output(is_output_list=True)])

        @classmethod
        def execute(cls, n):
            return io.NodeOutput(list(range(n)))


NODE_CLASS_MAPPINGS = {"PerItem": PerItem, "Gather": Gather, "Fan": Fan}
if io is not None:
    NODE_CLASS_MAPPINGS.update({"GatherV3": GatherV3, "FanV3": FanV3})
