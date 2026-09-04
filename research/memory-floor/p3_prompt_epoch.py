"""P3: the prompt epoch is readable without patching anything.

comfy-env used to class-patch ComfyUI's prompt tracker to learn when a new
prompt began. ComfyUI keeps the running prompt's id on a global progress
registry that it REPLACES per prompt, and comfy-env's own node code runs
inside prompt execution, so the same signal is simply readable.

Better than the patch on three counts: it carries ComfyUI's real prompt id
rather than a counter comfy-env increments, ComfyUI's own custom-node
unhooking pass cannot revert it, and the registry predates the prompt
tracker by more than a year, so it works on far more ComfyUI versions.
"""

import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import harness as H  # noqa: E402

H.bootstrap()


def main():
    r = H.Report("P3 prompt epoch without a patch")
    from comfy_env.isolation.workers.subprocess import _current_prompt_gen
    from comfy_execution.progress import (
        get_progress_state, reset_progress_state,
    )

    r.check("P3.1 no prompt running reads as None, not a frozen value",
            _current_prompt_gen() is None,
            "so workers use the decay fallback rather than one eternal prompt")

    class _DynPrompt:
        def get_node(self, node_id):
            return {}

    reset_progress_state("prompt-alpha", _DynPrompt())
    first = _current_prompt_gen()
    r.check("P3.1b the real prompt id is what crosses to the worker",
            first == "prompt-alpha", repr(first))

    again = _current_prompt_gen()
    r.check("P3.2 stable within one prompt",
            again == first, "two reads agree")

    reset_progress_state("prompt-beta", _DynPrompt())
    second = _current_prompt_gen()
    r.check("P3.3 a new prompt changes it, which is what retires old marks",
            second == "prompt-beta" and second != first,
            "{} -> {}".format(first, second))

    r.check("P3.4 nothing was patched to get any of this",
            getattr(get_progress_state(), "prompt_id", None) == "prompt-beta",
            "read straight off ComfyUI's own registry")
    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
