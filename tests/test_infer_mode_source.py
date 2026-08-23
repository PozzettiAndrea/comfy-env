"""The worker crosses the boundary as source text (ADR-0006), so the default it
selects is assertable without importing torch -- and must be, since CI lanes
without torch are exactly where a silent revert would slip through.

Rationale for the choice lives in tests/test_infer_mode.py.
"""

import re
from pathlib import Path

WORKER_SRC = (
    Path(__file__).resolve().parents[1]
    / "src/comfy_env/isolation/workers/_persistent_worker.py"
)


def test_worker_selects_no_grad_not_inference_mode():
    src = WORKER_SRC.read_text(encoding="utf-8")
    assert re.search(r"_infer_mode\s*=\s*_torch_worker\.no_grad", src), \
        "worker must default to torch.no_grad()"
    assert not re.search(r"_infer_mode\s*=\s*_torch_worker\.inference_mode", src), \
        "worker must not use torch.inference_mode() -- see tests/test_infer_mode.py"
