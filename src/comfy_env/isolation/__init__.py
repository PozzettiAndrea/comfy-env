"""
Isolation layer - Process isolation for node execution.

Wraps node classes to execute in isolated subprocess environments.
"""

from .wrap import (
    wrap_isolated_nodes,
    wrap_nodes,
)
from .workers import (
    Worker,
    WorkerError,
    SubprocessWorker,
)
from .tensor_utils import (
    TensorKeeper,
    keep_tensor,
    keep_tensors_recursive,
    prepare_tensor_for_ipc,
    prepare_for_ipc_recursive,
)

__all__ = [
    # Node wrapping
    "wrap_isolated_nodes",
    "wrap_nodes",
    # Workers
    "Worker",
    "WorkerError",
    "SubprocessWorker",
    # Tensor utilities
    "TensorKeeper",
    "keep_tensor",
    "keep_tensors_recursive",
    "prepare_tensor_for_ipc",
    "prepare_for_ipc_recursive",
]
