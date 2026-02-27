"""
Isolation layer - Process isolation for node execution.

Wraps node classes to execute in isolated subprocess environments.
"""

from .wrap import (
    register_nodes,
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
    release_tensor,
    release_tensors_recursive,
)

__all__ = [
    # Node registration
    "register_nodes",
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
    "release_tensor",
    "release_tensors_recursive",
]
