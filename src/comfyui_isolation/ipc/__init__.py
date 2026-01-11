"""IPC (Inter-Process Communication) for comfyui-isolation."""

from .bridge import WorkerBridge
from .worker import BaseWorker, register
from .protocol import (
    Request,
    Response,
    encode_object,
    decode_object,
    set_tensor_ipc_enabled,
    get_tensor_ipc_enabled,
)
from .transport import (
    Transport,
    UnixSocketTransport,
    StdioTransport,
    get_socket_path,
    cleanup_socket,
)

# TorchBridge is optional (requires PyTorch)
try:
    from .torch_bridge import TorchBridge, TorchWorker
    _TORCH_EXPORTS = ["TorchBridge", "TorchWorker"]
except ImportError:
    _TORCH_EXPORTS = []

# Tensor IPC is optional (requires PyTorch)
try:
    from .tensor import serialize_tensor, deserialize_tensor, TensorKeeper
    _TENSOR_EXPORTS = ["serialize_tensor", "deserialize_tensor", "TensorKeeper"]
except ImportError:
    _TENSOR_EXPORTS = []

__all__ = [
    "WorkerBridge",
    "BaseWorker",
    "register",
    "Request",
    "Response",
    "encode_object",
    "decode_object",
    "set_tensor_ipc_enabled",
    "get_tensor_ipc_enabled",
    # Transport
    "Transport",
    "UnixSocketTransport",
    "StdioTransport",
    "get_socket_path",
    "cleanup_socket",
] + _TORCH_EXPORTS + _TENSOR_EXPORTS
