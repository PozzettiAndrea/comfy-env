"""
IPC Protocol - Message format for bridge-worker communication.

Uses JSON for simplicity and debuggability. Large binary data (images, tensors)
is serialized efficiently:
- Tensors: Zero-copy via CUDA IPC or shared memory (see tensor.py)
- Images: PNG encoded + base64
- Other: pickle + base64 fallback
"""

import json
import base64
import pickle
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Flag to enable/disable IPC tensor sharing (set based on process context)
_use_tensor_ipc = True


def set_tensor_ipc_enabled(enabled: bool) -> None:
    """Enable or disable IPC tensor sharing."""
    global _use_tensor_ipc
    _use_tensor_ipc = enabled


def get_tensor_ipc_enabled() -> bool:
    """Check if IPC tensor sharing is enabled."""
    return _use_tensor_ipc


@dataclass
class Request:
    """
    Request message from bridge to worker.

    Attributes:
        id: Unique request ID for matching responses
        method: Method name to call on worker
        args: Keyword arguments for the method
    """
    id: str
    method: str
    args: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "Request":
        """Deserialize from JSON string."""
        d = json.loads(data)
        return cls(**d)


@dataclass
class Response:
    """
    Response message from worker to bridge.

    Attributes:
        id: Request ID this is responding to
        result: Result value (None if error)
        error: Error message (None if success)
        traceback: Full traceback string (only if error)
    """
    id: str
    result: Any = None
    error: Optional[str] = None
    traceback: Optional[str] = None

    @property
    def success(self) -> bool:
        """Check if response indicates success."""
        return self.error is None

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "Response":
        """Deserialize from JSON string."""
        d = json.loads(data)
        return cls(**d)


def encode_binary(data: bytes) -> str:
    """Encode binary data as base64 string."""
    return base64.b64encode(data).decode('utf-8')


def decode_binary(encoded: str) -> bytes:
    """Decode base64 string to binary data."""
    return base64.b64decode(encoded)


def encode_object(obj: Any) -> Dict[str, Any]:
    """
    Encode a Python object for JSON serialization.

    Returns a dict with _type and _data keys for special types,
    or the original object if it's JSON-serializable.

    Special handling:
    - PyTorch tensors: Zero-copy via CUDA IPC or shared memory
    - PIL Images: PNG encoded
    - Complex objects: pickle fallback
    """
    if obj is None:
        return None

    # Handle torch tensors - try zero-copy IPC first
    if hasattr(obj, 'cpu') and hasattr(obj, 'numpy'):
        try:
            import torch
            if isinstance(obj, torch.Tensor) and _use_tensor_ipc:
                try:
                    from .tensor import serialize_tensor
                    return serialize_tensor(obj)
                except Exception as e:
                    # Fall back to pickle method if IPC fails
                    logger.debug(f"Tensor IPC failed, using pickle: {e}")
        except ImportError:
            pass

        # Fallback: pickle the numpy array
        arr = obj.cpu().numpy()
        return {
            "_type": "tensor_pickle",
            "_dtype": str(arr.dtype),
            "_shape": list(arr.shape),
            "_data": encode_binary(pickle.dumps(arr)),
        }

    # Handle numpy arrays
    if hasattr(obj, '__array__'):
        import numpy as np
        arr = np.asarray(obj)
        return {
            "_type": "numpy",
            "_dtype": str(arr.dtype),
            "_shape": list(arr.shape),
            "_data": encode_binary(pickle.dumps(arr)),
        }

    # Handle PIL Images
    if hasattr(obj, 'save') and hasattr(obj, 'mode'):
        import io
        buffer = io.BytesIO()
        obj.save(buffer, format="PNG")
        return {
            "_type": "image",
            "_format": "PNG",
            "_data": encode_binary(buffer.getvalue()),
        }

    # Handle bytes
    if isinstance(obj, bytes):
        return {
            "_type": "bytes",
            "_data": encode_binary(obj),
        }

    # Handle lists/tuples recursively
    if isinstance(obj, (list, tuple)):
        encoded = [encode_object(item) for item in obj]
        return {
            "_type": "list" if isinstance(obj, list) else "tuple",
            "_data": encoded,
        }

    # Handle dicts recursively
    if isinstance(obj, dict):
        return {k: encode_object(v) for k, v in obj.items()}

    # For simple objects with __dict__, serialize as dict
    # This avoids pickle module path issues across process boundaries
    if hasattr(obj, '__dict__') and not hasattr(obj, '__slots__'):
        return {
            "_type": "object",
            "_class": obj.__class__.__name__,
            "_data": {k: encode_object(v) for k, v in obj.__dict__.items()},
        }

    # For complex objects that can't be JSON serialized, use pickle
    try:
        json.dumps(obj)
        return obj  # JSON-serializable, return as-is
    except (TypeError, ValueError):
        return {
            "_type": "pickle",
            "_data": encode_binary(pickle.dumps(obj)),
        }


def decode_object(obj: Any) -> Any:
    """
    Decode a JSON-deserialized object back to Python types.

    Reverses the encoding done by encode_object.
    """
    if obj is None:
        return None

    if not isinstance(obj, dict):
        return obj

    # Check for special encoded types
    obj_type = obj.get("_type")

    # Handle zero-copy tensor IPC
    if obj_type == "tensor_ipc":
        try:
            from .tensor import deserialize_tensor
            return deserialize_tensor(obj)
        except Exception as e:
            logger.error(f"Failed to deserialize tensor via IPC: {e}")
            raise

    # Handle pickle fallback for tensors
    if obj_type == "tensor_pickle":
        import torch
        arr = pickle.loads(decode_binary(obj["_data"]))
        return torch.from_numpy(arr)

    # Legacy types for backwards compatibility
    if obj_type == "numpy":
        return pickle.loads(decode_binary(obj["_data"]))

    if obj_type in ("tensor", "comfyui_image", "comfyui_mask"):
        import torch
        arr = pickle.loads(decode_binary(obj["_data"]))
        return torch.from_numpy(arr)

    if obj_type == "image":
        import io
        from PIL import Image
        buffer = io.BytesIO(decode_binary(obj["_data"]))
        return Image.open(buffer)

    if obj_type == "bytes":
        return decode_binary(obj["_data"])

    if obj_type == "pickle":
        return pickle.loads(decode_binary(obj["_data"]))

    # Simple object serialized as dict - restore as SimpleNamespace
    if obj_type == "object":
        from types import SimpleNamespace
        data = {k: decode_object(v) for k, v in obj["_data"].items()}
        ns = SimpleNamespace(**data)
        ns._class_name = obj.get("_class", "unknown")
        return ns

    if obj_type in ("list", "tuple"):
        decoded = [decode_object(item) for item in obj["_data"]]
        return decoded if obj_type == "list" else tuple(decoded)

    # Regular dict - decode values recursively
    return {k: decode_object(v) for k, v in obj.items()}
