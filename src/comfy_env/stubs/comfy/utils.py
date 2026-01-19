"""
Stub for comfy.utils in isolated worker processes.

Provides utility classes like ProgressBar without requiring
the full ComfyUI installation.
"""


class ProgressBar:
    """
    No-op progress bar for isolated workers.

    In isolated subprocess, we can't update the main ComfyUI progress bar,
    so this just tracks progress internally. Nodes can still use the same API.
    """

    def __init__(self, total):
        self.total = total
        self.current = 0

    def update(self, value):
        """Increment progress by value."""
        self.current += value

    def update_absolute(self, value, total=None, preview=None):
        """Set progress to absolute value."""
        self.current = value
        if total is not None:
            self.total = total
