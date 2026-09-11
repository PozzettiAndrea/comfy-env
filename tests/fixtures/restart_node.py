"""The crt-nodes shape: __init__ builds process-bound resources.

`CRT_ImageLoaderCrawlBatch` keeps a ThreadPoolExecutor and a threading.Lock on
self, built once in __init__, and its execute uses them on every call. Neither
pickles, so the parent holds only a marker for each, and a marker from a dead
process names a value that no longer exists anywhere. The old rule was that
__init__ runs when the PARENT says so, once per instance ever: after a
restart the new process had the marker, no executor, and no way to build one.
"""

import threading
from concurrent.futures import ThreadPoolExecutor


class RestartNode:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.lock = threading.Lock()
        self.calls = 0
        self.cache = {}

    def run(self, key="k"):
        with self.lock:
            self.calls += 1
        # a real use of the resource, so a stale marker cannot hide behind
        # an attribute the call never reads
        squared = self.executor.submit(lambda: self.calls ** 2).result()
        self.cache[key] = squared
        return {"calls": self.calls, "squared": squared,
                "cache_keys": sorted(self.cache)}


class PlainNode:
    """No process-bound state: a restart must NOT cost this node anything."""

    def __init__(self):
        self.calls = 0

    def run(self):
        self.calls += 1
        return {"calls": self.calls}
