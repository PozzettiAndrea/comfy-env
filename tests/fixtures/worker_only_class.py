"""Plain classes for worker-only serializer fixtures.

Kept SEPARATE from the serializer files (like a real pack, where the class
comes from a library such as trimesh): the serializer file is loaded by
path under a mangled module name, while node code imports the class via
sys.path -- both must resolve to the SAME class object for isinstance.
"""


class WorkerOnly:
    def __init__(self, secret):
        self.secret = secret


class WorkerOnlyArr:
    def __init__(self, data):
        self.data = data  # numpy array
