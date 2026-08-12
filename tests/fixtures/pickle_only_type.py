"""Plain class with NO registered serializer -- rides the pickle rung.

The parent deliberately never has this module importable: unpickling there
must degrade to OpaquePickle (held bytes), not an ImportError, so a bare
host (only comfy-env installed) can hold and forward the value.
"""


class PickleOnly:
    def __init__(self, n):
        self.n = n
