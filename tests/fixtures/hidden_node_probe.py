"""Reads the real class's `hidden` after a call, to prove the clone was
per-call and nothing was written onto the shared class."""


def real_class_hidden():
    import hidden_node
    return hidden_node.V3SaveNode.hidden
