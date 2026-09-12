"""Fixture for the log-level and models_dir contracts."""
import logging


def emit_at_every_level():
    """Returns nothing useful; the test watches what the host RECEIVES."""
    logging.debug("DEBUG-LINE")
    logging.info("INFO-LINE")
    logging.warning("WARNING-LINE")
    return "done"


def read_models_dir():
    import folder_paths
    return folder_paths.models_dir
