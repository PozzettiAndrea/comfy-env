"""Contract: unquoted `python = 3.10` (TOML float -> 3.1) is rejected."""

import pytest

from comfy_env.config import parse_config


def test_float_python_is_rejected():
    with pytest.raises(ValueError, match=r'python = 3\.1 .*[Qq]uote'):
        parse_config({"python": 3.10})  # TOML would deliver the float 3.1


def test_int_python_is_rejected():
    with pytest.raises(ValueError, match="not a string"):
        parse_config({"python": 3})


def test_quoted_python_ok():
    assert parse_config({"python": "3.10"}).python == "3.10"


def test_python_below_310_is_rejected():
    """comfy-env supports 3.10+ only: ComfyUI itself is requires-python
    >= 3.10, no fleet pack pins lower, and the worker program is only kept
    parseable down to 3.10. A lower pin must die at config load, not at
    worker startup after a multi-GB env build."""
    for pin in ("3.9", "3.9.*", "==3.8", ">=3.9,<3.10"):
        with pytest.raises(ValueError, match="below 3.10"):
            parse_config({"python": pin})


def test_python_310_and_up_pass_the_floor():
    for pin in ("3.10", "3.12", "3.13.*", ">=3.10"):
        assert parse_config({"python": pin}).python == pin
