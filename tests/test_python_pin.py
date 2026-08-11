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
