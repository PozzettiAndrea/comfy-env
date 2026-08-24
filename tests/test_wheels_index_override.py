"""Contract: the cuda-wheels index is overridable, for mirrors and air-gapped sites."""

import pytest

import comfy_env.packages.cuda_wheels as cw


def test_defaults_to_the_public_index(monkeypatch):
    monkeypatch.delenv("COMFY_ENV_CUDA_WHEELS_INDEX", raising=False)
    assert cw.cuda_wheels_index() == cw.CUDA_WHEELS_INDEX_DEFAULT


def test_env_var_overrides(monkeypatch):
    monkeypatch.setenv("COMFY_ENV_CUDA_WHEELS_INDEX", "https://mirror.internal/wheels/v2/")
    assert cw.cuda_wheels_index() == "https://mirror.internal/wheels/v2/"


def test_missing_trailing_slash_is_repaired(monkeypatch):
    """Every call site does f"{index}{pkg}/" -- without this, an override
    without a slash silently builds ".../v2flash_attn/" and 404s."""
    monkeypatch.setenv("COMFY_ENV_CUDA_WHEELS_INDEX", "https://mirror.internal/wheels/v2")
    assert cw.cuda_wheels_index().endswith("/v2/")


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_falls_back_to_the_default(monkeypatch, blank):
    """An empty value in settings.env must not produce a relative URL."""
    monkeypatch.setenv("COMFY_ENV_CUDA_WHEELS_INDEX", blank)
    assert cw.cuda_wheels_index() == cw.CUDA_WHEELS_INDEX_DEFAULT


def test_override_reaches_the_url_builder(monkeypatch):
    """The point of the setting: lookups actually go to the mirror."""
    monkeypatch.setenv("COMFY_ENV_CUDA_WHEELS_INDEX", "https://mirror.internal/w/")
    urls = cw.get_find_links_urls("flash-attn")
    assert urls and all(u.startswith("https://mirror.internal/w/") for u in urls), urls


def test_resolve_index_url_honours_the_override(monkeypatch):
    monkeypatch.setenv("COMFY_ENV_CUDA_WHEELS_INDEX", "https://mirror.internal/w/")
    assert cw.resolve_index_url("cuda") == "https://mirror.internal/w/"
    with pytest.raises(ValueError, match="rocm"):
        cw.resolve_index_url("rocm")
