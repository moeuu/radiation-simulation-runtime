"""Tests for the sole packaged shared-runtime defaults namespace."""

from __future__ import annotations

from runtime import defaults


def test_runtime_defaults_are_owned_by_the_runtime_package() -> None:
    """The package namespace must expose every declared runtime default."""
    names = tuple(defaults.__all__)

    assert names
    assert all(hasattr(defaults, name) for name in names)
