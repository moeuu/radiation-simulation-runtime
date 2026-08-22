"""Tests for packaged shared runtime defaults."""

from __future__ import annotations

from pathlib import Path
import tomllib

import runtime_defaults

from runtime import defaults


def test_legacy_defaults_module_reexports_packaged_values() -> None:
    """The installable package should own legacy runtime default values."""
    names = tuple(defaults.__all__)

    assert names
    assert all(
        getattr(runtime_defaults, name) == getattr(defaults, name)
        for name in names
    )


def test_legacy_defaults_shim_is_declared_as_a_wheel_module() -> None:
    """Wheel installs must retain the pre-package compatibility import."""
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as handle:
        configuration = tomllib.load(handle)

    assert "runtime_defaults" in configuration["tool"]["setuptools"]["py-modules"]
