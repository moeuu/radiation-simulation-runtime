"""Tests for shared physical asset ownership."""

from __future__ import annotations

from runtime.assets import simulation_runtime_root, standard_geant4_config_path


def test_standard_config_is_owned_by_shared_runtime() -> None:
    """The standard Geant4 config must resolve inside this repository."""
    root = simulation_runtime_root()
    config = standard_geant4_config_path()
    assert config.is_relative_to(root)
    assert config.is_file()
