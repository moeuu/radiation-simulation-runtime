"""Resolve physical runtime assets owned by this package's source repository."""

from __future__ import annotations

import os
from pathlib import Path


_ROOT_ENVIRONMENT_VARIABLE = "ROTATING_SHIELD_SIMULATION_RUNTIME_ROOT"


def simulation_runtime_root() -> Path:
    """Return the configured or editable-checkout shared-runtime root."""
    override = os.environ.get(_ROOT_ENVIRONMENT_VARIABLE)
    root = (
        Path(override).expanduser().resolve()
        if override
        else Path(__file__).resolve().parents[2]
    )
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError(
            "Shared runtime assets are unavailable. Set "
            f"{_ROOT_ENVIRONMENT_VARIABLE} to the runtime repository root."
        )
    return root


def standard_geant4_config_path() -> Path:
    """Return the authoritative standard no-GUI Geant4 configuration path."""
    path = (
        simulation_runtime_root()
        / "configs"
        / "geant4"
        / "variance_reduction_external_no_isaac_32threads.json"
    )
    if not path.is_file():
        raise RuntimeError(f"Standard shared-runtime config is missing: {path}")
    return path


__all__ = ["simulation_runtime_root", "standard_geant4_config_path"]
