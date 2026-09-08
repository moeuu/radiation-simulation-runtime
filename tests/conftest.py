"""Share expensive immutable test tools without sharing simulation state."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def _build_native_sidecar(
    tmp_path_factory: pytest.TempPathFactory,
    *,
    diagnostic: bool,
) -> Path:
    """Build one selected native tool in a disposable session directory."""
    if shutil.which("g++") is None or shutil.which("geant4-config") is None:
        pytest.skip("g++ and geant4-config are required for native integration.")
    root = Path(__file__).resolve().parents[1]
    executable = tmp_path_factory.mktemp("native_sidecar") / "sidecar"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_geant4_sidecar.py",
            "--profile",
            "portable",
            "--output",
            str(executable),
            *(["--trajectory-diagnostic"] if diagnostic else []),
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return executable


@pytest.fixture(scope="session")
def native_sidecar_executable(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Share only the immutable production executable across native tests."""
    return _build_native_sidecar(tmp_path_factory, diagnostic=False)


@pytest.fixture(scope="session")
def native_trajectory_executable(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Share the separate opt-in diagnostic executable across trajectory tests."""
    return _build_native_sidecar(tmp_path_factory, diagnostic=True)
