"""Share expensive immutable test tools without sharing simulation state."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture(scope="session")
def native_sidecar_executable(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile one portable sidecar per pytest invocation for native tests."""
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
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return executable
