"""Fail-closed integration test for the retired Geant4 collision shortcut."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def geant4_sidecar(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the complete sidecar once when Geant4 is available."""

    if shutil.which("g++") is None or shutil.which("geant4-config") is None:
        pytest.skip("g++ and geant4-config are required for this integration.")
    executable = tmp_path_factory.mktemp("force_collision_sidecar") / "sidecar"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_geant4_sidecar.py",
            "--profile",
            "portable",
            "--output",
            executable.as_posix(),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return executable


def test_unvalidated_forced_collision_fails_before_transport(
    geant4_sidecar: Path,
    tmp_path: Path,
) -> None:
    """The biased shortcut must not silently enter calibration or runtime."""

    response = tmp_path / "rejected.response"
    completed = subprocess.run(
        [
            geant4_sidecar.as_posix(),
            "--scene",
            (
                REPOSITORY_ROOT
                / "tests/native/force_collision_one_leaf.scene"
            ).as_posix(),
            "--request",
            (
                REPOSITORY_ROOT
                / "tests/native/force_collision_proof.request"
            ).as_posix(),
            "--response",
            response.as_posix(),
            "--physics-profile",
            "balanced",
            "--threads",
            "1",
            "--dead-time-tau-s",
            "0",
            "--source-rate-model",
            "detector_cps_1m",
            "--source-bias-mode",
            "detector_cone",
            "--detector-scoring-mode",
            "incident_gamma_energy",
            "--secondary-transport-mode",
            "full_transport",
            "--primary-sampling-fraction",
            "1",
            "--mean-calibration-histories-per-source-line",
            "256",
            "--mean-calibration-angle-strata-mu",
            "1",
            "--mean-calibration-angle-strata-phi",
            "1",
            "--validation-entry-class-spectra",
            "--background-cps",
            "0",
            "--mean-calibration-forced-collision",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "failed its analog-mean exactness test" in completed.stderr
    assert not response.exists()
