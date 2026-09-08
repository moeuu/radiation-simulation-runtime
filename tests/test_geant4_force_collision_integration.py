"""Fail-closed integration test for the retired Geant4 collision shortcut."""

from __future__ import annotations

from pathlib import Path
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_unvalidated_forced_collision_fails_before_transport(
    native_sidecar_executable: Path,
    tmp_path: Path,
) -> None:
    """The biased shortcut must not silently enter calibration or runtime."""

    response = tmp_path / "rejected.response"
    completed = subprocess.run(
        [
            native_sidecar_executable.as_posix(),
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
