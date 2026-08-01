"""Repository ownership tests for simulation/estimator separation."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_simulation_repository_contains_no_estimator_package() -> None:
    """The common runtime must not acquire PF, MLE, or hybrid implementation."""
    forbidden = (
        ROOT / "src" / "pf",
        ROOT / "src" / "planning",
        ROOT / "src" / "three_d_estimation",
        ROOT / "src" / "orchestrator",
    )

    assert all(not path.exists() for path in forbidden)


def test_native_and_measurement_log_owners_are_present() -> None:
    """Production transport and raw-log publication live in this repository."""
    assert (ROOT / "native" / "geant4_sidecar" / "geant4_sidecar.cpp").is_file()
    assert (ROOT / "src" / "runtime" / "measurement_log.py").is_file()

