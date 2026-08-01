"""Tests for the Geant4 paired all-64 capture and replay integration."""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def geant4_integration_driver(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Compile the real Geant4 parallel-world integration driver."""

    compiler = shutil.which("g++")
    geant4_config = shutil.which("geant4-config")
    if compiler is None or geant4_config is None:
        pytest.skip(
            "g++ and geant4-config are required for the Geant4 integration."
        )
    repository_root = Path(__file__).resolve().parents[1]
    output_path = (
        tmp_path_factory.mktemp("paired_all64_geant4") / "driver"
    )
    cflags = shlex.split(
        subprocess.run(
            [geant4_config, "--cflags"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    libraries = shlex.split(
        subprocess.run(
            [geant4_config, "--libs"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    command = [
        compiler,
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-Werror",
        *cflags,
        "-I",
        str(repository_root / "native/geant4_sidecar"),
        str(
            repository_root
            / "tests/native/paired_all64_geant4_test_driver.cpp"
        ),
        str(
            repository_root
            / "native/geant4_sidecar/paired_all64_geant4.cpp"
        ),
        str(
            repository_root
            / "native/geant4_sidecar/paired_all64_phase_space.cpp"
        ),
        "-o",
        str(output_path),
        *libraries,
    ]
    subprocess.run(
        command,
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return output_path


def test_parallel_world_captures_exact_first_inward_state(
    geant4_integration_driver: Path,
) -> None:
    """The boundary state, zero history, replay, and all-64 scores must pass."""

    result = subprocess.run(
        [str(geant4_integration_driver), "normal"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "paired_all64_geant4_ok" in result.stdout


def test_non_gamma_transport_state_is_preserved(
    geant4_integration_driver: Path,
) -> None:
    """An electron crossing must retain species, mass, and charge for replay."""

    result = subprocess.run(
        [str(geant4_integration_driver), "non_gamma"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "paired_all64_geant4_electron_ok" in result.stdout


def test_positron_transport_state_is_preserved(
    geant4_integration_driver: Path,
) -> None:
    """A positron crossing must retain species, mass, and charge for replay."""

    result = subprocess.run(
        [str(geant4_integration_driver), "positron"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "paired_all64_geant4_positron_ok" in result.stdout


def test_worker_local_capture_merges_complete_mt_history_set(
    geant4_integration_driver: Path,
) -> None:
    """Two Geant4 workers must retain the same histories and boundary state."""

    result = subprocess.run(
        [str(geant4_integration_driver), "mt"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "paired_all64_geant4_mt_ok" in result.stdout


def test_weighted_boundary_track_fails_closed(
    geant4_integration_driver: Path,
) -> None:
    """A weighted capture branch must never produce an authenticated bank."""

    result = subprocess.run(
        [str(geant4_integration_driver), "weighted"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert (
        "Exact paired replay rejects weighted boundary crossings."
        in result.stderr
    )


def test_integration_remains_outside_standard_sidecar_source() -> None:
    """The dedicated profile must not become a standard-runtime option."""

    repository_root = Path(__file__).resolve().parents[1]
    standard_sidecar = (
        repository_root / "native/geant4_sidecar/geant4_sidecar.cpp"
    ).read_text(encoding="utf-8")
    assert "PairedAll64CaptureParallelWorld" not in standard_sidecar
    assert "geant4_phase_space_paired_all64_v3" not in standard_sidecar


def test_replay_resolves_the_serialized_particle_definition() -> None:
    """Replay must not replace all captured species with gamma primaries."""

    repository_root = Path(__file__).resolve().parents[1]
    integration_source = (
        repository_root / "native/geant4_sidecar/paired_all64_geant4.cpp"
    ).read_text(encoding="utf-8")
    assert "FindParticle(" in integration_source
    assert "crossing.particle_name" in integration_source
    assert "new G4PrimaryParticle(G4Gamma" not in integration_source
