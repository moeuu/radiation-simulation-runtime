"""Real-Geant4 tests for persistent movable detector geometry."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from measurement.shielding import (
    SHIELD_POSE_CONTRACT_ID,
    SHIELD_POSE_CONTRACT_SHA256,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = (
    REPOSITORY_ROOT / "tests/native/force_collision_thin_leaf.scene"
)


@dataclass(frozen=True)
class SidecarResult:
    """Hold the response fields used by persistent-geometry assertions."""

    metadata: dict[str, str]
    spectrum: tuple[float, ...]
    variance: tuple[float, ...]


@pytest.fixture(scope="module")
def geant4_sidecar(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the complete native sidecar once when Geant4 is available."""

    if shutil.which("g++") is None or shutil.which("geant4-config") is None:
        pytest.skip("g++ and geant4-config are required for this integration.")
    executable = (
        tmp_path_factory.mktemp("persistent_geometry_sidecar") / "sidecar"
    )
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


def _write_request(
    path: Path,
    *,
    step_id: int,
    seed: int,
    detector_x_m: float,
) -> None:
    """Write one deterministic detector-pose request."""

    path.write_text(
        "\n".join(
            (
                (
                    f"STEP step_id={step_id} dwell_time_s=1 seed={seed} "
                    f"shield_pose_contract_id={SHIELD_POSE_CONTRACT_ID} "
                    "shield_pose_contract_sha256="
                    f"{SHIELD_POSE_CONTRACT_SHA256} "
                    "fe_orientation_index=7 pb_orientation_index=7"
                ),
                (
                    "POSE kind=detector "
                    f"x={detector_x_m} y=0 z=0 "
                    "qw=1 qx=0 qy=0 qz=0"
                ),
                (
                    "POSE kind=fe "
                    f"x={detector_x_m} y=0 z=0 "
                    "qw=1 qx=0 qy=0 qz=0"
                ),
                (
                    "POSE kind=pb "
                    f"x={detector_x_m} y=0 z=0 "
                    "qw=1 qx=0 qy=0 qz=0"
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )


def _parse_response(path: Path) -> SidecarResult:
    """Parse native response metadata and spectrum arrays."""

    metadata: dict[str, str] = {}
    spectrum: tuple[float, ...] = ()
    variance: tuple[float, ...] = ()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("META "):
            key, value = line[5:].split("=", maxsplit=1)
            metadata[key] = value
        elif line.startswith("SPECTRUM "):
            spectrum = tuple(float(value) for value in line[9:].split(","))
        elif line.startswith("SPECTRUM_VARIANCE "):
            variance = tuple(float(value) for value in line[18:].split(","))
    assert spectrum
    assert len(variance) == len(spectrum)
    return SidecarResult(metadata, spectrum, variance)


def _sidecar_arguments(
    executable: Path,
    *,
    threads: int,
    secondary_transport_mode: str,
) -> list[str]:
    """Return standard unit-weight observation arguments."""

    return [
        executable.as_posix(),
        "--physics-profile",
        "balanced",
        "--threads",
        str(threads),
        "--dead-time-tau-s",
        "5.813e-9",
        "--source-rate-model",
        "detector_cps_1m",
        "--source-bias-mode",
        "detector_cone",
        "--detector-scoring-mode",
        "incident_gamma_energy",
        "--secondary-transport-mode",
        secondary_transport_mode,
        "--primary-sampling-fraction",
        "1",
        "--sample-detector-response",
        "--background-cps",
        "12",
    ]


def _run_fresh(
    executable: Path,
    request_path: Path,
    response_path: Path,
    *,
    threads: int,
    secondary_transport_mode: str = "full_transport",
) -> SidecarResult:
    """Run one request in a newly initialized Geant4 process."""

    completed = subprocess.run(
        [
            *_sidecar_arguments(
                executable,
                threads=threads,
                secondary_transport_mode=secondary_transport_mode,
            ),
            "--scene",
            SCENE_PATH.as_posix(),
            "--request",
            request_path.as_posix(),
            "--response",
            response_path.as_posix(),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return _parse_response(response_path)


def _run_persistent_pair(
    executable: Path,
    first_request: Path,
    second_request: Path,
    first_response: Path,
    second_response: Path,
    *,
    threads: int,
    secondary_transport_mode: str = "full_transport",
) -> tuple[SidecarResult, SidecarResult, str]:
    """Run two poses through one persistent Geant4 session."""

    command = [
        *_sidecar_arguments(
            executable,
            threads=threads,
            secondary_transport_mode=secondary_transport_mode,
        ),
        "--persistent",
    ]
    input_payload = "\n".join(
        (
            (
                "RUN "
                f"scene={SCENE_PATH.as_posix()} "
                f"request={first_request.as_posix()} "
                f"response={first_response.as_posix()}"
            ),
            (
                "RUN "
                f"scene={SCENE_PATH.as_posix()} "
                f"request={second_request.as_posix()} "
                f"response={second_response.as_posix()}"
            ),
            "SHUTDOWN",
        )
    )
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        input=input_payload + "\n",
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.count("SIMBRIDGE_OK response=") == 2
    return (
        _parse_response(first_response),
        _parse_response(second_response),
        completed.stdout,
    )


@pytest.mark.parametrize("threads", (1, 16, 32))
def test_persistent_pose_update_reuses_session_and_matches_fresh_transport(
    geant4_sidecar: Path,
    tmp_path: Path,
    threads: int,
) -> None:
    """A moved detector must match a fresh session at 1/16/32 threads."""

    first_request = tmp_path / f"first_{threads}.request"
    second_request = tmp_path / f"second_{threads}.request"
    _write_request(first_request, step_id=0, seed=1949, detector_x_m=3.0)
    _write_request(second_request, step_id=1, seed=2753, detector_x_m=2.5)
    first, cached, _ = _run_persistent_pair(
        geant4_sidecar,
        first_request,
        second_request,
        tmp_path / f"first_{threads}.response",
        tmp_path / f"cached_{threads}.response",
        threads=threads,
    )
    fresh = _run_fresh(
        geant4_sidecar,
        second_request,
        tmp_path / f"fresh_{threads}.response",
        threads=threads,
    )

    assert first.metadata["geometry_cache_hit"] == "false"
    assert first.metadata["movable_geometry_updated"] == "false"
    assert cached.metadata["geometry_cache_hit"] == "true"
    assert cached.metadata["movable_geometry_updated"] == "true"
    assert cached.metadata["transport_history_mode"] == "full_unit_weight"
    assert cached.metadata["transport_tally_weighted"] == "false"
    assert fresh.metadata["geometry_cache_hit"] == "false"
    assert fresh.metadata["movable_geometry_updated"] == "false"
    if threads == 1:
        assert cached.spectrum == fresh.spectrum
        assert cached.variance == fresh.variance
    else:
        difference = abs(math.fsum(cached.spectrum) - math.fsum(fresh.spectrum))
        standard_error = math.sqrt(
            math.fsum(cached.variance) + math.fsum(fresh.variance)
        )
        assert difference <= 6.0 * max(1.0, standard_error)


def test_gamma_only_stacking_uses_the_moved_detector_center(
    geant4_sidecar: Path,
    tmp_path: Path,
) -> None:
    """Gamma-only secondary filtering must follow the runtime detector pose."""

    first_request = tmp_path / "gamma_first.request"
    second_request = tmp_path / "gamma_second.request"
    _write_request(first_request, step_id=0, seed=3253, detector_x_m=3.0)
    _write_request(second_request, step_id=1, seed=4093, detector_x_m=2.5)
    _, cached, _ = _run_persistent_pair(
        geant4_sidecar,
        first_request,
        second_request,
        tmp_path / "gamma_first.response",
        tmp_path / "gamma_cached.response",
        threads=1,
        secondary_transport_mode="gamma_only",
    )
    fresh = _run_fresh(
        geant4_sidecar,
        second_request,
        tmp_path / "gamma_fresh.response",
        threads=1,
        secondary_transport_mode="gamma_only",
    )

    assert cached.spectrum == fresh.spectrum
    assert cached.variance == fresh.variance
    assert cached.metadata["geometry_cache_hit"] == "true"
    assert cached.metadata["movable_geometry_updated"] == "true"


def test_persistent_pose_outside_world_fails_before_transport(
    geant4_sidecar: Path,
    tmp_path: Path,
) -> None:
    """A later pose outside the fixed world must fail closed."""

    valid_request = tmp_path / "valid.request"
    invalid_request = tmp_path / "invalid.request"
    valid_response = tmp_path / "valid.response"
    invalid_response = tmp_path / "invalid.response"
    _write_request(valid_request, step_id=0, seed=5011, detector_x_m=3.0)
    _write_request(invalid_request, step_id=1, seed=5011, detector_x_m=100.0)
    command = [
        *_sidecar_arguments(
            geant4_sidecar,
            threads=1,
            secondary_transport_mode="full_transport",
        ),
        "--persistent",
    ]
    input_payload = "\n".join(
        (
            (
                "RUN "
                f"scene={SCENE_PATH.as_posix()} "
                f"request={valid_request.as_posix()} "
                f"response={valid_response.as_posix()}"
            ),
            (
                "RUN "
                f"scene={SCENE_PATH.as_posix()} "
                f"request={invalid_request.as_posix()} "
                f"response={invalid_response.as_posix()}"
            ),
            "SHUTDOWN",
        )
    )
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        input=input_payload + "\n",
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert valid_response.is_file()
    assert not invalid_response.exists()
    assert "SIMBRIDGE_ERR Detector pose lies outside" in completed.stdout


@pytest.mark.parametrize(
    "unsupported_option",
    ("--exact-free-flight", "--first-flight-forcing"),
)
def test_actual_observation_free_flight_switches_fail_closed(
    geant4_sidecar: Path,
    unsupported_option: str,
) -> None:
    """Unproven unit-weight free-flight routes must remain unselectable."""

    completed = subprocess.run(
        [geant4_sidecar.as_posix(), unsupported_option],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert (
        f"Unsupported Geant4 sidecar option: {unsupported_option}"
        in completed.stderr
    )
