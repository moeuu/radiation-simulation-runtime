"""Real-Geant4 contracts for evaluated decay cascades and coincidence sums."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from measurement.source_boundary import (
    SURFACE_EMISSION_EPSILON_M,
    surface_emission_policy_sha256,
)
from sim.geant4_app.io_format import write_request_file, write_scene_file
from sim.geant4_app.engine import Geant4StepRequest
from sim.geant4_app.scene_export import (
    ExportedDetectorModel,
    ExportedGeant4Scene,
    ExportedGeant4Source,
)
from sim.isaacsim_app.scene_builder import StagePrimPaths


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def radioactive_decay_sidecar(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the complete sidecar once when Geant4 is available."""
    if shutil.which("g++") is None or shutil.which("geant4-config") is None:
        pytest.skip("g++ and geant4-config are required for this integration.")
    executable = tmp_path_factory.mktemp("radioactive_decay") / "sidecar"
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


def _surface_source(
    isotope: str,
    y_m: float,
    *,
    activity_bq: float = 1_000.0,
) -> ExportedGeant4Source:
    """Return one wall-bound source for a small cascade smoke scene."""
    epsilon = SURFACE_EMISSION_EPSILON_M
    return ExportedGeant4Source(
        isotope=isotope,
        position_xyz=(epsilon, y_m, 1.0),
        anchor_position_xyz=(0.0, y_m, 1.0),
        activity_bq=activity_bq,
        surface_chart_id=0,
        surface_uv=(y_m / 2.0, 0.5),
        surface_normal_xyz=(1.0, 0.0, 0.0),
        surface_emission_policy_sha256=surface_emission_policy_sha256(),
    )


def _parse_response(path: Path) -> tuple[dict[str, str], tuple[float, ...]]:
    """Parse the native metadata and spectrum needed by this contract test."""
    metadata: dict[str, str] = {}
    spectrum: tuple[float, ...] = ()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("META "):
            key, value = line[5:].split("=", maxsplit=1)
            metadata[key] = value
        elif line.startswith("SPECTRUM "):
            spectrum = tuple(float(value) for value in line[9:].split(","))
    return metadata, spectrum


def test_radioactive_decay_tracks_cascade_as_one_detector_event(
    radioactive_decay_sidecar: Path,
    tmp_path: Path,
) -> None:
    """Co-60 and long-lived Eu-152 must run through evaluated RDM cascades."""
    scene_path = tmp_path / "cascade.scene"
    request_path = tmp_path / "cascade.request"
    response_path = tmp_path / "cascade.response"
    scene = ExportedGeant4Scene(
        scene_hash="d" * 64,
        usd_path=None,
        room_size_xyz=(2.0, 2.0, 2.0),
        static_volumes=(),
        sources=(
            _surface_source("Co-60", 0.8),
            _surface_source("Eu-152", 1.2),
        ),
        detector_model=ExportedDetectorModel(),
        fe_shield=None,
        pb_shield=None,
        prim_paths=StagePrimPaths(),
    )
    write_scene_file(scene, scene_path)
    write_request_file(
        Geant4StepRequest(
            step_id=0,
            dwell_time_s=1.0,
            seed=9841,
            detector_pose_xyz=(0.2, 1.0, 1.0),
            detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            fe_shield_pose_xyz=(0.2, 1.0, 1.0),
            fe_shield_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            pb_shield_pose_xyz=(0.2, 1.0, 1.0),
            pb_shield_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
        request_path,
    )
    completed = subprocess.run(
        [
            radioactive_decay_sidecar.as_posix(),
            "--scene",
            scene_path.as_posix(),
            "--request",
            request_path.as_posix(),
            "--response",
            response_path.as_posix(),
            "--physics-profile",
            "balanced",
            "--threads",
            "2",
            "--source-rate-model",
            "parent_decay_activity_bq",
            "--primary-emission-model",
            "geant4_radioactive_decay",
            "--source-bias-mode",
            "analog",
            "--detector-scoring-mode",
            "full_transport",
            "--secondary-transport-mode",
            "full_transport",
            "--primary-sampling-fraction",
            "1",
            "--background-cps",
            "0",
            "--dead-time-tau-s",
            "0",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120.0,
    )
    assert completed.returncode == 0, completed.stderr
    metadata, spectrum = _parse_response(response_path)

    assert spectrum
    assert sum(spectrum) > 0.0
    assert metadata["primary_emission_model"] == "geant4_radioactive_decay"
    assert metadata["prompt_decay_cascade_transport"] == "true"
    assert metadata["radioactive_source_state_semantics"] == (
        "scheduled_parent_decays_no_preexisting_daughter_inventory"
    )
    assert metadata["true_coincidence_summing"] == (
        "global_time_window_energy_deposit_sum"
    )
    assert float(metadata["detector_coincidence_window_s"]) == pytest.approx(
        1.0e-6
    )
    assert metadata["delayed_decay_pulse_separation"] == "true"
    assert metadata["radioactive_decay_time_window"] == (
        "parent_events_uniform_in_acquisition_prompt_parent_forced_"
        "daughters_geant4_timed_out_of_window_rejected"
    )
    assert metadata["expected_primary_semantics"] == (
        "parent_activity_bq_times_live_time"
    )
    assert int(metadata["primary_history_batch_count"]) == 2


def test_decay_comparison_axis_preserves_co60_sum_peak(
    radioactive_decay_sidecar: Path,
    tmp_path: Path,
) -> None:
    """The diagnostic-only wide axis must retain the 2506-keV Co-60 sum peak."""
    scene_path = tmp_path / "wide_cascade.scene"
    request_path = tmp_path / "wide_cascade.request"
    response_path = tmp_path / "wide_cascade.response"
    scene = ExportedGeant4Scene(
        scene_hash="f" * 64,
        usd_path=None,
        room_size_xyz=(2.0, 2.0, 2.0),
        static_volumes=(),
        sources=(
            _surface_source(
                "Co-60",
                1.0,
                activity_bq=15_000.0,
            ),
        ),
        detector_model=ExportedDetectorModel(),
        fe_shield=None,
        pb_shield=None,
        prim_paths=StagePrimPaths(),
    )
    write_scene_file(scene, scene_path)
    write_request_file(
        Geant4StepRequest(
            step_id=0,
            dwell_time_s=1.0,
            seed=74291,
            detector_pose_xyz=(0.08, 1.0, 1.0),
            detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            fe_shield_pose_xyz=(0.08, 1.0, 1.0),
            fe_shield_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            pb_shield_pose_xyz=(0.08, 1.0, 1.0),
            pb_shield_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
        request_path,
    )
    completed = subprocess.run(
        [
            radioactive_decay_sidecar.as_posix(),
            "--scene",
            scene_path.as_posix(),
            "--request",
            request_path.as_posix(),
            "--response",
            response_path.as_posix(),
            "--physics-profile",
            "balanced",
            "--threads",
            "2",
            "--source-rate-model",
            "parent_decay_activity_bq",
            "--primary-emission-model",
            "geant4_radioactive_decay",
            "--source-bias-mode",
            "analog",
            "--detector-scoring-mode",
            "full_transport",
            "--secondary-transport-mode",
            "full_transport",
            "--primary-sampling-fraction",
            "1",
            "--background-cps",
            "0",
            "--dead-time-tau-s",
            "0",
            "--decay-comparison-diagnostic",
            "--decay-comparison-energy-max-kev",
            "3400",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120.0,
    )
    assert completed.returncode == 0, completed.stderr
    metadata, spectrum = _parse_response(response_path)
    assert len(spectrum) == 1701
    assert metadata["decay_comparison_diagnostic"] == "true"
    assert metadata["spectrum_energy_max_keV"] == "3400"
    co60_sum_window = spectrum[1235:1272]
    assert sum(co60_sum_window) > 0.0


@pytest.mark.parametrize(
    "isotope",
    (
        "Cs-137",
        "Co-60",
        "Eu-154",
        "Eu-152",
        "Nb-94",
        "Cs-134",
        "Sb-125",
        "Am-241",
    ),
)
def test_requested_nuclide_has_live_evaluated_radioactive_decay(
    radioactive_decay_sidecar: Path,
    tmp_path: Path,
    isotope: str,
) -> None:
    """Every requested candidate must produce detected RDM decay radiation."""
    scene_path = tmp_path / f"{isotope}.scene"
    request_path = tmp_path / f"{isotope}.request"
    response_path = tmp_path / f"{isotope}.response"
    scene = ExportedGeant4Scene(
        scene_hash="e" * 64,
        usd_path=None,
        room_size_xyz=(2.0, 2.0, 2.0),
        static_volumes=(),
        sources=(_surface_source(isotope, 1.0),),
        detector_model=ExportedDetectorModel(),
        fe_shield=None,
        pb_shield=None,
        prim_paths=StagePrimPaths(),
    )
    write_scene_file(scene, scene_path)
    write_request_file(
        Geant4StepRequest(
            step_id=0,
            dwell_time_s=1.0,
            seed=8123,
            detector_pose_xyz=(0.2, 1.0, 1.0),
            detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            fe_shield_pose_xyz=(0.2, 1.0, 1.0),
            fe_shield_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            pb_shield_pose_xyz=(0.2, 1.0, 1.0),
            pb_shield_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
        request_path,
    )
    completed = subprocess.run(
        [
            radioactive_decay_sidecar.as_posix(),
            "--scene",
            scene_path.as_posix(),
            "--request",
            request_path.as_posix(),
            "--response",
            response_path.as_posix(),
            "--physics-profile",
            "balanced",
            "--threads",
            "2",
            "--source-rate-model",
            "parent_decay_activity_bq",
            "--primary-emission-model",
            "geant4_radioactive_decay",
            "--source-bias-mode",
            "analog",
            "--detector-scoring-mode",
            "full_transport",
            "--secondary-transport-mode",
            "full_transport",
            "--primary-sampling-fraction",
            "1",
            "--background-cps",
            "0",
            "--dead-time-tau-s",
            "0",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120.0,
    )
    assert completed.returncode == 0, completed.stderr
    metadata, spectrum = _parse_response(response_path)

    if isotope == "Cs-137":
        assert sum(spectrum) == 0.0
        assert int(
            metadata["acquisition_window_rejected_delayed_pulses"]
        ) > 0
    else:
        assert sum(spectrum) > 0.0
        assert float(metadata[f"transport_detected_counts_{isotope}"]) > 0.0
    assert metadata["prompt_decay_cascade_transport"] == "true"
    assert metadata["radioactive_source_state_semantics"] == (
        "scheduled_parent_decays_no_preexisting_daughter_inventory"
    )
    assert metadata["true_coincidence_summing"] == (
        "global_time_window_energy_deposit_sum"
    )
    assert metadata["delayed_decay_pulse_separation"] == "true"
