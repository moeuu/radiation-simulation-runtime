"""Exercise opt-in trajectory recording with real native Geant4 transport."""

from pathlib import Path
import subprocess

import numpy as np
import pytest

from measurement.source_boundary import (
    SURFACE_EMISSION_EPSILON_M,
    surface_emission_policy_sha256,
)
from scripts.export_ral_geant4_trajectories import parse_trajectory_file
from sim.geant4_app.engine import Geant4StepRequest
from sim.geant4_app.io_format import (
    read_response_file,
    write_request_file,
    write_scene_file,
)
from sim.geant4_app.scene_export import (
    ExportedDetectorModel,
    ExportedGeant4Material,
    ExportedGeant4Scene,
    ExportedGeant4Source,
    ExportedGeant4Volume,
)
from sim.isaacsim_app.scene_builder import StagePrimPaths


def _inputs(directory: Path) -> list[str]:
    """Write two weak sources and an air boundary to exercise multi-step tracks."""
    sources = tuple(
        ExportedGeant4Source(
            isotope=isotope,
            position_xyz=(SURFACE_EMISSION_EPSILON_M, y_m, 1.0),
            anchor_position_xyz=(0.0, y_m, 1.0),
            intensity_cps_1m=1.0,
            surface_chart_id=0,
            surface_uv=(y_m / 2.0, 0.5),
            surface_normal_xyz=(1.0, 0.0, 0.0),
            surface_emission_policy_sha256=surface_emission_policy_sha256(),
        )
        for isotope, y_m in (("Cs-137", 0.95), ("Co-60", 1.05))
    )
    scene = ExportedGeant4Scene(
        scene_hash="d" * 64,
        usd_path=None,
        room_size_xyz=(2.0, 2.0, 2.0),
        static_volumes=(
            ExportedGeant4Volume(
                path="/World/AirBoundary",
                shape="box",
                translation_xyz=(0.04, 1.0, 1.0),
                orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
                size_xyz=(0.002, 0.4, 0.4),
                material=ExportedGeant4Material(name="air"),
            ),
        ),
        sources=sources,
        detector_model=ExportedDetectorModel(),
        fe_shield=None,
        pb_shield=None,
        prim_paths=StagePrimPaths(),
    )
    request = Geant4StepRequest(
        step_id=0,
        dwell_time_s=1.0,
        seed=41357,
        detector_pose_xyz=(0.1, 1.0, 1.0),
        detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        fe_shield_pose_xyz=(0.1, 1.0, 1.0),
        fe_shield_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        pb_shield_pose_xyz=(0.1, 1.0, 1.0),
        pb_shield_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    scene_path = directory / "input.scene"
    request_path = directory / "input.request"
    write_scene_file(scene, scene_path)
    write_request_file(request, request_path)
    return ["--scene", str(scene_path), "--request", str(request_path)]


@pytest.mark.parametrize("mode", ("isotropic_emission", "detector_entering"))
def test_recording_preserves_transport_and_caps_only_recorded_points(
    native_sidecar_executable: Path,
    native_trajectory_executable: Path,
    tmp_path: Path,
    mode: str,
) -> None:
    """Both diagnostics must leave the seeded spectrum and its variance unchanged."""
    inputs = _inputs(tmp_path)
    common = [
        *inputs,
        "--physics-profile",
        "balanced",
        "--threads",
        "1",
        "--secondary-transport-mode",
        "full_transport",
        "--primary-sampling-fraction",
        "1",
        "--background-cps",
        "0",
        "--dead-time-tau-s",
        "0",
    ]
    if mode == "isotropic_emission":
        common += [
            "--source-rate-model",
            "isotropic_emission_equivalent",
            "--source-bias-mode",
            "analog",
            "--detector-scoring-mode",
            "full_transport",
        ]
    else:
        common += [
            "--source-rate-model",
            "detector_cps_1m",
            "--source-bias-mode",
            "detector_cone",
            "--detector-scoring-mode",
            "incident_gamma_energy",
            "--mean-calibration-histories-per-source-line",
            "32",
            "--mean-calibration-angle-strata-mu",
            "4",
            "--mean-calibration-angle-strata-phi",
            "4",
            "--validation-entry-class-spectra",
        ]
    tracks_path = tmp_path / "native.tracks"
    recording = [
        "--geant4-trajectory-output",
        str(tracks_path),
        "--geant4-trajectory-max-tracks-per-source",
        "8",
        "--geant4-trajectory-max-points-per-track",
        "2",
    ]
    outputs = []
    for name, executable, options in (
        ("production", native_sidecar_executable, []),
        ("diagnostic_disabled", native_trajectory_executable, []),
        ("diagnostic_recording", native_trajectory_executable, recording),
    ):
        response = tmp_path / f"{name}.response"
        completed = subprocess.run(
            [str(executable), *common, "--response", str(response), *options],
            check=False,
            capture_output=True,
            text=True,
            timeout=60.0,
        )
        assert completed.returncode == 0, completed.stderr
        outputs.append(read_response_file(response))
    reference_spectrum, reference_metadata = outputs[0]
    assert reference_spectrum.sum() > 0
    for spectrum, metadata in outputs[1:]:
        np.testing.assert_array_equal(spectrum, reference_spectrum)
        assert (
            metadata["spectrum_count_variance"]
            == reference_metadata["spectrum_count_variance"]
        )
        assert metadata["num_primaries"] == reference_metadata["num_primaries"]
    assert "geant4_trajectory_recording" not in reference_metadata
    assert outputs[1][1]["geant4_trajectory_recording"] is False
    assert outputs[2][1]["geant4_trajectory_recording"] is True
    parsed = parse_trajectory_file(tracks_path, mode=mode)
    assert parsed["metadata"]["scene_hash"] == "d" * 64
    assert len(parsed["tracks"]) == 16
    assert outputs[2][1]["geant4_trajectory_track_count"] == 16
    for source_index, y_m in enumerate((0.95, 1.05)):
        tracks = [t for t in parsed["tracks"] if t["source_index"] == source_index]
        assert len(tracks) == 8
        for track in tracks:
            assert track["parent_id"] == 0
            assert len(track["points_xyz_m"]) == 2
            np.testing.assert_allclose(
                track["points_xyz_m"][0],
                [SURFACE_EMISSION_EPSILON_M, y_m, 1.0],
                rtol=0.0,
                atol=1.0e-15,
            )
    if mode == "detector_entering":
        assert any(t["detector_entered"] for t in parsed["tracks"])
        assert any(t["points_truncated"] for t in parsed["tracks"])


def test_recording_requires_opt_in_build_and_one_shot_mode(
    native_sidecar_executable: Path,
    native_trajectory_executable: Path,
    tmp_path: Path,
) -> None:
    """Production rejects recording options and diagnostics reject persistent use."""
    path = tmp_path / "forbidden.tracks"
    for executable, extra, message in (
        (native_sidecar_executable, [], "Unsupported Geant4 sidecar option"),
        (
            native_trajectory_executable,
            ["--persistent"],
            "cannot be combined with --persistent",
        ),
    ):
        completed = subprocess.run(
            [str(executable), "--geant4-trajectory-output", str(path), *extra],
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        assert completed.returncode != 0
        assert message in completed.stderr
        assert not path.exists()
