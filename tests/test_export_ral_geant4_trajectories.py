"""Tests for the actual Geant4 trajectory export utility."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.export_ral_geant4_trajectories import (
    _prepare_output_directory,
    _recorded_station,
    _diagnostic_config,
    parse_args,
    parse_trajectory_file,
)


def test_diagnostic_config_uses_a_dedicated_executable(tmp_path: Path) -> None:
    """Trajectory export must explicitly select its nonproduction binary."""
    executable = tmp_path / "geant4_trajectory_sidecar"
    trajectory = tmp_path / "track-output.txt"

    config = _diagnostic_config(
        {"executable_path": "build/geant4_sidecar"},
        mode="isotropic_emission",
        trajectory_path=trajectory,
        executable_path=executable,
        seed=17,
    )

    assert config["executable_path"] == executable.resolve().as_posix()
    assert config["executable_args"][0] == "--geant4-trajectory-output"
    assert config["executable_args"][1] == trajectory.resolve().as_posix()


def test_parse_trajectory_file_preserves_native_step_points(tmp_path: Path) -> None:
    """The parser should preserve native coordinates and track semantics."""
    path = tmp_path / "tracks.txt"
    path.write_text(
        "\n".join(
            (
                "FORMAT geant4_primary_gamma_step_trajectory_v1",
                "META scene_hash=abc",
                "META recorded_track_count=1",
                "TRACK source_index=2 isotope=Cs-137 primary_batch_index=4 "
                "primary_history_index=9 bias_branch_lineage_id=0 track_id=1 "
                "parent_id=0 initial_energy_keV=661.7 raw_step_count=1 "
                "detector_entered=1 interacted=0 points_truncated=0 point_count=2",
                "POINT index=0 x_m=1.25 y_m=2.5 z_m=0.75",
                "POINT index=1 x_m=4.0 y_m=5.0 z_m=1.5",
                "END_TRACK",
                "",
            )
        ),
        encoding="utf-8",
    )

    parsed = parse_trajectory_file(path, mode="detector_entering")

    assert parsed["metadata"] == {
        "scene_hash": "abc",
        "recorded_track_count": "1",
    }
    tracks = parsed["tracks"]
    assert isinstance(tracks, list)
    assert tracks == [
        {
            "mode": "detector_entering",
            "source_index": 2,
            "isotope": "Cs-137",
            "primary_batch_index": 4,
            "primary_history_index": 9,
            "bias_branch_lineage_id": 0,
            "track_id": 1,
            "parent_id": 0,
            "initial_energy_keV": 661.7,
            "raw_step_count": 1,
            "detector_entered": True,
            "interacted": False,
            "points_truncated": False,
            "points_xyz_m": [[1.25, 2.5, 0.75], [4.0, 5.0, 1.5]],
        }
    ]


def test_parse_trajectory_file_rejects_missing_points(tmp_path: Path) -> None:
    """A track without both Geant4 endpoints must fail closed."""
    path = tmp_path / "tracks.txt"
    path.write_text(
        "\n".join(
            (
                "FORMAT geant4_primary_gamma_step_trajectory_v1",
                "META recorded_track_count=1",
                "TRACK source_index=0 isotope=Co-60 primary_batch_index=0 "
                "primary_history_index=0 bias_branch_lineage_id=0 track_id=1 "
                "parent_id=0 initial_energy_keV=1173.2 raw_step_count=0 "
                "detector_entered=0 interacted=0 points_truncated=0 point_count=0",
                "END_TRACK",
                "",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid track"):
        parse_trajectory_file(path, mode="isotropic_emission")


def test_run_selection_drives_all_defaults_and_creates_fresh_output_names() -> None:
    """Changing runs or stations must never select the old hardcoded evidence."""
    arguments = ["--run-id", "example_run", "--station-index", "12"]
    first = parse_args(arguments)
    second = parse_args(arguments)
    for name in ("runtime_config", "scenario", "truth_manifest"):
        assert getattr(first, name).name == "example_run.json"
    assert first.observations.parent.name == "example_run"
    assert "example_run_station_12_" in first.output.parent.name
    assert first.output.name == "trajectories.json"
    assert first.output != second.output


@pytest.mark.parametrize("existing", ("json", "raw"))
def test_output_reservation_preserves_existing_artifacts(
    tmp_path: Path, existing: str
) -> None:
    """A previous completed or interrupted export must not be overwritten."""
    output = tmp_path / "trajectories.json"
    if existing == "json":
        output.write_text("keep current evidence")
        protected = output
    else:
        raw = _prepare_output_directory(output)
        protected = raw / "isotropic_emission.tracks"
        protected.write_text("keep current evidence")
    with pytest.raises(FileExistsError):
        _prepare_output_directory(output)
    assert protected.read_text() == "keep current evidence"


def test_station_selection_rejects_another_measurement_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pose must not be silently joined to another private scene."""
    monkeypatch.setattr(
        "scripts.export_ral_geant4_trajectories.load_measurement_log",
        lambda _path: SimpleNamespace(run_id="another_run", records=()),
    )
    with pytest.raises(ValueError, match="MeasurementLog run ID disagrees"):
        _recorded_station(tmp_path / "observations.npz", 0, run_id="selected_run")


def test_trajectory_metadata_rejects_duplicate_identity(tmp_path: Path) -> None:
    """Duplicate identity records must not silently replace native provenance."""
    path = tmp_path / "tracks.txt"
    path.write_text(
        "FORMAT geant4_primary_gamma_step_trajectory_v1\n"
        "META recorded_track_count=0\nMETA recorded_track_count=0\n"
    )
    with pytest.raises(ValueError, match="malformed metadata"):
        parse_trajectory_file(path, mode="isotropic_emission")
