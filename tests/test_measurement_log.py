"""Contract tests for raw full-spectrum MeasurementLog schema version 2."""

from __future__ import annotations

from dataclasses import replace
import errno
from hashlib import sha256
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import pytest

from runtime.measurement_log import (
    MEASUREMENT_LOG_SCHEMA_VERSION,
    MeasurementLogStreamWriter,
    MeasurementLogValidationError,
    _validate_full_spectrum_contract_alignment,
    _validate_environment_payload,
    _validate_record_sequence,
    load_measurement_log,
)
from runtime.records import validate_truth_free_estimator_input
from measurement.obstacle_assets import KnownObstacleInstance, ObstacleComponent
from measurement.obstacles import ObstacleGrid
from tests.runtime_test_support import (
    TEST_COMMIT,
    TEST_ISOTOPES,
    environment,
    make_measurement_log,
    records,
    runtime_config,
)


def test_measurement_log_round_trip_preserves_raw_integer_spectra(tmp_path) -> None:
    """Canonical storage must preserve the event histogram without projection."""
    root = make_measurement_log(tmp_path / "measurement-log")

    loaded = load_measurement_log(root)

    assert loaded.schema_version == MEASUREMENT_LOG_SCHEMA_VERSION == 2
    assert len(loaded.records) == 4
    for expected, actual in zip(records(), loaded.records, strict=True):
        assert actual.spectrum_counts.dtype == np.int64
        np.testing.assert_array_equal(
            actual.spectrum_counts,
            expected.spectrum_counts,
        )
    with np.load(root / "observations.npz", allow_pickle=False) as arrays:
        assert "spectrum_counts" in arrays.files
        assert not any("isotope_count" in key for key in arrays.files)


def test_measurement_log_publication_never_replaces_a_racing_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A target created after staging begins must remain untouched."""
    target = tmp_path / "measurement-log"

    def _inject_racing_target(
        source: Path,
        destination: Path,
        *,
        flags: int,
    ) -> None:
        """Create a competing generation at the final no-replace operation."""
        del source
        assert flags == 1
        destination.mkdir()
        (destination / "owner.txt").write_text("racer", encoding="utf-8")
        raise FileExistsError(errno.EEXIST, "target exists", destination)

    monkeypatch.setattr("runtime.artifacts._linux_renameat2", _inject_racing_target)

    with pytest.raises(FileExistsError):
        make_measurement_log(target)

    assert (target / "owner.txt").read_text(encoding="utf-8") == "racer"
    assert not tuple(tmp_path.glob(".measurement-log.bundle-*"))


def test_stream_cleanup_failure_prevents_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A WAL cleanup failure must abort before public bundle publication."""
    writer = object.__new__(MeasurementLogStreamWriter)
    writer.output_dir = tmp_path / "measurement-log"
    writer.stage_dir = tmp_path / f".measurement-log.stream-{os.getpid()}"
    writer.stage_dir.mkdir()
    writer.run_id = "run"
    writer.repository_commit = "a" * 40
    writer.runtime_config = {}
    writer.environment = {}
    writer.forward_model_manifest = {}
    writer.isotopes = ("Cs-137",)
    writer.records = []
    writer.metadata = {}
    writer.obstacle_layout_path = None
    writer.source_layout_path = None
    publication_attempted = False

    def _publish(*args: object, **kwargs: object) -> object:
        """Record an invalid publication attempt after failed cleanup."""
        nonlocal publication_attempted
        del args, kwargs
        publication_attempted = True
        return object()

    def _fail_cleanup(path: Path) -> None:
        """Model an OS cleanup error before publication."""
        assert path == writer.stage_dir
        raise OSError("injected cleanup failure")

    monkeypatch.setattr("runtime.measurement_log.write_measurement_log", _publish)
    monkeypatch.setattr("runtime.measurement_log.shutil.rmtree", _fail_cleanup)

    with pytest.raises(OSError, match="injected cleanup failure"):
        writer.finalize()
    assert not publication_attempted


def test_stream_abort_removes_owned_wal_idempotently(tmp_path: Path) -> None:
    """An aborted fresh run must leave no recoverable partial stage."""
    writer = object.__new__(MeasurementLogStreamWriter)
    writer.output_dir = tmp_path / "measurement-log"
    writer.stage_dir = tmp_path / f".measurement-log.stream-{os.getpid()}"
    writer.stage_dir.mkdir()
    (writer.stage_dir / "record_00000000.npz").write_bytes(b"partial")

    writer.abort()
    writer.abort()

    assert not writer.stage_dir.exists()


def test_stream_constructor_failure_removes_its_private_wal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial writer constructor must roll back its newly owned WAL."""
    output_dir = tmp_path / "measurement-log"

    def fail_manifest_write(*args: object, **kwargs: object) -> None:
        """Inject failure after the private staging directory is created."""
        del args, kwargs
        raise OSError("injected WAL manifest write failure")

    monkeypatch.setattr("runtime.measurement_log._write_json", fail_manifest_write)

    with pytest.raises(OSError, match="injected WAL manifest write failure"):
        MeasurementLogStreamWriter(
            output_dir,
            run_id="stream-constructor-failure-test",
            repository_commit=TEST_COMMIT,
            runtime_config=runtime_config(),
            environment=environment(),
            forward_model_manifest={},
            isotopes=TEST_ISOTOPES,
        )

    assert not output_dir.exists()
    assert not tuple(tmp_path.glob(".measurement-log.stream-*"))


def test_record_rejects_fractional_or_projected_counts() -> None:
    """A weighted or corrected histogram cannot enter the production log."""
    valid = records(1)[0]

    with pytest.raises(
        MeasurementLogValidationError,
        match="unit-weight integer",
    ):
        replace(
            valid,
            spectrum_counts=np.full(
                valid.spectrum_counts.shape,
                0.5,
                dtype=np.float64,
            ),
        )


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    (
        ("detector_pose_xyz", ("0.25", 0.25, 0.4)),
        ("detector_pose_xyz", (True, 0.25, 0.4)),
        ("detector_quat_wxyz", (True, 0.0, 0.0, 0.0)),
        ("live_time_s", "1.0"),
        ("live_time_s", True),
        ("travel_time_s", "0.25"),
        ("shield_actuation_time_s", False),
    ),
)
def test_record_rejects_pose_and_time_scalar_coercion(
    field_name: str,
    invalid: object,
) -> None:
    """Pose and timing semantics cannot change through scalar conversion."""
    with pytest.raises(MeasurementLogValidationError, match="finite JSON number"):
        replace(records(1)[0], **{field_name: invalid})


def test_record_rejects_a_reshaped_energy_axis() -> None:
    """A matrix-shaped axis cannot become a valid vector through flattening."""
    valid = records(1)[0]

    with pytest.raises(MeasurementLogValidationError, match="one-dimensional"):
        replace(
            valid,
            energy_bin_edges_keV=valid.energy_bin_edges_keV.reshape(1, -1),
        )


@pytest.mark.parametrize("dtype", (np.int32, np.uint64))
def test_record_requires_exact_int64_spectrum_storage(
    dtype: type[np.generic],
) -> None:
    """Integer-looking arrays cannot be silently retyped as event histograms."""
    valid = records(1)[0]

    with pytest.raises(MeasurementLogValidationError, match="unit-weight integer"):
        replace(
            valid,
            spectrum_counts=valid.spectrum_counts.astype(dtype),
        )


def test_record_rejects_removed_isotope_count_metadata() -> None:
    """Removed per-isotope inference routes cannot be hidden in metadata."""
    valid = records(1)[0]

    with pytest.raises(
        MeasurementLogValidationError,
        match="removed metadata",
    ):
        replace(
            valid,
            metadata={
                **dict(valid.metadata),
                "runtime_likelihood_route_by_isotope": {"Cs-137": "legacy"},
            },
        )


@pytest.mark.parametrize(
    "invalid",
    (
        Path("implicit-path.json"),
        np.int64(7),
        object(),
        {1: "non-string-key"},
    ),
)
def test_record_metadata_rejects_lossy_json_coercions(invalid: object) -> None:
    """Production metadata must already be JSON-native at its write boundary."""
    valid = records(1)[0]

    with pytest.raises(
        MeasurementLogValidationError,
        match="JSON",
    ):
        replace(
            valid,
            metadata={**dict(valid.metadata), "invalid": invalid},
        )


@pytest.mark.parametrize("invalid", (Path("config.json"), np.float32(1.0), object()))
def test_runtime_config_rejects_lossy_json_coercions_before_publication(
    tmp_path: Path,
    invalid: object,
) -> None:
    """A non-JSON runtime value must fail before any log directory is published."""
    target = tmp_path / "measurement-log"

    with pytest.raises((TypeError, MeasurementLogValidationError)):
        make_measurement_log(
            target,
            runtime_overrides={"invalid_runtime_value": invalid},
        )

    assert not target.exists()


def test_record_accepts_truth_free_source_position_contract_label() -> None:
    """A coordinate semantic label is not a realized source position."""
    valid = records(1)[0]

    updated = replace(
        valid,
        metadata={
            **dict(valid.metadata),
            "source_position_semantics": "air_side_native_emission_xyz",
        },
    )

    assert (
        updated.metadata["source_position_semantics"] == "air_side_native_emission_xyz"
    )


def test_record_rejects_realized_source_positions() -> None:
    """Actual source coordinates must remain outside estimator-visible records."""
    valid = records(1)[0]

    with pytest.raises(
        MeasurementLogValidationError,
        match="realized truth",
    ):
        replace(
            valid,
            metadata={
                **dict(valid.metadata),
                "source_positions": [[1.0, 2.0, 3.0]],
            },
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "private_source_profile",
        "scenario_family",
        "scene_seed",
        "scene_rng_provenance",
        "source_profile",
    ),
)
def test_record_rejects_private_scenario_metadata(field_name: str) -> None:
    """Estimator-visible metadata must not identify private scene generation."""
    valid = records(1)[0]

    with pytest.raises(MeasurementLogValidationError, match="realized truth"):
        replace(
            valid,
            metadata={**dict(valid.metadata), field_name: "private"},
        )


@pytest.mark.parametrize(
    "field_name",
    ("private_source_profile", "scene_seed", "scene_rng_provenance"),
)
def test_estimator_runtime_config_rejects_private_scene_fields(
    field_name: str,
) -> None:
    """Current-scene generation inputs cannot hide in physical config."""
    with pytest.raises(MeasurementLogValidationError, match="realized truth"):
        validate_truth_free_estimator_input(
            {"context": {"runtime_config": {field_name: "private"}}},
            path="adaptive.event",
        )


def test_record_rejects_truthy_string_station_completion_marker() -> None:
    """The causal station boundary accepts an exact JSON boolean only."""
    valid = records(1)[0]

    with pytest.raises(
        MeasurementLogValidationError,
        match="station_complete must be a boolean",
    ):
        replace(
            valid,
            metadata={
                **dict(valid.metadata),
                "station_complete": "false",
            },
        )


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    (
        ("record_count", "4"),
        ("record_count", 4.0),
        ("energy_bin_count", "851"),
        ("energy_bin_count", 851.0),
        ("full_spectrum_contract_schema_version", "3"),
        ("full_spectrum_contract_schema_version", 3.0),
    ),
)
def test_run_manifest_rejects_coerced_integer_fields(
    tmp_path: Path,
    field_name: str,
    invalid: object,
) -> None:
    """Manifest dimensions and schema identifiers are exact JSON integers."""
    root = make_measurement_log(tmp_path / "measurement-log")
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field_name] = invalid
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MeasurementLogValidationError, match="JSON integer"):
        load_measurement_log(root)


def test_metadata_row_rejects_string_array_index(tmp_path: Path) -> None:
    """A string index cannot be coerced into causal record order."""
    root = make_measurement_log(tmp_path / "measurement-log")
    metadata_path = root / "observation_metadata.jsonl"
    rows = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["array_index"] = "0"
    metadata_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_hashes"]["observation_metadata.jsonl"] = sha256(
        metadata_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MeasurementLogValidationError, match="JSON integer"):
        load_measurement_log(root)


def test_loader_rejects_duplicate_json_members(tmp_path: Path) -> None:
    """Duplicate manifest keys cannot be resolved with last-value-wins parsing."""
    root = make_measurement_log(tmp_path / "measurement-log")
    manifest_path = root / "run_manifest.json"
    original = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(
        '{"schema_version":2,' + original[1:],
        encoding="utf-8",
    )

    with pytest.raises(MeasurementLogValidationError, match="duplicate JSON key"):
        load_measurement_log(root)


def test_loader_rejects_nonfinite_json_constants(tmp_path: Path) -> None:
    """NaN and infinity extensions cannot enter a scientific JSON artifact."""
    root = make_measurement_log(tmp_path / "measurement-log")
    config_path = root / "runtime_config.resolved.json"
    original = config_path.read_text(encoding="utf-8")
    config_path.write_text(
        '{"unexpected_nonfinite":NaN,' + original[1:],
        encoding="utf-8",
    )

    with pytest.raises(MeasurementLogValidationError, match="non-finite JSON"):
        load_measurement_log(root)


def test_loader_rejects_unknown_run_manifest_fields(tmp_path: Path) -> None:
    """A misspelled or future manifest field cannot be silently ignored."""
    root = make_measurement_log(tmp_path / "measurement-log")
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["record_counts"] = manifest["record_count"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MeasurementLogValidationError, match="unknown"):
        load_measurement_log(root)


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    (
        ("size_x", "2.0"),
        ("size_y", True),
        ("detector_position", [True, 0.25, 0.4]),
    ),
)
def test_loader_rejects_environment_numeric_coercion(
    tmp_path: Path,
    field_name: str,
    invalid: object,
) -> None:
    """Logged room geometry must preserve exact JSON numeric types."""
    root = make_measurement_log(tmp_path / "measurement-log")
    environment_path = root / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment[field_name] = invalid
    environment_path.write_text(json.dumps(environment), encoding="utf-8")
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["environment"] = environment
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        MeasurementLogValidationError,
        match="environment is incompatible",
    ):
        load_measurement_log(root)


def test_environment_rejects_hollow_asset_serialized_as_solid_box() -> None:
    """MeasurementLog must bind estimator inputs to authored shell components."""
    component = ObstacleComponent(
        name="cabinet_left_wall",
        center_xyz=(1.05, 1.5, 1.0),
        size_xyz=(0.1, 1.0, 2.0),
        material="steel",
    )
    instance = KnownObstacleInstance(
        name="cabinet_0",
        template="steel_cabinet_hollow",
        footprint_xy=(1.0, 2.0, 1.0, 2.0),
        footprint_cells=((1, 1),),
        components=(component,),
    )
    solid_envelope = (1.0, 1.0, 0.0, 2.0, 2.0, 2.0)
    invalid_grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(3, 3),
        blocked_cells=((1, 1),),
        collision_boxes_m=(solid_envelope,),
        transport_boxes_m=(solid_envelope,),
    )
    payload = {
        "environment_model_id": "hollow-contract-test",
        "size_x": 3.0,
        "size_y": 3.0,
        "size_z": 2.5,
        "detector_position": [0.5, 0.5, 0.5],
        "obstacle_grid": invalid_grid.to_dict(),
        "obstacle_instances": [instance.to_dict()],
    }

    with pytest.raises(
        MeasurementLogValidationError,
        match="authored component boxes",
    ):
        _validate_environment_payload(payload)


def test_loader_rejects_non_string_manifest_isotope(tmp_path: Path) -> None:
    """An isotope identifier cannot be made valid through stringification."""
    root = make_measurement_log(tmp_path / "measurement-log")
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["isotopes"][0] = 60
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MeasurementLogValidationError, match="string array"):
        load_measurement_log(root)


def test_loader_rejects_duplicate_npz_member_names(tmp_path: Path) -> None:
    """Duplicate archive members cannot be hidden by dictionary conversion."""
    root = make_measurement_log(tmp_path / "measurement-log")
    observations_path = root / "observations.npz"
    with zipfile.ZipFile(observations_path, mode="r") as archive:
        duplicate = archive.read("step_id.npy")
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(observations_path, mode="a") as archive:
            archive.writestr("step_id.npy", duplicate)
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_hashes"]["observations.npz"] = sha256(
        observations_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MeasurementLogValidationError, match="duplicate array"):
        load_measurement_log(root)


def test_record_sequence_requires_canonical_ids_and_station_pose() -> None:
    """Causal IDs, station groups, and station geometry are not advisory."""
    valid = records(4)
    with pytest.raises(MeasurementLogValidationError, match="step_id"):
        _validate_record_sequence(
            (valid[0], replace(valid[1], step_id=2), *valid[2:]),
            ("Co-60", "Cs-137", "Eu-154"),
        )
    with pytest.raises(MeasurementLogValidationError, match="contiguous"):
        _validate_record_sequence(
            (
                valid[0],
                valid[1],
                replace(valid[2], station_id=2),
                replace(valid[3], station_id=2),
            ),
            ("Co-60", "Cs-137", "Eu-154"),
        )
    with pytest.raises(MeasurementLogValidationError, match="pose and quaternion"):
        _validate_record_sequence(
            (
                valid[0],
                replace(
                    valid[1],
                    detector_quat_wxyz=(0.0, 1.0, 0.0, 0.0),
                ),
                *valid[2:],
            ),
            ("Co-60", "Cs-137", "Eu-154"),
        )


@pytest.mark.parametrize("record_count", (True, "1", 1.5, -1, 5))
def test_prefix_rejects_coercion_and_range_clamping(
    tmp_path: Path,
    record_count: object,
) -> None:
    """A requested causal prefix cannot be truncated or silently clamped."""
    loaded = load_measurement_log(make_measurement_log(tmp_path / "measurement-log"))

    with pytest.raises(MeasurementLogValidationError):
        loaded.prefix(record_count)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("location", "field_name", "invalid"),
    (
        ("model", "schema_version", "3"),
        ("model", "energy_bin_count", 851.0),
        ("runtime", "energy_bin_count", "851"),
    ),
)
def test_full_spectrum_alignment_rejects_coerced_schema_dimensions(
    tmp_path: Path,
    location: str,
    field_name: str,
    invalid: object,
) -> None:
    """Schema-v3 dimensions stay exact across model, runtime, and log."""
    loaded = load_measurement_log(make_measurement_log(tmp_path / "measurement-log"))
    runtime = json.loads(json.dumps(dict(loaded.runtime_config)))
    if location == "model":
        runtime["full_spectrum_generative_model"][field_name] = invalid
    else:
        runtime[field_name] = invalid

    with pytest.raises(MeasurementLogValidationError, match="JSON integer"):
        _validate_full_spectrum_contract_alignment(
            run_manifest=loaded.run_manifest,
            runtime_config=runtime,
            records=loaded.records,
        )


@pytest.mark.parametrize(
    ("location", "invalid"),
    (
        ("manifest", 137),
        ("manifest", ""),
        ("line", 137),
        ("line", ""),
    ),
)
def test_full_spectrum_alignment_requires_exact_isotope_strings(
    tmp_path: Path,
    location: str,
    invalid: object,
) -> None:
    """Numeric and empty isotope IDs cannot align through string coercion."""
    loaded = load_measurement_log(make_measurement_log(tmp_path / "measurement-log"))
    manifest = json.loads(json.dumps(dict(loaded.run_manifest)))
    runtime = json.loads(json.dumps(dict(loaded.runtime_config)))
    if location == "manifest":
        manifest["isotopes"][0] = invalid
    else:
        runtime["full_spectrum_generative_model"]["line_identity"][0]["isotope"] = (
            invalid
        )

    with pytest.raises(
        MeasurementLogValidationError,
        match="JSON strings",
    ):
        _validate_full_spectrum_contract_alignment(
            run_manifest=manifest,
            runtime_config=runtime,
            records=loaded.records,
        )


def test_full_spectrum_alignment_rejects_duplicate_manifest_isotopes(
    tmp_path: Path,
) -> None:
    """The canonical run-manifest isotope set cannot contain duplicates."""
    loaded = load_measurement_log(make_measurement_log(tmp_path / "measurement-log"))
    manifest = json.loads(json.dumps(dict(loaded.run_manifest)))
    manifest["isotopes"].append(manifest["isotopes"][0])

    with pytest.raises(
        MeasurementLogValidationError,
        match="unique nonempty JSON strings",
    ):
        _validate_full_spectrum_contract_alignment(
            run_manifest=manifest,
            runtime_config=loaded.runtime_config,
            records=loaded.records,
        )


def test_full_spectrum_alignment_allows_authenticated_isotope_superset(
    tmp_path: Path,
) -> None:
    """A candidate subset may use a model with additional authenticated lines."""
    loaded = load_measurement_log(make_measurement_log(tmp_path / "measurement-log"))
    manifest = json.loads(json.dumps(dict(loaded.run_manifest)))
    runtime = json.loads(json.dumps(dict(loaded.runtime_config)))
    manifest["isotopes"] = ["Co-60", "Cs-137"]
    runtime["candidate_isotopes"] = ["Co-60", "Cs-137"]

    _validate_full_spectrum_contract_alignment(
        run_manifest=manifest,
        runtime_config=runtime,
        records=loaded.records,
    )


def test_full_spectrum_alignment_requires_every_candidate_line(
    tmp_path: Path,
) -> None:
    """A run isotope absent from the authenticated line basis must fail closed."""
    loaded = load_measurement_log(make_measurement_log(tmp_path / "measurement-log"))
    manifest = json.loads(json.dumps(dict(loaded.run_manifest)))
    manifest["isotopes"] = ["Co-60", "Cs-137", "Eu-154", "Xe-133"]

    with pytest.raises(
        MeasurementLogValidationError,
        match="must cover",
    ):
        _validate_full_spectrum_contract_alignment(
            run_manifest=manifest,
            runtime_config=loaded.runtime_config,
            records=loaded.records,
        )
