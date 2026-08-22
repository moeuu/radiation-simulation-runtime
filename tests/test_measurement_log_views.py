"""Tests for public immutable MeasurementLog views and inventories."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from runtime import (
    ArtifactInventory,
    MeasurementLogArrayView,
    MeasurementLogStationView,
    MeasurementLogView,
    build_artifact_inventory,
    measurement_log_artifact_inventory,
    measurement_records_content_sha256,
)
from runtime.measurement_log import (
    MeasurementLogValidationError,
    load_measurement_log,
)
from runtime.prefix import measurement_records_sha256
from tests.runtime_test_support import make_measurement_log


def test_array_view_preserves_exact_dtypes_alignment_and_immutability(
    tmp_path: Path,
) -> None:
    """The public dense view must preserve raw arrays without writable aliases."""
    log = load_measurement_log(make_measurement_log(tmp_path / "measurement-log"))

    view = log.array_view()

    assert isinstance(view, MeasurementLogArrayView)
    assert view.record_count == len(log.records) == 4
    assert view.energy_bin_count == 851
    assert view.spectrum_counts.dtype == np.int64
    for name in (
        "step_id",
        "action_id",
        "station_id",
        "fe_orientation_index",
        "pb_orientation_index",
    ):
        assert getattr(view, name).dtype == np.int64
    for name in (
        "detector_pose_xyz",
        "detector_quat_wxyz",
        "live_time_s",
        "travel_time_s",
        "shield_actuation_time_s",
        "energy_bin_edges_keV",
    ):
        assert getattr(view, name).dtype == np.float64
    assert all(not array.flags.writeable for array in view.as_mapping().values())
    np.testing.assert_array_equal(
        view.spectrum_counts,
        np.stack([record.spectrum_counts for record in log.records]),
    )
    with pytest.raises(ValueError, match="read-only"):
        view.spectrum_counts[0, 0] = 0
    with pytest.raises(TypeError):
        view.as_mapping()["step_id"] = np.zeros(4, dtype=np.int64)  # type: ignore[index]
    with pytest.raises(TypeError, match="exact dtype int64"):
        replace(view, spectrum_counts=view.spectrum_counts.astype(np.int32))


def test_empty_prefix_views_keep_the_energy_contract_and_separate_identity(
    tmp_path: Path,
) -> None:
    """An empty in-memory prefix must not masquerade as its source bundle."""
    log = load_measurement_log(make_measurement_log(tmp_path / "measurement-log"))
    empty = log.prefix(0)

    arrays = empty.array_view()
    stations = empty.station_view()

    assert arrays.record_count == 0
    assert arrays.spectrum_counts.shape == (0, 851)
    assert arrays.spectrum_counts.dtype == np.int64
    assert arrays.energy_bin_edges_keV.shape == (852,)
    assert stations.station_count == 0
    assert stations.complete_station_count == 0
    assert stations.source_log_sha256 == log.log_sha256
    assert stations.records_content_sha256 == measurement_records_content_sha256(())
    assert stations.records_content_sha256 != stations.source_log_sha256
    assert empty.log_sha256 == log.log_sha256


def test_records_view_avoids_synthesizing_a_measurement_log_manifest(
    tmp_path: Path,
) -> None:
    """Live records should become shared views directly from their RunContext."""
    log = load_measurement_log(make_measurement_log(tmp_path / "measurement-log"))

    live = MeasurementLogView.from_records(log.context, log.records[:3])
    prefix = log.prefix_view(3)

    assert live.source_log_sha256 is None
    assert prefix.source_log_sha256 == log.log_sha256
    assert live.records_content_sha256 == prefix.records_content_sha256
    np.testing.assert_array_equal(live.array_view().step_id, [0, 1, 2])
    assert [station.station_id for station in live.station_view().stations] == [0, 1]
    empty = MeasurementLogView.from_records(log.context, ())
    assert empty.array_view().spectrum_counts.shape == (0, 851)


def test_station_view_preserves_durable_prefix_semantics(tmp_path: Path) -> None:
    """Station prefixes must end only at explicitly durable writer markers."""
    log = load_measurement_log(
        make_measurement_log(
            tmp_path / "measurement-log",
            station_complete_markers=True,
        )
    )
    partial = log.prefix(3).station_view()

    assert isinstance(partial, MeasurementLogStationView)
    assert [station.station_id for station in partial.stations] == [0, 1]
    assert [station.record_count for station in partial.stations] == [2, 1]
    assert [station.marked_complete for station in partial.stations] == [True, False]
    assert partial.complete_station_count == 1
    complete = partial.complete_prefix()
    assert complete.station_count == 1
    assert complete.record_count == 2
    assert complete.source_log_sha256 == partial.source_log_sha256
    assert complete.records_content_sha256 == measurement_records_content_sha256(
        log.records[:2]
    )
    assert complete.records_content_sha256 == measurement_records_sha256(
        log.records[:2]
    )
    np.testing.assert_array_equal(complete.array_view().step_id, [0, 1])
    with pytest.raises(ValueError, match="durable station prefix"):
        partial.prefix(2)
    assert partial.prefix(2, require_complete=False).record_count == 3


def test_station_view_does_not_infer_legacy_completion_markers(
    tmp_path: Path,
) -> None:
    """Station-id grouping must remain distinct from durable completion evidence."""
    log = load_measurement_log(make_measurement_log(tmp_path / "measurement-log"))

    view = log.station_view()

    assert view.station_count == 2
    assert view.complete_station_count == 0
    with pytest.raises(ValueError, match="station_complete=true"):
        view.prefix(1)
    assert view.prefix(1, require_complete=False).record_count == 2


def test_station_view_rejects_an_early_completion_marker(tmp_path: Path) -> None:
    """A true completion marker cannot precede another record in its station."""
    log = load_measurement_log(make_measurement_log(tmp_path / "measurement-log"))
    first = log.records[0]
    bad_first = replace(
        first,
        metadata={**dict(first.metadata), "station_complete": True},
    )
    bad_log = replace(log, records=(bad_first, *log.records[1:]))

    with pytest.raises(MeasurementLogValidationError, match="final record"):
        bad_log.station_view()


def test_station_view_rejects_completion_after_an_unmarked_station(
    tmp_path: Path,
) -> None:
    """A marker-aware prefix cannot skip a prior durable station boundary."""
    log = load_measurement_log(make_measurement_log(tmp_path / "measurement-log"))
    terminal = log.records[-1]
    bad_terminal = replace(
        terminal,
        metadata={**dict(terminal.metadata), "station_complete": True},
    )
    bad_log = replace(log, records=(*log.records[:-1], bad_terminal))

    with pytest.raises(MeasurementLogValidationError, match="only its final station"):
        bad_log.station_view()


def test_measurement_log_inventory_is_immutable_and_digest_compatible(
    tmp_path: Path,
) -> None:
    """The public inventory must expose actual files without changing log identity."""
    log = load_measurement_log(make_measurement_log(tmp_path / "measurement-log"))

    inventory = log.artifact_inventory()

    assert isinstance(inventory, ArtifactInventory)
    assert inventory.file_count == 7
    assert set(inventory.sha256_by_path) == {
        "environment.json",
        "forward_model_manifest.json",
        "observation_metadata.jsonl",
        "observations.npz",
        "repository_commit.txt",
        "run_manifest.json",
        "runtime_config.resolved.json",
    }
    assert inventory == measurement_log_artifact_inventory(log.path)
    assert inventory.sha256 == log.log_sha256
    with pytest.raises(TypeError):
        inventory.sha256_by_path["extra"] = "0" * 64  # type: ignore[index]


def test_generic_artifact_inventory_has_no_estimator_schema_policy(
    tmp_path: Path,
) -> None:
    """Generic inventory mechanics should hash arbitrary ordinary files only."""
    root = tmp_path / "artifacts"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "result.bin").write_bytes(b"result")
    (nested / "diagnostics.any").write_bytes(b"diagnostics")

    inventory = build_artifact_inventory(root)

    assert tuple(inventory.sha256_by_path) == (
        "nested/diagnostics.any",
        "result.bin",
    )
    assert inventory == build_artifact_inventory(root)


def test_generic_artifact_inventory_rejects_symlinks(tmp_path: Path) -> None:
    """An inventory must not silently hash a file outside its declared root."""
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "escape.bin").symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        build_artifact_inventory(root)
