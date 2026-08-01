"""Causal prefix publication for raw full-spectrum MeasurementLog v2."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Sequence

from runtime.measurement_log import (
    MeasurementLogRecord,
    load_measurement_log,
    measurement_log_sha256,
    write_measurement_log,
)
from runtime.provenance import canonical_json_bytes


@dataclass(frozen=True, slots=True)
class MeasurementLogPrefix:
    """Describe one published causal raw-log prefix."""

    output_dir: Path
    record_count: int
    covered_step_ids: tuple[int, ...]
    data_cutoff_step: int
    data_cutoff_station: int
    covered_records_sha256: str
    measurement_log_sha256: str


def _record_payload(record: MeasurementLogRecord) -> dict[str, object]:
    """Return all raw estimator-visible record fields deterministically."""
    return {
        "step_id": record.step_id,
        "action_id": record.action_id,
        "station_id": record.station_id,
        "detector_pose_xyz": list(record.detector_pose_xyz),
        "detector_quat_wxyz": list(record.detector_quat_wxyz),
        "fe_orientation_index": record.fe_orientation_index,
        "pb_orientation_index": record.pb_orientation_index,
        "live_time_s": record.live_time_s,
        "travel_time_s": record.travel_time_s,
        "shield_actuation_time_s": record.shield_actuation_time_s,
        "energy_bin_edges_keV": record.energy_bin_edges_keV.tolist(),
        "spectrum_counts": record.spectrum_counts.tolist(),
        "metadata": dict(record.metadata),
    }


def measurement_records_sha256(
    records: Sequence[MeasurementLogRecord],
) -> str:
    """Hash an ordered sequence of complete raw observation records."""
    rows = tuple(records)
    if not rows:
        raise ValueError("At least one record is required for a lineage digest.")
    return sha256(
        canonical_json_bytes([_record_payload(record) for record in rows])
    ).hexdigest()


def covered_station_boundaries_sha256(
    records: Sequence[MeasurementLogRecord],
    *,
    source_run_id: str,
) -> str:
    """Hash explicit station-end markers in one causal prefix."""
    selected = tuple(records)
    if not selected:
        raise ValueError("Station-boundary coverage requires at least one record.")
    entries: list[dict[str, int]] = []
    for index, record in enumerate(selected):
        if record.metadata.get("station_complete") is not True:
            continue
        if (
            index + 1 < len(selected)
            and selected[index + 1].station_id == record.station_id
        ):
            raise ValueError("A station_complete marker precedes another station row.")
        entries.append(
            {"station_id": record.station_id, "terminal_step_id": record.step_id}
        )
    if not entries or entries[-1]["terminal_step_id"] != selected[-1].step_id:
        raise ValueError("A causal prefix must end at station_complete=true.")
    if {entry["station_id"] for entry in entries} != {
        record.station_id for record in selected
    }:
        raise ValueError("Every station in a causal prefix must declare its end marker.")
    payload = {
        "schema_version": 1,
        "source_run_id": str(source_run_id),
        "station_end_steps": entries,
    }
    return sha256(canonical_json_bytes(payload)).hexdigest()


def _prefix_stop_index(
    records: tuple[MeasurementLogRecord, ...],
    *,
    cutoff_step: int,
    cutoff_station: int,
    assert_station_complete: bool,
) -> int:
    """Validate an exact station boundary and return its record index."""
    matching = [
        index for index, record in enumerate(records)
        if record.step_id == int(cutoff_step)
    ]
    if not matching:
        raise ValueError(f"cutoff_step {cutoff_step} is absent from the log.")
    stop = matching[0]
    record = records[stop]
    if record.station_id != int(cutoff_station):
        raise ValueError(
            f"cutoff_step {cutoff_step} belongs to station {record.station_id}, "
            f"not cutoff_station {cutoff_station}."
        )
    if stop + 1 < len(records) and records[stop + 1].station_id == record.station_id:
        raise ValueError(f"cutoff_step {cutoff_step} is not station-complete.")
    if record.metadata.get("station_complete") is not True and not assert_station_complete:
        raise ValueError(
            "The cutoff record lacks metadata.station_complete=true; an external "
            "validated schedule must explicitly attest this boundary."
        )
    return stop


def materialize_measurement_log_prefix(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    cutoff_step: int,
    cutoff_station: int,
    assert_station_complete: bool = False,
) -> MeasurementLogPrefix:
    """Publish one truth-free raw-log prefix using the shared writer."""
    source = Path(run_dir).resolve()
    target = Path(output_dir).resolve()
    if not isinstance(assert_station_complete, bool):
        raise TypeError("assert_station_complete must be a boolean.")
    if source == target or source in target.parents:
        raise ValueError("output_dir must not be the source log or its descendant.")
    log = load_measurement_log(source)
    stop = _prefix_stop_index(
        log.records,
        cutoff_step=cutoff_step,
        cutoff_station=cutoff_station,
        assert_station_complete=assert_station_complete,
    )
    records = log.records[: stop + 1]
    writer_marked_complete = records[-1].metadata.get("station_complete") is True
    prefix_metadata = {
        "schema_version": 1,
        "source_run_id": log.context.run_id,
        "data_cutoff_step": records[-1].step_id,
        "data_cutoff_station": records[-1].station_id,
        "station_boundary_attestation": (
            "covered_prefix_markers_v1"
            if writer_marked_complete
            else "external_validated_schedule"
        ),
    }
    if writer_marked_complete:
        prefix_metadata["covered_station_boundaries_sha256"] = (
            covered_station_boundaries_sha256(
                records,
                source_run_id=log.context.run_id,
            )
        )
    metadata = dict(log.context.metadata)
    metadata.pop("station_boundary_attestation", None)
    metadata["measurement_log_prefix"] = prefix_metadata
    saved = write_measurement_log(
        target,
        run_id=log.context.run_id,
        repository_commit=log.context.repository_commit,
        runtime_config=log.context.runtime_config,
        environment=log.context.environment,
        forward_model_manifest=log.context.forward_model_manifest,
        isotopes=log.context.isotopes,
        records=records,
        metadata=metadata,
        obstacle_layout_path=log.context.obstacle_layout_path,
        source_layout_path=None,
    )
    return MeasurementLogPrefix(
        output_dir=saved.path,
        record_count=len(records),
        covered_step_ids=tuple(record.step_id for record in records),
        data_cutoff_step=records[-1].step_id,
        data_cutoff_station=records[-1].station_id,
        covered_records_sha256=measurement_records_sha256(records),
        measurement_log_sha256=measurement_log_sha256(saved.path),
    )


__all__ = [
    "MeasurementLogPrefix",
    "covered_station_boundaries_sha256",
    "materialize_measurement_log_prefix",
    "measurement_records_sha256",
]
