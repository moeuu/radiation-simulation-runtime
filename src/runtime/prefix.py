"""Stable digests for ordered live MeasurementLog record prefixes."""

from __future__ import annotations

from hashlib import sha256
from collections.abc import Sequence

from runtime.measurement_log import (
    MeasurementLogRecord,
    measurement_records_content_sha256,
)
from runtime.provenance import DigestIdentity, canonical_json_bytes


MEASUREMENT_RECORDS_DIGEST_ALGORITHM = (
    "rotating-shield-runtime.measurement-records-v2+canonical-json-sha256"
)
STATION_BOUNDARIES_DIGEST_ALGORITHM = (
    "rotating-shield-runtime.station-boundaries-v1+canonical-json-sha256"
)


def measurement_records_sha256(
    records: Sequence[MeasurementLogRecord],
) -> str:
    """Hash an ordered sequence of complete raw observation records."""
    rows = tuple(records)
    if not rows:
        raise ValueError("At least one record is required for a lineage digest.")
    return measurement_records_content_sha256(rows)


def measurement_records_digest(
    records: Sequence[MeasurementLogRecord],
) -> DigestIdentity:
    """Return the v2 record digest together with its stable algorithm ID."""
    return DigestIdentity(
        algorithm=MEASUREMENT_RECORDS_DIGEST_ALGORITHM,
        sha256=measurement_records_sha256(records),
    )


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


def covered_station_boundaries_digest(
    records: Sequence[MeasurementLogRecord],
    *,
    source_run_id: str,
) -> DigestIdentity:
    """Return the station-boundary digest with its stable algorithm ID."""
    return DigestIdentity(
        algorithm=STATION_BOUNDARIES_DIGEST_ALGORITHM,
        sha256=covered_station_boundaries_sha256(
            records,
            source_run_id=source_run_id,
        ),
    )


__all__ = [
    "MEASUREMENT_RECORDS_DIGEST_ALGORITHM",
    "STATION_BOUNDARIES_DIGEST_ALGORITHM",
    "covered_station_boundaries_digest",
    "covered_station_boundaries_sha256",
    "measurement_records_digest",
    "measurement_records_sha256",
]
