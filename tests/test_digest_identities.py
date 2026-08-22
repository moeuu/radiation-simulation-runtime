"""Golden compatibility tests for algorithm-bound runtime digests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime.prefix import (
    covered_station_boundaries_digest,
    covered_station_boundaries_sha256,
    measurement_records_digest,
    measurement_records_sha256,
)
from runtime.provenance import DigestIdentity
from runtime.records import measurement_record_from_payload


FIXTURE = Path(__file__).parent / "fixtures" / "digest_identity_v1.json"


def test_runtime_digest_algorithms_match_golden_fixture() -> None:
    """Persisted digest bytes and algorithm IDs must remain stable together."""
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    record = measurement_record_from_payload(payload["record"])
    records = (record,)

    record_digest = measurement_records_digest(records)
    boundary_digest = covered_station_boundaries_digest(
        records,
        source_run_id=payload["source_run_id"],
    )

    assert record_digest.to_payload() == payload["record_digest"]
    assert boundary_digest.to_payload() == payload["station_boundaries_digest"]
    assert measurement_records_sha256(records) == record_digest.sha256
    assert covered_station_boundaries_sha256(
        records,
        source_run_id=payload["source_run_id"],
    ) == boundary_digest.sha256


def test_digest_identity_round_trip_is_exact() -> None:
    """Digest DTOs must reject missing, unknown, and untyped wire fields."""
    expected = DigestIdentity(
        algorithm="example.contract-v1+sha256",
        sha256="0" * 64,
    )

    assert DigestIdentity.from_payload(expected.to_payload()) == expected
    with pytest.raises(ValueError, match="exactly"):
        DigestIdentity.from_payload({"algorithm": expected.algorithm})
    with pytest.raises(TypeError, match="strings"):
        DigestIdentity.from_payload(
            {"algorithm": expected.algorithm, "sha256": 0}
        )
