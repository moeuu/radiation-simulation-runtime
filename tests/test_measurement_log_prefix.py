"""Tests for shared causal MeasurementLog prefix publication."""

import json
from pathlib import Path

from runtime.measurement_log import load_measurement_log
from runtime.prefix import materialize_measurement_log_prefix
from tests.runtime_test_support import make_measurement_log


PREFIX_FIXTURE = (
    Path(__file__).parent / "fixtures" / "measurement_log_prefix_v1.json"
)


def test_prefix_is_published_by_shared_runtime(tmp_path) -> None:
    """The shared writer must publish a strict station-complete raw prefix."""
    source = make_measurement_log(
        tmp_path / "source",
        station_complete_markers=True,
    )

    prefix = materialize_measurement_log_prefix(
        source,
        tmp_path / "prefix",
        cutoff_step=1,
        cutoff_station=0,
    )
    loaded = load_measurement_log(prefix.output_dir)

    assert prefix.record_count == 2
    assert prefix.covered_step_ids == (0, 1)
    assert prefix.covered_records_digest.sha256 == prefix.covered_records_sha256
    assert prefix.measurement_log_digest.sha256 == prefix.measurement_log_sha256
    assert len(loaded.records) == 2
    manifest_prefix = loaded.context.metadata["measurement_log_prefix"]
    expected_prefix = json.loads(PREFIX_FIXTURE.read_text(encoding="utf-8"))

    assert manifest_prefix == expected_prefix
    assert manifest_prefix["covered_records_digest"] == (
        prefix.covered_records_digest.to_payload()
    )
    assert manifest_prefix["covered_records_sha256"] == (
        prefix.covered_records_digest.sha256
    )
