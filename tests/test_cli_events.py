"""Tests for estimator-neutral CLI JSON event framing."""

from __future__ import annotations

import pytest

from runtime.cli_events import CLIJSONEventFraming


def test_cli_event_framing_round_trips_deterministically() -> None:
    """Machine events must retain one stable frame distinct from diagnostics."""
    framing = CLIJSONEventFraming("machine-event ")

    encoded = framing.encode({"type": "ready", "schema_version": 1})

    assert encoded == (
        'machine-event {"schema_version": 1, "type": "ready"}\n'
    )
    assert framing.parse(encoded) == {"schema_version": 1, "type": "ready"}
    assert framing.try_parse("ordinary diagnostic\n") is None


@pytest.mark.parametrize(
    "line",
    [
        'machine-event {"type":"ready","type":"again"}',
        'machine-event {"value":NaN}',
        "machine-event []",
    ],
)
def test_cli_event_framing_rejects_non_strict_objects(line: str) -> None:
    """Malformed or non-object machine events must never pass as valid frames."""
    framing = CLIJSONEventFraming("machine-event ")

    with pytest.raises((TypeError, ValueError)):
        framing.parse(line)
