"""Tests for the renderer-only private CUI truth socket."""

from __future__ import annotations

from collections.abc import Callable
import stat
from pathlib import Path

import numpy as np
import pytest

from runtime.cui_truth_overlay import (
    CUITruthOverlay,
    CUITruthOverlaySocketServer,
    load_cui_truth_overlay,
)


def _truth_payload() -> dict[str, object]:
    """Return one valid multi-isotope private overlay payload."""
    return {
        "schema_version": 1,
        "semantics": "evaluation_cui_overlay_only_not_estimator_input",
        "true_sources": {
            "Co-60": [[1.0, 2.0, 3.0]],
            "Cs-137": [[4.0, 5.0, 0.5], [6.0, 7.0, 1.5]],
        },
        "true_strengths": {
            "Co-60": [700_000.0],
            "Cs-137": [400_000.0, 500_000.0],
        },
    }


def test_private_cui_truth_socket_round_trip_and_cleanup(tmp_path: Path) -> None:
    """Only the renderer socket should deliver aligned immutable truth arrays."""
    endpoint = tmp_path / "cui-truth.sock"
    server = CUITruthOverlaySocketServer(endpoint, _truth_payload())
    try:
        assert stat.S_IMODE(endpoint.stat().st_mode) == 0o600
        overlay = load_cui_truth_overlay(endpoint, connect_timeout_s=2.0)
        assert server.served
        np.testing.assert_array_equal(
            overlay.true_sources["Cs-137"],
            np.asarray([[4.0, 5.0, 0.5], [6.0, 7.0, 1.5]]),
        )
        np.testing.assert_array_equal(
            overlay.true_strengths["Co-60"],
            np.asarray([700_000.0]),
        )
        assert not overlay.true_sources["Cs-137"].flags.writeable
        assert not overlay.true_strengths["Co-60"].flags.writeable
    finally:
        server.close()
    assert not endpoint.exists()


@pytest.mark.parametrize(
    "mutation",
    (
        lambda payload: payload.update({"legacy_sources": {}}),
        lambda payload: payload["true_strengths"].update({"Am-241": [1.0]}),
        lambda payload: payload["true_strengths"].update({"Co-60": [0.0]}),
    ),
)
def test_cui_truth_overlay_rejects_unknown_misaligned_or_zero_values(
    mutation: Callable[[dict[str, object]], object],
) -> None:
    """Private truth parsing must fail closed without aliases or defaults."""
    payload = _truth_payload()
    mutation(payload)
    with pytest.raises((TypeError, ValueError)):
        CUITruthOverlay.from_truth_payload(payload)


def test_cui_truth_socket_refuses_to_replace_existing_endpoint(
    tmp_path: Path,
) -> None:
    """A stale or foreign endpoint must not be overwritten."""
    endpoint = tmp_path / "occupied.sock"
    endpoint.write_text("foreign", encoding="utf-8")
    with pytest.raises(FileExistsError):
        CUITruthOverlaySocketServer(endpoint, _truth_payload())
    assert endpoint.read_text(encoding="utf-8") == "foreign"
