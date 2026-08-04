"""Tests for training-only physical-component candidate selection."""

from __future__ import annotations

import pytest

from scripts.build_physical_component_full_spectrum_candidate import (
    _best_calibrated_pair,
)


def test_best_pair_rejects_higher_likelihood_uncalibrated_candidate() -> None:
    """Predictive density cannot override the predeclared coverage contract."""
    scores = {
        "direct=1000000;scatter=1000": 10.0,
        "direct=1000;scatter=100": 9.0,
    }
    coverages = {
        "direct=1000000;scatter=1000": 0.79,
        "direct=1000;scatter=100": 0.80,
    }

    assert _best_calibrated_pair(
        scores=scores,
        coverages=coverages,
    ) == (1000.0, 100.0)


def test_best_pair_fails_when_training_calibration_is_impossible() -> None:
    """A candidate must not be emitted when all cross-fit coverages fail."""
    with pytest.raises(RuntimeError, match="coverage contract"):
        _best_calibrated_pair(
            scores={"direct=1000;scatter=100": 1.0},
            coverages={"direct=1000;scatter=100": 0.79},
        )
