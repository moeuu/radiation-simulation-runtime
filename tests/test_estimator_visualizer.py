"""Fail-close tests for visual-only estimator payload parsing."""

from __future__ import annotations

import numpy as np
import pytest

from sim.isaacsim_app.estimator_visualizer import (
    _require_point_mapping,
    _require_vector,
)


@pytest.mark.parametrize(
    "payload",
    (
        {"Cs-137": [[1.0, 2.0]]},
        {"Cs-137": [[1.0, 2.0, 3.0, 4.0]]},
        {"Cs-137": [["1", "2", "3"]]},
        {"Cs-137": [[1.0, 2.0, np.nan]]},
        {7: [[1.0, 2.0, 3.0]]},
    ),
)
def test_estimator_points_reject_lossy_shape_and_type_recovery(
    payload: object,
) -> None:
    """Malformed marker positions must not be truncated, emptied, or stringified."""
    with pytest.raises((TypeError, ValueError)):
        _require_point_mapping(payload, location="sample_positions")


@pytest.mark.parametrize(
    ("value", "size"),
    (
        (None, 1),
        ([], 1),
        ([1.0, 2.0], 1),
        (["1"], 1),
        ([np.nan], 1),
        ([-1.0], 1),
    ),
)
def test_estimator_vectors_reject_padding_truncation_and_coercion(
    value: object,
    size: int,
) -> None:
    """Marker weights and strengths must have one exact finite vector shape."""
    with pytest.raises(ValueError):
        _require_vector(value, size, location="sample_weights.Cs-137")


def test_estimator_visual_vectors_accept_exact_empty_and_nonnegative_values() -> None:
    """Exact empty and populated vectors remain valid visual payloads."""
    assert _require_vector([], 0, location="empty").shape == (0,)
    assert np.array_equal(
        _require_vector([0.0, 0.25], 2, location="weights"),
        np.asarray((0.0, 0.25)),
    )
