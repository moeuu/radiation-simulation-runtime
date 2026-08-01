"""Tests for the heterogeneous first-collision statistical oracle."""

from __future__ import annotations

import numpy as np
import pytest

from sim.geant4_app.first_collision_oracle import (
    build_piecewise_first_collision_law,
    conditional_collision_cdf,
    sample_piecewise_first_collisions,
)


def test_piecewise_law_matches_two_material_analytic_probabilities() -> None:
    """Segment, process, and escape masses must match closed-form values."""

    lengths = np.asarray([2.0, 1.0], dtype=np.float64)
    cross_sections = np.asarray(
        [
            [0.1, 0.2],
            [0.4, 0.0],
        ],
        dtype=np.float64,
    )
    law = build_piecewise_first_collision_law(
        lengths,
        cross_sections,
        initial_weight=2.5,
    )

    expected_optical_depths = np.asarray([0.6, 0.4])
    expected_segment_probabilities = np.asarray(
        [
            1.0 - np.exp(-0.6),
            np.exp(-0.6) * (1.0 - np.exp(-0.4)),
        ]
    )
    expected_process_probabilities = np.asarray(
        [
            [
                expected_segment_probabilities[0] / 3.0,
                2.0 * expected_segment_probabilities[0] / 3.0,
            ],
            [expected_segment_probabilities[1], 0.0],
        ]
    )

    np.testing.assert_allclose(
        law.optical_depths,
        expected_optical_depths,
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        law.segment_collision_probabilities,
        expected_segment_probabilities,
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        law.segment_process_probabilities,
        expected_process_probabilities,
        rtol=0.0,
        atol=1.0e-15,
    )
    assert law.no_collision_probability == pytest.approx(np.exp(-1.0))
    assert (
        law.total_collision_probability + law.no_collision_probability
    ) == pytest.approx(1.0)
    np.testing.assert_allclose(
        law.forced_segment_weights,
        2.5 * expected_segment_probabilities,
    )
    np.testing.assert_allclose(
        law.forced_process_weights,
        2.5 * expected_process_probabilities,
    )
    assert (
        np.sum(law.forced_segment_weights) + law.no_collision_weight
    ) == pytest.approx(2.5)


def test_unconditional_samples_match_segment_process_and_escape_masses() -> None:
    """The analog sampler must reproduce every joint categorical outcome."""

    law = build_piecewise_first_collision_law(
        [1.5, 0.75, 2.0],
        [
            [0.15, 0.05],
            [0.0, 0.4],
            [0.03, 0.07],
        ],
    )
    sample_count = 300_000
    samples = sample_piecewise_first_collisions(
        law,
        rng=np.random.default_rng(20260729),
        sample_count=sample_count,
        condition_on_collision=False,
    )

    process_count = law.process_cross_sections.shape[1]
    observed = np.zeros(
        law.segment_process_probabilities.size + 1,
        dtype=np.float64,
    )
    collided_outcomes = (
        samples.segment_indices[samples.collided] * process_count
        + samples.process_indices[samples.collided]
    )
    observed[:-1] = np.bincount(
        collided_outcomes,
        minlength=law.segment_process_probabilities.size,
    )
    observed[-1] = np.count_nonzero(~samples.collided)
    observed /= sample_count
    expected = np.concatenate(
        (
            law.segment_process_probabilities.reshape(-1),
            np.asarray([law.no_collision_probability]),
        )
    )

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=2.5e-3)
    assert np.all(samples.weights == 1.0)
    assert np.all(samples.segment_indices[~samples.collided] == -1)
    assert np.all(samples.process_indices[~samples.collided] == -1)
    assert np.all(np.isnan(samples.local_distances[~samples.collided]))


def test_conditioned_samples_match_process_mixture_and_distance_cdf() -> None:
    """Forced samples must retain the exact process and truncated-path laws."""

    segment_length = 2.0
    total_cross_section = 1.0
    law = build_piecewise_first_collision_law(
        [segment_length],
        [[0.3, 0.7]],
        initial_weight=4.0,
    )
    samples = sample_piecewise_first_collisions(
        law,
        rng=np.random.default_rng(8417),
        sample_count=250_000,
        condition_on_collision=True,
    )

    observed_process_zero = np.mean(samples.process_indices == 0)
    assert observed_process_zero == pytest.approx(0.3, abs=2.5e-3)
    assert np.all(samples.collided)
    assert np.all(samples.segment_indices == 0)
    assert np.all(
        samples.weights
        == 4.0 * law.total_collision_probability
    )
    assert np.all(samples.local_distances >= 0.0)
    assert np.all(samples.local_distances <= segment_length)

    for distance in (0.25, 0.75, 1.5):
        empirical_cdf = np.mean(samples.local_distances <= distance)
        expected_cdf = conditional_collision_cdf(
            [distance],
            segment_length=segment_length,
            total_cross_section=total_cross_section,
        )[0]
        assert empirical_cdf == pytest.approx(expected_cdf, abs=2.5e-3)


def test_conditioned_sampler_selects_heterogeneous_segments_by_mass() -> None:
    """Global forced sampling must include upstream survival in segment choice."""

    law = build_piecewise_first_collision_law(
        [1.0, 2.0, 0.5],
        [
            [0.2, 0.1],
            [0.0, 0.8],
            [0.5, 0.5],
        ],
        initial_weight=3.0,
    )
    samples = sample_piecewise_first_collisions(
        law,
        rng=np.random.default_rng(9031),
        sample_count=250_000,
        condition_on_collision=True,
    )
    observed = np.bincount(
        samples.segment_indices,
        minlength=law.segment_lengths.size,
    ) / samples.segment_indices.size
    expected = (
        law.segment_collision_probabilities
        / law.total_collision_probability
    )

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=2.5e-3)


def test_zero_cross_section_path_has_only_no_collision_mass() -> None:
    """A vacuum-like path must escape and reject collision conditioning."""

    law = build_piecewise_first_collision_law(
        [1.0, 2.0],
        [[0.0, 0.0], [0.0, 0.0]],
        initial_weight=7.0,
    )

    assert law.total_collision_probability == 0.0
    assert law.no_collision_probability == 1.0
    assert law.no_collision_weight == 7.0
    np.testing.assert_array_equal(law.forced_segment_weights, [0.0, 0.0])

    analog_samples = sample_piecewise_first_collisions(
        law,
        rng=np.random.default_rng(2),
        sample_count=32,
        condition_on_collision=False,
    )
    assert not np.any(analog_samples.collided)

    with pytest.raises(ValueError, match="zero collision probability"):
        sample_piecewise_first_collisions(
            law,
            rng=np.random.default_rng(3),
            sample_count=1,
            condition_on_collision=True,
        )


def test_extreme_optical_depths_remain_finite_and_conserve_mass() -> None:
    """Small and large optical depths must avoid cancellation and NaNs."""

    law = build_piecewise_first_collision_law(
        [1.0e-12, 1000.0],
        [[1.0e-3, 0.0], [0.2, 0.8]],
    )

    assert np.all(np.isfinite(law.segment_collision_probabilities))
    assert np.all(np.isfinite(law.segment_process_probabilities))
    assert (
        law.total_collision_probability + law.no_collision_probability
    ) == pytest.approx(1.0)
    assert law.segment_collision_probabilities[0] == pytest.approx(
        1.0e-15,
        rel=1.0e-12,
    )
    assert law.no_collision_probability == 0.0


@pytest.mark.parametrize(
    ("lengths", "cross_sections", "initial_weight", "message"),
    [
        ([], [], 1.0, "non-empty 1-D"),
        ([1.0], [0.1, 0.2], 1.0, "shape"),
        ([1.0, 2.0], [[0.1]], 1.0, "shape"),
        ([0.0], [[0.1]], 1.0, "positive"),
        ([np.nan], [[0.1]], 1.0, "finite"),
        ([1.0], [[-0.1]], 1.0, "non-negative"),
        ([1.0], [[np.inf]], 1.0, "finite"),
        ([1.0], [[0.1]], -1.0, "non-negative"),
    ],
)
def test_invalid_piecewise_paths_fail_fast(
    lengths: object,
    cross_sections: object,
    initial_weight: float,
    message: str,
) -> None:
    """Malformed path contracts must fail instead of changing physics."""

    with pytest.raises(ValueError, match=message):
        build_piecewise_first_collision_law(
            lengths,
            cross_sections,
            initial_weight=initial_weight,
        )


def test_sample_count_contract_fails_fast() -> None:
    """The sampler must reject non-positive and non-integral batch sizes."""

    law = build_piecewise_first_collision_law([1.0], [[0.2]])

    with pytest.raises(ValueError, match="positive"):
        sample_piecewise_first_collisions(
            law,
            rng=np.random.default_rng(1),
            sample_count=0,
            condition_on_collision=False,
        )
    with pytest.raises(TypeError, match="integer"):
        sample_piecewise_first_collisions(
            law,
            rng=np.random.default_rng(1),
            sample_count=1.5,  # type: ignore[arg-type]
            condition_on_collision=False,
        )
