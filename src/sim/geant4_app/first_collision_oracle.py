"""Statistical oracle for exact first collisions along piecewise paths.

This module is deliberately isolated from the production Geant4 runtime.  It
provides a small, batched NumPy reference for validating a native
heterogeneous first-collision implementation without approximating Geant4
transport or replacing any physics process.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class PiecewiseFirstCollisionLaw:
    """Analytic first-collision law for constant-material path segments."""

    segment_lengths: NDArray[np.float64]
    process_cross_sections: NDArray[np.float64]
    total_cross_sections: NDArray[np.float64]
    optical_depths: NDArray[np.float64]
    survival_probabilities_before: NDArray[np.float64]
    survival_probabilities_after: NDArray[np.float64]
    segment_collision_probabilities: NDArray[np.float64]
    segment_process_probabilities: NDArray[np.float64]
    no_collision_probability: float
    total_collision_probability: float
    initial_weight: float
    forced_segment_weights: NDArray[np.float64]
    forced_process_weights: NDArray[np.float64]
    no_collision_weight: float


@dataclass(frozen=True)
class FirstCollisionSamples:
    """A batched sample from an analog or collision-conditioned path law."""

    collided: NDArray[np.bool_]
    segment_indices: NDArray[np.int64]
    process_indices: NDArray[np.int64]
    local_distances: NDArray[np.float64]
    path_distances: NDArray[np.float64]
    weights: NDArray[np.float64]
    conditioned_on_collision: bool


def build_piecewise_first_collision_law(
    segment_lengths: ArrayLike,
    process_cross_sections: ArrayLike,
    *,
    initial_weight: float = 1.0,
) -> PiecewiseFirstCollisionLaw:
    """Return the exact first-collision probabilities for a segmented path.

    ``process_cross_sections[r, j]`` is the macroscopic cross-section of
    process ``j`` in segment ``r``.  Its reciprocal unit must match the unit of
    ``segment_lengths``.  Before the first interaction, a gamma's energy and
    direction are constant within each Geant4 material segment, so the
    cross-sections are constant over that segment.
    """

    lengths = np.asarray(segment_lengths, dtype=np.float64)
    cross_sections = np.asarray(process_cross_sections, dtype=np.float64)
    weight = float(initial_weight)
    _validate_inputs(lengths, cross_sections, weight)

    total_cross_sections = np.sum(cross_sections, axis=1, dtype=np.float64)
    optical_depths = lengths * total_cross_sections
    segment_conditional_probabilities = -np.expm1(-optical_depths)

    cumulative_depths = np.cumsum(optical_depths, dtype=np.float64)
    survival_after = np.exp(-cumulative_depths)
    survival_before = np.concatenate(
        (np.ones(1, dtype=np.float64), survival_after[:-1])
    )
    segment_collision_probabilities = (
        survival_before * segment_conditional_probabilities
    )

    process_fractions = np.divide(
        cross_sections,
        total_cross_sections[:, np.newaxis],
        out=np.zeros_like(cross_sections),
        where=total_cross_sections[:, np.newaxis] > 0.0,
    )
    segment_process_probabilities = (
        segment_collision_probabilities[:, np.newaxis] * process_fractions
    )
    no_collision_probability = float(survival_after[-1])
    total_collision_probability = float(
        np.sum(segment_collision_probabilities, dtype=np.float64)
    )

    probability_sum = total_collision_probability + no_collision_probability
    if not np.isclose(probability_sum, 1.0, rtol=2.0e-13, atol=2.0e-15):
        raise FloatingPointError(
            "Piecewise first-collision probabilities do not conserve mass: "
            f"{probability_sum!r}."
        )

    return PiecewiseFirstCollisionLaw(
        segment_lengths=_readonly_copy(lengths),
        process_cross_sections=_readonly_copy(cross_sections),
        total_cross_sections=_readonly_copy(total_cross_sections),
        optical_depths=_readonly_copy(optical_depths),
        survival_probabilities_before=_readonly_copy(survival_before),
        survival_probabilities_after=_readonly_copy(survival_after),
        segment_collision_probabilities=_readonly_copy(
            segment_collision_probabilities
        ),
        segment_process_probabilities=_readonly_copy(
            segment_process_probabilities
        ),
        no_collision_probability=no_collision_probability,
        total_collision_probability=total_collision_probability,
        initial_weight=weight,
        forced_segment_weights=_readonly_copy(
            weight * segment_collision_probabilities
        ),
        forced_process_weights=_readonly_copy(
            weight * segment_process_probabilities
        ),
        no_collision_weight=weight * no_collision_probability,
    )


def sample_piecewise_first_collisions(
    law: PiecewiseFirstCollisionLaw,
    *,
    rng: np.random.Generator,
    sample_count: int,
    condition_on_collision: bool,
) -> FirstCollisionSamples:
    """Sample exact segment, process, and within-segment collision distance.

    When ``condition_on_collision`` is true, every returned track represents
    the collision branch and has weight
    ``initial_weight * P(any collision)``.  The complementary no-collision
    branch has ``law.no_collision_weight``.  When false, the function samples
    the ordinary analog first-collision law and each sample retains
    ``initial_weight``.
    """

    count = _validate_sample_count(sample_count)
    process_count = law.process_cross_sections.shape[1]
    joint_probabilities = law.segment_process_probabilities.reshape(-1)

    if condition_on_collision:
        if law.total_collision_probability <= 0.0:
            raise ValueError(
                "Cannot condition on collision when the path has zero "
                "collision probability."
            )
        probabilities = (
            joint_probabilities / law.total_collision_probability
        )
        outcome_indices = rng.choice(
            joint_probabilities.size,
            size=count,
            p=_normalized_probabilities(probabilities),
        )
        collided = np.ones(count, dtype=np.bool_)
        sample_weights = np.full(
            count,
            law.initial_weight * law.total_collision_probability,
            dtype=np.float64,
        )
    else:
        probabilities = np.concatenate(
            (
                joint_probabilities,
                np.asarray(
                    [law.no_collision_probability],
                    dtype=np.float64,
                ),
            )
        )
        outcome_indices = rng.choice(
            probabilities.size,
            size=count,
            p=_normalized_probabilities(probabilities),
        )
        collided = outcome_indices < joint_probabilities.size
        sample_weights = np.full(
            count,
            law.initial_weight,
            dtype=np.float64,
        )

    segment_indices = np.full(count, -1, dtype=np.int64)
    process_indices = np.full(count, -1, dtype=np.int64)
    local_distances = np.full(count, np.nan, dtype=np.float64)
    path_distances = np.full(
        count,
        float(np.sum(law.segment_lengths, dtype=np.float64)),
        dtype=np.float64,
    )

    collision_outcomes = outcome_indices[collided]
    collision_segments = np.asarray(
        collision_outcomes // process_count,
        dtype=np.int64,
    )
    collision_processes = np.asarray(
        collision_outcomes % process_count,
        dtype=np.int64,
    )
    uniforms = rng.random(collision_segments.size)
    collision_local_distances = _sample_conditional_distances(
        law.segment_lengths[collision_segments],
        law.total_cross_sections[collision_segments],
        uniforms,
    )
    segment_offsets = np.concatenate(
        (
            np.zeros(1, dtype=np.float64),
            np.cumsum(law.segment_lengths[:-1], dtype=np.float64),
        )
    )

    segment_indices[collided] = collision_segments
    process_indices[collided] = collision_processes
    local_distances[collided] = collision_local_distances
    path_distances[collided] = (
        segment_offsets[collision_segments] + collision_local_distances
    )

    return FirstCollisionSamples(
        collided=_readonly_copy(collided),
        segment_indices=_readonly_copy(segment_indices),
        process_indices=_readonly_copy(process_indices),
        local_distances=_readonly_copy(local_distances),
        path_distances=_readonly_copy(path_distances),
        weights=_readonly_copy(sample_weights),
        conditioned_on_collision=bool(condition_on_collision),
    )


def conditional_collision_cdf(
    distances: ArrayLike,
    *,
    segment_length: float,
    total_cross_section: float,
) -> NDArray[np.float64]:
    """Evaluate the collision-distance CDF conditional on a segment hit."""

    values = np.asarray(distances, dtype=np.float64)
    length = float(segment_length)
    cross_section = float(total_cross_section)
    if not np.all(np.isfinite(values)):
        raise ValueError("Collision distances must be finite.")
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("segment_length must be finite and positive.")
    if not np.isfinite(cross_section) or cross_section <= 0.0:
        raise ValueError(
            "total_cross_section must be finite and positive."
        )

    clipped = np.clip(values, 0.0, length)
    denominator = -np.expm1(-cross_section * length)
    cdf = -np.expm1(-cross_section * clipped) / denominator
    return np.where(values < 0.0, 0.0, np.where(values >= length, 1.0, cdf))


def _sample_conditional_distances(
    segment_lengths: NDArray[np.float64],
    total_cross_sections: NDArray[np.float64],
    uniforms: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Sample truncated exponentials with stable small-depth arithmetic."""

    optical_depths = segment_lengths * total_cross_sections
    collision_probabilities = -np.expm1(-optical_depths)
    optical_distances = -np.log1p(-uniforms * collision_probabilities)
    distances = optical_distances / total_cross_sections
    return np.minimum(distances, segment_lengths)


def _validate_inputs(
    segment_lengths: NDArray[np.float64],
    process_cross_sections: NDArray[np.float64],
    initial_weight: float,
) -> None:
    """Validate shapes and physical domains for a piecewise path."""

    if segment_lengths.ndim != 1 or segment_lengths.size == 0:
        raise ValueError("segment_lengths must be a non-empty 1-D array.")
    if (
        process_cross_sections.ndim != 2
        or process_cross_sections.shape[0] != segment_lengths.size
        or process_cross_sections.shape[1] == 0
    ):
        raise ValueError(
            "process_cross_sections must have shape "
            "(segment_count, process_count)."
        )
    if not np.all(np.isfinite(segment_lengths)):
        raise ValueError("segment_lengths must be finite.")
    if np.any(segment_lengths <= 0.0):
        raise ValueError("segment_lengths must be positive.")
    if not np.all(np.isfinite(process_cross_sections)):
        raise ValueError("process_cross_sections must be finite.")
    if np.any(process_cross_sections < 0.0):
        raise ValueError("process_cross_sections must be non-negative.")
    if not np.isfinite(initial_weight) or initial_weight < 0.0:
        raise ValueError("initial_weight must be finite and non-negative.")


def _validate_sample_count(sample_count: int) -> int:
    """Return a validated positive integer sample count."""

    if isinstance(sample_count, bool) or not isinstance(
        sample_count, (int, np.integer)
    ):
        raise TypeError("sample_count must be an integer.")
    count = int(sample_count)
    if count <= 0:
        raise ValueError("sample_count must be positive.")
    return count


def _normalized_probabilities(
    probabilities: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Normalize a finite non-negative categorical probability vector."""

    total = float(np.sum(probabilities, dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Categorical probabilities must have positive mass.")
    normalized = probabilities / total
    return normalized


def _readonly_copy(values: NDArray) -> NDArray:
    """Return an owned NumPy array whose contents cannot be mutated."""

    result = np.array(values, copy=True)
    result.setflags(write=False)
    return result
