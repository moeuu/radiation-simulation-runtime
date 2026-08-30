"""Tests for the same-physics Geant4 thread-scaling benchmark."""

from __future__ import annotations

from dataclasses import replace

import pytest

from scripts.benchmark_geant4_thread_scaling import (
    PHYSICS_METADATA_KEYS,
    BenchmarkRun,
    SidecarResponse,
    _affinity_by_thread_count,
    _summaries,
    _validate_same_physics,
)


def _response() -> SidecarResponse:
    """Return one minimal physics-complete benchmark response."""
    metadata = {key: f"value-{key}" for key in PHYSICS_METADATA_KEYS}
    metadata["num_primaries"] = "1000"
    metadata["total_track_steps"] = "5000"
    return SidecarResponse(
        metadata=metadata,
        spectrum=(10.0, 20.0),
        variance=(10.0, 20.0),
    )


def _run(thread_count: int, repeat_index: int, rate: float) -> BenchmarkRun:
    """Return one compact deterministic throughput record."""
    return BenchmarkRun(
        thread_count=thread_count,
        cpu_affinity=tuple(range(thread_count)),
        repeat_index=repeat_index,
        process_wall_s=2.0,
        num_primaries=1000,
        primaries_per_s=rate,
        effective_entries_per_s=0.5 * rate,
        total_track_steps=5000,
        total_spectrum_counts=30.0,
        spectrum_sha256="0" * 64,
    )


def test_thread_benchmark_affinity_uses_one_then_two_siblings() -> None:
    """The benchmark must distinguish physical cores from SMT siblings."""
    topology = {core: [core, core + 16] for core in range(16)}

    affinity = _affinity_by_thread_count(topology)

    assert affinity[16] == tuple(range(16))
    assert affinity[32] == tuple(range(32))


def test_thread_benchmark_validates_physics_and_monte_carlo_agreement() -> None:
    """Thread comparisons must fail closed on physics or spectral mismatch."""
    reference = _response()
    equivalent = replace(
        reference,
        spectrum=(11.0, 19.0),
        variance=(11.0, 19.0),
    )

    _validate_same_physics([reference, equivalent])

    changed_metadata = dict(reference.metadata)
    changed_metadata["physics_profile"] = "changed"
    with pytest.raises(RuntimeError, match="physics_profile"):
        _validate_same_physics(
            [reference, replace(reference, metadata=changed_metadata)]
        )
    with pytest.raises(RuntimeError, match="six-sigma"):
        _validate_same_physics(
            [
                reference,
                replace(
                    reference,
                    spectrum=(10_000.0, 20_000.0),
                    variance=(1.0, 1.0),
                ),
            ]
        )


def test_thread_benchmark_summary_selects_measured_throughput_winner() -> None:
    """The recommendation must follow median native transport throughput."""
    runs = [
        _run(16, 0, 100.0),
        _run(16, 1, 110.0),
        _run(32, 0, 125.0),
        _run(32, 1, 130.0),
    ]

    summary = _summaries(runs)

    assert summary["recommended_thread_count"] == 32
    assert summary["smt_over_physical_speedup"] == pytest.approx(127.5 / 105.0)
