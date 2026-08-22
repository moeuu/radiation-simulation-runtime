"""Tests for the RDM-versus-independent-line comparison contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pytest
import spectrum.decay_cascade_comparison as comparison_module

from spectrum.decay_cascade_comparison import (
    DecayCascadeComparisonDesign,
    _execute_comparison_cases,
    analyze_decay_cascade_cases,
    detector_reference_acceptance,
    load_decay_cascade_comparison_design,
    planned_parent_decays,
    sphere_solid_angle_fraction,
)


def _design(*, minimum_pulses: int = 300) -> DecayCascadeComparisonDesign:
    """Return one small deterministic unit-test design."""
    return DecayCascadeComparisonDesign(
        isotopes=("Co-60",),
        distances_m=(1.5,),
        target_expected_gamma_intersections=1000,
        maximum_parent_decays_per_case=25_000_000,
        independent_line_histories_per_case=200_000,
        energy_max_keV=3400.0,
        comparison_bin_width_keV=20.0,
        minimum_rdm_detected_pulses=minimum_pulses,
        maximum_common_band_tv=0.03,
        maximum_coincidence_excess_fraction=0.01,
        bootstrap_samples=512,
        seed=12345,
        total_geant4_threads=6,
        case_workers=2,
        timeout_s=30.0,
    )


def _case_records(
    tmp_path: Path,
    *,
    rdm: np.ndarray,
    independent: np.ndarray,
) -> list[dict[str, object]]:
    """Write two spectra and return the minimal acquisition records."""
    records: list[dict[str, object]] = []
    for model, spectrum in (
        ("geant4_radioactive_decay", rdm),
        ("independent_gamma_lines", independent),
    ):
        path = tmp_path / f"{model}.npy"
        np.save(path, spectrum)
        records.append(
            {
                "isotope": "Co-60",
                "distance_m": 1.5,
                "emission_model": model,
                "spectrum_path": path.as_posix(),
            }
        )
    return records


def test_predeclared_design_loads_without_hidden_defaults() -> None:
    """The repository design must be complete, finite, and hashable."""
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "validation"
        / "decay_cascade_comparison.json"
    )
    design = load_decay_cascade_comparison_design(path)
    assert design.isotopes == ("Co-60", "Eu-152", "Eu-154")
    assert design.distances_m == (1.5, 2.5, 4.0)
    assert design.energy_max_keV >= 2.0 * 1332.492
    assert len(design.design_sha256) == 64
    assert json.loads(path.read_text(encoding="utf-8"))["case_workers"] == 3


def test_parent_decay_quota_uses_geometry_and_has_a_hard_cap() -> None:
    """Long-distance RDM cases must receive more but bounded analog histories."""
    design = _design()
    near = planned_parent_decays(
        design,
        isotope="Co-60",
        distance_m=1.5,
        detector_radius_m=0.038,
    )
    far = planned_parent_decays(
        design,
        isotope="Co-60",
        distance_m=4.0,
        detector_radius_m=0.038,
    )
    assert 0 < near < far <= design.maximum_parent_decays_per_case
    assert detector_reference_acceptance(0.038) == pytest.approx(
        sphere_solid_angle_fraction(1.0, 0.038)
    )
    assert sphere_solid_angle_fraction(4.0, 0.038) < (
        sphere_solid_angle_fraction(1.5, 0.038)
    )
    assert design.threads_per_case == 3


def test_identical_high_count_mark_spectra_pass(tmp_path: Path) -> None:
    """Matched high-count line spectra must pass the predeclared equivalence gate."""
    design = _design()
    spectrum = np.zeros(1701, dtype=np.float64)
    spectrum[586] = 50_000
    spectrum[666] = 50_000
    analysis = analyze_decay_cascade_cases(
        design,
        _case_records(tmp_path, rdm=spectrum, independent=spectrum.copy()),
    )
    assert analysis["overall_status"] == "independent_basis_adequate"


def test_co60_sum_peak_requires_cascade_aware_model(tmp_path: Path) -> None:
    """A material 2506-keV RDM-only sum peak must fail the coincidence gate."""
    design = _design()
    independent = np.zeros(1701, dtype=np.float64)
    independent[586] = 50_000
    independent[666] = 50_000
    rdm = independent.copy()
    rdm[1253] = 20_000
    analysis = analyze_decay_cascade_cases(
        design,
        _case_records(tmp_path, rdm=rdm, independent=independent),
    )
    case = analysis["case_results"][0]
    assert analysis["overall_status"] == "cascade_aware_model_required"
    assert case["coincidence_excess_lower_95"] > 0.01


def test_low_count_rdm_case_is_inconclusive(tmp_path: Path) -> None:
    """A statistically weak RDM case must never certify the line basis."""
    design = _design(minimum_pulses=300)
    independent = np.zeros(1701, dtype=np.float64)
    independent[586] = 10_000
    rdm = np.zeros(1701, dtype=np.float64)
    rdm[586] = 20
    analysis = analyze_decay_cascade_cases(
        design,
        _case_records(tmp_path, rdm=rdm, independent=independent),
    )
    assert analysis["overall_status"] == "inconclusive"


def test_case_runtime_selects_bounded_ordered_process_pool(monkeypatch) -> None:
    """The heavy isotope/distance dimension must select process parallelism."""
    observed: dict[str, object] = {}

    class RecordingExecutor:
        """Record pool selection while evaluating the tiny oracle serially."""

        def __init__(self, *, max_workers: int) -> None:
            """Store the requested process bound."""
            observed["max_workers"] = max_workers

        def __enter__(self) -> "RecordingExecutor":
            """Return this deterministic context manager."""
            return self

        def __exit__(self, *args: object) -> None:
            """Leave the deterministic test context."""

        def map(
            self,
            function: Callable[[object], object],
            values: Iterable[object],
        ) -> list[object]:
            """Evaluate the ordered scalar oracle for comparison."""
            observed["function"] = function
            return [function(value) for value in values]

    monkeypatch.setattr(
        comparison_module,
        "ProcessPoolExecutor",
        RecordingExecutor,
    )
    monkeypatch.setattr(
        comparison_module,
        "_run_comparison_case",
        lambda value: {"value": value},
    )
    result = _execute_comparison_cases([1, 2, 3], max_workers=2)
    assert result == [{"value": 1}, {"value": 2}, {"value": 3}]
    assert observed["max_workers"] == 2
