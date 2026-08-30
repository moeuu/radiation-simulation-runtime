"""Tests for the isotope-independent full-detector Green operator."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from spectrum.detector_green_construction import (
    DETECTOR_ENERGY_RESOLUTION_CONTRACT_SHA256,
    build_detector_green_operator,
    catalog_independent_energy_nodes_keV,
    gaussian_resolution_operator,
    impact_parameter_edges_for_equal_solid_angle_strata,
)
from spectrum.detector_green_operator import (
    DETECTOR_GREEN_BINARY_BASENAME,
    DetectorGreenOperator,
    canonical_json_bytes,
)
from spectrum.detector_cone_scatter import (
    DETECTOR_CONE_SCATTER_MAXIMUM_DISTANCE_M,
    DETECTOR_CONE_SCATTER_RESPONSE_ID,
    build_detector_cone_scatter_grid,
)
from spectrum.detector_green_provenance import (
    DETECTOR_GREEN_IMPLEMENTATION_PATHS,
    detector_green_implementation_bundle_sha256,
)
from spectrum.detector_green_validation import (
    DETECTOR_GREEN_VALIDATION_CONTRACT_SHA256,
    DETECTOR_GREEN_VALIDATION_ID,
    DETECTOR_GREEN_VALIDATION_MANIFEST_BASENAME,
    DETECTOR_GREEN_VALIDATION_RAW_BASENAME,
    DETECTOR_GREEN_VALIDATION_SCHEMA_VERSION,
    build_detector_green_validation_manifest,
    detector_green_holdout_energies_keV,
    load_detector_green_validation_manifest,
)
from spectrum.additive_scatter import PhysicsOnlyNoncollidedTransportResponse
from spectrum.library import Nuclide, NuclideLine, default_library
from spectrum.transport_spectral import (
    DETECTOR_IMPACT_PHASE_COUNT,
    GeometryConditionedSpectralModel,
    PhysicalComponentDiscrepancy,
)
from tests.green_test_support import synthetic_detector_green_operator


ROOT = Path(__file__).resolve().parents[1]


def test_detector_implementation_provenance_excludes_application_contracts() -> None:
    """Scene seeds, PF code, and generated artifacts must not stale Green data."""
    paths = set(DETECTOR_GREEN_IMPLEMENTATION_PATHS)

    assert len(paths) == len(DETECTOR_GREEN_IMPLEMENTATION_PATHS)
    assert Path("src/spectrum/transport_spectral.py") not in paths
    assert Path("configs/validation/full_spectrum_acceptance.json") not in paths
    assert not any("/assets/" in path.as_posix() for path in paths)
    digest = detector_green_implementation_bundle_sha256(ROOT)
    assert len(digest) == 64
    assert all(character in "0123456789abcdef" for character in digest)


def _construction() -> dict[str, object]:
    """Return complete synthetic native provenance for artifact tests."""
    return {
        "method": "native_geant4_monoenergetic_full_detector",
        "raw_corpus_sha256": "1" * 64,
        "native_executable_sha256": "2" * 64,
        "native_execution_environment_sha256": "3" * 64,
        "detector_implementation_bundle_sha256": "4" * 64,
        "detector_model_sha256": "5" * 64,
        "geant4_physics_contract_sha256": "6" * 64,
        "energy_resolution_contract_sha256": (
            DETECTOR_ENERGY_RESOLUTION_CONTRACT_SHA256
        ),
        "construction_seed": 991_337,
        "histories_per_energy": 100_000,
        "energy_node_design": (
            "catalog_independent_deterministic_continuous_domain_v1"
        ),
        "phase_strata": 2,
        "detector_target_radius_m": 0.0395,
        "completed": True,
    }


def _operator() -> DetectorGreenOperator:
    """Return a small physically shaped operator for unit tests."""
    nodes = np.asarray((0.0, 400.0, 900.0, 1700.0), dtype=np.float64)
    edges = np.asarray((0.0, 0.55, 1.0), dtype=np.float64)
    raw = np.zeros((4, 2, 851), dtype=np.float64)
    for node_index, energy in enumerate(nodes[1:], start=1):
        raw_bin = int(energy // 2.0)
        raw[node_index, 0, raw_bin] = 7_000.0
        raw[node_index, 0, max(raw_bin // 3, 1)] = 1_000.0
        raw[node_index, 1, raw_bin] = 4_000.0
        raw[node_index, 1, max(raw_bin // 3, 1)] = 3_000.0
    histories = np.full((4, 2), 10_000.0, dtype=np.float64)
    return build_detector_green_operator(
        energy_nodes_keV=nodes,
        impact_parameter_edges_fraction=edges,
        raw_deposit_histograms_ncb=raw,
        sampled_histories_nc=histories,
        construction=_construction(),
    )


def _integer_histogram(
    probability_b: np.ndarray,
    total: int,
) -> np.ndarray:
    """Allocate one exact integer total by deterministic largest remainder."""
    expected = np.asarray(probability_b, dtype=np.float64) * int(total)
    result = np.floor(expected).astype(np.int64)
    missing = int(total) - int(np.sum(result))
    if missing:
        order = np.argsort(-(expected - result), kind="stable")
        result[order[:missing]] += 1
    return result


def _validation_corpus(
    operator: DetectorGreenOperator,
    *,
    design_seed: int = 818_181,
    transport_seed: int = 919_191,
) -> dict[str, object]:
    """Return synthetic independent evidence sampled from one operator."""
    energies = detector_green_holdout_energies_keV(
        design_seed,
        operator=operator,
    )
    phase_cbe, _ = operator.phase_response_for_axis(energies)
    detection_ce = operator.phase_detection_probability_for_axis(energies)
    histories_per_energy = 100_000
    histories_per_cell = histories_per_energy // phase_cbe.shape[0]
    cells = []
    for energy_index in range(energies.size):
        for impact_index in range(phase_cbe.shape[0]):
            pulses = int(
                round(
                    float(detection_ce[impact_index, energy_index]) * histories_per_cell
                )
            )
            histogram = _integer_histogram(
                phase_cbe[impact_index, :, energy_index],
                pulses,
            )
            cells.append(
                {
                    "energy_index": energy_index,
                    "impact_bin_index": impact_index,
                    "sampled_histories": histories_per_cell,
                    "registered_pulses": pulses,
                    "sparse_raw_deposit_histogram": [
                        [int(index), int(histogram[index])]
                        for index in np.flatnonzero(histogram)
                    ],
                }
            )
    construction = operator.construction
    assert construction is not None
    return {
        "schema_version": DETECTOR_GREEN_VALIDATION_SCHEMA_VERSION,
        "validation": DETECTOR_GREEN_VALIDATION_ID,
        "validation_contract_sha256": (DETECTOR_GREEN_VALIDATION_CONTRACT_SHA256),
        "operator_contract_sha256": operator.contract_hash_sha256,
        "operator_binary_sha256": operator.binary_sha256,
        "operator_construction_seed": construction["construction_seed"],
        "design_seed": design_seed,
        "transport_seed": transport_seed,
        "histories_per_energy": histories_per_energy,
        "impact_parameter_edges_fraction": (
            operator.impact_parameter_edges_fraction.tolist()
        ),
        "holdout_energies_keV": energies.tolist(),
        "output_energy_axis": {
            "minimum_keV": 0.0,
            "bin_width_keV": 2.0,
            "bin_count": 851,
        },
        "cells": cells,
        "runtime_config_sha256": "7" * 64,
        "native_executable_sha256": construction["native_executable_sha256"],
        "native_execution_environment_sha256": construction[
            "native_execution_environment_sha256"
        ],
        "detector_implementation_bundle_sha256": construction[
            "detector_implementation_bundle_sha256"
        ],
        "detector_model_sha256": construction["detector_model_sha256"],
        "geant4_physics_contract_sha256": construction[
            "geant4_physics_contract_sha256"
        ],
        "energy_resolution_contract_sha256": construction[
            "energy_resolution_contract_sha256"
        ],
        "reference_scene_sha256": "8" * 64,
        "reference_source_contract_sha256": "9" * 64,
    }


def test_energy_design_is_catalog_independent_and_covers_domain() -> None:
    """Construction nodes cover the domain without reusing catalog energies."""
    nodes = catalog_independent_energy_nodes_keV()
    catalog_energies = {
        float(line.energy_keV)
        for nuclide in default_library().values()
        for line in nuclide.lines
    }

    assert nodes[0] == 0.0
    assert nodes[-1] == 1700.0
    assert not catalog_energies.intersection(set(nodes.tolist()))


def test_catalog_lines_are_inputs_but_cannot_extend_operator_domain() -> None:
    """Known line metadata is accepted while extrapolation and cascades fail."""
    operator = _operator()
    operator.validate_catalog_profile(tuple(default_library()))
    outside = Nuclide(
        name="Outside",
        lines=(NuclideLine(energy_keV=1800.0, intensity=1.0),),
        representative_energy_keV=1800.0,
    )

    with pytest.raises(ValueError, match="outside the detector Green domain"):
        operator.validate_catalog_profile(
            ("Outside",),
            library={"Outside": outside},
        )
    with pytest.raises(ValueError, match="prompt-cascade"):
        operator.validate_catalog_profile(
            ("Cs-137",),
            primary_emission_model="geant4_radioactive_decay",
        )


def test_interpolation_preserves_counts_and_scales_peak_energy() -> None:
    """Conditional and absolute interpolation preserve their probability laws."""
    operator = _operator()
    axis = np.arange(851, dtype=np.float64) * 2.0
    phase, concentration = operator.phase_response_for_axis(axis)
    response, marginal_concentration = operator.marginal_response_for_axis(axis)
    absolute_phase, absolute_concentration = operator.phase_absolute_response_for_axis(
        axis
    )
    absolute, absolute_marginal_concentration = (
        operator.marginal_absolute_response_for_axis(axis)
    )
    detection = operator.phase_detection_probability_for_axis(axis)

    assert phase.shape == (2, 851, 851)
    assert np.allclose(np.sum(phase, axis=1), 1.0)
    assert np.allclose(np.sum(response, axis=0), 1.0)
    assert np.all(concentration >= 2.0)
    assert np.all(marginal_concentration >= 2.0)
    assert np.allclose(np.sum(absolute_phase, axis=1), detection)
    assert np.all(np.sum(absolute, axis=0) <= 1.0)
    assert np.any(np.sum(absolute, axis=0) < 1.0)
    assert np.all(absolute_concentration >= 2.0)
    assert np.all(absolute_marginal_concentration >= 2.0)
    assert abs(int(np.argmax(response[:, 250])) - 250) <= 2
    with pytest.raises(ValueError, match="in-domain"):
        operator.phase_response_for_axis((-1.0, 2.0))


def test_reference_efficiency_is_catalog_weighted_and_isotope_generic() -> None:
    """Source-rate normalization must derive only from lines and Green physics."""
    operator = _operator()
    radius = 0.0395
    edges = operator.impact_parameter_edges_fraction
    weights = (
        np.sqrt(1.0 - np.square(radius * edges[:-1]))
        - np.sqrt(1.0 - np.square(radius * edges[1:]))
    ) / (1.0 - np.sqrt(1.0 - radius * radius))

    for isotope in ("Cs-137", "Co-60", "Eu-154"):
        nuclide = default_library()[isotope]
        energies = np.asarray(
            [line.energy_keV for line in nuclide.lines],
            dtype=np.float64,
        )
        intensities = np.asarray(
            [line.intensity for line in nuclide.lines],
            dtype=np.float64,
        )
        phase = operator.phase_detection_probability_for_axis(energies)
        per_line = np.sum(weights[:, np.newaxis] * phase, axis=0)
        expected = float(np.sum(intensities * per_line) / np.sum(intensities))

        assert operator.catalog_weighted_reference_efficiency(
            nuclide,
            detector_target_radius_m=radius,
        ) == pytest.approx(expected, rel=1.0e-14, abs=1.0e-15)


def test_interpolation_aligns_the_exact_target_raw_bin_anchor() -> None:
    """A narrow peak cannot drift because source nodes occupy other raw bins."""
    nodes = np.asarray(
        (0.0, 51.62908865961538, 66.62627689791759, 1700.0),
        dtype=np.float64,
    )
    edges = np.asarray((0.0, 0.55, 1.0), dtype=np.float64)
    raw = np.zeros((4, 2, 851), dtype=np.float64)
    for node_index, energy in enumerate(nodes[1:], start=1):
        raw[node_index, :, int(np.floor(energy / 2.0))] = 8_000.0
    operator = build_detector_green_operator(
        energy_nodes_keV=nodes,
        impact_parameter_edges_fraction=edges,
        raw_deposit_histograms_ncb=raw,
        sampled_histories_nc=np.full((4, 2), 10_000.0),
        construction=_construction(),
    )
    target_energy_keV = 54.31604357552256

    phase, _ = operator.phase_response_for_axis((target_energy_keV,))

    target_raw_bin = int(np.floor(target_energy_keV / 2.0))
    assert tuple(np.argmax(phase[:, :, 0], axis=1)) == (
        target_raw_bin,
        target_raw_bin,
    )


def test_batched_and_single_target_interpolation_are_equivalent() -> None:
    """The standard batched path equals its one-target deterministic oracle."""
    operator = _operator()
    axis = np.arange(851, dtype=np.float64) * 2.0

    batched = operator.phase_response_for_axis(axis, batch_size=73)
    serial_oracle = operator.phase_response_for_axis(axis, batch_size=1)
    absolute_batched = operator.phase_absolute_response_for_axis(
        axis,
        batch_size=73,
    )
    absolute_serial = operator.phase_absolute_response_for_axis(
        axis,
        batch_size=1,
    )

    assert np.array_equal(batched[0], serial_oracle[0])
    assert np.array_equal(batched[1], serial_oracle[1])
    assert np.array_equal(absolute_batched[0], absolute_serial[0])
    assert np.array_equal(absolute_batched[1], absolute_serial[1])


def test_aligned_energy_phase_interpolation_matches_full_phase_tensor() -> None:
    """Joint energy/impact interpolation must equal exact phase selection."""
    operator = _operator()
    energies = np.asarray((25.0, 400.0, 713.5, 1699.0), dtype=np.float64)
    phases = np.asarray((0, 1, 0, 1), dtype=np.int64)

    paired, paired_concentration = (
        operator.absolute_response_for_energy_phase_pairs(
            energies,
            phases,
            batch_size=3,
        )
    )
    full, full_concentration = operator.phase_absolute_response_for_axis(
        energies,
        batch_size=3,
    )
    expected = np.stack(
        [full[phase, :, index] for index, phase in enumerate(phases)],
        axis=0,
    )
    expected_concentration = np.asarray(
        [
            full_concentration[phase, index]
            for index, phase in enumerate(phases)
        ],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(paired, expected)
    np.testing.assert_array_equal(paired_concentration, expected_concentration)
    with pytest.raises(ValueError, match="Energy/impact response pairs"):
        operator.absolute_response_for_energy_phase_pairs(
            energies,
            np.asarray((0, 1, 0, 2), dtype=np.int64),
        )


def test_detector_cone_scatter_grid_is_joint_isotope_free_and_bounded() -> None:
    """Scatter construction must retain joint energy/impact Green physics."""
    operator = _operator()
    energies = np.asarray((662.0, 1173.0, 1332.0), dtype=np.float64)
    reference = np.asarray((0.6, 0.55, 0.5), dtype=np.float64)

    grid = build_detector_cone_scatter_grid(
        operator=operator,
        incident_energies_keV=energies,
        source_reference_efficiencies=reference,
        fixed_scatter_distances_m=(0.08, 0.12),
    )
    payload = grid.contract_payload()

    assert grid.marked_response_dlb.shape == (
        grid.distance_nodes_m.size,
        energies.size,
        operator.output_bin_count,
    )
    assert grid.effective_histories_dl.shape == (
        grid.distance_nodes_m.size,
        energies.size,
    )
    assert grid.distance_nodes_m[-1] == DETECTOR_CONE_SCATTER_MAXIMUM_DISTANCE_M
    assert np.all(grid.marked_response_dlb >= 0.0)
    assert np.all(grid.effective_histories_dl >= 1.0)
    assert payload["response"] == DETECTOR_CONE_SCATTER_RESPONSE_ID
    serialized = json.dumps(payload, sort_keys=True)
    assert all(name not in serialized for name in default_library())


def test_detector_cone_scatter_cache_requires_exact_physics_identity() -> None:
    """Immutable grids may be reused only for an exact physical cache key."""
    operator = _operator()
    energies = np.asarray((662.0, 1173.0, 1332.0), dtype=np.float64)
    reference = np.asarray((0.6, 0.55, 0.5), dtype=np.float64)
    keywords = {
        "operator": operator,
        "incident_energies_keV": energies,
        "source_reference_efficiencies": reference,
        "fixed_scatter_distances_m": (0.08, 0.12),
    }

    first = build_detector_cone_scatter_grid(**keywords)
    repeated = build_detector_cone_scatter_grid(**keywords)
    changed = build_detector_cone_scatter_grid(
        **{
            **keywords,
            "source_reference_efficiencies": np.asarray(
                (np.nextafter(0.6, 1.0), 0.55, 0.5),
                dtype=np.float64,
            ),
        }
    )

    assert repeated is first
    assert changed is not first
    assert not np.array_equal(
        changed.marked_response_dlb,
        first.marked_response_dlb,
    )
    assert first.distance_nodes_m.flags.writeable is False
    assert first.marked_response_dlb.flags.writeable is False
    assert first.effective_histories_dl.flags.writeable is False


def test_artifact_round_trip_is_strict_and_contains_no_isotope_names(
    tmp_path: Path,
) -> None:
    """Binary readback is exact and the response artifact is isotope-free."""
    manifest_path = _operator().write_artifact(tmp_path / "operator")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    serialized = json.dumps(payload, sort_keys=True)

    assert all(name not in serialized for name in default_library())
    reloaded = DetectorGreenOperator.from_artifact(manifest_path)
    assert reloaded.runtime_ready
    assert reloaded.contract_hash_sha256 == payload["contract_hash_sha256"]

    binary_path = manifest_path.parent / DETECTOR_GREEN_BINARY_BASENAME
    binary_path.write_bytes(binary_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="binary hash is stale"):
        DetectorGreenOperator.from_artifact(manifest_path)


def test_resolution_and_impact_construction_are_normalized() -> None:
    """Numerical construction primitives preserve their physical measures."""
    resolution = gaussian_resolution_operator(
        output_bin_count=851,
        output_bin_width_keV=2.0,
    )
    edges = impact_parameter_edges_for_equal_solid_angle_strata(
        source_distance_m=0.05,
        detector_target_radius_m=0.0395,
        stratum_count=8,
    )

    assert np.allclose(np.sum(resolution, axis=0), 1.0)
    assert np.all(np.diff(edges) > 0.0)
    assert edges[0] == 0.0
    assert edges[-1] == 1.0


def test_zero_pulse_cell_remains_an_explicit_no_pulse_outcome() -> None:
    """A physical zero-pulse cell adds no hidden event or response mass."""
    nodes = np.asarray((0.0, 400.0, 900.0, 1700.0), dtype=np.float64)
    edges = np.asarray((0.0, 0.55, 1.0), dtype=np.float64)
    raw = np.zeros((4, 2, 851), dtype=np.float64)
    for node_index, energy in enumerate(nodes[1:], start=1):
        raw_bin = int(energy // 2.0)
        raw[node_index, :, raw_bin] = 5_000.0
    raw[1, 0] = 0.0
    histories = np.full((4, 2), 10_000.0, dtype=np.float64)

    operator = build_detector_green_operator(
        energy_nodes_keV=nodes,
        impact_parameter_edges_fraction=edges,
        raw_deposit_histograms_ncb=raw,
        sampled_histories_nc=histories,
        construction=_construction(),
    )
    conditional, _ = operator.phase_response_for_axis((400.0,))
    absolute, _ = operator.phase_absolute_response_for_axis((400.0,))

    assert operator.pulse_detection_probability_nc[1, 0] == 0.0
    assert operator.effective_histories_nc[1, 0] == 10_000.0
    assert np.sum(conditional[0, :, 0]) == pytest.approx(1.0)
    assert np.sum(absolute[0, :, 0]) == 0.0
    assert np.sum(absolute[1, :, 0]) == pytest.approx(0.5)


def test_catalog_lines_drive_one_generic_phase_conditioned_model(
    tmp_path: Path,
) -> None:
    """Known Cs/Co lines use one operator and exact batched phase weighting."""
    manifest_path = synthetic_detector_green_operator().write_artifact(
        tmp_path / "green"
    )
    operator = DetectorGreenOperator.from_artifact(manifest_path)
    model = GeometryConditionedSpectralModel.nonproduction_native(
        ("Cs-137", "Co-60"),
        dead_time_tau_s=0.0,
        background_rate_cps=0.0,
        detector_green_operator=operator,
    )
    line_rows = model.line_identity
    line_count = len(line_rows)
    total = np.zeros((3, 2, line_count), dtype=np.float64)
    for line_index, row in enumerate(line_rows):
        source_index = 0 if row["isotope"] == "Cs-137" else 1
        total[:, source_index, line_index] = float(row["branching_weight"])
    features = np.zeros(
        total.shape + (5 + DETECTOR_IMPACT_PHASE_COUNT,),
        dtype=np.float64,
    )
    features[..., 4] = 1.0
    features[0, ..., 5] = 1.0
    features[1, ..., 5:] = 1.0 / DETECTOR_IMPACT_PHASE_COUNT
    features[2, ..., -1] = 1.0
    live_times = np.asarray((20.0, 20.0, 20.0), dtype=np.float64)

    predicted = model.predict_mean_numpy(
        total,
        total,
        features,
        live_times,
    )
    serial_weights = np.stack(
        [
            model._direct_phase_weights_numpy(
                features[index : index + 1],
                active_xvsl=np.ones_like(total[index : index + 1], dtype=bool),
            )[0]
            for index in range(3)
        ]
    )
    batched_weights = model._direct_phase_weights_numpy(
        features,
        active_xvsl=np.ones_like(total, dtype=bool),
    )

    assert predicted.shape == (3, 851)
    assert np.isclose(np.sum(predicted[1]), 40.0, rtol=0.0, atol=1.0e-10)
    assert np.allclose(np.sum(predicted, axis=1), 40.0, rtol=0.0, atol=5.0e-7)
    assert not np.allclose(predicted[0], predicted[1])
    assert np.array_equal(batched_weights, serial_weights)
    assert model.manifest_payload()["detector_green_operator_id"] == (
        "isotope_independent_full_detector_green_operator_v3"
    )


def test_finite_green_detection_covariance_enters_count_uncertainty(
    tmp_path: Path,
) -> None:
    """No-pulse corpus uncertainty contributes to NumPy and Torch counts."""
    manifest_path = synthetic_detector_green_operator().write_artifact(
        tmp_path / "green"
    )
    model = GeometryConditionedSpectralModel.nonproduction_native(
        ("Cs-137", "Co-60"),
        dead_time_tau_s=0.0,
        background_rate_cps=0.0,
        physical_component_discrepancy=(
            PhysicalComponentDiscrepancy.physics_only_budget()
        ),
        additive_scatter_response=PhysicsOnlyNoncollidedTransportResponse(
            detector_radius_m=0.038,
            fe_scatter_distance_m=0.08,
            pb_scatter_distance_m=0.12,
        ),
        detector_green_operator=DetectorGreenOperator.from_artifact(manifest_path),
    )
    line_rows = model.line_identity
    total = np.zeros((1, 2, len(line_rows)), dtype=np.float64)
    for line_index, row in enumerate(line_rows):
        source_index = 0 if row["isotope"] == "Cs-137" else 1
        total[:, source_index, line_index] = float(row["branching_weight"])
    features = np.zeros(
        total.shape + (5 + DETECTOR_IMPACT_PHASE_COUNT,),
        dtype=np.float64,
    )
    features[..., 4] = 1.0
    features[..., 5:] = 1.0 / DETECTOR_IMPACT_PHASE_COUNT

    numpy_concentration = model._component_count_concentration_numpy(
        total,
        total,
        features,
    )

    assert numpy_concentration.shape == (1,)
    assert np.all(np.isfinite(numpy_concentration))
    assert np.all(numpy_concentration > 0.0)
    assert np.all(
        numpy_concentration
        < model.physical_component_discrepancy.count_uncollided_concentration
    )
    torch = pytest.importorskip("torch")
    torch_concentration = model._component_count_concentration_torch(
        torch.as_tensor(total, dtype=torch.float64),
        torch.as_tensor(total, dtype=torch.float64),
        torch.as_tensor(features, dtype=torch.float64),
    )
    assert np.allclose(
        numpy_concentration,
        torch_concentration.detach().cpu().numpy(),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_independent_holdout_recomputes_metrics_and_detects_tampering(
    tmp_path: Path,
) -> None:
    """Validation reloads raw evidence instead of trusting reported metrics."""
    operator_path = _operator().write_artifact(tmp_path / "operator")
    operator = DetectorGreenOperator.from_artifact(operator_path)
    corpus = _validation_corpus(operator)
    manifest = build_detector_green_validation_manifest(
        corpus,
        operator=operator,
    )
    validation = tmp_path / "validation"
    validation.mkdir()
    raw_path = validation / DETECTOR_GREEN_VALIDATION_RAW_BASENAME
    raw_path.write_bytes(canonical_json_bytes(corpus))
    manifest_path = validation / DETECTOR_GREEN_VALIDATION_MANIFEST_BASENAME
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    loaded = load_detector_green_validation_manifest(
        manifest_path,
        operator=operator,
        require_passed=False,
    )
    assert loaded == manifest

    tampered = json.loads(raw_path.read_text(encoding="utf-8"))
    tampered["cells"][0]["registered_pulses"] += 1
    raw_path.write_bytes(canonical_json_bytes(tampered))
    assert (
        hashlib.sha256(raw_path.read_bytes()).hexdigest()
        != (manifest["raw_corpus_sha256"])
    )
    with pytest.raises(ValueError, match="raw corpus hash is stale"):
        load_detector_green_validation_manifest(
            manifest_path,
            operator=operator,
            require_passed=False,
        )
