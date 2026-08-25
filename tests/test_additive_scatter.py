"""Tests for the physical additive noncollided transport response."""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

from spectrum.additive_scatter import (
    ADDITIVE_SCATTER_FEATURE_ORDER,
    ADDITIVE_SCATTER_INCIDENT_LABEL_SEMANTICS,
    ADDITIVE_SCATTER_RIDGE_LAMBDA_GRID,
    DETECTOR_CONE_AIR_XCOM_SINGLE_SCATTER_BASIS_SEMANTICS,
    EXACT_SINGLE_SCATTER_BASIS_SEMANTICS,
    DETECTOR_CONE_SINGLE_SCATTER_BASIS_SEMANTICS,
    AdditiveNoncollidedTransportResponse,
    PhysicsOnlyNoncollidedTransportResponse,
    fit_additive_noncollided_transport_response,
    physical_scatter_basis_numpy,
    physical_scatter_basis_torch,
    klein_nishina_forward_cone_fraction_numpy,
    scatter_basis_from_stored_geometry_numpy,
)
from spectrum.air_attenuation import (
    NIST_XCOM_DRY_AIR_TOTAL_CONTRACT_SHA256,
    dry_air_total_linear_attenuation_numpy,
    dry_air_total_linear_attenuation_torch,
)
from spectrum.transport_spectral import (
    DESIGNATED_HOLDOUT_SCENE_SEEDS,
    DESIGNATED_TRAINING_SCENE_SEEDS,
    FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256,
    VALIDATION_SCENARIO_IDS,
)


def _training_manifest() -> dict[str, object]:
    """Return deterministic designated-training provenance for unit tests."""
    return {
        "schema_version": 1,
        "acceptance_contract_sha256": (
            FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256
        ),
        "training_scene_seeds": list(DESIGNATED_TRAINING_SCENE_SEEDS),
        "scenario_ids": list(VALIDATION_SCENARIO_IDS),
        "artifact_sha256_by_scene": {
            "2026072701": "a" * 64,
            "2026072702": "b" * 64,
            "2026072703": "c" * 64,
        },
        "pair_ids_by_scene": {
            "2026072701": list(range(64)),
            "2026072702": list(range(64)),
            "2026072703": list(range(64)),
        },
        "label_space": ADDITIVE_SCATTER_INCIDENT_LABEL_SEMANTICS,
        "selection_objective": (
            "leave_one_training_scene_out_weighted_log1p_mse"
        ),
    }


def _response() -> AdditiveNoncollidedTransportResponse:
    """Return a valid nonzero additive response for deterministic tests."""
    manifest = _training_manifest()
    manifest.update(
        {
            "fit_sample_count": 210,
            "loso_scene_ids": [
                str(seed) for seed in DESIGNATED_TRAINING_SCENE_SEEDS
            ],
            "candidate_validation_scores": {
                format(value, ".12g"): (
                    0.5 if value == 0.1 else 1.0 + float(value)
                )
                for value in ADDITIVE_SCATTER_RIDGE_LAMBDA_GRID
            },
            "selected_validation_score": 0.5,
            "selected_ridge_lambda": 0.1,
            "selection_completed": True,
        }
    )
    return AdditiveNoncollidedTransportResponse(
        coefficients=(0.8, 0.5, 0.4, 0.2, 0.1, 0.05, 0.025),
        ridge_lambda=0.1,
        training_manifest=manifest,
    )


def test_physical_scatter_basis_matches_torch() -> None:
    """The batched NumPy and Torch physical bases must be equivalent."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(418)
    shape = (3, 4, 5)
    tau_fe = rng.uniform(0.0, 4.0, shape)
    tau_pb = rng.uniform(0.0, 5.0, shape)
    tau_obstacle = rng.uniform(0.0, 3.0, shape)
    tau_obstacle_compton = tau_obstacle * rng.uniform(0.1, 0.9, shape)
    distance = rng.uniform(0.2, 30.0, shape)
    energy = rng.uniform(250.0, 1600.0, shape)
    mu_fe = rng.uniform(0.35, 1.5, shape)
    mu_pb = rng.uniform(0.6, 3.0, shape)
    numpy_basis = physical_scatter_basis_numpy(
        tau_fe=tau_fe,
        tau_pb=tau_pb,
        tau_obstacle=tau_obstacle,
        tau_obstacle_compton=tau_obstacle_compton,
        distance_m=distance,
        energy_keV=energy,
        mu_fe_cm_inv=mu_fe,
        mu_pb_cm_inv=mu_pb,
    )
    torch_basis = physical_scatter_basis_torch(
        tau_fe=torch.as_tensor(tau_fe, dtype=torch.float64),
        tau_pb=torch.as_tensor(tau_pb, dtype=torch.float64),
        tau_obstacle=torch.as_tensor(tau_obstacle, dtype=torch.float64),
        tau_obstacle_compton=torch.as_tensor(
            tau_obstacle_compton,
            dtype=torch.float64,
        ),
        distance_m=torch.as_tensor(distance, dtype=torch.float64),
        energy_keV=torch.as_tensor(energy, dtype=torch.float64),
        mu_fe_cm_inv=torch.as_tensor(mu_fe, dtype=torch.float64),
        mu_pb_cm_inv=torch.as_tensor(mu_pb, dtype=torch.float64),
    )
    assert numpy_basis.shape == shape + (len(ADDITIVE_SCATTER_FEATURE_ORDER),)
    np.testing.assert_allclose(
        torch_basis.detach().cpu().numpy(),
        numpy_basis,
        rtol=2.0e-13,
        atol=2.0e-15,
    )


def test_exact_single_scatter_basis_matches_torch_and_occludes_air() -> None:
    """Exact-one features must match on CPU/GPU and vanish behind opaque LOS."""
    torch = pytest.importorskip("torch")
    shape = (2, 3)
    tau_obstacle = np.full(shape, 25.0, dtype=np.float64)
    inputs = {
        "tau_fe": np.zeros(shape, dtype=np.float64),
        "tau_pb": np.zeros(shape, dtype=np.float64),
        "tau_obstacle": tau_obstacle,
        "tau_obstacle_compton": 0.45 * tau_obstacle,
        "distance_m": np.full(shape, 12.0, dtype=np.float64),
        "energy_keV": np.full(shape, 662.0, dtype=np.float64),
        "mu_fe_cm_inv": np.full(shape, 0.58, dtype=np.float64),
        "mu_pb_cm_inv": np.full(shape, 1.29, dtype=np.float64),
    }
    legacy = physical_scatter_basis_numpy(**inputs)
    exact = physical_scatter_basis_numpy(
        **inputs,
        semantics=EXACT_SINGLE_SCATTER_BASIS_SEMANTICS,
    )
    exact_torch = physical_scatter_basis_torch(
        **{
            key: torch.as_tensor(value, dtype=torch.float64)
            for key, value in inputs.items()
        },
        semantics=EXACT_SINGLE_SCATTER_BASIS_SEMANTICS,
    )
    np.testing.assert_allclose(
        exact_torch.detach().cpu().numpy(),
        exact,
        rtol=2.0e-13,
        atol=2.0e-15,
    )
    assert np.all(legacy[..., 3] > 0.0)
    assert np.all(exact[..., 3] < legacy[..., 3] * 1.0e-9)


def test_detector_cone_single_scatter_matches_torch() -> None:
    """The standard physics-only cone integral must match on CPU and GPU."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(9281)
    shape = (2, 3, 4)
    obstacle_tau = rng.uniform(0.0, 2.0, shape)
    inputs = {
        "tau_fe": rng.uniform(0.0, 1.0, shape),
        "tau_pb": rng.uniform(0.0, 1.5, shape),
        "tau_obstacle": obstacle_tau,
        "tau_obstacle_compton": obstacle_tau * rng.uniform(0.1, 0.9, shape),
        "distance_m": rng.uniform(0.5, 20.0, shape),
        "energy_keV": rng.uniform(120.0, 1600.0, shape),
        "mu_fe_cm_inv": rng.uniform(0.35, 1.5, shape),
        "mu_pb_cm_inv": rng.uniform(0.6, 3.0, shape),
    }
    geometry = {
        "detector_radius_m": 0.025,
        "fe_scatter_distance_m": 0.14,
        "pb_scatter_distance_m": 0.10,
    }
    numpy_basis = physical_scatter_basis_numpy(
        **inputs,
        **geometry,
        semantics=DETECTOR_CONE_SINGLE_SCATTER_BASIS_SEMANTICS,
    )
    torch_basis = physical_scatter_basis_torch(
        **{
            key: torch.as_tensor(value, dtype=torch.float64)
            for key, value in inputs.items()
        },
        **geometry,
        semantics=DETECTOR_CONE_SINGLE_SCATTER_BASIS_SEMANTICS,
    )
    np.testing.assert_allclose(
        torch_basis.detach().cpu().numpy(),
        numpy_basis,
        rtol=2.0e-13,
        atol=2.0e-15,
    )
    assert np.all(numpy_basis[..., 4:] == 0.0)


def test_detector_cone_compact_line_constants_match_expanded_tensors() -> None:
    """Compact immutable line tensors must preserve the expanded GPU result."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1942)
    shape = (2, 3, 4)
    line_shape = (1, 1, shape[-1])
    obstacle_tau = rng.uniform(0.0, 2.0, shape)
    common = {
        "tau_fe": torch.as_tensor(
            rng.uniform(0.0, 1.0, shape),
            dtype=torch.float64,
        ),
        "tau_pb": torch.as_tensor(
            rng.uniform(0.0, 1.5, shape),
            dtype=torch.float64,
        ),
        "tau_obstacle": torch.as_tensor(
            obstacle_tau,
            dtype=torch.float64,
        ),
        "tau_obstacle_compton": torch.as_tensor(
            obstacle_tau * rng.uniform(0.1, 0.9, shape),
            dtype=torch.float64,
        ),
        "distance_m": torch.as_tensor(
            rng.uniform(0.5, 20.0, shape),
            dtype=torch.float64,
        ),
        "semantics": DETECTOR_CONE_SINGLE_SCATTER_BASIS_SEMANTICS,
        "detector_radius_m": 0.038,
        "fe_scatter_distance_m": 0.14,
        "pb_scatter_distance_m": 0.10,
    }
    energy = rng.uniform(120.0, 1600.0, line_shape)
    mu_fe = rng.uniform(0.35, 1.5, line_shape)
    mu_pb = rng.uniform(0.6, 3.0, line_shape)
    compact = physical_scatter_basis_torch(
        **common,
        energy_keV=torch.as_tensor(energy, dtype=torch.float64),
        mu_fe_cm_inv=torch.as_tensor(mu_fe, dtype=torch.float64),
        mu_pb_cm_inv=torch.as_tensor(mu_pb, dtype=torch.float64),
    )
    expanded = physical_scatter_basis_torch(
        **common,
        energy_keV=torch.as_tensor(
            np.broadcast_to(energy, shape).copy(),
            dtype=torch.float64,
        ),
        mu_fe_cm_inv=torch.as_tensor(
            np.broadcast_to(mu_fe, shape).copy(),
            dtype=torch.float64,
        ),
        mu_pb_cm_inv=torch.as_tensor(
            np.broadcast_to(mu_pb, shape).copy(),
            dtype=torch.float64,
        ),
    )
    np.testing.assert_array_equal(
        compact.detach().cpu().numpy(),
        expanded.detach().cpu().numpy(),
    )


def test_xcom_air_attenuation_and_scatter_basis_match_torch() -> None:
    """Authenticated dry-air loss must be batched and CPU/GPU equivalent."""
    torch = pytest.importorskip("torch")
    energy = np.asarray([59.5, 122.0, 662.0, 1332.0], dtype=np.float64)
    numpy_mu = dry_air_total_linear_attenuation_numpy(energy)
    torch_mu = dry_air_total_linear_attenuation_torch(
        torch.as_tensor(energy, dtype=torch.float64)
    )
    np.testing.assert_allclose(
        torch_mu.detach().cpu().numpy(),
        numpy_mu,
        rtol=2.0e-13,
        atol=0.0,
    )
    inputs = {
        "tau_fe": np.zeros(energy.shape, dtype=np.float64),
        "tau_pb": np.zeros(energy.shape, dtype=np.float64),
        "tau_obstacle": np.zeros(energy.shape, dtype=np.float64),
        "tau_obstacle_compton": np.zeros(energy.shape, dtype=np.float64),
        "distance_m": np.full(energy.shape, 18.0, dtype=np.float64),
        "energy_keV": energy,
        "mu_fe_cm_inv": np.ones(energy.shape, dtype=np.float64),
        "mu_pb_cm_inv": np.ones(energy.shape, dtype=np.float64),
    }
    geometry = {
        "detector_radius_m": 0.025,
        "fe_scatter_distance_m": 0.14,
        "pb_scatter_distance_m": 0.10,
    }
    numpy_basis = physical_scatter_basis_numpy(
        **inputs,
        **geometry,
        semantics=DETECTOR_CONE_AIR_XCOM_SINGLE_SCATTER_BASIS_SEMANTICS,
    )
    torch_basis = physical_scatter_basis_torch(
        **{
            key: torch.as_tensor(value, dtype=torch.float64)
            for key, value in inputs.items()
        },
        **geometry,
        semantics=DETECTOR_CONE_AIR_XCOM_SINGLE_SCATTER_BASIS_SEMANTICS,
    )
    np.testing.assert_allclose(
        torch_basis.detach().cpu().numpy(),
        numpy_basis,
        rtol=2.0e-13,
        atol=2.0e-15,
    )
    assert np.all(numpy_basis[..., 3] > 0.0)


def test_klein_nishina_cone_fraction_is_bounded_and_geometric() -> None:
    """A larger detector cone must capture at least as much Compton mass."""
    energy = np.asarray([122.0, 662.0, 1332.0], dtype=np.float64)
    narrow = klein_nishina_forward_cone_fraction_numpy(
        energy,
        detector_radius_m=0.01,
        scatter_distance_m=1.0,
    )
    wide = klein_nishina_forward_cone_fraction_numpy(
        energy,
        detector_radius_m=0.05,
        scatter_distance_m=1.0,
    )
    assert np.all((0.0 <= narrow) & (narrow <= 1.0))
    assert np.all((0.0 <= wide) & (wide <= 1.0))
    assert np.all(wide > narrow)


def test_physics_only_response_round_trips_without_training_fields() -> None:
    """A physics-only response must be authenticated without scene artifacts."""
    response = PhysicsOnlyNoncollidedTransportResponse(
        detector_radius_m=0.025,
        fe_scatter_distance_m=0.14,
        pb_scatter_distance_m=0.10,
    )
    payload = response.to_payload()
    assert payload["fit_family"] == "none_physics_only"
    assert payload["schema_version"] == 2
    assert payload["dry_air_total_attenuation_contract_sha256"] == (
        NIST_XCOM_DRY_AIR_TOTAL_CONTRACT_SHA256
    )
    assert "training_manifest" not in payload
    assert response.training_ready is True
    assert PhysicsOnlyNoncollidedTransportResponse.from_payload(
        payload
    ).to_payload() == payload


def test_stored_legacy_geometry_reconstructs_exact_versioned_basis() -> None:
    """Stored ray features must reconstruct the exact runtime basis losslessly."""
    features = np.asarray(
        [
            [[0.2, 0.4, 1.5, 4.0], [0.0, 0.8, 6.0, 9.0]],
            [[0.5, 0.0, 0.0, 2.0], [0.1, 0.3, 18.0, 15.0]],
        ],
        dtype=np.float64,
    )
    lines = (
        {
            "energy_keV": 662.0,
            "mu_fe_cm_inv": 0.58,
            "mu_pb_cm_inv": 1.29,
        },
        {
            "energy_keV": 1173.0,
            "mu_fe_cm_inv": 0.43,
            "mu_pb_cm_inv": 0.76,
        },
    )
    line_shape = (1, 2)
    obstacle_compton = features[..., 2] * np.asarray(
        [[0.4, 0.7]],
        dtype=np.float64,
    )
    inputs = {
        "tau_fe": features[..., 0],
        "tau_pb": features[..., 1],
        "tau_obstacle": features[..., 2],
        "tau_obstacle_compton": obstacle_compton,
        "distance_m": features[..., 3],
        "energy_keV": np.asarray([662.0, 1173.0]).reshape(line_shape),
        "mu_fe_cm_inv": np.asarray([0.58, 0.43]).reshape(line_shape),
        "mu_pb_cm_inv": np.asarray([1.29, 0.76]).reshape(line_shape),
    }
    stored = physical_scatter_basis_numpy(**inputs)
    expected = physical_scatter_basis_numpy(
        **inputs,
        semantics=EXACT_SINGLE_SCATTER_BASIS_SEMANTICS,
    )
    reconstructed = scatter_basis_from_stored_geometry_numpy(
        stored_basis=stored,
        transport_features=features,
        line_identity=lines,
        target_semantics=EXACT_SINGLE_SCATTER_BASIS_SEMANTICS,
    )
    np.testing.assert_allclose(reconstructed, expected, rtol=2.0e-13, atol=1e-15)


def test_exact_basis_payload_is_schema_three_and_round_trips() -> None:
    """New basis semantics must be explicit and byte-authenticated."""
    response = replace(
        _response(),
        feature_basis_semantics=EXACT_SINGLE_SCATTER_BASIS_SEMANTICS,
    )
    payload = response.to_payload()
    assert payload["schema_version"] == 3
    assert payload["feature_basis_semantics"] == (
        EXACT_SINGLE_SCATTER_BASIS_SEMANTICS
    )
    assert AdditiveNoncollidedTransportResponse.from_payload(
        payload
    ).to_payload() == payload


def test_detector_cone_basis_reconstruction_uses_response_geometry() -> None:
    """Stored rays must reconstruct detector-cone physics without fitting."""
    features = np.asarray(
        [[[0.4, 0.7, 0.2, 3.0], [0.3, 0.5, 0.1, 4.0]]],
        dtype=np.float64,
    )
    lines = (
        {
            "energy_keV": 662.0,
            "mu_fe_cm_inv": 0.58,
            "mu_pb_cm_inv": 1.29,
        },
        {
            "energy_keV": 1173.0,
            "mu_fe_cm_inv": 0.43,
            "mu_pb_cm_inv": 0.76,
        },
    )
    energy = np.asarray([662.0, 1173.0], dtype=np.float64).reshape(1, 2)
    mu_fe = np.asarray([0.58, 0.43], dtype=np.float64).reshape(1, 2)
    mu_pb = np.asarray([1.29, 0.76], dtype=np.float64).reshape(1, 2)
    obstacle_compton = features[..., 2] * np.asarray(
        [[0.4, 0.7]],
        dtype=np.float64,
    )
    stored = physical_scatter_basis_numpy(
        tau_fe=features[..., 0],
        tau_pb=features[..., 1],
        tau_obstacle=features[..., 2],
        tau_obstacle_compton=obstacle_compton,
        distance_m=features[..., 3],
        energy_keV=energy,
        mu_fe_cm_inv=mu_fe,
        mu_pb_cm_inv=mu_pb,
    )
    expected = physical_scatter_basis_numpy(
        tau_fe=features[..., 0],
        tau_pb=features[..., 1],
        tau_obstacle=features[..., 2],
        tau_obstacle_compton=obstacle_compton,
        distance_m=features[..., 3],
        energy_keV=energy,
        mu_fe_cm_inv=mu_fe,
        mu_pb_cm_inv=mu_pb,
        semantics=DETECTOR_CONE_SINGLE_SCATTER_BASIS_SEMANTICS,
        detector_radius_m=0.038,
        fe_scatter_distance_m=0.057,
        pb_scatter_distance_m=0.082,
    )
    reconstructed = scatter_basis_from_stored_geometry_numpy(
        stored_basis=stored,
        transport_features=features,
        line_identity=lines,
        target_semantics=DETECTOR_CONE_SINGLE_SCATTER_BASIS_SEMANTICS,
        detector_radius_m=0.038,
        fe_scatter_distance_m=0.057,
        pb_scatter_distance_m=0.082,
    )
    np.testing.assert_allclose(
        reconstructed,
        expected,
        rtol=2.0e-13,
        atol=1.0e-15,
    )


def test_additive_kernel_is_nonnegative_and_cpu_torch_equivalent() -> None:
    """The response must add nonnegative scatter without changing direct counts."""
    torch = pytest.importorskip("torch")
    response = _response()
    rng = np.random.default_rng(67)
    unattenuated = rng.uniform(0.1, 5.0, (2, 3, 4))
    uncollided = unattenuated * rng.uniform(0.0, 1.0, unattenuated.shape)
    basis = rng.uniform(
        0.0,
        0.5,
        unattenuated.shape + (len(ADDITIVE_SCATTER_FEATURE_ORDER),),
    )
    numpy_total = response.total_kernel_numpy(
        unattenuated,
        uncollided,
        basis,
    )
    torch_total = response.total_kernel_torch(
        torch.as_tensor(unattenuated, dtype=torch.float64),
        torch.as_tensor(uncollided, dtype=torch.float64),
        torch.as_tensor(basis, dtype=torch.float64),
    )
    assert np.all(numpy_total >= uncollided)
    assert np.any(numpy_total > uncollided)
    np.testing.assert_allclose(
        torch_total.detach().cpu().numpy(),
        numpy_total,
        rtol=2.0e-15,
        atol=2.0e-15,
    )


def test_fit_uses_only_declared_training_arrays_and_global_coefficients() -> None:
    """Training-scene cross validation must recover one global nonzero model."""
    rng = np.random.default_rng(82)
    sample_count = 210
    features = rng.uniform(
        0.0,
        0.5,
        (sample_count, len(ADDITIVE_SCATTER_FEATURE_ORDER)),
    )
    true_coefficients = np.asarray(
        (0.6, 0.4, 0.2, 0.1, 0.05, 0.03, 0.01),
        dtype=np.float64,
    )
    targets = features @ true_coefficients
    scene_ids = np.repeat(
        ("2026072701", "2026072702", "2026072703"),
        sample_count // 3,
    )
    response = fit_additive_noncollided_transport_response(
        features,
        targets,
        np.ones(sample_count, dtype=np.float64),
        scene_ids,
        training_manifest=_training_manifest(),
    )
    assert np.all(np.asarray(response.coefficients) >= 0.0)
    assert np.any(np.asarray(response.coefficients) > 0.0)
    assert "holdout" not in response.to_payload()["training"]
    np.testing.assert_allclose(
        features @ np.asarray(response.coefficients),
        targets,
        rtol=1.0e-4,
        atol=1.0e-5,
    )


def test_direct_transport_fit_is_signed_authenticated_and_batched() -> None:
    """Direct attenuation correction must preserve CPU/GPU contract parity."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(114)
    sample_count = 210
    features = rng.uniform(
        0.0,
        0.5,
        (sample_count, len(ADDITIVE_SCATTER_FEATURE_ORDER)),
    )
    scatter_coefficients = np.linspace(0.03, 0.15, features.shape[1])
    direct_coefficients = np.asarray(
        (-0.30, -0.55, -0.10, 0.04, -0.06, -0.03, 0.02),
        dtype=np.float64,
    )
    scene_ids = np.repeat(
        ("2026072701", "2026072702", "2026072703"),
        sample_count // 3,
    )
    response = fit_additive_noncollided_transport_response(
        features,
        features @ scatter_coefficients,
        np.ones(sample_count, dtype=np.float64),
        scene_ids,
        training_manifest=_training_manifest(),
        direct_log_ratio_n=features @ direct_coefficients,
    )
    uncollided = rng.uniform(0.1, 2.0, (5, 6))
    evaluation_basis = rng.uniform(
        0.0,
        0.5,
        uncollided.shape + (len(ADDITIVE_SCATTER_FEATURE_ORDER),),
    )
    numpy_direct = response.corrected_uncollided_kernel_numpy(
        uncollided,
        evaluation_basis,
    )
    torch_direct = response.corrected_uncollided_kernel_torch(
        torch.as_tensor(uncollided, dtype=torch.float64),
        torch.as_tensor(evaluation_basis, dtype=torch.float64),
    )

    assert response.to_payload()["schema_version"] == 2
    assert response.direct_training_manifest is not None
    assert np.any(numpy_direct < uncollided)
    np.testing.assert_allclose(
        torch_direct.detach().cpu().numpy(),
        numpy_direct,
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    assert AdditiveNoncollidedTransportResponse.from_payload(
        response.to_payload()
    ).to_payload() == response.to_payload()


def test_payload_authentication_rejects_legacy_pair_categorical_model() -> None:
    """Old pair-indexed multiplicative calibration must not enter production."""
    response = _response()
    payload = response.to_payload()
    assert AdditiveNoncollidedTransportResponse.from_payload(
        payload
    ).to_payload() == payload
    legacy = {
        "schema_version": 3,
        "model": "legacy_pair_categorical_regression",
        "scale_by_pair": {"0": 2.0},
    }
    with pytest.raises(ValueError, match="schema"):
        AdditiveNoncollidedTransportResponse.from_payload(legacy)


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    (
        (("schema_version",), True),
        (
            ("artifact_sha256_by_scene", "2026072701"),
            int("a" * 64, 16),
        ),
        (("pair_ids_by_scene", "2026072701", 1), True),
        (("fit_sample_count",), "210"),
        (("fit_sample_count",), True),
        (("selected_validation_score",), "0.5"),
        (("selected_validation_score",), True),
        (("selected_ridge_lambda",), "0.1"),
        (("selected_ridge_lambda",), True),
        (("candidate_validation_scores", "0.1"), "0.5"),
        (("candidate_validation_scores", "0.1"), True),
    ),
)
def test_training_manifest_rejects_json_scalar_coercion(
    field_path: tuple[object, ...],
    replacement: object,
) -> None:
    """Training evidence must preserve exact JSON scalar types."""
    response = _response()
    manifest = copy.deepcopy(dict(response.training_manifest))
    target: object = manifest
    for key in field_path[:-1]:
        target = target[key]  # type: ignore[index]
    target[field_path[-1]] = replacement  # type: ignore[index]

    with pytest.raises(ValueError, match="training provenance"):
        AdditiveNoncollidedTransportResponse(
            coefficients=response.coefficients,
            ridge_lambda=response.ridge_lambda,
            training_manifest=manifest,
        )


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    (
        (("schema_version",), True),
        (("coefficients", 0), "0.8"),
        (("coefficients", 0), True),
        (("selected_ridge_lambda",), "0.1"),
        (("selected_ridge_lambda",), True),
        (("ridge_lambda_grid", 5), True),
        (("contract_hash_sha256",), int("a" * 64, 16)),
    ),
)
def test_payload_rejects_json_scalar_coercion(
    field_path: tuple[object, ...],
    replacement: object,
) -> None:
    """External response payloads must reject strings and booleans as numbers."""
    payload = copy.deepcopy(_response().to_payload())
    target: object = payload
    for key in field_path[:-1]:
        target = target[key]  # type: ignore[index]
    target[field_path[-1]] = replacement  # type: ignore[index]

    with pytest.raises(ValueError, match="schema"):
        AdditiveNoncollidedTransportResponse.from_payload(payload)


@pytest.mark.parametrize(
    "tamper",
    ("holdout_seed", "missing_pair", "extra_key"),
)
def test_training_provenance_rejects_leakage_and_schema_tampering(
    tamper: str,
) -> None:
    """Only the designated training all-64 LOSO fit may enter production."""
    manifest = copy.deepcopy(dict(_response().training_manifest))
    if tamper == "holdout_seed":
        manifest["training_scene_seeds"][-1] = (
            DESIGNATED_HOLDOUT_SCENE_SEEDS[0]
        )
    elif tamper == "missing_pair":
        manifest["pair_ids_by_scene"]["2026072701"] = list(range(63))
    else:
        manifest["holdout_metrics"] = {"score": 0.0}

    with pytest.raises(ValueError, match="training provenance"):
        AdditiveNoncollidedTransportResponse(
            coefficients=(0.8, 0.5, 0.4, 0.2, 0.1, 0.05, 0.025),
            ridge_lambda=0.1,
            training_manifest=manifest,
        )


@pytest.mark.parametrize(
    "basis_case",
    ("air", "shield", "obstacle"),
)
def test_nonzero_physical_opportunities_cannot_silently_return_zero_scatter(
    basis_case: str,
) -> None:
    """Air, shield, and obstacle opportunities must produce positive scatter."""
    response = _response()
    shape = (1,)
    tau_fe = np.zeros(shape)
    tau_pb = np.zeros(shape)
    tau_obstacle = np.zeros(shape)
    tau_obstacle_compton = np.zeros(shape)
    distance = np.zeros(shape)
    if basis_case == "air":
        distance[...] = 10.0
    elif basis_case == "shield":
        tau_fe[...] = 1.0
    else:
        tau_obstacle[...] = 1.0
        tau_obstacle_compton[...] = 0.5
    basis = physical_scatter_basis_numpy(
        tau_fe=tau_fe,
        tau_pb=tau_pb,
        tau_obstacle=tau_obstacle,
        tau_obstacle_compton=tau_obstacle_compton,
        distance_m=distance,
        energy_keV=np.full(shape, 662.0),
        mu_fe_cm_inv=np.full(shape, 0.58),
        mu_pb_cm_inv=np.full(shape, 1.29),
    )
    assert float(response.scatter_fraction_numpy(basis)[0]) > 0.0
