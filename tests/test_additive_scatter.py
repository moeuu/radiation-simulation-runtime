"""Tests for the physical additive noncollided transport response."""

from __future__ import annotations


import numpy as np
import pytest

from spectrum.additive_scatter import (
    ADDITIVE_SCATTER_FEATURE_ORDER,
    PHYSICAL_SCATTER_BASIS_SEMANTICS,
    PhysicsOnlyNoncollidedTransportResponse,
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


def _response() -> PhysicsOnlyNoncollidedTransportResponse:
    """Return the current response with deterministic detector geometry."""
    return PhysicsOnlyNoncollidedTransportResponse(
        detector_radius_m=0.025,
        fe_scatter_distance_m=0.14,
        pb_scatter_distance_m=0.10,
    )


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
        semantics=PHYSICAL_SCATTER_BASIS_SEMANTICS,
    )
    torch_basis = physical_scatter_basis_torch(
        **{
            key: torch.as_tensor(value, dtype=torch.float64)
            for key, value in inputs.items()
        },
        **geometry,
        semantics=PHYSICAL_SCATTER_BASIS_SEMANTICS,
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
        "semantics": PHYSICAL_SCATTER_BASIS_SEMANTICS,
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
        semantics=PHYSICAL_SCATTER_BASIS_SEMANTICS,
    )
    torch_basis = physical_scatter_basis_torch(
        **{
            key: torch.as_tensor(value, dtype=torch.float64)
            for key, value in inputs.items()
        },
        **geometry,
        semantics=PHYSICAL_SCATTER_BASIS_SEMANTICS,
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
    assert (
        PhysicsOnlyNoncollidedTransportResponse.from_payload(payload).to_payload()
        == payload
    )


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
    features = np.concatenate(
        (features[..., :3], obstacle_compton[..., None], features[..., 3:]),
        axis=-1,
    )
    stored = physical_scatter_basis_numpy(
        tau_fe=features[..., 0],
        tau_pb=features[..., 1],
        tau_obstacle=features[..., 2],
        tau_obstacle_compton=obstacle_compton,
        distance_m=features[..., 4],
        energy_keV=energy,
        mu_fe_cm_inv=mu_fe,
        mu_pb_cm_inv=mu_pb,
        detector_radius_m=0.038,
        fe_scatter_distance_m=0.057,
        pb_scatter_distance_m=0.082,
    )
    expected = physical_scatter_basis_numpy(
        tau_fe=features[..., 0],
        tau_pb=features[..., 1],
        tau_obstacle=features[..., 2],
        tau_obstacle_compton=obstacle_compton,
        distance_m=features[..., 4],
        energy_keV=energy,
        mu_fe_cm_inv=mu_fe,
        mu_pb_cm_inv=mu_pb,
        semantics=PHYSICAL_SCATTER_BASIS_SEMANTICS,
        detector_radius_m=0.038,
        fe_scatter_distance_m=0.057,
        pb_scatter_distance_m=0.082,
    )
    reconstructed = scatter_basis_from_stored_geometry_numpy(
        stored_basis=stored,
        transport_features=features,
        transport_feature_order=(
            "tau_fe",
            "tau_pb",
            "tau_obstacle",
            "tau_obstacle_compton",
            "distance_m",
        ),
        line_identity=lines,
        target_semantics=PHYSICAL_SCATTER_BASIS_SEMANTICS,
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
        detector_radius_m=response.detector_radius_m,
        fe_scatter_distance_m=response.fe_scatter_distance_m,
        pb_scatter_distance_m=response.pb_scatter_distance_m,
    )
    assert float(response.scatter_fraction_numpy(basis)[0]) > 0.0


@pytest.mark.parametrize(
    "semantics",
    (
        "at_least_one_interaction_opportunity_v1",
        "exactly_one_compton_with_zero_other_los_interactions_v2",
        "detector_cone_path_quadrature_single_compton_v1",
    ),
)
def test_retired_scatter_bases_cannot_be_constructed(semantics: str) -> None:
    """Direct construction and both compute backends must reject retired bases."""
    torch = pytest.importorskip("torch")
    geometry = dict(
        detector_radius_m=0.025, fe_scatter_distance_m=0.14, pb_scatter_distance_m=0.10
    )
    with pytest.raises(ValueError, match="current XCOM-air"):
        PhysicsOnlyNoncollidedTransportResponse(
            **geometry, feature_basis_semantics=semantics
        )
    inputs = dict(
        tau_fe=np.array([0.2]),
        tau_pb=np.array([0.3]),
        tau_obstacle=np.array([0.4]),
        tau_obstacle_compton=np.array([0.1]),
        distance_m=np.array([4.0]),
        energy_keV=np.array([662.0]),
        mu_fe_cm_inv=np.array([0.58]),
        mu_pb_cm_inv=np.array([1.29]),
    )
    with pytest.raises(ValueError, match="current XCOM-air"):
        physical_scatter_basis_numpy(**inputs, **geometry, semantics=semantics)
    with pytest.raises(ValueError, match="current XCOM-air"):
        physical_scatter_basis_torch(
            **{key: torch.as_tensor(value) for key, value in inputs.items()},
            **geometry,
            semantics=semantics,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 1),
        ("schema_version", True),
        ("detector_radius_m", "0.025"),
        ("detector_radius_m", True),
    ],
)
def test_current_response_rejects_retired_or_coerced_payloads(
    field: str, value: object
) -> None:
    """A retained response must keep its current schema and physical types."""
    payload = _response().to_payload()
    payload[field] = value
    with pytest.raises((ValueError, TypeError)):
        PhysicsOnlyNoncollidedTransportResponse.from_payload(payload)
