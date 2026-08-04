"""Physical additive noncollided-response model for full-spectrum inference.

The model predicts detector-entry scatter before detector-response marking,
background injection, and electronics dead time.  It is deliberately additive:

``total = uncollided + unattenuated_geometric * scatter_fraction``.

Every learned coefficient is global and nonnegative.  The feature basis uses
only known interaction probabilities and contains no shield-pair, scene-seed,
source-index, or isotope categorical term.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import nnls

from spectrum.air_attenuation import (
    G4_AIR_REFERENCE_DENSITY_G_CM3,
    NIST_XCOM_DRY_AIR_TOTAL_CONTRACT_ID,
    NIST_XCOM_DRY_AIR_TOTAL_CONTRACT_SHA256,
    dry_air_total_linear_attenuation_numpy,
    dry_air_total_linear_attenuation_torch,
)
from spectrum.physics_contracts import (
    OBSTACLE_MATERIAL_CONTRACT_ID,
    OBSTACLE_MATERIAL_CONTRACT_SHA256,
    TRANSPORT_PHYSICS_TABLE_CONTRACT_ID,
    TRANSPORT_PHYSICS_TABLE_CONTRACT_SHA256,
)


ELECTRON_REST_ENERGY_KEV = 510.99895
CLASSICAL_ELECTRON_RADIUS_CM = 2.8179403262e-13
AVOGADRO_CONSTANT_MOL_INV = 6.02214076e23
AIR_DENSITY_G_CM3 = 1.225e-3
AIR_EFFECTIVE_Z_OVER_A = 0.49919
IRON_DENSITY_G_CM3 = 7.874
IRON_Z_OVER_A = 26.0 / 55.845
LEAD_DENSITY_G_CM3 = 11.34
LEAD_Z_OVER_A = 82.0 / 207.2
ELEMENT_Z_AND_ATOMIC_MASS = MappingProxyType(
    {
        "H": (1.0, 1.00794),
        "C": (6.0, 12.0107),
        "N": (7.0, 14.0067),
        "O": (8.0, 15.9994),
        "Al": (13.0, 26.9815385),
        "Si": (14.0, 28.0855),
        "Ar": (18.0, 39.948),
        "Ca": (20.0, 40.078),
        "Cr": (24.0, 51.9961),
        "Fe": (26.0, 55.845),
        "Ni": (28.0, 58.6934),
        "Pb": (82.0, 207.2),
    }
)

ADDITIVE_SCATTER_FEATURE_ORDER = (
    "fe_single_compton_probability",
    "pb_single_compton_probability",
    "obstacle_single_compton_probability",
    "air_single_compton_probability",
    "fe_pb_interaction_probability",
    "shield_obstacle_interaction_probability",
    "material_air_interaction_probability",
)
ADDITIVE_SCATTER_RIDGE_LAMBDA_GRID = (
    0.0,
    1.0e-4,
    1.0e-3,
    1.0e-2,
    1.0e-1,
    1.0,
    10.0,
)
ADDITIVE_SCATTER_INCIDENT_LABEL_SEMANTICS = (
    "pre_dead_time_raw_incident_gamma_source_line_entry_class"
)
ADDITIVE_SCATTER_TARGET_SEMANTICS = (
    "interacted_plus_secondary_pre_dead_time_incident_counts_divided_by_"
    "unattenuated_geometric_line_counts"
)
ADDITIVE_SCATTER_MODEL_ID = "additive_noncollided_transport_response_v1"
LEGACY_SCATTER_BASIS_SEMANTICS = "at_least_one_interaction_opportunity_v1"
EXACT_SINGLE_SCATTER_BASIS_SEMANTICS = (
    "exactly_one_compton_with_zero_other_los_interactions_v2"
)
DETECTOR_CONE_SINGLE_SCATTER_BASIS_SEMANTICS = (
    "detector_cone_path_quadrature_single_compton_v1"
)
DETECTOR_CONE_AIR_XCOM_SINGLE_SCATTER_BASIS_SEMANTICS = (
    "detector_cone_path_quadrature_single_compton_air_xcom_v2"
)
SCATTER_BASIS_SEMANTICS = (
    LEGACY_SCATTER_BASIS_SEMANTICS,
    EXACT_SINGLE_SCATTER_BASIS_SEMANTICS,
    DETECTOR_CONE_SINGLE_SCATTER_BASIS_SEMANTICS,
    DETECTOR_CONE_AIR_XCOM_SINGLE_SCATTER_BASIS_SEMANTICS,
)
DETECTOR_CONE_SCATTER_BASIS_SEMANTICS = frozenset(
    {
        DETECTOR_CONE_SINGLE_SCATTER_BASIS_SEMANTICS,
        DETECTOR_CONE_AIR_XCOM_SINGLE_SCATTER_BASIS_SEMANTICS,
    }
)
LEGACY_PHYSICS_ONLY_TRANSPORT_RESPONSE_ID = (
    "physics_only_detector_cone_transport_response_v1"
)
PHYSICS_ONLY_TRANSPORT_RESPONSE_ID = (
    "physics_only_detector_cone_transport_response_v2"
)
_CONE_QUADRATURE_NODES, _CONE_QUADRATURE_WEIGHTS = (
    np.polynomial.legendre.leggauss(16)
)
_TORCH_CONE_QUADRATURE_CACHE: dict[tuple[str, str], tuple[object, object]] = {}


def _torch_cone_quadrature_constants(
    *,
    device: object,
    dtype: object,
) -> tuple[object, object]:
    """Return cached Gauss-Legendre constants on one Torch device."""
    import torch

    key = (str(device), str(dtype))
    cached = _TORCH_CONE_QUADRATURE_CACHE.get(key)
    if cached is not None:
        return cached
    nodes = torch.as_tensor(
        _CONE_QUADRATURE_NODES,
        device=device,
        dtype=dtype,
    )
    weights = torch.as_tensor(
        _CONE_QUADRATURE_WEIGHTS,
        device=device,
        dtype=dtype,
    )
    result = (nodes, weights)
    _TORCH_CONE_QUADRATURE_CACHE[key] = result
    return result
DIRECT_TRANSPORT_TARGET_SEMANTICS = (
    "log_native_uncollided_primary_counts_divided_by_"
    "analytic_uncollided_detector_entry_counts"
)
DIRECT_TRANSPORT_MAXIMUM_ABS_LOG_CORRECTION = 2.0
_ADDITIVE_SCATTER_TRAINING_BASE_KEYS = frozenset(
    {
        "schema_version",
        "acceptance_contract_sha256",
        "training_scene_seeds",
        "scenario_ids",
        "pair_ids_by_scene",
        "artifact_sha256_by_scene",
        "label_space",
        "selection_objective",
    }
)
_ADDITIVE_SCATTER_TRAINING_FIT_KEYS = frozenset(
    {
        "fit_sample_count",
        "loso_scene_ids",
        "candidate_validation_scores",
        "selected_validation_score",
        "selected_ridge_lambda",
        "selection_completed",
    }
)
_ADDITIVE_SCATTER_TRAINING_CONTRACT_KEYS = frozenset(
    {
        "shield_pose_contract_sha256",
        "detector_response_contract_sha256",
        "obstacle_material_contract_sha256",
        "transport_physics_table_contract_sha256",
        "artifact_contract_sha256_by_scene",
    }
)


def _canonical_json_sha256(payload: object) -> str:
    """Return a canonical lowercase SHA-256 for one JSON-compatible payload."""
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _is_lower_sha256(value: object) -> bool:
    """Return whether a value is one canonical lowercase SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(
            character in "0123456789abcdef" for character in value
        )
    )


def _is_finite_json_number(value: object) -> bool:
    """Return whether one value is a finite JSON number, excluding booleans."""
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and np.isfinite(float(value))
    )


def _strict_json_number(value: object, *, field_name: str) -> float:
    """Return one finite JSON number without accepting scalar coercion."""
    if not _is_finite_json_number(value):
        raise TypeError(f"{field_name} must be a finite JSON number.")
    return float(value)


def _training_manifest_base_ready(
    manifest: Mapping[str, object],
) -> bool:
    """Validate immutable training-only scatter provenance."""
    from spectrum.transport_spectral import (
        FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256,
        VALIDATION_SCENARIO_IDS,
    )

    schema_version = manifest.get("schema_version")
    expected_base = _ADDITIVE_SCATTER_TRAINING_BASE_KEYS
    if schema_version == 2:
        expected_base = (
            expected_base | _ADDITIVE_SCATTER_TRAINING_CONTRACT_KEYS
        )
    if set(manifest).difference(
        _ADDITIVE_SCATTER_TRAINING_FIT_KEYS
    ) != expected_base:
        return False
    training_seeds = manifest.get("training_scene_seeds")
    scenarios = manifest.get("scenario_ids")
    if (
        type(manifest.get("schema_version")) is not int
        or schema_version not in (1, 2)
        or manifest.get("acceptance_contract_sha256")
        != FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256
        or not isinstance(training_seeds, list)
        or not training_seeds
        or any(type(seed) is not int or seed < 0 for seed in training_seeds)
        or len(set(training_seeds)) != len(training_seeds)
        or not isinstance(scenarios, list)
        or not scenarios
        or any(
            type(scenario) is not str
            or scenario not in VALIDATION_SCENARIO_IDS
            for scenario in scenarios
        )
        or len(set(scenarios)) != len(scenarios)
        or manifest.get("label_space")
        != ADDITIVE_SCATTER_INCIDENT_LABEL_SEMANTICS
        or manifest.get("selection_objective")
        != "leave_one_training_scene_out_weighted_log1p_mse"
    ):
        return False
    seed_keys = {str(seed) for seed in training_seeds}
    pair_ids = manifest.get("pair_ids_by_scene")
    artifact_hashes = manifest.get("artifact_sha256_by_scene")
    declared_pair_ids = (
        pair_ids.get(str(training_seeds[0]))
        if isinstance(pair_ids, Mapping)
        else None
    )
    base_ready = bool(
        isinstance(pair_ids, Mapping)
        and set(pair_ids) == seed_keys
        and isinstance(declared_pair_ids, list)
        and all(
            isinstance(pair_ids[str(seed)], list)
            and bool(pair_ids[str(seed)])
            and all(
                type(pair_id) is int
                and 0 <= pair_id < 64
                for pair_id in pair_ids[str(seed)]
            )
            and len(set(pair_ids[str(seed)])) == len(pair_ids[str(seed)])
            and pair_ids[str(seed)] == declared_pair_ids
            for seed in training_seeds
        )
        and isinstance(artifact_hashes, Mapping)
        and set(artifact_hashes) == seed_keys
        and all(
            _is_lower_sha256(artifact_hashes[str(seed)])
            for seed in training_seeds
        )
    )
    if not base_ready or schema_version == 1:
        return base_ready
    from measurement.shielding import SHIELD_POSE_CONTRACT_SHA256
    from spectrum.response_matrix import (
        NATIVE_GEANT4_DETECTOR_RESPONSE_CONTRACT_SHA256,
    )

    artifact_contracts = manifest.get("artifact_contract_sha256_by_scene")
    return bool(
        manifest.get("shield_pose_contract_sha256")
        == SHIELD_POSE_CONTRACT_SHA256
        and manifest.get("detector_response_contract_sha256")
        == NATIVE_GEANT4_DETECTOR_RESPONSE_CONTRACT_SHA256
        and manifest.get("obstacle_material_contract_sha256")
        == OBSTACLE_MATERIAL_CONTRACT_SHA256
        and manifest.get("transport_physics_table_contract_sha256")
        == TRANSPORT_PHYSICS_TABLE_CONTRACT_SHA256
        and isinstance(artifact_contracts, Mapping)
        and set(artifact_contracts) == seed_keys
        and all(
            _is_lower_sha256(artifact_contracts[str(seed)])
            for seed in training_seeds
        )
    )


def _training_manifest_fit_ready(
    manifest: Mapping[str, object],
    *,
    selected_ridge_lambda: float,
) -> bool:
    """Validate LOSO selection outputs without accepting holdout provenance."""
    expected_keys = (
        _ADDITIVE_SCATTER_TRAINING_BASE_KEYS
        | _ADDITIVE_SCATTER_TRAINING_FIT_KEYS
    )
    if manifest.get("schema_version") == 2:
        expected_keys |= _ADDITIVE_SCATTER_TRAINING_CONTRACT_KEYS
    if set(manifest) != expected_keys:
        return False
    if not _training_manifest_base_ready(manifest):
        return False
    scores = manifest.get("candidate_validation_scores")
    expected_score_keys = {
        format(value, ".12g")
        for value in ADDITIVE_SCATTER_RIDGE_LAMBDA_GRID
    }
    raw_fit_sample_count = manifest.get("fit_sample_count")
    raw_selected_score = manifest.get("selected_validation_score")
    raw_manifest_ridge = manifest.get("selected_ridge_lambda")
    loso_scene_ids = manifest.get("loso_scene_ids")
    if (
        type(raw_fit_sample_count) is not int
        or not _is_finite_json_number(raw_selected_score)
        or not _is_finite_json_number(raw_manifest_ridge)
    ):
        return False
    fit_sample_count = raw_fit_sample_count
    selected_score = float(raw_selected_score)
    manifest_ridge = float(raw_manifest_ridge)
    if (
        fit_sample_count < len(ADDITIVE_SCATTER_FEATURE_ORDER)
        or not isinstance(loso_scene_ids, list)
        or any(type(scene_id) is not str for scene_id in loso_scene_ids)
        or tuple(loso_scene_ids)
        != tuple(str(seed) for seed in manifest["training_scene_seeds"])
        or not isinstance(scores, Mapping)
        or set(scores) != expected_score_keys
        or any(type(key) is not str for key in scores)
        or any(
            not _is_finite_json_number(value) or float(value) < 0.0
            for value in scores.values()
        )
        or not np.isfinite(selected_score)
        or selected_score < 0.0
        or manifest_ridge != float(selected_ridge_lambda)
        or manifest.get("selection_completed") is not True
    ):
        return False
    minimum_score = min(float(value) for value in scores.values())
    tied_ridges = [
        ridge
        for ridge in ADDITIVE_SCATTER_RIDGE_LAMBDA_GRID
        if float(scores[format(ridge, ".12g")])
        <= minimum_score + 1.0e-12
    ]
    return bool(
        np.isclose(selected_score, minimum_score, rtol=0.0, atol=0.0)
        and manifest_ridge == max(tied_ridges)
    )


def klein_nishina_total_cross_section_cm2(
    energy_keV: NDArray[np.float64] | Sequence[float] | float,
) -> NDArray[np.float64]:
    """Return the total Klein-Nishina cross section per electron."""
    energy = np.asarray(energy_keV, dtype=np.float64)
    if np.any(~np.isfinite(energy)) or np.any(energy <= 0.0):
        raise ValueError("Gamma-line energies must be finite and positive.")
    alpha = energy / ELECTRON_REST_ENERGY_KEV
    log_term = np.log1p(2.0 * alpha)
    bracket = (
        (1.0 + alpha)
        / np.square(alpha)
        * (
            2.0 * (1.0 + alpha) / (1.0 + 2.0 * alpha)
            - log_term / alpha
        )
        + log_term / (2.0 * alpha)
        - (1.0 + 3.0 * alpha) / np.square(1.0 + 2.0 * alpha)
    )
    return (
        2.0
        * np.pi
        * CLASSICAL_ELECTRON_RADIUS_CM**2
        * np.maximum(bracket, 0.0)
    )


def material_compton_fraction_numpy(
    energy_keV: NDArray[np.float64] | Sequence[float] | float,
    total_mu_cm_inv: NDArray[np.float64] | Sequence[float] | float,
    *,
    density_g_cm3: float,
    z_over_a: float,
) -> NDArray[np.float64]:
    """Return the known Compton-to-total linear attenuation fraction."""
    energy, total_mu = np.broadcast_arrays(
        np.asarray(energy_keV, dtype=np.float64),
        np.asarray(total_mu_cm_inv, dtype=np.float64),
    )
    density = float(density_g_cm3)
    electron_ratio = float(z_over_a)
    if (
        np.any(~np.isfinite(total_mu))
        or np.any(total_mu < 0.0)
        or not np.isfinite(density)
        or density <= 0.0
        or not np.isfinite(electron_ratio)
        or electron_ratio <= 0.0
    ):
        raise ValueError("Material attenuation inputs must be physical.")
    compton_mu = (
        density
        * AVOGADRO_CONSTANT_MOL_INV
        * electron_ratio
        * klein_nishina_total_cross_section_cm2(energy)
    )
    return np.clip(
        np.divide(
            compton_mu,
            np.maximum(total_mu, np.finfo(np.float64).tiny),
        ),
        0.0,
        1.0,
    )


def composition_effective_z_over_a(
    composition_by_mass: Mapping[str, float],
) -> float:
    """Return the electron-per-atomic-mass ratio of a mass composition."""
    if not isinstance(composition_by_mass, Mapping) or not composition_by_mass:
        raise ValueError("Material composition must be a nonempty mapping.")
    normalized: dict[str, float] = {}
    for element, raw_weight in composition_by_mass.items():
        symbol = str(element)
        if symbol not in ELEMENT_Z_AND_ATOMIC_MASS:
            raise KeyError(
                f"No atomic-number contract exists for element {symbol!r}."
            )
        weight = float(raw_weight)
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError("Material mass fractions must be nonnegative.")
        normalized[symbol] = weight
    total = float(sum(normalized.values()))
    if total <= 0.0:
        raise ValueError("Material mass fractions must have positive mass.")
    return float(
        sum(
            (weight / total)
            * ELEMENT_Z_AND_ATOMIC_MASS[element][0]
            / ELEMENT_Z_AND_ATOMIC_MASS[element][1]
            for element, weight in normalized.items()
        )
    )


def material_compton_mu_cm_inv_numpy(
    energy_keV: NDArray[np.float64] | Sequence[float] | float,
    *,
    density_g_cm3: float,
    composition_by_mass: Mapping[str, float],
) -> NDArray[np.float64]:
    """Return physical Compton linear attenuation for a known composition."""
    density = float(density_g_cm3)
    if not np.isfinite(density) or density <= 0.0:
        raise ValueError("Material density must be finite and positive.")
    return (
        density
        * AVOGADRO_CONSTANT_MOL_INV
        * composition_effective_z_over_a(composition_by_mass)
        * klein_nishina_total_cross_section_cm2(energy_keV)
    )


def klein_nishina_forward_cone_fraction_numpy(
    energy_keV: NDArray[np.float64] | Sequence[float] | float,
    *,
    detector_radius_m: float,
    scatter_distance_m: NDArray[np.float64] | Sequence[float] | float,
) -> NDArray[np.float64]:
    """Integrate the normalized Klein-Nishina law over a detector cone."""
    energy, distance = np.broadcast_arrays(
        np.asarray(energy_keV, dtype=np.float64),
        np.asarray(scatter_distance_m, dtype=np.float64),
    )
    radius = float(detector_radius_m)
    if (
        radius <= 0.0
        or not np.isfinite(radius)
        or np.any(~np.isfinite(energy))
        or np.any(energy <= 0.0)
        or np.any(~np.isfinite(distance))
        or np.any(distance <= 0.0)
    ):
        raise ValueError("Klein-Nishina detector-cone inputs are invalid.")
    ratio = np.clip(radius / np.maximum(distance, radius), 0.0, 1.0)
    mu_min = np.sqrt(np.maximum(1.0 - np.square(ratio), 0.0))
    nodes = _CONE_QUADRATURE_NODES.reshape(
        (1,) * energy.ndim + (-1,)
    )
    weights = _CONE_QUADRATURE_WEIGHTS.reshape(
        (1,) * energy.ndim + (-1,)
    )
    midpoint = 0.5 * (1.0 + mu_min)[..., None]
    half_width = 0.5 * (1.0 - mu_min)[..., None]
    cosine = midpoint + half_width * nodes
    alpha = energy[..., None] / ELECTRON_REST_ENERGY_KEV
    scattered_ratio = 1.0 / (1.0 + alpha * (1.0 - cosine))
    angular = np.square(scattered_ratio) * (
        scattered_ratio
        + 1.0 / np.maximum(scattered_ratio, np.finfo(np.float64).tiny)
        - (1.0 - np.square(cosine))
    )
    numerator = (
        np.pi
        * CLASSICAL_ELECTRON_RADIUS_CM**2
        * np.sum(weights * half_width * angular, axis=-1)
    )
    denominator = klein_nishina_total_cross_section_cm2(energy)
    return np.clip(
        numerator / np.maximum(denominator, np.finfo(np.float64).tiny),
        0.0,
        1.0,
    )


def klein_nishina_forward_cone_fraction_torch(
    energy_keV: object,
    *,
    detector_radius_m: float,
    scatter_distance_m: object,
) -> object:
    """Return the Torch equivalent of the detector-cone KN integral."""
    import torch

    energy = torch.as_tensor(energy_keV)
    if energy.dtype != torch.float64:
        raise TypeError("Production scatter evaluation requires torch.float64.")
    distance = torch.as_tensor(
        scatter_distance_m,
        device=energy.device,
        dtype=energy.dtype,
    )
    energy, distance = torch.broadcast_tensors(energy, distance)
    radius = float(detector_radius_m)
    invalid_tensor_inputs = torch.stack(
        (
            torch.any(~torch.isfinite(energy)),
            torch.any(energy <= 0.0),
            torch.any(~torch.isfinite(distance)),
            torch.any(distance <= 0.0),
        )
    )
    if (
        not np.isfinite(radius)
        or radius <= 0.0
        or bool(torch.any(invalid_tensor_inputs))
    ):
        raise ValueError("Klein-Nishina detector-cone inputs are invalid.")
    ratio = torch.clamp(
        radius / torch.clamp(distance, min=radius),
        min=0.0,
        max=1.0,
    )
    mu_min = torch.sqrt(torch.clamp(1.0 - torch.square(ratio), min=0.0))
    nodes, weights = _torch_cone_quadrature_constants(
        device=energy.device,
        dtype=energy.dtype,
    )
    midpoint = 0.5 * (1.0 + mu_min).unsqueeze(-1)
    half_width = 0.5 * (1.0 - mu_min).unsqueeze(-1)
    cosine = midpoint + half_width * nodes
    alpha = energy.unsqueeze(-1) / ELECTRON_REST_ENERGY_KEV
    scattered_ratio = 1.0 / (1.0 + alpha * (1.0 - cosine))
    angular = torch.square(scattered_ratio) * (
        scattered_ratio
        + 1.0
        / torch.clamp(
            scattered_ratio,
            min=torch.finfo(energy.dtype).tiny,
        )
        - (1.0 - torch.square(cosine))
    )
    numerator = (
        np.pi
        * CLASSICAL_ELECTRON_RADIUS_CM**2
        * torch.sum(weights * half_width * angular, dim=-1)
    )
    alpha_total = energy / ELECTRON_REST_ENERGY_KEV
    log_term = torch.log1p(2.0 * alpha_total)
    bracket = (
        (1.0 + alpha_total)
        / torch.square(alpha_total)
        * (
            2.0
            * (1.0 + alpha_total)
            / (1.0 + 2.0 * alpha_total)
            - log_term / alpha_total
        )
        + log_term / (2.0 * alpha_total)
        - (1.0 + 3.0 * alpha_total)
        / torch.square(1.0 + 2.0 * alpha_total)
    )
    denominator = (
        2.0
        * np.pi
        * CLASSICAL_ELECTRON_RADIUS_CM**2
        * torch.clamp(bracket, min=0.0)
    )
    return torch.clamp(
        numerator
        / torch.clamp(
            denominator,
            min=torch.finfo(energy.dtype).tiny,
        ),
        min=0.0,
        max=1.0,
    )


def physical_scatter_basis_numpy(
    *,
    tau_fe: NDArray[np.float64],
    tau_pb: NDArray[np.float64],
    tau_obstacle: NDArray[np.float64],
    tau_obstacle_compton: NDArray[np.float64],
    distance_m: NDArray[np.float64],
    energy_keV: NDArray[np.float64],
    mu_fe_cm_inv: NDArray[np.float64],
    mu_pb_cm_inv: NDArray[np.float64],
    semantics: str = LEGACY_SCATTER_BASIS_SEMANTICS,
    detector_radius_m: float | None = None,
    fe_scatter_distance_m: float | None = None,
    pb_scatter_distance_m: float | None = None,
    obstacle_single_scatter_probability: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Return the batched physical scatter-opportunity feature basis."""
    (
        fe_tau,
        pb_tau,
        obstacle_tau,
        obstacle_compton_tau,
        distance,
        energy,
        mu_fe,
        mu_pb,
    ) = np.broadcast_arrays(
        np.asarray(tau_fe, dtype=np.float64),
        np.asarray(tau_pb, dtype=np.float64),
        np.asarray(tau_obstacle, dtype=np.float64),
        np.asarray(tau_obstacle_compton, dtype=np.float64),
        np.asarray(distance_m, dtype=np.float64),
        np.asarray(energy_keV, dtype=np.float64),
        np.asarray(mu_fe_cm_inv, dtype=np.float64),
        np.asarray(mu_pb_cm_inv, dtype=np.float64),
    )
    arrays = (
        fe_tau,
        pb_tau,
        obstacle_tau,
        obstacle_compton_tau,
        distance,
        energy,
        mu_fe,
        mu_pb,
    )
    if any(np.any(~np.isfinite(array)) for array in arrays) or any(
        np.any(array < 0.0) for array in arrays
    ):
        raise ValueError("Scatter basis inputs must be finite and nonnegative.")
    if np.any(energy <= 0.0):
        raise ValueError("Scatter basis line energies must be positive.")
    tolerance = 64.0 * np.finfo(np.float64).eps
    if np.any(obstacle_compton_tau > obstacle_tau * (1.0 + tolerance)):
        raise ValueError(
            "Obstacle Compton optical depth cannot exceed total optical depth."
        )
    fe_fraction = material_compton_fraction_numpy(
        energy,
        mu_fe,
        density_g_cm3=IRON_DENSITY_G_CM3,
        z_over_a=IRON_Z_OVER_A,
    )
    pb_fraction = material_compton_fraction_numpy(
        energy,
        mu_pb,
        density_g_cm3=LEAD_DENSITY_G_CM3,
        z_over_a=LEAD_Z_OVER_A,
    )
    obstacle_fraction = np.divide(
        obstacle_compton_tau,
        np.maximum(obstacle_tau, np.finfo(np.float64).tiny),
        out=np.zeros_like(obstacle_tau),
        where=obstacle_tau > 0.0,
    )
    air_compton_mu = (
        AIR_DENSITY_G_CM3
        * AVOGADRO_CONSTANT_MOL_INV
        * AIR_EFFECTIVE_Z_OVER_A
        * klein_nishina_total_cross_section_cm2(energy)
    )
    if semantics not in SCATTER_BASIS_SEMANTICS:
        raise ValueError("Scatter feature-basis semantics are invalid.")
    obstacle_fraction = np.clip(obstacle_fraction, 0.0, 1.0)
    air_compton_tau = distance * 100.0 * air_compton_mu
    air_survival_tau = air_compton_tau
    if semantics == DETECTOR_CONE_AIR_XCOM_SINGLE_SCATTER_BASIS_SEMANTICS:
        air_compton_tau = (
            distance
            * 100.0
            * G4_AIR_REFERENCE_DENSITY_G_CM3
            * AVOGADRO_CONSTANT_MOL_INV
            * AIR_EFFECTIVE_Z_OVER_A
            * klein_nishina_total_cross_section_cm2(energy)
        )
        air_survival_tau = (
            distance * 100.0 * dry_air_total_linear_attenuation_numpy(energy)
        )
    if semantics == LEGACY_SCATTER_BASIS_SEMANTICS:
        p_fe = -np.expm1(-fe_tau) * fe_fraction
        p_pb = -np.expm1(-pb_tau) * pb_fraction
        p_obstacle = -np.expm1(-obstacle_tau) * obstacle_fraction
        p_air = -np.expm1(-air_compton_tau)
        p_shield = p_fe + p_pb
        p_material = p_shield + p_obstacle
        interactions = (
            p_fe,
            p_pb,
            p_obstacle,
            p_air,
            p_fe * p_pb,
            p_shield * p_obstacle,
            p_material * p_air,
        )
    else:
        survival = np.exp(
            -(fe_tau + pb_tau + obstacle_tau + air_survival_tau)
        )
        fe_compton_tau = fe_tau * fe_fraction
        pb_compton_tau = pb_tau * pb_fraction
        obstacle_compton = obstacle_tau * obstacle_fraction
        shield_compton_tau = fe_compton_tau + pb_compton_tau
        material_compton_tau = shield_compton_tau + obstacle_compton
        interactions = (
            fe_compton_tau * survival,
            pb_compton_tau * survival,
            obstacle_compton * survival,
            air_compton_tau * survival,
            fe_compton_tau * pb_compton_tau * survival,
            shield_compton_tau * obstacle_compton * survival,
            material_compton_tau * air_compton_tau * survival,
        )
        if semantics in DETECTOR_CONE_SCATTER_BASIS_SEMANTICS:
            if (
                detector_radius_m is None
                or fe_scatter_distance_m is None
                or pb_scatter_distance_m is None
            ):
                raise ValueError(
                    "Detector-cone scatter requires detector and shield "
                    "geometry distances."
                )
            radius = float(detector_radius_m)
            fe_distance = float(fe_scatter_distance_m)
            pb_distance = float(pb_scatter_distance_m)
            if (
                not np.isfinite(radius)
                or radius <= 0.0
                or not np.isfinite(fe_distance)
                or fe_distance <= 0.0
                or not np.isfinite(pb_distance)
                or pb_distance <= 0.0
            ):
                raise ValueError("Detector-cone scatter geometry is invalid.")
            obstacle_distance = np.maximum(0.5 * distance, radius)
            acceptance = (
                klein_nishina_forward_cone_fraction_numpy(
                    energy,
                    detector_radius_m=radius,
                    scatter_distance_m=fe_distance,
                ),
                klein_nishina_forward_cone_fraction_numpy(
                    energy,
                    detector_radius_m=radius,
                    scatter_distance_m=pb_distance,
                ),
                klein_nishina_forward_cone_fraction_numpy(
                    energy,
                    detector_radius_m=radius,
                    scatter_distance_m=obstacle_distance,
                ),
                klein_nishina_forward_cone_fraction_numpy(
                    energy,
                    detector_radius_m=radius,
                    scatter_distance_m=obstacle_distance,
                ),
            )
            interactions = tuple(
                interaction * acceptance[index]
                if index < len(acceptance)
                else np.zeros_like(interaction)
                for index, interaction in enumerate(interactions)
            )
            if obstacle_single_scatter_probability is not None:
                obstacle_probability = np.asarray(
                    obstacle_single_scatter_probability,
                    dtype=np.float64,
                )
                if (
                    obstacle_probability.shape != obstacle_tau.shape
                    or np.any(~np.isfinite(obstacle_probability))
                    or np.any(obstacle_probability < 0.0)
                ):
                    raise ValueError(
                        "Obstacle single-scatter probability is invalid."
                    )
                interactions = (
                    interactions[0],
                    interactions[1],
                    obstacle_probability,
                    interactions[3],
                    interactions[4],
                    interactions[5],
                    interactions[6],
                )
    return np.stack(interactions, axis=-1)


def physical_scatter_basis_torch(
    *,
    tau_fe: object,
    tau_pb: object,
    tau_obstacle: object,
    tau_obstacle_compton: object,
    distance_m: object,
    energy_keV: object,
    mu_fe_cm_inv: object,
    mu_pb_cm_inv: object,
    semantics: str = LEGACY_SCATTER_BASIS_SEMANTICS,
    detector_radius_m: float | None = None,
    fe_scatter_distance_m: float | None = None,
    pb_scatter_distance_m: float | None = None,
    obstacle_single_scatter_probability: object | None = None,
) -> object:
    """Return the Torch equivalent of the physical scatter feature basis."""
    import torch

    fe_tau = torch.as_tensor(tau_fe)
    if fe_tau.dtype != torch.float64:
        raise TypeError("Production scatter evaluation requires torch.float64.")
    tensors = [
        torch.as_tensor(value, device=fe_tau.device, dtype=fe_tau.dtype)
        for value in (
            tau_pb,
            tau_obstacle,
            tau_obstacle_compton,
            distance_m,
            energy_keV,
            mu_fe_cm_inv,
            mu_pb_cm_inv,
        )
    ]
    compact_energy = tensors[4]
    compact_mu_fe = tensors[5]
    compact_mu_pb = tensors[6]
    (
        fe_tau,
        pb_tau,
        obstacle_tau,
        obstacle_compton_tau,
        distance,
        energy,
        mu_fe,
        mu_pb,
    ) = torch.broadcast_tensors(fe_tau, *tensors)
    values = (
        fe_tau,
        pb_tau,
        obstacle_tau,
        obstacle_compton_tau,
        distance,
        energy,
        mu_fe,
        mu_pb,
    )
    invalid_physical_inputs = torch.stack(
        tuple(
            check
            for value in values
            for check in (
                torch.any(~torch.isfinite(value)),
                torch.any(value < 0.0),
            )
        )
        + (
            torch.any(energy <= 0.0),
            torch.any(
                obstacle_compton_tau
                > obstacle_tau
                * (
                    1.0
                    + 64.0 * torch.finfo(fe_tau.dtype).eps
                )
            ),
        )
    )
    invalid_flags = (
        invalid_physical_inputs.detach().cpu().numpy().astype(bool, copy=False)
    )
    if np.any(invalid_flags[:16]):
        raise ValueError("Torch scatter basis inputs must be physical.")
    if invalid_flags[16]:
        raise ValueError("Torch scatter line energies must be positive.")
    if invalid_flags[17]:
        raise ValueError(
            "Obstacle Compton optical depth cannot exceed total optical depth."
        )

    def _material_fraction(
        total_mu: object,
        *,
        energy_value: object,
        density_g_cm3: float,
        z_over_a: float,
    ) -> object:
        """Return a material Compton fraction on the active Torch device."""
        active_energy = torch.as_tensor(
            energy_value,
            device=fe_tau.device,
            dtype=fe_tau.dtype,
        )
        active_total_mu = torch.as_tensor(
            total_mu,
            device=fe_tau.device,
            dtype=fe_tau.dtype,
        )
        alpha = active_energy / ELECTRON_REST_ENERGY_KEV
        log_term = torch.log1p(2.0 * alpha)
        bracket = (
            (1.0 + alpha)
            / torch.square(alpha)
            * (
                2.0 * (1.0 + alpha) / (1.0 + 2.0 * alpha)
                - log_term / alpha
            )
            + log_term / (2.0 * alpha)
            - (1.0 + 3.0 * alpha) / torch.square(1.0 + 2.0 * alpha)
        )
        sigma = (
            2.0
            * np.pi
            * CLASSICAL_ELECTRON_RADIUS_CM**2
            * torch.clamp(bracket, min=0.0)
        )
        compton_mu = (
            float(density_g_cm3)
            * AVOGADRO_CONSTANT_MOL_INV
            * float(z_over_a)
            * sigma
        )
        return torch.clamp(
            compton_mu
            / torch.clamp(
                active_total_mu,
                min=torch.finfo(fe_tau.dtype).tiny,
            ),
            min=0.0,
            max=1.0,
        )

    fe_fraction = _material_fraction(
        compact_mu_fe,
        energy_value=compact_energy,
        density_g_cm3=IRON_DENSITY_G_CM3,
        z_over_a=IRON_Z_OVER_A,
    )
    pb_fraction = _material_fraction(
        compact_mu_pb,
        energy_value=compact_energy,
        density_g_cm3=LEAD_DENSITY_G_CM3,
        z_over_a=LEAD_Z_OVER_A,
    )
    obstacle_fraction = torch.where(
        obstacle_tau > 0.0,
        obstacle_compton_tau
        / torch.clamp(obstacle_tau, min=torch.finfo(fe_tau.dtype).tiny),
        torch.zeros_like(obstacle_tau),
    )
    air_compton_mu = _material_fraction(
        torch.ones_like(compact_energy),
        energy_value=compact_energy,
        density_g_cm3=AIR_DENSITY_G_CM3,
        z_over_a=AIR_EFFECTIVE_Z_OVER_A,
    )
    if semantics not in SCATTER_BASIS_SEMANTICS:
        raise ValueError("Scatter feature-basis semantics are invalid.")
    obstacle_fraction = torch.clamp(
        obstacle_fraction,
        min=0.0,
        max=1.0,
    )
    air_compton_tau = distance * 100.0 * air_compton_mu
    air_survival_tau = air_compton_tau
    if semantics == DETECTOR_CONE_AIR_XCOM_SINGLE_SCATTER_BASIS_SEMANTICS:
        air_compton_mu = _material_fraction(
            torch.ones_like(compact_energy),
            energy_value=compact_energy,
            density_g_cm3=G4_AIR_REFERENCE_DENSITY_G_CM3,
            z_over_a=AIR_EFFECTIVE_Z_OVER_A,
        )
        air_compton_tau = distance * 100.0 * air_compton_mu
        air_survival_tau = (
            distance
            * 100.0
            * dry_air_total_linear_attenuation_torch(energy)
        )
    if semantics == LEGACY_SCATTER_BASIS_SEMANTICS:
        p_fe = -torch.expm1(-fe_tau) * fe_fraction
        p_pb = -torch.expm1(-pb_tau) * pb_fraction
        p_obstacle = -torch.expm1(-obstacle_tau) * obstacle_fraction
        p_air = -torch.expm1(-air_compton_tau)
        p_shield = p_fe + p_pb
        p_material = p_shield + p_obstacle
        interactions = (
            p_fe,
            p_pb,
            p_obstacle,
            p_air,
            p_fe * p_pb,
            p_shield * p_obstacle,
            p_material * p_air,
        )
    else:
        survival = torch.exp(
            -(fe_tau + pb_tau + obstacle_tau + air_survival_tau)
        )
        fe_compton_tau = fe_tau * fe_fraction
        pb_compton_tau = pb_tau * pb_fraction
        obstacle_compton = obstacle_tau * obstacle_fraction
        shield_compton_tau = fe_compton_tau + pb_compton_tau
        material_compton_tau = shield_compton_tau + obstacle_compton
        interactions = (
            fe_compton_tau * survival,
            pb_compton_tau * survival,
            obstacle_compton * survival,
            air_compton_tau * survival,
            fe_compton_tau * pb_compton_tau * survival,
            shield_compton_tau * obstacle_compton * survival,
            material_compton_tau * air_compton_tau * survival,
        )
        if semantics in DETECTOR_CONE_SCATTER_BASIS_SEMANTICS:
            if (
                detector_radius_m is None
                or fe_scatter_distance_m is None
                or pb_scatter_distance_m is None
            ):
                raise ValueError(
                    "Detector-cone scatter requires detector and shield "
                    "geometry distances."
                )
            radius = float(detector_radius_m)
            fe_distance = float(fe_scatter_distance_m)
            pb_distance = float(pb_scatter_distance_m)
            if (
                not np.isfinite(radius)
                or radius <= 0.0
                or not np.isfinite(fe_distance)
                or fe_distance <= 0.0
                or not np.isfinite(pb_distance)
                or pb_distance <= 0.0
            ):
                raise ValueError("Detector-cone scatter geometry is invalid.")

            obstacle_distance = torch.clamp(
                0.5 * distance,
                min=radius,
            )
            variable_distance_acceptance = (
                klein_nishina_forward_cone_fraction_torch(
                    energy,
                    detector_radius_m=radius,
                    scatter_distance_m=obstacle_distance,
                )
            )
            acceptance = (
                klein_nishina_forward_cone_fraction_torch(
                    compact_energy,
                    detector_radius_m=radius,
                    scatter_distance_m=fe_distance,
                ),
                klein_nishina_forward_cone_fraction_torch(
                    compact_energy,
                    detector_radius_m=radius,
                    scatter_distance_m=pb_distance,
                ),
                variable_distance_acceptance,
                variable_distance_acceptance,
            )
            interactions = tuple(
                interaction * acceptance[index]
                if index < len(acceptance)
                else torch.zeros_like(interaction)
                for index, interaction in enumerate(interactions)
            )
            if obstacle_single_scatter_probability is not None:
                obstacle_probability = torch.as_tensor(
                    obstacle_single_scatter_probability,
                    device=energy.device,
                    dtype=energy.dtype,
                )
                invalid_obstacle_probability = torch.stack(
                    (
                        torch.any(~torch.isfinite(obstacle_probability)),
                        torch.any(obstacle_probability < 0.0),
                    )
                )
                if (
                    obstacle_probability.shape != obstacle_tau.shape
                    or bool(torch.any(invalid_obstacle_probability))
                ):
                    raise ValueError(
                        "Obstacle single-scatter probability is invalid."
                    )
                interactions = (
                    interactions[0],
                    interactions[1],
                    obstacle_probability,
                    interactions[3],
                    interactions[4],
                    interactions[5],
                    interactions[6],
                )
    return torch.stack(interactions, dim=-1)


def scatter_basis_from_stored_geometry_numpy(
    *,
    stored_basis: NDArray[np.float64],
    transport_features: NDArray[np.float64],
    line_identity: Sequence[Mapping[str, object]],
    target_semantics: str,
    detector_radius_m: float | None = None,
    fe_scatter_distance_m: float | None = None,
    pb_scatter_distance_m: float | None = None,
) -> NDArray[np.float64]:
    """Return a versioned basis reconstructed from stored ray geometry.

    Mean-calibration and acceptance artifacts store the original seven-feature
    basis together with the four lossless ray features ``tau_fe``, ``tau_pb``,
    ``tau_obstacle``, and ``distance_m``.  The original obstacle feature also
    preserves the obstacle Compton fraction, so newer feature semantics can be
    reconstructed without rerunning or approximating Geant4 transport.
    """
    basis = np.asarray(stored_basis, dtype=np.float64)
    features = np.asarray(transport_features, dtype=np.float64)
    line_rows = tuple(line_identity)
    if (
        target_semantics not in SCATTER_BASIS_SEMANTICS
        or features.ndim < 2
        or features.shape[-1] != 4
        or basis.shape
        != features.shape[:-1] + (len(ADDITIVE_SCATTER_FEATURE_ORDER),)
        or features.shape[-2] != len(line_rows)
        or np.any(~np.isfinite(features))
        or np.any(features < 0.0)
        or np.any(~np.isfinite(basis))
        or np.any(basis < 0.0)
    ):
        raise ValueError("Stored scatter geometry is invalid.")
    if target_semantics == LEGACY_SCATTER_BASIS_SEMANTICS:
        return basis.copy()
    energies = np.asarray(
        [float(row["energy_keV"]) for row in line_rows],
        dtype=np.float64,
    )
    mu_fe = np.asarray(
        [float(row["mu_fe_cm_inv"]) for row in line_rows],
        dtype=np.float64,
    )
    mu_pb = np.asarray(
        [float(row["mu_pb_cm_inv"]) for row in line_rows],
        dtype=np.float64,
    )
    if (
        np.any(~np.isfinite(energies))
        or np.any(energies <= 0.0)
        or np.any(~np.isfinite(mu_fe))
        or np.any(mu_fe <= 0.0)
        or np.any(~np.isfinite(mu_pb))
        or np.any(mu_pb <= 0.0)
    ):
        raise ValueError("Stored scatter line identity is invalid.")
    line_shape = (1,) * (features.ndim - 2) + (len(line_rows),)
    obstacle_tau = features[..., 2]
    legacy_obstacle_probability = -np.expm1(-obstacle_tau)
    obstacle_fraction = np.divide(
        basis[..., 2],
        legacy_obstacle_probability,
        out=np.zeros_like(obstacle_tau),
        where=legacy_obstacle_probability > np.finfo(np.float64).tiny,
    )
    tolerance = 256.0 * np.finfo(np.float64).eps
    if np.any(obstacle_fraction > 1.0 + tolerance):
        raise ValueError(
            "Stored obstacle scatter feature exceeds its interaction bound."
        )
    obstacle_compton_tau = obstacle_tau * np.clip(
        obstacle_fraction,
        0.0,
        1.0,
    )
    return physical_scatter_basis_numpy(
        tau_fe=features[..., 0],
        tau_pb=features[..., 1],
        tau_obstacle=obstacle_tau,
        tau_obstacle_compton=obstacle_compton_tau,
        distance_m=features[..., 3],
        energy_keV=energies.reshape(line_shape),
        mu_fe_cm_inv=mu_fe.reshape(line_shape),
        mu_pb_cm_inv=mu_pb.reshape(line_shape),
        semantics=target_semantics,
        detector_radius_m=detector_radius_m,
        fe_scatter_distance_m=fe_scatter_distance_m,
        pb_scatter_distance_m=pb_scatter_distance_m,
    )


def _validate_direct_training_manifest(
    direct_manifest: Mapping[str, object],
    *,
    scatter_manifest: Mapping[str, object],
    selected_ridge_lambda: float,
) -> dict[str, object]:
    """Validate signed direct-response LOSO provenance."""
    manifest = json.loads(
        json.dumps(dict(direct_manifest), sort_keys=True, allow_nan=False)
    )
    expected_keys = {
        "schema_version",
        "base_training_manifest_sha256",
        "target_semantics",
        "fit_sample_count",
        "loso_scene_ids",
        "candidate_validation_scores",
        "selected_validation_score",
        "selected_ridge_lambda",
        "selection_completed",
    }
    score_keys = {
        format(value, ".12g")
        for value in ADDITIVE_SCATTER_RIDGE_LAMBDA_GRID
    }
    scores = manifest.get("candidate_validation_scores")
    base_keys = _ADDITIVE_SCATTER_TRAINING_BASE_KEYS
    if scatter_manifest.get("schema_version") == 2:
        base_keys |= _ADDITIVE_SCATTER_TRAINING_CONTRACT_KEYS
    base_manifest = {
        key: scatter_manifest[key]
        for key in base_keys
    }
    if (
        set(manifest) != expected_keys
        or manifest.get("schema_version") != 1
        or manifest.get("base_training_manifest_sha256")
        != _canonical_json_sha256(base_manifest)
        or manifest.get("target_semantics")
        != DIRECT_TRANSPORT_TARGET_SEMANTICS
        or manifest.get("fit_sample_count")
        != scatter_manifest.get("fit_sample_count")
        or manifest.get("loso_scene_ids")
        != scatter_manifest.get("loso_scene_ids")
        or not isinstance(scores, Mapping)
        or set(scores) != score_keys
        or any(
            not _is_finite_json_number(value) or float(value) < 0.0
            for value in scores.values()
        )
        or not _is_finite_json_number(
            manifest.get("selected_validation_score")
        )
        or not _is_finite_json_number(manifest.get("selected_ridge_lambda"))
        or manifest.get("selection_completed") is not True
    ):
        raise ValueError("Direct transport training provenance is invalid.")
    minimum_score = min(float(value) for value in scores.values())
    tied = [
        value
        for value in ADDITIVE_SCATTER_RIDGE_LAMBDA_GRID
        if float(scores[format(value, ".12g")])
        <= minimum_score + 1.0e-12
    ]
    if (
        float(manifest["selected_validation_score"]) != minimum_score
        or float(manifest["selected_ridge_lambda"]) != max(tied)
        or float(selected_ridge_lambda) != max(tied)
    ):
        raise ValueError("Direct transport ridge selection is not reproducible.")
    return manifest


@dataclass(frozen=True)
class AdditiveNoncollidedTransportResponse:
    """Store physical direct correction and nonnegative additive scatter."""

    coefficients: tuple[float, ...]
    ridge_lambda: float
    training_manifest: Mapping[str, object]
    direct_log_coefficients: tuple[float, ...] = ()
    direct_ridge_lambda: float | None = None
    direct_training_manifest: Mapping[str, object] | None = None
    feature_basis_semantics: str = LEGACY_SCATTER_BASIS_SEMANTICS

    def __post_init__(self) -> None:
        """Validate and freeze model coefficients and training provenance."""
        coefficients = tuple(float(value) for value in self.coefficients)
        direct_coefficients = tuple(
            float(value) for value in self.direct_log_coefficients
        )
        basis_semantics = str(self.feature_basis_semantics)
        ridge_lambda = float(self.ridge_lambda)
        if (
            len(coefficients) != len(ADDITIVE_SCATTER_FEATURE_ORDER)
            or any(not np.isfinite(value) or value < 0.0 for value in coefficients)
            or not any(value > 0.0 for value in coefficients)
            or ridge_lambda not in ADDITIVE_SCATTER_RIDGE_LAMBDA_GRID
        ):
            raise ValueError(
                "Additive scatter coefficients must be nonnegative, nonzero, "
                "and use the predeclared ridge grid."
            )
        if basis_semantics not in SCATTER_BASIS_SEMANTICS:
            raise ValueError("Scatter feature-basis semantics are invalid.")
        if not isinstance(self.training_manifest, Mapping):
            raise TypeError("Scatter training provenance must be a mapping.")
        manifest = json.loads(
            json.dumps(
                dict(self.training_manifest),
                sort_keys=True,
                allow_nan=False,
            )
        )
        if not _training_manifest_fit_ready(
            manifest,
            selected_ridge_lambda=ridge_lambda,
        ):
            raise ValueError(
                "Additive scatter training provenance must exactly bind the "
                "designated training scenes, declared shield-pair subset and "
                "scenarios, and the predeclared LOSO ridge selection."
            )
        has_direct = bool(direct_coefficients)
        if has_direct != (self.direct_ridge_lambda is not None) or has_direct != (
            self.direct_training_manifest is not None
        ):
            raise ValueError(
                "Direct coefficients, ridge, and provenance must be declared "
                "together."
            )
        direct_manifest = None
        if has_direct:
            if (
                len(direct_coefficients) != len(ADDITIVE_SCATTER_FEATURE_ORDER)
                or any(not np.isfinite(value) for value in direct_coefficients)
                or float(self.direct_ridge_lambda)
                not in ADDITIVE_SCATTER_RIDGE_LAMBDA_GRID
            ):
                raise ValueError("Direct transport correction is invalid.")
            direct_manifest = _validate_direct_training_manifest(
                self.direct_training_manifest,
                scatter_manifest=manifest,
                selected_ridge_lambda=float(self.direct_ridge_lambda),
            )
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "ridge_lambda", ridge_lambda)
        object.__setattr__(self, "direct_log_coefficients", direct_coefficients)
        object.__setattr__(self, "feature_basis_semantics", basis_semantics)
        object.__setattr__(
            self,
            "direct_ridge_lambda",
            None if not has_direct else float(self.direct_ridge_lambda),
        )
        object.__setattr__(
            self,
            "training_manifest",
            MappingProxyType(manifest),
        )
        object.__setattr__(
            self,
            "direct_training_manifest",
            None if direct_manifest is None else MappingProxyType(direct_manifest),
        )
        object.__setattr__(
            self,
            "_contract_hash_sha256",
            _canonical_json_sha256(self._contract_payload()),
        )

    def _contract_payload(self) -> dict[str, object]:
        """Return fields that define the immutable fitted response."""
        payload = {
            "model": ADDITIVE_SCATTER_MODEL_ID,
            "feature_order": list(ADDITIVE_SCATTER_FEATURE_ORDER),
            "coefficients": list(self.coefficients),
            "fit_family": "nonnegative_ridge_nnls",
            "ridge_lambda_grid": list(ADDITIVE_SCATTER_RIDGE_LAMBDA_GRID),
            "selected_ridge_lambda": float(self.ridge_lambda),
            "selection_objective": (
                "leave_one_training_scene_out_weighted_log1p_mse"
            ),
            "tie_break": "largest_ridge_lambda_within_1e-12",
            "coefficient_scope": (
                "one_global_vector_all_scenes_pairs_sources_isotopes_lines"
            ),
            "incident_label_semantics": (
                ADDITIVE_SCATTER_INCIDENT_LABEL_SEMANTICS
            ),
            "target_semantics": ADDITIVE_SCATTER_TARGET_SEMANTICS,
            "kernel_equation": (
                "total=uncollided+unattenuated_geometric*max(0,basis@coefficients)"
            ),
            "training": dict(self.training_manifest),
        }
        if self.direct_log_coefficients:
            payload.update(
                {
                    "direct_log_coefficients": list(
                        self.direct_log_coefficients
                    ),
                    "direct_selected_ridge_lambda": float(
                        self.direct_ridge_lambda
                    ),
                    "direct_maximum_abs_log_correction": (
                        DIRECT_TRANSPORT_MAXIMUM_ABS_LOG_CORRECTION
                    ),
                    "direct_target_semantics": (
                        DIRECT_TRANSPORT_TARGET_SEMANTICS
                    ),
                    "direct_training": dict(self.direct_training_manifest),
                    "kernel_equation": (
                        "corrected_uncollided=uncollided*exp(clip(basis@"
                        "direct_log_coefficients,-2,2));total="
                        "corrected_uncollided+unattenuated_geometric*"
                        "max(0,basis@coefficients)"
                    ),
                }
            )
        if self.feature_basis_semantics != LEGACY_SCATTER_BASIS_SEMANTICS:
            payload["feature_basis_semantics"] = self.feature_basis_semantics
        return payload

    @property
    def contract_hash_sha256(self) -> str:
        """Return the immutable fitted response contract hash."""
        return str(self._contract_hash_sha256)

    @property
    def training_ready(self) -> bool:
        """Return whether strict training-only provenance authenticates the fit."""
        return bool(
            self.training_manifest.get("schema_version") == 2
            and _training_manifest_fit_ready(
                self.training_manifest,
                selected_ridge_lambda=self.ridge_lambda,
            )
            and (
                not self.direct_log_coefficients
                or self.direct_training_manifest is not None
            )
        )

    def to_payload(self) -> dict[str, object]:
        """Return an authenticated JSON-compatible response payload."""
        payload = {
            "schema_version": (
                3
                if self.feature_basis_semantics
                != LEGACY_SCATTER_BASIS_SEMANTICS
                else 2 if self.direct_log_coefficients else 1
            ),
            **self._contract_payload(),
            "contract_hash_sha256": self.contract_hash_sha256,
        }
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "AdditiveNoncollidedTransportResponse":
        """Reconstruct and byte-authenticate one additive response payload."""
        raw_coefficients = (
            payload.get("coefficients")
            if isinstance(payload, Mapping)
            else None
        )
        raw_ridge = (
            payload.get("selected_ridge_lambda")
            if isinstance(payload, Mapping)
            else None
        )
        raw_ridge_grid = (
            payload.get("ridge_lambda_grid")
            if isinstance(payload, Mapping)
            else None
        )
        has_direct_payload = isinstance(payload, Mapping) and (
            "direct_log_coefficients" in payload
        )
        if (
            not isinstance(payload, Mapping)
            or type(payload.get("schema_version")) is not int
            or payload.get("schema_version") not in (1, 2, 3)
            or payload.get("model") != ADDITIVE_SCATTER_MODEL_ID
            or tuple(payload.get("feature_order", ()))
            != ADDITIVE_SCATTER_FEATURE_ORDER
            or payload.get("fit_family") != "nonnegative_ridge_nnls"
            or not isinstance(raw_ridge_grid, list)
            or any(
                not _is_finite_json_number(value)
                for value in raw_ridge_grid
            )
            or tuple(raw_ridge_grid)
            != ADDITIVE_SCATTER_RIDGE_LAMBDA_GRID
            or not isinstance(raw_coefficients, list)
            or any(
                not _is_finite_json_number(value)
                for value in raw_coefficients
            )
            or not _is_finite_json_number(raw_ridge)
            or payload.get("incident_label_semantics")
            != ADDITIVE_SCATTER_INCIDENT_LABEL_SEMANTICS
            or payload.get("target_semantics")
            != ADDITIVE_SCATTER_TARGET_SEMANTICS
            or not isinstance(payload.get("training"), Mapping)
            or not _is_lower_sha256(payload.get("contract_hash_sha256"))
        ):
            raise ValueError("Additive scatter response schema is invalid.")
        model = cls(
            coefficients=tuple(
                _strict_json_number(
                    value,
                    field_name=f"coefficients[{index}]",
                )
                for index, value in enumerate(raw_coefficients)
            ),
            ridge_lambda=_strict_json_number(
                raw_ridge,
                field_name="selected_ridge_lambda",
            ),
            training_manifest=payload["training"],
            direct_log_coefficients=tuple(
                _strict_json_number(
                    value,
                    field_name=f"direct_log_coefficients[{index}]",
                )
                for index, value in enumerate(
                    payload.get("direct_log_coefficients", ())
                )
            ),
            direct_ridge_lambda=(
                None
                if not has_direct_payload
                else _strict_json_number(
                    payload.get("direct_selected_ridge_lambda"),
                    field_name="direct_selected_ridge_lambda",
                )
            ),
            direct_training_manifest=payload.get("direct_training"),
            feature_basis_semantics=(
                LEGACY_SCATTER_BASIS_SEMANTICS
                if payload.get("schema_version") in (1, 2)
                else str(payload.get("feature_basis_semantics"))
            ),
        )
        if model.to_payload() != dict(payload):
            raise ValueError(
                "Additive scatter response does not exactly reconstruct its "
                "declared contract."
            )
        return model

    def scatter_fraction_numpy(
        self,
        feature_basis: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return batched nonnegative scatter fractions using NumPy."""
        basis = np.asarray(feature_basis, dtype=np.float64)
        if (
            basis.ndim < 1
            or basis.shape[-1] != len(self.coefficients)
            or np.any(~np.isfinite(basis))
            or np.any(basis < 0.0)
        ):
            raise ValueError("Scatter feature basis is invalid.")
        return np.maximum(
            np.einsum(
                "...f,f->...",
                basis,
                np.asarray(self.coefficients, dtype=np.float64),
                optimize=True,
            ),
            0.0,
        )

    def scatter_fraction_torch(self, feature_basis: object) -> object:
        """Return batched nonnegative scatter fractions using Torch."""
        import torch

        basis = torch.as_tensor(feature_basis)
        if basis.dtype != torch.float64:
            raise TypeError("Production scatter evaluation requires torch.float64.")
        if (
            basis.ndim < 1
            or basis.shape[-1] != len(self.coefficients)
            or bool(torch.any(~torch.isfinite(basis)))
            or bool(torch.any(basis < 0.0))
        ):
            raise ValueError("Torch scatter feature basis is invalid.")
        coefficients = torch.as_tensor(
            self.coefficients,
            device=basis.device,
            dtype=basis.dtype,
        )
        return torch.clamp(
            torch.einsum("...f,f->...", basis, coefficients),
            min=0.0,
        )

    def total_kernel_numpy(
        self,
        unattenuated_geometric_kernel: NDArray[np.float64],
        uncollided_kernel: NDArray[np.float64],
        feature_basis: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return the additive total detector-entry kernel using NumPy."""
        unattenuated = np.asarray(
            unattenuated_geometric_kernel,
            dtype=np.float64,
        )
        uncollided = np.asarray(uncollided_kernel, dtype=np.float64)
        if (
            unattenuated.shape != uncollided.shape
            or feature_basis.shape != unattenuated.shape + (len(self.coefficients),)
            or np.any(~np.isfinite(unattenuated))
            or np.any(unattenuated < 0.0)
            or np.any(~np.isfinite(uncollided))
            or np.any(uncollided < 0.0)
        ):
            raise ValueError("Additive scatter kernel inputs are invalid.")
        corrected = self.corrected_uncollided_kernel_numpy(
            uncollided,
            feature_basis,
        )
        return corrected + unattenuated * self.scatter_fraction_numpy(
            feature_basis
        )

    def corrected_uncollided_kernel_numpy(
        self,
        uncollided_kernel: NDArray[np.float64],
        feature_basis: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return the geometry-conditioned direct detector-entry kernel."""
        uncollided = np.asarray(uncollided_kernel, dtype=np.float64)
        basis = np.asarray(feature_basis, dtype=np.float64)
        if (
            basis.shape
            != uncollided.shape + (len(ADDITIVE_SCATTER_FEATURE_ORDER),)
            or np.any(~np.isfinite(uncollided))
            or np.any(uncollided < 0.0)
            or np.any(~np.isfinite(basis))
            or np.any(basis < 0.0)
        ):
            raise ValueError("Direct transport kernel inputs are invalid.")
        if not self.direct_log_coefficients:
            return uncollided.copy()
        coefficients = np.asarray(
            self.direct_log_coefficients,
            dtype=np.float64,
        )
        log_scale = np.einsum(
            "...f,f->...",
            basis,
            coefficients,
            optimize=True,
        )
        return uncollided * np.exp(
            np.clip(
                log_scale,
                -DIRECT_TRANSPORT_MAXIMUM_ABS_LOG_CORRECTION,
                DIRECT_TRANSPORT_MAXIMUM_ABS_LOG_CORRECTION,
            )
        )

    def total_kernel_torch(
        self,
        unattenuated_geometric_kernel: object,
        uncollided_kernel: object,
        feature_basis: object,
    ) -> object:
        """Return the additive total detector-entry kernel using Torch."""
        import torch

        unattenuated = torch.as_tensor(unattenuated_geometric_kernel)
        if unattenuated.dtype != torch.float64:
            raise TypeError("Production scatter evaluation requires torch.float64.")
        uncollided = torch.as_tensor(
            uncollided_kernel,
            device=unattenuated.device,
            dtype=unattenuated.dtype,
        )
        basis = torch.as_tensor(
            feature_basis,
            device=unattenuated.device,
            dtype=unattenuated.dtype,
        )
        if (
            unattenuated.shape != uncollided.shape
            or basis.shape
            != unattenuated.shape + (len(self.coefficients),)
            or bool(torch.any(~torch.isfinite(unattenuated)))
            or bool(torch.any(unattenuated < 0.0))
            or bool(torch.any(~torch.isfinite(uncollided)))
            or bool(torch.any(uncollided < 0.0))
        ):
            raise ValueError("Torch additive scatter kernel inputs are invalid.")
        corrected = self.corrected_uncollided_kernel_torch(
            uncollided,
            basis,
        )
        return (
            corrected
            + unattenuated * self.scatter_fraction_torch(basis)
        )

    def corrected_uncollided_kernel_torch(
        self,
        uncollided_kernel: object,
        feature_basis: object,
    ) -> object:
        """Return the Torch geometry-conditioned direct entry kernel."""
        import torch

        uncollided = torch.as_tensor(uncollided_kernel)
        basis = torch.as_tensor(
            feature_basis,
            device=uncollided.device,
            dtype=uncollided.dtype,
        )
        if (
            uncollided.dtype != torch.float64
            or basis.shape
            != uncollided.shape + (len(ADDITIVE_SCATTER_FEATURE_ORDER),)
            or bool(torch.any(~torch.isfinite(uncollided)))
            or bool(torch.any(uncollided < 0.0))
            or bool(torch.any(~torch.isfinite(basis)))
            or bool(torch.any(basis < 0.0))
        ):
            raise ValueError("Torch direct transport inputs are invalid.")
        if not self.direct_log_coefficients:
            return uncollided.clone()
        coefficients = torch.as_tensor(
            self.direct_log_coefficients,
            device=uncollided.device,
            dtype=uncollided.dtype,
        )
        log_scale = torch.einsum("...f,f->...", basis, coefficients)
        return uncollided * torch.exp(
            torch.clamp(
                log_scale,
                min=-DIRECT_TRANSPORT_MAXIMUM_ABS_LOG_CORRECTION,
                max=DIRECT_TRANSPORT_MAXIMUM_ABS_LOG_CORRECTION,
            )
        )


@dataclass(frozen=True)
class PhysicsOnlyNoncollidedTransportResponse:
    """Evaluate direct plus single-Compton transport without scene fitting."""

    detector_radius_m: float
    fe_scatter_distance_m: float
    pb_scatter_distance_m: float
    feature_basis_semantics: str = (
        DETECTOR_CONE_AIR_XCOM_SINGLE_SCATTER_BASIS_SEMANTICS
    )

    def __post_init__(self) -> None:
        """Validate immutable detector-cone integration geometry."""
        for name in (
            "detector_radius_m",
            "fe_scatter_distance_m",
            "pb_scatter_distance_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        if self.feature_basis_semantics not in (
            DETECTOR_CONE_SINGLE_SCATTER_BASIS_SEMANTICS,
            DETECTOR_CONE_AIR_XCOM_SINGLE_SCATTER_BASIS_SEMANTICS,
        ):
            raise ValueError("Physics-only scatter semantics are invalid.")
        object.__setattr__(
            self,
            "_contract_hash_sha256",
            _canonical_json_sha256(self._contract_payload()),
        )

    def _contract_payload(self) -> dict[str, object]:
        """Return the universal physical response contract fields."""
        uses_xcom_air = (
            self.feature_basis_semantics
            == DETECTOR_CONE_AIR_XCOM_SINGLE_SCATTER_BASIS_SEMANTICS
        )
        payload: dict[str, object] = {
            "model": (
                PHYSICS_ONLY_TRANSPORT_RESPONSE_ID
                if uses_xcom_air
                else LEGACY_PHYSICS_ONLY_TRANSPORT_RESPONSE_ID
            ),
            "feature_order": list(ADDITIVE_SCATTER_FEATURE_ORDER),
            "feature_basis_semantics": self.feature_basis_semantics,
            "mean_model": (
                "beer_lambert_uncollided_with_nist_xcom_dry_air_plus_"
                "detector_cone_single_compton"
                if uses_xcom_air
                else "beer_lambert_uncollided_plus_detector_cone_single_compton"
            ),
            "fit_family": "none_physics_only",
            "detector_radius_m": float(self.detector_radius_m),
            "fe_scatter_distance_m": float(self.fe_scatter_distance_m),
            "pb_scatter_distance_m": float(self.pb_scatter_distance_m),
            "angular_quadrature_order": int(
                _CONE_QUADRATURE_NODES.size
            ),
            "material_path_quadrature_order": 2,
            "higher_order_mean": "excluded_positive_nuisance_owned_by_likelihood",
            "obstacle_material_contract_id": OBSTACLE_MATERIAL_CONTRACT_ID,
            "obstacle_material_contract_sha256": (
                OBSTACLE_MATERIAL_CONTRACT_SHA256
            ),
            "transport_physics_table_contract_id": (
                TRANSPORT_PHYSICS_TABLE_CONTRACT_ID
            ),
            "transport_physics_table_contract_sha256": (
                TRANSPORT_PHYSICS_TABLE_CONTRACT_SHA256
            ),
        }
        if uses_xcom_air:
            payload.update(
                {
                    "dry_air_total_attenuation_contract_id": (
                        NIST_XCOM_DRY_AIR_TOTAL_CONTRACT_ID
                    ),
                    "dry_air_total_attenuation_contract_sha256": (
                        NIST_XCOM_DRY_AIR_TOTAL_CONTRACT_SHA256
                    ),
                }
            )
        return payload

    @property
    def contract_hash_sha256(self) -> str:
        """Return the immutable physics-only response digest."""
        return str(self._contract_hash_sha256)

    @property
    def training_ready(self) -> bool:
        """Return true because the response has no empirical fit."""
        return True

    @property
    def physics_only(self) -> bool:
        """Return whether this response excludes scene-trained coefficients."""
        return True

    def to_payload(self) -> dict[str, object]:
        """Return an authenticated JSON-compatible physics payload."""
        return {
            "schema_version": (
                2
                if self.feature_basis_semantics
                == DETECTOR_CONE_AIR_XCOM_SINGLE_SCATTER_BASIS_SEMANTICS
                else 1
            ),
            **self._contract_payload(),
            "contract_hash_sha256": self.contract_hash_sha256,
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "PhysicsOnlyNoncollidedTransportResponse":
        """Reconstruct and authenticate a physics-only response payload."""
        if not isinstance(payload, Mapping):
            raise ValueError("Physics-only transport response is invalid.")
        model_id = payload.get("model")
        schema_version = payload.get("schema_version")
        if (
            (model_id, schema_version)
            not in (
                (LEGACY_PHYSICS_ONLY_TRANSPORT_RESPONSE_ID, 1),
                (PHYSICS_ONLY_TRANSPORT_RESPONSE_ID, 2),
            )
            or not _is_lower_sha256(payload.get("contract_hash_sha256"))
        ):
            raise ValueError("Physics-only transport response is invalid.")
        model = cls(
            detector_radius_m=_strict_json_number(
                payload.get("detector_radius_m"),
                field_name="detector_radius_m",
            ),
            fe_scatter_distance_m=_strict_json_number(
                payload.get("fe_scatter_distance_m"),
                field_name="fe_scatter_distance_m",
            ),
            pb_scatter_distance_m=_strict_json_number(
                payload.get("pb_scatter_distance_m"),
                field_name="pb_scatter_distance_m",
            ),
            feature_basis_semantics=str(payload.get("feature_basis_semantics")),
        )
        if model.to_payload() != dict(payload):
            raise ValueError(
                "Physics-only transport response does not reconstruct exactly."
            )
        return model

    def scatter_fraction_numpy(
        self,
        feature_basis: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return the single-scatter fraction from physical basis terms."""
        basis = np.asarray(feature_basis, dtype=np.float64)
        if (
            basis.ndim < 1
            or basis.shape[-1] != len(ADDITIVE_SCATTER_FEATURE_ORDER)
            or np.any(~np.isfinite(basis))
            or np.any(basis < 0.0)
        ):
            raise ValueError("Physics-only scatter feature basis is invalid.")
        return np.sum(basis[..., :4], axis=-1)

    def scatter_fraction_torch(self, feature_basis: object) -> object:
        """Return the single-scatter fraction using Torch."""
        import torch

        basis = torch.as_tensor(feature_basis)
        if (
            basis.dtype != torch.float64
            or basis.ndim < 1
            or basis.shape[-1] != len(ADDITIVE_SCATTER_FEATURE_ORDER)
            or bool(torch.any(~torch.isfinite(basis)))
            or bool(torch.any(basis < 0.0))
        ):
            raise ValueError("Torch physics-only scatter basis is invalid.")
        return torch.sum(basis[..., :4], dim=-1)

    def corrected_uncollided_kernel_numpy(
        self,
        uncollided_kernel: NDArray[np.float64],
        feature_basis: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return exact Beer-Lambert uncollided transport unchanged."""
        uncollided = np.asarray(uncollided_kernel, dtype=np.float64)
        basis = np.asarray(feature_basis, dtype=np.float64)
        if (
            basis.shape
            != uncollided.shape + (len(ADDITIVE_SCATTER_FEATURE_ORDER),)
            or np.any(~np.isfinite(uncollided))
            or np.any(uncollided < 0.0)
        ):
            raise ValueError("Physics-only direct transport input is invalid.")
        return uncollided.copy()

    def corrected_uncollided_kernel_torch(
        self,
        uncollided_kernel: object,
        feature_basis: object,
    ) -> object:
        """Return exact Torch Beer-Lambert transport unchanged."""
        import torch

        uncollided = torch.as_tensor(uncollided_kernel)
        basis = torch.as_tensor(
            feature_basis,
            device=uncollided.device,
            dtype=uncollided.dtype,
        )
        if (
            uncollided.dtype != torch.float64
            or basis.shape
            != uncollided.shape + (len(ADDITIVE_SCATTER_FEATURE_ORDER),)
            or bool(torch.any(~torch.isfinite(uncollided)))
            or bool(torch.any(uncollided < 0.0))
        ):
            raise ValueError("Torch physics-only direct transport is invalid.")
        return uncollided.clone()

    def total_kernel_numpy(
        self,
        unattenuated_geometric_kernel: NDArray[np.float64],
        uncollided_kernel: NDArray[np.float64],
        feature_basis: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return Beer-Lambert direct plus physical single scatter."""
        unattenuated = np.asarray(
            unattenuated_geometric_kernel,
            dtype=np.float64,
        )
        uncollided = self.corrected_uncollided_kernel_numpy(
            uncollided_kernel,
            feature_basis,
        )
        if unattenuated.shape != uncollided.shape:
            raise ValueError("Physics-only total kernel shapes disagree.")
        return uncollided + unattenuated * self.scatter_fraction_numpy(
            feature_basis
        )

    def total_kernel_torch(
        self,
        unattenuated_geometric_kernel: object,
        uncollided_kernel: object,
        feature_basis: object,
    ) -> object:
        """Return Torch Beer-Lambert direct plus physical single scatter."""
        import torch

        unattenuated = torch.as_tensor(unattenuated_geometric_kernel)
        uncollided = self.corrected_uncollided_kernel_torch(
            uncollided_kernel,
            feature_basis,
        )
        if unattenuated.shape != uncollided.shape:
            raise ValueError("Torch physics-only total kernel shapes disagree.")
        return uncollided + unattenuated * self.scatter_fraction_torch(
            feature_basis
        )


def _fit_nonnegative_ridge(
    features_nf: NDArray[np.float64],
    targets_n: NDArray[np.float64],
    weights_n: NDArray[np.float64],
    *,
    ridge_lambda: float,
) -> NDArray[np.float64]:
    """Fit one weighted nonnegative ridge model with an augmented NNLS solve."""
    features = np.asarray(features_nf, dtype=np.float64)
    targets = np.asarray(targets_n, dtype=np.float64)
    weights = np.asarray(weights_n, dtype=np.float64)
    root_weights = np.sqrt(weights)
    weighted_features = features * root_weights[:, np.newaxis]
    weighted_targets = targets * root_weights
    ridge = float(ridge_lambda)
    if ridge > 0.0:
        weighted_features = np.concatenate(
            (
                weighted_features,
                np.sqrt(ridge)
                * np.eye(features.shape[1], dtype=np.float64),
            ),
            axis=0,
        )
        weighted_targets = np.concatenate(
            (
                weighted_targets,
                np.zeros(features.shape[1], dtype=np.float64),
            )
        )
    coefficients, _ = nnls(weighted_features, weighted_targets)
    return np.asarray(coefficients, dtype=np.float64)


def _fit_signed_ridge(
    features_nf: NDArray[np.float64],
    targets_n: NDArray[np.float64],
    weights_n: NDArray[np.float64],
    *,
    ridge_lambda: float,
) -> NDArray[np.float64]:
    """Fit one weighted signed ridge model without categorical features."""
    features = np.asarray(features_nf, dtype=np.float64)
    targets = np.asarray(targets_n, dtype=np.float64)
    weights = np.asarray(weights_n, dtype=np.float64)
    root = np.sqrt(weights)
    design = features * root[:, np.newaxis]
    response = targets * root
    ridge = float(ridge_lambda)
    gram = design.T @ design + ridge * np.eye(
        design.shape[1],
        dtype=np.float64,
    )
    return np.asarray(
        np.linalg.solve(gram, design.T @ response),
        dtype=np.float64,
    )


def fit_additive_noncollided_transport_response(
    feature_basis_nf: NDArray[np.float64],
    target_scatter_fraction_n: NDArray[np.float64],
    sample_weights_n: NDArray[np.float64],
    training_scene_ids_n: Sequence[object],
    *,
    training_manifest: Mapping[str, object],
    direct_log_ratio_n: NDArray[np.float64] | None = None,
    feature_basis_semantics: str = LEGACY_SCATTER_BASIS_SEMANTICS,
) -> AdditiveNoncollidedTransportResponse:
    """Fit the predeclared global model using training scenes only.

    Ridge selection uses leave-one-training-scene-out weighted log1p mean
    squared error.  Holdout observations are intentionally not accepted by this
    API, so they cannot influence coefficients or regularization selection.
    """
    expected_training_keys = _ADDITIVE_SCATTER_TRAINING_BASE_KEYS
    if isinstance(training_manifest, Mapping) and (
        training_manifest.get("schema_version") == 2
    ):
        expected_training_keys |= _ADDITIVE_SCATTER_TRAINING_CONTRACT_KEYS
    if (
        not isinstance(training_manifest, Mapping)
        or set(training_manifest) != expected_training_keys
        or not _training_manifest_base_ready(training_manifest)
    ):
        raise ValueError(
            "Scatter fitting requires the exact designated-training manifest."
        )
    features = np.asarray(feature_basis_nf, dtype=np.float64)
    targets = np.asarray(target_scatter_fraction_n, dtype=np.float64).reshape(-1)
    weights = np.asarray(sample_weights_n, dtype=np.float64).reshape(-1)
    scene_ids = np.asarray(
        [str(value) for value in training_scene_ids_n],
        dtype=object,
    ).reshape(-1)
    if (
        features.ndim != 2
        or features.shape[1] != len(ADDITIVE_SCATTER_FEATURE_ORDER)
        or targets.shape != (features.shape[0],)
        or weights.shape != targets.shape
        or scene_ids.shape != targets.shape
        or features.shape[0] < len(ADDITIVE_SCATTER_FEATURE_ORDER)
        or np.unique(scene_ids).size < 2
        or np.any(~np.isfinite(features))
        or np.any(features < 0.0)
        or np.any(~np.isfinite(targets))
        or np.any(targets < 0.0)
        or np.any(~np.isfinite(weights))
        or np.any(weights <= 0.0)
    ):
        raise ValueError("Additive scatter training arrays are invalid.")
    unique_scenes = np.unique(scene_ids)
    candidate_scores: list[tuple[float, float]] = []
    log_targets = np.log1p(targets)
    for ridge_lambda in ADDITIVE_SCATTER_RIDGE_LAMBDA_GRID:
        squared_error_sum = 0.0
        weight_sum = 0.0
        for held_out_scene in unique_scenes:
            validation_mask = scene_ids == held_out_scene
            training_mask = ~validation_mask
            coefficients = _fit_nonnegative_ridge(
                features[training_mask],
                targets[training_mask],
                weights[training_mask],
                ridge_lambda=float(ridge_lambda),
            )
            predictions = np.maximum(
                features[validation_mask] @ coefficients,
                0.0,
            )
            errors = np.square(
                np.log1p(predictions) - log_targets[validation_mask]
            )
            squared_error_sum += float(
                np.sum(weights[validation_mask] * errors)
            )
            weight_sum += float(np.sum(weights[validation_mask]))
        candidate_scores.append(
            (
                squared_error_sum / max(weight_sum, np.finfo(np.float64).tiny),
                float(ridge_lambda),
            )
        )
    best_score = min(score for score, _ in candidate_scores)
    tied = [
        ridge
        for score, ridge in candidate_scores
        if score <= best_score + 1.0e-12
    ]
    selected_lambda = max(tied)
    coefficients = _fit_nonnegative_ridge(
        features,
        targets,
        weights,
        ridge_lambda=selected_lambda,
    )
    if not np.any(coefficients > 0.0):
        raise RuntimeError(
            "Designated training produced a zero-scatter model; production "
            "approval is forbidden."
        )
    provenance = dict(training_manifest)
    provenance.update(
        {
            "fit_sample_count": int(features.shape[0]),
            "loso_scene_ids": [
                str(value) for value in unique_scenes.tolist()
            ],
            "candidate_validation_scores": {
                format(ridge, ".12g"): float(score)
                for score, ridge in candidate_scores
            },
            "selected_validation_score": float(best_score),
            "selected_ridge_lambda": float(selected_lambda),
            "selection_completed": True,
        }
    )
    direct_coefficients: tuple[float, ...] = ()
    direct_ridge: float | None = None
    direct_manifest: dict[str, object] | None = None
    if direct_log_ratio_n is not None:
        direct_targets = np.asarray(
            direct_log_ratio_n,
            dtype=np.float64,
        ).reshape(-1)
        if (
            direct_targets.shape != targets.shape
            or np.any(~np.isfinite(direct_targets))
        ):
            raise ValueError("Direct transport log-ratio targets are invalid.")
        direct_scores: list[tuple[float, float]] = []
        for ridge_lambda in ADDITIVE_SCATTER_RIDGE_LAMBDA_GRID:
            squared_error_sum = 0.0
            weight_sum = 0.0
            for held_out_scene in unique_scenes:
                validation_mask = scene_ids == held_out_scene
                training_mask = ~validation_mask
                direct_fold = _fit_signed_ridge(
                    features[training_mask],
                    direct_targets[training_mask],
                    weights[training_mask],
                    ridge_lambda=float(ridge_lambda),
                )
                prediction = np.clip(
                    features[validation_mask] @ direct_fold,
                    -DIRECT_TRANSPORT_MAXIMUM_ABS_LOG_CORRECTION,
                    DIRECT_TRANSPORT_MAXIMUM_ABS_LOG_CORRECTION,
                )
                errors = np.square(
                    prediction - direct_targets[validation_mask]
                )
                squared_error_sum += float(
                    np.sum(weights[validation_mask] * errors)
                )
                weight_sum += float(np.sum(weights[validation_mask]))
            direct_scores.append(
                (
                    squared_error_sum
                    / max(weight_sum, np.finfo(np.float64).tiny),
                    float(ridge_lambda),
                )
            )
        direct_best = min(score for score, _ in direct_scores)
        direct_tied = [
            ridge
            for score, ridge in direct_scores
            if score <= direct_best + 1.0e-12
        ]
        direct_ridge = max(direct_tied)
        fitted_direct = _fit_signed_ridge(
            features,
            direct_targets,
            weights,
            ridge_lambda=direct_ridge,
        )
        direct_coefficients = tuple(float(value) for value in fitted_direct)
        direct_manifest = {
            "schema_version": 1,
            "base_training_manifest_sha256": _canonical_json_sha256(
                dict(training_manifest)
            ),
            "target_semantics": DIRECT_TRANSPORT_TARGET_SEMANTICS,
            "fit_sample_count": int(features.shape[0]),
            "loso_scene_ids": [
                str(value) for value in unique_scenes.tolist()
            ],
            "candidate_validation_scores": {
                format(ridge, ".12g"): float(score)
                for score, ridge in direct_scores
            },
            "selected_validation_score": float(direct_best),
            "selected_ridge_lambda": float(direct_ridge),
            "selection_completed": True,
        }
    return AdditiveNoncollidedTransportResponse(
        coefficients=tuple(float(value) for value in coefficients),
        ridge_lambda=float(selected_lambda),
        training_manifest=provenance,
        direct_log_coefficients=direct_coefficients,
        direct_ridge_lambda=direct_ridge,
        direct_training_manifest=direct_manifest,
        feature_basis_semantics=feature_basis_semantics,
    )
