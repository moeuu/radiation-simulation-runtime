"""Physical additive noncollided-response model for full-spectrum inference.

The model predicts detector-entry scatter before detector-response marking,
background injection, and electronics dead time.  It is deliberately additive:

``total = uncollided + unattenuated_geometric * scatter_fraction``.

The single supported basis uses XCOM dry-air attenuation and detector-cone
Compton transport. It has no learned coefficients or historical basis selector.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

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
ADDITIVE_SCATTER_INCIDENT_LABEL_SEMANTICS = (
    "pre_dead_time_raw_incident_gamma_source_line_entry_class"
)
ADDITIVE_SCATTER_TARGET_SEMANTICS = (
    "interacted_plus_secondary_pre_dead_time_incident_counts_divided_by_"
    "unattenuated_geometric_line_counts"
)
PHYSICAL_SCATTER_BASIS_SEMANTICS = (
    "detector_cone_path_quadrature_single_compton_air_xcom_v2"
)
PHYSICS_ONLY_TRANSPORT_RESPONSE_ID = "physics_only_detector_cone_transport_response_v2"
_CONE_QUADRATURE_NODES, _CONE_QUADRATURE_WEIGHTS = np.polynomial.legendre.leggauss(16)
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
        and all(character in "0123456789abcdef" for character in value)
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
        * (2.0 * (1.0 + alpha) / (1.0 + 2.0 * alpha) - log_term / alpha)
        + log_term / (2.0 * alpha)
        - (1.0 + 3.0 * alpha) / np.square(1.0 + 2.0 * alpha)
    )
    return 2.0 * np.pi * CLASSICAL_ELECTRON_RADIUS_CM**2 * np.maximum(bracket, 0.0)


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
            raise KeyError(f"No atomic-number contract exists for element {symbol!r}.")
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
    nodes = _CONE_QUADRATURE_NODES.reshape((1,) * energy.ndim + (-1,))
    weights = _CONE_QUADRATURE_WEIGHTS.reshape((1,) * energy.ndim + (-1,))
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
            2.0 * (1.0 + alpha_total) / (1.0 + 2.0 * alpha_total)
            - log_term / alpha_total
        )
        + log_term / (2.0 * alpha_total)
        - (1.0 + 3.0 * alpha_total) / torch.square(1.0 + 2.0 * alpha_total)
    )
    denominator = (
        2.0 * np.pi * CLASSICAL_ELECTRON_RADIUS_CM**2 * torch.clamp(bracket, min=0.0)
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
    semantics: str = PHYSICAL_SCATTER_BASIS_SEMANTICS,
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
    if semantics != PHYSICAL_SCATTER_BASIS_SEMANTICS:
        raise ValueError(
            "Scatter transport requires the current XCOM-air detector-cone basis."
        )
    obstacle_fraction = np.clip(obstacle_fraction, 0.0, 1.0)
    air_compton_tau = (
        distance
        * 100.0
        * G4_AIR_REFERENCE_DENSITY_G_CM3
        * AVOGADRO_CONSTANT_MOL_INV
        * AIR_EFFECTIVE_Z_OVER_A
        * klein_nishina_total_cross_section_cm2(energy)
    )
    air_survival_tau = distance * 100.0 * dry_air_total_linear_attenuation_numpy(energy)
    survival = np.exp(-(fe_tau + pb_tau + obstacle_tau + air_survival_tau))
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
    if (
        detector_radius_m is None
        or fe_scatter_distance_m is None
        or pb_scatter_distance_m is None
    ):
        raise ValueError(
            "Detector-cone scatter requires detector and shield geometry distances."
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
            raise ValueError("Obstacle single-scatter probability is invalid.")
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
    semantics: str = PHYSICAL_SCATTER_BASIS_SEMANTICS,
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
                > obstacle_tau * (1.0 + 64.0 * torch.finfo(fe_tau.dtype).eps)
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
            * (2.0 * (1.0 + alpha) / (1.0 + 2.0 * alpha) - log_term / alpha)
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
            float(density_g_cm3) * AVOGADRO_CONSTANT_MOL_INV * float(z_over_a) * sigma
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
    if semantics != PHYSICAL_SCATTER_BASIS_SEMANTICS:
        raise ValueError(
            "Scatter transport requires the current XCOM-air detector-cone basis."
        )
    obstacle_fraction = torch.clamp(obstacle_fraction, min=0.0, max=1.0)
    air_compton_mu = _material_fraction(
        torch.ones_like(compact_energy),
        energy_value=compact_energy,
        density_g_cm3=G4_AIR_REFERENCE_DENSITY_G_CM3,
        z_over_a=AIR_EFFECTIVE_Z_OVER_A,
    )
    air_compton_tau = distance * 100.0 * air_compton_mu
    air_survival_tau = distance * 100.0 * dry_air_total_linear_attenuation_torch(energy)
    survival = torch.exp(-(fe_tau + pb_tau + obstacle_tau + air_survival_tau))
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
    if (
        detector_radius_m is None
        or fe_scatter_distance_m is None
        or pb_scatter_distance_m is None
    ):
        raise ValueError(
            "Detector-cone scatter requires detector and shield geometry distances."
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
    variable_distance_acceptance = klein_nishina_forward_cone_fraction_torch(
        energy,
        detector_radius_m=radius,
        scatter_distance_m=obstacle_distance,
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
        if obstacle_probability.shape != obstacle_tau.shape or bool(
            torch.any(invalid_obstacle_probability)
        ):
            raise ValueError("Obstacle single-scatter probability is invalid.")
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
    transport_feature_order: Sequence[str],
    line_identity: Sequence[Mapping[str, object]],
    target_semantics: str,
    detector_radius_m: float | None = None,
    fe_scatter_distance_m: float | None = None,
    pb_scatter_distance_m: float | None = None,
) -> NDArray[np.float64]:
    """Reconstruct the current physical basis from stored ray geometry.

    Acceptance artifacts store the seven-feature basis
    together with lossless ray features, including the material-integrated
    obstacle Compton optical depth and detector-impact phase fractions.
    """
    basis = np.asarray(stored_basis, dtype=np.float64)
    features = np.asarray(transport_features, dtype=np.float64)
    line_rows = tuple(line_identity)
    feature_order = tuple(transport_feature_order)
    base_order = (
        "tau_fe",
        "tau_pb",
        "tau_obstacle",
        "tau_obstacle_compton",
        "distance_m",
    )
    phase_order = feature_order[len(base_order) :]
    expected_phase_order = tuple(
        f"uncollided_impact_fraction_{index}" for index in range(len(phase_order))
    )
    if (
        target_semantics != PHYSICAL_SCATTER_BASIS_SEMANTICS
        or feature_order[: len(base_order)] != base_order
        or phase_order != expected_phase_order
        or features.ndim < 2
        or features.shape[-1] != len(feature_order)
        or basis.shape != features.shape[:-1] + (len(ADDITIVE_SCATTER_FEATURE_ORDER),)
        or features.shape[-2] != len(line_rows)
        or np.any(~np.isfinite(features))
        or np.any(features < 0.0)
        or np.any(~np.isfinite(basis))
        or np.any(basis < 0.0)
    ):
        raise ValueError("Stored scatter geometry is invalid.")
    if phase_order:
        phase_values = features[..., len(base_order) :]
        phase_sum = np.sum(phase_values, axis=-1)
        tolerance = 1.0e-10
        if np.any(phase_values > 1.0) or np.any(
            (np.abs(phase_sum) > tolerance) & (np.abs(phase_sum - 1.0) > tolerance)
        ):
            raise ValueError(
                "Stored detector-impact fractions must sum to zero or one."
            )
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
    obstacle_compton_tau = features[..., 3]
    if np.any(obstacle_compton_tau > obstacle_tau + 1.0e-12):
        raise ValueError("Stored obstacle Compton depth exceeds total optical depth.")
    return physical_scatter_basis_numpy(
        tau_fe=features[..., 0],
        tau_pb=features[..., 1],
        tau_obstacle=obstacle_tau,
        tau_obstacle_compton=obstacle_compton_tau,
        distance_m=features[..., 4],
        energy_keV=energies.reshape(line_shape),
        mu_fe_cm_inv=mu_fe.reshape(line_shape),
        mu_pb_cm_inv=mu_pb.reshape(line_shape),
        semantics=target_semantics,
        detector_radius_m=detector_radius_m,
        fe_scatter_distance_m=fe_scatter_distance_m,
        pb_scatter_distance_m=pb_scatter_distance_m,
    )


@dataclass(frozen=True)
class PhysicsOnlyNoncollidedTransportResponse:
    """Evaluate direct plus single-Compton transport without scene fitting."""

    detector_radius_m: float
    fe_scatter_distance_m: float
    pb_scatter_distance_m: float
    feature_basis_semantics: str = PHYSICAL_SCATTER_BASIS_SEMANTICS

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
        if self.feature_basis_semantics != PHYSICAL_SCATTER_BASIS_SEMANTICS:
            raise ValueError(
                "Physics-only transport requires the current XCOM-air detector-cone basis."
            )
        object.__setattr__(
            self,
            "_contract_hash_sha256",
            _canonical_json_sha256(self._contract_payload()),
        )

    def _contract_payload(self) -> dict[str, object]:
        """Return the universal physical response contract fields."""
        payload: dict[str, object] = {
            "model": PHYSICS_ONLY_TRANSPORT_RESPONSE_ID,
            "feature_order": list(ADDITIVE_SCATTER_FEATURE_ORDER),
            "feature_basis_semantics": self.feature_basis_semantics,
            "mean_model": (
                "beer_lambert_uncollided_with_nist_xcom_dry_air_plus_"
                "detector_cone_single_compton"
            ),
            "fit_family": "none_physics_only",
            "detector_radius_m": float(self.detector_radius_m),
            "fe_scatter_distance_m": float(self.fe_scatter_distance_m),
            "pb_scatter_distance_m": float(self.pb_scatter_distance_m),
            "angular_quadrature_order": int(_CONE_QUADRATURE_NODES.size),
            "material_path_quadrature_order": 2,
            "higher_order_mean": "excluded_positive_nuisance_owned_by_likelihood",
            "obstacle_material_contract_id": OBSTACLE_MATERIAL_CONTRACT_ID,
            "obstacle_material_contract_sha256": (OBSTACLE_MATERIAL_CONTRACT_SHA256),
            "transport_physics_table_contract_id": (
                TRANSPORT_PHYSICS_TABLE_CONTRACT_ID
            ),
            "transport_physics_table_contract_sha256": (
                TRANSPORT_PHYSICS_TABLE_CONTRACT_SHA256
            ),
        }
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
    def physics_only(self) -> bool:
        """Return whether this response excludes scene-trained coefficients."""
        return True

    def to_payload(self) -> dict[str, object]:
        """Return an authenticated JSON-compatible physics payload."""
        return {
            "schema_version": 2,
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
        if (model_id, schema_version) != (
            PHYSICS_ONLY_TRANSPORT_RESPONSE_ID,
            2,
        ) or not _is_lower_sha256(payload.get("contract_hash_sha256")):
            raise ValueError(
                "Runtime requires the XCOM-air physics-only transport "
                "response contract."
            )
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
            basis.shape != uncollided.shape + (len(ADDITIVE_SCATTER_FEATURE_ORDER),)
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
            or basis.shape != uncollided.shape + (len(ADDITIVE_SCATTER_FEATURE_ORDER),)
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
        return uncollided + unattenuated * self.scatter_fraction_numpy(feature_basis)

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
        return uncollided + unattenuated * self.scatter_fraction_torch(feature_basis)
