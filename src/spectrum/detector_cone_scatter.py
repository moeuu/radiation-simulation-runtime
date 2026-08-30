"""Isotope-independent joint Compton-energy and detector-impact response."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from threading import RLock
from typing import Final, Sequence

import numpy as np
from numpy.typing import NDArray

from spectrum.additive_scatter import ELECTRON_REST_ENERGY_KEV
from spectrum.detector_green_operator import DetectorGreenOperator


DETECTOR_CONE_SCATTER_RESPONSE_ID: Final = (
    "detector_cone_joint_energy_impact_single_compton_v1"
)
DETECTOR_CONE_SCATTER_QUADRATURE_ORDER: Final = 24
DETECTOR_CONE_SCATTER_LOG_DISTANCE_NODE_COUNT: Final = 17
DETECTOR_CONE_SCATTER_MAXIMUM_DISTANCE_M: Final = 100.0
_DETECTOR_CONE_SCATTER_CACHE_MAX_ENTRIES: Final = 16
_DETECTOR_CONE_SCATTER_CACHE: OrderedDict[
    tuple[object, ...],
    "DetectorConeScatterGrid",
] = OrderedDict()
_DETECTOR_CONE_SCATTER_CACHE_LOCK = RLock()


def _canonical_sha256(payload: object) -> str:
    """Return one canonical JSON SHA-256 digest."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class DetectorConeScatterGrid:
    """Store a deterministic distance-conditioned marked scatter operator."""

    distance_nodes_m: NDArray[np.float64]
    marked_response_dlb: NDArray[np.float64]
    effective_histories_dl: NDArray[np.float64]
    contract_hash_sha256: str

    def __post_init__(self) -> None:
        """Validate and freeze the response grid arrays."""
        distances = np.ascontiguousarray(self.distance_nodes_m, dtype=np.float64)
        response = np.ascontiguousarray(self.marked_response_dlb, dtype=np.float64)
        histories = np.ascontiguousarray(self.effective_histories_dl, dtype=np.float64)
        if (
            distances.ndim != 1
            or distances.size < 3
            or np.any(~np.isfinite(distances))
            or np.any(distances <= 0.0)
            or np.any(np.diff(distances) <= 0.0)
            or response.ndim != 3
            or response.shape[0] != distances.size
            or histories.shape != response.shape[:2]
            or np.any(~np.isfinite(response))
            or np.any(response < 0.0)
            or np.any(~np.isfinite(histories))
            or np.any(histories <= 0.0)
            or not isinstance(self.contract_hash_sha256, str)
            or len(self.contract_hash_sha256) != 64
        ):
            raise ValueError("Detector-cone scatter response grid is invalid.")
        distances.setflags(write=False)
        response.setflags(write=False)
        histories.setflags(write=False)
        object.__setattr__(self, "distance_nodes_m", distances)
        object.__setattr__(self, "marked_response_dlb", response)
        object.__setattr__(self, "effective_histories_dl", histories)

    def contract_payload(self) -> dict[str, object]:
        """Return the compact physical construction contract."""
        return {
            "response": DETECTOR_CONE_SCATTER_RESPONSE_ID,
            "quadrature_order": DETECTOR_CONE_SCATTER_QUADRATURE_ORDER,
            "distance_domain_m": [
                float(self.distance_nodes_m[0]),
                float(self.distance_nodes_m[-1]),
            ],
            "distance_nodes_m": self.distance_nodes_m.tolist(),
            "distance_interpolation": "piecewise_linear_in_log_distance",
            "single_scatter_conditioning": (
                "klein_nishina_energy_and_detector_impact_jointly_conditioned_"
                "on_housing_intersection"
            ),
            "higher_order_scatter_mean": "excluded",
            "contract_hash_sha256": self.contract_hash_sha256,
        }


def detector_cone_scatter_distance_nodes(
    *,
    detector_radius_m: float,
    fixed_scatter_distances_m: Sequence[float],
) -> NDArray[np.float64]:
    """Return physics-defined log-distance nodes including shield radii."""
    radius = float(detector_radius_m)
    fixed = np.asarray(tuple(fixed_scatter_distances_m), dtype=np.float64)
    if (
        not np.isfinite(radius)
        or radius <= 0.0
        or radius >= DETECTOR_CONE_SCATTER_MAXIMUM_DISTANCE_M
        or fixed.ndim != 1
        or fixed.size == 0
        or np.any(~np.isfinite(fixed))
        or np.any(fixed < radius)
        or np.any(fixed > DETECTOR_CONE_SCATTER_MAXIMUM_DISTANCE_M)
    ):
        raise ValueError("Detector-cone scatter distances are outside the domain.")
    logarithmic = np.geomspace(
        radius,
        DETECTOR_CONE_SCATTER_MAXIMUM_DISTANCE_M,
        DETECTOR_CONE_SCATTER_LOG_DISTANCE_NODE_COUNT,
        dtype=np.float64,
    )
    result = np.unique(np.concatenate((logarithmic, fixed)))
    result.setflags(write=False)
    return result


def build_detector_cone_scatter_grid(
    *,
    operator: DetectorGreenOperator,
    incident_energies_keV: Sequence[float],
    source_reference_efficiencies: Sequence[float],
    fixed_scatter_distances_m: Sequence[float],
) -> DetectorConeScatterGrid:
    """Build the batched joint scatter/impact detector response.

    The additive transport kernel already supplies the probability that one
    Compton interaction sends a photon into the detector cone.  This routine
    therefore constructs the conditional pulse-height distribution given that
    cone intersection, retaining the exact correlation between Compton energy
    loss and detector impact stratum.
    """
    energies = np.asarray(tuple(incident_energies_keV), dtype=np.float64)
    reference = np.asarray(tuple(source_reference_efficiencies), dtype=np.float64)
    radius = float(operator.detector_target_radius_m)
    if (
        energies.ndim != 1
        or energies.size == 0
        or reference.shape != energies.shape
        or np.any(~np.isfinite(energies))
        or np.any(energies <= 0.0)
        or np.any(~np.isfinite(reference))
        or np.any(reference <= 0.0)
        or float(np.min(energies)) < operator.input_energy_domain_keV[0]
        or float(np.max(energies)) > operator.input_energy_domain_keV[1]
    ):
        raise ValueError("Scatter response lines or reference efficiencies are invalid.")
    cache_key = (
        str(operator.contract_hash_sha256),
        str(operator.binary_sha256),
        tuple(float(value).hex() for value in energies),
        tuple(float(value).hex() for value in reference),
        tuple(float(value).hex() for value in fixed_scatter_distances_m),
    )
    with _DETECTOR_CONE_SCATTER_CACHE_LOCK:
        cached = _DETECTOR_CONE_SCATTER_CACHE.get(cache_key)
        if cached is not None:
            _DETECTOR_CONE_SCATTER_CACHE.move_to_end(cache_key)
            return cached
    distances = detector_cone_scatter_distance_nodes(
        detector_radius_m=radius,
        fixed_scatter_distances_m=fixed_scatter_distances_m,
    )
    impact_edges = operator.impact_parameter_edges_fraction
    phase_count = int(impact_edges.size - 1)
    quadrature_nodes, quadrature_weights = np.polynomial.legendre.leggauss(
        DETECTOR_CONE_SCATTER_QUADRATURE_ORDER
    )
    ratio_edges_dc = (
        radius
        * impact_edges[np.newaxis, :]
        / distances[:, np.newaxis]
    )
    cosine_edges_dc = np.sqrt(np.maximum(1.0 - np.square(ratio_edges_dc), 0.0))
    cosine_lower_dc = cosine_edges_dc[:, 1:]
    cosine_upper_dc = cosine_edges_dc[:, :-1]
    midpoint_dc = 0.5 * (cosine_lower_dc + cosine_upper_dc)
    half_width_dc = 0.5 * (cosine_upper_dc - cosine_lower_dc)
    cosine_dcq = (
        midpoint_dc[..., np.newaxis]
        + half_width_dc[..., np.newaxis]
        * quadrature_nodes[np.newaxis, np.newaxis, :]
    )
    alpha_l = energies / ELECTRON_REST_ENERGY_KEV
    energy_ratio_dlcq = 1.0 / (
        1.0
        + alpha_l[np.newaxis, :, np.newaxis, np.newaxis]
        * (1.0 - cosine_dcq[:, np.newaxis, :, :])
    )
    scattered_energy_dlcq = (
        energies[np.newaxis, :, np.newaxis, np.newaxis] * energy_ratio_dlcq
    )
    angular_dlcq = np.square(energy_ratio_dlcq) * (
        energy_ratio_dlcq
        + 1.0 / np.maximum(energy_ratio_dlcq, np.finfo(np.float64).tiny)
        - (1.0 - np.square(cosine_dcq))[:, np.newaxis, :, :]
    )
    raw_weight_dlcq = (
        half_width_dc[:, np.newaxis, :, np.newaxis]
        * quadrature_weights[np.newaxis, np.newaxis, np.newaxis, :]
        * angular_dlcq
    )
    normalizer_dl = np.sum(raw_weight_dlcq, axis=(-2, -1))
    if np.any(normalizer_dl <= 0.0) or np.any(~np.isfinite(normalizer_dl)):
        raise RuntimeError("Detector-cone scatter quadrature has zero mass.")
    normalized_weight_dlcq = raw_weight_dlcq / normalizer_dl[..., np.newaxis, np.newaxis]
    phase_dlcq = np.broadcast_to(
        np.arange(phase_count, dtype=np.int64)[np.newaxis, np.newaxis, :, np.newaxis],
        scattered_energy_dlcq.shape,
    )
    flat_response, flat_concentration = (
        operator.absolute_response_for_energy_phase_pairs(
            scattered_energy_dlcq.reshape(-1),
            phase_dlcq.reshape(-1),
        )
    )
    response_dlcqb = flat_response.reshape(
        scattered_energy_dlcq.shape + (operator.output_bin_count,)
    )
    concentration_dlcq = flat_concentration.reshape(scattered_energy_dlcq.shape)
    absolute_response_dlb = np.einsum(
        "dlcq,dlcqb->dlb",
        normalized_weight_dlcq,
        response_dlcqb,
        # This reduction defines an authenticated finite-MC concentration.
        # Keep a fixed NumPy contraction order across host BLAS kernels.
        optimize=False,
    )
    no_pulse_dlcq = 1.0 - np.sum(response_dlcqb, axis=-1)
    component_trace_dlcq = (
        1.0
        - np.sum(np.square(response_dlcqb), axis=-1)
        - np.square(no_pulse_dlcq)
    ) / (concentration_dlcq + 1.0)
    mixture_trace_dl = np.sum(
        np.square(normalized_weight_dlcq) * component_trace_dlcq,
        axis=(-2, -1),
    )
    mixture_no_pulse_dl = 1.0 - np.sum(absolute_response_dlb, axis=-1)
    mixture_numerator_dl = (
        1.0
        - np.sum(np.square(absolute_response_dlb), axis=-1)
        - np.square(mixture_no_pulse_dl)
    )
    effective_histories_dl = (
        np.divide(
            mixture_numerator_dl,
            mixture_trace_dl,
            out=np.full_like(mixture_numerator_dl, 1.0e15),
            where=mixture_trace_dl > np.finfo(np.float64).tiny,
        )
        - 1.0
    )
    effective_histories_dl = np.maximum(effective_histories_dl, 1.0)
    marked_response_dlb = (
        absolute_response_dlb / reference[np.newaxis, :, np.newaxis]
    )
    contract_payload = {
        "response": DETECTOR_CONE_SCATTER_RESPONSE_ID,
        "quadrature_order": DETECTOR_CONE_SCATTER_QUADRATURE_ORDER,
        "distance_nodes_m": distances.tolist(),
        "distance_interpolation": "piecewise_linear_in_log_distance",
        "detector_green_operator_contract_sha256": operator.contract_hash_sha256,
        "detector_green_operator_binary_sha256": operator.binary_sha256,
        "incident_energies_keV": energies.tolist(),
        # Reference efficiencies are derived reductions over Green columns.
        # Canonicalize only their contract representation so BLAS scheduling
        # cannot change artifact identity; inference retains full float64.
        "source_reference_efficiencies": np.round(
            reference,
            decimals=12,
        ).tolist(),
    }
    grid = DetectorConeScatterGrid(
        distance_nodes_m=distances,
        marked_response_dlb=marked_response_dlb,
        effective_histories_dl=effective_histories_dl,
        contract_hash_sha256=_canonical_sha256(contract_payload),
    )
    with _DETECTOR_CONE_SCATTER_CACHE_LOCK:
        existing = _DETECTOR_CONE_SCATTER_CACHE.get(cache_key)
        if existing is not None:
            _DETECTOR_CONE_SCATTER_CACHE.move_to_end(cache_key)
            return existing
        _DETECTOR_CONE_SCATTER_CACHE[cache_key] = grid
        if (
            len(_DETECTOR_CONE_SCATTER_CACHE)
            > _DETECTOR_CONE_SCATTER_CACHE_MAX_ENTRIES
        ):
            _DETECTOR_CONE_SCATTER_CACHE.popitem(last=False)
    return grid


__all__ = [
    "DETECTOR_CONE_SCATTER_MAXIMUM_DISTANCE_M",
    "DETECTOR_CONE_SCATTER_RESPONSE_ID",
    "DetectorConeScatterGrid",
    "build_detector_cone_scatter_grid",
    "detector_cone_scatter_distance_nodes",
]
