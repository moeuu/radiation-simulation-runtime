"""Geometry-conditioned full-spectrum model for the pure particle filter.

The model keeps source contributions separate until their physical direct and
scattered incident-gamma spectra have been formed.  Detector-response marking
is applied once, background is added once, and nonparalyzable detector dead
time is represented by a renewal total-count law with conditional multinomial
energy marks.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import time
from types import MappingProxyType
from collections.abc import Mapping, Sequence
from threading import RLock

import numpy as np
from numpy.typing import NDArray
from scipy import special, stats

from measurement.geometry_family import (
    GEOMETRY_FAMILY_APPLICABILITY_SHA256,
    validate_geometry_family_descriptor,
)
from measurement.source_boundary import (
    SURFACE_EMISSION_EPSILON_M,
    surface_emission_policy_sha256,
)
from measurement.shielding import (
    DEFAULT_FE_SHIELD_INNER_RADIUS_CM,
    DEFAULT_FE_SHIELD_THICKNESS_CM,
    DEFAULT_PB_SHIELD_INNER_RADIUS_CM,
    DEFAULT_PB_SHIELD_THICKNESS_CM,
    SHIELD_POSE_CONTRACT_ID,
    SHIELD_POSE_CONTRACT_SHA256,
    line_resolved_shield_mu_by_isotope,
)
from runtime.experiment_profiles import (
    STANDARD_ACQUISITION_LIVE_TIME_S,
    STANDARD_OBSTACLE_MATERIAL,
    STANDARD_ROOM_BOUNDARY_THICKNESS_M,
)
from runtime.contracts import FULL_SPECTRUM_MODEL_SCHEMA_VERSION
from runtime.forward_model_manifest import resolve_file_backed_model_asset
from spectrum.additive_scatter import (
    ADDITIVE_SCATTER_INCIDENT_LABEL_SEMANTICS,
    DETECTOR_CONE_AIR_XCOM_SINGLE_SCATTER_BASIS_SEMANTICS,
    PHYSICS_ONLY_TRANSPORT_RESPONSE_ID,
    AdditiveNoncollidedTransportResponse,
    PhysicsOnlyNoncollidedTransportResponse,
    klein_nishina_forward_cone_fraction_numpy,
    klein_nishina_forward_cone_fraction_torch,
)
from spectrum.detector_cone_scatter import (
    DETECTOR_CONE_SCATTER_MAXIMUM_DISTANCE_M,
    DETECTOR_CONE_SCATTER_RESPONSE_ID,
    build_detector_cone_scatter_grid,
)
from spectrum.physics_contracts import (
    OBSTACLE_MATERIAL_CONTRACT_ID,
    OBSTACLE_MATERIAL_CONTRACT_SHA256,
    TRANSPORT_PHYSICS_TABLE_CONTRACT_ID,
    TRANSPORT_PHYSICS_TABLE_CONTRACT_SHA256,
)
from spectrum.library import default_library
from spectrum.detector_green_operator import (
    DETECTOR_GREEN_COINCIDENCE_SEMANTICS,
    DETECTOR_GREEN_OPERATOR_ID,
    DETECTOR_GREEN_SAMPLING_MODE,
    DetectorGreenOperator,
)
from spectrum.response_matrix import (
    NATIVE_GEANT4_BIN_COUNT,
    NATIVE_GEANT4_BIN_WIDTH_KEV,
    native_geant4_background_shape,
)
from spectrum.detector_green_validation import (
    detector_green_validation_manifest_sha256,
    validate_detector_green_validation_manifest,
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
DETECTOR_IMPACT_PHASE_COUNT = 8
DETECTOR_IMPACT_FEATURE_ORDER = tuple(
    f"uncollided_impact_fraction_{index}"
    for index in range(DETECTOR_IMPACT_PHASE_COUNT)
)
TRANSPORT_PHYSICS_FEATURE_ORDER = (
    "tau_fe",
    "tau_pb",
    "tau_obstacle",
    "tau_obstacle_compton",
    "distance_m",
)
TRANSPORT_FEATURE_ORDER = (
    *TRANSPORT_PHYSICS_FEATURE_ORDER,
    *DETECTOR_IMPACT_FEATURE_ORDER,
)
TRANSPORT_DISTANCE_FEATURE_INDEX = TRANSPORT_PHYSICS_FEATURE_ORDER.index("distance_m")
TRANSPORT_IMPACT_FEATURE_OFFSET = len(TRANSPORT_PHYSICS_FEATURE_ORDER)
CANONICAL_DETECTOR_GREEN_OPERATOR_MANIFEST = (
    Path(__file__).resolve().parent
    / "assets"
    / "detector_green_operator"
    / "manifest.json"
)
_DETECTOR_GREEN_MODEL_RESPONSE_CACHE_MAX_BYTES = 32 * 1024 * 1024
_DETECTOR_GREEN_MODEL_RESPONSE_CACHE: OrderedDict[
    tuple[object, ...],
    tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        int,
    ],
] = OrderedDict()
_DETECTOR_GREEN_MODEL_RESPONSE_CACHE_BYTES = 0
_DETECTOR_GREEN_MODEL_RESPONSE_CACHE_LOCK = RLock()
DEFAULT_DETECTOR_GREEN_OPERATOR_MANIFEST = CANONICAL_DETECTOR_GREEN_OPERATOR_MANIFEST
BIRTH_PROPOSAL_WORKING_SET_BYTES = 512 * 1024 * 1024
CROSS_LIKELIHOOD_ACTION_CHUNK_SIZE = 1
CROSS_LIKELIHOOD_SAMPLE_CHUNK_SIZE = 64
CROSS_LIKELIHOOD_STATE_CHUNK_SIZE = 256
CROSS_LIKELIHOOD_STATE_AUTOTUNE_MAX_CHUNK_SIZE = 1024
CROSS_LIKELIHOOD_BIN_CHUNK_SIZE = 128
SUBSET_LIKELIHOOD_VIEW_CHUNK_SIZE = 8
CONTINUUM_NUISANCE_BAND_WIDTH_KEV = 50.0
CANONICAL_LOG_GAMMA_LANCZOS_G = 7.0
CANONICAL_LOG_GAMMA_COEFFICIENTS = (
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
)
CANONICAL_LOG_GAMMA_HALF_LOG_TWO_PI = 0.9189385332046727
MARK_EXACT_CONCENTRATION = 1.0e15
RENEWAL_LOG_GAMMA_MAX_ITERATIONS = 2_048
RENEWAL_GAMMA_INTERVAL_QUADRATURE_ORDER = 32
(
    _RENEWAL_GAMMA_INTERVAL_NODES,
    _RENEWAL_GAMMA_INTERVAL_WEIGHTS,
) = np.polynomial.legendre.leggauss(RENEWAL_GAMMA_INTERVAL_QUADRATURE_ORDER)
# Historical scene-fit tooling remains benchmark-only and is not referenced by
# the schema-v7 production runtime or schema-v7 application-approval contract.
DESIGNATED_TRAINING_SCENE_SEEDS = (2026072701, 2026072702, 2026072703)
DESIGNATED_VALIDATION_SCENE_SEEDS = (
    3646699724,
    4620708915,
    5193545889,
    7235536511,
    7325752837,
)
FULL_SPECTRUM_ACCEPTANCE_EXPERIMENT_ID = "cs_co_full_spectrum_acceptance"
CATALOG_INDEPENDENT_APPROVAL_SCOPE = (
    "catalog_independent_detector_green_transport_v1"
)
TRANSFERRED_VALIDATION_SCHEMA_VERSION = 7
_TRANSFERRED_VALIDATION_FIELDS = frozenset(
    {
        "approval_scope",
        "approved_catalog_independent_contract_sha256",
        "application_validation_isotopes",
        "source_validation_manifest_sha256",
    }
)
ACCEPTANCE_ROOM_SIZE_XYZ = (10.0, 20.0, 10.0)
ACCEPTANCE_DETECTOR_POSE_XYZ = (1.0, 1.0, 0.5)
ACCEPTANCE_OBSTACLE_BLOCKED_FRACTION = 0.4
ACCEPTANCE_PASSAGE_WIDTH_M = 1.0
ACCEPTANCE_SURFACE_CHART_MAX_EDGE_M = 1.0
ACCEPTANCE_GEOMETRY_USE_GPU = False
ACCEPTANCE_GEOMETRY_DEVICE = "cpu"
ACCEPTANCE_GEOMETRY_DTYPE = "float64"
ACCEPTANCE_PERTURBATION_TANGENT_MAGNITUDES_M = (0.5, 1.0, 2.0, 4.0)
ACCEPTANCE_PERTURBATION_TANGENT_DIRECTIONS_UV = (
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (0.7071067811865476, 0.7071067811865476),
    (-0.7071067811865476, 0.7071067811865476),
    (0.7071067811865476, -0.7071067811865476),
    (-0.7071067811865476, -0.7071067811865476),
)
ACCEPTANCE_PERTURBATION_MINIMUM_DISPLACEMENT_M = 0.5
ACCEPTANCE_PERTURBATION_MINIMUM_LOG_RATE_SEPARATION = math.log(1.2)
ACCEPTANCE_PERTURBATION_MINIMUM_BEARING_ANGLE_RAD = math.radians(10.0)
RATE_SCALE_HALF_WIDTH_GRID = (0.0, 0.02, 0.05, 0.10, 0.20)
RATE_SCALE_MIXTURE_WEIGHTS = (0.25, 0.50, 0.25)
RATE_SCALE_UNIFORM_QUADRATURE_ORDER = 9
MARK_CONCENTRATION_GRID = (
    100.0,
    300.0,
    1_000.0,
    3_000.0,
    10_000.0,
    100_000.0,
)


def _detector_green_model_response_bundle(
    operator: DetectorGreenOperator,
    energy_axis_keV: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Return exact model-axis Green reductions from a bounded process cache."""
    global _DETECTOR_GREEN_MODEL_RESPONSE_CACHE_BYTES

    axis = np.ascontiguousarray(energy_axis_keV, dtype=np.float64)
    cache_key = (
        str(operator.contract_hash_sha256),
        str(operator.binary_sha256),
        axis.shape,
        axis.tobytes(order="C"),
    )
    with _DETECTOR_GREEN_MODEL_RESPONSE_CACHE_LOCK:
        cached = _DETECTOR_GREEN_MODEL_RESPONSE_CACHE.get(cache_key)
        if cached is not None:
            _DETECTOR_GREEN_MODEL_RESPONSE_CACHE.move_to_end(cache_key)
            response, conditional_concentration, absolute_concentration, _ = cached
            return (
                response.copy(),
                conditional_concentration.copy(),
                absolute_concentration.copy(),
            )
    response, absolute_concentration = operator.marginal_absolute_response_for_axis(
        axis
    )
    _, conditional_concentration = operator.marginal_response_for_axis(axis)
    stored_arrays = tuple(
        np.ascontiguousarray(value, dtype=np.float64).copy()
        for value in (
            response,
            conditional_concentration,
            absolute_concentration,
        )
    )
    entry_bytes = int(sum(value.nbytes for value in stored_arrays))
    if entry_bytes <= _DETECTOR_GREEN_MODEL_RESPONSE_CACHE_MAX_BYTES:
        for value in stored_arrays:
            value.setflags(write=False)
        with _DETECTOR_GREEN_MODEL_RESPONSE_CACHE_LOCK:
            existing = _DETECTOR_GREEN_MODEL_RESPONSE_CACHE.get(cache_key)
            if existing is None:
                while (
                    _DETECTOR_GREEN_MODEL_RESPONSE_CACHE
                    and _DETECTOR_GREEN_MODEL_RESPONSE_CACHE_BYTES + entry_bytes
                    > _DETECTOR_GREEN_MODEL_RESPONSE_CACHE_MAX_BYTES
                ):
                    _, removed = _DETECTOR_GREEN_MODEL_RESPONSE_CACHE.popitem(
                        last=False
                    )
                    _DETECTOR_GREEN_MODEL_RESPONSE_CACHE_BYTES -= int(removed[3])
                _DETECTOR_GREEN_MODEL_RESPONSE_CACHE[cache_key] = (
                    stored_arrays[0],
                    stored_arrays[1],
                    stored_arrays[2],
                    entry_bytes,
                )
                _DETECTOR_GREEN_MODEL_RESPONSE_CACHE_BYTES += entry_bytes
            else:
                _DETECTOR_GREEN_MODEL_RESPONSE_CACHE.move_to_end(cache_key)
    return (
        np.ascontiguousarray(response, dtype=np.float64),
        np.ascontiguousarray(conditional_concentration, dtype=np.float64),
        np.ascontiguousarray(absolute_concentration, dtype=np.float64),
    )


def canonical_log_gamma_numpy(value: object) -> NDArray[np.float64]:
    """Return positive-argument log-gamma with the canonical Lanczos path.

    NumPy and Torch likelihoods deliberately use the same coefficients,
    recurrence, operation order, and float64 arithmetic.  This avoids backend
    library differences accumulating across hundreds of spectrum bins while
    retaining a device-resident Torch implementation.
    """
    argument = np.asarray(value, dtype=np.float64)
    if np.any(~np.isfinite(argument)) or np.any(argument <= 0.0):
        raise ValueError("Canonical log-gamma requires finite positive arguments.")
    recurrent = argument < 0.5
    shifted = np.where(recurrent, argument + 1.0, argument)
    z = shifted - 1.0
    accumulator = np.full_like(
        z,
        CANONICAL_LOG_GAMMA_COEFFICIENTS[0],
        dtype=np.float64,
    )
    for index, coefficient in enumerate(
        CANONICAL_LOG_GAMMA_COEFFICIENTS[1:],
        start=1,
    ):
        accumulator = accumulator + float(coefficient) / (z + float(index))
    scale = z + CANONICAL_LOG_GAMMA_LANCZOS_G + 0.5
    result = (
        CANONICAL_LOG_GAMMA_HALF_LOG_TWO_PI
        + (z + 0.5) * np.log(scale)
        - scale
        + np.log(accumulator)
    )
    return np.asarray(
        np.where(recurrent, result - np.log(argument), result),
        dtype=np.float64,
    )


def canonical_log_gamma_torch(value: object) -> object:
    """Return Torch log-gamma through the canonical float64 Lanczos path."""
    import torch

    argument = torch.as_tensor(value)
    if argument.dtype != torch.float64:
        raise TypeError("Canonical Torch log-gamma requires float64 tensors.")
    invalid = torch.any(~torch.isfinite(argument)) | torch.any(argument <= 0.0)
    if bool(invalid.item()):
        raise ValueError("Canonical log-gamma requires finite positive arguments.")
    return _canonical_log_gamma_torch_unchecked(argument)


def _canonical_log_gamma_torch_unchecked(argument: object) -> object:
    """Evaluate the canonical Lanczos path after caller-owned validation."""
    import torch

    argument = torch.as_tensor(argument)
    if argument.dtype != torch.float64:
        raise TypeError("Canonical Torch log-gamma requires float64 tensors.")
    recurrent = argument < 0.5
    shifted = torch.where(recurrent, argument + 1.0, argument)
    z = shifted - 1.0
    accumulator = torch.full_like(
        z,
        float(CANONICAL_LOG_GAMMA_COEFFICIENTS[0]),
    )
    for index, coefficient in enumerate(
        CANONICAL_LOG_GAMMA_COEFFICIENTS[1:],
        start=1,
    ):
        accumulator = accumulator + float(coefficient) / (z + float(index))
    scale = z + CANONICAL_LOG_GAMMA_LANCZOS_G + 0.5
    result = (
        CANONICAL_LOG_GAMMA_HALF_LOG_TWO_PI
        + (z + 0.5) * torch.log(scale)
        - scale
        + torch.log(accumulator)
    )
    return torch.where(recurrent, result - torch.log(argument), result)


def _combined_dirichlet_concentration_numpy(
    first: object,
    second: object,
) -> NDArray[np.float64]:
    """Combine two independent compositional covariance concentrations."""
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    reciprocal = 1.0 / (left + 1.0) + 1.0 / (right + 1.0)
    return np.maximum(
        1.0 / np.maximum(reciprocal, np.finfo(np.float64).tiny) - 1.0,
        np.finfo(np.float64).tiny,
    )


def _combined_dirichlet_concentration_torch(
    first: object,
    second: object,
) -> object:
    """Return the Torch equivalent of combined compositional concentration."""
    import torch

    left = torch.as_tensor(first)
    right = torch.as_tensor(second, device=left.device, dtype=left.dtype)
    reciprocal = 1.0 / (left + 1.0) + 1.0 / (right + 1.0)
    return torch.clamp(
        1.0 / torch.clamp(reciprocal, min=torch.finfo(left.dtype).tiny) - 1.0,
        min=torch.finfo(left.dtype).tiny,
    )


def _build_mark_partition_tree(
    photopeak_mask_b: NDArray[np.bool_],
    continuum_group_mask_gb: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
]:
    """Build one immutable balanced tree over physical energy partitions.

    The root separates catalog-defined photopeak support from continuum.
    Connected photopeak intervals and fixed-width continuum bands are leaves.
    Construction runs once per model; runtime likelihood evaluation is fully
    batched over every tree node and leaf.
    """
    peak = np.asarray(photopeak_mask_b, dtype=np.bool_)
    continuum_groups = np.asarray(continuum_group_mask_gb, dtype=np.float64)
    peak_indices = np.flatnonzero(peak)
    peak_segments = np.split(
        peak_indices,
        np.flatnonzero(np.diff(peak_indices) > 1) + 1,
    )
    peak_groups = np.zeros((len(peak_segments), peak.size), dtype=np.float64)
    for index, segment in enumerate(peak_segments):
        peak_groups[index, segment] = 1.0
    leaf_groups = np.concatenate((peak_groups, continuum_groups), axis=0)
    peak_leaf_count = int(peak_groups.shape[0])
    if (
        peak_leaf_count <= 0
        or int(continuum_groups.shape[0]) <= 0
        or not np.array_equal(np.sum(leaf_groups, axis=0), np.ones(peak.size))
    ):
        raise RuntimeError("Physical mark leaves do not partition the energy axis.")

    nodes: list[dict[str, object]] = [
        {
            "left_child": None,
            "right_child": None,
            "left_leaves": tuple(range(peak_leaf_count)),
            "right_leaves": tuple(range(peak_leaf_count, len(leaf_groups))),
            "domain": -1,
            "depth": 0,
        }
    ]

    def _subtree(
        leaf_ids: tuple[int, ...],
        *,
        domain: int,
        depth: int,
    ) -> int:
        """Return a node index or negative encoded leaf for one subtree."""
        if len(leaf_ids) == 1:
            return -int(leaf_ids[0]) - 1
        split = len(leaf_ids) // 2
        left_ids = leaf_ids[:split]
        right_ids = leaf_ids[split:]
        node_index = len(nodes)
        nodes.append(
            {
                "left_child": None,
                "right_child": None,
                "left_leaves": left_ids,
                "right_leaves": right_ids,
                "domain": int(domain),
                "depth": int(depth),
            }
        )
        left_child = _subtree(left_ids, domain=domain, depth=depth + 1)
        right_child = _subtree(right_ids, domain=domain, depth=depth + 1)
        nodes[node_index]["left_child"] = left_child
        nodes[node_index]["right_child"] = right_child
        return node_index

    peak_ids = tuple(range(peak_leaf_count))
    continuum_ids = tuple(range(peak_leaf_count, len(leaf_groups)))
    nodes[0]["left_child"] = _subtree(peak_ids, domain=0, depth=1)
    nodes[0]["right_child"] = _subtree(continuum_ids, domain=1, depth=1)
    left_masks = np.asarray(
        [np.sum(leaf_groups[list(node["left_leaves"])], axis=0) for node in nodes],
        dtype=np.float64,
    )
    right_masks = np.asarray(
        [np.sum(leaf_groups[list(node["right_leaves"])], axis=0) for node in nodes],
        dtype=np.float64,
    )
    return (
        leaf_groups,
        left_masks,
        right_masks,
        np.asarray([int(node["domain"]) for node in nodes], dtype=np.int64),
        np.asarray([int(node["depth"]) for node in nodes], dtype=np.int64),
        np.asarray([int(node["left_child"]) for node in nodes], dtype=np.int64),
        np.asarray([int(node["right_child"]) for node in nodes], dtype=np.int64),
    )


@dataclass(frozen=True)
class LikelihoodDecomposition:
    """Hold exact per-view production likelihood contributions by role."""

    total_count_nv: NDArray[np.float64]
    background_mark_nv: NDArray[np.float64]
    source_mark_nv: NDArray[np.float64]

    @property
    def total_log_likelihood_n(self) -> NDArray[np.float64]:
        """Return the exact station likelihood reconstructed from all roles."""
        return np.sum(
            self.total_count_nv + self.background_mark_nv + self.source_mark_nv,
            axis=-1,
        )


@dataclass(frozen=True)
class PreparedTorchCrossObservation:
    """Hold immutable CUDA observation terms reused across exact states."""

    leading_shape: tuple[int, ...]
    observed_asvb: object
    observed_total_asv: object
    multinomial_constant_asv: object
    peak_observed_asvp: object
    continuum_observed_asvc: object
    peak_count_asv: object
    continuum_count_asv: object
    beta_binomial_constant_asv: object
    peak_multinomial_constant_asv: object
    continuum_group_observed_asvg: object
    continuum_group_constant_asv: object
    continuum_within_constant_asv: object
    mark_observed_projection_asvm: object | None
    mark_leaf_log_factorial_asvh: object | None

    def restored(self, values: object) -> object:
        """Restore the original leading action axes of a prepared tensor."""
        trailing_shape = tuple(int(value) for value in values.shape[1:])
        return values.reshape(self.leading_shape + trailing_shape)

    def block(
        self,
        *,
        action_start: int,
        action_stop: int,
        sample_start: int,
        sample_stop: int,
        view_start: int = 0,
        view_stop: int | None = None,
    ) -> "PreparedTorchCrossObservation":
        """Return one action/sample/view slab without recomputation."""

        resolved_view_stop = (
            int(self.observed_asvb.shape[2]) if view_stop is None else int(view_stop)
        )

        def _slice(values: object) -> object:
            """Slice a prepared tensor on action, sample, and view axes."""
            return values[
                int(action_start) : int(action_stop),
                int(sample_start) : int(sample_stop),
                int(view_start) : resolved_view_stop,
            ]

        def _slice_optional(values: object | None) -> object | None:
            """Slice one optional hierarchical observation tensor."""
            return None if values is None else _slice(values)

        return PreparedTorchCrossObservation(
            leading_shape=(int(action_stop) - int(action_start),),
            observed_asvb=_slice(self.observed_asvb),
            observed_total_asv=_slice(self.observed_total_asv),
            multinomial_constant_asv=_slice(self.multinomial_constant_asv),
            peak_observed_asvp=_slice(self.peak_observed_asvp),
            continuum_observed_asvc=_slice(self.continuum_observed_asvc),
            peak_count_asv=_slice(self.peak_count_asv),
            continuum_count_asv=_slice(self.continuum_count_asv),
            beta_binomial_constant_asv=_slice(self.beta_binomial_constant_asv),
            peak_multinomial_constant_asv=_slice(self.peak_multinomial_constant_asv),
            continuum_group_observed_asvg=_slice(self.continuum_group_observed_asvg),
            continuum_group_constant_asv=_slice(self.continuum_group_constant_asv),
            continuum_within_constant_asv=_slice(self.continuum_within_constant_asv),
            mark_observed_projection_asvm=_slice_optional(
                self.mark_observed_projection_asvm
            ),
            mark_leaf_log_factorial_asvh=_slice_optional(
                self.mark_leaf_log_factorial_asvh
            ),
        )


@dataclass(frozen=True)
class PreparedNumpySubsetCrossLikelihood:
    """Cache exact view-resolved NumPy terms for arbitrary view subsets.

    The cached axes are action, predictive sample, hypothesis state,
    station-shared rate node, station-shared physical-mark node, and view.
    A subset evaluation gathers only selected views, combines their sufficient
    statistics, and marginalizes station-shared nuisance nodes exactly once.
    """

    leading_shape: tuple[int, ...]
    view_node_log_aqnjrv: NDArray[np.float64]
    latent_log_weights_jr: NDArray[np.float64]
    shared_gamma_concentration: float | None = None
    shared_observed_counts_aqv: NDArray[np.float64] | None = None
    shared_expected_counts_anjv: NDArray[np.float64] | None = None

    @property
    def action_count(self) -> int:
        """Return the flattened number of action or pose candidates."""
        return int(self.view_node_log_aqnjrv.shape[0])

    @property
    def sample_count(self) -> int:
        """Return the number of predictive observation samples."""
        return int(self.view_node_log_aqnjrv.shape[1])

    @property
    def state_count(self) -> int:
        """Return the number of likelihood hypothesis states."""
        return int(self.view_node_log_aqnjrv.shape[2])

    @property
    def view_count(self) -> int:
        """Return the number of cached views or shield pairs."""
        return int(self.view_node_log_aqnjrv.shape[-1])

    @property
    def pair_count(self) -> int:
        """Return the cached view count under the shield-pair terminology."""
        return self.view_count

    @property
    def dtype(self) -> np.dtype[np.float64]:
        """Return the floating-point dtype used by cached likelihood terms."""
        return self.view_node_log_aqnjrv.dtype

    def _normalized_indices(
        self,
        subset_indices: object,
    ) -> NDArray[np.int64]:
        """Validate and broadcast subset indices to action/candidate/view."""
        raw = np.asarray(subset_indices)
        if raw.dtype == np.bool_ or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("Subset view indices must be integer-valued.")
        if raw.ndim == 2:
            indices = np.broadcast_to(
                raw[np.newaxis, ...],
                (self.action_count,) + tuple(raw.shape),
            )
        elif raw.ndim == 3 and int(raw.shape[0]) == self.action_count:
            indices = raw
        else:
            raise ValueError(
                "Subset view indices must be shaped candidate/view or "
                "action/candidate/view."
            )
        if int(indices.shape[1]) <= 0:
            raise ValueError("At least one subset candidate is required.")
        normalized = np.asarray(indices, dtype=np.int64)
        if normalized.size and (
            np.any(normalized < 0) or np.any(normalized >= self.view_count)
        ):
            raise ValueError("Subset view indices are outside the cache range.")
        if int(normalized.shape[-1]) > 1 and np.any(
            np.diff(np.sort(normalized, axis=-1), axis=-1) == 0
        ):
            raise ValueError("One subset cannot contain duplicate views.")
        return normalized

    def _shared_gamma_adjustment(
        self,
        selection_acv: NDArray[np.float64],
    ) -> NDArray[np.float64] | None:
        """Return exact negative-multinomial subset normalization terms."""
        if self.shared_gamma_concentration is None:
            return None
        if (
            self.shared_observed_counts_aqv is None
            or self.shared_expected_counts_anjv is None
        ):
            raise RuntimeError("Shared-Gamma subset statistics are incomplete.")
        observed_by_view = np.moveaxis(
            self.shared_observed_counts_aqv,
            -1,
            1,
        )
        total_observed = np.matmul(selection_acv, observed_by_view)
        expected_by_view = np.moveaxis(
            self.shared_expected_counts_anjv,
            -1,
            1,
        )
        expected_shape = tuple(int(value) for value in expected_by_view.shape)
        total_expected = np.matmul(
            selection_acv,
            expected_by_view.reshape((self.action_count, self.view_count, -1)),
        ).reshape(
            (
                self.action_count,
                int(selection_acv.shape[1]),
            )
            + expected_shape[2:]
        )
        concentration = float(self.shared_gamma_concentration)
        counts = total_observed[:, :, :, np.newaxis, np.newaxis]
        means = total_expected[:, :, np.newaxis, :, :]
        return (
            canonical_log_gamma_numpy(concentration + counts)
            - canonical_log_gamma_numpy(concentration)
            + concentration * np.log(concentration)
            - (concentration + counts) * np.log(concentration + means)
        )

    def _marginalize_nodes(
        self,
        node_log_acqnjr: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Integrate the shared rate and physical-mark quadrature nodes."""
        weights = self.latent_log_weights_jr.reshape(
            (1,) * (node_log_acqnjr.ndim - 2) + tuple(self.latent_log_weights_jr.shape)
        )
        return special.logsumexp(
            node_log_acqnjr + weights,
            axis=(-2, -1),
        )

    def evaluate(
        self,
        subset_indices: object,
    ) -> NDArray[np.float64]:
        """Return exact likelihoods for batched arbitrary view subsets.

        ``subset_indices`` may be shaped ``(C, K)`` and shared by all actions,
        or ``(A, C, K)`` for action-specific candidates.  The result is shaped
        ``(A, C, Q, N)``.  View order has no statistical effect and unselected
        views do not enter any sufficient statistic.
        """
        indices = self._normalized_indices(subset_indices)
        candidate_count = int(indices.shape[1])
        selection = np.zeros(
            (self.action_count, candidate_count, self.view_count),
            dtype=np.float64,
        )
        np.put_along_axis(selection, indices, 1.0, axis=-1)
        view_first = np.moveaxis(self.view_node_log_aqnjrv, -1, 1)
        trailing_shape = tuple(int(value) for value in view_first.shape[2:])
        node_log = np.matmul(
            selection,
            view_first.reshape((self.action_count, self.view_count, -1)),
        ).reshape((self.action_count, candidate_count) + trailing_shape)
        shared_adjustment = self._shared_gamma_adjustment(selection)
        if shared_adjustment is not None:
            node_log = node_log + shared_adjustment[..., np.newaxis]
        return np.asarray(self._marginalize_nodes(node_log), dtype=np.float64)

    def full(self) -> NDArray[np.float64]:
        """Return the exact likelihood using every cached view."""
        node_log = np.sum(self.view_node_log_aqnjrv, axis=-1)
        if self.shared_gamma_concentration is not None:
            if (
                self.shared_observed_counts_aqv is None
                or self.shared_expected_counts_anjv is None
            ):
                raise RuntimeError("Shared-Gamma full statistics are incomplete.")
            counts = np.sum(
                self.shared_observed_counts_aqv,
                axis=-1,
            )[:, :, np.newaxis, np.newaxis]
            means = np.sum(
                self.shared_expected_counts_anjv,
                axis=-1,
            )[:, np.newaxis, :, :]
            concentration = float(self.shared_gamma_concentration)
            node_log = (
                node_log
                + (
                    canonical_log_gamma_numpy(concentration + counts)
                    - canonical_log_gamma_numpy(concentration)
                    + concentration * np.log(concentration)
                    - (concentration + counts) * np.log(concentration + means)
                )[..., np.newaxis]
            )
        flat = self._marginalize_nodes(node_log)
        return flat.reshape(self.leading_shape + (self.sample_count, self.state_count))

@dataclass(frozen=True)
class PreparedTorchSubsetCrossLikelihood:
    """Cache device-resident exact Torch terms for arbitrary view subsets."""

    leading_shape: tuple[int, ...]
    view_node_log_aqnjrv: object
    latent_log_weights_jr: object
    shared_gamma_concentration: float | None = None
    shared_observed_counts_aqv: object | None = None
    shared_expected_counts_anjv: object | None = None

    @property
    def action_count(self) -> int:
        """Return the flattened number of action or pose candidates."""
        return int(self.view_node_log_aqnjrv.shape[0])

    @property
    def sample_count(self) -> int:
        """Return the number of predictive observation samples."""
        return int(self.view_node_log_aqnjrv.shape[1])

    @property
    def state_count(self) -> int:
        """Return the number of likelihood hypothesis states."""
        return int(self.view_node_log_aqnjrv.shape[2])

    @property
    def view_count(self) -> int:
        """Return the number of cached views or shield pairs."""
        return int(self.view_node_log_aqnjrv.shape[-1])

    @property
    def pair_count(self) -> int:
        """Return the cached view count under the shield-pair terminology."""
        return self.view_count

    @property
    def device(self) -> object:
        """Return the device holding all cached likelihood terms."""
        return self.view_node_log_aqnjrv.device

    @property
    def dtype(self) -> object:
        """Return the floating-point dtype used by cached likelihood terms."""
        return self.view_node_log_aqnjrv.dtype

    def _normalized_indices(self, subset_indices: object) -> object:
        """Validate and broadcast subset indices on the cache device."""
        import torch

        raw = torch.as_tensor(subset_indices, device=self.device)
        if raw.dtype == torch.bool or raw.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError("Subset view indices must be integer-valued.")
        if raw.ndim == 2:
            indices = raw.unsqueeze(0).expand(self.action_count, -1, -1)
        elif raw.ndim == 3 and int(raw.shape[0]) == self.action_count:
            indices = raw
        else:
            raise ValueError(
                "Subset view indices must be shaped candidate/view or "
                "action/candidate/view."
            )
        if int(indices.shape[1]) <= 0:
            raise ValueError("At least one subset candidate is required.")
        indices = indices.to(dtype=torch.int64)
        if int(indices.numel()) > 0:
            invalid_range = torch.any((indices < 0) | (indices >= self.view_count))
            if bool(invalid_range.item()):
                raise ValueError("Subset view indices are outside the cache range.")
        if int(indices.shape[-1]) > 1:
            ordered = torch.sort(indices, dim=-1).values
            if bool(torch.any(torch.diff(ordered, dim=-1) == 0).item()):
                raise ValueError("One subset cannot contain duplicate views.")
        return indices

    def _shared_gamma_adjustment(self, selection_acv: object) -> object | None:
        """Return exact negative-multinomial subset normalization terms."""
        import torch

        if self.shared_gamma_concentration is None:
            return None
        if (
            self.shared_observed_counts_aqv is None
            or self.shared_expected_counts_anjv is None
        ):
            raise RuntimeError("Shared-Gamma subset statistics are incomplete.")
        candidate_count = int(selection_acv.shape[1])
        observed_by_view = torch.movedim(
            self.shared_observed_counts_aqv,
            -1,
            1,
        )
        total_observed = torch.bmm(selection_acv, observed_by_view)
        node_count = int(self.shared_expected_counts_anjv.shape[-2])
        expected_by_view = torch.movedim(
            self.shared_expected_counts_anjv,
            -1,
            1,
        )
        total_expected = torch.bmm(
            selection_acv,
            expected_by_view.reshape(
                self.action_count,
                self.view_count,
                self.state_count * node_count,
            ),
        ).reshape(
            self.action_count,
            candidate_count,
            self.state_count,
            node_count,
        )
        concentration = torch.as_tensor(
            float(self.shared_gamma_concentration),
            device=self.device,
            dtype=self.dtype,
        )
        counts = total_observed[:, :, :, None, None]
        means = total_expected[:, :, None, :, :]
        return (
            canonical_log_gamma_torch(concentration + counts)
            - canonical_log_gamma_torch(concentration)
            + concentration * torch.log(concentration)
            - (concentration + counts) * torch.log(concentration + means)
        )

    def _marginalize_nodes(self, node_log_acqnjr: object) -> object:
        """Integrate the shared rate and physical-mark quadrature nodes."""
        import torch

        weights = self.latent_log_weights_jr.reshape(
            (1,) * (node_log_acqnjr.ndim - 2) + tuple(self.latent_log_weights_jr.shape)
        )
        return torch.logsumexp(
            node_log_acqnjr + weights,
            dim=(-2, -1),
        )

    def evaluate(self, subset_indices: object) -> object:
        """Return device-resident exact likelihoods for arbitrary subsets.

        Input shapes are ``(C, K)`` or ``(A, C, K)`` and output shape is
        ``(A, C, Q, N)``.  All arithmetic stays on the prepared response
        device and uses its floating-point dtype.
        """
        import torch

        indices = self._normalized_indices(subset_indices)
        candidate_count = int(indices.shape[1])
        node_count = int(self.view_node_log_aqnjrv.shape[-3])
        mark_node_count = int(self.view_node_log_aqnjrv.shape[-2])
        selection = torch.zeros(
            (self.action_count, candidate_count, self.view_count),
            device=self.device,
            dtype=self.dtype,
        )
        selection.scatter_(-1, indices, 1.0)
        view_first = torch.movedim(self.view_node_log_aqnjrv, -1, 1)
        node_log = torch.bmm(
            selection,
            view_first.reshape(
                self.action_count,
                self.view_count,
                self.sample_count * self.state_count * node_count * mark_node_count,
            ),
        ).reshape(
            self.action_count,
            candidate_count,
            self.sample_count,
            self.state_count,
            node_count,
            mark_node_count,
        )
        shared_adjustment = self._shared_gamma_adjustment(selection)
        if shared_adjustment is not None:
            node_log = node_log + shared_adjustment.unsqueeze(-1)
        return self._marginalize_nodes(node_log)

    def full(self) -> object:
        """Return the exact likelihood using every cached view."""
        import torch

        node_log = torch.sum(self.view_node_log_aqnjrv, dim=-1)
        if self.shared_gamma_concentration is not None:
            if (
                self.shared_observed_counts_aqv is None
                or self.shared_expected_counts_anjv is None
            ):
                raise RuntimeError("Shared-Gamma full statistics are incomplete.")
            counts = torch.sum(
                self.shared_observed_counts_aqv,
                dim=-1,
            )[:, :, None, None]
            means = torch.sum(
                self.shared_expected_counts_anjv,
                dim=-1,
            )[:, None, :, :]
            concentration = torch.as_tensor(
                float(self.shared_gamma_concentration),
                device=self.device,
                dtype=self.dtype,
            )
            node_log = node_log + (
                canonical_log_gamma_torch(concentration + counts)
                - canonical_log_gamma_torch(concentration)
                + concentration * torch.log(concentration)
                - (concentration + counts) * torch.log(concentration + means)
            ).unsqueeze(-1)
        flat = self._marginalize_nodes(node_log)
        return flat.reshape(self.leading_shape + (self.sample_count, self.state_count))

VALIDATION_SCENARIO_IDS = (
    "background_only",
    "single_line_source_resolved",
    "dominant_plus_absent_isotope",
    "multi_isotope_superposition",
    "continuous_surface_perturbation_ranking",
)
SURFACE_BOUNDARY_GATE_SCHEMA_VERSION = 3
SURFACE_BOUNDARY_PROBE_DWELL_TIME_S = 1.0e-2
ACCEPTANCE_METRIC_CONTRACT = MappingProxyType(
    {
        "detector_green_contract_mismatch_count": ("le", 0.0),
        "native_deadtime_contract_mismatch_count": ("le", 0.0),
        "cpu_torch_mean_max_abs_error": ("le", 1.0e-8),
        "cpu_torch_log_likelihood_max_abs_error": ("le", 1.0e-6),
        "background_pairwise_95_coverage_fraction": ("ge", 0.85),
        "background_k_positive_decision_rate_at_p0p95": ("le", 0.05),
        "single_source_pairwise_95_coverage_fraction": ("ge", 0.80),
        "dominant_absent_pairwise_95_coverage_fraction": ("ge", 0.80),
        "absent_isotope_k_positive_decision_rate_at_p0p95": ("le", 0.05),
        "superposition_pairwise_95_coverage_fraction": ("ge", 0.80),
        "truth_vs_perturbed_joint_log_bayes_factor": ("ge", 0.0),
        "pairwise_standardized_total_abs_q95": ("le", 3.0),
        "conditional_mark_upper_tail_ge_0p01_fraction": ("ge", 0.80),
        "source_rate_reference_normalization_max_relative_error": ("le", 1.0e-12),
        "validation_label_production_influence_max_abs": ("le", 0.0),
    }
)


def full_spectrum_acceptance_contract_payload() -> Mapping[str, object]:
    """Return the complete predeclared acceptance execution contract."""
    detector_operator = DetectorGreenOperator.from_artifact(
        CANONICAL_DETECTOR_GREEN_OPERATOR_MANIFEST
    )
    detector_operator.require_runtime_ready()
    construction = detector_operator.construction
    if construction is None:
        raise RuntimeError(
            "Formal acceptance requires detector Green construction provenance."
        )
    return {
        "contract_id": "generic_detector_green_cs_co_acceptance",
        "experiment_profile_id": FULL_SPECTRUM_ACCEPTANCE_EXPERIMENT_ID,
        "dwell_time_s": STANDARD_ACQUISITION_LIVE_TIME_S,
        "validation_scene_seeds": list(DESIGNATED_VALIDATION_SCENE_SEEDS),
        "validation_seed_set": {
            "seed_set_id": "independent_cs_co_validation_20260827",
            "generation_method": "os_csprng_uniform_10_digit",
            "generated_utc_date": "2026-08-27",
            "predeclared_before_acquisition": True,
        },
        "environment": {
            "room_size_xyz_m": list(ACCEPTANCE_ROOM_SIZE_XYZ),
            "detector_pose_xyz_m": list(ACCEPTANCE_DETECTOR_POSE_XYZ),
            "target_blocked_fraction": (ACCEPTANCE_OBSTACLE_BLOCKED_FRACTION),
            "passage_width_m": ACCEPTANCE_PASSAGE_WIDTH_M,
            "surface_chart_max_edge_m": (ACCEPTANCE_SURFACE_CHART_MAX_EDGE_M),
            "obstacle_material": STANDARD_OBSTACLE_MATERIAL,
            "room_boundary_thickness_m": (STANDARD_ROOM_BOUNDARY_THICKNESS_M),
        },
        "geometry_compute": {
            "use_gpu": ACCEPTANCE_GEOMETRY_USE_GPU,
            "device": ACCEPTANCE_GEOMETRY_DEVICE,
            "dtype": ACCEPTANCE_GEOMETRY_DTYPE,
        },
        "continuous_surface_perturbation": {
            "selection": ("first_valid_fixed_order_geometry_only_separable_tangent_v1"),
            "tangent_magnitudes_m": list(ACCEPTANCE_PERTURBATION_TANGENT_MAGNITUDES_M),
            "tangent_directions_uv": [
                list(direction)
                for direction in ACCEPTANCE_PERTURBATION_TANGENT_DIRECTIONS_UV
            ],
            "minimum_surface_displacement_m": (
                ACCEPTANCE_PERTURBATION_MINIMUM_DISPLACEMENT_M
            ),
            "minimum_absolute_log_inverse_square_rate_ratio": (
                ACCEPTANCE_PERTURBATION_MINIMUM_LOG_RATE_SEPARATION
            ),
            "minimum_detector_bearing_angle_rad": (
                ACCEPTANCE_PERTURBATION_MINIMUM_BEARING_ANGLE_RAD
            ),
            "separability_logic": "inverse_square_rate_or_detector_bearing",
            "uses_observation_counts": False,
            "uses_detector_response": False,
            "uses_candidate_model_likelihood": False,
        },
        "surface_boundary_probe": {
            "schema_version": SURFACE_BOUNDARY_GATE_SCHEMA_VERSION,
            "dwell_time_s": SURFACE_BOUNDARY_PROBE_DWELL_TIME_S,
            "surface_emission_epsilon_m": SURFACE_EMISSION_EPSILON_M,
            "native_position_variants": [
                "exact_surface_anchor",
                "air_plus_epsilon",
                "solid_minus_epsilon",
            ],
            "require_nonempty_transport_process_counts": True,
        },
        "native_process_counter_policy": {
            "background_only": "exact_empty_counter_map",
            "source_present": "nonempty_positive_counter_map",
        },
        "detector_cps_green_reference_efficiency_policy": (
            "recomputed_from_authenticated_catalog_and_operator_strict_tolerance_v1"
        ),
        "detector_response_event_policy": {
            "primary_emission_model": "independent_gamma_lines",
            "source_bias_cone_policy": "detector_covering",
            "catalog_line_semantics": (
                "positive_intensity_lines_normalized_per_isotope"
            ),
            "prompt_decay_cascade_transport": False,
            "true_coincidence_summing": "disabled",
            "coincidence_window_s": 1.0e-6,
            "sampling_mode": DETECTOR_GREEN_SAMPLING_MODE,
            "coincidence_semantics": DETECTOR_GREEN_COINCIDENCE_SEMANTICS,
            "counter_semantics": (
                "incident_ge_registered_ge_pulses_and_merged_entry_excess_"
                "ge_multi_entry_pulses_v1"
            ),
        },
        "shield_pair_ids": list(range(64)),
        "scenario_ids": list(VALIDATION_SCENARIO_IDS),
        "metrics": {
            metric_id: {
                "comparison": comparison,
                "threshold": float(threshold),
            }
            for metric_id, (comparison, threshold) in (
                ACCEPTANCE_METRIC_CONTRACT.items()
            )
        },
        "candidate_selection": {
            "method": "none_predeclared_physics_only",
            "isotopes": ["Co-60", "Cs-137"],
            "scene_fit": False,
            "validation_feedback": False,
            "detector_operator_validation": (
                "independent_catalog_excluded_monoenergetic_holdout"
            ),
        },
        "detector_green_operator": {
            "operator_id": DETECTOR_GREEN_OPERATOR_ID,
            "contract_sha256": detector_operator.contract_hash_sha256,
            "binary_sha256": detector_operator.binary_sha256,
            "construction_raw_corpus_sha256": construction["raw_corpus_sha256"],
            "construction_implementation_bundle_sha256": construction[
                "detector_implementation_bundle_sha256"
            ],
        },
        "selection_policy": ("thresholds_fixed_before_validation_no_validation_tuning"),
    }


def _freeze_json_value(value: object) -> object:
    """Return an immutable recursively copied JSON-compatible value."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError("Validation and line manifests must be JSON-compatible.")


def _thaw_json_value(value: object) -> object:
    """Return a detached mutable JSON-compatible copy."""
    if isinstance(value, Mapping):
        return {str(key): _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


def _canonical_json_sha256(value: object) -> str:
    """Return the SHA-256 of one canonical JSON-compatible value."""
    return hashlib.sha256(
        json.dumps(
            _thaw_json_value(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _reject_nonfinite_json_constant(value: str) -> None:
    """Reject non-standard non-finite constants in model JSON."""
    raise ValueError(
        "Full-spectrum model JSON must contain finite standard-JSON numbers; "
        f"found {value}."
    )


def _strict_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    """Build one model JSON object while rejecting duplicate keys."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(
                f"Full-spectrum model JSON contains duplicate key {key!r}."
            )
        result[key] = value
    return result


def _strict_json_number(value: object, *, field_name: str) -> float:
    """Return one finite JSON number without accepting strings or booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a JSON number.")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite.")
    return parsed


def _strict_json_number_sequence(
    value: object,
    *,
    field_name: str,
) -> tuple[float, ...]:
    """Return one nonempty finite JSON-number sequence without coercion."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise TypeError(f"{field_name} must be a nonempty JSON array.")
    return tuple(
        _strict_json_number(
            item,
            field_name=f"{field_name}[{index}]",
        )
        for index, item in enumerate(value)
    )


FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256 = _canonical_json_sha256(
    full_spectrum_acceptance_contract_payload()
)


def rate_scale_mixture_for_half_width(
    half_width: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return the predeclared symmetric positive mean-one scale mixture."""
    width = float(half_width)
    if width not in RATE_SCALE_HALF_WIDTH_GRID:
        raise ValueError(
            "Rate-scale half width must belong to the predeclared training grid."
        )
    return (
        (1.0 - width, 1.0, 1.0 + width),
        RATE_SCALE_MIXTURE_WEIGHTS,
    )


def continuous_rate_scale_quadrature_for_half_width(
    half_width: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return a mean-one quadrature for one continuous uniform rate scale."""
    width = float(half_width)
    if width not in RATE_SCALE_HALF_WIDTH_GRID:
        raise ValueError(
            "Rate-scale half width must belong to the predeclared training grid."
        )
    if width == 0.0:
        return (1.0,), (1.0,)
    abscissas, raw_weights = np.polynomial.legendre.leggauss(
        RATE_SCALE_UNIFORM_QUADRATURE_ORDER
    )
    nodes = 1.0 + width * abscissas
    weights = raw_weights / 2.0
    weights /= np.sum(weights)
    return (
        tuple(float(value) for value in nodes),
        tuple(float(value) for value in weights),
    )


def _is_sha256(value: object) -> bool:
    """Return whether one value is a lowercase hexadecimal SHA-256 string."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _array_digest(array: NDArray[np.float64]) -> bytes:
    """Return shape-sensitive bytes for one immutable float64 array."""
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(tuple(int(value) for value in contiguous.shape)).encode())
    digest.update(contiguous.tobytes())
    return digest.digest()


_DERIVED_CONTRACT_ARRAY_DECIMALS = 12


def _portable_derived_array_digest(array: NDArray[np.float64]) -> bytes:
    """Hash a derived float64 array independently of CPU math kernels."""
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    if np.any(~np.isfinite(contiguous)):
        raise ValueError("Contract arrays must contain only finite values.")
    # Canonicalize only the digest copy. Inference and observation arrays retain
    # their original float64 values and therefore their existing physics.
    canonical = np.asarray(
        np.round(
            contiguous,
            decimals=_DERIVED_CONTRACT_ARRAY_DECIMALS,
        ),
        dtype="<f8",
        order="C",
    ).copy()
    canonical[canonical == 0.0] = 0.0
    digest = hashlib.sha256()
    digest.update(
        b"portable-derived-float64-rounded-"
        + str(_DERIVED_CONTRACT_ARRAY_DECIMALS).encode("ascii")
        + b"-v2"
    )
    digest.update(str(tuple(int(value) for value in canonical.shape)).encode())
    digest.update(canonical.tobytes())
    return digest.digest()


def _logdiffexp_numpy(
    log_large: NDArray[np.float64],
    log_small: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return ``log(exp(log_large) - exp(log_small))`` stably."""
    large = np.asarray(log_large, dtype=np.float64)
    small = np.asarray(log_small, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.minimum(small - large, 0.0)
        result = large + np.log(-np.expm1(delta))
    return np.where(small < large, result, -np.inf)


def _logdiffexp_torch(log_large: object, log_small: object) -> object:
    """Return the Torch stable logarithm of a positive exponential difference."""
    import torch

    delta = torch.minimum(log_small - log_large, torch.zeros_like(log_large))
    result = log_large + torch.log(-torch.expm1(delta))
    return torch.where(log_small < log_large, result, -torch.inf)


def _regularized_gamma_interval_log_numpy(
    shape: NDArray[np.float64],
    lower_argument: NDArray[np.float64],
    upper_argument: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return a regularized gamma interval integral in stable log space."""
    a, lower, upper = np.broadcast_arrays(
        np.asarray(shape, dtype=np.float64),
        np.asarray(lower_argument, dtype=np.float64),
        np.asarray(upper_argument, dtype=np.float64),
    )
    width = upper - lower
    result = np.full(a.shape, -np.inf, dtype=np.float64)
    valid = width > 0.0
    if not np.any(valid):
        return result
    half_width = 0.5 * width[valid]
    midpoint = 0.5 * (upper[valid] + lower[valid])
    points = (
        midpoint[:, None] + half_width[:, None] * _RENEWAL_GAMMA_INTERVAL_NODES[None, :]
    )
    log_density = (
        special.xlogy(a[valid, None] - 1.0, points)
        - points
        - canonical_log_gamma_numpy(a[valid, None])
    )
    result[valid] = np.log(half_width) + special.logsumexp(
        np.log(_RENEWAL_GAMMA_INTERVAL_WEIGHTS)[None, :] + log_density,
        axis=-1,
    )
    return result


def _regularized_gamma_interval_log_torch(
    shape: object,
    lower_argument: object,
    upper_argument: object,
) -> object:
    """Return a Torch regularized gamma interval integral in log space."""
    import torch

    a, lower, upper = torch.broadcast_tensors(
        shape,
        lower_argument,
        upper_argument,
    )
    width = upper - lower
    half_width = 0.5 * width
    midpoint = 0.5 * (upper + lower)
    nodes = torch.as_tensor(
        _RENEWAL_GAMMA_INTERVAL_NODES,
        dtype=a.dtype,
        device=a.device,
    )
    weights = torch.as_tensor(
        _RENEWAL_GAMMA_INTERVAL_WEIGHTS,
        dtype=a.dtype,
        device=a.device,
    )
    points = midpoint.unsqueeze(-1) + half_width.unsqueeze(-1) * nodes
    log_density = (
        torch.xlogy(a.unsqueeze(-1) - 1.0, points)
        - points
        - canonical_log_gamma_torch(a).unsqueeze(-1)
    )
    log_interval = torch.log(half_width) + torch.logsumexp(
        torch.log(weights) + log_density,
        dim=-1,
    )
    return torch.where(width > 0.0, log_interval, -torch.inf)


def _renewal_positive_decomposition_numpy(
    counts: NDArray[np.float64],
    first_arguments: NDArray[np.float64],
    second_arguments: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate the positive-term renewal PMF decomposition in log space.

    For integer ``m >= 1`` and ``a >= b >= 0``, the renewal probability is

    ``[P(m, a) - P(m, b)] + exp(-b) b**m / Gamma(m + 1)``.

    Both terms are nonnegative.  The first is evaluated directly as the
    gamma-density integral over the finite dead-time interval.  This avoids
    subtracting nearly equal regularized-gamma tails and remains stable when
    the incident mean and observed count are both very large.
    """
    m, first, second = np.broadcast_arrays(
        np.asarray(counts, dtype=np.float64),
        np.asarray(first_arguments, dtype=np.float64),
        np.asarray(second_arguments, dtype=np.float64),
    )
    starts_at_zero = second == 0.0
    log_interval = np.full(m.shape, -np.inf, dtype=np.float64)
    if np.any(starts_at_zero):
        log_interval[starts_at_zero] = _regularized_gamma_lower_log_numpy(
            m[starts_at_zero],
            first[starts_at_zero],
        )
    if np.any(~starts_at_zero):
        log_interval[~starts_at_zero] = _regularized_gamma_interval_log_numpy(
            m[~starts_at_zero],
            second[~starts_at_zero],
            first[~starts_at_zero],
        )
    log_boundary = (
        special.xlogy(m, second) - second - canonical_log_gamma_numpy(m + 1.0)
    )
    return np.asarray(
        np.logaddexp(log_interval, log_boundary),
        dtype=np.float64,
    )


def _regularized_gamma_lower_log_numpy(
    shape: NDArray[np.float64],
    argument: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return a zero-origin lower gamma interval by a positive series."""
    a, x = np.broadcast_arrays(
        np.asarray(shape, dtype=np.float64),
        np.asarray(argument, dtype=np.float64),
    )
    term = np.ones(a.shape, dtype=np.float64)
    series = np.ones(a.shape, dtype=np.float64)
    tolerance = 8.0 * np.finfo(np.float64).eps
    converged = np.zeros(a.shape, dtype=np.bool_)
    for iteration in range(1, RENEWAL_LOG_GAMMA_MAX_ITERATIONS + 1):
        term *= x / (a + float(iteration))
        series += term
        converged = term <= tolerance * series
        if np.all(converged):
            break
    if not np.all(converged):
        raise RuntimeError("Lower regularized-gamma log series did not converge.")
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (
            special.xlogy(a, x)
            - x
            - canonical_log_gamma_numpy(a + 1.0)
            + np.log(series)
        )
    return np.where(x > 0.0, result, -np.inf)


def _renewal_positive_decomposition_torch(
    counts: object,
    first_arguments: object,
    second_arguments: object,
) -> object:
    """Return the Torch positive-term renewal PMF decomposition."""
    import torch

    m, first, second = torch.broadcast_tensors(
        counts,
        first_arguments,
        second_arguments,
    )
    starts_at_zero = second == 0.0
    log_interval = torch.full_like(m, -torch.inf)
    if bool(torch.any(starts_at_zero)):
        log_interval = log_interval.masked_scatter(
            starts_at_zero,
            _regularized_gamma_lower_log_torch(
                m[starts_at_zero],
                first[starts_at_zero],
            ),
        )
    if bool(torch.any(~starts_at_zero)):
        log_interval = log_interval.masked_scatter(
            ~starts_at_zero,
            _regularized_gamma_interval_log_torch(
                m[~starts_at_zero],
                second[~starts_at_zero],
                first[~starts_at_zero],
            ),
        )
    log_boundary = torch.xlogy(m, second) - second - canonical_log_gamma_torch(m + 1.0)
    return torch.logaddexp(log_interval, log_boundary)


def _regularized_gamma_lower_log_torch(
    shape: object,
    argument: object,
) -> object:
    """Return a Torch zero-origin lower gamma interval by positive series."""
    import torch

    a, x = torch.broadcast_tensors(shape, argument)
    term = torch.ones_like(a)
    series = torch.ones_like(a)
    tolerance = 8.0 * torch.finfo(a.dtype).eps
    converged = torch.zeros_like(a, dtype=torch.bool)
    for iteration in range(1, RENEWAL_LOG_GAMMA_MAX_ITERATIONS + 1):
        term = term * x / (a + float(iteration))
        series = series + term
        converged = term <= tolerance * series
        if bool(torch.all(converged)):
            break
    if not bool(torch.all(converged)):
        raise RuntimeError("Torch lower regularized-gamma log series did not converge.")
    result = (
        torch.xlogy(a, x) - x - canonical_log_gamma_torch(a + 1.0) + torch.log(series)
    )
    return torch.where(x > 0.0, result, -torch.inf)


def nonparalyzable_count_log_probability_numpy(
    observed_counts: NDArray[np.float64],
    incident_rates_cps: NDArray[np.float64],
    live_times_s: NDArray[np.float64],
    *,
    dead_time_tau_s: float,
) -> NDArray[np.float64]:
    """Return exact nonparalyzable renewal-count log probabilities.

    The detector starts live.  For ``m >= 1`` the probability is

    ``F_Gamma(m, rate; T-(m-1)tau) - F_Gamma(m+1, rate; T-m tau)``.

    The implementation switches between lower and upper regularized gamma
    tails before applying ``logdiffexp``.  At exactly zero dead time it uses
    the algebraically equivalent Poisson law directly.
    """
    counts, rates, live_times = np.broadcast_arrays(
        np.asarray(observed_counts, dtype=np.float64),
        np.asarray(incident_rates_cps, dtype=np.float64),
        np.asarray(live_times_s, dtype=np.float64),
    )
    tau = float(dead_time_tau_s)
    if (
        np.any(~np.isfinite(counts))
        or np.any(counts < 0.0)
        or np.any(counts != np.floor(counts))
        or np.any(~np.isfinite(rates))
        or np.any(rates < 0.0)
        or np.any(~np.isfinite(live_times))
        or np.any(live_times <= 0.0)
        or not np.isfinite(tau)
        or tau < 0.0
    ):
        raise ValueError(
            "Renewal counts/rates/live times must be finite with exact "
            "nonnegative integer counts, nonnegative rates, and positive times."
        )
    mean = rates * live_times
    if tau == 0.0:
        return np.asarray(
            special.xlogy(counts, mean)
            - mean
            - canonical_log_gamma_numpy(counts + 1.0),
            dtype=np.float64,
        )
    result = np.full(counts.shape, -np.inf, dtype=np.float64)
    zero = counts == 0.0
    result[zero] = -mean[zero]
    positive = ~zero
    if not np.any(positive):
        return result
    m = counts[positive]
    rate = rates[positive]
    live = live_times[positive]
    raw_second_window = live - m * tau
    first_window = np.maximum(raw_second_window + tau, 0.0)
    second_window = np.maximum(raw_second_window, 0.0)
    first_argument = rate * first_window
    second_argument = rate * second_window
    lower_first = special.gammainc(m, first_argument)
    lower_second = special.gammainc(m + 1.0, second_argument)
    upper_first = special.gammaincc(m, first_argument)
    upper_second = special.gammaincc(m + 1.0, second_argument)
    with np.errstate(divide="ignore"):
        lower_log = _logdiffexp_numpy(
            np.log(lower_first),
            np.log(lower_second),
        )
        upper_log = _logdiffexp_numpy(
            np.log(upper_second),
            np.log(upper_first),
        )
    use_lower = lower_first <= 0.5
    selected = np.where(use_lower, lower_log, upper_log)
    needs_exact_recovery = ~np.isfinite(selected)
    if np.any(needs_exact_recovery):
        recovered = _renewal_positive_decomposition_numpy(
            m[needs_exact_recovery],
            first_argument[needs_exact_recovery],
            second_argument[needs_exact_recovery],
        )
        if np.any(np.isnan(recovered)) or np.any(np.isposinf(recovered)):
            raise RuntimeError("Positive-term renewal likelihood recovery was invalid.")
        selected[needs_exact_recovery] = recovered
    result[positive] = selected
    return result


def nonparalyzable_count_log_probability_torch(
    observed_counts: object,
    incident_rates_cps: object,
    live_times_s: object,
    *,
    dead_time_tau_s: float,
    validate_inputs: bool = True,
) -> object:
    """Return the Torch equivalent renewal-count log probabilities."""
    import torch

    rates = torch.as_tensor(incident_rates_cps)
    if rates.dtype != torch.float64:
        raise TypeError("Production renewal likelihood requires torch.float64.")
    dtype = rates.dtype
    device = rates.device
    counts = torch.as_tensor(
        observed_counts,
        dtype=dtype,
        device=device,
    )
    live_times = torch.as_tensor(
        live_times_s,
        dtype=dtype,
        device=device,
    )
    counts, rates, live_times = torch.broadcast_tensors(
        counts,
        rates,
        live_times,
    )
    tau = float(dead_time_tau_s)
    if not np.isfinite(tau) or tau < 0.0:
        raise ValueError("Torch renewal-count inputs are invalid.")
    if validate_inputs:
        invalid = torch.stack(
            (
                torch.any(~torch.isfinite(counts)),
                torch.any(counts < 0.0),
                torch.any(counts != torch.floor(counts)),
                torch.any(~torch.isfinite(rates)),
                torch.any(rates < 0.0),
                torch.any(~torch.isfinite(live_times)),
                torch.any(live_times <= 0.0),
            )
        ).any()
        if bool(invalid.item()):
            raise ValueError("Torch renewal-count inputs are invalid.")
    mean = rates * live_times
    if tau == 0.0:
        safe_mean = torch.clamp(mean, min=torch.finfo(dtype).tiny)
        poisson = (
            torch.xlogy(counts, safe_mean)
            - mean
            - canonical_log_gamma_torch(counts + 1.0)
        )
        return torch.where(
            (mean == 0.0) & (counts == 0.0),
            torch.zeros_like(poisson),
            torch.where(
                (mean == 0.0) & (counts > 0.0),
                torch.full_like(poisson, -torch.inf),
                poisson,
            ),
        )
    positive = counts > 0.0
    m = torch.where(positive, counts, torch.ones_like(counts))
    raw_second_window = live_times - m * tau
    first_window = torch.clamp(raw_second_window + tau, min=0.0)
    second_window = torch.clamp(raw_second_window, min=0.0)
    first_argument = rates * first_window
    second_argument = rates * second_window
    lower_first = torch.special.gammainc(m, first_argument)
    lower_second = torch.special.gammainc(m + 1.0, second_argument)
    upper_first = torch.special.gammaincc(m, first_argument)
    upper_second = torch.special.gammaincc(m + 1.0, second_argument)
    lower_log = _logdiffexp_torch(
        torch.log(lower_first),
        torch.log(lower_second),
    )
    upper_log = _logdiffexp_torch(
        torch.log(upper_second),
        torch.log(upper_first),
    )
    selected = torch.where(lower_first <= 0.5, lower_log, upper_log)
    needs_exact_recovery = positive & ~torch.isfinite(selected)
    if bool(torch.any(needs_exact_recovery)):
        recovered = _renewal_positive_decomposition_torch(
            m[needs_exact_recovery],
            first_argument[needs_exact_recovery],
            second_argument[needs_exact_recovery],
        )
        invalid_recovery = torch.stack(
            (
                torch.any(torch.isnan(recovered)),
                torch.any(torch.isposinf(recovered)),
            )
        ).any()
        if bool(invalid_recovery.item()):
            raise RuntimeError(
                "Torch positive-term renewal likelihood recovery was invalid."
            )
        selected = selected.masked_scatter(needs_exact_recovery, recovered)
    return torch.where(positive, selected, -mean)


def station_shared_gamma_poisson_count_log_increments_numpy(
    observed_counts_qv: NDArray[np.float64],
    expected_counts_njv: NDArray[np.float64],
    *,
    concentration: float,
) -> NDArray[np.float64]:
    """Return per-view increments of a shared-Gamma Poisson count law.

    A mean-one Gamma latent variable is shared by every view in a station.
    Conditional view counts are Poisson with the supplied recorded-count
    means. Marginalizing that scalar gives a negative-multinomial law. The
    returned increments telescope exactly to each view-prefix likelihood.
    """
    observed = np.asarray(observed_counts_qv, dtype=np.float64)
    expected = np.asarray(expected_counts_njv, dtype=np.float64)
    shape = float(concentration)
    if (
        observed.ndim < 2
        or expected.ndim < 3
        or observed.shape[:-2] != expected.shape[:-3]
        or observed.shape[-1] != expected.shape[-1]
        or np.any(~np.isfinite(observed))
        or np.any(observed < 0.0)
        or np.any(observed != np.floor(observed))
        or np.any(~np.isfinite(expected))
        or np.any(expected < 0.0)
        or not np.isfinite(shape)
        or shape <= 0.0
    ):
        raise ValueError("Shared-Gamma count inputs are invalid.")
    counts = observed[..., :, np.newaxis, np.newaxis, :]
    means = expected[..., np.newaxis, :, :, :]
    cumulative_counts = np.cumsum(counts, axis=-1)
    cumulative_means = np.cumsum(means, axis=-1)
    component_terms = special.xlogy(counts, means) - canonical_log_gamma_numpy(
        counts + 1.0
    )
    cumulative_components = np.cumsum(component_terms, axis=-1)
    prefix_log = (
        canonical_log_gamma_numpy(shape + cumulative_counts)
        - canonical_log_gamma_numpy(shape)
        + shape * np.log(shape)
        - (shape + cumulative_counts) * np.log(shape + cumulative_means)
        + cumulative_components
    )
    return np.diff(
        prefix_log,
        axis=-1,
        prepend=np.zeros(prefix_log.shape[:-1] + (1,), dtype=np.float64),
    )


def station_shared_gamma_poisson_count_log_increments_torch(
    observed_counts_qv: object,
    expected_counts_njv: object,
    *,
    concentration: float,
    validate_inputs: bool = True,
) -> object:
    """Return Torch increments of the shared-Gamma Poisson count law."""
    import torch

    expected = torch.as_tensor(expected_counts_njv)
    observed = torch.as_tensor(
        observed_counts_qv,
        device=expected.device,
        dtype=expected.dtype,
    )
    shape = float(concentration)
    if (
        observed.ndim < 2
        or expected.ndim < 3
        or tuple(observed.shape[:-2]) != tuple(expected.shape[:-3])
        or int(observed.shape[-1]) != int(expected.shape[-1])
        or not np.isfinite(shape)
        or shape <= 0.0
    ):
        raise ValueError("Torch shared-Gamma count inputs are invalid.")
    if validate_inputs:
        invalid = torch.stack(
            (
                torch.any(~torch.isfinite(observed)),
                torch.any(observed < 0.0),
                torch.any(observed != torch.floor(observed)),
                torch.any(~torch.isfinite(expected)),
                torch.any(expected < 0.0),
            )
        ).any()
        if bool(invalid.item()):
            raise ValueError("Torch shared-Gamma count inputs are invalid.")
    counts = observed.unsqueeze(-2).unsqueeze(-2)
    means = expected.unsqueeze(-4)
    cumulative_counts = torch.cumsum(counts, dim=-1)
    cumulative_means = torch.cumsum(means, dim=-1)
    component_terms = torch.xlogy(counts, means) - canonical_log_gamma_torch(
        counts + 1.0
    )
    cumulative_components = torch.cumsum(component_terms, dim=-1)
    shape_tensor = torch.as_tensor(
        shape,
        device=expected.device,
        dtype=expected.dtype,
    )
    prefix_log = (
        canonical_log_gamma_torch(shape_tensor + cumulative_counts)
        - canonical_log_gamma_torch(shape_tensor)
        + shape_tensor * torch.log(shape_tensor)
        - (shape_tensor + cumulative_counts)
        * torch.log(shape_tensor + cumulative_means)
        + cumulative_components
    )
    zero = torch.zeros(
        prefix_log.shape[:-1] + (1,),
        device=expected.device,
        dtype=expected.dtype,
    )
    return torch.diff(prefix_log, dim=-1, prepend=zero)


def view_independent_gamma_poisson_count_log_increments_numpy(
    observed_counts_qv: NDArray[np.float64],
    expected_counts_njv: NDArray[np.float64],
    *,
    concentration: float | NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return per-view Gamma-Poisson count log probabilities."""
    observed = np.asarray(observed_counts_qv, dtype=np.float64)
    expected = np.asarray(expected_counts_njv, dtype=np.float64)
    raw_shape = np.asarray(concentration, dtype=np.float64)
    try:
        shape = np.broadcast_to(raw_shape, expected.shape)
    except ValueError as exc:
        raise ValueError(
            "View-independent Gamma concentrations are not broadcastable."
        ) from exc
    if (
        observed.ndim < 2
        or expected.ndim < 3
        or observed.shape[:-2] != expected.shape[:-3]
        or observed.shape[-1] != expected.shape[-1]
        or np.any(~np.isfinite(observed))
        or np.any(observed < 0.0)
        or np.any(observed != np.floor(observed))
        or np.any(~np.isfinite(expected))
        or np.any(expected < 0.0)
        or np.any(~np.isfinite(shape))
        or np.any(shape <= 0.0)
    ):
        raise ValueError("View-independent Gamma count inputs are invalid.")
    counts = observed[..., :, np.newaxis, np.newaxis, :]
    means = expected[..., np.newaxis, :, :, :]
    shapes = shape[..., np.newaxis, :, :, :]
    return (
        canonical_log_gamma_numpy(shapes + counts)
        - canonical_log_gamma_numpy(shapes)
        - canonical_log_gamma_numpy(counts + 1.0)
        + shapes * np.log(shapes)
        + special.xlogy(counts, means)
        - (shapes + counts) * np.log(shapes + means)
    )


def view_independent_gamma_poisson_count_log_increments_torch(
    observed_counts_qv: object,
    expected_counts_njv: object,
    *,
    concentration: object,
    validate_inputs: bool = True,
) -> object:
    """Return Torch per-view Gamma-Poisson count log probabilities."""
    import torch

    expected = torch.as_tensor(expected_counts_njv)
    observed = torch.as_tensor(
        observed_counts_qv,
        device=expected.device,
        dtype=expected.dtype,
    )
    raw_shape = torch.as_tensor(
        concentration,
        device=expected.device,
        dtype=expected.dtype,
    )
    try:
        shape = torch.broadcast_to(raw_shape, expected.shape)
    except RuntimeError as exc:
        raise ValueError(
            "Torch view-independent Gamma concentrations are not broadcastable."
        ) from exc
    if (
        observed.ndim < 2
        or expected.ndim < 3
        or tuple(observed.shape[:-2]) != tuple(expected.shape[:-3])
        or int(observed.shape[-1]) != int(expected.shape[-1])
    ):
        raise ValueError("Torch view-independent Gamma count inputs are invalid.")
    if validate_inputs:
        invalid = torch.stack(
            (
                torch.any(~torch.isfinite(observed)),
                torch.any(observed < 0.0),
                torch.any(observed != torch.floor(observed)),
                torch.any(~torch.isfinite(expected)),
                torch.any(expected < 0.0),
                torch.any(~torch.isfinite(shape)),
                torch.any(shape <= 0.0),
            )
        ).any()
        if bool(invalid.item()):
            raise ValueError("Torch view-independent Gamma count inputs are invalid.")
    counts = observed.unsqueeze(-2).unsqueeze(-2)
    means = expected.unsqueeze(-4)
    shape_tensor = shape.unsqueeze(-4)
    gamma_arguments = torch.broadcast_tensors(
        shape_tensor + counts,
        shape_tensor,
        counts + 1.0,
    )
    gamma_terms = _canonical_log_gamma_torch_unchecked(
        torch.stack(gamma_arguments, dim=0)
    )
    return (
        gamma_terms[0]
        - gamma_terms[1]
        - gamma_terms[2]
        + shape_tensor * torch.log(shape_tensor)
        + torch.xlogy(counts, means)
        - (shape_tensor + counts) * torch.log(shape_tensor + means)
    )


def nonparalyzable_count_cdf_numpy(
    count_threshold: NDArray[np.int64],
    incident_rates_cps: NDArray[np.float64],
    live_times_s: NDArray[np.float64],
    *,
    dead_time_tau_s: float,
) -> NDArray[np.float64]:
    """Return ``P(M <= m)`` for a nonparalyzable detector starting live."""
    threshold, rates, live_times = np.broadcast_arrays(
        np.asarray(count_threshold, dtype=np.int64),
        np.asarray(incident_rates_cps, dtype=np.float64),
        np.asarray(live_times_s, dtype=np.float64),
    )
    tau = float(dead_time_tau_s)
    remaining = live_times - threshold.astype(np.float64) * tau
    argument = rates * np.maximum(remaining, 0.0)
    cdf = special.gammaincc(threshold.astype(np.float64) + 1.0, argument)
    cdf = np.where(threshold < 0, 0.0, cdf)
    cdf = np.where(remaining <= 0.0, 1.0, cdf)
    return np.asarray(np.clip(cdf, 0.0, 1.0), dtype=np.float64)


def sample_nonparalyzable_counts_numpy(
    incident_rates_cps: NDArray[np.float64],
    live_times_s: NDArray[np.float64],
    *,
    dead_time_tau_s: float,
    sample_count: int,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Draw renewal totals by vectorized inverse-CDF integer bisection."""
    rates, live_times = np.broadcast_arrays(
        np.asarray(incident_rates_cps, dtype=np.float64),
        np.asarray(live_times_s, dtype=np.float64),
    )
    if int(sample_count) <= 0:
        raise ValueError("sample_count must be positive.")
    if (
        np.any(~np.isfinite(rates))
        or np.any(rates < 0.0)
        or np.any(~np.isfinite(live_times))
        or np.any(live_times <= 0.0)
    ):
        raise ValueError("Renewal sampling rates/times are invalid.")
    sample_shape = rates.shape + (int(sample_count),)
    expanded_rates = np.broadcast_to(rates[..., np.newaxis], sample_shape)
    expanded_times = np.broadcast_to(
        live_times[..., np.newaxis],
        sample_shape,
    )
    if float(dead_time_tau_s) == 0.0:
        return np.asarray(
            rng.poisson(expanded_rates * expanded_times),
            dtype=np.int64,
        )
    uniform = rng.random(sample_shape)
    poisson_mean = expanded_rates * expanded_times
    initial_high = np.ceil(poisson_mean + 10.0 * np.sqrt(poisson_mean + 1.0) + 10.0)
    if np.any(initial_high >= float(np.iinfo(np.int64).max // 2)):
        raise OverflowError("Renewal count support exceeds int64.")
    high = np.asarray(np.maximum(initial_high, 0.0), dtype=np.int64)
    support_maximum = np.asarray(
        np.floor(expanded_times / float(dead_time_tau_s)) + 1.0,
        dtype=np.int64,
    )
    high = np.minimum(high, support_maximum)
    high_cdf = nonparalyzable_count_cdf_numpy(
        high,
        expanded_rates,
        expanded_times,
        dead_time_tau_s=float(dead_time_tau_s),
    )
    unresolved = high_cdf < uniform
    while bool(np.any(unresolved)):
        expanded_high = np.minimum(
            np.maximum(2 * high + 1, 1),
            support_maximum,
        )
        if np.array_equal(expanded_high[unresolved], high[unresolved]):
            raise RuntimeError(
                "Renewal inverse-CDF upper support failed to bracket a draw."
            )
        high = np.where(unresolved, expanded_high, high)
        high_cdf = nonparalyzable_count_cdf_numpy(
            high,
            expanded_rates,
            expanded_times,
            dead_time_tau_s=float(dead_time_tau_s),
        )
        unresolved = high_cdf < uniform
    low = np.full(sample_shape, -1, dtype=np.int64)
    while bool(np.any(high - low > 1)):
        midpoint = low + (high - low) // 2
        cdf = nonparalyzable_count_cdf_numpy(
            midpoint,
            expanded_rates,
            expanded_times,
            dead_time_tau_s=float(dead_time_tau_s),
        )
        move_high = cdf >= uniform
        high = np.where(move_high, midpoint, high)
        low = np.where(move_high, low, midpoint)
    return high


def _klein_nishina_total_cross_section_cm2(
    energy_keV: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return total Klein-Nishina cross section per electron."""
    energy = np.asarray(energy_keV, dtype=np.float64)
    alpha = np.maximum(energy / ELECTRON_REST_ENERGY_KEV, 1.0e-12)
    log_term = np.log1p(2.0 * alpha)
    bracket = (
        (1.0 + alpha)
        / np.square(alpha)
        * (2.0 * (1.0 + alpha) / (1.0 + 2.0 * alpha) - log_term / alpha)
        + log_term / (2.0 * alpha)
        - (1.0 + 3.0 * alpha) / np.square(1.0 + 2.0 * alpha)
    )
    return 2.0 * np.pi * CLASSICAL_ELECTRON_RADIUS_CM**2 * np.maximum(bracket, 0.0)


def _klein_nishina_transition_matrix(
    energy_axis_keV: NDArray[np.float64],
    *,
    quadrature_order: int,
) -> NDArray[np.float64]:
    """Build a column-stochastic single-Compton-scatter energy operator."""
    axis = np.asarray(energy_axis_keV, dtype=np.float64)
    if axis.ndim != 1 or axis.size < 2 or np.any(np.diff(axis) <= 0.0):
        raise ValueError("Klein-Nishina transition requires an increasing axis.")
    mu, quadrature_weights = np.polynomial.legendre.leggauss(int(quadrature_order))
    incident = axis[:, np.newaxis]
    alpha = incident / ELECTRON_REST_ENERGY_KEV
    ratio = 1.0 / (1.0 + alpha * (1.0 - mu[np.newaxis, :]))
    scattered = incident * ratio
    differential = np.square(ratio) * (
        ratio
        + 1.0 / np.maximum(ratio, np.finfo(np.float64).tiny)
        - (1.0 - np.square(mu))[np.newaxis, :]
    )
    weights = np.maximum(
        differential * quadrature_weights[np.newaxis, :],
        0.0,
    )
    weights /= np.maximum(
        np.sum(weights, axis=1, keepdims=True),
        np.finfo(np.float64).tiny,
    )
    bin_width = float(axis[1] - axis[0])
    fractional = (scattered - float(axis[0])) / bin_width
    lower = np.floor(fractional).astype(np.int64)
    upper_fraction = fractional - lower
    lower = np.clip(lower, 0, axis.size - 1)
    upper = np.clip(lower + 1, 0, axis.size - 1)
    input_indices = np.broadcast_to(
        np.arange(axis.size, dtype=np.int64)[:, np.newaxis],
        lower.shape,
    )
    transition = np.zeros((axis.size, axis.size), dtype=np.float64)
    np.add.at(
        transition,
        (lower.reshape(-1), input_indices.reshape(-1)),
        (weights * (1.0 - upper_fraction)).reshape(-1),
    )
    np.add.at(
        transition,
        (upper.reshape(-1), input_indices.reshape(-1)),
        (weights * upper_fraction).reshape(-1),
    )
    column_sums = np.sum(transition, axis=0)
    zero_energy = axis <= 0.0
    transition[:, zero_energy] = 0.0
    transition[0, zero_energy] = 1.0
    column_sums = np.sum(transition, axis=0)
    transition /= np.maximum(
        column_sums[np.newaxis, :],
        np.finfo(np.float64).tiny,
    )
    return transition


def _line_order_shapes(
    energy_axis_keV: NDArray[np.float64],
    raw_bin_indices: NDArray[np.int64],
    *,
    maximum_scatter_order: int,
    quadrature_order: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return direct and successive Klein-Nishina line shapes."""
    axis = np.asarray(energy_axis_keV, dtype=np.float64)
    indices = np.asarray(raw_bin_indices, dtype=np.int64)
    direct = np.zeros((indices.size, axis.size), dtype=np.float64)
    direct[np.arange(indices.size), indices] = 1.0
    transition = _klein_nishina_transition_matrix(
        axis,
        quadrature_order=int(quadrature_order),
    )
    current = direct.T
    orders: list[NDArray[np.float64]] = []
    for _ in range(int(maximum_scatter_order)):
        current = transition @ current
        current /= np.maximum(
            np.sum(current, axis=0, keepdims=True),
            np.finfo(np.float64).tiny,
        )
        orders.append(current.T.copy())
    return direct, np.stack(orders, axis=1)


def low_rank_spectral_mean_descriptor_numpy(
    total_xvsl: NDArray[np.float64],
    uncollided_xvsl: NDArray[np.float64],
    features_xvslf: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return rate, line-mixture, and transport descriptors in one batch."""
    total = np.asarray(total_xvsl, dtype=np.float64)
    uncollided = np.asarray(uncollided_xvsl, dtype=np.float64)
    features = np.asarray(features_xvslf, dtype=np.float64)
    if (
        total.ndim < 2
        or uncollided.shape != total.shape
        or features.shape != total.shape + (len(TRANSPORT_FEATURE_ORDER),)
        or np.any(~np.isfinite(total))
        or np.any(total < 0.0)
        or np.any(~np.isfinite(uncollided))
        or np.any(uncollided < 0.0)
        or np.any(~np.isfinite(features))
        or np.any(features < 0.0)
    ):
        raise ValueError("Low-rank descriptor inputs are invalid.")
    line_rates = np.sum(total, axis=-2)
    total_rate = np.sum(line_rates, axis=-1)
    line_fractions = np.divide(
        line_rates,
        total_rate[..., np.newaxis],
        out=np.zeros_like(line_rates),
        where=total_rate[..., np.newaxis] > 0.0,
    )
    uncollided_fraction = np.divide(
        np.sum(uncollided, axis=(-2, -1)),
        total_rate,
        out=np.zeros_like(total_rate),
        where=total_rate > 0.0,
    )
    feature_numerator = np.sum(
        total[..., np.newaxis] * features,
        axis=(-3, -2),
    )
    feature_mean = np.divide(
        feature_numerator,
        total_rate[..., np.newaxis],
        out=np.zeros_like(feature_numerator),
        where=total_rate[..., np.newaxis] > 0.0,
    )
    return np.concatenate(
        (
            np.log1p(total_rate)[..., np.newaxis],
            line_fractions,
            uncollided_fraction[..., np.newaxis],
            feature_mean,
        ),
        axis=-1,
    )


@dataclass
class LowRankSpectralMeanCorrection:
    """Apply a count-preserving correction to conditional spectral marks."""

    descriptor_order: tuple[str, ...]
    descriptor_center_d: NDArray[np.float64]
    descriptor_scale_d: NDArray[np.float64]
    regression_qk: NDArray[np.float64]
    basis_kb: NDArray[np.float64]
    maximum_abs_log_correction: float
    training_manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        """Validate, own, and freeze the learned low-rank arrays."""
        self.descriptor_order = tuple(str(value) for value in self.descriptor_order)
        self.descriptor_center_d = np.ascontiguousarray(
            self.descriptor_center_d,
            dtype=np.float64,
        )
        self.descriptor_scale_d = np.ascontiguousarray(
            self.descriptor_scale_d,
            dtype=np.float64,
        )
        self.regression_qk = np.ascontiguousarray(
            self.regression_qk,
            dtype=np.float64,
        )
        self.basis_kb = np.ascontiguousarray(
            self.basis_kb,
            dtype=np.float64,
        )
        self.training_manifest = _freeze_json_value(dict(self.training_manifest))
        descriptor_count = len(self.descriptor_order)
        rank = int(self.basis_kb.shape[0]) if self.basis_kb.ndim == 2 else 0
        if (
            descriptor_count == 0
            or len(set(self.descriptor_order)) != descriptor_count
            or self.descriptor_center_d.shape != (descriptor_count,)
            or self.descriptor_scale_d.shape != (descriptor_count,)
            or self.regression_qk.shape != (descriptor_count + 1, rank)
            or rank <= 0
            or self.basis_kb.shape[1] <= 1
            or np.any(~np.isfinite(self.descriptor_center_d))
            or np.any(~np.isfinite(self.descriptor_scale_d))
            or np.any(self.descriptor_scale_d <= 0.0)
            or np.any(~np.isfinite(self.regression_qk))
            or np.any(~np.isfinite(self.basis_kb))
            or not np.isfinite(self.maximum_abs_log_correction)
            or not 0.0 < float(self.maximum_abs_log_correction) <= 4.0
        ):
            raise ValueError("Low-rank spectral mean correction is invalid.")
        for array in (
            self.descriptor_center_d,
            self.descriptor_scale_d,
            self.regression_qk,
            self.basis_kb,
        ):
            array.setflags(write=False)
        self._contract_hash_sha256 = self._build_contract_hash()

    @property
    def contract_hash_sha256(self) -> str:
        """Return the immutable correction identity."""
        return self._contract_hash_sha256

    @property
    def training_ready(self) -> bool:
        """Return whether only the designated fixed-quota training was used."""
        manifest = self.training_manifest
        legacy_keys = {
            "schema_version",
            "training_policy",
            "training_scene_seeds",
            "scenario_ids",
            "pair_ids_by_scene",
            "artifact_sha256_by_scene",
            "rank_grid",
            "ridge_lambda_grid",
            "selected_rank",
            "selected_ridge_lambda",
            "selection_objective",
            "selected_validation_score",
            "selection_completed",
            "holdout_artifacts_consumed",
        }
        exact_basis_keys = legacy_keys | {
            "base_additive_response_contract_sha256",
            "feature_basis_semantics",
        }
        policy = manifest.get("training_policy")
        training_seeds = tuple(manifest.get("training_scene_seeds", ()))
        legacy_training = bool(
            policy == "fixed_quota_loso_training_only_low_rank_log_mean_v1"
            and training_seeds == (2026072701, 2026072702)
        )
        randomized_family_training = bool(
            policy == "randomized_geometry_family_loso_low_rank_log_mean_v2"
            and training_seeds == DESIGNATED_TRAINING_SCENE_SEEDS
            and tuple(manifest.get("scenario_ids", ()))
            == tuple(
                scenario
                for scenario in VALIDATION_SCENARIO_IDS
                if scenario != "background_only"
            )
        )
        exact_basis_training = bool(
            policy == "randomized_geometry_family_loso_low_rank_log_mean_v3"
            and manifest.get("schema_version") == 2
            and set(manifest) == exact_basis_keys
            and _is_sha256(manifest.get("base_additive_response_contract_sha256"))
            and manifest.get("feature_basis_semantics")
            == "exactly_one_compton_with_zero_other_los_interactions_v2"
            and training_seeds == DESIGNATED_TRAINING_SCENE_SEEDS
            and tuple(manifest.get("scenario_ids", ()))
            == tuple(
                scenario
                for scenario in VALIDATION_SCENARIO_IDS
                if scenario != "background_only"
            )
        )
        return bool(
            isinstance(manifest, Mapping)
            and set(manifest) in (legacy_keys, exact_basis_keys)
            and (
                (
                    manifest.get("schema_version") == 1
                    and (legacy_training or randomized_family_training)
                )
                or exact_basis_training
            )
            and manifest.get("selection_objective")
            == "leave_one_scene_out_target_probability_weighted_log_mse"
            and manifest.get("selection_completed") is True
            and manifest.get("holdout_artifacts_consumed") is False
            and int(manifest.get("selected_rank", 0)) == self.basis_kb.shape[0]
            and np.isfinite(float(manifest.get("selected_ridge_lambda", np.nan)))
            and np.isfinite(float(manifest.get("selected_validation_score", np.nan)))
            and all(
                _is_sha256(value)
                for value in dict(manifest.get("artifact_sha256_by_scene", {})).values()
            )
        )

    def _build_contract_hash(self) -> str:
        """Hash the correction semantics, training, and numeric arrays."""
        digest = hashlib.sha256()
        digest.update(b"low_rank_spectral_mean_correction_v1")
        digest.update(
            json.dumps(
                {
                    "descriptor_order": list(self.descriptor_order),
                    "maximum_abs_log_correction": float(
                        self.maximum_abs_log_correction
                    ),
                    "training_manifest_sha256": _canonical_json_sha256(
                        self.training_manifest
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        for array in (
            self.descriptor_center_d,
            self.descriptor_scale_d,
            self.regression_qk,
            self.basis_kb,
        ):
            digest.update(_array_digest(array))
        return digest.hexdigest()

    def _descriptor_numpy(
        self,
        total_xvsl: NDArray[np.float64],
        uncollided_xvsl: NDArray[np.float64],
        features_xvslf: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return physical rate, line-mixture, and transport descriptors."""
        descriptor = low_rank_spectral_mean_descriptor_numpy(
            total_xvsl,
            uncollided_xvsl,
            features_xvslf,
        )
        if descriptor.shape[-1] != len(self.descriptor_order):
            raise ValueError("Low-rank correction descriptor width is invalid.")
        return descriptor

    def apply_numpy(
        self,
        marked_source_xvb: NDArray[np.float64],
        total_xvsl: NDArray[np.float64],
        uncollided_xvsl: NDArray[np.float64],
        features_xvslf: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Apply the bounded learned log-mean correction in one batch."""
        marked = np.asarray(marked_source_xvb, dtype=np.float64)
        descriptor = self._descriptor_numpy(total_xvsl, uncollided_xvsl, features_xvslf)
        standardized = (descriptor - self.descriptor_center_d) / self.descriptor_scale_d
        design = np.concatenate(
            (np.ones(standardized.shape[:-1] + (1,)), standardized),
            axis=-1,
        )
        log_correction = np.einsum(
            "...q,qk,kb->...b",
            design,
            self.regression_qk,
            self.basis_kb,
            optimize=True,
        )
        bound = float(self.maximum_abs_log_correction)
        log_correction = np.clip(log_correction, -bound, bound)
        floor = np.sum(marked, axis=-1, keepdims=True) * 1.0e-12
        floor /= float(marked.shape[-1])
        corrected = np.maximum(
            (marked + floor) * np.exp(log_correction),
            0.0,
        )
        marked_total = np.sum(marked, axis=-1, keepdims=True)
        corrected_total = np.sum(corrected, axis=-1, keepdims=True)
        return np.divide(
            corrected * marked_total,
            corrected_total,
            out=np.zeros_like(corrected),
            where=corrected_total > 0.0,
        )

    def apply_torch(
        self,
        marked_source_xvb: object,
        total_xvsl: object,
        uncollided_xvsl: object,
        features_xvslf: object,
    ) -> object:
        """Return the Torch-equivalent corrected source spectrum."""
        import torch

        marked = torch.as_tensor(marked_source_xvb)
        total = torch.as_tensor(total_xvsl, device=marked.device, dtype=marked.dtype)
        uncollided = torch.as_tensor(
            uncollided_xvsl,
            device=marked.device,
            dtype=marked.dtype,
        )
        features = torch.as_tensor(
            features_xvslf,
            device=marked.device,
            dtype=marked.dtype,
        )
        line_rates = torch.sum(total, dim=-2)
        total_rate = torch.sum(line_rates, dim=-1)
        tiny = torch.finfo(marked.dtype).tiny
        line_fractions = torch.where(
            total_rate.unsqueeze(-1) > 0.0,
            line_rates / torch.clamp(total_rate.unsqueeze(-1), min=tiny),
            torch.zeros_like(line_rates),
        )
        uncollided_fraction = torch.where(
            total_rate > 0.0,
            torch.sum(uncollided, dim=(-2, -1)) / torch.clamp(total_rate, min=tiny),
            torch.zeros_like(total_rate),
        )
        feature_numerator = torch.sum(
            total.unsqueeze(-1) * features,
            dim=(-3, -2),
        )
        feature_mean = torch.where(
            total_rate.unsqueeze(-1) > 0.0,
            feature_numerator / torch.clamp(total_rate.unsqueeze(-1), min=tiny),
            torch.zeros_like(feature_numerator),
        )
        descriptor = torch.cat(
            (
                torch.log1p(total_rate).unsqueeze(-1),
                line_fractions,
                uncollided_fraction.unsqueeze(-1),
                feature_mean,
            ),
            dim=-1,
        )
        center = torch.as_tensor(
            np.array(self.descriptor_center_d, copy=True),
            device=marked.device,
            dtype=marked.dtype,
        )
        scale = torch.as_tensor(
            np.array(self.descriptor_scale_d, copy=True),
            device=marked.device,
            dtype=marked.dtype,
        )
        standardized = (descriptor - center) / scale
        design = torch.cat(
            (torch.ones_like(standardized[..., :1]), standardized),
            dim=-1,
        )
        regression = torch.as_tensor(
            np.array(self.regression_qk, copy=True),
            device=marked.device,
            dtype=marked.dtype,
        )
        basis = torch.as_tensor(
            np.array(self.basis_kb, copy=True),
            device=marked.device,
            dtype=marked.dtype,
        )
        log_correction = torch.einsum(
            "...q,qk,kb->...b",
            design,
            regression,
            basis,
        )
        bound = float(self.maximum_abs_log_correction)
        log_correction = torch.clamp(log_correction, min=-bound, max=bound)
        floor = torch.sum(marked, dim=-1, keepdim=True) * 1.0e-12
        floor = floor / float(marked.shape[-1])
        corrected = torch.clamp(
            (marked + floor) * torch.exp(log_correction),
            min=0.0,
        )
        marked_total = torch.sum(marked, dim=-1, keepdim=True)
        corrected_total = torch.sum(corrected, dim=-1, keepdim=True)
        return torch.where(
            corrected_total > 0.0,
            corrected
            * marked_total
            / torch.clamp(corrected_total, min=torch.finfo(marked.dtype).tiny),
            torch.zeros_like(corrected),
        )

    def to_payload(self) -> dict[str, object]:
        """Return the authenticated JSON representation."""
        return {
            "schema_version": 1,
            "model": "geometry_conditioned_low_rank_log_mean_correction_v1",
            "contract_hash_sha256": self.contract_hash_sha256,
            "descriptor_order": list(self.descriptor_order),
            "descriptor_center": self.descriptor_center_d.tolist(),
            "descriptor_scale": self.descriptor_scale_d.tolist(),
            "regression": self.regression_qk.tolist(),
            "basis": self.basis_kb.tolist(),
            "maximum_abs_log_correction": float(self.maximum_abs_log_correction),
            "training_ready": self.training_ready,
            "training": _thaw_json_value(self.training_manifest),
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "LowRankSpectralMeanCorrection":
        """Reconstruct and authenticate one low-rank correction."""
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") != 1
            or payload.get("model")
            != "geometry_conditioned_low_rank_log_mean_correction_v1"
        ):
            raise ValueError("Low-rank correction payload is invalid.")
        correction = cls(
            descriptor_order=tuple(payload.get("descriptor_order", ())),
            descriptor_center_d=np.asarray(
                payload.get("descriptor_center"),
                dtype=np.float64,
            ),
            descriptor_scale_d=np.asarray(
                payload.get("descriptor_scale"),
                dtype=np.float64,
            ),
            regression_qk=np.asarray(payload.get("regression"), dtype=np.float64),
            basis_kb=np.asarray(payload.get("basis"), dtype=np.float64),
            maximum_abs_log_correction=float(
                payload.get("maximum_abs_log_correction", np.nan)
            ),
            training_manifest=(
                payload.get("training")
                if isinstance(payload.get("training"), Mapping)
                else {}
            ),
        )
        if correction.to_payload() != dict(payload):
            raise ValueError("Low-rank correction does not reconstruct exactly.")
        return correction


@dataclass(frozen=True)
class PhysicalComponentDiscrepancy:
    """Define physical count and component-aware mark uncertainty.

    Source-direct, source-scatter, and isotope-independent background marks
    remain separate until their covariance is propagated into the immutable
    energy-partition tree.  This prevents a bright low-energy source from
    making a background-owned high-energy branch spuriously exact.  Detector
    Green finite-corpus covariance is combined with these concentrations by
    the model and is never treated as a fixed response matrix.
    """

    count_uncollided_concentration: float
    count_scatter_concentration: float
    mark_uncollided_concentration: float
    mark_scatter_concentration: float
    mark_background_group_concentration: float
    mark_background_within_concentration: float
    count_scope: str = "view_independent"
    provenance: str = "empirical_training"
    mark_latent_model: str = "component_dirichlet_tree_hierarchical"

    def __post_init__(self) -> None:
        """Validate the component-latent statistical contract."""
        values = (
            self.count_uncollided_concentration,
            self.count_scatter_concentration,
            self.mark_uncollided_concentration,
            self.mark_scatter_concentration,
            self.mark_background_group_concentration,
            self.mark_background_within_concentration,
        )
        if any(not np.isfinite(value) or float(value) <= 0.0 for value in values):
            raise ValueError(
                "Physical-component discrepancy concentrations must be "
                "finite and positive."
            )
        if self.count_scope != "view_independent":
            raise ValueError(
                "Physical-component count discrepancy currently requires "
                "view_independent Gamma latents."
            )
        if self.provenance not in (
            "empirical_training",
            "physics_only_uncertainty_budget_v1",
        ):
            raise ValueError("Physical-component provenance is invalid.")
        if self.mark_latent_model != "component_dirichlet_tree_hierarchical":
            raise ValueError(
                "Physical-component marks require the component-aware "
                "Dirichlet-tree hierarchy."
            )
        if self.provenance == "physics_only_uncertainty_budget_v1" and values != (
            2500.0,
            4.0,
            9999.0,
            23.999999999999996,
            23.999999999999996,
            9999.0,
        ):
            raise ValueError(
                "Physics-only uncertainty concentrations are immutable; "
                "scene-tuned replacements require a different nonproduction "
                "model family."
            )

    def to_payload(self) -> Mapping[str, object]:
        """Return the authenticated JSON representation."""
        payload: dict[str, object] = {
            "schema_version": 5,
            "model": "uncollided_scatter_background_component_latents_v2",
            "count_scope": self.count_scope,
            "count_uncollided_concentration": float(
                self.count_uncollided_concentration
            ),
            "count_scatter_concentration": float(self.count_scatter_concentration),
            "mark_uncollided_concentration": float(self.mark_uncollided_concentration),
            "mark_scatter_concentration": float(self.mark_scatter_concentration),
            "mark_background_group_concentration": float(
                self.mark_background_group_concentration
            ),
            "mark_background_within_concentration": float(
                self.mark_background_within_concentration
            ),
            "fraction_contract": (
                "minimum_total_uncollided_and_total_minus_uncollided"
            ),
            "provenance": self.provenance,
            "mark_latent_model": self.mark_latent_model,
            "mark_latent_scope": "station_view_component_energy_partition_tree",
            "mark_latent_factorization": (
                "beta_binomial_balanced_partition_tree_plus_leaf_dirichlet_multinomial"
            ),
            "photopeak_partition_contract": (
                "detector_response_contiguous_three_sigma_support_v1"
            ),
            "continuum_partition_contract": (
                "fixed_50kev_detector_resolution_bands_v1"
            ),
            "component_covariance_contract": (
                "direct_scatter_background_moment_propagation_v1"
            ),
            "detector_green_finite_mc_contract": (
                "pulse_plus_no_pulse_categorical_covariance_all_tree_levels_v1"
            ),
            "background_mark_contract": (
                "isotope_independent_group_and_within_group_dirichlet_v1"
            ),
        }
        if self.provenance == "physics_only_uncertainty_budget_v1":
            payload.update(
                {
                    "count_uncollided_relative_standard_uncertainty": 0.02,
                    "count_scatter_relative_standard_uncertainty": 0.5,
                    "mark_uncollided_probability_standard_uncertainty": 0.01,
                    "mark_scatter_probability_standard_uncertainty": 0.2,
                    "mark_background_group_probability_standard_uncertainty": 0.2,
                    "mark_background_within_probability_standard_uncertainty": 0.01,
                    "higher_order_scatter_nuisance": (
                        "positive_mean_one_gamma_component"
                    ),
                    "obstacle_material_contract_sha256": (
                        OBSTACLE_MATERIAL_CONTRACT_SHA256
                    ),
                    "transport_physics_table_contract_sha256": (
                        TRANSPORT_PHYSICS_TABLE_CONTRACT_SHA256
                    ),
                }
            )
        return payload

    @classmethod
    def physics_only_budget(cls) -> "PhysicalComponentDiscrepancy":
        """Return the predeclared non-empirical physical uncertainty budget."""
        scatter_concentration = 1.0 / (0.2**2) - 1.0
        sharp_concentration = 1.0 / (0.01**2) - 1.0
        return cls(
            count_uncollided_concentration=1.0 / (0.02**2),
            count_scatter_concentration=1.0 / (0.5**2),
            mark_uncollided_concentration=sharp_concentration,
            mark_scatter_concentration=scatter_concentration,
            mark_background_group_concentration=scatter_concentration,
            mark_background_within_concentration=sharp_concentration,
            provenance="physics_only_uncertainty_budget_v1",
            mark_latent_model="component_dirichlet_tree_hierarchical",
        )

    @property
    def physics_only(self) -> bool:
        """Return whether uncertainty parameters were fixed without scenes."""
        return self.provenance == "physics_only_uncertainty_budget_v1"

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "PhysicalComponentDiscrepancy":
        """Reconstruct one strict physical-component latent contract."""
        expected_keys = {
            "schema_version",
            "model",
            "count_scope",
            "count_uncollided_concentration",
            "count_scatter_concentration",
            "mark_uncollided_concentration",
            "mark_scatter_concentration",
            "mark_background_group_concentration",
            "mark_background_within_concentration",
            "fraction_contract",
            "provenance",
            "mark_latent_model",
            "mark_latent_scope",
            "mark_latent_factorization",
            "photopeak_partition_contract",
            "continuum_partition_contract",
            "component_covariance_contract",
            "detector_green_finite_mc_contract",
            "background_mark_contract",
            "count_uncollided_relative_standard_uncertainty",
            "count_scatter_relative_standard_uncertainty",
            "mark_uncollided_probability_standard_uncertainty",
            "mark_scatter_probability_standard_uncertainty",
            "mark_background_group_probability_standard_uncertainty",
            "mark_background_within_probability_standard_uncertainty",
            "higher_order_scatter_nuisance",
            "obstacle_material_contract_sha256",
            "transport_physics_table_contract_sha256",
        }
        if (
            not isinstance(payload, Mapping)
            or set(payload) != expected_keys
            or payload.get("schema_version") != 5
            or payload.get("model")
            != "uncollided_scatter_background_component_latents_v2"
            or payload.get("fraction_contract")
            != "minimum_total_uncollided_and_total_minus_uncollided"
        ):
            raise ValueError("Physical-component discrepancy payload is invalid.")
        result = cls(
            count_uncollided_concentration=_strict_json_number(
                payload.get("count_uncollided_concentration"),
                field_name="count_uncollided_concentration",
            ),
            count_scatter_concentration=_strict_json_number(
                payload.get("count_scatter_concentration"),
                field_name="count_scatter_concentration",
            ),
            mark_uncollided_concentration=_strict_json_number(
                payload.get("mark_uncollided_concentration"),
                field_name="mark_uncollided_concentration",
            ),
            mark_scatter_concentration=_strict_json_number(
                payload.get("mark_scatter_concentration"),
                field_name="mark_scatter_concentration",
            ),
            mark_background_group_concentration=_strict_json_number(
                payload.get("mark_background_group_concentration"),
                field_name="mark_background_group_concentration",
            ),
            mark_background_within_concentration=_strict_json_number(
                payload.get("mark_background_within_concentration"),
                field_name="mark_background_within_concentration",
            ),
            count_scope=str(payload.get("count_scope")),
            provenance=str(payload.get("provenance")),
            mark_latent_model=str(payload.get("mark_latent_model")),
        )
        if result.to_payload() != dict(payload):
            raise ValueError(
                "Physical-component discrepancy does not reconstruct exactly."
            )
        return result


@dataclass
class GeometryConditionedSpectralModel:
    """Represent the shared source-resolved PF/DSS spectrum distribution."""

    _energy_axis_keV: NDArray[np.float64]
    _line_identity: tuple[Mapping[str, object], ...]
    detector_green_operator: DetectorGreenOperator
    background_shape_b: NDArray[np.float64]
    dead_time_tau_s: float
    background_rate_cps: float
    maximum_scatter_order: int = 5
    klein_nishina_quadrature_order: int = 64
    rate_scale_nodes_j: tuple[float, ...] = (1.0,)
    rate_scale_weights_j: tuple[float, ...] = (1.0,)
    count_discrepancy_concentration: float | None = None
    count_discrepancy_scope: str | None = None
    mark_concentration_source: float | None = None
    mark_concentration_multi_isotope: float | None = None
    physical_component_discrepancy: PhysicalComponentDiscrepancy | None = None
    discrepancy_training_manifest: Mapping[str, object] | None = None
    validation_manifest: Mapping[str, object] | None = None
    additive_scatter_response: (
        AdditiveNoncollidedTransportResponse
        | PhysicsOnlyNoncollidedTransportResponse
        | None
    ) = None
    low_rank_spectral_mean_correction: LowRankSpectralMeanCorrection | None = None
    _torch_cache: dict[tuple[str, str], tuple[object, ...]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _torch_likelihood_cache: dict[tuple[str, str], tuple[object, ...]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _torch_mark_tree_cache: dict[tuple[str, str], tuple[object, ...]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _torch_component_likelihood_cache: dict[
        tuple[str, str],
        tuple[object, ...],
    ] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _torch_cross_state_chunk_cache: dict[tuple[object, ...], int] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    last_torch_cross_chunk_diagnostics: dict[str, object] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _proposal_basis_cache: dict[bytes, NDArray[np.float64]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _photopeak_mask_b: NDArray[np.bool_] = field(
        init=False,
        repr=False,
    )
    _continuum_group_mask_gb: NDArray[np.float64] = field(
        init=False,
        repr=False,
    )
    _mark_leaf_group_mask_hb: NDArray[np.float64] = field(
        init=False,
        repr=False,
    )
    _mark_tree_left_mask_tb: NDArray[np.float64] = field(
        init=False,
        repr=False,
    )
    _mark_tree_right_mask_tb: NDArray[np.float64] = field(
        init=False,
        repr=False,
    )
    _mark_tree_domain_t: NDArray[np.int64] = field(
        init=False,
        repr=False,
    )
    _mark_tree_depth_t: NDArray[np.int64] = field(
        init=False,
        repr=False,
    )
    _mark_tree_left_child_t: NDArray[np.int64] = field(
        init=False,
        repr=False,
    )
    _mark_tree_right_child_t: NDArray[np.int64] = field(
        init=False,
        repr=False,
    )
    response_operator_br: NDArray[np.float64] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Validate and freeze the physical model arrays."""
        self._line_identity = tuple(
            _freeze_json_value(dict(item)) for item in tuple(self._line_identity)
        )
        if not all(isinstance(item, Mapping) for item in self._line_identity):
            raise TypeError("Line identity rows must be mappings.")
        self.validation_manifest = (
            None
            if self.validation_manifest is None
            else _freeze_json_value(dict(self.validation_manifest))
        )
        self.discrepancy_training_manifest = (
            None
            if self.discrepancy_training_manifest is None
            else _freeze_json_value(dict(self.discrepancy_training_manifest))
        )
        if self.additive_scatter_response is not None and not isinstance(
            self.additive_scatter_response,
            (
                AdditiveNoncollidedTransportResponse,
                PhysicsOnlyNoncollidedTransportResponse,
            ),
        ):
            raise TypeError(
                "additive_scatter_response must use the authenticated additive "
                "noncollided schema."
            )
        if self.low_rank_spectral_mean_correction is not None and not isinstance(
            self.low_rank_spectral_mean_correction,
            LowRankSpectralMeanCorrection,
        ):
            raise TypeError(
                "low_rank_spectral_mean_correction must use its authenticated schema."
            )
        component_discrepancy = self.physical_component_discrepancy
        if component_discrepancy is not None and not isinstance(
            component_discrepancy,
            PhysicalComponentDiscrepancy,
        ):
            raise TypeError(
                "physical_component_discrepancy must use its authenticated schema."
            )
        self._discrepancy_training_manifest_sha256 = (
            None
            if self.discrepancy_training_manifest is None
            else _canonical_json_sha256(self.discrepancy_training_manifest)
        )
        self._validation_manifest_sha256 = (
            None
            if self.validation_manifest is None
            else _canonical_json_sha256(self.validation_manifest)
        )
        self._energy_axis_keV = np.ascontiguousarray(
            self._energy_axis_keV,
            dtype=np.float64,
        )
        if not isinstance(self.detector_green_operator, DetectorGreenOperator):
            raise TypeError(
                "Geometry-conditioned spectra require a detector Green operator."
            )
        self.detector_green_operator.require_runtime_ready()
        if (
            int(self.detector_green_operator.impact_parameter_edges_fraction.size - 1)
            != DETECTOR_IMPACT_PHASE_COUNT
        ):
            raise ValueError(
                "The detector Green impact partition does not match the "
                "transport feature contract."
            )
        (
            response_operator,
            conditional_response_concentration,
            response_concentration,
        ) = _detector_green_model_response_bundle(
            self.detector_green_operator,
            self._energy_axis_keV,
        )
        self.response_operator_br = np.ascontiguousarray(
            response_operator,
            dtype=np.float64,
        )
        self._response_concentration_r = np.ascontiguousarray(
            conditional_response_concentration,
            dtype=np.float64,
        )
        self._absolute_response_concentration_r = np.ascontiguousarray(
            response_concentration,
            dtype=np.float64,
        )
        self.background_shape_b = np.ascontiguousarray(
            self.background_shape_b,
            dtype=np.float64,
        )
        line_count = len(self._line_identity)
        bin_count = int(self._energy_axis_keV.size)
        if (
            bin_count < 2
            or line_count == 0
            or self.detector_green_operator.output_energy_min_keV
            != float(self._energy_axis_keV[0])
            or not np.isclose(
                self.detector_green_operator.output_bin_width_keV,
                float(self._energy_axis_keV[1] - self._energy_axis_keV[0]),
                rtol=0.0,
                atol=0.0,
            )
            or self.response_operator_br.shape != (bin_count, bin_count)
            or self._response_concentration_r.shape != (bin_count,)
            or self._absolute_response_concentration_r.shape != (bin_count,)
            or self.background_shape_b.shape != (bin_count,)
            or np.any(~np.isfinite(self._energy_axis_keV))
            or np.any(np.diff(self._energy_axis_keV) <= 0.0)
            or np.any(~np.isfinite(self.response_operator_br))
            or np.any(self.response_operator_br < 0.0)
            or np.any(~np.isfinite(self._response_concentration_r))
            or np.any(self._response_concentration_r <= 0.0)
            or np.any(~np.isfinite(self._absolute_response_concentration_r))
            or np.any(self._absolute_response_concentration_r <= 0.0)
            or np.any(~np.isfinite(self.background_shape_b))
            or np.any(self.background_shape_b < 0.0)
        ):
            raise ValueError("Geometry-conditioned spectrum arrays are invalid.")
        response_column_sums = np.sum(self.response_operator_br, axis=0)
        if np.any(response_column_sums < 0.0) or np.any(
            response_column_sums > 1.0 + 1.0e-12
        ):
            raise ValueError(
                "Absolute detector-response columns must be sub-probabilities."
            )
        if not np.isclose(
            np.sum(self.background_shape_b),
            1.0,
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise ValueError("Background mark probabilities must sum to one.")
        correction = self.low_rank_spectral_mean_correction
        if correction is not None and (
            len(correction.descriptor_order)
            != 2 + line_count + len(TRANSPORT_FEATURE_ORDER)
            or correction.basis_kb.shape[1] != bin_count
        ):
            raise ValueError(
                "Low-rank spectral mean correction dimensions do not match "
                "the physical spectrum model."
            )
        if (
            not np.isfinite(self.dead_time_tau_s)
            or self.dead_time_tau_s < 0.0
            or not np.isfinite(self.background_rate_cps)
            or self.background_rate_cps < 0.0
            or int(self.maximum_scatter_order) < 1
            or int(self.klein_nishina_quadrature_order) < 8
        ):
            raise ValueError("Spectrum scalar physical parameters are invalid.")
        nodes = np.asarray(self.rate_scale_nodes_j, dtype=np.float64)
        weights = np.asarray(self.rate_scale_weights_j, dtype=np.float64)
        if (
            nodes.ndim != 1
            or nodes.size == 0
            or weights.shape != nodes.shape
            or np.any(~np.isfinite(nodes))
            or np.any(nodes <= 0.0)
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
            or not np.isclose(np.sum(weights), 1.0, atol=1.0e-12)
            or not np.isclose(
                np.sum(nodes * weights),
                1.0,
                atol=1.0e-12,
            )
        ):
            raise ValueError(
                "Rate-scale mixture must be positive, normalized, and mean one."
            )
        concentration = self.mark_concentration_source
        if concentration is not None and (
            not np.isfinite(concentration) or float(concentration) <= 0.0
        ):
            raise ValueError(
                "mark_concentration_source must be positive when configured."
            )
        multi_concentration = self.mark_concentration_multi_isotope
        if multi_concentration is not None and (
            concentration is None
            or not np.isfinite(multi_concentration)
            or float(multi_concentration) <= 0.0
            or float(multi_concentration) < float(concentration)
        ):
            raise ValueError(
                "mark_concentration_multi_isotope must be finite, positive, "
                "and no smaller than mark_concentration_source."
            )
        count_concentration = self.count_discrepancy_concentration
        if count_concentration is not None and (
            not np.isfinite(count_concentration)
            or float(count_concentration) <= 0.0
            or not np.array_equal(nodes, np.asarray((1.0,), dtype=np.float64))
            or not np.array_equal(weights, np.asarray((1.0,), dtype=np.float64))
        ):
            raise ValueError(
                "count_discrepancy_concentration requires a positive "
                "value and no additional discrete rate-scale mixture."
            )
        if (count_concentration is None) != (self.count_discrepancy_scope is None):
            raise ValueError(
                "Count discrepancy concentration and scope must be configured together."
            )
        if self.count_discrepancy_scope not in (
            None,
            "station_shared",
            "view_independent",
        ):
            raise ValueError("Count discrepancy scope is invalid.")
        if component_discrepancy is not None and (
            count_concentration is not None
            or self.count_discrepancy_scope is not None
            or self.mark_concentration_source is not None
            or self.mark_concentration_multi_isotope is not None
            or not np.array_equal(nodes, np.asarray((1.0,), dtype=np.float64))
            or not np.array_equal(weights, np.asarray((1.0,), dtype=np.float64))
        ):
            raise ValueError(
                "Physical-component discrepancy cannot be combined with the "
                "retired global discrepancy or rate-scale mixture."
            )
        physics_response = isinstance(
            self.additive_scatter_response,
            PhysicsOnlyNoncollidedTransportResponse,
        )
        if physics_response and (
            self.low_rank_spectral_mean_correction is not None
            or self.discrepancy_training_manifest is not None
            or component_discrepancy is None
            or not component_discrepancy.physics_only
        ):
            raise ValueError(
                "Physics-only transport requires its predeclared physical "
                "uncertainty budget and forbids trained mean/discrepancy terms."
            )
        if (
            not physics_response
            and component_discrepancy is not None
            and component_discrepancy.physics_only
        ):
            raise ValueError(
                "Physics-only uncertainty cannot accompany a fitted response."
            )
        self._rate_scale_nodes_j = np.ascontiguousarray(nodes)
        self._rate_scale_weights_j = np.ascontiguousarray(weights)
        isotope_names = tuple(
            sorted({str(item["isotope"]) for item in self._line_identity})
        )
        self._mark_isotope_names = isotope_names
        self._line_to_mark_isotope_li = np.ascontiguousarray(
            np.asarray(
                [
                    [float(str(row["isotope"]) == isotope) for isotope in isotope_names]
                    for row in self._line_identity
                ],
                dtype=np.float64,
            )
        )
        energies = np.asarray(
            [float(item["energy_keV"]) for item in self._line_identity],
            dtype=np.float64,
        )
        self._line_energies_keV_l = np.ascontiguousarray(energies)
        raw_indices = np.asarray(
            [int(item["raw_bin_index"]) for item in self._line_identity],
            dtype=np.int64,
        )
        if np.any(raw_indices < 0) or np.any(raw_indices >= bin_count):
            raise ValueError("Transport-line raw bins are outside the energy axis.")
        if physics_response:
            direct = np.zeros((line_count, bin_count), dtype=np.float64)
            direct[np.arange(line_count), raw_indices] = 1.0
            scatter = np.empty((line_count, 0, bin_count), dtype=np.float64)
        else:
            direct, scatter = _line_order_shapes(
                self._energy_axis_keV,
                raw_indices,
                maximum_scatter_order=int(self.maximum_scatter_order),
                quadrature_order=int(self.klein_nishina_quadrature_order),
            )
        self._direct_line_shapes_lb = direct
        self._scatter_order_shapes_lob = scatter
        unique_energies, inverse_energy = np.unique(
            energies,
            return_inverse=True,
        )
        phase_response_cbu, phase_concentration_cu = (
            self.detector_green_operator.phase_absolute_response_for_axis(
                unique_energies
            )
        )
        _, phase_conditional_concentration_cu = (
            self.detector_green_operator.phase_response_for_axis(unique_energies)
        )
        radius = float(self.detector_green_operator.detector_target_radius_m)
        if radius <= 0.0 or radius >= 1.0:
            raise ValueError(
                "Detector Green reference normalization requires a housing "
                "radius strictly between zero and one metre."
            )
        edges = self.detector_green_operator.impact_parameter_edges_fraction
        ratio = radius
        lower_cosine = np.sqrt(np.maximum(1.0 - np.square(ratio * edges[:-1]), 0.0))
        upper_cosine = np.sqrt(np.maximum(1.0 - np.square(ratio * edges[1:]), 0.0))
        reference_phase_weights = lower_cosine - upper_cosine
        reference_phase_weights /= np.sum(reference_phase_weights)
        phase_detection_cu = np.sum(phase_response_cbu, axis=1)
        line_phase_detection_cl = phase_detection_cu[:, inverse_energy]
        line_phase_detection_variance_cl = (
            line_phase_detection_cl
            * (1.0 - line_phase_detection_cl)
            / (phase_concentration_cu[:, inverse_energy] + 1.0)
        )
        line_reference_detection_l = np.einsum(
            "c,cl->l",
            reference_phase_weights,
            line_phase_detection_cl,
            optimize=True,
        )
        branching_l = np.asarray(
            [float(item["branching_weight"]) for item in self._line_identity],
            dtype=np.float64,
        )
        reference_efficiency_l = np.empty(line_count, dtype=np.float64)
        reference_efficiency_std_l = np.empty(line_count, dtype=np.float64)
        for isotope in isotope_names:
            isotope_mask = np.asarray(
                [str(item["isotope"]) == isotope for item in self._line_identity],
                dtype=np.bool_,
            )
            isotope_branching = branching_l[isotope_mask]
            if np.any(isotope_branching <= 0.0) or not np.isclose(
                np.sum(isotope_branching),
                1.0,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise ValueError(
                    f"Catalog branching weights are invalid for {isotope!r}."
                )
            efficiency = float(
                np.dot(
                    isotope_branching,
                    line_reference_detection_l[isotope_mask],
                )
            )
            if not np.isfinite(efficiency) or efficiency <= 0.0:
                raise ValueError(
                    f"Catalog-weighted detector efficiency is zero for {isotope!r}."
                )
            reference_efficiency_l[isotope_mask] = efficiency
            # A perfect-positive-correlation bound remains conservative when
            # nearby line energies share interpolation nodes.
            reference_std = float(
                np.sum(
                    isotope_branching[np.newaxis, :]
                    * reference_phase_weights[:, np.newaxis]
                    * np.sqrt(line_phase_detection_variance_cl[:, isotope_mask])
                )
            )
            reference_efficiency_std_l[isotope_mask] = reference_std
        self._source_rate_reference_efficiency_l = np.ascontiguousarray(
            reference_efficiency_l,
            dtype=np.float64,
        )
        self._line_reference_detection_efficiency_l = np.ascontiguousarray(
            line_reference_detection_l,
            dtype=np.float64,
        )
        self._source_rate_reference_efficiency_std_l = np.ascontiguousarray(
            reference_efficiency_std_l,
            dtype=np.float64,
        )
        if physics_response:
            if not isinstance(
                self.additive_scatter_response,
                PhysicsOnlyNoncollidedTransportResponse,
            ):
                raise RuntimeError("Physics-only scatter response identity was lost.")
            detector_cone_scatter = build_detector_cone_scatter_grid(
                operator=self.detector_green_operator,
                incident_energies_keV=energies,
                source_reference_efficiencies=reference_efficiency_l,
                fixed_scatter_distances_m=(
                    self.additive_scatter_response.fe_scatter_distance_m,
                    self.additive_scatter_response.pb_scatter_distance_m,
                ),
            )
            self._detector_cone_scatter_contract = (
                detector_cone_scatter.contract_payload()
            )
            self._detector_cone_scatter_distance_nodes_d = np.ascontiguousarray(
                detector_cone_scatter.distance_nodes_m,
                dtype=np.float64,
            )
            self._marked_detector_cone_scatter_shapes_dlb = np.ascontiguousarray(
                detector_cone_scatter.marked_response_dlb,
                dtype=np.float64,
            )
            self._detector_cone_scatter_response_concentration_dl = (
                np.ascontiguousarray(
                    detector_cone_scatter.effective_histories_dl,
                    dtype=np.float64,
                )
            )
        else:
            self._detector_cone_scatter_contract = None
            self._detector_cone_scatter_distance_nodes_d = np.empty(
                0,
                dtype=np.float64,
            )
            self._marked_detector_cone_scatter_shapes_dlb = np.empty(
                (0, line_count, bin_count),
                dtype=np.float64,
            )
            self._detector_cone_scatter_response_concentration_dl = np.empty(
                (0, line_count),
                dtype=np.float64,
            )
        direct_factor_cl = (
            line_phase_detection_cl / reference_efficiency_l[np.newaxis, :]
        )
        numerator_relative_std_cl = np.divide(
            np.sqrt(line_phase_detection_variance_cl),
            line_phase_detection_cl,
            out=np.zeros_like(line_phase_detection_cl),
            where=line_phase_detection_cl > 0.0,
        )
        reference_relative_std_l = reference_efficiency_std_l / reference_efficiency_l
        self._direct_detection_factor_cl = np.ascontiguousarray(
            direct_factor_cl,
            dtype=np.float64,
        )
        self._direct_detection_factor_std_cl = np.ascontiguousarray(
            direct_factor_cl
            * (numerator_relative_std_cl + reference_relative_std_l[np.newaxis, :]),
            dtype=np.float64,
        )
        self._marked_direct_line_shapes_clb = np.ascontiguousarray(
            np.transpose(
                phase_response_cbu[:, :, inverse_energy],
                (0, 2, 1),
            )
            / reference_efficiency_l[np.newaxis, :, np.newaxis],
            dtype=np.float64,
        )
        self._direct_response_concentration_cl = np.ascontiguousarray(
            phase_conditional_concentration_cu[:, inverse_energy],
            dtype=np.float64,
        )
        self._direct_absolute_response_concentration_cl = np.ascontiguousarray(
            phase_concentration_cu[:, inverse_energy],
            dtype=np.float64,
        )
        self._marked_direct_line_shapes_lb = np.einsum(
            "c,clb->lb",
            reference_phase_weights,
            self._marked_direct_line_shapes_clb,
            optimize=True,
        )
        self._marked_scatter_order_shapes_lob = (
            np.einsum(
                "br,lor->lob",
                self.response_operator_br,
                scatter,
                # This array participates in the authenticated model digest.
                # Keep NumPy's fixed contraction order instead of delegating
                # the reduction schedule to a host-specific BLAS kernel.
                optimize=False,
            )
            / reference_efficiency_l[:, np.newaxis, np.newaxis]
        )
        marginal_detection_r = np.sum(self.response_operator_br, axis=0)
        marginal_detection_variance_r = (
            marginal_detection_r
            * (1.0 - marginal_detection_r)
            / (self._absolute_response_concentration_r + 1.0)
        )
        scatter_detection_factor_lo = (
            np.einsum(
                "lor,r->lo",
                scatter,
                marginal_detection_r,
                optimize=True,
            )
            / reference_efficiency_l[:, np.newaxis]
        )
        scatter_numerator_std_lo = (
            np.einsum(
                "lor,r->lo",
                scatter,
                np.sqrt(marginal_detection_variance_r),
                optimize=True,
            )
            / reference_efficiency_l[:, np.newaxis]
        )
        self._scatter_detection_factor_lo = np.ascontiguousarray(
            scatter_detection_factor_lo,
            dtype=np.float64,
        )
        self._scatter_detection_factor_std_lo = np.ascontiguousarray(
            scatter_numerator_std_lo
            + scatter_detection_factor_lo * reference_relative_std_l[:, np.newaxis],
            dtype=np.float64,
        )
        detector_cone_detection_factor_dl = np.sum(
            self._marked_detector_cone_scatter_shapes_dlb,
            axis=-1,
        )
        detector_cone_absolute_detection_dl = (
            detector_cone_detection_factor_dl * reference_efficiency_l[np.newaxis, :]
        )
        detector_cone_detection_std_dl = np.divide(
            np.sqrt(
                np.maximum(
                    detector_cone_absolute_detection_dl
                    * (1.0 - detector_cone_absolute_detection_dl),
                    0.0,
                )
                / (self._detector_cone_scatter_response_concentration_dl + 1.0)
            ),
            reference_efficiency_l[np.newaxis, :],
            out=np.zeros_like(detector_cone_detection_factor_dl),
            where=reference_efficiency_l[np.newaxis, :] > 0.0,
        )
        self._detector_cone_scatter_detection_factor_dl = np.ascontiguousarray(
            detector_cone_detection_factor_dl,
            dtype=np.float64,
        )
        self._detector_cone_scatter_detection_factor_std_dl = np.ascontiguousarray(
            detector_cone_detection_std_dl
            + detector_cone_detection_factor_dl
            * reference_relative_std_l[np.newaxis, :],
            dtype=np.float64,
        )
        active_scatter_energy = scatter > np.finfo(np.float64).tiny
        self._scatter_response_concentration_lo = np.min(
            np.where(
                active_scatter_energy,
                self._response_concentration_r[np.newaxis, np.newaxis, :],
                np.inf,
            ),
            axis=-1,
        )
        self._direct_response_concentration_l = np.min(
            self._direct_response_concentration_cl,
            axis=0,
        )
        self._scatter_response_concentration_l = (
            np.min(
                self._detector_cone_scatter_response_concentration_dl,
                axis=0,
            )
            if physics_response
            else np.min(
                self._scatter_response_concentration_lo,
                axis=1,
            )
        )
        photopeak_mask = np.zeros(bin_count, dtype=np.bool_)
        three_sigma_relative_height = math.exp(-4.5)
        for line_index, identity in enumerate(self._line_identity):
            shape = self._marked_direct_line_shapes_lb[line_index]
            peak = int(np.argmax(shape))
            threshold = float(shape[peak]) * three_sigma_relative_height
            lower = peak
            upper = peak
            while lower > 0 and float(shape[lower - 1]) >= threshold:
                lower -= 1
            while upper + 1 < bin_count and float(shape[upper + 1]) >= threshold:
                upper += 1
            photopeak_mask[lower : upper + 1] = True
        if not np.any(photopeak_mask) or np.all(photopeak_mask):
            raise RuntimeError(
                "Detector response did not define a valid peak/continuum partition."
            )
        photopeak_mask.setflags(write=False)
        self._photopeak_mask_b = photopeak_mask
        raw_continuum_groups = np.floor(
            (self._energy_axis_keV - float(self._energy_axis_keV[0]))
            / CONTINUUM_NUISANCE_BAND_WIDTH_KEV
        ).astype(np.int64)
        active_group_ids = np.unique(raw_continuum_groups[~photopeak_mask])
        continuum_group_mask = np.asarray(
            [
                (~photopeak_mask) & (raw_continuum_groups == group_id)
                for group_id in active_group_ids
            ],
            dtype=np.float64,
        )
        if (
            continuum_group_mask.ndim != 2
            or continuum_group_mask.shape[1] != bin_count
            or np.any(np.sum(continuum_group_mask, axis=0) > 1.0)
            or not np.array_equal(
                np.sum(continuum_group_mask, axis=0) > 0.0,
                ~photopeak_mask,
            )
        ):
            raise RuntimeError("Detector-resolution continuum grouping is invalid.")
        continuum_group_mask.setflags(write=False)
        self._continuum_group_mask_gb = continuum_group_mask
        (
            self._mark_leaf_group_mask_hb,
            self._mark_tree_left_mask_tb,
            self._mark_tree_right_mask_tb,
            self._mark_tree_domain_t,
            self._mark_tree_depth_t,
            self._mark_tree_left_child_t,
            self._mark_tree_right_child_t,
        ) = _build_mark_partition_tree(
            self._photopeak_mask_b,
            self._continuum_group_mask_gb,
        )
        sigma_kn = _klein_nishina_total_cross_section_cm2(energies)
        self._air_mu_compton_l = (
            AIR_DENSITY_G_CM3
            * AVOGADRO_CONSTANT_MOL_INV
            * AIR_EFFECTIVE_Z_OVER_A
            * sigma_kn
        )
        self._fe_compton_fraction_l = self._material_compton_fraction(
            sigma_kn,
            density_g_cm3=IRON_DENSITY_G_CM3,
            z_over_a=IRON_Z_OVER_A,
            material_key="mu_fe_cm_inv",
        )
        self._pb_compton_fraction_l = self._material_compton_fraction(
            sigma_kn,
            density_g_cm3=LEAD_DENSITY_G_CM3,
            z_over_a=LEAD_Z_OVER_A,
            material_key="mu_pb_cm_inv",
        )
        self._obstacle_compton_fraction_l = np.ones(
            line_count,
            dtype=np.float64,
        )
        for array in (
            self._energy_axis_keV,
            self.response_operator_br,
            self.background_shape_b,
            self._direct_line_shapes_lb,
            self._scatter_order_shapes_lob,
            self._marked_direct_line_shapes_lb,
            self._marked_direct_line_shapes_clb,
            self._direct_response_concentration_cl,
            self._direct_response_concentration_l,
            self._marked_scatter_order_shapes_lob,
            self._detector_cone_scatter_distance_nodes_d,
            self._marked_detector_cone_scatter_shapes_dlb,
            self._detector_cone_scatter_response_concentration_dl,
            self._detector_cone_scatter_detection_factor_dl,
            self._detector_cone_scatter_detection_factor_std_dl,
            self._scatter_response_concentration_lo,
            self._scatter_response_concentration_l,
            self._response_concentration_r,
            self._absolute_response_concentration_r,
            self._direct_absolute_response_concentration_cl,
            self._source_rate_reference_efficiency_l,
            self._line_reference_detection_efficiency_l,
            self._source_rate_reference_efficiency_std_l,
            self._direct_detection_factor_cl,
            self._direct_detection_factor_std_cl,
            self._scatter_detection_factor_lo,
            self._scatter_detection_factor_std_lo,
            self._air_mu_compton_l,
            self._fe_compton_fraction_l,
            self._pb_compton_fraction_l,
            self._obstacle_compton_fraction_l,
            self._line_energies_keV_l,
            self._rate_scale_nodes_j,
            self._rate_scale_weights_j,
            self._line_to_mark_isotope_li,
            self._mark_leaf_group_mask_hb,
            self._mark_tree_left_mask_tb,
            self._mark_tree_right_mask_tb,
            self._mark_tree_domain_t,
            self._mark_tree_depth_t,
            self._mark_tree_left_child_t,
            self._mark_tree_right_child_t,
        ):
            array.setflags(write=False)
        self._catalog_independent_contract_hash_sha256 = (
            self._build_catalog_independent_contract_hash()
        )
        self._contract_hash_sha256 = self._build_contract_hash()

    @classmethod
    def physics_only_native(
        cls,
        isotopes: Sequence[str],
        *,
        dead_time_tau_s: float,
        background_rate_cps: float,
        detector_green_operator: DetectorGreenOperator,
        validation_manifest: Mapping[str, object] | None = None,
    ) -> GeometryConditionedSpectralModel:
        """Build the sole model family accepted by production live runtime."""
        operator = detector_green_operator
        operator.require_runtime_ready()
        physical_transport = PhysicsOnlyNoncollidedTransportResponse(
            detector_radius_m=operator.detector_target_radius_m,
            fe_scatter_distance_m=(
                DEFAULT_FE_SHIELD_INNER_RADIUS_CM + 0.5 * DEFAULT_FE_SHIELD_THICKNESS_CM
            )
            / 100.0,
            pb_scatter_distance_m=(
                DEFAULT_PB_SHIELD_INNER_RADIUS_CM + 0.5 * DEFAULT_PB_SHIELD_THICKNESS_CM
            )
            / 100.0,
        )
        return cls.nonproduction_native(
            isotopes,
            dead_time_tau_s=dead_time_tau_s,
            background_rate_cps=background_rate_cps,
            physical_component_discrepancy=(
                PhysicalComponentDiscrepancy.physics_only_budget()
            ),
            validation_manifest=validation_manifest,
            additive_scatter_response=physical_transport,
            detector_green_operator=operator,
        )

    @classmethod
    def nonproduction_native(
        cls,
        isotopes: Sequence[str],
        *,
        dead_time_tau_s: float,
        background_rate_cps: float,
        rate_scale_nodes_j: Sequence[float] = (1.0,),
        rate_scale_weights_j: Sequence[float] = (1.0,),
        count_discrepancy_concentration: float | None = None,
        count_discrepancy_scope: str | None = None,
        mark_concentration_source: float | None = None,
        mark_concentration_multi_isotope: float | None = None,
        physical_component_discrepancy: (PhysicalComponentDiscrepancy | None) = None,
        discrepancy_training_manifest: Mapping[str, object] | None = None,
        validation_manifest: Mapping[str, object] | None = None,
        additive_scatter_response: (
            AdditiveNoncollidedTransportResponse
            | PhysicsOnlyNoncollidedTransportResponse
            | None
        ) = None,
        low_rank_spectral_mean_correction: (
            LowRankSpectralMeanCorrection | None
        ) = None,
        detector_green_operator: DetectorGreenOperator | None = None,
    ) -> GeometryConditionedSpectralModel:
        """Build a research model that production loaders never infer."""
        isotope_order = tuple(sorted(str(value) for value in isotopes))
        if not isotope_order or len(set(isotope_order)) != len(isotope_order):
            raise ValueError("Spectrum model isotopes must be nonempty and unique.")
        bin_width = float(NATIVE_GEANT4_BIN_WIDTH_KEV)
        energy_axis = np.arange(NATIVE_GEANT4_BIN_COUNT, dtype=np.float64) * bin_width
        library = default_library()
        shield_lines = line_resolved_shield_mu_by_isotope(
            isotope_order,
            normalize_line_intensities=True,
        )
        line_identity: list[dict[str, object]] = []
        for isotope in isotope_order:
            nuclide = library.get(isotope)
            if nuclide is None:
                raise KeyError(f"Missing physical line library for {isotope!r}.")
            positive_lines = [
                line for line in nuclide.lines if float(line.intensity) > 0.0
            ]
            isotope_shield_lines = shield_lines.get(isotope, ())
            if len(isotope_shield_lines) != len(positive_lines):
                raise RuntimeError(
                    f"Shield and spectrum line libraries disagree for {isotope!r}."
                )
            total_weight = sum(float(line.intensity) for line in positive_lines)
            for local_index, line in enumerate(positive_lines):
                shield_entry = isotope_shield_lines[local_index]
                raw_bin = int(
                    np.floor(
                        (float(line.energy_keV) - float(energy_axis[0])) / bin_width
                    )
                )
                line_identity.append(
                    {
                        "isotope": isotope,
                        "transport_line_index": int(local_index),
                        "energy_keV": float(line.energy_keV),
                        "branching_weight": (
                            float(line.intensity) / float(total_weight)
                        ),
                        "raw_bin_index": raw_bin,
                        "raw_bin_energy_keV": float(energy_axis[raw_bin]),
                        "mu_fe_cm_inv": float(shield_entry["fe"]),
                        "mu_pb_cm_inv": float(shield_entry["pb"]),
                    }
                )
        operator = (
            DetectorGreenOperator.from_artifact(
                DEFAULT_DETECTOR_GREEN_OPERATOR_MANIFEST
            )
            if detector_green_operator is None
            else detector_green_operator
        )
        operator.require_runtime_ready()
        operator.validate_catalog_profile(isotope_order, library=library)
        background_shape = native_geant4_background_shape(
            energy_axis,
            bin_width,
        )
        return cls(
            _energy_axis_keV=energy_axis,
            _line_identity=tuple(line_identity),
            detector_green_operator=operator,
            background_shape_b=background_shape,
            dead_time_tau_s=float(dead_time_tau_s),
            background_rate_cps=float(background_rate_cps),
            rate_scale_nodes_j=tuple(float(value) for value in rate_scale_nodes_j),
            rate_scale_weights_j=tuple(float(value) for value in rate_scale_weights_j),
            count_discrepancy_concentration=(
                None
                if count_discrepancy_concentration is None
                else float(count_discrepancy_concentration)
            ),
            count_discrepancy_scope=(
                None
                if count_discrepancy_scope is None
                else str(count_discrepancy_scope)
            ),
            mark_concentration_source=(
                None
                if mark_concentration_source is None
                else float(mark_concentration_source)
            ),
            mark_concentration_multi_isotope=(
                None
                if mark_concentration_multi_isotope is None
                else float(mark_concentration_multi_isotope)
            ),
            physical_component_discrepancy=physical_component_discrepancy,
            discrepancy_training_manifest=discrepancy_training_manifest,
            validation_manifest=validation_manifest,
            additive_scatter_response=additive_scatter_response,
            low_rank_spectral_mean_correction=(low_rank_spectral_mean_correction),
        )

    @classmethod
    def from_manifest_payload(
        cls,
        payload: Mapping[str, object],
        *,
        detector_green_operator: DetectorGreenOperator,
    ) -> GeometryConditionedSpectralModel:
        """Reconstruct and authenticate one runtime-ready schema-v7 model."""
        if not isinstance(payload, Mapping):
            raise TypeError("Full-spectrum model manifest must be a mapping.")
        if (
            payload.get("schema_version") != FULL_SPECTRUM_MODEL_SCHEMA_VERSION
            or payload.get("model") != "geometry_conditioned_full_spectrum"
        ):
            raise ValueError(
                "Runtime requires a geometry-conditioned schema-v7 spectrum manifest."
            )
        line_rows = payload.get("line_identity")
        mixture = payload.get("rate_scale_mixture")
        if (
            not isinstance(line_rows, Sequence)
            or isinstance(line_rows, (str, bytes))
            or not line_rows
            or not all(isinstance(row, Mapping) for row in line_rows)
            or not isinstance(mixture, Mapping)
            or set(mixture) != {"scope", "nodes", "weights", "weighted_mean"}
            or mixture.get("scope") != "station_shared_source_only"
        ):
            raise ValueError(
                "Full-spectrum manifest line or discrepancy identity is invalid."
            )
        raw_isotopes = tuple(row.get("isotope") for row in line_rows)
        if any(not isinstance(value, str) or not value for value in raw_isotopes):
            raise ValueError("Full-spectrum manifest requires nonempty line isotopes.")
        isotope_order = tuple(sorted(set(raw_isotopes)))
        additive_payload = payload.get("additive_noncollided_transport_response")
        if not isinstance(additive_payload, Mapping):
            raise ValueError(
                "Schema-v7 full-spectrum manifests require the authenticated "
                "additive noncollided transport response."
            )
        mixture_nodes = _strict_json_number_sequence(
            mixture.get("nodes"),
            field_name="rate_scale_mixture.nodes",
        )
        mixture_weights = _strict_json_number_sequence(
            mixture.get("weights"),
            field_name="rate_scale_mixture.weights",
        )
        mixture_mean = _strict_json_number(
            mixture.get("weighted_mean"),
            field_name="rate_scale_mixture.weighted_mean",
        )
        if mixture_nodes != (1.0,) or mixture_weights != (1.0,) or mixture_mean != 1.0:
            raise ValueError("Production schema-v7 forbids rate-scale mixtures.")
        dead_time_tau_s = _strict_json_number(
            payload.get("dead_time_tau_s"),
            field_name="dead_time_tau_s",
        )
        background_rate_cps = _strict_json_number(
            payload.get("background_rate_cps"),
            field_name="background_rate_cps",
        )
        physical_component_payload = payload.get("physical_component_discrepancy")
        physical_component_discrepancy = (
            PhysicalComponentDiscrepancy.from_payload(physical_component_payload)
            if isinstance(physical_component_payload, Mapping)
            else None
        )
        response_model_id = additive_payload.get("model")
        if response_model_id != PHYSICS_ONLY_TRANSPORT_RESPONSE_ID:
            raise ValueError(
                "Production schema-v7 forbids scene-fitted and legacy "
                "transport responses."
            )
        if (
            any(
                payload.get(field_name) is not None
                for field_name in (
                    "discrepancy_training",
                    "discrepancy_training_manifest_sha256",
                    "low_rank_spectral_mean_correction",
                    "count_discrepancy_concentration",
                    "count_discrepancy_scope",
                    "mark_concentration_multi_isotope",
                )
            )
            or payload.get("mark_concentration_source") is not None
        ):
            raise ValueError(
                "Production schema-v7 forbids trained, global, and low-rank "
                "response corrections."
            )
        additive_response = PhysicsOnlyNoncollidedTransportResponse.from_payload(
            additive_payload
        )
        if (
            physical_component_discrepancy is None
            or not physical_component_discrepancy.physics_only
            or physical_component_discrepancy.mark_latent_model
            != "component_dirichlet_tree_hierarchical"
            or payload.get("mark_model")
            != "component_background_source_dirichlet_tree_hierarchical"
            or payload.get("scatter_shape") != DETECTOR_CONE_SCATTER_RESPONSE_ID
            or payload.get("higher_order_scatter_mean")
            != "excluded_positive_nuisance_owned_by_likelihood"
            or not isinstance(
                payload.get("detector_cone_scatter_response"),
                Mapping,
            )
            or payload.get("detector_green_phase_conditioning")
            != (
                "transport_resolved_direct_impact_and_detector_cone_"
                "scatter_joint_state_v3"
            )
            or "maximum_scatter_order" in payload
        ):
            raise ValueError(
                "Production schema-v7 requires the generic physics-only "
                "transport uncertainty contract."
            )
        model = cls.physics_only_native(
            isotope_order,
            dead_time_tau_s=dead_time_tau_s,
            background_rate_cps=background_rate_cps,
            validation_manifest=(
                payload.get("validation")
                if isinstance(payload.get("validation"), Mapping)
                else None
            ),
            detector_green_operator=detector_green_operator,
        )
        if (
            model.additive_scatter_response is None
            or model.additive_scatter_response.to_payload()
            != additive_response.to_payload()
            or model.physical_component_discrepancy is None
            or model.physical_component_discrepancy.to_payload()
            != physical_component_discrepancy.to_payload()
        ):
            raise ValueError(
                "Schema-v7 physical transport parameters are not canonical."
            )
        reconstructed = model.manifest_payload()
        supplied = _thaw_json_value(_freeze_json_value(dict(payload)))
        if reconstructed != supplied:
            raise ValueError(
                "Full-spectrum manifest does not exactly reconstruct the "
                "declared physical and statistical contract."
            )
        model.require_runtime_ready()
        return model

    def _material_compton_fraction(
        self,
        sigma_kn_l: NDArray[np.float64],
        *,
        density_g_cm3: float,
        z_over_a: float,
        material_key: str,
    ) -> NDArray[np.float64]:
        """Return Compton/total attenuation fractions from line provenance."""
        compton_mu = (
            float(density_g_cm3)
            * AVOGADRO_CONSTANT_MOL_INV
            * float(z_over_a)
            * np.asarray(sigma_kn_l, dtype=np.float64)
        )
        total_mu = np.asarray(
            [
                float(item.get(material_key, compton_mu[index]))
                for index, item in enumerate(self._line_identity)
            ],
            dtype=np.float64,
        )
        return np.clip(
            np.divide(
                compton_mu,
                np.maximum(total_mu, np.finfo(np.float64).tiny),
            ),
            0.0,
            1.0,
        )

    def _build_contract_hash(self) -> str:
        """Return the physical model digest independent of validation results."""
        digest = hashlib.sha256()
        physics_response = isinstance(
            self.additive_scatter_response,
            PhysicsOnlyNoncollidedTransportResponse,
        )
        digest.update(
            b"geometry_conditioned_spectral_model_v7_"
            b"isotope_independent_detector_green_v3"
        )
        digest.update(
            json.dumps(
                {
                    "line_identity": [dict(item) for item in self._line_identity],
                    "source_rate_semantics": (
                        "pre_dead_time_detector_pulse_rate_at_1m"
                    ),
                    "source_rate_green_normalization": (
                        "catalog_branching_weighted_absolute_detection_"
                        "efficiency_at_1m_v1"
                    ),
                    "dead_time_tau_s": float(self.dead_time_tau_s),
                    "background_rate_cps": float(self.background_rate_cps),
                    "legacy_maximum_scatter_order": (
                        None if physics_response else int(self.maximum_scatter_order)
                    ),
                    "legacy_klein_nishina_quadrature_order": (
                        None
                        if physics_response
                        else int(self.klein_nishina_quadrature_order)
                    ),
                    "transport_feature_order": TRANSPORT_FEATURE_ORDER,
                    "detector_green_operator_id": (DETECTOR_GREEN_OPERATOR_ID),
                    "detector_green_operator_contract_sha256": (
                        self.detector_green_operator.contract_hash_sha256
                    ),
                    "detector_green_operator_binary_sha256": (
                        self.detector_green_operator.binary_sha256
                    ),
                    "detector_green_phase_conditioning": (
                        "transport_resolved_direct_impact_and_detector_cone_"
                        "scatter_joint_state_v3"
                    ),
                    "detector_cone_scatter_response": (
                        self._detector_cone_scatter_contract
                    ),
                    "detector_green_finite_mc_uncertainty": (
                        "pulse_plus_no_pulse_categorical_covariance_v1"
                    ),
                    "shield_pose_contract_sha256": (SHIELD_POSE_CONTRACT_SHA256),
                    "obstacle_material_contract_sha256": (
                        OBSTACLE_MATERIAL_CONTRACT_SHA256
                    ),
                    "transport_physics_table_contract_sha256": (
                        TRANSPORT_PHYSICS_TABLE_CONTRACT_SHA256
                    ),
                    "dead_time_model": (
                        "nonparalyzable_renewal_total_conditional_multinomial"
                    ),
                    "birth_proposal_score_method": (
                        "background_whitened_non_target_line_subspace_matched_filter_v1"
                    ),
                    "birth_proposal_background_regularization_counts": 1.0,
                    "rate_scale_mixture": "station_shared_finite_positive",
                    "mark_discrepancy": (
                        self.physical_component_discrepancy.mark_latent_model
                        if self.physical_component_discrepancy is not None
                        else "source_fraction_dirichlet_multinomial"
                        if self.mark_concentration_source is not None
                        else "finite_detector_corpus_dirichlet_multinomial"
                    ),
                    "mark_concentration_source": (
                        None
                        if self.mark_concentration_source is None
                        else float(self.mark_concentration_source)
                    ),
                    "discrepancy_training_manifest_sha256": (
                        self._discrepancy_training_manifest_sha256
                    ),
                    "additive_scatter_contract_sha256": (
                        None
                        if self.additive_scatter_response is None
                        else self.additive_scatter_response.contract_hash_sha256
                    ),
                    "transport_training_label_semantics": (
                        ADDITIVE_SCATTER_INCIDENT_LABEL_SEMANTICS
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        correction = self.low_rank_spectral_mean_correction
        if correction is not None:
            digest.update(b"\0low_rank_spectral_mean_correction_sha256\0")
            digest.update(correction.contract_hash_sha256.encode("ascii"))
        if self.count_discrepancy_concentration is not None:
            digest.update(b"\0count_discrepancy_concentration\0")
            digest.update(
                repr(float(self.count_discrepancy_concentration)).encode("ascii")
            )
            digest.update(b"\0count_discrepancy_scope\0")
            digest.update(str(self.count_discrepancy_scope).encode("ascii"))
        if self.mark_concentration_multi_isotope is not None:
            digest.update(b"\0mark_concentration_multi_isotope\0")
            digest.update(
                repr(float(self.mark_concentration_multi_isotope)).encode("ascii")
            )
        if self.physical_component_discrepancy is not None:
            digest.update(b"\0physical_component_discrepancy\0")
            digest.update(
                _canonical_json_sha256(
                    self.physical_component_discrepancy.to_payload()
                ).encode("ascii")
            )
        digest.update(_array_digest(self._energy_axis_keV))
        common_arrays = (
            self.response_operator_br,
            self.background_shape_b,
            self._direct_line_shapes_lb,
            self._marked_direct_line_shapes_lb,
            self._marked_direct_line_shapes_clb,
            self._direct_response_concentration_cl,
            self._source_rate_reference_efficiency_l,
            self._line_reference_detection_efficiency_l,
            self._source_rate_reference_efficiency_std_l,
            self._direct_detection_factor_cl,
            self._direct_detection_factor_std_cl,
            self._absolute_response_concentration_r,
            self._direct_absolute_response_concentration_cl,
            self._air_mu_compton_l,
            self._fe_compton_fraction_l,
            self._pb_compton_fraction_l,
            self._obstacle_compton_fraction_l,
            self._line_energies_keV_l,
        )
        scatter_arrays = (
            (
                self._detector_cone_scatter_distance_nodes_d,
                self._marked_detector_cone_scatter_shapes_dlb,
                self._detector_cone_scatter_response_concentration_dl,
                self._detector_cone_scatter_detection_factor_dl,
                self._detector_cone_scatter_detection_factor_std_dl,
            )
            if physics_response
            else (
                self._scatter_order_shapes_lob,
                self._marked_scatter_order_shapes_lob,
                self._scatter_response_concentration_lo,
                self._scatter_detection_factor_lo,
                self._scatter_detection_factor_std_lo,
            )
        )
        for array in (*common_arrays, *scatter_arrays):
            digest.update(_portable_derived_array_digest(array))
        digest.update(_array_digest(self._rate_scale_nodes_j))
        digest.update(_array_digest(self._rate_scale_weights_j))
        return digest.hexdigest()

    def _build_catalog_independent_contract_hash(self) -> str:
        """Return the production algorithm digest excluding catalog line choices.

        Line energies and branching weights are application inputs.  Their
        exact derived tensors remain protected by ``contract_hash_sha256``.
        This second digest intentionally covers only the isotope-independent
        detector, transport, uncertainty, and count-semantics implementation
        so adding an in-domain catalog profile does not invalidate physical
        approval evidence.
        """
        scatter_contract = (
            None
            if self._detector_cone_scatter_contract is None
            else dict(self._detector_cone_scatter_contract)
        )
        if scatter_contract is not None:
            scatter_contract.pop("contract_hash_sha256", None)
        payload = {
            "contract": "catalog_independent_full_spectrum_runtime_v1",
            "model_schema_version": FULL_SPECTRUM_MODEL_SCHEMA_VERSION,
            "source_rate_semantics": (
                "pre_dead_time_detector_pulse_rate_at_1m"
            ),
            "source_rate_green_normalization": (
                "catalog_branching_weighted_absolute_detection_efficiency_"
                "at_1m_v1"
            ),
            "catalog_line_projection": (
                "canonical_line_energy_branching_and_xcom_projection_v1"
            ),
            "dead_time_tau_s": float(self.dead_time_tau_s),
            "background_rate_cps": float(self.background_rate_cps),
            "transport_feature_order": TRANSPORT_FEATURE_ORDER,
            "detector_green_operator_id": DETECTOR_GREEN_OPERATOR_ID,
            "detector_green_operator_contract_sha256": (
                self.detector_green_operator.contract_hash_sha256
            ),
            "detector_green_operator_binary_sha256": (
                self.detector_green_operator.binary_sha256
            ),
            "detector_green_input_energy_domain_keV": (
                self.detector_green_operator.input_energy_domain_keV
            ),
            "detector_cone_scatter_response": scatter_contract,
            "shield_pose_contract_sha256": SHIELD_POSE_CONTRACT_SHA256,
            "obstacle_material_contract_sha256": (
                OBSTACLE_MATERIAL_CONTRACT_SHA256
            ),
            "transport_physics_table_contract_sha256": (
                TRANSPORT_PHYSICS_TABLE_CONTRACT_SHA256
            ),
            "dead_time_model": (
                "nonparalyzable_renewal_total_conditional_multinomial"
            ),
            "detector_green_finite_mc_uncertainty": (
                "pulse_plus_no_pulse_categorical_covariance_v1"
            ),
            "additive_scatter_response": (
                None
                if self.additive_scatter_response is None
                else self.additive_scatter_response.to_payload()
            ),
            "physical_component_discrepancy": (
                None
                if self.physical_component_discrepancy is None
                else self.physical_component_discrepancy.to_payload()
            ),
            "rate_scale_nodes": self._rate_scale_nodes_j.tolist(),
            "rate_scale_weights": self._rate_scale_weights_j.tolist(),
            "mark_concentration_source": self.mark_concentration_source,
            "mark_concentration_multi_isotope": (
                self.mark_concentration_multi_isotope
            ),
            "count_discrepancy_concentration": (
                self.count_discrepancy_concentration
            ),
            "count_discrepancy_scope": self.count_discrepancy_scope,
        }
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        for array in (
            self._energy_axis_keV,
            self.response_operator_br,
            self.background_shape_b,
            self._response_concentration_r,
            self._absolute_response_concentration_r,
        ):
            digest.update(_portable_derived_array_digest(array))
        return digest.hexdigest()

    @property
    def discrepancy_training_ready(self) -> bool:
        """Return whether global discrepancy parameters used training only."""
        manifest = self.discrepancy_training_manifest
        if not isinstance(manifest, Mapping):
            return False
        if manifest.get("schema_version") in (3, 4, 5):
            return self._physical_component_training_ready(manifest)
        if manifest.get("schema_version") == 2:
            return self._short_discrepancy_training_ready(manifest)
        expected_keys = {
            "schema_version",
            "acceptance_contract_sha256",
            "training_scene_seeds",
            "scenario_ids",
            "pair_ids_by_scene",
            "artifact_sha256_by_scene",
            "rate_scale_family",
            "mark_family",
            "selection_objective",
            "selected_rate_scale_half_width",
            "selected_mark_concentration_source",
            "candidate_count",
            "selected_training_log_predictive_density",
            "selection_artifact_sha256",
            "selection_completed",
        }
        if set(manifest) != expected_keys:
            return False
        if (
            manifest.get("schema_version") != 1
            or manifest.get("acceptance_contract_sha256")
            != FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256
            or tuple(manifest.get("training_scene_seeds", ()))
            != DESIGNATED_TRAINING_SCENE_SEEDS
            or tuple(manifest.get("scenario_ids", ())) != VALIDATION_SCENARIO_IDS
            or manifest.get("rate_scale_family")
            != "station_shared_three_node_symmetric_mean_one"
            or manifest.get("mark_family") != "source_fraction_dirichlet_multinomial"
            or manifest.get("selection_objective")
            != "maximum_joint_training_log_predictive_density"
            or manifest.get("selection_completed") is not True
            or manifest.get("candidate_count")
            != len(RATE_SCALE_HALF_WIDTH_GRID) * len(MARK_CONCENTRATION_GRID)
            or not _is_sha256(manifest.get("selection_artifact_sha256"))
        ):
            return False
        pair_ids = manifest.get("pair_ids_by_scene")
        artifact_hashes = manifest.get("artifact_sha256_by_scene")
        expected_seed_keys = {str(seed) for seed in DESIGNATED_TRAINING_SCENE_SEEDS}
        if (
            not isinstance(pair_ids, Mapping)
            or set(pair_ids) != expected_seed_keys
            or any(
                tuple(pair_ids[str(seed)]) != tuple(range(64))
                for seed in DESIGNATED_TRAINING_SCENE_SEEDS
            )
            or not isinstance(artifact_hashes, Mapping)
            or set(artifact_hashes) != expected_seed_keys
            or any(
                not _is_sha256(artifact_hashes[str(seed)])
                for seed in DESIGNATED_TRAINING_SCENE_SEEDS
            )
        ):
            return False
        try:
            width = float(manifest["selected_rate_scale_half_width"])
            concentration = float(manifest["selected_mark_concentration_source"])
            selected_score = float(manifest["selected_training_log_predictive_density"])
        except (TypeError, ValueError):
            return False
        if (
            width not in RATE_SCALE_HALF_WIDTH_GRID
            or concentration not in MARK_CONCENTRATION_GRID
            or not np.isfinite(selected_score)
            or self.mark_concentration_source is None
            or float(self.mark_concentration_source) != concentration
        ):
            return False
        expected_nodes, expected_weights = rate_scale_mixture_for_half_width(width)
        return bool(
            np.array_equal(
                self._rate_scale_nodes_j,
                np.asarray(expected_nodes, dtype=np.float64),
            )
            and np.array_equal(
                self._rate_scale_weights_j,
                np.asarray(expected_weights, dtype=np.float64),
            )
        )

    def _physical_component_training_ready(
        self,
        manifest: Mapping[str, object],
    ) -> bool:
        """Validate randomized-family training for component latents."""
        legacy_keys = {
            "schema_version",
            "training_policy",
            "acceptance_contract_sha256",
            "geometry_family_applicability_sha256",
            "training_scene_seeds",
            "scenario_ids",
            "artifact_sha256_by_scene_and_scenario",
            "component_family",
            "selected_concentrations",
            "selection_objective",
            "selection_completed",
            "holdout_artifacts_consumed",
        }
        exact_basis_keys = legacy_keys | {
            "base_additive_response_contract_sha256",
            "low_rank_mean_correction_contract_sha256",
            "feature_basis_semantics",
        }
        calibrated_exact_basis_keys = exact_basis_keys | {
            "mark_tail_probability_threshold",
            "mark_cross_fitted_coverage_threshold",
            "selected_mark_cross_fitted_coverage",
        }
        component = self.physical_component_discrepancy
        selected = manifest.get("selected_concentrations")
        selection_contract = (
            manifest.get("training_policy"),
            manifest.get("selection_objective"),
        )
        if (
            component is None
            or set(manifest)
            not in (
                legacy_keys,
                exact_basis_keys,
                calibrated_exact_basis_keys,
            )
            or manifest.get("schema_version") not in (3, 4, 5)
            or selection_contract
            not in {
                (
                    "randomized_geometry_family_training_only_v1",
                    "maximum_training_log_predictive_density_regularized",
                ),
                (
                    "randomized_geometry_family_cross_fitted_component_v2",
                    "leave_one_geometry_out_log_predictive_density_regularized",
                ),
                (
                    "randomized_geometry_family_cross_fitted_component_v3",
                    "leave_one_geometry_out_log_predictive_density_regularized",
                ),
                (
                    "randomized_geometry_family_cross_fitted_component_v4",
                    "leave_one_geometry_out_log_predictive_density_regularized_"
                    "subject_to_predeclared_pairwise_mark_coverage",
                ),
            }
            or manifest.get("acceptance_contract_sha256")
            != FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256
            or manifest.get("geometry_family_applicability_sha256")
            != GEOMETRY_FAMILY_APPLICABILITY_SHA256
            or tuple(manifest.get("training_scene_seeds", ()))
            != DESIGNATED_TRAINING_SCENE_SEEDS
            or tuple(manifest.get("scenario_ids", ())) != VALIDATION_SCENARIO_IDS
            or manifest.get("component_family")
            != "uncollided_scatter_component_latents_v1"
            or manifest.get("selection_completed") is not True
            or manifest.get("holdout_artifacts_consumed") is not False
            or not isinstance(
                manifest.get("artifact_sha256_by_scene_and_scenario"),
                Mapping,
            )
            or not isinstance(selected, Mapping)
        ):
            return False
        if manifest.get("schema_version") in (4, 5):
            additive = self.additive_scatter_response
            correction = self.low_rank_spectral_mean_correction
            if (
                set(manifest)
                != (
                    exact_basis_keys
                    if manifest.get("schema_version") == 4
                    else calibrated_exact_basis_keys
                )
                or additive is None
                or correction is None
                or manifest.get("base_additive_response_contract_sha256")
                != additive.contract_hash_sha256
                or manifest.get("low_rank_mean_correction_contract_sha256")
                != correction.contract_hash_sha256
                or manifest.get("feature_basis_semantics")
                != additive.feature_basis_semantics
            ):
                return False
        if manifest.get("schema_version") == 5:
            try:
                mark_tail_threshold = float(manifest["mark_tail_probability_threshold"])
                mark_coverage_threshold = float(
                    manifest["mark_cross_fitted_coverage_threshold"]
                )
                selected_mark_coverage = float(
                    manifest["selected_mark_cross_fitted_coverage"]
                )
            except (TypeError, ValueError):
                return False
            required_coverage = float(
                ACCEPTANCE_METRIC_CONTRACT[
                    "conditional_mark_upper_tail_ge_0p01_fraction"
                ][1]
            )
            if (
                mark_tail_threshold != 0.01
                or mark_coverage_threshold != required_coverage
                or not np.isfinite(selected_mark_coverage)
                or selected_mark_coverage + 1.0e-12 < mark_coverage_threshold
                or selected_mark_coverage > 1.0
            ):
                return False
        expected_selected = {
            "count_uncollided_concentration": float(
                component.count_uncollided_concentration
            ),
            "count_scatter_concentration": float(component.count_scatter_concentration),
            "mark_uncollided_concentration": float(
                component.mark_uncollided_concentration
            ),
            "mark_scatter_concentration": float(component.mark_scatter_concentration),
            "count_scope": component.count_scope,
        }
        return dict(selected) == expected_selected

    def _short_discrepancy_training_ready(
        self,
        manifest: Mapping[str, object],
    ) -> bool:
        """Validate a declared short diagnostic training-only discrepancy.

        This contract authorizes runtime diagnosis but never production
        approval.  It exists to decouple a logically complete model from the
        optional multi-day all-64 release evaluation.  Holdout artifacts are
        explicitly forbidden from parameter selection.
        """
        expected_keys = {
            "schema_version",
            "training_policy",
            "acceptance_contract_sha256",
            "training_scene_seeds",
            "scenario_ids",
            "pair_ids_by_scene_and_scenario",
            "artifact_sha256_by_scene_and_scenario",
            "rate_scale_family",
            "mark_family",
            "mark_calibration",
            "selection_objective",
            "selected_rate_scale_half_width",
            "selected_count_discrepancy_scope",
            "selected_mark_concentration_source",
            "selected_mark_concentration_multi_isotope",
            "candidate_count",
            "selected_training_log_predictive_density",
            "selection_artifact_sha256",
            "selection_completed",
            "holdout_artifacts_consumed",
        }
        if set(manifest) != expected_keys:
            return False
        training_policy = manifest.get("training_policy")
        rate_scale_family = manifest.get("rate_scale_family")
        legacy_training = (
            training_policy == "declared_short_diagnostic_training_no_holdout_feedback"
        )
        runtime_training = (
            training_policy == "declared_runtime_training_no_holdout_feedback_v2"
        )
        expected_candidate_count = (
            1
            + (1 if runtime_training else 2) * (len(RATE_SCALE_HALF_WIDTH_GRID) - 1)
            + 2 * len(MARK_CONCENTRATION_GRID)
        )
        if (
            not (legacy_training or runtime_training)
            or manifest.get("acceptance_contract_sha256")
            != FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256
            or (
                legacy_training
                and rate_scale_family
                != "selected_scope_gamma_poisson_recorded_count_mean_one"
            )
            or (
                runtime_training
                and rate_scale_family
                != "view_conditioned_gamma_poisson_recorded_count_mean_one"
            )
            or manifest.get("mark_family") != "source_fraction_dirichlet_multinomial"
            or manifest.get("selection_objective")
            != "maximum_joint_training_log_predictive_density"
            or manifest.get("selection_completed") is not True
            or manifest.get("holdout_artifacts_consumed") is not False
            or manifest.get("candidate_count") != expected_candidate_count
            or not _is_sha256(manifest.get("selection_artifact_sha256"))
        ):
            return False
        raw_seeds = manifest.get("training_scene_seeds")
        raw_scenarios = manifest.get("scenario_ids")
        pair_ids = manifest.get("pair_ids_by_scene_and_scenario")
        artifact_hashes = manifest.get("artifact_sha256_by_scene_and_scenario")
        mark_calibration = manifest.get("mark_calibration")
        if (
            not isinstance(raw_seeds, tuple)
            or not raw_seeds
            or any(type(seed) is not int for seed in raw_seeds)
            or len(set(raw_seeds)) != len(raw_seeds)
            or any(seed in DESIGNATED_VALIDATION_SCENE_SEEDS for seed in raw_seeds)
            or not isinstance(raw_scenarios, tuple)
            or len(raw_scenarios) < 2
            or any(
                scenario not in VALIDATION_SCENARIO_IDS for scenario in raw_scenarios
            )
            or "single_line_source_resolved" not in raw_scenarios
            or "dominant_plus_absent_isotope" not in raw_scenarios
            or not isinstance(pair_ids, Mapping)
            or not isinstance(artifact_hashes, Mapping)
            or not isinstance(mark_calibration, Mapping)
        ):
            return False
        expected_seed_keys = {str(seed) for seed in raw_seeds}
        if set(pair_ids) != expected_seed_keys or set(artifact_hashes) != (
            expected_seed_keys
        ):
            return False
        for seed in raw_seeds:
            seed_key = str(seed)
            scenario_pairs = pair_ids.get(seed_key)
            scenario_hashes = artifact_hashes.get(seed_key)
            if (
                not isinstance(scenario_pairs, Mapping)
                or set(scenario_pairs) != set(raw_scenarios)
                or not isinstance(scenario_hashes, Mapping)
                or set(scenario_hashes) != set(raw_scenarios)
            ):
                return False
            for scenario in raw_scenarios:
                values = scenario_pairs[scenario]
                hashes = scenario_hashes[scenario]
                if (
                    not isinstance(values, tuple)
                    or not values
                    or any(
                        type(value) is not int or value < 0 or value >= 64
                        for value in values
                    )
                    or len(set(values)) != len(values)
                    or not isinstance(hashes, Mapping)
                    or set(hashes) != {str(value) for value in values}
                    or any(not _is_sha256(value) for value in hashes.values())
                ):
                    return False
        try:
            width = float(manifest["selected_rate_scale_half_width"])
            selected_scope = manifest["selected_count_discrepancy_scope"]
            concentration = float(manifest["selected_mark_concentration_source"])
            multi_concentration = float(
                manifest["selected_mark_concentration_multi_isotope"]
            )
            selected_score = float(manifest["selected_training_log_predictive_density"])
        except (TypeError, ValueError):
            return False
        expected_mark_keys = {
            "method",
            "lower_quantile",
            "lower_quantile_moment_concentration_by_scenario",
            "selected_concentration",
            "selected_multi_isotope_concentration",
            "training_scene_seeds",
            "scenario_ids",
            "pair_ids",
            "artifact_sha256_by_scene_and_scenario",
            "design_sha256",
            "holdout_artifacts_consumed",
        }
        if (
            set(mark_calibration) != expected_mark_keys
            or mark_calibration.get("method")
            not in {
                "training_mean_dirichlet_moment_lower_quantile_v1",
                (
                    "training_mean_dirichlet_moment_lower_quantile_"
                    "cardinality_conservative_v2"
                ),
            }
            or float(mark_calibration.get("lower_quantile", -1.0)) != 0.05
            or mark_calibration.get("holdout_artifacts_consumed") is not False
            or not _is_sha256(mark_calibration.get("design_sha256"))
            or tuple(mark_calibration.get("training_scene_seeds", ()))
            != (2026072701, 2026072702)
            or tuple(mark_calibration.get("scenario_ids", ()))
            != (
                "dominant_plus_absent_isotope",
                "multi_isotope_superposition",
                "continuous_surface_perturbation_ranking",
            )
            or len(tuple(mark_calibration.get("pair_ids", ()))) < 16
            or float(mark_calibration.get("selected_concentration", -1.0))
            != concentration
            or float(
                mark_calibration.get(
                    "selected_multi_isotope_concentration",
                    -1.0,
                )
            )
            != multi_concentration
            or not isinstance(
                mark_calibration.get("lower_quantile_moment_concentration_by_scenario"),
                Mapping,
            )
            or not isinstance(
                mark_calibration.get("artifact_sha256_by_scene_and_scenario"),
                Mapping,
            )
        ):
            return False
        if (
            width not in RATE_SCALE_HALF_WIDTH_GRID
            or concentration not in MARK_CONCENTRATION_GRID
            or not np.isfinite(selected_score)
            or self.mark_concentration_source is None
            or float(self.mark_concentration_source) != concentration
            or self.mark_concentration_multi_isotope is None
            or float(self.mark_concentration_multi_isotope) != multi_concentration
        ):
            return False
        expected_count_concentration = None if width == 0.0 else 3.0 / float(width**2)
        if selected_scope not in (
            None,
            "station_shared",
            "view_independent",
        ) or (width == 0.0) != (selected_scope is None):
            return False
        if runtime_training and (
            selected_scope != (None if width == 0.0 else "view_independent")
            or mark_calibration.get("method")
            != (
                "training_mean_dirichlet_moment_lower_quantile_"
                "cardinality_conservative_v2"
            )
            or concentration != multi_concentration
        ):
            return False
        return bool(
            np.array_equal(self._rate_scale_nodes_j, np.asarray((1.0,)))
            and np.array_equal(self._rate_scale_weights_j, np.asarray((1.0,)))
            and self.count_discrepancy_concentration == expected_count_concentration
            and self.count_discrepancy_scope == selected_scope
        )

    @property
    def exact_physical_statistics_ready(self) -> bool:
        """Return whether no empirical likelihood discrepancy is configured."""
        component = self.physical_component_discrepancy
        return bool(
            self.discrepancy_training_manifest is None
            and self.low_rank_spectral_mean_correction is None
            and self.mark_concentration_source is None
            and np.array_equal(
                self._rate_scale_nodes_j,
                np.asarray((1.0,), dtype=np.float64),
            )
            and np.array_equal(
                self._rate_scale_weights_j,
                np.asarray((1.0,), dtype=np.float64),
            )
            and self.count_discrepancy_concentration is None
            and self.count_discrepancy_scope is None
            and self.mark_concentration_multi_isotope is None
            and (component is None or component.physics_only)
            and not isinstance(
                self.additive_scatter_response,
                AdditiveNoncollidedTransportResponse,
            )
        )

    @property
    def runtime_ready(self) -> bool:
        """Return whether physics-only contracts authorize runtime use."""
        additive_response = self.additive_scatter_response
        return bool(
            isinstance(
                additive_response,
                PhysicsOnlyNoncollidedTransportResponse,
            )
            and additive_response.training_ready
            and self.low_rank_spectral_mean_correction is None
            and self.discrepancy_training_manifest is None
            and self.physical_component_discrepancy is not None
            and self.physical_component_discrepancy.physics_only
            and self.physical_component_discrepancy.mark_latent_model
            == "component_dirichlet_tree_hierarchical"
            and self.physical_component_discrepancy.mark_background_group_concentration
            == self.physical_component_discrepancy.mark_scatter_concentration
            and self.physical_component_discrepancy.mark_background_within_concentration
            == self.physical_component_discrepancy.mark_uncollided_concentration
            and isinstance(self._detector_cone_scatter_contract, Mapping)
            and self._detector_cone_scatter_contract.get("response")
            == DETECTOR_CONE_SCATTER_RESPONSE_ID
            and _is_sha256(
                self._detector_cone_scatter_contract.get("contract_hash_sha256")
            )
            and self.mark_concentration_source is None
            and self.mark_concentration_multi_isotope is None
            and self.count_discrepancy_concentration is None
            and self.count_discrepancy_scope is None
            and np.array_equal(
                self._rate_scale_nodes_j,
                np.asarray((1.0,), dtype=np.float64),
            )
            and np.array_equal(
                self._rate_scale_weights_j,
                np.asarray((1.0,), dtype=np.float64),
            )
        )

    @property
    def production_ready(self) -> bool:
        """Return whether independent evidence approves this runtime algorithm."""
        if not self.runtime_ready:
            return False
        additive_response = self.additive_scatter_response
        if (
            additive_response is None
            or additive_response.feature_basis_semantics
            != DETECTOR_CONE_AIR_XCOM_SINGLE_SCATTER_BASIS_SEMANTICS
        ):
            return False
        manifest = self.validation_manifest
        if not isinstance(manifest, Mapping):
            return False
        base_expected_keys = {
            "schema_version",
            "validation_contract_sha256",
            "approved_model_contract_sha256",
            "acceptance_run_contract_sha256",
            "runtime_config_sha256",
            "native_executable_sha256",
            "native_execution_environment_sha256",
            "implementation_bundle_sha256",
            "detector_green_operator_contract_sha256",
            "detector_green_operator_binary_sha256",
            "detector_green_validation",
            "detector_green_validation_manifest_sha256",
            "additive_scatter_contract_sha256",
            "surface_emission_policy_sha256",
            "validation_scene_seeds",
            "candidate_selection",
            "scene_calibration_count",
            "metric_scene_seeds",
            "metric_split",
            "metric_aggregation",
            "scenario_ids",
            "pair_ids_by_scene",
            "artifact_sha256_by_scene",
            "scene_hash_by_scene_and_scenario",
            "surface_source_contract_sha256_by_scene_and_scenario",
            "metrics",
            "all_passed",
        }
        schema_version = manifest.get("schema_version")
        expected_keys = (
            base_expected_keys
            if schema_version == 6
            else base_expected_keys | _TRANSFERRED_VALIDATION_FIELDS
            if schema_version == TRANSFERRED_VALIDATION_SCHEMA_VERSION
            else frozenset()
        )
        if set(manifest) != expected_keys:
            return False
        exact_application_approval = (
            manifest.get("approved_model_contract_sha256")
            == self.contract_hash_sha256
        )
        if schema_version == 6 and not exact_application_approval:
            return False
        if schema_version == TRANSFERRED_VALIDATION_SCHEMA_VERSION:
            source_manifest = _thaw_json_value(manifest)
            for field_name in _TRANSFERRED_VALIDATION_FIELDS:
                source_manifest.pop(field_name, None)
            source_manifest["schema_version"] = 6
            validated_isotopes = manifest.get("application_validation_isotopes")
            line_energies = np.asarray(
                [float(row["energy_keV"]) for row in self._line_identity],
                dtype=np.float64,
            )
            lower_energy, upper_energy = (
                self.detector_green_operator.input_energy_domain_keV
            )
            if (
                manifest.get("approval_scope")
                != CATALOG_INDEPENDENT_APPROVAL_SCOPE
                or manifest.get(
                    "approved_catalog_independent_contract_sha256"
                )
                != self.catalog_independent_contract_hash_sha256
                or manifest.get("source_validation_manifest_sha256")
                != _canonical_json_sha256(source_manifest)
                or not isinstance(validated_isotopes, tuple)
                or not validated_isotopes
                or any(
                    type(value) is not str or not value
                    for value in validated_isotopes
                )
                or tuple(sorted(set(validated_isotopes))) != validated_isotopes
                or np.any(~np.isfinite(line_energies))
                or np.any(line_energies < float(lower_energy))
                or np.any(line_energies > float(upper_energy))
            ):
                return False
        if (
            not _is_sha256(manifest.get("validation_contract_sha256"))
            or not _is_sha256(manifest.get("approved_model_contract_sha256"))
            or not _is_sha256(manifest.get("acceptance_run_contract_sha256"))
            or not _is_sha256(manifest.get("runtime_config_sha256"))
            or not _is_sha256(manifest.get("native_executable_sha256"))
            or not _is_sha256(manifest.get("native_execution_environment_sha256"))
            or not _is_sha256(manifest.get("implementation_bundle_sha256"))
            or manifest.get("detector_green_operator_contract_sha256")
            != self.detector_green_operator.contract_hash_sha256
            or manifest.get("detector_green_operator_binary_sha256")
            != self.detector_green_operator.binary_sha256
            or manifest.get("additive_scatter_contract_sha256")
            != self.additive_scatter_response.contract_hash_sha256
            or manifest.get("surface_emission_policy_sha256")
            != surface_emission_policy_sha256()
            or manifest.get("candidate_selection") != "none_predeclared_physics_only"
            or isinstance(manifest.get("scene_calibration_count"), bool)
            or manifest.get("scene_calibration_count") != 0
            or manifest.get("metric_split") != "independent_validation_only"
            or manifest.get("metric_aggregation")
            != "validation_scene_conservative_worst_case"
            or manifest.get("all_passed") is not True
        ):
            return False
        raw_validation_seeds = manifest.get("validation_scene_seeds")
        raw_metric_seeds = manifest.get("metric_scene_seeds")
        raw_scenario_ids = manifest.get("scenario_ids")
        if (
            not isinstance(raw_validation_seeds, tuple)
            or len(raw_validation_seeds) < 5
            or any(type(seed) is not int for seed in raw_validation_seeds)
            or len(set(raw_validation_seeds)) != len(raw_validation_seeds)
            or raw_metric_seeds != raw_validation_seeds
            or not isinstance(raw_scenario_ids, tuple)
            or not raw_scenario_ids
            or any(type(value) is not str or not value for value in raw_scenario_ids)
            or len(set(raw_scenario_ids)) != len(raw_scenario_ids)
        ):
            return False
        try:
            detector_green_validation = validate_detector_green_validation_manifest(
                _thaw_json_value(manifest.get("detector_green_validation")),
                operator=self.detector_green_operator,
            )
        except (TypeError, ValueError):
            return False
        if (
            manifest.get("detector_green_validation_manifest_sha256")
            != detector_green_validation_manifest_sha256(
                detector_green_validation,
                operator=self.detector_green_operator,
            )
            or detector_green_validation["native_executable_sha256"]
            != manifest["native_executable_sha256"]
            or detector_green_validation["native_execution_environment_sha256"]
            != manifest["native_execution_environment_sha256"]
        ):
            return False
        all_seeds = raw_validation_seeds
        scenario_ids = raw_scenario_ids
        pair_ids = manifest.get("pair_ids_by_scene")
        artifact_hashes = manifest.get("artifact_sha256_by_scene")
        scene_hashes = manifest.get("scene_hash_by_scene_and_scenario")
        source_hashes = manifest.get(
            "surface_source_contract_sha256_by_scene_and_scenario"
        )
        expected_seed_keys = {str(seed) for seed in all_seeds}
        if (
            not isinstance(pair_ids, Mapping)
            or set(pair_ids) != expected_seed_keys
            or any(tuple(pair_ids[str(seed)]) != tuple(range(64)) for seed in all_seeds)
            or not isinstance(artifact_hashes, Mapping)
            or set(artifact_hashes) != expected_seed_keys
            or any(not _is_sha256(artifact_hashes[str(seed)]) for seed in all_seeds)
            or not isinstance(scene_hashes, Mapping)
            or set(scene_hashes) != expected_seed_keys
            or any(
                not isinstance(scene_hashes[str(seed)], Mapping)
                or set(scene_hashes[str(seed)]) != set(scenario_ids)
                or any(
                    not _is_sha256(scene_hashes[str(seed)][scenario])
                    for scenario in scenario_ids
                )
                for seed in all_seeds
            )
            or not isinstance(source_hashes, Mapping)
            or set(source_hashes) != expected_seed_keys
            or any(
                not isinstance(source_hashes[str(seed)], Mapping)
                or set(source_hashes[str(seed)]) != set(scenario_ids)
                or any(
                    not _is_sha256(source_hashes[str(seed)][scenario])
                    for scenario in scenario_ids
                )
                for seed in all_seeds
            )
        ):
            return False
        metrics = manifest.get("metrics")
        if not isinstance(metrics, Mapping) or not metrics:
            return False
        for result in metrics.values():
            if not isinstance(result, Mapping) or set(result) != {
                "value",
                "comparison",
                "threshold",
                "passed",
            }:
                return False
            raw_value = result["value"]
            raw_threshold = result["threshold"]
            if (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, (int, float))
                or isinstance(raw_threshold, bool)
                or not isinstance(raw_threshold, (int, float))
            ):
                return False
            value = float(raw_value)
            reported_threshold = float(raw_threshold)
            if (
                not np.isfinite(value)
                or not np.isfinite(reported_threshold)
                or result["comparison"] not in {"le", "ge"}
                or result["passed"] is not True
            ):
                return False
            expected_pass = (
                value <= reported_threshold
                if result["comparison"] == "le"
                else value >= reported_threshold
            )
            if not expected_pass:
                return False
        return True

    @property
    def contract_hash_sha256(self) -> str:
        """Return the immutable physical model hash."""
        return self._contract_hash_sha256

    @property
    def catalog_independent_contract_hash_sha256(self) -> str:
        """Return the isotope-independent detector/transport algorithm hash."""
        return self._catalog_independent_contract_hash_sha256

    @property
    def energy_axis_keV(self) -> NDArray[np.float64]:
        """Return a defensive copy of the native analysis axis."""
        return self._energy_axis_keV.copy()

    @property
    def line_identity(self) -> tuple[Mapping[str, object], ...]:
        """Return the global positive transport-line order."""
        return tuple(dict(item) for item in self._line_identity)

    @property
    def transport_feature_order(self) -> tuple[str, ...]:
        """Return the canonical geometry feature order."""
        return TRANSPORT_FEATURE_ORDER

    @property
    def detector_impact_parameter_edges_fraction(self) -> NDArray[np.float64]:
        """Return the authenticated detector-impact partition."""
        return np.array(
            self.detector_green_operator.impact_parameter_edges_fraction,
            dtype=np.float64,
            copy=True,
        )

    @property
    def detector_target_radius_m(self) -> float:
        """Return the detector housing radius used by the Green operator."""
        return float(self.detector_green_operator.detector_target_radius_m)

    def require_production_ready(self) -> None:
        """Fail closed until independent evidence approves the runtime algorithm."""
        if not self.production_ready:
            raise RuntimeError(
                "Geometry-conditioned spectrum model has neither exact "
                "independent all-64 validation nor transferable "
                "catalog-independent algorithm approval for every configured "
                "line energy."
            )

    def require_runtime_ready(self) -> None:
        """Fail closed until physics-only runtime contracts are complete."""
        if not self.runtime_ready:
            raise RuntimeError(
                "Geometry-conditioned spectrum model has not passed its "
                "physics-only transport, uncertainty, and detector-Green "
                "runtime gates."
            )

    def require_environment_applicable(
        self,
        environment_payload: Mapping[str, object],
    ) -> None:
        """Reject empirical component latents outside their geometry family."""
        if (
            self.physical_component_discrepancy is None
            or self.physical_component_discrepancy.physics_only
        ):
            return
        descriptor = environment_payload.get("geometry_family")
        if not isinstance(descriptor, Mapping):
            raise RuntimeError(
                "Physical-component discrepancy requires an authenticated "
                "geometry_family descriptor in the MeasurementLog."
            )
        try:
            validate_geometry_family_descriptor(
                descriptor,
                require_in_domain=True,
            )
        except ValueError as exc:
            raise RuntimeError(
                "Measurement environment is outside the trained geometry "
                "family; refusing empirical full-spectrum discrepancy."
            ) from exc

    def _validated_numpy_inputs(
        self,
        total_line_contributions_xvsl: NDArray[np.float64],
        uncollided_line_contributions_xvsl: NDArray[np.float64],
        transport_features_xvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Return aligned finite source-resolved NumPy transport inputs."""
        total = np.asarray(total_line_contributions_xvsl, dtype=np.float64)
        uncollided = np.asarray(
            uncollided_line_contributions_xvsl,
            dtype=np.float64,
        )
        features = np.asarray(transport_features_xvslf, dtype=np.float64)
        live_times = np.asarray(live_times_s_v, dtype=np.float64)
        line_count = len(self._line_identity)
        if (
            total.ndim < 3
            or total.shape[-1] != line_count
            or uncollided.shape != total.shape
            or features.shape != total.shape + (len(TRANSPORT_FEATURE_ORDER),)
            or live_times.shape != (total.shape[-3],)
            or np.any(~np.isfinite(total))
            or np.any(total < 0.0)
            or np.any(~np.isfinite(uncollided))
            or np.any(uncollided < 0.0)
            or np.any(~np.isfinite(features))
            or np.any(features < 0.0)
            or np.any(~np.isfinite(live_times))
            or np.any(live_times <= 0.0)
        ):
            raise ValueError(
                "Spectrum transport inputs must be finite nonnegative "
                "...view/source/line arrays with positive view live times."
            )
        tolerance = (
            128.0
            * np.finfo(np.float64).eps
            * np.maximum(
                1.0,
                np.maximum(np.abs(total), np.abs(uncollided)),
            )
        )
        if np.any(uncollided > total + tolerance):
            raise ValueError(
                "Uncollided line contributions cannot exceed total incident "
                "line contributions."
            )
        uncollided = np.minimum(uncollided, total)
        return total, uncollided, features, live_times

    def _interaction_order_weights_numpy(
        self,
        features_xvslf: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return conditional positive-interaction order probabilities."""
        features = np.asarray(features_xvslf, dtype=np.float64)
        tau = (
            features[..., 0] * self._fe_compton_fraction_l
            + features[..., 1] * self._pb_compton_fraction_l
            + features[..., 3]
            + features[..., TRANSPORT_DISTANCE_FEATURE_INDEX]
            * 100.0
            * self._air_mu_compton_l
        )
        tau = np.maximum(tau, 0.0)
        exact_orders = np.arange(
            1,
            int(self.maximum_scatter_order),
            dtype=np.float64,
        )
        denominator = -np.expm1(-tau)
        safe_tau = np.maximum(tau, np.finfo(np.float64).tiny)
        log_exact = (
            -tau[..., np.newaxis]
            + np.log(safe_tau)[..., np.newaxis] * exact_orders
            - special.gammaln(exact_orders + 1.0)
            - np.log(np.maximum(denominator, np.finfo(np.float64).tiny))[
                ..., np.newaxis
            ]
        )
        exact = np.exp(log_exact)
        exact = np.where(tau[..., np.newaxis] > 0.0, exact, 0.0)
        tail = np.maximum(1.0 - np.sum(exact, axis=-1), 0.0)
        weights = np.concatenate(
            (exact, tail[..., np.newaxis]),
            axis=-1,
        )
        zero_tau = tau <= 0.0
        weights[..., 0] = np.where(zero_tau, 1.0, weights[..., 0])
        weights[..., 1:] = np.where(
            zero_tau[..., np.newaxis],
            0.0,
            weights[..., 1:],
        )
        weights /= np.maximum(
            np.sum(weights, axis=-1, keepdims=True),
            np.finfo(np.float64).tiny,
        )
        return weights

    def _direct_phase_weights_numpy(
        self,
        features_xvslf: NDArray[np.float64],
        *,
        active_xvsl: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """Return authenticated transport-conditioned impact probabilities."""
        features = np.asarray(features_xvslf, dtype=np.float64)
        active = np.asarray(active_xvsl, dtype=np.bool_)
        phase = features[..., TRANSPORT_IMPACT_FEATURE_OFFSET:]
        expected_phase_count = int(
            self.detector_green_operator.impact_parameter_edges_fraction.size - 1
        )
        if (
            active.shape != features.shape[:-1]
            or phase.shape != active.shape + (expected_phase_count,)
            or expected_phase_count != DETECTOR_IMPACT_PHASE_COUNT
            or np.any(~np.isfinite(phase))
            or np.any(phase < 0.0)
            or np.any(phase > 1.0)
        ):
            raise ValueError(
                "Detector Green phase conditioning requires the complete "
                "finite transport-resolved impact contract."
            )
        sums = np.sum(phase, axis=-1)
        if np.any(active & ~np.isclose(sums, 1.0, rtol=0.0, atol=1.0e-10)):
            raise ValueError(
                "Every positive direct contribution requires normalized "
                "transport-resolved detector-impact probabilities."
            )
        return phase

    def _direct_phase_weights_torch(
        self,
        features_xvslf: object,
        *,
        active_xvsl: object,
    ) -> object:
        """Return Torch transport-conditioned impact probabilities."""
        import torch

        features = torch.as_tensor(features_xvslf)
        active = torch.as_tensor(
            active_xvsl,
            device=features.device,
            dtype=torch.bool,
        )
        phase = features[..., TRANSPORT_IMPACT_FEATURE_OFFSET:]
        expected_phase_count = int(
            self.detector_green_operator.impact_parameter_edges_fraction.size - 1
        )
        if (
            tuple(active.shape) != tuple(features.shape[:-1])
            or tuple(phase.shape) != tuple(active.shape) + (expected_phase_count,)
            or expected_phase_count != DETECTOR_IMPACT_PHASE_COUNT
        ):
            raise ValueError("Detector Green active-ray mask is misaligned.")
        if (
            not bool(torch.all(torch.isfinite(phase)))
            or bool(torch.any(phase < 0.0))
            or bool(torch.any(phase > 1.0))
        ):
            raise ValueError(
                "Detector Green phase conditioning requires the complete "
                "finite transport-resolved impact contract."
            )
        sums = torch.sum(phase, dim=-1)
        normalized = torch.isclose(
            sums,
            torch.ones_like(sums),
            rtol=0.0,
            atol=1.0e-10,
        )
        if bool(torch.any(active & ~normalized)):
            raise ValueError(
                "Every positive direct contribution requires normalized "
                "transport-resolved detector-impact probabilities."
            )
        return phase

    def _detector_cone_scatter_coefficients_numpy(
        self,
        scatter_xvsl: NDArray[np.float64],
        features_xvslf: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Bin physical single-scatter rates by line and distance node."""
        scatter = np.asarray(scatter_xvsl, dtype=np.float64)
        features = np.asarray(features_xvslf, dtype=np.float64)
        response = self.additive_scatter_response
        if not isinstance(
            response, PhysicsOnlyNoncollidedTransportResponse
        ) or features.shape != scatter.shape + (len(TRANSPORT_FEATURE_ORDER),):
            raise RuntimeError(
                "Detector-cone scatter coefficients require the physics-only "
                "transport contract."
            )
        distance = features[..., TRANSPORT_DISTANCE_FEATURE_INDEX]
        radius = float(response.detector_radius_m)
        variable_distance = np.maximum(0.5 * distance, radius)
        if (
            np.any(~np.isfinite(variable_distance))
            or np.any(variable_distance < radius)
            or np.any(variable_distance > DETECTOR_CONE_SCATTER_MAXIMUM_DISTANCE_M)
        ):
            raise ValueError(
                "Scatter-to-detector distance is outside the authenticated "
                "detector-cone response domain."
            )
        energy_shape = (1,) * (distance.ndim - 1) + (
            int(self._line_energies_keV_l.size),
        )
        energies = self._line_energies_keV_l.reshape(energy_shape)
        fe_acceptance = klein_nishina_forward_cone_fraction_numpy(
            energies,
            detector_radius_m=radius,
            scatter_distance_m=float(response.fe_scatter_distance_m),
        )
        pb_acceptance = klein_nishina_forward_cone_fraction_numpy(
            energies,
            detector_radius_m=radius,
            scatter_distance_m=float(response.pb_scatter_distance_m),
        )
        variable_acceptance = klein_nishina_forward_cone_fraction_numpy(
            energies,
            detector_radius_m=radius,
            scatter_distance_m=variable_distance,
        )
        component_mass = np.stack(
            (
                features[..., 0] * self._fe_compton_fraction_l * fe_acceptance,
                features[..., 1] * self._pb_compton_fraction_l * pb_acceptance,
                features[..., 3] * variable_acceptance,
                distance * 100.0 * self._air_mu_compton_l * variable_acceptance,
            ),
            axis=-1,
        )
        mass_sum = np.sum(component_mass, axis=-1)
        if np.any((scatter > 0.0) & (mass_sum <= np.finfo(np.float64).tiny)):
            raise RuntimeError(
                "Positive single-scatter rate has no physical material component."
            )
        fractions = np.divide(
            component_mass,
            mass_sum[..., np.newaxis],
            out=np.zeros_like(component_mass),
            where=mass_sum[..., np.newaxis] > np.finfo(np.float64).tiny,
        )
        component_distance = np.stack(
            (
                np.full_like(distance, float(response.fe_scatter_distance_m)),
                np.full_like(distance, float(response.pb_scatter_distance_m)),
                variable_distance,
                variable_distance,
            ),
            axis=-1,
        )
        nodes = self._detector_cone_scatter_distance_nodes_d
        log_nodes = np.log(nodes)
        log_distance = np.log(component_distance)
        upper = np.searchsorted(log_nodes, log_distance, side="right")
        upper = np.clip(upper, 1, int(nodes.size) - 1)
        lower = upper - 1
        upper_weight = (log_distance - log_nodes[lower]) / (
            log_nodes[upper] - log_nodes[lower]
        )
        upper_weight = np.clip(upper_weight, 0.0, 1.0)
        component_rate = scatter[..., np.newaxis] * fractions
        group_shape = scatter.shape[:-2]
        group_count = int(np.prod(group_shape, dtype=np.int64))
        source_count = int(scatter.shape[-2])
        line_count = int(scatter.shape[-1])
        distance_count = int(nodes.size)
        rate_nslk = component_rate.reshape(
            group_count,
            source_count,
            line_count,
            4,
        )
        lower_nslk = lower.reshape(rate_nslk.shape)
        upper_nslk = upper.reshape(rate_nslk.shape)
        weight_nslk = upper_weight.reshape(rate_nslk.shape)
        base_nslk = (
            np.arange(group_count, dtype=np.int64)[
                :, np.newaxis, np.newaxis, np.newaxis
            ]
            * line_count
            * distance_count
            + np.arange(line_count, dtype=np.int64)[
                np.newaxis, np.newaxis, :, np.newaxis
            ]
            * distance_count
        )
        output_size = group_count * line_count * distance_count
        lower_output = np.bincount(
            (base_nslk + lower_nslk).reshape(-1),
            weights=(rate_nslk * (1.0 - weight_nslk)).reshape(-1),
            minlength=output_size,
        )
        upper_output = np.bincount(
            (base_nslk + upper_nslk).reshape(-1),
            weights=(rate_nslk * weight_nslk).reshape(-1),
            minlength=output_size,
        )
        return (lower_output + upper_output).reshape(
            group_shape + (line_count, distance_count)
        )

    def _detector_cone_scatter_coefficients_torch(
        self,
        scatter_xvsl: object,
        features_xvslf: object,
        *,
        distance_nodes_d: object,
        line_energies_l: object,
        air_mu_l: object,
        fe_fraction_l: object,
        pb_fraction_l: object,
    ) -> object:
        """Return Torch line/distance coefficients without scalar loops."""
        import torch

        scatter = torch.as_tensor(scatter_xvsl)
        features = torch.as_tensor(
            features_xvslf,
            device=scatter.device,
            dtype=scatter.dtype,
        )
        response = self.additive_scatter_response
        if not isinstance(response, PhysicsOnlyNoncollidedTransportResponse) or tuple(
            features.shape
        ) != tuple(scatter.shape) + (len(TRANSPORT_FEATURE_ORDER),):
            raise RuntimeError(
                "Torch detector-cone coefficients require physics-only transport."
            )
        distance = features[..., TRANSPORT_DISTANCE_FEATURE_INDEX]
        radius = float(response.detector_radius_m)
        variable_distance = torch.clamp(0.5 * distance, min=radius)
        if bool(
            torch.any(~torch.isfinite(variable_distance))
            or torch.any(variable_distance < radius)
            or torch.any(variable_distance > DETECTOR_CONE_SCATTER_MAXIMUM_DISTANCE_M)
        ):
            raise ValueError(
                "Torch scatter distance is outside the authenticated response domain."
            )
        line_energies = torch.as_tensor(
            line_energies_l,
            device=scatter.device,
            dtype=scatter.dtype,
        )
        energy_shape = (1,) * (distance.ndim - 1) + (int(line_energies.numel()),)
        energies = line_energies.reshape(energy_shape)
        fe_acceptance = klein_nishina_forward_cone_fraction_torch(
            energies,
            detector_radius_m=radius,
            scatter_distance_m=torch.full_like(
                energies,
                float(response.fe_scatter_distance_m),
            ),
        )
        pb_acceptance = klein_nishina_forward_cone_fraction_torch(
            energies,
            detector_radius_m=radius,
            scatter_distance_m=torch.full_like(
                energies,
                float(response.pb_scatter_distance_m),
            ),
        )
        variable_acceptance = klein_nishina_forward_cone_fraction_torch(
            energies,
            detector_radius_m=radius,
            scatter_distance_m=variable_distance,
        )
        component_mass = torch.stack(
            (
                features[..., 0] * fe_fraction_l * fe_acceptance,
                features[..., 1] * pb_fraction_l * pb_acceptance,
                features[..., 3] * variable_acceptance,
                distance * 100.0 * air_mu_l * variable_acceptance,
            ),
            dim=-1,
        )
        mass_sum = torch.sum(component_mass, dim=-1)
        if bool(
            torch.any((scatter > 0.0) & (mass_sum <= torch.finfo(scatter.dtype).tiny))
        ):
            raise RuntimeError(
                "Positive Torch single-scatter rate has no material component."
            )
        fractions = torch.where(
            mass_sum.unsqueeze(-1) > torch.finfo(scatter.dtype).tiny,
            component_mass
            / torch.clamp(mass_sum.unsqueeze(-1), min=torch.finfo(scatter.dtype).tiny),
            torch.zeros_like(component_mass),
        )
        component_distance = torch.stack(
            (
                torch.full_like(distance, float(response.fe_scatter_distance_m)),
                torch.full_like(distance, float(response.pb_scatter_distance_m)),
                variable_distance,
                variable_distance,
            ),
            dim=-1,
        )
        nodes = torch.as_tensor(
            distance_nodes_d,
            device=scatter.device,
            dtype=scatter.dtype,
        )
        log_nodes = torch.log(nodes)
        log_distance = torch.log(component_distance)
        upper = torch.searchsorted(log_nodes, log_distance, right=True)
        upper = torch.clamp(upper, min=1, max=int(nodes.numel()) - 1)
        lower = upper - 1
        upper_weight = (log_distance - log_nodes[lower]) / (
            log_nodes[upper] - log_nodes[lower]
        )
        upper_weight = torch.clamp(upper_weight, min=0.0, max=1.0)
        component_rate = scatter.unsqueeze(-1) * fractions
        group_shape = tuple(scatter.shape[:-2])
        group_count = int(np.prod(group_shape, dtype=np.int64))
        source_count = int(scatter.shape[-2])
        line_count = int(scatter.shape[-1])
        distance_count = int(nodes.numel())
        rate_nslk = component_rate.reshape(group_count, source_count, line_count, 4)
        lower_nslk = lower.reshape(rate_nslk.shape)
        upper_nslk = upper.reshape(rate_nslk.shape)
        weight_nslk = upper_weight.reshape(rate_nslk.shape)
        base_nslk = (
            torch.arange(
                line_count,
                device=scatter.device,
                dtype=torch.long,
            )[None, None, :, None]
            * distance_count
        )
        lower_index = (base_nslk + lower_nslk).reshape(group_count, -1)
        upper_index = (base_nslk + upper_nslk).reshape(group_count, -1)
        output = torch.zeros(
            (group_count, line_count * distance_count),
            device=scatter.device,
            dtype=scatter.dtype,
        )
        output = output.scatter_add(
            1,
            lower_index,
            (rate_nslk * (1.0 - weight_nslk)).reshape(group_count, -1),
        )
        output = output.scatter_add(
            1,
            upper_index,
            (rate_nslk * weight_nslk).reshape(group_count, -1),
        )
        return output.reshape(group_shape + (line_count, distance_count))

    def _pre_dead_time_mean_numpy(
        self,
        total_line_contributions_xvsl: NDArray[np.float64],
        uncollided_line_contributions_xvsl: NDArray[np.float64],
        transport_features_xvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
        *,
        return_components: bool = False,
        return_physical_components: bool = False,
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return expected marked spectra before detector dead time."""
        total, uncollided, features, live_times = self._validated_numpy_inputs(
            total_line_contributions_xvsl,
            uncollided_line_contributions_xvsl,
            transport_features_xvslf,
            live_times_s_v,
        )
        live_scale = live_times.reshape(
            (1,) * (total.ndim - 3) + (int(total.shape[-3]), 1, 1)
        )
        total_counts = total * live_scale
        uncollided_counts = uncollided * live_scale
        direct = np.minimum(total_counts, uncollided_counts)
        scatter = total_counts - direct
        direct_phase_weights = self._direct_phase_weights_numpy(
            features,
            active_xvsl=direct > 0.0,
        )
        marked_direct = np.einsum(
            "...vsl,...vslc,clb->...vb",
            direct,
            direct_phase_weights,
            self._marked_direct_line_shapes_clb,
            optimize=True,
        )
        if isinstance(
            self.additive_scatter_response,
            PhysicsOnlyNoncollidedTransportResponse,
        ):
            scatter_coefficients = self._detector_cone_scatter_coefficients_numpy(
                scatter,
                features,
            )
            marked_scatter = np.einsum(
                "...vld,dlb->...vb",
                scatter_coefficients,
                self._marked_detector_cone_scatter_shapes_dlb,
                optimize=False,
            )
        else:
            order_weights = self._interaction_order_weights_numpy(features)
            scatter_by_line_order = np.sum(
                scatter[..., np.newaxis] * order_weights,
                axis=-3,
            )
            marked_scatter = np.einsum(
                "...vlo,lob->...vb",
                scatter_by_line_order,
                self._marked_scatter_order_shapes_lob,
                optimize=True,
            )
        marked_source = marked_direct + marked_scatter
        correction = self.low_rank_spectral_mean_correction
        if correction is not None:
            if return_physical_components:
                raise RuntimeError(
                    "Physical mark-component latents cannot be combined with "
                    "an undecomposed learned spectral correction."
                )
            marked_source = correction.apply_numpy(
                marked_source,
                total_counts,
                uncollided_counts,
                features,
            )
        background = (
            float(self.background_rate_cps)
            * live_times[:, np.newaxis]
            * self.background_shape_b[np.newaxis, :]
        )
        background = np.broadcast_to(
            background,
            marked_source.shape,
        ).copy()
        mean = marked_source + background
        if np.any(~np.isfinite(marked_source)) or np.any(marked_source < 0.0):
            raise RuntimeError(
                "Absolute detector Green marking produced invalid counts."
            )
        if return_components:
            return np.maximum(marked_source, 0.0), background
        if return_physical_components:
            return (
                np.maximum(marked_direct, 0.0),
                np.maximum(marked_scatter, 0.0),
                background,
            )
        return np.maximum(mean, 0.0)

    def predict_mean_numpy(
        self,
        total_line_contributions_xvsl: NDArray[np.float64],
        uncollided_line_contributions_xvsl: NDArray[np.float64],
        transport_features_xvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return asymptotic renewal means with exact conditional mark means."""
        source_mean, background_mean = self._pre_dead_time_mean_numpy(
            total_line_contributions_xvsl,
            uncollided_line_contributions_xvsl,
            transport_features_xvslf,
            live_times_s_v,
            return_components=True,
        )
        live_times = np.asarray(live_times_s_v, dtype=np.float64)
        pre_mean = background_mean[..., np.newaxis, :, :] + source_mean[
            ..., np.newaxis, :, :
        ] * self._rate_scale_nodes_j.reshape((1,) * (source_mean.ndim - 2) + (-1, 1, 1))
        pre_total = np.sum(pre_mean, axis=-1)
        rates = pre_total / live_times
        expected_total = pre_total / (1.0 + rates * float(self.dead_time_tau_s))
        probabilities = np.divide(
            pre_mean,
            pre_total[..., np.newaxis],
            out=np.zeros_like(pre_mean),
            where=pre_total[..., np.newaxis] > 0.0,
        )
        node_means = probabilities * expected_total[..., np.newaxis]
        return np.sum(
            node_means
            * self._rate_scale_weights_j.reshape(
                (1,) * (source_mean.ndim - 2) + (-1, 1, 1)
            ),
            axis=-3,
        )

    def pre_dead_time_components_numpy(
        self,
        total_line_contributions_xvsl: NDArray[np.float64],
        uncollided_line_contributions_xvsl: NDArray[np.float64],
        transport_features_xvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return source and background marked means before detector dead time."""
        source, background = self._pre_dead_time_mean_numpy(
            total_line_contributions_xvsl,
            uncollided_line_contributions_xvsl,
            transport_features_xvslf,
            live_times_s_v,
            return_components=True,
        )
        return (
            np.asarray(source, dtype=np.float64).copy(),
            np.asarray(background, dtype=np.float64).copy(),
        )

    @property
    def rate_scale_nodes(self) -> NDArray[np.float64]:
        """Return a defensive copy of the shared source-rate mixture nodes."""
        return self._rate_scale_nodes_j.copy()

    @property
    def rate_scale_weights(self) -> NDArray[np.float64]:
        """Return a defensive copy of the shared source-rate mixture weights."""
        return self._rate_scale_weights_j.copy()

    def _torch_constants(self, reference: object) -> tuple[object, ...]:
        """Return cached immutable constants matching a reference Torch tensor."""
        import torch

        tensor = torch.as_tensor(reference)
        key = (str(tensor.device), str(tensor.dtype))
        cached = self._torch_cache.get(key)
        if cached is not None:
            return cached
        arrays = (
            self.background_shape_b,
            self._marked_direct_line_shapes_clb,
            self._marked_direct_line_shapes_lb,
            self._marked_scatter_order_shapes_lob,
            self._air_mu_compton_l,
            self._fe_compton_fraction_l,
            self._pb_compton_fraction_l,
            self._obstacle_compton_fraction_l,
            self._direct_detection_factor_cl,
            self._direct_detection_factor_std_cl,
            self._scatter_detection_factor_lo,
            self._scatter_detection_factor_std_lo,
            self._detector_cone_scatter_distance_nodes_d,
            self._marked_detector_cone_scatter_shapes_dlb,
            self._detector_cone_scatter_detection_factor_dl,
            self._detector_cone_scatter_detection_factor_std_dl,
            self._line_energies_keV_l,
        )
        cached = tuple(
            torch.as_tensor(
                np.array(value, dtype=np.float64, copy=True),
                device=tensor.device,
                dtype=tensor.dtype,
            )
            for value in arrays
        )
        self._torch_cache[key] = cached
        return cached

    def _torch_likelihood_constants(
        self,
        reference: object,
    ) -> tuple[object, ...]:
        """Return cached quadrature nodes, weights, and spectral masks."""
        import torch

        tensor = torch.as_tensor(reference)
        key = (str(tensor.device), str(tensor.dtype))
        cached = self._torch_likelihood_cache.get(key)
        if cached is not None:
            return cached
        peak_mask = torch.as_tensor(
            np.array(self._photopeak_mask_b, copy=True),
            device=tensor.device,
            dtype=torch.bool,
        )
        continuum_group_mask = torch.as_tensor(
            np.array(
                self._continuum_group_mask_gb[:, ~self._photopeak_mask_b],
                dtype=np.float64,
                copy=True,
            ),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        cached = (
            torch.as_tensor(
                np.array(self._rate_scale_nodes_j, dtype=np.float64, copy=True),
                device=tensor.device,
                dtype=tensor.dtype,
            ),
            torch.as_tensor(
                np.array(
                    self._rate_scale_weights_j,
                    dtype=np.float64,
                    copy=True,
                ),
                device=tensor.device,
                dtype=tensor.dtype,
            ),
            peak_mask,
            continuum_group_mask,
        )
        self._torch_likelihood_cache[key] = cached
        return cached

    def _torch_mark_tree_constants(
        self,
        reference: object,
    ) -> tuple[object, ...]:
        """Return cached device-resident mark-tree masks and topology."""
        import torch

        tensor = torch.as_tensor(reference)
        key = (str(tensor.device), str(tensor.dtype))
        cached = self._torch_mark_tree_cache.get(key)
        if cached is not None:
            return cached
        floating = (
            self._mark_leaf_group_mask_hb,
            self._mark_tree_left_mask_tb,
            self._mark_tree_right_mask_tb,
        )
        integer = (
            self._mark_tree_domain_t,
            self._mark_tree_depth_t,
            self._mark_tree_left_child_t,
            self._mark_tree_right_child_t,
        )
        cached = tuple(
            torch.as_tensor(
                np.array(value, dtype=np.float64, copy=True),
                device=tensor.device,
                dtype=tensor.dtype,
            )
            for value in floating
        ) + tuple(
            torch.as_tensor(
                np.array(value, dtype=np.int64, copy=True),
                device=tensor.device,
                dtype=torch.long,
            )
            for value in integer
        )
        self._torch_mark_tree_cache[key] = cached
        return cached

    def _torch_component_likelihood_constants(
        self,
        reference: object,
    ) -> tuple[object, ...]:
        """Return cached response concentrations and fused mark projections."""
        import torch

        tensor = torch.as_tensor(reference)
        key = (str(tensor.device), str(tensor.dtype))
        cached = self._torch_component_likelihood_cache.get(key)
        if cached is not None:
            return cached
        (
            leaf_masks,
            left_masks,
            right_masks,
            domains,
            _depths,
            _left_children,
            _right_children,
        ) = self._torch_mark_tree_constants(tensor)
        peak_mask = self._torch_likelihood_constants(tensor)[2].to(
            dtype=tensor.dtype
        )
        domain_masks = torch.where(
            domains.unsqueeze(-1) < 0,
            torch.ones_like(left_masks),
            torch.where(
                domains.unsqueeze(-1) == 0,
                peak_mask.unsqueeze(0),
                (1.0 - peak_mask).unsqueeze(0),
            ),
        )
        cached = (
            torch.as_tensor(
                np.array(self._direct_response_concentration_l, copy=True),
                device=tensor.device,
                dtype=tensor.dtype,
            ),
            torch.as_tensor(
                np.array(self._scatter_response_concentration_l, copy=True),
                device=tensor.device,
                dtype=tensor.dtype,
            ),
            torch.cat((left_masks, right_masks, leaf_masks), dim=0).contiguous(),
            torch.cat(
                (left_masks, right_masks, domain_masks),
                dim=0,
            ).contiguous(),
            (torch.sum(leaf_masks, dim=1) > 1.0),
        )
        self._torch_component_likelihood_cache[key] = cached
        return cached

    def prepare_cross_observation_torch(
        self,
        observed_spectra_xqvb: object,
        *,
        reference: object,
    ) -> PreparedTorchCrossObservation:
        """Prepare exact observation-only likelihood terms on one device."""
        import torch

        tensor = torch.as_tensor(reference)
        observed = torch.as_tensor(
            observed_spectra_xqvb,
            device=tensor.device,
            dtype=tensor.dtype,
        )
        if observed.ndim < 3:
            raise ValueError("Torch cross-spectrum observations need three axes.")
        if int(observed.shape[-1]) != int(self._energy_axis_keV.size):
            raise ValueError("Torch cross-spectrum energy bins are misaligned.")
        invalid = torch.stack(
            (
                torch.any(~torch.isfinite(observed)),
                torch.any(observed < 0.0),
                torch.any(observed != torch.floor(observed)),
            )
        ).any()
        if bool(invalid.item()):
            raise ValueError("Torch cross-spectrum observations are invalid.")
        leading_shape = tuple(int(value) for value in observed.shape[:-3])
        action_count = int(np.prod(leading_shape, dtype=np.int64))
        if not leading_shape:
            action_count = 1
        sample_count = int(observed.shape[-3])
        view_count = int(observed.shape[-2])
        bin_count = int(observed.shape[-1])
        observed_flat = observed.reshape(
            action_count,
            sample_count,
            view_count,
            bin_count,
        )
        observed_total = torch.sum(observed_flat, dim=-1)
        log_factorial_by_bin = canonical_log_gamma_torch(observed_flat + 1.0)
        log_factorial_sum = torch.sum(log_factorial_by_bin, dim=-1)
        multinomial_constant = (
            canonical_log_gamma_torch(observed_total + 1.0) - log_factorial_sum
        )
        _, _, peak_mask, continuum_group_mask = self._torch_likelihood_constants(
            observed_flat
        )
        peak_observed = observed_flat[..., peak_mask]
        continuum_observed = observed_flat[..., ~peak_mask]
        peak_count = torch.sum(peak_observed, dim=-1)
        continuum_count = observed_total - peak_count
        continuum_group_observed = torch.einsum(
            "...b,gb->...g",
            continuum_observed,
            continuum_group_mask,
        )
        beta_binomial_constant = (
            canonical_log_gamma_torch(observed_total + 1.0)
            - canonical_log_gamma_torch(peak_count + 1.0)
            - canonical_log_gamma_torch(continuum_count + 1.0)
        )
        peak_multinomial_constant = canonical_log_gamma_torch(
            peak_count + 1.0
        ) - torch.sum(canonical_log_gamma_torch(peak_observed + 1.0), dim=-1)
        continuum_group_constant = canonical_log_gamma_torch(
            continuum_count + 1.0
        ) - torch.sum(
            canonical_log_gamma_torch(continuum_group_observed + 1.0),
            dim=-1,
        )
        continuum_within_constant = torch.sum(
            canonical_log_gamma_torch(continuum_group_observed + 1.0),
            dim=-1,
        ) - torch.sum(
            canonical_log_gamma_torch(continuum_observed + 1.0),
            dim=-1,
        )
        mark_observed_projection = None
        mark_leaf_log_factorial = None
        component = self.physical_component_discrepancy
        if (
            component is not None
            and component.mark_latent_model
            == "component_dirichlet_tree_hierarchical"
        ):
            leaf_masks = self._torch_mark_tree_constants(observed_flat)[0]
            projection_masks = self._torch_component_likelihood_constants(
                observed_flat
            )[2]
            mark_observed_projection = torch.einsum(
                "...b,mb->...m",
                observed_flat,
                projection_masks,
            )
            mark_leaf_log_factorial = torch.einsum(
                "...b,hb->...h",
                log_factorial_by_bin,
                leaf_masks,
            )
        return PreparedTorchCrossObservation(
            leading_shape=leading_shape,
            observed_asvb=observed_flat,
            observed_total_asv=observed_total,
            multinomial_constant_asv=multinomial_constant,
            peak_observed_asvp=peak_observed,
            continuum_observed_asvc=continuum_observed,
            peak_count_asv=peak_count,
            continuum_count_asv=continuum_count,
            beta_binomial_constant_asv=beta_binomial_constant,
            peak_multinomial_constant_asv=peak_multinomial_constant,
            continuum_group_observed_asvg=continuum_group_observed,
            continuum_group_constant_asv=continuum_group_constant,
            continuum_within_constant_asv=continuum_within_constant,
            mark_observed_projection_asvm=mark_observed_projection,
            mark_leaf_log_factorial_asvh=mark_leaf_log_factorial,
        )

    def _pre_dead_time_mean_torch(
        self,
        total_line_contributions_xvsl: object,
        uncollided_line_contributions_xvsl: object,
        transport_features_xvslf: object,
        live_times_s_v: object,
        *,
        return_components: bool = False,
        return_physical_components: bool = False,
        return_component_count_concentration: bool = False,
    ) -> object:
        """Return the Torch pre-dead-time mean and requested physical terms.

        The optional component count concentration reuses the exact direct
        phase weights and scatter coefficients already needed for the marked
        spectrum.  It is returned only with the three physical mean
        components, avoiding a second transport pass in the hierarchical
        production likelihood.
        """
        import torch

        total = torch.as_tensor(total_line_contributions_xvsl)
        if total.dtype != torch.float64:
            raise TypeError(
                "Production full-spectrum inference requires torch.float64."
            )
        uncollided = torch.as_tensor(
            uncollided_line_contributions_xvsl,
            device=total.device,
            dtype=total.dtype,
        )
        features = torch.as_tensor(
            transport_features_xvslf,
            device=total.device,
            dtype=total.dtype,
        )
        live_times = torch.as_tensor(
            live_times_s_v,
            device=total.device,
            dtype=total.dtype,
        )
        if (
            total.ndim < 3
            or total.shape[-1] != len(self._line_identity)
            or uncollided.shape != total.shape
            or features.shape != total.shape + (len(TRANSPORT_FEATURE_ORDER),)
            or tuple(live_times.shape) != (int(total.shape[-3]),)
        ):
            raise ValueError("Torch spectrum transport inputs are invalid.")
        if return_component_count_concentration and not return_physical_components:
            raise ValueError(
                "Component count concentration requires physical components."
            )
        tolerance = (
            128.0
            * torch.finfo(total.dtype).eps
            * torch.maximum(
                torch.ones((), device=total.device, dtype=total.dtype),
                torch.maximum(torch.abs(total), torch.abs(uncollided)),
            )
        )
        invalid = torch.stack(
            (
                torch.any(~torch.isfinite(total)),
                torch.any(total < 0.0),
                torch.any(~torch.isfinite(uncollided)),
                torch.any(uncollided < 0.0),
                torch.any(~torch.isfinite(features)),
                torch.any(features < 0.0),
                torch.any(~torch.isfinite(live_times)),
                torch.any(live_times <= 0.0),
                torch.any(uncollided > total + tolerance),
            )
        ).any()
        if bool(invalid.item()):
            raise ValueError(
                "Torch spectrum transport values are invalid or uncollided "
                "contributions cannot exceed total incident contributions."
            )
        uncollided = torch.minimum(uncollided, total)
        (
            background_shape,
            direct_phase_shapes,
            _direct_marginal_shapes,
            scatter_shapes,
            air_mu,
            fe_fraction,
            pb_fraction,
            obstacle_fraction,
            direct_detection_factor,
            direct_detection_factor_std,
            scatter_detection_factor,
            scatter_detection_factor_std,
            detector_cone_distance_nodes,
            detector_cone_scatter_shapes,
            detector_cone_detection_factor,
            detector_cone_detection_factor_std,
            line_energies,
        ) = self._torch_constants(total)
        live_scale = live_times.reshape(
            (1,) * (total.ndim - 3) + (int(total.shape[-3]), 1, 1)
        )
        total_counts = total * live_scale
        uncollided_counts = uncollided * live_scale
        direct = uncollided_counts
        scatter = total_counts - direct
        direct_phase_weights = self._direct_phase_weights_torch(
            features,
            active_xvsl=direct > 0.0,
        )
        marked_direct = torch.einsum(
            "...vsl,...vslc,clb->...vb",
            direct,
            direct_phase_weights,
            direct_phase_shapes,
        )
        if isinstance(
            self.additive_scatter_response,
            PhysicsOnlyNoncollidedTransportResponse,
        ):
            scatter_coefficients = self._detector_cone_scatter_coefficients_torch(
                scatter,
                features,
                distance_nodes_d=detector_cone_distance_nodes,
                line_energies_l=line_energies,
                air_mu_l=air_mu,
                fe_fraction_l=fe_fraction,
                pb_fraction_l=pb_fraction,
            )
            marked_scatter = torch.einsum(
                "...vld,dlb->...vb",
                scatter_coefficients,
                detector_cone_scatter_shapes,
            )
        else:
            tau = (
                features[..., 0] * fe_fraction
                + features[..., 1] * pb_fraction
                + features[..., 3]
                + features[..., TRANSPORT_DISTANCE_FEATURE_INDEX] * 100.0 * air_mu
            )
            tau = torch.clamp(tau, min=0.0)
            exact_orders = torch.arange(
                1,
                int(self.maximum_scatter_order),
                device=total.device,
                dtype=total.dtype,
            )
            denominator = -torch.expm1(-tau)
            tiny = torch.finfo(total.dtype).tiny
            log_exact = (
                -tau.unsqueeze(-1)
                + torch.log(torch.clamp(tau, min=tiny)).unsqueeze(-1) * exact_orders
                - torch.lgamma(exact_orders + 1.0)
                - torch.log(torch.clamp(denominator, min=tiny)).unsqueeze(-1)
            )
            exact = torch.where(
                tau.unsqueeze(-1) > 0.0,
                torch.exp(log_exact),
                torch.zeros_like(log_exact),
            )
            tail = torch.clamp(1.0 - torch.sum(exact, dim=-1), min=0.0)
            order_weights = torch.cat((exact, tail.unsqueeze(-1)), dim=-1)
            zero_tau = tau <= 0.0
            first = torch.where(
                zero_tau,
                torch.ones_like(order_weights[..., 0]),
                order_weights[..., 0],
            )
            rest = torch.where(
                zero_tau.unsqueeze(-1),
                torch.zeros_like(order_weights[..., 1:]),
                order_weights[..., 1:],
            )
            order_weights = torch.cat((first.unsqueeze(-1), rest), dim=-1)
            order_weights = order_weights / torch.clamp(
                torch.sum(order_weights, dim=-1, keepdim=True),
                min=tiny,
            )
            scatter_by_line_order = torch.sum(
                scatter.unsqueeze(-1) * order_weights,
                dim=-3,
            )
            marked_scatter = torch.einsum(
                "...vlo,lob->...vb",
                scatter_by_line_order,
                scatter_shapes,
            )
        count_concentration = None
        if return_component_count_concentration:
            component = self.physical_component_discrepancy
            if component is None:
                raise RuntimeError(
                    "Physical-component discrepancy is not configured."
                )
            direct_rate = torch.einsum(
                "...vsl,...vslc,cl->...v",
                uncollided,
                direct_phase_weights,
                direct_detection_factor,
            )
            direct_green_std = torch.einsum(
                "...vsl,...vslc,cl->...v",
                uncollided,
                direct_phase_weights,
                direct_detection_factor_std,
            )
            if isinstance(
                self.additive_scatter_response,
                PhysicsOnlyNoncollidedTransportResponse,
            ):
                coefficient_live_scale = live_times.reshape(
                    (1,) * (scatter_coefficients.ndim - 3)
                    + (int(scatter_coefficients.shape[-3]), 1, 1)
                )
                scatter_rate_coefficients = (
                    scatter_coefficients / coefficient_live_scale
                )
                scatter_rate = torch.einsum(
                    "...vld,dl->...v",
                    scatter_rate_coefficients,
                    detector_cone_detection_factor,
                )
                scatter_green_std = torch.einsum(
                    "...vld,dl->...v",
                    scatter_rate_coefficients,
                    detector_cone_detection_factor_std,
                )
            else:
                scatter_rate_by_line_order = torch.sum(
                    (total - uncollided).unsqueeze(-1) * order_weights,
                    dim=-3,
                )
                scatter_rate = torch.einsum(
                    "...vlo,lo->...v",
                    scatter_rate_by_line_order,
                    scatter_detection_factor,
                )
                scatter_green_std = torch.einsum(
                    "...vlo,lo->...v",
                    scatter_rate_by_line_order,
                    scatter_detection_factor_std,
                )
            total_rate = direct_rate + scatter_rate
            denominator = (
                torch.square(direct_rate)
                / float(component.count_uncollided_concentration)
                + torch.square(scatter_rate)
                / float(component.count_scatter_concentration)
                + torch.square(direct_green_std + scatter_green_std)
            )
            count_concentration = torch.where(
                total_rate > 0.0,
                torch.square(total_rate)
                / torch.clamp(
                    denominator,
                    min=torch.finfo(total.dtype).tiny,
                ),
                torch.full_like(total_rate, 1.0e15),
            )
        marked_source = marked_direct + marked_scatter
        correction = self.low_rank_spectral_mean_correction
        if correction is not None:
            if return_physical_components:
                raise RuntimeError(
                    "Physical mark-component latents cannot be combined with "
                    "an undecomposed learned spectral correction."
                )
            marked_source = correction.apply_torch(
                marked_source,
                total_counts,
                uncollided_counts,
                features,
            )
        background = (
            float(self.background_rate_cps)
            * live_times[:, None]
            * background_shape[None, :]
        )
        background = torch.broadcast_to(
            background,
            marked_source.shape,
        )
        if correction is None and bool(
            torch.any(~torch.isfinite(marked_source)) or torch.any(marked_source < 0.0)
        ):
            raise RuntimeError(
                "Torch absolute detector Green marking produced invalid counts."
            )
        if return_components:
            return torch.clamp(marked_source, min=0.0), background
        if return_physical_components:
            physical_components = (
                torch.clamp(marked_direct, min=0.0),
                torch.clamp(marked_scatter, min=0.0),
                background,
            )
            if return_component_count_concentration:
                return physical_components + (count_concentration,)
            return physical_components
        return torch.clamp(marked_source + background, min=0.0)

    def predict_mean_torch(
        self,
        total_line_contributions_xvsl: object,
        uncollided_line_contributions_xvsl: object,
        transport_features_xvslf: object,
        live_times_s_v: object,
    ) -> object:
        """Return the Torch renewal/mark predictive mean."""
        import torch

        source_mean, background_mean = self._pre_dead_time_mean_torch(
            total_line_contributions_xvsl,
            uncollided_line_contributions_xvsl,
            transport_features_xvslf,
            live_times_s_v,
            return_components=True,
        )
        live_times = torch.as_tensor(
            live_times_s_v,
            device=source_mean.device,
            dtype=source_mean.dtype,
        )
        nodes = torch.as_tensor(
            np.array(self._rate_scale_nodes_j, copy=True),
            device=source_mean.device,
            dtype=torch.float64,
        )
        node_weights = torch.as_tensor(
            np.array(self._rate_scale_weights_j, copy=True),
            device=source_mean.device,
            dtype=torch.float64,
        )
        node_shape = (1,) * (source_mean.ndim - 2) + (int(nodes.numel()), 1, 1)
        pre_mean = background_mean.unsqueeze(-3) + source_mean.unsqueeze(
            -3
        ) * nodes.reshape(node_shape)
        pre_total = torch.sum(pre_mean, dim=-1)
        expected_total = pre_total / (
            1.0 + pre_total / live_times * float(self.dead_time_tau_s)
        )
        probabilities = torch.where(
            pre_total.unsqueeze(-1) > 0.0,
            pre_mean
            / torch.clamp(
                pre_total.unsqueeze(-1),
                min=torch.finfo(pre_mean.dtype).tiny,
            ),
            torch.zeros_like(pre_mean),
        )
        node_means = probabilities * expected_total.unsqueeze(-1)
        return torch.sum(
            node_means * node_weights.reshape(node_shape),
            dim=-3,
        )

    def _detector_response_mark_concentration_numpy(
        self,
        total_line_contributions_xnvsl: NDArray[np.float64],
        uncollided_line_contributions_xnvsl: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return conservative finite-corpus concentration per state/view."""
        total = np.asarray(total_line_contributions_xnvsl, dtype=np.float64)
        uncollided = np.minimum(
            np.asarray(
                uncollided_line_contributions_xnvsl,
                dtype=np.float64,
            ),
            total,
        )
        direct_by_line = np.sum(uncollided, axis=-2)
        scatter_by_line = np.sum(total - uncollided, axis=-2)
        concentrations = np.concatenate(
            (
                np.broadcast_to(
                    self._direct_response_concentration_l,
                    direct_by_line.shape,
                ),
                np.broadcast_to(
                    self._scatter_response_concentration_l,
                    scatter_by_line.shape,
                ),
            ),
            axis=-1,
        )
        active = np.concatenate(
            (direct_by_line > 0.0, scatter_by_line > 0.0),
            axis=-1,
        )
        return np.min(
            np.where(active, concentrations, 1.0e15),
            axis=-1,
        )

    def _base_mark_concentration_numpy(
        self,
        total_line_contributions_xnvsl: NDArray[np.float64],
        uncollided_line_contributions_xnvsl: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Combine physical and finite-response mark concentrations."""
        if uncollided_line_contributions_xnvsl is None:
            raise ValueError(
                "Detector-response uncertainty requires uncollided line contributions."
            )
        response_concentration = self._detector_response_mark_concentration_numpy(
            total_line_contributions_xnvsl,
            uncollided_line_contributions_xnvsl,
        )
        component = self.physical_component_discrepancy
        if component is not None:
            direct_fraction, scatter_fraction = self._source_component_fractions_numpy(
                total_line_contributions_xnvsl,
                uncollided_line_contributions_xnvsl,
            )
            reciprocal = np.square(direct_fraction) / (
                float(component.mark_uncollided_concentration) + 1.0
            ) + np.square(scatter_fraction) / (
                float(component.mark_scatter_concentration) + 1.0
            )
            physical_concentration = np.maximum(
                np.divide(
                    1.0,
                    np.maximum(reciprocal, np.finfo(np.float64).tiny),
                )
                - 1.0,
                np.finfo(np.float64).tiny,
            )
        else:
            low = self.mark_concentration_source
            total = np.asarray(
                total_line_contributions_xnvsl,
                dtype=np.float64,
            )
            output_shape = total.shape[:-2]
            high = self.mark_concentration_multi_isotope
            if low is None:
                physical_concentration = np.full(
                    output_shape,
                    1.0e15,
                    dtype=np.float64,
                )
            elif high is None or len(self._mark_isotope_names) < 2:
                physical_concentration = np.full(
                    output_shape,
                    float(low),
                    dtype=np.float64,
                )
            else:
                isotope_totals = np.einsum(
                    "...vsl,li->...vi",
                    total,
                    self._line_to_mark_isotope_li,
                    optimize=True,
                )
                total_rate = np.sum(isotope_totals, axis=-1, keepdims=True)
                fractions = np.divide(
                    isotope_totals,
                    total_rate,
                    out=np.zeros_like(isotope_totals),
                    where=total_rate > 0.0,
                )
                entropy = -np.sum(
                    special.xlogy(fractions, fractions),
                    axis=-1,
                ) / np.log(float(len(self._mark_isotope_names)))
                entropy = np.clip(entropy, 0.0, 1.0)
                physical_concentration = np.exp(
                    np.log(float(low))
                    + entropy * (np.log(float(high)) - np.log(float(low)))
                )
        reciprocal = 1.0 / (response_concentration + 1.0) + 1.0 / (
            physical_concentration + 1.0
        )
        return np.maximum(
            1.0 / np.maximum(reciprocal, np.finfo(np.float64).tiny) - 1.0,
            np.finfo(np.float64).tiny,
        )

    def _sample_component_tree_probabilities_numpy(
        self,
        probabilities_xb: NDArray[np.float64],
        tree_concentration_xt: NDArray[np.float64],
        leaf_concentration_xh: NDArray[np.float64],
        *,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Draw conditional mark probabilities from the production tree.

        All leading dimensions are sampled in one batched path.  The only
        loop is over the immutable levels of the balanced energy tree; it does
        not scale with particles, stations, sources, shield poses, or bins.
        """
        probabilities = np.asarray(probabilities_xb, dtype=np.float64)
        tree_concentration = np.asarray(tree_concentration_xt, dtype=np.float64)
        leaf_concentration = np.asarray(leaf_concentration_xh, dtype=np.float64)
        tree_count = int(self._mark_tree_left_mask_tb.shape[0])
        leaf_count = int(self._mark_leaf_group_mask_hb.shape[0])
        if (
            probabilities.ndim < 1
            or probabilities.shape[-1] != self._energy_axis_keV.size
            or tree_concentration.shape != probabilities.shape[:-1] + (tree_count,)
            or leaf_concentration.shape != probabilities.shape[:-1] + (leaf_count,)
            or np.any(~np.isfinite(probabilities))
            or np.any(probabilities < 0.0)
            or np.any(~np.isfinite(tree_concentration))
            or np.any(tree_concentration <= 0.0)
            or np.any(~np.isfinite(leaf_concentration))
            or np.any(leaf_concentration <= 0.0)
        ):
            raise ValueError("Component-tree sampling inputs are invalid.")
        probability_total = np.sum(probabilities, axis=-1)
        if np.any(np.abs(probability_total - 1.0) > 1.0e-10):
            raise ValueError("Component-tree probabilities must sum to one.")

        left_mass = np.einsum(
            "...b,tb->...t",
            probabilities,
            self._mark_tree_left_mask_tb,
            optimize=True,
        )
        right_mass = np.einsum(
            "...b,tb->...t",
            probabilities,
            self._mark_tree_right_mask_tb,
            optimize=True,
        )
        parent_mass = left_mass + right_mass
        branch_probability = np.divide(
            left_mass,
            parent_mass,
            out=np.zeros_like(left_mass),
            where=parent_mass > 0.0,
        )
        interior = (branch_probability > 0.0) & (branch_probability < 1.0)
        sampled_branch_probability = branch_probability.copy()
        sampled_branch_probability[interior] = rng.beta(
            tree_concentration[interior] * branch_probability[interior],
            tree_concentration[interior] * (1.0 - branch_probability[interior]),
        )
        node_mass = np.zeros_like(sampled_branch_probability)
        node_mass[..., 0] = 1.0
        sampled_leaf_mass = np.zeros(
            probabilities.shape[:-1] + (leaf_count,),
            dtype=np.float64,
        )
        maximum_depth = int(np.max(self._mark_tree_depth_t))
        for depth in range(maximum_depth + 1):
            node_ids = np.flatnonzero(self._mark_tree_depth_t == depth)
            if node_ids.size == 0:
                continue
            parent = node_mass[..., node_ids]
            left_value = parent * sampled_branch_probability[..., node_ids]
            right_value = parent - left_value
            left_target = self._mark_tree_left_child_t[node_ids]
            right_target = self._mark_tree_right_child_t[node_ids]
            left_nodes = left_target >= 0
            right_nodes = right_target >= 0
            if np.any(left_nodes):
                node_mass[..., left_target[left_nodes]] = left_value[..., left_nodes]
            if np.any(right_nodes):
                node_mass[..., right_target[right_nodes]] = right_value[
                    ..., right_nodes
                ]
            if np.any(~left_nodes):
                sampled_leaf_mass[..., -left_target[~left_nodes] - 1] = left_value[
                    ..., ~left_nodes
                ]
            if np.any(~right_nodes):
                sampled_leaf_mass[..., -right_target[~right_nodes] - 1] = right_value[
                    ..., ~right_nodes
                ]

        base_leaf_mass = np.einsum(
            "...b,hb->...h",
            probabilities,
            self._mark_leaf_group_mask_hb,
            optimize=True,
        )
        mapped_base_leaf_mass = np.einsum(
            "...h,hb->...b",
            base_leaf_mass,
            self._mark_leaf_group_mask_hb,
            optimize=True,
        )
        within_probability = np.divide(
            probabilities,
            mapped_base_leaf_mass,
            out=np.zeros_like(probabilities),
            where=mapped_base_leaf_mass > 0.0,
        )
        mapped_leaf_concentration = np.einsum(
            "...h,hb->...b",
            leaf_concentration,
            self._mark_leaf_group_mask_hb,
            optimize=True,
        )
        within_alpha = within_probability * mapped_leaf_concentration
        positive_within = within_alpha > 0.0
        within_gamma = rng.gamma(shape=np.where(positive_within, within_alpha, 1.0))
        within_gamma = np.where(positive_within, within_gamma, 0.0)
        within_group_sum = np.einsum(
            "...b,hb->...h",
            within_gamma,
            self._mark_leaf_group_mask_hb,
            optimize=True,
        )
        within_sum_by_bin = np.einsum(
            "...h,hb->...b",
            within_group_sum,
            self._mark_leaf_group_mask_hb,
            optimize=True,
        )
        sampled_within = np.divide(
            within_gamma,
            within_sum_by_bin,
            out=within_probability.copy(),
            where=within_sum_by_bin > 0.0,
        )
        sampled_leaf_mass_by_bin = np.einsum(
            "...h,hb->...b",
            sampled_leaf_mass,
            self._mark_leaf_group_mask_hb,
            optimize=True,
        )
        sampled = sampled_leaf_mass_by_bin * sampled_within
        normalization = np.sum(sampled, axis=-1, keepdims=True)
        sampled = np.divide(
            sampled,
            normalization,
            out=probabilities.copy(),
            where=normalization > 0.0,
        )
        if (
            np.any(~np.isfinite(sampled))
            or np.any(sampled < 0.0)
            or np.any(np.abs(np.sum(sampled, axis=-1) - 1.0) > 1.0e-10)
        ):
            raise RuntimeError("Component-tree probability sampling failed.")
        return np.asarray(sampled, dtype=np.float64)

    def _component_tree_mark_log_numpy(
        self,
        observed_xqvb: NDArray[np.float64],
        probabilities_xnjvb: NDArray[np.float64],
        tree_concentration_xnjvt: NDArray[np.float64],
        leaf_concentration_xnjvh: NDArray[np.float64],
        *,
        return_factors: bool = False,
    ) -> (
        NDArray[np.float64]
        | tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ]
    ):
        """Return the component-aware Dirichlet-tree mark likelihood."""
        observed = np.asarray(observed_xqvb, dtype=np.float64)
        probabilities = np.asarray(probabilities_xnjvb, dtype=np.float64)
        tree_concentration = np.asarray(
            tree_concentration_xnjvt,
            dtype=np.float64,
        )
        leaf_concentration = np.asarray(
            leaf_concentration_xnjvh,
            dtype=np.float64,
        )
        if (
            observed.ndim < 3
            or probabilities.ndim != observed.ndim + 1
            or probabilities.shape[:-4] != observed.shape[:-3]
            or probabilities.shape[-2:] != observed.shape[-2:]
            or tree_concentration.shape
            != probabilities.shape[:-1] + (self._mark_tree_left_mask_tb.shape[0],)
            or leaf_concentration.shape
            != probabilities.shape[:-1] + (self._mark_leaf_group_mask_hb.shape[0],)
            or np.any(~np.isfinite(tree_concentration))
            or np.any(tree_concentration <= 0.0)
            or np.any(~np.isfinite(leaf_concentration))
            or np.any(leaf_concentration <= 0.0)
        ):
            raise ValueError("Component-tree NumPy likelihood inputs are invalid.")
        observed_expanded = observed[..., :, np.newaxis, np.newaxis, :, :]
        probability_expanded = probabilities[..., np.newaxis, :, :, :, :]
        tree_concentration_expanded = tree_concentration[..., np.newaxis, :, :, :, :]
        leaf_concentration_expanded = leaf_concentration[..., np.newaxis, :, :, :, :]
        left_count = np.einsum(
            "...b,tb->...t",
            observed_expanded,
            self._mark_tree_left_mask_tb,
            optimize=True,
        )
        right_count = np.einsum(
            "...b,tb->...t",
            observed_expanded,
            self._mark_tree_right_mask_tb,
            optimize=True,
        )
        left_probability_mass = np.einsum(
            "...b,tb->...t",
            probability_expanded,
            self._mark_tree_left_mask_tb,
            optimize=True,
        )
        right_probability_mass = np.einsum(
            "...b,tb->...t",
            probability_expanded,
            self._mark_tree_right_mask_tb,
            optimize=True,
        )
        parent_probability_mass = left_probability_mass + right_probability_mass
        left_probability = np.divide(
            left_probability_mass,
            parent_probability_mass,
            out=np.zeros_like(left_probability_mass),
            where=parent_probability_mass > 0.0,
        )
        parent_count = left_count + right_count
        interior = (left_probability > 0.0) & (left_probability < 1.0)
        safe_alpha = np.where(
            interior,
            tree_concentration_expanded * left_probability,
            1.0,
        )
        safe_beta = np.where(
            interior,
            tree_concentration_expanded * (1.0 - left_probability),
            1.0,
        )
        tree_log = (
            canonical_log_gamma_numpy(parent_count + 1.0)
            - canonical_log_gamma_numpy(left_count + 1.0)
            - canonical_log_gamma_numpy(right_count + 1.0)
            + canonical_log_gamma_numpy(left_count + safe_alpha)
            + canonical_log_gamma_numpy(right_count + safe_beta)
            - canonical_log_gamma_numpy(parent_count + safe_alpha + safe_beta)
            - canonical_log_gamma_numpy(safe_alpha)
            - canonical_log_gamma_numpy(safe_beta)
            + canonical_log_gamma_numpy(safe_alpha + safe_beta)
        )
        tree_log = np.where(
            interior,
            tree_log,
            np.where(
                (left_probability <= 0.0) & (left_count == 0.0),
                0.0,
                np.where(
                    (left_probability >= 1.0) & (right_count == 0.0),
                    0.0,
                    -np.inf,
                ),
            ),
        )

        leaf_probability_mass = np.einsum(
            "...b,hb->...h",
            probability_expanded,
            self._mark_leaf_group_mask_hb,
            optimize=True,
        )
        mapped_leaf_probability = np.einsum(
            "...h,hb->...b",
            leaf_probability_mass,
            self._mark_leaf_group_mask_hb,
            optimize=True,
        )
        within_probability = np.divide(
            probability_expanded,
            mapped_leaf_probability,
            out=np.zeros_like(probability_expanded),
            where=mapped_leaf_probability > 0.0,
        )
        mapped_leaf_concentration = np.einsum(
            "...h,hb->...b",
            leaf_concentration_expanded,
            self._mark_leaf_group_mask_hb,
            optimize=True,
        )
        alpha = within_probability * mapped_leaf_concentration
        active_increment = (observed_expanded > 0.0) & (alpha > 0.0)
        safe_increment_alpha = np.where(active_increment, alpha, 1.0)
        safe_increment_observed = np.where(
            active_increment,
            observed_expanded,
            0.0,
        )
        increment = np.where(
            active_increment,
            canonical_log_gamma_numpy(safe_increment_alpha + safe_increment_observed)
            - canonical_log_gamma_numpy(safe_increment_alpha),
            0.0,
        )
        leaf_count = np.einsum(
            "...b,hb->...h",
            observed_expanded,
            self._mark_leaf_group_mask_hb,
            optimize=True,
        )
        observed_factorial_by_leaf = np.einsum(
            "...b,hb->...h",
            canonical_log_gamma_numpy(observed_expanded + 1.0),
            self._mark_leaf_group_mask_hb,
            optimize=True,
        )
        increment_by_leaf = np.einsum(
            "...b,hb->...h",
            increment,
            self._mark_leaf_group_mask_hb,
            optimize=True,
        )
        safe_leaf_concentration = np.where(
            np.sum(self._mark_leaf_group_mask_hb, axis=1) > 1.0,
            leaf_concentration_expanded,
            1.0,
        )
        leaf_log = (
            canonical_log_gamma_numpy(leaf_count + 1.0)
            - observed_factorial_by_leaf
            + canonical_log_gamma_numpy(safe_leaf_concentration)
            - canonical_log_gamma_numpy(safe_leaf_concentration + leaf_count)
            + increment_by_leaf
        )
        leaf_log = np.where(
            np.sum(self._mark_leaf_group_mask_hb, axis=1) > 1.0,
            leaf_log,
            0.0,
        )
        impossible = np.any(
            (observed_expanded > 0.0) & (probability_expanded <= 0.0),
            axis=-1,
        )
        total_observed = np.sum(observed_expanded, axis=-1)
        result = np.sum(tree_log, axis=-1) + np.sum(leaf_log, axis=-1)
        result = np.where(
            total_observed == 0.0,
            0.0,
            np.where(impossible, -np.inf, result),
        )
        if np.any(np.isnan(result)) or np.any(np.isposinf(result)):
            raise RuntimeError("Component-tree NumPy likelihood is invalid.")
        resolved_result = np.asarray(result, dtype=np.float64)
        if return_factors:
            return (
                resolved_result,
                np.asarray(tree_log, dtype=np.float64),
                np.asarray(leaf_log, dtype=np.float64),
            )
        return resolved_result

    def _component_tree_mark_log_torch(
        self,
        observed_xqvb: object,
        probabilities_xnjvb: object,
        tree_concentration_xnjvt: object,
        leaf_concentration_xnjvh: object,
        *,
        prepared_observation: PreparedTorchCrossObservation | None = None,
    ) -> object:
        """Return the canonical component-tree mark likelihood."""
        import torch

        probabilities = torch.as_tensor(probabilities_xnjvb)
        prepared = prepared_observation
        if prepared is None:
            prepared = self.prepare_cross_observation_torch(
                observed_xqvb,
                reference=probabilities,
            )
        observed = torch.as_tensor(
            prepared.restored(prepared.observed_asvb),
            device=probabilities.device,
            dtype=probabilities.dtype,
        )
        tree_concentration = torch.as_tensor(
            tree_concentration_xnjvt,
            device=probabilities.device,
            dtype=probabilities.dtype,
        )
        leaf_concentration = torch.as_tensor(
            leaf_concentration_xnjvh,
            device=probabilities.device,
            dtype=probabilities.dtype,
        )
        leaf_masks, left_masks, right_masks, *_ = self._torch_mark_tree_constants(
            probabilities
        )
        (
            _direct_reference,
            _scatter_reference,
            projection_masks,
            _tree_projection_masks,
            nondegenerate_leaf,
        ) = self._torch_component_likelihood_constants(probabilities)
        if (
            observed.ndim < 3
            or probabilities.ndim != observed.ndim + 1
            or tuple(probabilities.shape[:-4]) != tuple(observed.shape[:-3])
            or tuple(probabilities.shape[-2:]) != tuple(observed.shape[-2:])
            or tuple(tree_concentration.shape)
            != tuple(probabilities.shape[:-1]) + (int(left_masks.shape[0]),)
            or tuple(leaf_concentration.shape)
            != tuple(probabilities.shape[:-1]) + (int(leaf_masks.shape[0]),)
        ):
            raise ValueError("Component-tree Torch likelihood inputs are invalid.")
        concentration_invalid = torch.stack(
            (
                torch.any(~torch.isfinite(tree_concentration)),
                torch.any(tree_concentration <= 0.0),
                torch.any(~torch.isfinite(leaf_concentration)),
                torch.any(leaf_concentration <= 0.0),
            )
        ).any()
        if bool(concentration_invalid.item()):
            raise ValueError("Component-tree Torch concentrations are invalid.")
        observed_expanded = observed.unsqueeze(-3).unsqueeze(-3)
        probability_expanded = probabilities.unsqueeze(-5)
        tree_concentration_expanded = tree_concentration.unsqueeze(-5)
        leaf_concentration_expanded = leaf_concentration.unsqueeze(-5)
        tree_count = int(left_masks.shape[0])
        leaf_count_value = int(leaf_masks.shape[0])
        if prepared.mark_observed_projection_asvm is None:
            observed_projection = torch.einsum(
                "...b,mb->...m",
                observed_expanded,
                projection_masks,
            )
        else:
            observed_projection = prepared.restored(
                prepared.mark_observed_projection_asvm
            ).unsqueeze(-3).unsqueeze(-3)
        probability_projection = torch.einsum(
            "...b,mb->...m",
            probability_expanded,
            projection_masks,
        )
        left_count, right_count, leaf_count = torch.split(
            observed_projection,
            (tree_count, tree_count, leaf_count_value),
            dim=-1,
        )
        (
            left_probability_mass,
            right_probability_mass,
            leaf_probability_mass,
        ) = torch.split(
            probability_projection,
            (tree_count, tree_count, leaf_count_value),
            dim=-1,
        )
        parent_probability_mass = left_probability_mass + right_probability_mass
        left_probability = torch.where(
            parent_probability_mass > 0.0,
            left_probability_mass
            / torch.clamp(
                parent_probability_mass,
                min=torch.finfo(probabilities.dtype).tiny,
            ),
            torch.zeros_like(left_probability_mass),
        )
        parent_count = left_count + right_count
        interior = (left_probability > 0.0) & (left_probability < 1.0)
        safe_alpha = torch.where(
            interior,
            tree_concentration_expanded * left_probability,
            torch.ones_like(left_probability),
        )
        safe_beta = torch.where(
            interior,
            tree_concentration_expanded * (1.0 - left_probability),
            torch.ones_like(left_probability),
        )
        tree_arguments = torch.broadcast_tensors(
            parent_count + 1.0,
            left_count + 1.0,
            right_count + 1.0,
            left_count + safe_alpha,
            right_count + safe_beta,
            parent_count + safe_alpha + safe_beta,
            safe_alpha,
            safe_beta,
            safe_alpha + safe_beta,
        )
        tree_gamma = _canonical_log_gamma_torch_unchecked(
            torch.stack(tree_arguments, dim=0)
        )
        tree_terms = torch.unbind(tree_gamma, dim=0)
        tree_log = (
            tree_terms[0]
            - tree_terms[1]
            - tree_terms[2]
            + tree_terms[3]
            + tree_terms[4]
            - tree_terms[5]
            - tree_terms[6]
            - tree_terms[7]
            + tree_terms[8]
        )
        tree_log = torch.where(
            interior,
            tree_log,
            torch.where(
                (left_probability <= 0.0) & (left_count == 0.0),
                torch.zeros_like(tree_log),
                torch.where(
                    (left_probability >= 1.0) & (right_count == 0.0),
                    torch.zeros_like(tree_log),
                    torch.full_like(tree_log, float("-inf")),
                ),
            ),
        )
        mapped_leaf_probability = torch.einsum(
            "...h,hb->...b",
            leaf_probability_mass,
            leaf_masks,
        )
        within_probability = torch.where(
            mapped_leaf_probability > 0.0,
            probability_expanded
            / torch.clamp(
                mapped_leaf_probability,
                min=torch.finfo(probabilities.dtype).tiny,
            ),
            torch.zeros_like(probability_expanded),
        )
        mapped_leaf_concentration = torch.einsum(
            "...h,hb->...b",
            leaf_concentration_expanded,
            leaf_masks,
        )
        alpha = within_probability * mapped_leaf_concentration
        active_increment = (observed_expanded > 0.0) & (alpha > 0.0)
        safe_increment_alpha = torch.where(
            active_increment,
            alpha,
            torch.ones_like(alpha),
        )
        safe_increment_observed = torch.where(
            active_increment,
            observed_expanded,
            torch.zeros_like(observed_expanded),
        )
        increment_arguments = torch.broadcast_tensors(
            safe_increment_alpha + safe_increment_observed,
            safe_increment_alpha,
        )
        increment_gamma = _canonical_log_gamma_torch_unchecked(
            torch.stack(increment_arguments, dim=0)
        )
        increment = torch.where(
            active_increment,
            increment_gamma[0] - increment_gamma[1],
            torch.zeros_like(alpha),
        )
        if prepared.mark_leaf_log_factorial_asvh is None:
            observed_factorial_by_leaf = torch.einsum(
                "...b,hb->...h",
                _canonical_log_gamma_torch_unchecked(observed_expanded + 1.0),
                leaf_masks,
            )
        else:
            observed_factorial_by_leaf = prepared.restored(
                prepared.mark_leaf_log_factorial_asvh
            ).unsqueeze(-3).unsqueeze(-3)
        increment_by_leaf = torch.einsum("...b,hb->...h", increment, leaf_masks)
        safe_leaf_concentration = torch.where(
            nondegenerate_leaf,
            leaf_concentration_expanded,
            torch.ones_like(leaf_concentration_expanded),
        )
        leaf_arguments = torch.broadcast_tensors(
            leaf_count + 1.0,
            safe_leaf_concentration,
            safe_leaf_concentration + leaf_count,
        )
        leaf_gamma = _canonical_log_gamma_torch_unchecked(
            torch.stack(leaf_arguments, dim=0)
        )
        leaf_log = (
            leaf_gamma[0]
            - observed_factorial_by_leaf
            + leaf_gamma[1]
            - leaf_gamma[2]
            + increment_by_leaf
        )
        leaf_log = torch.where(
            nondegenerate_leaf,
            leaf_log,
            torch.zeros_like(leaf_log),
        )
        impossible = torch.any(
            (observed_expanded > 0.0) & (probability_expanded <= 0.0),
            dim=-1,
        )
        total_observed = torch.sum(observed_expanded, dim=-1)
        result = torch.sum(tree_log, dim=-1) + torch.sum(leaf_log, dim=-1)
        result = torch.where(
            total_observed == 0.0,
            torch.zeros_like(result),
            torch.where(
                impossible,
                torch.full_like(result, float("-inf")),
                result,
            ),
        )
        invalid = torch.any(torch.isnan(result)) | torch.any(
            torch.isinf(result) & (result > 0.0)
        )
        if bool(invalid.item()):
            raise RuntimeError("Component-tree Torch likelihood is invalid.")
        return result

    def _detector_response_mark_concentration_torch(
        self,
        total_line_contributions_xnvsl: object,
        uncollided_line_contributions_xnvsl: object,
    ) -> object:
        """Return Torch finite-corpus concentration per state/view."""
        import torch

        total = torch.as_tensor(total_line_contributions_xnvsl)
        uncollided = torch.minimum(
            torch.as_tensor(
                uncollided_line_contributions_xnvsl,
                device=total.device,
                dtype=total.dtype,
            ),
            total,
        )
        direct_by_line = torch.sum(uncollided, dim=-2)
        scatter_by_line = torch.sum(total - uncollided, dim=-2)
        direct_concentration = torch.as_tensor(
            np.array(self._direct_response_concentration_l, copy=True),
            device=total.device,
            dtype=total.dtype,
        )
        scatter_concentration = torch.as_tensor(
            np.array(self._scatter_response_concentration_l, copy=True),
            device=total.device,
            dtype=total.dtype,
        )
        concentrations = torch.cat(
            (
                torch.broadcast_to(
                    direct_concentration,
                    direct_by_line.shape,
                ),
                torch.broadcast_to(
                    scatter_concentration,
                    scatter_by_line.shape,
                ),
            ),
            dim=-1,
        )
        active = torch.cat(
            (direct_by_line > 0.0, scatter_by_line > 0.0),
            dim=-1,
        )
        return torch.min(
            torch.where(
                active,
                concentrations,
                torch.full_like(concentrations, 1.0e15),
            ),
            dim=-1,
        ).values

    def _base_mark_concentration_torch(
        self,
        total_line_contributions_xnvsl: object,
        uncollided_line_contributions_xnvsl: object | None = None,
    ) -> object:
        """Combine Torch physical and finite-response concentrations."""
        import torch

        if uncollided_line_contributions_xnvsl is None:
            raise ValueError(
                "Detector-response uncertainty requires uncollided line contributions."
            )
        response_concentration = self._detector_response_mark_concentration_torch(
            total_line_contributions_xnvsl,
            uncollided_line_contributions_xnvsl,
        )
        component = self.physical_component_discrepancy
        if component is not None:
            direct_fraction, scatter_fraction = self._source_component_fractions_torch(
                total_line_contributions_xnvsl,
                uncollided_line_contributions_xnvsl,
            )
            reciprocal = torch.square(direct_fraction) / (
                float(component.mark_uncollided_concentration) + 1.0
            ) + torch.square(scatter_fraction) / (
                float(component.mark_scatter_concentration) + 1.0
            )
            physical_concentration = torch.clamp(
                1.0
                / torch.clamp(
                    reciprocal,
                    min=torch.finfo(direct_fraction.dtype).tiny,
                )
                - 1.0,
                min=torch.finfo(direct_fraction.dtype).tiny,
            )
        else:
            low = self.mark_concentration_source
            total = torch.as_tensor(total_line_contributions_xnvsl)
            output_shape = total.shape[:-2]
            high = self.mark_concentration_multi_isotope
            if low is None:
                physical_concentration = torch.full(
                    output_shape,
                    1.0e15,
                    device=total.device,
                    dtype=total.dtype,
                )
            elif high is None or len(self._mark_isotope_names) < 2:
                physical_concentration = torch.full(
                    output_shape,
                    float(low),
                    device=total.device,
                    dtype=total.dtype,
                )
            else:
                mapping = torch.as_tensor(
                    np.array(self._line_to_mark_isotope_li, copy=True),
                    device=total.device,
                    dtype=total.dtype,
                )
                isotope_totals = torch.einsum(
                    "...vsl,li->...vi",
                    total,
                    mapping,
                )
                total_rate = torch.sum(
                    isotope_totals,
                    dim=-1,
                    keepdim=True,
                )
                fractions = torch.where(
                    total_rate > 0.0,
                    isotope_totals
                    / torch.clamp(
                        total_rate,
                        min=torch.finfo(total.dtype).tiny,
                    ),
                    torch.zeros_like(isotope_totals),
                )
                entropy = -torch.sum(
                    torch.xlogy(fractions, fractions),
                    dim=-1,
                ) / float(np.log(float(len(self._mark_isotope_names))))
                entropy = torch.clamp(entropy, 0.0, 1.0)
                physical_concentration = torch.exp(
                    float(np.log(float(low)))
                    + entropy * (float(np.log(float(high))) - float(np.log(float(low))))
                )
        reciprocal = 1.0 / (response_concentration + 1.0) + 1.0 / (
            physical_concentration + 1.0
        )
        return torch.clamp(
            1.0
            / torch.clamp(
                reciprocal,
                min=torch.finfo(response_concentration.dtype).tiny,
            )
            - 1.0,
            min=torch.finfo(response_concentration.dtype).tiny,
        )

    def _detector_response_component_concentrations_numpy(
        self,
        total_line_contributions_xnvsl: NDArray[np.float64],
        uncollided_line_contributions_xnvsl: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return separate conservative direct/scatter Green concentrations."""
        total = np.asarray(total_line_contributions_xnvsl, dtype=np.float64)
        uncollided = np.minimum(
            np.asarray(uncollided_line_contributions_xnvsl, dtype=np.float64),
            total,
        )
        direct_by_line = np.sum(uncollided, axis=-2)
        scatter_by_line = np.sum(total - uncollided, axis=-2)
        direct = np.min(
            np.where(
                direct_by_line > 0.0,
                self._direct_response_concentration_l,
                MARK_EXACT_CONCENTRATION,
            ),
            axis=-1,
        )
        scatter = np.min(
            np.where(
                scatter_by_line > 0.0,
                self._scatter_response_concentration_l,
                MARK_EXACT_CONCENTRATION,
            ),
            axis=-1,
        )
        return np.stack((direct, scatter), axis=-1)

    def _detector_response_component_concentrations_torch(
        self,
        total_line_contributions_xnvsl: object,
        uncollided_line_contributions_xnvsl: object,
    ) -> object:
        """Return Torch direct/scatter finite-Green concentrations."""
        import torch

        total = torch.as_tensor(total_line_contributions_xnvsl)
        uncollided = torch.minimum(
            torch.as_tensor(
                uncollided_line_contributions_xnvsl,
                device=total.device,
                dtype=total.dtype,
            ),
            total,
        )
        direct_by_line = torch.sum(uncollided, dim=-2)
        scatter_by_line = torch.sum(total - uncollided, dim=-2)
        direct_reference, scatter_reference, *_ = (
            self._torch_component_likelihood_constants(total)
        )
        direct = torch.min(
            torch.where(
                direct_by_line > 0.0,
                direct_reference,
                torch.full_like(direct_by_line, MARK_EXACT_CONCENTRATION),
            ),
            dim=-1,
        ).values
        scatter = torch.min(
            torch.where(
                scatter_by_line > 0.0,
                scatter_reference,
                torch.full_like(scatter_by_line, MARK_EXACT_CONCENTRATION),
            ),
            dim=-1,
        ).values
        return torch.stack((direct, scatter), dim=-1)

    def _component_tree_mark_concentrations_numpy(
        self,
        total_line_contributions_xnvsl: NDArray[np.float64],
        uncollided_line_contributions_xnvsl: NDArray[np.float64],
        component_means_xnjvkb: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Propagate component and finite-MC covariance to tree and leaves."""
        component = self.physical_component_discrepancy
        means = np.asarray(component_means_xnjvkb, dtype=np.float64)
        if (
            component is None
            or component.mark_latent_model != "component_dirichlet_tree_hierarchical"
            or means.shape[-2:] != (3, self._energy_axis_keV.size)
            or np.any(~np.isfinite(means))
            or np.any(means < 0.0)
        ):
            raise ValueError("Component-tree NumPy mark inputs are invalid.")
        response = np.expand_dims(
            self._detector_response_component_concentrations_numpy(
                total_line_contributions_xnvsl,
                uncollided_line_contributions_xnvsl,
            ),
            axis=-3,
        )
        direct_group = _combined_dirichlet_concentration_numpy(
            response[..., 0],
            float(component.mark_uncollided_concentration),
        )
        scatter_group = _combined_dirichlet_concentration_numpy(
            response[..., 1],
            float(component.mark_scatter_concentration),
        )
        direct_local = direct_group
        scatter_local = _combined_dirichlet_concentration_numpy(
            response[..., 1],
            float(component.mark_uncollided_concentration),
        )
        group_component = np.stack(
            (
                direct_group,
                scatter_group,
                np.full_like(
                    direct_group,
                    float(component.mark_background_group_concentration),
                ),
            ),
            axis=-1,
        )
        local_component = np.stack(
            (
                direct_local,
                scatter_local,
                np.full_like(
                    direct_local,
                    float(component.mark_background_within_concentration),
                ),
            ),
            axis=-1,
        )
        group_component = np.broadcast_to(group_component, means.shape[:-1])
        local_component = np.broadcast_to(local_component, means.shape[:-1])
        left_mass = np.einsum(
            "...vkb,tb->...vkt",
            means,
            self._mark_tree_left_mask_tb,
            optimize=True,
        )
        right_mass = np.einsum(
            "...vkb,tb->...vkt",
            means,
            self._mark_tree_right_mask_tb,
            optimize=True,
        )
        parent_mass = left_mass + right_mass
        domain_masks = np.where(
            self._mark_tree_domain_t[:, np.newaxis] < 0,
            1.0,
            np.where(
                self._mark_tree_domain_t[:, np.newaxis] == 0,
                self._photopeak_mask_b[np.newaxis, :],
                (~self._photopeak_mask_b)[np.newaxis, :],
            ),
        )
        domain_mass = np.einsum(
            "...vkb,tb->...vkt",
            means,
            domain_masks,
            optimize=True,
        )
        parent_probability = np.divide(
            parent_mass,
            domain_mass,
            out=np.zeros_like(parent_mass),
            where=domain_mass > 0.0,
        )
        component_node_concentration = np.maximum(
            group_component[..., np.newaxis] * parent_probability,
            np.finfo(np.float64).tiny,
        )
        total_parent = np.sum(parent_mass, axis=-2)
        weights = np.divide(
            parent_mass,
            total_parent[..., np.newaxis, :],
            out=np.zeros_like(parent_mass),
            where=total_parent[..., np.newaxis, :] > 0.0,
        )
        component_left_probability = np.divide(
            left_mass,
            parent_mass,
            out=np.zeros_like(left_mass),
            where=parent_mass > 0.0,
        )
        left_probability = np.sum(weights * component_left_probability, axis=-2)
        binary_variance = np.sum(
            np.square(weights)
            * component_left_probability
            * (1.0 - component_left_probability)
            / (component_node_concentration + 1.0),
            axis=-2,
        )
        tree_concentration = np.maximum(
            np.divide(
                left_probability * (1.0 - left_probability),
                binary_variance,
                out=np.full_like(left_probability, MARK_EXACT_CONCENTRATION),
                where=binary_variance > 0.0,
            )
            - 1.0,
            np.finfo(np.float64).tiny,
        )

        leaf_mass = np.einsum(
            "...vkb,hb->...vkh",
            means,
            self._mark_leaf_group_mask_hb,
            optimize=True,
        )
        total_leaf = np.sum(leaf_mass, axis=-2)
        leaf_weights = np.divide(
            leaf_mass,
            total_leaf[..., np.newaxis, :],
            out=np.zeros_like(leaf_mass),
            where=total_leaf[..., np.newaxis, :] > 0.0,
        )
        component_square_mass = np.einsum(
            "...vkb,hb->...vkh",
            np.square(means),
            self._mark_leaf_group_mask_hb,
            optimize=True,
        )
        component_square_probability = np.divide(
            component_square_mass,
            np.square(leaf_mass),
            out=np.zeros_like(component_square_mass),
            where=leaf_mass > 0.0,
        )
        combined = np.sum(means, axis=-2)
        combined_square_mass = np.einsum(
            "...vb,hb->...vh",
            np.square(combined),
            self._mark_leaf_group_mask_hb,
            optimize=True,
        )
        combined_square_probability = np.divide(
            combined_square_mass,
            np.square(total_leaf),
            out=np.zeros_like(combined_square_mass),
            where=total_leaf > 0.0,
        )
        trace_variance = np.sum(
            np.square(leaf_weights)
            * np.maximum(1.0 - component_square_probability, 0.0)
            / (local_component[..., np.newaxis] + 1.0),
            axis=-2,
        )
        leaf_concentration = np.maximum(
            np.divide(
                np.maximum(1.0 - combined_square_probability, 0.0),
                trace_variance,
                out=np.full_like(trace_variance, MARK_EXACT_CONCENTRATION),
                where=trace_variance > 0.0,
            )
            - 1.0,
            np.finfo(np.float64).tiny,
        )
        return tree_concentration, leaf_concentration

    def _component_tree_mark_concentrations_torch(
        self,
        total_line_contributions_xnvsl: object,
        uncollided_line_contributions_xnvsl: object,
        component_means_xnjvkb: object,
    ) -> tuple[object, object]:
        """Return Torch tree and within-leaf concentrations."""
        import torch

        component = self.physical_component_discrepancy
        means = torch.as_tensor(component_means_xnjvkb)
        if (
            component is None
            or component.mark_latent_model != "component_dirichlet_tree_hierarchical"
            or tuple(means.shape[-2:]) != (3, int(self._energy_axis_keV.size))
        ):
            raise ValueError("Component-tree Torch mark inputs are invalid.")
        invalid = torch.any(~torch.isfinite(means)) | torch.any(means < 0.0)
        if bool(invalid.item()):
            raise ValueError("Component-tree Torch mark inputs are invalid.")
        response = self._detector_response_component_concentrations_torch(
            total_line_contributions_xnvsl,
            uncollided_line_contributions_xnvsl,
        ).unsqueeze(-3)
        direct_group = _combined_dirichlet_concentration_torch(
            response[..., 0],
            float(component.mark_uncollided_concentration),
        )
        scatter_group = _combined_dirichlet_concentration_torch(
            response[..., 1],
            float(component.mark_scatter_concentration),
        )
        scatter_local = _combined_dirichlet_concentration_torch(
            response[..., 1],
            float(component.mark_uncollided_concentration),
        )
        group_component = torch.stack(
            (
                direct_group,
                scatter_group,
                torch.full_like(
                    direct_group,
                    float(component.mark_background_group_concentration),
                ),
            ),
            dim=-1,
        ).expand(means.shape[:-1])
        local_component = torch.stack(
            (
                direct_group,
                scatter_local,
                torch.full_like(
                    direct_group,
                    float(component.mark_background_within_concentration),
                ),
            ),
            dim=-1,
        ).expand(means.shape[:-1])
        (
            leaf_masks,
            left_masks,
            right_masks,
            _domains,
            _depths,
            _left_children,
            _right_children,
        ) = self._torch_mark_tree_constants(means)
        (
            _direct_reference,
            _scatter_reference,
            _mark_projection_masks,
            tree_projection_masks,
            _nondegenerate_leaf,
        ) = self._torch_component_likelihood_constants(means)
        tree_count = int(left_masks.shape[0])
        tree_projection = torch.einsum(
            "...vkb,mb->...vkm",
            means,
            tree_projection_masks,
        )
        left_mass, right_mass, domain_mass = torch.split(
            tree_projection,
            (tree_count, tree_count, tree_count),
            dim=-1,
        )
        parent_mass = left_mass + right_mass
        parent_probability = torch.where(
            domain_mass > 0.0,
            parent_mass / torch.clamp(domain_mass, min=torch.finfo(means.dtype).tiny),
            torch.zeros_like(parent_mass),
        )
        component_node_concentration = torch.clamp(
            group_component.unsqueeze(-1) * parent_probability,
            min=torch.finfo(means.dtype).tiny,
        )
        total_parent = torch.sum(parent_mass, dim=-2)
        weights = torch.where(
            total_parent.unsqueeze(-2) > 0.0,
            parent_mass
            / torch.clamp(
                total_parent.unsqueeze(-2),
                min=torch.finfo(means.dtype).tiny,
            ),
            torch.zeros_like(parent_mass),
        )
        component_left_probability = torch.where(
            parent_mass > 0.0,
            left_mass / torch.clamp(parent_mass, min=torch.finfo(means.dtype).tiny),
            torch.zeros_like(left_mass),
        )
        left_probability = torch.sum(weights * component_left_probability, dim=-2)
        binary_variance = torch.sum(
            torch.square(weights)
            * component_left_probability
            * (1.0 - component_left_probability)
            / (component_node_concentration + 1.0),
            dim=-2,
        )
        tree_concentration = torch.clamp(
            torch.where(
                binary_variance > 0.0,
                left_probability
                * (1.0 - left_probability)
                / torch.clamp(binary_variance, min=torch.finfo(means.dtype).tiny)
                - 1.0,
                torch.full_like(left_probability, MARK_EXACT_CONCENTRATION),
            ),
            min=torch.finfo(means.dtype).tiny,
        )
        component_leaf_projection = torch.einsum(
            "r...vkb,hb->r...vkh",
            torch.stack((means, torch.square(means)), dim=0),
            leaf_masks,
        )
        leaf_mass = component_leaf_projection[0]
        total_leaf = torch.sum(leaf_mass, dim=-2)
        leaf_weights = torch.where(
            total_leaf.unsqueeze(-2) > 0.0,
            leaf_mass
            / torch.clamp(
                total_leaf.unsqueeze(-2),
                min=torch.finfo(means.dtype).tiny,
            ),
            torch.zeros_like(leaf_mass),
        )
        component_square_mass = component_leaf_projection[1]
        component_square_probability = torch.where(
            leaf_mass > 0.0,
            component_square_mass
            / torch.clamp(torch.square(leaf_mass), min=torch.finfo(means.dtype).tiny),
            torch.zeros_like(component_square_mass),
        )
        combined = torch.sum(means, dim=-2)
        combined_square_mass = torch.einsum(
            "...vb,hb->...vh",
            torch.square(combined),
            leaf_masks,
        )
        combined_square_probability = torch.where(
            total_leaf > 0.0,
            combined_square_mass
            / torch.clamp(torch.square(total_leaf), min=torch.finfo(means.dtype).tiny),
            torch.zeros_like(combined_square_mass),
        )
        trace_variance = torch.sum(
            torch.square(leaf_weights)
            * torch.clamp(1.0 - component_square_probability, min=0.0)
            / (local_component.unsqueeze(-1) + 1.0),
            dim=-2,
        )
        leaf_concentration = torch.clamp(
            torch.where(
                trace_variance > 0.0,
                torch.clamp(1.0 - combined_square_probability, min=0.0)
                / torch.clamp(trace_variance, min=torch.finfo(means.dtype).tiny)
                - 1.0,
                torch.full_like(trace_variance, MARK_EXACT_CONCENTRATION),
            ),
            min=torch.finfo(means.dtype).tiny,
        )
        return tree_concentration, leaf_concentration

    @staticmethod
    def _source_component_fractions_numpy(
        total_line_contributions_xvsl: NDArray[np.float64],
        uncollided_line_contributions_xvsl: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return uncollided and scatter fractions of source incident rate."""
        total = np.asarray(total_line_contributions_xvsl, dtype=np.float64)
        uncollided = np.minimum(
            np.asarray(
                uncollided_line_contributions_xvsl,
                dtype=np.float64,
            ),
            total,
        )
        total_rate = np.sum(total, axis=(-2, -1))
        direct_rate = np.sum(uncollided, axis=(-2, -1))
        direct_fraction = np.divide(
            direct_rate,
            total_rate,
            out=np.ones_like(total_rate),
            where=total_rate > 0.0,
        )
        return direct_fraction, np.maximum(1.0 - direct_fraction, 0.0)

    @staticmethod
    def _source_component_fractions_torch(
        total_line_contributions_xvsl: object,
        uncollided_line_contributions_xvsl: object,
    ) -> tuple[object, object]:
        """Return Torch uncollided and scatter source-rate fractions."""
        import torch

        total = torch.as_tensor(total_line_contributions_xvsl)
        uncollided = torch.minimum(
            torch.as_tensor(
                uncollided_line_contributions_xvsl,
                device=total.device,
                dtype=total.dtype,
            ),
            total,
        )
        total_rate = torch.sum(total, dim=(-2, -1))
        direct_rate = torch.sum(uncollided, dim=(-2, -1))
        direct_fraction = torch.where(
            total_rate > 0.0,
            direct_rate / torch.clamp(total_rate, min=torch.finfo(total.dtype).tiny),
            torch.ones_like(total_rate),
        )
        return direct_fraction, torch.clamp(1.0 - direct_fraction, min=0.0)

    def _component_count_concentration_numpy(
        self,
        total_line_contributions_xvsl: NDArray[np.float64],
        uncollided_line_contributions_xvsl: NDArray[np.float64],
        transport_features_xvslf: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return physical plus finite-Green count concentration per view."""
        component = self.physical_component_discrepancy
        if component is None:
            raise RuntimeError("Physical-component discrepancy is not configured.")
        total = np.asarray(total_line_contributions_xvsl, dtype=np.float64)
        uncollided = np.minimum(
            np.asarray(
                uncollided_line_contributions_xvsl,
                dtype=np.float64,
            ),
            total,
        )
        features = np.asarray(transport_features_xvslf, dtype=np.float64)
        if features.shape != total.shape + (len(TRANSPORT_FEATURE_ORDER),):
            raise ValueError("Count concentration transport features are invalid.")
        direct = uncollided
        scatter = np.maximum(total - direct, 0.0)
        direct_phase_weights = self._direct_phase_weights_numpy(
            features,
            active_xvsl=direct > 0.0,
        )
        direct_rate = np.einsum(
            "...vsl,...vslc,cl->...v",
            direct,
            direct_phase_weights,
            self._direct_detection_factor_cl,
            optimize=True,
        )
        direct_green_std = np.einsum(
            "...vsl,...vslc,cl->...v",
            direct,
            direct_phase_weights,
            self._direct_detection_factor_std_cl,
            optimize=True,
        )
        if isinstance(
            self.additive_scatter_response,
            PhysicsOnlyNoncollidedTransportResponse,
        ):
            scatter_coefficients = self._detector_cone_scatter_coefficients_numpy(
                scatter,
                features,
            )
            scatter_rate = np.einsum(
                "...vld,dl->...v",
                scatter_coefficients,
                self._detector_cone_scatter_detection_factor_dl,
                optimize=False,
            )
            scatter_green_std = np.einsum(
                "...vld,dl->...v",
                scatter_coefficients,
                self._detector_cone_scatter_detection_factor_std_dl,
                optimize=False,
            )
        else:
            order_weights = self._interaction_order_weights_numpy(features)
            scatter_by_line_order = np.sum(
                scatter[..., np.newaxis] * order_weights,
                axis=-3,
            )
            scatter_rate = np.einsum(
                "...vlo,lo->...v",
                scatter_by_line_order,
                self._scatter_detection_factor_lo,
                optimize=True,
            )
            scatter_green_std = np.einsum(
                "...vlo,lo->...v",
                scatter_by_line_order,
                self._scatter_detection_factor_std_lo,
                optimize=True,
            )
        total_rate = direct_rate + scatter_rate
        denominator = (
            np.square(direct_rate) / float(component.count_uncollided_concentration)
            + np.square(scatter_rate) / float(component.count_scatter_concentration)
            + np.square(direct_green_std + scatter_green_std)
        )
        return np.where(
            total_rate > 0.0,
            np.square(total_rate) / np.maximum(denominator, np.finfo(np.float64).tiny),
            1.0e15,
        )

    def _component_count_concentration_torch(
        self,
        total_line_contributions_xvsl: object,
        uncollided_line_contributions_xvsl: object,
        transport_features_xvslf: object,
    ) -> object:
        """Return Torch physical plus finite-Green count concentration."""
        import torch

        component = self.physical_component_discrepancy
        if component is None:
            raise RuntimeError("Physical-component discrepancy is not configured.")
        total = torch.as_tensor(total_line_contributions_xvsl)
        uncollided = torch.minimum(
            torch.as_tensor(
                uncollided_line_contributions_xvsl,
                device=total.device,
                dtype=total.dtype,
            ),
            total,
        )
        features = torch.as_tensor(
            transport_features_xvslf,
            device=total.device,
            dtype=total.dtype,
        )
        if tuple(features.shape) != tuple(total.shape) + (
            len(TRANSPORT_FEATURE_ORDER),
        ):
            raise ValueError("Torch count concentration features are invalid.")
        direct = uncollided
        scatter = torch.clamp(total - direct, min=0.0)
        direct_phase_weights = self._direct_phase_weights_torch(
            features,
            active_xvsl=direct > 0.0,
        )
        (
            _background,
            _direct_shapes,
            _direct_marginal,
            _scatter_shapes,
            air_mu,
            fe_fraction,
            pb_fraction,
            _obstacle,
            direct_factor,
            direct_factor_std,
            scatter_factor,
            scatter_factor_std,
            detector_cone_distance_nodes,
            _detector_cone_scatter_shapes,
            detector_cone_factor,
            detector_cone_factor_std,
            line_energies,
        ) = self._torch_constants(total)
        direct_rate = torch.einsum(
            "...vsl,...vslc,cl->...v",
            direct,
            direct_phase_weights,
            direct_factor,
        )
        direct_green_std = torch.einsum(
            "...vsl,...vslc,cl->...v",
            direct,
            direct_phase_weights,
            direct_factor_std,
        )
        if isinstance(
            self.additive_scatter_response,
            PhysicsOnlyNoncollidedTransportResponse,
        ):
            scatter_coefficients = self._detector_cone_scatter_coefficients_torch(
                scatter,
                features,
                distance_nodes_d=detector_cone_distance_nodes,
                line_energies_l=line_energies,
                air_mu_l=air_mu,
                fe_fraction_l=fe_fraction,
                pb_fraction_l=pb_fraction,
            )
            scatter_rate = torch.einsum(
                "...vld,dl->...v",
                scatter_coefficients,
                detector_cone_factor,
            )
            scatter_green_std = torch.einsum(
                "...vld,dl->...v",
                scatter_coefficients,
                detector_cone_factor_std,
            )
        else:
            tau = (
                features[..., 0] * fe_fraction
                + features[..., 1] * pb_fraction
                + features[..., 3]
                + features[..., TRANSPORT_DISTANCE_FEATURE_INDEX] * 100.0 * air_mu
            )
            tau = torch.clamp(tau, min=0.0)
            exact_orders = torch.arange(
                1,
                int(self.maximum_scatter_order),
                device=total.device,
                dtype=total.dtype,
            )
            tiny = torch.finfo(total.dtype).tiny
            denominator_positive = -torch.expm1(-tau)
            log_exact = (
                -tau.unsqueeze(-1)
                + torch.log(torch.clamp(tau, min=tiny)).unsqueeze(-1) * exact_orders
                - torch.lgamma(exact_orders + 1.0)
                - torch.log(torch.clamp(denominator_positive, min=tiny)).unsqueeze(-1)
            )
            exact = torch.where(
                tau.unsqueeze(-1) > 0.0,
                torch.exp(log_exact),
                torch.zeros_like(log_exact),
            )
            tail = torch.clamp(1.0 - torch.sum(exact, dim=-1), min=0.0)
            order_weights = torch.cat((exact, tail.unsqueeze(-1)), dim=-1)
            zero_tau = tau <= 0.0
            order_weights = torch.cat(
                (
                    torch.where(
                        zero_tau,
                        torch.ones_like(order_weights[..., 0]),
                        order_weights[..., 0],
                    ).unsqueeze(-1),
                    torch.where(
                        zero_tau.unsqueeze(-1),
                        torch.zeros_like(order_weights[..., 1:]),
                        order_weights[..., 1:],
                    ),
                ),
                dim=-1,
            )
            order_weights = order_weights / torch.clamp(
                torch.sum(order_weights, dim=-1, keepdim=True),
                min=tiny,
            )
            scatter_by_line_order = torch.sum(
                scatter.unsqueeze(-1) * order_weights,
                dim=-3,
            )
            scatter_rate = torch.einsum(
                "...vlo,lo->...v",
                scatter_by_line_order,
                scatter_factor,
            )
            scatter_green_std = torch.einsum(
                "...vlo,lo->...v",
                scatter_by_line_order,
                scatter_factor_std,
            )
        total_rate = direct_rate + scatter_rate
        denominator = (
            torch.square(direct_rate) / float(component.count_uncollided_concentration)
            + torch.square(scatter_rate) / float(component.count_scatter_concentration)
            + torch.square(direct_green_std + scatter_green_std)
        )
        return torch.where(
            total_rate > 0.0,
            torch.square(total_rate)
            / torch.clamp(
                denominator,
                min=torch.finfo(total.dtype).tiny,
            ),
            torch.full_like(total_rate, 1.0e15),
        )

    def _prepare_subset_cross_likelihood_numpy_unchunked(
        self,
        observed_spectra_xqvb: NDArray[np.float64],
        total_line_contributions_xnvsl: NDArray[np.float64],
        uncollided_line_contributions_xnvsl: NDArray[np.float64],
        transport_features_xnvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
    ) -> PreparedNumpySubsetCrossLikelihood:
        """Prepare exact view-resolved NumPy likelihood sufficient terms."""
        observed = np.asarray(observed_spectra_xqvb, dtype=np.float64)
        component_discrepancy = self.physical_component_discrepancy
        component_tree_marks = bool(
            component_discrepancy is not None
            and component_discrepancy.mark_latent_model
            == "component_dirichlet_tree_hierarchical"
        )
        if component_tree_marks:
            direct_mean, scatter_mean, background_mean = self._pre_dead_time_mean_numpy(
                total_line_contributions_xnvsl,
                uncollided_line_contributions_xnvsl,
                transport_features_xnvslf,
                live_times_s_v,
                return_physical_components=True,
            )
            source_mean = direct_mean + scatter_mean
        else:
            source_mean, background_mean = self._pre_dead_time_mean_numpy(
                total_line_contributions_xnvsl,
                uncollided_line_contributions_xnvsl,
                transport_features_xnvslf,
                live_times_s_v,
                return_components=True,
            )
        if (
            observed.ndim < 3
            or observed.shape[:-3] != source_mean.shape[:-3]
            or observed.shape[-2:] != source_mean.shape[-2:]
            or np.any(~np.isfinite(observed))
            or np.any(observed < 0.0)
            or np.any(observed != np.floor(observed))
        ):
            raise ValueError(
                "Cross spectra must be exact nonnegative counts shaped "
                "...sample/view/bin with common model leading axes."
            )
        node_shape = (1,) * (source_mean.ndim - 3) + (
            1,
            int(self._rate_scale_nodes_j.size),
            1,
            1,
        )
        node_source = source_mean[
            ..., :, np.newaxis, :, :
        ] * self._rate_scale_nodes_j.reshape(node_shape)
        pre_mean = background_mean[..., :, np.newaxis, :, :] + node_source
        observed_total = np.sum(observed, axis=-1)
        pre_total = np.sum(pre_mean, axis=-1)
        live = np.asarray(live_times_s_v, dtype=np.float64)
        shared_gamma_concentration: float | None = None
        shared_observed_counts = None
        shared_expected_counts = None
        if (
            self.count_discrepancy_concentration is None
            and component_discrepancy is None
        ):
            count_log = nonparalyzable_count_log_probability_numpy(
                observed_total[..., :, np.newaxis, np.newaxis, :],
                pre_total[..., np.newaxis, :, :, :] / live,
                live,
                dead_time_tau_s=float(self.dead_time_tau_s),
            )
        else:
            dead_time_scale = 1.0 + pre_total / live * float(self.dead_time_tau_s)
            recorded_total_mean = pre_total / dead_time_scale
            component_count_concentration = None
            if component_discrepancy is not None:
                component_count_concentration = (
                    self._component_count_concentration_numpy(
                        total_line_contributions_xnvsl,
                        uncollided_line_contributions_xnvsl,
                        transport_features_xnvslf,
                    )[..., np.newaxis, :]
                )
            if (
                component_discrepancy is None
                and self.count_discrepancy_scope != "view_independent"
            ):
                shared_gamma_concentration = float(self.count_discrepancy_concentration)
                shared_observed_counts = observed_total
                shared_expected_counts = recorded_total_mean
                counts = observed_total[..., :, np.newaxis, np.newaxis, :]
                means = recorded_total_mean[..., np.newaxis, :, :, :]
                count_log = special.xlogy(counts, means) - canonical_log_gamma_numpy(
                    counts + 1.0
                )
            else:
                count_log = view_independent_gamma_poisson_count_log_increments_numpy(
                    observed_total,
                    recorded_total_mean,
                    concentration=(
                        component_count_concentration
                        if component_count_concentration is not None
                        else float(self.count_discrepancy_concentration)
                    ),
                )
                if component_discrepancy is not None:
                    source_active = np.sum(node_source, axis=-1) > 0.0
                    background_only_log = nonparalyzable_count_log_probability_numpy(
                        observed_total[..., :, np.newaxis, np.newaxis, :],
                        pre_total[..., np.newaxis, :, :, :] / live,
                        live,
                        dead_time_tau_s=float(self.dead_time_tau_s),
                    )
                    count_log = np.where(
                        source_active[..., np.newaxis, :, :, :],
                        count_log,
                        background_only_log,
                    )
        probabilities = np.divide(
            pre_mean,
            pre_total[..., np.newaxis],
            out=np.zeros_like(pre_mean),
            where=pre_total[..., np.newaxis] > 0.0,
        )
        log_probabilities = np.log(np.maximum(probabilities, np.finfo(np.float64).tiny))
        multinomial_log = (
            canonical_log_gamma_numpy(observed_total + 1.0)[
                ..., :, np.newaxis, np.newaxis, :
            ]
            - np.sum(
                canonical_log_gamma_numpy(observed + 1.0),
                axis=-1,
            )[..., :, np.newaxis, np.newaxis, :]
            + np.einsum(
                "...qvb,...njvb->...qnjv",
                observed,
                log_probabilities,
                optimize=True,
            )
        )
        impossible_marks = (
            np.einsum(
                "...qvb,...njvb->...qnjv",
                observed,
                probabilities <= 0.0,
                optimize=True,
            )
            > 0.0
        )
        multinomial_log = np.where(
            impossible_marks,
            -np.inf,
            multinomial_log,
        )
        mark_log = multinomial_log
        if component_tree_marks:
            node_direct = direct_mean[
                ..., :, np.newaxis, :, :
            ] * self._rate_scale_nodes_j.reshape(node_shape)
            node_scatter = scatter_mean[
                ..., :, np.newaxis, :, :
            ] * self._rate_scale_nodes_j.reshape(node_shape)
            component_means = np.stack(
                (
                    node_direct,
                    node_scatter,
                    np.broadcast_to(
                        background_mean[..., :, np.newaxis, :, :],
                        node_direct.shape,
                    ),
                ),
                axis=-2,
            )
            tree_concentration, leaf_concentration = (
                self._component_tree_mark_concentrations_numpy(
                    total_line_contributions_xnvsl,
                    uncollided_line_contributions_xnvsl,
                    component_means,
                )
            )
            mark_log = self._component_tree_mark_log_numpy(
                observed,
                probabilities,
                tree_concentration,
                leaf_concentration,
            )
        if not component_tree_marks:
            base_concentration = self._base_mark_concentration_numpy(
                total_line_contributions_xnvsl,
                uncollided_line_contributions_xnvsl,
            )
            source_total = np.sum(node_source, axis=-1)
            source_fraction = np.divide(
                source_total,
                pre_total,
                out=np.zeros_like(source_total),
                where=pre_total > 0.0,
            )
            base_concentration = self._base_mark_concentration_numpy(
                total_line_contributions_xnvsl,
                uncollided_line_contributions_xnvsl,
            )
            concentration = base_concentration[..., np.newaxis, :] / np.maximum(
                np.square(source_fraction),
                1.0e-12,
            )
            alpha = probabilities * concentration[..., np.newaxis]
            dirichlet_sum = np.zeros_like(multinomial_log)
            for start in range(
                0,
                observed.shape[-1],
                CROSS_LIKELIHOOD_BIN_CHUNK_SIZE,
            ):
                stop = min(
                    start + CROSS_LIKELIHOOD_BIN_CHUNK_SIZE,
                    observed.shape[-1],
                )
                observed_chunk = observed[..., start:stop]
                alpha_chunk = alpha[..., start:stop]
                expanded_alpha = alpha_chunk[..., np.newaxis, :, :, :, :]
                expanded_observed = observed_chunk[..., :, np.newaxis, np.newaxis, :, :]
                active_increment = (expanded_alpha > 0.0) & (expanded_observed > 0.0)
                safe_alpha = np.where(
                    active_increment,
                    expanded_alpha,
                    1.0,
                )
                safe_observed = np.where(
                    active_increment,
                    expanded_observed,
                    1.0,
                )
                dirichlet_sum += np.sum(
                    np.where(
                        active_increment,
                        np.log(safe_alpha)
                        + special.gammaln(safe_alpha + safe_observed)
                        - special.gammaln(safe_alpha + 1.0),
                        0.0,
                    ),
                    axis=-1,
                )
            dirichlet_log = (
                special.gammaln(observed_total + 1.0)[..., :, np.newaxis, np.newaxis, :]
                - np.sum(
                    special.gammaln(observed + 1.0),
                    axis=-1,
                )[..., :, np.newaxis, np.newaxis, :]
                + special.gammaln(concentration)[..., np.newaxis, :, :, :]
                - special.gammaln(
                    concentration[..., np.newaxis, :, :, :]
                    + observed_total[..., :, np.newaxis, np.newaxis, :]
                )
                + dirichlet_sum
            )
            dirichlet_log = np.where(
                impossible_marks,
                -np.inf,
                dirichlet_log,
            )
            mark_log = np.where(
                source_fraction[..., np.newaxis, :, :, :] > 0.0,
                dirichlet_log,
                multinomial_log,
            )
        zero_mark_total = observed_total[..., :, np.newaxis, np.newaxis, :]
        zero_mark_total = zero_mark_total == 0.0
        mark_log = np.where(zero_mark_total, 0.0, mark_log)
        view_node_log = (count_log + mark_log)[..., np.newaxis, :]
        latent_log_weights = np.log(self._rate_scale_weights_j)[:, np.newaxis]
        leading_shape = tuple(int(value) for value in observed.shape[:-3])
        action_count = int(np.prod(leading_shape, dtype=np.int64))
        if not leading_shape:
            action_count = 1
        sample_count = int(observed.shape[-3])
        state_count = int(total_line_contributions_xnvsl.shape[-4])
        return PreparedNumpySubsetCrossLikelihood(
            leading_shape=leading_shape,
            view_node_log_aqnjrv=np.ascontiguousarray(
                view_node_log.reshape(
                    (
                        action_count,
                        sample_count,
                        state_count,
                    )
                    + tuple(view_node_log.shape[-3:])
                ),
                dtype=np.float64,
            ),
            latent_log_weights_jr=np.ascontiguousarray(
                latent_log_weights,
                dtype=np.float64,
            ),
            shared_gamma_concentration=shared_gamma_concentration,
            shared_observed_counts_aqv=(
                None
                if shared_observed_counts is None
                else np.ascontiguousarray(
                    shared_observed_counts.reshape(
                        (action_count, sample_count, int(observed.shape[-2]))
                    ),
                    dtype=np.float64,
                )
            ),
            shared_expected_counts_anjv=(
                None
                if shared_expected_counts is None
                else np.ascontiguousarray(
                    shared_expected_counts.reshape(
                        (
                            action_count,
                            state_count,
                            int(self._rate_scale_nodes_j.size),
                            int(observed.shape[-2]),
                        )
                    ),
                    dtype=np.float64,
                )
            ),
        )

    def _cross_log_likelihood_numpy_unchunked(
        self,
        observed_spectra_xqvb: NDArray[np.float64],
        total_line_contributions_xnvsl: NDArray[np.float64],
        uncollided_line_contributions_xnvsl: NDArray[np.float64],
        transport_features_xnvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return exact likelihoods by reducing prepared view terms."""
        prepared = self._prepare_subset_cross_likelihood_numpy_unchunked(
            observed_spectra_xqvb,
            total_line_contributions_xnvsl,
            uncollided_line_contributions_xnvsl,
            transport_features_xnvslf,
            live_times_s_v,
        )
        return prepared.full()

    def _prepare_subset_cross_likelihood_torch_unchunked(
        self,
        observed_spectra_xqvb: object,
        total_line_contributions_xnvsl: object,
        uncollided_line_contributions_xnvsl: object,
        transport_features_xnvslf: object,
        live_times_s_v: object,
        *,
        prepared_observation: PreparedTorchCrossObservation | None = None,
    ) -> PreparedTorchSubsetCrossLikelihood:
        """Prepare exact device-resident Torch view sufficient terms."""
        import torch

        total = torch.as_tensor(total_line_contributions_xnvsl)
        prepared = prepared_observation
        if prepared is None:
            prepared = self.prepare_cross_observation_torch(
                observed_spectra_xqvb,
                reference=total,
            )
        observed = torch.as_tensor(
            prepared.restored(prepared.observed_asvb),
            device=total.device,
            dtype=total.dtype,
        )
        component_discrepancy = self.physical_component_discrepancy
        component_tree_marks = bool(
            component_discrepancy is not None
            and component_discrepancy.mark_latent_model
            == "component_dirichlet_tree_hierarchical"
        )
        component_count_concentration = None
        if component_tree_marks:
            (
                direct_mean,
                scatter_mean,
                background_mean,
                component_count_concentration,
            ) = self._pre_dead_time_mean_torch(
                total,
                uncollided_line_contributions_xnvsl,
                transport_features_xnvslf,
                live_times_s_v,
                return_physical_components=True,
                return_component_count_concentration=True,
            )
            source_mean = direct_mean + scatter_mean
        else:
            source_mean, background_mean = self._pre_dead_time_mean_torch(
                total,
                uncollided_line_contributions_xnvsl,
                transport_features_xnvslf,
                live_times_s_v,
                return_components=True,
            )
        if (
            observed.ndim < 3
            or tuple(observed.shape[:-3]) != tuple(source_mean.shape[:-3])
            or tuple(observed.shape[-2:]) != tuple(source_mean.shape[-2:])
        ):
            raise ValueError("Torch cross-spectrum observations are invalid.")
        nodes, node_weights, _, _ = self._torch_likelihood_constants(total)
        node_shape = (1,) * (source_mean.ndim - 3) + (1, int(nodes.numel()), 1, 1)
        node_source = source_mean.unsqueeze(-3) * nodes.reshape(node_shape)
        pre_mean = background_mean.unsqueeze(-3) + node_source
        observed_total = prepared.restored(prepared.observed_total_asv)
        pre_total = torch.sum(pre_mean, dim=-1)
        live = torch.as_tensor(
            live_times_s_v,
            device=total.device,
            dtype=total.dtype,
        )
        shared_gamma_concentration: float | None = None
        shared_observed_counts = None
        shared_expected_counts = None
        if (
            self.count_discrepancy_concentration is None
            and component_discrepancy is None
        ):
            count_log = nonparalyzable_count_log_probability_torch(
                observed_total.unsqueeze(-2).unsqueeze(-2),
                pre_total.unsqueeze(-4) / live,
                live,
                dead_time_tau_s=float(self.dead_time_tau_s),
                validate_inputs=False,
            )
        else:
            dead_time_scale = 1.0 + pre_total / live * float(self.dead_time_tau_s)
            recorded_total_mean = pre_total / dead_time_scale
            if (
                component_discrepancy is not None
                and component_count_concentration is None
            ):
                component_count_concentration = (
                    self._component_count_concentration_torch(
                        total,
                        uncollided_line_contributions_xnvsl,
                        transport_features_xnvslf,
                    )
                )
            if component_count_concentration is not None:
                component_count_concentration = (
                    component_count_concentration.unsqueeze(-2)
                )
            if (
                component_discrepancy is None
                and self.count_discrepancy_scope != "view_independent"
            ):
                shared_gamma_concentration = float(self.count_discrepancy_concentration)
                shared_observed_counts = observed_total
                shared_expected_counts = recorded_total_mean
                counts = observed_total.unsqueeze(-2).unsqueeze(-2)
                means = recorded_total_mean.unsqueeze(-4)
                count_log = torch.xlogy(counts, means) - canonical_log_gamma_torch(
                    counts + 1.0
                )
            else:
                count_log = view_independent_gamma_poisson_count_log_increments_torch(
                    observed_total,
                    recorded_total_mean,
                    concentration=(
                        component_count_concentration
                        if component_count_concentration is not None
                        else float(self.count_discrepancy_concentration)
                    ),
                    validate_inputs=False,
                )
                if component_discrepancy is not None:
                    source_active = torch.sum(node_source, dim=-1) > 0.0
                    all_source_active = bool(torch.all(source_active).item())
                    if not all_source_active:
                        background_only_log = (
                            nonparalyzable_count_log_probability_torch(
                                observed_total.unsqueeze(-2).unsqueeze(-2),
                                pre_total.unsqueeze(-4) / live,
                                live,
                                dead_time_tau_s=float(self.dead_time_tau_s),
                                validate_inputs=False,
                            )
                        )
                        count_log = torch.where(
                            source_active.unsqueeze(-4),
                            count_log,
                            background_only_log,
                        )
        tiny = torch.finfo(total.dtype).tiny
        probabilities = torch.where(
            pre_total.unsqueeze(-1) > 0.0,
            pre_mean / torch.clamp(pre_total.unsqueeze(-1), min=tiny),
            torch.zeros_like(pre_mean),
        )
        if component_tree_marks:
            node_direct = direct_mean.unsqueeze(-3) * nodes.reshape(node_shape)
            node_scatter = scatter_mean.unsqueeze(-3) * nodes.reshape(node_shape)
            component_means = torch.stack(
                (
                    node_direct,
                    node_scatter,
                    background_mean.unsqueeze(-3).expand(node_direct.shape),
                ),
                dim=-2,
            )
            tree_concentration, leaf_concentration = (
                self._component_tree_mark_concentrations_torch(
                    total,
                    uncollided_line_contributions_xnvsl,
                    component_means,
                )
            )
            mark_log = self._component_tree_mark_log_torch(
                observed,
                probabilities,
                tree_concentration,
                leaf_concentration,
                prepared_observation=prepared,
            )
        else:
            log_probabilities = torch.log(torch.clamp(probabilities, min=tiny))
            multinomial_log = prepared.restored(
                prepared.multinomial_constant_asv
            ).unsqueeze(-2).unsqueeze(-2) + torch.einsum(
                "...qvb,...njvb->...qnjv",
                observed,
                log_probabilities,
            )
            impossible = (
                torch.einsum(
                    "...qvb,...njvb->...qnjv",
                    observed,
                    (probabilities <= 0.0).to(dtype=observed.dtype),
                )
                > 0.0
            )
            multinomial_log = torch.where(
                impossible,
                -torch.inf,
                multinomial_log,
            )
            base_concentration = self._base_mark_concentration_torch(
                total,
                uncollided_line_contributions_xnvsl,
            )
            source_total = torch.sum(node_source, dim=-1)
            source_fraction = torch.where(
                pre_total > 0.0,
                source_total / torch.clamp(pre_total, min=tiny),
                torch.zeros_like(source_total),
            )
            base_concentration = self._base_mark_concentration_torch(
                total,
                uncollided_line_contributions_xnvsl,
            )
            concentration = base_concentration.unsqueeze(-2) / torch.clamp(
                torch.square(source_fraction),
                min=1.0e-12,
            )
            alpha = probabilities * concentration.unsqueeze(-1)
            dirichlet_sum = torch.zeros_like(multinomial_log)
            for start in range(
                0,
                int(observed.shape[-1]),
                CROSS_LIKELIHOOD_BIN_CHUNK_SIZE,
            ):
                stop = min(
                    start + CROSS_LIKELIHOOD_BIN_CHUNK_SIZE,
                    int(observed.shape[-1]),
                )
                observed_chunk = observed[..., start:stop]
                alpha_chunk = alpha[..., start:stop]
                expanded_alpha = alpha_chunk.unsqueeze(-5)
                expanded_observed = observed_chunk.unsqueeze(-3).unsqueeze(-3)
                active_increment = (expanded_alpha > 0.0) & (expanded_observed > 0.0)
                safe_alpha = torch.where(
                    active_increment,
                    expanded_alpha,
                    torch.ones_like(expanded_alpha),
                )
                safe_observed = torch.where(
                    active_increment,
                    expanded_observed,
                    torch.ones_like(expanded_observed),
                )
                dirichlet_sum = dirichlet_sum + torch.sum(
                    torch.where(
                        active_increment,
                        torch.log(safe_alpha)
                        + torch.lgamma(safe_alpha + safe_observed)
                        - torch.lgamma(safe_alpha + 1.0),
                        torch.zeros_like(safe_alpha),
                    ),
                    dim=-1,
                )
            dirichlet_log = (
                prepared.restored(prepared.multinomial_constant_asv)
                .unsqueeze(-2)
                .unsqueeze(-2)
                + torch.lgamma(concentration).unsqueeze(-4)
                - torch.lgamma(
                    concentration.unsqueeze(-4)
                    + observed_total.unsqueeze(-2).unsqueeze(-2)
                )
                + dirichlet_sum
            )
            dirichlet_log = torch.where(
                impossible,
                -torch.inf,
                dirichlet_log,
            )
            mark_log = torch.where(
                source_fraction.unsqueeze(-4) > 0.0,
                dirichlet_log,
                multinomial_log,
            )
        zero_mark_total = observed_total.unsqueeze(-2).unsqueeze(-2)
        zero_mark_total = zero_mark_total == 0.0
        mark_log = torch.where(
            zero_mark_total,
            torch.zeros_like(mark_log),
            mark_log,
        )
        view_node_log = (count_log + mark_log).unsqueeze(-2)
        latent_log_weights = torch.log(node_weights).unsqueeze(-1)
        leading_shape = tuple(int(value) for value in observed.shape[:-3])
        action_count = int(np.prod(leading_shape, dtype=np.int64))
        if not leading_shape:
            action_count = 1
        sample_count = int(observed.shape[-3])
        state_count = int(total.shape[-4])
        return PreparedTorchSubsetCrossLikelihood(
            leading_shape=leading_shape,
            view_node_log_aqnjrv=view_node_log.reshape(
                (action_count, sample_count, state_count)
                + tuple(view_node_log.shape[-3:])
            ).contiguous(),
            latent_log_weights_jr=latent_log_weights.contiguous(),
            shared_gamma_concentration=shared_gamma_concentration,
            shared_observed_counts_aqv=(
                None
                if shared_observed_counts is None
                else shared_observed_counts.reshape(
                    (action_count, sample_count, int(observed.shape[-2]))
                ).contiguous()
            ),
            shared_expected_counts_anjv=(
                None
                if shared_expected_counts is None
                else shared_expected_counts.reshape(
                    (
                        action_count,
                        state_count,
                        int(node_weights.numel()),
                        int(observed.shape[-2]),
                    )
                ).contiguous()
            ),
        )

    def _cross_log_likelihood_torch_unchunked(
        self,
        observed_spectra_xqvb: object,
        total_line_contributions_xnvsl: object,
        uncollided_line_contributions_xnvsl: object,
        transport_features_xnvslf: object,
        live_times_s_v: object,
        *,
        prepared_observation: PreparedTorchCrossObservation | None = None,
    ) -> object:
        """Return exact Torch likelihoods by reducing prepared view terms."""
        prepared = self._prepare_subset_cross_likelihood_torch_unchunked(
            observed_spectra_xqvb,
            total_line_contributions_xnvsl,
            uncollided_line_contributions_xnvsl,
            transport_features_xnvslf,
            live_times_s_v,
            prepared_observation=prepared_observation,
        )
        return prepared.full()

    @staticmethod
    def _resolved_cross_chunk_size(
        value: int | None,
        *,
        total: int,
        default: int,
        label: str,
    ) -> int:
        """Return a positive cross-likelihood chunk size bounded by its axis."""
        if int(total) <= 0:
            raise ValueError(f"{label} axis must be nonempty.")
        if value is None:
            resolved = int(default)
        else:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{label} chunk size must be a positive integer.")
            resolved = int(value)
        if resolved <= 0:
            raise ValueError(f"{label} chunk size must be a positive integer.")
        return min(resolved, int(total))

    def estimate_cross_likelihood_working_set_bytes(
        self,
        *,
        num_actions: int,
        num_samples: int,
        num_particles: int,
        num_isotopes: int,
        num_views: int,
        action_chunk_size: int | None = None,
        sample_chunk_size: int | None = None,
        state_chunk_size: int | None = None,
        dtype_bytes: int = 8,
    ) -> int:
        """Conservatively estimate one chunk of exact likelihood workspace."""
        counts = {
            "num_actions": num_actions,
            "num_samples": num_samples,
            "num_particles": num_particles,
            "num_isotopes": num_isotopes,
            "num_views": num_views,
            "dtype_bytes": dtype_bytes,
        }
        for label, raw_value in counts.items():
            if (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, (int, np.integer))
                or int(raw_value) <= 0
            ):
                raise ValueError(f"{label} must be a positive integer.")
        action_chunk = self._resolved_cross_chunk_size(
            action_chunk_size,
            total=int(num_actions),
            default=CROSS_LIKELIHOOD_ACTION_CHUNK_SIZE,
            label="action",
        )
        sample_chunk = self._resolved_cross_chunk_size(
            sample_chunk_size,
            total=int(num_samples),
            default=CROSS_LIKELIHOOD_SAMPLE_CHUNK_SIZE,
            label="sample",
        )
        state_chunk = self._resolved_cross_chunk_size(
            state_chunk_size,
            total=int(num_particles),
            default=CROSS_LIKELIHOOD_STATE_CHUNK_SIZE,
            label="state",
        )
        bin_count = int(np.asarray(self.energy_axis_keV).size)
        bin_chunk = min(CROSS_LIKELIHOOD_BIN_CHUNK_SIZE, bin_count)
        node_count = int(self._rate_scale_nodes_j.size)
        line_count = len(self._line_identity)
        expanded = (
            action_chunk
            * sample_chunk
            * state_chunk
            * node_count
            * int(num_views)
            * bin_chunk
        )
        # lgamma/where evaluation can hold several simultaneous dense
        # temporaries.  Eight copies is deliberately conservative for both
        # NumPy and Torch allocator behaviour.
        dirichlet_temporaries = 8 * expanded
        marked_state = action_chunk * state_chunk * int(num_views) * bin_count
        observed = action_chunk * sample_chunk * int(num_views) * bin_count
        transport_inputs = (
            action_chunk
            * state_chunk
            * int(num_views)
            * int(num_isotopes)
            * line_count
            * (2 + len(TRANSPORT_FEATURE_ORDER))
        )
        output_and_reductions = (
            6 * action_chunk * sample_chunk * state_chunk * node_count * int(num_views)
        )
        total_elements = (
            dirichlet_temporaries
            + 10 * marked_state
            + 3 * observed
            + transport_inputs
            + output_and_reductions
        )
        return int(total_elements * int(dtype_bytes))

    def estimate_subset_cross_likelihood_working_set_bytes(
        self,
        *,
        num_actions: int,
        num_samples: int,
        num_particles: int,
        num_source_slots: int,
        num_views: int,
        num_candidates: int,
        subset_size: int,
        action_chunk_size: int | None = None,
        sample_chunk_size: int | None = None,
        state_chunk_size: int | None = None,
        view_chunk_size: int | None = None,
        dtype_bytes: int = 8,
    ) -> int:
        """Estimate resident cache plus peak arbitrary-subset workspace.

        The estimate includes source-resolved inputs, predictive observations,
        the reusable view/node cache, one bounded cache-preparation slab, and
        one fully batched candidate contraction. It is intentionally suitable
        for planner scheduling and never changes nuisance integration or the
        set of evaluated candidates.
        """
        counts = {
            "num_actions": num_actions,
            "num_samples": num_samples,
            "num_particles": num_particles,
            "num_source_slots": num_source_slots,
            "num_views": num_views,
            "num_candidates": num_candidates,
            "subset_size": subset_size,
            "dtype_bytes": dtype_bytes,
        }
        for label, raw_value in counts.items():
            if (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, (int, np.integer))
                or int(raw_value) <= 0
            ):
                raise ValueError(f"{label} must be a positive integer.")
        if int(subset_size) > int(num_views):
            raise ValueError("subset_size cannot exceed num_views.")
        action_chunk = self._resolved_cross_chunk_size(
            action_chunk_size,
            total=int(num_actions),
            default=CROSS_LIKELIHOOD_ACTION_CHUNK_SIZE,
            label="action",
        )
        sample_chunk = self._resolved_cross_chunk_size(
            sample_chunk_size,
            total=int(num_samples),
            default=CROSS_LIKELIHOOD_SAMPLE_CHUNK_SIZE,
            label="sample",
        )
        state_chunk = self._resolved_cross_chunk_size(
            state_chunk_size,
            total=int(num_particles),
            default=CROSS_LIKELIHOOD_STATE_CHUNK_SIZE,
            label="state",
        )
        view_chunk = self._resolved_cross_chunk_size(
            view_chunk_size,
            total=int(num_views),
            default=SUBSET_LIKELIHOOD_VIEW_CHUNK_SIZE,
            label="view",
        )
        action_count = int(num_actions)
        sample_count = int(num_samples)
        particle_count = int(num_particles)
        view_count = int(num_views)
        candidate_count = int(num_candidates)
        node_count = int(self._rate_scale_nodes_j.size)
        component = self.physical_component_discrepancy
        mark_node_count = 1
        line_count = len(self._line_identity)
        bin_count = int(np.asarray(self.energy_axis_keV).size)
        cache_elements = (
            action_count
            * sample_count
            * particle_count
            * node_count
            * mark_node_count
            * view_count
        )
        shared_gamma = bool(
            self.count_discrepancy_concentration is not None
            and component is None
            and self.count_discrepancy_scope != "view_independent"
        )
        if shared_gamma:
            cache_elements += (
                action_count * sample_count * view_count
                + action_count * particle_count * node_count * view_count
            )
        transport_elements = (
            action_count
            * particle_count
            * view_count
            * int(num_source_slots)
            * line_count
            * (2 + len(TRANSPORT_FEATURE_ORDER))
        )
        predictive_transport_elements = (
            action_count
            * sample_count
            * view_count
            * int(num_source_slots)
            * line_count
            * (2 + len(TRANSPORT_FEATURE_ORDER))
        )
        # Predictive integer spectra coexist briefly with their float64 cache
        # input and prepared observation statistics.
        predictive_elements = 3 * action_count * sample_count * view_count * bin_count
        preparation_workspace = self.estimate_cross_likelihood_working_set_bytes(
            num_actions=action_count,
            num_samples=sample_count,
            num_particles=particle_count,
            num_isotopes=int(num_source_slots),
            num_views=view_chunk,
            action_chunk_size=action_chunk,
            sample_chunk_size=sample_chunk,
            state_chunk_size=state_chunk,
            dtype_bytes=int(dtype_bytes),
        )
        candidate_node_elements = (
            action_count
            * candidate_count
            * sample_count
            * particle_count
            * node_count
            * mark_node_count
        )
        selection_elements = action_count * candidate_count * view_count
        eig_reduction_elements = (
            8 * action_count * candidate_count * sample_count * particle_count
        )
        gamma_elements = (
            4
            * action_count
            * candidate_count
            * sample_count
            * particle_count
            * node_count
            if shared_gamma
            else 0
        )
        candidate_workspace = int(dtype_bytes) * (
            2 * candidate_node_elements
            + selection_elements
            + eig_reduction_elements
            + gamma_elements
        )
        resident_bytes = int(dtype_bytes) * (
            cache_elements
            + transport_elements
            + predictive_transport_elements
            + predictive_elements
        )
        return int(
            resident_bytes + max(int(preparation_workspace), int(candidate_workspace))
        )

    @staticmethod
    def _torch_cross_state_autotune_candidates(
        *,
        state_count: int,
        memory_limited_maximum: int,
    ) -> tuple[int, ...]:
        """Return real-work CUDA state slabs used for one-time tuning."""
        maximum = min(
            int(state_count),
            int(memory_limited_maximum),
            CROSS_LIKELIHOOD_STATE_AUTOTUNE_MAX_CHUNK_SIZE,
        )
        candidates: list[int] = []
        consumed = 0
        candidate = CROSS_LIKELIHOOD_STATE_CHUNK_SIZE
        while candidate <= maximum and consumed + candidate <= int(state_count):
            candidates.append(candidate)
            consumed += candidate
            candidate *= 2
        return tuple(candidates)

    def _torch_cross_state_autotune_key(
        self,
        *,
        total: object,
        action_count: int,
        sample_count: int,
        action_chunk: int,
        sample_chunk: int,
    ) -> tuple[object, ...]:
        """Return a reusable exact-likelihood CUDA workload key."""
        tensor = total
        component = self.physical_component_discrepancy
        return (
            str(tensor.device),
            str(tensor.dtype),
            int(action_count),
            int(sample_count),
            int(action_chunk),
            int(sample_chunk),
            (
                1024
                if int(tensor.shape[-4]) >= 1792
                else 512
                if int(tensor.shape[-4]) >= 768
                else 256
            ),
            int(tensor.shape[-3]),
            int(tensor.shape[-2]),
            int(tensor.shape[-1]),
            None if component is None else str(component.mark_latent_model),
        )

    def prepare_subset_cross_likelihood_numpy(
        self,
        observed_spectra_xqvb: NDArray[np.float64],
        total_line_contributions_xnvsl: NDArray[np.float64],
        uncollided_line_contributions_xnvsl: NDArray[np.float64],
        transport_features_xnvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
        *,
        action_chunk_size: int | None = None,
        sample_chunk_size: int | None = None,
        state_chunk_size: int | None = None,
        view_chunk_size: int | None = None,
    ) -> PreparedNumpySubsetCrossLikelihood:
        """Prepare reusable exact NumPy likelihood terms for view subsets.

        Preparation is chunked over actions, samples, states, and views.
        Subsequent candidates use one bounded selection-matrix contraction
        without recomputing full-spectrum response marking.
        """
        observed = np.asarray(observed_spectra_xqvb, dtype=np.float64)
        total = np.asarray(total_line_contributions_xnvsl, dtype=np.float64)
        uncollided = np.asarray(
            uncollided_line_contributions_xnvsl,
            dtype=np.float64,
        )
        features = np.asarray(
            transport_features_xnvslf,
            dtype=np.float64,
        )
        if observed.ndim < 3 or total.ndim < 4:
            raise ValueError("Subset likelihood inputs have too few dimensions.")
        leading_shape = tuple(int(value) for value in observed.shape[:-3])
        if tuple(int(value) for value in total.shape[:-4]) != leading_shape:
            raise ValueError("Subset spectra and states require identical action axes.")
        if (
            uncollided.shape != total.shape
            or features.shape != total.shape + (len(TRANSPORT_FEATURE_ORDER),)
            or int(observed.shape[-2]) != int(total.shape[-3])
            or int(observed.shape[-1]) != int(np.asarray(self.energy_axis_keV).size)
        ):
            raise ValueError("Subset likelihood tensor shapes are inconsistent.")
        action_count = int(np.prod(leading_shape, dtype=np.int64))
        if not leading_shape:
            action_count = 1
        sample_count = int(observed.shape[-3])
        state_count = int(total.shape[-4])
        view_count = int(total.shape[-3])
        action_chunk = self._resolved_cross_chunk_size(
            action_chunk_size,
            total=action_count,
            default=CROSS_LIKELIHOOD_ACTION_CHUNK_SIZE,
            label="action",
        )
        sample_chunk = self._resolved_cross_chunk_size(
            sample_chunk_size,
            total=sample_count,
            default=CROSS_LIKELIHOOD_SAMPLE_CHUNK_SIZE,
            label="sample",
        )
        state_chunk = self._resolved_cross_chunk_size(
            state_chunk_size,
            total=state_count,
            default=CROSS_LIKELIHOOD_STATE_CHUNK_SIZE,
            label="state",
        )
        view_chunk = self._resolved_cross_chunk_size(
            view_chunk_size,
            total=view_count,
            default=SUBSET_LIKELIHOOD_VIEW_CHUNK_SIZE,
            label="view",
        )
        live = np.asarray(live_times_s_v, dtype=np.float64)
        observed_flat = observed.reshape((action_count,) + tuple(observed.shape[-3:]))
        total_flat = total.reshape((action_count,) + tuple(total.shape[-4:]))
        uncollided_flat = uncollided.reshape(total_flat.shape)
        features_flat = features.reshape((action_count,) + tuple(features.shape[-5:]))
        component = self.physical_component_discrepancy
        mark_node_count = 1
        node_count = int(self._rate_scale_nodes_j.size)
        view_node_log = np.empty(
            (
                action_count,
                sample_count,
                state_count,
                node_count,
                mark_node_count,
                view_count,
            ),
            dtype=np.float64,
        )
        shared_gamma = bool(
            self.count_discrepancy_concentration is not None
            and component is None
            and self.count_discrepancy_scope != "view_independent"
        )
        shared_observed = np.sum(observed_flat, axis=-1) if shared_gamma else None
        shared_expected = (
            np.empty(
                (action_count, state_count, node_count, view_count),
                dtype=np.float64,
            )
            if shared_gamma
            else None
        )
        latent_log_weights = None
        for action_start in range(0, action_count, action_chunk):
            action_stop = min(action_start + action_chunk, action_count)
            for state_start in range(0, state_count, state_chunk):
                state_stop = min(state_start + state_chunk, state_count)
                for sample_start in range(0, sample_count, sample_chunk):
                    sample_stop = min(sample_start + sample_chunk, sample_count)
                    for view_start in range(0, view_count, view_chunk):
                        view_stop = min(view_start + view_chunk, view_count)
                        block = self._prepare_subset_cross_likelihood_numpy_unchunked(
                            observed_flat[
                                action_start:action_stop,
                                sample_start:sample_stop,
                                view_start:view_stop,
                            ],
                            total_flat[
                                action_start:action_stop,
                                state_start:state_stop,
                                view_start:view_stop,
                            ],
                            uncollided_flat[
                                action_start:action_stop,
                                state_start:state_stop,
                                view_start:view_stop,
                            ],
                            features_flat[
                                action_start:action_stop,
                                state_start:state_stop,
                                view_start:view_stop,
                            ],
                            live[view_start:view_stop],
                        )
                        view_node_log[
                            action_start:action_stop,
                            sample_start:sample_stop,
                            state_start:state_stop,
                            ...,
                            view_start:view_stop,
                        ] = block.view_node_log_aqnjrv
                        if latent_log_weights is None:
                            latent_log_weights = block.latent_log_weights_jr
                        if (
                            shared_expected is not None
                            and sample_start == 0
                            and block.shared_expected_counts_anjv is not None
                        ):
                            shared_expected[
                                action_start:action_stop,
                                state_start:state_stop,
                                ...,
                                view_start:view_stop,
                            ] = block.shared_expected_counts_anjv
        if latent_log_weights is None:
            raise RuntimeError("Subset likelihood preparation produced no blocks.")
        return PreparedNumpySubsetCrossLikelihood(
            leading_shape=leading_shape,
            view_node_log_aqnjrv=view_node_log,
            latent_log_weights_jr=np.ascontiguousarray(latent_log_weights),
            shared_gamma_concentration=(
                float(self.count_discrepancy_concentration) if shared_gamma else None
            ),
            shared_observed_counts_aqv=shared_observed,
            shared_expected_counts_anjv=shared_expected,
        )

    def prepare_subset_cross_likelihood_torch(
        self,
        observed_spectra_xqvb: object,
        total_line_contributions_xnvsl: object,
        uncollided_line_contributions_xnvsl: object,
        transport_features_xnvslf: object,
        live_times_s_v: object,
        *,
        action_chunk_size: int | None = None,
        sample_chunk_size: int | None = None,
        state_chunk_size: int | None = None,
        view_chunk_size: int | None = None,
        prepared_observation: PreparedTorchCrossObservation | None = None,
    ) -> PreparedTorchSubsetCrossLikelihood:
        """Prepare reusable device-resident Torch terms for view subsets.

        Full-spectrum marking is evaluated in bounded GPU slabs once.  Every
        candidate subset thereafter uses bounded matrix contractions and
        retains the exact station-shared nuisance integration.
        """
        import torch

        total = torch.as_tensor(total_line_contributions_xnvsl)
        observation_terms = prepared_observation
        if observation_terms is None:
            observation_terms = self.prepare_cross_observation_torch(
                observed_spectra_xqvb,
                reference=total,
            )
        observed = torch.as_tensor(
            observation_terms.restored(observation_terms.observed_asvb),
            device=total.device,
            dtype=total.dtype,
        )
        uncollided = torch.as_tensor(
            uncollided_line_contributions_xnvsl,
            device=total.device,
            dtype=total.dtype,
        )
        features = torch.as_tensor(
            transport_features_xnvslf,
            device=total.device,
            dtype=total.dtype,
        )
        if observed.ndim < 3 or total.ndim < 4:
            raise ValueError("Torch subset likelihood inputs have too few dimensions.")
        leading_shape = tuple(int(value) for value in observation_terms.leading_shape)
        if tuple(int(value) for value in total.shape[:-4]) != leading_shape:
            raise ValueError(
                "Torch subset spectra and states require identical action axes."
            )
        if (
            tuple(uncollided.shape) != tuple(total.shape)
            or tuple(features.shape)
            != tuple(total.shape) + (len(TRANSPORT_FEATURE_ORDER),)
            or int(observed.shape[-2]) != int(total.shape[-3])
            or int(observed.shape[-1]) != int(np.asarray(self.energy_axis_keV).size)
        ):
            raise ValueError("Torch subset likelihood tensor shapes are inconsistent.")
        action_count = int(np.prod(leading_shape, dtype=np.int64))
        if not leading_shape:
            action_count = 1
        sample_count = int(observed.shape[-3])
        state_count = int(total.shape[-4])
        view_count = int(total.shape[-3])
        action_chunk = self._resolved_cross_chunk_size(
            action_chunk_size,
            total=action_count,
            default=CROSS_LIKELIHOOD_ACTION_CHUNK_SIZE,
            label="action",
        )
        sample_chunk = self._resolved_cross_chunk_size(
            sample_chunk_size,
            total=sample_count,
            default=CROSS_LIKELIHOOD_SAMPLE_CHUNK_SIZE,
            label="sample",
        )
        state_chunk = self._resolved_cross_chunk_size(
            state_chunk_size,
            total=state_count,
            default=CROSS_LIKELIHOOD_STATE_CHUNK_SIZE,
            label="state",
        )
        view_chunk = self._resolved_cross_chunk_size(
            view_chunk_size,
            total=view_count,
            default=SUBSET_LIKELIHOOD_VIEW_CHUNK_SIZE,
            label="view",
        )
        live = torch.as_tensor(
            live_times_s_v,
            device=total.device,
            dtype=total.dtype,
        )
        observed_flat = observed.reshape((action_count,) + tuple(observed.shape[-3:]))
        total_flat = total.reshape((action_count,) + tuple(total.shape[-4:]))
        uncollided_flat = uncollided.reshape(total_flat.shape)
        features_flat = features.reshape((action_count,) + tuple(features.shape[-5:]))
        component = self.physical_component_discrepancy
        mark_node_count = 1
        node_count = int(self._rate_scale_nodes_j.size)
        view_node_log = torch.empty(
            (
                action_count,
                sample_count,
                state_count,
                node_count,
                mark_node_count,
                view_count,
            ),
            device=total.device,
            dtype=total.dtype,
        )
        shared_gamma = bool(
            self.count_discrepancy_concentration is not None
            and component is None
            and self.count_discrepancy_scope != "view_independent"
        )
        shared_observed = torch.sum(observed_flat, dim=-1) if shared_gamma else None
        shared_expected = (
            torch.empty(
                (action_count, state_count, node_count, view_count),
                device=total.device,
                dtype=total.dtype,
            )
            if shared_gamma
            else None
        )
        latent_log_weights = None
        for action_start in range(0, action_count, action_chunk):
            action_stop = min(action_start + action_chunk, action_count)
            for state_start in range(0, state_count, state_chunk):
                state_stop = min(state_start + state_chunk, state_count)
                for sample_start in range(0, sample_count, sample_chunk):
                    sample_stop = min(sample_start + sample_chunk, sample_count)
                    for view_start in range(0, view_count, view_chunk):
                        view_stop = min(view_start + view_chunk, view_count)
                        observation_block = observation_terms.block(
                            action_start=action_start,
                            action_stop=action_stop,
                            sample_start=sample_start,
                            sample_stop=sample_stop,
                            view_start=view_start,
                            view_stop=view_stop,
                        )
                        block = self._prepare_subset_cross_likelihood_torch_unchunked(
                            observed_flat[
                                action_start:action_stop,
                                sample_start:sample_stop,
                                view_start:view_stop,
                            ],
                            total_flat[
                                action_start:action_stop,
                                state_start:state_stop,
                                view_start:view_stop,
                            ],
                            uncollided_flat[
                                action_start:action_stop,
                                state_start:state_stop,
                                view_start:view_stop,
                            ],
                            features_flat[
                                action_start:action_stop,
                                state_start:state_stop,
                                view_start:view_stop,
                            ],
                            live[view_start:view_stop],
                            prepared_observation=observation_block,
                        )
                        view_node_log[
                            action_start:action_stop,
                            sample_start:sample_stop,
                            state_start:state_stop,
                            ...,
                            view_start:view_stop,
                        ] = block.view_node_log_aqnjrv
                        if latent_log_weights is None:
                            latent_log_weights = block.latent_log_weights_jr
                        if (
                            shared_expected is not None
                            and sample_start == 0
                            and block.shared_expected_counts_anjv is not None
                        ):
                            shared_expected[
                                action_start:action_stop,
                                state_start:state_stop,
                                ...,
                                view_start:view_stop,
                            ] = block.shared_expected_counts_anjv
        if latent_log_weights is None:
            raise RuntimeError("Torch subset preparation produced no blocks.")
        return PreparedTorchSubsetCrossLikelihood(
            leading_shape=leading_shape,
            view_node_log_aqnjrv=view_node_log,
            latent_log_weights_jr=latent_log_weights,
            shared_gamma_concentration=(
                float(self.count_discrepancy_concentration) if shared_gamma else None
            ),
            shared_observed_counts_aqv=shared_observed,
            shared_expected_counts_anjv=shared_expected,
        )

    def cross_log_likelihood_numpy(
        self,
        observed_spectra_xqvb: NDArray[np.float64],
        total_line_contributions_xnvsl: NDArray[np.float64],
        uncollided_line_contributions_xnvsl: NDArray[np.float64],
        transport_features_xnvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
        *,
        action_chunk_size: int | None = None,
        sample_chunk_size: int | None = None,
        state_chunk_size: int | None = None,
    ) -> NDArray[np.float64]:
        """Return exact cross likelihoods with bounded action/sample/state memory."""
        observed = np.asarray(observed_spectra_xqvb, dtype=np.float64)
        total = np.asarray(total_line_contributions_xnvsl, dtype=np.float64)
        uncollided = np.asarray(
            uncollided_line_contributions_xnvsl,
            dtype=np.float64,
        )
        features = np.asarray(transport_features_xnvslf, dtype=np.float64)
        if observed.ndim < 3 or total.ndim < 4:
            raise ValueError("Cross-likelihood inputs have too few dimensions.")
        leading_shape = tuple(int(value) for value in observed.shape[:-3])
        if tuple(int(value) for value in total.shape[:-4]) != leading_shape:
            raise ValueError(
                "Cross spectra and transport states require identical action axes."
            )
        if (
            uncollided.shape != total.shape
            or features.shape != total.shape + (len(TRANSPORT_FEATURE_ORDER),)
            or int(observed.shape[-2]) != int(total.shape[-3])
            or int(observed.shape[-1]) != int(np.asarray(self.energy_axis_keV).size)
        ):
            raise ValueError("Cross-likelihood tensor shapes are inconsistent.")
        action_count = int(np.prod(leading_shape, dtype=np.int64))
        if not leading_shape:
            action_count = 1
        sample_count = int(observed.shape[-3])
        state_count = int(total.shape[-4])
        action_chunk = self._resolved_cross_chunk_size(
            action_chunk_size,
            total=action_count,
            default=CROSS_LIKELIHOOD_ACTION_CHUNK_SIZE,
            label="action",
        )
        sample_chunk = self._resolved_cross_chunk_size(
            sample_chunk_size,
            total=sample_count,
            default=CROSS_LIKELIHOOD_SAMPLE_CHUNK_SIZE,
            label="sample",
        )
        state_chunk = self._resolved_cross_chunk_size(
            state_chunk_size,
            total=state_count,
            default=CROSS_LIKELIHOOD_STATE_CHUNK_SIZE,
            label="state",
        )
        observed_flat = observed.reshape((action_count,) + tuple(observed.shape[-3:]))
        total_flat = total.reshape((action_count,) + tuple(total.shape[-4:]))
        uncollided_flat = uncollided.reshape(total_flat.shape)
        features_flat = features.reshape((action_count,) + tuple(features.shape[-5:]))
        result = np.empty(
            (action_count, sample_count, state_count),
            dtype=np.float64,
        )
        for action_start in range(0, action_count, action_chunk):
            action_stop = min(action_start + action_chunk, action_count)
            for state_start in range(0, state_count, state_chunk):
                state_stop = min(state_start + state_chunk, state_count)
                total_block = total_flat[
                    action_start:action_stop,
                    state_start:state_stop,
                ]
                uncollided_block = uncollided_flat[
                    action_start:action_stop,
                    state_start:state_stop,
                ]
                features_block = features_flat[
                    action_start:action_stop,
                    state_start:state_stop,
                ]
                for sample_start in range(0, sample_count, sample_chunk):
                    sample_stop = min(sample_start + sample_chunk, sample_count)
                    result[
                        action_start:action_stop,
                        sample_start:sample_stop,
                        state_start:state_stop,
                    ] = self._cross_log_likelihood_numpy_unchunked(
                        observed_flat[
                            action_start:action_stop,
                            sample_start:sample_stop,
                        ],
                        total_block,
                        uncollided_block,
                        features_block,
                        live_times_s_v,
                    )
        return result.reshape(leading_shape + (sample_count, state_count))

    def cross_log_likelihood_torch(
        self,
        observed_spectra_xqvb: object,
        total_line_contributions_xnvsl: object,
        uncollided_line_contributions_xnvsl: object,
        transport_features_xnvslf: object,
        live_times_s_v: object,
        *,
        action_chunk_size: int | None = None,
        sample_chunk_size: int | None = None,
        state_chunk_size: int | None = None,
        prepared_observation: PreparedTorchCrossObservation | None = None,
    ) -> object:
        """Return exact Torch likelihoods with cached CUDA chunk selection."""
        import torch

        total = torch.as_tensor(total_line_contributions_xnvsl)
        prepared = prepared_observation
        if prepared is None:
            prepared = self.prepare_cross_observation_torch(
                observed_spectra_xqvb,
                reference=total,
            )
        observed = torch.as_tensor(
            prepared.restored(prepared.observed_asvb),
            device=total.device,
            dtype=total.dtype,
        )
        uncollided = torch.as_tensor(
            uncollided_line_contributions_xnvsl,
            device=total.device,
            dtype=total.dtype,
        )
        features = torch.as_tensor(
            transport_features_xnvslf,
            device=total.device,
            dtype=total.dtype,
        )
        if observed.ndim < 3 or total.ndim < 4:
            raise ValueError("Torch cross-likelihood inputs have too few dimensions.")
        leading_shape = tuple(int(value) for value in prepared.leading_shape)
        if tuple(int(value) for value in total.shape[:-4]) != leading_shape:
            raise ValueError("Torch spectra and states require identical action axes.")
        if (
            tuple(uncollided.shape) != tuple(total.shape)
            or tuple(features.shape)
            != tuple(total.shape) + (len(TRANSPORT_FEATURE_ORDER),)
            or int(observed.shape[-2]) != int(total.shape[-3])
            or int(observed.shape[-1]) != int(np.asarray(self.energy_axis_keV).size)
        ):
            raise ValueError("Torch cross-likelihood tensor shapes are inconsistent.")
        action_count = int(np.prod(leading_shape, dtype=np.int64))
        if not leading_shape:
            action_count = 1
        sample_count = int(observed.shape[-3])
        state_count = int(total.shape[-4])
        action_chunk = self._resolved_cross_chunk_size(
            action_chunk_size,
            total=action_count,
            default=CROSS_LIKELIHOOD_ACTION_CHUNK_SIZE,
            label="action",
        )
        sample_chunk = self._resolved_cross_chunk_size(
            sample_chunk_size,
            total=sample_count,
            default=CROSS_LIKELIHOOD_SAMPLE_CHUNK_SIZE,
            label="sample",
        )
        state_chunk = self._resolved_cross_chunk_size(
            state_chunk_size,
            total=state_count,
            default=CROSS_LIKELIHOOD_STATE_CHUNK_SIZE,
            label="state",
        )
        observed_flat = observed.reshape((action_count,) + tuple(observed.shape[-3:]))
        total_flat = total.reshape((action_count,) + tuple(total.shape[-4:]))
        uncollided_flat = uncollided.reshape(total_flat.shape)
        features_flat = features.reshape((action_count,) + tuple(features.shape[-5:]))
        result = torch.empty(
            (action_count, sample_count, state_count),
            device=total.device,
            dtype=total.dtype,
        )

        def _evaluate_state_slab(state_start: int, state_stop: int) -> None:
            """Evaluate one state slab across every action and sample chunk."""
            for action_start in range(0, action_count, action_chunk):
                action_stop = min(action_start + action_chunk, action_count)
                total_block = total_flat[
                    action_start:action_stop,
                    state_start:state_stop,
                ]
                uncollided_block = uncollided_flat[
                    action_start:action_stop,
                    state_start:state_stop,
                ]
                features_block = features_flat[
                    action_start:action_stop,
                    state_start:state_stop,
                ]
                for sample_start in range(0, sample_count, sample_chunk):
                    sample_stop = min(sample_start + sample_chunk, sample_count)
                    prepared_block = prepared.block(
                        action_start=action_start,
                        action_stop=action_stop,
                        sample_start=sample_start,
                        sample_stop=sample_stop,
                    )
                    result[
                        action_start:action_stop,
                        sample_start:sample_stop,
                        state_start:state_stop,
                    ] = self._cross_log_likelihood_torch_unchunked(
                        observed_flat[
                            action_start:action_stop,
                            sample_start:sample_stop,
                        ],
                        total_block,
                        uncollided_block,
                        features_block,
                        live_times_s_v,
                        prepared_observation=prepared_block,
                    )

        autotune_start = 0
        if state_chunk_size is None and bool(total.is_cuda) and state_count > 256:
            key = self._torch_cross_state_autotune_key(
                total=total,
                action_count=action_count,
                sample_count=sample_count,
                action_chunk=action_chunk,
                sample_chunk=sample_chunk,
            )
            cached_chunk = self._torch_cross_state_chunk_cache.get(key)
            if cached_chunk is not None:
                state_chunk = min(int(cached_chunk), state_count)
                self.last_torch_cross_chunk_diagnostics = {
                    "mode": "cached_cuda_autotune",
                    "selected_state_chunk_size": int(state_chunk),
                    "shape_key": key,
                }
            else:
                free_bytes, _ = torch.cuda.mem_get_info(total.device)
                memory_limit = max(1, int(free_bytes) // 2)
                memory_maximum = CROSS_LIKELIHOOD_STATE_CHUNK_SIZE
                for candidate in (256, 512, 1024):
                    estimate = self.estimate_cross_likelihood_working_set_bytes(
                        num_actions=action_count,
                        num_samples=sample_count,
                        num_particles=state_count,
                        num_isotopes=int(total.shape[-2]),
                        num_views=int(total.shape[-3]),
                        action_chunk_size=action_chunk,
                        sample_chunk_size=sample_chunk,
                        state_chunk_size=min(candidate, state_count),
                        dtype_bytes=int(total.element_size()),
                    )
                    if estimate <= memory_limit:
                        memory_maximum = candidate
                candidates = self._torch_cross_state_autotune_candidates(
                    state_count=state_count,
                    memory_limited_maximum=memory_maximum,
                )
                trials: list[dict[str, float | int | str]] = []
                for candidate in candidates:
                    state_stop = autotune_start + candidate
                    torch.cuda.synchronize(total.device)
                    trial_start = time.perf_counter()
                    try:
                        _evaluate_state_slab(autotune_start, state_stop)
                        torch.cuda.synchronize(total.device)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        trials.append(
                            {
                                "state_chunk_size": int(candidate),
                                "states_per_second": 0.0,
                                "elapsed_s": float("inf"),
                                "status": "cuda_oom",
                            }
                        )
                        break
                    elapsed = time.perf_counter() - trial_start
                    trials.append(
                        {
                            "state_chunk_size": int(candidate),
                            "states_per_second": float(
                                candidate / max(elapsed, 1.0e-12)
                            ),
                            "elapsed_s": float(elapsed),
                            "status": "ok",
                        }
                    )
                    autotune_start = state_stop
                successful = [trial for trial in trials if trial["status"] == "ok"]
                if successful:
                    selected = max(
                        successful,
                        key=lambda trial: float(trial["states_per_second"]),
                    )
                    state_chunk = int(selected["state_chunk_size"])
                else:
                    state_chunk = min(
                        CROSS_LIKELIHOOD_STATE_CHUNK_SIZE,
                        state_count,
                    )
                self._torch_cross_state_chunk_cache[key] = state_chunk
                self.last_torch_cross_chunk_diagnostics = {
                    "mode": "empirical_cuda_autotune",
                    "selected_state_chunk_size": int(state_chunk),
                    "memory_limited_maximum": int(memory_maximum),
                    "trials": trials,
                    "shape_key": key,
                }
                trial_summary = ",".join(
                    f"{int(trial['state_chunk_size'])}:"
                    f"{float(trial['states_per_second']):.3g}state/s"
                    for trial in successful
                )
                print(
                    "[full-spectrum] state-chunk-autotune "
                    f"trials={trial_summary or 'none'} "
                    f"selected={state_chunk}",
                    flush=True,
                )
        else:
            self.last_torch_cross_chunk_diagnostics = {
                "mode": "explicit_or_non_cuda",
                "selected_state_chunk_size": int(state_chunk),
            }
        for state_start in range(autotune_start, state_count, state_chunk):
            state_stop = min(state_start + state_chunk, state_count)
            _evaluate_state_slab(state_start, state_stop)
        return result.reshape(leading_shape + (sample_count, state_count))

    def cross_log_likelihood_replace_slots_torch(
        self,
        observed_spectra_xqvb: object,
        accepted_total_line_contributions_xNvsl: object,
        accepted_uncollided_line_contributions_xNvsl: object,
        accepted_transport_features_xNvslf: object,
        replacement_total_line_contributions_xnvrl: object,
        replacement_uncollided_line_contributions_xnvrl: object,
        replacement_transport_features_xnvrlf: object,
        live_times_s_v: object,
        *,
        particle_indices_n: object,
        slot_start: int,
        slot_stop: int,
        action_chunk_size: int | None = None,
        sample_chunk_size: int | None = None,
        state_chunk_size: int | None = None,
        prepared_observation: PreparedTorchCrossObservation | None = None,
    ) -> object:
        """Evaluate exact likelihoods after one source-slot block replacement.

        The accepted source-resolved tensors remain immutable.  Only one
        bounded state slab is gathered and overlaid at a time before entering
        :meth:`cross_log_likelihood_torch`, so proposal scoring never clones a
        complete particle-by-history tensor.  The unchanged slots and the
        replacement slots still enter the same hierarchical likelihood kernel;
        this method changes memory scheduling only.
        """
        import torch

        accepted_total = torch.as_tensor(
            accepted_total_line_contributions_xNvsl
        )
        accepted_uncollided = torch.as_tensor(
            accepted_uncollided_line_contributions_xNvsl
        )
        accepted_features = torch.as_tensor(
            accepted_transport_features_xNvslf
        )
        replacement_total = torch.as_tensor(
            replacement_total_line_contributions_xnvrl
        )
        replacement_uncollided = torch.as_tensor(
            replacement_uncollided_line_contributions_xnvrl
        )
        replacement_features = torch.as_tensor(
            replacement_transport_features_xnvrlf
        )
        tensors = (
            accepted_total,
            accepted_uncollided,
            accepted_features,
            replacement_total,
            replacement_uncollided,
            replacement_features,
        )
        if any(
            value.device != accepted_total.device
            or value.dtype != accepted_total.dtype
            for value in tensors[1:]
        ):
            raise ValueError(
                "Accepted and replacement slot tensors must share device and dtype."
            )
        if accepted_total.ndim < 4:
            raise ValueError("Accepted slot-overlay tensors have too few dimensions.")
        leading_shape = tuple(int(value) for value in accepted_total.shape[:-4])
        accepted_state_count = int(accepted_total.shape[-4])
        view_count = int(accepted_total.shape[-3])
        source_slot_count = int(accepted_total.shape[-2])
        line_count = int(accepted_total.shape[-1])
        feature_count = len(TRANSPORT_FEATURE_ORDER)
        if (
            tuple(accepted_uncollided.shape) != tuple(accepted_total.shape)
            or tuple(accepted_features.shape)
            != tuple(accepted_total.shape) + (feature_count,)
        ):
            raise ValueError("Accepted slot-overlay tensor shapes are inconsistent.")
        if type(slot_start) is not int or type(slot_stop) is not int:
            raise TypeError("Slot-overlay bounds must be exact integers.")
        start = slot_start
        stop = slot_stop
        if start < 0 or stop <= start or stop > source_slot_count:
            raise ValueError("Slot-overlay bounds are outside the accepted state.")
        replacement_slot_count = stop - start
        proposal_state_count = int(replacement_total.shape[-4])
        expected_replacement_shape = leading_shape + (
            proposal_state_count,
            view_count,
            replacement_slot_count,
            line_count,
        )
        if (
            tuple(replacement_total.shape) != expected_replacement_shape
            or tuple(replacement_uncollided.shape) != expected_replacement_shape
            or tuple(replacement_features.shape)
            != expected_replacement_shape + (feature_count,)
        ):
            raise ValueError(
                "Replacement slot tensors do not align with the accepted state."
            )
        if not torch.is_tensor(particle_indices_n):
            raise TypeError("Slot-overlay particle indices must be a Torch tensor.")
        if (
            particle_indices_n.device != accepted_total.device
            or particle_indices_n.dtype != torch.long
        ):
            raise ValueError(
                "Slot-overlay particle indices must share the accepted device "
                "and use torch.long."
            )
        indices = particle_indices_n.reshape(-1)
        if int(indices.numel()) != proposal_state_count:
            raise ValueError(
                "Slot-overlay particle indices must align with proposal states."
            )
        if proposal_state_count == 0:
            raise ValueError("Slot-overlay likelihood requires at least one state.")
        if bool(
            torch.any(
                (indices < 0) | (indices >= accepted_state_count)
            ).item()
        ):
            raise IndexError("Slot-overlay particle index is out of range.")
        prepared = prepared_observation
        if prepared is None:
            prepared = self.prepare_cross_observation_torch(
                observed_spectra_xqvb,
                reference=accepted_total,
            )
        if tuple(int(value) for value in prepared.leading_shape) != leading_shape:
            raise ValueError(
                "Slot-overlay observations and transport require identical action axes."
            )
        sample_count = int(prepared.observed_asvb.shape[-3])
        action_count = int(np.prod(leading_shape, dtype=np.int64))
        if not leading_shape:
            action_count = 1
        chunk_selection_mode = "explicit_or_non_cuda"
        if state_chunk_size is None and bool(accepted_total.is_cuda):
            free_bytes, _ = torch.cuda.mem_get_info(accepted_total.device)
            memory_limit = max(1, int(free_bytes) // 2)
            resolved_state_chunk = min(32, proposal_state_count)
            minimum_fits = False
            for candidate in (32, 64, 128, 256, 512, 1024):
                candidate_count = min(candidate, proposal_state_count)
                likelihood_bytes = self.estimate_cross_likelihood_working_set_bytes(
                    num_actions=action_count,
                    num_samples=sample_count,
                    num_particles=proposal_state_count,
                    num_isotopes=source_slot_count,
                    num_views=view_count,
                    action_chunk_size=action_chunk_size,
                    sample_chunk_size=sample_chunk_size,
                    state_chunk_size=candidate_count,
                    dtype_bytes=int(accepted_total.element_size()),
                )
                scratch_elements = (
                    action_count
                    * candidate_count
                    * view_count
                    * source_slot_count
                    * line_count
                    * (2 + feature_count)
                )
                scratch_bytes = scratch_elements * int(
                    accepted_total.element_size()
                )
                if likelihood_bytes + scratch_bytes <= memory_limit:
                    resolved_state_chunk = candidate_count
                    minimum_fits = True
            if not minimum_fits:
                raise torch.cuda.OutOfMemoryError(
                    "Exact slot-overlay likelihood cannot fit its minimum "
                    "bounded CUDA state slab."
                )
            chunk_selection_mode = "cuda_memory_bounded"
        else:
            resolved_state_chunk = self._resolved_cross_chunk_size(
                state_chunk_size,
                total=proposal_state_count,
                default=CROSS_LIKELIHOOD_STATE_CHUNK_SIZE,
                label="state",
            )
        result = torch.empty(
            leading_shape + (sample_count, proposal_state_count),
            device=accepted_total.device,
            dtype=accepted_total.dtype,
        )
        scratch_peak_bytes = 0
        slab_count = 0
        for state_start in range(0, proposal_state_count, resolved_state_chunk):
            state_stop = min(
                state_start + resolved_state_chunk,
                proposal_state_count,
            )
            slab_indices = indices[state_start:state_stop]
            total_slab = torch.index_select(
                accepted_total,
                dim=-4,
                index=slab_indices,
            )
            uncollided_slab = torch.index_select(
                accepted_uncollided,
                dim=-4,
                index=slab_indices,
            )
            features_slab = torch.index_select(
                accepted_features,
                dim=-5,
                index=slab_indices,
            )
            total_slab[..., start:stop, :] = replacement_total[
                ..., state_start:state_stop, :, :, :
            ]
            uncollided_slab[..., start:stop, :] = replacement_uncollided[
                ..., state_start:state_stop, :, :, :
            ]
            features_slab[..., start:stop, :, :] = replacement_features[
                ..., state_start:state_stop, :, :, :, :
            ]
            scratch_peak_bytes = max(
                scratch_peak_bytes,
                int(
                    sum(
                        int(value.numel()) * int(value.element_size())
                        for value in (
                            total_slab,
                            uncollided_slab,
                            features_slab,
                        )
                    )
                ),
            )
            slab_result = self.cross_log_likelihood_torch(
                observed_spectra_xqvb,
                total_slab,
                uncollided_slab,
                features_slab,
                live_times_s_v,
                action_chunk_size=action_chunk_size,
                sample_chunk_size=sample_chunk_size,
                state_chunk_size=state_stop - state_start,
                prepared_observation=prepared,
            )
            result[..., state_start:state_stop] = slab_result
            slab_count += 1
        self.last_torch_slot_overlay_diagnostics = {
            "mode": "bounded_exact_slot_overlay",
            "chunk_selection_mode": chunk_selection_mode,
            "proposal_state_count": proposal_state_count,
            "accepted_state_count": accepted_state_count,
            "state_chunk_size": resolved_state_chunk,
            "slab_count": slab_count,
            "replacement_slot_count": replacement_slot_count,
            "source_slot_count": source_slot_count,
            "scratch_peak_bytes": scratch_peak_bytes,
            "full_history_clone_count": 0,
        }
        return result

    def log_likelihood_numpy(
        self,
        observed_spectrum_vb: NDArray[np.float64],
        total_line_contributions_nvsl: NDArray[np.float64],
        uncollided_line_contributions_nvsl: NDArray[np.float64],
        transport_features_nvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return one joint full-spectrum log likelihood per particle."""
        observed = np.asarray(observed_spectrum_vb, dtype=np.float64)
        if observed.ndim != 2:
            raise ValueError("One station observation must be view x bin.")
        return self.cross_log_likelihood_numpy(
            observed[np.newaxis, ...],
            total_line_contributions_nvsl,
            uncollided_line_contributions_nvsl,
            transport_features_nvslf,
            live_times_s_v,
        )[0]

    def count_log_likelihood_numpy(
        self,
        observed_spectrum_vb: NDArray[np.float64],
        total_line_contributions_nvsl: NDArray[np.float64],
        uncollided_line_contributions_nvsl: NDArray[np.float64],
        transport_features_nvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return the station total-count likelihood without spectral marks.

        This diagnostic integrates the same station-shared rate-scale latent
        variable and uses the same nonparalyzable renewal law as the full
        likelihood. It intentionally omits only the conditional energy-mark
        term, so it distinguishes total-rate pressure from spectral-shape
        pressure without changing the inference target.
        """
        observed = np.asarray(observed_spectrum_vb, dtype=np.float64)
        total = np.asarray(total_line_contributions_nvsl, dtype=np.float64)
        if observed.ndim != 2 or total.ndim != 4:
            raise ValueError(
                "Count diagnostics require view x bin observations and "
                "particle x view x isotope x line transport arrays."
            )
        source_mean, background_mean = self._pre_dead_time_mean_numpy(
            total,
            uncollided_line_contributions_nvsl,
            transport_features_nvslf,
            live_times_s_v,
            return_components=True,
        )
        if observed.shape != source_mean.shape[-2:]:
            raise ValueError(
                "Count diagnostic spectra and model means are inconsistent."
            )
        node_source = (
            source_mean[:, np.newaxis, :, :]
            * self._rate_scale_nodes_j[
                np.newaxis,
                :,
                np.newaxis,
                np.newaxis,
            ]
        )
        pre_mean = background_mean[:, np.newaxis, :, :] + node_source
        pre_total = np.sum(pre_mean, axis=-1)
        observed_total = np.sum(observed, axis=-1)
        live = np.asarray(live_times_s_v, dtype=np.float64)
        component_discrepancy = self.physical_component_discrepancy
        if (
            self.count_discrepancy_concentration is None
            and component_discrepancy is None
        ):
            count_log = nonparalyzable_count_log_probability_numpy(
                observed_total[np.newaxis, np.newaxis, :],
                pre_total / live[np.newaxis, np.newaxis, :],
                live[np.newaxis, np.newaxis, :],
                dead_time_tau_s=float(self.dead_time_tau_s),
            )
        else:
            dead_time_scale = 1.0 + pre_total / live[np.newaxis, np.newaxis, :] * float(
                self.dead_time_tau_s
            )
            component_count_concentration = None
            if component_discrepancy is not None:
                component_count_concentration = (
                    self._component_count_concentration_numpy(
                        total,
                        uncollided_line_contributions_nvsl,
                        transport_features_nvslf,
                    )[:, np.newaxis, :]
                )
            count_function = (
                view_independent_gamma_poisson_count_log_increments_numpy
                if component_discrepancy is not None
                or self.count_discrepancy_scope == "view_independent"
                else station_shared_gamma_poisson_count_log_increments_numpy
            )
            count_log = count_function(
                observed_total[np.newaxis, :],
                pre_total / dead_time_scale,
                concentration=(
                    component_count_concentration
                    if component_count_concentration is not None
                    else float(self.count_discrepancy_concentration)
                ),
            )[0]
            if component_discrepancy is not None:
                source_active = np.sum(node_source, axis=-1) > 0.0
                background_only_log = nonparalyzable_count_log_probability_numpy(
                    observed_total[np.newaxis, np.newaxis, :],
                    pre_total / live[np.newaxis, np.newaxis, :],
                    live[np.newaxis, np.newaxis, :],
                    dead_time_tau_s=float(self.dead_time_tau_s),
                )
                count_log = np.where(
                    source_active,
                    count_log,
                    background_only_log,
                )
        node_log = np.sum(count_log, axis=-1)
        return special.logsumexp(
            node_log + np.log(self._rate_scale_weights_j)[np.newaxis, :],
            axis=-1,
        )

    def decompose_log_likelihood_numpy(
        self,
        observed_spectrum_vb: NDArray[np.float64],
        total_line_contributions_nvsl: NDArray[np.float64],
        uncollided_line_contributions_nvsl: NDArray[np.float64],
        transport_features_nvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
    ) -> LikelihoodDecomposition:
        """Decompose production likelihood into count/background/source roles.

        The conditional mark law is one joint component-aware tree, so its
        factors are attributed to background and source in proportion to the
        expected component mass of each tree node or leaf.  This attribution
        is exactly additive and diagnostic only; inference continues to use
        the unsplit joint likelihood.
        """
        observed = np.asarray(observed_spectrum_vb, dtype=np.float64)
        total = np.asarray(total_line_contributions_nvsl, dtype=np.float64)
        uncollided = np.asarray(
            uncollided_line_contributions_nvsl,
            dtype=np.float64,
        )
        features = np.asarray(transport_features_nvslf, dtype=np.float64)
        live = np.asarray(live_times_s_v, dtype=np.float64)
        component = self.physical_component_discrepancy
        if (
            observed.ndim != 2
            or total.ndim != 4
            or uncollided.shape != total.shape
            or features.shape != total.shape + (len(TRANSPORT_FEATURE_ORDER),)
            or observed.shape != (total.shape[1], self._energy_axis_keV.size)
            or live.shape != (total.shape[1],)
        ):
            raise ValueError("Likelihood decomposition inputs are misaligned.")
        if (
            component is None
            or component.mark_latent_model != "component_dirichlet_tree_hierarchical"
            or not np.array_equal(
                self._rate_scale_nodes_j,
                np.asarray((1.0,), dtype=np.float64),
            )
            or not np.array_equal(
                self._rate_scale_weights_j,
                np.asarray((1.0,), dtype=np.float64),
            )
        ):
            raise RuntimeError(
                "Likelihood decomposition requires the production component "
                "tree without retired rate mixtures."
            )
        direct_mean, scatter_mean, background_mean = self._pre_dead_time_mean_numpy(
            total,
            uncollided,
            features,
            live,
            return_physical_components=True,
        )
        pre_mean = direct_mean + scatter_mean + background_mean
        pre_total = np.sum(pre_mean, axis=-1)
        probabilities = np.divide(
            pre_mean,
            pre_total[..., np.newaxis],
            out=np.zeros_like(pre_mean),
            where=pre_total[..., np.newaxis] > 0.0,
        )
        observed_total = np.sum(observed, axis=-1)
        recorded_total_mean = pre_total / (
            1.0 + pre_total / live * float(self.dead_time_tau_s)
        )
        count_concentration = self._component_count_concentration_numpy(
            total,
            uncollided,
            features,
        )
        count_log = view_independent_gamma_poisson_count_log_increments_numpy(
            observed_total[np.newaxis, :],
            recorded_total_mean[:, np.newaxis, :],
            concentration=count_concentration[:, np.newaxis, :],
        )[0, :, 0, :]
        source_active = np.sum(direct_mean + scatter_mean, axis=-1) > 0.0
        background_only_log = nonparalyzable_count_log_probability_numpy(
            observed_total[np.newaxis, :],
            pre_total / live,
            live,
            dead_time_tau_s=float(self.dead_time_tau_s),
        )
        count_log = np.where(source_active, count_log, background_only_log)

        component_means = np.stack(
            (direct_mean, scatter_mean, background_mean),
            axis=-2,
        )[:, np.newaxis, ...]
        tree_concentration, leaf_concentration = (
            self._component_tree_mark_concentrations_numpy(
                total,
                uncollided,
                component_means,
            )
        )
        mark_result = self._component_tree_mark_log_numpy(
            observed[np.newaxis, ...],
            probabilities[:, np.newaxis, ...],
            tree_concentration,
            leaf_concentration,
            return_factors=True,
        )
        if not isinstance(mark_result, tuple):  # pragma: no cover - type guard
            raise RuntimeError("Component-tree factors were not returned.")
        _mark_log, tree_log, leaf_log = mark_result
        tree_log_nvt = tree_log[0, :, 0, :, :]
        leaf_log_nvh = leaf_log[0, :, 0, :, :]
        tree_component_mass = np.einsum(
            "nvkb,tb->nvkt",
            component_means[:, 0],
            self._mark_tree_left_mask_tb + self._mark_tree_right_mask_tb,
            optimize=True,
        )
        leaf_component_mass = np.einsum(
            "nvkb,hb->nvkh",
            component_means[:, 0],
            self._mark_leaf_group_mask_hb,
            optimize=True,
        )
        tree_total_mass = np.sum(tree_component_mass, axis=-2)
        leaf_total_mass = np.sum(leaf_component_mass, axis=-2)
        tree_background_fraction = np.divide(
            tree_component_mass[..., 2, :],
            tree_total_mass,
            out=np.zeros_like(tree_total_mass),
            where=tree_total_mass > 0.0,
        )
        leaf_background_fraction = np.divide(
            leaf_component_mass[..., 2, :],
            leaf_total_mass,
            out=np.zeros_like(leaf_total_mass),
            where=leaf_total_mass > 0.0,
        )
        background_mark = np.sum(
            tree_log_nvt * tree_background_fraction,
            axis=-1,
        ) + np.sum(
            leaf_log_nvh * leaf_background_fraction,
            axis=-1,
        )
        source_mark = np.sum(
            tree_log_nvt * (1.0 - tree_background_fraction),
            axis=-1,
        ) + np.sum(
            leaf_log_nvh * (1.0 - leaf_background_fraction),
            axis=-1,
        )
        decomposition = LikelihoodDecomposition(
            total_count_nv=np.asarray(count_log, dtype=np.float64),
            background_mark_nv=np.asarray(background_mark, dtype=np.float64),
            source_mark_nv=np.asarray(source_mark, dtype=np.float64),
        )
        full = self.log_likelihood_numpy(
            observed,
            total,
            uncollided,
            features,
            live,
        )
        if not np.allclose(
            decomposition.total_log_likelihood_n,
            full,
            rtol=0.0,
            atol=1.0e-8,
        ):
            raise RuntimeError("Likelihood role decomposition is not exact.")
        return decomposition

    def log_likelihood_torch(
        self,
        observed_spectrum_vb: object,
        total_line_contributions_nvsl: object,
        uncollided_line_contributions_nvsl: object,
        transport_features_nvslf: object,
        live_times_s_v: object,
    ) -> object:
        """Return the Torch station likelihood for every particle."""
        import torch

        total = torch.as_tensor(total_line_contributions_nvsl)
        observed = torch.as_tensor(
            observed_spectrum_vb,
            device=total.device,
            dtype=total.dtype,
        )
        if observed.ndim != 2:
            raise ValueError("One Torch station observation must be view x bin.")
        return self.cross_log_likelihood_torch(
            observed.unsqueeze(0),
            total,
            uncollided_line_contributions_nvsl,
            transport_features_nvslf,
            live_times_s_v,
        )[0]

    def sample_predictive_torch(
        self,
        total_line_contributions_xvsl: object,
        uncollided_line_contributions_xvsl: object,
        transport_features_xvslf: object,
        live_times_s_v: object,
        *,
        sample_count: int,
        generator: object | None = None,
        action_seeds_a: object | None = None,
    ) -> object:
        """Draw exact future spectra without leaving the Torch device.

        An explicit device-matched generator is required for an unseeded
        action batch.  When ``action_seeds_a`` is supplied, its leading action
        axis is sampled with independent canonical streams, so action ordering
        and outer chunking cannot alter any action's draws.
        """
        from spectrum.predictive_torch import (
            sample_geometry_conditioned_predictive_torch,
        )

        return sample_geometry_conditioned_predictive_torch(
            self,
            total_line_contributions_xvsl,
            uncollided_line_contributions_xvsl,
            transport_features_xvslf,
            live_times_s_v,
            sample_count=sample_count,
            generator=generator,
            action_seeds_a=action_seeds_a,
        )

    def sample_predictive_numpy(
        self,
        total_line_contributions_xvsl: NDArray[np.float64],
        uncollided_line_contributions_xvsl: NDArray[np.float64],
        transport_features_xvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
        *,
        sample_count: int,
        rng: np.random.Generator,
        action_seeds_a: NDArray[np.int64] | None = None,
    ) -> NDArray[np.int64]:
        """Draw shared-scale renewal totals and calibrated energy marks."""
        if int(sample_count) <= 0:
            raise ValueError("sample_count must be positive.")
        if action_seeds_a is not None:
            total = np.asarray(
                total_line_contributions_xvsl,
                dtype=np.float64,
            )
            uncollided = np.asarray(
                uncollided_line_contributions_xvsl,
                dtype=np.float64,
            )
            features = np.asarray(
                transport_features_xvslf,
                dtype=np.float64,
            )
            seeds = np.asarray(action_seeds_a)
            if (
                total.ndim < 4
                or seeds.ndim != 1
                or seeds.shape != (int(total.shape[0]),)
                or not np.issubdtype(seeds.dtype, np.integer)
                or uncollided.shape != total.shape
                or features.shape != total.shape + (len(TRANSPORT_FEATURE_ORDER),)
            ):
                raise ValueError(
                    "action_seeds_a must provide one integer seed for the "
                    "leading action axis."
                )
            action_samples = []
            for action_index, raw_seed in enumerate(seeds):
                seed = int(raw_seed) & ((1 << 64) - 1)
                action_rng = np.random.Generator(np.random.Philox(seed))
                action_samples.append(
                    self.sample_predictive_numpy(
                        total[action_index],
                        uncollided[action_index],
                        features[action_index],
                        live_times_s_v,
                        sample_count=int(sample_count),
                        rng=action_rng,
                    )
                )
            return np.stack(action_samples, axis=0).astype(
                np.int64,
                copy=False,
            )
        component_discrepancy = self.physical_component_discrepancy
        component_tree_marks = bool(
            component_discrepancy is not None
            and component_discrepancy.mark_latent_model
            == "component_dirichlet_tree_hierarchical"
        )
        if component_tree_marks:
            direct_mean, scatter_mean, background_mean = self._pre_dead_time_mean_numpy(
                total_line_contributions_xvsl,
                uncollided_line_contributions_xvsl,
                transport_features_xvslf,
                live_times_s_v,
                return_physical_components=True,
            )
            source_mean = direct_mean + scatter_mean
        else:
            source_mean, background_mean = self._pre_dead_time_mean_numpy(
                total_line_contributions_xvsl,
                uncollided_line_contributions_xvsl,
                transport_features_xvslf,
                live_times_s_v,
                return_components=True,
            )
        live = np.asarray(live_times_s_v, dtype=np.float64)
        leading_shape = source_mean.shape[:-2]
        node_indices = rng.choice(
            self._rate_scale_nodes_j.size,
            size=leading_shape + (int(sample_count),),
            p=self._rate_scale_weights_j,
        )
        sampled_scale = self._rate_scale_nodes_j[node_indices]
        node_source = (
            source_mean[..., np.newaxis, :, :]
            * sampled_scale[..., np.newaxis, np.newaxis]
        )
        pre_mean = background_mean[..., np.newaxis, :, :] + node_source
        pre_total = np.sum(pre_mean, axis=-1)
        if (
            self.count_discrepancy_concentration is None
            and component_discrepancy is None
        ):
            rates = pre_total / live
            totals = sample_nonparalyzable_counts_numpy(
                rates,
                np.broadcast_to(live, rates.shape),
                dead_time_tau_s=float(self.dead_time_tau_s),
                sample_count=1,
                rng=rng,
            )[..., 0]
        else:
            dead_time_scale = 1.0 + pre_total / live * float(self.dead_time_tau_s)
            recorded_total_mean = pre_total / dead_time_scale
            if component_discrepancy is not None:
                base_count_concentration = self._component_count_concentration_numpy(
                    total_line_contributions_xvsl,
                    uncollided_line_contributions_xvsl,
                    transport_features_xvslf,
                )[..., np.newaxis, :]
                count_concentration = np.broadcast_to(
                    base_count_concentration,
                    recorded_total_mean.shape,
                )
                sampled_count_scale = rng.gamma(
                    shape=count_concentration,
                    scale=1.0 / count_concentration,
                )
            else:
                count_concentration = float(self.count_discrepancy_concentration)
                scale_shape = leading_shape + (int(sample_count),)
                if self.count_discrepancy_scope == "view_independent":
                    scale_shape += (int(recorded_total_mean.shape[-1]),)
                sampled_count_scale = rng.gamma(
                    shape=count_concentration,
                    scale=1.0 / count_concentration,
                    size=scale_shape,
                )
            if (
                component_discrepancy is None
                and self.count_discrepancy_scope == "station_shared"
            ):
                sampled_count_scale = sampled_count_scale[..., np.newaxis]
            totals = rng.poisson(recorded_total_mean * sampled_count_scale)
            if component_discrepancy is not None:
                source_active = np.sum(node_source, axis=-1) > 0.0
                if np.any(~source_active):
                    background_only_totals = sample_nonparalyzable_counts_numpy(
                        pre_total / live,
                        np.broadcast_to(live, pre_total.shape),
                        dead_time_tau_s=float(self.dead_time_tau_s),
                        sample_count=1,
                        rng=rng,
                    )[..., 0]
                    totals = np.where(
                        source_active,
                        totals,
                        background_only_totals,
                    )
        mark_pre_mean = pre_mean
        mark_total = np.sum(mark_pre_mean, axis=-1)
        probabilities = np.divide(
            mark_pre_mean,
            mark_total[..., np.newaxis],
            out=np.zeros_like(mark_pre_mean),
            where=mark_total[..., np.newaxis] > 0.0,
        )
        zero_rate = pre_total <= 0.0
        if np.any(zero_rate):
            zero_count_sentinel = np.zeros_like(probabilities)
            zero_count_sentinel[..., 0] = 1.0
            probabilities = np.where(
                zero_rate[..., np.newaxis],
                zero_count_sentinel,
                probabilities,
            )
        if component_tree_marks:
            sampled_direct = (
                direct_mean[..., np.newaxis, :, :]
                * sampled_scale[..., np.newaxis, np.newaxis]
            )
            sampled_scatter = (
                scatter_mean[..., np.newaxis, :, :]
                * sampled_scale[..., np.newaxis, np.newaxis]
            )
            component_means = np.stack(
                (
                    sampled_direct,
                    sampled_scatter,
                    np.broadcast_to(
                        background_mean[..., np.newaxis, :, :],
                        sampled_direct.shape,
                    ),
                ),
                axis=-2,
            )
            tree_concentration, leaf_concentration = (
                self._component_tree_mark_concentrations_numpy(
                    total_line_contributions_xvsl,
                    uncollided_line_contributions_xvsl,
                    component_means,
                )
            )
            random_probability = self._sample_component_tree_probabilities_numpy(
                probabilities,
                tree_concentration,
                leaf_concentration,
                rng=rng,
            )
            return np.asarray(
                rng.multinomial(totals, random_probability),
                dtype=np.int64,
            )
        if not component_tree_marks:
            base_concentration = self._base_mark_concentration_numpy(
                total_line_contributions_xvsl,
                uncollided_line_contributions_xvsl,
            )
            source_total = np.sum(node_source, axis=-1)
            source_fraction = np.divide(
                source_total,
                pre_total,
                out=np.zeros_like(source_total),
                where=pre_total > 0.0,
            )
            base_concentration = self._base_mark_concentration_numpy(
                total_line_contributions_xvsl,
                uncollided_line_contributions_xvsl,
            )
            concentration = base_concentration[..., np.newaxis, :] / np.maximum(
                np.square(source_fraction),
                1.0e-12,
            )
            alpha = probabilities * concentration[..., np.newaxis]
            positive_alpha = alpha > 0.0
            gamma_draws = rng.gamma(
                shape=np.where(positive_alpha, alpha, 1.0),
            )
            gamma_draws = np.where(
                positive_alpha,
                gamma_draws,
                0.0,
            )
            random_probabilities = np.divide(
                gamma_draws,
                np.sum(gamma_draws, axis=-1, keepdims=True),
                out=probabilities.copy(),
                where=np.sum(
                    gamma_draws,
                    axis=-1,
                    keepdims=True,
                )
                > 0.0,
            )
            probabilities = np.where(
                source_fraction[..., np.newaxis] > 0.0,
                random_probabilities,
                probabilities,
            )
        samples = rng.multinomial(
            totals,
            probabilities,
        )
        return np.asarray(samples, dtype=np.int64)

    def _birth_proposal_nuisance_basis_numpy(
        self,
        target_line_mask_l: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """Return a fixed whitened orthonormal non-target line subspace."""
        mask = np.asarray(target_line_mask_l, dtype=np.bool_)
        if mask.shape != (len(self._line_identity),) or not np.any(mask):
            raise ValueError("target_line_mask_l must select at least one global line.")
        key = np.ascontiguousarray(mask).tobytes()
        cached = self._proposal_basis_cache.get(key)
        if cached is not None:
            return cached
        nuisance_direct = self._marked_direct_line_shapes_lb[~mask]
        nuisance_scatter = (
            self._marked_detector_cone_scatter_shapes_dlb[:, ~mask, :].reshape(
                -1,
                self._energy_axis_keV.size,
            )
            if isinstance(
                self.additive_scatter_response,
                PhysicsOnlyNoncollidedTransportResponse,
            )
            else self._marked_scatter_order_shapes_lob[~mask].reshape(
                -1,
                self._energy_axis_keV.size,
            )
        )
        nuisance = np.concatenate(
            (nuisance_direct, nuisance_scatter),
            axis=0,
        )
        whitening = 1.0 / np.sqrt(
            self.background_shape_b + 1.0 / float(self._energy_axis_keV.size)
        )
        whitened = nuisance * whitening[np.newaxis, :]
        if whitened.shape[0] == 0:
            basis = np.zeros(
                (self._energy_axis_keV.size, 0),
                dtype=np.float64,
            )
        else:
            basis, _ = np.linalg.qr(whitened.T, mode="reduced")
        basis = np.ascontiguousarray(basis, dtype=np.float64)
        basis.setflags(write=False)
        self._proposal_basis_cache[key] = basis
        return basis

    def _birth_proposal_candidate_chunk_size(self, view_count: int) -> int:
        """Return a conservative candidate chunk under the memory cap."""
        values_per_candidate = (
            int(view_count)
            * int(self._energy_axis_keV.size)
            * (8 + 3 * int(self._rate_scale_nodes_j.size))
        )
        bytes_per_candidate = max(values_per_candidate * 8, 1)
        return max(
            1,
            int(BIRTH_PROPOSAL_WORKING_SET_BYTES // bytes_per_candidate),
        )

    def birth_proposal_log_scores_numpy(
        self,
        observed_spectrum_vb: NDArray[np.float64],
        candidate_total_line_contributions_gvsl: NDArray[np.float64],
        candidate_uncollided_line_contributions_gvsl: NDArray[np.float64],
        candidate_transport_features_gvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
        *,
        target_line_mask_l: NDArray[np.bool_],
        reference_mean_vb: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Return deterministic proposal-only chart-by-strength log scores.

        ``reference_mean_vb`` may contain the spectrum already explained by
        earlier blocks of a sequential importance proposal.  It changes only
        the proposal residual and whitening; candidate templates remain the
        target-isotope increment over the physical background.  The caller is
        responsible for retaining this conditional proposal density in the
        importance correction.
        """
        observed = np.asarray(observed_spectrum_vb, dtype=np.float64)
        total = np.asarray(
            candidate_total_line_contributions_gvsl,
            dtype=np.float64,
        )
        mask = np.asarray(target_line_mask_l, dtype=np.bool_)
        if (
            observed.shape
            != (
                int(total.shape[-3]),
                int(self._energy_axis_keV.size),
            )
            or np.any(~np.isfinite(observed))
            or np.any(observed < 0.0)
            or np.any(observed != np.floor(observed))
            or mask.shape != (len(self._line_identity),)
            or np.any(total[..., ~mask] != 0.0)
        ):
            raise ValueError(
                "Birth proposal candidates must contain exact observed counts "
                "and target-isotope line rates only."
            )
        uncollided = np.asarray(
            candidate_uncollided_line_contributions_gvsl,
            dtype=np.float64,
        )
        features = np.asarray(
            candidate_transport_features_gvslf,
            dtype=np.float64,
        )
        zero_total = np.zeros(
            (1,) + total.shape[-3:],
            dtype=np.float64,
        )
        zero_features = np.zeros(
            zero_total.shape + (len(TRANSPORT_FEATURE_ORDER),),
            dtype=np.float64,
        )
        baseline = self.predict_mean_numpy(
            zero_total,
            zero_total,
            zero_features,
            live_times_s_v,
        )[0]
        if reference_mean_vb is None:
            reference = baseline
        else:
            reference = np.asarray(reference_mean_vb, dtype=np.float64)
            if (
                reference.shape != observed.shape
                or np.any(~np.isfinite(reference))
                or np.any(reference < 0.0)
            ):
                raise ValueError(
                    "Birth proposal reference mean must be finite, "
                    "nonnegative, and observation-aligned."
                )
        whitening = 1.0 / np.sqrt(reference + 1.0)
        residual = (observed - reference) * whitening
        basis = self._birth_proposal_nuisance_basis_numpy(mask)
        if basis.shape[1] > 0:
            residual = residual - (residual @ basis) @ basis.T
        scores = np.empty(int(total.shape[0]), dtype=np.float64)
        chunk_size = self._birth_proposal_candidate_chunk_size(int(total.shape[-3]))
        for start in range(0, int(total.shape[0]), chunk_size):
            stop = min(start + chunk_size, int(total.shape[0]))
            candidate_mean = self.predict_mean_numpy(
                total[start:stop],
                uncollided[start:stop],
                features[start:stop],
                live_times_s_v,
            )
            templates = (candidate_mean - baseline[np.newaxis, ...]) * whitening
            if basis.shape[1] > 0:
                coefficients = np.einsum(
                    "gvb,bj->gvj",
                    templates,
                    basis,
                    optimize=True,
                )
                templates = templates - np.einsum(
                    "gvj,bj->gvb",
                    coefficients,
                    basis,
                    optimize=True,
                )
            correlation = np.einsum(
                "vb,gvb->g",
                residual,
                templates,
                optimize=True,
            )
            energy = np.einsum(
                "gvb,gvb->g",
                templates,
                templates,
                optimize=True,
            )
            scores[start:stop] = correlation - 0.5 * energy
        if scores.shape != (int(total.shape[0]),) or np.any(~np.isfinite(scores)):
            raise RuntimeError("Birth proposal score is not finite and aligned.")
        return np.asarray(scores, dtype=np.float64)

    def birth_proposal_log_scores_torch(
        self,
        observed_spectrum_vb: object,
        candidate_total_line_contributions_gvsl: object,
        candidate_uncollided_line_contributions_gvsl: object,
        candidate_transport_features_gvslf: object,
        live_times_s_v: object,
        *,
        target_line_mask_l: object,
        reference_mean_vb: object | None = None,
    ) -> object:
        """Return the Torch-equivalent proposal-only matched-filter scores."""
        import torch

        total = torch.as_tensor(candidate_total_line_contributions_gvsl)
        if total.dtype != torch.float64:
            raise TypeError("Birth proposal scoring requires torch.float64.")
        observed = torch.as_tensor(
            observed_spectrum_vb,
            device=total.device,
            dtype=torch.float64,
        )
        mask = torch.as_tensor(
            target_line_mask_l,
            device=total.device,
            dtype=torch.bool,
        )
        if (
            observed.shape != (int(total.shape[-3]), int(self._energy_axis_keV.size))
            or tuple(mask.shape) != (len(self._line_identity),)
            or bool(torch.any(~torch.isfinite(observed)))
            or bool(torch.any(observed < 0.0))
            or bool(torch.any(observed != torch.floor(observed)))
            or bool(torch.any(total[..., ~mask] != 0.0))
        ):
            raise ValueError("Torch birth proposal inputs are invalid.")
        uncollided = torch.as_tensor(
            candidate_uncollided_line_contributions_gvsl,
            device=total.device,
            dtype=torch.float64,
        )
        features = torch.as_tensor(
            candidate_transport_features_gvslf,
            device=total.device,
            dtype=torch.float64,
        )
        zero_total = torch.zeros(
            (1,) + tuple(total.shape[-3:]),
            device=total.device,
            dtype=torch.float64,
        )
        zero_features = torch.zeros(
            tuple(zero_total.shape) + (len(TRANSPORT_FEATURE_ORDER),),
            device=total.device,
            dtype=torch.float64,
        )
        baseline = self.predict_mean_torch(
            zero_total,
            zero_total,
            zero_features,
            live_times_s_v,
        )[0]
        if reference_mean_vb is None:
            reference = baseline
        else:
            reference = torch.as_tensor(
                reference_mean_vb,
                device=total.device,
                dtype=torch.float64,
            )
            if (
                tuple(reference.shape) != tuple(observed.shape)
                or bool(torch.any(~torch.isfinite(reference)))
                or bool(torch.any(reference < 0.0))
            ):
                raise ValueError("Torch birth proposal reference mean is invalid.")
        whitening = torch.rsqrt(reference + 1.0)
        residual = (observed - reference) * whitening
        basis = torch.as_tensor(
            np.array(
                self._birth_proposal_nuisance_basis_numpy(
                    mask.detach().cpu().numpy(),
                ),
                copy=True,
            ),
            device=total.device,
            dtype=torch.float64,
        )
        if int(basis.shape[1]) > 0:
            residual = residual - (residual @ basis) @ basis.T
        scores = torch.empty(
            int(total.shape[0]),
            device=total.device,
            dtype=torch.float64,
        )
        chunk_size = self._birth_proposal_candidate_chunk_size(int(total.shape[-3]))
        for start in range(0, int(total.shape[0]), chunk_size):
            stop = min(start + chunk_size, int(total.shape[0]))
            candidate_mean = self.predict_mean_torch(
                total[start:stop],
                uncollided[start:stop],
                features[start:stop],
                live_times_s_v,
            )
            templates = (candidate_mean - baseline.unsqueeze(0)) * whitening
            if int(basis.shape[1]) > 0:
                coefficients = torch.einsum(
                    "gvb,bj->gvj",
                    templates,
                    basis,
                )
                templates = templates - torch.einsum(
                    "gvj,bj->gvb",
                    coefficients,
                    basis,
                )
            correlation = torch.einsum(
                "vb,gvb->g",
                residual,
                templates,
            )
            energy = torch.einsum(
                "gvb,gvb->g",
                templates,
                templates,
            )
            scores[start:stop] = correlation - 0.5 * energy
        if tuple(scores.shape) != (int(total.shape[0]),) or bool(
            torch.any(~torch.isfinite(scores))
        ):
            raise RuntimeError("Torch birth proposal scores are invalid.")
        return scores

    def posterior_predictive_innovation_numpy(
        self,
        observed_spectrum_vb: NDArray[np.float64],
        total_line_contributions_nvsl: NDArray[np.float64],
        uncollided_line_contributions_nvsl: NDArray[np.float64],
        transport_features_nvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
        particle_weights_n: NDArray[np.float64],
        *,
        confidence: float,
    ) -> Mapping[str, float | int | bool | None]:
        """Return renewal-total and conditional-mark posterior diagnostics."""
        observed = np.asarray(observed_spectrum_vb, dtype=np.float64)
        weights = np.asarray(particle_weights_n, dtype=np.float64)
        if (
            self.physical_component_discrepancy is not None
            or not self.exact_physical_statistics_ready
        ):
            return self._discrepancy_innovation_numpy(
                observed,
                total_line_contributions_nvsl,
                uncollided_line_contributions_nvsl,
                transport_features_nvslf,
                live_times_s_v,
                weights,
                confidence=confidence,
            )
        means = self.predict_mean_numpy(
            total_line_contributions_nvsl,
            uncollided_line_contributions_nvsl,
            transport_features_nvslf,
            live_times_s_v,
        )
        if (
            weights.shape != (means.shape[0],)
            or np.any(~np.isfinite(weights))
            or np.any(weights < 0.0)
            or float(np.sum(weights)) <= 0.0
        ):
            raise ValueError("Posterior innovation weights are invalid.")
        normalized = weights / float(np.sum(weights))
        posterior_mean = np.einsum(
            "n,nvb->vb",
            normalized,
            means,
            optimize=True,
        )
        observed_total = np.sum(observed, axis=-1)
        predicted_total = np.sum(posterior_mean, axis=-1)
        live = np.asarray(live_times_s_v, dtype=np.float64)
        incident_rate = predicted_total / np.maximum(
            live - predicted_total * float(self.dead_time_tau_s),
            np.finfo(np.float64).tiny,
        )
        renewal_variance = (
            incident_rate
            * live
            / np.power(
                1.0 + incident_rate * float(self.dead_time_tau_s),
                3.0,
            )
        )
        total_z = (observed_total - predicted_total) / np.sqrt(
            np.maximum(renewal_variance, 1.0)
        )
        probabilities = np.divide(
            posterior_mean,
            predicted_total[:, np.newaxis],
            out=np.zeros_like(posterior_mean),
            where=predicted_total[:, np.newaxis] > 0.0,
        )
        expected_marks = observed_total[:, np.newaxis] * probabilities
        mark_pearson = float(
            np.sum(
                np.square(observed - expected_marks) / np.maximum(expected_marks, 1.0)
            )
        )
        degrees = int(np.sum(expected_marks >= 1.0) - observed.shape[0])
        mark_tail_probability = None
        mark_upper_tail_probability = None
        if degrees > 0:
            upper_tail = float(stats.chi2.sf(mark_pearson, degrees))
            lower_tail = float(stats.chi2.cdf(mark_pearson, degrees))
            mark_upper_tail_probability = upper_tail
            mark_tail_probability = min(
                1.0,
                2.0 * min(upper_tail, lower_tail),
            )
        threshold = float(stats.norm.ppf(0.5 + float(confidence) / 2.0))
        maximum_total_z = float(np.max(np.abs(total_z)))
        return {
            "renewal_total_max_abs_z": maximum_total_z,
            "renewal_total_within_confidence": maximum_total_z <= threshold,
            "conditional_mark_pearson": mark_pearson,
            "conditional_mark_degrees_of_freedom": degrees,
            "conditional_mark_tail_probability": mark_tail_probability,
            "conditional_mark_upper_tail_probability": (mark_upper_tail_probability),
            "confidence": float(confidence),
        }

    def _discrepancy_innovation_numpy(
        self,
        observed_spectrum_vb: NDArray[np.float64],
        total_line_contributions_nvsl: NDArray[np.float64],
        uncollided_line_contributions_nvsl: NDArray[np.float64],
        transport_features_nvslf: NDArray[np.float64],
        live_times_s_v: NDArray[np.float64],
        particle_weights_n: NDArray[np.float64],
        *,
        confidence: float,
    ) -> Mapping[str, float | int | bool | None]:
        """Return innovation using the same latent discrepancy as likelihood.

        Renewal variance integrates both the conditional renewal count noise
        and the station-shared rate-scale mixture.  Conditional-mark variance
        integrates the configured Dirichlet-multinomial dispersion and the
        mixture of particle/rate-node mark probabilities.  This prevents a
        calibrated likelihood from being judged by a contradictory Poisson /
        multinomial-only convergence gate.
        """
        observed = np.asarray(observed_spectrum_vb, dtype=np.float64)
        weights = np.asarray(particle_weights_n, dtype=np.float64)
        component_discrepancy = self.physical_component_discrepancy
        component_tree_marks = bool(
            component_discrepancy is not None
            and component_discrepancy.mark_latent_model
            == "component_dirichlet_tree_hierarchical"
        )
        if component_tree_marks:
            direct_mean, scatter_mean, background_mean = self._pre_dead_time_mean_numpy(
                total_line_contributions_nvsl,
                uncollided_line_contributions_nvsl,
                transport_features_nvslf,
                live_times_s_v,
                return_physical_components=True,
            )
            source_mean = direct_mean + scatter_mean
        else:
            source_mean, background_mean = self._pre_dead_time_mean_numpy(
                total_line_contributions_nvsl,
                uncollided_line_contributions_nvsl,
                transport_features_nvslf,
                live_times_s_v,
                return_components=True,
            )
        if (
            observed.shape != source_mean.shape[-2:]
            or weights.shape != source_mean.shape[:1]
            or np.any(~np.isfinite(weights))
            or np.any(weights < 0.0)
            or float(np.sum(weights)) <= 0.0
        ):
            raise ValueError("Posterior discrepancy innovation inputs are invalid.")
        normalized = weights / float(np.sum(weights))
        component_weights = (
            normalized[:, np.newaxis] * self._rate_scale_weights_j[np.newaxis, :]
        )
        pre_mean = (
            background_mean[:, np.newaxis, :, :]
            + source_mean[:, np.newaxis, :, :]
            * self._rate_scale_nodes_j[np.newaxis, :, np.newaxis, np.newaxis]
        )
        pre_total = np.sum(pre_mean, axis=-1)
        live = np.asarray(live_times_s_v, dtype=np.float64)
        dead_time_scale = 1.0 + pre_total / live[np.newaxis, np.newaxis, :] * float(
            self.dead_time_tau_s
        )
        component_total_mean = pre_total / dead_time_scale
        if (
            self.count_discrepancy_concentration is None
            and component_discrepancy is None
        ):
            component_total_variance = pre_total / np.power(
                dead_time_scale,
                3.0,
            )
        else:
            count_concentration = (
                self._component_count_concentration_numpy(
                    total_line_contributions_nvsl,
                    uncollided_line_contributions_nvsl,
                    transport_features_nvslf,
                )[:, np.newaxis, :]
                if component_discrepancy is not None
                else float(self.count_discrepancy_concentration)
            )
            component_total_variance = (
                component_total_mean
                + np.square(component_total_mean) / count_concentration
            )
            if component_discrepancy is not None:
                source_active = np.sum(source_mean, axis=-1) > 0.0
                renewal_variance = pre_total / np.power(dead_time_scale, 3.0)
                component_total_variance = np.where(
                    source_active[:, np.newaxis, :],
                    component_total_variance,
                    renewal_variance,
                )
        posterior_total_mean = np.einsum(
            "nj,njv->v",
            component_weights,
            component_total_mean,
            optimize=True,
        )
        posterior_total_variance = np.einsum(
            "nj,njv->v",
            component_weights,
            component_total_variance + np.square(component_total_mean),
            optimize=True,
        ) - np.square(posterior_total_mean)
        posterior_total_variance = np.maximum(
            posterior_total_variance,
            1.0,
        )
        observed_total = np.sum(observed, axis=-1)
        total_z = (observed_total - posterior_total_mean) / np.sqrt(
            posterior_total_variance
        )

        conditional_predictive: NDArray[np.int64] | None = None
        if component_tree_marks:
            probabilities = np.divide(
                pre_mean,
                pre_total[..., np.newaxis],
                out=np.zeros_like(pre_mean),
                where=pre_total[..., np.newaxis] > 0.0,
            )
            posterior_probabilities = np.einsum(
                "nj,njvb->vb",
                component_weights,
                probabilities,
                optimize=True,
            )
            expected_marks = observed_total[:, np.newaxis] * posterior_probabilities
            seed_hash = hashlib.sha256()
            seed_hash.update(self.contract_hash_sha256.encode("ascii"))
            seed_hash.update(np.ascontiguousarray(observed, dtype=np.float64).tobytes())
            diagnostic_seed = int.from_bytes(
                seed_hash.digest()[:8],
                byteorder="little",
                signed=False,
            )
            diagnostic_rng = np.random.Generator(np.random.Philox(diagnostic_seed))
            predictive_sample_count = 256
            selected_indices = diagnostic_rng.choice(
                normalized.size,
                size=predictive_sample_count,
                replace=True,
                p=normalized,
            )
            selected_direct = direct_mean[selected_indices]
            selected_scatter = scatter_mean[selected_indices]
            selected_background = background_mean[selected_indices]
            selected_pre_mean = selected_direct + selected_scatter + selected_background
            selected_total = np.sum(selected_pre_mean, axis=-1, keepdims=True)
            selected_probabilities = np.divide(
                selected_pre_mean,
                selected_total,
                out=np.zeros_like(selected_pre_mean),
                where=selected_total > 0.0,
            )
            zero_probability = np.sum(selected_probabilities, axis=-1) <= 0.0
            selected_probabilities[zero_probability, 0] = 1.0
            selected_component_means = np.stack(
                (selected_direct, selected_scatter, selected_background),
                axis=-2,
            )[:, np.newaxis, ...]
            selected_tree, selected_leaf = (
                self._component_tree_mark_concentrations_numpy(
                    np.asarray(total_line_contributions_nvsl, dtype=np.float64)[
                        selected_indices
                    ],
                    np.asarray(
                        uncollided_line_contributions_nvsl,
                        dtype=np.float64,
                    )[selected_indices],
                    selected_component_means,
                )
            )
            sampled_probabilities = self._sample_component_tree_probabilities_numpy(
                selected_probabilities,
                selected_tree[:, 0],
                selected_leaf[:, 0],
                rng=diagnostic_rng,
            )
            conditional_predictive = np.asarray(
                diagnostic_rng.multinomial(
                    np.broadcast_to(
                        observed_total[np.newaxis, :],
                        (predictive_sample_count, observed_total.size),
                    ).astype(np.int64),
                    sampled_probabilities,
                ),
                dtype=np.int64,
            )
            mark_variance = np.var(
                conditional_predictive.astype(np.float64),
                axis=0,
            )
        else:
            probabilities = np.divide(
                pre_mean,
                pre_total[..., np.newaxis],
                out=np.zeros_like(pre_mean),
                where=pre_total[..., np.newaxis] > 0.0,
            )
            posterior_probabilities = np.einsum(
                "nj,njvb->vb",
                component_weights,
                probabilities,
                optimize=True,
            )
            expected_marks = observed_total[:, np.newaxis] * posterior_probabilities
            node_source_total = np.sum(
                source_mean[:, np.newaxis, :, :]
                * self._rate_scale_nodes_j[np.newaxis, :, np.newaxis, np.newaxis],
                axis=-1,
            )
            source_fraction = np.divide(
                node_source_total,
                pre_total,
                out=np.zeros_like(pre_total),
                where=pre_total > 0.0,
            )
            base_concentration = self._base_mark_concentration_numpy(
                total_line_contributions_nvsl,
                uncollided_line_contributions_nvsl,
            )
            concentration = base_concentration[:, np.newaxis, :] / np.maximum(
                np.square(source_fraction),
                1.0e-12,
            )
            sample_size = observed_total[np.newaxis, np.newaxis, :]
            dispersion = np.where(
                source_fraction > 0.0,
                (sample_size + concentration) / (1.0 + concentration),
                1.0,
            )
            component_mark_variance = (
                observed_total[np.newaxis, np.newaxis, :, np.newaxis]
                * probabilities
                * (1.0 - probabilities)
                * dispersion[..., np.newaxis]
            )
            mark_variance = np.einsum(
                "nj,njvb->vb",
                component_weights,
                component_mark_variance
                + np.square(
                    observed_total[np.newaxis, np.newaxis, :, np.newaxis]
                    * probabilities
                ),
                optimize=True,
            ) - np.square(expected_marks)
        mark_variance = np.maximum(mark_variance, 1.0)
        mark_pearson = float(
            np.sum(np.square(observed - expected_marks) / mark_variance)
        )
        degrees = int(np.sum(expected_marks >= 1.0) - observed.shape[0])
        mark_tail_probability = None
        mark_upper_tail_probability = None
        if component_tree_marks:
            if conditional_predictive is None:  # pragma: no cover - guarded above
                raise RuntimeError("Component-tree diagnostic samples disappeared.")
            predictive_statistics = np.sum(
                np.square(
                    conditional_predictive.astype(np.float64)
                    - expected_marks[np.newaxis, :, :]
                )
                / mark_variance[np.newaxis, :, :],
                axis=(-2, -1),
            )
            predictive_sample_count = int(conditional_predictive.shape[0])
            upper_tail = (
                1.0 + float(np.sum(predictive_statistics >= mark_pearson))
            ) / float(predictive_sample_count + 1)
            lower_tail = (
                1.0 + float(np.sum(predictive_statistics <= mark_pearson))
            ) / float(predictive_sample_count + 1)
            mark_upper_tail_probability = upper_tail
            mark_tail_probability = min(
                1.0,
                2.0 * min(upper_tail, lower_tail),
            )
        elif degrees > 0:
            upper_tail = float(stats.chi2.sf(mark_pearson, degrees))
            lower_tail = float(stats.chi2.cdf(mark_pearson, degrees))
            mark_upper_tail_probability = upper_tail
            mark_tail_probability = min(
                1.0,
                2.0 * min(upper_tail, lower_tail),
            )
        threshold = float(stats.norm.ppf(0.5 + float(confidence) / 2.0))
        maximum_total_z = float(np.max(np.abs(total_z)))
        return {
            "renewal_total_max_abs_z": maximum_total_z,
            "renewal_total_within_confidence": maximum_total_z <= threshold,
            "conditional_mark_pearson": mark_pearson,
            "conditional_mark_degrees_of_freedom": degrees,
            "conditional_mark_tail_probability": mark_tail_probability,
            "conditional_mark_upper_tail_probability": (mark_upper_tail_probability),
            "confidence": float(confidence),
        }

    def manifest_payload(self) -> Mapping[str, object]:
        """Return immutable physics and validation provenance."""
        bin_width = float(self._energy_axis_keV[1] - self._energy_axis_keV[0])
        mark_model = (
            "component_background_source_dirichlet_tree_hierarchical"
            if self.physical_component_discrepancy is not None
            and self.physical_component_discrepancy.mark_latent_model
            == "component_dirichlet_tree_hierarchical"
            else "physical_component_fraction_dirichlet_multinomial"
            if self.physical_component_discrepancy is not None
            else "source_fraction_dirichlet_multinomial"
            if self.mark_concentration_source is not None
            else "finite_detector_corpus_dirichlet_multinomial"
        )
        physics_response = isinstance(
            self.additive_scatter_response,
            PhysicsOnlyNoncollidedTransportResponse,
        )
        payload: dict[str, object] = {
            "schema_version": FULL_SPECTRUM_MODEL_SCHEMA_VERSION,
            "model": "geometry_conditioned_full_spectrum",
            "contract_hash_sha256": self.contract_hash_sha256,
            "shield_pose_contract_id": SHIELD_POSE_CONTRACT_ID,
            "shield_pose_contract_sha256": SHIELD_POSE_CONTRACT_SHA256,
            "obstacle_material_contract_id": OBSTACLE_MATERIAL_CONTRACT_ID,
            "obstacle_material_contract_sha256": (OBSTACLE_MATERIAL_CONTRACT_SHA256),
            "transport_physics_table_contract_id": (
                TRANSPORT_PHYSICS_TABLE_CONTRACT_ID
            ),
            "transport_physics_table_contract_sha256": (
                TRANSPORT_PHYSICS_TABLE_CONTRACT_SHA256
            ),
            "runtime_ready": self.runtime_ready,
            "production_ready": self.production_ready,
            "energy_bin_count": int(self._energy_axis_keV.size),
            "energy_min_keV": float(self._energy_axis_keV[0]),
            "energy_max_keV": float(self._energy_axis_keV[-1]),
            "bin_width_keV": bin_width,
            "transport_feature_order": list(TRANSPORT_FEATURE_ORDER),
            "additive_noncollided_transport_response": (
                None
                if self.additive_scatter_response is None
                else self.additive_scatter_response.to_payload()
            ),
            "line_identity": [dict(item) for item in self._line_identity],
            "source_rate_semantics": ("pre_dead_time_detector_pulse_rate_at_1m"),
            "source_rate_green_normalization": (
                "catalog_branching_weighted_absolute_detection_efficiency_at_1m_v1"
            ),
            "direct_partition": "minimum_of_total_and_uncollided",
            "scatter_partition": "total_minus_direct",
            "scatter_shape": (
                DETECTOR_CONE_SCATTER_RESPONSE_ID
                if physics_response
                else "klein_nishina_optical_depth_orders"
            ),
            "higher_order_scatter_mean": (
                "excluded_positive_nuisance_owned_by_likelihood"
                if physics_response
                else "legacy_explicit_optical_depth_orders"
            ),
            "detector_cone_scatter_response": (self._detector_cone_scatter_contract),
            "detector_response_sampling": (
                "physical_component_gamma_poisson_and_dirichlet_marking"
                if self.physical_component_discrepancy is not None
                else "multinomial_marking_with_station_shared_gamma_poisson_"
                "recorded_total"
                if self.count_discrepancy_concentration is not None
                else DETECTOR_GREEN_SAMPLING_MODE
            ),
            "detector_green_operator_contract_sha256": (
                self.detector_green_operator.contract_hash_sha256
            ),
            "detector_green_operator_binary_sha256": (
                self.detector_green_operator.binary_sha256
            ),
            "detector_green_operator_id": DETECTOR_GREEN_OPERATOR_ID,
            "detector_green_boundary_state": (
                "normalized_impact_parameter_at_detector_housing_entry_v1"
            ),
            "detector_green_phase_conditioning": (
                "transport_resolved_direct_impact_and_detector_cone_"
                "scatter_joint_state_v3"
            ),
            "detector_green_finite_mc_uncertainty": (
                "pulse_plus_no_pulse_categorical_covariance_v1"
            ),
            "dead_time_model": (
                "nonparalyzable_mean_then_component_gamma_poisson_total"
                if self.physical_component_discrepancy is not None
                else "nonparalyzable_mean_then_gamma_poisson_recorded_total"
                if self.count_discrepancy_concentration is not None
                else "nonparalyzable_event_time_renewal_total"
            ),
            "dead_time_tau_s": float(self.dead_time_tau_s),
            "dead_time_application_count": 1,
            "background_rate_cps": float(self.background_rate_cps),
            "background_model": ("native_geant4_background_shape_v1_bin_centres"),
            "background_semantics": ("independent_pre_dead_time_pulse_rate_added_once"),
            "rate_scale_mixture": {
                "scope": "station_shared_source_only",
                "nodes": self._rate_scale_nodes_j.tolist(),
                "weights": self._rate_scale_weights_j.tolist(),
                "weighted_mean": float(
                    np.sum(self._rate_scale_nodes_j * self._rate_scale_weights_j)
                ),
            },
            "mark_model": mark_model,
            "mark_concentration_source": (
                None
                if self.mark_concentration_source is None
                else float(self.mark_concentration_source)
            ),
            "validation": (
                None
                if self.validation_manifest is None
                else _thaw_json_value(self.validation_manifest)
            ),
            "validation_manifest_sha256": self._validation_manifest_sha256,
        }
        if not physics_response:
            payload["maximum_scatter_order"] = int(self.maximum_scatter_order)
        correction = self.low_rank_spectral_mean_correction
        if self.count_discrepancy_concentration is not None:
            payload["count_discrepancy_concentration"] = float(
                self.count_discrepancy_concentration
            )
            payload["count_discrepancy_scope"] = str(self.count_discrepancy_scope)
        if self.mark_concentration_multi_isotope is not None:
            payload["mark_concentration_multi_isotope"] = float(
                self.mark_concentration_multi_isotope
            )
        if self.physical_component_discrepancy is not None:
            payload["physical_component_discrepancy"] = (
                self.physical_component_discrepancy.to_payload()
            )
        if correction is not None:
            payload["low_rank_spectral_mean_correction"] = correction.to_payload()
        return payload


def with_catalog_independent_production_approval(
    model: GeometryConditionedSpectralModel,
    *,
    approved_source: GeometryConditionedSpectralModel,
) -> GeometryConditionedSpectralModel:
    """Attach transferable approval without changing the target physics hash.

    The source must carry literal schema-v6 all-64 evidence.  Transfer is
    permitted only when the target has the same isotope-independent detector,
    transport, background, dead-time, and uncertainty contract.  The target's
    catalog lines remain application inputs and must lie inside the validated
    continuous detector-energy domain.
    """
    model.require_runtime_ready()
    if model.production_ready:
        return model
    approved_source.require_production_ready()
    source_validation = approved_source.validation_manifest
    if (
        not isinstance(source_validation, Mapping)
        or source_validation.get("schema_version") != 6
    ):
        raise RuntimeError(
            "Catalog-independent approval must originate from literal schema-v6 "
            "all-64 application evidence; chained approval transfer is forbidden."
        )
    if (
        approved_source.catalog_independent_contract_hash_sha256
        != model.catalog_independent_contract_hash_sha256
    ):
        raise RuntimeError(
            "Full-spectrum catalog-independent detector/transport algorithm "
            "differs from the independently approved source model."
        )
    source_payload = _thaw_json_value(source_validation)
    source_validation_sha256 = _canonical_json_sha256(source_payload)
    validated_isotopes = tuple(
        sorted({str(row["isotope"]) for row in approved_source.line_identity})
    )
    transferred_validation = dict(source_payload)
    transferred_validation.update(
        {
            "schema_version": TRANSFERRED_VALIDATION_SCHEMA_VERSION,
            "approval_scope": CATALOG_INDEPENDENT_APPROVAL_SCOPE,
            "approved_catalog_independent_contract_sha256": (
                model.catalog_independent_contract_hash_sha256
            ),
            "application_validation_isotopes": list(validated_isotopes),
            "source_validation_manifest_sha256": source_validation_sha256,
        }
    )
    target_isotopes = tuple(
        sorted({str(row["isotope"]) for row in model.line_identity})
    )
    approved = GeometryConditionedSpectralModel.physics_only_native(
        target_isotopes,
        dead_time_tau_s=float(model.dead_time_tau_s),
        background_rate_cps=float(model.background_rate_cps),
        detector_green_operator=model.detector_green_operator,
        validation_manifest=transferred_validation,
    )
    if approved.contract_hash_sha256 != model.contract_hash_sha256:
        raise RuntimeError(
            "Approval attachment changed the target full-spectrum physics hash."
        )
    approved.require_production_ready()
    return approved


def geometry_conditioned_model_from_runtime_config(
    runtime_config: Mapping[str, object],
    *,
    run_root: str | Path | None = None,
) -> GeometryConditionedSpectralModel:
    """Reconstruct and verify the estimator-visible spectrum contract."""
    if not isinstance(runtime_config, Mapping):
        raise TypeError("Resolved runtime configuration must be a mapping.")
    from spectrum.isotope_profiles import resolve_profile_model_runtime_config

    runtime_config = resolve_profile_model_runtime_config(
        runtime_config,
        run_root=run_root,
    )
    inline_payload = runtime_config.get("full_spectrum_generative_model")
    path_value = runtime_config.get("full_spectrum_generative_model_path")
    if inline_payload is not None and path_value is not None:
        raise ValueError(
            "Full-spectrum runtime must select exactly one inline or "
            "file-backed generative model."
        )
    if inline_payload is None and path_value is None:
        raise ValueError(
            "Resolved runtime requires one full-spectrum generative model."
        )
    if path_value is None:
        if not isinstance(inline_payload, Mapping):
            raise ValueError("full_spectrum_generative_model must be a mapping.")
        if "full_spectrum_generative_model_file_sha256" in runtime_config:
            raise ValueError(
                "Inline full-spectrum models cannot declare a file digest."
            )
        payload = inline_payload
    else:
        declared_file_hash = runtime_config.get(
            "full_spectrum_generative_model_file_sha256"
        )
        if not _is_sha256(declared_file_hash):
            raise ValueError(
                "File-backed full-spectrum models require an exact SHA-256."
            )
        resolved_path = resolve_file_backed_model_asset(
            path_value,
            field_name="full_spectrum_generative_model_path",
            run_root=run_root,
        )
        raw_bytes = resolved_path.read_bytes()
        if hashlib.sha256(raw_bytes).hexdigest() != declared_file_hash:
            raise ValueError(
                "Full-spectrum model file SHA-256 does not match the configured digest."
            )
        try:
            decoded_payload = json.loads(
                raw_bytes,
                object_pairs_hook=_strict_json_object,
                parse_constant=_reject_nonfinite_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                "Full-spectrum model asset must be canonical JSON."
            ) from exc
        if not isinstance(decoded_payload, Mapping):
            raise ValueError(
                "Full-spectrum model asset must contain one manifest mapping."
            )
        payload = decoded_payload
    operator_manifest_value = runtime_config.get("detector_green_operator_manifest")
    if (
        not isinstance(operator_manifest_value, str)
        or not operator_manifest_value.strip()
    ):
        raise ValueError(
            "Resolved runtime requires an explicit detector Green operator "
            "manifest path."
        )
    operator_manifest_path = resolve_file_backed_model_asset(
        operator_manifest_value,
        field_name="detector_green_operator_manifest",
        run_root=run_root,
    )
    detector_green_operator = DetectorGreenOperator.from_artifact(
        operator_manifest_path
    )
    detector_green_operator.require_runtime_ready()
    model = GeometryConditionedSpectralModel.from_manifest_payload(
        payload,
        detector_green_operator=detector_green_operator,
    )
    declared_hash = runtime_config.get("full_spectrum_contract_hash_sha256")
    if declared_hash != model.contract_hash_sha256:
        raise ValueError(
            "Resolved runtime full-spectrum hash does not match its model."
        )
    expected_numeric = {
        "energy_min_keV": float(model.energy_axis_keV[0]),
        "energy_max_keV": float(model.energy_axis_keV[-1]),
        "bin_width_keV": float(model.energy_axis_keV[1] - model.energy_axis_keV[0]),
        "energy_bin_count": int(model.energy_axis_keV.size),
        "background_cps": float(model.background_rate_cps),
        "dead_time_tau_s": float(model.dead_time_tau_s),
    }
    for key, expected in expected_numeric.items():
        value = runtime_config.get(key)
        if key == "energy_bin_count":
            valid = (
                not isinstance(value, bool)
                and isinstance(value, int)
                and value == expected
            )
        else:
            valid = (
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and np.isfinite(float(value))
                and np.isclose(
                    float(value),
                    expected,
                    rtol=0.0,
                    atol=1.0e-15,
                )
            )
        if not valid:
            raise ValueError(
                f"Resolved runtime {key} disagrees with the full-spectrum model."
            )
    if runtime_config.get("source_rate_model") != "detector_cps_1m":
        raise ValueError(
            "Full-spectrum runtime requires source_rate_model=detector_cps_1m."
        )
    return model
