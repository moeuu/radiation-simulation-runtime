"""Isotope-independent full-detector Green-operator artifacts.

The operator maps a photon state at the detector-housing boundary to an
absolute registered-pulse sub-probability.  Its stored decomposition contains
both the pulse-detection probability and the conditional pulse-height law.
It does not own nuclide identities or decay-line probabilities.  Those remain
explicit catalog inputs to the transport model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Final

import numpy as np
from numpy.typing import NDArray

from spectrum.library import Nuclide, default_library


DETECTOR_GREEN_OPERATOR_SCHEMA_VERSION: Final = 3
DETECTOR_GREEN_OPERATOR_ID: Final = (
    "isotope_independent_full_detector_green_operator_v3"
)
DETECTOR_GREEN_BOUNDARY_STATE: Final = (
    "normalized_impact_parameter_at_detector_housing_entry_v1"
)
DETECTOR_GREEN_CONDITIONING: Final = (
    "registered_pulse_subprobability_given_housing_incident_gamma_v1"
)
DETECTOR_GREEN_SAMPLING_MODE: Final = (
    "independent_green_mark_then_same_history_coincidence_sum_"
    "nonparalyzable_v1"
)
DETECTOR_GREEN_COINCIDENCE_SEMANTICS: Final = (
    "sample_each_housing_incident_gamma_then_sum_registered_energy_"
    "within_same_history_branch_and_detector_window_v1"
)
DETECTOR_GREEN_INTERPOLATION: Final = (
    "raw_bin_anchor_energy_scaled_pulse_height_linear_probability_v2"
)
DETECTOR_GREEN_BINARY_MAGIC: Final = b"RSGKV3\x00\x00"
DETECTOR_GREEN_BINARY_BASENAME: Final = "operator.bin"
DETECTOR_GREEN_MANIFEST_BASENAME: Final = "manifest.json"
DETECTOR_GREEN_BINARY_HEADER = struct.Struct("<8sIIII4d")
DETECTOR_GREEN_INPUT_DOMAIN_KEV: Final = (0.0, 1700.0)
DETECTOR_GREEN_MINIMUM_EFFECTIVE_HISTORIES: Final = 2.0
DETECTOR_GREEN_MINIMUM_CONSTRUCTION_HISTORIES: Final = 100_000


def canonical_json_bytes(payload: object) -> bytes:
    """Return strict canonical JSON bytes with one trailing newline."""
    return (
        json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    """Return the lowercase SHA-256 digest of bytes."""
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    """Return whether a value is one lowercase SHA-256 digest."""
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_array(
    value: object,
    *,
    field_name: str,
    ndim: int,
) -> NDArray[np.float64]:
    """Return one finite float64 array with the requested rank."""
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc
    if result.ndim != ndim or np.any(~np.isfinite(result)):
        raise ValueError(f"{field_name} must be a finite rank-{ndim} array.")
    return np.ascontiguousarray(result)


def _require_exact_keys(
    payload: Mapping[str, object],
    expected: set[str],
    *,
    label: str,
) -> None:
    """Reject missing, unknown, and ignored artifact fields."""
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        unknown = sorted(set(payload) - expected)
        raise ValueError(
            f"{label} schema is incompatible; missing={missing}, unknown={unknown}."
        )


def _validate_construction_payload(payload: object) -> dict[str, object]:
    """Validate isotope-free native construction provenance."""
    if not isinstance(payload, Mapping):
        raise TypeError("Detector Green construction must be an object.")
    expected = {
        "method",
        "raw_corpus_sha256",
        "native_executable_sha256",
        "native_execution_environment_sha256",
        "detector_implementation_bundle_sha256",
        "detector_model_sha256",
        "geant4_physics_contract_sha256",
        "energy_resolution_contract_sha256",
        "construction_seed",
        "histories_per_energy",
        "energy_node_design",
        "phase_strata",
        "detector_target_radius_m",
        "completed",
    }
    _require_exact_keys(payload, expected, label="Detector Green construction")
    if (
        payload.get("method") != "native_geant4_monoenergetic_full_detector"
        or payload.get("energy_node_design")
        != "catalog_independent_deterministic_continuous_domain_v1"
        or payload.get("completed") is not True
    ):
        raise ValueError("Detector Green construction method is incompatible.")
    for field_name in (
        "raw_corpus_sha256",
        "native_executable_sha256",
        "native_execution_environment_sha256",
        "detector_implementation_bundle_sha256",
        "detector_model_sha256",
        "geant4_physics_contract_sha256",
        "energy_resolution_contract_sha256",
    ):
        if not _is_sha256(payload.get(field_name)):
            raise ValueError(f"Detector Green construction {field_name} is invalid.")
    for field_name in (
        "construction_seed",
        "histories_per_energy",
        "phase_strata",
    ):
        value = payload.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"Detector Green construction {field_name} must be positive."
            )
    histories_per_energy = int(payload["histories_per_energy"])
    phase_strata = int(payload["phase_strata"])
    radius = payload.get("detector_target_radius_m")
    if (
        histories_per_energy < DETECTOR_GREEN_MINIMUM_CONSTRUCTION_HISTORIES
        or histories_per_energy % phase_strata != 0
        or isinstance(radius, bool)
        or not isinstance(radius, (int, float))
        or not math.isfinite(float(radius))
        or float(radius) <= 0.0
    ):
        raise ValueError(
            "Detector Green construction histories, phase strata, or "
            "detector radius are invalid."
        )
    return json.loads(json.dumps(dict(payload), allow_nan=False))


def _validate_manifest_payload(payload: object) -> dict[str, object]:
    """Validate a strict Green-operator manifest without reading its binary."""
    if not isinstance(payload, Mapping):
        raise TypeError("Detector Green manifest must be an object.")
    expected = {
        "schema_version",
        "operator",
        "contract_hash_sha256",
        "input_energy_domain_keV",
        "output_energy_axis",
        "boundary_state",
        "conditioning",
        "interpolation",
        "binary",
        "construction",
    }
    _require_exact_keys(payload, expected, label="Detector Green manifest")
    if (
        payload.get("schema_version") != DETECTOR_GREEN_OPERATOR_SCHEMA_VERSION
        or payload.get("operator") != DETECTOR_GREEN_OPERATOR_ID
        or payload.get("boundary_state") != DETECTOR_GREEN_BOUNDARY_STATE
        or payload.get("conditioning") != DETECTOR_GREEN_CONDITIONING
        or payload.get("interpolation") != DETECTOR_GREEN_INTERPOLATION
        or payload.get("input_energy_domain_keV")
        != list(DETECTOR_GREEN_INPUT_DOMAIN_KEV)
        or not _is_sha256(payload.get("contract_hash_sha256"))
    ):
        raise ValueError("Detector Green manifest contract is incompatible.")
    output_axis = payload.get("output_energy_axis")
    if not isinstance(output_axis, Mapping):
        raise TypeError("Detector Green output axis must be an object.")
    _require_exact_keys(
        output_axis,
        {"minimum_keV", "bin_width_keV", "bin_count"},
        label="Detector Green output axis",
    )
    minimum = output_axis.get("minimum_keV")
    width = output_axis.get("bin_width_keV")
    count = output_axis.get("bin_count")
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, (int, float))
        or not math.isfinite(float(minimum))
        or float(minimum) != 0.0
        or isinstance(width, bool)
        or not isinstance(width, (int, float))
        or not math.isfinite(float(width))
        or float(width) <= 0.0
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 2
    ):
        raise ValueError("Detector Green output energy axis is invalid.")
    binary = payload.get("binary")
    if not isinstance(binary, Mapping):
        raise TypeError("Detector Green binary descriptor must be an object.")
    _require_exact_keys(
        binary,
        {
            "basename",
            "sha256",
            "probability_dtype",
            "layout",
            "energy_node_count",
            "impact_bin_count",
            "output_bin_count",
        },
        label="Detector Green binary descriptor",
    )
    if (
        binary.get("basename") != DETECTOR_GREEN_BINARY_BASENAME
        or not _is_sha256(binary.get("sha256"))
        or binary.get("probability_dtype") != "float32_le"
        or binary.get("layout") != "energy_node_impact_bin_output_bin_c_order"
        or any(
            isinstance(binary.get(key), bool)
            or not isinstance(binary.get(key), int)
            or int(binary[key]) <= 0
            for key in (
                "energy_node_count",
                "impact_bin_count",
                "output_bin_count",
            )
        )
        or int(binary["output_bin_count"]) != int(count)
    ):
        raise ValueError("Detector Green binary descriptor is invalid.")
    construction = _validate_construction_payload(payload.get("construction"))
    result = json.loads(json.dumps(dict(payload), allow_nan=False))
    result["construction"] = construction
    return result


@dataclass(frozen=True)
class DetectorGreenOperator:
    """Represent an isotope-independent housing-to-pulse Green operator."""

    energy_nodes_keV: NDArray[np.float64]
    impact_parameter_edges_fraction: NDArray[np.float64]
    conditional_response_ncb: NDArray[np.float64]
    effective_histories_nc: NDArray[np.float64]
    pulse_detection_probability_nc: NDArray[np.float64]
    output_energy_min_keV: float = 0.0
    output_bin_width_keV: float = 2.0
    construction: Mapping[str, object] | None = None
    binary_sha256: str | None = None
    contract_hash_sha256: str | None = None

    def __post_init__(self) -> None:
        """Validate, canonically quantize, and freeze every operator array."""
        nodes = _finite_array(
            self.energy_nodes_keV,
            field_name="energy_nodes_keV",
            ndim=1,
        )
        edges = _finite_array(
            self.impact_parameter_edges_fraction,
            field_name="impact_parameter_edges_fraction",
            ndim=1,
        )
        response = _finite_array(
            self.conditional_response_ncb,
            field_name="conditional_response_ncb",
            ndim=3,
        )
        histories = _finite_array(
            self.effective_histories_nc,
            field_name="effective_histories_nc",
            ndim=2,
        )
        detection = _finite_array(
            self.pulse_detection_probability_nc,
            field_name="pulse_detection_probability_nc",
            ndim=2,
        )
        output_minimum = float(self.output_energy_min_keV)
        output_width = float(self.output_bin_width_keV)
        node_count, impact_count, output_count = response.shape
        if (
            nodes.shape != (node_count,)
            or edges.shape != (impact_count + 1,)
            or histories.shape != (node_count, impact_count)
            or detection.shape != histories.shape
            or node_count < 2
            or impact_count < 1
            or output_count < 2
            or np.any(np.diff(nodes) <= 0.0)
            or float(nodes[0]) != DETECTOR_GREEN_INPUT_DOMAIN_KEV[0]
            or float(nodes[-1]) != DETECTOR_GREEN_INPUT_DOMAIN_KEV[1]
            or np.any(np.diff(edges) <= 0.0)
            or float(edges[0]) != 0.0
            or float(edges[-1]) != 1.0
            or np.any(response < 0.0)
            or np.any(histories < DETECTOR_GREEN_MINIMUM_EFFECTIVE_HISTORIES)
            or np.any(detection < 0.0)
            or np.any(detection > 1.0)
            or not math.isfinite(output_minimum)
            or output_minimum != 0.0
            or not math.isfinite(output_width)
            or output_width <= 0.0
        ):
            raise ValueError("Detector Green operator arrays are incompatible.")
        column_sums = np.sum(response, axis=-1)
        if not np.allclose(column_sums, 1.0, rtol=1.0e-7, atol=1.0e-8):
            raise ValueError(
                "Detector Green conditional response must preserve pulse counts."
            )
        # The native sampler consumes float32 probabilities.  Quantize before
        # hashing so Python reconstruction and Geant4 sample the same operator.
        # Do not renormalize an already valid column here: doing so again after
        # float32 readback can move probabilities by one ULP and would make
        # artifact serialization non-idempotent.  Both interpolation paths and
        # the native CDF normalize their working columns explicitly.
        response = np.asarray(response, dtype="<f4").astype(np.float64)
        construction = (
            None
            if self.construction is None
            else _validate_construction_payload(self.construction)
        )
        if construction is not None and int(construction["phase_strata"]) != (
            impact_count
        ):
            raise ValueError(
                "Detector Green construction phase count disagrees with its axis."
            )
        if self.binary_sha256 is not None and not _is_sha256(self.binary_sha256):
            raise ValueError("Detector Green binary_sha256 is invalid.")
        computed_hash = self._contract_hash(
            nodes=nodes,
            edges=edges,
            response=response,
            histories=histories,
            detection=detection,
            construction=construction,
        )
        if (
            self.contract_hash_sha256 is not None
            and self.contract_hash_sha256 != computed_hash
        ):
            raise ValueError("Detector Green contract hash is stale.")
        for array in (nodes, edges, response, histories, detection):
            array.setflags(write=False)
        object.__setattr__(self, "energy_nodes_keV", nodes)
        object.__setattr__(self, "impact_parameter_edges_fraction", edges)
        object.__setattr__(self, "conditional_response_ncb", response)
        object.__setattr__(self, "effective_histories_nc", histories)
        object.__setattr__(self, "pulse_detection_probability_nc", detection)
        object.__setattr__(self, "output_energy_min_keV", output_minimum)
        object.__setattr__(self, "output_bin_width_keV", output_width)
        object.__setattr__(self, "construction", construction)
        object.__setattr__(self, "contract_hash_sha256", computed_hash)

    @property
    def input_energy_domain_keV(self) -> tuple[float, float]:
        """Return the closed incident-energy domain supported by the operator."""
        return (
            float(self.energy_nodes_keV[0]),
            float(self.energy_nodes_keV[-1]),
        )

    @property
    def detector_target_radius_m(self) -> float:
        """Return the housing-boundary radius used by the phase coordinate."""
        if self.construction is None:
            raise RuntimeError("Detector Green operator has no construction geometry.")
        return float(self.construction["detector_target_radius_m"])

    @property
    def output_bin_count(self) -> int:
        """Return the number of observed pulse-height bins."""
        return int(self.conditional_response_ncb.shape[-1])

    @property
    def runtime_ready(self) -> bool:
        """Return whether native immutable construction provenance is complete."""
        return bool(
            self.construction is not None
            and self.binary_sha256 is not None
            and _is_sha256(self.contract_hash_sha256)
        )

    def require_runtime_ready(self) -> None:
        """Reject an in-memory or test-only operator in a runtime model."""
        if not self.runtime_ready:
            raise RuntimeError(
                "Detector Green operator lacks native construction provenance."
            )

    def validate_catalog_profile(
        self,
        isotopes: Sequence[str],
        *,
        library: Mapping[str, Nuclide] | None = None,
        primary_emission_model: str = "independent_gamma_lines",
    ) -> None:
        """Validate known catalog lines without granting application approval."""
        if primary_emission_model != "independent_gamma_lines":
            raise ValueError(
                "The detector Green operator supports independent catalog "
                "gamma lines only; prompt-cascade transport requires a "
                "separate approved contract."
            )
        catalog = default_library() if library is None else library
        names = tuple(str(value) for value in isotopes)
        if not names or len(set(names)) != len(names):
            raise ValueError("Detector Green profile isotopes must be unique.")
        lower, upper = self.input_energy_domain_keV
        for isotope in names:
            nuclide = catalog.get(isotope)
            if nuclide is None:
                raise KeyError(f"Catalog profile contains unknown isotope {isotope!r}.")
            for line in nuclide.lines:
                energy = float(line.energy_keV)
                if energy < lower or energy > upper:
                    raise ValueError(
                        f"Catalog line {isotope} {energy:g} keV is outside the "
                        f"detector Green domain [{lower:g}, {upper:g}] keV."
                    )

    def phase_response_for_axis(
        self,
        incident_energy_axis_keV: Sequence[float],
        *,
        batch_size: int = 64,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Interpolate phase-resolved responses and effective concentrations."""
        targets = _finite_array(
            incident_energy_axis_keV,
            field_name="incident_energy_axis_keV",
            ndim=1,
        )
        if (
            targets.size == 0
            or np.any(np.diff(targets) <= 0.0)
            or float(targets[0]) < float(self.energy_nodes_keV[0])
            or float(targets[-1]) > float(self.energy_nodes_keV[-1])
            or isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError(
                "Incident response axis must be increasing, in-domain, and "
                "use a positive batch size."
            )
        output = np.empty(
            (
                self.conditional_response_ncb.shape[1],
                self.output_bin_count,
                targets.size,
            ),
            dtype=np.float64,
        )
        concentration = np.empty(
            (self.conditional_response_ncb.shape[1], targets.size),
            dtype=np.float64,
        )
        for start in range(0, targets.size, batch_size):
            stop = min(start + batch_size, targets.size)
            block_response, block_concentration = self._interpolate_block(
                targets[start:stop]
            )
            output[:, :, start:stop] = np.moveaxis(
                block_response,
                0,
                -1,
            )
            concentration[:, start:stop] = block_concentration.T
        output = np.maximum(output, 0.0)
        output /= np.maximum(
            np.sum(output, axis=1, keepdims=True),
            np.finfo(np.float64).tiny,
        )
        return output, concentration

    def marginal_response_for_axis(
        self,
        incident_energy_axis_keV: Sequence[float],
        *,
        phase_weights: Sequence[float] | None = None,
        batch_size: int = 64,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return a phase-marginal response and effective column concentration."""
        phase, concentration = self.phase_response_for_axis(
            incident_energy_axis_keV,
            batch_size=batch_size,
        )
        if phase_weights is None:
            edges = self.impact_parameter_edges_fraction
            weights = np.diff(np.square(edges))
        else:
            weights = _finite_array(
                phase_weights,
                field_name="phase_weights",
                ndim=1,
            )
        if (
            weights.shape != (phase.shape[0],)
            or np.any(weights < 0.0)
            or float(np.sum(weights)) <= 0.0
        ):
            raise ValueError("Detector Green phase weights are invalid.")
        weights = weights / np.sum(weights)
        # A fixed-axis NumPy reduction keeps the serialized model contract
        # independent of whichever BLAS kernel happens to back ``einsum``.
        response = np.sum(
            weights[:, np.newaxis, np.newaxis] * phase,
            axis=0,
            dtype=np.float64,
        )
        component_trace = (
            np.square(weights)[:, np.newaxis]
            * (1.0 - np.sum(np.square(phase), axis=1))
            / (concentration + 1.0)
        )
        trace = np.sum(component_trace, axis=0)
        numerator = 1.0 - np.sum(np.square(response), axis=0)
        effective = (
            np.divide(
                numerator,
                trace,
                out=np.full_like(numerator, 1.0e15),
                where=trace > np.finfo(np.float64).tiny,
            )
            - 1.0
        )
        effective = np.maximum(
            effective,
            DETECTOR_GREEN_MINIMUM_EFFECTIVE_HISTORIES,
        )
        response /= np.maximum(
            np.sum(response, axis=0, keepdims=True),
            np.finfo(np.float64).tiny,
        )
        return response, effective

    def phase_absolute_response_for_axis(
        self,
        incident_energy_axis_keV: Sequence[float],
        *,
        batch_size: int = 64,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return phase-resolved absolute pulse sub-probabilities.

        The response shape is ``(phase, observed_bin, energy)``.  A column sum
        is the probability that one photon crossing the detector housing
        produces a registered pulse; the omitted mass is the no-pulse outcome.
        Concentrations cover observed bins and that no-pulse outcome jointly.
        """
        targets = _finite_array(
            incident_energy_axis_keV,
            field_name="incident_energy_axis_keV",
            ndim=1,
        )
        if (
            targets.size == 0
            or np.any(np.diff(targets) <= 0.0)
            or float(targets[0]) < float(self.energy_nodes_keV[0])
            or float(targets[-1]) > float(self.energy_nodes_keV[-1])
            or isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError(
                "Incident response axis must be increasing, in-domain, and "
                "use a positive batch size."
            )
        output = np.empty(
            (
                self.conditional_response_ncb.shape[1],
                self.output_bin_count,
                targets.size,
            ),
            dtype=np.float64,
        )
        concentration = np.empty(
            (self.conditional_response_ncb.shape[1], targets.size),
            dtype=np.float64,
        )
        for start in range(0, targets.size, batch_size):
            stop = min(start + batch_size, targets.size)
            block_response, block_concentration = self._interpolate_absolute_block(
                targets[start:stop]
            )
            output[:, :, start:stop] = np.moveaxis(
                block_response,
                0,
                -1,
            )
            concentration[:, start:stop] = block_concentration.T
        if np.any(output < 0.0) or np.any(np.sum(output, axis=1) > 1.0 + 1.0e-12):
            raise RuntimeError(
                "Interpolated detector Green sub-probability is invalid."
            )
        return np.maximum(output, 0.0), concentration

    def absolute_response_for_energy_phase_pairs(
        self,
        incident_energies_keV: Sequence[float],
        impact_phase_indices: Sequence[int],
        *,
        batch_size: int = 256,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return absolute responses for aligned energy/impact pairs.

        Unlike :meth:`phase_absolute_response_for_axis`, this method does not
        materialize every impact phase for every requested energy.  It is the
        batched construction primitive for a joint transport state in which
        Compton energy loss and detector impact phase are correlated.
        """
        targets = _finite_array(
            incident_energies_keV,
            field_name="incident_energies_keV",
            ndim=1,
        )
        phases = np.asarray(impact_phase_indices)
        phase_count = int(self.conditional_response_ncb.shape[1])
        if (
            targets.size == 0
            or phases.ndim != 1
            or phases.shape != targets.shape
            or phases.dtype.kind not in "iu"
            or np.any(phases < 0)
            or np.any(phases >= phase_count)
            or float(np.min(targets)) < float(self.energy_nodes_keV[0])
            or float(np.max(targets)) > float(self.energy_nodes_keV[-1])
            or isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError(
                "Energy/impact response pairs must be aligned, in-domain, "
                "and use valid phase indices and a positive batch size."
            )
        phases = phases.astype(np.int64, copy=False)
        output = np.empty((targets.size, self.output_bin_count), dtype=np.float64)
        concentration = np.empty(targets.size, dtype=np.float64)
        for start in range(0, targets.size, batch_size):
            stop = min(start + batch_size, targets.size)
            block_response, block_concentration = (
                self._interpolate_absolute_selected_phase_block(
                    targets[start:stop],
                    phases[start:stop],
                )
            )
            output[start:stop] = block_response
            concentration[start:stop] = block_concentration
        if np.any(output < 0.0) or np.any(np.sum(output, axis=1) > 1.0 + 1.0e-12):
            raise RuntimeError(
                "Interpolated paired detector Green sub-probability is invalid."
            )
        return np.maximum(output, 0.0), concentration

    def marginal_absolute_response_for_axis(
        self,
        incident_energy_axis_keV: Sequence[float],
        *,
        phase_weights: Sequence[float] | None = None,
        batch_size: int = 64,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return phase-marginal absolute pulse sub-probabilities.

        Default phase weights are uniform over projected detector area.  The
        effective concentration includes pulse-height and pulse-detection
        Monte Carlo uncertainty.
        """
        phase, concentration = self.phase_absolute_response_for_axis(
            incident_energy_axis_keV,
            batch_size=batch_size,
        )
        if phase_weights is None:
            edges = self.impact_parameter_edges_fraction
            weights = np.diff(np.square(edges))
        else:
            weights = _finite_array(
                phase_weights,
                field_name="phase_weights",
                ndim=1,
            )
        if (
            weights.shape != (phase.shape[0],)
            or np.any(weights < 0.0)
            or float(np.sum(weights)) <= 0.0
        ):
            raise ValueError("Detector Green phase weights are invalid.")
        weights = weights / np.sum(weights)
        # Keep phase marginalization on NumPy's fixed-axis reduction.  BLAS
        # dispatch is permitted to change execution speed, not model identity.
        response = np.sum(
            weights[:, np.newaxis, np.newaxis] * phase,
            axis=0,
            dtype=np.float64,
        )
        phase_no_pulse = 1.0 - np.sum(phase, axis=1)
        phase_square_sum = np.sum(np.square(phase), axis=1) + np.square(phase_no_pulse)
        trace = np.sum(
            np.square(weights)[:, np.newaxis]
            * (1.0 - phase_square_sum)
            / (concentration + 1.0),
            axis=0,
        )
        no_pulse = 1.0 - np.sum(response, axis=0)
        numerator = 1.0 - np.sum(np.square(response), axis=0) - np.square(no_pulse)
        effective = (
            np.divide(
                numerator,
                trace,
                out=np.full_like(numerator, 1.0e15),
                where=trace > np.finfo(np.float64).tiny,
            )
            - 1.0
        )
        effective = np.maximum(
            effective,
            DETECTOR_GREEN_MINIMUM_EFFECTIVE_HISTORIES,
        )
        return np.maximum(response, 0.0), effective

    def phase_detection_probability_for_axis(
        self,
        incident_energies_keV: Sequence[float],
    ) -> NDArray[np.float64]:
        """Interpolate registered-pulse probabilities by energy and phase."""
        targets = _finite_array(
            incident_energies_keV,
            field_name="incident_energies_keV",
            ndim=1,
        )
        if (
            targets.size == 0
            or np.any(np.diff(targets) <= 0.0)
            or targets[0] < self.energy_nodes_keV[0]
            or targets[-1] > self.energy_nodes_keV[-1]
        ):
            raise ValueError(
                "Detection energies must be increasing and inside the "
                "detector Green domain."
            )
        upper = np.searchsorted(self.energy_nodes_keV, targets, side="left")
        upper = np.clip(upper, 1, self.energy_nodes_keV.size - 1)
        exact = self.energy_nodes_keV[upper] == targets
        lower = np.where(exact, upper, upper - 1)
        denominator = self.energy_nodes_keV[upper] - self.energy_nodes_keV[lower]
        upper_weight = np.divide(
            targets - self.energy_nodes_keV[lower],
            denominator,
            out=np.zeros_like(targets),
            where=denominator > 0.0,
        )
        upper_weight = np.where(exact, 0.0, upper_weight)
        result = (1.0 - upper_weight)[
            :, np.newaxis
        ] * self.pulse_detection_probability_nc[lower] + upper_weight[
            :, np.newaxis
        ] * self.pulse_detection_probability_nc[upper]
        return np.ascontiguousarray(result.T, dtype=np.float64)

    def reference_pulse_detection_probability_for_axis(
        self,
        incident_energies_keV: Sequence[float],
        *,
        detector_target_radius_m: float,
    ) -> NDArray[np.float64]:
        """Return 1 m point-source pulse efficiencies for incident energies."""
        radius = float(detector_target_radius_m)
        if not math.isfinite(radius) or radius <= 0.0 or radius >= 1.0:
            raise ValueError(
                "Detector Green reference radius must be strictly between "
                "zero and one metre."
            )
        phase_detection = self.phase_detection_probability_for_axis(
            incident_energies_keV
        )
        edges = self.impact_parameter_edges_fraction
        normalization = max(
            1.0 - math.sqrt(max(0.0, 1.0 - radius * radius)),
            np.finfo(np.float64).tiny,
        )
        lower_cosine = np.sqrt(
            np.maximum(1.0 - np.square(radius * edges[:-1]), 0.0)
        )
        upper_cosine = np.sqrt(
            np.maximum(1.0 - np.square(radius * edges[1:]), 0.0)
        )
        phase_weights = (lower_cosine - upper_cosine) / normalization
        efficiencies = np.sum(
            phase_weights[:, np.newaxis] * phase_detection,
            axis=0,
            dtype=np.float64,
        )
        if (
            np.any(~np.isfinite(efficiencies))
            or np.any(efficiencies <= 0.0)
            or np.any(efficiencies > 1.0 + 1.0e-12)
        ):
            raise RuntimeError(
                "Detector Green reference pulse efficiencies are invalid."
            )
        return np.minimum(efficiencies, 1.0)

    def catalog_weighted_reference_efficiency(
        self,
        nuclide: Nuclide,
        *,
        detector_target_radius_m: float,
    ) -> float:
        """Return one catalog-line-weighted 1 m source-rate efficiency."""
        if not isinstance(nuclide, Nuclide):
            raise TypeError("nuclide must be an authenticated Nuclide catalog row.")
        energies = np.asarray(
            [line.energy_keV for line in nuclide.lines],
            dtype=np.float64,
        )
        intensities = np.asarray(
            [line.intensity for line in nuclide.lines],
            dtype=np.float64,
        )
        total_intensity = float(np.sum(intensities, dtype=np.float64))
        if (
            energies.size == 0
            or np.any(np.diff(energies) <= 0.0)
            or np.any(~np.isfinite(intensities))
            or np.any(intensities <= 0.0)
            or not math.isfinite(total_intensity)
            or total_intensity <= 0.0
        ):
            raise ValueError("Nuclide transport lines are invalid for normalization.")
        efficiencies = self.reference_pulse_detection_probability_for_axis(
            energies,
            detector_target_radius_m=detector_target_radius_m,
        )
        result = float(
            np.sum(intensities * efficiencies, dtype=np.float64) / total_intensity
        )
        if not math.isfinite(result) or result <= 0.0 or result > 1.0 + 1.0e-12:
            raise RuntimeError(
                "Catalog-weighted detector Green reference efficiency is invalid."
            )
        return min(result, 1.0)

    def manifest_payload(self) -> dict[str, object]:
        """Return the strict portable manifest for an artifact-backed operator."""
        self.require_runtime_ready()
        return {
            "schema_version": DETECTOR_GREEN_OPERATOR_SCHEMA_VERSION,
            "operator": DETECTOR_GREEN_OPERATOR_ID,
            "contract_hash_sha256": self.contract_hash_sha256,
            "input_energy_domain_keV": list(self.input_energy_domain_keV),
            "output_energy_axis": {
                "minimum_keV": self.output_energy_min_keV,
                "bin_width_keV": self.output_bin_width_keV,
                "bin_count": self.output_bin_count,
            },
            "boundary_state": DETECTOR_GREEN_BOUNDARY_STATE,
            "conditioning": DETECTOR_GREEN_CONDITIONING,
            "interpolation": DETECTOR_GREEN_INTERPOLATION,
            "binary": {
                "basename": DETECTOR_GREEN_BINARY_BASENAME,
                "sha256": self.binary_sha256,
                "probability_dtype": "float32_le",
                "layout": "energy_node_impact_bin_output_bin_c_order",
                "energy_node_count": int(self.energy_nodes_keV.size),
                "impact_bin_count": int(self.impact_parameter_edges_fraction.size - 1),
                "output_bin_count": self.output_bin_count,
            },
            "construction": dict(self.construction or {}),
        }

    def binary_bytes(self) -> bytes:
        """Serialize the exact custom binary consumed by Python and Geant4."""
        node_count, impact_count, output_count = self.conditional_response_ncb.shape
        header = DETECTOR_GREEN_BINARY_HEADER.pack(
            DETECTOR_GREEN_BINARY_MAGIC,
            DETECTOR_GREEN_OPERATOR_SCHEMA_VERSION,
            node_count,
            impact_count,
            output_count,
            self.output_energy_min_keV,
            self.output_bin_width_keV,
            float(self.energy_nodes_keV[0]),
            float(self.energy_nodes_keV[-1]),
        )
        return b"".join(
            (
                header,
                np.asarray(self.energy_nodes_keV, dtype="<f8").tobytes(order="C"),
                np.asarray(
                    self.impact_parameter_edges_fraction,
                    dtype="<f8",
                ).tobytes(order="C"),
                np.asarray(
                    self.conditional_response_ncb,
                    dtype="<f4",
                ).tobytes(order="C"),
                np.asarray(
                    self.effective_histories_nc,
                    dtype="<f8",
                ).tobytes(order="C"),
                np.asarray(
                    self.pulse_detection_probability_nc,
                    dtype="<f8",
                ).tobytes(order="C"),
            )
        )

    def write_artifact(self, output_directory: str | Path) -> Path:
        """Atomically publish one new immutable operator artifact directory."""
        import os
        import shutil
        import tempfile

        if self.construction is None:
            raise RuntimeError(
                "Detector Green artifact publication requires construction provenance."
            )
        destination = Path(output_directory).resolve()
        if destination.exists():
            raise FileExistsError(
                "Detector Green publication requires a new output directory: "
                f"{destination}."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        binary = self.binary_bytes()
        binary_sha256 = _sha256_bytes(binary)
        publishable = DetectorGreenOperator(
            energy_nodes_keV=self.energy_nodes_keV,
            impact_parameter_edges_fraction=(self.impact_parameter_edges_fraction),
            conditional_response_ncb=self.conditional_response_ncb,
            effective_histories_nc=self.effective_histories_nc,
            pulse_detection_probability_nc=(self.pulse_detection_probability_nc),
            output_energy_min_keV=self.output_energy_min_keV,
            output_bin_width_keV=self.output_bin_width_keV,
            construction=self.construction,
            binary_sha256=binary_sha256,
            contract_hash_sha256=self.contract_hash_sha256,
        )
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.staging.",
                dir=destination.parent,
            )
        )
        try:
            (staging / DETECTOR_GREEN_BINARY_BASENAME).write_bytes(binary)
            manifest_path = staging / DETECTOR_GREEN_MANIFEST_BASENAME
            manifest_path.write_bytes(
                canonical_json_bytes(publishable.manifest_payload())
            )
            reloaded = DetectorGreenOperator.from_artifact(manifest_path)
            if reloaded.contract_hash_sha256 != publishable.contract_hash_sha256:
                raise RuntimeError("Published detector Green operator failed readback.")
            os.replace(staging, destination)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return destination / DETECTOR_GREEN_MANIFEST_BASENAME

    @classmethod
    def from_artifact(cls, manifest_path: str | Path) -> "DetectorGreenOperator":
        """Load and authenticate one immutable operator artifact directory."""
        path = Path(manifest_path).resolve()
        payload = _validate_manifest_payload(
            json.loads(path.read_text(encoding="utf-8"))
        )
        binary_path = path.parent / str(payload["binary"]["basename"])
        binary = binary_path.read_bytes()
        if _sha256_bytes(binary) != payload["binary"]["sha256"]:
            raise ValueError("Detector Green binary hash is stale.")
        operator = cls._from_binary(
            binary,
            construction=payload["construction"],
            binary_sha256=str(payload["binary"]["sha256"]),
            contract_hash_sha256=str(payload["contract_hash_sha256"]),
        )
        if operator.manifest_payload() != payload:
            raise ValueError(
                "Detector Green artifact does not reconstruct its manifest."
            )
        return operator

    @classmethod
    def _from_binary(
        cls,
        payload: bytes,
        *,
        construction: Mapping[str, object],
        binary_sha256: str,
        contract_hash_sha256: str,
    ) -> "DetectorGreenOperator":
        """Decode a validated custom operator binary."""
        if len(payload) < DETECTOR_GREEN_BINARY_HEADER.size:
            raise ValueError("Detector Green binary is truncated.")
        (
            magic,
            schema_version,
            node_count,
            impact_count,
            output_count,
            output_minimum,
            output_width,
            domain_minimum,
            domain_maximum,
        ) = DETECTOR_GREEN_BINARY_HEADER.unpack_from(payload)
        if (
            magic != DETECTOR_GREEN_BINARY_MAGIC
            or schema_version != DETECTOR_GREEN_OPERATOR_SCHEMA_VERSION
            or node_count < 2
            or impact_count < 1
            or output_count < 2
            or (domain_minimum, domain_maximum) != DETECTOR_GREEN_INPUT_DOMAIN_KEV
        ):
            raise ValueError("Detector Green binary header is incompatible.")
        offset = DETECTOR_GREEN_BINARY_HEADER.size

        def take(dtype: str, count: int) -> NDArray[np.float64]:
            """Take one typed contiguous array from the binary payload."""
            nonlocal offset
            itemsize = np.dtype(dtype).itemsize
            stop = offset + itemsize * count
            if stop > len(payload):
                raise ValueError("Detector Green binary is truncated.")
            result = np.frombuffer(
                payload,
                dtype=dtype,
                count=count,
                offset=offset,
            ).astype(np.float64)
            offset = stop
            return result

        nodes = take("<f8", node_count)
        edges = take("<f8", impact_count + 1)
        response = take("<f4", node_count * impact_count * output_count)
        histories = take("<f8", node_count * impact_count)
        detection = take("<f8", node_count * impact_count)
        if offset != len(payload):
            raise ValueError("Detector Green binary contains trailing bytes.")
        return cls(
            energy_nodes_keV=nodes,
            impact_parameter_edges_fraction=edges,
            conditional_response_ncb=response.reshape(
                node_count,
                impact_count,
                output_count,
            ),
            effective_histories_nc=histories.reshape(
                node_count,
                impact_count,
            ),
            pulse_detection_probability_nc=detection.reshape(
                node_count,
                impact_count,
            ),
            output_energy_min_keV=output_minimum,
            output_bin_width_keV=output_width,
            construction=construction,
            binary_sha256=binary_sha256,
            contract_hash_sha256=contract_hash_sha256,
        )

    def _interpolate_block(
        self,
        targets_keV: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Interpolate one bounded target-energy block without scalar bin loops."""
        nodes = self.energy_nodes_keV
        upper = np.searchsorted(nodes, targets_keV, side="left")
        upper = np.clip(upper, 1, nodes.size - 1)
        exact_upper = nodes[upper] == targets_keV
        lower = np.where(exact_upper, upper, upper - 1)
        denominator = nodes[upper] - nodes[lower]
        upper_weight = np.divide(
            targets_keV - nodes[lower],
            denominator,
            out=np.zeros_like(targets_keV),
            where=denominator > 0.0,
        )
        upper_weight = np.where(exact_upper, 0.0, upper_weight)
        lower_weight = 1.0 - upper_weight
        lower_response = self._rescale_response(
            self.conditional_response_ncb[lower],
            source_energies_keV=nodes[lower],
            target_energies_keV=targets_keV,
        )
        upper_response = self._rescale_response(
            self.conditional_response_ncb[upper],
            source_energies_keV=nodes[upper],
            target_energies_keV=targets_keV,
        )
        response = (
            lower_weight[:, np.newaxis, np.newaxis] * lower_response
            + upper_weight[:, np.newaxis, np.newaxis] * upper_response
        )
        response /= np.maximum(
            np.sum(response, axis=-1, keepdims=True),
            np.finfo(np.float64).tiny,
        )
        pulse_histories = np.maximum(
            self.effective_histories_nc * self.pulse_detection_probability_nc,
            DETECTOR_GREEN_MINIMUM_EFFECTIVE_HISTORIES,
        )
        lower_trace = (1.0 - np.sum(np.square(lower_response), axis=-1)) / (
            pulse_histories[lower] + 1.0
        )
        upper_trace = (1.0 - np.sum(np.square(upper_response), axis=-1)) / (
            pulse_histories[upper] + 1.0
        )
        trace = (
            np.square(lower_weight)[:, np.newaxis] * lower_trace
            + np.square(upper_weight)[:, np.newaxis] * upper_trace
        )
        numerator = 1.0 - np.sum(np.square(response), axis=-1)
        concentration = (
            np.divide(
                numerator,
                trace,
                out=np.full_like(numerator, 1.0e15),
                where=trace > np.finfo(np.float64).tiny,
            )
            - 1.0
        )
        concentration = np.maximum(
            concentration,
            DETECTOR_GREEN_MINIMUM_EFFECTIVE_HISTORIES,
        )
        return response, concentration

    def _interpolate_absolute_block(
        self,
        targets_keV: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Interpolate pulse and no-pulse probabilities with MC covariance."""
        nodes = self.energy_nodes_keV
        upper = np.searchsorted(nodes, targets_keV, side="left")
        upper = np.clip(upper, 1, nodes.size - 1)
        exact_upper = nodes[upper] == targets_keV
        lower = np.where(exact_upper, upper, upper - 1)
        denominator = nodes[upper] - nodes[lower]
        upper_weight = np.divide(
            targets_keV - nodes[lower],
            denominator,
            out=np.zeros_like(targets_keV),
            where=denominator > 0.0,
        )
        upper_weight = np.where(exact_upper, 0.0, upper_weight)
        lower_weight = 1.0 - upper_weight
        lower_conditional = self._rescale_response(
            self.conditional_response_ncb[lower],
            source_energies_keV=nodes[lower],
            target_energies_keV=targets_keV,
        )
        upper_conditional = self._rescale_response(
            self.conditional_response_ncb[upper],
            source_energies_keV=nodes[upper],
            target_energies_keV=targets_keV,
        )
        lower_detection = self.pulse_detection_probability_nc[lower]
        upper_detection = self.pulse_detection_probability_nc[upper]
        lower_response = lower_conditional * lower_detection[..., np.newaxis]
        upper_response = upper_conditional * upper_detection[..., np.newaxis]
        response = (
            lower_weight[:, np.newaxis, np.newaxis] * lower_response
            + upper_weight[:, np.newaxis, np.newaxis] * upper_response
        )
        lower_square_sum = np.sum(np.square(lower_response), axis=-1) + np.square(
            1.0 - lower_detection
        )
        upper_square_sum = np.sum(np.square(upper_response), axis=-1) + np.square(
            1.0 - upper_detection
        )
        trace = np.square(lower_weight)[:, np.newaxis] * (1.0 - lower_square_sum) / (
            self.effective_histories_nc[lower] + 1.0
        ) + np.square(upper_weight)[:, np.newaxis] * (1.0 - upper_square_sum) / (
            self.effective_histories_nc[upper] + 1.0
        )
        detection = np.sum(response, axis=-1)
        numerator = (
            1.0 - np.sum(np.square(response), axis=-1) - np.square(1.0 - detection)
        )
        concentration = (
            np.divide(
                numerator,
                trace,
                out=np.full_like(numerator, 1.0e15),
                where=trace > np.finfo(np.float64).tiny,
            )
            - 1.0
        )
        concentration = np.maximum(
            concentration,
            DETECTOR_GREEN_MINIMUM_EFFECTIVE_HISTORIES,
        )
        return response, concentration

    def _interpolate_absolute_selected_phase_block(
        self,
        targets_keV: NDArray[np.float64],
        phases: NDArray[np.int64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Interpolate aligned energy/phase rows without a phase cross-product."""
        nodes = self.energy_nodes_keV
        upper = np.searchsorted(nodes, targets_keV, side="left")
        upper = np.clip(upper, 1, nodes.size - 1)
        exact_upper = nodes[upper] == targets_keV
        lower = np.where(exact_upper, upper, upper - 1)
        denominator = nodes[upper] - nodes[lower]
        upper_weight = np.divide(
            targets_keV - nodes[lower],
            denominator,
            out=np.zeros_like(targets_keV),
            where=denominator > 0.0,
        )
        upper_weight = np.where(exact_upper, 0.0, upper_weight)
        lower_weight = 1.0 - upper_weight
        lower_conditional = self._rescale_response(
            self.conditional_response_ncb[lower, phases, :][:, np.newaxis, :],
            source_energies_keV=nodes[lower],
            target_energies_keV=targets_keV,
        )[:, 0, :]
        upper_conditional = self._rescale_response(
            self.conditional_response_ncb[upper, phases, :][:, np.newaxis, :],
            source_energies_keV=nodes[upper],
            target_energies_keV=targets_keV,
        )[:, 0, :]
        lower_detection = self.pulse_detection_probability_nc[lower, phases]
        upper_detection = self.pulse_detection_probability_nc[upper, phases]
        lower_response = lower_conditional * lower_detection[:, np.newaxis]
        upper_response = upper_conditional * upper_detection[:, np.newaxis]
        response = (
            lower_weight[:, np.newaxis] * lower_response
            + upper_weight[:, np.newaxis] * upper_response
        )
        lower_square_sum = np.sum(np.square(lower_response), axis=-1) + np.square(
            1.0 - lower_detection
        )
        upper_square_sum = np.sum(np.square(upper_response), axis=-1) + np.square(
            1.0 - upper_detection
        )
        trace = np.square(lower_weight) * (1.0 - lower_square_sum) / (
            self.effective_histories_nc[lower, phases] + 1.0
        ) + np.square(upper_weight) * (1.0 - upper_square_sum) / (
            self.effective_histories_nc[upper, phases] + 1.0
        )
        detection = np.sum(response, axis=-1)
        numerator = (
            1.0 - np.sum(np.square(response), axis=-1) - np.square(1.0 - detection)
        )
        concentration = (
            np.divide(
                numerator,
                trace,
                out=np.full_like(numerator, 1.0e15),
                where=trace > np.finfo(np.float64).tiny,
            )
            - 1.0
        )
        return response, np.maximum(
            concentration,
            DETECTOR_GREEN_MINIMUM_EFFECTIVE_HISTORIES,
        )

    def _rescale_response(
        self,
        response_tcb: NDArray[np.float64],
        *,
        source_energies_keV: NDArray[np.float64],
        target_energies_keV: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Align morphology while preserving exact raw-bin peak anchors."""
        target_count, impact_count, output_count = response_tcb.shape
        result = np.zeros_like(response_tcb, dtype=np.float64)
        positive = source_energies_keV > 0.0
        source_anchor = (
            np.floor(
                (source_energies_keV - self.output_energy_min_keV)
                / self.output_bin_width_keV
            )
            + 0.5
        )
        target_anchor = (
            np.floor(
                (target_energies_keV - self.output_energy_min_keV)
                / self.output_bin_width_keV
            )
            + 0.5
        )
        scale = np.divide(
            target_anchor,
            source_anchor,
            out=np.ones_like(target_energies_keV),
            where=positive,
        )
        coordinate = (np.arange(output_count, dtype=np.float64) + 0.5)[
            np.newaxis, :
        ] * scale[:, np.newaxis] - 0.5
        raw_lower = np.floor(coordinate).astype(np.int64)
        fraction = coordinate - raw_lower
        lower = np.clip(raw_lower, 0, output_count - 1)
        upper = np.clip(raw_lower + 1, 0, output_count - 1)
        fraction = np.where(lower == upper, 0.0, fraction)
        target_index = np.broadcast_to(
            np.arange(target_count)[:, np.newaxis, np.newaxis],
            response_tcb.shape,
        )
        impact_index = np.broadcast_to(
            np.arange(impact_count)[np.newaxis, :, np.newaxis],
            response_tcb.shape,
        )
        lower_index = np.broadcast_to(
            lower[:, np.newaxis, :],
            response_tcb.shape,
        )
        upper_index = np.broadcast_to(
            upper[:, np.newaxis, :],
            response_tcb.shape,
        )
        fraction_3d = fraction[:, np.newaxis, :]
        np.add.at(
            result,
            (target_index, impact_index, lower_index),
            response_tcb * (1.0 - fraction_3d),
        )
        np.add.at(
            result,
            (target_index, impact_index, upper_index),
            response_tcb * fraction_3d,
        )
        if np.any(~positive):
            result[~positive] = 0.0
            result[~positive, :, 0] = 1.0
        return result

    @staticmethod
    def _contract_hash(
        *,
        nodes: NDArray[np.float64],
        edges: NDArray[np.float64],
        response: NDArray[np.float64],
        histories: NDArray[np.float64],
        detection: NDArray[np.float64],
        construction: Mapping[str, object] | None,
    ) -> str:
        """Return the physical and numerical operator contract digest."""
        digest = hashlib.sha256()
        digest.update(
            canonical_json_bytes(
                {
                    "schema_version": DETECTOR_GREEN_OPERATOR_SCHEMA_VERSION,
                    "operator": DETECTOR_GREEN_OPERATOR_ID,
                    "input_energy_domain_keV": list(DETECTOR_GREEN_INPUT_DOMAIN_KEV),
                    "boundary_state": DETECTOR_GREEN_BOUNDARY_STATE,
                    "conditioning": DETECTOR_GREEN_CONDITIONING,
                    "interpolation": DETECTOR_GREEN_INTERPOLATION,
                    "construction": (
                        None if construction is None else dict(construction)
                    ),
                }
            )
        )
        for array in (nodes, edges, response, histories, detection):
            canonical = np.ascontiguousarray(array, dtype="<f8")
            digest.update(str(canonical.shape).encode("ascii"))
            digest.update(canonical.tobytes(order="C"))
        return digest.hexdigest()


__all__ = [
    "DETECTOR_GREEN_BINARY_BASENAME",
    "DETECTOR_GREEN_CONDITIONING",
    "DETECTOR_GREEN_INPUT_DOMAIN_KEV",
    "DETECTOR_GREEN_MANIFEST_BASENAME",
    "DETECTOR_GREEN_OPERATOR_ID",
    "DetectorGreenOperator",
    "canonical_json_bytes",
]
