"""Construct an isotope-independent detector Green operator from Geant4 counts."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy import special

from spectrum.detector_green_operator import DetectorGreenOperator


DETECTOR_ENERGY_RESOLUTION_SPEC: Final = (
    "registered_pulse_gaussian_resolution_v1|"
    "sigma_keV=max(0.5*sqrt(E_keV)-1.5,0.5)|"
    "raw_deposit_bin_center_quadrature|conditional_in_axis_normalization"
)
DETECTOR_ENERGY_RESOLUTION_CONTRACT_SHA256: Final = hashlib.sha256(
    DETECTOR_ENERGY_RESOLUTION_SPEC.encode("ascii")
).hexdigest()
DETECTOR_GREEN_DEFAULT_POSITIVE_NODE_COUNT: Final = 65


def catalog_independent_energy_nodes_keV(
    positive_node_count: int = DETECTOR_GREEN_DEFAULT_POSITIVE_NODE_COUNT,
) -> NDArray[np.float64]:
    """Return deterministic continuous-domain nodes unrelated to catalog lines."""
    if (
        isinstance(positive_node_count, bool)
        or not isinstance(positive_node_count, int)
        or positive_node_count < 3
    ):
        raise ValueError("positive_node_count must be an integer of at least 3.")
    minimum = 2.0
    maximum = 1700.0
    index = np.arange(positive_node_count, dtype=np.float64)
    # Chebyshev--Lobatto placement resolves both low-energy threshold behavior
    # and the high-energy endpoint without consulting any isotope catalog.
    positive = 0.5 * (minimum + maximum) - 0.5 * (maximum - minimum) * np.cos(
        np.pi * index / float(positive_node_count - 1)
    )
    nodes = np.concatenate((np.asarray((0.0,), dtype=np.float64), positive))
    nodes[0] = 0.0
    nodes[1] = minimum
    nodes[-1] = maximum
    if np.any(np.diff(nodes) <= 0.0):
        raise RuntimeError("Detector Green energy-node design is not increasing.")
    return np.ascontiguousarray(nodes)


def impact_parameter_edges_for_equal_solid_angle_strata(
    *,
    source_distance_m: float,
    detector_target_radius_m: float,
    stratum_count: int,
) -> NDArray[np.float64]:
    """Return normalized impact edges induced by native equal-mu cone strata."""
    distance = float(source_distance_m)
    radius = float(detector_target_radius_m)
    if (
        not math.isfinite(distance)
        or not math.isfinite(radius)
        or radius <= 0.0
        or distance <= radius
        or isinstance(stratum_count, bool)
        or not isinstance(stratum_count, int)
        or stratum_count <= 0
    ):
        raise ValueError(
            "Impact strata require a source outside a positive detector radius."
        )
    cosine_limit = math.sqrt(max(0.0, 1.0 - (radius / distance) ** 2))
    fractions = np.linspace(0.0, 1.0, stratum_count + 1)
    cosine = cosine_limit + (1.0 - cosine_limit) * fractions
    descending = distance * np.sqrt(np.maximum(0.0, 1.0 - cosine**2)) / radius
    edges = np.ascontiguousarray(descending[::-1], dtype=np.float64)
    edges[0] = 0.0
    edges[-1] = 1.0
    if np.any(np.diff(edges) <= 0.0):
        raise RuntimeError("Native cone strata do not define impact bins.")
    return edges


def gaussian_resolution_operator(
    *,
    output_bin_count: int,
    output_bin_width_keV: float,
) -> NDArray[np.float64]:
    """Return the count-preserving detector-resolution folding matrix."""
    if (
        isinstance(output_bin_count, bool)
        or not isinstance(output_bin_count, int)
        or output_bin_count < 2
        or not math.isfinite(float(output_bin_width_keV))
        or float(output_bin_width_keV) <= 0.0
    ):
        raise ValueError("Detector resolution axis is invalid.")
    width = float(output_bin_width_keV)
    centres = (np.arange(output_bin_count, dtype=np.float64) + 0.5) * width
    sigma = np.maximum(0.5 * np.sqrt(centres) - 1.5, 0.5)
    lower_edges = np.arange(output_bin_count, dtype=np.float64) * width
    upper_edges = lower_edges + width
    z_lower = (lower_edges[:, np.newaxis] - centres[np.newaxis, :]) / sigma[
        np.newaxis, :
    ]
    z_upper = (upper_edges[:, np.newaxis] - centres[np.newaxis, :]) / sigma[
        np.newaxis, :
    ]
    operator = special.ndtr(z_upper) - special.ndtr(z_lower)
    operator = np.maximum(operator, 0.0)
    operator /= np.maximum(
        np.sum(operator, axis=0, keepdims=True),
        np.finfo(np.float64).tiny,
    )
    return np.ascontiguousarray(operator, dtype=np.float64)


def build_detector_green_operator(
    *,
    energy_nodes_keV: NDArray[np.float64],
    impact_parameter_edges_fraction: NDArray[np.float64],
    raw_deposit_histograms_ncb: NDArray[np.float64],
    sampled_histories_nc: NDArray[np.float64],
    construction: Mapping[str, object],
    output_bin_width_keV: float = 2.0,
) -> DetectorGreenOperator:
    """Build the conditional pulse operator and finite-MC uncertainty budget."""
    nodes = np.asarray(energy_nodes_keV, dtype=np.float64)
    edges = np.asarray(impact_parameter_edges_fraction, dtype=np.float64)
    raw = np.asarray(raw_deposit_histograms_ncb, dtype=np.float64)
    histories = np.asarray(sampled_histories_nc, dtype=np.float64)
    if (
        raw.ndim != 3
        or nodes.shape != (raw.shape[0],)
        or edges.shape != (raw.shape[1] + 1,)
        or histories.shape != raw.shape[:2]
        or np.any(~np.isfinite(raw))
        or np.any(raw < 0.0)
        or np.any(~np.isfinite(histories))
        or np.any(histories < 2.0)
    ):
        raise ValueError("Detector Green construction arrays are invalid.")
    pulse_counts = np.sum(raw, axis=-1)
    positive_nodes = nodes > 0.0
    resolution = gaussian_resolution_operator(
        output_bin_count=raw.shape[-1],
        output_bin_width_keV=output_bin_width_keV,
    )
    marked = np.einsum("br,ncr->ncb", resolution, raw, optimize=True)
    response = np.divide(
        marked,
        np.sum(marked, axis=-1, keepdims=True),
        out=np.zeros_like(marked),
        where=np.sum(marked, axis=-1, keepdims=True) > 0.0,
    )
    # A zero-pulse cell is a valid Monte Carlo outcome, especially for
    # low-energy photons absorbed by the detector housing.  Its absolute
    # registered-pulse sub-probability is exactly zero.  The conditional
    # pulse-height law is therefore mathematically unidentified and cannot
    # affect an absolute response; use one explicit sentinel distribution so
    # the serialized conditional tensor remains normalized.  No pseudo-event
    # is added to either the pulse count or the detection probability.
    no_pulse_cells = pulse_counts == 0.0
    response[no_pulse_cells] = 0.0
    response[no_pulse_cells, 0] = 1.0
    # Zero energy is a numerical underflow sentinel rather than a physical
    # photon.  It is explicit and cannot be selected as a catalog line.
    zero_nodes = ~positive_nodes
    response[zero_nodes] = 0.0
    response[zero_nodes, :, 0] = 1.0
    # Keep the incident-history count.  Together with the pulse probability it
    # defines a categorical corpus over every observed bin plus the no-pulse
    # outcome.  Conditional pulse-shape precision is derived as N * p_detect.
    effective = np.maximum(histories, 2.0)
    detection = np.divide(
        pulse_counts,
        histories,
        out=np.zeros_like(pulse_counts),
        where=histories > 0.0,
    )
    if np.any(detection > 1.0 + 1.0e-12):
        raise ValueError(
            "Detector Green corpus reports more than one pulse per history."
        )
    detection = np.clip(detection, 0.0, 1.0)
    return DetectorGreenOperator(
        energy_nodes_keV=nodes,
        impact_parameter_edges_fraction=edges,
        conditional_response_ncb=response,
        effective_histories_nc=effective,
        pulse_detection_probability_nc=detection,
        output_energy_min_keV=0.0,
        output_bin_width_keV=float(output_bin_width_keV),
        construction=dict(construction),
    )


def detector_green_raw_corpus_sha256(payload: object) -> str:
    """Return the strict digest of one isotope-free raw construction corpus."""
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "DETECTOR_ENERGY_RESOLUTION_CONTRACT_SHA256",
    "build_detector_green_operator",
    "catalog_independent_energy_nodes_keV",
    "detector_green_raw_corpus_sha256",
    "gaussian_resolution_operator",
    "impact_parameter_edges_for_equal_solid_angle_strata",
]
