"""Acquire an isotope-independent monoenergetic full-detector corpus."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import tempfile
from types import MappingProxyType

import numpy as np

from measurement.source_boundary import (
    SURFACE_EMISSION_EPSILON_M,
    surface_emission_policy_sha256,
    surface_source_runtime_contract_sha256,
)
from runtime.experiment_profiles import STANDARD_ACQUISITION_LIVE_TIME_S
from sim.geant4_app.app import (
    Geant4AppConfig,
    Geant4Application,
    validate_mean_calibration_transport_metadata,
)
from sim.geant4_app.engine import Geant4StepRequest
from sim.geant4_app.scene_export import (
    ExportedGeant4Scene,
    ExportedGeant4Source,
)
from sim.isaacsim_app.scene_builder import StagePrimPaths
from sim.runtime import load_production_runtime_config_with_digest
from spectrum.detector_green_construction import (
    DETECTOR_ENERGY_RESOLUTION_CONTRACT_SHA256,
    build_detector_green_operator,
    catalog_independent_energy_nodes_keV,
    detector_green_raw_corpus_sha256,
    impact_parameter_edges_for_equal_solid_angle_strata,
)
from spectrum.detector_green_operator import canonical_json_bytes
from spectrum.detector_green_provenance import (
    detector_green_implementation_bundle_sha256,
)
from spectrum.geant4_physics import GEANT4_PHYSICS_CONTRACT_SHA256
from spectrum.library import Nuclide, NuclideLine
from spectrum.mean_calibration import parse_mean_calibration_metadata
from spectrum.native_metadata import native_source_line_token
from spectrum.response_matrix import (
    NATIVE_GEANT4_BIN_COUNT,
    NATIVE_GEANT4_BIN_WIDTH_KEV,
)
from spectrum.runtime_model_keys import FULL_SPECTRUM_MODEL_RUNTIME_KEYS


DETECTOR_GREEN_RAW_CORPUS_SCHEMA_VERSION = 2
DETECTOR_GREEN_RAW_CORPUS_ID = "monoenergetic_full_detector_phase_space_corpus_v2"
DETECTOR_GREEN_RAW_CORPUS_BASENAME = "raw_corpus.json"
DETECTOR_GREEN_MINIMUM_HISTORIES_PER_ENERGY = 100_000
DETECTOR_GREEN_DEFAULT_IMPACT_STRATA = 8
DETECTOR_GREEN_REFERENCE_SOURCE_DISTANCE_M = 0.05
_REFERENCE_DETECTOR_POSE_XYZ = (1.0, 1.0, 0.5)
_PROBE_NAME = "MonoenergeticPhotonProbe"


def _probe_library(
    energy_nodes_keV: np.ndarray,
) -> Mapping[str, Nuclide]:
    """Return an offline-only independent-photon line schedule."""
    positive = np.asarray(energy_nodes_keV, dtype=np.float64)
    positive = positive[positive > 0.0]
    lines = tuple(
        NuclideLine(energy_keV=float(energy), intensity=1.0) for energy in positive
    )
    probe = Nuclide(
        name=_PROBE_NAME,
        lines=lines,
        decay_lines=lines,
        representative_energy_keV=float(np.median(positive)),
        source_origin="detector_operator_probe",
        eligible_materials=("*",),
        decay_data_reference="numerical_monoenergetic_probe",
        prompt_cascade_model="independent_monoenergetic_probe",
    )
    return MappingProxyType({_PROBE_NAME: probe})


def _build_app_payload(
    runtime_config: Mapping[str, object],
    *,
    repository_root: Path,
    histories_per_energy: int,
    impact_strata: int,
) -> dict[str, object]:
    """Build the isolated fixed-quota full-detector construction payload."""
    payload = {
        key: value
        for key, value in dict(runtime_config).items()
        if key not in FULL_SPECTRUM_MODEL_RUNTIME_KEYS
    }
    executable_raw = payload.get("executable_path")
    if not isinstance(executable_raw, str) or not executable_raw:
        raise ValueError(
            "Detector Green construction requires explicit executable_path."
        )
    executable = Path(executable_raw)
    if not executable.is_absolute():
        executable = repository_root / executable
    payload.update(
        {
            "executable_path": executable.resolve().as_posix(),
            "engine_mode": "external",
            "source_rate_model": "detector_cps_1m",
            "primary_emission_model": "independent_gamma_lines",
            "detector_scoring_mode": "full_transport",
            "secondary_transport_mode": "full_transport",
            "sample_detector_response": False,
            "detector_green_operator_manifest": None,
            "validation_entry_class_spectra": True,
            "background_cps": 0.0,
            "dead_time_tau_s": 0.0,
            "primary_sampling_fraction": 1.0,
            "target_sampled_primaries": None,
            "accelerated_weighted_transport_enable": False,
            "mean_calibration_histories_per_source_line": int(histories_per_energy),
            "mean_calibration_angle_strata_mu": int(impact_strata),
            "mean_calibration_angle_strata_phi": 1,
            "mean_calibration_forced_collision": False,
            "persistent_process": False,
        }
    )
    config = Geant4AppConfig.from_dict(payload)
    if (
        config.detector_scoring_mode != "full_transport"
        or config.secondary_transport_mode != "full_transport"
        or config.sample_detector_response
        or not config.validation_entry_class_spectra
        or config.background_cps != 0.0
        or config.dead_time_tau_s != 0.0
        or config.mean_calibration_histories_per_source_line != histories_per_energy
        or config.mean_calibration_angle_strata_mu != impact_strata
        or config.mean_calibration_angle_strata_phi != 1
        or config.mean_calibration_forced_collision
    ):
        raise RuntimeError("Detector Green construction payload drifted.")
    return payload


def _reference_source() -> ExportedGeant4Source:
    """Return one air-side monoenergetic probe source."""
    detector = np.asarray(_REFERENCE_DETECTOR_POSE_XYZ, dtype=np.float64)
    normal = np.asarray((-1.0, 0.0, 0.0), dtype=np.float64)
    transport = detector + np.asarray(
        (DETECTOR_GREEN_REFERENCE_SOURCE_DISTANCE_M, 0.0, 0.0),
        dtype=np.float64,
    )
    anchor = transport - SURFACE_EMISSION_EPSILON_M * normal
    return ExportedGeant4Source(
        isotope=_PROBE_NAME,
        position_xyz=tuple(float(value) for value in transport),
        intensity_cps_1m=1.0,
        anchor_position_xyz=tuple(float(value) for value in anchor),
        surface_chart_id=0,
        surface_uv=(0.25, 0.25),
        surface_normal_xyz=tuple(float(value) for value in normal),
        surface_emission_policy_sha256=surface_emission_policy_sha256(),
    )


def _source_contract_sha256(source: ExportedGeant4Source) -> str:
    """Return the exact shared source-surface digest for the probe."""
    return surface_source_runtime_contract_sha256(
        [
            {
                "isotope": source.isotope,
                "position": list(source.anchor_position_xyz),
                "transport_position": list(source.position_xyz),
                "intensity_cps_1m": float(source.intensity_cps_1m),
                "surface_chart_id": source.surface_chart_id,
                "surface_uv": list(source.surface_uv),
                "surface_normal": list(source.surface_normal_xyz),
                "surface_emission_policy_sha256": (
                    source.surface_emission_policy_sha256
                ),
            }
        ]
    )


def _reference_scene(config: Geant4AppConfig) -> ExportedGeant4Scene:
    """Build a detector-only scene with no room, obstacle, or shield material."""
    source = _reference_source()
    provisional = ExportedGeant4Scene(
        scene_hash="",
        usd_path=None,
        room_size_xyz=(2.0, 2.0, 2.0),
        static_volumes=(),
        sources=(source,),
        detector_model=config.detector_model,
        fe_shield=None,
        pb_shield=None,
        prim_paths=StagePrimPaths(),
    )
    stable = provisional.to_dict()
    stable.pop("scene_hash")
    scene_hash = hashlib.sha256(
        json.dumps(
            stable,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return replace(provisional, scene_hash=scene_hash)


def _orientation_zero_quaternion_wxyz() -> tuple[float, float, float, float]:
    """Return the exact orientation-zero octant placement quaternion."""
    root_half = math.sqrt(0.5)
    return (0.0, root_half, -root_half, 0.0)


def _reference_request(*, transport_seed: int) -> Geant4StepRequest:
    """Return one standard-live-time fixed detector-only request."""
    shield_quaternion = _orientation_zero_quaternion_wxyz()
    return Geant4StepRequest(
        step_id=0,
        dwell_time_s=STANDARD_ACQUISITION_LIVE_TIME_S,
        seed=int(transport_seed),
        detector_pose_xyz=_REFERENCE_DETECTOR_POSE_XYZ,
        detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        fe_shield_pose_xyz=_REFERENCE_DETECTOR_POSE_XYZ,
        fe_shield_quat_wxyz=shield_quaternion,
        pb_shield_pose_xyz=_REFERENCE_DETECTOR_POSE_XYZ,
        pb_shield_quat_wxyz=shield_quaternion,
        fe_orientation_index=0,
        pb_orientation_index=0,
    )


def _dense_corpus_arrays(
    metadata: Mapping[str, object],
    *,
    energy_nodes_keV: np.ndarray,
    impact_strata: int,
    histories_per_energy: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract energy/impact raw pulse histograms from native batch statistics."""
    calibration = parse_mean_calibration_metadata(
        metadata,
        bin_count=NATIVE_GEANT4_BIN_COUNT,
    )
    positive = np.asarray(energy_nodes_keV, dtype=np.float64)
    positive = positive[positive > 0.0]
    expected_histories_per_stratum = histories_per_energy // impact_strata
    batches_by_line: dict[str, list[object]] = {}
    for batch in calibration.batches:
        batches_by_line.setdefault(batch.line_token, []).append(batch)
    expected_tokens = {
        native_source_line_token(
            source_index=0,
            isotope=_PROBE_NAME,
            energy_keV=float(energy),
        )
        for energy in positive
    }
    if set(batches_by_line) != expected_tokens:
        raise RuntimeError("Detector Green native energy schedule is incomplete.")
    raw = np.zeros(
        (energy_nodes_keV.size, impact_strata, NATIVE_GEANT4_BIN_COUNT),
        dtype=np.float64,
    )
    sampled = np.full(
        (energy_nodes_keV.size, impact_strata),
        float(expected_histories_per_stratum),
        dtype=np.float64,
    )
    for energy_index, energy in enumerate(positive, start=1):
        token = native_source_line_token(
            source_index=0,
            isotope=_PROBE_NAME,
            energy_keV=float(energy),
        )
        batches = batches_by_line[token]
        if len(batches) != impact_strata:
            raise RuntimeError(
                "Detector Green energy does not cover every impact stratum."
            )
        seen: set[int] = set()
        for batch in batches:
            native_index = int(batch.angle_stratum_index)
            phase_index = impact_strata - 1 - native_index
            if (
                native_index < 0
                or native_index >= impact_strata
                or phase_index in seen
                or int(batch.sampled_histories) != expected_histories_per_stratum
            ):
                raise RuntimeError(
                    "Detector Green impact-stratum batch metadata is invalid."
                )
            seen.add(phase_index)
            for bin_index, count in batch.combined_histogram():
                raw[energy_index, phase_index, bin_index] = float(count)
    return raw, sampled


def _sparse_raw_corpus(
    *,
    energy_nodes_keV: np.ndarray,
    impact_edges: np.ndarray,
    raw_histograms: np.ndarray,
    sampled_histories: np.ndarray,
    runtime_config_sha256: str,
    app: Geant4Application,
    scene: ExportedGeant4Scene,
    transport_seed: int,
    histories_per_energy: int,
) -> dict[str, object]:
    """Return a compact isotope-free raw construction corpus."""
    provenance = (
        app.native_executable_sha256,
        app.native_execution_environment_sha256,
        app.detector_implementation_bundle_sha256,
    )
    if any(value is None for value in provenance):
        raise RuntimeError("Detector Green native provenance is incomplete.")
    cells = []
    for node_index in range(1, energy_nodes_keV.size):
        for phase_index in range(impact_edges.size - 1):
            histogram = raw_histograms[node_index, phase_index]
            cells.append(
                {
                    "energy_node_index": node_index,
                    "impact_bin_index": phase_index,
                    "sampled_histories": int(
                        sampled_histories[node_index, phase_index]
                    ),
                    "registered_pulses": int(np.sum(histogram)),
                    "sparse_raw_deposit_histogram": [
                        [int(index), int(histogram[index])]
                        for index in np.flatnonzero(histogram)
                    ],
                }
            )
    detector_model_sha256 = hashlib.sha256(
        canonical_json_bytes(scene.detector_model.to_dict())
    ).hexdigest()
    return {
        "schema_version": DETECTOR_GREEN_RAW_CORPUS_SCHEMA_VERSION,
        "corpus": DETECTOR_GREEN_RAW_CORPUS_ID,
        "energy_node_design": (
            "catalog_independent_deterministic_continuous_domain_v1"
        ),
        "energy_nodes_keV": energy_nodes_keV.tolist(),
        "impact_parameter_edges_fraction": impact_edges.tolist(),
        "output_energy_axis": {
            "minimum_keV": 0.0,
            "bin_width_keV": NATIVE_GEANT4_BIN_WIDTH_KEV,
            "bin_count": NATIVE_GEANT4_BIN_COUNT,
        },
        "dwell_time_s": STANDARD_ACQUISITION_LIVE_TIME_S,
        "transport_seed": int(transport_seed),
        "histories_per_energy": int(histories_per_energy),
        "phase_strata": int(impact_edges.size - 1),
        "cells": cells,
        "runtime_config_sha256": runtime_config_sha256,
        "native_executable_sha256": str(provenance[0]),
        "native_execution_environment_sha256": str(provenance[1]),
        "detector_implementation_bundle_sha256": str(provenance[2]),
        "detector_model_sha256": detector_model_sha256,
        "geant4_physics_contract_sha256": GEANT4_PHYSICS_CONTRACT_SHA256,
        "energy_resolution_contract_sha256": (
            DETECTOR_ENERGY_RESOLUTION_CONTRACT_SHA256
        ),
        "reference_scene_sha256": scene.scene_hash,
        "reference_source_contract_sha256": _source_contract_sha256(scene.sources[0]),
    }


def _publish(
    *,
    output_root: Path,
    raw_corpus: Mapping[str, object],
    operator: object,
) -> tuple[Path, Path]:
    """Publish raw evidence and the operator with one atomic root rename."""
    destination = output_root.resolve()
    if destination.exists():
        raise FileExistsError(
            f"Detector Green construction requires a new output root: {destination}."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging.",
            dir=destination.parent,
        )
    )
    try:
        raw_path = staging / DETECTOR_GREEN_RAW_CORPUS_BASENAME
        raw_path.write_bytes(canonical_json_bytes(dict(raw_corpus)))
        manifest_path = operator.write_artifact(staging / "operator")
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return (
        destination / DETECTOR_GREEN_RAW_CORPUS_BASENAME,
        destination / "operator" / manifest_path.name,
    )


def run_detector_green_construction(
    *,
    runtime_config_path: str | Path,
    output_root: str | Path,
    repository_root: str | Path,
    histories_per_energy: int = DETECTOR_GREEN_MINIMUM_HISTORIES_PER_ENERGY,
    impact_strata: int = DETECTOR_GREEN_DEFAULT_IMPACT_STRATA,
    transport_seed: int | None = None,
) -> tuple[Path, Path]:
    """Acquire and atomically publish a formal detector Green operator."""
    if (
        isinstance(histories_per_energy, bool)
        or not isinstance(histories_per_energy, int)
        or histories_per_energy < DETECTOR_GREEN_MINIMUM_HISTORIES_PER_ENERGY
        or isinstance(impact_strata, bool)
        or not isinstance(impact_strata, int)
        or impact_strata <= 0
        or histories_per_energy % impact_strata != 0
        or histories_per_energy // impact_strata < 2
    ):
        raise ValueError(
            "Detector Green construction requires at least 100000 histories "
            "per energy, positive impact strata, and at least two histories "
            "per stratum."
        )
    if transport_seed is None:
        resolved_seed = secrets.randbelow(2_147_483_646) + 1
    elif (
        isinstance(transport_seed, bool)
        or not isinstance(transport_seed, int)
        or transport_seed <= 0
        or transport_seed >= 2_147_483_647
    ):
        raise ValueError("transport_seed must be a positive signed 32-bit integer.")
    else:
        resolved_seed = int(transport_seed)
    repository = Path(repository_root).resolve()
    config_path = Path(runtime_config_path).resolve()
    destination = Path(output_root).resolve()
    if destination.exists():
        raise FileExistsError(
            f"Detector Green construction requires a new output root: {destination}."
        )
    nodes = catalog_independent_energy_nodes_keV()
    library = _probe_library(nodes)
    runtime_config, runtime_config_sha256 = (
        load_production_runtime_config_with_digest(config_path)
    )
    app_payload = _build_app_payload(
        runtime_config,
        repository_root=repository,
        histories_per_energy=histories_per_energy,
        impact_strata=impact_strata,
    )
    app = Geant4Application(
        app_config=app_payload,
        production_runtime_config_sha256=runtime_config_sha256,
        offline_nuclide_library=library,
    )
    scene = _reference_scene(app.config)
    try:
        app.engine.load_scene(scene)
        _, raw_metadata = app.engine.simulate(
            _reference_request(transport_seed=resolved_seed)
        )
        metadata = dict(raw_metadata)
        validate_mean_calibration_transport_metadata(
            metadata,
            expected_histories_per_source_line=histories_per_energy,
            expected_angle_strata_mu=impact_strata,
            expected_angle_strata_phi=1,
            expected_forced_collision=False,
            expected_source_rate_model="detector_cps_1m",
            expected_thread_count=app.config.thread_count,
            expected_physics_profile=app.config.physics_profile,
            expected_detector_scoring_mode="full_transport",
            expected_secondary_transport_mode="full_transport",
            expected_source_bias_mode="detector_cone",
            expected_surface_source_contract_sha256=(
                _source_contract_sha256(scene.sources[0])
            ),
            expected_scene_hash=scene.scene_hash,
        )
        raw_histograms, sampled_histories = _dense_corpus_arrays(
            metadata,
            energy_nodes_keV=nodes,
            impact_strata=impact_strata,
            histories_per_energy=histories_per_energy,
        )
        impact_edges = impact_parameter_edges_for_equal_solid_angle_strata(
            source_distance_m=DETECTOR_GREEN_REFERENCE_SOURCE_DISTANCE_M,
            detector_target_radius_m=(
                app.config.detector_model.crystal_radius_m
                + app.config.detector_model.housing_thickness_m
            ),
            stratum_count=impact_strata,
        )
        raw_corpus = _sparse_raw_corpus(
            energy_nodes_keV=nodes,
            impact_edges=impact_edges,
            raw_histograms=raw_histograms,
            sampled_histories=sampled_histories,
            runtime_config_sha256=runtime_config_sha256,
            app=app,
            scene=scene,
            transport_seed=resolved_seed,
            histories_per_energy=histories_per_energy,
        )
    finally:
        app.close()
    expected_implementation = detector_green_implementation_bundle_sha256(repository)
    if (
        raw_corpus["detector_implementation_bundle_sha256"]
        != expected_implementation
    ):
        raise RuntimeError("Detector Green implementation changed during acquisition.")
    construction = {
        "method": "native_geant4_monoenergetic_full_detector",
        "raw_corpus_sha256": detector_green_raw_corpus_sha256(raw_corpus),
        "native_executable_sha256": raw_corpus["native_executable_sha256"],
        "native_execution_environment_sha256": raw_corpus[
            "native_execution_environment_sha256"
        ],
        "detector_implementation_bundle_sha256": raw_corpus[
            "detector_implementation_bundle_sha256"
        ],
        "detector_model_sha256": raw_corpus["detector_model_sha256"],
        "geant4_physics_contract_sha256": (
            raw_corpus["geant4_physics_contract_sha256"]
        ),
        "energy_resolution_contract_sha256": (
            raw_corpus["energy_resolution_contract_sha256"]
        ),
        "construction_seed": resolved_seed,
        "histories_per_energy": histories_per_energy,
        "energy_node_design": (
            "catalog_independent_deterministic_continuous_domain_v1"
        ),
        "phase_strata": impact_strata,
        "detector_target_radius_m": (
            app.config.detector_model.crystal_radius_m
            + app.config.detector_model.housing_thickness_m
        ),
        "completed": True,
    }
    operator = build_detector_green_operator(
        energy_nodes_keV=nodes,
        impact_parameter_edges_fraction=impact_edges,
        raw_deposit_histograms_ncb=raw_histograms,
        sampled_histories_nc=sampled_histories,
        construction=construction,
        output_bin_width_keV=NATIVE_GEANT4_BIN_WIDTH_KEV,
    )
    return _publish(
        output_root=destination,
        raw_corpus=raw_corpus,
        operator=operator,
    )


__all__ = [
    "DETECTOR_GREEN_MINIMUM_HISTORIES_PER_ENERGY",
    "run_detector_green_construction",
]
