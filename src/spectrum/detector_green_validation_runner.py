"""Acquire an independent monoenergetic detector Green holdout."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import secrets
import shutil
import tempfile

import numpy as np

from sim.geant4_app.app import (
    Geant4Application,
    validate_mean_calibration_transport_metadata,
)
from sim.runtime import load_production_runtime_config_with_digest
from spectrum.detector_green_construction import (
    impact_parameter_edges_for_equal_solid_angle_strata,
)
from spectrum.detector_green_construction_runner import (
    DETECTOR_GREEN_REFERENCE_SOURCE_DISTANCE_M,
    _build_app_payload,
    _dense_corpus_arrays,
    _probe_library,
    _reference_request,
    _reference_scene,
    _source_contract_sha256,
)
from spectrum.detector_green_operator import (
    DetectorGreenOperator,
    canonical_json_bytes,
)
from spectrum.detector_green_validation import (
    DETECTOR_GREEN_VALIDATION_CONTRACT_SHA256,
    DETECTOR_GREEN_VALIDATION_MANIFEST_BASENAME,
    DETECTOR_GREEN_VALIDATION_MINIMUM_HISTORIES_PER_ENERGY,
    DETECTOR_GREEN_VALIDATION_ID,
    DETECTOR_GREEN_VALIDATION_RAW_BASENAME,
    DETECTOR_GREEN_VALIDATION_RAW_SCHEMA_VERSION,
    build_detector_green_validation_manifest,
    detector_green_holdout_energies_keV,
    load_detector_green_validation_manifest,
    validate_detector_green_validation_raw_corpus,
)
from spectrum.detector_green_provenance import (
    detector_green_implementation_bundle_sha256,
)
from spectrum.geant4_physics import GEANT4_PHYSICS_CONTRACT_SHA256
from spectrum.detector_green_construction import (
    DETECTOR_ENERGY_RESOLUTION_CONTRACT_SHA256,
)
from spectrum.response_matrix import (
    NATIVE_GEANT4_BIN_COUNT,
    NATIVE_GEANT4_BIN_WIDTH_KEV,
)


def _new_seed(excluded: set[int]) -> int:
    """Return a fresh positive signed 32-bit seed outside one exclusion set."""
    for _ in range(1_024):
        value = secrets.randbelow(2_147_483_646) + 1
        if value not in excluded:
            return value
    raise RuntimeError("Could not allocate a distinct detector validation seed.")


def _resolve_seed(value: int | None, *, field_name: str) -> int | None:
    """Validate one optional positive signed 32-bit seed."""
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value >= 2_147_483_647
    ):
        raise ValueError(f"{field_name} must be a positive signed 32-bit integer.")
    return int(value)


def _raw_corpus(
    *,
    operator: DetectorGreenOperator,
    energies_keV: np.ndarray,
    raw_histograms_ecb: np.ndarray,
    sampled_histories_ec: np.ndarray,
    design_seed: int,
    transport_seed: int,
    histories_per_energy: int,
    runtime_config_sha256: str,
    app: Geant4Application,
    scene: object,
) -> dict[str, object]:
    """Return one compact isotope-free validation corpus."""
    construction = operator.construction
    if construction is None:
        raise RuntimeError("Detector Green construction provenance is absent.")
    cells: list[dict[str, object]] = []
    for energy_index in range(energies_keV.size):
        for impact_index in range(raw_histograms_ecb.shape[1]):
            histogram = raw_histograms_ecb[energy_index, impact_index]
            cells.append(
                {
                    "energy_index": energy_index,
                    "impact_bin_index": impact_index,
                    "sampled_histories": int(
                        sampled_histories_ec[energy_index, impact_index]
                    ),
                    "registered_pulses": int(np.sum(histogram)),
                    "sparse_raw_deposit_histogram": [
                        [int(index), int(histogram[index])]
                        for index in np.flatnonzero(histogram)
                    ],
                }
            )
    provenance = (
        app.native_executable_sha256,
        app.native_execution_environment_sha256,
        app.detector_implementation_bundle_sha256,
    )
    if any(value is None for value in provenance):
        raise RuntimeError("Detector Green validation provenance is incomplete.")
    detector_model_sha256 = hashlib.sha256(
        canonical_json_bytes(app.config.detector_model.to_dict())
    ).hexdigest()
    return {
        "schema_version": DETECTOR_GREEN_VALIDATION_RAW_SCHEMA_VERSION,
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
        "holdout_energies_keV": energies_keV.tolist(),
        "output_energy_axis": {
            "minimum_keV": 0.0,
            "bin_width_keV": NATIVE_GEANT4_BIN_WIDTH_KEV,
            "bin_count": NATIVE_GEANT4_BIN_COUNT,
        },
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
    corpus: dict[str, object],
    manifest: dict[str, object],
    operator: DetectorGreenOperator,
) -> tuple[Path, Path]:
    """Atomically publish and read back validation evidence."""
    destination = output_root.resolve()
    if destination.exists():
        raise FileExistsError(
            f"Detector Green validation output already exists: {destination}."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging.",
            dir=destination.parent,
        )
    )
    try:
        (staging / DETECTOR_GREEN_VALIDATION_RAW_BASENAME).write_bytes(
            canonical_json_bytes(corpus)
        )
        (staging / DETECTOR_GREEN_VALIDATION_MANIFEST_BASENAME).write_bytes(
            canonical_json_bytes(manifest)
        )
        load_detector_green_validation_manifest(
            staging / DETECTOR_GREEN_VALIDATION_MANIFEST_BASENAME,
            operator=operator,
            require_passed=False,
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return (
        destination / DETECTOR_GREEN_VALIDATION_RAW_BASENAME,
        destination / DETECTOR_GREEN_VALIDATION_MANIFEST_BASENAME,
    )


def run_detector_green_validation(
    *,
    runtime_config_path: str | Path,
    operator_manifest_path: str | Path,
    output_root: str | Path,
    repository_root: str | Path,
    histories_per_energy: int = (
        DETECTOR_GREEN_VALIDATION_MINIMUM_HISTORIES_PER_ENERGY
    ),
    design_seed: int | None = None,
    transport_seed: int | None = None,
) -> tuple[Path, Path]:
    """Acquire and atomically publish an independent detector holdout."""
    if (
        isinstance(histories_per_energy, bool)
        or not isinstance(histories_per_energy, int)
        or histories_per_energy < DETECTOR_GREEN_VALIDATION_MINIMUM_HISTORIES_PER_ENERGY
    ):
        raise ValueError(
            "Detector Green validation requires at least 100000 histories per energy."
        )
    operator = DetectorGreenOperator.from_artifact(operator_manifest_path)
    operator.require_runtime_ready()
    construction = operator.construction
    if construction is None:
        raise RuntimeError("Detector Green construction provenance is absent.")
    impact_strata = int(construction["phase_strata"])
    if histories_per_energy % impact_strata != 0:
        raise ValueError("Validation histories must divide over impact strata.")
    excluded = {int(construction["construction_seed"])}
    resolved_design = _resolve_seed(design_seed, field_name="design_seed")
    if resolved_design is None:
        resolved_design = _new_seed(excluded)
    excluded.add(resolved_design)
    resolved_transport = _resolve_seed(
        transport_seed,
        field_name="transport_seed",
    )
    if resolved_transport is None:
        resolved_transport = _new_seed(excluded)
    if resolved_transport in excluded:
        raise ValueError("Construction, design, and transport seeds must differ.")
    energies = detector_green_holdout_energies_keV(
        resolved_design,
        operator=operator,
    )
    schedule = np.concatenate((np.asarray((0.0,)), energies))
    library = _probe_library(schedule)
    repository = Path(repository_root).resolve()
    config_path = Path(runtime_config_path).resolve()
    destination = Path(output_root).resolve()
    if destination.exists():
        raise FileExistsError(
            f"Detector Green validation output already exists: {destination}."
        )
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
        current_provenance = {
            "native_executable_sha256": app.native_executable_sha256,
            "native_execution_environment_sha256": (
                app.native_execution_environment_sha256
            ),
            "detector_implementation_bundle_sha256": (
                app.detector_implementation_bundle_sha256
            ),
        }
        for field_name, current_value in current_provenance.items():
            if construction[field_name] != current_value:
                raise RuntimeError(
                    "Detector Green validation implementation differs from "
                    f"construction for {field_name}."
                )
        expected_edges = impact_parameter_edges_for_equal_solid_angle_strata(
            source_distance_m=DETECTOR_GREEN_REFERENCE_SOURCE_DISTANCE_M,
            detector_target_radius_m=(
                app.config.detector_model.crystal_radius_m
                + app.config.detector_model.housing_thickness_m
            ),
            stratum_count=impact_strata,
        )
        if not np.array_equal(
            expected_edges,
            operator.impact_parameter_edges_fraction,
        ):
            raise RuntimeError(
                "Detector Green validation phase geometry differs from construction."
            )
        app.engine.load_scene(scene)
        _, raw_metadata = app.engine.simulate(
            _reference_request(transport_seed=resolved_transport)
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
        raw_with_zero, sampled_with_zero = _dense_corpus_arrays(
            metadata,
            energy_nodes_keV=schedule,
            impact_strata=impact_strata,
            histories_per_energy=histories_per_energy,
        )
        corpus = _raw_corpus(
            operator=operator,
            energies_keV=energies,
            raw_histograms_ecb=raw_with_zero[1:],
            sampled_histories_ec=sampled_with_zero[1:],
            design_seed=resolved_design,
            transport_seed=resolved_transport,
            histories_per_energy=histories_per_energy,
            runtime_config_sha256=runtime_config_sha256,
            app=app,
            scene=scene,
        )
    finally:
        app.close()
    if (
        detector_green_implementation_bundle_sha256(repository)
        != corpus["detector_implementation_bundle_sha256"]
    ):
        raise RuntimeError("Detector Green implementation changed during holdout.")
    validated = validate_detector_green_validation_raw_corpus(
        corpus,
        operator=operator,
    )
    manifest = build_detector_green_validation_manifest(
        validated,
        operator=operator,
    )
    return _publish(
        output_root=destination,
        corpus=validated,
        manifest=manifest,
        operator=operator,
    )


__all__ = ["run_detector_green_validation"]
