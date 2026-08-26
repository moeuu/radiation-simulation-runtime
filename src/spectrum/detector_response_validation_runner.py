"""Acquire a shield-free native corpus for detector-response validation."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import tempfile
from collections.abc import Mapping

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
from sim.runtime import load_runtime_config
from spectrum.detector_response_validation import (
    DETECTOR_RESPONSE_MINIMUM_HISTORIES_PER_ENERGY,
    DETECTOR_RESPONSE_RAW_CORPUS_BASENAME,
    DETECTOR_RESPONSE_RAW_CORPUS_SCHEMA_VERSION,
    DETECTOR_RESPONSE_REFERENCE_GEOMETRY_SHA256,
    DETECTOR_RESPONSE_REFERENCE_SOURCE_DISTANCE_M,
    DETECTOR_RESPONSE_VALIDATION_CONTRACT_ID,
    DETECTOR_RESPONSE_VALIDATION_CONTRACT_SHA256,
    build_detector_response_validation_manifest,
    canonical_json_bytes,
    validate_detector_response_raw_corpus,
    validate_detector_response_validation_manifest,
)
from spectrum.full_spectrum_acceptance_runner import (
    acceptance_implementation_bundle_sha256,
)
from spectrum.geant4_acceptance_backend import _FULL_SPECTRUM_RUNTIME_KEYS
from spectrum.geant4_physics import (
    GEANT4_PHYSICS_CONTRACT_SHA256,
    geant4_physics_contract_payload,
)
from spectrum.library import default_library
from spectrum.mean_calibration import parse_mean_calibration_metadata
from spectrum.native_metadata import native_source_line_token
from spectrum.response_matrix import (
    NATIVE_GEANT4_BIN_COUNT,
    NATIVE_GEANT4_BIN_WIDTH_KEV,
    NATIVE_GEANT4_DETECTOR_RESPONSE_CONTRACT_SHA256,
)


DETECTOR_RESPONSE_VALIDATION_MANIFEST_BASENAME = "validation_manifest.json"
_REFERENCE_DETECTOR_POSE_XYZ = (1.0, 1.0, 0.5)
_ENTRY_CLASSES = (
    "uncollided_primary",
    "interacted_primary",
    "secondary",
)


def _file_sha256(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _build_app_payload(
    runtime_config: Mapping[str, object],
    *,
    repository_root: Path,
    histories_per_energy: int,
) -> dict[str, object]:
    """Build the isolated fixed-line full-detector application payload."""
    payload = {
        key: value
        for key, value in dict(runtime_config).items()
        if key not in _FULL_SPECTRUM_RUNTIME_KEYS
    }
    executable_raw = payload.get("executable_path")
    if not isinstance(executable_raw, str) or not executable_raw:
        raise ValueError(
            "Detector-response validation requires explicit executable_path."
        )
    executable = Path(executable_raw)
    if not executable.is_absolute():
        executable = repository_root / executable
    payload.update(
        {
            "executable_path": executable.resolve().as_posix(),
            "engine_mode": "external",
            "source_rate_model": "detector_cps_1m",
            "detector_scoring_mode": "full_transport",
            "secondary_transport_mode": "full_transport",
            "sample_detector_response": False,
            "validation_entry_class_spectra": True,
            "background_cps": 0.0,
            "dead_time_tau_s": 0.0,
            "primary_sampling_fraction": 1.0,
            "target_sampled_primaries": None,
            "accelerated_weighted_transport_enable": False,
            "mean_calibration_histories_per_source_line": int(
                histories_per_energy
            ),
            "mean_calibration_angle_strata_mu": 1,
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
        or config.mean_calibration_histories_per_source_line
        != histories_per_energy
        or config.mean_calibration_forced_collision
    ):
        raise RuntimeError("Detector-response application contract drifted.")
    return payload


def _reference_sources() -> tuple[ExportedGeant4Source, ...]:
    """Return three co-located known-nuclide sources for all formal lines."""
    detector = np.asarray(_REFERENCE_DETECTOR_POSE_XYZ, dtype=np.float64)
    normal = np.asarray((-1.0, 0.0, 0.0), dtype=np.float64)
    transport = detector + np.asarray(
        (DETECTOR_RESPONSE_REFERENCE_SOURCE_DISTANCE_M, 0.0, 0.0),
        dtype=np.float64,
    )
    anchor = transport - SURFACE_EMISSION_EPSILON_M * normal
    policy_hash = surface_emission_policy_sha256()
    return tuple(
        ExportedGeant4Source(
            isotope=isotope,
            position_xyz=tuple(float(value) for value in transport),
            intensity_cps_1m=1.0,
            anchor_position_xyz=tuple(float(value) for value in anchor),
            surface_chart_id=source_index,
            surface_uv=(0.25, 0.25 + 0.25 * source_index),
            surface_normal_xyz=tuple(float(value) for value in normal),
            surface_emission_policy_sha256=policy_hash,
        )
        for source_index, isotope in enumerate(("Cs-137", "Co-60", "Eu-154"))
    )


def _source_contract_sha256(
    sources: tuple[ExportedGeant4Source, ...],
) -> str:
    """Return the exact shared source-surface contract for the reference scene."""
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
            for source in sources
        ]
    )


def _reference_scene(config: Geant4AppConfig) -> ExportedGeant4Scene:
    """Build a detector-only scene with no wall, obstacle, Fe, or Pb volume."""
    provisional = ExportedGeant4Scene(
        scene_hash="",
        usd_path=None,
        room_size_xyz=(2.0, 2.0, 2.0),
        static_volumes=(),
        sources=_reference_sources(),
        detector_model=config.detector_model,
        fe_shield=None,
        pb_shield=None,
        prim_paths=StagePrimPaths(),
    )
    stable_payload = provisional.to_dict()
    stable_payload.pop("scene_hash")
    scene_hash = hashlib.sha256(
        json.dumps(stable_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return replace(provisional, scene_hash=scene_hash)


def _orientation_zero_quaternion_wxyz() -> tuple[float, float, float, float]:
    """Return the exact orientation-zero octant placement quaternion."""
    root_half = math.sqrt(0.5)
    return (0.0, root_half, -root_half, 0.0)


def _reference_request(*, transport_seed: int) -> Geant4StepRequest:
    """Return one 20-second fixed-pose request for the detector-only scene."""
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


def _metadata_vector(metadata: Mapping[str, object], key: str) -> np.ndarray:
    """Parse one finite nonnegative native comma-separated response vector."""
    raw = metadata.get(key)
    if not isinstance(raw, str):
        raise RuntimeError(f"Native detector-response metadata {key} is missing.")
    try:
        vector = np.asarray(
            [float(value) for value in raw.split(",")],
            dtype=np.float64,
        )
    except ValueError as exc:
        raise RuntimeError(
            f"Native detector-response metadata {key} is not numeric."
        ) from exc
    if (
        vector.shape != (NATIVE_GEANT4_BIN_COUNT,)
        or np.any(~np.isfinite(vector))
        or np.any(vector < 0.0)
    ):
        raise RuntimeError(
            f"Native detector-response metadata {key} is invalid."
        )
    return vector


def _line_descriptors() -> tuple[tuple[float, str, int, str], ...]:
    """Return the exact sorted formal line identities used by the native scene."""
    library = default_library()
    descriptors = [
        (
            float(line.energy_keV),
            isotope,
            source_index,
            native_source_line_token(
                source_index=source_index,
                isotope=isotope,
                energy_keV=float(line.energy_keV),
            ),
        )
        for source_index, isotope in enumerate(("Cs-137", "Co-60", "Eu-154"))
        for line in library[isotope].lines
        if float(line.intensity) > 0.0
    ]
    return tuple(sorted(descriptors))


def _line_corpora(
    metadata: Mapping[str, object],
    *,
    histories_per_energy: int,
) -> list[dict[str, object]]:
    """Extract per-line observed detector spectra and fixed-quota provenance."""
    calibration = parse_mean_calibration_metadata(
        metadata,
        bin_count=NATIVE_GEANT4_BIN_COUNT,
    )
    batches_by_line: dict[str, list[object]] = {}
    for batch in calibration.batches:
        batches_by_line.setdefault(batch.line_token, []).append(batch)
    expected_tokens = {descriptor[3] for descriptor in _line_descriptors()}
    if set(batches_by_line) != expected_tokens or any(
        len(batches) != 1 for batches in batches_by_line.values()
    ):
        raise RuntimeError(
            "Native detector-response fixed-quota line schedule is incomplete."
        )
    prefix = "validation_only_observed_entry_spectrum_"
    allowed_keys = {
        prefix + token + "_" + entry_class
        for token in expected_tokens
        for entry_class in _ENTRY_CLASSES
    }
    unexpected = {
        key
        for key in metadata
        if isinstance(key, str)
        and key.startswith(prefix)
        and key not in allowed_keys
    }
    if unexpected:
        raise RuntimeError(
            "Native detector-response metadata contains unexpected line labels: "
            f"{sorted(unexpected)}."
        )
    result: list[dict[str, object]] = []
    for energy, isotope, source_index, line_token in _line_descriptors():
        batch = batches_by_line[line_token][0]
        if batch.sampled_histories != histories_per_energy:
            raise RuntimeError("Native detector-response line quota drifted.")
        spectrum = np.zeros(NATIVE_GEANT4_BIN_COUNT, dtype=np.float64)
        for entry_class in _ENTRY_CLASSES:
            key = prefix + line_token + "_" + entry_class
            if key in metadata:
                spectrum += _metadata_vector(metadata, key)
        result.append(
            {
                "energy_keV": energy,
                "isotope": isotope,
                "source_index": source_index,
                "line_token": line_token,
                "sampled_histories": int(batch.sampled_histories),
                "history_weight": float(batch.history_weight),
                "observed_weighted_spectrum": spectrum.tolist(),
                "pulse_count_weighted": float(np.sum(spectrum)),
            }
        )
    return result


def _physics_provenance() -> dict[str, object]:
    """Return the exact Geant4 physics fields already checked in native metadata."""
    payload = geant4_physics_contract_payload()
    return {
        "geant4_physics_contract_id": payload["contract_id"],
        "geant4_physics_contract_sha256": GEANT4_PHYSICS_CONTRACT_SHA256,
        **{
            key: value
            for key, value in payload.items()
            if key != "contract_id"
        },
    }


def _raw_corpus(
    *,
    metadata: Mapping[str, object],
    scene: ExportedGeant4Scene,
    app: Geant4Application,
    runtime_config_sha256: str,
    transport_seed: int,
    histories_per_energy: int,
) -> dict[str, object]:
    """Build the canonical raw response corpus from native evidence."""
    native_executable_sha256 = app.native_executable_sha256
    native_environment_sha256 = app.native_execution_environment_sha256
    implementation_bundle_sha256 = app.implementation_bundle_sha256
    if any(
        value is None
        for value in (
            native_executable_sha256,
            native_environment_sha256,
            implementation_bundle_sha256,
        )
    ):
        raise RuntimeError("Detector-response native provenance is incomplete.")
    detector_model_sha256 = hashlib.sha256(
        canonical_json_bytes(scene.detector_model.to_dict())
    ).hexdigest()
    payload = {
        "schema_version": DETECTOR_RESPONSE_RAW_CORPUS_SCHEMA_VERSION,
        "validation_contract_id": DETECTOR_RESPONSE_VALIDATION_CONTRACT_ID,
        "validation_contract_sha256": (
            DETECTOR_RESPONSE_VALIDATION_CONTRACT_SHA256
        ),
        "detector_response_contract_sha256": (
            NATIVE_GEANT4_DETECTOR_RESPONSE_CONTRACT_SHA256
        ),
        "geant4_physics_contract_sha256": GEANT4_PHYSICS_CONTRACT_SHA256,
        "reference_geometry_sha256": (
            DETECTOR_RESPONSE_REFERENCE_GEOMETRY_SHA256
        ),
        "runtime_config_sha256": runtime_config_sha256,
        "native_executable_sha256": native_executable_sha256,
        "native_execution_environment_sha256": native_environment_sha256,
        "implementation_bundle_sha256": implementation_bundle_sha256,
        "reference_scene_sha256": scene.scene_hash,
        "reference_source_contract_sha256": _source_contract_sha256(
            scene.sources
        ),
        "reference_detector_model_sha256": detector_model_sha256,
        "reference_detector_scoring_mode": "full_transport",
        "candidate_detector_scoring_mode": "incident_gamma_energy",
        "dwell_time_s": STANDARD_ACQUISITION_LIVE_TIME_S,
        "transport_seed": int(transport_seed),
        "histories_per_energy": int(histories_per_energy),
        "energy_axis_keV": (
            np.arange(NATIVE_GEANT4_BIN_COUNT, dtype=np.float64)
            * NATIVE_GEANT4_BIN_WIDTH_KEV
        ).tolist(),
        "line_corpora": _line_corpora(
            metadata,
            histories_per_energy=histories_per_energy,
        ),
        "native_physics_provenance": _physics_provenance(),
    }
    return validate_detector_response_raw_corpus(payload)


def _write_atomic_artifact_directory(
    *,
    output_root: Path,
    raw_corpus: Mapping[str, object],
) -> tuple[Path, Path]:
    """Publish raw and derived artifacts with one atomic directory rename."""
    destination = output_root.resolve()
    if destination.exists():
        raise FileExistsError(
            "Detector-response validation requires a new empty output root: "
            f"{destination}."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging.",
            dir=destination.parent,
        )
    )
    try:
        corpus = validate_detector_response_raw_corpus(raw_corpus)
        manifest = build_detector_response_validation_manifest(corpus)
        validate_detector_response_validation_manifest(
            manifest,
            require_passed=False,
        )
        raw_path = temporary / DETECTOR_RESPONSE_RAW_CORPUS_BASENAME
        manifest_path = (
            temporary / DETECTOR_RESPONSE_VALIDATION_MANIFEST_BASENAME
        )
        raw_path.write_bytes(canonical_json_bytes(corpus))
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return (
        destination / DETECTOR_RESPONSE_RAW_CORPUS_BASENAME,
        destination / DETECTOR_RESPONSE_VALIDATION_MANIFEST_BASENAME,
    )


def run_detector_response_validation(
    *,
    runtime_config_path: str | Path,
    output_root: str | Path,
    repository_root: str | Path,
    histories_per_energy: int = (
        DETECTOR_RESPONSE_MINIMUM_HISTORIES_PER_ENERGY
    ),
    transport_seed: int | None = None,
) -> tuple[Path, Path]:
    """Acquire, evaluate, and atomically publish formal detector validation."""
    if (
        isinstance(histories_per_energy, bool)
        or not isinstance(histories_per_energy, int)
        or histories_per_energy
        < DETECTOR_RESPONSE_MINIMUM_HISTORIES_PER_ENERGY
    ):
        raise ValueError(
            "Formal detector-response validation requires at least "
            f"{DETECTOR_RESPONSE_MINIMUM_HISTORIES_PER_ENERGY} histories "
            "per energy."
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
            "Detector-response validation requires a new empty output root: "
            f"{destination}."
        )
    runtime_config = load_runtime_config(config_path)
    runtime_config_sha256 = _file_sha256(config_path)
    app_payload = _build_app_payload(
        runtime_config,
        repository_root=repository,
        histories_per_energy=histories_per_energy,
    )
    app = Geant4Application(
        app_config=app_payload,
        production_runtime_config_sha256=runtime_config_sha256,
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
            expected_angle_strata_mu=1,
            expected_angle_strata_phi=1,
            expected_forced_collision=False,
            expected_source_rate_model="detector_cps_1m",
            expected_thread_count=app.config.thread_count,
            expected_physics_profile=app.config.physics_profile,
            expected_detector_scoring_mode="full_transport",
            expected_secondary_transport_mode="full_transport",
            expected_source_bias_mode="detector_cone",
            expected_surface_source_contract_sha256=(
                _source_contract_sha256(scene.sources)
            ),
            expected_scene_hash=scene.scene_hash,
        )
        corpus = _raw_corpus(
            metadata=metadata,
            scene=scene,
            app=app,
            runtime_config_sha256=runtime_config_sha256,
            transport_seed=resolved_seed,
            histories_per_energy=histories_per_energy,
        )
    finally:
        app.close()
    expected_implementation = acceptance_implementation_bundle_sha256(
        repository
    )
    if corpus["implementation_bundle_sha256"] != expected_implementation:
        raise RuntimeError(
            "Detector-response implementation changed during acquisition."
        )
    return _write_atomic_artifact_directory(
        output_root=destination,
        raw_corpus=corpus,
    )


__all__ = [
    "DETECTOR_RESPONSE_VALIDATION_MANIFEST_BASENAME",
    "run_detector_response_validation",
]
