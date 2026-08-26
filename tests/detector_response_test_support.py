"""Test-only builders for authenticated detector-response validation corpora."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from runtime.experiment_profiles import STANDARD_ACQUISITION_LIVE_TIME_S
from spectrum.detector_response_validation import (
    DETECTOR_RESPONSE_MINIMUM_HISTORIES_PER_ENERGY,
    DETECTOR_RESPONSE_RAW_CORPUS_BASENAME,
    DETECTOR_RESPONSE_RAW_CORPUS_SCHEMA_VERSION,
    DETECTOR_RESPONSE_REFERENCE_GEOMETRY_SHA256,
    DETECTOR_RESPONSE_VALIDATION_CONTRACT_ID,
    DETECTOR_RESPONSE_VALIDATION_CONTRACT_SHA256,
    build_detector_response_validation_manifest,
    canonical_json_bytes,
)
from spectrum.detector_response_validation_runner import (
    DETECTOR_RESPONSE_VALIDATION_MANIFEST_BASENAME,
)
from spectrum.geant4_physics import (
    GEANT4_PHYSICS_CONTRACT_SHA256,
    geant4_physics_contract_payload,
)
from spectrum.library import default_library
from spectrum.native_metadata import native_source_line_token
from spectrum.response_matrix import (
    NATIVE_GEANT4_BIN_COUNT,
    NATIVE_GEANT4_BIN_WIDTH_KEV,
    NATIVE_GEANT4_DETECTOR_RESPONSE_CONTRACT_SHA256,
    build_native_geant4_detector_response_matrix,
)


def passing_detector_response_raw_corpus(
    *,
    native_executable_sha256: str = "b" * 64,
    native_execution_environment_sha256: str = "c" * 64,
    implementation_bundle_sha256: str = "d" * 64,
    runtime_config_sha256: str = "a" * 64,
) -> dict[str, object]:
    """Return a physically shaped synthetic corpus equal to the candidate."""
    axis = (
        np.arange(NATIVE_GEANT4_BIN_COUNT, dtype=np.float64)
        * NATIVE_GEANT4_BIN_WIDTH_KEV
    )
    response = build_native_geant4_detector_response_matrix(
        axis,
        NATIVE_GEANT4_BIN_WIDTH_KEV,
    )
    histories = DETECTOR_RESPONSE_MINIMUM_HISTORIES_PER_ENERGY
    library = default_library()
    descriptors = sorted(
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
    )
    physics = geant4_physics_contract_payload()
    return {
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
        "native_execution_environment_sha256": (
            native_execution_environment_sha256
        ),
        "implementation_bundle_sha256": implementation_bundle_sha256,
        "reference_scene_sha256": "1" * 64,
        "reference_source_contract_sha256": "2" * 64,
        "reference_detector_model_sha256": "3" * 64,
        "reference_detector_scoring_mode": "full_transport",
        "candidate_detector_scoring_mode": "incident_gamma_energy",
        "dwell_time_s": STANDARD_ACQUISITION_LIVE_TIME_S,
        "transport_seed": 123_456_789,
        "histories_per_energy": histories,
        "energy_axis_keV": axis.tolist(),
        "line_corpora": [
            {
                "energy_keV": energy,
                "isotope": isotope,
                "source_index": source_index,
                "line_token": line_token,
                "sampled_histories": histories,
                "history_weight": 1.0,
                "observed_weighted_spectrum": (
                    response[
                        :,
                        int(energy // NATIVE_GEANT4_BIN_WIDTH_KEV),
                    ]
                    * histories
                ).tolist(),
                "pulse_count_weighted": float(histories),
            }
            for energy, isotope, source_index, line_token in descriptors
        ],
        "native_physics_provenance": {
            "geant4_physics_contract_id": physics["contract_id"],
            "geant4_physics_contract_sha256": (
                GEANT4_PHYSICS_CONTRACT_SHA256
            ),
            **{
                key: value
                for key, value in physics.items()
                if key != "contract_id"
            },
        },
    }


def write_passing_detector_response_validation(
    directory: Path,
    *,
    native_executable_sha256: str = "b" * 64,
    native_execution_environment_sha256: str = "c" * 64,
    implementation_bundle_sha256: str = "d" * 64,
    runtime_config_sha256: str = "a" * 64,
) -> Path:
    """Write one canonical passing raw corpus and its derived manifest."""
    directory.mkdir(parents=True, exist_ok=False)
    corpus = passing_detector_response_raw_corpus(
        native_executable_sha256=native_executable_sha256,
        native_execution_environment_sha256=(
            native_execution_environment_sha256
        ),
        implementation_bundle_sha256=implementation_bundle_sha256,
        runtime_config_sha256=runtime_config_sha256,
    )
    manifest = build_detector_response_validation_manifest(corpus)
    (directory / DETECTOR_RESPONSE_RAW_CORPUS_BASENAME).write_bytes(
        canonical_json_bytes(corpus)
    )
    manifest_path = directory / DETECTOR_RESPONSE_VALIDATION_MANIFEST_BASENAME
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    return manifest_path


__all__ = [
    "passing_detector_response_raw_corpus",
    "write_passing_detector_response_validation",
]
