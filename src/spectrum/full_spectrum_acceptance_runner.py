"""Resumable real-Geant4 independent acceptance orchestration.

The runner evaluates one immutable physics-only model on predeclared unused
scenes.  Source-resolved detector-entry labels are validation diagnostics only;
this module contains no scene fitting, response calibration, or model-selection
phase.

The module contains no surrogate transport path.  A backend used by the
production CLI must advertise and satisfy the native external-Geant4
postconditions in :data:`NATIVE_ACCEPTANCE_FIDELITY`.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Protocol
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from measurement.geometry_family import validate_geometry_family_descriptor
from measurement.obstacle_assets import room_boundary_transport_components
from measurement.obstacles import (
    ROOM_ABSORBER_TRANSPORT_GROUP,
    ObstacleGrid,
)
from measurement.source_boundary import (
    SURFACE_EMISSION_EPSILON_M,
    canonical_surface_source_runtime_payload,
    surface_emission_policy_sha256,
    surface_source_runtime_contract_sha256,
)
from measurement.shielding import SHIELD_POSE_CONTRACT_SHA256
from runtime.experiment_profiles import (
    STANDARD_ACQUISITION_LIVE_TIME_S,
    STANDARD_OBSTACLE_MATERIAL,
    STANDARD_ROOM_BOUNDARY_THICKNESS_M,
)
from runtime.contracts import FULL_SPECTRUM_MODEL_SCHEMA_VERSION
from spectrum.additive_scatter import (
    ADDITIVE_SCATTER_FEATURE_ORDER,
    ADDITIVE_SCATTER_INCIDENT_LABEL_SEMANTICS,
    ADDITIVE_SCATTER_TARGET_SEMANTICS,
)
from spectrum.air_attenuation import (
    NIST_XCOM_DRY_AIR_TOTAL_CONTRACT_ID,
    NIST_XCOM_DRY_AIR_TOTAL_CONTRACT_SHA256,
)
from spectrum.full_spectrum_acceptance import (
    SURFACE_BOUNDARY_GATE_SCHEMA_VERSION,
)
from spectrum.geant4_physics import (
    GEANT4_EM_PHYSICS_CONSTRUCTOR,
    GEANT4_GAMMA_EM_SUBPROCESS_NAMES,
    GEANT4_GAMMA_PROCESS_NAMES,
    GEANT4_MATERIAL_RESOLUTION_CONTRACT_ID,
    GEANT4_PHYSICS_CONTRACT_ID,
    GEANT4_PHYSICS_CONTRACT_SHA256,
    GEANT4_PRODUCTION_CUT_RANGE_MM,
    GEANT4_REFERENCE_PHYSICS_LIST,
    GEANT4_VERSION_NUMBER,
    GEANT4_VERSION_TAG,
)
from spectrum.native_metadata import native_source_line_token
from spectrum.physics_contracts import (
    OBSTACLE_MATERIAL_CONTRACT_SHA256,
    TRANSPORT_PHYSICS_TABLE_CONTRACT_SHA256,
)
from spectrum.response_matrix import NATIVE_GEANT4_BIN_COUNT
from spectrum.detector_green_operator import (
    DETECTOR_GREEN_COINCIDENCE_SEMANTICS,
    DETECTOR_GREEN_SAMPLING_MODE,
    DetectorGreenOperator,
)
from spectrum.transport_spectral import (
    ACCEPTANCE_DETECTOR_POSE_XYZ,
    ACCEPTANCE_GEOMETRY_DEVICE,
    ACCEPTANCE_GEOMETRY_DTYPE,
    ACCEPTANCE_GEOMETRY_USE_GPU,
    ACCEPTANCE_OBSTACLE_BLOCKED_FRACTION,
    ACCEPTANCE_PASSAGE_WIDTH_M,
    ACCEPTANCE_ROOM_SIZE_XYZ,
    ACCEPTANCE_SURFACE_CHART_MAX_EDGE_M,
    DEFAULT_DETECTOR_GREEN_OPERATOR_MANIFEST,
    DESIGNATED_VALIDATION_SCENE_SEEDS,
    FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256,
    FULL_SPECTRUM_ACCEPTANCE_EXPERIMENT_ID,
    SURFACE_BOUNDARY_PROBE_DWELL_TIME_S,
    TRANSPORT_FEATURE_ORDER,
    VALIDATION_SCENARIO_IDS,
    GeometryConditionedSpectralModel,
)


ACCEPTANCE_RUN_CONTRACT_SCHEMA_VERSION = 13
ACCEPTANCE_PAIR_SCHEMA_VERSION = 8
ACCEPTANCE_SCENE_CORPUS_SCHEMA_VERSION = 2
ACCEPTANCE_ISOTOPES = ("Co-60", "Cs-137")
ACCEPTANCE_PAIR_IDS = tuple(range(64))
ACCEPTANCE_SCENARIO_SOURCE_SPEC = {
    "background_only": (),
    "single_line_source_resolved": (("Cs-137", 800_000.0),),
    "dominant_plus_absent_isotope": (("Cs-137", 1_500_000.0),),
    "multi_isotope_superposition": (
        ("Cs-137", 1_200_000.0),
        ("Co-60", 900_000.0),
    ),
    "continuous_surface_perturbation_ranking": (("Co-60", 900_000.0),),
}
NATIVE_ACCEPTANCE_FIDELITY = {
    "backend": "geant4",
    "engine_mode": "external",
    "physics_profile": "balanced",
    "requested_threads": 32,
    "multithreaded_run_manager": True,
    "source_rate_model": "detector_cps_1m",
    "primary_emission_model": "independent_gamma_lines",
    "emission_model": "detector_equivalent_cone",
    "source_strength_field": "intensity_cps_1m",
    "intensity_cps_1m_definition": "pre_dead_time_detector_pulse_rate_at_1m",
    "true_coincidence_summing": "disabled",
    "radioactive_decay_time_window": "disabled",
    "prompt_decay_cascade_transport": False,
    "delayed_decay_pulse_separation": False,
    "line_intensities_normalized": True,
    "source_bias_mode": "detector_cone",
    "source_bias_cone_policy": "detector_covering",
    "source_bias_isotropic_fraction": 1.0,
    "detector_coincidence_window_s": 1.0e-6,
    "detector_scoring_mode": "incident_gamma_energy",
    "secondary_transport_mode": "full_transport",
    "primary_sampling_fraction": 1.0,
    "primary_history_weight": 1.0,
    "primary_sampling_budget_enabled": False,
    "target_sampled_primaries": 0,
    "history_thinning_enabled": False,
    "transport_history_mode": "full_unit_weight",
    "transport_tally_weighted": False,
    "weighted_transport": False,
    "theory_tvl_attenuation": False,
    "sample_detector_response": True,
    "detector_response_applied_in_native": True,
    "detector_response_sampling_model": (
        "isotope_independent_full_detector_green_operator_v3"
    ),
    "detector_response_sampling_mode": DETECTOR_GREEN_SAMPLING_MODE,
    "detector_response_boundary_state": (
        "normalized_impact_parameter_at_detector_housing_entry_v1"
    ),
    "detector_response_conditioning": (
        "registered_pulse_subprobability_given_housing_incident_gamma_v1"
    ),
    "detector_response_coincidence_semantics": (DETECTOR_GREEN_COINCIDENCE_SEMANTICS),
    "detector_cps_green_reference_normalization": (
        "catalog_branching_weighted_absolute_detection_efficiency_at_1m_v1"
    ),
    "validation_entry_class_spectra": True,
    "validation_entry_spectrum_space": ("pre_dead_time_raw_incident_gamma"),
    "validation_entry_spectrum_grouping": (
        "source_token_initial_gamma_line_entry_class"
    ),
    "spectrum_bin_count": NATIVE_GEANT4_BIN_COUNT,
    "geant4_version_number": GEANT4_VERSION_NUMBER,
    "geant4_version_tag": GEANT4_VERSION_TAG,
    "reference_physics_list": GEANT4_REFERENCE_PHYSICS_LIST,
    "electromagnetic_physics_constructor": (GEANT4_EM_PHYSICS_CONSTRUCTOR),
    "production_cut_range_mm": GEANT4_PRODUCTION_CUT_RANGE_MM,
    "gamma_process_names": GEANT4_GAMMA_PROCESS_NAMES,
    "gamma_em_subprocess_names": GEANT4_GAMMA_EM_SUBPROCESS_NAMES,
    "geant4_physics_contract_id": GEANT4_PHYSICS_CONTRACT_ID,
    "geant4_physics_contract_sha256": GEANT4_PHYSICS_CONTRACT_SHA256,
    "material_resolution_contract_id": (GEANT4_MATERIAL_RESOLUTION_CONTRACT_ID),
}
_PAIR_KEYS = frozenset(
    {
        "schema_version",
        "acceptance_contract_sha256",
        "scene_seed",
        "split",
        "scenario_id",
        "shield_pair_id",
        "transport_seed",
        "dwell_time_s",
        "scene_hash",
        "surface_source_contract_sha256",
        "surface_boundary_gate",
        "detector_pose_xyz",
        "sources",
        "line_identity_contract_sha256",
        "observed_spectrum_counts",
        "geometry",
        "geometry_family",
        "validation_labels",
        "native_fidelity",
        "native_process_diagnostics",
        "detector_green_operator_contract_sha256",
        "detector_green_operator_binary_sha256",
        "shield_pose_contract_sha256",
        "obstacle_material_contract_sha256",
        "transport_physics_table_contract_sha256",
    }
)
_GEOMETRY_KEYS = frozenset(
    {
        "unattenuated_source_line_rate_vsl",
        "uncollided_source_line_rate_vsl",
        "transport_features_vslf",
        "additive_scatter_basis_vslf",
        "perturbed_unattenuated_source_line_rate_vsl",
        "perturbed_uncollided_source_line_rate_vsl",
        "perturbed_transport_features_vslf",
        "perturbed_additive_scatter_basis_vslf",
    }
)
_LABEL_KEYS = frozenset(
    {
        "label_space",
        "target_semantics",
        "entry_class_totals_by_source_line",
        "entry_spectrum_sha256_by_source_line_class",
        "background_entry_total",
        "background_entry_spectrum_sha256",
    }
)
_BOUNDARY_GATE_KEYS = frozenset(
    {
        "schema_version",
        "surface_emission_policy_sha256",
        "surface_emission_epsilon_m",
        "probe_dwell_time_s",
        "native_position_variants",
        "evidence_sha256_by_variant",
        "exact_anchor_vs_air_gate_passed",
        "solid_minus_air_gate_passed",
        "passed",
    }
)
_NATIVE_POSITION_VARIANTS = (
    "exact_surface_anchor",
    "air_plus_epsilon",
    "solid_minus_epsilon",
)
_ACCEPTANCE_TRANSPORT_SEED_DOMAIN = "full_spectrum_all64_native_transport_v3"
ACCEPTANCE_NATIVE_TRANSPORT_SEED_MAX = (1 << 63) - 1


def canonical_json_bytes(payload: object) -> bytes:
    """Return deterministic strict JSON bytes for one artifact."""
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(payload: object) -> str:
    """Return a deterministic strict JSON SHA-256 digest."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def acceptance_transport_seed(
    *,
    scene_seed: int,
    scenario_id: str,
    shield_pair_id: int,
) -> int:
    """Return the fixed independent native transport seed for one pair."""
    if (
        isinstance(scene_seed, bool)
        or not isinstance(scene_seed, int)
        or scene_seed not in DESIGNATED_VALIDATION_SCENE_SEEDS
        or not isinstance(scenario_id, str)
        or scenario_id not in VALIDATION_SCENARIO_IDS
        or isinstance(shield_pair_id, bool)
        or not isinstance(shield_pair_id, int)
        or shield_pair_id not in ACCEPTANCE_PAIR_IDS
    ):
        raise ValueError("Acceptance transport-seed identity is invalid.")
    payload = (
        f"{_ACCEPTANCE_TRANSPORT_SEED_DOMAIN}|{scene_seed}|"
        f"{scenario_id}|{shield_pair_id}"
    ).encode("utf-8")
    digest_integer = int.from_bytes(
        hashlib.sha256(payload).digest()[:8],
        byteorder="big",
        signed=False,
    )
    return digest_integer % ACCEPTANCE_NATIVE_TRANSPORT_SEED_MAX + 1


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of one existing file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=4)
def acceptance_implementation_bundle_sha256(
    repository_root: str | Path,
) -> str:
    """Hash only the production physics and inference algorithm contract.

    The approval gate intentionally excludes CLI wiring, scenario-profile names,
    documentation, tests, output paths, and the acceptance orchestrator itself.
    Those changes cannot alter a housing-to-spectrum prediction and therefore
    must not force another multi-hour physical validation.  Material model,
    detector, transport, likelihood-semantics, or canonical-array changes remain
    bound through the authenticated model and physics contracts below.
    """
    root = Path(repository_root).resolve()
    if not (root / "src" / "spectrum" / "transport_spectral.py").is_file():
        raise FileNotFoundError(
            f"Runtime repository does not contain the production model: {root}."
        )
    operator = canonical_detector_green_operator()
    reference_model = GeometryConditionedSpectralModel.physics_only_native(
        ACCEPTANCE_ISOTOPES,
        dead_time_tau_s=0.0,
        background_rate_cps=0.0,
        detector_green_operator=operator,
    )
    return canonical_json_sha256(
        {
            "contract": "production_full_spectrum_algorithm",
            "model_schema_version": FULL_SPECTRUM_MODEL_SCHEMA_VERSION,
            "reference_model_contract_sha256": (reference_model.contract_hash_sha256),
            "detector_green_operator_contract_sha256": (operator.contract_hash_sha256),
            "detector_green_operator_binary_sha256": operator.binary_sha256,
            "shield_pose_contract_sha256": SHIELD_POSE_CONTRACT_SHA256,
            "obstacle_material_contract_sha256": (OBSTACLE_MATERIAL_CONTRACT_SHA256),
            "transport_physics_table_contract_sha256": (
                TRANSPORT_PHYSICS_TABLE_CONTRACT_SHA256
            ),
            "geant4_physics_contract_sha256": GEANT4_PHYSICS_CONTRACT_SHA256,
            "dry_air_total_attenuation_contract_sha256": (
                NIST_XCOM_DRY_AIR_TOTAL_CONTRACT_SHA256
            ),
            "surface_emission_policy_sha256": surface_emission_policy_sha256(),
            "transport_feature_order": list(TRANSPORT_FEATURE_ORDER),
        }
    )


def _is_sha256(value: object) -> bool:
    """Return whether a value is one lowercase SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, *, field_name: str) -> str:
    """Return one strict SHA-256 digest or fail closed."""
    if not _is_sha256(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
    return str(value)


def _strict_finite_number(value: object, *, field_name: str) -> float:
    """Return one finite JSON number without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a JSON number.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite.")
    return parsed


def _atomic_write_immutable_json(path: Path, payload: object) -> Path:
    """Write one immutable canonical JSON artifact or verify an exact resume."""
    destination = Path(path)
    encoded = canonical_json_bytes(payload)
    if destination.exists():
        if destination.read_bytes() != encoded:
            raise RuntimeError(
                f"Refusing to overwrite incompatible frozen artifact: {destination}."
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, destination)
    return destination


def _load_json_mapping(path: str | Path) -> Mapping[str, object]:
    """Load one canonical JSON object or fail closed."""
    source = Path(path)
    raw = source.read_bytes()
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid JSON artifact: {source}.") from exc
    if not isinstance(payload, Mapping):
        raise TypeError(f"JSON artifact must contain an object: {source}.")
    try:
        canonical = canonical_json_bytes(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid JSON artifact: {source}.") from exc
    if raw != canonical:
        raise ValueError(f"JSON artifact is not immutable canonical JSON: {source}.")
    return payload


def _expected_split(scene_seed: int) -> str:
    """Return the immutable split assigned to one designated seed."""
    if isinstance(scene_seed, bool) or not isinstance(scene_seed, int):
        raise TypeError("Acceptance scene seed must be a JSON integer.")
    if scene_seed in DESIGNATED_VALIDATION_SCENE_SEEDS:
        return "validation"
    raise ValueError(f"Scene seed {scene_seed} is not designated.")


def line_identity_contract_sha256(
    model: GeometryConditionedSpectralModel,
) -> str:
    """Return the digest of the exact global line identity order."""
    return canonical_json_sha256([dict(row) for row in model.line_identity])


@lru_cache(maxsize=1)
def canonical_detector_green_operator() -> DetectorGreenOperator:
    """Load the one canonical operator authorized for formal acceptance."""
    operator = DetectorGreenOperator.from_artifact(
        DEFAULT_DETECTOR_GREEN_OPERATOR_MANIFEST
    )
    operator.require_runtime_ready()
    return operator


def build_acceptance_run_contract(
    *,
    runtime_config_sha256: str,
    native_executable_sha256: str,
    native_execution_environment_sha256: str,
    implementation_bundle_sha256: str,
) -> dict[str, object]:
    """Return the immutable pre-acquisition acceptance run contract."""
    return {
        "schema_version": ACCEPTANCE_RUN_CONTRACT_SCHEMA_VERSION,
        "acceptance_contract_sha256": (FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256),
        "experiment_profile_id": FULL_SPECTRUM_ACCEPTANCE_EXPERIMENT_ID,
        "runtime_config_sha256": _require_sha256(
            runtime_config_sha256,
            field_name="runtime_config_sha256",
        ),
        "native_executable_sha256": _require_sha256(
            native_executable_sha256,
            field_name="native_executable_sha256",
        ),
        "native_execution_environment_sha256": _require_sha256(
            native_execution_environment_sha256,
            field_name="native_execution_environment_sha256",
        ),
        "implementation_bundle_sha256": _require_sha256(
            implementation_bundle_sha256,
            field_name="implementation_bundle_sha256",
        ),
        "detector_green_operator_contract_sha256": (
            canonical_detector_green_operator().contract_hash_sha256
        ),
        "detector_green_operator_binary_sha256": (
            canonical_detector_green_operator().binary_sha256
        ),
        "validation_scene_seeds": list(DESIGNATED_VALIDATION_SCENE_SEEDS),
        "scenario_ids": list(VALIDATION_SCENARIO_IDS),
        "scenario_source_spec": {
            scenario: [
                {
                    "isotope": isotope,
                    "intensity_cps_1m": float(intensity),
                }
                for isotope, intensity in ACCEPTANCE_SCENARIO_SOURCE_SPEC[scenario]
            ]
            for scenario in VALIDATION_SCENARIO_IDS
        },
        "shield_pair_ids": list(ACCEPTANCE_PAIR_IDS),
        "dwell_time_s": STANDARD_ACQUISITION_LIVE_TIME_S,
        "environment": {
            "room_size_xyz_m": list(ACCEPTANCE_ROOM_SIZE_XYZ),
            "detector_pose_xyz_m": list(ACCEPTANCE_DETECTOR_POSE_XYZ),
            "target_blocked_fraction": (ACCEPTANCE_OBSTACLE_BLOCKED_FRACTION),
            "passage_width_m": ACCEPTANCE_PASSAGE_WIDTH_M,
            "surface_chart_max_edge_m": (ACCEPTANCE_SURFACE_CHART_MAX_EDGE_M),
            "obstacle_material": STANDARD_OBSTACLE_MATERIAL,
            "room_boundary_thickness_m": (STANDARD_ROOM_BOUNDARY_THICKNESS_M),
            "absorber_transport_mode": "absorber",
            "absorber_transport_group": ROOM_ABSORBER_TRANSPORT_GROUP,
            "absorber_transport_contract_sha256": (
                ObstacleGrid(
                    origin=(0.0, 0.0),
                    cell_size=1.0,
                    grid_shape=(1, 1),
                    blocked_cells=(),
                    absorber_transport_group=(ROOM_ABSORBER_TRANSPORT_GROUP),
                    absorber_transport_boxes_m=tuple(
                        component.box_m
                        for component in room_boundary_transport_components(
                            ACCEPTANCE_ROOM_SIZE_XYZ,
                            thickness_m=(STANDARD_ROOM_BOUNDARY_THICKNESS_M),
                            material=STANDARD_OBSTACLE_MATERIAL,
                        )
                    ),
                ).absorber_transport_contract_sha256
            ),
        },
        "geometry_compute": {
            "use_gpu": ACCEPTANCE_GEOMETRY_USE_GPU,
            "device": ACCEPTANCE_GEOMETRY_DEVICE,
            "dtype": ACCEPTANCE_GEOMETRY_DTYPE,
        },
        "native_fidelity": dict(NATIVE_ACCEPTANCE_FIDELITY),
        "dry_air_total_attenuation_contract": {
            "id": NIST_XCOM_DRY_AIR_TOTAL_CONTRACT_ID,
            "sha256": NIST_XCOM_DRY_AIR_TOTAL_CONTRACT_SHA256,
        },
        "surface_emission_policy_sha256": (surface_emission_policy_sha256()),
        "surface_emission_epsilon_m": SURFACE_EMISSION_EPSILON_M,
        "surface_boundary_probe_dwell_time_s": (SURFACE_BOUNDARY_PROBE_DWELL_TIME_S),
        "selection_policy": (
            "physics_only_candidate_frozen_before_all_validation_acquisition"
        ),
        "model_construction": "physics_only_no_scene_fit_or_calibration",
        "transport_seed_domain": _ACCEPTANCE_TRANSPORT_SEED_DOMAIN,
    }


def load_acceptance_run_contract(
    path: str | Path,
) -> tuple[dict[str, object], str]:
    """Load one exact acceptance run contract and return its file digest."""
    source = Path(path).resolve()
    payload = _load_json_mapping(source)
    required_hashes: dict[str, str] = {}
    for field_name in (
        "runtime_config_sha256",
        "native_executable_sha256",
        "native_execution_environment_sha256",
        "implementation_bundle_sha256",
    ):
        required_hashes[field_name] = _require_sha256(
            payload.get(field_name),
            field_name=field_name,
        )
    expected = build_acceptance_run_contract(
        runtime_config_sha256=required_hashes["runtime_config_sha256"],
        native_executable_sha256=required_hashes["native_executable_sha256"],
        native_execution_environment_sha256=required_hashes[
            "native_execution_environment_sha256"
        ],
        implementation_bundle_sha256=required_hashes["implementation_bundle_sha256"],
    )
    if dict(payload) != expected:
        raise ValueError(
            "Acceptance run contract does not match the current exact schema."
        )
    return expected, file_sha256(source)


def validate_surface_boundary_gate(payload: object) -> dict[str, object]:
    """Validate native signed-epsilon positive and negative probe evidence."""
    if not isinstance(payload, Mapping) or set(payload) != _BOUNDARY_GATE_KEYS:
        raise ValueError("surface_boundary_gate has an incompatible schema.")
    evidence = payload["evidence_sha256_by_variant"]
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != SURFACE_BOUNDARY_GATE_SCHEMA_VERSION
        or payload["surface_emission_policy_sha256"] != surface_emission_policy_sha256()
        or _strict_finite_number(
            payload["surface_emission_epsilon_m"],
            field_name="surface_boundary_gate.surface_emission_epsilon_m",
        )
        != SURFACE_EMISSION_EPSILON_M
        or _strict_finite_number(
            payload["probe_dwell_time_s"],
            field_name="surface_boundary_gate.probe_dwell_time_s",
        )
        != SURFACE_BOUNDARY_PROBE_DWELL_TIME_S
        or tuple(payload["native_position_variants"]) != _NATIVE_POSITION_VARIANTS
        or not isinstance(evidence, Mapping)
        or set(evidence) != set(_NATIVE_POSITION_VARIANTS)
        or any(not _is_sha256(evidence[key]) for key in evidence)
        or len(set(evidence.values())) != len(_NATIVE_POSITION_VARIANTS)
        or payload["exact_anchor_vs_air_gate_passed"] is not True
        or payload["solid_minus_air_gate_passed"] is not True
        or payload["passed"] is not True
    ):
        raise ValueError("Native signed-epsilon surface-boundary gate failed.")
    return json.loads(json.dumps(dict(payload), allow_nan=False))


def validate_native_fidelity(payload: object) -> dict[str, object]:
    """Require the exact unit-weight external-Geant4 postconditions."""
    if not isinstance(payload, Mapping):
        raise TypeError("native_fidelity must be an object.")
    if set(payload) != set(NATIVE_ACCEPTANCE_FIDELITY):
        raise ValueError("native_fidelity has an incompatible exact schema.")
    for key, expected in NATIVE_ACCEPTANCE_FIDELITY.items():
        actual = payload[key]
        if isinstance(expected, float):
            parsed = _strict_finite_number(
                actual,
                field_name=f"native_fidelity[{key!r}]",
            )
            if not np.isclose(parsed, expected, rtol=0.0, atol=1.0e-15):
                raise ValueError(f"Native fidelity mismatch for {key}: {parsed!r}.")
        elif type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"Native fidelity mismatch for {key}: {actual!r}.")
    return dict(payload)


def validate_native_process_diagnostics(
    payload: object,
    *,
    expect_transport_processes: bool,
) -> dict[str, object]:
    """Require exact nonnegative native gamma-process counter provenance."""
    if not isinstance(expect_transport_processes, bool):
        raise TypeError("expect_transport_processes must be boolean.")
    if not isinstance(payload, Mapping) or set(payload) != {
        "process_count_compton",
        "process_count_rayleigh",
        "process_count_photoelectric",
        "transport_process_counts",
        "detector_response_coincidence_semantics",
        "detector_response_incident_entry_count",
        "detector_response_registered_entry_count",
        "detector_response_coincidence_pulse_count",
        "detector_response_multi_entry_pulse_count",
        "pre_dead_time_total_pulse_count",
        "post_dead_time_total_pulse_count",
        "dead_time_observed_scale",
    }:
        raise ValueError("native_process_diagnostics has an incompatible exact schema.")
    parsed: dict[str, object] = {}
    for key in (
        "process_count_compton",
        "process_count_rayleigh",
        "process_count_photoelectric",
    ):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"native_process_diagnostics[{key!r}] is invalid.")
        parsed[key] = value
    counters = payload["transport_process_counts"]
    if not isinstance(counters, Mapping) or any(
        not isinstance(name, str)
        or not name
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for name, count in counters.items()
    ):
        raise ValueError("transport_process_counts must be a nonnegative map.")
    parsed["transport_process_counts"] = {
        name: counters[name] for name in sorted(counters)
    }
    if expect_transport_processes:
        if not counters or sum(counters.values()) <= 0:
            raise ValueError(
                "Source-bearing pair must contain positive transport-process counts."
            )
    elif counters:
        raise ValueError(
            "Background-only pair must contain no transport-process counts."
        )
    for field_name, process_name in (
        ("process_count_compton", "compt"),
        ("process_count_rayleigh", "Rayl"),
        ("process_count_photoelectric", "phot"),
    ):
        if parsed[field_name] != counters.get(process_name, 0):
            raise ValueError(f"{field_name} disagrees with transport_process_counts.")
    if payload["detector_response_coincidence_semantics"] != (
        DETECTOR_GREEN_COINCIDENCE_SEMANTICS
    ):
        raise ValueError(
            "native_process_diagnostics has incompatible coincidence semantics."
        )
    parsed["detector_response_coincidence_semantics"] = (
        DETECTOR_GREEN_COINCIDENCE_SEMANTICS
    )
    response_counter_names = (
        "detector_response_incident_entry_count",
        "detector_response_registered_entry_count",
        "detector_response_coincidence_pulse_count",
        "detector_response_multi_entry_pulse_count",
    )
    for field_name in response_counter_names:
        value = payload[field_name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"native_process_diagnostics[{field_name!r}] is invalid.")
        parsed[field_name] = value
    incident_count = int(parsed["detector_response_incident_entry_count"])
    registered_count = int(parsed["detector_response_registered_entry_count"])
    pulse_count = int(parsed["detector_response_coincidence_pulse_count"])
    multi_entry_count = int(parsed["detector_response_multi_entry_pulse_count"])
    if (
        incident_count < registered_count
        or registered_count < pulse_count
        or multi_entry_count > pulse_count
        or registered_count - pulse_count < multi_entry_count
        or (
            not expect_transport_processes
            and any(int(parsed[name]) != 0 for name in response_counter_names)
        )
    ):
        raise ValueError(
            "native_process_diagnostics has inconsistent coincidence counters."
        )
    for field_name in (
        "pre_dead_time_total_pulse_count",
        "post_dead_time_total_pulse_count",
    ):
        value = payload[field_name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"native_process_diagnostics[{field_name!r}] is invalid.")
        parsed[field_name] = value
    scale = _strict_finite_number(
        payload["dead_time_observed_scale"],
        field_name="native_process_diagnostics['dead_time_observed_scale']",
    )
    pre_dead_time_count = int(parsed["pre_dead_time_total_pulse_count"])
    post_dead_time_count = int(parsed["post_dead_time_total_pulse_count"])
    expected_scale = (
        float(post_dead_time_count) / float(pre_dead_time_count)
        if pre_dead_time_count > 0
        else 1.0
    )
    if (
        post_dead_time_count > pre_dead_time_count
        or scale <= 0.0
        or scale > 1.0
        or (pre_dead_time_count > 0 and post_dead_time_count == 0)
        or not np.isclose(scale, expected_scale, rtol=0.0, atol=1.0e-15)
    ):
        raise ValueError("native_process_diagnostics has inconsistent dead-time data.")
    parsed["dead_time_observed_scale"] = scale
    return parsed


def _validate_array_payload(
    value: object,
    *,
    shape: tuple[int, ...],
    field_name: str,
    nonnegative: bool = True,
) -> NDArray[np.float64]:
    """Return one exact-shape finite JSON-number array without coercion."""
    try:
        raw = np.asarray(value)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be a rectangular JSON-number array."
        ) from exc
    if raw.dtype == np.bool_ or not np.issubdtype(raw.dtype, np.number):
        raise TypeError(f"{field_name} must contain only JSON numbers.")
    array = np.asarray(raw, dtype=np.float64)
    if (
        array.shape != shape
        or np.any(~np.isfinite(array))
        or (nonnegative and np.any(array < 0.0))
    ):
        raise ValueError(
            f"{field_name} must have finite shape {shape}"
            + (" and be nonnegative." if nonnegative else ".")
        )
    return array


def _validate_vector_payload(
    value: object,
    *,
    shape: tuple[int, ...],
    field_name: str,
) -> NDArray[np.float64]:
    """Return one exact finite JSON-number vector."""
    return _validate_array_payload(
        value,
        shape=shape,
        field_name=field_name,
        nonnegative=False,
    )


@lru_cache(maxsize=1)
def _acceptance_line_identity() -> tuple[Mapping[str, object], ...]:
    """Return immutable native line rows without repeated matrix builds."""
    model = GeometryConditionedSpectralModel.physics_only_native(
        ACCEPTANCE_ISOTOPES,
        dead_time_tau_s=0.0,
        background_rate_cps=0.0,
        detector_green_operator=canonical_detector_green_operator(),
    )
    return model.line_identity


def _acceptance_line_count() -> int:
    """Return the immutable native line count."""
    return len(_acceptance_line_identity())


@dataclass(frozen=True)
class AcceptancePairRecord:
    """Store one fully authenticated native pair observation."""

    path: Path
    file_sha256: str
    scene_seed: int
    split: str
    scenario_id: str
    shield_pair_id: int
    transport_seed: int
    dwell_time_s: float
    scene_hash: str
    source_contract_sha256: str
    source_count: int
    observed_spectrum_counts: NDArray[np.int64]
    unattenuated_vsl: NDArray[np.float64]
    uncollided_vsl: NDArray[np.float64]
    features_vslf: NDArray[np.float64]
    scatter_basis_vslf: NDArray[np.float64]
    perturbed_unattenuated_vsl: NDArray[np.float64]
    perturbed_uncollided_vsl: NDArray[np.float64]
    perturbed_features_vslf: NDArray[np.float64]
    perturbed_scatter_basis_vslf: NDArray[np.float64]
    labels: Mapping[str, object]
    geometry_family: Mapping[str, object]


def load_acceptance_pair(
    path: str | Path,
    *,
    expected_line_identity_sha256: str,
) -> AcceptancePairRecord:
    """Load and strictly validate one resumable native pair artifact."""
    artifact_path = Path(path).resolve()
    raw = artifact_path.read_bytes()
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid pair artifact: {artifact_path}.") from exc
    try:
        canonical = canonical_json_bytes(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid pair artifact: {artifact_path}.") from exc
    if raw != canonical:
        raise ValueError(
            f"Pair artifact is not immutable canonical JSON: {artifact_path}."
        )
    if not isinstance(payload, Mapping):
        raise ValueError("Pair artifact has an incompatible exact schema.")
    if set(payload) != _PAIR_KEYS:
        raise ValueError("Pair artifact has an incompatible exact schema.")
    seed = payload["scene_seed"]
    pair_id = payload["shield_pair_id"]
    transport_seed = payload["transport_seed"]
    scenario = payload["scenario_id"]
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or isinstance(pair_id, bool)
        or not isinstance(pair_id, int)
        or pair_id not in ACCEPTANCE_PAIR_IDS
        or isinstance(transport_seed, bool)
        or not isinstance(transport_seed, int)
        or transport_seed
        != acceptance_transport_seed(
            scene_seed=seed,
            scenario_id=scenario,
            shield_pair_id=pair_id,
        )
    ):
        raise ValueError("Pair seed or shield-pair identity is invalid.")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != ACCEPTANCE_PAIR_SCHEMA_VERSION
        or payload["acceptance_contract_sha256"]
        != FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256
        or type(payload["split"]) is not str
        or payload["split"] != _expected_split(seed)
        or type(scenario) is not str
        or scenario not in VALIDATION_SCENARIO_IDS
        or payload["line_identity_contract_sha256"]
        != _require_sha256(
            expected_line_identity_sha256,
            field_name="expected_line_identity_sha256",
        )
    ):
        raise ValueError("Pair artifact contract identity is invalid.")
    dwell = _strict_finite_number(
        payload["dwell_time_s"],
        field_name="dwell_time_s",
    )
    if dwell != STANDARD_ACQUISITION_LIVE_TIME_S:
        raise ValueError("Acceptance dwell time differs from its fixed contract.")
    scene_hash = _require_sha256(
        payload["scene_hash"],
        field_name="scene_hash",
    )
    source_hash = _require_sha256(
        payload["surface_source_contract_sha256"],
        field_name="surface_source_contract_sha256",
    )
    expected_sources = ACCEPTANCE_SCENARIO_SOURCE_SPEC[str(scenario)]
    validate_surface_boundary_gate(payload["surface_boundary_gate"])
    validate_native_fidelity(payload["native_fidelity"])
    native_diagnostics = validate_native_process_diagnostics(
        payload["native_process_diagnostics"],
        expect_transport_processes=bool(expected_sources),
    )
    validate_geometry_family_descriptor(
        payload["geometry_family"],
        require_in_domain=True,
    )
    operator = canonical_detector_green_operator()
    expected_physics_contracts = {
        "detector_green_operator_contract_sha256": (operator.contract_hash_sha256),
        "detector_green_operator_binary_sha256": operator.binary_sha256,
        "shield_pose_contract_sha256": SHIELD_POSE_CONTRACT_SHA256,
        "obstacle_material_contract_sha256": (OBSTACLE_MATERIAL_CONTRACT_SHA256),
        "transport_physics_table_contract_sha256": (
            TRANSPORT_PHYSICS_TABLE_CONTRACT_SHA256
        ),
    }
    for field_name, expected_value in expected_physics_contracts.items():
        if payload[field_name] != expected_value:
            raise ValueError(f"Pair physics contract {field_name!r} is incompatible.")
    detector = _validate_vector_payload(
        payload["detector_pose_xyz"],
        shape=(3,),
        field_name="detector_pose_xyz",
    )
    sources = payload["sources"]
    if (
        detector.shape != (3,)
        or np.any(~np.isfinite(detector))
        or not isinstance(sources, Sequence)
        or isinstance(sources, (str, bytes))
        or len(sources) != len(expected_sources)
    ):
        raise ValueError("Pair detector or source truth payload is invalid.")
    normalized_sources = canonical_surface_source_runtime_payload(sources)
    if source_hash != surface_source_runtime_contract_sha256(normalized_sources):
        raise ValueError(
            "Pair surface-source hash does not authenticate its source payload."
        )
    for index, ((expected_isotope, expected_intensity), source) in enumerate(
        zip(expected_sources, normalized_sources, strict=True)
    ):
        if not isinstance(source, Mapping):
            raise ValueError(f"Pair source truth contract is invalid at index {index}.")
        chart_id = source.get("surface_chart_id")
        if (
            source.get("isotope") != expected_isotope
            or _strict_finite_number(
                source.get("intensity_cps_1m"),
                field_name=f"sources[{index}].intensity_cps_1m",
            )
            != expected_intensity
            or isinstance(chart_id, bool)
            or not isinstance(chart_id, int)
            or source.get("surface_emission_policy_sha256")
            != surface_emission_policy_sha256()
        ):
            raise ValueError(f"Pair source truth contract is invalid at index {index}.")
        for field_name, vector_shape in (
            ("surface_uv", (2,)),
            ("position", (3,)),
            ("transport_position", (3,)),
            ("surface_normal", (3,)),
        ):
            _validate_vector_payload(
                source.get(field_name),
                shape=vector_shape,
                field_name=f"sources[{index}].{field_name}",
            )
    raw_observed = np.asarray(payload["observed_spectrum_counts"])
    if (
        raw_observed.shape != (NATIVE_GEANT4_BIN_COUNT,)
        or raw_observed.dtype == np.bool_
        or not np.issubdtype(raw_observed.dtype, np.integer)
        or np.any(raw_observed < 0)
    ):
        raise ValueError(
            "Native acceptance spectrum must contain 851 nonnegative integers."
        )
    geometry = payload["geometry"]
    labels = payload["validation_labels"]
    if (
        not isinstance(geometry, Mapping)
        or set(geometry) != _GEOMETRY_KEYS
        or not isinstance(labels, Mapping)
        or set(labels) != _LABEL_KEYS
        or labels["label_space"] != ADDITIVE_SCATTER_INCIDENT_LABEL_SEMANTICS
        or labels["target_semantics"] != ADDITIVE_SCATTER_TARGET_SEMANTICS
    ):
        raise ValueError("Pair geometry or validation-label schema is invalid.")
    line_count = _acceptance_line_count()
    source_count = len(sources)
    shape = (1, source_count, line_count)
    feature_shape = shape + (len(TRANSPORT_FEATURE_ORDER),)
    scatter_shape = shape + (len(ADDITIVE_SCATTER_FEATURE_ORDER),)
    if source_count == 0:
        base_fields = (
            "unattenuated_source_line_rate_vsl",
            "uncollided_source_line_rate_vsl",
            "transport_features_vslf",
            "additive_scatter_basis_vslf",
        )
        if any(geometry[field_name] is not None for field_name in base_fields):
            raise ValueError("Background-only geometry fields must be encoded as null.")
        unattenuated = np.empty(shape, dtype=np.float64)
        uncollided = np.empty(shape, dtype=np.float64)
        features = np.empty(feature_shape, dtype=np.float64)
        scatter_basis = np.empty(scatter_shape, dtype=np.float64)
    else:
        unattenuated = _validate_array_payload(
            geometry["unattenuated_source_line_rate_vsl"],
            shape=shape,
            field_name="unattenuated_source_line_rate_vsl",
        )
        uncollided = _validate_array_payload(
            geometry["uncollided_source_line_rate_vsl"],
            shape=shape,
            field_name="uncollided_source_line_rate_vsl",
        )
        if np.any(uncollided > unattenuated * (1.0 + 1.0e-12)):
            raise ValueError("Uncollided line rates exceed unattenuated rates.")
        features = _validate_array_payload(
            geometry["transport_features_vslf"],
            shape=feature_shape,
            field_name="transport_features_vslf",
        )
        scatter_basis = _validate_array_payload(
            geometry["additive_scatter_basis_vslf"],
            shape=scatter_shape,
            field_name="additive_scatter_basis_vslf",
        )
    perturbation_expected = str(scenario) == "continuous_surface_perturbation_ranking"
    if perturbation_expected:
        perturbed_unattenuated = _validate_array_payload(
            geometry["perturbed_unattenuated_source_line_rate_vsl"],
            shape=shape,
            field_name="perturbed_unattenuated_source_line_rate_vsl",
        )
        perturbed_uncollided = _validate_array_payload(
            geometry["perturbed_uncollided_source_line_rate_vsl"],
            shape=shape,
            field_name="perturbed_uncollided_source_line_rate_vsl",
        )
        perturbed_features = _validate_array_payload(
            geometry["perturbed_transport_features_vslf"],
            shape=feature_shape,
            field_name="perturbed_transport_features_vslf",
        )
        perturbed_scatter = _validate_array_payload(
            geometry["perturbed_additive_scatter_basis_vslf"],
            shape=scatter_shape,
            field_name="perturbed_additive_scatter_basis_vslf",
        )
    else:
        perturbed_fields = (
            "perturbed_unattenuated_source_line_rate_vsl",
            "perturbed_uncollided_source_line_rate_vsl",
            "perturbed_transport_features_vslf",
            "perturbed_additive_scatter_basis_vslf",
        )
        if any(geometry[field_name] is not None for field_name in perturbed_fields):
            raise ValueError(
                "Non-perturbation scenarios must encode perturbed geometry "
                "fields as null."
            )
        perturbed_unattenuated = np.empty((0, 0, 0), dtype=np.float64)
        perturbed_uncollided = np.empty((0, 0, 0), dtype=np.float64)
        perturbed_features = np.empty((0, 0, 0, 0), dtype=np.float64)
        perturbed_scatter = np.empty((0, 0, 0, 0), dtype=np.float64)
    entry_totals = labels["entry_class_totals_by_source_line"]
    entry_hashes = labels["entry_spectrum_sha256_by_source_line_class"]
    expected_line_tokens = {
        native_source_line_token(
            source_index=source_index,
            isotope=str(source["isotope"]),
            energy_keV=float(line["energy_keV"]),
        )
        for source_index, source in enumerate(normalized_sources)
        for line in _acceptance_line_identity()
        if line["isotope"] == source["isotope"]
    }
    background_total = labels["background_entry_total"]
    if (
        not isinstance(entry_totals, Mapping)
        or not isinstance(entry_hashes, Mapping)
        or set(entry_totals) != expected_line_tokens
        or set(entry_hashes) != expected_line_tokens
        or any(
            not isinstance(row, Mapping)
            or set(row)
            != {
                "uncollided_primary",
                "interacted_primary",
                "secondary",
            }
            for row in entry_totals.values()
        )
        or any(
            not isinstance(row, Mapping)
            or set(row)
            != {
                "uncollided_primary",
                "interacted_primary",
                "secondary",
            }
            or any(not _is_sha256(value) for value in row.values())
            for row in entry_hashes.values()
        )
        or isinstance(background_total, bool)
        or not isinstance(background_total, int)
        or background_total < 0
        or not _is_sha256(labels["background_entry_spectrum_sha256"])
    ):
        raise ValueError("Validation entry-class label payload is invalid.")
    for token, row in entry_totals.items():
        for entry_class, value in row.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    "Entry-class totals must be exact nonnegative unit-weight "
                    f"integer counts: {token!r}/{entry_class!r}."
                )
    if int(native_diagnostics["pre_dead_time_total_pulse_count"]) != int(
        native_diagnostics["detector_response_coincidence_pulse_count"]
    ) + int(background_total) or int(
        native_diagnostics["post_dead_time_total_pulse_count"]
    ) != int(np.sum(raw_observed, dtype=np.int64)):
        raise ValueError(
            "Native dead-time pulse counts disagree with response, background, "
            "or observed-spectrum totals."
        )
    return AcceptancePairRecord(
        path=artifact_path,
        file_sha256=hashlib.sha256(raw).hexdigest(),
        scene_seed=int(seed),
        split=str(payload["split"]),
        scenario_id=str(scenario),
        shield_pair_id=int(pair_id),
        transport_seed=int(transport_seed),
        dwell_time_s=dwell,
        scene_hash=scene_hash,
        source_contract_sha256=source_hash,
        source_count=source_count,
        observed_spectrum_counts=raw_observed.astype(np.int64),
        unattenuated_vsl=unattenuated,
        uncollided_vsl=uncollided,
        features_vslf=features,
        scatter_basis_vslf=scatter_basis,
        perturbed_unattenuated_vsl=perturbed_unattenuated,
        perturbed_uncollided_vsl=perturbed_uncollided,
        perturbed_features_vslf=perturbed_features,
        perturbed_scatter_basis_vslf=perturbed_scatter,
        labels=json.loads(json.dumps(dict(labels), allow_nan=False)),
        geometry_family=json.loads(
            json.dumps(
                dict(payload["geometry_family"]),
                allow_nan=False,
            )
        ),
    )


class AcceptanceScenarioSession(Protocol):
    """Protocol for one cached native scene evaluated across all shield pairs."""

    def acquire_pair(self, shield_pair_id: int) -> Mapping[str, object]:
        """Return one complete pair payload satisfying :func:`load_acceptance_pair`."""


class AcceptanceTransportBackend(Protocol):
    """Protocol selected by the resumable orchestration layer."""

    backend_id: str

    def open_scenario(
        self,
        *,
        scene_seed: int,
        split: str,
        scenario_id: str,
        line_identity_sha256: str,
    ) -> AbstractContextManager[AcceptanceScenarioSession]:
        """Open one native scene session with Geant4 MT inside each request."""


@dataclass(frozen=True)
class AcceptanceRunLayout:
    """Resolve deterministic paths for every resumable phase artifact."""

    root: Path

    @property
    def run_contract_path(self) -> Path:
        """Return the immutable run-contract path."""
        return self.root / "run_contract.json"

    @property
    def candidate_model_path(self) -> Path:
        """Return the physics-only model frozen before validation."""
        return self.root / "candidate_model.json"

    @property
    def validation_manifest_path(self) -> Path:
        """Return the independent validation manifest path."""
        return self.root / "independent_validation.json"

    @property
    def production_model_path(self) -> Path:
        """Return the independently approved production-model path."""
        return self.root / "production_model.json"

    def pair_path(
        self,
        *,
        split: str,
        scene_seed: int,
        scenario_id: str,
        shield_pair_id: int,
    ) -> Path:
        """Return one deterministic pair checkpoint path."""
        return (
            self.root
            / split
            / f"scene_{int(scene_seed)}"
            / scenario_id
            / f"pair_{int(shield_pair_id):02d}.json"
        )

    def scene_corpus_path(self, *, split: str, scene_seed: int) -> Path:
        """Return one complete scene-corpus manifest path."""
        return self.root / split / f"scene_{int(scene_seed)}" / "corpus_manifest.json"

    def scene_acceptance_path(self, *, scene_seed: int) -> Path:
        """Return one evaluated scene-acceptance artifact path."""
        return self.root / "scene_acceptance" / f"scene_{scene_seed}.json"


def _validate_pair_identity(
    record: AcceptancePairRecord,
    *,
    scene_seed: int,
    split: str,
    scenario_id: str,
    shield_pair_id: int,
) -> None:
    """Require one checkpoint to belong to its exact scheduled slot."""
    if (
        record.scene_seed != int(scene_seed)
        or record.split != split
        or record.scenario_id != scenario_id
        or record.shield_pair_id != int(shield_pair_id)
    ):
        raise ValueError("Resumed pair artifact belongs to another schedule slot.")


def acquire_scene_corpus(
    *,
    layout: AcceptanceRunLayout,
    backend: AcceptanceTransportBackend,
    scene_seed: int,
    split: str,
    line_identity_sha256: str,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Acquire or authenticate all five scenario-by-all-64 native pairs."""
    if not isinstance(backend.backend_id, str) or not backend.backend_id:
        raise TypeError("Acceptance backend_id must be a nonempty string.")
    expected_split = _expected_split(scene_seed)
    if split != expected_split:
        raise ValueError("Acquisition split disagrees with designated seed.")
    pair_hashes: dict[str, dict[str, str]] = {}
    scene_hashes: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    gate_hashes: set[str] = set()
    for scenario in VALIDATION_SCENARIO_IDS:
        missing = [
            pair_id
            for pair_id in ACCEPTANCE_PAIR_IDS
            if not layout.pair_path(
                split=split,
                scene_seed=scene_seed,
                scenario_id=scenario,
                shield_pair_id=pair_id,
            ).exists()
        ]
        session_context: AbstractContextManager[AcceptanceScenarioSession] | None = None
        if missing:
            session_context = backend.open_scenario(
                scene_seed=scene_seed,
                split=split,
                scenario_id=scenario,
                line_identity_sha256=line_identity_sha256,
            )
        if session_context is None:
            records = []
            for pair_id in ACCEPTANCE_PAIR_IDS:
                pair_path = layout.pair_path(
                    split=split,
                    scene_seed=scene_seed,
                    scenario_id=scenario,
                    shield_pair_id=pair_id,
                )
                record = load_acceptance_pair(
                    pair_path,
                    expected_line_identity_sha256=line_identity_sha256,
                )
                _validate_pair_identity(
                    record,
                    scene_seed=scene_seed,
                    split=split,
                    scenario_id=scenario,
                    shield_pair_id=pair_id,
                )
                records.append(record)
        else:
            records = []
            with session_context as session:
                for pair_id in ACCEPTANCE_PAIR_IDS:
                    pair_path = layout.pair_path(
                        split=split,
                        scene_seed=scene_seed,
                        scenario_id=scenario,
                        shield_pair_id=pair_id,
                    )
                    if not pair_path.exists():
                        payload = session.acquire_pair(pair_id)
                        _atomic_write_immutable_json(pair_path, payload)
                        if progress is not None:
                            progress(
                                f"{split} scene={scene_seed} "
                                f"scenario={scenario} pair={pair_id}/63"
                            )
                    record = load_acceptance_pair(
                        pair_path,
                        expected_line_identity_sha256=(line_identity_sha256),
                    )
                    _validate_pair_identity(
                        record,
                        scene_seed=scene_seed,
                        split=split,
                        scenario_id=scenario,
                        shield_pair_id=pair_id,
                    )
                    records.append(record)
        if len(records) != 64:
            raise RuntimeError("Scenario corpus did not produce exactly 64 pairs.")
        scenario_scene_hashes = {record.scene_hash for record in records}
        scenario_source_hashes = {record.source_contract_sha256 for record in records}
        if len(scenario_scene_hashes) != 1 or len(scenario_source_hashes) != 1:
            raise ValueError(
                "One scenario changed scene or truth-source identity across pairs."
            )
        pair_hashes[scenario] = {
            str(record.shield_pair_id): record.file_sha256 for record in records
        }
        scene_hashes[scenario] = next(iter(scenario_scene_hashes))
        source_hashes[scenario] = next(iter(scenario_source_hashes))
        for record in records:
            raw_payload = _load_json_mapping(record.path)
            gate_hashes.add(canonical_json_sha256(raw_payload["surface_boundary_gate"]))
    if len(gate_hashes) != 1:
        raise ValueError(
            "One scene seed must use one authenticated signed-epsilon gate."
        )
    manifest = {
        "schema_version": ACCEPTANCE_SCENE_CORPUS_SCHEMA_VERSION,
        "acceptance_contract_sha256": (FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256),
        "backend_id": backend.backend_id,
        "scene_seed": int(scene_seed),
        "split": split,
        "scenario_ids": list(VALIDATION_SCENARIO_IDS),
        "shield_pair_ids": list(ACCEPTANCE_PAIR_IDS),
        "line_identity_contract_sha256": line_identity_sha256,
        "scene_hash_by_scenario": scene_hashes,
        "surface_source_contract_sha256_by_scenario": source_hashes,
        "surface_boundary_gate_sha256": next(iter(gate_hashes)),
        "pair_artifact_sha256_by_scenario": pair_hashes,
        "pair_count": len(VALIDATION_SCENARIO_IDS) * 64,
        "native_fidelity_postconditions_complete": True,
        "complete": True,
    }
    return _atomic_write_immutable_json(
        layout.scene_corpus_path(split=split, scene_seed=scene_seed),
        manifest,
    )


def validate_scene_corpus(
    path: str | Path,
    *,
    layout: AcceptanceRunLayout,
    expected_line_identity_sha256: str,
) -> tuple[AcceptancePairRecord, ...]:
    """Load one complete scene only after rehashing every pair checkpoint."""
    manifest_path = Path(path).resolve()
    payload = _load_json_mapping(manifest_path)
    expected_keys = {
        "schema_version",
        "acceptance_contract_sha256",
        "backend_id",
        "scene_seed",
        "split",
        "scenario_ids",
        "shield_pair_ids",
        "line_identity_contract_sha256",
        "scene_hash_by_scenario",
        "surface_source_contract_sha256_by_scenario",
        "surface_boundary_gate_sha256",
        "pair_artifact_sha256_by_scenario",
        "pair_count",
        "native_fidelity_postconditions_complete",
        "complete",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        raise ValueError("Scene corpus manifest has an incompatible schema.")
    seed = payload["scene_seed"]
    split = payload["split"]
    backend_id = payload["backend_id"]
    pair_count = payload["pair_count"]
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not isinstance(split, str)
        or split != _expected_split(seed)
        or not isinstance(backend_id, str)
        or not backend_id
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != ACCEPTANCE_SCENE_CORPUS_SCHEMA_VERSION
        or payload["acceptance_contract_sha256"]
        != FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256
        or tuple(payload["scenario_ids"]) != VALIDATION_SCENARIO_IDS
        or tuple(payload["shield_pair_ids"]) != ACCEPTANCE_PAIR_IDS
        or payload["line_identity_contract_sha256"] != expected_line_identity_sha256
        or type(pair_count) is not int
        or pair_count != len(VALIDATION_SCENARIO_IDS) * 64
        or payload["native_fidelity_postconditions_complete"] is not True
        or payload["complete"] is not True
    ):
        raise ValueError("Scene corpus manifest is incomplete or incompatible.")
    pair_hashes = payload["pair_artifact_sha256_by_scenario"]
    scene_hashes = payload["scene_hash_by_scenario"]
    source_hashes = payload["surface_source_contract_sha256_by_scenario"]
    if (
        not isinstance(pair_hashes, Mapping)
        or set(pair_hashes) != set(VALIDATION_SCENARIO_IDS)
        or not isinstance(scene_hashes, Mapping)
        or set(scene_hashes) != set(VALIDATION_SCENARIO_IDS)
        or not isinstance(source_hashes, Mapping)
        or set(source_hashes) != set(VALIDATION_SCENARIO_IDS)
        or not _is_sha256(payload["surface_boundary_gate_sha256"])
    ):
        raise ValueError("Scene corpus authentication mappings are invalid.")
    records: list[AcceptancePairRecord] = []
    gate_hashes: set[str] = set()
    for scenario in VALIDATION_SCENARIO_IDS:
        scenario_hashes = pair_hashes[scenario]
        if not isinstance(scenario_hashes, Mapping) or set(scenario_hashes) != {
            str(pair_id) for pair_id in ACCEPTANCE_PAIR_IDS
        }:
            raise ValueError("Scene corpus does not authenticate all 64 pairs.")
        for pair_id in ACCEPTANCE_PAIR_IDS:
            pair_path = layout.pair_path(
                split=str(split),
                scene_seed=int(seed),
                scenario_id=scenario,
                shield_pair_id=pair_id,
            )
            if (
                not pair_path.exists()
                or file_sha256(pair_path) != scenario_hashes[str(pair_id)]
            ):
                raise ValueError(f"Pair checkpoint hash mismatch: {pair_path}.")
            record = load_acceptance_pair(
                pair_path,
                expected_line_identity_sha256=(expected_line_identity_sha256),
            )
            _validate_pair_identity(
                record,
                scene_seed=int(seed),
                split=str(split),
                scenario_id=scenario,
                shield_pair_id=pair_id,
            )
            if (
                record.scene_hash != scene_hashes[scenario]
                or record.source_contract_sha256 != source_hashes[scenario]
            ):
                raise ValueError("Scene corpus pair identity is stale.")
            raw_pair = _load_json_mapping(pair_path)
            gate_hashes.add(canonical_json_sha256(raw_pair["surface_boundary_gate"]))
            records.append(record)
    if gate_hashes != {str(payload["surface_boundary_gate_sha256"])}:
        raise ValueError("Scene corpus signed-epsilon gate hashes disagree.")
    return tuple(records)


def freeze_candidate_model(
    *,
    layout: AcceptanceRunLayout,
    model: GeometryConditionedSpectralModel,
) -> Path:
    """Freeze the exact pre-validation model and require it to be unapproved."""
    if model.production_ready or model.validation_manifest is not None:
        raise ValueError(
            "Candidate model must be frozen before validation acquisition."
        )
    payload = dict(model.manifest_payload())
    if (
        payload["contract_hash_sha256"] != model.contract_hash_sha256
        or payload["production_ready"] is not False
        or payload["validation"] is not None
    ):
        raise RuntimeError("Candidate manifest violates pre-validation isolation.")
    return _atomic_write_immutable_json(layout.candidate_model_path, payload)


def load_frozen_candidate_model(
    layout: AcceptanceRunLayout,
) -> GeometryConditionedSpectralModel:
    """Reconstruct the strict physics-only pre-validation candidate."""
    payload = _load_json_mapping(layout.candidate_model_path)
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != FULL_SPECTRUM_MODEL_SCHEMA_VERSION
        or payload.get("model") != "geometry_conditioned_full_spectrum"
        or payload.get("production_ready") is not False
        or payload.get("validation") is not None
        or not _is_sha256(payload.get("contract_hash_sha256"))
    ):
        raise ValueError("Frozen candidate model is incompatible.")
    model = GeometryConditionedSpectralModel.from_manifest_payload(
        payload,
        detector_green_operator=canonical_detector_green_operator(),
    )
    if not model.runtime_ready or model.production_ready:
        raise ValueError("Frozen candidate is not a runtime-ready unapproved model.")
    return model


def acquire_designated_split(
    *,
    layout: AcceptanceRunLayout,
    backend: AcceptanceTransportBackend,
    seeds: Sequence[int],
    split: str,
    line_identity_sha256: str,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, ...]:
    """Acquire the exact validation split after freezing one candidate."""
    if split != "validation":
        raise ValueError(
            "Production acceptance has no training split; only the frozen "
            "candidate validation split is reachable."
        )
    expected = DESIGNATED_VALIDATION_SCENE_SEEDS
    if (
        not isinstance(seeds, Sequence)
        or isinstance(seeds, (str, bytes))
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or tuple(seeds) != expected
    ):
        raise ValueError(f"{split} acquisition must use its exact seed tuple.")
    candidate = load_frozen_candidate_model(layout)
    if candidate.production_ready:
        raise RuntimeError("Validation must evaluate the frozen candidate.")
    paths = []
    for seed in seeds:
        paths.append(
            acquire_scene_corpus(
                layout=layout,
                backend=backend,
                scene_seed=seed,
                split=split,
                line_identity_sha256=line_identity_sha256,
                progress=progress,
            )
        )
    return tuple(paths)


__all__ = [
    "ACCEPTANCE_ISOTOPES",
    "ACCEPTANCE_PAIR_IDS",
    "ACCEPTANCE_SCENARIO_SOURCE_SPEC",
    "AcceptancePairRecord",
    "AcceptanceRunLayout",
    "AcceptanceScenarioSession",
    "AcceptanceTransportBackend",
    "NATIVE_ACCEPTANCE_FIDELITY",
    "acceptance_implementation_bundle_sha256",
    "acceptance_transport_seed",
    "acquire_designated_split",
    "acquire_scene_corpus",
    "build_acceptance_run_contract",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "file_sha256",
    "freeze_candidate_model",
    "line_identity_contract_sha256",
    "load_acceptance_pair",
    "load_acceptance_run_contract",
    "load_frozen_candidate_model",
    "validate_native_fidelity",
    "validate_native_process_diagnostics",
    "validate_scene_corpus",
    "validate_surface_boundary_gate",
]
