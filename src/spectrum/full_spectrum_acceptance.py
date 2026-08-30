"""Fail-closed aggregation for independent all-64 full-spectrum validation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from collections.abc import Mapping, Sequence

from measurement.source_boundary import (
    SURFACE_EMISSION_EPSILON_M,
    surface_emission_policy_sha256,
)
from spectrum.detector_green_operator import DetectorGreenOperator
from spectrum.detector_green_validation import (
    detector_green_validation_manifest_sha256,
    validate_detector_green_validation_manifest,
)
from spectrum.transport_spectral import (
    ACCEPTANCE_METRIC_CONTRACT,
    CANONICAL_DETECTOR_GREEN_OPERATOR_MANIFEST,
    DESIGNATED_VALIDATION_SCENE_SEEDS,
    FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256,
    SURFACE_BOUNDARY_GATE_SCHEMA_VERSION,
    SURFACE_BOUNDARY_PROBE_DWELL_TIME_S,
    VALIDATION_SCENARIO_IDS,
)


ACCEPTANCE_SCENE_ARTIFACT_SCHEMA_VERSION = 5
SURFACE_BOUNDARY_NATIVE_POSITION_VARIANTS = (
    "exact_surface_anchor",
    "air_plus_epsilon",
    "solid_minus_epsilon",
)
_SCENE_ARTIFACT_KEYS = frozenset(
    {
        "schema_version",
        "acceptance_contract_sha256",
        "acceptance_run_contract_sha256",
        "runtime_config_sha256",
        "native_executable_sha256",
        "native_execution_environment_sha256",
        "implementation_bundle_sha256",
        "scene_seed",
        "split",
        "scenario_ids",
        "shield_pair_ids",
        "pair_ids_by_scenario",
        "observation_count_by_scenario",
        "approved_model_contract_sha256",
        "detector_green_operator_contract_sha256",
        "detector_green_operator_binary_sha256",
        "additive_scatter_contract_sha256",
        "surface_emission_policy_sha256",
        "scene_hash_by_scenario",
        "surface_source_contract_sha256_by_scenario",
        "surface_boundary_gate",
        "metrics",
    }
)
_SURFACE_BOUNDARY_GATE_KEYS = frozenset(
    {
        "schema_version",
        "surface_emission_policy_sha256",
        "surface_emission_epsilon_m",
        "probe_dwell_time_s",
        "native_position_variants",
        "exact_anchor_vs_air_gate_passed",
        "solid_minus_air_gate_passed",
        "passed",
    }
)


@dataclass(frozen=True)
class AcceptanceSceneArtifact:
    """Store one authenticated all-pair scene acceptance artifact."""

    path: Path
    file_sha256: str
    scene_seed: int
    split: str
    model_contract_sha256: str
    additive_scatter_contract_sha256: str
    acceptance_run_contract_sha256: str
    runtime_config_sha256: str
    native_executable_sha256: str
    native_execution_environment_sha256: str
    implementation_bundle_sha256: str
    detector_green_operator_contract_sha256: str
    detector_green_operator_binary_sha256: str
    scene_hash_by_scenario: Mapping[str, str]
    surface_source_contract_sha256_by_scenario: Mapping[str, str]
    metrics: Mapping[str, float]


def _is_sha256(value: object) -> bool:
    """Return whether a value is a lowercase hexadecimal SHA-256 digest."""
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


def _canonical_detector_green_operator() -> DetectorGreenOperator:
    """Load the canonical operator bound to production model assets."""
    operator = DetectorGreenOperator.from_artifact(
        CANONICAL_DETECTOR_GREEN_OPERATOR_MANIFEST
    )
    operator.require_runtime_ready()
    return operator


def _expected_split(scene_seed: int) -> str:
    """Return the predeclared split for one designated scene seed."""
    if scene_seed in DESIGNATED_VALIDATION_SCENE_SEEDS:
        return "validation"
    raise ValueError(
        f"Scene seed {scene_seed} is outside the predeclared acceptance split."
    )


def _validate_surface_boundary_gate(payload: object) -> None:
    """Require native exact/air/solid signed-epsilon boundary validation."""
    if not isinstance(payload, Mapping):
        raise TypeError("surface_boundary_gate must be a mapping.")
    if set(payload) != _SURFACE_BOUNDARY_GATE_KEYS:
        raise ValueError("surface_boundary_gate has an incompatible exact schema.")
    epsilon = payload["surface_emission_epsilon_m"]
    probe_dwell = payload["probe_dwell_time_s"]
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError(
            "surface_boundary_gate.surface_emission_epsilon_m must be a JSON number."
        )
    if isinstance(probe_dwell, bool) or not isinstance(probe_dwell, (int, float)):
        raise TypeError(
            "surface_boundary_gate.probe_dwell_time_s must be a JSON number."
        )
    if (
        payload["schema_version"] != SURFACE_BOUNDARY_GATE_SCHEMA_VERSION
        or payload["surface_emission_policy_sha256"] != surface_emission_policy_sha256()
        or float(epsilon) != SURFACE_EMISSION_EPSILON_M
        or float(probe_dwell) != SURFACE_BOUNDARY_PROBE_DWELL_TIME_S
        or tuple(payload["native_position_variants"])
        != SURFACE_BOUNDARY_NATIVE_POSITION_VARIANTS
        or payload["exact_anchor_vs_air_gate_passed"] is not True
        or payload["solid_minus_air_gate_passed"] is not True
        or payload["passed"] is not True
    ):
        raise ValueError(
            "Native signed-epsilon surface-boundary validation did not pass."
        )


def load_acceptance_scene_artifact(
    path: str | Path,
) -> AcceptanceSceneArtifact:
    """Load and strictly validate one Geant4 all-64 scene artifact."""
    operator = _canonical_detector_green_operator()
    artifact_path = Path(path).resolve()
    raw_bytes = artifact_path.read_bytes()
    try:
        payload = json.loads(raw_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"Acceptance artifact is not valid JSON: {artifact_path}."
        ) from exc
    try:
        canonical_bytes = (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Acceptance artifact is not strict JSON: {artifact_path}."
        ) from exc
    if raw_bytes != canonical_bytes:
        raise ValueError(
            f"Acceptance artifact is not immutable canonical JSON: {artifact_path}."
        )
    if not isinstance(payload, Mapping) or set(payload) != _SCENE_ARTIFACT_KEYS:
        raise ValueError(
            f"Acceptance artifact has an incompatible exact schema: {artifact_path}."
        )
    scene_seed = payload["scene_seed"]
    if isinstance(scene_seed, bool) or not isinstance(scene_seed, int):
        raise TypeError("scene_seed must be a JSON integer.")
    split = payload["split"]
    if not isinstance(split, str) or split != _expected_split(scene_seed):
        raise ValueError(
            "Acceptance artifact split disagrees with its predeclared seed."
        )
    if (
        payload["schema_version"] != ACCEPTANCE_SCENE_ARTIFACT_SCHEMA_VERSION
        or payload["acceptance_contract_sha256"]
        != FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256
        or tuple(payload["scenario_ids"]) != VALIDATION_SCENARIO_IDS
        or tuple(payload["shield_pair_ids"]) != tuple(range(64))
        or payload["detector_green_operator_contract_sha256"]
        != operator.contract_hash_sha256
        or payload["detector_green_operator_binary_sha256"] != operator.binary_sha256
        or payload["surface_emission_policy_sha256"] != surface_emission_policy_sha256()
    ):
        raise ValueError(
            "Acceptance artifact does not satisfy the fixed scenarios, "
            "all-64 pairs, or physics contracts."
        )
    pair_ids_by_scenario = payload["pair_ids_by_scenario"]
    observation_count_by_scenario = payload["observation_count_by_scenario"]
    expected_scenarios = set(VALIDATION_SCENARIO_IDS)
    if (
        not isinstance(pair_ids_by_scenario, Mapping)
        or set(pair_ids_by_scenario) != expected_scenarios
        or any(
            tuple(pair_ids_by_scenario[scenario]) != tuple(range(64))
            for scenario in VALIDATION_SCENARIO_IDS
        )
        or not isinstance(observation_count_by_scenario, Mapping)
        or set(observation_count_by_scenario) != expected_scenarios
    ):
        raise ValueError(
            "Every acceptance scenario must independently cover all 64 shield pairs."
        )
    for scenario in VALIDATION_SCENARIO_IDS:
        count = observation_count_by_scenario[scenario]
        if isinstance(count, bool) or not isinstance(count, int) or count < 64:
            raise ValueError(
                "Every acceptance scenario must contain at least one "
                "observation for each shield pair."
            )
    model_hash = _require_sha256(
        payload["approved_model_contract_sha256"],
        field_name="approved_model_contract_sha256",
    )
    additive_hash = _require_sha256(
        payload["additive_scatter_contract_sha256"],
        field_name="additive_scatter_contract_sha256",
    )
    run_contract_hash = _require_sha256(
        payload["acceptance_run_contract_sha256"],
        field_name="acceptance_run_contract_sha256",
    )
    runtime_config_hash = _require_sha256(
        payload["runtime_config_sha256"],
        field_name="runtime_config_sha256",
    )
    native_executable_hash = _require_sha256(
        payload["native_executable_sha256"],
        field_name="native_executable_sha256",
    )
    native_execution_environment_hash = _require_sha256(
        payload["native_execution_environment_sha256"],
        field_name="native_execution_environment_sha256",
    )
    implementation_bundle_hash = _require_sha256(
        payload["implementation_bundle_sha256"],
        field_name="implementation_bundle_sha256",
    )
    operator_contract_hash = _require_sha256(
        payload["detector_green_operator_contract_sha256"],
        field_name="detector_green_operator_contract_sha256",
    )
    operator_binary_hash = _require_sha256(
        payload["detector_green_operator_binary_sha256"],
        field_name="detector_green_operator_binary_sha256",
    )
    scene_hashes = payload["scene_hash_by_scenario"]
    source_hashes = payload["surface_source_contract_sha256_by_scenario"]
    if (
        not isinstance(scene_hashes, Mapping)
        or set(scene_hashes) != expected_scenarios
        or not isinstance(source_hashes, Mapping)
        or set(source_hashes) != expected_scenarios
    ):
        raise ValueError(
            "Acceptance artifacts must authenticate every scenario-specific "
            "native scene and source contract."
        )
    validated_scene_hashes = {
        scenario: _require_sha256(
            scene_hashes[scenario],
            field_name=f"scene_hash_by_scenario[{scenario!r}]",
        )
        for scenario in VALIDATION_SCENARIO_IDS
    }
    validated_source_hashes = {
        scenario: _require_sha256(
            source_hashes[scenario],
            field_name=(f"surface_source_contract_sha256_by_scenario[{scenario!r}]"),
        )
        for scenario in VALIDATION_SCENARIO_IDS
    }
    _validate_surface_boundary_gate(payload["surface_boundary_gate"])
    raw_metrics = payload["metrics"]
    if not isinstance(raw_metrics, Mapping) or set(raw_metrics) != set(
        ACCEPTANCE_METRIC_CONTRACT
    ):
        raise ValueError(
            "Acceptance scene metrics must exactly match the fixed contract."
        )
    metrics: dict[str, float] = {}
    for metric_id, value in raw_metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Acceptance metric {metric_id!r} must be a JSON number.")
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"Acceptance metric {metric_id!r} must be finite.")
        metrics[str(metric_id)] = parsed
    return AcceptanceSceneArtifact(
        path=artifact_path,
        file_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        scene_seed=int(scene_seed),
        split=split,
        model_contract_sha256=model_hash,
        additive_scatter_contract_sha256=additive_hash,
        acceptance_run_contract_sha256=run_contract_hash,
        runtime_config_sha256=runtime_config_hash,
        native_executable_sha256=native_executable_hash,
        native_execution_environment_sha256=(native_execution_environment_hash),
        implementation_bundle_sha256=implementation_bundle_hash,
        detector_green_operator_contract_sha256=operator_contract_hash,
        detector_green_operator_binary_sha256=operator_binary_hash,
        scene_hash_by_scenario=validated_scene_hashes,
        surface_source_contract_sha256_by_scenario=(validated_source_hashes),
        metrics=metrics,
    )


def _load_exact_scene_set(
    paths: Sequence[str | Path],
    *,
    expected_seeds: Sequence[int],
) -> tuple[AcceptanceSceneArtifact, ...]:
    """Load exactly one artifact for every expected scene seed."""
    artifacts = tuple(load_acceptance_scene_artifact(path) for path in paths)
    by_seed = {artifact.scene_seed: artifact for artifact in artifacts}
    if len(by_seed) != len(artifacts):
        raise ValueError("Acceptance artifacts contain a duplicate scene seed.")
    if set(by_seed) != set(expected_seeds):
        raise ValueError(
            "Acceptance artifacts do not contain the exact designated scene set."
        )
    ordered = tuple(by_seed[int(seed)] for seed in expected_seeds)
    model_hashes = {artifact.model_contract_sha256 for artifact in ordered}
    additive_hashes = {
        artifact.additive_scatter_contract_sha256 for artifact in ordered
    }
    provenance_sets = {
        "acceptance_run_contract_sha256": {
            artifact.acceptance_run_contract_sha256 for artifact in ordered
        },
        "runtime_config_sha256": {
            artifact.runtime_config_sha256 for artifact in ordered
        },
        "native_executable_sha256": {
            artifact.native_executable_sha256 for artifact in ordered
        },
        "native_execution_environment_sha256": {
            artifact.native_execution_environment_sha256 for artifact in ordered
        },
        "implementation_bundle_sha256": {
            artifact.implementation_bundle_sha256 for artifact in ordered
        },
        "detector_green_operator_contract_sha256": {
            artifact.detector_green_operator_contract_sha256 for artifact in ordered
        },
        "detector_green_operator_binary_sha256": {
            artifact.detector_green_operator_binary_sha256 for artifact in ordered
        },
    }
    if (
        len(model_hashes) != 1
        or len(additive_hashes) != 1
        or any(len(values) != 1 for values in provenance_sets.values())
    ):
        raise ValueError(
            "Acceptance artifacts disagree on the evaluated model or native "
            "execution contracts."
        )
    return ordered


def build_independent_validation_manifest(
    paths: Sequence[str | Path],
    *,
    detector_green_validation_manifest: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Aggregate validation-only metrics and preserve all-scene provenance."""
    all_seeds = DESIGNATED_VALIDATION_SCENE_SEEDS
    artifacts = _load_exact_scene_set(paths, expected_seeds=all_seeds)
    by_seed = {artifact.scene_seed: artifact for artifact in artifacts}
    validation = tuple(by_seed[seed] for seed in DESIGNATED_VALIDATION_SCENE_SEEDS)
    if any(artifact.split != "validation" for artifact in validation):
        raise ValueError("Validation metrics must originate from validation scenes.")
    model_hash = artifacts[0].model_contract_sha256
    additive_hash = artifacts[0].additive_scatter_contract_sha256
    if detector_green_validation_manifest is None:
        raise RuntimeError(
            "Production approval requires an independent monoenergetic "
            "detector Green validation manifest."
        )
    operator = _canonical_detector_green_operator()
    detector_green_validation = validate_detector_green_validation_manifest(
        detector_green_validation_manifest,
        operator=operator,
    )
    # The monoenergetic detector run and the Cs/Co application run have
    # intentionally different configs. Their shared native executable and
    # execution environment must still match exactly; each config hash remains
    # authenticated inside its own manifest.
    expected_provenance = {
        "native_executable_sha256": artifacts[0].native_executable_sha256,
        "native_execution_environment_sha256": (
            artifacts[0].native_execution_environment_sha256
        ),
    }
    if any(
        detector_green_validation[field_name] != expected_value
        for field_name, expected_value in expected_provenance.items()
    ):
        raise ValueError(
            "Detector Green validation provenance differs from application acceptance."
        )
    metrics: dict[str, dict[str, object]] = {}
    for metric_id, (comparison, threshold) in ACCEPTANCE_METRIC_CONTRACT.items():
        values = [artifact.metrics[metric_id] for artifact in validation]
        value = max(values) if comparison == "le" else min(values)
        passed = (
            value <= float(threshold)
            if comparison == "le"
            else value >= float(threshold)
        )
        metrics[metric_id] = {
            "value": float(value),
            "comparison": comparison,
            "threshold": float(threshold),
            "passed": bool(passed),
        }
    return {
        "schema_version": 6,
        "validation_contract_sha256": (FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256),
        "approved_model_contract_sha256": model_hash,
        "acceptance_run_contract_sha256": (artifacts[0].acceptance_run_contract_sha256),
        "runtime_config_sha256": artifacts[0].runtime_config_sha256,
        "native_executable_sha256": (artifacts[0].native_executable_sha256),
        "native_execution_environment_sha256": (
            artifacts[0].native_execution_environment_sha256
        ),
        "implementation_bundle_sha256": (artifacts[0].implementation_bundle_sha256),
        "detector_green_operator_contract_sha256": (operator.contract_hash_sha256),
        "detector_green_operator_binary_sha256": operator.binary_sha256,
        "detector_green_validation": detector_green_validation,
        "detector_green_validation_manifest_sha256": (
            detector_green_validation_manifest_sha256(
                detector_green_validation,
                operator=operator,
            )
        ),
        "additive_scatter_contract_sha256": additive_hash,
        "surface_emission_policy_sha256": (surface_emission_policy_sha256()),
        "validation_scene_seeds": list(DESIGNATED_VALIDATION_SCENE_SEEDS),
        "candidate_selection": "none_predeclared_physics_only",
        "scene_calibration_count": 0,
        "metric_scene_seeds": list(DESIGNATED_VALIDATION_SCENE_SEEDS),
        "metric_split": "independent_validation_only",
        "metric_aggregation": "validation_scene_conservative_worst_case",
        "scenario_ids": list(VALIDATION_SCENARIO_IDS),
        "pair_ids_by_scene": {str(seed): list(range(64)) for seed in all_seeds},
        "artifact_sha256_by_scene": {
            str(seed): by_seed[seed].file_sha256 for seed in all_seeds
        },
        "scene_hash_by_scene_and_scenario": {
            str(seed): dict(by_seed[seed].scene_hash_by_scenario) for seed in all_seeds
        },
        "surface_source_contract_sha256_by_scene_and_scenario": {
            str(seed): dict(by_seed[seed].surface_source_contract_sha256_by_scenario)
            for seed in all_seeds
        },
        "metrics": metrics,
        "all_passed": all(result["passed"] is True for result in metrics.values()),
    }


def write_independent_validation_manifest(
    paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    detector_green_validation_manifest: Mapping[str, object],
) -> Path:
    """Validate artifacts and write one deterministic validation manifest."""
    manifest = build_independent_validation_manifest(
        paths,
        detector_green_validation_manifest=(detector_green_validation_manifest),
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            manifest,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination
