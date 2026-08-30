"""Tests for fail-closed resumable full-spectrum acceptance artifacts."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from collections.abc import Iterator

import numpy as np
import pytest

import spectrum.full_spectrum_acceptance_runner as acceptance_runner

from measurement.geometry_family import (
    GEOMETRY_FAMILY_APPLICABILITY_SHA256,
    GEOMETRY_FAMILY_ID,
    GEOMETRY_FAMILY_SCHEMA_VERSION,
    GEOMETRY_GENERATOR_ALGORITHM_ID,
)
from measurement.source_boundary import (
    SURFACE_EMISSION_EPSILON_M,
    surface_emission_policy_sha256,
    surface_source_runtime_contract_sha256,
)
from measurement.shielding import SHIELD_POSE_CONTRACT_SHA256
from runtime.experiment_profiles import STANDARD_ACQUISITION_LIVE_TIME_S
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
    SURFACE_BOUNDARY_PROBE_DWELL_TIME_S,
)
from spectrum.full_spectrum_acceptance_runner import (
    ACCEPTANCE_PAIR_SCHEMA_VERSION,
    NATIVE_ACCEPTANCE_FIDELITY,
    acceptance_implementation_bundle_sha256,
    acceptance_transport_seed,
    build_acceptance_run_contract,
    canonical_json_bytes,
    line_identity_contract_sha256,
    load_acceptance_pair,
    load_acceptance_run_contract,
)
from spectrum.detector_green_operator import DETECTOR_GREEN_COINCIDENCE_SEMANTICS
from spectrum.native_metadata import native_source_line_token
from spectrum.physics_contracts import (
    OBSTACLE_MATERIAL_CONTRACT_SHA256,
    TRANSPORT_PHYSICS_TABLE_CONTRACT_SHA256,
)
from spectrum.response_matrix import NATIVE_GEANT4_BIN_COUNT
from spectrum.transport_spectral import (
    DESIGNATED_VALIDATION_SCENE_SEEDS,
    FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256,
    TRANSPORT_FEATURE_ORDER,
    GeometryConditionedSpectralModel,
)
from tests.green_test_support import synthetic_detector_green_operator


_TEST_OPERATOR = synthetic_detector_green_operator()


def test_transport_seed_schedule_uses_exact_signed_64_bit_integers() -> None:
    """Every predeclared native seed must be positive, bounded, and unique."""
    seeds = [
        acceptance_transport_seed(
            scene_seed=scene_seed,
            scenario_id=scenario_id,
            shield_pair_id=pair_id,
        )
        for scene_seed in DESIGNATED_VALIDATION_SCENE_SEEDS
        for scenario_id in acceptance_runner.VALIDATION_SCENARIO_IDS
        for pair_id in acceptance_runner.ACCEPTANCE_PAIR_IDS
    ]
    assert all(
        1 <= seed <= acceptance_runner.ACCEPTANCE_NATIVE_TRANSPORT_SEED_MAX
        for seed in seeds
    )
    assert len(set(seeds)) == len(seeds)


@pytest.fixture(autouse=True)
def _use_explicit_test_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Keep unit artifacts independent of the formal generated asset."""
    monkeypatch.setattr(
        acceptance_runner,
        "canonical_detector_green_operator",
        lambda: _TEST_OPERATOR,
    )
    acceptance_runner._acceptance_line_identity.cache_clear()
    yield
    acceptance_runner._acceptance_line_identity.cache_clear()


def _base_model() -> GeometryConditionedSpectralModel:
    """Return the fixed native line contract used by acceptance artifacts."""
    return GeometryConditionedSpectralModel.physics_only_native(
        ("Co-60", "Cs-137"),
        dead_time_tau_s=0.0,
        background_rate_cps=0.0,
        detector_green_operator=_TEST_OPERATOR,
    )


def test_acceptance_run_contract_authenticates_python_implementation() -> None:
    """Every resumable phase must bind the exact Python implementation."""
    contract = build_acceptance_run_contract(
        runtime_config_sha256="a" * 64,
        native_executable_sha256="b" * 64,
        native_execution_environment_sha256="d" * 64,
        implementation_bundle_sha256="c" * 64,
    )

    assert contract["schema_version"] == 13
    assert contract["surface_boundary_probe_dwell_time_s"] == (
        SURFACE_BOUNDARY_PROBE_DWELL_TIME_S
    )
    assert contract["environment"]["absorber_transport_group"] == "wall"
    assert len(contract["environment"]["absorber_transport_contract_sha256"]) == 64
    assert contract["dwell_time_s"] == STANDARD_ACQUISITION_LIVE_TIME_S
    assert contract["dry_air_total_attenuation_contract"] == {
        "id": NIST_XCOM_DRY_AIR_TOTAL_CONTRACT_ID,
        "sha256": NIST_XCOM_DRY_AIR_TOTAL_CONTRACT_SHA256,
    }
    assert contract["native_execution_environment_sha256"] == "d" * 64
    assert contract["implementation_bundle_sha256"] == "c" * 64


def test_algorithm_approval_ignores_nonphysical_repository_changes(
    tmp_path: Path,
) -> None:
    """CLI, profile, and output-name edits must not expire physical approval."""
    model_path = tmp_path / "src/spectrum/transport_spectral.py"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("# repository identity sentinel\n", encoding="utf-8")
    cli_path = tmp_path / "src/runtime/cli.py"
    cli_path.parent.mkdir(parents=True)
    cli_path.write_text("# before\n", encoding="utf-8")
    acceptance_implementation_bundle_sha256.cache_clear()
    before = acceptance_implementation_bundle_sha256(tmp_path)

    cli_path.write_text("# after\n", encoding="utf-8")
    (tmp_path / "results-v99.txt").write_text("renamed\n", encoding="utf-8")
    acceptance_implementation_bundle_sha256.cache_clear()
    after = acceptance_implementation_bundle_sha256(tmp_path)

    assert after == before


def test_acceptance_run_contract_rejects_invalid_implementation_digest() -> None:
    """A missing implementation digest must not permit mixed phase code."""
    with pytest.raises(ValueError, match="implementation_bundle_sha256"):
        build_acceptance_run_contract(
            runtime_config_sha256="a" * 64,
            native_executable_sha256="b" * 64,
            native_execution_environment_sha256="d" * 64,
            implementation_bundle_sha256="not-a-digest",
        )


def test_acceptance_run_contract_loader_rejects_schema_drift(
    tmp_path: Path,
) -> None:
    """Approval phases must reauthenticate the exact run-contract schema."""
    contract = build_acceptance_run_contract(
        runtime_config_sha256="a" * 64,
        native_executable_sha256="b" * 64,
        native_execution_environment_sha256="d" * 64,
        implementation_bundle_sha256="c" * 64,
    )
    path = tmp_path / "run_contract.json"
    path.write_bytes(canonical_json_bytes(contract))

    loaded, digest = load_acceptance_run_contract(path)

    assert loaded == contract
    assert len(digest) == 64
    contract["legacy_fallback"] = True
    path.write_bytes(canonical_json_bytes(contract))
    with pytest.raises(ValueError, match="current exact schema"):
        load_acceptance_run_contract(path)


def _boundary_gate() -> dict[str, object]:
    """Return deterministic distinct signed-epsilon evidence."""
    return {
        "schema_version": SURFACE_BOUNDARY_GATE_SCHEMA_VERSION,
        "surface_emission_policy_sha256": (surface_emission_policy_sha256()),
        "surface_emission_epsilon_m": SURFACE_EMISSION_EPSILON_M,
        "probe_dwell_time_s": SURFACE_BOUNDARY_PROBE_DWELL_TIME_S,
        "native_position_variants": [
            "exact_surface_anchor",
            "air_plus_epsilon",
            "solid_minus_epsilon",
        ],
        "evidence_sha256_by_variant": {
            "exact_surface_anchor": "1" * 64,
            "air_plus_epsilon": "2" * 64,
            "solid_minus_epsilon": "3" * 64,
        },
        "exact_anchor_vs_air_gate_passed": True,
        "solid_minus_air_gate_passed": True,
        "passed": True,
    }


def _source() -> dict[str, object]:
    """Return one valid continuous-surface Cs-137 source contract."""
    normal = [0.0, 0.0, 1.0]
    anchor = [3.0, 4.0, 0.0]
    transport = [
        anchor[index] + SURFACE_EMISSION_EPSILON_M * normal[index] for index in range(3)
    ]
    return {
        "isotope": "Cs-137",
        "position": anchor,
        "transport_position": transport,
        "intensity_cps_1m": 800_000.0,
        "surface_chart_id": 7,
        "surface_uv": [0.25, 0.75],
        "surface_normal": normal,
        "surface_emission_policy_sha256": (surface_emission_policy_sha256()),
    }


def _pair_payload(*, background_only: bool) -> dict[str, object]:
    """Return one exact pair payload for a source or background scenario."""
    model = _base_model()
    sources = [] if background_only else [_source()]
    source_count = len(sources)
    line_count = len(model.line_identity)
    if source_count == 0:
        geometry = {
            "unattenuated_source_line_rate_vsl": None,
            "uncollided_source_line_rate_vsl": None,
            "transport_features_vslf": None,
            "additive_scatter_basis_vslf": None,
            "perturbed_unattenuated_source_line_rate_vsl": None,
            "perturbed_uncollided_source_line_rate_vsl": None,
            "perturbed_transport_features_vslf": None,
            "perturbed_additive_scatter_basis_vslf": None,
        }
    else:
        unattenuated = np.zeros((1, 1, line_count), dtype=np.float64)
        uncollided = np.zeros_like(unattenuated)
        cs_line_index = next(
            index
            for index, line in enumerate(model.line_identity)
            if line["isotope"] == "Cs-137"
        )
        unattenuated[0, 0, cs_line_index] = 25_000.0
        uncollided[0, 0, cs_line_index] = 20_000.0
        geometry = {
            "unattenuated_source_line_rate_vsl": unattenuated.tolist(),
            "uncollided_source_line_rate_vsl": uncollided.tolist(),
            "transport_features_vslf": np.zeros(
                (1, 1, line_count, len(TRANSPORT_FEATURE_ORDER)),
                dtype=np.float64,
            ).tolist(),
            "additive_scatter_basis_vslf": np.zeros(
                (
                    1,
                    1,
                    line_count,
                    len(ADDITIVE_SCATTER_FEATURE_ORDER),
                ),
                dtype=np.float64,
            ).tolist(),
            "perturbed_unattenuated_source_line_rate_vsl": None,
            "perturbed_uncollided_source_line_rate_vsl": None,
            "perturbed_transport_features_vslf": None,
            "perturbed_additive_scatter_basis_vslf": None,
        }
    totals: dict[str, object] = {}
    hashes: dict[str, object] = {}
    for source_index, source in enumerate(sources):
        for line in model.line_identity:
            if line["isotope"] != source["isotope"]:
                continue
            token = native_source_line_token(
                source_index=source_index,
                isotope=str(source["isotope"]),
                energy_keV=float(line["energy_keV"]),
            )
            totals[token] = {
                "uncollided_primary": 10,
                "interacted_primary": 2,
                "secondary": 1,
            }
            hashes[token] = {
                "uncollided_primary": "4" * 64,
                "interacted_primary": "5" * 64,
                "secondary": "6" * 64,
            }
    scenario = "background_only" if background_only else "single_line_source_resolved"
    return {
        "schema_version": ACCEPTANCE_PAIR_SCHEMA_VERSION,
        "acceptance_contract_sha256": (FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256),
        "scene_seed": DESIGNATED_VALIDATION_SCENE_SEEDS[0],
        "split": "validation",
        "scenario_id": scenario,
        "shield_pair_id": 0,
        "transport_seed": acceptance_transport_seed(
            scene_seed=DESIGNATED_VALIDATION_SCENE_SEEDS[0],
            scenario_id=scenario,
            shield_pair_id=0,
        ),
        "dwell_time_s": STANDARD_ACQUISITION_LIVE_TIME_S,
        "scene_hash": "7" * 64,
        "surface_source_contract_sha256": (
            surface_source_runtime_contract_sha256(sources)
        ),
        "surface_boundary_gate": _boundary_gate(),
        "detector_pose_xyz": [1.0, 2.0, 1.0],
        "sources": sources,
        "line_identity_contract_sha256": (line_identity_contract_sha256(model)),
        "observed_spectrum_counts": [0] * NATIVE_GEANT4_BIN_COUNT,
        "geometry": geometry,
        "validation_labels": {
            "label_space": ADDITIVE_SCATTER_INCIDENT_LABEL_SEMANTICS,
            "target_semantics": ADDITIVE_SCATTER_TARGET_SEMANTICS,
            "entry_class_totals_by_source_line": totals,
            "entry_spectrum_sha256_by_source_line_class": hashes,
            "background_entry_total": 0,
            "background_entry_spectrum_sha256": "8" * 64,
        },
        "native_fidelity": dict(NATIVE_ACCEPTANCE_FIDELITY),
        "native_process_diagnostics": {
            "process_count_compton": 0,
            "process_count_rayleigh": 0,
            "process_count_photoelectric": 0,
            "transport_process_counts": (
                {} if background_only else {"Transportation": 1}
            ),
            "detector_response_coincidence_semantics": (
                DETECTOR_GREEN_COINCIDENCE_SEMANTICS
            ),
            "detector_response_incident_entry_count": 0,
            "detector_response_registered_entry_count": 0,
            "detector_response_coincidence_pulse_count": 0,
            "detector_response_multi_entry_pulse_count": 0,
            "pre_dead_time_total_pulse_count": 0,
            "post_dead_time_total_pulse_count": 0,
            "dead_time_observed_scale": 1.0,
        },
        "geometry_family": {
            "schema_version": GEOMETRY_FAMILY_SCHEMA_VERSION,
            "geometry_family_id": GEOMETRY_FAMILY_ID,
            "generator_algorithm_id": GEOMETRY_GENERATOR_ALGORITHM_ID,
            "transport_representation": "explicit_material_component_boxes",
            "room_size_xyz_m": [10.0, 20.0, 10.0],
            "cell_size_m": 1.0,
            "target_blocked_fraction": 0.4,
            "realized_blocked_fraction": 0.3,
            "passage_width_m": 2.0,
            "obstacle_height_limit_fraction": 0.5,
            "realized_max_component_height_fraction": 0.1,
            "instance_count": 1,
            "transport_component_count": 1,
            "template_names": ["fake_hollow_obstacle"],
            "component_materials": ["concrete"],
            "component_geometry_sha256": "9" * 64,
            "applicability_contract_sha256": (GEOMETRY_FAMILY_APPLICABILITY_SHA256),
        },
        "detector_green_operator_contract_sha256": (
            _TEST_OPERATOR.contract_hash_sha256
        ),
        "detector_green_operator_binary_sha256": (_TEST_OPERATOR.binary_sha256),
        "shield_pose_contract_sha256": SHIELD_POSE_CONTRACT_SHA256,
        "obstacle_material_contract_sha256": (OBSTACLE_MATERIAL_CONTRACT_SHA256),
        "transport_physics_table_contract_sha256": (
            TRANSPORT_PHYSICS_TABLE_CONTRACT_SHA256
        ),
    }


def _write_pair(path: Path, payload: dict[str, object]) -> Path:
    """Write one canonical test artifact."""
    path.write_bytes(canonical_json_bytes(payload))
    return path


@pytest.mark.parametrize("background_only", (True, False))
def test_pair_loader_accepts_exact_background_and_source_contracts(
    tmp_path: Path,
    background_only: bool,
) -> None:
    """Zero-source and nonzero-source tensors must reconstruct unambiguously."""
    model = _base_model()
    record = load_acceptance_pair(
        _write_pair(
            tmp_path / "pair.json",
            _pair_payload(background_only=background_only),
        ),
        expected_line_identity_sha256=line_identity_contract_sha256(model),
    )

    assert record.source_count == (0 if background_only else 1)
    assert record.unattenuated_vsl.shape == (
        1,
        record.source_count,
        len(model.line_identity),
    )


def test_pair_loader_rejects_transport_processes_without_sources(
    tmp_path: Path,
) -> None:
    """Persisted background evidence must have an exact empty process map."""
    payload = _pair_payload(background_only=True)
    payload["native_process_diagnostics"]["transport_process_counts"] = {
        "Transportation": 1
    }

    with pytest.raises(ValueError, match="Background-only"):
        load_acceptance_pair(
            _write_pair(tmp_path / "pair.json", payload),
            expected_line_identity_sha256=(
                line_identity_contract_sha256(_base_model())
            ),
        )


def test_pair_loader_rejects_empty_transport_processes_with_sources(
    tmp_path: Path,
) -> None:
    """Persisted source evidence must prove positive native transport."""
    payload = _pair_payload(background_only=False)
    payload["native_process_diagnostics"]["transport_process_counts"] = {}

    with pytest.raises(ValueError, match="Source-bearing"):
        load_acceptance_pair(
            _write_pair(tmp_path / "pair.json", payload),
            expected_line_identity_sha256=(
                line_identity_contract_sha256(_base_model())
            ),
        )


def test_pair_loader_rejects_retired_coincidence_semantics(
    tmp_path: Path,
) -> None:
    """Persisted pairs must not acquire a compatibility adapter for old marking."""
    payload = _pair_payload(background_only=False)
    payload["native_process_diagnostics"]["detector_response_coincidence_semantics"] = (
        "per_incident_gamma_without_same_history_coincidence"
    )

    with pytest.raises(ValueError, match="incompatible coincidence semantics"):
        load_acceptance_pair(
            _write_pair(tmp_path / "pair.json", payload),
            expected_line_identity_sha256=(
                line_identity_contract_sha256(_base_model())
            ),
        )


def test_pair_loader_rejects_inconsistent_coincidence_counters(
    tmp_path: Path,
) -> None:
    """Persisted pulse counters must prove their many-entry reductions."""
    payload = _pair_payload(background_only=False)
    diagnostics = payload["native_process_diagnostics"]
    diagnostics["detector_response_incident_entry_count"] = 2
    diagnostics["detector_response_registered_entry_count"] = 2
    diagnostics["detector_response_coincidence_pulse_count"] = 2
    diagnostics["detector_response_multi_entry_pulse_count"] = 1

    with pytest.raises(ValueError, match="inconsistent coincidence counters"):
        load_acceptance_pair(
            _write_pair(tmp_path / "pair.json", payload),
            expected_line_identity_sha256=(
                line_identity_contract_sha256(_base_model())
            ),
        )


def test_pair_loader_rejects_inconsistent_dead_time_scale(tmp_path: Path) -> None:
    """Persisted dead-time scale must equal the exact realized pulse ratio."""
    payload = _pair_payload(background_only=True)
    diagnostics = payload["native_process_diagnostics"]
    diagnostics["pre_dead_time_total_pulse_count"] = 2
    diagnostics["post_dead_time_total_pulse_count"] = 1
    diagnostics["dead_time_observed_scale"] = 0.75

    with pytest.raises(ValueError, match="inconsistent dead-time"):
        load_acceptance_pair(
            _write_pair(tmp_path / "pair.json", payload),
            expected_line_identity_sha256=(
                line_identity_contract_sha256(_base_model())
            ),
        )


def test_pair_loader_rejects_dead_time_count_not_bound_to_spectrum(
    tmp_path: Path,
) -> None:
    """Pre/post pulse counts must bind response, background, and spectrum totals."""
    payload = _pair_payload(background_only=True)
    diagnostics = payload["native_process_diagnostics"]
    diagnostics["pre_dead_time_total_pulse_count"] = 1
    diagnostics["post_dead_time_total_pulse_count"] = 1
    diagnostics["dead_time_observed_scale"] = 1.0

    with pytest.raises(ValueError, match="pulse counts disagree"):
        load_acceptance_pair(
            _write_pair(tmp_path / "pair.json", payload),
            expected_line_identity_sha256=(
                line_identity_contract_sha256(_base_model())
            ),
        )


def test_pair_loader_rejects_retired_schema_one_artifacts(tmp_path: Path) -> None:
    """The current acceptance command must have no legacy pair adapter."""
    payload = _pair_payload(background_only=True)
    payload["schema_version"] = 1

    with pytest.raises(ValueError, match="contract identity"):
        load_acceptance_pair(
            _write_pair(tmp_path / "legacy_pair.json", payload),
            expected_line_identity_sha256=(
                line_identity_contract_sha256(_base_model())
            ),
        )


def test_pair_loader_rejects_numeric_string_geometry(tmp_path: Path) -> None:
    """Geometry arrays must not gain physical meaning through float coercion."""
    payload = _pair_payload(background_only=False)
    payload["geometry"]["unattenuated_source_line_rate_vsl"][0][0][0] = "0"

    with pytest.raises(TypeError, match="JSON numbers"):
        load_acceptance_pair(
            _write_pair(tmp_path / "pair.json", payload),
            expected_line_identity_sha256=(
                line_identity_contract_sha256(_base_model())
            ),
        )


def test_pair_loader_authenticates_embedded_source_payload(tmp_path: Path) -> None:
    """A source coordinate change must invalidate the source-contract digest."""
    payload = _pair_payload(background_only=False)
    payload["sources"][0]["surface_uv"][0] = 0.5

    with pytest.raises(ValueError, match="source hash"):
        load_acceptance_pair(
            _write_pair(tmp_path / "pair.json", payload),
            expected_line_identity_sha256=(
                line_identity_contract_sha256(_base_model())
            ),
        )


def test_pair_loader_requires_every_native_source_line_label(
    tmp_path: Path,
) -> None:
    """Missing entry-class labels must not silently become zero training data."""
    payload = _pair_payload(background_only=False)
    totals = payload["validation_labels"]["entry_class_totals_by_source_line"]
    token = next(iter(totals))
    del totals[token]

    with pytest.raises(ValueError, match="label payload"):
        load_acceptance_pair(
            _write_pair(tmp_path / "pair.json", payload),
            expected_line_identity_sha256=(
                line_identity_contract_sha256(_base_model())
            ),
        )


def test_pair_loader_requires_null_for_absent_tensor_axes(
    tmp_path: Path,
) -> None:
    """JSON empty lists must not masquerade as a higher-rank zero-source tensor."""
    payload = _pair_payload(background_only=True)
    payload["geometry"]["unattenuated_source_line_rate_vsl"] = []

    with pytest.raises(ValueError, match="Background-only geometry"):
        load_acceptance_pair(
            _write_pair(tmp_path / "pair.json", payload),
            expected_line_identity_sha256=(
                line_identity_contract_sha256(_base_model())
            ),
        )


def test_pair_loader_requires_distinct_boundary_probe_evidence(
    tmp_path: Path,
) -> None:
    """One reused digest cannot authenticate three signed native probes."""
    payload = copy.deepcopy(_pair_payload(background_only=True))
    payload["surface_boundary_gate"]["evidence_sha256_by_variant"] = {
        name: "1" * 64
        for name in (
            "exact_surface_anchor",
            "air_plus_epsilon",
            "solid_minus_epsilon",
        )
    }

    with pytest.raises(ValueError, match="boundary gate"):
        load_acceptance_pair(
            _write_pair(tmp_path / "pair.json", payload),
            expected_line_identity_sha256=(
                line_identity_contract_sha256(_base_model())
            ),
        )


def test_pair_artifacts_remain_strict_json(tmp_path: Path) -> None:
    """The test fixture itself must use only round-trippable JSON values."""
    payload = _pair_payload(background_only=False)
    assert json.loads(canonical_json_bytes(payload)) == payload


def test_pair_loader_rejects_noncanonical_duplicate_keys(
    tmp_path: Path,
) -> None:
    """A resumable checkpoint must never use last-key-wins semantics."""
    payload = _pair_payload(background_only=False)
    canonical = canonical_json_bytes(payload).decode("utf-8")
    raw = ('{"schema_version":2,' + canonical.removeprefix("{")).encode("utf-8")
    path = tmp_path / "duplicate.json"
    path.write_bytes(raw)

    with pytest.raises(ValueError, match="canonical JSON"):
        load_acceptance_pair(
            path,
            expected_line_identity_sha256=(
                line_identity_contract_sha256(_base_model())
            ),
        )
