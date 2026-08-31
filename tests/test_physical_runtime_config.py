"""Ownership and fidelity checks for estimator-neutral runtime configuration."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from runtime.session import (
    _native_executable_sha256,
    estimator_neutral_physical_runtime_config,
    estimator_neutral_runtime_config,
    require_production_runtime_preflight,
)
from runtime.forward_model_manifest import forward_model_component_payloads
from measurement.observation_model import build_runtime_observation_model
from measurement.observation_model import build_nonproduction_observation_model
from spectrum.air_attenuation import (
    NIST_XCOM_DRY_AIR_TOTAL_CONTRACT_ID,
    NIST_XCOM_DRY_AIR_TOTAL_CONTRACT_SHA256,
)
from sim.geant4_app.app import Geant4Application
from sim.isaacsim_app.scene_builder import SceneDescription
from sim.runtime import load_production_runtime_config, load_runtime_config
from spectrum.transport_spectral import (
    CATALOG_INDEPENDENT_APPROVAL_SCOPE,
    GeometryConditionedSpectralModel,
)
from spectrum.additive_scatter import PhysicsOnlyNoncollidedTransportResponse
from tests.runtime_test_support import (
    approved_full_spectrum_model,
    runtime_config,
)


ROOT = Path(__file__).resolve().parents[1]
STANDARD_CONFIG = (
    ROOT / "configs" / "geant4" / "variance_reduction_external_no_isaac_32threads.json"
)


def test_standard_runtime_config_is_estimator_neutral() -> None:
    """The shared physical config must not contain PF or MLE controls."""
    payload = load_runtime_config(STANDARD_CONFIG)

    forbidden_prefixes = ("pf_", "mle_", "dss_", "structural_rj_")
    assert not [key for key in payload if str(key).startswith(forbidden_prefixes)]
    assert "estimator_profile" not in payload
    assert "pure_pf_schema_version" not in payload
    assert "variable_cardinality" not in payload


def test_combined_config_is_rejected_at_the_acquisition_boundary() -> None:
    """PF controls in a legacy combined config must fail production preflight."""
    payload = load_runtime_config(STANDARD_CONFIG)
    payload.update(
        {
            "cui_truth_display_mode": "evaluation_live",
            "joint_strength_block_probability": 0.25,
            "max_temper_steps": 256,
            "num_particles": 2000,
            "pf_max_sources": 5,
            "pure_pf_schema_version": 1,
            "structural_rj_merge_probability": 0.1,
            "target_ess_ratio": 0.4,
            "variable_cardinality": True,
        }
    )

    with pytest.raises(ValueError, match="estimator-owned or retired"):
        estimator_neutral_physical_runtime_config(payload)


def test_standard_runtime_preserves_full_transport_fidelity() -> None:
    """Repository separation must not introduce a lower-fidelity transport mode."""
    payload = load_runtime_config(STANDARD_CONFIG)

    assert payload["source_rate_model"] == "detector_cps_1m"
    assert payload["primary_sampling_fraction"] == pytest.approx(1.0)
    assert payload.get("accelerated_weighted_transport_enable", False) is False
    assert payload.get("weighted_transport", False) is False
    assert payload.get("theory_tvl_attenuation", False) is False
    assert payload["secondary_transport_mode"] == "full_transport"
    assert payload["sample_detector_response"] is True


def test_production_geant4_reset_rejects_scene_config_overrides() -> None:
    """Production reset must reject mismatches without mutating the scene."""
    app = Geant4Application.__new__(Geant4Application)
    app.production_runtime_config_sha256 = "0" * 64
    app.config = SimpleNamespace(
        usd_path="/approved/room.usda",
        author_obstacle_prims=True,
        author_room_boundary_prims=True,
    )
    fallback_scene = SceneDescription(
        usd_path=None,
        use_config_usd_fallback=True,
        author_obstacle_prims=True,
        author_room_boundary_prims=True,
    )

    with pytest.raises(ValueError, match="forbids config USD fallback"):
        app.reset(fallback_scene)

    assert fallback_scene.usd_path is None
    mismatched_authoring = SceneDescription(
        usd_path="/approved/room.usda",
        use_config_usd_fallback=False,
        author_obstacle_prims=False,
        author_room_boundary_prims=True,
    )

    with pytest.raises(ValueError, match="author_obstacle_prims differs"):
        app.reset(mismatched_authoring)

    assert mismatched_authoring.author_obstacle_prims is False


def test_standard_profile_reuses_only_catalog_independent_approval() -> None:
    """An in-domain profile may reuse physics approval without claiming all-64."""
    payload = load_runtime_config(STANDARD_CONFIG)

    resolved = estimator_neutral_runtime_config(
        payload,
        backend="geant4",
        isotopes=("Co-60", "Cs-137", "Eu-154"),
        run_root=ROOT,
    )
    model_payload = resolved["full_spectrum_generative_model"]
    validation = model_payload["validation"]

    assert model_payload["production_ready"] is True
    assert validation["schema_version"] == 7
    assert validation["approval_scope"] == CATALOG_INDEPENDENT_APPROVAL_SCOPE
    assert validation["application_validation_isotopes"] == ["Co-60", "Cs-137"]
    assert validation["approved_model_contract_sha256"] != (
        model_payload["contract_hash_sha256"]
    )


def test_production_session_rejects_analytic_requested_backend() -> None:
    """An exact Geant4 config cannot authorize an approximate runtime backend."""
    payload = load_runtime_config(STANDARD_CONFIG)

    with pytest.raises(ValueError, match="requires backend='geant4'"):
        estimator_neutral_runtime_config(
            payload,
            backend="analytic",
            isotopes=("Co-60", "Cs-137", "Eu-154"),
            run_root=ROOT,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("background_rate", "differs from the approved model rate"),
        ("background_model", "background_spectrum_model_id"),
    ),
)
def test_production_preflight_rejects_background_contract_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    """Transport background settings must match one approved spectral contract."""
    payload = load_production_runtime_config(STANDARD_CONFIG)
    approved = approved_full_spectrum_model()
    monkeypatch.setattr(
        "runtime.session.geometry_conditioned_model_from_runtime_config",
        lambda *args, **kwargs: approved,
    )
    payload["background_cps"] = float(approved.background_rate_cps)
    if mutation == "background_rate":
        payload["background_cps"] = float(approved.background_rate_cps) + 1.0
    else:
        payload["background_spectrum_model_id"] = "wrong-background-model"

    with pytest.raises(ValueError, match=message):
        require_production_runtime_preflight(
            payload,
            requested_backend="geant4",
        )


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("auto_start_sidecar", False),
        ("author_obstacle_prims", False),
        ("author_room_boundary_prims", False),
        ("detector_scoring_mode", "energy_deposit"),
        ("line_resolved_shield_attenuation", False),
        ("obstacle_attenuation_enabled", False),
        ("primary_emission_model", "geant4_radioactive_decay"),
        ("primary_sampling_fraction", 0.5),
        ("sample_detector_response", False),
        ("secondary_transport_mode", "primary_only"),
        ("source_bias_cone_policy", "fixed_angle"),
        ("source_bias_isotropic_fraction", 0.5),
        ("source_bias_mode", "analog"),
        ("source_rate_model", "activity_bq"),
    ),
)
def test_production_preflight_rejects_lower_fidelity_transport(
    field: str,
    invalid: object,
) -> None:
    """Canonical production cannot disable required full-spectrum physics."""
    payload = load_runtime_config(STANDARD_CONFIG)
    payload[field] = invalid

    with pytest.raises(ValueError, match="transport invariants"):
        require_production_runtime_preflight(
            payload,
            requested_backend="geant4",
        )


def test_native_executable_digest_hashes_one_exact_regular_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production executable identity must come from the selected file bytes."""
    executable = tmp_path / "geant4_sidecar"
    executable.write_bytes(b"approved-native-geant4-build")
    executable.chmod(0o755)
    monkeypatch.setattr(
        "runtime.session._RUNTIME_REPOSITORY_ROOT",
        tmp_path,
    )

    digest = _native_executable_sha256({"executable_path": executable.name})

    assert digest == sha256(executable.read_bytes()).hexdigest()
    symlink = tmp_path / "geant4_sidecar_link"
    symlink.symlink_to(executable)
    with pytest.raises(ValueError, match="cannot be a symlink"):
        _native_executable_sha256({"executable_path": symlink.name})


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("native", "native Geant4 executable SHA-256"),
        ("environment", "execution-environment SHA-256"),
        ("implementation", "implementation bundle SHA-256"),
    ),
)
def test_production_preflight_rejects_unapproved_execution_bundle(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    """A changed native binary or Python transport bundle must abort startup."""
    payload = load_runtime_config(STANDARD_CONFIG)
    approved = approved_full_spectrum_model()
    validation = approved.validation_manifest
    assert validation is not None
    monkeypatch.setattr(
        "runtime.session.geometry_conditioned_model_from_runtime_config",
        lambda *args, **kwargs: approved,
    )
    payload["background_cps"] = float(approved.background_rate_cps)
    monkeypatch.setattr(
        "runtime.session._native_executable_sha256",
        lambda _config: (
            "0" * 64 if mutation == "native" else validation["native_executable_sha256"]
        ),
    )
    monkeypatch.setattr(
        "runtime.session.native_execution_environment_bundle_sha256",
        lambda _path: (
            "0" * 64
            if mutation == "environment"
            else validation["native_execution_environment_sha256"]
        ),
    )
    monkeypatch.setattr(
        "runtime.session.acceptance_implementation_bundle_sha256",
        lambda _root: (
            "0" * 64
            if mutation == "implementation"
            else validation["implementation_bundle_sha256"]
        ),
    )

    with pytest.raises(RuntimeError, match=message):
        require_production_runtime_preflight(
            payload,
            requested_backend="geant4",
        )


def test_nonalgorithm_runtime_config_change_does_not_expire_model_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Path and orchestration changes may reuse the same approved algorithm."""
    payload = load_production_runtime_config(STANDARD_CONFIG)
    approved = approved_full_spectrum_model()
    validation = approved.validation_manifest
    assert validation is not None
    monkeypatch.setattr(
        "runtime.session.geometry_conditioned_model_from_runtime_config",
        lambda *args, **kwargs: approved,
    )
    payload["background_cps"] = float(approved.background_rate_cps)
    monkeypatch.setattr(
        "runtime.session._native_executable_sha256",
        lambda _config: validation["native_executable_sha256"],
    )
    monkeypatch.setattr(
        "runtime.session.native_execution_environment_bundle_sha256",
        lambda _path: validation["native_execution_environment_sha256"],
    )
    monkeypatch.setattr(
        "runtime.session.acceptance_implementation_bundle_sha256",
        lambda _root: validation["implementation_bundle_sha256"],
    )

    result = require_production_runtime_preflight(
        payload,
        requested_backend="geant4",
    )

    assert result is approved


def test_approved_profile_resolves_once_for_estimator_neutral_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An approved model must produce one immutable logged contract."""
    payload = load_runtime_config(STANDARD_CONFIG)
    approved = approved_full_spectrum_model()
    validation = approved.validation_manifest
    assert validation is not None

    def resolve_approved_profile(
        config: dict[str, object],
        *,
        run_root: Path,
    ) -> dict[str, object]:
        """Embed the approved synthetic model without altering physical fields."""
        del run_root
        resolved = dict(config)
        resolved["full_spectrum_generative_model"] = approved.manifest_payload()
        resolved["full_spectrum_contract_hash_sha256"] = approved.contract_hash_sha256
        return resolved

    monkeypatch.setattr(
        "runtime.session.geometry_conditioned_model_from_runtime_config",
        lambda *args, **kwargs: approved,
    )
    payload["background_cps"] = float(approved.background_rate_cps)
    monkeypatch.setattr(
        "runtime.session.resolve_profile_model_runtime_config",
        resolve_approved_profile,
    )
    monkeypatch.setattr(
        "runtime.session._native_executable_sha256",
        lambda _config: validation["native_executable_sha256"],
    )
    monkeypatch.setattr(
        "runtime.session.native_execution_environment_bundle_sha256",
        lambda _path: validation["native_execution_environment_sha256"],
    )
    monkeypatch.setattr(
        "runtime.session.acceptance_implementation_bundle_sha256",
        lambda _root: validation["implementation_bundle_sha256"],
    )
    resolved = estimator_neutral_runtime_config(
        payload,
        backend="geant4",
        isotopes=("Co-60", "Cs-137", "Eu-154"),
        run_root=ROOT,
    )

    assert resolved["simulation_runtime_schema_version"] == 1
    assert resolved["candidate_isotopes"] == ["Co-60", "Cs-137", "Eu-154"]
    assert "full_spectrum_generative_model" in resolved
    assert "full_spectrum_generative_model_path" not in resolved
    assert "full_spectrum_model_registry_path" not in resolved
    assert "full_spectrum_model_registry_file_sha256" not in resolved
    assert "isotope_experiment_profile" not in resolved
    assert "full_spectrum_profile_calibration_status" not in resolved
    model = GeometryConditionedSpectralModel.from_manifest_payload(
        resolved["full_spectrum_generative_model"],
        detector_green_operator=approved.detector_green_operator,
    )
    assert model.runtime_ready is True
    assert model.production_ready is True
    assert (
        model.contract_hash_sha256
        == approved_full_spectrum_model().contract_hash_sha256
    )


def test_standard_profile_reaches_shared_continuous_kernel_contract() -> None:
    """Only explicit non-production tooling may inspect an unapproved model."""
    payload = load_runtime_config(STANDARD_CONFIG)

    with pytest.raises(RuntimeError, match="independent all-64 validation"):
        build_runtime_observation_model(
            payload,
            isotopes=("Co-60", "Cs-137", "Eu-154"),
        )

    observation = build_nonproduction_observation_model(
        payload,
        isotopes=("Co-60", "Cs-137", "Eu-154"),
    )

    assert isinstance(
        observation.additive_scatter_response,
        PhysicsOnlyNoncollidedTransportResponse,
    )
    assert "air_xcom" in (observation.additive_scatter_response.feature_basis_semantics)
    assert (
        observation.dry_air_total_attenuation_contract_id
        == NIST_XCOM_DRY_AIR_TOTAL_CONTRACT_ID
    )
    assert (
        observation.dry_air_total_attenuation_contract_sha256
        == NIST_XCOM_DRY_AIR_TOTAL_CONTRACT_SHA256
    )


def test_runtime_observation_model_requires_literal_production_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A truthy non-boolean flag cannot authorize production observations."""
    payload = runtime_config()
    model = approved_full_spectrum_model()
    monkeypatch.setattr(
        type(model),
        "production_ready",
        property(lambda _self: "true"),
    )
    monkeypatch.setattr(
        type(model),
        "require_production_ready",
        lambda _self: None,
    )

    with pytest.raises(RuntimeError, match="production_ready=False"):
        build_runtime_observation_model(
            payload,
            isotopes=("Co-60", "Cs-137", "Eu-154"),
            authenticated_full_spectrum_model=model,
        )


def test_production_observation_model_requires_one_spectrum_model() -> None:
    """Production cannot silently construct a count-only observation model."""
    payload = runtime_config()
    del payload["full_spectrum_generative_model"]
    del payload["full_spectrum_contract_hash_sha256"]

    with pytest.raises(RuntimeError, match="authenticated full-spectrum model"):
        build_runtime_observation_model(
            payload,
            isotopes=("Co-60", "Cs-137", "Eu-154"),
        )


@pytest.mark.parametrize(
    "mutation",
    ("missing_source_rate", "missing_line_flag", "disabled_line_transport"),
)
def test_production_observation_model_rejects_implicit_line_semantics(
    mutation: str,
) -> None:
    """Catalog and source-rate semantics must be explicit and line resolved."""
    payload = runtime_config()
    model = approved_full_spectrum_model()
    if mutation == "missing_source_rate":
        del payload["source_rate_model"]
    elif mutation == "missing_line_flag":
        del payload["line_resolved_shield_attenuation"]
    else:
        payload["line_resolved_shield_attenuation"] = False

    with pytest.raises(
        (TypeError, ValueError), match="source_rate_model|line.resolved"
    ):
        build_runtime_observation_model(
            payload,
            isotopes=("Co-60", "Cs-137", "Eu-154"),
            authenticated_full_spectrum_model=model,
        )


def test_archived_embedded_model_uses_logged_registry_digest() -> None:
    """Registry updates must not invalidate a self-contained archived log."""
    declared_digest = "a" * 64
    payloads = forward_model_component_payloads(
        runtime_config={
            "full_spectrum_generative_model": {"schema_version": 3},
            "full_spectrum_contract_hash_sha256": "b" * 64,
            "full_spectrum_model_registry_path": (
                "configs/geant4/models/historical_registry.json"
            ),
            "full_spectrum_model_registry_file_sha256": declared_digest,
        },
        environment={},
        obstacle_layout_path=None,
        isotopes=("Co-60", "Cs-137", "Eu-154"),
        run_root=None,
        repository_root=ROOT,
    )

    assert payloads["spectrum"]["file_assets"] == {
        "full_spectrum_model_registry_path": {
            "path": "configs/geant4/models/historical_registry.json",
            "sha256": declared_digest,
        }
    }
