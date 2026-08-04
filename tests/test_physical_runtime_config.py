"""Ownership and fidelity checks for estimator-neutral runtime configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from runtime.session import (
    estimator_neutral_physical_runtime_config,
    estimator_neutral_runtime_config,
)
from runtime.forward_model_manifest import forward_model_component_payloads
from measurement.observation_model import build_runtime_observation_model
from sim.runtime import load_runtime_config
from spectrum.transport_spectral import GeometryConditionedSpectralModel
from spectrum.additive_scatter import PhysicsOnlyNoncollidedTransportResponse


ROOT = Path(__file__).resolve().parents[1]
STANDARD_CONFIG = (
    ROOT
    / "configs"
    / "geant4"
    / "variance_reduction_external_no_isaac_32threads.json"
)


def test_standard_runtime_config_is_estimator_neutral() -> None:
    """The shared physical config must not contain PF or MLE controls."""
    payload = load_runtime_config(STANDARD_CONFIG)

    forbidden_prefixes = ("pf_", "mle_", "dss_", "structural_rj_")
    assert not [
        key for key in payload if str(key).startswith(forbidden_prefixes)
    ]
    assert "estimator_profile" not in payload
    assert "pure_pf_schema_version" not in payload
    assert "variable_cardinality" not in payload


def test_combined_config_is_split_at_the_acquisition_boundary() -> None:
    """PF controls in a legacy combined config must not reach acquisition."""
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

    physical = estimator_neutral_physical_runtime_config(payload)

    assert physical["backend"] == "geant4"
    assert physical["primary_sampling_fraction"] == pytest.approx(1.0)
    assert "cui_truth_display_mode" not in physical
    assert "joint_strength_block_probability" not in physical
    assert "max_temper_steps" not in physical
    assert "num_particles" not in physical
    assert "pf_max_sources" not in physical
    assert "pure_pf_schema_version" not in physical
    assert "structural_rj_merge_probability" not in physical
    assert "target_ess_ratio" not in physical
    assert "variable_cardinality" not in physical


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


def test_standard_profile_resolves_once_for_estimator_neutral_log() -> None:
    """Profile registry selection must produce one immutable logged model."""
    payload = load_runtime_config(STANDARD_CONFIG)

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
        resolved["full_spectrum_generative_model"]
    )
    assert isinstance(
        model.additive_scatter_response,
        PhysicsOnlyNoncollidedTransportResponse,
    )
    assert model.discrepancy_training_manifest is None
    assert model.low_rank_spectral_mean_correction is None
    assert model.runtime_ready is True


def test_standard_profile_reaches_shared_continuous_kernel_contract() -> None:
    """Registry-backed physics must not disappear at observation construction."""
    payload = load_runtime_config(STANDARD_CONFIG)

    observation = build_runtime_observation_model(
        payload,
        isotopes=("Co-60", "Cs-137", "Eu-154"),
    )

    assert isinstance(
        observation.additive_scatter_response,
        PhysicsOnlyNoncollidedTransportResponse,
    )
    assert "air_xcom" in (
        observation.additive_scatter_response.feature_basis_semantics
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
