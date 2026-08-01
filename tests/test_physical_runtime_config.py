"""Ownership and fidelity checks for estimator-neutral runtime configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from runtime.session import estimator_neutral_runtime_config
from sim.runtime import load_runtime_config


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
