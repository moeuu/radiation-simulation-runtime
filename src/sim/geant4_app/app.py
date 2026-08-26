"""Geant4 sidecar application entry points."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from measurement.shielding import (
    SHIELD_POSE_CONTRACT_ID,
    SHIELD_POSE_CONTRACT_SHA256,
)
from measurement.source_boundary import (
    SURFACE_EMISSION_EPSILON_M,
    surface_emission_policy_sha256,
    surface_source_runtime_contract_sha256,
)
from sim.geant4_app.engine import (
    Geant4EngineConfig,
    Geant4StepRequest,
    build_geant4_engine,
)
from sim.geant4_app.execution_environment import (
    native_executable_sha256,
    native_execution_environment_bundle_sha256,
)
from sim.geant4_app.scene_export import (
    DEFAULT_DETECTOR_COINCIDENCE_WINDOW_S,
    DEFAULT_DETECTOR_CRYSTAL_LENGTH_M,
    DEFAULT_DETECTOR_CRYSTAL_RADIUS_M,
    DEFAULT_DETECTOR_HOUSING_THICKNESS_M,
    ExportedDetectorModel,
    export_scene_for_geant4,
)
from sim.isaacsim_app.app import IsaacAssetGeometry, StageMaterialRule
from sim.isaacsim_app.robot_controller import RobotController
from sim.isaacsim_app.scene_builder import SceneBuilder, SceneDescription
from sim.isaacsim_app.stage_backend import (
    FakeStageBackend,
    IsaacSimStageBackend,
    StageBackend,
)
from sim.protocol import SimulationCommand, SimulationObservation
from sim.radiation_visualization import RadiationVisualizationConfig
from sim.shield_geometry import ShieldThicknessConfig, resolve_shield_thickness_config
from spectrum.mean_calibration import (
    MEAN_CALIBRATION_ANALOG_COVARIANCE_SEMANTICS,
    MEAN_CALIBRATION_FORCED_COLLISION_COVARIANCE_SEMANTICS,
    parse_mean_calibration_metadata,
)
from spectrum.response_matrix import (
    NATIVE_GEANT4_BACKGROUND_MODEL_ID,
    NATIVE_GEANT4_BIN_COUNT,
    NATIVE_GEANT4_BIN_WIDTH_KEV,
    NATIVE_GEANT4_DETECTOR_RESPONSE_CONTRACT_SHA256,
    NATIVE_GEANT4_ENERGY_MAX_KEV,
    NATIVE_GEANT4_ENERGY_MIN_KEV,
)
from spectrum.geant4_physics import validate_geant4_physics_metadata


_MANAGED_GEANT4_EXECUTABLE_OPTIONS = frozenset(
    {
        "--background-cps",
        "--dead-time-tau-s",
        "--detector-scoring-mode",
        "--physics-profile",
        "--persistent",
        "--primary-emission-model",
        "--primary-sampling-fraction",
        "--target-sampled-primaries",
        "--mean-calibration-histories-per-source-line",
        "--mean-calibration-angle-strata-mu",
        "--mean-calibration-angle-strata-phi",
        "--mean-calibration-forced-collision",
        "--request",
        "--response",
        "--sample-detector-response",
        "--scene",
        "--secondary-transport-mode",
        "--source-bias-cone-half-angle-deg",
        "--source-bias-isotropic-fraction",
        "--source-bias-mode",
        "--source-rate-model",
        "--threads",
        "--validation-entry-class-spectra",
    }
)

_MIN_PRIMARY_SAMPLING_FRACTION = 1.0e-6


def require_primary_sampling_fraction(
    value: object,
    *,
    accelerated_weighted_transport_enable: bool = False,
    target_sampled_primaries: int | None = None,
) -> float:
    """Validate primary sampling and require an explicit weighted-mode opt-in."""
    if not isinstance(accelerated_weighted_transport_enable, bool):
        raise ValueError(
            "accelerated_weighted_transport_enable must be a JSON boolean."
        )
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("primary_sampling_fraction must be a JSON number.")
    fraction = float(value)
    if (
        not np.isfinite(fraction)
        or fraction < _MIN_PRIMARY_SAMPLING_FRACTION
        or fraction > 1.0
    ):
        raise ValueError("primary_sampling_fraction must be in the interval [1e-6, 1].")
    weighted_requested = fraction < 1.0
    if weighted_requested and not accelerated_weighted_transport_enable:
        raise ValueError(
            "Geant4 runtime requires primary_sampling_fraction=1.0; "
            "weighted history thinning requires the explicit "
            "accelerated_weighted_transport_enable=true opt-in."
        )
    if (
        accelerated_weighted_transport_enable
        and not weighted_requested
        and target_sampled_primaries is None
    ):
        raise ValueError(
            "accelerated_weighted_transport_enable=true requires "
            "primary_sampling_fraction<1.0 or target_sampled_primaries."
        )
    return fraction


def require_target_sampled_primaries(value: object) -> int | None:
    """Return a positive integer primary budget or the disabled sentinel."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("target_sampled_primaries must be a positive JSON integer.")
    if value <= 0:
        raise ValueError("target_sampled_primaries must be a positive JSON integer.")
    return int(value)


def require_full_history_primary_sampling_fraction(value: object) -> float:
    """Return a full-history fraction or reject weighted history thinning."""
    return require_primary_sampling_fraction(value)


def resolve_primary_sampling_fraction(
    maximum_fraction: float,
    target_sampled_primaries: int | None,
    expected_unthinned_primaries: float,
) -> tuple[float, str]:
    """Resolve an observation-specific sampling fraction and provenance label."""
    if target_sampled_primaries is None:
        return float(maximum_fraction), "fixed_fraction"
    if (
        not np.isfinite(expected_unthinned_primaries)
        or expected_unthinned_primaries < 0.0
    ):
        raise RuntimeError("Native Geant4 expected-primary provenance is invalid.")
    budget_fraction = (
        float(target_sampled_primaries) / expected_unthinned_primaries
        if expected_unthinned_primaries > 0.0
        else np.inf
    )
    if budget_fraction < maximum_fraction:
        return (
            float(
                np.clip(
                    budget_fraction,
                    _MIN_PRIMARY_SAMPLING_FRACTION,
                    1.0,
                )
            ),
            "target_budget_limited",
        )
    return float(maximum_fraction), "maximum_fraction_limited"


def validate_geant4_executable_args(values: tuple[str, ...]) -> tuple[str, ...]:
    """Reject executable arguments that override engine-managed fidelity options."""
    for value in values:
        option = str(value).split("=", maxsplit=1)[0]
        if option in _MANAGED_GEANT4_EXECUTABLE_OPTIONS:
            raise ValueError(
                f"executable_args cannot override managed Geant4 option {option}."
            )
    return values


def _required_metadata_bool(metadata: dict[str, Any], key: str) -> bool:
    """Return a strict boolean provenance value or fail closed."""
    if key not in metadata:
        raise RuntimeError(f"Native Geant4 response is missing {key} provenance.")
    value = metadata[key]
    if not isinstance(value, bool):
        raise RuntimeError(
            f"Native Geant4 response has invalid boolean {key}={value!r}."
        )
    return value


def _required_metadata_integer(metadata: dict[str, Any], key: str) -> int:
    """Return an exact JSON integer provenance value or fail closed."""
    if key not in metadata:
        raise RuntimeError(f"Native Geant4 response is missing {key} provenance.")
    value = metadata[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(
            f"Native Geant4 response has invalid integer {key}={value!r}."
        )
    return value


def _required_metadata_number(metadata: dict[str, Any], key: str) -> float:
    """Return one finite exact JSON number provenance value."""
    if key not in metadata:
        raise RuntimeError(f"Native Geant4 response is missing {key} provenance.")
    value = metadata[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(
            f"Native Geant4 response has invalid numeric {key}={value!r}."
        )
    parsed = float(value)
    if not np.isfinite(parsed):
        raise RuntimeError(
            f"Native Geant4 response has non-finite numeric {key}={value!r}."
        )
    return parsed


def _required_metadata_string(metadata: dict[str, Any], key: str) -> str:
    """Return one exact JSON string provenance value."""
    if key not in metadata:
        raise RuntimeError(f"Native Geant4 response is missing {key} provenance.")
    value = metadata[key]
    if not isinstance(value, str):
        raise RuntimeError(
            f"Native Geant4 response has invalid string {key}={value!r}."
        )
    return value


def _validate_native_weighted_response(
    spectrum_counts: object,
    metadata: dict[str, Any],
) -> None:
    """Validate weighted native spectrum and per-bin sum-w2 consistency."""
    try:
        spectrum = np.asarray(spectrum_counts, dtype=float)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Native Geant4 weighted spectrum is not numeric.") from exc
    if spectrum.ndim != 1 or spectrum.size == 0:
        raise RuntimeError(
            "Native Geant4 weighted spectrum must be a nonempty one-dimensional array."
        )
    if not np.all(np.isfinite(spectrum)):
        raise RuntimeError("Native Geant4 weighted spectrum is not finite.")
    if np.any(spectrum < 0.0):
        raise RuntimeError("Native Geant4 weighted spectrum contains negative counts.")

    try:
        reported_spectrum_total = float(metadata["total_spectrum_counts"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Native Geant4 weighted response is missing total_spectrum_counts."
        ) from exc
    spectrum_total = float(np.sum(spectrum, dtype=float))
    if not np.isfinite(reported_spectrum_total) or reported_spectrum_total < 0.0:
        raise RuntimeError("Native Geant4 weighted total_spectrum_counts is invalid.")
    if not np.isclose(
        spectrum_total,
        reported_spectrum_total,
        rtol=1.0e-9,
        atol=1.0e-6,
    ):
        raise RuntimeError(
            "Native Geant4 weighted spectrum sum is inconsistent with "
            "total_spectrum_counts."
        )

    try:
        spectrum_variance = np.asarray(
            metadata["spectrum_count_variance"],
            dtype=float,
        )
    except KeyError as exc:
        raise RuntimeError(
            "Native Geant4 weighted response is missing per-bin spectrum variance."
        ) from exc
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Native Geant4 weighted per-bin spectrum variance is not numeric."
        ) from exc
    if spectrum_variance.shape != spectrum.shape:
        raise RuntimeError(
            "Native Geant4 weighted per-bin spectrum variance shape does not match "
            "the spectrum."
        )
    if not np.all(np.isfinite(spectrum_variance)):
        raise RuntimeError(
            "Native Geant4 weighted per-bin spectrum variance is not finite."
        )
    if np.any(spectrum_variance < 0.0):
        raise RuntimeError(
            "Native Geant4 weighted per-bin spectrum variance is negative."
        )

    variance_total = float(np.sum(spectrum_variance, dtype=float))
    try:
        parsed_variance_total = float(metadata["spectrum_count_variance_total"])
        reported_variance_total = float(metadata["weighted_spectrum_sumw2"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Native Geant4 weighted response is missing aggregate sum-w2 provenance."
        ) from exc
    if not np.isfinite(parsed_variance_total) or parsed_variance_total < 0.0:
        raise RuntimeError("Native Geant4 spectrum_count_variance_total is invalid.")
    for reported_total, label in (
        (parsed_variance_total, "spectrum_count_variance_total"),
        (reported_variance_total, "weighted_spectrum_sumw2"),
    ):
        if not np.isfinite(reported_total) or reported_total < 0.0:
            raise RuntimeError(f"Native Geant4 {label} is invalid.")
        if not np.isclose(
            variance_total,
            reported_total,
            rtol=1.0e-9,
            atol=1.0e-8,
        ):
            raise RuntimeError(
                "Native Geant4 weighted per-bin spectrum variance sum is "
                f"inconsistent with {label}."
            )


def _validate_native_sampled_event_response(
    spectrum_counts: object,
    metadata: dict[str, Any],
) -> NDArray[np.int64]:
    """Return exact native sampled-event counts after provenance checks."""
    raw_spectrum = np.asarray(spectrum_counts)
    if raw_spectrum.dtype.kind not in {"i", "u", "f"}:
        raise RuntimeError(
            "Native Geant4 sampled-event spectrum must contain exact JSON "
            "numbers without string or boolean coercion."
        )
    spectrum = np.asarray(raw_spectrum, dtype=np.float64)
    if spectrum.ndim != 1 or spectrum.size == 0:
        raise RuntimeError(
            "Native Geant4 sampled-event spectrum must be a nonempty "
            "one-dimensional array."
        )
    if (
        np.any(~np.isfinite(spectrum))
        or np.any(spectrum < 0.0)
        or np.any(spectrum != np.rint(spectrum))
        or np.any(spectrum > float(2**53))
    ):
        raise RuntimeError(
            "Native Geant4 sampled detector response must contain exact "
            "nonnegative unit-weight integer event counts."
        )
    event_counts = np.ascontiguousarray(spectrum, dtype=np.int64)
    event_total = int(np.sum(event_counts, dtype=np.int64))
    reported_total = _required_metadata_number(
        metadata,
        "total_spectrum_counts",
    )
    reported_sumw2 = _required_metadata_number(
        metadata,
        "weighted_spectrum_sumw2",
    )
    if (
        not np.isfinite(reported_total)
        or not np.isfinite(reported_sumw2)
        or not np.isclose(
            reported_total,
            float(event_total),
            rtol=0.0,
            atol=1.0e-9,
        )
        or not np.isclose(
            reported_sumw2,
            float(event_total),
            rtol=0.0,
            atol=1.0e-9,
        )
    ):
        raise RuntimeError(
            "Native Geant4 sampled-event spectrum disagrees with its total "
            "count or unit-weight sum-w2 provenance."
        )
    return event_counts


def validate_transport_metadata(
    metadata: dict[str, Any],
    *,
    expected_primary_sampling_fraction: float = 1.0,
    expected_target_sampled_primaries: int | None = None,
    accelerated_weighted_transport_enable: bool = False,
    expected_source_rate_model: str | None = None,
    expected_primary_emission_model: str = "independent_gamma_lines",
    expected_thread_count: int | None = None,
    expected_physics_profile: str | None = None,
    expected_detector_scoring_mode: str | None = None,
    expected_secondary_transport_mode: str | None = None,
    expected_source_bias_mode: str | None = None,
    expected_background_cps: float | None = None,
    expected_dead_time_tau_s: float | None = None,
    expected_detector_response_sampling: bool = False,
    expected_surface_source_contract_sha256: str | None = None,
    expected_scene_hash: str | None = None,
) -> None:
    """Fail when native transport provenance disagrees with configured semantics."""
    if not isinstance(expected_detector_response_sampling, bool):
        raise ValueError("expected_detector_response_sampling must be a JSON boolean.")
    if expected_primary_emission_model not in {
        "independent_gamma_lines",
        "geant4_radioactive_decay",
    }:
        raise ValueError("expected_primary_emission_model has an unsupported value.")
    if expected_thread_count is not None and (
        isinstance(expected_thread_count, bool)
        or not isinstance(expected_thread_count, int)
        or expected_thread_count <= 0
    ):
        raise ValueError("expected_thread_count must be a positive JSON integer.")
    configured_fraction = require_primary_sampling_fraction(
        expected_primary_sampling_fraction,
        accelerated_weighted_transport_enable=(accelerated_weighted_transport_enable),
        target_sampled_primaries=expected_target_sampled_primaries,
    )
    expected_target = require_target_sampled_primaries(
        expected_target_sampled_primaries
    )
    if expected_target is not None and not accelerated_weighted_transport_enable:
        raise ValueError(
            "expected_target_sampled_primaries requires "
            "accelerated_weighted_transport_enable=true."
        )
    observed_fraction = _required_metadata_number(
        metadata,
        "primary_sampling_fraction",
    )
    observed_history_weight = _required_metadata_number(
        metadata,
        "primary_history_weight",
    )
    if (
        not np.isfinite(observed_fraction)
        or observed_fraction < _MIN_PRIMARY_SAMPLING_FRACTION
        or observed_fraction > 1.0
    ):
        raise RuntimeError("Native Geant4 primary sampling fraction is invalid.")
    if expected_target is None and not np.isclose(
        observed_fraction,
        configured_fraction,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "Native Geant4 response requires "
            f"primary_sampling_fraction={configured_fraction}, "
            f"got {observed_fraction}."
        )
    expected_observed_weight = 1.0 / observed_fraction
    if not np.isfinite(observed_history_weight) or not np.isclose(
        observed_history_weight,
        expected_observed_weight,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        if expected_target is None:
            raise RuntimeError(
                "Native Geant4 response requires "
                f"primary_history_weight={1.0 / configured_fraction}, "
                f"got {observed_history_weight}."
            )
        raise RuntimeError(
            "Native Geant4 primary sampling fraction and history weight are "
            "invalid or inconsistent."
        )

    if _required_metadata_string(metadata, "backend") != "geant4":
        raise RuntimeError("Native Geant4 response has invalid backend provenance.")
    if _required_metadata_string(metadata, "engine_mode") != "external":
        raise RuntimeError("Native Geant4 response has invalid engine_mode provenance.")

    source_rate_model = _required_metadata_string(
        metadata,
        "source_rate_model",
    )
    if (
        expected_source_rate_model is not None
        and source_rate_model != expected_source_rate_model
    ):
        raise RuntimeError(
            "Native Geant4 source-rate semantics disagree with runtime config: "
            f"expected {expected_source_rate_model}, got {source_rate_model or 'missing'}."
        )
    if source_rate_model not in {
        "detector_cps_1m",
        "isotropic_emission_equivalent",
        "parent_decay_activity_bq",
    }:
        raise RuntimeError(
            "Native Geant4 response has invalid source_rate_model provenance."
        )
    primary_emission_model = _required_metadata_string(
        metadata,
        "primary_emission_model",
    )
    if primary_emission_model != expected_primary_emission_model:
        raise RuntimeError(
            "Native Geant4 primary-emission provenance disagrees with runtime "
            f"config: expected {expected_primary_emission_model}, got "
            f"{primary_emission_model or 'missing'}."
        )
    radioactive_decay_emission = primary_emission_model == "geant4_radioactive_decay"
    if radioactive_decay_emission and source_rate_model != ("parent_decay_activity_bq"):
        raise RuntimeError(
            "Geant4 radioactive-decay emission requires parent activity in Bq."
        )
    history_thinning_enabled = observed_fraction < 1.0
    if history_thinning_enabled and source_rate_model != "detector_cps_1m":
        raise RuntimeError(
            "Accelerated weighted history thinning is currently restricted to "
            "source_rate_model=detector_cps_1m."
        )
    if radioactive_decay_emission:
        if (
            _required_metadata_string(
                metadata,
                "activity_bq_definition",
            )
            != "parent_decays_per_second_at_acquisition_start"
        ):
            raise RuntimeError(
                "Native Geant4 response has invalid activity_bq semantics."
            )
        if _required_metadata_string(
            metadata,
            "radioactive_source_state_semantics",
        ) != ("scheduled_parent_decays_no_preexisting_daughter_inventory"):
            raise RuntimeError(
                "Native Geant4 response has invalid radioactive source-state semantics."
            )
    elif (
        _required_metadata_string(
            metadata,
            "intensity_cps_1m_definition",
        )
        != "pre_dead_time_detector_pulse_rate_at_1m"
    ):
        raise RuntimeError(
            "Native Geant4 response has invalid intensity_cps_1m semantics."
        )
    if (
        _required_metadata_string(metadata, "source_position_semantics")
        != "air_side_native_emission_xyz"
        or _required_metadata_string(metadata, "source_anchor_semantics")
        != "exact_surface_chart_uv_evaluation_truth"
        or not _required_metadata_bool(
            metadata,
            "all_sources_surface_bound",
        )
        or _required_metadata_string(
            metadata,
            "surface_emission_policy_sha256",
        )
        != surface_emission_policy_sha256()
    ):
        raise RuntimeError(
            "Native Geant4 source positions do not satisfy the shared "
            "surface-anchor emission contract."
        )
    source_emission_epsilon_m = _required_metadata_number(
        metadata,
        "surface_emission_epsilon_m",
    )
    if not np.isclose(
        source_emission_epsilon_m,
        SURFACE_EMISSION_EPSILON_M,
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise RuntimeError(
            "Native Geant4 surface-emission epsilon differs from the shared model."
        )
    for key, expected_hash in (
        (
            "surface_source_contract_sha256",
            expected_surface_source_contract_sha256,
        ),
        ("scene_hash", expected_scene_hash),
    ):
        actual_hash = metadata.get(key)
        if (
            not isinstance(actual_hash, str)
            or len(actual_hash) != 64
            or any(character not in "0123456789abcdef" for character in actual_hash)
            or (expected_hash is not None and actual_hash != str(expected_hash))
        ):
            raise RuntimeError(f"Native Geant4 {key} is missing, invalid, or stale.")

    physics_profile = _required_metadata_string(
        metadata,
        "physics_profile",
    )
    validate_geant4_physics_metadata(metadata)
    if (
        expected_physics_profile is not None
        and physics_profile != expected_physics_profile
    ):
        raise RuntimeError(
            "Native Geant4 physics profile disagrees with runtime config: "
            f"expected {expected_physics_profile}, got {physics_profile or 'missing'}."
        )
    detector_scoring_mode = _required_metadata_string(
        metadata,
        "detector_scoring_mode",
    )
    if (
        expected_detector_scoring_mode is not None
        and detector_scoring_mode != expected_detector_scoring_mode
    ):
        raise RuntimeError(
            "Native Geant4 detector scoring disagrees with runtime config: "
            f"expected {expected_detector_scoring_mode}, "
            f"got {detector_scoring_mode or 'missing'}."
        )
    detector_response_applied = _required_metadata_bool(
        metadata,
        "detector_response_applied_in_native",
    )
    response_sampling_mode = _required_metadata_string(
        metadata,
        "detector_response_sampling_mode",
    )
    response_sampling_enabled = (
        response_sampling_mode == "multinomial_marking_with_nonparalyzable_event_time"
    )
    if response_sampling_enabled != bool(expected_detector_response_sampling):
        raise RuntimeError(
            "Native Geant4 detector-response sampling disagrees with runtime "
            "configuration."
        )
    if response_sampling_enabled and detector_scoring_mode != ("incident_gamma_energy"):
        raise RuntimeError(
            "Native detector-response marking requires incident-gamma scoring."
        )
    if (
        response_sampling_enabled
        and _required_metadata_string(
            metadata,
            "detector_response_sampling_contract_sha256",
        )
        != NATIVE_GEANT4_DETECTOR_RESPONSE_CONTRACT_SHA256
    ):
        raise RuntimeError(
            "Native detector-response sampling contract differs from the "
            "shared full-spectrum model."
        )
    if detector_response_applied != (
        detector_scoring_mode != "incident_gamma_energy" or response_sampling_enabled
    ):
        raise RuntimeError(
            "Native Geant4 detector-response provenance is inconsistent with "
            "detector_scoring_mode."
        )
    energy_min_keV = _required_metadata_number(
        metadata,
        "spectrum_energy_min_keV",
    )
    energy_max_keV = _required_metadata_number(
        metadata,
        "spectrum_energy_max_keV",
    )
    bin_width_keV = _required_metadata_number(
        metadata,
        "spectrum_bin_width_keV",
    )
    bin_count = _required_metadata_integer(metadata, "spectrum_bin_count")
    axis_values = (
        energy_min_keV,
        energy_max_keV,
        bin_width_keV,
    )
    if not all(np.isfinite(value) for value in axis_values):
        raise RuntimeError("Native Geant4 spectrum-axis provenance must be finite.")
    if (
        not np.isclose(
            energy_min_keV,
            NATIVE_GEANT4_ENERGY_MIN_KEV,
            rtol=0.0,
            atol=1.0e-12,
        )
        or not np.isclose(
            energy_max_keV,
            NATIVE_GEANT4_ENERGY_MAX_KEV,
            rtol=0.0,
            atol=1.0e-12,
        )
        or not np.isclose(
            bin_width_keV,
            NATIVE_GEANT4_BIN_WIDTH_KEV,
            rtol=0.0,
            atol=1.0e-12,
        )
        or bin_count != NATIVE_GEANT4_BIN_COUNT
    ):
        raise RuntimeError(
            "Native Geant4 spectrum axis differs from the fixed production contract."
        )
    if _required_metadata_string(
        metadata,
        "background_spectrum_model_id",
    ) != (NATIVE_GEANT4_BACKGROUND_MODEL_ID):
        raise RuntimeError(
            "Native Geant4 background-spectrum model provenance is invalid."
        )

    secondary_transport_mode = _required_metadata_string(
        metadata,
        "secondary_transport_mode",
    )
    if (
        expected_secondary_transport_mode is not None
        and secondary_transport_mode != expected_secondary_transport_mode
    ):
        raise RuntimeError(
            "Native Geant4 secondary transport disagrees with runtime config: "
            f"expected {expected_secondary_transport_mode}, "
            f"got {secondary_transport_mode or 'missing'}."
        )
    gamma_only = _required_metadata_bool(
        metadata,
        "gamma_only_secondary_transport",
    )
    if gamma_only != (secondary_transport_mode == "gamma_only"):
        raise RuntimeError(
            "Native Geant4 gamma-only provenance is inconsistent with "
            "secondary_transport_mode."
        )
    if _required_metadata_bool(metadata, "theory_tvl_attenuation"):
        raise RuntimeError("Native Geant4 runtime must not use theory-TVL attenuation.")
    if not _required_metadata_bool(metadata, "poisson_background"):
        raise RuntimeError(
            "Native Geant4 runtime must use Poisson background sampling."
        )
    background_cps = _required_metadata_number(metadata, "background_cps")
    if not np.isfinite(background_cps) or background_cps < 0.0:
        raise RuntimeError("Native Geant4 background_cps provenance is invalid.")
    if expected_background_cps is not None and not np.isclose(
        background_cps,
        float(expected_background_cps),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "Native Geant4 background rate disagrees with runtime config: "
            f"expected {expected_background_cps}, got {background_cps}."
        )

    requested_threads = _required_metadata_integer(
        metadata,
        "requested_threads",
    )
    if expected_thread_count is not None and requested_threads != expected_thread_count:
        raise RuntimeError(
            "Native Geant4 thread count disagrees with runtime config: "
            f"expected {expected_thread_count}, got {requested_threads}."
        )
    multithreaded = _required_metadata_bool(
        metadata,
        "multithreaded_run_manager",
    )
    if requested_threads > 1 and not multithreaded:
        raise RuntimeError(
            "Native Geant4 did not use a multithreaded run manager for a "
            f"{requested_threads}-thread runtime request."
        )

    if radioactive_decay_emission:
        expected_key = "expected_physical_primaries"
        expected_semantics = "parent_activity_bq_times_live_time"
        expected_emission_model = "isotropic_parent_radioactive_decay"
        expected_line_normalization = False
        resolved_source_bias_mode = "analog"
        expected_source_bias_weighting = False
    elif source_rate_model == "detector_cps_1m":
        expected_key = "expected_detector_equivalent_primaries"
        expected_semantics = "detector_equivalent_histories"
        expected_emission_model = "detector_equivalent_cone"
        expected_line_normalization = True
        resolved_source_bias_mode = "detector_cone"
        expected_source_bias_weighting = False
    else:
        expected_key = "expected_physical_primaries"
        expected_semantics = "isotropic_physical_histories"
        resolved_source_bias_mode = (
            expected_source_bias_mode
            if expected_source_bias_mode is not None
            else _required_metadata_string(metadata, "source_bias_mode")
        )
        expected_source_bias_weighting = resolved_source_bias_mode == "detector_cone"
        expected_emission_model = (
            "weighted_isotropic" if expected_source_bias_weighting else "isotropic"
        )
        expected_line_normalization = False
    reported_source_bias_mode = _required_metadata_string(
        metadata,
        "source_bias_mode",
    )
    if reported_source_bias_mode != resolved_source_bias_mode:
        raise RuntimeError(
            "Native Geant4 source bias disagrees with runtime config: "
            f"expected {resolved_source_bias_mode}, "
            f"got {reported_source_bias_mode}."
        )
    if _required_metadata_string(metadata, "emission_model") != expected_emission_model:
        raise RuntimeError(
            "Native Geant4 emission-model provenance disagrees with source-rate "
            "semantics."
        )
    expected_cascade_transport = radioactive_decay_emission
    if (
        _required_metadata_bool(metadata, "prompt_decay_cascade_transport")
        != expected_cascade_transport
    ):
        raise RuntimeError(
            "Native Geant4 prompt-cascade provenance disagrees with the "
            "configured primary-emission model."
        )
    expected_summing = (
        "global_time_window_energy_deposit_sum"
        if radioactive_decay_emission
        else "disabled"
    )
    if (
        _required_metadata_string(
            metadata,
            "true_coincidence_summing",
        )
        != expected_summing
    ):
        raise RuntimeError(
            "Native Geant4 coincidence-summing provenance is inconsistent."
        )
    coincidence_window_s = _required_metadata_number(
        metadata,
        "detector_coincidence_window_s",
    )
    if not np.isfinite(coincidence_window_s) or coincidence_window_s <= 0.0:
        raise RuntimeError("Native Geant4 detector coincidence window is invalid.")
    if (
        _required_metadata_bool(metadata, "delayed_decay_pulse_separation")
        != radioactive_decay_emission
    ):
        raise RuntimeError(
            "Native Geant4 delayed-decay pulse provenance is inconsistent."
        )
    expected_decay_time_window = (
        "conditioned_parent_decay_event_no_long_lifetime_cut"
        if radioactive_decay_emission
        else "disabled"
    )
    if (
        _required_metadata_string(
            metadata,
            "radioactive_decay_time_window",
        )
        != expected_decay_time_window
    ):
        raise RuntimeError(
            "Native Geant4 radioactive-decay time-window provenance is inconsistent."
        )
    if (
        _required_metadata_bool(
            metadata,
            "line_intensities_normalized",
        )
        != expected_line_normalization
    ):
        raise RuntimeError(
            "Native Geant4 line-intensity normalization disagrees with "
            "source-rate semantics."
        )
    if (
        _required_metadata_bool(
            metadata,
            "source_bias_weighted_transport",
        )
        != expected_source_bias_weighting
    ):
        raise RuntimeError(
            "Native Geant4 source-bias weighting disagrees with source-rate semantics."
        )
    expected_tally_weighted = history_thinning_enabled or expected_source_bias_weighting
    if (
        _required_metadata_bool(
            metadata,
            "weighted_transport",
        )
        != expected_tally_weighted
    ):
        raise RuntimeError(
            "Native Geant4 aggregate weighting provenance disagrees with "
            "configured transport semantics."
        )
    if (
        _required_metadata_bool(
            metadata,
            "transport_tally_weighted",
        )
        != expected_tally_weighted
    ):
        raise RuntimeError(
            "Native Geant4 tally-weighting provenance disagrees with configured "
            "transport semantics."
        )
    if (
        _required_metadata_bool(
            metadata,
            "history_thinning_enabled",
        )
        != history_thinning_enabled
    ):
        raise RuntimeError(
            "Native Geant4 history-thinning provenance disagrees with configured "
            "transport semantics."
        )
    expected_history_mode = (
        "weighted_thinning" if history_thinning_enabled else "full_unit_weight"
    )
    if (
        _required_metadata_string(metadata, "transport_history_mode")
        != expected_history_mode
    ):
        raise RuntimeError(
            "Native Geant4 transport_history_mode disagrees with configured "
            f"transport semantics: expected {expected_history_mode}."
        )
    expected_variance_semantics = (
        "renewal_total_conditional_multinomial_marks"
        if response_sampling_enabled
        else "compound_poisson_sumw2_includes_counting"
    )
    if _required_metadata_string(
        metadata,
        "spectrum_variance_semantics",
    ) != (expected_variance_semantics):
        raise RuntimeError(
            "Native Geant4 response is missing weighted sumw2 variance semantics."
        )
    expected_dead_time_semantics = (
        "event_time_nonparalyzable_global_stream"
        if response_sampling_enabled
        else "fixed_observed_scale"
    )
    if _required_metadata_string(
        metadata,
        "spectrum_variance_dead_time_propagation",
    ) != (expected_dead_time_semantics):
        raise RuntimeError(
            "Native Geant4 response has invalid dead-time variance provenance."
        )
    dead_time_tau_s = _required_metadata_number(metadata, "dead_time_tau_s")
    dead_time_observed_scale = _required_metadata_number(
        metadata,
        "dead_time_observed_scale",
    )
    dwell_time_s = _required_metadata_number(metadata, "dwell_time_s")
    pre_dead_time_counts = _required_metadata_number(
        metadata,
        "pre_dead_time_total_spectrum_counts",
    )
    pre_dead_time_sumw2 = _required_metadata_number(
        metadata,
        "pre_dead_time_weighted_spectrum_sumw2",
    )
    post_dead_time_sumw2 = _required_metadata_number(
        metadata,
        "weighted_spectrum_sumw2",
    )
    dead_time_values = (
        dead_time_tau_s,
        dead_time_observed_scale,
        dwell_time_s,
        pre_dead_time_counts,
        pre_dead_time_sumw2,
        post_dead_time_sumw2,
    )
    if not all(np.isfinite(value) for value in dead_time_values):
        raise RuntimeError("Native Geant4 dead-time variance provenance is not finite.")
    if (
        dead_time_tau_s < 0.0
        or dead_time_observed_scale <= 0.0
        or dead_time_observed_scale > 1.0
        or dwell_time_s <= 0.0
        or pre_dead_time_counts < 0.0
        or pre_dead_time_sumw2 < 0.0
        or post_dead_time_sumw2 < 0.0
    ):
        raise RuntimeError("Native Geant4 dead-time variance provenance is invalid.")
    if expected_dead_time_tau_s is not None and not np.isclose(
        dead_time_tau_s,
        float(expected_dead_time_tau_s),
        rtol=1.0e-12,
        atol=1.0e-18,
    ):
        raise RuntimeError(
            "Native Geant4 dead-time constant disagrees with runtime config: "
            f"expected {expected_dead_time_tau_s}, got {dead_time_tau_s}."
        )
    if response_sampling_enabled:
        if _required_metadata_string(
            metadata,
            "dead_time_scale_semantics",
        ) != ("realized_global_acceptance_fraction"):
            raise RuntimeError(
                "Native event-time dead-time acceptance provenance is invalid."
            )
        expected_realized_scale = (
            post_dead_time_sumw2 / pre_dead_time_counts
            if pre_dead_time_counts > 0.0
            else 1.0
        )
        if (
            not np.isclose(
                dead_time_observed_scale,
                expected_realized_scale,
                rtol=0.0,
                atol=1.0e-15,
            )
            or not np.isclose(
                pre_dead_time_sumw2,
                pre_dead_time_counts,
                rtol=0.0,
                atol=1.0e-9,
            )
            or post_dead_time_sumw2 > pre_dead_time_counts + 1.0e-9
            or not np.isclose(
                post_dead_time_sumw2,
                round(post_dead_time_sumw2),
                rtol=0.0,
                atol=1.0e-9,
            )
        ):
            raise RuntimeError(
                "Native marked-Poisson dead-time provenance is inconsistent "
                "with unit histories and stochastic thinning."
            )
    else:
        expected_dead_time_scale = 1.0 / (
            1.0 + pre_dead_time_counts * dead_time_tau_s / dwell_time_s
        )
        if not np.isclose(
            dead_time_observed_scale,
            expected_dead_time_scale,
            rtol=1.0e-12,
            atol=1.0e-15,
        ):
            raise RuntimeError(
                "Native Geant4 dead-time scale is inconsistent with its "
                "count-rate provenance."
            )
        if not np.isclose(
            post_dead_time_sumw2,
            pre_dead_time_sumw2 * dead_time_observed_scale**2,
            rtol=1.0e-12,
            atol=1.0e-9,
        ):
            raise RuntimeError(
                "Native Geant4 post-dead-time sumw2 is inconsistent with its raw "
                "variance provenance."
            )
    expected_primaries = _required_metadata_number(metadata, expected_key)
    expected_unthinned_primaries = _required_metadata_number(
        metadata,
        "expected_unthinned_primaries",
    )
    sampled_primaries = _required_metadata_number(
        metadata,
        "expected_sampled_primaries",
    )
    if (
        not np.isfinite(expected_primaries)
        or not np.isfinite(expected_unthinned_primaries)
        or not np.isfinite(sampled_primaries)
    ):
        raise RuntimeError("Native Geant4 expected-primary provenance is not finite.")
    if not np.isclose(
        expected_unthinned_primaries,
        expected_primaries,
        rtol=1.0e-12,
        atol=1.0e-9,
    ):
        raise RuntimeError(
            "Native Geant4 generic and source-model-specific unthinned primary "
            "expectations disagree."
        )
    resolved_fraction, expected_resolution = resolve_primary_sampling_fraction(
        configured_fraction,
        expected_target,
        expected_unthinned_primaries,
    )
    if not np.isclose(
        observed_fraction,
        resolved_fraction,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "Native Geant4 resolved primary sampling fraction disagrees with "
            f"the configured budget: expected {resolved_fraction}, "
            f"got {observed_fraction}."
        )
    budget_keys_present = any(
        key in metadata
        for key in (
            "requested_primary_sampling_fraction",
            "target_sampled_primaries",
            "primary_sampling_budget_enabled",
            "primary_sampling_fraction_resolution",
        )
    )
    if expected_target is not None or budget_keys_present:
        reported_requested_fraction = _required_metadata_number(
            metadata,
            "requested_primary_sampling_fraction",
        )
        reported_target = _required_metadata_integer(
            metadata,
            "target_sampled_primaries",
        )
        if reported_target < 0:
            raise RuntimeError(
                "Native Geant4 target sampled-primary provenance is invalid."
            )
        if not np.isclose(
            reported_requested_fraction,
            configured_fraction,
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise RuntimeError(
                "Native Geant4 requested sampling fraction disagrees with runtime "
                "configuration."
            )
        expected_reported_target = int(expected_target or 0)
        if reported_target != expected_reported_target:
            raise RuntimeError(
                "Native Geant4 target sampled-primary provenance disagrees with "
                "runtime configuration."
            )
        if _required_metadata_bool(
            metadata,
            "primary_sampling_budget_enabled",
        ) != (expected_target is not None):
            raise RuntimeError(
                "Native Geant4 primary-budget enable provenance disagrees with "
                "runtime configuration."
            )
        if _required_metadata_string(
            metadata,
            "primary_sampling_fraction_resolution",
        ) != (expected_resolution):
            raise RuntimeError(
                "Native Geant4 sampling-fraction resolution provenance disagrees "
                f"with runtime configuration: expected {expected_resolution}."
            )
    expected_sampled = expected_primaries * observed_fraction
    if not np.isclose(
        sampled_primaries,
        expected_sampled,
        rtol=1.0e-12,
        atol=1.0e-9,
    ):
        raise RuntimeError(
            "Native Geant4 sampled-primary expectation does not match the "
            "configured sampling fraction."
        )
    semantics = _required_metadata_string(
        metadata,
        "expected_primary_semantics",
    )
    if semantics != expected_semantics:
        raise RuntimeError(
            "Native Geant4 expected-primary semantics disagree with source-rate "
            f"model: expected {expected_semantics}, got {semantics or 'missing'}."
        )
    if source_rate_model == "detector_cps_1m" and metadata.get(
        "expected_physical_primaries"
    ) not in (None, ""):
        raise RuntimeError(
            "Detector-equivalent history metadata must not be labelled as "
            "physical isotropic primaries."
        )


def _mean_calibration_metadata_vector(
    metadata: dict[str, Any],
    key: str,
) -> NDArray[np.float64]:
    """Parse one comma-delimited native calibration vector."""
    value = metadata.get(key)
    if not isinstance(value, str):
        raise RuntimeError(f"Native Geant4 mean calibration is missing vector {key}.")
    try:
        vector = np.asarray(
            [float(token) for token in value.split(",")],
            dtype=np.float64,
        )
    except ValueError as exc:
        raise RuntimeError(
            f"Native Geant4 mean-calibration vector {key} is not numeric."
        ) from exc
    if (
        vector.shape != (NATIVE_GEANT4_BIN_COUNT,)
        or np.any(~np.isfinite(vector))
        or np.any(vector < 0.0)
    ):
        raise RuntimeError(f"Native Geant4 mean-calibration vector {key} is invalid.")
    return vector


def _validate_native_mean_calibration_response(
    spectrum_counts: object,
    metadata: dict[str, Any],
) -> None:
    """Validate a fixed-quota mean spectrum against its history statistics."""
    _validate_native_weighted_response(spectrum_counts, metadata)
    spectrum = np.asarray(spectrum_counts, dtype=np.float64)
    if spectrum.shape != (NATIVE_GEANT4_BIN_COUNT,):
        raise RuntimeError(
            "Native Geant4 mean-calibration spectrum has the wrong shape."
        )
    variance = metadata.get("spectrum_count_variance")
    try:
        calibration = parse_mean_calibration_metadata(
            metadata,
            bin_count=NATIVE_GEANT4_BIN_COUNT,
        )
        calibration.validate_native_arrays(spectrum, variance)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Native Geant4 mean-calibration spectrum disagrees with its "
            "original-history sufficient statistics."
        ) from exc


def validate_mean_calibration_transport_metadata(
    metadata: dict[str, Any],
    *,
    expected_histories_per_source_line: int,
    expected_angle_strata_mu: int,
    expected_angle_strata_phi: int,
    expected_forced_collision: bool,
    expected_source_rate_model: str = "detector_cps_1m",
    expected_thread_count: int | None = None,
    expected_physics_profile: str | None = None,
    expected_detector_scoring_mode: str = "incident_gamma_energy",
    expected_secondary_transport_mode: str = "full_transport",
    expected_source_bias_mode: str = "detector_cone",
    expected_surface_source_contract_sha256: str | None = None,
    expected_scene_hash: str | None = None,
) -> None:
    """Validate fixed-quota weighted-mean calibration provenance."""
    integer_inputs = {
        "expected_histories_per_source_line": (expected_histories_per_source_line),
        "expected_angle_strata_mu": expected_angle_strata_mu,
        "expected_angle_strata_phi": expected_angle_strata_phi,
    }
    for name, value in integer_inputs.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive JSON integer.")
    if not isinstance(expected_forced_collision, bool):
        raise ValueError("expected_forced_collision must be a JSON boolean.")
    if expected_thread_count is not None and (
        isinstance(expected_thread_count, bool)
        or not isinstance(expected_thread_count, int)
        or expected_thread_count <= 0
    ):
        raise ValueError("expected_thread_count must be a positive JSON integer.")
    stratum_count = expected_angle_strata_mu * expected_angle_strata_phi
    if expected_histories_per_source_line % stratum_count != 0:
        raise ValueError(
            "expected_histories_per_source_line must be divisible by the "
            "configured angular stratum count."
        )
    histories_per_stratum = expected_histories_per_source_line // stratum_count
    if histories_per_stratum < 2:
        raise ValueError(
            "Mean calibration requires at least two histories per stratum."
        )
    if expected_source_rate_model != "detector_cps_1m":
        raise ValueError("Mean calibration requires source_rate_model=detector_cps_1m.")
    if (
        expected_detector_scoring_mode
        not in {"incident_gamma_energy", "full_transport"}
        or expected_secondary_transport_mode != "full_transport"
        or expected_source_bias_mode != "detector_cone"
    ):
        raise ValueError(
            "Fixed source-line sampling requires implemented detector scoring, "
            "full secondary transport, and detector-cone source sampling."
        )
    if expected_detector_scoring_mode == "full_transport" and (
        expected_forced_collision
    ):
        raise ValueError(
            "Full-detector fixed source-line validation forbids forced collision."
        )

    validate_geant4_physics_metadata(metadata)

    exact_strings = {
        "backend": "geant4",
        "engine_mode": "external",
        "source_rate_model": expected_source_rate_model,
        "primary_emission_model": "independent_gamma_lines",
        "true_coincidence_summing": "disabled",
        "radioactive_decay_time_window": "disabled",
        "intensity_cps_1m_definition": ("pre_dead_time_detector_pulse_rate_at_1m"),
        "source_position_semantics": "air_side_native_emission_xyz",
        "source_anchor_semantics": ("exact_surface_chart_uv_evaluation_truth"),
        "detector_scoring_mode": expected_detector_scoring_mode,
        "detector_response_sampling_mode": "disabled",
        "secondary_transport_mode": expected_secondary_transport_mode,
        "source_bias_mode": expected_source_bias_mode,
        "emission_model": "detector_equivalent_cone",
        "expected_primary_semantics": "detector_equivalent_histories",
        "primary_schedule_mode": ("fixed_source_line_stratified_mean_calibration"),
        "transport_history_mode": ("fixed_source_line_stratified_weighted_mean"),
        "mean_calibration_history_weight_semantics": (
            "expected_source_line_mean_divided_by_fixed_quota"
        ),
        "mean_calibration_covariance_semantics": (
            MEAN_CALIBRATION_FORCED_COLLISION_COVARIANCE_SEMANTICS
            if expected_forced_collision
            else MEAN_CALIBRATION_ANALOG_COVARIANCE_SEMANTICS
        ),
        "spectrum_variance_semantics": (
            "stratified_fixed_quota_sample_mean_covariance"
        ),
        "spectrum_variance_dead_time_propagation": (
            "disabled_zero_dead_time_mean_calibration"
        ),
        "dead_time_scale_semantics": ("identity_zero_dead_time_mean_calibration"),
        "validation_entry_spectrum_space": (
            "pre_dead_time_raw_incident_gamma_weighted_mean"
            if expected_detector_scoring_mode == "incident_gamma_energy"
            else "pre_dead_time_raw_energy_deposition_weighted_mean"
        ),
        "validation_observed_entry_spectrum_space": (
            "pre_dead_time_sampled_detector_response_weighted_mean"
            if expected_detector_scoring_mode == "incident_gamma_energy"
            else "pre_dead_time_smeared_energy_deposition_weighted_mean"
        ),
        "validation_entry_spectrum_grouping": (
            "source_token_initial_gamma_line_entry_class"
        ),
        "shield_pose_contract_id": SHIELD_POSE_CONTRACT_ID,
        "shield_pose_contract_sha256": SHIELD_POSE_CONTRACT_SHA256,
    }
    if expected_physics_profile is not None:
        exact_strings["physics_profile"] = expected_physics_profile
    for key, expected in exact_strings.items():
        if _required_metadata_string(metadata, key) != expected:
            raise RuntimeError(
                "Native Geant4 mean-calibration provenance disagrees with "
                f"the configured contract for {key}."
            )

    exact_booleans = {
        "prompt_decay_cascade_transport": False,
        "delayed_decay_pulse_separation": False,
        "mean_calibration_enabled": True,
        "mean_calibration_forced_collision": expected_forced_collision,
        "detector_response_applied_in_native": (
            expected_detector_scoring_mode == "full_transport"
        ),
        "gamma_only_secondary_transport": False,
        "theory_tvl_attenuation": False,
        "poisson_background": False,
        "validation_entry_class_spectra": True,
        "source_bias_weighted_transport": False,
        "line_intensities_normalized": True,
        "weighted_transport": True,
        "transport_tally_weighted": True,
        "history_thinning_enabled": False,
        "primary_sampling_budget_enabled": False,
        "all_sources_surface_bound": True,
    }
    for key, expected in exact_booleans.items():
        if _required_metadata_bool(metadata, key) != expected:
            raise RuntimeError(
                "Native Geant4 mean-calibration provenance disagrees with "
                f"the configured contract for {key}."
            )
    coincidence_window_s = _required_metadata_number(
        metadata,
        "detector_coincidence_window_s",
    )
    if not np.isfinite(coincidence_window_s) or coincidence_window_s <= 0.0:
        raise RuntimeError(
            "Native Geant4 mean-calibration detector coincidence window is invalid."
        )

    exact_integers = {
        "mean_calibration_histories_per_source_line": (
            expected_histories_per_source_line
        ),
        "mean_calibration_angle_strata_mu": expected_angle_strata_mu,
        "mean_calibration_angle_strata_phi": expected_angle_strata_phi,
        "mean_calibration_angle_stratum_count": stratum_count,
        "target_sampled_primaries": 0,
    }
    for key, expected in exact_integers.items():
        if _required_metadata_integer(metadata, key) != expected:
            raise RuntimeError(
                "Native Geant4 mean-calibration provenance disagrees with "
                f"the configured contract for {key}."
            )
    force_leaf_count = _required_metadata_integer(
        metadata,
        "mean_calibration_force_collision_leaf_count",
    )
    force_split_count = _required_metadata_integer(
        metadata,
        "mean_calibration_force_collision_split_count",
    )
    force_weight_errors = (
        _required_metadata_number(
            metadata,
            "mean_calibration_force_collision_max_abs_weight_error",
        ),
        _required_metadata_number(
            metadata,
            "mean_calibration_force_collision_max_rel_weight_error",
        ),
    )
    if expected_forced_collision:
        absolute_error, relative_error = force_weight_errors
        if (
            force_leaf_count <= 0
            or force_split_count <= 0
            or any(error < 0.0 for error in force_weight_errors)
            or (absolute_error > 1.0e-9 and relative_error > 1.0e-9)
        ):
            raise RuntimeError(
                "Native Geant4 forced-collision diagnostics are invalid."
            )
    elif (
        force_leaf_count != 0
        or force_split_count != 0
        or any(error != 0.0 for error in force_weight_errors)
    ):
        raise RuntimeError(
            "Analog mean calibration contains forced-collision diagnostics."
        )

    if not np.isclose(
        _required_metadata_number(metadata, "primary_sampling_fraction"),
        1.0,
        rtol=0.0,
        atol=1.0e-15,
    ) or not np.isclose(
        _required_metadata_number(metadata, "primary_history_weight"),
        1.0,
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise RuntimeError(
            "Native Geant4 mean calibration must disable history thinning."
        )
    if (
        not np.isclose(
            _required_metadata_number(
                metadata,
                "requested_primary_sampling_fraction",
            ),
            1.0,
            rtol=0.0,
            atol=1.0e-15,
        )
        or _required_metadata_string(
            metadata,
            "primary_sampling_fraction_resolution",
        )
        != "fixed_source_line_mean_calibration"
    ):
        raise RuntimeError(
            "Native Geant4 mean-calibration sampling resolution is invalid."
        )

    if _required_metadata_string(
        metadata,
        "surface_emission_policy_sha256",
    ) != surface_emission_policy_sha256() or not np.isclose(
        _required_metadata_number(metadata, "surface_emission_epsilon_m"),
        SURFACE_EMISSION_EPSILON_M,
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise RuntimeError(
            "Native Geant4 mean calibration violates the shared surface "
            "emission contract."
        )
    for key, expected_hash in (
        (
            "surface_source_contract_sha256",
            expected_surface_source_contract_sha256,
        ),
        ("scene_hash", expected_scene_hash),
    ):
        actual_hash = metadata.get(key)
        if (
            not isinstance(actual_hash, str)
            or len(actual_hash) != 64
            or any(character not in "0123456789abcdef" for character in actual_hash)
            or (expected_hash is not None and actual_hash != str(expected_hash))
        ):
            raise RuntimeError(f"Native Geant4 {key} is missing, invalid, or stale.")

    axis_values = {
        "spectrum_energy_min_keV": NATIVE_GEANT4_ENERGY_MIN_KEV,
        "spectrum_energy_max_keV": NATIVE_GEANT4_ENERGY_MAX_KEV,
        "spectrum_bin_width_keV": NATIVE_GEANT4_BIN_WIDTH_KEV,
    }
    for key, expected in axis_values.items():
        if not np.isclose(
            _required_metadata_number(metadata, key),
            expected,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(
                "Native Geant4 mean-calibration spectrum axis is invalid."
            )
    if (
        _required_metadata_integer(metadata, "spectrum_bin_count")
        != NATIVE_GEANT4_BIN_COUNT
        or _required_metadata_string(
            metadata,
            "background_spectrum_model_id",
        )
        != NATIVE_GEANT4_BACKGROUND_MODEL_ID
    ):
        raise RuntimeError(
            "Native Geant4 mean-calibration spectrum contract is invalid."
        )

    requested_threads = _required_metadata_integer(
        metadata,
        "requested_threads",
    )
    if requested_threads <= 0 or (
        expected_thread_count is not None and requested_threads != expected_thread_count
    ):
        raise RuntimeError("Native Geant4 mean-calibration thread count is invalid.")
    if requested_threads > 1 and not _required_metadata_bool(
        metadata,
        "multithreaded_run_manager",
    ):
        raise RuntimeError(
            "Native Geant4 mean calibration did not use the requested MT run manager."
        )

    zero_numbers = (
        "background_cps",
        "dead_time_tau_s",
    )
    if any(
        not np.isclose(
            _required_metadata_number(metadata, key),
            0.0,
            rtol=0.0,
            atol=1.0e-18,
        )
        for key in zero_numbers
    ) or not np.isclose(
        _required_metadata_number(metadata, "dead_time_observed_scale"),
        1.0,
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise RuntimeError(
            "Native Geant4 mean calibration must exclude background and "
            "dead-time sampling."
        )
    dwell_time_s = _required_metadata_number(metadata, "dwell_time_s")
    total_counts = _required_metadata_number(
        metadata,
        "total_spectrum_counts",
    )
    pre_dead_time_counts = _required_metadata_number(
        metadata,
        "pre_dead_time_total_spectrum_counts",
    )
    variance_total = _required_metadata_number(
        metadata,
        "weighted_spectrum_sumw2",
    )
    pre_dead_time_variance = _required_metadata_number(
        metadata,
        "pre_dead_time_weighted_spectrum_sumw2",
    )
    if (
        dwell_time_s <= 0.0
        or total_counts < 0.0
        or variance_total < 0.0
        or not np.isclose(
            total_counts,
            pre_dead_time_counts,
            rtol=1.0e-9,
            atol=1.0e-6,
        )
        or not np.isclose(
            variance_total,
            pre_dead_time_variance,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    ):
        raise RuntimeError(
            "Native Geant4 mean-calibration zero-dead-time provenance is inconsistent."
        )

    batch_count = _required_metadata_integer(
        metadata,
        "primary_history_batch_count",
    )
    if batch_count <= 0 or batch_count % stratum_count != 0:
        raise RuntimeError("Native Geant4 mean-calibration batch count is invalid.")
    line_count = batch_count // stratum_count
    expected_num_primaries = line_count * expected_histories_per_source_line
    if (
        _required_metadata_integer(
            metadata,
            "num_primaries",
        )
        != expected_num_primaries
    ):
        raise RuntimeError(
            "Native Geant4 mean calibration did not run the fixed line quota."
        )
    expected_beam_on_calls = (expected_num_primaries + 999_999_999) // 1_000_000_000
    if (
        _required_metadata_integer(
            metadata,
            "primary_beam_on_calls",
        )
        != expected_beam_on_calls
    ):
        raise RuntimeError(
            "Native Geant4 mean calibration did not use the batched BeamOn schedule."
        )

    try:
        calibration = parse_mean_calibration_metadata(
            metadata,
            bin_count=NATIVE_GEANT4_BIN_COUNT,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Native Geant4 mean-calibration sufficient statistics are "
            f"malformed or semantically mixed: {exc}"
        ) from exc
    line_strata: dict[str, set[int]] = {}
    expected_total = 0.0
    sampled_total = 0
    observed_weights: list[float] = []
    for batch_index in range(batch_count):
        prefix = f"mean_calibration_batch_{batch_index}_"
        source_token = _required_metadata_string(
            metadata,
            prefix + "source_token",
        )
        line_token = _required_metadata_string(
            metadata,
            prefix + "line_token",
        )
        sampled = _required_metadata_integer(
            metadata,
            prefix + "sampled_histories",
        )
        stratum_index = _required_metadata_integer(
            metadata,
            prefix + "angle_stratum_index",
        )
        expected = _required_metadata_number(
            metadata,
            prefix + "expected_unthinned_histories",
        )
        weight = _required_metadata_number(
            metadata,
            prefix + "history_weight",
        )
        if (
            not source_token
            or not line_token.startswith(source_token + "_")
            or sampled != histories_per_stratum
            or stratum_index < 0
            or stratum_index >= stratum_count
            or expected <= 0.0
            or weight <= 0.0
            or not np.isclose(
                expected,
                weight * sampled,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        ):
            raise RuntimeError(
                "Native Geant4 mean-calibration batch provenance is invalid."
            )
        strata = line_strata.setdefault(line_token, set())
        if stratum_index in strata:
            raise RuntimeError(
                "Native Geant4 mean calibration duplicated an angular stratum."
            )
        strata.add(stratum_index)
        expected_total += expected
        sampled_total += sampled
        observed_weights.append(weight)
    reconstructed_variance = np.diag(calibration.raw_covariance())
    reconstructed_class_variance = {
        line_token + "_" + entry_class: variance
        for line_token, classes in (
            calibration.entry_class_line_bin_variances().items()
        )
        for entry_class, variance in classes.items()
    }

    expected_strata = set(range(stratum_count))
    if (
        len(line_strata) != line_count
        or any(strata != expected_strata for strata in line_strata.values())
        or sampled_total != expected_num_primaries
    ):
        raise RuntimeError(
            "Native Geant4 mean calibration did not cover every angular "
            "stratum exactly once per source line."
        )
    expected_detector_primaries = _required_metadata_number(
        metadata,
        "expected_detector_equivalent_primaries",
    )
    if (
        expected_total <= 0.0
        or not np.isclose(
            expected_total,
            expected_detector_primaries,
            rtol=1.0e-12,
            atol=1.0e-9,
        )
        or not np.isclose(
            _required_metadata_number(
                metadata,
                "expected_unthinned_primaries",
            ),
            expected_total,
            rtol=1.0e-12,
            atol=1.0e-9,
        )
        or not np.isclose(
            _required_metadata_number(
                metadata,
                "expected_sampled_primaries",
            ),
            float(expected_num_primaries),
            rtol=0.0,
            atol=1.0e-9,
        )
    ):
        raise RuntimeError(
            "Native Geant4 mean-calibration primary expectations are invalid."
        )
    if metadata.get("expected_physical_primaries") not in (None, ""):
        raise RuntimeError(
            "Detector-equivalent mean calibration must not claim physical "
            "isotropic primaries."
        )

    weight_min = min(observed_weights)
    weight_max = max(observed_weights)
    for key, expected in (
        ("mean_calibration_history_weight_min", weight_min),
        ("mean_calibration_history_weight_max", weight_max),
    ):
        if not np.isclose(
            _required_metadata_number(metadata, key),
            expected,
            rtol=1.0e-12,
            atol=1.0e-15,
        ):
            raise RuntimeError(
                "Native Geant4 mean-calibration history-weight range is "
                "inconsistent with its batches."
            )

    reported_variance = metadata.get("spectrum_count_variance")
    try:
        reported_variance_array = np.asarray(
            reported_variance,
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Native Geant4 mean calibration is missing spectrum variance."
        ) from exc
    if (
        reported_variance_array.shape != reconstructed_variance.shape
        or np.any(~np.isfinite(reported_variance_array))
        or np.any(reported_variance_array < 0.0)
        or not np.allclose(
            reported_variance_array,
            reconstructed_variance,
            rtol=1.0e-9,
            atol=1.0e-12,
        )
    ):
        raise RuntimeError(
            "Native Geant4 mean-calibration variance disagrees with its "
            "per-history sufficient statistics."
        )
    for class_key, expected_variance in reconstructed_class_variance.items():
        reported = _mean_calibration_metadata_vector(
            metadata,
            "validation_only_entry_variance_" + class_key,
        )
        if not np.allclose(
            reported,
            expected_variance,
            rtol=1.0e-9,
            atol=1.0e-12,
        ):
            raise RuntimeError(
                "Native Geant4 entry-class variance disagrees with its "
                "per-history sufficient statistics."
            )

    background = _mean_calibration_metadata_vector(
        metadata,
        "validation_only_background_incident_spectrum",
    )
    if np.any(background != 0.0):
        raise RuntimeError(
            "Native Geant4 mean calibration must exclude background counts."
        )


def validate_full_history_transport_metadata(
    metadata: dict[str, Any],
    *,
    expected_source_rate_model: str | None = None,
    expected_thread_count: int | None = None,
    expected_physics_profile: str | None = None,
    expected_detector_scoring_mode: str | None = None,
    expected_secondary_transport_mode: str | None = None,
    expected_source_bias_mode: str | None = None,
    expected_background_cps: float | None = None,
    expected_dead_time_tau_s: float | None = None,
    expected_detector_response_sampling: bool = False,
    expected_surface_source_contract_sha256: str | None = None,
    expected_scene_hash: str | None = None,
) -> None:
    """Validate standard full-history native transport provenance."""
    validate_transport_metadata(
        metadata,
        expected_primary_sampling_fraction=1.0,
        accelerated_weighted_transport_enable=False,
        expected_source_rate_model=expected_source_rate_model,
        expected_thread_count=expected_thread_count,
        expected_physics_profile=expected_physics_profile,
        expected_detector_scoring_mode=expected_detector_scoring_mode,
        expected_secondary_transport_mode=expected_secondary_transport_mode,
        expected_source_bias_mode=expected_source_bias_mode,
        expected_background_cps=expected_background_cps,
        expected_dead_time_tau_s=expected_dead_time_tau_s,
        expected_detector_response_sampling=(expected_detector_response_sampling),
        expected_surface_source_contract_sha256=(
            expected_surface_source_contract_sha256
        ),
        expected_scene_hash=expected_scene_hash,
    )


def _json_boolean(
    payload: Mapping[str, Any],
    key: str,
    *,
    default: bool | None,
) -> bool | None:
    """Return an exact JSON boolean or optional null without truthy coercion."""
    value = payload.get(key, default)
    if value is None and default is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a JSON boolean.")
    return value


def _json_integer(
    payload: Mapping[str, Any],
    key: str,
    *,
    default: int,
    minimum: int | None = None,
) -> int:
    """Return an exact JSON integer satisfying an optional lower bound."""
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be a JSON integer.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{key} must be at least {minimum}.")
    return value


def _optional_json_integer(
    payload: Mapping[str, Any],
    key: str,
    *,
    minimum: int | None = None,
) -> int | None:
    """Return an optional exact JSON integer with an optional lower bound."""
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be a JSON integer or null.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{key} must be at least {minimum}.")
    return value


def _json_number(
    payload: Mapping[str, Any],
    key: str,
    *,
    default: float,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    """Return a finite JSON number satisfying its physical domain."""
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{key} must be a JSON number.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{key} must be finite.")
    if strictly_positive and parsed <= 0.0:
        raise ValueError(f"{key} must be positive.")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{key} must be at least {minimum}.")
    return parsed


def _json_string(
    payload: Mapping[str, Any],
    key: str,
    *,
    default: str,
    choices: frozenset[str] | None = None,
) -> str:
    """Return an exact nonempty JSON string from an optional enum."""
    value = payload.get(key, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a nonempty JSON string.")
    if choices is not None and value not in choices:
        raise ValueError(f"{key} must be one of {sorted(choices)}, got {value!r}.")
    return value


def _json_vector3(
    payload: Mapping[str, Any],
    key: str,
    *,
    default: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return a positive finite three-number JSON array."""
    value = payload.get(key, default)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{key} must be a three-element JSON array.")
    parsed: list[float] = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, Real):
            raise ValueError(f"{key} must contain only JSON numbers.")
        numeric = float(component)
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(f"{key} entries must be finite and positive.")
        parsed.append(numeric)
    return parsed[0], parsed[1], parsed[2]


def _nonempty_string(value: object, *, field_name: str) -> str:
    """Return an exact nonempty string."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a nonempty string.")
    return value


@dataclass(frozen=True)
class Geant4AppConfig:
    """Collect sidecar configuration relevant to the Geant4 app."""

    use_mock_stage: bool = True
    headless: bool = True
    renderer: str = "RayTracedLighting"
    usd_path: str | None = None
    detector_height_m: float = 0.5
    robot_ground_z_m: float = 0.0
    obstacle_height_m: float = 2.0
    author_obstacle_prims: bool | None = None
    author_room_boundary_prims: bool | None = None
    fe_shield_size_xyz: tuple[float, float, float] = (0.25, 0.08, 0.25)
    pb_shield_size_xyz: tuple[float, float, float] = (0.25, 0.08, 0.25)
    stage_material_rules: tuple[StageMaterialRule, ...] = field(default_factory=tuple)
    engine_mode: str = "external"
    physics_profile: str = "balanced"
    thread_count: int = 1
    random_seed_base: int = 123
    dead_time_tau_s: float = 5.813e-9
    scatter_gain: float = 0.0
    executable_path: str | None = "build/geant4_sidecar"
    executable_args: tuple[str, ...] = field(default_factory=tuple)
    timeout_s: float = 120.0
    persistent_process: bool = False
    source_rate_model: str = "detector_cps_1m"
    primary_emission_model: str = "independent_gamma_lines"
    source_bias_mode: str = "detector_cone"
    source_bias_cone_half_angle_deg: float = 0.0
    source_bias_isotropic_fraction: float = 1.0
    detector_scoring_mode: str = "full_transport"
    secondary_transport_mode: str = "full_transport"
    primary_sampling_fraction: float = 1.0
    target_sampled_primaries: int | None = None
    mean_calibration_histories_per_source_line: int | None = None
    mean_calibration_angle_strata_mu: int = 1
    mean_calibration_angle_strata_phi: int = 1
    mean_calibration_forced_collision: bool = False
    accelerated_weighted_transport_enable: bool = False
    background_cps: float = 0.0
    sample_detector_response: bool = False
    validation_entry_class_spectra: bool = False
    detector_model: ExportedDetectorModel = field(default_factory=ExportedDetectorModel)
    shield_thickness: ShieldThicknessConfig = field(
        default_factory=resolve_shield_thickness_config
    )
    absorbing_transport_groups: tuple[str, ...] = field(default_factory=tuple)
    absorbing_path_prefixes: tuple[str, ...] = field(default_factory=tuple)
    radiation_visualization: RadiationVisualizationConfig = field(
        default_factory=RadiationVisualizationConfig
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Geant4AppConfig":
        """Normalize a JSON config payload into a strongly typed object."""
        if data is None:
            payload: dict[str, Any] = {}
        elif isinstance(data, Mapping):
            payload = dict(data)
        else:
            raise TypeError("Geant4 app config must be a mapping or None.")
        stage_material_rules_payload = payload.get("stage_material_rules", ())
        if not isinstance(stage_material_rules_payload, (list, tuple)):
            raise ValueError("stage_material_rules must be a list of objects.")
        stage_material_rules: list[StageMaterialRule] = []
        seen_material_prefixes: set[str] = set()
        for index, entry in enumerate(stage_material_rules_payload):
            if not isinstance(entry, Mapping) or set(entry) != {
                "path_prefix",
                "material",
            }:
                raise ValueError(
                    f"stage_material_rules[{index}] must contain exactly "
                    "path_prefix and material."
                )
            path_prefix = _nonempty_string(
                entry["path_prefix"],
                field_name=f"stage_material_rules[{index}].path_prefix",
            )
            material = _nonempty_string(
                entry["material"],
                field_name=f"stage_material_rules[{index}].material",
            )
            if path_prefix in seen_material_prefixes:
                raise ValueError("stage_material_rules path prefixes must be unique.")
            seen_material_prefixes.add(path_prefix)
            stage_material_rules.append(
                StageMaterialRule(
                    path_prefix=path_prefix,
                    material=material,
                )
            )
        detector_payload = payload.get("detector_model", {})
        if not isinstance(detector_payload, Mapping):
            raise ValueError("detector_model must be a JSON object.")
        detector_keys = {
            "crystal_radius_m",
            "crystal_length_m",
            "housing_thickness_m",
            "crystal_shape",
            "crystal_material",
            "housing_material",
        }
        unknown_detector_keys = sorted(set(detector_payload) - detector_keys)
        if unknown_detector_keys:
            raise ValueError(
                "Unsupported detector_model settings: "
                + ", ".join(str(key) for key in unknown_detector_keys)
            )
        visualization_payload = payload.get("radiation_visualization", {})
        if not isinstance(visualization_payload, dict):
            raise ValueError("radiation_visualization must be a JSON object.")
        executable_args = payload.get("executable_args", ())
        if not isinstance(executable_args, (list, tuple)):
            raise ValueError("executable_args must be a list of strings.")
        if any(not isinstance(value, str) or not value for value in executable_args):
            raise ValueError("executable_args entries must be nonempty strings.")
        normalized_executable_args = validate_geant4_executable_args(
            tuple(executable_args)
        )
        absorbing_transport_groups = payload.get("absorbing_transport_groups", ())
        if not isinstance(absorbing_transport_groups, (list, tuple)):
            raise ValueError("absorbing_transport_groups must be a list of strings.")
        if any(
            not isinstance(value, str) or not value
            for value in absorbing_transport_groups
        ):
            raise ValueError(
                "absorbing_transport_groups entries must be nonempty strings."
            )
        absorbing_path_prefixes = payload.get("absorbing_path_prefixes", ())
        if not isinstance(absorbing_path_prefixes, (list, tuple)):
            raise ValueError("absorbing_path_prefixes must be a list of strings.")
        if any(
            not isinstance(value, str) or not value for value in absorbing_path_prefixes
        ):
            raise ValueError(
                "absorbing_path_prefixes entries must be nonempty strings."
            )
        accelerated_weighted_transport_enable = payload.get(
            "accelerated_weighted_transport_enable",
            False,
        )
        if not isinstance(accelerated_weighted_transport_enable, bool):
            raise ValueError(
                "accelerated_weighted_transport_enable must be a JSON boolean."
            )
        sample_detector_response = payload.get(
            "sample_detector_response",
            False,
        )
        validation_entry_class_spectra = payload.get(
            "validation_entry_class_spectra",
            False,
        )
        if not isinstance(sample_detector_response, bool):
            raise ValueError("sample_detector_response must be a JSON boolean.")
        if not isinstance(validation_entry_class_spectra, bool):
            raise ValueError("validation_entry_class_spectra must be a JSON boolean.")
        use_mock_stage = _json_boolean(
            payload,
            "use_mock_stage",
            default=True,
        )
        headless = _json_boolean(
            payload,
            "headless",
            default=True,
        )
        author_obstacle_prims = _json_boolean(
            payload,
            "author_obstacle_prims",
            default=None,
        )
        author_room_boundary_prims = _json_boolean(
            payload,
            "author_room_boundary_prims",
            default=None,
        )
        persistent_process = _json_boolean(
            payload,
            "persistent_process",
            default=False,
        )
        mean_calibration_histories_per_source_line = _optional_json_integer(
            payload,
            "mean_calibration_histories_per_source_line",
            minimum=1,
        )
        mean_calibration_angle_strata_mu = _json_integer(
            payload,
            "mean_calibration_angle_strata_mu",
            default=1,
            minimum=1,
        )
        mean_calibration_angle_strata_phi = _json_integer(
            payload,
            "mean_calibration_angle_strata_phi",
            default=1,
            minimum=1,
        )
        mean_calibration_forced_collision = _json_boolean(
            payload,
            "mean_calibration_forced_collision",
            default=False,
        )
        mean_calibration_enabled = (
            mean_calibration_histories_per_source_line is not None
        )
        stratum_count = (
            mean_calibration_angle_strata_mu * mean_calibration_angle_strata_phi
        )
        if (
            mean_calibration_enabled
            and mean_calibration_histories_per_source_line % stratum_count != 0
        ):
            raise ValueError(
                "mean_calibration_histories_per_source_line must be divisible "
                "by the product of the mu and phi stratum counts."
            )
        if (
            mean_calibration_enabled
            and mean_calibration_histories_per_source_line // stratum_count < 2
        ):
            raise ValueError(
                "Mean calibration requires at least two histories per "
                "mu/phi stratum for covariance estimation."
            )
        if not mean_calibration_enabled and (
            mean_calibration_angle_strata_mu != 1
            or mean_calibration_angle_strata_phi != 1
            or mean_calibration_forced_collision
        ):
            raise ValueError(
                "Mean-calibration strata and forced collision require "
                "mean_calibration_histories_per_source_line."
            )
        target_sampled_primaries = require_target_sampled_primaries(
            payload.get("target_sampled_primaries")
        )
        if mean_calibration_enabled and target_sampled_primaries is not None:
            raise ValueError(
                "Mean calibration cannot combine with target_sampled_primaries."
            )
        if (
            target_sampled_primaries is not None
            and not accelerated_weighted_transport_enable
        ):
            raise ValueError(
                "target_sampled_primaries requires "
                "accelerated_weighted_transport_enable=true."
            )
        source_rate_model = _json_string(
            payload,
            "source_rate_model",
            default="detector_cps_1m",
            choices=frozenset(
                {
                    "detector_cps_1m",
                    "isotropic_emission_equivalent",
                    "parent_decay_activity_bq",
                }
            ),
        )
        primary_emission_model = _json_string(
            payload,
            "primary_emission_model",
            default="independent_gamma_lines",
            choices=frozenset(
                {
                    "independent_gamma_lines",
                    "geant4_radioactive_decay",
                }
            ),
        )
        source_bias_mode = _json_string(
            payload,
            "source_bias_mode",
            default="detector_cone",
            choices=frozenset({"analog", "detector_cone", "mixture_cone_isotropic"}),
        )
        source_bias_cone_half_angle_deg = _json_number(
            payload,
            "source_bias_cone_half_angle_deg",
            default=0.0,
            minimum=0.0,
        )
        if source_bias_cone_half_angle_deg > 180.0:
            raise ValueError("source_bias_cone_half_angle_deg must not exceed 180.")
        source_bias_isotropic_fraction = _json_number(
            payload,
            "source_bias_isotropic_fraction",
            default=1.0,
            strictly_positive=True,
        )
        if source_bias_isotropic_fraction > 1.0:
            raise ValueError("source_bias_isotropic_fraction must lie in (0, 1].")
        if source_rate_model == "detector_cps_1m":
            if source_bias_mode != "detector_cone":
                raise ValueError(
                    "source_rate_model=detector_cps_1m has the fixed effective "
                    "source_bias_mode=detector_cone."
                )
            if not np.isclose(source_bias_isotropic_fraction, 1.0):
                raise ValueError(
                    "source_rate_model=detector_cps_1m has the fixed effective "
                    "source_bias_isotropic_fraction=1.0."
                )
            source_bias_mode = "detector_cone"
            source_bias_isotropic_fraction = 1.0
        primary_sampling_fraction = require_primary_sampling_fraction(
            payload.get("primary_sampling_fraction", 1.0),
            accelerated_weighted_transport_enable=(
                accelerated_weighted_transport_enable
            ),
            target_sampled_primaries=target_sampled_primaries,
        )
        if (
            accelerated_weighted_transport_enable
            and source_rate_model != "detector_cps_1m"
        ):
            raise ValueError(
                "Accelerated weighted history thinning currently requires "
                "source_rate_model=detector_cps_1m."
            )
        detector_scoring_mode = _json_string(
            payload,
            "detector_scoring_mode",
            default="full_transport",
            choices=frozenset({"full_transport", "incident_gamma_energy"}),
        )
        secondary_transport_mode = _json_string(
            payload,
            "secondary_transport_mode",
            default="full_transport",
            choices=frozenset({"full_transport", "gamma_only"}),
        )
        if mean_calibration_enabled:
            if detector_scoring_mode not in {
                "incident_gamma_energy",
                "full_transport",
            }:
                raise ValueError(
                    "Fixed source-line sampling requires an implemented "
                    "detector scoring mode."
                )
            if (
                detector_scoring_mode == "full_transport"
                and mean_calibration_forced_collision
            ):
                raise ValueError(
                    "Full-detector fixed source-line validation forbids "
                    "forced-collision transport."
                )
            if sample_detector_response:
                raise ValueError(
                    "Mean calibration requires sample_detector_response=false."
                )
            if secondary_transport_mode != "full_transport":
                raise ValueError(
                    "Mean calibration requires secondary_transport_mode=full_transport."
                )
            if not validation_entry_class_spectra:
                raise ValueError(
                    "Mean calibration requires validation_entry_class_spectra=true."
                )
            if accelerated_weighted_transport_enable:
                raise ValueError(
                    "Mean calibration cannot combine with retired weighted "
                    "history thinning."
                )
            if not np.isclose(primary_sampling_fraction, 1.0):
                raise ValueError(
                    "Mean calibration requires primary_sampling_fraction=1.0."
                )
        elif sample_detector_response != (
            detector_scoring_mode == "incident_gamma_energy"
        ):
            raise ValueError(
                "sample_detector_response must be true exactly when "
                "detector_scoring_mode=incident_gamma_energy."
            )
        background_cps = _json_number(
            payload,
            "background_cps",
            default=0.0,
            minimum=0.0,
        )
        dead_time_tau_s = _json_number(
            payload,
            "dead_time_tau_s",
            default=5.813e-9,
            minimum=0.0,
        )
        if primary_emission_model == "geant4_radioactive_decay":
            if (
                source_rate_model != "parent_decay_activity_bq"
                or source_bias_mode != "analog"
                or detector_scoring_mode != "full_transport"
                or secondary_transport_mode != "full_transport"
                or not np.isclose(primary_sampling_fraction, 1.0)
                or target_sampled_primaries is not None
                or mean_calibration_enabled
                or sample_detector_response
            ):
                raise ValueError(
                    "primary_emission_model=geant4_radioactive_decay requires "
                    "source_rate_model=parent_decay_activity_bq, analog "
                    "source sampling, full detector and secondary transport, "
                    "unit primary sampling, and no target budget, response "
                    "sampling, or mean calibration."
                )
        if mean_calibration_enabled:
            if source_rate_model != "detector_cps_1m":
                raise ValueError(
                    "Mean calibration requires source_rate_model=detector_cps_1m."
                )
            if background_cps != 0.0:
                raise ValueError("Mean calibration requires background_cps=0.")
            if dead_time_tau_s != 0.0:
                raise ValueError("Mean calibration requires dead_time_tau_s=0.")
        renderer = _json_string(
            payload,
            "renderer",
            default="RayTracedLighting",
        )
        usd_path = payload.get("usd_path")
        if usd_path is not None:
            usd_path = _nonempty_string(usd_path, field_name="usd_path")
        engine_mode = _json_string(
            payload,
            "engine_mode",
            default="external",
            choices=frozenset({"external"}),
        )
        physics_profile = _json_string(
            payload,
            "physics_profile",
            default="balanced",
            choices=frozenset({"balanced"}),
        )
        executable_path_raw = payload.get(
            "executable_path",
            "build/geant4_sidecar",
        )
        executable_path = _nonempty_string(
            executable_path_raw,
            field_name="executable_path",
        )
        scatter_gain = _json_number(
            payload,
            "scatter_gain",
            default=0.0,
            minimum=0.0,
        )
        if scatter_gain != 0.0:
            raise ValueError(
                "scatter_gain is not a native Geant4 runtime option and must be 0.0."
            )
        crystal_shape = _nonempty_string(
            detector_payload.get("crystal_shape", "sphere"),
            field_name="detector_model.crystal_shape",
        )
        if crystal_shape != "sphere":
            raise ValueError(
                "detector_model.crystal_shape must be 'sphere'; native Geant4 "
                "constructs a spherical detector."
            )
        crystal_material = _nonempty_string(
            detector_payload.get("crystal_material", "cebr3"),
            field_name="detector_model.crystal_material",
        )
        housing_material = _nonempty_string(
            detector_payload.get("housing_material", "aluminum"),
            field_name="detector_model.housing_material",
        )
        return cls(
            use_mock_stage=use_mock_stage,
            headless=headless,
            renderer=renderer,
            usd_path=usd_path,
            detector_height_m=_json_number(
                payload,
                "detector_height_m",
                default=0.5,
                strictly_positive=True,
            ),
            robot_ground_z_m=_json_number(
                payload,
                "robot_ground_z_m",
                default=0.0,
            ),
            obstacle_height_m=_json_number(
                payload,
                "obstacle_height_m",
                default=2.0,
                strictly_positive=True,
            ),
            author_obstacle_prims=author_obstacle_prims,
            author_room_boundary_prims=author_room_boundary_prims,
            fe_shield_size_xyz=_json_vector3(
                payload,
                "fe_shield_size_xyz",
                default=(0.25, 0.08, 0.25),
            ),
            pb_shield_size_xyz=_json_vector3(
                payload,
                "pb_shield_size_xyz",
                default=(0.25, 0.08, 0.25),
            ),
            stage_material_rules=tuple(stage_material_rules),
            engine_mode=engine_mode,
            physics_profile=physics_profile,
            thread_count=_json_integer(
                payload,
                "thread_count",
                default=1,
                minimum=1,
            ),
            random_seed_base=_json_integer(
                payload,
                "random_seed_base",
                default=123,
                minimum=0,
            ),
            dead_time_tau_s=dead_time_tau_s,
            scatter_gain=scatter_gain,
            executable_path=executable_path,
            executable_args=normalized_executable_args,
            timeout_s=_json_number(
                payload,
                "timeout_s",
                default=120.0,
                strictly_positive=True,
            ),
            persistent_process=persistent_process,
            source_rate_model=source_rate_model,
            primary_emission_model=primary_emission_model,
            source_bias_mode=source_bias_mode,
            source_bias_cone_half_angle_deg=source_bias_cone_half_angle_deg,
            source_bias_isotropic_fraction=source_bias_isotropic_fraction,
            detector_scoring_mode=detector_scoring_mode,
            secondary_transport_mode=secondary_transport_mode,
            primary_sampling_fraction=primary_sampling_fraction,
            target_sampled_primaries=target_sampled_primaries,
            mean_calibration_histories_per_source_line=(
                mean_calibration_histories_per_source_line
            ),
            mean_calibration_angle_strata_mu=(mean_calibration_angle_strata_mu),
            mean_calibration_angle_strata_phi=(mean_calibration_angle_strata_phi),
            mean_calibration_forced_collision=(bool(mean_calibration_forced_collision)),
            accelerated_weighted_transport_enable=(
                accelerated_weighted_transport_enable
            ),
            background_cps=background_cps,
            sample_detector_response=sample_detector_response,
            validation_entry_class_spectra=(validation_entry_class_spectra),
            detector_model=ExportedDetectorModel(
                crystal_radius_m=_json_number(
                    detector_payload,
                    "crystal_radius_m",
                    default=DEFAULT_DETECTOR_CRYSTAL_RADIUS_M,
                    strictly_positive=True,
                ),
                crystal_length_m=_json_number(
                    detector_payload,
                    "crystal_length_m",
                    default=DEFAULT_DETECTOR_CRYSTAL_LENGTH_M,
                    strictly_positive=True,
                ),
                housing_thickness_m=_json_number(
                    detector_payload,
                    "housing_thickness_m",
                    default=DEFAULT_DETECTOR_HOUSING_THICKNESS_M,
                    minimum=0.0,
                ),
                coincidence_window_s=_json_number(
                    detector_payload,
                    "coincidence_window_s",
                    default=DEFAULT_DETECTOR_COINCIDENCE_WINDOW_S,
                    strictly_positive=True,
                ),
                crystal_shape=crystal_shape,
                crystal_material=crystal_material,
                housing_material=housing_material,
            ),
            shield_thickness=resolve_shield_thickness_config(payload),
            absorbing_transport_groups=tuple(absorbing_transport_groups),
            absorbing_path_prefixes=tuple(absorbing_path_prefixes),
            radiation_visualization=RadiationVisualizationConfig.from_dict(
                visualization_payload
            ),
        )


class Geant4Application:
    """Wrap Geant4 sidecar scene handling and spectrum generation."""

    def __init__(
        self,
        *,
        app_config: dict[str, Any] | None = None,
        stage_backend: StageBackend | None = None,
        production_runtime_config_sha256: str | None = None,
        expected_native_executable_sha256: str | None = None,
        expected_native_execution_environment_sha256: str | None = None,
        expected_implementation_bundle_sha256: str | None = None,
    ) -> None:
        """Create the application and initialize the requested stage backend."""
        if production_runtime_config_sha256 is not None and (
            not isinstance(production_runtime_config_sha256, str)
            or len(production_runtime_config_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in production_runtime_config_sha256
            )
        ):
            raise ValueError(
                "production_runtime_config_sha256 must be a lowercase SHA-256 string."
            )
        self.production_runtime_config_sha256 = production_runtime_config_sha256
        self.config = Geant4AppConfig.from_dict(app_config)
        self.native_executable_sha256: str | None = None
        self.native_execution_environment_sha256: str | None = None
        self.implementation_bundle_sha256: str | None = None
        for field_name, value in (
            (
                "expected_native_executable_sha256",
                expected_native_executable_sha256,
            ),
            (
                "expected_native_execution_environment_sha256",
                expected_native_execution_environment_sha256,
            ),
            (
                "expected_implementation_bundle_sha256",
                expected_implementation_bundle_sha256,
            ),
        ):
            if value is not None and (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 string.")
        expected_bundle_parts = (
            expected_native_executable_sha256,
            expected_native_execution_environment_sha256,
            expected_implementation_bundle_sha256,
        )
        if any(value is None for value in expected_bundle_parts) and any(
            value is not None for value in expected_bundle_parts
        ):
            raise ValueError(
                "Native execution provenance requires all approved digests."
            )
        if self.production_runtime_config_sha256 is not None or (
            expected_native_executable_sha256 is not None
        ):
            raw_executable = self.config.executable_path
            if raw_executable is None:
                raise RuntimeError(
                    "Production Geant4 sidecar has no native executable."
                )
            executable = Path(raw_executable).expanduser()
            if not executable.is_absolute():
                executable = Path(__file__).resolve().parents[3] / executable
            executable = executable.absolute()
            if self.production_runtime_config_sha256 is not None:
                self.native_executable_sha256 = native_executable_sha256(executable)
                self.native_execution_environment_sha256 = (
                    native_execution_environment_bundle_sha256(executable)
                )
            else:
                self.native_executable_sha256 = expected_native_executable_sha256
                self.native_execution_environment_sha256 = (
                    expected_native_execution_environment_sha256
                )
            native_engine_executable_path = executable.as_posix()
        else:
            native_engine_executable_path = self.config.executable_path
        if self.production_runtime_config_sha256 is not None:
            from spectrum.full_spectrum_acceptance_runner import (
                acceptance_implementation_bundle_sha256,
            )

            self.implementation_bundle_sha256 = acceptance_implementation_bundle_sha256(
                Path(__file__).resolve().parents[3]
            )
        elif expected_implementation_bundle_sha256 is not None:
            self.implementation_bundle_sha256 = expected_implementation_bundle_sha256
        self.scene = SceneDescription()
        self.asset_geometry = IsaacAssetGeometry(
            detector_height_m=self.config.detector_height_m,
            obstacle_height_m=self.config.obstacle_height_m,
            fe_shield_size_xyz=self.config.fe_shield_size_xyz,
            pb_shield_size_xyz=self.config.pb_shield_size_xyz,
        )
        backend = stage_backend
        if backend is None:
            if self.config.use_mock_stage:
                backend = FakeStageBackend()
            else:
                try:
                    backend = IsaacSimStageBackend(
                        headless=self.config.headless,
                        renderer=self.config.renderer,
                    )
                except ModuleNotFoundError as exc:
                    raise RuntimeError(
                        "Geant4 use_mock_stage=false requires Isaac Sim Python modules. "
                        "Run the bridge with Isaac Sim's python.sh or set "
                        "ISAACSIM_PYTHON=/path/to/isaacsim/python.sh for auto-start."
                    ) from exc
        self._stage_backend = backend
        self.scene_builder = SceneBuilder(
            backend,
            detector_height_m=self.config.detector_height_m,
            obstacle_height_m=self.config.obstacle_height_m,
            fe_shield_size_xyz=self.config.fe_shield_size_xyz,
            pb_shield_size_xyz=self.config.pb_shield_size_xyz,
        )
        self.robot_controller = RobotController(
            backend,
            self.scene.prim_paths,
            detector_height_m=self.config.detector_height_m,
            fe_offset_xyz=(0.0, 0.0, self.config.detector_height_m),
            pb_offset_xyz=(0.0, 0.0, self.config.detector_height_m),
            ground_z_m=self.config.robot_ground_z_m,
        )
        self.engine = build_geant4_engine(
            Geant4EngineConfig(
                physics_profile=self.config.physics_profile,
                thread_count=self.config.thread_count,
                random_seed_base=self.config.random_seed_base,
                dead_time_tau_s=self.config.dead_time_tau_s,
                scatter_gain=self.config.scatter_gain,
                executable_path=native_engine_executable_path,
                executable_args=self.config.executable_args,
                timeout_s=self.config.timeout_s,
                persistent_process=self.config.persistent_process,
                source_rate_model=self.config.source_rate_model,
                primary_emission_model=self.config.primary_emission_model,
                source_bias_mode=self.config.source_bias_mode,
                source_bias_cone_half_angle_deg=self.config.source_bias_cone_half_angle_deg,
                source_bias_isotropic_fraction=self.config.source_bias_isotropic_fraction,
                detector_scoring_mode=self.config.detector_scoring_mode,
                secondary_transport_mode=self.config.secondary_transport_mode,
                primary_sampling_fraction=self.config.primary_sampling_fraction,
                target_sampled_primaries=self.config.target_sampled_primaries,
                mean_calibration_histories_per_source_line=(
                    self.config.mean_calibration_histories_per_source_line
                ),
                mean_calibration_angle_strata_mu=(
                    self.config.mean_calibration_angle_strata_mu
                ),
                mean_calibration_angle_strata_phi=(
                    self.config.mean_calibration_angle_strata_phi
                ),
                mean_calibration_forced_collision=(
                    self.config.mean_calibration_forced_collision
                ),
                background_cps=self.config.background_cps,
                sample_detector_response=(self.config.sample_detector_response),
                validation_entry_class_spectra=(
                    self.config.validation_entry_class_spectra
                ),
                expected_native_executable_sha256=(self.native_executable_sha256),
                expected_native_execution_environment_sha256=(
                    self.native_execution_environment_sha256
                ),
                expected_implementation_bundle_sha256=(
                    self.implementation_bundle_sha256
                ),
                radiation_visualization=self.config.radiation_visualization,
            ),
            engine_mode=self.config.engine_mode,
        )
        self._last_cache_hit = False

    def reset(self, scene: SceneDescription) -> None:
        """Load a new scene description and rebuild or reuse the Geant4 world."""
        if self.production_runtime_config_sha256 is not None:
            if scene.use_config_usd_fallback:
                raise ValueError("Production Geant4 reset forbids config USD fallback.")
            if scene.usd_path != self.config.usd_path:
                raise ValueError(
                    "Production scene usd_path differs from its runtime config."
                )
            for field_name in (
                "author_obstacle_prims",
                "author_room_boundary_prims",
            ):
                if getattr(scene, field_name) is not getattr(
                    self.config,
                    field_name,
                ):
                    raise ValueError(
                        f"Production scene {field_name} differs from its runtime "
                        "config."
                    )
        else:
            if (
                scene.usd_path is None
                and scene.use_config_usd_fallback
                and self.config.usd_path is not None
            ):
                scene.usd_path = self.config.usd_path
            if self.config.author_obstacle_prims is not None:
                scene.author_obstacle_prims = self.config.author_obstacle_prims
            if self.config.author_room_boundary_prims is not None:
                scene.author_room_boundary_prims = (
                    self.config.author_room_boundary_prims
                )
        self.scene = scene
        self.scene_builder.load_scene(scene, usd_path_override=None)
        self.robot_controller = RobotController(
            self._stage_backend,
            scene.prim_paths,
            detector_height_m=self.config.detector_height_m,
            fe_offset_xyz=(0.0, 0.0, self.config.detector_height_m),
            pb_offset_xyz=(0.0, 0.0, self.config.detector_height_m),
            ground_z_m=self.config.robot_ground_z_m,
        )
        self.robot_controller.reset()
        exported_scene = export_scene_for_geant4(
            scene,
            stage_backend=self._stage_backend,
            asset_geometry=self.asset_geometry,
            detector_model=self.config.detector_model,
            shield_thickness=self.config.shield_thickness,
            stage_material_rules=self.config.stage_material_rules,
            absorbing_transport_groups=self.config.absorbing_transport_groups,
            absorbing_path_prefixes=self.config.absorbing_path_prefixes,
        )
        self._last_cache_hit = bool(self.engine.load_scene(exported_scene))

    def runtime_fidelity_metadata(self) -> dict[str, object]:
        """Return configured transport semantics for TCP reset handshakes."""
        weighted = bool(self.config.accelerated_weighted_transport_enable)
        budget_enabled = self.config.target_sampled_primaries is not None
        expected_policy_hash = surface_emission_policy_sha256()
        active_scene = getattr(self, "scene", None)
        scene_sources = () if active_scene is None else tuple(active_scene.sources)
        all_sources_surface_bound = bool(scene_sources) and all(
            source.transport_position_xyz is not None
            and source.surface_chart_id is not None
            and source.surface_uv is not None
            and source.surface_normal_xyz is not None
            and source.surface_emission_policy_sha256 == expected_policy_hash
            for source in scene_sources
        )
        source_contract_sha256 = ""
        if all_sources_surface_bound:
            source_contract_sha256 = surface_source_runtime_contract_sha256(
                [
                    {
                        "isotope": source.isotope,
                        "position": list(source.position_xyz),
                        "transport_position": list(source.transport_position_xyz),
                        "intensity_cps_1m": float(source.intensity_cps_1m),
                        "surface_chart_id": source.surface_chart_id,
                        "surface_uv": list(source.surface_uv),
                        "surface_normal": list(source.surface_normal_xyz),
                        "surface_emission_policy_sha256": (
                            source.surface_emission_policy_sha256
                        ),
                    }
                    for source in scene_sources
                ]
            )
        engine_scene = getattr(getattr(self, "engine", None), "scene", None)
        scene_hash = (
            "" if engine_scene is None else str(getattr(engine_scene, "scene_hash", ""))
        )
        metadata: dict[str, object] = {
            "primary_sampling_fraction": float(self.config.primary_sampling_fraction),
            "primary_history_weight": float(
                1.0 / self.config.primary_sampling_fraction
            ),
            "accelerated_weighted_transport_enable": weighted,
            "requested_primary_sampling_fraction": float(
                self.config.primary_sampling_fraction
            ),
            "target_sampled_primaries": int(self.config.target_sampled_primaries or 0),
            "mean_calibration_histories_per_source_line": int(
                self.config.mean_calibration_histories_per_source_line or 0
            ),
            "mean_calibration_angle_strata_mu": int(
                self.config.mean_calibration_angle_strata_mu
            ),
            "mean_calibration_angle_strata_phi": int(
                self.config.mean_calibration_angle_strata_phi
            ),
            "mean_calibration_forced_collision": bool(
                self.config.mean_calibration_forced_collision
            ),
            "primary_sampling_budget_enabled": budget_enabled,
            "primary_sampling_fraction_resolution": (
                "per_observation_pending" if budget_enabled else "fixed_fraction"
            ),
            "dead_time_tau_s": float(self.config.dead_time_tau_s),
            "source_rate_model": str(self.config.source_rate_model),
            "primary_emission_model": str(self.config.primary_emission_model),
            "intensity_cps_1m_definition": ("pre_dead_time_detector_pulse_rate_at_1m"),
            "requested_threads": int(self.config.thread_count),
            "physics_profile": str(self.config.physics_profile),
            "detector_scoring_mode": str(self.config.detector_scoring_mode),
            "secondary_transport_mode": str(self.config.secondary_transport_mode),
            "source_bias_mode": str(self.config.source_bias_mode),
            "source_bias_isotropic_fraction": float(
                self.config.source_bias_isotropic_fraction
            ),
            "background_cps": float(self.config.background_cps),
            "sample_detector_response": bool(self.config.sample_detector_response),
            "detector_response_sampling_contract_sha256": (
                NATIVE_GEANT4_DETECTOR_RESPONSE_CONTRACT_SHA256
            ),
            "spectrum_energy_min_keV": float(NATIVE_GEANT4_ENERGY_MIN_KEV),
            "spectrum_energy_max_keV": float(NATIVE_GEANT4_ENERGY_MAX_KEV),
            "spectrum_bin_width_keV": float(NATIVE_GEANT4_BIN_WIDTH_KEV),
            "spectrum_bin_count": int(NATIVE_GEANT4_BIN_COUNT),
            "background_spectrum_model_id": (NATIVE_GEANT4_BACKGROUND_MODEL_ID),
            "source_position_semantics": "air_side_native_emission_xyz",
            "source_anchor_semantics": ("exact_surface_chart_uv_evaluation_truth"),
            "all_sources_surface_bound": all_sources_surface_bound,
            "surface_emission_epsilon_m": SURFACE_EMISSION_EPSILON_M,
            "surface_emission_policy_sha256": (
                expected_policy_hash if all_sources_surface_bound else ""
            ),
            "surface_source_contract_sha256": source_contract_sha256,
            "scene_hash": scene_hash,
            "shield_pose_contract_id": SHIELD_POSE_CONTRACT_ID,
            "shield_pose_contract_sha256": SHIELD_POSE_CONTRACT_SHA256,
        }
        if self.production_runtime_config_sha256 is not None:
            metadata["production_runtime_config_sha256"] = (
                self.production_runtime_config_sha256
            )
            metadata["native_executable_sha256"] = self.native_executable_sha256
            metadata["native_execution_environment_sha256"] = (
                self.native_execution_environment_sha256
            )
            metadata["implementation_bundle_sha256"] = self.implementation_bundle_sha256
        if budget_enabled:
            metadata["history_thinning_resolution"] = "per_observation_pending"
        else:
            history_thinning_enabled = self.config.primary_sampling_fraction < 1.0
            metadata["history_thinning_enabled"] = history_thinning_enabled
            metadata["transport_history_mode"] = (
                "weighted_thinning" if history_thinning_enabled else "full_unit_weight"
            )
        return metadata

    def step(self, command: SimulationCommand) -> SimulationObservation:
        """Apply a command and return the resulting Geant4-backed observation."""
        self.robot_controller.apply_command(command)
        detector_pose = self.robot_controller.detector_world_pose()
        fe_pose = self._stage_backend.get_world_pose(
            self.scene.prim_paths.fe_shield_path
        )
        pb_pose = self._stage_backend.get_world_pose(
            self.scene.prim_paths.pb_shield_path
        )
        spectrum, metadata = self.engine.simulate(
            Geant4StepRequest(
                step_id=command.step_id,
                dwell_time_s=float(command.dwell_time_s),
                seed=int(self.config.random_seed_base + int(command.step_id)),
                detector_pose_xyz=detector_pose.translation_xyz,
                detector_quat_wxyz=detector_pose.orientation_wxyz,
                fe_shield_pose_xyz=fe_pose.translation_xyz,
                fe_shield_quat_wxyz=fe_pose.orientation_wxyz,
                pb_shield_pose_xyz=pb_pose.translation_xyz,
                pb_shield_quat_wxyz=pb_pose.orientation_wxyz,
                fe_orientation_index=int(command.fe_orientation_index),
                pb_orientation_index=int(command.pb_orientation_index),
            )
        )
        metadata = dict(metadata)
        mean_calibration_quota = self.config.mean_calibration_histories_per_source_line
        if mean_calibration_quota is not None:
            validate_mean_calibration_transport_metadata(
                metadata,
                expected_histories_per_source_line=mean_calibration_quota,
                expected_angle_strata_mu=(self.config.mean_calibration_angle_strata_mu),
                expected_angle_strata_phi=(
                    self.config.mean_calibration_angle_strata_phi
                ),
                expected_forced_collision=(
                    self.config.mean_calibration_forced_collision
                ),
                expected_source_rate_model=self.config.source_rate_model,
                expected_thread_count=self.config.thread_count,
                expected_physics_profile=self.config.physics_profile,
                expected_detector_scoring_mode=(self.config.detector_scoring_mode),
                expected_secondary_transport_mode=(
                    self.config.secondary_transport_mode
                ),
                expected_source_bias_mode=self.config.source_bias_mode,
            )
            _validate_native_mean_calibration_response(spectrum, metadata)
            canonical_spectrum: list[int | float] = np.asarray(
                spectrum,
                dtype=np.float64,
            ).tolist()
        else:
            validate_transport_metadata(
                metadata,
                expected_primary_sampling_fraction=(
                    self.config.primary_sampling_fraction
                ),
                expected_target_sampled_primaries=(
                    self.config.target_sampled_primaries
                ),
                accelerated_weighted_transport_enable=(
                    self.config.accelerated_weighted_transport_enable
                ),
                expected_source_rate_model=self.config.source_rate_model,
                expected_primary_emission_model=(self.config.primary_emission_model),
                expected_thread_count=self.config.thread_count,
                expected_physics_profile=self.config.physics_profile,
                expected_detector_scoring_mode=(self.config.detector_scoring_mode),
                expected_secondary_transport_mode=(
                    self.config.secondary_transport_mode
                ),
                expected_source_bias_mode=self.config.source_bias_mode,
                expected_background_cps=self.config.background_cps,
                expected_dead_time_tau_s=self.config.dead_time_tau_s,
                expected_detector_response_sampling=(
                    self.config.sample_detector_response
                ),
            )
        if (
            mean_calibration_quota is None
            and self.config.accelerated_weighted_transport_enable
        ):
            _validate_native_weighted_response(spectrum, metadata)
            canonical_spectrum = np.asarray(
                spectrum,
                dtype=np.float64,
            ).tolist()
        elif mean_calibration_quota is None and self.config.sample_detector_response:
            canonical_spectrum = _validate_native_sampled_event_response(
                spectrum,
                metadata,
            ).tolist()
        elif mean_calibration_quota is None:
            canonical_spectrum = np.asarray(
                spectrum,
                dtype=np.float64,
            ).tolist()
        metadata["accelerated_weighted_transport_enable"] = bool(
            self.config.accelerated_weighted_transport_enable
        )
        metadata["shield_pose_contract_id"] = SHIELD_POSE_CONTRACT_ID
        metadata["shield_pose_contract_sha256"] = SHIELD_POSE_CONTRACT_SHA256
        metadata.setdefault("cache_hit", self._last_cache_hit)
        metadata.setdefault("fe_orientation_index", int(command.fe_orientation_index))
        metadata.setdefault("pb_orientation_index", int(command.pb_orientation_index))
        metadata.setdefault("shield_num_orientations", 8)
        metadata.setdefault(
            "shield_pair_id",
            int(command.fe_orientation_index) * 8 + int(command.pb_orientation_index),
        )
        metadata.setdefault(
            "shield_thickness_scale",
            float(self.config.shield_thickness.thickness_scale),
        )
        metadata.setdefault(
            "shield_thickness_fe_cm",
            float(self.config.shield_thickness.thickness_fe_cm),
        )
        metadata.setdefault(
            "shield_thickness_pb_cm",
            float(self.config.shield_thickness.thickness_pb_cm),
        )
        energy = np.arange(NATIVE_GEANT4_BIN_COUNT, dtype=np.float64) * float(
            NATIVE_GEANT4_BIN_WIDTH_KEV
        )
        bin_width_keV = float(NATIVE_GEANT4_BIN_WIDTH_KEV)
        edges = list(energy) + [float(energy[-1] + bin_width_keV)]
        return SimulationObservation(
            step_id=command.step_id,
            detector_pose_xyz=detector_pose.translation_xyz,
            detector_quat_wxyz=detector_pose.orientation_wxyz,
            fe_orientation_index=command.fe_orientation_index,
            pb_orientation_index=command.pb_orientation_index,
            spectrum_counts=canonical_spectrum,
            energy_bin_edges_keV=[float(v) for v in edges],
            metadata=metadata,
        )

    def close(self) -> None:
        """Close the underlying engine and stage backend."""
        self.engine.close()
        self._stage_backend.close()
