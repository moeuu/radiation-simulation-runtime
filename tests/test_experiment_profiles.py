"""Tests for runtime-owned experiment and acquisition contracts."""

from __future__ import annotations

import pytest

from runtime.experiment_profiles import (
    AcquisitionContract,
    MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE,
    MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE_ID,
    STANDARD_ACQUISITION_LIVE_TIME_S,
    acquisition_contract_from_environment,
    experiment_profile_from_environment,
    require_experiment_profile,
    require_private_scene_variant,
)


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("schema_version", True),
        ("schema_version", 1.0),
        ("max_stations", True),
        ("views_per_station", 8.0),
        ("live_time_s", True),
        ("live_time_s", "20.0"),
        ("coverage_radius_m", float("nan")),
    ),
)
def test_acquisition_contract_rejects_coerced_wire_scalars(
    field: str,
    invalid: object,
) -> None:
    """The public acquisition contract must accept exact JSON scalar types only."""
    payload = MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE.acquisition.to_payload()
    payload[field] = invalid

    with pytest.raises((TypeError, ValueError)):
        AcquisitionContract.from_payload(payload)


def test_named_profile_owns_every_shared_acquisition_value() -> None:
    """The named experiment must expose its exact acquisition contract."""
    profile = MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE

    assert profile.profile_id == MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE_ID
    assert STANDARD_ACQUISITION_LIVE_TIME_S == 20.0
    assert (
        profile.environment.size_x,
        profile.environment.size_y,
        profile.environment.size_z,
    ) == (10.0, 15.0, 5.0)
    assert profile.acquisition.to_payload() == {
        "schema_version": 1,
        "max_stations": 16,
        "views_per_station": 8,
        "live_time_s": STANDARD_ACQUISITION_LIVE_TIME_S,
        "max_measurements": 128,
        "min_station_separation_m": 3.0,
        "coverage_radius_m": 3.0,
    }


def test_environment_contract_round_trip_rejects_profile_drift() -> None:
    """Consumers must fail when a declared profile carries altered limits."""
    profile = MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE
    environment = profile.public_environment_fields()

    assert acquisition_contract_from_environment(environment) == profile.acquisition
    assert experiment_profile_from_environment(environment) is profile

    changed = dict(environment)
    contract = dict(profile.acquisition.to_payload())
    contract["live_time_s"] = 30.0
    changed["acquisition_contract"] = contract
    with pytest.raises(ValueError, match="differs"):
        experiment_profile_from_environment(changed)


@pytest.mark.parametrize("value", (None, 7, ""))
def test_profile_selection_rejects_implicit_or_coerced_ids(value: object) -> None:
    """Profile and variant lookup must accept explicit nonempty strings only."""
    with pytest.raises(TypeError):
        require_experiment_profile(value)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        require_private_scene_variant(
            MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE_ID,
            value,  # type: ignore[arg-type]
        )
