"""Define runtime-owned physical experiments and public acquisition contracts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType

from measurement.model import EnvironmentConfig

MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE_ID = "multi_isotope_surface_search"
CS_CO_SURFACE_SEARCH_PROFILE_ID = "cs4_co3_surface_search"
ACQUISITION_CONTRACT_FIELD = "acquisition_contract"
EXPERIMENT_PROFILE_ID_FIELD = "experiment_profile_id"
STANDARD_ACQUISITION_LIVE_TIME_S = 20.0
STANDARD_OBSTACLE_MATERIAL = "concrete"
STANDARD_ROOM_BOUNDARY_THICKNESS_M = 0.1


@dataclass(frozen=True, slots=True)
class AcquisitionContract:
    """Describe estimator-neutral measurement and station limits."""

    max_stations: int
    views_per_station: int
    live_time_s: float
    max_measurements: int
    min_station_separation_m: float
    coverage_radius_m: float

    def __post_init__(self) -> None:
        """Validate one internally consistent acquisition contract."""
        for name in ("max_stations", "views_per_station", "max_measurements"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact JSON integer.")
            if value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.max_measurements != self.max_stations * self.views_per_station:
            raise ValueError(
                "max_measurements must equal max_stations * views_per_station."
            )
        for name in (
            "live_time_s",
            "min_station_separation_m",
            "coverage_radius_m",
        ):
            raw_value = getattr(self, name)
            if type(raw_value) not in (int, float):
                raise TypeError(f"{name} must be an exact JSON number.")
            value = float(raw_value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive.")
            object.__setattr__(self, name, value)

    def to_payload(self) -> dict[str, object]:
        """Return the strict truth-free environment payload."""
        return {
            "schema_version": 1,
            "max_stations": self.max_stations,
            "views_per_station": self.views_per_station,
            "live_time_s": self.live_time_s,
            "max_measurements": self.max_measurements,
            "min_station_separation_m": self.min_station_separation_m,
            "coverage_radius_m": self.coverage_radius_m,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "AcquisitionContract":
        """Parse one exact acquisition payload without implicit defaults."""
        if not isinstance(payload, Mapping):
            raise TypeError("acquisition_contract must be a JSON object.")
        expected = {
            "schema_version",
            "max_stations",
            "views_per_station",
            "live_time_s",
            "max_measurements",
            "min_station_separation_m",
            "coverage_radius_m",
        }
        if set(payload) != expected:
            raise ValueError("acquisition_contract must match schema version 1.")
        schema_version = payload["schema_version"]
        if type(schema_version) is not int or schema_version != 1:
            raise ValueError(
                "acquisition_contract schema_version must be exact integer 1."
            )
        return cls(
            max_stations=payload["max_stations"],
            views_per_station=payload["views_per_station"],
            live_time_s=payload["live_time_s"],
            max_measurements=payload["max_measurements"],
            min_station_separation_m=payload["min_station_separation_m"],
            coverage_radius_m=payload["coverage_radius_m"],
        )


@dataclass(frozen=True, slots=True)
class _PrivateSceneVariant:
    """Describe one private source realization family within an experiment."""

    variant_id: str
    isotope_sequence: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the private variant identifier and source sequence."""
        if not self.variant_id or not self.isotope_sequence:
            raise ValueError("A scene variant requires an ID and source sequence.")

    @property
    def source_counts(self) -> Mapping[str, int]:
        """Return immutable exact source counts for private validation."""
        return MappingProxyType(dict(Counter(self.isotope_sequence)))


@dataclass(frozen=True, slots=True)
class ExperimentProfile:
    """Keep every standard physical and acquisition choice in one object."""

    profile_id: str
    environment: EnvironmentConfig
    acquisition: AcquisitionContract
    candidate_isotopes: tuple[str, ...]
    runtime_config_relative_path: str
    isotope_experiment_profile: str
    candidate_count: int
    passage_width_m: float
    blocked_fraction: float
    same_isotope_min_distance_m: float
    intensity_cps_1m: tuple[float, float]
    environment_model_id: str
    obstacle_material: str
    room_boundary_thickness_m: float
    surface_chart_max_edge_m: float

    def __post_init__(self) -> None:
        """Validate the public physical and acquisition profile."""
        if (
            not self.profile_id
            or not self.environment_model_id
            or not self.obstacle_material
        ):
            raise ValueError(
                "profile_id, environment_model_id, and obstacle_material must "
                "be nonempty."
            )
        if not self.candidate_isotopes or len(set(self.candidate_isotopes)) != len(
            self.candidate_isotopes
        ):
            raise ValueError("candidate_isotopes must be nonempty and unique.")
        if (
            isinstance(self.candidate_count, bool)
            or not isinstance(self.candidate_count, int)
            or self.candidate_count < 8
        ):
            raise ValueError("candidate_count must be an integer of at least 8.")
        for name in (
            "passage_width_m",
            "room_boundary_thickness_m",
            "same_isotope_min_distance_m",
            "surface_chart_max_edge_m",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive.")
        if not 0.0 <= float(self.blocked_fraction) < 1.0:
            raise ValueError("blocked_fraction must lie in [0, 1).")
        if (
            len(self.intensity_cps_1m) != 2
            or float(self.intensity_cps_1m[0]) <= 0.0
            or float(self.intensity_cps_1m[1]) < float(self.intensity_cps_1m[0])
        ):
            raise ValueError("intensity_cps_1m must be one positive ordered range.")

    def public_environment_fields(self) -> dict[str, object]:
        """Return profile-specific fields added to one authored environment."""
        return {
            EXPERIMENT_PROFILE_ID_FIELD: self.profile_id,
            ACQUISITION_CONTRACT_FIELD: self.acquisition.to_payload(),
        }


MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE = ExperimentProfile(
    profile_id=MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE_ID,
    environment=EnvironmentConfig(
        size_x=10.0,
        size_y=15.0,
        size_z=5.0,
        detector_position=(1.0, 1.0, 0.5),
    ),
    acquisition=AcquisitionContract(
        max_stations=16,
        views_per_station=8,
        live_time_s=STANDARD_ACQUISITION_LIVE_TIME_S,
        max_measurements=128,
        min_station_separation_m=3.0,
        coverage_radius_m=3.0,
    ),
    candidate_isotopes=("Co-60", "Cs-137", "Eu-154"),
    runtime_config_relative_path=(
        "configs/geant4/variance_reduction_external_no_isaac_32threads.json"
    ),
    isotope_experiment_profile="unconditioned_eu154",
    candidate_count=256,
    passage_width_m=2.0,
    blocked_fraction=0.4,
    same_isotope_min_distance_m=3.0,
    intensity_cps_1m=(300_000.0, 2_000_000.0),
    environment_model_id="random_manchester_component_union",
    obstacle_material=STANDARD_OBSTACLE_MATERIAL,
    room_boundary_thickness_m=STANDARD_ROOM_BOUNDARY_THICKNESS_M,
    surface_chart_max_edge_m=1.0,
)

CS_CO_SURFACE_SEARCH_PROFILE = ExperimentProfile(
    profile_id=CS_CO_SURFACE_SEARCH_PROFILE_ID,
    environment=MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE.environment,
    acquisition=MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE.acquisition,
    candidate_isotopes=("Co-60", "Cs-137"),
    runtime_config_relative_path=(
        "configs/geant4/cs_co_external_no_isaac_32threads.json"
    ),
    isotope_experiment_profile="unconditioned_cs_co",
    candidate_count=MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE.candidate_count,
    passage_width_m=MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE.passage_width_m,
    blocked_fraction=MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE.blocked_fraction,
    same_isotope_min_distance_m=(
        MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE.same_isotope_min_distance_m
    ),
    intensity_cps_1m=MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE.intensity_cps_1m,
    environment_model_id="random_manchester_component_union",
    obstacle_material=MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE.obstacle_material,
    room_boundary_thickness_m=(
        MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE.room_boundary_thickness_m
    ),
    surface_chart_max_edge_m=(
        MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE.surface_chart_max_edge_m
    ),
)

_PRIVATE_SCENE_VARIANTS_BY_PROFILE: Mapping[str, Mapping[str, _PrivateSceneVariant]] = (
    MappingProxyType(
        {
            MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE_ID: MappingProxyType(
                {
                    "mix9": _PrivateSceneVariant(
                        variant_id="mix9",
                        isotope_sequence=(
                            "Cs-137",
                            "Cs-137",
                            "Cs-137",
                            "Cs-137",
                            "Co-60",
                            "Co-60",
                            "Co-60",
                            "Eu-154",
                            "Eu-154",
                        ),
                    ),
                }
            ),
            CS_CO_SURFACE_SEARCH_PROFILE_ID: MappingProxyType(
                {
                    "cs4-co3": _PrivateSceneVariant(
                        variant_id="cs4-co3",
                        isotope_sequence=(
                            "Cs-137",
                            "Cs-137",
                            "Cs-137",
                            "Cs-137",
                            "Co-60",
                            "Co-60",
                            "Co-60",
                        ),
                    ),
                }
            ),
        }
    )
)

_EXPERIMENT_PROFILES: Mapping[str, ExperimentProfile] = MappingProxyType(
    {
        MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE.profile_id: (
            MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE
        ),
        CS_CO_SURFACE_SEARCH_PROFILE.profile_id: CS_CO_SURFACE_SEARCH_PROFILE,
    }
)


def available_experiment_profiles() -> tuple[str, ...]:
    """Return public experiment-profile identifiers in deterministic order."""
    return tuple(sorted(_EXPERIMENT_PROFILES))


def require_experiment_profile(profile_id: str) -> ExperimentProfile:
    """Return one runtime-owned experiment profile by identifier."""
    if type(profile_id) is not str or not profile_id:
        raise TypeError("experiment profile ID must be a nonempty JSON string.")
    try:
        return _EXPERIMENT_PROFILES[profile_id]
    except KeyError as exc:
        expected = ", ".join(available_experiment_profiles())
        raise ValueError(
            f"Unknown experiment profile {profile_id!r}; expected: {expected}."
        ) from exc


def available_private_scene_variants(profile_id: str) -> tuple[str, ...]:
    """Return private variant IDs for runtime-owned scenario authoring."""
    require_experiment_profile(profile_id)
    return tuple(sorted(_PRIVATE_SCENE_VARIANTS_BY_PROFILE[profile_id]))


def require_private_scene_variant(
    profile_id: str,
    variant_id: str,
) -> _PrivateSceneVariant:
    """Return one private runtime variant without publishing it as package API."""
    require_experiment_profile(profile_id)
    if type(variant_id) is not str or not variant_id:
        raise TypeError("scene variant ID must be a nonempty JSON string.")
    try:
        return _PRIVATE_SCENE_VARIANTS_BY_PROFILE[profile_id][variant_id]
    except KeyError as exc:
        expected = ", ".join(available_private_scene_variants(profile_id))
        raise ValueError(
            f"Unknown private scene variant {variant_id!r}; expected: {expected}."
        ) from exc


def acquisition_contract_from_environment(
    environment: Mapping[str, object],
) -> AcquisitionContract:
    """Load the runtime-authored acquisition contract from an environment."""
    if not isinstance(environment, Mapping):
        raise TypeError("environment must be a mapping.")
    return AcquisitionContract.from_payload(environment.get(ACQUISITION_CONTRACT_FIELD))


def experiment_profile_from_environment(
    environment: Mapping[str, object],
) -> ExperimentProfile:
    """Resolve the declared runtime experiment from a truth-free environment."""
    if not isinstance(environment, Mapping):
        raise TypeError("environment must be a mapping.")
    profile_id = environment.get(EXPERIMENT_PROFILE_ID_FIELD)
    if not isinstance(profile_id, str) or not profile_id:
        raise ValueError("environment must declare experiment_profile_id.")
    profile = require_experiment_profile(profile_id)
    observed = acquisition_contract_from_environment(environment)
    if observed != profile.acquisition:
        raise ValueError("Environment acquisition contract differs from its profile.")
    return profile


__all__ = [
    "ACQUISITION_CONTRACT_FIELD",
    "AcquisitionContract",
    "CS_CO_SURFACE_SEARCH_PROFILE",
    "CS_CO_SURFACE_SEARCH_PROFILE_ID",
    "EXPERIMENT_PROFILE_ID_FIELD",
    "ExperimentProfile",
    "MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE",
    "MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE_ID",
    "STANDARD_ACQUISITION_LIVE_TIME_S",
    "STANDARD_OBSTACLE_MATERIAL",
    "STANDARD_ROOM_BOUNDARY_THICKNESS_M",
    "acquisition_contract_from_environment",
    "available_experiment_profiles",
    "experiment_profile_from_environment",
    "require_experiment_profile",
]
