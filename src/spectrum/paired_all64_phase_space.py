"""Fail-closed contracts for paired all-64 Geant4 phase-space replay.

This module deliberately contains no transport orchestration.  It defines the
artifact schemas and statistical bookkeeping required before a detector-
boundary phase-space bank can be connected to Geant4.  The paired bank is a
calibration/acceptance facility, not a standard-runtime observation mode:
reusing one upstream bank preserves each shield pair's marginal distribution
but intentionally correlates the 64 counterfactual results.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


PAIRED_ALL64_BANK_SCHEMA_VERSION = 3
PAIRED_ALL64_MANIFEST_SCHEMA_VERSION = 2
PAIRED_ALL64_COVARIANCE_SCHEMA_VERSION = 2
PAIRED_ALL64_PROFILE = "geant4_phase_space_paired_all64_v3"
PAIRED_ALL64_SEMANTICS = "paired_counterfactual_mean_v1"
PAIRED_ALL64_BANK_FORMAT = "event_grouped_detector_boundary_v3"
PAIRED_ALL64_PAIR_IDS = tuple(range(64))
PAIRED_ALL64_STANDARD_RUNTIME_SELECTABLE = False
PAIRED_ALL64_ESTIMATOR_COEFFICIENT_SEMANTICS = (
    "external_fixed_quota_history_coefficient_v1"
)
PAIRED_ALL64_REQUIRED_FIDELITY: Mapping[str, object] = {
    "physics_list": "FTFP_BERT",
    "em_physics": "G4EmStandardPhysics_option4",
    "secondary_transport_mode": "full_transport",
    "primary_sampling_fraction": 1.0,
    "primary_history_weight": 1.0,
    "crossing_track_weight": 1.0,
    "history_thinning_enabled": False,
    "weighted_transport": False,
    "theory_tvl_attenuation": False,
    "full_world_replay": True,
    "kill_outward_boundary_crossings": False,
    "event_grouping": "one_original_primary_history_per_g4event",
    "transported_particle_crossing_policy": "preserve_exact_restart_state",
    "weighted_geant4_crossing_policy": "fail_closed",
    "estimator_coefficient_application": "external_to_geant4_transport",
    "sample_detector_response": False,
    "background_cps": 0.0,
    "dead_time_tau_s": 0.0,
}

_BANK_SEED_DOMAIN = "paired_all64_phase_space_bank_seed_v2"
_REPLAY_SEED_DOMAIN = "paired_all64_phase_space_replay_seed_v2"
_HISTORY_IDENTITY_DOMAIN = "paired_all64_history_estimator_identity_v1"
_STRATUM_ASSIGNMENT_DOMAIN = "paired_all64_stratum_assignment_v1"
_COVARIANCE_SEMANTICS = (
    "stratified_fixed_quota_original_history_covariance_v1"
)
_HISTORY_ESTIMATOR_IDENTITY_KEYS = frozenset(
    {
        "original_history_id",
        "source_index",
        "line_index",
        "angle_stratum_index",
        "angle_stratum_count",
        "estimator_coefficient",
    }
)
_COVARIANCE_GROUP_KEYS = frozenset(
    {
        "source_index",
        "line_index",
        "angle_stratum_index",
        "angle_stratum_count",
        "history_count",
        "estimator_coefficient",
    }
)
_BANK_KEYS = frozenset(
    {
        "schema_version",
        "profile",
        "semantics",
        "bank_format",
        "endianness",
        "length_unit",
        "energy_unit",
        "time_unit",
        "geant4_version",
        "native_executable_sha256",
        "scene_sha256",
        "static_geometry_sha256",
        "material_contract_sha256",
        "source_contract_sha256",
        "source_schedule_sha256",
        "detector_center_m",
        "boundary_radius_m",
        "dwell_time_s",
        "root_seed",
        "bank_seed",
        "history_count",
        "estimator_group_count",
        "history_estimator_identity_sha256",
        "estimator_coefficient_semantics",
        "zero_crossing_history_count",
        "crossing_count",
        "non_gamma_crossing_count",
        "nonunit_crossing_weight_count",
        "species_counts",
        "bank_payload_sha256",
        "fidelity",
    }
)
_PAIR_KEYS = frozenset(
    {
        "shield_pair_id",
        "fe_orientation_index",
        "pb_orientation_index",
        "detector_center_m",
        "fe_center_m",
        "pb_center_m",
        "scene_sha256",
        "source_contract_sha256",
        "source_schedule_sha256",
        "dwell_time_s",
        "root_seed",
        "replay_seed",
        "fidelity",
    }
)
_COVARIANCE_KEYS = frozenset(
    {
        "schema_version",
        "semantics",
        "pair_ids",
        "history_count",
        "group_count",
        "feature_count",
        "score_semantics",
        "history_estimator_identity_sha256",
        "group_assignment_sha256",
        "estimator_coefficient_semantics",
        "groups",
        "original_history_ids_shape",
        "history_group_indices_shape",
        "estimate_shape",
        "first_sum_shape",
        "centered_factor_shape",
        "total_cross_pair_covariance_shape",
        "artifact_sha256",
    }
)
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "profile",
        "semantics",
        "standard_runtime_selectable",
        "complete",
        "bank",
        "pairs",
        "cross_pair_stratified_covariance",
    }
)


def _canonical_json_bytes(payload: object) -> bytes:
    """Return deterministic strict JSON bytes for hashing."""
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _require_exact_keys(
    payload: Mapping[str, object],
    expected: frozenset[str],
    *,
    field_name: str,
) -> None:
    """Reject missing or unknown keys in one versioned schema."""
    actual = frozenset(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"{field_name} keys differ from schema: "
            f"missing={missing}, unknown={unknown}."
        )


def _require_sha256(value: object, *, field_name: str) -> str:
    """Return one lowercase SHA-256 digest or fail closed."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
    return value


def _require_nonempty_string(value: object, *, field_name: str) -> str:
    """Return one nonempty string or fail closed."""
    if not isinstance(value, str) or not value:
        raise TypeError(f"{field_name} must be a nonempty string.")
    return value


def _require_integer(
    value: object,
    *,
    field_name: str,
    minimum: int = 0,
) -> int:
    """Return one bounded integer without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer.")
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}.")
    return parsed


def _require_bounded_integer(
    value: object,
    *,
    field_name: str,
    maximum: int,
) -> int:
    """Return one nonnegative integer bounded by a native wire type."""
    parsed = _require_integer(value, field_name=field_name)
    if parsed > maximum:
        raise ValueError(f"{field_name} must be at most {maximum}.")
    return parsed


def _require_seed(value: object, *, field_name: str) -> int:
    """Return one nonnegative signed-63-bit seed."""
    parsed = _require_integer(value, field_name=field_name)
    if parsed >= (1 << 63):
        raise ValueError(f"{field_name} must fit in a signed 63-bit integer.")
    return parsed


def _require_finite_number(
    value: object,
    *,
    field_name: str,
    positive: bool = False,
) -> float:
    """Return one finite JSON number under an optional positivity gate."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a JSON number.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite.")
    if positive and parsed <= 0.0:
        raise ValueError(f"{field_name} must be positive.")
    return parsed


def _require_center(value: object, *, field_name: str) -> tuple[float, ...]:
    """Return one exact finite three-dimensional center."""
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 3
    ):
        raise TypeError(f"{field_name} must contain exactly three numbers.")
    return tuple(
        _require_finite_number(item, field_name=f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )


def _required_fidelity() -> dict[str, object]:
    """Return a mutable copy of the immutable paired fidelity contract."""
    return dict(PAIRED_ALL64_REQUIRED_FIDELITY)


@dataclass(frozen=True)
class HistoryEstimatorIdentity:
    """Identify one unit-weight history and its external estimator role."""

    original_history_id: int
    source_index: int
    line_index: int
    angle_stratum_index: int
    angle_stratum_count: int
    estimator_coefficient: float

    def payload(self) -> dict[str, object]:
        """Return the canonical JSON-compatible history identity."""
        return {
            "original_history_id": self.original_history_id,
            "source_index": self.source_index,
            "line_index": self.line_index,
            "angle_stratum_index": self.angle_stratum_index,
            "angle_stratum_count": self.angle_stratum_count,
            "estimator_coefficient": self.estimator_coefficient,
        }


def _parse_history_estimator_identity(
    value: object,
    *,
    field_name: str,
) -> HistoryEstimatorIdentity:
    """Parse one exact native-v3 history estimator identity."""
    if isinstance(value, HistoryEstimatorIdentity):
        raw: Mapping[str, object] = value.payload()
    elif isinstance(value, Mapping):
        raw = value
    else:
        raise TypeError(f"{field_name} must be a mapping.")
    _require_exact_keys(
        raw,
        _HISTORY_ESTIMATOR_IDENTITY_KEYS,
        field_name=field_name,
    )
    original_history_id = _require_bounded_integer(
        raw["original_history_id"],
        field_name=f"{field_name}.original_history_id",
        maximum=(1 << 64) - 1,
    )
    source_index = _require_bounded_integer(
        raw["source_index"],
        field_name=f"{field_name}.source_index",
        maximum=(1 << 32) - 1,
    )
    line_index = _require_bounded_integer(
        raw["line_index"],
        field_name=f"{field_name}.line_index",
        maximum=(1 << 32) - 1,
    )
    angle_stratum_index = _require_bounded_integer(
        raw["angle_stratum_index"],
        field_name=f"{field_name}.angle_stratum_index",
        maximum=(1 << 32) - 1,
    )
    angle_stratum_count = _require_bounded_integer(
        raw["angle_stratum_count"],
        field_name=f"{field_name}.angle_stratum_count",
        maximum=(1 << 32) - 1,
    )
    if (
        angle_stratum_count == 0
        or angle_stratum_index >= angle_stratum_count
    ):
        raise ValueError(
            f"{field_name} has an invalid angle-stratum identity."
        )
    estimator_coefficient = _require_finite_number(
        raw["estimator_coefficient"],
        field_name=f"{field_name}.estimator_coefficient",
        positive=True,
    )
    return HistoryEstimatorIdentity(
        original_history_id=original_history_id,
        source_index=source_index,
        line_index=line_index,
        angle_stratum_index=angle_stratum_index,
        angle_stratum_count=angle_stratum_count,
        estimator_coefficient=estimator_coefficient,
    )


def _validated_history_estimator_identities(
    value: object,
    *,
    require_canonical_order: bool,
) -> tuple[HistoryEstimatorIdentity, ...]:
    """Validate complete fixed-quota source-line-angle identities."""
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or not value
    ):
        raise TypeError(
            "history_estimator_identities must be a nonempty sequence."
        )
    identities = tuple(
        _parse_history_estimator_identity(
            item,
            field_name=f"history_estimator_identities[{index}]",
        )
        for index, item in enumerate(value)
    )
    history_ids = tuple(
        identity.original_history_id for identity in identities
    )
    if len(set(history_ids)) != len(history_ids):
        raise ValueError("Original history IDs must be unique.")
    if require_canonical_order and history_ids != tuple(sorted(history_ids)):
        raise ValueError(
            "Original history IDs must be in canonical sorted order."
        )

    group_members: dict[
        tuple[int, int, int],
        list[HistoryEstimatorIdentity],
    ] = {}
    stratum_counts: dict[tuple[int, int], int] = {}
    for identity in identities:
        source_line = (identity.source_index, identity.line_index)
        previous_count = stratum_counts.setdefault(
            source_line,
            identity.angle_stratum_count,
        )
        if previous_count != identity.angle_stratum_count:
            raise ValueError(
                "One source-line schedule declares inconsistent angle "
                "stratum counts."
            )
        group_members.setdefault(
            (
                identity.source_index,
                identity.line_index,
                identity.angle_stratum_index,
            ),
            [],
        ).append(identity)

    for source_line, stratum_count in stratum_counts.items():
        group_sizes: list[int] = []
        for stratum_index in range(stratum_count):
            group = group_members.get(
                (source_line[0], source_line[1], stratum_index)
            )
            if group is None:
                raise ValueError(
                    "Every source-line must contain every declared angle "
                    "stratum."
                )
            coefficients = {
                item.estimator_coefficient for item in group
            }
            if len(coefficients) != 1:
                raise ValueError(
                    "One source-line-angle stratum cannot mix external "
                    "estimator coefficients."
                )
            if len(group) < 2:
                raise ValueError(
                    "Exact within-group covariance requires at least two "
                    "original histories per stratum."
                )
            group_sizes.append(len(group))
        if len(set(group_sizes)) != 1:
            raise ValueError(
                "Fixed-quota histories must be equal across angle strata "
                "of one source line."
            )
    return identities


def _history_identity_sha256(
    identities: Sequence[HistoryEstimatorIdentity],
) -> str:
    """Hash canonical full estimator identities including coefficients."""
    payload = {
        "domain": _HISTORY_IDENTITY_DOMAIN,
        "histories": [identity.payload() for identity in identities],
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _group_assignment(
    identities: Sequence[HistoryEstimatorIdentity],
) -> tuple[
    tuple[int, ...],
    tuple["StratumCovarianceDescriptor", ...],
    str,
]:
    """Return canonical group indices, descriptors, and assignment digest."""
    group_keys = sorted(
        {
            (
                identity.source_index,
                identity.line_index,
                identity.angle_stratum_index,
            )
            for identity in identities
        }
    )
    group_index = {
        key: index for index, key in enumerate(group_keys)
    }
    indices = tuple(
        group_index[
            (
                identity.source_index,
                identity.line_index,
                identity.angle_stratum_index,
            )
        ]
        for identity in identities
    )
    descriptors: list[StratumCovarianceDescriptor] = []
    for key in group_keys:
        members = [
            identity
            for identity in identities
            if (
                identity.source_index,
                identity.line_index,
                identity.angle_stratum_index,
            )
            == key
        ]
        descriptors.append(
            StratumCovarianceDescriptor(
                source_index=key[0],
                line_index=key[1],
                angle_stratum_index=key[2],
                angle_stratum_count=members[0].angle_stratum_count,
                history_count=len(members),
                estimator_coefficient=members[0].estimator_coefficient,
            )
        )
    assignment = {
        "domain": _STRATUM_ASSIGNMENT_DOMAIN,
        "histories": [
            {
                "group_index": group,
                "history_id": f"int:{identity.original_history_id}",
            }
            for identity, group in zip(identities, indices, strict=True)
        ],
    }
    digest = hashlib.sha256(_canonical_json_bytes(assignment)).hexdigest()
    return indices, tuple(descriptors), digest


def validate_paired_all64_fidelity(
    value: object,
    *,
    field_name: str = "fidelity",
) -> dict[str, object]:
    """Validate the exact unit-weight full-world replay fidelity contract."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping.")
    expected = _required_fidelity()
    _require_exact_keys(
        value,
        frozenset(expected),
        field_name=field_name,
    )
    actual = dict(value)
    if actual != expected:
        differences = {
            key: {"expected": expected[key], "actual": actual[key]}
            for key in expected
            if actual[key] != expected[key]
        }
        raise ValueError(
            f"{field_name} violates paired replay fidelity: {differences}."
        )
    return actual


def require_dedicated_paired_all64_profile(
    profile: object,
    *,
    standard_runtime: bool,
) -> str:
    """Permit the paired API only after explicit non-standard selection."""
    if standard_runtime:
        raise ValueError(
            "Paired all-64 phase-space replay is not a standard runtime mode."
        )
    if profile != PAIRED_ALL64_PROFILE:
        raise ValueError(
            f"profile must be exactly {PAIRED_ALL64_PROFILE!r}."
        )
    return PAIRED_ALL64_PROFILE


def phase_space_bank_seed(
    *,
    root_seed: int,
    scene_sha256: str,
    source_schedule_sha256: str,
    detector_center_m: Sequence[float],
    dwell_time_s: float,
) -> int:
    """Derive a stable pair-independent upstream bank seed."""
    root = _require_seed(root_seed, field_name="root_seed")
    scene = _require_sha256(scene_sha256, field_name="scene_sha256")
    schedule = _require_sha256(
        source_schedule_sha256,
        field_name="source_schedule_sha256",
    )
    center = _require_center(
        detector_center_m,
        field_name="detector_center_m",
    )
    dwell = _require_finite_number(
        dwell_time_s,
        field_name="dwell_time_s",
        positive=True,
    )
    identity = {
        "domain": _BANK_SEED_DOMAIN,
        "schema_version": PAIRED_ALL64_BANK_SCHEMA_VERSION,
        "profile": PAIRED_ALL64_PROFILE,
        "root_seed": root,
        "scene_sha256": scene,
        "source_schedule_sha256": schedule,
        "detector_center_m": list(center),
        "dwell_time_s": dwell,
    }
    return int.from_bytes(
        hashlib.sha256(_canonical_json_bytes(identity)).digest()[:8],
        byteorder="big",
        signed=False,
    ) & ((1 << 63) - 1)


def phase_space_replay_seed(
    *,
    root_seed: int,
    bank_payload_sha256: str,
    shield_pair_id: int,
) -> int:
    """Derive a stable pair-order-independent downstream replay seed."""
    root = _require_seed(root_seed, field_name="root_seed")
    bank_digest = _require_sha256(
        bank_payload_sha256,
        field_name="bank_payload_sha256",
    )
    pair_id = _require_integer(
        shield_pair_id,
        field_name="shield_pair_id",
    )
    if pair_id not in PAIRED_ALL64_PAIR_IDS:
        raise ValueError("shield_pair_id must belong to the canonical all-64.")
    identity = {
        "domain": _REPLAY_SEED_DOMAIN,
        "schema_version": PAIRED_ALL64_BANK_SCHEMA_VERSION,
        "profile": PAIRED_ALL64_PROFILE,
        "root_seed": root,
        "bank_payload_sha256": bank_digest,
        "shield_pair_id": pair_id,
    }
    return int.from_bytes(
        hashlib.sha256(_canonical_json_bytes(identity)).digest()[:8],
        byteorder="big",
        signed=False,
    ) & ((1 << 63) - 1)


def build_phase_space_bank_metadata(
    *,
    geant4_version: str,
    native_executable_sha256: str,
    scene_sha256: str,
    static_geometry_sha256: str,
    material_contract_sha256: str,
    source_contract_sha256: str,
    source_schedule_sha256: str,
    detector_center_m: Sequence[float],
    boundary_radius_m: float,
    dwell_time_s: float,
    root_seed: int,
    history_estimator_identities: Sequence[
        Mapping[str, object] | HistoryEstimatorIdentity
    ],
    zero_crossing_history_count: int,
    crossing_count: int,
    species_counts: Mapping[str, int],
    bank_payload_sha256: str,
    non_gamma_crossing_count: int = 0,
    nonunit_crossing_weight_count: int = 0,
    fidelity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build and validate metadata for one event-grouped upstream bank."""
    selected_fidelity = (
        _required_fidelity() if fidelity is None else dict(fidelity)
    )
    center = _require_center(
        detector_center_m,
        field_name="detector_center_m",
    )
    dwell = _require_finite_number(
        dwell_time_s,
        field_name="dwell_time_s",
        positive=True,
    )
    root = _require_seed(root_seed, field_name="root_seed")
    identities = _validated_history_estimator_identities(
        history_estimator_identities,
        require_canonical_order=True,
    )
    _, groups, _ = _group_assignment(identities)
    scene_digest = _require_sha256(
        scene_sha256,
        field_name="scene_sha256",
    )
    schedule_digest = _require_sha256(
        source_schedule_sha256,
        field_name="source_schedule_sha256",
    )
    payload: dict[str, object] = {
        "schema_version": PAIRED_ALL64_BANK_SCHEMA_VERSION,
        "profile": PAIRED_ALL64_PROFILE,
        "semantics": PAIRED_ALL64_SEMANTICS,
        "bank_format": PAIRED_ALL64_BANK_FORMAT,
        "endianness": "little",
        "length_unit": "m",
        "energy_unit": "MeV",
        "time_unit": "s",
        "geant4_version": geant4_version,
        "native_executable_sha256": native_executable_sha256,
        "scene_sha256": scene_digest,
        "static_geometry_sha256": static_geometry_sha256,
        "material_contract_sha256": material_contract_sha256,
        "source_contract_sha256": source_contract_sha256,
        "source_schedule_sha256": schedule_digest,
        "detector_center_m": list(center),
        "boundary_radius_m": boundary_radius_m,
        "dwell_time_s": dwell,
        "root_seed": root,
        "bank_seed": phase_space_bank_seed(
            root_seed=root,
            scene_sha256=scene_digest,
            source_schedule_sha256=schedule_digest,
            detector_center_m=center,
            dwell_time_s=dwell,
        ),
        "history_count": len(identities),
        "estimator_group_count": len(groups),
        "history_estimator_identity_sha256": (
            _history_identity_sha256(identities)
        ),
        "estimator_coefficient_semantics": (
            PAIRED_ALL64_ESTIMATOR_COEFFICIENT_SEMANTICS
        ),
        "zero_crossing_history_count": zero_crossing_history_count,
        "crossing_count": crossing_count,
        "non_gamma_crossing_count": non_gamma_crossing_count,
        "nonunit_crossing_weight_count": (
            nonunit_crossing_weight_count
        ),
        "species_counts": dict(species_counts),
        "bank_payload_sha256": bank_payload_sha256,
        "fidelity": selected_fidelity,
    }
    return validate_phase_space_bank_metadata(payload)


def validate_phase_space_bank_metadata(
    value: object,
) -> dict[str, object]:
    """Validate one complete phase-space bank metadata mapping."""
    if not isinstance(value, Mapping):
        raise TypeError("bank metadata must be a mapping.")
    _require_exact_keys(value, _BANK_KEYS, field_name="bank metadata")
    payload = dict(value)
    if payload["schema_version"] != PAIRED_ALL64_BANK_SCHEMA_VERSION:
        raise ValueError("Unsupported phase-space bank schema_version.")
    require_dedicated_paired_all64_profile(
        payload["profile"],
        standard_runtime=False,
    )
    if payload["semantics"] != PAIRED_ALL64_SEMANTICS:
        raise ValueError("Bank semantics are not paired counterfactual means.")
    if payload["bank_format"] != PAIRED_ALL64_BANK_FORMAT:
        raise ValueError("Bank format is not event-grouped detector-boundary.")
    expected_units = {
        "endianness": "little",
        "length_unit": "m",
        "energy_unit": "MeV",
        "time_unit": "s",
    }
    for key, expected in expected_units.items():
        if payload[key] != expected:
            raise ValueError(f"bank metadata {key} must be {expected!r}.")
    _require_nonempty_string(
        payload["geant4_version"],
        field_name="geant4_version",
    )
    for field_name in (
        "native_executable_sha256",
        "scene_sha256",
        "static_geometry_sha256",
        "material_contract_sha256",
        "source_contract_sha256",
        "source_schedule_sha256",
        "bank_payload_sha256",
    ):
        _require_sha256(payload[field_name], field_name=field_name)
    center = _require_center(
        payload["detector_center_m"],
        field_name="detector_center_m",
    )
    _require_finite_number(
        payload["boundary_radius_m"],
        field_name="boundary_radius_m",
        positive=True,
    )
    dwell = _require_finite_number(
        payload["dwell_time_s"],
        field_name="dwell_time_s",
        positive=True,
    )
    root_seed = _require_seed(payload["root_seed"], field_name="root_seed")
    bank_seed = _require_seed(payload["bank_seed"], field_name="bank_seed")
    expected_bank_seed = phase_space_bank_seed(
        root_seed=root_seed,
        scene_sha256=str(payload["scene_sha256"]),
        source_schedule_sha256=str(payload["source_schedule_sha256"]),
        detector_center_m=center,
        dwell_time_s=dwell,
    )
    if bank_seed != expected_bank_seed:
        raise ValueError("bank_seed disagrees with the stable bank identity.")
    history_count = _require_integer(
        payload["history_count"],
        field_name="history_count",
        minimum=1,
    )
    estimator_group_count = _require_integer(
        payload["estimator_group_count"],
        field_name="estimator_group_count",
        minimum=1,
    )
    if estimator_group_count > history_count // 2:
        raise ValueError(
            "Every exact covariance group requires at least two histories."
        )
    _require_sha256(
        payload["history_estimator_identity_sha256"],
        field_name="history_estimator_identity_sha256",
    )
    if (
        payload["estimator_coefficient_semantics"]
        != PAIRED_ALL64_ESTIMATOR_COEFFICIENT_SEMANTICS
    ):
        raise ValueError(
            "Unsupported external estimator coefficient semantics."
        )
    zero_count = _require_integer(
        payload["zero_crossing_history_count"],
        field_name="zero_crossing_history_count",
    )
    crossing_count = _require_integer(
        payload["crossing_count"],
        field_name="crossing_count",
    )
    non_gamma_count = _require_integer(
        payload["non_gamma_crossing_count"],
        field_name="non_gamma_crossing_count",
    )
    nonunit_crossing_weight_count = _require_integer(
        payload["nonunit_crossing_weight_count"],
        field_name="nonunit_crossing_weight_count",
    )
    if zero_count > history_count:
        raise ValueError(
            "zero_crossing_history_count cannot exceed history_count."
        )
    if crossing_count < history_count - zero_count:
        raise ValueError(
            "Every nonempty bank event must contain at least one crossing."
        )
    if nonunit_crossing_weight_count != 0:
        raise ValueError(
            "Exact paired replay rejects weighted Geant4 crossings."
        )
    species_raw = payload["species_counts"]
    if not isinstance(species_raw, Mapping) or not species_raw:
        raise TypeError("species_counts must be a nonempty mapping.")
    species_counts: dict[str, int] = {}
    for species, count in species_raw.items():
        token = _require_nonempty_string(
            species,
            field_name="species_counts key",
        )
        species_counts[token] = _require_integer(
            count,
            field_name=f"species_counts[{token!r}]",
        )
    if sum(species_counts.values()) != crossing_count:
        raise ValueError("species_counts must sum to crossing_count.")
    expected_non_gamma_count = sum(
        count
        for species, count in species_counts.items()
        if species != "gamma"
    )
    if non_gamma_count != expected_non_gamma_count:
        raise ValueError(
            "non_gamma_crossing_count must equal the non-gamma species total."
        )
    validate_paired_all64_fidelity(payload["fidelity"])
    payload["detector_center_m"] = list(center)
    payload["dwell_time_s"] = dwell
    payload["root_seed"] = root_seed
    payload["bank_seed"] = bank_seed
    payload["history_count"] = history_count
    payload["estimator_group_count"] = estimator_group_count
    payload["zero_crossing_history_count"] = zero_count
    payload["crossing_count"] = crossing_count
    payload["non_gamma_crossing_count"] = non_gamma_count
    payload["nonunit_crossing_weight_count"] = (
        nonunit_crossing_weight_count
    )
    payload["species_counts"] = species_counts
    payload["fidelity"] = _required_fidelity()
    return payload


def build_paired_all64_pair_request(
    *,
    bank_metadata: Mapping[str, object],
    shield_pair_id: int,
    detector_center_m: Sequence[float],
    fe_center_m: Sequence[float],
    pb_center_m: Sequence[float],
    root_seed: int,
    fidelity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build one shield-pair request bound to an authenticated bank."""
    bank = validate_phase_space_bank_metadata(bank_metadata)
    pair_id = _require_integer(
        shield_pair_id,
        field_name="shield_pair_id",
    )
    if pair_id not in PAIRED_ALL64_PAIR_IDS:
        raise ValueError("shield_pair_id must belong to the canonical all-64.")
    payload: dict[str, object] = {
        "shield_pair_id": pair_id,
        "fe_orientation_index": pair_id // 8,
        "pb_orientation_index": pair_id % 8,
        "detector_center_m": list(detector_center_m),
        "fe_center_m": list(fe_center_m),
        "pb_center_m": list(pb_center_m),
        "scene_sha256": bank["scene_sha256"],
        "source_contract_sha256": bank["source_contract_sha256"],
        "source_schedule_sha256": bank["source_schedule_sha256"],
        "dwell_time_s": bank["dwell_time_s"],
        "root_seed": root_seed,
        "replay_seed": phase_space_replay_seed(
            root_seed=root_seed,
            bank_payload_sha256=str(bank["bank_payload_sha256"]),
            shield_pair_id=pair_id,
        ),
        "fidelity": (
            _required_fidelity() if fidelity is None else dict(fidelity)
        ),
    }
    return _validate_pair_request(payload, bank=bank)


def _validate_pair_request(
    value: object,
    *,
    bank: Mapping[str, object],
) -> dict[str, object]:
    """Validate one replay request against its upstream bank."""
    if not isinstance(value, Mapping):
        raise TypeError("pair request must be a mapping.")
    _require_exact_keys(value, _PAIR_KEYS, field_name="pair request")
    payload = dict(value)
    pair_id = _require_integer(
        payload["shield_pair_id"],
        field_name="shield_pair_id",
    )
    if pair_id not in PAIRED_ALL64_PAIR_IDS:
        raise ValueError("shield_pair_id must belong to the canonical all-64.")
    if payload["fe_orientation_index"] != pair_id // 8:
        raise ValueError("Fe orientation disagrees with shield_pair_id.")
    if payload["pb_orientation_index"] != pair_id % 8:
        raise ValueError("Pb orientation disagrees with shield_pair_id.")
    detector_center = _require_center(
        payload["detector_center_m"],
        field_name="detector_center_m",
    )
    fe_center = _require_center(
        payload["fe_center_m"],
        field_name="fe_center_m",
    )
    pb_center = _require_center(
        payload["pb_center_m"],
        field_name="pb_center_m",
    )
    bank_center = _require_center(
        bank["detector_center_m"],
        field_name="bank.detector_center_m",
    )
    if not (
        detector_center == bank_center
        and fe_center == bank_center
        and pb_center == bank_center
    ):
        raise ValueError(
            "Detector, Fe, Pb, and bank boundary centers must be identical."
        )
    for field_name in (
        "scene_sha256",
        "source_contract_sha256",
        "source_schedule_sha256",
    ):
        digest = _require_sha256(payload[field_name], field_name=field_name)
        if digest != bank[field_name]:
            raise ValueError(f"{field_name} differs from the bank.")
    dwell = _require_finite_number(
        payload["dwell_time_s"],
        field_name="dwell_time_s",
        positive=True,
    )
    if dwell != bank["dwell_time_s"]:
        raise ValueError("dwell_time_s differs from the bank.")
    root_seed = _require_seed(payload["root_seed"], field_name="root_seed")
    if root_seed != bank["root_seed"]:
        raise ValueError("root_seed differs from the bank.")
    replay_seed = _require_seed(
        payload["replay_seed"],
        field_name="replay_seed",
    )
    expected_seed = phase_space_replay_seed(
        root_seed=root_seed,
        bank_payload_sha256=str(bank["bank_payload_sha256"]),
        shield_pair_id=pair_id,
    )
    if replay_seed != expected_seed:
        raise ValueError("replay_seed disagrees with pair identity.")
    validate_paired_all64_fidelity(payload["fidelity"])
    payload["shield_pair_id"] = pair_id
    payload["detector_center_m"] = list(detector_center)
    payload["fe_center_m"] = list(fe_center)
    payload["pb_center_m"] = list(pb_center)
    payload["dwell_time_s"] = dwell
    payload["root_seed"] = root_seed
    payload["replay_seed"] = replay_seed
    payload["fidelity"] = _required_fidelity()
    return payload


def _array_digest_bytes(value: NDArray[np.float64]) -> bytes:
    """Return canonical little-endian C-order bytes for one float64 array."""
    return np.asarray(value, dtype="<f8", order="C").tobytes(order="C")


@dataclass(frozen=True)
class StratumCovarianceDescriptor:
    """Describe one source-line-angle fixed-quota covariance group."""

    source_index: int
    line_index: int
    angle_stratum_index: int
    angle_stratum_count: int
    history_count: int
    estimator_coefficient: float

    def payload(self) -> dict[str, object]:
        """Return the canonical JSON-compatible group descriptor."""
        return {
            "source_index": self.source_index,
            "line_index": self.line_index,
            "angle_stratum_index": self.angle_stratum_index,
            "angle_stratum_count": self.angle_stratum_count,
            "history_count": self.history_count,
            "estimator_coefficient": self.estimator_coefficient,
        }


@dataclass(frozen=True)
class CrossPairStratifiedCovariance:
    """Exact original-history covariance for fixed-quota strata."""

    pair_ids: tuple[int, ...]
    history_count: int
    group_count: int
    feature_count: int
    score_semantics: str
    history_estimator_identity_sha256: str
    group_assignment_sha256: str
    original_history_ids: NDArray[np.uint64]
    history_group_indices: NDArray[np.uint32]
    groups: tuple[StratumCovarianceDescriptor, ...]
    estimate_by_pair_feature: NDArray[np.float64]
    first_sum_by_group_pair_feature: NDArray[np.float64]
    centered_factor_by_history: NDArray[np.float64]
    total_cross_pair_covariance: NDArray[np.float64]
    artifact_sha256: str

    def metadata(self) -> dict[str, object]:
        """Return strict JSON metadata referencing the numeric artifact."""
        return {
            "schema_version": PAIRED_ALL64_COVARIANCE_SCHEMA_VERSION,
            "semantics": _COVARIANCE_SEMANTICS,
            "pair_ids": list(self.pair_ids),
            "history_count": self.history_count,
            "group_count": self.group_count,
            "feature_count": self.feature_count,
            "score_semantics": self.score_semantics,
            "history_estimator_identity_sha256": (
                self.history_estimator_identity_sha256
            ),
            "group_assignment_sha256": self.group_assignment_sha256,
            "estimator_coefficient_semantics": (
                PAIRED_ALL64_ESTIMATOR_COEFFICIENT_SEMANTICS
            ),
            "groups": [group.payload() for group in self.groups],
            "original_history_ids_shape": list(
                self.original_history_ids.shape
            ),
            "history_group_indices_shape": list(
                self.history_group_indices.shape
            ),
            "estimate_shape": list(self.estimate_by_pair_feature.shape),
            "first_sum_shape": list(
                self.first_sum_by_group_pair_feature.shape
            ),
            "centered_factor_shape": list(
                self.centered_factor_by_history.shape
            ),
            "total_cross_pair_covariance_shape": list(
                self.total_cross_pair_covariance.shape
            ),
            "artifact_sha256": self.artifact_sha256,
        }


def _covariance_header(
    *,
    history_count: int,
    group_count: int,
    feature_count: int,
    score_semantics: str,
    group_assignment_sha256: str,
) -> dict[str, object]:
    """Return the exact header hashed by the native v2 artifact format."""
    return {
        "centered_factor_shape": [history_count, 64, feature_count],
        "estimate_shape": [64, feature_count],
        "feature_count": feature_count,
        "first_sum_shape": [group_count, 64, feature_count],
        "group_assignment_sha256": group_assignment_sha256,
        "group_count": group_count,
        "history_count": history_count,
        "pair_ids": list(PAIRED_ALL64_PAIR_IDS),
        "schema_version": PAIRED_ALL64_COVARIANCE_SCHEMA_VERSION,
        "score_semantics": score_semantics,
        "semantics": _COVARIANCE_SEMANTICS,
        "total_cross_pair_covariance_shape": [64, 64],
    }


def _covariance_artifact_sha256(
    *,
    header: Mapping[str, object],
    original_history_ids: NDArray[np.uint64],
    history_group_indices: NDArray[np.uint32],
    groups: Sequence[StratumCovarianceDescriptor],
    estimate: NDArray[np.float64],
    first_sum: NDArray[np.float64],
    centered_factor: NDArray[np.float64],
    total_covariance: NDArray[np.float64],
) -> str:
    """Hash the same canonical fields as the native covariance artifact."""
    hasher = hashlib.sha256()
    hasher.update(_canonical_json_bytes(dict(header)))
    hasher.update(
        np.asarray(
            original_history_ids,
            dtype="<u8",
            order="C",
        ).tobytes(order="C")
    )
    hasher.update(
        np.asarray(
            history_group_indices,
            dtype="<u4",
            order="C",
        ).tobytes(order="C")
    )
    for group in groups:
        hasher.update(
            struct.pack(
                "<IIIIQd",
                group.source_index,
                group.line_index,
                group.angle_stratum_index,
                group.angle_stratum_count,
                group.history_count,
                group.estimator_coefficient,
            )
        )
    for array in (estimate, first_sum, centered_factor, total_covariance):
        hasher.update(_array_digest_bytes(array))
    return hasher.hexdigest()


def _deterministic_total_cross_pair_covariance(
    centered_factor: NDArray[np.float64],
    *,
    history_batch_size: int = 256,
) -> NDArray[np.float64]:
    """Accumulate pair totals in native history order using bounded batches."""
    if centered_factor.ndim != 3 or centered_factor.shape[1] != 64:
        raise ValueError(
            "centered_factor must have shape (history, 64, feature)."
        )
    batch_size = _require_integer(
        history_batch_size,
        field_name="history_batch_size",
        minimum=1,
    )
    total_covariance = np.zeros((64, 64), dtype=np.float64)
    for start in range(0, centered_factor.shape[0], batch_size):
        stop = min(start + batch_size, centered_factor.shape[0])
        feature_prefix = np.add.accumulate(
            centered_factor[start:stop],
            axis=2,
        )
        pair_totals = feature_prefix[:, :, -1]
        products = (
            pair_totals[:, :, None] * pair_totals[:, None, :]
        )
        products[0] += total_covariance
        np.add.accumulate(products, axis=0, out=products)
        total_covariance[...] = products[-1]
    return total_covariance


def aggregate_cross_pair_stratified_covariance(
    *,
    history_estimator_identities: Sequence[
        Mapping[str, object] | HistoryEstimatorIdentity
    ],
    scores_by_history_pair_feature: NDArray[np.float64],
    score_semantics: str,
) -> CrossPairStratifiedCovariance:
    """Compute exact within-stratum original-history covariance factors."""
    semantics = _require_nonempty_string(
        score_semantics,
        field_name="score_semantics",
    )
    if not semantics.isascii():
        raise ValueError(
            "score_semantics must be ASCII for native artifact hashing."
        )
    identities = _validated_history_estimator_identities(
        history_estimator_identities,
        require_canonical_order=False,
    )
    scores = np.asarray(scores_by_history_pair_feature, dtype=np.float64)
    if scores.ndim != 3:
        raise ValueError(
            "scores_by_history_pair_feature must have shape "
            "(history, 64, feature)."
        )
    history_count, pair_count, feature_count = scores.shape
    if history_count != len(identities):
        raise ValueError(
            "History estimator identities differ from the score history axis."
        )
    if pair_count != len(PAIRED_ALL64_PAIR_IDS):
        raise ValueError("The score pair axis must contain exactly all 64.")
    if feature_count <= 0:
        raise ValueError("The score feature axis must be nonempty.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Cross-pair scores must all be finite.")

    permutation = np.argsort(
        np.asarray(
            [identity.original_history_id for identity in identities],
            dtype=np.uint64,
        ),
        kind="stable",
    )
    canonical_identities = tuple(identities[index] for index in permutation)
    canonical_identities = _validated_history_estimator_identities(
        canonical_identities,
        require_canonical_order=True,
    )
    canonical_scores = scores[permutation]
    group_indices, groups, assignment_sha256 = _group_assignment(
        canonical_identities
    )
    group_count = len(groups)
    first_sum = np.zeros(
        (group_count, pair_count, feature_count),
        dtype=np.float64,
    )
    estimate = np.zeros((pair_count, feature_count), dtype=np.float64)
    centered_factor = np.zeros_like(canonical_scores)
    group_index_array = np.asarray(group_indices, dtype=np.uint32)
    for group_index, descriptor in enumerate(groups):
        selected_indices = np.flatnonzero(
            group_index_array == group_index
        )
        if selected_indices.size != descriptor.history_count:
            raise RuntimeError(
                "Internal history-to-stratum assignment is inconsistent."
            )
        selected = canonical_scores[selected_indices]
        group_sum = np.sum(selected, axis=0)
        first_sum[group_index] = group_sum
        estimate += descriptor.estimator_coefficient * group_sum
        group_mean = group_sum / float(descriptor.history_count)
        scale = descriptor.estimator_coefficient * math.sqrt(
            descriptor.history_count / (descriptor.history_count - 1.0)
        )
        centered_factor[selected_indices] = (
            scale * (selected - group_mean[None, :, :])
        )

    total_covariance = _deterministic_total_cross_pair_covariance(
        centered_factor
    )
    original_history_ids = np.asarray(
        [
            identity.original_history_id
            for identity in canonical_identities
        ],
        dtype=np.uint64,
    )
    identity_sha256 = _history_identity_sha256(canonical_identities)
    header = _covariance_header(
        history_count=history_count,
        group_count=group_count,
        feature_count=feature_count,
        score_semantics=semantics,
        group_assignment_sha256=assignment_sha256,
    )
    artifact_sha256 = _covariance_artifact_sha256(
        header=header,
        original_history_ids=original_history_ids,
        history_group_indices=group_index_array,
        groups=groups,
        estimate=estimate,
        first_sum=first_sum,
        centered_factor=centered_factor,
        total_covariance=total_covariance,
    )
    for array in (
        original_history_ids,
        group_index_array,
        estimate,
        first_sum,
        centered_factor,
        total_covariance,
    ):
        array.setflags(write=False)
    return CrossPairStratifiedCovariance(
        pair_ids=PAIRED_ALL64_PAIR_IDS,
        history_count=history_count,
        group_count=group_count,
        feature_count=feature_count,
        score_semantics=semantics,
        history_estimator_identity_sha256=identity_sha256,
        group_assignment_sha256=assignment_sha256,
        original_history_ids=original_history_ids,
        history_group_indices=group_index_array,
        groups=groups,
        estimate_by_pair_feature=estimate,
        first_sum_by_group_pair_feature=first_sum,
        centered_factor_by_history=centered_factor,
        total_cross_pair_covariance=total_covariance,
        artifact_sha256=artifact_sha256,
    )


def validate_cross_pair_covariance_metadata(
    value: object,
) -> dict[str, object]:
    """Validate metadata for one paired cross-pair covariance artifact."""
    if not isinstance(value, Mapping):
        raise TypeError("cross-pair covariance metadata must be a mapping.")
    _require_exact_keys(
        value,
        _COVARIANCE_KEYS,
        field_name="cross-pair covariance metadata",
    )
    payload = dict(value)
    if payload["schema_version"] != PAIRED_ALL64_COVARIANCE_SCHEMA_VERSION:
        raise ValueError("Unsupported cross-pair covariance schema_version.")
    if payload["semantics"] != _COVARIANCE_SEMANTICS:
        raise ValueError("Unsupported cross-pair covariance semantics.")
    if payload["pair_ids"] != list(PAIRED_ALL64_PAIR_IDS):
        raise ValueError("Covariance pair_ids must be the canonical all-64.")
    history_count = _require_integer(
        payload["history_count"],
        field_name="history_count",
        minimum=2,
    )
    group_count = _require_integer(
        payload["group_count"],
        field_name="group_count",
        minimum=1,
    )
    if group_count > history_count // 2:
        raise ValueError(
            "Every exact covariance group requires at least two histories."
        )
    feature_count = _require_integer(
        payload["feature_count"],
        field_name="feature_count",
        minimum=1,
    )
    score_semantics = _require_nonempty_string(
        payload["score_semantics"],
        field_name="score_semantics",
    )
    if not score_semantics.isascii():
        raise ValueError("score_semantics must be ASCII.")
    _require_sha256(
        payload["history_estimator_identity_sha256"],
        field_name="history_estimator_identity_sha256",
    )
    _require_sha256(
        payload["group_assignment_sha256"],
        field_name="group_assignment_sha256",
    )
    if (
        payload["estimator_coefficient_semantics"]
        != PAIRED_ALL64_ESTIMATOR_COEFFICIENT_SEMANTICS
    ):
        raise ValueError(
            "Unsupported external estimator coefficient semantics."
        )
    groups_raw = payload["groups"]
    if (
        isinstance(groups_raw, (str, bytes))
        or not isinstance(groups_raw, Sequence)
        or len(groups_raw) != group_count
    ):
        raise ValueError("groups must contain exactly group_count entries.")
    groups: list[dict[str, object]] = []
    prior_key: tuple[int, int, int] | None = None
    group_history_total = 0
    for index, raw_group in enumerate(groups_raw):
        if not isinstance(raw_group, Mapping):
            raise TypeError(f"groups[{index}] must be a mapping.")
        _require_exact_keys(
            raw_group,
            _COVARIANCE_GROUP_KEYS,
            field_name=f"groups[{index}]",
        )
        identity = _parse_history_estimator_identity(
            {
                "original_history_id": 0,
                "source_index": raw_group["source_index"],
                "line_index": raw_group["line_index"],
                "angle_stratum_index": raw_group[
                    "angle_stratum_index"
                ],
                "angle_stratum_count": raw_group[
                    "angle_stratum_count"
                ],
                "estimator_coefficient": raw_group[
                    "estimator_coefficient"
                ],
            },
            field_name=f"groups[{index}]",
        )
        group_history_count = _require_integer(
            raw_group["history_count"],
            field_name=f"groups[{index}].history_count",
            minimum=2,
        )
        key = (
            identity.source_index,
            identity.line_index,
            identity.angle_stratum_index,
        )
        if prior_key is not None and key <= prior_key:
            raise ValueError("Covariance groups must be canonically ordered.")
        prior_key = key
        group_history_total += group_history_count
        groups.append(
            {
                "source_index": identity.source_index,
                "line_index": identity.line_index,
                "angle_stratum_index": identity.angle_stratum_index,
                "angle_stratum_count": identity.angle_stratum_count,
                "history_count": group_history_count,
                "estimator_coefficient": (
                    identity.estimator_coefficient
                ),
            }
        )
    if group_history_total != history_count:
        raise ValueError("Covariance group histories must cover all histories.")
    source_line_groups: dict[
        tuple[int, int],
        list[dict[str, object]],
    ] = {}
    for group in groups:
        source_line_groups.setdefault(
            (int(group["source_index"]), int(group["line_index"])),
            [],
        ).append(group)
    for source_line, descriptors in source_line_groups.items():
        declared_counts = {
            int(group["angle_stratum_count"]) for group in descriptors
        }
        if len(declared_counts) != 1:
            raise ValueError(
                "One covariance source-line has inconsistent stratum counts."
            )
        declared_count = next(iter(declared_counts))
        observed_indices = {
            int(group["angle_stratum_index"]) for group in descriptors
        }
        if observed_indices != set(range(declared_count)):
            raise ValueError(
                "Covariance groups must contain every declared stratum for "
                f"source-line {source_line}."
            )
        quotas = {int(group["history_count"]) for group in descriptors}
        if len(quotas) != 1:
            raise ValueError(
                "Covariance fixed quota must be equal across source-line "
                "angle strata."
            )
    expected_shapes = {
        "original_history_ids_shape": [history_count],
        "history_group_indices_shape": [history_count],
        "estimate_shape": [64, feature_count],
        "first_sum_shape": [group_count, 64, feature_count],
        "centered_factor_shape": [history_count, 64, feature_count],
        "total_cross_pair_covariance_shape": [64, 64],
    }
    for field_name, expected in expected_shapes.items():
        if payload[field_name] != expected:
            raise ValueError(f"{field_name} must be exactly {expected}.")
    _require_sha256(
        payload["artifact_sha256"],
        field_name="artifact_sha256",
    )
    payload["history_count"] = history_count
    payload["group_count"] = group_count
    payload["feature_count"] = feature_count
    payload["score_semantics"] = score_semantics
    payload["groups"] = groups
    return payload


def build_paired_all64_manifest(
    *,
    bank_metadata: Mapping[str, object],
    pair_requests: Sequence[Mapping[str, object]],
    covariance_metadata: Mapping[str, object],
) -> dict[str, object]:
    """Build a complete fail-closed paired all-64 replay manifest."""
    bank = validate_phase_space_bank_metadata(bank_metadata)
    if (
        isinstance(pair_requests, (str, bytes))
        or not isinstance(pair_requests, Sequence)
    ):
        raise TypeError("pair_requests must be a sequence.")
    if len(pair_requests) != len(PAIRED_ALL64_PAIR_IDS):
        raise ValueError("A paired manifest requires exactly 64 pair requests.")
    pairs = [
        _validate_pair_request(pair, bank=bank)
        for pair in pair_requests
    ]
    pair_ids = [int(pair["shield_pair_id"]) for pair in pairs]
    if len(set(pair_ids)) != 64 or set(pair_ids) != set(
        PAIRED_ALL64_PAIR_IDS
    ):
        raise ValueError("Pair requests must contain each canonical pair once.")
    pairs.sort(key=lambda pair: int(pair["shield_pair_id"]))
    covariance = validate_cross_pair_covariance_metadata(
        covariance_metadata
    )
    if covariance["history_count"] != bank["history_count"]:
        raise ValueError(
            "Covariance history_count differs from the phase-space bank."
        )
    if covariance["group_count"] != bank["estimator_group_count"]:
        raise ValueError(
            "Covariance group_count differs from the phase-space bank."
        )
    if (
        covariance["history_estimator_identity_sha256"]
        != bank["history_estimator_identity_sha256"]
    ):
        raise ValueError(
            "Covariance history estimator identities differ from the bank."
        )
    return {
        "schema_version": PAIRED_ALL64_MANIFEST_SCHEMA_VERSION,
        "profile": PAIRED_ALL64_PROFILE,
        "semantics": PAIRED_ALL64_SEMANTICS,
        "standard_runtime_selectable": False,
        "complete": True,
        "bank": bank,
        "pairs": pairs,
        "cross_pair_stratified_covariance": covariance,
    }


def validate_paired_all64_manifest(value: object) -> dict[str, object]:
    """Validate and canonicalize one complete paired all-64 manifest."""
    if not isinstance(value, Mapping):
        raise TypeError("paired all-64 manifest must be a mapping.")
    _require_exact_keys(value, _MANIFEST_KEYS, field_name="paired manifest")
    payload = dict(value)
    if payload["schema_version"] != PAIRED_ALL64_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported paired all-64 manifest schema_version.")
    require_dedicated_paired_all64_profile(
        payload["profile"],
        standard_runtime=bool(payload["standard_runtime_selectable"]),
    )
    if payload["standard_runtime_selectable"] is not False:
        raise ValueError("Paired replay must not be standard-runtime selectable.")
    if payload["complete"] is not True:
        raise ValueError("Incomplete paired replay manifests are not runnable.")
    if payload["semantics"] != PAIRED_ALL64_SEMANTICS:
        raise ValueError("Manifest semantics are not paired counterfactual.")
    rebuilt = build_paired_all64_manifest(
        bank_metadata=payload["bank"],
        pair_requests=payload["pairs"],
        covariance_metadata=payload["cross_pair_stratified_covariance"],
    )
    if _canonical_json_bytes(rebuilt) != _canonical_json_bytes(payload):
        raise ValueError("Paired manifest is not in canonical pair order.")
    return rebuilt


__all__ = [
    "CrossPairStratifiedCovariance",
    "HistoryEstimatorIdentity",
    "PAIRED_ALL64_BANK_FORMAT",
    "PAIRED_ALL64_BANK_SCHEMA_VERSION",
    "PAIRED_ALL64_COVARIANCE_SCHEMA_VERSION",
    "PAIRED_ALL64_ESTIMATOR_COEFFICIENT_SEMANTICS",
    "PAIRED_ALL64_MANIFEST_SCHEMA_VERSION",
    "PAIRED_ALL64_PAIR_IDS",
    "PAIRED_ALL64_PROFILE",
    "PAIRED_ALL64_REQUIRED_FIDELITY",
    "PAIRED_ALL64_SEMANTICS",
    "PAIRED_ALL64_STANDARD_RUNTIME_SELECTABLE",
    "StratumCovarianceDescriptor",
    "aggregate_cross_pair_stratified_covariance",
    "build_paired_all64_manifest",
    "build_paired_all64_pair_request",
    "build_phase_space_bank_metadata",
    "phase_space_bank_seed",
    "phase_space_replay_seed",
    "require_dedicated_paired_all64_profile",
    "validate_cross_pair_covariance_metadata",
    "validate_paired_all64_fidelity",
    "validate_paired_all64_manifest",
    "validate_phase_space_bank_metadata",
]
