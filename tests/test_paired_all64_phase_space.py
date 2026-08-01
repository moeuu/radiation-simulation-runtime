"""Tests for fail-closed paired all-64 phase-space v3 contracts."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from spectrum.paired_all64_phase_space import (
    HistoryEstimatorIdentity,
    PAIRED_ALL64_BANK_FORMAT,
    PAIRED_ALL64_BANK_SCHEMA_VERSION,
    PAIRED_ALL64_ESTIMATOR_COEFFICIENT_SEMANTICS,
    PAIRED_ALL64_PAIR_IDS,
    PAIRED_ALL64_PROFILE,
    PAIRED_ALL64_REQUIRED_FIDELITY,
    PAIRED_ALL64_STANDARD_RUNTIME_SELECTABLE,
    aggregate_cross_pair_stratified_covariance,
    build_paired_all64_manifest,
    build_paired_all64_pair_request,
    build_phase_space_bank_metadata,
    phase_space_bank_seed,
    phase_space_replay_seed,
    require_dedicated_paired_all64_profile,
    validate_cross_pair_covariance_metadata,
    validate_paired_all64_manifest,
    validate_phase_space_bank_metadata,
)


def _identities(
    *,
    history_count: int = 8,
    coefficient: float = 0.25,
) -> list[HistoryEstimatorIdentity]:
    """Return two complete equal-quota angle strata for one source line."""
    if history_count < 4 or history_count % 2 != 0:
        raise ValueError("history_count must be an even integer of at least 4.")
    quota = history_count // 2
    return [
        HistoryEstimatorIdentity(
            original_history_id=index,
            source_index=0,
            line_index=0,
            angle_stratum_index=0 if index < quota else 1,
            angle_stratum_count=2,
            estimator_coefficient=coefficient,
        )
        for index in range(history_count)
    ]


def _bank(*, history_count: int = 8) -> dict[str, object]:
    """Return one valid v3 event-grouped gamma-only phase-space bank."""
    return build_phase_space_bank_metadata(
        geant4_version="11.3.2",
        native_executable_sha256="a" * 64,
        scene_sha256="b" * 64,
        static_geometry_sha256="c" * 64,
        material_contract_sha256="d" * 64,
        source_contract_sha256="e" * 64,
        source_schedule_sha256="f" * 64,
        detector_center_m=(1.0, 2.0, 3.0),
        boundary_radius_m=0.5,
        dwell_time_s=30.0,
        root_seed=1729,
        history_estimator_identities=_identities(
            history_count=history_count
        ),
        zero_crossing_history_count=2,
        crossing_count=history_count,
        species_counts={"gamma": history_count},
        bank_payload_sha256="1" * 64,
    )


def _pairs(bank: dict[str, object]) -> list[dict[str, object]]:
    """Return the canonical 64 requests for one upstream bank."""
    center = tuple(bank["detector_center_m"])
    return [
        build_paired_all64_pair_request(
            bank_metadata=bank,
            shield_pair_id=pair_id,
            detector_center_m=center,
            fe_center_m=center,
            pb_center_m=center,
            root_seed=int(bank["root_seed"]),
        )
        for pair_id in PAIRED_ALL64_PAIR_IDS
    ]


def _covariance(
    *,
    history_count: int = 8,
    feature_count: int = 2,
) -> object:
    """Return one deterministic exact stratified covariance aggregate."""
    scores = np.arange(
        history_count * 64 * feature_count,
        dtype=np.float64,
    ).reshape(history_count, 64, feature_count)
    return aggregate_cross_pair_stratified_covariance(
        history_estimator_identities=_identities(
            history_count=history_count
        ),
        scores_by_history_pair_feature=scores,
        score_semantics="incident_gamma_bin_count_per_original_history",
    )


def _manifest() -> dict[str, object]:
    """Return one complete canonical v3 paired replay manifest."""
    bank = _bank()
    return build_paired_all64_manifest(
        bank_metadata=bank,
        pair_requests=_pairs(bank),
        covariance_metadata=_covariance().metadata(),
    )


def test_v3_tokens_and_external_coefficient_contract_are_exact() -> None:
    """Python metadata must identify only the native v3 bank semantics."""
    bank = _bank()

    assert PAIRED_ALL64_BANK_SCHEMA_VERSION == 3
    assert PAIRED_ALL64_PROFILE == "geant4_phase_space_paired_all64_v3"
    assert PAIRED_ALL64_BANK_FORMAT == "event_grouped_detector_boundary_v3"
    assert bank["schema_version"] == 3
    assert bank["estimator_group_count"] == 2
    assert (
        bank["estimator_coefficient_semantics"]
        == PAIRED_ALL64_ESTIMATOR_COEFFICIENT_SEMANTICS
    )
    assert bank["nonunit_crossing_weight_count"] == 0


def test_paired_seed_identity_is_stable_and_pair_order_independent() -> None:
    """Bank and replay seeds must depend only on authenticated identities."""
    bank_seed_a = phase_space_bank_seed(
        root_seed=7,
        scene_sha256="a" * 64,
        source_schedule_sha256="b" * 64,
        detector_center_m=(1.0, 2.0, 3.0),
        dwell_time_s=30.0,
    )
    bank_seed_b = phase_space_bank_seed(
        root_seed=7,
        scene_sha256="a" * 64,
        source_schedule_sha256="b" * 64,
        detector_center_m=[1, 2, 3],
        dwell_time_s=30,
    )
    seeds_forward = {
        pair_id: phase_space_replay_seed(
            root_seed=7,
            bank_payload_sha256="c" * 64,
            shield_pair_id=pair_id,
        )
        for pair_id in PAIRED_ALL64_PAIR_IDS
    }
    seeds_reverse = {
        pair_id: phase_space_replay_seed(
            root_seed=7,
            bank_payload_sha256="c" * 64,
            shield_pair_id=pair_id,
        )
        for pair_id in reversed(PAIRED_ALL64_PAIR_IDS)
    }

    assert bank_seed_a == bank_seed_b
    assert seeds_forward == seeds_reverse
    assert len(set(seeds_forward.values())) == 64


def test_manifest_accepts_exact_all64_and_canonicalizes_pair_order() -> None:
    """The dedicated schema must accept all and only the canonical 64 pairs."""
    bank = _bank()
    manifest = build_paired_all64_manifest(
        bank_metadata=bank,
        pair_requests=list(reversed(_pairs(bank))),
        covariance_metadata=_covariance().metadata(),
    )

    assert [pair["shield_pair_id"] for pair in manifest["pairs"]] == list(
        PAIRED_ALL64_PAIR_IDS
    )
    assert manifest["standard_runtime_selectable"] is False
    assert "cross_pair_stratified_covariance" in manifest
    assert validate_paired_all64_manifest(manifest) == manifest


def test_manifest_rejects_missing_or_duplicate_pairs() -> None:
    """A repeated pair must never masquerade as complete all-64 coverage."""
    bank = _bank()
    pairs = _pairs(bank)
    pairs[-1] = copy.deepcopy(pairs[0])

    with pytest.raises(ValueError, match="each canonical pair once"):
        build_paired_all64_manifest(
            bank_metadata=bank,
            pair_requests=pairs,
            covariance_metadata=_covariance().metadata(),
        )


@pytest.mark.parametrize(
    ("field_name", "replacement", "message"),
    (
        ("detector_center_m", [1.0, 2.0, 3.0001], "centers"),
        ("fe_center_m", [1.0, 2.0, 3.0001], "centers"),
        ("pb_center_m", [1.0, 2.0, 3.0001], "centers"),
        ("source_contract_sha256", "9" * 64, "source_contract"),
        ("source_schedule_sha256", "8" * 64, "source_schedule"),
        ("dwell_time_s", 29.0, "dwell_time"),
    ),
)
def test_manifest_rejects_pair_contract_mismatch(
    field_name: str,
    replacement: object,
    message: str,
) -> None:
    """Every pair must replay exactly the same upstream experiment."""
    bank = _bank()
    pairs = _pairs(bank)
    pairs[7][field_name] = replacement

    with pytest.raises(ValueError, match=message):
        build_paired_all64_manifest(
            bank_metadata=bank,
            pair_requests=pairs,
            covariance_metadata=_covariance().metadata(),
        )


def test_manifest_rejects_changed_full_transport_fidelity() -> None:
    """A lower-fidelity pair may not enter an exact paired artifact."""
    bank = _bank()
    pairs = _pairs(bank)
    fidelity = dict(PAIRED_ALL64_REQUIRED_FIDELITY)
    fidelity["secondary_transport_mode"] = "gamma_only"
    pairs[12]["fidelity"] = fidelity

    with pytest.raises(ValueError, match="violates paired replay fidelity"):
        build_paired_all64_manifest(
            bank_metadata=bank,
            pair_requests=pairs,
            covariance_metadata=_covariance().metadata(),
        )


def test_bank_preserves_species_but_rejects_weighted_crossings() -> None:
    """All species are valid, while any weighted crossing fails closed."""
    common = {
        "geant4_version": "11.3.2",
        "native_executable_sha256": "a" * 64,
        "scene_sha256": "b" * 64,
        "static_geometry_sha256": "c" * 64,
        "material_contract_sha256": "d" * 64,
        "source_contract_sha256": "e" * 64,
        "source_schedule_sha256": "f" * 64,
        "detector_center_m": (1.0, 2.0, 3.0),
        "boundary_radius_m": 0.5,
        "dwell_time_s": 30.0,
        "root_seed": 1729,
        "history_estimator_identities": _identities(),
        "zero_crossing_history_count": 2,
        "crossing_count": 8,
        "species_counts": {"gamma": 7, "e-": 1},
        "bank_payload_sha256": "1" * 64,
    }
    bank = build_phase_space_bank_metadata(
        **common,
        non_gamma_crossing_count=1,
    )
    assert bank["species_counts"] == {"gamma": 7, "e-": 1}
    assert bank["non_gamma_crossing_count"] == 1

    common["species_counts"] = {"gamma": 8}
    with pytest.raises(ValueError, match="weighted Geant4 crossings"):
        build_phase_space_bank_metadata(
            **common,
            nonunit_crossing_weight_count=1,
        )


def test_manifest_rejects_forged_replay_seed() -> None:
    """A pair result must remain independent of pair iteration order."""
    bank = _bank()
    pairs = _pairs(bank)
    pairs[22]["replay_seed"] = int(pairs[22]["replay_seed"]) + 1

    with pytest.raises(ValueError, match="replay_seed"):
        build_paired_all64_manifest(
            bank_metadata=bank,
            pair_requests=pairs,
            covariance_metadata=_covariance().metadata(),
        )


def test_exact_factor_matches_within_group_original_history_covariance() -> None:
    """Factor rows must reproduce the exact stratified estimator covariance."""
    rng = np.random.default_rng(20260729)
    identities = _identities(history_count=16, coefficient=0.125)
    scores = rng.normal(size=(16, 64, 3))
    aggregate = aggregate_cross_pair_stratified_covariance(
        history_estimator_identities=identities,
        scores_by_history_pair_feature=scores,
        score_semantics="test_score",
    )
    recovered = (
        aggregate.centered_factor_by_history.reshape(16, -1).T
        @ aggregate.centered_factor_by_history.reshape(16, -1)
    )

    expected = np.zeros_like(recovered)
    expected_estimate = np.zeros((64, 3), dtype=np.float64)
    for stratum_index in (0, 1):
        indices = [
            index
            for index, identity in enumerate(identities)
            if identity.angle_stratum_index == stratum_index
        ]
        group = scores[indices].reshape(len(indices), -1)
        centered = group - np.mean(group, axis=0, keepdims=True)
        coefficient = identities[indices[0]].estimator_coefficient
        expected += (
            coefficient**2
            * len(indices)
            / (len(indices) - 1)
            * (centered.T @ centered)
        )
        expected_estimate += coefficient * np.sum(scores[indices], axis=0)

    np.testing.assert_allclose(recovered, expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        aggregate.estimate_by_pair_feature,
        expected_estimate,
        rtol=1e-12,
        atol=1e-14,
    )
    total_factor = aggregate.centered_factor_by_history.sum(axis=2)
    np.testing.assert_allclose(
        aggregate.total_cross_pair_covariance,
        total_factor.T @ total_factor,
        rtol=1e-12,
        atol=1e-14,
    )


def test_exact_covariance_never_pools_between_group_mean_difference() -> None:
    """Different source-line means must not create fake sampling variance."""
    identities = [
        HistoryEstimatorIdentity(index, 0, 0, 0, 1, 0.5)
        for index in range(2)
    ] + [
        HistoryEstimatorIdentity(index, 1, 0, 0, 1, 0.25)
        for index in range(2, 4)
    ]
    scores = np.zeros((4, 64, 1), dtype=np.float64)
    scores[2:] = 100.0

    aggregate = aggregate_cross_pair_stratified_covariance(
        history_estimator_identities=identities,
        scores_by_history_pair_feature=scores,
        score_semantics="constant_within_group",
    )

    np.testing.assert_array_equal(
        aggregate.centered_factor_by_history,
        np.zeros_like(scores),
    )
    np.testing.assert_array_equal(
        aggregate.total_cross_pair_covariance,
        np.zeros((64, 64), dtype=np.float64),
    )
    np.testing.assert_array_equal(
        aggregate.estimate_by_pair_feature,
        np.full((64, 1), 50.0, dtype=np.float64),
    )


def test_exact_artifact_is_input_order_invariant() -> None:
    """Joint identity/score reordering must preserve the exact artifact."""
    rng = np.random.default_rng(17)
    scores = rng.normal(size=(8, 64, 2))
    identities = _identities()
    first = aggregate_cross_pair_stratified_covariance(
        history_estimator_identities=identities,
        scores_by_history_pair_feature=scores,
        score_semantics="test_score",
    )
    permutation = np.asarray([7, 2, 0, 6, 3, 1, 5, 4])
    second = aggregate_cross_pair_stratified_covariance(
        history_estimator_identities=[
            identities[index] for index in permutation
        ],
        scores_by_history_pair_feature=scores[permutation],
        score_semantics="test_score",
    )

    assert (
        first.history_estimator_identity_sha256
        == second.history_estimator_identity_sha256
    )
    assert first.group_assignment_sha256 == second.group_assignment_sha256
    assert first.artifact_sha256 == second.artifact_sha256
    np.testing.assert_array_equal(
        first.centered_factor_by_history,
        second.centered_factor_by_history,
    )


@pytest.mark.parametrize(
    ("identities", "message"),
    (
        (
            [
                HistoryEstimatorIdentity(0, 0, 0, 0, 1, 1.0),
                HistoryEstimatorIdentity(0, 0, 0, 0, 1, 1.0),
            ],
            "unique",
        ),
        (
            [
                HistoryEstimatorIdentity(0, 0, 0, 0, 1, 1.0),
            ],
            "at least two",
        ),
        (
            [
                HistoryEstimatorIdentity(0, 0, 0, 0, 2, 1.0),
                HistoryEstimatorIdentity(1, 0, 0, 0, 2, 1.0),
            ],
            "every declared angle stratum",
        ),
        (
            [
                HistoryEstimatorIdentity(0, 0, 0, 0, 1, 1.0),
                HistoryEstimatorIdentity(1, 0, 0, 0, 1, 2.0),
            ],
            "cannot mix",
        ),
    ),
)
def test_exact_covariance_rejects_invalid_estimator_groups(
    identities: list[HistoryEstimatorIdentity],
    message: str,
) -> None:
    """Invalid history identity or fixed-quota groups must fail closed."""
    scores = np.zeros((len(identities), 64, 1), dtype=np.float64)

    with pytest.raises(ValueError, match=message):
        aggregate_cross_pair_stratified_covariance(
            history_estimator_identities=identities,
            scores_by_history_pair_feature=scores,
            score_semantics="test_score",
        )


def test_manifest_binds_covariance_identity_to_bank() -> None:
    """A covariance from another estimator schedule must be rejected."""
    bank = _bank()
    changed = _identities(coefficient=0.5)
    scores = np.zeros((8, 64, 1), dtype=np.float64)
    covariance = aggregate_cross_pair_stratified_covariance(
        history_estimator_identities=changed,
        scores_by_history_pair_feature=scores,
        score_semantics="test_score",
    )

    with pytest.raises(ValueError, match="identities differ"):
        build_paired_all64_manifest(
            bank_metadata=bank,
            pair_requests=_pairs(bank),
            covariance_metadata=covariance.metadata(),
        )


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    (
        ("schema_version", 2, "schema_version"),
        ("profile", "geant4_phase_space_paired_all64_v2", "profile"),
        ("bank_format", "event_grouped_detector_boundary_v2", "format"),
    ),
)
def test_bank_rejects_unknown_or_mixed_v2_v3_contracts(
    field_name: str,
    value: object,
    message: str,
) -> None:
    """No v2 bank token may be accepted under the v3 Python schema."""
    bank = _bank()
    bank[field_name] = value

    with pytest.raises(ValueError, match=message):
        validate_phase_space_bank_metadata(bank)


def test_covariance_rejects_old_or_mixed_block_schema() -> None:
    """Pooled block fields must never masquerade as exact v2 covariance."""
    metadata = _covariance().metadata()
    metadata["schema_version"] = 1
    with pytest.raises(ValueError, match="schema_version"):
        validate_cross_pair_covariance_metadata(metadata)

    metadata = _covariance().metadata()
    metadata["block_count"] = 4
    with pytest.raises(ValueError, match="keys differ"):
        validate_cross_pair_covariance_metadata(metadata)


def test_standard_runtime_cannot_select_paired_phase_space_profile() -> None:
    """Standard main/config paths must remain disconnected from paired replay."""
    assert PAIRED_ALL64_STANDARD_RUNTIME_SELECTABLE is False
    with pytest.raises(ValueError, match="not a standard runtime"):
        require_dedicated_paired_all64_profile(
            PAIRED_ALL64_PROFILE,
            standard_runtime=True,
        )

    repository_root = Path(__file__).resolve().parents[1]
    standard_config_path = (
        repository_root
        / "configs/geant4/variance_reduction_external_no_isaac_32threads.json"
    )
    standard_config = json.loads(standard_config_path.read_text())
    assert PAIRED_ALL64_PROFILE not in json.dumps(
        standard_config,
        sort_keys=True,
    )
    assert "paired_all64_phase_space" not in (
        repository_root / "src/runtime/cli.py"
    ).read_text()
    assert "paired_all64_phase_space" not in (
        repository_root / "src/runtime/session.py"
    ).read_text()
