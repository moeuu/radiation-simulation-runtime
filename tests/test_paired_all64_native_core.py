"""Tests for the native paired all-64 phase-space replay core."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from spectrum.paired_all64_phase_space import (
    HistoryEstimatorIdentity,
    aggregate_cross_pair_stratified_covariance,
    phase_space_replay_seed,
)


@pytest.fixture(scope="module")
def native_driver(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile the dependency-free native phase-space core test driver."""
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the native core contract test.")
    repository_root = Path(__file__).resolve().parents[1]
    output_path = tmp_path_factory.mktemp("paired_all64_native") / "driver"
    command = [
        compiler,
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-Werror",
        "-I",
        str(repository_root / "native/geant4_sidecar"),
        str(
            repository_root
            / "tests/native/paired_all64_phase_space_test_driver.cpp"
        ),
        str(
            repository_root
            / "native/geant4_sidecar/paired_all64_phase_space.cpp"
        ),
        "-o",
        str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return output_path


def _run_driver(driver: Path) -> dict[str, str]:
    """Run the native contract driver and parse its scalar report."""
    result = subprocess.run(
        [str(driver)],
        check=True,
        capture_output=True,
        text=True,
    )
    return dict(
        line.split("=", maxsplit=1)
        for line in result.stdout.splitlines()
        if line
    )


def test_native_capture_bank_replay_and_fail_closed_contract(
    native_driver: Path,
) -> None:
    """The compiled core must exercise grouping, replay, and rejection gates."""
    report = _run_driver(native_driver)

    assert len(report["bank_sha256"]) == 64
    assert int(report["bank_size"]) > 0
    assert int(report["covariance_payload_size"]) > int(report["bank_size"])
    assert int(report["replay_seed"]) == phase_space_replay_seed(
        root_seed=7,
        bank_payload_sha256="c" * 64,
        shield_pair_id=12,
    )


def test_native_stratified_covariance_is_cross_language_equivalent(
    native_driver: Path,
) -> None:
    """C++ original-history strata must match Python byte-for-byte."""
    report = _run_driver(native_driver)
    identities = [
        HistoryEstimatorIdentity(0, 0, 0, 0, 2, 2.0),
        HistoryEstimatorIdentity(1, 0, 0, 0, 2, 2.0),
        HistoryEstimatorIdentity(2, 0, 0, 1, 2, 2.0),
        HistoryEstimatorIdentity(3, 0, 0, 1, 2, 2.0),
        HistoryEstimatorIdentity(4, 1, 2, 0, 1, 0.5),
        HistoryEstimatorIdentity(5, 1, 2, 0, 1, 0.5),
        HistoryEstimatorIdentity(6, 1, 2, 0, 1, 0.5),
        HistoryEstimatorIdentity(7, 1, 2, 0, 1, 0.5),
    ]
    scores = np.empty((8, 64, 2), dtype=np.float64)
    for pair in range(64):
        pair_value = float(pair)
        scores[:, pair, :] = np.asarray(
            [
                [pair_value + 1.0, 0.0],
                [pair_value + 3.0, 2.0],
                [10.0 + 2.0 * pair_value, 1.0],
                [14.0 + 2.0 * pair_value, 5.0],
                [0.0, 0.0],
                [2.0, 0.0],
                [4.0, 2.0],
                [6.0, 2.0],
            ],
            dtype=np.float64,
        )
    expected = aggregate_cross_pair_stratified_covariance(
        history_estimator_identities=identities,
        scores_by_history_pair_feature=scores,
        score_semantics="incident_gamma_bin_count_per_primary_history",
    )

    assert int(report["group_count"]) == expected.group_count
    assert (
        report["group_assignment_sha256"]
        == expected.group_assignment_sha256
    )
    assert report["artifact_sha256"] == expected.artifact_sha256
    assert (
        float(report["estimate_first"])
        == expected.estimate_by_pair_feature.flat[0]
    )
    assert float(report["estimate_pair63_feature0"]) == (
        expected.estimate_by_pair_feature[63, 0]
    )
    assert float(report["factor_first"]) == pytest.approx(
        expected.centered_factor_by_history.flat[0],
        rel=1e-5,
    )
    assert float(report["zero_history_factor"]) == pytest.approx(
        expected.centered_factor_by_history[4, 0, 0],
        rel=1e-5,
    )
    assert float(report["covariance_first"]) == pytest.approx(
        expected.total_cross_pair_covariance.flat[0],
        rel=1e-5,
    )
    assert float(report["covariance_last"]) == pytest.approx(
        expected.total_cross_pair_covariance.flat[-1],
        rel=1e-5,
    )
    assert (
        report["approximate_semantics"]
        == "approximate_pooled_hash_block_diagnostic_v1"
    )


def test_native_api_exposes_only_batched_scores_and_full_world_replay() -> None:
    """The integration surface must encode paired replay fidelity explicitly."""
    repository_root = Path(__file__).resolve().parents[1]
    header = (
        repository_root
        / "native/geant4_sidecar/paired_all64_phase_space.hpp"
    ).read_text(encoding="utf-8")

    assert "SubmitPairScores" in header
    assert "SubmitScore(" not in header
    assert "FullWorldReplayRequired" in header
    assert "KillOutwardCrossings" in header
    assert "standard_runtime" in header
    assert "RecordFirstInwardCrossing" in header
    assert "std::string particle_name" in header
    assert "double mass_mev" in header
    assert "double charge_eplus" in header
