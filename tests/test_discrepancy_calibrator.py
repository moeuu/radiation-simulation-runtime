"""Tests for estimator-neutral all-pair discrepancy calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from runtime.discrepancy_calibrator import calibrate_discrepancy


def _write_input(path: Path, *, all_pairs: bool = True) -> None:
    """Write two independent environments for every requested shield pair."""
    pairs = np.arange(64 if all_pairs else 63, dtype=np.int64)
    pair_ids = np.tile(pairs, 2)
    environments = np.repeat(np.asarray(["cal-a", "cal-b"]), pairs.size)
    expected = np.full((pair_ids.size, 4), 100.0)
    modulation = (pair_ids[:, None] % 5) * np.asarray([[1.0, 2.0, 1.0, 0.5]])
    observed = expected + modulation
    observed[pairs.size :] += np.asarray([2.0, -1.0, 3.0, 0.0])
    np.savez(
        path,
        observed_counts=observed,
        expected_counts=expected,
        energy_bin_edges_keV=np.arange(5, dtype=float),
        environment_ids=environments,
        shield_pair_ids=pair_ids,
    )


def test_calibrator_publishes_strict_all_pair_contract(tmp_path: Path) -> None:
    """Calibration output must validate and retain independent environments."""
    source = tmp_path / "input.npz"
    target = tmp_path / "calibration.json"
    _write_input(source)

    calibration = calibrate_discrepancy(
        source,
        target,
        calibration_id="holdout-v1",
        residual_rank=2,
    )

    assert target.is_file()
    assert calibration.independent_environment_ids == ("cal-a", "cal-b")
    assert calibration.shield_pair_feature_basis.shape[0] == 64
    assert np.all(calibration.overdispersion_alpha_by_bin >= 0.0)


def test_calibrator_rejects_incomplete_shield_pair_coverage(tmp_path: Path) -> None:
    """A subset of shield pairs cannot define production discrepancy physics."""
    source = tmp_path / "input.npz"
    _write_input(source, all_pairs=False)

    with pytest.raises(ValueError, match="all 64"):
        calibrate_discrepancy(
            source,
            tmp_path / "calibration.json",
            calibration_id="invalid",
        )


def test_calibrator_rejects_nonpositive_residual_rank(tmp_path: Path) -> None:
    """A low-rank residual family must contain at least one requested mode."""
    source = tmp_path / "input.npz"
    _write_input(source)

    with pytest.raises(ValueError, match="positive integer"):
        calibrate_discrepancy(
            source,
            tmp_path / "calibration.json",
            calibration_id="invalid-rank",
            residual_rank=0,
        )


def test_calibrator_requires_all_pairs_in_every_environment(tmp_path: Path) -> None:
    """All-pair calibration cannot be assembled from disjoint environment subsets."""
    source = tmp_path / "input.npz"
    _write_input(source)
    with np.load(source, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    keep = ~(
        (payload["environment_ids"] == "cal-b") & (payload["shield_pair_ids"] == 63)
    )
    np.savez(
        source,
        **{
            name: value[keep] if name != "energy_bin_edges_keV" else value
            for name, value in payload.items()
        },
    )

    with pytest.raises(ValueError, match="Every independent environment"):
        calibrate_discrepancy(
            source,
            tmp_path / "calibration.json",
            calibration_id="incomplete-environment",
        )
