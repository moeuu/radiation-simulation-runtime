"""Tests for public byte-stable runtime publication APIs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from runtime.measurement_log import write_deterministic_npz


def test_public_npz_writer_is_byte_stable_and_preserves_insertion_order(
    tmp_path: Path,
) -> None:
    """Repeated publications should have identical bytes and member order."""
    arrays = {
        "second": np.asarray([2, 3], dtype=np.int64),
        "first": np.asarray([1.5], dtype=np.float64),
    }
    first = write_deterministic_npz(tmp_path / "first.npz", arrays)
    second = write_deterministic_npz(tmp_path / "second.npz", arrays)

    assert first.read_bytes() == second.read_bytes()
    with np.load(first, allow_pickle=False) as loaded:
        assert loaded.files == ["second", "first"]
        np.testing.assert_array_equal(loaded["second"], arrays["second"])


@pytest.mark.parametrize(
    "arrays",
    [
        {},
        {"": np.asarray([1])},
        {"nested/name": np.asarray([1])},
        {".hidden": np.asarray([1])},
        {"nul\x00suffix": np.asarray([1])},
        {"line\nbreak": np.asarray([1])},
        {"object": np.asarray([object()], dtype=object)},
    ],
)
def test_public_npz_writer_rejects_invalid_array_mappings(
    tmp_path: Path,
    arrays: dict[str, np.ndarray],
) -> None:
    """Invalid public writer inputs should fail before creating an archive."""
    with pytest.raises((TypeError, ValueError)):
        write_deterministic_npz(tmp_path / "invalid.npz", arrays)

    assert not (tmp_path / "invalid.npz").exists()


def test_public_npz_writer_preserves_existing_target_on_failure(
    tmp_path: Path,
) -> None:
    """A rejected array must not truncate a previously published archive."""
    target = tmp_path / "existing.npz"
    target.write_bytes(b"old-artifact")

    with pytest.raises(TypeError, match="Python objects"):
        write_deterministic_npz(
            target,
            {
                "valid": np.asarray([1], dtype=np.int64),
                "invalid": np.asarray([object()], dtype=object),
            },
        )

    assert target.read_bytes() == b"old-artifact"
    assert not tuple(tmp_path.glob(".existing.npz.*.tmp"))
