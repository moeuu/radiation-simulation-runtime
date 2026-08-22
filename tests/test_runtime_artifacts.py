"""Tests for public atomic artifact publication helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime.artifacts import (
    atomic_copy_file,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
)


def test_atomic_writers_publish_complete_canonical_files(tmp_path: Path) -> None:
    """Public writers should replace files without leaving staging entries."""
    bytes_path = atomic_write_bytes(tmp_path / "bytes.bin", b"first")
    atomic_write_bytes(bytes_path, b"second")
    text_path = atomic_write_text(tmp_path / "value.txt", "value\n")
    json_path = atomic_write_json(
        tmp_path / "value.json",
        {"path": str(tmp_path / "asset"), "values": (2, 1)},
    )

    assert bytes_path.read_bytes() == b"second"
    assert text_path.read_text(encoding="utf-8") == "value\n"
    assert json.loads(json_path.read_text(encoding="utf-8")) == {
        "path": str(tmp_path / "asset"),
        "values": [2, 1],
    }
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_atomic_copy_preserves_existing_target_on_copy_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed staged copy must leave the prior publication untouched."""
    source = tmp_path / "source.bin"
    target = tmp_path / "target.bin"
    source.write_bytes(b"new")
    target.write_bytes(b"old")

    def fail_copy(*args: object, **kwargs: object) -> None:
        """Raise after the staging file has been created."""
        del args, kwargs
        raise OSError("injected copy failure")

    monkeypatch.setattr("runtime.artifacts.shutil.copyfileobj", fail_copy)
    with pytest.raises(OSError, match="injected copy failure"):
        atomic_copy_file(source, target)

    assert target.read_bytes() == b"old"
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_atomic_copy_rejects_missing_source(tmp_path: Path) -> None:
    """Copy publication should fail before changing the destination."""
    target = tmp_path / "target.bin"
    target.write_bytes(b"old")

    with pytest.raises(FileNotFoundError, match="Artifact source"):
        atomic_copy_file(tmp_path / "missing.bin", target)

    assert target.read_bytes() == b"old"


@pytest.mark.parametrize(
    ("payload", "error"),
    (
        ({1: "integer", "1": "string"}, TypeError),
        ({"value": float("nan")}, ValueError),
        ({"value": object()}, TypeError),
        ({"value": Path("asset")}, TypeError),
    ),
)
def test_atomic_json_rejects_lossy_or_unstable_values(
    tmp_path: Path,
    payload: object,
    error: type[Exception],
) -> None:
    """Strict JSON publication must preserve an existing artifact on failure."""
    target = tmp_path / "value.json"
    target.write_bytes(b"old-artifact")

    with pytest.raises(error):
        atomic_write_json(target, payload)

    assert target.read_bytes() == b"old-artifact"
