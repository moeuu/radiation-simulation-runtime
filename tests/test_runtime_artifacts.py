"""Tests for public atomic artifact publication helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime.artifacts import (
    ARTIFACT_INVENTORY_DIGEST_ALGORITHM,
    AtomicBundlePublisher,
    ArtifactInventory,
    DurableJSONLWriter,
    atomic_copy_file,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    build_artifact_inventory,
    publish_artifact_manifest,
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


def test_artifact_inventory_exposes_its_digest_algorithm(tmp_path: Path) -> None:
    """Inventory hashes must never be interpreted without their algorithm ID."""
    (tmp_path / "artifact.txt").write_text("content\n", encoding="utf-8")

    inventory = build_artifact_inventory(tmp_path)

    assert inventory.digest.algorithm == ARTIFACT_INVENTORY_DIGEST_ALGORITHM
    assert inventory.digest.sha256 == inventory.sha256


def test_artifact_inventory_allowlist_is_exact(tmp_path: Path) -> None:
    """Allowlist inventories must hash only declared regular files."""
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")

    inventory = ArtifactInventory.from_allowlist(tmp_path, ["b.txt"])

    assert tuple(inventory.sha256_by_path) == ("b.txt",)
    with pytest.raises(FileNotFoundError, match="Allowlisted"):
        ArtifactInventory.from_allowlist(tmp_path, ["missing.txt"])


def test_atomic_bundle_create_and_replace_known(tmp_path: Path) -> None:
    """Complete directory generations must appear through one atomic rename."""
    target = tmp_path / "bundle"
    with AtomicBundlePublisher(target) as publisher:
        publisher.write_bytes("data.bin", b"old")
        publisher.write_json("metadata.json", {"generation": 1})
        first = publisher.publish()

    assert target.joinpath("data.bin").read_bytes() == b"old"
    assert first == build_artifact_inventory(target)

    with AtomicBundlePublisher(
        target,
        policy="replace_known",
        known_members=("data.bin", "metadata.json"),
    ) as publisher:
        publisher.write_bytes("data.bin", b"new")
        publisher.write_json("metadata.json", {"generation": 2})
        second = publisher.publish()

    assert target.joinpath("data.bin").read_bytes() == b"new"
    assert json.loads(target.joinpath("metadata.json").read_text(encoding="utf-8")) == {
        "generation": 2
    }
    assert second == build_artifact_inventory(target)


def test_replace_known_preserves_unrelated_members(tmp_path: Path) -> None:
    """A report update must retain unrelated files byte-for-byte."""
    target = tmp_path / "bundle"
    target.mkdir()
    (target / "known.txt").write_bytes(b"old")
    (target / "notes.txt").write_bytes(b"user-owned")

    with AtomicBundlePublisher(
        target,
        policy="replace_known",
        known_members=("known.txt",),
    ) as publisher:
        publisher.write_bytes("known.txt", b"new")
        publisher.publish()

    assert (target / "known.txt").read_bytes() == b"new"
    assert (target / "notes.txt").read_bytes() == b"user-owned"


def test_replace_known_rejects_staging_path_mutation_of_unrelated_member(
    tmp_path: Path,
) -> None:
    """Specialized encoders cannot bypass the replacement member allowlist."""
    target = tmp_path / "bundle"
    target.mkdir()
    (target / "known.txt").write_bytes(b"old")
    (target / "notes.txt").write_bytes(b"user-owned")

    with pytest.raises(ValueError, match="undeclared"):
        with AtomicBundlePublisher(
            target,
            policy="replace_known",
            known_members=("known.txt",),
        ) as publisher:
            (publisher.staging_path / "notes.txt").write_bytes(b"changed")
            publisher.publish()

    assert (target / "known.txt").read_bytes() == b"old"
    assert (target / "notes.txt").read_bytes() == b"user-owned"
    assert not tuple(tmp_path.glob(".bundle.bundle-*"))


def test_atomic_bundle_failure_preserves_existing_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed exchange must leave the prior public generation unchanged."""
    target = tmp_path / "bundle"
    target.mkdir()
    (target / "known.txt").write_bytes(b"old")

    def fail_exchange(source: Path, destination: Path, *, flags: int) -> None:
        """Inject an exchange failure before the public directory changes."""
        del source, destination, flags
        raise OSError("injected exchange failure")

    monkeypatch.setattr("runtime.artifacts._linux_renameat2", fail_exchange)
    with pytest.raises(OSError, match="injected exchange failure"):
        with AtomicBundlePublisher(
            target,
            policy="replace_known",
            known_members=("known.txt",),
        ) as publisher:
            publisher.write_bytes("known.txt", b"new")
            publisher.publish()

    assert (target / "known.txt").read_bytes() == b"old"
    assert not tuple(tmp_path.glob(".bundle.bundle-*"))


def test_durable_jsonl_and_manifest_publication(tmp_path: Path) -> None:
    """JSONL appends and inventory manifests must use strict durable bytes."""
    events = tmp_path / "events.jsonl"
    with DurableJSONLWriter(events) as writer:
        assert writer.append({"event": "ready", "sequence": 0}) > 0
        writer.append({"event": "record", "sequence": 1})

    assert [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()] == [
        {"event": "ready", "sequence": 0},
        {"event": "record", "sequence": 1},
    ]
    inventory = ArtifactInventory.from_allowlist(tmp_path, ["events.jsonl"])
    manifest_path = publish_artifact_manifest(
        tmp_path / "artifact_manifest.json",
        inventory,
        metadata={"encoding_profile": "jsonl-v1"},
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["inventory_digest"] == inventory.digest.to_payload()
    assert manifest["metadata"] == {"encoding_profile": "jsonl-v1"}


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
