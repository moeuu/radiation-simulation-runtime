"""Atomic publication helpers for cross-repository runtime artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import re
import shutil
import tempfile
from types import MappingProxyType
from typing import Any

from runtime.provenance import strict_canonical_json_bytes, strict_sha256_json


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ArtifactInventory:
    """Store an immutable path-to-digest inventory for ordinary files."""

    sha256_by_path: Mapping[str, str]

    def __post_init__(self) -> None:
        """Validate, sort, and freeze the portable relative-path mapping."""
        normalized: dict[str, str] = {}
        for relative_path, digest in self.sha256_by_path.items():
            if not isinstance(relative_path, str) or not relative_path:
                raise TypeError("Artifact inventory paths must be nonempty strings.")
            path = Path(relative_path)
            if path.is_absolute() or relative_path in {".", ".."} or ".." in path.parts:
                raise ValueError(
                    "Artifact inventory paths must be root-relative without '..'."
                )
            if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
                raise ValueError(
                    "Artifact inventory digests must be lowercase SHA-256 strings."
                )
            normalized[relative_path] = digest
        object.__setattr__(
            self,
            "sha256_by_path",
            MappingProxyType(dict(sorted(normalized.items()))),
        )

    @property
    def sha256(self) -> str:
        """Return the canonical digest of the complete inventory mapping."""
        return strict_sha256_json(dict(self.sha256_by_path))

    @property
    def file_count(self) -> int:
        """Return the number of regular files represented by the inventory."""
        return len(self.sha256_by_path)


def build_artifact_inventory(root: str | Path) -> ArtifactInventory:
    """Hash every root-contained regular file without applying schema policy."""
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"Artifact inventory root is not a directory: {root_path}")
    digests: dict[str, str] = {}
    for candidate in sorted(root_path.rglob("*")):
        relative = candidate.relative_to(root_path).as_posix()
        if candidate.is_symlink():
            raise ValueError(f"Artifact inventory must not contain symlink {relative}.")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise ValueError(f"Artifact inventory entry is not a regular file: {relative}")
        digest = sha256()
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digests[relative] = digest.hexdigest()
    return ArtifactInventory(digests)


def atomic_write_bytes(path: str | Path, payload: bytes) -> Path:
    """Atomically replace one file with durable serialized bytes."""
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes.")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Atomically replace one text file using the requested encoding."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    return atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    """Serialize strict canonical JSON and publish it as one atomic file."""
    return atomic_write_bytes(path, strict_canonical_json_bytes(payload))


def atomic_copy_file(source: str | Path, target: str | Path) -> Path:
    """Copy a completed file and atomically replace its publication target."""
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(f"Artifact source does not exist: {source_path}")
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target_path.parent,
        prefix=f".{target_path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with source_path.open("rb") as source_handle, temporary.open(
            "wb"
        ) as target_handle:
            shutil.copyfileobj(source_handle, target_handle)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        os.replace(temporary, target_path)
        _fsync_directory(target_path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target_path


def _fsync_directory(path: Path) -> None:
    """Synchronize one directory after an atomic entry replacement."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "ArtifactInventory",
    "atomic_copy_file",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "build_artifact_inventory",
]
